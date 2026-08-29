//! The runner's side of the console: a way in, a way out, and the commands
//! that only the host can answer.
//!
//! The way in is **stdin**. There is no console widget yet - the editor is the
//! step that brings one, and when it does it will be a *view* onto the table
//! [`colby_core::abi::cvar`] already holds rather than a second system. A line
//! typed into the terminal `just hot` was started from reaches the same
//! [`console::run`] a config file does, which is what makes the feature usable
//! today instead of after the next two subsystems.
//!
//! The way out is `cvars.cfg` in the workspace, written when the process stops
//! and read when it starts. It is a config script, not a serialization format:
//! the same parser reads it, it is meant to be edited by hand, and colby gains
//! no dependency for it. Writing it as JSON would have cost one.
//!
//! @note: `--shot` has no console and does not read `cvars.cfg`. A screenshot
//! is meant to be the same picture on every machine, and a file holding
//! whatever someone last typed is the opposite of that.

use std::{
	fs,
	io::{BufRead, stdin},
	panic::{AssertUnwindSafe, catch_unwind},
	path::{Path, PathBuf},
	sync::mpsc::{self, Receiver, TryRecvError},
	thread,
};

use colby_core::{
	Error,
	abi::{Args, Value, World, console},
	error, info, warn,
};

/// The file archived variables are kept in.
const ARCHIVE: &str = "cvars.cfg";

/// The most steps one frame will run on top of real time.
///
/// Four seconds of simulation in one frame is already an odd thing to want;
/// this is here so that a typo cannot ask for an hour of it. The command
/// clamps what it is given and the loop clamps again on the way out, because
/// `World::owed_steps` is a public field and gameplay code is code that is
/// expected to be wrong sometimes.
pub(crate) const MAX_STEP: i64 = 240;

/// Lines typed at the terminal, on their way to [`console::run`].
pub(crate) struct Console {
	lines: Receiver<String>,
	archive: PathBuf,
}

impl Console {
	/// Registers the host's commands, reads the config, and starts listening.
	///
	/// Called after the game module is loaded, so that a `cvars.cfg` naming a
	/// variable the *game* registers finds it there.
	///
	/// @param world - the world whose table everything is registered into
	pub(crate) fn open(world: &mut World) -> Self {
		let archive = crate::workspace().join(ARCHIVE);

		if archive.is_file() {
			console::exec(world, &archive);
		}

		Self { lines: listen(), archive }
	}

	/// Runs whatever has been typed since the previous frame.
	///
	/// Never blocks: the reader is a thread of its own, and this takes what is
	/// waiting and returns.
	///
	/// @param world - the state commands act on
	pub(crate) fn poll(&self, world: &mut World) {
		loop {
			match self.lines.try_recv() {
				| Ok(line) => run(world, &line),
				| Err(TryRecvError::Empty | TryRecvError::Disconnected) => return,
			}
		}
	}

	/// Writes the archived variables back out.
	///
	/// @param world - the world to read them from
	pub(crate) fn close(&self, world: &World) {
		let mut lines = vec![
			"// Written by colby when it stopped, and read when it starts.".to_owned(),
			"// This is an ordinary config script: edit it, or `exec` it again.".to_owned(),
			String::new(),
		];

		for entry in world.cvars.iter() {
			let Some(value) = entry.value().filter(|_| entry.is_archived()) else {
				continue;
			};

			lines.push(format!("{} {}", entry.name(), value.quoted()));
		}

		lines.push(String::new());

		if let Err(failure) = fs::write(&self.archive, lines.join("\n")) {
			error!(path = %self.archive.display(), %failure, "could not write the config");
		}
	}
}

/// Registers everything the host answers for.
///
/// Called before the game module loads, so that these are the engine's and stay
/// through a reload. @ref [`colby_core::abi::cvar::Cvars::forget_module`].
///
/// @param world - the world to register into
pub(crate) fn install(world: &mut World) {
	world
		.cvars
		.command("help", help, "list commands and variables, optionally matching a word");
	world
		.cvars
		.command("echo", echo, "print the rest of the line");
	world
		.cvars
		.command("exec", exec, "run a config file, relative to the workspace");
	world
		.cvars
		.command("reset", reset, "put a variable back to the value its code registered");
	world
		.cvars
		.command("quit", quit, "stop the process");

	world.cvars.var(
		crate::app::PAUSE,
		Value::Bool(false),
		"hold the simulation still; the picture goes on being drawn",
	);
	world.cvars.var(
		crate::app::SPEED,
		Value::Float(1.0),
		"how fast simulated time runs against real time",
	);
	world
		.cvars
		.command("sim.step", step, "run this many simulation steps, paused or not");

	world.cvars.var(
		crate::app::GRAVITY,
		Value::Float(-crate::app::PULL),
		"how hard everything is pulled downwards, in units a second squared",
	);
	world.cvars.var(
		colby_physics::PASSES,
		Value::Float(passes()),
		"how many times the solver corrects contact velocities each step",
	);
	world
		.cvars
		.command("phys.bodies", bodies, "report what the solver is carrying");

	// the debug renderer. All off, because this is a tool and a window covered
	// in wireframe is worse than no wireframe for every question except the one
	// it answers. Not archived either: coming back to a session with the
	// outlines still on is a surprise rather than a setting.
	world.cvars.var(
		colby_physics::debug::SHAPES,
		Value::Bool(false),
		"outline what every body is really shaped like",
	);
	world.cvars.var(
		colby_physics::debug::CONTACTS,
		Value::Bool(false),
		"mark every contact the solver found and which way it pushes",
	);
	world.cvars.var(
		colby_physics::debug::JOINTS,
		Value::Bool(false),
		"mark every joint and the two anchors it holds",
	);
	// shadows, and these are on by default because they are a feature rather
	// than a tool: what the variable is for is turning them off on a machine
	// that cannot afford them. The cascade tint is the exception and is a tool.
	world.cvars.var(
		colby_engine::shadow::ENABLED,
		Value::Bool(true),
		"cast shadows from the one directional light",
	);
	world.cvars.var(
		colby_engine::shadow::DISTANCE,
		Value::Float(colby_engine::shadow::DEFAULT_DISTANCE),
		"how far from the camera anything is shadowed at all, in units",
	);
	world.cvars.var(
		colby_engine::shadow::TINT,
		Value::Bool(false),
		"color every pixel by the shadow cascade it read",
	);
	world.cvars.var(
		colby_ui::world_text::TEXT_SIZE,
		Value::Float(colby_ui::world_text::DEFAULT_TEXT_SIZE),
		"how big a label anchored in the world is drawn, in layout pixels",
	);
	world.cvars.command(
		"debug.clear",
		clear_debug,
		"throw away every debug line that has a lifetime",
	);

	world
		.cvars
		.command("scene.save", save_scene, "write the world into saves/<name>.cscene");
	world.cvars.command(
		"scene.load",
		load_scene,
		"put saves/<name>.cscene back, replacing this world",
	);
}

/// Runs one line, containing anything it throws.
///
/// A command can come from the game module, and gameplay code is code that is
/// expected to be wrong sometimes. A panic in one costs a message: unlike a
/// panic in `update` it does not park the module, because a bad command says
/// nothing about whether the rest of the build works.
fn run(world: &mut World, line: &str) {
	let result = catch_unwind(AssertUnwindSafe(|| console::run(world, line)));

	if let Err(payload) = result {
		let failure = Error::from_panic(&*payload);
		drop(payload);

		error!(%failure, "the command panicked");
	}
}

/// Starts the thread that reads the terminal.
///
/// A thread rather than a poll on the main one because reading a line blocks
/// until there is one, and the frame cannot wait for someone to type.
///
/// Nothing joins it. It ends when stdin does - a redirected build, a closed
/// pipe - and there is nothing it holds that the process needs back.
fn listen() -> Receiver<String> {
	let (sender, lines) = mpsc::channel();

	let spawned = thread::Builder::new()
		.name("console".to_owned())
		.spawn(move || {
			for line in stdin().lock().lines() {
				let Ok(line) = line else {
					return;
				};

				if sender.send(line).is_err() {
					return;
				}
			}
		});

	if let Err(failure) = spawned {
		error!(%failure, "no console: the terminal reader would not start");
	}

	lines
}

/// `help [word]` - lists commands and variables.
///
/// # Safety
///
/// As [`ConsoleFn`](colby_core::abi::ConsoleFn): both pointers are live for the
/// duration of the call.
unsafe extern "C-unwind" fn help(world: *mut World, args: *const Args) {
	// SAFETY: the console hands over a live world for the duration of the call,
	// and nothing else touches it while a command runs.
	let world = unsafe { &*world };
	// SAFETY: the argument list is built by the caller and outlives the call.
	let args = unsafe { &*args };
	let filter = args.word(0).unwrap_or_default();
	let mut shown = 0_usize;

	for entry in world.cvars.iter() {
		if !entry.name().contains(filter) {
			continue;
		}

		shown += 1;

		match entry.value() {
			| Some(value) => info!("{} = {} - {}", entry.name(), value.quoted(), entry.help()),
			| None => info!("{} - {}", entry.name(), entry.help()),
		}
	}

	info!("{shown} of {} entries", world.cvars.len());
}

/// `echo <text>` - prints the rest of the line.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn echo(_world: *mut World, args: *const Args) {
	// SAFETY: as help.
	let args = unsafe { &*args };

	info!("{}", args.rest());
}

/// `exec <path>` - runs a config file.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn exec(world: *mut World, args: *const Args) {
	// SAFETY: as help.
	let world = unsafe { &mut *world };
	// SAFETY: as help.
	let args = unsafe { &*args };

	let Some(name) = args.word(0) else {
		warn!("exec takes the path of a config file");

		return;
	};

	// resolved against the workspace, so that `exec scripts/demo.cfg` means the
	// same thing wherever the executable was started from.
	let path = Path::new(name);
	let path = if path.is_absolute() {
		path.to_owned()
	} else {
		crate::workspace().join(path)
	};

	console::exec(world, &path);
}

/// `reset <name>` - puts a variable back to its registered value.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn reset(world: *mut World, args: *const Args) {
	// SAFETY: as help.
	let world = unsafe { &mut *world };
	// SAFETY: as help.
	let args = unsafe { &*args };

	let Some(name) = args.word(0) else {
		warn!("reset takes the name of a variable");

		return;
	};

	if world.cvars.reset(name) {
		console::run(world, name);
	} else {
		warn!(name, "not a variable");
	}
}

/// `quit` - asks the runner to stop.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn quit(world: *mut World, _args: *const Args) {
	// SAFETY: as help.
	let world = unsafe { &mut *world };

	world.quit = true;
}

/// `debug.clear` - throws away lasting debug geometry.
///
/// Transient geometry is swept every step and needs no command; this is for the
/// marks somebody asked to keep, which by definition nothing else will take
/// away.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn clear_debug(world: *mut World, _args: *const Args) {
	// SAFETY: as help.
	let world = unsafe { &mut *world };

	world.debug.clear();
	info!("debug geometry cleared");
}

/// `scene.save <name>` - writes the world out.
///
/// Both of these leave a note rather than doing the work: a load has to happen
/// between frames and needs the solver, which a command cannot reach. @ref
/// [`crate::saves`] for why that is the shape rather than an accident.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn save_scene(_world: *mut World, args: *const Args) {
	// SAFETY: as help.
	let args = unsafe { &*args };

	crate::saves::ask(crate::saves::Request::Save(args.rest()));
}

/// `scene.load <name>` - puts a saved world back.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn load_scene(_world: *mut World, args: *const Args) {
	// SAFETY: as help.
	let args = unsafe { &*args };

	crate::saves::ask(crate::saves::Request::Load(args.rest()));
}

/// The solver's default pass count, as the number a console variable holds.
fn passes() -> f32 {
	f32::from(u16::try_from(colby_physics::VELOCITY_PASSES).unwrap_or(u16::MAX))
}

/// `phys.bodies` - reports what the solver is carrying.
///
/// A view onto the body table rather than a mechanism of its own, which is the
/// rule a console command here follows as much as an editor panel does.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn bodies(world: *mut World, _args: *const Args) {
	// SAFETY: as help.
	let world = unsafe { &mut *world };

	let mut asleep = 0_usize;
	let mut movable = 0_usize;

	for (_, body) in world.bodies.iter() {
		if body.movable() {
			movable += 1;
		}

		if body.sleeping {
			asleep += 1;
		}
	}

	info!(
		bodies = world.bodies.len(),
		movable,
		asleep,
		contacts = world.contacts,
		gravity = %world.gravity,
		"physics"
	);

	// a line each for the handful the solver is actually moving. This is the
	// question a person asks next - *which* one will not settle - and a table
	// of five numbers answers it where a total cannot.
	for (id, body) in world.bodies.iter() {
		if !body.movable() {
			continue;
		}

		info!(
			slot = id.slot(),
			at = %body.transform.position,
			speed = body.velocity.length(),
			spin = body.angular.length(),
			asleep = body.sleeping,
			"body"
		);
	}
}

/// `sim.step [count]` - runs simulation steps whether or not time is passing.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn step(world: *mut World, args: *const Args) {
	// SAFETY: as help.
	let world = unsafe { &mut *world };
	// SAFETY: as help.
	let args = unsafe { &*args };

	let asked = args.int(0).unwrap_or(1).clamp(1, MAX_STEP);
	let Ok(count) = u32::try_from(asked) else {
		return;
	};

	world.owed_steps = world.owed_steps.saturating_add(count);
}
