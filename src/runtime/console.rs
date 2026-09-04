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
	abi::{
		Aim, Args, Asked, Cvars, Mix, PeerId, Scripts, Sound, Value, Voice, World, console,
		cvar::Owner,
	},
	error, info, warn,
};

/// The file archived variables are kept in.
const ARCHIVE: &str = "cvars.cfg";

/// The name `script.status` waits under.
pub(crate) const SCRIPT_STATUS: &str = "script.status";

/// Takes every waiting line that was asked under one of these names.
///
/// **The frame loop's half of a line that waited.** A command that needs
/// something the runner owns - the solver, the socket, the interpreter - is
/// registered with [`console::defer`] and leaves its line on the world; each
/// of the runner's subsystems then asks for its own names here, once a frame,
/// and does the work with what it has. @ref [`Asked`] for the whole
/// arrangement, and [`of_scripts`] for the one taker that goes by owner rather
/// than by name.
///
/// @param world - where the lines wait
/// @param names - which of them to take
/// @return the lines, in the order they were asked
pub(crate) fn take(world: &mut World, names: &[&str]) -> Vec<Asked> {
	world
		.asked
		.extract_if(.., |asked| names.contains(&asked.name.as_str()))
		.collect()
}

/// Takes every waiting line that is a program's to answer.
///
/// By who registered the name rather than by what it is: the interpreter
/// attributes what a program publishes to itself, so a program's command is
/// whatever the table says is a program's. A name nothing is registered under
/// any more was a program's that has since been built again, and goes with
/// them - the interpreter answers one of those by doing nothing, deliberately.
/// What is left is the engine's, which the frame loop takes by name, and a
/// module's, which stay for the game's own `update` to read.
///
/// @param world - where the lines wait
/// @return the lines, in the order they were asked
pub(crate) fn of_scripts(world: &mut World) -> Vec<Asked> {
	let cvars = &world.cvars;

	world
		.asked
		.extract_if(.., |asked| {
			cvars
				.get(&asked.name)
				.is_none_or(|entry| entry.owner() == Owner::Script)
		})
		.collect()
}

/// How many playing voices `snd.list` writes a line for.
///
/// The table holds sixty-four and a listing of all of them is a screen of
/// identical lines. The count is what the question was about; the lines are for
/// telling one voice from another.
const LISTED_VOICES: usize = 12;

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
	/// @param workspace - where the config is kept
	pub(crate) fn open(world: &mut World, workspace: &Path) -> Self {
		let archive = workspace.join(ARCHIVE);

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
		.command(EXEC, console::defer, "run a config file, relative to the workspace");
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
	// **saved, unlike the two above it.** Pausing and scaling are things
	// somebody does for a minute and stops doing; a tick rate is a property of
	// the project, and coming back to a world running at a rate other than the
	// one it was left at is the surprise rather than the setting. It takes
	// effect from the frame after it is typed - a rate that could be
	// configured and not turned would be the worst of both.
	world.cvars.saved(
		crate::app::RATE,
		Value::Int(i64::from(colby_core::time::Rate::DEFAULT.hz())),
		"how many simulation steps there are in a second",
	);
	world.cvars.var(
		crate::mode::EDIT,
		Value::Bool(false),
		"edit the world instead of playing it; stopping puts back what play started from",
	);

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

	install_scenes(world);
	install_audio(world);
	install_net(world);
	install_scripts(world);
}

/// The three commands over the programs the host is running.
///
/// Split off like the scenes and the volumes, and for the same reason. Two of
/// them are pure functions of the world, which is worth noticing: listing
/// reads the table, and reloading moves a revision and lets whoever is running
/// the program notice on its own. The third asks what the interpreter is
/// running, and the interpreter is the runner's, so its line waits for the
/// step - which at sixty a second is not something a person notices.
///
/// @param world - the table to register into
fn install_scripts(world: &mut World) {
	world.cvars.command(
		"script.list",
		list_scripts,
		"report every program the host has loaded and whose it is",
	);
	world.cvars.command(
		"script.reload",
		reload_script,
		"build <name> again, or every program if no name is given",
	);
	world.cvars.command(
		SCRIPT_STATUS,
		console::defer,
		"report what each running program cost and whether it has been switched off",
	);
}

/// The six numbers that make the wire bad, and the two commands over it.
///
/// Split off like the scenes and the volumes, and for the same reason: an
/// installer with one more subsystem inlined in it is one nobody reads. The
/// numbers themselves are registered by the subsystem that reads them back, so
/// that the name and its meaning are written down once.
///
/// Both commands wait for the frame: the endpoint they want is the runner's,
/// like the renderer and the output device. @ref [`crate::net::serve`], which
/// takes them up.
///
/// @param world - the table to register into
fn install_net(world: &mut World) {
	crate::net::install(&mut world.cvars);

	world.cvars.command(
		crate::net::SAY,
		console::defer,
		"send the rest of the line to every peer, resent until each has it",
	);
	world.cvars.command(
		crate::net::STATUS,
		console::defer,
		"report the wire and every peer on it",
	);
}

/// The four volumes and the three commands over the voice table.
///
/// Split off like the scene commands and for the same reason as those: the
/// installer is long enough that one more subsystem in it is one too many.
/// Everything here is a write into plain data, so none of it needs the device -
/// which is also why `snd.play` works in a build whose device failed to open.
///
/// @param world - the table to register into
fn install_audio(world: &mut World) {
	// archived, unlike the debug variables and like the editor's own: somebody
	// who turned the sound down meant it, and coming back to a session with it
	// loud again is the surprise this avoids.
	world.cvars.saved(
		colby_audio::MASTER,
		Value::Float(1.0),
		"how loud everything is, before anything else scales it",
	);
	world.cvars.saved(
		colby_audio::EFFECTS,
		Value::Float(1.0),
		"how loud sounds in the world are",
	);
	world
		.cvars
		.saved(colby_audio::MUSIC, Value::Float(1.0), "how loud music is");
	world
		.cvars
		.saved(colby_audio::INTERFACE, Value::Float(1.0), "how loud clicks and beeps are");

	world.cvars.command(
		"snd.play",
		play_sound,
		"play a compiled sound by name, at an optional volume",
	);
	world
		.cvars
		.command("snd.stop", stop_sounds, "stop everything that is playing");
	world
		.cvars
		.command("snd.list", list_sounds, "report the sounds loaded and what is playing");
}

/// The four volumes, as the console table has them.
///
/// A function of the table and nothing else, which is what makes it the one
/// piece of the audio wiring a test can reach - everything else in this
/// subsystem's runner half is a call into a device or a write into a world.
/// Same split as the editor's: if it can be a function of state, it goes
/// somewhere it can be tested.
///
/// @param cvars - the table the four were registered into
/// @return what the game's `World::mix` should be, with a variable that is
/// missing or is not a number reading as full volume rather than as silence -
/// a typo should not turn the sound off
pub(crate) fn volumes(cvars: &Cvars) -> Mix {
	let volume = |name: &str| cvars.float(name).unwrap_or(1.0);

	Mix {
		master: volume(colby_audio::MASTER),
		effects: volume(colby_audio::EFFECTS),
		music: volume(colby_audio::MUSIC),
		interface: volume(colby_audio::INTERFACE),
	}
}

/// `snd.play <name> [volume]` - plays a sound with no place in the world.
///
/// Flat rather than positioned, because a name typed at a console is not
/// standing anywhere. It is the cheapest end-to-end check there is: one line
/// reaches the registry, the voice table, the snapshot, the mixer and a driver.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn play_sound(world: *mut World, args: *const Args) {
	// SAFETY: as help.
	let world = unsafe { &mut *world };
	// SAFETY: as help.
	let args = unsafe { &*args };

	// @note: this guard cannot be caught by any mutation and is kept for the
	// message. Without it the empty name reaches `find`, which answers
	// `SoundId::NONE`, which the check below refuses - so the outcome is the
	// same and only the sentence differs. "wants the name of a sound" is what
	// somebody who typed `snd.play` alone needs to read, and "no sound of that
	// name is loaded" is not.
	let Some(name) = args.word(0) else {
		warn!("snd.play wants the name of a sound; snd.list says which there are");

		return;
	};

	let sound = world.sounds.find(name);
	if !sound.is_some() {
		warn!(name, "no sound of that name is loaded");

		return;
	}

	let volume = args.float(1).unwrap_or(1.0);
	let id = world
		.audio
		.play(Voice::flat(sound).volume(volume));

	if !id.is_some() {
		warn!(name, "every voice is busy; nothing was played");

		return;
	}

	info!(
		name,
		slot = id.slot(),
		volume,
		seconds = world.sounds.data(sound).seconds(),
		"playing"
	);
}

/// `snd.stop` - stops every voice.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn stop_sounds(world: *mut World, _args: *const Args) {
	// SAFETY: as help.
	let world = unsafe { &mut *world };
	let stopped = world.audio.len();

	world.audio.stop_all();
	info!(stopped, "everything stopped");
}

/// `snd.list` - reports what is loaded and what is playing.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn list_sounds(world: *mut World, _args: *const Args) {
	// SAFETY: as help.
	let world = unsafe { &mut *world };

	info!(
		sounds = world.sounds.len().saturating_sub(1),
		playing = world.audio.len(),
		refused = world.audio.dropped(),
		master = world.mix.master,
		"audio"
	);

	// the null entry is slot zero and is silence, which is not a sound anybody
	// compiled and not one anybody can play.
	for entry in world.sounds.iter().skip(1) {
		let data = entry.value();

		info!(
			name = entry.name(),
			seconds = data.seconds(),
			rate = data.rate,
			channels = data.channels,
			"sound"
		);
	}

	// a line each, up to a point. Sixty-four of them is what filling the table
	// looks like and it is not what somebody asking what is playing wants to
	// read; the count above already answered that question.
	for (id, voice) in world.audio.iter().take(LISTED_VOICES) {
		info!(
			slot = id.slot(),
			sound = world
				.sounds
				.get(voice.sound)
				.map_or("", Sound::name),
			head = voice.head,
			volume = voice.volume,
			looping = voice.looping,
			positioned = voice.positioned,
			"voice"
		);
	}

	let rest = world.audio.len().saturating_sub(LISTED_VOICES);
	if rest > 0 {
		info!(rest, "and more, not listed");
	}
}

/// The four that read or write a file.
///
/// Split off for the lint rather than for the shape, and the shape is better
/// for it: putting a world back replaces every table in it and needs the
/// solver, which a command cannot reach, so every one of these waits for the
/// frame loop rather than answering inside the line. @ref [`crate::saves`],
/// which takes them up and is where the four names are written down.
///
/// @param world - the table to register into
fn install_scenes(world: &mut World) {
	world.cvars.command(
		crate::saves::SAVE,
		console::defer,
		"write the world into saves/<name>.cscene",
	);
	world.cvars.command(
		crate::saves::WRITE,
		console::defer,
		"write the world into assets/scenes/<name>.scene, which the compiler picks up",
	);
	world.cvars.command(
		crate::saves::LOAD,
		console::defer,
		"put saves/<name>.cscene back, replacing this world",
	);
	world.cvars.command(
		crate::saves::PROP,
		console::defer,
		"write the scene registered as props/<name> into assets/props/<name>.scene",
	);
}

/// Runs one line as this machine, pointing where this screen points.
///
/// The ordinary way in, and the one every caller but the wire uses. @ref
/// [`run_as`] for what a line that came from somewhere else goes through.
///
/// @param world - the world the line is run against
/// @param line - one console line
pub(crate) fn run(world: &mut World, line: &str) {
	let (peer, aim) = (world.peer, world.pointing());

	run_as(world, peer, aim, line);
}

/// Runs one line as somebody, pointing wherever they were, containing anything
/// it throws.
///
/// **Two fields swapped around one call, and they are the whole of what makes
/// the console an RPC layer.** [`World::peer`] is who is asking - every
/// gameplay command reads it to find out whose player a nameless request is
/// about - and [`World::aim`] is where they were pointing, which is the thing
/// two angles could not say. A line that arrived from a peer is run with that
/// peer's pair; a line typed here is run with this machine's, which is what
/// makes a command written once mean the same thing on both paths.
///
/// **Both are put back afterwards, whatever happened.** A command that ended
/// the process leaves the world half written and the fields have to name this
/// machine again either way, and a command that panicked has not said anything
/// about who the next one is from.
///
/// A command can come from the game module, and gameplay code is code that is
/// expected to be wrong sometimes. A panic in one costs a message: unlike a
/// panic in `update` it does not park the module, because a bad command says
/// nothing about whether the rest of the build works.
///
/// @param world - the world the line is run against
/// @param peer - who is asking
/// @param aim - where they were pointing, or [`Aim::NONE`]
/// @param line - one console line
pub(crate) fn run_as(world: &mut World, peer: PeerId, aim: Aim, line: &str) {
	let (was_peer, was_aim) = (world.peer, world.aim);

	world.peer = peer;
	world.aim = aim;

	let result = catch_unwind(AssertUnwindSafe(|| console::run(world, line)));

	world.peer = was_peer;
	world.aim = was_aim;

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
/// Waits for the frame rather than answering inside the line, because the
/// path is resolved against the workspace and the workspace is the runtime's
/// rather than the world's: `exec scripts/demo.cfg` has to mean the same thing
/// wherever the executable was started from, and a command is handed nothing
/// that says where that is. @ref [`serve`], which takes it up.
pub(crate) const EXEC: &str = "exec";

/// Runs every config file somebody asked for since the last frame.
///
/// The frame loop's, beside the scene and the wire requests and for the same
/// reason. The one thing a line loses by waiting is its place in a
/// multi-statement line: `exec a.cfg; sim.pause 1` pauses first and runs the
/// file when the frame gets to it.
///
/// @param world - where the lines wait, and what the file is run against
/// @param workspace - what a relative path is resolved against
pub(crate) fn serve(world: &mut World, workspace: &Path) {
	for asked in take(world, &[EXEC]) {
		let Some(name) = asked.words.first() else {
			warn!("exec takes the path of a config file");

			continue;
		};

		let path = Path::new(name);
		let path = if path.is_absolute() {
			path.to_owned()
		} else {
			workspace.join(path)
		};

		console::exec(world, &path);
	}
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

/// The solver's default pass count, as the number a console variable holds.
fn passes() -> f32 {
	f32::from(u16::try_from(colby_physics::VELOCITY_PASSES).unwrap_or(u16::MAX))
}

/// `script.list` - reports every program the host has loaded.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn list_scripts(world: *mut World, _args: *const Args) {
	// SAFETY: as help.
	let world = unsafe { &*world };

	for entry in world.scripts.iter() {
		if entry.name().is_empty() {
			continue;
		}

		info!(
			name = entry.name(),
			revision = entry.revision(),
			lines = entry.value().source.lines().count(),
			world = Scripts::is_world(entry.name()),
			"program"
		);
	}

	info!(programs = world.scripts.len().saturating_sub(1), "loaded");
}

/// `script.reload [name]` - asks for a program to be built again.
///
/// Nothing is read off disk: the file is fine and what this moves is the
/// revision, which is the signal whoever is running the program watches. With
/// no name it moves every one of them, which is the form worth having on a
/// keyboard.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn reload_script(world: *mut World, args: *const Args) {
	// SAFETY: as help.
	let world = unsafe { &mut *world };
	// SAFETY: as help.
	let args = unsafe { &*args };

	let Some(name) = args.word(0) else {
		let names: Vec<String> = world
			.scripts
			.iter()
			.map(|entry| entry.name().to_owned())
			.collect();
		let mut moved = 0_usize;

		for name in names {
			let id = world.scripts.find(&name);
			if world.scripts.touch(id) {
				moved += 1;
			}
		}

		info!(programs = moved, "every program will be built again");

		return;
	};

	let id = world.scripts.find(name);
	if !id.is_some() {
		warn!(name, "no program of that name is loaded");

		return;
	}

	if world.scripts.touch(id) {
		info!(name, "will be built again");
	}
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

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::SoundData,
		glam::{Vec2, Vec3},
	};

	use super::*;

	/// A console command that writes down where the world said its caller was
	/// pointing, and then throws if it was told to.
	///
	/// The ray goes into two vectors nothing else here writes, so that all six
	/// numbers survive to be compared rather than a summary of them.
	///
	/// # Safety
	///
	/// As `ConsoleFn`: both pointers are live for the duration of the call.
	unsafe extern "C-unwind" fn whence(world: *mut World, args: *const Args) {
		// SAFETY: the console hands over a live world for the duration of the
		// call.
		let world = unsafe { &mut *world };
		// SAFETY: and live arguments beside it. Two blocks because two
		// dereferences in one is a lint.
		let args = unsafe { &*args };

		world.clear = Vec3::from_array(world.aim.origin);
		world.light = Vec3::from_array(world.aim.direction);

		assert!(args.rest() != "throw", "asked to");
	}

	/// A world with `whence` registered and a camera somewhere particular.
	fn asked() -> World {
		let mut world = World::new();

		world.camera.position = Vec3::new(3.0, 5.0, 9.0);
		world.camera.target = Vec3::new(1.0, 0.5, 0.0);
		world
			.ui
			.set_viewport(Vec2::new(1280.0, 720.0), 1.0);
		world
			.cvars
			.command("whence", whence, "writes down where its caller was pointing");

		world
	}

	#[test]
	fn a_line_typed_here_is_run_pointing_where_this_screen_points() {
		let mut world = asked();
		let pointing = world.pointing();

		assert!(pointing.is_some(), "this screen points somewhere, or this proves nothing");
		run(&mut world, "whence");
		assert!(
			world.clear.abs_diff_eq(pointing.start(), 1e-6)
				&& world.light.abs_diff_eq(pointing.forward(), 1e-6),
			"the command saw this screen's own ray"
		);
		assert!(!world.aim.is_some(), "and the field is nobody's again once the line has run");
	}

	#[test]
	fn a_line_run_as_somebody_else_is_run_with_their_aim_and_their_name() {
		let mut world = asked();
		// slot three, third occupant, said the way a runner has to say one:
		// minting is the world's and `PeerId::at` is not public.
		let mine = PeerId::from_bits((7 << 32) | 3);
		// deliberately not what this screen points along, and six numbers that
		// are six different numbers.
		let theirs = Aim {
			origin: [-4.0, 0.5, 12.0],
			direction: [0.6, -0.8, 0.25],
		};

		assert!(
			!world
				.pointing()
				.start()
				.abs_diff_eq(theirs.start(), 1e-3),
			"the two rays differ, or this test cannot tell them apart"
		);
		run_as(&mut world, mine, theirs, "whence");
		assert!(
			world.clear.abs_diff_eq(theirs.start(), 1e-6)
				&& world.light.abs_diff_eq(theirs.forward(), 1e-6),
			"the command saw the ray it was run with rather than this screen's"
		);
		assert!(world.peer.is_host(), "and this machine is itself again");
		assert!(!world.aim.is_some(), "pointing nowhere again");
	}

	#[test]
	fn a_command_that_panicked_does_not_leave_its_callers_aim_behind() {
		// the reason the two fields are put back before the panic is reported
		// rather than after: a line that threw has said nothing about who the
		// next one is from, and a world left holding a departed peer's ray
		// would hand it to whatever ran next.
		let mut world = asked();

		run_as(
			&mut world,
			PeerId::from_bits((5 << 32) | 2),
			Aim {
				origin: [1.0; 3],
				direction: [0.0, 1.0, 0.0],
			},
			"whence throw",
		);
		assert!(world.clear.abs_diff_eq(Vec3::ONE, 1e-6), "it ran, and it saw the aim");
		assert!(world.peer.is_host(), "and this machine is itself again");
		assert!(!world.aim.is_some(), "and pointing nowhere, though the command threw");
	}

	/// A table with the four volumes registered, as `install_audio` leaves it.
	fn table() -> World {
		let mut world = World::new();
		install_audio(&mut world);

		world
	}

	/// A table with everything the host registers, as a running engine has it.
	///
	/// The whole of `install` rather than the piece under test, because what
	/// several of these ask is whether a variable is registered *at all* - and
	/// a fixture that registered it itself would answer that for the fixture.
	fn engine() -> World {
		let mut world = World::new();
		install(&mut world);

		world
	}

	#[test]
	fn a_table_nobody_has_touched_reads_as_full_volume() {
		assert_eq!(volumes(&table().cvars), Mix::FULL, "silence is not a sensible default");
	}

	#[test]
	fn a_table_with_no_audio_variables_at_all_still_reads_as_full_volume() {
		// what a build whose `install_audio` never ran would look like, and
		// what a config file naming a variable this build dropped looks like.
		// Reading nothing as silence would be an engine that went quiet for a
		// reason nobody could find.
		assert_eq!(volumes(&World::new().cvars), Mix::FULL);
	}

	#[test]
	fn each_variable_reaches_its_own_field_and_no_other() {
		// the mistake this catches is two of them wired to the same name,
		// which is invisible until somebody turns the music down and the
		// footsteps go with it.
		let cases = [
			(colby_audio::MASTER, Mix { master: 0.25, ..Mix::FULL }),
			(colby_audio::EFFECTS, Mix { effects: 0.25, ..Mix::FULL }),
			(colby_audio::MUSIC, Mix { music: 0.25, ..Mix::FULL }),
			(colby_audio::INTERFACE, Mix { interface: 0.25, ..Mix::FULL }),
		];

		for (name, expected) in cases {
			let mut world = table();
			console::run(&mut world, &format!("{name} 0.25"));

			assert_eq!(volumes(&world.cvars), expected, "{name} moved the wrong field");
		}
	}

	#[test]
	fn the_volumes_are_written_into_the_config_and_the_debug_variables_are_not() {
		// somebody who turned the sound down meant it. Somebody who turned the
		// collision outlines on did not mean to find them on tomorrow.
		let world = engine();
		let archived: Vec<&str> = world
			.cvars
			.iter()
			.filter(|entry| entry.is_archived())
			.map(colby_core::abi::cvar::Entry::name)
			.collect();

		for name in [
			colby_audio::MASTER,
			colby_audio::EFFECTS,
			colby_audio::MUSIC,
			colby_audio::INTERFACE,
			// and the tick rate, which is the same kind of thing: a property
			// of the project rather than a knob somebody turned for a minute.
			crate::app::RATE,
		] {
			assert!(archived.contains(&name), "{name} should survive a restart");
		}

		for name in [crate::app::PAUSE, crate::app::SPEED] {
			assert!(
				!archived.contains(&name),
				"{name} is a thing somebody does for a minute, not a setting"
			);
		}
	}

	#[test]
	fn the_rate_a_console_asks_for_is_the_rate_the_clock_is_given() {
		let cases = [
			("60", 60),
			("120", 120),
			("30", 30),
			// both ends of the range, and both of the ways past it. A console
			// takes what it is typed, so every number a person can type has to
			// come out as a rate somebody could have meant.
			("1", 1),
			("1000", 1000),
			("0", 1),
			("-9", 1),
			("99999", 1000),
			("9223372036854775807", 1000),
		];

		for (typed, expected) in cases {
			let mut world = engine();
			console::run(&mut world, &format!("{} {typed}", crate::app::RATE));

			assert_eq!(
				crate::app::rate(&world.cvars).hz(),
				expected,
				"typing {typed} should run at {expected} a second"
			);
		}
	}

	#[test]
	fn a_rate_that_is_not_a_number_leaves_the_one_that_was_there() {
		// the variable is an integer, so the table refuses the word rather
		// than taking it, and what a refusal has to leave behind is the rate
		// the world was already running at.
		let mut world = engine();
		console::run(&mut world, &format!("{} 120", crate::app::RATE));
		console::run(&mut world, &format!("{} soon", crate::app::RATE));

		assert_eq!(crate::app::rate(&world.cvars).hz(), 120, "the word changed nothing");
	}

	#[test]
	fn playing_a_sound_by_name_puts_a_voice_in_the_table() {
		// the console command is the cheapest end-to-end reach there is, and
		// it is also the one thing in this file that is a function of a world.
		let mut world = table();
		world.sounds.insert("sounds/test", SoundData {
			samples: vec![0; 100],
			rate: 1000,
			channels: 1,
		});

		console::run(&mut world, "snd.play sounds/test 0.5");

		assert_eq!(world.audio.len(), 1, "something is playing");

		let (_, voice) = world.audio.iter().next().expect("just played");

		assert!((voice.volume - 0.5).abs() < f32::EPSILON, "at the volume that was asked for");
		assert!(!voice.positioned, "and with no place in the world, having been typed");
	}

	#[test]
	fn playing_a_name_nothing_answers_to_plays_nothing() {
		let mut world = table();

		console::run(&mut world, "snd.play sounds/nothing");

		assert!(world.audio.is_empty(), "a typo is not a voice holding a slot");
	}

	#[test]
	fn playing_with_no_name_at_all_plays_nothing() {
		let mut world = table();

		console::run(&mut world, "snd.play");

		assert!(world.audio.is_empty());
	}

	#[test]
	fn stopping_empties_the_table() {
		let mut world = table();
		world.sounds.insert("sounds/test", SoundData {
			samples: vec![0; 100],
			rate: 1000,
			channels: 1,
		});

		for _ in 0..4 {
			console::run(&mut world, "snd.play sounds/test");
		}

		assert_eq!(world.audio.len(), 4);
		console::run(&mut world, "snd.stop");
		assert!(world.audio.is_empty(), "everything, not the first one");
	}
}
