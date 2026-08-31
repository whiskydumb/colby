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
	abi::{Args, Cvars, Mix, Sound, Value, Voice, World, console},
	error, info, warn,
};

/// The file archived variables are kept in.
const ARCHIVE: &str = "cvars.cfg";

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
/// for it: these are the only commands in the table that do not answer inside
/// the frame they were typed in. Every one of them leaves a note for the frame
/// loop instead. @ref [`crate::saves`].
///
/// @param world - the table to register into
fn install_scenes(world: &mut World) {
	world
		.cvars
		.command("scene.save", save_scene, "write the world into saves/<name>.cscene");
	world.cvars.command(
		"scene.write",
		write_scene,
		"write the world into assets/scenes/<name>.scene, which the compiler picks up",
	);
	world.cvars.command(
		"scene.load",
		load_scene,
		"put saves/<name>.cscene back, replacing this world",
	);
	world.cvars.command(
		"scene.prop",
		write_prop,
		"write the scene registered as props/<name> into assets/props/<name>.scene",
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

/// `scene.prop <name>` - writes one registered scene out as a prop.
///
/// The half of saving a contraption that needs a filesystem. The other half is
/// the game's: it cuts the piece out and registers it, because what is under
/// somebody's crosshair is not something the host can see. @ref
/// [`crate::saves::Request::Prop`].
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn write_prop(_world: *mut World, args: *const Args) {
	// SAFETY: as help.
	let args = unsafe { &*args };

	crate::saves::ask(crate::saves::Request::Prop(args.rest()));
}

/// `scene.write <name>` - writes the world out as a source somebody can edit.
///
/// The other end of the editor's work, and the one that goes back into the
/// repository rather than beside it. @ref [`crate::saves`] for the difference
/// between the two files this and `scene.save` produce.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn write_scene(_world: *mut World, args: *const Args) {
	// SAFETY: as help.
	let args = unsafe { &*args };

	crate::saves::ask(crate::saves::Request::Write(args.rest()));
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

#[cfg(test)]
mod tests {
	use colby_core::abi::SoundData;

	use super::*;

	/// A table with the four volumes registered, as `install_audio` leaves it.
	fn table() -> World {
		let mut world = World::new();
		install_audio(&mut world);

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
		let world = table();
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
		] {
			assert!(archived.contains(&name), "{name} should survive a restart");
		}
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
