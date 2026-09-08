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
//! The way out is the project's `settings.cfg`, written when the process stops
//! and read when it starts. It is a config script, not a serialization format:
//! the same parser reads it, it is meant to be edited by hand, and colby gains
//! no dependency for it. Writing it as JSON would have cost one.
//!
//! @note: `--shot` has no console and does not read `settings.cfg`. A
//! screenshot is meant to be the same picture on every machine, and a file
//! holding whatever someone last typed is the opposite of that.

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
		Aim, Args, Asked, Bound, Cvars, Mix, NavSettings, PeerId, Scripts, Sound, Value, Voice,
		World, console, cvar::Owner, input, navmesh,
	},
	error, info, warn,
};

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
	/// Called after the game module is loaded, so that a `settings.cfg` naming
	/// a variable the *game* registers finds it there.
	///
	/// @param world - the world whose table everything is registered into
	/// @param archive - the file the archived variables are kept in, which is
	/// the project's
	pub(crate) fn open(world: &mut World, archive: &Path) -> Self {
		let archive = archive.to_owned();

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

/// What the archive last said a variable was, without running it.
///
/// For the one variable that is needed before there is a table to run the
/// archive against: the graphics API a window is made with is decided before
/// the world comes up, because the screen that shows the world coming up
/// needs a device to be drawn on. The same parser the archive is run with,
/// so a value quoted in the file reads here as it reads there; the last line
/// naming the variable wins, as it does when the file is run.
///
/// @param archive - the file, which need not exist
/// @param name - the variable
/// @return its value's first word, if a line in the file sets it
pub(crate) fn archived(archive: &Path, name: &str) -> Option<String> {
	let text = fs::read_to_string(archive).ok()?;

	console::statements(&text)
		.into_iter()
		.rev()
		.find(|words| words.first().is_some_and(|word| word == name))
		.and_then(|words| words.get(1).cloned())
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
		.command(EXEC, console::defer, "run a config file, relative to the project");
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
	// saved, unlike the mode beside it: which mode a session opens in is that
	// session's business, and whether a stop keeps what the game did is a way
	// of working that somebody settles on once.
	world.cvars.saved(
		crate::mode::KEEP,
		Value::Bool(false),
		"keep what the game did when play stops, instead of putting the world back",
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
	world.cvars.var(
		colby_physics::debug::WATER,
		Value::Bool(false),
		"outline every fluid and hatch the surface it is filled to",
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

	install_render(world);
	install_input(world);
	install_nav(world);
	install_scenes(world);
	install_code(world);
	install_audio(world);
	install_net(world);
	install_scripts(world);
}

/// What every project starts with bound, and to what.
///
/// **An engine that shipped no actions at all would make every project write
/// these eight lines**, and one of them could not: a Lua program can register
/// a *command* and nothing else, so a program under `assets/scripts/` that
/// wanted an action would have had to ask somebody else to declare it. s&box
/// ships a list for exactly this reason - "games that don't define any input
/// actions will get a bunch of default actions given to them" - and Godot
/// ships its `ui_*` set.
///
/// Eight rather than s&box's twenty: the movement four are in every reference
/// there is, and jump, crouch, use and attack are the four after them. An
/// inventory slot is a decision about a *game*.
const ACTIONS: &[(&str, &str, &str)] = &[
	("forward", "w", "walk forwards"),
	("back", "s", "walk backwards"),
	("left", "a", "step to the left"),
	("right", "d", "step to the right"),
	("jump", "space", "jump"),
	("crouch", "control", "crouch"),
	("use", "e", "use whatever is in front"),
	("attack", "mouse1", "attack"),
];

/// The input map: one saved variable an action, and a command to read it back.
///
/// **Saved, like the tick rate and unlike the debug outlines.** Which key does
/// what is the most personal setting a project has, and coming back to a world
/// where it has been forgotten is the surprise. `reset in.jump` puts the
/// default back, and `help in.` lists every action there is - which is the
/// listing a `bind` command of its own would have had to grow.
///
/// @param world - the table to register into
fn install_input(world: &mut World) {
	for (action, key, help) in ACTIONS {
		world.cvars.saved(
			&format!("{}{action}", input::PREFIX),
			Value::Text((*key).to_owned()),
			help,
		);
	}

	world
		.cvars
		.command("in.list", actions, "report every action and the key it answers to");
}

/// `in.list` - reports every action, its key, and whether it is down.
///
/// One line rather than one an action, because the answer to "why is my jump
/// not working" is the whole table and not one row of it. **The state is in it
/// too**: an action bound to a key nobody is pressing and an action bound to
/// nothing look identical from outside, and this is the one place that can
/// tell them apart.
///
/// # Safety
///
/// As [`help`].
unsafe extern "C-unwind" fn actions(world: *mut World, _args: *const Args) {
	// SAFETY: as help.
	let world = unsafe { &mut *world };
	// collected first, because the loop below reads the whole world through
	// `bound` and cannot be holding a piece of it at the time.
	let names = input::actions(&world.cvars)
		.map(str::to_owned)
		.collect::<Vec<_>>();
	let mut bound = 0_usize;
	let mut listed = Vec::with_capacity(names.len());

	for action in &names {
		let key = world.bound(action);
		bound += usize::from(key.is_some());

		listed.push(format!(
			"{action}={}{}",
			key.map_or("-", Bound::name),
			if world.action(action) { "*" } else { "" }
		));
	}

	info!(actions = names.len(), bound, map = listed.join(" "), "the input map");
}
/// The navmesh's variables: how big the thing that walks is, and one tool.
///
/// **The four settings are saved and the drawing is not**, which is the split
/// `sim.rate` and the debug outlines already keep: how big an agent is belongs
/// to the project and coming back to a world baked for a different one is the
/// surprise; a grid of crosses over the ground is a tool, and coming back to a
/// session still covered in them is the surprise the other way.
///
/// There is deliberately no `nav.bake`. A bake happens when the world it is
/// baked from changes, and the settings are part of that world - so turning
/// any of the five *is* the way to ask for one, and a command that did the
/// same thing would be a second way for the two to disagree.
///
/// @param world - the table to register into
fn install_nav(world: &mut World) {
	world.cvars.saved(
		crate::nav::CELL,
		Value::Float(navmesh::CELL),
		"how wide one cell of the navmesh is, in world units",
	);
	world.cvars.saved(
		crate::nav::RADIUS,
		Value::Float(navmesh::RADIUS),
		"how far from a wall a thing that walks must stay, in world units",
	);
	world.cvars.saved(
		crate::nav::STEP,
		Value::Float(NavSettings::DEFAULT.step),
		"the tallest lip a thing that walks climbs, in world units",
	);
	world.cvars.saved(
		crate::nav::SLOPE,
		Value::Float(NavSettings::DEFAULT.slope),
		"the steepest ground it stands on, as the cosine of the angle from up",
	);
	world.cvars.var(
		crate::nav::DRAW,
		Value::Bool(false),
		"mark every cell of the navmesh anything may stand on",
	);
	world.cvars.var(
		crate::nav::SHOW,
		Value::Text(String::new()),
		"draw the path between two places on the ground: <x> <z> <x> <z>",
	);
}

/// The renderer's variables: the shadows, the lights, and which graphics APIs
/// to draw with.
///
/// Registered whether or not there is a window, like everything else here: a
/// host writes them into its config untouched, and a config is one file for a
/// project rather than one per kind of process.
///
/// @param world - the table to register into
fn install_render(world: &mut World) {
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
	// how many point and cone lights one frame may carry. A ceiling rather
	// than a switch: the array is the frame's whole budget for them, so the
	// number a machine can afford is the thing worth being able to say.
	world.cvars.var(
		colby_engine::scene::LAMPS,
		Value::Float(colby_engine::scene::DEFAULT_LAMPS),
		"how many of the nearest point and cone lights a frame draws with",
	);
	// how many samples a pixel of the world is drawn with. **Saved**, unlike
	// the two above and like `r.backend`: how much a machine can afford to
	// spend on smooth edges is a property of the machine rather than of a
	// session, which is what every engine that has this setting treats it as.
	// Unlike `r.backend` it takes effect at once - the scene rebuilds the
	// eight pipelines and the two targets that have to agree about it on the
	// next frame - because a picture setting nobody can compare by turning it
	// on and off is one nobody turns on.
	world.cvars.saved(
		colby_engine::scene::MSAA,
		Value::Float(colby_engine::scene::DEFAULT_MSAA),
		"how many samples a pixel is drawn with: one is off, anything more is four",
	);
	// which graphics APIs the window may draw with. Saved, because it is a
	// property of the machine rather than of a session; and read once, when
	// the device is made, so a value typed at a running window takes effect
	// at the next start - which is what every engine does with this setting.
	// The environment's `WGPU_BACKEND` overrides it for one run.
	world.cvars.saved(
		colby_engine::gpu::BACKEND,
		Value::Text(colby_engine::gpu::AUTO.to_owned()),
		"which graphics APIs to consider at the next start: auto, or vulkan, dx12, metal, gl in \
		 a comma list",
	);
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
/// which takes them up and is where the seven names are written down.
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
	world.cvars.command(
		crate::saves::MATERIAL,
		console::defer,
		"write the material registered as materials/<name> back into assets/",
	);
	world.cvars.command(
		crate::saves::MODEL,
		console::defer,
		"start an import sidecar beside the source that compiles to <name>",
	);
	world.cvars.command(
		crate::saves::BLOCKS,
		console::defer,
		"write the meshes baked under maps/<name>/ into assets/ as .obj sources",
	);
}

/// The one command and the two variables that open a source in an editor.
///
/// Saved rather than plain, because which editor somebody uses is a property
/// of the machine and not of a session - the same argument `r.backend` is
/// registered with. It is a *project's* `settings.cfg` all the same, which is
/// where every setting colby has lives; @ref [`crate::code`] for the rest.
///
/// @param world - the table to register into
fn install_code(world: &mut World) {
	world.cvars.command(
		crate::code::OPEN,
		console::defer,
		"open the source that compiles to <name> in an editor, at [line] if one is given",
	);
	world.cvars.saved(
		crate::code::EDITOR,
		Value::Text(String::new()),
		"the program `code.open` starts; empty hands the file to the platform instead",
	);
	world.cvars.saved(
		crate::code::ARGS,
		Value::Text(crate::code::DEFAULT_ARGS.to_owned()),
		"what to pass it, with {file}, {line}, {col} and {project} filled in; quote the whole \
		 value, as \"{project} {file}:{line}:{col}\" for zed or sublime",
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
/// path is resolved against the project and the project is the runtime's
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
/// @param root - the project directory a relative path is resolved against
pub(crate) fn serve(world: &mut World, root: &Path) {
	for asked in take(world, &[EXEC]) {
		let Some(name) = asked.words.first() else {
			warn!("exec takes the path of a config file");

			continue;
		};

		let path = Path::new(name);
		let path = if path.is_absolute() {
			path.to_owned()
		} else {
			root.join(path)
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

	/// What a reader falls back to when it finds no entry, beside the value
	/// the registration gives it.
	///
	/// **The two used to be able to drift and nothing said so.** A picture and
	/// a recording had no table at all, so every one of these readers took its
	/// own branch and the constant beside it was the only thing that ran;
	/// since `PERF-4` the table is registered in every front, so both branches
	/// are live and they have to agree. A row here is one reader, named in the
	/// comment beside it.
	const FALLBACKS: &[(&str, Value)] = &[
		// `crate::app::pay`
		(crate::app::PAUSE, Value::Bool(false)),
		(crate::app::SPEED, Value::Float(1.0)),
		// `crate::mode::wanted` and `crate::mode::kept`
		(crate::mode::EDIT, Value::Bool(false)),
		(crate::mode::KEEP, Value::Bool(false)),
		// `colby_physics::debug::draw`
		(colby_physics::debug::SHAPES, Value::Bool(false)),
		(colby_physics::debug::CONTACTS, Value::Bool(false)),
		(colby_physics::debug::JOINTS, Value::Bool(false)),
		(colby_physics::debug::WATER, Value::Bool(false)),
		// `colby_engine::scene::Scene::upload`
		(colby_engine::shadow::ENABLED, Value::Bool(true)),
		(colby_engine::shadow::TINT, Value::Bool(false)),
		// `crate::console::volumes`, all four
		(colby_audio::MASTER, Value::Float(1.0)),
		(colby_audio::EFFECTS, Value::Float(1.0)),
		(colby_audio::MUSIC, Value::Float(1.0)),
		(colby_audio::INTERFACE, Value::Float(1.0)),
	];

	#[test]
	fn every_reader_falls_back_to_what_the_registration_would_have_given_it() {
		let mut world = World::default();
		install(&mut world);

		for (name, expected) in FALLBACKS {
			let entry = world
				.cvars
				.iter()
				.find(|entry| entry.name() == *name)
				.unwrap_or_else(|| panic!("{name} is not registered at all"));

			assert_eq!(
				entry.value(),
				Some(expected),
				"{name} registers one number and its reader falls back to another"
			);
		}
	}

	#[test]
	fn the_four_numbers_a_reader_computes_come_out_the_same_either_way() {
		// the three rows above cannot cover: these readers do arithmetic on
		// what they find, so what has to match is the *answer* rather than the
		// value. A world with no table and a world with one have to agree.
		let mut world = World::default();
		let bare = volumes(&world.cvars);

		install(&mut world);

		assert_eq!(volumes(&world.cvars), bare, "the mixer hears something different");
		assert_eq!(
			world.cvars.float(colby_engine::scene::LAMPS),
			Some(colby_engine::scene::DEFAULT_LAMPS),
			"the lamp ceiling"
		);
		assert_eq!(
			world.cvars.float(colby_engine::scene::MSAA),
			Some(colby_engine::scene::DEFAULT_MSAA),
			"the sample count"
		);
		assert_eq!(
			world
				.cvars
				.float(colby_physics::PASSES)
				.map(asked_passes),
			Some(colby_physics::VELOCITY_PASSES),
			"the solver runs the same number of passes with a table and without one"
		);
	}

	/// What the solver would make of the registered value.
	///
	/// The solver's own reader is private to `colby_physics`; this is the same
	/// arithmetic, and the point of the row is that the registration lands on
	/// the default rather than a step either side of it.
	fn asked_passes(asked: f32) -> usize {
		let rounded = asked.round();

		if !rounded.is_finite() || rounded < 1.0 {
			return colby_physics::VELOCITY_PASSES;
		}

		(1..=colby_physics::MAX_PASSES)
			.rev()
			.find(|held| rounded >= f32::from(u16::try_from(*held).unwrap_or(u16::MAX)))
			.unwrap_or(colby_physics::VELOCITY_PASSES)
	}

	#[test]
	fn the_one_variable_a_window_needs_before_the_table_is_read_from_the_archive_by_name() {
		let dir = std::env::temp_dir().join("colby_console_archived");
		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(&dir).expect("a directory to work in");
		let archive = dir.join("settings.cfg");

		assert_eq!(archived(&archive, "r.backend"), None, "no file is no answer");

		fs::write(
			&archive,
			"// written by colby
sim.rate 60
r.backend \"dx12, vulkan\"; snd.volume 1
r.backend auto
",
		)
		.expect("the archive");

		assert_eq!(
			archived(&archive, "r.backend").as_deref(),
			Some("auto"),
			"the last line naming it wins, as it does when the file is run"
		);
		assert_eq!(archived(&archive, "sim.rate").as_deref(), Some("60"));
		assert_eq!(
			archived(&archive, "snd.volume").as_deref(),
			Some("1"),
			"after a semicolon too"
		);
		assert_eq!(archived(&archive, "r.shadows"), None, "a name not in the file is no answer");

		fs::write(
			&archive,
			"r.backend \"dx12, vulkan\"
",
		)
		.expect("a quoted value");

		assert_eq!(
			archived(&archive, "r.backend").as_deref(),
			Some("dx12, vulkan"),
			"quotes come off, the way the parser takes them off for the table"
		);
	}

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
			// and the graphics API, which is a property of the machine.
			colby_engine::gpu::BACKEND,
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

	#[test]
	fn every_default_action_is_registered_and_bound_to_something() {
		let world = engine();

		for (action, key, _) in ACTIONS {
			let name = format!("{}{action}", input::PREFIX);
			let entry = world
				.cvars
				.get(&name)
				.unwrap_or_else(|| panic!("{name} is not registered"));

			assert!(entry.is_archived(), "{name} has to survive a restart");
			assert_eq!(
				world.bound(action).map(Bound::name),
				Some(*key),
				"{name} does not answer to the key it was registered with"
			);
		}
	}

	#[test]
	fn the_listing_command_is_not_itself_an_action() {
		// **a live run counted nine actions where there are eight**, because
		// `in.list` is a command under the same prefix. `in.` is a namespace
		// rather than a list - `sim.` holds a command and three variables - so
		// what makes an entry an action is that it has a value.
		let world = engine();
		let named = input::actions(&world.cvars).collect::<Vec<_>>();

		assert_eq!(named.len(), ACTIONS.len(), "{named:?}");
		assert!(!named.contains(&"list"), "the command is not one of them: {named:?}");
		assert_eq!(world.bound("list"), None);
	}

	#[test]
	fn a_rebinding_survives_a_restart_and_a_reset_undoes_it() {
		// the two halves of what makes this an input *map* rather than a
		// constant: the file is where a person's choice lives, and `reset` is
		// where the project's default lives.
		let mut world = engine();

		console::run(&mut world, &format!("{}jump q", input::PREFIX));

		assert_eq!(world.bound("jump").map(Bound::name), Some("q"));

		let name = format!("{}jump", input::PREFIX);

		assert!(
			world
				.cvars
				.iter()
				.filter(|entry| entry.is_archived())
				.any(|entry| entry.name() == name),
			"and it is written into settings.cfg with the rest"
		);

		console::run(&mut world, &format!("reset {}jump", input::PREFIX));

		assert_eq!(world.bound("jump").map(Bound::name), Some("space"), "back to the default");
	}

	#[test]
	fn an_action_bound_to_a_word_that_is_not_a_key_is_quiet() {
		// a console takes what it is typed, so `in.jump banana` is a thing a
		// person can do. The answer is an action that answers to nothing, not
		// a refused line and not a panic.
		let mut world = engine();

		console::run(&mut world, &format!("{}jump banana", input::PREFIX));

		assert_eq!(world.bound("jump"), None, "nothing answers for it");
		assert!(!world.action("jump"), "and it is never held");

		console::run(&mut world, &format!("{}jump \"\"", input::PREFIX));

		assert_eq!(world.bound("jump"), None, "and clearing it is the same answer");
	}

	#[test]
	fn the_map_can_be_read_back_from_the_console() {
		let mut world = engine();
		world
			.input
			.set_key(colby_core::abi::Key::Space, true);

		console::run(&mut world, "in.list");

		// the command reports rather than returns, so what is asserted is that
		// it ran against a world it could read and left it alone.
		assert!(world.action("jump"), "space is down and jump is bound to it");
		assert_eq!(world.bound("attack").map(Bound::name), Some("mouse1"));
	}
}
