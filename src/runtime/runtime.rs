//! One runtime, however it is driven.
//!
//! Four things drive a world in this engine - a window, a socket with nothing
//! on screen, a picture and a sound - and each of them used to stand the world
//! up by hand: the same solver, the same asset loop, the same module, the same
//! console and the same interpreter, in four copies that agreed by care. This
//! is the one copy. A front says what kind of process it is with a [`Front`],
//! [`Runtime::open`] builds everything in the one order, and the front keeps
//! only what is genuinely its own: a window and a clock, a socket loop, or a
//! fixed count of steps.
//!
//! **What is shared is the state and the step; what is not is the pace.** A
//! window paces steps off a vertical blank, a windowless end off a clock that
//! sleeps, a picture runs ninety and stops - and that difference is the front's
//! to keep, because it is the whole of what makes a screenshot deterministic
//! and a window smooth. [`Runtime::step`] runs one step and puts a message on
//! the wire; when to call it is not its business.

use std::{
	net::SocketAddr,
	time::{Duration, Instant},
};

use colby_asset::Project;
use colby_audio::Device;
use colby_core::{
	Result,
	abi::{Input, World, console, scene},
	debug, err, error,
	glam::Vec2,
	info,
	time::Rate,
};
#[cfg(feature = "editor")]
use colby_editor::Editor;
use colby_net::Slot;
use colby_physics::Simulation;
use colby_script::Vm;
use colby_ui::Interface;

use crate::{
	Build,
	assets::Assets,
	console::Console,
	game::Game,
	net::{Net, Standing},
	step,
};

/// The viewport a process with no window lays its documents out against.
///
/// A document laid out against a different size would put its boxes somewhere
/// else, so a picture, a sound and a windowless end all use one number: the
/// picture's, because the picture is the one anybody looks at. Nothing draws it
/// on the other two.
pub const VIEWPORT: Vec2 = Vec2::new(1280.0, 720.0);

/// What kind of process is being run, which is everything a stand-up differs
/// by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Front {
	/// A window: a console, an output device, the editor's variables, and a
	/// wire if it was asked for.
	///
	/// The one front whose viewport is not known when it opens - the renderer
	/// says what it is, a moment later - and the one front that goes on
	/// without a socket it could not bind: a window is somebody's session, and
	/// a session is worth having on its own.
	Window(Standing),

	/// A picture or a sound: no console, no device, no wire, no clock.
	///
	/// No console means no config file and none of the host's variables, so
	/// what comes out depends on the build and on nothing anybody typed - which
	/// is the whole reason a hash of it means anything.
	Fixed,

	/// A socket instead of a window, at the authority's end.
	///
	/// A console and a wire, no device and no editor. A socket it cannot bind
	/// is a stop rather than a shrug: this process exists to be on the wire.
	Host(u16),

	/// The same, at the other end: a client with nothing on screen.
	Join(SocketAddr),
}

impl Front {
	/// The viewport to lay documents out against before the first step, if it
	/// is known before a renderer exists.
	const fn viewport(self) -> Option<Vec2> {
		match self {
			| Self::Window(_) => None,
			| Self::Fixed | Self::Host(_) | Self::Join(_) => Some(VIEWPORT),
		}
	}

	/// Whether this process reads a terminal and a config file.
	const fn has_console(self) -> bool { !matches!(self, Self::Fixed) }

	/// Whether this process is a window.
	const fn is_window(self) -> bool { matches!(self, Self::Window(_)) }
}

/// Every piece of state a world needs to run, whatever is driving it.
pub struct Runtime {
	/// The world. Boxed: the entity table alone is tens of kilobytes, and this
	/// is handed across the module boundary by pointer anyway.
	pub world: Box<World>,

	/// The physics. Boxed because the world holds its address: the table of
	/// queries installed into `world` points here, and a value that moved
	/// would leave that pointer behind. @ref `colby_physics`.
	pub(crate) simulation: Box<Simulation>,

	/// The compile-and-load loop over the two asset trees.
	pub(crate) assets: Assets,

	/// The loaded game. `None` only once [`close`](Self::close) has run: the
	/// module is dropped there rather than with the rest, because it may still
	/// be running code from the image and has to go before anything it could
	/// be holding does.
	pub(crate) game: Option<Game>,

	/// The interpreter, for the documents' logic and the world's own programs.
	pub(crate) scripts: Vm,

	/// The game's own interface. Always present, and only able to draw once a
	/// device exists to attach it to.
	pub(crate) interface: Interface,

	/// The terminal and the config file, for a front that has them.
	pub(crate) console: Option<Console>,

	/// The socket and the conversation over it, for a front that is on a wire.
	pub(crate) net: Option<Net>,

	/// The output device and the mixer feeding it. A window's, and only when
	/// one opened: an engine whose picture works and whose sound did not start
	/// is worth running, and the log says which half is missing.
	pub(crate) audio: Option<Device>,

	/// The world as a snapshot describes it, taken down once a step rather
	/// than allocated per snapshot. Empty off the wire.
	records: Vec<Slot>,

	/// When this runtime opened, which is the clock the wire is on.
	///
	/// Real time rather than simulated: a round-trip estimate measured in
	/// simulated seconds would shrink whenever the machine stalled, and the
	/// number is about the wire rather than about the world.
	started: Instant,

	/// The project everything on disk resolves against.
	project: Project,
}

impl Runtime {
	/// Brings a world up, the one way there is.
	///
	/// The order is the window's, which was the most demanding of the four:
	/// the assets before the module, so `init` finds its meshes by name; the
	/// host's variables before the module, so they are the engine's and
	/// survive a reload; the socket before the module, so a client has said
	/// what it is before `init` reads `World::peer`; the editor's variables
	/// before the module for the reason the host's are; the config after the
	/// module, because a line in it may name a variable the game registered.
	///
	/// @param front - what kind of process this is
	/// @param project - the project: where the assets, the saves, the config
	/// and the game crate are, and what the world starts as
	/// @param build - what the build script knew, which says where the engine
	/// is: the game crate is mounted there and built there
	/// @return the runtime, or the first thing that would not come up
	pub fn open(front: Front, project: &Project, build: &Build) -> Result<Self> {
		// boxed and installed before anything else touches the world: the
		// world keeps this address. Once, here, and never again - the pointers
		// in the table address this executable rather than the game module, so
		// no reload disturbs them, which is the whole difference between this
		// and a console command and is why one has to be forgotten on unload
		// and the other does not.
		let mut simulation = Box::new(Simulation::new());
		let mut world = Box::<World>::default();
		world.install_physics(simulation.table());

		let mut assets = Assets::of(project);
		assets.sync(&mut world);
		start(&mut world, &mut simulation, project)?;

		if let Some(viewport) = front.viewport() {
			world.ui.set_viewport(viewport, 1.0);
			world.aspect = viewport.x / viewport.y;
		}

		// after the assets rather than before them, so the first copy into the
		// mixer's bank finds a registry that is already full.
		let audio = if front.is_window() { listen(&world) } else { None };

		if front.has_console() {
			crate::console::install(&mut world);
		}

		let net = connect(front, &mut world)?;

		#[cfg(feature = "editor")]
		if front.is_window() {
			// registered here rather than in `Editor::new`, so that they exist
			// whether or not a window was ever made - and before the module,
			// so that they are the engine's.
			Editor::install(&mut world);
		}

		// a project without a game crate has no module, and runs on its scenes
		// and its programs. @ref `Game::open` for the build that links one in.
		let module = project.game().is_some().then(|| project.module());

		if module.is_none() {
			info!(
				project = project.id(),
				"no game crate; the world is its scenes and its programs"
			);
		}

		// mounted in the engine's workspace and built if no image of it exists
		// yet, before it is loaded. @ref `crate::mount` for why a module has
		// to be built there and nowhere else.
		prepare_module(project, build, module.as_deref())?;

		let game = Game::open(&mut world, module.as_deref())?;

		// last of the three that register, because a line in it may name a
		// variable the game registered a moment ago.
		let console = front
			.has_console()
			.then(|| Console::open(&mut world, &project.settings()));

		// a run that cannot start its interpreter is a run that stops, whatever
		// the front: a picture, a sound and a digest are only comparable when
		// the same programs ran, and a window without its documents' logic is
		// a different window rather than a lesser one.
		let scripts = Vm::new(console::defer)?;

		Ok(Self {
			world,
			simulation,
			assets,
			game,
			scripts,
			interface: Interface::new(),
			console,
			net,
			audio,
			records: Vec::new(),
			started: Instant::now(),
			project: project.clone(),
		})
	}

	/// Everything that happens once a frame and outside a step.
	///
	/// The asset trees, the terminal, every line a command left waiting, and
	/// the socket - in that order, and all of it before any step runs: a scene
	/// load replaces every table in the world, and doing that halfway through
	/// a step would leave the rest of the step running against a world its
	/// first half never saw. A window calls this after checking for a rebuilt
	/// module and before its own edges; a windowless end calls it and little
	/// else.
	pub fn poll(&mut self) {
		// the mixer keeps its own copy of every sound, so a pass over the tree
		// is also when it has to be told. Only when the pass really ran: four
		// times a second rather than sixty, which is what keeps the lock the
		// audio callback wants out of its way.
		if self.assets.poll(&mut self.world)
			&& let Some(audio) = self.audio.as_mut()
		{
			let copied = audio.load(&self.world.sounds);
			if copied > 0 {
				debug!(copied, samples = audio.samples(), "sounds copied into the mixer");
			}
		}

		if let Some(console) = self.console.as_ref() {
			console.poll(&mut self.world);
		}

		// then every line a command left waiting, each taken up by the
		// subsystem that registered its name: a config file to run, a scene to
		// write or put back, a word for the wire. @ref
		// `colby_core::abi::console::Asked`.
		crate::console::serve(&mut self.world, self.project.root());
		crate::saves::serve(&mut self.world, &mut self.simulation, &self.project);
		crate::net::serve(&mut self.world, self.net.as_mut());

		// and everything the socket is holding, before any step: what a step
		// runs against is what had arrived when it started rather than
		// whatever turned up halfway through. This is where a client learns
		// who it is, where what the host has already run stops being resent,
		// and where a line somebody else typed is run. @ref `crate::net::hear`.
		if let Some(net) = self.net.as_mut() {
			crate::net::hear(net, &mut self.world, self.started.elapsed());
		}
	}

	/// Advances the world by one step and puts a message out for it.
	///
	/// Out once a step rather than once a frame, because a message has to mean
	/// "this is where things stand at this moment" and a frame rate is not a
	/// moment. @ref `crate::step::run` for the step itself.
	///
	/// @param input - everything that has arrived since the previous step; the
	/// edges in it are consumed here
	/// @param rate - how long this step is
	/// @param time - the simulated time this step ends at, in seconds
	/// @param editing - whether the world is being edited rather than played
	/// @param moment - where the wire's clock stands for this step
	/// @return where it stands for the next one: exactly one step on, whatever
	/// the real clock says, so that the renderer's blend between two steps is
	/// asked to cover a gap that is one step wide
	pub fn step(
		&mut self,
		input: &mut Input,
		rate: Rate,
		time: f32,
		editing: bool,
		moment: Duration,
	) -> Duration {
		step::run(
			&mut self.world,
			step::Parts {
				game: self.game.as_mut(),
				interface: &mut self.interface,
				scripts: Some(&mut self.scripts),
				simulation: self.simulation.as_mut(),
				audio: self.audio.as_mut(),
				// the endpoint and the clock it is on, so that a world a host
				// described lands in this one. @ref `Net::arrive`, which
				// refuses a host anyway - so at that end this is which answer
				// is given rather than whether one is.
				wire: wired(self.net.as_mut(), moment),
			},
			input,
			rate,
			time,
			editing,
		);

		if let Some(net) = self.net.as_mut() {
			crate::net::tell(
				net,
				&self.world,
				&mut self.records,
				rate.hz(),
				self.started.elapsed(),
			);
		}

		moment.saturating_add(rate.step())
	}

	/// Writes the config out and shuts the game down, in that order.
	///
	/// The config before the game, while its variables are still in the table
	/// to be written out; and the module dropped here rather than with the
	/// rest, because it may still be running code from the image and has to
	/// go before anything it could be holding does.
	pub fn close(&mut self) {
		if let Some(console) = self.console.as_ref() {
			console.close(&self.world);
		}

		if let Some(game) = self.game.as_mut() {
			game.close(&mut self.world);
		}

		self.game = None;
	}

	/// How long this runtime has been open, which is the clock the wire is on.
	#[must_use]
	pub fn now(&self) -> Duration { self.started.elapsed() }

	/// Whether this end takes its world from somebody else.
	///
	/// What decides the rate: a client runs the one every host runs, whatever
	/// its own console says. @ref `crate::app::paced`.
	#[must_use]
	pub fn following(&self) -> bool {
		self.net
			.as_ref()
			.is_some_and(|net| !net.hosting())
	}

	/// The project everything on disk resolves against.
	#[must_use]
	pub fn project(&self) -> &Project { &self.project }
}

/// Makes sure the project's module can be loaded: mounted in the engine's
/// workspace, and built when no image of it exists yet.
///
/// A project opened for the first time on this engine has no `<id>_game.dll`
/// beside the executable, and loading has to wait for one; the build is the
/// same one the watcher runs on an edit, and it inherits the terminal so that
/// a compile error lands in front of whoever is opening the project. Dead
/// mounts are swept first, because one of those fails every cargo command.
///
/// @param project - whose module
/// @param build - where the engine is
/// @param module - the module's name, or nothing for a project with no crate
#[cfg(feature = "hot_reload")]
fn prepare_module(project: &Project, build: &Build, module: Option<&str>) -> Result {
	let Some(module) = module else {
		return Ok(());
	};

	crate::mount::sweep(&build.engine);
	crate::mount::mount(project, &build.engine)?;

	if !colby_core::mods::path::from_name(module)?.is_file() {
		crate::watch::build(build)?;
	}

	Ok(())
}

/// Nothing to prepare: the game is linked in, whatever a project says.
#[cfg(not(feature = "hot_reload"))]
fn prepare_module(_project: &Project, _build: &Build, _module: Option<&str>) -> Result { Ok(()) }

/// Puts the world the project starts as in place, if its file names one.
///
/// After the assets and before the module, so that a game's `init` finds the
/// map there rather than having to make it - which is also what lets a project
/// with no game crate at all have a world. A named scene the registry does not
/// hold is a stop rather than an empty world: the project said what it starts
/// as, and a picture of nothing is not that.
///
/// @param world - the world, empty until now
/// @param simulation - the solver, told to forget what it derived, which is the
/// obligation every restore leaves and here nothing yet
/// @param project - whose file may name a scene
fn start(world: &mut World, simulation: &mut Simulation, project: &Project) -> Result {
	let Some(name) = project.startup_scene() else {
		return Ok(());
	};

	let id = world.scenes.find(name);

	if !id.is_some() {
		return Err(err!(Asset(
			"the startup scene {name} is not among the compiled scenes; it compiles from \
			 assets/{name}.scene"
		)));
	}

	let data = world.scenes.data(id).clone();
	let put = scene::restore(world, &data)?;
	simulation.forget();

	info!(
		scene = name,
		entities = put.things,
		bodies = put.solids,
		joints = put.links,
		"the world starts as the project says"
	);

	Ok(())
}

/// Opens the output device, if the machine has one.
///
/// A log line rather than a stop when it has none: an engine whose picture
/// works and whose sound did not start is worth running.
fn listen(world: &World) -> Option<Device> {
	match Device::open() {
		| Ok(mut device) => {
			device.load(&world.sounds);

			Some(device)
		},
		| Err(error) => {
			error!(%error, "no output device; nothing will make a sound");

			None
		},
	}
}

/// Opens the socket a front asked for, and says which end of it this is.
///
/// **A window can be either end of a wire**, and which one it is has to be
/// decided before the module loads: a window that serves stays the authority
/// its world already thinks it is, and one that connects stops being it before
/// the game module ever reads the field. A window that could not bind goes on
/// alone with a line saying so; a windowless end that could not is a stop,
/// because being on the wire is the whole of what it is for.
///
/// @param front - what kind of process this is
/// @param world - the world, told when it is no longer the authority
/// @return the endpoint, or nothing for a process that is on no wire
fn connect(front: Front, world: &mut World) -> Result<Option<Net>> {
	// how bad the wire is is a console variable and the seed for it is one
	// too, which is why this comes after the host's variables are registered.
	let seed = crate::net::seed(&world.cvars);
	let opened = match front {
		| Front::Window(Standing::Serving(port)) | Front::Host(port) =>
			Net::host(port, seed).map(|net| (net, true)),
		| Front::Window(Standing::Talking(address)) | Front::Join(address) =>
			Net::connect(address, seed).map(|net| (net, false)),
		| Front::Window(Standing::Alone) | Front::Fixed => return Ok(None),
	};

	let (net, hosting) = match opened {
		| Ok(pair) => pair,
		| Err(error) if front.is_window() => {
			error!(%error, "no socket; this window is on its own");

			return Ok(None);
		},
		| Err(error) => return Err(error),
	};

	if !hosting {
		// and this process stops being the authority, which is the one thing
		// about an end that connected that nothing else could work out. @ref
		// `crate::net::joined`.
		crate::net::joined(world);
	}

	Ok(Some(net))
}

/// The endpoint and the moment, for the step about to run.
///
/// Its own function so that *which* clock the moment comes from is something
/// a test can ask about. It comes from the caller, which read it once for the
/// frame; reading it here would put a real-time sample inside the step loop,
/// and two steps in one frame would then be microseconds apart where the
/// renderer's blend assumes a step. @ref `step::Wired`.
///
/// @param net - the endpoint, if this process has one
/// @param moment - where the wire's clock stands for this step
/// @return the pair, or nothing when this process is on no wire
fn wired(net: Option<&mut Net>, moment: Duration) -> Option<step::Wired<'_>> {
	let net = net?;

	Some(step::Wired { net, now: moment })
}

#[cfg(test)]
mod tests {
	use std::{
		cell::RefCell,
		net::{IpAddr, Ipv4Addr},
		rc::Rc,
	};

	use super::*;
	use crate::net::{Loopback, Wire};

	impl Runtime {
		/// A runtime with a world in it and nothing loaded, for the questions
		/// that are about this file rather than about a game.
		fn empty() -> Self {
			let simulation = Box::new(Simulation::new());
			let mut world = Box::<World>::default();
			world.install_physics(simulation.table());

			let scratch = std::env::temp_dir().join("colby_runtime_empty");

			Self {
				world,
				simulation,
				assets: Assets::at(scratch.join("assets"), scratch.join("out")),
				game: None,
				scripts: Vm::new(console::defer).expect("the interpreter starts"),
				interface: Interface::new(),
				console: None,
				net: None,
				audio: None,
				records: Vec::new(),
				started: Instant::now(),
				project: Project::parse(
					&scratch,
					r#"{ "schema": 1, "engine": "0.1.0", "id": "empty", "name": "empty" }"#,
				)
				.expect("a project"),
			}
		}
	}

	#[test]
	fn a_window_has_no_viewport_until_its_renderer_says_and_the_rest_share_the_pictures() {
		assert_eq!(Front::Window(Standing::Alone).viewport(), None);

		for front in [
			Front::Fixed,
			Front::Host(1),
			Front::Join(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 1)),
		] {
			assert_eq!(front.viewport(), Some(VIEWPORT), "{front:?} lays out like a picture");
		}
	}

	#[test]
	fn only_a_picture_has_no_console() {
		// no config file and none of the host's variables, so that what comes
		// out depends on the build and on nothing anybody typed.
		assert!(!Front::Fixed.has_console());
		assert!(Front::Host(1).has_console());
		assert!(Front::Window(Standing::Alone).has_console());
	}

	#[test]
	fn the_wires_clock_moves_by_exactly_one_step_a_step() {
		// what the renderer's own blend between two steps assumes, and the
		// only thing that says the moment is not re-read from the real clock
		// inside the loop: two steps in one frame would then be microseconds
		// apart, and two a frame apart a whole frame apart, with one blend
		// asked to cover both.
		let mut runtime = Runtime::empty();
		let mut input = Input::default();
		let began = Duration::from_secs(7);
		let step = Rate::DEFAULT.step();
		let after = runtime.step(&mut input, Rate::DEFAULT, 0.0, false, began);

		assert_eq!(after, began + step, "one step on, whatever the clock says");
		assert_eq!(
			runtime.step(&mut input, Rate::DEFAULT, 0.0, false, after),
			began + step * 2,
			"and again"
		);
	}

	#[test]
	fn the_wires_clock_moves_by_the_step_it_was_told_about() {
		// and not by the one this file was compiled with. Two peers on the
		// same wire place a world against a moment each, and a moment that
		// advanced by a constant while the world advanced by something else
		// would be a delay nobody could measure.
		let mut runtime = Runtime::empty();
		let fast = Rate::from_hz(240);
		let began = Duration::from_secs(1);

		assert_eq!(
			runtime.step(&mut Input::default(), fast, 0.0, false, began),
			began + fast.step(),
			"a quarter of the usual step moves the wire a quarter as far"
		);
	}

	#[test]
	fn the_moment_a_step_is_given_is_the_one_it_was_handed() {
		// and not one this process read for itself. A process with no socket
		// gets no wire at all, which is the ordinary case.
		let asked = Duration::from_secs(3);

		assert!(wired(None, asked).is_none(), "no socket, no wire");

		let mut net = Net::over(
			Box::new(Loopback::at(
				SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 1),
				&Rc::new(RefCell::new(Wire::default())),
			)),
			false,
			1,
			1,
		);

		assert_eq!(
			wired(Some(&mut net), asked).map(|it| it.now),
			Some(asked),
			"the moment is the caller's, not a fresh reading"
		);
	}
}
