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
	thread,
	time::{Duration, Instant},
};

use colby_asset::Project;
use colby_audio::Device;
use colby_core::{
	Err, Result,
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
	launch::Asked,
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

	/// How long the last step's particles took.
	///
	/// Kept here rather than returned, because what reads it is the profiler
	/// one layer up and it wants the number a whole frame later. The physics
	/// keeps its own inside `Simulation`; the particles have no state to keep
	/// theirs in, so it lives beside the thing that drives them. @ref
	/// `crate::sparks`.
	pub(crate) sparked: Duration,

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
	pub(crate) project: Project,
}

impl Runtime {
	/// Brings a world up, the one way there is.
	///
	/// An [`Opening`] run to the end with nothing between its stages: what a
	/// picture, a sound and a windowless end do, and what a window does one
	/// stage a frame behind a screen that says which. @ref [`Stage`] for the
	/// order and the reasons.
	///
	/// @param front - what kind of process this is
	/// @param project - the project: where the assets, the saves, the config
	/// and the game crate are, and what the world starts as
	/// @param build - what the build script knew, which says where the engine
	/// is: the game crate is mounted there and built there
	/// @return the runtime, or the first thing that would not come up
	pub fn open(front: Front, project: &Project, build: &Build, asked: &Asked) -> Result<Self> {
		let mut opening = Opening::start(front, project, build, asked)?;

		loop {
			match opening.advance()? {
				| Progress::Moved(stage) => debug!(stage = stage.title(), "up"),
				| Progress::Waiting(_) => thread::sleep(BUILD_POLL),
				| Progress::Done => return opening.finish(),
			}
		}
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
		crate::code::serve(&mut self.world, &self.project);
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
				sparked: &mut self.sparked,
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

/// How long a windowless front waits between two looks at a build that is
/// still running.
const BUILD_POLL: Duration = Duration::from_millis(50);

/// The stages a world comes up in, in the order they run.
///
/// The order is the window's, which was the most demanding of the four
/// fronts: the assets before the module, so `init` finds its meshes by name;
/// the host's variables before the module, so they are the engine's and
/// survive a reload; the socket before the module, so a client has said what
/// it is before `init` reads `World::peer`; the editor's variables before the
/// module for the reason the host's are; the config after the module, because
/// a line in it may name a variable the game registered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
	/// The asset tree, compiled and loaded.
	Assets,

	/// The scene the project starts as, put in place.
	Scene,

	/// The host's variables, the output device, the editor's variables.
	Console,

	/// The socket, for a front that is on a wire.
	Wire,

	/// The game crate, mounted in the engine's workspace and built by cargo.
	Build,

	/// The module, loaded and initialized.
	Module,

	/// The settings archive, run against the table.
	Settings,

	/// The interpreter.
	Scripts,
}

impl Stage {
	/// Every stage, in the order they run.
	pub const ALL: [Self; 8] = [
		Self::Assets,
		Self::Scene,
		Self::Console,
		Self::Wire,
		Self::Build,
		Self::Module,
		Self::Settings,
		Self::Scripts,
	];

	/// What a screen says while the stage runs.
	#[must_use]
	pub const fn title(self) -> &'static str {
		match self {
			| Self::Assets => "compiling the assets",
			| Self::Scene => "the startup scene",
			| Self::Console => "the console and the output device",
			| Self::Wire => "the wire",
			| Self::Build => "building the game crate",
			| Self::Module => "loading the game module",
			| Self::Settings => "reading settings.cfg",
			| Self::Scripts => "starting the interpreter",
		}
	}

	/// The stage after this one, if there is one.
	const fn after(self) -> Option<Self> {
		match self {
			| Self::Assets => Some(Self::Scene),
			| Self::Scene => Some(Self::Console),
			| Self::Console => Some(Self::Wire),
			| Self::Wire => Some(Self::Build),
			| Self::Build => Some(Self::Module),
			| Self::Module => Some(Self::Settings),
			| Self::Settings => Some(Self::Scripts),
			| Self::Scripts => None,
		}
	}
}

/// What one call to [`Opening::advance`] came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
	/// This stage finished.
	Moved(Stage),

	/// This stage is still busy - a build cargo has not finished - and the
	/// next call will look again.
	Waiting(Stage),

	/// Every stage has run; @ref [`Opening::finish`].
	Done,
}

/// A world on its way up, one stage at a time.
///
/// A value rather than a function because a window wants to draw between the
/// stages: compiling an asset tree and building a game crate take seconds,
/// and a window that shows nothing for those seconds looks like a window that
/// died. So the stand-up is something a frame loop can hold and push one
/// stage a frame, drawing what the screen should say in between - and the
/// build, which is another process, is looked at rather than waited for, so
/// the window stays a window while cargo works. The fronts with no screen run
/// it to the end in one call, @ref [`Runtime::open`], which is what keeps this
/// the one order a world is brought up in.
pub struct Opening {
	front: Front,
	project: Project,
	/// The variables the command line asked for, applied at the last stage.
	asked: Asked,
	#[cfg(feature = "hot_reload")]
	build: Build,

	/// The stage the next call runs, or nothing once every stage has.
	next: Option<Stage>,

	world: Box<World>,
	simulation: Box<Simulation>,
	assets: Assets,
	audio: Option<Device>,
	net: Option<Net>,
	game: Option<Game>,
	console: Option<Console>,
	scripts: Option<Vm>,

	/// The module's name, or nothing for a project with no crate.
	module: Option<String>,

	/// The cargo building the module, while it runs.
	#[cfg(feature = "hot_reload")]
	building: Option<std::process::Child>,
}

impl Opening {
	/// The world and its solver, and nothing that takes time.
	///
	/// @param front - what kind of process this is
	/// @param project - the project everything on disk resolves against
	/// @param build - what the build script knew, for the build stage
	#[cfg_attr(
		not(feature = "hot_reload"),
		expect(
			unused_variables,
			reason = "with the game linked in there is no crate to build, and the two builds \
			          have to agree on a signature"
		)
	)]
	pub fn start(front: Front, project: &Project, build: &Build, asked: &Asked) -> Result<Self> {
		// boxed and installed before anything else touches the world: the
		// world keeps this address. Once, here, and never again - the pointers
		// in the table address this executable rather than the game module, so
		// no reload disturbs them, which is the whole difference between this
		// and a console command and is why one has to be forgotten on unload
		// and the other does not.
		let simulation = Box::new(Simulation::new());
		let mut world = Box::<World>::default();
		world.install_physics(simulation.table());

		if let Some(viewport) = front.viewport() {
			world.ui.set_viewport(viewport, 1.0);
			world.aspect = viewport.x / viewport.y;
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

		Ok(Self {
			front,
			project: project.clone(),
			asked: asked.clone(),
			#[cfg(feature = "hot_reload")]
			build: build.clone(),
			next: Some(Stage::Assets),
			world,
			simulation,
			assets: Assets::of(project),
			audio: None,
			net: None,
			game: None,
			console: None,
			scripts: None,
			module,
			#[cfg(feature = "hot_reload")]
			building: None,
		})
	}

	/// The stage the next call to [`advance`](Self::advance) runs, or nothing
	/// once every stage has run.
	#[must_use]
	pub const fn next(&self) -> Option<Stage> { self.next }

	/// The world as it stands, for a screen that draws it while it comes up.
	#[must_use]
	pub fn world(&self) -> &World { &self.world }

	/// The world as it stands, for a renderer that has to tell it how wide
	/// the window is.
	pub fn world_mut(&mut self) -> &mut World { &mut self.world }

	/// Runs the next stage, or looks again at one that is still busy.
	///
	/// @return what came of it: a stage finished, a stage still busy, or
	/// nothing left to run
	pub fn advance(&mut self) -> Result<Progress> {
		let Some(stage) = self.next else {
			return Ok(Progress::Done);
		};

		match stage {
			| Stage::Assets => self.assets.sync(&mut self.world),
			| Stage::Scene => start(&mut self.world, &mut self.simulation, &self.project)?,
			| Stage::Console => self.console_stage(),
			| Stage::Wire => self.net = connect(self.front, &mut self.world)?,
			| Stage::Build =>
				if !self.build_stage()? {
					return Ok(Progress::Waiting(stage));
				},
			| Stage::Module => self.game = Game::open(&mut self.world, self.module.as_deref())?,
			| Stage::Settings => self.settings_stage(),
			| Stage::Scripts => self.scripts = Some(Vm::new(console::defer)?),
		}

		self.next = stage.after();

		Ok(Progress::Moved(stage))
	}

	/// The runtime, once every stage has run.
	///
	/// # Errors
	///
	/// If a stage is still to run: the value is not a runtime yet.
	pub fn finish(self) -> Result<Runtime> {
		let Self {
			project,
			next,
			world,
			simulation,
			assets,
			audio,
			net,
			game,
			console,
			scripts,
			..
		} = self;

		if let Some(stage) = next {
			return Err!(Err("the world is not up yet; {} is still to run", stage.title()));
		}

		let Some(scripts) = scripts else {
			return Err!(Err("the world is not up yet; the interpreter is still to start"));
		};

		Ok(Runtime {
			world,
			simulation,
			assets,
			game,
			scripts,
			interface: Interface::new(),
			console,
			net,
			audio,
			sparked: Duration::ZERO,
			records: Vec::new(),
			started: Instant::now(),
			project,
		})
	}

	/// The output device, the host's variables and the editor's, and the
	/// window's own command.
	fn console_stage(&mut self) {
		// after the assets rather than before them, so the first copy into the
		// mixer's bank finds a registry that is already full.
		if self.front.is_window() {
			self.audio = listen(&self.world);
		}

		// **the table always, the terminal and the config file only where
		// there is one.** These are two different things and they used to be
		// one: a picture and a recording had no table at all, so every reader
		// in the engine carried its own copy of the default it would have
		// found there, and the only way to set one of them from outside was to
		// make a project whose game module claimed the name first. Registering
		// changes no answer - every fallback is the same constant the
		// registration uses, and a test says so - and it is what lets `--set`
		// mean something in a run with no console. @ref `PERF-4`.
		crate::console::install(&mut self.world);

		#[cfg(feature = "editor")]
		if self.front.is_window() {
			// registered here rather than in `Editor::new`, so that they exist
			// whether or not a window was ever made - and before the module,
			// so that they are the engine's.
			Editor::install(&mut self.world);
		}

		if self.front.is_window() {
			// the window's own: a picture of what it shows needs its device,
			// so the frame loop takes the line. Before the module, so that it
			// is the engine's. @ref `crate::screenshot`.
			self.world.cvars.command(
				crate::screenshot::COMMAND,
				console::defer,
				"write what the window shows to a png under screenshots/: a name, or the next \
				 number",
			);
		}
	}

	/// The settings archive, run against the table, and then the command line.
	///
	/// Last of the three that register, because a line in it may name a
	/// variable the game registered a moment ago - and the command line is
	/// after the file for the same reason one place further along: what
	/// somebody wrote on the line they started this run with beats what a file
	/// remembers and what a module asked for. @ref [`Asked`].
	///
	/// **The command line is applied whether or not there is a console**, and
	/// that is the whole of `PERF-4`: a picture, a recording and a measurement
	/// have no terminal and no config file, and they still have a table.
	fn settings_stage(&mut self) {
		if self.front.has_console() {
			self.console = Some(Console::open(&mut self.world, &self.project.settings()));
		}

		self.asked.apply(&mut self.world);
	}

	/// Mounts the project in the engine's workspace and builds its crate.
	///
	/// **Built at every open, not only when no image exists.** cargo is the
	/// one thing that knows whether the image on disk matches the engine this
	/// process runs and the sources as they stand now; a build that has
	/// nothing to do is a third of a second, and one that has something to do
	/// is the difference between a module that loads and a module built
	/// against yesterday's core that fails to. The build is another process,
	/// looked at rather than waited for, so a window can go on drawing.
	///
	/// @return whether the stage is finished: the build done, or nothing to
	/// build
	#[cfg(feature = "hot_reload")]
	fn build_stage(&mut self) -> Result<bool> {
		let Some(module) = self.module.as_deref() else {
			return Ok(true);
		};

		let Some(child) = self.building.as_mut() else {
			// dead mounts are swept first, because one of those fails every
			// cargo command. @ref `crate::mount` for why a module has to be
			// built in the engine's workspace and nowhere else.
			crate::mount::sweep(&self.build.engine);
			crate::mount::mount(&self.project, &self.build.engine)?;
			self.building = Some(crate::watch::start(&self.build)?);

			return Ok(false);
		};

		let Some(built) = crate::watch::finished(child)? else {
			return Ok(false);
		};

		self.building = None;

		if built {
			return Ok(true);
		}

		// a build that failed with an image on disk is a warning and the
		// image, because the image may well be the one that was running a
		// moment ago; with no image there is nothing to load and this is a
		// stop, with cargo's own words already on the terminal.
		if colby_core::mods::path::from_name(module)?.is_file() {
			colby_core::warn!(
				module,
				"building the game crate failed; loading the image that is there, which may be \
				 stale"
			);

			return Ok(true);
		}

		Err!(Module("building the game crate failed, and there is no image of it to load"))
	}

	/// Nothing to build: the game is linked in, whatever a project says.
	#[cfg(not(feature = "hot_reload"))]
	#[expect(
		clippy::unnecessary_wraps,
		clippy::unused_self,
		clippy::needless_pass_by_ref_mut,
		reason = "the hot-reload build of this stage can fail and works on what it holds, and \
		          the two have to agree on a signature"
	)]
	fn build_stage(&mut self) -> Result<bool> { Ok(true) }
}

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

	#[test]
	fn the_stages_run_in_the_order_the_stand_up_always_had() {
		// the assets before the module, the host's variables before the
		// module, the socket before the module, the config after it: what the
		// old one function did in one order, kept as a list a screen can show.
		let mut walked = vec![Stage::ALL[0]];

		while let Some(next) = walked.last().and_then(|stage| stage.after()) {
			walked.push(next);
		}

		assert_eq!(walked, Stage::ALL, "the chain and the list agree");
		assert_eq!(Stage::ALL.map(Stage::title), [
			"compiling the assets",
			"the startup scene",
			"the console and the output device",
			"the wire",
			"building the game crate",
			"loading the game module",
			"reading settings.cfg",
			"starting the interpreter",
		]);
	}

	#[test]
	fn an_opening_is_not_a_runtime_until_every_stage_has_run() {
		let scratch = std::env::temp_dir().join("colby_opening_early");
		let project = Project::parse(
			&scratch,
			r#"{ "schema": 1, "engine": "0.1.0", "id": "early", "name": "early" }"#,
		)
		.expect("a project");
		let build = Build {
			engine: scratch,
			cargo: "cargo".to_owned(),
			profile: "dev".to_owned(),
			rustflags: String::new(),
			package: "colby".to_owned(),
		};

		let opening =
			Opening::start(Front::Fixed, &project, &build, &Asked::default()).expect("started");

		assert_eq!(opening.next(), Some(Stage::Assets), "nothing has run");
		assert!(
			(opening.world().aspect - VIEWPORT.x / VIEWPORT.y).abs() < f32::EPSILON,
			"laid out like a picture"
		);

		let Err(error) = opening.finish() else {
			panic!("an opening with every stage still to run is not a runtime");
		};
		let text = error.to_string();

		assert!(text.contains("compiling the assets"), "it says which stage: {text}");
	}

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
				sparked: Duration::ZERO,
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
