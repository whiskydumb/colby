//! The window, and the loop that drives everything else.
//!
//! `App` is the one place the whole process's state sits: the world, the clock,
//! the loaded game, the renderer and the file watcher. Nothing here lives
//! inside the hot-reloaded module, which is why swapping the module is allowed
//! to be a two-line operation in the middle of a frame.

use std::{
	sync::Arc,
	time::{Duration, Instant},
};

use colby_audio::Device;
use colby_core::{
	Error, Result,
	abi::{Input, Mix, World, cvar::Cvars},
	debug, err, error,
	glam::{Vec2, Vec3},
	info,
	time::{Clock, Pace, Rate},
	warn,
};
#[cfg(feature = "editor")]
use colby_editor::Editor;
use colby_engine::{
	Overlay, Renderer,
	winit::{
		application::ApplicationHandler,
		dpi::LogicalSize,
		event::{ElementState, KeyEvent, WindowEvent},
		event_loop::ActiveEventLoop,
		keyboard::{Key, NamedKey},
		window::{Window, WindowId},
	},
};
use colby_net::Slot;
use colby_physics::Simulation;
use colby_script::Vm;
use colby_ui::Interface;

#[cfg(feature = "hot_reload")]
use crate::watch::Watch;
use crate::{
	assets::Assets,
	console::Console,
	game::Game,
	input,
	mode::Mode,
	net::{Net, Standing},
	step,
};

/// The window title.
const TITLE: &str = "colby";

/// The window's initial size, in logical pixels.
const SIZE: LogicalSize<f64> = LogicalSize::new(1280.0, 720.0);

/// The variable that holds the simulation still.
pub(crate) const PAUSE: &str = "sim.pause";

/// The variable that scales simulated time against real time.
pub(crate) const SPEED: &str = "sim.speed";

/// The variable that says how many simulation steps there are in a second.
pub(crate) const RATE: &str = "sim.rate";

/// The variable that says how hard everything falls.
pub(crate) const GRAVITY: &str = "phys.gravity";

/// How hard everything falls unless somebody says otherwise.
pub(crate) const PULL: f32 = 9.81;

/// What the console says the simulation rate is.
///
/// **Clamped on the way out rather than refused on the way in.** A config file
/// holding a rate this build will not run at should start the engine at the
/// nearest rate it will, rather than stop it: the variable is an integer and a
/// console takes what it is typed, so every number a person can type has to
/// mean something. @ref [`Rate::from_hz`], which is where the clamp lives, so
/// there is one of it rather than two.
///
/// @param cvars - the table [`RATE`] was registered into
/// @return the rate to run at; [`Rate::DEFAULT`] when nothing was said
pub(crate) fn rate(cvars: &Cvars) -> Rate {
	let default = i64::from(Rate::DEFAULT.hz());
	let asked = cvars
		.int(RATE)
		.unwrap_or(default)
		.clamp(i64::from(Rate::MIN_HZ), i64::from(Rate::MAX_HZ));

	Rate::from_hz(u16::try_from(asked).unwrap_or(Rate::DEFAULT.hz()))
}

/// Hands the clock whatever the console asked for on top of real time.
///
/// Taken rather than read: an owed step is owed once. Clamped because the field
/// is a public one, and a game that writes four billion into it should cost a
/// wrong number of steps rather than a loop that never comes back.
///
/// **In steps of whatever a step currently is**, which is the whole reason
/// this is a function anybody can call: `sim.step 8` has to be eight steps at
/// any rate, and a clock told to owe eight sixtieths of a second would run
/// sixteen of them at a hundred and twenty.
///
/// @param world - the world carrying `owed_steps`
/// @param clock - the clock to owe them to
fn pay(world: &mut World, clock: &mut Clock) {
	let owed = std::mem::take(&mut world.owed_steps)
		.min(u32::try_from(crate::console::MAX_STEP).unwrap_or(u32::MAX));

	if owed > 0 {
		clock.owe(clock.rate().step() * owed);
	}
}

/// The rate an endpoint actually runs at.
///
/// **The rate is the authority's, and a client is not one.** A client replays
/// its own unacknowledged moves at its own `World::dt` and the host applied
/// them at the host's, so two ends on different rates integrate the same
/// commands differently and the difference reads as a player being corrected
/// forever. Nothing on the wire says what rate the far end is running, so the
/// only answer that costs nothing is for every client to run the one rate
/// every host runs - which is what a host running anything else is then
/// choosing, knowingly.
///
/// @param cvars - the table [`RATE`] was registered into
/// @param client - whether this end takes its world from somebody else
/// @return the rate to put on the clock
pub(crate) fn paced(cvars: &Cvars, client: bool) -> Rate {
	if client { Rate::DEFAULT } else { rate(cvars) }
}

/// Every piece of state in the process.
pub(crate) struct App {
	/// Boxed: the entity table alone is tens of kilobytes, and this is handed
	/// across the module boundary by pointer anyway.
	world: Box<World>,
	clock: Clock,
	/// Frames drawn. Not the same number as `world.steps`, which is the point
	/// of the whole arrangement.
	frames: u64,
	input: Input,
	game: Option<Game>,
	renderer: Option<Renderer>,
	/// The game's own interface. Always present, and only able to draw once a
	/// device exists to attach it to.
	interface: Interface,
	/// The interface's own logic, in Lua. An `Option` for the same reason the
	/// renderer is one: an engine whose scene and interface work and whose
	/// scripts did not start is still worth looking at, and the log says which
	/// half is missing.
	scripts: Option<Vm>,
	/// The output device and the mixer feeding it. An `Option` for the same
	/// reason the scripts are: an engine whose picture works and whose sound
	/// did not start is worth running, and the log says which half is missing.
	audio: Option<Device>,
	/// The socket and the conversation over it, if this window was told to
	/// connect to a host or to be one.
	/// An `Option` for the third time and the third same reason: an engine
	/// whose window works and whose socket did not bind is worth running.
	net: Option<Net>,

	/// The world as a snapshot describes it, taken down once a step rather
	/// than allocated per snapshot. Empty in a window that serves nobody.
	records: Vec<Slot>,
	/// When this process started, which is the clock the wire is on.
	///
	/// Real time rather than simulated: a round-trip estimate measured in
	/// simulated seconds would shrink whenever the machine stalled, and the
	/// number is about the wire rather than about the world.
	started: Instant,
	/// The physics. Boxed because the world holds its address: the table of
	/// queries installed into `world` points here, and a value that moved
	/// would leave that pointer behind. @ref `colby_physics`.
	simulation: Box<Simulation>,
	assets: Assets,
	/// Whether the world is being played or edited, and the world play started
	/// from. @ref `crate::mode`.
	mode: Mode,
	console: Option<Console>,
	#[cfg(feature = "editor")]
	editor: Option<Editor>,
	#[cfg(feature = "hot_reload")]
	watch: Option<Watch>,
	/// The volumes the console last asked for. @ref
	/// [`gravity`](Self::gravity), which is the same arrangement and the same
	/// reason: `World::mix` is the game's field, so writing it every frame
	/// would argue with a game that ducked the music during a cutscene.
	mix: Mix,
	/// The gravity the console last asked for.
	///
	/// `World::gravity` is the *game's* field, so the console must not write it
	/// every frame or a game that points gravity sideways would be argued with
	/// sixty times a second. It is written only when the variable moves, which
	/// is the same rule the material and console tables follow: whoever said
	/// something last wins, and nobody says anything by standing still.
	gravity: f32,
	failure: Option<Error>,
}

impl App {
	/// An application that has not opened its window yet.
	pub(crate) fn new() -> Self {
		let simulation = Box::new(Simulation::new());
		let mut world = Box::<World>::default();

		// once, here, and never again. The pointers in the table address this
		// executable rather than the game module, so no reload disturbs them -
		// which is the whole difference between this and a console command, and
		// is why one has to be forgotten on unload and the other does not.
		world.install_physics(simulation.table());

		Self {
			world,
			clock: Clock::new(),
			frames: 0,
			input: Input::default(),
			game: None,
			renderer: None,
			interface: Interface::new(),
			scripts: None,
			audio: None,
			net: None,
			records: Vec::new(),
			started: Instant::now(),
			simulation,
			assets: Assets::new(&crate::workspace()),
			mode: Mode::new(),
			console: None,
			#[cfg(feature = "editor")]
			editor: None,
			#[cfg(feature = "hot_reload")]
			watch: None,
			mix: Mix::FULL,
			gravity: -PULL,
			failure: None,
		}
	}

	/// The reason the loop stopped, if it was not asked to.
	pub(crate) fn into_result(self) -> Result { self.failure.map_or(Ok(()), Err) }

	/// Opens the window, brings up the renderer and loads the game.
	fn start(&mut self, event_loop: &ActiveEventLoop) -> Result {
		let attributes = Window::default_attributes()
			.with_title(TITLE)
			.with_inner_size(SIZE);

		let window = event_loop
			.create_window(attributes)
			.map_err(|error| err!(Graphics("creating the window: {error}")))?;

		let renderer = Renderer::new(Arc::new(window))?;
		let (width, height) = renderer.size();

		self.world.aspect = renderer.aspect();
		self.input
			.set_viewport(f64::from(width), f64::from(height));

		if let Err(error) = self
			.interface
			.attach(renderer.device(), renderer.format())
		{
			// a log line rather than a stop: an engine whose scene draws and
			// whose interface does not is still worth looking at, and the
			// message says which half is missing.
			error!(%error, "the interface has no pipeline; nothing it draws will be on screen");
		}

		self.renderer = Some(renderer);

		// before the game, not after: `init` is where a game resolves the
		// meshes it wants by name, and a registry that fills up one frame later
		// would hand it nothing on the first one.
		self.assets.sync(&mut self.world);

		match Vm::new(crate::console::publisher()) {
			| Ok(scripts) => self.scripts = Some(scripts),
			| Err(error) =>
				error!(%error, "no interpreter; documents with a script have no logic"),
		}

		// after the assets rather than before them, so the first copy into the
		// mixer's bank finds a registry that is already full.
		match Device::open() {
			| Ok(mut device) => {
				device.load(&self.world.sounds);
				self.audio = Some(device);
			},
			| Err(error) => error!(%error, "no output device; nothing will make a sound"),
		}

		// and the host's console variables before *that*, so they belong to the
		// engine rather than to the module and survive a reload.
		crate::console::install(&mut self.world);

		// after them, because how bad the wire is is a console variable and the
		// seed for it is one too. A window that was told neither to connect
		// nor to serve simply has no socket, which is the ordinary case.
		//
		// **A window can be either end of a wire**, and which one it is has to
		// be decided here rather than later: a window that serves stays the
		// authority its world already thinks it is, and one that connects
		// stops being it before the game module ever reads the field.
		let seed = crate::net::seed(&self.world.cvars);
		let opened = match crate::net::standing() {
			| Standing::Serving(port) => Some((Net::host(port, seed), true)),
			| Standing::Talking(address) => Some((Net::connect(address, seed), false)),
			| Standing::Alone => None,
		};

		match opened {
			| None => {},
			| Some((Err(error), _)) => error!(%error, "no socket; this window is on its own"),
			| Some((Ok(net), true)) => self.net = Some(net),
			| Some((Ok(net), false)) => {
				self.net = Some(net);
				// and this process stops being the authority, which is the one
				// thing about a window that connected that nothing else could
				// work out. @ref `crate::net::joined`.
				crate::net::joined(&mut self.world);
			},
		}

		self.start_editor();

		self.game = Some(Game::open(&mut self.world)?);

		// the config is read last of the three, because a line in it may name a
		// variable the game registered a moment ago.
		self.console = Some(Console::open(&mut self.world));

		// after the config, because that is where a rate somebody asked for
		// arrives, and before the first step, because that is the last moment
		// saying anything about it is useful. @ref `set_pace` for why a client
		// does not get its own rate.
		let following = self
			.net
			.as_ref()
			.is_some_and(|net| !net.hosting());

		if following && rate(&self.world.cvars) != Rate::DEFAULT {
			warn!(
				asked = rate(&self.world.cvars).hz(),
				running = Rate::DEFAULT.hz(),
				"a window that connected runs at the host's rate, not its own"
			);
		}

		self.start_watching()?;

		// everything above took as long as it took: a window, an adapter, a
		// shader, the asset tree and a LoadLibrary. The clock has been running
		// since `new`, and without this the first frame would arrive owing the
		// simulation a second of catch-up it never actually missed.
		self.clock.reset();

		info!("colby is running; escape or the window close button stops it");

		Ok(())
	}

	/// Brings up the editor against the window and the device.
	///
	/// Its variables are registered here rather than in `Editor::new` so that
	/// they exist whether or not the editor was built.
	#[cfg(feature = "editor")]
	fn start_editor(&mut self) {
		Editor::install(&mut self.world);

		let Some(renderer) = self.renderer.as_ref() else {
			return;
		};

		self.editor = Some(Editor::new(renderer.window(), renderer.device(), renderer.format()));
	}

	/// Does nothing; this build has no editor.
	#[cfg(not(feature = "editor"))]
	#[expect(clippy::unused_self, reason = "as start_watching")]
	fn start_editor(&self) {}

	/// Builds the editor's interface for this frame, if it is on screen.
	///
	/// After the simulation and before the draw, so that what it shows is the
	/// state that is about to be drawn rather than the one before it.
	#[cfg(feature = "editor")]
	fn run_editor(&mut self) {
		if !Editor::shown(&self.world) {
			return;
		}

		// the window is behind an `Arc` for exactly this: the renderer holds
		// it, and the editor needs it while the world is borrowed.
		let Some(window) = self
			.renderer
			.as_ref()
			.map(|renderer| Arc::clone(renderer.window()))
		else {
			return;
		};

		if let Some(editor) = self.editor.as_mut() {
			editor.run(&window, &mut self.world, &self.clock, self.frames);
		}
	}

	/// Does nothing; this build has no editor.
	#[cfg(not(feature = "editor"))]
	#[expect(clippy::unused_self, reason = "as start_watching")]
	fn run_editor(&self) {}

	/// Offers an event to the editor before the game sees it.
	///
	/// @return whether the editor took it. A key typed into the console is not
	/// a key held to walk with, and escape closing a text field is not escape
	/// closing the window.
	#[cfg(feature = "editor")]
	fn editor_took(&mut self, event: &WindowEvent) -> bool {
		if !Editor::shown(&self.world) {
			return false;
		}

		let Some(window) = self
			.renderer
			.as_ref()
			.map(|renderer| Arc::clone(renderer.window()))
		else {
			return false;
		};

		self.editor
			.as_mut()
			.is_some_and(|editor| editor.on_event(&window, event))
	}

	/// Nothing takes it; this build has no editor.
	#[cfg(not(feature = "editor"))]
	#[expect(
		clippy::unused_self,
		clippy::needless_pass_by_ref_mut,
		reason = "as start_watching"
	)]
	fn editor_took(&mut self, _event: &WindowEvent) -> bool { false }

	/// Starts watching the game crate for changes.
	#[cfg(feature = "hot_reload")]
	fn start_watching(&mut self) -> Result {
		let sources = crate::workspace().join("src").join("game");
		self.watch = Some(Watch::new(crate::game::MODULE_NAME, sources)?);

		Ok(())
	}

	/// Does nothing; there is no module to watch.
	#[cfg(not(feature = "hot_reload"))]
	#[expect(
		clippy::unused_self,
		clippy::needless_pass_by_ref_mut,
		reason = "the hot-reload variant of this function needs both, and the two have to 		          agree on a signature"
	)]
	fn start_watching(&mut self) -> Result { Ok(()) }

	/// Runs whatever simulation this frame owes, then draws between the last
	/// two states of it.
	///
	/// Zero steps at a high refresh rate, one most of the time, several after a
	/// hitch - and a picture either way, because what makes the motion smooth
	/// is where the frame sits between two steps rather than how many of them
	/// it ran.
	/// The moment the next step is at, on the clock the wire is on.
	///
	/// Read once a frame and advanced a step at a time by the caller, rather
	/// than read again inside the step loop: what the wire is asked for has to
	/// move by exactly one step per step, or the renderer's own blend between
	/// two steps is asked to cover a gap that is not one step wide - and a
	/// body would be drawn speeding up and slowing down on a wire doing
	/// nothing of the kind. @ref `step::Wired`.
	///
	/// @return how long this process has been running
	fn moment(&self) -> Duration { self.started.elapsed() }

	/// The endpoint and the moment, for the step about to run.
	///
	/// Its own function so that *which* clock the moment comes from is
	/// something a test can ask about. It comes from the caller, which read it
	/// once for the frame; reading it here would put a real-time sample inside
	/// the step loop, and two steps in one frame would then be microseconds
	/// apart where the renderer's blend assumes a step. @ref `step::Wired`.
	///
	/// @param net - the endpoint, if this process has one
	/// @param moment - where the wire's clock stands for this step
	/// @return the pair, or nothing when this process is on no wire
	fn wired(net: Option<&mut Net>, moment: Duration) -> Option<step::Wired<'_>> {
		let net = net?;

		Some(step::Wired { net, now: moment })
	}

	/// Puts a message out for the step that has just run, and moves the wire's
	/// own clock on by one step.
	///
	/// Out once a step rather than once a frame, because a message has to mean
	/// "this is where things stand at this moment" and a frame rate is not a
	/// moment.
	///
	/// @param moment - where the wire's clock stood for the step just run
	/// @param rate - how long that step was
	/// @return where it stands for the next one
	fn stepped(&mut self, moment: Duration, rate: Rate) -> Duration {
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

	fn frame(&mut self) -> Result {
		// reading two variables is not the kind of work that moves the clock
		// sample around, and the pace has to be set before the time it applies
		// to is measured.
		self.set_pace();

		// then the sample, ahead of anything that touches a filesystem. Total
		// time is never lost whenever this is taken, but the *phase* of it is
		// what the drawn pose is a function of, so a sample that wanders by
		// however long a directory scan took is a picture that wanders with it.
		let pace = self.clock.tick();
		self.report(pace);

		self.reload_if_stale();

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

		// and whatever a command asked to be done with a scene, immediately
		// after it was typed and before any step runs: a load replaces every
		// table in the world, and doing that halfway through one would leave
		// the rest of the step running against a world its first half never
		// saw. @ref `crate::saves`.
		crate::saves::serve(&mut self.world, &mut self.simulation);

		// and whatever a command asked of the wire, beside it and for the same
		// reason. Then everything the socket is holding, before any step: what
		// a step runs against is what had arrived when it started rather than
		// whatever turned up halfway through. @ref `crate::net`.
		crate::net::serve(self.net.as_mut());

		if let Some(net) = self.net.as_mut() {
			// which is where this window learns who it is, where what the host
			// has already run stops being resent, and where a line somebody
			// else typed is run. @ref `crate::net::hear`, which a windowless
			// end calls too - and which is one function because the last of
			// those four was missing from this copy for as long as there were
			// two.
			crate::net::hear(net, &mut self.world, self.started.elapsed());
		}

		// and the mode's own edge, in the same place and for the same reason:
		// stopping play replaces every table in the world, so it happens
		// between steps rather than inside one.
		let editing = crate::mode::wanted(&self.world);
		self.mode
			.follow(&mut self.world, &mut self.simulation, editing);

		pay(&mut self.world, &mut self.clock);

		let asked = self.world.cvars.float(GRAVITY).unwrap_or(-PULL);
		if (asked - self.gravity).abs() > f32::EPSILON {
			self.gravity = asked;
			self.world.gravity = Vec3::new(0.0, asked, 0.0);
		}

		self.hear();

		if let Some(renderer) = self.renderer.as_ref() {
			// before the steps rather than after them: gameplay asking how
			// wide the window is should not be told last frame's answer, four
			// times in a row.
			self.world.aspect = renderer.aspect();

			let (width, height) = renderer.size();
			#[expect(
				clippy::as_conversions,
				clippy::cast_possible_truncation,
				reason = "a display scale is between one and four, and the only f32 it is ever 				          multiplied by is a pixel count"
			)]
			let scale = renderer.window().scale_factor() as f32;

			self.world.ui.set_viewport(
				Vec2::new(
					f32::from(u16::try_from(width).unwrap_or(u16::MAX)),
					f32::from(u16::try_from(height).unwrap_or(u16::MAX)),
				),
				scale,
			);
			self.world
				.ui
				.set_pointer(Vec2::from(self.input.cursor));
		}

		let mut moment = self.moment();
		// read once for the whole pass rather than per step: a rate that moved
		// between two steps of the same frame would make the second one a
		// different length from the first, which is the one thing a fixed step
		// is for not being.
		let rate = self.clock.rate();

		while let Some(time) = self.clock.step() {
			step::run(
				&mut self.world,
				step::Parts {
					game: self.game.as_mut(),
					interface: &mut self.interface,
					scripts: self.scripts.as_mut(),
					simulation: self.simulation.as_mut(),
					audio: self.audio.as_mut(),
					// the endpoint and the clock it is on, so that a world a
					// host described lands in this one. @ref `Net::arrive`.
					wire: Self::wired(self.net.as_mut(), moment),
				},
				&mut self.input,
				rate,
				time,
				editing,
			);

			moment = self.stepped(moment, rate);
		}

		self.frames = self.frames.saturating_add(1);
		// the mode has a say in this: a world being edited is drawn as it
		// stands rather than blended. @ref `Mode::interpolation`.
		self.world.set_interpolation(
			self.mode
				.interpolation(self.clock.interpolation()),
		);

		// the interface is laid out again here rather than reused from the step:
		// the window may have been resized since, and a document that is a share
		// of the screen should be the share the screen is now.
		self.interface.run(&self.world);

		if let Some(renderer) = self.renderer.as_ref() {
			self.interface
				.prepare(renderer.device(), renderer.queue(), &self.world);
		}

		self.run_editor();

		#[cfg(feature = "editor")]
		let shown = Editor::shown(&self.world);

		// disjoint borrows: the renderer is one field, the interface another and
		// the editor a third, which is what lets the frame be handed to all of
		// them.
		let mut overlays: Vec<&mut dyn Overlay> = vec![&mut self.interface];

		#[cfg(feature = "editor")]
		if shown && let Some(editor) = self.editor.as_mut() {
			overlays.push(editor);
		}

		let Some(renderer) = self.renderer.as_mut() else {
			return Ok(());
		};

		renderer.render(&self.world, &mut overlays)
	}

	/// Puts the console's pacing variables onto the clock.
	///
	/// `sim.pause` is not a separate mechanism from `sim.speed`, it is a speed
	/// of zero - which is what makes unpausing free of a lurch: no time
	/// accumulated while it was held, so there is nothing owed on the way out.
	fn set_pace(&mut self) {
		let paused = self.world.cvars.bool(PAUSE).unwrap_or(false);
		let speed = self.world.cvars.float(SPEED).unwrap_or(1.0);

		// a window that serves is the authority and runs what it was told; one
		// that connected takes the rate every host runs. @ref `paced`.
		let following = self
			.net
			.as_ref()
			.is_some_and(|net| !net.hosting());

		self.clock
			.set_rate(paced(&self.world.cvars, following));
		self.clock
			.set_speed(if paused { 0.0 } else { speed });
	}

	/// Writes the volumes the console asked for, if any of them moved.
	///
	/// The same shape gravity has, and the same rule: whoever said something
	/// last wins, and standing still says nothing. Four numbers rather than
	/// one, so the comparison is over the whole struct.
	fn hear(&mut self) {
		let asked = crate::console::volumes(&self.world.cvars);

		if asked != self.mix {
			self.mix = asked;
			self.world.mix = asked;
		}
	}

	/// Says something the first time the simulation falls behind, and the first
	/// time it catches up.
	///
	/// Nothing in between, deliberately: a machine slow enough to be
	/// permanently behind would otherwise spend what it has left writing a
	/// line about it every frame.
	fn report(&self, pace: Pace) {
		match pace {
			| Pace::FellBehind => warn!(
				stalls = self.clock.stalls(),
				"the simulation is behind real time; catch-up is capped and the excess is being \
				 dropped"
			),
			| Pace::CaughtUp => info!("the simulation is keeping up again"),
			| Pace::Keeping | Pace::Behind => {},
		}
	}

	/// Swaps the game module if a newer build has appeared.
	///
	/// A failed swap is a log line, not a stop: the next successful build gets
	/// another go, which is the entire point of editing code in a running
	/// process.
	#[cfg(feature = "hot_reload")]
	fn reload_if_stale(&mut self) {
		if !self.watch.as_mut().is_some_and(Watch::poll) {
			return;
		}

		if let Some(game) = self.game.as_mut()
			&& let Err(error) = game.reload(&mut self.world)
		{
			error!(%error, "reload failed; the game is parked until the next build");
		}

		// whether it worked or not, it took time: an unload, a copy, a
		// LoadLibrary and the game's `init`. None of that is time the
		// simulation owes, and billing it as arrears would make every edit
		// jump the scene forward by a tenth of a second.
		self.clock.reset();
	}

	/// Does nothing; the game is linked in.
	#[cfg(not(feature = "hot_reload"))]
	#[expect(
		clippy::unused_self,
		clippy::needless_pass_by_ref_mut,
		reason = "as start_watching"
	)]
	fn reload_if_stale(&mut self) {}

	/// Reacts to a named key going down, once the editor has had its chance.
	///
	/// @param event_loop - the loop to stop, if that is what was pressed
	/// @param key - the key
	#[cfg_attr(
		not(feature = "editor"),
		expect(
			clippy::unused_self,
			clippy::needless_pass_by_ref_mut,
			reason = "with the editor built in, this arm toggles it and needs the world; \
			          without 			          it there is only escape, and the two have to agree \
			          on a signature"
		)
	)]
	fn pressed(&mut self, event_loop: &ActiveEventLoop, key: NamedKey) {
		match key {
			| NamedKey::Escape => event_loop.exit(),
			#[cfg(feature = "editor")]
			| NamedKey::F1 => Editor::toggle(&mut self.world),
			// under the same feature as F1, and for the same reason: play and
			// stop is a tool's gesture. A build with no editor in it still has
			// the variable, and nothing in it presses this.
			#[cfg(feature = "editor")]
			| NamedKey::F5 => crate::mode::toggle(&mut self.world),
			| _ => {},
		}
	}

	/// Records a failure and asks the loop to stop.
	fn fail(&mut self, event_loop: &ActiveEventLoop, error: Error) {
		error!(%error, "stopping");
		self.failure = Some(error);
		event_loop.exit();
	}
}

impl ApplicationHandler for App {
	fn resumed(&mut self, event_loop: &ActiveEventLoop) {
		if self.renderer.is_some() {
			return;
		}

		if let Err(error) = self.start(event_loop) {
			self.fail(event_loop, error);
		}
	}

	fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
		// the editor first: what it takes, the game must not also act on.
		let taken = self.editor_took(&event);

		if !taken {
			input::apply(&mut self.input, &event);
		}

		match event {
			| WindowEvent::CloseRequested => event_loop.exit(),
			| WindowEvent::KeyboardInput {
				event:
					KeyEvent {
						logical_key: Key::Named(key),
						state: ElementState::Pressed,
						..
					},
				..
			} if !taken => self.pressed(event_loop, key),
			| WindowEvent::Resized(size) =>
				if let Some(renderer) = self.renderer.as_mut() {
					renderer.resize(size.width, size.height);
				},
			| WindowEvent::RedrawRequested => {
				if let Err(error) = self.frame() {
					self.fail(event_loop, error);
				} else if self.world.quit {
					// asked for by the `quit` command, or by the game itself.
					// The same way out as the close button, so `exiting` runs
					// and the config is written.
					info!("stopping, as asked");
					event_loop.exit();
				}
			},
			| _ => {},
		}
	}

	fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
		if let Some(renderer) = self.renderer.as_ref() {
			renderer.window().request_redraw();
		}
	}

	fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
		// before the game is shut down, while its variables are still in the
		// table to be written out.
		if let Some(console) = self.console.as_ref() {
			console.close(&self.world);
		}

		if let Some(game) = self.game.as_mut() {
			game.close(&mut self.world);
		}

		// order matters: the game goes first because it may still be running
		// code from the module, then the renderer, whose surface borrows the
		// window it is holding the last share of.
		self.game = None;
		self.renderer = None;

		info!(
			frames = self.frames,
			steps = self.world.steps,
			stalls = self.clock.stalls(),
			reloads = self.world.reloads,
			"colby stopped"
		);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// How many steps a clock has ready.
	fn drain(clock: &mut Clock) -> u32 {
		let mut ran = 0;
		while clock.step().is_some() {
			ran += 1;
		}

		ran
	}

	#[test]
	fn a_step_the_console_asked_for_is_a_step_at_whatever_the_rate_is() {
		for hz in [30_u16, 60, 120, 240] {
			let mut world = World::new();
			let mut clock = Clock::new();
			clock.set_rate(Rate::from_hz(hz));
			world.owed_steps = 8;

			pay(&mut world, &mut clock);

			assert_eq!(
				drain(&mut clock),
				8,
				"eight steps at {hz} a second, not eight of somebody else's"
			);
			assert_eq!(world.owed_steps, 0, "and asked for once");
		}
	}

	#[test]
	fn a_console_cannot_ask_for_more_steps_than_the_ceiling() {
		let mut world = World::new();
		let mut clock = Clock::new();
		world.owed_steps = u32::MAX;

		pay(&mut world, &mut clock);

		assert_eq!(
			u64::from(drain(&mut clock)),
			u64::try_from(crate::console::MAX_STEP)
				.expect("the ceiling is a small positive number"),
			"four billion steps is a loop that does not come back"
		);
	}

	#[test]
	fn an_end_that_takes_its_world_from_somebody_runs_the_rate_every_host_runs() {
		// the rate is the authority's. A client that ran its own would replay
		// its unacknowledged moves over a different step from the one the host
		// applied them over, and that is a correction on every snapshot rather
		// than an error anybody could see in a log.
		let mut world = World::new();
		crate::console::install(&mut world);
		crate::console::run(&mut world, &format!("{RATE} 240"));

		assert_eq!(rate(&world.cvars).hz(), 240, "the console was heard");
		assert_eq!(
			paced(&world.cvars, false).hz(),
			240,
			"an end that owns its world runs what it was told"
		);
		assert_eq!(
			paced(&world.cvars, true).hz(),
			Rate::DEFAULT.hz(),
			"and one that does not, does not"
		);
	}

	#[test]
	fn the_wires_clock_moves_by_exactly_one_step_a_step() {
		// what the renderer's own blend between two steps assumes, and the
		// only thing that says the moment is not re-read from the real clock
		// inside the loop: two steps in one frame would then be microseconds
		// apart, and two a frame apart a whole frame apart, with one blend
		// asked to cover both.
		let mut app = App::new();
		let began = Duration::from_secs(7);
		let step = Rate::DEFAULT.step();
		let after = app.stepped(began, Rate::DEFAULT);

		assert_eq!(after, began + step, "one step on, whatever the clock says");
		assert_eq!(app.stepped(after, Rate::DEFAULT), began + step * 2, "and again");
	}

	#[test]
	fn the_wires_clock_moves_by_the_step_it_was_told_about() {
		// and not by the one this file was compiled with. Two peers on the
		// same wire place a world against a moment each, and a moment that
		// advanced by a constant while the world advanced by something else
		// would be a delay nobody could measure.
		let mut app = App::new();
		let fast = Rate::from_hz(240);
		let began = Duration::from_secs(1);

		assert_eq!(
			app.stepped(began, fast),
			began + fast.step(),
			"a quarter of the usual step moves the wire a quarter as far"
		);
	}

	#[test]
	fn the_moment_a_step_is_given_is_the_one_it_was_handed() {
		// and not one this process read for itself. A window with no socket
		// gets no wire at all, which is the ordinary case.
		let mut app = App::new();
		let asked = Duration::from_secs(3);

		assert!(App::wired(app.net.as_mut(), asked).is_none(), "no socket, no wire");

		app.net = Some(Net::over(
			Box::new(crate::net::Loopback::at(
				std::net::SocketAddr::new(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST), 1),
				&std::rc::Rc::new(std::cell::RefCell::new(crate::net::Wire::default())),
			)),
			false,
			1,
			1,
		));

		assert_eq!(
			App::wired(app.net.as_mut(), asked).map(|it| it.now),
			Some(asked),
			"the moment is the caller's, not a fresh reading"
		);
	}
}
