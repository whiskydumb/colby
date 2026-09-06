//! The window, and the loop that drives everything else.
//!
//! Two halves, one after the other. First the window comes up over a world
//! that is still coming up - @ref [`Loader`], which runs the stand-up one
//! stage a frame behind a screen that says which stage, because compiling an
//! asset tree and building a game crate are seconds and a window that shows
//! nothing for seconds looks like a window that died. Then [`App`] takes the
//! window over: the runtime, the clock that paces it off the vertical blank,
//! the renderer, the editor and the file watcher. Everything a world needs to
//! run is in the runtime and is the same for a window as for a socket or a
//! picture - @ref [`Runtime`] - and nothing here lives inside the hot-reloaded
//! module, which is why swapping the module is allowed to be a two-line
//! operation in the middle of a frame.

use std::sync::Arc;

use colby_core::{
	Err, Error, Result,
	abi::{Input, Mix, World, cvar::Cvars},
	debug, err, error,
	glam::{Vec2, Vec3},
	info,
	time::{Clock, Pace, Rate},
	warn,
};
#[cfg(feature = "editor")]
use colby_editor::{Editor, Loading, State, Step};
use colby_engine::{
	Gpu, Overlay, Renderer, gpu,
	winit::{
		application::ApplicationHandler,
		dpi::LogicalSize,
		event::{ElementState, KeyEvent, WindowEvent},
		event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
		keyboard::{Key, NamedKey},
		window::{Window, WindowId},
	},
};

#[cfg(feature = "editor")]
use crate::runtime::Stage;
#[cfg(feature = "hot_reload")]
use crate::watch::Watch;
use crate::{
	Build, Front, Project, Runtime, input,
	mode::Mode,
	net::Standing,
	runtime::{Opening, Progress},
	screenshot,
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

/// Opens a window on a world and runs it until somebody closes it.
///
/// @param build - what the build script of the executable knew
/// @param standing - which end of a wire this window is, if either
/// @param project - the project to run
pub(crate) fn run(build: Build, standing: Standing, project: &Project) -> Result {
	let event_loop =
		EventLoop::new().map_err(|error| err!(Graphics("creating the event loop: {error}")))?;

	// @note: `Poll` rather than `Wait`. The frame is paced by the surface's
	// vsync, not by the event loop, and a game keeps simulating whether or not
	// the window has anything to say.
	event_loop.set_control_flow(ControlFlow::Poll);

	let mut boot = Boot::new(build, standing, project);

	event_loop
		.run_app(&mut boot)
		.map_err(|error| err!(Graphics("running the event loop: {error}")))?;

	boot.into_result()
}

/// Which half the window is in.
enum Phase {
	/// The world coming up behind its screen.
	Loading(Box<Loader>),

	/// The world running.
	Running(Box<App>),

	/// Neither: the moment between the two, and after either has stopped.
	Over,
}

/// The window's whole life: the loader, then the app, one handler over both.
struct Boot {
	phase: Phase,
	failure: Option<Error>,
}

impl Boot {
	/// A window waiting to be made, and a world waiting to come up behind it.
	fn new(build: Build, standing: Standing, project: &Project) -> Self {
		Self {
			phase: Phase::Loading(Box::new(Loader::new(build, standing, project))),
			failure: None,
		}
	}

	/// The reason the loop stopped, if it was not asked to.
	fn into_result(self) -> Result { self.failure.map_or(Ok(()), Err) }

	/// Records a failure and asks the loop to stop.
	fn fail(&mut self, event_loop: &ActiveEventLoop, error: Error) {
		error!(%error, "stopping");
		self.failure = Some(error);
		event_loop.exit();
	}

	/// Takes the window over from the loader, once the world is up.
	fn take_over(&mut self, event_loop: &ActiveEventLoop, runtime: Runtime) {
		let Phase::Loading(loader) = std::mem::replace(&mut self.phase, Phase::Over) else {
			return;
		};

		match App::new(*loader, runtime) {
			| Ok(app) => self.phase = Phase::Running(Box::new(app)),
			| Err(error) => self.fail(event_loop, error),
		}
	}
}

impl ApplicationHandler for Boot {
	fn resumed(&mut self, event_loop: &ActiveEventLoop) {
		let Phase::Loading(loader) = &mut self.phase else {
			return;
		};

		if loader.renderer.is_some() {
			return;
		}

		if let Err(error) = loader.start(event_loop) {
			self.fail(event_loop, error);
		}
	}

	fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
		match &mut self.phase {
			| Phase::Loading(loader) => match loader.window_event(event_loop, &event) {
				| Ok(Some(runtime)) => self.take_over(event_loop, runtime),
				| Ok(None) => {},
				| Err(error) => self.fail(event_loop, error),
			},
			| Phase::Running(app) => app.window_event(event_loop, id, &event),
			| Phase::Over => {},
		}
	}

	fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
		match &mut self.phase {
			| Phase::Loading(loader) => loader.request_redraw(),
			| Phase::Running(app) => app.about_to_wait(event_loop),
			| Phase::Over => {},
		}
	}

	fn exiting(&mut self, event_loop: &ActiveEventLoop) {
		match &mut self.phase {
			| Phase::Loading(loader) =>
				if let Some(error) = loader.close() {
					self.failure = Some(error);
				},
			| Phase::Running(app) => app.exiting(event_loop),
			| Phase::Over => {},
		}
	}
}

/// The window's first half: the window, the device, and the world coming up
/// behind a screen that says how far it has come.
///
/// One stage a frame, so that the screen is drawn between them; the build
/// stage is another process and is looked at rather than waited for, so the
/// window goes on answering while cargo works. A stage that fails stops the
/// stand-up and leaves the screen up with the failure on it, because a person
/// who opened the project from a list has nowhere else to read it.
struct Loader {
	build: Build,
	standing: Standing,
	project: Project,
	gpu: Option<Gpu>,
	renderer: Option<Renderer>,
	#[cfg(feature = "editor")]
	screen: Option<Loading>,
	opening: Option<Opening>,
	failure: Option<Error>,
}

impl Loader {
	/// A world waiting for its window.
	fn new(build: Build, standing: Standing, project: &Project) -> Self {
		Self {
			build,
			standing,
			project: project.clone(),
			gpu: None,
			renderer: None,
			#[cfg(feature = "editor")]
			screen: None,
			opening: None,
			failure: None,
		}
	}

	/// Opens the window and the device, and starts the world coming up.
	///
	/// The device before the world, which is the other way round from how a
	/// picture does it, because the screen the world comes up behind needs
	/// something to be drawn on. Which APIs the device may use is the
	/// archive's to say and there is no table yet to run the archive against,
	/// so the one variable is read from the file by name.
	fn start(&mut self, event_loop: &ActiveEventLoop) -> Result {
		let attributes = Window::default_attributes()
			.with_title(TITLE)
			.with_inner_size(SIZE);

		let window = Arc::new(
			event_loop
				.create_window(attributes)
				.map_err(|error| err!(Graphics("creating the window: {error}")))?,
		);

		// against this window, so that the adapter chosen is one that can
		// present to it; the surface itself is the renderer's. @ref
		// `colby_engine::gpu`.
		let asked = crate::console::archived(&self.project.settings(), gpu::BACKEND);
		let Some(gpu) = Gpu::open(gpu::backends(asked.as_deref()), Some(&window))? else {
			return Err!(Graphics("no usable adapter, so there is nothing to draw with"));
		};

		let renderer = Renderer::new(&gpu, window)?;

		self.start_screen(&renderer);

		let mut opening =
			Opening::start(Front::Window(self.standing), &self.project, &self.build)?;
		opening.world_mut().aspect = renderer.aspect();

		self.gpu = Some(gpu);
		self.renderer = Some(renderer);
		self.opening = Some(opening);

		info!(
			project = self.project.id(),
			"the window is open; the world is coming up behind it"
		);

		Ok(())
	}

	/// Reacts to one window event.
	///
	/// @return the runtime, the frame every stage has run
	fn window_event(
		&mut self,
		event_loop: &ActiveEventLoop,
		event: &WindowEvent,
	) -> Result<Option<Runtime>> {
		self.offer(event);

		match event {
			| WindowEvent::CloseRequested => {
				event_loop.exit();

				Ok(None)
			},
			| WindowEvent::Resized(size) => {
				if let Some(renderer) = self.renderer.as_mut() {
					renderer.resize(size.width, size.height);
				}

				Ok(None)
			},
			| WindowEvent::RedrawRequested => self.frame(),
			| _ => Ok(None),
		}
	}

	/// Brings up the screen against the window and the device.
	#[cfg(feature = "editor")]
	fn start_screen(&mut self, renderer: &Renderer) {
		self.screen = Some(Loading::new(
			renderer.window(),
			renderer.device(),
			renderer.format(),
			self.project.name(),
		));
	}

	/// Nothing to bring up; this build has no screen.
	#[cfg(not(feature = "editor"))]
	#[expect(
		clippy::unused_self,
		clippy::needless_pass_by_ref_mut,
		reason = "the editor build of this function needs both, and the two have to agree on a \
		          signature"
	)]
	fn start_screen(&mut self, _renderer: &Renderer) {}

	/// Offers an event to the screen, so that it follows the window.
	#[cfg(feature = "editor")]
	fn offer(&mut self, event: &WindowEvent) {
		if let (Some(screen), Some(renderer)) = (self.screen.as_mut(), self.renderer.as_ref()) {
			let window = Arc::clone(renderer.window());

			screen.on_event(&window, event);
		}
	}

	/// Nothing to offer it to; this build has no screen.
	#[cfg(not(feature = "editor"))]
	#[expect(
		clippy::unused_self,
		clippy::needless_pass_by_ref_mut,
		reason = "the editor build of this function needs both, and the two have to agree on a \
		          signature"
	)]
	fn offer(&mut self, _event: &WindowEvent) {}

	/// Asks for another frame.
	fn request_redraw(&self) {
		if let Some(renderer) = self.renderer.as_ref() {
			renderer.window().request_redraw();
		}
	}

	/// Draws the screen as things stand, then runs one stage.
	///
	/// The picture first, so that what is on screen while a stage runs is the
	/// line saying which stage; a stage that takes a second would otherwise
	/// be a second of the previous frame.
	///
	/// @return the runtime, once every stage has run
	fn frame(&mut self) -> Result<Option<Runtime>> {
		self.draw()?;

		if self.failure.is_some() {
			return Ok(None);
		}

		let Some(opening) = self.opening.as_mut() else {
			return Ok(None);
		};

		match opening.advance() {
			| Ok(Progress::Done) => {
				let Some(opening) = self.opening.take() else {
					return Ok(None);
				};

				Ok(Some(opening.finish()?))
			},
			| Ok(Progress::Moved(stage)) => {
				debug!(stage = stage.title(), "up");

				Ok(None)
			},
			| Ok(Progress::Waiting(_)) => Ok(None),
			| Err(error) => {
				// the screen stays, with the failure on it, until the window is
				// closed: somebody who opened this from a list has nowhere else
				// to read why it did not open.
				error!(%error, "the world did not come up; close the window");
				self.failure = Some(error);

				Ok(None)
			},
		}
	}

	/// Draws the world as it stands and the screen over it.
	fn draw(&mut self) -> Result {
		let (Some(renderer), Some(opening)) = (self.renderer.as_mut(), self.opening.as_ref())
		else {
			return Ok(());
		};

		let mut overlays: Vec<&mut dyn Overlay> = Vec::new();

		#[cfg(feature = "editor")]
		if let Some(screen) = self.screen.as_mut() {
			let window = Arc::clone(renderer.window());
			let steps = steps(opening.next());
			let failure = self.failure.as_ref().map(ToString::to_string);

			screen.run(&window, &steps, failure.as_deref());
			overlays.push(screen);
		}

		renderer.render(opening.world(), &mut overlays)
	}

	/// Takes the window down, the screen before the renderer whose surface
	/// borrows the window.
	///
	/// @return the failure that stopped the stand-up, if one did
	fn close(&mut self) -> Option<Error> {
		self.drop_screen();
		self.opening = None;
		self.renderer = None;

		self.failure.take()
	}

	/// Takes the screen down.
	#[cfg(feature = "editor")]
	fn drop_screen(&mut self) { self.screen = None; }

	/// Nothing to take down; this build has no screen.
	#[cfg(not(feature = "editor"))]
	#[expect(
		clippy::unused_self,
		clippy::needless_pass_by_ref_mut,
		reason = "as start_screen"
	)]
	fn drop_screen(&mut self) {}
}

/// Every stage as the screen shows it, given the one about to run.
#[cfg(feature = "editor")]
fn steps(next: Option<Stage>) -> Vec<Step<'static>> {
	let current = next.and_then(|stage| Stage::ALL.iter().position(|it| *it == stage));

	Stage::ALL
		.iter()
		.enumerate()
		.map(|(index, stage)| Step {
			title: stage.title(),
			state: match current {
				| Some(at) if index < at => State::Done,
				| Some(at) if index == at => State::Current,
				| Some(_) => State::Pending,
				| None => State::Done,
			},
		})
		.collect()
}

/// The window's half of the process.
pub(crate) struct App {
	/// Everything a world needs to run, which is the same for a window as for
	/// anything else. @ref [`Runtime`].
	runtime: Runtime,
	clock: Clock,
	/// Frames drawn. Not the same number as `world.steps`, which is the point
	/// of the whole arrangement.
	frames: u64,
	input: Input,
	/// The process's one device, made with the window and kept as long as
	/// anything draws: the renderer and every picture taken while the window
	/// is open are built on it. @ref `colby_engine::gpu`.
	gpu: Option<Gpu>,
	renderer: Option<Renderer>,
	/// Whether the world is being played or edited, and the world play started
	/// from. @ref `crate::mode`.
	mode: Mode,
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
	/// What the build script knew, handed down to the watcher.
	#[cfg(feature = "hot_reload")]
	build: Build,
	failure: Option<Error>,
}

impl App {
	/// Takes the window over from the loader, with the world up.
	///
	/// Everything that needs both a device and a runtime happens here: the
	/// interface's pipeline, the editor, the watcher. The clock is reset last,
	/// because everything before it took as long as it took and none of that
	/// is time the simulation owes.
	///
	/// @param loader - the window and the device, done loading
	/// @param runtime - the world, up
	#[cfg_attr(
		not(feature = "hot_reload"),
		expect(
			unused_variables,
			reason = "with hot-reload built in the loader's facts are kept for the watcher; \
			          without it there is nothing to rebuild, and the two have to agree on a \
			          signature"
		)
	)]
	fn new(loader: Loader, runtime: Runtime) -> Result<Self> {
		let Loader { build, gpu, renderer, .. } = loader;

		// after the config, because that is where a rate somebody asked for
		// arrives, and before the first step, because that is the last moment
		// saying anything about it is useful. @ref `set_pace` for why a client
		// does not get its own rate.
		if runtime.following() && rate(&runtime.world.cvars) != Rate::DEFAULT {
			warn!(
				asked = rate(&runtime.world.cvars).hz(),
				running = Rate::DEFAULT.hz(),
				"a window that connected runs at the host's rate, not its own"
			);
		}

		let mut app = Self {
			runtime,
			clock: Clock::new(),
			frames: 0,
			input: Input::default(),
			gpu,
			renderer,
			mode: Mode::new(),
			#[cfg(feature = "editor")]
			editor: None,
			#[cfg(feature = "hot_reload")]
			watch: None,
			mix: Mix::FULL,
			gravity: -PULL,
			#[cfg(feature = "hot_reload")]
			build,
			failure: None,
		};

		app.start()?;

		Ok(app)
	}

	/// Brings up everything that needs the device and the world at once.
	fn start(&mut self) -> Result {
		let Some(renderer) = self.renderer.as_ref() else {
			return Err!(Graphics("the loader handed over no renderer"));
		};
		let (width, height) = renderer.size();

		self.runtime.world.aspect = renderer.aspect();
		self.input
			.set_viewport(f64::from(width), f64::from(height));

		if let Err(error) = self
			.runtime
			.interface
			.attach(renderer.device(), renderer.format())
		{
			// a log line rather than a stop: an engine whose scene draws and
			// whose interface does not is still worth looking at, and the
			// message says which half is missing.
			error!(%error, "the interface has no pipeline; nothing it draws will be on screen");
		}

		self.start_editor();
		self.start_watching()?;

		// everything above and everything the loader did took as long as it
		// took: a module, the asset tree, a window, an adapter and a shader.
		// Without this the first frame would arrive owing the simulation
		// seconds of catch-up it never actually missed.
		self.clock.reset();

		info!("colby is running; escape or the window close button stops it");

		Ok(())
	}

	/// Brings up the editor against the window and the device.
	///
	/// Its variables were registered when the runtime opened, before the game
	/// module, so that they exist whether or not the editor was built and
	/// belong to the engine rather than to the module.
	#[cfg(feature = "editor")]
	fn start_editor(&mut self) {
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
		if !Editor::shown(&self.runtime.world) {
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
			editor.run(&window, &mut self.runtime.world, &self.clock, self.frames);
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
		if !Editor::shown(&self.runtime.world) {
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
		// a project with no game crate has nothing to watch: its world is its
		// scenes and its programs, and those the asset loop already watches.
		let Some(sources) = self.runtime.project().game() else {
			return Ok(());
		};
		let module = self.runtime.project().module();

		self.watch = Some(Watch::new(&module, sources, &self.build)?);

		Ok(())
	}

	/// Does nothing; there is no module to watch.
	#[cfg(not(feature = "hot_reload"))]
	#[expect(
		clippy::unused_self,
		clippy::needless_pass_by_ref_mut,
		reason = "the hot-reload variant of this function needs both, and the two have to agree \
		          on a signature"
	)]
	fn start_watching(&mut self) -> Result { Ok(()) }

	/// Runs whatever simulation this frame owes, then draws between the last
	/// two states of it.
	///
	/// Zero steps at a high refresh rate, one most of the time, several after a
	/// hitch - and a picture either way, because what makes the motion smooth
	/// is where the frame sits between two steps rather than how many of them
	/// it ran.
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

		// the asset trees, the terminal, every line a command left waiting, and
		// the socket - all of it before any step, and all of it the runtime's.
		// @ref `Runtime::poll` for the order and the reasons.
		self.runtime.poll();

		// and the mode's own edge, in the same place and for the same reason:
		// stopping play replaces every table in the world, so it happens
		// between steps rather than inside one.
		let editing = crate::mode::wanted(&self.runtime.world);
		self.mode
			.follow(&mut self.runtime.world, &mut self.runtime.simulation, editing);

		pay(&mut self.runtime.world, &mut self.clock);

		let asked = self
			.runtime
			.world
			.cvars
			.float(GRAVITY)
			.unwrap_or(-PULL);
		if (asked - self.gravity).abs() > f32::EPSILON {
			self.gravity = asked;
			self.runtime.world.gravity = Vec3::new(0.0, asked, 0.0);
		}

		self.hear();

		if let Some(renderer) = self.renderer.as_ref() {
			// before the steps rather than after them: gameplay asking how
			// wide the window is should not be told last frame's answer, four
			// times in a row.
			self.runtime.world.aspect = renderer.aspect();

			let (width, height) = renderer.size();
			#[expect(
				clippy::as_conversions,
				clippy::cast_possible_truncation,
				reason = "a display scale is between one and four, and the only f32 it is ever \
				          multiplied by is a pixel count"
			)]
			let scale = renderer.window().scale_factor() as f32;

			self.runtime.world.ui.set_viewport(
				Vec2::new(
					f32::from(u16::try_from(width).unwrap_or(u16::MAX)),
					f32::from(u16::try_from(height).unwrap_or(u16::MAX)),
				),
				scale,
			);
			self.runtime
				.world
				.ui
				.set_pointer(Vec2::from(self.input.cursor));
		}

		// the moment the next step is at, on the clock the wire is on. Read
		// once a frame and advanced a step at a time by the runtime, rather
		// than read again inside the step loop: what the wire is asked for has
		// to move by exactly one step per step, or the renderer's own blend
		// between two steps is asked to cover a gap that is not one step wide
		// - and a body would be drawn speeding up and slowing down on a wire
		// doing nothing of the kind. @ref `Runtime::step`.
		let mut moment = self.runtime.now();
		// read once for the whole pass rather than per step: a rate that moved
		// between two steps of the same frame would make the second one a
		// different length from the first, which is the one thing a fixed step
		// is for not being.
		let rate = self.clock.rate();

		while let Some(time) = self.clock.step() {
			moment = self
				.runtime
				.step(&mut self.input, rate, time, editing, moment);
		}

		self.frames = self.frames.saturating_add(1);
		// the mode has a say in this: a world being edited is drawn as it
		// stands rather than blended. @ref `Mode::interpolation`.
		self.runtime.world.set_interpolation(
			self.mode
				.interpolation(self.clock.interpolation()),
		);

		// the interface is laid out again here rather than reused from the step:
		// the window may have been resized since, and a document that is a share
		// of the screen should be the share the screen is now.
		self.runtime.interface.run(&self.runtime.world);

		if let Some(renderer) = self.renderer.as_ref() {
			self.runtime.interface.prepare(
				renderer.device(),
				renderer.queue(),
				&self.runtime.world,
			);
		}

		self.run_editor();
		self.screenshots();

		#[cfg(feature = "editor")]
		let shown = Editor::shown(&self.runtime.world);

		// disjoint borrows: the renderer is one field, the interface another and
		// the editor a third, which is what lets the frame be handed to all of
		// them.
		let mut overlays: Vec<&mut dyn Overlay> = vec![&mut self.runtime.interface];

		#[cfg(feature = "editor")]
		if shown && let Some(editor) = self.editor.as_mut() {
			overlays.push(editor);
		}

		let Some(renderer) = self.renderer.as_mut() else {
			return Ok(());
		};

		renderer.render(&self.runtime.world, &mut overlays)
	}

	/// Writes every picture the console asked for since the last frame.
	///
	/// After the interface has been prepared and before the window is drawn,
	/// so that what goes into the file is this frame: the same world and the
	/// same interface, drawn by a second scene on the same device while the
	/// window's own goes on drawing into the window. The editor is not in it,
	/// as it is not in `--shot`: a screenshot is of the game. @ref
	/// [`crate::screenshot`].
	fn screenshots(&mut self) {
		let asked = crate::console::take(&mut self.runtime.world, &[screenshot::COMMAND]);
		if asked.is_empty() {
			return;
		}

		let (Some(gpu), Some(renderer)) = (self.gpu.as_ref(), self.renderer.as_ref()) else {
			return;
		};
		let root = self.runtime.project().root().to_owned();

		for line in asked {
			let name = line.words.first().map(String::as_str);
			let outcome = screenshot::place(&root, name).and_then(|path| {
				screenshot::take(
					gpu,
					renderer.format(),
					renderer.size(),
					&mut self.runtime.world,
					&mut self.runtime.interface,
					&path,
				)
			});

			if let Err(error) = outcome {
				error!(%error, "no screenshot");
			}
		}

		// a second scene's pipelines, its uploads and a readback took as long
		// as they took, and none of it is time the simulation owes: billed as
		// arrears it would be a burst of catch-up steps and a warning about
		// falling behind, once per picture. The same arrangement a module
		// swap has. @ref `reload_if_stale`.
		self.clock.reset();
	}

	/// Puts the console's pacing variables onto the clock.
	///
	/// `sim.pause` is not a separate mechanism from `sim.speed`, it is a speed
	/// of zero - which is what makes unpausing free of a lurch: no time
	/// accumulated while it was held, so there is nothing owed on the way out.
	fn set_pace(&mut self) {
		let cvars = &self.runtime.world.cvars;
		let paused = cvars.bool(PAUSE).unwrap_or(false);
		let speed = cvars.float(SPEED).unwrap_or(1.0);

		// a window that serves is the authority and runs what it was told; one
		// that connected takes the rate every host runs. @ref `paced`.
		self.clock
			.set_rate(paced(cvars, self.runtime.following()));
		self.clock
			.set_speed(if paused { 0.0 } else { speed });
	}

	/// Writes the volumes the console asked for, if any of them moved.
	///
	/// The same shape gravity has, and the same rule: whoever said something
	/// last wins, and standing still says nothing. Four numbers rather than
	/// one, so the comparison is over the whole struct.
	fn hear(&mut self) {
		let asked = crate::console::volumes(&self.runtime.world.cvars);

		if asked != self.mix {
			self.mix = asked;
			self.runtime.world.mix = asked;
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

		if let Some(game) = self.runtime.game.as_mut()
			&& let Err(error) = game.reload(&mut self.runtime.world)
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
			| NamedKey::F1 => Editor::toggle(&mut self.runtime.world),
			// under the same feature as F1, and for the same reason: play and
			// stop is a tool's gesture. A build with no editor in it still has
			// the variable, and nothing in it presses this.
			#[cfg(feature = "editor")]
			| NamedKey::F5 => crate::mode::toggle(&mut self.runtime.world),
			| _ => {},
		}
	}

	/// Records a failure and asks the loop to stop.
	fn fail(&mut self, event_loop: &ActiveEventLoop, error: Error) {
		error!(%error, "stopping");
		self.failure = Some(error);
		event_loop.exit();
	}

	/// Reacts to one window event.
	fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: &WindowEvent) {
		// the editor first: what it takes, the game must not also act on.
		let taken = self.editor_took(event);

		if !taken {
			input::apply(&mut self.input, event);
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
			} if !taken => self.pressed(event_loop, *key),
			| WindowEvent::Resized(size) =>
				if let Some(renderer) = self.renderer.as_mut() {
					renderer.resize(size.width, size.height);
				},
			| WindowEvent::RedrawRequested => {
				if let Err(error) = self.frame() {
					self.fail(event_loop, error);
				} else if self.runtime.world.quit {
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

	/// Asks for another frame.
	fn about_to_wait(&self, _event_loop: &ActiveEventLoop) {
		if let Some(renderer) = self.renderer.as_ref() {
			renderer.window().request_redraw();
		}
	}

	/// Takes everything down, in order.
	fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
		// the config and then the game, and the module dropped with it: it may
		// still be running code from the image, so it goes before the renderer,
		// whose surface borrows the window it is holding the last share of.
		self.runtime.close();
		self.renderer = None;

		info!(
			frames = self.frames,
			steps = self.runtime.world.steps,
			stalls = self.clock.stalls(),
			reloads = self.runtime.world.reloads,
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

	#[cfg(feature = "editor")]
	#[test]
	fn the_screen_is_told_which_stage_is_running_and_which_are_done() {
		let marks = steps(Some(Stage::Build));

		assert_eq!(marks.len(), Stage::ALL.len(), "every stage has a line");
		assert_eq!(marks[0].state, State::Done, "the assets are done");
		assert_eq!(marks[3].state, State::Done, "and the wire");
		assert_eq!(marks[4].state, State::Current, "the build is running");
		assert_eq!(marks[4].title, "building the game crate");
		assert_eq!(marks[5].state, State::Pending, "the module is still to come");
		assert!(
			steps(None)
				.iter()
				.all(|step| step.state == State::Done),
			"nothing left to run is everything done"
		);
		assert!(
			steps(Some(Stage::Assets))
				.iter()
				.skip(1)
				.all(|step| step.state == State::Pending),
			"and the first stage running is nothing done"
		);
	}
}
