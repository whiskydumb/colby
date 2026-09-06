//! The launcher's window: a list and a form over no world at all.
//!
//! What `colby` opens when it is started with no project named and none in
//! the working directory - the project manager's own shape, one binary and a
//! mode when nothing else was asked for. It opens no runtime: there is no
//! project to open one on, and the window is a screen with a list in it. The
//! project a person picks runs in a process of its own, this same executable
//! started with `--project`, and this window closes the moment that process
//! has been started - which keeps one window on screen at a time and keeps
//! every way a world comes up in one place, @ref [`crate::run`].
//!
//! The screen itself is the editor crate's, @ref [`colby_editor::Launcher`]:
//! what is here is the window, the device, and the hand-off.

use std::{
	env,
	path::{Path, PathBuf},
	process::Command,
	sync::Arc,
};

use colby_core::{Err, Error, Result, abi::World, err, error, info};
use colby_editor::{Action, Launcher};
use colby_engine::{Gpu, Overlay, Renderer, gpu, winit};
use winit::{
	application::ApplicationHandler,
	dpi::LogicalSize,
	event::WindowEvent,
	event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
	window::{Window, WindowId},
};

use crate::{Build, launch};

/// The window title.
const TITLE: &str = "colby engine";

/// The window's size, in logical pixels: a screen, not a viewport.
const SIZE: LogicalSize<f64> = LogicalSize::new(600.0, 600.0);

/// Opens the launcher and runs it until a project is picked or the window is
/// closed.
///
/// @param build - what the build script knew, for where the engine is
pub(crate) fn run(build: &Build) -> Result {
	let event_loop =
		EventLoop::new().map_err(|error| err!(Graphics("creating the event loop: {error}")))?;

	// `Poll`, as the game's window does: a text box's caret blinks and a list
	// scrolls, and the surface's vsync is what paces it.
	event_loop.set_control_flow(ControlFlow::Poll);

	let mut hub = Hub::new(&build.engine);

	event_loop
		.run_app(&mut hub)
		.map_err(|error| err!(Graphics("running the event loop: {error}")))?;

	hub.into_result()
}

/// The launcher's half of the process: a window, the device and the screen.
struct Hub {
	/// The engine checkout, where the templates are.
	engine: PathBuf,

	/// A world with nothing in it, because the renderer draws one and this
	/// window has none: the clear color behind the page.
	world: Box<World>,
	gpu: Option<Gpu>,
	renderer: Option<Renderer>,
	launcher: Option<Launcher>,
	failure: Option<Error>,
}

impl Hub {
	/// A launcher waiting for its window.
	fn new(engine: &Path) -> Self {
		Self {
			engine: engine.to_owned(),
			world: Box::default(),
			gpu: None,
			renderer: None,
			launcher: None,
			failure: None,
		}
	}

	/// The reason the loop stopped, if it was not asked to.
	fn into_result(self) -> Result { self.failure.map_or(Ok(()), Err) }

	/// Opens the window and brings up everything that needs a device.
	fn start(&mut self, event_loop: &ActiveEventLoop) -> Result {
		let attributes = Window::default_attributes()
			.with_title(TITLE)
			.with_inner_size(SIZE)
			.with_resizable(false);

		let window = Arc::new(
			event_loop
				.create_window(attributes)
				.map_err(|error| err!(Graphics("creating the window: {error}")))?,
		);

		// no project, so no `settings.cfg` and no `r.backend`: the environment
		// or the default, which is what a picture gets too.
		let Some(gpu) = Gpu::open(gpu::backends(None), Some(&window))? else {
			return Err!(Graphics("no usable adapter, so there is nothing to draw with"));
		};

		let renderer = Renderer::new(&gpu, window)?;

		self.launcher = Some(Launcher::new(
			renderer.window(),
			renderer.device(),
			renderer.format(),
			&self.engine,
		));
		self.world.aspect = renderer.aspect();
		self.gpu = Some(gpu);
		self.renderer = Some(renderer);

		info!("the launcher is open; pick a project or make one");

		Ok(())
	}

	/// Builds and draws one frame, and hands a picked project off.
	fn frame(&mut self, event_loop: &ActiveEventLoop) -> Result {
		let (Some(renderer), Some(launcher)) = (self.renderer.as_mut(), self.launcher.as_mut())
		else {
			return Ok(());
		};

		// the window is behind an `Arc` for exactly this: the renderer holds
		// it, and the launcher needs it while the renderer is borrowed.
		let window = Arc::clone(renderer.window());

		if let Some(Action::Open(dir)) = launcher.run(&window) {
			open(&dir)?;
			event_loop.exit();

			return Ok(());
		}

		self.world.aspect = renderer.aspect();

		let overlay: &mut dyn Overlay = launcher;

		renderer.render(&self.world, &mut [overlay], None)
	}

	/// Records a failure and asks the loop to stop.
	fn fail(&mut self, event_loop: &ActiveEventLoop, error: Error) {
		error!(%error, "stopping");
		self.failure = Some(error);
		event_loop.exit();
	}
}

impl ApplicationHandler for Hub {
	fn resumed(&mut self, event_loop: &ActiveEventLoop) {
		if self.renderer.is_some() {
			return;
		}

		if let Err(error) = self.start(event_loop) {
			self.fail(event_loop, error);
		}
	}

	fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
		if let (Some(launcher), Some(renderer)) = (self.launcher.as_mut(), self.renderer.as_ref())
		{
			let window = Arc::clone(renderer.window());

			launcher.on_event(&window, &event);
		}

		match event {
			| WindowEvent::CloseRequested => event_loop.exit(),
			| WindowEvent::Resized(size) =>
				if let Some(renderer) = self.renderer.as_mut() {
					renderer.resize(size.width, size.height);
				},
			| WindowEvent::RedrawRequested =>
				if let Err(error) = self.frame(event_loop) {
					self.fail(event_loop, error);
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
		// the launcher before the renderer, whose surface borrows the window
		// the launcher's egui is also looking at.
		self.launcher = None;
		self.renderer = None;

		info!("the launcher closed");
	}
}

/// Starts this same executable on a project.
///
/// A process of its own rather than this one turning into the editor: the
/// world comes up the one way it always does, in a process that was started
/// on a project, and the launcher never has to know how. The child inherits
/// the terminal, so what it logs lands where the launcher's did.
///
/// @param dir - the project's directory
fn open(dir: &Path) -> Result {
	let exe = env::current_exe()
		.map_err(|error| err!(Err("this executable's own path cannot be read: {error}")))?;

	let child = Command::new(&exe)
		.arg(launch::PROJECT)
		.arg(dir)
		.spawn()
		.map_err(|error| err!(Err("{} could not be started: {error}", exe.display())))?;

	info!(project = %dir.display(), pid = child.id(), "opening in a process of its own");

	Ok(())
}
