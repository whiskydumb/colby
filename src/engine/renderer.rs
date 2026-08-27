//! The window half of drawing: a surface, its configuration, and presenting.
//!
//! Written straight against wgpu, with no `RenderContext` trait in sight. An
//! abstraction designed under exactly one backend puts its seams in the wrong
//! places; it can be extracted the day an ash backend exists to disagree with
//! it.
//!
//! Everything that is not about the window lives in [`Scene`], which is what
//! makes an offscreen render possible, and with it a test that reads the pixels
//! back. @ref [`capture`](crate::capture).

use std::sync::Arc;

use colby_core::{Err, Result, abi::World, debug, err};
use wgpu::{
	Backends, CurrentSurfaceTexture, Device, DeviceDescriptor, ExperimentalFeatures, Features,
	Instance, InstanceDescriptor, Limits, MemoryHints, PowerPreference, PresentMode,
	RequestAdapterOptions, Surface, SurfaceConfiguration, TextureFormat, TextureUsages,
	TextureViewDescriptor, Trace,
};
use winit::window::Window;

use crate::{overlay::Overlay, scene::Scene};

/// A window, its surface, and the scene drawn into it.
///
/// The surface borrows the window, so the window is held behind an `Arc` and
/// the renderer keeps a share of it. The renderer therefore has to be dropped
/// before the last other reference to the window goes away.
pub struct Renderer {
	surface: Surface<'static>,
	config: SurfaceConfiguration,
	scene: Scene,
	window: Arc<Window>,
}

impl Renderer {
	/// Brings up an adapter, a device, the pipeline and the built-in meshes.
	///
	/// @param window - the window to present into
	/// @return a renderer ready for [`render`](Self::render)
	pub fn new(window: Arc<Window>) -> Result<Self> {
		let mut renderer = pollster::block_on(Self::create(window))?;

		// before the first frame rather than after it. Windows hands out a
		// window at a default size and applies the requested one a moment
		// later, so the size read while the adapter and the device were being
		// asked for can be a size this window never had - and nothing reports
		// that as a resize, because from the window's side nothing resized.
		// Without this the first picture is drawn into a surface built for
		// something else, and the surface spends the rest of the run saying so.
		renderer.resync();

		Ok(renderer)
	}

	/// Reacts to the window changing size.
	///
	/// A minimized window reports a zero-sized surface, which is not
	/// configurable, so the request is dropped rather than turned into a
	/// validation error.
	///
	/// @param width - the new width in physical pixels
	/// @param height - the new height in physical pixels
	pub fn resize(&mut self, width: u32, height: u32) {
		if width == 0 || height == 0 {
			return;
		}

		self.config.width = width;
		self.config.height = height;
		self.reconfigure();
	}

	/// Draws one frame from what the game left in the world, and presents it.
	///
	/// @param world - the host state, with this frame's game output written
	/// @param overlays - what to draw on top of the scene, in order, @ref
	/// [`Overlay`]. The game's interface first and the editor over it: a tool
	/// that could be hidden behind the thing it is inspecting would be a tool
	/// nobody could use.
	/// @return `Ok` once the frame has been submitted and presented
	pub fn render(&mut self, world: &World, overlays: &mut [&mut dyn Overlay]) -> Result {
		// whether the swapchain stopped matching the window while this frame
		// was being handed out. @ref the note where it is acted on, below.
		let mut stale = false;

		let frame = match self.surface.get_current_texture() {
			| CurrentSurfaceTexture::Success(frame) => frame,
			| CurrentSurfaceTexture::Suboptimal(frame) => {
				// still good enough to draw into, so draw into it - but the
				// reconfigure has to wait until it has been handed back.
				stale = true;

				frame
			},
			| CurrentSurfaceTexture::Lost | CurrentSurfaceTexture::Outdated => {
				debug!("surface out of date, reconfiguring");
				self.reconfigure();

				return Ok(());
			},
			| state @ (CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded) => {
				// nothing is drawn and nothing is presented, so whatever was on
				// screen stays there. Worth a line: a run of these looks exactly
				// like the picture having frozen, and used to say nothing at all.
				debug!(?state, "no frame to draw into; skipping");

				return Ok(());
			},
			| CurrentSurfaceTexture::Validation => {
				return Err!(Graphics("acquiring the next frame raised a validation error"));
			},
		};

		let view = frame
			.texture
			.create_view(&TextureViewDescriptor::default());

		self.scene.render(&view, world);

		// after the scene and before the surface goes back: an overlay draws
		// into the frame the scene was just recorded into, and submits its own
		// work behind the scene's on the same queue.
		for overlay in overlays {
			overlay.draw(
				self.scene.device(),
				self.scene.queue(),
				&view,
				self.config.width,
				self.config.height,
			);
		}

		self.window.pre_present_notify();
		self.scene.queue().present(frame);

		// and only now. A surface cannot be configured while a texture it
		// handed out is still alive, and `present` is what hands this one back:
		// reconfiguring a few lines earlier, with the frame still in scope, is
		// a wgpu validation error and therefore a panic - which is what used to
		// happen the moment the window was minimized, resized or moved to a
		// display with another scale, all of which are `Suboptimal`.
		if stale {
			self.resync();
		}

		Ok(())
	}

	/// The surface size in physical pixels.
	#[must_use]
	pub const fn size(&self) -> (u32, u32) { (self.config.width, self.config.height) }

	/// The window this renderer presents into.
	#[must_use]
	pub const fn window(&self) -> &Arc<Window> { &self.window }

	/// The device every frame belongs to.
	///
	/// For an [`Overlay`], which has to build its own pipelines against the
	/// same device rather than a second one.
	#[must_use]
	pub const fn device(&self) -> &Device { self.scene.device() }

	/// The queue every frame's work is submitted on.
	///
	/// For an [`Overlay`], which writes its own buffers and textures.
	#[must_use]
	pub const fn queue(&self) -> &wgpu::Queue { self.scene.queue() }

	/// The color format the surface was configured with.
	#[must_use]
	pub const fn format(&self) -> TextureFormat { self.config.format }

	/// The surface's width divided by its height.
	///
	/// Never zero: the surface is never configured with a zero dimension, @ref
	/// [`resize`](Self::resize).
	#[must_use]
	#[expect(
		clippy::as_conversions,
		clippy::cast_precision_loss,
		reason = "u32 to f32 loses precision above 2^24, which is four thousand times wider \
		          than any surface this will ever be configured with"
	)]
	pub fn aspect(&self) -> f32 {
		let width = self.config.width.max(1) as f32;
		let height = self.config.height.max(1) as f32;

		width / height
	}

	/// Reconfigures the surface and the depth buffer to the current size.
	/// Reconfigures the surface, but only if the window is a different size.
	///
	/// The `Suboptimal` path. A swapchain can report itself suboptimal for
	/// reasons that reconfiguring does not fix, and reconfiguring is not free:
	/// doing it on every such frame is its own flicker, and one that survives
	/// the size being right. Two hundred reconfigures in a run became five.
	fn resync(&mut self) {
		let size = self.window.inner_size();

		if size.width == 0 || size.height == 0 {
			return;
		}

		if size.width != self.config.width || size.height != self.config.height {
			self.reconfigure();
		}
	}

	fn reconfigure(&mut self) {
		// the window is the authority on its own size, not the configuration
		// this last wrote. Reconfiguring from the stored numbers is how a
		// mismatch turns into a loop: the surface reports itself out of date,
		// this hands back the same size it already refused, and the next frame
		// says the same thing - once a frame, for as long as the window is
		// open. On screen that is a picture that flickers and will not settle,
		// and it clears the moment anything else corrects the size, which is
		// why going full screen and back used to "fix" it.
		let size = self.window.inner_size();
		let moved = size.width != self.config.width || size.height != self.config.height;

		if size.width > 0 && size.height > 0 && moved {
			debug!(
				width = size.width,
				height = size.height,
				was_width = self.config.width,
				was_height = self.config.height,
				"the surface was configured for a size the window is not"
			);

			self.config.width = size.width;
			self.config.height = size.height;
		}

		self.surface
			.configure(self.scene.device(), &self.config);
		self.scene
			.resize(self.config.width, self.config.height);
	}

	/// The async half of [`new`](Self::new).
	async fn create(window: Arc<Window>) -> Result<Self> {
		let instance = Instance::new(InstanceDescriptor {
			backends: Backends::DX12 | Backends::VULKAN,
			..InstanceDescriptor::new_without_display_handle()
		});

		let surface = instance
			.create_surface(Arc::clone(&window))
			.map_err(|error| err!(Graphics("creating the surface: {error}")))?;

		let adapter = instance
			.request_adapter(&RequestAdapterOptions {
				power_preference: PowerPreference::HighPerformance,
				compatible_surface: Some(&surface),
				..Default::default()
			})
			.await
			.map_err(|error| err!(Graphics("no usable adapter: {error}")))?;

		let info = adapter.get_info();
		debug!(adapter = %info.name, backend = ?info.backend, "adapter selected");

		let (device, queue) = adapter
			.request_device(&DeviceDescriptor {
				label: Some("colby"),
				required_features: Features::empty(),
				required_limits: Limits::default(),
				experimental_features: ExperimentalFeatures::disabled(),
				memory_hints: MemoryHints::Performance,
				trace: Trace::Off,
			})
			.await
			.map_err(|error| err!(Graphics("requesting a device: {error}")))?;

		// read again rather than reused: `size` was taken before the adapter and
		// the device were asked for, which on Windows is long enough for the
		// window to have settled on a different one - a scaled display gives
		// the size it was asked for first and the size it really is after. A
		// surface configured for the wrong one is out of date from its first
		// frame.
		let size = window.inner_size();

		// @note: taken from wgpu rather than filled in field by field, so that a
		// new field in `SurfaceConfiguration` is a default here instead of a
		// compile error in a crate that has no opinion about it.
		let mut config = surface
			.get_default_config(&adapter, size.width.max(1), size.height.max(1))
			.ok_or_else(|| err!(Graphics("the adapter cannot present to this surface")))?;

		config.usage = TextureUsages::RENDER_ATTACHMENT;
		config.present_mode = PresentMode::AutoVsync;

		let capabilities = surface.get_capabilities(&adapter);
		if let Some(srgb) = capabilities
			.formats
			.iter()
			.copied()
			.find(TextureFormat::is_srgb)
		{
			config.format = srgb;
		}

		surface.configure(&device, &config);

		let scene = Scene::new(device, queue, config.format, config.width, config.height)?;

		Ok(Self { surface, config, scene, window })
	}
}
