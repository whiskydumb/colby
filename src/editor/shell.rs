//! egui's plumbing, once: the context, the window's input, the painter and the
//! frame in flight between them.
//!
//! Three things in this crate draw with egui - the editor over a running
//! world, the launcher over nothing, and the loading screen over a world that
//! is still coming up - and every one of them needs the same six fields and
//! the same three steps: take the window's input, build a frame, paint it when
//! the renderer has a frame to put it in. This is that, and nothing about what
//! the frame holds.

use egui::{ClippedPrimitive, Context, TexturesDelta, Ui};
use egui_wgpu::{Renderer as Painter, RendererOptions, ScreenDescriptor};
use wgpu::{
	CommandEncoderDescriptor, Device, LoadOp, Operations, Queue, RenderPassColorAttachment,
	RenderPassDescriptor, StoreOp, TextureFormat, TextureView,
};
use winit::{event::WindowEvent, window::Window};

/// egui against one window and one device.
pub(crate) struct Shell {
	context: Context,
	state: egui_winit::State,
	painter: Painter,
	/// This frame's triangles, tessellated by [`Shell::run`] and painted by
	/// [`Shell::draw`] once the renderer has a frame to put them in.
	jobs: Vec<ClippedPrimitive>,
	textures: TexturesDelta,
	points: f32,
}

impl Shell {
	/// Brings up egui against the window and the device the frames belong to.
	///
	/// @param window - the window events come from
	/// @param device - the device the frames belong to
	/// @param format - the color format the surface was configured with
	pub(crate) fn new(window: &Window, device: &Device, format: TextureFormat) -> Self {
		let context = Context::default();
		let state = egui_winit::State::new(
			context.clone(),
			context.viewport_id(),
			window,
			None,
			None,
			None,
		);

		// the defaults are what an overlay wants: no multisampling and no depth
		// buffer, because this draws over a frame that is already finished and
		// nothing in it is behind anything else. Dithering stays on; the
		// surface is sRGB, which is what it assumes.
		let painter = Painter::new(device, format, RendererOptions::default());

		Self {
			context,
			state,
			painter,
			jobs: Vec::new(),
			textures: TexturesDelta::default(),
			points: 1.0,
		}
	}

	/// Offers one window event to egui.
	///
	/// @param window - the window the event came from
	/// @param event - the event
	/// @return whether egui took it, in which case whatever is underneath must
	/// not also act on it
	pub(crate) fn on_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
		self.state.on_window_event(window, event).consumed
	}

	/// Builds one frame.
	///
	/// Nothing is drawn here - that happens in [`draw`](Self::draw), when there
	/// is a frame to draw into.
	///
	/// @param window - the window, for input and for the cursor
	/// @param build - what the frame holds, given the context
	pub(crate) fn run<F: FnOnce(&mut Ui)>(&mut self, window: &Window, build: F) {
		let input = self.state.take_egui_input(window);

		// egui asks for a closure it may call more than once and calls it
		// once; a frame is built once, so the closure is taken on the way in.
		let mut build = Some(build);
		let output = self.context.run_ui(input, |ui| {
			if let Some(build) = build.take() {
				build(ui);
			}
		});

		self.state
			.handle_platform_output(window, output.platform_output);

		self.points = output.pixels_per_point;
		self.jobs = self
			.context
			.tessellate(output.shapes, output.pixels_per_point);

		// appended rather than assigned. A frame is not always drawn - the
		// surface can be lost, or the editor can be hidden between building it
		// and painting it - and a `TexturesDelta` that is dropped with anything
		// still in it is a panic, by epaint's own design. Merging means an
		// unpainted frame's font atlas is still applied by the next one that
		// does get painted.
		self.textures.append(output.textures_delta);
	}

	/// Paints the frame [`run`](Self::run) built into a target.
	///
	/// What an [`Overlay`](colby_engine::Overlay) does, for whichever type
	/// holds this shell to delegate to.
	///
	/// @param device - the device the target belongs to
	/// @param queue - its queue
	/// @param target - the frame, with the scene already in it
	/// @param width - the target's width in pixels
	/// @param height - its height
	pub(crate) fn draw(
		&mut self,
		device: &Device,
		queue: &Queue,
		target: &TextureView,
		width: u32,
		height: u32,
	) {
		let screen = ScreenDescriptor {
			size_in_pixels: [width, height],
			pixels_per_point: self.points,
		};

		// taken, so that what is applied here cannot also be applied again, and
		// so that the cleared remainder is what gets dropped.
		let mut textures = std::mem::take(&mut self.textures);

		for (id, deltas) in &textures.set {
			for delta in deltas {
				self.painter
					.update_texture(device, queue, *id, delta);
			}
		}

		let mut encoder =
			device.create_command_encoder(&CommandEncoderDescriptor { label: Some("egui") });

		self.painter
			.update_buffers(device, queue, &mut encoder, &self.jobs, &screen);

		// `Load`, not `Clear`: the scene is already in this frame, and the
		// point of an overlay is to be over something.
		let mut pass = encoder
			.begin_render_pass(&RenderPassDescriptor {
				label: Some("egui"),
				color_attachments: &[Some(RenderPassColorAttachment {
					view: target,
					depth_slice: None,
					resolve_target: None,
					ops: Operations {
						load: LoadOp::Load,
						store: StoreOp::Store,
					},
				})],
				depth_stencil_attachment: None,
				timestamp_writes: None,
				occlusion_query_set: None,
				multiview_mask: None,
			})
			.forget_lifetime();

		self.painter
			.render(&mut pass, &self.jobs, &screen);
		drop(pass);

		queue.submit([encoder.finish()]);

		// after the pass, not before it: a texture freed while the commands that
		// sample it are still queued is a texture the driver is entitled to
		// complain about.
		for id in &textures.free {
			self.painter.free_texture(id);
		}

		textures.clear();
	}
}

impl Drop for Shell {
	fn drop(&mut self) {
		// epaint asserts that a delta is applied rather than dropped, which is
		// the right rule while a frame is being built and the wrong one for a
		// process on its way out. Nothing is going to paint this.
		self.textures.clear();
	}
}
