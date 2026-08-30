//! The editor: egui drawn over the scene, in the same frame.
//!
//! Engine-side, not gameplay-side - editing this is a restart, the same as
//! editing the renderer. It is its own crate so that egui stays out of
//! `colby_engine` (a renderer with an opinion about buttons is a renderer that
//! cannot be reused) and so that a shipping build drops the whole thing by
//! turning off one feature.
//!
//! Three windows, and each is a view onto something that already existed rather
//! than a new system: the console shows the table and the log the *previous*
//! step built, the statistics show the clock and the world, and the scene tree
//! shows the three tables a world is made of. That is the whole design brief
//! for an editor here - if a panel needs the engine to grow a new mechanism to
//! feed it, the panel is wrong.
//!
//! What can be checked by running it rather than by looking at it lives in
//! [`select`], deliberately: a module with no egui in it is a module with
//! tests.
//!
//! @note: the game's own interface will not be this. egui is for tools; a game
//! draws its interface with HTML/CSS over taffy, which is a separate subsystem
//! that happens to arrive through the same [`Overlay`] seam.

use colby_core::{
	abi::{World, cvar::Value},
	time::Clock,
};
use egui_wgpu::{Renderer as Painter, RendererOptions, ScreenDescriptor};
use wgpu::{
	CommandEncoderDescriptor, Device, LoadOp, Operations, Queue, RenderPassColorAttachment,
	RenderPassDescriptor, StoreOp, TextureFormat, TextureView,
};
use winit::{event::WindowEvent, window::Window};

mod console;
mod select;
mod stats;
mod tree;

/// The variable that decides whether the editor is on screen.
///
/// Saved, so that closing it stays closed. `editor.show 1` from the console
/// works as well as the key, because it is the same variable either way.
pub const SHOW: &str = "editor.show";

/// egui, and everything colby keeps on its behalf.
pub struct Editor {
	context: egui::Context,
	state: egui_winit::State,
	painter: Painter,
	/// This frame's triangles, tessellated by [`Editor::run`] and painted by
	/// [`Overlay::draw`] once the renderer has a frame to put them in.
	jobs: Vec<egui::ClippedPrimitive>,
	textures: egui::TexturesDelta,
	points: f32,
	console: console::Console,
	tree: tree::Tree,
}

impl Editor {
	/// Brings up egui against the window and the device the scene draws with.
	///
	/// @param window - the window events come from
	/// @param device - the device the frames belong to
	/// @param format - the color format the surface was configured with
	#[must_use]
	pub fn new(window: &Window, device: &Device, format: TextureFormat) -> Self {
		let context = egui::Context::default();
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
			textures: egui::TexturesDelta::default(),
			points: 1.0,
			console: console::Console::default(),
			tree: tree::Tree::default(),
		}
	}

	/// Registers the editor's own console variables.
	///
	/// Called by the host before the game module loads, so they belong to the
	/// engine and survive a reload.
	///
	/// @param world - the world to register into
	pub fn install(world: &mut World) {
		world
			.cvars
			.saved(SHOW, Value::Bool(true), "show the editor; F1 does the same thing");
	}

	/// Whether the editor is on screen.
	///
	/// @param world - where the variable lives
	#[must_use]
	pub fn shown(world: &World) -> bool { world.cvars.bool(SHOW).unwrap_or(false) }

	/// Shows the editor if it is hidden, and hides it if it is not.
	///
	/// @param world - where the variable lives
	pub fn toggle(world: &mut World) {
		let shown = Self::shown(world);

		world
			.cvars
			.set(SHOW, if shown { "false" } else { "true" });
	}

	/// Offers one window event to the editor.
	///
	/// @param window - the window the event came from
	/// @param event - the event
	/// @return whether the editor took it, in which case the game must not also
	/// act on it: a key typed into the console is not a key held to walk with
	pub fn on_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
		self.state.on_window_event(window, event).consumed
	}

	/// Builds this frame's interface.
	///
	/// Called after the simulation has run, so that what it shows is this
	/// frame's state rather than the previous one's. Nothing is drawn here -
	/// that happens in [`Overlay::draw`], when there is a frame to draw into.
	///
	/// @param window - the window, for input and for the cursor
	/// @param world - the state the panels show and edit
	/// @param clock - the pacing, for the statistics
	/// @param frames - how many frames have been drawn
	pub fn run(&mut self, window: &Window, world: &mut World, clock: &Clock, frames: u64) {
		let input = self.state.take_egui_input(window);
		let console = &mut self.console;
		let tree = &mut self.tree;

		// cloned before the run rather than reached through the `Ui` egui hands
		// the closure: a `Context` is a handle, cloning it is a refcount, and
		// every window here wants the context rather than a root layout.
		let context = self.context.clone();

		let output = self.context.run_ui(input, |_ui| {
			stats::show(&context, world, clock, frames);
			console.show(&context, world);
			tree.show(&context, world);
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
}

impl Drop for Editor {
	fn drop(&mut self) {
		// epaint asserts that a delta is applied rather than dropped, which is
		// the right rule while a frame is being built and the wrong one for a
		// process on its way out. Nothing is going to paint this.
		self.textures.clear();
	}
}

impl colby_engine::Overlay for Editor {
	fn draw(
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
			device.create_command_encoder(&CommandEncoderDescriptor { label: Some("editor") });

		self.painter
			.update_buffers(device, queue, &mut encoder, &self.jobs, &screen);

		// `Load`, not `Clear`: the scene is already in this frame, and the
		// point of an overlay is to be over something.
		let mut pass = encoder
			.begin_render_pass(&RenderPassDescriptor {
				label: Some("editor"),
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
