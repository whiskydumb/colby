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
//! **The launcher and the loading screen live here too**, @ref [`launcher`]
//! and [`loading`]: the screen every project is opened from and the screen a
//! project comes up behind are tools in the same sense a panel is, drawn with
//! the same egui through the same [`Shell`](shell::Shell), and they are what
//! a build with no editor in it has no use for either.
//!
//! @note: the game's own interface will not be this. egui is for tools; a game
//! draws its interface with HTML/CSS over taffy, which is a separate subsystem
//! that happens to arrive through the same [`Overlay`] seam.

use colby_core::{
	abi::{World, cvar::Value},
	time::Clock,
};
use colby_engine::Overlay;
use wgpu::{Device, Queue, TextureFormat, TextureView};
use winit::{event::WindowEvent, window::Window};

mod aim;
mod console;
mod gizmo;
pub mod launcher;
pub mod loading;
mod select;
mod shell;
mod stats;
mod tree;
mod viewport;

pub use self::{
	launcher::{Action, Launcher},
	loading::{Loading, State, Step},
};

/// The variable that decides whether the editor is on screen.
///
/// Saved, so that closing it stays closed. `editor.show 1` from the console
/// works as well as the key, because it is the same variable either way.
pub const SHOW: &str = "editor.show";

/// egui, and everything colby keeps on its behalf.
pub struct Editor {
	shell: shell::Shell,
	console: console::Console,
	/// What is selected, held here rather than in a panel: the viewport picks
	/// into it and the tree draws it, and two copies could disagree.
	selection: select::Selection,
	tree: tree::Tree,
	viewport: viewport::Viewport,
}

impl Editor {
	/// Brings up egui against the window and the device the scene draws with.
	///
	/// @param window - the window events come from
	/// @param device - the device the frames belong to
	/// @param format - the color format the surface was configured with
	#[must_use]
	pub fn new(window: &Window, device: &Device, format: TextureFormat) -> Self {
		Self {
			shell: shell::Shell::new(window, device, format),
			console: console::Console::default(),
			selection: select::Selection::default(),
			tree: tree::Tree::default(),
			viewport: viewport::Viewport::default(),
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
		self.shell.on_event(window, event)
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
		let Self {
			shell,
			console,
			selection,
			tree,
			viewport,
		} = self;

		shell.run(window, |ui| {
			// every window here wants the context rather than the root layout:
			// a `Context` is a handle, and cloning it is a refcount.
			let context = ui.ctx().clone();
			let context = &context;

			stats::show(context, world, clock, frames);
			console.show(context, world);

			// the world may have been replaced since the last frame by a scene
			// load or by play being stopped. Once, here, before anything reads
			// the selection.
			selection.refresh(world);

			// the viewport before the tree, so that a click out in the world
			// is already in hand when the tree draws the row it selected.
			if let Some(pick) = viewport.run(context, world, selection) {
				selection.set(world, pick);
			}

			tree.show(context, world, selection, viewport.tool());
		});
	}
}

impl Overlay for Editor {
	fn draw(
		&mut self,
		device: &Device,
		queue: &Queue,
		target: &TextureView,
		width: u32,
		height: u32,
	) {
		self.shell
			.draw(device, queue, target, width, height);
	}
}
