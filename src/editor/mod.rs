//! The editor: egui drawn over the scene, in the same frame.
//!
//! Engine-side, not gameplay-side - editing this is a restart, the same as
//! editing the renderer. It is its own crate so that egui stays out of
//! `colby_engine` (a renderer with an opinion about buttons is a renderer that
//! cannot be reused) and so that a shipping build drops the whole thing by
//! turning off one feature.
//!
//! **Four panels around a picture.** The hierarchy on the left, the inspector
//! on the right, the console and the statistics along the bottom, a strip of
//! controls along the top, and in the middle what is left: the world, drawn
//! into exactly that rectangle - @ref [`Frame`], which is how the window is
//! told where. Each panel is a view onto something that already existed
//! rather than a new system: the console shows the table and the log the
//! *previous* step built, the statistics show the clock and the world, the
//! hierarchy shows the tables a world is made of and the inspector one row
//! of them. That is the whole design brief for an editor here - if a panel
//! needs the engine to grow a new mechanism to feed it, the panel is wrong.
//!
//! **A press is a value.** A panel hands back what was pressed as a
//! [`Change`] and [`Panels::apply`] does it, so that the world is written in
//! one place and a panel can be drawn in a test without a window - the shape
//! the launcher's pages have.
//!
//! **Everything written is written down first**, @ref [`history`], and an
//! undo is a description of the world to put back, carried out of the frame
//! for the runner to restore: the solver has to forget what it derived at the
//! same moment, and the solver is the runner's.
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
	abi::{EntityId, World, cvar::Value, scene::SceneData},
	debug, info,
	time::Clock,
};
use colby_engine::{Overlay, Viewport};
use egui::{Context, Key, Modifiers, Panel, Rect, Ui};
use wgpu::{Device, Queue, TextureFormat, TextureView};
use winit::{event::WindowEvent, window::Window};

mod aim;
mod bar;
mod console;
mod gizmo;
mod hierarchy;
mod history;
mod inspector;
pub mod launcher;
pub mod loading;
mod select;
mod shell;
mod stats;
mod viewport;

use self::{bar::Steps, gizmo::Tool, history::History, select::Pick};
pub use self::{
	launcher::{Action, Launcher},
	loading::{Loading, State, Step},
};

/// The variable that decides whether the editor is on screen.
///
/// Saved, so that closing it stays closed. `editor.show 1` from the console
/// works as well as the key, because it is the same variable either way.
pub const SHOW: &str = "editor.show";

/// The variable that decides whether the world is being edited rather than
/// played.
///
/// The runner's, registered by it and acted on by it between two frames; the
/// editor writes it the way a typed line would, so that the button, the key
/// and the console reach the mode by one path.
const EDIT: &str = "sim.edit";

/// How wide the hierarchy starts out, in points.
const HIERARCHY_WIDTH: f32 = 240.0;

/// How wide the inspector starts out, in points.
const INSPECTOR_WIDTH: f32 = 320.0;

/// How tall the bottom panel starts out, in points.
const BOTTOM_HEIGHT: f32 = 220.0;

/// What one frame of the editor came to, for the window that holds it.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
	/// The part of the window left for the world, in physical pixels: what
	/// the panels did not take.
	pub view: Viewport,

	/// A world to put back, because a step back or forward was asked for.
	///
	/// The runner's to restore, between this frame and the next, the way it
	/// puts a world back when play stops - and to tell the solver to forget
	/// what it derived, which nothing in this crate can reach.
	pub restore: Option<Box<SceneData>>,
}

/// Something that was pressed on a panel.
///
/// A value rather than a call, so that a panel is a function from state to
/// intent and the world is written in one place, [`Panels::apply`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Change {
	/// Select something and nothing else, or nothing.
	Select(Pick),

	/// Add something to the selection, or take it out.
	Toggle(Pick),

	/// Remove everything selected.
	Delete,

	/// Make a copy of everything selected, and select the copies.
	Duplicate,

	/// Put the keyboard in the primary's name field.
	Rename,

	/// Hang an entity off another, or stand it on its own.
	Hang {
		/// What to hang.
		child: EntityId,

		/// What to hang it off, or [`EntityId::NONE`].
		parent: EntityId,
	},

	/// Switch the gizmo to one of its three things.
	Tool(Tool),

	/// Edit the world, or play it.
	Edit(bool),

	/// Write the world out as a scene source under this name.
	Write(String),

	/// Take a step back.
	Undo,

	/// Take a step forward again.
	Redo,
}

/// Which of the bottom panel's tabs is up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Tab {
	/// The console.
	#[default]
	Console,

	/// The statistics.
	Statistics,
}

impl Tab {
	/// What the tab is called.
	const fn name(self) -> &'static str {
		match self {
			| Self::Console => "console",
			| Self::Statistics => "statistics",
		}
	}
}

/// egui, and everything colby keeps on its behalf.
pub struct Editor {
	shell: shell::Shell,
	panels: Panels,
}

/// Everything about the editor that is not egui's plumbing: the panels, what
/// is selected, and where the picture was left.
///
/// Built without a window in a test, and drawn against a headless context.
pub(crate) struct Panels {
	bar: bar::Bar,
	console: console::Console,
	/// What is selected, held here rather than in a panel: the viewport picks
	/// into it and the hierarchy draws it, and two copies could disagree.
	selection: select::Selection,
	hierarchy: hierarchy::Hierarchy,
	viewport: viewport::Viewport,
	history: History,
	tab: Tab,
	/// Whether the name field takes the keyboard this frame: F2 was pressed
	/// last frame, and the inspector is what answers it.
	rename: bool,
	/// Whether the world was being edited last frame, so that play starting
	/// is an edge: the records are dropped on it, @ref [`history`].
	was_editing: bool,
	/// A world to put back, because a step was taken this frame.
	restore: Option<Box<SceneData>>,
	/// The part of the screen the world was drawn into last frame, in
	/// points.
	///
	/// What the viewport's gestures are measured against this frame, before
	/// the panels have been laid out again: the viewport runs first, so that
	/// a click in the world is in hand when the hierarchy draws the row it
	/// selected, and a panel's edge moves at most a few points between two
	/// frames.
	view: Rect,
}

impl Default for Panels {
	fn default() -> Self {
		Self {
			bar: bar::Bar::default(),
			console: console::Console::default(),
			selection: select::Selection::default(),
			hierarchy: hierarchy::Hierarchy::default(),
			viewport: viewport::Viewport::default(),
			history: History::default(),
			tab: Tab::default(),
			rename: false,
			was_editing: false,
			restore: None,
			// nowhere, until the first frame has laid the panels out: a
			// rectangle of no size at the corner, so that a pointer measured
			// from it on that one frame is measured from the window's corner
			// rather than from infinity.
			view: Rect::ZERO,
		}
	}
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
			panels: Panels::default(),
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
	/// @return what the frame came to, which is where the world goes
	pub fn run(
		&mut self,
		window: &Window,
		world: &mut World,
		clock: &Clock,
		frames: u64,
	) -> Frame {
		let Self { shell, panels } = self;
		let mut view = Rect::NOTHING;

		shell.run(window, |ui| {
			view = panels.frame(ui, world, clock, frames);
		});

		Frame {
			view: physical(view, shell.points()),
			restore: panels.restore.take(),
		}
	}
}

impl Panels {
	/// Lays the four panels out and drives the world between them.
	///
	/// @param ui - the whole window, egui's root layout for this frame
	/// @param world - the state the panels show and edit
	/// @param clock - the pacing, for the statistics
	/// @param frames - how many frames have been drawn
	/// @return the part of the window left for the world, in points
	pub(crate) fn frame(
		&mut self,
		ui: &mut Ui,
		world: &mut World,
		clock: &Clock,
		frames: u64,
	) -> Rect {
		// every panel wants the context rather than the root layout: a
		// `Context` is a handle, and cloning it is a refcount.
		let context = ui.ctx().clone();

		// the world may have been replaced since the last frame by a scene
		// load or by play being stopped. Once, here, before anything reads
		// the selection.
		self.selection.refresh(world);
		self.follow(world);

		// the viewport before the panels, against last frame's rectangle, so
		// that a click out in the world is already in hand when the hierarchy
		// draws the row it selected. A ctrl-click adds to what is selected
		// rather than replacing it, out here as in the hierarchy.
		if let Some(pick) =
			self.viewport
				.run(&context, world, &self.selection, self.view, &mut self.history)
		{
			if context.input(|input| input.modifiers.command) {
				self.selection.toggle(world, pick);
			} else {
				self.selection.set(world, pick);
			}
		}

		let mut changes = Vec::new();
		let tool = self.viewport.tool();
		let steps = Steps {
			undo: self.history.undoable(),
			redo: self.history.redoable(),
		};

		Panel::top("bar").show(ui, |ui| {
			self.bar
				.show(ui, world, tool, steps, &mut changes);
		});
		Panel::left("hierarchy")
			.default_size(HIERARCHY_WIDTH)
			.show(ui, |ui| {
				self.hierarchy
					.show(ui, world, &self.selection, &mut changes);
			});
		Panel::right("inspector")
			.default_size(INSPECTOR_WIDTH)
			.show(ui, |ui| {
				inspector::show(ui, world, &self.selection, &mut self.history, self.rename);
			});
		// answered, whether or not the field took it
		self.rename = false;
		Panel::bottom("bottom")
			.resizable(true)
			.default_size(BOTTOM_HEIGHT)
			.show(ui, |ui| self.bottom(ui, world, clock, frames));

		changes.extend(stepped(&context));

		for change in changes {
			self.apply(world, change);
		}

		// the frame is over for the history: a gesture nothing wrote to this
		// frame is a record now.
		if self.history.settle(world) {
			debug!(undo = self.history.undoable(), "written down");
		}

		// what is left is the world's. Nothing is laid out there on purpose:
		// egui takes the root layout's leftover as the part of the screen it
		// does not own, and that is what lets a drag out there be a camera's
		// rather than a widget's.
		self.view = ui.available_rect_before_wrap();

		self.view
	}

	/// Removes everything selected, and everything that could not stand
	/// without it, as one record.
	fn delete(&mut self, world: &mut World) {
		let picks = self.selection.picks();
		if picks.is_empty() {
			return;
		}

		self.history.begin("delete", world);
		let went = select::delete(world, &picks);
		self.selection.clear();

		info!(entities = went.entities, bodies = went.bodies, joints = went.joints, "deleted");
	}

	/// Copies everything selected, as one record, and selects the copies.
	fn duplicate(&mut self, world: &mut World) {
		let picks = self.selection.picks();
		if picks.is_empty() {
			return;
		}

		self.history.begin("duplicate", world);
		let copies = select::duplicate(world, &picks);
		self.selection.clear();

		// in the order the originals were picked, so that the copy of the
		// primary is the primary
		for copy in &copies {
			self.selection.toggle(world, *copy);
		}

		info!(copies = copies.len(), "duplicated");
	}

	/// Acts on play having started or stopped since the last frame.
	///
	/// The records are dropped when play starts: what they describe is a
	/// world the game is about to rewrite, and the world that comes back when
	/// play stops is the one play started from - which the records were made
	/// before, not after. @ref [`history`].
	fn follow(&mut self, world: &World) {
		if self.was_editing && !world.editing {
			self.history.clear();
			info!("playing; what was done while editing can no longer be undone");
		}

		self.was_editing = world.editing;
	}

	/// The bottom panel: two tabs, and whichever is up.
	fn bottom(&mut self, ui: &mut Ui, world: &mut World, clock: &Clock, frames: u64) {
		ui.add_space(4.0);
		ui.horizontal(|ui| {
			tab(ui, &mut self.tab, Tab::Console);
			tab(ui, &mut self.tab, Tab::Statistics);
		});
		ui.separator();

		match self.tab {
			| Tab::Console => self.console.show(ui, world),
			| Tab::Statistics => stats::show(ui, world, clock, frames),
		}
	}

	/// Does what was pressed.
	///
	/// @param world - the world to write
	/// @param change - what
	pub(crate) fn apply(&mut self, world: &mut World, change: Change) {
		match change {
			| Change::Select(pick) => self.selection.set(world, pick),
			| Change::Toggle(pick) => self.selection.toggle(world, pick),
			| Change::Delete => self.delete(world),
			| Change::Duplicate => self.duplicate(world),
			| Change::Rename => self.rename = true,
			| Change::Hang { child, parent } => {
				self.history.begin("hang", world);

				if !select::hang(world, child, parent) {
					// a stale handle, a loop, or a thing hung off itself: the
					// hierarchy refuses the last with no highlight, and the
					// other two are a race with the world. Worth a line, not
					// a stop.
					debug!(?child, ?parent, "nothing was hung");
				}
			},
			| Change::Tool(tool) => self.viewport.set_tool(tool),
			| Change::Edit(editing) => {
				world
					.cvars
					.set(EDIT, if editing { "true" } else { "false" });
			},
			| Change::Write(name) =>
				colby_core::abi::console::run(world, &format!("scene.write {name}")),
			| Change::Undo => self.restore = self.history.undo(world),
			| Change::Redo => self.restore = self.history.redo(),
		}
	}
}

/// The keys, if they were pressed where nothing else wanted them: a step
/// back and forward, delete, duplicate, rename.
///
/// Skipped while a text field is taking typing, so that a ctrl+z or a
/// delete in it stays the field's - and only then: a row or a button that
/// was clicked holds egui's focus too, and a key pressed after a click is
/// exactly the case.
fn stepped(context: &Context) -> Vec<Change> {
	if context.text_edit_focused() {
		return Vec::new();
	}

	context.input_mut(|input| {
		let mut changes = Vec::new();

		if input.consume_key(Modifiers::COMMAND, Key::Z) {
			changes.push(Change::Undo);
		}

		if input.consume_key(Modifiers::COMMAND, Key::Y)
			|| input.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
		{
			changes.push(Change::Redo);
		}

		if input.consume_key(Modifiers::NONE, Key::Delete) {
			changes.push(Change::Delete);
		}

		if input.consume_key(Modifiers::COMMAND, Key::D) {
			changes.push(Change::Duplicate);
		}

		if input.consume_key(Modifiers::NONE, Key::F2) {
			changes.push(Change::Rename);
		}

		changes
	})
}

/// One tab's label, which brings its tab up when pressed.
fn tab(ui: &mut Ui, current: &mut Tab, this: Tab) {
	if ui
		.selectable_label(*current == this, this.name())
		.clicked()
	{
		*current = this;
	}
}

/// A rectangle in points, as a viewport in physical pixels.
///
/// @param rect - the rectangle, from egui
/// @param points - how many physical pixels one point is
fn physical(rect: Rect, points: f32) -> Viewport {
	let points = if points.is_finite() && points > 0.0 {
		points
	} else {
		1.0
	};

	Viewport {
		x: pixels(rect.min.x * points),
		y: pixels(rect.min.y * points),
		width: pixels(rect.width() * points),
		height: pixels(rect.height() * points),
	}
}

/// A length in pixels as a count of them: rounded, and never less than none.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "clamped to the range first, and a screen is a few thousand pixels across"
)]
fn pixels(value: f32) -> u32 {
	if value.is_finite() {
		value.round().clamp(0.0, 1.0e6) as u32
	} else {
		0
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

#[cfg(test)]
mod tests {
	use colby_core::{abi::Transform, glam::Vec3};
	use egui::{Pos2, RawInput, vec2};

	use super::*;

	/// One headless frame of the whole editor over a world, with these
	/// events in it.
	///
	/// Built once, whatever egui asks: a context may run the closure a
	/// second time in the same call when something asked for another pass,
	/// and a frame built twice would answer a key twice - the same guard
	/// `Shell::run` keeps.
	fn frame_with(panels: &mut Panels, world: &mut World, events: Vec<egui::Event>) -> Rect {
		let context = Context::default();
		let clock = Clock::new();
		let mut view = Rect::NOTHING;
		let mut built = false;

		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 720.0))),
				events,
				..Default::default()
			},
			|ui| {
				if !built {
					built = true;
					view = panels.frame(ui, world, &clock, 1);
				}
			},
		);
		// nothing paints this frame, and epaint asserts that a texture delta
		// is applied rather than dropped; cleared on purpose, the way the
		// shell does on its way out.
		output.textures_delta.clear();

		view
	}

	/// One headless frame with nothing pressed.
	fn frame(panels: &mut Panels, world: &mut World) -> Rect {
		frame_with(panels, world, Vec::new())
	}

	/// One headless frame with one key pressed in it.
	fn keyed(panels: &mut Panels, world: &mut World, key: Key, modifiers: Modifiers) {
		frame_with(panels, world, vec![egui::Event::Key {
			key,
			physical_key: None,
			pressed: true,
			repeat: false,
			modifiers,
		}]);
	}

	#[test]
	fn the_panels_leave_the_middle_of_the_window_for_the_world() {
		let mut world = World::new();
		world.editing = true;
		let mut panels = Panels::default();

		let view = frame(&mut panels, &mut world);

		assert!(view.min.x >= HIERARCHY_WIDTH, "the hierarchy is on the left: {view:?}");
		assert!(view.max.x <= 1280.0 - INSPECTOR_WIDTH, "the inspector on the right: {view:?}");
		assert!(view.min.y > 0.0, "the strip along the top: {view:?}");
		assert!(view.max.y <= 720.0 - BOTTOM_HEIGHT, "the bottom panel below: {view:?}");
		assert!(view.width() > 400.0 && view.height() > 200.0, "and room for a world: {view:?}");
		assert_eq!(panels.view, view, "and the viewport is told where for next frame");
	}

	#[test]
	fn a_rectangle_in_points_is_a_viewport_in_pixels() {
		let rect = Rect::from_min_size(Pos2::new(240.0, 36.5), vec2(720.0, 463.5));

		assert_eq!(physical(rect, 1.0), Viewport { x: 240, y: 37, width: 720, height: 464 });
		assert_eq!(physical(rect, 2.0), Viewport { x: 480, y: 73, width: 1440, height: 927 });
		assert_eq!(
			physical(rect, 0.0),
			physical(rect, 1.0),
			"a scale of nothing is a scale of one rather than a division by it"
		);
		assert_eq!(
			physical(Rect::NOTHING, 1.0),
			Viewport { x: 0, y: 0, width: 0, height: 0 },
			"and a rectangle of nothing is no pixels, not a panic"
		);
	}

	#[test]
	fn a_press_is_applied_to_the_world_in_one_place() {
		let mut world = World::new();
		let car = world.entities.spawn_at(Transform::at(Vec3::X));
		let wheel = world
			.entities
			.spawn_at(Transform::at(Vec3::new(3.0, 0.0, 0.0)));
		let mut panels = Panels::default();

		panels.apply(&mut world, Change::Select(Pick::Entity(wheel)));
		assert!(panels.selection.is(Pick::Entity(wheel)));

		panels.apply(&mut world, Change::Hang { child: wheel, parent: car });
		assert_eq!(world.entities.parent(wheel), car, "the wheel hangs off the car");
		assert_eq!(
			world.entities.placed(wheel).map(|it| it.position),
			Some(Vec3::new(3.0, 0.0, 0.0)),
			"and stayed where it was in the world"
		);

		panels.apply(&mut world, Change::Tool(Tool::Turn));
		assert_eq!(panels.viewport.tool(), Tool::Turn);

		world.cvars.var(EDIT, Value::Bool(false), "");
		panels.apply(&mut world, Change::Edit(true));
		assert_eq!(world.cvars.bool(EDIT), Some(true), "play and stop go through the variable");
	}

	#[test]
	fn a_hang_is_undone_by_the_world_the_frame_hands_back() {
		let mut world = World::new();
		world.editing = true;
		let car = world.entities.spawn_at(Transform::at(Vec3::X));
		world.entities.set_name(car, "car");
		let wheel = world
			.entities
			.spawn_at(Transform::at(Vec3::new(3.0, 0.0, 0.0)));
		world.entities.set_name(wheel, "wheel");
		let mut panels = Panels::default();

		panels.apply(&mut world, Change::Hang { child: wheel, parent: car });
		// two quiet frames: the one the hang was written in, and the one
		// that closes the record
		assert!(!panels.history.settle(&world));
		assert!(panels.history.settle(&world), "the hang is a record");
		assert_eq!(panels.history.undoable(), Some("hang"));

		panels.apply(&mut world, Change::Undo);
		let described = panels
			.restore
			.take()
			.expect("a world to put back");
		colby_core::abi::scene::restore(&mut world, &described).expect("the world takes it");

		assert!(!world.entities.parent(wheel).is_some(), "the wheel stands on its own again");
		assert_eq!(
			world.entities.placed(wheel).map(|it| it.position),
			Some(Vec3::new(3.0, 0.0, 0.0)),
			"where it was"
		);
		assert!(world.entities.alive(wheel), "and its handle still resolves");
		assert_eq!(panels.history.redoable(), Some("hang"), "and the hang can be done again");
	}

	#[test]
	fn ctrl_z_in_a_frame_hands_back_the_world_before_the_last_record() {
		let mut world = World::new();
		world.editing = true;
		let car = world.entities.spawn_at(Transform::at(Vec3::X));
		let wheel = world
			.entities
			.spawn_at(Transform::at(Vec3::new(3.0, 0.0, 0.0)));
		let mut panels = Panels::default();
		panels.apply(&mut world, Change::Hang { child: wheel, parent: car });

		// two frames with nothing pressed close the record, then one with the
		// key down
		frame(&mut panels, &mut world);
		frame(&mut panels, &mut world);
		assert_eq!(panels.history.undoable(), Some("hang"));

		keyed(&mut panels, &mut world, Key::Z, Modifiers::COMMAND);

		let described = panels
			.restore
			.take()
			.expect("the key took a step back");
		assert_eq!(
			described.things.len(),
			2,
			"the world before the hang, with both things in it"
		);
		assert!(
			described
				.things
				.iter()
				.all(|thing| thing.parent == colby_core::abi::scene::NO_INDEX),
			"and neither hanging off the other"
		);
	}

	#[test]
	fn the_delete_key_removes_what_is_selected_as_one_record() {
		let mut world = World::new();
		world.editing = true;
		let car = world.entities.spawn_at(Transform::at(Vec3::X));
		let wheel = world.entities.spawn_at(Transform::at(Vec3::Z));
		assert!(world.entities.set_parent(wheel, car));
		let bystander = world.entities.spawn_at(Transform::IDENTITY);
		let mut panels = Panels::default();
		panels.apply(&mut world, Change::Select(Pick::Entity(car)));

		keyed(&mut panels, &mut world, Key::Delete, Modifiers::NONE);
		frame(&mut panels, &mut world);

		assert!(!world.entities.alive(car) && !world.entities.alive(wheel), "the branch went");
		assert!(world.entities.alive(bystander), "and nothing else");
		assert_eq!(panels.selection.at(), Pick::Nothing, "nothing is selected now");
		assert_eq!(panels.history.undoable(), Some("delete"), "and it is one step back");
	}

	#[test]
	fn ctrl_d_copies_what_is_selected_and_selects_the_copies() {
		let mut world = World::new();
		world.editing = true;
		let car = world.entities.spawn_at(Transform::at(Vec3::X));
		world.entities.set_name(car, "car");
		let ball = world.entities.spawn_at(Transform::at(Vec3::Z));
		let mut panels = Panels::default();
		panels.apply(&mut world, Change::Select(Pick::Entity(ball)));
		panels.apply(&mut world, Change::Toggle(Pick::Entity(car)));

		keyed(&mut panels, &mut world, Key::D, Modifiers::COMMAND);
		frame(&mut panels, &mut world);

		assert_eq!(world.entities.len(), 4, "two copies beside the two");
		assert_eq!(panels.selection.len(), 2, "the copies are selected");
		let Pick::Entity(primary) = panels.selection.at() else {
			panic!("the copy of the car is the primary");
		};
		assert_ne!(primary, car);
		assert_eq!(world.entities.name(primary), "car", "the copy of what was picked last");
		assert_eq!(panels.history.undoable(), Some("duplicate"));
	}

	#[test]
	fn f2_puts_the_keyboard_in_the_name_field_for_one_frame() {
		let mut world = World::new();
		world.editing = true;
		let car = world.entities.spawn_at(Transform::at(Vec3::X));
		let mut panels = Panels::default();
		panels.apply(&mut world, Change::Select(Pick::Entity(car)));

		keyed(&mut panels, &mut world, Key::F2, Modifiers::NONE);
		assert!(panels.rename, "asked for");

		frame(&mut panels, &mut world);
		assert!(!panels.rename, "and answered by the next frame");
	}

	#[test]
	fn play_starting_drops_what_was_done_while_editing() {
		let mut world = World::new();
		world.editing = true;
		let car = world.entities.spawn_at(Transform::at(Vec3::X));
		let wheel = world.entities.spawn_at(Transform::at(Vec3::Z));
		let mut panels = Panels::default();
		panels.apply(&mut world, Change::Hang { child: wheel, parent: car });
		frame(&mut panels, &mut world);
		frame(&mut panels, &mut world);
		assert_eq!(panels.history.undoable(), Some("hang"));

		world.editing = false;
		frame(&mut panels, &mut world);

		assert_eq!(panels.history.undoable(), None, "the game owns the world now");
	}
}
