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

use std::time::Duration;

use colby_asset::{Project, compile::Kind};
use colby_core::{
	abi::{EntityId, World, cvar::Value, scene::SceneData},
	debug,
	glam::{Vec2, Vec3},
	info,
	time::Clock,
	warn,
};
use colby_engine::{Gpu, Overlay, Viewport, cull::Drawn};
use egui::{Context, DragAndDrop, Key, Modifiers, Panel, Rect, Ui};
use wgpu::{Device, Queue, TextureFormat, TextureView};
use winit::{event::WindowEvent, window::Window};

mod aim;
mod bake;
mod bar;
mod browser;
mod catalog;
mod console;
mod gizmo;
mod helper;
mod hierarchy;
mod history;
mod inspector;
pub mod launcher;
pub mod loading;
mod paint;
mod profiler;
mod select;
mod session;
mod settings;
mod shell;
mod stats;
mod tabs;
mod thumbs;
mod viewport;

use self::{bar::Steps, browser::Dropped, gizmo::Tool, select::Pick, tabs::Tabs};
pub use self::{
	launcher::{Action, Launcher},
	loading::{Loading, State, Step},
};

/// The variable that decides whether the editor is on screen.
///
/// Saved, so that closing it stays closed. `editor.show 1` from the console
/// works as well as the key, because it is the same variable either way.
pub const SHOW: &str = "editor.show";

/// How far apart the grid's lines are, in world units.
///
/// **The step every placement and every drag lands on.** Nought turns snapping
/// off, which is what the editor did before this existed and is still what
/// somebody moving a lamp by eye wants.
///
/// A variable rather than a constant because the right number is a property of
/// what is being built - Unreal keeps a whole power-of-two hierarchy of them
/// with hotkeys to climb it (`CubeGridTool.h:111`, `GridPower` 0 to 31) and
/// Godot's editor keeps one in the project. One number and a console line is
/// the smallest thing that is honestly a grid.
pub const GRID: &str = "editor.grid";

/// The variable that decides whether the things with nothing to look at are
/// drawn and can be clicked.
///
/// **On, and every editor read for this has the same switch** - a menu of
/// kinds, a key, a show flag. What it is for is the same thing in all of them:
/// a world full of lamps is a world full of glyphs, and somebody framing a
/// picture wants the picture. Off takes the marks, the outlines, the reaches
/// and the handles away *and* takes them out of a click, because a thing that
/// answers a click while being invisible is worse than one that does neither.
pub const HELPERS: &str = "editor.helpers";

/// The variable that decides whether the world is being edited rather than
/// played.
///
/// The runner's, registered by it and acted on by it between two frames; the
/// editor writes it the way a typed line would, so that the button, the key
/// and the console reach the mode by one path.
const EDIT: &str = "sim.edit";

/// What a tab is called before anybody has written it anywhere.
///
/// A window can come up on a world the project named, and it can come up on
/// one nothing named at all - a fresh project, a game that built its own. The
/// name is what `scene.write` writes to, so a world nobody named is offered
/// this one and whoever writes it may change it.
const UNNAMED: &str = "edited";

/// The variable that decides whether a stop keeps what the game did.
///
/// The runner's as well: the editor never writes it and only says which of
/// the two things a stop is about to do, because a promise about that is the
/// one thing a panel can get wrong without anybody noticing until the work is
/// gone.
const KEEP: &str = "sim.keep";

/// What the grid is set to when nobody has said otherwise.
///
/// Half a unit, which is half the built-in cube: two blocks side by side meet
/// exactly, and a block half a step off the grid is visibly off it rather than
/// arguably off it.
const GRID_STEP: f32 = 0.5;

/// What the window lends the editor for one frame: the pacing, the
/// project and the device.
///
/// The clock and the count are for the statistics; the project is where the
/// asset browser walks and keeps its pictures; the device is what a mesh's
/// picture is drawn with, and a window with no usable one draws none.
#[derive(Clone, Copy)]
pub struct Host<'a> {
	/// The pacing.
	pub clock: &'a Clock,

	/// How many frames have been drawn.
	pub frames: u64,

	/// The project, if the window has one.
	pub project: Option<&'a Project>,

	/// The device, if there is one.
	pub gpu: Option<&'a Gpu>,

	/// What a frame is costing, when anything is measuring it.
	///
	/// Filled by the runner, because the two halves of it live where this
	/// crate cannot reach: the hardware spans are on the scene's own
	/// apparatus and the simulation's are on the solver. What arrives here is
	/// already averaged - @ref `Part`.
	pub profile: Profile<'a>,

	/// How much of the world the last frame drew.
	///
	/// Filled by the runner off the window's own scene, a frame behind the one
	/// the panel is drawn in, which for a count of what a camera leaves out is
	/// the same answer. @ref [`Drawn`].
	pub drawn: Drawn,
}

/// What a frame is costing, as the panel shows it.
///
/// Owned by this crate rather than by the runner, because [`Host`] is this
/// crate's and a panel cannot name a type from the crate that holds it. The
/// runner fills it; nothing here knows where a number came from.
#[derive(Clone, Copy, Debug, Default)]
pub struct Profile<'a> {
	/// One per part of the frame, in the order a frame happens in.
	pub parts: &'a [Part],

	/// How many render passes the last measured frame recorded.
	///
	/// The one number here that does not move between two runs of the same
	/// scene, which is what makes it the part worth comparing.
	pub passes: Option<u32>,

	/// Whether the hardware side answered at all.
	///
	/// False on an adapter with no timestamp queries, where the six hardware
	/// rows are empty and the wall clock is the whole answer.
	pub hardware: bool,
}

/// One part of a frame, averaged over the frames it was measured in.
#[derive(Clone, Copy, Debug)]
pub struct Part {
	/// What it is called: `gpu scene`, `cpu solve`.
	pub name: &'static str,

	/// The mean, or nothing for a part that never ran.
	pub mean: Option<Duration>,

	/// The worst single one of them.
	///
	/// Beside the mean because they answer different questions, which is the
	/// reason `--profile` prints both: a part that is cheap on average and
	/// occasionally enormous is invisible in the first and obvious in the
	/// second.
	pub worst: Option<Duration>,
}

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

	/// Whether anything on screen wants a frame measured.
	///
	/// The profiler panel's whole gate, and it is a gate rather than a switch
	/// because measuring costs a query set, two buffers and a readback a
	/// frame: a window that showed the panel once should not pay for it for
	/// the rest of the run. The runner starts and stops the apparatus on this.
	pub measuring: bool,
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

	/// Put the entities selected into a new group, and select it.
	Group,

	/// Take the groups selected apart, and select what was in them.
	Ungroup,

	/// Select everything hanging off an entity, and not the entity.
	///
	/// What marks a whole wall in one gesture: the wall's group is the row the
	/// menu opened on, and a field changed afterwards is written to every
	/// brick.
	Inside(EntityId),

	/// Hide an entity and everything hanging off it, or show it again.
	Hide {
		/// Which.
		entity: EntityId,

		/// Whether it is to be hidden.
		hidden: bool,
	},

	/// Switch the gizmo to one of its three things.
	Tool(Tool),

	/// Put the brush out, or away.
	///
	/// Its own press rather than a fourth [`Tool`]: while the brush is out
	/// there are no handles at all, and a gizmo with no handles is not one of
	/// the three things the gizmo does. @ref `viewport::Viewport::stroke`.
	Brush(bool),

	/// Make the brush this wide and this hard.
	Stroke {
		/// How wide, in the ground's own units.
		radius: f32,

		/// How hard, a share of a whole cell a dab.
		strength: f32,
	},

	/// Edit the world, or play it.
	Edit(bool),

	/// Write the world out as a scene source under this name.
	Write(String),

	/// Take a step back.
	Undo,

	/// Take a step forward again.
	Redo,

	/// Move to a scene, opening it if it is not open.
	Open {
		/// The scene source's asset name.
		name: String,
	},

	/// Move to the tab in this place in the row.
	Show {
		/// Which tab.
		which: usize,
	},

	/// Close the tab in this place in the row.
	Shut {
		/// Which tab.
		which: usize,
	},

	/// Show an asset in the inspector, by the name it registers under.
	///
	/// A name rather than a handle, because the browser that asks is a view of
	/// a *directory* and holds no world: what it knows about a row is what the
	/// file is called. @ref `Panels::inspect`.
	Inspect {
		/// The asset name, `materials/brass`.
		name: String,
	},

	/// Rename an asset's source, and every sidecar standing beside it.
	///
	/// A name and a name, for the reason [`Inspect`](Self::Inspect) carries
	/// one: the browser knows what a row is called and the runner is the half
	/// that owns the project. @ref `colby_runtime`'s `rename` module for why a
	/// rename is a command rather than a file-manager gesture, and why there
	/// is no undo of one.
	RenameAsset {
		/// The asset name as it stands, `meshes/crystal`.
		name: String,

		/// What the last part of it is to become, `gem`.
		to: String,
	},

	/// Open an asset's source in whatever editor this person uses.
	///
	/// A name rather than a path, for the reason
	/// [`Inspect`](Self::Inspect) carries one: the browser knows what a row is
	/// called, and the runner is the half that owns the project and can turn a
	/// name into a file. @ref `colby_runtime`'s `code` module.
	Code {
		/// The asset name, `scripts/thruster`.
		name: String,
	},

	/// Put a body of water in the middle of what is being looked at.
	///
	/// No asset behind it and so not a [`Drop`](Self::Drop): a fluid is
	/// something the engine can make on its own, which is why it is a press
	/// rather than a row in the browser.
	Water,

	/// Put a block in the middle of what is being looked at, on the grid.
	///
	/// The same kind of press as [`Water`](Self::Water) and for the same
	/// reason: a cube is something the engine can make on its own.
	Block,

	/// Put a decal in the middle of what is being looked at, facing down.
	///
	/// The third press of its kind, for its reason: a decal is a box and a
	/// material, and the engine can make both on its own.
	Decal,

	/// Turn the selected blocks into one mesh per material.
	///
	/// The name is the bar's write field, which already names what is being
	/// built - a second name for the same room would be a second thing to keep
	/// in step.
	Bake(String),

	/// Work the world's still light out, and write the picture and the scene.
	///
	/// The name is the bar's write field, as the blocks' bake takes it: the
	/// picture and the scene it belongs to are called what the room is.
	Light(String),

	/// Put an asset into the world, where it was dropped.
	Drop {
		/// The asset name.
		name: String,

		/// What it is.
		kind: Kind,

		/// Where it lands.
		at: Vec3,
	},
}

/// Which of the bottom panel's panes is up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Pane {
	/// The console.
	#[default]
	Console,

	/// The statistics.
	Statistics,

	/// The asset browser.
	Assets,

	/// Every console variable, as something to turn.
	Settings,

	/// What a frame costs, one row per part of it.
	///
	/// **The last of the five, and the only one whose being up costs
	/// anything**: measuring a frame builds a query set and two buffers and
	/// reads one back. That is why it is a pane rather than a section of the
	/// statistics, which somebody may leave open all day.
	Profiler,
}

impl Pane {
	/// What the tab is called.
	const fn name(self) -> &'static str {
		match self {
			| Self::Console => "console",
			| Self::Statistics => "statistics",
			| Self::Assets => "assets",
			| Self::Settings => "settings",
			| Self::Profiler => "profiler",
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
	browser: browser::Browser,
	console: console::Console,
	/// What is selected, held here rather than in a panel: the viewport picks
	/// into it and the hierarchy draws it, and two copies could disagree.
	selection: select::Selection,
	hierarchy: hierarchy::Hierarchy,
	viewport: viewport::Viewport,
	settings: settings::Settings,
	/// Every scene open, and the history of each. The one on screen is the
	/// live world; @ref [`tabs`].
	tabs: Tabs,
	pane: Pane,
	/// Whether the name field takes the keyboard this frame: F2 was pressed
	/// last frame, and the inspector is what answers it.
	rename: bool,
	/// Whether the world was being edited last frame, so that play starting
	/// is an edge: the records are dropped on it, @ref [`history`].
	was_editing: bool,
	/// A world to put back, because a step was taken this frame.
	restore: Option<Box<SceneData>>,
	/// How wide the three panels are, as read off the last frame and written
	/// to `.colby/editor.json` when the window stops.
	sizes: session::Sizes,

	/// Scenes the last run had open, waiting for the registry to hold them.
	///
	/// A window comes up before its assets do, so a name read out of the file
	/// cannot be opened on the first frame. Each is tried once a frame until
	/// the compiler has produced it or the list gives up on it; @ref
	/// `catching_up`.
	waiting: Vec<String>,

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
			browser: browser::Browser::default(),
			console: console::Console::default(),
			selection: select::Selection::default(),
			hierarchy: hierarchy::Hierarchy::default(),
			viewport: viewport::Viewport::default(),
			settings: settings::Settings::default(),
			tabs: Tabs::new(UNNAMED),
			pane: Pane::default(),
			rename: false,
			was_editing: false,
			sizes: session::Sizes::default(),
			waiting: Vec::new(),
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

	/// Puts back what the editor was left with for this project.
	///
	/// Called once, before the first frame: the panels take the widths they
	/// were dragged to and the tabs are the scenes that were open, minus the
	/// one the window came up on, which is already the world. The scenes are
	/// only named here - each is opened by name on the frame that reaches it,
	/// because a scene the compiler no longer produces is a name and nothing
	/// more.
	///
	/// @param project - whose file
	/// @param scene - the scene the world came up on, which is tab one
	pub fn remember(&mut self, project: &Project, scene: &str) {
		let session = session::Session::open(project);

		self.panels.sizes = session.panels;
		self.panels.tabs = Tabs::new(if scene.is_empty() { UNNAMED } else { scene });
		self.panels.waiting = session
			.tabs
			.iter()
			.filter(|name| *name != scene)
			.cloned()
			.collect();
	}

	/// Writes down what to put back next time.
	///
	/// Called as the window goes down, beside the console's own config. A
	/// failure is a line and not a stop: this is a convenience, and a project
	/// whose derived tree cannot be written has worse problems than a panel
	/// width.
	///
	/// @param project - whose file
	pub fn forget_not(&self, project: &Project) {
		let session = session::Session {
			tabs: self
				.panels
				.tabs
				.names()
				.map(str::to_owned)
				.collect(),
			current: self.panels.tabs.at(),
			panels: self.panels.sizes,
		};

		if let Err(failure) = session.save(project) {
			warn!(%failure, "the editor's file was not written");
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
		world.cvars.saved(
			GRID,
			Value::Float(GRID_STEP),
			"how far apart the editor's grid is, in world units; 0 turns snapping off",
		);
		world.cvars.saved(
			HELPERS,
			Value::Bool(true),
			"draw and pick the things that have nothing to look at: lamps, throwers, decals, \
			 ties and bodies nobody draws",
		);
	}

	/// The grid step this world is being edited on, or nothing for no snapping.
	///
	/// @param world - where the variable lives
	#[must_use]
	pub fn grid(world: &World) -> Option<f32> {
		world.cvars.float(GRID).filter(|step| *step > 0.0)
	}

	/// Whether the editor is on screen.
	///
	/// @param world - where the variable lives
	#[must_use]
	pub fn shown(world: &World) -> bool { world.cvars.bool(SHOW).unwrap_or(false) }

	/// Whether the things with nothing to look at are drawn and clickable.
	///
	/// @param world - where the variable lives
	#[must_use]
	pub fn helpers(world: &World) -> bool { world.cvars.bool(HELPERS).unwrap_or(true) }

	/// Shows the editor if it is hidden, and hides it if it is not.
	///
	/// @param world - where the variable lives
	pub fn toggle(world: &mut World) {
		let shown = Self::shown(world);

		world
			.cvars
			.set(SHOW, if shown { "false" } else { "true" });
	}

	/// Starts play if the world is being edited, and stops it if it is not.
	///
	/// **The one path.** The key in the window and the button on the bar both
	/// come here, so the world before play is written down once and by the
	/// same hand - and so that it is written down whether the editor is on
	/// screen or hidden, which a key can be pressed either way.
	///
	/// @param world - the world, whose variable is written and whose state is
	/// what an undo of the play goes back to
	pub fn play(&mut self, world: &mut World) {
		let editing = world.editing;

		self.panels.set_mode(world, !editing);
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
	/// @param host - what the window lends for the frame
	/// @return what the frame came to, which is where the world goes
	pub fn run(&mut self, window: &Window, world: &mut World, host: &Host<'_>) -> Frame {
		let Self { shell, panels } = self;
		let mut view = Rect::NOTHING;

		shell.run(window, |ui| {
			view = panels.frame(ui, world, host);
		});

		Frame {
			view: physical(view, shell.points()),
			restore: panels.restore.take(),
			// the pane being up is the whole of the gate, and it is read here
			// rather than kept as a flag: a pane that was closed this frame
			// stops being measured for the next one, with nothing to keep in
			// step.
			measuring: panels.measuring(),
		}
	}
}

impl Panels {
	/// Lays the four panels out and drives the world between them.
	///
	/// @param ui - the whole window, egui's root layout for this frame
	/// @param world - the state the panels show and edit
	/// @param host - what the window lends for the frame
	/// @return the part of the window left for the world, in points
	pub(crate) fn frame(&mut self, ui: &mut Ui, world: &mut World, host: &Host<'_>) -> Rect {
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
				.run(&context, world, &self.selection, self.view, self.tabs.history())
		{
			// the outermost group around what is under the pointer, and exactly
			// what is under it with alt held
			let pick = if context.input(|input| input.modifiers.alt) {
				pick
			} else {
				select::grouped(world, pick)
			};

			if context.input(|input| input.modifiers.command) {
				self.selection.toggle(world, pick);
			} else {
				self.selection.set(world, pick);
			}
		}

		// the window as it stands, taken before a panel has eaten into it:
		// what a panel was dragged to is the difference between this and what
		// is left over at the end of the frame
		let whole = ui.max_rect();
		let mut changes = Vec::new();
		let tool = (
			self.viewport.tool(),
			self.viewport
				.brushing()
				.then(|| self.viewport.stroke_of()),
		);
		let scene = self.tabs.current().name.clone();
		let history = self.tabs.history();
		let steps = Steps {
			undo: history.undoable(),
			redo: history.redoable(),
		};

		Panel::top("bar").show(ui, |ui| {
			self.bar
				.show(ui, world, tool, steps, &scene, &mut changes);
		});
		Panel::top("tabs").show(ui, |ui| {
			tabs::strip(ui, &self.tabs, &mut changes);
		});
		Panel::left("hierarchy")
			.default_size(points(self.sizes.left))
			.show(ui, |ui| {
				self.hierarchy
					.show(ui, world, &self.selection, &mut changes);
			});
		Panel::right("inspector")
			.default_size(points(self.sizes.right))
			.show(ui, |ui| {
				inspector::show(
					ui,
					world,
					&self.selection,
					self.tabs.history(),
					self.rename,
					host.project,
				);
			});
		// answered, whether or not the field took it
		self.rename = false;
		Panel::bottom("bottom")
			.resizable(true)
			.default_size(points(self.sizes.bottom))
			.show(ui, |ui| self.bottom(ui, world, host, &mut changes));

		changes.extend(self.catching_up(world));
		changes.extend(stepped(&context));
		changes.extend(self.dropped(&context, world));

		for change in changes {
			self.apply(world, change);
		}

		// the frame is over for the history: a gesture nothing wrote to this
		// frame is a record now.
		if self.tabs.history().settle(world) {
			debug!(undo = self.tabs.history().undoable(), "written down");
		}

		// what is left is the world's. Nothing is laid out there on purpose:
		// egui takes the root layout's leftover as the part of the screen it
		// does not own, and that is what lets a drag out there be a camera's
		// rather than a widget's.
		self.view = ui.available_rect_before_wrap();
		// the three widths as they now stand, read off the leftover rather
		// than asked of egui: whatever the panels were dragged to is the
		// difference between the window and what they left for the picture,
		// and that is one subtraction rather than three lookups into a
		// memory whose keys this crate would have to keep in step
		let screen = whole;
		self.sizes = session::Sizes {
			left: pixels(self.view.min.x - screen.min.x),
			right: pixels(screen.max.x - self.view.max.x),
			bottom: pixels(screen.max.y - self.view.max.y),
		};

		self.view
	}

	/// Removes everything selected, and everything that could not stand
	/// without it, as one record.
	fn delete(&mut self, world: &mut World) {
		let picks = self.selection.picks();
		if picks.is_empty() {
			return;
		}

		self.tabs.history().begin("delete", world);
		let went = select::delete(world, &picks);
		self.selection.clear();

		info!(entities = went.entities, bodies = went.bodies, joints = went.joints, "deleted");
	}

	/// Puts the entities selected into a new group, as one record, and selects
	/// the group.
	fn group(&mut self, world: &mut World) {
		let picks = self.selection.picks();
		if !picks
			.iter()
			.any(|pick| matches!(pick, Pick::Entity(_)))
		{
			return;
		}

		self.tabs.history().begin("group", world);
		let made = select::group(world, &picks, Editor::grid(world));

		if made.is_empty() {
			warn!("there was no room in the world for a group");

			return;
		}

		self.selection.clear();
		for pick in &made {
			self.selection.toggle(world, *pick);
		}

		info!(picked = picks.len(), "grouped");
	}

	/// Takes the groups selected apart, as one record, and selects what was in
	/// them.
	fn ungroup(&mut self, world: &mut World) {
		let picks = self.selection.picks();
		if !picks.iter().any(|pick| match *pick {
			| Pick::Entity(id) => select::is_group(world, id),
			| Pick::Nothing
			| Pick::Body(_)
			| Pick::Joint(_)
			| Pick::Material(_)
			| Pick::Model(_) => false,
		}) {
			return;
		}

		self.tabs.history().begin("ungroup", world);
		let released = select::ungroup(world, &picks);

		self.selection.clear();
		for pick in &released {
			self.selection.toggle(world, *pick);
		}

		info!(released = released.len(), "ungrouped");
	}

	/// Copies everything selected, as one record, and selects the copies.
	fn duplicate(&mut self, world: &mut World) {
		let picks = self.selection.picks();
		if picks.is_empty() {
			return;
		}

		self.tabs.history().begin("duplicate", world);
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
	/// The record that spans the play was opened by @ref `set_mode`, before
	/// the mode moved; this is the other end of it. A stop that put the world
	/// back closes it with the world it opened with and writes nothing down,
	/// and a stop under `sim.keep` closes it as one step. Play that began
	/// somewhere else - a typed line, a config file - held nothing, and there
	/// is nothing here to close. @ref [`history`].
	fn follow(&mut self, world: &World) {
		if self.was_editing && !world.editing && !self.tabs.history().holding() {
			// play was started by something that is not the editor - a typed
			// `sim.edit 0`, a config file, a game that asked for it - so
			// nobody wrote the world down and there is nothing to close. The
			// records made before it stand, and what an undo of one does is
			// what it always does: put a whole world back.
			debug!("playing; nobody wrote this play down, so there is no step back over it");
		}

		if !self.was_editing && world.editing && self.tabs.history().release(world) {
			info!("editing; the play is one step to go back over, because sim.keep is on");
		}

		self.was_editing = world.editing;
	}

	/// The bottom panel: four panes, and whichever is up.
	fn bottom(
		&mut self,
		ui: &mut Ui,
		world: &mut World,
		host: &Host<'_>,
		changes: &mut Vec<Change>,
	) {
		ui.add_space(4.0);
		ui.horizontal(|ui| {
			pane(ui, &mut self.pane, Pane::Console);
			pane(ui, &mut self.pane, Pane::Statistics);
			pane(ui, &mut self.pane, Pane::Assets);
			pane(ui, &mut self.pane, Pane::Settings);
			pane(ui, &mut self.pane, Pane::Profiler);
		});
		ui.separator();

		match self.pane {
			| Pane::Console => self.console.show(ui, world),
			| Pane::Statistics => stats::show(ui, world, host.clock, host.frames, host.drawn),
			| Pane::Assets => self
				.browser
				.show(ui, host.project, host.gpu, changes),
			| Pane::Settings => self.settings.show(ui, world),
			| Pane::Profiler => profiler::show(ui, host.profile),
		}
	}

	/// Whether anything on screen wants a frame measured.
	///
	/// One pane and nothing else, which is what makes the gate cheap to reason
	/// about: the profiler is up or it is not, and there is no flag anywhere
	/// that can disagree with what is on screen.
	fn measuring(&self) -> bool { self.pane == Pane::Profiler }

	/// An asset let go of over the picture, if one was this frame.
	///
	/// The picture is not a widget, so nothing there can be asked whether a
	/// payload was released on it; the release is read off the pointer and
	/// the payload taken by hand, and where it lands is where the ray from
	/// the pointer meets the floor. @ref [`aim::floor`].
	fn dropped(&self, context: &Context, world: &World) -> Vec<Change> {
		if !context.input(|input| input.pointer.any_released()) {
			return Vec::new();
		}

		let Some(pointer) = context.pointer_latest_pos() else {
			return Vec::new();
		};

		if !self.view.contains(pointer) {
			return Vec::new();
		}

		let Some(dropped) = DragAndDrop::take_payload::<Dropped>(context) else {
			return Vec::new();
		};

		let local = Vec2::new(pointer.x - self.view.min.x, pointer.y - self.view.min.y);
		let size = Vec2::new(self.view.width().max(1.0), self.view.height().max(1.0));
		let at = aim::floor(&world.render_camera(), local, size);

		vec![Change::Drop {
			name: dropped.name.clone(),
			kind: dropped.kind,
			at,
		}]
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
				self.tabs.history().begin("hang", world);

				if !select::hang(world, child, parent) {
					// a stale handle, a loop, or a thing hung off itself: the
					// hierarchy refuses the last with no highlight, and the
					// other two are a race with the world. Worth a line, not
					// a stop.
					debug!(?child, ?parent, "nothing was hung");
				}
			},
			| Change::Group => self.group(world),
			| Change::Ungroup => self.ungroup(world),
			| Change::Inside(entity) => {
				self.selection.clear();

				for id in select::descendants(world, entity) {
					self.selection.toggle(world, Pick::Entity(id));
				}
			},
			| Change::Hide { entity, hidden } => {
				// a gesture like a hang and undone like one: the word is in the
				// world a record captures, so this is all a hide takes
				self.tabs.history().begin("hide", world);

				if !world.entities.set_hidden(entity, hidden) {
					debug!(?entity, "nothing was hidden");
				}
			},
			| Change::Tool(tool) => self.viewport.set_tool(tool),
			| Change::Brush(out) => self.viewport.set_brushing(out),
			| Change::Stroke { radius, strength } => self.viewport.set_stroke(radius, strength),
			| Change::Edit(editing) => self.set_mode(world, editing),
			| Change::Write(name) => {
				// the scene the tab is, from now on: a write under a new name
				// is a save-as, and what is on screen afterwards is what was
				// just written rather than what it came from
				colby_core::abi::console::run(world, &format!("scene.write {name}"));
				self.tabs.rename(&name);
			},
			| Change::Undo => self.restore = self.tabs.history().undo(world),
			| Change::Redo => self.restore = self.tabs.history().redo(),
			| Change::Open { name } => self.reach(world, &name),
			| Change::Show { which } => self.restore = self.tabs.switch(world, which),
			| Change::Shut { which } => self.restore = self.tabs.close(which),
			| Change::Inspect { name } => self.inspect(world, &name),
			| Change::RenameAsset { name, to } => self.rename_asset(world, &name, &to),
			| Change::Code { name } => {
				// straight to the console, as a write is: the runner is what
				// holds the project and the two variables, and the editor's
				// half of this is knowing which row was pressed.
				colby_core::abi::console::run(world, &format!("code.open {name}"));
			},
			| Change::Water => self.water(world),
			| Change::Block => self.block(world),
			| Change::Decal => self.decal(world),
			| Change::Bake(name) => self.bake(world, &name),
			| Change::Light(name) => self.light(world, &name),
			| Change::Drop { name, kind, at } => self.drop(world, &name, kind, at),
		}
	}

	/// Asks for a mode, and writes the play down when one is starting.
	///
	/// The variable is written rather than the state, so a typed `sim.edit 1`
	/// and this are the same thing one frame later; what this adds is the
	/// record. A play is one step to go back over, opened here and closed
	/// when the world comes back to being edited, @ref `follow`. A stop that
	/// puts the world back closes it with the world it opened with and
	/// nothing is written down; a stop under `sim.keep` closes it with what
	/// the game left, and one ctrl+z undoes the whole play.
	///
	/// @param world - the world to write the variable into
	/// @param editing - the mode being asked for
	fn set_mode(&mut self, world: &mut World, editing: bool) {
		if !editing {
			self.tabs.history().hold("play", world);
		}

		world
			.cvars
			.set(EDIT, if editing { "true" } else { "false" });
	}

	/// One scene from the last run, if the compiler has caught up with it.
	///
	/// A window comes up before its assets do, so a name out of the file
	/// cannot be opened on the first frame; one is tried a frame, in the
	/// order they were written down, and a name the registry still does not
	/// know stays in the list. A name it *never* knows stays there for the
	/// life of the window and costs one lookup a frame, which is a linear
	/// walk of tens of entries and not worth a second mechanism to avoid.
	///
	/// The scene opened this way does not become the one on screen: they are
	/// opened in order and the tab that was current is chosen at the end, so
	/// opening them one at a time would otherwise walk the window through
	/// every scene the last run had.
	fn catching_up(&mut self, world: &World) -> Vec<Change> {
		let Some(index) = self
			.waiting
			.iter()
			.position(|name| world.scenes.find(name).is_some())
		else {
			return Vec::new();
		};

		let name = self.waiting.remove(index);

		vec![Change::Open { name }]
	}

	/// Asks the runner to rename an asset, and follows it in the tabs.
	///
	/// Straight to the console, as a write and an open are: the runner is what
	/// holds the project, and the editor's half of this is knowing which row
	/// was pressed and what was typed into it.
	///
	/// **The tab moves with the file.** Renaming a scene that is open would
	/// otherwise leave a tab claiming to be a source that is not there, and the
	/// next write under that name would make a second one. Every other kind
	/// needs nothing: the asset loop hands the table the identity and the table
	/// moves the entry, so every handle this world is holding goes on
	/// resolving. @ref `colby_core::abi::registry::Registry::adopt`.
	///
	/// **There is no record of it.** The history is a stack of worlds, and a
	/// rename happens outside every world there is; the undo of a rename is a
	/// rename back.
	///
	/// @param world - where the line waits for the runner
	/// @param name - the asset name as it stands
	/// @param to - what the last part of it is to become
	fn rename_asset(&mut self, world: &mut World, name: &str, to: &str) {
		colby_core::abi::console::run(world, &format!("asset.rename {name} {to}"));

		let Some((held, _)) = name.rsplit_once('/') else {
			self.tabs.rename_named(name, to);

			return;
		};

		self.tabs
			.rename_named(name, &format!("{held}/{to}"));
	}

	/// Opens a scene by name, or moves to it if it is already open.
	///
	/// The compiled scene is the tab's world, taken out of the registry the
	/// asset loop fills. A name the registry does not know is a scene that
	/// has not compiled - the browser says so on the row - and nothing
	/// happens beyond a line saying which name it was.
	///
	/// @param world - the world being left, written down by the tabs
	/// @param name - the scene source's asset name
	fn reach(&mut self, world: &World, name: &str) {
		let id = world.scenes.find(name);
		if !id.is_some() {
			warn!(name, "no compiled scene under that name to open");

			return;
		}

		let data = world.scenes.data(id).clone();
		self.restore = self.tabs.open(world, name, &data);
		self.selection.clear();

		info!(name, tabs = self.tabs.len(), "opened");
	}

	/// Selects an asset so the inspector shows it.
	///
	/// A material or a model, which are the two the inspector has a panel for
	/// and the two members of [`Pick`] that are not in the world. A name
	/// nothing answers to selects nothing, which is what a browser row for a
	/// file the engine has not loaded yet would ask for.
	///
	/// **The material registry is asked first**, and the two cannot collide
	/// anyway: a material's name is either its own file's or is inside a
	/// model's, `models/lamp/brass`, and a model's is the model's.
	///
	/// @param world - the registries to look the name up in, read only: what a
	/// selection points at is not a change to the world
	/// @param name - the asset name
	fn inspect(&mut self, world: &World, name: &str) {
		let coat = world.materials.find(name);

		if coat.is_some() {
			self.selection.set(world, Pick::Material(coat));

			return;
		}

		let model = world.models.find(name);

		if model.is_some() {
			self.selection.set(world, Pick::Model(model));

			return;
		}

		debug!(name, "nothing in the world answers to that asset");
	}

	/// Puts a block down where the middle of the picture meets the floor.
	///
	/// The same arithmetic `water` does, and for its reason: a block put at the
	/// origin while somebody is looking somewhere else is a block they have to
	/// go and find.
	fn block(&mut self, world: &mut World) {
		let at = self.looked_at(world);

		self.tabs.history().begin("block", world);

		let made = select::block(world, at, Editor::grid(world));

		if made.is_empty() {
			warn!("there was no room in the world for a block");

			return;
		}

		self.selection.clear();
		for pick in &made {
			self.selection.toggle(world, *pick);
		}
	}

	/// Turns the selected blocks into a mesh, and writes it out.
	///
	/// **Two halves, and the seam is the mesh registry.** The world work is
	/// this crate's - @ref [`bake`] - and the files are the runner's, asked for
	/// with a console line carrying the bake's name. Nothing about a mesh
	/// crosses that line; both sides already share the registry it was left in.
	fn bake(&mut self, world: &mut World, name: &str) {
		// **the last part of the write name, not the whole of it.** The field
		// holds a *scene* - `scenes/room` - and a bake's name is a directory
		// under `maps/`, which may not have a slash in it. Taking the last part
		// is what a person means: the room described by `scenes/room` bakes
		// into `maps/room/`. Found by pressing the button, which wrote a mesh
		// into the world and then refused to write the file.
		let name = name
			.trim()
			.rsplit('/')
			.next()
			.unwrap_or_default()
			.to_owned();

		if name.is_empty() {
			warn!("name the room in the write field first; that is what the files are called");

			return;
		}

		self.tabs.history().begin("bake", world);

		let baked = bake::bake(world, &self.selection.picks(), &name);

		if baked.made.is_empty() {
			warn!(blocks = baked.blocks, "there was nothing to bake");

			return;
		}

		info!(
			name,
			blocks = baked.blocks,
			meshes = baked.made.len(),
			triangles = baked.triangles,
			buried = baked.buried,
			"blocks baked"
		);

		self.selection.clear();
		for pick in &baked.made {
			self.selection.toggle(world, *pick);
		}

		colby_core::abi::console::run(world, &format!("blocks.write {name}"));
	}

	/// Asks for the world's still light, the way the console does.
	///
	/// A step to go back over, because a bake writes every thing's place on the
	/// picture into its record and names the picture on the world; the files it
	/// writes stay, as the blocks' bake's do. The line is served before the
	/// next frame's panels, so the gesture opened here is still open when the
	/// bake lands in the world and closes on the frame after with the bake in
	/// it. The tab is the scene of that name afterwards, as a write makes it.
	///
	/// @param world - the world to bake
	/// @param name - the write field; its last part is the name, as the
	/// blocks' bake takes it
	fn light(&mut self, world: &mut World, name: &str) {
		let name = name
			.trim()
			.rsplit('/')
			.next()
			.unwrap_or_default()
			.to_owned();

		if name.is_empty() {
			warn!("name the room in the write field first; that is what the picture is called");

			return;
		}

		self.tabs.history().begin("bake light", world);
		colby_core::abi::console::run(world, &format!("light.bake {name}"));
		self.tabs.rename(&name);
	}

	fn water(&mut self, world: &mut World) {
		let at = self.looked_at(world);

		self.tabs.history().begin("water", world);
		let made = select::water(world, at);

		if made.is_empty() {
			warn!("there was no room in the world for a pool");

			return;
		}

		self.selection.clear();
		for pick in &made {
			self.selection.toggle(world, *pick);
		}

		info!(?at, "a pool was put in the world");
	}

	/// Puts a decal where the middle of the picture meets the ground, turned to
	/// throw straight down at it, and selects it.
	fn decal(&mut self, world: &mut World) {
		let at = self.looked_at(world);

		self.tabs.history().begin("decal", world);
		let made = select::decal(world, at);

		if made.is_empty() {
			warn!("there was no room in the world for a decal");

			return;
		}

		self.selection.clear();
		for pick in &made {
			self.selection.toggle(world, *pick);
		}

		info!(?at, "a decal was put in the world");
	}

	/// Where the middle of the picture meets the ground, which is where a thing
	/// made with no pointer lands: a pool put down at the origin while somebody
	/// is looking somewhere else is a pool they have to go and find. @ref
	/// `aim::floor`.
	fn looked_at(&self, world: &World) -> Vec3 {
		let middle = Vec2::new(self.view.width() * 0.5, self.view.height() * 0.5);
		let size = Vec2::new(self.view.width().max(1.0), self.view.height().max(1.0));

		aim::floor(&world.render_camera(), middle, size)
	}

	/// Puts an asset into the world where it was dropped, as one record, and
	/// selects what it became.
	fn drop(&mut self, world: &mut World, name: &str, kind: Kind, at: Vec3) {
		self.tabs.history().begin("drop", world);
		let landed = select::drop(world, name, kind, at);

		if landed.is_empty() {
			// a texture, a sound, or a scene the loop has not compiled yet:
			// the record closes empty, because nothing changed
			warn!(name, kind = catalog::word(kind), "nothing to put in the world for it");

			return;
		}

		self.selection.clear();
		for pick in &landed {
			self.selection.toggle(world, *pick);
		}

		info!(name, ?at, made = landed.len(), "dropped into the world");
	}
}

/// The keys, if they were pressed where nothing else wanted them: a step
/// back and forward, delete, duplicate, group and ungroup, rename.
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

		// the chord with a shift in it first: egui matches a key ignoring a shift
		// the pattern does not name, so an undo asked for first takes the redo
		// chord for itself
		if input.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
			|| input.consume_key(Modifiers::COMMAND, Key::Y)
		{
			changes.push(Change::Redo);
		}

		if input.consume_key(Modifiers::COMMAND, Key::Z) {
			changes.push(Change::Undo);
		}

		if input.consume_key(Modifiers::NONE, Key::Delete) {
			changes.push(Change::Delete);
		}

		if input.consume_key(Modifiers::COMMAND, Key::D) {
			changes.push(Change::Duplicate);
		}

		// the shift first again, for the reason the redo chord is
		if input.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::G) {
			changes.push(Change::Ungroup);
		}

		if input.consume_key(Modifiers::COMMAND, Key::G) {
			changes.push(Change::Group);
		}

		if input.consume_key(Modifiers::NONE, Key::F2) {
			changes.push(Change::Rename);
		}

		changes
	})
}

/// One pane's label, which brings its pane up when pressed.
fn pane(ui: &mut Ui, current: &mut Pane, this: Pane) {
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

/// A width the file holds, as the number of points a panel is given.
fn points(size: u32) -> f32 { f32::from(u16::try_from(size).unwrap_or(u16::MAX)) }

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
	use colby_core::abi::Transform;
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
		frame_holding(panels, world, events, Modifiers::NONE)
	}

	/// The same, with modifier keys held down through it.
	fn frame_holding(
		panels: &mut Panels,
		world: &mut World,
		events: Vec<egui::Event>,
		modifiers: Modifiers,
	) -> Rect {
		frame_on(&Context::default(), panels, world, events, modifiers).0
	}

	/// The same over a context that lasts, for what egui answers from the frame
	/// before: whether the pointer is over a panel or over what the panels
	/// left, which a fresh context cannot say and answers as a panel.
	fn frame_on(
		context: &Context,
		panels: &mut Panels,
		world: &mut World,
		events: Vec<egui::Event>,
		modifiers: Modifiers,
	) -> (Rect, Vec<egui::epaint::ClippedShape>) {
		let clock = Clock::new();
		let host = Host {
			clock: &clock,
			frames: 1,
			project: None,
			gpu: None,
			profile: Profile::default(),
			drawn: Drawn::default(),
		};
		let mut view = Rect::NOTHING;
		let mut built = false;
		// the keys held are an event of their own, ahead of what they are held
		// through
		let events = std::iter::once(egui::Event::ModifiersChanged(modifiers))
			.chain(events)
			.collect();

		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 720.0))),
				events,
				..Default::default()
			},
			|ui| {
				if !built {
					built = true;
					view = panels.frame(ui, world, &host);
				}
			},
		);
		// nothing paints this frame, and epaint asserts that a texture delta
		// is applied rather than dropped; cleared on purpose, the way the
		// shell does on its way out.
		output.textures_delta.clear();

		(view, output.shapes)
	}

	#[test]
	fn the_profiler_pane_being_up_is_the_whole_of_the_gate() {
		// what the runner starts and stops the apparatus on. A flag kept
		// beside the pane could disagree with what is on screen; this cannot,
		// because it is the pane.
		let mut panels = Panels::default();

		assert!(!panels.measuring(), "the console is up, and nothing is being measured");

		for pane in [Pane::Statistics, Pane::Assets, Pane::Settings] {
			panels.pane = pane;

			assert!(!panels.measuring(), "{} costs nothing", pane.name());
		}

		panels.pane = Pane::Profiler;

		assert!(panels.measuring(), "and the one pane that does say so");
	}

	#[test]
	fn a_frame_with_the_profiler_up_draws_it_and_asks_to_be_measured() {
		let mut panels = Panels::default();
		let mut world = World::new();

		panels.pane = Pane::Profiler;

		// the pane draws over a profile nothing has filled in, which is what
		// the first frame after it is opened always looks like
		let view = frame(&mut panels, &mut world);

		assert!(view.width() > 0.0, "the picture still has room beside the panels");

		assert!(panels.measuring(), "and it is still asking after a frame of it");
	}

	#[test]
	fn every_pane_has_a_name_of_its_own() {
		let named =
			[Pane::Console, Pane::Statistics, Pane::Assets, Pane::Settings, Pane::Profiler];
		let mut names: Vec<&str> = named.iter().map(|pane| pane.name()).collect();

		names.sort_unstable();
		names.dedup();

		assert_eq!(names.len(), named.len(), "two panes share a name");
	}

	#[test]
	fn a_bake_is_named_by_the_last_part_of_the_write_field() {
		// the write field holds a scene - `scenes/room` - and a bake's name is
		// a directory under `maps/`, which may not have a slash in it. Found
		// by pressing the button: the world baked and the file was refused.
		let mut panels = Panels::default();
		let mut world = World::new();

		drop(select::block(&mut world, Vec3::ZERO, Some(1.0)));
		panels.apply(&mut world, Change::Bake("scenes/room".to_owned()));

		assert!(
			world.meshes.find("maps/room/default").is_some(),
			"the mesh is under the room, not under a directory called scenes"
		);
		assert!(
			!world
				.meshes
				.find("maps/scenes/room/default")
				.is_some(),
			"and not under the whole of the field"
		);
	}

	#[test]
	fn a_bake_with_no_name_bakes_nothing() {
		let mut panels = Panels::default();
		let mut world = World::new();

		drop(select::block(&mut world, Vec3::ZERO, Some(1.0)));

		let before = world.entities.len();

		panels.apply(&mut world, Change::Bake("   ".to_owned()));

		assert_eq!(
			world.entities.len(),
			before,
			"the blocks are still blocks, because a bake with no name has nowhere to be written"
		);
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
	fn asking_for_water_puts_a_pool_in_front_of_the_camera_and_selects_it() {
		let mut world = World::new();
		world.editing = true;
		let mut panels = Panels::default();

		// one frame first, so the viewport has a size for the middle of it to
		// be worked out from: a pool asked for before anything was drawn
		// would land wherever a camera with no picture points.
		frame(&mut panels, &mut world);
		panels.apply(&mut world, Change::Water);

		let (id, body) = world
			.bodies
			.iter()
			.find(|(_, body)| body.water.is_wet())
			.expect("a pool was made");

		assert!(!body.solid(), "which nothing is pushed out of");
		assert!(body.entity.is_some(), "and it drives something to look at");
		assert_eq!(world.bodies.len(), 1, "one body and no more");
		assert!(
			panels.selection.is(Pick::Entity(body.entity)),
			"and what was made is what is selected"
		);

		// and it is a step, so it can be taken back
		panels.apply(&mut world, Change::Undo);
		let put_back = panels
			.restore
			.take()
			.expect("an undo hands a world back");
		colby_core::abi::scene::restore(&mut world, &put_back).expect("the layouts agree");

		assert!(
			world.bodies.get(id).is_none(),
			"a pool put down by mistake is one step back, like everything else"
		);
	}

	#[test]
	fn the_console_prompt_is_inside_the_window_and_not_under_its_floor() {
		let mut world = World::new();
		world.editing = true;
		let mut panels = Panels::default();
		let context = Context::default();
		let clock = Clock::new();
		let host = Host {
			clock: &clock,
			frames: 1,
			project: None,
			gpu: None,
			profile: Profile::default(),
			drawn: Drawn::default(),
		};
		let mut built = false;
		let mut view = Rect::NOTHING;
		// a scrollback with more in it than the panel can hold, which is the
		// state a window is in within a second of starting and the one a
		// small fixture cannot show
		for line in 0..300_u32 {
			info!(line, "a line in the log");
		}

		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 720.0))),
				..Default::default()
			},
			|ui| {
				if !built {
					built = true;
					view = panels.frame(ui, &mut world, &host);
				}
			},
		);
		output.textures_delta.clear();

		// the console is the tab that is up to begin with, so its prompt is
		// drawn; a bottom panel grows to fit what is in it, and anything
		// added above the prompt comes out of the scrollback's share rather
		// than pushing the prompt off the bottom of the window
		let prompt = context
			.read_response(egui::Id::new("console prompt"))
			.expect("the console was drawn")
			.rect;

		assert!(prompt.max.y <= 720.0, "the prompt is under the window's floor: {prompt:?}");
		assert!(view.max.y <= prompt.min.y, "and the picture stops above it: {view:?}");
	}

	#[test]
	fn opening_a_scene_that_never_compiled_leaves_the_tabs_where_they_are() {
		let mut world = World::new();
		world.editing = true;
		let mut panels = Panels::default();

		panels.apply(&mut world, Change::Open { name: "scenes/never".to_owned() });

		assert_eq!(panels.tabs.len(), 1, "no tab for a name the registry does not know");
		assert!(panels.restore.is_none(), "and no world to put back");
	}

	#[test]
	fn a_scene_the_last_run_had_open_waits_for_the_compiler_to_catch_up() {
		let mut world = World::new();
		world.editing = true;
		let mut panels = Panels {
			waiting: vec!["scenes/hangar".to_owned()],
			..Panels::default()
		};

		assert!(panels.catching_up(&world).is_empty(), "nothing to open yet");
		assert_eq!(panels.waiting.len(), 1, "and the name is still waiting");

		// the asset loop catches up
		let mut other = World::new();
		other
			.entities
			.spawn_at(Transform::at(Vec3::Y * 3.0));
		world
			.scenes
			.insert("scenes/hangar", colby_core::abi::scene::capture(&other));

		let asked = panels.catching_up(&world);

		assert_eq!(asked.len(), 1, "now it can be opened");
		assert!(panels.waiting.is_empty(), "and it is off the list");
	}

	#[test]
	fn asking_for_a_source_leaves_a_console_line_for_the_runner() {
		// the seam between the two halves: the editor knows which row was
		// pressed and the runner owns the project, so what crosses is a line.
		// A world with no such command registered - which is what a `World`
		// out of `new` is - drops it, so the name is registered here the way
		// the runner registers it.
		let mut world = World::new();
		world
			.cvars
			.command("code.open", colby_core::abi::console::defer, "");
		let mut panels = Panels::default();

		panels.apply(&mut world, Change::Code { name: "scripts/thruster".to_owned() });

		assert_eq!(world.asked.len(), 1, "one line waiting for the frame loop");
		assert_eq!(world.asked[0].name, "code.open");
		assert_eq!(world.asked[0].words, vec!["scripts/thruster".to_owned()]);
	}

	#[test]
	fn renaming_an_asset_leaves_a_console_line_and_moves_the_tab_that_was_open_on_it() {
		// the same seam a source opening uses, and one thing besides: a tab
		// open on a scene has to follow the file, or its next write makes a
		// second one under the name that is gone.
		let mut world = World::new();
		world
			.cvars
			.command("asset.rename", colby_core::abi::console::defer, "");
		let mut panels = Panels::default();
		panels.tabs.rename("scenes/yard");

		panels.apply(&mut world, Change::RenameAsset {
			name: "scenes/yard".to_owned(),
			to: "court".to_owned(),
		});

		assert_eq!(world.asked.len(), 1, "one line waiting for the frame loop");
		assert_eq!(world.asked[0].name, "asset.rename");
		assert_eq!(world.asked[0].words, vec!["scenes/yard".to_owned(), "court".to_owned()]);
		assert_eq!(
			panels.tabs.names().collect::<Vec<&str>>(),
			vec!["scenes/court"],
			"and the tab is called what the file is called"
		);
	}

	#[test]
	fn baking_the_light_leaves_a_line_named_by_the_last_part_of_the_write_field() {
		// the runner owns the project and the files, so what crosses is a line,
		// as with a write; the name is the room's, as with the blocks' bake
		let mut world = World::new();
		world.editing = true;
		world
			.cvars
			.command("light.bake", colby_core::abi::console::defer, "");
		let mut panels = Panels::default();

		panels.apply(&mut world, Change::Light("scenes/yard".to_owned()));

		assert_eq!(world.asked.len(), 1, "one line waiting for the frame loop");
		assert_eq!(world.asked[0].name, "light.bake");
		assert_eq!(world.asked[0].words, vec!["yard".to_owned()], "the room, not the directory");
		assert_eq!(panels.tabs.current().name, "yard", "and the tab is the scene it writes");

		world.asked.clear();
		panels.apply(&mut world, Change::Light("   ".to_owned()));

		assert!(world.asked.is_empty(), "a bake with no name asks for nothing");
	}

	#[test]
	fn a_write_renames_the_tab_it_was_written_from() {
		let mut world = World::new();
		world.editing = true;
		let mut panels = Panels::default();

		panels.apply(&mut world, Change::Write("scenes/yard".to_owned()));

		assert_eq!(panels.tabs.current().name, "scenes/yard", "a write is a save-as");
	}

	/// A world with one thing hung off another and a ball above the floor,
	/// with that hang already a record.
	fn hung() -> (World, Panels, EntityId) {
		let mut world = World::new();
		world.editing = true;
		let car = world.entities.spawn_at(Transform::at(Vec3::X));
		let wheel = world.entities.spawn_at(Transform::at(Vec3::Z));
		let ball = world.entities.spawn_at(Transform::at(Vec3::Y));
		let mut panels = Panels::default();
		panels.apply(&mut world, Change::Hang { child: wheel, parent: car });
		frame(&mut panels, &mut world);
		frame(&mut panels, &mut world);

		(world, panels, ball)
	}

	/// Moves a thing, the way a game does every step it runs.
	fn shift(world: &mut World, id: EntityId, to: Vec3) {
		if let Some(transform) = world.entities.transform_mut(id) {
			transform.position = to;
		}
	}

	#[test]
	fn the_panels_leave_the_middle_of_the_window_for_the_world() {
		let mut world = World::new();
		world.editing = true;
		let mut panels = Panels::default();

		let view = frame(&mut panels, &mut world);

		let starting = session::Sizes::default();
		assert!(view.min.x >= points(starting.left), "the hierarchy is on the left: {view:?}");
		assert!(
			view.max.x <= 1280.0 - points(starting.right),
			"the inspector on the right: {view:?}"
		);
		assert!(view.min.y > 0.0, "the strip along the top: {view:?}");
		assert!(
			view.max.y <= 720.0 - points(starting.bottom),
			"the bottom panel below: {view:?}"
		);
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

		panels.apply(&mut world, Change::Brush(true));
		assert!(panels.viewport.brushing(), "the brush is out");
		panels.apply(&mut world, Change::Tool(Tool::Move));
		assert!(!panels.viewport.brushing(), "and asking for the gizmo puts it away");

		panels.apply(&mut world, Change::Stroke { radius: 9.0, strength: 0.5 });
		assert_eq!(panels.viewport.stroke_of(), (9.0, 0.5), "the brush takes both numbers");
		panels.apply(&mut world, Change::Stroke { radius: 1.0e9, strength: 4.0 });
		assert_eq!(
			panels.viewport.stroke_of(),
			(paint::RANGE.1, 1.0),
			"and holds each inside what it may be"
		);

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
		assert!(!panels.tabs.history().settle(&world));
		assert!(panels.tabs.history().settle(&world), "the hang is a record");
		assert_eq!(panels.tabs.history().undoable(), Some("hang"));

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
		assert_eq!(
			panels.tabs.history().redoable(),
			Some("hang"),
			"and the hang can be done again"
		);
	}

	#[test]
	fn a_hide_from_the_tree_is_one_step_and_is_undone_like_any_other() {
		let mut world = World::new();
		world.editing = true;
		let car = world.entities.spawn_at(Transform::at(Vec3::X));
		let wheel = world.entities.spawn_at(Transform::at(Vec3::Z));
		assert!(world.entities.set_parent(wheel, car));
		let mut panels = Panels::default();

		panels.apply(&mut world, Change::Hide { entity: car, hidden: true });

		assert!(!world.entities.shown(wheel), "the car is hidden, and the wheel under it");
		// two quiet frames: the one the hide was written in, and the one that
		// closes the record
		assert!(!panels.tabs.history().settle(&world));
		assert!(panels.tabs.history().settle(&world), "the hide is a record");
		assert_eq!(panels.tabs.history().undoable(), Some("hide"));

		panels.apply(&mut world, Change::Undo);
		let described = panels
			.restore
			.take()
			.expect("a world to put back");
		colby_core::abi::scene::restore(&mut world, &described).expect("the world takes it");

		assert!(!world.entities.hidden(car), "one step back the car is drawn again");
		assert!(world.entities.shown(wheel), "and the wheel with it");
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
		assert_eq!(panels.tabs.history().undoable(), Some("hang"));

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
	fn ctrl_shift_z_and_ctrl_y_take_a_step_forward_again() {
		// egui matches a key ignoring a shift the pattern does not name, so the
		// chord with a shift in it has to be asked for before the one without, or
		// it is taken for an undo. A probe of the build before this test found
		// ctrl+shift+z handing back nothing at all
		let mut world = World::new();
		world.editing = true;
		let car = world.entities.spawn_at(Transform::at(Vec3::X));
		let wheel = world
			.entities
			.spawn_at(Transform::at(Vec3::new(3.0, 0.0, 0.0)));
		let mut panels = Panels::default();
		panels.apply(&mut world, Change::Hang { child: wheel, parent: car });
		frame(&mut panels, &mut world);
		frame(&mut panels, &mut world);

		for (key, chord) in
			[(Key::Z, Modifiers::COMMAND | Modifiers::SHIFT), (Key::Y, Modifiers::COMMAND)]
		{
			keyed(&mut panels, &mut world, Key::Z, Modifiers::COMMAND);
			let back = panels
				.restore
				.take()
				.expect("ctrl+z took a step back");
			colby_core::abi::scene::restore(&mut world, &back).expect("the world takes it");
			frame(&mut panels, &mut world);

			keyed(&mut panels, &mut world, key, chord);
			let forward = panels
				.restore
				.take()
				.expect("the chord took a step forward");

			assert!(
				forward
					.things
					.iter()
					.any(|thing| thing.parent != colby_core::abi::scene::NO_INDEX),
				"the world after the hang, the wheel hanging off the car"
			);

			colby_core::abi::scene::restore(&mut world, &forward).expect("the world takes it");
			frame(&mut panels, &mut world);
		}
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
		assert_eq!(panels.tabs.history().undoable(), Some("delete"), "and it is one step back");
	}

	#[test]
	fn ctrl_g_groups_what_is_selected_and_ctrl_shift_g_takes_it_apart_each_one_step() {
		let mut world = World::new();
		world.editing = true;
		let crate_ = world.entities.spawn_at(Transform::at(Vec3::X));
		let barrel = world
			.entities
			.spawn_at(Transform::at(Vec3::new(3.0, 0.0, 0.0)));
		let mut panels = Panels::default();
		panels.apply(&mut world, Change::Select(Pick::Entity(crate_)));
		panels.apply(&mut world, Change::Toggle(Pick::Entity(barrel)));

		keyed(&mut panels, &mut world, Key::G, Modifiers::COMMAND);
		frame(&mut panels, &mut world);

		let Pick::Entity(group) = panels.selection.at() else {
			panic!("the group is what is selected");
		};
		assert_eq!(panels.selection.len(), 1, "and nothing else");
		assert!(select::is_group(&world, group), "and it is one");
		assert_eq!(world.entities.parent(crate_), group, "with the crate in it");
		assert_eq!(world.entities.parent(barrel), group, "and the barrel");
		assert_eq!(panels.tabs.history().undoable(), Some("group"), "one step back");

		keyed(&mut panels, &mut world, Key::G, Modifiers::COMMAND | Modifiers::SHIFT);
		frame(&mut panels, &mut world);

		assert!(!world.entities.alive(group), "taken apart, the empty group went");
		assert_eq!(
			world.entities.parent(crate_),
			EntityId::NONE,
			"and the crate stands on its own"
		);
		assert!(
			panels.selection.is(Pick::Entity(crate_))
				&& panels.selection.is(Pick::Entity(barrel)),
			"with what was inside selected"
		);
		assert_eq!(panels.tabs.history().undoable(), Some("ungroup"), "one step back as well");
	}

	#[test]
	fn a_click_in_the_picture_selects_the_outermost_group_and_with_alt_what_is_under_it() {
		let mut world = World::new();
		world.editing = true;
		world.camera.position = Vec3::new(0.0, 0.0, 10.0);
		world.camera.target = Vec3::ZERO;
		let group = world.entities.spawn();
		let crate_ = world.entities.spawn_at(Transform::IDENTITY);
		world.entities.set_renderable(
			crate_,
			colby_core::abi::Renderable::new(colby_core::abi::MeshId::CUBE, Vec3::ONE),
		);
		assert!(world.entities.set_parent(crate_, group));

		if let Some(editing) = world
			.entities
			.record_mut(&colby_core::abi::EDITING, group)
		{
			editing.group = 1;
		}

		let mut panels = Panels::default();
		let context = Context::default();
		let (view, _) = frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE);
		let size = Vec2::new(view.width(), view.height());
		let at = gizmo::project(
			world
				.render_camera()
				.view_projection(size.x / size.y),
			Vec3::ZERO,
			size,
		)
		.expect("the crate is in front of the camera");
		let pointer = Pos2::new(view.min.x + at.x, view.min.y + at.y);
		let click = || {
			let mut events = vec![egui::Event::PointerMoved(pointer)];

			for pressed in [true, false] {
				events.push(egui::Event::PointerButton {
					pos: pointer,
					button: egui::PointerButton::Primary,
					pressed,
					modifiers: Modifiers::NONE,
				});
			}

			events
		};

		drop(frame_on(&context, &mut panels, &mut world, click(), Modifiers::NONE));

		assert_eq!(panels.selection.at(), Pick::Entity(group), "the group around the crate");

		panels.apply(&mut world, Change::Select(Pick::Nothing));
		drop(frame_on(&context, &mut panels, &mut world, click(), Modifiers::ALT));

		assert_eq!(
			panels.selection.at(),
			Pick::Entity(crate_),
			"and the crate itself with alt held"
		);
	}

	/// A lamp standing somewhere, drawing nothing at all.
	fn lit(world: &mut World, at: Vec3) -> EntityId {
		let id = world.entities.spawn_at(Transform::at(at));
		world
			.entities
			.set_light(id, colby_core::abi::Light::point(Vec3::ONE, 1.0, 4.0));

		id
	}

	/// A world being edited, looked at from nine back along z.
	fn looked_at() -> World {
		let mut world = World::new();
		world.editing = true;
		world.camera.position = Vec3::new(0.0, 0.0, 9.0);
		world.camera.target = Vec3::ZERO;

		world
	}

	/// Where something in the world lands in the window.
	fn on_screen(world: &World, view: Rect, at: Vec3) -> Pos2 {
		let size = Vec2::new(view.width(), view.height());
		let point = gizmo::project(
			world
				.render_camera()
				.view_projection(size.x / size.y),
			at,
			size,
		)
		.expect("it is in front of the camera");

		Pos2::new(view.min.x + point.x, view.min.y + point.y)
	}

	/// A press and a release in one frame.
	fn clicking(pointer: Pos2) -> Vec<egui::Event> {
		let mut events = vec![egui::Event::PointerMoved(pointer)];

		for pressed in [true, false] {
			events.push(egui::Event::PointerButton {
				pos: pointer,
				button: egui::PointerButton::Primary,
				pressed,
				modifiers: Modifiers::NONE,
			});
		}

		events
	}

	#[test]
	fn a_click_on_a_lamp_s_mark_selects_it_where_a_ray_would_find_nothing() {
		let mut world = looked_at();
		let lamp = lit(&mut world, Vec3::ZERO);
		let mut panels = Panels::default();
		let context = Context::default();
		let (view, _) = frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE);
		let pointer = on_screen(&world, view, Vec3::ZERO);

		assert_eq!(
			aim::under(&world, world.camera.position, Vec3::NEG_Z),
			Pick::Nothing,
			"a lamp has no mesh, so the ray finds nothing"
		);

		drop(frame_on(&context, &mut panels, &mut world, clicking(pointer), Modifiers::NONE));

		assert_eq!(panels.selection.at(), Pick::Entity(lamp), "and the mark answers the click");
	}

	#[test]
	fn a_click_on_a_mark_inside_a_group_selects_the_group_and_with_alt_the_lamp() {
		let mut world = looked_at();
		let lamp = lit(&mut world, Vec3::ZERO);
		let group = world.entities.spawn();
		assert!(world.entities.set_parent(lamp, group));

		if let Some(editing) = world
			.entities
			.record_mut(&colby_core::abi::EDITING, group)
		{
			editing.group = 1;
		}

		let mut panels = Panels::default();
		let context = Context::default();
		let (view, _) = frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE);
		let pointer = on_screen(&world, view, Vec3::ZERO);

		drop(frame_on(&context, &mut panels, &mut world, clicking(pointer), Modifiers::NONE));

		assert_eq!(panels.selection.at(), Pick::Entity(group), "the group around the lamp");

		panels.apply(&mut world, Change::Select(Pick::Nothing));
		drop(frame_on(&context, &mut panels, &mut world, clicking(pointer), Modifiers::ALT));

		assert_eq!(
			panels.selection.at(),
			Pick::Entity(lamp),
			"and the lamp itself with alt held"
		);
	}

	#[test]
	fn the_switch_takes_the_marks_off_the_screen_and_out_of_a_click() {
		let mut world = looked_at();
		world.cvars.var(HELPERS, Value::Bool(false), "");
		let lamp = lit(&mut world, Vec3::ZERO);
		let mut panels = Panels::default();
		let context = Context::default();
		let (view, _) = frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE);
		let (_, shapes) =
			frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE);
		let off = straight_lines(shapes);
		let pointer = on_screen(&world, view, Vec3::ZERO);

		drop(frame_on(&context, &mut panels, &mut world, clicking(pointer), Modifiers::NONE));

		assert_eq!(panels.selection.at(), Pick::Nothing, "nothing to click on");

		world.cvars.set(HELPERS, "true");
		drop(frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE));
		let (_, shapes) =
			frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE);

		assert_eq!(straight_lines(shapes), off + 4, "a lamp's mark is four rays");

		drop(frame_on(&context, &mut panels, &mut world, clicking(pointer), Modifiers::NONE));

		assert_eq!(panels.selection.at(), Pick::Entity(lamp), "and it answers a click again");
	}

	#[test]
	fn a_reach_dragged_by_its_handle_is_one_step_back_however_long_the_pointer_rests() {
		let mut world = looked_at();
		let lamp = lit(&mut world, Vec3::ZERO);
		let other = lit(&mut world, Vec3::new(0.0, 3.0, 0.0));
		let mut panels = Panels::default();
		panels.apply(&mut world, Change::Select(Pick::Entity(other)));
		panels.apply(&mut world, Change::Toggle(Pick::Entity(lamp)));

		let context = Context::default();
		let (view, _) = frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE);
		let handle = helper::handles(&world, &world.render_camera(), Pick::Entity(lamp))
			.first()
			.copied()
			.expect("a lit lamp offers its reach");
		let pointer = on_screen(&world, view, handle.at);
		let press = |pressed, at: Pos2| egui::Event::PointerButton {
			pos: at,
			button: egui::PointerButton::Primary,
			pressed,
			modifiers: Modifiers::NONE,
		};

		let held = |world: &World, id| {
			world
				.entities
				.light(id)
				.map(|light| light.range)
				.unwrap_or_default()
		};

		drop(frame_on(
			&context,
			&mut panels,
			&mut world,
			vec![egui::Event::PointerMoved(pointer), press(true, pointer)],
			Modifiers::NONE,
		));

		let out = Pos2::new(pointer.x + 40.0, pointer.y);
		drop(frame_on(
			&context,
			&mut panels,
			&mut world,
			vec![egui::Event::PointerMoved(out)],
			Modifiers::NONE,
		));

		let after = held(&world, lamp);
		assert!(after > 4.0, "the drag let the lamp out: {after}");

		// the pointer resting, which is what makes a drag several steps back
		// anywhere the record is opened only by a value moving
		for _ in 0..4 {
			drop(frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE));
		}

		let far = Pos2::new(pointer.x + 80.0, pointer.y);
		drop(frame_on(
			&context,
			&mut panels,
			&mut world,
			vec![egui::Event::PointerMoved(far)],
			Modifiers::NONE,
		));
		drop(frame_on(
			&context,
			&mut panels,
			&mut world,
			vec![press(false, far)],
			Modifiers::NONE,
		));
		drop(frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE));

		let ended = held(&world, lamp);

		assert!(ended > after, "and it kept going after the rest: {ended}");
		assert_eq!(panels.tabs.history().len(), 1, "one step back for the whole drag");
		assert_eq!(panels.tabs.history().undoable(), Some("reach"));
		assert!(
			(held(&world, other) - ended).abs() < 1.0e-4,
			"and the other lamp picked reaches as far: {}",
			held(&world, other)
		);
	}

	#[test]
	fn a_press_and_a_release_on_a_handle_are_not_a_click_on_what_is_behind_it() {
		let mut world = looked_at();
		let lamp = lit(&mut world, Vec3::ZERO);
		let mut panels = Panels::default();
		panels.apply(&mut world, Change::Select(Pick::Entity(lamp)));

		let context = Context::default();
		let (view, _) = frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE);
		let handle = helper::handles(&world, &world.render_camera(), Pick::Entity(lamp))
			.first()
			.copied()
			.expect("a lit lamp offers its reach");
		let pointer = on_screen(&world, view, handle.at);
		let button = |pressed| {
			vec![egui::Event::PointerButton {
				pos: pointer,
				button: egui::PointerButton::Primary,
				pressed,
				modifiers: Modifiers::NONE,
			}]
		};

		// a real pointer presses in one frame and lets go in another, and the
		// click is answered in the second: by then the drag is over, and what
		// keeps the click off the world is the handle still being under it
		drop(frame_on(
			&context,
			&mut panels,
			&mut world,
			vec![egui::Event::PointerMoved(pointer)],
			Modifiers::NONE,
		));
		drop(frame_on(&context, &mut panels, &mut world, button(true), Modifiers::NONE));
		drop(frame_on(&context, &mut panels, &mut world, button(false), Modifiers::NONE));

		assert_eq!(
			panels.selection.at(),
			Pick::Entity(lamp),
			"the lamp is still what is selected, and the empty world behind the handle is not"
		);
	}

	/// What a game keeps on every entity, with a field drawn each way.
	#[repr(C)]
	#[derive(
		Clone, Copy, Debug, PartialEq, colby_core::bytemuck::Pod, colby_core::bytemuck::Zeroable,
	)]
	#[bytemuck(crate = "::colby_core::bytemuck")]
	struct Door {
		reach: f32,
		target: [f32; 3],
	}

	/// The door as the game spells it.
	const DOOR: colby_core::abi::Record<Door> = colby_core::abi::Record {
		name: "door",
		help: "a thing that opens",
		rows: &[
			colby_core::row!(Float, Door, reach, "how far off it notices somebody")
				.drawn(colby_core::abi::Draw::Radius),
			colby_core::row!(Vec3, Door, target, "where it swings to")
				.drawn(colby_core::abi::Draw::Point),
		],
		default: Door { reach: 3.0, target: [1.0, 0.0, 0.0] },
	};

	#[test]
	fn what_a_game_s_record_draws_is_on_screen_and_goes_with_the_switch() {
		let mut world = looked_at();
		world.entities.declare(&DOOR).expect("a door");
		let door = world.entities.spawn_at(Transform::at(Vec3::ZERO));
		let mut panels = Panels::default();
		panels.apply(&mut world, Change::Select(Pick::Entity(door)));

		let context = Context::default();
		drop(frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE));
		let (_, shapes) =
			frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE);
		let drawn = straight_lines(shapes);

		world.cvars.var(HELPERS, Value::Bool(false), "");
		drop(frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE));
		let (_, shapes) =
			frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE);
		let off = straight_lines(shapes);

		assert!(
			drawn > off + 3 * 24,
			"three circles and a line out to the place: {drawn} against {off}"
		);
	}

	#[test]
	fn a_record_s_radius_is_dragged_by_its_own_handle_in_one_step_back() {
		let mut world = looked_at();
		world.entities.declare(&DOOR).expect("a door");
		let door = world.entities.spawn_at(Transform::at(Vec3::ZERO));
		let mut panels = Panels::default();
		panels.apply(&mut world, Change::Select(Pick::Entity(door)));

		let context = Context::default();
		let (view, _) = frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE);
		let handle = helper::noted(&world, &world.render_camera(), Pick::Entity(door))
			.into_iter()
			.find(|handle| matches!(handle.knob, helper::Knob::Radius { .. }))
			.expect("the record says one of its fields is a radius");
		let pointer = on_screen(&world, view, handle.at);
		let press = |pressed, at: Pos2| {
			vec![egui::Event::PointerButton {
				pos: at,
				button: egui::PointerButton::Primary,
				pressed,
				modifiers: Modifiers::NONE,
			}]
		};
		let mut events = vec![egui::Event::PointerMoved(pointer)];
		events.extend(press(true, pointer));

		drop(frame_on(&context, &mut panels, &mut world, events, Modifiers::NONE));

		let out = Pos2::new(pointer.x + 60.0, pointer.y);
		drop(frame_on(
			&context,
			&mut panels,
			&mut world,
			vec![egui::Event::PointerMoved(out)],
			Modifiers::NONE,
		));
		drop(frame_on(&context, &mut panels, &mut world, press(false, out), Modifiers::NONE));
		drop(frame_on(&context, &mut panels, &mut world, Vec::new(), Modifiers::NONE));

		let Some(colby_core::abi::field::Value::Float(reach)) =
			world
				.entities
				.field(door, colby_core::abi::record::ENGINE.len(), 0)
		else {
			panic!("a door keeps a number");
		};

		assert!(reach > 3.0, "the field the record drew grew with the drag: {reach}");
		assert_eq!(panels.tabs.history().len(), 1, "one step back for the drag");
		assert_eq!(
			panels.tabs.history().undoable(),
			Some("record"),
			"named as the panel names it"
		);
	}

	/// How many straight lines some shapes draw, a list of shapes opened up.
	fn straight_lines(shapes: Vec<egui::epaint::ClippedShape>) -> usize {
		let mut open: Vec<egui::Shape> = shapes
			.into_iter()
			.map(|clipped| clipped.shape)
			.collect();
		let mut count = 0;

		while let Some(shape) = open.pop() {
			match shape {
				| egui::Shape::Vec(inner) => open.extend(inner),
				| egui::Shape::LineSegment { .. } => count += 1,
				| _ => {},
			}
		}

		count
	}

	#[test]
	fn a_selected_group_is_drawn_in_a_box_and_a_selected_member_is_not() {
		let mut world = World::new();
		world.editing = true;
		world.camera.position = Vec3::new(0.0, 3.0, 10.0);
		world.camera.target = Vec3::ZERO;
		let group = world.entities.spawn();
		let crate_ = world.entities.spawn_at(Transform::at(Vec3::X));
		world.entities.set_renderable(
			crate_,
			colby_core::abi::Renderable::new(colby_core::abi::MeshId::CUBE, Vec3::ONE),
		);
		assert!(world.entities.set_parent(crate_, group));

		if let Some(editing) = world
			.entities
			.record_mut(&colby_core::abi::EDITING, group)
		{
			editing.group = 1;
		}

		// the second of two frames over one context, which is the one that laid
		// everything out; every straight line in it, the gizmo's arms among them
		let lines = |world: &mut World, pick: Pick| {
			let mut panels = Panels::default();
			panels.apply(world, Change::Select(pick));
			let context = Context::default();
			drop(frame_on(&context, &mut panels, world, Vec::new(), Modifiers::NONE));
			let (_, shapes) = frame_on(&context, &mut panels, world, Vec::new(), Modifiers::NONE);

			straight_lines(shapes)
		};

		let boxed = lines(&mut world, Pick::Entity(group));
		let plain = lines(&mut world, Pick::Entity(crate_));

		assert_eq!(
			boxed,
			plain + 12,
			"the group's box is twelve edges beside the gizmo both of them have"
		);
	}

	#[test]
	fn a_group_is_not_made_of_nothing_and_nothing_is_written_down_for_it() {
		let mut world = World::new();
		world.editing = true;
		let mut panels = Panels::default();
		let count = world.entities.len();

		keyed(&mut panels, &mut world, Key::G, Modifiers::COMMAND);
		frame(&mut panels, &mut world);
		keyed(&mut panels, &mut world, Key::G, Modifiers::COMMAND | Modifiers::SHIFT);
		frame(&mut panels, &mut world);

		assert_eq!(world.entities.len(), count, "no group of nothing");
		assert_eq!(panels.tabs.history().undoable(), None, "and no step back over it");
	}

	#[test]
	fn selecting_what_hangs_off_a_row_selects_the_branch_and_not_the_row() {
		let mut world = World::new();
		let wall = world.entities.spawn();
		let bricks =
			[Vec3::X, Vec3::Y, Vec3::Z].map(|at| world.entities.spawn_at(Transform::at(at)));
		let mortar = world.entities.spawn();

		for brick in bricks {
			assert!(world.entities.set_parent(brick, wall));
		}

		assert!(world.entities.set_parent(mortar, bricks[0]));
		let mut panels = Panels::default();
		panels.apply(&mut world, Change::Select(Pick::Entity(wall)));

		panels.apply(&mut world, Change::Inside(wall));

		assert!(!panels.selection.is(Pick::Entity(wall)), "not the wall");
		assert_eq!(panels.selection.len(), 4, "every brick, and what hangs off a brick");

		for id in bricks.into_iter().chain([mortar]) {
			assert!(panels.selection.is(Pick::Entity(id)));
		}
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
		assert_eq!(panels.tabs.history().undoable(), Some("duplicate"));
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
	fn a_dropped_scene_lands_as_one_record_and_is_selected() {
		let mut world = World::new();
		world.editing = true;
		let mut described = World::new();
		let crate_ = described
			.entities
			.spawn_at(Transform::at(Vec3::Y));
		described.entities.set_name(crate_, "crate");
		world
			.scenes
			.insert("scenes/box", colby_core::abi::scene::capture(&described));
		let mut panels = Panels::default();

		panels.apply(&mut world, Change::Drop {
			name: "scenes/box".to_owned(),
			kind: Kind::Scene,
			at: Vec3::X * 4.0,
		});
		frame(&mut panels, &mut world);
		frame(&mut panels, &mut world);

		assert_eq!(world.entities.len(), 1, "the crate landed");
		let Pick::Entity(landed) = panels.selection.at() else {
			panic!("and is selected");
		};
		assert_eq!(world.entities.name(landed), "crate");
		assert_eq!(panels.tabs.history().undoable(), Some("drop"));

		panels.apply(&mut world, Change::Drop {
			name: "textures/wall".to_owned(),
			kind: Kind::Texture,
			at: Vec3::ZERO,
		});
		frame(&mut panels, &mut world);
		frame(&mut panels, &mut world);
		assert_eq!(world.entities.len(), 1, "a texture is nothing to put in the world");
		assert_eq!(
			panels.tabs.history().undoable(),
			Some("drop"),
			"and no record was made of nothing"
		);
	}

	#[test]
	fn a_play_that_puts_the_world_back_leaves_what_was_done_before_it() {
		let (mut world, mut panels, ball) = hung();
		assert_eq!(panels.tabs.history().undoable(), Some("hang"));

		// play, through the same call the key and the button make
		panels.set_mode(&mut world, false);
		world.editing = false;
		for _ in 0..3 {
			frame(&mut panels, &mut world);
		}
		// the game moves something, every step it runs
		shift(&mut world, ball, Vec3::new(0.0, 9.0, 0.0));
		frame(&mut panels, &mut world);
		assert_eq!(panels.tabs.history().undoable(), Some("hang"), "and nothing is written down");

		// stopping puts the world back, which the runner does; the editor
		// sees the mode change on the frame after. The clock is not put
		// back with it, because it never stopped: a step runs in either
		// mode and only the simulation is skipped.
		shift(&mut world, ball, Vec3::Y);
		world.time += 4.0;
		world.steps += 240;
		world.editing = true;
		frame(&mut panels, &mut world);

		assert_eq!(
			panels.tabs.history().undoable(),
			Some("hang"),
			"a play that changed nothing is not a step, and the record before it stands"
		);
	}

	#[test]
	fn a_play_whose_world_is_kept_is_one_step_to_go_back_over() {
		let (mut world, mut panels, ball) = hung();

		panels.set_mode(&mut world, false);
		world.editing = false;
		frame(&mut panels, &mut world);
		// the game drops it, and the stop keeps where it landed
		shift(&mut world, ball, Vec3::new(0.0, -4.0, 0.0));
		frame(&mut panels, &mut world);
		world.editing = true;
		frame(&mut panels, &mut world);

		assert_eq!(panels.tabs.history().undoable(), Some("play"), "the whole play is one step");

		panels.apply(&mut world, Change::Undo);
		let back = panels
			.restore
			.take()
			.expect("a world to put back");
		let ball = back
			.things
			.iter()
			.find(|thing| thing.transform.position.y > 0.0)
			.expect("the world before the play, with the ball where it was");

		assert!((ball.transform.position.y - 1.0).abs() < 1.0e-5, "at {ball:?}");
		assert_eq!(
			panels.tabs.history().undoable(),
			Some("hang"),
			"and the edit before it is next"
		);
	}

	#[test]
	fn a_play_nobody_wrote_down_leaves_the_records_alone() {
		let (mut world, mut panels, ball) = hung();

		// a typed `sim.edit 0` rather than the key or the button: nothing
		// held a record open, so there is nothing to close
		world.editing = false;
		frame(&mut panels, &mut world);
		shift(&mut world, ball, Vec3::new(0.0, -4.0, 0.0));
		world.editing = true;
		frame(&mut panels, &mut world);

		assert_eq!(panels.tabs.history().undoable(), Some("hang"));
	}
}
