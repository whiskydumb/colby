//! The part of the screen that is not a panel: the world's picture, in the
//! middle of the four.
//!
//! Three gestures, a gizmo and a click, and none of them is a widget: what the
//! pointer does out here is what it does when no panel wanted it. That is the
//! whole of the guard - `Context::egui_wants_pointer_input`, whose own
//! documentation says it is false exactly when "you may be interested in what
//! it is doing" - and everything below is skipped while it is true. It also
//! stays false for a drag that *began* outside a panel and has since crossed
//! one, which is the behavior a camera drag needs and the reason this is the
//! right question to ask rather than "is the pointer over a panel". What
//! makes the middle count as outside egui is that nothing is laid out there:
//! egui takes the root layout's leftover rectangle as the part of the screen
//! it does not own, so the picture is left as exactly that.
//!
//! Everything measured here is measured from the picture's own corner and
//! against its own size, not the window's: the ray a click becomes, where a
//! handle is on screen, how wide the projection is. The window's corner is
//! added back only when a handle is painted.
//!
//! **What has nothing to look at is drawn anyway.** A lamp, a thrower, a decal,
//! a tie between two bodies and a body nobody draws are marked or outlined
//! whatever is selected, and a click lands on those marks before it is offered
//! to the meshes - what is painted on top is what is picked. @ref
//! [`helper`](crate::helper), which owns all of that arithmetic;
//! `editor.helpers` turns the lot off.
//!
//! - **right drag** turns the camera around what it is looking at;
//! - **middle drag** slides what it is looking at across the view;
//! - **the wheel** moves closer and further;
//! - **left drag on a handle** moves, turns or stretches what is selected;
//! - **left click on anything else** selects whatever is under the pointer;
//! - **left drag while the brush is out** paints the ground under the pointer,
//!   and holds the gizmo back for as long as it is out.
//!
//! The order of those last two matters: a press that lands on a handle starts a
//! drag and is *not* also a selection, or grabbing the arm of the thing you
//! have selected would immediately select whatever is behind it.
//!
//! **The brush is a mode and not a fourth thing the gizmo does**, which is what
//! the one engine in the field with a foliage brush does too: while it is out
//! there are no handles to grab and a click selects nothing, because every
//! gesture in the middle of the screen belongs to the stroke. `b` puts it out
//! and away again, and it is only ever out while something that strews is
//! selected. @ref [`paint`](crate::paint) for the arithmetic and the tests.
//!
//! The camera is the editor's **only while the world is being edited**. While
//! it is being played the game owns it and this holds nothing at all, so that
//! resuming an edit takes the view the game left rather than the one from
//! before it started.
//!
//! Everything here is egui talking to [`aim`](crate::aim) and
//! [`gizmo`](crate::gizmo), which is where the arithmetic and the tests are.

use colby_core::{
	abi::{Camera, EmitterKind, EntityId, LightKind, Transform, World},
	glam::{Mat4, Quat, Vec2, Vec3},
	trace,
};
use egui::{
	Color32, Context, Key, LayerId, Painter, PointerButton, Pos2, Rect, Stroke, StrokeKind, vec2,
};

use crate::{
	Editor,
	aim::{self, View},
	gizmo::{self, Axis, Tool},
	helper::{self, Drawn, Handle, Kind, Knob, MARK, Mark, Outline},
	history::History,
	paint::{self, Brush},
	select::{self, Pick, Selection},
};

/// How thick a handle is drawn, and how much thicker the grabbed one is.
const INK: (f32, f32) = (2.0, 3.5);

/// How wide the blob on the end of an arm is, in points.
const TIP: f32 = 4.5;

/// What a handle under the pointer is drawn in.
const LIT: Color32 = Color32::from_rgb(255, 235, 120);

/// What a lamp's reach is drawn in.
///
/// Dimmer than [`LIT`] and the same hue: it is a fact about the scene rather
/// than something under the pointer, and it has to read as ink that is not
/// asking to be grabbed.
const LAMP: Color32 = Color32::from_rgb(190, 170, 70);

/// What the box around a selected group is drawn in.
///
/// Dim like [`LAMP`] and of another hue: where a group reaches is a fact about
/// the scene, and a lamp's reach and a group's box can be on screen together.
const GROUPED: Color32 = Color32::from_rgb(120, 150, 190);

/// What a tie between two bodies is drawn in.
///
/// Its own hue, because a tie, a lamp's reach and a group's box can all be on
/// screen at once and the eye has to tell them apart.
const TIE: Color32 = Color32::from_rgb(110, 190, 200);

/// What the ring showing where the brush is is drawn in.
///
/// Its own hue again, and bright like [`LIT`] rather than dim like the facts
/// about the scene: it is where the pointer is, which is the one thing on
/// screen that answers to the hand.
const BRUSH: Color32 = Color32::from_rgb(230, 130, 200);

/// How many straight pieces the brush's ring is drawn as.
const BRUSH_STEPS: usize = 48;

/// What a body nobody draws is outlined in.
const SOLID: Color32 = Color32::from_rgb(145, 145, 155);

/// What a body full of fluid is outlined in.
const FLUID: Color32 = Color32::from_rgb(90, 155, 210);

/// What a thing that throws particles is marked in.
const THROWN: Color32 = Color32::from_rgb(215, 140, 80);

/// What a thing that paints is marked in.
const PAINT: Color32 = Color32::from_rgb(165, 140, 220);

/// What a field of a game's own record draws itself in.
const NOTED: Color32 = Color32::from_rgb(110, 200, 150);

/// How wide the blob in the middle of a mark is, in points.
const EYE: f32 = 3.5;

/// The eight corners of the unit cube a decal's box is, in its own space.
const BOX_CORNERS: [Vec3; 8] = [
	Vec3::new(-0.5, -0.5, -0.5),
	Vec3::new(0.5, -0.5, -0.5),
	Vec3::new(0.5, 0.5, -0.5),
	Vec3::new(-0.5, 0.5, -0.5),
	Vec3::new(-0.5, -0.5, 0.5),
	Vec3::new(0.5, -0.5, 0.5),
	Vec3::new(0.5, 0.5, 0.5),
	Vec3::new(-0.5, 0.5, 0.5),
];

/// Which pairs of [`BOX_CORNERS`] are the box's twelve edges.
const BOX_EDGES: [(usize, usize); 12] = [
	(0, 1),
	(1, 2),
	(2, 3),
	(3, 0),
	(4, 5),
	(5, 6),
	(6, 7),
	(7, 4),
	(0, 4),
	(1, 5),
	(2, 6),
	(3, 7),
];

/// What the pointer does outside every window.
#[derive(Debug)]
pub(crate) struct Viewport {
	/// The editor's camera, while it has one.
	view: Option<View>,

	/// Which of the three things the gizmo is doing.
	tool: Tool,

	/// The drag in progress, if one is.
	grab: Option<Grab>,

	/// Whether the brush is out, and the stroke in progress if one is.
	brush: Option<Brush>,

	/// How wide the brush is, in the ground's own units.
	radius: f32,

	/// How hard it paints, a share of a whole cell per dab.
	strength: f32,
}

impl Default for Viewport {
	fn default() -> Self {
		Self {
			view: None,
			tool: Tool::default(),
			grab: None,
			brush: None,
			radius: paint::RADIUS,
			strength: paint::STRENGTH,
		}
	}
}

/// A drag in progress: the gizmo moving everything selected, or one of the
/// selected thing's own handles writing one of its fields.
#[derive(Clone, Debug)]
enum Grab {
	/// An arm or a ring of the gizmo.
	Tool(Pulled),

	/// A handle of the thing itself. @ref [`helper`](crate::helper).
	Field(Held),
}

/// A drag of one field's handle, from the moment it was grabbed.
#[derive(Clone, Debug)]
struct Held {
	/// The handle as it stood when it was taken hold of: where its line is in
	/// the world, and what its field held then.
	handle: Handle,

	/// What it hangs off.
	pick: Pick,

	/// Where on the handle's line or plane the pointer read the first time.
	start: Vec3,

	/// Every other entity picked, which the change is written to as well - the
	/// inspector's rule for a changed field, through the same call.
	others: Vec<EntityId>,
}

/// A drag of one of the gizmo's handles, from the moment it was grabbed.
#[derive(Clone, Debug)]
struct Pulled {
	/// Which handle.
	axis: Axis,

	/// What it was doing when it was grabbed.
	///
	/// Kept rather than read again, so that pressing another tool's key
	/// mid-drag cannot apply one kind of change to another kind's numbers.
	tool: Tool,

	/// The transform the drag began at.
	from: Transform,

	/// How long the arm was in world units when it was grabbed, so that a
	/// stretch is measured against the thing's own size rather than in units.
	arm: f32,

	/// What the pointer read the first time.
	start: f32,

	/// What it read last frame.
	last: f32,

	/// How far the drag has gone in total.
	total: f32,

	/// Everything else that was selected when the handle was grabbed, each
	/// with where it was in the world at that moment: what a drag of any
	/// length lands them from. @ref [`select::drag_all`].
	others: Vec<(Pick, Transform)>,
}

impl Viewport {
	/// Which of the three things the gizmo is doing.
	pub(crate) const fn tool(&self) -> Tool { self.tool }

	/// Switches the gizmo to one of its three things, from a panel rather
	/// than a key.
	/// Switches the gizmo, which is also what puts the brush away: they are two
	/// modes and not four tools, so asking for one is saying no to the other.
	pub(crate) fn set_tool(&mut self, tool: Tool) {
		self.tool = tool;
		self.brush = None;
		self.grab = None;
	}

	/// Drives the camera and the gizmo, and answers what was clicked.
	///
	/// @param context - egui, mid-frame
	/// @param world - read for what is in it, written for the camera and for
	/// whatever the gizmo is dragging
	/// @param selection - what the gizmo is attached to
	/// @param view - where the world's picture is on the screen, in points
	/// @param history - where a drag is written down, so that it can be undone
	/// @return what a click landed on, or nothing at all if there was no click
	/// to answer for. A click on empty space answers [`Pick::Nothing`], which
	/// is a different thing from not having clicked and is what deselects
	pub(crate) fn run(
		&mut self,
		context: &Context,
		world: &mut World,
		selection: &Selection,
		view: Rect,
		history: &mut History,
	) -> Option<Pick> {
		if !world.editing {
			// the game's camera again. Whatever orbit this was holding
			// described a view from before the game started moving it.
			self.view = None;
			self.grab = None;

			return None;
		}

		let busy = context.egui_wants_pointer_input();
		let held = Gestures::read(context, view);

		self.pick_tool(context);
		self.fly(world, busy, held);

		// the brush before the gizmo and instead of it: while it is out there
		// are no handles to grab, so there is nothing for the two of them to
		// disagree about
		let painting = self.stroke(context, world, selection, busy, held, view, history);
		let dragging =
			painting || self.gizmo(context, world, selection, busy, held, view, history);

		let found = if busy || dragging || painting || !held.clicked {
			None
		} else {
			held.at.map(|at| picked(world, at, size(view)))
		};

		if held.clicked {
			// `busy` on it because the commonest answer to "why did my click
			// do nothing" is that a panel was under it, and `dragging` because
			// the second commonest is that it landed on a handle.
			trace!(busy, dragging, at = ?held.at, ?found, "a click in the world");
		}

		found
	}

	/// Switches between move, turn and stretch.
	///
	/// The three keys every editor with a gizmo uses. Skipped while a text
	/// field is taking typing, so that a `w` in the console stays a `w` - and
	/// only then, because a row that was clicked holds egui's focus as well.
	fn pick_tool(&mut self, context: &Context) {
		if context.text_edit_focused() {
			return;
		}

		let asked = context.input(|input| {
			[(Key::W, Tool::Move), (Key::E, Tool::Turn), (Key::R, Tool::Size)]
				.into_iter()
				.find(|&(key, _)| input.key_pressed(key))
				.map(|(_, tool)| tool)
		});

		if let Some(tool) = asked {
			self.set_tool(tool);
		}

		if context.input(|input| input.key_pressed(Key::B)) {
			self.set_brushing(self.brush.is_none());
		}
	}

	/// Whether the brush is out.
	pub(crate) const fn brushing(&self) -> bool { self.brush.is_some() }

	/// Puts the brush out, or away.
	pub(crate) fn set_brushing(&mut self, out: bool) {
		self.brush = out.then(Brush::default);
		// a gesture in the middle of the screen belongs to one of the two, and
		// changing which of them halfway through a drag would apply the first
		// half to the other one
		self.grab = None;
	}

	/// How wide the brush is, and how hard it paints.
	pub(crate) const fn stroke_of(&self) -> (f32, f32) { (self.radius, self.strength) }

	/// Makes the brush this wide and this hard.
	///
	/// @param radius - how wide, held inside [`paint::RANGE`]
	/// @param strength - how hard, held between nought and one
	pub(crate) const fn set_stroke(&mut self, radius: f32, strength: f32) {
		self.radius = radius.clamp(paint::RANGE.0, paint::RANGE.1);
		self.strength = strength.clamp(0.0, 1.0);
	}

	/// Moves the camera by whatever the pointer did.
	fn fly(&mut self, world: &mut World, busy: bool, held: Gestures) {
		let mut view = View::taken(self.view, &world.camera);

		if !busy && self.grab.is_none() {
			if held.turning {
				view.orbit().turn(held.drag);
			}

			if held.sliding {
				view.orbit().slide(held.drag);
			}

			if held.wheel.abs() > f32::EPSILON {
				// egui reports the wheel in points rather than in notches, and
				// one notch is fifty of them on every platform it runs on.
				view.orbit().dolly(held.wheel / 50.0);
			}
		}

		view.put(&mut world.camera);
		self.view = Some(view);
	}

	/// Draws the gizmo, everything with nothing to look at, and applies
	/// whatever is being dragged.
	///
	/// @return whether the pointer is on the gizmo's business rather than the
	/// world's
	#[expect(
		clippy::too_many_arguments,
		reason = "one frame of what the pointer did, against what it did it to; a struct for \
		          the seven would be this signature with a name"
	)]
	fn gizmo(
		&mut self,
		context: &Context,
		world: &mut World,
		selection: &Selection,
		busy: bool,
		held: Gestures,
		view: Rect,
		history: &mut History,
	) -> bool {
		// **Asked twice on purpose.** `run` above stops short of calling this at
		// all while the brush is out, so taking this out changes nothing today;
		// what it is for is that a gesture in the middle of the screen belongs
		// to one of the two, and a caller rearranged must not be able to let
		// the gizmo draw handles a stroke would then fight over.
		if self.brush.is_some() {
			return false;
		}

		let pick = selection.at();
		let camera = world.render_camera();
		let viewport = size(view);
		let shown = Editor::helpers(world);

		// whatever is selected, and even when nothing is: a click cannot land
		// on what is not drawn, which is the whole reason a mark exists
		if shown {
			helpers(context, world, selection, &camera, (viewport, view, held.at));
		}

		let mut knobs = if shown {
			helper::handles(world, &camera, pick)
		} else {
			Vec::new()
		};

		if shown {
			knobs.extend(helper::noted(world, &camera, pick));
		}
		let over_knob = held
			.at
			.and_then(|point| helper::grabbed(&knobs, &camera, viewport, point));

		let placed = select::transform(world, pick);
		let handles = placed.map(|at| Handles::of(&camera, at, self.tool, viewport));
		let over = handles
			.as_ref()
			.and_then(|drawn| held.at.and_then(|point| drawn.under(point)));

		if held.released || (placed.is_none() && knobs.is_empty()) {
			self.grab = None;
		}

		if held.pressed {
			// the one line that answers "why will it not grab". `middle` is
			// where the gizmo actually is, which is the thing a person cannot
			// tell from a screenshot when the answer is that they were
			// dragging twenty points away from it.
			trace!(
				busy,
				?over,
				?over_knob,
				at = ?held.at,
				middle = ?handles.as_ref().and_then(|drawn| drawn.middle),
				tool = self.tool.word(),
				"the pointer went down in the world"
			);
		}

		if held.pressed
			&& !busy && let Some(point) = held.at
		{
			// a field's own handle is asked before the arms: it is the smaller
			// target and it is drawn on top of them
			if let Some(handle) = over_knob.and_then(|index| knobs.get(index)) {
				self.hold_field(&camera, selection, handle, point, viewport);
			} else if let (Some(at), Some(axis)) = (placed, over) {
				// everything else selected, with where it stands now: the drag
				// lands them from here, however long it lasts
				let others = selection
					.others()
					.into_iter()
					.filter_map(|other| select::transform(world, other).map(|was| (other, was)))
					.collect();

				self.hold(&camera, at, axis, point, viewport, others);
			}
		}

		if held.down
			&& let Some(point) = held.at
		{
			self.pull(world, pick, &camera, point, viewport, history);
		}

		let lit = self.holding().or_else(|| {
			over_knob
				.and_then(|index| knobs.get(index))
				.map(|handle| handle.knob)
		});

		paint_knobs(context, &knobs, lit, (&camera, viewport, view));

		if let Some(drawn) = handles {
			drawn.paint(context, self.turning().or(over), view);
		}

		self.grab.is_some() || ((over.is_some() || over_knob.is_some()) && !busy)
	}

	/// Which of a thing's own handles the drag in progress is holding.
	fn holding(&self) -> Option<Knob> {
		match &self.grab {
			| Some(Grab::Field(grab)) => Some(grab.handle.knob),
			| Some(Grab::Tool(_)) | None => None,
		}
	}

	/// Which of the gizmo's arms or rings the drag in progress is holding.
	fn turning(&self) -> Option<Axis> {
		match &self.grab {
			| Some(Grab::Tool(grab)) => Some(grab.axis),
			| Some(Grab::Field(_)) | None => None,
		}
	}

	/// Takes hold of one of the gizmo's handles.
	///
	/// @param others - everything else selected, each with where it stands
	fn hold(
		&mut self,
		camera: &Camera,
		at: Transform,
		axis: Axis,
		point: Vec2,
		viewport: Vec2,
		others: Vec<(Pick, Transform)>,
	) {
		let Some(start) = read(camera, at, axis, self.tool, point, viewport) else {
			return;
		};

		self.grab = Some(Grab::Tool(Pulled {
			axis,
			tool: self.tool,
			from: at,
			arm: gizmo::reach(camera, at.position, viewport),
			start,
			last: start,
			total: 0.0,
			others,
		}));
	}

	/// Takes hold of one of the selected thing's own handles.
	///
	/// @param handle - the handle as it stands, which carries what its field
	/// holds: a drag is measured against both
	fn hold_field(
		&mut self,
		camera: &Camera,
		selection: &Selection,
		handle: &Handle,
		point: Vec2,
		viewport: Vec2,
	) {
		let Some(start) = helper::read(handle, camera, point, viewport) else {
			return;
		};

		self.grab = Some(Grab::Field(Held {
			handle: *handle,
			pick: selection.at(),
			start,
			others: selection
				.others()
				.into_iter()
				.filter_map(|other| match other {
					| Pick::Entity(id) => Some(id),
					| Pick::Nothing
					| Pick::Body(_)
					| Pick::Joint(_)
					| Pick::Material(_)
					| Pick::Model(_) => None,
				})
				.collect(),
		}));
	}

	/// Applies whichever drag is in progress.
	fn pull(
		&mut self,
		world: &mut World,
		pick: Pick,
		camera: &Camera,
		point: Vec2,
		viewport: Vec2,
		history: &mut History,
	) {
		if matches!(self.grab, Some(Grab::Tool(_))) {
			self.pull_tool(world, pick, camera, point, viewport, history);
		} else if matches!(self.grab, Some(Grab::Field(_))) {
			self.pull_field(world, camera, point, viewport, history);
		}
	}

	/// Paints with the brush, and draws where it is.
	///
	/// **A stroke is one step back, and nothing here arranges that**: a record
	/// opens on the first frame that writes and closes on the first frame that
	/// does not, so a drag from the moment the button goes down to the moment
	/// it comes up is one record by the rule every gesture in this editor
	/// already follows. @ref [`History::begin`].
	///
	/// @return whether the brush is out, which holds the gizmo and a click back
	#[expect(
		clippy::too_many_arguments,
		reason = "one frame of what the pointer did against what it did it to, which is the \
		          gizmo's own signature"
	)]
	fn stroke(
		&mut self,
		context: &Context,
		world: &mut World,
		selection: &Selection,
		busy: bool,
		held: Gestures,
		view: Rect,
		history: &mut History,
	) -> bool {
		let Some(mut brush) = self.brush else {
			return false;
		};
		let camera = world.render_camera();
		let viewport = size(view);
		let painting = strewings(world, selection);
		// where the brush is, which is a place on the *first* one's ground:
		// every strewing painted at once hangs off some ground, and what is
		// drawn has to be one ring rather than one a strewing
		let over = held
			.at
			.zip(painting.first().copied())
			.and_then(|(point, id)| {
				let (from, along) = aim::ray(&camera, point, viewport);

				paint::ground_under(world, id, from, along).map(|at| (id, at))
			});

		if held.released {
			brush.lift();
		}

		if held.down
			&& !busy && let Some((_, at)) = over
		{
			// **Written down before the first cell moves, and written every
			// frame the button is down.** Both halves were found by driving a
			// window and neither by a test. Before, because `begin` keeps the
			// world it is first handed, and a call after the paint would keep
			// one that already had a dab in it - an undo would then leave that
			// dab standing. Every frame, because a stroke dabs by *distance*:
			// a hand that slows down writes nothing for a frame or two, and a
			// record closes on the first frame in which nothing writes, so one
			// stroke would be several steps back. Every other gesture in this
			// editor writes every frame it lasts and needs neither.
			history.begin("paint", world);

			if brush.dab(Vec2::new(at.x, at.z), self.radius) {
				self.dab(context, world, Vec2::new(at.x, at.z), &painting);
			}
		}

		self.brush = Some(brush);
		paint_brush(context, world, &camera, over, (self.radius, viewport, view));

		true
	}

	/// Lands one dab of the brush on every strewing being painted.
	///
	/// A call of its own rather than the body of the stroke's own `if`, because
	/// two questions and a loop inside them is one level of nesting past what
	/// this workspace allows.
	///
	/// @param at - where the dab lands, east and south in the ground's space
	/// @param painting - every strewing selected that strews something
	fn dab(&self, context: &Context, world: &mut World, at: Vec2, painting: &[EntityId]) {
		// ctrl puts a field back where a plain stroke takes one away. Which way
		// round is not a preference: a rule lays its whole field until somebody
		// paints, so the stroke that does anything to a strewing nobody has
		// painted is the one that thins it.
		let back = context.input(|input| input.modifiers.command);
		let strength = if back { -self.strength } else { self.strength };
		let mut moved = false;

		for id in painting {
			// each in its own grid and each from the same place on the ground:
			// they hang off one ground, so one stroke is one clearing in every
			// field over it
			moved |= paint::paint(world, *id, at, self.radius, strength);
		}

		trace!(
			strewings = painting.len(),
			east = at.x,
			south = at.y,
			radius = self.radius,
			strength,
			moved,
			"a dab of the brush"
		);
	}

	/// Applies a drag of one of the gizmo's handles.
	///
	/// Written down first, every frame of it: the record opens on the first
	/// frame and stays open while the drag writes, so the whole drag is one
	/// step back. @ref [`History::begin`].
	fn pull_tool(
		&mut self,
		world: &mut World,
		pick: Pick,
		camera: &Camera,
		point: Vec2,
		viewport: Vec2,
		history: &mut History,
	) {
		let Some(Grab::Tool(grab)) = self.grab.as_mut() else {
			return;
		};

		let Some(now) = read(camera, grab.from, grab.axis, grab.tool, point, viewport) else {
			return;
		};

		grab.total = match grab.tool {
			// a distance along a line grows without wrapping, so the whole
			// drag is the difference from where it began - exact however many
			// frames it took, and unaffected by one of them being slow.
			| Tool::Move | Tool::Size => now - grab.start,
			// an angle is only known to within a whole turn, so this one has
			// to add the frames up. Each step is wrapped into the half turn
			// either side of nothing, which is right for any drag nobody can
			// make faster than half a turn between two frames.
			| Tool::Turn => grab.total + wrapped(now - grab.last),
		};
		grab.last = now;

		let put = match grab.tool {
			| Tool::Move => gizmo::moved(grab.from, grab.axis, grab.total),
			| Tool::Turn => gizmo::turned(grab.from, grab.axis, grab.total),
			| Tool::Size =>
				gizmo::sized(grab.from, grab.axis, 1.0 + grab.total / grab.arm.max(1.0e-4)),
		};
		let put = held(put, grab.tool, Editor::grid(world));
		let (tool, from, others) = (grab.tool, grab.from, grab.others.clone());

		history.begin(tool.word(), world);
		select::drag_all(world, pick, from, put, &others);
	}

	/// Applies a drag of one of the selected thing's own handles.
	///
	/// Written down every frame it is held, exactly as a gizmo drag is, and
	/// that is what makes a drag with a pause in it one step back rather than
	/// several: a record ends on the first frame nobody writes.
	fn pull_field(
		&self,
		world: &mut World,
		camera: &Camera,
		point: Vec2,
		viewport: Vec2,
		history: &mut History,
	) {
		let Some(Grab::Field(grab)) = self.grab.as_ref() else {
			return;
		};

		let Some(now) = helper::read(&grab.handle, camera, point, viewport) else {
			return;
		};

		let value = helper::dragged(&grab.handle, now - grab.start, Editor::grid(world));

		history.begin(grab.handle.knob.word(), world);
		helper::write(world, grab.pick, &grab.others, &grab.handle, value);
	}
}

/// Everything the editor draws for things that have nothing to look at.
///
/// @param at - where the pointer is, so that whatever it is resting on is drawn
/// as the thing that would answer a click
fn helpers(
	context: &Context,
	world: &World,
	selection: &Selection,
	camera: &Camera,
	(viewport, view, at): (Vec2, Rect, Option<Vec2>),
) {
	let marked = helper::marks(world, camera, viewport);
	let lines = helper::outlines(world);
	let lit = at.and_then(|point| helper::nearest(&marked, point));
	let touched = at
		.filter(|_| lit.is_none())
		.and_then(|point| helper::touched(&lines, camera, viewport, point));
	let detail: Vec<(Vec3, Vec3)> = selection
		.picks()
		.into_iter()
		.flat_map(|pick| helper::detail(world, pick))
		.collect();

	let sketched: Vec<(Vec3, Vec3)> = selection
		.picks()
		.into_iter()
		.flat_map(|pick| helper::sketch(world, pick))
		.collect();

	paint_marks(context, &marked, lit, view);
	paint_outlines(context, &lines, touched, &detail, camera, viewport, view);
	paint_sketch(context, &sketched, camera, viewport, view);
	lamps(context, world, selection, camera, viewport, view);
	throwers(context, world, selection, camera, viewport, view);
	decals(context, world, selection, camera, viewport, view);
	groups(context, world, selection, camera, viewport, view);
}

/// Every mark, drawn where the thing it stands for stands.
///
/// A glyph of lines and circles rather than a picture: a mark says which of
/// four things this is and nothing else, and an editor that had to load a
/// texture to say it would have one more thing to go wrong.
///
/// @param lit - which of them the pointer is resting on
fn paint_marks(context: &Context, marks: &[Mark], lit: Option<usize>, view: Rect) {
	let painter = context
		.layer_painter(LayerId::background())
		.with_clip_rect(view);
	let corner = Vec2::new(view.min.x, view.min.y);

	for (index, mark) in marks.iter().enumerate() {
		let color = if lit == Some(index) { LIT } else { tint(mark.kind) };
		let stroke = Stroke::new(INK.0, color);
		let at = spot(mark.at + corner);

		match mark.kind {
			| Kind::Lamp => {
				painter.circle_filled(at, EYE, color);

				for way in [Vec2::X, Vec2::NEG_X, Vec2::Y, Vec2::NEG_Y] {
					painter.line_segment(
						[
							spot(mark.at + corner + way * (EYE + 2.0)),
							spot(mark.at + corner + way * MARK),
						],
						stroke,
					);
				}
			},
			| Kind::Spot => {
				painter.circle_filled(at, EYE, color);

				// down the way it throws, so a cone aimed at the floor reads
				// as one from above without turning the camera
				let way = mark.aim.unwrap_or(Vec2::Y);

				painter.line_segment(
					[
						spot(mark.at + corner + way * (EYE + 1.5)),
						spot(mark.at + corner + way * (MARK + 4.0)),
					],
					stroke,
				);
			},
			| Kind::Thrower =>
				for offset in [
					Vec2::ZERO,
					Vec2::new(MARK * 0.7, -MARK * 0.6),
					Vec2::new(-MARK * 0.6, -MARK * 0.7),
				] {
					painter.circle_filled(spot(mark.at + corner + offset), 2.0, color);
				},
			| Kind::Painter => {
				let wide = MARK * 0.8;

				painter.rect_stroke(
					Rect::from_center_size(at, vec2(wide * 2.0, wide * 2.0)),
					1.0,
					stroke,
					StrokeKind::Middle,
				);
				painter.circle_filled(at, 1.5, color);
			},
		}
	}
}

/// What a kind of mark is drawn in.
const fn tint(kind: Kind) -> Color32 {
	match kind {
		| Kind::Lamp | Kind::Spot => LAMP,
		| Kind::Thrower => THROWN,
		| Kind::Painter => PAINT,
	}
}

/// Every outline, and whatever a selected thing adds to its own.
///
/// @param lit - which of them the pointer is resting on
fn paint_outlines(
	context: &Context,
	lines: &[Outline],
	lit: Option<usize>,
	detail: &[(Vec3, Vec3)],
	camera: &Camera,
	viewport: Vec2,
	view: Rect,
) {
	let painter = context
		.layer_painter(LayerId::background())
		.with_clip_rect(view);
	let corner = Vec2::new(view.min.x, view.min.y);
	let view_projection = camera.view_projection(viewport.x / viewport.y.max(1.0));

	for (index, outline) in lines.iter().enumerate() {
		let color = if lit == Some(index) {
			LIT
		} else {
			match outline.drawn {
				| Drawn::Tie => TIE,
				| Drawn::Solid => SOLID,
				| Drawn::Fluid => FLUID,
			}
		};
		let stroke = Stroke::new(INK.0, color);

		for (from, to) in &outline.segments {
			segment(&painter, view_projection, (*from, *to), (viewport, corner), stroke);
		}
	}

	let stroke = Stroke::new(INK.0, TIE);

	for (from, to) in detail {
		segment(&painter, view_projection, (*from, *to), (viewport, corner), stroke);
	}
}

/// What a selected thing's own records draw: a radius as circles, a place as a
/// line out to where it is.
///
/// Its own color, because it is the one thing on screen a *game* asked for: the
/// rest of what is drawn here is the engine saying what it knows about a lamp,
/// a tie or a box.
fn paint_sketch(
	context: &Context,
	sketched: &[(Vec3, Vec3)],
	camera: &Camera,
	viewport: Vec2,
	view: Rect,
) {
	let painter = context
		.layer_painter(LayerId::background())
		.with_clip_rect(view);
	let corner = Vec2::new(view.min.x, view.min.y);
	let view_projection = camera.view_projection(viewport.x / viewport.y.max(1.0));
	let stroke = Stroke::new(INK.0, NOTED);

	for (from, to) in sketched {
		segment(&painter, view_projection, (*from, *to), (viewport, corner), stroke);
	}
}

/// One line in the world, projected and drawn.
///
/// @param corner - where the picture is on the screen
fn segment(
	painter: &Painter,
	view_projection: Mat4,
	(from, to): (Vec3, Vec3),
	(viewport, corner): (Vec2, Vec2),
	stroke: Stroke,
) {
	let (Some(start), Some(end)) = (
		gizmo::project(view_projection, from, viewport),
		gizmo::project(view_projection, to, viewport),
	) else {
		return;
	};

	painter.line_segment([spot(start + corner), spot(end + corner)], stroke);
}

/// Every entity selected that strews something, in the order they were
/// selected.
fn strewings(world: &World, selection: &Selection) -> Vec<EntityId> {
	selection
		.picks()
		.into_iter()
		.filter_map(|pick| match pick {
			| Pick::Entity(id) => Some(id),
			| _ => None,
		})
		.filter(|id| colby_core::abi::strew::strews(&world.entities, *id))
		.collect()
}

/// The ring that shows where the brush is.
///
/// Drawn from a circle in the *ground's* own space, so that a ground turned or
/// scaled unevenly shows the oval the stroke really paints rather than a circle
/// it does not. What it does not follow is a hillside: every point of the ring
/// is at the height of the place under the pointer, so a ring on a steep slope
/// cuts into it. A ring that followed the ground would be a ray a point, which
/// is a hundred traces a frame for a line somebody is not looking at.
///
/// **The mask itself is not drawn**, and that is the card's one deliberate
/// gap: showing it wants a pass over the ground's own surface. What a person
/// needs while painting is where the brush is and what it did, and the second
/// of those is the field thinning under the stroke.
///
/// @param over - the strewing being painted and where the ray met its ground,
/// in the ground's own space
fn paint_brush(
	context: &Context,
	world: &World,
	camera: &Camera,
	over: Option<(EntityId, Vec3)>,
	(radius, viewport, view): (f32, Vec2, Rect),
) {
	let Some((id, at)) = over else {
		return;
	};

	let Some(ground) = paint::ground_of(world, id) else {
		return;
	};
	let painter = context.layer_painter(LayerId::background());
	let projection = camera.view_projection(viewport.x / viewport.y.max(1.0e-4));
	let corner = view.min.to_vec2();
	let mut last: Option<Vec2> = None;

	for point in paint::ring(at, radius, BRUSH_STEPS) {
		let now = gizmo::project(projection, ground.transform_point3(point), viewport);

		if let (Some(from), Some(to)) = (last, now) {
			painter.line_segment(
				[Pos2::new(from.x, from.y) + corner, Pos2::new(to.x, to.y) + corner],
				Stroke::new(INK.0, BRUSH),
			);
		}

		last = now;
	}
}

/// The handles a selected thing's own fields offer, drawn as blocks.
///
/// A block rather than a blob, which is what the size tool's arms end in: a
/// handle here stretches a number the same way, and the shape is the one thing
/// that says so before it is dragged.
///
/// @param lit - which field the pointer is resting on, or is dragging
fn paint_knobs(
	context: &Context,
	knobs: &[Handle],
	lit: Option<Knob>,
	(camera, viewport, view): (&Camera, Vec2, Rect),
) {
	let painter = context
		.layer_painter(LayerId::background())
		.with_clip_rect(view);
	let corner = Vec2::new(view.min.x, view.min.y);
	let view_projection = camera.view_projection(viewport.x / viewport.y.max(1.0));

	for handle in knobs {
		let Some(at) = gizmo::project(view_projection, handle.at, viewport) else {
			continue;
		};

		let color = if lit == Some(handle.knob) { LIT } else { LAMP };
		let block = Rect::from_center_size(spot(at + corner), vec2(TIP * 2.0, TIP * 2.0));

		painter.rect_filled(block, 1.0, color);
	}
}

/// How far every selected lamp reaches, drawn over the world.
///
/// The one thing in this file that is not a handle: it cannot be grabbed and
/// nothing hit-tests it. A light has no geometry, so without this the only
/// evidence that an entity is one is a row in a panel and whatever the picture
/// happens to look like - and the number a person is dragging, `range`, has no
/// visible meaning at all. Every reference editor draws exactly this shape for
/// exactly this reason.
///
/// Drawn for the whole selection rather than for the primary alone, so that
/// two lamps being moved together both say where they reach.
fn lamps(
	context: &Context,
	world: &World,
	selection: &Selection,
	camera: &Camera,
	viewport: Vec2,
	view: Rect,
) {
	let painter = context
		.layer_painter(LayerId::background())
		.with_clip_rect(view);
	let corner = Vec2::new(view.min.x, view.min.y);
	let stroke = Stroke::new(INK.0, LAMP);

	for pick in selection.picks() {
		let Pick::Entity(id) = pick else {
			continue;
		};

		let Some(light) = world
			.entities
			.light(id)
			.copied()
			.filter(|it| it.is_lit())
		else {
			continue;
		};

		let at = world.entities.placed(id).unwrap_or_default();

		if light.kind == LightKind::Spot {
			let (inner, outer) = light.cone();
			let way = (at.rotation * Vec3::NEG_Z).normalize_or(Vec3::NEG_Z);

			// four segments out of the apex rather than a polyline through all
			// five points, which would trace the rim as well and close nothing
			let edges = gizmo::cone(camera, at, light.range, outer, viewport);
			for rim in edges.iter().skip(1) {
				painter.line_segment([spot(edges[0] + corner), spot(*rim + corner)], stroke);
			}

			// and the mouth, in the plane the cone points at
			outline(
				&painter,
				&gizmo::circle(
					camera,
					at.position + way * light.range,
					way,
					light.range * outer.tan(),
					viewport,
				),
				corner,
				stroke,
			);

			// and the bright middle inside it, dimmer and only when there is
			// one: `inner` is a number with no meaning on screen anywhere else,
			// and it is nought until somebody sets it
			if inner > 0.0 {
				outline(
					&painter,
					&gizmo::circle(
						camera,
						at.position + way * light.range,
						way,
						light.range * inner.tan(),
						viewport,
					),
					corner,
					Stroke::new(INK.0, LAMP.gamma_multiply(0.55)),
				);
			}

			continue;
		}

		// three circles rather than one facing the eye: a sphere read off a
		// single circle is a disc, and which of the two it is happens to be
		// the whole question when a lamp is inside a wall.
		for normal in [Vec3::X, Vec3::Y, Vec3::Z] {
			outline(
				&painter,
				&gizmo::circle(camera, at.position, normal, light.range, viewport),
				corner,
				stroke,
			);
		}
	}
}

/// Every selected decal's box, drawn over the world.
///
/// The lamps' reason a second time: a decal has no geometry of its own, and
/// the box it paints inside is what a person stretches when they scale one, so
/// without this the only evidence of where it reaches is whatever it happens
/// to have painted.
fn decals(
	context: &Context,
	world: &World,
	selection: &Selection,
	camera: &Camera,
	viewport: Vec2,
	view: Rect,
) {
	let painter = context
		.layer_painter(LayerId::background())
		.with_clip_rect(view);
	let corner = Vec2::new(view.min.x, view.min.y);
	let stroke = Stroke::new(INK.0, LAMP);
	let view_projection = camera.view_projection(viewport.x / viewport.y.max(1.0));

	for pick in selection.picks() {
		let Pick::Entity(id) = pick else {
			continue;
		};

		if !world
			.entities
			.decal(id)
			.is_some_and(|it| it.paints())
		{
			continue;
		}

		let placed = world.entities.placed(id).unwrap_or_default();

		edges(&painter, view_projection, placed.matrix(), viewport, corner, stroke);

		// and which way it throws its picture, which is the one thing about a
		// decal the box does not say: a puddle and a picture on a wall are the
		// same box turned two different ways
		let way = (placed.rotation * Vec3::NEG_Z).normalize_or(Vec3::NEG_Z);
		let tip = placed.position + way * (placed.scale.z.abs() * 0.5);

		for (from, to) in helper::arrow(placed.position, tip) {
			segment(&painter, view_projection, (from, to), (viewport, corner), stroke);
		}
	}
}

/// What every selected thrower throws into, drawn over the world.
///
/// The lamps' reason a third time, and here it is the whole of what is on
/// screen: an emitter has no geometry, and a world being edited has no cloud
/// either - a cloud is the step's, and no step runs while somebody is editing.
/// The cone is drawn as far as a particle thrown at full speed gets before it
/// dies, @ref [`helper::reach`], so what `speed`, `life` and `drag` come to is
/// a distance rather than three numbers in a panel.
fn throwers(
	context: &Context,
	world: &World,
	selection: &Selection,
	camera: &Camera,
	viewport: Vec2,
	view: Rect,
) {
	let painter = context
		.layer_painter(LayerId::background())
		.with_clip_rect(view);
	let corner = Vec2::new(view.min.x, view.min.y);
	let stroke = Stroke::new(INK.0, THROWN);

	for pick in selection.picks() {
		let Pick::Entity(id) = pick else {
			continue;
		};

		let Some(emitter) = world
			.entities
			.emitter(id)
			.copied()
			.filter(|it| it.kind.throws())
		else {
			continue;
		};

		let at = world.entities.placed(id).unwrap_or_default();
		let far = helper::reach(&emitter);

		if far <= 0.0 {
			continue;
		}

		if emitter.kind == EmitterKind::Cone {
			let way = (at.rotation * Vec3::NEG_Z).normalize_or(Vec3::NEG_Z);
			let edges = gizmo::cone(camera, at, far, emitter.spread, viewport);

			for rim in edges.iter().skip(1) {
				painter.line_segment([spot(edges[0] + corner), spot(*rim + corner)], stroke);
			}

			outline(
				&painter,
				&gizmo::circle(
					camera,
					at.position + way * far,
					way,
					far * emitter.spread.tan(),
					viewport,
				),
				corner,
				stroke,
			);

			continue;
		}

		// a point throws every way, so it reads as a ball: the lamp's three
		// circles, for the lamp's reason
		for normal in [Vec3::X, Vec3::Y, Vec3::Z] {
			outline(
				&painter,
				&gizmo::circle(camera, at.position, normal, far, viewport),
				corner,
				stroke,
			);
		}
	}
}

/// The box around everything hanging off each selected group, drawn over the
/// world.
///
/// A group draws nothing and a click cannot land on it, so without this the one
/// thing on screen that says a group is selected is the gizmo in its middle,
/// which says nothing about how far the group reaches. Square to the world, the
/// way the box a group is put in the middle of is, and not a handle: nothing
/// hit-tests it. @ref [`select::bounds`].
fn groups(
	context: &Context,
	world: &World,
	selection: &Selection,
	camera: &Camera,
	viewport: Vec2,
	view: Rect,
) {
	let painter = context
		.layer_painter(LayerId::background())
		.with_clip_rect(view);
	let corner = Vec2::new(view.min.x, view.min.y);
	let stroke = Stroke::new(INK.0, GROUPED);
	let view_projection = camera.view_projection(viewport.x / viewport.y.max(1.0));

	for pick in selection.picks() {
		let Pick::Entity(id) = pick else {
			continue;
		};

		if !select::is_group(world, id) {
			continue;
		}

		let Some((low, high)) = select::bounds(world, &select::descendants(world, id)) else {
			continue;
		};

		let matrix =
			Mat4::from_scale_rotation_translation(high - low, Quat::IDENTITY, (low + high) * 0.5);

		edges(&painter, view_projection, matrix, viewport, corner, stroke);
	}
}

/// The twelve edges of the box a matrix takes the unit cube to, drawn.
///
/// @param corner - where the picture is on the screen
fn edges(
	painter: &Painter,
	view_projection: Mat4,
	matrix: Mat4,
	viewport: Vec2,
	corner: Vec2,
	stroke: Stroke,
) {
	let ends = BOX_CORNERS
		.map(|at| gizmo::project(view_projection, matrix.transform_point3(at), viewport));

	for (from, to) in BOX_EDGES {
		if let (Some(start), Some(end)) = (ends[from], ends[to]) {
			painter.line_segment([spot(start + corner), spot(end + corner)], stroke);
		}
	}
}

/// One closed polyline, moved to where the picture is and drawn.
///
/// An empty list is a shape some part of which is behind the eye. @ref
/// `gizmo::circle`, which is what decides that.
fn outline(painter: &Painter, points: &[Vec2], corner: Vec2, stroke: Stroke) {
	if points.is_empty() {
		return;
	}

	painter.line(
		points
			.iter()
			.map(|point| spot(*point + corner))
			.collect(),
		stroke,
	);
}

/// The gizmo's handles, projected, ready to be drawn and hit.
struct Handles {
	/// Which shape they are.
	tool: Tool,

	/// Where the thing it is attached to is, on screen.
	///
	/// Every arm starts here and every ring is centered on it, so it is the
	/// one number that says whether a gizmo that will not grab is broken or
	/// merely somewhere else.
	middle: Option<Vec2>,

	/// The three arms, for a move or a stretch.
	arms: [Option<(Vec2, Vec2)>; 3],

	/// The three rings, for a turn.
	rings: [Vec<Vec2>; 3],
}

impl Handles {
	/// Projects whichever handles this tool has.
	fn of(camera: &Camera, at: Transform, tool: Tool, viewport: Vec2) -> Self {
		let view = camera.view_projection(viewport.x.max(1.0) / viewport.y.max(1.0));
		let middle = gizmo::project(view, at.position, viewport);

		match tool {
			| Tool::Move | Tool::Size => Self {
				tool,
				middle,
				arms: gizmo::arms(camera, at, viewport),
				rings: [Vec::new(), Vec::new(), Vec::new()],
			},
			| Tool::Turn => Self {
				tool,
				middle,
				arms: [None, None, None],
				rings: Axis::ALL.map(|axis| gizmo::ring(camera, at, axis, viewport)),
			},
		}
	}

	/// Which handle the pointer is on.
	fn under(&self, at: Vec2) -> Option<Axis> {
		match self.tool {
			| Tool::Move | Tool::Size => gizmo::grabbed(&self.arms, at),
			| Tool::Turn => gizmo::grabbed_ring(&self.rings, at),
		}
	}

	/// Draws them, behind every panel and over the world.
	///
	/// The handles were projected against the picture, so the picture's
	/// corner is added back here, and the painter is cut to the picture so
	/// that an arm reaching under a panel stops at its edge.
	///
	/// @param view - where the picture is on the screen, in points
	fn paint(&self, context: &Context, lit: Option<Axis>, view: Rect) {
		let painter = context
			.layer_painter(LayerId::background())
			.with_clip_rect(view);
		let corner = Vec2::new(view.min.x, view.min.y);

		for (axis, arm) in Axis::ALL.into_iter().zip(&self.arms) {
			let Some((start, end)) = *arm else {
				continue;
			};

			let stroke = ink(axis, lit == Some(axis));
			painter.line_segment([spot(start + corner), spot(end + corner)], stroke);

			// a blob for a move and a block for a stretch, so the two tools
			// are told apart by the shape rather than by remembering which key
			// was last pressed.
			if self.tool == Tool::Size {
				let block =
					Rect::from_center_size(spot(end + corner), vec2(TIP * 2.0, TIP * 2.0));
				painter.rect_filled(block, 1.0, stroke.color);
			} else {
				painter.circle_filled(spot(end + corner), TIP, stroke.color);
			}
		}

		for (axis, ring) in Axis::ALL.into_iter().zip(&self.rings) {
			if ring.is_empty() {
				continue;
			}

			let points = ring
				.iter()
				.map(|point| spot(*point + corner))
				.collect();
			painter.line(points, ink(axis, lit == Some(axis)));
		}
	}
}

/// Everything the pointer was doing this frame.
///
/// One struct so that egui's input is read once, under one lock, rather than
/// nine times.
#[derive(Clone, Copy, Debug)]
struct Gestures {
	/// How far the pointer moved, in points.
	drag: Vec2,

	/// How far the wheel turned, in points.
	wheel: f32,

	/// Whether the button that turns the camera is down.
	turning: bool,

	/// Whether the button that slides it is down.
	sliding: bool,

	/// Whether the button that selects went down this frame.
	pressed: bool,

	/// Whether it is down at all.
	down: bool,

	/// Whether it came back up this frame.
	released: bool,

	/// Whether it was a click rather than the start of a drag.
	clicked: bool,

	/// Where the pointer is, in points from the picture's top left corner.
	at: Option<Vec2>,
}

impl Gestures {
	/// Reads the lot.
	///
	/// @param view - where the picture is on the screen, which the pointer is
	/// measured from
	fn read(context: &Context, view: Rect) -> Self {
		context.input(|input| Self {
			drag: Vec2::new(input.pointer.delta().x, input.pointer.delta().y),
			wheel: input.smooth_scroll_delta.y,
			turning: input
				.pointer
				.button_down(PointerButton::Secondary),
			sliding: input.pointer.button_down(PointerButton::Middle),
			pressed: input.pointer.primary_pressed(),
			down: input.pointer.primary_down(),
			released: input.pointer.primary_released(),
			clicked: input
				.pointer
				.button_clicked(PointerButton::Primary),
			at: input
				.pointer
				.interact_pos()
				.map(|pos| Vec2::new(pos.x - view.min.x, pos.y - view.min.y)),
		})
	}
}

/// What the pointer reads on one handle, in the units that handle is measured
/// in.
///
/// A distance along the arm for a move or a stretch, an angle in the ring's own
/// plane for a turn.
fn read(
	camera: &Camera,
	at: Transform,
	axis: Axis,
	tool: Tool,
	point: Vec2,
	viewport: Vec2,
) -> Option<f32> {
	let (from, ray) = aim::ray(camera, point, viewport);

	match tool {
		| Tool::Move | Tool::Size =>
			gizmo::along(at.position, at.rotation * axis.way(), from, ray),
		| Tool::Turn => {
			let (normal, first) = gizmo::plane(at, axis);

			gizmo::around(at.position, normal, first, from, ray)
		},
	}
}
/// A dragged transform put on the grid, if there is one.
///
/// **The grid holds a drag, not only a placement.** One that held only what is
/// put down would be a grid nobody could use a minute later, when the thing
/// needs moving - so a move lands on a line and a size lands on a whole number
/// of cells.
///
/// **Turning is deliberately not snapped.** An angle is a different unit: a
/// step in world units says nothing about degrees, and giving it one would be a
/// second number and a second decision. Unreal keeps its angle snap apart from
/// its grid for the same reason.
///
/// @param put - where the drag has got to
/// @param tool - which of the three is being dragged
/// @param step - the grid, or nothing for none
#[must_use]
fn held(put: Transform, tool: Tool, step: Option<f32>) -> Transform {
	let Some(step) = step else {
		return put;
	};

	match tool {
		| Tool::Move => Transform {
			position: select::snapped(put.position, step),
			..put
		},
		| Tool::Size => Transform {
			scale: select::sized(put.scale, step),
			..put
		},
		| Tool::Turn => put,
	}
}

/// An angle put back into the half turn either side of nothing.
fn wrapped(angle: f32) -> f32 {
	let turn = std::f32::consts::TAU;

	turn.mul_add(-(angle / turn).round(), angle)
}

/// One handle's ink.
fn ink(axis: Axis, lit: bool) -> Stroke {
	let [red, green, blue] = axis.tint();
	let color = if lit { LIT } else { Color32::from_rgb(red, green, blue) };

	Stroke::new(if lit { INK.1 } else { INK.0 }, color)
}

/// A point on the screen, in the type egui draws with.
fn spot(at: Vec2) -> Pos2 { Pos2::new(at.x, at.y) }

/// How big the picture is, in points; never less than a point each way, so
/// that a window squeezed down to its panels divides by nothing.
fn size(view: Rect) -> Vec2 { Vec2::new(view.width().max(1.0), view.height().max(1.0)) }

/// What is under the pointer.
///
/// Both the position and the size come from egui rather than from the window,
/// so that they are in one another's units whatever the display scale is doing.
/// The camera is the one the frame was *drawn* through rather than the one the
/// last step wrote, because what a person clicked on is what they were looking
/// at.
///
/// @param at - where the pointer is, from the picture's corner
/// @param viewport - how big the picture is, in the same units
fn picked(world: &World, at: Vec2, viewport: Vec2) -> Pick {
	let camera = world.render_camera();
	let (from, along) = aim::ray(&camera, at, viewport);

	if along.abs_diff_eq(Vec3::ZERO, f32::EPSILON) {
		return Pick::Nothing;
	}

	// what is painted over the picture is what a click lands on. A mark has no
	// depth at all, and a mesh is picked by its *bounds*, so a box the camera
	// stands inside answers every ray at no distance and would take every
	// click. @ref [`helper`](crate::helper) for the whole of that argument.
	if Editor::helpers(world)
		&& let Some(found) = helper::under(world, &camera, viewport, at)
	{
		return found;
	}

	aim::under(world, from, along)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_grid_holds_a_move_and_a_size_and_leaves_a_turn_alone() {
		let put = Transform {
			position: Vec3::new(1.2, 0.0, -0.4),
			rotation: Quat::from_rotation_y(0.3),
			scale: Vec3::new(1.1, 0.2, 2.4),
		};

		let moved = held(put, Tool::Move, Some(0.5));

		assert_eq!(moved.position, Vec3::new(1.0, 0.0, -0.5), "a move lands on a line");
		assert_eq!(moved.scale, put.scale, "and leaves the size where it was");

		let sized = held(put, Tool::Size, Some(0.5));

		assert_eq!(
			sized.scale,
			Vec3::new(1.0, 0.5, 2.5),
			"a size lands on whole cells, and never on nought"
		);
		assert_eq!(sized.position, put.position, "and leaves the place where it was");

		let turned = held(put, Tool::Turn, Some(0.5));

		assert_eq!(
			turned, put,
			"a turn is not snapped at all: an angle is a different unit and wants a step of its \
			 own"
		);
	}

	#[test]
	fn no_grid_holds_nothing() {
		let put = Transform {
			position: Vec3::new(1.237, -0.9, 0.04),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(0.31, 1.77, 0.02),
		};

		for tool in [Tool::Move, Tool::Turn, Tool::Size] {
			assert_eq!(held(put, tool, None), put, "{tool:?} is left alone with no grid");
		}
	}

	/// The meadow in miniature: a ground an entity strews cubes over, with the
	/// camera looking straight down at it.
	fn strewn_world() -> (World, EntityId) {
		use colby_core::abi::{
			MaterialId, MeshData, MeshVertex, Renderable, STREWING, Strewing, mesh,
		};

		let mut world = World::new();
		world
			.entities
			.declare(&STREWING)
			.expect("a world with nothing declared takes the record");
		world.meshes.insert("meshes/cube", mesh::cube());

		let corner = |x: f32, z: f32| MeshVertex::new(Vec3::new(x, 0.0, z), Vec3::Y, Vec2::ZERO);
		let floor = world.meshes.insert("meshes/floor", MeshData {
			vertices: vec![
				corner(-16.0, -16.0),
				corner(16.0, -16.0),
				corner(16.0, 16.0),
				corner(-16.0, 16.0),
			],
			indices: vec![0, 2, 1, 0, 3, 2],
			..MeshData::default()
		});
		let ground = world.entities.spawn();
		world
			.entities
			.set_renderable(ground, Renderable::of(floor, MaterialId::DEFAULT, Vec3::ONE));

		let grass = world.entities.spawn();
		world.entities.set_renderable(
			grass,
			Renderable::of(colby_core::abi::MeshId::CUBE, MaterialId::DEFAULT, Vec3::ONE),
		);
		world.entities.set_parent(grass, ground);

		if let Some(rule) = world.entities.record_mut(&STREWING, grass) {
			*rule = Strewing { strews: 1, ..Strewing::NONE };
		}

		// straight down the middle, so that the pointer in the middle of the
		// picture is the middle of the ground
		world.camera.position = Vec3::new(0.0, 20.0, 0.0);
		world.camera.target = Vec3::ZERO;
		world.camera.up = Vec3::NEG_Z;
		world.editing = true;

		(world, grass)
	}

	/// One frame of a viewport over a world, with whatever the pointer did.
	fn frame(
		context: &Context,
		viewport: &mut Viewport,
		world: &mut World,
		selection: &Selection,
		history: &mut History,
		events: Vec<egui::Event>,
	) {
		let view = Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0));
		let mut output = context.run_ui(
			egui::RawInput {
				screen_rect: Some(view),
				events,
				..Default::default()
			},
			// nothing is laid out in the middle of the screen: what the
			// viewport reads is what no panel wanted
			|_| {},
		);

		viewport.run(context, world, selection, view, history);
		// **the frame is over for the history**, exactly as `Panels::frame`
		// ends it: a gesture nothing wrote to this frame is a record now. A
		// helper that left this out would make every frame of a test one
		// gesture however long it rested, which is the one thing a stroke has
		// to be driven against.
		let _wrote = history.settle(world);
		output.textures_delta.clear();
	}

	/// What the pointer does over one place in the picture.
	fn pointing(at: Pos2, pressed: Option<bool>) -> Vec<egui::Event> {
		let mut events = vec![egui::Event::PointerMoved(at)];

		if let Some(down) = pressed {
			events.push(egui::Event::PointerButton {
				pos: at,
				button: PointerButton::Primary,
				pressed: down,
				modifiers: egui::Modifiers::NONE,
			});
		}

		events
	}

	#[test]
	fn a_stroke_over_the_ground_paints_the_selected_strewing_and_is_one_step_back() {
		let (mut world, grass) = strewn_world();
		let context = Context::default();
		let mut viewport = Viewport::default();
		let mut selection = Selection::default();
		let mut history = History::default();

		selection.set(&world, Pick::Entity(grass));
		viewport.set_brushing(true);

		// down in the middle of the picture, dragged a little, and up
		frame(
			&context,
			&mut viewport,
			&mut world,
			&selection,
			&mut history,
			pointing(Pos2::new(400.0, 300.0), Some(true)),
		);

		for step in 1..=6 {
			let along = f32::from(u8::try_from(step).expect("six steps")) * 12.0;

			frame(
				&context,
				&mut viewport,
				&mut world,
				&selection,
				&mut history,
				pointing(Pos2::new(400.0 + along, 300.0), None),
			);

			// **and a frame in which the pointer does not move**, which is
			// what a hand resting halfway through a stroke does. A stroke dabs
			// by distance, so a frame like this writes nothing at all - and a
			// record closes on the first frame in which nothing writes.
			frame(&context, &mut viewport, &mut world, &selection, &mut history, Vec::new());
		}

		frame(
			&context,
			&mut viewport,
			&mut world,
			&selection,
			&mut history,
			pointing(Pos2::new(472.0, 300.0), Some(false)),
		);

		let mask = world
			.entities
			.mask(grass)
			.expect("the stroke made one");

		assert!(mask.painted() > 0.0, "the stroke took some of the field away");
		assert!(mask.at(0.0, 0.0) < 1.0, "the middle of the picture is where it started");
		assert_eq!(mask.at(15.0, 15.0).to_bits(), 1.0_f32.to_bits(), "the far corner is open");

		// a record opens on the first frame that writes and closes on the
		// first frame that does not, so the whole drag is one step back
		history.settle(&world);
		history.settle(&world);

		assert_eq!(history.undoable(), Some("paint"), "the whole stroke is one step");

		// and **one** step whose world is the one before the stroke, with
		// nothing painted at all. Until a driven window said otherwise, a
		// stroke whose hand slowed down was several records, and each record's
		// world already had a dab in it - both of which pass every assertion
		// above.
		let back = history
			.undo(&world)
			.expect("there is a step to take");

		assert!(
			back.things
				.iter()
				.all(|thing| thing.mask.is_none()),
			"the step back is the world before the stroke, not one part way through it"
		);
		assert_eq!(history.undoable(), None, "and there is nothing behind it");
	}

	#[test]
	fn a_stroke_with_the_brush_away_moves_nothing_and_the_gizmo_takes_it_back() {
		let (mut world, grass) = strewn_world();
		let context = Context::default();
		let mut viewport = Viewport::default();
		let mut selection = Selection::default();
		let mut history = History::default();

		selection.set(&world, Pick::Entity(grass));

		for events in [
			pointing(Pos2::new(400.0, 300.0), Some(true)),
			pointing(Pos2::new(430.0, 300.0), None),
			pointing(Pos2::new(430.0, 300.0), Some(false)),
		] {
			frame(&context, &mut viewport, &mut world, &selection, &mut history, events);
		}

		assert!(world.entities.mask(grass).is_none(), "nothing painted with the brush away");

		// and the brush put out and taken back again paints nothing either
		viewport.set_brushing(true);
		viewport.set_tool(Tool::Move);

		for events in [
			pointing(Pos2::new(400.0, 300.0), Some(true)),
			pointing(Pos2::new(430.0, 300.0), None),
			pointing(Pos2::new(430.0, 300.0), Some(false)),
		] {
			frame(&context, &mut viewport, &mut world, &selection, &mut history, events);
		}

		assert!(world.entities.mask(grass).is_none(), "and none after the gizmo took it back");
	}

	#[test]
	fn nothing_is_grabbed_while_the_brush_is_out() {
		// the other half of the brush being a mode: a drag in the middle of
		// the screen belongs to the stroke, so the gizmo's own handles are not
		// drawn and cannot be held. The ground is what is selected here,
		// because a strewing draws nothing where it stands and so has no
		// handles of its own to be sure about.
		let (mut world, grass) = strewn_world();
		let ground = world.entities.parent(grass);
		let context = Context::default();
		let mut viewport = Viewport::default();
		let mut selection = Selection::default();
		let mut history = History::default();

		selection.set(&world, Pick::Entity(ground));

		let stood = world
			.entities
			.transform(ground)
			.copied()
			.expect("it stands somewhere");

		// the gizmo stands where the thing does, which is the middle of the
		// picture, and its arms reach out from there
		let drag = |viewport: &mut Viewport, world: &mut World, history: &mut History| {
			for events in [
				pointing(Pos2::new(440.0, 300.0), Some(true)),
				pointing(Pos2::new(500.0, 300.0), None),
				pointing(Pos2::new(500.0, 300.0), Some(false)),
			] {
				frame(&context, viewport, world, &selection, history, events);
			}
		};

		viewport.set_brushing(true);
		drag(&mut viewport, &mut world, &mut history);

		assert_eq!(
			world.entities.transform(ground).copied(),
			Some(stood),
			"the ground did not move: there was no handle to grab"
		);

		// and the same drag with the brush away does move it, which is what
		// says the drag was on a handle at all
		viewport.set_brushing(false);
		drag(&mut viewport, &mut world, &mut history);

		assert_ne!(
			world.entities.transform(ground).copied(),
			Some(stood),
			"the same drag with the gizmo out moves it"
		);
	}

	#[test]
	fn an_angle_comes_back_inside_the_half_turn_either_side_of_nothing() {
		let turn = std::f32::consts::TAU;

		assert!((wrapped(0.3) - 0.3).abs() < 1.0e-5, "a small one is itself");
		assert!(
			(wrapped(turn - 0.2) + 0.2).abs() < 1.0e-4,
			"nearly the whole way round forwards is a little way back"
		);
		assert!(
			(wrapped(0.2 - turn) - 0.2).abs() < 1.0e-4,
			"and the other way round is a little way forwards"
		);
		assert!(wrapped(turn * 3.0).abs() < 1.0e-4, "whole turns are nothing at all");
	}
}
