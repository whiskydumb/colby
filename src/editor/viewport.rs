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
//! - **right drag** turns the camera around what it is looking at;
//! - **middle drag** slides what it is looking at across the view;
//! - **the wheel** moves closer and further;
//! - **left drag on a handle** moves, turns or stretches what is selected;
//! - **left click on anything else** selects whatever is under the pointer.
//!
//! The order of those last two matters: a press that lands on a handle starts a
//! drag and is *not* also a selection, or grabbing the arm of the thing you
//! have selected would immediately select whatever is behind it.
//!
//! The camera is the editor's **only while the world is being edited**. While
//! it is being played the game owns it and this holds nothing at all, so that
//! resuming an edit takes the view the game left rather than the one from
//! before it started.
//!
//! Everything here is egui talking to [`aim`](crate::aim) and
//! [`gizmo`](crate::gizmo), which is where the arithmetic and the tests are.

use colby_core::{
	abi::{Camera, Transform, World},
	glam::{Vec2, Vec3},
	trace,
};
use egui::{Color32, Context, Key, LayerId, PointerButton, Pos2, Rect, Stroke, vec2};

use crate::{
	aim::{self, View},
	gizmo::{self, Axis, Tool},
	history::History,
	select::{self, Pick, Selection},
};

/// How thick a handle is drawn, and how much thicker the grabbed one is.
const INK: (f32, f32) = (2.0, 3.5);

/// How wide the blob on the end of an arm is, in points.
const TIP: f32 = 4.5;

/// What a handle under the pointer is drawn in.
const LIT: Color32 = Color32::from_rgb(255, 235, 120);

/// What the pointer does outside every window.
#[derive(Debug, Default)]
pub(crate) struct Viewport {
	/// The editor's camera, while it has one.
	view: Option<View>,

	/// Which of the three things the gizmo is doing.
	tool: Tool,

	/// The drag in progress, if one is.
	grab: Option<Grab>,
}

/// A drag of one handle, from the moment it was grabbed.
#[derive(Clone, Debug)]
struct Grab {
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
	pub(crate) const fn set_tool(&mut self, tool: Tool) { self.tool = tool; }

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

		let dragging = self.gizmo(context, world, selection, busy, held, view, history);

		let found = if busy || dragging || !held.clicked {
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
			self.tool = tool;
		}
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

	/// Draws the gizmo and applies whatever is being dragged with it.
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
		let pick = selection.at();
		let Some(at) = select::transform(world, pick) else {
			self.grab = None;

			return false;
		};

		let camera = world.render_camera();
		let viewport = size(view);
		let handles = Handles::of(&camera, at, self.tool, viewport);
		let over = held.at.and_then(|point| handles.under(point));

		if held.released {
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
				at = ?held.at,
				middle = ?handles.middle,
				tool = self.tool.word(),
				"the pointer went down in the world"
			);
		}

		if held.pressed
			&& !busy && let (Some(axis), Some(point)) = (over, held.at)
		{
			// everything else selected, with where it stands now: the drag
			// lands them from here, however long it lasts
			let others = selection
				.others()
				.into_iter()
				.filter_map(|other| select::transform(world, other).map(|was| (other, was)))
				.collect();

			self.hold(&camera, at, axis, point, viewport, others);
		}

		if held.down
			&& let Some(point) = held.at
		{
			self.pull(world, pick, &camera, point, viewport, history);
		}

		handles.paint(
			context,
			self.grab
				.as_ref()
				.map_or(over, |grab| Some(grab.axis)),
			view,
		);

		self.grab.is_some() || (over.is_some() && !busy)
	}

	/// Takes hold of a handle.
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

		self.grab = Some(Grab {
			axis,
			tool: self.tool,
			from: at,
			arm: gizmo::reach(camera, at.position, viewport),
			start,
			last: start,
			total: 0.0,
			others,
		});
	}

	/// Applies the drag in progress.
	///
	/// Written down first, every frame of it: the record opens on the first
	/// frame and stays open while the drag writes, so the whole drag is one
	/// step back. @ref [`History::begin`].
	fn pull(
		&mut self,
		world: &mut World,
		pick: Pick,
		camera: &Camera,
		point: Vec2,
		viewport: Vec2,
		history: &mut History,
	) {
		let Some(grab) = self.grab.as_mut() else {
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
		let (tool, from, others) = (grab.tool, grab.from, grab.others.clone());

		history.begin(tool.word(), world);
		select::drag_all(world, pick, from, put, &others);
	}
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

	aim::under(world, from, along)
}

#[cfg(test)]
mod tests {
	use super::*;

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
