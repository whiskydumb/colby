//! The part of the screen that is not a window.
//!
//! Three gestures and a click, and none of them is a widget: what the pointer
//! does out here is what it does when no panel wanted it. That is the whole of
//! the guard - `Context::egui_wants_pointer_input`, whose own documentation
//! says it is false exactly when "you may be interested in what it is doing" -
//! and everything below is skipped while it is true. It also stays false for a
//! drag that *began* outside a window and has since crossed one, which is the
//! behavior a camera drag needs and the reason this is the right question to
//! ask rather than "is the pointer over a window".
//!
//! - **right drag** turns the camera around what it is looking at;
//! - **middle drag** slides what it is looking at across the view;
//! - **the wheel** moves closer and further;
//! - **left click** selects whatever is under the pointer, or nothing.
//!
//! The camera is the editor's **only while the world is being edited**. While
//! it is being played the game owns it and this holds nothing at all, so that
//! resuming an edit takes the view the game left rather than the one from
//! before it started.
//!
//! Everything here is egui talking to [`aim`](crate::aim), which is where the
//! arithmetic and the tests are.

use colby_core::{
	abi::World,
	glam::{Vec2, Vec3},
	trace,
};
use egui::{Context, PointerButton};

use crate::{
	aim::{self, View},
	select::Pick,
};

/// What the pointer does outside every window.
#[derive(Debug, Default)]
pub(crate) struct Viewport {
	/// The editor's camera, while it has one.
	view: Option<View>,
}

impl Viewport {
	/// Drives the camera and answers what was clicked.
	///
	/// @param context - egui, mid-frame
	/// @param world - read for what is in it, written for the camera
	/// @return what a click landed on, or nothing at all if there was no click.
	/// A click on empty space answers [`Pick::Nothing`], which is a different
	/// thing from not having clicked and is what deselects
	pub(crate) fn run(&mut self, context: &Context, world: &mut World) -> Option<Pick> {
		if !world.editing {
			// the game's camera again. Whatever orbit this was holding
			// described a view from before the game started moving it.
			self.view = None;

			return None;
		}

		let mut view = View::taken(self.view, &world.camera);

		// a panel wanting the pointer means the pointer is a panel's. Read
		// before anything is done with it, so the camera cannot be turned by a
		// drag that started on a slider.
		let busy = context.egui_wants_pointer_input();
		let held = context.input(|input| Gestures {
			drag: Vec2::new(input.pointer.delta().x, input.pointer.delta().y),
			wheel: input.smooth_scroll_delta.y,
			turning: input
				.pointer
				.button_down(PointerButton::Secondary),
			sliding: input.pointer.button_down(PointerButton::Middle),
			clicked: input
				.pointer
				.button_clicked(PointerButton::Primary),
			at: input
				.pointer
				.interact_pos()
				.map(|pos| Vec2::new(pos.x, pos.y)),
		});

		if !busy {
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

		let found = if busy {
			None
		} else {
			held.at.map(|at| picked(context, world, at))
		};

		if held.clicked {
			// the same line the game's interface writes when a click reaches a
			// named box, and for the same reason: the chain from a pointer
			// through a projection to a thing in the world has no other way of
			// being checked, because what it produces is a highlighted row
			// that only a pair of eyes can read. `busy` is on it because the
			// commonest answer to "why did my click do nothing" is that a
			// panel was under it.
			trace!(busy, at = ?held.at, ?found, "a click in the world");
		}

		if held.clicked { found } else { None }
	}
}

/// Everything the pointer was doing this frame.
///
/// One struct so that egui's input is read once, under one lock, rather than
/// six times.
struct Gestures {
	/// How far the pointer moved, in points.
	drag: Vec2,

	/// How far the wheel turned, in points.
	wheel: f32,

	/// Whether the button that turns the camera is down.
	turning: bool,

	/// Whether the button that slides it is down.
	sliding: bool,

	/// Whether the button that selects was clicked.
	clicked: bool,

	/// Where the pointer is, in points from the top left.
	at: Option<Vec2>,
}

/// What is under the pointer.
///
/// Both the position and the size come from egui rather than from the window,
/// so that they are in one another's units whatever the display scale is doing.
/// The camera is the one the frame was *drawn* through rather than the one the
/// last step wrote, because what a person clicked on is what they were looking
/// at.
fn picked(context: &Context, world: &World, at: Vec2) -> Pick {
	let rect = context.viewport_rect();
	let viewport = Vec2::new(rect.width(), rect.height());
	let camera = world.render_camera();
	let (from, along) = aim::ray(&camera, at - Vec2::new(rect.left(), rect.top()), viewport);

	if along.abs_diff_eq(Vec3::ZERO, f32::EPSILON) {
		return Pick::Nothing;
	}

	aim::under(world, from, along)
}
