//! The arms and rings a person drags to move something, and the arithmetic
//! under them.
//!
//! **Drawn and hit-tested in screen space, not in the world.** A gizmo is a
//! tool rather than a thing in the scene: it has to be the same size however
//! far away what it is attached to is, it has to be grabbable at the width a
//! finger expects rather than at the width it happens to be drawn, and it must
//! not be hidden behind whatever it is attached to. All three of those are
//! properties of pixels, so the arms are projected and the pointer is compared
//! against them in pixels - which is what the two engines that hit-test gizmos
//! analytically both do, at eight and nine and a half pixels respectively.
//! Eight is what is used here.
//!
//! The alternative is a second render pass writing an identifier per pixel,
//! which is exact and needs a read back from the GPU, a stall, and a shape
//! nothing can unit-test. That was rejected. @ref `colby-known-gaps` for what
//! the pixel test costs.
//!
//! **The arms are the thing's own axes.** There is no global-or-local switch:
//! the arms point along the object's rotation, so a drag moves it the way the
//! arm it grabbed is pointing. That is the only choice that is correct for all
//! three tools - a scale along a world axis is not something a transform of
//! translate, rotate and scale can even represent once the thing is turned.
//!
//! **A drag is measured against where it started, never against the frame
//! before.** The transform at the moment the pointer went down is kept, and
//! every frame computes the whole change from it. Accumulating per-frame
//! deltas instead would drift, and would make a drag that crossed a slow frame
//! land somewhere else than one that did not.

use colby_core::{
	abi::{Camera, Transform},
	glam::{Mat4, Quat, Vec2, Vec3},
};

/// How long an arm is on screen, in points.
pub(crate) const ARM: f32 = 80.0;

/// How near the pointer has to be to grab something, in points.
///
/// The same order as the two analytic gizmo pickers in the field use for a
/// line. Wider than it looks: an arm is one pixel of ink and eight of reach.
pub(crate) const GRAB: f32 = 8.0;

/// How many straight pieces a ring is drawn with.
const RING_STEPS: usize = 32;

/// How square-on a ring has to be to the eye to be worth having.
///
/// The cosine between the way it faces and the way it is looked at. A ring
/// below this is edge on: on screen it is a line rather than a circle, and a
/// drag on it cannot be measured at all - [`around`] refuses one, because
/// there is no angle in a plane the ray never leaves. A ring that could be
/// grabbed and not dragged is a dead handle, so an edge-on one is neither
/// drawn nor grabbable.
const RING_FACING: f32 = 0.08;

/// Which of the three axes a handle belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Axis {
	/// The first one, drawn red.
	X,

	/// The second, drawn green.
	Y,

	/// The third, drawn blue.
	Z,
}

impl Axis {
	/// All three, in the order everything here returns them in.
	pub(crate) const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

	/// Which way it points, before the thing's own rotation is applied.
	pub(crate) const fn way(self) -> Vec3 {
		match self {
			| Self::X => Vec3::X,
			| Self::Y => Vec3::Y,
			| Self::Z => Vec3::Z,
		}
	}

	/// Its place in a three-component vector.
	pub(crate) const fn place(self) -> usize {
		match self {
			| Self::X => 0,
			| Self::Y => 1,
			| Self::Z => 2,
		}
	}

	/// What it is drawn in, as sRGB bytes.
	///
	/// The convention every editor shares, and the one the debug renderer
	/// already uses for its own axes: x is red, y is green, z is blue.
	pub(crate) const fn tint(self) -> [u8; 3] {
		match self {
			| Self::X => [230, 70, 70],
			| Self::Y => [90, 210, 90],
			| Self::Z => [80, 130, 240],
		}
	}
}

/// What the gizmo does with a drag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Tool {
	/// Slide it along an axis.
	#[default]
	Move,

	/// Turn it about an axis.
	Turn,

	/// Stretch it along an axis.
	Size,
}

impl Tool {
	/// What to call it on screen.
	pub(crate) const fn word(self) -> &'static str {
		match self {
			| Self::Move => "move",
			| Self::Turn => "turn",
			| Self::Size => "size",
		}
	}
}

/// Where a point in the world lands on the screen, in points.
///
/// @return `None` for a point on or behind the plane through the eye, where
/// the divide is not a wrong answer but a pair of values that pass every
/// comparison afterwards
pub(crate) fn project(view_projection: Mat4, at: Vec3, viewport: Vec2) -> Option<Vec2> {
	let clip = view_projection * at.extend(1.0);

	if clip.w <= f32::EPSILON {
		return None;
	}

	let ndc = clip.truncate() / clip.w;

	Some(Vec2::new(
		ndc.x.mul_add(0.5, 0.5) * viewport.x,
		// flipped: clip space has y up and a screen counts it down.
		ndc.y.mul_add(-0.5, 0.5) * viewport.y,
	))
}

/// How long an arm has to be in world units to be [`ARM`] points on screen.
///
/// Worked out from the height of the view at the gizmo's own distance, which
/// is exact in the middle of the picture and near enough at the edges. Without
/// this a gizmo on something far away is a dot and one up close fills the
/// window.
///
/// @param camera - what the picture is drawn through
/// @param at - where the gizmo is
/// @param viewport - how tall the picture is, in points
pub(crate) fn reach(camera: &Camera, at: Vec3, viewport: Vec2) -> f32 {
	let away = (at - camera.position).length().max(0.01);
	let half = (camera.fov_y * 0.5).tan();

	away * half * 2.0 * (ARM / viewport.y.max(1.0))
}

/// The three arms, projected, in the order of [`Axis::ALL`].
///
/// @return the start and end of each on screen, or `None` for an arm that
/// cannot be put there at all
pub(crate) fn arms(camera: &Camera, at: Transform, viewport: Vec2) -> [Option<(Vec2, Vec2)>; 3] {
	let view = camera.view_projection(ratio(viewport));
	let long = reach(camera, at.position, viewport);
	let Some(middle) = project(view, at.position, viewport) else {
		return [None, None, None];
	};

	Axis::ALL.map(|axis| {
		let tip = at.position + at.rotation * axis.way() * long;

		project(view, tip, viewport).map(|end| (middle, end))
	})
}

/// The plane one ring lies in, in world space.
///
/// One definition for the drawing and for the drag: [`ring`] walks the circle
/// with these two and [`around`] measures the pointer's angle against the same
/// pair, so the angle a person sees and the angle the arithmetic uses cannot
/// drift apart.
///
/// @param at - where the gizmo is and how it is turned
/// @param axis - which ring
/// @return the way the ring faces, and the direction its angles start from;
/// both unit vectors
pub(crate) fn plane(at: Transform, axis: Axis) -> (Vec3, Vec3) {
	let turn = at.rotation * Quat::from_rotation_arc(Vec3::Z, axis.way());

	((turn * Vec3::Z).normalize_or(Vec3::Z), (turn * Vec3::X).normalize_or(Vec3::X))
}

/// One ring, projected, in the plane an axis is the normal of.
///
/// @return the points of a closed polyline, or an empty list if any of it
/// cannot be put on the screen
pub(crate) fn ring(camera: &Camera, at: Transform, axis: Axis, viewport: Vec2) -> Vec<Vec2> {
	let view = camera.view_projection(ratio(viewport));
	let long = reach(camera, at.position, viewport);
	let (normal, first) = plane(at, axis);

	let looked = (at.position - camera.position).normalize_or(Vec3::NEG_Z);
	if looked.dot(normal).abs() < RING_FACING {
		return Vec::new();
	}

	let second = normal.cross(first);
	let mut points = Vec::with_capacity(RING_STEPS + 1);

	for step in 0..=RING_STEPS {
		let angle = std::f32::consts::TAU * step_of(step);
		let (sin, cos) = angle.sin_cos();
		let on = at.position + (first * cos + second * sin) * long;

		let Some(point) = project(view, on, viewport) else {
			// one point behind the eye and the whole ring is a shape whose
			// straight pieces cross the screen. Refusing the lot is the honest
			// answer and it is what a gizmo behind the camera should be.
			return Vec::new();
		};

		points.push(point);
	}

	points
}

/// Which arm the pointer is on, if it is on one.
///
/// The nearest within [`GRAB`], so that two arms crossing on screen resolve to
/// the one actually under the pointer rather than to whichever was tested
/// first.
///
/// @param arms - what [`arms`] returned
/// @param at - where the pointer is
pub(crate) fn grabbed(arms: &[Option<(Vec2, Vec2)>; 3], at: Vec2) -> Option<Axis> {
	let mut nearest = GRAB;
	let mut found = None;

	for (axis, arm) in Axis::ALL.into_iter().zip(arms) {
		let Some((start, end)) = *arm else {
			continue;
		};

		let away = to_segment(at, start, end);
		if away < nearest {
			nearest = away;
			found = Some(axis);
		}
	}

	found
}

/// Which ring the pointer is on, if it is on one.
///
/// @param rings - what [`ring`] returned for each axis, in order
/// @param at - where the pointer is
pub(crate) fn grabbed_ring(rings: &[Vec<Vec2>; 3], at: Vec2) -> Option<Axis> {
	let mut nearest = GRAB;
	let mut found = None;

	for (axis, points) in Axis::ALL.into_iter().zip(rings) {
		for pair in points.windows(2) {
			let (Some(&start), Some(&end)) = (pair.first(), pair.get(1)) else {
				continue;
			};

			let away = to_segment(at, start, end);
			if away < nearest {
				nearest = away;
				found = Some(axis);
			}
		}
	}

	found
}

/// How far a point is from a line between two others.
///
/// @return the distance in whatever units the three are in
pub(crate) fn to_segment(at: Vec2, start: Vec2, end: Vec2) -> f32 {
	let along = end - start;
	let length = along.length_squared();

	if length < f32::EPSILON {
		return at.distance(start);
	}

	let share = ((at - start).dot(along) / length).clamp(0.0, 1.0);

	at.distance(start + along * share)
}

/// How far along a line the point nearest a ray is.
///
/// The standard closest-approach of two lines, which is what turns a pointer
/// into a distance along the arm it grabbed. A drag then reads this once when
/// it starts and again every frame, and the difference is how far the thing
/// has moved.
///
/// @param origin - a point on the line
/// @param way - which way the line goes; need not be a unit vector
/// @param from - where the ray starts
/// @param ray - which way it goes
/// @return the multiple of `way` from `origin`, or nothing if the two are
/// parallel and there is no single nearest point
pub(crate) fn along(origin: Vec3, way: Vec3, from: Vec3, ray: Vec3) -> Option<f32> {
	let between = origin - from;
	let (aa, ab, bb) = (way.dot(way), way.dot(ray), ray.dot(ray));
	let under = aa.mul_add(bb, -(ab * ab));

	// zero when the two are parallel, and near zero when they are nearly so -
	// which is a drag along an arm pointing straight at the eye, where a pixel
	// of pointer movement is an unbounded distance in the world.
	if under.abs() < 1.0e-6 {
		return None;
	}

	let (ad, bd) = (way.dot(between), ray.dot(between));

	Some(bb.mul_add(-ad, ab * bd) / under)
}

/// What angle a ray meets a disc at, measured in the disc's own plane.
///
/// @param center - the middle of the disc
/// @param normal - which way it faces; a unit vector
/// @param first - the direction the angle is measured from; in the plane, a
/// unit vector
/// @param from - where the ray starts
/// @param ray - which way it goes
/// @return the angle in radians, or nothing if the ray runs along the disc and
/// never meets it
pub(crate) fn around(
	center: Vec3,
	normal: Vec3,
	first: Vec3,
	from: Vec3,
	ray: Vec3,
) -> Option<f32> {
	let facing = ray.dot(normal);

	// edge on. A ring seen edge on is a line, and where along it the pointer
	// is says nothing about an angle.
	if facing.abs() < 1.0e-4 {
		return None;
	}

	let reached = (center - from).dot(normal) / facing;
	let on = from + ray * reached - center;
	let second = normal.cross(first);

	Some(on.dot(second).atan2(on.dot(first)))
}

/// Where a slide along an arm puts something.
///
/// @param start - the transform the drag began at
/// @param axis - which arm
/// @param far - how far along it, in world units
pub(crate) fn moved(start: Transform, axis: Axis, far: f32) -> Transform {
	let mut moved = start;
	moved.position += start.rotation * axis.way() * far;

	moved
}

/// What a turn about an arm makes something.
///
/// Applied on the right, so the axis is the thing's own rather than the
/// world's - which is what makes the ring that was grabbed the ring it turns
/// about, whatever the thing was already doing.
///
/// @param start - the transform the drag began at
/// @param axis - which ring
/// @param angle - how far round, in radians
pub(crate) fn turned(start: Transform, axis: Axis, angle: f32) -> Transform {
	let mut turned = start;
	turned.rotation = (start.rotation * Quat::from_axis_angle(axis.way(), angle)).normalize();

	turned
}

/// What a stretch along an arm makes something.
///
/// @param start - the transform the drag began at
/// @param axis - which arm
/// @param factor - what to multiply that axis of the scale by
pub(crate) fn sized(start: Transform, axis: Axis, factor: f32) -> Transform {
	let mut sized = start;
	let place = axis.place();
	let was = sized.scale.to_array()[place];

	// held above zero rather than allowed through it: a scale of zero is a
	// matrix with no inverse and a thing that cannot be picked or unstretched
	// afterwards, and a negative one turns the geometry inside out.
	let mut axes = sized.scale.to_array();
	axes[place] = (was * factor).max(1.0e-3);
	sized.scale = Vec3::from_array(axes);

	sized
}

/// How far round a ring one step is.
fn step_of(step: usize) -> f32 {
	let step = u16::try_from(step).unwrap_or(u16::MAX);
	let steps = u16::try_from(RING_STEPS).unwrap_or(u16::MAX);

	f32::from(step) / f32::from(steps.max(1))
}

/// The picture's width over its height.
fn ratio(viewport: Vec2) -> f32 { viewport.x.max(1.0) / viewport.y.max(1.0) }

#[cfg(test)]
mod tests {
	use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

	use super::*;

	/// A camera nine back along z, looking at the origin.
	fn watching() -> Camera {
		let mut camera = Camera::DEFAULT;
		camera.target = Vec3::ZERO;
		camera.position = Vec3::Z * 9.0;

		camera
	}

	/// The size of the picture the tests project into.
	const VIEW: Vec2 = Vec2::new(1280.0, 720.0);

	#[test]
	fn the_middle_of_the_world_lands_in_the_middle_of_the_picture() {
		let camera = watching();
		let view = camera.view_projection(ratio(VIEW));
		let middle = project(view, Vec3::ZERO, VIEW).expect("it is in front of the camera");

		assert!(middle.abs_diff_eq(VIEW * 0.5, 0.5), "{middle} against {}", VIEW * 0.5);
	}

	#[test]
	fn something_above_the_middle_lands_higher_up_the_screen() {
		let camera = watching();
		let view = camera.view_projection(ratio(VIEW));
		let up = project(view, Vec3::Y, VIEW).expect("in front");
		let right = project(view, Vec3::X, VIEW).expect("in front");

		assert!(up.y < VIEW.y * 0.5, "up the world is up the screen: {up}");
		assert!(right.x > VIEW.x * 0.5, "and right is right: {right}");
	}

	#[test]
	fn a_point_behind_the_eye_has_nowhere_to_land() {
		let camera = watching();
		let view = camera.view_projection(ratio(VIEW));

		assert!(project(view, Vec3::Z * 20.0, VIEW).is_none(), "behind");
		assert!(
			project(view, camera.position, VIEW).is_none(),
			"and the eye itself, which is the divide by zero"
		);
	}

	#[test]
	fn an_arm_is_the_same_length_on_screen_however_far_away_it_is() {
		let camera = watching();
		let near = arms(&camera, Transform::at(Vec3::ZERO), VIEW);

		let mut far_camera = watching();
		far_camera.position = Vec3::Z * 60.0;
		let far = arms(&far_camera, Transform::at(Vec3::ZERO), VIEW);

		let length =
			|pair: Option<(Vec2, Vec2)>| pair.map_or(0.0, |(start, end)| start.distance(end));

		let (near_x, far_x) = (length(near[0]), length(far[0]));

		assert!(near_x > 1.0, "there is an arm at all: {near_x}");
		assert!(
			(near_x - far_x).abs() < 2.0,
			"and it is the same length from seven times as far: {near_x} against {far_x}"
		);
	}

	#[test]
	fn the_arms_point_the_way_the_thing_is_turned() {
		let camera = watching();
		let mut turned = Transform::at(Vec3::ZERO);
		turned.rotation = Quat::from_rotation_z(FRAC_PI_2);

		let straight = arms(&camera, Transform::at(Vec3::ZERO), VIEW);
		let after = arms(&camera, turned, VIEW);

		let (Some((from, straight_x)), Some((_, turned_x))) = (straight[0], after[0]) else {
			panic!("both arms are on screen");
		};

		assert!(straight_x.x > from.x + 10.0, "unturned, the first arm goes right");
		assert!(
			turned_x.y < from.y - 10.0,
			"and a quarter turn about z sends it up the screen instead: {turned_x}"
		);
	}

	#[test]
	fn the_pointer_grabs_the_arm_it_is_nearest() {
		let camera = watching();
		let at = Transform::at(Vec3::ZERO);
		let found = arms(&camera, at, VIEW);
		let Some((middle, tip)) = found[0] else {
			panic!("the first arm is on screen");
		};

		let along_it = middle + (tip - middle) * 0.6;

		assert_eq!(grabbed(&found, along_it), Some(Axis::X), "on the arm");
		assert_eq!(
			grabbed(&found, along_it + Vec2::new(0.0, GRAB * 0.5)),
			Some(Axis::X),
			"and within reach of it"
		);
		assert_eq!(
			grabbed(&found, along_it + Vec2::new(0.0, GRAB * 3.0)),
			None,
			"and not out beyond that"
		);
	}

	#[test]
	fn two_arms_crossing_resolve_to_the_nearer_one() {
		// built by hand rather than projected, because the point is the choice
		// between two arms both within reach and not the geometry that put
		// them there. Both start together and end six points apart; a pointer
		// five above the first is one below the second.
		let arms = [
			Some((Vec2::ZERO, Vec2::new(100.0, 0.0))),
			Some((Vec2::ZERO, Vec2::new(100.0, 6.0))),
			None,
		];

		assert_eq!(
			grabbed(&arms, Vec2::new(50.0, 5.0)),
			Some(Axis::Y),
			"the second is nearer, and it is second"
		);
		assert_eq!(
			grabbed(&arms, Vec2::new(50.0, 0.5)),
			Some(Axis::X),
			"and the first when the pointer is down on it"
		);
	}

	#[test]
	fn a_ring_lies_in_the_plane_the_thing_is_turned_into() {
		let mut turned = Transform::IDENTITY;
		turned.rotation = Quat::from_rotation_z(FRAC_PI_2);

		let (straight, _) = plane(Transform::IDENTITY, Axis::X);
		let (after, first) = plane(turned, Axis::X);

		assert!(straight.abs_diff_eq(Vec3::X, 1.0e-4), "unturned it faces along x");
		assert!(
			after.abs_diff_eq(Vec3::Y, 1.0e-3),
			"and a quarter turn about z sends it along y: {after}"
		);
		assert!(
			after.dot(first).abs() < 1.0e-4,
			"and the angles are measured from something in the plane, not out of it"
		);
	}

	#[test]
	fn an_arm_that_cannot_be_drawn_cannot_be_grabbed() {
		let arms = [None, None, None];

		assert_eq!(grabbed(&arms, Vec2::new(640.0, 360.0)), None);
	}

	#[test]
	fn a_ring_closes_and_is_grabbable_around_its_rim() {
		let camera = watching();
		let at = Transform::at(Vec3::ZERO);
		let rings = Axis::ALL.map(|axis| ring(&camera, at, axis, VIEW));

		assert_eq!(rings[2].len(), RING_STEPS + 1, "a ring is a closed polyline");
		assert!(
			rings[2][0].abs_diff_eq(rings[2][RING_STEPS], 0.01),
			"whose last point is its first"
		);

		// the camera is straight down z, so the other two rings are edge on
		// and are neither drawn nor grabbable. Without that they would be
		// two lines lying across this one's rim, and the pointer would grab
		// a ring it could not then turn.
		assert!(rings[0].is_empty(), "the first ring is edge on");
		assert!(rings[1].is_empty(), "and so is the second");

		let rim = VIEW * 0.5 + Vec2::new(ARM, 0.0);

		assert_eq!(grabbed_ring(&rings, rim), Some(Axis::Z), "the rim is grabbable");
		assert_eq!(
			grabbed_ring(&rings, VIEW * 0.5),
			None,
			"and the middle of it is not, because a ring is not a disc"
		);
	}

	#[test]
	fn a_point_is_no_distance_from_a_line_it_is_on() {
		let start = Vec2::new(10.0, 10.0);
		let end = Vec2::new(30.0, 10.0);

		assert!(to_segment(Vec2::new(20.0, 10.0), start, end).abs() < 1.0e-4, "on it");
		assert!(
			(to_segment(Vec2::new(20.0, 15.0), start, end) - 5.0).abs() < 1.0e-4,
			"beside it"
		);
		assert!(
			(to_segment(Vec2::new(40.0, 10.0), start, end) - 10.0).abs() < 1.0e-4,
			"past the end of it, which is a distance from the end and not from the line"
		);
		assert!(
			(to_segment(Vec2::new(3.0, 4.0), Vec2::ZERO, Vec2::ZERO) - 5.0).abs() < 1.0e-4,
			"and a line of no length is a point, measured to where it is"
		);
	}

	#[test]
	fn a_ray_meets_a_line_where_the_two_are_nearest() {
		// the x axis through the origin, and a ray straight down at x = 4.
		let found = along(Vec3::ZERO, Vec3::X, Vec3::new(4.0, 5.0, 0.0), Vec3::NEG_Y);

		assert!(
			found.is_some_and(|far| (far - 4.0).abs() < 1.0e-3),
			"four along the axis: {found:?}"
		);

		// and the same ray moved sideways, which is nearest the same point.
		let across = along(Vec3::ZERO, Vec3::X, Vec3::new(4.0, 5.0, 7.0), Vec3::NEG_Y);

		assert!(across.is_some_and(|far| (far - 4.0).abs() < 1.0e-3), "{across:?}");
	}

	#[test]
	fn a_ray_running_along_a_line_has_no_one_nearest_point() {
		assert_eq!(
			along(Vec3::ZERO, Vec3::X, Vec3::new(0.0, 1.0, 0.0), Vec3::X),
			None,
			"parallel"
		);
	}

	#[test]
	fn a_ray_meets_a_disc_at_the_angle_it_goes_through_it() {
		// a disc in the xy plane at the origin, measured from x.
		let quarter = around(Vec3::ZERO, Vec3::Z, Vec3::X, Vec3::new(0.0, 3.0, 5.0), Vec3::NEG_Z);

		assert!(
			quarter.is_some_and(|angle| (angle - FRAC_PI_2).abs() < 1.0e-3),
			"straight up from the middle is a quarter turn: {quarter:?}"
		);

		let none = around(Vec3::ZERO, Vec3::Z, Vec3::X, Vec3::new(0.0, 3.0, 5.0), Vec3::X);

		assert_eq!(none, None, "and a ray along the disc never meets it");
	}

	#[test]
	fn a_slide_goes_the_way_the_arm_points() {
		let start = Transform::at(Vec3::Y);
		let straight = moved(start, Axis::X, 3.0);

		assert!(
			straight
				.position
				.abs_diff_eq(Vec3::new(3.0, 1.0, 0.0), 1.0e-4)
		);

		let mut turned_start = start;
		turned_start.rotation = Quat::from_rotation_z(FRAC_PI_2);
		let turned = moved(turned_start, Axis::X, 3.0);

		assert!(
			turned
				.position
				.abs_diff_eq(Vec3::new(0.0, 4.0, 0.0), 1.0e-3),
			"a quarter turn about z points the first arm up: {}",
			turned.position
		);
	}

	#[test]
	fn a_turn_is_about_the_things_own_axis() {
		let mut start = Transform::IDENTITY;
		start.rotation = Quat::from_rotation_z(FRAC_PI_2);

		let after = turned(start, Axis::X, FRAC_PI_2);

		// a quarter turn about z has already sent the thing's own first axis
		// along the world's second. Turning about *its own* first axis leaves
		// that axis exactly where it is; turning about the *world's* first one
		// would swing it round to the third, which is what tells the two
		// apart.
		let own = after.rotation * Vec3::X;
		assert!(own.abs_diff_eq(Vec3::Y, 1.0e-3), "the axis turned about has not moved: {own}");

		// and something perpendicular to it has, or the turn did nothing.
		let across = after.rotation * Vec3::Y;
		assert!(across.abs_diff_eq(Vec3::Z, 1.0e-3), "and everything else has: {across}");

		assert!(
			(after.rotation.length() - 1.0).abs() < 1.0e-5,
			"and the rotation is still a rotation"
		);
	}

	#[test]
	fn a_turn_hands_back_a_rotation_even_when_it_was_given_something_else() {
		// `Transform::rotation` is a public field and nothing polices it: a
		// game may write whatever it likes there, and the scene *source*
		// refuses a quaternion that is not a rotation precisely because
		// nothing else does. So a turn normalizes on the way out rather than
		// trusting what it was handed.
		let mut start = Transform::IDENTITY;
		start.rotation = Quat::from_xyzw(0.0, 0.0, 0.0, 4.0);

		let after = turned(start, Axis::Y, FRAC_PI_4);

		assert!(
			(after.rotation.length() - 1.0).abs() < 1.0e-5,
			"what comes back is a rotation: {}",
			after.rotation.length()
		);
	}

	#[test]
	fn a_turn_of_nothing_changes_nothing() {
		let start = Transform::IDENTITY;
		let after = turned(start, Axis::Y, 0.0);

		assert!(after.rotation.abs_diff_eq(start.rotation, 1.0e-5));
	}

	#[test]
	fn a_stretch_changes_one_axis_and_leaves_the_others() {
		let mut start = Transform::IDENTITY;
		start.scale = Vec3::new(2.0, 3.0, 4.0);

		let after = sized(start, Axis::Y, 2.0);

		assert!(
			after
				.scale
				.abs_diff_eq(Vec3::new(2.0, 6.0, 4.0), 1.0e-4),
			"only the second one moved: {}",
			after.scale
		);
		assert!(after.position.abs_diff_eq(start.position, 1.0e-6), "and nothing else did");
	}

	#[test]
	fn a_stretch_cannot_be_taken_to_nothing_or_through_it() {
		let start = Transform::IDENTITY;

		assert!(sized(start, Axis::X, 0.0).scale.x > 0.0, "not to nothing");
		assert!(sized(start, Axis::X, -3.0).scale.x > 0.0, "and not through it");
	}

	#[test]
	fn every_axis_has_its_own_place_way_and_color() {
		let places: Vec<usize> = Axis::ALL.into_iter().map(Axis::place).collect();

		assert_eq!(places, vec![0, 1, 2], "in order");
		assert_eq!(Axis::X.way(), Vec3::X);
		assert_eq!(Axis::Y.way(), Vec3::Y);
		assert_eq!(Axis::Z.way(), Vec3::Z);
		assert_ne!(Axis::X.tint(), Axis::Y.tint(), "and they are told apart by eye");
	}

	#[test]
	fn the_angle_round_a_ring_covers_the_whole_of_it() {
		assert!(step_of(0).abs() < 1.0e-6, "the first step is none of the way round");
		assert!((step_of(RING_STEPS) - 1.0).abs() < 1.0e-6, "and the last is all of it");
		let eighth = std::f32::consts::TAU.mul_add(step_of(RING_STEPS / 8), -FRAC_PI_4);
		assert!(
			eighth.abs() < 1.0e-4,
			"an eighth of the steps is an eighth of the way: {eighth}"
		);
	}
}
