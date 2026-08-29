//! The narrow phase: which bodies touch, where, and how deeply.
//!
//! Six shape pairs, and only two kinds of code between them. A ball is close
//! enough to a point that every pair involving one is "find the nearest place
//! on the other shape and compare the distance to the radius" - exact, a few
//! lines each, and no separating axis anywhere. Everything else is a box or a
//! triangle, and those go through [`convex`](crate::convex), which does not
//! know a box from a triangle.
//!
//! A mesh is not one shape but a bag of them, so a pair involving one runs the
//! whole test per triangle and keeps every manifold it finds. That is why a
//! mesh body is never dynamic - see
//! [`Body::movable`](colby_core::abi::Body::movable) - and why the triangles it
//! produces are tested only against the bounds of the other body first.

use colby_core::{
	abi::{Bodies, Body, BodyId, Shape, ShapeKind, Transform},
	glam::{Mat3, Vec3},
};

use crate::{
	Simulation,
	convex::{Hull, MAX_CONTACTS, Touch, collide},
	query,
};

/// The identifier every contact involving a ball carries.
///
/// A ball has one contact with anything, wherever it has rolled to, so its
/// identity is a constant rather than a feature. @ref
/// [`Touch::id`](crate::convex::Touch::id).
const ROUND: u32 = 0;

/// Contacts shallower than this are ignored.
///
/// Two bodies that merely graze produce a manifold whose impulses are noise,
/// and a pile of noise is a pile that hums.
const SKIN: f32 = 1.0e-4;

/// One place two bodies touch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Contact {
	/// Where, in world space.
	pub(crate) position: Vec3,

	/// How far the two overlap. Always positive.
	pub(crate) depth: f32,

	/// Which feature made it, for finding the same point again next step.
	pub(crate) id: u32,
}

/// Everything the solver needs about one touching pair.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Manifold {
	/// One of the two bodies.
	pub(crate) first: BodyId,

	/// The other.
	pub(crate) second: BodyId,

	/// Which way to push them apart, pointing from the first at the second.
	pub(crate) normal: Vec3,

	/// Where they touch.
	pub(crate) points: [Contact; MAX_CONTACTS],

	/// How many of `points` are real.
	pub(crate) count: usize,

	/// How much of the impact comes back.
	pub(crate) restitution: f32,

	/// How hard the pair is to slide.
	pub(crate) friction: f32,

	/// Which piece of a many-piece shape this manifold is against.
	///
	/// Zero for everything but a collision mesh, where one pair of bodies
	/// produces one manifold per triangle and they would otherwise be
	/// indistinguishable to anything keeping notes on a pair.
	pub(crate) shard: u32,
}

impl Manifold {
	/// The points that are real.
	pub(crate) fn points(&self) -> &[Contact] { &self.points[..self.count] }
}

/// Every pair of bodies that touches.
///
/// Every pair is tested against every other, which is a broad phase in the
/// sense that a linear scan is a search. With a bounds rejection in front of it
/// that is a few hundred nanoseconds for the scenes this engine has; the day it
/// is not, a grid goes here and nothing above it changes.
///
/// What a sensor makes goes in a second list rather than being marked in the
/// first. Two lists because the solver must never be handed one, and a flag it
/// had to check would be a flag somewhere in it could be forgotten; and
/// separate lists rather than a partition of one, because reordering the
/// solver's input would change the order it visits pairs in, which is a
/// property a Gauss-Seidel sweep can feel.
///
/// @param bodies - the table
/// @param simulation - the baked collision meshes
/// @param into - cleared and filled with what the solver has to separate
/// @param sensed - cleared and filled with what a sensor merely noticed
pub(crate) fn find(
	bodies: &Bodies,
	simulation: &Simulation,
	into: &mut Vec<Manifold>,
	sensed: &mut Vec<Manifold>,
) {
	into.clear();
	sensed.clear();

	let handles: Vec<(BodyId, Body)> = bodies
		.iter()
		.map(|(id, body)| (id, *body))
		.collect();

	for (index, &(first, ref one)) in handles.iter().enumerate() {
		for &(second, ref other) in &handles[index + 1..] {
			// two sensors have nothing to tell each other. Neither pushes, so
			// neither has anything to notice, and a world full of trigger
			// volumes would otherwise test every one against every other.
			if !one.solid() && !other.solid() {
				continue;
			}

			let sensing = !one.solid() || !other.solid();

			// two bodies neither of which the solver moves can never need
			// separating, however much they overlap. A sensor is exempt rather
			// than overlooked: what it produces is an event and not a
			// separation, and a static body really does move here - it is
			// written from the entity it is bolted to at the top of every step,
			// so a trigger a moving prop is carried through has to be asked.
			if !sensing && !one.movable() && !other.movable() {
				continue;
			}

			if one.sleeping && other.sleeping {
				continue;
			}

			// and last, the filter a game set up on purpose. Last because it is
			// the only one of the four that is somebody's decision rather than an
			// arrangement of the world, so a pair skipped here is skipped for a
			// reason that can be read out of the two bodies.
			if !one.layers.meets(other.layers) {
				continue;
			}

			let list = if sensing { &mut *sensed } else { &mut *into };

			pair(simulation, (first, one), (second, other), list);
		}
	}
}

/// Tests one pair and appends whatever it finds.
///
/// @param simulation - the baked collision meshes
/// @param first - one body and its handle
/// @param second - the other
/// @param into - where to append
fn pair(
	simulation: &Simulation,
	first: (BodyId, &Body),
	second: (BodyId, &Body),
	into: &mut Vec<Manifold>,
) {
	let (one, other) = (first.1, second.1);

	// a mesh is the only shape that is many shapes, so it is worth putting it
	// on one side and looping there rather than writing the loop twice.
	match (one.shape.kind, other.shape.kind) {
		| (ShapeKind::Mesh, ShapeKind::Mesh) => (),
		| (ShapeKind::Mesh, _) => mesh(simulation, first, second, true, into),
		| (_, ShapeKind::Mesh) => mesh(simulation, second, first, false, into),
		| _ =>
			if let Some(found) = solids(first, second) {
				into.push(found);
			},
	}
}

/// Tests a box or a ball against every triangle of a collision mesh.
///
/// @param simulation - where the baked triangles live
/// @param mesh - the mesh body and its handle
/// @param solid - the other body and its handle
/// @param leading - whether the mesh is the *first* body of the reported pair,
/// which is what decides the sign of every normal produced
/// @param into - where to append
fn mesh(
	simulation: &Simulation,
	mesh: (BodyId, &Body),
	solid: (BodyId, &Body),
	leading: bool,
	into: &mut Vec<Manifold>,
) {
	let Some(collider) = simulation.collider(mesh.0) else {
		return;
	};

	let Some((low, high)) = query::world_bounds(solid.1, None) else {
		return;
	};

	let matrix = mesh.1.transform.matrix();
	let surface = surface_of(mesh.1, solid.1);

	for (shard, corners) in collider.triangles().enumerate() {
		let placed = corners.map(|corner| matrix.transform_point3(corner));

		if apart(placed, low, high) {
			continue;
		}

		let Some(hull) = Hull::triangle(placed) else {
			continue;
		};

		let Some(found) = against(&hull, solid.1) else {
			continue;
		};

		let (normal, points, count) = found;
		let (first, second, normal) = if leading {
			(mesh.0, solid.0, normal)
		} else {
			(solid.0, mesh.0, -normal)
		};

		let mut manifold = assemble(first, second, normal, points, count, surface);
		manifold.shard = u32::try_from(shard).unwrap_or(0);

		into.push(manifold);
	}
}

/// Whether a triangle is entirely outside a box.
///
/// @param corners - the triangle, in world space
/// @param low - the box's low corner
/// @param high - its high corner
pub(crate) fn apart(corners: [Vec3; 3], low: Vec3, high: Vec3) -> bool {
	let mut least = corners[0];
	let mut most = corners[0];

	for &corner in &corners[1..] {
		least = least.min(corner);
		most = most.max(corner);
	}

	most.cmplt(low).any() || least.cmpgt(high).any()
}

/// Tests one triangle against a box or a ball.
///
/// @param hull - the triangle
/// @param solid - the other body
/// @return the normal out of the triangle, the points, and how many are real
fn against(hull: &Hull, solid: &Body) -> Option<(Vec3, [Touch; MAX_CONTACTS], usize)> {
	match solid.shape.kind {
		| ShapeKind::Sphere => {
			let (normal, touch) =
				sphere_triangle(center(solid), radius(solid), hull.corners(), hull.normal())?;
			let mut points = [Touch::default(); MAX_CONTACTS];
			points[0] = touch;

			Some((normal, points, 1))
		},
		| _ => collide(hull, &Hull::cuboid(&solid.transform, solid.shape.extents)),
	}
}

/// Tests two bodies neither of which is a mesh.
///
/// @param first - one body and its handle
/// @param second - the other
fn solids(first: (BodyId, &Body), second: (BodyId, &Body)) -> Option<Manifold> {
	let (one, other) = (first.1, second.1);
	let surface = surface_of(one, other);

	let (normal, points, count) = match (one.shape.kind, other.shape.kind) {
		| (ShapeKind::Sphere, ShapeKind::Sphere) => single(sphere_sphere(one, other)?),
		| (ShapeKind::Sphere, _) => single(sphere_box(one, other, false)?),
		| (_, ShapeKind::Sphere) => single(sphere_box(other, one, true)?),
		| _ => collide(
			&Hull::cuboid(&one.transform, one.shape.extents),
			&Hull::cuboid(&other.transform, other.shape.extents),
		)?,
	};

	Some(assemble(first.0, second.0, normal, points, count, surface))
}

/// Wraps a single point up the way a clipped face arrives.
fn single(found: (Vec3, Touch)) -> (Vec3, [Touch; MAX_CONTACTS], usize) {
	let mut points = [Touch::default(); MAX_CONTACTS];
	points[0] = found.1;

	(found.0, points, 1)
}

/// Turns raw touches into a manifold, dropping the ones too shallow to matter.
///
/// @param first - one body's handle
/// @param second - the other's
/// @param normal - pointing from the first at the second
/// @param points - what the narrow phase found
/// @param count - how many are real
/// @param surface - the pair's restitution and friction
fn assemble(
	first: BodyId,
	second: BodyId,
	normal: Vec3,
	points: [Touch; MAX_CONTACTS],
	count: usize,
	surface: (f32, f32),
) -> Manifold {
	let mut contacts = [Contact::default(); MAX_CONTACTS];
	let mut found = 0;

	for touch in &points[..count] {
		if touch.depth <= SKIN {
			continue;
		}

		contacts[found] = Contact {
			position: touch.position,
			depth: touch.depth,
			id: touch.id,
		};
		found += 1;
	}

	Manifold {
		first,
		second,
		normal,
		points: contacts,
		count: found,
		restitution: surface.0,
		friction: surface.1,
		shard: 0,
	}
}

/// What a pair of surfaces does between them.
///
/// Restitution is the larger of the two and friction the geometric mean, which
/// is what every engine settles on: a bouncy ball stays bouncy on a dead floor,
/// and ice against rubber is somewhere between rather than either.
fn surface_of(first: &Body, second: &Body) -> (f32, f32) {
	let restitution = first
		.restitution
		.max(second.restitution)
		.clamp(0.0, 1.0);
	let friction = (first.friction.max(0.0) * second.friction.max(0.0)).sqrt();

	(restitution, friction)
}

/// Where a ball is.
pub(crate) fn center(body: &Body) -> Vec3 { body.transform.position }

/// How big a ball is in world space.
///
/// The largest axis of the scale, so a ball under a non-uniform one is the
/// sphere around the ellipsoid rather than the ellipsoid. A ray is exact about
/// this and the solver is not; @ref `colby-known-gaps`.
pub(crate) fn radius(body: &Body) -> f32 {
	body.shape.radius.abs() * body.transform.scale.abs().max_element()
}

/// Two balls.
fn sphere_sphere(first: &Body, second: &Body) -> Option<(Vec3, Touch)> {
	let (one, other) = (center(first), center(second));
	let (near, far) = (radius(first), radius(second));
	let between = other - one;
	let distance = between.length();

	if distance >= near + far {
		return None;
	}

	let normal = if distance > SKIN { between / distance } else { Vec3::Y };
	let depth = near + far - distance;

	Some((normal, Touch {
		position: normal.mul_add(Vec3::splat(depth.mul_add(-0.5, near)), one),
		depth,
		id: ROUND,
	}))
}

/// A ball against a box.
///
/// @param ball - the ball
/// @param solid - the box
/// @param flipped - whether the box is the *first* body of the reported pair
/// @return the normal from the first body at the second, and the point
fn sphere_box(ball: &Body, solid: &Body, flipped: bool) -> Option<(Vec3, Touch)> {
	let matrix = solid.transform.matrix();
	let inverse = matrix.inverse();

	if !inverse.is_finite() {
		return None;
	}

	let extents = solid.shape.extents.abs();
	let local = inverse.transform_point3(center(ball));
	let nearest = local.clamp(-extents, extents);
	let world = matrix.transform_point3(nearest);
	let reach = radius(ball);
	let between = center(ball) - world;
	let distance = between.length();

	// outside the box, which is the ordinary case: the nearest point on its
	// surface is the contact and the normal is the way back to the middle of
	// the ball.
	if distance > SKIN {
		if distance >= reach {
			return None;
		}

		let normal = between / distance;
		let depth = reach - distance;
		let touch = Touch { position: world, depth, id: ROUND };

		return Some(if flipped { (normal, touch) } else { (-normal, touch) });
	}

	// the middle of the ball is inside the box, so there is no direction to the
	// surface. The shallowest face is the way out.
	let (axis, depth) = shallowest(local, extents);
	let sign = if local[axis] < 0.0 { -1.0 } else { 1.0 };
	let normal = (Mat3::from_quat(solid.transform.rotation) * (Vec3::AXES[axis] * sign))
		.normalize_or(Vec3::Y);
	let touch = Touch {
		position: world,
		depth: depth + reach,
		id: ROUND,
	};

	Some(if flipped { (normal, touch) } else { (-normal, touch) })
}

/// Which face of a box a point inside it is nearest to.
///
/// @param local - the point, in the box's own space
/// @param extents - the box's half-extents
/// @return the axis and how far in the point is
fn shallowest(local: Vec3, extents: Vec3) -> (usize, f32) {
	let mut axis = 0;
	let mut least = f32::INFINITY;

	for index in 0..3 {
		let depth = extents[index] - local[index].abs();

		if depth < least {
			least = depth;
			axis = index;
		}
	}

	(axis, least.max(0.0))
}

/// A ball against one triangle.
///
/// @param middle - where the ball is
/// @param reach - its radius
/// @param corners - the triangle
/// @param face - the triangle's unit normal
/// @return the normal out of the triangle and the point, or `None`
fn sphere_triangle(
	middle: Vec3,
	reach: f32,
	corners: [Vec3; 3],
	face: Vec3,
) -> Option<(Vec3, Touch)> {
	// behind the triangle is not a contact, for the reason a box behind one is
	// not: a surface has no inside to be pushed out of.
	if (middle - corners[0]).dot(face) < 0.0 {
		return None;
	}

	let nearest = nearest_on(middle, corners);
	let between = middle - nearest;
	let distance = between.length();

	if distance >= reach {
		return None;
	}

	let normal = if distance > SKIN { between / distance } else { face };

	Some((normal, Touch {
		position: nearest,
		depth: reach - distance,
		id: ROUND,
	}))
}

/// The point of a triangle nearest another point.
///
/// The barycentric walk: try the face, then each edge, then each corner. Every
/// branch is one of the seven Voronoi regions of a triangle.
///
/// @param point - what to be near
/// @param corners - the triangle
fn nearest_on(point: Vec3, corners: [Vec3; 3]) -> Vec3 {
	let [first, second, third] = corners;
	let (along, across) = (second - first, third - first);
	let towards = point - first;

	let (d1, d2) = (along.dot(towards), across.dot(towards));
	if d1 <= 0.0 && d2 <= 0.0 {
		return first;
	}

	let towards_second = point - second;
	let (d3, d4) = (along.dot(towards_second), across.dot(towards_second));
	if d3 >= 0.0 && d4 <= d3 {
		return second;
	}

	let edge = d1.mul_add(d4, -(d3 * d2));
	if edge <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
		return first + along * (d1 / (d1 - d3));
	}

	let towards_third = point - third;
	let (d5, d6) = (along.dot(towards_third), across.dot(towards_third));
	if d6 >= 0.0 && d5 <= d6 {
		return third;
	}

	let other = d5.mul_add(d2, -(d1 * d6));
	if other <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
		return first + across * (d2 / (d2 - d6));
	}

	let far = d3.mul_add(d6, -(d5 * d4));
	if far <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
		return second + (third - second) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
	}

	let total = far + edge + other;
	let towards_second = edge / total;
	let towards_third = other / total;

	first + along * towards_third + across * towards_second
}

/// The world-space inverse inertia tensor of a body.
///
/// Worked out per step from the mass and the shape rather than stored, because
/// a stored one is wrong the moment somebody changes either. A body the solver
/// does not move has none at all, which is what makes an impulse against a wall
/// turn nothing.
///
/// @param body - the body
pub(crate) fn inverse_inertia(body: &Body) -> Mat3 {
	let inverse_mass = body.inverse_mass();

	if inverse_mass <= 0.0 {
		return Mat3::ZERO;
	}

	let diagonal = match body.shape.kind {
		| ShapeKind::Sphere => {
			// a solid ball is two fifths of m r squared about every axis, so
			// the inverse is the inverse mass over that. Writing it as the
			// moment and then inverting is how this was wrong by a factor of
			// the radius to the fourth for a while: at a radius of a third that
			// is a ball sixty times harder to spin than it should be, which
			// looks like a ball that will not roll.
			let reach = radius(body).max(SKIN);

			Vec3::splat(inverse_mass / (0.4 * reach * reach))
		},
		| _ => {
			let full = local_extents(&body.shape, &body.transform) * 2.0;
			let squared = full * full;
			let scale = inverse_mass * 12.0;

			Vec3::new(
				scale / (squared.y + squared.z).max(SKIN),
				scale / (squared.x + squared.z).max(SKIN),
				scale / (squared.x + squared.y).max(SKIN),
			)
		},
	};

	let rotation = Mat3::from_quat(body.transform.rotation);

	rotation * Mat3::from_diagonal(diagonal) * rotation.transpose()
}

/// A shape's half-extents after the transform's scale.
fn local_extents(shape: &Shape, transform: &Transform) -> Vec3 {
	shape.local_extents().abs() * transform.scale.abs()
}

#[cfg(test)]
mod tests {
	use super::*;

	fn ball(position: Vec3, reach: f32) -> Body {
		Body::dynamic(Shape::ball(reach), Transform::at(position), 1.0)
	}

	fn brick(position: Vec3, extents: Vec3) -> Body {
		Body::new(
			colby_core::abi::BodyKind::Static,
			Shape::cuboid(extents),
			Transform::at(position),
		)
	}

	#[test]
	fn two_balls_touching_meet_halfway_between_their_surfaces() {
		let (first, second) = (ball(Vec3::ZERO, 1.0), ball(Vec3::new(1.5, 0.0, 0.0), 1.0));
		let (normal, touch) = sphere_sphere(&first, &second).expect("they overlap by a half");

		assert!(normal.abs_diff_eq(Vec3::X, 1.0e-5), "from the first at the second");
		assert!((touch.depth - 0.5).abs() < 1.0e-5, "half a unit in, got {}", touch.depth);
		assert!(
			touch
				.position
				.abs_diff_eq(Vec3::new(0.75, 0.0, 0.0), 1.0e-5),
			"between the two surfaces, got {}",
			touch.position
		);
	}

	#[test]
	fn two_balls_that_do_not_reach_report_nothing() {
		let (first, second) = (ball(Vec3::ZERO, 1.0), ball(Vec3::new(2.5, 0.0, 0.0), 1.0));

		assert!(sphere_sphere(&first, &second).is_none(), "half a unit apart");
	}

	#[test]
	fn a_ball_on_a_slab_is_pushed_straight_up() {
		let slab = brick(Vec3::ZERO, Vec3::new(4.0, 0.5, 4.0));
		let resting = ball(Vec3::new(0.0, 1.4, 0.0), 1.0);
		let (normal, touch) = sphere_box(&resting, &slab, true).expect("it sinks in by a tenth");

		assert!(normal.abs_diff_eq(Vec3::Y, 1.0e-5), "out of the slab, got {normal}");
		assert!((touch.depth - 0.1).abs() < 1.0e-5, "a tenth, got {}", touch.depth);
	}

	#[test]
	fn a_ball_whose_middle_is_inside_a_box_leaves_by_the_nearest_face() {
		let slab = brick(Vec3::ZERO, Vec3::new(4.0, 0.5, 4.0));
		let sunk = ball(Vec3::new(0.0, 0.3, 0.0), 0.25);
		let (normal, touch) = sphere_box(&sunk, &slab, true).expect("the middle is inside");

		assert!(normal.abs_diff_eq(Vec3::Y, 1.0e-5), "up, which is the nearest way out");
		assert!(touch.depth > 0.25, "and at least a radius deep, got {}", touch.depth);
	}

	#[test]
	fn a_ball_on_a_triangle_rests_on_its_face() {
		let corners =
			[Vec3::new(-4.0, 0.0, -4.0), Vec3::new(0.0, 0.0, 6.0), Vec3::new(4.0, 0.0, -4.0)];
		let (normal, touch) =
			sphere_triangle(Vec3::new(0.0, 0.9, 0.0), 1.0, corners, Vec3::Y).expect("it touches");

		assert!(normal.abs_diff_eq(Vec3::Y, 1.0e-5), "out of the face, got {normal}");
		assert!((touch.depth - 0.1).abs() < 1.0e-5, "a tenth, got {}", touch.depth);
		assert!(touch.position.y.abs() < 1.0e-5, "on the plane");
	}

	#[test]
	fn a_ball_past_the_edge_of_a_triangle_touches_its_corner() {
		let corners = [Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0)];
		let (_, touch) = sphere_triangle(Vec3::new(-0.3, 0.3, -0.3), 0.6, corners, Vec3::Y)
			.expect("it reaches the corner");

		assert!(
			touch.position.abs_diff_eq(Vec3::ZERO, 1.0e-5),
			"the nearest place on the triangle is its corner, got {}",
			touch.position
		);
	}

	#[test]
	fn a_ball_behind_a_triangle_is_left_alone() {
		let corners =
			[Vec3::new(-4.0, 0.0, -4.0), Vec3::new(0.0, 0.0, 6.0), Vec3::new(4.0, 0.0, -4.0)];

		assert!(
			sphere_triangle(Vec3::new(0.0, -0.9, 0.0), 1.0, corners, Vec3::Y).is_none(),
			"a surface has no inside"
		);
	}

	#[test]
	fn a_static_body_has_no_inertia_to_speak_of() {
		let slab = brick(Vec3::ZERO, Vec3::ONE);

		assert_eq!(inverse_inertia(&slab), Mat3::ZERO, "an impulse against a wall turns nothing");
	}

	#[test]
	fn a_heavier_body_is_harder_to_turn() {
		let light = Body::dynamic(Shape::UNIT, Transform::IDENTITY, 1.0);
		let heavy = Body::dynamic(Shape::UNIT, Transform::IDENTITY, 10.0);

		assert!(
			inverse_inertia(&heavy).x_axis.x < inverse_inertia(&light).x_axis.x,
			"ten times the mass is a tenth of the inverse inertia"
		);
	}

	#[test]
	fn a_ball_and_a_cube_of_the_same_mass_do_not_turn_alike() {
		let cube = Body::dynamic(Shape::UNIT, Transform::IDENTITY, 1.0);
		let sphere = Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0);

		assert!(
			(inverse_inertia(&cube).x_axis.x - inverse_inertia(&sphere).x_axis.x).abs() > 1.0e-3,
			"the tensor comes from the shape and not only from the mass"
		);
	}
}
