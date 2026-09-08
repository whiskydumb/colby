//! The geometry behind the two traces.
//!
//! Everything here works in one of two spaces and says which. A body's own
//! space is where its shape is exactly what [`Shape`] says - the box really is
//! `extents` half-wide, the ball really is `radius` across, the triangles are
//! the ones the mesh was baked from - and getting there is one inverse of the
//! body's model matrix. That is worth doing rather than scaling the shape,
//! because it is exact under non-uniform scale where scaling a radius is not,
//! and because the fraction along a segment is preserved by an affine map, so
//! the answer comes back without being converted at all.
//!
//! A ray is exact against all three shapes. A swept box is **not**: it tests
//! the world-space bounds of each body, grown by the box's half-extents, which
//! is exact for an unrotated box and conservative - reporting contact slightly
//! early - for a ball, a rotated box and a mesh. @ref the module docs of
//! [`crate`] for why that is the trade taken.

use colby_core::{
	abi::{Bodies, Body, BodyId, Shape, ShapeKind, TraceInfo, TraceResult, Transform},
	glam::{Mat3, Mat4, Vec3},
};

use crate::{
	Simulation, contact,
	convex::{Hull, collide},
};

/// Below this the segment is a point and every division by it is a mistake.
const EPSILON: f32 = 1.0e-6;

/// How far apart two places along a sweep may be when the cast box has no
/// thickness of its own to go by.
const SWEEP_STEP: f32 = 0.1;

/// How many places along a sweep are tried before giving up on being exact.
///
/// A sweep wanting more than this is a box crossing many times its own size in
/// one call, which is not what an exact answer is for. That body keeps the
/// answer the bounds gave, which reports contact early rather than late - the
/// safe direction to be wrong in, and the direction this whole function used
/// to be wrong in for everything.
const MAX_SWEEP_SAMPLES: usize = 64;

/// How many times the bracket is halved once the first overlap is found.
///
/// Eight puts the answer within a two-hundred-and-fiftieth of the sample step,
/// which for a box a third of a unit thick is a tenth of a millimeter.
const SWEEP_REFINEMENTS: usize = 8;

/// What one body did to a trace.
#[derive(Clone, Copy, Debug)]
struct Hit {
	/// How far along the segment contact happened.
	fraction: f32,

	/// The surface normal there, in world space, already turned against the
	/// segment.
	normal: Vec3,

	/// Whether the segment began inside this body.
	started_solid: bool,

	/// Whether it ended inside.
	ended_solid: bool,
}

/// Traces a ray through every body.
///
/// @param bodies - the table to trace against
/// @param simulation - the baked collision meshes, for mesh bodies
/// @param info - the trace
/// @return the nearest contact, or a miss
pub(crate) fn ray(bodies: &Bodies, simulation: &Simulation, info: &TraceInfo) -> TraceResult {
	// one scratch for the whole trace rather than one per body, and empty
	// until a mesh body is actually reached: `Vec::new` allocates nothing, so a
	// world of boxes and balls pays for this exactly nothing. @ref
	// [`Collider::candidates`].
	let mut shards = Vec::new();

	nearest(bodies, info, |id, body| {
		let collider = simulation.collider(id);

		ray_body(body, collider, info.start, info.end, &mut shards)
	})
}

/// Sweeps an axis-aligned box through every body.
///
/// Two stages, and the first of them is what the whole function used to be.
/// Each body's world bounds, grown by the cast box's half-extents, say whether
/// the sweep can reach it at all and between which two fractions - exact for an
/// unrotated box and generous for everything else. The second stage asks the
/// shape itself, at places along that bracket, through the same [`collide`] the
/// narrow phase uses. Agreeing with the solver matters more here than the
/// accuracy does: a box told by the sweep that it is clear and told by the
/// solver that it is not has no stable place to stand.
///
/// @param bodies - the table to trace against
/// @param simulation - the baked collision meshes
/// @param info - the trace, whose `extents` are the box's half-extents
/// @return the nearest contact, or a miss
pub(crate) fn swept(bodies: &Bodies, simulation: &Simulation, info: &TraceInfo) -> TraceResult {
	let sweep = Sweep {
		start: info.start,
		direction: info.end - info.start,
		extents: info.extents.abs(),
	};

	let mut shards = Vec::new();

	nearest(bodies, info, |id, body| {
		let collider = simulation.collider(id);
		let (low, high) = world_bounds(body, collider)?;
		let (low, high) = (low - sweep.extents, high + sweep.extents);
		let (enter, exit, coarse) = span(sweep.start, sweep.direction, low, high)?;

		refine(body, collider, sweep, (enter, exit), coarse, &mut shards)
	})
}

/// The box a sweep moves, and where it moves it.
///
/// One struct because the three travel together through four functions, and a
/// parameter list of bare `Vec3`s is a place to swap two of them by mistake.
#[derive(Clone, Copy, Debug)]
struct Sweep {
	/// Where the box's middle begins.
	start: Vec3,

	/// The whole segment its middle travels, so a fraction of this is a
	/// fraction of the sweep.
	direction: Vec3,

	/// Its half-extents. Axis-aligned, and it stays that way throughout.
	extents: Vec3,
}

impl Sweep {
	/// Where the box's middle is, part of the way along.
	fn at(&self, fraction: f32) -> Vec3 { self.start + self.direction * fraction }
}

/// Finds where along a bracket the box first touches a body.
///
/// @param body - what to test against
/// @param collider - its baked triangles, if it is a mesh
/// @param sweep - the box and its path
/// @param bracket - the fractions between which the bounds say it is worth
/// asking the shape
/// @param coarse - the normal the bounds gave, kept for the case this gives up
fn refine(
	body: &Body,
	collider: Option<&Collider>,
	sweep: Sweep,
	bracket: (f32, f32),
	coarse: Vec3,
	shards: &mut Vec<u32>,
) -> Option<Hit> {
	// the scratch is borrowed by the test, so the test is `FnMut` and `walk`
	// below takes it by mutable reference. That is the whole cost of letting a
	// swept box ask the grid instead of every triangle.
	let mut touching =
		|fraction: f32| overlap(body, collider, sweep.at(fraction), sweep.extents, shards);
	let ended_solid = touching(1.0).is_some();

	if touching(0.0).is_some() {
		// there is nothing better to say than where it began, and a normal back
		// the way it came is the only one that does not point a caller further
		// into the solid.
		return Some(Hit {
			fraction: 0.0,
			normal: -sweep.direction.normalize_or(Vec3::Y),
			started_solid: true,
			ended_solid,
		});
	}

	let (enter, exit) = bracket;
	let thickness = sweep.extents.min_element();
	let reach = sweep.direction.length().max(EPSILON);
	let stride = if thickness > EPSILON { thickness } else { SWEEP_STEP } / reach;

	let Some((mut clear, mut solid, mut normal)) = walk(&mut touching, (enter, exit), stride)
	else {
		// either nothing along the bracket touched it, or the walk ran out of
		// samples before reaching the end of one. The second is the case that
		// keeps the answer the bounds gave.
		return stride
			.mul_add(count_of(MAX_SWEEP_SAMPLES), enter)
			.lt(&exit)
			.then_some(Hit {
				fraction: enter,
				normal: against(coarse, sweep.direction),
				started_solid: false,
				ended_solid,
			});
	};

	for _ in 0..SWEEP_REFINEMENTS {
		let middle = f32::midpoint(clear, solid);

		if let Some(found) = touching(middle) {
			solid = middle;
			normal = found;
		} else {
			clear = middle;
		}
	}

	Some(Hit {
		// the last place it was clear, which is where a caller can put it.
		fraction: clear.clamp(0.0, 1.0),
		normal: against(normal, sweep.direction),
		started_solid: false,
		ended_solid,
	})
}

/// Steps along a bracket until something overlaps.
///
/// @param touching - the overlap test
/// @param bracket - the fractions to walk between
/// @param stride - how far apart to try, as a fraction of the whole sweep
/// @return the last fraction that was clear, the first that was not, and the
/// normal there; `None` if nothing touched or the walk ran out of samples
fn walk(
	touching: &mut impl FnMut(f32) -> Option<Vec3>,
	bracket: (f32, f32),
	stride: f32,
) -> Option<(f32, f32, Vec3)> {
	let (enter, exit) = bracket;
	let mut clear = enter;

	for index in 1..=MAX_SWEEP_SAMPLES {
		let fraction = stride.mul_add(count_of(index), enter).min(exit);

		if let Some(normal) = touching(fraction) {
			return Some((clear, fraction, normal));
		}

		clear = fraction;

		if fraction >= exit {
			return None;
		}
	}

	None
}

/// A sample count as the float the arithmetic wants.
///
/// The bound is small enough that there is nothing here to lose, and going
/// through `u16` says so without an `expect` about it.
fn count_of(index: usize) -> f32 { f32::from(u16::try_from(index).unwrap_or(u16::MAX)) }

/// Whether an axis-aligned box at a place overlaps a body.
///
/// @param body - what to test against
/// @param collider - its baked triangles, if it is a mesh
/// @param at - where the box's middle is
/// @param extents - its half-extents
/// @param shards - the caller's scratch for a mesh's candidate triangles
/// @return the surface normal out of the body, or `None` if they are apart
fn overlap(
	body: &Body,
	collider: Option<&Collider>,
	at: Vec3,
	extents: Vec3,
	shards: &mut Vec<u32>,
) -> Option<Vec3> {
	match body.shape.kind {
		// the solver's ball rather than the ray's ellipsoid, deliberately. A
		// sweep that disagreed with the solver about how big a ball is would put
		// a body somewhere the next step pushes it out of.
		| ShapeKind::Sphere => ball(contact::center(body), contact::radius(body), at, extents),
		| ShapeKind::Box => collide(
			&Hull::cuboid(&Transform::at(at), extents),
			&Hull::cuboid(&body.transform, body.shape.extents),
		)
		.map(|(normal, ..)| -normal),
		| ShapeKind::Mesh => triangles(body, collider?, at, extents, shards),
	}
}

/// Whether an axis-aligned box overlaps a ball.
///
/// @param center - where the ball is
/// @param radius - how big it is
/// @param at - where the box's middle is
/// @param extents - its half-extents
/// @return the normal out of the ball, or `None`
fn ball(center: Vec3, radius: f32, at: Vec3, extents: Vec3) -> Option<Vec3> {
	let closest = center.clamp(at - extents, at + extents);
	let away = closest - center;

	if away.length() > radius {
		return None;
	}

	// far enough inside and there is no nearest surface at all, because every
	// direction is one. The line between the two middles is then the only
	// answer with a meaning.
	Some(
		away.try_normalize()
			.unwrap_or_else(|| (at - center).normalize_or(Vec3::Y)),
	)
}

/// Whether an axis-aligned box overlaps any triangle of a collision mesh.
///
/// Through the collider's grid, behind the same bounds rejection the narrow
/// phase uses - and in the same order it always was, because a sweep answers
/// with the **first** triangle it finds overlapping and another order would be
/// another normal.
///
/// @param body - the mesh body, for its transform
/// @param collider - its baked triangles
/// @param at - where the box's middle is
/// @param extents - its half-extents
/// @param shards - the caller's scratch for the candidates
/// @return the normal out of the first triangle that overlaps, or `None`
fn triangles(
	body: &Body,
	collider: &Collider,
	at: Vec3,
	extents: Vec3,
	shards: &mut Vec<u32>,
) -> Option<Vec3> {
	let matrix = body.transform.matrix();
	let cast = Hull::cuboid(&Transform::at(at), extents);
	let (low, high) = (at - extents, at + extents);
	let (near, far) = local_bounds(&matrix, low, high)?;

	collider.candidates(near, far, shards);

	for shard in 0..shards.len() {
		let Some(corners) = shards
			.get(shard)
			.and_then(|&it| collider.triangle(it))
		else {
			continue;
		};
		let placed = corners.map(|corner| matrix.transform_point3(corner));

		if contact::apart(placed, low, high) {
			continue;
		}

		let Some(hull) = Hull::triangle(placed) else {
			continue;
		};

		if let Some((normal, ..)) = collide(&cast, &hull) {
			return Some(-normal);
		}
	}

	None
}

/// Runs a per-body test over the whole table and keeps the nearest answer.
///
/// @param bodies - the table
/// @param info - the trace, for its start, end and ignore list
/// @param test - what one body does to the trace
fn nearest(
	bodies: &Bodies,
	info: &TraceInfo,
	mut test: impl FnMut(BodyId, &Body) -> Option<Hit>,
) -> TraceResult {
	let mut result = TraceResult::miss(info.start, info.end);
	let direction = info.end - info.start;

	for (id, body) in bodies.iter() {
		if info.ignores(id) {
			continue;
		}

		// a sensor is not there as far as a trace is concerned. A pick ray
		// stopping at an invisible box, or a bullet stopping at a trigger, is
		// the same bug as a trigger that pushes, seen from the other side.
		if !body.solid() {
			continue;
		}

		// the same symmetric rule the narrow phase uses, and it has to be the
		// same one: a sweep that reports clear where the solver reports contact
		// leaves whatever was sweeping with no stable place to stand.
		if !info.layers.meets(body.layers) {
			continue;
		}

		let Some(hit) = test(id, body) else {
			continue;
		};

		// solidity is a property of the trace rather than of whichever body
		// happened to be nearest, so it accumulates over all of them.
		result.started_solid |= hit.started_solid;
		result.ended_solid |= hit.ended_solid;

		if result.hit && hit.fraction >= result.fraction {
			continue;
		}

		result.hit = true;
		result.fraction = hit.fraction;
		result.normal = hit.normal;
		result.body = id;
		result.entity = body.entity;
	}

	if result.hit {
		result.end = info.start + direction * result.fraction;
	}

	result
}

/// Traces a ray against one body, exactly.
///
/// @param body - what to trace against
/// @param collider - its baked triangles, if it is a mesh
/// @param start - where the ray begins, in world space
/// @param end - where it stops
/// @param shards - the caller's scratch for a mesh's candidate triangles
fn ray_body(
	body: &Body,
	collider: Option<&Collider>,
	start: Vec3,
	end: Vec3,
	shards: &mut Vec<u32>,
) -> Option<Hit> {
	let matrix = body.transform.matrix();
	let inverse = matrix.inverse();

	if !inverse.is_finite() {
		// a zero on some axis of the scale. There is no body there to hit.
		return None;
	}

	let local_start = inverse.transform_point3(start);
	let local_end = inverse.transform_point3(end);
	let local = local_end - local_start;

	let (fraction, normal) = match body.shape.kind {
		| ShapeKind::Box => local_box(local_start, local, body.shape.extents.abs()),
		| ShapeKind::Sphere => local_sphere(local_start, local, body.shape.radius.abs()),
		| ShapeKind::Mesh => collider.and_then(|it| it.trace(local_start, local, shards)),
	}?;

	let normals = Mat3::from_mat4(matrix).inverse().transpose();
	let world = (normals * normal).normalize_or(Vec3::Y);

	Some(Hit {
		fraction,
		normal: against(world, end - start),
		started_solid: solid(&body.shape, collider, local_start),
		ended_solid: solid(&body.shape, collider, local_end),
	})
}

/// Whether a point in body space is inside the shape.
///
/// A mesh is never solid: deciding that needs a watertight winding number, and
/// nothing has asked. @ref `colby-known-gaps`.
fn solid(shape: &Shape, _collider: Option<&Collider>, point: Vec3) -> bool {
	match shape.kind {
		| ShapeKind::Box => {
			let extents = shape.extents.abs();

			contains(point, -extents, extents)
		},
		| ShapeKind::Sphere => point.length_squared() <= shape.radius * shape.radius,
		| ShapeKind::Mesh => false,
	}
}

/// A segment against a box centered on the origin.
///
/// @param origin - where the segment begins
/// @param direction - the whole segment, not a unit vector, so `t` is a
/// fraction of it
/// @param extents - the box's half-extents
/// @return the fraction and the face normal, or `None`
fn local_box(origin: Vec3, direction: Vec3, extents: Vec3) -> Option<(f32, Vec3)> {
	slab(origin, direction, -extents, extents)
}

/// A segment against an axis-aligned box.
///
/// The standard slab test, keeping which face produced the entry so that the
/// normal comes out of it rather than being worked out again afterwards.
///
/// @param origin - where the segment begins
/// @param direction - the whole segment
/// @param low - the box's low corner
/// @param high - its high corner
/// @return the fraction and the face normal, or `None`
fn slab(origin: Vec3, direction: Vec3, low: Vec3, high: Vec3) -> Option<(f32, Vec3)> {
	let (enter, _, normal) = span(origin, direction, low, high)?;

	Some((enter, normal))
}

/// Both ends of where a segment is inside an axis-aligned box.
///
/// What [`slab`] is built on, and what a sweep wants that a trace does not:
/// the fraction it leaves at, which is where there is no longer any point
/// asking the shape inside the box.
///
/// @param origin - where the segment begins
/// @param direction - the whole segment
/// @param low - the box's low corner
/// @param high - its high corner
/// @return the fractions it enters and leaves at, and the face normal it
/// entered through
fn span(origin: Vec3, direction: Vec3, low: Vec3, high: Vec3) -> Option<(f32, f32, Vec3)> {
	let mut enter = 0.0_f32;
	let mut exit = 1.0_f32;
	let mut axis = 0_usize;
	let mut sign = 0.0_f32;

	for index in 0..3 {
		let start = origin[index];
		let step = direction[index];
		let (near, far) = (low[index], high[index]);

		if step.abs() < EPSILON {
			if start < near || start > far {
				return None;
			}

			continue;
		}

		let low_hit = (near - start) / step;
		let high_hit = (far - start) / step;

		// which of the two planes the segment reaches first is which way it is
		// pointing, and that is also which face it enters through.
		let (first, second, face) = if low_hit <= high_hit {
			(low_hit, high_hit, -1.0_f32)
		} else {
			(high_hit, low_hit, 1.0_f32)
		};

		if first > enter {
			enter = first;
			axis = index;
			sign = face;
		}

		exit = exit.min(second);

		if enter > exit {
			return None;
		}
	}

	let mut normal = Vec3::ZERO;

	if sign == 0.0 {
		// the segment began inside, so there is no entry face. Report the
		// start, and a normal back along the segment, which is the only answer
		// that does not point a caller into the solid.
		return Some((enter, exit, -direction.normalize_or(Vec3::Y)));
	}

	normal[axis] = sign;

	Some((enter, exit, normal))
}

/// A segment against a ball centered on the origin.
///
/// @param origin - where the segment begins
/// @param direction - the whole segment
/// @param radius - the ball's radius
/// @return the fraction and the surface normal, or `None`
fn local_sphere(origin: Vec3, direction: Vec3, radius: f32) -> Option<(f32, Vec3)> {
	let a = direction.length_squared();

	if a < EPSILON {
		return None;
	}

	let b = 2.0 * origin.dot(direction);
	let c = radius.mul_add(-radius, origin.length_squared());
	let discriminant = b.mul_add(b, -(4.0 * a * c));

	if discriminant < 0.0 {
		return None;
	}

	let root = discriminant.sqrt();
	let first = (-b - root) / (2.0 * a);
	let second = (-b + root) / (2.0 * a);

	// the near root unless the segment began inside, in which case the far one
	// is where it leaves.
	let t = if first >= 0.0 { first } else { second };

	if !(0.0..=1.0).contains(&t) {
		return None;
	}

	let point = origin + direction * t;

	Some((t, point.normalize_or(Vec3::Y)))
}

/// A segment against one triangle, two-sided.
///
/// Moller-Trumbore. Two-sided on purpose: a collision mesh that is only solid
/// from outside is a mesh a player falls through the moment they are inside it,
/// and which side of a triangle is "out" is not a thing an OBJ reliably says.
///
/// @param origin - where the segment begins
/// @param direction - the whole segment
/// @param corners - the triangle's three corners
/// @return the fraction and the geometric normal, or `None`
fn triangle(origin: Vec3, direction: Vec3, corners: [Vec3; 3]) -> Option<(f32, Vec3)> {
	let [anchor, second, third] = corners;
	let (first_edge, second_edge) = (second - anchor, third - anchor);
	let across = direction.cross(second_edge);
	let determinant = first_edge.dot(across);

	if determinant.abs() < EPSILON {
		return None;
	}

	let inverse = 1.0 / determinant;
	let to_origin = origin - anchor;
	let first_weight = to_origin.dot(across) * inverse;

	if !(0.0..=1.0).contains(&first_weight) {
		return None;
	}

	let along = to_origin.cross(first_edge);
	let second_weight = direction.dot(along) * inverse;

	if second_weight < 0.0 || first_weight + second_weight > 1.0 {
		return None;
	}

	let fraction = second_edge.dot(along) * inverse;

	if !(0.0..=1.0).contains(&fraction) {
		return None;
	}

	Some((
		fraction,
		first_edge
			.cross(second_edge)
			.normalize_or(Vec3::Y),
	))
}

/// Turns a normal to face back along a segment.
///
/// A two-sided triangle and a segment leaving a shape both produce a normal
/// pointing the way the trace was going, which is the one direction a caller
/// can do nothing useful with.
fn against(normal: Vec3, direction: Vec3) -> Vec3 {
	if normal.dot(direction) > 0.0 { -normal } else { normal }
}

/// Whether a point is inside an axis-aligned box.
fn contains(point: Vec3, low: Vec3, high: Vec3) -> bool {
	point.cmpge(low).all() && point.cmple(high).all()
}

/// The smallest world-space axis-aligned box holding a body.
///
/// @param body - the body
/// @param collider - its baked triangles, if it is a mesh
/// @return `(min, max)`, or `None` if the body has no extent at all
pub(crate) fn world_bounds(body: &Body, collider: Option<&Collider>) -> Option<(Vec3, Vec3)> {
	if body.shape.kind != ShapeKind::Mesh {
		return body.bounds();
	}

	let collider = collider?;
	let matrix = body.transform.matrix();
	let mut low = Vec3::splat(f32::INFINITY);
	let mut high = Vec3::splat(f32::NEG_INFINITY);

	for index in 0..8_u32 {
		let corner = Vec3::new(
			if index & 1 == 0 {
				collider.low.x
			} else {
				collider.high.x
			},
			if index & 2 == 0 {
				collider.low.y
			} else {
				collider.high.y
			},
			if index & 4 == 0 {
				collider.low.z
			} else {
				collider.high.z
			},
		);
		let placed = matrix.transform_point3(corner);

		low = low.min(placed);
		high = high.max(placed);
	}

	Some((low, high))
}

/// A world-space box in a body's own space, made axis-aligned again.
///
/// The eight corners through the inverse, and the bounds of where they land -
/// conservative under rotation, which is exactly what a grid query wants and
/// is the same trick [`world_bounds`] plays in the other direction.
///
/// @param matrix - the body's transform
/// @param low - the box's low corner, in world space
/// @param high - its high corner
/// @return `(low, high)` in the body's own space, or `None` for a transform
/// with no inverse
pub(crate) fn local_bounds(matrix: &Mat4, low: Vec3, high: Vec3) -> Option<(Vec3, Vec3)> {
	let inverse = matrix.inverse();

	if !inverse.is_finite() {
		return None;
	}

	let mut near = Vec3::splat(f32::INFINITY);
	let mut far = Vec3::splat(f32::NEG_INFINITY);

	for index in 0..8_u32 {
		let corner = Vec3::new(
			if index & 1 == 0 { low.x } else { high.x },
			if index & 2 == 0 { low.y } else { high.y },
			if index & 4 == 0 { low.z } else { high.z },
		);
		let placed = inverse.transform_point3(corner);

		near = near.min(placed);
		far = far.max(placed);
	}

	Some((near, far))
}

/// A body's geometry, baked once and kept in the body's own space.
///
/// Its own copy rather than a look into [`Meshes`](colby_core::abi::Meshes),
/// and that is a rule rather than an accident: recompiling an `.obj` replaces
/// what is drawn and leaves what is collided against alone until the body is
/// created again. A collision mesh is a resource with its own preparation -
/// the grid below is that preparation - not a second view onto a vertex buffer
/// that may be halfway through being rewritten.
///
/// **The grid is what makes a mesh bigger than a floor usable at all.** Both
/// things that read a collider used to walk every triangle it holds: the
/// narrow phase transformed each one into world space to reject it against a
/// box, and [`Collider::trace`] tested each one against the segment behind a
/// single bounds check. At the thirty-two thousand triangles a default
/// [`Terrain`](colby_core::abi::Terrain) builds, that is a per-pair cost in
/// the hundreds of microseconds and a ray that pays for the whole landscape to
/// find the hill in front of it. Both comments in this file said a hierarchy
/// went here; this is it, and it is a uniform grid rather than a tree for the
/// reason the broad phase is a sweep rather than a tree - it fits a flat list,
/// it is a hundred lines, and the shape it is best at is exactly the shape
/// terrain has.
///
/// **What comes out of it is the same set, in the same order, as walking them
/// all.** [`candidates`](Self::candidates) hands back ascending triangle
/// indices with no repeats, so the manifolds the narrow phase builds are the
/// ones it built before and the solver walks them in the order it walked them
/// in. That is not decoration: a sequential-impulse solver settles a pile
/// differently if the order moves, and `--link` never steps a simulation, so a
/// change there is a change nothing could see. Same discipline
/// [`broad`](crate::broad) keeps for its pairs.
#[derive(Clone, Debug, Default)]
pub(crate) struct Collider {
	/// Every triangle, in the body's own space.
	triangles: Vec<[Vec3; 3]>,

	/// The low corner of their bounds.
	low: Vec3,

	/// The high corner.
	high: Vec3,

	/// How many cells the grid has on each axis, all at least one.
	divisions: [u32; 3],

	/// How wide one cell is on each axis.
	///
	/// Kept rather than divided out per query: a query does three multiplies
	/// by the reciprocal instead of three divisions, and the reciprocal of a
	/// span that is nil would be an infinity that indexes nothing.
	scale: Vec3,

	/// Where each cell's triangles start in [`members`](Self::members), one
	/// longer than the number of cells so the last cell has an end.
	starts: Vec<u32>,

	/// The triangle indices, cell by cell and ascending inside each cell.
	///
	/// A triangle whose bounds cross a cell boundary is filed in every cell it
	/// touches, which is what makes a query conservative and is why
	/// [`candidates`](Self::candidates) has to remove repeats.
	members: Vec<u32>,
}

/// About how many triangles one cell should hold.
///
/// Four. Below it the grid is mostly empty cells and the `starts` array costs
/// more than the walk it saves; above it a query hands the caller triangles it
/// is going to reject anyway.
const PER_CELL: usize = 4;

/// The most cells a grid may have.
///
/// Two hundred and sixty-two thousand one hundred and forty-four, so the
/// `starts` array is a megabyte at the very worst - which is the biggest mesh
/// [`MAX_SIDE`](colby_core::abi::terrain::MAX_SIDE) allows and then some.
const MAX_CELLS: usize = 262_144;

/// The most cells on any one axis.
///
/// Two hundred and fifty-six, so that a mesh which is a thin sheet - a
/// terrain is exactly that - cannot spend its whole cell budget along one
/// direction.
const MAX_AXIS: u32 = 256;

/// Spans below this are treated as nil, and their axis gets one cell.
const FLAT: f32 = 1.0e-6;

impl Collider {
	/// Bakes the triangles of a mesh.
	///
	/// @param triangles - three corners each, in the body's own space
	pub(crate) fn new(triangles: Vec<[Vec3; 3]>) -> Self {
		let mut low = Vec3::splat(f32::INFINITY);
		let mut high = Vec3::splat(f32::NEG_INFINITY);

		for corners in &triangles {
			for &corner in corners {
				low = low.min(corner);
				high = high.max(corner);
			}
		}

		if triangles.is_empty() {
			low = Vec3::ZERO;
			high = Vec3::ZERO;
		}

		let (divisions, scale) = shape_of(&triangles, low, high);
		let (starts, members) = file(&triangles, low, divisions, scale);

		Self {
			triangles,
			low,
			high,
			divisions,
			scale,
			starts,
			members,
		}
	}

	/// How many triangles this collides with.
	pub(crate) fn count(&self) -> usize { self.triangles.len() }

	/// Every triangle, in the body's own space.
	///
	/// The whole mesh, in order, with no grid in front of it: the one caller is
	/// the debug renderer, which is drawing all of them on purpose.
	pub(crate) fn triangles(&self) -> impl Iterator<Item = [Vec3; 3]> {
		self.triangles.iter().copied()
	}

	/// One triangle, by the index [`candidates`](Self::candidates) handed out.
	///
	/// @param shard - which triangle
	pub(crate) fn triangle(&self, shard: u32) -> Option<[Vec3; 3]> {
		usize::try_from(shard)
			.ok()
			.and_then(|at| self.triangles.get(at))
			.copied()
	}

	/// Which triangles could possibly meet a box, in the body's own space.
	///
	/// **Conservative and ordered.** Every triangle whose own bounds meet the
	/// box is in the answer; some that do not may be too, because a cell is
	/// bigger than a triangle. What is guaranteed is that the answer is the
	/// triangle indices in ascending order with no repeats, which is what
	/// makes a caller that filters it further produce exactly what walking all
	/// of them would have produced. @ref the type's own documentation for why
	/// the order is load-bearing.
	///
	/// @param low - the box's low corner, in the body's own space
	/// @param high - its high corner
	/// @param into - where to put them; cleared first, and the caller keeps
	/// the allocation between steps
	pub(crate) fn candidates(&self, low: Vec3, high: Vec3, into: &mut Vec<u32>) {
		into.clear();

		if self.triangles.is_empty() || low.cmpgt(self.high).any() || high.cmplt(self.low).any() {
			return;
		}

		let first = cell_in(low, self.low, self.divisions, self.scale);
		let last = cell_in(high, self.low, self.divisions, self.scale);

		for z in first[2]..=last[2] {
			for y in first[1]..=last[1] {
				self.row(first[0]..=last[0], (y, z), into);
			}
		}

		// a triangle straddling a boundary is filed in each cell it touches,
		// so the walk above can hand the same one out several times. Sorting
		// is what puts the list back into the order the linear scan produced,
		// and the sort is over tens of entries rather than tens of thousands.
		into.sort_unstable();
		into.dedup();
	}

	/// Appends one run of cells along the first axis.
	///
	/// A method rather than a third nested loop, which is one level past what
	/// this workspace allows and is the right call anyway - the innermost of
	/// three loops is where a reader stops being able to see the shape.
	///
	/// @param across - the first and last cell on the first axis
	/// @param at - which row and which layer
	/// @param into - where to append
	fn row(&self, across: core::ops::RangeInclusive<u32>, at: (u32, u32), into: &mut Vec<u32>) {
		let (y, z) = at;

		for x in across {
			if let Some(span) = self.cell(place(self.divisions, [x, y, z])) {
				into.extend_from_slice(span);
			}
		}
	}

	/// One cell's triangle indices.
	///
	/// @param at - where the cell is in [`starts`](Self::starts)
	fn cell(&self, at: usize) -> Option<&[u32]> {
		let from = usize::try_from(*self.starts.get(at)?).ok()?;
		let to = usize::try_from(*self.starts.get(at + 1)?).ok()?;

		self.members.get(from..to)
	}

	/// A segment against the triangles the grid says are near it.
	///
	/// The candidate box is the segment's own bounds, which is conservative
	/// and is not a walk along the ray: a ray straight down at a landscape
	/// touches a column of cells and pays for almost nothing, and a ray along
	/// one touches a slab. Walking the grid cell by cell is what goes here if
	/// a long horizontal trace ever measures badly; nothing above this
	/// function would notice it arriving.
	///
	/// @param origin - where the segment begins, in the body's own space
	/// @param direction - the whole segment
	/// @param shards - the caller's scratch, so a trace allocates nothing
	fn trace(&self, origin: Vec3, direction: Vec3, shards: &mut Vec<u32>) -> Option<(f32, Vec3)> {
		slab(origin, direction, self.low, self.high)?;

		let far = origin + direction;

		self.candidates(origin.min(far), origin.max(far), shards);

		let mut best: Option<(f32, Vec3)> = None;

		for &shard in shards.iter() {
			let Some(corners) = self.triangle(shard) else {
				continue;
			};
			let Some(hit) = triangle(origin, direction, corners) else {
				continue;
			};

			if best.is_none_or(|(fraction, _)| hit.0 < fraction) {
				best = Some(hit);
			}
		}

		best
	}
}

/// How many cells a mesh's grid gets on each axis, and how wide one is.
///
/// The cells are as near cubes as the bounds allow, and their count is aimed
/// at [`PER_CELL`] triangles each: a mesh whose bounds are a thin sheet gets
/// many cells across and few through, which is what a landscape wants and what
/// a cube grid over the same bounds would spend its budget on.
///
/// @param triangles - the mesh, for how many there are
/// @param low - the low corner of its bounds
/// @param high - the high corner
/// @return the divisions and the reciprocal of one cell's size on each axis
fn shape_of(triangles: &[[Vec3; 3]], low: Vec3, high: Vec3) -> ([u32; 3], Vec3) {
	let span = (high - low).max(Vec3::ZERO);
	let wanted = (triangles.len() / PER_CELL).clamp(1, MAX_CELLS);
	// the extent of the axes that have one, spread over the cells asked for
	// and rooted by how many those are: the edge of a cube of that volume. An
	// axis with no extent is left out, so a perfectly flat mesh still gets a
	// sensible edge rather than a nil one.
	let mut volume = 1.0_f32;
	let mut axes = 0_u32;

	for along in span.to_array() {
		if along > FLAT {
			volume *= along;
			axes += 1;
		}
	}

	let edge = match u8::try_from(axes) {
		| Ok(0) | Err(_) => 1.0,
		| Ok(axes) => (volume / fraction(wanted)).powf(1.0 / f32::from(axes)),
	}
	.max(FLAT);
	let mut divisions = [1_u32; 3];
	let mut scale = Vec3::ONE;

	for (axis, slot) in divisions.iter_mut().enumerate() {
		let along = span.to_array().get(axis).copied().unwrap_or(0.0);

		if along <= FLAT {
			continue;
		}

		let count = whole((along / edge).ceil()).clamp(1, MAX_AXIS);

		*slot = count;
		scale[axis] = fraction_of(count) / along;
	}

	(divisions, scale)
}

/// Files every triangle into every cell its bounds touch.
///
/// Two passes and two allocations: one to count what each cell holds, one to
/// fill it. Filling in triangle order is what makes each cell's own list
/// ascending, which is half of what [`Collider::candidates`] promises.
///
/// @param triangles - the mesh
/// @param low - the low corner of its bounds
/// @param divisions - how many cells on each axis
/// @param scale - the reciprocal of one cell's size on each axis
/// @return where each cell starts, and the members
fn file(
	triangles: &[[Vec3; 3]],
	low: Vec3,
	divisions: [u32; 3],
	scale: Vec3,
) -> (Vec<u32>, Vec<u32>) {
	let cells = cells_in(divisions);
	let mut counts = vec![0_u32; cells + 1];

	for corners in triangles {
		spread(corners, low, divisions, scale, |at| {
			if let Some(count) = counts.get_mut(at + 1) {
				*count = count.saturating_add(1);
			}
		});
	}

	for at in 1..counts.len() {
		counts[at] = counts[at].saturating_add(counts[at - 1]);
	}

	let starts = counts.clone();
	let total = usize::try_from(counts.last().copied().unwrap_or(0)).unwrap_or(0);
	let mut members = vec![0_u32; total];

	for (shard, corners) in triangles.iter().enumerate() {
		let Ok(shard) = u32::try_from(shard) else {
			continue;
		};

		spread(corners, low, divisions, scale, |at| {
			let Some(cursor) = counts.get_mut(at) else {
				return;
			};
			let Ok(free) = usize::try_from(*cursor) else {
				return;
			};

			if let Some(slot) = members.get_mut(free) {
				*slot = shard;
				*cursor += 1;
			}
		});
	}

	(starts, members)
}

/// Calls back once for every cell one triangle's bounds touch.
///
/// Named for what it does to a triangle rather than for the loop it is, because
/// the sweep above already has a `walk` and the two have nothing to do with
/// each other.
///
/// @param corners - the triangle, in the body's own space
/// @param low - the low corner of the mesh's bounds
/// @param divisions - how many cells on each axis
/// @param scale - the reciprocal of one cell's size on each axis
/// @param visit - what to do with each cell's place in the arrays
fn spread(
	corners: &[Vec3; 3],
	low: Vec3,
	divisions: [u32; 3],
	scale: Vec3,
	mut visit: impl FnMut(usize),
) {
	let least = corners[0].min(corners[1]).min(corners[2]);
	let most = corners[0].max(corners[1]).max(corners[2]);
	let first = cell_in(least, low, divisions, scale);
	let last = cell_in(most, low, divisions, scale);

	for z in first[2]..=last[2] {
		for y in first[1]..=last[1] {
			for x in first[0]..=last[0] {
				visit(place(divisions, [x, y, z]));
			}
		}
	}
}

/// How many cells a grid of these divisions has.
fn cells_in(divisions: [u32; 3]) -> usize {
	let count = u64::from(divisions[0]) * u64::from(divisions[1]) * u64::from(divisions[2]);

	usize::try_from(count).unwrap_or(1).max(1)
}

/// Where a cell is in the flat arrays.
///
/// @param divisions - how many cells on each axis
/// @param cell - the cell, already held inside the grid
fn place(divisions: [u32; 3], cell: [u32; 3]) -> usize {
	let [x, y, z] = cell;
	let [across, up, _] = divisions;
	let index = u64::from(z) * u64::from(across) * u64::from(up)
		+ u64::from(y) * u64::from(across)
		+ u64::from(x);

	usize::try_from(index).unwrap_or(0)
}

/// Which cell a point falls in, held inside the grid.
///
/// A free function rather than a method, because the grid has to be walked
/// while it is being built and before the collider that owns it exists.
///
/// @param point - where, in the body's own space
/// @param low - the low corner of the mesh's bounds
/// @param divisions - how many cells on each axis
/// @param scale - the reciprocal of one cell's size on each axis
fn cell_in(point: Vec3, low: Vec3, divisions: [u32; 3], scale: Vec3) -> [u32; 3] {
	let inside = ((point - low) * scale).to_array();
	let mut cell = [0_u32; 3];

	for (axis, slot) in cell.iter_mut().enumerate() {
		let along = inside
			.get(axis)
			.copied()
			.unwrap_or(0.0)
			.floor()
			.clamp(0.0, f32::from(u16::MAX));
		let last = divisions
			.get(axis)
			.copied()
			.unwrap_or(1)
			.saturating_sub(1);

		*slot = whole(along).min(last);
	}

	cell
}

/// A whole number out of a float that is already whole and already in range.
///
/// The binary decomposition, written out, because this workspace refuses `as`
/// and `try_from` is not implemented from a float. Each power of two is taken
/// at most once, which is what a binary representation *is*, so seventeen
/// subtractions give the exact answer for every whole number a caller clamps
/// itself to.
///
/// @param value - clamped by the caller into `0.0 ..= 65535.0`
fn whole(value: f32) -> u32 {
	let mut count = 0_u32;
	let mut left = value.max(0.0);

	for step in (0..=16_u32).rev() {
		let stride = 1_u32 << step;
		let width = fraction_of(stride);

		if left >= width {
			left -= width;
			count = count.saturating_add(stride);
		}
	}

	count
}

/// A count as a float.
///
/// Through two `u16`s, which is what `mesh`'s own helper does and for the same
/// reason: this workspace refuses `as`, and the halves are each exact.
///
/// @param count - how many
fn fraction(count: usize) -> f32 {
	f32::from(u16::try_from(count >> 16).unwrap_or(u16::MAX)) * 65_536.0
		+ f32::from(u16::try_from(count & 0xFFFF).unwrap_or(u16::MAX))
}

/// The same, for a `u32`.
fn fraction_of(count: u32) -> f32 { fraction(usize::try_from(count).unwrap_or(0)) }
#[cfg(test)]
mod tests {
	use colby_core::abi::{MeshData, Terrain};

	use super::*;

	/// The triangles of a mesh, in the body's own space, the way
	/// [`crate::bake`] makes them.
	fn shards_of(mesh: &MeshData) -> Vec<[Vec3; 3]> {
		mesh.indices
			.chunks_exact(3)
			.filter_map(|corners| {
				let mut shard = [Vec3::ZERO; 3];

				for (slot, &index) in shard.iter_mut().zip(corners) {
					let vertex = mesh.vertices.get(usize::try_from(index).ok()?)?;

					*slot = Vec3::from_array(vertex.position);
				}

				Some(shard)
			})
			.collect()
	}

	/// Every triangle whose own bounds meet a box, found by walking all of
	/// them. What the grid has to agree with.
	fn linearly(collider: &Collider, low: Vec3, high: Vec3) -> Vec<u32> {
		collider
			.triangles()
			.enumerate()
			.filter(|(_, corners)| !contact::apart(*corners, low, high))
			.filter_map(|(at, _)| u32::try_from(at).ok())
			.collect()
	}

	fn landscape(side: u32) -> Collider {
		let ground = Terrain { side, ..Terrain::of(4) };

		Collider::new(shards_of(&ground.build()))
	}

	#[test]
	fn the_grid_never_misses_a_triangle_the_linear_scan_finds() {
		let collider = landscape(33);
		let mut found = Vec::new();

		for step in 0..24_i16 {
			let along = f32::from(step).mul_add(5.0, -60.0);
			let low = Vec3::new(along, -20.0, along * 0.5);
			let high = low + Vec3::new(3.0, 40.0, 3.0);

			collider.candidates(low, high, &mut found);

			for shard in linearly(&collider, low, high) {
				assert!(found.contains(&shard), "the grid lost triangle {shard} at {low:?}");
			}
		}
	}

	#[test]
	fn what_the_grid_hands_back_is_ascending_and_has_no_repeats() {
		// the load-bearing half: a triangle straddling a cell boundary is filed
		// in each of them, and the narrow phase walks manifolds in the order it
		// is given them.
		let collider = landscape(33);
		let mut found = Vec::new();

		collider.candidates(Vec3::splat(-200.0), Vec3::splat(200.0), &mut found);

		assert!(!found.is_empty(), "the whole landscape is in a box that holds it");
		assert!(found.is_sorted(), "ascending, which is the order the linear scan had");

		let mut once = found.clone();
		once.dedup();

		assert_eq!(once, found, "and each one only once");
	}

	#[test]
	fn a_box_that_holds_the_whole_mesh_finds_every_triangle() {
		let collider = landscape(17);
		let mut found = Vec::new();

		collider.candidates(Vec3::splat(-500.0), Vec3::splat(500.0), &mut found);

		assert_eq!(found.len(), collider.count(), "all of them, filed exactly once each");
	}

	#[test]
	fn a_box_nowhere_near_the_mesh_finds_nothing() {
		let collider = landscape(17);
		let mut found = vec![7, 8, 9];

		collider.candidates(Vec3::splat(500.0), Vec3::splat(600.0), &mut found);

		assert!(found.is_empty(), "and the scratch is cleared even so");
	}

	#[test]
	fn a_collider_with_no_triangles_answers_nothing_rather_than_panicking() {
		let collider = Collider::new(Vec::new());
		let mut found = Vec::new();

		collider.candidates(Vec3::splat(-1.0), Vec3::splat(1.0), &mut found);

		assert!(found.is_empty());
		assert_eq!(collider.count(), 0);
	}

	#[test]
	fn the_grid_cuts_the_work_by_orders_of_magnitude() {
		// the whole reason this exists. A crate-sized box on a default-sized
		// landscape used to be tested against every triangle in it.
		let collider = landscape(65);
		let mut found = Vec::new();

		collider.candidates(Vec3::new(-1.0, -20.0, -1.0), Vec3::new(1.0, 20.0, 1.0), &mut found);

		assert!(
			found.len() * 50 < collider.count(),
			"{} of {} triangles is not a saving",
			found.len(),
			collider.count()
		);
	}

	#[test]
	fn a_flat_sheet_gets_its_cells_across_rather_than_through() {
		// a mesh with no thickness would spend its whole budget on one axis of
		// a cube grid, and a landscape is exactly that mesh.
		let collider = landscape(65);
		let [across, up, along] = collider.divisions;

		assert!(across > 1 && along > 1, "the wide axes are divided: {across} by {along}");
		assert!(up <= across, "and the thin one is not: {up}");
	}

	#[test]
	fn a_ray_through_the_grid_hits_where_the_surface_is() {
		let ground = Terrain { side: 33, ..Terrain::of(4) };
		let collider = Collider::new(shards_of(&ground.build()));
		let mut shards = Vec::new();
		let flat = colby_core::glam::Vec2::new(3.0, -5.0);
		let above = Vec3::new(flat.x, 50.0, flat.y);
		let hit = collider
			.trace(above, Vec3::NEG_Y * 100.0, &mut shards)
			.expect("a ray straight down at ground hits it");
		let landed = (Vec3::NEG_Y.y * 100.0).mul_add(hit.0, above.y);

		assert!(
			(landed - ground.height_at(flat)).abs() < 0.5,
			"landed at {landed}, the field says {}",
			ground.height_at(flat)
		);
	}

	#[test]
	fn a_ray_that_misses_the_mesh_misses_it() {
		let collider = landscape(17);
		let mut shards = Vec::new();

		assert!(
			collider
				.trace(Vec3::new(500.0, 50.0, 500.0), Vec3::NEG_Y * 100.0, &mut shards)
				.is_none()
		);
	}

	#[test]
	fn a_whole_number_out_of_a_float_is_the_number() {
		for value in [0.0, 1.0, 2.0, 7.0, 255.0, 256.0, 4095.0, 65_535.0] {
			let counted = whole(value);
			let back = fraction_of(counted);

			assert!((back - value).abs() < 0.5, "{value} came back as {back}");
		}
		assert_eq!(whole(-3.0), 0, "and nothing below nought");
	}
}
