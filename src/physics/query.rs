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
	glam::{Mat3, Vec3},
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
	nearest(bodies, info, |id, body| {
		let collider = simulation.collider(id);

		ray_body(body, collider, info.start, info.end)
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

	nearest(bodies, info, |id, body| {
		let collider = simulation.collider(id);
		let (low, high) = world_bounds(body, collider)?;
		let (low, high) = (low - sweep.extents, high + sweep.extents);
		let (enter, exit, coarse) = span(sweep.start, sweep.direction, low, high)?;

		refine(body, collider, sweep, (enter, exit), coarse)
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
) -> Option<Hit> {
	let touching = |fraction: f32| overlap(body, collider, sweep.at(fraction), sweep.extents);
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

	let Some((mut clear, mut solid, mut normal)) = walk(&touching, (enter, exit), stride) else {
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
	touching: &impl Fn(f32) -> Option<Vec3>,
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
/// @return the surface normal out of the body, or `None` if they are apart
fn overlap(body: &Body, collider: Option<&Collider>, at: Vec3, extents: Vec3) -> Option<Vec3> {
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
		| ShapeKind::Mesh => triangles(body, collider?, at, extents),
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
/// Linear, behind the same bounds rejection the narrow phase uses. A hierarchy
/// goes in [`Collider`] when something wants one, and this does not change.
///
/// @param body - the mesh body, for its transform
/// @param collider - its baked triangles
/// @param at - where the box's middle is
/// @param extents - its half-extents
/// @return the normal out of the first triangle that overlaps, or `None`
fn triangles(body: &Body, collider: &Collider, at: Vec3, extents: Vec3) -> Option<Vec3> {
	let matrix = body.transform.matrix();
	let cast = Hull::cuboid(&Transform::at(at), extents);
	let (low, high) = (at - extents, at + extents);

	for corners in collider.triangles() {
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
fn ray_body(body: &Body, collider: Option<&Collider>, start: Vec3, end: Vec3) -> Option<Hit> {
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
		| ShapeKind::Mesh => collider.and_then(|it| it.trace(local_start, local)),
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

/// A body's geometry, baked once and kept in the body's own space.
///
/// Its own copy rather than a look into [`Meshes`](colby_core::abi::Meshes),
/// and that is a rule rather than an accident: recompiling an `.obj` replaces
/// what is drawn and leaves what is collided against alone until the body is
/// created again. A collision mesh is a resource with its own preparation - a
/// hierarchy over it is the next thing that goes in here - not a second view
/// onto a vertex buffer that may be halfway through being rewritten.
#[derive(Clone, Debug, Default)]
pub(crate) struct Collider {
	/// Every triangle, in the body's own space.
	triangles: Vec<[Vec3; 3]>,

	/// The low corner of their bounds.
	low: Vec3,

	/// The high corner.
	high: Vec3,
}

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

		Self { triangles, low, high }
	}

	/// How many triangles this collides with.
	pub(crate) fn count(&self) -> usize { self.triangles.len() }

	/// Every triangle, in the body's own space.
	pub(crate) fn triangles(&self) -> impl Iterator<Item = [Vec3; 3]> {
		self.triangles.iter().copied()
	}

	/// A segment against every triangle.
	///
	/// Linear, with a bounds rejection in front of it. A hierarchy is what goes
	/// here when a collision mesh is bigger than a floor, and nothing above
	/// this function would notice it arriving.
	///
	/// @param origin - where the segment begins, in the body's own space
	/// @param direction - the whole segment
	fn trace(&self, origin: Vec3, direction: Vec3) -> Option<(f32, Vec3)> {
		slab(origin, direction, self.low, self.high)?;

		let mut best: Option<(f32, Vec3)> = None;

		for &corners in &self.triangles {
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
