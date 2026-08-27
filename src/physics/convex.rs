//! Convex shapes and the separating axis test over a pair of them.
//!
//! Two shapes are described here and one algorithm runs over both: a box, and a
//! single triangle of a collision mesh. Writing the test against a general
//! [`Hull`] rather than against "box versus box" is what makes a box landing on
//! a triangle the same code as a box landing on a box, and it is the reason
//! this file is the only place in the crate that knows what a face is.
//!
//! The test is textbook and the pieces are named after the textbook. Project
//! both hulls onto every candidate axis - the faces of each, and the cross
//! product of every pair of edge directions - and the smallest overlap is the
//! way out. Then build the contact: if the winning axis was a face, clip the
//! other hull's most opposed face against that face's sides, which is what
//! produces the four points a box needs to rest flat instead of rocking. If it
//! was a pair of edges, the contact is the one point where they cross.
//!
//! A triangle is **one-sided**. It has no inside, so a box that has got behind
//! one is not resolved back through it - the guard is in [`sat`], and without
//! it a prop that tunnels a floor is launched back through it rather than left
//! alone.

use colby_core::{
	abi::Transform,
	glam::{Mat3, Vec3},
};

/// How many corners the largest hull has.
const MAX_POINTS: usize = 8;

/// How many faces it has.
const MAX_FACES: usize = 6;

/// How many corners one face has.
const FACE_POINTS: usize = 4;

/// How many distinct edge directions a hull has.
const MAX_EDGES: usize = 3;

/// How many contact points one pair can produce.
pub(crate) const MAX_CONTACTS: usize = 4;

/// Axes shorter than this are two edges that were parallel, and normalizing
/// one is a division by nothing.
const EPSILON: f32 = 1.0e-6;

/// The corners of a box, indexed by which side of each axis they are on.
///
/// Bit zero is the first axis, bit one the second, bit two the third; a set bit
/// is the positive side. That is what makes a box's edges free: the neighbor of
/// a corner along axis `k` is the same index with bit `k` flipped, which is the
/// whole of the edge lookup in [`Hull::edge`].
const CORNERS: usize = 8;

/// The six faces of a box, as corner indices wound counter-clockwise seen from
/// outside, paired with which axis and sign their normal is.
///
/// Winding matters: the clipping planes are built from consecutive edges of a
/// face, so a face wound the wrong way clips against its own outside.
const BOX_FACES: [(usize, f32, [usize; FACE_POINTS]); MAX_FACES] = [
	(0, 1.0, [1, 3, 7, 5]),
	(0, -1.0, [0, 4, 6, 2]),
	(1, 1.0, [2, 6, 7, 3]),
	(1, -1.0, [0, 1, 5, 4]),
	(2, 1.0, [4, 5, 7, 6]),
	(2, -1.0, [0, 2, 3, 1]),
];

/// One face of a hull.
#[derive(Clone, Copy, Debug)]
struct Face {
	/// Which way it points, in world space.
	normal: Vec3,

	/// Its corners, as indices into the hull's points, wound
	/// counter-clockwise.
	corners: [usize; FACE_POINTS],

	/// How many of `corners` are real.
	count: usize,
}

/// A convex shape in world space, ready to be tested against another.
///
/// Built fresh per pair rather than cached: a hull is thirty-odd floats of
/// arithmetic over a transform that may have changed this step, and a cache
/// that has to be invalidated when a body moves is a cache that is wrong once.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Hull {
	/// Its corners, in world space.
	points: [Vec3; MAX_POINTS],

	/// How many of `points` are real.
	point_count: usize,

	/// Its faces.
	faces: [Face; MAX_FACES],

	/// How many of `faces` are real.
	face_count: usize,

	/// Its distinct edge directions, unit length.
	edges: [Vec3; MAX_EDGES],

	/// How many of `edges` are real.
	edge_count: usize,

	/// The middle of it, for orienting a separating axis.
	center: Vec3,

	/// Whether only the front of its first face is solid.
	///
	/// True for a triangle, which is a surface rather than a solid.
	one_sided: bool,
}

impl Hull {
	/// A box, from a body's transform and the shape's half-extents.
	///
	/// The transform's scale multiplies the extents rather than being applied
	/// to the corners afterwards, so a non-uniformly scaled box is still a box
	/// and its face normals are still its axes.
	///
	/// @param transform - where the body is
	/// @param extents - the shape's half-extents, before scale
	pub(crate) fn cuboid(transform: &Transform, extents: Vec3) -> Self {
		let rotation = Mat3::from_quat(transform.rotation);
		let axes = [rotation.x_axis, rotation.y_axis, rotation.z_axis];
		let half = extents.abs() * transform.scale.abs();
		let center = transform.position;

		let mut points = [Vec3::ZERO; MAX_POINTS];
		for (index, point) in points.iter_mut().enumerate() {
			let sign = |bit: usize| if index & (1 << bit) == 0 { -1.0 } else { 1.0 };

			*point = center
				+ axes[0] * (half.x * sign(0))
				+ axes[1] * (half.y * sign(1))
				+ axes[2] * (half.z * sign(2));
		}

		let mut faces = [Face {
			normal: Vec3::Y,
			corners: [0; FACE_POINTS],
			count: 0,
		}; MAX_FACES];

		for (face, &(axis, sign, corners)) in faces.iter_mut().zip(&BOX_FACES) {
			*face = Face {
				normal: axes[axis] * sign,
				corners,
				count: FACE_POINTS,
			};
		}

		Self {
			points,
			point_count: CORNERS,
			faces,
			face_count: MAX_FACES,
			edges: axes,
			edge_count: MAX_EDGES,
			center,
			one_sided: false,
		}
	}

	/// One triangle, already in world space.
	///
	/// @param corners - its three points
	/// @return the hull, or `None` if the triangle is degenerate and has no
	/// normal to speak of
	pub(crate) fn triangle(corners: [Vec3; 3]) -> Option<Self> {
		let [first, second, third] = corners;
		let normal = (second - first).cross(third - first);

		if normal.length_squared() < EPSILON {
			return None;
		}

		let normal = normal.normalize();
		let mut points = [Vec3::ZERO; MAX_POINTS];
		points[..3].copy_from_slice(&corners);

		let mut faces = [Face {
			normal,
			corners: [0; FACE_POINTS],
			count: 0,
		}; MAX_FACES];
		faces[0] = Face { normal, corners: [0, 1, 2, 0], count: 3 };

		let mut edges = [Vec3::X; MAX_EDGES];
		for (slot, (from, to)) in edges.iter_mut().zip([(0, 1), (1, 2), (2, 0)]) {
			*slot = (points[to] - points[from]).normalize_or(Vec3::X);
		}

		Some(Self {
			points,
			point_count: 3,
			faces,
			face_count: 1,
			edges,
			edge_count: MAX_EDGES,
			center: (first + second + third) / 3.0,
			one_sided: true,
		})
	}

	/// The three corners of a one-sided hull.
	///
	/// Only a triangle has an answer worth having; a box returns its first
	/// three corners, which is nothing anybody should ask for.
	pub(crate) fn corners(&self) -> [Vec3; 3] { [self.points[0], self.points[1], self.points[2]] }

	/// Which way the hull's first face points.
	pub(crate) fn normal(&self) -> Vec3 { self.faces[0].normal }

	/// How far this hull reaches along an axis.
	///
	/// @param axis - a unit direction
	/// @return the lowest and highest projection of any corner
	fn project(&self, axis: Vec3) -> (f32, f32) {
		let mut low = f32::INFINITY;
		let mut high = f32::NEG_INFINITY;

		for &point in &self.points[..self.point_count] {
			let along = point.dot(axis);

			low = low.min(along);
			high = high.max(along);
		}

		(low, high)
	}

	/// The corner that reaches furthest along an axis.
	///
	/// @param axis - a unit direction
	fn support(&self, axis: Vec3) -> usize {
		let mut best = 0;
		let mut furthest = f32::NEG_INFINITY;

		for (index, &point) in self.points[..self.point_count].iter().enumerate() {
			let along = point.dot(axis);

			if along > furthest {
				furthest = along;
				best = index;
			}
		}

		best
	}

	/// The edge in a given direction that reaches furthest along an axis.
	///
	/// For a box this is a corner lookup and a bit flip; for a triangle the
	/// direction *is* the edge. Both are exact, which is what keeps an
	/// edge-against-edge contact on the edges rather than on the nearest
	/// corner.
	///
	/// @param axis - which way the contact is
	/// @param direction - which of [`Hull::edges`] generated the axis
	/// @return the edge's two ends
	fn edge(&self, axis: Vec3, direction: usize) -> (Vec3, Vec3) {
		if self.one_sided {
			let from = direction.min(2);
			let to = (from + 1) % 3;

			return (self.points[from], self.points[to]);
		}

		let corner = self.support(axis);
		let other = corner ^ (1 << direction.min(2));

		(self.points[corner], self.points[other])
	}

	/// The face most opposed to a direction.
	///
	/// @param normal - the direction to be opposed to
	fn incident(&self, normal: Vec3) -> usize {
		let mut best = 0;
		let mut lowest = f32::INFINITY;

		for (index, face) in self.faces[..self.face_count].iter().enumerate() {
			let along = face.normal.dot(normal);

			if along < lowest {
				lowest = along;
				best = index;
			}
		}

		best
	}

	/// The world-space corners of one face, each tagged with which corner it
	/// is.
	///
	/// @param index - which face
	fn polygon(&self, index: usize) -> ([Corner; FACE_POINTS], usize) {
		let face = self.faces[index];
		let mut points = [Corner::default(); FACE_POINTS];

		for (slot, &corner) in points.iter_mut().zip(&face.corners[..face.count]) {
			*slot = Corner {
				point: self.points[corner],
				id: u32::try_from(corner).unwrap_or(0),
			};
		}

		(points, face.count)
	}
}

/// Which hull's feature produced the separating axis with the least overlap.
///
/// A face variant names the *hull* and not the face. It cannot name the face:
/// [`overlap`] turns every axis around to point from the first hull towards the
/// second, so the face whose normal was tested is as often as not the opposite
/// one. The reference face is re-derived from the oriented normal instead,
/// which is one lookup and cannot disagree with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
	/// A face of the first hull.
	First,

	/// A face of the second.
	Second,

	/// One edge direction from each.
	Edges(usize, usize),
}

/// The axis two hulls overlap least along.
#[derive(Clone, Copy, Debug)]
struct Separation {
	/// A unit vector pointing from the first hull towards the second.
	normal: Vec3,

	/// How far they overlap along it. Always positive.
	depth: f32,

	/// What produced it.
	source: Source,
}

/// Marks an identifier as belonging to a point cut out of an edge rather than
/// to a corner that survived.
const CUT: u32 = 1 << 8;

/// One point where two hulls touch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Touch {
	/// Where, in world space.
	pub(crate) position: Vec3,

	/// How far in. Positive when the two overlap.
	pub(crate) depth: f32,

	/// Which feature of the two hulls produced this point.
	///
	/// Not a value anybody reads - only compares. The manifold is rebuilt from
	/// scratch every step, so the solver's record of how hard each point pushed
	/// last time has to be matched to a point this time, and matching by
	/// *position* fails exactly when it matters: a box that has turned a
	/// fiftieth of a radian has moved every one of its corners further than any
	/// sensible tolerance, loses its whole warm start, and is then far more
	/// likely to turn again. An identifier survives that, because a corner is
	/// still the same corner however it has moved.
	pub(crate) id: u32,
}

/// One corner of a polygon being clipped, and where it came from.
#[derive(Clone, Copy, Debug, Default)]
struct Corner {
	/// Where it is, in world space.
	point: Vec3,

	/// Which feature it is. @ref [`Touch::id`].
	id: u32,
}

/// Where two hulls touch, if they do.
///
/// @param first - one hull
/// @param second - the other
/// @return the contact normal pointing from the first towards the second, the
/// points, and how many are real
pub(crate) fn collide(
	first: &Hull,
	second: &Hull,
) -> Option<(Vec3, [Touch; MAX_CONTACTS], usize)> {
	let separation = sat(first, second)?;

	let (points, count) = match separation.source {
		| Source::Edges(one, other) => (crossing(first, second, &separation, one, other), 1),
		| Source::First => clipped(first, second, separation.normal),
		| Source::Second => clipped(second, first, -separation.normal),
	};

	if count == 0 {
		return None;
	}

	Some((separation.normal, points, count))
}

/// The separating axis test.
///
/// @param first - one hull
/// @param second - the other
/// @return the least-overlapping axis, or `None` if any axis separates them
fn sat(first: &Hull, second: &Hull) -> Option<Separation> {
	let mut best: Option<Separation> = None;

	// a surface is measured against its own plane rather than as an interval:
	// a triangle has no thickness, so the overlap along its own normal is
	// exactly zero however deep something has sunk through it, and a plain
	// interval test reports every contact with a floor as a miss.
	if first.one_sided {
		best = keep(best, plane(first, second, 1.0, Source::First)?);
	}

	if second.one_sided {
		best = keep(best, plane(second, first, -1.0, Source::Second)?);
	}

	for index in 0..first.face_count {
		let axis = first.faces[index].normal;

		if first.one_sided || along_surface(second, axis) {
			continue;
		}

		best = keep(best, overlap(first, second, axis, Source::First)?);
	}

	for index in 0..second.face_count {
		let axis = second.faces[index].normal;

		if second.one_sided || along_surface(first, axis) {
			continue;
		}

		best = keep(best, overlap(first, second, axis, Source::Second)?);
	}

	for one in 0..first.edge_count {
		for other in 0..second.edge_count {
			let axis = first.edges[one].cross(second.edges[other]);

			if axis.length_squared() < EPSILON {
				continue;
			}

			let axis = axis.normalize();

			// an in-plane edge of a triangle crossed with an edge of the box
			// lying in that same plane reproduces the triangle's own normal
			// exactly, where the interval test is degenerate - and that
			// direction is what `plane` above already measured. Skipping it is
			// dropping a duplicate, not dropping a case.
			if along_surface(first, axis) || along_surface(second, axis) {
				continue;
			}

			let source = Source::Edges(one, other);
			best = keep(best, overlap(first, second, axis, source)?);
		}
	}

	let best = best?;

	// a triangle has no inside, so something that got behind one is not pushed
	// back out through it. Without this a prop that tunnels a floor in one step
	// is fired back up through it in the next, which reads as an explosion.
	if second.one_sided && (first.center - second.points[0]).dot(second.faces[0].normal) < 0.0 {
		return None;
	}

	if first.one_sided && (second.center - first.points[0]).dot(first.faces[0].normal) < 0.0 {
		return None;
	}

	Some(best)
}

/// How far two hulls overlap along one axis.
///
/// @param first - one hull
/// @param second - the other
/// @param axis - a unit direction, in either orientation
/// @param source - what produced the axis
/// @return the overlap, oriented from the first hull towards the second, or
/// `None` if this axis separates them
fn overlap(first: &Hull, second: &Hull, axis: Vec3, source: Source) -> Option<Separation> {
	let (low_first, high_first) = first.project(axis);
	let (low_second, high_second) = second.project(axis);

	let depth = high_first.min(high_second) - low_first.max(low_second);

	if depth <= 0.0 {
		return None;
	}

	// the axis arrives in whichever orientation the face or the cross product
	// happened to have. Turning it to point from the first hull to the second
	// is what lets everything downstream stop thinking about signs.
	let normal = if (second.center - first.center).dot(axis) < 0.0 {
		-axis
	} else {
		axis
	};

	Some(Separation { normal, depth, source })
}

/// Whether an axis is the normal of a one-sided hull.
///
/// @param hull - the hull to ask about
/// @param axis - a unit direction, in either orientation
fn along_surface(hull: &Hull, axis: Vec3) -> bool {
	hull.one_sided && axis.dot(hull.faces[0].normal).abs() > 1.0 - 1.0e-3
}

/// How far a solid has sunk through a surface.
///
/// The interval test in [`overlap`] cannot answer this: a triangle's own extent
/// along its own normal is a point, so the two intervals never overlap by
/// anything at all. What is wanted instead is how far past the plane the other
/// hull's nearest corner has got.
///
/// @param surface - the one-sided hull
/// @param solid - the other one
/// @param facing - `1.0` if the surface is the *first* hull of the pair and the
/// contact normal therefore points out of it, `-1.0` if it is the second
/// @param source - what produced the axis
/// @return the overlap, or `None` if the solid is entirely in front of the
/// plane
fn plane(surface: &Hull, solid: &Hull, facing: f32, source: Source) -> Option<Separation> {
	let normal = surface.faces[0].normal;
	let offset = surface.points[0].dot(normal);
	let (low, _) = solid.project(normal);
	let depth = offset - low;

	if depth <= 0.0 {
		return None;
	}

	Some(Separation { normal: normal * facing, depth, source })
}

/// Keeps whichever of two separations overlaps less.
fn keep(best: Option<Separation>, candidate: Separation) -> Option<Separation> {
	match best {
		| Some(best) if best.depth <= candidate.depth => Some(best),
		| _ => Some(candidate),
	}
}

/// The single point where two edges cross.
///
/// @param first - one hull
/// @param second - the other
/// @param separation - the winning axis
/// @param one - which edge direction of the first hull produced it
/// @param other - which of the second
fn crossing(
	first: &Hull,
	second: &Hull,
	separation: &Separation,
	one: usize,
	other: usize,
) -> [Touch; MAX_CONTACTS] {
	let (from_first, to_first) = first.edge(separation.normal, one);
	let (from_second, to_second) = second.edge(-separation.normal, other);
	let position = closest(from_first, to_first, from_second, to_second);

	let mut points = [Touch::default(); MAX_CONTACTS];
	points[0] = Touch {
		position,
		depth: separation.depth,
		// an edge against an edge is one point, and which two edges made it is
		// the whole of its identity.
		id: CUT | u32::try_from((one << 4) | other).unwrap_or(0),
	};

	points
}

/// The midpoint of the shortest line between two segments.
///
/// @param from_first - one end of the first segment
/// @param to_first - the other
/// @param from_second - one end of the second
/// @param to_second - the other
fn closest(from_first: Vec3, to_first: Vec3, from_second: Vec3, to_second: Vec3) -> Vec3 {
	let first = to_first - from_first;
	let second = to_second - from_second;
	let between = from_first - from_second;

	let (along_one, shared) = (first.dot(first), first.dot(second));
	let along_other = second.dot(second);
	let (offset_one, offset_other) = (first.dot(between), second.dot(between));
	let determinant = along_one.mul_add(along_other, -(shared * shared));

	// parallel edges do not produce a cross-product axis at all, so this is a
	// guard against arithmetic rather than against a case.
	if determinant.abs() < EPSILON {
		return from_first.midpoint(from_second);
	}

	let along_first = shared.mul_add(offset_other, -(along_other * offset_one)) / determinant;
	let along_second = along_one.mul_add(offset_other, -(shared * offset_one)) / determinant;

	let on_first = from_first + first * along_first.clamp(0.0, 1.0);
	let on_second = from_second + second * along_second.clamp(0.0, 1.0);

	on_first.midpoint(on_second)
}

/// The points of a face-against-face contact.
///
/// The other hull's most opposed face is clipped against the sides of the
/// reference face, and whatever is left below the reference plane is the
/// contact. This is what gives a box four points to rest on rather than one to
/// rock about.
///
/// @param reference - the hull whose face won the separating axis test
/// @param incident - the other hull
/// @param normal - the contact normal, pointing out of `reference`
fn clipped(reference: &Hull, incident: &Hull, normal: Vec3) -> ([Touch; MAX_CONTACTS], usize) {
	// the face pointing most nearly the way the contact does, which is what the
	// separating axis meant whichever orientation it was found in.
	let face = reference.incident(-normal);
	let (border, border_count) = reference.polygon(face);
	let target = incident.incident(normal);
	let (mut polygon, mut count) = incident.polygon(target);

	for index in 0..border_count {
		let from = border[index].point;
		let to = border[(index + 1) % border_count].point;
		let side = (to - from).cross(normal).normalize_or(Vec3::ZERO);

		if side == Vec3::ZERO {
			continue;
		}

		let plane = u32::try_from(index).unwrap_or(0);
		let (kept, left) = cut(&polygon, count, side, side.dot(from), plane);
		polygon = kept;
		count = left;

		if count == 0 {
			return ([Touch::default(); MAX_CONTACTS], 0);
		}
	}

	let plane = normal.dot(border[0].point);
	let mut points = [Touch::default(); MAX_CONTACTS];
	let mut found = 0;

	for &corner in &polygon[..count] {
		let separation = normal.dot(corner.point) - plane;

		if separation > 0.0 || found == MAX_CONTACTS {
			continue;
		}

		points[found] = Touch {
			position: corner.point,
			depth: -separation,
			id: corner.id,
		};
		found += 1;
	}

	(points, found)
}

/// Keeps the part of a polygon on the inner side of a plane.
///
/// Sutherland-Hodgman, one plane at a time.
///
/// @param polygon - the corners
/// @param count - how many are real
/// @param normal - the plane's outward normal
/// @param offset - the plane's distance along that normal
/// @param plane - which side of the reference face this is, for identifying
/// whatever gets cut out of an edge
/// @return the clipped corners and how many there are
fn cut(
	polygon: &[Corner; FACE_POINTS],
	count: usize,
	normal: Vec3,
	offset: f32,
	plane: u32,
) -> ([Corner; FACE_POINTS], usize) {
	let mut kept = [Corner::default(); FACE_POINTS];
	let mut found = 0;
	let mut push = |corner: Corner, found: &mut usize| {
		if *found < FACE_POINTS {
			kept[*found] = corner;
			*found += 1;
		}
	};

	for index in 0..count {
		let from = polygon[index];
		let to = polygon[(index + 1) % count];
		let (near, far) = (normal.dot(from.point) - offset, normal.dot(to.point) - offset);

		if near <= 0.0 {
			push(from, &mut found);
		}

		if (near > 0.0) == (far > 0.0) {
			continue;
		}

		let split = near / (near - far);
		push(
			Corner {
				point: from.point.lerp(to.point, split),
				// which edge was cut, and by which side. Both halves are needed:
				// one edge can be cut twice, and two edges can be cut by the
				// same side.
				id: CUT | (from.id << 4) | plane,
			},
			&mut found,
		);
	}

	(kept, found)
}

#[cfg(test)]
mod tests {
	use colby_core::glam::Quat;

	use super::*;

	/// A triangle in the xz plane, wound counter-clockwise seen from above, so
	/// its normal is +Y and its front is the side something stands on.
	const FLOOR: [Vec3; 3] =
		[Vec3::new(-4.0, 0.0, -4.0), Vec3::new(0.0, 0.0, 6.0), Vec3::new(4.0, 0.0, -4.0)];

	fn at(position: Vec3) -> Transform { Transform::at(position) }

	#[test]
	fn two_identical_boxes_stacked_touch_at_all_four_corners() {
		let lower = Hull::cuboid(&at(Vec3::ZERO), Vec3::splat(0.31));
		let upper = Hull::cuboid(&at(Vec3::new(0.0, 0.60, 0.0)), Vec3::splat(0.31));
		let (normal, points, count) = collide(&lower, &upper).expect("overlapping by a fiftieth");

		assert_eq!(count, 4, "clipping a face against one exactly its own size keeps all four");
		assert!(normal.abs_diff_eq(Vec3::Y, 1.0e-5), "pointing up out of the lower one");

		for point in &points[..count] {
			assert!(
				(point.depth - 0.02).abs() < 1.0e-4,
				"and each is the same depth, or the box above is pinched and turns: got {}",
				point.depth
			);
			assert!(
				(point.position.y - 0.29).abs() < 1.0e-4,
				"on the shared face, got {}",
				point.position
			);
		}
	}

	#[test]
	fn a_contact_keeps_its_identifier_when_the_body_turns() {
		let lower = Hull::cuboid(&at(Vec3::ZERO), Vec3::splat(0.31));
		let named = |angle: f32| {
			let mut turned = at(Vec3::new(0.0, 0.60, 0.0));
			turned.rotation = Quat::from_rotation_y(angle);
			let upper = Hull::cuboid(&turned, Vec3::splat(0.31));
			let (_, points, count) = collide(&lower, &upper).expect("still overlapping");
			let mut ids: Vec<u32> = points[..count].iter().map(|it| it.id).collect();
			ids.sort_unstable();

			ids
		};

		// what a step actually looks like: a body that has barely moved. Every
		// corner has traveled further than any position tolerance worth having,
		// and the warm start has to survive that.
		let before = named(0.02);
		let after = named(0.0201);

		assert_eq!(before.len(), 4, "four points either way");
		assert_eq!(before, after, "and the same four, named the same way");

		// a body that has *not* barely moved is a different overlap with
		// genuinely different points, which is also right: the corners of a
		// square face stop being the contact the moment the face is clipped.
		assert_ne!(
			named(0.0),
			before,
			"an unturned box meets face to face and is not clipped at all, so its points are 			 its own corners rather than pieces of its edges"
		);
	}

	#[test]
	fn the_four_corners_of_a_face_are_told_apart() {
		let lower = Hull::cuboid(&at(Vec3::ZERO), Vec3::splat(0.31));
		let upper = Hull::cuboid(&at(Vec3::new(0.0, 0.60, 0.0)), Vec3::splat(0.31));
		let (_, points, count) = collide(&lower, &upper).expect("overlapping");
		let mut ids: Vec<u32> = points[..count].iter().map(|it| it.id).collect();

		ids.sort_unstable();
		ids.dedup();

		assert_eq!(ids.len(), 4, "or two corners share a record of how hard they pushed");
	}

	#[test]
	fn two_boxes_that_do_not_touch_report_nothing() {
		let first = Hull::cuboid(&at(Vec3::ZERO), Vec3::splat(0.5));
		let second = Hull::cuboid(&at(Vec3::new(0.0, 4.0, 0.0)), Vec3::splat(0.5));

		assert!(collide(&first, &second).is_none(), "four units apart is apart");
	}

	#[test]
	fn a_box_resting_flat_on_another_has_four_points() {
		let ground = Hull::cuboid(&at(Vec3::ZERO), Vec3::new(4.0, 0.5, 4.0));
		let resting = Hull::cuboid(&at(Vec3::new(0.0, 0.9, 0.0)), Vec3::splat(0.5));
		let (normal, points, count) =
			collide(&ground, &resting).expect("they overlap by a tenth");

		assert_eq!(count, 4, "a face against a face is four points, or the box rocks");
		assert!(
			normal.abs_diff_eq(Vec3::Y, 1.0e-4),
			"pointing from the ground up at the box, got {normal}"
		);

		for point in &points[..count] {
			assert!((point.depth - 0.1).abs() < 1.0e-4, "each a tenth deep, got {}", point.depth);
		}
	}

	#[test]
	fn the_normal_turns_around_when_the_hulls_are_given_the_other_way_round() {
		let ground = Hull::cuboid(&at(Vec3::ZERO), Vec3::new(4.0, 0.5, 4.0));
		let resting = Hull::cuboid(&at(Vec3::new(0.0, 0.9, 0.0)), Vec3::splat(0.5));

		let (down, ..) = collide(&resting, &ground).expect("still overlapping");

		assert!(
			down.abs_diff_eq(Vec3::NEG_Y, 1.0e-4),
			"from the box down at the ground, got {down}"
		);
	}

	#[test]
	fn a_box_balanced_on_a_corner_touches_at_one_point() {
		let ground = Hull::cuboid(&at(Vec3::ZERO), Vec3::new(4.0, 0.5, 4.0));
		let mut tipped = at(Vec3::new(0.0, 1.29, 0.0));
		tipped.rotation =
			Quat::from_rotation_x(core::f32::consts::FRAC_PI_4) * Quat::from_rotation_z(0.6);
		let corner = Hull::cuboid(&tipped, Vec3::splat(0.5));

		let (normal, _, count) = collide(&ground, &corner).expect("the corner is in the slab");

		assert_eq!(count, 1, "a corner is one point, got {count}");
		assert!(normal.dot(Vec3::Y) > 0.9, "and it is pushed upwards, got {normal}");
	}

	#[test]
	fn a_box_on_a_triangle_is_pushed_along_the_triangles_normal() {
		let floor = Hull::triangle(FLOOR).expect("not degenerate");
		let resting = Hull::cuboid(&at(Vec3::new(0.0, 0.45, 0.0)), Vec3::splat(0.5));

		let (normal, points, count) =
			collide(&floor, &resting).expect("it sinks in by a twentieth");

		assert!(count >= 1, "at least one point, got {count}");
		assert!(normal.abs_diff_eq(Vec3::Y, 1.0e-4), "up out of the triangle, got {normal}");
		assert!(
			points[..count]
				.iter()
				.all(|point| (point.depth - 0.05).abs() < 1.0e-3),
			"each a twentieth deep"
		);
	}

	#[test]
	fn a_box_behind_a_triangle_is_left_alone() {
		let floor = Hull::triangle(FLOOR).expect("not degenerate");
		let under = Hull::cuboid(&at(Vec3::new(0.0, -0.45, 0.0)), Vec3::splat(0.5));

		assert!(
			collide(&floor, &under).is_none(),
			"a surface has no inside, so nothing is pushed back through it"
		);
	}

	#[test]
	fn the_winding_is_what_decides_which_side_of_a_triangle_is_solid() {
		let [first, second, third] = FLOOR;
		let backwards = Hull::triangle([first, third, second]).expect("still a triangle");
		let resting = Hull::cuboid(&at(Vec3::new(0.0, 0.45, 0.0)), Vec3::splat(0.5));

		assert!(
			collide(&backwards, &resting).is_none(),
			"the same three points wound the other way face downwards, and something above 			 \
			 them is behind them"
		);
	}

	#[test]
	fn a_degenerate_triangle_is_refused_rather_than_normalized() {
		let line = [Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)];

		assert!(Hull::triangle(line).is_none(), "three points on a line have no normal");
	}

	#[test]
	fn a_box_overlapping_a_corner_of_another_is_clipped_to_what_is_shared() {
		let ground = Hull::cuboid(&at(Vec3::ZERO), Vec3::new(1.0, 0.5, 1.0));
		// hanging off the edge, so most of the incident face is outside the
		// reference face and has to be cut away.
		let hanging = Hull::cuboid(&at(Vec3::new(0.75, 0.9, 0.0)), Vec3::splat(0.5));

		let (_, points, count) = collide(&ground, &hanging).expect("the overlap is real");

		assert!(count > 0, "something is touching");
		assert!(
			points[..count]
				.iter()
				.all(|point| point.position.x <= 1.0 + 1.0e-4),
			"and every point of it is over the ground rather than past its edge"
		);
	}
}
