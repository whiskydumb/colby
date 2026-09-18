//! The hierarchy of boxes a ray is traced through.
//!
//! A box around every triangle, a box around every few of those, and so on up
//! to one box around the world, so that a ray asks a few dozen boxes and a few
//! triangles where a walk over the whole list would ask every triangle there
//! is. The boxes are cut where the surface area says a ray is least likely to
//! have to look into both halves, which is the rule every tracer in the field
//! builds by; the centers are sorted into a dozen buckets rather than tried one
//! at a time, which is where nearly all of that rule's benefit is for a
//! fraction of its cost.
//!
//! **Built once, walked by many threads, and the same tree every time.** The
//! build is one thread over the triangles in the order they were handed in,
//! every tie broken by that order, so two builds of one scene are one tree and
//! a walk through it meets the same triangles in the same order. That is what
//! lets a bake promise one hash.
//!
//! **What a triangle is to a ray is a corner and two edges**, kept here in the
//! order the leaves read them rather than in the scene's order, so that the
//! triangles a leaf holds sit side by side in memory. [`Hit::triangle`] gives
//! the scene's index back.

use std::ops::ControlFlow;

use colby_core::glam::Vec3;

/// How many triangles a box may hold before it is cut in two.
const LEAF: usize = 4;

/// How many buckets the centers are sorted into to choose a cut.
const BINS: usize = 12;

/// How deep a box may be and still be cut where the buckets say.
///
/// A cut the buckets choose may leave one side much larger than the other, and
/// a pathological scene could chain such cuts into a tree as deep as it has
/// triangles. Past this depth a box is cut at its middle triangle instead,
/// which halves it: thirty more halvings take four thousand million triangles
/// down to a leaf, so no tree is deeper than this and thirty.
const HALVING: usize = 24;

/// How many boxes a walk may have waiting.
///
/// One a level at most - the nearer child is opened at once and only the
/// farther one waits - so a tree [`HALVING`] and thirty deep needs fifty-five.
const STACK: usize = 64;

/// A ray: where it starts and which way it goes.
///
/// The way it goes need not be of unit length, and a hit's distance is then in
/// lengths of it rather than in world units; everything in this crate hands
/// over a unit direction, so the two are the same.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
	/// Where it starts.
	pub origin: Vec3,

	/// Which way it goes.
	pub direction: Vec3,

	/// One over each axis of the direction, worked out once for every box it
	/// meets. An axis of nought gives an infinity, which is what the slab test
	/// wants: a ray running along a box's face is inside that slab everywhere
	/// or nowhere.
	inverse: Vec3,
}

impl Ray {
	/// A ray from a point in a direction.
	///
	/// @param origin - where it starts
	/// @param direction - which way it goes
	#[must_use]
	pub fn new(origin: Vec3, direction: Vec3) -> Self {
		Self {
			origin,
			direction,
			inverse: direction.recip(),
		}
	}
}

/// Where a ray met a triangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
	/// How far along the ray, in lengths of its direction.
	pub distance: f32,

	/// Which triangle, as the scene numbers them.
	pub triangle: u32,

	/// How much of the second corner the point is, from nought to one.
	pub along: f32,

	/// How much of the third corner. The first's share is what the two leave.
	pub across: f32,

	/// Whether the ray met the side the triangle's winding faces - the outside
	/// of whatever it is part of - rather than its back.
	pub front: bool,
}

/// One box of the tree.
#[derive(Clone, Copy, Debug)]
struct Node {
	/// The box's lowest corner.
	low: Vec3,

	/// Its highest.
	high: Vec3,

	/// For a leaf, where its triangles start; for a box with children, where
	/// the first of the two is.
	first: u32,

	/// How many triangles a leaf holds, or nought for a box with children,
	/// which sit at `first` and one after it.
	count: u32,
}

impl Node {
	/// A box that holds nothing yet, as a build reserves it.
	const EMPTY: Self = Self {
		low: Vec3::ZERO,
		high: Vec3::ZERO,
		first: 0,
		count: 0,
	};
}

/// The boxes and the triangles they hold.
#[derive(Clone, Debug, Default)]
pub struct Tree {
	/// Every box, the root first.
	nodes: Vec<Node>,

	/// Every triangle as a corner and the two edges from it, in the order the
	/// leaves read them.
	shapes: Vec<[Vec3; 3]>,

	/// The scene's number for each of those.
	ids: Vec<u32>,
}

/// What building the tree works on: each triangle's box and middle.
struct Build<'a> {
	lows: &'a [Vec3],
	highs: &'a [Vec3],
	centers: &'a [Vec3],
}

/// One bucket the centers are sorted into.
#[derive(Clone, Copy)]
struct Bin {
	low: Vec3,
	high: Vec3,
	count: usize,
}

impl Bin {
	/// A bucket nothing is in.
	const EMPTY: Self = Self {
		low: Vec3::INFINITY,
		high: Vec3::NEG_INFINITY,
		count: 0,
	};

	/// This bucket with one more box in it.
	fn grown(self, low: Vec3, high: Vec3) -> Self {
		Self {
			low: low.min(self.low),
			high: high.max(self.high),
			count: self.count + 1,
		}
	}

	/// This bucket and another, as one.
	fn joined(self, other: Self) -> Self {
		Self {
			low: self.low.min(other.low),
			high: self.high.max(other.high),
			count: self.count + other.count,
		}
	}

	/// Half the surface of its box, or nought when nothing is in it: what a
	/// cut is priced by.
	fn area(self) -> f32 {
		if self.count == 0 {
			return 0.0;
		}

		half_surface(self.low, self.high)
	}
}

impl Tree {
	/// Builds the tree over a list of triangles.
	///
	/// @param triangles - each as its three corners, in the order the scene
	/// numbers them
	#[must_use]
	pub fn build(triangles: &[[Vec3; 3]]) -> Self {
		let lows: Vec<Vec3> = triangles
			.iter()
			.map(|[a, b, c]| a.min(*b).min(*c))
			.collect();
		let highs: Vec<Vec3> = triangles
			.iter()
			.map(|[a, b, c]| a.max(*b).max(*c))
			.collect();
		let centers: Vec<Vec3> = lows
			.iter()
			.zip(&highs)
			.map(|(low, high)| (*low + *high) * 0.5)
			.collect();
		let build = Build {
			lows: &lows,
			highs: &highs,
			centers: &centers,
		};
		let mut order: Vec<u32> = (0..triangles.len())
			.filter_map(|index| u32::try_from(index).ok())
			.collect();
		let mut nodes = Vec::new();

		if !order.is_empty() {
			nodes.push(Node::EMPTY);
		}

		// what is still to be cut: a box's place, the run of the order it
		// holds, and how deep it is. Last in, first out, so a box's children
		// are finished before its neighbor is started - which does not change
		// the tree, only the order its boxes are written in.
		let mut pending = vec![(0_usize, 0_usize, order.len(), 0_usize)];

		while let Some((at, start, end, depth)) = pending.pop() {
			if start == end {
				continue;
			}

			let held = &mut order[start..end];
			let (low, high) = build.bounds(held);
			let cut = build.cut(held, depth);

			let Some(cut) = cut else {
				nodes[at] = Node {
					low,
					high,
					first: u32::try_from(start).unwrap_or(u32::MAX),
					count: u32::try_from(end - start).unwrap_or(u32::MAX),
				};

				continue;
			};

			let first = nodes.len();

			nodes.push(Node::EMPTY);
			nodes.push(Node::EMPTY);
			nodes[at] = Node {
				low,
				high,
				first: u32::try_from(first).unwrap_or(u32::MAX),
				count: 0,
			};
			pending.push((first + 1, start + cut, end, depth + 1));
			pending.push((first, start, start + cut, depth + 1));
		}

		let shapes = order
			.iter()
			.filter_map(|&id| triangles.get(usize::try_from(id).ok()?))
			.map(|[a, b, c]| [*a, *b - *a, *c - *a])
			.collect();

		Self { nodes, shapes, ids: order }
	}

	/// How many triangles the tree holds.
	#[must_use]
	pub fn len(&self) -> usize { self.ids.len() }

	/// Whether it holds none.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.ids.is_empty() }

	/// How many boxes it is made of.
	#[must_use]
	pub fn boxes(&self) -> usize { self.nodes.len() }

	/// The nearest triangle a ray meets before it has gone so far.
	///
	/// Either side of a triangle counts: which side it met is on the hit, and
	/// what a back face means is the asker's business.
	///
	/// @param ray - the ray
	/// @param reach - how far it goes, in lengths of its direction
	/// @return the nearest hit short of the reach, or `None`
	#[must_use]
	pub fn nearest(&self, ray: &Ray, reach: f32) -> Option<Hit> {
		let mut found = None;

		self.walk(ray, reach, |shape, id, reach| {
			let Some((distance, along, across, front)) = crossing(shape, ray, reach) else {
				return ControlFlow::Continue(None);
			};

			found = Some(Hit {
				distance,
				triangle: id,
				along,
				across,
				front,
			});

			ControlFlow::Continue(Some(distance))
		});

		found
	}

	/// Whether anything at all is in a ray's way before it has gone so far.
	///
	/// Cheaper than [`nearest`](Self::nearest): the walk stops at the first
	/// triangle it meets, which is all a shadow asks.
	///
	/// @param ray - the ray
	/// @param reach - how far it goes, in lengths of its direction
	#[must_use]
	pub fn blocked(&self, ray: &Ray, reach: f32) -> bool {
		let mut met = false;

		self.walk(ray, reach, |shape, _, reach| {
			if crossing(shape, ray, reach).is_some() {
				met = true;

				return ControlFlow::Break(());
			}

			ControlFlow::Continue(None)
		});

		met
	}

	/// Walks every box a ray passes through, nearer children first, and hands
	/// each triangle of each leaf to a visitor.
	///
	/// The visitor answers a new reach when the triangle shortened it, after
	/// which every box that starts past it is skipped, or stops the walk.
	///
	/// @param ray - the ray
	/// @param reach - how far it goes to begin with
	/// @param visit - handed a triangle, its scene number and the reach so far
	fn walk<Visit: FnMut(&[Vec3; 3], u32, f32) -> ControlFlow<(), Option<f32>>>(
		&self,
		ray: &Ray,
		reach: f32,
		mut visit: Visit,
	) {
		if self.nodes.is_empty() {
			return;
		}

		let mut reach = reach;
		let mut stack = [0_u32; STACK];
		let mut top = 1_usize;

		while top > 0 {
			top -= 1;

			let Some(node) = self.node(stack[top]) else {
				continue;
			};

			if entry(node, ray, reach).is_none() {
				continue;
			}

			if node.count > 0 {
				match self.leaf(node, reach, &mut visit) {
					| ControlFlow::Break(()) => return,
					| ControlFlow::Continue(shorter) => reach = shorter,
				}

				continue;
			}

			for child in self
				.children(node, ray, reach)
				.into_iter()
				.flatten()
			{
				top = pushed(&mut stack, top, child);
			}
		}
	}

	/// Hands every triangle of a leaf to a visitor.
	///
	/// @return the reach the triangles left, or a break when the visitor asked
	/// for one
	fn leaf<Visit: FnMut(&[Vec3; 3], u32, f32) -> ControlFlow<(), Option<f32>>>(
		&self,
		node: &Node,
		reach: f32,
		visit: &mut Visit,
	) -> ControlFlow<(), f32> {
		let mut reach = reach;

		for index in node.first..node.first.saturating_add(node.count) {
			let slot = usize::try_from(index).unwrap_or(usize::MAX);
			let (Some(shape), Some(&id)) = (self.shapes.get(slot), self.ids.get(slot)) else {
				continue;
			};

			match visit(shape, id, reach)? {
				| Some(shorter) => reach = shorter,
				| None => {},
			}
		}

		ControlFlow::Continue(reach)
	}

	/// The children of a box a ray enters, in the order they go on the stack:
	/// the farther first, so that the nearer is looked into first and may
	/// shorten the reach before the other is opened.
	fn children(&self, node: &Node, ray: &Ray, reach: f32) -> [Option<u32>; 2] {
		let (first, second) = (node.first, node.first.saturating_add(1));
		let near = self
			.node(first)
			.and_then(|child| entry(child, ray, reach));
		let far = self
			.node(second)
			.and_then(|child| entry(child, ray, reach));

		match (near, far) {
			| (Some(one), Some(two)) if two < one => [Some(first), Some(second)],
			| (Some(_), Some(_)) => [Some(second), Some(first)],
			| (Some(_), None) => [None, Some(first)],
			| (None, Some(_)) => [None, Some(second)],
			| (None, None) => [None, None],
		}
	}

	/// A box, by its place.
	fn node(&self, index: u32) -> Option<&Node> { self.nodes.get(usize::try_from(index).ok()?) }
}

impl Build<'_> {
	/// The box around a run of triangles.
	fn bounds(&self, held: &[u32]) -> (Vec3, Vec3) {
		held.iter()
			.filter_map(|&id| usize::try_from(id).ok())
			.fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(low, high), id| {
				(low.min(self.lows[id]), high.max(self.highs[id]))
			})
	}

	/// Where to cut a run of triangles in two, reordering it so the first half
	/// is the part before the cut.
	///
	/// @param held - the run, reordered in place
	/// @param depth - how deep the box holding it is
	/// @return how many go to the first child, or `None` for a leaf
	fn cut(&self, held: &mut [u32], depth: usize) -> Option<usize> {
		if held.len() <= LEAF {
			return None;
		}

		let (low, high) = held
			.iter()
			.filter_map(|&id| usize::try_from(id).ok())
			.fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(low, high), id| {
				(low.min(self.centers[id]), high.max(self.centers[id]))
			});
		let extent = high - low;
		let axis = longest(extent);
		let span = extent[axis];

		// every middle in one place: nothing to cut between, and a leaf of
		// many is the honest answer
		if span.is_nan() || span <= 0.0 {
			return None;
		}

		if depth >= HALVING {
			return Some(self.halved(held, axis));
		}

		let bin = |id: u32| {
			usize::try_from(id)
				.ok()
				.map_or(0, |id| bucket(self.centers[id][axis], low[axis], span))
		};
		let mut bins = [Bin::EMPTY; BINS];

		for &id in held.iter() {
			let Ok(slot) = usize::try_from(id) else {
				continue;
			};

			let at = bin(id);
			bins[at] = bins[at].grown(self.lows[slot], self.highs[slot]);
		}

		let Some(after) = cheapest(&bins) else {
			return Some(self.halved(held, axis));
		};

		// a stable partition: the order inside each half is the order they
		// arrived in, which is what makes two builds one tree
		let (first, second): (Vec<u32>, Vec<u32>) =
			held.iter().partition(|&&id| bin(id) <= after);
		let cut = first.len();

		for (slot, id) in held
			.iter_mut()
			.zip(first.into_iter().chain(second))
		{
			*slot = id;
		}

		(cut > 0 && cut < held.len()).then_some(cut)
	}

	/// Cuts a run at its middle triangle along an axis.
	///
	/// The fallback for when the buckets cannot tell the triangles apart, and
	/// for a box so deep that a lopsided cut could make the tree deeper than a
	/// walk's stack.
	fn halved(&self, held: &mut [u32], axis: usize) -> usize {
		held.sort_by(|&one, &two| {
			let place = |id: u32| {
				usize::try_from(id)
					.ok()
					.map_or(0.0, |id| self.centers[id][axis])
			};

			place(one)
				.total_cmp(&place(two))
				.then(one.cmp(&two))
		});

		held.len() / 2
	}
}

/// A box put on a walk's stack, and how many are on it after.
///
/// Never full: the stack is as deep as the tree can be, @ref [`STACK`].
fn pushed(stack: &mut [u32; STACK], top: usize, child: u32) -> usize {
	let Some(slot) = stack.get_mut(top) else {
		return top;
	};

	*slot = child;

	top + 1
}

/// Which bucket a middle falls into.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "a fraction of the bucket count, floored and held inside it"
)]
fn bucket(position: f32, low: f32, span: f32) -> usize {
	let bins = f32::from(u8::try_from(BINS).unwrap_or(u8::MAX));
	let place = ((position - low) / span * bins).floor();

	(place.max(0.0) as usize).min(BINS - 1)
}

/// The last bucket of the first half of the cheapest cut, or `None` when
/// every cut leaves one side empty.
///
/// A cut costs the area of each side's box times how many triangles it holds,
/// which is how many a ray that enters that box can be expected to have to
/// ask. The first of two equal costs wins.
fn cheapest(bins: &[Bin; BINS]) -> Option<usize> {
	let mut rising = [Bin::EMPTY; BINS];
	let mut running = Bin::EMPTY;

	for (at, bin) in bins.iter().enumerate() {
		running = running.joined(*bin);
		rising[at] = running;
	}

	let mut falling = Bin::EMPTY;
	let mut best: Option<(usize, f32)> = None;

	for after in (0..BINS - 1).rev() {
		falling = falling.joined(bins[after + 1]);

		let left = rising[after];

		if left.count == 0 || falling.count == 0 {
			continue;
		}

		let first = left.area() * count_of(left.count);
		let second = falling.area() * count_of(falling.count);
		let cost = first + second;

		if best.is_none_or(|(_, cheapest)| cost <= cheapest) {
			best = Some((after, cost));
		}
	}

	best.map(|(after, _)| after)
}

/// A count as a float, for pricing: past what a float holds exactly the price
/// does not need to be exact.
#[expect(
	clippy::as_conversions,
	clippy::cast_precision_loss,
	reason = "a count used as a weight, where a rounded weight prices the same"
)]
const fn count_of(count: usize) -> f32 { count as f32 }

/// Half the surface of a box, which is what a ray's chance of entering it goes
/// with.
fn half_surface(low: Vec3, high: Vec3) -> f32 {
	let size = (high - low).max(Vec3::ZERO);

	// x y + y z + z x, as one dot product rather than a sum a compiler may fold
	size.dot(Vec3::new(size.y, size.z, size.x))
}

/// The axis a box is longest along; the first of equals.
fn longest(extent: Vec3) -> usize {
	if extent.x >= extent.y && extent.x >= extent.z {
		0
	} else if extent.y >= extent.z {
		1
	} else {
		2
	}
}

/// Where a ray enters a box, if it does before the reach.
///
/// The slab test: the distances at which the ray crosses each pair of parallel
/// faces, the latest entry and the earliest exit. A ray starting inside enters
/// at nought.
///
/// **An axis at a time, and the one case the products cannot answer is
/// answered by hand.** A ray that runs parallel to a pair of faces and starts
/// exactly on one of them works out nought times infinity there, which is not a
/// number; it is inside that pair everywhere along its length, so that pair
/// says nothing about where it enters. Left to a vector's own least and
/// greatest, the answer would depend on which operand the not-a-number was -
/// and those are allowed to differ between one kind of processor and another,
/// which a bake that is the same bytes everywhere cannot let them do. It was a
/// ray straight down through a triangle's corner that found it.
fn entry(node: &Node, ray: &Ray, reach: f32) -> Option<f32> {
	let mut enter = 0.0_f32;
	let mut leave = reach;

	for axis in 0..3 {
		let near = (node.low[axis] - ray.origin[axis]) * ray.inverse[axis];
		let far = (node.high[axis] - ray.origin[axis]) * ray.inverse[axis];

		if near.is_nan() || far.is_nan() {
			continue;
		}

		enter = enter.max(near.min(far));
		leave = leave.min(near.max(far));
	}

	(enter <= leave).then_some(enter)
}

/// Where a ray crosses one triangle, if it does in front of its start and
/// short of the reach.
///
/// The test every tracer in the field uses: solve for the distance and the two
/// weights at once, and refuse as soon as either weight leaves the triangle.
/// A ray parallel to the triangle meets it nowhere.
///
/// @param shape - the corner and the two edges from it
/// @param ray - the ray
/// @param reach - how far it may go
/// @return the distance, the two weights and whether it met the front
fn crossing(shape: &[Vec3; 3], ray: &Ray, reach: f32) -> Option<(f32, f32, f32, bool)> {
	let [corner, first, second] = *shape;
	let push = ray.direction.cross(second);
	let determinant = first.dot(push);

	// not a number, or so near nought that the ray runs along the triangle
	if determinant.is_nan() || determinant.abs() <= f32::MIN_POSITIVE {
		return None;
	}

	let inverse = determinant.recip();
	let from = ray.origin - corner;
	let along = from.dot(push) * inverse;

	if !(0.0..=1.0).contains(&along) {
		return None;
	}

	let lift = from.cross(first);
	let across = ray.direction.dot(lift) * inverse;

	if !(across >= 0.0 && along + across <= 1.0) {
		return None;
	}

	let distance = second.dot(lift) * inverse;

	if !(distance > 0.0 && distance < reach) {
		return None;
	}

	// the winding's side: a ray coming at a triangle whose corners turn
	// counter-clockwise towards it has a positive determinant
	Some((distance, along, across, determinant > 0.0))
}

#[cfg(test)]
mod tests {
	use colby_core::random::Random;

	use super::*;

	/// A number in `-1..1` from a generator.
	fn signed(random: &mut Random) -> f32 {
		let top = u32::try_from(random.draw() >> 40).expect("twenty-four bits");

		f32::from(u16::try_from(top >> 8).expect("sixteen bits")) / 32768.0 - 1.0
	}

	/// A point in a cube two units on a side around the origin, from a
	/// generator.
	fn point(random: &mut Random) -> Vec3 {
		Vec3::new(signed(random), signed(random), signed(random))
	}

	/// Small triangles scattered through a box.
	fn soup(count: usize, seed: u64) -> Vec<[Vec3; 3]> {
		let mut random = Random::new(seed);

		std::iter::repeat_with(|| {
			let middle = point(&mut random) * 10.0;

			[
				middle + point(&mut random) * 1.5,
				middle + point(&mut random) * 1.5,
				middle + point(&mut random) * 1.5,
			]
		})
		.take(count)
		.collect()
	}

	/// The nearest hit by asking every triangle, which is what the tree has to
	/// agree with.
	fn every(triangles: &[[Vec3; 3]], ray: &Ray, reach: f32) -> Option<Hit> {
		let mut best: Option<Hit> = None;

		for (index, [a, b, c]) in triangles.iter().enumerate() {
			let limit = best.map_or(reach, |hit| hit.distance);

			if let Some((distance, along, across, front)) =
				crossing(&[*a, *b - *a, *c - *a], ray, limit)
			{
				best = Some(Hit {
					distance,
					triangle: u32::try_from(index).expect("a small soup"),
					along,
					across,
					front,
				});
			}
		}

		best
	}

	#[test]
	fn a_ray_through_a_triangle_meets_it_where_it_should_and_says_which_side() {
		let triangle =
			[Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0)];
		let tree = Tree::build(&[triangle]);
		// corners counter-clockwise seen from below, so the front faces down
		let up = tree
			.nearest(&Ray::new(Vec3::new(0.0, -2.0, 0.0), Vec3::Y), f32::INFINITY)
			.expect("straight up through its middle");
		let down = tree
			.nearest(&Ray::new(Vec3::new(0.0, 3.0, 0.0), Vec3::NEG_Y), f32::INFINITY)
			.expect("straight down through it");

		assert!((up.distance - 2.0).abs() < 1.0e-6, "two below it, {}", up.distance);
		assert!((down.distance - 3.0).abs() < 1.0e-6, "three above it, {}", down.distance);
		assert!(up.front, "the corners turn counter-clockwise seen from below");
		assert!(!down.front, "and clockwise seen from above");
		assert!(
			tree.nearest(&Ray::new(Vec3::new(3.0, -2.0, 0.0), Vec3::Y), f32::INFINITY)
				.is_none(),
			"a ray beside it meets nothing"
		);
		assert!(
			tree.nearest(&Ray::new(Vec3::new(0.0, -2.0, 0.0), Vec3::Y), 1.5)
				.is_none(),
			"and one that stops short of it meets nothing either"
		);
	}

	#[test]
	fn the_nearest_hit_is_the_one_asking_every_triangle_finds() {
		let triangles = soup(1500, 7);
		let tree = Tree::build(&triangles);
		let mut random = Random::new(11);
		let mut met = 0;

		for _ in 0..3000 {
			let origin = point(&mut random) * 14.0;
			let direction = point(&mut random).normalize_or(Vec3::X);
			let ray = Ray::new(origin, direction);
			let asked = tree.nearest(&ray, 40.0);
			let known = every(&triangles, &ray, 40.0);

			assert_eq!(asked, known, "from {origin} towards {direction}");
			assert_eq!(
				tree.blocked(&ray, 40.0),
				known.is_some(),
				"and whether anything is in the way agrees with it"
			);
			met += usize::from(known.is_some());
		}

		assert!(met > 800, "enough rays met something to mean anything: {met}");
	}

	#[test]
	fn two_builds_of_one_list_are_one_tree() {
		let triangles = soup(900, 3);
		let one = Tree::build(&triangles);
		let two = Tree::build(&triangles);

		assert_eq!(one.ids, two.ids, "the same order");
		assert_eq!(one.boxes(), two.boxes(), "and the same boxes");
		assert_eq!(one.len(), 900, "holding every triangle once");

		let mut seen = one.ids;

		seen.sort_unstable();
		seen.dedup();
		assert_eq!(seen.len(), 900, "none twice");
	}

	#[test]
	fn nothing_to_trace_meets_nothing() {
		let tree = Tree::build(&[]);
		let ray = Ray::new(Vec3::ZERO, Vec3::X);

		assert!(tree.is_empty(), "an empty list is an empty tree");
		assert!(tree.nearest(&ray, f32::INFINITY).is_none(), "which nothing hits");
		assert!(!tree.blocked(&ray, f32::INFINITY), "and nothing blocks");
	}

	#[test]
	fn a_pile_of_one_triangle_in_one_place_still_builds_and_is_found() {
		let triangle =
			[Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0)];
		let tree = Tree::build(&[triangle; 50]);
		let hit = tree
			.nearest(&Ray::new(Vec3::new(0.0, -1.0, 0.0), Vec3::Y), f32::INFINITY)
			.expect("the pile is in the way");

		assert_eq!(tree.len(), 50, "every copy is held");
		assert!(hit.triangle < 50, "and one of them answers");
	}

	#[test]
	fn a_ray_along_a_box_face_is_neither_lost_nor_invented() {
		// a floor of two triangles, and rays lying exactly in its plane and in
		// the plane of its edge: the slab test divides nought by nought there
		let floor = [
			[Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 1.0), Vec3::new(1.0, 0.0, -1.0)],
			[Vec3::new(-1.0, 0.0, -1.0), Vec3::new(-1.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 1.0)],
		];
		let tree = Tree::build(&floor);
		let along = Ray::new(Vec3::new(-3.0, 0.0, 0.0), Vec3::X);
		let down = Ray::new(Vec3::new(1.0, 2.0, 0.5), Vec3::NEG_Y);

		assert!(
			tree.nearest(&along, f32::INFINITY).is_none(),
			"a ray in the floor's plane meets no face of it"
		);
		assert_eq!(
			tree.nearest(&down, f32::INFINITY)
				.map(|hit| hit.distance),
			Some(2.0),
			"and one down its very edge meets it, two below where it started"
		);
	}

	#[test]
	fn a_ray_through_a_corner_or_down_an_edge_meets_the_triangle() {
		// corners and edges at whole numbers, so the weights a ray through
		// them works out are exactly nought and one
		let triangle =
			[Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 2.0)];
		let tree = Tree::build(&[triangle]);
		let down_through = |x: f32, z: f32| {
			tree.nearest(&Ray::new(Vec3::new(x, 4.0, z), Vec3::NEG_Y), f32::INFINITY)
		};

		for (x, z) in [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
			let hit = down_through(x, z).unwrap_or_else(|| panic!("through ({x}, {z})"));

			assert_eq!(
				hit.distance.to_bits(),
				4.0_f32.to_bits(),
				"four down, through ({x}, {z})"
			);
		}

		assert!(down_through(1.5, 1.5).is_none(), "and past the long edge, nothing");
	}
}
