//! Which pairs of bodies are worth asking about, out of all of them.
//!
//! **This is here because a pool of two hundred crates cost two and a quarter
//! milliseconds a step.** The narrow phase used to be a loop over every pair in
//! the world with four cheap filters and then, for anything that survived them,
//! two hulls built and a full separating-axis test - no bounding box anywhere.
//! Measured on an RX 9060 XT while closing `PERF-3`: forty crates in water 119
//! microseconds, a hundred 578, two hundred **2182**, which is a whole frame at
//! sixty hertz spent deciding that nothing touches anything.
//!
//! Two things were wrong and this fixes both.
//!
//! **A pair that cannot touch paid a hundred nanoseconds to find out.** Every
//! engine tests bounding boxes first; Jolt's own reference implementation of a
//! brute-force broad phase does exactly that and nothing else
//! (`BroadPhaseBruteForce.cpp:292-295`). A box against a box is six
//! comparisons.
//!
//! **And the loop itself was a square.** Even with every pair rejected, twenty
//! thousand iterations of the filters cost about two hundred microseconds - the
//! measured floor of the dry case, where the crates are asleep and almost
//! nothing reaches the narrow phase at all. So the bounding boxes are *sorted*
//! and swept rather than compared all against all: bodies in order of where
//! their box starts, and each one asked only about the ones that start before
//! its own box ends. Sweep and prune, which is the oldest trick there is and
//! the one that fits a flat list of bodies. Box2D and Godot both keep a tree
//! instead, which is better still and is five hundred lines with an
//! incremental rebalance; that is a step rather than a debt, and this is what
//! the crate docs promised when they said "a broadphase and a contact cache
//! later".
//!
//! **The order the pairs come out in is exactly the order the square produced
//! them**, and that is not decoration. A sequential-impulse solver walks its
//! manifolds in order, so a pile settles differently if the order moves - and
//! none of the three oracles would notice, because `--link` never steps a
//! simulation and the Blank fixture has no bodies. A change nothing can see is
//! a change nothing can check, so this one is made invisible on purpose: two
//! counting sorts put the candidates back in index order, and a test asserts
//! the whole list matches what the old square found.

use colby_core::glam::Vec3;

/// A body's axis-aligned bounds.
///
/// A triangle mesh has no bounds of its own without the baked collider, which
/// lives on the simulation; [`Broad::sweep`] is handed those already worked
/// out, so nothing here has to know where they came from.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Bounds {
	/// The low corner.
	low: Vec3,

	/// The high corner.
	high: Vec3,
}

impl Bounds {
	/// Bounds nothing is inside, for a body that has none to give.
	///
	/// Inverted on purpose: a low corner above its high corner overlaps
	/// nothing at all, including itself, so a body with no bounds is simply
	/// never a candidate rather than a special case in the sweep.
	const NOWHERE: Self = Self {
		low: Vec3::INFINITY,
		high: Vec3::NEG_INFINITY,
	};

	/// Whether two of them share any space.
	fn touches(self, other: Self) -> bool {
		self.low.x <= other.high.x
			&& other.low.x <= self.high.x
			&& self.low.y <= other.high.y
			&& other.low.y <= self.high.y
			&& self.low.z <= other.high.z
			&& other.low.z <= self.high.z
	}
}

/// The scratch a sweep needs, kept so that a step allocates nothing.
///
/// Owned by the [`Simulation`](crate::Simulation) and taken out and put back
/// around the call, the same way the manifold lists are and for the same
/// reason: the narrow phase borrows the simulation for its collision meshes
/// while filling a list that lives on it.
#[derive(Debug, Default)]
pub(crate) struct Broad {
	/// One per body, in the order they were handed over.
	boxes: Vec<Bounds>,

	/// Body indices, sorted by where their box starts along the sweep axis.
	order: Vec<u32>,

	/// Somewhere to scatter into while sorting.
	scratch: Vec<(u32, u32)>,

	/// How many candidates fall in each bucket, reused by both passes.
	counts: Vec<u32>,
}

impl Broad {
	/// Which axis the sweep sorts along.
	///
	/// **X, fixed, rather than the widest spread measured per step.** Choosing
	/// per step is one pass over the bounds and it makes the *order* depend on
	/// the world, which is a second thing that could differ between two runs of
	/// something that has to be reproducible. A world laid out along one axis
	/// and swept along another degrades to the square this replaces, which is
	/// the case to remember if a level ever turns out to be a corridor running
	/// north.
	const AXIS: usize = 0;

	/// Fills `into` with every pair whose bounds overlap, in index order.
	///
	/// @param bounds - each body's world bounds, or `None` where it has none
	/// @param into - cleared, then filled with `(first, second)` index pairs,
	/// first below second, ordered exactly as a loop over every pair would
	/// have produced them
	pub(crate) fn sweep(
		&mut self,
		bounds: impl ExactSizeIterator<Item = Option<(Vec3, Vec3)>>,
		into: &mut Vec<(u32, u32)>,
	) {
		into.clear();
		self.boxes.clear();
		self.boxes.extend(bounds.map(|found| {
			found.map_or(Bounds::NOWHERE, |(low, high)| Bounds { low, high })
		}));

		let count = self.boxes.len();

		// taken out and sorted on its own, because the comparison reads the
		// boxes and the list being sorted lives beside them on the same struct
		let mut order = core::mem::take(&mut self.order);
		order.clear();
		order.extend(0..u32::try_from(count).unwrap_or(u32::MAX));
		// by where each box starts, and by index where two start together, so
		// that the walk below is a total order and not a nearly-total one.
		// `sort_unstable_by` over a few hundred integers is nothing beside
		// what it saves.
		order.sort_unstable_by(|&one, &other| {
			let (here, there) = (self.start(one), self.start(other));

			here.total_cmp(&there).then(one.cmp(&other))
		});
		self.order = order;

		for (at, &first) in self.order.iter().enumerate() {
			let one = self.boxes.get(first as usize).copied();
			let Some(one) = one else {
				continue;
			};
			// the whole of the pruning: past the end of this box nothing that
			// starts later can reach back to touch it, so the walk stops
			// rather than running to the end of the list.
			let reach = one.high[Self::AXIS];

			for &second in self.order.iter().skip(at + 1) {
				if self.start(second) > reach {
					break;
				}

				let Some(other) = self.boxes.get(second as usize).copied() else {
					continue;
				};

				if one.touches(other) {
					into.push((first.min(second), first.max(second)));
				}
			}
		}

		self.tidy(into, count);
	}

	/// Where one body's box starts along the sweep axis.
	///
	/// Infinity for a body with no bounds, which sorts it to the end where its
	/// own missing box excludes it anyway.
	fn start(&self, index: u32) -> f32 {
		self.boxes
			.get(index as usize)
			.map_or(f32::INFINITY, |held| held.low[Self::AXIS])
	}

	/// Puts the candidates back in the order a loop over every pair would have
	/// produced them.
	///
	/// Two stable counting sorts, least significant key first: by the second
	/// index and then by the first. Linear in the candidates and in the bodies,
	/// against the twenty to forty microseconds a comparison sort of a few
	/// thousand pairs would cost - which would be most of what the sweep just
	/// saved.
	///
	/// @param into - the candidates, reordered in place
	/// @param count - how many bodies there are, which is how many buckets
	fn tidy(&mut self, into: &mut Vec<(u32, u32)>, count: usize) {
		self.pass(into, count, false);
		self.pass(into, count, true);
	}

	/// One stable counting sort over a key.
	///
	/// @param into - the candidates, reordered in place
	/// @param count - how many buckets
	/// @param leading - whether to sort by the first index or the second
	fn pass(&mut self, into: &mut Vec<(u32, u32)>, count: usize, leading: bool) {
		self.counts.clear();
		self.counts.resize(count + 1, 0);

		for &(first, second) in into.iter() {
			let key = if leading { first } else { second } as usize;

			if let Some(bucket) = self.counts.get_mut(key) {
				*bucket += 1;
			}
		}

		// the prefix sum, turning "how many land here" into "where this
		// bucket's run starts"
		let mut running = 0;
		for bucket in &mut self.counts {
			let held = *bucket;
			*bucket = running;
			running += held;
		}

		self.scratch.clear();
		self.scratch.resize(into.len(), (0, 0));

		// in input order, which is what makes this stable and therefore what
		// lets the second pass keep the first one's work
		for &pair in into.iter() {
			let key = if leading { pair.0 } else { pair.1 } as usize;
			let Some(bucket) = self.counts.get_mut(key) else {
				continue;
			};
			let at = *bucket as usize;
			*bucket += 1;

			if let Some(slot) = self.scratch.get_mut(at) {
				*slot = pair;
			}
		}

		into.clear();
		into.extend_from_slice(&self.scratch);
	}
}

/// Every pair of body indices, the way the square used to produce them.
///
/// The reference the sweep is checked against, and the fallback nothing uses:
/// it exists so that a test can say "these two agree" about a world rather than
/// asserting a list somebody typed out.
///
/// @param count - how many bodies there are
/// @return every `(first, second)` with first below second, in order
#[cfg(test)]
fn every_pair(count: usize) -> Vec<(u32, u32)> {
	let mut pairs = Vec::new();

	for first in 0..count {
		for second in first + 1..count {
			pairs.push((
				u32::try_from(first).unwrap_or(u32::MAX),
				u32::try_from(second).unwrap_or(u32::MAX),
			));
		}
	}

	pairs
}

/// Whether two bodies' bounds overlap at all.
///
/// The reference the sweep is checked against: a test asks this about every
/// pair and compares the answer with what the sweep produced.
///
/// @param one - one body's bounds
/// @param other - the other's
#[cfg(test)]
fn overlapping(one: Option<(Vec3, Vec3)>, other: Option<(Vec3, Vec3)>) -> bool {
	let (Some((low, high)), Some((other_low, other_high))) = (one, other) else {
		return false;
	};

	Bounds { low, high }.touches(Bounds {
		low: other_low,
		high: other_high,
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A unit box standing at a place.
	fn at(x: f32) -> Option<(Vec3, Vec3)> {
		Some((Vec3::new(x - 0.5, -0.5, -0.5), Vec3::new(x + 0.5, 0.5, 0.5)))
	}

	/// Runs a sweep over a list of bounds.
	fn swept(bounds: &[Option<(Vec3, Vec3)>]) -> Vec<(u32, u32)> {
		let mut broad = Broad::default();
		let mut pairs = Vec::new();

		broad.sweep(bounds.iter().copied(), &mut pairs);

		pairs
	}

	/// What a loop over every pair would have kept, with the same test.
	fn squared(bounds: &[Option<(Vec3, Vec3)>]) -> Vec<(u32, u32)> {
		every_pair(bounds.len())
			.into_iter()
			.filter(|&(first, second)| {
				overlapping(bounds[first as usize], bounds[second as usize])
			})
			.collect()
	}

	#[test]
	fn a_row_of_boxes_that_touch_nobody_makes_no_pairs() {
		let bounds = [at(0.0), at(10.0), at(20.0), at(30.0)];

		assert!(swept(&bounds).is_empty());
		assert_eq!(swept(&bounds), squared(&bounds));
	}

	#[test]
	fn boxes_that_overlap_are_found_and_the_ones_between_them_are_not() {
		// three in a row a little apart, so the first and second touch and the
		// first and third cannot: exactly the case the sweep's early stop is
		// for.
		let bounds = [at(0.0), at(0.5), at(4.0)];

		assert_eq!(swept(&bounds), vec![(0, 1)]);
		assert_eq!(swept(&bounds), squared(&bounds));
	}

	#[test]
	fn the_sweep_finds_what_the_square_finds_on_a_grid_of_boxes() {
		// the case that made `PERF-3`: a pile of things that mostly do not
		// touch, with a few that do.
		let mut bounds = Vec::new();

		for at in 0_u8..60 {
			let x = f32::from(at % 10) * 0.75;
			let z = f32::from(at / 10) * 0.75;

			bounds.push(Some((
				Vec3::new(x - 0.5, -0.5, z - 0.5),
				Vec3::new(x + 0.5, 0.5, z + 0.5),
			)));
		}

		assert_eq!(swept(&bounds), squared(&bounds), "the sweep lost or gained a pair");
		assert!(!swept(&bounds).is_empty(), "a fixture that finds nothing proves nothing");
	}

	#[test]
	fn one_enormous_box_is_a_candidate_for_everything_it_covers() {
		// the floor, and the pool: a body whose bounds reach across the world
		// is where a sweep along one axis is worth least, and it still has to
		// be right.
		let mut bounds = vec![Some((Vec3::splat(-100.0), Vec3::splat(100.0)))];

		for step in 1_u8..20 {
			bounds.push(at(f32::from(step) * 3.0));
		}

		let found = swept(&bounds);

		assert_eq!(found, squared(&bounds));
		assert_eq!(found.len(), 19, "the floor pairs with every one of them and they with nobody");
	}

	#[test]
	fn a_body_with_no_bounds_is_in_no_pair_at_all() {
		// a triangle mesh whose collider has not been baked yet, which is what
		// `None` means here. Overlapping nothing beats being tested against
		// everything.
		let bounds = [at(0.0), None, at(0.2)];

		assert_eq!(swept(&bounds), vec![(0, 2)]);
		assert_eq!(swept(&bounds), squared(&bounds));
	}

	#[test]
	fn the_pairs_come_out_in_the_order_the_square_produced_them() {
		// **the property the whole design turns on.** A sequential-impulse
		// solver walks its manifolds in order, so a pile settles differently
		// if this moves - and none of the three oracles would notice, because
		// `--link` never steps a simulation and the Blank fixture has no
		// bodies at all.
		let bounds = [
			Some((Vec3::splat(-50.0), Vec3::splat(50.0))),
			at(0.0),
			at(0.4),
			at(0.8),
			at(1.2),
		];
		let found = swept(&bounds);
		let mut sorted = found.clone();
		sorted.sort_unstable();

		assert_eq!(found, sorted, "the candidates came out unsorted");
		assert_eq!(found, squared(&bounds));
	}

	#[test]
	fn boxes_that_start_together_still_produce_every_pair() {
		// a tie on the sweep axis, which is what a stack of crates dropped
		// down one line is. The sort breaks it by index so the walk has a
		// total order, and nothing may be lost to the early stop.
		let bounds = [at(0.0), at(0.0), at(0.0), at(0.0)];

		assert_eq!(swept(&bounds), squared(&bounds));
		assert_eq!(swept(&bounds).len(), 6, "four boxes in one place are six pairs");
	}

	#[test]
	fn nothing_at_all_sweeps_to_nothing() {
		assert!(swept(&[]).is_empty());
		assert!(swept(&[at(0.0)]).is_empty(), "one body is no pair");
	}
}
