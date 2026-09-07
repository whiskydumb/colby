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
//! the one that fits a flat list of bodies. Jolt, Box3D and Godot all keep a
//! tree instead, which is better still and is between eight hundred and two
//! thousand lines with an incremental rebalance; that is a step rather than a
//! debt.
//!
//! **The order the pairs come out in is exactly the order the square produced
//! them**, and that is not decoration. A sequential-impulse solver walks its
//! manifolds in order, so a pile settles differently if the order moves - and
//! none of the three oracles would notice, because `--link` never steps a
//! simulation and the Blank fixture has no bodies. A change nothing can see is
//! a change nothing can check, so this one is made invisible on purpose: two
//! counting sorts put the candidates back in index order, and a test asserts
//! the whole list matches what the old square found.
//!
//! **And because of that, which axis is swept is free to be chosen from the
//! world.** @ref [`Broad::widest`]. The pairs a sweep finds are the same set
//! whatever it sorts along - a pair that touches overlaps on every axis, so it
//! can never fall past the early stop - and `tidy` below puts them back in the
//! same order afterwards. So the axis buys time and costs nothing that anything
//! outside this file can see, which is the opposite of what the debt that
//! opened this expected.

use std::time::{Duration, Instant};

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

	/// The middle of it, or nothing at all where there is no box.
	///
	/// [`NOWHERE`](Self::NOWHERE) has no middle - its corners average to a NaN
	/// on every axis - so a body with no bounds is left out of the spread the
	/// axis is chosen from rather than poisoning it.
	fn center(self) -> Option<Vec3> {
		(self.low.x <= self.high.x).then(|| (self.low + self.high) * 0.5)
	}

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
	///
	/// A position in the list the caller handed over rather than a
	/// [`BodyId`](colby_core::abi::BodyId): the sweep never looks at a body and
	/// has no reason to know what one is.
	order: Vec<usize>,

	/// Somewhere to scatter into while sorting.
	scratch: Vec<(usize, usize)>,

	/// How many candidates fall in each bucket, reused by both passes.
	counts: Vec<usize>,

	/// Which axis the last sweep sorted along. @ref [`Broad::widest`].
	axis: usize,

	/// How long the last sweep took, for [`Spent`](crate::Spent).
	///
	/// Kept here rather than clocked at the call site because the bounds are
	/// worked out lazily *inside* the sweep - the caller hands over an iterator
	/// - so a clock outside it would be timing the caller's own loop as well.
	spent: Duration,
}

impl Broad {
	/// How long the last sweep took, bounds and sorts and all.
	pub(crate) const fn spent(&self) -> Duration { self.spent }

	/// Which axis to sweep along: the one the bodies are most spread out on.
	///
	/// **Measured, and it is the whole of what the axis is worth.** A world
	/// laid out along one axis and swept along another degrades to a square of
	/// box tests, because no body's box ends before the next one's begins and
	/// the early stop in [`against`](Self::against) never fires. On this
	/// machine, at the thousand-and-twenty-four bodies a world can hold, in
	/// microseconds of the whole narrow phase - a rectangle of crates spaced
	/// three apart, none of them touching, so all of it is this:
	///
	/// | world, all of it along z | swept along x | swept along z |
	/// |---|---|---|
	/// | 32 x 32 square | 111 | 124 |
	/// | 8 x 128 hall | 172 | 67 |
	/// | 4 x 256 hall | 243 | 67 |
	/// | 1 x 1024 in single file | **624** | **65** |
	///
	/// The sweep's own span, measured apart from the tests it feeds, is 550
	/// against 27 on the last of those. **Twenty times**, and a third of a
	/// sixty-hertz frame spent finding out that nothing touches anything. The
	/// square is a tie and sweeps along x either way, so its two columns are
	/// the same work twice and are what run-to-run noise looks like.
	///
	/// **The spread of the centers, not of the low corners**, which is what
	/// Jolt does at every node of its tree
	/// (`QuadTree.cpp:440`, `GetHighestComponentIndex` of the center bounds),
	/// what Box3D does before its median split (`dynamic_tree.c:1501`), and
	/// what Godot tries first before falling back to comparing all three
	/// (`bvh_split.inc:33`, `:70-98`). Centers rather than corners because one
	/// enormous body - the floor, a pool - has a box that reaches across the
	/// world on every axis and a center that is simply in the middle of it, so
	/// it moves a corner spread and barely moves a center one.
	///
	/// A tie keeps the lower axis, so a square world sweeps along x exactly as
	/// it did before this existed.
	///
	/// **It costs about three microseconds** at the thousand-and-twenty-four
	/// bodies a world can hold, measured back to back in one build against a
	/// run that skipped the pass and took the x axis regardless: 102, 103, 101
	/// against 100, 102, 96. Three microseconds against five hundred is what
	/// makes this worth a pass over the boxes rather than an argument.
	///
	/// @return which axis of the boxes to sort along
	fn widest(&self) -> usize {
		let mut low = Vec3::INFINITY;
		let mut high = Vec3::NEG_INFINITY;

		for center in self.boxes.iter().filter_map(|held| held.center()) {
			low = low.min(center);
			high = high.max(center);
		}

		let spread = high - low;
		let mut axis = 0;

		for next in 1..3 {
			if spread[next] > spread[axis] {
				axis = next;
			}
		}

		axis
	}

	/// Fills `into` with every pair whose bounds overlap, in index order.
	///
	/// @param bounds - each body's world bounds, or `None` where it has none
	/// @param into - cleared, then filled with `(first, second)` index pairs,
	/// first below second, ordered exactly as a loop over every pair would
	/// have produced them
	pub(crate) fn sweep(
		&mut self,
		bounds: impl ExactSizeIterator<Item = Option<(Vec3, Vec3)>>,
		into: &mut Vec<(usize, usize)>,
	) {
		let began = Instant::now();

		into.clear();
		self.boxes.clear();
		self.boxes.extend(
			bounds.map(|found| found.map_or(Bounds::NOWHERE, |(low, high)| Bounds { low, high })),
		);

		let count = self.boxes.len();

		self.axis = self.widest();

		// taken out and sorted on its own, because the comparison reads the
		// boxes and the list being sorted lives beside them on the same struct
		let mut order = core::mem::take(&mut self.order);
		order.clear();
		order.extend(0..count);
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
			let Some(one) = self.boxes.get(first).copied() else {
				continue;
			};

			self.against(at, first, one, into);
		}

		self.tidy(into, count);
		self.spent = began.elapsed();
	}

	/// Every candidate for one body, out of the ones that start after it.
	///
	/// A method of its own rather than the inner half of a loop, which is what
	/// a walk that stops early wants anyway: the `return` here is the pruning.
	///
	/// @param at - where this body sits in the sorted order
	/// @param first - which body it is
	/// @param one - its bounds
	/// @param into - where a candidate pair is appended
	fn against(&self, at: usize, first: usize, one: Bounds, into: &mut Vec<(usize, usize)>) {
		// past the end of this box nothing that starts later can reach back to
		// touch it, so the walk stops rather than running to the end of the
		// list. That is the whole of what a sweep buys over a square.
		let reach = one.high[self.axis];

		for &second in self.order.iter().skip(at + 1) {
			if self.start(second) > reach {
				return;
			}

			if self
				.boxes
				.get(second)
				.is_some_and(|other| one.touches(*other))
			{
				into.push((first.min(second), first.max(second)));
			}
		}
	}

	/// Where one body's box starts along the sweep axis.
	///
	/// Infinity for a body with no bounds, which sorts it to the end where its
	/// own missing box excludes it anyway.
	fn start(&self, index: usize) -> f32 {
		self.boxes
			.get(index)
			.map_or(f32::INFINITY, |held| held.low[self.axis])
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
	fn tidy(&mut self, into: &mut Vec<(usize, usize)>, count: usize) {
		self.pass(into, count, false);
		self.pass(into, count, true);
	}

	/// One stable counting sort over a key.
	///
	/// @param into - the candidates, reordered in place
	/// @param count - how many buckets
	/// @param leading - whether to sort by the first index or the second
	fn pass(&mut self, into: &mut Vec<(usize, usize)>, count: usize, leading: bool) {
		self.counts.clear();
		self.counts.resize(count + 1, 0);

		for &(first, second) in into.iter() {
			let key = if leading { first } else { second };

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
			let key = if leading { pair.0 } else { pair.1 };
			let Some(bucket) = self.counts.get_mut(key) else {
				continue;
			};
			let at = *bucket;
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
fn every_pair(count: usize) -> Vec<(usize, usize)> {
	let mut pairs = Vec::new();

	for first in 0..count {
		for second in first + 1..count {
			pairs.push((first, second));
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

	Bounds { low, high }.touches(Bounds { low: other_low, high: other_high })
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A unit box standing at a place.
	fn at(x: f32) -> Option<(Vec3, Vec3)> {
		Some((Vec3::new(x - 0.5, -0.5, -0.5), Vec3::new(x + 0.5, 0.5, 0.5)))
	}

	/// Runs a sweep over a list of bounds.
	fn swept(bounds: &[Option<(Vec3, Vec3)>]) -> Vec<(usize, usize)> {
		let mut broad = Broad::default();
		let mut pairs = Vec::new();

		broad.sweep(bounds.iter().copied(), &mut pairs);

		pairs
	}

	/// What a loop over every pair would have kept, with the same test.
	fn squared(bounds: &[Option<(Vec3, Vec3)>]) -> Vec<(usize, usize)> {
		every_pair(bounds.len())
			.into_iter()
			.filter(|&(first, second)| overlapping(bounds[first], bounds[second]))
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
		assert_eq!(
			found.len(),
			19,
			"the floor pairs with every one of them and they with nobody"
		);
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

	/// A unit box standing anywhere.
	fn anywhere(position: Vec3) -> Option<(Vec3, Vec3)> {
		Some((position - Vec3::splat(0.5), position + Vec3::splat(0.5)))
	}

	/// A row of boxes along one axis, a little apart.
	fn line(along: Vec3, count: u8) -> Vec<Option<(Vec3, Vec3)>> {
		(0..count)
			.map(|step| anywhere(along * f32::from(step) * 3.0))
			.collect()
	}

	/// Which axis a sweep over these bounds would choose.
	fn chosen(bounds: &[Option<(Vec3, Vec3)>]) -> usize {
		let mut broad = Broad::default();
		let mut pairs = Vec::new();

		broad.sweep(bounds.iter().copied(), &mut pairs);

		broad.axis
	}

	#[test]
	fn the_axis_is_the_one_the_bodies_are_most_spread_out_on() {
		// the case `PERF-6` was opened for: a corridor running north, swept
		// along east, which is a square of box tests and was measured at
		// twenty-one times the cost of sweeping along the corridor.
		assert_eq!(chosen(&line(Vec3::X, 20)), 0, "a row along x");
		assert_eq!(chosen(&line(Vec3::Y, 20)), 1, "a column along y");
		assert_eq!(chosen(&line(Vec3::Z, 20)), 2, "a corridor along z");
	}

	#[test]
	fn a_world_with_no_longest_axis_sweeps_along_x_as_it_always_did() {
		// the tie, and it is kept deliberately: a square world has to behave
		// exactly as it did before an axis was ever chosen, or every recorded
		// number about one stops meaning anything.
		let cube = [
			anywhere(Vec3::ZERO),
			anywhere(Vec3::splat(10.0)),
			anywhere(Vec3::new(10.0, 0.0, 0.0)),
			anywhere(Vec3::new(0.0, 10.0, 0.0)),
		];

		assert_eq!(chosen(&cube), 0);
		assert_eq!(chosen(&[]), 0, "and a world with nothing in it");
		assert_eq!(chosen(&[None, None]), 0, "and one whose bodies have no bounds");
	}

	#[test]
	fn one_enormous_body_does_not_decide_the_axis_by_itself() {
		// the floor, and the pool. Its box reaches across the world on every
		// axis, so a spread of *corners* would be a tie it caused and the
		// corridor beside it would be swept the wrong way. Its center is simply
		// in the middle. @ref [`Broad::widest`].
		let mut bounds = vec![Some((Vec3::new(-60.0, -1.0, -60.0), Vec3::new(60.0, 0.0, 60.0)))];

		bounds.extend(line(Vec3::Z, 20));

		assert_eq!(chosen(&bounds), 2, "the crates decide, not the floor under them");
	}

	#[test]
	fn every_axis_finds_the_same_pairs_the_square_does() {
		// **the property the axis being chosen at all stands on.** A pair that
		// touches overlaps on every axis, so it can never fall past the early
		// stop whichever one is swept - and `tidy` puts what is found back in
		// index order regardless. Three corridors, one per axis, each with
		// overlaps in it, each checked against the loop over every pair.
		for along in [Vec3::X, Vec3::Y, Vec3::Z] {
			let mut bounds = Vec::new();

			for step in 0..12_u8 {
				// each pair of neighbors a third of a box apart, so half of
				// them overlap and half do not
				bounds.push(anywhere(along * f32::from(step) * 0.66));
			}

			let found = swept(&bounds);

			assert_eq!(found, squared(&bounds), "swept along {along}");
			assert!(!found.is_empty(), "a fixture that finds nothing proves nothing");
		}
	}

	#[test]
	fn the_sweep_says_how_long_it_took() {
		// the `cpu broad` row of `--profile`, which is what makes an axis that
		// went wrong visible without a debugger.
		let mut broad = Broad::default();
		let mut pairs = Vec::new();

		assert_eq!(broad.spent(), Duration::ZERO, "before it has ever run");

		broad.sweep(line(Vec3::Z, 40).iter().copied(), &mut pairs);

		assert!(broad.spent() > Duration::ZERO, "and after");
	}
}
