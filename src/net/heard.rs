//! What one peer has said the world looks like, and when it said it, so that
//! the world can be drawn a moment behind rather than in jumps.
//!
//! ```text
//!   heard.take(number, &world, now);
//!   let world = heard.at(now - DELAY);
//! ```
//!
//! **A snapshot arrives twenty times a second and a frame is drawn sixty or
//! more.** Drawing the newest snapshot as it stands means every moving thing
//! moves three times a frame and then stands still for two, which reads as a
//! stutter rather than as motion. The answer every system in the field reaches
//! for is the same one: keep two snapshots and draw the world *between* them,
//! at a time far enough behind that the later of the two has already arrived.
//!
//! **Behind is what makes it smooth, and behind is what it costs.** A viewer
//! sees the world as it was a delay ago, so a thing it is looking at is not
//! quite where the host says it is. That is a bill nothing here can avoid and
//! everything in the field pays: a client cannot draw a moment it has not been
//! told about without inventing it, and inventing it is a guess that has to be
//! taken back. The one thing this does *not* do is extrapolate past the newest
//! snapshot - asked for a moment it has not reached, it answers with the newest
//! world it has and lets the picture sit still for a frame, which is a smaller
//! lie than a body carrying on through a wall it was about to stop at.
//!
//! ## What the delay has to be
//!
//! Longer than the gap between snapshots, or the later of the two will not
//! have arrived and there is nothing to blend towards; longer again by however
//! much the wire varies, or a late one leaves the same hole. The number is not
//! chosen here - it is the caller's, because it is a trade between smoothness
//! and how far behind a player is willing to be, and that is a matter of taste
//! rather than of arithmetic.
//!
//! ## Time is when it arrived, not when it was sent
//!
//! A snapshot carries a number and no clock reading, so the only time this end
//! has for one is the moment it turned up. That is deliberate rather than an
//! omission: two machines' clocks do not agree, and a snapshot stamped with the
//! sender's would need them to. What arrival time costs is that the *spacing*
//! between two snapshots here is the spacing the wire delivered them at rather
//! than the spacing they were sent at, so a burst of delay stretches the world
//! and a burst of catching-up squeezes it.
//!
//! **Drawing behind does not absorb that**, and it is worth being exact,
//! because it is the obvious thing to assume and it is false: a moment is
//! placed between two arrival times, so the rate the world sweeps at is fixed
//! by those two times and moving the moment back by a constant cannot change
//! it. Three snapshots a wire delivers at nought, ninety and a hundred
//! milliseconds are drawn at that spacing whatever the delay is. What the delay
//! *does* buy is that there is a later snapshot to be between at all - it turns
//! a hole, where the world stands still with nothing to move towards, into a
//! stretch, where it moves at the wrong speed. That is a smaller fault, and it
//! is the whole of what the delay is for.

use core::time::Duration;

use crate::{
	ring::{DEPTH, Ring},
	snapshot::{NOTHING, Slot},
};

/// What one peer has said about the world, and when.
#[derive(Clone, Debug)]
pub struct Heard {
	/// The worlds themselves, by number.
	ring: Ring,

	/// When each of them arrived, in the same places the ring uses.
	///
	/// A place is only meaningful while the ring still holds the number that
	/// put it there, which is why nothing here is read without asking the ring
	/// about the number first.
	arrived: Vec<Duration>,

	/// The newest snapshot taken, or [`NOTHING`].
	holding: u32,

	/// Where a world is put together, and the two it is put together from.
	between: Vec<Slot>,
	earlier: Vec<Slot>,
	later: Vec<Slot>,
}

impl Heard {
	/// A peer that has said nothing yet.
	///
	/// @return a buffer holding no world at all
	#[must_use]
	pub fn new() -> Self {
		Self {
			ring: Ring::new(),
			arrived: vec![Duration::ZERO; DEPTH],
			holding: NOTHING,
			between: Vec::new(),
			earlier: Vec::new(),
			later: Vec::new(),
		}
	}

	/// The newest snapshot taken, which is what this end says it holds.
	///
	/// @return the number, or [`NOTHING`] if nothing has been taken
	#[must_use]
	pub const fn holding(&self) -> u32 { self.holding }

	/// Whether nothing has been taken yet.
	///
	/// @return whether there is no world here at all
	#[must_use]
	pub const fn is_empty(&self) -> bool { self.holding == NOTHING }

	/// The world one snapshot described, to read the next difference against.
	///
	/// @param against - which snapshot the next block names
	/// @return the world as it was, and the number it is - or an empty world
	/// and [`NOTHING`]
	pub fn base(&mut self, against: u32) -> (&[Slot], u32) { self.ring.base(against) }

	/// Remembers a world that arrived, and when.
	///
	/// @param number - which snapshot it is
	/// @param world - what it described, by slot
	/// @param now - how long this end has been running
	/// @return whether it was new, and so whether it was taken
	pub fn take(&mut self, number: u32, world: &[Slot], now: Duration) -> bool {
		if !self.ring.keep(number, world) {
			return false;
		}

		// after the ring, never before: the place is only somebody's while the
		// ring agrees it is, and writing the time first would leave a stamp on
		// a snapshot that was refused.
		self.arrived[Self::place(number)] = now;
		self.holding = number;
		true
	}

	/// Forgets everything, for a peer that has gone.
	///
	/// @note: the numbering is not restarted, because [`Ring::forget`] does not
	/// restart it - a number the last conversation used must not name anything
	/// in this one. The consequence is that a peer which comes back *counting
	/// from one again* is refused every snapshot it sends, and nothing here can
	/// tell that from a peer sending stale numbers. Which conversation a
	/// datagram belongs to is the connection handshake's question and it is not
	/// written yet; until it is, a reconnect does not converge either way.
	pub fn forget(&mut self) {
		self.ring.forget();
		self.holding = NOTHING;
		self.between.clear();
	}

	/// The world as it was at one moment.
	///
	/// Asked for a moment between two snapshots it blends them; asked for one
	/// past the newest it answers with the newest, and for one older than
	/// anything still here, with the oldest that is.
	///
	/// @param when - the moment to draw, on this end's own clock
	/// @return the world then, by slot
	pub fn at(&mut self, when: Duration) -> &[Slot] {
		let Some((earlier, later)) = self.straddling(when) else {
			self.between.clear();

			return &self.between;
		};

		if earlier == later {
			self.ring.world(later, &mut self.between);

			return &self.between;
		}

		let (from, to) = (self.arrived[Self::place(earlier)], self.arrived[Self::place(later)]);
		let span = to.saturating_sub(from).as_secs_f32();
		let t = if span > 0.0 {
			(when.saturating_sub(from).as_secs_f32() / span).clamp(0.0, 1.0)
		} else {
			// two snapshots that arrived in the same instant have no interval
			// to be anywhere inside of, so the later one is the answer.
			1.0
		};

		self.ring.world(earlier, &mut self.earlier);
		self.ring.world(later, &mut self.later);
		Self::blend(&self.earlier, &self.later, t, &mut self.between);

		&self.between
	}

	/// The two snapshots one moment lies between.
	///
	/// @param when - the moment
	/// @return the pair, the same number twice when there is nothing to blend
	/// towards, or nothing at all when this end holds no world
	fn straddling(&self, when: Duration) -> Option<(u32, u32)> {
		if self.holding == NOTHING {
			return None;
		}

		let mut later = self.holding;

		// backwards from the newest, over the [`DEPTH`] numbers below it, until
		// one that had already arrived.
		//
		// @note: `DEPTH` *numbers*, which is not quite every snapshot the ring
		// can be holding. A place is only reused when a number lands on it
		// again, so after a long run of losses the oldest number still here can
		// be further below the newest than the ring is deep. One that far back
		// is a second and a half old at the cadence, which is past any delay
		// anybody would set, so it is not blended towards - and saying that is
		// better than a walk whose length is a number nobody can state.
		//
		// **The numbers have gaps in them and that is the ordinary case.** A
		// snapshot is not resent when it is lost - the next one replaces it -
		// so on a wire that drops one in ten, one number in ten is simply
		// never taken. A walk that stepped back by one and stopped at the
		// first number missing would give up at the first loss and answer with
		// the newest world alone, which is the whole of what this module
		// exists to avoid: it would draw in jumps precisely on the wires that
		// need it not to.
		for back in 1..=DEPTH {
			let Some(step) = u32::try_from(back).ok() else {
				break;
			};
			let Some(earlier) = self.holding.checked_sub(step) else {
				break;
			};

			if !self.ring.has(earlier) {
				continue;
			}

			if self.arrived[Self::place(earlier)] <= when {
				return Some((earlier, later));
			}

			later = earlier;
		}

		// nothing still here had arrived by then, so the oldest this end holds
		// is the whole answer.
		Some((later, later))
	}

	/// One world blended into another, slot by slot.
	///
	/// **Which slots are occupied is the earlier snapshot's answer, not the
	/// later one's**, and that is the whole of what this decides. A thing only
	/// the later one holds is not drawn at all until the pair moves on.
	///
	/// The other way round is the tempting one and it is wrong twice over. A
	/// body that appears in the later snapshot has nothing to blend from, so it
	/// would be drawn at its *end of interval* position for the whole interval:
	/// a place it is not, standing still, and then a jump. And it would be
	/// drawn from the moment the *earlier* snapshot describes, which is a whole
	/// interval before the host said it was there. Following the earlier one
	/// costs a body appearing one interval late; following the later one costs
	/// it appearing early, in the wrong place, and then teleporting. Late and
	/// right beats early and wrong, and it is what the system this wire format
	/// comes from does.
	///
	/// Removal is the same rule seen from the other side: a thing the later
	/// snapshot has dropped stays where the earlier one put it until the pair
	/// moves past, rather than vanishing before it was taken away.
	///
	/// A slot whose occupant *changed* is neither blended nor swapped early -
	/// two different bodies have nothing between them, so the earlier one
	/// stands until the pair moves on. @ref `Solid::between`, which says the
	/// same thing about a field.
	fn blend(earlier: &[Slot], later: &[Slot], t: f32, into: &mut Vec<Slot>) {
		// as far as the earlier world reaches, because that is the one that
		// says what is here. Resized and then written all the way through, so
		// there is no emptying first: the loop below assigns every slot rather
		// than only the occupied ones, which is what `Ring::fill` cannot do and
		// why that one clears and this one does not.
		into.resize(earlier.len(), None);

		for (slot, held) in into.iter_mut().enumerate() {
			let before = earlier.get(slot).copied().flatten();
			let after = later.get(slot).copied().flatten();

			*held = match (before, after) {
				// the same occupant at both ends, which is the only case there
				// is anything to blend in.
				| (Some((was, from)), Some((generation, to))) if was == generation =>
					Some((generation, from.between(&to, t))),
				| (Some(occupant), _) => Some(occupant),
				| (None, _) => None,
			};
		}
	}

	/// Where one number's arrival time lives.
	const fn place(number: u32) -> usize {
		#[expect(
			clippy::as_conversions,
			reason = "a const fn, where try_from is not available"
		)]
		{
			(number as usize) % DEPTH
		}
	}
}

impl Default for Heard {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::snapshot::Solid;

	/// A tenth of a second, which is roughly two snapshots at the cadence.
	const TENTH: Duration = Duration::from_millis(100);

	/// A body at a distance along one axis.
	fn along(distance: f32) -> Solid {
		Solid {
			position: [distance, 0.0, 0.0],
			rotation: [0.0, 0.0, 0.0, 1.0],
			scale: [1.0, 1.0, 1.0],
			kind: 2,
			entity: [1, 1],
			..Solid::default()
		}
	}

	/// A world of one body at a distance.
	fn world(distance: f32) -> Vec<Slot> { vec![Some((1, along(distance)))] }

	/// A world whose body is at the *square* of a distance.
	///
	/// Straight-line motion is the fixture that cannot tell one pair of
	/// snapshots from another: if a body's position is a linear function of
	/// the time a snapshot arrived, then blending over the pair that straddles
	/// a moment and blending over the widest pair in the ring give the same
	/// answer to the bit, and a test that means to pin *which* pair was chosen
	/// pins nothing. A body that is speeding up tells them apart.
	fn curving(step: u32) -> Vec<Slot> {
		let along = f32::from(u16::try_from(step * step).unwrap_or(0));

		world(along)
	}

	/// Where the body in one slot of a world is.
	fn at(world: &[Slot], slot: usize) -> Option<f32> {
		world
			.get(slot)
			.copied()
			.flatten()
			.map(|(_, solid)| solid.position[0])
	}

	/// Where the one body in a world is.
	fn where_it_is(world: &[Slot]) -> Option<f32> {
		world
			.first()
			.copied()
			.flatten()
			.map(|(_, solid)| solid.position[0])
	}

	#[test]
	fn a_peer_that_has_said_nothing_has_no_world_at_any_moment() {
		let mut heard = Heard::new();

		assert!(heard.is_empty());
		assert_eq!(heard.holding(), NOTHING);
		assert!(heard.at(Duration::ZERO).is_empty());
		assert!(heard.at(TENTH).is_empty(), "including a moment it has not reached");
	}

	#[test]
	fn a_moment_between_two_snapshots_is_between_the_two_worlds() {
		let mut heard = Heard::new();

		assert!(heard.take(1, &world(0.0), Duration::ZERO));
		assert!(heard.take(2, &world(10.0), TENTH));

		assert_eq!(where_it_is(heard.at(Duration::ZERO)), Some(0.0), "at the first, the first");
		assert_eq!(where_it_is(heard.at(TENTH)), Some(10.0), "at the second, the second");
		assert_eq!(
			where_it_is(heard.at(Duration::from_millis(50))),
			Some(5.0),
			"and half way between them, half way between the worlds"
		);
		assert_eq!(where_it_is(heard.at(Duration::from_millis(25))), Some(2.5));
	}

	#[test]
	fn a_moment_the_far_end_has_not_reached_is_the_newest_world_and_not_a_guess() {
		// the one thing this refuses to do. Carrying the body on past where it
		// was last seen is a guess, and a guess about a body is a body that
		// walks through a wall and is then taken back.
		let mut heard = Heard::new();

		heard.take(1, &world(0.0), Duration::ZERO);
		heard.take(2, &world(10.0), TENTH);

		assert_eq!(
			where_it_is(heard.at(Duration::from_secs(9))),
			Some(10.0),
			"a moment far past the newest is the newest, standing still"
		);
	}

	#[test]
	fn a_moment_older_than_anything_still_here_is_the_oldest_there_is() {
		let mut heard = Heard::new();

		heard.take(1, &world(4.0), TENTH);
		heard.take(2, &world(8.0), TENTH * 2);

		assert_eq!(
			where_it_is(heard.at(Duration::ZERO)),
			Some(4.0),
			"before the first arrived, the first world"
		);
	}

	#[test]
	fn the_pair_a_moment_falls_between_is_the_pair_that_moment_is_inside() {
		// with several to choose from, the blend has to be over the *right*
		// two - the newest pair that has the moment inside it - or a world
		// half a second old would be drawn as though it were current.
		let mut heard = Heard::new();

		// nought, one, four, nine, sixteen, a tenth of a second apart. The
		// squares are the point: over the tight pair a moment half way between
		// the third and fourth is 6.5, and over the widest pair in the ring it
		// is 10.0, so this can say which was used. With a body moving in a
		// straight line both answers are the same number.
		for step in 0..5 {
			heard.take(step + 1, &curving(step), TENTH * step);
		}

		assert_eq!(heard.holding(), 5);
		assert_eq!(
			where_it_is(heard.at(Duration::from_millis(250))),
			Some(6.5),
			"between the third and the fourth, and not between the first and the last"
		);
		assert_eq!(where_it_is(heard.at(Duration::from_millis(50))), Some(0.5));
		assert_eq!(where_it_is(heard.at(Duration::from_millis(350))), Some(12.5));
	}

	#[test]
	fn a_gap_in_the_numbers_is_walked_over_rather_than_stopped_at() {
		// the ordinary case, not the exotic one: a snapshot that is lost is
		// never resent, so on a wire that drops one in ten, one number in ten
		// is never taken at all. A walk that stepped back by one and gave up
		// at the first number missing would answer with the newest world
		// alone - which is drawing in jumps, on exactly the wires that need it
		// not to.
		let mut heard = Heard::new();

		heard.take(1, &curving(0), Duration::ZERO);
		heard.take(2, &curving(1), TENTH);
		// three and four never arrive. Five turns up when it would have.
		heard.take(5, &curving(4), TENTH * 4);

		assert_eq!(heard.holding(), 5);
		assert_eq!(
			where_it_is(heard.at(Duration::from_millis(250))),
			Some(8.5),
			"half way between the two that did arrive, not the newest one alone"
		);
		assert_eq!(
			where_it_is(heard.at(Duration::from_millis(50))),
			Some(0.5),
			"and an older moment still finds the pair around it"
		);
	}

	#[test]
	fn a_run_of_losses_wider_than_the_ring_leaves_one_world_and_says_so() {
		// past `DEPTH` the walk has nowhere left to look, and the answer is
		// the newest world standing still rather than a blend with something
		// that is no longer here.
		let mut heard = Heard::new();

		heard.take(1, &world(0.0), Duration::ZERO);

		// two past the depth rather than twice it, so that the older snapshot
		// is *still in the ring* - `1 + DEPTH * 2` would land on nought's own
		// place and evict it, and the test would pass for that reason instead
		// of for the walk's.
		let far = 2 + u32::try_from(DEPTH).unwrap_or(0);

		heard.take(far, &world(90.0), TENTH * 9);

		assert_eq!(heard.holding(), far);
		assert_eq!(
			where_it_is(heard.at(Duration::from_millis(450))),
			Some(90.0),
			"the older one is further back than the walk looks, so there is nothing to come from"
		);
	}

	#[test]
	fn a_snapshot_that_is_not_new_is_not_taken_and_leaves_the_times_alone() {
		let mut heard = Heard::new();

		heard.take(1, &world(0.0), Duration::ZERO);
		heard.take(2, &world(10.0), TENTH);

		assert!(!heard.take(1, &world(99.0), TENTH * 9), "an old number is not news");
		assert_eq!(heard.holding(), 2);
		assert_eq!(
			where_it_is(heard.at(Duration::from_millis(50))),
			Some(5.0),
			"and the moment between the two is where it always was"
		);
	}

	#[test]
	fn what_is_there_is_the_earlier_snapshots_answer_and_not_the_later_ones() {
		// a body only the later snapshot holds is not drawn yet, because there
		// is nowhere to draw it: blending needs both ends and it has one, so
		// the alternative is its end-of-interval position held still for the
		// whole interval and then a jump. One the later snapshot has dropped
		// stays where it was until the pair moves past it.
		let mut heard = Heard::new();

		heard.take(1, &[Some((1, along(0.0))), None, Some((1, along(9.0)))], Duration::ZERO);
		heard.take(
			2,
			&[Some((1, along(10.0))), Some((1, along(50.0))), None, Some((1, along(70.0)))],
			TENTH,
		);

		let half = heard.at(Duration::from_millis(50));

		assert_eq!(half.len(), 3, "as far as the earlier world reaches and no further");
		assert_eq!(where_it_is(half), Some(5.0), "the one both hold is blended");
		assert_eq!(at(half, 1), None, "the one that appeared is not drawn a moment early");
		assert_eq!(at(half, 2), Some(9.0), "and the one that went is not taken away early");

		// and once the pair has moved past, each is what the newer world says.
		heard.take(3, &[Some((1, along(20.0))), Some((1, along(60.0))), None], TENTH * 2);

		let later = heard.at(Duration::from_millis(100));

		assert_eq!(at(later, 1), Some(50.0), "now it is here, at where it was said to be");
		assert_eq!(at(later, 2), None, "and now it is gone");
	}

	#[test]
	fn a_slot_that_changed_hands_is_one_body_or_the_other_and_never_between_them() {
		// the generation is what says these are two bodies rather than one that
		// moved, and there is nothing between two bodies. The earlier one
		// stands until the pair moves past it, like any other occupancy.
		let mut heard = Heard::new();

		heard.take(1, &[Some((1, along(0.0)))], Duration::ZERO);
		heard.take(2, &[Some((2, along(10.0)))], TENTH);

		let half = heard.at(Duration::from_millis(50));

		assert_eq!(where_it_is(half), Some(0.0), "not five, which is between two bodies");
		assert_eq!(half.first().copied().flatten().map(|(g, _)| g), Some(1));

		heard.take(3, &[Some((2, along(20.0)))], TENTH * 2);

		let later = heard.at(Duration::from_millis(100));

		assert_eq!(where_it_is(later), Some(10.0), "and then it is the new one");
		assert_eq!(later.first().copied().flatten().map(|(g, _)| g), Some(2));
	}

	#[test]
	fn a_peer_that_went_away_leaves_no_world_behind() {
		let mut heard = Heard::new();

		heard.take(1, &world(0.0), Duration::ZERO);
		heard.take(2, &world(10.0), TENTH);
		heard.forget();

		assert!(heard.is_empty());
		assert_eq!(heard.holding(), NOTHING);
		assert!(heard.at(TENTH).is_empty());
	}
}
