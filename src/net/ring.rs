//! What a host remembers having told one peer, so the next telling can be a
//! difference.
//!
//! ```text
//!   let number = ring.next();
//!   let (base, against) = ring.base(what_the_peer_says_it_holds);
//!
//!   Snapshot::write(number, against, ours, base, now, out)?;
//!   ring.keep(number, now);
//! ```
//!
//! **One of these per peer, not one per host.** A difference is only a
//! difference against something a *particular* far end has, and two peers that
//! joined a second apart are holding different worlds. Sharing one ring between
//! them would mean writing every snapshot against whatever the unluckiest peer
//! last acknowledged, which is a baseline most of the time.
//!
//! ## What is kept, and what it costs
//!
//! Only the slots that were occupied. A world of a thousand slots holding fifty
//! bodies is fifty records here, not a thousand - which matters because this is
//! the one structure in the subsystem whose size is multiplied by three things
//! at once: the depth of the ring, the number of peers, and the two of these a
//! conversation needs. One holds what was *sent* to a peer, to write the next
//! difference against; the other holds what was *taken* from it, to read its
//! next difference against. Every figure below is per ring, so double it for a
//! conversation. The channel's history of sent messages is multiplied the same
//! way and is a fraction of the size, which is the comparison that makes this
//! the one worth keeping sparse.
//!
//! A slot costs ninety six bytes, not the eighty eight a [`Solid`] weighs: it
//! is an `Option` of a generation and a body, and no field of a body has a
//! spare bit pattern to hide the tag in. Kept densely that is `32 * 1024 * 96 *
//! 8`, or twenty four mebibytes of mostly nothing. Kept as it is, the sandbox's
//! fifty one bodies cost about one and a quarter megabytes across eight peers.
//!
//! The one part of this that is not sparse is the table a base is spread back
//! into, which is dense out to the highest occupied slot and is never given
//! back. A world with one body in the last slot makes it ninety six kilobytes
//! per peer - real, and small beside the rest: the honest worst case for the
//! module is a world that genuinely fills its thousand slots, which is the
//! twenty four mebibytes above and not this.
//!
//! ## Choosing a base, and why there is no margin
//!
//! A peer says which snapshot it holds; if this ring still has that one, the
//! next difference is written against it, and otherwise against nothing at all.
//! That is the whole rule.
//!
//! The system this is modeled on needs a second, tighter test - it refuses a
//! base a few frames *before* the ring would lose it. The reason is worth
//! knowing so that nobody adds the same margin here out of sympathy: over there
//! a snapshot does not own the bodies it describes. They live in one circular
//! buffer shared by every client, which rolls over on its own schedule, so a
//! frame can still be in the frame ring while the bodies it pointed at have
//! been overwritten by somebody else's snapshot. The margin is a guess at how
//! far behind that can get. Here a kept snapshot owns its own copy of
//! everything it described, so there is nothing to roll off underneath it and
//! "still in the ring" is the entire question.
//!
//! One thing does get shared, and it is worth naming because it is the same
//! shape as the hazard being argued away: [`base`](Ring::base) hands out a
//! borrow of one scratch table that the next call to it clears. What makes that
//! safe is not ownership but the signature - it borrows the ring mutably, so
//! asking a second time, or keeping, or forgetting, while an answer is still
//! live is a compile error rather than a stale read. The margin over there is a
//! guess at a race; here the same race does not build.
//!
//! ## What the peer's word is worth
//!
//! It is a *lower bound* and must be treated as one - but not for the reason
//! the channel would suggest, and the difference matters because a reader who
//! goes looking for it in the channel will not find it. What a peer holds does
//! not come from an acknowledgement at all. It rides in the snapshot block as
//! `holding`, so the channel's rules about which messages it will and will not
//! acknowledge have no bearing here.
//!
//! What makes it a bound is plainer: it is a round trip old. The peer stated it
//! before this snapshot was written, and will have taken one or two more since.
//! Writing a difference against a base the peer does not have is a world that
//! never converges, so the bound is the safe way round and nothing here tries
//! to be cleverer than it.
//!
//! Writing against an *older* base than necessary costs bytes, and it is
//! correct only if the far end decodes against the base the block names rather
//! than against the newest world it has. A host sending twice a round trip
//! routinely writes snapshot `n + 2` against `n` while the peer has already
//! applied `n + 1`, and a receiver that decoded against what it currently held
//! would get a body whose unsent fields came from the wrong world.
//!
//! **So a receiver keeps one of these too**, holding what it applied rather
//! than what it sent, and reads each block against the number the block names.
//! That is why this type is not called something about sending: it is one half
//! of a conversation seen from either end, and both ends need it.

use crate::snapshot::{MAX_SLOTS, NOTHING, Slot, Solid};

/// How many snapshots one peer's ring remembers.
///
/// It matches the reach of the channel's acknowledgement field, and that is a
/// coincidence rather than a reason - worth saying because the two numbers
/// being equal invites the wrong story. A peer names what it holds with a whole
/// `u32` in the block itself, so it can name anything at all and nothing on the
/// wire bounds this.
///
/// What bounds it is memory against tolerance. Depth multiplies by peers, so
/// every step costs eight worlds; and it buys how far behind a peer may fall
/// and still be sent a difference rather than the whole world. At
/// [`EVERY`](crate::snapshot::EVERY) that is **one and a half seconds** of
/// history, which is the number to hold this against: a peer whose round trip
/// is worse than that is sent a baseline every time. Being inside it is not a
/// promise of the opposite - a peer whose own word about what it holds is lost
/// for long enough falls out of the ring just the same - but it is the
/// difference between that happening on a bad burst and happening always.
pub const DEPTH: usize = 32;

/// One snapshot as it was sent, kept only while it can still be named.
#[derive(Clone, Debug)]
struct Kept {
	/// Which snapshot it was, which is the whole of what a place in the ring
	/// proves: a number and its place agree only until the ring turns over.
	number: u32,

	/// Slot, generation and body, for the slots that held one. Sparse, and the
	/// vector is reused rather than rebuilt when this place is written again.
	held: Vec<(u16, u32, Solid)>,
}

/// The last [`DEPTH`] snapshots sent to one peer.
#[derive(Clone, Debug)]
pub struct Ring {
	/// One place per number modulo [`DEPTH`], so a snapshot both overwrites the
	/// one it displaces and can be found without a search.
	kept: Vec<Option<Kept>>,

	/// The number the next snapshot will take.
	///
	/// [`NOTHING`] once the numbers have run out, which is a state and not a
	/// number: nought says a peer holds none, so nothing can be sent under it
	/// and it is free to mean this end has counted as far as it can.
	next: u32,

	/// Where a kept snapshot is spread back out for the writer to read.
	///
	/// Kept between calls for the reason every other scratch in this engine is:
	/// a host talking to eight peers many times a second should not allocate a
	/// table per peer per snapshot.
	spread: Vec<Slot>,
}

impl Ring {
	/// A ring that has said nothing yet.
	///
	/// @return a ring holding nothing, whose first snapshot is number one
	#[must_use]
	pub fn new() -> Self {
		Self {
			kept: vec![None; DEPTH],
			next: 1,
			spread: Vec::new(),
		}
	}

	/// The number the next snapshot will take.
	///
	/// @return the number, or [`NOTHING`] if this ring has counted as far as it
	/// can and will keep nothing further
	#[must_use]
	pub const fn next(&self) -> u32 { self.next }

	/// How many snapshots are remembered.
	///
	/// @return how many of the [`DEPTH`] places are filled
	#[must_use]
	pub fn len(&self) -> usize { self.kept.iter().flatten().count() }

	/// Whether nothing has been sent yet.
	///
	/// @return whether no snapshot is remembered
	#[must_use]
	pub fn is_empty(&self) -> bool { self.len() == 0 }

	/// Whether one snapshot is still here to be written against.
	///
	/// [`NOTHING`] is answered no like any other number rather than by a guard
	/// of its own, because [`keep`](Self::keep) will not take it: it refuses
	/// anything below `next`, `next` begins at one, and once it has run out it
	/// refuses everything. So nought is never the number of anything here, and
	/// a peer saying it holds none matches nothing by arithmetic.
	///
	/// @param number - which snapshot, as the peer said it, so any u32 at all
	/// @return whether a difference may be written against it
	#[must_use]
	pub fn has(&self, number: u32) -> bool {
		self.kept[Self::slot(number)]
			.as_ref()
			.is_some_and(|kept| kept.number == number)
	}

	/// Remembers what was just sent, and moves the number on.
	///
	/// A number that is not new is refused outright rather than kept. The whole
	/// structure is an equality test on a number, so it is correct exactly
	/// while a number names one world - and the way that stops being true is a
	/// number used twice. Keeping an older one again would do it twice over: a
	/// stale world under a live name for the next difference to be written
	/// against, and a live snapshot in that slot evicted to make room for it.
	///
	/// @param number - the number it went out as, from [`next`](Self::next)
	/// @param world - what was described, by slot
	/// @return whether it was new, and so whether it was kept
	///
	/// The answer is worth having rather than working out again: "was this
	/// number new" and "is this number in the ring" are different questions,
	/// and a caller that asked [`has`](Self::has) instead would get a yes for
	/// a number that was refused *because* an older telling of it is still
	/// sitting there.
	pub fn keep(&mut self, number: u32, world: &[Slot]) -> bool {
		if self.next == NOTHING || number < self.next {
			return false;
		}

		// the vector already in this place is taken and refilled rather than
		// dropped for a fresh one. Collecting instead would be an allocation
		// per peer per snapshot, which is the thing the scratch above exists
		// to avoid, and it would be the same amount of code.
		let place = Self::slot(number);
		let mut held = self.kept[place]
			.take()
			.map_or_else(Vec::new, |kept| kept.held);

		held.clear();
		held.extend(
			world
				.iter()
				.take(MAX_SLOTS)
				.enumerate()
				.filter_map(|(slot, occupant)| {
					let (generation, solid) = (*occupant)?;
					let slot = u16::try_from(slot).ok()?;

					Some((slot, generation, solid))
				}),
		);

		self.kept[place] = Some(Kept { number, held });
		// there is no number after the last one, and a ring that went on from
		// there would hand the same one out forever. `NOTHING` is what says a
		// peer holds none, so it can never be a snapshot and is free to mean
		// this end has run out - which `keep` above refuses to go past.
		//
		// @note: `NOTHING` being nought makes this exactly `wrapping_add(1)`,
		// so no test can tell the two apart and a mutation pass will report
		// the swap surviving forever. It is written the long way because the
		// short way reads as an overflow that was shrugged at, and this one
		// was chosen.
		self.next = number.checked_add(1).unwrap_or(NOTHING);
		true
	}

	/// The world one snapshot described, to write the next difference against.
	///
	/// @param holding - what the peer says it has, or [`NOTHING`]
	/// @return the world as it was, and the number it is - or an empty world
	/// and [`NOTHING`], which is a baseline
	pub fn base(&mut self, holding: u32) -> (&[Slot], u32) {
		self.spread.clear();

		// one lookup, and the number checked where it is used. Asking `has`
		// first and then fetching again would put the check that stands
		// between a peer and its neighbor's world in another function.
		let kept = match self.kept[Self::slot(holding)].as_ref() {
			| Some(kept) if kept.number == holding => kept,
			| _ => return (&self.spread, NOTHING),
		};

		let reach = kept
			.held
			.iter()
			.map(|(slot, ..)| usize::from(*slot) + 1)
			.max()
			.unwrap_or(0);

		self.spread.resize(reach, None);

		for (slot, generation, solid) in &kept.held {
			self.spread[usize::from(*slot)] = Some((*generation, *solid));
		}

		(&self.spread, holding)
	}

	/// Forgets everything, for a peer that has gone.
	///
	/// **The numbering does not start over**, which is the whole point and is
	/// worth saying because starting it over is the obvious thing to write. A
	/// peer that reconnects into the same place is a different conversation,
	/// and a snapshot number it remembers from the last one must not name
	/// anything in this one. Counting on from where the last conversation
	/// stopped makes every number it used name nothing, forever. Counting from
	/// one again would do the exact opposite: within [`DEPTH`] ticks each of
	/// those numbers names a live world that the peer holding it never saw,
	/// and there is no handshake yet to tell the two conversations apart.
	///
	/// @note: a ring whose numbering has run out is not revived by this, so
	/// that place would take no further peer. It is the same six years as the
	/// rest of the ceiling and the alternative is worse - reusing the numbers
	/// is the one thing this method exists to prevent.
	pub fn forget(&mut self) {
		for slot in &mut self.kept {
			*slot = None;
		}

		self.spread.clear();
	}

	/// Where one number lives.
	///
	/// @param number - any number at all, including one off the wire
	/// @return its place, always below [`DEPTH`]
	const fn slot(number: u32) -> usize {
		// @note: `DEPTH` is a power of two, so this is a mask - but written as
		// the remainder it is, because the ring's correctness does not depend
		// on that and a later depth that is not a power of two should not
		// quietly break it.
		#[expect(
			clippy::as_conversions,
			reason = "a const fn, where try_from is not available"
		)]
		{
			(number as usize) % DEPTH
		}
	}
}

impl Default for Ring {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::snapshot::{Change, Snapshot};

	/// A body somewhere, with every word set to something a zero is not.
	fn somewhere() -> Solid {
		Solid {
			position: [1.0, 2.0, 3.0],
			rotation: [0.5, 0.5, 0.5, 0.5],
			velocity: [4.0, 5.0, 6.0],
			angular: [7.0, 8.0, 9.0],
			sleeping: 1,
			scale: [1.5, 2.5, 3.5],
			kind: 2,
			entity: [3, 1],
			owner: [5, 3],
		}
	}

	/// A table of slots, from what is in each of them.
	fn table(entries: &[(usize, u32, Solid)]) -> Vec<Slot> {
		let mut out = vec![
			None;
			entries
				.iter()
				.map(|it| it.0 + 1)
				.max()
				.unwrap_or(0)
		];

		for (slot, generation, solid) in entries {
			out[*slot] = Some((*generation, *solid));
		}

		out
	}

	#[test]
	fn a_ring_that_has_said_nothing_can_write_against_nothing() {
		let mut ring = Ring::new();

		assert!(ring.is_empty());
		assert_eq!(ring.next(), 1, "and the first snapshot is not nought, which means none");
		assert!(!ring.has(NOTHING), "nobody holds the number that means nobody holds one");

		let (base, against) = ring.base(7);

		assert!(base.is_empty());
		assert_eq!(against, NOTHING, "a peer claiming one that was never sent gets a baseline");
	}

	#[test]
	fn what_was_kept_comes_back_the_way_it_went_in() {
		let mut ring = Ring::new();
		let world = table(&[(0, 1, somewhere()), (4, 2, somewhere())]);

		ring.keep(ring.next(), &world);

		assert_eq!(ring.next(), 2, "the number moved on");
		assert_eq!(ring.len(), 1);
		assert!(ring.has(1));

		let (base, against) = ring.base(1);

		assert_eq!(against, 1);
		assert_eq!(base, world.as_slice(), "slot for slot, generation for generation");
	}

	#[test]
	fn only_the_slots_that_held_something_are_remembered() {
		// the point of keeping this sparsely: a world of a thousand slots with
		// two bodies in it costs two records, not a thousand.
		let mut ring = Ring::new();
		let mut world = vec![None; 1000];

		world[3] = Some((1, somewhere()));
		world[997] = Some((1, somewhere()));

		ring.keep(1, &world);

		let (base, _) = ring.base(1);

		assert_eq!(base.len(), 998, "spread back out only as far as the last body");
		assert_eq!(base[3], Some((1, somewhere())));
		assert_eq!(base[997], Some((1, somewhere())));
		assert!(base[4].is_none(), "and the holes are still holes");
	}

	#[test]
	fn a_snapshot_older_than_the_ring_is_a_baseline_again() {
		let mut ring = Ring::new();
		let world = table(&[(0, 1, somewhere())]);

		for _ in 0..=DEPTH {
			ring.keep(ring.next(), &world);
		}

		assert!(!ring.has(1), "the first has been written over");
		assert!(ring.has(2), "the one after it is the oldest still here");
		assert!(ring.has(u32::try_from(DEPTH + 1).unwrap_or(0)), "and the newest is here");

		let (base, against) = ring.base(1);

		assert!(base.is_empty());
		assert_eq!(against, NOTHING, "so a peer that fell that far behind is told everything");
	}

	#[test]
	fn a_number_the_ring_never_had_is_not_mistaken_for_the_one_in_its_place() {
		// the trap a ring of this shape sets: `1 + DEPTH` and `1` live in the
		// same slot, so a peer naming one must not be handed the other.
		let mut ring = Ring::new();
		let world = table(&[(0, 1, somewhere())]);

		ring.keep(1, &world);

		let wrapped = u32::try_from(1 + DEPTH).unwrap_or(0);

		assert!(ring.has(1));
		assert!(!ring.has(wrapped), "the same slot, a different number, and nobody has it");
		assert_eq!(ring.base(wrapped).1, NOTHING);
	}

	#[test]
	fn a_peer_that_went_away_leaves_nothing_behind() {
		let mut ring = Ring::new();
		let world = table(&[(0, 1, somewhere())]);

		ring.keep(ring.next(), &world);
		ring.keep(ring.next(), &world);
		ring.forget();

		assert!(ring.is_empty());
		assert!(!ring.has(1), "a number the last conversation used names nothing in this one");
		assert_eq!(ring.base(2).1, NOTHING);
	}

	#[test]
	fn a_peer_that_came_back_is_never_told_the_last_conversation_over_again() {
		// the reason `forget` leaves the numbering alone. Starting it over
		// would put a live world under every number the peer that left was
		// still holding, and there is no handshake to tell the two apart - so
		// a stale word from the old conversation would be believed.
		let mut ring = Ring::new();
		let old = table(&[(0, 1, somewhere())]);
		let mut fresh = somewhere();

		fresh.kind = 77;

		ring.keep(ring.next(), &old);
		ring.keep(ring.next(), &old);
		ring.forget();
		ring.keep(ring.next(), &table(&[(0, 1, fresh)]));

		assert!(!ring.has(1), "the numbers the peer that left was holding name nothing");
		assert!(!ring.has(2));
		assert_eq!(ring.base(2).1, NOTHING, "so it is told everything rather than a difference");
		assert_eq!(ring.next(), 4, "and this conversation counts on from where that one stopped");
		assert_eq!(
			ring.base(3).0[0].map(|(_, solid)| solid.kind),
			Some(77),
			"while the number this conversation did use names its own world"
		);
	}

	#[test]
	fn a_number_that_is_not_new_is_refused_rather_than_kept() {
		// three distinguishable worlds, because the damage a stale keep does
		// is to contents and not to the count: it would put `stale` under the
		// name of a snapshot the peer really holds, and evict a live one from
		// the place it landed in.
		let mut ring = Ring::new();
		let mut second = somewhere();
		let mut stale = somewhere();

		second.kind = 33;
		stale.kind = 99;

		// `1` and `1 + DEPTH` share a place, so keeping the old number again
		// would evict the live snapshot sitting there as well as lying about
		// what went out under it.
		let wrapped = 1 + u32::try_from(DEPTH).unwrap_or(0);

		ring.keep(1, &table(&[(0, 1, somewhere())]));
		ring.keep(wrapped, &table(&[(0, 1, second)]));
		ring.keep(1, &table(&[(0, 1, stale)]));

		assert_eq!(ring.next(), wrapped + 1, "the next number is past everything kept");
		assert_eq!(ring.len(), 1, "and the place they share holds one of them");
		assert!(ring.has(wrapped), "the newer snapshot is the one still here");
		assert!(!ring.has(1), "and the older number names nothing, rather than the stale world");
		assert_eq!(
			ring.base(wrapped).0[0].map(|(_, solid)| solid.kind),
			Some(33),
			"what went out under that number is what comes back"
		);
		assert_eq!(ring.base(1).1, NOTHING, "so a peer naming the old one is told everything");
	}

	#[test]
	fn every_snapshot_the_ring_holds_can_be_written_against() {
		let mut ring = Ring::new();

		for step in 0..DEPTH {
			let mut solid = somewhere();

			solid.kind = u32::try_from(step).unwrap_or(0);
			ring.keep(ring.next(), &table(&[(0, 1, solid)]));
		}

		assert_eq!(ring.len(), DEPTH, "the ring is full and nothing has been lost");

		// each of them, by number, and each giving back its own world rather
		// than its neighbor's.
		for step in 0..DEPTH {
			let number = u32::try_from(step + 1).unwrap_or(0);
			let (base, against) = ring.base(number);

			assert_eq!(against, number);
			assert_eq!(
				base[0].map(|(_, solid)| solid.kind),
				u32::try_from(step).ok(),
				"snapshot {number} came back as somebody else's"
			);
		}
	}

	/// A far end: what it holds, and nothing of what the ring holds.
	#[derive(Default)]
	struct Far {
		held: Vec<Slot>,
		holding: u32,
	}

	impl Far {
		/// Puts one change into what is held, growing to reach it.
		fn put(&mut self, change: &Change) {
			let slot = usize::from(change.slot);

			if self.held.len() <= slot {
				self.held.resize(slot + 1, None);
			}

			self.held[slot] = change
				.solid
				.map(|solid| (change.generation, solid));
		}

		/// Takes a block and puts what is in it into what is held.
		///
		/// @return what the sender said it was holding, which is its news and
		/// not this end's
		fn take(&mut self, bytes: &[u8]) -> u32 {
			let (snapshot, read) =
				Snapshot::read(&self.held, bytes).expect("what was written reads");

			assert_eq!(read, bytes.len(), "a block with nothing after it reads to its end");

			// a baseline is the whole world, and it has no way of saying that a
			// slot has emptied - so what is held goes before it is applied.
			if snapshot.against == NOTHING {
				self.held.clear();
			}

			for change in &snapshot.changes {
				self.put(change);
			}

			self.holding = snapshot.number;

			snapshot.holding
		}
	}

	/// Whether two tables describe the same world, ignoring trailing holes.
	fn same(left: &[Slot], right: &[Slot]) -> bool {
		let reach = left.len().max(right.len());

		(0..reach)
			.all(|slot| left.get(slot).copied().flatten() == right.get(slot).copied().flatten())
	}

	/// The world at one step, which changes shape as well as contents.
	///
	/// Words from three places in the table move, not one: a position early,
	/// where a change is cheap, `sleeping` in the middle and `kind` late, where
	/// a change drags every word below it onto the wire. A fixture that moved
	/// one word would exercise one column of the mask.
	fn moving(step: usize) -> Vec<Slot> {
		let mut body = somewhere();
		let mut other = somewhere();
		let along = f32::from(u16::try_from(step % 97).unwrap_or(0));

		body.position = [along, -along, 0.0];
		body.velocity[1] = along;
		body.kind = u32::try_from(step).unwrap_or(0);
		body.sleeping = u32::try_from(step % 2).unwrap_or(0);
		other.position[2] = along;
		other.kind = u32::try_from(step * 3).unwrap_or(0);

		table(&[(0, 1, body), (1 + step % 5, 1 + u32::try_from(step % 3).unwrap_or(0), other)])
	}

	#[test]
	fn what_the_ring_hands_out_is_a_base_the_far_end_can_read_against() {
		// the module's own usage, executed. Every other test here checks the
		// ring against a table this file built; only driving it through the
		// writer and into a receiver that holds nothing of the host's can show
		// a base that is well formed and wrong - which is the one kind of
		// mistake a ring makes.
		//
		// Delivery stops for a stretch longer than the ring is deep, so the
		// base it offers goes stale and the world has to be told again from
		// nothing without either end noticing the seam. The stretch ends on a
		// step whose second body is in a *different* slot from the one the far
		// end was left holding, so mending the world means taking a body away
		// as well as putting ones back - which is the half a baseline cannot
		// say and the receiver has to do for itself.
		let mut ring = Ring::new();
		let mut far = Far::default();
		let mut baselines = 0;

		for step in 0..DEPTH * 3 {
			let now = moving(step);
			let number = ring.next();
			let mut out = Vec::new();
			let (base, against) = ring.base(far.holding);

			// `NOTHING` is this end's own `holding`: the host is being told nothing
			// by anybody in this test, and the field says what the *sender* has.
			Snapshot::write(number, against, NOTHING, base, &now, &mut out).expect("it fits");
			ring.keep(number, &now);

			// nothing arrives for longer than the ring remembers, so what the
			// far end last said it held falls out of it.
			if (DEPTH..DEPTH * 2 + 3).contains(&step) {
				continue;
			}

			if against == NOTHING {
				baselines += 1;
			}

			let said = far.take(&out);

			assert_eq!(said, NOTHING, "the head carries the sender's own news, not the reader's");
			assert!(same(&far.held, &now), "the far end holds the world after step {step}");
			assert_eq!(far.holding, number, "and says so with the number it was given");
		}

		assert_eq!(baselines, 2, "one to open the conversation, one to mend it");
	}

	#[test]
	fn a_place_written_again_holds_the_new_snapshot_and_none_of_the_old_one() {
		// the vector in a place is reused rather than freshly allocated, so
		// what was in it has to go. A slot left behind would be a body the
		// far end is known to hold and does not - and if that slot were taken
		// again at the same generation, the difference would be written
		// against a body that was never sent.
		let mut ring = Ring::new();

		ring.keep(1, &table(&[(0, 1, somewhere()), (5, 1, somewhere())]));

		for number in 2..=1 + u32::try_from(DEPTH).unwrap_or(0) {
			ring.keep(number, &table(&[(0, 1, somewhere())]));
		}

		let last = 1 + u32::try_from(DEPTH).unwrap_or(0);
		let (base, against) = ring.base(last);

		assert_eq!(against, last, "the newest snapshot shares its place with the first");
		assert_eq!(base.len(), 1, "and reaches only as far as the body it holds");
		assert!(base.get(5).copied().flatten().is_none(), "the older one left nothing behind");
	}

	#[test]
	fn the_ring_speaks_of_no_more_slots_than_a_snapshot_can_carry() {
		// the ring's idea of a world and the codec's have to be the same one,
		// or the ring would remember a body that no snapshot could ever say
		// anything about.
		let mut ring = Ring::new();
		let mut world = vec![None; MAX_SLOTS + 2];

		world[0] = Some((1, somewhere()));
		world[MAX_SLOTS - 1] = Some((1, somewhere()));
		world[MAX_SLOTS] = Some((1, somewhere()));
		world[MAX_SLOTS + 1] = Some((1, somewhere()));

		ring.keep(1, &world);

		let (base, _) = ring.base(1);

		assert_eq!(base.len(), MAX_SLOTS, "the last slot under the ceiling is remembered");
		assert!(base[MAX_SLOTS - 1].is_some(), "and it is the body that was there");
		assert!(base[0].is_some(), "so is the first");
	}

	#[test]
	fn a_number_the_ring_does_not_have_gives_back_nothing_rather_than_the_last_answer() {
		// the empty slice and the `NOTHING` have to travel together. A caller
		// handed the previous answer under a number meaning "no base" would
		// write a difference from a world it has just said it is not using.
		let mut ring = Ring::new();

		ring.keep(1, &table(&[(0, 1, somewhere()), (4, 1, somewhere())]));

		assert_eq!(ring.base(1).0.len(), 5, "a base that is there fills the table");

		let (base, against) = ring.base(9);

		assert_eq!(against, NOTHING);
		assert!(base.is_empty(), "and one that is not leaves it empty");
	}

	#[test]
	fn a_base_does_not_show_the_last_one_through_its_holes() {
		// the table a base is spread into is kept between calls, so a slot the
		// previous answer filled has to be a hole in this one rather than the
		// last body that happened to land there.
		let mut ring = Ring::new();

		ring.keep(1, &table(&[(0, 1, somewhere())]));
		ring.keep(2, &table(&[(3, 1, somewhere())]));

		assert!(ring.base(1).0[0].is_some(), "the first fills slot nought");

		let (base, against) = ring.base(2);

		assert_eq!(against, 2);
		assert_eq!(base.len(), 4, "spread only as far as the body it holds");
		assert!(base[0].is_none(), "and the slot the first one filled is empty in this one");
		assert!(base[3].is_some());
	}

	#[test]
	fn nought_is_never_the_number_of_anything_the_ring_holds() {
		// `NOTHING` is nought and nought is a place in the ring like any
		// other, so a ring that took a snapshot under that number would answer
		// "I have what you hold" to a peer that holds none, and send it a
		// difference from a world it has never seen. Nothing guards against
		// that where it is asked; `keep` refusing it here is what makes the
		// question unanswerable in the first place.
		let mut ring = Ring::new();
		let world = table(&[(0, 1, somewhere())]);

		ring.keep(NOTHING, &world);

		assert!(ring.is_empty(), "nought is older than the first number and is not kept");
		assert!(!ring.has(NOTHING), "so nothing is held under the name for holding nothing");
		assert_eq!(ring.base(NOTHING).1, NOTHING, "and the answer is a baseline");
		assert!(ring.base(NOTHING).0.is_empty(), "written against an empty world");
	}

	#[test]
	fn a_ring_that_has_counted_as_far_as_it_can_stops_rather_than_starting_over() {
		// there is no number after the last one. Wrapping would hand a number
		// out twice, which is the one thing the whole structure cannot survive,
		// so it stops instead: what is already here can still be written
		// against, and nothing further is ever kept.
		let mut ring = Ring::new();
		let world = table(&[(0, 1, somewhere())]);

		ring.keep(u32::MAX, &world);

		assert_eq!(ring.next(), NOTHING, "the numbers ran out");
		assert!(ring.has(u32::MAX), "the last one is still here to be written against");

		ring.keep(ring.next(), &world);

		assert_eq!(ring.len(), 1, "and nothing further is kept");
		assert!(!ring.has(NOTHING));
	}

	#[test]
	fn a_slot_that_changed_hands_comes_back_as_the_occupant_of_its_own_snapshot() {
		// the generation is what tells two occupants of one slot apart, and a
		// base that gave back the wrong one would have the far end inherit a
		// dead body's fields for every word the new one did not send.
		let mut ring = Ring::new();
		let mut after = somewhere();

		after.kind = 5;

		ring.keep(1, &table(&[(0, 1, somewhere())]));
		ring.keep(2, &table(&[(0, 2, after)]));

		assert_eq!(ring.base(1).0[0], Some((1, somewhere())));
		assert_eq!(ring.base(2).0[0], Some((2, after)));
	}
}
