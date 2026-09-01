//! The arenas the game keeps its own state in.
//!
//! This exists so that adding a value to gameplay stops meaning a rebuild of
//! `colby_core` and a restart. The host owns a fixed run of bytes and never
//! looks inside; the game declares a `#[repr(C)]` struct, stamps it with a
//! layout number, and reads it back every frame.
//!
//! **There are three of them, and which one a value goes in is the question
//! this module exists to make somebody answer.**
//!
//! | arena | who writes it | who will be told | in a save |
//! | --- | --- | --- | --- |
//! | [`World::state`](crate::abi::World::state) | the host | everybody | yes |
//! | [`World::players`](crate::abi::World::players) | the host | that peer alone | yes |
//! | [`World::local`](crate::abi::World::local) | whoever is running | nobody | **no** |
//!
//! **Two of those columns are about a commit that has not happened.** Nothing
//! is sent anywhere yet - there is no snapshot - and every world this engine
//! builds is its own host, so "who writes it" is one process today. What is
//! true *now* is the last column, and the split itself: three arenas with
//! three layout numbers, and a save that carries two of them.
//!
//! The world's arena is what the world is: props, the map, the score. A
//! player's is what is true of one person: what they are holding, which tool
//! they picked, where their feet are. The local one is what is true of one
//! *screen*: a camera, a panel handle, the voice a gun is humming with. Put a
//! camera in the world's arena and everybody looks through one pair of eyes;
//! put a score in the local one and nobody else ever hears about it.
//!
//! **The rule for deciding is who would be wrong if it were copied.** If a
//! second person appearing would want their own copy, it is a player's. If a
//! second *window* onto the same person would want its own, it is local.
//! Everything else is the world's.
//!
//! Each is an ordinary [`GameState`] with its own layout number, so a game
//! bumps them independently and a change to one does not zero the other two.
//!
//! ```ignore
//! #[repr(C)]
//! #[derive(Clone, Copy, colby_core::bytemuck::Pod, colby_core::bytemuck::Zeroable)]
//! struct State { spin: f32, lives: u32 }
//!
//! const LAYOUT: u64 = 1;
//!
//! let (state, fresh) = world.state.get::<State>(LAYOUT);
//! ```
//!
//! Change the struct, bump `LAYOUT`, save: the arena zeroes itself, `fresh`
//! comes back `true`, and the game starts over from a known state - all without
//! the process restarting.
//!
//! Forgetting to bump it is not unsound. `T: Pod` means every bit pattern is a
//! valid `T`, so the worst case is reading yesterday's bytes as today's fields:
//! wrong numbers, not undefined behavior.

use super::net::{MAX_PEERS, PeerId};
use crate::bytemuck::{self, Pod};

/// How many bytes of state the game gets.
///
/// Fixed, because a growable arena would have to live behind a pointer the
/// game could outlive. Raising this is one constant, and a restart.
pub const STATE_BYTES: usize = 4096;

/// The alignment the arena guarantees.
///
/// Enough for anything with `f64`, `u64` or a 16-byte vector in it.
pub const STATE_ALIGN: usize = 16;

/// A fixed run of bytes the game interprets however it likes.
#[repr(C, align(16))]
#[derive(Clone)]
pub struct GameState {
	bytes: [u8; STATE_BYTES],

	/// The layout number the bytes were last written under.
	///
	/// Zero means nothing has claimed the arena yet, which is why a layout of
	/// zero is refused.
	layout: u64,
}

impl GameState {
	/// An arena nobody has claimed.
	#[must_use]
	pub const fn new() -> Self { Self { bytes: [0; STATE_BYTES], layout: 0 } }

	/// Reads the arena as `T`, resetting it if the layout has moved.
	///
	/// @param layout - the game's own version number for `T`, never zero
	/// @return the state, and whether it was just zeroed
	///
	/// # Panics
	///
	/// If `layout` is zero, which is reserved for "unclaimed".
	pub fn get<T: Pod>(&mut self, layout: u64) -> (&mut T, bool) {
		const {
			assert!(size_of::<T>() <= STATE_BYTES, "game state is larger than the arena");
			assert!(
				align_of::<T>() <= STATE_ALIGN,
				"game state needs more alignment than the arena has"
			);
		}

		assert!(layout != 0, "a layout number of zero is reserved for an unclaimed arena");

		let fresh = self.layout != layout;
		if fresh {
			self.bytes = [0; STATE_BYTES];
			self.layout = layout;
		}

		(bytemuck::from_bytes_mut(&mut self.bytes[..size_of::<T>()]), fresh)
	}

	/// The layout number currently stamped on the arena.
	#[must_use]
	pub const fn layout(&self) -> u64 { self.layout }

	/// The whole arena as bytes, whatever they currently mean.
	///
	/// For writing the arena down and reading it back, and for nothing else:
	/// the host still never looks *inside*, it only copies. A saved world that
	/// left this out would come back with every entity standing where it was
	/// and a game holding handles to none of them, because the handles a game
	/// keeps are in here.
	#[must_use]
	pub const fn raw(&self) -> &[u8; STATE_BYTES] { &self.bytes }

	/// Puts bytes back into the arena, with the number they were written
	/// under.
	///
	/// The other half of [`raw`](Self::raw), and it is deliberately blunt: it
	/// copies and stamps, and asks nothing about what the bytes mean. What
	/// keeps it honest is the layout number - a build whose own number has
	/// moved on will find these bytes stamped with the old one and zero them
	/// on the next [`get`](Self::get), reporting itself fresh, which is
	/// already what happens when a game changes its state struct.
	///
	/// A short slice leaves the rest of the arena zeroed; a long one is
	/// truncated. Neither is unsound, for the reason the whole arena is not:
	/// every bit pattern is a valid `T`.
	///
	/// @param bytes - what to copy in
	/// @param layout - the number to stamp them with, zero for unclaimed
	pub fn put_raw(&mut self, bytes: &[u8], layout: u64) {
		let taken = bytes.len().min(STATE_BYTES);

		self.bytes = [0; STATE_BYTES];
		self.bytes[..taken].copy_from_slice(&bytes[..taken]);
		self.layout = layout;
	}

	/// Forgets everything, so the next [`get`](Self::get) reports itself fresh.
	pub fn reset(&mut self) {
		self.bytes = [0; STATE_BYTES];
		self.layout = 0;
	}
}

impl Default for GameState {
	fn default() -> Self { Self::new() }
}

impl core::fmt::Debug for GameState {
	/// The layout number, and how many bytes there are rather than what they
	/// are.
	///
	/// Written by hand rather than derived because the derived one prints four
	/// thousand bytes, and because there is nothing to say about them: the host
	/// does not know what they mean and neither does anything that would be
	/// printing this.
	fn fmt(&self, out: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		out.debug_struct("GameState")
			.field("layout", &self.layout)
			.field("bytes", &STATE_BYTES)
			.finish()
	}
}

/// One block of gameplay state per peer, and the peers themselves.
///
/// Two things in one, and they are one thing because separating them is what
/// makes a stale handle read somebody else's state. The blocks are an array
/// indexed by [`PeerId::slot`]; the generations beside them are what say
/// whether the peer asking is the peer that block belongs to. A table of
/// blocks with no generations would be indexed by a bare number, and a peer
/// that left would be answered with whatever its replacement is holding -
/// which is the whole reason [`PeerId`] carries a generation at all.
///
/// **This is therefore the thing that mints a [`PeerId`]**, the way
/// [`Bodies`](super::Bodies) is the only thing that mints a
/// [`BodyId`](super::BodyId) - in both cases the table rather than one method
/// of it, since restoring and walking hand out handles too.
/// [`admit`](Self::admit) takes the lowest free slot *at one or above*, so
/// slot zero stays the host's and a client cannot be handed
/// [`PeerId::HOST`] by an off-by-one. That rule was a sentence of prose when
/// `PeerId` was written and it is code here.
///
/// **Slot zero is alive from the moment the table exists**, at the host's own
/// generation, because a world with nobody in it is still a world with a host
/// in it - and offline that host is the person playing.
#[derive(Clone, Debug)]
pub struct Players {
	blocks: Vec<GameState>,
	generations: Vec<u32>,
	alive: Vec<bool>,
}

impl Players {
	/// A table holding the host and nobody else.
	#[must_use]
	pub fn new() -> Self {
		let mut generations = vec![0; MAX_PEERS];
		let mut alive = vec![false; MAX_PEERS];

		// the host is not admitted, it is simply there. @ref the type comment.
		generations[PeerId::HOST.slot()] = PeerId::HOST.generation();
		alive[PeerId::HOST.slot()] = true;

		Self {
			blocks: vec![GameState::new(); MAX_PEERS],
			generations,
			alive,
		}
	}

	/// Takes the lowest free slot above the host's, for a peer that has
	/// arrived.
	///
	/// The block is not cleared here, because a free slot is always already a
	/// clear one: everything that frees a slot clears it on the way out. That
	/// is a smaller invariant than clearing at both ends and it is the one
	/// worth having, since only one of the two could ever be observed.
	///
	/// @return the peer's handle, or [`PeerId::NONE`] if the table is full
	pub fn admit(&mut self) -> PeerId {
		// from one: slot zero is the host's forever, and handing it out is the
		// one mistake this table exists to make impossible.
		//
		// @note: this bound is deliberately redundant. Slot zero is occupied
		// from the moment the table exists, so the aliveness check below skips
		// it anyway, and no test can tell the two apart - starting this range
		// at zero passes the whole suite. It is kept because the two guards
		// fail differently: one goes if somebody ever makes the host's slot
		// free, the other if somebody reorders this loop. What neither covers
		// is the saturation below, which is why that is a third check rather
		// than a comment.
		for slot in 1..MAX_PEERS {
			if self.alive[slot] {
				continue;
			}

			let Ok(index) = u32::try_from(slot) else {
				continue;
			};

			// a slot already at the top is retired rather than reused, and the
			// reason is not that its number reads as the host's - it does not,
			// because `is_host` reads the slot too and this loop never yields
			// slot zero. It is that the *next* one would have to saturate:
			// every occupant after this would be minted at `u32::MAX`, two
			// peers would be indistinguishable, and a handle to the one that
			// left would resolve straight onto its replacement. That is the
			// whole thing a generation is for.
			//
			// A slot one below the top still has one honest identity in it, so
			// the check is on what is there rather than on what would come
			// next. Retirement is permanent: nothing lowers a generation, and
			// nothing reports the loss except the table getting smaller.
			if self.generations[slot] == u32::MAX {
				continue;
			}

			let next = self.generations[slot].saturating_add(1);

			self.generations[slot] = next;
			self.alive[slot] = true;

			return PeerId::at(index, next);
		}

		PeerId::NONE
	}

	/// Lets a peer go, and forgets what it was holding.
	///
	/// The block is cleared rather than left: what a peer had is a claim about
	/// somebody who is no longer here, and the next occupant of the slot must
	/// not read it. Its generation stays where it is, so a handle to the peer
	/// that left resolves to nothing rather than to its replacement.
	///
	/// **That last sentence holds within a run and not across a
	/// [`restore`](Self::restore)**, which puts generations back as they were
	/// and therefore rewinds them. A handle minted after the description was
	/// captured can be minted a second time afterwards. That is the same
	/// bargain every other table here makes - putting a world back means
	/// handles from after it was written are not in it - and it is worth
	/// knowing because one arena is *not* put back, so a peer kept in
	/// [`World::local`](crate::abi::World::local) outlives the rewind.
	///
	/// The host is never let go: it is not a peer that arrived.
	///
	/// @note: `false` means two things - nobody by that name is here, and the
	/// name is the host's. A caller that wants to warn about a disconnect for
	/// somebody it never admitted has to ask [`here`](Self::here) as well, or
	/// it will warn about the host.
	///
	/// @param peer - who is leaving
	/// @return whether that peer was here to leave
	pub fn forget(&mut self, peer: PeerId) -> bool {
		if peer.is_host() || !self.here(peer) {
			return false;
		}

		self.alive[peer.slot()] = false;
		self.blocks[peer.slot()].reset();

		true
	}

	/// Whether this peer is one the table currently holds.
	///
	/// @note: the first of the three is redundant today and is kept for the
	/// reason the bound in [`admit`](Self::admit) is. A peer naming nobody has
	/// a generation of nothing, and no live slot has one - `restore` refuses
	/// to occupy a slot that does, and `admit` never mints one. So no test can
	/// tell this check from its absence. It is the statement that nobody is
	/// powerless, and it is the line that would still hold if either of the
	/// other two were ever loosened.
	#[must_use]
	pub fn here(&self, peer: PeerId) -> bool {
		peer.is_some()
			&& self
				.generations
				.get(peer.slot())
				.is_some_and(|&generation| generation == peer.generation())
			&& self.alive.get(peer.slot()).copied() == Some(true)
	}

	/// Reads one peer's block as `T`, resetting it if the layout has moved.
	///
	/// @param peer - whose block
	/// @param layout - the game's own version number for `T`, never zero
	/// @return the state and whether it was just zeroed, or nothing if that
	/// peer is not here
	pub fn get<T: Pod>(&mut self, peer: PeerId, layout: u64) -> Option<(&mut T, bool)> {
		if !self.here(peer) {
			return None;
		}

		Some(self.blocks[peer.slot()].get::<T>(layout))
	}

	/// One peer's block, whatever its bytes currently mean.
	#[must_use]
	pub fn block(&self, peer: PeerId) -> Option<&GameState> {
		self.here(peer).then(|| &self.blocks[peer.slot()])
	}

	/// The same, to write into.
	pub fn block_mut(&mut self, peer: PeerId) -> Option<&mut GameState> {
		self.here(peer)
			.then(|| &mut self.blocks[peer.slot()])
	}

	/// Every peer the table holds, in slot order, the host first.
	pub fn iter(&self) -> impl Iterator<Item = (PeerId, &GameState)> {
		(0..MAX_PEERS)
			.filter(|&slot| self.alive[slot])
			.filter_map(move |slot| {
				let index = u32::try_from(slot).ok()?;
				let peer = PeerId::at(index, self.generations[slot]);

				Some((peer, &self.blocks[slot]))
			})
	}

	/// How many peers are here, the host counted.
	#[must_use]
	pub fn len(&self) -> usize { self.alive.iter().filter(|here| **here).count() }

	/// Always `false`: the host is always here.
	#[must_use]
	pub fn is_empty(&self) -> bool { false }

	/// The generation living in a slot, whether or not anybody is in it.
	///
	/// For writing the table down and reading it back, and nothing else.
	///
	/// @param slot - which slot
	#[must_use]
	pub fn generation(&self, slot: usize) -> u32 {
		self.generations.get(slot).copied().unwrap_or(0)
	}

	/// Puts a whole table back, slot for slot and generation for generation.
	///
	/// Generations are put back rather than derived, which is the half of
	/// [`Bodies::restore`](super::Bodies::restore)'s contract that carries
	/// over: a handle a game kept has to mean afterwards what it meant before.
	/// The other half does not. That one resizes its table to the description
	/// and lifts a generation of zero to one; this one is always
	/// [`MAX_PEERS`] slots and *refuses* a zero instead, because a peer table
	/// has a fixed shape and a slot nobody can name is a slot nobody is in.
	///
	/// **A restore can leave peers here that nobody is connected to.** The
	/// description says who was in the world when it was written, and whether
	/// those people are still on the far end of a socket is not something this
	/// table can know. Reconciling the two is owed by whoever restored, the
	/// same way the solver's own caches are.
	///
	/// The host's slot is forced back to itself whatever the description says,
	/// because a file claiming somebody else is the host is a file this world
	/// cannot honor - the process reading it is the one deciding.
	///
	/// @param generations - one per slot, in slot order
	/// @param blocks - the occupied slots, each with its bytes and layout
	pub fn restore(&mut self, generations: &[u32], blocks: &[(usize, Vec<u8>, u64)]) {
		for slot in 0..MAX_PEERS {
			self.generations[slot] = generations.get(slot).copied().unwrap_or(0);
			self.alive[slot] = false;
			self.blocks[slot].reset();
		}

		for (slot, bytes, layout) in blocks {
			if *slot >= MAX_PEERS {
				continue;
			}

			// a description may say anything. A slot whose generation is
			// nothing is a slot nobody can hold a handle to, so marking it
			// occupied would put a block behind a peer that cannot exist. And
			// two records for one slot is a description disagreeing with
			// itself, where taking the last silently is the worst of the three
			// answers available.
			if self.generations[*slot] == 0 || self.alive[*slot] {
				continue;
			}

			self.blocks[*slot].put_raw(bytes, *layout);
			self.alive[*slot] = true;
		}

		self.generations[PeerId::HOST.slot()] = PeerId::HOST.generation();
		self.alive[PeerId::HOST.slot()] = true;
	}
}

impl Default for Players {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::bytemuck::Zeroable;

	#[repr(C)]
	#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
	struct Example {
		count: u32,
		flag: u32,
	}

	#[repr(C)]
	#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
	struct Wider {
		count: u32,
		flag: u32,
		extra: u64,
	}

	#[test]
	fn the_first_claim_hands_back_zeroes_and_says_so() {
		let mut state = GameState::new();
		let (value, fresh) = state.get::<Example>(1);

		assert!(fresh, "nobody had claimed the arena");
		assert_eq!(*value, Example { count: 0, flag: 0 }, "and it starts zeroed");
	}

	#[test]
	fn writes_survive_the_next_read() {
		let mut state = GameState::new();
		state.get::<Example>(1).0.count = 7;

		let (value, fresh) = state.get::<Example>(1);

		assert!(!fresh, "the layout has not moved");
		assert_eq!(value.count, 7, "so the bytes are still the game's");
	}

	#[test]
	fn bumping_the_layout_starts_over() {
		let mut state = GameState::new();
		state.get::<Example>(1).0.count = 7;

		let (value, fresh) = state.get::<Wider>(2);

		assert!(fresh, "a new layout number resets the arena");
		assert_eq!(value.count, 0, "and everything reads as zero");
		assert_eq!(state.layout(), 2, "the arena remembers who owns it now");
	}

	#[test]
	fn a_reset_arena_reports_itself_fresh_again() {
		let mut state = GameState::new();
		state.get::<Example>(1).0.count = 7;
		state.reset();

		let (value, fresh) = state.get::<Example>(1);

		assert!(fresh, "reset unclaims the arena");
		assert_eq!(value.count, 0, "and zeroes it");
	}

	#[test]
	#[should_panic(expected = "reserved for an unclaimed arena")]
	fn a_layout_of_zero_is_refused() {
		let mut state = GameState::new();
		let (value, _) = state.get::<Example>(0);

		assert_eq!(value.count, 0, "unreachable: the call above panics");
	}

	#[test]
	fn a_fresh_table_holds_the_host_and_nobody_else() {
		let players = Players::new();

		assert_eq!(players.len(), 1, "a world with nobody in it still has a host");
		assert!(!players.is_empty());
		assert!(players.here(PeerId::HOST));
		assert_eq!(
			players.generation(PeerId::HOST.slot()),
			PeerId::HOST.generation(),
			"at the generation the host's own handle names"
		);
		assert_eq!(players.iter().count(), 1, "and walking the table walks the host and stops");
	}

	#[test]
	fn a_client_is_never_handed_the_hosts_slot() {
		let mut players = Players::new();

		for _ in 0..MAX_PEERS {
			let peer = players.admit();

			if peer.is_some() {
				assert_ne!(peer.slot(), PeerId::HOST.slot(), "slot zero is not handed out");
				assert!(!peer.is_host(), "and neither is the host's own identity");
			}
		}

		assert_eq!(
			players.len(),
			MAX_PEERS,
			"the host plus every slot above it, and then no more"
		);
		assert_eq!(players.admit(), PeerId::NONE, "a full table refuses rather than wraps");
	}

	#[test]
	fn a_slot_that_changed_hands_answers_only_its_new_occupant() {
		let mut players = Players::new();
		let first = players.admit();

		players
			.get::<Example>(first, 1)
			.expect("it is here")
			.0
			.count = 7;

		assert!(players.forget(first), "it was here to leave");
		assert!(!players.here(first), "and is not afterwards");
		assert!(players.get::<Example>(first, 1).is_none(), "so it reads nothing");

		let second = players.admit();

		assert_eq!(second.slot(), first.slot(), "the slot is handed out again");
		assert_ne!(second, first, "to a different peer");
		assert!(!players.here(first), "and the one that left stays gone");
		assert_eq!(
			players
				.get::<Example>(second, 1)
				.expect("it is here")
				.0
				.count,
			0,
			"reading what the last occupant left is the thing this prevents"
		);
	}

	#[test]
	fn the_host_is_not_a_peer_that_can_leave() {
		let mut players = Players::new();

		assert!(!players.forget(PeerId::HOST), "the host did not arrive and does not leave");
		assert!(players.here(PeerId::HOST));
		assert!(!players.forget(PeerId::NONE), "and nobody was never here");
	}

	#[test]
	fn a_peer_reads_its_own_block_and_no_one_elses() {
		let mut players = Players::new();
		let mine = players.admit();
		let theirs = players.admit();

		players
			.get::<Example>(mine, 1)
			.expect("mine is here")
			.0
			.count = 4;
		players
			.get::<Example>(theirs, 1)
			.expect("theirs is here")
			.0
			.count = 9;

		assert_eq!(
			players
				.get::<Example>(mine, 1)
				.expect("here")
				.0
				.count,
			4
		);
		assert_eq!(
			players
				.get::<Example>(theirs, 1)
				.expect("here")
				.0
				.count,
			9
		);
		assert_eq!(
			players
				.get::<Example>(PeerId::HOST, 1)
				.expect("the host is here")
				.0
				.count,
			0,
			"and the host's own block is a third one"
		);
	}

	#[test]
	fn a_table_comes_back_slot_for_slot_and_generation_for_generation() {
		let mut players = Players::new();
		let first = players.admit();
		let second = players.admit();

		players
			.get::<Example>(second, 1)
			.expect("here")
			.0
			.count = 5;

		let generations: Vec<u32> = (0..MAX_PEERS)
			.map(|slot| players.generation(slot))
			.collect();
		let blocks: Vec<(usize, Vec<u8>, u64)> = players
			.iter()
			.map(|(peer, block)| (peer.slot(), block.raw().to_vec(), block.layout()))
			.collect();

		let mut put_back = Players::new();
		put_back.admit();
		put_back.restore(&generations, &blocks);

		assert!(put_back.here(first), "the handles a game kept still resolve");
		assert!(put_back.here(second));
		assert_eq!(
			put_back
				.get::<Example>(second, 1)
				.expect("here")
				.0
				.count,
			5,
			"and what that peer was holding came with it"
		);
		assert!(put_back.here(PeerId::HOST), "the host is always here afterwards");
	}

	#[test]
	fn a_restore_cannot_make_somebody_else_the_host() {
		let mut players = Players::new();
		let mut generations = vec![0; MAX_PEERS];

		// a file claiming the host's slot belongs to somebody else. The
		// process reading it is the one deciding what it is, so this is put
		// back rather than honored.
		generations[PeerId::HOST.slot()] = 3;
		players.restore(&generations, &[]);

		assert!(players.here(PeerId::HOST), "the host is the host whatever a file says");
		assert_eq!(players.generation(PeerId::HOST.slot()), PeerId::HOST.generation());
	}

	#[test]
	fn a_block_past_the_end_of_the_table_is_dropped_rather_than_taken() {
		let mut players = Players::new();

		// the boundary itself, not a number well past it: `MAX_PEERS` is the
		// first slot that is not one, and indexing with it is the panic this
		// guard exists to stop.
		players.restore(&[0; MAX_PEERS], &[
			(MAX_PEERS, vec![1; 8], 2),
			(MAX_PEERS + 4, vec![1; 8], 2),
		]);

		assert_eq!(players.len(), 1, "only the host, and nothing landed anywhere");
	}

	#[test]
	fn a_restore_forgets_whoever_the_description_does_not_mention() {
		let mut players = Players::new();
		let staying = players.admit();
		let going = players.admit();

		let generations: Vec<u32> = (0..MAX_PEERS)
			.map(|slot| players.generation(slot))
			.collect();
		let block = (staying.slot(), vec![0_u8; 8], 1_u64);

		// the description names one of the two. The other was here a moment
		// ago and is not in the world being loaded, so it has to go.
		players.restore(&generations, &[block]);

		assert!(players.here(staying), "the one the file knows about is here");
		assert!(
			!players.here(going),
			"and the one it does not is gone, rather than surviving the load"
		);
		assert_eq!(players.len(), 2, "the host and the one that was named");
	}

	#[test]
	fn a_slot_with_no_generation_is_nobody_however_a_description_dresses_it() {
		let mut players = Players::new();

		// every generation nothing: a file saying somebody is in slot three
		// cannot be honored, because no handle to slot three could exist.
		players.restore(&[0; MAX_PEERS], &[(3, vec![7; 8], 1)]);

		assert!(!players.here(PeerId::at(3, 0)), "a zero generation names nobody");
		assert_eq!(players.len(), 1, "so nothing was put there");
	}

	#[test]
	fn a_slot_whose_generations_have_run_out_is_retired_rather_than_reused() {
		let mut players = Players::new();
		let mut generations = vec![0; MAX_PEERS];

		// one below the top: there is still one honest identity in this slot,
		// so it is handed out rather than retired.
		generations[1] = u32::MAX - 1;
		players.restore(&generations, &[]);

		let last = players.admit();

		assert_eq!(last.slot(), 1, "the lowest free slot, as always");
		assert_eq!(last.generation(), u32::MAX, "and its last occupant");
		assert!(!last.is_host(), "which is not the host, because a slot is read too");

		// and now it is spent. Another turn would have to saturate, which
		// would mint this same handle a second time.
		assert!(players.forget(last));

		let next = players.admit();

		assert!(next.is_some(), "there are eight other slots");
		assert_ne!(next.slot(), 1, "but not that one, ever again");
		assert_ne!(next, last, "and nothing is minted twice");
	}

	#[test]
	fn two_records_for_one_slot_are_not_taken_in_turn() {
		let mut players = Players::new();
		let mut generations = vec![0; MAX_PEERS];

		generations[2] = 1;

		// a description disagreeing with itself. Taking the last silently is
		// the worst of the three answers available, so the first wins and the
		// rest are refused.
		players.restore(&generations, &[(2, vec![7; 8], 1), (2, vec![9; 8], 1)]);

		let peer = PeerId::at(2, 1);
		let held = players
			.block(peer)
			.expect("the first record landed")
			.raw()[0];

		assert_eq!(held, 7, "the first of the two, not the last");
		assert_eq!(players.len(), 2, "and it is one peer rather than two");
	}
}
