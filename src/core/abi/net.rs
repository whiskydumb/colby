//! Who somebody is, and what part this process plays in what they own.
//!
//! Two types and one rule between them. A [`PeerId`] names one endpoint of a
//! conversation; a [`Role`] says what this process is entitled to do about a
//! thing that peer owns. Nothing here opens a socket and nothing here is sent:
//! the wire is a crate of its own, and what crosses it is a later commit. What
//! is here is the vocabulary the rest of it needs to be written in.
//!
//! **The host is the only thing that simulates.** That is the whole authority
//! model in one line, and every role below follows from it: a host is the
//! [`Role::Authority`] over everything in its world, including a body a client
//! owns, and a client works out nothing except what it owns itself.
//!
//! **A role is worked out, never stored.** A body carries an owner and the
//! world carries who this process is; those two are the whole of the input,
//! and a third field saying "and my role is" is one that can disagree with
//! them, with one of the two then having to win. That is the argument that
//! took a redundant flag bit back out of the mesh format.
//!
//! Of the five systems read for this, exactly one stores a role - and it
//! stores a **pair**, its own and the far end's, replicates both, and swaps
//! them as they are read so that each side's "mine and yours" comes out right.
//! It has to: its notion of an owner is a pointer to another object, and
//! asking who that is means walking the chain of them up to a connection,
//! which exists only on the machine that accepted it. A client cannot work the
//! answer out from what it holds, so it is told instead. Here an owner is
//! a [`PeerId`], which is the same number on both machines, so both machines
//! reach the same answer out of what they already hold. That engine also pays
//! for the stored pair in a way worth knowing about: the far end's role is
//! *projected per connection* on the way out, so one thing is written as
//! autonomous to the peer that owns it and as simulated to everybody else, and
//! the real value is put back afterwards. A derived role gets that for
//! nothing, because each peer works out its own answer and no answer is ever
//! sent. Of the other four, one recomputes cached booleans inside the setter
//! that changes the owner - a derivation with a memo in front of it - and one
//! derives from which markers a receiver was handed. The last two have no
//! per-object owner at all: one compares two integers at each of the several
//! dozen places it wants to know whether something is the local player, and
//! the other is host-authoritative with no such notion in it anywhere.
//!
//! **Three roles and not four.** The engine with the enum has a fourth
//! meaning "replicated to nobody", and it earns its place there by gating two
//! real things: whether a remote call is absorbed, and whether a thing is in
//! the replication list at all. Both of those are deferred here with named
//! triggers rather than fields, so a fourth would be a variant every match
//! had to carry and nothing could ever produce.
//!
//! ```text
//!   let peer = world.peer;
//!
//!   for (_, body) in world.bodies.iter() {
//!       match body.role(peer) {
//!           | Role::SimulatedProxy  => ..  // be told where it went
//!           | Role::AutonomousProxy => ..  // work it out; expect a correction
//!           | Role::Authority       => ..  // work it out; tell everybody
//!       }
//!   }
//! ```

/// A handle to one endpoint of a conversation.
///
/// Generational, like a body's handle and unlike an asset's, and for a
/// stronger reason than either: a peer disconnects and its slot is handed to
/// whoever turns up next, while a body it owned may still be lying where it
/// was dropped. A bare slot number there is a prop that changes hands the
/// moment a stranger connects.
///
/// The field agrees from both sides. The two systems read that name an owner
/// on the object identify a peer with something nothing ever reuses - one
/// mints a fresh unique value per connection and says in a comment that it is
/// avoiding allocated sequential ids on purpose, the other counts upwards from
/// past a reserved range and recycles nothing until it wraps. The one whose
/// peer slots *are* reused is a bare array index with no generation anywhere
/// on it, and that system has no per-client owner on an entity at all - so it
/// has nothing that can go stale. What it does carry is an owner that is an
/// *entity* number, kept on the server side of its own structure and never put
/// on the wire.
///
/// Which makes a numeric peer on a body a step past what any of them do, and
/// what makes it workable here is the property the whole wire format will
/// stand on: the *world's* tables - entities, bodies, joints - are slotted and
/// generational, and a restore rebuilds them slot for slot and generation for
/// generation, so a number written down on one machine names the same thing
/// when it is read on another.
///
/// `Pod` for the reason every other handle here is: a game keeps its handles
/// in the arena, and a zeroed arena has to read back as [`NONE`](Self::NONE)
/// rather than as something that could resolve.
///
/// **A generation of zero means nobody, and nothing in this file can make that
/// true.** Every other handle here is guaranteed it by the table that hands it
/// out - a body's generation is bumped before the handle leaves and lifted to
/// one where a slot is reused, so zero can never escape. There is no table of
/// peers yet, so this is a reservation rather than a guarantee, and whoever
/// writes that table owes the same discipline. `Pod` means it cannot be an
/// enforcement in any case: any crate can cast eight bytes into one of these.
#[repr(C)]
#[derive(
	Clone,
	Copy,
	Debug,
	Default,
	PartialEq,
	Eq,
	Hash,
	crate::bytemuck::Pod,
	crate::bytemuck::Zeroable,
)]
pub struct PeerId {
	index: u32,
	generation: u32,
}

impl PeerId {
	/// The peer that simulates the world.
	///
	/// Slot zero, at a generation nothing hands out. Both halves are
	/// deliberate and they defend different things.
	///
	/// The **slot** is reserved so that a block kept per peer has somewhere to
	/// put the host's. Nothing in this file enforces that; the table that
	/// hands slots out owes it, and it does not exist yet.
	///
	/// The **generation** is `u32::MAX` because the tables in this engine all
	/// mint a handle the same way - take the lowest free slot, add one to its
	/// generation - and the first occupant of the first slot in such a table
	/// is therefore `{0, 1}`. Had that been this value, a peer table written
	/// in the house style would hand the *first client to connect* an identity
	/// that reads back as the host, and with it authority over every body in
	/// the world. So the two defenses are kept separate on purpose: getting
	/// the reservation wrong is then a client sharing the host's block, which
	/// is a bug, rather than a client being the host, which is the end of the
	/// authority model.
	///
	/// A world nobody has networked holds this in
	/// [`World::peer`](crate::abi::World::peer), because a process playing on
	/// its own is its own authority. That is a deliberate value rather than
	/// the zero one, and the difference is the whole of the safety here: see
	/// [`NONE`](Self::NONE).
	pub const HOST: Self = Self { index: 0, generation: u32::MAX };
	/// Nobody.
	///
	/// What a body nobody owns says - the map, a prop lying in a corner, every
	/// body in a world that has never heard of a network. Also what a client
	/// that has connected but has not yet been told who it is is meant to say
	/// about itself, and the reason this is the *zero* value rather than
	/// [`HOST`](Self::HOST): a peer read out of a freshly zeroed arena works
	/// nothing out and owns nothing. The powerless answer is the one a zero
	/// should give.
	pub const NONE: Self = Self { index: 0, generation: 0 };

	/// Which occupant of that slot this names.
	#[must_use]
	pub const fn generation(self) -> u32 { self.generation }

	/// Whether this is the peer that simulates the world.
	#[must_use]
	pub const fn is_host(self) -> bool {
		self.index == Self::HOST.index && self.generation == Self::HOST.generation
	}

	/// Whether this names anybody at all.
	///
	/// A `true` here does not mean that peer is still connected - only the
	/// table of them answers that.
	#[must_use]
	pub const fn is_some(self) -> bool { self.generation != 0 }

	/// The slot this addresses, whatever peer sits there now.
	///
	/// Unvalidated, and it has to be: every bit pattern is a legal `PeerId`,
	/// so this can be any number a `u32` holds. It is also meaningless unless
	/// [`is_some`](Self::is_some), because a peer naming nobody reports slot
	/// zero, which is the host's. Whoever indexes a table with this bounds
	/// checks it and asks that question first - the same arrangement a body's
	/// slot has, where every consumer goes through a lookup that rejects a
	/// generation of zero.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "lossless here, and try_from is not const"
	)]
	pub const fn slot(self) -> usize { self.index as usize }

	/// Some peer at a slot, as the table that hands them out would mint one.
	///
	/// Test-only, and deliberately: a handle is minted by the table that owns
	/// it, and the table of peers arrives with the arena that is kept per
	/// peer. Until then this is what lets the rest of the crate check what a
	/// client's answers are.
	#[cfg(test)]
	pub(crate) const fn at(index: u32, generation: u32) -> Self { Self { index, generation } }
}

/// What part this process plays in one thing.
///
/// Not stored anywhere: it is [`of`](Self::of) two peers, worked out wherever
/// somebody needs it. @ref the module comment for why, and note there is
/// deliberately no `Default` - a role is an answer to a question, and there is
/// no question to answer until both peers are in hand.
///
/// **Declared in order of how much this process decides**, which is the order
/// the one engine that has this enum declares its own in and compares them by.
/// Nothing here compares two roles, so nothing derives an ordering. What the
/// order does buy is that the discriminant a zero would read as is
/// `SimulatedProxy`, the powerless one - which matters not at all today,
/// because this is never stored, never sent and not `Pod`, and would matter a
/// great deal the day any of those three stopped being true.
///
/// `#[repr(C)]` for consistency with every other enum at this boundary rather
/// than because anything needs it: this crosses no `extern "C"` signature and
/// sits in no `#[repr(C)]` struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
	/// This process is told where it went.
	///
	/// Everything a client can see that is not its own. Nothing local moves
	/// it; it is driven towards the place the last snapshot put it.
	SimulatedProxy,

	/// This process works out where it goes and expects to be corrected.
	///
	/// What a client holds over the things it owns. The prediction is the
	/// client's own move re-run over what the host has not acknowledged yet,
	/// and the correction is a decay rather than a snap.
	AutonomousProxy,

	/// This process decides where it goes, and everybody else is told.
	///
	/// What the host holds over everything in its world, a client's own player
	/// included: one machine simulates, and that is the whole model.
	Authority,
}

impl Role {
	/// Whether a thing in this role is worked out here rather than reported to
	/// here.
	///
	/// True for the two that decide and false for the one that is told. That
	/// is the branch anything driving a body towards a snapshot actually
	/// wants, because a whole class of code cares which side of the line a
	/// role falls on rather than which of the two it is.
	///
	/// @note: written as the two that decide rather than as "not the one that
	/// is told", which is the same answer today and a different one the day a
	/// fourth role is added. A role nobody has thought about yet should fail
	/// to this side of the line, and `matches!` is checked for exhaustiveness
	/// in neither direction.
	#[must_use]
	pub const fn local(self) -> bool { matches!(self, Self::AutonomousProxy | Self::Authority) }

	/// The part one peer plays in a thing another peer owns.
	///
	/// @param peer - who this process is, @ref
	/// [`World::peer`](crate::abi::World::peer)
	/// @param owner - who owns the thing, or [`PeerId::NONE`] for nobody
	/// @return what this process may do about it
	#[must_use]
	pub fn of(peer: PeerId, owner: PeerId) -> Self {
		if peer.is_host() {
			return Self::Authority;
		}

		// @note: both halves are load-bearing, and the first is deliberately
		// `is_some` rather than a comparison against `PeerId::NONE`. The
		// obvious case it stops is a client that has connected and has not
		// been told who it is: it holds nobody, and every unowned body in the
		// world - which is most of them - would otherwise answer that it owns
		// it. The case only `is_some` stops is a peer of any slot at all with
		// a generation of zero, which is a handle naming nobody at a slot
		// somebody could be sitting in, and every bit pattern is one of those.
		if peer.is_some() && peer == owner {
			return Self::AutonomousProxy;
		}

		Self::SimulatedProxy
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Some client, as the table handing slots out would mint one.
	const fn client(index: u32, generation: u32) -> PeerId { PeerId::at(index, generation) }

	#[test]
	fn nobody_is_the_zero_value_and_the_host_is_not() {
		// the bytes a game's own memory actually keeps a handle as, rather
		// than a constructor's opinion of them.
		let arena = [0_u8; 8];
		let read: PeerId = bytemuck::pod_read_unaligned(&arena);

		assert_eq!(size_of::<PeerId>(), size_of::<u64>(), "two words, like a BodyId");
		assert_eq!(PeerId::default(), PeerId::NONE);
		assert_eq!(read, PeerId::NONE, "a freshly zeroed arena reads as nobody");

		assert!(!PeerId::NONE.is_some());
		assert!(!PeerId::NONE.is_host(), "the powerless answer is the one a zero gives");
		assert!(PeerId::HOST.is_some());
		assert!(PeerId::HOST.is_host());
		assert_ne!(PeerId::HOST, PeerId::NONE);
	}

	#[test]
	fn a_slot_with_no_occupant_in_it_is_nobody_whatever_slot_it_is() {
		let empty = client(3, 0);

		assert!(!empty.is_some(), "a generation of zero is nobody at any slot at all");
		assert!(!empty.is_host());
		assert_eq!(
			Role::of(empty, empty),
			Role::SimulatedProxy,
			"so a handle naming nobody does not own itself"
		);
	}

	/// The trap this constant exists to avoid, written as a test.
	///
	/// Every table in this engine mints the first occupant of a slot at
	/// generation one, so `{0, 1}` is what a peer table in the house style
	/// hands the first client that connects. If that were the host, that
	/// client would be handed the world.
	#[test]
	fn what_a_table_would_mint_for_a_first_client_is_not_the_host() {
		let first = client(0, 1);

		assert!(!first.is_host(), "the host is at a generation nothing hands out");
		assert_ne!(first, PeerId::HOST);
		assert_eq!(
			Role::of(first, PeerId::NONE),
			Role::SimulatedProxy,
			"so the first peer to connect decides nothing until it is given something"
		);
		assert_eq!(PeerId::HOST.generation(), u32::MAX);
	}

	#[test]
	fn a_slot_alone_is_not_an_identity() {
		assert_eq!(PeerId::HOST.slot(), 0, "the host's is the first");
		assert_eq!(client(3, 1).slot(), 3, "and a client's is the one it was minted at");
		assert_eq!(client(3, 2).generation(), 2, "as is the occupant");

		let stranger = client(0, 2);

		assert_eq!(stranger.slot(), PeerId::HOST.slot(), "the same slot as the host");
		assert!(!stranger.is_host(), "and not the host");
		assert!(
			!client(7, u32::MAX).is_host(),
			"and neither is the host's generation somewhere else, so both halves are read"
		);
		assert_eq!(
			PeerId::NONE.slot(),
			PeerId::HOST.slot(),
			"and nobody reports the host's slot too, which is why a slot is read only once \
			 is_some has been asked"
		);
	}

	#[test]
	fn a_slot_that_changed_hands_is_not_the_peer_that_left_it() {
		let before = client(3, 1);
		let after = client(3, 2);

		assert_eq!(before.slot(), after.slot());
		assert_ne!(before, after, "which is the whole reason a generation is carried");
		assert_eq!(
			Role::of(after, before),
			Role::SimulatedProxy,
			"so a body the old peer owned does not change hands with the slot"
		);
	}

	#[test]
	fn a_host_is_the_authority_over_everything_it_can_see() {
		for owner in [PeerId::NONE, PeerId::HOST, client(1, 1), client(7, 4)] {
			assert_eq!(
				Role::of(PeerId::HOST, owner),
				Role::Authority,
				"including what a client owns"
			);
		}
	}

	#[test]
	fn a_client_works_out_its_own_and_is_told_the_rest() {
		let mine = client(1, 1);
		let theirs = client(2, 1);

		assert_eq!(Role::of(mine, mine), Role::AutonomousProxy);
		assert_eq!(Role::of(mine, theirs), Role::SimulatedProxy);
		assert_eq!(
			Role::of(mine, PeerId::NONE),
			Role::SimulatedProxy,
			"and a prop nobody owns is the host's business"
		);
		assert_eq!(
			Role::of(mine, PeerId::HOST),
			Role::SimulatedProxy,
			"as is anything the host holds itself"
		);
	}

	#[test]
	fn a_client_that_has_not_been_told_who_it_is_owns_nothing() {
		assert_eq!(
			Role::of(PeerId::NONE, PeerId::NONE),
			Role::SimulatedProxy,
			"nobody owning nothing is not a peer owning its own"
		);
		assert_eq!(Role::of(PeerId::NONE, client(1, 1)), Role::SimulatedProxy);
		assert_eq!(Role::of(PeerId::NONE, PeerId::HOST), Role::SimulatedProxy);
	}

	#[test]
	fn the_two_roles_that_decide_are_the_two_that_are_local() {
		assert!(Role::Authority.local());
		assert!(Role::AutonomousProxy.local());
		assert!(!Role::SimulatedProxy.local(), "and the one that is told is not");
	}
}
