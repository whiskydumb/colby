//! A handle as a value a program can keep.
//!
//! Every table in this engine is addressed by a slot and a generation, and
//! neither half means anything without the other: the slot says where, the
//! generation says which occupant, and a handle to something that has been
//! despawned is *detectably* stale rather than quietly somebody else's. What
//! crosses into Lua has to keep that property and three more.
//!
//! - **It has to be usable as a table key.** A program keeping "the props I am
//!   pushing" writes `mine[thing] = true`, and two values naming one body have
//!   to be one key. That rules out userdata, where two of them for the same
//!   body are two different keys.
//! - **It has to cost nothing to make.** A program that walks every body each
//!   step should not allocate once per body.
//! - **It has to say which table it belongs to.** Both tables here are dense
//!   and both start at slot nought generation one, so a body handle and an
//!   entity handle collide constantly. Untagged, `entity.name(a_body)` is a
//!   *wrong answer* rather than an error, which is the worst of the three
//!   outcomes.
//!
//! All four together give one answer: **a tagged integer**, which is what the
//! engine whose object ids survive as script values also settled on - it packs
//! a slot, a validator and one more bit into a single sixty-four bit number for
//! exactly these reasons.
//!
//! ```text
//!   bit 63      unused, so every handle is a positive Lua integer
//!   bits 56..62 which table this addresses, never nought
//!   bits 24..55 the generation
//!   bits  0..23 the slot
//! ```
//!
//! Two properties fall out of the layout and both are worth having. **A small
//! number is never a handle**: the tag is never nought, so anything a program
//! could have arrived at by counting is below the smallest handle there is and
//! is refused by name. And **a handle is positive**, so it prints and compares
//! the way a number is expected to.

use colby_core::abi::{BodyId, EntityId, MAX_BODIES, MAX_ENTITIES};

/// How many bits of the number the slot gets.
const INDEX_BITS: u32 = 24;

/// How many the generation gets, which is all of it.
const GENERATION_BITS: u32 = 32;

/// Where the generation starts.
const GENERATION_SHIFT: u32 = INDEX_BITS;

/// Where the tag starts.
const KIND_SHIFT: u32 = INDEX_BITS + GENERATION_BITS;

/// The largest slot the layout can carry.
const MAX_INDEX: usize = (1 << INDEX_BITS) - 1;

/// Everything below the tag.
const PAYLOAD: i64 = (1 << KIND_SHIFT) - 1;

// the two tables this addresses are bounded, and the bound is what says the
// layout is lossless. Written as a comparison of constants that are declared
// apart rather than as a restatement of one of them, so it can actually fail.
const _: () = assert!(MAX_ENTITIES <= MAX_INDEX, "an entity slot has to fit in the layout");
const _: () = assert!(MAX_BODIES <= MAX_INDEX, "and so does a body slot");
const _: () =
	assert!(KIND_SHIFT + Kind::BITS < i64::BITS, "and the tag has to fit under the sign bit");

/// Which table a handle addresses.
///
/// Never nought, which is what makes a number a program arrived at by counting
/// refusable rather than a handle to slot nought.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
	/// Something in the world that is drawn and moved.
	Entity = 1,

	/// Something the solver knows about.
	Body = 2,
}

impl Kind {
	/// How many bits the tag needs.
	const BITS: u32 = 7;

	/// What a program calls this table.
	pub(crate) const fn table(self) -> &'static str {
		match self {
			| Self::Entity => "entity",
			| Self::Body => "body",
		}
	}

	/// The tag a number carries, if it carries one this build knows.
	const fn of(bits: i64) -> Option<Self> {
		match bits >> KIND_SHIFT {
			| 1 => Some(Self::Entity),
			| 2 => Some(Self::Body),
			| _ => None,
		}
	}
}

/// One handle, as a program holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Handle {
	kind: Kind,
	index: u32,
	generation: u32,
}

impl Handle {
	/// Packs one up.
	#[expect(
		clippy::as_conversions,
		reason = "the build asserts above that every field fits inside the layout, so none of \
		          these widenings can lose anything"
	)]
	pub(crate) const fn to_bits(self) -> i64 {
		((self.kind as i64) << KIND_SHIFT)
			| ((self.generation as i64) << GENERATION_SHIFT)
			| (self.index as i64)
	}

	/// Reads one back, if that is what the number is.
	///
	/// @param bits - what a program handed over
	/// @return the handle, or `None` for a number carrying no tag this build
	/// knows - which is every number small enough for a program to have
	/// counted to
	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "taking each field back out is what this is for, and the masks are what make \
		          every one of them exact"
	)]
	pub(crate) const fn from_bits(bits: i64) -> Option<Self> {
		let Some(kind) = Kind::of(bits) else {
			return None;
		};
		let payload = bits & PAYLOAD;

		Some(Self {
			kind,
			index: (payload as u64 & MAX_INDEX as u64) as u32,
			generation: (payload >> GENERATION_SHIFT) as u32,
		})
	}

	/// Which table it addresses.
	pub(crate) const fn kind(self) -> Kind { self.kind }

	/// The entity it names, whether or not one is there.
	pub(crate) const fn entity(self) -> EntityId { EntityId::at(self.index, self.generation) }

	/// The body it names, whether or not one is there.
	pub(crate) const fn body(self) -> BodyId { BodyId::at(self.index, self.generation) }

	/// The handle for one entity.
	pub(crate) fn of_entity(id: EntityId) -> Self {
		Self {
			kind: Kind::Entity,
			index: narrowed(id.slot()),
			generation: id.generation(),
		}
	}

	/// The handle for one body.
	pub(crate) fn of_body(id: BodyId) -> Self {
		Self {
			kind: Kind::Body,
			index: narrowed(id.slot()),
			generation: id.generation(),
		}
	}
}

/// A slot as the layout carries it.
///
/// Both tables are bounded well below four billion, which the build asserts
/// above, so this can never actually narrow anything. Nought is the slot a dead
/// handle would name anyway.
fn narrowed(slot: usize) -> u32 { u32::try_from(slot).unwrap_or(0) }

impl std::fmt::Display for Handle {
	/// What `colby.describe` writes, and what a person reads in a log.
	fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(out, "{} {}:{}", self.kind.table(), self.index, self.generation)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A handle whose two numbers are two different numbers, neither of them
	/// one, and both with bits set high enough that a field swapped or a mask
	/// one bit short is visible.
	fn awkward(kind: Kind) -> Handle {
		Handle {
			kind,
			index: 0x00AB_CDEF & u32::try_from(MAX_INDEX).unwrap_or(u32::MAX),
			generation: 0xFEDC_BA98,
		}
	}

	#[test]
	fn a_handle_written_down_as_a_number_comes_back_itself() {
		for kind in [Kind::Entity, Kind::Body] {
			let handle = awkward(kind);
			let back = Handle::from_bits(handle.to_bits()).expect("it is a handle");

			assert_eq!(back, handle, "{kind:?} survived the round trip");
		}
	}

	#[test]
	fn a_handle_is_a_positive_number_however_high_its_generation_is() {
		// the sign bit is deliberately left out of the layout, so a handle
		// prints and compares the way a number is expected to rather than
		// arriving in Lua as a negative.
		let handle = Handle {
			kind: Kind::Body,
			index: u32::try_from(MAX_INDEX).unwrap_or(u32::MAX),
			generation: u32::MAX,
		};

		assert!(handle.to_bits() > 0, "and the highest one there is: {}", handle.to_bits());
	}

	#[test]
	fn a_number_a_program_could_have_counted_to_is_not_a_handle() {
		// the whole reason the tag is never nought. Without it a program
		// writing `body.name(1)` would be asking about slot one rather than
		// being told it is holding a number.
		for bits in [0, 1, 2, 1000, i64::from(u32::MAX)] {
			assert!(Handle::from_bits(bits).is_none(), "{bits} is not a handle");
		}

		assert!(Handle::from_bits(-1).is_none(), "and neither is a negative");
	}

	#[test]
	fn a_body_and_an_entity_in_the_same_slot_are_two_different_numbers() {
		// both tables are dense and both start at slot nought generation one,
		// so this pair happens constantly. Untagged they would be one number
		// and every lookup would answer about the wrong table.
		let entity = Handle::of_entity(EntityId::at(3, 1));
		let body = Handle::of_body(BodyId::at(3, 1));

		assert_ne!(entity.to_bits(), body.to_bits());
		assert_eq!(Handle::from_bits(entity.to_bits()).map(Handle::kind), Some(Kind::Entity));
		assert_eq!(Handle::from_bits(body.to_bits()).map(Handle::kind), Some(Kind::Body));
	}

	#[test]
	fn a_handle_resolves_to_the_slot_and_generation_it_was_made_from() {
		let id = EntityId::at(7, 12);
		let handle = Handle::of_entity(id);

		assert_eq!(handle.entity(), id, "the same entity");
		assert_eq!(handle.entity().slot(), 7);
		assert_eq!(handle.entity().generation(), 12);
	}

	#[test]
	fn a_handle_says_what_it_is_in_a_way_a_person_can_read() {
		assert_eq!(Handle::of_body(BodyId::at(12, 3)).to_string(), "body 12:3");
		assert_eq!(Handle::of_entity(EntityId::at(0, 1)).to_string(), "entity 0:1");
	}
}
