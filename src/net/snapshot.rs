//! What moved, said as briefly as saying it allows.
//!
//! A snapshot is a numbered set of records, one per slot of a table, and it is
//! written as a *difference* from a snapshot the far end has already got. What
//! did not move is not in it at all; what moved carries only the fields that
//! did.
//!
//! ```text
//!   Snapshot::write(number, against, &before, &now, &mut out)
//!   Snapshot::read(&before, bytes) -> the records, and how far it read
//! ```
//!
//! **A baseline is a delta against nothing**, not a second format. Writing a
//! thing for the first time is writing it against a record of all zeroes, so
//! every field that is not *already* zero is marked and sent - and a field that
//! is zero costs its bit and no more, because the far end is starting from
//! zeroes too. That is one code path instead of two, and it is what makes "the
//! first snapshot a client gets" and "the snapshot after a long silence" the
//! same thing as far as this module is concerned.
//!
//! The one thing a zeroed base cannot express is a slot that is *occupied* and
//! entirely zero, which differs from its base in nothing at all. That is why a
//! record is written for a slot the far end has never heard of even when it has
//! nothing to say: the record's existence is the news.
//!
//! ## The record, and why it is shaped like this
//!
//! Every field is exactly four bytes. That is the assumption the whole
//! mechanism rests on: a record is compared as an array of words, so the
//! comparison neither knows nor cares which of them are floats, and adding a
//! field is adding a word and a line to the table.
//!
//! The build checks two things about that, and it is worth being exact about
//! which: that [`Solid`] is [`WORDS`] words with no padding, and that
//! [`FIELDS`] names every word from nought upwards with no gap, repeat or swap.
//! Together those mean a field added to the struct and forgotten in the table
//! does not compile. What they do **not** check is that any entry's *name* has
//! anything to do with the word it sits on - the names are for a person reading
//! a table, and a wrong one is caught by nobody.
//!
//! **The table is sorted by how often a field changes**, and the sorting is
//! load-bearing rather than tidy. One byte says how far up the table the
//! changes reach, and only the fields *below* that carry a changed bit. A body
//! that is falling touches the first thirteen words and stops; its scale, its
//! kind, the entity it drives and the peer that owns it cost nothing at all.
//! Put a rarely-changing field early and every record pays for it forever.
//!
//! **Full `f32`, and no small-value shortcut.** The system this is modeled on
//! spends two bits on a field that is zero and sixteen on one that happens to
//! be a small whole number, and it wins on both because its game rounds what it
//! sends first - though not everywhere, and the exception is instructive: the
//! *player's own* position goes through the same codec and is snapped nowhere,
//! so the half of its traffic the shortcut was tuned for pays the full
//! thirty-five bits anyway. Nothing here rounds anything at all, so the same
//! shortcut would spend two extra bits on every changed field to buy a saving
//! that never applies. Rounding is quantization by another name, quantization
//! is deferred until a measured number hurts, and the two decisions want taking
//! together.
//!
//! **A field is compared as a word, never as a number**, and that is
//! load-bearing rather than lazy. Comparing floats would call an unchanged
//! `NaN` changed on every snapshot forever, dragging the reach up to whatever
//! field held it and every field below it along. Worse, it would call `-0.0`
//! and `0.0` equal and so *elide* the change - after which the two ends hold
//! different bits, and since every later difference is taken against the
//! writer's copy, they would never agree again. Exactness is what the whole
//! scheme stands on; the cost is four bytes when a sign flips over a zero.
//!
//! ## What is not here
//!
//! **The shape, the mass, the surface, and anything naming an asset.** A body's
//! shape names a mesh when it is a triangle soup, and a mesh handle is where an
//! asset landed in a registry this run. A saved scene writes such a handle as a
//! *name*, and its recorded reason is that the next run may fill the registry
//! in another order; a snapshot has that reason and a second one on top of it,
//! because the machine being told filled its own registry entirely separately.
//!
//! A name per record at twenty snapshots a second is not affordable, and the
//! answer is the one the distributed-ownership engine in the field uses: the
//! whole of what a thing *is* rides in the message that says it has appeared,
//! and its per-tick delta carries a transform and nothing else. So the shape is
//! not in this record, and the commit that makes a thing appear is where it
//! belongs.
//!
//! **Entities.** Thirty-seven of the sandbox's fifty-one bodies drive one, and
//! at the end of a step the two hold one transform between them - so sending
//! both would be about a third of a snapshot spent on saying it twice. Which of
//! the pair is the copy depends on the kind and is worth knowing before relying
//! on it: the solver writes the entity from a body it moves, and writes the
//! body from the entity for every body it does not, which in this sandbox is
//! thirty one of the thirty seven. What an entity has that a body does not is
//! what it looks like, and that is an asset name, which is the paragraph above.
//!
//! The entities that move and have *no* body are the character's two pieces.
//! Their transforms are walked along a beat by the game itself and their bones
//! are moved by a pose, and a pose is out of this step by agreement - so what a
//! remote viewer would need for them is the half that is deferred anyway.
//!
//! **Anything hostile is refused rather than absorbed.** A short block, a field
//! count past the end of the table, a slot past the end of a world: each is a
//! [`Fault`], and no input makes this module panic or reserve memory in
//! proportion to a number it was handed. The system this is modeled on reads
//! zeroes past the end of a message instead, so a truncated packet there
//! decodes as a well-formed record of nothing - which is the one part of its
//! design worth deliberately not copying.

use colby_core::bytemuck;

use crate::packet::{u16_at, u32_at};

/// How many bodies one snapshot can speak about.
///
/// The world's own ceiling, taken from it rather than restated beside it, so
/// the two cannot drift into disagreeing about what a slot number means.
pub const MAX_SLOTS: usize = colby_core::abi::MAX_BODIES;

/// The largest a written snapshot may be, in bytes.
///
/// Not [`MAX_MESSAGE`](crate::MAX_MESSAGE), and the difference matters. A
/// message carries the reliable ring *and* a snapshot, the ring may be sixty
/// four commands of a kilobyte each, and a channel that is handed a message too
/// long **refuses the whole thing** rather than trimming it. So a snapshot
/// sized against the message ceiling would, exactly when a peer has stopped
/// acknowledging and the ring has filled, throw the ring away with itself. This
/// is what is left over once the ring has all it can ask for, rounded down
/// hard.
///
/// **This is a ceiling on how many bodies can be described at once, and the
/// number is [`MAX_BASELINE`].** A world with more moving parts than that
/// cannot be sent in full at all: there is no continuation and no resume, so a
/// baseline either fits or is refused. That is not an oversight to be fixed
/// here - a world of hundreds of bodies is the stated trigger for deciding
/// *which* of them a given peer is told about, and until something decides
/// that, sending all of them is the only thing on offer.
pub const MAX_SNAPSHOT: usize = 8192;

/// How many bodies one snapshot can carry when every one of them is new.
///
/// The number [`MAX_SNAPSHOT`] really means, worked out rather than guessed:
/// a record nothing can be delta'd against carries its key, its reach, a full
/// mask and every word. Below this a baseline always fits; above it, a baseline
/// is impossible rather than merely large, which is why the number is worth
/// having a name and a test.
pub const MAX_BASELINE: usize = (MAX_SNAPSHOT - HEAD) / (KEY + 1 + WORDS.div_ceil(8) + WORDS * 4);

/// A slot that has emptied since the last snapshot, in place of a field count.
///
/// Out of the table's range by a long way. The reader checks for this *before*
/// it checks the range, so the thing keeping a grown table from reading a
/// removal as a record is the build assertion below and nothing else.
const GONE: u8 = u8::MAX;

/// What a snapshot number of nothing means: this is a delta against no
/// snapshot at all, which is to say a baseline.
pub const NOTHING: u32 = 0;

/// Bytes at the head of a snapshot, before the first record.
const HEAD: usize = 10;

/// The most the reliable ring can ask of one message, in bytes.
///
/// Every slot full of the longest command there is, **and the block's own
/// head**. Worked out here rather than guessed, because it is the other half of
/// what [`MAX_SNAPSHOT`] has to leave room for - and the head is easy to leave
/// out, which makes the sum ten bytes optimistic exactly when it matters.
#[expect(
	clippy::as_conversions,
	reason = "a const, where try_from is not available"
)]
const RING: usize =
	crate::reliable::HEAD + crate::MAX_COMMANDS as usize * (2 + crate::MAX_COMMAND);

/// Bytes of key at the head of every record: a slot and a generation.
const KEY: usize = 6;

/// One field of a record: what it is called, and which word it is.
///
/// A name because a table nobody can read is a table nobody checks, and it is
/// what a fault reports. There is deliberately no width: every field is one
/// four-byte word, which is what lets a record be compared as an array without
/// knowing what any of it means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field {
	/// What the field is called, for a person reading a fault.
	pub name: &'static str,

	/// Which word of the record it is.
	pub word: usize,
}

impl Field {
	/// One field of a record.
	const fn new(name: &'static str, word: usize) -> Self { Self { name, word } }
}

/// One body, as a snapshot holds it.
///
/// Only what moves. @ref the module comment for what is deliberately absent
/// and where each absent thing belongs instead.
///
/// `#[repr(C)]` of nothing but four-byte fields, so it is exactly
/// [`WORDS`] words long with no padding anywhere - which the build checks,
/// because the whole codec casts it to an array of words.
#[repr(C)]
#[derive(
	Clone,
	Copy,
	Debug,
	Default,
	PartialEq,
	colby_core::bytemuck::Pod,
	colby_core::bytemuck::Zeroable,
)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Solid {
	/// Where it is.
	pub position: [f32; 3],

	/// Which way it is turned, xyzw.
	pub rotation: [f32; 4],

	/// How fast it is moving, in units a second.
	pub velocity: [f32; 3],

	/// How fast it is turning, in radians a second.
	pub angular: [f32; 3],

	/// Whether the solver has stopped integrating it.
	pub sleeping: u32,

	/// How big it is.
	pub scale: [f32; 3],

	/// What the solver may do with it.
	///
	/// Not fixed once spawned, which is the thing a snapshot design would
	/// naturally assume and would be wrong about: freezing a prop turns it
	/// kinematic and letting it go turns it back, and a ragdoll does the same
	/// to thirteen bodies at once.
	pub kind: u32,

	/// The entity it drives, as a slot and a generation.
	pub entity: [u32; 2],

	/// The peer that owns it, as a slot and a generation.
	pub owner: [u32; 2],
}

/// How many four-byte words one [`Solid`] is.
pub const WORDS: usize = 22;

/// Every field of a [`Solid`], in the order they are sent.
///
/// **Sorted by how often each changes**, most often first, because one byte
/// says how far up this list a record's changes reach and everything past that
/// point is free. @ref the module comment.
pub const FIELDS: [Field; WORDS] = [
	Field::new("position.x", 0),
	Field::new("position.y", 1),
	Field::new("position.z", 2),
	Field::new("rotation.x", 3),
	Field::new("rotation.y", 4),
	Field::new("rotation.z", 5),
	Field::new("rotation.w", 6),
	Field::new("velocity.x", 7),
	Field::new("velocity.y", 8),
	Field::new("velocity.z", 9),
	Field::new("angular.x", 10),
	Field::new("angular.y", 11),
	Field::new("angular.z", 12),
	Field::new("sleeping", 13),
	Field::new("scale.x", 14),
	Field::new("scale.y", 15),
	Field::new("scale.z", 16),
	Field::new("kind", 17),
	Field::new("entity.slot", 18),
	Field::new("entity.generation", 19),
	Field::new("owner.slot", 20),
	Field::new("owner.generation", 21),
];

// the table has to cover the record exactly: every word once, in order. The
// system this is modeled on asserts only that the *count* matches, which lets
// a duplicated word and a missing one cancel each other out and pass. This
// checks the thing that actually has to be true, and it checks it in the build
// rather than in a test, because a table that has drifted is not a failing
// assertion somewhere - it is two machines quietly disagreeing about what byte
// nine means.
const _: () = assert!(
	size_of::<Solid>() == WORDS * 4,
	"a snapshot record is words and nothing but words"
);
const _: () = {
	let mut word = 0;

	while word < WORDS {
		assert!(FIELDS[word].word == word, "the field table has a gap, a repeat or a swap in it");
		word += 1;
	}
};
// widening rather than narrowing, which is the whole point: `WORDS as u8`
// truncates, so at two hundred and fifty six words it would compare 255 with 0
// and pass while a full record's reach did not fit in a byte at all.
#[expect(
	clippy::as_conversions,
	reason = "a const, where try_from is not available"
)]
const _: () = assert!(WORDS < GONE as usize, "the removal mark is out of the table's range");
// and the two ceilings have to fit in one message together, or a full ring of
// commands and a full snapshot are a message a channel refuses.
// no `+ HEAD`: `MAX_SNAPSHOT` is measured over the whole block, its own head
// included, because that is what `write` compares against.
const _: () = assert!(
	MAX_SNAPSHOT + RING < crate::MAX_MESSAGE,
	"a full ring and a full snapshot have to fit in one message"
);
const _: () = assert!(MAX_BASELINE > 0, "a snapshot has to be able to carry at least one body");

/// What was wrong with a snapshot somebody sent.
///
/// A small enumeration rather than an error carrying a sentence, for the reason
/// the datagram head has one: a peer able to reach the socket can send a bad
/// snapshot as fast as the wire allows, and building a message per bad one is a
/// way of letting them fill somebody's disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
	/// It ends in the middle of something.
	Short,

	/// It says it holds more records than a world has slots.
	TooMany,

	/// A record is about a slot no world has.
	NoSuchSlot,

	/// A record claims more fields than the table has.
	NoSuchField,

	/// It is longer than a snapshot is allowed to be.
	TooLong,

	/// Its records are not in slot order, or one slot is in it twice.
	OutOfOrder,
}

impl core::fmt::Display for Fault {
	/// A sentence, for the one thing a small enumeration still owes whoever is
	/// reading a log.
	fn fmt(&self, out: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		let said = match self {
			| Self::Short => "it ends in the middle of something",
			| Self::TooMany => "it claims more records than a world has slots",
			| Self::NoSuchSlot => "it is about a slot no world has",
			| Self::NoSuchField => "it claims more fields than the table has",
			| Self::TooLong => "it is longer than a snapshot may be",
			| Self::OutOfOrder => "its slots are out of order or repeated",
		};

		out.write_str(said)
	}
}

/// One slot's worth of a snapshot, as it was read back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Change {
	/// Which slot it is about.
	pub slot: u16,

	/// Which occupant of that slot, so that a slot changing hands is not read
	/// as the same thing moving.
	pub generation: u32,

	/// What it now holds, or nothing if the slot has emptied.
	pub solid: Option<Solid>,
}

/// A numbered set of changes, and the snapshot they are a difference from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
	/// Which snapshot this is.
	pub number: u32,

	/// Which one it is a difference from, or [`NOTHING`] for a baseline.
	pub against: u32,

	/// What changed, in slot order.
	pub changes: Vec<Change>,
}

impl Snapshot {
	/// Writes a snapshot as the difference between two sets of records.
	///
	/// Both are indexed by slot. A slot occupied in `now` and not in `before`
	/// is written in full, because a delta against nothing is a baseline; a
	/// slot occupied in `before` and not in `now` is written as gone; a slot in
	/// both is written as the fields that differ, and if none differ it is not
	/// written at all.
	///
	/// @param number - which snapshot this is
	/// @param against - which one it is a difference from, or [`NOTHING`]
	/// @param before - what the far end is known to have, by slot
	/// @param now - what is true, by slot
	/// @param out - where the bytes go, appended
	/// @return how many slots were spoken about
	///
	/// # Errors
	///
	/// If the result would be longer than [`MAX_SNAPSHOT`].
	pub fn write(
		number: u32,
		against: u32,
		before: &[Option<(u32, Solid)>],
		now: &[Option<(u32, Solid)>],
		out: &mut Vec<u8>,
	) -> Result<usize, Fault> {
		let began = out.len();
		let slots = before.len().max(now.len()).min(MAX_SLOTS);

		out.extend_from_slice(&number.to_le_bytes());
		out.extend_from_slice(&against.to_le_bytes());
		// filled in once the records are counted, because a count that has to
		// be right is better worked out than predicted.
		out.extend_from_slice(&0_u16.to_le_bytes());

		let mut spoken = 0;

		for slot in 0..slots {
			let was = before.get(slot).copied().flatten();
			let is = now.get(slot).copied().flatten();
			let Ok(index) = u16::try_from(slot) else {
				continue;
			};

			if slotted(index, was, is, out) {
				spoken += 1;
			}
		}

		if out.len() - began > MAX_SNAPSHOT {
			out.truncate(began);

			return Err(Fault::TooLong);
		}

		let count = u16::try_from(spoken).unwrap_or(u16::MAX);

		out[began + 8..began + HEAD].copy_from_slice(&count.to_le_bytes());

		Ok(spoken)
	}

	/// Reads a snapshot back, against what the reader already had.
	///
	/// @param before - what this end holds, by slot, for the fields a record
	/// does not carry
	/// @param bytes - the block, which may have anything after it
	/// @return the snapshot, and how many bytes it took
	///
	/// # Errors
	///
	/// @ref [`Fault`] - and every one of them is a peer that is broken or
	/// lying, so there is nothing to salvage from the rest of the message.
	pub fn read(before: &[Option<(u32, Solid)>], bytes: &[u8]) -> Result<(Self, usize), Fault> {
		if bytes.len() < HEAD {
			return Err(Fault::Short);
		}

		let number = u32_at(bytes, 0);
		let against = u32_at(bytes, 4);
		let count = usize::from(u16_at(bytes, 8));

		if count > MAX_SLOTS {
			return Err(Fault::TooMany);
		}

		// grown rather than reserved. `count` came off the wire, and a vector
		// sized from it is memory reserved in proportion to a number somebody
		// sent - which is the one thing this module undertakes not to do, and
		// a ceiling makes survivable rather than untrue. Ten bytes claiming a
		// thousand records would otherwise reserve a hundred kilobytes before
		// reading one of them.
		let mut changes: Vec<Change> = Vec::new();
		let mut at = HEAD;
		let mut last: Option<u16> = None;

		for _ in 0..count {
			let (change, next) = get(before, bytes, at)?;

			// ascending and each slot once, which is what `changes` claims to
			// be. A writer produces nothing else; a peer can, and anything
			// walking this beside its own table in one pass would be wrong.
			if last.is_some_and(|had| had >= change.slot) {
				return Err(Fault::OutOfOrder);
			}

			last = Some(change.slot);

			changes.push(change);
			at = next;

			// the writer's ceiling, enforced on the way in as well. Without it
			// a peer may send nine times what this module will ever write, and
			// the number everything else is sized against means nothing.
			if at > MAX_SNAPSHOT {
				return Err(Fault::TooLong);
			}
		}

		Ok((Self { number, against, changes }, at))
	}
}

/// Writes whatever one slot has to say, if it has anything.
///
/// @param slot - which slot
/// @param was - what the far end has for it, if anything
/// @param is - what is true of it, if anything
/// @param out - where the bytes go
/// @return whether the slot was spoken about
fn slotted(
	slot: u16,
	was: Option<(u32, Solid)>,
	is: Option<(u32, Solid)>,
	out: &mut Vec<u8>,
) -> bool {
	match (was, is) {
		| (None, None) => false,
		| (Some((generation, _)), None) => {
			out.extend_from_slice(&slot.to_le_bytes());
			out.extend_from_slice(&generation.to_le_bytes());
			out.push(GONE);

			true
		},
		| (was, Some((generation, solid))) => {
			// a slot that changed hands is written in full: the fields the last
			// occupant left are not a starting point for the next one, they are
			// somebody else's.
			let (base, fresh) = match was {
				| Some((held, solid)) if held == generation => (solid, false),
				| _ => (Solid::default(), true),
			};

			// and a slot the far end has never heard of is spoken about even
			// when every one of its fields is zero. Without this, a body that
			// appears at the origin holding nothing is indistinguishable from a
			// body that is not there - the occupancy and the generation would
			// both be lost, and the far end would never learn of it at all.
			put(slot, generation, &base, &solid, fresh, out)
		},
	}
}

/// Writes one record, or nothing at all when there is nothing to say.
///
/// @param fresh - whether the far end has never heard of this occupant, in
/// which case it is told even if every field is zero
/// @return whether anything was written
fn put(
	slot: u16,
	generation: u32,
	before: &Solid,
	now: &Solid,
	fresh: bool,
	out: &mut Vec<u8>,
) -> bool {
	let was: &[u32; WORDS] = bytemuck::cast_ref(before);
	let is: &[u32; WORDS] = bytemuck::cast_ref(now);

	// how far up the table the changes reach. Not how many changed: a change to
	// the last word alone costs a changed bit for every word before it, which
	// is the whole reason the table is sorted the way it is.
	// typed, rather than left to inference. An inferred integer here compiles
	// into whatever the first use demands, so a change to how this is worked
	// out can be caught by the type checker for an incidental reason and look
	// like a test having noticed.
	let mut reach: usize = 0;

	for word in 0..WORDS {
		if was[word] != is[word] {
			reach = word + 1;
		}
	}

	if reach == 0 && !fresh {
		return false;
	}

	let mask = reach.div_ceil(8);

	out.extend_from_slice(&slot.to_le_bytes());
	out.extend_from_slice(&generation.to_le_bytes());
	// the cast cannot lose anything: `reach` is at most WORDS, and the build
	// asserts WORDS is below the removal mark. The fallback is nought rather
	// than the mark, because a mark here would say "removed" and then keep
	// writing a mask and words the reader would take for the next record's
	// key - one overflow desynchronizing everything after it.
	out.push(u8::try_from(reach).unwrap_or(0));

	let head = out.len();

	out.resize(head + mask, 0);

	for word in 0..reach {
		if was[word] == is[word] {
			continue;
		}

		out[head + word / 8] |= 1 << (word % 8);
		out.extend_from_slice(&is[word].to_le_bytes());
	}

	true
}

/// Reads one record, against whatever the reader already had for that slot.
fn get(
	before: &[Option<(u32, Solid)>],
	bytes: &[u8],
	at: usize,
) -> Result<(Change, usize), Fault> {
	if at + KEY + 1 > bytes.len() {
		return Err(Fault::Short);
	}

	let slot = u16_at(bytes, at);
	let generation = u32_at(bytes, at + 2);
	let reach = bytes[at + KEY];

	if usize::from(slot) >= MAX_SLOTS {
		return Err(Fault::NoSuchSlot);
	}

	if reach == GONE {
		return Ok((Change { slot, generation, solid: None }, at + KEY + 1));
	}

	let reach = usize::from(reach);

	if reach > WORDS {
		return Err(Fault::NoSuchField);
	}

	// the same rule the writer follows: a slot that changed hands is read
	// against nothing, because what the last occupant left is not this one's.
	let held = before.get(usize::from(slot)).copied().flatten();
	let mut solid = match held {
		| Some((was, solid)) if was == generation => solid,
		| _ => Solid::default(),
	};

	let mask = reach.div_ceil(8);
	let mut read = at + KEY + 1;

	if read + mask > bytes.len() {
		return Err(Fault::Short);
	}

	let flags = &bytes[read..read + mask];

	read += mask;

	let words: &mut [u32; WORDS] = bytemuck::cast_mut(&mut solid);

	for word in 0..reach {
		if flags[word / 8] & (1 << (word % 8)) == 0 {
			continue;
		}

		if read + 4 > bytes.len() {
			return Err(Fault::Short);
		}

		words[word] = u32_at(bytes, read);
		read += 4;
	}

	Ok((Change { slot, generation, solid: Some(solid) }, read))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A body somewhere, with **every** word set to something a zero is not.
	///
	/// That matters more here than it looks. A baseline is a delta against
	/// zeroes, so a field whose fixture value happens to be zero is a field no
	/// baseline ever puts on the wire - and a reader that dropped it would give
	/// the right answer for the wrong reason. An earlier version of this
	/// fixture left the quaternion at the identity, `sleeping` at nought and
	/// the owner at nobody, which is six of the twenty two words, and five
	/// tests never reached past word twenty because of it.
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
	fn table(entries: &[(usize, u32, Solid)]) -> Vec<Option<(u32, Solid)>> {
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

	/// The far end: what it holds, and nothing of what the writer holds.
	///
	/// **A receiver with its own copy is the whole point of this type.** A test
	/// that reads back against the writer's own table proves only that a pair
	/// of functions agree with each other, which is exactly what a matched
	/// pair of bugs also does - and one such pair, the generation guard in
	/// both halves, passed an entire suite shaped that way.
	#[derive(Clone, Debug, Default, PartialEq)]
	struct Far {
		held: Vec<Option<(u32, Solid)>>,
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
		fn take(&mut self, bytes: &[u8]) -> Result<Snapshot, Fault> {
			let (snapshot, read) = Snapshot::read(&self.held, bytes)?;

			assert_eq!(read, bytes.len(), "a block with nothing after it reads to its end");

			for change in &snapshot.changes {
				self.put(change);
			}

			Ok(snapshot)
		}
	}

	/// Writes one snapshot from a receiver's own idea of the world, and gives
	/// the bytes back.
	fn tell(far: &Far, now: &[Option<(u32, Solid)>], out: &mut Vec<u8>) -> usize {
		Snapshot::write(1, NOTHING, &far.held, now, out).expect("it fits")
	}

	/// Whether two tables describe the same world, ignoring trailing holes.
	fn same(left: &[Option<(u32, Solid)>], right: &[Option<(u32, Solid)>]) -> bool {
		let reach = left.len().max(right.len());

		(0..reach)
			.all(|slot| left.get(slot).copied().flatten() == right.get(slot).copied().flatten())
	}

	#[test]
	fn a_baseline_is_a_delta_against_nothing() {
		let now = table(&[(0, 1, somewhere())]);
		let mut far = Far::default();
		let mut out = Vec::new();

		assert_eq!(tell(&far, &now, &mut out), 1, "one slot spoken about");

		let back = far.take(&out).expect("what was written reads");

		assert_eq!(back.number, 1);
		assert_eq!(back.against, NOTHING, "and it says what it is a difference from");
		assert_eq!(back.changes.len(), 1);
		assert_eq!(
			back.changes[0].solid,
			Some(somewhere()),
			"whole, because every word of it differs from the zeroes it is written against"
		);
		assert!(same(&far.held, &now), "and the far end now holds the world");
	}

	#[test]
	fn nothing_that_did_not_move_is_in_it() {
		let world = table(&[(0, 1, somewhere()), (4, 1, somewhere())]);
		let far = Far { held: world.clone() };
		let mut out = Vec::new();

		assert_eq!(tell(&far, &world, &mut out), 0, "two bodies, neither of them moved");
		assert_eq!(out.len(), HEAD, "so the snapshot is its own head and nothing else");
		assert_eq!(u16_at(&out, 8), 0, "with a count of none");
	}

	#[test]
	fn a_body_that_only_moved_pays_for_nothing_above_it() {
		let mut moved = somewhere();

		moved.position[0] += 1.0;

		let far = Far { held: table(&[(0, 1, somewhere())]) };
		let now = table(&[(0, 1, moved)]);
		let mut out = Vec::new();

		tell(&far, &now, &mut out);

		// the whole block: a head, a key, a reach of one, one mask byte and one
		// word. The other twenty one words are past the reach and cost nothing.
		assert_eq!(out.len(), HEAD + KEY + 1 + 1 + 4, "twenty two bytes for a step of one axis");
		// and the bytes themselves, not only how many there are. Nothing else
		// in this file would notice a mask a byte wide when it should be two,
		// or the bits written from the wrong end of a byte.
		assert_eq!(out[HEAD + KEY], 1, "a reach of one");
		assert_eq!(out[HEAD + KEY + 1], 0b0000_0001, "and the low bit of the first mask byte");
	}

	#[test]
	fn a_late_field_drags_every_field_below_it_along() {
		let mut claimed = somewhere();

		// exactly one word: the last of the table.
		claimed.owner[1] += 1;

		let far = Far { held: table(&[(0, 1, somewhere())]) };
		let now = table(&[(0, 1, claimed)]);
		let mut out = Vec::new();

		tell(&far, &now, &mut out);

		assert_eq!(out.len(), HEAD + KEY + 1 + 3 + 4, "three mask bytes for one changed word");
		assert_eq!(out[HEAD + KEY], 22, "the reach is the whole table");
		assert_eq!(
			&out[HEAD + KEY + 1..HEAD + KEY + 4],
			&[0b0000_0000, 0b0000_0000, 0b0010_0000],
			"and the only bit set is the twenty second, in the third byte"
		);
	}

	#[test]
	fn a_body_coming_to_rest_puts_zeroes_on_the_wire() {
		let mut resting = somewhere();

		// the commonest thing that happens in a world of falling props, and the
		// one a codec that skipped zero words would get right by accident.
		resting.velocity = [0.0; 3];
		resting.angular = [0.0; 3];

		let mut far = Far { held: table(&[(0, 1, somewhere())]) };
		let now = table(&[(0, 1, resting)]);
		let mut out = Vec::new();

		tell(&far, &now, &mut out);
		far.take(&out).expect("it reads");

		assert!(same(&far.held, &now), "a body at rest is at rest on both sides");
		assert_eq!(
			far.held[0].map(|(_, solid)| solid.velocity),
			Some([0.0; 3]),
			"which means the zeroes were sent rather than left to be inferred"
		);
	}

	#[test]
	fn a_slot_that_emptied_is_said_rather_than_left_out() {
		let mut far = Far { held: table(&[(2, 1, somewhere())]) };
		let mut out = Vec::new();

		tell(&far, &[], &mut out);

		let back = far.take(&out).expect("it reads");

		assert_eq!(out.len(), HEAD + KEY + 1, "a removal is a key and a mark");
		assert_eq!(back.changes.len(), 1);
		assert_eq!(back.changes[0].slot, 2, "and it is about the slot it was in");
		assert_eq!(back.changes[0].generation, 1, "and says which occupant went");
		assert_eq!(back.changes[0].solid, None);
		assert!(far.held[2].is_none(), "so the far end lets go of it");
	}

	#[test]
	fn a_slot_that_changed_hands_is_written_and_read_against_nothing() {
		let mut moved = somewhere();

		moved.position[0] += 1.0;

		// the same slot, a different occupant, and only one word differs from
		// what the last one left. Written against that, twenty one words would
		// be inherited from somebody else - and the *value* would come out
		// right, which is why this asserts the byte count as well.
		let mut far = Far { held: table(&[(0, 1, somewhere())]) };
		let now = table(&[(0, 2, moved)]);
		let mut out = Vec::new();

		tell(&far, &now, &mut out);

		assert_eq!(
			out.len(),
			HEAD + KEY + 1 + 3 + WORDS * 4,
			"every word of the new occupant went on the wire, not one"
		);

		let back = far.take(&out).expect("it reads");

		assert_eq!(back.changes[0].generation, 2);
		assert!(same(&far.held, &now));
	}

	/// The reader's own guard, which a matched writer can never exercise.
	///
	/// A writer that knows the slot changed hands sends the record in full, so
	/// every word is overwritten and the reader's base does not matter. What
	/// the guard is for is a peer that is broken or lying: a *partial* record
	/// carrying a generation the reader does not hold. Without it the missing
	/// words are inherited from whoever had the slot before, which is the one
	/// thing a generation exists to stop.
	///
	/// Built by hand for that reason - this is a block no honest writer here
	/// produces.
	#[test]
	fn a_partial_record_for_an_occupant_the_reader_never_had_inherits_nothing() {
		let mut moved = somewhere();

		moved.position[0] += 1.0;

		let far = Far { held: table(&[(0, 1, somewhere())]) };
		let mut out = Vec::new();

		tell(&far, &table(&[(0, 1, moved)]), &mut out);
		assert_eq!(out.len(), HEAD + KEY + 1 + 1 + 4, "one changed word, as a delta");

		// and now the lie: the same partial record, relabeled as a different
		// occupant of the same slot.
		out[HEAD + 2..HEAD + KEY].copy_from_slice(&2_u32.to_le_bytes());

		let (back, _) = Snapshot::read(&far.held, &out).expect("well formed, only untrue");
		let bare = Solid {
			position: [moved.position[0], 0.0, 0.0],
			..Solid::default()
		};

		assert_eq!(back.changes[0].generation, 2);
		assert_eq!(
			back.changes[0].solid,
			Some(bare),
			"the one word it sent, and zeroes for the rest - not the last occupant's"
		);
	}

	#[test]
	fn a_body_that_appears_holding_nothing_is_still_announced() {
		// every word zero, which is what a body spawned at the origin with no
		// velocity and no owner looks like. Written against zeroes it differs
		// in nothing at all, so the only thing that says it is there is the
		// record's own existence.
		let mut far = Far::default();
		let now = table(&[(0, 4, Solid::default())]);
		let mut out = Vec::new();

		assert_eq!(tell(&far, &now, &mut out), 1, "it is spoken about");
		assert_eq!(out.len(), HEAD + KEY + 1, "as a key and a reach of nothing");
		assert_eq!(out[HEAD + KEY], 0, "which is not the removal mark");

		let back = far.take(&out).expect("it reads");

		assert_eq!(back.changes[0].solid, Some(Solid::default()));
		assert_eq!(back.changes[0].generation, 4, "and the far end learns whose slot it is");
		assert!(same(&far.held, &now));
	}

	#[test]
	fn several_slots_in_one_snapshot_each_keep_their_own() {
		let mut second = somewhere();
		let mut third = somewhere();

		second.position = [9.0, 9.0, 9.0];
		third.kind = 7;

		// three slots, none of them slot zero for the changed ones, one
		// unchanged in the middle, and a removal at the end.
		let mut far = Far {
			held: table(&[
				(1, 1, somewhere()),
				(2, 1, somewhere()),
				(5, 1, somewhere()),
				(7, 1, somewhere()),
			]),
		};
		let now = table(&[(1, 1, second), (2, 1, somewhere()), (5, 1, third)]);
		let mut out = Vec::new();

		assert_eq!(tell(&far, &now, &mut out), 3, "two changed and one gone");

		let back = far.take(&out).expect("it reads");

		assert_eq!(back.changes.len(), 3);
		assert_eq!(
			back.changes
				.iter()
				.map(|change| change.slot)
				.collect::<Vec<u16>>(),
			vec![1, 5, 7],
			"in slot order, and slot two is not in it because it did not move"
		);
		assert!(same(&far.held, &now), "and the far end holds exactly what is true");
	}

	#[test]
	fn a_world_kept_up_over_many_snapshots_stays_the_world() {
		// the test the rest of this file is a special case of: a receiver with
		// its own state, told only differences, over a run long enough that a
		// single dropped or wrongly scoped field compounds into a world that has
		// drifted. Bodies appear, move, come to rest, change hands, wake, are
		// disowned and are removed.
		let mut world: Vec<Option<(u32, Solid)>> = vec![None; 12];
		let mut far = Far::default();
		let mut random = crate::Random::new(9);

		for step in 0..120_u32 {
			let slot = usize::try_from(random.below(12)).unwrap_or(0);

			happen(&mut world[slot], random.below(6), step, &mut random);

			let mut out = Vec::new();

			Snapshot::write(step + 1, step, &far.held, &world, &mut out).expect("it fits");
			far.take(&out).expect("what was written reads");

			assert!(
				same(&far.held, &world),
				"the far end has drifted from the world at step {step}"
			);
		}
	}

	/// One thing happening to one slot, in the long run below.
	///
	/// A free function rather than six arms nested inside a loop, which is what
	/// the nesting rule here wants and what makes each of the six readable.
	fn happen(slot: &mut Option<(u32, Solid)>, what: u64, step: u32, random: &mut crate::Random) {
		if what == 0 {
			// somebody arrives, sometimes holding nothing at all
			let generation = slot.map_or(1, |(had, _)| had + 1);
			let bare = random.below(4) == 0;
			let solid = if bare {
				Solid::default()
			} else {
				Solid { kind: step, ..somewhere() }
			};

			*slot = Some((generation, solid));

			return;
		}

		if what == 1 {
			*slot = None;

			return;
		}

		let Some((_, solid)) = slot.as_mut() else {
			return;
		};

		match what {
			| 2 => {
				solid.position[0] += 0.5;
				solid.velocity = [1.0, 2.0, 3.0];
				solid.sleeping = 0;
			},
			// coming to rest: the zeroes case, every time
			| 3 => {
				solid.velocity = [0.0; 3];
				solid.angular = [0.0; 3];
				solid.sleeping = 1;
			},
			| 4 => solid.owner = [u32::try_from(random.below(9)).unwrap_or(0), 1],
			| _ => solid.owner = [0, 0],
		}
	}

	#[test]
	fn every_field_survives_the_way_out_and_back() {
		let full = Solid {
			position: [-1.5, 0.25, 1.0e9],
			rotation: [0.5, -0.5, 0.5, -0.5],
			velocity: [f32::MIN, 0.125, f32::MAX],
			angular: [1.0e-9, -2.0, 3.0],
			sleeping: 1,
			scale: [2.0, 3.0, 4.0],
			kind: 1,
			entity: [7, 9],
			owner: [1, u32::MAX],
		};
		let mut far = Far::default();
		let now = table(&[(0, 1, full)]);
		let mut out = Vec::new();

		tell(&far, &now, &mut out);

		assert_eq!(
			out.len(),
			HEAD + KEY + 1 + 3 + WORDS * 4,
			"all twenty two words, none of them equal to the zero they are written against"
		);
		far.take(&out).expect("it reads");
		assert!(same(&far.held, &now), "including the ones a reader would round");
	}

	#[test]
	fn a_field_compares_as_bytes_rather_than_as_a_number() {
		// two values that are equal as floats and different as words. Comparing
		// as numbers would elide this, and the two ends would then disagree
		// forever, because every later difference is taken against the writer's
		// copy rather than the reader's.
		let mut signed = somewhere();

		signed.position[0] = -0.0;

		let mut zero = signed;

		zero.position[0] = 0.0;

		let mut far = Far { held: table(&[(0, 1, signed)]) };
		let now = table(&[(0, 1, zero)]);
		let mut out = Vec::new();

		assert_eq!(tell(&far, &now, &mut out), 1, "a change of sign is a change");
		far.take(&out).expect("it reads");
		assert_eq!(
			far.held[0].map(|(_, solid)| solid.position[0].is_sign_negative()),
			Some(false),
			"and the far end has the sign the writer has"
		);
	}

	#[test]
	fn the_table_names_every_word_of_the_record_once() {
		let mut names: Vec<&str> = FIELDS.iter().map(|field| field.name).collect();

		names.sort_unstable();
		names.dedup();

		assert_eq!(names.len(), WORDS, "no two fields share a name");
	}

	#[test]
	fn a_block_cut_short_anywhere_is_refused_rather_than_read_as_zeroes() {
		// two records and a removal between them, so that a cut lands inside a
		// key, inside a mask, inside a word, and between two records - which a
		// snapshot of one record cannot test at all.
		let far = Far {
			held: table(&[(0, 1, somewhere()), (3, 1, somewhere()), (6, 1, somewhere())]),
		};
		let mut moved = somewhere();

		moved.position[0] += 1.0;

		let mut out = Vec::new();

		tell(&far, &table(&[(0, 1, moved), (6, 2, somewhere())]), &mut out);
		assert!(out.len() > HEAD + 3 * (KEY + 1), "three records, one of them a removal");

		for cut in 0..out.len() {
			assert_eq!(
				Snapshot::read(&far.held, &out[..cut]).map(|(back, _)| back.changes.len()),
				Err(Fault::Short),
				"a block cut to {cut} of {} bytes",
				out.len()
			);
		}

		assert!(Snapshot::read(&far.held, &out).is_ok(), "and the whole of it still reads");
	}

	#[test]
	fn a_record_naming_a_field_the_table_does_not_have_is_refused() {
		let far = Far { held: Vec::new() };
		let mut out = Vec::new();

		tell(&far, &table(&[(0, 1, somewhere())]), &mut out);
		out[HEAD + KEY] = u8::try_from(WORDS + 1).unwrap_or(GONE - 1);

		assert_eq!(Snapshot::read(&far.held, &out), Err(Fault::NoSuchField));
	}

	#[test]
	fn a_record_about_a_slot_no_world_has_is_refused_and_the_last_one_is_not() {
		let far = Far { held: Vec::new() };
		let mut out = Vec::new();

		tell(&far, &table(&[(0, 1, somewhere())]), &mut out);

		let last = u16::try_from(MAX_SLOTS - 1).unwrap_or(0);

		out[HEAD..HEAD + 2].copy_from_slice(&last.to_le_bytes());
		assert!(
			Snapshot::read(&far.held, &out).is_ok(),
			"the last slot a world has is a slot a snapshot may name"
		);

		out[HEAD..HEAD + 2].copy_from_slice(
			&u16::try_from(MAX_SLOTS)
				.unwrap_or(0)
				.to_le_bytes(),
		);
		assert_eq!(
			Snapshot::read(&far.held, &out),
			Err(Fault::NoSuchSlot),
			"and one past it is not"
		);
	}

	#[test]
	fn a_snapshot_claiming_more_records_than_a_world_has_slots_is_refused() {
		let far = Far { held: Vec::new() };
		let mut out = Vec::new();

		tell(&far, &[], &mut out);

		out[8..HEAD].copy_from_slice(
			&u16::try_from(MAX_SLOTS)
				.unwrap_or(0)
				.to_le_bytes(),
		);
		assert_eq!(
			Snapshot::read(&far.held, &out),
			Err(Fault::Short),
			"a count of exactly the ceiling is allowed, and then runs out of block"
		);

		out[8..HEAD].copy_from_slice(
			&u16::try_from(MAX_SLOTS + 1)
				.unwrap_or(0)
				.to_le_bytes(),
		);
		assert_eq!(Snapshot::read(&far.held, &out), Err(Fault::TooMany), "one past it is not");
	}

	#[test]
	fn records_out_of_order_or_repeated_are_refused() {
		let mut first = somewhere();

		first.position[0] += 1.0;

		let far = Far {
			held: table(&[(1, 1, somewhere()), (2, 1, somewhere())]),
		};
		let mut out = Vec::new();

		tell(&far, &table(&[(1, 1, first), (2, 1, first)]), &mut out);

		let record = KEY + 1 + 1 + 4;

		assert_eq!(out.len(), HEAD + 2 * record, "two records of the same shape");

		// the second record relabeled as the first's slot: a peer saying one
		// slot twice, which a writer never does and a one-pass consumer would
		// be wrong about.
		let (a, b) = (HEAD, HEAD + record);

		out[b..b + 2].copy_from_slice(&1_u16.to_le_bytes());
		assert_eq!(
			Snapshot::read(&far.held, &out),
			Err(Fault::OutOfOrder),
			"the same slot twice"
		);

		out[a..a + 2].copy_from_slice(&2_u16.to_le_bytes());
		assert_eq!(Snapshot::read(&far.held, &out), Err(Fault::OutOfOrder), "and backwards");
	}

	#[test]
	fn a_snapshot_too_long_to_send_is_refused_rather_than_cut() {
		let full: Vec<Option<(u32, Solid)>> = (0..=MAX_BASELINE)
			.map(|slot| {
				Some((1, Solid {
					kind: u32::try_from(slot).unwrap_or(0),
					..somewhere()
				}))
			})
			.collect();
		let mut out = vec![0xAB; 3];

		assert_eq!(
			Snapshot::write(1, NOTHING, &[], &full, &mut out),
			Err(Fault::TooLong),
			"one body more than a baseline holds"
		);
		assert_eq!(out, vec![0xAB; 3], "and what was already in the buffer is untouched");

		// and the one below it goes through, into the same non-empty buffer,
		// appended rather than written over.
		assert_eq!(
			Snapshot::write(1, NOTHING, &[], &full[..MAX_BASELINE], &mut out),
			Ok(MAX_BASELINE),
			"exactly a baseline's worth fits"
		);
		assert_eq!(&out[..3], &[0xAB; 3], "after what the caller already had");
		assert!(out.len() - 3 <= MAX_SNAPSHOT);
	}

	#[test]
	fn a_snapshot_longer_than_may_be_written_is_refused_on_the_way_in_too() {
		// a peer is not bound by the writer's politeness, so the reader has the
		// same ceiling. Without it a message may carry nine times what this
		// module will ever produce, and the number the rest is sized against
		// means nothing.
		let far = Far { held: Vec::new() };
		let big: Vec<Option<(u32, Solid)>> = (0..MAX_BASELINE)
			.map(|slot| {
				Some((1, Solid {
					kind: u32::try_from(slot).unwrap_or(0),
					..somewhere()
				}))
			})
			.collect();
		// a second baseline over the slots *above* the first, so that the two
		// spliced together are still in order and still every slot once - only
		// twice as long as a snapshot may be.
		let mut above: Vec<Option<(u32, Solid)>> = vec![None; MAX_BASELINE];

		above.extend(big.iter().copied());

		let mut out = Vec::new();
		let mut rest = Vec::new();

		tell(&far, &big, &mut out);
		tell(&far, &above, &mut rest);

		let count = u16::try_from(MAX_BASELINE * 2).unwrap_or(u16::MAX);

		out.extend_from_slice(&rest[HEAD..]);
		out[8..HEAD].copy_from_slice(&count.to_le_bytes());

		assert!(out.len() > MAX_SNAPSHOT, "the block really is past the ceiling");
		assert_eq!(Snapshot::read(&far.held, &out), Err(Fault::TooLong));
	}

	#[test]
	fn what_follows_a_snapshot_is_left_where_it_is() {
		let far = Far { held: Vec::new() };
		let mut out = Vec::new();

		tell(&far, &table(&[(0, 1, somewhere())]), &mut out);

		let written = out.len();
		let tail: &[u8] = b"and then something else entirely";

		out.extend_from_slice(tail);

		let (back, read) = Snapshot::read(&far.held, &out).expect("it reads");

		assert_eq!(read, written, "so a reader knows where the next thing begins");
		assert_eq!(&out[read..], tail, "and what is there is untouched");
		assert_eq!(
			back.changes[0].solid,
			Some(somewhere()),
			"and the record still reads correctly with something behind it"
		);
	}

	#[test]
	fn a_fault_says_what_it_was_in_words() {
		let said = [
			Fault::Short,
			Fault::TooMany,
			Fault::NoSuchSlot,
			Fault::NoSuchField,
			Fault::TooLong,
			Fault::OutOfOrder,
		]
		.map(|fault| fault.to_string());
		let mut sorted = said.to_vec();

		sorted.sort();
		sorted.dedup();

		assert_eq!(sorted.len(), said.len(), "every fault says something of its own");
		assert!(said.iter().all(|line| !line.is_empty()));
	}

	#[test]
	fn a_baseline_holds_the_number_of_bodies_it_says_it_does() {
		// the ceiling is a real limit on how big a world may be, so the number
		// is written down and checked rather than left to be discovered by a
		// world that quietly stopped being sendable.
		let record = KEY + 1 + WORDS.div_ceil(8) + WORDS * 4;

		assert_eq!(MAX_BASELINE, (MAX_SNAPSHOT - HEAD) / record);
		assert!(MAX_BASELINE >= 51, "the sandbox has fifty one bodies and has to fit");
		assert!(
			MAX_BASELINE < MAX_SLOTS,
			"and a world of every slot does not, which is what interest management is for"
		);
	}
}
