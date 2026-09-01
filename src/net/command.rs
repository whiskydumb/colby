//! What a client is asking for, and what the host has done about it.
//!
//! The other direction from [`snapshot`](crate::snapshot), and deliberately a
//! much smaller thing. A snapshot is a description of a world and is written as
//! a difference because a whole one is eight kilobytes; a block of commands is
//! at most a ring's worth of twenty-four-byte records and is written out whole,
//! every time, because that is the *point* - the redundancy is the error
//! correction.
//!
//! ```text
//!   Block::write(yours, settled, &commands, &mut out)
//!   Block::read(bytes, &mut commands) -> what the head said, and how far it read
//! ```
//!
//! **Every message carries one**, the way every message carries a snapshot
//! block, and for the same reason: the head is news even when the body is
//! empty. A host with nothing to ask still has to say which of a client's
//! commands it has run, or the client can never let go of one; and a client
//! that has asked for nothing still has to be told who it is.
//!
//! ## The head, and why each field is in it
//!
//! Fourteen bytes: eight for a peer, four for a number, two for a count.
//!
//! **`yours`** is who the *far end* is, as this end knows it, and it is the
//! whole of the naming there is. A host mints a [`PeerId`] for everybody who
//! turns up and says it here in every message; a client reads it and stops
//! being nobody. It is repeated rather than sent once because **nothing
//! acknowledges it**: the two blocks either side of this one are each answered
//! by a number coming back the other way - the reliable ring by what has been
//! run, a snapshot by what is held - and a client has no field to answer a
//! name with. So a name sent once is a name lost with one datagram, and a name
//! sent sixty times a second costs eight bytes and cannot be. A client writes
//! [`PeerId::NONE`] here, because a client names nobody.
//!
//! **`settled`** is the highest of the *far end's* command numbers this end is
//! done with, so it is read the same way in both directions and written by
//! only one of them: a host writes what it has run of its client's commands,
//! and a client writes nought, because a host asks a client for nothing. It is
//! the same field [`Commands::settle`](colby_core::abi::Commands::settle) is,
//! read from the two sides, and it is the only thing that lets a sender stop
//! sending a command. It is not the only thing that makes one go: a ring drops
//! its oldest when it fills and throws everything away on a number far enough
//! out, neither of which anybody is told about.
//!
//! **`count`** is how many records follow, and it is on the wire rather than
//! implied by the length left over for the reason the datagram's piece count
//! is: a block whose length says where it ends cannot be followed by anything,
//! and a snapshot block follows this one.
//!
//! ## What is not here
//!
//! **No acknowledgement of the acknowledgement.** A host says what it has
//! settled and never learns whether the client heard, which is right: the
//! client's own ring drops what is settled, and a settled number that arrives
//! late is refused for being old rather than acted on twice.
//!
//! **No compression and no delta.** A command is twenty-four bytes and at most
//! a ring's worth cross at once, so the whole block is at most seven hundred
//! and eighty-two bytes against a payload of eleven hundred and eighty-four.
//! Delta-coding a stream whose whole purpose is to be sent several times over
//! would be spending the saving twice.
//!
//! **Nothing is authenticated**, as everywhere else in this crate. A stranger
//! who can reach the socket can ask for anything a peer can ask for, and
//! nothing in this module pretends otherwise. What it does undertake is that
//! no block, however malformed, can make it panic or reserve memory in
//! proportion to a number it was handed.

use colby_core::abi::{Command, PeerId, net::BACKUP};

use crate::packet::{u16_at, u32_at, u64_at};

/// How many bytes a block's head takes.
const HEAD: usize = 14;

/// How many bytes one command takes on the wire.
///
/// The struct's own size, and the same five fields in the same order - but
/// written a field at a time, little-endian, rather than copied whole. A copy
/// would be native-endian and would be a different block on a machine nobody
/// here has tried. @ref [`crate::packet`], which says the same of the head.
const RECORD: usize = 24;

/// How many commands one block may carry.
///
/// A ring's worth, which is every command a far end could still be waiting to
/// have settled. Sending more than that is sending something the far end has
/// already forgotten it asked for.
pub const MAX_ASKED: usize = BACKUP;

/// The largest a block may be, in bytes.
pub const MAX_BLOCK: usize = HEAD + MAX_ASKED * RECORD;

// @note: that this fits beside a full snapshot and a full reliable ring is
// asserted in `snapshot`, where the ring's own size is worked out. All three
// go in one message and a channel refuses a message too long whole, so the
// three ceilings are one question and are checked in one place.

/// What was wrong with a block somebody sent.
///
/// A small enumeration rather than an error carrying a sentence, for
/// [`Fault`](crate::snapshot::Fault)'s reason: a peer able to reach the socket
/// can send a bad block as fast as the wire allows, and building a message per
/// bad one is a way of letting them fill somebody's disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
	/// It ends in the middle of something.
	Short,

	/// It says it holds more commands than a ring can want.
	TooMany,

	/// Two of its commands are out of order, or one is in it twice.
	OutOfOrder,

	/// One of its commands is numbered nothing, which is not a command.
	Nothing,
}

/// What a block's head said.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Told {
	/// Who the far end says this end is, or [`PeerId::NONE`] from a client.
	pub yours: PeerId,

	/// The highest of this end's command numbers the far end is done with.
	pub settled: u32,

	/// How many bytes the block took.
	pub used: usize,
}

/// One block of commands, written and read.
///
/// A unit struct rather than a value, exactly as
/// [`Snapshot`](crate::snapshot::Snapshot)'s writing half is: there is no state
/// between one block and the next, and the ring that *has* the state is
/// [`Commands`](colby_core::abi::Commands) in the world.
#[derive(Clone, Copy, Debug)]
pub struct Block;

impl Block {
	/// Writes a block onto the end of a message.
	///
	/// Never fails. Anything past [`MAX_ASKED`] is dropped from the *front*,
	/// which is the same choice the ring itself makes and for the same reason:
	/// a far end that has fallen this far behind has lost the old ones
	/// whatever happens, and dropping the new ones would lose the present as
	/// well as the past.
	///
	/// @param yours - who the far end is, or [`PeerId::NONE`]
	/// @param settled - the highest of their numbers this end is done with
	/// @param commands - what to ask for, oldest first
	/// @param out - the message so far, appended
	/// @return how many commands went in
	pub fn write(yours: PeerId, settled: u32, commands: &[Command], out: &mut Vec<u8>) -> usize {
		// indexed rather than `get`, because `saturating_sub` has already made
		// the start no larger than the length and a fallback here would be a
		// branch nothing can reach.
		let asking = &commands[commands.len().saturating_sub(MAX_ASKED)..];
		// and `expect` rather than a fallback for a sharper reason: the count
		// and the body have to agree, and there is no wrong count that is safe
		// to write - a head claiming more records than follow it is a block
		// the far end reads as short, and one claiming fewer is a snapshot
		// read from the wrong offset.
		//
		// @note: the slice above is already no longer than [`MAX_ASKED`], so
		// this cannot fire and swapping it for any fallback at all survives
		// the whole suite. It is here to say which invariant is holding the
		// count and the body together, and the day that invariant goes this
		// is the line that stops a malformed block rather than writing one.
		let count = u16::try_from(asking.len()).expect("at most a ring's worth is ever asked");

		out.extend_from_slice(&yours.to_bits().to_le_bytes());
		out.extend_from_slice(&settled.to_le_bytes());
		out.extend_from_slice(&count.to_le_bytes());

		for command in asking {
			out.extend_from_slice(&command.step.to_le_bytes());
			out.extend_from_slice(&command.number.to_le_bytes());
			out.extend_from_slice(&command.buttons.to_le_bytes());
			out.extend_from_slice(&command.yaw.to_bits().to_le_bytes());
			out.extend_from_slice(&command.pitch.to_bits().to_le_bytes());
		}

		asking.len()
	}

	/// Reads a block off the front of an arriving message.
	///
	/// **Refused whole or taken whole.** A block whose fifth command is
	/// nonsense is not four commands and a fault: the numbers are what order
	/// them and what a receiver settles by, so half a block would be a peer
	/// whose next real command reads as one it has already had. `out` is left
	/// empty when this refuses.
	///
	/// The order is checked here rather than left to the ring, and the two
	/// checks are not the same one. The ring refuses a number that is not
	/// above what it *holds*, which is about the conversation; this refuses a
	/// block that is not in order *within itself*, which is about the block.
	/// A block out of its own order would have the ring take its first command
	/// and silently drop the rest, and the peer would look as though the wire
	/// had eaten them.
	///
	/// @param bytes - the block, and whatever follows it
	/// @param out - where the commands go, emptied first
	/// @return what the head said, or why the block is not one
	///
	/// # Errors
	///
	/// [`Fault`] for a block too short for what it claims, one claiming more
	/// commands than a ring can want, one whose numbers do not increase, and
	/// one carrying a command numbered nothing.
	pub fn read(bytes: &[u8], out: &mut Vec<Command>) -> Result<Told, Fault> {
		out.clear();

		if bytes.len() < HEAD {
			return Err(Fault::Short);
		}

		let yours = PeerId::from_bits(u64_at(bytes, 0));
		let settled = u32_at(bytes, 8);
		let count = usize::from(u16_at(bytes, 12));

		// before anything is reserved, and against the ring's own depth rather
		// than against what is left in the message: a count is a number the
		// far end chose.
		if count > MAX_ASKED {
			return Err(Fault::TooMany);
		}

		let used = HEAD + count * RECORD;

		if bytes.len() < used {
			return Err(Fault::Short);
		}

		let mut newest = 0;

		for index in 0..count {
			let at = HEAD + index * RECORD;
			let command = Command {
				step: u64_at(bytes, at),
				number: u32_at(bytes, at + 8),
				buttons: u32_at(bytes, at + 12),
				yaw: f32::from_bits(u32_at(bytes, at + 16)),
				pitch: f32::from_bits(u32_at(bytes, at + 20)),
			};

			if !command.is_some() {
				out.clear();

				return Err(Fault::Nothing);
			}

			if command.number <= newest {
				out.clear();

				return Err(Fault::OutOfOrder);
			}

			newest = command.number;
			out.push(command);
		}

		Ok(Told { yours, settled, used })
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A command whose five numbers are five different numbers.
	fn asked(number: u32) -> Command {
		let turn = [0.3_f32, -1.4, 2.05, 0.75, -2.9];
		let yaw = turn[usize::try_from(number % 5).unwrap_or(0)];

		Command {
			step: u64::from(number) * 7 + 1000,
			number,
			buttons: 0b1001 << (number % 5),
			yaw,
			pitch: yaw * -0.375,
		}
	}

	/// Some client, built the way the wire builds one.
	fn client(index: u32, generation: u32) -> PeerId {
		PeerId::from_bits((u64::from(generation) << 32) | u64::from(index))
	}

	/// The layout, written down as the bytes a datagram would carry.
	///
	/// Every other test here goes out through `write` and back through `read`
	/// and would not notice the two of them agreeing on the wrong thing. This
	/// one is a known answer.
	#[test]
	fn a_block_is_fourteen_bytes_of_head_and_twenty_four_a_command() {
		let mut out = vec![0xAA_u8];
		let one = Command {
			step: 0x0102_0304_0506_0708,
			number: 0x090A_0B0C,
			buttons: 0x0D0E_0F10,
			yaw: 1.0,
			pitch: -2.0,
		};

		// the numbers themselves, because every other assertion in this file
		// spells the ceilings symbolically and would move with them. A block
		// that carried a command more or less than a ring holds is not a
		// failure any of them could see.
		assert_eq!(MAX_ASKED, 32, "exactly what a ring holds, and the ring holds thirty-two");
		assert_eq!(MAX_BLOCK, 782, "fourteen of head and twenty-four apiece");

		assert_eq!(Block::write(client(6, 0x1234_5678), 0x1122_3344, &[one], &mut out), 1);
		assert_eq!(out.len(), 1 + HEAD + RECORD);
		assert_eq!(&out[1..], &[
			0x06, 0x00, 0x00, 0x00, 0x78, 0x56, 0x34, 0x12, // yours
			0x44, 0x33, 0x22, 0x11, // settled
			0x01, 0x00, // count
			0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // step
			0x0C, 0x0B, 0x0A, 0x09, // number
			0x10, 0x0F, 0x0E, 0x0D, // buttons
			0x00, 0x00, 0x80, 0x3F, // yaw
			0x00, 0x00, 0x00, 0xC0, // pitch
		]);

		let mut back = Vec::new();
		let told = Block::read(&out[1..], &mut back).expect("it is a block");

		assert_eq!(told.yours, client(6, 0x1234_5678));
		assert_eq!(told.settled, 0x1122_3344);
		assert_eq!(told.used, HEAD + RECORD);
		assert_eq!(back, vec![one], "and every word of it came back");
	}

	#[test]
	fn a_block_with_nothing_in_it_still_says_what_the_head_says() {
		let mut out = Vec::new();

		assert_eq!(Block::write(PeerId::NONE, 91, &[], &mut out), 0);
		assert_eq!(out.len(), HEAD, "the head and not a byte more");

		let mut back = vec![asked(1)];
		let told = Block::read(&out, &mut back).expect("a head is a block");

		assert_eq!(told.yours, PeerId::NONE, "which is what a client writes");
		assert_eq!(told.settled, 91, "and the number is the news");
		assert_eq!(told.used, HEAD);
		assert!(back.is_empty(), "and whatever was in the way is gone");
	}

	#[test]
	fn a_block_says_where_it_ends_so_something_can_follow_it() {
		let mut out = Vec::new();
		let commands: Vec<Command> = [4_u32, 9, 10].iter().map(|n| asked(*n)).collect();

		Block::write(PeerId::NONE, 0, &commands, &mut out);
		// what a snapshot block would be, and the bytes are deliberately not a
		// block of commands: a reader that ignored the count and read to the
		// end would take them for records.
		out.extend_from_slice(&[0x7F; 40]);

		let mut back = Vec::new();
		let told = Block::read(&out, &mut back).expect("it is a block");

		assert_eq!(told.used, HEAD + 3 * RECORD, "and the rest is somebody else's");
		assert_eq!(back, commands);
		// the bytes themselves rather than how many there are, which would be
		// arithmetic on the line above: what matters is that `used` points at
		// the thing that follows and not one byte either side of it.
		assert_eq!(&out[told.used..], &[0x7F; 40], "and it starts exactly where it should");

		// and the same with nothing in the block at all, which is what almost
		// every message actually carries: a head, then a snapshot. A reader
		// that took a head-only block's length from the buffer rather than
		// from its count would swallow the snapshot whole.
		let mut empty = Vec::new();

		Block::write(PeerId::NONE, 5, &[], &mut empty);
		empty.extend_from_slice(&[0x7F; 40]);

		let told = Block::read(&empty, &mut back).expect("a head is a block");

		assert_eq!(told.used, HEAD);
		assert!(back.is_empty());
		assert_eq!(&empty[told.used..], &[0x7F; 40]);
	}

	#[test]
	fn a_ring_and_a_half_is_cut_down_to_the_newest_ring() {
		let mut out = Vec::new();
		// five past the depth, so the boundary is crossed rather than landed
		// on, and numbered from something other than one.
		let many: Vec<Command> = (100..100 + u32::try_from(MAX_ASKED + 5).expect("small"))
			.map(asked)
			.collect();

		assert_eq!(Block::write(PeerId::NONE, 0, &many, &mut out), MAX_ASKED);
		assert_eq!(out.len(), MAX_BLOCK);

		let mut back = Vec::new();

		Block::read(&out, &mut back).expect("it is a block");
		assert_eq!(back.len(), MAX_ASKED);
		assert_eq!(back.first().copied(), Some(asked(105)), "the oldest five went");
		assert_eq!(back.last().copied(), many.last().copied(), "and the newest stayed");
	}

	#[test]
	fn a_block_claiming_more_than_a_ring_holds_reserves_nothing() {
		let mut out = Vec::new();

		Block::write(PeerId::NONE, 0, &[], &mut out);
		// the first count past the ceiling, not a number far past it: the
		// bound is what is being tested rather than arithmetic on a big
		// number.
		let over = u16::try_from(MAX_ASKED + 1).expect("small");

		out[12..14].copy_from_slice(&over.to_le_bytes());

		// something in the way, so that "left empty" is a thing that happened
		// rather than a thing that was already true.
		let mut back = vec![asked(7), asked(8)];

		assert_eq!(Block::read(&out, &mut back), Err(Fault::TooMany));
		assert!(back.is_empty());

		// and the largest a two-byte count can be, which is sixty-five
		// thousand records of twenty-four bytes - a megabyte and a half
		// reserved on one datagram from anybody at all.
		out[12..14].copy_from_slice(&u16::MAX.to_le_bytes());
		assert_eq!(Block::read(&out, &mut back), Err(Fault::TooMany));
	}

	#[test]
	fn a_block_that_stops_in_the_middle_of_a_command_is_refused() {
		let mut out = Vec::new();

		Block::write(PeerId::NONE, 0, &[asked(3), asked(4)], &mut out);

		// as above: what is in the way has to go, and a fresh vector cannot
		// show that.
		let mut back = vec![asked(9)];

		// one byte short of the whole thing, which is the only interesting
		// place to cut it: the head is intact and the count is honest.
		assert_eq!(Block::read(&out[..out.len() - 1], &mut back), Err(Fault::Short));
		assert!(back.is_empty(), "and nothing of it is handed over");

		for cut in 0..HEAD {
			assert_eq!(Block::read(&out[..cut], &mut back), Err(Fault::Short), "cut at {cut}");
		}

		assert_eq!(
			Block::read(&out[..HEAD], &mut back),
			Err(Fault::Short),
			"a head promising two commands and no commands after it"
		);
	}

	#[test]
	fn a_block_out_of_its_own_order_is_refused_whole() {
		let mut out = Vec::new();

		Block::write(PeerId::NONE, 0, &[asked(3), asked(9), asked(4)], &mut out);

		let mut back = Vec::new();

		assert_eq!(Block::read(&out, &mut back), Err(Fault::OutOfOrder));
		assert!(back.is_empty(), "the two before the fault go with it");

		// and the same number twice, which is a peer repeating itself inside
		// one block rather than across two.
		let mut twice = Vec::new();

		Block::write(PeerId::NONE, 0, &[asked(3), asked(3)], &mut twice);
		assert_eq!(Block::read(&twice, &mut back), Err(Fault::OutOfOrder));
	}

	#[test]
	fn a_command_numbered_nothing_makes_the_block_nothing() {
		let mut out = Vec::new();

		Block::write(PeerId::NONE, 0, &[asked(3), Command { number: 0, ..asked(4) }], &mut out);

		let mut back = Vec::new();

		assert_eq!(Block::read(&out, &mut back), Err(Fault::Nothing));
		assert!(back.is_empty());
	}

	/// Every bit pattern is a legal command, and the block has to survive the
	/// ones that are not sensible.
	#[test]
	fn a_block_of_nonsense_is_read_rather_than_refused_or_believed() {
		let mut out = Vec::new();
		let odd = Command {
			step: u64::MAX,
			number: u32::MAX,
			buttons: u32::MAX,
			yaw: f32::NAN,
			pitch: f32::INFINITY,
		};

		Block::write(client(1, 1), u32::MAX, &[odd], &mut out);

		let mut back = Vec::new();
		let told = Block::read(&out, &mut back).expect("nonsense is still a block");

		assert_eq!(told.settled, u32::MAX);
		assert_eq!(back.len(), 1);
		assert_eq!(back[0].step, u64::MAX);
		assert_eq!(back[0].number, u32::MAX);
		// the bits rather than the value, because a NaN is not equal to
		// itself and the wire's promise is about bits.
		assert_eq!(back[0].yaw.to_bits(), f32::NAN.to_bits());
		assert!(back[0].pitch.is_infinite());
	}
}
