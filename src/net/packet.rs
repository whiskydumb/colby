//! The sixteen bytes at the head of a datagram, and what it takes to be one.
//!
//! ```text
//!    0  magic      u16   the two bytes that say this is one of ours
//!    2  version    u16   the protocol, and a mismatch is refused over it
//!    4  sequence   u16   which message this datagram is part of
//!    6  ack        u16   the newest message that has arrived from the far end
//!    8  ack_bits   u32   which of the thirty-two before `ack` arrived
//!   12  fragment   u16   which piece of the message this is, counting from nil
//!   14  fragments  u16   how many pieces the message has, one meaning whole
//!   16  the piece, up to [`MAX_PAYLOAD`] bytes of it
//! ```
//!
//! **Every field is written and read a byte at a time, little-endian.** Nothing
//! here is a `#[repr(C)]` block cast in place the way an asset is, and the
//! reason is that the bytes come off a socket rather than out of a file: they
//! are not aligned, they were not written by this build, and the writer may not
//! even have been this program. So the layout is spelled out and the padding a
//! compiler would insert never gets a say. It costs a dozen shifts per datagram
//! and it removes a whole class of question.
//!
//! **A message is the unit, not a datagram.** All the pieces of one message
//! carry the same `sequence`; the number only moves when the next message goes
//! out. That is what makes an acknowledgement mean something - half a message
//! is worth nothing to whoever asked for it, so there is nothing useful to
//! acknowledge until the whole of it is there.
//!
//! **The piece count is on the wire rather than implied by a short tail.** The
//! obvious alternative, and the one the engine this wire is modeled on takes,
//! is to say that a piece smaller than the maximum is the last one - which
//! forces the sender to append an *empty* piece whenever the message divides
//! exactly, and hangs the receiver forever if it forgets. Two more bytes buy
//! that whole case away, and they buy out-of-order reassembly with it.

use std::fmt;

/// The two bytes every datagram starts with.
pub const MAGIC: u16 = u16::from_le_bytes(*b"CN");

/// The revision of everything in this module.
///
/// Bump it whenever the header, the fragment rules or the reliable block change
/// shape. A datagram carrying a different number is refused with the number in
/// the message rather than read as if it agreed.
pub const PROTOCOL_VERSION: u16 = 1;

const MAGIC_AT: usize = 0;
const VERSION_AT: usize = 2;
const SEQUENCE_AT: usize = 4;
const ACK_AT: usize = 6;
const ACK_BITS_AT: usize = 8;
const FRAGMENT_AT: usize = 12;
const FRAGMENTS_AT: usize = 14;

/// How big the head of a datagram is, and where the piece starts.
pub const HEADER_BYTES: usize = FRAGMENTS_AT + 2;

/// The most bytes one datagram may be, head and piece together.
///
/// The smallest maximum transmission unit anything is required to carry is
/// 1280 bytes; forty of those go to an address header and eight more to the
/// datagram header, which leaves 1232. Twelve hundred sits under that with room
/// for a tunnel, which is why every library in the field picked the same
/// number.
pub const MAX_DATAGRAM: usize = 1200;

/// The most bytes of a message one datagram may carry.
pub const MAX_PAYLOAD: usize = MAX_DATAGRAM - HEADER_BYTES;

/// The most pieces one message may be cut into.
///
/// Sixty-four, because that is how many bits a `u64` has and the receiving side
/// tracks which pieces have arrived in exactly one of those. A ceiling here is
/// not tidiness: without it a peer could claim a million pieces and make the
/// far end reserve room for all of them, which is the shape of the one memory
/// bug worth naming in this whole subsystem.
pub const MAX_FRAGMENTS: usize = 64;

/// The largest message that can be sent, in bytes.
///
/// Derived rather than chosen. It is comfortably more than a whole world sent
/// at once and more than a full ring of reliable commands.
pub const MAX_MESSAGE: usize = MAX_PAYLOAD * MAX_FRAGMENTS;

/// How many messages behind [`Header::ack`] the acknowledgement field covers.
pub const ACK_BITS: u16 = 32;

/// Half the sequence space, which is where "newer" stops meaning anything.
const HALF: u16 = 1 << 15;

/// Why a datagram was not worth anything.
///
/// Each of these is a fact rather than a failure: a stray datagram from
/// somebody else's program is not an error the way a truncated file is, and
/// building an [`Error`](colby_core::Error) per stray packet would be a way of
/// letting whoever is sending them fill somebody's log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
	/// Fewer bytes than a header.
	Short,

	/// The first two bytes are not [`MAGIC`].
	Foreign,

	/// A colby speaking a different [`PROTOCOL_VERSION`], carried here.
	Version(u16),

	/// The sequence number kept back to mean "nothing has arrived yet".
	Reserved,

	/// More bytes than [`MAX_DATAGRAM`].
	Oversize,

	/// A piece count of nil, one past [`MAX_FRAGMENTS`], or an index outside
	/// the count it declares.
	Fragments,

	/// A piece that is not the size its place in the message demands.
	Piece,

	/// A message at or behind one that has already been handed over whole.
	Stale,

	/// A piece claiming a different piece count from the message being built.
	Divided,
}

impl fmt::Display for Reason {
	fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
		match *self {
			| Self::Short => out.write_str("shorter than a header"),
			| Self::Foreign => out.write_str("not one of ours"),
			| Self::Version(version) => write!(out, "protocol {version}, not {PROTOCOL_VERSION}"),
			| Self::Reserved => out.write_str("the sequence number that means nothing yet"),
			| Self::Oversize => out.write_str("longer than a datagram may be"),
			| Self::Fragments => out.write_str("a piece count that cannot be right"),
			| Self::Piece => out.write_str("a piece of the wrong size for its place"),
			| Self::Stale => out.write_str("older than what has already arrived"),
			| Self::Divided => out.write_str("a piece of a differently divided message"),
		}
	}
}

/// The head of a datagram, once it has been read and found to be one.
///
/// The magic and the version are not fields: writing always writes this
/// build's, and reading refuses anything else, so there is nothing to carry and
/// nothing to get wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
	/// Which message this datagram is part of. @ref [`after`].
	pub sequence: u16,

	/// The newest message the sender has taken whole from its far end, or nil
	/// if it has taken none.
	pub ack: u16,

	/// One bit per message before [`ack`](Self::ack), the lowest bit being
	/// `ack - 1`. A bit that is set means that message arrived whole.
	pub ack_bits: u32,

	/// Which piece of the message this is, counting from nil.
	pub fragment: u16,

	/// How many pieces the message was cut into. One means it is all here.
	pub fragments: u16,
}

impl Header {
	/// Lays the head down in the first [`HEADER_BYTES`] of a datagram.
	///
	/// @param out - where the datagram is being built
	pub fn write(&self, out: &mut [u8; HEADER_BYTES]) {
		out[MAGIC_AT..VERSION_AT].copy_from_slice(&MAGIC.to_le_bytes());
		out[VERSION_AT..SEQUENCE_AT].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
		out[SEQUENCE_AT..ACK_AT].copy_from_slice(&self.sequence.to_le_bytes());
		out[ACK_AT..ACK_BITS_AT].copy_from_slice(&self.ack.to_le_bytes());
		out[ACK_BITS_AT..FRAGMENT_AT].copy_from_slice(&self.ack_bits.to_le_bytes());
		out[FRAGMENT_AT..FRAGMENTS_AT].copy_from_slice(&self.fragment.to_le_bytes());
		out[FRAGMENTS_AT..HEADER_BYTES].copy_from_slice(&self.fragments.to_le_bytes());
	}

	/// Reads the head of a datagram and checks that the rest of it could be
	/// one.
	///
	/// Everything checkable from the bytes alone is checked here, the length of
	/// the piece included, so that whoever calls this has one branch rather
	/// than six. What is *not* checkable here is whether the message is one
	/// worth having, which needs to know what has already arrived. @ref
	/// [`Channel::receive`](crate::Channel::receive).
	///
	/// @param datagram - the whole of what arrived
	/// @return the head, or why the datagram is worth nothing
	pub fn read(datagram: &[u8]) -> Result<Self, Reason> {
		if datagram.len() < HEADER_BYTES {
			return Err(Reason::Short);
		}
		if datagram.len() > MAX_DATAGRAM {
			return Err(Reason::Oversize);
		}
		if u16_at(datagram, MAGIC_AT) != MAGIC {
			return Err(Reason::Foreign);
		}

		let version = u16_at(datagram, VERSION_AT);

		if version != PROTOCOL_VERSION {
			return Err(Reason::Version(version));
		}

		let header = Self {
			sequence: u16_at(datagram, SEQUENCE_AT),
			ack: u16_at(datagram, ACK_AT),
			ack_bits: u32_at(datagram, ACK_BITS_AT),
			fragment: u16_at(datagram, FRAGMENT_AT),
			fragments: u16_at(datagram, FRAGMENTS_AT),
		};
		// the sending side steps over nil so that it can mean "nothing has
		// arrived yet" in the acknowledgement field, and the reading side has to
		// agree with it. A datagram carrying nil would otherwise set the far
		// end back to that reading, after which every real message behind it is
		// older than what has already arrived.
		if header.sequence == 0 {
			return Err(Reason::Reserved);
		}

		let fragments = usize::from(header.fragments);

		// @note: the first of the three cannot be observed on its own, and it
		// is kept because it is the one that says what the rule is. A piece
		// index is never below nil, so a count of nil already fails the third,
		// and no mutation of this clause alone can break a test.
		if fragments == 0 || fragments > MAX_FRAGMENTS || header.fragment >= header.fragments {
			return Err(Reason::Fragments);
		}

		// every piece but the last is exactly full, so the length of the whole
		// message follows from the count and the length of the last one. A peer
		// that sent short middle pieces could otherwise describe a message two
		// different ways.
		let piece = datagram.len() - HEADER_BYTES;
		let last = usize::from(header.fragment) + 1 == fragments;

		if !last && piece != MAX_PAYLOAD {
			return Err(Reason::Piece);
		}

		Ok(header)
	}

	/// The part of a datagram that is not the head.
	///
	/// Hands back nothing rather than refusing when there is no head to be
	/// past, because the precondition below is a doc comment and this is a
	/// public function taking bytes off a wire. A caller that has not been
	/// through [`read`](Self::read) has a bug either way; it should not be a
	/// panic.
	///
	/// @param datagram - the whole of what arrived
	#[must_use]
	pub fn piece(datagram: &[u8]) -> &[u8] { datagram.get(HEADER_BYTES..).unwrap_or_default() }
}

/// Whether one sequence number is newer than another.
///
/// Sequence numbers wrap, so this cannot be `>`. Anything within half the
/// space ahead is newer and anything else is older, which is the only reading
/// that survives the wrap and is what every protocol with a short sequence
/// number does. Half an hour of datagrams at sixty a second is what it takes to
/// go round, and being behind by more than thirty-two thousand of them is a
/// connection that is over.
///
/// @param sequence - the one being judged
/// @param than - what it is being judged against
/// @return whether `sequence` comes after `than`; a number is not after itself
#[must_use]
pub const fn after(sequence: u16, than: u16) -> bool {
	sequence != than && sequence.wrapping_sub(than) < HALF
}

/// How far ahead one sequence number is of another.
///
/// Only meaningful when [`after`] says it is ahead.
///
/// @param sequence - the newer one
/// @param from - the older one
#[must_use]
pub const fn distance(sequence: u16, from: u16) -> u16 { sequence.wrapping_sub(from) }

/// A little-endian short at a place the caller has already checked is there.
pub(crate) fn u16_at(bytes: &[u8], at: usize) -> u16 {
	u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

/// A little-endian long at a place the caller has already checked is there.
pub(crate) fn u32_at(bytes: &[u8], at: usize) -> u32 {
	u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// A little-endian eight-byte number at a place already checked.
pub(crate) fn u64_at(bytes: &[u8], at: usize) -> u64 {
	u64::from_le_bytes([
		bytes[at],
		bytes[at + 1],
		bytes[at + 2],
		bytes[at + 3],
		bytes[at + 4],
		bytes[at + 5],
		bytes[at + 6],
		bytes[at + 7],
	])
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A datagram with the given head and a piece of the given length.
	fn datagram(header: &Header, piece: usize) -> Vec<u8> {
		let mut head = [0_u8; HEADER_BYTES];

		header.write(&mut head);

		let mut out = head.to_vec();

		out.resize(HEADER_BYTES + piece, 0xAB);
		out
	}

	fn whole() -> Header {
		Header {
			sequence: 7,
			ack: 3,
			ack_bits: 0b1011,
			fragment: 0,
			fragments: 1,
		}
	}

	#[test]
	fn a_head_survives_the_round_trip_field_for_field() {
		let header = Header {
			sequence: 0x1234,
			ack: 0x5678,
			ack_bits: 0x9ABC_DEF0,
			fragment: 2,
			fragments: 3,
		};
		let bytes = datagram(&header, MAX_PAYLOAD);
		let read = Header::read(&bytes).expect("it is a datagram");

		assert_eq!(read, header, "every field comes back as it went in");
	}

	#[test]
	fn the_head_is_sixteen_bytes_and_the_rest_is_the_piece() {
		assert_eq!(HEADER_BYTES, 16, "two shorts, a long and two more shorts");
		assert_eq!(MAX_PAYLOAD, MAX_DATAGRAM - 16, "and the piece is whatever is left");

		let bytes = datagram(&whole(), 5);

		assert_eq!(Header::piece(&bytes), &[0xAB; 5], "the piece is what follows the head");
	}

	#[test]
	fn a_datagram_shorter_than_a_head_is_refused() {
		for length in 0..HEADER_BYTES {
			let bytes = vec![0_u8; length];

			assert_eq!(Header::read(&bytes), Err(Reason::Short), "{length} bytes is not a head");
		}
	}

	#[test]
	fn a_datagram_longer_than_the_ceiling_is_refused() {
		let mut bytes = datagram(&whole(), MAX_PAYLOAD);

		bytes.push(0);
		assert_eq!(Header::read(&bytes), Err(Reason::Oversize), "one byte over is over");
	}

	#[test]
	fn something_that_is_not_ours_is_refused_before_anything_else_is_read() {
		let mut bytes = datagram(&whole(), 0);

		bytes[MAGIC_AT] ^= 0xFF;
		assert_eq!(Header::read(&bytes), Err(Reason::Foreign), "the magic is the first gate");
	}

	#[test]
	fn a_different_protocol_is_refused_and_says_which_one() {
		let mut bytes = datagram(&whole(), 0);

		bytes[VERSION_AT..VERSION_AT + 2].copy_from_slice(&99_u16.to_le_bytes());
		assert_eq!(
			Header::read(&bytes),
			Err(Reason::Version(99)),
			"the number is carried so a log line can name it"
		);
	}

	#[test]
	fn the_sequence_kept_back_to_mean_nothing_yet_is_refused() {
		// the read side has to hold the invariant the write side keeps, or a
		// datagram carrying nil sets the far end back to "nothing has arrived"
		// and everything real behind it then reads as older than that.
		let header = Header { sequence: 0, ..whole() };

		assert_eq!(Header::read(&datagram(&header, 4)), Err(Reason::Reserved));
		assert!(Header::read(&datagram(&whole(), 4)).is_ok(), "and one is a real message");
	}

	#[test]
	fn the_piece_of_something_that_is_not_a_datagram_is_nothing() {
		assert!(Header::piece(&[]).is_empty());
		assert!(Header::piece(&[1_u8, 2, 3]).is_empty(), "shorter than the head it is past");
	}

	#[test]
	fn the_head_is_laid_down_at_the_offsets_the_module_publishes() {
		// the offsets are the contract rather than a detail, because the thing
		// on the far end may not have been built here. Every other test in this
		// file writes and reads through the same constants and would not notice
		// two of them swapping.
		let header = Header {
			sequence: 0x0201,
			ack: 0x0403,
			ack_bits: 0x0807_0605,
			fragment: 0x0A09,
			fragments: 0x0C0B,
		};
		let mut head = [0_u8; HEADER_BYTES];

		header.write(&mut head);
		assert_eq!(
			head,
			[b'C', b'N', 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0x0A, 0x0B, 0x0C],
			"the magic, version one, then the five fields little-endian in order"
		);
	}

	#[test]
	fn every_reason_a_datagram_is_refused_for_reads_as_a_sentence() {
		// the only thing this module puts in front of a person, and the one
		// place the protocol number a mismatch carries is actually shown.
		assert_eq!(Reason::Short.to_string(), "shorter than a header");
		assert_eq!(Reason::Foreign.to_string(), "not one of ours");
		assert_eq!(Reason::Version(7).to_string(), format!("protocol 7, not {PROTOCOL_VERSION}"));
		assert_eq!(Reason::Reserved.to_string(), "the sequence number that means nothing yet");
		assert_eq!(Reason::Oversize.to_string(), "longer than a datagram may be");
		assert_eq!(Reason::Fragments.to_string(), "a piece count that cannot be right");
		assert_eq!(Reason::Piece.to_string(), "a piece of the wrong size for its place");
		assert_eq!(Reason::Stale.to_string(), "older than what has already arrived");
		assert_eq!(Reason::Divided.to_string(), "a piece of a differently divided message");
	}

	#[test]
	fn a_piece_count_of_nil_is_refused() {
		let header = Header { fragments: 0, ..whole() };
		let bytes = datagram(&header, 0);

		assert_eq!(Header::read(&bytes), Err(Reason::Fragments), "no message has no pieces");
	}

	#[test]
	fn a_piece_count_past_the_ceiling_is_refused_before_anything_is_reserved() {
		// the whole point of the ceiling: a peer saying it has sent a great
		// many pieces must not be able to make this side put anything aside for
		// them.
		let header = Header { fragment: 0, fragments: 65, ..whole() };
		let bytes = datagram(&header, MAX_PAYLOAD);

		assert_eq!(Header::read(&bytes), Err(Reason::Fragments), "sixty-five is one too many");

		let header = Header { fragment: 0, fragments: 64, ..whole() };
		let bytes = datagram(&header, MAX_PAYLOAD);

		assert!(Header::read(&bytes).is_ok(), "and sixty-four is exactly enough");
	}

	#[test]
	fn a_piece_index_outside_its_own_count_is_refused() {
		let header = Header { fragment: 3, fragments: 3, ..whole() };
		let bytes = datagram(&header, 10);

		assert_eq!(Header::read(&bytes), Err(Reason::Fragments), "three of three is the fourth");
	}

	#[test]
	fn a_piece_that_is_not_the_last_has_to_be_full() {
		let header = Header { fragment: 0, fragments: 2, ..whole() };

		assert_eq!(
			Header::read(&datagram(&header, MAX_PAYLOAD - 1)),
			Err(Reason::Piece),
			"a short middle piece would make the message two lengths at once"
		);
		assert!(Header::read(&datagram(&header, MAX_PAYLOAD)).is_ok(), "a full one is fine");
	}

	#[test]
	fn the_last_piece_may_be_any_length_up_to_a_full_one() {
		let header = Header { fragment: 1, fragments: 2, ..whole() };

		for piece in [0, 1, 400, MAX_PAYLOAD] {
			assert!(Header::read(&datagram(&header, piece)).is_ok(), "{piece} bytes is a tail");
		}
	}

	#[test]
	fn a_message_of_no_bytes_at_all_is_a_datagram() {
		// what an acknowledgement rides on when there is nothing to say.
		let bytes = datagram(&whole(), 0);

		assert!(Header::read(&bytes).is_ok(), "a head and nothing else is still one");
		assert!(Header::piece(&bytes).is_empty());
	}

	#[test]
	fn newer_means_newer_and_a_number_is_not_newer_than_itself() {
		assert!(after(2, 1), "two follows one");
		assert!(!after(1, 2), "and one does not follow two");
		assert!(!after(5, 5), "nor itself");
	}

	#[test]
	fn newer_survives_the_wrap() {
		assert!(after(1, 65_535), "the number after the last one is the first one");
		assert!(!after(65_535, 1), "and not the other way");
		assert_eq!(distance(1, 65_535), 2, "with nil counted between them");
	}

	#[test]
	fn half_the_space_away_is_where_newer_stops_meaning_anything() {
		// not a bug and worth saying out loud. At exactly half the space
		// neither number is newer than the other, and past it the reading turns
		// around - a connection thirty-two thousand messages behind is over
		// long before this decides anything.
		assert!(after(HALF - 1, 0), "just under half is ahead");
		assert!(!after(HALF, 0), "exactly half is neither");
		assert!(!after(0, HALF), "in either direction");
		assert!(after(0, HALF + 1), "and one past it reads the other way round");
	}

	#[test]
	fn distance_counts_forward_across_the_wrap() {
		assert_eq!(distance(10, 3), 7);
		assert_eq!(distance(3, 65_533), 6, "over the top");
	}

	#[test]
	fn the_numbers_this_wire_was_designed_around_are_the_ones_it_uses() {
		// each of these is a decision with an argument above it, and each is
		// invisible to every other test here because writing and reading agree
		// on whatever the numbers happen to be.
		assert_eq!(
			MAX_DATAGRAM, 1200,
			"under the smallest transmission unit anything must carry"
		);
		assert_eq!(HEADER_BYTES, 16, "two shorts, a long and two more shorts");
		assert_eq!(MAX_FRAGMENTS, 64, "which is how many bits are used to track them");
		assert_eq!(ACK_BITS, 32, "and the acknowledgement field is a long");
		assert_eq!(MAX_MESSAGE, 75_776, "so the largest message is this many bytes");
	}
}
