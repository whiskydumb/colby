//! Cutting a message into datagrams, and putting one back together.
//!
//! Both halves are here because they are one rule read in two directions: the
//! sender cuts at [`MAX_PAYLOAD`] and stops, and the receiver knows where each
//! piece goes because every piece but the last is exactly that long. Nothing
//! carries a byte offset, the way the engine this is modeled on does, because
//! the index and a fixed stride say the same thing in half the space and
//! without a second number to disagree with the first.
//!
//! **The room for a message being rebuilt is taken once and never grows.** A
//! peer says how many pieces its message has and this side has already refused
//! anything past [`MAX_FRAGMENTS`] by then, so the reservation is the same
//! whatever arrives. That is deliberate: the shape of the one memory bug worth
//! naming in this subsystem is a receiver that believes a piece count and
//! reserves room for it, and there is no arithmetic here that a datagram can
//! reach.
//!
//! **One message at a time.** A piece of a newer message throws away whatever
//! was half built, which is right for everything this wire carries - a snapshot
//! that is missing a piece is a snapshot nobody wants once a newer one has
//! started arriving. What is kept is out-of-order tolerance *within* one
//! message, which the piece index buys for the price of a `u64` of flags.

use crate::packet::{MAX_FRAGMENTS, MAX_MESSAGE, MAX_PAYLOAD};

const _: () = assert!(
	MAX_FRAGMENTS == 64,
	"which pieces of a message have arrived is tracked in the bits of one u64"
);

/// How many datagrams a message of this length takes.
///
/// A message of no bytes still takes one, because a datagram with nothing in it
/// is how an acknowledgement travels when there is nothing else to say.
///
/// @param length - the whole message, in bytes
#[must_use]
pub fn count(length: usize) -> usize {
	if length == 0 { 1 } else { length.div_ceil(MAX_PAYLOAD) }
}

/// The bytes of a message that belong in one datagram.
///
/// @param payload - the whole message
/// @param index - which piece, counting from nil, below [`count`]
#[must_use]
pub fn piece(payload: &[u8], index: usize) -> &[u8] {
	// the multiplication is checked before the comparison rather than after it,
	// because a shipping build here keeps overflow checks on: an index far
	// enough out would otherwise be an arithmetic panic on the line that exists
	// to hand such an index nothing.
	let Some(at) = index.checked_mul(MAX_PAYLOAD) else {
		return &[];
	};

	// @note: what the comparison stops is an index past the end of the message,
	// which would otherwise be a slice starting after it ends. Its exact
	// boundary cannot be observed - at exactly the end the clamp below agrees
	// with it and hands back the same nothing - so no mutation of the
	// comparison alone can fail a test, while removing the line does.
	if at >= payload.len() {
		return &[];
	}

	let end = payload.len().min(at + MAX_PAYLOAD);

	&payload[at..end]
}

/// A message being put back together out of the datagrams carrying it.
///
/// Holds room for the largest message there is, taken once when the channel is
/// made. @ref [`MAX_MESSAGE`].
#[derive(Clone, Debug)]
pub struct Assembly {
	bytes: Vec<u8>,
	sequence: u16,
	fragments: u16,
	present: u64,
	length: usize,
}

impl Default for Assembly {
	fn default() -> Self { Self::new() }
}

impl Assembly {
	/// Room for one message, with nothing being built in it.
	#[must_use]
	pub fn new() -> Self {
		Self {
			bytes: vec![0; MAX_MESSAGE],
			sequence: 0,
			fragments: 0,
			present: 0,
			length: 0,
		}
	}

	/// Whether a message is part way here.
	#[must_use]
	pub const fn building(&self) -> bool { self.fragments != 0 }

	/// Which message is part way here. Only meaningful while
	/// [`building`](Self::building).
	#[must_use]
	pub const fn sequence(&self) -> u16 { self.sequence }

	/// How many pieces the message being built was cut into.
	#[must_use]
	pub const fn fragments(&self) -> u16 { self.fragments }

	/// Throws away whatever was being built and starts on a new message.
	///
	/// @note: a count of nil or one past [`MAX_FRAGMENTS`] leaves this holding
	/// nothing rather than trusting it. Nothing arriving through a channel can
	/// reach that - the head is checked before this is called - but this is the
	/// place the arithmetic below depends on, so it is checked where it is
	/// depended on.
	///
	/// @param sequence - which message
	/// @param fragments - how many pieces it was cut into
	pub fn start(&mut self, sequence: u16, fragments: u16) {
		self.sequence = sequence;
		self.fragments = if fragments == 0 || usize::from(fragments) > MAX_FRAGMENTS {
			0
		} else {
			fragments
		};
		self.present = 0;
		self.length = 0;
	}

	/// Puts one piece where it belongs.
	///
	/// @param index - which piece, counting from nil
	/// @param piece - its bytes
	/// @return whether that was the last one missing
	pub fn put(&mut self, index: u16, piece: &[u8]) -> bool {
		if index >= self.fragments || piece.len() > MAX_PAYLOAD {
			return false;
		}

		let at = usize::from(index) * MAX_PAYLOAD;

		self.bytes[at..at + piece.len()].copy_from_slice(piece);
		self.present |= 1_u64 << u32::from(index);

		if index + 1 == self.fragments {
			self.length = at + piece.len();
		}

		self.present == mask(self.fragments)
	}

	/// Says the message is no longer being built, leaving it readable.
	pub const fn finish(&mut self) { self.fragments = 0; }

	/// The message, as far as it has been finished.
	///
	/// Only worth reading after [`put`](Self::put) has said the last piece
	/// landed.
	#[must_use]
	pub fn message(&self) -> &[u8] { &self.bytes[..self.length] }
}

/// One bit per piece a message of this many has.
fn mask(fragments: u16) -> u64 {
	if fragments == 0 || usize::from(fragments) > MAX_FRAGMENTS {
		return 0;
	}

	u64::MAX >> (u64::BITS - u32::from(fragments))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A message of `length` bytes, every one of them different from its
	/// neighbors, so that a piece landing in the wrong place shows.
	fn message(length: usize) -> Vec<u8> {
		(0..length)
			.map(|at| u8::try_from(at % 251).unwrap_or(0))
			.collect()
	}

	#[test]
	fn a_message_that_fits_is_one_piece_and_a_message_of_nothing_is_still_one() {
		assert_eq!(count(0), 1, "an empty message is what an acknowledgement rides on");
		assert_eq!(count(1), 1);
		assert_eq!(count(MAX_PAYLOAD), 1, "exactly full is still one");
		assert_eq!(count(MAX_PAYLOAD + 1), 2, "one byte over is two");
	}

	#[test]
	fn a_message_that_divides_exactly_needs_no_extra_empty_piece() {
		// the whole reason the count is on the wire. Told only that a short
		// piece ends the message, a sender has to append an empty one here or
		// the far end waits forever.
		assert_eq!(count(MAX_PAYLOAD * 2), 2, "two full pieces and no terminator");
		assert_eq!(count(MAX_PAYLOAD * 64), 64, "and the largest message is sixty-four");
		assert_eq!(count(MAX_MESSAGE), MAX_FRAGMENTS);
	}

	#[test]
	fn the_pieces_put_end_to_end_are_the_message() {
		for length in [0, 1, MAX_PAYLOAD - 1, MAX_PAYLOAD, MAX_PAYLOAD + 1, MAX_PAYLOAD * 3 + 7] {
			let payload = message(length);
			let mut back = Vec::new();

			for index in 0..count(length) {
				back.extend_from_slice(piece(&payload, index));
			}

			assert_eq!(back, payload, "{length} bytes came back as they went");
		}
	}

	#[test]
	fn every_piece_but_the_last_is_exactly_full() {
		let payload = message(MAX_PAYLOAD * 2 + 5);
		let pieces = count(payload.len());

		assert_eq!(pieces, 3);
		assert_eq!(piece(&payload, 0).len(), MAX_PAYLOAD);
		assert_eq!(piece(&payload, 1).len(), MAX_PAYLOAD);
		assert_eq!(piece(&payload, 2).len(), 5, "and the tail is whatever is left");
	}

	#[test]
	fn a_piece_past_the_end_of_the_message_is_nothing() {
		let payload = message(10);

		assert!(piece(&payload, 1).is_empty(), "there is no second piece of ten bytes");
		assert!(piece(&[], 0).is_empty(), "and none at all of nothing");
	}

	#[test]
	fn a_piece_index_far_past_the_end_is_nothing_rather_than_arithmetic() {
		// the stride is over a thousand and the index is a usize, so a large
		// one goes over in the multiplication before it ever reaches the
		// comparison that is there to hand it nothing.
		assert!(piece(&message(10), usize::MAX).is_empty());
		assert!(piece(&message(10), usize::MAX / 2).is_empty());
		assert!(piece(&message(10), usize::MAX / MAX_PAYLOAD).is_empty());
	}

	#[test]
	fn a_message_of_one_piece_is_finished_by_that_piece() {
		let mut assembly = Assembly::new();

		assembly.start(4, 1);
		assert!(assembly.building());
		assert!(assembly.put(0, b"hello"), "one of one finishes it");
		assembly.finish();
		assert!(!assembly.building());
		assert_eq!(assembly.message(), b"hello");
	}

	#[test]
	fn the_pieces_may_arrive_in_any_order() {
		let payload = message(MAX_PAYLOAD * 3 + 11);
		let mut assembly = Assembly::new();

		assembly.start(9, 4);
		assert!(!assembly.put(2, piece(&payload, 2)));
		assert!(!assembly.put(0, piece(&payload, 0)));
		assert!(!assembly.put(3, piece(&payload, 3)), "the tail is not the last to arrive");
		assert!(assembly.put(1, piece(&payload, 1)), "the one that fills the gap finishes it");
		assert_eq!(assembly.message(), payload.as_slice(), "and every byte is where it was");
	}

	#[test]
	fn the_same_piece_twice_does_not_finish_a_message() {
		let mut assembly = Assembly::new();

		assembly.start(1, 2);
		assert!(!assembly.put(0, &[7; MAX_PAYLOAD]));
		assert!(!assembly.put(0, &[7; MAX_PAYLOAD]), "a repeat says nothing new");
		assert!(assembly.put(1, b"tail"));
	}

	#[test]
	fn a_message_that_divides_exactly_comes_back_at_the_right_length() {
		// the length is read off the last piece, so a last piece that is
		// exactly full is the case to get wrong.
		let payload = message(MAX_PAYLOAD * 2);
		let mut assembly = Assembly::new();

		assembly.start(1, 2);
		assert!(!assembly.put(0, piece(&payload, 0)));
		assert!(assembly.put(1, piece(&payload, 1)));
		assert_eq!(assembly.message().len(), MAX_PAYLOAD * 2);
		assert_eq!(assembly.message(), payload.as_slice());
	}

	#[test]
	fn the_largest_message_there_is_goes_through_whole() {
		let payload = message(MAX_MESSAGE);
		let mut assembly = Assembly::new();

		assembly.start(1, 64);

		for index in 0..64_u16 {
			let last = assembly.put(index, piece(&payload, usize::from(index)));

			assert_eq!(last, index == 63, "only the sixty-fourth finishes it");
		}

		assert_eq!(assembly.message(), payload.as_slice());
	}

	#[test]
	fn starting_a_new_message_throws_away_whatever_was_half_built() {
		let mut assembly = Assembly::new();

		assembly.start(1, 2);
		assert!(!assembly.put(0, &[1; MAX_PAYLOAD]));
		assembly.start(2, 2);
		assert_eq!(assembly.sequence(), 2);
		assert!(!assembly.put(1, b"tail"), "the piece kept from the old one does not count");
		assert!(assembly.put(0, &[2; MAX_PAYLOAD]), "and the new one needs both of its own");
		assert_eq!(assembly.message()[0], 2, "with the new bytes rather than the old");
	}

	#[test]
	fn a_piece_count_that_cannot_be_right_leaves_it_holding_nothing() {
		// nothing arriving through a channel reaches this, because the head is
		// checked first. It is checked again here because this is where the
		// arithmetic that depends on it lives.
		let mut assembly = Assembly::new();

		assembly.start(1, 0);
		assert!(!assembly.building(), "no message has no pieces");
		assert!(!assembly.put(0, b"x"));

		assembly.start(1, 65);
		assert!(!assembly.building(), "and none has more than sixty-four");
		assert!(!assembly.put(0, b"x"));
	}

	#[test]
	fn a_piece_outside_the_count_or_too_long_is_ignored() {
		let mut assembly = Assembly::new();

		assembly.start(1, 2);
		assert!(!assembly.put(2, b"x"), "there is no third piece of two");

		// past the whole ceiling rather than merely past this message's count:
		// the room set aside is sixty-four pieces and the flags are sixty-four
		// bits, so a larger index is a write outside both.
		assert!(!assembly.put(64, b"x"), "nor a sixty-fifth of anything");
		assert!(!assembly.put(u16::MAX, b"x"), "however far outside it is");
		assert!(!assembly.put(0, &[0; MAX_PAYLOAD + 1]), "nor a piece bigger than a datagram");
		assert!(!assembly.put(0, &[0; MAX_PAYLOAD]), "and the real one still leaves one missing");
	}

	#[test]
	fn the_flags_cover_exactly_as_many_pieces_as_there_are() {
		assert_eq!(mask(1), 1, "one piece is one bit");
		assert_eq!(mask(2), 0b11);
		assert_eq!(mask(64), u64::MAX, "and sixty-four fills the word");
		assert_eq!(mask(0), 0, "nothing is not a message");
		assert_eq!(mask(65), 0, "and neither is more than there is room for");
	}
}
