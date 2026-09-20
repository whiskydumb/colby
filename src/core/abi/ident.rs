//! An asset's identity: the number, and the way it is written down.
//!
//! **Only the half the engine needs is here.** Where an identity comes from -
//! how one is born out of a project and a name, where it is kept beside a
//! source, how a whole tree of them is read back - is the compiler's, and lives
//! in `colby_asset::ident`, which re-exports everything below so there is one
//! `Id` and not two. What is here is what a *running* engine has to be able to
//! do: hold one in a registry entry, compare it, and read or write the
//! thirteen letters a person copies out of a panel.
//!
//! ```text
//!   id://abcdefghijklm      the whole of how one is written
//! ```
//!
//! An identity answers the question a name cannot: **which asset is this**,
//! across a rename. A name is an address - rename the file and every reference
//! by name points at nothing - and a registry entry carries both, so that the
//! asset loop can move an entry to a new name rather than appending a second
//! one, and every handle the world is holding stays valid. @ref
//! [`Registry::adopt`](super::registry::Registry::adopt).

use std::fmt::{self, Write as _};

/// What an id is spelled with, in a file a person writes.
///
/// A scheme rather than a second key beside every reference: one field then
/// takes either spelling, a name cannot be mistaken for an id, and an id cannot
/// be mistaken for a name - an asset name is built out of path components, and
/// a colon is not a character a file name may hold on the platform this is
/// developed on.
pub const SCHEME: &str = "id://";

/// The digits an id is spelled with: lowercase letters and the digits that no
/// letter is easily read as.
///
/// Thirty-two of them, so each stands for five bits and the whole of an id is
/// thirteen of them.
const ALPHABET: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";

/// How many digits there are, as an id is read and written with.
const RADIX: u64 = 32;

/// How many digits an id is spelled with.
///
/// Thirteen at five bits each is sixty-five, which covers the sixty-three an id
/// holds with room to spare in the first digit. Fixed width rather than the
/// shortest that fits, so that ids sort and compare as text exactly as they do
/// as numbers.
pub const DIGITS: usize = 13;

/// Every bit an id may use: all but the topmost.
///
/// The top bit is left clear so that an id is a positive number in every
/// language that may one day have to read the table, and so that the spelling
/// never needs a fourteenth digit.
const MASK: u64 = u64::MAX >> 1;

/// An asset's identity.
///
/// Sixty-three bits, of which zero means "no id at all" - the same convention
/// the registries use for slot zero, and for the same reason: something that
/// was never set resolves to a value that is harmless rather than to an
/// absence that has to be spelled everywhere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Id(u64);

impl Id {
	/// No asset at all.
	pub const NONE: Self = Self(0);

	/// The number behind it.
	#[must_use]
	pub const fn number(self) -> u64 { self.0 }

	/// Whether this names nothing.
	#[must_use]
	pub const fn is_none(self) -> bool { self.0 == 0 }

	/// The identity a run of bits stands for, with the top one dropped.
	///
	/// The one way to make an id out of a number, so that the bit nothing may
	/// use is cleared in exactly one place. What comes out may be
	/// [`NONE`](Self::NONE), which is a caller's business: whoever is drawing
	/// one counts that as taken and draws again.
	///
	/// @param bits - whatever a draw produced
	#[must_use]
	pub const fn from_bits(bits: u64) -> Self { Self(bits & MASK) }

	/// Whether a written reference is spelled as an id rather than as a name.
	///
	/// Asked before parsing, because a reference that means to be an id and is
	/// misspelled has to be an error rather than a name nothing answers to.
	///
	/// @param written - what the file said
	#[must_use]
	pub fn spelled(written: &str) -> bool { written.starts_with(SCHEME) }

	/// Reads one back from the way it is written down.
	///
	/// @param written - the whole reference, scheme and all
	/// @return the id, or nothing when it is not one this build can read
	#[must_use]
	pub fn parse(written: &str) -> Option<Self> {
		let body = written.strip_prefix(SCHEME)?;

		if body.len() != DIGITS {
			return None;
		}

		let mut held: u64 = 0;

		for byte in body.bytes() {
			let digit = ALPHABET.iter().position(|it| *it == byte)?;

			held = held
				.checked_mul(RADIX)?
				.checked_add(u64::try_from(digit).ok()?)?;
		}

		(held != 0 && held <= MASK).then_some(Self(held))
	}

	/// Which digit stands at a place, counting up from the last.
	fn digit(self, place: usize) -> char {
		let shifted = (self.0 >> (place * 5)) % RADIX;
		let index = usize::try_from(shifted).unwrap_or_default();

		char::from(ALPHABET[index])
	}
}

impl fmt::Display for Id {
	/// The way an id is written down: the scheme and thirteen digits.
	fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
		out.write_str(SCHEME)?;

		for place in (0..DIGITS).rev() {
			out.write_char(self.digit(place))?;
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn an_id_reads_back_the_way_it_was_written() {
		for bits in [1_u64, 2, 0x1234_5678_9ABC_DEF0, u64::MAX, u64::MAX >> 1] {
			let id = Id::from_bits(bits);
			let written = id.to_string();

			assert_eq!(written.len(), SCHEME.len() + DIGITS, "{written} is the width it is");
			assert!(written.starts_with(SCHEME), "{written} carries the scheme");
			assert_eq!(Id::parse(&written), Some(id), "{written} reads back");
		}
	}

	#[test]
	fn the_top_bit_is_never_part_of_one() {
		assert_eq!(Id::from_bits(u64::MAX).number(), MASK, "the topmost is dropped");
		assert_eq!(
			Id::from_bits(u64::MAX),
			Id::from_bits(MASK),
			"so two draws a bit apart up there are one id"
		);
	}

	#[test]
	fn nothing_is_the_number_nought_and_says_so() {
		assert!(Id::NONE.is_none(), "the one that names nothing");
		assert_eq!(Id::NONE.number(), 0);
		assert_eq!(Id::default(), Id::NONE, "which is what an unset one is");
		assert!(!Id::from_bits(1).is_none());
	}

	#[test]
	fn a_spelling_this_build_does_not_read_is_refused() {
		let good = Id::from_bits(0x0123_4567_89AB_CDEF).to_string();

		assert!(Id::parse(&good).is_some(), "the one it writes itself is read");
		assert!(Id::spelled(&good), "and it is spelled as an id");
		assert!(!Id::spelled("meshes/crystal"), "where a name is not");

		let bare = good.trim_start_matches(SCHEME).to_owned();
		let shouted = good.to_uppercase();

		for bad in [
			"meshes/crystal",
			"id://",
			"id://abcdefghijkl",
			"id://abcdefghijklmn",
			"id://abcdefghijkl!",
			// 0, 1, 8 and 9 are not digits an id is spelled with
			"id://abcdefghijkl0",
			"id://abcdefghijkl9",
			// nought is no identity, and every bit set is one more than
			// sixty-three bits hold
			"id://aaaaaaaaaaaaa",
			"id://7777777777777",
			shouted.as_str(),
			bare.as_str(),
		] {
			assert_eq!(Id::parse(bad), None, "{bad} is not an id");
		}
	}
}
