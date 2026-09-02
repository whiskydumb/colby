//! The importer for a `.lua`, which is almost nothing and is deliberately a
//! module anyway.
//!
//! Every other source kind is turned into something else by hundreds of lines -
//! a font into a distance field, a picture into mip levels, a model into a
//! tree of assets. A program is turned into itself, so the whole of the work
//! here is the one thing that has to happen offline rather than at load time:
//! stripping a byte order mark.
//!
//! **The mark is not a hypothetical.** Every editor and every shell on this
//! platform will happily write one at the head of a text file, and to an
//! interpreter it is not whitespace - it is a character before the first
//! character, so the file fails to compile with a message pointing at line one
//! and nothing visibly wrong on line one. The console's own tokenizer learned
//! the same lesson from a hand-written `cvars.cfg`.
//!
//! It is a module rather than a line inside the compiler because the pair is
//! the convention: a source extension lives with its importer and an output
//! extension lives with its format, and a script that ever grows a real check -
//! a syntax pass, say, which would need an interpreter the compiler does not
//! link - has an obvious place to grow it.

/// The extension the importer reads.
pub const EXTENSION: &str = "lua";

/// What a text file that starts with a byte order mark starts with.
const MARK: char = '\u{feff}';

/// Cleans up one program on its way into a `.clua`.
///
/// @param text - the file, as it was written
/// @return the program the interpreter should be handed
#[must_use]
pub fn import(text: &str) -> String { text.strip_prefix(MARK).unwrap_or(text).to_owned() }

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_program_is_itself() {
		let source = "local a = 1\nfunction tick(dt) a = a + dt end\n";

		assert_eq!(import(source), source);
	}

	#[test]
	fn a_byte_order_mark_is_taken_off_the_front() {
		// what an editor on this platform leaves behind, and what an
		// interpreter reads as a character before the first character.
		let source = format!("{MARK}local a = 1");

		assert_eq!(import(&source), "local a = 1");
		assert!(!import(&source).starts_with(MARK), "and only the front");
	}

	#[test]
	fn only_the_first_one_is_a_mark_and_the_rest_are_the_programs_business() {
		// the second one is inside a string literal as far as anything here
		// can tell, and a compiler that quietly edited the middle of a file
		// would be a compiler nobody could debug against.
		let source = format!("{MARK}local a = \"{MARK}\"");

		assert_eq!(import(&source), format!("local a = \"{MARK}\""));
	}

	#[test]
	fn a_program_with_nothing_in_it_stays_that_way() {
		assert_eq!(import(""), "");
		assert_eq!(import(&MARK.to_string()), "");
	}
}
