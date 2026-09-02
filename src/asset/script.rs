//! colby's runtime script format: `.clua`.
//!
//! The twin of [`document`](crate::document), and for the same reason: a
//! program is kilobytes of text, and a compiled form that is not text buys
//! nothing measurable while costing the ability to open the file and read it.
//! So a `.clua` is a versioned header and the program itself.
//!
//! ```text
//!   0  MAGIC                            8 bytes
//!   8  version                          4 bytes
//!  12  flags                            4 bytes
//!  16  the program, UTF-8
//! ```
//!
//! **Not bytecode**, which is the one alternative worth naming. Shipping
//! bytecode saves a parse per program per load, which for something this size
//! is not measurable; against that it makes the file unreadable, ties the asset
//! tree to one interpreter's build, and makes an asset that a different build
//! of the same engine cannot load. The trigger to revisit is a measured load
//! cost, and there is nowhere near enough script in the tree to produce one.
//!
//! What the compiler *does* do offline is the part that has to happen offline:
//! it strips a byte order mark, which anything that writes a file on Windows is
//! liable to leave at the head and which the interpreter reads as a syntax
//! error a line before the first line. @ref [`lua`](crate::lua).

use std::path::Path;

use colby_core::{Result, abi::ScriptData, err};

/// The eight bytes every `.clua` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYLUA";

/// The revision of everything in this module.
pub const FORMAT_VERSION: u32 = 1;

/// The extension a compiled program is written with.
pub const EXTENSION: &str = "clua";

/// How big the header is, and where the program starts.
pub const HEADER_BYTES: usize = 16;

/// The largest program the reader will accept, in bytes.
///
/// A program is kilobytes. This is how wrong a file has to be before the reader
/// stops rather than reading what it was handed - the same ceiling a document
/// has, for the same reason.
pub const MAX_BYTES: usize = 4 << 20;

/// A `.clua` read off disk and checked.
#[derive(Clone, Debug)]
pub struct ScriptFile {
	source: String,
}

impl ScriptFile {
	/// Reads and checks a compiled program.
	pub fn open(path: &Path) -> Result<Self> {
		let bytes = std::fs::read(path)?;

		Self::from_bytes(&bytes).map_err(|error| err!(Asset("{}: {error}", path.display())))
	}

	/// Checks bytes that are already in memory.
	pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
		if bytes.len() > MAX_BYTES {
			return Err(err!(Asset(
				"is {} bytes, and this build reads at most {MAX_BYTES}",
				bytes.len()
			)));
		}

		let head = bytes
			.get(..HEADER_BYTES)
			.ok_or_else(|| err!(Asset("is too short to hold a {HEADER_BYTES}-byte header")))?;

		if head.get(..MAGIC.len()) != Some(&MAGIC[..]) {
			return Err(err!(Asset("is not a colby script")));
		}

		let version: [u8; 4] = head
			.get(8..12)
			.and_then(|slice| slice.try_into().ok())
			.unwrap_or([0; 4]);
		let version = u32::from_le_bytes(version);

		if version != FORMAT_VERSION {
			return Err(err!(Asset(
				"was written by asset format version {version}, and this build reads version \
				 {FORMAT_VERSION}; run `just assets --force` to recompile it"
			)));
		}

		let flags: [u8; 4] = head
			.get(12..16)
			.and_then(|slice| slice.try_into().ok())
			.unwrap_or([0; 4]);

		if u32::from_le_bytes(flags) != 0 {
			return Err(err!(Asset("sets flag bits this build does not know about")));
		}

		let body = bytes.get(HEADER_BYTES..).unwrap_or_default();
		let source = std::str::from_utf8(body)
			.map_err(|error| err!(Asset("is not valid UTF-8: {error}")))?;

		Ok(Self { source: source.to_owned() })
	}

	/// The program, as text.
	#[must_use]
	pub fn source(&self) -> &str { &self.source }

	/// The program as the table wants it.
	#[must_use]
	pub fn to_script_data(&self) -> ScriptData { ScriptData { source: self.source.clone() } }
}

/// Wraps a program in a header.
///
/// @param source - the program, already cleaned up by the importer
#[must_use]
pub fn encode(source: &str) -> Vec<u8> {
	let mut out = Vec::with_capacity(HEADER_BYTES + source.len());
	out.extend_from_slice(&MAGIC);
	out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
	out.extend_from_slice(&0_u32.to_le_bytes());
	out.extend_from_slice(source.as_bytes());

	out
}

/// The format version a file on disk was written by.
#[must_use]
pub fn version_of(path: &Path) -> Option<u32> {
	let mut head = [0_u8; 12];
	let mut file = std::fs::File::open(path).ok()?;
	std::io::Read::read_exact(&mut file, &mut head).ok()?;

	if head.get(..MAGIC.len()) != Some(&MAGIC[..]) {
		return None;
	}

	let version: [u8; 4] = head.get(8..12)?.try_into().ok()?;

	Some(u32::from_le_bytes(version))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_program_survives_the_round_trip_as_text() {
		let source = "-- a comment\nlocal a = 1\nfunction tick(dt) a = a + dt end\n";
		let file = ScriptFile::from_bytes(&encode(source)).expect("it reads back");

		assert_eq!(file.source(), source);
		assert_eq!(file.to_script_data().source, source);
	}

	#[test]
	fn a_file_of_another_format_is_refused_by_name() {
		let error = ScriptFile::from_bytes(&crate::document::encode("<div></div>"))
			.expect_err("a document is not a program");

		assert!(
			error.to_string().contains("not a colby script"),
			"saying which it is not: {error}"
		);
	}

	#[test]
	fn a_file_from_another_version_says_how_to_fix_it() {
		let mut bytes = encode("local a = 1");
		bytes.splice(8..12, 99_u32.to_le_bytes());

		let error = ScriptFile::from_bytes(&bytes).expect_err("this build does not read 99");

		assert!(error.to_string().contains("just assets"), "and says what to run: {error}");
	}

	#[test]
	fn a_flag_bit_nobody_knows_is_refused_rather_than_ignored() {
		// the opposite of the rule a `.cscene` record follows, and the reason
		// is the same one that distinguishes a code from a flag there: this
		// header has no smaller shape to fall back to, so a bit set here means
		// the body is not what this build thinks it is.
		let mut bytes = encode("local a = 1");
		bytes.splice(12..16, 1_u32.to_le_bytes());

		let error = ScriptFile::from_bytes(&bytes).expect_err("nothing sets a flag yet");

		assert!(error.to_string().contains("flag"), "naming what it met: {error}");
	}

	#[test]
	fn a_header_that_is_cut_short_is_refused_rather_than_read_past() {
		let bytes = encode("local a = 1");
		let error =
			ScriptFile::from_bytes(&bytes[..8]).expect_err("half a header is not a header");

		assert!(error.to_string().contains("too short"), "saying so: {error}");
	}

	#[test]
	fn a_program_that_is_not_text_is_refused() {
		let mut bytes = encode("");
		bytes.push(0xFF);

		let error = ScriptFile::from_bytes(&bytes).expect_err("0xFF starts no UTF-8 sequence");

		assert!(error.to_string().contains("UTF-8"), "saying why: {error}");
	}

	#[test]
	fn the_version_is_readable_without_reading_the_program() {
		let path = std::env::temp_dir().join("colby-clua-version.clua");

		std::fs::write(&path, encode("local a = 1")).expect("the temp directory is writable");

		assert_eq!(version_of(&path), Some(FORMAT_VERSION));
		assert_eq!(
			version_of(&std::env::temp_dir().join("colby-clua-nothing.clua")),
			None,
			"and a file that is not there has no version"
		);

		std::fs::remove_file(&path).ok();
	}
}
