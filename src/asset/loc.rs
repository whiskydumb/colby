//! colby's runtime translation format: `.cloc`, and the `.loc` it comes from.
//!
//! The twin of [`script`](crate::script), and the argument is the same one:
//! a language is kilobytes of text, and a compiled form that is not text buys
//! nothing measurable while costing the ability to open the file and read it.
//! So a `.cloc` is a versioned header and the pairs, one to a line.
//!
//! ```text
//!   0  MAGIC                            8 bytes
//!   8  version                          4 bytes
//!  12  flags                            4 bytes
//!  16  the pairs, UTF-8, one a line: key, a tab, the words
//! ```
//!
//! **What compiling buys is a sorted, checked table**, which is the same trade
//! a `.cmat` makes: the reader binary-searches without sorting, a key written
//! twice is refused where it is written rather than silently answered by one
//! of the two, and a byte order mark is gone before anything sees it. The
//! source is a flat JSON object because that is the shape every reference with
//! a translation file uses, and colby already reads JSON for `.material` and
//! `.scene`.
//!
//! ```json
//! {
//!   "menu.play": "Play",
//!   "menu.quit": "Quit"
//! }
//! ```
//!
//! **The language is the file's name, not a field inside it.**
//! `assets/lang/ru.loc` compiles to `lang/ru`, and the `loc.language` variable
//! holding `ru` names the tail of that. A field would be a second place for
//! the same fact to be written, and the two would disagree the first time
//! somebody copied a file.
//!
//! **One file a language**, and that is a decision worth its sentence: s&box
//! reads every `*.json` under a language's directory and merges them, because
//! each of its addons ships its own. colby has no addons, so one file is one
//! language and merging is machinery with no case behind it yet.
//!
//! A tab separates a key from its words because a key never holds one - the
//! compiler refuses one that does - and a newline inside either is written the
//! way JSON writes it and kept escaped in the file.

use std::path::Path;

use colby_core::{
	Result,
	abi::{LangData, loc::SIGIL},
	err,
};

use crate::json::{self, Value};

/// The eight bytes every `.cloc` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYLOC";

/// The revision of everything in this module.
pub const FORMAT_VERSION: u32 = 1;

/// The extension a source translation is written with.
pub const SOURCE_EXTENSION: &str = "loc";

/// The extension a compiled one is written with.
pub const EXTENSION: &str = "cloc";

/// How big the header is, and where the pairs start.
pub const HEADER_BYTES: usize = 16;

/// The largest table the reader will accept, in bytes.
///
/// A language is kilobytes. The same ceiling a program and a document have, and
/// for the same reason: this is how wrong a file has to be before the reader
/// stops rather than reading what it was handed.
pub const MAX_BYTES: usize = 4 << 20;

/// What separates a key from its words in the compiled file.
const SEPARATOR: char = '\t';

/// Reads a `.loc` and gives back the table it describes.
///
/// @param text - the whole source file
/// @return the pairs, sorted and checked
pub fn import(text: &str) -> Result<LangData> {
	let value = json::parse(text)?;
	let Value::Object(fields) = &value else {
		return Err(err!(Asset(
			"is not a flat object of keys and words; a translation is `{{ \"menu.play\": \
			 \"Play\" }}`"
		)));
	};

	let mut strings = Vec::with_capacity(fields.len());

	for (key, words) in fields {
		let Some(words) = words.as_str() else {
			return Err(err!(Asset(
				"{key:?} stands for something that is not a string; every value in a \
				 translation is the words a person reads"
			)));
		};

		check_key(key)?;
		strings.push((key.clone(), words.to_owned()));
	}

	strings.sort_by(|left, right| left.0.cmp(&right.0));

	// after the sort, so the two that clash are neighbors. Refused rather than
	// resolved: JSON readers disagree about which of two wins, and a
	// translation with one key twice is a mistake somebody wants told about
	// rather than a shape with a meaning.
	if let Some(pair) = strings
		.windows(2)
		.find(|pair| pair[0].0 == pair[1].0)
	{
		return Err(err!(Asset(
			"writes the key {:?} twice, and a translation may say one thing a key",
			pair[0].0
		)));
	}

	Ok(LangData { strings })
}

/// Whether a key is one this format can write down.
fn check_key(key: &str) -> Result<()> {
	if key.is_empty() {
		return Err(err!(Asset("has an empty key, which nothing could ever ask for")));
	}

	if key.contains(SEPARATOR) || key.contains('\n') {
		return Err(err!(Asset(
			"has a key holding a tab or a newline ({key:?}), and a compiled table separates the \
			 two halves of a pair with a tab"
		)));
	}

	if key.starts_with(SIGIL) {
		return Err(err!(Asset(
			"has a key beginning with a sigil ({key:?}); the sigil marks a lookup where the \
			 words are used and is not part of the key"
		)));
	}

	Ok(())
}

/// A `.cloc` read off disk and checked.
#[derive(Clone, Debug)]
pub struct LangFile {
	data: LangData,
}

impl LangFile {
	/// Reads and checks a compiled translation.
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
			return Err(err!(Asset("is not a colby translation")));
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
		let text = std::str::from_utf8(body)
			.map_err(|error| err!(Asset("is not valid UTF-8: {error}")))?;

		Ok(Self { data: decode(text)? })
	}

	/// The table.
	#[must_use]
	pub const fn data(&self) -> &LangData { &self.data }

	/// The table, to hand to the registry.
	#[must_use]
	pub fn to_lang_data(&self) -> LangData { self.data.clone() }
}

/// Turns the body of a compiled file back into pairs.
///
/// **Checked rather than trusted, even though the compiler wrote it.** The
/// order is what makes the lookup a binary search, and a file edited by hand or
/// truncated by a full disk would otherwise answer some keys and not others -
/// which is the kind of wrong that looks like a missing translation.
fn decode(text: &str) -> Result<LangData> {
	let mut strings = Vec::new();

	for line in text.lines() {
		if line.is_empty() {
			continue;
		}

		let (key, words) = line.split_once(SEPARATOR).ok_or_else(|| {
			err!(Asset("holds a line with no tab in it, so it is not a pair: {line:?}"))
		})?;

		strings.push((key.to_owned(), unescaped(words)));
	}

	if strings
		.windows(2)
		.any(|pair| pair[0].0 >= pair[1].0)
	{
		return Err(err!(Asset(
			"is not in key order, and a reader binary searches it; run `just assets --force`"
		)));
	}

	Ok(LangData { strings })
}

/// Wraps a table in a header.
///
/// @param data - the pairs, already sorted and checked by [`import`]
#[must_use]
pub fn encode(data: &LangData) -> Vec<u8> {
	let mut body = String::new();

	for (key, words) in &data.strings {
		body.push_str(key);
		body.push(SEPARATOR);
		body.push_str(&escaped(words));
		body.push('\n');
	}

	let mut out = Vec::with_capacity(HEADER_BYTES + body.len());
	out.extend_from_slice(&MAGIC);
	out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
	out.extend_from_slice(&0_u32.to_le_bytes());
	out.extend_from_slice(body.as_bytes());

	out
}

/// The words with the two characters a line cannot hold spelled out.
///
/// Only a newline and a backslash: a tab in the *words* is fine, because the
/// split takes the first one and a key may not hold any.
fn escaped(words: &str) -> String {
	let mut out = String::with_capacity(words.len());

	for letter in words.chars() {
		match letter {
			| '\\' => out.push_str("\\\\"),
			| '\n' => out.push_str("\\n"),
			| _ => out.push(letter),
		}
	}

	out
}

/// The inverse of [`escaped`].
fn unescaped(words: &str) -> String {
	if !words.contains('\\') {
		return words.to_owned();
	}

	let mut out = String::with_capacity(words.len());
	let mut letters = words.chars();

	while let Some(letter) = letters.next() {
		if letter != '\\' {
			out.push(letter);

			continue;
		}

		match letters.next() {
			| Some('n') => out.push('\n'),
			// a doubled one, and a lone one at the very end - which `escaped`
			// never writes, so it is a person's and stays one backslash.
			| Some('\\') | None => out.push('\\'),
			// a backslash before anything else was never written by `escaped`
			// either, and is kept exactly as typed rather than eaten.
			| Some(other) => {
				out.push('\\');
				out.push(other);
			},
		}
	}

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
	fn a_flat_object_becomes_pairs_in_key_order() {
		let data = import(r#"{ "menu.quit": "Quit", "menu.play": "Play" }"#).expect("it reads");

		assert_eq!(
			data.strings,
			vec![
				("menu.play".to_owned(), "Play".to_owned()),
				("menu.quit".to_owned(), "Quit".to_owned()),
			],
			"sorted by the compiler, because the reader binary searches"
		);
	}

	#[test]
	fn a_table_survives_the_round_trip() {
		let data = import(r#"{ "a": "one", "b": "two" }"#).expect("it reads");
		let file = LangFile::from_bytes(&encode(&data)).expect("it reads back");

		assert_eq!(file.data(), &data);
		assert_eq!(file.to_lang_data().get("b"), Some("two"));
	}

	#[test]
	fn words_holding_a_newline_survive_it_too() {
		let data = import("{ \"a\": \"one\\ntwo\", \"b\": \"back\\\\slash\" }")
			.expect("both escapes are JSON's");
		let file = LangFile::from_bytes(&encode(&data)).expect("it reads back");

		assert_eq!(
			file.data().get("a"),
			Some("one\ntwo"),
			"a newline would otherwise end the pair halfway"
		);
		assert_eq!(file.data().get("b"), Some("back\\slash"));
	}

	#[test]
	fn cyrillic_words_survive_as_themselves() {
		// the whole point of the feature, and the one assertion that would
		// catch a byte-wise round trip that is not character-wise.
		let data = import("{ \"menu.play\": \"Играть\" }").expect("it reads");
		let file = LangFile::from_bytes(&encode(&data)).expect("it reads back");

		assert_eq!(file.data().get("menu.play"), Some("Играть"));
	}

	#[test]
	fn a_key_written_twice_is_refused_rather_than_resolved() {
		let error = import(r#"{ "a": "one", "a": "two" }"#).expect_err("one key, one meaning");

		assert!(error.to_string().contains("twice"), "saying which: {error}");
	}

	#[test]
	fn a_value_that_is_not_a_string_is_refused() {
		let error = import(r#"{ "a": 3 }"#).expect_err("a number is not words");

		assert!(error.to_string().contains("not a string"), "saying so: {error}");
	}

	#[test]
	fn something_that_is_not_an_object_is_refused_with_the_shape_it_wanted() {
		let error = import("[1, 2]").expect_err("an array is not a translation");

		assert!(error.to_string().contains("menu.play"), "showing one: {error}");
	}

	#[test]
	fn a_key_that_could_not_be_written_down_is_refused_where_it_is_written() {
		assert!(
			import("{ \"a\\tb\": \"words\" }")
				.expect_err("a tab is the separator")
				.to_string()
				.contains("tab"),
		);
		assert!(
			import(r#"{ "": "words" }"#)
				.expect_err("nothing could ask for it")
				.to_string()
				.contains("empty key"),
		);
		assert!(
			import(r##"{ "#menu.play": "words" }"##)
				.expect_err("the sigil marks the use, not the key")
				.to_string()
				.contains("sigil"),
		);
	}

	#[test]
	fn a_compiled_file_out_of_key_order_is_refused_rather_than_half_searched() {
		// the failure this is here for: a binary search over an unsorted table
		// answers some keys and not others, which reads exactly like a
		// translation somebody forgot to finish.
		let mut bytes = encode(&LangData::default());
		bytes.extend_from_slice(b"b\ttwo\na\tone\n");

		let error = LangFile::from_bytes(&bytes).expect_err("b comes after a");

		assert!(error.to_string().contains("key order"), "saying so: {error}");
	}

	#[test]
	fn a_line_with_no_tab_is_refused() {
		let mut bytes = encode(&LangData::default());
		bytes.extend_from_slice(b"not a pair\n");

		let error = LangFile::from_bytes(&bytes).expect_err("there is no separator");

		assert!(error.to_string().contains("no tab"), "saying so: {error}");
	}

	#[test]
	fn a_file_of_another_format_is_refused_by_name() {
		let error = LangFile::from_bytes(&crate::script::encode("local a = 1"))
			.expect_err("a program is not a translation");

		assert!(
			error
				.to_string()
				.contains("not a colby translation"),
			"saying which it is not: {error}"
		);
	}

	#[test]
	fn a_file_from_another_version_says_how_to_fix_it() {
		let mut bytes = encode(&LangData::default());
		bytes.splice(8..12, 99_u32.to_le_bytes());

		let error = LangFile::from_bytes(&bytes).expect_err("this build does not read 99");

		assert!(error.to_string().contains("just assets"), "and says what to run: {error}");
	}

	#[test]
	fn a_flag_bit_nobody_knows_is_refused_rather_than_ignored() {
		let mut bytes = encode(&LangData::default());
		bytes.splice(12..16, 1_u32.to_le_bytes());

		let error = LangFile::from_bytes(&bytes).expect_err("nothing sets a flag yet");

		assert!(error.to_string().contains("flag"), "naming what it met: {error}");
	}

	#[test]
	fn the_version_is_readable_without_reading_the_table() {
		let path = std::env::temp_dir().join("colby-cloc-version.cloc");

		std::fs::write(&path, encode(&LangData::default())).expect("the temp dir is writable");

		assert_eq!(version_of(&path), Some(FORMAT_VERSION));
		assert_eq!(
			version_of(&std::env::temp_dir().join("colby-cloc-nothing.cloc")),
			None,
			"and a file that is not there has no version"
		);

		std::fs::remove_file(&path).ok();
	}
}
