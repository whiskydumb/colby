//! colby's runtime document format: `.cdoc`.
//!
//! The one compiled format that is **not** designed to be read rather than
//! parsed, and that is deliberate. A `.cmesh` holds bytes the GPU takes as they
//! are because parsing an OBJ at load time is slow and decoding one is a
//! dependency; a document is two kilobytes of text, and a tree with strings in
//! it serializes to something nobody can open in an editor for no measurable
//! gain. So a `.cdoc` is a versioned header and the document itself.
//!
//! ```text
//!   0  MAGIC                            8 bytes
//!   8  version                          4 bytes
//!  12  flags                            4 bytes
//!  16  the document, UTF-8
//! ```
//!
//! What the compiler *does* do is the part that has to happen offline: it
//! resolves every `<link href>` against the source tree and folds those
//! stylesheets into the head of the file. One output, no second file to find at
//! load time, and no question about what happens when one is missing - that is
//! a compile error naming it. It also parses the whole thing once, so that a
//! misspelled property is a line in the log at compile time rather than a box
//! that is silently the wrong size.
//!
//! A `.css` is therefore **not an asset**: it is an input to one, which is what
//! makes editing a shared stylesheet recompile every document naming it instead
//! of changing the picture only for whichever of them somebody happens to touch
//! next.
//!
//! **A `.lua` is not folded in, and is an asset of its own.** The two cases
//! look alike and are not: a stylesheet has to be folded because rules of equal
//! weight are applied in the order they were written, so where a sheet lands in
//! the file is part of what it means, while a program means the same thing
//! wherever it sits. Making it an asset buys three things folding cannot - a
//! program several documents share reloads once instead of rebuilding all of
//! them, editing a stylesheet stops restarting a program that did not change,
//! and a program belonging to no document has somewhere to live.
//! `<script src="ui/hud">` therefore names it the way `<img src>` names a
//! texture. @ref [`script`](crate::script).
//!
//! Folding a linked sheet in as a `<style>` block *before* the document keeps
//! the cascade honest: rules of equal weight are applied in the order they were
//! written, so a document's own `<style>` still overrides the sheet it shares
//! with everything else.

use std::path::Path;

use colby_core::{Result, abi::ui::DocumentData, err, warn};

use crate::{compile, html};

/// The eight bytes every `.cdoc` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYDOC";

/// The revision of everything in this module.
///
/// Two since a program stopped being folded in: a file written by version one
/// holds its program as a `<script>` block with a body, which this build
/// refuses outright rather than reads differently, so the version has to move
/// or every document already in an output tree fails to load with a message
/// about markup.
pub const FORMAT_VERSION: u32 = 2;

/// The extension a compiled document is written with.
pub const EXTENSION: &str = "cdoc";

/// How big the header is, and where the document starts.
pub const HEADER_BYTES: usize = 16;

/// The largest document the reader will accept, in bytes.
///
/// An interface is kilobytes. This is how wrong a file has to be before the
/// reader stops rather than reading what it was handed.
pub const MAX_BYTES: usize = 4 << 20;

/// A `.cdoc` read off disk and checked.
#[derive(Clone, Debug)]
pub struct DocumentFile {
	text: String,
}

impl DocumentFile {
	/// Reads and checks a compiled document.
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
			return Err(err!(Asset("is not a colby document")));
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

		Ok(Self { text: text.to_owned() })
	}

	/// The document, as text.
	#[must_use]
	pub fn text(&self) -> &str { &self.text }

	/// Parses it into a tree and a set of rules.
	///
	/// Every `<link>` was already folded in by the compiler, so nothing is
	/// linked here and nothing is read off disk.
	pub fn to_document_data(&self) -> Result<DocumentData> {
		Ok(html::parse(&self.text, &[])?.document)
	}
}

/// Wraps a document in a header.
///
/// @param text - the document, with its linked stylesheets already folded in
#[must_use]
pub fn encode(text: &str) -> Vec<u8> {
	let mut out = Vec::with_capacity(HEADER_BYTES + text.len());
	out.extend_from_slice(&MAGIC);
	out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
	out.extend_from_slice(&0_u32.to_le_bytes());
	out.extend_from_slice(text.as_bytes());

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

/// Every stylesheet a document links to, as paths beside it.
///
/// @param source - the `.html`, in the source tree
/// @param text - its contents
/// @param root - the source tree, which a link may not leave
/// @return one path per `<link href>`, in order
#[must_use]
pub fn stylesheets(source: &Path, text: &str, root: &Path) -> Vec<std::path::PathBuf> {
	beside(source, root, &html::links(text))
}

/// Resolves what a document names against its own directory, the way a browser
/// would.
///
/// Anything that climbs out of the source tree is dropped rather than followed:
/// an asset that reads a file from somewhere else is not something the
/// compiler's staleness check can see. @ref [`compile::within`], which is the
/// rule.
///
/// @param source - the `.html`, in the source tree
/// @param root - the source tree, which a reference may not leave
/// @param named - what the document asked for, as written
fn beside(source: &Path, root: &Path, named: &[String]) -> Vec<std::path::PathBuf> {
	let Some(directory) = source.parent() else {
		return Vec::new();
	};

	named
		.iter()
		.filter_map(|href| compile::within(&directory.join(href), root))
		.collect()
}

/// Reads a document and everything it links to, ready to be written out.
///
/// @param source - the `.html`, in the source tree
/// @param root - the source tree
/// @return the merged text, or why one of its parts could not be read
pub fn merge(source: &Path, root: &Path) -> Result<String> {
	let text = std::fs::read_to_string(source)?;
	let named = html::links(&text);
	let sheets = beside(source, root, &named);
	let mut merged = String::new();

	// a link that was dropped is a document that draws unstyled, and this is
	// the one moment anybody can be told: the staleness check asks for the
	// same list four times a second and is no place for a warning.
	if sheets.len() < named.len() {
		warn!(
			document = %source.display(),
			dropped = named.len() - sheets.len(),
			"a stylesheet outside the source tree is not followed"
		);
	}

	for path in sheets {
		let sheet = read_part(source, &path)?;

		merged.push_str("<style>\n");
		merged.push_str(&sheet);
		merged.push_str("\n</style>\n");
	}

	merged.push_str(&text);

	Ok(merged)
}

/// Reads one of the files a document is built out of.
///
/// @param source - the document that named it, for the error message
/// @param path - the stylesheet
fn read_part(source: &Path, path: &Path) -> Result<String> {
	std::fs::read_to_string(path).map_err(|error| {
		err!(Asset(
			"{} links to {}, which could not be read: {error}",
			source.display(),
			path.display()
		))
	})
}

#[cfg(test)]
mod tests {
	use std::{fs, path::PathBuf};

	use super::*;

	/// A directory under the temp folder, emptied first.
	fn workspace(name: &str) -> PathBuf {
		let path = std::env::temp_dir().join("colby-cdoc").join(name);

		fs::remove_dir_all(&path).ok();
		fs::create_dir_all(&path).expect("the temp directory is writable");

		path
	}

	#[test]
	fn a_document_survives_the_round_trip_as_text() {
		let source = "<div id=\"a\">hello</div>";
		let file = DocumentFile::from_bytes(&encode(source)).expect("it reads");

		assert_eq!(file.text(), source, "byte for byte");
		assert!(
			file.to_document_data()
				.expect("and parses")
				.find("a")
				.is_some(),
			"and the tree comes out of it"
		);
	}

	#[test]
	fn a_file_from_another_version_says_how_to_fix_it() {
		let mut bytes = encode("<div></div>");
		bytes[8] = 9;

		let error =
			DocumentFile::from_bytes(&bytes).expect_err("the version does not match this build");

		assert!(error.to_string().contains("--force"), "{error}");
	}

	#[test]
	fn a_file_that_is_not_one_of_these_is_refused() {
		assert!(DocumentFile::from_bytes(b"<div></div>").is_err(), "no header at all");
		assert!(DocumentFile::from_bytes(b"COLBYFNT\0\0\0\0\0\0\0\0").is_err(), "a font's magic");
	}

	#[test]
	fn a_linked_sheet_is_folded_in_ahead_of_the_document() {
		let root = workspace("linked");
		fs::write(root.join("theme.css"), ".panel { width: 10px; }").expect("it writes");
		fs::write(
			root.join("hud.html"),
			"<link rel=\"stylesheet\" href=\"theme.css\">\n<div class=\"panel\"></div>",
		)
		.expect("it writes");

		let merged = merge(&root.join("hud.html"), &root).expect("both parts read");

		assert!(merged.contains(".panel { width: 10px; }"), "the sheet is in the output");
		assert!(
			merged.find(".panel {").unwrap_or(usize::MAX) < merged.find("<div").unwrap_or(0),
			"and it comes first, so the document's own rules still win a tie"
		);
	}

	#[test]
	fn a_link_to_a_sheet_that_is_not_there_is_an_error_naming_it() {
		let root = workspace("missing");
		fs::write(root.join("hud.html"), "<link href=\"nope.css\">").expect("it writes");

		let error = merge(&root.join("hud.html"), &root).expect_err("there is no such sheet");

		assert!(error.to_string().contains("nope.css"), "and it says which: {error}");
	}

	#[test]
	fn a_link_that_climbs_out_of_the_source_tree_is_not_followed() {
		let root = workspace("escape");
		fs::write(root.join("hud.html"), "<link href=\"../secrets.css\">").expect("it writes");

		let found = stylesheets(&root.join("hud.html"), "<link href=\"../secrets.css\">", &root);

		assert!(
			found.is_empty(),
			"the compiler cannot watch a file outside the tree it walks, so following one would \
			 be a stale output nobody could explain"
		);
	}

	#[test]
	fn a_link_is_followed_when_the_source_tree_itself_is_named_by_a_climbing_path() {
		// a project opened as `--project ../elsewhere` names its tree with a
		// step up in it, and every one of its documents lost its stylesheet
		// to a guard that folded that step away on one side of the comparison
		// and not the other.
		let root = Path::new("../elsewhere/assets");
		let document = root.join("ui").join("hud.html");

		let found = stylesheets(&document, "<link href=\"theme.css\">", root);

		assert_eq!(found, vec![root.join("ui").join("theme.css")], "beside the document");
		assert!(
			stylesheets(&document, "<link href=\"../../secrets.css\">", root).is_empty(),
			"and a link that climbs out of that tree is still refused"
		);
	}

	#[test]
	fn a_document_carries_the_name_of_its_program_and_not_the_program() {
		// the whole of what a program becoming an asset changed: nothing is
		// read off disk beside the document, and what lands in the output is
		// a name for the host to look up.
		let root = workspace("script");
		fs::write(root.join("hud.lua"), "ui.on(\"a\", \"click\", function() end)")
			.expect("it writes");
		fs::write(
			root.join("hud.html"),
			"<script src=\"ui/hud\"></script>\n<div id=\"a\"></div>",
		)
		.expect("it writes");

		let merged = merge(&root.join("hud.html"), &root).expect("the document reads");
		let document = DocumentFile::from_bytes(&encode(&merged))
			.expect("it reads")
			.to_document_data()
			.expect("and parses");

		assert_eq!(document.program, "ui/hud", "the name, resolved by nobody");
		assert!(!merged.contains("ui.on"), "and the program itself was never opened: {merged}");
		assert!(document.find("a").is_some(), "and the boxes are still boxes");
	}

	#[test]
	fn a_program_beside_a_document_is_not_something_the_document_is_built_out_of() {
		// the staleness machinery used to track a `.lua` the way it tracks a
		// `.css`, and a program that is an asset is stale on its own. A
		// document that still listed one would rebuild every time somebody
		// edited a program it merely names.
		let root = workspace("not-an-input");
		fs::write(root.join("hud.lua"), "local a = 1").expect("it writes");
		let text = "<link href=\"theme.css\"><script src=\"ui/hud\"></script>";

		let inputs = stylesheets(&root.join("hud.html"), text, &root);

		assert_eq!(inputs.len(), 1, "the stylesheet and nothing else");
		assert!(inputs[0].ends_with("theme.css"), "and it is the stylesheet: {inputs:?}");
	}

	#[test]
	fn the_sheets_a_document_links_to_are_listed_for_the_staleness_check() {
		let root = workspace("listed");
		let text = "<link href=\"a.css\"><link href=\"sub/b.css\">";

		let found = stylesheets(&root.join("hud.html"), text, &root);

		assert_eq!(found.len(), 2, "both of them");
		assert!(found[1].ends_with("b.css"), "resolved against the document's own directory");
	}
}
