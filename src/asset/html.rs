//! A deliberate subset of HTML.
//!
//! Elements, attributes, text and comments. No scripts, no tables, no forms, no
//! doctype, and no error recovery worth the name - a document that does not
//! close a tag is a compile error naming the tag, rather than a tree a browser
//! guessed at. A reader of that shape is a hundred and fifty lines; this is
//! longer only because it reports where things went wrong.
//!
//! Six elements mean something:
//!
//! - **`<style>`** holds a stylesheet, read by [`css`](crate::css).
//! - **`<link href="theme.css">`** names one that lives beside the document.
//!   The *compiler* resolves it and hands the result in, so nothing at runtime
//!   has to find a second file. @ref [`links`].
//! - **`<script src="ui/hud">`** names the program this document's logic is in,
//!   **by the name it is registered under** rather than by a file beside the
//!   document - the same way an image names a texture. A program is an asset of
//!   its own, so there is nothing here to fold in and nothing to resolve; what
//!   the host does with the name is look it up. A `<script>` with a *body* is
//!   refused, because a program that lived inside a document could not be
//!   shared, could not reload on its own, and would be a second kind of
//!   program.
//! - **`<img src="textures/construct/floor">`** draws a texture, by the name it
//!   is registered under. The one part of the interface that was already free.
//! - **`<body>`** at the top level, if there is one, is the root itself rather
//!   than a box inside it.
//! - anything else is a box. `div`, `span`, `button`, `p` - the tag is not
//!   checked against a list, because the only thing a tag does here is give a
//!   stylesheet something to match on.
//!
//! **Whitespace between elements is dropped.** In a browser the space between
//! two `<span>`s is a space; here every child of a box is a flex item, and a
//! text node made of one space would be an item that pushes the others around
//! by however the file happens to be indented. `gap` is how boxes are spaced.

use colby_core::{
	Result,
	abi::ui::{
		DocumentData, Node,
		document::{NONE, ROOT},
	},
	err,
};

use crate::css::{self, Sheet};

/// The extension the importer reads.
pub const EXTENSION: &str = "html";

/// The extension a stylesheet a document links to has.
pub const STYLESHEET_EXTENSION: &str = "css";

/// The tag whose body is a script.
const SCRIPT: &str = "script";

/// What closes one.
const SCRIPT_CLOSE: &str = "</script>";

/// Elements that never have children and are never closed.
const VOID: &[&str] = &["img", "br", "hr", "link", "meta", "input"];

/// What a document turned into.
#[derive(Clone, Debug, Default)]
pub struct Parsed {
	/// The tree and the rules over it.
	pub document: DocumentData,

	/// Everything that was read but not understood.
	pub warnings: Vec<String>,
}

/// Every stylesheet a document links to, in the order they appear.
///
/// Read without parsing the document, because the compiler needs this before it
/// decides whether the document needs compiling at all: an output older than a
/// stylesheet it was built from is stale however new the `.html` is.
///
/// @param text - the whole document
/// @return the `href` of each `<link>`, as written
#[must_use]
pub fn links(text: &str) -> Vec<String> { referenced(text, "link", "href") }

/// Every file one kind of tag names, without parsing the document.
///
/// @param text - the whole document
/// @param tag - the element to look for, lowercase
/// @param attribute - the attribute naming the file
fn referenced(text: &str, tag: &str, attribute: &str) -> Vec<String> {
	let mut found = Vec::new();
	let mut rest = text;

	while let Some(open) = rest.find('<') {
		let after = rest.get(open + 1..).unwrap_or_default();
		let Some(close) = after.find('>') else {
			break;
		};

		let element = after.get(..close).unwrap_or_default();
		rest = after.get(close + 1..).unwrap_or_default();

		let mut words = element.split_whitespace();
		if !words
			.next()
			.is_some_and(|name| name.eq_ignore_ascii_case(tag))
		{
			continue;
		}

		let mut attributes = Attributes::new(element);
		while let Some((name, value)) = attributes.next_pair() {
			if name == attribute {
				found.push(value);
			}
		}
	}

	found
}

/// Reads a document.
///
/// @param text - the whole `.html`
/// @param linked - the stylesheets its `<link>`s named, already parsed, in the
/// same order; their rules go in before the document's own `<style>`
/// @return the tree and the rules, or why the document could not be read
pub fn parse(text: &str, linked: &[Sheet]) -> Result<Parsed> {
	let mut builder = Builder::new();

	for sheet in linked {
		builder
			.document
			.rules
			.extend(sheet.rules.iter().cloned());
		builder
			.warnings
			.extend(sheet.warnings.iter().cloned());
	}

	builder.run(text)?;
	builder.finish()
}

/// The tree being built, and where in the document it is.
struct Builder {
	document: DocumentData,
	/// The elements that are open, innermost last.
	stack: Vec<u32>,
	/// The last child appended to each open element, or [`NONE`].
	last: Vec<u32>,
	warnings: Vec<String>,
	line: u32,
}

impl Builder {
	fn new() -> Self {
		Self {
			document: DocumentData::empty(),
			stack: vec![ROOT],
			last: vec![NONE],
			warnings: Vec::new(),
			line: 1,
		}
	}

	/// Walks the whole document.
	fn run(&mut self, text: &str) -> Result<()> {
		let mut rest = text;

		while let Some(open) = rest.find('<') {
			let before = rest.get(..open).unwrap_or_default();
			self.text(before);
			self.line += newlines(before);

			rest = rest.get(open..).unwrap_or_default();
			rest = self.element(rest)?;
		}

		self.text(rest);

		if self.stack.len() > 1 {
			let open = self
				.stack
				.last()
				.and_then(|index| self.document.node(*index))
				.map_or_else(String::new, |node| node.tag.clone());

			return Err(err!(Asset("`<{open}>` is never closed")));
		}

		Ok(())
	}

	/// Handles whatever starts at a `<`.
	///
	/// @return the rest of the document, after the tag
	fn element<'a>(&mut self, rest: &'a str) -> Result<&'a str> {
		if let Some(after) = rest.strip_prefix("<!--") {
			let Some(close) = after.find("-->") else {
				return Err(err!(Asset("line {}: a comment is never closed", self.line)));
			};

			self.line += newlines(after.get(..close).unwrap_or_default());

			return Ok(after.get(close + 3..).unwrap_or_default());
		}

		let after = rest.get(1..).unwrap_or_default();
		let Some(close) = after.find('>') else {
			return Err(err!(Asset("line {}: a `<` is never closed by a `>`", self.line)));
		};

		let tag = after.get(..close).unwrap_or_default();
		let following = after.get(close + 1..).unwrap_or_default();
		self.line += newlines(tag);

		if let Some(name) = tag.strip_prefix('/') {
			self.close(name.trim())?;

			return Ok(following);
		}

		let name = tag
			.split(|character: char| character.is_whitespace() || character == '/')
			.next()
			.unwrap_or_default()
			.to_ascii_lowercase();

		if name == "style" {
			return Ok(self.stylesheet(following));
		}

		if name == SCRIPT {
			return self.script(tag, following);
		}

		self.open(&name, tag, following)
	}

	/// Reads a `<style>` block and everything up to its closing tag.
	fn stylesheet<'a>(&mut self, rest: &'a str) -> &'a str {
		let end = find_ignoring_case(rest, "</style>").unwrap_or(rest.len());
		let body = rest.get(..end).unwrap_or_default();

		let sheet = css::parse(body);
		self.document.rules.extend(sheet.rules);
		self.warnings.extend(sheet.warnings);
		self.line += newlines(body);

		rest.get(end + "</style>".len()..)
			.unwrap_or_default()
	}

	/// Reads a `<script>` tag and everything up to its closing one.
	///
	/// Stricter than [`stylesheet`](Self::stylesheet), which treats a block
	/// nobody closed as running to the end of the file: an unclosed `<style>`
	/// turns the rest of a document into stylesheet nobody can match, and an
	/// unclosed `<script>` would turn it into a program. Say so instead.
	///
	/// What the tag carries is a **name**, not a program: `src` is the name the
	/// program is registered under, exactly as an image's `src` is the name of
	/// a texture. A body is refused rather than kept, because a program written
	/// inside a document could not be shared with a second document, could not
	/// reload without the document being rebuilt, and would be a second kind of
	/// program for everything downstream to know about.
	fn script<'a>(&mut self, tag: &str, rest: &'a str) -> Result<&'a str> {
		let mut attributes = Attributes::new(tag);
		let mut named = String::new();
		while let Some((name, value)) = attributes.next_pair() {
			if name == "src" {
				named = value;
			}
		}

		let Some(end) = find_ignoring_case(rest, SCRIPT_CLOSE) else {
			return Err(err!(Asset("line {}: `<script>` is never closed", self.line)));
		};

		let body = rest.get(..end).unwrap_or_default();
		let following = rest
			.get(end + SCRIPT_CLOSE.len()..)
			.unwrap_or_default();

		// the tag's own line, kept before the count moves past it, so that a
		// complaint names where the tag was written rather than where it ended.
		let at = self.line;
		self.line += newlines(body);

		if !body.trim().is_empty() {
			return Err(err!(Asset(
				"line {at}: `<script>` has a body; a program is an asset, so put it in a `.lua` \
				 and name it with `src`"
			)));
		}

		if named.is_empty() {
			return Ok(following);
		}

		// one panel shows one document and one document runs one program, so a
		// second name is a document asking for something this engine has no
		// answer to rather than a document asking for two of something.
		if !self.document.program.is_empty() {
			return Err(err!(Asset(
				"line {at}: this document already runs `{}`, and a document runs one program",
				self.document.program
			)));
		}

		self.document.program = named;

		Ok(following)
	}

	/// Adds one element, and opens it if it has children.
	fn open<'a>(&mut self, name: &str, tag: &str, following: &'a str) -> Result<&'a str> {
		let closed = tag.trim_end().ends_with('/');
		let void = VOID.contains(&name) || closed;

		let mut node = Node {
			tag: name.to_owned(),
			parent: NONE,
			first_child: NONE,
			next_sibling: NONE,
			..Node::default()
		};

		let mut attributes = Attributes::new(tag);
		while let Some((key, value)) = attributes.next_pair() {
			self.attribute(&mut node, &key, &value, name);
		}

		// a `<body>` at the top level is the root rather than a box in it: the
		// root exists before the file is read, and a document that wraps
		// everything in one would otherwise have two boxes where it wrote one.
		if name == "body" && self.stack.len() == 1 {
			if let Some(root) = self.document.nodes.first_mut() {
				root.id = node.id;
				root.classes = node.classes;
				root.inline = node.inline;
			}

			return Ok(following);
		}

		let index = self.append(node);

		if !void {
			self.stack.push(index);
			self.last.push(NONE);
		}

		Ok(following)
	}

	/// Puts one attribute on a node.
	fn attribute(&mut self, node: &mut Node, key: &str, value: &str, tag: &str) {
		match key {
			| "id" => value.clone_into(&mut node.id),
			| "class" => value.clone_into(&mut node.classes),
			| "style" => node.inline = css::declarations(value, &mut self.warnings),
			| "src" if tag == Node::IMAGE => value.clone_into(&mut node.source),
			// the *default* value, the way the attribute is in a browser: what
			// the field says once somebody has typed in it lives in the panel's
			// bindings, and the document goes on saying what it always said.
			// @ref [`Bind`](colby_core::abi::ui::Bind).
			| "value" if tag == Node::INPUT => value.clone_into(&mut node.text),
			| "href" | "rel" | "type" if tag == "link" => {},
			| _ => self.warnings.push(format!(
				"line {}: `<{tag}>` has a `{key}` attribute, which colby does not read",
				self.line
			)),
		}
	}

	/// Closes the innermost open element, which had better be this one.
	fn close(&mut self, name: &str) -> Result<()> {
		let name = name.to_ascii_lowercase();

		if VOID.contains(&name.as_str()) {
			// `</img>` is redundant rather than wrong. Nothing was opened, so
			// nothing is closed.
			return Ok(());
		}

		if name == "body" && self.stack.len() == 1 {
			return Ok(());
		}

		let Some(index) = self.stack.last().copied() else {
			return Err(err!(Asset("line {}: `</{name}>` closes nothing", self.line)));
		};

		if index == ROOT {
			return Err(err!(Asset("line {}: `</{name}>` closes nothing", self.line)));
		}

		let open = self
			.document
			.node(index)
			.map_or_else(String::new, |node| node.tag.clone());

		if open != name {
			return Err(err!(Asset(
				"line {}: `</{name}>` closes `<{open}>`, which is not what is open",
				self.line
			)));
		}

		self.stack.pop();
		self.last.pop();

		Ok(())
	}

	/// Adds a run of text, unless it is only the file's indentation.
	fn text(&mut self, raw: &str) {
		if raw.trim().is_empty() {
			return;
		}

		let text = collapse(&entities(raw));

		self.append(Node {
			tag: Node::TEXT.to_owned(),
			text,
			parent: NONE,
			first_child: NONE,
			next_sibling: NONE,
			..Node::default()
		});
	}

	/// Hangs a node off whatever is open, and returns its index.
	fn append(&mut self, mut node: Node) -> u32 {
		let parent = self.stack.last().copied().unwrap_or(ROOT);
		node.parent = parent;

		let index = u32::try_from(self.document.nodes.len()).unwrap_or(NONE);
		self.document.nodes.push(node);

		match self.last.last().copied() {
			| Some(previous) if previous != NONE => {
				if let Some(sibling) = self
					.document
					.nodes
					.get_mut(usize::try_from(previous).unwrap_or(0))
				{
					sibling.next_sibling = index;
				}
			},
			| _ =>
				if let Some(parent) = self
					.document
					.nodes
					.get_mut(usize::try_from(parent).unwrap_or(0))
				{
					parent.first_child = index;
				},
		}

		if let Some(slot) = self.last.last_mut() {
			*slot = index;
		}

		index
	}

	/// The finished document.
	fn finish(self) -> Result<Parsed> {
		Ok(Parsed {
			document: self.document,
			warnings: self.warnings,
		})
	}
}

/// Walks the `name="value"` pairs inside a tag.
struct Attributes<'a> {
	rest: &'a str,
}

impl<'a> Attributes<'a> {
	/// Everything after the tag's own name.
	fn new(tag: &'a str) -> Self {
		let rest = tag
			.find(char::is_whitespace)
			.and_then(|at| tag.get(at..))
			.unwrap_or("");

		Self { rest }
	}

	/// The next pair, lowercased name and decoded value.
	fn next_pair(&mut self) -> Option<(String, String)> {
		let start = self
			.rest
			.find(|character: char| !character.is_whitespace() && character != '/')?;
		let rest = self.rest.get(start..)?;

		let end = rest
			.find(|character: char| character.is_whitespace() || character == '=')
			.unwrap_or(rest.len());
		let name = rest.get(..end)?.to_ascii_lowercase();
		let after = rest.get(end..).unwrap_or("").trim_start();

		let Some(after) = after.strip_prefix('=') else {
			// a bare attribute, `disabled`. Not something colby reads, but it
			// should not swallow the rest of the tag either.
			self.rest = after;

			return Some((name, String::new()));
		};

		let after = after.trim_start();
		let (value, tail) = match after.chars().next() {
			| Some(quote @ ('"' | '\'')) => {
				let inside = after.get(1..)?;
				let close = inside.find(quote).unwrap_or(inside.len());

				(inside.get(..close)?, inside.get(close + 1..).unwrap_or(""))
			},
			| _ => {
				let close = after
					.find(char::is_whitespace)
					.unwrap_or(after.len());

				(after.get(..close)?, after.get(close..).unwrap_or(""))
			},
		};

		self.rest = tail;

		Some((name, entities(value)))
	}
}

/// Replaces the handful of entities a document actually uses.
///
/// Five named ones and the numeric forms. A full table is four hundred entries
/// nobody types; a document that needs one can hold the character itself, since
/// the file is UTF-8 either way.
fn entities(text: &str) -> String {
	if !text.contains('&') {
		return text.to_owned();
	}

	let mut out = String::with_capacity(text.len());
	let mut rest = text;

	while let Some(at) = rest.find('&') {
		out.push_str(rest.get(..at).unwrap_or_default());

		let after = rest.get(at..).unwrap_or_default();
		let Some(end) = after.find(';').filter(|end| *end <= 10) else {
			out.push('&');
			rest = after.get(1..).unwrap_or_default();

			continue;
		};

		let name = after.get(1..end).unwrap_or_default();
		match named(name) {
			| Some(character) => out.push(character),
			| None => {
				out.push('&');
				out.push_str(name);
				out.push(';');
			},
		}

		rest = after.get(end + 1..).unwrap_or_default();
	}

	out.push_str(rest);

	out
}

/// One entity by name, or by number.
fn named(name: &str) -> Option<char> {
	match name {
		| "amp" => return Some('&'),
		| "lt" => return Some('<'),
		| "gt" => return Some('>'),
		| "quot" => return Some('"'),
		| "apos" => return Some('\''),
		| "nbsp" => return Some('\u{A0}'),
		| _ => {},
	}

	let digits = name.strip_prefix('#')?;
	let code = match digits.strip_prefix(['x', 'X']) {
		| Some(hex) => u32::from_str_radix(hex, 16).ok()?,
		| None => digits.parse().ok()?,
	};

	char::from_u32(code)
}

/// Turns every run of whitespace into one space and trims the ends.
fn collapse(text: &str) -> String {
	let mut out = String::with_capacity(text.len());

	for word in text.split_whitespace() {
		if !out.is_empty() {
			out.push(' ');
		}

		out.push_str(word);
	}

	out
}

/// How many lines a piece of the document spans.
fn newlines(text: &str) -> u32 {
	u32::try_from(text.bytes().filter(|byte| *byte == b'\n').count()).unwrap_or(0)
}

/// Where a closing tag starts, whatever case it was written in.
fn find_ignoring_case(text: &str, needle: &str) -> Option<usize> {
	text.to_ascii_lowercase().find(needle)
}

#[cfg(test)]
mod tests {
	use colby_core::abi::ui::style::Length;

	use super::*;

	/// A document, with nothing linked to it.
	fn read(text: &str) -> Parsed { parse(text, &[]).expect("it reads") }

	#[test]
	fn a_tree_of_boxes_comes_out_as_a_tree() {
		let parsed = read("<div id=\"outer\"><div id=\"inner\"></div><span></span></div>");
		let data = &parsed.document;

		let outer = data
			.find("outer")
			.expect("the outer box is there");
		assert_eq!(data.node(outer).map(|node| node.parent), Some(ROOT), "hung off the root");
		assert_eq!(data.children(outer).count(), 2, "with both of its children");
		assert_eq!(
			data.children(outer)
				.filter_map(|index| data.node(index))
				.map(|node| node.tag.clone())
				.collect::<Vec<_>>(),
			vec!["div", "span"],
			"in the order they were written"
		);
	}

	#[test]
	fn text_becomes_a_node_of_its_own() {
		let parsed = read("<div id=\"label\">hello</div>");
		let data = &parsed.document;
		let label = data.find("label").expect("the box is there");
		let text = data
			.children(label)
			.next()
			.expect("and holds something");

		assert!(data.node(text).is_some_and(Node::is_text), "which is a run of text");
		assert_eq!(data.node(text).map(|node| node.text.clone()), Some("hello".to_owned()));
	}

	#[test]
	fn the_indentation_of_the_file_does_not_become_boxes() {
		let parsed = read("<div id=\"outer\">\n\t<div></div>\n\t<div></div>\n</div>");
		let outer = parsed
			.document
			.find("outer")
			.expect("the outer box is there");

		assert_eq!(
			parsed.document.children(outer).count(),
			2,
			"two boxes, and not five with three runs of whitespace between them"
		);
	}

	#[test]
	fn a_run_of_whitespace_inside_text_becomes_one_space() {
		let parsed = read("<p id=\"line\">two   words\n\there</p>");
		let line = parsed.document.find("line").expect("it is there");
		let text = parsed
			.document
			.children(line)
			.next()
			.expect("with text in it");

		assert_eq!(
			parsed
				.document
				.node(text)
				.map(|node| node.text.clone()),
			Some("two words here".to_owned()),
			"so that a document can be indented without changing what it says"
		);
	}

	#[test]
	fn attributes_are_read_and_the_style_one_is_parsed() {
		let parsed = read("<div id=\"a\" class=\"panel wide\" style=\"width: 20px\"></div>");
		let index = parsed.document.find("a").expect("it is there");
		let node = parsed.document.node(index).expect("it is there");

		assert_eq!(node.classes, "panel wide", "the class list, as written");
		assert_eq!(node.inline.width, Some(Length::Px(20.0)), "and the style, parsed");
	}

	#[test]
	fn single_quotes_and_no_quotes_both_work() {
		let parsed = read("<div id='a' class=panel></div>");

		assert!(parsed.document.find("a").is_some(), "single quotes");
		assert_eq!(
			parsed
				.document
				.find("a")
				.and_then(|index| parsed.document.node(index))
				.map(|node| node.classes.clone()),
			Some("panel".to_owned()),
			"and none at all"
		);
	}

	#[test]
	fn an_attribute_colby_does_not_read_is_a_warning_rather_than_silence() {
		let parsed = read("<div onclick=\"go()\"></div>");

		assert_eq!(parsed.warnings.len(), 1, "one thing was not understood");
		assert!(
			parsed.warnings[0].contains("onclick"),
			"and it says which: {}",
			parsed.warnings[0]
		);
	}

	#[test]
	fn a_style_block_becomes_rules_on_the_document() {
		let parsed = read("<style>.panel { width: 40px; }</style><div class=\"panel\"></div>");

		assert_eq!(parsed.document.rules.len(), 1, "the rule was read");
		assert_eq!(
			parsed.document.rules[0].style.width,
			Some(Length::Px(40.0)),
			"and it means what it says"
		);
		assert_eq!(parsed.document.nodes.len(), 2, "and the block is not a box");
	}

	#[test]
	fn a_script_names_the_program_it_runs_and_is_not_a_box() {
		let parsed = read("<script src=\"ui/hud\"></script><div></div>");

		assert_eq!(parsed.document.program, "ui/hud", "the name, as written");
		assert_eq!(parsed.document.nodes.len(), 2, "and the tag is not a box");
		assert!(
			parsed.warnings.is_empty(),
			"nor is the `src` an attribute nobody reads: {:?}",
			parsed.warnings
		);
	}

	#[test]
	fn a_script_with_a_body_is_refused_rather_than_kept() {
		// a program that lived inside a document could not be shared with a
		// second document and could not reload without the document being
		// rebuilt, so there is one place a program can be and it is a file.
		let error = parse("<script>local a = 1</script>", &[])
			.expect_err("a program does not live in a document");

		assert!(error.to_string().contains(".lua"), "saying where to put it: {error}");
	}

	#[test]
	fn a_second_program_is_refused_rather_than_quietly_winning() {
		// one panel shows one document and one document runs one program. The
		// trap this closes is the *silent* form: taking the last one, so that
		// a document naming two runs whichever happens to be written second.
		let error = parse("<script src=\"ui/a\"></script><script src=\"ui/b\"></script>", &[])
			.expect_err("two is not a number of programs a document may have");

		assert!(error.to_string().contains("ui/a"), "naming the one it already has: {error}");
	}

	#[test]
	fn an_empty_script_tag_names_nothing_and_is_not_an_error() {
		let parsed = read("<script></script><div></div>");

		assert!(parsed.document.program.is_empty(), "there is nothing to run");
		assert_eq!(parsed.document.nodes.len(), 2, "and still no box");
	}

	#[test]
	fn a_script_that_is_never_closed_is_an_error_rather_than_the_rest_of_the_file() {
		let error =
			parse("<script src=\"ui/hud\">\n<div></div>", &[]).expect_err("nothing closes it");

		assert!(error.to_string().contains("script"), "naming the tag: {error}");
	}

	#[test]
	fn a_linked_sheet_comes_before_the_documents_own_rules() {
		let linked = css::parse(".panel { width: 10px; }");
		let parsed =
			parse("<style>.panel { width: 20px; }</style>", &[linked]).expect("it reads");

		assert_eq!(parsed.document.rules.len(), 2, "both sheets are in");
		assert_eq!(
			parsed.document.rules[0].style.width,
			Some(Length::Px(10.0)),
			"the linked one first, so that a document overrides the sheet it shares"
		);
	}

	#[test]
	fn the_links_are_readable_without_reading_the_document() {
		let found = links(
			"<link rel=\"stylesheet\" href=\"theme.css\">\n<link href='other.css'>\n<div></div>",
		);

		assert_eq!(found, vec!["theme.css", "other.css"], "both, in order");
		assert!(links("<div></div>").is_empty(), "and a document with none has none");
	}

	#[test]
	fn a_void_element_needs_no_closing_tag() {
		let parsed = read("<div id=\"a\"><img src=\"textures/tiles\"><br></div>");
		let outer = parsed.document.find("a").expect("it is there");

		assert_eq!(parsed.document.children(outer).count(), 2, "both are children of the box");
		assert!(
			parsed
				.document
				.children(outer)
				.filter_map(|index| parsed.document.node(index))
				.any(|node| node.is_image() && node.source == "textures/tiles"),
			"and the image knows which texture it draws"
		);
	}

	#[test]
	fn a_self_closing_tag_is_closed() {
		let parsed = read("<div id=\"a\"/><div id=\"b\"></div>");

		assert_eq!(
			parsed
				.document
				.find("b")
				.and_then(|index| parsed.document.node(index))
				.map(|node| node.parent),
			Some(ROOT),
			"the second box is a sibling, not a child of the first"
		);
	}

	#[test]
	fn a_body_at_the_top_is_the_root_rather_than_a_box_in_it() {
		let parsed = read("<body class=\"screen\"><div id=\"a\"></div></body>");

		assert_eq!(
			parsed
				.document
				.node(ROOT)
				.map(|node| node.classes.clone()),
			Some("screen".to_owned()),
			"its attributes are the root's"
		);
		assert_eq!(
			parsed
				.document
				.find("a")
				.and_then(|index| parsed.document.node(index))
				.map(|node| node.parent),
			Some(ROOT),
			"and what was inside it hangs off the root directly"
		);
	}

	#[test]
	fn comments_are_skipped_whatever_is_in_them() {
		let parsed = read("<!-- <div id=\"ghost\"></div> --><div id=\"real\"></div>");

		assert!(parsed.document.find("ghost").is_none(), "nothing in a comment is a box");
		assert!(parsed.document.find("real").is_some(), "and what follows one still is");
	}

	#[test]
	fn entities_are_replaced_in_text_and_in_attributes() {
		let parsed = read("<div id=\"a\" class=\"&quot;q&quot;\">&lt;b&gt; &amp; &#65;</div>");
		let index = parsed.document.find("a").expect("it is there");
		let text = parsed
			.document
			.children(index)
			.next()
			.expect("with text in it");

		assert_eq!(
			parsed
				.document
				.node(text)
				.map(|node| node.text.clone()),
			Some("<b> & A".to_owned()),
			"named and numeric both"
		);
		assert_eq!(
			parsed
				.document
				.node(index)
				.map(|node| node.classes.clone()),
			Some("\"q\"".to_owned()),
			"and in an attribute value too"
		);
	}

	#[test]
	fn an_unclosed_element_is_an_error_naming_it() {
		let error = parse("<div><span></div>", &[]).expect_err("the span is still open");

		assert!(
			error.to_string().contains("span"),
			"and the message says which tag disagrees: {error}"
		);
	}

	#[test]
	fn a_closing_tag_with_nothing_open_is_an_error() {
		assert!(parse("</div>", &[]).is_err(), "it closes nothing");
		assert!(parse("<div></div></div>", &[]).is_err(), "and neither does the second one");
	}

	#[test]
	fn an_unclosed_element_at_the_end_of_the_file_is_an_error() {
		let error = parse("<div id=\"a\">", &[]).expect_err("it is never closed");

		assert!(error.to_string().contains("div"), "naming the tag: {error}");
	}

	#[test]
	fn a_document_with_nothing_in_it_is_a_document() {
		let parsed = read("");

		assert_eq!(parsed.document.nodes.len(), 1, "the root, and nothing else");
		assert!(parsed.warnings.is_empty(), "and nothing wrong with that");
	}
}
