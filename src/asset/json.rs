//! Reading JSON, because one import format is a JSON document.
//!
//! Nothing colby loads at runtime is text, and nothing it compiled before now
//! was either: an `.obj` is line-oriented, an `.html` is a tag soup, a `.css`
//! is declarations. glTF is the first source format whose structure is JSON,
//! and this module is the whole of what reading one needs.
//!
//! **Why this is written rather than taken.** Every engine that already owns a
//! JSON parser writes its own glTF reader on top of it, and every engine that
//! does not reaches for a library. colby is in the first camp on principle and
//! the second by accident: taking a JSON crate means taking a derive stack,
//! and it lands in the runner's binary rather than only in the compiler. What
//! is here instead is a recursive-descent reader of about the size of the
//! importers already in this crate, and it has no dependency at all.
//!
//! **The dialect is strict.** RFC 8259 and nothing beside it: no comments, no
//! trailing commas, no unquoted field names, no single quotes, no `NaN` and no
//! `Infinity`, no leading zeros and no leading plus. Everything a generator
//! writes is inside that, and everything outside it is a file saying something
//! it does not mean. A leading byte order mark is the one thing dropped rather
//! than refused, because every editor on this platform writes one and finding
//! that out cost this project a session already.
//!
//! **Every way a document can be wrong is one sentence naming a line and a
//! column.** A `.gltf` is a file a person can open, so the parser is expected
//! to answer "where" and not only "no". Nothing here panics on bad input, and
//! nothing recurses without a bound: a document nested past [`MAX_DEPTH`] is
//! refused, which is what keeps a hostile file from taking the stack with it.
//!
//! ```text
//!   { "meshes": [ { "name": "shade" } ] }
//!
//!   parse(text)?                      -> Value::Object
//!     .get("meshes")                  -> Some(Value::Array)
//!     .map(Value::as_array)[0]        -> Value::Object
//!     .get("name").and_then(as_str)   -> Some("shade")
//! ```

use colby_core::{Result, err};

/// How deeply a document may nest before it is refused.
///
/// The parser is recursive, so this is what stands between a hostile file and
/// the stack. Real documents are nowhere near it: the deepest thing a glTF
/// nests is a material's texture reference inside a primitive inside a mesh,
/// which is six.
pub const MAX_DEPTH: usize = 128;

/// The largest whole number an `f64` holds exactly.
///
/// JSON has one numeric type and it is a double, so an index past this point
/// is a number that cannot be trusted to be the one that was written.
const WHOLE_LIMIT: f64 = 9_007_199_254_740_992.0;

/// A JSON value.
///
/// An object is a list of pairs rather than a map, for two reasons. Objects in
/// the documents this reads hold a handful of fields, so a scan beats hashing
/// the way it does in the asset registries; and the order a generator wrote
/// them in survives, which is what makes an error message about the third
/// field mean anything.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Value {
	/// `null`, and also what a missing field reads as when a caller asks for
	/// one that is not there.
	#[default]
	Null,

	/// `true` or `false`.
	Bool(bool),

	/// A number. JSON has exactly one, and it is a double.
	Number(f64),

	/// A string, with every escape already undone.
	String(String),

	/// An array, in the order it was written.
	Array(Vec<Self>),

	/// An object, in the order it was written.
	///
	/// A name written twice is stored twice and [`Value::get`] answers with the
	/// last of them, which is what every other reader of this format does. The
	/// alternative, replacing on the way in, costs a scan per field and buys a
	/// case no generator produces.
	Object(Vec<(String, Self)>),
}

impl Value {
	/// One field of an object, by name.
	///
	/// @param name - the field to look for
	/// @return the value written last under that name, if this is an object
	/// that has one at all
	#[must_use]
	pub fn get(&self, name: &str) -> Option<&Self> {
		let Self::Object(fields) = self else {
			return None;
		};

		fields
			.iter()
			.rev()
			.find(|(written, _)| written == name)
			.map(|(_, value)| value)
	}

	/// The items, or nothing when this is not an array.
	///
	/// An empty slice rather than an `Option` because every caller wants to
	/// walk it, and "there were none" and "it was not an array" lead to the
	/// same loop. A caller that needs to tell them apart matches the variant.
	#[must_use]
	pub fn as_array(&self) -> &[Self] {
		match self {
			| Self::Array(items) => items,
			| _ => &[],
		}
	}

	/// The fields, or nothing when this is not an object.
	///
	/// For the one case a name cannot answer: walking what an object holds
	/// without knowing what to ask for.
	#[must_use]
	pub fn as_object(&self) -> &[(String, Self)] {
		match self {
			| Self::Object(fields) => fields,
			| _ => &[],
		}
	}

	/// The text, if this is a string.
	#[must_use]
	pub fn as_str(&self) -> Option<&str> {
		match self {
			| Self::String(text) => Some(text),
			| _ => None,
		}
	}

	/// The flag, if this is a boolean.
	#[must_use]
	pub const fn as_bool(&self) -> Option<bool> {
		match self {
			| Self::Bool(flag) => Some(*flag),
			| _ => None,
		}
	}

	/// The number, if this is one.
	#[must_use]
	pub const fn as_f64(&self) -> Option<f64> {
		match self {
			| Self::Number(number) => Some(*number),
			| _ => None,
		}
	}

	/// The same, narrowed to what the engine stores.
	///
	/// Every number that reaches a vertex, a color or a transform is an `f32`,
	/// so the narrowing happens once here rather than at each of them.
	#[must_use]
	pub fn as_f32(&self) -> Option<f32> { self.as_f64().map(narrow) }

	/// The number as a whole one, if that is what it is.
	///
	/// A number with a fractional part, a negative one, or one past the point
	/// where a double stops counting exactly is refused rather than rounded.
	/// Indices into a file's own tables come through here, and a rounded index
	/// is a mesh quietly reading the wrong buffer.
	#[must_use]
	pub fn as_u64(&self) -> Option<u64> {
		let number = self.as_f64()?;

		if !(0.0..=WHOLE_LIMIT).contains(&number) {
			return None;
		}

		if is_fractional(number) {
			return None;
		}

		Some(whole(number))
	}

	/// The same, as the width the file's own tables are indexed with.
	#[must_use]
	pub fn as_u32(&self) -> Option<u32> { u32::try_from(self.as_u64()?).ok() }

	/// The same, as the width this machine indexes a slice with.
	#[must_use]
	pub fn as_usize(&self) -> Option<usize> { usize::try_from(self.as_u64()?).ok() }

	/// Whether this is `null`.
	///
	/// The one question the other readers cannot answer: they say `None` both
	/// for a field that is absent and for one that was written as `null`.
	#[must_use]
	pub const fn is_null(&self) -> bool { matches!(self, Self::Null) }
}

/// Reads a whole JSON document.
///
/// @param text - the document, already decoded from UTF-8
/// @return the value it holds, or the line and column that could not be read
pub fn parse(text: &str) -> Result<Value> {
	// a byte order mark is not whitespace, so left in place it would be the
	// first thing the parser sees and nothing in the grammar starts with one.
	// The same trap the console tokenizer and the OBJ importer already learned.
	let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
	let mut parser = Parser { text, at: 0, depth: 0 };

	let value = parser
		.value()
		.map_err(|reason| err!(Asset("{reason}")))?;

	parser.space();

	if parser.at < text.len() {
		let reason = parser.fault("there is more in the file after the value ends");

		return Err(err!(Asset("{reason}")));
	}

	Ok(value)
}

/// What one step of the parse produces: a value, or a sentence saying where it
/// stopped and why.
type Step<T> = std::result::Result<T, String>;

/// The state of one parse.
struct Parser<'a> {
	/// The whole document, with any byte order mark already gone.
	text: &'a str,

	/// How far in, in bytes. Always on a character boundary: every byte the
	/// scanner stops on is ASCII, and a UTF-8 continuation byte is not.
	at: usize,

	/// How many arrays and objects are open above this point.
	depth: usize,
}

impl Parser<'_> {
	/// Reads one value, whatever it turns out to be.
	fn value(&mut self) -> Step<Value> {
		self.space();

		let Some(byte) = self.peek() else {
			return Err(self.fault("a value was expected and the file ended"));
		};

		match byte {
			| b'{' | b'[' => self.nested(byte),
			| b'"' => self.string().map(Value::String),
			| b'-' | b'0'..=b'9' => self.number(),
			| b't' => self.word("true", Value::Bool(true)),
			| b'f' => self.word("false", Value::Bool(false)),
			| b'n' => self.word("null", Value::Null),

			| b']' | b'}' =>
				Err(self.fault("a value was expected and a closing bracket came instead")),
			| _ => Err(self.fault(
				"a value has to be an object, an array, a string, a number, true, false or null",
			)),
		}
	}

	/// Reads an array or an object, one level further down.
	///
	/// The bound lives here rather than in each of the two, so there is one
	/// place that knows the parser is recursive.
	fn nested(&mut self, opener: u8) -> Step<Value> {
		self.depth += 1;

		if self.depth > MAX_DEPTH {
			return Err(self.fault("this document nests deeper than colby will follow"));
		}

		let value = if opener == b'{' { self.object() } else { self.array() };
		self.depth -= 1;

		value
	}

	/// Reads an object, starting at its brace.
	fn object(&mut self) -> Step<Value> {
		self.at += 1;
		let mut fields = Vec::new();
		self.space();

		if self.eat(b'}') {
			return Ok(Value::Object(fields));
		}

		loop {
			self.space();

			if self.peek() != Some(b'"') {
				return Err(self.fault("a field name has to be a string in double quotes"));
			}

			let name = self.string()?;
			self.space();

			if !self.eat(b':') {
				return Err(self.fault("a field name has to be followed by a colon"));
			}

			fields.push((name, self.value()?));
			self.space();

			if self.eat(b',') {
				continue;
			}

			if self.eat(b'}') {
				return Ok(Value::Object(fields));
			}

			return Err(self.fault("a field has to be followed by a comma or by a closing brace"));
		}
	}

	/// Reads an array, starting at its bracket.
	fn array(&mut self) -> Step<Value> {
		self.at += 1;
		let mut items = Vec::new();
		self.space();

		if self.eat(b']') {
			return Ok(Value::Array(items));
		}

		loop {
			items.push(self.value()?);
			self.space();

			if self.eat(b',') {
				continue;
			}

			if self.eat(b']') {
				return Ok(Value::Array(items));
			}

			return Err(
				self.fault("an item has to be followed by a comma or by a closing bracket")
			);
		}
	}

	/// Reads a string, starting at its opening quote, with escapes undone.
	///
	/// Everything between two escapes is copied as one slice rather than a
	/// character at a time, so a document of plain names costs one copy per
	/// string.
	fn string(&mut self) -> Step<String> {
		self.at += 1;
		let mut text = String::new();
		let mut chunk = self.at;

		loop {
			let Some(byte) = self.peek() else {
				return Err(self.fault("the file ended inside a string"));
			};

			match byte {
				| b'"' => {
					text.push_str(self.slice(chunk, self.at));
					self.at += 1;

					return Ok(text);
				},
				| b'\\' => {
					text.push_str(self.slice(chunk, self.at));
					self.at += 1;
					text.push(self.escape()?);
					chunk = self.at;
				},
				| 0x00..=0x1F => {
					return Err(
						self.fault("a control character inside a string has to be escaped")
					);
				},
				| _ => self.at += 1,
			}
		}
	}

	/// Reads what follows a backslash.
	fn escape(&mut self) -> Step<char> {
		let Some(byte) = self.peek() else {
			return Err(self.fault("the file ended inside an escape"));
		};

		self.at += 1;

		let plain = match byte {
			| b'"' => '"',
			| b'\\' => '\\',
			| b'/' => '/',
			| b'b' => '\u{8}',
			| b'f' => '\u{C}',
			| b'n' => '\n',
			| b'r' => '\r',
			| b't' => '\t',
			| b'u' => return self.escaped_char(),
			| _ => return Err(self.fault("this is not an escape JSON has")),
		};

		Ok(plain)
	}

	/// Reads a `\u` escape, and the second half of a pair when there is one.
	fn escaped_char(&mut self) -> Step<char> {
		let first = self.four_hex_digits()?;

		// anything outside the basic plane is written as two escapes, and a
		// half on its own is not a character at all. Refusing rather than
		// substituting a replacement character is the same rule the rest of
		// this module follows: say where the file is wrong.
		if (0xD800..0xDC00).contains(&first) {
			if !self.rest().starts_with("\\u") {
				return Err(self.fault("this escape starts a pair and nothing finishes it"));
			}

			self.at += 2;
			let second = self.four_hex_digits()?;

			if !(0xDC00..0xE000).contains(&second) {
				return Err(self.fault("this escape finishes a pair and is not a second half"));
			}

			let joined = 0x1_0000 + ((first - 0xD800) << 10) + (second - 0xDC00);

			return char::from_u32(joined)
				.ok_or_else(|| self.fault("this pair of escapes is not a character"));
		}

		if (0xDC00..0xE000).contains(&first) {
			return Err(self.fault("this escape is a second half with no first half before it"));
		}

		char::from_u32(first).ok_or_else(|| self.fault("this escape is not a character"))
	}

	/// Reads the four digits of a `\u` escape.
	fn four_hex_digits(&mut self) -> Step<u32> {
		let mut value = 0;

		for _ in 0..4 {
			let Some(byte) = self.peek() else {
				return Err(self.fault("the file ended inside an escape"));
			};

			let Some(digit) = char::from(byte).to_digit(16) else {
				return Err(self.fault("an escape needs four hexadecimal digits"));
			};

			value = value * 16 + digit;
			self.at += 1;
		}

		Ok(value)
	}

	/// Reads a number, having checked the grammar by hand first.
	///
	/// The shape is checked here and the digits are turned into a double by the
	/// standard library, which is the division of labor that matters: the
	/// grammar is what keeps `NaN`, `0x10` and `+1` out, and the conversion is
	/// somebody else's correctly rounded arithmetic.
	fn number(&mut self) -> Step<Value> {
		let start = self.at;

		if self.eat(b'-') && self.peek().is_none() {
			return Err(self.fault("a minus sign needs a number after it"));
		}

		match self.peek() {
			| Some(b'0') => {
				self.at += 1;

				if matches!(self.peek(), Some(b'0'..=b'9')) {
					return Err(self.fault("a number may not have a leading zero"));
				}
			},
			| Some(b'1'..=b'9') => self.digits(),
			| _ => return Err(self.fault("a number needs at least one digit")),
		}

		if self.eat(b'.') {
			if !matches!(self.peek(), Some(b'0'..=b'9')) {
				return Err(self.fault("a decimal point needs a digit after it"));
			}

			self.digits();
		}

		if matches!(self.peek(), Some(b'e' | b'E')) {
			self.at += 1;

			if matches!(self.peek(), Some(b'+' | b'-')) {
				self.at += 1;
			}

			if !matches!(self.peek(), Some(b'0'..=b'9')) {
				return Err(self.fault("an exponent needs at least one digit"));
			}

			self.digits();
		}

		let written = self.slice(start, self.at);
		let Ok(number) = written.parse::<f64>() else {
			return Err(self.fault("this is not a number this machine can hold"));
		};

		if !number.is_finite() {
			return Err(self.fault("this number is too large to hold"));
		}

		Ok(Value::Number(number))
	}

	/// Reads one of the three bare words.
	fn word(&mut self, word: &str, value: Value) -> Step<Value> {
		if !self.rest().starts_with(word) {
			return Err(self.fault("this is not true, false or null"));
		}

		self.at += word.len();

		Ok(value)
	}

	/// Walks past every digit there is.
	fn digits(&mut self) {
		while matches!(self.peek(), Some(b'0'..=b'9')) {
			self.at += 1;
		}
	}

	/// Walks past the four bytes JSON calls whitespace, and no others.
	fn space(&mut self) {
		while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
			self.at += 1;
		}
	}

	/// Walks past one byte if it is the one expected.
	fn eat(&mut self, byte: u8) -> bool {
		if self.peek() == Some(byte) {
			self.at += 1;

			return true;
		}

		false
	}

	/// The byte the parse is standing on.
	fn peek(&self) -> Option<u8> { self.text.as_bytes().get(self.at).copied() }

	/// Everything from here to the end.
	fn rest(&self) -> &str { self.text.get(self.at..).unwrap_or("") }

	/// A piece of the document.
	///
	/// Both ends are byte offsets the scanner produced, so both are on
	/// character boundaries; the fallback is unreachable and is an empty string
	/// rather than a panic because a name that comes out short is a better
	/// failure than a dead compiler.
	fn slice(&self, from: usize, to: usize) -> &str { self.text.get(from..to).unwrap_or("") }

	/// One sentence saying where the parse stopped and why.
	fn fault(&self, reason: &str) -> String {
		let before = self.text.get(..self.at).unwrap_or("");
		let line = before.matches('\n').count() + 1;
		let opened = before.rfind('\n').map_or(0, |index| index + 1);
		let column = before.get(opened..).unwrap_or("").chars().count() + 1;

		format!("line {line} column {column}: {reason}")
	}
}

/// Whether a number has anything after its decimal point.
///
/// The comparison is exact on purpose and is correct that way: the fractional
/// part of a whole double is zero itself rather than nearly zero, so a
/// tolerance here would accept a file that wrote an index as `3.0000001`.
fn is_fractional(number: f64) -> bool { number.fract() != 0.0 }

/// A whole number that has already been checked to fit.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "the only conversion from a float in the language, guarded by the caller against \
	          every case it would round: not finite, negative, fractional, or past where a \
	          double still counts one at a time"
)]
fn whole(number: f64) -> u64 { number as u64 }

/// A double as the width the engine stores.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "the only narrowing from a double in the language, and every number in a vertex, a \
	          color or a transform is stored at this width"
)]
fn narrow(number: f64) -> f32 { number as f32 }

#[cfg(test)]
mod tests {
	use super::*;

	/// The value a document holds, or the message saying why it does not.
	fn read(text: &str) -> Value { parse(text).expect("the document should read") }

	/// Why a document was refused, without the wrapper around it.
	fn refusal(text: &str) -> String {
		parse(text)
			.expect_err("the document should be refused")
			.to_string()
	}

	#[test]
	fn an_object_reads_back_the_fields_that_were_written() {
		let value = read(r#"{ "name": "shade", "index": 3, "skip": true, "gone": null }"#);

		assert_eq!(value.get("name").and_then(Value::as_str), Some("shade"));
		assert_eq!(value.get("index").and_then(Value::as_usize), Some(3));
		assert_eq!(value.get("skip").and_then(Value::as_bool), Some(true));
		assert!(value.get("gone").is_some_and(Value::is_null), "a null field is present");
		assert!(value.get("absent").is_none(), "and one nobody wrote is not");
	}

	#[test]
	fn an_array_keeps_its_order() {
		let value = read("[1, 2, 3]");
		let read_back: Vec<u32> = value
			.as_array()
			.iter()
			.filter_map(Value::as_u32)
			.collect();

		assert_eq!(read_back, vec![1, 2, 3]);
	}

	#[test]
	fn asking_the_wrong_question_answers_nothing_rather_than_guessing() {
		let value = read(r#"{ "a": 1 }"#);

		assert_eq!(value.as_str(), None, "an object is not a string");
		assert!(value.as_array().is_empty(), "and it is not an array");
		assert_eq!(read("[]").get("a"), None, "an array has no fields");
		assert!(read("1").as_object().is_empty(), "and a number has none either");
	}

	#[test]
	fn every_shape_of_number_is_read() {
		assert_eq!(read("0").as_f64(), Some(0.0));
		assert_eq!(read("-17").as_f64(), Some(-17.0));
		assert_eq!(read("1.5").as_f64(), Some(1.5));
		assert_eq!(read("1e3").as_f64(), Some(1000.0));
		assert_eq!(read("1E+3").as_f64(), Some(1000.0));
		assert_eq!(read("15e-1").as_f64(), Some(1.5));
		assert_eq!(read("-0.25").as_f32(), Some(-0.25));
	}

	#[test]
	fn an_index_has_to_be_a_whole_number_that_fits() {
		assert_eq!(read("7").as_u32(), Some(7), "a whole number is one");
		assert_eq!(read("7.0").as_u32(), Some(7), "and so is one written with a point");
		assert_eq!(read("3.5").as_u32(), None, "a fraction is not an index");
		assert_eq!(read("-1").as_u32(), None, "nor is a negative one");
		assert_eq!(read("4294967296").as_u32(), None, "nor is one past the width");
		assert_eq!(
			read("4294967296").as_u64(),
			Some(4_294_967_296),
			"which is a width, not a bug"
		);
		assert_eq!(read("1e300").as_u64(), None, "nor is one past where a double counts");
	}

	#[test]
	fn a_number_too_large_to_hold_is_refused_rather_than_read_as_infinite() {
		let message = refusal("1e400");

		assert!(message.contains("too large"), "got {message}");
	}

	#[test]
	fn every_escape_is_undone() {
		let value = read(r#""a\"b\\c\/d\be\ff\ng\rh\tiA""#);

		assert_eq!(value.as_str(), Some("a\"b\\c/d\u{8}e\u{C}f\ng\rh\tiA"));
	}

	#[test]
	fn a_pair_of_escapes_becomes_one_character_outside_the_basic_plane() {
		let value = read(r#""\uD83D\uDE00""#);

		assert_eq!(value.as_str(), Some("\u{1F600}"));
	}

	#[test]
	fn half_of_a_pair_on_its_own_is_refused() {
		assert!(refusal(r#""\uD83D""#).contains("nothing finishes it"), "a first half alone");
		assert!(refusal(r#""\uD83DA""#).contains("nothing finishes it"), "or with letters after");
		assert!(refusal(r#""\uDE00""#).contains("no first half"), "a second half alone");
		assert!(
			refusal(r#""\uD83D\u0041""#).contains("not a second half"),
			"a first half followed by an escape that is not one"
		);
	}

	#[test]
	fn a_control_character_inside_a_string_is_refused() {
		let message = refusal("\"a\nb\"");

		assert!(message.contains("control character"), "got {message}");
	}

	#[test]
	fn the_dialect_is_strict_about_everything_a_generator_never_writes() {
		assert!(refusal("[1, 2, ]").contains("closing bracket"), "no trailing comma");
		assert!(refusal(r#"{ "a": 1, }"#).contains("field name"), "not in an object either");
		assert!(refusal("{ a: 1 }").contains("double quotes"), "a field name is quoted");
		assert!(refusal("'a'").contains("a value has to be"), "and quoted with the right quote");
		assert!(refusal("01").contains("leading zero"), "no leading zero");
		assert!(refusal("+1").contains("a value has to be"), "no leading plus");
		assert!(refusal("NaN").contains("a value has to be"), "no NaN");
		assert!(refusal("nul").contains("not true, false or null"), "and no near miss");
		assert!(refusal("1.").contains("decimal point"), "a point needs a digit");
		assert!(refusal("1e").contains("exponent"), "so does an exponent");
	}

	#[test]
	fn anything_after_the_value_is_refused() {
		let message = refusal("{} {}");

		assert!(message.contains("after the value ends"), "got {message}");
		assert_eq!(read("  {}   "), Value::Object(Vec::new()), "but space around it is fine");
	}

	#[test]
	fn nesting_past_the_cap_is_refused_rather_than_taking_the_stack() {
		let deep = format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1));
		let message = refusal(&deep);

		assert!(message.contains("nests deeper"), "got {message}");

		let allowed = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));

		assert!(parse(&allowed).is_ok(), "and one exactly at the cap is read");
	}

	#[test]
	fn a_byte_order_mark_at_the_head_is_dropped() {
		let value = read("\u{FEFF}{ \"a\": 1 }");

		assert_eq!(value.get("a").and_then(Value::as_u32), Some(1));
	}

	#[test]
	fn a_field_written_twice_reads_as_the_last_one() {
		let value = read(r#"{ "a": 1, "a": 2 }"#);

		assert_eq!(value.get("a").and_then(Value::as_u32), Some(2));
		assert_eq!(value.as_object().len(), 2, "and both are still there to be walked");
	}

	#[test]
	fn a_refusal_names_the_line_and_the_column() {
		let message = refusal("{\n  \"a\": 1,\n  \"b\": x\n}");

		assert!(message.contains("line 3"), "the third line is where it is wrong: {message}");
		assert!(message.contains("column 8"), "under the x: {message}");
	}

	#[test]
	fn a_column_is_counted_in_characters_rather_than_bytes() {
		let message = refusal("[\"\u{444}\u{444}\", x]");

		assert!(
			message.contains("column 8"),
			"two letters are two columns, not four bytes: {message}"
		);
	}

	#[test]
	fn the_shape_a_real_document_has_reads_the_way_it_is_walked() {
		let text = r#"
			{
				"asset": { "version": "2.0", "generator": "something" },
				"meshes": [
					{
						"name": "shade",
						"primitives": [
							{ "attributes": { "POSITION": 0, "NORMAL": 1 }, "indices": 2 }
						]
					}
				],
				"materials": [ { "pbrMetallicRoughness": { "metallicFactor": 0.0 } } ]
			}
		"#;
		let document = read(text);

		assert_eq!(
			document
				.get("asset")
				.and_then(|asset| asset.get("version"))
				.and_then(Value::as_str),
			Some("2.0")
		);

		let primitive = &document
			.get("meshes")
			.map(Value::as_array)
			.unwrap_or_default()[0]
			.get("primitives")
			.map(Value::as_array)
			.unwrap_or_default()[0];

		assert_eq!(primitive.get("indices").and_then(Value::as_usize), Some(2));

		let attributes = primitive
			.get("attributes")
			.map(Value::as_object)
			.unwrap_or_default();
		let named: Vec<&str> = attributes
			.iter()
			.map(|(name, _)| name.as_str())
			.collect();

		assert_eq!(named, vec!["POSITION", "NORMAL"], "in the order they were written");
		assert_eq!(
			document
				.get("materials")
				.map(Value::as_array)
				.unwrap_or_default()[0]
				.get("pbrMetallicRoughness")
				.and_then(|pbr| pbr.get("metallicFactor"))
				.and_then(Value::as_f32),
			Some(0.0)
		);
	}
}
