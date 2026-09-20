//! Translations: a table of strings a person reads, and the language to read
//! them in.
//!
//! Here rather than in the interface crate for the reason [`font`](super::font)
//! is: the words a box holds are an input to the layout, and the layout runs
//! wherever the world is. A translation reached only from the renderer would be
//! a translation the game could not ask for.
//!
//! **A key is marked in the text itself.** A string beginning with [`SIGIL`] is
//! a key, whether it is a run of words in a `.html` or whatever a game last
//! wrote with [`Ui::set_text`](super::ui::Ui::set_text). That is the Source
//! convention and s&box still keeps it, and it is what makes one hook cover
//! both: the alternative, an attribute in the markup, reaches the document and
//! not the half of the words a game writes from code. A literal `#` at the
//! head of a string is written twice.
//!
//! **A field's value is not translated**, and that is the one exception with a
//! reason rather than an omission: an `<input>` holds what somebody typed, the
//! editing keys write it back through the same `set_text`, and a value put
//! through a lookup on its way to the screen would be replaced by its own key
//! the first time a letter was pressed.
//!
//! **A key nobody translated is answered with the key**, after the fallback
//! language has been asked. Three engines end in exactly that place - the
//! fourth leaves the widget holding whatever it had - and none of the four
//! refuses to build. A translation is always missing something; that is its
//! normal state and not an error.
//!
//! ```text
//! "#menu.play"  ->  lang/ru has it   ->  the Russian
//!               ->  lang/en has it   ->  the English
//!               ->  nobody has it    ->  "menu.play"
//! ```
//!
//! What language is being read, and what to fall back to, are two saved
//! console variables. Asking the operating system was considered and refused:
//! @ref `colby_runtime::console` for the whole of the reason.

use super::registry::{Entry, Registry};
use crate::registry_handle;

/// The character that marks a string as a key rather than as words.
pub const SIGIL: char = '#';

/// The console variable holding the language being read.
pub const LANGUAGE: &str = "loc.language";

/// The one holding the language to fall back to.
pub const FALLBACK: &str = "loc.fallback";

/// The one that shows keys instead of the words they stand for.
pub const SHOW_KEYS: &str = "loc.showkeys";

/// The language both variables start out naming.
pub const DEFAULT_LANGUAGE: &str = "en";

/// What a language's asset name starts with.
///
/// A language is an ordinary asset, so `assets/lang/ru.loc` compiles to
/// `lang/ru` and the variable holding `ru` names the tail of it. The variable
/// is the short word rather than the whole name because it is a thing a person
/// types.
pub const PREFIX: &str = "lang/";

registry_handle! {
	/// Which translation in [`Translations`].
	///
	/// Not generational, like every other resource handle here. @ref
	/// [`registry`](super::registry).
	LangId
}

/// One entry of the translation registry.
pub type Lang = Entry<LangData>;

/// One language: every key somebody translated, and what into.
///
/// The pairs are **sorted by key**, so a lookup is a binary search. Sorted by
/// the compiler rather than here: a `.cloc` is written once and read on every
/// load and every reload, and the order is a property of the file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LangData {
	/// Every pair, sorted by key, with no key written twice.
	pub strings: Vec<(String, String)>,
}

/// The empty table, so [`Translations::data`] has something to borrow.
static EMPTY: LangData = LangData { strings: Vec::new() };

impl LangData {
	/// A language nobody has translated anything into.
	#[must_use]
	pub fn empty() -> Self { EMPTY.clone() }

	/// Whether there is anything in it.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.strings.is_empty() }

	/// How many keys it holds.
	#[must_use]
	pub fn len(&self) -> usize { self.strings.len() }

	/// What one key stands for here.
	///
	/// @param key - the key, without [`SIGIL`]
	/// @return the words, or `None` when this language does not have it
	#[must_use]
	pub fn get(&self, key: &str) -> Option<&str> {
		self.strings
			.binary_search_by(|(written, _)| written.as_str().cmp(key))
			.ok()
			.and_then(|index| self.strings.get(index))
			.map(|(_, words)| words.as_str())
	}
}

/// Every language the host has loaded, addressed by [`LangId`].
///
/// Slot zero is [`LangId::NONE`] and is empty, so a project with no
/// translations at all answers every key with the key rather than with nothing.
#[derive(Clone, Debug)]
pub struct Translations {
	entries: Registry<LangData>,
}

impl Translations {
	/// A registry holding nothing but the empty language.
	#[must_use]
	pub fn new() -> Self {
		Self {
			entries: Registry::new(LangData::empty()),
		}
	}

	/// Looks a language up by its whole asset name, e.g. `lang/ru`.
	#[must_use]
	pub fn find(&self, name: &str) -> LangId { LangId::new(self.entries.find(name)) }

	/// Looks one up by the short word a variable holds, e.g. `ru`.
	///
	/// @param language - the tail of the asset name, without [`PREFIX`]
	#[must_use]
	pub fn of_language(&self, language: &str) -> LangId {
		if language.is_empty() {
			return LangId::NONE;
		}

		self.find(&format!("{PREFIX}{language}"))
	}

	/// Registers a table under a name, replacing whatever was there.
	pub fn insert(&mut self, name: &str, data: LangData) -> LangId {
		LangId::new(self.entries.insert(name, data))
	}

	/// One language, by handle.
	#[must_use]
	pub fn get(&self, id: LangId) -> Option<&Lang> { self.entries.entry(id.index()) }

	/// One language's table, by handle, falling back to the empty one.
	#[must_use]
	pub fn data(&self, id: LangId) -> &LangData {
		self.entries
			.entry(id.index())
			.map_or(&EMPTY, Entry::value)
	}

	/// How many languages there are, counting the null one.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Always `false`: slot zero always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// Every language, in slot order.
	pub fn iter(&self) -> impl Iterator<Item = &Lang> { self.entries.iter() }
}

impl Default for Translations {
	fn default() -> Self { Self::new() }
}

/// Whether a string is a key rather than words, and which key.
///
/// A doubled [`SIGIL`] is how a string that really does start with one is
/// written, and it is undone by [`unescape`] rather than here: this answers
/// "is this a lookup" and the two doubled cases are not.
///
/// @param text - whatever somebody wrote
/// @return the key, without the sigil, or `None` when the text is words
#[must_use]
pub fn key_of(text: &str) -> Option<&str> {
	let rest = text.strip_prefix(SIGIL)?;

	if rest.is_empty() || rest.starts_with(SIGIL) {
		return None;
	}

	Some(rest)
}

/// What a string that is not a key is drawn as.
///
/// Only ever different from its input for a string that begins with a doubled
/// [`SIGIL`], which is how a literal one is written.
///
/// @param text - whatever somebody wrote
#[must_use]
pub fn unescape(text: &str) -> &str {
	match text.strip_prefix(SIGIL) {
		| Some(rest) if rest.starts_with(SIGIL) => rest,
		| _ => text,
	}
}

// the two an identity needs, for the table above. @ref `registry_identity!`
crate::registry_identity!(Translations, LangId, entries);

#[cfg(test)]
mod tests {
	use super::*;

	/// A table of the pairs given, sorted the way the compiler leaves one.
	fn table(pairs: &[(&str, &str)]) -> LangData {
		let mut strings: Vec<(String, String)> = pairs
			.iter()
			.map(|(key, words)| ((*key).to_owned(), (*words).to_owned()))
			.collect();

		strings.sort_by(|left, right| left.0.cmp(&right.0));

		LangData { strings }
	}

	#[test]
	fn a_key_is_found_by_a_binary_search_over_sorted_pairs() {
		let data = table(&[("menu.quit", "Quit"), ("menu.play", "Play")]);

		assert_eq!(data.get("menu.play"), Some("Play"));
		assert_eq!(data.get("menu.quit"), Some("Quit"));
		assert_eq!(data.get("menu.save"), None, "and one nobody wrote is not there");
	}

	#[test]
	fn a_string_beginning_with_the_sigil_is_a_key() {
		assert_eq!(key_of("#menu.play"), Some("menu.play"));
		assert_eq!(key_of("menu.play"), None, "and one without it is words");
		assert_eq!(key_of("#"), None, "a sigil on its own names nothing");
	}

	#[test]
	fn a_doubled_sigil_is_how_a_literal_one_is_written() {
		assert_eq!(key_of("##1"), None, "so it is not a lookup");
		assert_eq!(unescape("##1"), "#1", "and one of the two is dropped when it is drawn");
		assert_eq!(unescape("#menu.play"), "#menu.play", "a key is left alone here");
		assert_eq!(unescape("plain"), "plain", "and so is everything else");
	}

	#[test]
	fn the_null_slot_is_a_language_that_translates_nothing() {
		let translations = Translations::new();

		assert_eq!(
			translations.find("lang/ru"),
			LangId::NONE,
			"nothing resolves in an empty table"
		);
		assert!(translations.data(LangId::NONE).is_empty());
	}

	#[test]
	fn a_language_is_found_by_the_short_word_a_variable_holds() {
		let mut translations = Translations::new();
		let id = translations.insert("lang/ru", table(&[("menu.play", "Play")]));

		assert_eq!(translations.of_language("ru"), id, "the tail of the asset name");
		assert_eq!(translations.of_language("de"), LangId::NONE, "one nobody compiled");
		assert_eq!(translations.of_language(""), LangId::NONE, "and the empty word is not one");
	}

	/// A world with two languages loaded and the two variables registered.
	///
	/// Registered here rather than borrowed from the runner, because the whole
	/// point of the chain below is what it does when a variable is *missing*,
	/// and a helper that always registers them could not show that.
	fn worlds() -> crate::abi::World {
		let mut world = crate::abi::World::new();

		world
			.translations
			.insert("lang/en", table(&[("menu.play", "Play"), ("menu.quit", "Quit")]));
		world
			.translations
			.insert("lang/ru", table(&[("menu.play", "Igrat")]));

		world
			.cvars
			.saved(LANGUAGE, crate::abi::Value::Text("ru".to_owned()), "");
		world
			.cvars
			.saved(FALLBACK, crate::abi::Value::Text("en".to_owned()), "");

		world
	}

	#[test]
	fn a_key_is_answered_by_the_language_then_the_fallback_then_itself() {
		let world = worlds();

		assert_eq!(world.text("menu.play"), "Igrat", "the language being read has it");
		assert_eq!(
			world.text("menu.quit"),
			"Quit",
			"nobody translated this one, so the fallback language answers"
		);
		assert_eq!(
			world.text("menu.save"),
			"menu.save",
			"and a key nobody at all has reads as itself, which is the last resort every 			 \
			 engine with one of these ends at"
		);
	}

	#[test]
	fn a_world_with_no_variables_registered_still_answers_every_key() {
		// the case a project that has never heard of translations is in, and
		// the one that must not be a blank screen: no `loc.language`, no
		// `loc.fallback`, no `assets/lang/` at all.
		let world = crate::abi::World::new();

		assert_eq!(world.text("menu.play"), "menu.play");
		assert_eq!(world.worded("#menu.play"), "menu.play", "the sigil comes off either way");
		assert_eq!(world.worded("Play"), "Play", "and plain words are left alone");
	}

	#[test]
	fn a_language_nobody_compiled_falls_all_the_way_through() {
		// how a misspelled `loc.language de` looks, and it must look like the
		// fallback rather than like a blank screen.
		let mut world = worlds();

		world.cvars.set(LANGUAGE, "de");

		assert_eq!(world.text("menu.play"), "Play", "the fallback still answers");
		assert_eq!(world.text("menu.save"), "menu.save");
	}

	#[test]
	fn showing_keys_wins_over_every_language_there_is() {
		let mut world = worlds();

		world
			.cvars
			.saved(SHOW_KEYS, crate::abi::Value::Bool(true), "");

		assert_eq!(
			world.text("menu.play"),
			"menu.play",
			"which is the whole of what the variable is for: seeing which words on the screen 			 are reachable at all"
		);
		assert_eq!(world.worded("Play"), "Play", "words that are not a key are still words");
	}

	#[test]
	fn a_fallback_that_is_the_language_is_not_searched_twice() {
		let mut world = worlds();

		world.cvars.set(FALLBACK, "ru");

		assert_eq!(
			world.text("menu.quit"),
			"menu.quit",
			"the English table is not consulted when nobody named it"
		);
	}

	#[test]
	fn what_a_game_wrote_goes_through_the_same_rule_as_the_document() {
		let world = worlds();

		assert_eq!(world.worded("#menu.play"), "Igrat", "a key wherever it came from");
		assert_eq!(world.worded("##menu.play"), "#menu.play", "and a literal sigil is doubled");
	}

	#[test]
	fn a_recompiled_language_keeps_its_slot() {
		let mut translations = Translations::new();
		let first = translations.insert("lang/ru", table(&[("a", "1")]));
		let again = translations.insert("lang/ru", table(&[("a", "2"), ("b", "3")]));

		assert_eq!(first, again, "the same name is the same handle");
		assert_eq!(translations.len(), 2, "and nothing was appended");
		assert_eq!(translations.data(first).len(), 2, "with the new table in it");
	}
}
