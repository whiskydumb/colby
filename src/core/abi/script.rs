//! Programs the host can run, and the registry they live in.
//!
//! A script is an asset like a mesh or a document: the compiler turns a `.lua`
//! into a `.clua`, the runner registers it under the path it was written at,
//! and everything downstream names it - `ui/hud`, `scripts/thruster`. Nothing
//! in `colby_core` runs one; this is the table, and the interpreter is the
//! host's. @ref `colby_script`.
//!
//! **Why a script is an asset rather than text carried inside whatever uses
//! it.** A document used to fold its program in, which made three things true
//! and all three were wrong: editing a shared program recompiled every document
//! naming it, editing a stylesheet restarted a program that had not changed,
//! and a program that belonged to no document had nowhere at all to live. One
//! table answers all three, and it is the table a program the *world* runs is
//! reached through as well.
//!
//! The handle is not generational, like every other resource handle here: a
//! name resolved once stays resolved for the life of the process, and
//! recompiling the source rewrites the entry the handle already points at. The
//! entry's [`revision`](Entry::revision) moving is the whole reload mechanism -
//! whoever is running the program compares it and builds again. @ref
//! [`registry`](super::registry).

use super::registry::{Entry, Registry};
use crate::registry_handle;

/// The source of one program.
///
/// Text rather than bytecode, which is the same decision `.cdoc` made: what a
/// compiled format buys is a load that does not parse, and a program is
/// kilobytes. Bytecode would also stop the file being readable and would tie it
/// to one interpreter's build.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScriptData {
	/// The program, as it was written.
	pub source: String,
}

impl ScriptData {
	/// A program with nothing in it.
	///
	/// What slot zero holds, and what an entry whose file has been deleted
	/// becomes. Running it does nothing, which is the honest reading of a
	/// program that is not there.
	#[must_use]
	pub const fn empty() -> Self { Self { source: String::new() } }

	/// Whether there is anything to run.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.source.trim().is_empty() }
}

registry_handle! {
	/// Which program in [`Scripts`].
	ScriptId
}

/// One entry of the script registry.
pub type Script = Entry<ScriptData>;

/// Every program the host has been handed, addressed by [`ScriptId`].
///
/// Slot zero is [`ScriptId::NONE`] and holds nothing, so a document naming a
/// program nobody compiled runs no program rather than failing to load.
#[derive(Clone, Debug)]
pub struct Scripts {
	entries: Registry<ScriptData>,
}

impl Scripts {
	/// A registry holding nothing but the empty program.
	#[must_use]
	pub fn new() -> Self {
		Self {
			entries: Registry::new(ScriptData::empty()),
		}
	}

	/// Looks a program up by name.
	///
	/// @return its handle, or [`ScriptId::NONE`] if nothing answers to it
	#[must_use]
	pub fn find(&self, name: &str) -> ScriptId { ScriptId::new(self.entries.find(name)) }

	/// Registers a program under a name, replacing whatever was there.
	///
	/// @return the handle, the same one as last time if the name is known
	pub fn insert(&mut self, name: &str, data: ScriptData) -> ScriptId {
		ScriptId::new(self.entries.insert(name, data))
	}

	/// One program, by handle.
	#[must_use]
	pub fn get(&self, id: ScriptId) -> Option<&Script> { self.entries.entry(id.index()) }

	/// How many programs there are, counting the empty one.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Always `false`: slot zero always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// Every program, in slot order.
	pub fn iter(&self) -> impl Iterator<Item = &Script> { self.entries.iter() }
}

impl Default for Scripts {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A program with something in it.
	fn program(text: &str) -> ScriptData { ScriptData { source: text.to_owned() } }

	#[test]
	fn a_fresh_registry_holds_the_empty_program_at_slot_zero() {
		let scripts = Scripts::new();

		assert_eq!(scripts.len(), 1, "slot zero and nothing else");
		assert_eq!(scripts.find("scripts/nobody"), ScriptId::NONE);
		assert!(
			scripts
				.get(ScriptId::NONE)
				.is_some_and(|entry| entry.value().is_empty()),
			"and it resolves to a program with nothing in it"
		);
	}

	#[test]
	fn a_name_keeps_its_handle_when_the_program_is_rewritten() {
		// the whole reload mechanism: a handle resolved once stays resolved,
		// and what moves is the revision. A generational handle here would
		// break exactly the property that makes editing a file under a running
		// process work.
		let mut scripts = Scripts::new();
		let first = scripts.insert("scripts/thruster", program("local a = 1"));
		let was = scripts.get(first).map_or(0, Entry::revision);

		let again = scripts.insert("scripts/thruster", program("local a = 2"));

		assert_eq!(first, again, "the same slot answers to the same name");
		assert_eq!(
			scripts
				.get(again)
				.map(|entry| entry.value().source.as_str()),
			Some("local a = 2"),
			"holding what was written last"
		);
		assert!(
			scripts
				.get(again)
				.is_some_and(|entry| entry.revision() > was),
			"and saying that it moved"
		);
	}

	#[test]
	fn a_program_of_only_whitespace_is_a_program_with_nothing_in_it() {
		// what the runner does with it is skip building an environment for it,
		// so the question is asked here rather than in three places. A file
		// somebody emptied should behave as a file that was never there.
		assert!(program("").is_empty());
		assert!(program("\n\t  \n").is_empty());
		assert!(!program("-- a comment is a program").is_empty());
	}

	#[test]
	fn every_program_is_walked_in_slot_order() {
		let mut scripts = Scripts::new();

		for name in ["scripts/b", "scripts/a", "scripts/c"] {
			scripts.insert(name, program("return 1"));
		}

		let names: Vec<&str> = scripts.iter().map(Entry::name).collect();

		assert_eq!(
			names,
			["", "scripts/b", "scripts/a", "scripts/c"],
			"the order they were registered in, which is the order the tree was walked in - not \
			 alphabetical, and not a hash order"
		);
	}
}
