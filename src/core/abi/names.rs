//! What things in the world are called.
//!
//! Three tables - entities, bodies and joints - are generational arrays of
//! plain data, and all three now carry a name beside everything else they
//! carry. A name is not an identifier: the identifier is the handle, and it is
//! unique by construction. A name is the human half, and nothing here enforces
//! that two of them differ.
//!
//! **Why the world holds it and not only the scene file.** A scene record has
//! had a `name` field since the format existed, and nothing could ever fill it
//! on the way out, because an entity had nothing to be called. So a world
//! written down and read back lost every name in it, which makes a name
//! useless for the one thing it is for: pointing at a particular thing across
//! a save, a reload or a restart. A tree that lists what is in the world, and
//! a selection that survives the world being replaced, both stand on this.
//!
//! **Why uniqueness is not enforced here.** Duplicating a scene twice would
//! have to fail or rename, and both are worse than allowing it: the handle is
//! what has to be unique and already is. Where uniqueness genuinely matters -
//! a scene *source*, whose bodies name their entities in text - it is made at
//! the moment of writing rather than carried around all the time.
//!
//! The storage is one `Vec<String>` per table, indexed by slot exactly like
//! the arrays beside it. It is deliberately not a map: the tables are dense,
//! and a slot with no name costs an empty `String` rather than an absence that
//! has to be spelled.

/// The longest a name may be, in characters.
///
/// Anything past this is cut, by characters rather than by bytes so that
/// cutting cannot leave half of one. The same rule and the same reason as a
/// debug label: a name is something a person reads in a list, and one that
/// does not fit in the list was not a name.
pub const MAX_NAME: usize = 64;

/// One name per slot of a table.
///
/// Crate-private on purpose: the three tables expose it through their own
/// handles, and a name reached by raw slot index would be a second way to
/// address something the whole ABI addresses by handle.
#[derive(Clone, Debug, Default)]
pub(crate) struct Names {
	names: Vec<String>,
}

impl Names {
	/// No names at all.
	pub(crate) const fn new() -> Self { Self { names: Vec::new() } }

	/// What lives in a slot, or the empty string.
	///
	/// A slot past the end reads as unnamed rather than panicking: this is
	/// reached through a handle that has already been resolved, so being out
	/// of range means the tables disagree with each other, and answering
	/// "nothing" is the honest report of that.
	///
	/// @param slot - the array index, not a handle
	pub(crate) fn at(&self, slot: usize) -> &str {
		self.names.get(slot).map_or("", String::as_str)
	}

	/// Names a slot, cutting anything past [`MAX_NAME`].
	///
	/// The empty name clears it, and that is the *only* way a name is ever
	/// cleared: a table clears the name at the one moment a slot becomes
	/// somebody's, which is when it is handed out. Clearing on the way out as
	/// well - at a despawn, or at a table-wide clear - was tried and removed,
	/// because each of the three covered the other two and no mutation of any
	/// one of them could fail a test. What is left is the one that does not
	/// depend on how the slot reached the free list.
	///
	/// @param slot - the array index, not a handle
	/// @param name - what to call it; empty clears the name
	pub(crate) fn set(&mut self, slot: usize, name: &str) {
		let Some(held) = self.names.get_mut(slot) else {
			return;
		};

		held.clear();
		held.extend(name.chars().take(MAX_NAME));
	}

	/// How many slots there is a name for.
	///
	/// Spelled the way the tables spell it rather than `len`, because it is
	/// the same number: this array has one entry per slot the table has ever
	/// handed out. Nothing in the engine asks - the three tables index it with
	/// a slot they have already resolved - so this exists for the three tests
	/// that assert the arrays have not drifted apart, which is the failure
	/// that would otherwise show up as an index panic in the host.
	#[cfg(test)]
	pub(crate) fn slots(&self) -> usize { self.names.len() }

	/// Adds a slot, unnamed.
	///
	/// Called from the same place a table grows its other arrays, so the
	/// lengths cannot drift apart.
	pub(crate) fn push(&mut self) { self.names.push(String::new()); }

	/// Sizes the array for a restore, with everything unnamed.
	///
	/// The names a restore puts back are written afterwards through the
	/// table's own handles, because that is the only way a caller can address
	/// what a restore just handed it.
	///
	/// @param slots - how many slots the table ends up with
	pub(crate) fn reset(&mut self, slots: usize) {
		self.names.clear();
		self.names.resize(slots, String::new());
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_slot_with_nothing_in_it_is_unnamed() {
		let mut names = Names::new();
		names.push();

		assert_eq!(names.at(0), "", "a fresh slot has no name");
		assert_eq!(names.at(7), "", "and neither does one that does not exist");
	}

	#[test]
	fn a_name_is_kept_and_replaced() {
		let mut names = Names::new();
		names.push();

		names.set(0, "crate");
		assert_eq!(names.at(0), "crate");

		names.set(0, "barrel");
		assert_eq!(names.at(0), "barrel", "naming again replaces rather than appends");

		names.set(0, "");
		assert_eq!(names.at(0), "", "and the empty name clears it");
	}

	#[test]
	fn naming_a_slot_that_is_not_there_does_nothing() {
		let mut names = Names::new();
		names.set(3, "nowhere");

		assert_eq!(names.at(3), "", "there was no slot to write into");
	}

	#[test]
	fn a_name_is_cut_at_the_limit_by_characters() {
		let mut names = Names::new();
		names.push();

		// two-byte characters, so a cut by bytes would land in the middle of
		// one and the string would not be valid at all.
		let long: String = std::iter::repeat_n('\u{044F}', MAX_NAME + 20).collect();
		names.set(0, &long);

		assert_eq!(names.at(0).chars().count(), MAX_NAME, "cut to the limit");
		assert!(names.at(0).chars().all(|c| c == '\u{044F}'), "and cut cleanly");
	}

	#[test]
	fn a_reset_sizes_the_array_and_empties_it() {
		let mut names = Names::new();
		names.push();
		names.set(0, "crate");

		names.reset(3);

		assert_eq!(names.at(0), "", "the old name is gone");
		assert_eq!(names.at(2), "", "and the new slots are there to be written");

		names.set(2, "floor");
		assert_eq!(names.at(2), "floor");
	}
}
