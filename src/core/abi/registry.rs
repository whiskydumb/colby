//! The shape every named, reloadable resource table shares.
//!
//! Meshes were the first. Textures and materials are the second and third, and
//! three is where the pattern stops being a coincidence: a table the host fills
//! from `assets/`, addressed by a handle the game keeps, where reloading a file
//! **replaces an entry in place** rather than appending a new one. The handle
//! staying valid is the whole mechanism - the game never re-resolves, and the
//! renderer notices only because it compares [`Entry::revision`] against what
//! it last uploaded.
//!
//! Slot zero is always present and always means "nothing", so a handle that was
//! never set resolves to something harmless rather than to nothing at all.
//!
//! Handles are per-table newtypes rather than one generic `Handle<T>`: they
//! cross the ABI as `#[repr(C)]` plain data, and a `MeshId` that could be
//! passed where a `TextureId` belongs is a bug this costs nothing to make
//! impossible. Only the storage is shared.
//!
//! An entry also carries the **identity** of the asset in it, and that is what
//! makes a rename survivable. Rename a source and it compiles under a new name,
//! so a table keyed only by name appends a second entry and leaves every handle
//! the world is holding pointing at the first - which goes on resolving and
//! draws nothing, because the file behind it is gone. @ref [`Registry::adopt`],
//! and [`ident`](super::ident) for what an identity is and is not.

use super::ident::Id;

/// Gives one of the wrapper tables the two methods an identity needs.
///
/// Not a general-purpose macro, and here for the reason
/// [`registry_handle!`](crate::registry_handle) is: twelve tables wrap a
/// [`Registry`] behind a handle type of their own, and the two forwards below
/// are identical in every one of them but the handle. Written out twelve times
/// they would drift, and a table that quietly lost `adopt` would be a table
/// whose assets stop drawing when somebody renames one.
///
/// The handle must be one [`registry_handle!`](crate::registry_handle) made,
/// and the field is named because one table calls its storage something else.
#[macro_export]
macro_rules! registry_identity {
	($table:ty, $handle:ty, $held:ident) => {
		impl $table {
			/// Which of these an identity belongs to, or nothing.
			///
			/// @param id - the identity
			#[must_use]
			pub fn find_by_id(&self, id: $crate::abi::ident::Id) -> $handle {
				<$handle>::new(self.$held.find_by_id(id))
			}

			/// Files a name against an identity, moving an entry that already
			/// has it.
			///
			/// @ref [`Registry::adopt`](
			/// $crate::abi::registry::Registry::adopt) for the whole of what
			/// this means; it is called by the host's asset loop and by
			/// nothing else.
			///
			/// @param name - the asset name it is known by now
			/// @param id - its identity
			pub fn adopt(&mut self, name: &str, id: $crate::abi::ident::Id) -> $handle {
				<$handle>::new(self.$held.adopt(name, id))
			}
		}
	};
}

/// Declares a `#[repr(C)]` handle into one of these tables.
///
/// Not a general-purpose macro. It exists because the three registries need
/// three handle types identical in every way except which table they index, and
/// writing that out three times invites the fourth to drift.
#[macro_export]
macro_rules! registry_handle {
	($(#[$attribute:meta])* $name:ident) => {
		$(#[$attribute])*
		#[repr(C)]
		#[derive(
			Clone,
			Copy,
			Debug,
			PartialEq,
			Eq,
			Hash,
			$crate::bytemuck::Pod,
			$crate::bytemuck::Zeroable,
		)]
		pub struct $name(u32);

		impl $name {
			/// Refers to nothing. What anything unset holds.
			pub const NONE: Self = Self(0);

			/// A handle to a slot.
			#[must_use]
			pub const fn new(index: u32) -> Self { Self(index) }

			/// The slot this addresses.
			#[must_use]
			pub const fn index(self) -> u32 { self.0 }

			/// The same slot, as an index into a slice.
			#[must_use]
			#[expect(
				clippy::as_conversions,
				reason = "u32 to usize is lossless on every target this builds for, and \
				          try_from is not available in a const fn"
			)]
			pub const fn slot(self) -> usize { self.0 as usize }

			/// Whether it refers to anything at all.
			#[must_use]
			pub const fn is_some(self) -> bool { self.0 != 0 }
		}

		impl Default for $name {
			fn default() -> Self { Self::NONE }
		}
	};
}

/// One entry: what it is called, what it *is*, what it holds, and how many
/// times it has changed.
#[derive(Clone, Debug)]
pub struct Entry<T> {
	name: String,
	id: Id,
	value: T,
	revision: u32,
}

impl<T> Entry<T> {
	/// The name this entry is registered under.
	#[must_use]
	pub fn name(&self) -> &str { &self.name }

	/// The identity of the asset in it, or [`Id::NONE`].
	///
	/// Filled by whoever loads the table - for a project's assets that is the
	/// host's asset loop, out of the tree's own table of identities - and left
	/// as nothing for a table filled by hand, which is what a test and a bake
	/// do. @ref [`Registry::adopt`].
	#[must_use]
	pub const fn id(&self) -> Id { self.id }

	/// What is in it.
	#[must_use]
	pub const fn value(&self) -> &T { &self.value }

	/// What is in it, to change.
	///
	/// Taking this counts as a change: the revision goes up whether or not
	/// anything is written, because there is no way to find out afterwards and
	/// an unnecessary re-upload is cheaper than a missed one.
	pub const fn value_mut(&mut self) -> &mut T {
		self.revision = self.revision.saturating_add(1);

		&mut self.value
	}

	/// How many times the value has been replaced or handed out mutably.
	///
	/// Whoever turns this into something on the GPU keeps the number it last
	/// saw and acts when the two disagree. Nothing else needs to know.
	#[must_use]
	pub const fn revision(&self) -> u32 { self.revision }
}

/// A table of named values, addressed by index, that only ever grows.
#[derive(Clone, Debug)]
pub struct Registry<T> {
	entries: Vec<Entry<T>>,
}

impl<T> Registry<T> {
	/// A table holding nothing but its null entry.
	///
	/// @param nothing - what slot zero holds; whatever "draws nothing" means
	/// for this kind of resource
	pub fn new(nothing: T) -> Self {
		let mut registry = Self { entries: Vec::with_capacity(4) };
		registry.push("", nothing);

		registry
	}

	/// Looks an entry up by name.
	///
	/// A linear scan. With the number of resources one scene has that beats a
	/// hash map, and callers are expected to resolve a name once and keep the
	/// index.
	///
	/// @param name - the name it was registered under
	/// @return its index, or zero if nothing answers to that name
	#[must_use]
	pub fn find(&self, name: &str) -> u32 {
		if name.is_empty() {
			return 0;
		}

		self.entries
			.iter()
			.position(|entry| entry.name == name)
			.and_then(|index| u32::try_from(index).ok())
			.unwrap_or(0)
	}

	/// Registers a value under a name, replacing whatever was there.
	///
	/// A name already in the table keeps its index; the entry's contents are
	/// replaced and its revision goes up.
	///
	/// @param name - what to register it as
	/// @param value - the value
	/// @return the index, the same one as last time if the name is known
	pub fn insert(&mut self, name: &str, value: T) -> u32 {
		let existing = self.find(name);
		if existing == 0 {
			return self.push(name, value);
		}

		let Some(entry) = self.entry_mut(existing) else {
			return self.push(name, value);
		};

		entry.value = value;
		entry.revision = entry.revision.saturating_add(1);

		existing
	}

	/// One entry, by index.
	#[must_use]
	pub fn entry(&self, index: u32) -> Option<&Entry<T>> {
		self.entries.get(usize::try_from(index).ok()?)
	}

	/// One entry, by index, to change.
	pub fn entry_mut(&mut self, index: u32) -> Option<&mut Entry<T>> {
		self.entries.get_mut(usize::try_from(index).ok()?)
	}

	/// How many entries there are, counting the null one.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Always `false`: slot zero always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// Every entry, in slot order, starting with the null one.
	pub fn iter(&self) -> impl Iterator<Item = &Entry<T>> { self.entries.iter() }

	/// Looks an entry up by the identity of the asset in it.
	///
	/// A linear scan, as [`find`](Self::find) is and for its reason. Nothing
	/// resolves an asset this way to *draw* it - a scene, the console, a
	/// program and the wire all name things - so this is asked once by a person
	/// pasting an identity into a panel, and by the loop that moves an entry
	/// when its file is renamed.
	///
	/// @param id - the identity
	/// @return its index, or zero when nothing here carries it
	#[must_use]
	pub fn find_by_id(&self, id: Id) -> u32 {
		if id.is_none() {
			return 0;
		}

		self.entries
			.iter()
			.position(|entry| entry.id == id)
			.and_then(|index| u32::try_from(index).ok())
			.unwrap_or(0)
	}

	/// Files a name against an identity, moving an entry that already has it.
	///
	/// **This is how a rename reaches a world that is already running.** Called
	/// by whoever loads the table, both before the value goes in and after:
	///
	/// - an entry already carries this identity under a **different** name, so
	///   the file behind it was renamed. The entry is renamed in place, which
	///   is the whole point - it keeps its slot, so every handle the world is
	///   holding goes on resolving, and the value compiled under the new name
	///   lands in the entry it always had;
	/// - an entry under this name carries it already, or takes it now;
	/// - neither, and nothing happens. An asset nothing has loaded yet has no
	///   entry to file anything against, which is why this is called a second
	///   time once the value is in.
	///
	/// **Nothing is ever made here**, and that is not tidiness: an entry made
	/// empty and filled an instant later would be an entry *replaced*, which
	/// moves its revision, and a revision that moves is a re-upload. A first
	/// load has to read as a first load.
	///
	/// [`Id::NONE`] takes the identity away from the entry under this name, if
	/// there is one. That is an asset whose file has gone.
	///
	/// @param name - the asset name it is known by now
	/// @param id - its identity
	/// @return the entry's index, or zero when there was nothing to do
	pub fn adopt(&mut self, name: &str, id: Id) -> u32 {
		if id.is_none() {
			let index = self.find(name);

			if index != 0
				&& let Some(entry) = self.entry_mut(index)
			{
				entry.id = Id::NONE;
			}

			return index;
		}

		let index = match self.find_by_id(id) {
			| 0 => self.find(name),
			| held => held,
		};

		// slot zero is the null entry and belongs to nobody: renaming it would
		// give "nothing" a name and an identity, and every handle that was
		// never set points at it.
		if index == 0 {
			return 0;
		}

		let Some(entry) = self.entry_mut(index) else {
			return 0;
		};

		name.clone_into(&mut entry.name);
		entry.id = id;

		index
	}

	/// Appends a new entry.
	fn push(&mut self, name: &str, value: T) -> u32 {
		let Ok(index) = u32::try_from(self.entries.len()) else {
			return 0;
		};

		self.entries.push(Entry {
			name: name.to_owned(),
			id: Id::NONE,
			value,
			revision: 0,
		});

		index
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn slot_zero_is_the_null_entry_and_answers_to_nothing() {
		let registry = Registry::new(7_u32);

		assert_eq!(registry.len(), 1, "a new table holds only its null entry");
		assert_eq!(registry.find(""), 0, "the empty name is not a name");
		assert_eq!(registry.find("anything"), 0, "and neither is one nobody used");
		assert_eq!(
			registry.entry(0).map(|entry| *entry.value()),
			Some(7),
			"slot zero holds what it was given"
		);
	}

	#[test]
	fn a_name_takes_a_slot_and_keeps_it() {
		let mut registry = Registry::new(0_u32);
		let first = registry.insert("thing", 1);
		let second = registry.insert("thing", 2);

		assert_ne!(first, 0, "a registered value is not the null one");
		assert_eq!(first, second, "the same name is the same slot");
		assert_eq!(registry.len(), 2, "so nothing was appended the second time");
		assert_eq!(
			registry.entry(first).map(|entry| *entry.value()),
			Some(2),
			"and the value really is the new one"
		);
		assert_eq!(
			registry.entry(first).map(Entry::revision),
			Some(1),
			"which is what the revision reports"
		);
	}

	#[test]
	fn taking_a_value_mutably_counts_as_changing_it() {
		let mut registry = Registry::new(0_u32);
		let index = registry.insert("thing", 1);

		*registry
			.entry_mut(index)
			.expect("the entry is there")
			.value_mut() = 9;

		assert_eq!(
			registry.entry(index).map(Entry::revision),
			Some(1),
			"there is no way to find out afterwards, so it is assumed"
		);
		assert_eq!(
			registry.entry(index).map(|entry| *entry.value()),
			Some(9),
			"and the write landed"
		);
	}

	#[test]
	fn entries_come_back_in_slot_order() {
		let mut registry = Registry::new(0_u32);
		registry.insert("one", 1);
		registry.insert("two", 2);

		let seen: Vec<u32> = registry
			.iter()
			.map(|entry| *entry.value())
			.collect();

		assert_eq!(seen, vec![0, 1, 2], "the null entry first, then the order they arrived");
	}

	#[test]
	fn an_entry_carries_no_identity_until_one_is_filed() {
		let mut registry = Registry::new(0_u32);
		let index = registry.insert("thing", 1);

		assert_eq!(registry.entry(index).map(Entry::id), Some(Id::NONE), "nobody said one");
		assert_eq!(registry.find_by_id(Id::NONE), 0, "and nothing is not something to find");

		let id = Id::from_bits(7);
		registry.adopt("thing", id);

		assert_eq!(registry.entry(index).map(Entry::id), Some(id), "now it has one");
		assert_eq!(registry.find_by_id(id), index, "and it answers to it");
		assert_eq!(registry.len(), 2, "with nothing appended");
	}

	#[test]
	fn a_name_that_is_not_there_yet_is_left_alone_and_takes_its_identity_after() {
		// what a first load looks like, and why nothing is made here: an entry
		// made empty and filled an instant later would be an entry replaced,
		// and a replaced entry's revision moves, and a revision that moves is
		// a re-upload of something nobody had.
		let mut registry = Registry::new(0_u32);
		let id = Id::from_bits(3);

		assert_eq!(registry.adopt("thing", id), 0, "there is nothing to file it against");
		assert_eq!(registry.len(), 1, "and nothing was made");

		let index = registry.insert("thing", 5);

		assert_eq!(registry.adopt("thing", id), index, "the second call finds it");
		assert_eq!(registry.find_by_id(id), index);
		assert_eq!(
			registry.entry(index).map(Entry::revision),
			Some(0),
			"and it is still a first load"
		);
	}

	#[test]
	fn slot_zero_takes_no_name_and_no_identity() {
		// every handle that was never set points at it, so a table that let it
		// be renamed would let "nothing" become an asset.
		let mut registry = Registry::new(0_u32);
		let id = Id::from_bits(9);

		assert_eq!(registry.adopt("", id), 0, "the empty name is not a name");
		assert_eq!(registry.adopt("nobody", id), 0, "and neither is one nothing answers to");
		assert_eq!(registry.entry(0).map(Entry::id), Some(Id::NONE), "so it has none");
		assert_eq!(registry.entry(0).map(Entry::name), Some(""), "and is still called nothing");
	}

	#[test]
	fn an_entry_whose_file_was_renamed_moves_rather_than_being_appended_to() {
		// the whole reason an entry carries an identity. A handle the world is
		// holding has to go on resolving to the thing it resolved to, and a
		// table keyed only by name would leave it pointing at the old entry.
		let mut registry = Registry::new(0_u32);
		let id = Id::from_bits(11);
		let held = registry.insert("meshes/crystal", 1);
		registry.adopt("meshes/crystal", id);

		let moved = registry.adopt("meshes/gem", id);
		registry.insert("meshes/gem", 2);

		assert_eq!(moved, held, "the same slot, so the same handle");
		assert_eq!(registry.len(), 2, "and nothing was appended");
		assert_eq!(registry.find("meshes/gem"), held, "the new name answers");
		assert_eq!(registry.find("meshes/crystal"), 0, "and the old one does not");
		assert_eq!(
			registry.entry(held).map(|entry| *entry.value()),
			Some(2),
			"holding what was compiled under the new name"
		);
	}

	#[test]
	fn nothing_filed_against_a_name_takes_its_identity_away() {
		// what an asset whose file has gone looks like: the entry stays, so a
		// handle still resolves, and it answers to nobody's identity.
		let mut registry = Registry::new(0_u32);
		let id = Id::from_bits(5);
		registry.adopt("thing", id);
		let index = registry.insert("thing", 1);

		assert_eq!(registry.adopt("thing", Id::NONE), index, "the same entry");
		assert_eq!(registry.entry(index).map(Entry::id), Some(Id::NONE));
		assert_eq!(registry.find_by_id(id), 0, "and the identity is nobody's");
		assert_eq!(registry.adopt("nowhere", Id::NONE), 0, "a name nobody knows makes nothing");
		assert_eq!(registry.len(), 2);
	}

	#[test]
	fn an_index_past_the_end_reaches_nothing() {
		let mut registry = Registry::new(0_u32);

		assert!(registry.entry(99).is_none(), "there is no slot 99");
		assert!(registry.entry_mut(99).is_none(), "not to write to either");
	}
}
