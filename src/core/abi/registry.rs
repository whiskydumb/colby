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

/// One entry: what it is called, what it is, and how many times it has changed.
#[derive(Clone, Debug)]
pub struct Entry<T> {
	name: String,
	value: T,
	revision: u32,
}

impl<T> Entry<T> {
	/// The name this entry is registered under.
	#[must_use]
	pub fn name(&self) -> &str { &self.name }

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

	/// Appends a new entry.
	fn push(&mut self, name: &str, value: T) -> u32 {
		let Ok(index) = u32::try_from(self.entries.len()) else {
			return 0;
		};

		self.entries.push(Entry {
			name: name.to_owned(),
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
	fn an_index_past_the_end_reaches_nothing() {
		let mut registry = Registry::new(0_u32);

		assert!(registry.entry(99).is_none(), "there is no slot 99");
		assert!(registry.entry_mut(99).is_none(), "not to write to either");
	}
}
