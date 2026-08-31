//! The mixer's own copy of every sound.
//!
//! A copy rather than a borrow, and that is the design rather than an
//! oversight. The registry the samples come from lives in `World`, which the
//! game writes and the asset loader rewrites while the simulation is running;
//! whatever is turning samples into a device buffer cannot hold a reference
//! into that and cannot be stopped while it is being replaced. So the mixer
//! keeps its own, and the copies are made where a copy is allowed to be slow.
//!
//! **What tells it a sound has changed is the registry's revision**, exactly
//! as the renderer decides which meshes to upload again. A recompiled `.wav`
//! rewrites the entry the handle already points at, the revision moves, and
//! the next [`Bank::sync`] copies it. Nothing else has to be told anything.
//!
//! The cost is the samples held twice. That is a few megabytes for what a
//! sandbox has and it is the same bargain every engine makes with its GPU
//! buffers.

use colby_core::abi::{SoundData, SoundId, Sounds};

/// One sound, and which version of it this is.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Recording {
	/// The samples, as the registry had them.
	pub data: SoundData,

	/// The registry revision this was copied at.
	pub revision: u32,
}

/// Every sound the mixer can reach, by the same slot the registry uses.
///
/// Addressed by [`SoundId`] and nothing else, so a voice carries a handle
/// across rather than a name or a pointer.
#[derive(Clone, Debug, Default)]
pub struct Bank {
	recordings: Vec<Option<Recording>>,
}

impl Bank {
	/// A bank with nothing in it.
	#[must_use]
	pub const fn new() -> Self { Self { recordings: Vec::new() } }

	/// Copies whatever has changed since the last time this was asked.
	///
	/// Called from wherever assets are loaded, which is a place a copy of a
	/// megabyte is unremarkable. Never from anything filling a device buffer.
	///
	/// @param sounds - the registry, as the world has it
	/// @return how many recordings were copied, for a log line; zero is the
	/// answer on all but the first call and the ones after an edit
	pub fn sync(&mut self, sounds: &Sounds) -> usize {
		self.recordings.resize(sounds.len(), None);

		let mut copied = 0;
		for (slot, entry) in sounds.iter().enumerate() {
			let current = self.recordings.get(slot).and_then(Option::as_ref);
			if current.is_some_and(|held| held.revision == entry.revision()) {
				continue;
			}

			self.recordings[slot] = Some(Recording {
				data: entry.value().clone(),
				revision: entry.revision(),
			});
			copied += 1;
		}

		copied
	}

	/// One recording, by the handle a voice carries.
	///
	/// @return nothing for a handle this bank has not been told about, which
	/// is what a sound compiled since the last [`sync`](Self::sync) looks like
	#[must_use]
	pub fn get(&self, id: SoundId) -> Option<&SoundData> {
		self.recordings
			.get(id.slot())
			.and_then(Option::as_ref)
			.map(|recording| &recording.data)
	}

	/// How many slots it holds, copied or not.
	#[must_use]
	pub fn len(&self) -> usize { self.recordings.len() }

	/// Whether it holds nothing at all.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.recordings.is_empty() }

	/// How many samples the whole bank is holding.
	///
	/// What a statistics panel asks. Counted rather than kept, because it is
	/// asked once a frame at most and kept numbers go stale.
	#[must_use]
	pub fn samples(&self) -> usize {
		self.recordings
			.iter()
			.flatten()
			.map(|recording| recording.data.samples.len())
			.sum()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A sound of `frames` mono frames, all holding `value`.
	fn tone(frames: usize, value: i16) -> SoundData {
		SoundData {
			samples: vec![value; frames],
			rate: 1000,
			channels: 1,
		}
	}

	#[test]
	fn a_new_bank_copies_everything_including_the_null_sound() {
		let mut sounds = Sounds::new();
		let id = sounds.insert("sounds/thud", tone(10, 7));
		let mut bank = Bank::new();

		assert_eq!(bank.sync(&sounds), 2, "the null entry and the one that was registered");
		assert_eq!(bank.len(), 2);
		assert!(!bank.is_empty());
		assert_eq!(bank.get(id).map(|data| data.samples.len()), Some(10));
		assert!(
			bank.get(SoundId::NONE)
				.is_some_and(SoundData::is_empty),
			"and slot zero is the silence the registry holds"
		);
	}

	#[test]
	fn syncing_twice_over_a_registry_that_did_not_move_copies_nothing() {
		let mut sounds = Sounds::new();
		sounds.insert("sounds/thud", tone(10, 7));
		let mut bank = Bank::new();

		assert_eq!(bank.sync(&sounds), 2, "the first pass copies");
		assert_eq!(bank.sync(&sounds), 0, "and the second has nothing to do");
	}

	#[test]
	fn a_recompiled_sound_is_copied_again_and_a_still_one_is_not() {
		// the whole mechanism: the registry rewrites the entry the handle
		// already points at and moves its revision, and that is the only
		// signal there is.
		let mut sounds = Sounds::new();
		let thud = sounds.insert("sounds/thud", tone(10, 7));
		sounds.insert("sounds/click", tone(4, 1));
		let mut bank = Bank::new();
		bank.sync(&sounds);

		sounds.insert("sounds/thud", tone(20, 9));

		assert_eq!(bank.sync(&sounds), 1, "one of the two moved");
		assert_eq!(
			bank.get(thud).map(|data| data.samples.len()),
			Some(20),
			"and the bank has the new samples under the same handle"
		);
	}

	#[test]
	fn a_sound_registered_since_the_last_sync_is_reached_only_after_the_next() {
		let mut sounds = Sounds::new();
		let mut bank = Bank::new();
		bank.sync(&sounds);

		let late = sounds.insert("sounds/late", tone(10, 7));

		assert!(bank.get(late).is_none(), "the mixer has not been told yet");
		bank.sync(&sounds);
		assert!(bank.get(late).is_some(), "and now it has");
	}

	#[test]
	fn a_handle_the_bank_has_never_heard_of_reaches_nothing() {
		let bank = Bank::new();

		assert!(bank.get(SoundId::new(9)).is_none(), "rather than panicking");
		assert!(bank.is_empty());
		assert_eq!(bank.samples(), 0);
	}

	#[test]
	fn the_bank_can_say_how_much_it_is_holding() {
		let mut sounds = Sounds::new();
		sounds.insert("sounds/one", tone(10, 1));
		sounds.insert("sounds/two", tone(25, 2));
		let mut bank = Bank::new();
		bank.sync(&sounds);

		assert_eq!(bank.samples(), 35, "both of them, and nothing for the empty slot");
	}
}
