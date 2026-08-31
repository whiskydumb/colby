//! What crosses from the simulation to whatever is filling a device buffer.
//!
//! One of these is built at the end of a step and read by the mixer. It carries
//! **finished numbers**: the gain each voice gets in each ear is already worked
//! out, so nothing on the far side has to know where the listener is, what the
//! distance model is, or how loud the music is meant to be. That is a decision
//! about where arithmetic happens rather than an optimization - the far side
//! runs on somebody else's thread at somebody else's cadence, and every value
//! computed there is a value that cannot be unit-tested.
//!
//! It carries the sound as a handle rather than as samples, because the samples
//! are in a [`Bank`](crate::bank::Bank) the mixer already owns.
//!
//! **A slot and a generation, not a handle.** The mixer keeps one playhead per
//! slot; the generation is how it notices that a slot is somebody else's sound
//! now and starts from the beginning instead of carrying on from wherever the
//! last one had got to.

use colby_core::abi::{Listener, Mix, SoundId, Voices, World};

use crate::pan;

/// One voice, as the mixer needs it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Playing {
	/// Which slot of the voice table this is.
	pub slot: u32,

	/// Which occupant of that slot. A change here means a new sound.
	pub generation: u32,

	/// What to play, as a handle into the bank.
	pub sound: SoundId,

	/// How far in the simulation thinks it is, in seconds.
	pub head: f32,

	/// How fast, as a multiple of the recording's own rate.
	pub speed: f32,

	/// What it is worth in the left ear.
	pub left: f32,

	/// And in the right.
	pub right: f32,

	/// Whether it starts over at the end.
	pub looping: bool,

	/// Whether the recording is mixed down to one channel before panning.
	///
	/// True for anything with a place in the world: a stereo recording played
	/// through a panner would be two sources at once, one of which is in the
	/// wrong ear. False for music and menu sounds, which are played as they
	/// were recorded.
	pub downmix: bool,
}

/// Every voice as of the end of one step.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
	/// One per voice still playing, in slot order.
	pub voices: Vec<Playing>,

	/// Which step this was taken at, for a log line and for telling two of
	/// them apart.
	pub step: u64,
}

impl Snapshot {
	/// An empty snapshot: nothing playing, no step taken.
	#[must_use]
	pub const fn new() -> Self { Self { voices: Vec::new(), step: 0 } }

	/// Fills this in from a world, reusing whatever it already allocated.
	///
	/// Taken once a step, on the simulation's thread. Reusing the vector is
	/// the whole reason this is a method rather than a constructor: the host
	/// keeps two of these and swaps them, so neither ever allocates after the
	/// first few steps.
	///
	/// @param world - the world as the step left it
	pub fn take(&mut self, world: &World) {
		self.fill(&world.audio, &world.listener, &world.mix);
		self.step = world.steps;
	}

	/// The same, from the three pieces rather than from a whole world.
	///
	/// @param voices - what is playing
	/// @param listener - where the world is heard from
	/// @param mix - how loud each category is
	pub fn fill(&mut self, voices: &Voices, listener: &Listener, mix: &Mix) {
		self.voices.clear();

		for (id, voice) in voices.iter() {
			let (left, right) = pan::gains(voice, listener, mix);

			self.voices.push(Playing {
				slot: id.slot(),
				generation: id.generation(),
				sound: voice.sound,
				head: voice.head,
				speed: voice.speed(),
				left,
				right,
				looping: voice.looping,
				downmix: voice.positioned,
			});
		}
	}

	/// How many voices it holds.
	#[must_use]
	pub fn len(&self) -> usize { self.voices.len() }

	/// Whether nothing is playing.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.voices.is_empty() }
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{Category, SoundData, Voice},
		glam::Vec3,
	};

	use super::*;

	/// A world with one sound registered and nothing playing.
	fn world() -> (World, SoundId) {
		let mut world = World::new();
		let id = world.sounds.insert("sounds/thud", SoundData {
			samples: vec![0; 1000],
			rate: 1000,
			channels: 1,
		});

		(world, id)
	}

	#[test]
	fn an_empty_world_makes_an_empty_snapshot() {
		let (world, _) = world();
		let mut snapshot = Snapshot::new();
		snapshot.take(&world);

		assert!(snapshot.is_empty(), "nothing is playing");
		assert_eq!(snapshot.len(), 0);
	}

	#[test]
	fn a_voice_crosses_with_its_slot_its_generation_and_its_gains() {
		let (mut world, sound) = world();
		// a slot that has been used before, so that the generation is not the
		// number one - which is what a fixture playing into a fresh table
		// gives, and which cannot tell a real generation from a constant.
		let first = world.audio.play(Voice::flat(sound));
		world.audio.stop(first);

		let id = world.audio.play(Voice::flat(sound).volume(0.5));

		assert!(id.generation() > 1, "the fixture is a reused slot: {}", id.generation());
		let mut snapshot = Snapshot::new();
		snapshot.take(&world);

		let playing = snapshot.voices[0];

		assert_eq!(playing.slot, id.slot(), "the slot the mixer keys its playhead on");
		assert_eq!(playing.generation, id.generation(), "and which occupant of it");
		assert_eq!(playing.sound, sound, "a handle, not samples");
		assert!((playing.left - 0.5).abs() < 1e-6, "with the volume already applied");
		assert!((playing.right - 0.5).abs() < 1e-6);
		assert!(!playing.downmix, "and nothing to downmix, having no place in the world");
	}

	#[test]
	fn the_listener_and_the_mix_are_baked_in_rather_than_carried() {
		// the point of the whole module: the far side gets numbers, not a
		// distance model.
		let (mut world, sound) = world();
		world.mix.effects = 0.5;
		world.listener.at = Vec3::ZERO;
		world.audio.play(
			Voice::at(sound, Vec3::X * 4.0)
				.volume(1.0)
				.range(2.0, 100.0)
				.category(Category::Effect),
		);

		let mut snapshot = Snapshot::new();
		snapshot.take(&world);

		let playing = snapshot.voices[0];

		assert!(playing.left.abs() < 1e-6, "hard right, so nothing on the left");
		assert!(
			(playing.right - 0.25).abs() < 1e-6,
			"half for the category, half again for the distance: {}",
			playing.right
		);
		assert!(
			playing.downmix,
			"and a thing standing somewhere is panned, so it is one channel"
		);
	}

	#[test]
	fn moving_the_listener_moves_what_the_next_snapshot_says() {
		let (mut world, sound) = world();
		world
			.audio
			.play(Voice::at(sound, Vec3::X * 4.0).range(10.0, 100.0));
		let mut snapshot = Snapshot::new();

		snapshot.take(&world);
		let before = snapshot.voices[0];

		world.listener.at = Vec3::X * 8.0;
		snapshot.take(&world);
		let after = snapshot.voices[0];

		assert!(before.right > before.left, "it was on the right");
		assert!(after.left > after.right, "and walking past it puts it on the left");
	}

	#[test]
	fn the_step_number_comes_across_so_two_snapshots_can_be_told_apart() {
		let (mut world, _) = world();
		let mut snapshot = Snapshot::new();

		world.steps = 41;
		snapshot.take(&world);

		assert_eq!(snapshot.step, 41);
	}

	#[test]
	fn taking_a_snapshot_again_replaces_what_was_in_it() {
		// it is reused rather than rebuilt, so a voice that stopped has to
		// actually leave.
		let (mut world, sound) = world();
		let id = world.audio.play(Voice::flat(sound));
		let mut snapshot = Snapshot::new();

		snapshot.take(&world);
		assert_eq!(snapshot.len(), 1);

		world.audio.stop(id);
		snapshot.take(&world);

		assert!(snapshot.is_empty(), "the stopped voice is not still in there");
	}

	#[test]
	fn the_speed_that_crosses_is_the_clamped_one() {
		let (mut world, sound) = world();
		world.audio.play(Voice::flat(sound).pitch(1000.0));
		let mut snapshot = Snapshot::new();
		snapshot.take(&world);

		assert!(
			(snapshot.voices[0].speed - colby_core::abi::audio::MAX_PITCH).abs() < f32::EPSILON,
			"a mixer stepping by a thousand would read past the end of everything"
		);
	}
}
