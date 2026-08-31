//! Sounds: the samples a noise is made of, and the host's registry of them.
//!
//! Here rather than in the mixer for the same reason meshes and textures are in
//! `colby_core` rather than in the renderer: [`SoundId`] crosses the boundary,
//! and a handle means nothing away from the table it indexes. @ref
//! [`registry`](super::registry) for the shape all five tables share.
//!
//! **Nothing in this module reads a `.wav`.** A sound arrives already decoded:
//! the asset compiler turns whatever a recorder wrote into interleaved
//! sixteen-bit samples, and what reaches here is the result. That is the same
//! bargain textures and fonts made - the decoder lands offline, and neither the
//! engine nor the runner links an audio library to open a file.
//!
//! **Sixteen bits, at the file's own rate.** Sixteen because that is what every
//! engine stores and because a ten-second stereo sound is 1.7 megabytes rather
//! than 3.5; the mixer widens to `f32` on the way past, which is one multiply
//! per sample. The file's own rate rather than one fixed rate because the
//! device's rate is not knowable when the file is compiled - and a mixer that
//! steps a fractional index has to exist anyway, since that is also what
//! playing something at another pitch is.
//!
//! **Everything is held whole in memory.** There is no streaming and no
//! compression, which is why [`MAX_SAMPLES`] is what it is: about three minutes
//! of stereo at forty-eight kilohertz. The thing that forces a codec is music,
//! and there is no music yet.

use super::registry::{Entry, Registry};
use crate::registry_handle;

/// The most samples one sound may hold, counting every channel.
///
/// About three minutes of stereo at forty-eight kilohertz, or thirty-two
/// megabytes. This is the number that says the engine does not stream: a sound
/// is decompressed once and then held, so the limit is a limit on memory rather
/// than on length as such.
pub const MAX_SAMPLES: usize = 1 << 24;

/// The most channels one sound may have.
///
/// Mono or stereo. A positioned sound is mixed down to one channel anyway, and
/// nothing here knows what to do with the fifth speaker of a file that has one.
pub const MAX_CHANNELS: u16 = 2;

/// The slowest sample rate that is taken seriously.
pub const MIN_RATE: u32 = 1000;

/// The fastest.
///
/// The same ceiling the one engine that resamples on import uses.
pub const MAX_RATE: u32 = 192_000;

/// The rate an empty sound reports, so that nothing divides by nothing.
pub const DEFAULT_RATE: u32 = 48_000;

registry_handle! {
	/// Which sound in [`Sounds`].
	///
	/// Not generational, like every other resource handle here: a sound
	/// resolved by name in `init` stays valid for the life of the process, and
	/// recompiling the source rewrites the entry the handle already points at.
	/// @ref [`registry`](super::registry).
	SoundId
}

/// One entry of the sound registry.
pub type Sound = Entry<SoundData>;

/// A decoded sound: interleaved samples and what they mean.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoundData {
	/// Every sample, interleaved, [`channels`](Self::channels) to a frame.
	///
	/// A stereo sound runs left, right, left, right. The length is always a
	/// whole number of frames, which [`check`](Self::check) is what enforces.
	pub samples: Vec<i16>,

	/// Frames a second, as the file was recorded.
	pub rate: u32,

	/// How many samples make one frame: one for mono, two for stereo.
	pub channels: u16,
}

impl SoundData {
	/// A sound with nothing in it.
	///
	/// What slot zero holds, and what a name nobody compiled resolves to. It
	/// plays as silence and ends immediately rather than being a handle that
	/// resolves to nothing, which is the same discipline `MeshId::NONE` and
	/// the empty font follow.
	#[must_use]
	pub const fn silence() -> Self {
		Self {
			samples: Vec::new(),
			rate: DEFAULT_RATE,
			channels: 1,
		}
	}

	/// How many frames there are, whatever the channel count is.
	#[must_use]
	pub fn frames(&self) -> usize {
		if self.channels == 0 {
			return 0;
		}

		self.samples.len() / usize::from(self.channels)
	}

	/// How long it lasts, in seconds.
	///
	/// The number a voice's playhead is measured against, so it is worked out
	/// the same way everywhere rather than at each call site.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		clippy::cast_precision_loss,
		reason = "a frame count past what an f32 counts exactly is past MAX_SAMPLES, which is \
		          refused, and a rate is at most MAX_RATE"
	)]
	pub fn seconds(&self) -> f32 {
		if self.rate == 0 {
			return 0.0;
		}

		self.frames() as f32 / self.rate as f32
	}

	/// Whether there is anything to play.
	#[must_use]
	pub const fn is_empty(&self) -> bool { self.samples.is_empty() }

	/// Everything that has to hold before this is a sound rather than numbers.
	///
	/// One definition, called by whatever builds a [`SoundData`] - the
	/// importer on the way in and the format writer on the way out - so that
	/// the two cannot come to different conclusions about the same file.
	///
	/// @return nothing, or why these are not samples anybody can play
	///
	/// # Errors
	///
	/// If the channel count, the rate or the length is one nothing downstream
	/// can work with.
	pub fn check(&self) -> Result<(), String> {
		if self.channels == 0 || self.channels > MAX_CHANNELS {
			return Err(format!(
				"has {} channels, and a sound here is mono or stereo",
				self.channels
			));
		}

		if self.rate < MIN_RATE || self.rate > MAX_RATE {
			return Err(format!(
				"is recorded at {} frames a second, outside the {MIN_RATE} to {MAX_RATE} a \
				 sound may be",
				self.rate
			));
		}

		if self.samples.len() > MAX_SAMPLES {
			return Err(format!(
				"holds {} samples, past the {MAX_SAMPLES} one sound may have; nothing here \
				 streams, so a sound is held whole",
				self.samples.len()
			));
		}

		if !self
			.samples
			.len()
			.is_multiple_of(usize::from(self.channels))
		{
			return Err(format!(
				"holds {} samples, which is not a whole number of {}-channel frames",
				self.samples.len(),
				self.channels
			));
		}

		Ok(())
	}
}

impl Default for SoundData {
	fn default() -> Self { Self::silence() }
}

/// Every sound the mixer can play, addressed by [`SoundId`].
///
/// Slot zero is [`SoundId::NONE`] and is [`SoundData::silence`], so a game
/// naming a sound nobody compiled plays nothing rather than reaching for an
/// entry that is not there.
#[derive(Clone, Debug)]
pub struct Sounds {
	entries: Registry<SoundData>,
}

impl Sounds {
	/// A registry holding nothing but silence.
	#[must_use]
	pub fn new() -> Self {
		Self {
			entries: Registry::new(SoundData::silence()),
		}
	}

	/// Looks a sound up by name.
	///
	/// @param name - what the compiler registered it as, e.g. `sounds/thud`
	/// @return its handle, or [`SoundId::NONE`] if nothing answers to it
	#[must_use]
	pub fn find(&self, name: &str) -> SoundId { SoundId::new(self.entries.find(name)) }

	/// Registers a decoded sound under a name, replacing whatever was there.
	///
	/// @return the handle, the same one as last time if the name is known
	pub fn insert(&mut self, name: &str, data: SoundData) -> SoundId {
		SoundId::new(self.entries.insert(name, data))
	}

	/// One sound, by handle.
	#[must_use]
	pub fn get(&self, id: SoundId) -> Option<&Sound> { self.entries.entry(id.index()) }

	/// One sound's samples, by handle, falling back to silence.
	///
	/// What the mixer calls: it has nothing useful to do about a handle from
	/// another table, and would rather play nothing than branch.
	///
	/// @note: no second look at slot zero, unlike the font table's version of
	/// this. Slot zero is unreachable - `insert` never returns it, because the
	/// empty name is not a name - so it always holds what it was built with,
	/// and a fall back to it and a fall back to [`SILENCE`] are the same
	/// answer. The line was written and taken out again because no mutation of
	/// it could fail a test.
	#[must_use]
	pub fn data(&self, id: SoundId) -> &SoundData {
		self.entries
			.entry(id.index())
			.map_or(&SILENCE, Entry::value)
	}

	/// How many sounds there are, counting the null one.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Always `false`: slot zero always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// Every sound, in slot order, starting with the null one.
	pub fn iter(&self) -> impl Iterator<Item = &Sound> { self.entries.iter() }
}

impl Default for Sounds {
	fn default() -> Self { Self::new() }
}

/// What [`Sounds::data`] hands back when even slot zero is missing.
///
/// It cannot be, but the alternative is an `unwrap` in the one call the mixer
/// makes per voice per block.
static SILENCE: SoundData = SoundData {
	samples: Vec::new(),
	rate: DEFAULT_RATE,
	channels: 1,
};

#[cfg(test)]
mod tests {
	use super::*;

	/// A second of stereo at a rate that divides evenly.
	fn beep() -> SoundData {
		SoundData {
			samples: vec![0; 2000],
			rate: 1000,
			channels: 2,
		}
	}

	#[test]
	fn a_sound_reports_its_length_from_its_samples_and_its_rate() {
		let sound = beep();

		assert_eq!(sound.frames(), 1000, "two samples to a frame");
		assert!(
			(sound.seconds() - 1.0).abs() < 1e-6,
			"a thousand frames at a thousand a second is a second"
		);
		assert!(!sound.is_empty(), "and there is something in it");
	}

	#[test]
	fn silence_is_empty_without_being_a_sound_of_no_rate() {
		let quiet = SoundData::silence();

		assert!(quiet.is_empty(), "there is nothing to play");
		assert_eq!(quiet.frames(), 0, "not one frame");
		assert!(
			(quiet.seconds() - 0.0).abs() < f32::EPSILON,
			"and it lasts no time at all rather than dividing by nothing"
		);
		assert_eq!(quiet.rate, DEFAULT_RATE, "the rate is still a number the mixer can use");
	}

	#[test]
	fn a_well_formed_sound_passes_the_check_and_the_broken_ones_say_why() {
		beep()
			.check()
			.expect("a second of stereo is a sound");

		let cases = [
			(SoundData { channels: 0, ..beep() }, "mono or stereo"),
			(SoundData { channels: 6, ..beep() }, "mono or stereo"),
			(SoundData { rate: 0, ..beep() }, "frames a second"),
			(SoundData { rate: MAX_RATE + 1, ..beep() }, "frames a second"),
			(SoundData { samples: vec![0; 999], ..beep() }, "whole number"),
			(
				SoundData {
					samples: vec![0; MAX_SAMPLES + 2],
					..beep()
				},
				"streams",
			),
		];

		for (sound, expected) in cases {
			let message = sound
				.check()
				.expect_err("this one is not a sound anybody can play");

			assert!(
				message.contains(expected),
				"the message names what is wrong: wanted {expected:?}, got {message:?}"
			);
		}
	}

	#[test]
	fn a_sound_too_long_to_hold_is_measured_before_it_is_refused() {
		// the length check has to come before the frame check, or a file that
		// is both too long and ragged reports the cheaper complaint and
		// somebody fixes the wrong thing.
		let ragged = SoundData {
			samples: vec![0; MAX_SAMPLES + 1],
			..beep()
		};

		assert!(
			ragged
				.check()
				.expect_err("it is too long")
				.contains("streams"),
			"length is the complaint, not the odd sample"
		);
	}

	#[test]
	fn slot_zero_is_silence_and_answers_to_nothing() {
		let sounds = Sounds::new();

		assert_eq!(sounds.len(), 1, "a new table holds only its null entry");
		assert!(!sounds.is_empty(), "which is why it is never empty");
		assert_eq!(sounds.find("sounds/thud"), SoundId::NONE, "nothing answers to that yet");
		assert!(sounds.data(SoundId::NONE).is_empty(), "and the null sound plays nothing");
	}

	#[test]
	fn a_name_takes_a_slot_and_keeps_it_across_a_recompile() {
		let mut sounds = Sounds::new();
		let first = sounds.insert("sounds/thud", beep());
		let again = sounds.insert("sounds/thud", SoundData { rate: 2000, ..beep() });

		assert_eq!(first, again, "the handle a game kept still points at the same slot");
		assert_eq!(sounds.data(first).rate, 2000, "and now at the new samples");
		assert_eq!(
			sounds.get(first).map(Entry::revision),
			Some(1),
			"which is what tells the mixer to start the voice over"
		);
	}

	#[test]
	fn a_handle_from_another_table_falls_back_to_silence_rather_than_nothing() {
		let sounds = Sounds::new();

		assert!(sounds.get(SoundId::new(9)).is_none(), "there is no slot nine");
		assert!(sounds.data(SoundId::new(9)).is_empty(), "and asking for it plays nothing");
	}
}
