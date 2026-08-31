//! Turning a snapshot and a bank into a buffer of samples.
//!
//! The one thing in this crate that has state, and the state is small: a
//! playhead and a pair of gains per slot. Everything else it needs arrives as
//! arguments, which is what makes this testable without a device - the real
//! output path is a caller that hands it the buffer a driver asked for, and
//! `just hear` is a caller that hands it a buffer it is going to write to a
//! file.
//!
//! **Two clocks, and knowing which is which is the whole of it.** The
//! simulation's playhead moves by `dt` a step and decides when a voice *ends*.
//! The one in here moves by one output frame at a time and decides which sample
//! is heard. They agree to within a block, and where they disagree by more than
//! that block plus [`SLEW_SECONDS`] this one jumps - which is what makes a game
//! writing `Voice::head` a seek rather than a suggestion, and what stops a long
//! run from drifting.
//!
//! **Stereo out, and nothing else.** A device that wants another channel count
//! is the caller's problem, not this one's.

use colby_core::abi::SoundData;

use crate::{bank::Bank, snapshot::Snapshot};

/// How many samples make one frame of output.
pub const CHANNELS: usize = 2;

/// How far this playhead may drift from the simulation's, over and above the
/// block being rendered, before it jumps.
///
/// Three steps *plus* however long the block itself is, which is the part that
/// is easy to leave out: a device asking for a hundred milliseconds at a time
/// runs a hundred milliseconds ahead of a snapshot taken before it, and a fixed
/// threshold below that would correct a playhead every single block. Above the
/// sum, something happened that is not drift: a game seeking, or a process that
/// stalled long enough for the simulation's clock - which is capped - to fall
/// behind the wall clock.
pub const SLEW_SECONDS: f32 = 0.05;

/// What one sixteen-bit sample is worth.
///
/// Divided rather than multiplied by its reciprocal so that the quietest
/// possible sample is exactly negative one.
const SCALE: f32 = 32768.0;

/// The most slots a mixer keeps a playhead for.
const SLOTS: usize = colby_core::abi::MAX_VOICES;

/// Where one slot has got to, and how loud it was at the end of the last
/// block.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Head {
	/// Which occupant of the slot this is about.
	generation: u32,

	/// How far into the recording, in its own frames.
	///
	/// `f64` rather than `f32`, and the reason is a measurement rather than
	/// caution: an `f32` runs out of whole numbers at about sixteen million,
	/// which at forty-eight kilohertz is five and a half minutes. A loop left
	/// running would stop advancing.
	frame: f64,

	/// What the left ear was getting.
	left: f32,

	/// And the right.
	right: f32,
}

/// The mixer: one playhead per slot, and a way to fill a buffer.
#[derive(Clone, Debug)]
pub struct Mixer {
	rate: u32,
	/// Which snapshot was last rendered from, or nothing before the first.
	///
	/// The drift correction below runs **only when this moves**, and that is
	/// not an optimization. Rendering several blocks against one snapshot is
	/// the normal case - a device asking every ten milliseconds against a step
	/// every sixteen - and each of those blocks legitimately runs further
	/// ahead of a playhead that has not been updated since. Correcting against
	/// a snapshot that is standing still drags the sound back to where it was
	/// a moment ago, over and over, and what comes out is the first fiftieth
	/// of a second on a loop. It was found by playing a real recording rather
	/// than a ramp, which is what the envelope test beside this crate exists
	/// for.
	last: Option<u64>,
	heads: Vec<Head>,
}

impl Mixer {
	/// A mixer that will be asked for samples at this rate.
	///
	/// @param rate - frames a second, as the device asked for them. Zero is
	/// taken as one, because the alternative is dividing by it.
	#[must_use]
	pub fn new(rate: u32) -> Self {
		Self {
			rate: rate.max(1),
			last: None,
			heads: vec![Head::default(); SLOTS],
		}
	}

	/// The rate it was built for.
	#[must_use]
	pub const fn rate(&self) -> u32 { self.rate }

	/// Fills a buffer with everything the snapshot says is playing.
	///
	/// Interleaved stereo, so `out` is twice as many samples as frames. What
	/// is already in it is overwritten rather than added to.
	///
	/// @param bank - the mixer's own copy of the samples
	/// @param snapshot - what was playing at the end of the last step
	/// @param out - the buffer to fill; a length that is not a whole number of
	/// frames has its last sample left alone
	pub fn render(&mut self, bank: &Bank, snapshot: &Snapshot, out: &mut [f32]) {
		out.fill(0.0);

		let frames = out.len() / CHANNELS;
		if frames == 0 {
			return;
		}

		let rate = f64::from(self.rate);
		let moved = self.last != Some(snapshot.step);
		self.last = Some(snapshot.step);

		for playing in &snapshot.voices {
			let Some(data) = bank.get(playing.sound) else {
				continue;
			};

			let Ok(slot) = usize::try_from(playing.slot) else {
				continue;
			};

			let Some(head) = self.heads.get_mut(slot) else {
				continue;
			};

			mix_one(head, data, playing, Block { rate, frames, moved }, out);
		}

		// the sum of sixty-four voices is not bounded by one, and a driver
		// handed a sample outside the range wraps it into a very loud noise.
		// Clamping is what every mixer without a limiter in it does, and a
		// limiter is a subject rather than a line.
		for sample in out.iter_mut() {
			*sample = sample.clamp(-1.0, 1.0);
		}
	}

	/// Forgets where everything had got to.
	///
	/// For a device that was reopened: the slots mean the same thing but the
	/// samples between then and now were never played, so carrying a playhead
	/// across would resume a sound in the middle of a silence nobody heard.
	pub fn forget(&mut self) {
		self.last = None;
		self.heads.clear();
		self.heads.resize(SLOTS, Head::default());
	}
}

/// Adds one voice into the buffer, and leaves its playhead where it got to.
#[expect(
	clippy::as_conversions,
	clippy::cast_precision_loss,
	reason = "a frame count comes from a sample count capped at MAX_SAMPLES, which an f64 \
	          counts exactly, and a block length is a few thousand"
)]
fn mix_one(
	head: &mut Head,
	data: &SoundData,
	playing: &crate::snapshot::Playing,
	about: Block,
	out: &mut [f32],
) {
	let Block { rate, frames, moved } = about;
	let block = frames as f64 / rate;
	let source_frames = data.frames();
	if source_frames == 0 || data.rate == 0 {
		return;
	}

	let end = source_frames as f64;
	let theirs = f64::from(playing.head) * f64::from(data.rate);

	if head.generation == playing.generation {
		// the two clocks are allowed to disagree by a block. Past that it is
		// not drift, and following is better than being right about a sound
		// nobody asked to still be at that point. Only against a snapshot that
		// has actually moved, though - @ref `Mixer::last`.
		let slack = (f64::from(SLEW_SECONDS) + block) * f64::from(data.rate);

		if moved && apart(head.frame, theirs, end, playing.looping) > slack {
			head.frame = theirs;
		}
	} else {
		// somebody else's sound is in this slot now. Start from where the
		// simulation says, and take the gains without a ramp: ramping from
		// whatever the previous occupant was worth is a fade from a stranger.
		head.generation = playing.generation;
		head.frame = theirs;
		head.left = playing.left;
		head.right = playing.right;
	}

	let step = f64::from(data.rate) / rate * f64::from(playing.speed);
	let span = frames as f32;
	let (mut left, mut right) = (head.left, head.right);
	let toward_left = (playing.left - left) / span;
	let toward_right = (playing.right - right) / span;

	for frame in 0..frames {
		if head.frame >= end {
			if !playing.looping {
				break;
			}

			head.frame = head.frame.rem_euclid(end);
		}

		let (mut sample_left, mut sample_right) = sample(data, head.frame, playing.looping);
		if playing.downmix {
			let both = (sample_left + sample_right) * 0.5;
			sample_left = both;
			sample_right = both;
		}

		if let Some(slot) = out.get_mut(frame * CHANNELS) {
			*slot = sample_left.mul_add(left, *slot);
		}

		if let Some(slot) = out.get_mut(frame * CHANNELS + 1) {
			*slot = sample_right.mul_add(right, *slot);
		}

		left += toward_left;
		right += toward_right;
		head.frame += step;
	}

	// whatever happened above, the ramp was towards these and the next block
	// starts from them. Setting them from `left` and `right` instead would
	// leave a voice that ran out of samples half-faded forever.
	head.left = playing.left;
	head.right = playing.right;
}

/// What one call to [`Mixer::render`] is about, as far as one voice cares.
///
/// A struct rather than three more arguments, which is what the house rules
/// ask for once a signature reaches seven and what both readers of this one
/// wanted anyway: the three of them are one fact about the block.
#[derive(Clone, Copy, Debug)]
struct Block {
	/// Frames a second, as the device asked for them.
	rate: f64,

	/// How many frames this call is for.
	frames: usize,

	/// Whether the snapshot is a different one from the last call's.
	moved: bool,
}

/// How far apart two playheads are, going the short way round a loop.
///
/// Without the short way, a looping voice whose two clocks wrapped on either
/// side of the same block would look a whole recording apart and be corrected
/// every time - which is a click at every loop point.
///
/// @param mine - the mixer's playhead, in source frames
/// @param theirs - the simulation's, in the same
/// @param length - how many frames the recording is
/// @param looping - whether the ends are joined
/// @return the distance, never negative
#[must_use]
fn apart(mine: f64, theirs: f64, length: f64, looping: bool) -> f64 {
	let straight = (mine - theirs).abs();
	if !looping || length <= 0.0 {
		return straight;
	}

	let wrapped = straight.rem_euclid(length);

	wrapped.min(length - wrapped)
}

/// One frame of a recording, at a position between two of them.
///
/// Linear interpolation, which is what every engine that resamples by stepping
/// an index does. It is not the best resampler there is; it is the one whose
/// cost is two multiplies and whose error nobody can hear on a thud.
///
/// @param data - the recording
/// @param at - where to read, in frames, fractional
/// @param looping - whether the frame after the last one is the first
/// @return `(left, right)`, both between -1 and 1, with a mono recording
/// answering the same in both
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "a float to integer cast saturates, and the result is bounded by the frame count \
	          on the line after it"
)]
fn sample(data: &SoundData, at: f64, looping: bool) -> (f32, f32) {
	let frames = data.frames();
	if frames == 0 {
		return (0.0, 0.0);
	}

	let base = at.max(0.0);
	let whole = base.floor();
	let fraction = (base - whole) as f32;
	let first = (whole as usize).min(frames - 1);
	let second = if first + 1 < frames {
		first + 1
	} else if looping {
		0
	} else {
		first
	};

	let (left, right) = frame_of(data, first);
	let (next_left, next_right) = frame_of(data, second);

	(
		fraction.mul_add(next_left - left, left),
		fraction.mul_add(next_right - right, right),
	)
}

/// One whole frame, as two floats.
fn frame_of(data: &SoundData, frame: usize) -> (f32, f32) {
	let channels = usize::from(data.channels.max(1));
	let at = frame * channels;
	let left = data.samples.get(at).copied().unwrap_or(0);
	let right = if channels > 1 {
		data.samples.get(at + 1).copied().unwrap_or(left)
	} else {
		left
	};

	(f32::from(left) / SCALE, f32::from(right) / SCALE)
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{Listener, Mix, SoundId, Sounds, Voice, Voices},
		glam::Vec3,
	};

	use super::*;
	use crate::snapshot::Playing;

	/// A bank holding one mono recording, under handle one.
	fn banked(samples: Vec<i16>, rate: u32, channels: u16) -> (Bank, SoundId) {
		let mut sounds = Sounds::new();
		let id = sounds.insert("sounds/test", SoundData { samples, rate, channels });
		let mut bank = Bank::new();
		bank.sync(&sounds);

		(bank, id)
	}

	/// A snapshot of exactly one voice, at slot zero generation one.
	fn one(playing: Playing) -> Snapshot { Snapshot { voices: vec![playing], step: 1 } }

	/// A voice playing this sound at full volume in both ears.
	fn plain(sound: SoundId) -> Playing {
		Playing {
			slot: 0,
			generation: 1,
			sound,
			head: 0.0,
			speed: 1.0,
			left: 1.0,
			right: 1.0,
			looping: false,
			downmix: false,
		}
	}

	/// The left channel of a buffer.
	fn lefts(out: &[f32]) -> Vec<f32> { out.iter().step_by(CHANNELS).copied().collect() }

	#[test]
	fn a_recording_played_at_the_device_rate_comes_out_sample_for_sample() {
		// the case nothing is allowed to be clever about: the same rate, no
		// pitch, so every output frame is one input frame and the arithmetic
		// has to be the identity.
		let (bank, sound) = banked(vec![0, 8192, 16384, -8192], 1000, 1);
		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.0; 4 * CHANNELS];

		mixer.render(&bank, &one(plain(sound)), &mut out);

		let heard = lefts(&out);
		let wanted = [0.0, 0.25, 0.5, -0.25];

		for (index, (got, want)) in heard.iter().zip(wanted).enumerate() {
			assert!((got - want).abs() < 1e-4, "frame {index}: {got} against {want}");
		}
	}

	#[test]
	fn a_recording_slower_than_the_device_is_stretched_and_interpolated() {
		let (bank, sound) = banked(vec![0, 16384], 1000, 1);
		let mut mixer = Mixer::new(2000);
		let mut out = vec![0.0; 4 * CHANNELS];

		mixer.render(&bank, &one(plain(sound)), &mut out);

		let heard = lefts(&out);

		assert!(heard[0].abs() < 1e-4, "the first frame is the first sample");
		assert!(
			(heard[1] - 0.25).abs() < 1e-4,
			"and the second is halfway to the next: {}",
			heard[1]
		);
		assert!((heard[2] - 0.5).abs() < 1e-4, "then the second sample itself");
		// half a frame past the last one there is nothing to interpolate
		// towards, so the last sample is held rather than faded into a zero
		// nobody recorded. It matters for exactly one output frame and only
		// when the rates differ.
		assert!((heard[3] - 0.5).abs() < 1e-4, "and the last sample is held: {}", heard[3]);
	}

	#[test]
	fn a_faster_pitch_reads_further_into_the_recording_each_frame() {
		let (bank, sound) = banked(vec![0, 8192, 16384, 24576], 1000, 1);
		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.0; 2 * CHANNELS];

		mixer.render(&bank, &one(Playing { speed: 2.0, ..plain(sound) }), &mut out);

		let heard = lefts(&out);

		assert!(heard[0].abs() < 1e-4, "the first");
		assert!(
			(heard[1] - 0.5).abs() < 1e-4,
			"and then the third, not the second: {}",
			heard[1]
		);
	}

	#[test]
	fn the_two_ears_get_the_gains_they_were_given() {
		let (bank, sound) = banked(vec![16384; 4], 1000, 1);
		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.0; 4 * CHANNELS];

		mixer.render(&bank, &one(Playing { left: 1.0, right: 0.0, ..plain(sound) }), &mut out);

		assert!(out[0] > 0.4, "the left ear has it");
		assert!(out[1].abs() < 1e-6, "and the right has none of it");
	}

	#[test]
	fn a_stereo_recording_keeps_its_channels_when_it_is_not_positioned() {
		let (bank, sound) = banked(vec![16384, -16384, 16384, -16384], 1000, 2);
		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.0; 2 * CHANNELS];

		mixer.render(&bank, &one(plain(sound)), &mut out);

		assert!((out[0] - 0.5).abs() < 1e-4, "left is left");
		assert!((out[1] + 0.5).abs() < 1e-4, "and right is the other one");
	}

	#[test]
	fn a_positioned_stereo_recording_is_mixed_down_before_it_is_panned() {
		// two channels through a panner would be two sources, one of them in
		// the wrong ear. Averaging first is what makes a stereo file usable as
		// a thud without anybody having to re-export it.
		let (bank, sound) = banked(vec![16384, -16384], 1000, 2);
		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.0; CHANNELS];

		mixer.render(&bank, &one(Playing { downmix: true, ..plain(sound) }), &mut out);

		assert!(out[0].abs() < 1e-4, "the two cancel: {}", out[0]);
		assert!(out[1].abs() < 1e-4);
	}

	#[test]
	fn a_recording_that_runs_out_leaves_the_rest_of_the_buffer_alone() {
		let (bank, sound) = banked(vec![16384, 16384], 1000, 1);
		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.0; 6 * CHANNELS];

		mixer.render(&bank, &one(plain(sound)), &mut out);

		let heard = lefts(&out);

		assert!(heard[0] > 0.4 && heard[1] > 0.4, "the two samples there were");
		for (index, sample) in heard.iter().enumerate().skip(2) {
			assert!(sample.abs() < 1e-6, "frame {index} should be silence, got {sample}");
		}
	}

	#[test]
	fn a_looping_recording_comes_round_again_rather_than_stopping() {
		let (bank, sound) = banked(vec![16384, -16384], 1000, 1);
		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.0; 6 * CHANNELS];

		mixer.render(&bank, &one(Playing { looping: true, ..plain(sound) }), &mut out);

		let heard = lefts(&out);

		for (index, sample) in heard.iter().enumerate() {
			let wanted = if index % 2 == 0 { 0.5 } else { -0.5 };

			assert!((sample - wanted).abs() < 1e-4, "frame {index}: {sample} against {wanted}");
		}
	}

	#[test]
	fn a_looping_recording_interpolates_from_its_last_frame_into_its_first() {
		// only visible when the rates differ: at the same rate the playhead
		// lands on whole frames and there is nothing to interpolate, so a
		// fixture that does not resample cannot tell wrapping from holding.
		// Three output frames to each input one, so the two frames on either
		// side of the join are read a third and two thirds of the way across.
		let (bank, sound) = banked(vec![16384, -16384], 1000, 1);
		let mut mixer = Mixer::new(3000);
		let mut out = vec![0.0; 6 * CHANNELS];

		mixer.render(&bank, &one(Playing { looping: true, ..plain(sound) }), &mut out);

		let heard = lefts(&out);
		let wanted = [0.5, 1.0 / 6.0, -1.0 / 6.0, -0.5, -1.0 / 6.0, 1.0 / 6.0];

		for (index, (got, want)) in heard.iter().zip(wanted).enumerate() {
			assert!(
				(got - want).abs() < 1e-3,
				"frame {index}: {got} against {want}; the last two are the join, and 				 \
				 holding the final sample instead of reading into the first gives -0.5"
			);
		}
	}

	#[test]
	fn a_playhead_carries_on_from_block_to_block() {
		// the reason the mixer has state at all. Without it every block would
		// start the sound again and a recording longer than one buffer would
		// be its first few milliseconds, over and over.
		let (bank, sound) = banked(vec![0, 8192, 16384, 24576], 1000, 1);
		let mut mixer = Mixer::new(1000);
		let snapshot = one(plain(sound));

		let mut first = vec![0.0; 2 * CHANNELS];
		mixer.render(&bank, &snapshot, &mut first);

		let mut second = vec![0.0; 2 * CHANNELS];
		mixer.render(&bank, &snapshot, &mut second);

		let heard = lefts(&second);

		assert!((heard[0] - 0.5).abs() < 1e-4, "the third sample, not the first: {}", heard[0]);
		assert!((heard[1] - 0.75).abs() < 1e-4, "and then the fourth");
	}

	#[test]
	fn a_new_occupant_of_a_slot_starts_from_the_beginning() {
		// what the generation is for. Without the check, the second sound
		// would carry on from wherever the first one had got to.
		let (bank, sound) = banked(vec![0, 8192, 16384, 24576], 1000, 1);
		let mut mixer = Mixer::new(1000);

		let mut out = vec![0.0; 2 * CHANNELS];
		mixer.render(&bank, &one(plain(sound)), &mut out);

		let mut again = vec![0.0; 2 * CHANNELS];
		mixer.render(&bank, &one(Playing { generation: 2, ..plain(sound) }), &mut again);

		let heard = lefts(&again);

		assert!(heard[0].abs() < 1e-4, "the first sample again: {}", heard[0]);
		assert!((heard[1] - 0.25).abs() < 1e-4, "and the second");
	}

	#[test]
	fn a_gain_that_changes_between_blocks_is_ramped_rather_than_stepped() {
		// a gain applied as a step is a click, and a game moving a looping
		// sound changes its gains every step. The ramp is what makes that
		// inaudible.
		let (bank, sound) = banked(vec![16384; 64], 1000, 1);
		let mut mixer = Mixer::new(1000);

		let mut first = vec![0.0; 4 * CHANNELS];
		mixer.render(&bank, &one(plain(sound)), &mut first);

		let mut second = vec![0.0; 4 * CHANNELS];
		mixer.render(&bank, &one(Playing { left: 0.0, right: 0.0, ..plain(sound) }), &mut second);

		let heard = lefts(&second);

		assert!((heard[0] - 0.5).abs() < 1e-4, "it starts where it left off: {}", heard[0]);
		assert!(heard[0] > heard[1], "and comes down");
		assert!(heard[1] > heard[2]);
		assert!(heard[2] > heard[3]);
		assert!(heard[3].abs() < 0.2, "reaching about nothing by the end: {}", heard[3]);

		// and a third block, which is what says the ramp *ended* where it was
		// told rather than starting over from where it began. Without the two
		// lines that write the target back, every block would fade the same
		// way from the same place and the voice would never actually go quiet.
		let mut third = vec![0.0; 4 * CHANNELS];
		mixer.render(&bank, &one(Playing { left: 0.0, right: 0.0, ..plain(sound) }), &mut third);

		for (index, sample) in lefts(&third).iter().enumerate() {
			assert!(sample.abs() < 1e-6, "frame {index} should be silent, got {sample}");
		}
	}

	#[test]
	fn a_new_voice_takes_its_gains_at_once_rather_than_fading_up_from_a_strangers() {
		let (bank, sound) = banked(vec![16384; 64], 1000, 1);
		let mut mixer = Mixer::new(1000);

		let mut first = vec![0.0; 4 * CHANNELS];
		mixer.render(&bank, &one(plain(sound)), &mut first);

		let mut second = vec![0.0; 4 * CHANNELS];
		mixer.render(
			&bank,
			&one(Playing {
				generation: 2,
				left: 0.25,
				right: 0.25,
				..plain(sound)
			}),
			&mut second,
		);

		assert!(
			(lefts(&second)[0] - 0.125).abs() < 1e-4,
			"at its own volume from the first frame: {}",
			lefts(&second)[0]
		);
	}

	#[test]
	fn several_voices_are_summed() {
		let (bank, sound) = banked(vec![8192; 8], 1000, 1);
		let mut mixer = Mixer::new(1000);
		let snapshot = Snapshot {
			voices: vec![plain(sound), Playing { slot: 1, ..plain(sound) }],
			step: 1,
		};
		let mut out = vec![0.0; 2 * CHANNELS];

		mixer.render(&bank, &snapshot, &mut out);

		assert!((out[0] - 0.5).abs() < 1e-4, "two quarters: {}", out[0]);
	}

	#[test]
	fn a_sum_past_the_range_is_clamped_rather_than_wrapped() {
		// a sample outside the range handed to a driver is not a loud sound,
		// it is a different sound: the bits wrap and what comes out is noise.
		let (bank, sound) = banked(vec![24576; 8], 1000, 1);
		let mut mixer = Mixer::new(1000);
		let mut voices = Vec::new();
		for slot in 0..8 {
			voices.push(Playing { slot, ..plain(sound) });
		}

		let mut out = vec![0.0; 2 * CHANNELS];
		mixer.render(&bank, &Snapshot { voices, step: 1 }, &mut out);

		for sample in &out {
			assert!((-1.0..=1.0).contains(sample), "{sample} is outside what a device takes");
		}

		assert!((out[0] - 1.0).abs() < 1e-6, "and it is at the top rather than somewhere else");
	}

	#[test]
	fn a_buffer_is_overwritten_rather_than_added_to() {
		let (bank, sound) = banked(vec![0; 8], 1000, 1);
		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.5; 4 * CHANNELS];

		mixer.render(&bank, &one(plain(sound)), &mut out);

		for sample in &out {
			assert!(sample.abs() < 1e-6, "whatever a driver left in it is not music");
		}
	}

	#[test]
	fn a_voice_plays_the_sound_it_names_and_not_another_one_in_the_bank() {
		let mut sounds = Sounds::new();
		let quiet = sounds.insert("sounds/quiet", SoundData {
			samples: vec![4096; 8],
			rate: 1000,
			channels: 1,
		});
		let loud = sounds.insert("sounds/loud", SoundData {
			samples: vec![24576; 8],
			rate: 1000,
			channels: 1,
		});
		let mut bank = Bank::new();
		bank.sync(&sounds);

		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.0; 2 * CHANNELS];
		mixer.render(&bank, &one(plain(quiet)), &mut out);

		assert!((out[0] - 0.125).abs() < 1e-4, "the quiet one: {}", out[0]);

		mixer.forget();
		mixer.render(&bank, &one(plain(loud)), &mut out);

		assert!((out[0] - 0.75).abs() < 1e-4, "and the loud one: {}", out[0]);
	}

	#[test]
	fn a_voice_naming_a_sound_the_bank_has_never_heard_of_is_skipped() {
		let (bank, _) = banked(vec![16384; 4], 1000, 1);
		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.0; 2 * CHANNELS];

		mixer.render(&bank, &one(plain(SoundId::new(99))), &mut out);

		for sample in &out {
			assert!(sample.abs() < 1e-6, "silence, not a panic and not somebody else's samples");
		}
	}

	#[test]
	fn an_empty_buffer_and_an_empty_snapshot_are_both_fine() {
		let (bank, sound) = banked(vec![16384; 4], 1000, 1);
		let mut mixer = Mixer::new(1000);

		mixer.render(&bank, &one(plain(sound)), &mut []);
		let mut out = vec![0.0; 2 * CHANNELS];
		mixer.render(&bank, &Snapshot::new(), &mut out);

		for sample in &out {
			assert!(sample.abs() < 1e-6);
		}
	}

	#[test]
	fn a_playhead_that_has_drifted_far_from_the_simulation_jumps_to_it() {
		// which is what makes writing `Voice::head` a seek. A game that sets
		// it forward wants the sound to be there, not to get there eventually.
		let (bank, sound) = banked(vec![0; 4000], 1000, 1);
		let mut mixer = Mixer::new(1000);

		let mut out = vec![0.0; 100 * CHANNELS];
		mixer.render(&bank, &one(plain(sound)), &mut out);
		assert!((mixer.heads[0].frame - 100.0).abs() < 1e-6, "a hundred frames in");

		mixer.render(
			&bank,
			&Snapshot {
				voices: vec![Playing { head: 3.0, ..plain(sound) }],
				// a different step, because a snapshot that has not moved is
				// not evidence about anything. @ref `Mixer::last`.
				step: 2,
			},
			&mut out,
		);
		assert!(
			mixer.heads[0].frame > 3000.0,
			"and now three seconds in, plus the block: {}",
			mixer.heads[0].frame
		);
	}

	#[test]
	fn a_playhead_within_a_block_of_the_simulation_is_left_alone() {
		// the other half, and the one that matters more: correcting every
		// block would make every block start with a jump. Twenty-frame blocks
		// at a thousand a second are twenty milliseconds each, and the
		// snapshot is five milliseconds behind, which is what the real thing
		// looks like.
		let (bank, sound) = banked(vec![0; 4000], 1000, 1);
		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.0; 20 * CHANNELS];

		mixer.render(&bank, &one(plain(sound)), &mut out);
		mixer.render(
			&bank,
			&Snapshot {
				voices: vec![Playing { head: 0.015, ..plain(sound) }],
				step: 2,
			},
			&mut out,
		);

		assert!(
			(mixer.heads[0].frame - 40.0).abs() < 1e-6,
			"two blocks in, not corrected back to fifteen milliseconds: {}",
			mixer.heads[0].frame
		);
	}

	#[test]
	fn a_playhead_is_not_dragged_back_by_a_snapshot_that_has_not_moved() {
		// found by playing a real recording: a device asking for samples
		// faster than the simulation produces snapshots renders several blocks
		// against one of them, and correcting against a standing snapshot each
		// time replays the same fiftieth of a second forever. What came out
		// was an envelope that restarted every seven blocks.
		let (bank, sound) = banked(vec![0; 40_000], 1000, 1);
		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.0; 100 * CHANNELS];
		let standing = one(plain(sound));

		for _ in 0..10 {
			mixer.render(&bank, &standing, &mut out);
		}

		assert!(
			(mixer.heads[0].frame - 1000.0).abs() < 1e-6,
			"ten blocks of a hundred frames, played through: {}",
			mixer.heads[0].frame
		);
	}

	#[test]
	fn a_device_asking_for_a_long_block_does_not_correct_itself_every_time() {
		// the bug the fixed threshold had, found by a test that was wrong for
		// the same reason: a block longer than the slew is a disagreement the
		// block itself explains, and jumping back at the start of every one of
		// them is a stutter that would only appear on somebody else's hardware.
		let (bank, sound) = banked(vec![0; 40_000], 1000, 1);
		let mut mixer = Mixer::new(1000);
		// a fifth of a second at a time, four times the slew.
		let mut out = vec![0.0; 200 * CHANNELS];

		mixer.render(&bank, &one(plain(sound)), &mut out);

		// the snapshot is a *tenth* of a second along while the mixer has done
		// a whole fifth, which is the shape of the real thing: a block is
		// rendered ahead of the step that describes it. A hundred frames apart
		// is past the slew on its own and inside the slew plus the block, so
		// this is the case that tells the two thresholds apart.
		mixer.render(
			&bank,
			&Snapshot {
				voices: vec![Playing { head: 0.1, ..plain(sound) }],
				step: 2,
			},
			&mut out,
		);

		assert!(
			(mixer.heads[0].frame - 400.0).abs() < 1e-6,
			"it played two whole blocks rather than being pulled back: {}",
			mixer.heads[0].frame
		);
	}

	#[test]
	fn the_two_clocks_wrapping_on_either_side_of_a_loop_is_not_a_drift() {
		// the arithmetic that stops a click at every loop point: near the
		// join, one clock is at the end and the other at the start, and a
		// straight subtraction calls that a whole recording apart.
		let length = 100.0;

		assert!(apart(99.0, 1.0, length, true) < 3.0, "two frames apart, the short way round");
		assert!(
			apart(99.0, 1.0, length, false) > 90.0,
			"and ninety-eight the long way, which is what a one-shot means"
		);
		assert!(apart(10.0, 40.0, length, true) > 25.0, "a real gap is still a real gap");
		assert!(apart(5.0, 5.0, length, true).abs() < 1e-9, "and none is none");
	}

	#[test]
	fn forgetting_starts_every_slot_over() {
		let (bank, sound) = banked(vec![0, 8192, 16384, 24576], 1000, 1);
		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.0; 2 * CHANNELS];

		mixer.render(&bank, &one(plain(sound)), &mut out);
		mixer.forget();
		mixer.render(&bank, &one(plain(sound)), &mut out);

		assert!(lefts(&out)[0].abs() < 1e-4, "the first sample, not the third");
	}

	#[test]
	fn a_rate_of_nothing_is_taken_as_one_rather_than_divided_by() {
		let mixer = Mixer::new(0);

		assert_eq!(mixer.rate(), 1, "a device that reported nothing is still a device");
	}

	#[test]
	fn a_whole_world_reaches_the_buffer() {
		// end to end through the real path: a world, a voice, a snapshot, a
		// bank, and samples out. Every piece above is tested on its own; this
		// is the one that would catch two of them disagreeing about what a
		// slot or a handle means.
		let mut sounds = Sounds::new();
		let sound = sounds.insert("sounds/test", SoundData {
			samples: vec![16384; 100],
			rate: 1000,
			channels: 1,
		});
		let mut bank = Bank::new();
		bank.sync(&sounds);

		let mut voices = Voices::new();
		voices.play(Voice::at(sound, Vec3::X * 2.0).range(2.0, 100.0));

		let mut snapshot = Snapshot::new();
		snapshot.fill(&voices, &Listener::DEFAULT, &Mix::FULL);

		let mut mixer = Mixer::new(1000);
		let mut out = vec![0.0; 4 * CHANNELS];
		mixer.render(&bank, &snapshot, &mut out);

		assert!(out[1] > 0.3, "it is to the right of somebody facing negative z: {}", out[1]);
		assert!(out[0].abs() < out[1], "and quieter on the left: {}", out[0]);
	}
}
