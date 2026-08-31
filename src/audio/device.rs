//! The output device, and the one place in this crate a test cannot reach.
//!
//! Everything beside this file is arithmetic over slices and is checked by
//! running it. This is a driver, a thread that is not ours, and a callback we
//! do not decide the cadence of; what it can do is be small, and be the only
//! thing that is like that.
//!
//! **What crosses the thread, and how.** One mutex holding the bank and the
//! latest snapshot. The simulation takes it for the few microseconds it takes
//! to copy sixty-four records; the callback **tries** for it and fills silence
//! if somebody else has it, because a callback that waits is a callback that
//! misses its deadline and a driver that misses its deadline is a click either
//! way. The refusals are counted rather than hidden, which is the same rule the
//! debug table's dropped lines follow.
//!
//! The other thing that could have crossed and does not is the mixer: it lives
//! inside the callback, so its playheads need no lock at all.
//!
//! **Floating-point output only.** Shared-mode WASAPI hands out `f32` on every
//! machine this has been run on, and a device wanting sixteen-bit integers
//! would need a second closure rather than a second design. It is named here
//! rather than silently unsupported: the error says which format was offered.

use std::sync::{
	Arc, Mutex,
	atomic::{AtomicU32, Ordering},
};

use colby_core::{
	Result,
	abi::{Sounds, World},
	err, info,
};
use cpal::{
	SampleFormat, Stream, StreamConfig,
	traits::{DeviceTrait, HostTrait, StreamTrait},
};

use crate::{
	bank::Bank,
	mix::{CHANNELS, Mixer},
	snapshot::Snapshot,
};

/// The most frames the fan-out scratch is sized for.
///
/// Eighty-five milliseconds at forty-eight kilohertz, which is several times
/// any block a driver has been seen to ask for. A callback wanting more than
/// this is served in pieces rather than by growing a buffer, because growing
/// one means allocating on a thread that must not.
pub const MAX_BLOCK_FRAMES: usize = 4096;

/// What the simulation writes and the callback reads.
struct Shared {
	/// The mixer's own copy of every sound.
	bank: Bank,

	/// What was playing at the end of the last step.
	snapshot: Snapshot,
}

/// An open output device, and the mixer feeding it.
///
/// Dropping this stops the stream. It is deliberately not `Send`: a `Stream` is
/// tied to the thread that made it on some platforms, and the runner has only
/// ever had one thread that owns state.
pub struct Device {
	/// First, so it stops before the state behind it goes away.
	stream: Stream,
	shared: Arc<Mutex<Shared>>,
	filled: Arc<AtomicU32>,
	missed: Arc<AtomicU32>,
	rate: u32,
	channels: u16,
	description: String,
}

impl Device {
	/// Opens the machine's default output.
	///
	/// @return the device, or why there is not going to be any sound. Every
	/// failure here is one the runner carries on past: an engine whose picture
	/// works and whose audio did not start is still worth looking at.
	pub fn open() -> Result<Self> {
		let host = cpal::default_host();
		let device = host
			.default_output_device()
			.ok_or_else(|| err!(Audio("this machine reports no default output device")))?;
		let description = device
			.description()
			.map_or_else(|_| "an output device".to_owned(), |it| it.to_string());
		let chosen = device.default_output_config().map_err(|error| {
			err!(Audio("{description} would not say what format it wants: {error}"))
		})?;

		if chosen.sample_format() != SampleFormat::F32 {
			return Err(err!(Audio(
				"{description} wants {} samples and this build writes floating-point ones",
				chosen.sample_format()
			)));
		}

		let config = chosen.config();
		let rate = chosen.sample_rate();
		let channels = chosen.channels();
		let shared = Arc::new(Mutex::new(Shared {
			bank: Bank::new(),
			snapshot: Snapshot::new(),
		}));
		let filled = Arc::new(AtomicU32::new(0));
		let missed = Arc::new(AtomicU32::new(0));
		let stream = build(&device, &config, Counters {
			rate,
			channels,
			shared: &shared,
			filled: &filled,
			missed: &missed,
		})?;

		stream
			.play()
			.map_err(|error| err!(Audio("{description} would not start: {error}")))?;

		info!(device = %description, rate, channels, "audio device open");

		Ok(Self {
			stream,
			shared,
			filled,
			missed,
			rate,
			channels,
			description,
		})
	}

	/// Copies whatever sounds have changed into the mixer's own bank.
	///
	/// Holds the lock for the length of the copy, so a callback landing in the
	/// middle of one fills silence. That is a block, it happens when somebody
	/// recompiles a `.wav` while the engine is running, and the alternative is
	/// holding a third copy of every sound.
	///
	/// @param sounds - the registry, as the world has it
	/// @return how many were copied
	pub fn load(&mut self, sounds: &Sounds) -> usize {
		let Ok(mut shared) = self.shared.lock() else {
			return 0;
		};

		shared.bank.sync(sounds)
	}

	/// Hands the callback what is playing as of the end of this step.
	///
	/// @param world - the world the step just finished with
	pub fn publish(&mut self, world: &World) {
		let Ok(mut shared) = self.shared.lock() else {
			return;
		};

		shared.snapshot.take(world);
	}

	/// Stops the stream without dropping the device.
	///
	/// What a pause would call, if anything called it. Kept because the pair
	/// is what a stream has and because starting one again is the only way to
	/// recover from a device that stopped itself.
	pub fn pause(&mut self) -> Result<()> {
		self.stream
			.pause()
			.map_err(|error| err!(Audio("{}: {error}", self.description)))
	}

	/// Starts it again.
	pub fn resume(&mut self) -> Result<()> {
		self.stream
			.play()
			.map_err(|error| err!(Audio("{}: {error}", self.description)))
	}

	/// Frames a second, as the device asked for them.
	#[must_use]
	pub const fn rate(&self) -> u32 { self.rate }

	/// How many speakers it has.
	#[must_use]
	pub const fn channels(&self) -> u16 { self.channels }

	/// What the machine calls it.
	#[must_use]
	pub fn description(&self) -> &str { &self.description }

	/// How many blocks the device has asked for since it opened.
	///
	/// The one thing that says a stream is alive without anybody listening to
	/// it: a device that opened and then never called back reads zero here,
	/// and that is a completely different failure from one that is playing
	/// silence.
	#[must_use]
	pub fn blocks(&self) -> u32 { self.filled.load(Ordering::Relaxed) }

	/// How many blocks have been filled with silence because the lock was
	/// taken.
	///
	/// Counted for the run rather than for a step, unlike the voice table's
	/// own refusals: a dropout is rare enough that the interesting question is
	/// whether there have been any at all.
	#[must_use]
	pub fn missed(&self) -> u32 { self.missed.load(Ordering::Relaxed) }

	/// How many samples the bank is holding.
	#[must_use]
	pub fn samples(&self) -> usize {
		self.shared
			.lock()
			.map_or(0, |shared| shared.bank.samples())
	}
}

/// Everything the callback is built out of besides the device itself.
///
/// A struct rather than five more arguments, which is what the house rules ask
/// for once a signature gets long and what this one wanted anyway.
#[derive(Clone, Copy)]
struct Counters<'a> {
	/// Frames a second, as the device asked for them.
	rate: u32,

	/// How many speakers it has.
	channels: u16,

	/// The bank and the snapshot.
	shared: &'a Arc<Mutex<Shared>>,

	/// Blocks asked for.
	filled: &'a Arc<AtomicU32>,

	/// Blocks the lock was busy for.
	missed: &'a Arc<AtomicU32>,
}

/// Builds the stream, with the mixer and the scratch buffer moved inside it.
fn build(device: &cpal::Device, config: &StreamConfig, about: Counters<'_>) -> Result<Stream> {
	let mut mixer = Mixer::new(about.rate);
	// only the fan-out needs it, and only when the device is not stereo. Sized
	// once, here, because the callback may not allocate.
	let mut scratch = vec![0.0_f32; MAX_BLOCK_FRAMES * CHANNELS];
	let voices = Arc::clone(about.shared);
	let counted = Arc::clone(about.filled);
	let refused = Arc::clone(about.missed);
	let speakers = usize::from(about.channels).max(1);

	device
		.build_output_stream::<f32, _, _>(
			config,
			move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
				counted.fetch_add(1, Ordering::Relaxed);

				let Ok(state) = voices.try_lock() else {
					// somebody is writing a snapshot or copying a sound.
					// Waiting for them is how a callback misses its deadline.
					refused.fetch_add(1, Ordering::Relaxed);
					out.fill(0.0);

					return;
				};

				if speakers == CHANNELS {
					mixer.render(&state.bank, &state.snapshot, out);

					return;
				}

				for piece in out.chunks_mut(MAX_BLOCK_FRAMES * speakers) {
					let frames = piece.len() / speakers;
					let stereo = scratch
						.get_mut(..frames * CHANNELS)
						.unwrap_or_default();

					mixer.render(&state.bank, &state.snapshot, stereo);
					spread(stereo, piece, speakers);
				}
			},
			|error| colby_core::warn!(%error, "the audio stream failed"),
			None,
		)
		.map_err(|error| err!(Audio("the output stream would not open: {error}")))
}

/// Writes a stereo block out across however many speakers a device has.
///
/// The three cases are all a device has ever turned out to be, and the third is
/// a guess that is at least not wrong: a surround device gets the front pair
/// and silence everywhere else, rather than the same thing out of every speaker
/// with the room's worth of comb filtering that produces.
///
/// @param stereo - interleaved left and right
/// @param out - the device's buffer, `channels` samples to a frame
/// @param channels - how many speakers
pub fn spread(stereo: &[f32], out: &mut [f32], channels: usize) {
	if channels == 0 {
		return;
	}

	for (frame, speakers) in out.chunks_mut(channels).enumerate() {
		let left = stereo
			.get(frame * CHANNELS)
			.copied()
			.unwrap_or(0.0);
		let right = stereo
			.get(frame * CHANNELS + 1)
			.copied()
			.unwrap_or(0.0);

		match channels {
			| 1 => speakers[0] = (left + right) * 0.5,
			| _ => {
				speakers[0] = left;
				speakers[1] = right;

				for spare in speakers.iter_mut().skip(CHANNELS) {
					*spare = 0.0;
				}
			},
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_stereo_device_gets_the_block_as_it_is() {
		let stereo = [1.0, -1.0, 0.5, -0.5];
		let mut out = vec![0.0; 4];

		spread(&stereo, &mut out, 2);

		// exact: the fan-out copies rather than computes, and anything but exact
		// equality here would let a stray multiply through.
		assert_eq!(out, vec![1.0, -1.0, 0.5, -0.5]);
	}

	#[test]
	fn a_mono_device_gets_the_two_averaged() {
		let stereo = [1.0, -1.0, 0.5, 0.5];
		let mut out = vec![0.0; 2];

		spread(&stereo, &mut out, 1);

		assert!(out[0].abs() < 1e-6, "one against minus one is nothing: {}", out[0]);
		assert!((out[1] - 0.5).abs() < 1e-6, "and a half against a half is a half");
	}

	#[test]
	fn a_surround_device_gets_the_front_pair_and_silence() {
		let stereo = [1.0, -1.0];
		let mut out = vec![9.0; 6];

		spread(&stereo, &mut out, 6);

		assert!((out[0] - 1.0).abs() < f32::EPSILON, "left");
		assert!((out[1] + 1.0).abs() < f32::EPSILON, "right");
		for (index, sample) in out.iter().enumerate().skip(2) {
			assert!(
				sample.abs() < f32::EPSILON,
				"speaker {index} was left holding {sample} from whatever was in the buffer"
			);
		}
	}

	#[test]
	fn a_block_the_stereo_side_is_too_short_for_comes_out_as_silence() {
		// the fan-out is handed a slice the mixer filled, so a mismatch is a
		// bug rather than an input - but it must not be an index out of range
		// on somebody's audio thread.
		let mut out = vec![9.0; 8];

		spread(&[1.0, -1.0], &mut out, 2);

		assert!((out[0] - 1.0).abs() < f32::EPSILON, "what there was");
		for sample in out.iter().skip(2) {
			assert!(sample.abs() < f32::EPSILON, "and silence rather than a panic");
		}
	}

	#[test]
	fn a_device_reporting_no_channels_at_all_writes_nothing() {
		let mut out = vec![9.0; 4];

		spread(&[1.0, -1.0], &mut out, 0);

		assert_eq!(out, vec![9.0; 4], "rather than dividing by it");
	}

	#[test]
	fn the_scratch_is_big_enough_for_a_block_anybody_asks_for() {
		// the number that keeps the callback from allocating. If a driver ever
		// asks for more than this the fan-out serves it in pieces, which is
		// what the loop in `build` is for.
		assert!(
			MAX_BLOCK_FRAMES >= 2048,
			"a driver asking for a whole frame's worth at ninety-six kilohertz wants about \
			 sixteen hundred"
		);
	}
}
