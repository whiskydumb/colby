//! Recording what a run sounds like, instead of what it looks like.
//!
//! `colby --record sound.wav` is the exact mirror of `--shot`: it brings the
//! world up the way the window path does, runs it for a fixed number of
//! simulation steps, and writes what came out. No window, no output device, and
//! no clock - the mixer is asked for exactly one step's worth of samples after
//! every step, so the same build produces the same file on every machine.
//!
//! **That is the whole point.** A screenshot is how anybody reviewing a change
//! to the renderer sees the result without being at the machine; there was no
//! equivalent for sound, and "run it and listen" is not a thing a test, a
//! reviewer or a person on the other end of a shell can do. A `.wav` with a
//! known hash is.
//!
//! It runs the same number of steps a screenshot does by default, so
//! `just shot` and `just hear` describe the same second and a half of the same
//! world - one as a picture and one as a sound.

use std::{fs, path::Path, time::Duration};

use colby_asset::wav;
use colby_audio::{Bank, CHANNELS, Mixer, Snapshot};
use colby_core::{
	Result,
	abi::{Input, SoundData},
	err, info,
	time::{Rate, STEP},
};

use crate::{Build, Front, Project, Runtime};

/// Where the sound goes when the flag is given without a path.
pub(crate) const DEFAULT_PATH: &str = "colby.wav";

/// How many frames a second the file holds.
///
/// Fixed rather than the device's, because there is no device and because a
/// file whose rate depended on what the machine had would not be comparable
/// against one taken anywhere else. Forty-eight thousand divides by sixty
/// exactly, which is what makes one step exactly [`FRAMES_PER_STEP`] frames
/// with nothing left over.
const RATE: u32 = 48_000;

/// How many steps there are in a second.
///
/// The rate a recording is pinned at, whatever a console was told: a `.wav`
/// this mode wrote has to be comparable against one written before the rate
/// was turnable. @ref [`Rate::DEFAULT`], which is the same number, and the
/// assertion below, which is what says so.
const STEPS_A_SECOND: u32 = 60;

const _: () = assert!(
	Rate::DEFAULT.hz() == 60,
	"a recording is written at the default rate, so the two have to be the same number"
);

/// How many frames one simulation step is worth.
#[expect(
	clippy::as_conversions,
	reason = "a u32 to usize is lossless on every target this builds for, and try_from is not \
	          available in a const item"
)]
const FRAMES_PER_STEP: usize = (RATE / STEPS_A_SECOND) as usize;

const _: () = assert!(
	RATE.is_multiple_of(STEPS_A_SECOND),
	"the rate has to divide by the step rate, or a recording drifts against the simulation"
);

/// How many steps to run when nobody says.
///
/// The same ninety a screenshot takes, so the two describe the same moment.
pub(crate) const DEFAULT_STEPS: u32 = 90;

/// The most steps one recording may be.
///
/// Nothing streams here, so the file is built in memory and then written, and
/// the sample cap a sound may have is what this is derived from: about three
/// minutes of stereo at this rate. Past it the encoder would refuse the whole
/// thing after doing all the work.
pub(crate) const MAX_STEPS: u32 = 10_000;

/// Runs the game for a moment and writes what came out of the mixer.
///
/// @param path - where to write the file
/// @param steps - how many simulation steps to run
/// @param project - the project to record
/// @param build - what the build script knew
/// @return `Ok` once the file is on disk
pub(crate) fn take(path: &Path, steps: u32, project: &Project, build: &Build) -> Result {
	// no console and no device here either, for the screenshot's reasons and
	// one more: what replaces the device is the two lines after every step
	// below, which are what a device's callback would have done. The runtime
	// lays the interface out against the picture's size at a scale of one, so
	// a document laid out against another size does not put its buttons
	// somewhere else and a script that reacts to one does not react
	// differently. Nothing here draws it. @ref `Front::Fixed`.
	let mut runtime = Runtime::open(Front::Fixed, project, build)?;
	let mut input = Input::default();

	let mut bank = Bank::new();
	let mut mixer = Mixer::new(RATE);
	let mut snapshot = Snapshot::new();
	let mut block = vec![0.0_f32; FRAMES_PER_STEP * CHANNELS];
	let mut samples: Vec<i16> = Vec::with_capacity(
		usize::try_from(steps)
			.unwrap_or(0)
			.saturating_mul(FRAMES_PER_STEP)
			.saturating_mul(CHANNELS),
	);

	for number in 1..=steps {
		// the simulated time this step ends at, computed rather than
		// accumulated, exactly as a screenshot does it.
		let time = (STEP * number).as_secs_f32();

		runtime.step(&mut input, Rate::DEFAULT, time, false, Duration::ZERO);

		// a game may register a sound of its own, and this is the only place
		// that would notice. It is a scan over a handful of revisions.
		bank.sync(&runtime.world.sounds);
		snapshot.take(&runtime.world);
		mixer.render(&bank, &snapshot, &mut block);
		samples.extend(block.iter().map(|sample| quantize(*sample)));
	}

	let data = SoundData {
		samples,
		rate: RATE,
		channels: u16::try_from(CHANNELS).unwrap_or(2),
	};
	let (peak, loudness) = measure(&data.samples);
	let bytes = wav::encode(&data)?;

	if let Some(parent) = path
		.parent()
		.filter(|it| !it.as_os_str().is_empty())
	{
		fs::create_dir_all(parent)?;
	}

	fs::write(path, &bytes)
		.map_err(|error| err!(Asset("writing {}: {error}", path.display())))?;
	runtime.close();

	// peak and loudness in the line, because they are what somebody asks next
	// and because a file of silence and a file of noise are the same size.
	info!(
		path = %path.display(),
		steps,
		seconds = data.seconds(),
		rate = RATE,
		peak,
		loudness,
		"recording written"
	);

	Ok(())
}

/// One mixed sample as the file holds it.
///
/// Clamped rather than wrapped, for the reason the mixer clamps: a sample
/// outside the range is not a loud sound, it is a different one.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "the value is clamped to a range an i16 holds on the line it is cast on"
)]
fn quantize(sample: f32) -> i16 { (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16 }

/// The loudest sample and the root mean square of all of them.
///
/// @return `(peak, loudness)`, both between zero and one
#[expect(
	clippy::as_conversions,
	clippy::cast_precision_loss,
	reason = "a sample count is bounded by MAX_STEPS times the frames in a step"
)]
fn measure(samples: &[i16]) -> (f32, f32) {
	if samples.is_empty() {
		return (0.0, 0.0);
	}

	let mut peak = 0.0_f32;
	let mut total = 0.0_f64;

	for sample in samples {
		let value = f32::from(*sample) / f32::from(i16::MAX);
		peak = peak.max(value.abs());
		total = f64::from(value).mul_add(f64::from(value), total);
	}

	#[expect(
		clippy::cast_possible_truncation,
		reason = "a root mean square of values between zero and one is between zero and one"
	)]
	let loudness = (total / samples.len() as f64).sqrt() as f32;

	(peak, loudness)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_step_is_a_whole_number_of_frames() {
		// the property the fixed rate is chosen for. Anything else and a
		// recording would gain or lose a sample every few steps, which is a
		// click at that rate and a drift over a minute.
		assert_eq!(
			FRAMES_PER_STEP * usize::try_from(STEPS_A_SECOND).unwrap_or(60),
			usize::try_from(RATE).unwrap_or(0),
			"sixty steps make exactly one second"
		);
	}

	#[test]
	fn a_sample_past_the_range_is_clamped_rather_than_wrapped() {
		assert_eq!(quantize(0.0), 0);
		assert_eq!(quantize(1.0), i16::MAX);
		assert_eq!(quantize(-1.0), -i16::MAX);
		assert_eq!(quantize(9.0), i16::MAX, "a loud sound, not a different one");
		assert_eq!(quantize(-9.0), -i16::MAX);
	}

	#[test]
	fn silence_measures_as_silence_and_a_square_wave_as_everything() {
		let (peak, loudness) = measure(&[0; 100]);

		assert!(peak.abs() < f32::EPSILON, "nothing is loud");
		assert!(loudness.abs() < f32::EPSILON);

		let (peak, loudness) = measure(&[i16::MAX, -i16::MAX, i16::MAX, -i16::MAX]);

		assert!((peak - 1.0).abs() < 1e-6, "and everything is: {peak}");
		assert!((loudness - 1.0).abs() < 1e-6, "with nothing in between: {loudness}");
	}

	#[test]
	fn nothing_at_all_measures_as_nothing_rather_than_dividing_by_it() {
		assert_eq!(measure(&[]), (0.0, 0.0));
	}

	#[test]
	fn a_half_scale_tone_measures_below_a_full_one() {
		let quiet = measure(&[i16::MAX / 2, -i16::MAX / 2]);
		let loud = measure(&[i16::MAX, -i16::MAX]);

		assert!(quiet.0 < loud.0, "the peak is lower");
		assert!(quiet.1 < loud.1, "and so is the loudness");
		assert!((quiet.1 - 0.5).abs() < 0.01, "by about half: {}", quiet.1);
	}
}
