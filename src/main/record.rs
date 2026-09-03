//! Recording what a run sounds like, instead of what it looks like.
//!
//! `colby --record sound.wav` is the exact mirror of `--shot`: it loads the
//! game module the window path does, runs it for a fixed number of simulation
//! steps, and writes what came out. No window, no output device, and no clock -
//! the mixer is asked for exactly one step's worth of samples after every step,
//! so the same build produces the same file on every machine.
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

use std::{fs, path::PathBuf};

use colby_asset::wav;
use colby_audio::{Bank, CHANNELS, Mixer, Snapshot};
use colby_core::{
	Result,
	abi::{Input, SoundData, World},
	err, info,
	time::{Rate, STEP},
};
use colby_physics::Simulation;
use colby_script::Vm;
use colby_ui::Interface;

use crate::{assets::Assets, game::Game, step};

/// The flag that asks for a recording.
const FLAG: &str = "--record";

/// Where the sound goes when the flag is given without a path.
const DEFAULT_PATH: &str = "colby.wav";

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
const DEFAULT_STEPS: u32 = 90;

/// The most steps one recording may be.
///
/// Nothing streams here, so the file is built in memory and then written, and
/// the sample cap a sound may have is what this is derived from: about three
/// minutes of stereo at this rate. Past it the encoder would refuse the whole
/// thing after doing all the work.
const MAX_STEPS: u32 = 10_000;

/// What the command line asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Request {
	/// Where to write the file.
	pub(crate) path: PathBuf,

	/// How many simulation steps to run.
	pub(crate) steps: u32,
}

/// Reads the command line for a recording request.
///
/// Accepts `--record` on its own, `--record path`, `--record=path`, and either
/// of the last two followed by a number of steps.
///
/// @return what to record, if anything was asked for
#[must_use]
pub(crate) fn requested() -> Option<Request> {
	let arguments: Vec<String> = std::env::args().skip(1).collect();

	parse(&arguments)
}

/// The same, over arguments already collected.
///
/// Split out so that it can be tested, which the environment cannot be.
fn parse(arguments: &[String]) -> Option<Request> {
	for (index, argument) in arguments.iter().enumerate() {
		let path = if let Some(rest) = argument.strip_prefix(&format!("{FLAG}=")) {
			Some(PathBuf::from(rest))
		} else if argument == FLAG {
			arguments
				.get(index + 1)
				.filter(|next| !next.starts_with('-'))
				.map(PathBuf::from)
		} else {
			continue;
		};

		let Some(path) = path else {
			return Some(Request {
				path: PathBuf::from(DEFAULT_PATH),
				steps: DEFAULT_STEPS,
			});
		};

		// a number after the path, and only if it is one: `--record out.wav`
		// followed by another flag must not eat it.
		let steps = arguments
			.get(index + if argument == FLAG { 2 } else { 1 })
			.and_then(|word| word.parse::<u32>().ok())
			.unwrap_or(DEFAULT_STEPS);

		return Some(Request { path, steps: steps.clamp(1, MAX_STEPS) });
	}

	None
}

/// Runs the game for a moment and writes what came out of the mixer.
///
/// @param request - where to write and how long for
/// @return `Ok` once the file is on disk
pub(crate) fn take(request: &Request) -> Result {
	// boxed and installed before anything else touches the world, for the same
	// reason the window and the screenshot box it: the world keeps this
	// address.
	let mut simulation = Box::new(Simulation::new());
	let mut world = Box::<World>::default();
	world.install_physics(simulation.table());

	// assets first, so the game's `init` resolves its meshes and its sounds by
	// name. A recording that skipped this would be a recording of a game that
	// found nothing to play.
	Assets::new(&crate::workspace()).sync(&mut world);

	let mut game = Game::open(&mut world)?;
	let mut input = Input::default();
	let mut scripts = Vm::new(crate::console::publisher())?;
	let mut interface = Interface::new();

	// the viewport a screenshot uses, at a scale of one, and for the same
	// reason: a document laid out against another size would put its buttons
	// somewhere else, and a script that reacts to one would react differently.
	// Nothing here draws it.
	let viewport = colby_core::glam::Vec2::new(1280.0, 720.0);
	world.ui.set_viewport(viewport, 1.0);
	world.aspect = viewport.x / viewport.y;

	let mut bank = Bank::new();
	let mut mixer = Mixer::new(RATE);
	let mut snapshot = Snapshot::new();
	let mut block = vec![0.0_f32; FRAMES_PER_STEP * CHANNELS];
	let mut samples: Vec<i16> = Vec::with_capacity(
		usize::try_from(request.steps)
			.unwrap_or(0)
			.saturating_mul(FRAMES_PER_STEP)
			.saturating_mul(CHANNELS),
	);

	for number in 1..=request.steps {
		// the simulated time this step ends at, computed rather than
		// accumulated, exactly as a screenshot does it.
		let time = (STEP * number).as_secs_f32();

		step::run(
			&mut world,
			step::Parts {
				game: Some(&mut game),
				interface: &mut interface,
				scripts: Some(&mut scripts),
				simulation: simulation.as_mut(),
				// no device here either. What replaces it is the two lines
				// below, which are what a device's callback would have done.
				audio: None,
				// no wire either, for the same reason.
				wire: None,
			},
			&mut input,
			Rate::DEFAULT,
			time,
			false,
		);

		// a game may register a sound of its own, and this is the only place
		// that would notice. It is a scan over a handful of revisions.
		bank.sync(&world.sounds);
		snapshot.take(&world);
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

	if let Some(parent) = request
		.path
		.parent()
		.filter(|it| !it.as_os_str().is_empty())
	{
		fs::create_dir_all(parent)?;
	}

	fs::write(&request.path, &bytes)
		.map_err(|error| err!(Asset("writing {}: {error}", request.path.display())))?;
	game.close(&mut world);

	// peak and loudness in the line, because they are what somebody asks next
	// and because a file of silence and a file of noise are the same size.
	info!(
		path = %request.path.display(),
		steps = request.steps,
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

	/// The command line, as strings.
	fn line(words: &[&str]) -> Vec<String> {
		words
			.iter()
			.map(|word| (*word).to_owned())
			.collect()
	}

	#[test]
	fn a_command_line_with_no_flag_asks_for_nothing() {
		assert_eq!(parse(&line(&["--shot", "picture.png"])), None);
		assert_eq!(parse(&line(&[])), None);
	}

	#[test]
	fn the_flag_on_its_own_writes_where_it_says_it_will() {
		assert_eq!(
			parse(&line(&["--record"])),
			Some(Request {
				path: PathBuf::from(DEFAULT_PATH),
				steps: DEFAULT_STEPS,
			})
		);
	}

	#[test]
	fn a_path_is_taken_either_way_round() {
		for words in [&["--record", "out.wav"][..], &["--record=out.wav"][..]] {
			assert_eq!(
				parse(&line(words)),
				Some(Request {
					path: PathBuf::from("out.wav"),
					steps: DEFAULT_STEPS,
				}),
				"{words:?}"
			);
		}
	}

	#[test]
	fn a_number_after_the_path_is_a_step_count() {
		for words in [&["--record", "out.wav", "300"][..], &["--record=out.wav", "300"][..]] {
			assert_eq!(
				parse(&line(words)),
				Some(Request {
					path: PathBuf::from("out.wav"),
					steps: 300,
				}),
				"{words:?}"
			);
		}
	}

	#[test]
	fn a_flag_after_the_path_is_not_a_step_count() {
		assert_eq!(
			parse(&line(&["--record", "out.wav", "--other"])),
			Some(Request {
				path: PathBuf::from("out.wav"),
				steps: DEFAULT_STEPS,
			}),
			"a word that is not a number leaves the default alone"
		);
	}

	#[test]
	fn a_flag_where_the_path_would_be_is_not_a_path() {
		// `--record --shot out.png` asks for a recording with no path and a
		// screenshot, not for a recording called `--shot`.
		assert_eq!(
			parse(&line(&["--record", "--shot", "out.png"])),
			Some(Request {
				path: PathBuf::from(DEFAULT_PATH),
				steps: DEFAULT_STEPS,
			})
		);
	}

	#[test]
	fn a_step_count_nothing_could_hold_is_clamped_rather_than_refused() {
		assert_eq!(
			parse(&line(&["--record", "out.wav", "9999999"])).map(|it| it.steps),
			Some(MAX_STEPS),
			"the encoder would refuse the whole thing after doing all the work"
		);
		assert_eq!(parse(&line(&["--record", "out.wav", "0"])).map(|it| it.steps), Some(1));
	}

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
