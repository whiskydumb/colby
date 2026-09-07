//! What a frame of this project costs, printed as a table.
//!
//! `colby --profile` is the fourth of the family, after `--shot`, `--record`
//! and `--link`, and it is here for the reason all three of those are: a
//! change to something nobody can look at is a change nobody can review. A
//! screenshot is how a renderer change is seen from the other end of a shell;
//! a recording is how a mixer change is; a two-endpoint run is how a
//! networking change is; **this is how a change that costs something is**.
//!
//! **It is the fourth oracle and it is not one of the three.** Those three
//! print a number that is the same on every machine, and their whole value is
//! that a moved number means somebody changed something. A time is not that
//! and cannot be made into it: the same build on the same machine gives a
//! different millisecond on a warm afternoon. So this prints a table nobody
//! compares byte for byte - and, beside it, **the one number here that is
//! stable**: how many render passes the frame recorded. Fifteen with the
//! defaults, twenty-six with bloom, and a frame that grew one grew it for a
//! reason somebody can name.
//!
//! **No window and no console**, which is `--shot`'s arrangement and is taken
//! for `--shot`'s reason: what comes out has to depend on the build, on the
//! project and on the command line, not on what somebody last typed at a
//! terminal or left in a config file. **The two knobs are the scene and
//! `--set`**: lights, sky, tonemap, exposure, bloom and fog live in the
//! `.cscene` since step five, so asking what thirty-two lamps cost is a
//! question answered with a project that has thirty-two lamps in it - while
//! `r.msaa`, `r.lights` and `r.shadows` are properties of the machine and are
//! set on the command line, which is what `--set` was built for in `PERF-4`.
//!
//! **Warm frames first.** The pipelines compile on the frame that first needs
//! them, the eye starts unadapted, and the allocator has not yet grown the
//! scratch every frame reuses. Measuring any of that is measuring a start-up,
//! so the first [`WARMUP`] frames are drawn and thrown away. Wicked averages
//! the twenty most recent frames for the same reason (`wiProfiler.cpp:48`),
//! and Unreal one-poles every number in `stat unit`
//! (`UnrealClient.cpp:381-400`).
//!
//! Nothing is read back off the GPU except the ten timestamps. A picture
//! copied to a mappable buffer is three and a half megabytes that no frame
//! anybody plays ever pays, and it would be the largest row in the table.

use std::time::Duration;

use colby_core::{
	Err, Result,
	abi::Input,
	info,
	time::{Rate, STEP},
	warn,
};
use colby_engine::{
	Capture, Gpu, Overlay, gpu,
	timing::{Frame, Pass, Work},
};
use colby_physics::Spent;

use crate::{Asked, Build, Front, Project, Runtime};

/// How big the frames are.
///
/// The same as `--shot`, deliberately: a fill-rate answer is a function of how
/// many pixels there are, so the two tools have to be talking about the same
/// picture or a screenshot is no guide to what a measurement meant.
const SIZE: (u32, u32) = (1280, 720);

/// How many frames to measure when the command line does not say.
///
/// Two seconds at the fixed step, which is long enough for a mean to stop
/// wandering and short enough that nobody minds waiting for it.
pub(crate) const FRAMES: u32 = 120;

/// The most frames anybody may ask for.
///
/// Every frame blocks on the queue, so a run is about as long as it says it
/// is; ten thousand of them is three minutes, which is past the point where
/// somebody meant it.
pub(crate) const MAX_FRAMES: u32 = 10_000;

/// How many frames are drawn and thrown away first.
///
/// Half a second. What it buys is every pipeline compiled, the eye adapted and
/// every scratch buffer grown, none of which happens twice.
const WARMUP: u32 = 30;

/// How many parts of a frame the table has rows for.
///
/// Five spans of hardware, two of recording and five of simulation.
const ROWS: usize = 12;

/// One row of the table: what a part of the frame cost over the whole run.
#[derive(Clone, Copy, Debug, Default)]
struct Row {
	/// Every frame's answer added together.
	total: Duration,

	/// The worst single frame.
	///
	/// Kept beside the mean because they answer different questions: a mean
	/// says what the run cost and a worst says what a hitch looks like, and a
	/// part of the frame that is cheap on average and occasionally enormous is
	/// invisible in the first and obvious in the second. Unreal shows both for
	/// every number in `stat unit` (`UnrealClient.h:299-330`).
	worst: Duration,

	/// How many frames had this part in them at all.
	///
	/// A bloom chain in a project that does not bloom is nought here rather
	/// than a mean of zero milliseconds.
	frames: u32,
}

impl Row {
	/// Folds one frame in.
	fn add(&mut self, took: Duration) {
		self.total = self.total.saturating_add(took);
		self.worst = self.worst.max(took);
		self.frames = self.frames.saturating_add(1);
	}

	/// The mean over the frames that had it.
	fn mean(&self) -> Duration {
		self.total
			.checked_div(self.frames)
			.unwrap_or_default()
	}
}

/// Every row, and what the run was.
#[derive(Debug)]
struct Table {
	/// One per name in [`NAMES`], in that order.
	rows: [Row; ROWS],

	/// How many render passes each frame recorded, and `None` before the first
	/// one.
	passes: Option<u32>,

	/// Whether a frame ever recorded a different number of passes from the
	/// one before it.
	///
	/// Worth knowing rather than worth averaging: a project whose bloom comes
	/// and going mid-run is a project whose table describes two things.
	steady: bool,
}

/// What each row is called, in the order they are reported.
///
/// The five hardware spans first, then what this thread spent recording them,
/// then what the step underneath cost - which is the order a frame actually
/// happens in, read from the outside in.
const NAMES: [&str; ROWS] = [
	"gpu shadow",
	"gpu scene",
	"gpu meter",
	"gpu glow",
	"gpu composite",
	"cpu upload",
	"cpu record",
	"cpu broad",
	"cpu narrow",
	"cpu buoyancy",
	"cpu solve",
	"cpu step",
];

impl Table {
	/// A table nothing has been folded into.
	const fn new() -> Self {
		Self {
			rows: [Row {
				total: Duration::ZERO,
				worst: Duration::ZERO,
				frames: 0,
			}; ROWS],
			passes: None,
			steady: true,
		}
	}

	/// Folds one frame's answers in.
	///
	/// @param frame - what the renderer measured
	/// @param spent - what the last simulation step cost
	fn fold(&mut self, frame: &Frame, spent: Spent) {
		for (at, pass) in Pass::ALL.into_iter().enumerate() {
			if let (Some(row), Some(took)) = (self.rows.get_mut(at), frame.pass(pass)) {
				row.add(took);
			}
		}

		for (at, work) in Work::ALL.into_iter().enumerate() {
			if let (Some(row), Some(took)) =
				(self.rows.get_mut(Pass::ALL.len() + at), frame.work(work))
			{
				row.add(took);
			}
		}

		let under = Pass::ALL.len() + Work::ALL.len();

		for (at, took) in [spent.broad, spent.narrow, spent.buoyancy, spent.solve, spent.total]
			.into_iter()
			.enumerate()
		{
			if let Some(row) = self.rows.get_mut(under + at) {
				row.add(took);
			}
		}

		match self.passes {
			| Some(before) if before != frame.passes() => self.steady = false,
			| _ => {},
		}

		self.passes = Some(frame.passes());
	}

	/// Prints it.
	///
	/// One line a row with the same field names throughout, which is what
	/// makes it read as a table in a terminal that prints structured lines.
	///
	/// @param frames - how many frames were folded in
	fn report(&self, frames: u32) {
		for (name, row) in NAMES.into_iter().zip(self.rows) {
			if row.frames == 0 {
				info!(part = name, "never ran");

				continue;
			}

			info!(
				part = name,
				mean_us = row.mean().as_micros(),
				worst_us = row.worst.as_micros(),
				frames = row.frames,
			);
		}

		// the three numbers that add up to a frame, and they are three rather
		// than two because the simulation's rows overlap: `cpu step` is the
		// whole step and the four above it are parts of it - and `cpu broad` is
		// in turn a part of `cpu narrow` - so adding every row would count the
		// same microseconds twice over.
		info!(
			frames,
			passes = self.passes.unwrap_or_default(),
			steady = self.steady,
			gpu_us = self.slice(0..Pass::ALL.len()).as_micros(),
			cpu_us = self
				.slice(Pass::ALL.len()..Pass::ALL.len() + Work::ALL.len())
				.as_micros(),
			step_us = self.slice(ROWS - 1..ROWS).as_micros(),
			"a frame"
		);

		if !self.steady {
			warn!(
				"the number of passes moved during the run, so these means describe more than \
				 one kind of frame"
			);
		}
	}

	/// The mean of several rows added together.
	fn slice(&self, range: core::ops::Range<usize>) -> Duration {
		self.rows
			.get(range)
			.unwrap_or_default()
			.iter()
			.fold(Duration::ZERO, |sum, row| sum.saturating_add(row.mean()))
	}
}

/// Runs the project for a while and prints what a frame of it costs.
///
/// @param project - the project to measure
/// @param build - what the build script knew
/// @param asked - the variables the command line set, which since `PERF-4` is
/// how this run is told to draw with one sample or eight lamps
/// @param frames - how many frames to measure, after the warm-up
/// @return `Ok` once the table has been printed
pub(crate) fn take(project: &Project, build: &Build, asked: &Asked, frames: u32) -> Result {
	// the adapter first, exactly as a screenshot does: a machine with nothing
	// to render on has no business loading a module to find that out.
	let Some(gpu) = Gpu::open(gpu::backends(None), None)? else {
		return Err!(Graphics("no usable adapter, so there is nothing to measure"));
	};
	let mut capture = Capture::new(&gpu, SIZE.0, SIZE.1)?;
	let hardware = capture.scene_mut().measure();

	if !hardware {
		warn!(
			"this adapter has no timestamp queries, so the five hardware rows will be empty and \
			 only the wall clock is answering"
		);
	}

	// no console and no device, which is `Front::Fixed` and is the whole of
	// why the scene is the only knob. @ref the module note.
	let mut runtime = Runtime::open(Front::Fixed, project, build, asked)?;
	let mut input = Input::default();

	runtime
		.interface
		.attach(capture.device(), capture.format())?;

	info!(
		project = %project.root().display(),
		frames,
		warmup = WARMUP,
		width = SIZE.0,
		height = SIZE.1,
		hardware,
		"measuring"
	);

	let mut table = Table::new();

	for number in 1..=WARMUP.saturating_add(frames) {
		// one step a frame, which is what a window running at the tick rate
		// does. The simulated time is computed rather than accumulated, for
		// the reason a screenshot computes it: however the arithmetic rounds,
		// the hundredth frame is the hundredth step.
		let time = (STEP * number).as_secs_f32();

		runtime.step(&mut input, Rate::DEFAULT, time, false, Duration::ZERO);
		runtime.world.set_interpolation(1.0);
		runtime.interface.run(&runtime.world);
		runtime
			.interface
			.prepare(capture.device(), capture.queue(), &runtime.world);

		let overlay: &mut dyn Overlay = &mut runtime.interface;

		capture.draw(&mut runtime.world, &mut [overlay]);

		// after the submit and before the next frame is recorded: this blocks
		// on the queue, which is what makes each frame's numbers its own.
		// @ref [`Timings::settle`](colby_engine::Timings::settle).
		let frame = capture.scene_mut().settle();

		if number > WARMUP {
			table.fold(&frame, runtime.simulation.spent());
		}
	}

	table.report(frames);
	runtime.close();

	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_row_reports_the_mean_of_what_went_into_it_and_the_worst_of_it() {
		let mut row = Row::default();

		row.add(Duration::from_micros(100));
		row.add(Duration::from_micros(300));
		row.add(Duration::from_micros(200));

		assert_eq!(row.mean(), Duration::from_micros(200));
		assert_eq!(row.worst, Duration::from_micros(300));
		assert_eq!(row.frames, 3);
	}

	#[test]
	fn a_row_nothing_went_into_is_no_answer_rather_than_a_division_by_nought() {
		// the glow chain in a project that does not bloom, which is every
		// project by default: `Post::DEFAULT` has bloom at nought.
		let row = Row::default();

		assert_eq!(row.mean(), Duration::ZERO);
		assert_eq!(row.frames, 0);
	}

	#[test]
	fn every_row_of_the_table_has_a_name_and_no_two_share_one() {
		let mut names = NAMES.to_vec();
		names.sort_unstable();
		names.dedup();

		assert_eq!(names.len(), ROWS, "two rows answer to one name");
		assert_eq!(
			Pass::ALL.len() + Work::ALL.len() + 5,
			ROWS,
			"the table has a row for every span the renderer and the simulation report"
		);
	}

	#[test]
	fn a_run_whose_pass_count_never_moved_is_steady_and_one_whose_did_is_not() {
		// the one number here worth comparing between two runs, and it is only
		// worth comparing while it holds still for the whole of one.
		let mut table = Table::new();

		table.fold(&Frame::default(), Spent::default());
		table.fold(&Frame::default(), Spent::default());

		assert!(table.steady, "two frames that recorded the same passes");
		assert_eq!(table.passes, Some(0));
	}

	#[test]
	fn the_simulation_rows_are_folded_in_beside_the_renderers() {
		let mut table = Table::new();
		let spent = Spent {
			broad: Duration::from_micros(12),
			narrow: Duration::from_micros(40),
			buoyancy: Duration::from_micros(7),
			solve: Duration::from_micros(90),
			total: Duration::from_micros(200),
		};

		table.fold(&Frame::default(), spent);

		let under = Pass::ALL.len() + Work::ALL.len();

		assert_eq!(table.rows[under].mean(), Duration::from_micros(12), "broad");
		assert_eq!(table.rows[under + 1].mean(), Duration::from_micros(40), "narrow");
		assert_eq!(table.rows[under + 2].mean(), Duration::from_micros(7), "buoyancy");
		assert_eq!(table.rows[under + 3].mean(), Duration::from_micros(90), "solve");
		assert_eq!(table.rows[under + 4].mean(), Duration::from_micros(200), "the step");
	}

	#[test]
	fn the_warm_up_is_shorter_than_the_run_it_is_in_front_of() {
		// a table whose warm-up is most of it is a table of a start-up.
		assert!(WARMUP < FRAMES, "the default run is mostly measurement");
		assert!(FRAMES <= MAX_FRAMES);
	}
}
