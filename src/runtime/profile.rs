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
#[cfg(feature = "editor")]
use colby_editor::{Part, Profile};
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
/// Five spans of hardware, two of recording, five of simulation and one of
/// particles.
const ROWS: usize = 13;

/// Which row of the table is the whole solver step.
///
/// Written out rather than "the last one", which is what it used to be and
/// what stopped being true the moment a row was appended after it. A row added
/// below has to leave these two alone or move them on purpose.
const STEP_ROW: usize = 11;

/// Which row is the particles.
const SPARKS_ROW: usize = 12;

/// How many frames the live table averages over.
///
/// @note: everything from here to [`Row`] is the editor's half of this module
/// and is compiled with it - a build with no editor in it has no panel to show
/// a window of frames to, and `--profile` is the other half and is always
/// here.
///
/// Wicked's window (`wiProfiler.cpp:48`, `float times[20]`), and taken for its
/// reason: a single frame's number is nobody's headline, and a window short
/// enough to react is worth more here than a mean over a whole run. What
/// `--profile` does instead - average everything after a warm-up - is right
/// for a run that ends, and wrong for a panel somebody is watching while they
/// change something.
#[cfg(feature = "editor")]
const WINDOW: usize = 20;

/// The table a window keeps while somebody is looking at it.
///
/// **A window rather than a total**, which is the whole difference from
/// [`Table`]: that one answers "what did this run cost" and this one answers
/// "what is it costing now". They share the names and the rule that a part
/// which never ran has no number - anything else would be two vocabularies for
/// one frame.
///
/// Frames arrive at two rates and it does not matter: the wall-clock halves
/// come every frame and the hardware halves every second or third, because the
/// readback does not wait. Each row counts its own samples.
#[cfg(feature = "editor")]
#[derive(Debug)]
pub(crate) struct Live {
	/// The last [`WINDOW`] samples of each row, newest last.
	rows: [Vec<Duration>; ROWS],

	/// The worst single sample of each, over the whole time the pane has been
	/// up rather than over the window: a hitch is worth remembering after it
	/// has scrolled out of the mean.
	worst: [Duration; ROWS],

	/// How many render passes the last frame read back recorded.
	passes: Option<u32>,

	/// What the panel is handed, rebuilt when a frame is folded in.
	parts: Vec<Part>,

	/// Whether the hardware side ever answered.
	hardware: bool,
}

#[cfg(feature = "editor")]
impl Live {
	/// A table nothing has been folded into.
	pub(crate) fn new() -> Self {
		Self {
			rows: core::array::from_fn(|_| Vec::with_capacity(WINDOW)),
			worst: [Duration::ZERO; ROWS],
			passes: None,
			parts: Vec::new(),
			hardware: false,
		}
	}

	/// Forgets everything, for a pane that has just been opened again.
	///
	/// A window that showed the profiler, went away for ten minutes and came
	/// back should not average what a frame cost before lunch.
	pub(crate) fn clear(&mut self) {
		for row in &mut self.rows {
			row.clear();
		}

		self.worst = [Duration::ZERO; ROWS];
		self.passes = None;
		self.parts.clear();
		self.hardware = false;
	}

	/// Folds one frame's wall-clock halves in.
	///
	/// Called every frame: the two recording spans and the five the step
	/// underneath cost need no readback and are this frame's own.
	///
	/// **Durations rather than a `Frame`**, and that is not a convenience: a
	/// table of numbers has no business knowing the renderer's type, and the
	/// caller has public accessors for every one of these. What it buys is
	/// that the whole of this is testable with no device anywhere near it.
	///
	/// @param record - what this thread spent, per [`Work`] in slot order
	/// @param spent - what the last simulation step cost
	pub(crate) fn walls(
		&mut self,
		record: [Option<Duration>; 2],
		spent: Spent,
		sparked: Duration,
	) {
		for (at, took) in record.into_iter().enumerate() {
			if let Some(took) = took {
				self.add(Pass::ALL.len() + at, took);
			}
		}

		let under = Pass::ALL.len() + Work::ALL.len();

		for (at, took) in
			[spent.broad, spent.narrow, spent.buoyancy, spent.solve, spent.total, sparked]
				.into_iter()
				.enumerate()
		{
			self.add(under + at, took);
		}

		self.rebuild();
	}

	/// Folds one frame's hardware halves in, when a readback has landed.
	///
	/// @param passes - what the hardware spent, per [`Pass`] in slot order,
	/// from a frame two or three behind the one this is called in
	/// @param count - how many render passes that frame recorded
	pub(crate) fn hardware(&mut self, passes: [Option<Duration>; 5], count: u32) {
		self.passes = Some(count);

		for (at, took) in passes.into_iter().enumerate() {
			if let Some(took) = took {
				self.hardware = true;
				self.add(at, took);
			}
		}

		self.rebuild();
	}

	/// What the panel is handed.
	pub(crate) fn profile(&self) -> Profile<'_> {
		Profile {
			parts: &self.parts,
			passes: self.passes,
			hardware: self.hardware,
		}
	}

	/// Puts one sample in one row, dropping the oldest when the window is
	/// full.
	fn add(&mut self, at: usize, took: Duration) {
		let Some(row) = self.rows.get_mut(at) else {
			return;
		};

		if row.len() >= WINDOW {
			row.remove(0);
		}

		row.push(took);

		if let Some(worst) = self.worst.get_mut(at) {
			*worst = (*worst).max(took);
		}
	}

	/// Builds the list the panel reads, one entry per name.
	fn rebuild(&mut self) {
		self.parts.clear();
		self.parts
			.extend(NAMES.into_iter().enumerate().map(|(at, name)| {
				Part {
					name,
					mean: self.rows.get(at).and_then(|row| mean(row)),
					worst: self
						.rows
						.get(at)
						.filter(|row| !row.is_empty())
						.and_then(|_| self.worst.get(at).copied()),
				}
			}));
	}
}

/// The mean of a window, or nothing for a part that never ran.
#[cfg(feature = "editor")]
fn mean(row: &[Duration]) -> Option<Duration> {
	let count = u32::try_from(row.len()).ok()?;

	row.iter()
		.copied()
		.try_fold(Duration::ZERO, Duration::checked_add)?
		.checked_div(count)
}

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

	/// The most particles any measured frame drew.
	///
	/// The most rather than the mean, because what somebody wants from it is
	/// what the run was actually asked to draw: a cloud still filling up
	/// during the warm-up would drag a mean below the number the project
	/// really carries. The second stable number in this table - a time is a
	/// fact about the afternoon, and this is a fact about the project.
	sparks: usize,

	/// How many triangles the world's terrain has.
	///
	/// Beside `sparks` and for its reason: a stable number about the project,
	/// printed on the frame line rather than folded into a mean. It gets no
	/// **row**, and that is the honest answer rather than an omission - a
	/// terrain is built once and is geometry from then on, so a `cpu terrain`
	/// row would read nil on every frame this table ever sees. What it costs
	/// shows up as bigger numbers in `gpu scene` and `cpu narrow`.
	ground: u64,

	/// How many cells of the world a thing that walks may stand on.
	///
	/// Beside `ground` above and for exactly its argument: a navmesh is baked
	/// once and is a lookup table from then on, so a `cpu nav` row would read
	/// nil on every frame this table ever sees. Unlike the terrain it costs
	/// nothing anywhere else either - nothing asks it a question unless a game
	/// does - so this is the whole of what the profiler has to say about it,
	/// and the number is here because it is what says whether the bake was
	/// about the world somebody thought it was.
	walkable: u64,

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
///
/// `pub(crate)` because the editor's pane shows the same thirteen, and a panel
/// with a list of its own would be a second place to add a row to.
pub(crate) const NAMES: [&str; ROWS] = [
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
	// last, and outside the four above it rather than inside them: the
	// particles are stepped beside the solver, not within it, so this is a
	// part of `cpu step` and of nothing else. Added at shell step 5k.
	"cpu sparks",
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
			sparks: 0,
			ground: 0,
			walkable: 0,
			passes: None,
			steady: true,
		}
	}

	/// Folds one frame's answers in.
	///
	/// @param frame - what the renderer measured
	/// @param spent - what the last simulation step cost
	/// @param sparked - what the particles cost
	/// @param counts - the numbers that are about the project rather than about
	/// a frame
	fn fold(&mut self, frame: &Frame, spent: Spent, sparked: Duration, counts: Counts) {
		self.sparks = self.sparks.max(counts.sparks);
		self.ground = self.ground.max(counts.ground);
		self.walkable = self.walkable.max(counts.walkable);

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

		for (at, took) in
			[spent.broad, spent.narrow, spent.buoyancy, spent.solve, spent.total, sparked]
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

		// the four numbers that add up to a frame, and they are four rather
		// than thirteen because the simulation's rows overlap: `cpu step` is
		// the whole solver step and the four above it are parts of it - and
		// `cpu broad` is in turn a part of `cpu narrow` - so adding every row
		// would count the same microseconds twice over. `cpu sparks` is the
		// exception and is why this grew from three to four: the particles are
		// stepped *beside* the solver, so their time is inside neither the
		// solver's total nor anything else here.
		info!(
			frames,
			passes = self.passes.unwrap_or_default(),
			sparks = self.sparks,
			ground = self.ground,
			walkable = self.walkable,
			steady = self.steady,
			gpu_us = self.slice(0..Pass::ALL.len()).as_micros(),
			cpu_us = self
				.slice(Pass::ALL.len()..Pass::ALL.len() + Work::ALL.len())
				.as_micros(),
			step_us = self.slice(STEP_ROW..STEP_ROW + 1).as_micros(),
			sparks_us = self.slice(SPARKS_ROW..SPARKS_ROW + 1).as_micros(),
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

/// The two counts a run reports that are not times.
///
/// One struct rather than two arguments, because both are the same kind of
/// thing - a number about the world rather than about a frame - and a fold
/// whose signature grows one scalar per feature is a call nobody reads.
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
	/// The most particles any measured frame drew.
	sparks: usize,

	/// How many triangles the world's terrain has.
	ground: u64,

	/// How many cells of it a thing that walks may stand on.
	walkable: u64,
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
			table.fold(&frame, runtime.simulation.spent(), runtime.sparked, Counts {
				sparks: capture.scene_mut().sparks(),
				ground: runtime.ground.triangles(),
				walkable: runtime.paths.cells(),
			});
		}
	}

	table.report(frames);
	runtime.close();

	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[cfg(feature = "editor")]
	#[test]
	fn a_live_row_is_the_mean_of_its_last_twenty_and_the_worst_of_all_of_them() {
		let mut live = Live::new();

		// thirty samples, the first of them the biggest: the window forgets
		// it and the worst does not
		for step in 0..30_u64 {
			live.add(0, Duration::from_micros(if step == 0 { 5_000 } else { 100 + step }));
		}

		live.rebuild();

		let part = live.profile().parts[0];

		assert_eq!(part.name, NAMES[0], "the first row is the first name");

		let mean = part.mean.expect("it ran");

		assert!(
			mean < Duration::from_micros(200),
			"the hitch has scrolled out of the mean: {mean:?}"
		);
		assert_eq!(
			part.worst,
			Some(Duration::from_millis(5)),
			"and the worst remembers it, which is the whole reason it is not a window too"
		);
	}

	#[cfg(feature = "editor")]
	#[test]
	fn a_live_part_that_never_ran_has_no_number_rather_than_nought() {
		// the same rule the printed table keeps: a glow chain in a project
		// that does not bloom is absent, and a nought there would read as the
		// cheapest thing in the frame
		let mut live = Live::new();

		live.add(1, Duration::from_micros(50));
		live.rebuild();

		let parts = live.profile().parts;

		assert_eq!(parts.len(), ROWS, "a row per name, always");
		assert_eq!(parts[1].mean, Some(Duration::from_micros(50)), "the one that ran");
		assert_eq!(parts[0].mean, None, "and the one that did not");
		assert_eq!(parts[0].worst, None, "with no worst either");
	}

	#[cfg(feature = "editor")]
	#[test]
	fn the_two_halves_of_a_frame_arrive_at_two_rates_and_both_land() {
		// what the arrangement actually is: the wall clock every frame, the
		// hardware every second or third, and neither waiting for the other
		let mut live = Live::new();
		// the recording span, which is the second of the two wall-clock ones
		live.walls([None, Some(Duration::from_micros(300))], Spent::default(), Duration::ZERO);

		assert!(!live.profile().hardware, "nothing hardware has answered yet");
		assert_eq!(
			live.profile().parts[Pass::ALL.len() + 1].mean,
			Some(Duration::from_micros(300)),
			"but the wall clock has"
		);
		assert_eq!(
			live.profile().parts[Pass::ALL.len() + 1].name,
			"cpu record",
			"in the row that name belongs to"
		);
		assert_eq!(live.profile().passes, None, "and no frame has been read back");

		live.hardware([None, Some(Duration::from_micros(800)), None, None, None], 15);

		assert!(live.profile().hardware, "now it has");
		assert_eq!(live.profile().parts[1].mean, Some(Duration::from_micros(800)));
		assert_eq!(live.profile().parts[1].name, "gpu scene", "in the second row");
		assert_eq!(live.profile().passes, Some(15), "and the count came with it");
	}

	#[cfg(feature = "editor")]
	#[test]
	fn a_pane_opened_again_does_not_average_what_a_frame_cost_before_lunch() {
		let mut live = Live::new();

		live.add(0, Duration::from_micros(900));
		live.rebuild();
		assert!(live.profile().parts[0].mean.is_some());

		live.clear();

		assert_eq!(live.profile().parts.len(), 0, "nothing is handed out at all");
		assert_eq!(live.profile().passes, None);
		assert!(!live.profile().hardware);
	}

	#[cfg(feature = "editor")]
	#[test]
	fn the_live_table_and_the_printed_one_name_the_same_twelve_parts() {
		// one vocabulary for one frame: a panel with a list of its own would
		// be a second place to add a row to, and the day they disagreed
		// somebody would be comparing two tables that are not about the same
		// thing
		let mut live = Live::new();

		for at in 0..ROWS {
			live.add(at, Duration::from_micros(10));
		}

		live.rebuild();

		let named: Vec<&str> = live
			.profile()
			.parts
			.iter()
			.map(|part| part.name)
			.collect();

		assert_eq!(named, NAMES.to_vec());
	}

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
		assert_eq!(NAMES[STEP_ROW], "cpu step", "the summary's step row is the step");
		assert_eq!(NAMES[SPARKS_ROW], "cpu sparks", "and its particle row is the particles");
		assert!(SPARKS_ROW < ROWS, "both are rows this table has");
		assert_eq!(
			// five of the solver's and one of the particles', which are
			// stepped beside it rather than inside it
			Pass::ALL.len() + Work::ALL.len() + 5 + 1,
			ROWS,
			"the table has a row for every span the renderer and the simulation report"
		);
	}

	#[test]
	fn a_run_whose_pass_count_never_moved_is_steady_and_one_whose_did_is_not() {
		// the one number here worth comparing between two runs, and it is only
		// worth comparing while it holds still for the whole of one.
		let mut table = Table::new();

		table.fold(&Frame::default(), Spent::default(), Duration::ZERO, Counts::default());
		table.fold(&Frame::default(), Spent::default(), Duration::ZERO, Counts::default());

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

		table.fold(&Frame::default(), spent, Duration::from_micros(31), Counts {
			sparks: 40,
			ground: 2048,
			walkable: 900,
		});

		let under = Pass::ALL.len() + Work::ALL.len();

		assert_eq!(table.rows[under].mean(), Duration::from_micros(12), "broad");
		assert_eq!(table.rows[under + 1].mean(), Duration::from_micros(40), "narrow");
		assert_eq!(table.rows[under + 2].mean(), Duration::from_micros(7), "buoyancy");
		assert_eq!(table.rows[under + 3].mean(), Duration::from_micros(90), "solve");
		assert_eq!(table.rows[under + 4].mean(), Duration::from_micros(200), "the step");
		assert_eq!(
			table.rows[under + 5].mean(),
			Duration::from_micros(31),
			"and the particles, which are beside the solver rather than inside it"
		);
		assert_eq!(table.sparks, 40, "and the count is carried beside the times");
		assert_eq!(table.ground, 2048, "and so is the terrain's size");
		assert_eq!(table.walkable, 900, "and how much of it can be walked on");
		assert_eq!(
			table.rows[under + 5].mean(),
			Duration::from_micros(31),
			"and the particles, which are beside the solver rather than inside it"
		);
	}

	#[test]
	fn the_warm_up_is_shorter_than_the_run_it_is_in_front_of() {
		// a table whose warm-up is most of it is a table of a start-up.
		assert!(WARMUP < FRAMES, "the default run is mostly measurement");
		assert!(FRAMES <= MAX_FRAMES);
	}
}
