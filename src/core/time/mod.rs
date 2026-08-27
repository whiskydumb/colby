//! The pace the simulation runs at.
//!
//! The engine simulates at a constant rate and draws whenever it can, which are
//! two different rates. [`Clock`] is what keeps them apart: real time goes in
//! through [`tick`](Clock::tick), whole steps come out through
//! [`step`](Clock::step), and whatever is left over is
//! [`interpolation`](Clock::interpolation) - how far the frame about to be
//! drawn sits between the last two simulated states.
//!
//! This is in `colby_core` rather than in the engine because none of it is
//! graphics: [`STEP_SECONDS`] is what a game reads as
//! [`World::dt`](crate::abi::World::dt), so the step length is part of the
//! host/game contract, and a dedicated server would want the loop without
//! wanting wgpu. It is deliberately *not* under `abi` - a game never sees a
//! `Clock`, only the numbers the host writes into the world.
//!
//! @note: the rate is a constant on purpose. Reading it from a per-project
//! manifest once before the loop starts is a knob that cannot actually be
//! turned; colby's moves into the manifest when there is a CVar system to turn
//! it with.

use std::time::{Duration, Instant};

/// How long one simulation step is, in nanoseconds.
///
/// Sixty a second. `1_000_000_000 / 60` is not an integer, so this is the
/// nearest nanosecond and the simulation gains a third of a nanosecond per
/// step against the wall clock - about a tenth of a second per day, which
/// nothing here can measure.
const STEP_NANOS: u64 = 16_666_667;

/// How long one simulation step is.
pub const STEP: Duration = Duration::from_nanos(STEP_NANOS);

/// The same step, in seconds, which is what a game reads as `World::dt`.
///
/// Written as a literal rather than derived from [`STEP`] because it has to be
/// exactly the number the old variable-timestep screenshots were taken with.
/// The two agree bit for bit; there is a test that says so.
pub const STEP_SECONDS: f32 = 1.0 / 60.0;

/// How many steps of arrears one frame is allowed to carry.
pub const MAX_ARREARS_STEPS: u64 = 4;

/// The most real time one frame may owe the simulation.
///
/// Anything past this is dropped rather than queued, which is the difference
/// between a process that stalls and then runs slow for a moment and a process
/// that stalls and then spends the rest of its life catching up. Because a
/// frame always drains the accumulator below [`STEP`] before it returns, this
/// one clamp is also what bounds the number of steps a frame can run - at most
/// [`MAX_ARREARS_STEPS`] plus the one the remainder pays for. No second cap is
/// needed, and a second cap would only be able to fire where nothing was wrong.
pub const MAX_ARREARS: Duration = Duration::from_nanos(STEP_NANOS * MAX_ARREARS_STEPS);

/// The fastest simulated time is allowed to run against real time.
///
/// Not taste: at a scale of N a frame runs N times as many steps, and there is
/// a number past which asking for more is asking the loop not to return.
pub const MAX_SPEED: f32 = 16.0;

/// Whether the simulation is keeping up with real time.
///
/// Returned by [`Clock::tick`] so the host can say something the first time it
/// falls behind and the first time it recovers, rather than once per frame for
/// as long as a slow machine stays slow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pace {
	/// Real time is being kept.
	Keeping,

	/// This frame owed more than [`MAX_ARREARS`] and the excess was dropped.
	/// The frame before it was keeping up.
	FellBehind,

	/// Still behind, and already reported.
	Behind,

	/// Caught up again after falling behind.
	CaughtUp,
}

/// Real time in, whole simulation steps out.
///
/// The accumulator is a [`Duration`] rather than an `f32` of seconds so that
/// simulated time is exact: nanoseconds are integers, and a float accumulated
/// sixty times a second stops being able to represent the increment long before
/// anyone stops playing.
#[derive(Clone, Copy, Debug)]
pub struct Clock {
	/// When the previous [`tick`](Self::tick) was.
	previous: Instant,

	/// Real time that has arrived but not yet been simulated.
	accumulator: Duration,

	/// Time the simulation has actually run. Behind real time by whatever the
	/// stalls dropped, and by whatever is in the accumulator.
	simulated: Duration,

	/// How fast simulated time runs against real time. One normally, zero
	/// while paused, and whatever `sim.speed` says otherwise.
	speed: f32,

	/// How many frames have been over [`MAX_ARREARS`].
	stalls: u64,

	/// Whether the previous frame was one of them.
	behind: bool,
}

impl Clock {
	/// A clock that starts now, with nothing owed.
	#[must_use]
	pub fn new() -> Self {
		Self {
			previous: Instant::now(),
			accumulator: Duration::ZERO,
			simulated: Duration::ZERO,
			speed: 1.0,
			stalls: 0,
			behind: false,
		}
	}

	/// Forgets the real time since the previous tick, keeping the accumulator.
	///
	/// For the moments the host knows are not the simulation's fault: opening
	/// the window, bringing up the adapter, and swapping the game module. Left
	/// alone, each of those would be measured as a frame that took a second and
	/// charged to gameplay as a stall.
	pub fn reset(&mut self) { self.previous = Instant::now(); }

	/// Sets how fast simulated time runs against real time.
	///
	/// Zero pauses: real time goes on being measured and accounted for, and
	/// none of it turns into steps, so nothing accumulates to be paid back on
	/// the way out. Anything that is not a number is taken as one, because the
	/// value comes from a console and a console takes what it is typed.
	///
	/// @param speed - the scale, clamped into `0.0 ..=` [`MAX_SPEED`]
	pub fn set_speed(&mut self, speed: f32) {
		self.speed = if speed.is_finite() {
			speed.clamp(0.0, MAX_SPEED)
		} else {
			1.0
		};
	}

	/// How fast simulated time is running against real time.
	#[must_use]
	pub const fn speed(&self) -> f32 { self.speed }

	/// Adds simulated time that nobody spent.
	///
	/// The console's single step. Deliberately not subject to the arrears
	/// clamp: this is not time the process fell behind by, it is time someone
	/// asked for, and refusing it would make `sim.step 8` mean four.
	///
	/// @param time - how much simulated time to owe
	pub fn owe(&mut self, time: Duration) {
		self.accumulator = self.accumulator.saturating_add(time);
	}

	/// Adds the real time since the previous call.
	///
	/// @return whether the simulation is keeping up, @ref [`Pace`]
	pub fn tick(&mut self) -> Pace {
		let now = Instant::now();
		let delta = now.duration_since(self.previous);
		self.previous = now;

		self.tick_with(delta)
	}

	/// The same, with the elapsed time supplied rather than measured.
	///
	/// Everything [`tick`](Self::tick) does apart from reading the clock, which
	/// is what makes the pacing testable without sleeping.
	///
	/// @param delta - how much real time to account for
	/// @return whether the simulation is keeping up
	pub fn tick_with(&mut self, delta: Duration) -> Pace {
		// the stall is judged on the *real* delta, before any scaling: falling
		// behind is a property of the process, not of how fast the game was
		// asked to run. Scaling afterwards is what keeps `sim.speed 8` from
		// being capped at the arrears clamp and quietly running at four.
		let over = delta > MAX_ARREARS;
		self.accumulator = self
			.accumulator
			.saturating_add(delta.min(MAX_ARREARS).mul_f32(self.speed));

		if over {
			self.stalls = self.stalls.saturating_add(1);
		}

		let pace = match (self.behind, over) {
			| (false, false) => Pace::Keeping,
			| (false, true) => Pace::FellBehind,
			| (true, true) => Pace::Behind,
			| (true, false) => Pace::CaughtUp,
		};

		self.behind = over;

		pace
	}

	/// Takes one whole step out of the accumulator, if there is one in it.
	///
	/// @return the simulated time after that step, in seconds, or `None` when
	/// the accumulator is short of a whole step
	pub fn step(&mut self) -> Option<f32> {
		if self.accumulator < STEP {
			return None;
		}

		self.accumulator = self.accumulator.saturating_sub(STEP);
		self.simulated = self.simulated.saturating_add(STEP);

		Some(self.simulated.as_secs_f32())
	}

	/// How far the frame about to be drawn sits past the last simulated state.
	///
	/// Zero means draw the state as it was simulated; one would mean draw the
	/// next one, which never happens because a frame drains the accumulator
	/// below a step before it draws.
	#[must_use]
	pub fn interpolation(&self) -> f32 {
		self.accumulator
			.div_duration_f32(STEP)
			.clamp(0.0, 1.0)
	}

	/// How many frames have fallen behind since the clock started.
	#[must_use]
	pub const fn stalls(&self) -> u64 { self.stalls }

	/// Time the simulation has run, in seconds.
	#[must_use]
	pub fn simulated(&self) -> f32 { self.simulated.as_secs_f32() }
}

impl Default for Clock {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Drives a clock with supplied time and counts what it does.
	///
	/// @param clock - the clock to advance
	/// @param delta - how much real time the frame took
	/// @return how many steps that frame ran
	fn frame(clock: &mut Clock, delta: Duration) -> u32 {
		clock.tick_with(delta);

		let mut ran = 0;
		while clock.step().is_some() {
			ran += 1;
		}

		ran
	}

	#[test]
	fn the_step_is_the_same_length_in_both_forms() {
		assert_eq!(
			STEP.as_secs_f32().to_bits(),
			STEP_SECONDS.to_bits(),
			"the Duration and the f32 have to agree, or a screenshot moves"
		);
	}

	#[test]
	fn a_frame_of_one_step_runs_exactly_one_step() {
		let mut clock = Clock::new();

		assert_eq!(frame(&mut clock, STEP), 1, "one step of real time is one step");
		assert!(clock.interpolation() < 1.0e-6, "and leaves nothing over to interpolate");
	}

	#[test]
	fn a_frame_shorter_than_a_step_runs_nothing_and_shows_up_as_interpolation() {
		let mut clock = Clock::new();

		assert_eq!(frame(&mut clock, STEP / 4), 0, "a quarter of a step is not a step");
		assert!(
			(clock.interpolation() - 0.25).abs() < 1.0e-3,
			"but it is a quarter of the way to one, got {}",
			clock.interpolation()
		);
		assert!(clock.simulated() < f32::EPSILON, "and no simulated time has passed");
	}

	#[test]
	fn short_frames_add_up_to_a_step_rather_than_being_lost() {
		let mut clock = Clock::new();
		// @note: three eight-millisecond frames, not four quarters of a step:
		// `Duration` divides by truncation, so four quarters come to three
		// nanoseconds short of a step and this would test the wrong thing.
		let mut ran = 0;
		for _ in 0..3 {
			ran += frame(&mut clock, Duration::from_millis(8));
		}

		assert_eq!(ran, 1, "the remainder carries between frames rather than being dropped");
		assert!(clock.interpolation() > 0.0, "and what is left over is still there");
	}

	#[test]
	fn a_long_frame_catches_up_in_one_go() {
		let mut clock = Clock::new();

		assert_eq!(frame(&mut clock, STEP * 3), 3, "three steps of arrears, three steps");
	}

	#[test]
	fn a_stalled_frame_drops_the_arrears_instead_of_queuing_them() {
		let mut clock = Clock::new();
		let ran = frame(&mut clock, Duration::from_secs(10));

		assert!(
			u64::from(ran) <= MAX_ARREARS_STEPS + 1,
			"ten seconds of stall must not become six hundred steps, got {ran}"
		);
		assert_eq!(clock.stalls(), 1, "and it is counted rather than hidden");
		assert!(
			clock.simulated() < 0.2,
			"the dropped time is dropped: the simulation runs slow, it does not spiral"
		);
	}

	#[test]
	fn a_stall_is_reported_once_going_in_and_once_coming_out() {
		let mut clock = Clock::new();

		assert_eq!(clock.tick_with(STEP), Pace::Keeping, "a normal frame says nothing");
		assert_eq!(
			clock.tick_with(Duration::from_secs(2)),
			Pace::FellBehind,
			"the first slow frame is the one worth a log line"
		);
		assert_eq!(
			clock.tick_with(Duration::from_secs(2)),
			Pace::Behind,
			"the next hundred are not"
		);
		assert_eq!(clock.tick_with(STEP), Pace::CaughtUp, "and recovery is worth one more");
		assert_eq!(clock.tick_with(STEP), Pace::Keeping, "then quiet again");
	}

	#[test]
	fn a_stall_never_leaves_more_than_a_frames_worth_of_arrears() {
		let mut clock = Clock::new();
		clock.tick_with(Duration::from_secs(30));

		let mut ran = 0;
		while clock.step().is_some() {
			ran += 1;
		}

		assert!(ran <= MAX_ARREARS_STEPS + 1, "the clamp is what bounds the loop, got {ran}");
		assert!(clock.interpolation() >= 0.0, "and what is left is a fraction of a step");
		assert!(clock.interpolation() <= 1.0, "never more than one");
	}

	#[test]
	fn interpolation_stays_inside_its_range_however_the_frames_land() {
		let mut clock = Clock::new();

		for length in [1_u32, 3, 7, 16, 17, 33, 200, 1] {
			frame(&mut clock, Duration::from_millis(u64::from(length)));

			let interpolation = clock.interpolation();
			assert!(
				(0.0..=1.0).contains(&interpolation),
				"a {length}ms frame left interpolation at {interpolation}"
			);
		}
	}

	#[test]
	fn the_moment_being_drawn_keeps_pace_with_real_time_whatever_the_frames_do() {
		let mut clock = Clock::new();
		let mut real = Duration::ZERO;

		// deliberately uneven, and deliberately all under MAX_ARREARS: frames
		// well above the step rate, frames well below it, and frames that land
		// on it. Nothing here is a stall, so nothing here is allowed to lose
		// time.
		for length in [4_u64, 4, 4, 16, 16, 40, 7, 3, 3, 33, 16, 16, 9] {
			let delta = Duration::from_millis(length);
			real = real.saturating_add(delta);
			frame(&mut clock, delta);

			// where the renderer is asked to draw: the last simulated state,
			// plus the fraction of a step this frame sits past it. The *pose*
			// it draws is one step behind that, because it blends the two
			// states either side of the moment - and one step behind, moving
			// at exactly the speed of real time, is what smooth means here.
			let drawn = STEP_SECONDS.mul_add(clock.interpolation(), clock.simulated());

			assert!(
				(drawn - real.as_secs_f32()).abs() < 1.0e-4,
				"a {length}ms frame put the drawn moment at {drawn} with {} of real time 				 \
				 gone: the two have to move together, or motion is only smooth at one frame 				 \
				 rate",
				real.as_secs_f32()
			);
		}
	}

	#[test]
	fn a_paused_clock_measures_time_without_simulating_it() {
		let mut clock = Clock::new();
		clock.set_speed(0.0);

		assert_eq!(frame(&mut clock, STEP * 10), 0, "nothing runs while the speed is zero");
		assert!(
			clock.interpolation() < f32::EPSILON,
			"and nothing is owed on the way out, or unpausing would lurch"
		);
	}

	#[test]
	fn a_scaled_clock_runs_that_many_steps() {
		let mut half = Clock::new();
		half.set_speed(0.5);

		let mut double = Clock::new();
		double.set_speed(2.0);

		// four steps of real time is exactly the arrears clamp, so this also
		// says the clamp does not cap the scale.
		assert_eq!(frame(&mut half, STEP * 4), 2, "half speed, half the steps");
		assert_eq!(frame(&mut double, STEP * 4), 8, "double speed, twice as many");
	}

	#[test]
	fn a_speed_is_taken_as_typed_or_not_at_all() {
		let mut clock = Clock::new();

		clock.set_speed(-3.0);
		assert!(clock.speed() < f32::EPSILON, "backwards is paused, not backwards");

		clock.set_speed(1.0e9);
		assert!(
			(clock.speed() - MAX_SPEED).abs() < f32::EPSILON,
			"and there is a ceiling on how much work one frame can be asked for"
		);

		clock.set_speed(f32::NAN);
		assert!(
			(clock.speed() - 1.0).abs() < f32::EPSILON,
			"something that is not a number is real time, not a frozen loop"
		);
	}

	#[test]
	fn an_owed_step_runs_without_any_time_passing() {
		let mut clock = Clock::new();
		clock.set_speed(0.0);
		clock.tick_with(STEP * 10);

		clock.owe(STEP * 3);

		let mut ran = 0;
		while clock.step().is_some() {
			ran += 1;
		}

		assert_eq!(ran, 3, "three steps asked for, three steps, paused or not");
	}

	#[test]
	fn simulated_time_does_not_drift_over_a_long_run() {
		let mut clock = Clock::new();
		for _ in 0..10_000 {
			frame(&mut clock, STEP);
		}

		let expected = (STEP * 10_000).as_secs_f32();

		assert!(
			(clock.simulated() - expected).abs() < 1.0e-4,
			"ten thousand steps of exact time is exact time, got {}",
			clock.simulated()
		);
	}

	#[test]
	fn resetting_forgets_the_stall_without_forgetting_the_remainder() {
		let mut clock = Clock::new();
		clock.tick_with(STEP / 2);
		clock.reset();

		assert!(
			(clock.interpolation() - 0.5).abs() < 1.0e-3,
			"a reset is about the real clock, not about what is already owed"
		);
	}
}
