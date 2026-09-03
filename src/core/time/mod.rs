//! The pace the simulation runs at.
//!
//! The engine simulates at a fixed rate and draws whenever it can, which are
//! two different rates. [`Clock`] is what keeps them apart: real time goes in
//! through [`tick`](Clock::tick), whole steps come out through
//! [`step`](Clock::step), and whatever is left over is
//! [`interpolation`](Clock::interpolation) - how far the frame about to be
//! drawn sits between the last two simulated states.
//!
//! This is in `colby_core` rather than in the engine because none of it is
//! graphics: a [`Rate`]'s [`seconds`](Rate::seconds) is what a game reads as
//! [`World::dt`](crate::abi::World::dt), so the step length is part of the
//! host/game contract, and a dedicated server would want the loop without
//! wanting wgpu. It is deliberately *not* under `abi` - a game never sees a
//! `Clock`, only the numbers the host writes into the world.
//!
//! @note: the rate is a number somebody can turn, and it is turned while the
//! process runs rather than only before it starts. [`Rate`] is what carries it
//! and [`Clock::set_rate`] is how it moves; the constants below are only the
//! value nobody has said anything about.

use std::time::{Duration, Instant};

/// How many nanoseconds a second holds.
const NANOS_A_SECOND: u64 = 1_000_000_000;

/// How long one simulation step is, in nanoseconds.
///
/// Sixty a second. `1_000_000_000 / 60` is not an integer, so this is the
/// nearest nanosecond and the simulation gains a third of a nanosecond per
/// step against the wall clock - about a tenth of a second per day, which
/// nothing here can measure.
const STEP_NANOS: u64 = 16_666_667;

/// How long one simulation step is, unless somebody says otherwise.
pub const STEP: Duration = Duration::from_nanos(STEP_NANOS);

/// The same step, in seconds, which is what a game reads as `World::dt`.
///
/// Written as a literal rather than derived from [`STEP`] because it has to be
/// exactly the number the old variable-timestep screenshots were taken with.
/// The two agree bit for bit; there is a test that says so.
pub const STEP_SECONDS: f32 = 1.0 / 60.0;

/// How many steps of arrears one frame is allowed to carry.
pub const MAX_ARREARS_STEPS: u32 = 4;

/// The fastest simulated time is allowed to run against real time.
///
/// Not taste: at a scale of N a frame runs N times as many steps, and there is
/// a number past which asking for more is asking the loop not to return.
pub const MAX_SPEED: f32 = 16.0;

/// How long a simulation step is, in the two forms anybody wants it in.
///
/// One value rather than a number passed around, because the three fields have
/// to agree and there is exactly one arithmetic that makes them agree. A rate
/// is built from a whole number of hertz or it is [`DEFAULT`](Self::DEFAULT);
/// there is no third way to make one, so the invariant holds by construction.
///
/// It is `Copy` and tiny on purpose: the host hands it to every simulation
/// step, and a step that had to reach for it would be a step with an opinion
/// about where it is kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rate {
	/// Steps a second, which is the number a person types.
	hz: u16,

	/// How long one of them is.
	step: Duration,

	/// The same, in seconds, which is what a game reads as `World::dt`.
	seconds: f32,
}

impl Rate {
	/// Sixty a second: the rate everything in this engine was tuned at.
	///
	/// Written out of the two constants rather than through
	/// [`from_hz`](Self::from_hz) so that it is a `const`. The two agree bit
	/// for bit and there is a test that says so.
	pub const DEFAULT: Self = Self {
		hz: 60,
		step: STEP,
		seconds: STEP_SECONDS,
	};
	/// The fastest.
	///
	/// Taste rather than safety, and the safety is what makes taste enough:
	/// the arrears clamp bounds how many steps a frame may run, so a rate too
	/// high for the machine makes simulated time fall behind real time rather
	/// than making the loop stop returning. A thousand is where a step is a
	/// round million nanoseconds and asking for more stops meaning anything on
	/// a clock this process can read.
	pub const MAX_HZ: u16 = 1000;
	/// The slowest rate that can be asked for.
	///
	/// Under one a step is longer than the whole arrears window, so a frame
	/// could owe time it is never allowed to pay. Zero would be a division by
	/// it.
	pub const MIN_HZ: u16 = 1;

	/// The rate that runs this many steps a second.
	///
	/// The step is rounded to the nearest nanosecond rather than truncated,
	/// which is what makes sixty come out at exactly [`STEP`] rather than one
	/// nanosecond short of it.
	///
	/// @param hz - steps a second, clamped into the range from
	/// [`MIN_HZ`](Self::MIN_HZ) to [`MAX_HZ`](Self::MAX_HZ)
	/// @return the rate, with its three fields in agreement
	#[must_use]
	pub fn from_hz(hz: u16) -> Self {
		let hz = hz.clamp(Self::MIN_HZ, Self::MAX_HZ);
		let divisor = u64::from(hz);

		Self {
			hz,
			step: Duration::from_nanos((NANOS_A_SECOND + divisor / 2) / divisor),
			seconds: 1.0 / f32::from(hz),
		}
	}

	/// How many steps a second this is.
	#[must_use]
	pub const fn hz(self) -> u16 { self.hz }

	/// How long one step is.
	#[must_use]
	pub const fn step(self) -> Duration { self.step }

	/// The same, in seconds, which is what the host writes into `World::dt`.
	#[must_use]
	pub const fn seconds(self) -> f32 { self.seconds }

	/// The most real time one frame may owe the simulation.
	///
	/// Anything past this is dropped rather than queued, which is the
	/// difference between a process that stalls and then runs slow for a
	/// moment and a process that stalls and then spends the rest of its life
	/// catching up. Because a frame always drains the accumulator below one
	/// step before it returns, this one clamp is also what bounds the number
	/// of steps a frame can run - at most [`MAX_ARREARS_STEPS`] plus the one
	/// the remainder pays for. No second cap is needed, and a second cap would
	/// only be able to fire where nothing was wrong.
	#[must_use]
	pub const fn arrears(self) -> Duration { self.step.saturating_mul(MAX_ARREARS_STEPS) }
}

impl Default for Rate {
	fn default() -> Self { Self::DEFAULT }
}

/// Whether the simulation is keeping up with real time.
///
/// Returned by [`Clock::tick`] so the host can say something the first time it
/// falls behind and the first time it recovers, rather than once per frame for
/// as long as a slow machine stays slow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pace {
	/// Real time is being kept.
	Keeping,

	/// This frame owed more than [`Rate::arrears`] and the excess was dropped.
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

	/// How long one step is, and how much may be owed before it is dropped.
	rate: Rate,

	/// Time the simulation has actually run. Behind real time by whatever the
	/// stalls dropped, and by whatever is in the accumulator.
	simulated: Duration,

	/// How fast simulated time runs against real time. One normally, zero
	/// while paused, and whatever `sim.speed` says otherwise.
	speed: f32,

	/// How many frames have been over [`Rate::arrears`].
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
			rate: Rate::DEFAULT,
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

	/// Sets how long a step is from here on.
	///
	/// **Nothing is converted, and that is the whole reason this is safe to
	/// call between two steps.** The accumulator holds *real* time that has
	/// arrived and not been simulated, which is a quantity with no step length
	/// in it, and `simulated` holds elapsed simulated time, which is a
	/// quantity with no step count in it. Neither means anything different
	/// after the rate moves; only how much of the first one buys a step does.
	///
	/// @param rate - the new step length
	pub const fn set_rate(&mut self, rate: Rate) { self.rate = rate; }

	/// How long a step is on this clock.
	#[must_use]
	pub const fn rate(&self) -> Rate { self.rate }

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
		let arrears = self.rate.arrears();
		let over = delta > arrears;
		self.accumulator = self
			.accumulator
			.saturating_add(delta.min(arrears).mul_f32(self.speed));

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
		let step = self.rate.step;

		if self.accumulator < step {
			return None;
		}

		self.accumulator = self.accumulator.saturating_sub(step);
		self.simulated = self.simulated.saturating_add(step);

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
			.div_duration_f32(self.rate.step)
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

	/// Takes every whole step a clock is already holding, without ticking it.
	///
	/// @param clock - the clock to drain
	/// @return how many steps came out
	fn drained(clock: &mut Clock) -> u32 {
		let mut ran = 0;
		while clock.step().is_some() {
			ran += 1;
		}

		ran
	}

	#[test]
	fn sixty_hertz_is_the_default_rate_bit_for_bit() {
		let built = Rate::from_hz(60);

		// the whole reason `from_hz` rounds rather than truncates. A step one
		// nanosecond short of `STEP` would move every screenshot this project
		// has ever taken, and it would move them by an amount too small to see
		// and too large to ignore.
		assert_eq!(built.step(), Rate::DEFAULT.step(), "the same step");
		assert_eq!(
			built.seconds().to_bits(),
			Rate::DEFAULT.seconds().to_bits(),
			"and the same number of seconds, to the bit"
		);
		assert_eq!(built.hz(), 60, "and it remembers what it was asked for");
	}

	#[test]
	fn a_rate_agrees_with_itself_in_both_forms() {
		for hz in [1_u16, 20, 30, 50, 60, 64, 120, 128, 240, 1000] {
			let rate = Rate::from_hz(hz);
			let apart = (rate.step().as_secs_f32() - rate.seconds()).abs();
			// half a nanosecond of rounding in the step, and one f32 ulp in
			// the seconds. Anything wider than that is the two being computed
			// from different numbers rather than from the same one.
			let allowed = rate.seconds().mul_add(f32::EPSILON, 1.0e-9);

			assert!(
				apart <= allowed,
				"{hz} hertz: a step of {:?} against {} seconds",
				rate.step(),
				rate.seconds()
			);
		}
	}

	#[test]
	fn a_rate_outside_the_range_is_pulled_back_into_it() {
		assert_eq!(Rate::from_hz(0).hz(), Rate::MIN_HZ, "nothing divides by zero");
		assert_eq!(
			Rate::from_hz(u16::MAX).hz(),
			Rate::MAX_HZ,
			"and nothing runs at sixty-five thousand"
		);
	}

	#[test]
	fn a_faster_rate_runs_more_steps_for_the_same_real_time() {
		// a hundred and twenty five and two hundred and fifty rather than
		// sixty and a hundred and twenty, because the steps of those two are
		// whole milliseconds: the count then says something about the rate
		// rather than about which side of a nanosecond the rounding fell.
		let mut slow = Clock::new();
		let mut fast = Clock::new();
		slow.set_rate(Rate::from_hz(125));
		fast.set_rate(Rate::from_hz(250));

		let mut slow_ran = 0;
		let mut fast_ran = 0;
		for _ in 0..250 {
			slow_ran += frame(&mut slow, Duration::from_millis(4));
			fast_ran += frame(&mut fast, Duration::from_millis(4));
		}

		assert_eq!(slow_ran, 125, "a second at a hundred and twenty five is that many steps");
		assert_eq!(fast_ran, 250, "and the same second at twice the rate is twice as many");
	}

	#[test]
	fn the_blend_between_two_steps_is_a_fraction_of_the_step_it_is_between() {
		// what the renderer draws with, and it is a fraction rather than a
		// duration - so a clock that measured it against a step it is not
		// running would hand the renderer a number between nought and one that
		// meant something else entirely, and the picture would lag or lead
		// with nothing anywhere reporting a fault.
		let mut clock = Clock::new();
		let fast = Rate::from_hz(240);
		clock.set_rate(fast);

		clock.tick_with(fast.step() / 2);

		let blend = clock.interpolation();
		assert!(
			(blend - 0.5).abs() < 1.0e-3,
			"half a step of the rate being run is half way between two of them, got {blend}"
		);
	}

	#[test]
	fn what_a_stalled_frame_may_owe_is_measured_against_the_rate_being_run() {
		// four steps of arrears at two hundred and forty is a sixtieth of a
		// second, not the fifteenth it would be at sixty. A clock that clamped
		// against the wrong one would take a frame that really did stall,
		// count it as keeping up, and run seven steps to catch up on it.
		let mut clock = Clock::new();
		let fast = Rate::from_hz(240);
		clock.set_rate(fast);

		let pace = clock.tick_with(fast.step() * 8);

		assert_eq!(pace, Pace::FellBehind, "eight steps of arrears is a stall at this rate");
		assert_eq!(clock.stalls(), 1, "and it is counted");
		assert_eq!(
			drained(&mut clock),
			MAX_ARREARS_STEPS,
			"and what is left to run is the window rather than what arrived"
		);
	}

	#[test]
	fn the_arrears_window_follows_the_rate() {
		let slow = Rate::from_hz(30);
		let fast = Rate::from_hz(120);

		assert_eq!(
			slow.arrears(),
			slow.step() * MAX_ARREARS_STEPS,
			"four steps, whatever a step is"
		);
		assert!(fast.arrears() < slow.arrears(), "and a shorter step is a shorter window");
	}

	#[test]
	fn changing_the_rate_neither_loses_nor_duplicates_what_is_owed() {
		let mut clock = Clock::new();

		// three quarters of a step at sixty: not enough to run, and the whole
		// of what the clock is holding.
		clock.tick_with(STEP * 3 / 4);
		assert!(clock.step().is_none(), "three quarters of a step is not a step");

		// the same real time is a step and a half at twice the rate, so
		// exactly one step comes out and a quarter of the old one is still
		// owed. If the accumulator were cleared, or measured in steps rather
		// than in time, this would be nothing at all.
		clock.set_rate(Rate::from_hz(120));
		assert!(clock.step().is_some(), "what was owed is still owed, and now buys a step");
		assert!(clock.step().is_none(), "but only the one");

		assert!(
			clock.interpolation() > 0.0,
			"and the remainder is still there, as a fraction of a shorter step"
		);
	}

	#[test]
	fn simulated_time_carries_across_a_rate_change() {
		let mut clock = Clock::new();

		for _ in 0..60 {
			frame(&mut clock, STEP);
		}

		let before = clock.simulated();
		clock.set_rate(Rate::from_hz(240));

		// to the bit on purpose: turning the rate must not touch this number
		// at all, and a tolerance here would let a small restatement through.
		assert_eq!(
			clock.simulated().to_bits(),
			before.to_bits(),
			"the seconds already run are not restated"
		);

		for _ in 0..240 {
			frame(&mut clock, Rate::from_hz(240).step());
		}

		// one second at sixty and one at two hundred and forty is two seconds,
		// and it is two seconds because the clock counts time rather than
		// steps.
		assert!(
			(clock.simulated() - 2.0).abs() < 1.0e-3,
			"two seconds of simulated time, got {}",
			clock.simulated()
		);
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
			ran <= MAX_ARREARS_STEPS + 1,
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

		// deliberately uneven, and deliberately all under the arrears cap: frames
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
