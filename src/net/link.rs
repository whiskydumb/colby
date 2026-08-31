//! A link that lies: the wire between two endpoints, made as bad as somebody
//! wants it.
//!
//! ```text
//!   link.receive(datagram, now)          a datagram off the socket goes in
//!   while let Some(d) = link.poll(now)   whatever is due comes out
//! ```
//!
//! **Only the receiving side lies**, which is what every simulator in this
//! field does and is worth saying why: a datagram that is delayed on the way
//! out and a datagram that is delayed on the way in are the same datagram, and
//! doing both halves at once means every number has to be read as half of
//! itself. So a round trip of a hundred milliseconds is a link at each end
//! holding fifty, and each endpoint's own numbers describe what *it* sees.
//!
//! **The same seed says the same thing.** Nothing here reads a clock or asks
//! the operating system for anything; time arrives as a [`Duration`] the caller
//! measured and chance arrives from a shift register somebody seeded. Two runs
//! of the same steps produce the same deliveries in the same order at the same
//! moments, which is the whole reason this is worth building rather than
//! borrowing - the libraries in this field all seed from the operating system,
//! so none of them can repeat a run, and a networking change with no repeatable
//! run is a change nobody can review.
//!
//! The draw order is part of that contract, so it is written down: per arriving
//! datagram, first whether the link changes state, then whether the datagram is
//! lost, then how long it is held, then whether it arrives twice - and one more
//! hold if it does. A datagram that is lost stops after the second draw.
//!
//! **Loss is two states rather than one probability**, so that what is lost
//! comes in runs the way it does on a real link. The chain is the standard one:
//! a good state and a bad state, a probability of moving between them each way,
//! and everything that arrives while it is bad is lost. What crosses from the
//! console is not those probabilities but the two numbers a person can hold -
//! [`LOSS`], the share that never arrives, and [`BURST`], how many times longer
//! a run of losses is than chance alone would make it. Both transitions are
//! then that share divided by the burst, which is the whole derivation: **the
//! burst stretches the timescale of the chain and leaves the share exactly
//! where it was.**
//!
//! @note: the chain only moves when a datagram arrives. A burst is measured in
//! datagrams, not in seconds, so a link left quiet is still in whatever state
//! the last datagram left it in - and a very large burst on a short run may
//! never enter the bad state at all, which reads as a link that is not lossy.
//! Past a burst of a few thousand million the way in rounds to nothing and it
//! never will, whatever the run length; past the same on the way out, the bad
//! state is where it stays. Neither is a burst anybody types.

use std::time::Duration;

use crate::random::{CERTAIN, Random, threshold};

/// The variable that sets how long a datagram is held.
pub const LAG: &str = "net.lag";

/// The variable that sets how much that hold varies either way.
pub const JITTER: &str = "net.jitter";

/// The variable that sets the share of datagrams that never arrive.
pub const LOSS: &str = "net.loss";

/// The variable that sets how clumped those losses are.
pub const BURST: &str = "net.burst";

/// The variable that sets the share of datagrams that arrive twice.
pub const DUPLICATE: &str = "net.duplicate";

/// How bad the link is.
///
/// Everything here is what a person writes; what the draws actually compare
/// against is worked out from it once, when it is set. @ref
/// [`Link::set`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conditions {
	/// How long a datagram is held before it is handed on, one way.
	pub lag: Duration,

	/// How far either side of [`lag`](Self::lag) that hold varies.
	///
	/// This is also the only thing that reorders anything: two datagrams a
	/// moment apart can be held for different lengths, and the second can come
	/// out first. The furthest apart two of them can be moved is twice this -
	/// unless it is larger than the lag, in which case everything that would
	/// have been held for less than no time is handed over at once instead, and
	/// the widest gap is the lag plus this. @ref the note on the hold.
	pub jitter: Duration,

	/// The share of datagrams that never arrive, nil through one.
	pub loss: f32,

	/// How many times longer a run of losses is than chance alone makes it.
	///
	/// One is chance alone, which still comes in runs - a link losing a quarter
	/// of everything loses two in a row one time in four. Four is runs four
	/// times that long with the same share lost overall.
	pub burst: f32,

	/// The share of datagrams that arrive twice, nil through one.
	///
	/// The second copy is held for its own length, so with any jitter at all it
	/// arrives apart from the first. With none it is held for exactly as long
	/// and the two come out together, which is what a link with only this set
	/// does. A datagram that was lost is not duplicated: there is nothing left
	/// of it to copy.
	pub duplicate: f32,
}

impl Default for Conditions {
	fn default() -> Self { Self::PERFECT }
}

impl Conditions {
	/// A link that does nothing at all to what crosses it.
	pub const PERFECT: Self = Self {
		lag: Duration::ZERO,
		jitter: Duration::ZERO,
		loss: 0.0,
		burst: 1.0,
		duplicate: 0.0,
	};

	/// Whether this link would leave everything exactly as it found it.
	#[must_use]
	pub fn is_perfect(&self) -> bool { *self == Self::PERFECT }
}

/// Which of the two states the loss chain is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
	/// Nothing is being lost.
	Good,

	/// Everything is.
	Bad,
}

impl State {
	/// The other one.
	const fn other(self) -> Self {
		match self {
			| Self::Good => Self::Bad,
			| Self::Bad => Self::Good,
		}
	}
}

/// What the draws compare against, worked out from [`Conditions`] once.
#[derive(Clone, Copy, Debug, Default)]
struct Chances {
	good_to_bad: u64,
	bad_to_good: u64,
	good_loss: u64,
	bad_loss: u64,
	duplicate: u64,
}

/// A datagram waiting for its moment.
#[derive(Clone, Debug)]
struct Held {
	due: Duration,
	order: u64,
	bytes: Vec<u8>,
}

/// One direction of a wire, with whatever is wrong with it.
#[derive(Clone, Debug)]
pub struct Link {
	conditions: Conditions,
	chances: Chances,
	state: State,
	random: Random,
	held: Vec<Held>,
	spare: Vec<Vec<u8>>,
	ready: Vec<u8>,
	arrivals: u64,
	received: u32,
	dropped: u32,
	duplicated: u32,
	delivered: u32,
}

impl Link {
	/// A link that does nothing to what crosses it, ready to be told otherwise.
	///
	/// @param seed - what the chance in it starts from; @ref
	/// [`Random::new`](crate::random::Random::new)
	#[must_use]
	pub fn new(seed: u64) -> Self {
		Self {
			conditions: Conditions::PERFECT,
			chances: chances(&Conditions::PERFECT),
			state: State::Good,
			random: Random::new(seed),
			held: Vec::new(),
			spare: Vec::new(),
			ready: Vec::new(),
			arrivals: 0,
			received: 0,
			dropped: 0,
			duplicated: 0,
			delivered: 0,
		}
	}

	/// Says how bad the link is from now on.
	///
	/// Whatever is already being held is left alone: it was held for as long as
	/// the conditions at the time said, and rewriting that would make a change
	/// to the numbers reach backwards. **Which state the loss chain is in is
	/// left alone as well**, so a link changed in the middle of a run of losses
	/// is still in the middle of one - which is why a share of nil has to mean
	/// nothing lost in *either* state rather than only in the good one.
	///
	/// @param conditions - what a person wrote
	pub fn set(&mut self, conditions: Conditions) {
		self.conditions = conditions;
		self.chances = chances(&conditions);
	}

	/// How bad the link is.
	#[must_use]
	pub const fn conditions(&self) -> &Conditions { &self.conditions }

	/// Takes a datagram off the wire and decides what becomes of it.
	///
	/// @param datagram - the bytes as they arrived
	/// @param now - how long this endpoint has been running
	pub fn receive(&mut self, datagram: &[u8], now: Duration) {
		self.received = self.received.saturating_add(1);

		let (loss, transition) = match self.state {
			| State::Good => (self.chances.good_loss, self.chances.good_to_bad),
			| State::Bad => (self.chances.bad_loss, self.chances.bad_to_good),
		};

		// the state is stepped first and the loss is drawn against the state
		// this datagram *found*, so a change takes effect on the one after it.
		//
		// @note: the second draw does no work today and is kept deliberately. A
		// state's loss is either nil or certain, so the outcome follows from
		// the state alone and one shared draw would give the same answer for
		// every seed. It is two because the standard form of this model has a
		// small residual loss in the good state as a fourth number, and the day
		// that is added the two decisions have to be independent. Until then it
		// advances the stream and keeps the draw count the module doc promises.
		if self.random.chance(transition) {
			self.state = self.state.other();
		}

		if self.random.chance(loss) {
			self.dropped = self.dropped.saturating_add(1);
			return;
		}

		self.hold(datagram, now);

		if self.random.chance(self.chances.duplicate) {
			self.duplicated = self.duplicated.saturating_add(1);
			self.hold(datagram, now);
		}
	}

	/// The next datagram that is due, if one is.
	///
	/// Call it until it says nothing. What comes out is in order of when it
	/// became due, and two that became due at the same moment come out in the
	/// order they arrived - which is the difference between a link that can be
	/// hashed and one that cannot.
	///
	/// @param now - how long this endpoint has been running
	pub fn poll(&mut self, now: Duration) -> Option<&[u8]> {
		let index = self.due(now)?;
		let held = self.held.swap_remove(index);
		let previous = std::mem::replace(&mut self.ready, held.bytes);

		self.spare.push(previous);
		self.delivered = self.delivered.saturating_add(1);
		Some(&self.ready)
	}

	/// How many datagrams are being held.
	#[must_use]
	pub fn waiting(&self) -> usize { self.held.len() }

	/// How many datagrams have arrived.
	#[must_use]
	pub const fn received(&self) -> u32 { self.received }

	/// How many of them were thrown away.
	#[must_use]
	pub const fn dropped(&self) -> u32 { self.dropped }

	/// How many second copies were made.
	#[must_use]
	pub const fn duplicated(&self) -> u32 { self.duplicated }

	/// How many datagrams have been handed on.
	#[must_use]
	pub const fn delivered(&self) -> u32 { self.delivered }

	/// Which held datagram is next, by when it came due and then by when it
	/// arrived.
	fn due(&self, now: Duration) -> Option<usize> {
		let mut best: Option<usize> = None;

		for (index, held) in self.held.iter().enumerate() {
			if held.due > now {
				continue;
			}

			let sooner = best.is_none_or(|at| {
				(held.due, held.order) < (self.held[at].due, self.held[at].order)
			});

			if sooner {
				best = Some(index);
			}
		}

		best
	}

	/// Puts one copy of a datagram aside until its moment.
	fn hold(&mut self, datagram: &[u8], now: Duration) {
		let due = now.saturating_add(self.wait());
		let mut bytes = self.spare.pop().unwrap_or_default();

		bytes.clear();
		bytes.extend_from_slice(datagram);
		self.arrivals = self.arrivals.saturating_add(1);
		self.held
			.push(Held { due, order: self.arrivals, bytes });
	}

	/// How long this copy is held for.
	///
	/// @note: a draw is taken even when there is no jitter to spend it on, so
	/// that the number of draws a datagram costs depends on what became of it
	/// and not on what the conditions happen to be. That is what makes the draw
	/// order in the module doc a contract rather than a description.
	fn wait(&mut self) -> Duration {
		let lag = nanos(self.conditions.lag);
		let jitter = nanos(self.conditions.jitter);

		// anywhere in the span from a jitter early to a jitter late, both ends
		// included - so the span is odd and the middle of it is no offset at
		// all. With no jitter the span is one and the offset is always nil.
		let span = jitter.saturating_mul(2).saturating_add(1);
		let offset = self.random.below(span);

		// @note: a hold that would come out negative is no hold instead, and
		// what that costs is worth writing down rather than discovering. It
		// only happens when the jitter is larger than the lag, and then the
		// whole early part of the span lands on the same moment: at no lag and
		// ten milliseconds of jitter, half the datagrams are handed over at
		// once, the average hold is a quarter of the jitter rather than the
		// nothing that was asked for, and the widest two of them can be parted
		// is the jitter rather than twice it. The alternative is handing a
		// datagram over before it arrived, which is not a thing a wire does.
		Duration::from_nanos(lag.saturating_add(offset).saturating_sub(jitter))
	}
}

/// What the draws compare against, worked out from what a person wrote.
///
/// The share lost is the share of the time the chain spends in the bad state,
/// and the burst divides both ways out of it - which stretches how long each
/// visit lasts without touching how much of the time is spent there. So the two
/// numbers a person writes do not fight each other, which is the whole reason
/// they are the two that cross.
fn chances(conditions: &Conditions) -> Chances {
	// a share that is not a number reads as nothing rather than falling through
	// both ends of the comparisons below - which it would, since neither
	// ordering holds against one, and it would land in the general derivation
	// with both ways out of the chain rounded to nil and the bad state's loss
	// certain. A link that happened to be in the bad state would then drop
	// everything for the rest of its life, and one in the good state would lose
	// nothing at all, from the same numbers.
	let loss = if conditions.loss.is_nan() {
		0.0
	} else {
		conditions.loss.clamp(0.0, 1.0)
	};

	// `max` hands back the other side when one of them is not a number, so this
	// is the same guard for the burst.
	let burst = conditions.burst.max(1.0);
	let duplicate = threshold(conditions.duplicate);

	// nothing is lost in either state, and this is *not* a short circuit for
	// the derivation below. The state of the chain survives a change of
	// conditions, so a link somebody has just told to stop losing may well be
	// in the bad state - where the derivation would put a certain loss, and it
	// would go on dropping everything until the chain happened to leave, which
	// at a large burst is a very long time.
	if loss <= 0.0 {
		return Chances { duplicate, ..Chances::default() };
	}

	// everything is lost in either state, and the chain would otherwise have to
	// reach the bad state before any of it was.
	if loss >= 1.0 {
		return Chances {
			good_loss: CERTAIN,
			bad_loss: CERTAIN,
			duplicate,
			..Chances::default()
		};
	}

	Chances {
		good_to_bad: threshold(loss / burst),
		bad_to_good: threshold((1.0 - loss) / burst),
		good_loss: 0,
		bad_loss: CERTAIN,
		duplicate,
	}
}

/// A span as whole nanoseconds, saturating rather than wrapping.
fn nanos(span: Duration) -> u64 { u64::try_from(span.as_nanos()).unwrap_or(u64::MAX) }

#[cfg(test)]
mod tests {
	use super::*;

	fn at(millis: u64) -> Duration { Duration::from_millis(millis) }

	/// Hands one datagram in and takes out everything that is due.
	fn cross(link: &mut Link, datagram: &[u8], now: Duration) -> Vec<Vec<u8>> {
		link.receive(datagram, now);
		drain(link, now)
	}

	/// Takes out everything that is due.
	fn drain(link: &mut Link, now: Duration) -> Vec<Vec<u8>> {
		let mut out = Vec::new();

		while let Some(datagram) = link.poll(now) {
			out.push(datagram.to_vec());
		}

		out
	}

	/// Every delivery over a run, with the moment it happened and all of it.
	///
	/// A tenth of a millisecond at a time and the whole payload, which is what
	/// the reproducibility claim is about - a run compared only at millisecond
	/// boundaries and only on its first byte would miss almost every way two
	/// runs could differ.
	fn timeline(link: &mut Link, count: u64) -> Vec<(u64, Vec<u8>)> {
		let mut out = Vec::new();

		for tick in 0..count * 10 + 20_000 {
			let now = Duration::from_micros(tick * 100);

			if tick < count * 10 && tick.is_multiple_of(10) {
				let number = u8::try_from(tick / 10 % 251).unwrap_or(0);

				link.receive(&[number, number ^ 0xFF, 0x5A, number], now);
			}

			while let Some(datagram) = link.poll(now) {
				out.push((tick, datagram.to_vec()));
			}
		}

		out
	}

	/// One number standing for a whole run, so that any part of it moving shows
	/// as one comparison.
	fn digest(timeline: &[(u64, Vec<u8>)]) -> u64 {
		let mut hash = 0xCBF2_9CE4_8422_2325_u64;

		for (tick, datagram) in timeline {
			for byte in tick.to_le_bytes().iter().chain(datagram) {
				hash ^= u64::from(*byte);
				hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
			}
		}

		hash
	}

	/// A link with the given conditions and seed.
	fn lying(seed: u64, conditions: Conditions) -> Link {
		let mut link = Link::new(seed);

		link.set(conditions);
		link
	}

	/// Sends `count` numbered datagrams a millisecond apart and reports which
	/// of them came out, in the order they did.
	fn run(link: &mut Link, count: u32) -> Vec<u8> {
		let mut out = Vec::new();

		for step in 0..count {
			let now = at(u64::from(step));
			let number = u8::try_from(step % 251).unwrap_or(0);

			link.receive(&[number], now);
			out.extend(
				drain(link, now)
					.into_iter()
					.filter_map(|d| d.first().copied()),
			);
		}

		// long enough after that anything still held has come due.
		out.extend(
			drain(link, at(u64::from(count) + 100_000))
				.into_iter()
				.filter_map(|d| d.first().copied()),
		);
		out
	}

	#[test]
	fn a_perfect_link_hands_back_what_it_was_given_at_once_and_in_order() {
		let mut link = Link::new(1);

		assert!(link.conditions().is_perfect());
		assert_eq!(cross(&mut link, b"one", at(0)), [b"one"]);
		assert_eq!(cross(&mut link, b"two", at(1)), [b"two"]);
		assert_eq!(link.received(), 2);
		assert_eq!(link.delivered(), 2);
		assert_eq!(link.dropped(), 0);
		assert_eq!(link.waiting(), 0);
	}

	#[test]
	fn a_datagram_of_no_bytes_crosses() {
		let mut link = Link::new(1);

		assert_eq!(cross(&mut link, b"", at(0)), [b""]);
	}

	#[test]
	fn a_lagged_datagram_comes_out_exactly_a_lag_later_and_not_before() {
		let mut link = lying(1, Conditions { lag: at(50), ..Conditions::PERFECT });

		link.receive(b"slow", at(100));
		assert!(drain(&mut link, at(149)).is_empty(), "a millisecond early is early");
		assert_eq!(link.waiting(), 1);
		assert_eq!(drain(&mut link, at(150)), [b"slow"], "and on the moment it is due");
		assert_eq!(link.waiting(), 0);
	}

	#[test]
	fn several_lagged_datagrams_come_out_in_the_order_they_went_in() {
		let mut link = lying(1, Conditions { lag: at(20), ..Conditions::PERFECT });

		for (step, text) in [b"one", b"two", b"six"].into_iter().enumerate() {
			link.receive(text, at(u64::try_from(step).unwrap_or(0)));
		}

		assert_eq!(drain(&mut link, at(100)), [b"one", b"two", b"six"]);
	}

	#[test]
	fn two_that_came_due_at_the_same_moment_come_out_in_the_order_they_arrived() {
		// what makes a run of this hashable rather than nearly so. Everything
		// in this field keys a queue on the moment alone and leaves two at the
		// same moment in whatever order a heap felt like.
		let mut link = lying(1, Conditions { lag: at(10), ..Conditions::PERFECT });

		link.receive(b"one", at(0));
		link.receive(b"two", at(0));
		link.receive(b"six", at(0));
		assert_eq!(drain(&mut link, at(10)), [b"one", b"two", b"six"]);
	}

	#[test]
	fn the_same_seed_and_the_same_steps_give_the_same_run() {
		// the contract the whole crate is built to keep, and the thing every
		// library in this field cannot say about its own simulator.
		let bad = Conditions {
			lag: at(30),
			jitter: at(15),
			loss: 0.2,
			burst: 3.0,
			duplicate: 0.1,
		};

		let mut one = lying(777, bad);
		let mut two = lying(777, bad);
		let first = timeline(&mut one, 500);
		let second = timeline(&mut two, 500);

		assert_eq!(first, second, "the same deliveries, whole, at the same moments");
		assert_eq!(one.dropped(), two.dropped());
		assert_eq!(one.duplicated(), two.duplicated());
		assert!(one.dropped() > 0 && one.duplicated() > 0, "and it really did lie");
		assert!(
			first.len() > 400 && first.len() < 600,
			"with a plausible number of deliveries: {}",
			first.len()
		);

		// the moments really do vary, or the grid would be measuring nothing.
		let spread: Vec<u64> = first
			.iter()
			.take(20)
			.map(|(tick, _)| *tick)
			.collect();

		assert!(spread.windows(2).any(|two| two[1] - two[0] != 10), "{spread:?}");
	}

	#[test]
	fn a_run_of_a_lying_link_is_the_run_it_has_always_been() {
		// two runs of the same build in one process agree because they are the
		// same code, so that on its own re-derives what the structure already
		// guarantees. What the claim is actually about is another machine and
		// another commit, and the only thing that pins it is a number written
		// down - the same thing the head of a datagram does with its sixteen
		// bytes, and for the same reason.
		//
		// This moves when the draw order moves, which is the point: the draw
		// order is a contract, and a change to it invalidates every recorded
		// run. Nothing here reads a clock, an environment or a hashed
		// container, which is what makes the number the same elsewhere.
		let bad = Conditions {
			lag: at(30),
			jitter: at(15),
			loss: 0.2,
			burst: 3.0,
			duplicate: 0.1,
		};
		let mut link = lying(777, bad);

		assert_eq!(digest(&timeline(&mut link, 500)), 0xA681_6D12_934E_096D);

		// and again with no jitter at all, because that is the path where the
		// number of draws a datagram costs could quietly change: a hold with
		// nothing to vary still takes its draw, and without a number written
		// down over this run nothing would say if it stopped.
		let steady = Conditions { jitter: Duration::ZERO, ..bad };
		let mut plain = lying(777, steady);

		assert_eq!(digest(&timeline(&mut plain, 500)), 0x78DA_1714_AD1C_5DBA);
	}

	#[test]
	fn a_different_seed_gives_a_different_run() {
		let bad = Conditions { loss: 0.2, ..Conditions::PERFECT };
		let mut one = lying(1, bad);
		let mut two = lying(2, bad);

		assert_ne!(run(&mut one, 500), run(&mut two, 500));
	}

	#[test]
	fn the_first_datagram_over_a_link_finds_it_good() {
		// a datagram's fate is drawn against the state it *found*, so a change
		// of state takes effect on the one after it. A link that has just been
		// made is good, so the first thing across it is never lost by the
		// chain - and drawing the fate against the state after the step would
		// lose that one as readily as any other. A share of one is the
		// exception and is not the chain: it loses in both states on purpose,
		// which the test two below this one asserts.
		for seed in 1..=64_u64 {
			let mut link = lying(seed, Conditions { loss: 0.9, ..Conditions::PERFECT });

			link.receive(b"first", Duration::ZERO);
			assert_eq!(link.dropped(), 0, "seed {seed} lost the first one");
		}
	}

	#[test]
	fn nothing_is_lost_when_nothing_is_meant_to_be() {
		let mut link = lying(5, Conditions { loss: 0.0, ..Conditions::PERFECT });

		assert_eq!(run(&mut link, 1000).len(), 1000);
		assert_eq!(link.dropped(), 0);
	}

	#[test]
	fn everything_is_lost_when_everything_is_meant_to_be() {
		// the chain would otherwise have to reach the bad state before any of
		// it went, so the first few would get through.
		let mut link = lying(5, Conditions { loss: 1.0, ..Conditions::PERFECT });

		assert!(run(&mut link, 1000).is_empty());
		assert_eq!(link.dropped(), 1000);
		assert_eq!(link.delivered(), 0);
	}

	#[test]
	fn the_share_that_is_lost_is_the_share_that_was_asked_for() {
		// a grid across seeds and shares, because one run of a lossy link is
		// one sample of a wobbly measurement.
		for seed in 1..=6_u64 {
			for share in [0.05_f32, 0.2, 0.5, 0.8] {
				let mut link =
					lying(seed * 104_729, Conditions { loss: share, ..Conditions::PERFECT });

				run(&mut link, 4000);

				let lost = f64::from(link.dropped()) / f64::from(link.received());

				assert!(
					(lost - f64::from(share)).abs() < 0.03,
					"seed {seed} wanted {share} lost and lost {lost}"
				);
			}
		}
	}

	#[test]
	fn the_burst_makes_the_losses_come_in_runs_without_losing_any_more_of_them() {
		// the property the two numbers were chosen for: one of them says how
		// much goes and the other says how clumped it is, and turning the
		// second one up must not turn the first one up with it.
		let share = 0.25_f32;
		let mut lengths = Vec::new();
		let mut shares = Vec::new();

		for burst in [1.0_f32, 4.0, 16.0] {
			let mut runs = Vec::new();
			let mut lost = Vec::new();

			for seed in 1..=6_u64 {
				let mut link = lying(seed * 15_485_863, Conditions {
					loss: share,
					burst,
					..Conditions::PERFECT
				});

				runs.push(mean_run(&mut link, 20_000));
				lost.push(f64::from(link.dropped()) / f64::from(link.received()));
			}

			lengths.push(mean(&runs));
			shares.push(mean(&lost));
		}

		for (burst, lost) in [1.0_f32, 4.0, 16.0].into_iter().zip(&shares) {
			assert!(
				(lost - f64::from(share)).abs() < 0.03,
				"a burst of {burst} lost {lost} rather than {share}"
			);
		}

		// and the run lengths are the formula rather than merely longer than
		// each other: chance alone gives runs of one over one minus the share,
		// and the burst multiplies that.
		let alone = 1.0 / (1.0 - f64::from(share));

		for (burst, length) in [1.0_f64, 4.0, 16.0].into_iter().zip(&lengths) {
			let wanted = burst * alone;

			assert!(
				(length - wanted).abs() < wanted * 0.15,
				"a burst of {burst} gave runs of {length} rather than {wanted}"
			);
		}
	}

	#[test]
	fn a_share_of_nil_loses_nothing_in_the_bad_state_either() {
		// which state the chain is in survives a change of conditions, so a
		// link somebody has just told to stop losing may well be in the middle
		// of a run of losses. The derivation on its own would give the bad
		// state a certain loss, and the link would go on dropping everything
		// until the chain happened to leave it.
		let lossless = chances(&Conditions {
			loss: 0.0,
			burst: 50.0,
			..Conditions::PERFECT
		});

		assert_eq!(lossless.good_loss, 0);
		assert_eq!(lossless.bad_loss, 0, "and the bad state loses nothing as well");
	}

	#[test]
	fn a_link_told_to_stop_losing_stops_at_once_even_from_the_bad_state() {
		let mut link = lying(3, Conditions {
			loss: 0.5,
			burst: 50.0,
			..Conditions::PERFECT
		});

		link.state = State::Bad;
		link.receive(b"gone", at(0));
		assert_eq!(link.dropped(), 1, "it is in the middle of a run of losses");

		link.set(Conditions::PERFECT);
		assert_eq!(link.state, State::Bad, "and the change of conditions did not move it");

		for step in 1..200_u64 {
			link.receive(b"kept", at(step));
			assert_eq!(drain(&mut link, at(step)).len(), 1, "at step {step}");
		}

		assert_eq!(link.dropped(), 1, "so nothing went after it was told not to");
	}

	#[test]
	fn a_share_that_is_not_a_number_is_read_as_nothing_lost() {
		// a console variable can be handed anything, and a share that is not a
		// number satisfies neither end of a comparison - so without the guard
		// it falls into the general derivation with both ways out of the chain
		// rounded to nil and the bad state's loss certain. A link in the bad
		// state would then drop everything for the rest of its life and one in
		// the good state would lose nothing, from the same numbers.
		let nonsense = Conditions {
			loss: f32::NAN,
			burst: f32::NAN,
			duplicate: f32::NAN,
			..Conditions::PERFECT
		};
		let worked = chances(&nonsense);

		assert_eq!(worked.good_loss, 0);
		assert_eq!(worked.bad_loss, 0);
		assert_eq!(worked.duplicate, 0);

		for state in [State::Good, State::Bad] {
			let mut link = lying(1, nonsense);

			link.state = state;
			assert_eq!(run(&mut link, 200).len(), 200, "from {state:?}");
			assert_eq!(link.dropped(), 0);
		}
	}

	#[test]
	fn a_burst_below_one_is_taken_as_one() {
		let mut lower = lying(9, Conditions {
			loss: 0.3,
			burst: 0.01,
			..Conditions::PERFECT
		});
		let mut one = lying(9, Conditions {
			loss: 0.3,
			burst: 1.0,
			..Conditions::PERFECT
		});

		assert_eq!(run(&mut lower, 500), run(&mut one, 500), "there is nothing under chance");
	}

	#[test]
	fn jitter_spreads_the_hold_either_side_of_the_lag_and_no_further() {
		let mut link = lying(3, Conditions {
			lag: at(40),
			jitter: at(10),
			..Conditions::PERFECT
		});
		let mut earliest = Duration::MAX;
		let mut latest = Duration::ZERO;

		for step in 0..2000_u64 {
			let sent = at(step * 1000);

			link.receive(b"x", sent);

			let mut out = sent;
			let ceiling = sent + at(200);

			// bounded on purpose. The hold here is at most fifty milliseconds,
			// so two hundred is room to spare - and an unbounded search is a
			// test that *hangs* rather than fails the moment anything stops a
			// datagram arriving, which is what a mutation that loses everything
			// does to it. A suite that hangs is worse than one that is wrong.
			while drain(&mut link, out).is_empty() {
				out += Duration::from_micros(100);

				assert!(out <= ceiling, "at step {step} nothing arrived inside {ceiling:?}");
			}

			let held = out.saturating_sub(sent);

			earliest = earliest.min(held);
			latest = latest.max(held);
		}

		assert!(earliest >= at(30), "never held for less than a jitter under the lag");
		assert!(latest <= at(50) + Duration::from_micros(100), "nor more than one over");
		assert!(earliest < at(31), "and the whole of the span is reached: {earliest:?}");
		assert!(latest > at(49), "at both ends: {latest:?}");
	}

	#[test]
	fn jitter_is_the_only_thing_that_reorders_anything() {
		let steady = Conditions { lag: at(40), ..Conditions::PERFECT };
		let shaken = Conditions { jitter: at(10), ..steady };
		let mut still = lying(4, steady);
		let mut moving = lying(4, shaken);
		let sent: Vec<u8> = (0..200)
			.map(|step| u8::try_from(step % 251).unwrap_or(0))
			.collect();

		assert_eq!(run(&mut still, 200), sent, "with no jitter, the order it went in");
		assert_ne!(run(&mut moving, 200), sent, "and with it, not");
	}

	#[test]
	fn a_duplicated_datagram_arrives_twice_and_only_jitter_parts_the_two() {
		// worth pinning both ways round rather than only the one the field doc
		// wants. With no jitter the two copies are held for exactly as long as
		// each other and come out together, and that is what a link with only
		// this number set does.
		let together = Conditions {
			lag: at(20),
			duplicate: 1.0,
			..Conditions::PERFECT
		};
		let mut side_by_side = lying(8, together);

		side_by_side.receive(b"twice", at(0));
		assert_eq!(side_by_side.duplicated(), 1);
		assert_eq!(side_by_side.waiting(), 2, "both copies are being held");
		assert_eq!(
			side_by_side.held[0].due, side_by_side.held[1].due,
			"and for the same length, so they arrive at the same moment"
		);
		assert!(drain(&mut side_by_side, at(19)).is_empty(), "neither is due yet");
		assert_eq!(drain(&mut side_by_side, at(20)), [b"twice", b"twice"]);
		assert_eq!(side_by_side.delivered(), 2);

		let mut apart = lying(8, Conditions { jitter: at(10), ..together });

		apart.receive(b"twice", at(0));
		assert_eq!(apart.waiting(), 2);
		assert_ne!(
			apart.held[0].due, apart.held[1].due,
			"with jitter each copy is held for its own length"
		);
		assert_eq!(drain(&mut apart, at(100)), [b"twice", b"twice"]);
	}

	#[test]
	fn nothing_is_duplicated_when_nothing_is_meant_to_be() {
		let mut link = lying(8, Conditions { duplicate: 0.0, ..Conditions::PERFECT });

		assert_eq!(run(&mut link, 500).len(), 500);
		assert_eq!(link.duplicated(), 0);
	}

	#[test]
	fn a_datagram_that_was_lost_is_not_duplicated() {
		// there is nothing left of it to copy, and a link that duplicated what
		// it had just thrown away would deliver more than it received.
		let mut link = lying(6, Conditions {
			loss: 1.0,
			duplicate: 1.0,
			..Conditions::PERFECT
		});

		run(&mut link, 500);
		assert_eq!(link.dropped(), 500);
		assert_eq!(link.duplicated(), 0);
		assert_eq!(link.delivered(), 0);
	}

	#[test]
	fn the_share_that_arrives_twice_is_the_share_that_was_asked_for() {
		for seed in 1..=6_u64 {
			for share in [0.1_f32, 0.5] {
				let mut link = lying(seed * 32_452_843, Conditions {
					duplicate: share,
					..Conditions::PERFECT
				});

				run(&mut link, 4000);

				let twice = f64::from(link.duplicated()) / f64::from(link.received());

				assert!(
					(twice - f64::from(share)).abs() < 0.03,
					"seed {seed} wanted {share} twice and got {twice}"
				);
			}
		}
	}

	#[test]
	fn the_chain_only_moves_when_a_datagram_arrives() {
		// so a burst is a run of datagrams rather than a stretch of time, and a
		// link left quiet is still in whatever state the last one left it in.
		let bad = Conditions {
			loss: 0.4,
			burst: 5.0,
			..Conditions::PERFECT
		};
		let mut quick = lying(21, bad);
		let mut slow = lying(21, bad);
		let mut one = Vec::new();
		let mut two = Vec::new();

		for step in 0..400_u64 {
			quick.receive(b"x", at(step));
			one.push(quick.waiting());
			drain(&mut quick, at(step));

			// the same datagrams, a whole minute apart each time.
			slow.receive(b"x", at(step * 60_000));
			two.push(slow.waiting());
			drain(&mut slow, at(step * 60_000));
		}

		assert_eq!(one, two, "time passing is not what moves it");
		assert_eq!(quick.dropped(), slow.dropped());
		assert!(quick.dropped() > 0);
	}

	#[test]
	fn changing_the_conditions_leaves_what_is_already_held_alone() {
		// a datagram was held for as long as the numbers at the time said, and
		// rewriting that would make a change to the numbers reach backwards.
		let mut link = lying(1, Conditions { lag: at(100), ..Conditions::PERFECT });

		link.receive(b"slow", at(0));
		link.set(Conditions::PERFECT);
		link.receive(b"quick", at(0));
		assert_eq!(drain(&mut link, at(0)), [b"quick"], "the new one is not held");
		assert_eq!(drain(&mut link, at(100)), [b"slow"], "and the old one still is");
	}

	#[test]
	fn the_buffers_a_link_holds_are_reused_rather_than_taken_again() {
		// a link sits in the receive path of an ordinary run, so it must not
		// take a fresh buffer for every datagram that crosses it.
		let mut link = Link::new(1);

		for step in 0..1000_u64 {
			link.receive(b"x", at(step));
			assert_eq!(drain(&mut link, at(step)).len(), 1);

			// exactly one rather than at most one: a link that threw its
			// buffers away instead of putting them back would hold none, and a
			// ceiling would not notice.
			assert_eq!(link.spare.len(), 1, "one buffer put back, at step {step}");
		}

		assert!(link.spare[0].capacity() > 0, "and it is a real one rather than an empty one");
		assert_eq!(link.waiting(), 0);
		assert_eq!(link.delivered(), 1000);
	}

	#[test]
	fn a_shorter_datagram_through_a_reused_buffer_keeps_none_of_the_longer_one() {
		// every other test here sends datagrams of one length, so a buffer that
		// was shortened rather than emptied would deliver the tail of whatever
		// was in it before and nothing would say so.
		let mut link = Link::new(1);

		assert_eq!(cross(&mut link, b"aaaaaaaaaaaaaaaa", at(0)), [b"aaaaaaaaaaaaaaaa"]);
		assert_eq!(cross(&mut link, b"bb", at(1)), [b"bb"]);
		assert_eq!(cross(&mut link, b"", at(2)), [b""]);
		assert_eq!(cross(&mut link, b"ccc", at(3)), [b"ccc"]);
		assert_eq!(cross(&mut link, b"dddddddddddddddd", at(4)), [b"dddddddddddddddd"]);
	}

	#[test]
	fn a_jitter_larger_than_the_lag_piles_up_on_no_hold_at_all() {
		// a wire cannot hand a datagram over before it arrived, so the early
		// part of the span is clamped rather than wrapped. What lands on no
		// hold at all is exactly the share of the span that was negative, which
		// at no lag is half of it - and that is worth pinning, because it means
		// the average hold is not the lag that was asked for.
		let mut link = lying(17, Conditions {
			lag: Duration::ZERO,
			jitter: at(10),
			..Conditions::PERFECT
		});
		let mut at_once = 0_u32;

		for step in 0..4000_u64 {
			let now = at(step * 100);

			drain(&mut link, now);
			link.receive(b"x", now);
			at_once += u32::try_from(drain(&mut link, now).len()).unwrap_or(0);
		}

		let share = f64::from(at_once) / 4000.0;

		assert!((share - 0.5).abs() < 0.03, "{share} of them were handed over at once");
	}

	#[test]
	fn the_counts_add_up() {
		let mut link = lying(13, Conditions {
			loss: 0.3,
			duplicate: 0.2,
			..Conditions::PERFECT
		});

		run(&mut link, 2000);
		assert_eq!(link.received(), 2000);
		assert_eq!(
			link.delivered(),
			link.received() - link.dropped() + link.duplicated(),
			"everything that was not thrown away, plus every second copy"
		);
	}

	/// The mean of a handful of numbers.
	fn mean(values: &[f64]) -> f64 {
		values.iter().sum::<f64>() / f64::from(u32::try_from(values.len()).unwrap_or(1))
	}

	/// How long a run of consecutive losses is, on average, over one run.
	fn mean_run(link: &mut Link, count: u32) -> f64 {
		let mut runs = 0_u32;
		let mut lost = 0_u32;
		let mut inside = false;

		for step in 0..count {
			let now = at(u64::from(step));
			let before = link.dropped();

			link.receive(b"x", now);
			drain(link, now);

			let gone = link.dropped() > before;

			lost += u32::from(gone);
			runs += u32::from(gone && !inside);
			inside = gone;
		}

		if runs == 0 {
			return 0.0;
		}

		f64::from(lost) / f64::from(runs)
	}
}
