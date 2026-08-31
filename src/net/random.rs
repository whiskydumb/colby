//! Numbers that look random and are not.
//!
//! Twelve lines of shift and multiply, seeded by hand, because the whole point
//! of the lying link above it is that a run of it can be repeated. Every
//! networking library in this field reaches for a thread-local generator seeded
//! from the operating system, and every one of them therefore cannot say what
//! its own simulated link will do twice - which is fine for a demo and useless
//! as a tool. A seed and a shift register cost nothing and buy a hash.
//!
//! **A probability crosses as a threshold rather than as a fraction.** The
//! caller writes 0.02 and [`threshold`] turns it into the share of the draw
//! space below which a draw counts, once, when the conditions are set; every
//! draw after that is one integer comparison. The same arrangement the mixer
//! has with its gains, and for the same reason: the arithmetic that decides
//! something happens where it can be looked at rather than in the middle of a
//! loop.

/// What a seed of nil is replaced with.
///
/// Nil is a fixed point of the shift register below - it maps to itself
/// forever, and every draw from it is the same number. That is worth a line
/// rather than a note, because nil is exactly the seed somebody writes first.
const AWAKE: u64 = 0x9E37_79B9_7F4A_7C15;

/// The odd multiplier that scrambles the low bits of the shift register.
const SCRAMBLE: u64 = 0x2545_F491_4F6C_DD1D;

/// How many bits of a draw a probability is judged on.
///
/// Read in three places - the size of the space, the float that size is scaled
/// through, and the shift that reduces a draw into it - and the assertion below
/// is what stops two of them being changed and the third left behind.
const WIDTH: u32 = 32;

/// The size of the space a draw is judged against.
///
/// Two to the thirty-second, so a probability is resolved to about one part in
/// four thousand million and a certainty is exact rather than nearly exact.
pub const CERTAIN: u64 = 1 << WIDTH;

/// The same size, written out, because a float is where a share is scaled.
const SPACE: f64 = 4_294_967_296.0;

#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "two constants compared where the build can see both of them"
)]
const _: () =
	assert!(SPACE as u64 == CERTAIN, "a share is scaled across the space a draw covers");

/// A generator that always says the same thing when it is seeded the same way.
///
/// @note: deliberately not `Copy`. A generator that could be copied by accident
/// would hand the same numbers out twice from what looks like two of them, and
/// the whole value of the thing is that its stream is a single sequence
/// somebody can point at.
#[expect(
	missing_copy_implementations,
	reason = "a generator copied by accident would hand out the same stream twice"
)]
#[derive(Clone, Debug)]
pub struct Random {
	state: u64,
}

impl Random {
	/// A generator started from a seed.
	///
	/// @param seed - anything at all; nil becomes [`AWAKE`]
	#[must_use]
	pub const fn new(seed: u64) -> Self {
		Self {
			state: if seed == 0 { AWAKE } else { seed },
		}
	}

	/// The next sixty-four bits.
	///
	/// @note: the three shifts are one of the published triples that give this
	/// register its full period, and nothing here can check the *period* - it
	/// is eighteen million million million draws long. What can be checked,
	/// and is, is that these are the numbers: a handful of draws from a fixed
	/// seed are written down as literals below, so changing a shift or the
	/// multiplier is a failing test rather than a silently different generator
	/// whose runs all look equally plausible.
	pub const fn draw(&mut self) -> u64 {
		let mut state = self.state;

		state ^= state >> 12;
		state ^= state << 25;
		state ^= state >> 27;
		self.state = state;
		state.wrapping_mul(SCRAMBLE)
	}

	/// A draw somewhere below a bound.
	///
	/// The whole draw is multiplied by the bound and the top of the product
	/// taken, rather than a remainder. The reason is which bits each of those
	/// reads: a remainder is decided by the *low* end of the draw, and the
	/// multiply that scrambles this register does not reach the lowest of them
	/// at all - the bottom bit of a product with an odd number is the bottom
	/// bit of what went in, which here is a raw shift-register bit. So a
	/// remainder would hand out the weak end of the stream while every
	/// probability above is judged on the well-mixed top half. Taking the top
	/// of a wide product reads the same end as those do, and has no division in
	/// it.
	///
	/// @note: very slightly biased, by about the bound over the whole draw
	/// space. For the spans this is used on that is a part in eighteen million
	/// million million, which is smaller than anything downstream can tell.
	///
	/// @param bound - one past the largest answer; nil answers nil
	pub fn below(&mut self, bound: u64) -> u64 {
		let wide = u128::from(self.draw()) * u128::from(bound);

		u64::try_from(wide >> u64::BITS).expect("the top of that product is below the bound")
	}

	/// A draw reduced to the space a probability is measured in.
	///
	/// @return somewhere from nil up to but not including [`CERTAIN`]
	pub const fn share(&mut self) -> u64 { self.draw() >> WIDTH }

	/// Whether this draw fell below a threshold.
	///
	/// @param threshold - what [`threshold`] made of a probability
	pub const fn chance(&mut self, threshold: u64) -> bool { happens(self.share(), threshold) }
}

/// Whether a share of the space falls below a threshold.
///
/// Its own function rather than a comparison inside [`Random::chance`] so that
/// both ends of it can be asserted. Sampling cannot reach them: the difference
/// between this comparison and the one that includes its own threshold is one
/// draw in four thousand million, and it is exactly the difference between a
/// probability of nil never happening and it happening.
///
/// @param share - a draw, from [`Random::share`]
/// @param threshold - from [`threshold`]
#[must_use]
pub const fn happens(share: u64, threshold: u64) -> bool { share < threshold }

/// The share of the draw space a probability covers.
///
/// Nil and one are both exact: nothing is below nil, and everything is below
/// [`CERTAIN`] because a draw only ever has thirty-two bits in it.
///
/// @param probability - nil through one; anything outside is clamped
#[must_use]
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "a float has no fallible form of this, and a clamped share cannot be out of range"
)]
pub fn threshold(probability: f32) -> u64 {
	// @note: nothing can observe this, and it is kept because of what it says.
	// A float that is not a number saturates to nil on its way into an integer,
	// so the answer is the same either way - but then "a share that is not a
	// number is nothing" would be a property of how a cast happens to behave
	// rather than something anybody decided, and the test below asserts it as
	// though it were decided. The guard is what makes that honest. The place
	// where a share that is not a number really does change an answer is the
	// chain it feeds, and that one has a guard nothing else can stand in for.
	if probability.is_nan() {
		return 0;
	}

	let share = f64::from(probability.clamp(0.0, 1.0));

	(share * SPACE) as u64
}

#[cfg(test)]
mod tests {
	use super::*;

	/// How many draws the counting tests take. Enough that a share is steady to
	/// about a part in a thousand, and small enough to be instant.
	const MANY: u32 = 100_000;

	#[test]
	fn the_same_seed_says_the_same_thing() {
		let mut one = Random::new(12345);
		let mut two = Random::new(12345);

		for _ in 0..1000 {
			assert_eq!(one.draw(), two.draw());
		}
	}

	#[test]
	fn a_different_seed_says_something_else() {
		let mut one = Random::new(1);
		let mut two = Random::new(2);
		let mut same = 0;

		for _ in 0..1000 {
			if one.draw() == two.draw() {
				same += 1;
			}
		}

		assert_eq!(same, 0, "two streams agreeing at all would be a broken register");
	}

	#[test]
	fn a_seed_of_nil_is_replaced_rather_than_repeating_one_number_forever() {
		// nil is a fixed point of the register, so without the replacement
		// every draw from it is the same value - and nil is the seed somebody
		// writes first.
		let mut zero = Random::new(0);
		let first = zero.draw();
		let second = zero.draw();

		assert_ne!(first, second, "a seed of nil still moves");
		assert_eq!(Random::new(0).state, AWAKE);
		assert_eq!(Random::new(7).state, 7, "and anything else is taken as it is");
	}

	#[test]
	fn the_stream_from_a_fixed_seed_is_the_stream_it_has_always_been() {
		// a known answer, written down, which is the only thing that pins the
		// three shifts and the multiplier. Two runs of the same build agreeing
		// says nothing about them: swap a shift for its neighbor and both runs
		// still agree with each other, every statistical test still passes, and
		// every recorded run ever taken is quietly worthless.
		//
		// If this fails, the generator was changed. That is a decision to take
		// on purpose - every hash of a recorded run moves with it.
		let mut random = Random::new(1);
		let first: Vec<u64> = std::iter::repeat_with(|| random.draw())
			.take(6)
			.collect();

		assert_eq!(first, [
			0x47E4_CE4B_896C_DD1D,
			0xABCF_A6A8_E079_651D,
			0xB9D1_0D8F_EB73_1F57,
			0x4DB4_18A0_BB1B_019D,
			0x0E61_99B0_4D5A_A600,
			0xC867_4BCB_42E3_AAD9,
		]);

		// and from the seed a nil is replaced with, which is the other one
		// anybody will meet.
		let mut awake = Random::new(0);
		let second: Vec<u64> = std::iter::repeat_with(|| awake.draw())
			.take(3)
			.collect();

		assert_eq!(second, [0x0D83_B3E2_9A21_487A, 0x54C4_4C79_F1FE_9D67, 0xA845_F342_007A_0E78]);

		// the share is the top half of the draw, and that is written down too.
		assert_eq!(Random::new(1).share(), 0x47E4_CE4B);
	}

	#[test]
	fn a_small_seed_does_not_hand_back_a_small_first_number() {
		// what the multiply at the end of a draw is for. The shift register on
		// its own is linear, so a state of one is a first draw with three bits
		// set in it - and everything drawn from a low-numbered seed would then
		// be lopsided until the state had spread out.
		for seed in 1..=16_u64 {
			let first = Random::new(seed).draw();
			let bits = first.count_ones();

			assert!(
				(16..=48).contains(&bits),
				"seed {seed} drew {first:#018x}, which has {bits} bits set in it"
			);
		}
	}

	#[test]
	fn both_ends_of_the_comparison_a_probability_turns_into_are_exact() {
		// one draw in four thousand million, so no amount of sampling reaches
		// it - and it is the whole difference between a probability of nil
		// never happening and it happening.
		assert!(!happens(0, 0), "nothing at all is below nothing at all");
		assert!(happens(0, 1), "and the lowest draw there is falls below one");
		assert!(happens(CERTAIN - 1, CERTAIN), "the highest draw is still below certain");
		assert!(!happens(CERTAIN - 1, CERTAIN - 1), "and not below itself");
	}

	#[test]
	fn a_draw_is_reduced_into_the_space_a_probability_is_measured_in() {
		let mut random = Random::new(31);

		for _ in 0..MANY {
			assert!(random.share() < CERTAIN);
		}
	}

	#[test]
	fn the_register_does_not_settle_on_a_number() {
		let mut random = Random::new(99);
		let mut seen = std::collections::HashSet::new();

		for _ in 0..10_000 {
			seen.insert(random.draw());
		}

		assert_eq!(seen.len(), 10_000, "ten thousand draws and no two alike");
	}

	#[test]
	fn a_probability_of_nil_never_happens_and_one_always_does() {
		let mut random = Random::new(4);
		let never = threshold(0.0);
		let always = threshold(1.0);

		assert_eq!(never, 0);
		assert_eq!(always, CERTAIN, "and the certainty is exact rather than nearly so");

		for _ in 0..MANY {
			assert!(!random.chance(never));
			assert!(random.chance(always));
		}
	}

	#[test]
	fn a_probability_outside_nil_through_one_is_clamped() {
		assert_eq!(threshold(-5.0), 0);
		assert_eq!(threshold(5.0), CERTAIN);
		assert_eq!(threshold(f32::NAN), 0, "and something that is not a number is nothing");
	}

	/// What share of [`MANY`] draws fell below the threshold for `wanted`.
	fn measure(seed: u64, wanted: f32) -> f64 {
		let mut random = Random::new(seed);
		let cut = threshold(wanted);
		let mut below = 0_u32;

		for _ in 0..MANY {
			below += u32::from(random.chance(cut));
		}

		f64::from(below) / f64::from(MANY)
	}

	#[test]
	fn a_share_of_the_draws_falls_below_the_share_that_was_asked_for() {
		// a grid rather than one run: every seed has to land in the band, or
		// what is being reported is one sample of a wobbly measurement.
		for seed in 1..=8_u64 {
			for wanted in [0.1_f32, 0.25, 0.5, 0.75, 0.9] {
				let share = measure(seed * 7919, wanted);
				let off = (share - f64::from(wanted)).abs();

				assert!(off < 0.01, "seed {seed} wanted {wanted} and got {share}");
			}
		}
	}

	#[test]
	fn a_draw_below_a_bound_stays_below_it_and_reaches_both_ends() {
		let mut random = Random::new(11);
		let mut low = false;
		let mut high = false;

		for _ in 0..MANY {
			let drawn = random.below(7);

			assert!(drawn < 7);
			low |= drawn == 0;
			high |= drawn == 6;
		}

		assert!(low && high, "and the whole of the span is reachable");
	}

	#[test]
	fn a_bound_of_nil_answers_nil() {
		let mut random = Random::new(3);

		for _ in 0..MANY {
			assert_eq!(random.below(0), 0);
		}
	}

	#[test]
	fn a_draw_below_a_bound_reads_the_same_end_of_the_stream_as_a_probability_does() {
		// the bottom bit of this register is the one its scramble cannot reach,
		// so a bound that is a power of two is where taking the low end would
		// show. Both of these have to look like coins.
		let mut random = Random::new(41);
		let mut heads = 0_u32;
		let mut runs = 0_u32;
		let mut last = 2_u64;

		for _ in 0..MANY {
			let flip = random.below(2);

			heads += u32::try_from(flip).unwrap_or(0);
			runs += u32::from(flip != last);
			last = flip;
		}

		let share = f64::from(heads) / f64::from(MANY);
		let changes = f64::from(runs) / f64::from(MANY);

		assert!((share - 0.5).abs() < 0.01, "heads came up {share} of the time");
		assert!((changes - 0.5).abs() < 0.01, "and it changed {changes} of the time");
	}

	#[test]
	fn a_draw_below_a_bound_is_spread_evenly_across_it() {
		let mut random = Random::new(2024);
		let mut counts = [0_u32; 10];

		for _ in 0..MANY {
			let drawn = usize::try_from(random.below(10)).unwrap_or(0);

			counts[drawn] += 1;
		}

		for (face, count) in counts.iter().enumerate() {
			let share = f64::from(*count) / f64::from(MANY);

			assert!((share - 0.1).abs() < 0.01, "face {face} came up {share} of the time");
		}
	}
}
