//! Where a sound sits between the two ears, and how loud it is when it gets
//! there.
//!
//! Two functions and no state. Both are called on the *simulation's* side of
//! the boundary, once a step per voice, so that what crosses to whatever is
//! filling a device buffer is two numbers rather than a position, a listener
//! and a distance model. That is the whole reason this is its own module: the
//! arithmetic that decides how loud something is has no business running sixty
//! times a second inside an audio callback.
//!
//! **Stereo panning and nothing more.** A source directly in front and one
//! directly behind pan to the same place, which is a real limitation and the
//! one thing head-related transfer functions exist to fix. The engine in this
//! field that has them says outright that they cost five to six times a plain
//! panner and need a measured impulse-response database; that is a different
//! project.

use colby_core::{
	abi::{Listener, Mix, Voice},
	glam::Vec3,
};

/// How far left or right a source has to be before it is entirely in one ear.
///
/// Not a distance: this is the dot product of the direction to the source with
/// the listener's own right, so it is already the sine of the angle off the
/// line of sight. One means "exactly to the right".
pub const HARD: f32 = 1.0;

/// A quarter turn, which is the whole width of the pan law below.
const QUARTER: f32 = std::f32::consts::FRAC_PI_4;

/// The two gains a source at this position gets.
///
/// A constant-power law: the gains are the cosine and the sine of an angle
/// running from zero to a quarter turn, so their squares always sum to one and
/// a source swept from ear to ear does not get louder in the middle. The cost,
/// and it is worth knowing rather than discovering: **a centered source is at
/// about 0.707 in each ear rather than 1.0**, three decibels down on a voice
/// with no place in the world. That is what every panner does, and the two are
/// different things - one is a recording, the other is a thing standing
/// somewhere.
///
/// @param position - where it is between the ears, -1 fully left through 0
/// ahead to +1 fully right. Anything outside is clamped.
/// @return `(left, right)`
#[must_use]
pub fn stereo(position: f32) -> (f32, f32) {
	let angle = (position.clamp(-HARD, HARD) + HARD) * QUARTER;

	(angle.cos(), angle.sin())
}

/// What one voice is worth in each ear, everything applied.
///
/// The one place volume, category, distance and direction are put together, so
/// that nothing downstream has to know any of them.
///
/// @param voice - what is playing
/// @param listener - where the world is being heard from
/// @param mix - how loud each category is
/// @return `(left, right)`, both at or above zero
#[must_use]
pub fn gains(voice: &Voice, listener: &Listener, mix: &Mix) -> (f32, f32) {
	let level = voice.gain() * mix.of(voice.category) * voice.attenuation(listener.at);

	if !voice.positioned {
		return (level, level);
	}

	let (left, right) = stereo(sideness(voice.at, listener));

	(level * left, level * right)
}

/// How far to the right of the listener something is, as a number the pan law
/// takes.
///
/// @param at - where the source is, in world space
/// @param listener - where the ears are and which way they face
/// @return -1 fully left through 0 ahead or behind to +1 fully right, and zero
/// for a source standing exactly where the listener is - which is the one case
/// that has no direction at all and would otherwise be whatever the
/// normalization of a zero vector happened to give
#[must_use]
pub fn sideness(at: Vec3, listener: &Listener) -> f32 {
	let (right, ..) = listener.frame();

	// @note: `normalize_or_zero` rather than a branch on the zero case. A
	// source standing exactly on the listener has no direction, and the zero
	// vector's dot with anything is zero, which is the middle - so the guard
	// that was written here first said nothing the one-liner does not, and no
	// mutation of it could fail a test.
	//
	// @note: the clamp cannot be observed either, and is kept anyway. Both
	// vectors are normalized, so their dot product is inside the range by
	// construction to within whatever a square root costs, and the pan law
	// clamps its own argument regardless. What it defends is the range this
	// function *documents*, against a float that comes back at 1.0000001 -
	// which is a case a test cannot construct and a caller may still meet.
	(at - listener.at)
		.normalize_or_zero()
		.dot(right)
		.clamp(-HARD, HARD)
}

#[cfg(test)]
mod tests {
	use colby_core::abi::{Category, SoundId};

	use super::*;

	/// Facing down negative z from the origin, y up, so right is positive x.
	fn ears() -> Listener { Listener::DEFAULT }

	#[test]
	fn a_source_in_the_middle_is_the_same_in_both_ears() {
		let (left, right) = stereo(0.0);

		assert!((left - right).abs() < 1e-6, "the middle is the middle");
		assert!(
			(left.mul_add(left, right * right) - 1.0).abs() < 1e-6,
			"and the two of them carry all of the power"
		);
	}

	#[test]
	fn a_source_at_one_side_is_only_in_that_ear() {
		let (left, right) = stereo(-1.0);

		assert!((left - 1.0).abs() < 1e-6, "hard left is all left");
		assert!(right.abs() < 1e-6, "and none of it is right");

		let (left, right) = stereo(1.0);

		assert!(left.abs() < 1e-6);
		assert!((right - 1.0).abs() < 1e-6);
	}

	#[test]
	fn the_power_is_the_same_wherever_it_is_panned() {
		// the property the law is named after, and the one a linear pan does
		// not have: a source swept from one ear to the other must not dip in
		// the middle.
		let mut steps = 0;

		for tenth in -10_i16..=10 {
			let position = f32::from(tenth) / 10.0;
			let (left, right) = stereo(position);
			let power = left.mul_add(left, right * right);

			assert!(
				(power - 1.0).abs() < 1e-6,
				"at {position} the power is {power} rather than one"
			);
			steps += 1;
		}

		assert_eq!(steps, 21, "and it was asked across the whole sweep");
	}

	#[test]
	fn a_position_past_the_ends_is_clamped_rather_than_wrapped() {
		assert_eq!(stereo(-50.0), stereo(-1.0), "further left than left is left");
		assert_eq!(stereo(50.0), stereo(1.0), "and the same the other way");
	}

	#[test]
	fn which_side_a_source_is_on_is_read_off_the_listeners_own_axes() {
		let listener = ears();

		assert!(
			(sideness(Vec3::X * 5.0, &listener) - 1.0).abs() < 1e-6,
			"positive x is to the right of somebody facing negative z"
		);
		assert!(
			(sideness(Vec3::NEG_X * 5.0, &listener) + 1.0).abs() < 1e-6,
			"and negative x left"
		);
		assert!(sideness(Vec3::NEG_Z * 5.0, &listener).abs() < 1e-6, "straight ahead is neither");
	}

	#[test]
	fn a_source_in_front_and_one_behind_pan_to_the_same_place() {
		// not a bug, and worth a test so that it is a decision somebody took
		// rather than something nobody noticed. Telling them apart is what a
		// head-related transfer function is for.
		let listener = ears();
		let ahead = sideness(Vec3::NEG_Z * 5.0, &listener);
		let behind = sideness(Vec3::Z * 5.0, &listener);

		assert!((ahead - behind).abs() < 1e-6, "{ahead} against {behind}");
	}

	#[test]
	fn turning_the_listener_turns_where_everything_is() {
		let facing_x = Listener { forward: Vec3::X, ..Listener::DEFAULT };

		assert!(
			(sideness(Vec3::Z * 5.0, &facing_x) - 1.0).abs() < 1e-6,
			"facing positive x, positive z is on the right"
		);
	}

	#[test]
	fn a_source_standing_on_the_listener_is_in_the_middle() {
		// no direction at all, and normalizing that would hand back whatever
		// a zero vector normalizes to. Everything in one ear because somebody
		// walked into it is the bug this prevents.
		assert!(sideness(Vec3::ZERO, &ears()).abs() < f32::EPSILON);
	}

	#[test]
	fn a_voice_with_no_place_in_the_world_is_the_same_in_both_ears_at_full_level() {
		let voice = Voice::flat(SoundId::NONE).volume(0.5);
		let (left, right) = gains(&voice, &ears(), &Mix::FULL);

		assert!((left - 0.5).abs() < 1e-6, "no panning and no attenuation");
		assert!((right - 0.5).abs() < 1e-6, "in either ear");
	}

	#[test]
	fn a_positioned_voice_carries_its_volume_its_category_and_its_distance() {
		let mix = Mix { effects: 0.5, ..Mix::FULL };
		let voice = Voice::at(SoundId::NONE, Vec3::X * 4.0)
			.volume(0.5)
			.range(2.0, 100.0);
		let (left, right) = gains(&voice, &ears(), &mix);

		// half the volume, half the category, half again for being at twice
		// the reference distance, and then hard right.
		assert!(left.abs() < 1e-6, "it is entirely on the right: {left}");
		assert!(
			(right - 0.125).abs() < 1e-6,
			"and one eighth as loud as it was recorded: {right}"
		);
	}

	#[test]
	fn a_muted_category_silences_the_voices_in_it_and_nothing_else() {
		let mix = Mix { music: 0.0, ..Mix::FULL };
		let song = Voice::flat(SoundId::NONE).category(Category::Music);
		let thud = Voice::flat(SoundId::NONE);

		assert_eq!(gains(&song, &ears(), &mix), (0.0, 0.0), "the music is off");
		assert_eq!(gains(&thud, &ears(), &mix), (1.0, 1.0), "and everything else is not");
	}
}
