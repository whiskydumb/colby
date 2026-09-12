//! One value as the sixteen bits a half-precision format stores.
//!
//! **Here rather than beside the texel format it encodes**, which is where it
//! used to live. Two things now write half-precision texels - the asset
//! compiler, which filters an environment into a cube offline, and the renderer
//! itself, which bakes the split-sum table of its own lobe at startup - and
//! they are in crates that do not know about each other. The only crate both
//! see is this one, and the module the host and a game agree on is not the
//! place for an encoder: what crosses that boundary is the *format*, and this
//! is one way of writing it.
//!
//! Rust has no stable half-precision type, so the bits are assembled by hand.

/// One value as the sixteen bits the format stores.
///
/// The sign, the exponent held inside what the format reaches, and the top ten
/// bits of the mantissa, rounded to nearest with ties going to even - which is
/// what every other encoder of this format does, and therefore what a second
/// answer will agree with.
///
/// @param value - the number to narrow
/// @return its sixteen bits
#[must_use]
pub fn half(value: f32) -> u16 {
	let bits = value.to_bits();
	let sign = u16::try_from(bits >> 31).unwrap_or(0) << 15;
	let exponent = i32::try_from((bits >> 23) & 0xFF).unwrap_or(0) - 127;
	let mantissa = bits & 0x007F_FFFF;

	// a value that is not a number at all, or one past what the format reaches,
	// both come back as the largest finite value of their sign: a sky with an
	// infinity in it is a sky nobody can filter, and clipping it is better than
	// a texel the GPU reads as not-a-number and spreads over the picture.
	if exponent == 128 {
		return sign | 0x7BFF;
	}

	if exponent > 15 {
		return sign | 0x7BFF;
	}

	// below what the format holds with a full mantissa, the value is stored
	// with a smaller one and an exponent of nought
	if exponent < -14 {
		let shift = u32::try_from(-14 - exponent).unwrap_or(32);

		if shift > 24 {
			return sign;
		}

		let widened = (mantissa | 0x0080_0000) >> shift;

		return sign | u16::try_from(rounded(widened) & 0x03FF).unwrap_or(0);
	}

	let stored = u32::try_from(exponent + 15).unwrap_or(0) << 10;
	let rounded = rounded(mantissa);

	// rounding the mantissa up can carry into the exponent, and adding the two
	// together is what makes that carry land where it should
	sign | u16::try_from((stored + rounded).min(0x7BFF)).unwrap_or(0x7BFF)
}

/// A twenty-three bit mantissa rounded down to ten, ties to even.
fn rounded(mantissa: u32) -> u32 {
	let kept = mantissa >> 13;
	let dropped = mantissa & 0x1FFF;

	if dropped > 0x1000 || (dropped == 0x1000 && kept & 1 == 1) {
		kept + 1
	} else {
		kept
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The sixteen bits read back as a number.
	///
	/// Written out rather than taken from a crate, because the point of the
	/// tests below is that the encoder above agrees with the format and not
	/// that it agrees with itself.
	fn widened(bits: u16) -> f32 {
		let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
		let exponent = i32::from((bits >> 10) & 0x1F);
		let mantissa = f32::from(bits & 0x03FF);

		if exponent == 0 {
			return sign * mantissa * 2.0_f32.powi(-24);
		}

		sign * (1.0 + mantissa / 1024.0) * 2.0_f32.powi(exponent - 15)
	}

	#[test]
	fn half_precision_round_trips_the_values_that_fit_in_it() {
		for value in [0.0_f32, 1.0, 0.5, 2.0, 0.25, 1024.0, 65504.0, -1.0, -0.5] {
			assert!(
				(widened(half(value)) - value).abs() <= value.abs() * 1.0e-3,
				"{value} came back as {}",
				widened(half(value))
			);
		}
	}

	#[test]
	fn a_value_past_what_the_format_reaches_clips_rather_than_becoming_an_infinity() {
		assert_eq!(half(1.0e30), 0x7BFF, "the largest finite value");
		assert_eq!(half(-1.0e30), 0xFBFF, "and its negative");
		assert_eq!(half(f32::INFINITY), 0x7BFF, "an infinity clips as well");
		assert_eq!(half(f32::NAN), 0x7BFF, "and so does a value that is not a number");
	}

	#[test]
	fn a_value_too_small_for_a_full_mantissa_is_stored_with_a_shorter_one() {
		// the smallest value the format holds with a full mantissa, and the
		// range below it where the exponent stops moving and the mantissa
		// shortens instead
		assert_eq!(half(6.103_515_6e-5), 0x0400, "the smallest normal value");
		assert_eq!(half(3.051_757_8e-5), 0x0200, "half of it, held with one bit fewer");
		assert_eq!(half(5.960_464_5e-8), 0x0001, "the smallest value of all");
		assert_eq!(half(1.0e-10), 0x0000, "and below that there is nothing left");
	}
}
