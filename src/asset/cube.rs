//! Turning one equirectangular picture into a prefiltered environment cube.
//!
//! **The whole of the expensive half of an environment map happens here, once,
//! offline.** That is the same bargain [`texture`](crate::texture) already
//! makes - a `.ctex` is what the GPU wants before anything loads it - and the
//! reason to keep it for this of all things is that the alternative costs a
//! compute pipeline, a set of storage views per level and a pass that has to
//! run before the first frame that reads it. None of that buys anything a game
//! here can use, because nothing here captures an environment at run time.
//!
//! ```text
//!   one .hdr, equirectangular      ->    six faces, a full mip chain
//!   level 0                              roughness 0, which is the picture
//!   level 1                              roughness 1/(levels-1)
//!   ...                                  ...
//!   level n-1                            roughness 1
//! ```
//!
//! **The mip chain is a roughness chain**, which is what makes one texture do
//! two jobs: the sky is drawn out of level 0, where the roughness is nought and
//! the filter is the identity, and a reflection reads the level its surface's
//! roughness names. There is no second texture and no second binding.
//!
//! **Every tap reads the source picture, never the cube being built.** That is
//! not an optimization, it is what makes the seams not exist: a direction that
//! crosses from one face to the next is two directions into one continuous
//! picture, so there is nothing to match up. A filter that read the cube would
//! have to do the cross-face lookup itself and would be wrong along twelve
//! edges.
//!
//! The integral is the standard one - the microfacet distribution sampled by
//! importance with a low-discrepancy sequence, weighted by how much the surface
//! faces each sample - and it is written here in the same arithmetic a second
//! answer can be written in away from this code, because that is how the result
//! gets checked. @ref [`radiance`](crate::radiance) for what is read in.

use colby_core::{
	Result,
	abi::texture::{CUBE_FACES, Texel, TextureData},
	err,
};

use crate::radiance::Radiance;

/// How many texels a face of the cube is, on a side.
///
/// A hundred and twenty-eight, which is what the two engines in the field that
/// offer a choice both start at. It is a constant rather than a setting because
/// a setting would have to live somewhere - a manifest beside every picture, or
/// a second rule about file names - and neither is worth having for a number
/// nobody has yet wanted to change.
pub const FACE_SIDE: u32 = 128;

/// How many samples each texel of each filtered level integrates.
///
/// Fixed rather than growing with the roughness, which is what two of the
/// engines in the field do: the level that would want the most samples is also
/// the smallest, so the saving is on the level that needs it least. One number
/// is also one number for a second answer to agree with.
pub const SAMPLES: u32 = 128;

/// The six faces, in the order the renderer's own cube arithmetic uses.
///
/// Each is the direction out of the middle of that face, with the two axes that
/// walk across it: right, then down. A direction picks its face by its largest
/// component, so the order here is `+x -x +y -y +z -z` and nothing may reorder
/// it without the shader agreeing.
const FACES: [Face; 6] = [
	Face {
		forward: [1.0, 0.0, 0.0],
		right: [0.0, 0.0, -1.0],
		down: [0.0, -1.0, 0.0],
	},
	Face {
		forward: [-1.0, 0.0, 0.0],
		right: [0.0, 0.0, 1.0],
		down: [0.0, -1.0, 0.0],
	},
	Face {
		forward: [0.0, 1.0, 0.0],
		right: [1.0, 0.0, 0.0],
		down: [0.0, 0.0, 1.0],
	},
	Face {
		forward: [0.0, -1.0, 0.0],
		right: [1.0, 0.0, 0.0],
		down: [0.0, 0.0, -1.0],
	},
	Face {
		forward: [0.0, 0.0, 1.0],
		right: [1.0, 0.0, 0.0],
		down: [0.0, -1.0, 0.0],
	},
	Face {
		forward: [0.0, 0.0, -1.0],
		right: [-1.0, 0.0, 0.0],
		down: [0.0, -1.0, 0.0],
	},
];

/// Which way one face of the cube looks and how its texels are laid out on it.
#[derive(Clone, Copy)]
struct Face {
	/// The direction out of the middle of it.
	forward: [f32; 3],

	/// The axis a column walks along.
	right: [f32; 3],

	/// The axis a row walks along.
	down: [f32; 3],
}

/// Builds the whole environment out of one equirectangular picture.
///
/// @param source - the picture, in linear light
/// @return a cube of [`FACE_SIDE`] with a full chain, or why it could not be
/// built
pub fn build(source: &Radiance) -> Result<TextureData> {
	if source.width < 4 || source.height < 2 {
		return Err(err!(Asset(
			"is {}x{}, which is too small to be a whole sky",
			source.width,
			source.height
		)));
	}

	let count = TextureData::full_chain(FACE_SIDE, FACE_SIDE);
	let last = f32::from(u16::try_from(count.saturating_sub(1)).unwrap_or(1)).max(1.0);
	let mut levels = Vec::with_capacity(usize::try_from(count).unwrap_or(1));

	for level in 0..count {
		let side = (FACE_SIDE >> level.min(31)).max(1);
		let roughness = f32::from(u16::try_from(level).unwrap_or(0)) / last;

		levels.push(filtered(source, side, roughness));
	}

	let data = TextureData {
		width: FACE_SIDE,
		height: FACE_SIDE,
		faces: CUBE_FACES,
		texel: Texel::Rgba16Float,
		levels,
	};

	if !data.is_consistent() {
		return Err(err!(Asset("built a cube whose levels do not add up")));
	}

	Ok(data)
}

/// One level: six faces of `side` square, each texel integrated at `roughness`.
fn filtered(source: &Radiance, side: u32, roughness: f32) -> Vec<u8> {
	let across = usize::try_from(side).unwrap_or(1);
	let mut out = Vec::with_capacity(across * across * 6 * 8);

	for face in FACES {
		one_face(source, &face, side, roughness, &mut out);
	}

	out
}

/// One face of one level, appended to the level's bytes.
fn one_face(source: &Radiance, face: &Face, side: u32, roughness: f32, out: &mut Vec<u8>) {
	for y in 0..side {
		for x in 0..side {
			let normal = direction(face, side, x, y);
			let color = if roughness <= 0.0 {
				sample(source, normal)
			} else {
				convolve(source, normal, roughness)
			};

			for channel in color {
				out.extend_from_slice(&half(channel).to_le_bytes());
			}

			// opaque, because the alpha of a sky means nothing and a
			// three-channel layout would be a fourth texel size to carry
			out.extend_from_slice(&half(1.0).to_le_bytes());
		}
	}
}

/// The direction out of the middle of one texel of one face.
///
/// @param forward - the direction out of the middle of the face
/// @param right - the axis that walks across it
/// @param down - the axis that walks down it
/// @param side - how many texels the face is on a side
/// @param x - the column
/// @param y - the row
fn direction(face: &Face, side: u32, x: u32, y: u32) -> [f32; 3] {
	let span = f32::from(u16::try_from(side).unwrap_or(1)).max(1.0);
	let along = middle(x, span);
	let deep = middle(y, span);

	normalize(added(face.forward, added(scaled(face.right, along), scaled(face.down, deep))))
}

/// Where one texel's middle falls across a face, from minus one to one.
fn middle(index: u32, span: f32) -> f32 {
	let fraction = (f32::from(u16::try_from(index).unwrap_or(0)) + 0.5) / span;

	fraction * 2.0 - 1.0
}

/// The environment as a surface of this roughness facing this way sees it.
///
/// The view direction is taken to be the normal, which is the approximation
/// every prefiltered environment in the field makes: the alternative is a
/// fourth dimension in the texture, because what a surface reflects depends on
/// where it is looked at from as well as where it faces.
fn convolve(source: &Radiance, normal: [f32; 3], roughness: f32) -> [f32; 3] {
	let alpha = roughness * roughness;
	let (right, up) = basis(normal);
	let mut total = [0.0_f32; 3];
	let mut weight = 0.0_f32;

	for index in 0..SAMPLES {
		let (first, second) = hammersley(index, SAMPLES);
		let facet = half_vector(first, second, alpha, normal, right, up);
		// the reflection of the view - which is the normal here - about the
		// microfacet's own normal
		let twice = scaled(facet, 2.0 * dot(normal, facet));
		let light = normalize(added(twice, scaled(normal, -1.0)));
		let facing = dot(normal, light);

		if facing <= 0.0 {
			continue;
		}

		let weighted = scaled(sample(source, light), facing);
		total = added(total, weighted);
		weight += facing;
	}

	if weight <= 0.0 {
		return sample(source, normal);
	}

	[total[0] / weight, total[1] / weight, total[2] / weight]
}

/// One microfacet normal out of the distribution, for a pair in `0..1`.
fn half_vector(
	first: f32,
	second: f32,
	alpha: f32,
	normal: [f32; 3],
	right: [f32; 3],
	up: [f32; 3],
) -> [f32; 3] {
	let phi = 2.0 * std::f32::consts::PI * first;
	// the distribution's own inverse, written in two steps so nothing folds
	// into a fused instruction a second answer could not fold. @ref `dot`.
	let squared = alpha * alpha;
	let spread = (squared - 1.0) * second;
	let denominator = 1.0 + spread;
	let cosine = ((1.0 - second) / denominator.max(1.0e-8))
		.max(0.0)
		.sqrt();
	let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
	let (along, across) = (phi.cos() * sine, phi.sin() * sine);

	normalize(added(scaled(right, along), added(scaled(up, across), scaled(normal, cosine))))
}

/// Two directions added together.
fn added(first: [f32; 3], second: [f32; 3]) -> [f32; 3] {
	std::array::from_fn(|axis| {
		first.get(axis).copied().unwrap_or(0.0) + second.get(axis).copied().unwrap_or(0.0)
	})
}

/// A direction stretched by a number.
fn scaled(way: [f32; 3], amount: f32) -> [f32; 3] {
	std::array::from_fn(|axis| way.get(axis).copied().unwrap_or(0.0) * amount)
}

/// Two axes across a direction, picked so neither is close to parallel with it.
fn basis(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
	let helper = if normal[2].abs() < 0.999 {
		[0.0, 0.0, 1.0]
	} else {
		[1.0, 0.0, 0.0]
	};
	let right = normalize(cross(helper, normal));

	(right, cross(normal, right))
}

/// The low-discrepancy pair at one place in a sequence of `count`.
///
/// The second coordinate is the index's bits reversed, read as a fraction: a
/// sequence whose samples spread themselves rather than clumping, which is what
/// lets a hundred and twenty-eight of them stand in for many more.
fn hammersley(index: u32, count: u32) -> (f32, f32) {
	let first = f32::from(u16::try_from(index).unwrap_or(0))
		/ f32::from(u16::try_from(count).unwrap_or(1)).max(1.0);

	(first, reversed(index))
}

/// One index's bits reversed, read as a fraction of one.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "a fraction of one worked out in double precision and narrowed, which is the one 	          conversion the arithmetic is for"
)]
fn reversed(index: u32) -> f32 {
	// 2^-32, which is what turns the reversed bits into a fraction
	(f64::from(index.reverse_bits()) * 2.328_306_436_538_696_3e-10) as f32
}

/// The source picture in one direction, filtered between its four nearest
/// texels.
///
/// The picture is equirectangular: the horizontal coordinate is the angle
/// around, wrapped, and the vertical one is the angle from straight up, held at
/// both poles. A direction of no length answers black rather than dividing by
/// nothing.
fn sample(source: &Radiance, way: [f32; 3]) -> [f32; 3] {
	let way = normalize(way);
	let around = way[0].atan2(-way[2]);
	let up = way[1].clamp(-1.0, 1.0).acos();
	let across = (around / (2.0 * std::f32::consts::PI) + 0.5)
		.rem_euclid(1.0)
		.clamp(0.0, 1.0);
	let down = (up / std::f32::consts::PI).clamp(0.0, 1.0);

	let width = f32::from(u16::try_from(source.width).unwrap_or(1)).max(1.0);
	let height = f32::from(u16::try_from(source.height).unwrap_or(1)).max(1.0);
	let x = across.mul_add(width, -0.5);
	let y = down.mul_add(height, -0.5);
	let (low_x, low_y) = (x.floor(), y.floor());
	let (fraction_x, fraction_y) = (x - low_x, y - low_y);

	let column = |offset: f32| wrapped(low_x + offset, source.width);
	let row = |offset: f32| held(low_y + offset, source.height);

	let (left, right) = (column(0.0), column(1.0));
	let (top, bottom) = (row(0.0), row(1.0));

	mix(
		mix(source.at(left, top), source.at(right, top), fraction_x),
		mix(source.at(left, bottom), source.at(right, bottom), fraction_x),
		fraction_y,
	)
}

/// A column, wrapped around the picture: the sky joins up behind the camera.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "taken modulo the width first, so the value is a whole number inside it"
)]
fn wrapped(value: f32, width: u32) -> u32 {
	let span = f64::from(width).max(1.0);
	let wrapped = f64::from(value).rem_euclid(span);

	(wrapped as u32).min(width.saturating_sub(1))
}

/// A row, held at the poles: above straight up there is no more sky.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "clamped into the height first, so the value is a whole number inside it"
)]
fn held(value: f32, height: u32) -> u32 {
	let last = f64::from(height.saturating_sub(1));

	(f64::from(value).clamp(0.0, last) as u32).min(height.saturating_sub(1))
}

/// Between two colors.
fn mix(from: [f32; 3], to: [f32; 3], amount: f32) -> [f32; 3] {
	let step = std::array::from_fn(|axis| {
		(to.get(axis).copied().unwrap_or(0.0) - from.get(axis).copied().unwrap_or(0.0)) * amount
	});

	added(from, step)
}

/// A direction of length one, or straight up for one of no length at all.
fn normalize(way: [f32; 3]) -> [f32; 3] {
	let length = dot(way, way).sqrt();

	if length <= 1.0e-20 {
		return [0.0, 1.0, 0.0];
	}

	scaled(way, 1.0 / length)
}

/// How much two directions agree.
///
/// Summed over an iterator rather than written as three products added
/// together, so that nothing here folds a multiply and an add into one
/// instruction: a second answer working the same integral out elsewhere cannot
/// fold them, and the two agreeing to the last bit is worth more than the
/// instruction.
fn dot(first: [f32; 3], second: [f32; 3]) -> f32 {
	first
		.iter()
		.zip(second)
		.map(|(left, right)| left * right)
		.sum()
}

/// A direction at right angles to two others.
fn cross(first: [f32; 3], second: [f32; 3]) -> [f32; 3] {
	std::array::from_fn(|axis| {
		let next = (axis + 1) % 3;
		let last = (axis + 2) % 3;
		let (ahead, behind) = (first[next] * second[last], first[last] * second[next]);

		ahead - behind
	})
}

/// One value as the sixteen bits the format stores.
///
/// Rust has no half-precision type that is stable, so the bits are assembled:
/// the sign, the exponent held inside what the format reaches, and the top ten
/// bits of the mantissa, rounded to nearest with ties going to even - which is
/// what every other encoder of this format does, and therefore what a second
/// answer will agree with.
fn half(value: f32) -> u16 {
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

	/// Whether three channels are the same to the last bit that matters.
	fn same(got: [f32; 3], wanted: [f32; 3]) -> bool {
		got.iter()
			.zip(wanted)
			.all(|(left, right)| (left - right).abs() <= 1.0e-9)
	}

	/// The same, to what a sixteen-bit value can hold rather than exactly.
	fn near(got: [f32; 3], wanted: [f32; 3]) -> bool {
		got.iter()
			.zip(wanted)
			.all(|(left, right)| (left - right).abs() < 2.0e-3)
	}

	/// Paints one texel of a picture, all three channels.
	fn paint(source: &mut Radiance, x: u32, y: u32, value: f32) {
		let at = (usize::try_from(y).unwrap_or(0) * usize::try_from(source.width).unwrap_or(0)
			+ usize::try_from(x).unwrap_or(0))
			* 3;

		for slot in source
			.texels
			.get_mut(at..at + 3)
			.unwrap_or_default()
		{
			*slot = value;
		}
	}

	/// A picture of one color everywhere.
	fn flat(value: [f32; 3]) -> Radiance {
		Radiance {
			width: 8,
			height: 4,
			texels: value
				.iter()
				.copied()
				.cycle()
				.take(8 * 4 * 3)
				.collect(),
		}
	}

	/// One texel of one face of one level, as three floats.
	fn texel_of(data: &TextureData, level: usize, face: u32, x: u32, y: u32) -> [f32; 3] {
		let side = (data.width >> u32::try_from(level).unwrap_or(0).min(31)).max(1);
		let across = usize::try_from(side).unwrap_or(1);
		let at = (usize::try_from(face).unwrap_or(0) * across * across
			+ usize::try_from(y).unwrap_or(0) * across
			+ usize::try_from(x).unwrap_or(0))
			* 8;
		let bytes = data
			.levels
			.get(level)
			.and_then(|level| level.get(at..at + 8))
			.expect("the texel is inside the level");

		std::array::from_fn(|channel| {
			let pair = bytes
				.get(channel * 2..channel * 2 + 2)
				.and_then(|slice| <[u8; 2]>::try_from(slice).ok())
				.unwrap_or([0, 0]);

			widened(u16::from_le_bytes(pair))
		})
	}

	/// One of the format's sixteen-bit values back as a float.
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
	fn a_flat_sky_filters_to_the_same_color_at_every_level_and_every_face() {
		let built = build(&flat([0.25, 0.5, 0.75])).expect("a flat sky");

		assert_eq!(built.faces, CUBE_FACES, "six faces");
		assert_eq!((built.width, built.height), (FACE_SIDE, FACE_SIDE), "of the fixed side");
		assert_eq!(
			built.levels.len(),
			usize::try_from(TextureData::full_chain(FACE_SIDE, FACE_SIDE))
				.expect("a short chain"),
			"and a full chain, because the chain is the roughness"
		);

		for level in 0..built.levels.len() {
			for face in 0..CUBE_FACES {
				let read = texel_of(&built, level, face, 0, 0);

				assert!(
					near(read, [0.25, 0.5, 0.75]),
					"level {level} face {face} is {read:?}: a filter of a constant is that \n					 constant, whatever the roughness"
				);
			}
		}
	}

	#[test]
	fn the_first_level_is_the_picture_itself_rather_than_a_filter_of_it() {
		// a sky that is bright straight up and dark straight down, so the two
		// faces along that axis are far apart and a filter of either would
		// fetch the other
		let mut source = flat([0.0; 3]);
		for column in 0..source.width {
			paint(&mut source, column, 0, 8.0);
		}

		let built = build(&source).expect("a sky with a top to it");
		let up = texel_of(&built, 0, 2, FACE_SIDE / 2, FACE_SIDE / 2);
		let down = texel_of(&built, 0, 3, FACE_SIDE / 2, FACE_SIDE / 2);

		assert!(
			up[0] > 4.0,
			"the middle of the +y face looks straight up, and up is bright: {up:?}"
		);
		assert!(
			down[0] < 1.0,
			"and the middle of -y looks straight down, which is not: {down:?}"
		);
	}

	#[test]
	fn a_level_further_down_the_chain_gathers_from_further_around() {
		// bright along the top row of the picture only, which is the sky within
		// forty-five degrees of straight up
		let mut source = flat([0.0; 3]);
		for column in 0..source.width {
			paint(&mut source, column, 0, 4.0);
		}

		let built = build(&source).expect("a sky with a bright band");
		let last = built.levels.len() - 1;
		// the middle of the +x face looks along the horizon, where the picture
		// is black
		let sharp = texel_of(&built, 0, 0, FACE_SIDE / 2, FACE_SIDE / 2);
		let rough = texel_of(&built, last, 0, 0, 0);

		assert!(
			same(sharp, [0.0; 3]),
			"a roughness of nought reads one direction, and it is dark"
		);
		assert!(
			rough[0] > 0.05,
			"and a roughness of one gathers the hemisphere around it, which reaches the bright 			 band: {rough:?}"
		);
		assert!(
			rough[0] < 4.0,
			"but it is an average over that hemisphere rather than the band itself: {rough:?}"
		);
	}

	#[test]
	fn a_picture_too_small_to_be_a_sky_is_refused_rather_than_stretched() {
		let tiny = Radiance {
			width: 2,
			height: 1,
			texels: vec![0.0; 6],
		};

		assert!(build(&tiny).is_err(), "two texels across is not a sky");
	}

	#[test]
	fn the_six_faces_are_in_the_order_the_renderer_picks_them_in() {
		for (index, face) in FACES.into_iter().enumerate() {
			let corner = direction(&face, 2, 0, 0);

			assert!(
				dot(corner, face.forward) > 0.5,
				"face {index} looks the way it says it does"
			);
			assert!(
				dot(face.forward, face.right).abs() < 1.0e-6,
				"and its axes are at right angles"
			);
			assert!(dot(face.forward, face.down).abs() < 1.0e-6, "both of them");
			assert!(dot(face.right, face.down).abs() < 1.0e-6, "and to each other");
		}
	}

	#[test]
	fn the_sequence_spreads_its_samples_rather_than_clumping_them() {
		let mut halves = 0;

		for index in 0..SAMPLES {
			let (_, second) = hammersley(index, SAMPLES);

			assert!((0.0..1.0).contains(&second), "{index} is inside the unit interval");

			if second < 0.5 {
				halves += 1;
			}
		}

		assert_eq!(
			halves,
			SAMPLES / 2,
			"exactly half of them fall in the lower half, which a clumping sequence would not"
		);
	}

	#[test]
	fn the_sky_joins_up_behind_the_camera_and_holds_at_the_poles() {
		let source = flat([1.0, 2.0, 3.0]);

		assert!(same(sample(&source, [0.0, 0.0, -1.0]), [1.0, 2.0, 3.0]), "one way");
		assert!(same(sample(&source, [0.0, 0.0, 1.0]), [1.0, 2.0, 3.0]), "and the other");
		assert!(same(sample(&source, [0.0, 1.0, 0.0]), [1.0, 2.0, 3.0]), "straight up");
		assert!(same(sample(&source, [0.0, -1.0, 0.0]), [1.0, 2.0, 3.0]), "and straight down");
		assert!(
			same(sample(&source, [0.0; 3]), [1.0, 2.0, 3.0]),
			"a direction of no length is up"
		);
	}
}
