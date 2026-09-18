//! A texture read on the processor: the color of a surface a ray lands on.
//!
//! **One small level, not the picture.** A ray that lands on a wall stands for
//! a patch of it as wide as the ray's share of the hemisphere, which a few
//! meters off is far wider than a texel of the sharpest level; what that patch
//! sends back is its average color, and a small level of the chain already is
//! that average. The level read is the first no more than [`SIDE`] texels
//! either way, which costs a picture a few kilobytes however large it is.
//!
//! **Decoded once, into linear light.** An eight-bit color is stored on the
//! sRGB curve, and what bounces is light, so every texel is taken off the
//! curve when the picture is read rather than when a ray asks. The curve is a
//! power, which is one of the things two libraries may round apart; it is
//! worked out once for all 256 bytes in double precision and narrowed, which
//! agrees to the bit wherever the two libraries agree to within a part in a
//! thousand million.

use std::sync::OnceLock;

use colby_core::{
	abi::{
		material::Wrap,
		texture::{Texel, TextureData},
	},
	glam::{Vec2, Vec4},
};

use crate::sky::widened;

/// The most texels a side of the level read may have.
pub const SIDE: u32 = 64;

/// One level of a flat texture, decoded.
#[derive(Clone, Debug, PartialEq)]
pub struct Picture {
	/// Its width, in texels.
	width: u32,

	/// Its height.
	height: u32,

	/// Every texel, row after row, linear, with its alpha.
	texels: Vec<Vec4>,
}

impl Picture {
	/// The level of a texture a bake reads, decoded.
	///
	/// @param data - what the registry holds
	/// @return the picture, or `None` for a cube or a texture whose levels do
	/// not add up
	#[must_use]
	pub fn of(data: &TextureData) -> Option<Self> {
		if data.is_cube() || !data.is_consistent() {
			return None;
		}

		let count = u32::try_from(data.levels.len()).ok()?;
		let level = (0..count)
			.find(|&level| {
				let (width, height) = data.level_size(level);

				width.max(height) <= SIDE
			})
			.unwrap_or_else(|| count.saturating_sub(1));
		let (width, height) = data.level_size(level);
		let bytes = data.levels.get(usize::try_from(level).ok()?)?;
		let texels: Vec<Vec4> = bytes
			.chunks_exact(data.texel.bytes())
			.map(|texel| decoded(data.texel, texel))
			.collect();

		(texels.len() == usize::try_from(width).ok()? * usize::try_from(height).ok()?)
			.then_some(Self { width, height, texels })
	}

	/// The color at one place, blended between its four nearest texels.
	///
	/// @param uv - where, origin top left, a whole picture from nought to one
	/// @param wrap - whether a place past the edge wraps round or is held
	#[must_use]
	pub fn at(&self, uv: Vec2, wrap: Wrap) -> Vec4 {
		let width = f32::from(u16::try_from(self.width).unwrap_or(u16::MAX)).max(1.0);
		let height = f32::from(u16::try_from(self.height).unwrap_or(u16::MAX)).max(1.0);
		let (x, y) = (uv.x * width, uv.y * height);
		// the texel whose middle is at a place is the one it falls in, less a half
		let (x, y) = (x - 0.5, y - 0.5);
		let (left, top) = (x.floor(), y.floor());
		let (towards_x, towards_y) = (x - left, y - top);
		let texel = |column: f32, row: f32| {
			let column = place(column, self.width, wrap);
			let row = place(row, self.height, wrap);
			let index = row * usize::try_from(self.width).unwrap_or(0) + column;

			self.texels
				.get(index)
				.copied()
				.unwrap_or(Vec4::ONE)
		};

		let upper = texel(left, top).lerp(texel(left + 1.0, top), towards_x);
		let lower = texel(left, top + 1.0).lerp(texel(left + 1.0, top + 1.0), towards_x);

		upper.lerp(lower, towards_y)
	}
}

/// One texel as linear numbers.
fn decoded(layout: Texel, bytes: &[u8]) -> Vec4 {
	let byte = |at: usize| bytes.get(at).copied().unwrap_or(0);

	match layout {
		| Texel::Rgba8Srgb => Vec4::new(
			linear(byte(0)),
			linear(byte(1)),
			linear(byte(2)),
			f32::from(byte(3)) / 255.0,
		),
		| Texel::Rgba8Unorm => Vec4::new(
			f32::from(byte(0)) / 255.0,
			f32::from(byte(1)) / 255.0,
			f32::from(byte(2)) / 255.0,
			f32::from(byte(3)) / 255.0,
		),
		| Texel::Rgba16Float => {
			let channel =
				|at: usize| widened(u16::from_le_bytes([byte(at * 2), byte(at * 2 + 1)]));

			Vec4::new(channel(0), channel(1), channel(2), channel(3))
		},
	}
}

/// A byte on the sRGB curve as the linear light it stands for.
#[must_use]
pub fn linear(byte: u8) -> f32 {
	static CURVE: OnceLock<[f32; 256]> = OnceLock::new();

	CURVE.get_or_init(curve)[usize::from(byte)]
}

/// The whole curve, once.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "a value worked out in double precision and narrowed once, where it is stored"
)]
fn curve() -> [f32; 256] {
	std::array::from_fn(|index| {
		let value = f64::from(u8::try_from(index).unwrap_or(u8::MAX)) / 255.0;
		let light = if value <= 0.040_45 {
			value / 12.92
		} else {
			((value + 0.055) / 1.055).powf(2.4)
		};

		light as f32
	})
}

/// A column or a row, wrapped round the picture or held at its edge.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "taken modulo the side or clamped into it first, so the value is a whole number \
	          inside it"
)]
fn place(value: f32, side: u32, wrap: Wrap) -> usize {
	let last = side.saturating_sub(1);
	let placed = match wrap {
		| Wrap::Repeat => f64::from(value).rem_euclid(f64::from(side.max(1))),
		| Wrap::Clamp => f64::from(value).clamp(0.0, f64::from(last)),
	};

	(placed as usize).min(usize::try_from(last).unwrap_or(0))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A picture of two by two, colors given row after row.
	fn square(texel: Texel, texels: [[u8; 4]; 4]) -> TextureData {
		TextureData {
			width: 2,
			height: 2,
			faces: 1,
			texel,
			levels: vec![texels.concat(), vec![0, 0, 0, 0]],
		}
	}

	#[test]
	fn the_curve_is_the_standard_one_at_its_ends_and_its_joint() {
		assert_eq!(linear(0).to_bits(), 0.0_f32.to_bits(), "black");
		assert_eq!(linear(255).to_bits(), 1.0_f32.to_bits(), "white");
		assert!(
			(linear(10) - 10.0 / 255.0 / 12.92).abs() < 1.0e-9,
			"the straight part below the joint"
		);
		assert!(
			(linear(128) - 0.215_860_5).abs() < 1.0e-6,
			"the middle byte is about a fifth of the light: {}",
			linear(128)
		);
	}

	#[test]
	fn a_texel_middle_reads_that_texel_and_the_seam_between_reads_both() {
		let picture = Picture::of(&square(Texel::Rgba8Unorm, [
			[255, 0, 0, 255],
			[0, 255, 0, 255],
			[0, 0, 255, 255],
			[255, 255, 255, 255],
		]))
		.expect("a picture");

		assert_eq!(
			picture.at(Vec2::new(0.25, 0.25), Wrap::Clamp),
			Vec4::new(1.0, 0.0, 0.0, 1.0),
			"top left"
		);
		assert_eq!(picture.at(Vec2::new(0.75, 0.75), Wrap::Clamp), Vec4::ONE, "bottom right");
		assert_eq!(
			picture.at(Vec2::new(0.5, 0.25), Wrap::Clamp),
			Vec4::new(0.5, 0.5, 0.0, 1.0),
			"half way between the top two"
		);
	}

	#[test]
	fn repeating_wraps_the_far_edge_round_and_clamping_holds_it() {
		let picture = Picture::of(&square(Texel::Rgba8Unorm, [
			[255, 0, 0, 255],
			[0, 0, 255, 255],
			[255, 0, 0, 255],
			[0, 0, 255, 255],
		]))
		.expect("a picture");
		// at the right edge: half the right column and half of what lies past it
		let edge = Vec2::new(1.0, 0.5);

		assert_eq!(
			picture.at(edge, Wrap::Repeat),
			Vec4::new(0.5, 0.0, 0.5, 1.0),
			"past it is the left column"
		);
		assert_eq!(
			picture.at(edge, Wrap::Clamp),
			Vec4::new(0.0, 0.0, 1.0, 1.0),
			"past it is itself"
		);
		assert_eq!(
			picture.at(Vec2::new(1.25, 0.25), Wrap::Repeat),
			picture.at(Vec2::new(0.25, 0.25), Wrap::Repeat),
			"a whole picture over is the same place"
		);
	}

	#[test]
	fn a_color_is_taken_off_its_curve_and_a_number_is_not() {
		let color =
			Picture::of(&square(Texel::Rgba8Srgb, [[128, 128, 128, 128]; 4])).expect("a color");
		let number =
			Picture::of(&square(Texel::Rgba8Unorm, [[128, 128, 128, 128]; 4])).expect("a number");
		let middle = Vec2::splat(0.5);

		let bits = |value: f32| value.to_bits();

		assert_eq!(bits(color.at(middle, Wrap::Repeat).x), bits(linear(128)), "a color is light");
		assert_eq!(
			bits(number.at(middle, Wrap::Repeat).x),
			bits(128.0 / 255.0),
			"a number is itself"
		);
		assert_eq!(
			bits(color.at(middle, Wrap::Repeat).w),
			bits(128.0 / 255.0),
			"and an alpha is never on the curve"
		);
	}

	#[test]
	fn a_large_picture_is_read_at_a_small_level() {
		let mut data = TextureData {
			width: 256,
			height: 128,
			faces: 1,
			texel: Texel::Rgba8Unorm,
			levels: Vec::new(),
		};

		for level in 0..TextureData::full_chain(256, 128) {
			let bytes = data.level_bytes(level);

			data.levels.push(vec![200; bytes]);
		}

		let picture = Picture::of(&data).expect("a picture");

		assert_eq!(
			(picture.width, picture.height),
			(64, 32),
			"the first level no wider than {SIDE}"
		);
		assert!(Picture::of(&TextureData::white_cube()).is_none(), "and a cube is not a picture");
	}
}
