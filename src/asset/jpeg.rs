//! Reading JPEG, the other texture import format.
//!
//! Here for one reason: the glTF specification allows a picture to be a PNG or
//! a JPEG and says nothing about preferring either, so a model handed over by
//! somebody else carries whichever their tool wrote. Refusing one of the two
//! means half of those models arrive with no textures on them at all.
//!
//! Everything else about this module is [`png`](crate::png)'s argument
//! repeated: the decoding happens **offline**, in the compiler, because a
//! `.ctex` already holds what the GPU takes, so nothing in the engine or the
//! runner links a JPEG decoder. And what the channels *mean* is not in the
//! file, so the caller says - @ref `crate::compile` for how a loose file is
//! judged and `crate::gltf` for how a material says it outright.
//!
//! **A JPEG has no alpha and never will.** Every pixel comes out opaque, which
//! is not a limitation of this reader but of the format; a picture that needs a
//! cut-out is a PNG. Grayscale is expanded to three channels here, the way
//! `png` does it and for the same reason: the decoder normalizes a depth and
//! will not invent a channel.
//!
//! **What is refused, by name: anything that is not gray or color.** A
//! print-shop JPEG in four inks is a different color space with a convention
//! about inversion that not even the tools agree on, and guessing at it makes a
//! picture that is subtly wrong everywhere rather than one that is missing.

use std::{fs, io::Cursor, path::Path};

use colby_core::{
	Result,
	abi::texture::{Texel, TextureData},
	err,
};
use jpeg_decoder::{Decoder, PixelFormat};

use crate::texture::{MAX_SIZE, build_chain};

/// The extension this importer claims.
pub const EXTENSION: &str = "jpg";

/// The other spelling of it, which is the one the specification writes.
pub const LONG_EXTENSION: &str = "jpeg";

/// Reads a JPEG file into a texture and its mip chain.
///
/// @param path - the `.jpg` to read
/// @param texel - what the channels are to be taken as
/// @return the texture, or why it could not be read
pub fn import_file(path: &Path, texel: Texel) -> Result<TextureData> {
	let bytes =
		fs::read(path).map_err(|error| err!(Asset("reading {}: {error}", path.display())))?;

	import(&bytes, texel).map_err(|error| err!(Asset("{}: {error}", path.display())))
}

/// Reads JPEG bytes into a texture and its mip chain.
///
/// @param bytes - the whole file
/// @param texel - what the channels are to be taken as
/// @return the texture, or why it could not be read
pub fn import(bytes: &[u8], texel: Texel) -> Result<TextureData> {
	let (width, height, rgba) = decode(bytes)?;

	Ok(TextureData {
		width,
		height,
		texel,
		levels: build_chain(width, height, rgba, texel)?,
	})
}

/// Decodes a JPEG into eight-bit RGBA.
fn decode(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
	let mut decoder = Decoder::new(Cursor::new(bytes));

	decoder
		.read_info()
		.map_err(|error| err!(Asset("not a jpeg this build can read: {error}")))?;

	let info = decoder
		.info()
		.ok_or_else(|| err!(Asset("says nothing about its own size")))?;
	let (width, height) = (u32::from(info.width), u32::from(info.height));

	if width == 0 || height == 0 {
		return Err(err!(Asset("is {width}x{height}, which is not an image")));
	}

	if width > MAX_SIZE || height > MAX_SIZE {
		return Err(err!(Asset(
			"is {width}x{height}, past the {MAX_SIZE} the texture format will hold"
		)));
	}

	let pixels = decoder
		.decode()
		.map_err(|error| err!(Asset("decoding failed: {error}")))?;

	expand(&pixels, info.pixel_format, width, height)
}

/// Turns whatever the decoder produced into opaque RGBA.
fn expand(
	pixels: &[u8],
	format: PixelFormat,
	width: u32,
	height: u32,
) -> Result<(u32, u32, Vec<u8>)> {
	let count = usize::try_from(width)
		.ok()
		.and_then(|width| {
			usize::try_from(height)
				.ok()
				.and_then(|h| width.checked_mul(h))
		})
		.ok_or_else(|| err!(Asset("is larger than this machine can hold")))?;
	let lanes = match format {
		| PixelFormat::L8 => 1,
		| PixelFormat::L16 => 2,
		| PixelFormat::RGB24 => 3,
		| PixelFormat::CMYK32 => {
			return Err(err!(Asset(
				"is in four inks rather than in light, which colby does not convert; re-save it \
				 as color"
			)));
		},
	};

	if pixels.len() < count * lanes {
		return Err(err!(Asset("decoded to fewer pixels than it said it had")));
	}

	let mut rgba = Vec::with_capacity(count * 4);

	for pixel in 0..count {
		let at = pixel * lanes;
		let (red, green, blue) = match format {
			| PixelFormat::L8 => (pixels[at], pixels[at], pixels[at]),
			// twelve bits of gray, read down to its top eight - the same loss a
			// sixteen-bit PNG takes, and for the same reason: there is no texel
			// layout that could hold the rest.
			| PixelFormat::L16 => {
				let gray = u16::from_ne_bytes([pixels[at], pixels[at + 1]]).to_le_bytes()[1];

				(gray, gray, gray)
			},
			| PixelFormat::RGB24 => (pixels[at], pixels[at + 1], pixels[at + 2]),
			| PixelFormat::CMYK32 => (0, 0, 0),
		};

		rgba.extend_from_slice(&[red, green, blue, 0xFF]);
	}

	Ok((width, height, rgba))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A thirty-two square checker in color, written by a tool that is not this
	/// one.
	const COLOR: &[u8] = include_bytes!("gltf/fixtures/tiles.jpg");

	/// The same checker with no color in it at all, so the one-channel path is
	/// read rather than assumed.
	const GRAY: &[u8] = include_bytes!("gltf/fixtures/gray.jpg");

	/// How far a decoded pixel may sit from the value that was drawn.
	///
	/// This format throws information away on purpose, so an exact comparison
	/// would be a test of the encoder's mood. Well inside a tile the error is a
	/// few counts; at an edge it is ringing, which is why nothing here samples
	/// one.
	const NEAR: i32 = 12;

	/// One pixel of a decoded image.
	fn pixel(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
		let at = usize::try_from((y * width + x) * 4).expect("the offset fits");

		[rgba[at], rgba[at + 1], rgba[at + 2], rgba[at + 3]]
	}

	/// Whether two colors are as close as this format allows.
	fn close(read: [u8; 4], drawn: [u8; 3]) -> bool {
		read.iter().zip(drawn).all(|(read, drawn)| {
			i32::from(*read).abs_diff(i32::from(drawn)) <= NEAR.unsigned_abs()
		})
	}

	#[test]
	fn a_color_jpeg_comes_out_as_the_picture_that_was_drawn() {
		let (width, height, rgba) = decode(COLOR).expect("it decodes");

		assert_eq!((width, height), (32, 32));
		assert_eq!(rgba.len(), 32 * 32 * 4);
		assert!(
			close(pixel(&rgba, width, 4, 4), [120, 115, 110]),
			"the dark tile came out {:?}",
			pixel(&rgba, width, 4, 4)
		);
		assert!(
			close(pixel(&rgba, width, 12, 4), [200, 190, 175]),
			"the light one came out {:?}",
			pixel(&rgba, width, 12, 4)
		);
	}

	#[test]
	fn every_pixel_of_a_jpeg_is_opaque_because_the_format_has_no_alpha() {
		let (.., rgba) = decode(COLOR).expect("it decodes");

		assert!(rgba.chunks_exact(4).all(|pixel| pixel[3] == 0xFF));
	}

	#[test]
	fn a_gray_jpeg_is_expanded_into_three_channels_that_agree() {
		let (width, height, rgba) = decode(GRAY).expect("it decodes");

		assert_eq!((width, height), (32, 32));

		for pixel in rgba.chunks_exact(4) {
			assert_eq!(pixel[0], pixel[1], "gray means the channels are one number");
			assert_eq!(pixel[1], pixel[2]);
		}

		let light = pixel(&rgba, width, 12, 4);

		assert!(close(light, [230, 230, 230]), "and it is still the picture: {light:?}");
	}

	#[test]
	fn what_the_channels_mean_is_the_callers_to_say() {
		let color = import(COLOR, Texel::Rgba8Srgb).expect("it reads");
		let numbers = import(COLOR, Texel::Rgba8Unorm).expect("and reads again");

		assert_eq!(color.texel, Texel::Rgba8Srgb);
		assert_eq!(numbers.texel, Texel::Rgba8Unorm);
		// the two ways of averaging only disagree where neighbors differ, and
		// this picture's tiles are eight across - so every block of the first
		// few levels is one flat color and the two agree exactly. The last
		// level is the whole image in one texel, which is where they cannot.
		assert_ne!(
			color.levels[5], numbers.levels[5],
			"a color is averaged in light and a number is averaged as it is stored"
		);
	}

	#[test]
	fn a_picture_arrives_with_its_whole_chain() {
		let data = import(COLOR, Texel::Rgba8Srgb).expect("it reads");

		assert_eq!(data.levels.len(), 6, "thirty-two down to one");
		assert_eq!(data.levels[0].len(), 32 * 32 * 4);
		assert_eq!(data.levels[5].len(), 4);
	}

	#[test]
	fn something_that_is_not_a_jpeg_is_refused_by_name() {
		let message = import(b"not a picture at all", Texel::Rgba8Srgb)
			.expect_err("it is refused")
			.to_string();

		assert!(message.contains("not a jpeg"), "got {message}");
	}

	#[test]
	fn a_truncated_jpeg_is_refused_rather_than_read_as_half_a_picture() {
		let message = import(&COLOR[..COLOR.len() / 2], Texel::Rgba8Srgb)
			.expect_err("it is refused")
			.to_string();

		assert!(
			message.contains("decoding failed") || message.contains("not a jpeg"),
			"got {message}"
		);
	}
}
