//! Reading PNG, the texture import format.
//!
//! **Why a crate here and a hand-written encoder in `colby_engine`.** The
//! encoder there writes one stored deflate block - sixty lines of arithmetic
//! against pulling a compressor into a renderer. Reading one means a real
//! inflate, which is a different size of problem, and it runs *offline*: a
//! `.ctex` is already what the GPU wants, so nothing in the engine or the
//! runner links this. The dependency lands where it costs least.
//!
//! Everything comes out as eight-bit RGBA, whatever the file was. Palettes and
//! low bit depths are expanded by the decoder; grayscale and missing alpha are
//! expanded here, because the decoder will normalize a depth but will not
//! invent a channel.
//!
//! **What the bytes mean is not in the file.** A PNG carries no answer to "is
//! this a color or a direction", and neither does anything the decoder returns,
//! so the caller says - and the answer changes both the format the GPU is given
//! and how the mip chain is built. @ref `crate::compile` for how the compiler
//! decides.
//!
//! Sixteen-bit images are read down to eight. That is a real loss, and it is
//! the right one until there is a texel layout that could hold them - @ref
//! [`Texel`](colby_core::abi::Texel).

use std::{fs, io::Cursor, path::Path};

use colby_core::{
	Result,
	abi::texture::{Texel, TextureData},
	err,
};
use png::{ColorType, Decoder, Transformations};

use crate::texture::{MAX_SIZE, build_chain};

/// The extension this importer claims.
pub const EXTENSION: &str = "png";

/// Reads a PNG file into a texture and its mip chain.
///
/// @param path - the `.png` to read
/// @param texel - what the channels are to be taken as
/// @return the texture, or why it could not be read
pub fn import_file(path: &Path, texel: Texel) -> Result<TextureData> {
	let bytes =
		fs::read(path).map_err(|error| err!(Asset("reading {}: {error}", path.display())))?;

	import(&bytes, texel).map_err(|error| err!(Asset("{}: {error}", path.display())))
}

/// Reads PNG bytes into a texture and its mip chain.
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

/// Decodes to eight-bit RGBA, whatever the file held.
fn decode(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
	// wrapped in a cursor because the decoder wants to seek: PNG 0.18 reads the
	// header, decides what transformations apply, and goes back for the data.
	let mut decoder = Decoder::new(Cursor::new(bytes));

	// expands a palette to color and a low bit depth to eight, and takes a
	// sixteen-bit image down to eight. It does not add an alpha channel, which
	// is why the match below exists.
	decoder.set_transformations(Transformations::normalize_to_color8());

	let mut reader = decoder
		.read_info()
		.map_err(|error| err!(Asset("not a png this build can read: {error}")))?;

	let (width, height) = (reader.info().width, reader.info().height);
	if width == 0 || height == 0 {
		return Err(err!(Asset("is {width}x{height}, which is not an image")));
	}

	if width > MAX_SIZE || height > MAX_SIZE {
		return Err(err!(Asset(
			"is {width}x{height}, past the {MAX_SIZE} the texture format will hold"
		)));
	}

	let size = reader
		.output_buffer_size()
		.ok_or_else(|| err!(Asset("declares a size that does not fit in memory")))?;

	let mut buffer = vec![0; size];
	let info = reader
		.next_frame(&mut buffer)
		.map_err(|error| err!(Asset("decoding failed: {error}")))?;

	buffer.truncate(info.buffer_size());

	let rgba = expand(&buffer, info.color_type, width, height)?;

	Ok((width, height, rgba))
}

/// Widens whatever channels the file had into four.
fn expand(bytes: &[u8], color: ColorType, width: u32, height: u32) -> Result<Vec<u8>> {
	let count = usize::try_from(width)
		.ok()
		.and_then(|width| width.checked_mul(usize::try_from(height).ok()?))
		.ok_or_else(|| err!(Asset("is too large to hold in memory")))?;

	let source = color.samples();
	if bytes.len() < count * source {
		return Err(err!(Asset(
			"decoded to {} bytes, and {width}x{height} of {source}-channel texels is {}",
			bytes.len(),
			count * source
		)));
	}

	let mut rgba = Vec::with_capacity(count * 4);
	for texel in bytes.chunks_exact(source).take(count) {
		let (red, green, blue, alpha) = match color {
			| ColorType::Grayscale => (texel[0], texel[0], texel[0], 0xFF),
			| ColorType::GrayscaleAlpha => (texel[0], texel[0], texel[0], texel[1]),
			| ColorType::Rgb => (texel[0], texel[1], texel[2], 0xFF),
			| ColorType::Rgba => (texel[0], texel[1], texel[2], texel[3]),
			| ColorType::Indexed =>
				return Err(err!(Asset("still has a palette after being asked to expand one"))),
		};

		rgba.extend([red, green, blue, alpha]);
	}

	Ok(rgba)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A two-by-two truecolor PNG with no alpha: red, green over blue, white.
	///
	/// Written by an encoder that is not this project's, and pasted here as
	/// bytes, so that the test does not rest on colby's own PNG writer and this
	/// reader agreeing with each other about anything.
	const RGB_QUAD: [u8; 77] = [
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
		0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x08, 0x02, 0x00, 0x00, 0x00, 0xFD,
		0xD4, 0x9A, 0x73, 0x00, 0x00, 0x00, 0x14, 0x49, 0x44, 0x41, 0x54, 0x78, 0xDA, 0x63, 0xF8,
		0xCF, 0xC0, 0xC0, 0x00, 0xC2, 0x0C, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x00, 0x1F, 0xEE, 0x05,
		0xFB, 0x60, 0x6C, 0x70, 0xF2, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42,
		0x60, 0x82,
	];

	/// The same size in grayscale, with no alpha either: black, white over
	/// white, black. Two channels this reader has to invent.
	const GREY_QUAD: [u8; 71] = [
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
		0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x08, 0x00, 0x00, 0x00, 0x00, 0x57,
		0xDD, 0x52, 0xF8, 0x00, 0x00, 0x00, 0x0E, 0x49, 0x44, 0x41, 0x54, 0x78, 0xDA, 0x63, 0x60,
		0xF8, 0xCF, 0xF0, 0x9F, 0x01, 0x00, 0x06, 0x00, 0x01, 0xFF, 0x92, 0x99, 0xB2, 0xEC, 0x00,
		0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
	];

	#[test]
	fn a_png_with_no_alpha_channel_comes_out_with_one() {
		let texture = import(&RGB_QUAD, Texel::Rgba8Srgb).expect("the quad decodes");

		assert_eq!((texture.width, texture.height), (2, 2), "two by two");
		assert_eq!(texture.texel, Texel::Rgba8Srgb, "widened to four channels");
		assert_eq!(
			texture.levels[0],
			vec![
				0xFF, 0x00, 0x00, 0xFF, // red
				0x00, 0xFF, 0x00, 0xFF, // green
				0x00, 0x00, 0xFF, 0xFF, // blue
				0xFF, 0xFF, 0xFF, 0xFF, // white
			],
			"in reading order, each made opaque"
		);
	}

	#[test]
	fn a_grayscale_png_comes_out_with_three_color_channels() {
		let texture = import(&GREY_QUAD, Texel::Rgba8Srgb).expect("the grey quad decodes");

		assert_eq!(
			texture.levels[0],
			vec![
				0x00, 0x00, 0x00, 0xFF, // black
				0xFF, 0xFF, 0xFF, 0xFF, // white
				0xFF, 0xFF, 0xFF, 0xFF, // white
				0x00, 0x00, 0x00, 0xFF, // black
			],
			"the one channel is replicated across three"
		);
	}

	#[test]
	fn a_chain_is_built_on_the_way_in() {
		let texture = import(&RGB_QUAD, Texel::Rgba8Srgb).expect("the quad decodes");

		assert_eq!(texture.levels.len(), 2, "two by two has a level below it");
		assert!(texture.is_consistent(), "and every level is its right size");
		assert_eq!(texture.levels[1].len(), 4, "which is one texel");
	}

	#[test]
	fn the_smallest_level_is_the_average_of_the_image() {
		let texture = import(&GREY_QUAD, Texel::Rgba8Srgb).expect("the grey quad decodes");
		let average = texture.levels[1][0];

		// half black, half white: mid grey in light, which is about 188 as an
		// sRGB byte rather than the 127 that averaging bytes would give.
		assert!(average.abs_diff(188) <= 2, "averaged in linear light, got {average}");
	}

	#[test]
	fn something_that_is_not_a_png_is_refused() {
		let error =
			import(b"this is not a png", Texel::Rgba8Srgb).expect_err("nor will it become one");

		assert!(error.to_string().contains("png"), "and it says so: {error}");
	}

	#[test]
	fn a_truncated_png_is_refused_rather_than_read_short() {
		let error =
			import(&RGB_QUAD[..40], Texel::Rgba8Srgb).expect_err("half a png is not a png");

		assert!(!error.to_string().is_empty(), "with a reason: {error}");
	}

	#[test]
	fn a_file_that_does_not_exist_names_itself() {
		let path = std::env::temp_dir().join("colby-no-such-texture.png");
		let error = import_file(&path, Texel::Rgba8Srgb).expect_err("there is nothing to read");

		assert!(
			error
				.to_string()
				.contains("colby-no-such-texture.png"),
			"{error}"
		);
	}
}
