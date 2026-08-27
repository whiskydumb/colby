//! An image in memory, and the smallest PNG writer that can put it on disk.
//!
//! @note: hand-written rather than a dependency. A PNG whose one data block is
//! stored uncompressed is about sixty lines of arithmetic, and the alternative
//! is a compression crate in a graphics engine's dependency list for the sake
//! of debug screenshots. The files are large; nothing keeps them.

use std::{fs, io::Write, path::Path};

use colby_core::{Result, err};

/// The bytes every PNG starts with.
const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// The largest a stored deflate block may be.
const BLOCK_MAX: usize = 0xFFFF;

/// An RGBA image, one byte per channel, rows top to bottom.
#[derive(Clone, Debug, Default)]
pub struct Image {
	/// Width in pixels.
	pub width: u32,

	/// Height in pixels.
	pub height: u32,

	/// `width * height * 4` bytes.
	pub pixels: Vec<u8>,
}

impl Image {
	/// One pixel, as `[r, g, b, a]`.
	///
	/// Reading outside the image is a black transparent pixel rather than a
	/// panic: callers here are tests sampling coordinates they computed, and a
	/// wrong coordinate should fail an assertion with a value in it.
	///
	/// @param x - column, from the left
	/// @param y - row, from the top
	#[must_use]
	pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
		let Some(offset) = self.offset(x, y) else {
			return [0, 0, 0, 0];
		};

		self.pixels
			.get(offset..offset + 4)
			.and_then(|slice| <[u8; 4]>::try_from(slice).ok())
			.unwrap_or_default()
	}

	/// Writes the image to a file as a PNG.
	///
	/// @param path - where to write it
	pub fn write_png(&self, path: &Path) -> Result {
		let mut file = fs::File::create(path)?;
		file.write_all(&self.encode_png())?;

		Ok(())
	}

	/// The whole PNG, as bytes.
	#[must_use]
	pub fn encode_png(&self) -> Vec<u8> {
		let mut png = Vec::from(SIGNATURE);

		let mut header = Vec::with_capacity(13);
		header.extend(self.width.to_be_bytes());
		header.extend(self.height.to_be_bytes());
		// eight bits per channel, color type 6 (truecolor with alpha),
		// deflate, no filtering beyond per-scanline, no interlacing.
		header.extend([8, 6, 0, 0, 0]);
		push_chunk(&mut png, *b"IHDR", &header);

		push_chunk(&mut png, *b"IDAT", &zlib_stored(&self.scanlines()));
		push_chunk(&mut png, *b"IEND", &[]);

		png
	}

	/// The image with a filter byte in front of every row, as PNG wants it.
	fn scanlines(&self) -> Vec<u8> {
		let stride = self.stride();
		let mut out = Vec::with_capacity(self.pixels.len() + stride);

		for row in self.pixels.chunks(stride) {
			// filter type zero: this row is stored as it is.
			out.push(0);
			out.extend_from_slice(row);
		}

		out
	}

	/// How many bytes one row of pixels takes.
	fn stride(&self) -> usize { usize::try_from(self.width).unwrap_or(0) * 4 }

	/// Where a pixel starts in [`pixels`](Self::pixels).
	fn offset(&self, x: u32, y: u32) -> Option<usize> {
		if x >= self.width || y >= self.height {
			return None;
		}

		let x = usize::try_from(x).ok()?;
		let y = usize::try_from(y).ok()?;

		y.checked_mul(self.stride())?.checked_add(x * 4)
	}
}

/// Reads a color back as it would be seen, given a path that has to exist.
///
/// @param path - where the image was written
/// @return the reason it could not be read, if it could not
pub fn require_written(path: &Path) -> Result {
	let size = fs::metadata(path)?.len();
	if size < 100 {
		return Err(err!(Graphics("{path:?} is only {size} bytes, which is not an image")));
	}

	Ok(())
}

/// Wraps bytes in a zlib stream whose deflate blocks are stored uncompressed.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
	// deflate, 32k window, no preset dictionary, and a check byte that makes
	// the two-byte header a multiple of 31.
	let mut out = vec![0x78, 0x01];

	if data.is_empty() {
		out.extend([0x01, 0x00, 0x00, 0xFF, 0xFF]);
	}

	for (index, block) in data.chunks(BLOCK_MAX).enumerate() {
		let last = (index + 1) * BLOCK_MAX >= data.len();
		let length = u16::try_from(block.len()).unwrap_or(0);

		out.push(u8::from(last));
		out.extend(length.to_le_bytes());
		out.extend((!length).to_le_bytes());
		out.extend_from_slice(block);
	}

	out.extend(adler32(data).to_be_bytes());

	out
}

/// Appends one PNG chunk: length, type, payload, CRC.
fn push_chunk(png: &mut Vec<u8>, kind: [u8; 4], payload: &[u8]) {
	png.extend(
		u32::try_from(payload.len())
			.unwrap_or(0)
			.to_be_bytes(),
	);
	png.extend_from_slice(&kind);
	png.extend_from_slice(payload);

	let mut crc = Vec::with_capacity(kind.len() + payload.len());
	crc.extend_from_slice(&kind);
	crc.extend_from_slice(payload);
	png.extend(crc32(&crc).to_be_bytes());
}

/// The CRC-32 PNG puts at the end of every chunk.
fn crc32(bytes: &[u8]) -> u32 {
	let mut crc = 0xFFFF_FFFF_u32;
	for byte in bytes {
		crc ^= u32::from(*byte);
		for _ in 0..8 {
			crc = if crc & 1 == 0 {
				crc >> 1
			} else {
				(crc >> 1) ^ 0xEDB8_8320
			};
		}
	}

	!crc
}

/// The Adler-32 a zlib stream ends with.
fn adler32(bytes: &[u8]) -> u32 {
	let (mut low, mut high) = (1_u32, 0_u32);
	for byte in bytes {
		low = (low + u32::from(*byte)) % 65521;
		high = (high + low) % 65521;
	}

	(high << 16) | low
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A two by two image: red, green on the top row, blue, white below.
	fn swatch() -> Image {
		Image {
			width: 2,
			height: 2,
			pixels: vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255],
		}
	}

	#[test]
	fn pixels_are_addressed_from_the_top_left() {
		let image = swatch();

		assert_eq!(
			image.pixel(0, 0),
			[255, 0, 0, 255],
			"the first pixel is the first four bytes"
		);
		assert_eq!(image.pixel(1, 0), [0, 255, 0, 255], "x walks along a row");
		assert_eq!(image.pixel(0, 1), [0, 0, 255, 255], "y walks down rows");
	}

	#[test]
	fn a_pixel_outside_the_image_reads_as_nothing() {
		let image = swatch();

		assert_eq!(image.pixel(2, 0), [0, 0, 0, 0], "past the right edge");
		assert_eq!(image.pixel(0, 2), [0, 0, 0, 0], "past the bottom edge");
	}

	#[test]
	fn the_png_starts_the_way_a_png_has_to() {
		let png = swatch().encode_png();

		assert_eq!(png.get(..8), Some(&SIGNATURE[..]), "signature");
		assert_eq!(png.get(12..16), Some(&b"IHDR"[..]), "the header comes first");
		assert_eq!(png.get(png.len() - 8..png.len() - 4), Some(&b"IEND"[..]), "and ends last");
	}

	#[test]
	fn the_crc_matches_the_one_png_defines() {
		// the CRC of "IEND" with an empty payload is fixed by the format.
		assert_eq!(crc32(b"IEND"), 0xAE42_6082, "IEND has a well known CRC");
	}

	#[test]
	fn the_adler_matches_the_one_zlib_defines() {
		// worked by hand from RFC 1950: `low` starts at one and accumulates the
		// bytes, so 1 + 97 + 98 + 99 = 295; `high` accumulates `low` after each
		// byte, so 98 + 196 + 295 = 589. The result is the two packed together.
		assert_eq!(adler32(b"abc"), (589 << 16) | 0x0127, "a + b*65536 over 'abc'");
		assert_eq!(adler32(b""), 1, "an empty stream is the initial low word alone");
	}
}
