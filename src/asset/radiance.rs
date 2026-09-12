//! Reading Radiance RGBE, the environment import format.
//!
//! **Why a second image reader at all.** A PNG holds eight bits a channel and a
//! sky holds a sun, and a sun is not a number between nought and one. An
//! eight-bit sky clips it to white, and then every reflection of that sky comes
//! back the same grey as the cloud beside it - which is the whole thing an
//! environment map exists to avoid. So the import format has to carry range,
//! and RGBE is the one that does and is small enough to read by hand: three
//! mantissas and one shared exponent, one byte each.
//!
//! ```text
//!   #?RADIANCE            the signature
//!   FORMAT=32-bit_rle_rgbe
//!                         a blank line ends the header
//!   -Y 512 +X 1024        the resolution, rows top to bottom
//!   <scanlines>
//! ```
//!
//! **One value per byte quadruple, and the decode is exact.** A texel is
//! `mantissa * 2^(exponent - 136)` per channel, with nought for an exponent of
//! nought - the convention every reader in the field uses, and the reason it
//! matters here is that a second answer worked out away from this code has to
//! agree bit for bit with it. Integer arithmetic and one power of two have that
//! property; a curve fitted to look right would not.
//!
//! Only `-Y +X` is read, which is every file anything writes: the four other
//! orientations are legal in the format and are refused by name rather than
//! transposed, because a picture read sideways is worse than a picture refused.
//!
//! This runs offline like every other importer. @ref
//! [`texture`](crate::texture) for what is made of the result.

use std::{fs, path::Path};

use colby_core::{Result, err};

/// The extension this importer claims.
pub const EXTENSION: &str = "hdr";

/// The bytes every one of these files starts with.
const SIGNATURE: &[u8] = b"#?";

/// The largest image this will read, on either side.
///
/// A limit on how wrong a file has to be before the reader stops rather than
/// allocating what a header asked for. Four thousand across is a whole sky at
/// more detail than a cube of a hundred and twenty-eight will ever show.
pub const MAX_SIZE: u32 = 8192;

/// How long a header may be before the reader gives up looking for its end.
const MAX_HEADER_LINES: usize = 128;

/// An equirectangular picture in linear light.
#[derive(Clone, Debug, PartialEq)]
pub struct Radiance {
	/// How many texels across.
	pub width: u32,

	/// How many texels down, the first row being the top of the sky.
	pub height: u32,

	/// Three floats a texel, row by row: red, green, blue, linear.
	pub texels: Vec<f32>,
}

impl Radiance {
	/// One texel's three channels, or black outside the picture.
	///
	/// @param x - the column
	/// @param y - the row, nought being the top
	#[must_use]
	pub fn at(&self, x: u32, y: u32) -> [f32; 3] {
		if x >= self.width || y >= self.height {
			return [0.0; 3];
		}

		let index = (usize::try_from(y).unwrap_or(0) * usize::try_from(self.width).unwrap_or(0)
			+ usize::try_from(x).unwrap_or(0))
			* 3;

		match self.texels.get(index..index + 3) {
			| Some(&[red, green, blue]) => [red, green, blue],
			| _ => [0.0; 3],
		}
	}
}

/// Reads a Radiance file into linear floats.
///
/// @param path - the `.hdr` to read
/// @return the picture, or why it could not be read
pub fn import_file(path: &Path) -> Result<Radiance> {
	let bytes =
		fs::read(path).map_err(|error| err!(Asset("reading {}: {error}", path.display())))?;

	import(&bytes).map_err(|error| err!(Asset("{}: {error}", path.display())))
}

/// The same, given the bytes.
///
/// @param bytes - the whole file
/// @return the picture, or why it could not be read
pub fn import(bytes: &[u8]) -> Result<Radiance> {
	if !bytes.starts_with(SIGNATURE) {
		return Err(err!(Asset("does not start with {}", "#?")));
	}

	let (width, height, mut at) = head(bytes)?;
	let across = usize::try_from(width).unwrap_or(0);
	let down = usize::try_from(height).unwrap_or(0);
	let mut texels = Vec::with_capacity(across * down * 3);
	let mut row = vec![0_u8; across * 4];

	for line in 0..down {
		at = scanline(bytes, at, &mut row)
			.map_err(|reason| err!(Asset("row {line} of {height}: {reason}")))?;

		for texel in row.chunks_exact(4) {
			let [red, green, blue] = widen(texel);
			texels.push(red);
			texels.push(green);
			texels.push(blue);
		}
	}

	Ok(Radiance { width, height, texels })
}

/// One RGBE quadruple as three linear floats.
///
/// @param texel - four bytes: three mantissas and the shared exponent
fn widen(texel: &[u8]) -> [f32; 3] {
	let &[red, green, blue, exponent] = texel else {
		return [0.0; 3];
	};

	if exponent == 0 {
		return [0.0; 3];
	}

	// the shared exponent is biased by a hundred and twenty-eight, and the
	// mantissas are whole numbers out of two hundred and fifty-six rather than
	// out of one - which is the other eight.
	let scale = (f64::from(exponent) - 136.0).exp2();

	[
		f64_to_f32(f64::from(red) * scale),
		f64_to_f32(f64::from(green) * scale),
		f64_to_f32(f64::from(blue) * scale),
	]
}

/// A double narrowed to a float, saturating rather than reaching infinity.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "the whole point of the call is the narrowing, and the value is held inside the \
	          range first"
)]
fn f64_to_f32(value: f64) -> f32 { value.clamp(f64::from(-f32::MAX), f64::from(f32::MAX)) as f32 }

/// Reads the header and the resolution line.
///
/// @param bytes - the whole file
/// @return the width, the height, and where the first scanline starts
fn head(bytes: &[u8]) -> Result<(u32, u32, usize)> {
	let mut at = 0;
	let mut blank = false;

	for _ in 0..MAX_HEADER_LINES {
		let (line, next) = line_at(bytes, at);
		at = next;

		if line.is_empty() {
			blank = true;

			continue;
		}

		if !blank {
			continue;
		}

		return resolution(line).map(|(width, height)| (width, height, at));
	}

	Err(err!(Asset("has no resolution line in its first {MAX_HEADER_LINES} lines")))
}

/// The resolution line, which says the orientation as well as the size.
///
/// @param line - the line after the header's blank one
fn resolution(line: &[u8]) -> Result<(u32, u32)> {
	let text = std::str::from_utf8(line)
		.map_err(|_| err!(Asset("has a resolution line that is not text")))?;
	let words: Vec<&str> = text.split_whitespace().collect();

	let &["-Y", height, "+X", width] = words.as_slice() else {
		return Err(err!(Asset(
			"is laid out as `{text}`, and this build reads `-Y <height> +X <width>` only"
		)));
	};

	let height: u32 = height
		.parse()
		.map_err(|_| err!(Asset("says its height is `{height}`")))?;
	let width: u32 = width
		.parse()
		.map_err(|_| err!(Asset("says its width is `{width}`")))?;

	if width == 0 || height == 0 {
		return Err(err!(Asset("is {width}x{height}, which is not an image")));
	}

	if width > MAX_SIZE || height > MAX_SIZE {
		return Err(err!(Asset("is {width}x{height}, past the {MAX_SIZE} this build will read")));
	}

	Ok((width, height))
}

/// One line of text and where the next starts, newlines eaten.
fn line_at(bytes: &[u8], from: usize) -> (&[u8], usize) {
	let rest = bytes.get(from..).unwrap_or_default();
	let end = rest
		.iter()
		.position(|byte| *byte == b'\n')
		.unwrap_or(rest.len());
	let line = rest.get(..end).unwrap_or_default();
	// a file written on one platform and read on another carries the other's
	// line ending, and a stray carriage return would end up inside a number.
	let line = match line {
		| [head @ .., b'\r'] => head,
		| whole => whole,
	};

	(line, from + end + 1)
}

/// Reads one scanline into four bytes a texel, flat or run-length.
///
/// @param bytes - the whole file
/// @param from - where this scanline starts
/// @param row - filled with `width * 4` bytes
/// @return where the next scanline starts
fn scanline(bytes: &[u8], from: usize, row: &mut [u8]) -> std::result::Result<usize, String> {
	let width = row.len() / 4;
	// the run-length form announces itself with a red of two, a green of two
	// and the width in the other two bytes; anything else is four bytes a
	// texel, in order. A width outside 8..=0x7FFF cannot use the run-length
	// form at all, which is why the check is on the header rather than on the
	// first byte alone.
	let head = bytes
		.get(from..from + 4)
		.ok_or("the file ends before it")?;
	let packed = matches!(head, [2, 2, hi, _] if usize::from(*hi) < 0x80)
		&& usize::from(head[2]) * 256 + usize::from(head[3]) == width
		&& (8..=0x7FFF).contains(&width);

	if !packed {
		let flat = bytes
			.get(from..from + row.len())
			.ok_or("the file ends inside it")?;
		row.copy_from_slice(flat);

		return Ok(from + row.len());
	}

	let mut at = from + 4;

	// a packed scanline holds one channel at a time, all the reds and then all
	// the greens: the exponents of a sky are nearly all the same, so a run of
	// them compresses and a run of interleaved quadruples would not.
	for channel in 0..4 {
		let mut filled = 0;

		while filled < width {
			let (taken, next) = run(bytes, at, row, &Run { channel, filled, width })?;
			at = next;
			filled += taken;
		}
	}

	Ok(at)
}

/// Where one run of a packed scanline goes.
struct Run {
	/// Which of the four channels is being filled.
	channel: usize,

	/// How many texels of it are filled already.
	filled: usize,

	/// How many there are to fill.
	width: usize,
}

/// Reads one run - repeated or literal - into one channel of a row.
///
/// @param bytes - the whole file
/// @param at - where the run's length byte is
/// @param row - the row being filled
/// @param into - which channel, how far along, and how wide
/// @return how many texels it filled, and where the next run starts
fn run(
	bytes: &[u8],
	at: usize,
	row: &mut [u8],
	into: &Run,
) -> std::result::Result<(usize, usize), String> {
	let &Run { channel, filled, width } = into;
	let count = usize::from(
		*bytes
			.get(at)
			.ok_or("the file ends inside a run")?,
	);
	// a length over a hundred and twenty-eight means one byte repeated that
	// many times past it; anything else is that many bytes in order
	let repeated = count > 128;
	let taken = if repeated { count - 128 } else { count };

	if taken == 0 {
		return Err("has a run of no texels at all".to_owned());
	}

	if filled + taken > width {
		return Err("has a run that reaches past its width".to_owned());
	}

	if repeated {
		let byte = *bytes
			.get(at + 1)
			.ok_or("the file ends inside a run")?;

		for step in 0..taken {
			row[(filled + step) * 4 + channel] = byte;
		}

		return Ok((taken, at + 2));
	}

	let literal = bytes
		.get(at + 1..at + 1 + taken)
		.ok_or("the file ends inside a literal run")?;
	for (step, byte) in literal.iter().enumerate() {
		row[(filled + step) * 4 + channel] = *byte;
	}

	Ok((taken, at + 1 + taken))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Whether three channels are the same to the last bit that matters.
	///
	/// A helper rather than an equality, because comparing floating-point
	/// arrays for exact equality is a habit worth not having even where, as
	/// here, the arithmetic really is exact.
	fn same(got: [f32; 3], wanted: [f32; 3]) -> bool {
		got.iter()
			.zip(wanted)
			.all(|(left, right)| (left - right).abs() <= 1.0e-9)
	}

	/// A flat two-by-one file: one texel at one, one at a quarter.
	fn flat() -> Vec<u8> {
		let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n".to_vec();

		// a mantissa of 128 with an exponent of 129 is 128 * 2^(129-136) = 1.0
		bytes.extend_from_slice(&[128, 128, 128, 129]);
		bytes.extend_from_slice(&[128, 128, 128, 127]);

		bytes
	}

	#[test]
	fn a_flat_file_reads_the_values_its_bytes_stand_for() {
		let read = import(&flat()).expect("a two-texel file");

		assert_eq!((read.width, read.height), (2, 1), "the resolution line said so");
		assert!(same(read.at(0, 0), [1.0; 3]), "a mantissa of 128 at an exponent of 129 is one");
		assert!(same(read.at(1, 0), [0.25; 3]), "and two exponents down is a quarter");
		assert!(same(read.at(2, 0), [0.0; 3]), "outside the picture is black");
	}

	#[test]
	fn an_exponent_of_nothing_is_black_rather_than_very_small() {
		let mut bytes = b"#?RADIANCE\n\n-Y 1 +X 1\n".to_vec();
		bytes.extend_from_slice(&[255, 255, 255, 0]);

		assert!(
			same(import(&bytes).expect("one texel").at(0, 0), [0.0; 3]),
			"the format says an exponent of nought is nothing, whatever the mantissas hold"
		);
	}

	#[test]
	fn a_value_above_one_survives_which_is_the_whole_reason_for_this_format() {
		let mut bytes = b"#?RADIANCE\n\n-Y 1 +X 1\n".to_vec();
		// 128 * 2^(138-136) = 512
		bytes.extend_from_slice(&[128, 64, 32, 138]);

		let read = import(&bytes).expect("one texel");

		assert!(same(read.at(0, 0), [512.0, 256.0, 128.0]), "a sun is not held down to one");
	}

	#[test]
	fn a_run_length_scanline_reads_the_same_as_the_flat_one_it_stands_for() {
		const WIDE: usize = 16;

		let mut bytes = b"#?RADIANCE\n\n-Y 1 +X 16\n".to_vec();
		#[expect(
			clippy::as_conversions,
			clippy::cast_possible_truncation,
			reason = "the width is sixteen, which fits a byte with room to spare"
		)]
		bytes.extend_from_slice(&[2, 2, (WIDE >> 8) as u8, (WIDE & 0xFF) as u8]);

		// red: eight literal bytes then a run of eight; the other three
		// channels are one run each
		bytes.push(8);
		bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
		bytes.extend_from_slice(&[128 + 8, 9]);
		bytes.extend_from_slice(&[128 + 16, 64]);
		bytes.extend_from_slice(&[128 + 16, 32]);
		bytes.extend_from_slice(&[128 + 16, 136]);

		let read = import(&bytes).expect("a packed scanline");

		assert_eq!(read.width, 16, "sixteen across");
		assert!(
			same(read.at(0, 0), [1.0, 64.0, 32.0]),
			"the first literal byte, at an exponent of one"
		);
		assert!(same(read.at(7, 0), [8.0, 64.0, 32.0]), "the last of them");
		assert!(same(read.at(8, 0), [9.0, 64.0, 32.0]), "the first of the run");
		assert!(same(read.at(15, 0), [9.0, 64.0, 32.0]), "and the last");
	}

	#[test]
	fn a_run_that_reaches_past_the_width_is_refused_rather_than_written_past_the_row() {
		let mut bytes = b"#?RADIANCE\n\n-Y 1 +X 16\n".to_vec();
		bytes.extend_from_slice(&[2, 2, 0, 16]);
		bytes.extend_from_slice(&[128 + 32, 7]);

		let refused = import(&bytes).expect_err("a run of thirty-two into sixteen");

		assert!(format!("{refused}").contains("past its width"), "and it says which: {refused}");
	}

	#[test]
	fn the_four_orientations_this_build_does_not_read_are_refused_by_name() {
		for line in ["+Y 4 +X 4", "-Y 4 -X 4", "+X 4 -Y 4"] {
			let bytes = format!("#?RADIANCE\n\n{line}\n").into_bytes();
			let refused = import(&bytes).expect_err("only one orientation is read");

			assert!(
				format!("{refused}").contains("-Y <height> +X <width>"),
				"{line} says what is read instead: {refused}"
			);
		}
	}

	#[test]
	fn a_file_that_is_not_one_is_refused_at_the_first_two_bytes() {
		assert!(import(b"\x89PNG\r\n\x1a\n").is_err(), "a png is not a radiance file");
		assert!(import(b"").is_err(), "and nor is nothing");
	}

	#[test]
	fn a_header_with_no_blank_line_does_not_read_a_resolution_out_of_it() {
		let bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n-Y 1 +X 1\n".to_vec();

		assert!(
			import(&bytes).is_err(),
			"the blank line is what says the header is over, so without it there is no \
			 resolution line"
		);
	}

	#[test]
	fn a_carriage_return_before_the_newline_does_not_end_up_inside_a_number() {
		let mut bytes = b"#?RADIANCE\r\nFORMAT=32-bit_rle_rgbe\r\n\r\n-Y 1 +X 1\r\n".to_vec();
		bytes.extend_from_slice(&[128, 128, 128, 129]);

		let read = import(&bytes).expect("a file written on the other platform");

		assert_eq!((read.width, read.height), (1, 1), "the size read the same");
		assert!(same(read.at(0, 0), [1.0; 3]), "and so did the texel");
	}

	#[test]
	fn a_picture_that_claims_more_rows_than_it_holds_is_refused() {
		let mut bytes = b"#?RADIANCE\n\n-Y 4 +X 2\n".to_vec();
		bytes.extend_from_slice(&[128, 128, 128, 129]);

		let refused = import(&bytes).expect_err("one texel of eight");

		assert!(format!("{refused}").contains("row 0 of 4"), "and says where: {refused}");
	}

	#[test]
	fn a_size_past_what_this_build_reads_is_refused_before_anything_is_allocated() {
		let bytes = b"#?RADIANCE\n\n-Y 99999 +X 99999\n".to_vec();
		let refused = import(&bytes).expect_err("a file nobody wrote");

		assert!(format!("{refused}").contains("past the"), "with the ceiling named: {refused}");
	}
}
