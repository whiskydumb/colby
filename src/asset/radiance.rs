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
//!
//! **And the one format here that is written as well as read.** A bake of the
//! world's light is a picture of light, as a sky is, and it is kept as a
//! source under `assets/lightmaps/` for the reason a baked block is kept as an
//! `.obj`: what the engine made is then an input like any other, compiled,
//! watched and versioned. @ref [`encode`] and [`lightmap`].

use std::{fs, path::Path};

use colby_core::{
	Result,
	abi::{Texel, TextureData},
	err,
	utils::half::half,
};

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

/// Writes a picture in linear light as a Radiance file.
///
/// **Run-length scanlines**, the form every writer of the format emits and
/// the reader above takes either way: a lightmap is mostly the dark between
/// its charts, and a run of one byte is two bytes. A row too narrow or too
/// wide for that form is written flat, as the format says it must be.
///
/// **Each texel is the quadruple nearest it.** The shared exponent is the
/// largest channel's, found from its bits rather than from a logarithm, and
/// each mantissa is its channel times a power of two rounded to the nearest
/// whole number - so the value the reader decodes, `mantissa * 2^(e - 136)`
/// with no half step added, is within half a step of the one written, and
/// the same bytes come out on every machine. Nothing below nought and nothing
/// that is not a number is light, and both are written as black.
///
/// @param width - how many texels across
/// @param height - how many down, the first row being the top
/// @param texels - three floats a texel, row by row
/// @return the whole file
///
/// # Errors
///
/// If the picture is empty, larger than [`MAX_SIZE`] on either side, or has
/// other than `width * height` texels.
pub fn encode(width: u32, height: u32, texels: &[[f32; 3]]) -> Result<Vec<u8>> {
	if width == 0 || height == 0 || width > MAX_SIZE || height > MAX_SIZE {
		return Err(err!(Asset(
			"a picture of {width}x{height} cannot be written; each side is 1 to {MAX_SIZE}"
		)));
	}

	let across = usize::try_from(width).unwrap_or(0);
	let down = usize::try_from(height).unwrap_or(0);

	if texels.len() != across * down {
		return Err(err!(Asset(
			"a picture of {width}x{height} has {} texels, not {}",
			texels.len(),
			across * down
		)));
	}

	let mut bytes =
		format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n").into_bytes();
	let packs = (8..=0x7FFF).contains(&across);
	let mut channel = Vec::with_capacity(across);

	for row in texels.chunks_exact(across) {
		let quadruples: Vec<[u8; 4]> = row.iter().map(|texel| narrowed(*texel)).collect();

		if !packs {
			bytes.extend(quadruples.iter().flatten());

			continue;
		}

		bytes.extend_from_slice(&[2, 2, byte(across >> 8), byte(across & 0xFF)]);

		for which in 0..4 {
			channel.clear();
			channel.extend(
				quadruples
					.iter()
					.map(|quadruple| quadruple[which]),
			);
			packed(&channel, &mut bytes);
		}
	}

	Ok(bytes)
}

/// The shortest run of one byte written as a run rather than as it is.
///
/// Four: a run is two bytes, and a literal of three bytes is four, so below
/// this a run saves nothing and breaks a literal in two.
const SHORTEST_RUN: usize = 4;

/// The longest run one length byte can say: a length byte over a hundred and
/// twenty-eight is a run of that many less a hundred and twenty-eight.
const LONGEST_RUN: usize = 127;

/// The longest stretch of bytes one length byte can say as they are.
const LONGEST_LITERAL: usize = 128;

/// One channel of one scanline in the run-length form.
///
/// @param channel - the channel's bytes, one a texel
/// @param out - where the runs are written
fn packed(channel: &[u8], out: &mut Vec<u8>) {
	let mut at = 0;

	while at < channel.len() {
		let run = run_at(channel, at);

		if run >= SHORTEST_RUN {
			out.push(byte(128 + run));
			out.push(channel[at]);
			at += run;

			continue;
		}

		let start = at;

		while at < channel.len()
			&& at - start < LONGEST_LITERAL
			&& run_at(channel, at) < SHORTEST_RUN
		{
			at += 1;
		}

		out.push(byte(at - start));
		out.extend_from_slice(&channel[start..at]);
	}
}

/// How many bytes from here on are the byte here, up to the longest run.
fn run_at(channel: &[u8], at: usize) -> usize {
	channel
		.get(at..)
		.unwrap_or_default()
		.iter()
		.take(LONGEST_RUN)
		.take_while(|byte| **byte == channel[at])
		.count()
}

/// A count that the callers hold below 256, as the byte it is written as.
fn byte(count: usize) -> u8 { u8::try_from(count).unwrap_or(u8::MAX) }

/// Three linear channels as the RGBE quadruple nearest them.
fn narrowed(texel: [f32; 3]) -> [u8; 4] {
	// nothing below nought is light, and a comparison a not-a-number fails
	// makes it nought as well
	let channels = texel.map(|channel| if channel > 0.0 { f64::from(channel) } else { 0.0 });
	let largest = channels[0].max(channels[1]).max(channels[2]);

	// below this the largest mantissa would be a sliver of one step, which is
	// what every reader and writer of the format takes as black
	if largest < 1.0e-32 {
		return [0; 4];
	}

	let exponent = exponent_of(largest);

	// the next exponent as well: rounding the largest channel up can reach
	// 256, which is the next exponent's 128
	for exponent in [exponent, exponent + 1] {
		let scale = two_to(8 - exponent);
		let [red, green, blue] = channels.map(|channel| {
			let scaled = channel * scale;

			(scaled + 0.5).floor()
		});

		if red.max(green).max(blue) < 256.0 {
			let Ok(shared) = u8::try_from(exponent + 128) else {
				// past what one byte of exponent says: as bright as the format
				// goes
				return [255; 4];
			};

			return [mantissa(red), mantissa(green), mantissa(blue), shared];
		}
	}

	[255; 4]
}

/// The exponent `e` a positive, finite number is `m * 2^e` at, with `m` from a
/// half up to one: read from its bits, so no logarithm is asked.
fn exponent_of(value: f64) -> i32 {
	let biased = (value.to_bits() >> 52) & 0x7FF;

	i32::try_from(biased).unwrap_or(0) - 1022
}

/// Two to a whole power, built from its bits.
///
/// @param power - between -1022 and 1023, which every caller is
fn two_to(power: i32) -> f64 {
	u64::try_from(power + 1023).map_or(0.0, |biased| f64::from_bits(biased << 52))
}

/// A mantissa already rounded to a whole number below 256.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "a whole number from nought to 255, rounded and held on the lines that make it"
)]
const fn mantissa(value: f64) -> u8 { value as u8 }

/// A lightmap as the texture it compiles to.
///
/// **Flat, one level and half precision, and that is all three of its
/// differences from a picture.** Flat, because a lightmap is not a sky; one
/// level, because a coarser one would average the two texels between two
/// charts together and every chart would bleed into the next; half precision,
/// because light is not a number between nought and one. The alpha is one.
///
/// @param source - the lightmap as read
/// @return the texture, or why it could not be made
///
/// # Errors
///
/// If the picture's sides do not add up to the texels it holds.
pub fn lightmap(source: &Radiance) -> Result<TextureData> {
	let one = half(1.0).to_le_bytes();
	let level: Vec<u8> = source
		.texels
		.chunks_exact(3)
		.flat_map(|texel| {
			[texel[0], texel[1], texel[2]]
				.map(|channel| half(channel).to_le_bytes())
				.into_iter()
				.flatten()
				.chain(one)
		})
		.collect();
	let data = TextureData {
		width: source.width,
		height: source.height,
		faces: 1,
		texel: Texel::Rgba16Float,
		levels: vec![level],
	};

	if !data.is_consistent() {
		return Err(err!(Asset(
			"a lightmap of {}x{} does not hold that many texels",
			source.width,
			source.height
		)));
	}

	Ok(data)
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

	/// A spread of light: nought, tiny, dim, one, bright, a sun, and every
	/// channel different from the other two.
	fn spread() -> Vec<[f32; 3]> {
		let mut texels = vec![
			[0.0, 0.0, 0.0],
			[1.0e-30, 2.0e-31, 0.0],
			[0.013, 0.2, 0.0071],
			[1.0, 1.0, 1.0],
			[0.999, 0.5, 0.25],
			[3.75, 12.5, 0.001],
			[40_000.0, 1.0, 0.5],
			// dim, and far above what the format takes as black
			[0.0004, 0.0002, 0.0001],
		];

		// and a ramp long enough to run and to break a run
		for step in 0..57_u16 {
			let level = f32::from(step / 9) * 0.0625;

			texels.push([level, level * 0.5, 1.0 - level]);
		}

		texels
	}

	/// The step one mantissa is worth at the exponent the largest channel
	/// takes.
	fn step_of(texel: [f32; 3]) -> f64 {
		let largest = f64::from(texel[0].max(texel[1]).max(texel[2]));

		two_to(exponent_of(largest) - 8)
	}

	#[test]
	fn a_written_picture_reads_back_within_half_a_step_of_every_channel() {
		let texels = spread();
		let width = u32::try_from(texels.len()).expect("a short row");
		let read = import(&encode(width, 1, &texels).expect("it is written")).expect("and read");

		for (x, texel) in (0_u32..).zip(&texels) {
			let got = read.at(x, 0);

			if texel.iter().all(|channel| *channel < 1.0e-29) {
				assert!(same(got, [0.0; 3]), "{texel:?} is below what the format holds");

				continue;
			}

			for (wanted, came) in texel.iter().zip(got) {
				let off = (f64::from(*wanted) - f64::from(came)).abs();

				assert!(
					off <= step_of(*texel) * 0.5,
					"{wanted} came back as {came}, off by {off} of a step {}",
					step_of(*texel)
				);
			}
		}
	}

	#[test]
	fn what_the_format_holds_exactly_comes_back_to_the_bit() {
		// a mantissa of 128 at any exponent, and 192 of 256 at one
		let texels = [[1.0, 0.25, 0.5], [0.75, 0.375, 0.0]];
		let read = import(&encode(2, 1, &texels).expect("written")).expect("read");

		for (x, texel) in (0_u32..).zip(&texels) {
			assert_eq!(
				read.at(x, 0).map(f32::to_bits),
				texel.map(f32::to_bits),
				"texel {x} to the bit"
			);
		}
	}

	#[test]
	fn a_channel_that_rounds_up_to_the_next_power_takes_the_next_exponent() {
		// 0.999 is 255.7 of 256 at an exponent of nought, which rounds to 256:
		// the quadruple is 128 at the exponent above instead, which is one
		let read = import(&encode(1, 1, &[[0.999, 0.0, 0.0]]).expect("written")).expect("read");

		assert!(
			same(read.at(0, 0), [1.0, 0.0, 0.0]),
			"not a mantissa of 256: {:?}",
			read.at(0, 0)
		);
	}

	#[test]
	fn nothing_below_nought_and_nothing_that_is_not_a_number_is_light() {
		let texels = [[-1.0, 0.5, 0.25], [f32::NAN, f32::NAN, f32::NAN], [-0.0, 0.0, -3.0]];
		let read = import(&encode(3, 1, &texels).expect("written")).expect("read");

		assert!(same(read.at(0, 0), [0.0, 0.5, 0.25]), "the one below nought is nought");
		assert!(same(read.at(1, 0), [0.0; 3]), "and a not-a-number is black");
		assert!(same(read.at(2, 0), [0.0; 3]), "and so is everything below it");
	}

	#[test]
	fn a_long_row_of_changing_light_is_written_as_literals_and_reads_back_to_the_bit() {
		// three hundred texels no two alike: no run anywhere, so every channel
		// is literals, and a literal says at most a hundred and twenty-eight
		let row: Vec<[f32; 3]> = (1..=300_u16)
			.map(|step| {
				let level = f32::from(step);

				[level, level * 0.5, level * 0.25]
			})
			.collect();
		let bytes = encode(300, 1, &row).expect("written");
		let read = import(&bytes).expect("read");

		for (x, texel) in (0_u32..).zip(&row) {
			let step = step_of(*texel);

			for (wanted, came) in texel.iter().zip(read.at(x, 0)) {
				assert!(
					(f64::from(*wanted) - f64::from(came)).abs() <= step * 0.5,
					"texel {x}: {wanted} came back as {came}"
				);
			}
		}

		// each channel: 128 and 128 and 44 bytes, each run of them one byte
		// longer for its length; and the four of the marker
		let header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 300\n".len();
		let red_green_blue_exponent = [300 + 3, 300 + 3, 300 + 3, 300 + 3];
		let exponents: std::collections::HashSet<u8> = row
			.iter()
			.map(|texel| narrowed(*texel)[3])
			.collect();

		assert!(exponents.len() > 3, "the exponents change along the row: {exponents:?}");
		assert!(
			bytes.len() <= header + 4 + red_green_blue_exponent.iter().sum::<usize>(),
			"written as literals, never longer than that: {}",
			bytes.len()
		);
	}

	#[test]
	fn a_row_of_one_value_is_written_as_runs_and_a_narrow_one_as_it_is() {
		let dark = vec![[0.0_f32; 3]; 300];
		let header = b"#?RADIANCE
FORMAT=32-bit_rle_rgbe

-Y 1 +X 300
"
		.len();
		let bytes = encode(300, 1, &dark).expect("written");

		// four bytes of marker, then per channel three runs of 127, 127 and 46,
		// two bytes each
		assert_eq!(bytes.len(), header + 4 + 4 * 3 * 2, "three hundred texels of nothing");
		assert!(same(import(&bytes).expect("read").at(299, 0), [0.0; 3]), "and it reads back");

		let narrow = encode(3, 1, &[[1.0; 3], [0.5; 3], [0.25; 3]]).expect("written");
		let header = b"#?RADIANCE
FORMAT=32-bit_rle_rgbe

-Y 1 +X 3
"
		.len();

		assert_eq!(narrow.len(), header + 3 * 4, "a row narrower than eight is flat");
		assert!(same(import(&narrow).expect("read").at(2, 0), [0.25; 3]), "and reads back");
	}

	#[test]
	fn a_picture_whose_sides_do_not_match_its_texels_is_not_written() {
		assert!(encode(2, 2, &[[1.0; 3]; 3]).is_err(), "three texels are not two by two");
		assert!(encode(0, 1, &[]).is_err(), "and nothing is not a picture");
		assert!(
			encode(MAX_SIZE + 1, 1, &vec![[0.0; 3]; 8193]).is_err(),
			"and nothing is written that this build would not read"
		);
	}

	#[test]
	fn a_lightmap_is_flat_of_one_level_and_half_precision_with_an_alpha_of_one() {
		let read =
			import(&encode(2, 1, &[[1.0, 0.5, 0.25], [2.0, 0.0, 0.125]]).expect("written"))
				.expect("read");
		let data = lightmap(&read).expect("a lightmap of two texels");

		assert_eq!((data.width, data.height, data.faces), (2, 1, 1), "flat, as drawn");
		assert_eq!(data.texel, Texel::Rgba16Float, "half precision");
		assert_eq!(data.levels.len(), 1, "one level, so no chart bleeds into the next");

		let halves: Vec<u16> = data.levels[0]
			.chunks_exact(2)
			.map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
			.collect();

		assert_eq!(
			halves,
			[1.0, 0.5, 0.25, 1.0, 2.0, 0.0, 0.125, 1.0].map(half),
			"each channel the half it is, and one for alpha"
		);
	}
}
