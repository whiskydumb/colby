//! The WAV importer: a RIFF container into interleaved sixteen-bit samples.
//!
//! Written by hand, and the argument is the one the OBJ importer made rather
//! than the one `png` made. A RIFF file is a four-character tag, a length, and
//! the bytes; the whole of reading one is walking that list and knowing what
//! four of the tags mean. There is no entropy coder and no predictor to
//! reproduce, so a library here would be a dependency taken to avoid two
//! hundred lines of chunk walking.
//!
//! ```text
//!   0  "RIFF"  <u32 length>  "WAVE"
//!  12  a chunk: <4 bytes tag> <u32 length> <length bytes, padded to even>
//!   .  ... until the file runs out
//! ```
//!
//! Two chunks matter and every other one is walked past: `fmt ` says what the
//! samples are, and `data` is them. That is why an exporter's `LIST`, `fact`,
//! `cue ` or embedded picture costs nothing here - a reader that walks the list
//! properly does not have to know what any of them are.
//!
//! **Everything becomes signed sixteen-bit on the way in**, whatever the file
//! held, because that is what [`SoundData`] is. Eight-bit WAV samples are
//! *unsigned* and biased by 128, which is the one conversion in here that is
//! not a truncation and the one that is silently wrong if it is forgotten.
//!
//! What is deliberately refused, by name rather than by silence: A-law, mu-law
//! and ADPCM. Each of those is a decoder rather than a reinterpretation, and a
//! file in one of them is better recognized and named than half-read.

use std::{fs, path::Path};

use colby_core::{Result, abi::audio::SoundData, err};

/// The extension this importer claims.
pub const EXTENSION: &str = "wav";

/// The bytes a RIFF file starts with.
const RIFF: [u8; 4] = *b"RIFF";

/// The form a RIFF file has to be for this importer to want it.
const WAVE: [u8; 4] = *b"WAVE";

/// The chunk saying what the samples are.
const FMT: [u8; 4] = *b"fmt ";

/// The chunk that is them.
const DATA: [u8; 4] = *b"data";

/// Plain integer samples.
const FORMAT_PCM: u16 = 1;

/// Samples that are already floating point.
const FORMAT_FLOAT: u16 = 3;

/// A wrapper whose real format code lives in a sub-format GUID.
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// The shortest a `fmt ` chunk can be and still say anything.
const FMT_BYTES: usize = 16;

/// How far into a `WAVE_FORMAT_EXTENSIBLE` chunk the real format code sits.
///
/// Sixteen bytes of common fields, then `cbSize`, `wValidBitsPerSample` and
/// `dwChannelMask`, and then a sixteen-byte GUID whose first two bytes are the
/// code this file would have carried if it had not needed the wrapper.
const SUB_FORMAT_AT: usize = 24;

/// Reads a WAV file into samples.
///
/// @param path - the `.wav` to read
/// @return the sound, or why it could not be read
pub fn import_file(path: &Path) -> Result<SoundData> {
	let bytes =
		fs::read(path).map_err(|error| err!(Asset("reading {}: {error}", path.display())))?;

	import(&bytes).map_err(|error| err!(Asset("{}: {error}", path.display())))
}

/// Reads a WAV that is already in memory.
///
/// @param bytes - the whole file
/// @return the sound, or why it could not be read
pub fn import(bytes: &[u8]) -> Result<SoundData> {
	let body = container(bytes)?;
	let mut format: Option<Format> = None;
	let mut samples: Option<&[u8]> = None;
	let mut at = 0_usize;

	while let Some(found) = chunk(body, at) {
		match found.tag {
			| FMT => format = Some(read_format(found.body)?),
			// the first one only: a file with two data chunks is a file this
			// reader does not understand, and taking the last would quietly
			// play the wrong half of it.
			| DATA if samples.is_none() => samples = Some(found.body),
			| _ => {},
		}

		at = found.next;
	}

	let format =
		format.ok_or_else(|| err!(Asset("has no fmt chunk, so nothing says what is in it")))?;
	let samples =
		samples.ok_or_else(|| err!(Asset("has no data chunk, so there is nothing in it")))?;
	let data = SoundData {
		samples: widen(samples, format),
		rate: format.rate,
		channels: format.channels,
	};

	data.check()
		.map_err(|reason| err!(Asset("{reason}")))?;

	Ok(data)
}

/// Writes samples out as a canonical sixteen-bit PCM WAV.
///
/// The one format this project *writes*, and it is written rather than a
/// `.csnd` for one reason: a `.csnd` is read by colby and by nothing else,
/// and the thing a recording is for is being opened by whatever somebody
/// already has. Every part of it is the shape the reader above expects, which
/// is what makes `import(encode(x))` an exact round trip and therefore a test.
///
/// @param data - the samples to write
/// @return the whole file, ready to put on disk
///
/// # Errors
///
/// If the samples are not ones anything can play, or if there are more of them
/// than a thirty-two bit length can describe.
pub fn encode(data: &SoundData) -> Result<Vec<u8>> {
	data.check()
		.map_err(|reason| err!(Asset("{reason}")))?;

	let bytes = data.samples.len().saturating_mul(2);
	let length = u32::try_from(bytes)
		.map_err(|_| err!(Asset("{bytes} bytes of samples is more than a WAV can describe")))?;
	let align = data.channels.saturating_mul(2);

	let mut out = Vec::with_capacity(bytes + 44);
	out.extend_from_slice(&RIFF);
	// everything after this field: the four bytes of "WAVE", two chunk heads
	// of eight, sixteen of format, and the samples.
	out.extend_from_slice(&length.saturating_add(36).to_le_bytes());
	out.extend_from_slice(&WAVE);

	out.extend_from_slice(&FMT);
	out.extend_from_slice(&(u32::try_from(FMT_BYTES).unwrap_or(16)).to_le_bytes());
	out.extend_from_slice(&FORMAT_PCM.to_le_bytes());
	out.extend_from_slice(&data.channels.to_le_bytes());
	out.extend_from_slice(&data.rate.to_le_bytes());
	out.extend_from_slice(
		&data
			.rate
			.saturating_mul(u32::from(align))
			.to_le_bytes(),
	);
	out.extend_from_slice(&align.to_le_bytes());
	out.extend_from_slice(&16_u16.to_le_bytes());

	out.extend_from_slice(&DATA);
	out.extend_from_slice(&length.to_le_bytes());
	for sample in &data.samples {
		out.extend_from_slice(&sample.to_le_bytes());
	}

	Ok(out)
}

/// What a `fmt ` chunk said.
#[derive(Clone, Copy, Debug)]
struct Format {
	/// One of [`FORMAT_PCM`] or [`FORMAT_FLOAT`], the wrapper resolved.
	code: u16,

	/// How many samples make a frame.
	channels: u16,

	/// Frames a second.
	rate: u32,

	/// How wide one sample is.
	bits: u16,
}

/// Checks the outer wrapper and hands back everything inside it.
///
/// @param bytes - the whole file
/// @return the chunk list, starting at the first chunk
fn container(bytes: &[u8]) -> Result<&[u8]> {
	let head = bytes.get(..12).ok_or_else(|| {
		err!(Asset("is {} bytes long, too short to be a RIFF file", bytes.len()))
	})?;

	if head.get(..4) != Some(&RIFF[..]) {
		return Err(err!(Asset("does not start with RIFF, so it is not a WAV file at all")));
	}

	if head.get(8..12) != Some(&WAVE[..]) {
		return Err(err!(Asset(
			"is a RIFF file of some other form than WAVE: {:?}",
			String::from_utf8_lossy(head.get(8..12).unwrap_or_default())
		)));
	}

	// the length in the header is believed only as far as the file goes. An
	// exporter that wrote it before knowing the final size, or a transfer that
	// truncated the file, both produce one that is too large.
	let declared = u32::from_le_bytes(
		head.get(4..8)
			.and_then(|slice| slice.try_into().ok())
			.unwrap_or([0; 4]),
	);
	let available = bytes.len().saturating_sub(8);
	let length = usize::try_from(declared)
		.unwrap_or(usize::MAX)
		.min(available);

	Ok(bytes.get(12..8 + length).unwrap_or_default())
}

/// One chunk out of the list.
struct Chunk<'a> {
	/// The four characters naming what it is.
	tag: [u8; 4],

	/// Its contents, without the eight-byte head.
	body: &'a [u8],

	/// Where the next chunk starts, relative to the list.
	next: usize,
}

/// Reads one chunk out of the list.
///
/// @param body - everything after the twelve-byte header
/// @param at - where to start looking, relative to `body`
/// @return the chunk, or nothing once the list is done
fn chunk(body: &[u8], at: usize) -> Option<Chunk<'_>> {
	let header = body.get(at..at.checked_add(8)?)?;
	let tag: [u8; 4] = header.get(..4)?.try_into().ok()?;
	let length = usize::try_from(u32::from_le_bytes(header.get(4..8)?.try_into().ok()?)).ok()?;
	let start = at.checked_add(8)?;
	// a chunk that claims more than is left is read as far as the file goes
	// rather than refused: a recording cut off at the end is still worth
	// playing, and the alternative is a file that is fine except for its last
	// millisecond being no file at all.
	let end = start.saturating_add(length).min(body.len());
	// chunks sit on even offsets, and the pad byte is not counted in the
	// length. Forgetting this reads the next tag one byte late, which looks
	// like a corrupt file rather than like a rounding rule.
	let next = start.saturating_add(length + (length % 2));

	Some(Chunk { tag, body: body.get(start..end)?, next })
}

/// Reads a `fmt ` chunk, resolving the wrapper if there is one.
fn read_format(chunk: &[u8]) -> Result<Format> {
	if chunk.len() < FMT_BYTES {
		return Err(err!(Asset(
			"has a {}-byte fmt chunk, and {FMT_BYTES} is the least one can be",
			chunk.len()
		)));
	}

	let word = |at: usize| -> u16 {
		u16::from_le_bytes(
			chunk
				.get(at..at + 2)
				.and_then(|slice| slice.try_into().ok())
				.unwrap_or([0; 2]),
		)
	};
	let long = |at: usize| -> u32 {
		u32::from_le_bytes(
			chunk
				.get(at..at + 4)
				.and_then(|slice| slice.try_into().ok())
				.unwrap_or([0; 4]),
		)
	};

	let declared = word(0);
	let code = if declared == FORMAT_EXTENSIBLE {
		if chunk.len() < SUB_FORMAT_AT + 2 {
			return Err(err!(Asset(
				"says it is an extensible format and then stops before saying which one"
			)));
		}

		word(SUB_FORMAT_AT)
	} else {
		declared
	};

	if code != FORMAT_PCM && code != FORMAT_FLOAT {
		return Err(err!(Asset(
			"is in {}, which is a decoder rather than a way of writing samples down",
			name_of(code)
		)));
	}

	let format = Format {
		code,
		channels: word(2),
		rate: long(4),
		bits: word(14),
	};

	if width_of(format).is_none() {
		return Err(err!(Asset(
			"has {}-bit {} samples, which is not a width anything writes",
			format.bits,
			name_of(code)
		)));
	}

	Ok(format)
}

/// What to call a format code in a message.
///
/// The four that turn up in the wild are named, because "format 17" tells
/// whoever is looking at the log nothing about what to do next.
fn name_of(code: u16) -> String {
	match code {
		| FORMAT_PCM => "PCM".to_owned(),
		| FORMAT_FLOAT => "floating-point PCM".to_owned(),
		| 6 => "A-law".to_owned(),
		| 7 => "mu-law".to_owned(),
		| 0x11 => "IMA ADPCM".to_owned(),
		| 0x02 => "Microsoft ADPCM".to_owned(),
		| other => format!("format {other:#06X}"),
	}
}

/// How many bytes one sample of this format takes, if it is one this reads.
const fn width_of(format: Format) -> Option<usize> {
	match (format.code, format.bits) {
		| (FORMAT_PCM, 8) => Some(1),
		| (FORMAT_PCM, 16) => Some(2),
		| (FORMAT_PCM, 24) => Some(3),
		| (FORMAT_PCM | FORMAT_FLOAT, 32) => Some(4),
		| (FORMAT_FLOAT, 64) => Some(8),
		| _ => None,
	}
}

/// Turns however the file wrote its samples into signed sixteen-bit ones.
///
/// Every integer case is a truncation to the top two bytes, which is what
/// dropping precision means when the samples are little-endian. Eight-bit is
/// the exception and is the one worth reading twice: those samples are
/// *unsigned*, centered on 128, so the bias has to come off before the shift.
fn widen(bytes: &[u8], format: Format) -> Vec<i16> {
	let Some(width) = width_of(format) else {
		return Vec::new();
	};

	let pair = |sample: &[u8], at: usize| -> i16 {
		i16::from_le_bytes(
			sample
				.get(at..at + 2)
				.and_then(|slice| slice.try_into().ok())
				.unwrap_or([0; 2]),
		)
	};

	bytes
		.chunks_exact(width)
		.map(|sample| match (format.code, format.bits) {
			| (FORMAT_PCM, 8) => i16::from(sample.first().copied().unwrap_or(128))
				.saturating_sub(128)
				.saturating_mul(256),
			| (FORMAT_PCM, 16) => pair(sample, 0),
			| (FORMAT_PCM, 24) => pair(sample, 1),
			| (FORMAT_PCM, 32) => pair(sample, 2),
			| (FORMAT_FLOAT, 32) => from_float(f32::from_le_bytes(
				sample
					.get(..4)
					.and_then(|slice| slice.try_into().ok())
					.unwrap_or([0; 4]),
			)),
			| (FORMAT_FLOAT, 64) => {
				let wide = f64::from_le_bytes(
					sample
						.get(..8)
						.and_then(|slice| slice.try_into().ok())
						.unwrap_or([0; 8]),
				);

				#[expect(
					clippy::as_conversions,
					clippy::cast_possible_truncation,
					reason = "a sample outside the f32 range is outside -1 to 1 and is clamped \
					          by the line below either way"
				)]
				from_float(wide as f32)
			},
			| _ => 0,
		})
		.collect()
}

/// One floating-point sample as an integer one.
///
/// Clamped rather than wrapped: a sample past one is what a mix that clipped
/// looks like, and the loudest sixteen-bit value is a better answer to it than
/// the quietest.
/// @note: a sample that is not a number needs no branch of its own. `clamp`
/// hands NaN back unchanged and a float-to-integer `as` cast saturates, which
/// makes NaN zero - silence, which is the only sensible answer. A guard here
/// was written first and taken out again because no mutation of it could fail
/// a test: it said nothing the cast does not already do.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "the value is clamped to a range that fits in an i16, and a cast that saturates 	          is what turns a sample that is not a number into silence"
)]
fn from_float(sample: f32) -> i16 { (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16 }

#[cfg(test)]
mod tests {
	use super::*;

	/// Three frames of stereo at 22050, values 1, -1, 2, -2, 3, -3, written by
	/// python's `wave` module.
	///
	/// Not this project's own encoder, which is the whole point: a reader
	/// checked against a writer they were both written against proves only
	/// that the two agree. It is used twice - once to prove the reader takes
	/// what a real recorder wrote, and once to prove the writer produces the
	/// same bytes.
	const RECORDED: &[u8] = &[
		0x52, 0x49, 0x46, 0x46, 0x30, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74,
		0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x22, 0x56, 0x00, 0x00, 0x88, 0x58,
		0x01, 0x00, 0x04, 0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61, 0x0C, 0x00, 0x00, 0x00, 0x01,
		0x00, 0xFF, 0xFF, 0x02, 0x00, 0xFE, 0xFF, 0x03, 0x00, 0xFD, 0xFF,
	];

	/// Builds a WAV around a block of sample bytes.
	///
	/// A writer rather than a fixture, so the tests can vary one field at a
	/// time. It is deliberately *not* what the importer reads back through:
	/// what it writes is checked against a file a real recorder wrote, in
	/// [`a_file_from_an_outside_tool_reads_back`].
	fn wav(code: u16, channels: u16, rate: u32, bits: u16, samples: &[u8]) -> Vec<u8> {
		let mut fmt = Vec::new();
		let align = channels * bits / 8;

		fmt.extend_from_slice(&code.to_le_bytes());
		fmt.extend_from_slice(&channels.to_le_bytes());
		fmt.extend_from_slice(&rate.to_le_bytes());
		fmt.extend_from_slice(&(rate * u32::from(align)).to_le_bytes());
		fmt.extend_from_slice(&align.to_le_bytes());
		fmt.extend_from_slice(&bits.to_le_bytes());

		let mut body = Vec::new();
		body.extend_from_slice(&WAVE);
		body.extend_from_slice(&FMT);
		body.extend_from_slice(
			&u32::try_from(fmt.len())
				.unwrap_or(0)
				.to_le_bytes(),
		);
		body.extend_from_slice(&fmt);
		body.extend_from_slice(&DATA);
		body.extend_from_slice(
			&u32::try_from(samples.len())
				.unwrap_or(0)
				.to_le_bytes(),
		);
		body.extend_from_slice(samples);

		let mut out = Vec::new();
		out.extend_from_slice(&RIFF);
		out.extend_from_slice(
			&u32::try_from(body.len())
				.unwrap_or(0)
				.to_le_bytes(),
		);
		out.extend_from_slice(&body);

		out
	}

	/// Four sixteen-bit mono samples at eight kilohertz.
	fn simple() -> Vec<u8> {
		let mut samples = Vec::new();

		for value in [0_i16, 1000, -1000, i16::MAX] {
			samples.extend_from_slice(&value.to_le_bytes());
		}

		wav(FORMAT_PCM, 1, 8000, 16, &samples)
	}

	/// What reading these bytes went wrong with.
	fn refused(bytes: &[u8]) -> String {
		import(bytes)
			.expect_err("the file is not readable")
			.to_string()
	}

	#[test]
	fn sixteen_bit_samples_arrive_exactly_as_they_were_written() {
		let sound = import(&simple()).expect("four mono samples");

		assert_eq!(sound.samples, vec![0, 1000, -1000, i16::MAX], "every one of them");
		assert_eq!(sound.rate, 8000, "at the rate the file said");
		assert_eq!(sound.channels, 1, "and mono");
		assert_eq!(sound.frames(), 4, "which makes four frames");
	}

	#[test]
	fn eight_bit_samples_have_their_bias_taken_off() {
		// unsigned, centered on 128. Getting this wrong turns silence into a
		// loud constant offset, which sounds like nothing at all until it
		// clicks at the end.
		let sound = import(&wav(FORMAT_PCM, 1, 8000, 8, &[128, 255, 0, 129]))
			.expect("four eight-bit samples");

		assert_eq!(sound.samples[0], 0, "the middle of the range is silence");
		assert_eq!(sound.samples[1], 32512, "the top of it is nearly the loudest");
		assert_eq!(sound.samples[2], -32768, "and the bottom is the quietest");
		assert_eq!(sound.samples[3], 256, "one step above the middle is one step up");
	}

	#[test]
	fn wider_integer_samples_keep_their_top_two_bytes() {
		let deep = import(&wav(FORMAT_PCM, 1, 8000, 24, &[0xFF, 0x00, 0x40, 0x11, 0x22, 0xC0]))
			.expect("two twenty-four bit samples");

		assert_eq!(deep.samples, vec![0x4000, -0x3FDE], "the low byte is dropped, not folded in");

		let wide = import(&wav(FORMAT_PCM, 1, 8000, 32, &[
			0x11, 0x22, 0x00, 0x40, 0x33, 0x44, 0x00, 0xC0,
		]))
		.expect("two thirty-two bit samples");

		assert_eq!(wide.samples, vec![0x4000, -0x4000], "the low two bytes are dropped");
	}

	#[test]
	fn floating_point_samples_are_scaled_and_clipping_is_clamped() {
		let mut bytes = Vec::new();

		for value in [0.0_f32, 1.0, -1.0, 4.0, -4.0, f32::NAN] {
			bytes.extend_from_slice(&value.to_le_bytes());
		}

		let sound = import(&wav(FORMAT_FLOAT, 1, 8000, 32, &bytes)).expect("six float samples");

		assert_eq!(sound.samples[0], 0, "zero is silence");
		assert_eq!(sound.samples[1], i16::MAX, "one is the loudest");
		assert_eq!(sound.samples[2], -i16::MAX, "and minus one the other way");
		assert_eq!(sound.samples[3], i16::MAX, "four is clamped rather than wrapped");
		assert_eq!(sound.samples[4], -i16::MAX, "in both directions");
		assert_eq!(sound.samples[5], 0, "and a sample that is not a number is silence");
	}

	#[test]
	fn sixty_four_bit_floats_are_read_as_well() {
		let mut bytes = Vec::new();

		for value in [0.5_f64, -0.5] {
			bytes.extend_from_slice(&value.to_le_bytes());
		}

		let sound = import(&wav(FORMAT_FLOAT, 1, 8000, 64, &bytes)).expect("two double samples");

		assert_eq!(sound.samples, vec![16383, -16383], "half as loud, both ways");
	}

	#[test]
	fn stereo_samples_stay_interleaved() {
		let mut bytes = Vec::new();

		for value in [100_i16, -100, 200, -200] {
			bytes.extend_from_slice(&value.to_le_bytes());
		}

		let sound = import(&wav(FORMAT_PCM, 2, 44_100, 16, &bytes)).expect("two stereo frames");

		assert_eq!(sound.channels, 2, "two samples to a frame");
		assert_eq!(sound.frames(), 2, "so four samples are two frames");
		assert_eq!(sound.samples, vec![100, -100, 200, -200], "left, right, left, right");
	}

	#[test]
	fn an_extensible_header_is_followed_to_the_format_it_wraps() {
		// the sub-format GUID for PCM, whose first two bytes are the code the
		// file would have carried without the wrapper.
		let guid = [
			0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38,
			0x9B, 0x71,
		];
		let mut file = simple();
		let mut extension = Vec::new();

		extension.extend_from_slice(&22_u16.to_le_bytes());
		extension.extend_from_slice(&16_u16.to_le_bytes());
		extension.extend_from_slice(&3_u32.to_le_bytes());
		extension.extend_from_slice(&guid);

		// the fmt chunk's length is at 16, its code at 20, and the extension
		// goes on the end of its sixteen bytes.
		let at = 20 + FMT_BYTES;
		let grown = u32::try_from(FMT_BYTES + extension.len()).unwrap_or(0);
		file.splice(at..at, extension);
		file[16..20].copy_from_slice(&grown.to_le_bytes());
		file[20..22].copy_from_slice(&FORMAT_EXTENSIBLE.to_le_bytes());
		let length = u32::try_from(file.len() - 8).unwrap_or(0);
		file[4..8].copy_from_slice(&length.to_le_bytes());

		let sound = import(&file).expect("a wrapped PCM file is a PCM file");

		assert_eq!(sound.samples, vec![0, 1000, -1000, i16::MAX], "read exactly as an unwrapped");
	}

	#[test]
	fn a_chunk_nobody_here_knows_is_walked_past() {
		let mut file = simple();
		let mut extra = Vec::new();

		extra.extend_from_slice(b"LIST");
		extra.extend_from_slice(&5_u32.to_le_bytes());
		extra.extend_from_slice(b"INFOx");
		// the odd length forces the pad byte, which is the half of this rule
		// that is easy to leave out.
		extra.push(0);

		file.splice(12..12, extra);
		let length = u32::try_from(file.len() - 8).unwrap_or(0);
		file[4..8].copy_from_slice(&length.to_le_bytes());

		let sound = import(&file).expect("an unknown chunk is not an error");

		assert_eq!(sound.samples.len(), 4, "and the data after it still reads");
	}

	#[test]
	fn a_chunk_of_odd_length_is_followed_by_a_pad_byte() {
		// the same rule as above with nothing else going on, so that a reader
		// which ignores the padding fails this one and nothing else.
		let mut file = simple();
		let mut extra = Vec::new();

		extra.extend_from_slice(b"cue ");
		extra.extend_from_slice(&1_u32.to_le_bytes());
		extra.push(0xAB);
		extra.push(0);

		file.splice(12..12, extra);
		let length = u32::try_from(file.len() - 8).unwrap_or(0);
		file[4..8].copy_from_slice(&length.to_le_bytes());

		assert_eq!(
			import(&file)
				.expect("the padding is accounted for")
				.samples
				.len(),
			4,
			"or the next tag is read one byte late and nothing is found"
		);
	}

	#[test]
	fn the_first_data_chunk_wins_rather_than_the_last() {
		let mut file = simple();
		let mut second = Vec::new();

		second.extend_from_slice(&DATA);
		second.extend_from_slice(&2_u32.to_le_bytes());
		second.extend_from_slice(&999_i16.to_le_bytes());

		file.extend_from_slice(&second);
		let length = u32::try_from(file.len() - 8).unwrap_or(0);
		file[4..8].copy_from_slice(&length.to_le_bytes());

		let sound = import(&file).expect("it still reads");

		assert_eq!(sound.samples.len(), 4, "the first block, not the second");
	}

	#[test]
	fn a_data_chunk_claiming_more_than_the_file_holds_is_read_as_far_as_it_goes() {
		let mut file = simple();
		// the data chunk's length sits eight bytes before its contents, which
		// start after the header and the fmt chunk.
		let at = 12 + 8 + FMT_BYTES + 4;
		file[at..at + 4].copy_from_slice(&9000_u32.to_le_bytes());
		let length = u32::try_from(file.len() - 8).unwrap_or(0);
		file[4..8].copy_from_slice(&length.to_le_bytes());

		let sound = import(&file).expect("a truncated recording is still a recording");

		assert_eq!(sound.samples.len(), 4, "as much of it as there is");
	}

	#[test]
	fn a_file_with_no_format_chunk_says_which_one_is_missing() {
		let mut file = simple();
		file[12..16].copy_from_slice(b"junk");

		assert!(refused(&file).contains("no fmt chunk"), "and it is named");
	}

	#[test]
	fn a_file_with_no_samples_says_so_too() {
		let mut file = simple();
		let at = 12 + 8 + FMT_BYTES;
		file[at..at + 4].copy_from_slice(b"junk");

		assert!(refused(&file).contains("no data chunk"));
	}

	#[test]
	fn something_that_is_not_a_riff_file_is_refused_first() {
		let mut file = simple();
		file[0] = b'X';

		assert!(refused(&file).contains("not a WAV file"), "the outer tag is checked first");
	}

	#[test]
	fn a_riff_file_of_another_form_is_refused_by_name() {
		let mut file = simple();
		file[8..12].copy_from_slice(b"AVI ");

		let message = refused(&file);

		assert!(message.contains("some other form"), "it is RIFF but not WAVE: {message}");
		assert!(message.contains("AVI"), "and the form is named: {message}");
	}

	#[test]
	fn a_compressed_file_is_named_rather_than_half_read() {
		for (code, expected) in
			[(6_u16, "A-law"), (7, "mu-law"), (0x11, "IMA ADPCM"), (0x02, "Microsoft ADPCM")]
		{
			let file = wav(code, 1, 8000, 4, &[0; 8]);
			let message = refused(&file);

			assert!(
				message.contains(expected),
				"a person reading the log has to know what to convert: {message}"
			);
			assert!(message.contains("decoder"), "and why it was refused: {message}");
		}
	}

	#[test]
	fn a_sample_width_nothing_writes_is_refused() {
		assert!(
			refused(&wav(FORMAT_PCM, 1, 8000, 12, &[0; 8])).contains("12-bit"),
			"the width is named"
		);
		assert!(
			refused(&wav(FORMAT_FLOAT, 1, 8000, 16, &[0; 8])).contains("16-bit"),
			"and a float is not sixteen bits wide here"
		);
	}

	#[test]
	fn a_file_whose_channel_count_or_rate_is_impossible_is_refused_by_the_shared_check() {
		// the same complaints `SoundData::check` makes, reached through the
		// importer, so that neither side can drift into accepting what the
		// other refuses.
		assert!(refused(&wav(FORMAT_PCM, 6, 8000, 16, &[0; 24])).contains("mono or stereo"));
		assert!(refused(&wav(FORMAT_PCM, 1, 0, 16, &[0; 8])).contains("frames a second"));
	}

	#[test]
	fn a_truncated_file_is_refused_rather_than_read_short() {
		assert!(refused(&simple()[..8]).contains("too short to be a RIFF file"));
		assert!(
			refused(&wav(FORMAT_PCM, 1, 8000, 16, &[0; 8])[..20]).contains("least one can be"),
			"a fmt chunk cut in half is not a fmt chunk"
		);
	}

	#[test]
	fn a_file_from_an_outside_tool_reads_back() {
		let sound = import(RECORDED).expect("a real recorder's file");

		assert_eq!(sound.rate, 22_050, "the rate it was recorded at");
		assert_eq!(sound.channels, 2, "in stereo");
		assert_eq!(sound.samples, vec![1, -1, 2, -2, 3, -3], "and every sample of it");
	}

	#[test]
	fn what_this_writes_is_what_it_reads() {
		// the round trip is exact because both halves are the same eight
		// numbers; anything that made it approximate would be a bug in one of
		// them.
		let cases = [
			SoundData {
				samples: vec![0, 1000, -1000, i16::MAX, i16::MIN],
				rate: 8000,
				channels: 1,
			},
			SoundData {
				samples: vec![1, -1, 2, -2],
				rate: 48_000,
				channels: 2,
			},
			SoundData {
				samples: Vec::new(),
				rate: 44_100,
				channels: 2,
			},
		];

		for original in cases {
			let bytes = encode(&original).expect("it writes");
			let back = import(&bytes).expect("and reads back");

			assert_eq!(back, original, "every sample, the rate and the channels");
		}
	}

	#[test]
	fn what_this_writes_has_the_head_a_wav_file_has() {
		let bytes = encode(&SoundData {
			samples: vec![7; 6],
			rate: 22_050,
			channels: 2,
		})
		.expect("it writes");

		assert_eq!(bytes.len(), 44 + 12, "a forty-four byte head and six samples");
		assert_eq!(&bytes[..4], b"RIFF");
		assert_eq!(&bytes[8..12], b"WAVE");
		assert_eq!(&bytes[12..16], b"fmt ");
		assert_eq!(&bytes[36..40], b"data");
		assert_eq!(
			u32::from_le_bytes(bytes[4..8].try_into().expect("four bytes")),
			u32::try_from(bytes.len() - 8).expect("it fits"),
			"the length names everything after itself"
		);
		assert_eq!(
			u32::from_le_bytes(bytes[40..44].try_into().expect("four bytes")),
			12,
			"and the data chunk names the samples"
		);
	}

	#[test]
	fn what_a_python_recorder_writes_and_what_this_writes_agree() {
		// the same three frames of stereo the fixture above holds, written by
		// this encoder. If the two ever disagree it is this one that is wrong,
		// because the other came from outside the project.
		let ours = encode(&SoundData {
			samples: vec![1, -1, 2, -2, 3, -3],
			rate: 22_050,
			channels: 2,
		})
		.expect("it writes");

		assert_eq!(ours.len(), RECORDED.len(), "the same length");
		assert_eq!(ours, RECORDED, "and the same bytes, head and all");
	}

	#[test]
	fn samples_nothing_can_play_are_not_written() {
		let ragged = SoundData {
			samples: vec![0; 3],
			rate: 8000,
			channels: 2,
		};

		assert!(
			encode(&ragged)
				.expect_err("it is not writable")
				.to_string()
				.contains("whole number"),
			"the shared check refuses it on the way out as it does on the way in"
		);
	}

	#[test]
	fn a_missing_file_is_an_error_naming_it() {
		let message = import_file(Path::new("nothing-here.wav"))
			.expect_err("there is no such file")
			.to_string();

		assert!(message.contains("nothing-here.wav"), "the path is in the message: {message}");
	}
}
