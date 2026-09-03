//! colby's runtime sound format: `.csnd`.
//!
//! The simplest of the formats here, and deliberately so: a header and then the
//! samples, with nothing between them to decode. Loading one is a read into an
//! aligned buffer and a single cast, which is the same bargain `.cmesh` made
//! with the vertex buffer.
//!
//! ```text
//!   0  SoundHeader                      32 bytes
//!  32  [i16; sample_count]              interleaved, channels to a frame
//! ```
//!
//! **No codec field, and adding one will move [`FORMAT_VERSION`].** That is the
//! distinction the scene format already draws: a *code* says which of several
//! things a record is and has nothing smaller to fall back to, while a *flag
//! bit* says a record has a property, so a build that does not know one reads a
//! smaller record rather than a wrong one. A reader that met an unknown codec
//! and carried on would play noise at full volume, which is the worst failure
//! available to an audio format.
//!
//! **The samples keep the rate the recording was made at.** Resampling on the
//! way in would need the device's rate, which is not knowable here - the file
//! is compiled once and played on whatever machine opens it. The mixer steps a
//! fractional index instead, which it has to do for pitch anyway.

use std::path::Path;

use colby_core::{
	Result,
	abi::audio::{MAX_SAMPLES, SoundData},
	bytemuck::{self, Pod, Zeroable},
	err,
};

use crate::bytes::{AlignedBytes, count, fits, span};

/// The eight bytes every `.csnd` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYSND";

/// The revision of everything in this module.
///
/// Bump it whenever the header changes shape, or whenever what the sample
/// block holds stops being plain interleaved sixteen-bit samples. A file
/// carrying a different number is refused with a message rather than read as if
/// it agreed.
pub const FORMAT_VERSION: u32 = 1;

/// The extension a compiled sound is written with.
pub const EXTENSION: &str = "csnd";

/// How big [`SoundHeader`] is, and where the samples start.
pub const HEADER_BYTES: usize = 32;

/// The fixed head of a `.csnd`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct SoundHeader {
	/// [`MAGIC`]. Anything else is not one of these files.
	pub magic: [u8; 8],

	/// [`FORMAT_VERSION`] at the time the file was written.
	pub version: u32,

	/// Reserved for optional blocks - loop points are the likely first one.
	/// Every bit is zero in version one, and a reader refuses a bit it does
	/// not know rather than ignoring it.
	pub flags: u32,

	/// Frames a second, as the recording was made.
	pub rate: u32,

	/// How many samples make one frame: one for mono, two for stereo.
	pub channels: u16,

	/// Nothing yet, and a reader refuses a file that puts something here.
	pub reserved: u16,

	/// How many samples there are, counting every channel.
	pub sample_count: u32,

	/// Where the samples start, in bytes from the start of the file.
	pub sample_offset: u32,
}

const _: () = assert!(
	size_of::<SoundHeader>() == HEADER_BYTES,
	"the header has to stay thirty-two bytes for the samples after it to be readable"
);

/// A `.csnd` held in memory, checked, and ready to be read in place.
#[derive(Clone, Debug)]
pub struct SoundFile {
	bytes: AlignedBytes,
	header: SoundHeader,
}

impl SoundFile {
	/// Reads and checks a compiled sound.
	///
	/// @param path - the `.csnd` to read
	/// @return the file, or why it could not be used
	pub fn open(path: &Path) -> Result<Self> {
		let bytes = AlignedBytes::read(path)?;
		let header = check(bytes.as_slice())
			.map_err(|reason| err!(Asset("{}: {reason}", path.display())))?;

		Ok(Self { bytes, header })
	}

	/// Checks bytes that are already in memory.
	///
	/// @param bytes - the whole file
	/// @return the file, or why it could not be used
	pub fn from_bytes(bytes: AlignedBytes) -> Result<Self> {
		let header = check(bytes.as_slice()).map_err(|reason| err!(Asset("{reason}")))?;

		Ok(Self { bytes, header })
	}

	/// The header, as it was read.
	#[must_use]
	pub const fn header(&self) -> &SoundHeader { &self.header }

	/// The samples, borrowed out of the buffer.
	///
	/// The whole point of the format: what is on disk is already what a mixer
	/// reads, so this is a cast rather than a decode.
	#[must_use]
	pub fn samples(&self) -> &[i16] {
		let Some(range) = span::<i16>(self.header.sample_offset, self.header.sample_count) else {
			return &[];
		};

		self.bytes
			.as_slice()
			.get(range)
			.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
			.unwrap_or(&[])
	}

	/// Copies the whole file into owned data.
	///
	/// The one copy in the path, and it is here for the same reason a mesh's
	/// is: what the host holds can also be built rather than read.
	#[must_use]
	pub fn to_sound_data(&self) -> SoundData {
		SoundData {
			samples: self.samples().to_vec(),
			rate: self.header.rate,
			channels: self.header.channels,
		}
	}
}

/// Writes a sound out as a `.csnd`.
///
/// @param data - the samples to write
/// @return the whole file, ready to put on disk
pub fn encode(data: &SoundData) -> Result<Vec<u8>> {
	// the same check the importer ran, run again on the way out, because a
	// `SoundData` may also be built by hand and this is the last place before
	// it becomes a file somebody else's build will read.
	data.check()
		.map_err(|reason| err!(Asset("{reason}")))?;

	let header = SoundHeader {
		magic: MAGIC,
		version: FORMAT_VERSION,
		flags: 0,
		rate: data.rate,
		channels: data.channels,
		reserved: 0,
		sample_count: count(data.samples.len(), "a sound's samples")?,
		sample_offset: count(HEADER_BYTES, "a sound's samples")?,
	};

	let mut out = Vec::with_capacity(HEADER_BYTES + size_of_val(data.samples.as_slice()));
	out.extend_from_slice(bytemuck::bytes_of(&header));
	out.extend_from_slice(bytemuck::cast_slice(&data.samples));

	Ok(out)
}

/// The version a `.csnd` claims, without reading the rest of it.
///
/// What the compiler asks so that a file written by another build of the
/// engine is stale however new it is.
///
/// @param path - the file to look at
/// @return its version, or nothing when it is not one of these at all
#[must_use]
pub fn version_of(path: &Path) -> Option<u32> {
	let mut head = [0_u8; 12];
	let mut file = std::fs::File::open(path).ok()?;
	std::io::Read::read_exact(&mut file, &mut head).ok()?;

	if head.get(..MAGIC.len()) != Some(&MAGIC[..]) {
		return None;
	}

	let version: [u8; 4] = head.get(8..12)?.try_into().ok()?;

	Some(u32::from_le_bytes(version))
}

/// Everything that has to hold before a [`SoundFile`] exists.
fn check(bytes: &[u8]) -> std::result::Result<SoundHeader, String> {
	let head = bytes.get(..HEADER_BYTES).ok_or_else(|| {
		format!(
			"only {} bytes long, too short to hold a {HEADER_BYTES}-byte header",
			bytes.len()
		)
	})?;
	let header: &SoundHeader = bytemuck::try_from_bytes(head)
		.map_err(|error| format!("the header could not be read: {error}"))?;

	if header.magic != MAGIC {
		return Err(format!(
			"not a colby sound: expected {:?} at the start, found {:?}",
			String::from_utf8_lossy(&MAGIC),
			String::from_utf8_lossy(&header.magic)
		));
	}

	if header.version != FORMAT_VERSION {
		return Err(format!(
			"written by asset format version {}, and this build reads version {FORMAT_VERSION}; \
			 run `just assets --force` to recompile it",
			header.version
		));
	}

	if header.flags != 0 {
		return Err(format!(
			"sets flag bits {:#010X} that this build does not know about",
			header.flags
		));
	}

	if header.reserved != 0 {
		return Err(format!(
			"puts {:#06X} in a word of the header that has no meaning yet",
			header.reserved
		));
	}

	if header.sample_count > u32::try_from(MAX_SAMPLES).unwrap_or(u32::MAX) {
		return Err(format!(
			"declares {} samples, past the {MAX_SAMPLES} one sound may have",
			header.sample_count
		));
	}

	fits::<i16>(bytes, HEADER_BYTES, (header.sample_offset, header.sample_count), "samples")?;

	// and the fields the header shares with what it describes, asked through
	// the one definition of what a sound is rather than repeated here.
	SoundData {
		samples: Vec::new(),
		rate: header.rate,
		channels: header.channels,
	}
	.check()?;

	if !header
		.sample_count
		.is_multiple_of(u32::from(header.channels))
	{
		return Err(format!(
			"declares {} samples, which is not a whole number of {}-channel frames",
			header.sample_count, header.channels
		));
	}

	Ok(*header)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Three frames of stereo.
	fn beep() -> SoundData {
		SoundData {
			samples: vec![1, -1, 2, -2, 3, -3],
			rate: 22_050,
			channels: 2,
		}
	}

	/// The beep, encoded.
	fn encoded() -> Vec<u8> { encode(&beep()).expect("a beep encodes") }

	/// The encoded beep, read back.
	fn opened() -> SoundFile {
		SoundFile::from_bytes(AlignedBytes::from_slice(&encoded())).expect("and reads back")
	}

	/// What reading these bytes went wrong with.
	fn refused(bytes: &[u8]) -> String {
		SoundFile::from_bytes(AlignedBytes::from_slice(bytes))
			.expect_err("the file is not valid")
			.to_string()
	}

	#[test]
	fn the_header_is_the_size_the_layout_depends_on() {
		assert_eq!(size_of::<SoundHeader>(), HEADER_BYTES, "thirty-two bytes, exactly");
		assert_eq!(align_of::<SoundHeader>(), 4, "with no padding in it");
		assert!(
			HEADER_BYTES.is_multiple_of(align_of::<i16>()),
			"so the samples after it can be read where they lie"
		);
	}

	#[test]
	fn a_sound_survives_the_trip_to_bytes_and_back() {
		for original in [beep(), SoundData::silence()] {
			let bytes = encode(&original).expect("it encodes");
			let file =
				SoundFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("and reads back");

			assert_eq!(file.to_sound_data(), original, "every sample, the rate and the channels");
		}
	}

	#[test]
	fn the_samples_are_borrowed_out_of_the_buffer_rather_than_copied() {
		let file = opened();
		let base = file.bytes.as_slice().as_ptr().addr();

		assert_eq!(
			file.samples().as_ptr().addr(),
			base + HEADER_BYTES,
			"the samples are the file's own bytes"
		);
		assert_eq!(file.samples(), &[1, -1, 2, -2, 3, -3], "all six of them");
		assert_eq!(file.header().rate, 22_050, "and the header is beside them");
	}

	#[test]
	fn a_sound_nothing_can_play_is_not_written() {
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
			"the shared check refuses it on the way out as well as on the way in"
		);
	}

	#[test]
	fn a_file_from_another_version_says_so_instead_of_panicking() {
		let mut bytes = encoded();
		bytes[8..12].copy_from_slice(&99_u32.to_le_bytes());

		let message = refused(&bytes);

		assert!(message.contains("version 99"), "it names the version found: {message}");
		assert!(message.contains("just assets"), "and what to do about it: {message}");
	}

	#[test]
	fn something_that_is_not_a_sound_at_all_is_refused() {
		let mut bytes = encoded();
		bytes[0] = b'X';

		assert!(refused(&bytes).contains("not a colby sound"), "the magic is checked first");
	}

	#[test]
	fn a_flag_this_build_does_not_know_is_refused() {
		let mut bytes = encoded();
		bytes[12..16].copy_from_slice(&1_u32.to_le_bytes());

		let message = refused(&bytes);

		assert!(
			message.contains("flag bits"),
			"an unknown flag means an unknown block, and a mixer that guessed would play noise: \
			 {message}"
		);
	}

	#[test]
	fn a_word_the_header_has_no_meaning_for_yet_is_refused() {
		let mut bytes = encoded();
		// the spare half-word sits twenty-two bytes in, after the channels.
		bytes[22..24].copy_from_slice(&1_u16.to_le_bytes());

		assert!(refused(&bytes).contains("no meaning yet"), "the same rule the flags follow");
	}

	#[test]
	fn a_block_that_runs_past_the_end_is_refused() {
		let mut bytes = encoded();
		// where the samples start, at twenty-eight.
		bytes[28..32].copy_from_slice(&9000_u32.to_le_bytes());

		assert!(refused(&bytes).contains("the file is"), "counted against its real length");
	}

	#[test]
	fn a_block_starting_inside_the_header_is_refused() {
		let mut bytes = encoded();
		bytes[28..32].copy_from_slice(&4_u32.to_le_bytes());

		assert!(
			refused(&bytes).contains("samples run from"),
			"a block overlapping the header is not a block"
		);
	}

	#[test]
	fn a_file_claiming_more_samples_than_any_sound_may_have_is_refused() {
		let mut bytes = encoded();
		// how many samples there are, at twenty-four.
		let past = u32::try_from(MAX_SAMPLES).unwrap_or(u32::MAX);
		bytes[24..28].copy_from_slice(&past.saturating_add(2).to_le_bytes());

		let message = refused(&bytes);

		assert!(message.contains("past the"), "it says what the limit is: {message}");
		assert!(
			message.contains(&MAX_SAMPLES.to_string()),
			"and refuses it before allocating for it: {message}"
		);
	}

	#[test]
	fn a_file_whose_rate_or_channels_are_impossible_is_refused_by_the_shared_check() {
		let mut zero_rate = encoded();
		zero_rate[16..20].copy_from_slice(&0_u32.to_le_bytes());

		assert!(refused(&zero_rate).contains("frames a second"));

		let mut too_many = encoded();
		too_many[20..22].copy_from_slice(&6_u16.to_le_bytes());

		assert!(refused(&too_many).contains("mono or stereo"));
	}

	#[test]
	fn a_file_whose_samples_do_not_divide_into_frames_is_refused() {
		let mut bytes = encoded();
		bytes[24..28].copy_from_slice(&5_u32.to_le_bytes());

		assert!(
			refused(&bytes).contains("whole number"),
			"half a stereo frame is a mixer reading one channel out of step forever"
		);
	}

	#[test]
	fn a_truncated_file_is_refused_rather_than_read_short() {
		assert!(refused(&encoded()[..20]).contains("too short"));
	}

	#[test]
	fn the_version_of_a_file_can_be_read_without_reading_the_file() {
		let path = std::env::temp_dir().join("colby-sound-version.csnd");

		std::fs::write(&path, encoded()).expect("the fixture is written");

		assert_eq!(version_of(&path), Some(FORMAT_VERSION));
		assert_eq!(version_of(Path::new("nothing.csnd")), None, "and a missing file is none");

		drop(std::fs::remove_file(&path));
	}
}
