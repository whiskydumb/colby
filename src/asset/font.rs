//! colby's runtime font format: `.cfont`.
//!
//! The same bargain the other two formats make: a fixed header, then blocks
//! that are already the shape the program wants. Nothing is decoded at load
//! time - the glyph table is an array of [`Glyph`] and the atlas is one byte
//! per texel, so reading a font is a file read and one `bytemuck` cast.
//!
//! ```text
//!   0  FontHeader                       64 bytes
//!  64  [Glyph; glyph_count]             24 bytes each, sorted by codepoint
//!   .  atlas                            atlas_width * atlas_height bytes
//! ```
//!
//! The atlas is a signed distance field rather than a picture of the letters:
//! `128` is on the outline, above is inside and below is outside, and
//! [`spread`](FontHeader::spread) says how far either side of the edge the
//! range reaches. @ref [`ttf`](crate::ttf) for what puts it there and why.

use std::path::Path;

use colby_core::{
	Result,
	abi::font::{FontData, Glyph},
	bytemuck::{self, Pod, Zeroable},
	err,
};

use crate::bytes::{ALIGNMENT, AlignedBytes};

/// The eight bytes every `.cfont` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYFNT";

/// The revision of everything in this module.
pub const FORMAT_VERSION: u32 = 1;

/// The extension a compiled font is written with.
pub const EXTENSION: &str = "cfont";

/// How big [`FontHeader`] is, and where the glyph table starts.
pub const HEADER_BYTES: usize = 64;

/// The largest atlas the reader will accept, on either side.
///
/// Not a GPU limit. It is how wrong a header has to be before the reader stops
/// rather than allocating what it was told to.
pub const MAX_ATLAS: u32 = 8192;

/// The most glyphs one font may hold.
pub const MAX_GLYPHS: u32 = 65536;

/// The fixed head of a `.cfont`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct FontHeader {
	/// [`MAGIC`]. Anything else is not one of these files.
	pub magic: [u8; 8],

	/// [`FORMAT_VERSION`] at the time the file was written.
	pub version: u32,

	/// Reserved. Every bit is zero, and a reader refuses a bit it does not
	/// know rather than ignoring it - which is what leaves room for a
	/// multi-channel field later without a format break.
	pub flags: u32,

	/// Bytes per glyph record. Must be `size_of::<Glyph>()`.
	pub glyph_stride: u32,

	/// How many glyphs the table holds.
	pub glyph_count: u32,

	/// Where the glyph table starts, in bytes from the start of the file.
	pub glyph_offset: u32,

	/// The atlas width, in texels.
	pub atlas_width: u32,

	/// The atlas height, in texels.
	pub atlas_height: u32,

	/// Where the atlas starts, in bytes from the start of the file.
	pub atlas_offset: u32,

	/// The em size the glyphs were rasterized at, in pixels.
	pub pixel_size: f32,

	/// How far above the baseline the tallest letters reach, in pixels.
	pub ascent: f32,

	/// How far below it the descenders drop, in pixels. Positive.
	pub descent: f32,

	/// Baseline to baseline, in pixels.
	pub line_height: f32,

	/// How far either side of an outline the field reaches, in pixels.
	pub spread: f32,

	/// Padding to a multiple of the buffer alignment. Always zero.
	pub reserved: [u32; 1],
}

/// A `.cfont` held in memory, checked, and ready to be read in place.
#[derive(Clone, Debug)]
pub struct FontFile {
	bytes: AlignedBytes,
	header: FontHeader,
}

impl FontFile {
	/// Reads and checks a compiled font.
	pub fn open(path: &Path) -> Result<Self> {
		let bytes = AlignedBytes::read(path)?;
		let header = check(bytes.as_slice())
			.map_err(|reason| err!(Asset("{}: {reason}", path.display())))?;

		Ok(Self { bytes, header })
	}

	/// Checks bytes that are already in memory.
	pub fn from_bytes(bytes: AlignedBytes) -> Result<Self> {
		let header = check(bytes.as_slice()).map_err(|reason| err!(Asset("{reason}")))?;

		Ok(Self { bytes, header })
	}

	/// The header, as it was read.
	#[must_use]
	pub const fn header(&self) -> &FontHeader { &self.header }

	/// The glyph table, borrowed out of the buffer.
	#[must_use]
	pub fn glyphs(&self) -> &[Glyph] {
		let start = length(self.header.glyph_offset);
		let end = start + length(self.header.glyph_count) * size_of::<Glyph>();

		self.bytes
			.as_slice()
			.get(start..end)
			.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
			.unwrap_or(&[])
	}

	/// The atlas, borrowed out of the buffer.
	#[must_use]
	pub fn atlas(&self) -> &[u8] {
		let start = length(self.header.atlas_offset);
		let end = start + length(self.header.atlas_width) * length(self.header.atlas_height);

		self.bytes
			.as_slice()
			.get(start..end)
			.unwrap_or(&[])
	}

	/// Copies the file into an owned font.
	///
	/// The one copy in the path, and it is here for the same reason a mesh's
	/// is: the registry owns its fonts, and an entry that sometimes borrows a
	/// file would be two types wearing one name.
	#[must_use]
	pub fn to_font_data(&self) -> FontData {
		FontData {
			pixel_size: self.header.pixel_size,
			ascent: self.header.ascent,
			descent: self.header.descent,
			line_height: self.header.line_height,
			spread: self.header.spread,
			atlas_width: self.header.atlas_width,
			atlas_height: self.header.atlas_height,
			atlas: self.atlas().to_vec(),
			glyphs: self.glyphs().to_vec(),
		}
	}
}

/// Writes a baked font out as a `.cfont`.
///
/// @param data - what [`ttf::import`](crate::ttf::import) produced
/// @return the whole file, ready to put on disk
pub fn encode(data: &FontData) -> Result<Vec<u8>> {
	if data.glyphs.is_empty() {
		return Err(err!(Asset("a font with no glyphs in it would draw nothing")));
	}

	if data
		.glyphs
		.windows(2)
		.any(|pair| pair[0].codepoint >= pair[1].codepoint)
	{
		return Err(err!(Asset(
			"the glyph table is not in codepoint order, and a reader binary searches it"
		)));
	}

	let expected = length(data.atlas_width) * length(data.atlas_height);
	if data.atlas.len() != expected {
		return Err(err!(Asset(
			"the atlas is {} bytes and its size says it should be {expected}",
			data.atlas.len()
		)));
	}

	let glyph_count = size(data.glyphs.len(), "glyphs")?;
	let glyph_offset = size(HEADER_BYTES, "header")?;
	let atlas_offset = glyph_offset
		.checked_add(glyph_count.saturating_mul(stride::<Glyph>()))
		.ok_or_else(|| err!(Asset("the font is too large to address with 32-bit offsets")))?;

	let header = FontHeader {
		magic: MAGIC,
		version: FORMAT_VERSION,
		flags: 0,
		glyph_stride: stride::<Glyph>(),
		glyph_count,
		glyph_offset,
		atlas_width: data.atlas_width,
		atlas_height: data.atlas_height,
		atlas_offset,
		pixel_size: data.pixel_size,
		ascent: data.ascent,
		descent: data.descent,
		line_height: data.line_height,
		spread: data.spread,
		reserved: [0; 1],
	};

	let mut out = Vec::with_capacity(HEADER_BYTES + data.atlas.len());
	out.extend_from_slice(bytemuck::bytes_of(&header));
	out.extend_from_slice(bytemuck::cast_slice(&data.glyphs));
	out.extend_from_slice(&data.atlas);

	Ok(out)
}

/// The format version a file on disk was written by.
///
/// @param path - a `.cfont`
/// @return the version it claims, if it claims one
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

/// Everything that has to hold before a [`FontFile`] exists.
fn check(bytes: &[u8]) -> std::result::Result<FontHeader, String> {
	const {
		assert!(size_of::<FontHeader>() == HEADER_BYTES, "the header changed size");
		assert!(HEADER_BYTES.is_multiple_of(ALIGNMENT), "the glyph table would lose alignment");
		assert!(size_of::<Glyph>() == 24, "a glyph record changed shape");
	}

	let head = bytes.get(..HEADER_BYTES).ok_or_else(|| {
		format!(
			"only {} bytes long, too short to hold a {HEADER_BYTES}-byte header",
			bytes.len()
		)
	})?;

	let header: &FontHeader = bytemuck::try_from_bytes(head)
		.map_err(|error| format!("the header could not be read: {error}"))?;

	if header.magic != MAGIC {
		return Err(format!(
			"not a colby font: expected {:?} at the start, found {:?}",
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

	if header.glyph_stride != stride::<Glyph>() {
		return Err(format!(
			"has {}-byte glyph records, and this build reads {}-byte ones",
			header.glyph_stride,
			stride::<Glyph>()
		));
	}

	check_sizes(header)?;
	check_blocks(header, bytes.len())?;

	Ok(*header)
}

/// Checks the numbers the reader is about to allocate against.
fn check_sizes(header: &FontHeader) -> std::result::Result<(), String> {
	if header.glyph_count == 0 {
		return Err("holds no glyphs, so nothing in it could be drawn".to_owned());
	}

	if header.glyph_count > MAX_GLYPHS {
		return Err(format!(
			"claims {} glyphs, and this build reads at most {MAX_GLYPHS}",
			header.glyph_count
		));
	}

	if header.atlas_width == 0 || header.atlas_height == 0 {
		return Err("has an atlas with no texels in it".to_owned());
	}

	if header.atlas_width > MAX_ATLAS || header.atlas_height > MAX_ATLAS {
		return Err(format!(
			"has a {}x{} atlas, and this build reads at most {MAX_ATLAS} on a side",
			header.atlas_width, header.atlas_height
		));
	}

	if !(header.pixel_size.is_finite() && header.pixel_size > 0.0) {
		return Err(format!(
			"was baked at a size of {}, which nothing can be scaled against",
			header.pixel_size
		));
	}

	if !(header.line_height.is_finite() && header.line_height > 0.0) {
		return Err(format!(
			"has a line height of {}, so every line of text would land on the last one",
			header.line_height
		));
	}

	Ok(())
}

/// Checks that both blocks are inside the file and do not overlap.
fn check_blocks(header: &FontHeader, len: usize) -> std::result::Result<(), String> {
	let glyphs_end = length(header.glyph_offset)
		.checked_add(length(header.glyph_count) * size_of::<Glyph>())
		.ok_or_else(|| "has a glyph table that runs past the end of memory".to_owned())?;

	let atlas_end = length(header.atlas_offset)
		.checked_add(length(header.atlas_width) * length(header.atlas_height))
		.ok_or_else(|| "has an atlas that runs past the end of memory".to_owned())?;

	if length(header.glyph_offset) < HEADER_BYTES {
		return Err("has a glyph table that starts inside the header".to_owned());
	}

	if glyphs_end > length(header.atlas_offset) {
		return Err("has a glyph table that runs into the atlas".to_owned());
	}

	if glyphs_end > len || atlas_end > len {
		return Err(format!(
			"says it is {} bytes long and the file is {len}",
			glyphs_end.max(atlas_end)
		));
	}

	Ok(())
}

/// A `usize` as the `u32` the format stores.
fn size(value: usize, what: &str) -> Result<u32> {
	u32::try_from(value).map_err(|_| err!(Asset("the font has more {what} than a u32 can count")))
}

/// The size of a record, for the header.
fn stride<T>() -> u32 { u32::try_from(size_of::<T>()).unwrap_or(0) }

/// A stored offset or count as a length.
fn length(value: u32) -> usize { usize::try_from(value).unwrap_or(0) }

#[cfg(test)]
mod tests {
	use super::*;

	/// A font of two glyphs and a four-texel atlas.
	fn tiny() -> FontData {
		FontData {
			pixel_size: 16.0,
			ascent: 12.0,
			descent: 4.0,
			line_height: 18.0,
			spread: 4.0,
			atlas_width: 2,
			atlas_height: 2,
			atlas: vec![1, 2, 3, 4],
			glyphs: vec![
				Glyph {
					codepoint: u32::from('a'),
					advance: 9.0,
					bearing_x: 1.0,
					bearing_y: 11.0,
					atlas_x: 0,
					atlas_y: 0,
					atlas_width: 1,
					atlas_height: 1,
				},
				Glyph {
					codepoint: u32::from('b'),
					advance: 9.5,
					bearing_x: 1.5,
					bearing_y: 12.0,
					atlas_x: 1,
					atlas_y: 0,
					atlas_width: 1,
					atlas_height: 1,
				},
			],
		}
	}

	#[test]
	fn a_font_survives_the_round_trip_unchanged() {
		let data = tiny();
		let bytes = encode(&data).expect("it encodes");
		let file = FontFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("and reads");

		assert_eq!(file.to_font_data(), data, "what came back is what went in");
	}

	#[test]
	fn the_glyph_table_is_read_in_place_rather_than_parsed() {
		let bytes = encode(&tiny()).expect("it encodes");
		let file = FontFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("and reads");

		assert_eq!(file.glyphs().len(), 2, "both glyphs are there");
		assert_eq!(
			file.glyphs().first().map(|glyph| glyph.codepoint),
			Some(u32::from('a')),
			"in the order they were written"
		);
		assert_eq!(file.atlas(), &[1, 2, 3, 4], "and the atlas follows them");
	}

	#[test]
	fn a_table_out_of_codepoint_order_is_refused_rather_than_written() {
		let mut data = tiny();
		data.glyphs.reverse();

		let error = encode(&data).expect_err("it is out of order");

		assert!(
			error.to_string().contains("order"),
			"a reader binary searches this table, so the message should say so: {error}"
		);
	}

	#[test]
	fn an_atlas_that_does_not_match_its_size_is_refused() {
		let mut data = tiny();
		data.atlas.push(5);

		assert!(encode(&data).is_err(), "five bytes is not a two by two atlas");
	}

	#[test]
	fn a_file_from_another_version_says_so_instead_of_being_read() {
		let mut bytes = encode(&tiny()).expect("it encodes");
		bytes[8] = FORMAT_VERSION.to_le_bytes()[0].wrapping_add(1);

		let error = FontFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("the version does not match");

		assert!(
			error.to_string().contains("--force"),
			"and the message should say how to fix it: {error}"
		);
	}

	#[test]
	fn a_file_that_is_not_one_of_these_is_refused_by_its_first_bytes() {
		let bytes = AlignedBytes::from_slice(b"COLBYMSH\0\0\0\0not a font at all, really");

		assert!(FontFile::from_bytes(bytes).is_err(), "the magic is a mesh's");
	}

	#[test]
	fn a_header_promising_more_than_the_file_holds_is_refused() {
		let mut bytes = encode(&tiny()).expect("it encodes");
		// four thousand glyphs in a file that holds two.
		bytes[16..20].copy_from_slice(&4000_u32.to_le_bytes());

		assert!(
			FontFile::from_bytes(AlignedBytes::from_slice(&bytes)).is_err(),
			"a reader that believed this would walk off the end of the buffer"
		);
	}

	#[test]
	fn a_version_is_readable_without_reading_the_whole_file() {
		let directory = std::env::temp_dir().join("colby-cfont-version");
		std::fs::create_dir_all(&directory).expect("the temp directory is writable");
		let path = directory.join("one.cfont");

		std::fs::write(&path, encode(&tiny()).expect("it encodes")).expect("it writes");

		assert_eq!(version_of(&path), Some(FORMAT_VERSION), "the head of the file says so");

		std::fs::write(&path, b"not a font").expect("it writes");
		assert_eq!(version_of(&path), None, "and something else says nothing");

		std::fs::remove_dir_all(&directory).ok();
	}
}
