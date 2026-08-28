//! colby's runtime texture format: `.ctex`.
//!
//! The same bargain as [`format`](crate::format): a fixed header and then bytes
//! the GPU takes as they are. Nothing decodes anything at runtime - a `.ctex`
//! is already RGBA8, already sRGB, already a whole mip chain.
//!
//! ```text
//!   0  TextureHeader                    64 bytes
//!  64  level 0                          width * height * 4
//!   .  level 1                          (width/2) * (height/2) * 4
//!   .  ...                              down to one texel
//! ```
//!
//! **The mip chain is built here, not at load time and not on the GPU.** It is
//! a box filter, and it runs in linear space: averaging sRGB bytes directly
//! makes every reduction darker than the image it came from, which shows up as
//! a floor that dims as it recedes. Decoding to linear, averaging and encoding
//! back costs nothing anyone will notice in a compiler that runs once per edit.

use std::path::Path;

use colby_core::{
	Result,
	abi::texture::{Texel, TextureData},
	bytemuck::{self, Pod, Zeroable},
	err,
};

use crate::bytes::{ALIGNMENT, AlignedBytes};

/// The eight bytes every `.ctex` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYTEX";

/// The revision of everything in this module.
pub const FORMAT_VERSION: u32 = 1;

/// The extension a compiled texture is written with.
pub const EXTENSION: &str = "ctex";

/// How big [`TextureHeader`] is, and where the first level starts.
pub const HEADER_BYTES: usize = 64;

/// The largest image the compiler will accept, on either side.
///
/// Not a GPU limit - it is well under every one of those. It is a limit on how
/// wrong a file has to be before the compiler stops rather than allocating
/// gigabytes because a header said to.
pub const MAX_SIZE: u32 = 16384;

/// The fixed head of a `.ctex`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct TextureHeader {
	/// [`MAGIC`]. Anything else is not one of these files.
	pub magic: [u8; 8],

	/// [`FORMAT_VERSION`] at the time the file was written.
	pub version: u32,

	/// Reserved. Every bit is zero, and a reader refuses a bit it does not
	/// know.
	pub flags: u32,

	/// Which [`Texel`] layout the levels are in.
	pub texel: u32,

	/// Width of the largest level.
	pub width: u32,

	/// Height of the largest level.
	pub height: u32,

	/// How many levels follow, at least one.
	pub levels: u32,

	/// Where the first level starts, in bytes from the start of the file.
	pub data_offset: u32,

	/// How many bytes every level takes together.
	pub data_bytes: u32,

	/// Padding to a multiple of the buffer alignment. Always zero.
	pub reserved: [u32; 6],
}

/// A `.ctex` held in memory, checked, and ready to be read in place.
#[derive(Clone, Debug)]
pub struct TextureFile {
	bytes: AlignedBytes,
	header: TextureHeader,
}

impl TextureFile {
	/// Reads and checks a compiled texture.
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
	pub const fn header(&self) -> &TextureHeader { &self.header }

	/// One mip level, borrowed out of the buffer.
	///
	/// @param level - which level, zero being the largest
	#[must_use]
	pub fn level(&self, level: u32) -> &[u8] {
		let Some(range) = self.level_range(level) else {
			return &[];
		};

		self.bytes
			.as_slice()
			.get(range)
			.unwrap_or_default()
	}

	/// Copies every level into an owned texture.
	///
	/// The one copy in the path, and it is here for the same reason it is in
	/// the mesh format: the registry owns its pixels, because a texture can
	/// also be generated.
	#[must_use]
	pub fn to_texture_data(&self) -> TextureData {
		let texel = Texel::from_code(self.header.texel).unwrap_or_default();
		let levels = (0..self.header.levels)
			.map(|level| self.level(level).to_vec())
			.collect();

		TextureData {
			width: self.header.width,
			height: self.header.height,
			texel,
			levels,
		}
	}

	/// Where one level sits in the file.
	fn level_range(&self, level: u32) -> Option<std::ops::Range<usize>> {
		if level >= self.header.levels {
			return None;
		}

		let texel = Texel::from_code(self.header.texel)?;
		let mut at = usize::try_from(self.header.data_offset).ok()?;

		for index in 0..=level {
			let bytes = level_bytes(self.header.width, self.header.height, index, texel)?;
			if index == level {
				return Some(at..at.checked_add(bytes)?);
			}

			at = at.checked_add(bytes)?;
		}

		None
	}
}

/// Writes a texture out as a `.ctex`.
///
/// @param data - the pixels and their chain
/// @return the whole file, ready to put on disk
pub fn encode(data: &TextureData) -> Result<Vec<u8>> {
	if !data.is_consistent() {
		return Err(err!(Asset(
			"the texture is {}x{} with {} levels, which do not add up",
			data.width,
			data.height,
			data.levels.len()
		)));
	}

	let levels = u32::try_from(data.levels.len())
		.map_err(|_| err!(Asset("a texture with more levels than a u32 can count")))?;
	let data_offset = u32::try_from(HEADER_BYTES)
		.map_err(|_| err!(Asset("the header does not fit in a u32 offset")))?;
	let data_bytes = u32::try_from(data.bytes())
		.map_err(|_| err!(Asset("the texture is larger than a u32 can address")))?;

	let header = TextureHeader {
		magic: MAGIC,
		version: FORMAT_VERSION,
		flags: 0,
		texel: data.texel.code(),
		width: data.width,
		height: data.height,
		levels,
		data_offset,
		data_bytes,
		reserved: [0; 6],
	};

	let mut out = Vec::with_capacity(HEADER_BYTES + data.bytes());
	out.extend_from_slice(bytemuck::bytes_of(&header));
	for level in &data.levels {
		out.extend_from_slice(level);
	}

	Ok(out)
}

/// The format version a file on disk was written by.
///
/// @ref [`format::version_of`](crate::format::version_of); this is its twin,
/// and it exists for the same reason: so a format bump rebuilds the tree
/// instead of leaving a message telling someone to pass `--force`.
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

/// Builds a whole mip chain from one image.
///
/// Each level is the box average of the one above it. Odd sizes clamp rather
/// than drop a row, which biases the last column very slightly and is invisible
/// next to the alternative of losing it.
///
/// **What is averaged depends on the layout, and it is not a nicety.** A color
/// is averaged in linear light, because averaging sRGB bytes directly makes
/// every reduction darker than the image it came from - a floor that dims as it
/// recedes. Anything else is averaged as it is stored, because its bytes are
/// numbers and the transfer function would bend them: a normal map put through
/// it comes back with every level tilted towards the flat direction.
///
/// A normal map's levels are not renormalized afterwards, on purpose. The
/// shader normalizes what it reads anyway, so the only thing renormalizing
/// would change is that the shortening a mip introduces would stop happening -
/// and that shortening is what makes a bumpy surface read smoother from far
/// away, which is what it should do.
///
/// @param width - the image's width
/// @param height - its height
/// @param base - `width * height * 4` bytes of RGBA
/// @param texel - which layout those bytes are in
/// @return every level, largest first
pub fn build_chain(width: u32, height: u32, base: Vec<u8>, texel: Texel) -> Result<Vec<Vec<u8>>> {
	let expected = level_bytes(width, height, 0, texel)
		.ok_or_else(|| err!(Asset("a texture of {width}x{height} does not fit in memory")))?;

	if base.len() != expected {
		return Err(err!(Asset(
			"a {width}x{height} image should be {expected} bytes, and this one is {}",
			base.len()
		)));
	}

	let count = TextureData::full_chain(width, height);
	let mut levels = Vec::with_capacity(usize::try_from(count).unwrap_or(1));
	levels.push(base);

	for level in 1..count {
		let (from_width, from_height) = size_at(width, height, level - 1);
		let (to_width, to_height) = size_at(width, height, level);
		let Some(previous) = levels.last() else {
			break;
		};

		let source = Level {
			bytes: previous,
			width: from_width,
			height: from_height,
			texel,
		};

		levels.push(halve(&source, to_width, to_height));
	}

	Ok(levels)
}

/// One level of a chain, as the level below it reads it.
struct Level<'a> {
	bytes: &'a [u8],
	width: u32,
	height: u32,
	texel: Texel,
}

/// Averages an image down to the next level.
fn halve(source: &Level<'_>, width: u32, height: u32) -> Vec<u8> {
	let mut out = Vec::with_capacity(
		usize::try_from(width).unwrap_or(0) * usize::try_from(height).unwrap_or(0) * 4,
	);

	for y in 0..height {
		for x in 0..width {
			out.extend(average(source, x, y));
		}
	}

	out
}

/// The four texels of the level above that one texel of this one covers.
///
/// Clamped at the edges, so an odd size biases its last row or column very
/// slightly rather than dropping it.
fn average(source: &Level<'_>, x: u32, y: u32) -> [u8; 4] {
	let color = source.texel.is_color();
	let mut sum = [0.0_f32; 4];

	for (offset_x, offset_y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
		let source_x = (x * 2 + offset_x).min(source.width.saturating_sub(1));
		let source_y = (y * 2 + offset_y).min(source.height.saturating_sub(1));
		let texel = texel_at(source.bytes, source.width, source_x, source_y);

		// the three color channels of an sRGB image go through the transfer
		// function and its alpha does not, because alpha was never a color. In
		// a layout that is not a color at all, neither does anything else.
		for (channel, total) in sum.iter_mut().enumerate().take(3) {
			*total += if color {
				to_linear(texel[channel])
			} else {
				f32::from(texel[channel]) / 255.0
			};
		}

		sum[3] += f32::from(texel[3]) / 255.0;
	}

	let encode = |value: f32| if color { from_srgb(value) } else { quantize(value) };

	[
		encode(sum[0] / 4.0),
		encode(sum[1] / 4.0),
		encode(sum[2] / 4.0),
		quantize(sum[3] / 4.0),
	]
}

/// One texel of an image, or opaque black outside it.
fn texel_at(bytes: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
	let offset = usize::try_from(y)
		.unwrap_or(0)
		.saturating_mul(usize::try_from(width).unwrap_or(0))
		.saturating_add(usize::try_from(x).unwrap_or(0))
		.saturating_mul(4);

	bytes
		.get(offset..offset + 4)
		.and_then(|slice| <[u8; 4]>::try_from(slice).ok())
		.unwrap_or([0, 0, 0, 255])
}

/// One sRGB byte as a linear value.
fn to_linear(value: u8) -> f32 {
	let value = f32::from(value) / 255.0;

	if value <= 0.040_45 {
		value / 12.92
	} else {
		((value + 0.055) / 1.055).powf(2.4)
	}
}

/// One linear value back as an sRGB byte.
fn from_srgb(value: f32) -> u8 {
	let encoded = if value <= 0.003_130_8 {
		value * 12.92
	} else {
		1.055_f32.mul_add(value.powf(1.0 / 2.4), -0.055)
	};

	quantize(encoded)
}

/// A value in `0.0 ..= 1.0` as a byte.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "clamped to 0..=255 on the line above the cast, which is the check the cast itself \
	          would not do"
)]
fn quantize(value: f32) -> u8 { (value * 255.0).clamp(0.0, 255.0).round() as u8 }

/// The size of a level, never smaller than one texel either way.
const fn size_at(width: u32, height: u32, level: u32) -> (u32, u32) {
	let shift = if level > 31 { 31 } else { level };
	let (width, height) = (width >> shift, height >> shift);

	(if width > 0 { width } else { 1 }, if height > 0 { height } else { 1 })
}

/// How many bytes one level of an image takes.
fn level_bytes(width: u32, height: u32, level: u32, texel: Texel) -> Option<usize> {
	let (width, height) = size_at(width, height, level);

	usize::try_from(width)
		.ok()?
		.checked_mul(usize::try_from(height).ok()?)?
		.checked_mul(texel.bytes())
}

/// Everything that has to hold before a [`TextureFile`] exists.
fn check(bytes: &[u8]) -> std::result::Result<TextureHeader, String> {
	const {
		assert!(size_of::<TextureHeader>() == HEADER_BYTES, "the header changed size");
		assert!(HEADER_BYTES.is_multiple_of(ALIGNMENT), "the levels would lose alignment");
	}

	let head = bytes.get(..HEADER_BYTES).ok_or_else(|| {
		format!(
			"only {} bytes long, too short to hold a {HEADER_BYTES}-byte header",
			bytes.len()
		)
	})?;

	let header: &TextureHeader = bytemuck::try_from_bytes(head)
		.map_err(|error| format!("the header could not be read: {error}"))?;

	if header.magic != MAGIC {
		return Err(format!(
			"not a colby texture: expected {:?} at the start, found {:?}",
			String::from_utf8_lossy(&MAGIC),
			String::from_utf8_lossy(&header.magic)
		));
	}

	if header.version != FORMAT_VERSION {
		return Err(format!(
			"written by texture format version {}, and this build reads version \
			 {FORMAT_VERSION}; run `just assets --force` to recompile it",
			header.version
		));
	}

	if header.flags != 0 {
		return Err(format!(
			"sets flag bits {:#010X} that this build does not know about",
			header.flags
		));
	}

	let texel = Texel::from_code(header.texel).ok_or_else(|| {
		format!("has texels laid out as {}, which this build cannot read", header.texel)
	})?;

	check_size(header)?;
	check_levels(header, texel, bytes.len())?;

	Ok(*header)
}

/// Checks that the header describes an image at all.
fn check_size(header: &TextureHeader) -> std::result::Result<(), String> {
	if header.width == 0 || header.height == 0 {
		return Err(format!("is {}x{}, which is not an image", header.width, header.height));
	}

	if header.width > MAX_SIZE || header.height > MAX_SIZE {
		return Err(format!(
			"is {}x{}, past the {MAX_SIZE} this build will read",
			header.width, header.height
		));
	}

	if header.levels == 0 {
		return Err("has no mip levels at all".to_owned());
	}

	let full = TextureData::full_chain(header.width, header.height);
	if header.levels > full {
		return Err(format!(
			"claims {} mip levels, and {}x{} has room for {full}",
			header.levels, header.width, header.height
		));
	}

	Ok(())
}

/// Checks that every level is inside the file.
fn check_levels(
	header: &TextureHeader,
	texel: Texel,
	len: usize,
) -> std::result::Result<(), String> {
	let mut total: usize = 0;
	for level in 0..header.levels {
		let bytes = level_bytes(header.width, header.height, level, texel)
			.ok_or_else(|| format!("has a level {level} whose size overflows"))?;

		total = total
			.checked_add(bytes)
			.ok_or_else(|| "has levels whose sizes overflow together".to_owned())?;
	}

	if usize::try_from(header.data_bytes).unwrap_or(usize::MAX) != total {
		return Err(format!(
			"says its levels are {} bytes, and {} levels of {}x{} are {total}",
			header.data_bytes, header.levels, header.width, header.height
		));
	}

	let offset = usize::try_from(header.data_offset).unwrap_or(usize::MAX);
	let end = offset
		.checked_add(total)
		.ok_or_else(|| "puts its levels past the end of memory".to_owned())?;

	if end > len {
		return Err(format!("declares levels ending at {end} in a file {len} bytes long"));
	}

	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	/// An image of a size, filled with one color.
	fn filled(width: u32, height: u32, color: [u8; 4]) -> Vec<u8> {
		let count = usize::try_from(width).unwrap_or(0) * usize::try_from(height).unwrap_or(0);

		color
			.iter()
			.copied()
			.cycle()
			.take(count * 4)
			.collect()
	}

	/// A texture of a size, with a chain built for it.
	fn chained(width: u32, height: u32, color: [u8; 4]) -> TextureData {
		TextureData {
			width,
			height,
			texel: Texel::Rgba8Srgb,
			levels: build_chain(width, height, filled(width, height, color), Texel::Rgba8Srgb)
				.expect("the chain builds"),
		}
	}

	/// The encoded fixture, read back.
	fn opened(data: &TextureData) -> TextureFile {
		let bytes = encode(data).expect("it encodes");

		TextureFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("and reads back")
	}

	#[test]
	fn the_header_is_the_size_the_layout_depends_on() {
		assert_eq!(size_of::<TextureHeader>(), HEADER_BYTES, "sixty-four bytes, exactly");
		assert_eq!(HEADER_BYTES % ALIGNMENT, 0, "so the levels stay aligned");
	}

	#[test]
	fn a_texture_survives_the_trip_to_bytes_and_back() {
		for data in [chained(8, 8, [10, 20, 30, 255]), chained(1, 1, [1, 2, 3, 4])] {
			assert_eq!(opened(&data).to_texture_data(), data, "every level, byte for byte");
		}
	}

	#[test]
	fn the_file_says_what_is_in_it() {
		let data = chained(8, 4, [255, 0, 0, 255]);
		let file = opened(&data);
		let header = file.header();

		assert_eq!(header.magic, MAGIC, "the magic is there");
		assert_eq!((header.width, header.height), (8, 4), "and the size");
		assert_eq!(header.levels, 4, "8, 4, 2, 1 - the longer side decides");
		assert_eq!(header.data_offset, 64, "the levels follow the header");
		assert_eq!(header.data_bytes, 4 * (32 + 8 + 2 + 1), "and add up to the total");
	}

	#[test]
	fn a_level_is_borrowed_out_of_the_buffer_rather_than_copied() {
		let data = chained(8, 8, [1, 2, 3, 4]);
		let file = opened(&data);
		let base = file.bytes.as_slice().as_ptr().addr();

		assert_eq!(file.level(0).as_ptr().addr(), base + 64, "the first level is the file");
		assert_eq!(
			file.level(1).as_ptr().addr(),
			base + 64 + 8 * 8 * 4,
			"and the second follows it"
		);
		assert_eq!(file.level(0).len(), 8 * 8 * 4, "at the length it should be");
		assert!(file.level(99).is_empty(), "and a level past the end is nothing");
	}

	#[test]
	fn a_flat_image_averages_to_the_same_color_all_the_way_down() {
		let levels = build_chain(8, 8, filled(8, 8, [128, 64, 32, 255]), Texel::Rgba8Srgb)
			.expect("it builds");

		assert_eq!(levels.len(), 4, "a whole chain");

		for (index, level) in levels.iter().enumerate() {
			// the round trip through linear costs at most one unit of the last
			// place; a filter with a gamma bug would be tens of units dark.
			assert!(
				level[0].abs_diff(128) <= 1 && level[1].abs_diff(64) <= 1,
				"level {index} drifted: {:?}",
				&level[..4]
			);
		}
	}

	#[test]
	fn averaging_happens_in_linear_light_not_in_srgb() {
		// half the texels black and half white. The correct average of those in
		// light is 0.5 linear, which is about 188 as an sRGB byte - not 127,
		// which is what averaging the bytes directly would give.
		let mut base = Vec::new();
		for index in 0..4 {
			let value = if index % 2 == 0 { 0 } else { 255 };
			base.extend([value, value, value, 255]);
		}

		let levels = build_chain(2, 2, base, Texel::Rgba8Srgb).expect("it builds");
		let average = levels[1][0];

		assert!(
			average.abs_diff(188) <= 2,
			"a checkerboard averages to mid grey in light, got {average}"
		);
	}

	#[test]
	fn a_layout_that_is_not_a_color_averages_its_bytes_as_they_are() {
		// the same checkerboard, declared as numbers. Half of nothing and half
		// of everything is half, which is 128 - and putting it through the
		// transfer function instead would answer 188, which as a normal map is
		// every mip level leaning the same way.
		let mut base = Vec::new();
		for index in 0..4 {
			let value = if index % 2 == 0 { 0 } else { 255 };
			base.extend([value, value, value, 255]);
		}

		let levels = build_chain(2, 2, base, Texel::Rgba8Unorm).expect("it builds");
		let average = levels[1][0];

		assert!(average.abs_diff(128) <= 1, "numbers average as numbers, got {average}");
	}

	#[test]
	fn every_level_of_a_chain_is_the_length_it_should_be() {
		for (width, height) in [(1, 1), (8, 8), (5, 3), (16, 2), (1, 32)] {
			let data = chained(width, height, [7, 7, 7, 255]);

			assert!(
				data.is_consistent(),
				"a {width}x{height} chain is consistent, got {:?}",
				data.levels
					.iter()
					.map(Vec::len)
					.collect::<Vec<_>>()
			);
		}
	}

	#[test]
	fn an_image_of_the_wrong_length_is_refused() {
		let error = build_chain(8, 8, vec![0; 10], Texel::Rgba8Srgb)
			.expect_err("ten bytes is not an 8x8 image");

		assert!(error.to_string().contains("256 bytes"), "and it says so: {error}");
	}

	#[test]
	fn a_texture_from_another_version_says_so_instead_of_panicking() {
		let mut bytes = encode(&chained(4, 4, [0, 0, 0, 255])).expect("it encodes");
		bytes[8..12].copy_from_slice(&99_u32.to_le_bytes());

		let error = TextureFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("version 99 is not this one");
		let message = error.to_string();

		assert!(message.contains("version 99"), "it names the version found: {message}");
		assert!(
			message.contains(&format!("version {FORMAT_VERSION}")),
			"and the one it wanted: {message}"
		);
	}

	#[test]
	fn a_texel_layout_this_build_does_not_know_is_refused() {
		let mut bytes = encode(&chained(4, 4, [0, 0, 0, 255])).expect("it encodes");
		bytes[16..20].copy_from_slice(&9_u32.to_le_bytes());

		let error = TextureFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("layout 9 does not exist");

		assert!(error.to_string().contains("laid out as 9"), "{error}");
	}

	#[test]
	fn a_level_count_the_size_cannot_hold_is_refused() {
		let mut bytes = encode(&chained(4, 4, [0, 0, 0, 255])).expect("it encodes");
		bytes[28..32].copy_from_slice(&9_u32.to_le_bytes());

		let error = TextureFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("4x4 has three levels, not nine");

		assert!(error.to_string().contains("room for 3"), "{error}");
	}

	#[test]
	fn a_truncated_texture_is_refused_rather_than_read_short() {
		let bytes = encode(&chained(8, 8, [0, 0, 0, 255])).expect("it encodes");

		for keep in [0, 32, 64, 100] {
			assert!(
				TextureFile::from_bytes(AlignedBytes::from_slice(&bytes[..keep])).is_err(),
				"a file cut to {keep} bytes is refused"
			);
		}
	}

	#[test]
	fn the_version_of_a_file_can_be_read_without_reading_the_file() {
		let path = std::env::temp_dir().join("colby-version-of.ctex");
		let bytes = encode(&chained(4, 4, [0, 0, 0, 255])).expect("it encodes");
		std::fs::write(&path, &bytes).expect("the fixture is written");

		assert_eq!(version_of(&path), Some(FORMAT_VERSION), "this build's version");

		std::fs::write(&path, b"not a texture").expect("the fixture is overwritten");

		assert_eq!(version_of(&path), None, "and something else reports nothing");

		drop(std::fs::remove_file(&path));
	}
}
