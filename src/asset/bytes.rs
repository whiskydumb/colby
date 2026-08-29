//! A byte buffer that is aligned enough to be cast in place.
//!
//! `bytemuck` will not hand out a `&[MeshVertex]` over a `&[u8]` whose address
//! is not a multiple of four, and a `Vec<u8>` promises nothing about its
//! address at all. That is the one thing standing between "read the file" and
//! "the file *is* the vertex buffer", so it is worth sixteen lines to remove.
//!
//! The trick is to allocate a `Vec` of an over-aligned element type and then
//! look at it as bytes: `Vec<T>` allocates with `align_of::<T>()`, so a vector
//! of sixteen-byte chunks starts on a sixteen-byte boundary and every offset
//! inside it that is a multiple of sixteen does too. No unsafe is involved -
//! `bytemuck` does the reinterpretation, both ways, and it is the same
//! reinterpretation it would refuse on a plain `Vec<u8>` only because that one
//! cannot promise anything.

use std::{fs::File, io::Read, ops::Range, path::Path};

use colby_core::{
	Result,
	bytemuck::{self, Pod, Zeroable},
	err,
};

/// The alignment [`AlignedBytes`] guarantees, in bytes.
///
/// Sixteen rather than four: it covers any `#[repr(C)]` block a future version
/// of the format might hold, including one with a `vec4` in it, and the header
/// is sized to a multiple of it so every block after the header inherits it.
pub const ALIGNMENT: usize = 16;

/// The allocation unit, whose alignment the whole buffer inherits.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct Chunk([u8; ALIGNMENT]);

/// Bytes guaranteed to start on an [`ALIGNMENT`]-byte boundary.
#[derive(Clone, Debug, Default)]
pub struct AlignedBytes {
	chunks: Vec<Chunk>,
	len: usize,
}

impl AlignedBytes {
	/// Reads a whole file into an aligned buffer.
	///
	/// @param path - the file to read
	/// @return its contents, ready to be cast in place
	pub fn read(path: &Path) -> Result<Self> {
		let mut file = File::open(path)
			.map_err(|error| err!(Asset("opening {}: {error}", path.display())))?;

		let len = usize::try_from(
			file.metadata()
				.map_err(|error| err!(Asset("reading the size of {}: {error}", path.display())))?
				.len(),
		)
		.map_err(|_| err!(Asset("{} does not fit in memory", path.display())))?;

		let mut buffer = Self::zeroed(len);
		file.read_exact(buffer.as_mut_slice())
			.map_err(|error| err!(Asset("reading {}: {error}", path.display())))?;

		Ok(buffer)
	}

	/// An aligned copy of some bytes.
	#[must_use]
	pub fn from_slice(bytes: &[u8]) -> Self {
		let mut buffer = Self::zeroed(bytes.len());
		buffer.as_mut_slice().copy_from_slice(bytes);

		buffer
	}

	/// A buffer of zeroes, `len` bytes long.
	#[must_use]
	pub fn zeroed(len: usize) -> Self {
		Self {
			chunks: vec![Chunk([0; ALIGNMENT]); len.div_ceil(ALIGNMENT)],
			len,
		}
	}

	/// The bytes. Their address is a multiple of [`ALIGNMENT`].
	#[must_use]
	pub fn as_slice(&self) -> &[u8] {
		let bytes: &[u8] = bytemuck::cast_slice(&self.chunks);

		bytes.get(..self.len).unwrap_or(bytes)
	}

	/// How many bytes there are, not counting the padding to a whole chunk.
	#[must_use]
	pub const fn len(&self) -> usize { self.len }

	/// Whether there are no bytes at all.
	#[must_use]
	pub const fn is_empty(&self) -> bool { self.len == 0 }

	/// The bytes, to write into.
	fn as_mut_slice(&mut self) -> &mut [u8] {
		let len = self.len;
		let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut self.chunks);

		if len > bytes.len() {
			return bytes;
		}

		&mut bytes[..len]
	}
}

/// Where a block of `T` sits in a file, when it sits anywhere sensible.
///
/// Shared by every format here that stores an offset and a count rather than
/// implying them, which is all of them: what a header says is arithmetic on
/// numbers a file supplied, and doing it in `usize` without checking is how a
/// corrupt file becomes a panic.
///
/// @note: on a sixty-four bit target the checked arithmetic here cannot
/// actually fail - a `u32` offset plus four thousand million records of any
/// reasonable size is nowhere near what a `usize` holds - so `None` is
/// unreachable and untested on the machine this is built on. It is not
/// decoration: on a thirty-two bit one the same numbers wrap, and the caller's
/// bounds check would then be comparing a range that came out backwards.
///
/// @param offset - where the block starts, as the header stores it
/// @param count - how many records, as the header stores it
/// @return the byte range, or nothing when the arithmetic does not fit
#[must_use]
pub fn span<T>(offset: u32, count: u32) -> Option<Range<usize>> {
	let start = usize::try_from(offset).ok()?;
	let length = usize::try_from(count)
		.ok()?
		.checked_mul(size_of::<T>())?;

	Some(start..start.checked_add(length)?)
}

/// Whether one block is inside the file it claims to be in, and readable
/// where it claims to be.
///
/// Three questions in one, and all three have to be asked before anything is
/// cast: the arithmetic has to fit, the range has to be inside the file and
/// clear of the header, and the start has to be on a boundary a `&[T]` may
/// begin at. @ref [`ALIGNMENT`], which is what every buffer here promises and
/// therefore the most a block can be asked to be.
///
/// @param bytes - the whole file
/// @param head - how many bytes the format's header is
/// @param block - where it starts and how many records, as the header stores
/// them
/// @param what - what to call the block in the message, in the plural
/// @return nothing, or why the block cannot be read
///
/// # Errors
///
/// If the block is not wholly inside the file, overlaps the header, or starts
/// somewhere a `&[T]` cannot.
pub fn fits<T>(
	bytes: &[u8],
	head: usize,
	block: (u32, u32),
	what: &str,
) -> std::result::Result<(), String> {
	let range = span::<T>(block.0, block.1)
		.ok_or_else(|| format!("the {what} are past the end of memory"))?;

	if range.start < head || range.end > bytes.len() {
		return Err(format!(
			"the {what} run from {} to {} and the file is {} bytes",
			range.start,
			range.end,
			bytes.len()
		));
	}

	if !range
		.start
		.is_multiple_of(align_of::<T>().min(ALIGNMENT))
	{
		return Err(format!("the {what} do not start on a boundary they can be read from"));
	}

	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_buffer_starts_on_the_alignment_it_promises() {
		for len in [0, 1, 15, 16, 17, 64, 1793] {
			let buffer = AlignedBytes::zeroed(len);
			let address = buffer.as_slice().as_ptr().addr();

			assert_eq!(
				address % ALIGNMENT,
				0,
				"a {len}-byte buffer starts at {address}, which is not {ALIGNMENT}-aligned"
			);
		}
	}

	#[test]
	fn the_length_is_the_one_asked_for_not_the_one_allocated() {
		let buffer = AlignedBytes::zeroed(17);

		assert_eq!(buffer.len(), 17, "seventeen bytes, in two chunks");
		assert_eq!(buffer.as_slice().len(), 17, "and the padding is not visible");
		assert!(!buffer.is_empty(), "seventeen is not none");
		assert!(AlignedBytes::zeroed(0).is_empty(), "and none is");
	}

	#[test]
	fn bytes_survive_the_trip_through_the_chunks() {
		let original: Vec<u8> = (0..70_u8).collect();
		let buffer = AlignedBytes::from_slice(&original);

		assert_eq!(buffer.as_slice(), original.as_slice(), "every byte, in order");
	}

	#[test]
	fn a_file_reads_back_exactly_as_written() {
		let path = std::env::temp_dir().join("colby-aligned-bytes.bin");
		let original: Vec<u8> = (0..=255_u8).cycle().take(1000).collect();
		std::fs::write(&path, &original).expect("the fixture is written");

		let buffer = AlignedBytes::read(&path).expect("and read back");

		assert_eq!(buffer.as_slice(), original.as_slice(), "byte for byte");
		assert_eq!(buffer.as_slice().as_ptr().addr() % ALIGNMENT, 0, "and still aligned");

		drop(std::fs::remove_file(&path));
	}

	#[test]
	fn a_missing_file_is_an_asset_error_naming_it() {
		let path = std::env::temp_dir().join("colby-no-such-file.bin");
		let error = AlignedBytes::read(&path).expect_err("there is nothing to read");

		assert!(
			error
				.to_string()
				.contains("colby-no-such-file.bin"),
			"the message says which file: {error}"
		);
	}
	/// A four-byte record, so a misaligned start is a thing that can happen.
	#[repr(C)]
	#[derive(Clone, Copy, Pod, Zeroable)]
	#[bytemuck(crate = "::colby_core::bytemuck")]
	struct Word(u32);

	#[test]
	fn a_block_inside_the_file_and_clear_of_the_header_reads() {
		let file = [0_u8; 64];

		assert!(fits::<Word>(&file, 16, (16, 12), "words").is_ok(), "sixteen to sixty-four");
		assert!(fits::<Word>(&file, 16, (16, 0), "words").is_ok(), "and an empty block is fine");
	}

	#[test]
	fn a_block_overlapping_the_header_is_refused() {
		let file = [0_u8; 64];
		let refused = fits::<Word>(&file, 16, (12, 4), "words").expect_err("it starts too early");

		assert!(refused.contains("words"), "the message names the block: {refused}");
	}

	#[test]
	fn a_block_past_the_end_of_the_file_is_refused() {
		let file = [0_u8; 64];
		let refused = fits::<Word>(&file, 16, (48, 8), "words").expect_err("it runs off the end");

		assert!(refused.contains("the file is 64 bytes"), "and says how long it is: {refused}");
	}

	#[test]
	fn a_block_that_could_not_be_cast_where_it_starts_is_refused() {
		let file = [0_u8; 64];
		let refused = fits::<Word>(&file, 16, (18, 4), "words").expect_err("eighteen is odd");

		assert!(refused.contains("boundary"), "the reason is the boundary: {refused}");
	}

	#[test]
	fn a_count_nothing_could_address_is_refused_rather_than_wrapping() {
		let file = [0_u8; 64];
		let refused = fits::<Word>(&file, 16, (16, u32::MAX), "words")
			.expect_err("four thousand million records are not in a sixty-four byte file");

		assert!(refused.contains("the file is 64 bytes"), "and it says so: {refused}");
		assert_eq!(
			span::<Word>(16, u32::MAX).map(|it| it.end),
			Some(16 + 4 * 0xFFFF_FFFF),
			"the arithmetic itself is exact here, which is what the bounds check then catches"
		);
	}
}
