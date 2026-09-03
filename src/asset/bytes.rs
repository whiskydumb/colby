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

/// A count as a header stores it.
///
/// @param value - how many of something there are
/// @param what - a noun phrase for them, which becomes the message: "a scene's
/// entities", "the mesh's vertices". The format's own voice is the argument
/// rather than a copy of the function
/// @return the count as a `u32`, or an asset error naming it
pub fn count(value: usize, what: &str) -> Result<u32> {
	u32::try_from(value)
		.map_err(|_| err!(Asset("{what} is {value}, more than a u32 can address")))
}

/// The width of one record as a header stores it.
///
/// @param what - a noun phrase for the record, as [`count`]
/// @return the size of `T` as a `u32`, or an asset error naming it
pub fn width<T>(what: &str) -> Result<u32> { count(size_of::<T>(), what) }

/// The string blob a compiled format's names live in.
///
/// Every one of colby's compiled formats holds fixed-size records, and a name
/// is not fixed-size, so each of them puts its names in one NUL-separated blob
/// at the end and stores an offset into it. This is that blob, and it was
/// written four separate times before it was written here.
///
/// **Offset zero is always the empty string.** That is what a record naming
/// nothing stores, so the blob opens with a terminator nobody asked for and a
/// reader looking there finds the end of a string immediately.
///
/// A name written twice is stored once: the second `put` finds the first and
/// hands back the same offset, so a picture shared by six materials costs one
/// copy of its name and six offsets. The lookup is a linear scan, which is the
/// right shape for the few dozen names a file has and would not be for
/// thousands.
#[derive(Clone, Debug, Default)]
pub struct Names {
	blob: Vec<u8>,
	written: Vec<(String, u32)>,
}

impl Names {
	/// Puts a name in, or finds the one already there.
	///
	/// @param name - what to write down, or the empty string for nothing
	/// @return where it starts in the blob
	pub fn put(&mut self, name: &str) -> u32 {
		if self.blob.is_empty() {
			self.blob.push(0);
		}

		if name.is_empty() {
			return 0;
		}

		if let Some((_, already)) = self
			.written
			.iter()
			.find(|(written, _)| written == name)
		{
			return *already;
		}

		let at = u32::try_from(self.blob.len()).unwrap_or(0);

		self.blob.extend_from_slice(name.as_bytes());
		self.blob.push(0);
		self.written.push((name.to_owned(), at));

		at
	}

	/// The whole blob, ready to be written at the end of a file.
	#[must_use]
	pub fn blob(&self) -> &[u8] { &self.blob }
}

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
	#[test]
	fn a_count_past_what_a_header_can_hold_is_refused_by_name() {
		assert_eq!(count(7, "a scene's records").ok(), Some(7), "an ordinary one passes");

		// only reachable on a target where a usize is wider than a u32, which
		// is every one this is built for. The four copies this function
		// replaced could not be tested here at all: each was private to a
		// format, and reaching this arm through one of them would have meant
		// four billion records.
		if usize::BITS > u32::BITS {
			let refused = count(usize::MAX, "a scene's records")
				.expect_err("more than four billion of anything is not a count");
			let said = format!("{refused}");

			assert!(said.contains("a scene's records"), "the caller's own voice: {said}");
			assert!(said.contains("more than a u32"), "and what is wrong with it: {said}");
		}
	}

	#[test]
	fn a_record_is_as_wide_as_the_type_that_describes_it() {
		assert_eq!(width::<u32>("a clip's records").ok(), Some(4));
		assert_eq!(width::<[u8; 44]>("a model's records").ok(), Some(44));
	}

	#[test]
	fn the_blob_opens_with_a_terminator_nobody_wrote() {
		let mut names = Names::default();

		assert_eq!(names.put(""), 0, "naming nothing is offset zero");
		assert_eq!(names.blob(), &[0], "and what is there is a terminator to find");
	}

	#[test]
	#[expect(
		clippy::naive_bytecount,
		reason = "the lint wants a crate for counting three bytes in a blob of forty"
	)]
	fn a_name_written_twice_is_stored_once_and_answers_the_same() {
		let mut names = Names::default();
		let first = names.put("models/lamp/tiles");
		let second = names.put("models/lamp/base");
		let again = names.put("models/lamp/tiles");

		assert_eq!(first, again, "the second asking finds the first copy");
		assert_ne!(first, second, "and two different names are two places");
		assert_eq!(
			names
				.blob()
				.iter()
				.filter(|byte| **byte == 0)
				.count(),
			3,
			"one terminator at the head and one after each of the two names"
		);
	}

	#[test]
	fn an_offset_is_where_a_name_starts_and_the_name_ends_at_a_nul() {
		let mut names = Names::default();
		// different lengths, so an offset that pointed at the wrong end of one
		// would land inside the other rather than on a terminator.
		let short = names.put("hub");
		let long = names.put("models/lamp/column");

		for (at, expected) in [(short, "hub"), (long, "models/lamp/column")] {
			let from = usize::try_from(at).expect("an offset is small");
			let rest = &names.blob()[from..];
			let end = rest
				.iter()
				.position(|byte| *byte == 0)
				.expect("every name is terminated");

			assert_eq!(
				std::str::from_utf8(&rest[..end]).ok(),
				Some(expected),
				"reading from {at} gives back what was put there"
			);
		}
	}
}
