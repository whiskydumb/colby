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

use std::{fs::File, io::Read, path::Path};

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
}
