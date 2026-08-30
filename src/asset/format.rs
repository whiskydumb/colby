//! colby's runtime mesh format: `.cmesh`.
//!
//! The whole file is a fixed [`MeshHeader`] followed by two `#[repr(C)]`
//! blocks - vertices, then indices - and, for a mesh bones move, a third.
//! Everything is little-endian, which is what every target the engine builds
//! for already is; a big-endian port would byte-swap on load and pay for it
//! there rather than making every loader on every machine pay for a decode
//! step.
//!
//! ```text
//!   0  MeshHeader                        80 bytes
//!  80  [MeshVertex; vertex_count]        48 bytes each
//!   .  [u32;        index_count]          4 bytes each
//!   .  [SkinVertex; skin_count]          12 bytes each, and only sometimes
//! ```
//!
//! **The skin block is the only optional one, and `skin_count` is the only
//! thing that says whether it is there.** It was going to be a bit in
//! [`MeshHeader::flags`], and that turned out to be redundant state: two fields
//! that can disagree, and one of them would have had to win. The flags word
//! keeps its documented job instead - it is what lets a *later* block be added
//! without moving the header, which this one could not do anyway, because the
//! header itself grew.
//!
//! The header is eighty bytes so that the vertex block inherits the buffer's
//! sixteen-byte alignment, and every block is exactly the layout the GPU
//! wants. Reading a mesh is therefore a file read into an
//! [`AlignedBytes`](crate::AlignedBytes) and two `bytemuck` casts -
//! [`MeshFile::vertices`] and [`MeshFile::indices`] borrow straight out of the
//! buffer and copy nothing.
//!
//! Every way the file can be wrong is checked once, in
//! [`MeshFile::from_bytes`], and reported as an
//! [`Error::Asset`](colby_core::Error::Asset) naming the file. Nothing in this
//! module panics on bad input: a mesh compiled by another version of the engine
//! is a message telling you to recompile, not a crash in a loader.

use std::path::Path;

use colby_core::{
	Result,
	abi::{
		mesh::{MeshData, MeshVertex, SkinVertex},
		skeleton::MAX_BONES,
	},
	bytemuck::{self, Pod, Zeroable},
	err,
	glam::Vec3,
};

use crate::bytes::{ALIGNMENT, AlignedBytes, span};

/// The eight bytes every `.cmesh` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYMSH";

/// The revision of everything in this module.
///
/// Bump it whenever the header or either block changes shape. A file carrying a
/// different number is refused with a message rather than read as if it agreed.
pub const FORMAT_VERSION: u32 = 4;

/// The extension a compiled mesh is written with.
pub const EXTENSION: &str = "cmesh";

/// How big [`MeshHeader`] is, and where the vertex block starts.
pub const HEADER_BYTES: usize = 80;

/// The fixed head of a `.cmesh`.
///
/// Offsets are stored rather than implied so that a later version can insert a
/// block without moving the ones after it, and counts are stored separately
/// from offsets so a reader can size its allocations before it looks at
/// anything else.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct MeshHeader {
	/// [`MAGIC`]. Anything else is not one of these files.
	pub magic: [u8; 8],

	/// [`FORMAT_VERSION`] at the time the file was written.
	pub version: u32,

	/// Reserved for optional blocks. Every bit is zero in version one, and a
	/// reader refuses a bit it does not know rather than ignoring it.
	pub flags: u32,

	/// Bytes per vertex. Must be `size_of::<MeshVertex>()`.
	pub vertex_stride: u32,

	/// Bytes per index. Must be four.
	pub index_stride: u32,

	/// How many vertices the vertex block holds.
	pub vertex_count: u32,

	/// How many indices the index block holds. Always a multiple of three.
	pub index_count: u32,

	/// Where the vertex block starts, in bytes from the start of the file.
	pub vertex_offset: u32,

	/// Where the index block starts, in bytes from the start of the file.
	pub index_offset: u32,

	/// The low corner of the mesh's axis-aligned bounding box.
	pub bounds_min: [f32; 3],

	/// The high corner of the same box.
	pub bounds_max: [f32; 3],

	/// Bytes per skin entry. Must be `size_of::<SkinVertex>()`, or zero when
	/// there is no skin block.
	pub skin_stride: u32,

	/// How many skin entries there are: either zero or `vertex_count`.
	///
	/// The only thing that says whether the third block is here at all.
	pub skin_count: u32,

	/// Where the skin block starts, or zero when there is none.
	pub skin_offset: u32,

	/// Nothing yet, and a reader refuses a file that puts something here.
	///
	/// It exists because the header has to be a multiple of sixteen bytes for
	/// the vertex block to stay aligned, and three new fields left four bytes
	/// over. Refusing rather than ignoring is the same rule
	/// [`flags`](Self::flags) follows.
	pub reserved: u32,
}

/// A `.cmesh` held in memory, checked, and ready to be read in place.
#[derive(Clone, Debug)]
pub struct MeshFile {
	bytes: AlignedBytes,
	header: MeshHeader,
}

impl MeshFile {
	/// Reads and checks a compiled mesh.
	///
	/// @param path - the `.cmesh` to read
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
	pub const fn header(&self) -> &MeshHeader { &self.header }

	/// The vertex block, borrowed out of the buffer.
	#[must_use]
	pub fn vertices(&self) -> &[MeshVertex] {
		self.block(self.header.vertex_offset, self.header.vertex_count)
	}

	/// The index block, borrowed out of the buffer.
	#[must_use]
	pub fn indices(&self) -> &[u32] {
		self.block(self.header.index_offset, self.header.index_count)
	}

	/// The skin block, borrowed out of the buffer.
	///
	/// Empty for a mesh nothing moves, which is almost all of them.
	#[must_use]
	pub fn skin(&self) -> &[SkinVertex] {
		self.block(self.header.skin_offset, self.header.skin_count)
	}

	/// The bounding box the compiler measured.
	#[must_use]
	pub fn bounds(&self) -> (Vec3, Vec3) {
		(
			Vec3::from_array(self.header.bounds_min),
			Vec3::from_array(self.header.bounds_max),
		)
	}

	/// Copies the two blocks into an owned mesh.
	///
	/// The one copy in the whole path, and it is here rather than in the format
	/// because the registry owns its geometry: a mesh can also be generated,
	/// and an entry that sometimes borrows a file and sometimes does not would
	/// be two types wearing one name.
	#[must_use]
	pub fn to_mesh_data(&self) -> MeshData {
		MeshData {
			vertices: self.vertices().to_vec(),
			indices: self.indices().to_vec(),
			skin: self.skin().to_vec(),
		}
	}

	/// One block, borrowed and reinterpreted.
	///
	/// Both offsets and both counts were checked in [`check`] before this
	/// struct existed, so the fallback is unreachable. It is an empty slice
	/// rather than a panic because a mesh that draws nothing is a better
	/// failure than a dead process, and because the check that makes it
	/// unreachable is a few lines away rather than in this function.
	fn block<T: Pod>(&self, offset: u32, count: u32) -> &[T] {
		let Some(range) = span::<T>(offset, count) else {
			return &[];
		};

		self.bytes
			.as_slice()
			.get(range)
			.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
			.unwrap_or(&[])
	}
}

/// Writes a mesh out as a `.cmesh`.
///
/// @param data - the geometry to write
/// @return the whole file, ready to put on disk
pub fn encode(data: &MeshData) -> Result<Vec<u8>> {
	sound(data)?;

	let vertex_count = count(data.vertices.len(), "vertices")?;
	let index_count = count(data.indices.len(), "indices")?;
	let vertex_offset = count(HEADER_BYTES, "header")?;
	let skin_count = count(data.skin.len(), "skin entries")?;
	let too_large = || err!(Asset("the mesh is too large to address with 32-bit offsets"));
	let index_offset = vertex_offset
		.checked_add(vertex_count.saturating_mul(stride::<MeshVertex>()))
		.ok_or_else(too_large)?;

	// zero rather than "where it would have been", because the offset of a
	// block that is not there is not a fact about the file.
	let skin_offset = if skin_count == 0 {
		0
	} else {
		index_offset
			.checked_add(index_count.saturating_mul(stride::<u32>()))
			.ok_or_else(too_large)?
	};

	let (bounds_min, bounds_max) = data.bounds();
	let header = MeshHeader {
		magic: MAGIC,
		version: FORMAT_VERSION,
		flags: 0,
		vertex_stride: stride::<MeshVertex>(),
		index_stride: stride::<u32>(),
		vertex_count,
		index_count,
		vertex_offset,
		index_offset,
		bounds_min: bounds_min.to_array(),
		bounds_max: bounds_max.to_array(),
		skin_stride: if skin_count == 0 { 0 } else { stride::<SkinVertex>() },
		skin_count,
		skin_offset,
		reserved: 0,
	};

	let mut out = Vec::with_capacity(
		HEADER_BYTES
			+ data.vertices.len() * size_of::<MeshVertex>()
			+ data.indices.len() * size_of::<u32>()
			+ data.skin.len() * size_of::<SkinVertex>(),
	);
	out.extend_from_slice(bytemuck::bytes_of(&header));
	out.extend_from_slice(bytemuck::cast_slice(&data.vertices));
	out.extend_from_slice(bytemuck::cast_slice(&data.indices));
	out.extend_from_slice(bytemuck::cast_slice(&data.skin));

	Ok(out)
}

/// Everything about a mesh that has to be true before it is worth writing.
///
/// Here rather than at load because it is the compiler's job to refuse
/// nonsense, and because a file colby wrote is trusted further than one it
/// merely found - @ref [`check`] for what is checked again anyway.
fn sound(data: &MeshData) -> Result<()> {
	if !data.indices_are_in_range() {
		return Err(err!(Asset(
			"the mesh has an index past the end of its {} vertices",
			data.vertices.len()
		)));
	}

	if !data.indices.len().is_multiple_of(3) {
		return Err(err!(Asset(
			"the mesh has {} indices, which is not a whole number of triangles",
			data.indices.len()
		)));
	}

	if !data.skin_fits() {
		return Err(err!(Asset(
			"the mesh has {} skin entries against {} vertices, and a mesh is either skinned all \
			 the way through or not at all",
			data.skin.len(),
			data.vertices.len()
		)));
	}

	if !data.weights_are_whole() {
		return Err(err!(Asset(
			"the mesh has a vertex whose bone weights do not add up to {}",
			SkinVertex::WHOLE
		)));
	}

	if !data.bones_are_in_range(MAX_BONES) {
		return Err(err!(Asset(
			"the mesh names a bone past the {MAX_BONES} a skeleton may hold"
		)));
	}

	Ok(())
}

/// The format version a file on disk was written by.
///
/// Reads the head of the file rather than the whole of it, and answers `None`
/// for anything that is not one of these files at all. The compiler uses this
/// to treat an output written by another version as stale - which is what turns
/// a `FORMAT_VERSION` bump into "it rebuilds" instead of "run it with --force".
///
/// @param path - a `.cmesh`
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

/// Everything that has to hold before a [`MeshFile`] exists.
///
/// @param bytes - the whole file
/// @return the header, or a sentence saying what is wrong with it
fn check(bytes: &[u8]) -> std::result::Result<MeshHeader, String> {
	const {
		assert!(size_of::<MeshHeader>() == HEADER_BYTES, "the header changed size");
		assert!(HEADER_BYTES.is_multiple_of(ALIGNMENT), "the vertex block would lose alignment");
		assert!(
			size_of::<MeshVertex>() == 48,
			"MeshVertex is no longer two vec3s, a vec2 and a vec4"
		);
		assert!(
			size_of::<SkinVertex>() == 12,
			"SkinVertex is no longer four shorts and four bytes"
		);
	}

	let head = bytes.get(..HEADER_BYTES).ok_or_else(|| {
		format!(
			"only {} bytes long, too short to hold a {HEADER_BYTES}-byte header",
			bytes.len()
		)
	})?;

	let header: &MeshHeader = bytemuck::try_from_bytes(head)
		.map_err(|error| format!("the header could not be read: {error}"))?;

	if header.magic != MAGIC {
		return Err(format!(
			"not a colby mesh: expected {:?} at the start, found {:?}",
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
			"puts {:#010X} in a header word that has no meaning yet",
			header.reserved
		));
	}

	check_strides(header)?;
	check_blocks(header, bytes.len())?;
	check_indices(bytes, header)?;
	check_skin(bytes, header)?;

	Ok(*header)
}

/// Checks that both blocks are made of the elements this build expects.
fn check_strides(header: &MeshHeader) -> std::result::Result<(), String> {
	if header.vertex_stride != stride::<MeshVertex>() {
		return Err(format!(
			"has {}-byte vertices, and this build reads {}-byte ones",
			header.vertex_stride,
			stride::<MeshVertex>()
		));
	}

	if header.index_stride != stride::<u32>() {
		return Err(format!(
			"has {}-byte indices, and this build reads {}-byte ones",
			header.index_stride,
			stride::<u32>()
		));
	}

	if !header.index_count.is_multiple_of(3) {
		return Err(format!(
			"has {} indices, which is not a whole number of triangles",
			header.index_count
		));
	}

	// zero and zero for a mesh nothing moves: the stride of a block that is
	// not there is not a fact about the file either.
	let skin_stride = if header.skin_count == 0 {
		0
	} else {
		stride::<SkinVertex>()
	};

	if header.skin_stride != skin_stride {
		return Err(format!(
			"has {}-byte skin entries, and this build reads {skin_stride}-byte ones",
			header.skin_stride
		));
	}

	Ok(())
}

/// Checks that both blocks are inside the file and aligned where they sit.
fn check_blocks(header: &MeshHeader, len: usize) -> std::result::Result<(), String> {
	let blocks = [
		(
			"vertex",
			header.vertex_offset,
			span::<MeshVertex>(header.vertex_offset, header.vertex_count),
		),
		(
			"index",
			header.index_offset,
			span::<u32>(header.index_offset, header.index_count),
		),
		(
			"skin",
			header.skin_offset,
			span::<SkinVertex>(header.skin_offset, header.skin_count),
		),
	];

	for (name, offset, range) in blocks {
		let range =
			range.ok_or_else(|| format!("declares a {name} block that overflows its offsets"))?;

		if range.end > len {
			return Err(format!(
				"declares a {name} block ending at {} in a file {len} bytes long",
				range.end
			));
		}

		if usize::try_from(offset).unwrap_or(usize::MAX) % 4 != 0 {
			return Err(format!("puts its {name} block at {offset}, which is not four-aligned"));
		}
	}

	Ok(())
}

/// Checks that every index addresses a vertex the file actually holds.
///
/// The GPU would not: it would read whatever is at the offset. One pass over
/// the indices at load time is cheap, and it turns a corrupt file into a
/// message instead of into geometry made of noise.
fn check_indices(bytes: &[u8], header: &MeshHeader) -> std::result::Result<(), String> {
	let Some(range) = span::<u32>(header.index_offset, header.index_count) else {
		return Err("declares an index block that overflows its offsets".to_owned());
	};

	let indices: &[u32] = bytes
		.get(range)
		.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
		.ok_or_else(|| "has an index block that cannot be read in place".to_owned())?;

	// the vertex block was checked to be readable by the same rule, so a
	// failure here would be the check disagreeing with itself.
	if bytes
		.get(span::<MeshVertex>(header.vertex_offset, header.vertex_count).unwrap_or(0..0))
		.and_then(|slice| bytemuck::try_cast_slice::<u8, MeshVertex>(slice).ok())
		.is_none()
	{
		return Err("has a vertex block that cannot be read in place".to_owned());
	}

	if let Some(past) = indices
		.iter()
		.find(|index| **index >= header.vertex_count)
	{
		return Err(format!(
			"has an index of {past} against only {} vertices",
			header.vertex_count
		));
	}

	Ok(())
}

/// Checks that a skin block, if there is one, could move this mesh.
///
/// The same argument [`check_indices`] makes: the GPU draws whatever the bytes
/// say, and what a garbled weight looks like on screen is a limb stretched to
/// the origin rather than an error. One pass at load turns that into a
/// sentence.
fn check_skin(bytes: &[u8], header: &MeshHeader) -> std::result::Result<(), String> {
	if header.skin_count == 0 {
		if header.skin_offset != 0 {
			return Err(format!(
				"says its skin block is at {} and then says it has no entries",
				header.skin_offset
			));
		}

		return Ok(());
	}

	if header.skin_count != header.vertex_count {
		return Err(format!(
			"has {} skin entries against {} vertices, and a mesh is either skinned all the way \
			 through or not at all",
			header.skin_count, header.vertex_count
		));
	}

	let Some(range) = span::<SkinVertex>(header.skin_offset, header.skin_count) else {
		return Err("declares a skin block that overflows its offsets".to_owned());
	};

	let skin: &[SkinVertex] = bytes
		.get(range)
		.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
		.ok_or_else(|| "has a skin block that cannot be read in place".to_owned())?;

	if let Some((at, entry)) = skin
		.iter()
		.enumerate()
		.find(|(_, entry)| !entry.is_sound())
	{
		return Err(format!(
			"has a vertex at {at} whose bone weights add up to {} rather than {}",
			entry.total(),
			SkinVertex::WHOLE
		));
	}

	if let Some(at) = skin
		.iter()
		.position(|entry| !entry.bones_below(MAX_BONES))
	{
		return Err(format!(
			"has a vertex at {at} naming a bone past the {MAX_BONES} a skeleton may hold"
		));
	}

	Ok(())
}

/// The size of `T` as the header stores it.
///
/// Saturating rather than checked: every `T` this is called with is two dozen
/// bytes, so the only way to reach the fallback would be a type that could not
/// be a vertex in the first place - and a stride of `u32::MAX` fails the
/// comparison it is used in, which is the right answer anyway.
fn stride<T>() -> u32 { u32::try_from(size_of::<T>()).unwrap_or(u32::MAX) }

/// A count as the header stores it.
fn count(value: usize, what: &str) -> Result<u32> {
	u32::try_from(value)
		.map_err(|_| err!(Asset("the mesh has {value} {what}, more than a u32 can address")))
}

#[cfg(test)]
mod tests {
	use colby_core::abi::mesh::{cube, quad};

	use super::*;

	/// A cube, encoded.
	fn encoded() -> Vec<u8> { encode(&cube()).expect("a cube encodes") }

	/// Where each of the four fields added in version four starts.
	const SKIN_STRIDE_AT: usize = 64;
	const SKIN_COUNT_AT: usize = 68;
	const SKIN_OFFSET_AT: usize = 72;
	const RESERVED_AT: usize = 76;

	/// A quad every vertex of which is pulled by bones, one of them by four.
	fn skinned() -> MeshData {
		let pulls = [
			SkinVertex::rigid(0),
			SkinVertex::rigid(3),
			SkinVertex {
				bones: [1, 2, 0, 0],
				weights: [128, 127, 0, 0],
			},
			SkinVertex {
				bones: [0, 1, 2, 3],
				weights: [64, 64, 64, 63],
			},
		];
		let mut mesh = quad();
		mesh.skin = pulls
			.iter()
			.copied()
			.cycle()
			.take(mesh.vertices.len())
			.collect();

		mesh
	}

	/// A skinned quad, encoded.
	fn skinned_bytes() -> Vec<u8> { encode(&skinned()).expect("a skinned quad encodes") }

	/// What encoding a mesh went wrong with.
	fn refused(data: &MeshData) -> String {
		encode(data)
			.expect_err("the mesh should not be written")
			.to_string()
	}

	/// The encoded cube, read back.
	fn opened() -> MeshFile {
		MeshFile::from_bytes(AlignedBytes::from_slice(&encoded())).expect("and reads back")
	}

	/// The encoded cube with one byte changed.
	fn tampered(offset: usize, value: u8) -> colby_core::Error {
		let mut bytes = encoded();
		bytes[offset] = value;

		MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("the file is no longer valid")
	}

	#[test]
	fn the_header_is_the_size_the_layout_depends_on() {
		assert_eq!(size_of::<MeshHeader>(), HEADER_BYTES, "eighty bytes, exactly");
		assert_eq!(align_of::<MeshHeader>(), 4, "and no padding beyond its fields");
		assert_eq!(HEADER_BYTES % ALIGNMENT, 0, "so the vertex block stays aligned");
	}

	#[test]
	fn a_mesh_survives_the_trip_to_bytes_and_back() {
		for original in [cube(), quad(), MeshData::default()] {
			let bytes = encode(&original).expect("it encodes");
			let file =
				MeshFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("and reads back");

			assert_eq!(file.to_mesh_data(), original, "every vertex and every index");
		}
	}

	#[test]
	fn the_file_says_what_is_in_it() {
		let file = opened();
		let header = file.header();

		assert_eq!(header.magic, MAGIC, "the magic is there");
		assert_eq!(header.version, FORMAT_VERSION, "and this build's version");
		assert_eq!(header.vertex_count, 24, "a cube's twenty-four vertices");
		assert_eq!(header.index_count, 36, "and its thirty-six indices");
		assert_eq!(
			header.vertex_offset,
			u32::try_from(HEADER_BYTES).expect("the header is small"),
			"the vertex block follows the header"
		);
		assert_eq!(
			header.index_offset,
			u32::try_from(HEADER_BYTES + 24 * size_of::<MeshVertex>()).expect("a cube is small"),
			"and the index block follows that"
		);
	}

	#[test]
	fn the_bounds_the_compiler_measured_come_back() {
		let (min, max) = opened().bounds();

		assert!(min.abs_diff_eq(Vec3::splat(-0.5), 1.0e-6), "the low corner, got {min}");
		assert!(max.abs_diff_eq(Vec3::splat(0.5), 1.0e-6), "the high corner, got {max}");
	}

	#[test]
	fn the_blocks_are_borrowed_out_of_the_buffer_rather_than_copied() {
		let file = opened();
		let base = file.bytes.as_slice().as_ptr().addr();
		let vertices = file.vertices().as_ptr().addr();
		let indices = file.indices().as_ptr().addr();

		assert_eq!(vertices, base + HEADER_BYTES, "the vertices are the file's own bytes");
		assert_eq!(
			indices,
			base + HEADER_BYTES + 24 * size_of::<MeshVertex>(),
			"and so are the indices"
		);
		assert_eq!(vertices % align_of::<MeshVertex>(), 0, "aligned where they sit");
		assert_eq!(indices % align_of::<u32>(), 0, "both of them");
	}

	#[test]
	fn a_file_from_another_version_says_so_instead_of_panicking() {
		let mut bytes = encoded();
		bytes[8..12].copy_from_slice(&99_u32.to_le_bytes());

		let error = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("version 99 is not this one");
		let message = error.to_string();

		assert!(message.contains("version 99"), "it names the version found: {message}");
		assert!(
			message.contains(&format!("version {FORMAT_VERSION}")),
			"and the one it wanted: {message}"
		);
		assert!(message.contains("just assets"), "and what to do about it: {message}");
	}

	#[test]
	fn something_that_is_not_a_mesh_at_all_is_refused() {
		let error = tampered(0, b'X');

		assert!(
			error.to_string().contains("not a colby mesh"),
			"the magic is checked first: {error}"
		);
	}

	#[test]
	fn a_mesh_nothing_moves_carries_no_skin_block_at_all() {
		let file = MeshFile::from_bytes(AlignedBytes::from_slice(&encoded()))
			.expect("a cube reads back");
		let header = file.header();

		assert_eq!(header.skin_count, 0, "a cube is moved by nothing");
		assert_eq!(header.skin_offset, 0, "so there is no offset to give");
		assert_eq!(header.skin_stride, 0, "and no entry size either");
		assert!(file.skin().is_empty(), "and nothing to read");
	}

	#[test]
	fn a_skinned_mesh_survives_the_trip_to_bytes_and_back() {
		let original = skinned();
		let file = MeshFile::from_bytes(AlignedBytes::from_slice(&skinned_bytes()))
			.expect("a skinned quad reads back");

		assert_eq!(file.to_mesh_data(), original, "every vertex, index and weight");
		assert_eq!(
			file.header().skin_count,
			file.header().vertex_count,
			"one entry per vertex, which is the only shape there is"
		);
		assert_eq!(file.header().skin_stride, 12, "four shorts and four bytes");
	}

	#[test]
	fn the_skin_block_is_borrowed_in_place_like_the_other_two() {
		let bytes = AlignedBytes::from_slice(&skinned_bytes());
		let file = MeshFile::from_bytes(bytes).expect("it reads back");
		let base = file.bytes.as_slice().as_ptr().addr();
		let skin = file.skin().as_ptr().addr();

		assert_eq!(
			skin,
			base + usize::try_from(file.header().skin_offset).expect("a quad is small"),
			"the skin is the file's own bytes"
		);
		assert_eq!(skin % align_of::<SkinVertex>(), 0, "aligned where it sits");
		assert!(
			skin > base + usize::try_from(file.header().index_offset).expect("still small"),
			"and it is the last block, so the two that were always there did not move"
		);
	}

	#[test]
	fn a_mesh_skinned_only_part_of_the_way_through_is_not_written() {
		let mut half = skinned();
		half.skin.pop();

		let message = refused(&half);

		assert!(message.contains("skin entries"), "it says what is short: {message}");
		assert!(
			message.contains("all the way through"),
			"and that there is no half measure: {message}"
		);
	}

	#[test]
	fn weights_that_do_not_add_up_to_a_whole_vertex_are_not_written() {
		let mut light = skinned();
		light.skin[0].weights[0] = 254;

		assert!(
			refused(&light).contains("do not add up"),
			"a vertex pulled by less than one bone's worth would drift towards the origin"
		);

		let mut heavy = skinned();
		heavy.skin[0] = SkinVertex {
			bones: [0, 1, 0, 0],
			weights: [255, 255, 0, 0],
		};

		assert!(refused(&heavy).contains("do not add up"), "and so is one pulled by two");
	}

	#[test]
	fn a_bone_past_what_any_skeleton_may_hold_is_not_written() {
		let mut wild = skinned();
		wild.skin[0].bones[0] = 4000;

		assert!(
			refused(&wild).contains("past the"),
			"the mesh does not know its own skeleton, so this is the only bound there is"
		);
	}

	#[test]
	fn a_bone_index_beside_a_weight_of_nothing_is_not_a_claim() {
		let mut idle = skinned();
		idle.skin[0].bones[3] = 9000;

		assert!(
			encode(&idle).is_ok(),
			"a bone with no weight is never read, so whatever sits beside it means nothing"
		);
	}

	#[test]
	fn a_file_claiming_a_skin_block_it_does_not_hold_is_refused() {
		let mut bytes = encoded();
		bytes[SKIN_COUNT_AT..SKIN_COUNT_AT + 4].copy_from_slice(&24_u32.to_le_bytes());

		let error = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("the block is not there");

		assert!(
			error.to_string().contains("skin entries"),
			"the stride gives it away first, which is fine: {error}"
		);
	}

	#[test]
	fn a_file_that_says_where_a_skin_is_and_then_that_there_is_none_is_refused() {
		let mut bytes = encoded();
		bytes[SKIN_OFFSET_AT..SKIN_OFFSET_AT + 4].copy_from_slice(&256_u32.to_le_bytes());

		let error = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("it contradicts itself");

		assert!(
			error.to_string().contains("no entries"),
			"and the message says which half is empty: {error}"
		);
	}

	#[test]
	fn a_skin_written_by_a_build_with_a_different_entry_is_refused() {
		let mut bytes = skinned_bytes();
		bytes[SKIN_STRIDE_AT..SKIN_STRIDE_AT + 4].copy_from_slice(&16_u32.to_le_bytes());

		let error = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("sixteen-byte entries are not these");

		assert!(error.to_string().contains("16-byte skin"), "and says so: {error}");
	}

	#[test]
	fn weights_garbled_on_disk_are_caught_at_load_rather_than_drawn() {
		let mut bytes = skinned_bytes();
		let weights = usize::try_from(
			MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
				.expect("it reads before it is broken")
				.header()
				.skin_offset,
		)
		.expect("a quad is small")
			+ 8;
		bytes[weights] = 3;

		let error = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("the first vertex no longer adds up");
		let message = error.to_string();

		assert!(message.contains("vertex at 0"), "it names the vertex: {message}");
		assert!(message.contains("255"), "and what the sum should have been: {message}");
	}

	#[test]
	fn a_word_the_header_has_no_meaning_for_yet_is_refused() {
		let mut bytes = encoded();
		bytes[RESERVED_AT..RESERVED_AT + 4].copy_from_slice(&1_u32.to_le_bytes());

		let error = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("something is in a word that means nothing");

		assert!(
			error.to_string().contains("no meaning yet"),
			"the same rule the flags word follows: {error}"
		);
	}

	#[test]
	fn a_flag_this_build_does_not_know_is_refused() {
		let mut bytes = encoded();
		bytes[12..16].copy_from_slice(&1_u32.to_le_bytes());

		let error = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("an unknown flag means an unknown block");

		assert!(error.to_string().contains("flag bits"), "and says which: {error}");
	}

	#[test]
	fn a_vertex_of_the_wrong_size_is_refused() {
		let mut bytes = encoded();
		bytes[16..20].copy_from_slice(&64_u32.to_le_bytes());

		let error = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("sixty-four byte vertices are not these ones");

		assert!(
			error
				.to_string()
				.contains(&format!("{}-byte", size_of::<MeshVertex>())),
			"and says what it wanted: {error}"
		);
	}

	#[test]
	fn a_truncated_file_is_refused_rather_than_read_short() {
		for keep in [0, 8, 63, 64, 100] {
			let bytes = &encoded()[..keep];
			let error = MeshFile::from_bytes(AlignedBytes::from_slice(bytes))
				.expect_err("a truncated file is not a whole cube");

			assert!(
				!error.to_string().is_empty(),
				"a file cut to {keep} bytes reports something"
			);
		}
	}

	#[test]
	fn an_index_past_the_last_vertex_is_refused() {
		let mut bytes = encoded();
		let first_index = HEADER_BYTES + 24 * size_of::<MeshVertex>();
		bytes[first_index..first_index + 4].copy_from_slice(&999_u32.to_le_bytes());

		let error = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("999 is past the twenty-fourth vertex");

		assert!(error.to_string().contains("999"), "and says which index: {error}");
	}

	#[test]
	fn a_block_that_runs_past_the_end_is_refused() {
		// bytes 24..28 are `vertex_count`: magic, version, flags and the two
		// strides come first, four bytes each after the eight-byte magic.
		let mut bytes = encoded();
		bytes[24..28].copy_from_slice(&9999_u32.to_le_bytes());

		let error = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("the file does not hold 9999 vertices");

		assert!(
			error.to_string().contains("bytes long"),
			"and says the file is not that big: {error}"
		);
	}

	#[test]
	fn the_version_of_a_file_can_be_read_without_reading_the_file() {
		let path = std::env::temp_dir().join("colby-version-of.cmesh");
		std::fs::write(&path, encoded()).expect("the fixture is written");

		assert_eq!(
			version_of(&path),
			Some(FORMAT_VERSION),
			"a file this build wrote reports this build's version"
		);

		std::fs::write(&path, b"not a mesh at all").expect("the fixture is overwritten");

		assert_eq!(version_of(&path), None, "and something else reports nothing");

		std::fs::write(&path, b"COLBY").expect("the fixture is truncated");

		assert_eq!(version_of(&path), None, "including something too short to have a version");

		drop(std::fs::remove_file(&path));
	}

	#[test]
	fn a_mesh_with_a_bad_index_is_refused_before_it_is_written() {
		let mut mesh = cube();
		mesh.indices[0] = 500;

		let error =
			encode(&mesh).expect_err("the compiler does not write geometry it cannot read");

		assert!(error.to_string().contains("past the end"), "{error}");
	}

	#[test]
	fn a_mesh_with_a_partial_triangle_is_refused_before_it_is_written() {
		let mut mesh = cube();
		mesh.indices.pop();

		let error = encode(&mesh).expect_err("thirty-five indices is not a triangle list");

		assert!(
			error
				.to_string()
				.contains("whole number of triangles"),
			"{error}"
		);
	}
}
