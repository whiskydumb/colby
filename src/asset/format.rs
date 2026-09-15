//! colby's runtime mesh format: `.cmesh`.
//!
//! The whole file is a fixed [`MeshHeader`] followed by two `#[repr(C)]`
//! blocks - vertices, then indices - and, for a mesh that needs them, three
//! more. Everything is little-endian, which is what every target the engine
//! builds for already is; a big-endian port would byte-swap on load and pay for
//! it there rather than making every loader on every machine pay for a decode
//! step.
//!
//! ```text
//!   0  MeshHeader                        96 bytes
//!  96  [MeshVertex; vertex_count]        48 bytes each
//!   .  [u32;        index_count]          4 bytes each
//!   .  [SkinVertex; skin_count]          12 bytes each, and only sometimes
//!   .  [MeshLevel;  level_count]         16 bytes each, and only sometimes
//!   .  [u32;        coarse_count]         4 bytes each, and only sometimes
//! ```
//!
//! **An optional block is there when its own count says so, and nothing else
//! says it.** The skin block was going to be a bit in [`MeshHeader::flags`],
//! and that turned out to be redundant state: two fields that can disagree, and
//! one of them would have had to win. The levels follow the skin's rule.
//!
//! **A coarser level is a record and a run of indices, not a mesh of its own.**
//! The triangles a level keeps are drawn over vertices the mesh already has, so
//! all a level adds is which of them: one [`MeshLevel`] each, saying where its
//! indices start in the run after the records, how many there are and how far
//! the level may stand from the mesh; then every level's indices back to back.
//! The index block before them stays the whole mesh, which is what every reader
//! that is not a picture reads - @ref
//! [`MeshData::levels`](colby_core::abi::MeshData::levels).
//!
//! The header is ninety-six bytes so that the vertex block inherits the
//! buffer's sixteen-byte alignment, and every block is exactly the layout the
//! GPU wants. Reading a mesh is therefore a file read into an
//! [`AlignedBytes`](crate::AlignedBytes) and a few `bytemuck` casts -
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
		mesh::{Level, MAX_LEVELS, MeshData, MeshVertex, SkinVertex},
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
///
/// Six since a mesh carries coarser levels: the header grew by four words to
/// say where two more blocks are, and the word it had kept in reserve became
/// the first of them.
pub const FORMAT_VERSION: u32 = 6;

/// The extension a compiled mesh is written with.
pub const EXTENSION: &str = "cmesh";

/// [`MeshHeader::flags`]: a sidecar beside the source was read into this.
///
/// The same bit a `.cmodel` carries and for the same reason: a `lamp.obj.model`
/// deleted moves nothing in the source tree, so without this the mesh would go
/// on standing at the scale of a file nobody can find. @ref
/// [`model::GUIDED`](crate::model::GUIDED), `crate::compile::is_stale`.
pub const GUIDED: u32 = 1 << 0;

/// Every flag bit this build knows.
const KNOWN_FLAGS: u32 = GUIDED;

/// How big [`MeshHeader`] is, and where the vertex block starts.
pub const HEADER_BYTES: usize = 96;

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

	/// What was true of this file when it was written, one bit each. A reader
	/// refuses a bit it does not know rather than ignoring it.
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

	/// Bytes per level record. Must be `size_of::<MeshLevel>()`, or zero when
	/// there are no levels.
	pub level_stride: u32,

	/// How many coarser levels the mesh carries, from none to
	/// [`MAX_LEVELS`].
	///
	/// The only thing that says whether the last two blocks are here at all.
	pub level_count: u32,

	/// Where the level records start, or zero when there are none.
	pub level_offset: u32,

	/// How many indices every level holds, added together.
	pub coarse_count: u32,

	/// Where those indices start, or zero when there are none.
	pub coarse_offset: u32,
}

/// One coarser level, as the file holds it.
///
/// @ref [`Level`](colby_core::abi::mesh::Level) for what a level is; this is
/// where its indices are in the file.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct MeshLevel {
	/// Where the level's indices start in the coarse block, counted in indices.
	///
	/// Always where the level before it stopped, so the records say nothing
	/// the counts do not - and a reader checks that they agree rather than
	/// trusting either.
	pub first: u32,

	/// How many indices the level draws. Always a multiple of three.
	pub count: u32,

	/// How far the level may stand from the mesh itself, in the mesh's units.
	pub error: f32,

	/// Nothing yet, and a reader refuses a record that puts something here: the
	/// rule [`MeshHeader::flags`] follows.
	pub spare: u32,
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

	/// The level records, borrowed out of the buffer.
	///
	/// Empty for a mesh that is only ever drawn whole.
	#[must_use]
	pub fn levels(&self) -> &[MeshLevel] {
		self.block(self.header.level_offset, self.header.level_count)
	}

	/// Every level's indices back to back, borrowed out of the buffer.
	#[must_use]
	pub fn coarse(&self) -> &[u32] {
		self.block(self.header.coarse_offset, self.header.coarse_count)
	}

	/// The bounding box the compiler measured.
	#[must_use]
	pub fn bounds(&self) -> (Vec3, Vec3) {
		(
			Vec3::from_array(self.header.bounds_min),
			Vec3::from_array(self.header.bounds_max),
		)
	}

	/// Copies the blocks into an owned mesh.
	///
	/// The one copy in the whole path, and it is here rather than in the format
	/// because the registry owns its geometry: a mesh can also be generated,
	/// and an entry that sometimes borrows a file and sometimes does not would
	/// be two types wearing one name.
	#[must_use]
	pub fn to_mesh_data(&self) -> MeshData {
		let coarse = self.coarse();

		MeshData {
			vertices: self.vertices().to_vec(),
			indices: self.indices().to_vec(),
			skin: self.skin().to_vec(),
			// every record was checked against the run before this struct
			// existed, so a range that does not fit is unreachable; an empty
			// level would be refused by the next writer rather than drawn
			levels: self
				.levels()
				.iter()
				.map(|level| Level {
					indices: run(level)
						.and_then(|range| coarse.get(range))
						.map(<[u32]>::to_vec)
						.unwrap_or_default(),
					error: level.error,
				})
				.collect(),
		}
	}

	/// One block, borrowed and reinterpreted.
	///
	/// Every offset and count was checked in [`check`] before this struct
	/// existed, so the fallback is unreachable. It is an empty slice rather
	/// than a panic because a mesh that draws nothing is a better failure than
	/// a dead process, and because the check that makes it unreachable is a
	/// few lines away rather than in this function.
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
pub fn encode(data: &MeshData) -> Result<Vec<u8>> { encode_marked(data, 0) }

/// The same, marked as having been imported through a sidecar.
///
/// A second entry point rather than a second parameter on [`encode`], because
/// the mark is a fact about one caller - the compiler, when a `.obj.model`
/// stood beside the source - and every other caller in the project would have
/// had to pass `false` forever. What it costs is that the two cannot drift:
/// this is the one that writes a header, and [`encode`] is it with no bits.
///
/// @param data - the geometry to write
/// @return the whole file, ready to put on disk
pub fn encode_guided(data: &MeshData) -> Result<Vec<u8>> { encode_marked(data, GUIDED) }

/// Writes a mesh out with the flag bits a caller asked for.
///
/// @param data - the geometry to write
/// @param flags - what to put in [`MeshHeader::flags`]
fn encode_marked(data: &MeshData, flags: u32) -> Result<Vec<u8>> {
	sound(data)?;

	let coarse: Vec<u32> = data
		.levels
		.iter()
		.flat_map(|level| level.indices.iter().copied())
		.collect();
	let records = level_records(data)?;

	let vertex_count = count(data.vertices.len(), "vertices")?;
	let index_count = count(data.indices.len(), "indices")?;
	let vertex_offset = count(HEADER_BYTES, "header")?;
	let skin_count = count(data.skin.len(), "skin entries")?;
	let level_count = count(records.len(), "levels")?;
	let coarse_count = count(coarse.len(), "indices of its levels")?;
	let too_large = || err!(Asset("the mesh is too large to address with 32-bit offsets"));
	let index_offset = vertex_offset
		.checked_add(vertex_count.saturating_mul(stride::<MeshVertex>()))
		.ok_or_else(too_large)?;
	let after_indices = index_offset
		.checked_add(index_count.saturating_mul(stride::<u32>()))
		.ok_or_else(too_large)?;
	let after_skin = after_indices
		.checked_add(skin_count.saturating_mul(stride::<SkinVertex>()))
		.ok_or_else(too_large)?;
	let after_levels = after_skin
		.checked_add(level_count.saturating_mul(stride::<MeshLevel>()))
		.ok_or_else(too_large)?;

	after_levels
		.checked_add(coarse_count.saturating_mul(stride::<u32>()))
		.ok_or_else(too_large)?;

	// zero rather than "where it would have been", because the offset of a
	// block that is not there is not a fact about the file.
	let (bounds_min, bounds_max) = data.bounds();
	let header = MeshHeader {
		magic: MAGIC,
		version: FORMAT_VERSION,
		flags,
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
		skin_offset: if skin_count == 0 { 0 } else { after_indices },
		level_stride: if level_count == 0 { 0 } else { stride::<MeshLevel>() },
		level_count,
		level_offset: if level_count == 0 { 0 } else { after_skin },
		coarse_count,
		coarse_offset: if coarse_count == 0 { 0 } else { after_levels },
	};

	let mut out = Vec::with_capacity(
		HEADER_BYTES
			+ data.vertices.len() * size_of::<MeshVertex>()
			+ data.indices.len() * size_of::<u32>()
			+ data.skin.len() * size_of::<SkinVertex>()
			+ records.len() * size_of::<MeshLevel>()
			+ coarse.len() * size_of::<u32>(),
	);
	out.extend_from_slice(bytemuck::bytes_of(&header));
	out.extend_from_slice(bytemuck::cast_slice(&data.vertices));
	out.extend_from_slice(bytemuck::cast_slice(&data.indices));
	out.extend_from_slice(bytemuck::cast_slice(&data.skin));
	out.extend_from_slice(bytemuck::cast_slice(&records));
	out.extend_from_slice(bytemuck::cast_slice(&coarse));

	Ok(out)
}

/// One record per level, each starting where the one before it stopped.
///
/// @param data - the mesh, whose levels were already found sound
fn level_records(data: &MeshData) -> Result<Vec<MeshLevel>> {
	let mut first = 0_u32;

	data.levels
		.iter()
		.map(|level| {
			let length = count(level.indices.len(), "indices of one level")?;
			let record = MeshLevel {
				first,
				count: length,
				error: level.error,
				spare: 0,
			};

			first = first
				.checked_add(length)
				.ok_or_else(|| err!(Asset("the mesh's levels hold more indices than a u32")))?;

			Ok(record)
		})
		.collect()
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

	if !data.levels_are_in_range() {
		return Err(err!(Asset(
			"the mesh has {} levels, and each has to be whole triangles over its {} vertices, \
			 at most {MAX_LEVELS} of them",
			data.levels.len(),
			data.vertices.len()
		)));
	}

	if !data.levels_thin_out() {
		return Err(err!(Asset(
			"the mesh has a level no thinner than the one before it, or nearer the mesh than \
			 that one"
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

/// The flag bits the file at this path sets, if it is one at all.
///
/// The head alone, so the staleness sweep can ask whether an output was built
/// through a sidecar without reading a file it may be about to rewrite. @ref
/// [`GUIDED`], `crate::compile::is_stale`.
///
/// @param path - the `.cmesh` to look at
#[must_use]
pub fn flags_of(path: &Path) -> Option<u32> {
	let mut head = [0_u8; 16];
	let mut file = std::fs::File::open(path).ok()?;
	std::io::Read::read_exact(&mut file, &mut head).ok()?;

	if head.get(..MAGIC.len()) != Some(&MAGIC[..]) {
		return None;
	}

	let flags: [u8; 4] = head.get(12..16)?.try_into().ok()?;

	Some(u32::from_le_bytes(flags))
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
		assert!(size_of::<MeshLevel>() == 16, "MeshLevel is no longer four words");
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

	if header.flags & !KNOWN_FLAGS != 0 {
		return Err(format!(
			"sets flag bits {:#010X} that this build does not know about",
			header.flags & !KNOWN_FLAGS
		));
	}

	check_strides(header)?;
	check_blocks(header, bytes.len())?;
	check_indices(bytes, header)?;
	check_skin(bytes, header)?;
	check_levels(bytes, header)?;

	Ok(*header)
}

/// Checks that every block is made of the elements this build expects.
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

	// and the same for a mesh that is only ever drawn whole
	let level_stride = if header.level_count == 0 {
		0
	} else {
		stride::<MeshLevel>()
	};

	if header.level_stride != level_stride {
		return Err(format!(
			"has {}-byte level records, and this build reads {level_stride}-byte ones",
			header.level_stride
		));
	}

	Ok(())
}

/// Checks that every block is inside the file and aligned where it sits.
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
		(
			"level",
			header.level_offset,
			span::<MeshLevel>(header.level_offset, header.level_count),
		),
		(
			"coarse index",
			header.coarse_offset,
			span::<u32>(header.coarse_offset, header.coarse_count),
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

/// Checks that the levels, if there are any, are levels of this mesh.
///
/// Everything [`MeshData::levels_are_in_range`] and
/// [`MeshData::levels_thin_out`] ask of a mesh before it is written, asked of
/// the bytes, plus what only the file can get wrong: a record that does not
/// start where the one before it stopped, a run the records do not add up to,
/// and a word that has no meaning yet.
fn check_levels(bytes: &[u8], header: &MeshHeader) -> std::result::Result<(), String> {
	if header.level_count == 0 {
		if header.level_offset != 0 || header.coarse_count != 0 || header.coarse_offset != 0 {
			return Err("says it has no levels and then says where their records or indices are"
				.to_owned());
		}

		return Ok(());
	}

	if usize::try_from(header.level_count).unwrap_or(usize::MAX) > MAX_LEVELS {
		return Err(format!(
			"has {} levels, and a mesh carries at most {MAX_LEVELS}",
			header.level_count
		));
	}

	let (records, coarse) = level_blocks(bytes, header)?;
	let mut before = (header.index_count, 0.0_f32);
	let mut first = 0_u32;

	for (at, record) in records.iter().enumerate() {
		if record.spare != 0 {
			return Err(format!("puts {:#010X} in level {at}'s spare word", record.spare));
		}

		if record.first != first {
			return Err(format!(
				"says level {at} starts at index {} of its run, where the level before it \
				 stopped at {first}",
				record.first
			));
		}

		if record.count == 0 || !record.count.is_multiple_of(3) {
			return Err(format!(
				"has {} indices in level {at}, which is not a whole number of triangles",
				record.count
			));
		}

		if record.count >= before.0 || !record.error.is_finite() || record.error < before.1 {
			return Err(format!(
				"has a level {at} of {} indices standing {} from the mesh, after one of {} \
				 standing {}: each level has to be thinner and no nearer",
				record.count, record.error, before.0, before.1
			));
		}

		first = first.saturating_add(record.count);
		before = (record.count, record.error);
	}

	if first != header.coarse_count {
		return Err(format!(
			"has levels holding {first} indices between them and a run of {}",
			header.coarse_count
		));
	}

	if let Some(past) = coarse
		.iter()
		.find(|index| **index >= header.vertex_count)
	{
		return Err(format!(
			"has a level index of {past} against only {} vertices",
			header.vertex_count
		));
	}

	Ok(())
}

/// The level records and the run of their indices, borrowed out of a file whose
/// blocks were already found to fit.
fn level_blocks<'a>(
	bytes: &'a [u8],
	header: &MeshHeader,
) -> std::result::Result<(&'a [MeshLevel], &'a [u32]), String> {
	let records: &[MeshLevel] = span::<MeshLevel>(header.level_offset, header.level_count)
		.and_then(|range| bytes.get(range))
		.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
		.ok_or_else(|| "has level records that cannot be read in place".to_owned())?;
	let coarse: &[u32] = span::<u32>(header.coarse_offset, header.coarse_count)
		.and_then(|range| bytes.get(range))
		.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
		.ok_or_else(|| "has level indices that cannot be read in place".to_owned())?;

	Ok((records, coarse))
}

/// Which indices of the coarse run one level holds, when the arithmetic fits.
///
/// Counted in indices rather than bytes: the run is already a `&[u32]` by the
/// time a level is looked up in it.
fn run(level: &MeshLevel) -> Option<std::ops::Range<usize>> {
	let start = usize::try_from(level.first).ok()?;

	Some(start..start.checked_add(usize::try_from(level.count).ok()?)?)
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

	/// Where each of the fields after the bounds starts.
	const SKIN_STRIDE_AT: usize = 64;
	const SKIN_COUNT_AT: usize = 68;
	const SKIN_OFFSET_AT: usize = 72;
	const LEVEL_STRIDE_AT: usize = 76;
	const LEVEL_COUNT_AT: usize = 80;
	const LEVEL_OFFSET_AT: usize = 84;
	const COARSE_COUNT_AT: usize = 88;
	const COARSE_OFFSET_AT: usize = 92;

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

	/// The cube with two coarser levels: ten of its triangles, then four.
	fn leveled() -> MeshData {
		let mut mesh = cube();

		mesh.levels = vec![
			Level {
				indices: mesh.indices[..30].to_vec(),
				error: 0.125,
			},
			Level {
				indices: mesh.indices[6..18].to_vec(),
				error: 0.5,
			},
		];

		mesh
	}

	/// The leveled cube, encoded.
	fn leveled_bytes() -> Vec<u8> { encode(&leveled()).expect("a leveled cube encodes") }

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

	/// What reading some bytes of a leveled cube went wrong with, after an
	/// edit.
	fn misread(edit: impl FnOnce(&mut Vec<u8>)) -> String {
		let mut bytes = leveled_bytes();
		edit(&mut bytes);

		MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("the file is no longer valid")
			.to_string()
	}

	/// A header word of a file, as a number.
	fn word(bytes: &[u8], at: usize) -> u32 {
		u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
	}

	/// Writes a header word of a file.
	fn put(bytes: &mut [u8], at: usize, value: u32) {
		bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
	}

	#[test]
	fn the_header_is_the_size_the_layout_depends_on() {
		assert_eq!(size_of::<MeshHeader>(), HEADER_BYTES, "ninety-six bytes, exactly");
		assert_eq!(align_of::<MeshHeader>(), 4, "and no padding beyond its fields");
		assert_eq!(HEADER_BYTES % ALIGNMENT, 0, "so the vertex block stays aligned");
		assert_eq!(size_of::<MeshLevel>(), 16, "and a level record is four words");
	}

	#[test]
	fn a_mesh_survives_the_trip_to_bytes_and_back() {
		for original in [cube(), quad(), MeshData::default(), leveled()] {
			let bytes = encode(&original).expect("it encodes");
			let file =
				MeshFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("and reads back");

			assert_eq!(file.to_mesh_data(), original, "every vertex, index and level");
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
			"and it comes after the index block, so the two that were always there did not move"
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
	fn a_flag_this_build_does_not_know_is_refused() {
		let mut bytes = encoded();

		// the lowest bit nothing has claimed, worked out rather than typed,
		// so that this still tests what it says the day a second flag lands
		bytes[12..16]
			.copy_from_slice(&(!KNOWN_FLAGS & KNOWN_FLAGS.wrapping_add(1)).to_le_bytes());

		let error = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("an unknown flag means an unknown block");

		assert!(error.to_string().contains("flag bits"), "and says which: {error}");
	}

	#[test]
	fn the_mark_a_sidecar_leaves_is_read_back_rather_than_refused() {
		let bytes = encode_guided(&cube()).expect("it writes");

		assert!(
			MeshFile::from_bytes(AlignedBytes::from_slice(&bytes)).is_ok(),
			"a bit this build knows is not an unknown block"
		);
		assert_eq!(
			bytes
				.get(12..16)
				.and_then(|four| four.try_into().ok()),
			Some(GUIDED.to_le_bytes()),
			"and it is the bit the compiler asked for"
		);
		assert_eq!(
			encode(&cube())
				.expect("it writes")
				.get(12..16)
				.and_then(|four| four.try_into().ok()),
			Some(0_u32.to_le_bytes()),
			"while a mesh with no sidecar beside it says nothing"
		);
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
		for keep in [0, 8, 63, 64, 95, 96, 100] {
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

	#[test]
	fn a_mesh_drawn_only_whole_carries_no_level_blocks_at_all() {
		let bytes = encoded();

		for (at, what) in [
			(LEVEL_STRIDE_AT, "record size"),
			(LEVEL_COUNT_AT, "count"),
			(LEVEL_OFFSET_AT, "offset"),
			(COARSE_COUNT_AT, "index count"),
			(COARSE_OFFSET_AT, "index offset"),
		] {
			assert_eq!(word(&bytes, at), 0, "a cube has no levels, so no {what} either");
		}

		assert_eq!(
			bytes.len(),
			HEADER_BYTES + 24 * size_of::<MeshVertex>() + 36 * size_of::<u32>(),
			"and the file ends where its indices do"
		);
		assert!(
			opened().levels().is_empty() && opened().coarse().is_empty(),
			"so there is nothing to borrow"
		);
	}

	#[test]
	fn the_levels_are_a_record_each_and_every_index_back_to_back_after_the_rest() {
		let bytes = leveled_bytes();
		let file =
			MeshFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("a leveled cube reads");
		let header = *file.header();
		let after_indices = HEADER_BYTES + 24 * size_of::<MeshVertex>() + 36 * size_of::<u32>();

		assert_eq!(header.level_stride, 16, "four words a record");
		assert_eq!(header.level_count, 2, "two levels");
		assert_eq!(
			usize::try_from(header.level_offset).ok(),
			Some(after_indices),
			"straight after the indices, because a cube has no skin"
		);
		assert_eq!(header.coarse_count, 42, "thirty indices and twelve");
		assert_eq!(
			usize::try_from(header.coarse_offset).ok(),
			Some(after_indices + 2 * 16),
			"straight after the two records"
		);
		assert_eq!(
			file.levels(),
			&[
				MeshLevel {
					first: 0,
					count: 30,
					error: 0.125,
					spare: 0
				},
				MeshLevel {
					first: 30,
					count: 12,
					error: 0.5,
					spare: 0
				},
			],
			"each starting where the one before stopped"
		);
		assert_eq!(&file.coarse()[30..], &cube().indices[6..18], "and the second's own run");
		assert_eq!(file.indices(), cube().indices.as_slice(), "while the mesh is still whole");
		assert_eq!(bytes.len(), after_indices + 2 * 16 + 42 * 4, "and nothing else follows");
	}

	#[test]
	fn a_skinned_mesh_with_levels_puts_the_levels_after_the_skin() {
		let mut mesh = skinned();
		mesh.levels = vec![Level {
			indices: mesh.indices[..3].to_vec(),
			error: 0.25,
		}];

		let bytes = encode(&mesh).expect("a skinned quad with a level encodes");
		let file = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("and reads");

		assert_eq!(file.to_mesh_data(), mesh, "every weight and every level");
		assert!(
			file.header().level_offset > file.header().skin_offset,
			"the records come after the skin, so the blocks before them did not move"
		);
	}

	#[test]
	fn levels_that_are_not_levels_of_the_mesh_are_not_written() {
		let mut past = leveled();
		past.levels[0].indices[0] = 24;

		assert!(refused(&past).contains("whole triangles"), "an index past the vertices");

		let mut thick = leveled();
		thick.levels[1].indices = thick.levels[0].indices.clone();

		assert!(refused(&thick).contains("no thinner"), "a level as thick as the one before");

		let mut near = leveled();
		near.levels[1].error = 0.0625;

		assert!(refused(&near).contains("nearer"), "a level nearer than the one before");

		let mut many = cube();
		many.levels = vec![Level { indices: vec![0, 1, 2], error: 0.0 }; MAX_LEVELS + 1];

		assert!(refused(&many).contains("at most"), "more levels than a mesh carries");
	}

	#[test]
	fn a_file_whose_levels_disagree_with_themselves_is_refused() {
		let records = HEADER_BYTES + 24 * size_of::<MeshVertex>() + 36 * size_of::<u32>();

		assert!(
			misread(|bytes| put(bytes, records + 16, 6)).contains("starts at index 6"),
			"a second level that does not start where the first stopped"
		);
		assert!(
			misread(|bytes| put(bytes, records + 4, 29)).contains("whole number of triangles"),
			"a level of twenty-nine indices"
		);
		assert!(
			misread(|bytes| {
				put(bytes, records + 20, 0);
				put(bytes, COARSE_COUNT_AT, 30);
			})
			.contains("whole number of triangles"),
			"a level of no indices at all, with a run that agrees with it"
		);
		assert!(
			misread(|bytes| put(bytes, records + 20, 36)).contains("thinner"),
			"a second level thicker than the first"
		);
		assert!(
			misread(|bytes| put(bytes, records + 24, 0.0625_f32.to_bits())).contains("no nearer"),
			"a second level nearer the mesh than the first"
		);
		assert!(
			misread(|bytes| put(bytes, records + 8, f32::NAN.to_bits())).contains("thinner"),
			"an error that is not a number"
		);
		assert!(
			misread(|bytes| put(bytes, records + 12, 1)).contains("spare word"),
			"a word that means nothing yet"
		);
		assert!(
			misread(|bytes| put(bytes, COARSE_COUNT_AT, 39)).contains("run of 39"),
			"a run the records do not add up to"
		);
		assert!(
			misread(|bytes| put(bytes, records + 2 * 16 + 30 * 4, 24)).contains("level index"),
			"a level index past the last vertex"
		);
		assert!(
			misread(|bytes| put(bytes, LEVEL_STRIDE_AT, 20)).contains("20-byte level"),
			"records written by a build with a different record"
		);
		assert!(
			misread(|bytes| {
				let near_the_end = u32::try_from(bytes.len() - 16).expect("a cube is small");

				put(bytes, LEVEL_OFFSET_AT, near_the_end);
			})
			.contains("level block"),
			"records that run past the end of the file"
		);
	}

	#[test]
	fn a_file_with_no_levels_that_says_where_they_are_is_refused() {
		for at in [LEVEL_OFFSET_AT, COARSE_COUNT_AT, COARSE_OFFSET_AT] {
			let mut bytes = encoded();
			put(&mut bytes, at, 8);

			let error = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
				.expect_err("it contradicts itself");

			assert!(
				error.to_string().contains("no levels"),
				"a word at {at} naming a block that is not there: {error}"
			);
		}
	}

	#[test]
	fn a_file_with_more_levels_than_a_mesh_carries_is_refused() {
		let mut mesh = cube();

		// seven levels of one triangle each would not thin out, so build seven
		// real ones out of a mesh big enough to thin out seven times
		mesh.indices = mesh.indices.repeat(8);
		mesh.levels = (1..=MAX_LEVELS)
			.map(|level| Level {
				indices: mesh.indices[..3 * (MAX_LEVELS + 1 - level) * 4].to_vec(),
				error: f32::from(u8::try_from(level).expect("a small number")),
			})
			.collect();

		let mut bytes = encode(&mesh).expect("as many levels as a mesh may carry");
		put(&mut bytes, LEVEL_COUNT_AT, u32::try_from(MAX_LEVELS + 1).expect("small"));

		let error = MeshFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("one more than that");

		assert!(error.to_string().contains("at most"), "and says the limit: {error}");
	}
}
