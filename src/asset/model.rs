//! colby's runtime model format: `.cmodel`.
//!
//! A model is not geometry and holds none. Its meshes are `.cmesh` files and
//! its pictures are `.ctex` files, written beside it under a directory of its
//! own name, and every one of them is an asset the engine already knows how to
//! load. What is left over is the two things nothing else could carry: **what
//! the file's surfaces are made of**, and **where each piece of it stands**.
//! That is the whole of this format.
//!
//! ```text
//!   0  ModelHeader                      64 bytes
//!  64  [Coat;  material_count]          36 bytes each
//!   .  [Stand; placement_count]         52 bytes each
//!   .  the string blob, NUL-separated UTF-8
//! ```
//!
//! Both record blocks are `#[repr(C)]` and cast in place out of an
//! [`AlignedBytes`](crate::AlignedBytes), the same trick a `.cmesh` uses. Names
//! cannot be, because they vary in length, so every name in a record is an
//! offset into one blob of NUL-terminated text at the end of the file. Offset
//! zero is always the empty string, which is what "this record names nothing"
//! means and is why the blob starts with a NUL nobody wrote.
//!
//! **Everything a record names, it names by asset name.** A placement says
//! `models/lamp/shade`, not "the third mesh"; a material says
//! `models/lamp/tiles`, not "the first picture". That is what lets the loader
//! resolve them through the registries every other asset already goes through,
//! and it is what makes a model survive its own meshes being recompiled: the
//! name stays, the registry entry is rewritten under it, and nothing here has
//! to be told.
//!
//! **A placement is world space.** glTF's node tree was flattened by the
//! importer, because an entity in this engine has no parent to hang a local
//! transform on - @ref `crate::gltf`. So a game reading this table spawns one
//! entity per placement and writes the transform it is handed.

use std::path::Path;

use colby_core::{
	Result,
	abi::{Transform, material::Wrap},
	bytemuck::{self, Pod, Zeroable},
	err,
	glam::{Quat, Vec3},
};

use crate::bytes::{AlignedBytes, fits, span};

/// The eight bytes every `.cmodel` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYMDL";

/// The revision of everything in this module.
///
/// Bump it whenever the header or either block changes shape. A file carrying a
/// different number is refused with a message rather than read as if it agreed.
pub const FORMAT_VERSION: u32 = 1;

/// The extension a compiled model is written with.
pub const EXTENSION: &str = "cmodel";

/// How big [`ModelHeader`] is, and where the first block starts.
pub const HEADER_BYTES: usize = 64;

/// The largest string blob the reader will accept, in bytes.
///
/// A model's names are a few dozen short paths. This is how wrong a file has to
/// be before the reader stops rather than allocating what it was told to.
pub const MAX_NAMES: usize = 1 << 20;

/// The fixed head of a `.cmodel`.
///
/// Offsets are stored rather than implied so that a later version can insert a
/// block without moving the ones after it - the same reasoning `.cmesh`'s
/// header follows, and the reason both have room to grow a joints block when
/// there is something to put in one.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct ModelHeader {
	/// [`MAGIC`]. Anything else is not one of these files.
	pub magic: [u8; 8],

	/// [`FORMAT_VERSION`] at the time the file was written.
	pub version: u32,

	/// Reserved for optional blocks. Every bit is zero in version one, and a
	/// reader refuses a bit it does not know rather than ignoring it.
	pub flags: u32,

	/// Bytes per material record. Must be `size_of::<Coat>()`.
	pub coat_stride: u32,

	/// Bytes per placement record. Must be `size_of::<Stand>()`.
	pub stand_stride: u32,

	/// How many materials the file declares.
	pub coat_count: u32,

	/// How many pieces stand somewhere.
	pub stand_count: u32,

	/// Where the material block starts, in bytes from the start of the file.
	pub coat_offset: u32,

	/// Where the placement block starts.
	pub stand_offset: u32,

	/// Where the string blob starts.
	pub names_offset: u32,

	/// How long the string blob is.
	pub names_length: u32,

	/// Spare, so the header is sixty-four bytes and the blocks after it inherit
	/// the buffer's alignment.
	pub reserved: [u32; 4],
}

// the whole point of the spare word is that the blocks after the header
// inherit the buffer's alignment, and a field added without shrinking it
// would move them without anybody noticing until a cast failed.
const _: () = assert!(
	size_of::<ModelHeader>() == HEADER_BYTES,
	"the header has to stay sixty-four bytes for the blocks after it to be readable"
);

/// What one surface of a model is made of.
///
/// The same numbers `abi::Material` holds, with an offset where each of its two
/// texture handles goes. Nothing here is a handle because no handle exists
/// until the host has registered what these names point at.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Coat {
	/// Offset into the blob of the name it registers under.
	pub name: u32,

	/// Offset of the albedo's asset name, or zero for none.
	pub albedo: u32,

	/// Offset of the normal map's asset name, or zero for none.
	pub normal: u32,

	/// What happens past the edge of both, as [`Wrap::code`].
	pub wrap: u32,

	/// Linear RGB.
	pub base_color: [f32; 3],

	/// Zero for a dielectric, one for a metal.
	pub metallic: f32,

	/// Zero is a mirror, one is chalk.
	pub roughness: f32,
}

/// One piece of a model standing somewhere.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Stand {
	/// Offset into the blob of what this piece is called.
	pub name: u32,

	/// Offset of the asset name of the mesh that stands here.
	pub mesh: u32,

	/// Offset of the material's name, or zero for the default one.
	pub material: u32,

	/// Where it stands, in world space.
	pub position: [f32; 3],

	/// How it is turned, xyzw.
	pub rotation: [f32; 4],

	/// How big it is along each axis. A negative one mirrors, and the mesh it
	/// names was written wound to match - @ref `crate::gltf`.
	pub scale: [f32; 3],
}

/// A model as plain data, before it is written or after it is read.
///
/// Names are owned strings here rather than offsets: the blob is an on-disk
/// detail and nothing above this module should have to know it exists.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelData {
	/// Every material, in the order the importer read them.
	pub materials: Vec<Material>,

	/// Every piece of the model and where it stands.
	pub placements: Vec<Placement>,
}

/// One material, with its pictures named.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Material {
	/// What it registers under.
	pub name: String,

	/// The albedo's asset name, or empty for none.
	pub albedo: String,

	/// The normal map's asset name, or empty for none.
	pub normal: String,

	/// Linear RGB.
	pub base_color: Vec3,

	/// Zero for a dielectric, one for a metal.
	pub metallic: f32,

	/// Zero is a mirror, one is chalk.
	pub roughness: f32,

	/// What happens past the edge of its pictures.
	pub wrap: Wrap,
}

/// One piece of a model, and where it stands.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Placement {
	/// What this piece is called.
	pub name: String,

	/// The asset name of the mesh that stands here.
	pub mesh: String,

	/// The material's name, or empty for the default one.
	pub material: String,

	/// Where it stands, with the whole tree above it already worked in.
	pub transform: Transform,
}

/// A `.cmodel` held in memory, checked, and ready to be read in place.
#[derive(Clone, Debug)]
pub struct ModelFile {
	bytes: AlignedBytes,
	header: ModelHeader,
}

impl ModelFile {
	/// Reads and checks a compiled model.
	///
	/// @param path - the `.cmodel` to read
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
	pub const fn header(&self) -> &ModelHeader { &self.header }

	/// The material block, borrowed out of the buffer.
	#[must_use]
	pub fn coats(&self) -> &[Coat] { self.block(self.header.coat_offset, self.header.coat_count) }

	/// The placement block, borrowed out of the buffer.
	#[must_use]
	pub fn stands(&self) -> &[Stand] {
		self.block(self.header.stand_offset, self.header.stand_count)
	}

	/// One name out of the blob.
	///
	/// @param offset - what a record stored
	/// @return the text up to its terminator, or nothing when the offset is not
	/// one this file wrote
	#[must_use]
	pub fn name(&self, offset: u32) -> &str {
		let Ok(start) = usize::try_from(offset) else {
			return "";
		};
		let Ok(base) = usize::try_from(self.header.names_offset) else {
			return "";
		};
		let Ok(length) = usize::try_from(self.header.names_length) else {
			return "";
		};

		let blob = self
			.bytes
			.as_slice()
			.get(base..base + length)
			.unwrap_or_default();
		let rest = blob.get(start..).unwrap_or_default();
		let end = rest
			.iter()
			.position(|byte| *byte == 0)
			.unwrap_or(rest.len());

		std::str::from_utf8(&rest[..end]).unwrap_or("")
	}

	/// Copies the whole file into owned data.
	///
	/// The one copy in the path, and it is here for the same reason a mesh's
	/// is: what the host holds can also be built rather than read, and an entry
	/// that sometimes borrows a file would be two types wearing one name.
	#[must_use]
	pub fn to_model_data(&self) -> ModelData {
		ModelData {
			materials: self
				.coats()
				.iter()
				.map(|coat| Material {
					name: self.name(coat.name).to_owned(),
					albedo: self.name(coat.albedo).to_owned(),
					normal: self.name(coat.normal).to_owned(),
					base_color: Vec3::from_array(coat.base_color),
					metallic: coat.metallic,
					roughness: coat.roughness,
					wrap: if coat.wrap == Wrap::Clamp.code() {
						Wrap::Clamp
					} else {
						Wrap::Repeat
					},
				})
				.collect(),
			placements: self
				.stands()
				.iter()
				.map(|stand| Placement {
					name: self.name(stand.name).to_owned(),
					mesh: self.name(stand.mesh).to_owned(),
					material: self.name(stand.material).to_owned(),
					transform: Transform {
						position: Vec3::from_array(stand.position),
						rotation: Quat::from_array(stand.rotation),
						scale: Vec3::from_array(stand.scale),
					},
				})
				.collect(),
		}
	}

	/// One block, borrowed and reinterpreted.
	///
	/// Every offset and count was checked in [`check`] before this struct
	/// existed, so the fallback is unreachable. It is an empty slice rather
	/// than a panic because a model that places nothing is a better failure
	/// than a dead process.
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

/// Writes a model out as a `.cmodel`.
///
/// Every distinct name is written into the blob once, so a picture shared by
/// six materials costs one copy of its name and six offsets.
///
/// @param data - the materials and placements to write
/// @return the whole file, ready to put on disk
pub fn encode(data: &ModelData) -> Result<Vec<u8>> {
	let mut names = Names::default();
	let coats: Vec<Coat> = data
		.materials
		.iter()
		.map(|material| Coat {
			name: names.put(&material.name),
			albedo: names.put(&material.albedo),
			normal: names.put(&material.normal),
			wrap: material.wrap.code(),
			base_color: material.base_color.to_array(),
			metallic: material.metallic,
			roughness: material.roughness,
		})
		.collect();
	let stands: Vec<Stand> = data
		.placements
		.iter()
		.map(|placement| Stand {
			name: names.put(&placement.name),
			mesh: names.put(&placement.mesh),
			material: names.put(&placement.material),
			position: placement.transform.position.to_array(),
			rotation: placement.transform.rotation.to_array(),
			scale: placement.transform.scale.to_array(),
		})
		.collect();

	let coat_offset = HEADER_BYTES;
	let stand_offset = coat_offset + size_of_val(coats.as_slice());
	let names_offset = stand_offset + size_of_val(stands.as_slice());
	let header = ModelHeader {
		magic: MAGIC,
		version: FORMAT_VERSION,
		flags: 0,
		coat_stride: width::<Coat>()?,
		stand_stride: width::<Stand>()?,
		coat_count: count(coats.len())?,
		stand_count: count(stands.len())?,
		coat_offset: count(coat_offset)?,
		stand_offset: count(stand_offset)?,
		names_offset: count(names_offset)?,
		names_length: count(names.blob.len())?,
		reserved: [0; 4],
	};

	let mut out = Vec::with_capacity(names_offset + names.blob.len());
	out.extend_from_slice(bytemuck::bytes_of(&header));
	out.extend_from_slice(bytemuck::cast_slice(&coats));
	out.extend_from_slice(bytemuck::cast_slice(&stands));
	out.extend_from_slice(&names.blob);

	Ok(out)
}

/// The version a `.cmodel` claims, without reading the rest of it.
///
/// What the compiler asks so that a file written by another build of the engine
/// is stale however new it is.
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

/// The blob being built, and where each name already in it starts.
#[derive(Default)]
struct Names {
	blob: Vec<u8>,
	written: Vec<(String, u32)>,
}

impl Names {
	/// Puts a name in, or finds the one already there.
	///
	/// The empty name is offset zero and is written once, at the head, because
	/// that is what a record naming nothing stores and a reader has to find a
	/// terminator there.
	fn put(&mut self, name: &str) -> u32 {
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
}

/// A count that has to fit in the header.
fn count(value: usize) -> Result<u32> {
	u32::try_from(value)
		.map_err(|_| err!(Asset("a model of {value} is more than one file holds")))
}

/// The width of a record, as the header stores it.
fn width<T>() -> Result<u32> { count(size_of::<T>()) }

/// Every way a `.cmodel` can be wrong, checked once.
fn check(bytes: &[u8]) -> std::result::Result<ModelHeader, String> {
	let head = bytes.get(..HEADER_BYTES).ok_or_else(|| {
		format!("a model is at least {HEADER_BYTES} bytes and this is {}", bytes.len())
	})?;
	let header: ModelHeader = *bytemuck::try_from_bytes(head)
		.map_err(|error| format!("the header could not be read: {error}"))?;

	if header.magic != MAGIC {
		return Err("this is not a colby model".to_owned());
	}

	if header.version != FORMAT_VERSION {
		return Err(format!(
			"this model is version {} and this build reads version {FORMAT_VERSION}; recompile \
			 it",
			header.version
		));
	}

	if header.flags != 0 {
		return Err(format!(
			"this model uses feature {:#x}, which this build does not",
			header.flags
		));
	}

	if usize::try_from(header.coat_stride) != Ok(size_of::<Coat>())
		|| usize::try_from(header.stand_stride) != Ok(size_of::<Stand>())
	{
		return Err("this model's records are not the size this build reads".to_owned());
	}

	if usize::try_from(header.names_length).unwrap_or(usize::MAX) > MAX_NAMES {
		return Err("this model's names are longer than any real one's".to_owned());
	}

	fits::<Coat>(bytes, HEADER_BYTES, (header.coat_offset, header.coat_count), "materials")?;
	fits::<Stand>(bytes, HEADER_BYTES, (header.stand_offset, header.stand_count), "placements")?;
	fits::<u8>(bytes, HEADER_BYTES, (header.names_offset, header.names_length), "names")?;

	Ok(header)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Half of a quarter turn, in the two places a unit quaternion holds it.
	const TURN: f32 = std::f32::consts::FRAC_1_SQRT_2;

	/// A model with everything a record can hold in it.
	fn sample() -> ModelData {
		ModelData {
			materials: vec![
				Material {
					name: "models/lamp/brass".to_owned(),
					albedo: "models/lamp/tiles".to_owned(),
					normal: "models/lamp/tiles_normal".to_owned(),
					base_color: Vec3::new(0.8, 0.6, 0.2),
					metallic: 1.0,
					roughness: 0.25,
					wrap: Wrap::Clamp,
				},
				Material {
					name: "models/lamp/glass".to_owned(),
					// the same picture, so the blob has one copy of its name
					albedo: "models/lamp/tiles".to_owned(),
					normal: String::new(),
					base_color: Vec3::ONE,
					metallic: 0.0,
					roughness: 0.1,
					wrap: Wrap::Repeat,
				},
			],
			placements: vec![
				Placement {
					name: "shade".to_owned(),
					mesh: "models/lamp/shade".to_owned(),
					material: "models/lamp/brass".to_owned(),
					transform: Transform {
						position: Vec3::new(1.0, 2.0, 3.0),
						rotation: Quat::from_xyzw(0.0, TURN, 0.0, TURN),
						scale: Vec3::new(1.0, 1.0, -1.0),
					},
				},
				Placement {
					name: "stem".to_owned(),
					mesh: "models/lamp/stem".to_owned(),
					material: String::new(),
					transform: Transform::IDENTITY,
				},
			],
		}
	}

	/// The sample, written and read back.
	fn round_trip(data: &ModelData) -> ModelFile {
		let bytes = encode(data).expect("it writes");

		ModelFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("and reads")
	}

	/// The sample with one byte of its header changed.
	fn corrupt(at: usize, to: u8) -> String {
		let mut bytes = encode(&sample()).expect("it writes");
		bytes[at] = to;

		ModelFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("and is refused")
			.to_string()
	}

	#[test]
	fn a_model_comes_back_as_what_was_written() {
		let data = sample();

		assert_eq!(round_trip(&data).to_model_data(), data);
	}

	#[test]
	fn a_name_used_twice_is_written_once() {
		let file = round_trip(&sample());
		let coats = file.coats();

		assert_eq!(coats[0].albedo, coats[1].albedo, "one offset, one copy in the blob");
		assert_eq!(file.name(coats[0].albedo), "models/lamp/tiles");
		assert!(
			file.header().names_length < 128,
			"the blob holds each name once and it is {} bytes",
			file.header().names_length
		);
	}

	#[test]
	fn naming_nothing_is_offset_zero_and_reads_as_nothing() {
		let file = round_trip(&sample());

		assert_eq!(file.coats()[1].normal, 0, "the material with no normal map");
		assert_eq!(file.stands()[1].material, 0, "and the placement with no material");
		assert_eq!(file.name(0), "");
	}

	#[test]
	fn the_records_are_the_width_the_header_promises() {
		let file = round_trip(&sample());

		assert_eq!(usize::try_from(file.header().coat_stride), Ok(size_of::<Coat>()));
		assert_eq!(usize::try_from(file.header().stand_stride), Ok(size_of::<Stand>()));
		assert_eq!(file.coats().len(), 2, "and both blocks cast in place");
		assert_eq!(file.stands().len(), 2);
	}

	#[test]
	fn a_transform_survives_the_trip_including_a_mirror() {
		let read = round_trip(&sample()).to_model_data();
		let shade = &read.placements[0];

		assert!(
			shade
				.transform
				.scale
				.abs_diff_eq(Vec3::new(1.0, 1.0, -1.0), 1e-6)
		);
		assert!(
			(shade.transform.rotation * Vec3::X).abs_diff_eq(-Vec3::Z, 1e-5),
			"and it still turns what it turned"
		);
	}

	#[test]
	fn a_wrap_this_build_does_not_know_reads_as_the_one_it_does() {
		let mut bytes = encode(&sample()).expect("it writes");
		let at = HEADER_BYTES + 12;

		bytes[at] = 0xFF;

		let file =
			ModelFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("it still reads");

		assert_eq!(file.to_model_data().materials[0].wrap, Wrap::Repeat);
	}

	#[test]
	fn a_file_that_is_not_one_of_these_is_refused_by_name() {
		assert!(corrupt(0, b'X').contains("not a colby model"));
		assert!(corrupt(8, 9).contains("version 9"));
		assert!(corrupt(12, 1).contains("feature"));
		assert!(corrupt(16, 99).contains("not the size this build reads"), "a record width");
	}

	#[test]
	fn a_block_that_does_not_fit_in_the_file_is_refused() {
		let mut bytes = encode(&sample()).expect("it writes");

		bytes[24..28].copy_from_slice(&9999_u32.to_le_bytes());

		let message = ModelFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("it is refused")
			.to_string();

		assert!(message.contains("materials run from"), "got {message}");
	}

	#[test]
	fn a_file_too_short_to_hold_a_header_is_refused() {
		let message = ModelFile::from_bytes(AlignedBytes::from_slice(&[0; 8]))
			.expect_err("it is refused")
			.to_string();

		assert!(message.contains("at least"), "got {message}");
	}

	#[test]
	fn a_model_with_nothing_in_it_is_still_a_model() {
		let file = round_trip(&ModelData::default());

		assert!(file.coats().is_empty());
		assert!(file.stands().is_empty());
		assert_eq!(file.to_model_data(), ModelData::default());
	}

	#[test]
	fn a_version_can_be_read_without_reading_the_rest() {
		let dir = std::env::temp_dir().join("colby-model-tests");

		std::fs::create_dir_all(&dir).expect("the directory is made");

		let path = dir.join("lamp.cmodel");

		std::fs::write(&path, encode(&sample()).expect("it writes")).expect("it is written");

		assert_eq!(version_of(&path), Some(FORMAT_VERSION));

		let other = dir.join("lamp.txt");

		std::fs::write(&other, b"not a model at all").expect("it is written");

		assert_eq!(version_of(&other), None);

		drop(std::fs::remove_dir_all(&dir));
	}
}
