//! colby's runtime model format: `.cmodel`.
//!
//! A model is not geometry and holds none. Its meshes are `.cmesh` files and
//! its pictures are `.ctex` files, written beside it under a directory of its
//! own name, and every one of them is an asset the engine already knows how to
//! load. What is left over is the three things nothing else could carry: **what
//! the file's surfaces are made of**, **where each piece of it stands**, and
//! **what the file's lamps shine**. That is the whole of this format.
//!
//! ```text
//!   0  ModelHeader                      64 bytes
//!  64  [Coat;  material_count]         100 bytes each
//!   .  [Stand; placement_count]         56 bytes each
//!   .  [Lit;   lamp_count]              40 bytes each
//!   .  the string blob, NUL-separated UTF-8
//! ```
//!
//! Every record block is `#[repr(C)]` and cast in place out of an
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
//!
//! **A lamp is a placement with nothing to draw and a light.** Its light is a
//! scene's [`Lit`] record keyed by the placement's place in the block, the way
//! a scene keys it by the entity's: one light written down is one record in
//! either file, read by the same rules. Nearly every piece of a model shines
//! nothing, so a block beside the placements costs a model of forty pieces and
//! two lamps eighty bytes rather than forty bytes a piece.

use std::path::Path;

use colby_core::{
	Result,
	abi::{
		Light, Material as Surface, TextureId, Transform,
		material::{Blend, Wrap},
	},
	bytemuck::{self, Pod, Zeroable},
	err,
	glam::{Quat, Vec2, Vec3},
};

use crate::{
	bytes::{AlignedBytes, Names, count, fits, span, width},
	scene::{Lit, light_of, lit_of},
};

/// The eight bytes every `.cmodel` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYMDL";

/// The revision of everything in this module.
///
/// Bump it whenever the header or a block changes shape. A file carrying a
/// different number is refused with a message rather than read as if it agreed.
pub const FORMAT_VERSION: u32 = 7;

/// The extension a compiled model is written with.
pub const EXTENSION: &str = "cmodel";

/// [`ModelHeader::flags`]: a sidecar beside the source was read into this.
///
/// The file's own record of how it came to be, for somebody looking at a
/// compiled model and asking why it stands at the scale it does. **Nothing in
/// the compiler reads it.** It used to be a second answer to "the sidecar was
/// deleted", and that question is settled by the input list a pass writes
/// down, which names a sidecar only while it is there: deleting one shortens
/// the list and the model rebuilds. @ref `crate::compile::extra_inputs`,
/// `crate::import`.
pub const GUIDED: u32 = 1 << 0;

/// Every flag bit this build knows.
const KNOWN_FLAGS: u32 = GUIDED;

/// [`Coat::flags`]: the surface is drawn as its own color, with no light on it.
pub const UNLIT: u32 = 1 << 0;

/// [`Coat::flags`]: the occlusion picture is read from the second set of
/// coordinates.
pub const OCCLUSION_UV2: u32 = 1 << 1;

/// [`Coat::flags`]: the glow picture is read from the second set of coordinates.
pub const GLOW_UV2: u32 = 1 << 2;

/// Every bit of [`Coat::flags`] this build knows.
///
/// **A bit it does not know is refused**, the header's rule rather than the
/// scene's: every one of these changes how a surface is drawn, so a surface
/// read without one it carried would be drawn as something it is not, and the
/// version number has already caught every file this build is older than.
const KNOWN_COAT_FLAGS: u32 = UNLIT | OCCLUSION_UV2 | GLOW_UV2;

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

	/// What was true of this file when it was written, one bit each.
	///
	/// [`GUIDED`] is the only one, and a reader refuses a bit it does not know
	/// rather than ignoring it.
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

	/// Bytes per lamp record. Must be `size_of::<Lit>()`.
	pub lit_stride: u32,

	/// How many placements shine something.
	pub lit_count: u32,

	/// Where the lamp block starts.
	pub lit_offset: u32,

	/// Spare, so the header is sixty-four bytes and the blocks after it inherit
	/// the buffer's alignment. The last of four; the lamps took the other
	/// three.
	pub reserved: u32,
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
/// The same numbers `abi::Material` holds, with an offset where each of its
/// five picture handles goes. Nothing here is a handle because no handle exists
/// until the host has registered what these names point at.
///
/// **The record a `.cmat` writes too**, with its name at nought because a file
/// of its own is named by its path: one material written down is one record
/// whichever file it is in. @ref `crate::material`.
///
/// The fields after [`uv_scale`](Self::uv_scale) arrived together, with the
/// rest of the exchange format's material, and are appended rather than put
/// beside the fields they belong with, so that every offset before them stayed
/// where it was.
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

	/// What happens past the edge of every picture, as [`Wrap::code`].
	pub wrap: u32,

	/// Linear RGB.
	pub base_color: [f32; 3],

	/// Zero for a dielectric, one for a metal.
	pub metallic: f32,

	/// Nought is as smooth as a surface is drawn, and one is chalk.
	pub roughness: f32,

	/// How its alpha is read, as [`Blend::code`].
	///
	/// A code rather than a flag, so a build that meets one it does not know
	/// refuses the file - @ref [`Blend::from_code`] and [`check`].
	pub blend: u32,

	/// How much of the surface there is, where the mode above reads it.
	pub opacity: f32,

	/// How many times the textures repeat across the mesh's own `0..1`.
	///
	/// Added with the `.material` source, and the reason it is here rather
	/// than only there is that this record is *the* described material - the
	/// live one has had the field since materials had textures, and two
	/// described forms of one record is two things to keep in step.
	pub uv_scale: [f32; 2],

	/// Offset of the metal and roughness picture's asset name, or zero for
	/// none.
	pub finish: u32,

	/// Offset of the occlusion picture's asset name, or zero for none.
	pub occlusion: u32,

	/// Offset of the glow picture's asset name, or zero for none.
	pub glow: u32,

	/// How much of the occlusion picture is applied.
	pub occlusion_strength: f32,

	/// The light the surface gives off, linear RGB.
	pub emissive: [f32; 3],

	/// How bright that light is, as a multiplier.
	pub emissive_strength: f32,

	/// Where the pictures on the first set of coordinates start.
	pub uv_offset: [f32; 2],

	/// How far they are turned, in radians.
	pub uv_rotation: f32,

	/// [`UNLIT`], [`OCCLUSION_UV2`] and [`GLOW_UV2`]; a bit this build does
	/// not know is refused.
	pub flags: u32,
}

// a record's size is written into every file's header and checked against
// this build's, so a field added without meaning to be is caught here first
const _: () = assert!(size_of::<Coat>() == 100, "a material record is a hundred bytes");

/// One piece of a model standing somewhere.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Stand {
	/// Offset into the blob of what this piece is called.
	pub name: u32,

	/// Offset of the asset name of the mesh that stands here, or zero for a
	/// lamp, whose light is in the lamp block.
	pub mesh: u32,

	/// Offset of the material's name, or zero for the default one.
	pub material: u32,

	/// Offset of the skeleton's name, or zero for a piece bones do not move.
	pub skeleton: u32,

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

	/// Whether a sidecar beside the source was read into this.
	///
	/// Not a description of the model - nothing about what it is made of or
	/// where it stands depends on it - but a fact about how this file came to
	/// say what it says, and the only one the compiler cannot work out again
	/// by looking at the source tree. @ref [`GUIDED`].
	pub guided: bool,
}

/// One material, with its pictures named.
///
/// **The numbers are the live record's own**: an `abi::Material` whose five
/// picture handles are left at what its default holds, because no handle exists
/// until the host has registered what a name points at, and the names beside it
/// in the five places the handles will go. So a field the live record grows is
/// a field this form carries without being taught it, and the one thing that
/// has to know both halves is [`live`](Self::live).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Material {
	/// What it registers under.
	pub name: String,

	/// The albedo's asset name, or empty for none.
	pub albedo: String,

	/// The normal map's asset name, or empty for none.
	pub normal: String,

	/// The metal and roughness picture's asset name, or empty for none.
	pub finish: String,

	/// The occlusion picture's asset name, or empty for none.
	pub occlusion: String,

	/// The glow picture's asset name, or empty for none.
	pub glow: String,

	/// Every number, with the five pictures' handles at the default's.
	pub surface: Surface,
}

impl Material {
	/// The live record this describes, each picture's handle whatever `handle`
	/// makes of its name.
	///
	/// **The one conversion**, and every material on its way into a world goes
	/// through it: a file somebody wrote, a model's own, and the editor's
	/// pictures of either. Four copies of it by hand had already dropped a
	/// field between them - a model's material lost its `uv_scale` on the way
	/// in - which is what one function is for.
	///
	/// @param handle - what a picture's asset name resolves to, asked for the
	/// albedo, the normal map, the finish, the occlusion and the glow in that
	/// order, an empty name included
	/// @return the record, with a normal map that resolves to nothing held as
	/// the flat one, which is the ABI's own rewrite
	pub fn live<F>(&self, mut handle: F) -> Surface
	where
		F: FnMut(&str) -> TextureId,
	{
		let albedo = handle(&self.albedo);
		let normal = handle(&self.normal);
		let finish = handle(&self.finish);
		let occlusion = handle(&self.occlusion);
		let glow = handle(&self.glow);

		Surface {
			albedo,
			finish,
			occlusion,
			glow,
			..self.surface
		}
		.bumped(normal)
	}

	/// A live record written down: its numbers, and its pictures by name.
	///
	/// The other way round from [`live`](Self::live), for whatever puts a
	/// world's material back into a file.
	///
	/// @param name - what it registers under
	/// @param live - the record
	/// @param named - a picture's asset name, or empty for none
	#[must_use]
	pub fn described<F>(name: &str, live: &Surface, mut named: F) -> Self
	where
		F: FnMut(TextureId) -> String,
	{
		Self {
			name: name.to_owned(),
			albedo: named(live.albedo),
			normal: named(live.normal),
			finish: named(live.finish),
			occlusion: named(live.occlusion),
			glow: named(live.glow),
			surface: Surface {
				albedo: Surface::DEFAULT.albedo,
				normal: Surface::DEFAULT.normal,
				finish: Surface::DEFAULT.finish,
				occlusion: Surface::DEFAULT.occlusion,
				glow: Surface::DEFAULT.glow,
				..*live
			},
		}
	}

	/// This material as the record a file holds.
	///
	/// @param name - where its name went in the file's blob, nought for a
	/// `.cmat`, whose material is named by its path
	/// @param names - the blob its pictures' names go into
	pub(crate) fn coat(&self, name: u32, names: &mut Names) -> Coat {
		let surface = &self.surface;
		let flags = [
			(surface.unlit, UNLIT),
			(surface.occlusion_uv2, OCCLUSION_UV2),
			(surface.glow_uv2, GLOW_UV2),
		]
		.iter()
		.filter(|(set, _)| *set)
		.fold(0, |held, (_, bit)| held | bit);

		Coat {
			name,
			albedo: names.put(&self.albedo),
			normal: names.put(&self.normal),
			wrap: surface.wrap.code(),
			base_color: surface.base_color.to_array(),
			metallic: surface.metallic,
			roughness: surface.roughness,
			blend: surface.blend.code(),
			opacity: surface.opacity,
			uv_scale: surface.uv_scale.to_array(),
			finish: names.put(&self.finish),
			occlusion: names.put(&self.occlusion),
			glow: names.put(&self.glow),
			occlusion_strength: surface.occlusion_strength,
			emissive: surface.emissive.to_array(),
			emissive_strength: surface.emissive_strength,
			uv_offset: surface.uv_offset.to_array(),
			uv_rotation: surface.uv_rotation,
			flags,
		}
	}

	/// The material a file's record describes.
	///
	/// A wrap this build has no name for reads as the ordinary one, because
	/// both of its answers are sensible; a mode or a flag it does not know was
	/// refused before this was asked, by whichever file the record is in.
	///
	/// @param coat - the record
	/// @param name - what it registers under
	/// @param named - the text at an offset into the file's blob
	pub(crate) fn of_coat<'a, F>(coat: &Coat, name: String, named: F) -> Self
	where
		F: Fn(u32) -> &'a str,
	{
		Self {
			name,
			albedo: named(coat.albedo).to_owned(),
			normal: named(coat.normal).to_owned(),
			finish: named(coat.finish).to_owned(),
			occlusion: named(coat.occlusion).to_owned(),
			glow: named(coat.glow).to_owned(),
			surface: Surface {
				base_color: Vec3::from_array(coat.base_color),
				metallic: coat.metallic,
				roughness: coat.roughness,
				occlusion_strength: coat.occlusion_strength,
				occlusion_uv2: coat.flags & OCCLUSION_UV2 != 0,
				glow_uv2: coat.flags & GLOW_UV2 != 0,
				emissive: Vec3::from_array(coat.emissive),
				emissive_strength: coat.emissive_strength,
				uv_scale: Vec2::from_array(coat.uv_scale),
				uv_offset: Vec2::from_array(coat.uv_offset),
				uv_rotation: coat.uv_rotation,
				wrap: Wrap::at(coat.wrap).unwrap_or_default(),
				blend: Blend::from_code(coat.blend).unwrap_or_default(),
				opacity: coat.opacity,
				unlit: coat.flags & UNLIT != 0,
				..Surface::DEFAULT
			},
		}
	}
}

/// Refuses a record naming an alpha mode or a flag this build does not have.
///
/// Asked by both files the record is written into, because both have to refuse
/// the same records: a code has nothing smaller to fall back to, and a flag
/// changes how the surface is drawn.
///
/// @param coat - the record
/// @return nothing, or what was wrong with it
pub(crate) fn unknown_in(coat: &Coat) -> std::result::Result<(), String> {
	if Blend::from_code(coat.blend).is_none() {
		return Err(format!(
			"reads its alpha in mode {}, which this build does not have",
			coat.blend
		));
	}

	if coat.flags & !KNOWN_COAT_FLAGS != 0 {
		return Err(format!(
			"uses feature {:#x}, which this build does not",
			coat.flags & !KNOWN_COAT_FLAGS
		));
	}

	Ok(())
}

/// One piece of a model, and where it stands.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Placement {
	/// What this piece is called.
	pub name: String,

	/// The asset name of the mesh that stands here, or empty for a lamp.
	pub mesh: String,

	/// The material's name, or empty for the default one.
	pub material: String,

	/// The skeleton's asset name, or empty for a piece bones do not move.
	///
	/// A piece that names one is drawn by a pose rather than by its own
	/// transform, and the transform beside this is then the identity - which
	/// is what the exchange format says a skinned node's transform means.
	pub skeleton: String,

	/// Where it stands, with the whole tree above it already worked in.
	pub transform: Transform,

	/// What it shines, or [`Light::NONE`] for a piece that is geometry.
	///
	/// Written as a [`Lit`] record keyed by this placement's place in the
	/// block, and only for a light of a kind that shines.
	pub light: Light,
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

	/// The lamp block, borrowed out of the buffer.
	#[must_use]
	pub fn lit(&self) -> &[Lit] { self.block(self.header.lit_offset, self.header.lit_count) }

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
	///
	/// A lamp record naming a placement the block does not have is dropped,
	/// the rule a scene's reader has for an entity: a model missing one lamp is
	/// a better answer than a load that did not happen.
	#[must_use]
	pub fn to_model_data(&self) -> ModelData {
		let mut placements: Vec<Placement> = self
			.stands()
			.iter()
			.map(|stand| Placement {
				name: self.name(stand.name).to_owned(),
				mesh: self.name(stand.mesh).to_owned(),
				material: self.name(stand.material).to_owned(),
				skeleton: self.name(stand.skeleton).to_owned(),
				transform: Transform {
					position: Vec3::from_array(stand.position),
					rotation: Quat::from_array(stand.rotation),
					scale: Vec3::from_array(stand.scale),
				},
				light: Light::NONE,
			})
			.collect();

		for record in self.lit() {
			if let Some(placement) = usize::try_from(record.thing)
				.ok()
				.and_then(|at| placements.get_mut(at))
			{
				placement.light = light_of(record);
			}
		}

		ModelData {
			guided: self.header.flags & GUIDED != 0,
			materials: self
				.coats()
				.iter()
				.map(|coat| {
					Material::of_coat(coat, self.name(coat.name).to_owned(), |at| self.name(at))
				})
				.collect(),
			placements,
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
		.map(|material| {
			let name = names.put(&material.name);

			material.coat(name, &mut names)
		})
		.collect();
	let stands: Vec<Stand> = data
		.placements
		.iter()
		.map(|placement| Stand {
			name: names.put(&placement.name),
			mesh: names.put(&placement.mesh),
			material: names.put(&placement.material),
			skeleton: names.put(&placement.skeleton),
			position: placement.transform.position.to_array(),
			rotation: placement.transform.rotation.to_array(),
			scale: placement.transform.scale.to_array(),
		})
		.collect();
	let lamps: Vec<Lit> = data
		.placements
		.iter()
		.enumerate()
		.filter(|(_, placement)| placement.light.kind.is_lit())
		.map(|(index, placement)| {
			count(index, "a model's records").map(|at| lit_of(at, placement.light))
		})
		.collect::<Result<_>>()?;

	let coat_offset = HEADER_BYTES;
	let stand_offset = coat_offset + size_of_val(coats.as_slice());
	let lit_offset = stand_offset + size_of_val(stands.as_slice());
	let names_offset = lit_offset + size_of_val(lamps.as_slice());
	let header = ModelHeader {
		magic: MAGIC,
		version: FORMAT_VERSION,
		flags: if data.guided { GUIDED } else { 0 },
		coat_stride: width::<Coat>("a model's records")?,
		stand_stride: width::<Stand>("a model's records")?,
		coat_count: count(coats.len(), "a model's records")?,
		stand_count: count(stands.len(), "a model's records")?,
		coat_offset: count(coat_offset, "a model's records")?,
		stand_offset: count(stand_offset, "a model's records")?,
		names_offset: count(names_offset, "a model's records")?,
		names_length: count(names.blob().len(), "a model's records")?,
		lit_stride: width::<Lit>("a model's records")?,
		lit_count: count(lamps.len(), "a model's records")?,
		lit_offset: count(lit_offset, "a model's records")?,
		reserved: 0,
	};

	let mut out = Vec::with_capacity(names_offset + names.blob().len());
	out.extend_from_slice(bytemuck::bytes_of(&header));
	out.extend_from_slice(bytemuck::cast_slice(&coats));
	out.extend_from_slice(bytemuck::cast_slice(&stands));
	out.extend_from_slice(bytemuck::cast_slice(&lamps));
	out.extend_from_slice(names.blob());

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

/// The flag bits the file at this path sets, if it is one at all.
///
/// The head alone, so a compiled model can be asked how it was built without
/// reading a file that may be about to be rewritten. @ref [`GUIDED`].
///
/// @param path - the `.cmodel` to look at
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

	if header.flags & !KNOWN_FLAGS != 0 {
		return Err(format!(
			"this model uses feature {:#x}, which this build does not",
			header.flags & !KNOWN_FLAGS
		));
	}

	if usize::try_from(header.coat_stride) != Ok(size_of::<Coat>())
		|| usize::try_from(header.stand_stride) != Ok(size_of::<Stand>())
		|| usize::try_from(header.lit_stride) != Ok(size_of::<Lit>())
	{
		return Err("this model's records are not the size this build reads".to_owned());
	}

	if usize::try_from(header.names_length).unwrap_or(usize::MAX) > MAX_NAMES {
		return Err("this model's names are longer than any real one's".to_owned());
	}

	fits::<Coat>(bytes, HEADER_BYTES, (header.coat_offset, header.coat_count), "materials")?;

	// the one field in either record that is a *code*, and the flags beside it,
	// are the only things here that have to be looked at rather than measured:
	// a wrap that is not one is read as the ordinary answer, because it has two
	// values and both are sensible, while a mode this build does not know has
	// nothing smaller to fall back to. @ref `colby-scene-format` for the same
	// line drawn in the other format that has both.
	unknown_coats(bytes, &header)?;
	fits::<Stand>(bytes, HEADER_BYTES, (header.stand_offset, header.stand_count), "placements")?;
	// read by a scene's rules, which look at nothing inside a record: a kind or
	// a flag this build does not know reads as less light, not as a refusal.
	// @ref `crate::scene::light_of`.
	fits::<Lit>(bytes, HEADER_BYTES, (header.lit_offset, header.lit_count), "lamps")?;
	fits::<u8>(bytes, HEADER_BYTES, (header.names_offset, header.names_length), "names")?;

	Ok(header)
}

/// Refuses a file naming an alpha mode or a flag this build does not have.
///
/// @param bytes - the whole file
/// @param header - its already-checked header
/// @return nothing, or which coat named what
fn unknown_coats(bytes: &[u8], header: &ModelHeader) -> std::result::Result<(), String> {
	let Some(range) = span::<Coat>(header.coat_offset, header.coat_count) else {
		return Ok(());
	};
	let coats: &[Coat] = bytes
		.get(range)
		.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
		.unwrap_or(&[]);

	for (index, coat) in coats.iter().enumerate() {
		unknown_in(coat).map_err(|what| format!("material {index} of this model {what}"))?;
	}

	Ok(())
}

#[cfg(test)]
mod tests {
	use core::mem::offset_of;

	use colby_core::abi::LightKind;

	use super::*;

	/// Half of a quarter turn, in the two places a unit quaternion holds it.
	const TURN: f32 = std::f32::consts::FRAC_1_SQRT_2;

	/// A model with everything a record can hold in it.
	fn sample() -> ModelData {
		ModelData {
			// on, so that the round trip below covers the mark as well as the
			// records: a file written without it and read back as `false`
			// would still have compared equal to a fixture holding `false`.
			guided: true,
			materials: vec![brass(), glass()],
			placements: vec![
				Placement {
					name: "shade".to_owned(),
					mesh: "models/lamp/shade".to_owned(),
					material: "models/lamp/brass".to_owned(),
					skeleton: String::new(),
					transform: Transform {
						position: Vec3::new(1.0, 2.0, 3.0),
						rotation: Quat::from_xyzw(0.0, TURN, 0.0, TURN),
						scale: Vec3::new(1.0, 1.0, -1.0),
					},
					light: Light::NONE,
				},
				Placement {
					name: "stem".to_owned(),
					mesh: "models/lamp/stem".to_owned(),
					material: String::new(),
					skeleton: "models/lamp/rig".to_owned(),
					transform: Transform::IDENTITY,
					light: Light::NONE,
				},
				Placement {
					name: "bulb".to_owned(),
					transform: Transform {
						position: Vec3::new(0.0, 2.5, 0.0),
						rotation: Quat::from_xyzw(TURN, 0.0, 0.0, TURN),
						scale: Vec3::ONE,
					},
					light: bulb(),
					..Placement::default()
				},
			],
		}
	}

	/// A lamp with every number off the one a light starts with, so that a
	/// round trip that dropped a field comes back unequal.
	fn bulb() -> Light {
		Light {
			kind: LightKind::Spot,
			color: Vec3::new(1.0, 0.8, 0.6),
			intensity: 2.5,
			range: 7.0,
			inner: 0.2,
			outer: 0.5,
			shadow: false,
		}
	}

	/// A material with every picture named and every number off its default.
	///
	/// Its flags and the glass's are deliberately each other's opposites, and
	/// so are the two alpha modes: a round trip that dropped a field, or wrote
	/// one material's answer to both, would come back equal to a fixture where
	/// the two matched.
	fn brass() -> Material {
		Material {
			name: "models/lamp/brass".to_owned(),
			albedo: "models/lamp/tiles".to_owned(),
			normal: "models/lamp/tiles_normal".to_owned(),
			// one picture for both, which is how an exporter packs them
			finish: "models/lamp/tiles_orm".to_owned(),
			occlusion: "models/lamp/tiles_orm".to_owned(),
			glow: "models/lamp/embers".to_owned(),
			surface: Surface {
				base_color: Vec3::new(0.8, 0.6, 0.2),
				metallic: 1.0,
				roughness: 0.25,
				occlusion_strength: 0.5,
				occlusion_uv2: true,
				emissive: Vec3::new(1.0, 0.5, 0.25),
				emissive_strength: 3.0,
				uv_scale: Vec2::new(4.0, 2.0),
				uv_offset: Vec2::new(0.25, -0.5),
				uv_rotation: 0.75,
				wrap: Wrap::Clamp,
				blend: Blend::Mask,
				opacity: 1.0,
				..Surface::DEFAULT
			},
		}
	}

	/// A second material, sharing a picture with the first.
	fn glass() -> Material {
		Material {
			name: "models/lamp/glass".to_owned(),
			// the same picture, so the blob has one copy of its name
			albedo: "models/lamp/tiles".to_owned(),
			surface: Surface {
				roughness: 0.1,
				glow_uv2: true,
				blend: Blend::Alpha,
				opacity: 0.35,
				unlit: true,
				..Surface::DEFAULT
			},
			..Material::default()
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

		// measured against the fixture rather than against a number somebody
		// calibrated once: what sharing buys is a blob smaller than writing
		// every field's name out, and that stays true as records grow fields.
		let data = sample();
		let apart: usize =
			data.materials
				.iter()
				.map(|coat| {
					[
						&coat.name,
						&coat.albedo,
						&coat.normal,
						&coat.finish,
						&coat.occlusion,
						&coat.glow,
					]
					.iter()
					.map(|name| name.len() + 1)
					.sum::<usize>()
				})
				.chain(data.placements.iter().map(|stand| {
					stand.name.len()
						+ stand.mesh.len() + stand.material.len()
						+ stand.skeleton.len()
						+ 4
				}))
				.sum();

		assert!(
			usize::try_from(file.header().names_length).expect("the blob is small") < apart,
			"the blob holds each name once, so it is under the {apart} bytes writing them all \
			 out would take, and it is {}",
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
		assert_eq!(usize::try_from(file.header().lit_stride), Ok(size_of::<Lit>()));
		assert_eq!(file.coats().len(), 2, "and every block casts in place");
		assert_eq!(file.stands().len(), 3, "a lamp among the placements");
		assert_eq!(file.lit().len(), 1, "and one lamp record, for it alone");
	}

	#[test]
	fn a_lamp_is_a_placement_with_nothing_to_draw_and_a_record_keyed_by_it() {
		let file = round_trip(&sample());

		assert_eq!(file.lit()[0].thing, 2, "keyed by the bulb's place among the placements");
		assert_eq!(file.stands()[2].mesh, 0, "which names no mesh");
		assert_eq!(file.to_model_data().placements[2].light, bulb(), "and it shines what it did");
		assert!(
			file.to_model_data().placements[..2]
				.iter()
				.all(|piece| piece.light == Light::NONE),
			"while the geometry shines nothing"
		);
	}

	#[test]
	fn the_sample_moves_every_number_a_lamp_has() {
		// the round trip's argument again, for the lamp record
		for field in Light::FIELDS {
			assert!(
				field.get(&bulb()) != field.get(&Light::NONE),
				"the bulb leaves {} where a light starts",
				field.name
			);
		}
	}

	#[test]
	fn a_model_with_no_lamps_writes_no_lamp_records() {
		let mut data = sample();

		data.placements.truncate(2);

		let file = round_trip(&data);

		assert_eq!(file.header().lit_count, 0, "nothing shines, so nothing is written down");
		assert_eq!(file.to_model_data(), data, "and nothing comes back shining");
	}

	/// Where the bulb's lamp record starts in the sample's bytes.
	fn bulb_record(bytes: &[u8]) -> usize {
		let header: ModelHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);

		usize::try_from(header.lit_offset).expect("an offset")
	}

	#[test]
	fn a_lamp_record_naming_a_placement_that_is_not_there_is_dropped() {
		let mut bytes = encode(&sample()).expect("it writes");
		let at = bulb_record(&bytes) + offset_of!(Lit, thing);

		bytes[at..at + 4].copy_from_slice(&99_u32.to_le_bytes());

		let read = ModelFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect("a model missing one lamp still reads")
			.to_model_data();

		assert!(
			read.placements
				.iter()
				.all(|piece| piece.light == Light::NONE),
			"and the lamp is the one thing missing"
		);
		assert_eq!(read.placements.len(), 3, "its placement still stands");
	}

	#[test]
	fn a_lamp_of_a_kind_this_build_does_not_know_reads_as_no_light() {
		// the scene's rule, because it is the scene's record
		let mut bytes = encode(&sample()).expect("it writes");
		let at = bulb_record(&bytes) + offset_of!(Lit, kind);

		bytes[at..at + 4].copy_from_slice(&7_u32.to_le_bytes());

		let read = ModelFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect("a kind is not refused")
			.to_model_data();

		assert_eq!(read.placements[2].light.kind, LightKind::None);
	}

	#[test]
	fn a_lamp_block_the_header_mismeasures_is_refused() {
		let stride = offset_of!(ModelHeader, lit_stride);

		assert!(corrupt(stride, 99).contains("not the size this build reads"), "a record width");

		let mut bytes = encode(&sample()).expect("it writes");
		let at = offset_of!(ModelHeader, lit_count);

		bytes[at..at + 4].copy_from_slice(&9999_u32.to_le_bytes());

		let message = ModelFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("it is refused")
			.to_string();

		assert!(message.contains("lamps run from"), "got {message}");
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

		assert_eq!(file.to_model_data().materials[0].surface.wrap, Wrap::Repeat);
	}

	#[test]
	fn a_mode_this_build_does_not_know_is_refused_rather_than_read_as_the_nearest_one() {
		// the field right after `wrap` in the record, which is what the test
		// above pokes. A *code* and a *flag* are treated differently on
		// purpose: the wrap above falls back because both of its answers are
		// sensible, and this one cannot, because a mode nobody here has means
		// a surface nobody here can draw.
		let mut bytes = encode(&sample()).expect("it writes");
		let at = HEADER_BYTES + offset_of!(Coat, blend);

		bytes[at] = 0x7F;

		let refused = ModelFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("a mode of 127 is not one of three");

		assert!(format!("{refused}").contains("127"), "and the message names it, got {refused}");

		// the half that says the check is about the value rather than about the
		// byte having been touched at all.
		bytes[at] = u8::try_from(Blend::Alpha.code()).expect("two fits in a byte");

		assert!(
			ModelFile::from_bytes(AlignedBytes::from_slice(&bytes)).is_ok(),
			"a mode this build does have reads"
		);
	}

	#[test]
	fn a_file_that_is_not_one_of_these_is_refused_by_name() {
		assert!(corrupt(0, b'X').contains("not a colby model"));
		assert!(corrupt(8, 9).contains("version 9"));
		// the lowest bit nothing has claimed, so this still tests what it
		// says the day a second flag lands
		let unknown = !KNOWN_FLAGS & KNOWN_FLAGS.wrapping_add(1);

		assert!(corrupt(12, u8::try_from(unknown).expect("the low byte")).contains("feature"));
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

	#[test]
	fn the_sample_moves_every_number_a_material_has() {
		// the round trip can only see a field the fixture moved off its
		// default: a field the record forgot would come back as the default and
		// compare equal to a fixture that left it there. So every number of the
		// live record is off its default in one of the two, and a field the
		// record grows later fails here until the fixture moves it too.
		for field in Surface::FIELDS
			.iter()
			.filter(|field| !field.kind.is_reference())
		{
			assert!(
				[brass(), glass()]
					.iter()
					.any(|material| field.get(&material.surface) != field.get(&Surface::DEFAULT)),
				"neither material moves {}",
				field.name
			);
		}
	}

	#[test]
	fn a_flag_this_build_does_not_know_is_refused() {
		let mut bytes = encode(&sample()).expect("it writes");
		let at = HEADER_BYTES + offset_of!(Coat, flags);
		// the lowest bit nothing has claimed, so this still tests what it says
		// the day a fourth flag lands
		let unknown = !KNOWN_COAT_FLAGS & KNOWN_COAT_FLAGS.wrapping_add(1);

		bytes[at..at + 4].copy_from_slice(&unknown.to_le_bytes());

		let refused = ModelFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("a flag nobody here has changes how the surface is drawn");

		assert!(format!("{refused}").contains("feature"), "and it says so: {refused}");
	}

	/// The handle a test's registry gives a picture: which of `PICTURES` it is,
	/// past the built-in ones, or nothing.
	fn handle_of(name: &str) -> TextureId {
		PICTURES
			.iter()
			.position(|known| *known == name)
			.and_then(|at| u32::try_from(at).ok())
			.map_or(TextureId::NONE, |at| TextureId::new(at + 10))
	}

	/// And back: a handle's name, or nothing for none and for the flat
	/// stand-in.
	fn name_of(id: TextureId) -> String {
		usize::try_from(id.index())
			.ok()
			.and_then(|at| at.checked_sub(10))
			.and_then(|at| PICTURES.get(at))
			.map_or_else(String::new, |name| (*name).to_owned())
	}

	/// Every picture the two sample materials name.
	const PICTURES: [&str; 4] = [
		"models/lamp/tiles",
		"models/lamp/tiles_normal",
		"models/lamp/tiles_orm",
		"models/lamp/embers",
	];

	#[test]
	fn the_one_conversion_carries_every_picture_and_every_number() {
		let brass = brass();
		let live = brass.live(handle_of);

		assert_eq!(live.albedo, handle_of("models/lamp/tiles"), "the color picture");
		assert_eq!(live.normal, handle_of("models/lamp/tiles_normal"), "the normal map");
		assert_eq!(live.finish, handle_of("models/lamp/tiles_orm"), "the finish");
		assert_eq!(live.occlusion, handle_of("models/lamp/tiles_orm"), "the occlusion");
		assert_eq!(live.glow, handle_of("models/lamp/embers"), "and the glow");

		for field in Surface::FIELDS
			.iter()
			.filter(|field| !field.kind.is_reference())
		{
			assert_eq!(
				field.get(&live),
				field.get(&brass.surface),
				"{} came across as it was",
				field.name
			);
		}

		assert_eq!(
			Material::described(&brass.name, &live, name_of),
			brass,
			"and written down again it is the material it was"
		);
	}

	#[test]
	fn a_normal_map_nobody_named_is_the_flat_one_and_is_written_down_as_nothing() {
		let glass = glass();
		let live = glass.live(handle_of);

		assert_eq!(live.normal, TextureId::FLAT_NORMAL, "the ABI's own rewrite, not a white one");
		assert!(!live.finish.is_some(), "and a picture nobody named is nothing");
		assert_eq!(Material::described(&glass.name, &live, name_of), glass);
	}
}
