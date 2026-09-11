//! colby's runtime material format: `.cmat`, and the `.material` it comes
//! from.
//!
//! **A material is nine numbers and two names**, which is why this is the
//! smallest format in the crate with a record in it at all: a header, one
//! fixed record, and the blob the two names live in.
//!
//! ```text
//!   0  MAGIC                            8 bytes
//!   8  version                          4 bytes
//!  12  flags                            4 bytes
//!  16  the record                      48 bytes
//!  64  the names, NUL-terminated
//! ```
//!
//! **It names no shader**, and that is a decision rather than an omission.
//! Four of the six engines read for this put one in the file - Defold's
//! `vertex_program` and `fragment_program`, Fyrox's `Material { shader, .. }`,
//! s&box's `Layer0 { shader "complex.vfx" .. }`, and an Unreal instance names
//! its parent - and the two that do not are exactly the case colby is in:
//! Godot's `StandardMaterial3D`, a hundred and thirty-four properties and no
//! shader name, and bevy's `StandardMaterial`. This engine has one shader, so
//! writing its name in every file would be writing the same word in every
//! file. The day there are more, the field has two answers and both are
//! additive: Godot's is a second kind of resource and s&box's is a string in
//! the same file.
//!
//! **What a `.cmat` describes is [`Material`](crate::model::Material)**, the
//! record a model has written its own materials down as since models existed.
//! A material described is a material described whoever wrote it, and a second
//! struct of the same nine fields would be a second thing to keep in step.
//!
//! The source is a `.material`, one flat JSON object read through
//! [`Material::FIELDS`](colby_core::abi::Material::FIELDS) the way a `.scene`
//! is read through the tables of the records in it, plus the two texture names
//! by hand - a reference is described by a table and never spelled by one.
//! @ref [`level`](crate::level).

use std::path::Path;

use colby_core::{
	Result,
	abi::{
		Material as Surface,
		material::{Blend, Wrap},
	},
	bytemuck::{self, Pod, Zeroable},
	err,
	glam::{Vec2, Vec3},
};

use crate::{
	bytes::Names,
	json::Value,
	level::{Rows, Writing, check, named_row, names, put_all, read},
	model::Material,
};

/// The eight bytes every `.cmat` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYMAT";

/// The revision of everything in this module.
pub const FORMAT_VERSION: u32 = 1;

/// The extension a compiled material is written with.
pub const EXTENSION: &str = "cmat";

/// The extension the source is written with.
pub const SOURCE_EXTENSION: &str = "material";

/// How big the header is, before the record.
pub const HEADER_BYTES: usize = 16;

/// The largest name blob the reader will accept, in bytes.
///
/// Two asset names. This is how wrong a file has to be before the reader stops
/// rather than reading what it was handed.
pub const MAX_NAMES: usize = 4 << 10;

/// One material, as the file holds it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Coat {
	/// Offset into the blob of the albedo's asset name, or zero for none.
	pub albedo: u32,

	/// The same for the normal map.
	pub normal: u32,

	/// What happens past the edge of both, as [`Wrap`] in declaration order.
	pub wrap: u32,

	/// How the alpha is read, as [`Blend`] in declaration order.
	pub blend: u32,

	/// Zero for a dielectric, one for a metal.
	pub metallic: f32,

	/// Nought is as smooth as a surface is drawn, and one is chalk.
	pub roughness: f32,

	/// How much of the surface there is, where the mode above reads it.
	pub opacity: f32,

	/// How many times the textures repeat across the mesh's own `0..1`.
	pub uv_scale: [f32; 2],

	/// Linear RGB.
	pub base_color: [f32; 3],
}

const _: () = assert!(size_of::<Coat>() == 48, "the record has to stay forty-eight bytes");

/// A `.cmat` read off disk and checked.
#[derive(Clone, Debug)]
pub struct MaterialFile {
	coat: Coat,
	blob: Vec<u8>,
}

impl MaterialFile {
	/// Reads and checks a compiled material.
	///
	/// @param path - the `.cmat` to read
	pub fn open(path: &Path) -> Result<Self> {
		let bytes = std::fs::read(path)?;

		Self::from_bytes(&bytes).map_err(|error| err!(Asset("{}: {error}", path.display())))
	}

	/// Checks bytes that are already in memory.
	///
	/// @param bytes - the whole file
	pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
		let head = bytes
			.get(..HEADER_BYTES)
			.ok_or_else(|| err!(Asset("is too short to hold a {HEADER_BYTES}-byte header")))?;

		if head.get(..MAGIC.len()) != Some(&MAGIC[..]) {
			return Err(err!(Asset("is not a colby material")));
		}

		let version = word(head, MAGIC.len())?;
		if version != FORMAT_VERSION {
			return Err(err!(Asset(
				"is version {version} and this build reads version {FORMAT_VERSION}; a compiled \
				 one is rebuilt from its source"
			)));
		}

		let record = bytes
			.get(HEADER_BYTES..HEADER_BYTES + size_of::<Coat>())
			.ok_or_else(|| err!(Asset("is too short to hold its record")))?;
		let coat: Coat = *bytemuck::try_from_bytes(record)
			.map_err(|reason| err!(Asset("its record could not be read: {reason}")))?;

		let blob = bytes
			.get(HEADER_BYTES + size_of::<Coat>()..)
			.unwrap_or_default();

		if blob.len() > MAX_NAMES {
			return Err(err!(Asset(
				"names {} bytes of texture and no material names that much",
				blob.len()
			)));
		}

		if Wrap::at(coat.wrap).is_none() {
			return Err(err!(Asset("wraps in a way this build has no name for: {}", coat.wrap)));
		}

		if Blend::at(coat.blend).is_none() {
			return Err(err!(Asset(
				"blends in a way this build has no name for: {}",
				coat.blend
			)));
		}

		Ok(Self { coat, blob: blob.to_vec() })
	}

	/// The record, as it was read.
	#[must_use]
	pub const fn coat(&self) -> &Coat { &self.coat }

	/// The material this describes, with both its pictures named.
	///
	/// @param name - what it registers under, which is the asset's own name
	/// and is therefore the caller's
	#[must_use]
	pub fn to_material(&self, name: &str) -> Material {
		Material {
			name: name.to_owned(),
			albedo: self.name(self.coat.albedo).to_owned(),
			normal: self.name(self.coat.normal).to_owned(),
			base_color: Vec3::from_array(self.coat.base_color),
			metallic: self.coat.metallic,
			roughness: self.coat.roughness,
			wrap: Wrap::at(self.coat.wrap).unwrap_or_default(),
			blend: Blend::at(self.coat.blend).unwrap_or_default(),
			opacity: self.coat.opacity,
			uv_scale: Vec2::from_array(self.coat.uv_scale),
		}
	}

	/// One name out of the blob, or nothing for offset zero.
	fn name(&self, at: u32) -> &str {
		let Ok(start) = usize::try_from(at) else {
			return "";
		};
		let rest = self.blob.get(start..).unwrap_or_default();
		let end = rest
			.iter()
			.position(|byte| *byte == 0)
			.unwrap_or(rest.len());

		std::str::from_utf8(rest.get(..end).unwrap_or_default()).unwrap_or("")
	}
}

/// Writes a material out as a `.cmat`.
///
/// @param material - the description to write; its name is not written, being
/// the asset's own
/// @return the whole file, ready to put on disk
#[must_use]
pub fn encode(material: &Material) -> Vec<u8> {
	let mut names = Names::default();
	let coat = Coat {
		albedo: names.put(&material.albedo),
		normal: names.put(&material.normal),
		wrap: material.wrap.index(),
		blend: material.blend.index(),
		metallic: material.metallic,
		roughness: material.roughness,
		opacity: material.opacity,
		uv_scale: material.uv_scale.to_array(),
		base_color: material.base_color.to_array(),
	};

	let mut out = Vec::with_capacity(HEADER_BYTES + size_of::<Coat>() + names.blob().len());
	out.extend_from_slice(&MAGIC);
	out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
	out.extend_from_slice(&0_u32.to_le_bytes());
	out.extend_from_slice(bytemuck::bytes_of(&coat));
	out.extend_from_slice(names.blob());

	out
}

/// Reads a `.material` source.
///
/// @param text - the whole file
/// @return the material, with its two pictures named
pub fn import(text: &str) -> Result<Material> {
	let root = crate::json::parse(text)?;
	let table = names(Surface::FIELDS, &REFERENCES);

	check(&root, &[table], &REFERENCES, "a material")?;

	let mut surface = Surface::DEFAULT;
	read(&mut surface, &root, Surface::FIELDS, "a material")?;

	Ok(described(&surface, text_of(&root, "albedo")?, text_of(&root, "normal")?))
}

/// Writes a `.material` source.
///
/// @param material - what to write
/// @return the text, or why it could not be written
pub fn export(material: &Material) -> Result<String> {
	let mut rows = Rows::default();
	let surface = surface_of(material);

	put_all(
		&mut rows,
		&surface,
		&Surface::DEFAULT,
		Surface::FIELDS,
		&Writing {
			prefix: "",
			what: "a material",
			skipped: &[],
		},
		|field| match field {
			| "albedo" => named_row("albedo", &material.albedo),
			| "normal" => named_row("normal", &material.normal),
			| _ => None,
		},
	)?;

	Ok(rows.text())
}

/// The two fields a table describes and cannot spell.
const REFERENCES: [&str; 2] = ["albedo", "normal"];

/// A described material as the live record, so that one table serves both.
///
/// The two handles are left at what the default holds: nothing here can
/// resolve a name, and the writer names them beside the table.
///
/// @param material - the described form
fn surface_of(material: &Material) -> Surface {
	Surface {
		base_color: material.base_color,
		metallic: material.metallic,
		roughness: material.roughness,
		wrap: material.wrap,
		blend: material.blend,
		opacity: material.opacity,
		uv_scale: material.uv_scale,
		..Surface::DEFAULT
	}
}

/// The other half: a live record and two names as the described form.
///
/// @param surface - everything but the two pictures
/// @param albedo - the color picture's asset name, or empty
/// @param normal - the normal map's, or empty
fn described(surface: &Surface, albedo: String, normal: String) -> Material {
	Material {
		name: String::new(),
		albedo,
		normal,
		base_color: surface.base_color,
		metallic: surface.metallic,
		roughness: surface.roughness,
		wrap: surface.wrap,
		blend: surface.blend,
		opacity: surface.opacity,
		uv_scale: surface.uv_scale,
	}
}

/// One name out of a JSON object, or empty for a key it does not have.
fn text_of(root: &Value, key: &str) -> Result<String> {
	let Some(value) = root.get(key) else {
		return Ok(String::new());
	};

	value
		.as_str()
		.map(str::to_owned)
		.ok_or_else(|| err!(Asset("a material's {key} should be the name of a texture")))
}

/// The version the file at this path claims, if it is one at all.
///
/// The head alone, so the staleness sweep can ask without reading a file it is
/// about to rewrite. @ref `crate::compile::Kind::version_of`.
///
/// @param path - the `.cmat` to look at
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

/// One little-endian word at an offset, or a message.
fn word(head: &[u8], at: usize) -> Result<u32> {
	let four: [u8; 4] = head
		.get(at..at + 4)
		.and_then(|slice| slice.try_into().ok())
		.ok_or_else(|| err!(Asset("is too short to hold its header")))?;

	Ok(u32::from_le_bytes(four))
}

#[cfg(test)]
mod tests {
	use core::mem::offset_of;

	use super::*;

	/// A material with every field off its default and both pictures named.
	fn sample() -> Material {
		Material {
			name: "materials/brass".to_owned(),
			albedo: "textures/brass".to_owned(),
			normal: "textures/brass_normal".to_owned(),
			base_color: Vec3::new(0.8, 0.6, 0.2),
			metallic: 1.0,
			roughness: 0.25,
			wrap: Wrap::Clamp,
			blend: Blend::Mask,
			opacity: 0.75,
			uv_scale: Vec2::new(4.0, 2.0),
		}
	}

	#[test]
	fn everything_written_comes_back() {
		let written = encode(&sample());
		let read = MaterialFile::from_bytes(&written)
			.expect("what this module wrote, this module reads")
			.to_material("materials/brass");

		assert_eq!(read, sample(), "a material through the file is the material");
	}

	#[test]
	fn a_material_naming_no_pictures_names_none() {
		let bare = Material {
			albedo: String::new(),
			normal: String::new(),
			..sample()
		};
		let read = MaterialFile::from_bytes(&encode(&bare))
			.expect("readable")
			.to_material(&bare.name);

		assert_eq!(read.albedo, "", "an empty name is offset zero");
		assert_eq!(read.normal, "", "for both of them");
		assert_eq!(read, bare, "and nothing else moved");
	}

	#[test]
	fn one_picture_used_twice_is_named_once() {
		let twice = Material {
			normal: "textures/brass".to_owned(),
			..sample()
		};
		let once = encode(&twice);
		let apart = encode(&sample());

		assert!(once.len() < apart.len(), "the blob holds one copy of a shared name");
		assert_eq!(
			MaterialFile::from_bytes(&once)
				.expect("readable")
				.to_material(&twice.name),
			twice,
			"and both fields still come back naming it"
		);
	}

	#[test]
	fn a_file_from_another_version_is_refused_with_a_message() {
		let mut bytes = encode(&sample());
		bytes[MAGIC.len()..MAGIC.len() + 4].copy_from_slice(&99_u32.to_le_bytes());

		let reason = MaterialFile::from_bytes(&bytes).expect_err("a version nobody reads");

		assert!(format!("{reason}").contains("99"), "and the message says which: {reason}");
	}

	#[test]
	fn a_word_this_build_has_no_name_for_is_refused_rather_than_read_as_the_first() {
		// the rule a `.cscene` keeps for a body's kind, and for its reason: a
		// code is refused where a flag bit is ignored, because a build that
		// silently read an unknown mode as `opaque` would draw a world of
		// glass as a world of walls.
		for at in [offset_of!(Coat, wrap), offset_of!(Coat, blend)] {
			let mut bytes = encode(&sample());
			let field = HEADER_BYTES + at;

			bytes[field..field + 4].copy_from_slice(&9_u32.to_le_bytes());

			assert!(
				MaterialFile::from_bytes(&bytes).is_err(),
				"a code nothing answers to is refused"
			);
		}
	}

	#[test]
	fn a_source_reads_and_writes_back_the_same_material() {
		let text = export(&sample()).expect("it can be written");
		let read = import(&text).unwrap_or_else(|failure| {
			panic!("what was written did not read back: {failure}\n{text}")
		});

		assert_eq!(
			Material { name: sample().name, ..read },
			sample(),
			"the text is the material: {text}"
		);

		// and every one of those keys is the field's own name
		for field in Surface::FIELDS {
			assert!(
				text.contains(&format!("\"{}\"", field.name)),
				"{} is in the text",
				field.name
			);
		}
	}

	#[test]
	fn a_source_that_says_nothing_is_the_default_material() {
		let read = import("{}").expect("an empty object is a material");

		assert_eq!(read, described(&Surface::DEFAULT, String::new(), String::new()));
	}

	#[test]
	fn a_field_nobody_declared_is_refused_by_name() {
		let refused = import(r#"{ "shininess": 3 }"#).expect_err("a material has no shininess");

		assert!(
			format!("{refused}").contains("shininess"),
			"and the message says which word it was: {refused}"
		);
	}

	#[test]
	fn a_word_the_source_does_not_know_is_refused_by_name() {
		let refused = import(r#"{ "blend": "frosted" }"#).expect_err("no such mode");

		assert!(
			format!("{refused}").contains("frosted"),
			"and the message says which: {refused}"
		);
	}

	#[test]
	fn a_picture_that_is_not_a_name_is_refused() {
		assert!(import(r#"{ "albedo": 7 }"#).is_err(), "a texture is named, not numbered");
		assert_eq!(
			import(r#"{ "albedo": "textures/brass" }"#)
				.expect("a name is a name")
				.albedo,
			"textures/brass"
		);
	}
}
