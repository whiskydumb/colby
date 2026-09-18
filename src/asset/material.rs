//! colby's runtime material format: `.cmat`, and the `.material` it comes
//! from.
//!
//! **A material is one record and the names of its five pictures**, which is
//! why this is the smallest format in the crate with a record in it at all: a
//! header, one fixed record, and the blob the names live in.
//!
//! ```text
//!   0  MAGIC                            8 bytes
//!   8  version                          4 bytes
//!  12  flags                            4 bytes
//!  16  the record                     100 bytes
//! 116  the names, NUL-terminated
//! ```
//!
//! **The record is a model's**, [`Coat`], with its own name at nought: a file
//! of its own is named by its path. One material written down is one record
//! whichever file it is in, so a field a material grows is one field in one
//! place and not two records kept in step by hand, which is what the two were
//! until the rest of the exchange format's material arrived.
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
//! form a model has written its own materials down as since models existed.
//! A material described is a material described whoever wrote it.
//!
//! The source is a `.material`, one flat JSON object read through
//! [`Material::FIELDS`](colby_core::abi::Material::FIELDS) the way a `.scene`
//! is read through the tables of the records in it, plus the five picture names
//! by hand - a reference is described by a table and never spelled by one.
//! @ref [`level`](crate::level).

use std::path::Path;

use colby_core::{
	Result,
	abi::{Material as Surface, material::Wrap},
	bytemuck, err,
};

use crate::{
	bytes::Names,
	json::Value,
	level::{Rows, Writing, check, named_row, names, put_all, read},
	model::{Coat, Material, unknown_in},
};

/// The eight bytes every `.cmat` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYMAT";

/// The revision of everything in this module.
pub const FORMAT_VERSION: u32 = 2;

/// The extension a compiled material is written with.
pub const EXTENSION: &str = "cmat";

/// The extension the source is written with.
pub const SOURCE_EXTENSION: &str = "material";

/// How big the header is, before the record.
pub const HEADER_BYTES: usize = 16;

/// The largest name blob the reader will accept, in bytes.
///
/// Five asset names. This is how wrong a file has to be before the reader
/// stops rather than reading what it was handed.
pub const MAX_NAMES: usize = 4 << 10;

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

		// stricter than a model about the wrap, and on purpose: a file somebody
		// wrote by hand is refused where it is wrong, rather than read as the
		// nearest thing it could have meant. @ref `colby-scene-format`.
		if Wrap::at(coat.wrap).is_none() {
			return Err(err!(Asset("wraps in a way this build has no name for: {}", coat.wrap)));
		}

		unknown_in(&coat).map_err(|what| err!(Asset("{what}")))?;

		Ok(Self { coat, blob: blob.to_vec() })
	}

	/// The record, as it was read.
	#[must_use]
	pub const fn coat(&self) -> &Coat { &self.coat }

	/// The material this describes, with its pictures named.
	///
	/// @param name - what it registers under, which is the asset's own name
	/// and is therefore the caller's
	#[must_use]
	pub fn to_material(&self, name: &str) -> Material {
		Material::of_coat(&self.coat, name.to_owned(), |at| self.name(at))
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
	let coat = material.coat(0, &mut names);

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
/// @return the material, with its pictures named
pub fn import(text: &str) -> Result<Material> {
	let root = crate::json::parse(text)?;
	let table = names(Surface::FIELDS, &REFERENCES);

	check(&root, &[table], &REFERENCES, "a material")?;

	let mut surface = Surface::DEFAULT;
	read(&mut surface, &root, Surface::FIELDS, "a material")?;

	Ok(Material {
		name: String::new(),
		albedo: text_of(&root, "albedo")?,
		normal: text_of(&root, "normal")?,
		finish: text_of(&root, "finish")?,
		occlusion: text_of(&root, "occlusion")?,
		glow: text_of(&root, "glow")?,
		surface,
	})
}

/// Writes a `.material` source.
///
/// @param material - what to write
/// @return the text, or why it could not be written
pub fn export(material: &Material) -> Result<String> {
	let mut rows = Rows::default();

	put_all(
		&mut rows,
		&material.surface,
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
			| "finish" => named_row("finish", &material.finish),
			| "occlusion" => named_row("occlusion", &material.occlusion),
			| "glow" => named_row("glow", &material.glow),
			| _ => None,
		},
	)?;

	Ok(rows.text())
}

/// The five fields a table describes and cannot spell.
const REFERENCES: [&str; 5] = ["albedo", "normal", "finish", "occlusion", "glow"];

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

	use colby_core::{
		abi::material::Blend,
		glam::{Vec2, Vec3},
	};

	use super::*;

	/// A material with every field off its default and every picture named.
	fn sample() -> Material {
		Material {
			name: "materials/brass".to_owned(),
			albedo: "textures/brass".to_owned(),
			normal: "textures/brass_normal".to_owned(),
			finish: "textures/brass_orm".to_owned(),
			occlusion: "textures/brass_ao".to_owned(),
			glow: "textures/brass_embers".to_owned(),
			surface: Surface {
				base_color: Vec3::new(0.8, 0.6, 0.2),
				metallic: 1.0,
				roughness: 0.25,
				occlusion_strength: 0.5,
				occlusion_uv2: true,
				glow_uv2: true,
				emissive: Vec3::new(1.0, 0.5, 0.25),
				emissive_strength: 4.0,
				uv_scale: Vec2::new(4.0, 2.0),
				uv_offset: Vec2::new(0.25, 0.5),
				uv_rotation: 1.5,
				wrap: Wrap::Clamp,
				blend: Blend::Mask,
				opacity: 0.75,
				unlit: true,
				..Surface::DEFAULT
			},
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
	fn the_record_is_a_models_with_no_name_of_its_own() {
		let written = encode(&sample());
		let file = MaterialFile::from_bytes(&written).expect("readable");

		assert_eq!(file.coat().name, 0, "a file of its own is named by its path");
		assert_eq!(
			written.len() - HEADER_BYTES - file.blob.len(),
			size_of::<Coat>(),
			"and the record between the header and the names is exactly one of a model's"
		);
	}

	#[test]
	fn a_material_naming_no_pictures_names_none() {
		let bare = Material {
			surface: sample().surface,
			name: sample().name,
			..Material::default()
		};
		let read = MaterialFile::from_bytes(&encode(&bare))
			.expect("readable")
			.to_material(&bare.name);

		for (named, what) in [
			(&read.albedo, "albedo"),
			(&read.normal, "normal"),
			(&read.finish, "finish"),
			(&read.occlusion, "occlusion"),
			(&read.glow, "glow"),
		] {
			assert_eq!(named, "", "an empty name is offset zero, for the {what} as well");
		}

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
	fn a_flag_this_build_does_not_know_is_refused_as_a_model_refuses_it() {
		let mut bytes = encode(&sample());
		let field = HEADER_BYTES + offset_of!(Coat, flags);

		bytes[field..field + 4].copy_from_slice(&(1_u32 << 20).to_le_bytes());

		let refused = MaterialFile::from_bytes(&bytes).expect_err("a flag nobody here has");

		assert!(format!("{refused}").contains("feature"), "and it says so: {refused}");
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

		assert_eq!(read, Material::default());
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
		for key in REFERENCES {
			assert!(
				import(&format!("{{ \"{key}\": 7 }}")).is_err(),
				"a texture is named, not numbered, the {key} as well"
			);
		}

		assert_eq!(
			import(r#"{ "albedo": "textures/brass" }"#)
				.expect("a name is a name")
				.albedo,
			"textures/brass"
		);
	}
}
