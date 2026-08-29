//! Reading a glTF's materials, and the pictures they name.
//!
//! glTF's material model is the metallic-roughness pair, which is the one
//! `abi::material` was built around, so the numbers cross with no translation
//! at all: a base color, how metallic, how rough, an albedo, a normal map. That
//! is the whole of what arrives. Everything else a material may declare is
//! named in a warning and dropped, because this renderer has nowhere to put it.
//!
//! **A picture reaches colby one of two ways, and which one decides who
//! compiles it.** An image written as a file beside the document is already an
//! asset: the compiler finds it on its own walk and turns it into a `.ctex`,
//! and a material only has to say which file. An image stored *inside* the
//! model has no such file, so it is decoded here and handed back to be written
//! out beside the model's meshes.
//!
//! **That split is what decides the texel layout, and it is why the naming rule
//! survives.** What a picture's channels mean is not in a PNG, so a loose one
//! is judged by its name - the `_normal` suffix, the only such rule in the
//! project. A material *says*, so an extracted picture needs no rule at all and
//! is written with the layout its use asks for. What is left is the seam
//! between the two, and it gets a warning: a file used as a normal map whose
//! name does not say so is about to be compiled as a color, and the other way
//! round.
//!
//! **One difference is deliberately not warned about: `doubleSided`.** colby
//! culls back faces and always will, so a material that asks for both sides
//! loses one - but the flag is on by default in the tool most models come
//! from, so warning about it would put a line in the log for every material
//! of every model and say nothing about any of them. A warning that always
//! fires teaches people to ignore the ones that do not.
//!
//! **The same picture used both ways comes out twice.** Once as a color and
//! once as numbers, under two names, because the two are different files by the
//! time the GPU sees them.

use std::path::PathBuf;

use colby_core::{
	abi::{
		material::Wrap,
		texture::{Texel, TextureData},
	},
	glam::Vec3,
};

use super::Gltf;
use crate::{compile, jpeg, json::Value, png};

/// What a sampler means by each of its two wrap modes.
const REPEAT: u32 = 10497;

/// The one other mode colby has.
const CLAMP: u32 = 33071;

/// The mode it does not have.
const MIRROR: u32 = 33648;

/// What a picture written as a PNG says it is.
const PNG: &str = "image/png";

/// And the other one the specification allows.
const JPEG: &str = "image/jpeg";

/// Every material a file declares, and what came out with them.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Coats {
	/// One per material, in the file's own order.
	pub surfaces: Vec<Surface>,

	/// Pictures that were inside the file and now need writing out.
	pub pictures: Vec<Extracted>,

	/// What the materials said that could not be used.
	pub warnings: Vec<String>,
}

/// One of the file's materials, as colby's own numbers.
///
/// Not an `abi::Material`, because that names its textures by handle and no
/// handle exists until the host has registered them. This is the same thing
/// with names in the two places a handle will go.
#[derive(Clone, Debug, PartialEq)]
pub struct Surface {
	/// What it registers under, inside the model's own name.
	pub name: String,

	/// Linear RGB, as the file wrote it.
	pub base_color: Vec3,

	/// Zero for a dielectric, one for a metal.
	pub metallic: f32,

	/// Zero is a mirror, one is chalk.
	pub roughness: f32,

	/// The color picture, if it has one.
	pub albedo: Option<Picture>,

	/// The normal map, if it has one.
	pub normal: Option<Picture>,

	/// What happens past the edge of both.
	pub wrap: Wrap,
}

impl Default for Surface {
	fn default() -> Self {
		Self {
			name: String::new(),
			base_color: Vec3::ONE,
			metallic: 1.0,
			roughness: 1.0,
			albedo: None,
			normal: None,
			wrap: Wrap::Repeat,
		}
	}
}

/// Where one of a material's pictures comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Picture {
	/// A file in the asset tree, which the compiler already turns into a
	/// texture on its own walk. The material only has to name it.
	Beside(PathBuf),

	/// One that was inside the model, by its index in [`Coats::pictures`].
	Inside(usize),
}

/// A picture that was inside the file and has to become a texture of its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extracted {
	/// What it registers under, inside the model's own name.
	pub name: String,

	/// The decoded image with its mip chain, ready to be written.
	pub data: TextureData,
}

/// Reads every material a file declares.
///
/// @param file - the document with its buffers
/// @return the materials, the pictures taken out of the file, and what was
/// dropped on the way
#[must_use]
pub(super) fn read(file: &Gltf) -> Coats {
	let mut coats = Coats::default();
	let mut taken = Vec::new();
	let mut pulled = Vec::new();

	for index in 0..file.table("materials").len() {
		let mut reading = Reading {
			file,
			index,
			coats: &mut coats,
			taken: &mut taken,
			pulled: &mut pulled,
		};
		let surface = reading.surface();

		coats.surfaces.push(surface);
	}

	coats
}

/// One material being read.
struct Reading<'a> {
	file: &'a Gltf,
	index: usize,
	coats: &'a mut Coats,
	taken: &'a mut Vec<String>,
	/// Which extracted picture each image and layout became, so a picture used
	/// twice the same way is decoded once.
	pulled: &'a mut Vec<(usize, Texel, usize)>,
}

impl Reading<'_> {
	/// The whole of one material.
	fn surface(&mut self) -> Surface {
		let entry = self
			.file
			.table("materials")
			.get(self.index)
			.cloned()
			.unwrap_or_default();
		let pbr = entry.get("pbrMetallicRoughness").cloned();
		let pbr = pbr.unwrap_or_default();

		self.complain(&entry, &pbr);

		let albedo = self.picture(pbr.get("baseColorTexture"), Texel::Rgba8Srgb);
		let normal = self.picture(entry.get("normalTexture"), Texel::Rgba8Unorm);
		let wrap = self.wrap(pbr.get("baseColorTexture"));

		Surface {
			name: self.name(&entry),
			base_color: color(pbr.get("baseColorFactor")),
			metallic: number(&pbr, "metallicFactor"),
			roughness: number(&pbr, "roughnessFactor"),
			albedo,
			normal,
			wrap,
		}
	}

	/// The name a material registers under.
	fn name(&mut self, entry: &Value) -> String {
		let written = entry
			.get("name")
			.and_then(Value::as_str)
			.unwrap_or("");
		let mut base = super::tidy(written);

		if base.is_empty() {
			base = format!("material{}", self.index);
		}

		super::unique(self.taken, &base)
	}

	/// One of a material's pictures, however it is stored.
	fn picture(&mut self, reference: Option<&Value>, texel: Texel) -> Option<Picture> {
		let reference = reference?;

		if reference
			.get("texCoord")
			.and_then(Value::as_usize)
			> Some(0)
		{
			self.note("names a second set of texture coordinates, and colby has one");
		}

		let texture = reference.get("index").and_then(Value::as_usize)?;
		let image = self
			.file
			.table("textures")
			.get(texture)
			.and_then(|entry| entry.get("source"))
			.and_then(Value::as_usize)?;
		let entry = self.file.table("images").get(image)?.clone();

		match entry.get("uri").and_then(Value::as_str) {
			| Some(uri) if !uri.starts_with(super::DATA_PREFIX) => self.file_beside(uri, texel),
			| _ => self.extract(image, &entry, texel),
		}
	}

	/// A picture that is a file of its own, which the compiler already knows
	/// how to turn into a texture.
	fn file_beside(&mut self, uri: &str, texel: Texel) -> Option<Picture> {
		let path = self.file.beside(uri).or_else(|| {
			self.note("names a picture outside the asset tree, and it is left out");

			None
		})?;

		// the loose file will be compiled by the naming rule, which cannot see
		// what this material says about it. When the two disagree the picture
		// ends up bent one way or lit from the wrong side, and this is the only
		// moment anybody can be told.
		if compile::texel_of(&path) != texel {
			let wanted = if texel.is_color() {
				"a color"
			} else {
				"a set of directions"
			};

			self.note(&format!(
				"uses {} as {wanted}, and its name says otherwise; the {} suffix is what \
				 decides a loose picture",
				path.display(),
				compile::NORMAL_SUFFIX
			));
		}

		Some(Picture::Beside(path))
	}

	/// A picture stored inside the model, decoded and kept to be written out.
	fn extract(&mut self, image: usize, entry: &Value, texel: Texel) -> Option<Picture> {
		if let Some((.., already)) = self
			.pulled
			.iter()
			.find(|(which, layout, _)| *which == image && *layout == texel)
		{
			return Some(Picture::Inside(*already));
		}

		let kind = entry
			.get("mimeType")
			.and_then(Value::as_str)
			.unwrap_or(PNG);

		if kind != PNG && kind != JPEG {
			self.note(&format!("holds a {kind} picture, which colby does not decode"));

			return None;
		}

		let bytes = self.bytes(entry)?;
		let read = if kind == JPEG {
			jpeg::import(&bytes, texel)
		} else {
			png::import(&bytes, texel)
		};
		let data = match read {
			| Ok(data) => data,
			| Err(error) => {
				self.note(&format!("holds a picture that will not decode: {error}"));

				return None;
			},
		};

		let index = self.coats.pictures.len();
		let name = self.picture_name(image, entry, texel);

		self.coats.pictures.push(Extracted { name, data });
		self.pulled.push((image, texel, index));

		Some(Picture::Inside(index))
	}

	/// The bytes of a picture that is inside the file.
	fn bytes(&mut self, entry: &Value) -> Option<Vec<u8>> {
		if let Some(view) = entry.get("bufferView").and_then(Value::as_usize) {
			return match self.file.view(view) {
				| Ok(bytes) => Some(bytes.to_vec()),
				| Err(error) => {
					self.note(&format!("names a picture that cannot be reached: {error}"));

					None
				},
			};
		}

		let uri = entry.get("uri").and_then(Value::as_str)?;
		let bytes = super::inline(uri);

		if bytes.is_none() {
			self.note("holds a picture written as an address colby cannot read");
		}

		bytes
	}

	/// The name an extracted picture registers under.
	fn picture_name(&mut self, image: usize, entry: &Value, texel: Texel) -> String {
		let written = entry
			.get("name")
			.and_then(Value::as_str)
			.unwrap_or("");
		let mut base = super::tidy(written);

		if base.is_empty() {
			base = format!("picture{image}");
		}

		// the same suffix a loose file would carry, so that two layouts of one
		// picture are two names and anybody reading the output tree can tell
		// which is which.
		if !texel.is_color() && !base.ends_with(compile::NORMAL_SUFFIX) {
			base = format!("{base}{}", compile::NORMAL_SUFFIX);
		}

		super::unique(self.taken, &base)
	}

	/// What a material's sampler does past the edge of its pictures.
	fn wrap(&mut self, reference: Option<&Value>) -> Wrap {
		let Some(sampler) = reference
			.and_then(|texture| texture.get("index"))
			.and_then(Value::as_usize)
			.and_then(|texture| self.file.table("textures").get(texture))
			.and_then(|texture| texture.get("sampler"))
			.and_then(Value::as_usize)
			.and_then(|index| self.file.table("samplers").get(index))
		else {
			return Wrap::Repeat;
		};

		let across = mode(sampler, "wrapS");
		let down = mode(sampler, "wrapT");

		if across != down {
			self.note(
				"wraps one way across and another down, and colby has one setting for both",
			);
		}

		if across == MIRROR || down == MIRROR {
			self.note("asks for a mirrored wrap, which colby does not have; it repeats instead");
		}

		if across == CLAMP { Wrap::Clamp } else { Wrap::Repeat }
	}

	/// Names everything the material declares that this renderer has nowhere to
	/// put.
	///
	/// One place for all of them so the list is readable as a list, which is
	/// also what it is: the gap between glTF's material and colby's.
	fn complain(&mut self, entry: &Value, pbr: &Value) {
		let missing = [
			(
				pbr.get("metallicRoughnessTexture").is_some(),
				"a metallic and roughness picture",
			),
			(entry.get("emissiveTexture").is_some(), "an emissive picture"),
			(entry.get("occlusionTexture").is_some(), "an occlusion picture"),
			(lit(entry.get("emissiveFactor")), "an emissive color"),
		];

		for (present, what) in missing {
			if present {
				self.note(&format!("has {what}, and colby has no slot for one"));
			}
		}

		if entry
			.get("alphaMode")
			.and_then(Value::as_str)
			.is_some_and(|mode| mode != "OPAQUE")
		{
			self.note("is not opaque, and every material colby draws is");
		}

		if entry.get("normalTexture").is_some_and(|texture| {
			texture
				.get("scale")
				.and_then(Value::as_f32)
				.is_some_and(|scale| (scale - 1.0).abs() > 1e-6)
		}) {
			self.note("scales its normal map, and colby applies one as it was authored");
		}
	}

	/// One line about this material.
	fn note(&mut self, what: &str) {
		self.coats
			.warnings
			.push(format!("material {} {what}", self.index));
	}
}

/// A wrap mode, or the one a sampler that says nothing means.
fn mode(sampler: &Value, name: &str) -> u32 {
	sampler
		.get(name)
		.and_then(Value::as_u32)
		.unwrap_or(REPEAT)
}

/// A factor that is one when the file leaves it out, which is what glTF says.
fn number(pbr: &Value, name: &str) -> f32 {
	pbr.get(name)
		.and_then(Value::as_f32)
		.unwrap_or(1.0)
}

/// The first three of four numbers, or white.
fn color(written: Option<&Value>) -> Vec3 {
	let Some(cells) = written.map(Value::as_array) else {
		return Vec3::ONE;
	};

	if cells.len() < 3 {
		return Vec3::ONE;
	}

	Vec3::new(
		cells[0].as_f32().unwrap_or(1.0),
		cells[1].as_f32().unwrap_or(1.0),
		cells[2].as_f32().unwrap_or(1.0),
	)
}

/// Whether an emissive color is anything but black.
fn lit(written: Option<&Value>) -> bool {
	written.map(Value::as_array).is_some_and(|cells| {
		cells
			.iter()
			.any(|cell| cell.as_f32().unwrap_or(0.0) > 0.0)
	})
}

#[cfg(test)]
mod tests {
	use std::{fs, path::Path};

	use super::*;
	use crate::gltf::import;

	/// A thirty-two square checker, written by a tool that is not this one.
	const PICTURE: &str = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAARElEQVR42mOoKM77jw+f2LceL6ZUP8OoA0YdMOqAAXcArS0gpH/UAaMOGHXAwDtgtCQcdcCoA0YdMFoSjjpg1AEj3gEAp+wYptPc9nMAAAAASUVORK5CYII=";

	/// The same checker written the other way the specification allows.
	const PHOTO: &str = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAAgACADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDn6+yKK+N68/8Aj+Vjs+AK+yKK+N6P4/lYPgCvsiivjej+P5WD4Ar7Ior43o/j+Vg+A//Z";

	/// How many bytes that is once it is decoded.
	const PHOTO_BYTES: usize = 675;

	/// The scene an exporter wrote, with its pictures inside it.
	fn packed() -> Coats {
		let file = Gltf::read(
			include_bytes!("fixtures/model.glb"),
			Path::new("model.glb"),
			Path::new(""),
		)
		.expect("the fixture reads");

		read(&file)
	}

	/// A document with one picture in it, and whatever a test asks for beside.
	fn document(body: &str) -> Coats {
		let text = format!(
			"{{ \"asset\": {{ \"version\": \"2.0\" }}, \"buffers\": [ {{ \"byteLength\": 125, \
			 \"uri\": \"data:application/octet-stream;base64,{PICTURE}\" }} ], \"bufferViews\": \
			 [ {{ \"buffer\": 0, \"byteLength\": 125 }} ], \"images\": [ {{ \"name\": \
			 \"picture\", \"bufferView\": 0, \"mimeType\": \"image/png\" }} ], {body} }}"
		);
		let file = Gltf::read(text.as_bytes(), Path::new("model.gltf"), Path::new(""))
			.expect("the document reads");

		read(&file)
	}

	/// Whether any warning says a thing.
	fn complained(coats: &Coats, about: &str) -> bool {
		coats
			.warnings
			.iter()
			.any(|line| line.contains(about))
	}

	#[test]
	fn the_numbers_of_a_material_cross_with_no_translation() {
		let coats = packed();
		let brass = &coats.surfaces[0];

		assert_eq!(coats.surfaces.len(), 2);
		assert_eq!(brass.name, "brass");
		assert!(
			brass
				.base_color
				.abs_diff_eq(Vec3::new(0.8, 0.6, 0.2), 1e-6),
			"the color it was given, in the space it was written in: {}",
			brass.base_color
		);
		assert!((brass.metallic - 0.0).abs() < 1e-6);
		assert!((brass.roughness - 0.5).abs() < 1e-6);
		assert_eq!(brass.albedo, None, "it wears no picture at all");
	}

	#[test]
	fn a_picture_stored_inside_the_model_is_decoded_and_kept() {
		let coats = packed();
		let stone = &coats.surfaces[1];

		assert_eq!(stone.name, "stone");
		assert_eq!(coats.pictures.len(), 2, "a color and a normal map");

		let names: Vec<&str> = coats
			.pictures
			.iter()
			.map(|picture| picture.name.as_str())
			.collect();

		assert_eq!(names, vec!["tiles", "tiles_normal"], "in the order they were reached for");
		assert_eq!(stone.albedo, Some(Picture::Inside(0)));
		assert_eq!(stone.normal, Some(Picture::Inside(1)));
	}

	#[test]
	fn what_a_picture_is_used_for_decides_how_its_channels_are_read() {
		// the whole reason a material can say what a file name has to guess at.
		let coats = packed();
		let color = &coats.pictures[0];
		let bump = &coats.pictures[1];

		assert_eq!(bump.data.texel, Texel::Rgba8Unorm, "a normal map is numbers");
		assert_eq!(color.data.texel, Texel::Rgba8Srgb, "and a color is a color");
		assert_eq!((color.data.width, color.data.height), (32, 32));
		assert_eq!(color.data.levels.len(), 6, "and it arrives with its whole chain");
	}

	#[test]
	fn a_picture_written_as_a_file_is_left_for_the_compiler_to_find() {
		// the same scene exported the other way. Nothing is decoded here,
		// because the loose file is an asset already and is compiled on the
		// walk that finds it.
		let dir = std::env::temp_dir()
			.join("colby-gltf-tests")
			.join("pictures");

		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(dir.join("models")).expect("the fixture is made");

		for (name, bytes) in [
			("model.gltf", include_bytes!("fixtures/model.gltf").as_slice()),
			("model.bin", include_bytes!("fixtures/model.bin").as_slice()),
			("tiles.png", include_bytes!("fixtures/tiles.png").as_slice()),
			("tiles_normal.png", include_bytes!("fixtures/tiles_normal.png").as_slice()),
		] {
			fs::write(dir.join("models").join(name), bytes).expect("the fixture is written");
		}

		let file =
			Gltf::open(&dir.join("models").join("model.gltf"), &dir).expect("the document reads");
		let coats = read(&file);

		assert!(coats.pictures.is_empty(), "nothing had to be taken out of it");
		assert_eq!(
			coats.surfaces[1].albedo,
			Some(Picture::Beside(dir.join("models").join("tiles.png")))
		);
		assert_eq!(coats.warnings, Vec::<String>::new(), "and the two names agree");

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn a_loose_picture_whose_name_disagrees_with_its_use_is_named_in_a_warning() {
		let dir = std::env::temp_dir()
			.join("colby-gltf-tests")
			.join("disagree");

		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(dir.join("models")).expect("the fixture is made");
		fs::write(dir.join("models").join("bumps.png"), include_bytes!("fixtures/tiles.png"))
			.expect("the picture is written");

		let text = "{ \"asset\": { \"version\": \"2.0\" }, \"images\": [ { \"uri\": \
		            \"bumps.png\" } ], \"textures\": [ { \"source\": 0 } ], \"materials\": [ { \
		            \"normalTexture\": { \"index\": 0 } } ] }";

		fs::write(dir.join("models").join("model.gltf"), text).expect("the document is written");

		let file = Gltf::open(&dir.join("models").join("model.gltf"), &dir).expect("it reads");
		let coats = read(&file);

		assert!(complained(&coats, "_normal"), "got {:?}", coats.warnings);
		assert!(complained(&coats, "bumps.png"), "and names the file");

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn one_picture_used_both_ways_comes_out_twice() {
		let coats = document(
			"\"textures\": [ { \"source\": 0 } ], \"materials\": [ { \"pbrMetallicRoughness\": \
			 { \"baseColorTexture\": { \"index\": 0 } } }, { \"normalTexture\": { \"index\": 0 \
			 } } ]",
		);
		let names: Vec<&str> = coats
			.pictures
			.iter()
			.map(|picture| picture.name.as_str())
			.collect();

		assert_eq!(names, vec!["picture", "picture_normal"]);
		assert_eq!(coats.pictures[0].data.texel, Texel::Rgba8Srgb);
		assert_eq!(coats.pictures[1].data.texel, Texel::Rgba8Unorm);
	}

	#[test]
	fn the_same_picture_used_the_same_way_twice_is_decoded_once() {
		let coats = document(
			"\"textures\": [ { \"source\": 0 } ], \"materials\": [ { \"pbrMetallicRoughness\": \
			 { \"baseColorTexture\": { \"index\": 0 } } }, { \"pbrMetallicRoughness\": { \
			 \"baseColorTexture\": { \"index\": 0 } } } ]",
		);

		assert_eq!(coats.pictures.len(), 1);
		assert_eq!(coats.surfaces[0].albedo, coats.surfaces[1].albedo);
	}

	#[test]
	fn a_material_that_says_nothing_is_the_one_the_specification_describes() {
		let coats = document("\"materials\": [ {} ]");
		let only = &coats.surfaces[0];

		assert!(only.base_color.abs_diff_eq(Vec3::ONE, 1e-6), "white");
		assert!((only.metallic - 1.0).abs() < 1e-6, "and metal, which is glTF's default");
		assert!((only.roughness - 1.0).abs() < 1e-6, "and rough");
		assert_eq!(only.name, "material0", "and numbered, having no name");
	}

	#[test]
	fn a_sampler_says_what_happens_past_the_edge() {
		let clamped = document(
			"\"samplers\": [ { \"wrapS\": 33071, \"wrapT\": 33071 } ], \"textures\": [ { \
			 \"source\": 0, \"sampler\": 0 } ], \"materials\": [ { \"pbrMetallicRoughness\": { \
			 \"baseColorTexture\": { \"index\": 0 } } } ]",
		);

		assert_eq!(clamped.surfaces[0].wrap, Wrap::Clamp);

		let mirrored = document(
			"\"samplers\": [ { \"wrapS\": 33648, \"wrapT\": 33648 } ], \"textures\": [ { \
			 \"source\": 0, \"sampler\": 0 } ], \"materials\": [ { \"pbrMetallicRoughness\": { \
			 \"baseColorTexture\": { \"index\": 0 } } } ]",
		);

		assert_eq!(mirrored.surfaces[0].wrap, Wrap::Repeat, "the nearest thing colby has");
		assert!(complained(&mirrored, "mirrored"), "got {:?}", mirrored.warnings);
	}

	#[test]
	fn everything_this_renderer_has_no_slot_for_is_named_rather_than_dropped_in_silence() {
		let coats = document(
			"\"textures\": [ { \"source\": 0 } ], \"materials\": [ { \"emissiveFactor\": [ 1, \
			 0, 0 ], \"occlusionTexture\": { \"index\": 0 }, \"alphaMode\": \"BLEND\", \
			 \"normalTexture\": { \"index\": 0, \"scale\": 2 }, \"pbrMetallicRoughness\": { \
			 \"metallicRoughnessTexture\": { \"index\": 0 }, \"baseColorTexture\": { \"index\": \
			 0, \"texCoord\": 1 } } } ]",
		);

		for about in [
			"a metallic and roughness picture",
			"an occlusion picture",
			"an emissive color",
			"is not opaque",
			"scales its normal map",
			"a second set of texture coordinates",
		] {
			assert!(complained(&coats, about), "nothing said {about}: {:?}", coats.warnings);
		}
	}

	#[test]
	fn a_picture_of_a_kind_colby_cannot_decode_is_named_and_left_out() {
		let coats = document(
			"\"textures\": [ { \"source\": 1 } ], \"images\": [ { \"bufferView\": 0, \
			 \"mimeType\": \"image/png\" }, { \"bufferView\": 0, \"mimeType\": \"image/webp\" } \
			 ], \"materials\": [ { \"pbrMetallicRoughness\": { \"baseColorTexture\": { \
			 \"index\": 0 } } } ]",
		);

		assert_eq!(coats.surfaces[0].albedo, None);
		assert!(coats.pictures.is_empty());
		assert!(complained(&coats, "image/webp"), "got {:?}", coats.warnings);
	}

	#[test]
	fn a_material_is_read_even_when_the_geometry_beside_it_is_not() {
		// the two halves of the importer meet only in the model, so a file with
		// materials and no meshes is a thing that has to come out whole.
		let file = Gltf::read(
			include_bytes!("fixtures/model.glb"),
			Path::new("model.glb"),
			Path::new(""),
		)
		.expect("the fixture reads");
		let model = import(&file).expect("it imports");

		assert_eq!(model.materials.len(), 2);
		assert_eq!(model.textures.len(), 2);
		assert_eq!(
			model.meshes[0].material,
			Some(0),
			"the arm is made of the first material the file declares"
		);
		assert_eq!(model.meshes[1].material, Some(1), "and the column of the second");
	}

	#[test]
	fn a_picture_inside_a_model_may_be_a_jpeg() {
		// the specification allows either, so a model handed over by somebody
		// else carries whichever their tool wrote.
		let text = format!(
			"{{ \"asset\": {{ \"version\": \"2.0\" }}, \"buffers\": [ {{ \"byteLength\": \
			 {PHOTO_BYTES}, \"uri\": \"data:application/octet-stream;base64,{PHOTO}\" }} ], \
			 \"bufferViews\": [ {{ \"buffer\": 0, \"byteLength\": {PHOTO_BYTES} }} ], \
			 \"images\": [ {{ \"name\": \"wall\", \"bufferView\": 0, \"mimeType\": \
			 \"image/jpeg\" }} ], \"textures\": [ {{ \"source\": 0 }} ], \"materials\": [ {{ \
			 \"pbrMetallicRoughness\": {{ \"baseColorTexture\": {{ \"index\": 0 }} }} }} ] }}"
		);
		let file = Gltf::read(text.as_bytes(), Path::new("model.gltf"), Path::new(""))
			.expect("the document reads");
		let coats = read(&file);

		assert_eq!(coats.warnings, Vec::<String>::new(), "nothing was dropped");
		assert_eq!(coats.pictures.len(), 1);
		assert_eq!(coats.pictures[0].name, "wall");
		assert_eq!(coats.pictures[0].data.width, 32);
		assert_eq!(coats.pictures[0].data.texel, Texel::Rgba8Srgb);
		assert_eq!(coats.surfaces[0].albedo, Some(Picture::Inside(0)));
	}
}
