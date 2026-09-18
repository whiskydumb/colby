//! Reading a glTF's materials, and the pictures they name.
//!
//! glTF's material model is the metallic-roughness pair, which is the one
//! `abi::material` was built around, so the numbers cross with no translation
//! at all: a base color, how metallic, how rough, the color picture, the normal
//! map, the picture of how metal and how rough, the occlusion picture, and the
//! light a surface gives off with the picture of where it gives it off. Three
//! extensions finish the record - the strength that light is given at, a
//! transform of the pictures' coordinates, and a surface drawn with no light on
//! it - and what a material says beyond them is named in a warning and dropped,
//! because this renderer has nowhere to put it.
//!
//! **A picture reaches colby one of two ways, and which one decides who
//! compiles it.** An image written as a file beside the document is already an
//! asset: the compiler finds it on its own walk and turns it into a `.ctex`,
//! and a material only has to say which file. An image stored *inside* the
//! model has no such file, so it is decoded here and handed back to be written
//! out beside the model's meshes.
//!
//! **That split is what decides the texel layout.** What a picture's channels
//! mean is not in a PNG, so a loose one is judged by its name - the `_normal`
//! and `_orm` suffixes, the only such rule in the project. A material *says*,
//! so an extracted picture needs no rule at all and is written with the layout
//! its use asks for. A loose picture whose name disagrees with its use is
//! **copied out** the same way, in the layout its use asks for: its own file is
//! still compiled by its name, and the model wears the copy.
//!
//! **One transform for the first set of coordinates, and none for the second.**
//! The exchange format moves each picture's coordinates on its own; colby moves
//! all of a material's pictures on the first set together, by the color
//! picture's transform - or by the first other picture's when it has no color
//! picture - and names in a warning any picture moved another way. The second
//! set is one unwrap of the whole mesh, laid out for a baked picture, and
//! nothing moves it. Which set a picture is read from is the file's to say for
//! the occlusion and the glow, the two pictures a bake lays out; the others are
//! read from the first set, with a warning when the file said otherwise.
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
		Material,
		material::{Blend, MASK_CUTOFF, Wrap},
		texture::{Texel, TextureData},
	},
	glam::{Vec2, Vec3},
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

/// The extension that gives the emitted light a strength of its own.
const EMISSIVE_STRENGTH: &str = "KHR_materials_emissive_strength";

/// The extension that moves a picture's coordinates.
const TRANSFORM: &str = "KHR_texture_transform";

/// The extension that draws a surface with no light on it.
const UNLIT: &str = "KHR_materials_unlit";

/// Every material a file declares, and what came out with them.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Coats {
	/// One per material, in the file's own order.
	pub surfaces: Vec<Surface>,

	/// Pictures that were inside the file, or beside it under a name that says
	/// otherwise, and now need writing out.
	pub pictures: Vec<Extracted>,

	/// What the materials said that could not be used.
	pub warnings: Vec<String>,
}

/// One of the file's materials, as colby's own numbers.
///
/// **The numbers are the live record's**, an `abi::Material` with its five
/// picture handles left at what the default holds: no handle exists until the
/// host has registered the pictures. Where each picture comes from is beside
/// it, in the five places a handle will go.
#[derive(Clone, Debug, PartialEq)]
pub struct Surface {
	/// What it registers under, inside the model's own name.
	pub name: String,

	/// Every number: the file's where it said one, the exchange format's
	/// default where it did not.
	pub numbers: Material,

	/// The color picture, if it has one.
	pub albedo: Option<Picture>,

	/// The normal map, if it has one.
	pub normal: Option<Picture>,

	/// The picture of how metal and how rough, if it has one.
	pub finish: Option<Picture>,

	/// The occlusion picture, if it has one.
	pub occlusion: Option<Picture>,

	/// The picture of where it gives off light, if it has one.
	pub glow: Option<Picture>,
}

impl Default for Surface {
	fn default() -> Self {
		Self {
			name: String::new(),
			// metal and rough, which is what the exchange format says a factor
			// the file leaves out is
			numbers: Material {
				metallic: 1.0,
				roughness: 1.0,
				..Material::DEFAULT
			},
			albedo: None,
			normal: None,
			finish: None,
			occlusion: None,
			glow: None,
		}
	}
}

/// Where one of a material's pictures comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Picture {
	/// A file in the asset tree, which the compiler already turns into a
	/// texture on its own walk. The material only has to name it.
	Beside(PathBuf),

	/// One that was taken out of the model, or copied out of a file beside it,
	/// by its index in [`Coats::pictures`].
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

/// What a material uses one of its pictures for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Use {
	/// The color.
	Albedo,

	/// Which way the surface faces.
	Normal,

	/// How metal, in blue, and how rough, in green.
	Finish,

	/// How much light from everywhere reaches it, in red.
	Occlusion,

	/// Where it gives off light.
	Glow,
}

impl Use {
	/// What a warning calls it.
	const fn what(self) -> &'static str {
		match self {
			| Self::Albedo => "color picture",
			| Self::Normal => "normal map",
			| Self::Finish => "metal and roughness picture",
			| Self::Occlusion => "occlusion picture",
			| Self::Glow => "emissive picture",
		}
	}

	/// How its channels are stored: a color is decoded out of sRGB, and
	/// everything else is numbers.
	const fn texel(self) -> Texel {
		match self {
			| Self::Albedo | Self::Glow => Texel::Rgba8Srgb,
			| Self::Normal | Self::Finish | Self::Occlusion => Texel::Rgba8Unorm,
		}
	}

	/// What a copy of it is named with, so that the layout shows in the tree.
	const fn suffix(self) -> &'static str {
		match self {
			| Self::Albedo | Self::Glow => "",
			| Self::Normal => compile::NORMAL_SUFFIX,
			| Self::Finish | Self::Occlusion => compile::ORM_SUFFIX,
		}
	}

	/// Whether colby reads it from the second set of coordinates when the file
	/// asks: the two pictures a bake lays out over a mesh's own unwrap.
	const fn takes_second(self) -> bool { matches!(self, Self::Occlusion | Self::Glow) }
}

/// How a picture's coordinates are moved before it is read: scaled, turned,
/// then offset, as the exchange format says.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Moved {
	offset: Vec2,
	rotation: f32,
	scale: Vec2,
}

impl Moved {
	/// Coordinates left where they are.
	const NONE: Self = Self {
		offset: Vec2::ZERO,
		rotation: 0.0,
		scale: Vec2::ONE,
	};

	/// What a picture's reference says about its coordinates, or nothing moved.
	fn of(reference: &Value) -> Self {
		let Some(written) = transform_of(reference) else {
			return Self::NONE;
		};

		Self {
			offset: pair(written.get("offset")).unwrap_or(Vec2::ZERO),
			rotation: written
				.get("rotation")
				.and_then(Value::as_f32)
				.unwrap_or(0.0),
			scale: pair(written.get("scale")).unwrap_or(Vec2::ONE),
		}
	}

	/// Whether two move a picture the same way, to a millionth.
	fn agrees(self, other: Self) -> bool {
		self.offset.abs_diff_eq(other.offset, 1.0e-6)
			&& (self.rotation - other.rotation).abs() <= 1.0e-6
			&& self.scale.abs_diff_eq(other.scale, 1.0e-6)
	}
}

/// One of a material's pictures, as the material refers to it.
struct Sampled {
	/// Which picture, when there is one colby can use.
	picture: Option<Picture>,

	/// Which set of coordinates the file says it is read from.
	set: usize,

	/// How those coordinates are moved first.
	moved: Moved,
}

impl Sampled {
	/// A picture the material does not have.
	const NONE: Self = Self {
		picture: None,
		set: 0,
		moved: Moved::NONE,
	};
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

		self.complain(&entry);

		let [albedo, normal, finish, occlusion, glow] = [
			(pbr.get("baseColorTexture"), Use::Albedo),
			(entry.get("normalTexture"), Use::Normal),
			(pbr.get("metallicRoughnessTexture"), Use::Finish),
			(entry.get("occlusionTexture"), Use::Occlusion),
			(entry.get("emissiveTexture"), Use::Glow),
		]
		.map(|(reference, used)| self.sampled(reference, used));
		// asked of all five, because the three colby reads from the first set
		// alone have a word to say when the file said otherwise
		let [_, _, _, occlusion_uv2, glow_uv2] = [
			self.second(&albedo, Use::Albedo),
			self.second(&normal, Use::Normal),
			self.second(&finish, Use::Finish),
			self.second(&occlusion, Use::Occlusion),
			self.second(&glow, Use::Glow),
		];
		let moved = self.moved(&[
			(&albedo, Use::Albedo, false),
			(&normal, Use::Normal, false),
			(&finish, Use::Finish, false),
			(&occlusion, Use::Occlusion, occlusion_uv2),
			(&glow, Use::Glow, glow_uv2),
		]);
		let extensions = entry.get("extensions");

		Surface {
			name: self.name(&entry),
			numbers: Material {
				base_color: color(pbr.get("baseColorFactor")).unwrap_or(Vec3::ONE),
				metallic: number(&pbr, "metallicFactor"),
				roughness: number(&pbr, "roughnessFactor"),
				occlusion_strength: entry
					.get("occlusionTexture")
					.and_then(|reference| reference.get("strength"))
					.and_then(Value::as_f32)
					.unwrap_or(1.0),
				occlusion_uv2,
				glow_uv2,
				// black and at one where the file says nothing, which is what the
				// exchange format and its extension say
				emissive: color(entry.get("emissiveFactor")).unwrap_or(Vec3::ZERO),
				emissive_strength: extensions
					.and_then(|all| all.get(EMISSIVE_STRENGTH))
					.and_then(|written| written.get("emissiveStrength"))
					.and_then(Value::as_f32)
					.unwrap_or(1.0),
				uv_scale: moved.scale,
				uv_offset: moved.offset,
				uv_rotation: moved.rotation,
				wrap: self.wrap(pbr.get("baseColorTexture")),
				blend: self.blend(&entry),
				// the fourth channel of the same factor the color came from. The
				// exchange format says it is ignored outside the blended mode,
				// and so does everything that reads it; it is carried across
				// anyway, because a mode changed later should find the number
				// already there rather than at one.
				opacity: opacity(pbr.get("baseColorFactor")),
				unlit: extensions
					.and_then(|all| all.get(UNLIT))
					.is_some(),
				..Material::DEFAULT
			},
			albedo: albedo.picture,
			normal: normal.picture,
			finish: finish.picture,
			occlusion: occlusion.picture,
			glow: glow.picture,
		}
	}

	/// One of a material's pictures, with the set it is read from and how its
	/// coordinates are moved.
	fn sampled(&mut self, reference: Option<&Value>, used: Use) -> Sampled {
		let Some(reference) = reference else {
			return Sampled::NONE;
		};
		// the transform's own set, when it names one, is the one the file means:
		// that is how an exporter keeps a fallback for a reader that ignores it
		let set = transform_of(reference)
			.and_then(|written| written.get("texCoord"))
			.or_else(|| reference.get("texCoord"))
			.and_then(Value::as_usize)
			.unwrap_or(0);

		Sampled {
			picture: self.picture(reference, used),
			set,
			moved: Moved::of(reference),
		}
	}

	/// Whether a picture is read from the second set of coordinates, and a word
	/// about a set colby does not read it from.
	fn second(&mut self, sampled: &Sampled, used: Use) -> bool {
		if sampled.picture.is_none() || sampled.set == 0 {
			return false;
		}

		if sampled.set == 1 && used.takes_second() {
			return true;
		}

		if sampled.set == 1 {
			self.note(&format!(
				"reads its {} from a second set of texture coordinates, and colby reads it from \
				 the first",
				used.what()
			));
		} else {
			self.note(&format!(
				"reads its {} from texture coordinates {}, and colby has two sets; it reads the \
				 first",
				used.what(),
				sampled.set
			));
		}

		false
	}

	/// How the first set of coordinates is moved: by the color picture's
	/// transform, or the first other picture's when there is no color picture,
	/// with a word about every picture moved another way.
	///
	/// @param pictures - each picture, what it is for, and whether it is read
	/// from the second set
	fn moved(&mut self, pictures: &[(&Sampled, Use, bool); 5]) -> Moved {
		for (sampled, used, second) in pictures {
			if *second && !sampled.moved.agrees(Moved::NONE) {
				self.note(&format!(
					"moves its {}, which it reads from the second set of texture coordinates, \
					 and colby moves only the first",
					used.what()
				));
			}
		}

		let first = || {
			pictures
				.iter()
				.filter(|(sampled, _, second)| sampled.picture.is_some() && !*second)
		};
		let Some((moved, from)) = first()
			.next()
			.map(|(sampled, used, _)| (sampled.moved, *used))
		else {
			return Moved::NONE;
		};

		for (sampled, used, _) in first() {
			if !sampled.moved.agrees(moved) {
				self.note(&format!(
					"moves its {} differently from its {}, and colby moves every picture on the \
					 first set of coordinates one way",
					used.what(),
					from.what()
				));
			}
		}

		moved
	}

	/// How a material says its alpha should be read.
	///
	/// The exchange format's three modes are colby's three, which is not a
	/// coincidence: both took the set that the hardware actually distinguishes.
	/// A word this reader does not know is a warning and an opaque surface,
	/// which is the answer that draws something rather than nothing.
	fn blend(&mut self, entry: &Value) -> Blend {
		let written = entry
			.get("alphaMode")
			.and_then(Value::as_str)
			.unwrap_or("OPAQUE");
		let mode = match written {
			| "OPAQUE" => Blend::Opaque,
			| "MASK" => Blend::Mask,
			| "BLEND" => Blend::Alpha,
			| _ => {
				self.note(&format!("reads its alpha as {written}, which is not a mode"));

				Blend::Opaque
			},
		};

		// the format's own default is the half colby cuts at, so a mask that
		// says nothing about it imports exactly. One that says something else
		// cannot: the threshold is a constant in the shader rather than a
		// number on the material. @ref `colby-pre-commit-audit`, TR-2.
		if mode == Blend::Mask {
			let cutoff = entry
				.get("alphaCutoff")
				.and_then(Value::as_f32)
				.unwrap_or(MASK_CUTOFF);

			if (cutoff - MASK_CUTOFF).abs() > 1.0e-6 {
				self.note(&format!(
					"cuts its holes out at {cutoff} and colby cuts every one of them at \
					 {MASK_CUTOFF}"
				));
			}
		}

		mode
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
	fn picture(&mut self, reference: &Value, used: Use) -> Option<Picture> {
		let texture = reference.get("index").and_then(Value::as_usize)?;
		let image = self
			.file
			.table("textures")
			.get(texture)
			.and_then(|entry| entry.get("source"))
			.and_then(Value::as_usize)?;
		let entry = self.file.table("images").get(image)?.clone();

		match entry.get("uri").and_then(Value::as_str) {
			| Some(uri) if !uri.starts_with(super::DATA_PREFIX) =>
				self.file_beside(image, &entry, uri, used),
			| _ => self.extract(image, &entry, used),
		}
	}

	/// A picture that is a file of its own, which the compiler already knows
	/// how to turn into a texture - unless its name says it is something else.
	fn file_beside(
		&mut self,
		image: usize,
		entry: &Value,
		uri: &str,
		used: Use,
	) -> Option<Picture> {
		let path = self.file.beside(uri).or_else(|| {
			self.note("names a picture outside the asset tree, and it is left out");

			None
		})?;

		// the loose file will be compiled by the naming rule, which cannot see
		// what this material says about it. When the two disagree the file
		// would end up bent one way or lit from the wrong side, so the model
		// wears a copy of it in the layout this material asks for instead,
		// made the way a picture inside the model is.
		if compile::texel_of(&path) != used.texel() {
			return self.extract(image, entry, used);
		}

		Some(Picture::Beside(path))
	}

	/// A picture that has to be written out beside the model, decoded and kept.
	fn extract(&mut self, image: usize, entry: &Value, used: Use) -> Option<Picture> {
		let texel = used.texel();

		if let Some((.., already)) = self
			.pulled
			.iter()
			.find(|(which, layout, _)| *which == image && *layout == texel)
		{
			return Some(Picture::Inside(*already));
		}

		let bytes = self.bytes(entry)?;
		// what the document says the bytes are, or what they say they are: a
		// picture beside the file has no type written down for it
		let kind = entry
			.get("mimeType")
			.and_then(Value::as_str)
			.unwrap_or_else(|| if bytes.starts_with(&[0xFF, 0xD8]) { JPEG } else { PNG });

		if kind != PNG && kind != JPEG {
			self.note(&format!("holds a {kind} picture, which colby does not decode"));

			return None;
		}

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
		let name = self.picture_name(image, entry, used);

		self.coats.pictures.push(Extracted { name, data });
		self.pulled.push((image, texel, index));

		Some(Picture::Inside(index))
	}

	/// The bytes of a picture, wherever the document keeps them.
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

		if !uri.starts_with(super::DATA_PREFIX) {
			let path = self.file.beside(uri)?;

			return match std::fs::read(&path) {
				| Ok(bytes) => Some(bytes),
				| Err(error) => {
					self.note(&format!(
						"names {} and it cannot be read: {error}",
						path.display()
					));

					None
				},
			};
		}

		let bytes = super::inline(uri);

		if bytes.is_none() {
			self.note("holds a picture written as an address colby cannot read");
		}

		bytes
	}

	/// The name an extracted picture registers under.
	fn picture_name(&mut self, image: usize, entry: &Value, used: Use) -> String {
		let written = entry
			.get("name")
			.and_then(Value::as_str)
			.or_else(|| {
				// a picture beside the file is called what the file is called
				entry
					.get("uri")
					.and_then(Value::as_str)
					.filter(|uri| !uri.starts_with(super::DATA_PREFIX))
					.and_then(|uri| std::path::Path::new(uri).file_stem())
					.and_then(|stem| stem.to_str())
			})
			.unwrap_or("");
		let mut base = super::tidy(written);

		if base.is_empty() {
			base = format!("picture{image}");
		}

		// the suffix a loose file of that layout would carry, so that two
		// layouts of one picture are two names and anybody reading the output
		// tree can tell which is which.
		let said = base.ends_with(compile::NORMAL_SUFFIX) || base.ends_with(compile::ORM_SUFFIX);

		if !used.texel().is_color() && !said {
			base = format!("{base}{}", used.suffix());
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

	/// Names what the material declares that this renderer has nowhere to put.
	///
	/// One place for it so the list is readable as a list, which is also what
	/// it is: the gap between glTF's material and colby's, which is down to
	/// one number since the rest of the material arrived.
	fn complain(&mut self, entry: &Value) {
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

/// A picture's `KHR_texture_transform`, when it has one.
fn transform_of(reference: &Value) -> Option<&Value> {
	reference
		.get("extensions")
		.and_then(|all| all.get(TRANSFORM))
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

/// The fourth channel of a base color factor, or all of it.
fn opacity(written: Option<&Value>) -> f32 {
	written
		.map(Value::as_array)
		.and_then(|cells| cells.get(3).and_then(Value::as_f32))
		.unwrap_or(1.0)
}

/// The first three of three or four numbers, or nothing for a factor the file
/// did not write whole.
pub(super) fn color(written: Option<&Value>) -> Option<Vec3> {
	let cells = written.map(Value::as_array)?;

	if cells.len() < 3 {
		return None;
	}

	Some(Vec3::new(
		cells[0].as_f32().unwrap_or(1.0),
		cells[1].as_f32().unwrap_or(1.0),
		cells[2].as_f32().unwrap_or(1.0),
	))
}

/// Two numbers, or nothing for anything else.
fn pair(written: Option<&Value>) -> Option<Vec2> {
	let cells = written.map(Value::as_array)?;

	match cells {
		| [across, down] => Some(Vec2::new(across.as_f32()?, down.as_f32()?)),
		| _ => None,
	}
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

	/// One material over the one picture, with its references written by the
	/// test and a texture that names the picture.
	fn material(references: &str) -> Coats {
		document(&format!(
			"\"textures\": [ {{ \"source\": 0 }} ], \"materials\": [ {{ {references} }} ]"
		))
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
				.numbers
				.base_color
				.abs_diff_eq(Vec3::new(0.8, 0.6, 0.2), 1e-6),
			"the color it was given, in the space it was written in: {}",
			brass.numbers.base_color
		);
		assert!((brass.numbers.metallic - 0.0).abs() < 1e-6);
		assert!((brass.numbers.roughness - 0.5).abs() < 1e-6);
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
	fn a_loose_picture_whose_name_disagrees_with_its_use_is_worn_as_a_copy_in_the_right_layout() {
		// what used to be a warning and a picture bent the wrong way: the file
		// beside the model is still compiled by its name, and the model wears
		// a copy of it made the way a picture inside the model is.
		let dir = std::env::temp_dir()
			.join("colby-gltf-tests")
			.join("disagree");

		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(dir.join("models")).expect("the fixture is made");
		fs::write(dir.join("models").join("bumps.png"), include_bytes!("fixtures/tiles.png"))
			.expect("the picture is written");

		let text = "{ \"asset\": { \"version\": \"2.0\" }, \"images\": [ { \"uri\": \
		            \"bumps.png\" } ], \"textures\": [ { \"source\": 0 } ], \"materials\": [ { \
		            \"normalTexture\": { \"index\": 0 }, \"occlusionTexture\": { \"index\": 0 \
		            }, \"pbrMetallicRoughness\": { \"baseColorTexture\": { \"index\": 0 } } } ] \
		            }";

		fs::write(dir.join("models").join("model.gltf"), text).expect("the document is written");

		let file = Gltf::open(&dir.join("models").join("model.gltf"), &dir).expect("it reads");
		let coats = read(&file);
		let surface = &coats.surfaces[0];

		assert_eq!(coats.warnings, Vec::<String>::new(), "nothing is wrong any more");
		assert_eq!(
			surface.albedo,
			Some(Picture::Beside(dir.join("models").join("bumps.png"))),
			"a color named as a color is the file itself"
		);
		assert_eq!(surface.normal, Some(Picture::Inside(0)), "a normal map is a copy");
		assert_eq!(
			surface.occlusion,
			Some(Picture::Inside(0)),
			"and the occlusion is the same copy: one layout, decoded once"
		);
		assert_eq!(coats.pictures.len(), 1);
		assert_eq!(
			coats.pictures[0].data.texel,
			Texel::Rgba8Unorm,
			"in the layout its use asks for"
		);
		assert_eq!(coats.pictures[0].name, "bumps_normal", "named after the file and the layout");

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

		assert!(
			only.numbers
				.base_color
				.abs_diff_eq(Vec3::ONE, 1e-6),
			"white"
		);
		assert!((only.numbers.metallic - 1.0).abs() < 1e-6, "and metal, which is glTF's default");
		assert!((only.numbers.roughness - 1.0).abs() < 1e-6, "and rough");
		assert_eq!(only.numbers.emissive, Vec3::ZERO, "giving off nothing");
		assert!((only.numbers.emissive_strength - 1.0).abs() < 1e-6, "at a strength of one");
		assert!((only.numbers.occlusion_strength - 1.0).abs() < 1e-6, "all of its occlusion");
		assert!(!only.numbers.unlit, "lit");
		assert_eq!(
			(only.numbers.uv_scale, only.numbers.uv_offset),
			(Vec2::ONE, Vec2::ZERO),
			"and its pictures where the mesh puts them"
		);
		assert_eq!(only.name, "material0", "and numbered, having no name");
		assert!(coats.warnings.is_empty(), "none of which is a complaint: {:?}", coats.warnings);
	}

	#[test]
	fn a_sampler_says_what_happens_past_the_edge() {
		let clamped = document(
			"\"samplers\": [ { \"wrapS\": 33071, \"wrapT\": 33071 } ], \"textures\": [ { \
			 \"source\": 0, \"sampler\": 0 } ], \"materials\": [ { \"pbrMetallicRoughness\": { \
			 \"baseColorTexture\": { \"index\": 0 } } } ]",
		);

		assert_eq!(clamped.surfaces[0].numbers.wrap, Wrap::Clamp);

		let mirrored = document(
			"\"samplers\": [ { \"wrapS\": 33648, \"wrapT\": 33648 } ], \"textures\": [ { \
			 \"source\": 0, \"sampler\": 0 } ], \"materials\": [ { \"pbrMetallicRoughness\": { \
			 \"baseColorTexture\": { \"index\": 0 } } } ]",
		);

		assert_eq!(
			mirrored.surfaces[0].numbers.wrap,
			Wrap::Repeat,
			"the nearest thing colby has"
		);
		assert!(complained(&mirrored, "mirrored"), "got {:?}", mirrored.warnings);
	}

	#[test]
	fn the_rest_of_the_material_crosses_and_nothing_about_it_is_a_complaint() {
		// every one of these was a warning and a dropped picture until this
		// card: a metal and roughness picture, an occlusion picture and its
		// strength, an emissive color, its picture and its strength, and a
		// surface with no light on it.
		let coats = material(
			"\"emissiveFactor\": [ 1, 0.5, 0 ], \"emissiveTexture\": { \"index\": 0 }, \
			 \"occlusionTexture\": { \"index\": 0, \"strength\": 0.25 }, \"extensions\": { \
			 \"KHR_materials_emissive_strength\": { \"emissiveStrength\": 5 }, \
			 \"KHR_materials_unlit\": {} }, \"pbrMetallicRoughness\": { \
			 \"metallicRoughnessTexture\": { \"index\": 0 } }",
		);
		let only = &coats.surfaces[0];

		assert_eq!(coats.warnings, Vec::<String>::new(), "nothing is dropped");
		assert!(
			only.numbers
				.emissive
				.abs_diff_eq(Vec3::new(1.0, 0.5, 0.0), 1e-6)
		);
		assert!(
			(only.numbers.emissive_strength - 5.0).abs() < 1e-6,
			"the strength is a number of its own and the factor is kept as it was written"
		);
		assert!((only.numbers.occlusion_strength - 0.25).abs() < 1e-6);
		assert!(only.numbers.unlit, "and the surface is drawn as its color");

		// the picture is the same image three times over: numbers for the finish
		// and the occlusion, decoded once, and a color for the glow
		assert_eq!(only.finish, only.occlusion, "one numbers copy for both");
		assert_ne!(only.finish, only.glow, "and a color one for the glow");
		assert_eq!(coats.pictures.len(), 2);

		let layout = |picture: &Option<Picture>| match picture {
			| Some(Picture::Inside(at)) => coats.pictures[*at].data.texel,
			| other => panic!("an extracted picture was expected, not {other:?}"),
		};

		assert_eq!(layout(&only.finish), Texel::Rgba8Unorm, "a finish is numbers");
		assert_eq!(layout(&only.glow), Texel::Rgba8Srgb, "and a glow is a color");
		assert!(
			coats
				.pictures
				.iter()
				.any(|picture| picture.name == "picture_orm"),
			"and the numbers are named the way a loose one would be: {:?}",
			coats
				.pictures
				.iter()
				.map(|picture| &picture.name)
				.collect::<Vec<_>>()
		);
	}

	#[test]
	fn a_picture_on_the_second_set_is_read_from_it_where_colby_can() {
		let baked = material(
			"\"occlusionTexture\": { \"index\": 0, \"texCoord\": 1 }, \"emissiveTexture\": { \
			 \"index\": 0, \"texCoord\": 1 }",
		);

		assert!(baked.surfaces[0].numbers.occlusion_uv2, "the occlusion, laid out by a bake");
		assert!(baked.surfaces[0].numbers.glow_uv2, "and the glow");
		assert_eq!(baked.warnings, Vec::<String>::new());

		let elsewhere = material(
			"\"normalTexture\": { \"index\": 0, \"texCoord\": 1 }, \"occlusionTexture\": { \
			 \"index\": 0, \"texCoord\": 2 }, \"pbrMetallicRoughness\": { \"baseColorTexture\": \
			 { \"index\": 0, \"texCoord\": 1 } }",
		);

		assert!(!elsewhere.surfaces[0].numbers.occlusion_uv2, "a third set is read as the first");
		assert!(
			complained(&elsewhere, "a second set of texture coordinates"),
			"and the color and the normal map, which colby reads from the first, say so: {:?}",
			elsewhere.warnings
		);
		assert!(complained(&elsewhere, "coordinates 2"), "as does the third set");
		assert_eq!(elsewhere.warnings.len(), 3, "one line each: {:?}", elsewhere.warnings);
	}

	#[test]
	fn a_transform_on_the_color_picture_moves_the_first_set() {
		let coats = material(
			"\"normalTexture\": { \"index\": 0, \"extensions\": { \"KHR_texture_transform\": { \
			 \"offset\": [ 0.25, 0.5 ], \"rotation\": 0.5, \"scale\": [ 2, 3 ] } } }, \
			 \"pbrMetallicRoughness\": { \"baseColorTexture\": { \"index\": 0, \"extensions\": \
			 { \"KHR_texture_transform\": { \"offset\": [ 0.25, 0.5 ], \"rotation\": 0.5, \
			 \"scale\": [ 2, 3 ] } } } }",
		);
		let only = &coats.surfaces[0].numbers;

		assert_eq!(only.uv_offset, Vec2::new(0.25, 0.5));
		assert!((only.uv_rotation - 0.5).abs() < 1e-6);
		assert_eq!(only.uv_scale, Vec2::new(2.0, 3.0));
		assert_eq!(coats.warnings, Vec::<String>::new(), "moved alike, so nothing to say");
	}

	#[test]
	fn a_picture_moved_another_way_is_named_and_moved_with_the_rest() {
		let coats = material(
			"\"normalTexture\": { \"index\": 0, \"extensions\": { \"KHR_texture_transform\": { \
			 \"scale\": [ 4, 4 ] } } }, \"occlusionTexture\": { \"index\": 0, \"texCoord\": 1, \
			 \"extensions\": { \"KHR_texture_transform\": { \"offset\": [ 0.5, 0 ] } } }, \
			 \"pbrMetallicRoughness\": { \"baseColorTexture\": { \"index\": 0 } }",
		);

		assert_eq!(
			coats.surfaces[0].numbers.uv_scale,
			Vec2::ONE,
			"the color picture's transform, which is none"
		);
		assert!(complained(&coats, "normal map differently from its color picture"));
		assert!(
			complained(&coats, "reads from the second set"),
			"and the second set is not moved at all: {:?}",
			coats.warnings
		);
	}

	#[test]
	fn a_material_with_no_color_picture_is_moved_by_the_first_picture_it_has() {
		let coats = material(
			"\"normalTexture\": { \"index\": 0, \"extensions\": { \"KHR_texture_transform\": { \
			 \"scale\": [ 4, 2 ] } } }",
		);

		assert_eq!(coats.surfaces[0].numbers.uv_scale, Vec2::new(4.0, 2.0));
		assert_eq!(coats.warnings, Vec::<String>::new());
	}

	#[test]
	fn a_transform_may_move_a_picture_to_the_second_set_by_itself() {
		// the exchange format's fallback: the reference keeps a set a reader that
		// ignores the transform can use, and the transform names the one it means
		let coats = material(
			"\"occlusionTexture\": { \"index\": 0, \"texCoord\": 0, \"extensions\": { \
			 \"KHR_texture_transform\": { \"texCoord\": 1 } } }",
		);

		assert!(coats.surfaces[0].numbers.occlusion_uv2);
	}

	#[test]
	fn only_the_normal_maps_scale_is_left_to_complain_about() {
		let coats = material(
			"\"normalTexture\": { \"index\": 0, \"scale\": 2 }, \"occlusionTexture\": { \
			 \"index\": 0 }, \"emissiveFactor\": [ 1, 0, 0 ], \"alphaMode\": \"BLEND\", \
			 \"pbrMetallicRoughness\": { \"metallicRoughnessTexture\": { \"index\": 0 } }",
		);

		// what it says about its alpha is *not* a complaint either, and has not
		// been since colby blended: the file asks to be blended and is
		assert_eq!(coats.surfaces[0].numbers.blend, Blend::Alpha, "the mode came across");
		assert!(complained(&coats, "scales its normal map"));
		assert_eq!(coats.warnings.len(), 1, "and that is all: {:?}", coats.warnings);
	}

	#[test]
	fn the_three_words_a_material_can_say_about_its_alpha_are_the_three_modes() {
		let coats = document(
			"\"materials\": [ { \"alphaMode\": \"OPAQUE\" }, { \"alphaMode\": \"MASK\" }, { \
			 \"alphaMode\": \"BLEND\" }, { } ]",
		);
		let modes: Vec<Blend> = coats
			.surfaces
			.iter()
			.map(|surface| surface.numbers.blend)
			.collect();

		assert_eq!(
			modes,
			vec![Blend::Opaque, Blend::Mask, Blend::Alpha, Blend::Opaque],
			"and a material that says nothing is opaque, which is what the format says"
		);
		assert!(coats.warnings.is_empty(), "none of them is a complaint: {:?}", coats.warnings);
	}

	#[test]
	fn a_word_this_reader_does_not_know_is_named_and_drawn_solid() {
		let coats = document("\"materials\": [ { \"alphaMode\": \"STOCHASTIC\" } ]");

		assert_eq!(
			coats.surfaces[0].numbers.blend,
			Blend::Opaque,
			"drawing it solid is the answer that draws something"
		);
		assert!(complained(&coats, "STOCHASTIC"), "and it is named: {:?}", coats.warnings);
	}

	#[test]
	fn the_fourth_channel_of_a_base_color_is_how_much_of_the_surface_there_is() {
		let coats = document(
			"\"materials\": [ { \"alphaMode\": \"BLEND\", \"pbrMetallicRoughness\": { \
			 \"baseColorFactor\": [ 0.2, 0.4, 0.6, 0.25 ] } }, { \"pbrMetallicRoughness\": { \
			 \"baseColorFactor\": [ 1, 1, 1 ] } }, { } ]",
		);

		assert!((coats.surfaces[0].numbers.opacity - 0.25).abs() < 1e-6, "the alpha written");
		assert!(
			coats.surfaces[0]
				.numbers
				.base_color
				.abs_diff_eq(Vec3::new(0.2, 0.4, 0.6), 1e-6),
			"and the three channels beside it are untouched by reading it"
		);
		assert!(
			(coats.surfaces[1].numbers.opacity - 1.0).abs() < 1e-6,
			"a factor of three numbers has no alpha in it, so the surface is all there"
		);
		assert!(
			(coats.surfaces[2].numbers.opacity - 1.0).abs() < 1e-6,
			"and so has no factor at all"
		);
	}

	#[test]
	fn a_mask_that_cuts_somewhere_else_is_named_rather_than_quietly_moved() {
		let quiet = document("\"materials\": [ { \"alphaMode\": \"MASK\" } ]");

		assert!(
			quiet.warnings.is_empty(),
			"the format's own default is the half colby cuts at, so saying nothing is exact: \
			 {:?}",
			quiet.warnings
		);

		let stated =
			document("\"materials\": [ { \"alphaMode\": \"MASK\", \"alphaCutoff\": 0.5 } ]");

		assert!(stated.warnings.is_empty(), "and so is saying it: {:?}", stated.warnings);

		let moved =
			document("\"materials\": [ { \"alphaMode\": \"MASK\", \"alphaCutoff\": 0.9 } ]");

		assert_eq!(moved.surfaces[0].numbers.blend, Blend::Mask, "it is still a mask");
		assert!(
			complained(&moved, "0.9"),
			"and the number it wanted is named: {:?}",
			moved.warnings
		);

		// the same number on a material that is not a mask says nothing about
		// anything, and the format says so outright.
		let elsewhere =
			document("\"materials\": [ { \"alphaMode\": \"BLEND\", \"alphaCutoff\": 0.9 } ]");

		assert!(
			elsewhere.warnings.is_empty(),
			"a cutoff outside a mask is not a cutoff: {:?}",
			elsewhere.warnings
		);
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
