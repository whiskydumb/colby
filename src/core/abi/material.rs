//! Materials: what a surface is made of, and the host's registry of them.
//!
//! The smallest thing that deserves the word, and it is deliberately the
//! metallic-roughness pair rather than anything more expressive: it is what
//! glTF and every real-time renderer of the last decade agree on, so anything
//! exported from anywhere already speaks it.
//!
//! A material names a texture by handle rather than owning one, so reloading an
//! image reaches every material using it without any of them being told. @ref
//! [`registry`](super::registry) for the shape all three tables share, and
//! [`textures`](super::texture) for what the handle points at.

use super::{
	field::{Field, field, word},
	registry::{Entry, Registry},
	texture::TextureId,
};
use crate::{
	glam::{Vec2, Vec3},
	registry_handle,
};

/// The name the always-present default material is registered under.
pub const DEFAULT_NAME: &str = "default";

/// The name the always-present water material is registered under.
///
/// A second built-in, and the only other one, because water is the only thing
/// this engine can make on its own that has to be *drawn* a particular way:
/// every other surface is what somebody's material says it is, and a fluid
/// that came out opaque would hide the very thing it is there to show. @ref
/// [`Material::WATER`] and `colby_core::abi::water`.
pub const WATER_NAME: &str = "water";

/// What a sampler does past the edge of a texture.
///
/// Two, because there are two answers anybody wants: a tiled surface repeats
/// and a decal does not. Mirroring and a border color exist in every graphics
/// API and neither has ever been the thing that was missing.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Wrap {
	/// The texture tiles, which is what a floor wants.
	#[default]
	Repeat = 0,

	/// The last row and column go on forever, which is what a decal or an
	/// atlas wants: a repeating one bleeds the far edge into the near one.
	Clamp = 1,
}

impl Wrap {
	/// The word each one is written as, in declaration order.
	///
	/// A file's vocabulary and an inspector's drop-down. @ref
	/// [`field::Kind::Word`](super::field::Kind::Word).
	pub const WORDS: &[&str] = &["repeat", "clamp"];

	/// The one at a place in [`WORDS`](Self::WORDS), if there is one.
	///
	/// @param index - the place
	#[must_use]
	pub const fn at(index: u32) -> Option<Self> {
		match index {
			| 0 => Some(Self::Repeat),
			| 1 => Some(Self::Clamp),
			| _ => None,
		}
	}

	/// Where this one is in [`WORDS`](Self::WORDS).
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "the discriminant is the place in the list, by declaration order"
	)]
	pub const fn index(self) -> u32 { self as u32 }

	/// The word this one is written as.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "u32 to usize is lossless on every target this builds for, and try_from is not \
		          available in a const fn"
	)]
	pub const fn word(self) -> &'static str { Self::WORDS[self.index() as usize] }

	/// The number the renderer indexes its sampler table with.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "a fieldless repr(u32) enum casts to its own discriminant, which is exactly 		          the number being asked for"
	)]
	pub const fn code(self) -> u32 { self as u32 }
}

/// How a surface's alpha is read.
///
/// Two things wear the word "transparent" and they are different mechanisms
/// rather than different arithmetic. A *mask* either throws a texel away or
/// keeps it whole, so the geometry goes on writing depth, goes on casting, and
/// is never sorted against anything - which is what a fence, a grate and a leaf
/// want. Blending wants all three of those the other way round: it writes no
/// depth, casts nothing, and has to be drawn after everything solid and in the
/// right order, or it composites over a picture that is not finished yet.
///
/// A mode rather than a flag, because a flag cannot say which of the three is
/// meant, and every renderer that has this at all has between five and seven
/// values of it.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Blend {
	/// The alpha is not read at all and the surface is solid.
	#[default]
	Opaque = 0,

	/// A texel whose alpha is below half is not drawn; every other texel is
	/// drawn exactly as [`Opaque`](Self::Opaque) would draw it.
	///
	/// The threshold lives in the shader rather than on the material, because
	/// moving the picture's own alpha does the same job and a knob nobody can
	/// turn is worse than no knob. It is the same half that alpha to coverage
	/// falls back to where the hardware has none.
	Mask = 1,

	/// The alpha says how much of what is behind the surface still shows
	/// through, which is glass, water and a beam of light.
	///
	/// Three things follow and none of them is optional: the depth buffer is
	/// read and not written, so what is drawn later is not held out by it; the
	/// surface is drawn in a second pass after everything solid, sorted far to
	/// near; and it casts no shadow at all, which is what every engine checked
	/// does with a surface that writes no depth.
	Alpha = 2,
}

impl Blend {
	/// How many rows there are, which is how wide that table is.
	///
	/// @note: a mode added above has to be counted here as well - nothing in
	/// the language can do it, and the test below is what notices.
	pub const COUNT: usize = 3;
	/// The word each mode is written as, in declaration order.
	///
	/// A file's vocabulary and an inspector's drop-down. @ref
	/// [`field::Kind::Word`](super::field::Kind::Word).
	pub const WORDS: &[&str] = &["opaque", "mask", "alpha"];

	/// The mode at a place in [`WORDS`](Self::WORDS), if there is one.
	///
	/// @param index - the place
	#[must_use]
	pub const fn at(index: u32) -> Option<Self> {
		match index {
			| 0 => Some(Self::Opaque),
			| 1 => Some(Self::Mask),
			| 2 => Some(Self::Alpha),
			| _ => None,
		}
	}

	/// Where this mode is in [`WORDS`](Self::WORDS).
	///
	/// The same number [`code`](Self::code) hands back, and deliberately so:
	/// the file's word, the inspector's row and the byte a `.cmesh` carries
	/// are one order, so a mode appended below is one line in three places
	/// instead of a translation table.
	#[must_use]
	pub const fn index(self) -> u32 { self.code() }

	/// The word this mode is written as.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "u32 to usize is lossless on every target this builds for, and try_from is not \
		          available in a const fn"
	)]
	pub const fn word(self) -> &'static str { Self::WORDS[self.index() as usize] }

	/// The row of the renderer's pipeline table this mode is built into.
	///
	/// A match rather than a cast of the discriminant, so that a mode added
	/// above is a compile error here rather than a row nobody built.
	#[must_use]
	pub const fn row(self) -> usize {
		match self {
			| Self::Opaque => 0,
			| Self::Mask => 1,
			| Self::Alpha => 2,
		}
	}

	/// The number a file writes this mode down as.
	///
	/// The same shape [`Wrap::code`] has and for the same reason: a format has
	/// to name a mode with something that does not move when the enum is
	/// reordered.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "a fieldless repr(u32) enum casts to its own discriminant"
	)]
	pub const fn code(self) -> u32 { self as u32 }

	/// The mode a file wrote down, or nothing if this build does not know it.
	///
	/// An `Option` rather than a fallback, because this is a *code* and not a
	/// flag: it says which of several things a record is and has nothing
	/// smaller to fall back to, so a build meeting one it does not know has to
	/// refuse the file rather than guess. The other format that has both draws
	/// the same line in the same place.
	#[must_use]
	pub const fn from_code(code: u32) -> Option<Self> {
		match code {
			| 0 => Some(Self::Opaque),
			| 1 => Some(Self::Mask),
			| 2 => Some(Self::Alpha),
			| _ => None,
		}
	}
}

/// How much alpha a texel needs before [`Blend::Mask`] draws it at all.
///
/// **Written down three times and it has to stay one number**: here, in the
/// scene's shader and in the cascade pass's, because a shader module cannot
/// include another one and neither can read a constant out of Rust. What keeps
/// them level is a test rather than anybody's memory - a fence whose shadow had
/// different holes in it than the fence has is invisible except on a lit scene.
///
/// A constant rather than a number on the material, for the reason written on
/// [`Blend::Mask`]. It is also the exchange format's own default, so a masked
/// material exported from anywhere imports exactly.
pub const MASK_CUTOFF: f32 = 0.5;

/// How rough an unspecified surface is.
///
/// Not zero: a perfectly smooth dielectric is a mirror, which is a strange
/// thing for a material nobody configured to be.
pub const DEFAULT_ROUGHNESS: f32 = 0.8;

/// What a surface is made of.
///
/// Plain data with public fields: a game is expected to build one inline, and
/// there is nothing here that could be put into an invalid state that the
/// shader would not simply clamp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
	/// Linear RGB, multiplied with the albedo texture and with the entity's own
	/// color.
	pub base_color: Vec3,

	/// Zero for a dielectric, one for a metal. The values between are for
	/// blending across a texture, not for describing a real substance.
	pub metallic: f32,

	/// Zero is a mirror, one is chalk.
	pub roughness: f32,

	/// The albedo texture, or [`TextureId::NONE`] for a flat color.
	///
	/// `NONE` samples one white texel, so there is no branch in the shader and
	/// no second pipeline - a flat material is a textured one whose texture
	/// happens to be white.
	pub albedo: TextureId,

	/// The normal map, or [`TextureId::FLAT_NORMAL`] for a surface that is as
	/// flat as its geometry says.
	///
	/// The same trick the albedo plays, with a different constant: the default
	/// is one texel meaning "straight out", so a material with no map goes
	/// through the same shader as one with a map and comes out unchanged.
	/// **Not `NONE`** - `NONE` is the white texel, and white read as a
	/// direction is a diagonal.
	///
	/// The image has to have been compiled as numbers rather than as a color;
	/// @ref `colby_asset::compile::NORMAL_SUFFIX` for how a file says so.
	pub normal: TextureId,

	/// How many times the texture repeats across the mesh's own `0..1`.
	///
	/// Here rather than in the mesh because it is a property of the *surface*:
	/// the same floor quad is one tile of marble or forty of brick depending on
	/// what it is made of, and baking that into the geometry would mean a mesh
	/// per material.
	pub uv_scale: Vec2,

	/// What happens past the edge of both of its textures.
	///
	/// One setting for the pair rather than one each: they are the same surface
	/// under the same unwrap, and a normal map that tiled differently from the
	/// color over it would be a bug with no use.
	pub wrap: Wrap,

	/// How the albedo's alpha is read.
	///
	/// Here rather than on the entity because it is a property of what the
	/// surface is *made of*: the same crate is solid or is a grate depending on
	/// its material, and two entities sharing one material can never disagree
	/// about it. It is also what the renderer already batches by, so a mode
	/// here costs it no second lookup.
	pub blend: Blend,

	/// How much of this surface there is, from nothing to all of it.
	///
	/// Multiplied by the albedo texture's own alpha, so a pane of frosted glass
	/// is a picture with an alpha channel and a material at one, and a whole
	/// wall fading out is a picture with none and a material that moves.
	///
	/// **Read only where [`blend`](Self::blend) is [`Blend::Alpha`]**, and
	/// deliberately: a surface that is solid is solid whatever somebody typed
	/// here, which is what every renderer with an alpha mode does with the
	/// number in its other modes.
	pub opacity: f32,
}

impl Material {
	/// A plain white dielectric.
	pub const DEFAULT: Self = Self {
		base_color: Vec3::ONE,
		metallic: 0.0,
		roughness: DEFAULT_ROUGHNESS,
		albedo: TextureId::NONE,
		normal: TextureId::FLAT_NORMAL,
		uv_scale: Vec2::ONE,
		wrap: Wrap::Repeat,
		blend: Blend::Opaque,
		opacity: 1.0,
	};
	/// Its fields, for an inspector, a reader and a writer. @ref
	/// [`field`](super::field).
	///
	/// **The whole record**, unlike every other table in this crate: a
	/// material has no transform, no handle to itself and no counter the host
	/// keeps, so there is nothing about it that is a moment rather than a
	/// thing. The two textures are references, described here and named by
	/// hand beside the table, exactly as a body's entity is.
	pub const FIELDS: &[Field<Self>] = &[
		field!(Color, "base_color", base_color, "the surface's own color, linear RGB"),
		field!(Float, "metallic", metallic, "nought for a dielectric, one for a metal"),
		field!(Float, "roughness", roughness, "nought is a mirror, one is chalk"),
		field!(Texture, "albedo", albedo, "the color picture, or none for a flat color"),
		field!(Texture, "normal", normal, "the normal map, or none for a flat surface"),
		field!(
			Vec2,
			"uv_scale",
			uv_scale,
			"how many times the textures repeat across the mesh's own nought to one"
		),
		word!(
			"wrap",
			wrap,
			Wrap::WORDS,
			Wrap::at,
			Wrap::index,
			"what happens past the edge of both textures"
		),
		word!(
			"blend",
			blend,
			Blend::WORDS,
			Blend::at,
			Blend::index,
			"how the albedo's alpha is read"
		),
		field!(Float, "opacity", opacity, "how much of the surface there is; alpha mode only"),
	];
	/// What a fluid is drawn as: blue-green, smooth, and half see-through.
	///
	/// [`Blend::Alpha`], so it reads the depth buffer and does not write it,
	/// is drawn after everything solid, sorted far to near, and casts no
	/// shadow - which is what a surface you can see the bottom through has to
	/// do. @ref [`Blend::Alpha`] for the three rules and why none is optional.
	///
	/// Smooth rather than rough, because the one thing a person recognizes
	/// water by at a glance is that the sky is in it.
	pub const WATER: Self = Self {
		base_color: Vec3::new(0.16, 0.38, 0.45),
		roughness: 0.08,
		blend: Blend::Alpha,
		opacity: 0.55,
		..Self::DEFAULT
	};

	/// A material in a color, with nothing else set.
	#[must_use]
	pub const fn colored(base_color: Vec3) -> Self { Self { base_color, ..Self::DEFAULT } }

	/// A material sampling a texture, tinted white.
	#[must_use]
	pub const fn textured(albedo: TextureId) -> Self { Self { albedo, ..Self::DEFAULT } }

	/// The same material, with a normal map over it.
	///
	/// [`TextureId::NONE`] means the flat map rather than the null texture,
	/// which is the one place a handle is rewritten on the way in. The reason
	/// is that `NONE` is the *white* texel and white read as a direction is a
	/// diagonal, so a game that asked for a map the registry does not have yet
	/// would get a surface lit from the wrong side rather than an unmapped one.
	/// Everywhere else a missing handle costs nothing; here it costs the
	/// picture.
	#[must_use]
	pub const fn bumped(self, normal: TextureId) -> Self {
		Self {
			normal: if normal.is_some() {
				normal
			} else {
				TextureId::FLAT_NORMAL
			},
			..self
		}
	}

	/// The same material, with its metallic and roughness set.
	#[must_use]
	pub const fn finished(self, metallic: f32, roughness: f32) -> Self {
		Self { metallic, roughness, ..self }
	}

	/// The same material, with its texture repeated.
	#[must_use]
	pub const fn tiled(self, times: f32) -> Self { Self { uv_scale: Vec2::splat(times), ..self } }

	/// The same material, with its textures held at their edges.
	#[must_use]
	pub const fn clamped(self) -> Self { Self { wrap: Wrap::Clamp, ..self } }

	/// The same material, with the holes in its picture left as holes.
	///
	/// @note: what decides is the *albedo texture's* alpha, so a material with
	/// no picture is masked against one white texel and nothing is ever cut out
	/// of it. That is the same rule [`textured`](Self::textured) already relies
	/// on, read the other way round.
	#[must_use]
	pub const fn masked(self) -> Self { Self { blend: Blend::Mask, ..self } }

	/// The same material, with what is behind it showing through.
	///
	/// @param opacity - how much of the surface there is, one being all of it
	#[must_use]
	pub const fn translucent(self, opacity: f32) -> Self {
		Self { blend: Blend::Alpha, opacity, ..self }
	}
}

impl Default for Material {
	fn default() -> Self { Self::DEFAULT }
}

registry_handle! {
	/// A handle to a material in the world's [`Materials`] registry.
	///
	/// Never removed, so an entity holding one keeps drawing across a reload of
	/// whatever the material is made of.
	MaterialId
}

impl MaterialId {
	/// The material every entity gets before it is given another.
	pub const DEFAULT: Self = Self::new(1);
}

/// One entry of the material registry.
pub type MaterialEntry = Entry<Material>;

/// Every material the renderer knows, addressed by [`MaterialId`].
///
/// Slot zero is [`MaterialId::NONE`] and slot one is
/// [`MaterialId::DEFAULT`]; both are plain white, so an entity that names
/// neither still draws.
#[derive(Clone, Debug)]
pub struct Materials {
	entries: Registry<Material>,
}

impl Materials {
	/// A registry holding the null material and the default one.
	#[must_use]
	pub fn new() -> Self {
		let mut materials = Self {
			entries: Registry::new(Material::DEFAULT),
		};
		materials.insert(DEFAULT_NAME, Material::DEFAULT);
		materials.insert(WATER_NAME, Material::WATER);

		materials
	}

	/// Looks a material up by name.
	///
	/// @return its handle, or [`MaterialId::NONE`] if nothing answers to it
	#[must_use]
	pub fn find(&self, name: &str) -> MaterialId { MaterialId::new(self.entries.find(name)) }

	/// Registers a material under a name, replacing whatever was there.
	///
	/// @return the handle, the same one as last time if the name is known
	pub fn insert(&mut self, name: &str, material: Material) -> MaterialId {
		MaterialId::new(self.entries.insert(name, material))
	}

	/// One material, by handle.
	#[must_use]
	pub fn get(&self, id: MaterialId) -> Option<&Material> {
		self.entries.entry(id.index()).map(Entry::value)
	}

	/// One material, by handle, to change.
	///
	/// Taking this bumps the entry's revision, so a game that tunes a roughness
	/// every frame makes the renderer re-upload every frame. Read it back with
	/// [`get`](Self::get) unless you mean to write.
	pub fn get_mut(&mut self, id: MaterialId) -> Option<&mut Material> {
		self.entries
			.entry_mut(id.index())
			.map(Entry::value_mut)
	}

	/// One entry, by handle, with its revision.
	#[must_use]
	pub fn entry(&self, id: MaterialId) -> Option<&MaterialEntry> {
		self.entries.entry(id.index())
	}

	/// What one is called, or the empty string for a handle nothing answers
	/// to.
	///
	/// The question a panel asks: a material is picked by name in a browser
	/// and shown by handle in an inspector, and this is what closes the loop.
	///
	/// @param id - which material
	#[must_use]
	pub fn name(&self, id: MaterialId) -> &str {
		self.entries
			.entry(id.index())
			.map_or("", Entry::name)
	}

	/// How many materials there are, counting the null one.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Always `false`: slot zero always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// Every material, in slot order.
	pub fn iter(&self) -> impl Iterator<Item = &MaterialEntry> { self.entries.iter() }
}

impl Default for Materials {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::{
		super::field::{Kind, Value},
		*,
	};

	#[test]
	fn the_default_material_is_a_plain_white_dielectric() {
		let material = Material::DEFAULT;

		assert_eq!(material.base_color, Vec3::ONE, "white, so a tint shows through unchanged");
		assert!(material.metallic.abs() < f32::EPSILON, "not a metal");
		assert!(!material.albedo.is_some(), "and no image, which samples as white");
		assert!(
			material.roughness > 0.5,
			"rough rather than a mirror, which is a strange default"
		);
		assert_eq!(material.uv_scale, Vec2::ONE, "and one tile across the mesh");
	}

	#[test]
	fn asking_for_a_normal_map_that_is_not_there_leaves_the_surface_flat() {
		let missing = Material::DEFAULT.bumped(TextureId::NONE);

		assert_eq!(
			missing.normal,
			TextureId::FLAT_NORMAL,
			"the null texture is white, and white read as a direction is a diagonal"
		);
		assert_eq!(
			Material::DEFAULT.bumped(TextureId::new(7)).normal,
			TextureId::new(7),
			"and a handle that names something is left alone"
		);
	}

	#[test]
	fn a_material_nobody_configured_reads_no_alpha_at_all() {
		assert_eq!(
			Material::DEFAULT.blend,
			Blend::Opaque,
			"a surface is solid until something says otherwise"
		);
		assert_eq!(Blend::default(), Blend::Opaque, "and the enum's own default agrees");
	}

	#[test]
	fn a_material_nobody_configured_is_all_there() {
		assert!(
			(Material::DEFAULT.opacity - 1.0).abs() < f32::EPSILON,
			"a surface is whole until something fades it, got {}",
			Material::DEFAULT.opacity
		);
	}

	#[test]
	fn fading_a_material_sets_the_mode_that_reads_the_number() {
		let plain = Material::colored(Vec3::X).finished(0.25, 0.3);
		let faded = plain.translucent(0.4);

		assert_eq!(faded.blend, Blend::Alpha, "an opacity nothing reads would be a lie");
		assert!((faded.opacity - 0.4).abs() < f32::EPSILON, "and the number is set");
		assert_eq!(
			Material {
				blend: Blend::Opaque,
				opacity: 1.0,
				..faded
			},
			plain,
			"and putting both back gives the material it was made from"
		);
	}

	#[test]
	fn masking_a_material_changes_that_and_nothing_else() {
		let plain = Material::textured(TextureId::new(4)).finished(0.25, 0.3);
		let masked = plain.masked();

		assert_eq!(masked.blend, Blend::Mask, "the mode is set");
		assert_eq!(
			Material { blend: Blend::Opaque, ..masked },
			plain,
			"and putting the mode back gives the material it was made from"
		);
	}

	#[test]
	fn a_mode_survives_being_written_down_as_a_number() {
		for mode in [Blend::Opaque, Blend::Mask, Blend::Alpha] {
			assert_eq!(
				Blend::from_code(mode.code()),
				Some(mode),
				"{mode:?} comes back as itself"
			);
		}

		assert_eq!(
			Blend::from_code(u32::try_from(Blend::COUNT).unwrap_or(u32::MAX)),
			None,
			"and a code this build does not know is refused rather than guessed at"
		);
	}

	#[test]
	fn every_mode_has_a_row_of_its_own_and_together_they_fill_the_table() {
		let mut taken = vec![None; Blend::COUNT];

		for mode in [Blend::Opaque, Blend::Mask, Blend::Alpha] {
			let row = mode.row();

			assert!(row < Blend::COUNT, "{mode:?} names row {row}, which is past the table");
			assert_eq!(taken[row], None, "{mode:?} shares row {row} with {:?}", taken[row]);
			taken[row] = Some(mode);
		}

		assert!(
			taken.iter().all(Option::is_some),
			"a row no mode names is a pipeline built for nobody: {taken:?}"
		);
	}

	#[test]
	fn tiling_is_a_property_of_the_material_not_the_mesh() {
		let tiled = Material::DEFAULT.tiled(8.0);

		assert_eq!(tiled.uv_scale, Vec2::splat(8.0), "eight across and eight down");
		assert_eq!(tiled.base_color, Material::DEFAULT.base_color, "and nothing else moved");
	}

	#[test]
	fn the_builders_change_one_thing_each() {
		let colored = Material::colored(Vec3::X);

		assert_eq!(colored.base_color, Vec3::X, "the color is set");
		assert_eq!(colored.albedo, Material::DEFAULT.albedo, "and nothing else moved");

		let finished = colored.finished(1.0, 0.2);

		assert_eq!(finished.base_color, Vec3::X, "the color survives");
		assert!((finished.metallic - 1.0).abs() < f32::EPSILON, "and the metal is set");
		assert!((finished.roughness - 0.2).abs() < f32::EPSILON, "and the roughness");
	}

	#[test]
	fn a_new_registry_has_a_default_at_the_handle_that_names_it() {
		let materials = Materials::new();

		assert_eq!(materials.len(), 3, "the null material, the default one and water");
		assert_eq!(
			materials.find(DEFAULT_NAME),
			MaterialId::DEFAULT,
			"the constant and the name agree"
		);
		assert_eq!(materials.get(MaterialId::DEFAULT), Some(&Material::DEFAULT), "and match");
		assert_eq!(materials.find("stone"), MaterialId::NONE, "nothing else is registered");
	}

	#[test]
	fn every_field_reads_back_what_it_was_written() {
		let mut material = Material::DEFAULT;

		for entry in Material::FIELDS {
			let written = match entry.kind {
				| Kind::Word(words) =>
					Value::Word(u32::try_from(words.len()).expect("a short list") - 1),
				| Kind::Float => Value::Float(0.75),
				| Kind::Color => Value::Color(Vec3::new(0.25, 0.5, 0.75)),
				| Kind::Vec2 => Value::Vec2(Vec2::new(3.0, 4.0)),
				// a reference is described and never written down as text; a
				// value of one still has to survive the table.
				| Kind::Texture => Value::Texture(TextureId::new(7)),
				| kind => panic!("a material has no field of {kind:?}"),
			};

			assert!(
				entry.set(&mut material, written.clone()),
				"{} takes its own kind",
				entry.name
			);
			assert_eq!(entry.get(&material), written, "{} hands back what it took", entry.name);
		}

		assert_eq!(material.wrap, Wrap::Clamp, "the last word is the last one");
		assert_eq!(material.blend, Blend::Alpha, "for both of them");
	}

	#[test]
	fn the_two_word_lists_are_their_enumerations_in_order() {
		for (index, word) in Wrap::WORDS.iter().enumerate() {
			let index = u32::try_from(index).expect("two of them");
			let held = Wrap::at(index).expect("every place has one");

			assert_eq!(held.index(), index, "{word} is at {index}");
			assert_eq!(held.word(), *word, "and is written as itself");
		}

		for (index, word) in Blend::WORDS.iter().enumerate() {
			let index = u32::try_from(index).expect("three of them");
			let held = Blend::at(index).expect("every place has a mode");

			assert_eq!(held.index(), index, "{word} is at {index}");
			assert_eq!(held.word(), *word, "and is written as itself");
			// the file's word, the inspector's row and the byte a `.cmesh`
			// carries are deliberately one order
			assert_eq!(held.code(), index, "and the code is the same number");
		}

		assert_eq!(Blend::WORDS.len(), Blend::COUNT, "every mode has a word");
		assert!(Wrap::at(2).is_none() && Blend::at(3).is_none(), "one past the end is nothing");
	}

	#[test]
	fn water_is_registered_too_and_is_the_one_built_in_surface_that_is_not_solid() {
		let materials = Materials::new();
		let found = materials.find(WATER_NAME);

		assert!(found.is_some(), "a world can name it without compiling anything");
		assert_eq!(materials.get(found), Some(&Material::WATER), "and it is the constant");
		assert_eq!(
			Material::WATER.blend,
			Blend::Alpha,
			"a fluid that came out opaque would hide what it is there to show"
		);
		assert_eq!(Material::DEFAULT.blend, Blend::Opaque, "while everything else is solid");
	}

	#[test]
	fn the_null_material_still_draws() {
		let materials = Materials::new();

		assert_eq!(
			materials.get(MaterialId::NONE),
			Some(&Material::DEFAULT),
			"an entity that names no material is white, not invisible"
		);
	}

	#[test]
	fn replacing_a_material_keeps_its_handle_and_bumps_its_revision() {
		let mut materials = Materials::new();
		let first = materials.insert("stone", Material::colored(Vec3::X));
		let second = materials.insert("stone", Material::colored(Vec3::Y));

		assert_eq!(first, second, "the handle survives");
		assert_eq!(
			materials
				.entry(first)
				.map(MaterialEntry::revision),
			Some(1),
			"and the revision moved"
		);
		assert_eq!(
			materials
				.get(first)
				.map(|material| material.base_color),
			Some(Vec3::Y),
			"and the value really is the new one"
		);
	}

	#[test]
	fn writing_through_a_handle_is_a_change_the_renderer_will_see() {
		let mut materials = Materials::new();
		let id = materials.insert("stone", Material::DEFAULT);

		materials
			.get_mut(id)
			.expect("the material is there")
			.roughness = 0.1;

		assert_eq!(
			materials.entry(id).map(MaterialEntry::revision),
			Some(1),
			"a mutable borrow counts, because there is no way to find out afterwards"
		);
		assert!(
			materials
				.get(id)
				.is_some_and(|material| (material.roughness - 0.1).abs() < f32::EPSILON),
			"and the write landed"
		);
	}
}
