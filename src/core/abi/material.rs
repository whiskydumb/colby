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
	registry::{Entry, Registry},
	texture::TextureId,
};
use crate::{
	glam::{Vec2, Vec3},
	registry_handle,
};

/// The name the always-present default material is registered under.
pub const DEFAULT_NAME: &str = "default";

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

	/// How many times the texture repeats across the mesh's own `0..1`.
	///
	/// Here rather than in the mesh because it is a property of the *surface*:
	/// the same floor quad is one tile of marble or forty of brick depending on
	/// what it is made of, and baking that into the geometry would mean a mesh
	/// per material.
	pub uv_scale: Vec2,
}

impl Material {
	/// A plain white dielectric.
	pub const DEFAULT: Self = Self {
		base_color: Vec3::ONE,
		metallic: 0.0,
		roughness: DEFAULT_ROUGHNESS,
		albedo: TextureId::NONE,
		uv_scale: Vec2::ONE,
	};

	/// A material in a color, with nothing else set.
	#[must_use]
	pub const fn colored(base_color: Vec3) -> Self { Self { base_color, ..Self::DEFAULT } }

	/// A material sampling a texture, tinted white.
	#[must_use]
	pub const fn textured(albedo: TextureId) -> Self { Self { albedo, ..Self::DEFAULT } }

	/// The same material, with its metallic and roughness set.
	#[must_use]
	pub const fn finished(self, metallic: f32, roughness: f32) -> Self {
		Self { metallic, roughness, ..self }
	}

	/// The same material, with its texture repeated.
	#[must_use]
	pub const fn tiled(self, times: f32) -> Self { Self { uv_scale: Vec2::splat(times), ..self } }
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
	use super::*;

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

		assert_eq!(materials.len(), 2, "the null material and the default one");
		assert_eq!(
			materials.find(DEFAULT_NAME),
			MaterialId::DEFAULT,
			"the constant and the name agree"
		);
		assert_eq!(materials.get(MaterialId::DEFAULT), Some(&Material::DEFAULT), "and match");
		assert_eq!(materials.find("stone"), MaterialId::NONE, "nothing else is registered");
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
