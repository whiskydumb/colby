//! Models: a piece of a world that arrived through one door, and the host's
//! registry of them.
//!
//! A model is what a `.gltf` or a `.glb` becomes. It holds no geometry and no
//! pixels: those are meshes and textures in the tables beside this one,
//! registered under their own names by the same loader that registers anything
//! a person made by hand. What is here is the part nothing else could carry -
//! **a list of what stands where**.
//!
//! **Every handle in a placement is already resolved.** The compiler wrote
//! names and the host looked them up, so a game reads a [`MeshId`] and a
//! [`MaterialId`] and never sees a string. A name that had not loaded yet was
//! *reserved* on the way through, which is what makes the order the asset tree
//! happens to be walked in stop mattering.
//!
//! **A placement is world space and there is no tree.** The importer folded
//! glTF's node hierarchy flat, because an entity in this engine has no parent
//! to hang a local transform on. So spawning a model is a loop:
//!
//! ```text
//!   let lamp = world.models.find("models/lamp");
//!
//!   for placement in world.models.placements(lamp) {
//!       let entity = world.entities.spawn();
//!       world.entities.set_transform(entity, placement.transform);
//!       world.entities.set_renderable(entity, Renderable::of(..));
//!   }
//! ```
//!
//! That loop is the whole surface, and it is deliberately a loop rather than a
//! call: what a game does with a placement - whether it gives it a body, a
//! name, a piece of gameplay state - is the game's, and a `spawn_model` that
//! guessed would be wrong for every game that wanted something else.

use super::{
	entity::Transform,
	material::MaterialId,
	mesh::MeshId,
	registry::{Entry, Registry},
};
use crate::registry_handle;

registry_handle! {
	/// A handle to a model in the world's [`Models`] registry.
	///
	/// No generation, like every other asset handle: entries are never removed,
	/// so reloading a model rewrites the entry the id already points at and a
	/// game holding one does not re-resolve.
	ModelId
}

/// One piece of a model, and where it stands.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Placement {
	/// What the file called this piece. For a game that wants to find one in
	/// particular rather than spawning all of them.
	pub name: String,

	/// The geometry that stands here.
	pub mesh: MeshId,

	/// What it is made of, or [`MaterialId::DEFAULT`] when the file said
	/// nothing.
	pub material: MaterialId,

	/// Where it stands, with the whole tree above it already worked in.
	pub transform: Transform,
}

/// A model as the world holds it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelData {
	/// Every piece of it, in the order the file listed them.
	pub placements: Vec<Placement>,
}

/// One entry of the model registry.
pub type Model = Entry<ModelData>;

/// Every model the host has loaded, addressed by [`ModelId`].
///
/// Slot zero is [`ModelId::NONE`] and stands nothing anywhere, so a game asking
/// for a model that is not there spawns nothing rather than failing.
#[derive(Clone, Debug)]
pub struct Models {
	entries: Registry<ModelData>,
}

impl Models {
	/// A registry holding the null model and nothing else.
	#[must_use]
	pub fn new() -> Self {
		Self {
			entries: Registry::new(ModelData::default()),
		}
	}

	/// Looks a model up by name.
	///
	/// @param name - the name it was registered under, e.g. `models/lamp`
	/// @return its handle, or [`ModelId::NONE`] if nothing answers to it
	#[must_use]
	pub fn find(&self, name: &str) -> ModelId { ModelId::new(self.entries.find(name)) }

	/// Registers a model under a name, replacing whatever was there.
	///
	/// @param name - what the game will ask for
	/// @param data - what stands where
	/// @return the handle, the same one as last time if the name is known
	pub fn insert(&mut self, name: &str, data: ModelData) -> ModelId {
		ModelId::new(self.entries.insert(name, data))
	}

	/// One model, by handle.
	#[must_use]
	pub fn get(&self, id: ModelId) -> Option<&Model> { self.entries.entry(id.index()) }

	/// What one model stands, by handle.
	///
	/// The one call every consumer makes, so it is here rather than in each of
	/// them. A handle to nothing gives nothing, which is what makes spawning a
	/// model that failed to load a loop over an empty list.
	#[must_use]
	pub fn placements(&self, id: ModelId) -> &[Placement] {
		self.get(id)
			.map_or(&[], |model| model.value().placements.as_slice())
	}

	/// How many models there are, counting the null one.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Always `false`: slot zero always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// Every model, in slot order, starting with the null one.
	pub fn iter(&self) -> impl Iterator<Item = &Model> { self.entries.iter() }
}

impl Default for Models {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::glam::Vec3;

	/// A model with two pieces in it.
	fn lamp() -> ModelData {
		ModelData {
			placements: vec![
				Placement {
					name: "shade".to_owned(),
					mesh: MeshId::new(4),
					material: MaterialId::new(2),
					transform: Transform::at(Vec3::Y),
				},
				Placement {
					name: "stem".to_owned(),
					mesh: MeshId::new(5),
					material: MaterialId::DEFAULT,
					transform: Transform::IDENTITY,
				},
			],
		}
	}

	#[test]
	fn a_model_comes_back_by_the_name_it_went_in_under() {
		let mut models = Models::new();
		let id = models.insert("models/lamp", lamp());

		assert_eq!(models.find("models/lamp"), id);
		assert_eq!(models.placements(id).len(), 2);
		assert_eq!(models.placements(id)[0].name, "shade");
	}

	#[test]
	fn asking_for_a_model_nobody_registered_stands_nothing_anywhere() {
		let models = Models::new();

		assert_eq!(models.find("models/nothing"), ModelId::NONE);
		assert!(models.placements(ModelId::NONE).is_empty());
		assert!(models.placements(ModelId::new(77)).is_empty(), "and neither does a wild one");
	}

	#[test]
	fn reloading_a_model_keeps_the_handle_a_game_is_holding() {
		let mut models = Models::new();
		let id = models.insert("models/lamp", lamp());
		let before = models
			.get(id)
			.map(Model::revision)
			.unwrap_or_default();

		let again = models.insert("models/lamp", ModelData::default());

		assert_eq!(again, id, "the same slot");
		assert!(models.placements(id).is_empty(), "with new contents");
		assert!(
			models
				.get(id)
				.is_some_and(|model| model.revision() > before),
			"and a revision that says so"
		);
	}

	#[test]
	fn slot_zero_is_the_one_that_stands_nothing() {
		let models = Models::new();

		assert_eq!(models.len(), 1);
		assert!(!models.is_empty(), "there is always the null model");
		assert_eq!(models.iter().count(), 1);
	}
}
