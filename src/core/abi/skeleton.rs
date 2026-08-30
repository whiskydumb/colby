//! Skeletons: the bones a skinned mesh is moved by, and the host's registry of
//! them.
//!
//! A skeleton holds no geometry. It is a list of bones, each naming its parent,
//! carrying the local transform it has when nothing is animating it, and
//! carrying the matrix that takes a vertex from the mesh's own space into that
//! bone's. What a mesh does with it is a bone index and a weight per vertex;
//! what moves it is a pose. Neither is here.
//!
//! **They are bones rather than joints**, and the word is not decoration:
//! [`joint`](super::joint) in this engine is a physical constraint between two
//! bodies, and a ragdoll is going to have both kinds in one function. Every
//! large engine calls these bones too; the exchange format calls them joints
//! and is the odd one out.
//!
//! **A parent always comes before its child.** That one rule is what lets a
//! pose be resolved into world matrices by walking the list once, forwards,
//! with no recursion and no scratch: by the time a bone is reached its parent
//! is already done. The exchange format guarantees nothing of the sort, so the
//! importer sorts and the loader refuses a file that is not sorted - @ref
//! [`SkeletonData::is_ordered`]. Checking it here rather than making it
//! impossible to build a bad one is deliberate: this is plain data read out of
//! a file, and a fallible constructor would only move the check to the same
//! place.
//!
//! The registry follows the rules every other asset table follows - slot zero
//! is nothing, a name keeps its slot for the life of the process, and
//! reloading rewrites the entry a handle already points at. @ref
//! [`registry`](super::registry).

use super::{
	entity::Transform,
	registry::{Entry, Registry},
};
use crate::{glam::Mat4, registry_handle};

/// The most bones one skeleton may have.
///
/// A humanoid rig is fifty to a hundred bones, a crowd character twenty to
/// thirty-five, and a film-grade one with fingers and a face around two
/// hundred. This is above all of those and is a bound on a file rather than a
/// budget: the matrices go to the GPU in a storage buffer, so nothing about
/// the number is expensive until there are a great many characters.
pub const MAX_BONES: usize = 256;

/// What [`Bone::parent`] holds when a bone is the top of its tree.
///
/// A sentinel rather than an `Option` because this is written into a file as a
/// number, and because [`SkeletonData::is_ordered`] then has one rule instead
/// of two.
pub const NO_PARENT: u16 = u16::MAX;

registry_handle! {
	/// A handle to a skeleton in the world's [`Skeletons`] registry.
	///
	/// No generation, like every other asset handle: entries are never
	/// removed, so reloading a skeleton rewrites the entry the id already
	/// points at and a game holding one does not re-resolve.
	SkeletonId
}

/// One bone.
#[derive(Clone, Debug, PartialEq)]
pub struct Bone {
	/// What the file called it.
	///
	/// The only way an animation finds a bone: a clip names the bones it
	/// turns, and the two are matched by name once, when the clip is loaded.
	/// Indices could not do it - a clip and a skeleton are separate files and
	/// either may be rebuilt without the other.
	pub name: String,

	/// Which bone this one hangs off, or [`NO_PARENT`].
	///
	/// Always less than this bone's own index, @ref
	/// [`SkeletonData::is_ordered`].
	pub parent: u16,

	/// The mesh's own space into this bone's, as the mesh was authored.
	///
	/// The other half of skinning: a vertex is carried into bone space by
	/// this, moved by wherever the bone is now, and the results are mixed by
	/// the vertex's weights. So the matrix a pose hands the GPU is
	/// `model_of(bone) * inverse_bind`, and in the pose the mesh was authored
	/// in that product is the identity and the mesh does not move.
	pub inverse_bind: Mat4,

	/// Where this bone sits relative to its parent when nothing has moved it.
	///
	/// Needed because a clip does not have to name every bone: whatever it
	/// leaves alone stays here. It is a local transform, unlike
	/// [`inverse_bind`](Self::inverse_bind), which is against the whole mesh.
	pub rest: Transform,
}

impl Default for Bone {
	fn default() -> Self {
		Self {
			name: String::new(),
			parent: NO_PARENT,
			inverse_bind: Mat4::IDENTITY,
			rest: Transform::IDENTITY,
		}
	}
}

/// A skeleton as the world holds it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SkeletonData {
	/// Every bone, parents before children.
	pub bones: Vec<Bone>,
}

impl SkeletonData {
	/// Looks a bone up by name.
	///
	/// A linear scan, like every other name lookup in the ABI, and for the
	/// same reason: it happens once when something is loaded, and after that
	/// everything holds an index.
	///
	/// @param name - the name the file gave it
	/// @return its index, or `None` if no bone answers to it
	#[must_use]
	pub fn find(&self, name: &str) -> Option<u16> {
		if name.is_empty() {
			return None;
		}

		self.bones
			.iter()
			.position(|bone| bone.name == name)
			.and_then(|index| u16::try_from(index).ok())
	}

	/// Whether every bone's parent comes before it.
	///
	/// The invariant everything downstream stands on. It also proves every
	/// parent is a real index, which is the same statement: a parent below a
	/// bone's own index is below the length as well, so a file that passes
	/// this cannot point at a bone that is not there.
	///
	/// @return `true` if the list can be resolved in one forward pass
	#[must_use]
	pub fn is_ordered(&self) -> bool {
		self.bones
			.iter()
			.enumerate()
			.all(|(index, bone)| bone.parent == NO_PARENT || usize::from(bone.parent) < index)
	}

	/// How many bones there are.
	#[must_use]
	pub fn len(&self) -> usize { self.bones.len() }

	/// Whether it has no bones at all, which is what the null skeleton is.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.bones.is_empty() }
}

/// Every bone's model-space matrix with nothing animating it.
///
/// The rest pose composed all the way down, which is the shape a mesh was
/// authored in. One forward pass, no recursion and no scratch, for the reason
/// everything else here manages the same: a parent is always written first.
///
/// It is what a ragdoll's layout is worked out from, and it is the honest way
/// to answer "which way does this model face" - compose the rests and look at
/// where the toe sits relative to the ankle, rather than guessing at a sign.
///
/// @param bones - the skeleton, parents before children
/// @param out - cleared and filled, one matrix a bone
pub fn rests(bones: &[Bone], out: &mut Vec<Mat4>) {
	out.clear();
	out.reserve(bones.len());

	for bone in bones {
		let local = bone.rest.matrix();
		let model = if bone.parent == NO_PARENT {
			local
		} else {
			out.get(usize::from(bone.parent))
				.copied()
				.unwrap_or(Mat4::IDENTITY)
				* local
		};

		out.push(model);
	}
}

/// One entry of the skeleton registry.
pub type Skeleton = Entry<SkeletonData>;

/// Every skeleton the host has loaded, addressed by [`SkeletonId`].
///
/// Slot zero is [`SkeletonId::NONE`] and has no bones, so a game asking for a
/// skeleton that is not there gets an empty list rather than failing.
#[derive(Clone, Debug)]
pub struct Skeletons {
	entries: Registry<SkeletonData>,
}

impl Skeletons {
	/// A registry holding the null skeleton and nothing else.
	#[must_use]
	pub fn new() -> Self {
		Self {
			entries: Registry::new(SkeletonData::default()),
		}
	}

	/// Looks a skeleton up by name.
	///
	/// @param name - the name it was registered under, e.g. `models/hero/rig`
	/// @return its handle, or [`SkeletonId::NONE`] if nothing answers to it
	#[must_use]
	pub fn find(&self, name: &str) -> SkeletonId { SkeletonId::new(self.entries.find(name)) }

	/// Registers a skeleton under a name, replacing whatever was there.
	///
	/// @param name - what the game will ask for
	/// @param data - its bones
	/// @return the handle, the same one as last time if the name is known
	pub fn insert(&mut self, name: &str, data: SkeletonData) -> SkeletonId {
		SkeletonId::new(self.entries.insert(name, data))
	}

	/// One skeleton, by handle.
	#[must_use]
	pub fn get(&self, id: SkeletonId) -> Option<&Skeleton> { self.entries.entry(id.index()) }

	/// The bones of one skeleton, by handle.
	///
	/// The call every consumer makes, so it is here rather than in each of
	/// them. A handle to nothing gives no bones, which is what makes posing
	/// something whose skeleton failed to load a loop over an empty list.
	#[must_use]
	pub fn bones(&self, id: SkeletonId) -> &[Bone] {
		self.get(id)
			.map_or(&[], |skeleton| skeleton.value().bones.as_slice())
	}

	/// How many skeletons there are, counting the null one.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Always `false`: slot zero always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// Every skeleton, in slot order, starting with the null one.
	pub fn iter(&self) -> impl Iterator<Item = &Skeleton> { self.entries.iter() }
}

impl Default for Skeletons {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::glam::{Quat, Vec3};

	/// A bone hanging off another one, a stride along `x` from it.
	fn bone(name: &str, parent: u16, along: f32) -> Bone {
		Bone {
			name: name.to_owned(),
			parent,
			rest: Transform::at(Vec3::new(along, 0.0, 0.0)),
			..Bone::default()
		}
	}

	/// Three bones in a row: a shoulder, an elbow below it, a wrist below that.
	///
	/// The inverse binds are worked out from the rests rather than written by
	/// hand, which is what an importer does and is the only way they can be
	/// right: an inverse bind is against the whole mesh, so it has to undo
	/// every rest above the bone as well as its own.
	fn arm() -> SkeletonData {
		let mut data = SkeletonData {
			bones: vec![
				bone("shoulder", NO_PARENT, 0.0),
				bone("elbow", 0, 1.0),
				bone("wrist", 1, 2.0),
			],
		};
		let mut model: Vec<Mat4> = Vec::with_capacity(data.bones.len());

		for bone in &data.bones {
			let local = bone.rest.matrix();
			let world = if bone.parent == NO_PARENT {
				local
			} else {
				model[usize::from(bone.parent)] * local
			};

			model.push(world);
		}

		for (bone, world) in data.bones.iter_mut().zip(&model) {
			bone.inverse_bind = world.inverse();
		}

		data
	}

	#[test]
	fn a_bone_is_found_by_the_name_the_file_gave_it() {
		let arm = arm();

		assert_eq!(arm.find("shoulder"), Some(0));
		assert_eq!(arm.find("wrist"), Some(2));
		assert_eq!(arm.find("tail"), None, "and one nobody named is not there");
		assert_eq!(arm.find(""), None, "the empty name is not a name");
	}

	#[test]
	fn a_bone_with_no_name_is_not_found_by_asking_for_nothing() {
		let unnamed = SkeletonData {
			bones: vec![bone("", NO_PARENT, 0.0), bone("elbow", 0, 1.0)],
		};

		assert_eq!(
			unnamed.find(""),
			None,
			"a name is optional in the exchange format, so a bone really can have none, and \
			 asking for nothing must not land on the first one that also has nothing"
		);
		assert_eq!(unnamed.find("elbow"), Some(1), "while a real name still answers");
	}

	#[test]
	fn a_parent_before_its_child_is_what_ordered_means() {
		assert!(arm().is_ordered(), "the shoulder is written before the elbow");

		let mut backwards = arm();
		backwards.bones.swap(0, 1);

		assert!(!backwards.is_ordered(), "and the elbow written before its own parent is not");
	}

	#[test]
	fn a_bone_that_is_its_own_parent_is_not_ordered_either() {
		let mut looped = arm();
		looped.bones[1].parent = 1;

		assert!(!looped.is_ordered(), "a parent has to be strictly before its child");
	}

	#[test]
	fn a_parent_that_is_not_a_bone_at_all_fails_the_same_check() {
		let mut wild = arm();
		wild.bones[2].parent = 40;

		assert!(
			!wild.is_ordered(),
			"an index past the end is above its own index, so the one rule catches it"
		);
	}

	#[test]
	fn roots_are_allowed_anywhere_and_there_may_be_several() {
		let two = SkeletonData {
			bones: vec![
				bone("first", NO_PARENT, 0.0),
				bone("second", NO_PARENT, 0.0),
				bone("under the second", 1, 1.0),
			],
		};

		assert!(two.is_ordered(), "nothing says a skeleton is one tree");
	}

	#[test]
	fn a_skeleton_comes_back_by_the_name_it_went_in_under() {
		let mut skeletons = Skeletons::new();
		let id = skeletons.insert("models/hero/rig", arm());

		assert_eq!(skeletons.find("models/hero/rig"), id);
		assert_eq!(skeletons.bones(id).len(), 3);
		assert_eq!(skeletons.bones(id)[1].name, "elbow");
	}

	#[test]
	fn asking_for_a_skeleton_nobody_registered_has_no_bones() {
		let skeletons = Skeletons::new();

		assert_eq!(skeletons.find("models/nothing"), SkeletonId::NONE);
		assert!(skeletons.bones(SkeletonId::NONE).is_empty());
		assert!(skeletons.bones(SkeletonId::new(77)).is_empty(), "nor does a wild one");
		assert!(
			skeletons
				.get(SkeletonId::NONE)
				.is_some_and(|null| null.value().is_empty()),
			"the null entry is there and is empty, which is not the same as absent"
		);
	}

	#[test]
	fn reloading_a_skeleton_keeps_the_handle_a_game_is_holding() {
		let mut skeletons = Skeletons::new();
		let id = skeletons.insert("models/hero/rig", arm());
		let again = skeletons.insert("models/hero/rig", SkeletonData::default());

		assert_eq!(again, id, "the same slot");
		assert!(skeletons.bones(id).is_empty(), "holding what arrived second");
		assert_eq!(
			skeletons.get(id).map(Skeleton::revision),
			Some(1),
			"and saying that it moved"
		);
	}

	#[test]
	fn the_bind_pose_is_the_one_where_nothing_moves() {
		// what a pose will do per bone, done by hand: walk the rests down from
		// the root, then undo the bind. The identity coming out of every one
		// of them is the whole definition of an inverse bind matrix, and it is
		// what makes an unanimated character stand in the shape it was drawn
		// in rather than folded up at the origin.
		let arm = arm();
		let mut model: Vec<Mat4> = Vec::with_capacity(arm.len());

		for bone in &arm.bones {
			let local = bone.rest.matrix();
			let world = if bone.parent == NO_PARENT {
				local
			} else {
				model[usize::from(bone.parent)] * local
			};

			model.push(world);
		}

		for (index, bone) in arm.bones.iter().enumerate() {
			let skinning = model[index] * bone.inverse_bind;

			assert!(
				skinning.abs_diff_eq(Mat4::IDENTITY, 1.0e-5),
				"{} leaves its vertices where they were, got {skinning:?}",
				bone.name
			);
		}
	}

	#[test]
	fn a_rest_is_composed_all_the_way_down_rather_than_read_off_the_bone() {
		// a quarter turn at the shoulder and a unit along x under it, so an
		// implementation that forgot the parent would put the hand at (1, 0, 0)
		// and the right one puts it a unit up from the shoulder.
		let bones = vec![
			Bone {
				name: "shoulder".to_owned(),
				parent: NO_PARENT,
				inverse_bind: Mat4::IDENTITY,
				rest: Transform {
					position: Vec3::new(0.0, 2.0, 0.0),
					rotation: Quat::from_rotation_z(core::f32::consts::FRAC_PI_2),
					scale: Vec3::ONE,
				},
			},
			Bone {
				name: "hand".to_owned(),
				parent: 0,
				inverse_bind: Mat4::IDENTITY,
				rest: Transform::at(Vec3::X),
			},
		];
		let mut at = Vec::new();

		rests(&bones, &mut at);

		assert_eq!(at.len(), 2, "one matrix a bone");
		assert!(
			at[0]
				.w_axis
				.truncate()
				.abs_diff_eq(Vec3::new(0.0, 2.0, 0.0), 1e-5),
			"the root is where its own rest puts it"
		);
		assert!(
			at[1]
				.w_axis
				.truncate()
				.abs_diff_eq(Vec3::new(0.0, 3.0, 0.0), 1e-5),
			"and the child is carried by its parent's turn, got {}",
			at[1].w_axis.truncate()
		);
	}

	#[test]
	fn rests_clears_what_it_is_given_rather_than_appending_to_it() {
		let mut at = vec![Mat4::ZERO; 5];

		rests(&arm().bones, &mut at);

		assert_eq!(at.len(), arm().bones.len(), "the buffer holds this skeleton and no other");
	}

	#[test]
	fn a_skeleton_of_no_bones_rests_at_nothing() {
		let mut at = vec![Mat4::ZERO];

		rests(&[], &mut at);

		assert!(at.is_empty(), "and the buffer is emptied rather than left as it was");
	}
}
