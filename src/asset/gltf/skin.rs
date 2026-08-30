//! Skins: the bones a file's meshes are moved by, turned into skeletons.
//!
//! The exchange format has no skeleton object. What it has is a *skin*: a flat
//! list of node indices called joints, an optional matrix per joint, and the
//! rule that the hierarchy is simply the node hierarchy - so the tree has to be
//! read back out of the nodes' `children`, which is the whole of this module.
//!
//! Three things have to happen on the way, and each of them is a place the
//! format and colby disagree:
//!
//! - **the list is sorted so a parent comes before its child.** Nothing in the
//!   format says it will be, and everything downstream walks a skeleton
//!   forwards exactly once. @ref
//!   [`SkeletonData::is_ordered`](colby_core::abi::skeleton::SkeletonData::is_ordered).
//! - **a joint whose parent node is not itself a joint keeps that node's
//!   transform anyway.** The format allows a skin to name a grandparent and a
//!   grandchild and skip what is between them; what is between them still moves
//!   the bone, so the skipped transforms are folded into the child's rest.
//! - **the sort renumbers everything**, so [`Skin::slots`] says where each of
//!   the file's joint slots ended up. A vertex's bone indices address the
//!   file's order and have to be carried over.
//!
//! Everything here reports rather than refuses. A skin that cannot be read
//! becomes a skeleton with no bones, which leaves its meshes unskinned and
//! standing where they were, and says so once in a warning.

use colby_core::{
	abi::{
		Transform,
		skeleton::{Bone, MAX_BONES, NO_PARENT, SkeletonData},
	},
	glam::Mat4,
};

use super::{Gltf, geometry::local, tidy, unique};
use crate::json::Value;

/// What one glTF skin becomes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Skin {
	/// What it registers under, inside the model's own name.
	pub name: String,

	/// Its bones, parents before children. Empty if it could not be read.
	pub data: SkeletonData,

	/// Where each of the file's joint slots ended up after the sort.
	///
	/// A vertex's `JOINTS_0` indexes the file's `skin.joints`; this carries
	/// that over to the bone it became.
	pub slots: Vec<u16>,
}

/// Every skin of one file, in the file's own order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Skins {
	/// One per entry of the document's `skins`, so a node's `skin` indexes it
	/// directly. A skin that could not be read is here with no bones in it.
	pub skins: Vec<Skin>,

	/// What could not be used. Not a failure.
	pub warnings: Vec<String>,
}

/// Reads every skin a file declares.
///
/// @param file - the document with its buffers, from [`Gltf::open`]
/// @return one skeleton per skin, and what could not be read
#[must_use]
pub(super) fn read(file: &Gltf) -> Skins {
	let declared = file.table("skins").len();

	if declared == 0 {
		return Skins::default();
	}

	let parents = parents(file);
	let mut taken = Vec::with_capacity(declared);
	let mut out = Skins::default();

	for index in 0..declared {
		let skin = one(file, index, &parents, &mut taken, &mut out.warnings);

		out.skins.push(skin);
	}

	out
}

/// Which node each node hangs off, over the whole document.
///
/// Worked out once for every skin, because `children` points downwards and
/// every question here is asked upwards.
fn parents(file: &Gltf) -> Vec<Option<usize>> {
	let nodes = file.table("nodes");
	let mut out = vec![None; nodes.len()];

	for (index, node) in nodes.iter().enumerate() {
		for child in node
			.get("children")
			.map_or(&[][..], Value::as_array)
			.iter()
			.filter_map(Value::as_usize)
		{
			if let Some(slot) = out.get_mut(child) {
				*slot = Some(index);
			}
		}
	}

	out
}

/// Reads one skin, or says why it has no bones.
fn one(
	file: &Gltf,
	index: usize,
	parents: &[Option<usize>],
	taken: &mut Vec<String>,
	warnings: &mut Vec<String>,
) -> Skin {
	let entry = file.table("skins").get(index).cloned();
	let written = entry
		.as_ref()
		.and_then(|skin| skin.get("name"))
		.and_then(Value::as_str)
		.unwrap_or("");
	let mut base = tidy(written);

	if base.is_empty() {
		base = format!("skin{index}");
	}

	let name = unique(taken, &base);
	let joints: Vec<usize> = entry
		.as_ref()
		.and_then(|skin| skin.get("joints"))
		.map_or(&[][..], Value::as_array)
		.iter()
		.filter_map(Value::as_usize)
		.collect();

	if joints.is_empty() {
		warnings.push(format!("skin {index} names no joints, and moves nothing"));

		return Skin { name, ..Skin::default() };
	}

	if joints.len() > MAX_BONES {
		warnings.push(format!(
			"skin {index} has {} joints, past the {MAX_BONES} a skeleton may hold, and is left \
			 out",
			joints.len()
		));

		return Skin { name, ..Skin::default() };
	}

	let Some(where_of) = slots_of(index, &joints, file.table("nodes").len(), warnings) else {
		return Skin { name, ..Skin::default() };
	};
	let binds = binds(file, entry.as_ref(), index, joints.len(), warnings);
	let raw = raw_bones(file, &joints, &where_of, parents, &binds);
	let (data, slots) = sorted(&raw, index, warnings);

	Skin { name, data, slots }
}

/// Which joint slot each node is, refusing a file that names one twice.
fn slots_of(
	index: usize,
	joints: &[usize],
	nodes: usize,
	warnings: &mut Vec<String>,
) -> Option<Vec<Option<usize>>> {
	let mut out = vec![None; nodes];

	for (slot, node) in joints.iter().enumerate() {
		let Some(cell) = out.get_mut(*node) else {
			warnings.push(format!(
				"skin {index} names node {node} as a joint, and there is no such node"
			));

			return None;
		};

		if cell.is_some() {
			warnings.push(format!("skin {index} names node {node} twice, and is left out"));

			return None;
		}

		*cell = Some(slot);
	}

	Some(out)
}

/// One inverse bind matrix per joint, defaulting to the identity.
///
/// The specification's own default: a skin with no `inverseBindMatrices` is one
/// whose mesh is already in every bone's space.
fn binds(
	file: &Gltf,
	entry: Option<&Value>,
	index: usize,
	joints: usize,
	warnings: &mut Vec<String>,
) -> Vec<Mat4> {
	let mut out = vec![Mat4::IDENTITY; joints];
	let Some(accessor) = entry
		.and_then(|skin| skin.get("inverseBindMatrices"))
		.and_then(Value::as_usize)
	else {
		return out;
	};

	let read = match file.floats(accessor) {
		| Ok(read) => read,
		| Err(error) => {
			warnings.push(format!(
				"skin {index} has bind matrices that could not be read ({error}), and is placed \
				 as though it had none"
			));

			return out;
		},
	};

	if read.lanes() != 16 || read.rows() < joints {
		warnings.push(format!(
			"skin {index} has {} bind matrices of {} numbers for {joints} joints, and is placed \
			 as though it had none",
			read.rows(),
			read.lanes()
		));

		return out;
	}

	for (slot, matrix) in out.iter_mut().enumerate() {
		let mut cells = [0.0_f32; 16];

		for (cell, value) in cells.iter_mut().zip(read.row(slot)) {
			*cell = *value;
		}

		*matrix = Mat4::from_cols_array(&cells);
	}

	out
}

/// One bone per joint slot, in the file's order, before the sort.
///
/// Where the two things the format leaves to the reader happen: a bone's
/// parent is its nearest ancestor that is also a joint, and every node skipped
/// on the way up is folded into its rest, because those nodes move it too.
fn raw_bones(
	file: &Gltf,
	joints: &[usize],
	where_of: &[Option<usize>],
	parents: &[Option<usize>],
	binds: &[Mat4],
) -> Vec<(Bone, u16)> {
	let nodes = file.table("nodes");
	let mut out = Vec::with_capacity(joints.len());
	let mut named = Vec::with_capacity(joints.len());

	for (slot, node) in joints.iter().enumerate() {
		let mut rest = nodes.get(*node).map_or(Mat4::IDENTITY, local);
		let mut parent = NO_PARENT;
		let mut above = parents.get(*node).copied().flatten();

		// bounded by the number of nodes, so a file whose `children` form a
		// cycle stops rather than climbing forever.
		for _ in 0..nodes.len() {
			let Some(index) = above else {
				break;
			};

			if let Some(Some(found)) = where_of.get(index) {
				parent = u16::try_from(*found).unwrap_or(NO_PARENT);

				break;
			}

			rest = nodes.get(index).map_or(Mat4::IDENTITY, local) * rest;
			above = parents.get(index).copied().flatten();
		}

		let written = nodes
			.get(*node)
			.and_then(|entry| entry.get("name"))
			.and_then(Value::as_str)
			.unwrap_or("");
		let mut base = tidy(written);

		if base.is_empty() {
			base = format!("bone{slot}");
		}

		let (scale, rotation, position) = rest.to_scale_rotation_translation();

		out.push((
			Bone {
				name: unique(&mut named, &base),
				parent,
				inverse_bind: binds.get(slot).copied().unwrap_or(Mat4::IDENTITY),
				rest: Transform { position, rotation, scale },
			},
			u16::try_from(slot).unwrap_or(0),
		));
	}

	out
}

/// The same bones with every parent written before its child.
///
/// A breadth-first walk down from the roots, which is enough because the nodes
/// a skeleton is made of are a forest. Anything the walk does not reach is a
/// file whose `children` contradict themselves; it is appended as a root
/// rather than dropped, so the mesh still has the bone its vertices name.
///
/// @return the sorted skeleton, and where each original slot ended up
fn sorted(
	raw: &[(Bone, u16)],
	index: usize,
	warnings: &mut Vec<String>,
) -> (SkeletonData, Vec<u16>) {
	let mut order: Vec<usize> = Vec::with_capacity(raw.len());
	let mut placed = vec![false; raw.len()];
	let mut wave: Vec<usize> = (0..raw.len())
		.filter(|slot| raw[*slot].0.parent == NO_PARENT)
		.collect();

	while let Some(slot) = wave.pop() {
		if placed[slot] {
			continue;
		}

		placed[slot] = true;
		order.push(slot);

		for (child, (bone, _)) in raw.iter().enumerate() {
			if !placed[child] && usize::from(bone.parent) == slot {
				wave.push(child);
			}
		}
	}

	if order.len() != raw.len() {
		warnings.push(format!(
			"skin {index} has {} joints nothing leads down to, and they are placed at its root",
			raw.len() - order.len()
		));

		order.extend((0..raw.len()).filter(|slot| !placed[*slot]));
	}

	let mut landed = vec![0_u16; raw.len()];

	for (position, slot) in order.iter().enumerate() {
		landed[*slot] = u16::try_from(position).unwrap_or(0);
	}

	let bones = order
		.iter()
		.map(|slot| {
			let mut bone = raw[*slot].0.clone();

			if bone.parent != NO_PARENT {
				bone.parent = landed
					.get(usize::from(bone.parent))
					.copied()
					.unwrap_or(NO_PARENT);
			}

			bone
		})
		.collect();
	let mut slots = vec![0_u16; raw.len()];

	for (position, slot) in order.iter().enumerate() {
		slots[usize::from(raw[*slot].1)] = u16::try_from(position).unwrap_or(0);
	}

	(SkeletonData { bones }, slots)
}

#[cfg(test)]
mod tests {
	use std::path::Path;

	use super::*;

	/// Reads a document written as text.
	fn read_text(text: &str) -> Skins {
		let file = Gltf::read(text.as_bytes(), Path::new("skin.gltf"), Path::new(""))
			.expect("the document reads");

		read(&file)
	}

	/// A three-node chain with a skin over all of it, written child-first so
	/// the sort has something to do.
	const CHAIN: &str = r#"{
		"asset": { "version": "2.0" },
		"nodes": [
			{ "name": "hips", "children": [1], "translation": [0.0, 1.0, 0.0] },
			{ "name": "spine", "children": [2], "translation": [0.0, 0.5, 0.0] },
			{ "name": "head", "translation": [0.0, 0.25, 0.0] }
		],
		"skins": [{ "name": "rig", "joints": [2, 1, 0] }]
	}"#;

	#[test]
	fn a_skin_becomes_a_skeleton_with_its_parents_written_first() {
		let skins = read_text(CHAIN);
		let skin = &skins.skins[0];

		assert_eq!(skins.warnings, Vec::<String>::new(), "nothing is wrong with it");
		assert_eq!(skin.name, "rig", "named after the skin");
		assert!(skin.data.is_ordered(), "and sorted, whatever order the file listed");
		assert_eq!(
			skin.data
				.bones
				.iter()
				.map(|bone| bone.name.as_str())
				.collect::<Vec<_>>(),
			vec!["hips", "spine", "head"],
			"the chain, top down, out of a file that listed it bottom up"
		);
		assert_eq!(skin.data.bones[0].parent, NO_PARENT);
		assert_eq!(skin.data.bones[1].parent, 0);
		assert_eq!(skin.data.bones[2].parent, 1);
	}

	#[test]
	fn the_sort_says_where_every_slot_a_vertex_names_ended_up() {
		let skins = read_text(CHAIN);

		assert_eq!(
			skins.skins[0].slots,
			vec![2, 1, 0],
			"slot zero was the head and the head sorted last, which is what a vertex naming \
			 slot zero has to be told"
		);
	}

	#[test]
	fn a_rest_is_the_node_transform_the_file_wrote() {
		let skins = read_text(CHAIN);
		let bones = &skins.skins[0].data.bones;

		assert!(
			bones[1]
				.rest
				.position
				.abs_diff_eq(colby_core::glam::Vec3::new(0.0, 0.5, 0.0), 1.0e-6),
			"the spine sits half a unit above the hips, not one and a half above the floor"
		);
	}

	#[test]
	fn a_node_between_two_joints_still_moves_the_lower_one() {
		// the skin names the hips and the head and skips the spine, which the
		// format allows and which does not stop the spine moving the head.
		let skins = read_text(&CHAIN.replace("\"joints\": [2, 1, 0]", "\"joints\": [0, 2]"));
		let bones = &skins.skins[0].data.bones;

		assert_eq!(bones.len(), 2, "two joints, two bones");
		assert_eq!(bones[1].parent, 0, "the head hangs off the hips");
		assert!(
			bones[1]
				.rest
				.position
				.abs_diff_eq(colby_core::glam::Vec3::new(0.0, 0.75, 0.0), 1.0e-6),
			"and carries the skipped spine's half unit, got {}",
			bones[1].rest.position
		);
	}

	#[test]
	fn a_skin_with_no_bind_matrices_is_read_as_though_the_mesh_were_already_in_place() {
		let skins = read_text(CHAIN);

		assert!(
			skins.skins[0]
				.data
				.bones
				.iter()
				.all(|bone| bone.inverse_bind == Mat4::IDENTITY),
			"which is what the specification says the default is"
		);
	}

	#[test]
	fn a_skin_that_names_no_joints_moves_nothing_and_says_so() {
		let skins = read_text(&CHAIN.replace("\"joints\": [2, 1, 0]", "\"joints\": []"));

		assert!(skins.skins[0].data.is_empty(), "no bones");
		assert_eq!(skins.skins[0].name, "rig", "but it is still there and still named");
		assert!(skins.warnings[0].contains("names no joints"), "got {:?}", skins.warnings);
	}

	#[test]
	fn a_skin_that_names_one_node_twice_is_left_out() {
		let skins = read_text(&CHAIN.replace("\"joints\": [2, 1, 0]", "\"joints\": [1, 1]"));

		assert!(skins.skins[0].data.is_empty());
		assert!(skins.warnings[0].contains("twice"), "got {:?}", skins.warnings);
	}

	#[test]
	fn a_skin_that_names_a_node_that_is_not_there_is_left_out() {
		let skins = read_text(&CHAIN.replace("\"joints\": [2, 1, 0]", "\"joints\": [0, 40]"));

		assert!(skins.skins[0].data.is_empty());
		assert!(skins.warnings[0].contains("no such node"), "got {:?}", skins.warnings);
	}

	#[test]
	fn a_file_with_no_skins_at_all_reads_as_nothing() {
		let skins = read_text("{ \"asset\": { \"version\": \"2.0\" } }");

		assert_eq!(skins, Skins::default());
	}

	#[test]
	fn two_skins_cannot_take_the_same_name() {
		let text = CHAIN.replace(
			"\"skins\": [{ \"name\": \"rig\", \"joints\": [2, 1, 0] }]",
			"\"skins\": [{ \"name\": \"rig\", \"joints\": [0] }, { \"name\": \"rig\", \
			 \"joints\": [1] }]",
		);
		let skins = read_text(&text);

		assert_eq!(skins.skins[0].name, "rig");
		assert_eq!(skins.skins[1].name, "rig_1", "because both become files in one directory");
	}
}
