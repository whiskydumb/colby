//! Turning a glTF's meshes and nodes into geometry colby can draw.
//!
//! The other half of the importer. [`super`] reads the file; this reads the
//! *scene* in it, and what comes out is plain colby types with no glTF left in
//! them. Five decisions are worth knowing before reading the code, because
//! each is a place where this engine and the format disagree about shape.
//!
//! **One primitive is one mesh.** A glTF mesh holds primitives, each with its
//! own material; every other engine folds those into one mesh with several
//! surfaces and draws them together. colby cannot: a `Renderable` is one mesh
//! and one material, so an object made of two materials arrives as two meshes
//! and stands in the world as two entities. That is a cost of the ABI rather
//! than a reading of the file, and it is the reason a `panel` comes out twice.
//!
//! **The tree is flattened.** glTF nodes have children and colby's entities do
//! not, so every node's place in the world is worked out here and what comes
//! out is a flat list. The specification guarantees a *local* transform is
//! always a translation, a rotation and a scale, so the only way to end up with
//! something a [`Transform`] cannot hold is this flattening: a rotation between
//! two uneven scales shears. That is checked for and warned about rather than
//! silently rounded off.
//!
//! **A mirrored instance gets its own copy of the mesh.** By the specification
//! the determinant of a node's transform decides which way its triangles wind,
//! and mirroring by a negative scale is a legal thing for an artist to do. The
//! renderer culls back faces one way only, so a mirrored placement would draw
//! inside out. What it gets instead is the same geometry with every triangle
//! turned around, made once and shared by every mirrored placement of it.
//!
//! **A name comes from the file, and an index is the fallback.** Names in glTF
//! are optional and need not be unique, and they end up as asset names here, so
//! they are folded to lowercase, everything outside a small alphabet becomes an
//! underscore, and a collision gets a number. A mesh with several primitives
//! numbers them.
//!
//! **What is missing is computed.** No normals means flat shading, which the
//! specification asks for and which costs the shared vertices; no texture
//! coordinates means zeros; no tangents means the generator every other mesh in
//! this engine already goes through.

use colby_core::{
	Result,
	abi::{
		Transform,
		mesh::{self, MeshData, MeshVertex},
	},
	err,
	glam::{Mat4, Quat, Vec2, Vec3},
};

use super::Gltf;
use crate::json::Value;

/// The drawing mode colby reads. Everything else is skipped with a warning.
const TRIANGLES: u32 = 4;

/// How far a rebuilt transform may drift before the flattening is called shear.
const SQUARE_ENOUGH: f32 = 1e-4;

/// What a name is turned into when the file has none, or none worth having.
const UNNAMED: &str = "mesh";

/// What one glTF file holds, once it is colby's.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Model {
	/// Every primitive of every mesh, and every mirrored copy that was needed.
	pub meshes: Vec<Piece>,

	/// Where each of them stands, in world space.
	pub placements: Vec<Placement>,

	/// What the file said that could not be used. Not a failure: the rest of it
	/// imported, and this is the one moment anybody is told.
	pub warnings: Vec<String>,
}

/// One primitive, as geometry with a name of its own.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Piece {
	/// What it registers under, inside the model's own name.
	pub name: String,

	/// The geometry, with tangents already on it.
	pub data: MeshData,

	/// Which of the file's materials it is made of, if it named one.
	///
	/// Read here and used later: what a material *is* is the next thing this
	/// importer learns, and the index is the only part of it that belongs to
	/// the geometry.
	pub material: Option<usize>,
}

/// One piece of geometry standing somewhere in the world.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Placement {
	/// The node's name, numbered when its mesh had more than one primitive.
	pub name: String,

	/// Which of [`Model::meshes`] stands here.
	pub mesh: usize,

	/// Where it stands, with the whole tree above it already worked in.
	pub transform: Transform,
}

/// Reads a file's scene into geometry.
///
/// @param file - the document with its buffers, from [`Gltf::open`]
/// @return every mesh, where each stands, and what could not be read
pub fn import(file: &Gltf) -> Result<Model> {
	let mut build = Build {
		file,
		meshes: Vec::new(),
		upright: Vec::new(),
		mirrored: Vec::new(),
		named: Vec::new(),
		placed: Vec::new(),
		warnings: Vec::new(),
	};

	build.pieces()?;
	let placements = build.walk();

	Ok(Model {
		meshes: build.meshes,
		placements,
		warnings: build.warnings,
	})
}

/// One import in progress.
struct Build<'a> {
	file: &'a Gltf,
	meshes: Vec<Piece>,
	/// Which piece each primitive became, by mesh and then by primitive.
	upright: Vec<Vec<Option<usize>>>,
	/// The turned-around copy of each, made only when something mirrors it.
	mirrored: Vec<Vec<Option<usize>>>,
	named: Vec<String>,
	placed: Vec<String>,
	warnings: Vec<String>,
}

impl Build<'_> {
	/// Builds a piece for every primitive of every mesh.
	fn pieces(&mut self) -> Result<()> {
		for index in 0..self.file.table("meshes").len() {
			let mut made = Vec::new();

			for primitive in 0..primitives(self.file, index).len() {
				made.push(self.piece(index, primitive)?);
			}

			self.mirrored.push(vec![None; made.len()]);
			self.upright.push(made);
		}

		Ok(())
	}

	/// Builds one primitive, or says why it was left out.
	fn piece(&mut self, mesh: usize, primitive: usize) -> Result<Option<usize>> {
		let entry = primitives(self.file, mesh)[primitive].clone();
		let mode = entry
			.get("mode")
			.and_then(Value::as_u32)
			.unwrap_or(TRIANGLES);

		if mode != TRIANGLES {
			self.warnings.push(format!(
				"mesh {mesh} primitive {primitive} is drawn as mode {mode} rather than as \
				 triangles, and is left out"
			));

			return Ok(None);
		}

		let Some(data) = self.build(mesh, primitive, &entry)? else {
			return Ok(None);
		};

		let name = self.name_for(mesh, primitive);
		let index = self.meshes.len();

		self.meshes.push(Piece {
			name,
			data,
			material: entry.get("material").and_then(Value::as_usize),
		});

		Ok(Some(index))
	}

	/// Reads one primitive's attributes into geometry.
	fn build(
		&mut self,
		mesh: usize,
		primitive: usize,
		entry: &Value,
	) -> Result<Option<MeshData>> {
		let attributes = entry.get("attributes");
		let Some(positions) = attributes
			.and_then(|named| named.get("POSITION"))
			.and_then(Value::as_usize)
		else {
			self.warnings.push(format!(
				"mesh {mesh} primitive {primitive} has no positions, and is left out"
			));

			return Ok(None);
		};

		let positions = self.file.floats(positions)?;
		let count = positions.rows();

		if positions.lanes() != 3 || count == 0 {
			self.warnings.push(format!(
				"mesh {mesh} primitive {primitive} has positions that are not points, and is \
				 left out"
			));

			return Ok(None);
		}

		let normals = self.lanes(attributes, "NORMAL", 3, count);
		let uvs = self.lanes(attributes, "TEXCOORD_0", 2, count);
		let tangents = self.lanes(attributes, "TANGENT", 4, count);
		let mut data = MeshData {
			vertices: (0..count)
				.map(|vertex| {
					let uv = uvs.as_ref().map_or(Vec2::ZERO, |read| {
						Vec2::new(read.row(vertex)[0], read.row(vertex)[1])
					});
					let normal = normals
						.as_ref()
						.map_or(Vec3::Y, |read| point(read.row(vertex)));

					MeshVertex::new(point(positions.row(vertex)), normal, uv)
				})
				.collect(),
			indices: self.indices(entry, count)?,
		};

		if !data.indices.len().is_multiple_of(3) {
			return Err(err!(Asset(
				"mesh {mesh} primitive {primitive} has {} indices, which is not a whole number \
				 of triangles",
				data.indices.len()
			)));
		}

		if !data.indices_are_in_range() {
			return Err(err!(Asset(
				"mesh {mesh} primitive {primitive} has an index past the end of its {count} \
				 vertices"
			)));
		}

		if normals.is_none() {
			flatten(&mut data);
		}

		match tangents {
			| Some(read) =>
				for (vertex, stored) in data.vertices.iter_mut().enumerate() {
					stored.tangent = [
						read.row(vertex)[0],
						read.row(vertex)[1],
						read.row(vertex)[2],
						read.row(vertex)[3],
					];
				},
			| None => mesh::tangents(&mut data),
		}

		Ok(Some(data))
	}

	/// One named attribute, when it is there and is the shape it should be.
	fn lanes(
		&mut self,
		attributes: Option<&Value>,
		name: &str,
		wanted: usize,
		count: usize,
	) -> Option<super::Floats> {
		let index = attributes
			.and_then(|named| named.get(name))
			.and_then(Value::as_usize)?;
		let read = self.file.floats(index).ok()?;

		if read.lanes() != wanted || read.rows() != count {
			self.warnings.push(format!(
				"attribute {name} of accessor {index} does not match the positions beside it, \
				 and is left out"
			));

			return None;
		}

		Some(read)
	}

	/// A primitive's indices, or the ones it implies by not having any.
	fn indices(&self, entry: &Value, count: usize) -> Result<Vec<u32>> {
		match entry.get("indices").and_then(Value::as_usize) {
			| Some(accessor) => self.file.integers(accessor),
			| None => Ok((0..count)
				.map(|vertex| u32::try_from(vertex).unwrap_or(u32::MAX))
				.collect()),
		}
	}

	/// The name a primitive registers under.
	fn name_for(&mut self, mesh: usize, primitive: usize) -> String {
		let written = self
			.file
			.table("meshes")
			.get(mesh)
			.and_then(|entry| entry.get("name"))
			.and_then(Value::as_str)
			.unwrap_or("");
		let mut base = tidy(written);

		if base.is_empty() {
			base = format!("{UNNAMED}{mesh}");
		}

		if primitives(self.file, mesh).len() > 1 {
			base = format!("{base}.{primitive}");
		}

		unique(&mut self.named, &base)
	}

	/// Walks the scene, working out where every piece stands.
	fn walk(&mut self) -> Vec<Placement> {
		let file = self.file;
		let nodes = file.table("nodes");
		let mut seen = vec![false; nodes.len()];
		let mut placements = Vec::new();
		let mut stack: Vec<(usize, Mat4)> = roots(self.file)
			.into_iter()
			.rev()
			.map(|index| (index, Mat4::IDENTITY))
			.collect();

		while let Some((index, above)) = stack.pop() {
			let Some(node) = nodes.get(index) else {
				continue;
			};

			if seen[index] {
				self.warnings.push(format!(
					"node {index} is in the scene more than once, and is placed once"
				));

				continue;
			}

			seen[index] = true;
			let world = above * local(node);

			self.stand(index, node, world, &mut placements);

			for child in node
				.get("children")
				.map_or(&[][..], Value::as_array)
				.iter()
				.rev()
				.filter_map(Value::as_usize)
			{
				stack.push((child, world));
			}
		}

		placements
	}

	/// Puts one node's mesh where the node is.
	fn stand(&mut self, index: usize, node: &Value, world: Mat4, out: &mut Vec<Placement>) {
		let Some(mesh) = node.get("mesh").and_then(Value::as_usize) else {
			return;
		};

		if mesh >= self.upright.len() {
			self.warnings
				.push(format!("node {index} names mesh {mesh}, which is not there"));

			return;
		}

		let transform = self.decompose(index, world);
		let turned = world.determinant() < 0.0;
		let written = node
			.get("name")
			.and_then(Value::as_str)
			.unwrap_or("");
		let several = self.upright[mesh].len() > 1;

		for primitive in 0..self.upright[mesh].len() {
			let Some(piece) = self.piece_for(mesh, primitive, turned) else {
				continue;
			};
			let mut base = tidy(written);

			if base.is_empty() {
				base = format!("node{index}");
			}

			if several {
				base = format!("{base}.{primitive}");
			}

			out.push(Placement {
				name: unique(&mut self.placed, &base),
				mesh: piece,
				transform,
			});
		}
	}

	/// The piece a placement draws, making the turned-around copy on demand.
	fn piece_for(&mut self, mesh: usize, primitive: usize, turned: bool) -> Option<usize> {
		let upright = self.upright[mesh][primitive]?;

		if !turned {
			return Some(upright);
		}

		if let Some(already) = self.mirrored[mesh][primitive] {
			return Some(already);
		}

		let name = unique(&mut self.named, &format!("{}.mirrored", self.meshes[upright].name));
		let index = self.meshes.len();

		self.meshes.push(Piece {
			name,
			data: turn_around(&self.meshes[upright].data),
			material: self.meshes[upright].material,
		});
		self.mirrored[mesh][primitive] = Some(index);

		Some(index)
	}

	/// A world matrix as a transform, complaining if it is not one.
	fn decompose(&mut self, index: usize, world: Mat4) -> Transform {
		let (scale, rotation, position) = world.to_scale_rotation_translation();
		let rebuilt = Mat4::from_scale_rotation_translation(scale, rotation, position);
		let drift = rebuilt
			.to_cols_array()
			.iter()
			.zip(world.to_cols_array())
			.map(|(made, was)| (made - was).abs())
			.fold(0.0_f32, f32::max);

		if drift > SQUARE_ENOUGH {
			self.warnings.push(format!(
				"node {index} is sheared once its parents are folded in, by {drift}, and colby \
				 places it square"
			));
		}

		Transform { position, rotation, scale }
	}
}

/// Which nodes the scene starts from.
fn roots(file: &Gltf) -> Vec<usize> {
	let scenes = file.table("scenes");

	if scenes.is_empty() {
		return orphans(file);
	}

	let which = file
		.document()
		.get("scene")
		.and_then(Value::as_usize)
		.unwrap_or(0);

	scenes
		.get(which)
		.or_else(|| scenes.first())
		.and_then(|scene| scene.get("nodes"))
		.map_or(&[][..], Value::as_array)
		.iter()
		.filter_map(Value::as_usize)
		.collect()
}

/// Every node nobody claims as a child, for a file with no scene in it.
fn orphans(file: &Gltf) -> Vec<usize> {
	let nodes = file.table("nodes");
	let mut claimed = vec![false; nodes.len()];

	for node in nodes {
		for child in node
			.get("children")
			.map_or(&[][..], Value::as_array)
			.iter()
			.filter_map(Value::as_usize)
		{
			if let Some(flag) = claimed.get_mut(child) {
				*flag = true;
			}
		}
	}

	(0..nodes.len())
		.filter(|index| !claimed[*index])
		.collect()
}

/// One node's own transform, however it wrote it.
fn local(node: &Value) -> Mat4 {
	if let Some(written) = node.get("matrix") {
		let cells = written.as_array();

		if cells.len() == 16 {
			let mut columns = [0.0_f32; 16];

			for (cell, value) in columns.iter_mut().zip(cells) {
				*cell = value.as_f32().unwrap_or(0.0);
			}

			return Mat4::from_cols_array(&columns);
		}
	}

	Mat4::from_scale_rotation_translation(
		triple(node.get("scale")).unwrap_or(Vec3::ONE),
		rotation(node),
		triple(node.get("translation")).unwrap_or(Vec3::ZERO),
	)
}

/// A node's rotation, which glTF writes with its scalar part last.
fn rotation(node: &Value) -> Quat {
	let Some(written) = node.get("rotation") else {
		return Quat::IDENTITY;
	};
	let cells = written.as_array();

	if cells.len() != 4 {
		return Quat::IDENTITY;
	}

	let at = |index: usize| cells[index].as_f32().unwrap_or(0.0);
	let quaternion = Quat::from_xyzw(at(0), at(1), at(2), at(3));

	if quaternion.is_normalized() {
		quaternion
	} else {
		Quat::IDENTITY
	}
}

/// Three numbers, when a field holds exactly three.
fn triple(written: Option<&Value>) -> Option<Vec3> {
	let cells = written?.as_array();

	if cells.len() != 3 {
		return None;
	}

	Some(Vec3::new(cells[0].as_f32()?, cells[1].as_f32()?, cells[2].as_f32()?))
}

/// The first three of however many numbers a row holds.
fn point(row: &[f32]) -> Vec3 {
	Vec3::new(
		row.first().copied().unwrap_or(0.0),
		row.get(1).copied().unwrap_or(0.0),
		row.get(2).copied().unwrap_or(0.0),
	)
}

/// Gives every triangle its own vertices and its own normal.
///
/// What the specification asks for when a primitive declares no normals, and it
/// cannot be done any other way: a flat face needs a normal per face, and a
/// shared vertex belongs to several.
fn flatten(data: &mut MeshData) {
	let mut vertices = Vec::with_capacity(data.indices.len());

	for triangle in data.indices.chunks_exact(3) {
		let corners: Vec<MeshVertex> = triangle
			.iter()
			.filter_map(|index| {
				usize::try_from(*index)
					.ok()
					.and_then(|index| data.vertices.get(index))
					.copied()
			})
			.collect();

		if corners.len() != 3 {
			continue;
		}

		let edge = Vec3::from_array(corners[1].position) - Vec3::from_array(corners[0].position);
		let other = Vec3::from_array(corners[2].position) - Vec3::from_array(corners[0].position);
		let normal = edge.cross(other).normalize_or_zero();

		for mut corner in corners {
			corner.normal = normal.to_array();
			vertices.push(corner);
		}
	}

	data.indices = (0..vertices.len())
		.map(|index| u32::try_from(index).unwrap_or(u32::MAX))
		.collect();
	data.vertices = vertices;
}

/// The same geometry with every triangle wound the other way.
///
/// For a placement whose transform mirrors it. The handedness stored in the
/// tangent goes with the winding: the third axis of the frame is a cross
/// product, and a cross product changes sign under a mirror.
fn turn_around(data: &MeshData) -> MeshData {
	let mut copy = data.clone();

	for triangle in copy.indices.chunks_exact_mut(3) {
		triangle.swap(1, 2);
	}

	for vertex in &mut copy.vertices {
		vertex.tangent[3] = -vertex.tangent[3];
	}

	copy
}

/// A name from a file, as something that can be part of an asset's name.
///
/// Lowercase, and everything outside letters, digits, a dash, an underscore and
/// a dot becomes an underscore. Leading and trailing separators go, so a name
/// that was only punctuation comes back empty and the caller numbers it
/// instead. Nothing here may produce a `.` or a `..`, because these end up as
/// pieces of a path.
fn tidy(name: &str) -> String {
	let mut out = String::with_capacity(name.len());

	for letter in name.chars() {
		out.push(match letter {
			| 'a'..='z' | '0'..='9' | '-' | '_' | '.' => letter,
			| 'A'..='Z' => letter.to_ascii_lowercase(),
			| _ => '_',
		});
	}

	out.trim_matches(['_', '.', '-'].as_slice())
		.to_owned()
}

/// A name nothing else in the list has, remembered.
fn unique(taken: &mut Vec<String>, wanted: &str) -> String {
	let mut name = wanted.to_owned();
	let mut attempt = 1;

	while taken.contains(&name) {
		name = format!("{wanted}.{attempt}");
		attempt += 1;
	}

	taken.push(name.clone());

	name
}

/// The primitives of one mesh.
fn primitives(file: &Gltf, mesh: usize) -> &[Value] {
	file.table("meshes")
		.get(mesh)
		.and_then(|entry| entry.get("primitives"))
		.map_or(&[][..], Value::as_array)
}

#[cfg(test)]
mod tests {
	use std::path::Path;

	use colby_core::glam::Vec4;

	use super::*;

	/// Three points in the xy plane, wound the way a front face is.
	const TRIANGLE: &str = "AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA";

	/// The scene an exporter wrote, imported.
	fn exported() -> Model {
		let file = Gltf::read(
			include_bytes!("fixtures/model.glb"),
			Path::new("model.glb"),
			Path::new(""),
		)
		.expect("the fixture reads");

		import(&file).expect("the fixture imports")
	}

	/// A document holding one triangle, and whatever else a test asks for.
	fn scene(body: &str) -> Model {
		let text = format!(
			"{{ \"asset\": {{ \"version\": \"2.0\" }}, \"buffers\": [ {{ \"byteLength\": 36, \
			 \"uri\": \"data:application/octet-stream;base64,{TRIANGLE}\" }} ], \
			 \"bufferViews\": [ {{ \"buffer\": 0, \"byteLength\": 36 }} ], \"accessors\": [ {{ \
			 \"bufferView\": 0, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\" }} ], \
			 {body} }}"
		);
		let file = Gltf::read(text.as_bytes(), Path::new("model.gltf"), Path::new(""))
			.expect("the document reads");

		import(&file).expect("the document imports")
	}

	/// One piece by name.
	fn piece<'a>(model: &'a Model, name: &str) -> &'a Piece {
		model
			.meshes
			.iter()
			.find(|piece| piece.name == name)
			.unwrap_or_else(|| panic!("there is no piece called {name}"))
	}

	/// Where a named placement stands.
	fn stands(model: &Model, name: &str) -> Transform {
		model
			.placements
			.iter()
			.find(|placement| placement.name == name)
			.unwrap_or_else(|| panic!("nothing called {name} stands anywhere"))
			.transform
	}

	#[test]
	fn every_primitive_becomes_a_mesh_of_its_own() {
		let model = exported();
		let names: Vec<&str> = model
			.meshes
			.iter()
			.map(|piece| piece.name.as_str())
			.collect();

		// the panel wore two materials, so it is two; the arm is instanced
		// twice and one of those is mirrored, so it is two as well.
		assert_eq!(names, vec!["arm", "column", "panel.0", "panel.1", "arm.mirrored"]);
	}

	#[test]
	fn every_node_with_a_mesh_stands_somewhere() {
		let model = exported();
		let names: Vec<&str> = model
			.placements
			.iter()
			.map(|placement| placement.name.as_str())
			.collect();

		assert_eq!(names, vec!["column", "arm", "arm_mirror", "panel.0", "panel.1"]);
	}

	#[test]
	fn a_child_is_placed_where_its_parents_put_it() {
		// the numbers are the scene's own, from before it was exported: the
		// column stands at the origin and the two arms are one unit either side
		// of it and half a unit up. Nothing but a correct flattening arrives
		// there, because the arms are written in the file relative to a parent
		// that is scaled unevenly.
		let model = exported();

		for (name, at) in [
			("column", Vec3::new(0.0, 1.5, 0.0)),
			("arm", Vec3::new(1.0, 0.5, 0.0)),
			("arm_mirror", Vec3::new(-1.0, 0.5, 0.0)),
			("panel.0", Vec3::new(0.0, 0.0, -2.0)),
		] {
			let stood = stands(&model, name).position;

			assert!(stood.abs_diff_eq(at, 1e-5), "{name} stands at {stood} rather than {at}");
		}
	}

	#[test]
	fn the_exported_scene_needs_no_apology() {
		assert_eq!(exported().warnings, Vec::<String>::new());
	}

	#[test]
	fn a_mirrored_placement_draws_a_copy_wound_the_other_way() {
		let model = exported();
		let upright = piece(&model, "arm");
		let turned = piece(&model, "arm.mirrored");

		let places = |piece: &Piece| -> Vec<[f32; 3]> {
			piece
				.data
				.vertices
				.iter()
				.map(|vertex| vertex.position)
				.collect()
		};

		assert_eq!(places(upright), places(turned), "the same points, in the same order");

		for (was, now) in upright
			.data
			.indices
			.chunks_exact(3)
			.zip(turned.data.indices.chunks_exact(3))
		{
			assert_eq!([was[0], was[2], was[1]], [now[0], now[1], now[2]], "turned around");
		}
	}

	#[test]
	fn the_mirrored_copy_is_what_the_mirrored_node_stands_on() {
		let model = exported();
		let turned = model
			.meshes
			.iter()
			.position(|piece| piece.name == "arm.mirrored")
			.expect("the copy was made");
		let upright = model
			.meshes
			.iter()
			.position(|piece| piece.name == "arm")
			.expect("and the original is still there");

		assert_eq!(
			model
				.placements
				.iter()
				.find(|placement| placement.name == "arm_mirror")
				.map(|placement| placement.mesh),
			Some(turned)
		);
		assert_eq!(
			model
				.placements
				.iter()
				.find(|placement| placement.name == "arm")
				.map(|placement| placement.mesh),
			Some(upright),
			"and the one that is not mirrored still draws the original"
		);
	}

	#[test]
	fn a_mirror_turns_the_tangent_frame_over_with_the_winding() {
		let model = exported();
		let upright = piece(&model, "arm");
		let turned = piece(&model, "arm.mirrored");

		for (was, now) in upright
			.data
			.vertices
			.iter()
			.zip(&turned.data.vertices)
		{
			assert!(
				(was.tangent[3] + now.tangent[3]).abs() < 1e-6,
				"the handedness is the other one"
			);
		}
	}

	#[test]
	fn every_imported_mesh_arrives_with_a_frame_on_every_vertex() {
		for piece in exported().meshes {
			for vertex in &piece.data.vertices {
				let normal = Vec3::from_array(vertex.normal);
				let tangent = Vec4::from_array(vertex.tangent);

				assert!(
					(normal.length() - 1.0).abs() < 1e-3,
					"{}: a normal of {}",
					piece.name,
					normal.length()
				);
				assert!(tangent.is_finite(), "{}: a tangent of {tangent}", piece.name);
				assert!(
					vertex.tangent[3].abs() > 0.5,
					"{}: a handedness of {}",
					piece.name,
					vertex.tangent[3]
				);
			}
		}
	}

	#[test]
	fn a_primitive_with_no_normals_is_shaded_flat() {
		// the specification asks for this, and it cannot be done any other way:
		// a face needs one normal and a shared vertex belongs to several, so
		// the vertices stop being shared.
		let model = scene(
			"\"meshes\": [ { \"primitives\": [ { \"attributes\": { \"POSITION\": 0 } } ] } ], \
			 \"nodes\": [ { \"mesh\": 0 } ], \"scenes\": [ { \"nodes\": [ 0 ] } ]",
		);
		let only = &model.meshes[0].data;

		assert_eq!(only.vertices.len(), 3, "one triangle, three vertices of its own");
		assert_eq!(only.indices, vec![0, 1, 2], "and they are its own");

		for vertex in &only.vertices {
			assert!(
				Vec3::from_array(vertex.normal).abs_diff_eq(Vec3::Z, 1e-6),
				"the face points at the viewer, and so does every corner of it"
			);
		}
	}

	#[test]
	fn a_primitive_with_no_indices_draws_its_vertices_in_order() {
		let model = scene(
			"\"meshes\": [ { \"primitives\": [ { \"attributes\": { \"POSITION\": 0 } } ] } ], \
			 \"nodes\": [ { \"mesh\": 0 } ]",
		);

		assert_eq!(model.meshes[0].data.indices, vec![0, 1, 2]);
		assert_eq!(model.placements.len(), 1, "a file with no scene still has its nodes");
	}

	#[test]
	fn a_rotation_is_read_with_its_scalar_part_last() {
		// glTF writes a quaternion xyzw and glam takes it xyzw, and the only way
		// to find out that one of them changed is to turn something with it: a
		// quarter turn about the vertical takes what pointed right to what
		// points at the viewer's back. Read the other way round these four
		// numbers are still a unit quaternion, so nothing but the result of
		// turning a vector by it says which order it was.
		let model = scene(
			"\"meshes\": [ { \"primitives\": [ { \"attributes\": { \"POSITION\": 0 } } ] } ], \
			 \"nodes\": [ { \"mesh\": 0, \"rotation\": [ 0, 0.70710678, 0, 0.70710678 ] } ]",
		);
		let turned = model.placements[0].transform.rotation * Vec3::X;

		assert!(turned.abs_diff_eq(-Vec3::Z, 1e-5), "x went to {turned} rather than -z");
	}

	#[test]
	fn a_node_may_write_its_place_as_a_matrix_instead() {
		// the other half of the specification's transform, which no exporter
		// this project has read uses and which is legal all the same. Column
		// major, so the translation is the last four numbers.
		let model = scene(
			"\"meshes\": [ { \"primitives\": [ { \"attributes\": { \"POSITION\": 0 } } ] } ], \
			 \"nodes\": [ { \"mesh\": 0, \"matrix\": [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, \
			 2, 3, 1 ] } ]",
		);
		let stood = model.placements[0].transform.position;

		assert!(stood.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-5), "it stands at {stood}");
	}

	#[test]
	fn a_shear_that_only_the_flattening_could_have_made_is_reported() {
		// a rotation between two uneven scales. Neither node on its own is
		// anything a transform cannot hold, which is the whole point: the
		// specification promises a local transform decomposes, and this is the
		// one way to end up with something that does not.
		let model = scene(
			"\"meshes\": [ { \"primitives\": [ { \"attributes\": { \"POSITION\": 0 } } ] } ], \
			 \"nodes\": [ { \"mesh\": 0, \"rotation\": [ 0, 0, 0.3826834, 0.9238795 ] }, { \
			 \"children\": [ 0 ], \"scale\": [ 1, 3, 1 ] } ], \"scenes\": [ { \"nodes\": [ 1 ] \
			 } ]",
		);

		assert!(
			model
				.warnings
				.iter()
				.any(|line| line.contains("sheared")),
			"got {:?}",
			model.warnings
		);
	}

	#[test]
	fn a_primitive_drawn_some_other_way_is_left_out_and_said_so() {
		let model = scene(
			"\"meshes\": [ { \"primitives\": [ { \"mode\": 1, \"attributes\": { \"POSITION\": 0 \
			 } } ] } ], \"nodes\": [ { \"mesh\": 0 } ]",
		);

		assert!(model.meshes.is_empty(), "nothing was built");
		assert!(model.placements.is_empty(), "so nothing stands anywhere");
		assert!(model.warnings[0].contains("mode 1"), "and it says which: {:?}", model.warnings);
	}

	#[test]
	fn names_are_folded_and_a_collision_is_numbered() {
		let model = scene(
			"\"meshes\": [ { \"name\": \"Front Wall\", \"primitives\": [ { \"attributes\": { \
			 \"POSITION\": 0 } } ] }, { \"name\": \"Front Wall\", \"primitives\": [ { \
			 \"attributes\": { \"POSITION\": 0 } } ] }, { \"name\": \"!!!\", \"primitives\": [ \
			 { \"attributes\": { \"POSITION\": 0 } } ] } ], \"nodes\": [ { \"mesh\": 0 } ]",
		);
		let names: Vec<&str> = model
			.meshes
			.iter()
			.map(|piece| piece.name.as_str())
			.collect();

		assert_eq!(names, vec!["front_wall", "front_wall.1", "mesh2"]);
	}

	#[test]
	fn a_node_that_reaches_itself_is_placed_once_rather_than_forever() {
		let model = scene(
			"\"meshes\": [ { \"primitives\": [ { \"attributes\": { \"POSITION\": 0 } } ] } ], \
			 \"nodes\": [ { \"mesh\": 0, \"children\": [ 0 ] } ], \"scenes\": [ { \"nodes\": [ \
			 0 ] } ]",
		);

		assert_eq!(model.placements.len(), 1);
		assert!(
			model.warnings[0].contains("more than once"),
			"and it says so: {:?}",
			model.warnings
		);
	}

	#[test]
	fn a_node_naming_a_mesh_that_is_not_there_stands_nowhere() {
		let model = scene("\"nodes\": [ { \"mesh\": 7 } ]");

		assert!(model.placements.is_empty());
		assert!(model.warnings[0].contains("mesh 7"), "got {:?}", model.warnings);
	}

	#[test]
	fn an_index_past_the_end_of_a_primitive_is_refused() {
		let text = format!(
			"{{ \"asset\": {{ \"version\": \"2.0\" }}, \"buffers\": [ {{ \"byteLength\": 36, \
			 \"uri\": \"data:application/octet-stream;base64,{TRIANGLE}\" }} ], \
			 \"bufferViews\": [ {{ \"buffer\": 0, \"byteLength\": 36 }} ], \"accessors\": [ {{ \
			 \"bufferView\": 0, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\" }}, \
			 {{ \"bufferView\": 0, \"byteOffset\": 12, \"componentType\": 5125, \"count\": 3, \
			 \"type\": \"SCALAR\" }} ], \"meshes\": [ {{ \"primitives\": [ {{ \"attributes\": \
			 {{ \"POSITION\": 0 }}, \"indices\": 1 }} ] }} ] }}"
		);
		let file = Gltf::read(text.as_bytes(), Path::new("model.gltf"), Path::new(""))
			.expect("the document reads");
		let message = import(&file)
			.expect_err("the primitive is refused")
			.to_string();

		assert!(message.contains("past the end"), "got {message}");
	}
}
