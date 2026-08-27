//! Reading Wavefront OBJ, the import format.
//!
//! **Why OBJ and not glTF.** glTF carries normals, materials and a scene graph
//! that OBJ has to be told about, and the day this engine has materials worth
//! importing that will matter - but OBJ is a line-oriented text format that
//! parses in two hundred lines with no dependency, and it stays editable in a
//! text editor, which is exactly what the hot-reload loop is for. A `.glb` is a
//! JSON document inside a binary container: hand-editing one is not a thing
//! anybody does, and reading one means either a JSON crate or a weekend.
//!
//! What is read: `v`, `vt`, `vn`, `f`. What is skipped without complaint: `o`,
//! `g`, `s`, `usemtl`, `mtllib`, comments and blank lines. Everything from
//! every group lands in one mesh, because a [`MeshId`](colby_core::abi::MeshId)
//! is one mesh; splitting by group is what the format's `flags` word is for
//! once there is a reason.
//!
//! **`v` is flipped on the way in.** OBJ measures the second texture coordinate
//! from the bottom of the image and every graphics API this will run on samples
//! from the top, so the importer stores `1 - v`. Getting this wrong is the
//! classic upside-down-texture bug, and it is cheaper to fix here, once, than
//! in every shader forever.
//!
//! Faces may have any number of corners and are triangulated as a fan, which is
//! correct for the convex faces a modeling tool emits and visibly wrong for
//! concave ones. A corner with no normal index makes the whole face flat: the
//! face's own normal is computed and given to every one of its vertices, which
//! is the same convention the built-in primitives use.
//!
//! Winding and handedness pass straight through. OBJ is right-handed with
//! counter-clockwise front faces, which is what the engine's pipeline already
//! expects, so nothing is flipped on the way in.

use std::{collections::HashMap, fs, path::Path, str::SplitWhitespace};

use colby_core::{
	Result,
	abi::mesh::{MeshData, MeshVertex},
	err,
	glam::{Vec2, Vec3},
};

/// The extension this importer claims.
pub const EXTENSION: &str = "obj";

/// Reads an OBJ file into geometry.
///
/// @param path - the `.obj` to read
/// @return the mesh, or the line that could not be read
pub fn import_file(path: &Path) -> Result<MeshData> {
	let text = fs::read_to_string(path)
		.map_err(|error| err!(Asset("reading {}: {error}", path.display())))?;

	parse(&text).map_err(|reason| err!(Asset("{}:{reason}", path.display())))
}

/// Reads OBJ text into geometry.
///
/// @param text - the contents of an `.obj`
/// @return the mesh, or the line that could not be read
pub fn import(text: &str) -> Result<MeshData> {
	parse(text).map_err(|reason| err!(Asset("line {reason}")))
}

/// The attribute triple a corner names, which is what two corners have to agree
/// on before they may share a vertex.
type CornerKey = (usize, Option<usize>, Option<usize>);

/// One corner of a face: a position, and whatever else the file named for it.
#[derive(Clone, Copy, Debug)]
struct Corner {
	position: usize,
	uv: Option<usize>,
	normal: Option<usize>,
}

/// The state of a parse in progress.
#[derive(Debug, Default)]
struct Builder {
	positions: Vec<Vec3>,
	texcoords: Vec<Vec2>,
	normals: Vec<Vec3>,
	mesh: MeshData,
	/// Vertices already emitted for an explicit attribute triple, so a
	/// smooth-shaded model shares them the way its author meant it to. Two
	/// corners that agree on the position and disagree on the texture
	/// coordinate are two vertices, which is what a UV seam is.
	shared: HashMap<CornerKey, u32>,
	/// Reused by every face, so a mesh with ten thousand of them allocates
	/// once.
	corners: Vec<Corner>,
	/// The vertex slots this face's corners ended up in. Reused for the same
	/// reason.
	slots: Vec<u32>,
}

/// Reads OBJ text.
///
/// @param text - the whole file
/// @return the mesh, or `line <n>: <what is wrong>`
fn parse(text: &str) -> std::result::Result<MeshData, String> {
	let mut builder = Builder::default();

	// a byte order mark is not whitespace and not a comment, so left in place it
	// would make the first keyword of a file saved by a Windows editor unknown.
	// Every text editor on this platform is capable of writing one.
	let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);

	for (index, line) in text.lines().enumerate() {
		let number = index + 1;
		builder
			.line(line)
			.map_err(|reason| format!("{number}: {reason}"))?;
	}

	Ok(builder.mesh)
}

impl Builder {
	/// Reads one line.
	fn line(&mut self, line: &str) -> std::result::Result<(), String> {
		let line = line.split('#').next().unwrap_or("");
		let mut tokens = line.split_whitespace();
		let Some(keyword) = tokens.next() else {
			return Ok(());
		};

		match keyword {
			| "v" => self.positions.push(vector(tokens, "v")?),
			| "vn" => self.normals.push(vector(tokens, "vn")?),
			| "vt" => self.texcoords.push(texcoord(tokens)?),
			| "f" => self.face(tokens)?,
			// o, g, s and the material keywords describe structure this format
			// does not carry. Skipping them quietly is deliberate: a real export
			// is full of them and none of them are errors.
			| "o" | "g" | "s" | "usemtl" | "mtllib" => {},
			| other => return Err(format!("unknown keyword {other:?}")),
		}

		Ok(())
	}

	/// Reads an `f` line and appends its triangles.
	fn face(&mut self, tokens: SplitWhitespace<'_>) -> std::result::Result<(), String> {
		self.corners.clear();
		for token in tokens {
			let corner = self.corner(token)?;
			self.corners.push(corner);
		}

		if self.corners.len() < 3 {
			return Err(format!(
				"a face needs at least three corners, this one has {}",
				self.corners.len()
			));
		}

		match self.flat_normal() {
			| Some(normal) => self.push_flat(normal),
			| None => self.push_shared(),
		}

		Ok(())
	}

	/// Reads one `v`, `v/vt`, `v//vn` or `v/vt/vn` corner.
	fn corner(&self, token: &str) -> std::result::Result<Corner, String> {
		let mut parts = token.split('/');
		let position = parts
			.next()
			.filter(|part| !part.is_empty())
			.ok_or_else(|| format!("corner {token:?} has no position index"))?;

		let position = resolve(position, self.positions.len(), "vertex")?;

		let uv = parts
			.next()
			.filter(|part| !part.is_empty())
			.map(|part| resolve(part, self.texcoords.len(), "texture"))
			.transpose()?;

		let normal = parts
			.next()
			.filter(|part| !part.is_empty())
			.map(|part| resolve(part, self.normals.len(), "normal"))
			.transpose()?;

		Ok(Corner { position, uv, normal })
	}

	/// The normal to give every corner, when the file did not give them all
	/// one.
	///
	/// @return the face's own normal, or `None` when every corner named one
	fn flat_normal(&self) -> Option<Vec3> {
		if self
			.corners
			.iter()
			.all(|corner| corner.normal.is_some())
		{
			return None;
		}

		// Newell's method rather than one cross product: it is the same answer
		// for a triangle and the right one for a polygon whose corners are not
		// exactly coplanar, which is most of them once a tool has rounded the
		// coordinates to six decimals.
		let mut normal = Vec3::ZERO;
		for (index, corner) in self.corners.iter().enumerate() {
			let next = self.corners[(index + 1) % self.corners.len()];
			let (current, next) = (self.position(corner.position), self.position(next.position));

			normal += Vec3::new(
				(current.y - next.y) * (current.z + next.z),
				(current.z - next.z) * (current.x + next.x),
				(current.x - next.x) * (current.y + next.y),
			);
		}

		// a face with no area has no normal. It draws nothing either way, so it
		// gets an arbitrary one rather than stopping the compile.
		Some(normal.try_normalize().unwrap_or(Vec3::Y))
	}

	/// Emits a face whose corners all carry their own normal.
	///
	/// Vertices are shared between faces that name the same pair, which is what
	/// makes a smooth-shaded export come out smooth and small.
	fn push_shared(&mut self) {
		self.slots.clear();

		for index in 0..self.corners.len() {
			let corner = self.corners[index];
			let key = (corner.position, corner.uv, corner.normal);

			if let Some(existing) = self.shared.get(&key) {
				self.slots.push(*existing);
				continue;
			}

			let vertex = MeshVertex::new(
				self.position(corner.position),
				self.normal(corner.normal),
				self.texcoord(corner.uv),
			);
			let Ok(slot) = u32::try_from(self.mesh.vertices.len()) else {
				return;
			};

			self.mesh.vertices.push(vertex);
			self.shared.insert(key, slot);
			self.slots.push(slot);
		}

		fan(&mut self.mesh.indices, &self.slots);
	}

	/// Emits a flat face: one set of vertices, all carrying the face's normal.
	///
	/// Nothing is shared with any other face, which is the point - a shared
	/// vertex could only carry one normal, and a flat face wants its own.
	fn push_flat(&mut self, normal: Vec3) {
		self.slots.clear();

		for index in 0..self.corners.len() {
			let Ok(slot) = u32::try_from(self.mesh.vertices.len()) else {
				return;
			};

			let corner = self.corners[index];
			let vertex =
				MeshVertex::new(self.position(corner.position), normal, self.texcoord(corner.uv));

			self.mesh.vertices.push(vertex);
			self.slots.push(slot);
		}

		fan(&mut self.mesh.indices, &self.slots);
	}

	/// A position by index, or the origin if the index is somehow past the end.
	fn position(&self, index: usize) -> Vec3 {
		self.positions
			.get(index)
			.copied()
			.unwrap_or(Vec3::ZERO)
	}

	/// A normal by index, or straight up if there is none.
	fn normal(&self, index: Option<usize>) -> Vec3 {
		index
			.and_then(|index| self.normals.get(index))
			.copied()
			.unwrap_or(Vec3::Y)
	}

	/// A texture coordinate by index, or the top left if there is none.
	fn texcoord(&self, index: Option<usize>) -> Vec2 {
		index
			.and_then(|index| self.texcoords.get(index))
			.copied()
			.unwrap_or(Vec2::ZERO)
	}
}

/// Triangulates a polygon as a fan around its first corner.
fn fan(indices: &mut Vec<u32>, slots: &[u32]) {
	for corner in 1..slots.len().saturating_sub(1) {
		indices.extend([slots[0], slots[corner], slots[corner + 1]]);
	}
}

/// Reads the first three numbers of a line as a vector.
fn vector(tokens: SplitWhitespace<'_>, keyword: &str) -> std::result::Result<Vec3, String> {
	let mut values = [0.0_f32; 3];
	let mut seen = 0;

	for (slot, token) in values.iter_mut().zip(tokens) {
		*slot = token
			.parse()
			.map_err(|error| format!("{keyword:?} has {token:?} where a number goes: {error}"))?;
		seen += 1;
	}

	if seen < 3 {
		return Err(format!("{keyword:?} needs three numbers, found {seen}"));
	}

	Ok(Vec3::from_array(values))
}

/// Reads a `vt` line, flipping the second coordinate.
///
/// A third value is legal in OBJ and describes a depth into a 3D texture; there
/// are none of those here, so it is read past and dropped.
fn texcoord(tokens: SplitWhitespace<'_>) -> std::result::Result<Vec2, String> {
	let mut values = [0.0_f32; 2];
	let mut seen = 0;

	for (slot, token) in values.iter_mut().zip(tokens) {
		*slot = token
			.parse()
			.map_err(|error| format!("\"vt\" has {token:?} where a number goes: {error}"))?;
		seen += 1;
	}

	if seen < 2 {
		return Err(format!("\"vt\" needs two numbers, found {seen}"));
	}

	// v counts up from the bottom in OBJ and down from the top everywhere the
	// engine will sample it. @ref the module docs.
	Ok(Vec2::new(values[0], 1.0 - values[1]))
}

/// Turns one OBJ index into an offset into the list it addresses.
///
/// OBJ counts from one, and a negative index counts back from whatever has been
/// declared so far - which is why this needs the current length rather than the
/// final one.
///
/// @param token - the index as written
/// @param declared - how many of that thing have been declared above this line
/// @param what - the word to use in the error
fn resolve(token: &str, declared: usize, what: &str) -> std::result::Result<usize, String> {
	let value: i64 = token
		.parse()
		.map_err(|error| format!("{what} index {token:?} is not a whole number: {error}"))?;

	let declared = i64::try_from(declared).unwrap_or(i64::MAX);
	let index = match value {
		| 0 => return Err(format!("{what} index 0 is not valid; obj counts from one")),
		| value if value > 0 => value - 1,
		| value => declared + value,
	};

	if index < 0 || index >= declared {
		return Err(format!(
			"{what} index {value} does not address any of the {declared} above it"
		));
	}

	usize::try_from(index).map_err(|error| format!("{what} index {value} does not fit: {error}"))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The two triangles of a unit square in the xz plane, wound facing up, and
	/// written the way a tool would write them.
	const SQUARE: &str = "\
# a floor
v -0.5 0.0  0.5
v  0.5 0.0  0.5
v  0.5 0.0 -0.5
v -0.5 0.0 -0.5
f 1 2 3 4
";

	/// The normal a triangle's winding implies, by the right-hand rule.
	fn wound_normal(mesh: &MeshData, triangle: usize) -> Vec3 {
		let corner = |offset: usize| {
			let index = usize::try_from(mesh.indices[triangle * 3 + offset]).unwrap_or(0);

			Vec3::from_array(mesh.vertices[index].position)
		};

		(corner(1) - corner(0))
			.cross(corner(2) - corner(0))
			.normalize()
	}

	#[test]
	fn a_quad_becomes_two_triangles() {
		let mesh = import(SQUARE).expect("the square parses");

		assert_eq!(mesh.vertices.len(), 4, "one set of corners, shared by the fan");
		assert_eq!(mesh.triangles(), 2, "a quad is two triangles");
		assert!(mesh.indices_are_in_range(), "and the fan addresses its own vertices");
	}

	#[test]
	fn a_face_with_no_normals_gets_the_one_its_winding_implies() {
		let mesh = import(SQUARE).expect("the square parses");

		for vertex in &mesh.vertices {
			let normal = Vec3::from_array(vertex.normal);

			assert!(normal.abs_diff_eq(Vec3::Y, 1.0e-5), "wound facing up, got {normal}");
		}

		assert!(wound_normal(&mesh, 0).dot(Vec3::Y) > 0.99, "and the triangles agree");
	}

	#[test]
	fn declared_normals_are_used_and_their_vertices_shared() {
		let text = "\
v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
f 1//1 2//1 3//1
f 1//1 3//1 2//1
";
		let mesh = import(text).expect("it parses");

		assert_eq!(
			mesh.vertices.len(),
			3,
			"the second face names the same pairs, so it reuses the same vertices"
		);
		assert_eq!(mesh.triangles(), 2, "and still draws two triangles");
		assert!(
			Vec3::from_array(mesh.vertices[0].normal).abs_diff_eq(Vec3::Z, 1.0e-6),
			"with the normal the file gave, got {:?}",
			mesh.vertices[0].normal
		);
	}

	#[test]
	fn a_flat_face_does_not_share_vertices_with_anything() {
		let text = "\
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
f 1 2 3
";
		let mesh = import(text).expect("it parses");

		assert_eq!(
			mesh.vertices.len(),
			6,
			"a shared vertex can only carry one normal, so a flat face keeps its own"
		);
	}

	#[test]
	fn every_corner_spelling_is_accepted() {
		let text = "\
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vn 0 0 1
f 1/1/1 2/1/1 3/1/1
f 1/1 2/1 3/1
f 1//1 2//1 3//1
f 1 2 3
";
		let mesh = import(text).expect("all four spellings parse");

		assert_eq!(mesh.triangles(), 4, "one triangle per face");
	}

	#[test]
	fn texture_coordinates_are_read_and_flipped() {
		let text = "v 0 0 0
v 1 0 0
v 0 1 0
vt 0.25 0.0
vt 1.0 1.0
vt 0.5 0.75
f 1/1 2/2 3/3
";
		let mesh = import(text).expect("it parses");
		let uvs: Vec<Vec2> = mesh
			.vertices
			.iter()
			.map(|vertex| Vec2::from_array(vertex.uv))
			.collect();

		// v counts up from the bottom in the file and down from the top here.
		let close = |left: Vec2, right: Vec2| left.abs_diff_eq(right, 1.0e-6);

		assert!(close(uvs[0], Vec2::new(0.25, 1.0)), "v = 0 is the bottom: {}", uvs[0]);
		assert!(close(uvs[1], Vec2::new(1.0, 0.0)), "v = 1 is the top: {}", uvs[1]);
		assert!(close(uvs[2], Vec2::new(0.5, 0.25)), "and between is flipped: {}", uvs[2]);
	}

	#[test]
	fn a_seam_splits_a_vertex_that_two_faces_would_otherwise_share() {
		let text = "v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 1
vn 0 0 1
f 1/1/1 2/1/1 3/1/1
f 1/1/1 3/1/1 2/1/1
f 1/2/1 2/1/1 3/1/1
";
		let mesh = import(text).expect("it parses");

		assert_eq!(
			mesh.vertices.len(),
			4,
			"the first two faces name the same triples and share three vertices; the third 			 \
			 gives one corner a different texture coordinate, which is a fourth vertex"
		);
		assert_eq!(mesh.triangles(), 3, "and all three faces still draw");
	}

	#[test]
	fn a_third_texture_coordinate_is_read_past_and_dropped() {
		let text = "v 0 0 0
v 1 0 0
v 0 1 0
vt 0.5 0.5 0.9
f 1/1 2/1 3/1
";
		let mesh = import(text).expect("a depth into a 3D texture is legal and unused");

		assert!(
			Vec2::from_array(mesh.vertices[0].uv).abs_diff_eq(Vec2::splat(0.5), 1.0e-6),
			"the first two are what is kept, got {:?}",
			mesh.vertices[0].uv
		);
	}

	#[test]
	fn a_corner_with_no_texture_coordinate_lands_at_the_origin() {
		let mesh = import(SQUARE).expect("the square parses");

		for vertex in &mesh.vertices {
			assert!(
				Vec2::from_array(vertex.uv).abs_diff_eq(Vec2::ZERO, 1.0e-6),
				"nothing was named, so nothing is invented: {:?}",
				vertex.uv
			);
		}
	}

	#[test]
	fn a_short_texture_coordinate_is_refused() {
		let error = import(
			"vt 0.5
",
		)
		.expect_err("a texture coordinate needs two numbers");
		let message = error.to_string();

		assert!(message.contains("line 1"), "with its line: {message}");
		assert!(message.contains("found 1"), "and what was found: {message}");
	}

	#[test]
	fn a_texture_index_past_the_end_is_refused() {
		let text = "v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
f 1/1 2/2 3/1
";
		let error = import(text).expect_err("there is only one texture coordinate");

		assert!(error.to_string().contains("texture index 2"), "and it says which one: {error}");
	}

	#[test]
	fn negative_indices_count_back_from_here() {
		let text = "\
v 0 0 0
v 1 0 0
v 0 1 0
f -3 -2 -1
";
		let mesh = import(text).expect("it parses");
		let positions: Vec<Vec3> = mesh
			.vertices
			.iter()
			.map(|vertex| Vec3::from_array(vertex.position))
			.collect();

		assert_eq!(
			positions,
			vec![Vec3::ZERO, Vec3::X, Vec3::Y],
			"minus one is the last vertex declared, not the first"
		);
	}

	#[test]
	fn comments_groups_and_materials_are_skipped() {
		let text = "\
# a comment
mtllib nothing.mtl
o thing
g part
usemtl stone
s 1
v 0 0 0  # trailing comment
v 1 0 0
v 0 1 0
vt 0.5 0.5
f 1 2 3
";
		let mesh = import(text).expect("none of that is an error");

		assert_eq!(mesh.triangles(), 1, "and the one face still lands");
	}

	#[test]
	fn every_group_lands_in_one_mesh() {
		let text = "\
v 0 0 0
v 1 0 0
v 0 1 0
g first
f 1 2 3
g second
f 3 2 1
";
		let mesh = import(text).expect("it parses");

		assert_eq!(mesh.triangles(), 2, "a MeshId is one mesh, so the groups are merged");
	}

	#[test]
	fn a_polygon_is_fanned_from_its_first_corner() {
		let text = "\
v 0 0 0
v 1 0 0
v 2 1 0
v 1 2 0
v 0 2 0
f 1 2 3 4 5
";
		let mesh = import(text).expect("a pentagon parses");

		assert_eq!(mesh.triangles(), 3, "five corners fan into three triangles");
		assert_eq!(
			&mesh.indices,
			&[0, 1, 2, 0, 2, 3, 0, 3, 4],
			"each triangle keeps the first corner"
		);
	}

	#[test]
	fn a_file_saved_with_a_byte_order_mark_still_parses() {
		let mesh = import(&format!("\u{FEFF}{SQUARE}")).expect("a BOM is not geometry");

		assert_eq!(mesh.triangles(), 2, "and does not hide the first line");
	}

	#[test]
	fn an_empty_file_is_an_empty_mesh_rather_than_an_error() {
		let mesh = import("\n# nothing here\n\n").expect("nothing is not wrong");

		assert!(mesh.is_empty(), "and it draws nothing");
	}

	#[test]
	fn an_index_past_the_end_is_refused_with_its_line() {
		let error = import("v 0 0 0\nf 1 2 3\n").expect_err("there is only one vertex");
		let message = error.to_string();

		assert!(message.contains("line 2"), "the line is named: {message}");
		assert!(message.contains("vertex index 2"), "and so is the index: {message}");
	}

	#[test]
	fn a_zero_index_is_refused() {
		let error =
			import("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n").expect_err("obj counts from one");

		assert!(error.to_string().contains("counts from one"), "{error}");
	}

	#[test]
	fn a_short_vertex_is_refused() {
		let error = import("v 1 2\n").expect_err("a position needs three numbers");
		let message = error.to_string();

		assert!(message.contains("line 1"), "with its line: {message}");
		assert!(message.contains("found 2"), "and what was found: {message}");
	}

	#[test]
	fn a_number_that_is_not_a_number_is_refused() {
		let error = import("v 1 two 3\n").expect_err("`two` is not a float");

		assert!(error.to_string().contains("\"two\""), "and it is quoted back: {error}");
	}

	#[test]
	fn a_face_of_two_corners_is_refused() {
		let error =
			import("v 0 0 0\nv 1 0 0\nf 1 2\n").expect_err("two corners is not a triangle");

		assert!(error.to_string().contains("three corners"), "{error}");
	}

	#[test]
	fn an_unknown_keyword_is_refused_rather_than_ignored() {
		let error = import("curv 0 1 2 3\n").expect_err("free-form geometry is not supported");
		let message = error.to_string();

		assert!(message.contains("unknown keyword"), "and says so: {message}");
		assert!(message.contains("curv"), "naming it: {message}");
	}

	#[test]
	fn a_face_with_no_area_does_not_stop_the_compile() {
		let text = "\
v 0 0 0
v 0 0 0
v 0 0 0
f 1 2 3
";
		let mesh = import(text).expect("a degenerate face is geometry, not an error");

		assert_eq!(mesh.triangles(), 1, "it is emitted and rasterizes to nothing");
	}

	#[test]
	fn a_file_that_does_not_exist_names_itself() {
		let path = std::env::temp_dir().join("colby-no-such-mesh.obj");
		let error = import_file(&path).expect_err("there is nothing to read");

		assert!(
			error
				.to_string()
				.contains("colby-no-such-mesh.obj"),
			"{error}"
		);
	}
}
