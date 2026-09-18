//! Turning a glTF's meshes and nodes into geometry colby can draw.
//!
//! The other half of the importer. [`super`] reads the file; this reads the
//! *scene* in it, and what comes out is plain colby types with no glTF left in
//! them. Six decisions are worth knowing before reading the code, because
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
//! this engine already goes through. A vertex color or a second set of
//! coordinates is a block of its own, written only when the file has either:
//! the other is white or nought beside it.
//!
//! **A lamp stands on its own.** A node naming one of the file's lights stands
//! a lamp where the node stands, facing the way the node faces, beside
//! whatever mesh the node also carries: a lamp is a placement with no mesh,
//! because a `Renderable` is one mesh and a light is not one. Its scale is
//! dropped, which is what the lights extension says a node's scale does to a
//! light. @ref `super::light` for what a lamp's numbers become.

use colby_core::{
	Result,
	abi::{
		Light, Transform,
		mesh::{self, BONES_PER_VERTEX, MeshData, MeshVertex, PaintVertex, SkinVertex},
	},
	err,
	glam::{Mat4, Quat, Vec2, Vec3, Vec4},
};

use super::{Clip, Extracted, Gltf, Skin, Surface, clip, light, skin};
use crate::json::Value;

/// The drawing mode colby reads. Everything else is skipped with a warning.
const TRIANGLES: u32 = 4;

/// Every vertex attribute this importer reads, plus the one that has its own
/// complaint. Anything else a primitive names is said out loud, @ref
/// [`Build::unread_attributes`].
const READ_ATTRIBUTES: &[&str] = &[
	"POSITION",
	"NORMAL",
	"TANGENT",
	"TEXCOORD_0",
	"TEXCOORD_1",
	"COLOR_0",
	"JOINTS_0",
	"WEIGHTS_0",
	"JOINTS_1",
	"WEIGHTS_1",
];

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

	/// Every lamp a node stands, in the order the walk met them.
	pub lamps: Vec<Lamp>,

	/// Every material the file declares, in its own order, which is what a
	/// [`Piece::material`] indexes.
	pub materials: Vec<Surface>,

	/// Pictures that were stored inside the file and have to be written out
	/// beside its meshes, because nothing else will.
	pub textures: Vec<Extracted>,

	/// Every skin the file declares, in its own order, which is what a
	/// [`Placement::skeleton`] indexes. One that could not be read is here
	/// with no bones in it and is named by nothing.
	pub skins: Vec<Skin>,

	/// Every animation the file declares, in its own order.
	///
	/// A clip names the bones it moves with text and names no skeleton at
	/// all, so nothing indexes this: it is written out beside the skins and
	/// found later by name.
	pub clips: Vec<Clip>,

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

	/// Which of [`Model::skins`] moves it, when anything does.
	///
	/// A skinned node's own transform is deliberately not in
	/// [`transform`](Self::transform): the specification says the transform of
	/// a skinned mesh node is ignored, because every vertex of it is placed by
	/// its bones instead.
	pub skeleton: Option<usize>,
}

/// One lamp standing somewhere in the world.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lamp {
	/// The node's name, numbered when a piece of geometry already has it.
	pub name: String,

	/// What it shines, in colby's own units.
	pub light: Light,

	/// Where it stands and which way it faces, with the whole tree above it
	/// worked in. The scale is always one.
	pub transform: Transform,
}

/// Reads a file's scene into geometry.
///
/// @param file - the document with its buffers, from [`Gltf::open`]
/// @return every mesh, where each stands, and what could not be read
pub fn import(file: &Gltf) -> Result<Model> {
	let mut rigs = skin::read(file);
	let mut lamps = light::read(file);

	rigs.warnings.append(&mut lamps.warnings);

	let mut build = Build {
		file,
		mesh_skin: mesh_skins(file, rigs.skins.len(), &mut rigs.warnings),
		skins: rigs.skins,
		lights: lamps.lights,
		meshes: Vec::new(),
		upright: Vec::new(),
		mirrored: Vec::new(),
		named: Vec::new(),
		placed: Vec::new(),
		warnings: rigs.warnings,
	};

	build.pieces()?;
	let (placements, lamps) = build.walk();
	let mut coats = super::material::read(file);
	let mut warnings = build.warnings;

	warnings.append(&mut coats.warnings);

	// after the skins and not before them: a clip's tracks are named by
	// whatever the skins decided each joint is called, so there is nothing to
	// name them against until that has happened.
	let mut moves = clip::read(file, &build.skins);

	warnings.append(&mut moves.warnings);
	warnings.extend(super::unread_extensions(file.document()));
	warnings.extend(unread_cameras(file));

	Ok(Model {
		meshes: build.meshes,
		placements,
		lamps,
		materials: coats.surfaces,
		textures: coats.pictures,
		skins: build.skins,
		clips: moves.clips,
		warnings,
	})
}

/// Says so when the file carries cameras, which colby does not read.
///
/// A camera is a core part of glTF rather than an extension, so the warning
/// about what a file uses cannot catch it: a scene exported whole rather than
/// a prop carries the one it was framed with, and every one of them was passed
/// over without a word. colby's camera belongs to the world and not to a
/// model, so there is nowhere to put one; the point is that the file said
/// something and the import did not answer.
///
/// A light is not here because a light is read: it is an extension in glTF
/// (`KHR_lights_punctual`), and a sun, the one kind colby leaves out, is warned
/// about where the lights are read. @ref `super::light`.
///
/// @param file - the document
/// @return one line, or none when the file has no cameras
fn unread_cameras(file: &Gltf) -> Option<String> {
	let cameras = file.table("cameras").len();

	(cameras > 0).then(|| {
		format!(
			"the file carries {cameras} camera{}, which colby does not read: a camera belongs \
			 to a world here rather than to a model",
			if cameras == 1 { "" } else { "s" }
		)
	})
}

/// Which skin moves each mesh, worked out from the nodes that stand them.
///
/// A mesh's vertices name bones, and which bones those are is a property of
/// the skin the *node* points at - so the pairing has to be settled before any
/// geometry is read. The format allows one mesh to be stood by two nodes with
/// different skins; nothing exports that, and the first one wins with a word
/// about it rather than a refusal.
fn mesh_skins(file: &Gltf, skins: usize, warnings: &mut Vec<String>) -> Vec<Option<usize>> {
	let mut out = vec![None; file.table("meshes").len()];

	for (index, node) in file.table("nodes").iter().enumerate() {
		let (Some(mesh), Some(skin)) = (
			node.get("mesh").and_then(Value::as_usize),
			node.get("skin").and_then(Value::as_usize),
		) else {
			continue;
		};

		if skin >= skins {
			warnings.push(format!("node {index} names skin {skin}, which is not there"));

			continue;
		}

		let Some(slot) = out.get_mut(mesh) else {
			continue;
		};

		match *slot {
			| Some(already) if already != skin => warnings.push(format!(
				"mesh {mesh} is moved by skin {already} and by skin {skin}, and colby gives it \
				 the first"
			)),
			| Some(_) => (),
			| None => *slot = Some(skin),
		}
	}

	out
}

/// One import in progress.
struct Build<'a> {
	file: &'a Gltf,
	/// Every skin of the file, already sorted into skeletons.
	skins: Vec<Skin>,
	/// Every light the file declares, or nothing where colby does not take one.
	lights: Vec<Option<Light>>,
	/// Which skin moves each mesh, or nothing for a mesh bones do not move.
	mesh_skin: Vec<Option<usize>>,
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
		if entry.get("targets").is_some() {
			self.warnings.push(format!(
				"mesh {mesh} primitive {primitive} has shapes it can be morphed into, which \
				 colby does not read, and stands in the one it was modeled in"
			));
		}

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

		self.unread_attributes(mesh, primitive, attributes);

		let normals = self.lanes(attributes, "NORMAL", &[3], count);
		let uvs = self.lanes(attributes, "TEXCOORD_0", &[2], count);
		// the specification says a tangent means nothing without the normal it
		// is measured against, and that one written beside no normals is to be
		// ignored: the flat normals made below are not the ones it was made for
		let tangents = if normals.is_some() {
			self.lanes(attributes, "TANGENT", &[4], count)
		} else {
			None
		};
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
			skin: Vec::new(),
			levels: Vec::new(),
			paint: self.paint_of(attributes, count),
			sheet: [0, 0],
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

		// read before the flattening and carried through it, like the paint:
		// every block is a vertex's, and a vertex is what flattening copies
		data.skin = self.skin_of(mesh, primitive, attributes, count);

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

	/// Names every vertex attribute this importer does not read.
	///
	/// A rule rather than a list of the ones somebody thought of: whatever a
	/// primitive names that is not in [`READ_ATTRIBUTES`] is said out loud, so
	/// a mesh whose paint went into a second color set, or whose lightmap
	/// coordinates went into a third, says why instead of looking like an
	/// exporter bug. It catches an exporter's own `_SOMETHING` too, which is
	/// the case nobody would have written a line for.
	///
	/// `JOINTS_1` is left out of the complaint although it is not read: it has
	/// a warning of its own further down that says the useful thing - that a
	/// vertex named more than four bones and the first four were taken -
	/// and `WEIGHTS_1` only ever appears beside it.
	///
	/// @param mesh - which mesh, for the message
	/// @param primitive - which primitive of it
	/// @param attributes - the primitive's `attributes` object
	fn unread_attributes(&mut self, mesh: usize, primitive: usize, attributes: Option<&Value>) {
		let dropped: Vec<&str> = attributes
			.map_or(&[][..], Value::as_object)
			.iter()
			.map(|(name, _)| name.as_str())
			.filter(|name| !READ_ATTRIBUTES.contains(name))
			.collect();

		if dropped.is_empty() {
			return;
		}

		self.warnings.push(format!(
			"mesh {mesh} primitive {primitive} names {}, which colby does not read",
			dropped.join(", ")
		));
	}

	/// What moves each vertex of one primitive, when bones do.
	///
	/// Two things happen here that nothing above this function knows about.
	/// The bone indices are the file's joint slots and are carried over to the
	/// bones those slots became, because sorting the skeleton renumbered them.
	/// And the weights are quantized into bytes adding to exactly 255, which
	/// is the shape a skin block holds and which the exchange format offers as
	/// one of its own three.
	fn skin_of(
		&mut self,
		mesh: usize,
		primitive: usize,
		attributes: Option<&Value>,
		count: usize,
	) -> Vec<SkinVertex> {
		// taken by value rather than borrowed: everything below this wants
		// `&mut self` to say what it could not read.
		let Some(slots) = self
			.mesh_skin
			.get(mesh)
			.copied()
			.flatten()
			.and_then(|index| self.skins.get(index))
			.filter(|rig| !rig.data.is_empty())
			.map(|rig| rig.slots.clone())
		else {
			return Vec::new();
		};

		if attributes.is_some_and(|named| named.get("JOINTS_1").is_some()) {
			self.warnings.push(format!(
				"mesh {mesh} primitive {primitive} names more than four bones for a vertex, and \
				 colby reads the first four"
			));
		}

		let Some(accessor) = attributes
			.and_then(|named| named.get("JOINTS_0"))
			.and_then(Value::as_usize)
		else {
			self.warnings.push(format!(
				"mesh {mesh} primitive {primitive} is moved by a skin and says which bones for \
				 none of its vertices, and is left where it was"
			));

			return Vec::new();
		};

		let bones = match self.file.wholes(accessor) {
			| Ok(read) if read.lanes() == BONES_PER_VERTEX && read.rows() == count => read,
			| Ok(read) => {
				self.warnings.push(format!(
					"mesh {mesh} primitive {primitive} names bones {} at a time for {} of its \
					 {count} vertices, and is left where it was",
					read.lanes(),
					read.rows()
				));

				return Vec::new();
			},
			| Err(error) => {
				self.warnings.push(format!(
					"mesh {mesh} primitive {primitive} has bone indices that could not be read \
					 ({error}), and is left where it was"
				));

				return Vec::new();
			},
		};

		let Some(weights) = self.lanes(attributes, "WEIGHTS_0", &[BONES_PER_VERTEX], count)
		else {
			self.warnings.push(format!(
				"mesh {mesh} primitive {primitive} names bones and says how much each pulls for \
				 none of its vertices, and is left where it was"
			));

			return Vec::new();
		};

		let mut wild = false;
		let mut limp = false;
		let out = (0..count)
			.map(|vertex| {
				pulled(bones.row(vertex), weights.row(vertex), &slots, &mut limp, &mut wild)
			})
			.collect();

		if wild {
			self.warnings.push(format!(
				"mesh {mesh} primitive {primitive} names a bone its own skin does not have, and \
				 colby reads that pull as the skin's first bone"
			));
		}

		if limp {
			self.warnings.push(format!(
				"mesh {mesh} primitive {primitive} has a vertex nothing pulls on, and colby \
				 hangs it rigidly off the first bone it names"
			));
		}

		out
	}

	/// What each vertex of one primitive was painted with and where it samples
	/// the second set, when the file says either.
	///
	/// One block for the two, because the vertex stage reads them as one: a
	/// file that has only a color gets the second set at nought beside it, and
	/// one that has only a second set gets white. A color with three channels
	/// is opaque, which is what the specification says it means.
	///
	/// @param attributes - the primitive's `attributes` object
	/// @param count - how many vertices it has
	/// @return one entry per vertex, or nothing for a primitive that has
	/// neither
	fn paint_of(&mut self, attributes: Option<&Value>, count: usize) -> Vec<PaintVertex> {
		let colors = self.lanes(attributes, "COLOR_0", &[3, 4], count);
		let seconds = self.lanes(attributes, "TEXCOORD_1", &[2], count);

		if colors.is_none() && seconds.is_none() {
			return Vec::new();
		}

		(0..count)
			.map(|vertex| {
				let color = colors.as_ref().map_or(Vec4::ONE, |read| {
					let row = read.row(vertex);

					Vec4::new(row[0], row[1], row[2], row.get(3).copied().unwrap_or(1.0))
				});
				let uv2 = seconds.as_ref().map_or(Vec2::ZERO, |read| {
					Vec2::new(read.row(vertex)[0], read.row(vertex)[1])
				});

				PaintVertex::new(color, uv2)
			})
			.collect()
	}

	/// One named attribute, when it is there and is one of the shapes it
	/// should be.
	fn lanes(
		&mut self,
		attributes: Option<&Value>,
		name: &str,
		wanted: &[usize],
		count: usize,
	) -> Option<super::Floats> {
		let index = attributes
			.and_then(|named| named.get(name))
			.and_then(Value::as_usize)?;
		let read = self.file.floats(index).ok()?;

		if !wanted.contains(&read.lanes()) || read.rows() != count {
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
		let mut base = super::tidy(written);

		if base.is_empty() {
			base = format!("{UNNAMED}{mesh}");
		}

		if primitives(self.file, mesh).len() > 1 {
			base = format!("{base}_{primitive}");
		}

		super::unique(&mut self.named, &base)
	}

	/// Walks the scene, working out where every piece and every lamp stands.
	fn walk(&mut self) -> (Vec<Placement>, Vec<Lamp>) {
		let file = self.file;
		let nodes = file.table("nodes");
		let mut seen = vec![false; nodes.len()];
		let mut placements = Vec::new();
		let mut lamps = Vec::new();
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
			self.lamp(index, node, world, &mut lamps);

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

		(placements, lamps)
	}

	/// Puts the lamp a node names where the node is, facing the way it faces.
	///
	/// The scale comes off: the extension says a node's scale touches neither
	/// a light's reach nor its brightness. What a cone needs of the rest is its
	/// -z, and the rotation carries that whatever the scale was - a mirror
	/// included, because the decomposition puts a mirror's sign on x and leaves
	/// z its length, so the rotation's -z is the matrix's -z made one long.
	fn lamp(&mut self, index: usize, node: &Value, world: Mat4, out: &mut Vec<Lamp>) {
		let Some(which) = light::named_by(node) else {
			return;
		};
		let Some(named) = self.lights.get(which).copied() else {
			self.warnings
				.push(format!("node {index} names light {which}, which is not there"));

			return;
		};
		// a sun, or a kind colby has no word for: said once, where the list
		// was read, rather than once a node
		let Some(light) = named else {
			return;
		};
		let placed = self.decompose(index, world);
		let written = node
			.get("name")
			.and_then(Value::as_str)
			.unwrap_or("");
		let mut base = super::tidy(written);

		if base.is_empty() {
			base = format!("node{index}");
		}

		out.push(Lamp {
			name: super::unique(&mut self.placed, &base),
			light,
			transform: Transform { scale: Vec3::ONE, ..placed },
		});
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

		let moved = node
			.get("skin")
			.and_then(Value::as_usize)
			.filter(|index| {
				self.skins
					.get(*index)
					.is_some_and(|rig| !rig.data.is_empty())
			});
		// a skinned node's own transform is ignored, which the specification
		// says outright: every vertex of it is placed by its bones instead, and
		// applying both would place it twice.
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
			let mut base = super::tidy(written);

			if base.is_empty() {
				base = format!("node{index}");
			}

			if several {
				base = format!("{base}_{primitive}");
			}

			out.push(Placement {
				name: super::unique(&mut self.placed, &base),
				mesh: piece,
				transform: if moved.is_some() {
					Transform::IDENTITY
				} else {
					transform
				},
				skeleton: moved,
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

		let name =
			super::unique(&mut self.named, &format!("{}_mirrored", self.meshes[upright].name));
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
pub(super) fn local(node: &Value) -> Mat4 {
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
///
/// **Every block goes with its vertex.** A corner copied for a face takes its
/// bones and its paint along, because both are the vertex's and the copy is the
/// vertex; leaving them behind would be a block as long as the vertices were
/// before, which is no longer the mesh.
fn flatten(data: &mut MeshData) {
	let mut vertices = Vec::with_capacity(data.indices.len());
	let mut skin = Vec::new();
	let mut paint = Vec::new();

	for triangle in data.indices.chunks_exact(3) {
		let Some(slots) = corners_of(triangle, data.vertices.len()) else {
			continue;
		};

		let [first, second, third] = slots.map(|slot| data.vertices[slot]);
		let edge = Vec3::from_array(second.position) - Vec3::from_array(first.position);
		let other = Vec3::from_array(third.position) - Vec3::from_array(first.position);
		let normal = edge.cross(other).normalize_or_zero();

		for slot in slots {
			let mut corner = data.vertices[slot];
			corner.normal = normal.to_array();
			vertices.push(corner);
			skin.extend(data.skin.get(slot).copied());
			paint.extend(data.paint.get(slot).copied());
		}
	}

	data.indices = (0..vertices.len())
		.map(|index| u32::try_from(index).unwrap_or(u32::MAX))
		.collect();
	data.vertices = vertices;
	data.skin = skin;
	data.paint = paint;
}

/// The three vertices one triangle names, when all three are there.
fn corners_of(triangle: &[u32], vertices: usize) -> Option<[usize; 3]> {
	let mut slots = [0_usize; 3];

	for (slot, index) in slots.iter_mut().zip(triangle) {
		*slot = usize::try_from(*index)
			.ok()
			.filter(|at| *at < vertices)?;
	}

	Some(slots)
}

/// One vertex's bones and weights, with the file's joint slots carried over.
///
/// The weights are settled before the bones, because a bone with none of them
/// is never read: whatever index sat beside it in the file is not carried over
/// and is left at zero, which is what a skin block says such a slot holds and
/// what makes two exports of one mesh come out byte for byte the same.
///
/// @param names - what the file said, indexing its own `skin.joints`
/// @param shares - how much each pulls, already read as floats
/// @param slots - where each of the file's joint slots ended up after sorting
/// @param limp - set when nothing pulls on this vertex at all
/// @param wild - set when it names a bone its skin does not have
fn pulled(
	names: &[u32],
	shares: &[f32],
	slots: &[u16],
	limp: &mut bool,
	wild: &mut bool,
) -> SkinVertex {
	let mut pull = [0.0_f32; BONES_PER_VERTEX];

	for (share, read) in pull.iter_mut().zip(shares) {
		*share = read.max(0.0);
	}

	if pull.iter().sum::<f32>() <= 0.0 {
		*limp = true;
		pull[0] = 1.0;
	}

	let given = quantize(pull);
	let mut bones = [0_u16; BONES_PER_VERTEX];

	for (bone, (wanted, weight)) in bones.iter_mut().zip(names.iter().zip(given)) {
		if weight == 0 {
			continue;
		}

		match usize::try_from(*wanted)
			.ok()
			.and_then(|at| slots.get(at))
		{
			| Some(found) => *bone = *found,
			| None => *wild = true,
		}
	}

	SkinVertex { bones, weights: given }
}

/// Four weights as the bytes a skin block holds, adding to exactly 255.
///
/// Normalized first, because the specification only asks an exporter to come
/// reasonably close to one, and then handed out by largest remainder: every
/// share takes its whole part and what is left over goes to whichever shares
/// the rounding cheated most. That is what makes the four add up to exactly
/// [`SkinVertex::WHOLE`], which is the invariant `.cmesh` refuses a file over.
///
/// A vertex nothing pulls on, or one whose weights are not finite numbers, is
/// hung rigidly off the first bone it names. Leaving it at zero would put it
/// at the origin rather than on the character.
///
/// @param weights - one share per bone, none of them negative
/// @return the same shares as bytes, summing to 255
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "each value is floored and clamped into 0..=255 on the line above the cast, and \
	          try_from is not available for a float"
)]
fn quantize(weights: [f32; BONES_PER_VERTEX]) -> [u8; BONES_PER_VERTEX] {
	let whole = f32::from(SkinVertex::WHOLE);
	let total: f32 = weights.iter().sum();

	if !total.is_finite() || total <= 0.0 {
		let mut out = [0_u8; BONES_PER_VERTEX];
		out[0] = 255;

		return out;
	}

	let mut given = [0_u16; BONES_PER_VERTEX];
	let mut owed = [0.0_f32; BONES_PER_VERTEX];

	for (index, share) in weights.iter().enumerate() {
		let exact = ((share / total) * whole).clamp(0.0, whole);
		let floor = exact.floor();

		given[index] = floor as u16;
		owed[index] = exact - floor;
	}

	let mut order = [0_usize, 1, 2, 3];
	order.sort_by(|first, second| owed[*second].total_cmp(&owed[*first]));

	let short = SkinVertex::WHOLE.saturating_sub(given.iter().sum());

	for index in order.iter().take(usize::from(short)) {
		given[*index] = given[*index].saturating_add(1);
	}

	given.map(|share| u8::try_from(share).unwrap_or(u8::MAX))
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

	use colby_core::abi::LightKind;

	use super::*;

	/// Three points in the xy plane, wound the way a front face is.
	const TRIANGLE: &str = "AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA";

	/// The same triangle with bones and weights after it, ninety-six bytes.
	const SKINNED: &str = "AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAABAAABAAAAAACAPwAAAAAAAAAAAAAAAAAAAD8AAAA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAA";

	/// A document holding that triangle, moved by whatever the body declares.
	///
	/// Accessor zero is the positions, one the bone indices as unsigned bytes,
	/// two the weights as floats - which is what an exporter actually writes.
	fn rigged(body: &str) -> Model {
		let text = format!(
			"{{ \"asset\": {{ \"version\": \"2.0\" }}, \"buffers\": [ {{ \"byteLength\": 96, \
			 \"uri\": \"data:application/octet-stream;base64,{SKINNED}\" }} ], \"bufferViews\": \
			 [ {{ \"buffer\": 0, \"byteLength\": 36 }}, {{ \"buffer\": 0, \"byteOffset\": 36, \
			 \"byteLength\": 12 }}, {{ \"buffer\": 0, \"byteOffset\": 48, \"byteLength\": 48 }} \
			 ], \"accessors\": [ {{ \"bufferView\": 0, \"componentType\": 5126, \"count\": 3, \
			 \"type\": \"VEC3\" }}, {{ \"bufferView\": 1, \"componentType\": 5121, \"count\": \
			 3, \"type\": \"VEC4\" }}, {{ \"bufferView\": 2, \"componentType\": 5126, \
			 \"count\": 3, \"type\": \"VEC4\" }} ], {body} }}"
		);
		let file = Gltf::read(text.as_bytes(), Path::new("rig.gltf"), Path::new(""))
			.expect("the fixture reads");

		import(&file).expect("and imports")
	}

	/// Two joints and a mesh hanging off them, with the joints listed
	/// child-first so the sort has to renumber them.
	const RIG: &str = "\"meshes\": [ { \"primitives\": [ { \"attributes\": { \"POSITION\": 0, \
	                   \"JOINTS_0\": 1, \"WEIGHTS_0\": 2 } } ] } ], \"nodes\": [ { \"name\": \
	                   \"root\", \"children\": [1] }, { \"name\": \"tip\", \"translation\": \
	                   [0.0, 1.0, 0.0] }, { \"name\": \"body\", \"mesh\": 0, \"skin\": 0, \
	                   \"translation\": [5.0, 0.0, 0.0] } ], \"skins\": [ { \"name\": \"rig\", \
	                   \"joints\": [1, 0] } ], \"scenes\": [ { \"nodes\": [0, 2] } ]";

	#[test]
	fn a_skinned_primitive_comes_back_with_a_bone_and_a_weight_per_vertex() {
		let model = rigged(RIG);
		let skin = &model.meshes[0].data.skin;

		assert_eq!(model.warnings, Vec::<String>::new(), "nothing is wrong with it");
		assert_eq!(skin.len(), 3, "one entry per vertex");
		assert!(model.meshes[0].data.weights_are_whole(), "and every one of them adds up");
	}

	#[test]
	fn the_bones_a_vertex_names_are_carried_over_to_where_the_sort_put_them() {
		let model = rigged(RIG);
		let skin = &model.meshes[0].data.skin;

		// the file lists its joints tip first, so slot zero is the tip and the
		// sort moves the tip to bone one. A vertex naming slot zero has to
		// come back naming bone one, or the character bends at the wrong end.
		assert_eq!(model.skins[0].slots, vec![1, 0]);
		assert_eq!(skin[0].bones[0], 1, "vertex zero hangs off slot zero, which is the tip");
		assert_eq!(skin[2].bones[0], 0, "and vertex two off slot one, which is the root");
	}

	#[test]
	fn two_bones_sharing_a_vertex_are_quantized_to_bytes_that_still_add_up() {
		let model = rigged(RIG);
		let shared = model.meshes[0].data.skin[1];

		assert_eq!(
			shared.weights,
			[128, 127, 0, 0],
			"half and half does not divide 255, so the rounding gives the odd one away"
		);
		assert_eq!(shared.total(), SkinVertex::WHOLE);
		assert_eq!(shared.bones, [1, 0, 0, 0], "both carried over, in the file's own order");
	}

	#[test]
	fn a_skinned_piece_stands_at_the_origin_whatever_its_node_said() {
		let model = rigged(RIG);
		let placement = &model.placements[0];

		assert_eq!(placement.skeleton, Some(0), "it names the skin that moves it");
		assert_eq!(
			placement.transform,
			Transform::IDENTITY,
			"and drops the five units its node was moved by, because the specification says a \
			 skinned node's transform is ignored and its bones place every vertex instead"
		);
	}

	#[test]
	fn a_mesh_nothing_moves_keeps_its_transform_and_carries_no_skin() {
		let model = rigged(&RIG.replace("\"skin\": 0, ", ""));

		assert!(model.meshes[0].data.skin.is_empty(), "no skin block");
		assert_eq!(model.placements[0].skeleton, None, "and nothing names one");
		assert!(
			model.placements[0]
				.transform
				.position
				.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1.0e-6),
			"so its own node transform is the whole of where it stands"
		);
	}

	#[test]
	fn a_skinned_primitive_with_no_weights_is_left_where_it_was() {
		let model = rigged(&RIG.replace(", \"WEIGHTS_0\": 2", ""));

		assert!(model.meshes[0].data.skin.is_empty(), "half a skin is not a skin");
		assert!(
			model
				.warnings
				.iter()
				.any(|said| said.contains("how much each pulls")),
			"and it says so: {:?}",
			model.warnings
		);
	}

	#[test]
	fn shapes_a_primitive_can_be_morphed_into_are_named_rather_than_read() {
		let model = rigged(&RIG.replace(
			"\"WEIGHTS_0\": 2 } } ]",
			"\"WEIGHTS_0\": 2 }, \"targets\": [ { \"POSITION\": 0 } ] } ]",
		));

		assert_eq!(model.meshes[0].data.vertices.len(), 3, "the base shape is still read");
		assert!(
			model
				.warnings
				.iter()
				.any(|said| said.contains("morphed into")),
			"and the rest are named rather than silently dropped: {:?}",
			model.warnings
		);
	}

	#[test]
	fn a_second_set_of_bones_is_named_rather_than_read() {
		let model = rigged(&RIG.replace("\"WEIGHTS_0\": 2", "\"WEIGHTS_0\": 2, \"JOINTS_1\": 1"));

		assert_eq!(model.meshes[0].data.skin.len(), 3, "the first four bones are still read");
		assert!(
			model
				.warnings
				.iter()
				.any(|said| said.contains("more than four bones")),
			"and the rest are named: {:?}",
			model.warnings
		);
	}

	#[test]
	fn a_vertex_naming_a_bone_its_skin_does_not_have_is_pulled_by_the_first_one() {
		// one joint, and the vertices still name slot one.
		let model = rigged(&RIG.replace("\"joints\": [1, 0]", "\"joints\": [0]"));
		let skin = &model.meshes[0].data.skin;

		assert!(skin.iter().all(|vertex| vertex.bones_below(1)), "nothing points past the bone");
		assert!(
			model
				.warnings
				.iter()
				.any(|said| said.contains("does not have")),
			"and it says so: {:?}",
			model.warnings
		);
	}

	#[test]
	fn a_mesh_two_skins_claim_goes_to_the_first_of_them() {
		let model = rigged(
			&RIG.replace(
				"\"skins\": [ { \"name\": \"rig\", \"joints\": [1, 0] } ]",
				"\"skins\": [ { \"name\": \"rig\", \"joints\": [1, 0] }, { \"name\": \"other\", \
				 \"joints\": [0] } ]",
			)
			.replace("\"nodes\": [0, 2]", "\"nodes\": [0, 2, 3]")
			.replace(
				"\"translation\": [5.0, 0.0, 0.0] } ]",
				"\"translation\": [5.0, 0.0, 0.0] }, { \"name\": \"twin\", \"mesh\": 0, \
				 \"skin\": 1 } ]",
			),
		);

		assert!(
			model
				.warnings
				.iter()
				.any(|said| said.contains("gives it the first")),
			"got {:?}",
			model.warnings
		);
	}

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

	/// The same three points, stored as normalized signed shorts instead of
	/// floats, which is what `KHR_mesh_quantization` lets an exporter do.
	const QUANTIZED: &str = "AAAAAAAA/38AAAAAAAD/fwAA";

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
		assert_eq!(names, vec!["arm", "column", "panel_0", "panel_1", "arm_mirrored"]);
	}

	#[test]
	fn every_node_with_a_mesh_stands_somewhere() {
		let model = exported();
		let names: Vec<&str> = model
			.placements
			.iter()
			.map(|placement| placement.name.as_str())
			.collect();

		assert_eq!(names, vec!["column", "arm", "arm_mirror", "panel_0", "panel_1"]);
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
			("panel_0", Vec3::new(0.0, 0.0, -2.0)),
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
		let turned = piece(&model, "arm_mirrored");

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
			.position(|piece| piece.name == "arm_mirrored")
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
		let turned = piece(&model, "arm_mirrored");

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

		assert_eq!(names, vec!["front_wall", "front_wall_1", "mesh2"]);
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

	#[test]
	fn a_mesh_quantized_into_shorts_is_the_same_triangle_as_one_written_in_floats() {
		// what `KHR_mesh_quantization` is on the reading side, and the whole
		// reason accepting it needed nothing built: the same three points
		// stored as normalized shorts, widened by the machinery that was
		// already there for texture coordinates
		let text = format!(
			"{{ \"asset\": {{ \"version\": \"2.0\" }}, \"extensionsRequired\": [ \
			 \"KHR_mesh_quantization\" ], \"extensionsUsed\": [ \"KHR_mesh_quantization\" ], \
			 \"buffers\": [ {{ \"byteLength\": 18, \"uri\": \
			 \"data:application/octet-stream;base64,{QUANTIZED}\" }} ], \"bufferViews\": [ {{ \
			 \"buffer\": 0, \"byteLength\": 18 }} ], \"accessors\": [ {{ \"bufferView\": 0, \
			 \"componentType\": 5122, \"normalized\": true, \"count\": 3, \"type\": \"VEC3\" }} \
			 ], \"meshes\": [ {{ \"name\": \"small\", \"primitives\": [ {{ \"attributes\": {{ \
			 \"POSITION\": 0 }} }} ] }} ] }}"
		);
		let file = Gltf::read(text.as_bytes(), Path::new("model.gltf"), Path::new(""))
			.expect("a file that requires it reads");
		let model = import(&file).expect("and imports");
		let plain = scene(
			"\"meshes\": [ { \"name\": \"small\", \"primitives\": [ { \"attributes\": { \
			 \"POSITION\": 0 } } ] } ]",
		);

		assert_eq!(
			piece(&model, "small").data.vertices,
			piece(&plain, "small").data.vertices,
			"stored small and stored as floats, the same triangle to the vertex"
		);
		assert_eq!(model.warnings, Vec::<String>::new(), "and nothing to complain about");
	}

	#[test]
	fn an_attribute_this_importer_does_not_read_is_named_rather_than_dropped() {
		// a second color set is the one that costs somebody an afternoon: the
		// paint they meant went into it, and nothing at all says why the mesh
		// arrived without it
		let model = scene(
			"\"meshes\": [ { \"name\": \"painted\", \"primitives\": [ { \"attributes\": { \
			 \"POSITION\": 0, \"COLOR_1\": 0, \"TEXCOORD_2\": 0 } } ] } ]",
		);

		assert_eq!(model.warnings.len(), 1, "one line for the primitive: {:?}", model.warnings);
		assert!(
			model.warnings[0].contains("COLOR_1") && model.warnings[0].contains("TEXCOORD_2"),
			"naming each of them: {:?}",
			model.warnings
		);
		assert!(
			scene(
				"\"meshes\": [ { \"primitives\": [ { \"attributes\": { \"POSITION\": 0 } } ] } ]"
			)
			.warnings
			.is_empty(),
			"and a primitive with nothing extra on it says nothing"
		);
	}

	/// Bytes as the base64 a `data:` address carries.
	fn base64(bytes: &[u8]) -> String {
		const ALPHABET: &[u8; 64] =
			b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
		let digit = |value: u32| char::from(ALPHABET[usize::try_from(value & 0x3F).unwrap_or(0)]);
		let mut out = String::new();

		for chunk in bytes.chunks(3) {
			let word = chunk
				.iter()
				.enumerate()
				.fold(0_u32, |word, (at, byte)| word | (u32::from(*byte) << (16 - 8 * at)));
			let digits = chunk.len() + 1;
			out.extend((0..digits).map(|place| digit(word >> (18 - 6 * place))));
			out.extend(core::iter::repeat_n('=', 4 - digits));
		}

		out
	}

	/// Little-endian bytes of some numbers, each written as `T` writes itself.
	fn bytes_of<const N: usize, T, F>(values: &[T], each: F) -> Vec<u8>
	where
		T: Copy,
		F: Fn(T) -> [u8; N],
	{
		values
			.iter()
			.flat_map(|value| each(*value))
			.collect()
	}

	/// A triangle with its paint in one of the three ways a file may store a
	/// color, a second set of coordinates beside it, and normals or none.
	///
	/// @param color - the color accessor's component type and shape, and its
	/// bytes; `None` for a triangle with no color at all
	/// @param second - whether it has a second set of coordinates
	/// @param normals - whether it declares normals
	fn painted(color: Option<(u32, &str, Vec<u8>)>, second: bool, normals: bool) -> Model {
		let positions: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
		let pointing: [f32; 9] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
		let seconds: [f32; 6] = [0.5, 0.25, 0.75, 0.25, 0.5, 1.0];
		let mut buffer = bytes_of(&positions, f32::to_le_bytes);
		let mut views = vec![(0, 36)];
		let mut accessors = vec![
			"{ \"bufferView\": 0, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\" }"
				.to_owned(),
		];
		let mut attributes = vec!["\"POSITION\": 0".to_owned()];

		let mut add = |bytes: Vec<u8>, accessor: String, attribute: &str| {
			while buffer.len() % 4 != 0 {
				buffer.push(0);
			}

			let view = views.len();
			views.push((buffer.len(), bytes.len()));
			buffer.extend(bytes);
			attributes.push(format!("\"{attribute}\": {}", accessors.len()));
			accessors.push(accessor.replace("VIEW", &view.to_string()));
		};

		if let Some((component, shape, bytes)) = color {
			let normalized = component != 5126;

			add(
				bytes,
				format!(
					"{{ \"bufferView\": VIEW, \"componentType\": {component}, \"normalized\": \
					 {normalized}, \"count\": 3, \"type\": \"{shape}\" }}"
				),
				"COLOR_0",
			);
		}

		if second {
			add(
				bytes_of(&seconds, f32::to_le_bytes),
				"{ \"bufferView\": VIEW, \"componentType\": 5126, \"count\": 3, \"type\": \
				 \"VEC2\" }"
					.to_owned(),
				"TEXCOORD_1",
			);
		}

		if normals {
			add(
				bytes_of(&pointing, f32::to_le_bytes),
				"{ \"bufferView\": VIEW, \"componentType\": 5126, \"count\": 3, \"type\": \
				 \"VEC3\" }"
					.to_owned(),
				"NORMAL",
			);
		}

		let views: Vec<String> = views
			.iter()
			.map(|(start, length)| {
				format!("{{ \"buffer\": 0, \"byteOffset\": {start}, \"byteLength\": {length} }}")
			})
			.collect();
		let text = format!(
			"{{ \"asset\": {{ \"version\": \"2.0\" }}, \"buffers\": [ {{ \"byteLength\": {}, \
			 \"uri\": \"data:application/octet-stream;base64,{}\" }} ], \"bufferViews\": [ {} \
			 ], \"accessors\": [ {} ], \"meshes\": [ {{ \"name\": \"painted\", \"primitives\": \
			 [ {{ \"attributes\": {{ {} }} }} ] }} ], \"nodes\": [ {{ \"mesh\": 0 }} ] }}",
			buffer.len(),
			base64(&buffer),
			views.join(", "),
			accessors.join(", "),
			attributes.join(", ")
		);
		let file = Gltf::read(text.as_bytes(), Path::new("painted.gltf"), Path::new(""))
			.expect("the painted triangle reads");

		import(&file).expect("and imports")
	}

	#[test]
	fn a_vertex_color_is_read_whichever_of_the_three_ways_the_file_stored_it() {
		// the same three colors, as floats with an alpha, as normalized bytes
		// with none, and as normalized shorts with one
		let floats: [f32; 12] = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.25];
		let wanted = [
			[PaintVertex::WHOLE, 0, 0, PaintVertex::WHOLE],
			[0, PaintVertex::WHOLE, 0, 32768],
			[0, 0, PaintVertex::WHOLE, 16384],
		];
		let bytes: [u8; 9] = [255, 0, 0, 0, 255, 0, 0, 0, 255];
		let shorts: [u16; 12] = [65535, 0, 0, 65535, 0, 65535, 0, 32768, 0, 0, 65535, 16384];

		let as_floats =
			painted(Some((5126, "VEC4", bytes_of(&floats, f32::to_le_bytes))), false, true);
		let as_bytes = painted(Some((5121, "VEC3", bytes.to_vec())), false, true);
		let as_shorts =
			painted(Some((5123, "VEC4", bytes_of(&shorts, u16::to_le_bytes))), false, true);

		for (label, model) in
			[("floats", &as_floats), ("bytes", &as_bytes), ("shorts", &as_shorts)]
		{
			let paint = &model.meshes[0].data.paint;

			assert_eq!(model.warnings, Vec::<String>::new(), "{label}: nothing to say");
			assert_eq!(paint.len(), 3, "{label}: one entry per vertex");
			assert!(
				paint.iter().all(|entry| entry.uv2 == [0.0, 0.0]),
				"{label}: and no second set, so nought beside the color"
			);

			// three channels are opaque, whatever the other two said
			let opaque = usize::from(label == "bytes");
			let alpha = |at: usize| [wanted[at][3], PaintVertex::WHOLE][opaque];

			for (at, entry) in paint.iter().enumerate() {
				let color = [wanted[at][0], wanted[at][1], wanted[at][2], alpha(at)];

				assert_eq!(entry.color, color, "{label}: vertex {at}");
			}
		}
	}

	#[test]
	fn a_second_set_of_coordinates_comes_in_beside_a_white_vertex() {
		let model = painted(None, true, true);
		let paint = &model.meshes[0].data.paint;

		assert_eq!(model.warnings, Vec::<String>::new(), "nothing to say about it");
		assert_eq!(
			paint
				.iter()
				.map(|entry| entry.uv2)
				.collect::<Vec<_>>(),
			vec![[0.5, 0.25], [0.75, 0.25], [0.5, 1.0]],
			"each vertex's own second coordinates, as the file wrote them"
		);
		assert!(
			paint
				.iter()
				.all(|entry| entry.color == PaintVertex::PLAIN.color),
			"and white beside them, which changes nothing"
		);
	}

	#[test]
	fn a_tangent_written_beside_no_normals_is_left_out_as_the_specification_says() {
		// the triangle in the xy plane, and a tangent along a direction no
		// generator would pick for it: the flat normals made for it are not the
		// normals the tangent was measured against, so it has to go
		let positions: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
		let written: [f32; 12] = [0.6, 0.8, 0.0, 1.0, 0.6, 0.8, 0.0, 1.0, 0.6, 0.8, 0.0, 1.0];
		let mut buffer = bytes_of(&positions, f32::to_le_bytes);
		buffer.extend(bytes_of(&written, f32::to_le_bytes));

		let text = format!(
			"{{ \"asset\": {{ \"version\": \"2.0\" }}, \"buffers\": [ {{ \"byteLength\": {}, \
			 \"uri\": \"data:application/octet-stream;base64,{}\" }} ], \"bufferViews\": [ {{ \
			 \"buffer\": 0, \"byteLength\": 36 }}, {{ \"buffer\": 0, \"byteOffset\": 36, \
			 \"byteLength\": 48 }} ], \"accessors\": [ {{ \"bufferView\": 0, \"componentType\": \
			 5126, \"count\": 3, \"type\": \"VEC3\" }}, {{ \"bufferView\": 1, \
			 \"componentType\": 5126, \"count\": 3, \"type\": \"VEC4\" }} ], \"meshes\": [ {{ \
			 \"primitives\": [ {{ \"attributes\": {{ \"POSITION\": 0, \"TANGENT\": 1 }} }} ] }} \
			 ], \"nodes\": [ {{ \"mesh\": 0 }} ] }}",
			buffer.len(),
			base64(&buffer)
		);
		let file = Gltf::read(text.as_bytes(), Path::new("leaning.gltf"), Path::new(""))
			.expect("the triangle reads");
		let model = import(&file).expect("and imports");

		for vertex in &model.meshes[0].data.vertices {
			assert!(
				!vertex
					.tangent_axis()
					.abs_diff_eq(Vec3::new(0.6, 0.8, 0.0), 1.0e-3),
				"the file's tangent is not the one the vertex carries: {:?}",
				vertex.tangent
			);
			assert!(
				vertex
					.tangent_axis()
					.dot(Vec3::from_array(vertex.normal))
					.abs() < 1.0e-5,
				"and the one it carries lies in the flat face: {:?}",
				vertex.tangent
			);
		}
	}

	#[test]
	fn a_primitive_with_neither_carries_no_paint_at_all() {
		let model = painted(None, false, true);

		assert!(model.meshes[0].data.paint.is_empty(), "no block, rather than a white one");
	}

	#[test]
	fn a_primitive_with_no_normals_still_imports_its_paint_whole() {
		let floats: [f32; 12] = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
		let model =
			painted(Some((5126, "VEC4", bytes_of(&floats, f32::to_le_bytes))), true, false);
		let data = &model.meshes[0].data;

		assert_eq!(model.warnings, Vec::<String>::new(), "nothing to say about it");
		assert_eq!(data.paint.len(), data.vertices.len(), "a block as long as the vertices");
		assert!(data.paint_fits(), "which is the only shape a block has");
	}

	#[test]
	fn flattening_copies_every_block_of_a_vertex_with_it() {
		// two triangles sharing an edge: four vertices, and the two on the edge
		// belong to both, so flattening makes six and copies those two
		let mut data = MeshData {
			vertices: [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0)]
				.iter()
				.map(|at| MeshVertex::new(*at, Vec3::Z, Vec2::ZERO))
				.collect(),
			indices: vec![0, 1, 2, 2, 1, 3],
			paint: (0..4_u8)
				.map(|at| {
					PaintVertex::new(Vec4::splat(f32::from(at) / 4.0), Vec2::splat(f32::from(at)))
				})
				.collect(),
			skin: (0..4_u16).map(SkinVertex::rigid).collect(),
			..MeshData::default()
		};

		flatten(&mut data);

		let order = [0_u8, 1, 2, 2, 1, 3];

		assert_eq!(data.vertices.len(), 6, "a vertex a corner");
		assert_eq!(
			data.paint
				.iter()
				.map(|entry| entry.uv2[0])
				.collect::<Vec<_>>(),
			order.map(f32::from).to_vec(),
			"each corner's paint is the paint of the vertex it was copied from"
		);
		assert_eq!(
			data.skin
				.iter()
				.map(|entry| entry.bones[0])
				.collect::<Vec<_>>(),
			order.map(u16::from).to_vec(),
			"and so are its bones, which used to be read after the copying and fell out of step"
		);
		assert!(data.paint_fits() && data.skin_fits(), "both blocks as long as the vertices");
	}

	#[test]
	fn a_mirrored_copy_keeps_the_paint_of_the_piece_it_was_copied_from() {
		let data = MeshData {
			vertices: vec![MeshVertex::default(); 3],
			indices: vec![0, 1, 2],
			paint: vec![
				PaintVertex::new(Vec4::new(1.0, 0.0, 0.0, 1.0), Vec2::ZERO),
				PaintVertex::PLAIN,
				PaintVertex::new(Vec4::ZERO, Vec2::ONE),
			],
			..MeshData::default()
		};

		let turned = turn_around(&data);

		assert_eq!(turned.indices, vec![0, 2, 1], "the winding turned");
		assert_eq!(turned.paint, data.paint, "and the paint is still each vertex's own");
	}

	/// The lights a lamp test declares: a spot, a point, and a sun.
	const LIGHTS: &str = "\"extensionsUsed\": [ \"KHR_lights_punctual\" ], \"extensions\": { \
	                      \"KHR_lights_punctual\": { \"lights\": [ { \"type\": \"spot\", \
	                      \"intensity\": 900 }, { \"type\": \"point\" }, { \"type\": \
	                      \"directional\" } ] } }";

	/// A document with those lights, one mesh, and these nodes under these
	/// roots.
	fn lit(nodes: &str, roots: &str) -> Model {
		scene(&format!(
			"{LIGHTS}, \"meshes\": [ {{ \"primitives\": [ {{ \"attributes\": {{ \"POSITION\": 0 \
			 }} }} ] }} ], \"nodes\": [ {nodes} ], \"scenes\": [ {{ \"nodes\": [ {roots} ] }} ]"
		))
	}

	/// The one lamp called this.
	fn lamp<'a>(model: &'a Model, name: &str) -> &'a Lamp {
		model
			.lamps
			.iter()
			.find(|lamp| lamp.name == name)
			.unwrap_or_else(|| panic!("no lamp called {name}: {:?}", model.lamps))
	}

	/// A turn of twenty degrees about y and a tilt of thirty-five down about x,
	/// xyzw, as a node writes one.
	const TURNED: &str = "[ -0.2961374, 0.16561121, 0.052217014, 0.93922785 ]";

	#[test]
	fn a_lamp_stands_where_its_node_stands_and_faces_the_way_it_faces() {
		let model = lit(
			&format!(
				"{{ \"name\": \"cone\", \"translation\": [ 1, 2, 3 ], \"rotation\": {TURNED}, \
				 \"extensions\": {{ \"KHR_lights_punctual\": {{ \"light\": 0 }} }} }}"
			),
			"0",
		);
		let cone = lamp(&model, "cone");

		assert_eq!(cone.light.kind, LightKind::Spot, "the spot it names");
		assert!(
			cone.transform
				.position
				.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-6),
			"{cone:?}"
		);
		assert!(
			cone.transform.rotation.abs_diff_eq(
				Quat::from_xyzw(-0.296_137_4, 0.165_611_21, 0.052_217_014, 0.939_227_9),
				1e-6
			),
			"turned as the node is: {cone:?}"
		);
		assert_eq!(cone.transform.scale, Vec3::ONE, "with no scale of its own");
		assert!(model.placements.is_empty(), "and nothing drawn, the node having no mesh");
	}

	#[test]
	fn a_mirrored_lamp_points_down_its_nodes_own_minus_z_and_leans_the_other_way() {
		// the one case the rotation alone might get wrong: a node mirrored
		// along z, whose -z is the world's +z. The lamp is right when its
		// rotation's -z is the node's whole matrix's -z made one long.
		let model = lit(
			&format!(
				"{{ \"name\": \"upright\", \"rotation\": {TURNED}, \"extensions\": {{ \
				 \"KHR_lights_punctual\": {{ \"light\": 0 }} }} }}, {{ \"name\": \"mirrored\", \
				 \"rotation\": {TURNED}, \"scale\": [ 1, 1, -1 ], \"extensions\": {{ \
				 \"KHR_lights_punctual\": {{ \"light\": 0 }} }} }}"
			),
			"0, 1",
		);
		let turn = Quat::from_xyzw(-0.296_137_4, 0.165_611_21, 0.052_217_014, 0.939_227_9);
		let matrix =
			Mat4::from_scale_rotation_translation(Vec3::new(1.0, 1.0, -1.0), turn, Vec3::ZERO);
		let wanted = matrix.transform_vector3(-Vec3::Z).normalize();
		let mirrored = lamp(&model, "mirrored").transform;
		let upright = lamp(&model, "upright").transform;

		assert!(
			(mirrored.rotation * -Vec3::Z).abs_diff_eq(wanted, 1e-5),
			"down the matrix's -z: {} against {wanted}",
			mirrored.rotation * -Vec3::Z
		);
		assert_eq!(mirrored.scale, Vec3::ONE, "and the mirror is not carried as a scale");
		assert!(
			(upright.rotation * -Vec3::Z).abs_diff_eq(-wanted, 1e-5),
			"the unmirrored one points the other way along the same line"
		);
	}

	#[test]
	fn a_lamp_under_a_scaled_parent_moves_with_it_and_takes_none_of_the_scale() {
		let model = lit(
			"{ \"name\": \"stand\", \"scale\": [ 2, 3, 2 ], \"children\": [ 1 ] }, { \"name\": \
			 \"bulb\", \"translation\": [ 0, 1, 0 ], \"extensions\": { \"KHR_lights_punctual\": \
			 { \"light\": 1 } } }",
			"0",
		);
		let bulb = lamp(&model, "bulb");

		assert!(
			bulb.transform
				.position
				.abs_diff_eq(Vec3::new(0.0, 3.0, 0.0), 1e-6),
			"{bulb:?}"
		);
		assert_eq!(bulb.transform.scale, Vec3::ONE, "{bulb:?}");
		assert_eq!(bulb.light.kind, LightKind::Point, "the point it names");
		assert!(
			model.warnings.len() == 1 && model.warnings[0].contains("directional"),
			"nothing to say but about the sun the fixture declares: {:?}",
			model.warnings
		);
	}

	#[test]
	fn a_node_with_a_mesh_and_a_lamp_stands_both_under_two_names() {
		let model = lit(
			"{ \"name\": \"Lantern\", \"mesh\": 0, \"extensions\": { \"KHR_lights_punctual\": { \
			 \"light\": 1 } } }",
			"0",
		);

		assert_eq!(model.placements.len(), 1, "the glass");
		assert_eq!(model.placements[0].name, "lantern", "which keeps the node's name");
		assert_eq!(model.lamps.len(), 1, "and the flame");
		assert_eq!(model.lamps[0].name, "lantern_1", "numbered, names being one list");
	}

	#[test]
	fn a_sun_stands_nowhere_and_a_light_that_is_not_there_is_said() {
		let model = lit(
			"{ \"name\": \"sky\", \"extensions\": { \"KHR_lights_punctual\": { \"light\": 2 } } \
			 }, { \"name\": \"lost\", \"extensions\": { \"KHR_lights_punctual\": { \"light\": 9 \
			 } } }",
			"0, 1",
		);

		assert!(model.lamps.is_empty(), "neither stands: {:?}", model.lamps);
		assert_eq!(model.warnings.len(), 2, "one word each: {:?}", model.warnings);
		assert!(
			model
				.warnings
				.iter()
				.any(|said| said.contains("directional"))
				&& model
					.warnings
					.iter()
					.any(|said| said.contains("light 9")),
			"{:?}",
			model.warnings
		);
	}

	#[test]
	fn a_camera_in_the_file_is_named_rather_than_passed_over() {
		// a scene exported whole rather than a prop carries the camera it was
		// framed with, and every one of them used to go by without a word
		let model = scene(
			"\"cameras\": [ { \"type\": \"perspective\", \"perspective\": { \"yfov\": 1.0, \
			 \"znear\": 0.1 } } ]",
		);

		assert_eq!(model.warnings.len(), 1, "one line: {:?}", model.warnings);
		assert!(
			model.warnings[0].contains("1 camera") && !model.warnings[0].contains("cameras"),
			"counted, and in English: {:?}",
			model.warnings
		);
	}
}
