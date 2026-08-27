//! Geometry: the vertex layout, the container it comes in, and the host's
//! registry of every mesh the renderer can draw.
//!
//! This lives in the ABI rather than in the engine because [`MeshId`] is part
//! of the boundary. The game addresses geometry by handle, and a handle is only
//! meaningful next to the table it indexes - so the table is here too, hanging
//! off [`World`](super::World) where the renderer reads it.
//!
//! The registry only ever grows. An id handed out once keeps pointing at the
//! same entry for the life of the process, and reloading an asset replaces that
//! entry's contents in place and bumps its revision. That is what lets a mesh
//! change on disk without the game noticing or the renderer being told: it
//! compares revisions and re-uploads what moved.
//!
//! Built-in primitives are seeded first, in [`MeshId`] constant order, so
//! `MeshId::CUBE` and `MeshId::QUAD` mean the same thing in every world.

use super::registry::{Entry, Registry};
use crate::{
	bytemuck::{Pod, Zeroable},
	glam::{Vec2, Vec3},
	registry_handle,
};

/// The name [`MeshId::CUBE`] is registered under.
pub const CUBE_NAME: &str = "cube";

/// The name [`MeshId::QUAD`] is registered under.
pub const QUAD_NAME: &str = "quad";

/// The name [`MeshId::SPHERE`] is registered under.
pub const SPHERE_NAME: &str = "sphere";

/// How many bands of latitude the built-in sphere has.
///
/// Sixteen by twenty-four is 768 triangles, which is fine for the handful of
/// them a scene has and coarse enough that the shading shows it is a mesh
/// rather than a shader trick.
const SPHERE_RINGS: usize = 16;

/// How many segments of longitude it has.
const SPHERE_SEGMENTS: usize = 24;

/// One vertex of a mesh, as the vertex stage reads it.
///
/// No color: that comes per entity, so one cube mesh serves every cube in the
/// world. No tangent yet - normal mapping is the next thing that will grow this
/// struct, and growing it means the asset format's version goes up with it.
///
/// @note: `#[repr(C)]` and `Pod` because this is exactly the layout the vertex
/// buffer holds and exactly the layout the asset format stores. The three are
/// the same bytes on purpose: an asset is read into memory and handed to the
/// GPU with nothing in between.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MeshVertex {
	/// Position in the mesh's own space.
	pub position: [f32; 3],

	/// Unit normal, in the same space.
	pub normal: [f32; 3],

	/// Where this vertex samples a texture.
	///
	/// Origin top left, the way every graphics API this will ever run on
	/// samples. OBJ measures `v` from the bottom, so the importer flips it -
	/// @ref `colby_asset::obj`.
	pub uv: [f32; 2],
}

impl MeshVertex {
	/// A vertex from its three attributes.
	#[must_use]
	pub fn new(position: Vec3, normal: Vec3, uv: Vec2) -> Self {
		Self {
			position: position.to_array(),
			normal: normal.to_array(),
			uv: uv.to_array(),
		}
	}
}

/// A mesh as plain data, before it reaches the GPU.
///
/// Indices are `u32` rather than `u16`. Sixteen bits is enough for everything
/// the engine generates and for very little anyone exports from a modeling
/// tool, and one index format for both is worth more than the two bytes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshData {
	/// The vertices, in no particular order.
	pub vertices: Vec<MeshVertex>,

	/// Three indices per triangle, each addressing
	/// [`vertices`](Self::vertices).
	pub indices: Vec<u32>,
}

impl MeshData {
	/// How many triangles this draws.
	#[must_use]
	pub const fn triangles(&self) -> usize { self.indices.len() / 3 }

	/// Whether there is anything to draw at all.
	#[must_use]
	pub const fn is_empty(&self) -> bool { self.indices.is_empty() }

	/// The smallest axis-aligned box containing every vertex.
	///
	/// @return `(min, max)`, both [`Vec3::ZERO`] for a mesh with no vertices
	#[must_use]
	pub fn bounds(&self) -> (Vec3, Vec3) {
		if self.vertices.is_empty() {
			return (Vec3::ZERO, Vec3::ZERO);
		}

		let mut min = Vec3::splat(f32::INFINITY);
		let mut max = Vec3::splat(f32::NEG_INFINITY);

		for vertex in &self.vertices {
			let position = Vec3::from_array(vertex.position);
			min = min.min(position);
			max = max.max(position);
		}

		(min, max)
	}

	/// Whether every index addresses a vertex that exists.
	///
	/// The renderer never checks this; the GPU would simply read whatever is at
	/// the offset. It is checked where geometry enters the process instead.
	#[must_use]
	pub fn indices_are_in_range(&self) -> bool {
		let count = u32::try_from(self.vertices.len()).unwrap_or(0);

		self.indices.iter().all(|index| *index < count)
	}
}

registry_handle! {
	/// A handle to a mesh in the world's [`Meshes`] registry.
	///
	/// No generation: entries are never removed, so a handle stays valid for
	/// the life of the process. That is deliberate - reloading an asset
	/// rewrites the entry the id already points at, which is what makes
	/// geometry hot-swappable without the game re-resolving anything.
	MeshId
}

impl MeshId {
	/// The unit cube, centered on the origin. Always registered.
	pub const CUBE: Self = Self::new(1);
	/// A unit square in the xz plane, facing up. Always registered.
	pub const QUAD: Self = Self::new(2);
	/// A ball of radius a half, at the origin. Always registered.
	///
	/// Here rather than in `assets/` because a ball is the one shape physics
	/// cannot be shown without - a box resting on a floor and a box asleep look
	/// the same, and a ball rolling does not look like anything else.
	pub const SPHERE: Self = Self::new(3);
}

/// One entry of the mesh registry.
pub type Mesh = Entry<MeshData>;

/// Every mesh the renderer can draw, addressed by [`MeshId`].
///
/// Slot zero is [`MeshId::NONE`] and is empty, so an entity that has not been
/// given a shape costs a lookup and nothing else.
#[derive(Clone, Debug)]
pub struct Meshes {
	entries: Registry<MeshData>,
}

impl Meshes {
	/// A registry holding the null mesh and the built-in primitives.
	#[must_use]
	pub fn new() -> Self {
		let mut meshes = Self {
			entries: Registry::new(MeshData::default()),
		};
		meshes.insert(CUBE_NAME, cube());
		meshes.insert(QUAD_NAME, quad());
		meshes.insert(SPHERE_NAME, sphere());

		meshes
	}

	/// Looks a mesh up by name.
	///
	/// @param name - the name it was registered under
	/// @return its handle, or [`MeshId::NONE`] if nothing answers to that name
	#[must_use]
	pub fn find(&self, name: &str) -> MeshId { MeshId::new(self.entries.find(name)) }

	/// Registers geometry under a name, replacing whatever was there.
	///
	/// A name already in the registry keeps its handle: the entry's contents
	/// are replaced and its revision goes up. That is what makes reloading an
	/// asset invisible to everything holding the id.
	///
	/// @param name - what the game will ask for
	/// @param data - the geometry
	/// @return the handle, the same one as last time if the name is known
	pub fn insert(&mut self, name: &str, data: MeshData) -> MeshId {
		MeshId::new(self.entries.insert(name, data))
	}

	/// One mesh, by handle.
	#[must_use]
	pub fn get(&self, id: MeshId) -> Option<&Mesh> { self.entries.entry(id.index()) }

	/// How many meshes there are, counting the null one.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Always `false`: slot zero always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// Every mesh, in slot order, starting with the null one.
	pub fn iter(&self) -> impl Iterator<Item = &Mesh> { self.entries.iter() }
}

impl Default for Meshes {
	fn default() -> Self { Self::new() }
}

/// A face as `(normal, right, up)`, each a unit vector.
///
/// `right` crossed with `up` gives `normal`, which is what makes the corner
/// order below come out counter-clockwise seen from outside.
type Face = ([f32; 3], [f32; 3], [f32; 3]);

/// The six faces of a cube.
const CUBE_FACES: [Face; 6] = [
	([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
	([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
	([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
	([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
	([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
	([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
];

/// The corners of a face, in counter-clockwise order, as multiples of a half
/// edge along `right` and `up`.
const FACE_CORNERS: [(f32, f32); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

/// A cube one unit on a side, centered on the origin.
///
/// Every face is wound counter-clockwise seen from outside, and every vertex
/// carries the flat normal of the face it belongs to. Flat rather than smooth
/// because a cube with averaged normals looks like a bad sphere.
#[must_use]
pub fn cube() -> MeshData {
	let mut mesh = MeshData {
		vertices: Vec::with_capacity(CUBE_FACES.len() * FACE_CORNERS.len()),
		indices: Vec::with_capacity(CUBE_FACES.len() * 6),
	};

	for (normal, right, up) in CUBE_FACES {
		let (normal, right, up) =
			(Vec3::from_array(normal), Vec3::from_array(right), Vec3::from_array(up));

		push_face(&mut mesh, normal * 0.5, right * 0.5, up * 0.5, normal);
	}

	mesh
}

/// A square one unit on a side in the xz plane, facing up.
#[must_use]
pub fn quad() -> MeshData {
	let mut mesh = MeshData::default();
	push_face(&mut mesh, Vec3::ZERO, Vec3::X * 0.5, Vec3::NEG_Z * 0.5, Vec3::Y);

	mesh
}

/// A ball of radius a half, centered on the origin.
///
/// A plain latitude-longitude sphere: every ring but the poles is a band of
/// quadrilaterals, and the normal at a point on a unit sphere is the point
/// itself, which is what makes this shorter than a cube.
#[must_use]
pub fn sphere() -> MeshData {
	let mut mesh = MeshData {
		vertices: Vec::with_capacity((SPHERE_RINGS + 1) * (SPHERE_SEGMENTS + 1)),
		indices: Vec::with_capacity(SPHERE_RINGS * SPHERE_SEGMENTS * 6),
	};

	let rings = fraction(SPHERE_RINGS);
	let segments = fraction(SPHERE_SEGMENTS);

	for ring in 0..=SPHERE_RINGS {
		let down = fraction(ring) / rings;
		let angle = down * core::f32::consts::PI;
		let (radius, height) = (angle.sin(), angle.cos());

		for segment in 0..=SPHERE_SEGMENTS {
			let around = fraction(segment) / segments;
			let turn = around * core::f32::consts::TAU;
			let normal = Vec3::new(radius * turn.cos(), height, radius * turn.sin());

			mesh.vertices
				.push(MeshVertex::new(normal * 0.5, normal, Vec2::new(around, down)));
		}
	}

	let stride = SPHERE_SEGMENTS + 1;
	for ring in 0..SPHERE_RINGS {
		for segment in 0..SPHERE_SEGMENTS {
			let top = ring * stride + segment;
			let bottom = top + stride;

			for corner in [top, bottom, top + 1, top + 1, bottom, bottom + 1] {
				mesh.indices
					.push(u32::try_from(corner).unwrap_or(0));
			}
		}
	}

	mesh
}

/// A small count as a float, for the sphere's parameters.
fn fraction(count: usize) -> f32 { f32::from(u16::try_from(count).unwrap_or(u16::MAX)) }

/// Adds one flat quadrilateral face to a mesh.
///
/// @param mesh - what to add to
/// @param center - the middle of the face
/// @param right - half an edge along the face's first axis
/// @param up - half an edge along its second
/// @param normal - the direction the face points
fn push_face(mesh: &mut MeshData, center: Vec3, right: Vec3, up: Vec3, normal: Vec3) {
	let Ok(base) = u32::try_from(mesh.vertices.len()) else {
		return;
	};

	for (across, along) in FACE_CORNERS {
		// the face's own corners map to the whole of a texture, with v counted
		// downwards so that `up` on the face is the top of the image.
		let uv = Vec2::new(across.mul_add(0.5, 0.5), along.mul_add(-0.5, 0.5));

		mesh.vertices
			.push(MeshVertex::new(center + right * across + up * along, normal, uv));
	}

	mesh.indices
		.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[cfg(test)]
mod tests {
	use super::*;

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
	fn the_cube_has_six_faces_of_two_triangles() {
		let cube = cube();

		assert_eq!(cube.vertices.len(), 24, "four vertices per face, not shared between faces");
		assert_eq!(cube.indices.len(), 36, "two triangles per face");
		assert_eq!(cube.triangles(), 12, "which is twelve triangles");
	}

	#[test]
	fn every_cube_triangle_is_wound_to_face_outwards() {
		let cube = cube();

		for triangle in 0..cube.triangles() {
			let wound = wound_normal(&cube, triangle);
			let first = usize::try_from(cube.indices[triangle * 3]).expect("the index fits");
			let declared = Vec3::from_array(cube.vertices[first].normal);

			assert!(
				wound.dot(declared) > 0.99,
				"triangle {triangle} is wound {wound}, but its vertices claim {declared}"
			);
		}
	}

	#[test]
	fn every_cube_vertex_sits_on_the_surface_of_a_unit_cube() {
		for vertex in cube().vertices {
			let position = Vec3::from_array(vertex.position);

			assert!(
				(position.abs().max_element() - 0.5).abs() < 1.0e-6,
				"{position} is not on the surface of a cube one unit across"
			);
		}
	}

	#[test]
	fn the_quad_faces_up() {
		let quad = quad();

		assert_eq!(quad.vertices.len(), 4, "one face");
		assert_eq!(quad.indices.len(), 6, "two triangles");

		for triangle in 0..2 {
			assert!(
				wound_normal(&quad, triangle).dot(Vec3::Y) > 0.99,
				"a floor is only a floor if it is wound facing up"
			);
		}

		for vertex in quad.vertices {
			assert!(vertex.position[1].abs() < 1.0e-6, "and flat in the xz plane");
		}
	}

	#[test]
	fn every_face_of_a_primitive_maps_to_a_whole_texture() {
		for mesh in [cube(), quad()] {
			for face in mesh.vertices.chunks(4) {
				let mut corners: Vec<[f32; 2]> = face.iter().map(|vertex| vertex.uv).collect();
				corners.sort_by(|left, right| left.partial_cmp(right).expect("no NaN"));

				assert_eq!(
					corners,
					vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
					"a face's four corners are the four corners of the image"
				);
			}
		}
	}

	#[test]
	fn the_top_of_a_face_is_the_top_of_its_texture() {
		// v counts down. The quad's corners are laid out along `up`, which is
		// -z for a floor, so the corner furthest along it must be v = 0.
		let quad = quad();
		let top = quad
			.vertices
			.iter()
			.find(|vertex| vertex.uv[1] < 0.5)
			.expect("some corner is at the top");

		assert!(
			top.position[2] < 0.0,
			"the -z edge of the floor is the top of the image, got {:?}",
			top.position
		);
	}

	#[test]
	fn a_cube_measures_one_unit_in_every_direction() {
		let (min, max) = cube().bounds();

		assert!(min.abs_diff_eq(Vec3::splat(-0.5), 1.0e-6), "the low corner, got {min}");
		assert!(max.abs_diff_eq(Vec3::splat(0.5), 1.0e-6), "the high corner, got {max}");
	}

	#[test]
	fn an_empty_mesh_has_a_box_rather_than_an_infinite_one() {
		let (min, max) = MeshData::default().bounds();

		assert_eq!(min, Vec3::ZERO, "nothing in it, so nothing to bound");
		assert_eq!(max, Vec3::ZERO, "and the box is a point rather than an infinity");
	}

	#[test]
	fn a_new_registry_answers_to_the_built_in_names() {
		let meshes = Meshes::new();

		assert_eq!(meshes.find(CUBE_NAME), MeshId::CUBE, "the cube keeps its constant");
		assert_eq!(meshes.find(QUAD_NAME), MeshId::QUAD, "and so does the quad");
		assert_eq!(meshes.find(SPHERE_NAME), MeshId::SPHERE, "and so does the sphere");
		assert_eq!(meshes.find("crystal"), MeshId::NONE, "an unknown name resolves to nothing");
		assert_eq!(meshes.find(""), MeshId::NONE, "and so does the empty one");
		assert_eq!(meshes.len(), 4, "null, cube, quad, sphere");
	}

	#[test]
	fn the_built_in_sphere_is_a_ball_of_radius_a_half() {
		let ball = sphere();
		let (low, high) = ball.bounds();

		assert!(!ball.is_empty(), "it has triangles");
		assert!(
			low.abs_diff_eq(Vec3::splat(-0.5), 1.0e-4),
			"reaching a half in every direction, got {low}"
		);
		assert!(high.abs_diff_eq(Vec3::splat(0.5), 1.0e-4), "and a half the other way");

		for vertex in &ball.vertices {
			let position = Vec3::from_array(vertex.position);
			let normal = Vec3::from_array(vertex.normal);

			assert!(
				(position.length() - 0.5).abs() < 1.0e-4,
				"every vertex is on the sphere, got {position}"
			);
			assert!(
				(normal.length() - 1.0).abs() < 1.0e-4 || position.length() < 1.0e-4,
				"and its normal is a unit vector, got {normal}"
			);
		}
	}

	#[test]
	fn the_null_mesh_draws_nothing() {
		let meshes = Meshes::new();
		let none = meshes
			.get(MeshId::NONE)
			.expect("slot zero always exists");

		assert!(none.value().is_empty(), "the null mesh has no triangles");
		assert!(!MeshId::NONE.is_some(), "and its handle knows it is null");
	}

	#[test]
	fn a_new_name_takes_a_new_slot() {
		let mut meshes = Meshes::new();
		let id = meshes.insert("crystal", cube());

		assert_ne!(id, MeshId::NONE, "a registered mesh has a real handle");
		assert_eq!(meshes.len(), 5, "and took a slot of its own");
		assert_eq!(meshes.find("crystal"), id, "which the name now resolves to");
		assert_eq!(
			meshes.get(id).map(Mesh::revision),
			Some(0),
			"a mesh registered once has never been replaced"
		);
	}

	#[test]
	fn replacing_a_mesh_keeps_its_handle_and_bumps_its_revision() {
		let mut meshes = Meshes::new();
		let first = meshes.insert("crystal", cube());
		let second = meshes.insert("crystal", quad());

		assert_eq!(first, second, "the handle survives, which is what makes a reload invisible");
		assert_eq!(meshes.len(), 5, "and nothing was appended");
		assert_eq!(
			meshes.get(first).map(Mesh::revision),
			Some(1),
			"but the revision moved, which is how the renderer notices"
		);
		assert_eq!(
			meshes
				.get(first)
				.map(|mesh| mesh.value().triangles()),
			Some(2),
			"and the geometry really is the new one"
		);
	}

	#[test]
	fn a_built_in_can_be_replaced_like_any_other() {
		let mut meshes = Meshes::new();
		let id = meshes.insert(CUBE_NAME, quad());

		assert_eq!(id, MeshId::CUBE, "an asset named `cube` overrides the primitive");
		assert_eq!(meshes.len(), 4, "without taking a new slot");
	}

	#[test]
	fn indices_are_checked_against_the_vertices_they_address() {
		let mut mesh = quad();

		assert!(mesh.indices_are_in_range(), "a generated mesh addresses its own vertices");

		mesh.indices[0] = 99;

		assert!(!mesh.indices_are_in_range(), "and one past the end is caught");
	}
}
