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

/// How small a triangle's UV area has to be before it is treated as collapsed.
///
/// The area is in texture units squared, and a triangle covering a hundredth of
/// a map on each side has an area around `1e-4`, so this is six orders of
/// magnitude below anything an unwrap means. Below it the reciprocal is what
/// the answer would be made of, and the answer is then noise.
const DEGENERATE_UV: f32 = 1.0e-10;

/// One vertex of a mesh, as the vertex stage reads it.
///
/// No color: that comes per entity, so one cube mesh serves every cube in the
/// world.
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

	/// The direction `u` grows in, in the mesh's own space, and a sign.
	///
	/// `xyz` is the unit tangent and `w` is `+1` or `-1`, so the third axis of
	/// the frame is `cross(normal, tangent) * w`. Storing the sign rather than
	/// the bitangent costs eight bytes less and is what every exchange format
	/// settled on, for the same reason: the two are the same information unless
	/// the frame is skewed, and a skewed frame is a broken unwrap rather than
	/// something to carry.
	///
	/// Zero until [`tangents`] has been over the mesh. Everything that builds
	/// geometry calls it, and a mesh read from a `.cmesh` was built by
	/// something that did.
	pub tangent: [f32; 4],
}

impl MeshVertex {
	/// A vertex from its three attributes, with no tangent.
	///
	/// The tangent follows from the whole mesh rather than from one vertex, so
	/// it is filled by [`tangents`] once every triangle is known.
	#[must_use]
	pub fn new(position: Vec3, normal: Vec3, uv: Vec2) -> Self {
		Self {
			position: position.to_array(),
			normal: normal.to_array(),
			uv: uv.to_array(),
			tangent: [0.0; 4],
		}
	}

	/// The tangent frame's first axis.
	#[must_use]
	pub fn tangent_axis(&self) -> Vec3 {
		Vec3::new(self.tangent[0], self.tangent[1], self.tangent[2])
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

/// Fills in every vertex's tangent frame from the way the mesh is unwrapped.
///
/// A tangent is a property of the *triangle*, not of the vertex: it is the
/// direction the texture's `u` axis points once the triangle is laid out in the
/// mesh's own space. So every triangle is measured, each of its three vertices
/// is given a share, and what a vertex ends up with is the sum over every
/// triangle it belongs to. Vertices that a face does not share with its
/// neighbors therefore get that face's own frame, and vertices that are shared
/// get the average - which is exactly the split the normals already have, and
/// for the same reason.
///
/// The result is made orthogonal to the vertex's normal before it is stored,
/// because the shader builds the frame around the normal and a tangent leaning
/// off it would tilt the whole thing.
///
/// Two ways this can have no answer, and both fall back rather than refuse:
/// a triangle whose unwrap collapsed to a line contributes nothing, and a
/// vertex left with nothing at all is given any frame perpendicular to its
/// normal. A wrong tangent on a surface with no normal map costs nothing; a
/// `NaN` costs the whole triangle.
///
/// @param mesh - read for its triangles, written for its tangents
pub fn tangents(mesh: &mut MeshData) {
	let mut along_u = vec![Vec3::ZERO; mesh.vertices.len()];
	let mut along_v = vec![Vec3::ZERO; mesh.vertices.len()];

	for triangle in mesh.indices.chunks_exact(3) {
		spread(mesh, triangle, &mut along_u, &mut along_v);
	}

	for (index, vertex) in mesh.vertices.iter_mut().enumerate() {
		let normal = Vec3::from_array(vertex.normal).normalize_or(Vec3::Y);
		let axis = orthogonal(normal, along_u[index]);

		// which way the third axis turns. The bitangent the unwrap implies is
		// `along_v`; the one the shader will build is `cross(normal, axis)`.
		// When they point apart the unwrap is mirrored, and the sign is how the
		// shader is told.
		let sign = if normal.cross(axis).dot(along_v[index]) < 0.0 {
			-1.0
		} else {
			1.0
		};

		vertex.tangent = axis.extend(sign).to_array();
	}
}

/// Adds one triangle's frame to each of its three vertices.
///
/// @param mesh - where the corners are read from
/// @param triangle - three indices into its vertices
/// @param along_u - the running sum of tangents, one per vertex
/// @param along_v - the running sum of bitangents, one per vertex
fn spread(mesh: &MeshData, triangle: &[u32], along_u: &mut [Vec3], along_v: &mut [Vec3]) {
	let mut slots = [0_usize; 3];
	for (slot, index) in slots.iter_mut().zip(triangle) {
		match usize::try_from(*index) {
			| Ok(at) if at < mesh.vertices.len() => *slot = at,
			| _ => return,
		}
	}

	let position = |slot: usize| Vec3::from_array(mesh.vertices[slots[slot]].position);
	let texcoord = |slot: usize| Vec2::from_array(mesh.vertices[slots[slot]].uv);

	let (edge_a, edge_b) = (position(1) - position(0), position(2) - position(0));
	let (uv_a, uv_b) = (texcoord(1) - texcoord(0), texcoord(2) - texcoord(0));

	// twice the signed area of the triangle as it lies on the texture. Zero
	// when the unwrap put its three corners on one line, and there is then no
	// direction for `u` to grow in.
	let area = uv_a.x.mul_add(uv_b.y, -(uv_b.x * uv_a.y));
	if area.abs() < DEGENERATE_UV {
		return;
	}

	let scale = 1.0 / area;
	let u = (edge_a * uv_b.y - edge_b * uv_a.y) * scale;
	let v = (edge_b * uv_a.x - edge_a * uv_b.x) * scale;

	if !(u.is_finite() && v.is_finite()) {
		return;
	}

	for slot in slots {
		along_u[slot] += u;
		along_v[slot] += v;
	}
}

/// The part of an accumulated tangent that is perpendicular to the normal.
///
/// @param normal - the vertex's own unit normal
/// @param accumulated - the sum of every triangle's tangent at this vertex
/// @return a unit vector across the normal, arbitrary if there was no sum
fn orthogonal(normal: Vec3, accumulated: Vec3) -> Vec3 {
	(accumulated - normal * normal.dot(accumulated))
		.try_normalize()
		.unwrap_or_else(|| normal.any_orthonormal_vector())
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

	tangents(&mut mesh);

	mesh
}

/// A square one unit on a side in the xz plane, facing up.
#[must_use]
pub fn quad() -> MeshData {
	let mut mesh = MeshData::default();
	push_face(&mut mesh, Vec3::ZERO, Vec3::X * 0.5, Vec3::NEG_Z * 0.5, Vec3::Y);
	tangents(&mut mesh);

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

			for corner in [top, top + 1, bottom, top + 1, bottom + 1, bottom] {
				mesh.indices
					.push(u32::try_from(corner).unwrap_or(0));
			}
		}
	}

	tangents(&mut mesh);

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

	/// Twice the area of a triangle, which is zero when it has no direction.
	fn wound_area(mesh: &MeshData, triangle: usize) -> f32 {
		let corner = |offset: usize| {
			let index = usize::try_from(mesh.indices[triangle * 3 + offset]).unwrap_or(0);

			Vec3::from_array(mesh.vertices[index].position)
		};

		(corner(1) - corner(0))
			.cross(corner(2) - corner(0))
			.length()
	}

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

	/// A mesh of loose triangles from `(position, uv)` corners, all facing up.
	fn loose(corners: &[(Vec3, Vec2)]) -> MeshData {
		MeshData {
			vertices: corners
				.iter()
				.map(|(position, uv)| MeshVertex::new(*position, Vec3::Y, *uv))
				.collect(),
			indices: (0..u32::try_from(corners.len()).expect("the fixture is small")).collect(),
		}
	}

	#[test]
	fn the_quad_tangent_points_the_way_its_u_axis_grows() {
		let quad = quad();

		for vertex in &quad.vertices {
			assert!(
				vertex.tangent_axis().abs_diff_eq(Vec3::X, 1.0e-5),
				"the quad's u runs along +x, got {:?}",
				vertex.tangent
			);
			assert!(
				(vertex.tangent[3] + 1.0).abs() < 1.0e-6,
				"and its v runs along +z, which is the mirrored turn: got {}",
				vertex.tangent[3]
			);
		}
	}

	#[test]
	fn every_cube_tangent_lies_flat_in_the_face_it_belongs_to() {
		for vertex in cube().vertices {
			let (normal, tangent) = (Vec3::from_array(vertex.normal), vertex.tangent_axis());

			assert!(
				(tangent.length() - 1.0).abs() < 1.0e-5,
				"a tangent is a unit vector, got {}",
				tangent.length()
			);
			assert!(
				normal.dot(tangent).abs() < 1.0e-5,
				"and lies across the normal, got {} against {normal}",
				normal.dot(tangent)
			);
			assert!(
				(vertex.tangent[3].abs() - 1.0).abs() < 1.0e-6,
				"and the sign is a sign, got {}",
				vertex.tangent[3]
			);
		}
	}

	#[test]
	fn the_spheres_tangents_run_around_it_rather_than_over_it() {
		let sphere = sphere();

		for vertex in &sphere.vertices {
			let normal = Vec3::from_array(vertex.normal);

			// the poles are where a latitude-longitude unwrap has no answer:
			// every segment of the top ring is the same point.
			if normal.y.abs() > 0.99 {
				continue;
			}

			let tangent = vertex.tangent_axis();
			let around = Vec3::new(-normal.z, 0.0, normal.x).normalize();

			assert!(
				normal.dot(tangent).abs() < 1.0e-5,
				"square with the normal, got {}",
				normal.dot(tangent)
			);
			// @note: not tighter than this, and the slack is the seam. The
			// first and last segment of a ring are the same point under two
			// texture coordinates, so each is shared by the triangles on one
			// side only and its averaged tangent sits half a segment - about
			// seven degrees - around from where the analytic one is.
			assert!(
				tangent.dot(around) > 0.98,
				"u runs around the equator rather than over the pole: got {tangent} against \
				 {around}"
			);
		}
	}

	#[test]
	fn a_nearly_collapsed_triangle_does_not_shout_down_the_ones_beside_it() {
		// two triangles sharing their first corner. The second is unwrapped
		// onto a patch a ten-millionth across, which is not collapsed and is
		// not meant either - a modeling tool that welded two texture
		// coordinates produces exactly this. Its share of the tangent is
		// divided by that patch's area, so without a floor under the divisor it
		// arrives ten million times louder than its neighbor and the shared
		// corner ends up believing it.
		let tiny = 1.0e-7;
		let mut mesh = MeshData {
			vertices: vec![
				MeshVertex::new(Vec3::ZERO, Vec3::Y, Vec2::ZERO),
				MeshVertex::new(Vec3::X, Vec3::Y, Vec2::new(1.0, 0.0)),
				MeshVertex::new(Vec3::Z, Vec3::Y, Vec2::new(0.0, 1.0)),
				MeshVertex::new(Vec3::Z, Vec3::Y, Vec2::new(tiny, 0.0)),
				MeshVertex::new(Vec3::X, Vec3::Y, Vec2::new(0.0, tiny)),
			],
			indices: vec![0, 1, 2, 0, 3, 4],
		};

		tangents(&mut mesh);

		assert!(
			mesh.vertices[0]
				.tangent_axis()
				.abs_diff_eq(Vec3::X, 1.0e-4),
			"the shared corner keeps the frame the honest triangle gave it, got {:?}",
			mesh.vertices[0].tangent
		);
	}

	#[test]
	fn a_coordinate_that_is_not_a_number_does_not_spread() {
		// nothing in the engine writes one; a file can. What matters is that
		// the corner it poisons is the only one, rather than every vertex the
		// triangle touches and then every triangle they touch.
		let mut mesh = loose(&[
			(Vec3::ZERO, Vec2::ZERO),
			(Vec3::X, Vec2::new(1.0, 0.0)),
			(Vec3::Z, Vec2::new(0.0, 1.0)),
			(Vec3::ZERO, Vec2::new(f32::NAN, 0.0)),
			(Vec3::X, Vec2::new(1.0, 0.0)),
			(Vec3::Z, Vec2::new(0.0, 1.0)),
		]);
		mesh.indices = vec![0, 1, 2, 3, 1, 2];

		tangents(&mut mesh);

		for vertex in &mesh.vertices {
			assert!(
				vertex.tangent_axis().is_finite(),
				"every frame is a real direction, got {:?}",
				vertex.tangent
			);
		}
		assert!(
			mesh.vertices[1]
				.tangent_axis()
				.abs_diff_eq(Vec3::X, 1.0e-5),
			"and the corners the good triangle shares with the bad one are unharmed, got {:?}",
			mesh.vertices[1].tangent
		);
	}

	#[test]
	fn a_mirrored_unwrap_turns_the_frame_the_other_way() {
		let (a, b, c) = (Vec3::ZERO, Vec3::X, Vec3::Z);
		let mut mesh = loose(&[
			(a, Vec2::new(0.0, 0.0)),
			(b, Vec2::new(1.0, 0.0)),
			(c, Vec2::new(0.0, 1.0)),
			(a, Vec2::new(1.0, 0.0)),
			(b, Vec2::new(0.0, 0.0)),
			(c, Vec2::new(1.0, 1.0)),
		]);

		tangents(&mut mesh);

		assert!(
			mesh.vertices[0]
				.tangent_axis()
				.abs_diff_eq(Vec3::X, 1.0e-5),
			"the first triangle's u grows along +x, got {:?}",
			mesh.vertices[0].tangent
		);
		assert!(
			mesh.vertices[3]
				.tangent_axis()
				.abs_diff_eq(Vec3::NEG_X, 1.0e-5),
			"the mirrored one's grows the other way, got {:?}",
			mesh.vertices[3].tangent
		);
		assert!(
			mesh.vertices[0].tangent[3] * mesh.vertices[3].tangent[3] < 0.0,
			"and the two frames turn opposite ways, got {} and {}",
			mesh.vertices[0].tangent[3],
			mesh.vertices[3].tangent[3]
		);
	}

	#[test]
	fn a_mesh_with_no_unwrap_at_all_still_gets_a_usable_frame() {
		let mut mesh =
			loose(&[(Vec3::ZERO, Vec2::ZERO), (Vec3::X, Vec2::ZERO), (Vec3::Z, Vec2::ZERO)]);

		tangents(&mut mesh);

		for vertex in &mesh.vertices {
			let tangent = vertex.tangent_axis();

			assert!(tangent.is_finite(), "no dividing by a collapsed unwrap, got {tangent}");
			assert!(
				(tangent.length() - 1.0).abs() < 1.0e-5,
				"still a unit vector, got {}",
				tangent.length()
			);
			assert!(
				Vec3::Y.dot(tangent).abs() < 1.0e-5,
				"still across the normal, got {}",
				Vec3::Y.dot(tangent)
			);
		}
	}

	#[test]
	fn a_tangent_is_made_square_with_the_normal_it_is_stored_beside() {
		// an unwrap that leans: u grows along x and also along y, while the
		// normal insists the surface is flat. The stored frame has to follow
		// the normal, because that is the axis the shader builds around.
		let mut mesh = loose(&[
			(Vec3::ZERO, Vec2::ZERO),
			(Vec3::new(1.0, 1.0, 0.0), Vec2::new(1.0, 0.0)),
			(Vec3::Z, Vec2::new(0.0, 1.0)),
		]);

		tangents(&mut mesh);

		for vertex in &mesh.vertices {
			assert!(
				Vec3::Y.dot(vertex.tangent_axis()).abs() < 1.0e-5,
				"the lean is projected out, got {:?}",
				vertex.tangent
			);
		}
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
	fn every_sphere_triangle_is_wound_to_face_outwards() {
		let sphere = sphere();

		let mut checked = 0;
		for triangle in 0..sphere.triangles() {
			// half of each pole band is degenerate by construction: a pole ring
			// is one point repeated once per segment, so every triangle with two
			// corners in it has no area to take a direction from. Forty-eight of
			// the seven hundred and sixty-eight. The threshold has three orders
			// of magnitude either side of it: the smallest real triangle here
			// measures about 2.5e-3 and the slivers about 1e-9 - not zero,
			// because the sine of a single-precision pi is not quite nothing, so
			// the bottom pole is a ring of radius 4e-8 rather than a point.
			if wound_area(&sphere, triangle) < 1.0e-6 {
				continue;
			}

			let wound = wound_normal(&sphere, triangle);
			let first = usize::try_from(sphere.indices[triangle * 3]).expect("the index fits");
			let declared = Vec3::from_array(sphere.vertices[first].normal);

			// the bands next to the poles are slivers, so their wound normal
			// leans a fair way off the corner's own. What matters is which side
			// of the surface it is on.
			assert!(
				wound.dot(declared) > 0.0,
				"triangle {triangle} is wound {wound}, and its corner claims {declared}"
			);
			checked += 1;
		}

		assert_eq!(checked, 720, "and the rest of them really were checked");
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
