//! Turning blocks into a mesh.
//!
//! **Blocks are the editing form and a mesh is the shipping form**, which is
//! the verdict of every engine read for the block tool: Unreal kept blocks in
//! the level and marked the operations that built one deprecated, Godot's own
//! documentation calls its shape tree prototyping and ships a bake beside it,
//! and s&box keeps a mesh in the scene and no blocks at all. @ref
//! `colby-block-references`.
//!
//! **What forces it here is not the frame.** A room of three hundred blocks
//! costs about three percent of a frame at sixty, which is affordable - and it
//! spends a third of [`MAX_ENTITIES`](colby_core::abi::entity::MAX_ENTITIES)
//! and a third of [`MAX_BODIES`](colby_core::abi::physics::MAX_BODIES), which
//! is not. A twenty-by-twenty room with a ceiling is eleven hundred blocks and
//! does not fit in the world at all. Baking is what makes a level buildable,
//! and the milliseconds are beside the point.
//!
//! **A face nobody can see is not written.** Two blocks side by side put two
//! faces in the same place pointing opposite ways, and both are dropped - which
//! for a wall of blocks is most of the triangles. The test is that the faces
//! *coincide*, not that the blocks are neighbors, so it needs no notion of
//! adjacency and works whatever the blocks are sized.
//!
//! **This module does the world and nothing else.** The meshes it builds are
//! registered under `maps/<name>/<material>` and the files are written by the
//! runner, which is asked with a console line - because a name crosses that
//! boundary and a mesh does not have to: the registry is already shared.

use std::collections::HashMap;

use colby_core::{
	abi::{
		Body, BodyKind, MaterialId, MeshId, Renderable, Shape, Transform, World,
		mesh::{MeshData, MeshVertex},
	},
	debug,
	glam::{Vec2, Vec3},
	unwrap,
};

use crate::select::{self, Pick};

/// How finely two faces have to agree before they are the same face.
///
/// A thousandth of a unit. Blocks land on a grid and their faces are therefore
/// exactly on top of one another or nowhere near, so this only has to be
/// smaller than the smallest grid anybody would use and larger than the error
/// in rotating a corner.
const SAME: f32 = 1.0e-3;

/// The six faces of a box: the way it points, and the two ways across it.
///
/// The same table [`colby_core::abi::mesh::cube`] is built from, and in the
/// same order, so a baked face and a cube's face are wound the same way round.
const FACES: [(Vec3, Vec3, Vec3); 6] = [
	(Vec3::X, Vec3::NEG_Z, Vec3::Y),
	(Vec3::NEG_X, Vec3::Z, Vec3::Y),
	(Vec3::Y, Vec3::X, Vec3::NEG_Z),
	(Vec3::NEG_Y, Vec3::X, Vec3::Z),
	(Vec3::Z, Vec3::X, Vec3::Y),
	(Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
];

/// The corners of a face, as multiples of a half edge across and along.
const CORNERS: [(f32, f32); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

/// One face of one block, in world space.
struct Side {
	/// Where its middle is, which is what decides whether another face is it.
	middle: Vec3,

	/// Which way it points.
	normal: Vec3,

	/// Its four corners, counter-clockwise seen from outside.
	corners: [Vec3; 4],

	/// What the block it belongs to is made of.
	material: MaterialId,
}

/// What one bake came to.
pub(crate) struct Baked {
	/// The entities it made, one per material.
	pub(crate) made: Vec<Pick>,

	/// How many blocks went into it.
	pub(crate) blocks: usize,

	/// How many triangles came out.
	pub(crate) triangles: usize,

	/// How many faces were dropped for being buried.
	pub(crate) buried: usize,
}

/// Bakes blocks into one mesh per material.
///
/// **What counts as a block**: an entity drawing the built-in cube. Not the
/// name it was given, which a person may change and often should, and not a
/// flag, which would be a field on every entity in the world for the sake of
/// this one gesture.
///
/// The blocks are gone afterwards, and so are the bodies that drove them; what
/// stands in their place is one entity per material with a mesh collider under
/// it. Godot leaves its shape tree beside the bake and colby does not, for the
/// reason the world has: a room kept twice is a room that spends its entity
/// budget twice.
///
/// @param world - the world to bake in
/// @param picks - what is selected; empty bakes every block there is
/// @param name - what to call the meshes, `room` for `maps/room/default`
/// @return what was made, and what it came to
pub(crate) fn bake(world: &mut World, picks: &[Pick], name: &str) -> Baked {
	let blocks = gathered(world, picks);
	let mut baked = Baked {
		made: Vec::new(),
		blocks: blocks.len(),
		triangles: 0,
		buried: 0,
	};

	if blocks.is_empty() {
		return baked;
	}

	let (sides, buried) = standing(&blocks);

	baked.buried = buried;

	// the blocks go first, so that the entities and bodies the meshes need are
	// slots the blocks have just given back rather than slots on top of them.
	// A room that is a third of the world cannot be doubled even for a frame.
	let gone = select::delete(
		world,
		&blocks
			.iter()
			.map(|(pick, ..)| *pick)
			.collect::<Vec<_>>(),
	);

	debug!(
		entities = gone.entities,
		bodies = gone.bodies,
		"the blocks a bake is made of are gone"
	);

	for (material, data) in grouped(sides) {
		baked.triangles += data.triangles();

		let under = format!("maps/{name}/{}", plain(world, material));
		let mesh = world.meshes.insert(&under, data);
		let entity = world.entities.spawn();

		if !entity.is_some() {
			continue;
		}

		world
			.entities
			.set_renderable(entity, Renderable::of(mesh, material, Vec3::ONE));
		world
			.entities
			.set_name(entity, &plain(world, material));

		// **one body for the whole of it.** A mesh shape is baked into a
		// collision mesh once, when the solver first sees the body, which is
		// exactly right for geometry that will never move again - and it is
		// what turns three hundred bodies into one.
		let body = world.bodies.spawn(
			Body::new(BodyKind::Static, Shape::mesh(mesh), Transform::IDENTITY).driving(entity),
		);

		if body.is_some() {
			world
				.bodies
				.set_name(body, &plain(world, material));
		}

		baked.made.push(Pick::Entity(entity));
	}

	baked
}

/// The blocks a bake is about: what is selected, or every one of them.
fn gathered(world: &World, picks: &[Pick]) -> Vec<(Pick, Transform, MaterialId)> {
	let looking: Vec<Pick> = if picks.is_empty() {
		world
			.entities
			.iter()
			.map(|(id, ..)| Pick::Entity(id))
			.collect()
	} else {
		picks.to_vec()
	};

	looking
		.into_iter()
		.filter_map(|pick| {
			let Pick::Entity(id) = pick else {
				return None;
			};
			let look = world.entities.renderable(id)?;

			if look.mesh != MeshId::CUBE {
				return None;
			}

			Some((pick, world.entities.placed(id)?, look.material))
		})
		.collect()
}

/// Every face of every block that something else is not standing against.
///
/// @param blocks - what to take apart
/// @return the faces that survived, and how many were dropped
fn standing(blocks: &[(Pick, Transform, MaterialId)]) -> (Vec<Side>, usize) {
	let mut sides = Vec::with_capacity(blocks.len() * FACES.len());

	for (_, at, material) in blocks {
		for (normal, across, along) in FACES {
			sides.push(side(*at, normal, across, along, *material));
		}
	}

	// how many faces sit in each place. Two is a pair of blocks touching and
	// both of them are buried; three or more cannot happen with solid boxes and
	// is dropped just the same, because whatever it is nobody can see it.
	let mut seen: HashMap<[i64; 3], usize> = HashMap::new();

	for face in &sides {
		*seen.entry(key(face.middle)).or_insert(0) += 1;
	}

	let before = sides.len();

	sides.retain(|face| seen.get(&key(face.middle)) == Some(&1));

	let buried = before - sides.len();

	(sides, buried)
}

/// One face of one block, in world space.
fn side(at: Transform, normal: Vec3, across: Vec3, along: Vec3, material: MaterialId) -> Side {
	let put = |local: Vec3| at.position + at.rotation * (local * at.scale);
	let middle = normal * 0.5;
	let corners = CORNERS.map(|(x, y)| put(middle + across * 0.5 * x + along * 0.5 * y));

	Side {
		middle: put(middle),
		// exact for a box: the faces are along the axes and so is the scale, so
		// nothing here needs the inverse transpose a sheared normal would
		normal: (at.rotation * normal).normalize_or_zero(),
		corners,
		material,
	}
}

/// A place, rounded to something two faces can agree on.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "a world is a few thousand units across and a thousandth of one fits in an i64many \
	          times over"
)]
fn key(at: Vec3) -> [i64; 3] {
	[
		(f64::from(at.x) / f64::from(SAME)).round() as i64,
		(f64::from(at.y) / f64::from(SAME)).round() as i64,
		(f64::from(at.z) / f64::from(SAME)).round() as i64,
	]
}

/// The surviving faces, gathered into one mesh per material.
///
/// In the order the materials were first met, so that two bakes of the same
/// room put the same geometry under the same name.
fn grouped(sides: Vec<Side>) -> Vec<(MaterialId, MeshData)> {
	let mut out: Vec<(MaterialId, MeshData)> = Vec::new();

	for face in sides {
		let at = match out
			.iter()
			.position(|(material, _)| *material == face.material)
		{
			| Some(at) => at,
			| None => {
				out.push((face.material, MeshData::default()));

				out.len() - 1
			},
		};

		if let Some((_, data)) = out.get_mut(at) {
			push(data, &face);
		}
	}

	for (_, data) in &mut out {
		colby_core::abi::mesh::tangents(data);
		// the second set the compiler gives the file this is written to, so
		// that a room baked here can have its light baked before its files
		// come back as meshes
		unwrap::second(data, Vec3::ONE);
	}

	out
}

/// Puts one face into a mesh.
fn push(data: &mut MeshData, face: &Side) {
	let Ok(base) = u32::try_from(data.vertices.len()) else {
		return;
	};

	for (at, corner) in face.corners.iter().enumerate() {
		let (across, along) = CORNERS[at];
		// the face's own corners map to the whole of a texture, with v counted
		// downwards so that the top of the face is the top of the image - the
		// same convention the built-in cube keeps
		let uv = Vec2::new(across.mul_add(0.5, 0.5), along.mul_add(-0.5, 0.5));

		data.vertices
			.push(MeshVertex::new(*corner, face.normal, uv));
	}

	data.indices
		.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// What to call the file a material's faces go into.
///
/// The last part of the material's name, because an asset name has slashes in
/// it and a file name may not have them mean directories here:
/// `materials/brass` becomes `brass`, and the built-in one becomes `default`.
fn plain(world: &World, material: MaterialId) -> String {
	let name = world.materials.name(material);
	let last = name.rsplit('/').next().unwrap_or(name);

	if last.is_empty() {
		"default".to_owned()
	} else {
		last.to_owned()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A world with blocks in a row along x, each one unit, touching.
	fn row(many: usize) -> (World, Vec<Pick>) {
		let mut world = World::new();
		let mut picks = Vec::new();

		for step in 0..many {
			let along = f32::from(u8::try_from(step).expect("a handful of blocks"));

			picks.extend(select::block(&mut world, Vec3::new(along, 0.0, 0.0), Some(1.0)));
		}

		(world, picks)
	}

	#[test]
	fn a_baked_room_has_the_second_set_its_file_will_come_back_with() {
		let (mut world, picks) = row(3);

		bake(&mut world, &picks, "room");

		let baked = world
			.meshes
			.get(world.meshes.find("maps/room/default"))
			.expect("the room is registered")
			.value()
			.clone();
		let mut read = colby_asset::obj::import(&colby_asset::obj::export(&baked))
			.expect("what the runner writes reads back");

		unwrap::second(&mut read, Vec3::ONE);

		assert!(baked.sheet != [0, 0], "a sheet laid out when it was baked: {:?}", baked.sheet);
		assert_eq!(read.sheet, baked.sheet, "the one the compiler lays out from the file");
		assert_eq!(read.indices, baked.indices, "over the same triangles");
		assert_eq!(read.paint, baked.paint, "and every vertex in the same place on it");
	}

	#[test]
	fn a_stretched_block_is_laid_out_as_long_as_it_is() {
		let mut world = World::new();
		let picks = select::block(&mut world, Vec3::ZERO, Some(1.0));
		let [Pick::Entity(block)] = picks[..] else {
			panic!("one block: {picks:?}");
		};

		assert!(
			world.entities.set_placed(block, Transform {
				scale: Vec3::new(4.0, 1.0, 1.0),
				..Transform::IDENTITY
			}),
			"the block is stretched"
		);
		bake(&mut world, &picks, "long");

		let baked = world
			.meshes
			.get(world.meshes.find("maps/long/default"))
			.expect("the block is registered")
			.value()
			.clone();
		let longest = (0..baked.triangles())
			.map(|triangle| {
				let spots: Vec<f32> = baked.indices[triangle * 3..triangle * 3 + 3]
					.iter()
					.map(|index| {
						let vertex = usize::try_from(*index).expect("a small mesh");

						baked.paint[vertex].uv2[0]
							* f32::from(u16::try_from(baked.sheet[0]).expect("small"))
					})
					.collect();

				spots.iter().copied().fold(f32::MIN, f32::max)
					- spots.iter().copied().fold(f32::MAX, f32::min)
			})
			.fold(0.0_f32, f32::max);

		assert!(
			(longest - 20.0).abs() < 1.0e-3,
			"a face four units long is twenty texels long, not laid square: {longest}"
		);
	}

	#[test]
	fn one_block_bakes_into_six_faces() {
		let (mut world, picks) = row(1);
		let baked = bake(&mut world, &picks, "room");

		assert_eq!(baked.blocks, 1);
		assert_eq!(baked.buried, 0, "a block on its own has nothing against it");
		assert_eq!(baked.triangles, 12, "six faces, two triangles each");
		assert_eq!(baked.made.len(), 1, "one material, one entity");
	}

	#[test]
	fn two_blocks_side_by_side_lose_the_faces_between_them() {
		// the whole reason this is worth doing: a wall of blocks is mostly
		// faces nobody can see
		let (mut world, picks) = row(2);
		let baked = bake(&mut world, &picks, "room");

		assert_eq!(baked.blocks, 2);
		assert_eq!(baked.buried, 2, "the two faces that meet, both of them");
		assert_eq!(baked.triangles, 20, "ten faces of the twelve");
	}

	#[test]
	fn a_row_of_blocks_keeps_only_its_skin() {
		let (mut world, picks) = row(10);
		let baked = bake(&mut world, &picks, "room");

		// six faces each, less two for every join
		assert_eq!(baked.buried, 18, "nine joins, two faces each");
		assert_eq!(baked.triangles, (10 * 6 - 18) * 2);
	}

	#[test]
	fn the_blocks_are_gone_and_one_thing_stands_in_their_place() {
		let (mut world, picks) = row(6);

		assert_eq!(world.entities.len(), 6, "six blocks");
		assert_eq!(world.bodies.len(), 6, "and six bodies");

		let baked = bake(&mut world, &picks, "room");

		assert_eq!(world.entities.len(), 1, "one entity for the one material");
		assert_eq!(world.bodies.len(), 1, "and one body, which is most of the point");
		assert_eq!(baked.made.len(), 1);

		let [Pick::Entity(entity)] = baked.made[..] else {
			panic!("one entity: {:?}", baked.made);
		};
		let look = world
			.entities
			.renderable(entity)
			.expect("it draws something");

		assert_ne!(look.mesh, MeshId::CUBE, "and it is not a block any more");
		assert!(
			world.meshes.find("maps/room/default").is_some(),
			"the mesh is registered under the name the files will be written from"
		);
	}

	#[test]
	fn the_body_left_behind_collides_against_the_mesh() {
		let (mut world, picks) = row(3);

		drop(bake(&mut world, &picks, "room"));

		let (_, held) = world
			.bodies
			.iter()
			.next()
			.expect("one body stands");

		assert_eq!(held.kind, BodyKind::Static, "a room does not fall");
		assert_eq!(
			held.shape.kind,
			colby_core::abi::ShapeKind::Mesh,
			"and it collides against what is drawn"
		);
		assert_eq!(held.shape.mesh, world.meshes.find("maps/room/default"));
	}

	#[test]
	fn baking_nothing_makes_nothing() {
		let mut world = World::new();
		let baked = bake(&mut world, &[], "room");

		assert_eq!(baked.blocks, 0);
		assert_eq!(baked.made.len(), 0);
		assert_eq!(world.entities.len(), 0, "and the world is untouched");
	}

	#[test]
	fn an_empty_selection_bakes_every_block_there_is() {
		let (mut world, _) = row(4);
		let baked = bake(&mut world, &[], "room");

		assert_eq!(baked.blocks, 4, "all of them, because nothing was picked");
	}

	#[test]
	fn a_thing_that_is_not_a_block_is_not_baked() {
		// a block is an entity drawing the built-in cube, and nothing else in
		// the world should be swept into a bake by being selected with one
		let (mut world, mut picks) = row(2);
		let other = world.entities.spawn();

		world
			.entities
			.set_renderable(other, Renderable::new(MeshId::SPHERE, Vec3::ONE));
		picks.push(Pick::Entity(other));

		let baked = bake(&mut world, &picks, "room");

		assert_eq!(baked.blocks, 2, "the two cubes and not the ball");
		assert!(world.entities.alive(other), "and the ball is still there afterwards");
	}

	#[test]
	fn two_materials_are_two_meshes_and_two_files() {
		let (mut world, picks) = row(2);
		let brass = world
			.materials
			.insert("materials/brass", colby_core::abi::Material::DEFAULT);

		let [Pick::Entity(first), ..] = picks[..] else {
			panic!("a block: {picks:?}");
		};
		let mut look = *world
			.entities
			.renderable(first)
			.expect("it draws");

		look.material = brass;
		world.entities.set_renderable(first, look);

		let baked = bake(&mut world, &picks, "room");

		assert_eq!(baked.made.len(), 2, "one entity per material");
		assert!(world.meshes.find("maps/room/brass").is_some(), "named by its material");
		assert!(world.meshes.find("maps/room/default").is_some());
		assert_eq!(
			baked.buried, 2,
			"and the faces between them are still buried, because a face is hidden by what \
			 stands against it whatever that is made of"
		);
	}

	#[test]
	fn two_faces_in_one_place_are_the_same_face_whatever_the_arithmetic() {
		// the whole cull is this: a place, rounded to something two faces can
		// agree on
		assert_eq!(key(Vec3::new(1.0, 2.0, 3.0)), key(Vec3::new(1.000_02, 2.0, 2.999_98)));
		assert_ne!(key(Vec3::ZERO), key(Vec3::new(0.01, 0.0, 0.0)));
	}
}
