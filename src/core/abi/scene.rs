//! A whole world as plain data: what stands where, what collides, what is
//! held together, and what the game itself remembers.
//!
//! Three things wear the word "serialization" and only one format is worth
//! writing, so this module is shaped by the difference between them.
//!
//! - **A save** has to come back exactly: every handle a game wrote into its
//!   own arena must resolve to the same entity afterwards, or the world is
//!   restored and nobody in it knows anything.
//! - **A dupe** has to come back *beside* what is already there, with new
//!   handles, because the point is a second copy rather than the same one.
//! - **A network snapshot** is the first of those, small and often.
//!
//! What follows from that: one description, and two ways to put it back. A
//! restore rebuilds the tables slot for slot, generation for generation, and
//! puts the arena back with them, so the handles in it are the handles they
//! were. An instantiate ignores all of that and creates fresh ones, handing
//! back a map from what the file called something to what it turned into.
//! This module holds the description and the capture; the two loaders and the
//! file itself come after it.
//!
//! **Everything a resource handle names is written as a name.** A `MeshId`
//! is where an asset landed in a registry this run, and the next run may fill
//! that registry in another order; the name is the thing that is stable, and
//! it is what the compiled model format already does for the same reason. The
//! handles that are *not* written as names are the ones into the world's own
//! tables - an entity, a body, a joint - and those are written as an index
//! into this description plus the slot and generation they held, which is what
//! lets both loaders work off one record.
//!
//! **What is deliberately not here**: the asset registries themselves, which
//! are the compiled tree and would be a copy of it; the console variables,
//! which have a file of their own and a different lifetime; the interface's
//! panels and binds, which a game shows again from `init`; and everything a
//! step derives and clears - input, the debug table, the touch and overlap
//! queues. @ref `colby-known-gaps` for what that costs.

use crate::{
	abi::{
		BodyId, BodyKind, Camera, JointKind, Layers, Shape, ShapeKind, Transform, World,
		state::STATE_BYTES,
	},
	glam::{Quat, Vec3},
};

/// What a record holds instead of an index when it names nothing.
///
/// Not zero, because zero is a perfectly good index into these lists - unlike
/// a registry, where slot zero is the null entry by construction.
pub const NO_INDEX: u32 = u32::MAX;

/// The world's own settings, as opposed to anything standing in it.
///
/// Only the fields a game writes and the renderer reads. The window's aspect
/// is not here because it belongs to the window, and neither is anything the
/// host counts about itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stage {
	/// Where the camera looks from.
	pub camera: Camera,

	/// The clear color, linear RGB.
	pub clear: Vec3,

	/// The direction the light travels.
	pub light: Vec3,

	/// How lit a surface facing away from the light still is.
	pub ambient: Vec3,

	/// What every dynamic body accelerates by.
	pub gravity: Vec3,

	/// Simulated seconds so far.
	///
	/// Written down because gameplay integrates against it: a world restored
	/// at time zero is a world whose animations all jump.
	pub time: f32,

	/// Simulation steps so far.
	pub steps: u64,
}

impl Stage {
	/// The settings a world starts with.
	pub const DEFAULT: Self = Self {
		camera: Camera::DEFAULT,
		clear: Vec3::ZERO,
		light: Vec3::new(-0.4, -1.0, -0.3),
		ambient: Vec3::splat(0.25),
		gravity: Vec3::new(0.0, -9.81, 0.0),
		time: 0.0,
		steps: 0,
	};
}

impl Default for Stage {
	fn default() -> Self { Self::DEFAULT }
}

/// One entity: where it is and what it looks like.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Thing {
	/// What this is called, or empty.
	///
	/// Written by whoever authored the scene and never by the capture, which
	/// has nothing to call anything: an entity in this engine has no name of
	/// its own. A name is what an instantiate hands back a handle under, and
	/// it is how a scene file refers to a thing a person typed.
	pub name: String,

	/// The slot it occupied, for a restore.
	pub slot: u32,

	/// Which occupant of that slot it was.
	pub generation: u32,

	/// Where it is.
	pub transform: Transform,

	/// The asset name of its mesh, or empty for nothing to draw.
	pub mesh: String,

	/// The asset name of its material, or empty for the default one.
	pub material: String,

	/// Its own tint, linear RGB.
	pub color: Vec3,
}

/// A shape with its mesh named rather than handled.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Form {
	/// Which of the three kinds it is.
	pub kind: ShapeKind,

	/// The radius of a ball.
	pub radius: f32,

	/// The half-extents of a box.
	pub extents: Vec3,

	/// The asset name of a mesh shape's geometry, or empty.
	pub mesh: String,
}

/// One body: everything the solver reads about it.
///
/// A sensor is one of these too - [`sensor`](Self::sensor) is a field rather
/// than a kind, exactly as it is on the live body.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Solid {
	/// What this is called, or empty. @ref [`Thing::name`].
	pub name: String,

	/// The slot it occupied, for a restore.
	pub slot: u32,

	/// Which occupant of that slot it was.
	pub generation: u32,

	/// What the solver may do with it.
	pub kind: BodyKind,

	/// What it is shaped like.
	pub shape: Form,

	/// Where it is.
	pub transform: Transform,

	/// How fast it is moving.
	pub velocity: Vec3,

	/// How fast it is turning.
	pub angular: Vec3,

	/// How heavy it is.
	pub mass: f32,

	/// How much of an impact comes back.
	pub restitution: f32,

	/// How hard it is to slide along.
	pub friction: f32,

	/// Whether it notices rather than pushes.
	pub sensor: bool,

	/// Whether the solver had stopped integrating it.
	pub sleeping: bool,

	/// Which layers it is on and which it interacts with.
	pub layers: Layers,

	/// Which entry of [`SceneData::things`] it drives, or [`NO_INDEX`].
	pub thing: u32,
}

/// One joint, naming the two bodies it holds by their place in the file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Link {
	/// What this is called, or empty. @ref [`Thing::name`].
	pub name: String,

	/// The slot it occupied, for a restore.
	pub slot: u32,

	/// Which occupant of that slot it was.
	pub generation: u32,

	/// Which of the three it is.
	pub kind: JointKind,

	/// Which entry of [`SceneData::solids`] it holds, or [`NO_INDEX`].
	pub first: u32,

	/// The other, or [`NO_INDEX`] for a joint pinned to a point in the world.
	pub second: u32,

	/// Where it attaches on the first body, in that body's own space.
	pub first_anchor: Vec3,

	/// Where it attaches on the second, or in the world if there is none.
	pub second_anchor: Vec3,

	/// The axis a hinge turns about.
	pub axis: Vec3,

	/// How far apart a rope lets them get.
	pub length: f32,

	/// The relative rotation the joint was made at.
	pub rest: Quat,

	/// How much of the pull is given up each step.
	pub give: f32,
}

/// The game's own bytes, and the layout number they were written under.
///
/// Opaque here and opaque to the host: what makes this safe to copy around is
/// that the arena is `Pod` by contract, so every bit pattern is a valid
/// whatever-the-game-declared. The layout number is what makes it safe to
/// *use*: a build whose number has moved on gets a fresh arena rather than
/// yesterday's bytes read through today's fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arena {
	/// The number the game stamped on the bytes.
	pub layout: u64,

	/// The bytes themselves, [`STATE_BYTES`] of them.
	pub bytes: Vec<u8>,
}

impl Arena {
	/// An arena nobody has claimed.
	#[must_use]
	pub fn empty() -> Self { Self { layout: 0, bytes: vec![0; STATE_BYTES] } }
}

impl Default for Arena {
	fn default() -> Self { Self::empty() }
}

/// A world, or a piece of one, as plain data.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneData {
	/// The world's own settings.
	pub stage: Stage,

	/// Every entity that was alive.
	pub things: Vec<Thing>,

	/// Every body that was alive.
	pub solids: Vec<Solid>,

	/// Every joint that was alive.
	pub links: Vec<Link>,

	/// The generation each entity slot was on, dead ones included.
	///
	/// One per slot the table had ever handed out, which is what makes a
	/// restored world refuse a handle that was already stale when it was
	/// saved. The free list is deliberately *not* here: a restore rebuilds it
	/// from whichever slots nothing occupies, in order, and the only thing
	/// that changes is which slot the next spawn takes.
	pub thing_generations: Vec<u32>,

	/// The same for body slots.
	pub solid_generations: Vec<u32>,

	/// The same for joint slots.
	pub link_generations: Vec<u32>,

	/// The game's own state, or nothing.
	///
	/// Present in a capture and absent in a scene somebody wrote by hand:
	/// there is nothing an author could put here, and an instantiate would
	/// have nowhere to put it if there were.
	pub arena: Option<Arena>,
}

impl SceneData {
	/// Whether there is nothing in it at all.
	#[must_use]
	pub fn is_empty(&self) -> bool {
		self.things.is_empty() && self.solids.is_empty() && self.links.is_empty()
	}
}

/// Writes down everything about a world that a world is made of.
///
/// @param world - what to describe
/// @return the description, with the arena in it
#[must_use]
pub fn capture(world: &World) -> SceneData {
	let things = things(world);
	// slot to index, so a body naming an entity and a joint naming a body both
	// look their answer up rather than searching. Sized by the table rather
	// than by what is alive, so a slot is an index into it directly.
	let mut thing_of = vec![NO_INDEX; world.entities.slots()];
	for (index, thing) in things.iter().enumerate() {
		let Ok(index) = u32::try_from(index) else {
			continue;
		};

		let Ok(slot) = usize::try_from(thing.slot) else {
			continue;
		};

		if let Some(entry) = thing_of.get_mut(slot) {
			*entry = index;
		}
	}

	let solids = solids(world, &thing_of);
	let mut solid_of = vec![NO_INDEX; world.bodies.slots()];
	for (index, solid) in solids.iter().enumerate() {
		let Ok(index) = u32::try_from(index) else {
			continue;
		};

		let Ok(slot) = usize::try_from(solid.slot) else {
			continue;
		};

		if let Some(entry) = solid_of.get_mut(slot) {
			*entry = index;
		}
	}

	SceneData {
		stage: stage(world),
		links: links(world, &solid_of),
		thing_generations: (0..world.entities.slots())
			.map(|slot| world.entities.generation(slot))
			.collect(),
		solid_generations: (0..world.bodies.slots())
			.map(|slot| world.bodies.generation(slot))
			.collect(),
		link_generations: (0..world.joints.slots())
			.map(|slot| world.joints.generation(slot))
			.collect(),
		arena: Some(Arena {
			layout: world.state.layout(),
			bytes: world.state.raw().to_vec(),
		}),
		things,
		solids,
	}
}

/// The world's own settings.
fn stage(world: &World) -> Stage {
	Stage {
		camera: world.camera,
		clear: world.clear,
		light: world.light,
		ambient: world.ambient,
		gravity: world.gravity,
		time: world.time,
		steps: world.steps,
	}
}

/// Every living entity, with its handles resolved back to names.
fn things(world: &World) -> Vec<Thing> {
	world
		.entities
		.iter()
		.map(|(id, transform, renderable)| Thing {
			name: String::new(),
			slot: u32::try_from(id.slot()).unwrap_or(0),
			generation: id.generation(),
			transform: *transform,
			mesh: world
				.meshes
				.get(renderable.mesh)
				.map_or_else(String::new, |entry| entry.name().to_owned()),
			material: world
				.materials
				.entry(renderable.material)
				.map_or_else(String::new, |entry| entry.name().to_owned()),
			color: renderable.color,
		})
		.collect()
}

/// Every living body, with its entity turned into an index.
fn solids(world: &World, thing_of: &[u32]) -> Vec<Solid> {
	world
		.bodies
		.iter()
		.map(|(id, body)| Solid {
			name: String::new(),
			slot: u32::try_from(id.slot()).unwrap_or(0),
			generation: id.generation(),
			kind: body.kind,
			shape: form(world, &body.shape),
			transform: body.transform,
			velocity: body.velocity,
			angular: body.angular,
			mass: body.mass,
			restitution: body.restitution,
			friction: body.friction,
			sensor: body.sensor,
			sleeping: body.sleeping,
			layers: body.layers,
			thing: thing_of
				.get(body.entity.slot())
				.copied()
				.filter(|_| body.entity.is_some())
				.unwrap_or(NO_INDEX),
		})
		.collect()
}

/// Every living joint, with both bodies turned into indices.
fn links(world: &World, solid_of: &[u32]) -> Vec<Link> {
	let named = |body: BodyId| {
		solid_of
			.get(body.slot())
			.copied()
			.filter(|_| body.is_some())
			.unwrap_or(NO_INDEX)
	};

	world
		.joints
		.iter()
		.map(|(id, joint)| Link {
			name: String::new(),
			slot: u32::try_from(id.slot()).unwrap_or(0),
			generation: id.generation(),
			kind: joint.kind,
			first: named(joint.first),
			second: named(joint.second),
			first_anchor: joint.first_anchor,
			second_anchor: joint.second_anchor,
			axis: joint.axis,
			length: joint.length,
			rest: joint.rest,
			give: joint.give,
		})
		.collect()
}

/// A shape with its mesh named.
fn form(world: &World, shape: &Shape) -> Form {
	Form {
		kind: shape.kind,
		radius: shape.radius,
		extents: shape.extents,
		mesh: world
			.meshes
			.get(shape.mesh)
			.map_or_else(String::new, |entry| entry.name().to_owned()),
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::abi::{Body, Joint, Material, MeshData, Renderable, mesh};

	/// A world with a mesh, a material and one entity standing on a body.
	fn furnished() -> World {
		let mut world = World::new();
		world
			.meshes
			.insert("meshes/crystal", MeshData::default());
		world
			.materials
			.insert("brass", Material::colored(Vec3::new(0.7, 0.5, 0.2)));

		world
	}

	#[test]
	fn an_empty_world_captures_as_an_empty_scene() {
		let scene = capture(&World::new());

		assert!(scene.is_empty(), "nothing stands in it");
		assert_eq!(
			scene.stage,
			Stage::DEFAULT,
			"and its settings are the ones a world starts with"
		);
	}

	#[test]
	fn an_entity_is_written_down_with_its_assets_named() {
		let mut world = furnished();
		let mesh = world.meshes.find("meshes/crystal");
		let material = world.materials.find("brass");
		let id = world
			.entities
			.spawn_at(Transform::at(Vec3::new(1.0, 2.0, 3.0)));
		world
			.entities
			.set_renderable(id, Renderable::of(mesh, material, Vec3::X));

		let scene = capture(&world);
		let thing = scene.things.first().expect("one entity");

		assert_eq!(thing.mesh, "meshes/crystal", "the mesh is named rather than numbered");
		assert_eq!(thing.material, "brass", "and so is the material");
		assert_eq!(thing.slot, u32::try_from(id.slot()).unwrap(), "its slot is written down");
		assert_eq!(thing.generation, id.generation(), "with the generation that goes with it");
		assert_eq!(thing.transform.position, Vec3::new(1.0, 2.0, 3.0), "and where it is");
	}

	#[test]
	fn an_entity_drawing_nothing_names_nothing() {
		let mut world = World::new();
		world.entities.spawn();

		let scene = capture(&world);
		let thing = scene.things.first().expect("one entity");

		assert!(thing.mesh.is_empty(), "the null mesh is the empty name");
		assert_eq!(thing.material, "default", "and the default material is a real one");
	}

	#[test]
	fn a_built_in_primitive_is_named_like_any_other_mesh() {
		let mut world = World::new();
		let id = world.entities.spawn();
		world
			.entities
			.set_renderable(id, Renderable::new(crate::abi::MeshId::CUBE, Vec3::ONE));

		let scene = capture(&world);

		assert_eq!(
			scene.things[0].mesh,
			mesh::CUBE_NAME,
			"the primitives are registry entries and are written the same way"
		);
	}

	#[test]
	fn a_body_names_the_entity_it_drives_by_its_place_in_the_file() {
		let mut world = World::new();
		world.entities.spawn();
		let driven = world.entities.spawn();
		let body = world
			.bodies
			.spawn(Body::dynamic(Shape::UNIT, Transform::IDENTITY, 2.0).driving(driven));

		let scene = capture(&world);
		let solid = scene.solids.first().expect("one body");

		assert_eq!(solid.thing, 1, "the second entity is the second thing in the list");
		assert_eq!(
			scene.things[usize::try_from(solid.thing).unwrap()].slot,
			u32::try_from(driven.slot()).unwrap(),
			"and it is the one the body actually drives"
		);
		assert!((solid.mass - 2.0).abs() < f32::EPSILON, "the rest of the body comes with it");
		assert_eq!(solid.slot, u32::try_from(body.slot()).unwrap(), "including its own slot");
	}

	#[test]
	fn a_body_that_drives_nothing_names_nothing() {
		// with an entity standing in slot zero, which is the case that makes
		// this worth testing: a handle to nothing *is* slot zero, so a lookup
		// that does not check first hands back whoever lives there.
		let mut world = World::new();
		world.entities.spawn();
		world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY));

		let scene = capture(&world);

		assert_eq!(scene.things.len(), 1, "there is an entity in slot zero");
		assert_eq!(
			scene.solids[0].thing, NO_INDEX,
			"and the body drives it no more than it drives anything else"
		);
	}

	#[test]
	fn a_mesh_shape_names_its_geometry() {
		let mut world = furnished();
		let mesh = world.meshes.find("meshes/crystal");
		world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::mesh(mesh), Transform::IDENTITY));

		let scene = capture(&world);

		assert_eq!(scene.solids[0].shape.kind, ShapeKind::Mesh, "the kind survives");
		assert_eq!(scene.solids[0].shape.mesh, "meshes/crystal", "and so does what it is");
	}

	#[test]
	fn a_joint_names_both_bodies_by_their_place_in_the_file() {
		let mut world = World::new();
		let first = world
			.bodies
			.spawn(Body::dynamic(Shape::UNIT, Transform::IDENTITY, 1.0));
		let second = world.bodies.spawn(Body::dynamic(
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 2.0, 0.0)),
			1.0,
		));
		world.join(Joint::weld(first, second, (Vec3::Y, -Vec3::Y)));
		world.join(Joint::rope(second, BodyId::NONE, (Vec3::ZERO, Vec3::Y * 4.0), 1.5));

		let scene = capture(&world);

		assert_eq!(scene.links.len(), 2, "both joints are written");
		assert_eq!((scene.links[0].first, scene.links[0].second), (0, 1), "the weld holds both");
		assert_eq!(
			(scene.links[1].first, scene.links[1].second),
			(1, NO_INDEX),
			"and the rope holds one body and a point in the world"
		);
		assert!(
			scene.links[0].rest.is_normalized(),
			"the angle a weld was made at comes with it"
		);
	}

	#[test]
	fn a_dead_slot_keeps_its_generation_in_the_written_scene() {
		let mut world = World::new();
		let first = world.entities.spawn();
		world.entities.spawn();
		world.entities.despawn(first);

		let scene = capture(&world);

		assert_eq!(scene.things.len(), 1, "only the living one is described");
		assert_eq!(
			scene.thing_generations.len(),
			2,
			"and the generations cover every slot the table ever handed out"
		);
		assert_eq!(scene.thing_generations[0], 1, "the dead slot's generation is kept");
	}

	#[test]
	fn the_arena_comes_along_with_its_layout_number() {
		#[repr(C)]
		#[derive(Clone, Copy, crate::bytemuck::Pod, crate::bytemuck::Zeroable)]
		struct Held {
			count: u32,
			pad: u32,
		}

		let mut world = World::new();
		world.state.get::<Held>(7).0.count = 42;

		let scene = capture(&world);
		let arena = scene.arena.expect("a capture always carries one");

		assert_eq!(arena.layout, 7, "stamped with the number the game claimed it under");
		assert_eq!(arena.bytes.len(), STATE_BYTES, "the whole arena rather than a prefix");
		assert_eq!(arena.bytes[0], 42, "and the bytes are the game's own");
	}

	#[test]
	fn the_stage_carries_what_a_game_wrote_and_not_what_the_host_counts() {
		let mut world = World::new();
		world.camera.position = Vec3::new(9.0, 8.0, 7.0);
		world.gravity = Vec3::new(0.0, 1.0, 0.0);
		world.time = 12.5;
		world.steps = 750;
		world.reloads = 3;

		let scene = capture(&world);

		assert_eq!(scene.stage.camera.position, Vec3::new(9.0, 8.0, 7.0), "the camera is saved");
		assert_eq!(scene.stage.gravity, Vec3::new(0.0, 1.0, 0.0), "so is a game's own gravity");
		assert!(
			(scene.stage.time - 12.5).abs() < f32::EPSILON,
			"and the clock gameplay integrates against"
		);
	}
}
