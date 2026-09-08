//! Turning a terrain description into geometry, and unmaking it again.
//!
//! [`Terrain`] is a record on an entity and nothing else - @ref
//! `colby_core::abi::terrain` for why it is a description rather than a buffer.
//! This is what reads it: once per step, before the solver and **on both sides
//! of the edit-mode guard**, because a person adding ground in the editor has
//! to see it appear. That is the one place this differs from
//! [`sparks`](crate::sparks), which is inside the guard: a cloud is a thing
//! that moves and ground is a thing that is there.
//!
//! **Three things are ensured, never created blindly.** The mesh under a
//! deterministic name, the entity's [`Renderable::mesh`], and one static body
//! if the terrain is solid. Each is checked before it is written, and that is
//! what makes this idempotent - which is what makes a saved world load
//! correctly. A `.cscene` writes the terrain record *and* the body it built, so
//! a restore hands this a body that is already right; a `.scene` source names a
//! mesh that does not exist yet, so a restore hands it a body pointing at
//! [`MeshId::NONE`] and this corrects it on the first step. Neither path needs
//! the scene writer to know what a terrain is.
//!
//! **The name is the entity's slot**, so a world restored slot for slot
//! rewrites the entries it had rather than growing new ones - a registry never
//! drops anything, and a terrain rebuilt on every load would be a leak with a
//! reload timer on it. An entity duplicated into a new slot gets a new entry,
//! which is what a copy means.

use std::time::Instant;

use colby_core::{
	abi::{
		Body, BodyId, BodyKind, EntityId, MeshId, Shape, ShapeKind, Terrain, World,
		mesh::MeshData,
	},
	trace, warn,
};

/// What a name begins with, so that everything one terrain owns is findable.
pub(crate) const PREFIX: &str = "terrain.";

/// What has been built, so that nothing is built twice.
///
/// One entry per entity slot, holding the record it was built from and the
/// *handle* of the entity that asked. Both are needed, and the handle rather
/// than the slot number: the record answers "has anything about the ground
/// changed", and the handle carries the generation, which answers "is this
/// even the same entity" - a question a slot that changed hands would
/// otherwise get wrong in the one way that matters, by leaving a hill under
/// something that is not a hill.
///
/// The shape [`Simulation`](colby_physics::Simulation) keeps for its baked
/// colliders, for the same reason and with the same two keys.
#[derive(Clone, Debug, Default)]
pub(crate) struct Ground {
	/// The record and the entity each slot was last built for.
	built: Vec<Option<(Terrain, EntityId)>>,

	/// How many triangles the built terrains have between them.
	///
	/// A number about the project rather than about a frame, which is exactly
	/// what `--profile` prints beside its means: the terrain has no per-frame
	/// cost of its own to give a row to - it is built once and then is
	/// geometry - and what it *does* cost shows up in `gpu scene` and
	/// `cpu narrow` as a bigger number. How long a build took is logged where
	/// it is spent instead.
	triangles: u64,
}

impl Ground {
	/// Nothing built yet.
	#[must_use]
	pub(crate) fn new() -> Self { Self::default() }

	/// How many triangles the world's terrain has.
	pub(crate) const fn triangles(&self) -> u64 { self.triangles }
}

/// Brings the world's geometry in line with its terrain records.
///
/// @param world - the host state: the entity table for the records, the mesh
/// registry for the geometry and the body table for the collision
/// @param ground - what was built last time
pub(crate) fn sync(world: &mut World, ground: &mut Ground) {
	let slots = world.entities.slots();

	ground.built.resize(slots, None);
	ground.triangles = 0;

	// collected before anything is written, because building writes to the two
	// tables this reads. One pass over the living rather than over the slots:
	// almost every world has no terrain at all and this is the whole of what it
	// then costs.
	let mut wanted: Vec<Option<(EntityId, Terrain)>> = vec![None; slots];

	for (id, ..) in world.entities.iter() {
		let Some(terrain) = world
			.entities
			.terrain(id)
			.copied()
			.filter(|it| it.is_ground())
		else {
			continue;
		};

		if let Some(entry) = wanted.get_mut(id.slot()) {
			*entry = Some((id, terrain));
		}
	}

	for slot in 0..slots {
		one(world, ground, slot, wanted.get(slot).copied().flatten());
	}
}

/// Brings one slot in line.
///
/// @param world - the host state
/// @param ground - what was built last time
/// @param slot - the entity slot to look at
/// @param wanted - the living entity there and the ground it is, if it is any
fn one(world: &mut World, ground: &mut Ground, slot: usize, wanted: Option<(EntityId, Terrain)>) {
	let was = ground.built.get(slot).copied().flatten();

	match (wanted, was) {
		| (Some((id, now)), Some((before, built))) if now == before && built == id => {
			// the usual case by an enormous margin, and the reason this is
			// cheap enough to run every step: the record is eight words of
			// plain data and comparing it is a memcmp.
			ground.triangles = ground.triangles.saturating_add(now.triangles());
		},
		| (Some((id, now)), was) => {
			// a slot that changed hands has to give its old occupant's ground
			// up before the new one's is built, or the body the old one made
			// would be found and reused for a hill somewhere else entirely.
			if let Some((_, before)) = was.filter(|&(_, before)| before != id) {
				unmake(world, before, slot);
			}

			build(world, id, slot, now);
			set(ground, slot, Some((now, id)));
			ground.triangles = ground.triangles.saturating_add(now.triangles());
		},
		| (None, Some((_, before))) => {
			unmake(world, before, slot);
			set(ground, slot, None);
		},
		| (None, None) => (),
	}
}

/// Builds a terrain's geometry, its renderable and its body.
///
/// @param world - the host state
/// @param id - the entity carrying the terrain
/// @param slot - its slot, which names the mesh
/// @param terrain - what to build
fn build(world: &mut World, id: EntityId, slot: usize, terrain: Terrain) {
	let name = named(slot);
	let began = Instant::now();
	let mesh = world.meshes.insert(&name, terrain.build());

	// the one measurement this module makes, and it is here rather than on a
	// profiler row because it is spent once rather than per frame - a row
	// reading nil on every frame but the first would be noise where this is an
	// answer. @ref `crate::profile`.
	trace!(
		name,
		triangles = terrain.triangles(),
		side = terrain.vertices_across(),
		took_us = began.elapsed().as_micros(),
		"terrain built"
	);

	if let Some(renderable) = world.entities.renderable_mut(id) {
		renderable.mesh = mesh;
	}

	// exactly one, whatever arrived: a body a save restored, a body an earlier
	// step made, and - the case nothing here can rehearse - a body a host
	// described to a client that had already built its own. All three are the
	// same statement, "this entity's ground collides", and two of them is a
	// second collision mesh nobody asked for. @ref `colby-known-gaps` for the
	// wire half, which no oracle in this workspace can see.
	match (terrain.solid, only_body(world, id)) {
		| (true, Some(body)) =>
			if let Some(body) = world.bodies.get_mut(body) {
				body.shape = Shape::mesh(mesh);
			},
		| (true, None) => {
			let transform = world.entities.placed(id).unwrap_or_default();
			let mut body = Body::new(BodyKind::Static, Shape::mesh(mesh), transform);
			body.entity = id;

			let made = world.bodies.spawn(body);

			world.bodies.set_name(made, &name);
		},
		| (false, Some(body)) => {
			world.bodies.despawn(body);
		},
		| (false, None) => (),
	}
}

/// Takes back everything a terrain built.
///
/// The mesh entry itself stays, emptied: a registry never drops a slot, and an
/// empty entry is what [`MeshId::NONE`] already means to everything that reads
/// one. What goes is the body and the entity's claim on the geometry.
///
/// @param world - the host state
/// @param id - the entity that carried the terrain, alive or not
/// @param slot - its slot
fn unmake(world: &mut World, id: EntityId, slot: usize) {
	let name = named(slot);
	let mesh = world.meshes.find(&name);

	world.meshes.insert(&name, MeshData::default());

	if let Some(renderable) = world.entities.renderable_mut(id)
		&& renderable.mesh == mesh
	{
		renderable.mesh = MeshId::NONE;
	}

	while let Some(body) = only_body(world, id) {
		world.bodies.despawn(body);
	}

	trace!(name, "terrain unmade");
}

/// The body a terrain built for an entity, if it has one, with any others
/// taken away.
///
/// **Found rather than remembered**, and that is the whole of why a saved world
/// loads correctly: the `.cscene` wrote this body down like any other, so after
/// a restore it is already there and this finds it instead of making a second.
/// A mesh body on a terrain entity is a terrain's body by construction -
/// nothing else puts one there, because a terrain's mesh is the only geometry
/// the entity has.
///
/// **And it is one**, which is the part that is a guard rather than a lookup:
/// whatever put a second there, the first is kept and the rest go. Keeping the
/// first rather than the last is deliberate - a slot handed out earlier is the
/// one anything else in the world has had a chance to name.
///
/// @param world - the host state
/// @param id - the entity
fn only_body(world: &mut World, id: EntityId) -> Option<BodyId> {
	let found: Vec<BodyId> = world
		.bodies
		.iter()
		.filter(|(_, body)| body.entity == id && body.shape.kind == ShapeKind::Mesh)
		.map(|(handle, _)| handle)
		.collect();
	let kept = found.first().copied();

	for extra in found.into_iter().skip(1) {
		warn!(entity = id.slot(), "a terrain had two bodies; the later one is taken away");
		world.bodies.despawn(extra);
	}

	kept
}

/// Writes what a slot was built from.
fn set(ground: &mut Ground, slot: usize, built: Option<(Terrain, EntityId)>) {
	if let Some(entry) = ground.built.get_mut(slot) {
		*entry = built;
	}
}

/// What a slot's mesh is registered under.
///
/// @param slot - the entity slot
fn named(slot: usize) -> String { format!("{PREFIX}{slot}") }

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{Entry, TerrainKind, Transform},
		glam::Vec3,
	};

	use super::*;

	/// A world with one entity that is ground, and the ground state beside it.
	fn hilly() -> (World, Ground, EntityId) {
		let mut world = World::new();
		let id = world.entities.spawn_at(Transform::IDENTITY);
		let ground = Ground::new();

		world
			.entities
			.set_terrain(id, Terrain { side: 9, ..Terrain::of(3) });

		(world, ground, id)
	}

	#[test]
	fn a_terrain_gets_a_mesh_a_renderable_and_a_body() {
		let (mut world, mut ground, id) = hilly();

		sync(&mut world, &mut ground);

		let mesh = world.meshes.find("terrain.0");

		assert!(mesh.is_some(), "the geometry is registered under the slot's name");
		assert_eq!(
			world.entities.renderable(id).map(|it| it.mesh),
			Some(mesh),
			"and the entity draws it"
		);
		assert_eq!(world.bodies.len(), 1, "and one body stands on it");
		assert_eq!(
			world
				.bodies
				.iter()
				.next()
				.map(|(_, body)| body.shape.mesh),
			Some(mesh),
			"collided against the same geometry it is drawn with"
		);
	}

	#[test]
	fn a_second_step_builds_nothing() {
		let (mut world, mut ground, _) = hilly();

		sync(&mut world, &mut ground);

		let after = world.meshes.len();
		let before = world
			.meshes
			.get(world.meshes.find("terrain.0"))
			.map(Entry::revision);

		sync(&mut world, &mut ground);

		assert_eq!(world.meshes.len(), after, "no second entry");
		assert_eq!(world.bodies.len(), 1, "and no second body");
		assert_eq!(
			world
				.meshes
				.get(world.meshes.find("terrain.0"))
				.map(Entry::revision),
			before,
			"and the revision did not move, which is what the renderer reads"
		);
	}

	#[test]
	fn turning_a_knob_rebuilds_in_place() {
		let (mut world, mut ground, id) = hilly();

		sync(&mut world, &mut ground);

		let mesh = world.meshes.find("terrain.0");
		let before = world.meshes.get(mesh).map(Entry::revision);

		if let Some(terrain) = world.entities.terrain_mut(id) {
			terrain.seed = 99;
		}

		sync(&mut world, &mut ground);

		assert_eq!(world.meshes.find("terrain.0"), mesh, "the same handle");
		assert_ne!(
			world.meshes.get(mesh).map(Entry::revision),
			before,
			"with a new revision, which is what makes the renderer re-upload"
		);
		assert_eq!(world.bodies.len(), 1, "and still one body");
	}

	#[test]
	fn turning_the_ground_off_takes_the_body_and_the_geometry() {
		let (mut world, mut ground, id) = hilly();

		sync(&mut world, &mut ground);
		world.entities.set_terrain(id, Terrain::NONE);
		sync(&mut world, &mut ground);

		assert_eq!(world.bodies.len(), 0, "nothing to stand on");
		assert_eq!(
			world.entities.renderable(id).map(|it| it.mesh),
			Some(MeshId::NONE),
			"and nothing to draw"
		);
		assert!(
			world
				.meshes
				.get(world.meshes.find("terrain.0"))
				.is_some_and(|entry| entry.value().is_empty()),
			"the entry stays and is emptied, because a registry drops nothing"
		);
	}

	#[test]
	fn a_terrain_that_is_not_solid_gets_no_body() {
		let (mut world, mut ground, id) = hilly();

		if let Some(terrain) = world.entities.terrain_mut(id) {
			terrain.solid = false;
		}

		sync(&mut world, &mut ground);

		assert_eq!(world.bodies.len(), 0, "a backdrop pays for no collision mesh");
		assert!(
			world
				.entities
				.renderable(id)
				.is_some_and(|it| it.mesh.is_some()),
			"and is still drawn"
		);
	}

	#[test]
	fn making_it_solid_afterwards_gives_it_a_body() {
		let (mut world, mut ground, id) = hilly();

		if let Some(terrain) = world.entities.terrain_mut(id) {
			terrain.solid = false;
		}
		sync(&mut world, &mut ground);

		if let Some(terrain) = world.entities.terrain_mut(id) {
			terrain.solid = true;
		}
		sync(&mut world, &mut ground);

		assert_eq!(world.bodies.len(), 1);
	}

	#[test]
	fn a_body_that_was_restored_beside_the_record_is_not_doubled() {
		// the case that makes a saved world load correctly: a `.cscene` writes
		// the terrain's body down like any other, so the sync finds one waiting
		// and must not make a second.
		let (mut world, mut ground, id) = hilly();
		let mut body =
			Body::new(BodyKind::Static, Shape::mesh(MeshId::NONE), Transform::IDENTITY);
		body.entity = id;
		world.bodies.spawn(body);

		sync(&mut world, &mut ground);

		assert_eq!(world.bodies.len(), 1, "the one that was already there");
		assert_eq!(
			world
				.bodies
				.iter()
				.next()
				.map(|(_, it)| it.shape.mesh),
			Some(world.meshes.find("terrain.0")),
			"pointed at the geometry this step built"
		);
	}

	#[test]
	fn a_terrain_that_somehow_has_two_bodies_keeps_one() {
		// the case a save cannot make and a wire can: a client that built its
		// own ground and was then told about the host's. No oracle here can
		// reach it, so it is a test and a warning rather than a screenshot.
		let (mut world, mut ground, id) = hilly();

		sync(&mut world, &mut ground);

		let mut second =
			Body::new(BodyKind::Static, Shape::mesh(MeshId::NONE), Transform::IDENTITY);
		second.entity = id;
		world.bodies.spawn(second);

		assert_eq!(world.bodies.len(), 2, "two, for a moment");

		if let Some(terrain) = world.entities.terrain_mut(id) {
			terrain.seed = 4;
		}
		sync(&mut world, &mut ground);

		assert_eq!(world.bodies.len(), 1, "and one afterwards");
		assert_eq!(
			world
				.bodies
				.iter()
				.next()
				.map(|(_, it)| it.shape.mesh),
			Some(world.meshes.find("terrain.0")),
			"the one that was there first, pointed at the ground"
		);
	}

	#[test]
	fn a_dead_entity_takes_its_ground_with_it() {
		let (mut world, mut ground, id) = hilly();

		sync(&mut world, &mut ground);
		world.entities.despawn(id);
		sync(&mut world, &mut ground);

		assert_eq!(world.bodies.len(), 0, "nothing is left standing");
	}

	#[test]
	fn a_slot_that_changed_hands_is_rebuilt_rather_than_inherited() {
		let (mut world, mut ground, id) = hilly();

		sync(&mut world, &mut ground);
		world.entities.despawn(id);

		let again = world.entities.spawn_at(Transform::at(Vec3::Y));

		assert_eq!(again.slot(), id.slot(), "the same slot, a new generation");

		world
			.entities
			.set_terrain(again, Terrain { side: 9, seed: 42, ..Terrain::hills() });
		sync(&mut world, &mut ground);

		assert_eq!(world.entities.terrain(again).map(|it| it.kind), Some(TerrainKind::Noise));
		assert_eq!(world.bodies.len(), 1, "one body, made for the new occupant");
	}

	#[test]
	fn something_dropped_on_a_terrain_comes_to_rest_on_it() {
		// the one thing no unit test above proves and the whole card asks for:
		// that the geometry this builds is geometry the solver collides
		// against. Nothing here knows what a terrain is except the record on
		// the entity - the body, the collider and the contact are all the
		// ordinary path a mesh takes.
		let mut world = World::new();
		let mut ground = Ground::new();
		let mut simulation = Box::new(colby_physics::Simulation::new());

		world.install_physics(simulation.table());

		let hill = world.entities.spawn_at(Transform::IDENTITY);
		let terrain = Terrain {
			size: 32.0,
			height: 6.0,
			side: 33,
			..Terrain::of(11)
		};

		world.entities.set_terrain(hill, terrain);

		let crate_at = Vec3::new(2.5, 12.0, -3.5);
		let falling = world.entities.spawn_at(Transform::at(crate_at));
		let mut body = Body::new(
			BodyKind::Dynamic,
			Shape::cuboid(Vec3::splat(0.5)),
			Transform::at(crate_at),
		);
		body.entity = falling;

		let dropped = world.bodies.spawn(body);

		world.dt = 1.0 / 60.0;
		for _ in 0..240 {
			sync(&mut world, &mut ground);
			simulation.step(&mut world);
		}

		let rest = world
			.bodies
			.get(dropped)
			.expect("it is still a body")
			.transform
			.position;
		let under = terrain.height_at(colby_core::glam::Vec2::new(rest.x, rest.z));

		assert!(
			rest.y > under,
			"it ended at {} and the ground there is at {under}: it fell through",
			rest.y
		);
		assert!(
			rest.y - under < 1.5,
			"it ended at {} and the ground there is at {under}: it is hovering",
			rest.y
		);
		assert!(
			(rest.x - crate_at.x).abs() < 6.0,
			"and it landed near where it was dropped rather than being flung: {rest:?}"
		);
	}

	#[test]
	fn a_ray_fired_down_at_a_terrain_hits_the_surface() {
		// the other half of "you can stand on it": a trace answers about the
		// ground, which is what anything aiming at it needs.
		let mut world = World::new();
		let mut ground = Ground::new();
		let mut simulation = Box::new(colby_physics::Simulation::new());

		world.install_physics(simulation.table());

		let hill = world.entities.spawn_at(Transform::IDENTITY);
		let terrain = Terrain {
			size: 32.0,
			height: 6.0,
			side: 33,
			..Terrain::of(5)
		};

		world.entities.set_terrain(hill, terrain);
		sync(&mut world, &mut ground);
		// a step, because a collider is baked when the solver first sees the
		// body and a trace reads the baked one.
		world.dt = 1.0 / 60.0;
		simulation.step(&mut world);

		let flat = colby_core::glam::Vec2::new(1.5, -2.5);
		let above = Vec3::new(flat.x, 20.0, flat.y);
		let hit =
			world.trace_ray(&colby_core::abi::TraceInfo::ray(above, above + Vec3::NEG_Y * 40.0));

		assert!(hit.hit, "a ray straight down at ground hits it");
		assert!(
			(hit.end.y - terrain.height_at(flat)).abs() < 0.5,
			"it hit at {} and the field says {}",
			hit.end.y,
			terrain.height_at(flat)
		);
		assert!(hit.normal.y > 0.5, "and the ground faces upwards: {:?}", hit.normal);
	}

	#[test]
	fn a_world_with_no_terrain_at_all_does_nothing_and_spends_nothing() {
		let mut world = World::new();
		let mut ground = Ground::new();

		world.entities.spawn_at(Transform::IDENTITY);
		sync(&mut world, &mut ground);

		assert_eq!(world.bodies.len(), 0);
		assert_eq!(ground.triangles(), 0);
	}
}
