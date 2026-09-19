//! Laying every strewing's copies out, and taking them away again.
//!
//! [`Strewing`] is a record on an entity and a rule, and this is what reads it:
//! once a step, right after the terrain's geometry and before the navmesh and
//! the solver, and **on both sides of the edit-mode guard**, for the terrain's
//! reason - somebody dragging a density in a panel has to see the field
//! thicken while the world is stopped, and ground built this step is ground
//! this step's copies stand on. @ref `colby_core::abi::strew` for the rule.
//!
//! **What was laid is kept with what it was laid from**, in the world's own
//! table rather than here. So a world put back by a load, a tab or a step back
//! is compared against what it holds and not against a memory of another
//! world: a key that matches is a layout that is still true, whatever happened
//! in between, and one that does not is laid again.
//!
//! **A solid strewing's body is found rather than remembered**, the terrain's
//! arrangement and for its reason: a save writes the body down like any other,
//! a copy of the entity brings one along, and neither knows what a strewing
//! is. A mesh body driving the strewing entity whose mesh is one of the meshes
//! laid here is its body; there is exactly one, and it collides with the mesh
//! laid for its own slot.

use std::time::Instant;

use colby_core::{
	abi::{
		Body, BodyId, BodyKind, EntityId, Entry, MeshId, STREWING, Shape, ShapeKind, Strewing,
		Transform, World,
		mesh::MeshData,
		strew::{self, Key, Layout},
	},
	trace, warn,
};

/// What a name begins with, for every mesh a solid strewing's body collides
/// with.
pub(crate) const PREFIX: &str = "strewn.";

/// One strewing entity, as this step finds it.
struct Wanted {
	/// Its slot, which names what it lays.
	slot: usize,

	/// The entity.
	id: EntityId,

	/// What it would be laid from now.
	key: Key,
}

/// Brings every strewing's copies in line with the records.
///
/// @param world - the host state: the records and the ground to read, the
/// table the copies go in, the mesh registry and the bodies a solid strewing
/// needs
pub(crate) fn sync(world: &mut World) {
	let wanted = wanted(world);

	// what no strewing asks for any more first: an entity that stopped
	// strewing or went, and a slot that changed hands, whose old occupant's
	// copies and body are not the new one's
	let stale: Vec<usize> = world
		.strewn
		.iter()
		.filter(|&(slot, layout)| {
			!wanted
				.iter()
				.any(|want| want.slot == slot && want.id == layout.entity)
		})
		.map(|(slot, _)| slot)
		.collect();

	for slot in stale {
		unlay(world, slot);
	}

	for want in &wanted {
		let current = world
			.strewn
			.get(want.slot)
			.is_some_and(|layout| layout.entity == want.id && layout.key.same(&want.key));

		if !current {
			lay(world, want);
		}

		body(world, want);
	}
}

/// Every living entity that strews, and what it would be laid from.
///
/// @param world - the host state
fn wanted(world: &World) -> Vec<Wanted> {
	let Some(rules) = world.entities.column(&STREWING) else {
		return Vec::new();
	};

	world
		.entities
		.iter()
		.filter_map(|(id, local, renderable)| {
			let rule = *rules.get(id.slot())?;

			if !rule.strews() {
				return None;
			}

			let ground = world
				.entities
				.renderable(world.entities.parent(id))
				.map_or(MeshId::NONE, |parent| parent.mesh);

			Some(Wanted {
				slot: id.slot(),
				id,
				key: Key {
					rule,
					ground: (ground, revision(world, ground)),
					mesh: (renderable.mesh, revision(world, renderable.mesh)),
					local: *local,
				},
			})
		})
		.collect()
}

/// Which revision of a mesh the registry holds, nought for none.
fn revision(world: &World, mesh: MeshId) -> u32 {
	world.meshes.get(mesh).map_or(0, Entry::revision)
}

/// Lays one strewing's copies, and the mesh its body collides with.
///
/// @param world - the host state
/// @param want - the strewing, and what to lay it from
fn lay(world: &mut World, want: &Wanted) {
	let began = Instant::now();
	let rule = want.key.rule;
	let nothing = MeshData::default();
	let ground = world
		.meshes
		.get(want.key.ground.0)
		.map_or(&nothing, Entry::value);
	let mesh = world
		.meshes
		.get(want.key.mesh.0)
		.map_or(&nothing, Entry::value);

	if !world.entities.parent(want.id).is_some() {
		warn!(
			entity = want.slot,
			"a strewing hangs off nothing, so there is no ground to strew its mesh over"
		);
	} else if ground.indices.is_empty() {
		warn!(
			entity = want.slot,
			"a strewing hangs off something with no mesh, so there is no ground to strew over"
		);
	}

	let laid = strew::lay_out(ground, &rule, &want.key.local, mesh.bounds());
	let solid = rule
		.solid()
		.then(|| strew::solid(&laid, mesh, &want.key.local))
		.filter(|merged| !merged.indices.is_empty());

	if laid.thinned {
		warn!(
			entity = want.slot,
			most = strew::MOST,
			"a strewing would lay more copies than one holds, and is laid at a density that fits"
		);
	}

	trace!(
		entity = want.slot,
		pieces = laid.pieces.len(),
		patches = laid.patches.len(),
		drawn = laid.drawn,
		digest = format!("{:#018X}", laid.digest),
		took_us = began.elapsed().as_micros(),
		"strewn"
	);

	let name = named(want.slot);
	let solid = match solid {
		| Some(merged) => world.meshes.insert(&name, merged),
		// a registry never drops an entry, so one a solid rule made before is
		// emptied rather than left colliding
		| None => {
			if world.meshes.find(&name).is_some() {
				world.meshes.insert(&name, MeshData::default());
			}

			MeshId::NONE
		},
	};

	world.strewn.put(want.slot, Layout {
		entity: want.id,
		key: want.key,
		laid,
		solid,
		revision: 0,
	});
}

/// Takes away everything one slot laid: its copies, the mesh its body collided
/// with, and the body.
///
/// @param world - the host state
/// @param slot - the entity slot
fn unlay(world: &mut World, slot: usize) {
	let Some(layout) = world.strewn.take(slot) else {
		return;
	};
	let name = named(slot);

	if world.meshes.find(&name).is_some() {
		world.meshes.insert(&name, MeshData::default());
	}

	for body in bodies_of(world, layout.entity) {
		world.bodies.despawn(body);
	}

	trace!(entity = slot, "strewing taken away");
}

/// Makes a strewing's body what its rule says: exactly one if it is solid, and
/// none if it is not.
///
/// @param world - the host state
/// @param want - the strewing
fn body(world: &mut World, want: &Wanted) {
	let solid = world
		.strewn
		.get(want.slot)
		.map_or(MeshId::NONE, |layout| layout.solid);
	let found = bodies_of(world, want.id);

	if !solid.is_some() {
		for extra in found {
			world.bodies.despawn(extra);
		}

		return;
	}

	match found.split_first() {
		| Some((&kept, extras)) => {
			for &extra in extras {
				warn!(
					entity = want.slot,
					"a strewing had two bodies; the later one is taken away"
				);
				world.bodies.despawn(extra);
			}

			// a copy brought its source's body along, still colliding with the
			// source's mesh: this slot's is the one it stands for
			if let Some(body) = world.bodies.get_mut(kept) {
				body.shape = Shape::mesh(solid);
			}
		},
		| None => {
			let placed = world.entities.placed(want.id).unwrap_or_default();
			let mut body = Body::new(BodyKind::Static, Shape::mesh(solid), placed);
			body.entity = want.id;

			let made = world.bodies.spawn(body);

			world.bodies.set_name(made, &named(want.slot));
		},
	}
}

/// Every body a strewing made for an entity: a mesh body driving it whose mesh
/// is one of the meshes laid here, in the order the table holds them.
fn bodies_of(world: &World, id: EntityId) -> Vec<BodyId> {
	world
		.bodies
		.iter()
		.filter(|(_, body)| {
			body.entity == id
				&& body.shape.kind == ShapeKind::Mesh
				&& world
					.meshes
					.get(body.shape.mesh)
					.is_some_and(|mesh| mesh.name().starts_with(PREFIX))
		})
		.map(|(handle, _)| handle)
		.collect()
}

/// What a slot's collision mesh is registered under.
fn named(slot: usize) -> String { format!("{PREFIX}{slot}") }

/// Everything every strewing laid, as text a script can read.
///
/// **For a check made outside the engine**, which is what an oracle is: the
/// ground a strewing stood on, the rule it was laid by and every copy and patch
/// it laid, so that a count per area, a slope and a band can be checked against
/// the triangles by arithmetic that shares nothing with the laying. A number is
/// written as the shortest text that reads back to the same float.
///
/// ```text
/// strewn <slot> <name> pieces=<n> patches=<n> drawn=<n> thinned=<0|1> digest=<hex>
/// rule <every field of the record, as field=value>
/// local <x y z> <qx qy qz qw> <sx sy sz>
/// bounds <low x y z> <high x y z>
/// ground <triangles>
/// <x y z of each of three corners>          one line a triangle
/// pieces <n>
/// <x y z> <qx qy qz qw> <size> <shade>      one line a copy
/// patches <n>
/// <first> <count> <low x y z> <high x y z> <largest>
/// end
/// ```
///
/// @param world - the world whose strewings to write
/// @return the text, one block a strewing, in slot order
pub(crate) fn written(world: &World) -> String {
	let mut lines: Vec<String> = Vec::new();

	for (slot, layout) in world.strewn.iter() {
		let laid = &layout.laid;
		let nothing = MeshData::default();
		let ground = world
			.meshes
			.get(layout.key.ground.0)
			.map_or(&nothing, Entry::value);
		let (low, high) = world
			.meshes
			.get(layout.key.mesh.0)
			.map_or_else(|| nothing.bounds(), |mesh| mesh.value().bounds());

		lines.push(format!(
			"strewn {slot} {} pieces={} patches={} drawn={} thinned={} digest={:#018X}",
			world.entities.name(layout.entity),
			laid.pieces.len(),
			laid.patches.len(),
			laid.drawn,
			u8::from(laid.thinned),
			laid.digest
		));
		lines.push(format!("rule {}", rule_of(&layout.key.rule)));
		lines.push(format!("local {}", transform_of(&layout.key.local)));
		lines.push(format!("bounds {}", floats(low.to_array().iter().chain(&high.to_array()))));
		lines.push(format!("ground {}", ground.indices.len() / 3));

		for corners in ground.indices.chunks_exact(3) {
			let places: Vec<f32> = corners
				.iter()
				.flat_map(|&index| {
					usize::try_from(index)
						.ok()
						.and_then(|at| ground.vertices.get(at))
						.map_or([f32::NAN; 3], |vertex| vertex.position)
				})
				.collect();

			lines.push(floats(places.iter()));
		}

		lines.push(format!("pieces {}", laid.pieces.len()));
		lines.extend(laid.pieces.iter().map(|piece| {
			floats(
				piece
					.at
					.iter()
					.chain(&piece.turn)
					.chain([&piece.size, &piece.shade]),
			)
		}));
		lines.push(format!("patches {}", laid.patches.len()));
		lines.extend(laid.patches.iter().map(|patch| {
			format!(
				"{} {} {}",
				patch.first,
				patch.count,
				floats(
					patch
						.low
						.iter()
						.chain(&patch.high)
						.chain([&patch.largest])
				)
			)
		}));
		lines.push("end".to_owned());
	}

	// a line ends every line, the last one included, so a file is lines
	lines.push(String::new());

	lines.join("\n")
}

/// A rule's every field, as `field=value` words.
fn rule_of(rule: &Strewing) -> String {
	format!(
		"strews={} seed={} density={} slope={} band={},{} size={},{} sink={},{} turns={} \
		 align={} tilt={} shade={} reach={} fade={} shadows={} solid={}",
		rule.strews,
		rule.seed,
		rule.density,
		rule.slope,
		rule.band[0],
		rule.band[1],
		rule.size[0],
		rule.size[1],
		rule.sink[0],
		rule.sink[1],
		rule.turns,
		rule.align,
		rule.tilt,
		rule.shade,
		rule.reach,
		rule.fade,
		rule.shadows,
		rule.solid
	)
}

/// A transform as ten numbers: place, turn, scale.
fn transform_of(transform: &Transform) -> String {
	floats(
		transform
			.position
			.to_array()
			.iter()
			.chain(&transform.rotation.to_array())
			.chain(&transform.scale.to_array()),
	)
}

/// Numbers as the shortest text that reads back to each, a space between.
fn floats<'a>(numbers: impl Iterator<Item = &'a f32>) -> String {
	numbers
		.map(ToString::to_string)
		.collect::<Vec<String>>()
		.join(" ")
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{MaterialId, Renderable, Terrain, mesh::cube},
		glam::Vec3,
	};

	use super::*;

	/// A world with ground and a strewing hung off it, the ground built.
	fn meadow(rule: Strewing) -> (World, EntityId, EntityId) {
		let mut world = World::new();
		let ground = world.entities.spawn_at(Transform::IDENTITY);
		let mesh = world
			.meshes
			.insert("ground", Terrain::of(5).build());
		let strewn = world.entities.spawn_at(Transform::IDENTITY);

		if let Some(renderable) = world.entities.renderable_mut(ground) {
			*renderable = Renderable::of(mesh, MaterialId::DEFAULT, Vec3::ONE);
		}

		if let Some(renderable) = world.entities.renderable_mut(strewn) {
			*renderable = Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, Vec3::ONE);
		}

		world.entities.set_parent(strewn, ground);

		if let Some(held) = world.entities.record_mut(&STREWING, strewn) {
			*held = rule;
		}

		(world, ground, strewn)
	}

	/// A rule that strews a few copies a unit.
	fn sparse() -> Strewing {
		Strewing {
			strews: 1,
			density: 0.25,
			..Strewing::NONE
		}
	}

	#[test]
	fn a_strewing_is_laid_once_and_then_only_when_something_it_stands_on_changes() {
		let (mut world, ground, strewn) = meadow(sparse());

		sync(&mut world);

		let first = world
			.strewn
			.get(strewn.slot())
			.map(|layout| (layout.revision, layout.laid.pieces.len()));

		assert!(first.is_some_and(|(_, pieces)| pieces > 1000), "laid: {first:?}");

		sync(&mut world);

		assert_eq!(
			world
				.strewn
				.get(strewn.slot())
				.map(|layout| layout.revision),
			first.map(|(revision, _)| revision),
			"a step that changed nothing lays nothing"
		);

		// the ground built again under the same name is another ground
		let rebuilt = world
			.meshes
			.insert("ground", Terrain::of(6).build());
		assert_eq!(
			Some(rebuilt),
			world
				.entities
				.renderable(ground)
				.map(|it| it.mesh),
			"the same handle"
		);

		sync(&mut world);

		assert_ne!(
			world
				.strewn
				.get(strewn.slot())
				.map(|layout| layout.revision),
			first.map(|(revision, _)| revision),
			"a ground built again lays the copies again"
		);
	}

	#[test]
	fn a_rule_changed_lays_again_and_one_turned_off_takes_everything_away() {
		let (mut world, _, strewn) = meadow(sparse());

		sync(&mut world);
		let before = world
			.strewn
			.get(strewn.slot())
			.map(|layout| layout.laid.digest);

		if let Some(rule) = world.entities.record_mut(&STREWING, strewn) {
			rule.seed = 9;
		}
		sync(&mut world);

		assert_ne!(
			world
				.strewn
				.get(strewn.slot())
				.map(|layout| layout.laid.digest),
			before,
			"another seed, other copies"
		);

		if let Some(rule) = world.entities.record_mut(&STREWING, strewn) {
			rule.strews = 0;
		}
		sync(&mut world);

		assert!(world.strewn.get(strewn.slot()).is_none(), "nothing strewn");
		assert_eq!(world.strewn.pieces(), 0);
	}

	#[test]
	fn a_strewing_that_hangs_off_nothing_lays_nothing_and_says_it_is_laid() {
		let (mut world, _, strewn) = meadow(sparse());

		world.entities.set_parent(strewn, EntityId::NONE);
		sync(&mut world);

		let layout = world.strewn.get(strewn.slot());

		assert!(
			layout.is_some_and(|layout| layout.laid.pieces.is_empty()),
			"nothing to stand on"
		);

		let revision = layout.map(|layout| layout.revision);
		sync(&mut world);

		assert_eq!(
			world
				.strewn
				.get(strewn.slot())
				.map(|layout| layout.revision),
			revision,
			"and it is not laid again every step"
		);
	}

	#[test]
	fn a_slot_handed_to_another_entity_is_laid_for_the_new_one() {
		let (mut world, ground, strewn) = meadow(sparse());

		sync(&mut world);
		world.entities.despawn(strewn);

		let other = world.entities.spawn_at(Transform::IDENTITY);
		assert_eq!(other.slot(), strewn.slot(), "the freed slot is handed out again");

		world.entities.set_parent(other, ground);
		if let Some(renderable) = world.entities.renderable_mut(other) {
			*renderable = Renderable::of(MeshId::SPHERE, MaterialId::DEFAULT, Vec3::ONE);
		}
		if let Some(rule) = world.entities.record_mut(&STREWING, other) {
			*rule = Strewing { seed: 3, ..sparse() };
		}

		sync(&mut world);

		assert_eq!(
			world
				.strewn
				.get(other.slot())
				.map(|layout| layout.entity),
			Some(other),
			"laid for whoever holds the slot now"
		);
	}

	#[test]
	fn a_solid_strewing_has_one_body_colliding_with_what_it_laid_and_none_when_it_is_not() {
		let (mut world, _, strewn) = meadow(Strewing { solid: 1, ..sparse() });

		sync(&mut world);

		let bodies = bodies_of(&world, strewn);
		assert_eq!(bodies.len(), 1, "one body");

		let solid = world
			.strewn
			.get(strewn.slot())
			.map_or(MeshId::NONE, |layout| layout.solid);
		let body = bodies
			.first()
			.and_then(|&body| world.bodies.get(body));

		assert!(solid.is_some(), "a mesh to collide with");
		assert_eq!(
			body.map(|body| body.shape.mesh),
			Some(solid),
			"and the body collides with it"
		);
		assert_eq!(body.map(|body| body.kind), Some(BodyKind::Static));

		// a second body a copy brought along is taken away, and a step changes
		// nothing else
		let extra = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::mesh(solid),
			Transform::IDENTITY,
		));
		if let Some(body) = world.bodies.get_mut(extra) {
			body.entity = strewn;
		}

		sync(&mut world);
		assert_eq!(bodies_of(&world, strewn).len(), 1, "still one");

		// the one body left, pointed at the mesh another slot laid - what a copy
		// of the entity brings along - is pointed back at this slot's
		let other = world
			.meshes
			.insert("strewn.99", MeshData::default());
		for body in bodies_of(&world, strewn) {
			if let Some(body) = world.bodies.get_mut(body) {
				body.shape = Shape::mesh(other);
			}
		}

		sync(&mut world);
		assert_eq!(
			bodies_of(&world, strewn)
				.first()
				.and_then(|&body| world.bodies.get(body))
				.map(|body| body.shape.mesh),
			Some(solid),
			"a copy's body collides with its own slot's mesh"
		);

		if let Some(rule) = world.entities.record_mut(&STREWING, strewn) {
			rule.solid = 0;
		}
		sync(&mut world);

		assert!(bodies_of(&world, strewn).is_empty(), "no body once it is not solid");
		assert!(
			world
				.meshes
				.get(solid)
				.is_some_and(|mesh| mesh.value().indices.is_empty()),
			"and the mesh it collided with emptied"
		);
	}

	#[test]
	fn a_ball_dropped_on_a_solid_copy_comes_to_rest_on_it() {
		// the ground has no body at all, so whatever holds the ball up is the
		// strewing's: the one thing the tests above cannot say, that the mesh
		// laid back together is geometry the solver collides against
		let mut world = World::new();
		let mut simulation = Box::new(colby_physics::Simulation::new());

		world.install_physics(simulation.table());

		let floor = world.meshes.insert(
			"floor",
			Terrain {
				size: 16.0,
				side: 17,
				height: 0.0,
				..Terrain::hills()
			}
			.build(),
		);
		let ground = world.entities.spawn_at(Transform::IDENTITY);
		let strewn = world.entities.spawn_at(Transform::IDENTITY);

		if let Some(renderable) = world.entities.renderable_mut(ground) {
			*renderable = Renderable::of(floor, MaterialId::DEFAULT, Vec3::ONE);
		}
		if let Some(renderable) = world.entities.renderable_mut(strewn) {
			*renderable = Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, Vec3::ONE);
		}
		world.entities.set_parent(strewn, ground);
		if let Some(rule) = world.entities.record_mut(&STREWING, strewn) {
			*rule = Strewing {
				density: 0.05,
				turns: 0,
				solid: 1,
				..sparse()
			};
		}

		sync(&mut world);

		let under = world
			.strewn
			.get(strewn.slot())
			.and_then(|layout| layout.laid.pieces.first().copied())
			.map(|piece| Vec3::from_array(piece.at))
			.expect("the floor holds a few copies");
		let dropped_at = under + Vec3::new(0.0, 5.0, 0.0);
		let ball = world.entities.spawn_at(Transform::at(dropped_at));
		let mut body = Body::new(BodyKind::Dynamic, Shape::ball(0.25), Transform::at(dropped_at));
		body.entity = ball;

		let dropped = world.bodies.spawn(body);

		world.dt = 1.0 / 60.0;
		for _ in 0..240 {
			sync(&mut world);
			simulation.step(&mut world);
		}

		let rest = world
			.bodies
			.get(dropped)
			.map(|body| body.transform.position)
			.expect("it is still a body");

		// the copy is a unit cube standing on the floor with its middle on it,
		// so its top is half a unit up and the ball's middle a quarter above that
		assert!(
			(rest.y - (under.y + 0.75)).abs() < 0.1,
			"it ended at {rest} over a copy at {under}: it did not come to rest on the copy"
		);
	}

	#[test]
	fn a_strewing_that_goes_takes_its_body_with_it() {
		let (mut world, _, strewn) = meadow(Strewing { solid: 1, ..sparse() });

		sync(&mut world);
		assert_eq!(bodies_of(&world, strewn).len(), 1);

		let solid = world
			.strewn
			.get(strewn.slot())
			.map_or(MeshId::NONE, |layout| layout.solid);

		if let Some(rule) = world.entities.record_mut(&STREWING, strewn) {
			rule.strews = 0;
		}
		sync(&mut world);

		assert!(bodies_of(&world, strewn).is_empty(), "taken away with the copies");
		assert!(
			world
				.meshes
				.get(solid)
				.is_some_and(|mesh| mesh.value().indices.is_empty()),
			"and the mesh it collided with emptied"
		);
	}

	#[test]
	fn turning_the_strewing_lays_its_copies_again_held_the_new_way() {
		let (mut world, _, strewn) = meadow(Strewing { turns: 0, ..sparse() });

		sync(&mut world);

		let turned = colby_core::glam::Quat::from_rotation_z(0.6);
		if let Some(local) = world.entities.transform_mut(strewn) {
			local.rotation = turned;
		}
		sync(&mut world);

		let first = world
			.strewn
			.get(strewn.slot())
			.and_then(|layout| layout.laid.pieces.first().copied())
			.expect("the hills hold copies");

		assert!(
			colby_core::glam::Quat::from_array(first.turn)
				.dot(turned)
				.abs() > 1.0 - 1.0e-6,
			"a copy that neither turns nor leans is held as the entity is now"
		);
	}

	#[test]
	fn the_text_carries_every_copy_and_the_ground_it_stood_on() {
		let (mut world, _, strewn) = meadow(sparse());

		world.entities.set_name(strewn, "grass");
		sync(&mut world);

		let text = written(&world);
		let layout = world.strewn.get(strewn.slot());
		let pieces = layout.map_or(0, |layout| layout.laid.pieces.len());
		let lines: Vec<&str> = text.lines().collect();

		assert!(
			lines.first().is_some_and(|line| line
				.starts_with(&format!("strewn {} grass pieces={pieces}", strewn.slot()))),
			"{:?}",
			lines.first()
		);
		assert_eq!(lines.last(), Some(&"end"));

		let triangles = Terrain::of(5).build().indices.len() / 3;
		assert!(
			lines.contains(&format!("ground {triangles}").as_str()),
			"the ground's own triangles"
		);
		assert!(lines.contains(&format!("pieces {pieces}").as_str()));

		// every number reads back to the float it was written from
		let first = layout
			.and_then(|layout| layout.laid.pieces.first())
			.copied();
		let at = lines
			.iter()
			.position(|line| line.starts_with("pieces "))
			.and_then(|index| lines.get(index + 1));
		let read: Vec<f32> = at
			.map(|line| {
				line.split(' ')
					.filter_map(|word| word.parse().ok())
					.collect()
			})
			.unwrap_or_default();

		assert_eq!(
			read,
			first
				.map(|piece| {
					let mut all = piece.at.to_vec();
					all.extend(piece.turn);
					all.extend([piece.size, piece.shade]);
					all
				})
				.unwrap_or_default(),
			"the first copy, to the bit"
		);
	}

	#[test]
	fn a_ground_with_its_mesh_taken_away_lays_nothing_and_the_copies_go() {
		let (mut world, ground, strewn) = meadow(sparse());

		sync(&mut world);
		assert!(world.strewn.pieces() > 0);

		if let Some(renderable) = world.entities.renderable_mut(ground) {
			renderable.mesh = MeshId::NONE;
		}
		sync(&mut world);

		assert_eq!(
			world
				.strewn
				.get(strewn.slot())
				.map(|layout| layout.laid.pieces.len()),
			Some(0),
			"a ground with no mesh holds no copies"
		);
	}

	#[test]
	fn a_ground_of_cubes_is_strewn_on_their_tops_alone() {
		let mut world = World::new();
		let ground = world.entities.spawn_at(Transform {
			scale: Vec3::new(8.0, 1.0, 8.0),
			..Transform::IDENTITY
		});
		let strewn = world.entities.spawn_at(Transform::IDENTITY);

		if let Some(renderable) = world.entities.renderable_mut(ground) {
			*renderable = Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, Vec3::ONE);
		}
		world.entities.set_parent(strewn, ground);
		if let Some(rule) = world.entities.record_mut(&STREWING, strewn) {
			*rule = Strewing { density: 400.0, ..sparse() };
		}

		sync(&mut world);

		let layout = world.strewn.get(strewn.slot());
		let pieces = layout.map_or(0, |layout| layout.laid.pieces.len());

		// the unit cube's top is one square unit of its own space: four hundred
		assert!(pieces.abs_diff(400) < 40, "{pieces} on the top");
		assert!(
			layout.is_some_and(|layout| layout
				.laid
				.pieces
				.iter()
				.all(|piece| (piece.at[1] - 0.5).abs() < 1.0e-6)),
			"every copy on the top face, in the cube's own space"
		);
		assert_eq!(cube().indices.len(), 36, "and a cube is what it was");
	}
}
