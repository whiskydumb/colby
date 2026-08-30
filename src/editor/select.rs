//! What a panel is pointing at, and what it does to what it points at.
//!
//! Everything here is a function of a [`World`] and there is no egui in it,
//! which is the whole reason it is its own module: a panel is checked by
//! looking at it, and this is the half that can be checked by running it.
//!
//! **A handle is the identity and a name is the way back to it.** A
//! [`Selection`] holds both. The handle is what everything is done through,
//! because it is unique and a name is not; the name is what finds the thing
//! again when the world is replaced underneath the panel - a scene loaded from
//! the console, a module reloaded onto a fresh arena, a play that was stopped
//! into a world put back around it. Without the name the selection would
//! simply go out every time any of those happened, which for the two of them
//! that put back *the same world* is plainly wrong.
//!
//! **An entity and the body driving it are one thing to a person and two
//! tables to the engine.** So moving either moves both, which is what
//! [`place`] is for. Nothing else would work: in edit mode no step runs, and
//! the step is the only thing that otherwise copies one into the other, so an
//! entity dragged on its own would snap back the moment play started.

use colby_core::abi::{BodyId, EntityId, JointId, Transform, World};

/// One thing in the world, whichever of the three tables it lives in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Pick {
	/// Nothing is selected.
	#[default]
	Nothing,

	/// An entity.
	Entity(EntityId),

	/// A body.
	Body(BodyId),

	/// A joint.
	Joint(JointId),
}

impl Pick {
	/// Whether the world still holds what this names.
	pub(crate) fn alive(self, world: &World) -> bool {
		match self {
			| Self::Nothing => false,
			| Self::Entity(id) => world.entities.alive(id),
			| Self::Body(id) => world.bodies.alive(id),
			| Self::Joint(id) => world.joints.alive(id),
		}
	}

	/// What it is called, or the empty string.
	pub(crate) fn name(self, world: &World) -> &str {
		match self {
			| Self::Nothing => "",
			| Self::Entity(id) => world.entities.name(id),
			| Self::Body(id) => world.bodies.name(id),
			| Self::Joint(id) => world.joints.name(id),
		}
	}
}

/// What the tree is pointing at, and what it was called when it was picked.
#[derive(Clone, Debug, Default)]
pub(crate) struct Selection {
	/// The handle, which is what everything is done through.
	at: Pick,

	/// The name it answered to when it was picked, or empty.
	///
	/// Only ever read by [`refresh`](Self::refresh), and only when the handle
	/// has stopped resolving.
	name: String,
}

impl Selection {
	/// What is selected.
	pub(crate) const fn at(&self) -> Pick { self.at }

	/// Whether a particular thing is the selected one.
	pub(crate) fn is(&self, pick: Pick) -> bool { self.at == pick }

	/// Selects something, remembering what it is called.
	///
	/// @param world - where the name is read from
	/// @param pick - what to select
	pub(crate) fn set(&mut self, world: &World, pick: Pick) {
		self.at = pick;
		pick.name(world).clone_into(&mut self.name);
	}

	/// Selects nothing.
	pub(crate) fn clear(&mut self) {
		self.at = Pick::Nothing;
		self.name.clear();
	}

	/// Finds the selection again if the world was replaced under it.
	///
	/// Called once a frame, before anything is drawn. A handle that still
	/// resolves is left exactly alone - that is the ordinary case and it costs
	/// one lookup. A handle that does not is looked for by name in the table
	/// it came from, and a name nothing answers to clears the selection rather
	/// than leaving it pointing at a thing that is gone.
	///
	/// @param world - the world as it now is
	pub(crate) fn refresh(&mut self, world: &World) {
		if self.at.alive(world) {
			// the name may have been edited since, here or by anything else
			// holding the world. What is remembered is what it is called now.
			self.at.name(world).clone_into(&mut self.name);

			return;
		}

		match again(world, self.at, &self.name) {
			// nothing answers to it any more, so neither the handle nor the
			// name is worth holding: a name with no handle beside it could
			// only ever match something that has not been created yet, which
			// is not the same thing and would be a surprise.
			| Pick::Nothing => self.clear(),
			| found => self.at = found,
		}
	}
}

/// Whatever now answers to a name, in the table a pick came from.
///
/// The first match wins. Names are not unique - two copies of one prop share a
/// name by construction - and there is nothing better to do about that here:
/// the alternative is refusing to find either.
fn again(world: &World, was: Pick, name: &str) -> Pick {
	if name.is_empty() {
		return Pick::Nothing;
	}

	match was {
		| Pick::Nothing => Pick::Nothing,
		| Pick::Entity(_) => world
			.entities
			.iter()
			.map(|(id, ..)| id)
			.find(|&id| world.entities.name(id) == name)
			.map_or(Pick::Nothing, Pick::Entity),
		| Pick::Body(_) => world
			.bodies
			.iter()
			.map(|(id, _)| id)
			.find(|&id| world.bodies.name(id) == name)
			.map_or(Pick::Nothing, Pick::Body),
		| Pick::Joint(_) => world
			.joints
			.iter()
			.map(|(id, _)| id)
			.find(|&id| world.joints.name(id) == name)
			.map_or(Pick::Nothing, Pick::Joint),
	}
}

/// Where something is, if it is the sort of thing that is anywhere.
///
/// A joint is not: it is a relationship between two bodies, and its anchors are
/// in their spaces rather than in the world's.
///
/// @param world - what to look in
/// @param at - what to look for
pub(crate) fn transform(world: &World, at: Pick) -> Option<Transform> {
	match at {
		| Pick::Entity(id) => world.entities.transform(id).copied(),
		| Pick::Body(id) => world.bodies.get(id).map(|body| body.transform),
		| Pick::Nothing | Pick::Joint(_) => None,
	}
}

/// Puts something where it is asked to go, and everything describing it with
/// it.
///
/// An entity with a body under it is moved through the body, because that is
/// the call that writes both and says the thing cut rather than traveled. An
/// entity with no body is written directly and snapped, which is the same
/// thing without the body half.
///
/// @param world - the world to write
/// @param at - what to move
/// @param transform - where it now is
/// @return `true` if anything was moved
pub(crate) fn place(world: &mut World, at: Pick, transform: Transform) -> bool {
	match at {
		| Pick::Entity(id) => {
			if let Some(body) = driver(world, id) {
				return world.teleport_body(body, transform);
			}

			let Some(held) = world.entities.transform_mut(id) else {
				return false;
			};

			*held = transform;
			// dragged, not traveled. Only play mode blends at all - a world
			// being edited is drawn as it stands - so this is about the
			// inspector being used while the game runs. @ref `crate::mode`
			// in the runner.
			world.entities.snap(id);

			true
		},
		| Pick::Body(id) => world.teleport_body(id, transform),
		| Pick::Nothing | Pick::Joint(_) => false,
	}
}

/// Renames whatever is picked.
///
/// @param world - the world to write
/// @param at - what to rename
/// @param name - what to call it
/// @return `true` if the handle resolved
pub(crate) fn rename(world: &mut World, at: Pick, name: &str) -> bool {
	match at {
		| Pick::Entity(id) => world.entities.set_name(id, name),
		| Pick::Body(id) => world.bodies.set_name(id, name),
		| Pick::Joint(id) => world.joints.set_name(id, name),
		| Pick::Nothing => false,
	}
}

/// The body driving an entity, if one does.
///
/// The first one found. Nothing stops two bodies naming one entity, and if two
/// do then the entity is being driven by whichever the solver visits last -
/// so picking the first here is no more arbitrary than the situation already
/// is.
pub(crate) fn driver(world: &World, id: EntityId) -> Option<BodyId> {
	if !id.is_some() {
		return None;
	}

	world
		.bodies
		.iter()
		.find(|(_, body)| body.entity == id)
		.map(|(body, _)| body)
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{Body, BodyKind, Joint, Shape},
		glam::Vec3,
	};

	use super::*;

	/// A world with a named entity, the body under it, and a joint.
	fn peopled() -> (World, EntityId, BodyId, JointId) {
		let mut world = World::new();

		let entity = world.entities.spawn_at(Transform::at(Vec3::Y));
		world.entities.set_name(entity, "crate");

		let body = world
			.bodies
			.spawn(Body::dynamic(Shape::ball(0.5), Transform::at(Vec3::Y), 1.0).driving(entity));
		world.bodies.set_name(body, "crate body");

		let joint =
			world
				.joints
				.spawn(Joint::rope(body, BodyId::NONE, (Vec3::ZERO, Vec3::Y * 4.0), 2.0));
		world.joints.set_name(joint, "rope");

		(world, entity, body, joint)
	}

	#[test]
	fn a_selection_that_still_resolves_is_left_alone() {
		let (world, entity, ..) = peopled();
		let mut selection = Selection::default();
		selection.set(&world, Pick::Entity(entity));

		selection.refresh(&world);

		assert_eq!(selection.at(), Pick::Entity(entity), "nothing happened to it");
		assert!(selection.is(Pick::Entity(entity)));
	}

	#[test]
	fn a_selection_finds_itself_again_in_a_world_that_was_replaced() {
		let (mut world, entity, ..) = peopled();
		let mut selection = Selection::default();
		selection.set(&world, Pick::Entity(entity));

		// the same thing by name, in a different slot: what a scene loaded
		// from the console, or a module reload onto a fresh arena, leaves
		// behind.
		world.entities.clear();
		let first = world.entities.spawn_at(Transform::at(Vec3::X));
		world.entities.set_name(first, "floor");
		let moved = world.entities.spawn_at(Transform::at(Vec3::Z));
		world.entities.set_name(moved, "crate");

		selection.refresh(&world);

		assert_ne!(moved, entity, "it really is a different handle");
		assert_eq!(selection.at(), Pick::Entity(moved), "and the name found it again");
	}

	#[test]
	fn a_selection_with_no_name_goes_out_when_its_handle_does() {
		let (mut world, ..) = peopled();
		let unnamed = world.entities.spawn_at(Transform::IDENTITY);
		// a second thing with no name, and it outlives the first. Without it
		// this test passes for the wrong reason: looking the empty name up
		// would find nothing anyway, and the code that refuses to look would
		// never be the thing under test. @ref `again`.
		let other = world.entities.spawn_at(Transform::at(Vec3::Z));

		let mut selection = Selection::default();
		selection.set(&world, Pick::Entity(unnamed));

		world.entities.despawn(unnamed);
		selection.refresh(&world);

		assert!(world.entities.alive(other), "something unnamed is still standing there");
		assert_eq!(
			selection.at(),
			Pick::Nothing,
			"and it is not what was selected: the empty name is the absence of one"
		);
	}

	#[test]
	fn a_selection_whose_name_nothing_answers_to_goes_out() {
		let (mut world, entity, ..) = peopled();
		let mut selection = Selection::default();
		selection.set(&world, Pick::Entity(entity));

		world.entities.clear();
		let other = world.entities.spawn_at(Transform::IDENTITY);
		world.entities.set_name(other, "something else");

		selection.refresh(&world);

		assert_eq!(selection.at(), Pick::Nothing, "the name is gone, so the selection is");
	}

	#[test]
	fn a_name_is_looked_for_in_the_table_it_came_from() {
		let (mut world, _, body, _) = peopled();
		let mut selection = Selection::default();
		selection.set(&world, Pick::Body(body));

		// an *entity* now answers to the body's name, and the body does not.
		world.bodies.clear();
		let decoy = world.entities.spawn_at(Transform::IDENTITY);
		world.entities.set_name(decoy, "crate body");

		selection.refresh(&world);

		assert_eq!(
			selection.at(),
			Pick::Nothing,
			"a body is not found again by something else wearing its name"
		);
	}

	#[test]
	fn moving_an_entity_moves_the_body_under_it() {
		let (mut world, entity, body, _) = peopled();
		let put = Transform::at(Vec3::new(3.0, 4.0, 5.0));

		assert!(place(&mut world, Pick::Entity(entity), put), "something moved");

		assert_eq!(
			world
				.entities
				.transform(entity)
				.map(|it| it.position),
			Some(put.position),
			"the entity went where it was put"
		);
		assert_eq!(
			world
				.bodies
				.get(body)
				.map(|it| it.transform.position),
			Some(put.position),
			"and the body went with it, or play would snap it back"
		);
	}

	#[test]
	fn moving_a_body_moves_the_entity_it_drives() {
		let (mut world, entity, body, _) = peopled();
		let put = Transform::at(Vec3::new(-1.0, 2.0, -3.0));

		assert!(place(&mut world, Pick::Body(body), put));

		assert_eq!(
			world
				.entities
				.transform(entity)
				.map(|it| it.position),
			Some(put.position),
			"the same pair, written from the other end"
		);
	}

	#[test]
	fn an_entity_with_nothing_under_it_moves_on_its_own() {
		let (mut world, ..) = peopled();
		let lone = world.entities.spawn_at(Transform::IDENTITY);
		let put = Transform::at(Vec3::X);

		assert!(place(&mut world, Pick::Entity(lone), put), "it still moves");
		assert_eq!(
			world
				.entities
				.transform(lone)
				.map(|it| it.position),
			Some(put.position)
		);
		assert!(driver(&world, lone).is_none(), "and there was nothing under it");
	}

	#[test]
	fn nothing_and_a_joint_are_not_things_that_are_anywhere() {
		let (mut world, _, _, joint) = peopled();

		assert!(transform(&world, Pick::Joint(joint)).is_none(), "a joint is a relationship");
		assert!(transform(&world, Pick::Nothing).is_none());
		assert!(!place(&mut world, Pick::Joint(joint), Transform::IDENTITY));
		assert!(!place(&mut world, Pick::Nothing, Transform::IDENTITY));
	}

	#[test]
	fn an_entity_moved_on_its_own_is_not_drawn_traveling() {
		let (mut world, ..) = peopled();
		let lone = world.entities.spawn_at(Transform::IDENTITY);

		// a step boundary, so that where it was and where it is are the same
		// and the move below is the only thing between them.
		world.entities.advance();
		place(&mut world, Pick::Entity(lone), Transform::at(Vec3::new(9.0, 0.0, 0.0)));
		world.entities.settle();

		assert_eq!(
			world
				.entities
				.interpolated(lone, 0.0)
				.map(|it| it.position),
			Some(Vec3::new(9.0, 0.0, 0.0)),
			"the start of the blend is where it was put, not where it used to be"
		);
	}

	#[test]
	fn a_handle_to_nothing_does_not_reach_the_bodies_that_drive_nothing() {
		let (mut world, ..) = peopled();
		let floor = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::ball(1.0),
			Transform::IDENTITY,
		));

		assert!(driver(&world, EntityId::NONE).is_none(), "a null handle drives nothing");
		assert!(
			!place(&mut world, Pick::Entity(EntityId::NONE), Transform::at(Vec3::X)),
			"and moving it moves nothing"
		);
		assert_eq!(
			world
				.bodies
				.get(floor)
				.map(|it| it.transform.position),
			Some(Vec3::ZERO),
			"the floor in particular, whose entity is also nothing, stayed put"
		);
	}

	#[test]
	fn a_transform_is_read_from_the_table_it_was_asked_about() {
		let (mut world, entity, body, _) = peopled();

		// the two are usually equal, so make them differ: this is the write a
		// game makes when it moves a body without telling the entity.
		if let Some(held) = world.bodies.get_mut(body) {
			held.transform.position = Vec3::new(7.0, 0.0, 0.0);
		}

		assert_eq!(
			transform(&world, Pick::Body(body)).map(|it| it.position),
			Some(Vec3::new(7.0, 0.0, 0.0)),
			"the body's own"
		);
		assert_eq!(
			transform(&world, Pick::Entity(entity)).map(|it| it.position),
			Some(Vec3::Y),
			"and the entity's own, which is a different number"
		);
	}

	#[test]
	fn everything_can_be_renamed_in_its_own_table() {
		let (mut world, entity, body, joint) = peopled();

		assert!(rename(&mut world, Pick::Entity(entity), "box"));
		assert!(rename(&mut world, Pick::Body(body), "box body"));
		assert!(rename(&mut world, Pick::Joint(joint), "string"));

		assert_eq!(Pick::Entity(entity).name(&world), "box");
		assert_eq!(Pick::Body(body).name(&world), "box body");
		assert_eq!(Pick::Joint(joint).name(&world), "string");
		assert!(!rename(&mut world, Pick::Nothing, "nowhere"));
	}

	#[test]
	fn a_rename_is_what_the_selection_remembers_afterwards() {
		let (mut world, entity, ..) = peopled();
		let mut selection = Selection::default();
		selection.set(&world, Pick::Entity(entity));

		rename(&mut world, Pick::Entity(entity), "barrel");
		selection.refresh(&world);

		// and now the handle goes stale, so only the remembered name can find
		// it. If `refresh` had kept the name it was picked under, this would
		// look for "crate" and find nothing.
		world.entities.clear();
		let again = world.entities.spawn_at(Transform::IDENTITY);
		world.entities.set_name(again, "barrel");
		selection.refresh(&world);

		assert_eq!(selection.at(), Pick::Entity(again), "it followed the rename");
	}
}
