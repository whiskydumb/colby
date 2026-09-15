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

use colby_asset::compile::Kind;
use colby_core::{
	abi::{
		Body, BodyId, BodyKind, Decal, EntityId, JointId, MaterialId, MeshId, ModelId,
		Renderable, Shape, Transform, Water, World, material, record, scene,
	},
	glam::Vec3,
};

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

	/// A material, which is an *asset* rather than a thing standing anywhere.
	///
	/// The one member of this enumeration that is not in the world in the
	/// sense the others are: a material is a row in a registry that a file
	/// filled, and what selects one is a click in the asset browser rather
	/// than a click in the picture. It is here all the same, because what
	/// follows from being selected - the inspector shows you its fields, and
	/// the field table draws them - is the same thing, and a second mechanism
	/// beside [`Selection`] would be a second mechanism for one word.
	///
	/// **A [`MaterialId`] is not generational**, so this never stops
	/// resolving the way the three above do; what saves it is the name kept
	/// beside it in [`Selection`], which is re-read every frame and is what a
	/// registry rebuilt by a reload would have moved.
	Material(MaterialId),

	/// A model, which is an *asset* like a material and not a thing standing
	/// anywhere either.
	///
	/// The sixth, and the second of the two that are not in the world. What
	/// it is worth selecting for is different from a material's: a material
	/// is shown so that its numbers can be *changed*, and a model is shown so
	/// that what the compiler made of a file can be *read* - what pieces came
	/// out of it, what each is made of, and whether a sidecar beside the
	/// source had a hand in any of it. @ref `colby_asset::import`.
	///
	/// A [`ModelId`] is not generational, for the reason a [`MaterialId`] is
	/// not, and the name beside it in [`Selection`] does the same work.
	Model(ModelId),
}

impl Pick {
	/// Whether the world still holds what this names.
	pub(crate) fn alive(self, world: &World) -> bool {
		match self {
			| Self::Nothing => false,
			| Self::Entity(id) => world.entities.alive(id),
			| Self::Body(id) => world.bodies.alive(id),
			| Self::Joint(id) => world.joints.alive(id),
			// a registry hands out no slot it does not hold, and never takes
			// one back: what a reload does to a material is replace what is
			// in the slot, which is a different question and the one the name
			// beside it in `Selection` answers.
			| Self::Material(id) => world.materials.get(id).is_some(),
			| Self::Model(id) => world.models.get(id).is_some(),
		}
	}

	/// What it is called, or the empty string.
	pub(crate) fn name(self, world: &World) -> &str {
		match self {
			| Self::Nothing => "",
			| Self::Entity(id) => world.entities.name(id),
			| Self::Body(id) => world.bodies.name(id),
			| Self::Joint(id) => world.joints.name(id),
			| Self::Material(id) => world.materials.name(id),
			| Self::Model(id) => world.models.name(id),
		}
	}
}

/// What the panels are pointing at: one thing or several, each with the
/// name it answered to when it was picked.
///
/// **The last one picked is the primary**: the gizmo hangs off it, the
/// inspector shows it, and a drag moves everything else by what it did to
/// the primary. Godot and Unity both put the handles on the last thing
/// clicked, and it is the one the person is looking at.
#[derive(Clone, Debug, Default)]
pub(crate) struct Selection {
	/// Everything selected, in the order it was picked, the primary last.
	///
	/// The name is only ever read by [`refresh`](Self::refresh), and only
	/// when the handle has stopped resolving.
	held: Vec<(Pick, String)>,
}

impl Selection {
	/// The primary: the last thing picked, or nothing.
	pub(crate) fn at(&self) -> Pick {
		self.held
			.last()
			.map_or(Pick::Nothing, |(pick, _)| *pick)
	}

	/// Whether a particular thing is among the selected.
	pub(crate) fn is(&self, pick: Pick) -> bool {
		pick != Pick::Nothing && self.held.iter().any(|(held, _)| *held == pick)
	}

	/// How many things are selected.
	pub(crate) fn len(&self) -> usize { self.held.len() }

	/// Everything selected, the primary last.
	pub(crate) fn picks(&self) -> Vec<Pick> { self.held.iter().map(|(pick, _)| *pick).collect() }

	/// Everything selected but the primary.
	pub(crate) fn others(&self) -> Vec<Pick> {
		let count = self.held.len().saturating_sub(1);

		self.held
			.iter()
			.take(count)
			.map(|(pick, _)| *pick)
			.collect()
	}

	/// Selects one thing and nothing else, remembering what it is called.
	///
	/// [`Pick::Nothing`] selects nothing at all, which is what a click on
	/// empty space means.
	///
	/// @param world - where the name is read from
	/// @param pick - what to select
	pub(crate) fn set(&mut self, world: &World, pick: Pick) {
		self.held.clear();

		if pick != Pick::Nothing {
			self.held
				.push((pick, pick.name(world).to_owned()));
		}
	}

	/// Adds a thing to the selection, or takes it out if it was in.
	///
	/// A thing added becomes the primary; a thing taken out leaves whatever
	/// was picked before it as the primary. Nothing at all is neither added
	/// nor taken out.
	///
	/// @param world - where the name is read from
	/// @param pick - what to add or take out
	pub(crate) fn toggle(&mut self, world: &World, pick: Pick) {
		if pick == Pick::Nothing {
			return;
		}

		if let Some(index) = self
			.held
			.iter()
			.position(|(held, _)| *held == pick)
		{
			self.held.remove(index);
		} else {
			self.held
				.push((pick, pick.name(world).to_owned()));
		}
	}

	/// Selects nothing.
	pub(crate) fn clear(&mut self) { self.held.clear(); }

	/// Finds the selection again if the world was replaced under it.
	///
	/// Called once a frame, before anything is drawn. A handle that still
	/// resolves is left exactly alone - that is the ordinary case and it costs
	/// one lookup. A handle that does not is looked for by name in the table
	/// it came from, and a name nothing answers to drops that entry rather
	/// than leaving it pointing at a thing that is gone.
	///
	/// @param world - the world as it now is
	pub(crate) fn refresh(&mut self, world: &World) {
		for (pick, name) in &mut self.held {
			if pick.alive(world) {
				// the name may have been edited since, here or by anything
				// else holding the world. What is remembered is what it is
				// called now.
				pick.name(world).clone_into(name);
			} else {
				*pick = again(world, *pick, name);
			}
		}

		// nothing answers to it any more, so neither the handle nor the name
		// is worth holding: a name with no handle beside it could only ever
		// match something that has not been created yet, which is not the
		// same thing and would be a surprise.
		self.held
			.retain(|(pick, _)| *pick != Pick::Nothing);
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
		// a registry's slot for a name is the answer, and a material's is
		// the only one of the four that a lookup can give directly.
		| Pick::Material(_) => match world.materials.find(name) {
			| found if found.is_some() => Pick::Material(found),
			| _ => Pick::Nothing,
		},
		| Pick::Model(_) => match world.models.find(name) {
			| found if found.is_some() => Pick::Model(found),
			| _ => Pick::Nothing,
		},
		| Pick::Joint(_) => world
			.joints
			.iter()
			.map(|(id, _)| id)
			.find(|&id| world.joints.name(id) == name)
			.map_or(Pick::Nothing, Pick::Joint),
	}
}

/// Where something is in the world, if it is the sort of thing that is
/// anywhere.
///
/// A joint is not: it is a relationship between two bodies, and its anchors are
/// in their spaces rather than in the world's. An entity hanging off another
/// is placed through it, because this is what a gizmo works in; @ref
/// [`local`] for what an inspector shows.
///
/// @param world - what to look in
/// @param at - what to look for
pub(crate) fn transform(world: &World, at: Pick) -> Option<Transform> {
	match at {
		| Pick::Entity(id) => world.entities.placed(id),
		| Pick::Body(id) => world.bodies.get(id).map(|body| body.transform),
		// a material does not stand anywhere, which is the one thing that
		// separates it from the three above.
		| Pick::Nothing | Pick::Joint(_) | Pick::Material(_) | Pick::Model(_) => None,
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

			// as a place in the world, whatever it hangs off
			if !world.entities.set_placed(id, transform) {
				return false;
			}

			// dragged, not traveled. Only play mode blends at all - a world
			// being edited is drawn as it stands - so this is about the
			// inspector being used while the game runs. @ref `crate::mode`
			// in the runner.
			world.entities.snap(id);

			true
		},
		| Pick::Body(id) => world.teleport_body(id, transform),
		| Pick::Nothing | Pick::Joint(_) | Pick::Material(_) | Pick::Model(_) => false,
	}
}

/// Where something is in its own terms: inside its parent for an entity that
/// hangs off one, and the same as [`transform`] for everything else.
///
/// What an inspector shows and edits, the way every editor checked shows a
/// child's numbers relative to its parent while its gizmo works in the world.
///
/// @param world - what to look in
/// @param at - what to look for
pub(crate) fn local(world: &World, at: Pick) -> Option<Transform> {
	match at {
		| Pick::Entity(id) if world.entities.parent(id).is_some() =>
			world.entities.transform(id).copied(),
		| _ => transform(world, at),
	}
}

/// Puts something where it is asked to go, in its own terms. @ref [`local`].
///
/// @param world - the world to write
/// @param at - what to move
/// @param transform - where it now is, inside its parent for an entity that
/// hangs off one
/// @return `true` if anything was moved
pub(crate) fn place_local(world: &mut World, at: Pick, transform: Transform) -> bool {
	match at {
		| Pick::Entity(id) if world.entities.parent(id).is_some() => {
			let Some(parent) = world.entities.placed(world.entities.parent(id)) else {
				return false;
			};

			place(world, at, parent.then(transform))
		},
		| _ => place(world, at, transform),
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
		// a material and a model are called what their *files* are called,
		// and renaming a file is not something a panel does behind
		// somebody's back.
		| Pick::Nothing | Pick::Material(_) | Pick::Model(_) => false,
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

/// The entity a body drives, if it is still there.
pub(crate) fn drives(world: &World, body: BodyId) -> Option<EntityId> {
	let entity = world.bodies.get(body)?.entity;

	world.entities.alive(entity).then_some(entity)
}

/// Hangs an entity off another, or stands it on its own, without moving it.
///
/// The thing stays exactly where it is in the world and only what it is
/// measured from changes - which is what every editor checked does when a
/// row is dropped onto another, and what a person dragging a wheel under a
/// car means. The entity's own transform is rewritten as its place inside
/// the new parent, @ref `Entities::set_placed`; a body under it is where it
/// was and stays there, because a body's place is in the world already.
///
/// @param world - the world to write
/// @param child - what to hang, or to take down
/// @param parent - what to hang it off, or [`EntityId::NONE`] to stand it on
/// its own
/// @return `true` if it was done; `false` for a stale handle, a parent that
/// is not alive, an entity hanging off itself, or a loop
pub(crate) fn hang(world: &mut World, child: EntityId, parent: EntityId) -> bool {
	let Some(placed) = world.entities.placed(child) else {
		return false;
	};

	if !world.entities.set_parent(child, parent) {
		return false;
	}

	if !world.entities.set_placed(child, placed) {
		return false;
	}

	// dragged, not traveled: the same rule `place` follows, and for the same
	// reason.
	world.entities.snap(child);

	true
}

/// What to call an entity in a panel: its name, or what it is made of in
/// angle brackets, so that a name and a description can never be mistaken
/// for each other.
///
/// From the world rather than from a panel, because a panel that kept its
/// own names would lose them the moment a scene was loaded.
pub(crate) fn entity_label(world: &World, id: EntityId) -> String {
	let name = world.entities.name(id);
	if !name.is_empty() {
		return name.to_owned();
	}

	let mesh = world
		.entities
		.renderable(id)
		.map(|renderable| renderable.mesh)
		.and_then(|mesh| world.meshes.get(mesh))
		.map_or("", |entry| entry.name());

	if mesh.is_empty() {
		format!("<entity {}>", id.slot())
	} else {
		format!("<{mesh}>")
	}
}

/// What to call a body in a panel. @ref [`entity_label`].
pub(crate) fn body_label(world: &World, id: BodyId) -> String {
	let name = world.bodies.name(id);
	if !name.is_empty() {
		return name.to_owned();
	}

	world.bodies.get(id).map_or_else(
		|| format!("<body {}>", id.slot()),
		|body| format!("<{} {}>", body_words(body), id.slot()),
	)
}

/// What to call a joint in a panel. @ref [`entity_label`].
pub(crate) fn joint_label(world: &World, id: JointId) -> String {
	let name = world.joints.name(id);
	if !name.is_empty() {
		return name.to_owned();
	}

	world.joints.get(id).map_or_else(
		|| format!("<joint {}>", id.slot()),
		|joint| format!("<{} {}>", joint.kind.word(), id.slot()),
	)
}

/// What a body is, in the fewest words that say it.
///
/// The words the format writes a body with, so that a row in a panel and a
/// line in a scene file agree; the tree used to have a vocabulary of its own,
/// and the moment the two had to be the same word was the moment the words
/// went on the kinds themselves.
pub(crate) fn body_words(body: &Body) -> String {
	let kind = body.kind.word();
	let shape = body.shape.kind.word();

	if body.sensor {
		// the first thing anybody wants to know about one, because a sensor is
		// the body that is there and does not push.
		format!("{kind} {shape} sensor")
	} else {
		format!("{kind} {shape}")
	}
}

/// Every entity hanging off one, directly or through others, in slot order.
///
/// A scan of the table rather than a list kept on the parent: the table has
/// no children walk, and the panels ask this a few times a frame at most.
///
/// @param world - what to look in
/// @param id - the ancestor
pub(crate) fn descendants(world: &World, id: EntityId) -> Vec<EntityId> {
	if !world.entities.alive(id) {
		return Vec::new();
	}

	world
		.entities
		.iter()
		.map(|(candidate, ..)| candidate)
		.filter(|&candidate| candidate != id && hangs_off(world, candidate, id))
		.collect()
}

/// Whether an entity hangs off another, at any distance.
fn hangs_off(world: &World, child: EntityId, ancestor: EntityId) -> bool {
	let mut above = world.entities.parent(child);

	// bounded like the walk in `Entities::set_parent`, and for the same
	// reason: a loop cannot be made, and a walk that could not end must.
	for _ in 0..colby_core::abi::MAX_ENTITIES {
		if !above.is_some() {
			return false;
		}

		if above == ancestor {
			return true;
		}

		above = world.entities.parent(above);
	}

	false
}

/// What a deletion took with it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Deleted {
	/// Entities, the selected ones and everything hanging off them.
	pub(crate) entities: usize,

	/// Bodies, the selected ones and every one driving an entity that went.
	pub(crate) bodies: usize,

	/// Joints, the selected ones and every one holding a body that went.
	pub(crate) joints: usize,
}

/// Every entity picked, with everything hanging off each: once, alive, in
/// slot order.
///
/// What a deletion takes and what a duplicate copies, because Godot and
/// Unity both act on a branch rather than a node. Slot order so that a
/// parent comes before its child whenever the table's order says so.
fn branches(world: &World, picks: &[Pick]) -> Vec<EntityId> {
	let mut found: Vec<EntityId> = Vec::new();

	for &pick in picks {
		let Pick::Entity(id) = pick else {
			continue;
		};

		for each in std::iter::once(id).chain(descendants(world, id)) {
			if world.entities.alive(each) && !found.contains(&each) {
				found.push(each);
			}
		}
	}

	found.sort_by_key(|id| id.slot());

	found
}

/// The bodies picked on their own, alive, once.
fn picked_bodies(world: &World, picks: &[Pick]) -> Vec<BodyId> {
	let mut found: Vec<BodyId> = Vec::new();

	for &pick in picks {
		if let Pick::Body(id) = pick
			&& world.bodies.alive(id)
			&& !found.contains(&id)
		{
			found.push(id);
		}
	}

	found
}

/// The joints picked on their own, alive, once.
fn picked_joints(world: &World, picks: &[Pick]) -> Vec<JointId> {
	let mut found: Vec<JointId> = Vec::new();

	for &pick in picks {
		if let Pick::Joint(id) = pick
			&& world.joints.alive(id)
			&& !found.contains(&id)
		{
			found.push(id);
		}
	}

	found
}

/// Removes what is picked, and everything that could not stand without it.
///
/// An entity goes with everything hanging off it - Godot and Unity both
/// delete a branch, not a node - and with the bodies driving any of them,
/// because a body left behind would be a collider nobody can see. A body
/// goes with the joints holding it, because a joint holding a body that is
/// gone holds nothing. A joint goes alone.
///
/// @param world - the world to write
/// @param picks - what was selected
/// @return how many of each went
pub(crate) fn delete(world: &mut World, picks: &[Pick]) -> Deleted {
	let entities = branches(world, picks);
	let mut bodies = picked_bodies(world, picks);
	let mut joints = picked_joints(world, picks);

	// the bodies driving an entity that goes, then the joints holding a body
	// that goes: each list grows from the one before it.
	for (body, held) in world.bodies.iter() {
		if entities.contains(&held.entity) && !bodies.contains(&body) {
			bodies.push(body);
		}
	}

	for (joint, held) in world.joints.iter() {
		if (bodies.contains(&held.first) || bodies.contains(&held.second))
			&& !joints.contains(&joint)
		{
			joints.push(joint);
		}
	}

	for joint in &joints {
		world.joints.despawn(*joint);
	}

	for body in &bodies {
		world.bodies.despawn(*body);
	}

	for entity in &entities {
		world.entities.despawn(*entity);
	}

	Deleted {
		entities: entities.len(),
		bodies: bodies.len(),
		joints: joints.len(),
	}
}

/// What a handle became when its thing was copied, if it was.
fn became<T: PartialEq + Copy>(copies: &[(T, T)], was: T) -> Option<T> {
	copies
		.iter()
		.find(|(from, _)| *from == was)
		.map(|(_, to)| *to)
}

/// Makes a copy of what is picked, beside the original and in its place.
///
/// An entity is copied with everything hanging off it, its name, its look
/// and its place; a copy hangs off the copy of its parent when the parent
/// was copied and off the same parent otherwise, which is where Godot and
/// Unity put a duplicate. The bodies driving copied entities are copied
/// driving the copies; a body picked on its own is copied driving nothing;
/// a joint is copied when the body it holds first was copied, holding the
/// copies, and skipped otherwise. Names are kept as they are - the world
/// does not mind two things with one name, and the scene writer numbers
/// them on the way out.
///
/// @param world - the world to write
/// @param picks - what was selected
/// @return the copies, in the order the originals were picked, the copy of
/// the primary last: what the selection becomes
pub(crate) fn duplicate(world: &mut World, picks: &[Pick]) -> Vec<Pick> {
	let copies = copy_entities(world, &branches(world, picks));
	let body_copies = copy_bodies(world, picks, &copies);
	let joint_copies = copy_joints(world, &body_copies);

	picks
		.iter()
		.filter_map(|pick| match *pick {
			| Pick::Entity(id) => became(&copies, id).map(Pick::Entity),
			| Pick::Body(id) => became(&body_copies, id).map(Pick::Body),
			| Pick::Joint(id) => became(&joint_copies, id).map(Pick::Joint),
			// duplicating a material or a model would be duplicating a
			// *file*, which is the browser's business and not a selection's.
			| Pick::Nothing | Pick::Material(_) | Pick::Model(_) => None,
		})
		.collect()
}

/// Copies entities, then hangs the copies the way the originals hang.
///
/// @param sources - what to copy, a parent before its child where the
/// table's order says so; hung in a second pass so that it does not matter
/// where it does not
/// @return each original with its copy
fn copy_entities(world: &mut World, sources: &[EntityId]) -> Vec<(EntityId, EntityId)> {
	let mut copies: Vec<(EntityId, EntityId)> = Vec::new();

	for &source in sources {
		let Some(transform) = world.entities.transform(source).copied() else {
			continue;
		};

		let copy = world.entities.spawn_at(transform);
		if !copy.is_some() {
			continue;
		}

		if let Some(renderable) = world.entities.renderable(source).copied() {
			world.entities.set_renderable(copy, renderable);
		}

		if let Some(light) = world.entities.light(source).copied() {
			world.entities.set_light(copy, light);
		}

		// and its emitter, beside the light and on the same terms. What it has
		// already thrown is *not* copied: a cloud belongs to the emitter that
		// threw it, and a duplicate that arrived with somebody else's smoke
		// around it would be a copy of a moment rather than of a thing.
		if let Some(emitter) = world.entities.emitter(source).copied() {
			world.entities.set_emitter(copy, emitter);
		}

		// and its ground, beside both and on the same terms. What it *built*
		// is not copied either, and for a reason nearer the emitter's than it
		// looks: a mesh and a body are what a record turned into, and the copy
		// gets its own pair on the next step under its own slot's name.
		if let Some(terrain) = world.entities.terrain(source).copied() {
			world.entities.set_terrain(copy, terrain);
		}

		// and what its records hold, by name, the way a save carries it: the copy
		// is the same thing to every record the world declares, and to a record
		// nobody has declared yet it waits the way the original's value does
		let noted = world.entities.noted(source);
		record::report(&world.entities.note(copy, &noted), "a duplicate");

		let name = world.entities.name(source).to_owned();
		world.entities.set_name(copy, &name);
		copies.push((source, copy));
	}

	// off the copy of the parent, or off the same parent when it was not
	// copied
	for &(source, copy) in &copies {
		let parent = world.entities.parent(source);
		let hung = became(&copies, parent).unwrap_or(parent);

		world.entities.set_parent(copy, hung);
	}

	copies
}

/// Copies every body driving a copied entity, driving the copy, and every
/// body picked on its own, driving nothing.
///
/// @return each original with its copy
fn copy_bodies(
	world: &mut World,
	picks: &[Pick],
	copies: &[(EntityId, EntityId)],
) -> Vec<(BodyId, BodyId)> {
	let mut sources: Vec<BodyId> = world
		.bodies
		.iter()
		.filter(|(_, body)| became(copies, body.entity).is_some())
		.map(|(id, _)| id)
		.collect();

	for picked in picked_bodies(world, picks) {
		if !sources.contains(&picked) {
			sources.push(picked);
		}
	}

	sources.sort_by_key(|id| id.slot());

	let mut body_copies: Vec<(BodyId, BodyId)> = Vec::new();

	for source in sources {
		let Some(mut body) = world.bodies.get(source).copied() else {
			continue;
		};

		body.entity = became(copies, body.entity).unwrap_or(EntityId::NONE);
		let copy = world.bodies.spawn(body);
		if !copy.is_some() {
			continue;
		}

		let name = world.bodies.name(source).to_owned();
		world.bodies.set_name(copy, &name);
		body_copies.push((source, copy));
	}

	body_copies
}

/// Copies every joint holding a copied body first: holding the copy, and at
/// the far end the copy when there is one, the same body when there is not,
/// and the world when it was the world.
///
/// @return each original with its copy
fn copy_joints(world: &mut World, body_copies: &[(BodyId, BodyId)]) -> Vec<(JointId, JointId)> {
	let mut sources: Vec<JointId> = world
		.joints
		.iter()
		.filter(|(_, joint)| became(body_copies, joint.first).is_some())
		.map(|(id, _)| id)
		.collect();
	sources.sort_by_key(|id| id.slot());

	let mut joint_copies: Vec<(JointId, JointId)> = Vec::new();

	for source in sources {
		let Some(mut joint) = world.joints.get(source).copied() else {
			continue;
		};

		let Some(first) = became(body_copies, joint.first) else {
			continue;
		};

		joint.first = first;
		if joint.second.is_some() {
			joint.second = became(body_copies, joint.second).unwrap_or(joint.second);
		}

		let copy = world.joints.spawn(joint);
		if !copy.is_some() {
			continue;
		}

		let name = world.joints.name(source).to_owned();
		world.joints.set_name(copy, &name);
		joint_copies.push((source, copy));
	}

	joint_copies
}

/// How big a pool is when somebody asks for one, in world units.
///
/// Wide, shallow and not square: a pool that came out a cube would have to be
/// dragged into shape before it looked like anything, and the shape a person
/// is going to want is a body of water rather than a tank.
pub(crate) const POOL: Vec3 = Vec3::new(8.0, 3.0, 8.0);

/// Puts a body of water in the world, and gives it something to look at.
///
/// **One entity and one body, and the two agree by construction.** The body's
/// shape is the unit cube and its size is the transform's scale, which is
/// exactly what the cube mesh the entity draws is - so the box that is drawn
/// and the box that floats things are the same box, and the gizmo's size tool
/// moves both at once. A shape with its own extents beside a mesh with its own
/// scale would be two numbers a person has to keep equal by hand.
///
/// A cube rather than a quad at the surface, and it is worth saying why: a
/// fluid is a *volume*, its walls are where things stop being in it, and a
/// single plane at the top would draw a world where a pool and a puddle look
/// identical. @ref [`Material::WATER`](colby_core::abi::Material::WATER) for
/// the three rules that make it see-through.
///
/// @param world - the world to write
/// @param at - where the middle of it goes
/// @return what was made, for the selection; empty if the tables are full
/// The nearest point on a grid of this step.
///
/// **Half away rounds away from nought**, which is `f32::round`'s rule and is
/// the one a person expects: a block dragged to 0.25 on a step of 0.5 lands on
/// 0.5 rather than on nought, and one dragged to -0.25 lands on -0.5. A step of
/// nought or less is no grid and the point is its own.
///
/// @param at - the point
/// @param step - how far apart the lines are
#[must_use]
pub(crate) fn snapped(at: Vec3, step: f32) -> Vec3 {
	if !step.is_finite() || step <= 0.0 {
		return at;
	}

	(at / step).round() * step
}

/// A size on a grid of this step, never smaller than one step.
///
/// **Not [`snapped`], and the difference is what a grid is for.** A size that
/// rounded to nought would be a block with no thickness, which is a block
/// nobody can see, click or collide with - so the smallest a block gets is one
/// cell. The sign is kept, because a negative scale is a mirror and taking that
/// away is not this function's business.
///
/// @param size - how big it is along each axis
/// @param step - how far apart the lines are
#[must_use]
pub(crate) fn sized(size: Vec3, step: f32) -> Vec3 {
	if !step.is_finite() || step <= 0.0 {
		return size;
	}

	Vec3::new(one_cell(size.x, step), one_cell(size.y, step), one_cell(size.z, step))
}

/// One axis of a size, rounded to the grid and held at one cell.
fn one_cell(size: f32, step: f32) -> f32 {
	let rounded = (size.abs() / step).round().max(1.0) * step;

	if size < 0.0 { -rounded } else { rounded }
}

/// A block: a cube standing on the grid, solid, and nothing else.
///
/// **The whole of the block tool, and it is short on purpose.** A block is an
/// entity with the built-in cube on it and a static box body driving it, which
/// means the gizmo moves it, the hierarchy lists it, undo remembers it,
/// duplicate copies it, delete removes it and a saved scene carries it - all
/// of that for nothing, because a block *is* an entity and every one of those
/// already works on entities.
///
/// **The collider needs no size of its own**: the solver scales a shape by the
/// transform it stands in (`contact.rs:650`), so a unit box in a transform
/// scaled to the block is the block.
///
/// What the field says about keeping blocks: nobody ships them. Unreal's
/// brushes kept them in the level and `CSG_Add` and `CSG_Subtract` are marked
/// "(deprecated, do not use.)" in `Brush.h`; Godot's CSG keeps a tree of
/// shapes and its own documentation says prototyping only, bake to static
/// geometry; s&box keeps a half-edge mesh in the scene and no blocks at all.
/// So this is the editing form and a bake is the shipping one - a card of its
/// own, and until it exists the honest thing is that a room of these is a room
/// of entities.
///
/// **How big a new one is**: a unit cube whenever the grid divides a unit, and
/// one cell when the cells are bigger than that - which is [`sized`] applied to
/// [`Vec3::ONE`] and needs no rule of its own. So a step of a quarter, a half
/// or a whole all give a unit block, and a step of two gives a two-unit one.
///
/// @param world - the world to put it in
/// @param at - where the pointer is looking, before snapping
/// @param step - the grid, or nothing for none
/// @return what was made, for the selection; empty when there was no room
pub(crate) fn block(world: &mut World, at: Vec3, step: Option<f32>) -> Vec<Pick> {
	let grid = step.unwrap_or(0.0);
	let size = sized(Vec3::ONE, grid);
	// half a block up, so that a block put down on the floor stands on it
	// rather than half through it - which is where the point under the
	// pointer is, and is not where a person means
	let standing = Transform {
		position: snapped(at, grid) + Vec3::Y * size.y * 0.5,
		rotation: colby_core::glam::Quat::IDENTITY,
		scale: size,
	};
	let entity = world.entities.spawn_at(standing);

	if !entity.is_some() {
		return Vec::new();
	}

	world
		.entities
		.set_renderable(entity, Renderable::new(MeshId::CUBE, Vec3::splat(0.8)));
	world.entities.set_name(entity, "block");

	let body = world
		.bodies
		.spawn(Body::new(BodyKind::Static, Shape::UNIT, standing).driving(entity));

	if !body.is_some() {
		world.entities.despawn(entity);

		return Vec::new();
	}

	world.bodies.set_name(body, "block");

	vec![Pick::Entity(entity)]
}

pub(crate) fn water(world: &mut World, at: Vec3) -> Vec<Pick> {
	let mut standing = Transform::at(at);
	standing.scale = POOL;

	let entity = world.entities.spawn_at(standing);
	if !entity.is_some() {
		return Vec::new();
	}

	let mut look = Renderable::new(MeshId::CUBE, Vec3::ONE);
	look.material = world.materials.find(material::WATER_NAME);
	world.entities.set_renderable(entity, look);
	world.entities.set_name(entity, "water");

	// static, because a body of water that fell would be a body of water with
	// nowhere to be. A person who wants one that moves says so in the
	// inspector, and everything about it already works if they do.
	let body = world
		.bodies
		.spawn(Body::new(BodyKind::Static, Shape::UNIT, standing).driving(entity));

	if !body.is_some() {
		world.entities.despawn(entity);

		return Vec::new();
	}

	if let Some(held) = world.bodies.get_mut(body) {
		held.water = Water::pool();
	}

	world.bodies.set_name(body, "water");

	vec![Pick::Entity(entity)]
}

/// A decal, standing where the middle of the view meets the ground and turned
/// to throw its picture straight down.
///
/// A unit across and a quarter deep, which puts the ground under the pointer in
/// the middle of its box; no mesh, so nothing is drawn for it but what it
/// paints; and the default material under a white tint, so what it paints is
/// white until it is given a material of its own. @ref
/// `colby_core::abi::decal`.
///
/// @param world - the world to put it in
/// @param at - where the pointer is looking
/// @return what was made, for the selection; empty when there was no room
pub(crate) fn decal(world: &mut World, at: Vec3) -> Vec<Pick> {
	let standing = Transform {
		position: at,
		// a quarter turn about x takes the box's -z, the way it throws, to -y
		rotation: colby_core::glam::Quat::from_rotation_x(-core::f32::consts::FRAC_PI_2),
		scale: Vec3::new(1.0, 1.0, 0.25),
	};
	let entity = world.entities.spawn_at(standing);

	if !entity.is_some() {
		return Vec::new();
	}

	world.entities.set_decal(entity, Decal::BOX);
	world.entities.set_name(entity, "decal");

	vec![Pick::Entity(entity)]
}

/// What a row of the asset browser becomes when it is dropped into the
/// world: a scene laid down there, a mesh as an entity standing there, a
/// model as an entity with a child per piece. Anything else - a texture, a
/// sound, a font, a document, a program, a translation - is not a thing that
/// stands anywhere, and becomes nothing.
///
/// @param world - the world to write
/// @param name - the asset name, `meshes/crystal`
/// @param kind - what it is
/// @param at - where it lands
/// @return what was made, for the selection; empty when nothing was
pub(crate) fn drop(world: &mut World, name: &str, kind: Kind, at: Vec3) -> Vec<Pick> {
	match kind {
		| Kind::Scene => drop_scene(world, name, at),
		| Kind::Mesh => drop_mesh(world, name, at),
		| Kind::Model => drop_model(world, name, at),
		| Kind::Texture
		| Kind::Font
		| Kind::Document
		| Kind::Material
		| Kind::Sound
		| Kind::Skeleton
		| Kind::Clip
		| Kind::Script
		| Kind::Translation => Vec::new(),
	}
}

/// The last part of an asset name, which is what a thing made from it is
/// called: `crystal` for `meshes/crystal`.
fn stem(name: &str) -> &str { name.rsplit('/').next().unwrap_or(name) }

/// A scene, created beside what is there with its roots moved to the drop.
fn drop_scene(world: &mut World, name: &str, at: Vec3) -> Vec<Pick> {
	let id = world.scenes.find(name);
	if !id.is_some() {
		return Vec::new();
	}

	let data = world.scenes.data(id).clone();
	let remap = scene::instantiate(world, &data, at);
	let mut landed: Vec<Pick> = remap.entities().map(Pick::Entity).collect();

	// a scene of bodies alone, a set of colliders, is selected by them
	if landed.is_empty() {
		landed.extend(remap.bodies().map(Pick::Body));
	}

	landed
}

/// A mesh, as an entity drawing it in the default material, standing where
/// it was dropped.
fn drop_mesh(world: &mut World, name: &str, at: Vec3) -> Vec<Pick> {
	let mesh = world.meshes.find(name);
	if !mesh.is_some() {
		return Vec::new();
	}

	let entity = world.entities.spawn_at(Transform::at(at));
	if !entity.is_some() {
		return Vec::new();
	}

	world
		.entities
		.set_renderable(entity, Renderable::new(mesh, Vec3::ONE));
	world.entities.set_name(entity, stem(name));

	vec![Pick::Entity(entity)]
}

/// A model, as an entity standing where it was dropped with a child per
/// piece hanging off it, each drawing its mesh in its material where the
/// model puts it.
///
/// What the table of placements is for: a game writes this loop itself,
/// and the editor writes it once here. A pose is not given, so a skinned
/// piece stands in its bind pose.
fn drop_model(world: &mut World, name: &str, at: Vec3) -> Vec<Pick> {
	let id = world.models.find(name);
	let Some(model) = world.models.get(id) else {
		return Vec::new();
	};
	let placements = model.value().placements.clone();

	let parent = world.entities.spawn_at(Transform::at(at));
	if !parent.is_some() {
		return Vec::new();
	}

	world.entities.set_name(parent, stem(name));

	for placement in placements {
		let child = world.entities.spawn_at(placement.transform);
		if !child.is_some() {
			continue;
		}

		world
			.entities
			.set_renderable(child, Renderable::of(placement.mesh, placement.material, Vec3::ONE));
		world.entities.set_name(child, &placement.name);
		world.entities.set_parent(child, parent);
	}

	vec![Pick::Entity(parent)]
}

/// Puts the primary where a drag took it, and everything else selected
/// along with it.
///
/// What the primary did is worked out as a change in the world - a shift, a
/// turn about the primary, a stretch away from it - and the same change is
/// applied to where each of the others was when the drag began, so that a
/// drag of any length lands them where one frame of it would. Turning and
/// stretching happen about the primary rather than about each thing's own
/// middle, which is what Godot does with several nodes under one gizmo.
///
/// @param world - the world to write
/// @param primary - what the gizmo is attached to
/// @param from - where the primary was when the drag began
/// @param put - where the drag has taken it
/// @param others - everything else selected, each with where it was when
/// the drag began, in the world
pub(crate) fn drag_all(
	world: &mut World,
	primary: Pick,
	from: Transform,
	put: Transform,
	others: &[(Pick, Transform)],
) {
	place(world, primary, put);

	let shift = put.position - from.position;
	let turn = (put.rotation * from.rotation.inverse()).normalize();
	let stretch = Vec3::new(
		ratio(put.scale.x, from.scale.x),
		ratio(put.scale.y, from.scale.y),
		ratio(put.scale.z, from.scale.z),
	);

	for &(other, was) in others {
		// the other's offset from the primary, in the primary's own axes,
		// stretched as the primary was, turned as it was, and shifted
		let offset = from.rotation.inverse() * (was.position - from.position);
		let moved = from.rotation * (offset * stretch);

		let landed = Transform {
			position: from.position + turn * moved + shift,
			rotation: (turn * was.rotation).normalize(),
			scale: was.scale * stretch,
		};

		place(world, other, landed);
	}
}

/// How much longer one length is than another; one when either is nothing.
fn ratio(now: f32, before: f32) -> f32 {
	if before.abs() < f32::EPSILON || !now.is_finite() {
		1.0
	} else {
		now / before
	}
}

#[cfg(test)]
mod tests {
	use colby_core::abi::{Joint, ShapeKind};

	use super::*;

	#[test]
	fn a_pool_is_one_body_and_one_entity_whose_two_boxes_are_the_same_box() {
		let mut world = World::new();
		let made = water(&mut world, Vec3::new(3.0, -1.0, 0.0));

		assert_eq!(made.len(), 1, "one thing is selected, and it is the entity");

		let Some(Pick::Entity(entity)) = made.first().copied() else {
			panic!("a pool is picked by its entity");
		};
		let (_, body) = world
			.bodies
			.iter()
			.find(|(_, body)| body.water.is_wet())
			.expect("and there is a body under it");

		assert_eq!(body.entity, entity, "the body drives the entity");
		assert_eq!(body.shape.kind, ShapeKind::Box, "and it is a box");
		assert_eq!(
			body.shape.extents.abs() * body.transform.scale.abs(),
			POOL * 0.5,
			"whose half-extents are half the drawn cube, which is what makes the two agree"
		);
		assert_eq!(body.transform.position, Vec3::new(3.0, -1.0, 0.0), "where it was asked for");
		assert!(!body.solid(), "and nothing is pushed out of it");

		let look = world
			.entities
			.renderable(entity)
			.expect("the entity draws something");

		assert_eq!(look.mesh, MeshId::CUBE, "a volume rather than a plane at the top of one");
		assert_eq!(
			look.material,
			world.materials.find(material::WATER_NAME),
			"in the one built-in material that is not solid"
		);
	}

	#[test]
	fn a_decal_is_a_box_turned_to_throw_straight_down_and_nothing_else() {
		let mut world = World::new();
		let Some(Pick::Entity(entity)) = decal(&mut world, Vec3::new(3.0, -1.0, 0.0))
			.first()
			.copied()
		else {
			panic!("a decal is picked by its entity");
		};

		assert_eq!(world.entities.decal(entity).copied(), Some(Decal::BOX), "it paints");

		let at = world
			.entities
			.transform(entity)
			.copied()
			.expect("it stands somewhere");

		assert!(
			(at.rotation * Vec3::NEG_Z - Vec3::NEG_Y).length() < 1.0e-5,
			"and throws its picture straight down"
		);
		assert_eq!(at.position, Vec3::new(3.0, -1.0, 0.0), "from where it was asked for");

		let look = world
			.entities
			.renderable(entity)
			.copied()
			.expect("it has a look");

		assert_eq!(look.mesh, MeshId::NONE, "with nothing drawn for it but what it paints");
		assert_eq!(world.bodies.iter().count(), 0, "and no body");
	}

	#[test]
	fn a_pool_can_be_resized_by_the_one_number_that_moves_both_halves() {
		// the whole reason the shape is the unit cube: a size gizmo writes the
		// transform's scale, and both the box that is drawn and the box that
		// floats things are that scale.
		let mut world = World::new();
		let Some(Pick::Entity(entity)) = water(&mut world, Vec3::ZERO).first().copied() else {
			panic!("a pool is picked by its entity");
		};
		let (id, _) = world
			.bodies
			.iter()
			.find(|(_, body)| body.water.is_wet())
			.expect("a body is under it");

		if let Some(body) = world.bodies.get_mut(id) {
			body.transform.scale = Vec3::new(20.0, 1.0, 20.0);
		}

		let body = world.bodies.get(id).expect("it is alive");

		assert_eq!(
			body.surface(),
			Some(0.5),
			"the surface follows the scale, so a shallower pool floats things lower"
		);
		assert!(
			world.entities.placed(entity).is_some(),
			"and the entity the renderer walks is still there to be scaled with it"
		);
	}

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
	fn a_child_is_picked_up_in_the_world_and_put_down_as_a_local() {
		let (mut world, ..) = peopled();
		let car = world
			.entities
			.spawn_at(Transform::at(Vec3::new(5.0, 0.0, 0.0)));
		let wheel = world.entities.spawn_at(Transform::at(Vec3::X));
		assert!(world.entities.set_parent(wheel, car));

		assert_eq!(
			transform(&world, Pick::Entity(wheel)).map(|it| it.position),
			Some(Vec3::new(6.0, 0.0, 0.0)),
			"the gizmo sees where it is in the world"
		);
		assert_eq!(
			local(&world, Pick::Entity(wheel)).map(|it| it.position),
			Some(Vec3::X),
			"and the inspector sees its place inside the car"
		);

		assert!(place(&mut world, Pick::Entity(wheel), Transform::at(Vec3::new(9.0, 0.0, 0.0))));
		assert_eq!(
			world
				.entities
				.transform(wheel)
				.map(|it| it.position),
			Some(Vec3::new(4.0, 0.0, 0.0)),
			"a world drop lands as a local"
		);

		assert!(place_local(&mut world, Pick::Entity(wheel), Transform::at(Vec3::NEG_X)));
		assert_eq!(
			transform(&world, Pick::Entity(wheel)).map(|it| it.position),
			Some(Vec3::new(4.0, 0.0, 0.0)),
			"and a local edit is a world place through the car"
		);
	}

	#[test]
	fn a_body_under_a_child_is_teleported_and_the_child_lands_as_a_local() {
		let (mut world, ..) = peopled();
		let car = world
			.entities
			.spawn_at(Transform::at(Vec3::new(5.0, 0.0, 0.0)));
		let wheel = world.entities.spawn_at(Transform::at(Vec3::X));
		assert!(world.entities.set_parent(wheel, car));
		let body = world.attach_body(wheel, BodyKind::Dynamic, Shape::UNIT);

		assert!(place(&mut world, Pick::Body(body), Transform::at(Vec3::new(9.0, 0.0, 0.0))));

		assert_eq!(
			world
				.bodies
				.get(body)
				.map(|it| it.transform.position),
			Some(Vec3::new(9.0, 0.0, 0.0)),
			"the body is where it was put, in the world"
		);
		assert_eq!(
			world
				.entities
				.transform(wheel)
				.map(|it| it.position),
			Some(Vec3::new(4.0, 0.0, 0.0)),
			"and the wheel took it as its place inside the car"
		);
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

	#[test]
	fn several_things_are_selected_and_the_last_picked_is_the_primary() {
		let (world, entity, body, joint) = peopled();
		let mut selection = Selection::default();

		selection.set(&world, Pick::Entity(entity));
		selection.toggle(&world, Pick::Body(body));
		selection.toggle(&world, Pick::Joint(joint));

		assert_eq!(selection.len(), 3);
		assert_eq!(selection.at(), Pick::Joint(joint), "the last picked");
		assert_eq!(selection.others(), vec![Pick::Entity(entity), Pick::Body(body)]);
		assert!(selection.is(Pick::Body(body)), "and the others are selected too");

		selection.toggle(&world, Pick::Joint(joint));
		assert_eq!(
			selection.at(),
			Pick::Body(body),
			"taking the primary out leaves the one before"
		);

		selection.toggle(&world, Pick::Nothing);
		assert_eq!(selection.len(), 2, "nothing is neither added nor taken out");

		selection.set(&world, Pick::Nothing);
		assert_eq!(selection.len(), 0, "a click on empty space clears the lot");
		assert!(!selection.is(Pick::Nothing), "and nothing is never selected");
	}

	#[test]
	fn a_selection_of_several_keeps_the_ones_that_still_resolve() {
		let (mut world, entity, body, _) = peopled();
		let mut selection = Selection::default();
		selection.set(&world, Pick::Entity(entity));
		selection.toggle(&world, Pick::Body(body));

		world.bodies.despawn(body);
		selection.refresh(&world);

		assert_eq!(
			selection.picks(),
			vec![Pick::Entity(entity)],
			"the body is gone, the entity stays"
		);
	}

	#[test]
	fn descendants_are_every_entity_down_the_chain_and_not_the_ancestor_itself() {
		let mut world = World::new();
		let car = world.entities.spawn_at(Transform::IDENTITY);
		let wheel = world.entities.spawn_at(Transform::at(Vec3::X));
		let hub = world.entities.spawn_at(Transform::at(Vec3::Y));
		let other = world.entities.spawn_at(Transform::at(Vec3::Z));
		assert!(world.entities.set_parent(wheel, car));
		assert!(world.entities.set_parent(hub, wheel));

		assert_eq!(descendants(&world, car), vec![wheel, hub]);
		assert_eq!(descendants(&world, wheel), vec![hub]);
		assert!(descendants(&world, other).is_empty());
		assert!(descendants(&world, EntityId::NONE).is_empty(), "nothing hangs off nothing");
	}

	#[test]
	fn deleting_an_entity_takes_its_branch_its_bodies_and_their_joints() {
		let (mut world, entity, body, joint) = peopled();
		let wheel = world.entities.spawn_at(Transform::at(Vec3::X));
		assert!(world.entities.set_parent(wheel, entity));
		let wheel_body = world.attach_body(wheel, BodyKind::Dynamic, Shape::UNIT);
		let bystander = world.entities.spawn_at(Transform::at(Vec3::Z));
		let floor = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::ball(1.0),
			Transform::IDENTITY,
		));

		let went = delete(&mut world, &[Pick::Entity(entity)]);

		assert_eq!(went, Deleted { entities: 2, bodies: 2, joints: 1 });
		assert!(!world.entities.alive(entity) && !world.entities.alive(wheel));
		assert!(!world.bodies.alive(body) && !world.bodies.alive(wheel_body));
		assert!(!world.joints.alive(joint), "the rope held a body that went");
		assert!(world.entities.alive(bystander) && world.bodies.alive(floor), "the rest stands");
	}

	#[test]
	fn deleting_a_body_takes_its_joints_and_leaves_its_entity() {
		let (mut world, entity, body, joint) = peopled();

		let went = delete(&mut world, &[Pick::Body(body)]);

		assert_eq!(went, Deleted { entities: 0, bodies: 1, joints: 1 });
		assert!(
			world.entities.alive(entity),
			"the thing drawn stays; the collider under it went"
		);
		assert!(!world.joints.alive(joint));

		assert_eq!(
			delete(&mut world, &[Pick::Body(body), Pick::Nothing]),
			Deleted::default(),
			"a stale handle and nothing at all delete nothing"
		);
	}

	#[test]
	fn a_duplicate_is_a_second_branch_with_its_bodies_and_joints_beside_the_first() {
		let (mut world, car, car_body, rope) = peopled();
		let wheel = world.entities.spawn_at(Transform::at(Vec3::X));
		world.entities.set_name(wheel, "wheel");
		assert!(world.entities.set_parent(wheel, car));
		let wheel_body = world.attach_body(wheel, BodyKind::Dynamic, Shape::UNIT);
		let axle = world
			.joints
			.spawn(Joint::weld(car_body, wheel_body, (Vec3::X, Vec3::ZERO)));
		let before = (world.entities.len(), world.bodies.len(), world.joints.len());

		let copies = duplicate(&mut world, &[Pick::Entity(car)]);

		assert_eq!(copies.len(), 1, "one thing was picked, so one copy is selected");
		let Some(Pick::Entity(car_copy)) = copies.first().copied() else {
			panic!("the copy of the car");
		};
		assert_ne!(car_copy, car);
		assert_eq!(world.entities.name(car_copy), "crate", "the name is kept");
		assert_eq!(
			(world.entities.len(), world.bodies.len(), world.joints.len()),
			(before.0 + 2, before.1 + 2, before.2 + 2),
			"the car and the wheel, their two bodies, the rope and the axle"
		);
		assert!(world.entities.alive(car) && world.bodies.alive(car_body), "the originals stand");

		let wheel_copy = descendants(&world, car_copy);
		assert_eq!(wheel_copy.len(), 1, "the wheel came along");
		assert_eq!(world.entities.name(wheel_copy[0]), "wheel");
		assert_eq!(
			world
				.entities
				.placed(wheel_copy[0])
				.map(|it| it.position),
			world.entities.placed(wheel).map(|it| it.position),
			"in the same place as the original"
		);

		let driving: Vec<BodyId> = world
			.bodies
			.iter()
			.filter(|(_, body)| body.entity == car_copy || body.entity == wheel_copy[0])
			.map(|(id, _)| id)
			.collect();
		assert_eq!(driving.len(), 2, "each copy has its body");

		let holding: Vec<&Joint> = world
			.joints
			.iter()
			.filter(|(id, _)| *id != rope && *id != axle)
			.map(|(_, joint)| joint)
			.collect();
		assert_eq!(holding.len(), 2, "the rope to the world and the axle between the copies");
		assert!(
			holding
				.iter()
				.all(|joint| driving.contains(&joint.first)),
			"each copied joint holds a copied body first"
		);
		assert!(
			holding
				.iter()
				.any(|joint| !joint.second.is_some())
				&& holding
					.iter()
					.any(|joint| driving.contains(&joint.second)),
			"one still to the world, one to the other copy"
		);
	}

	#[test]
	fn a_duplicate_carries_what_its_records_hold_and_what_waits_for_one() {
		// the copy is a fourth path a thing takes out of the world and back into
		// it, beside a save, a paste and a piece that crossed, and the only one
		// that does not go through a description: found by a live run of the
		// editor whose copy came back at every default
		let (mut world, car, ..) = peopled();

		if let Some(drawing) = world
			.entities
			.record_mut(&colby_core::abi::DRAWING, car)
		{
			drawing.covers = 1;
		}

		let waits = colby_core::abi::Noted {
			record: "door".to_owned(),
			field: "open".to_owned(),
			value: colby_core::abi::Spelled::Truth(true),
		};

		assert!(
			world
				.entities
				.note(car, std::slice::from_ref(&waits))
				.is_empty(),
			"a door nobody declared"
		);

		let copies = duplicate(&mut world, &[Pick::Entity(car)]);
		let Some(Pick::Entity(copy)) = copies.first().copied() else {
			panic!("the copy of the car");
		};

		assert!(
			world
				.entities
				.record(&colby_core::abi::DRAWING, copy)
				.is_some_and(|drawing| drawing.covers()),
			"the copy covers what the original covers"
		);
		assert_eq!(world.entities.waiting(copy), &[waits], "and waits for what it waited for");
		assert_eq!(world.entities.noted(copy), world.entities.noted(car), "every value, by name");
	}

	#[test]
	fn a_body_picked_on_its_own_is_copied_driving_nothing() {
		let (mut world, entity, body, _) = peopled();

		let copies = duplicate(&mut world, &[Pick::Body(body)]);

		let Some(Pick::Body(copy)) = copies.first().copied() else {
			panic!("the copy of the body");
		};
		assert_eq!(
			world.bodies.get(copy).map(|it| it.entity),
			Some(EntityId::NONE),
			"two bodies driving one entity would be a fight"
		);
		assert_eq!(
			world.bodies.get(body).map(|it| it.entity),
			Some(entity),
			"the original keeps its entity"
		);
	}

	/// Two things a unit apart along x, the first the primary.
	fn pair() -> (World, Pick, Pick, Transform, Transform) {
		let mut world = World::new();
		let first = world.entities.spawn_at(Transform::IDENTITY);
		let second = world.entities.spawn_at(Transform::at(Vec3::X));

		(
			world,
			Pick::Entity(first),
			Pick::Entity(second),
			Transform::IDENTITY,
			Transform::at(Vec3::X),
		)
	}

	#[test]
	fn a_dropped_mesh_is_an_entity_drawing_it_where_it_was_dropped() {
		let mut world = World::new();
		let mesh = world
			.meshes
			.insert("meshes/crystal", colby_core::abi::mesh::cube());

		let landed = drop(&mut world, "meshes/crystal", Kind::Mesh, Vec3::new(2.0, 0.0, -1.0));

		let [Pick::Entity(entity)] = landed[..] else {
			panic!("one entity: {landed:?}");
		};
		assert_eq!(world.entities.name(entity), "crystal", "called after the mesh");
		assert_eq!(
			world
				.entities
				.transform(entity)
				.map(|it| it.position),
			Some(Vec3::new(2.0, 0.0, -1.0))
		);
		assert_eq!(
			world
				.entities
				.renderable(entity)
				.map(|it| it.mesh),
			Some(mesh)
		);
		assert!(
			drop(&mut world, "meshes/nothing", Kind::Mesh, Vec3::ZERO).is_empty(),
			"a mesh nobody registered makes nothing"
		);
	}

	#[test]
	fn a_dropped_scene_is_laid_down_with_its_roots_moved_to_the_drop() {
		let mut world = World::new();
		let mut described = World::new();
		let crate_ = described
			.entities
			.spawn_at(Transform::at(Vec3::Y));
		described.entities.set_name(crate_, "crate");
		let lid = described
			.entities
			.spawn_at(Transform::at(Vec3::Y));
		described.entities.set_name(lid, "lid");
		assert!(described.entities.set_parent(lid, crate_));
		world
			.scenes
			.insert("scenes/box", scene::capture(&described));

		let landed = drop(&mut world, "scenes/box", Kind::Scene, Vec3::new(5.0, 0.0, 0.0));

		assert_eq!(landed.len(), 2, "both things of the scene");
		let root = landed
			.iter()
			.find_map(|pick| match *pick {
				| Pick::Entity(id) if world.entities.name(id) == "crate" => Some(id),
				| _ => None,
			})
			.expect("the crate landed");
		assert_eq!(
			world.entities.placed(root).map(|it| it.position),
			Some(Vec3::new(5.0, 1.0, 0.0)),
			"the root is moved to the drop"
		);
		let child = landed
			.iter()
			.find_map(|pick| match *pick {
				| Pick::Entity(id) if world.entities.name(id) == "lid" => Some(id),
				| _ => None,
			})
			.expect("the lid landed");
		assert_eq!(world.entities.parent(child), root, "still hanging off the crate");
		assert_eq!(
			world.entities.placed(child).map(|it| it.position),
			Some(Vec3::new(5.0, 2.0, 0.0)),
			"and moved with it, not twice"
		);
	}

	#[test]
	fn a_dropped_model_is_an_entity_with_a_child_per_piece() {
		use colby_core::abi::{
			MaterialId, MeshId,
			model::{ModelData, Placement},
		};

		let mut world = World::new();
		world.models.insert("models/lamp", ModelData {
			placements: vec![
				Placement {
					name: "base".to_owned(),
					mesh: MeshId::CUBE,
					material: MaterialId::DEFAULT,
					transform: Transform::at(Vec3::ZERO),
					..Placement::default()
				},
				Placement {
					name: "shade".to_owned(),
					mesh: MeshId::CUBE,
					material: MaterialId::DEFAULT,
					transform: Transform::at(Vec3::Y * 2.0),
					..Placement::default()
				},
			],
		});

		let landed = drop(&mut world, "models/lamp", Kind::Model, Vec3::X * 3.0);

		let [Pick::Entity(parent)] = landed[..] else {
			panic!("the model's own entity: {landed:?}");
		};
		assert_eq!(world.entities.name(parent), "lamp");
		let pieces = descendants(&world, parent);
		assert_eq!(pieces.len(), 2, "a child per piece");
		let shade = pieces
			.iter()
			.copied()
			.find(|id| world.entities.name(*id) == "shade")
			.expect("the shade");
		assert_eq!(
			world.entities.placed(shade).map(|it| it.position),
			Some(Vec3::new(3.0, 2.0, 0.0)),
			"where the model puts it, from where the model was dropped"
		);
		assert_eq!(world.entities.renderable(shade).map(|it| it.mesh), Some(MeshId::CUBE));
	}

	#[test]
	fn a_point_lands_on_the_nearest_line_and_half_rounds_away_from_nought() {
		assert_eq!(snapped(Vec3::new(0.24, 0.0, -0.24), 0.5), Vec3::ZERO, "under half");
		assert_eq!(
			snapped(Vec3::new(0.25, 0.0, -0.25), 0.5),
			Vec3::new(0.5, 0.0, -0.5),
			"half away from nought, which is what a person dragging expects"
		);
		assert_eq!(snapped(Vec3::new(1.4, 2.6, -3.9), 1.0), Vec3::new(1.0, 3.0, -4.0));
	}

	#[test]
	fn a_step_of_nothing_is_no_grid_at_all() {
		let loose = Vec3::new(0.137, -2.9, 41.0);

		for step in [0.0, -1.0, f32::NAN, f32::INFINITY] {
			assert_eq!(snapped(loose, step), loose, "a step of {step} snaps nothing");
			assert_eq!(sized(loose, step), loose, "and sizes nothing");
		}
	}

	#[test]
	fn a_size_never_rounds_down_to_nothing() {
		// the difference between a size and a point, and the reason `sized`
		// exists at all: a block with no thickness cannot be seen, clicked or
		// collided with, so the smallest one is a cell
		assert_eq!(sized(Vec3::splat(0.01), 0.5), Vec3::splat(0.5), "one cell, not nought");
		assert_eq!(sized(Vec3::splat(0.6), 0.5), Vec3::splat(0.5));
		assert_eq!(sized(Vec3::new(1.2, 2.4, 0.1), 0.5), Vec3::new(1.0, 2.5, 0.5));
	}

	#[test]
	fn a_size_keeps_the_sign_it_had() {
		// a negative scale is a mirror, and taking one away is not this
		// function's business
		assert_eq!(sized(Vec3::new(-1.2, 1.2, -0.01), 0.5), Vec3::new(-1.0, 1.0, -0.5));
	}

	#[test]
	fn a_block_stands_on_the_floor_rather_than_half_through_it() {
		let mut world = World::new();
		let made = block(&mut world, Vec3::new(2.3, 0.0, -1.1), Some(0.5));

		let [Pick::Entity(entity)] = made[..] else {
			panic!("one entity: {made:?}");
		};
		let standing = world
			.entities
			.placed(entity)
			.expect("it stands somewhere");

		assert_eq!(
			standing.position,
			Vec3::new(2.5, 0.5, -1.0),
			"snapped along the floor and lifted by half its height"
		);
		assert_eq!(
			standing.scale,
			Vec3::ONE,
			"a unit cube, because a step of half divides a unit; a step bigger than one gives 			 one cell instead"
		);
		assert_eq!(world.entities.name(entity), "block");
	}

	#[test]
	fn a_block_is_solid_and_its_collider_is_the_block() {
		// the whole reason this is cheap: the solver scales a shape by the
		// transform it stands in, so a unit box in the block's transform is
		// the block, and nothing here has to size a collider
		let mut world = World::new();
		let made = block(&mut world, Vec3::ZERO, Some(2.0));

		assert_eq!(made.len(), 1, "one thing selected, and it is the entity");

		let (_, held) = world
			.bodies
			.iter()
			.find(|(id, _)| world.bodies.name(*id) == "block")
			.expect("a block is solid");

		assert_eq!(held.kind, BodyKind::Static, "and static, because a wall does not fall");
		assert_eq!(held.shape.kind, ShapeKind::Box);
		assert_eq!(held.shape.extents, Vec3::splat(0.5), "a unit box");
		assert_eq!(held.transform.scale, Vec3::splat(2.0), "in a transform that is the block");
	}

	#[test]
	fn a_block_put_down_with_no_grid_lands_where_it_was_asked_for() {
		let mut world = World::new();
		let made = block(&mut world, Vec3::new(2.3, 0.0, -1.1), None);

		let [Pick::Entity(entity)] = made[..] else {
			panic!("one entity: {made:?}");
		};

		assert_eq!(
			world
				.entities
				.placed(entity)
				.expect("it stands")
				.position,
			Vec3::new(2.3, 0.5, -1.1),
			"unsnapped along the floor, and still lifted by half a unit block"
		);
	}

	#[test]
	fn a_dropped_texture_is_nothing() {
		let mut world = World::new();

		assert!(drop(&mut world, "textures/wall", Kind::Texture, Vec3::ZERO).is_empty());
		assert_eq!(world.entities.len(), 0);
	}

	#[test]
	fn a_shift_of_the_primary_shifts_the_others_by_the_same_amount() {
		let (mut world, first, second, from, other) = pair();

		drag_all(&mut world, first, from, Transform::at(Vec3::Y * 3.0), &[(second, other)]);

		assert_eq!(
			transform(&world, second).map(|it| it.position),
			Some(Vec3::new(1.0, 3.0, 0.0))
		);
	}

	#[test]
	fn a_turn_of_the_primary_turns_the_others_around_it() {
		let (mut world, first, second, from, other) = pair();
		let quarter = colby_core::glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);

		drag_all(&mut world, first, from, Transform { rotation: quarter, ..from }, &[(
			second, other,
		)]);

		let landed = transform(&world, second).expect("still there");
		assert!(
			landed
				.position
				.abs_diff_eq(Vec3::new(0.0, 0.0, -1.0), 1.0e-5),
			"a unit along x, a quarter turn about y, lands a unit along -z: {landed:?}"
		);
		assert!(landed.rotation.abs_diff_eq(quarter, 1.0e-5), "and it turned with the primary");
	}

	#[test]
	fn a_stretch_of_the_primary_stretches_the_others_away_from_it() {
		let (mut world, first, second, from, other) = pair();

		drag_all(
			&mut world,
			first,
			from,
			Transform { scale: Vec3::new(2.0, 1.0, 1.0), ..from },
			&[(second, other)],
		);

		let landed = transform(&world, second).expect("still there");
		assert_eq!(landed.position, Vec3::new(2.0, 0.0, 0.0), "twice as far along x");
		assert_eq!(landed.scale, Vec3::new(2.0, 1.0, 1.0), "and twice as wide");
	}
}
