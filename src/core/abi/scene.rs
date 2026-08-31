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
	Result,
	abi::{
		Body, BodyId, BodyKind, Camera, EntityId, Entry, Joint, JointId, JointKind, Layers,
		MaterialId, Pose, PoseId, Registry, Renderable, Shape, ShapeKind, Transform, World,
		state::STATE_BYTES,
	},
	err,
	glam::{Quat, Vec3},
	warn,
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
	/// Written by whoever authored the scene and by the capture alike: an
	/// entity carries a name in the world, so a world written down keeps every
	/// name in it and a world read back puts them all on again. A name is also
	/// what an instantiate hands back a handle under, and it is how a scene
	/// file refers to a thing a person typed. @ref
	/// [`names`](crate::abi::names) for why it is not an identifier.
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

	/// Which entry of [`SceneData::posed`] moves it, or [`NO_INDEX`].
	pub pose: u32,
}

impl Thing {
	/// The slot it occupied and the generation it held there.
	///
	/// What a restore keys the table's rebuilt generation array on.
	#[must_use]
	pub const fn key(&self) -> (u32, u32) { (self.slot, self.generation) }
}

/// A posed skeleton, with its skeleton named rather than handled.
///
/// A pose is not an asset and not an entity: it is one character's own
/// attitude, shared by every entity the character is drawn as. So it is a list
/// of its own and a [`Thing`] points into it by index, the way a [`Solid`]
/// points at its entity - two entities naming the same index are two entities
/// moved by one set of bones, which is what a model of two materials is.
///
/// Only the present is written. A pose's past is what the picture interpolates
/// through, and a world put back is a world nobody has stepped since: its past
/// is its present, and a restore says so rather than storing a second copy of
/// every bone.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Posed {
	/// What this is called, or empty.
	///
	/// A pose has no name in the world - nothing points at one except an
	/// entity, and an entity points by handle. It has one *here* because a
	/// source refers to it by name, the way a body refers to its entity, and
	/// because the writer then has something to invent a name into.
	pub name: String,

	/// The slot it occupied, for a restore.
	pub slot: u32,

	/// Which occupant of that slot it was.
	pub generation: u32,

	/// The asset name of the skeleton it poses, or empty.
	pub skeleton: String,

	/// Where each bone is, relative to its parent.
	pub locals: Vec<Transform>,
}

impl Posed {
	/// The slot it occupied and the generation it held there.
	#[must_use]
	pub const fn key(&self) -> (u32, u32) { (self.slot, self.generation) }
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

	/// Whether gravity reaches it.
	pub weightless: bool,

	/// Whether the solver had stopped integrating it.
	pub sleeping: bool,

	/// Which layers it is on and which it interacts with.
	pub layers: Layers,

	/// Which entry of [`SceneData::things`] it drives, or [`NO_INDEX`].
	pub thing: u32,
}

impl Solid {
	/// The slot it occupied and the generation it held there.
	///
	/// What a restore keys the table's rebuilt generation array on.
	#[must_use]
	pub const fn key(&self) -> (u32, u32) { (self.slot, self.generation) }
}

/// One joint, naming the two bodies it holds by their place in the file.
#[derive(Clone, Debug, PartialEq)]
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

	/// How stiff the spring holding it together is, in hertz. Zero is rigid.
	pub stiffness: f32,

	/// How quickly that spring stops ringing, as a ratio.
	pub damping: f32,

	/// The most it may pull with over one step, or zero for no ceiling.
	pub max_impulse: f32,

	/// The most it may turn with, or zero for no ceiling.
	pub max_torque: f32,

	/// Whether the two bodies it holds still collide with each other. @ref
	/// [`Joint::collide`].
	pub collide: bool,
}

impl Default for Link {
	/// The joint [`Joint::default`] describes, written down.
	///
	/// Hand-written rather than derived, and the reason is one field: a
	/// derived `Default` gives a damping of zero, which is a spring that rings
	/// forever, and `Link` is used as a template with `..Default::default()`.
	/// Every other field's zero is the value a joint actually has.
	fn default() -> Self {
		Self {
			name: String::new(),
			slot: 0,
			generation: 0,
			kind: JointKind::default(),
			first: 0,
			second: 0,
			first_anchor: Vec3::ZERO,
			second_anchor: Vec3::ZERO,
			axis: Vec3::ZERO,
			length: 0.0,
			rest: Quat::IDENTITY,
			stiffness: Joint::RIGID,
			damping: Joint::DAMPING,
			max_impulse: Joint::NO_CEILING,
			max_torque: Joint::NO_CEILING,
			collide: false,
		}
	}
}

impl Link {
	/// The slot it occupied and the generation it held there.
	///
	/// What a restore keys the table's rebuilt generation array on.
	#[must_use]
	pub const fn key(&self) -> (u32, u32) { (self.slot, self.generation) }
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

	/// Every pose that was alive.
	pub posed: Vec<Posed>,

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

	/// The same for pose slots.
	pub pose_generations: Vec<u32>,

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

	/// Which bodies are held to this one, directly or through others.
	///
	/// The joints are a graph and this is its connected component: a body, what
	/// is welded to it, what is welded to *that*, and so on. Everything a
	/// sandbox calls a contraption is one of these, and the reason it is here
	/// rather than in a game is that the question is about a description rather
	/// than about a world. The editor asks it to work out what "the thing I
	/// clicked" is; the duplicator asks it to work out what to copy; neither
	/// needs a `World` to answer it and neither should have to write it.
	///
	/// **A joint pinned to the world joins nothing to anything.** Its second
	/// end is [`NO_INDEX`] rather than a body, so a prop bolted to a wall is
	/// its own component and is not dragged along by whatever else is bolted to
	/// the same wall.
	///
	/// @param seed - which of [`solids`](Self::solids) to start from
	/// @return every index in its component, in order, seed included; empty if
	/// the seed is not an index into this description
	#[must_use]
	pub fn connected(&self, seed: u32) -> Vec<u32> {
		let Ok(count) = u32::try_from(self.solids.len()) else {
			return Vec::new();
		};

		if seed >= count {
			return Vec::new();
		}

		let mut found = vec![seed];
		let mut walked = 0;

		// breadth first over the joints, which for a contraption of a few dozen
		// parts is cheaper than building an adjacency list to walk it with.
		while walked < found.len() {
			let here = found[walked];
			walked += 1;

			let mut reached: Vec<u32> = self
				.links
				.iter()
				.filter_map(|link| across(link, here))
				.filter(|&far| far < count && !found.contains(&far))
				.collect();
			// two joints between the same pair would otherwise add it twice
			reached.sort_unstable();
			reached.dedup();

			found.extend(reached);
		}

		found.sort_unstable();

		found
	}

	/// A description of exactly these bodies, what they drive and what holds
	/// them together.
	///
	/// The other half of cutting a piece out. What comes back is a `SceneData`
	/// like any other, so [`instantiate`] pastes it beside what is already
	/// there and `level::export` writes it down as a file - which is what makes
	/// a duplicate and a saved contraption the same operation rather than two.
	///
	/// Three things are decided here and each could have gone the other way:
	///
	/// - **the stage comes along.** A piece cut out of a world under a
	///   particular gravity is a piece that was settled under it, and dropping
	///   that would make a saved contraption behave differently from the one
	///   that was saved. [`instantiate`] ignores it; a restore does not.
	/// - **the arena does not.** It is the game's own bytes and the handles in
	///   them name things that are not here. Same rule a paste already follows.
	/// - **a joint pinned to the world comes along**, because it is a property
	///   of the body it holds rather than of the pair. A paste offsets its
	///   anchor with everything else, so a copy of a prop bolted to a wall is
	///   bolted to the matching point beside it.
	///
	/// @param solids - which of [`solids`](Self::solids) to keep; anything out
	/// of range or named twice is ignored
	/// @return a description of them alone
	#[must_use]
	pub fn subset(&self, solids: &[u32]) -> Self {
		let mut keeping: Vec<u32> = Vec::new();
		for &index in solids {
			if usize::try_from(index).is_ok_and(|it| it < self.solids.len())
				&& !keeping.contains(&index)
			{
				keeping.push(index);
			}
		}

		// old index to new, for both tables. Sized by the description rather
		// than by what is kept, so a lookup is an index into it directly.
		let mut solid_of = vec![NO_INDEX; self.solids.len()];
		let mut thing_of = vec![NO_INDEX; self.things.len()];
		let (mut things, mut kept) = (Vec::new(), Vec::new());

		for &index in &keeping {
			let Some(solid) = self
				.solids
				.get(usize::try_from(index).unwrap_or(usize::MAX))
			else {
				continue;
			};

			solid_of[usize::try_from(index).unwrap_or(0)] = count(kept.len());
			let mut copy = solid.clone();
			copy.slot = count(kept.len());
			copy.generation = 1;
			// the entity it drives, if it has not already come along behind
			// another body. Two bodies naming one entity is legal and is what a
			// prop with a second collider would be.
			copy.thing = drag_along(&self.things, solid.thing, &mut thing_of, &mut things);

			kept.push(copy);
		}

		let mut links = Vec::new();
		for link in &self.links {
			let first = at_new(&solid_of, link.first);
			let second = at_new(&solid_of, link.second);

			// both ends, or one end and the world. A joint whose far end was
			// left behind is dropped: half a joint holds nothing and a paste
			// would refuse it anyway.
			if first == NO_INDEX || (link.second != NO_INDEX && second == NO_INDEX) {
				continue;
			}

			let mut copy = link.clone();
			copy.slot = count(links.len());
			copy.generation = 1;
			copy.first = first;
			copy.second = second;
			links.push(copy);
		}

		// the poses of whatever came along, renumbered like everything else. A
		// character cut out of a world is cut out standing as it stood; two of
		// its entities naming one pose still name one pose afterwards.
		let mut pose_of = vec![NO_INDEX; self.posed.len()];
		let mut posed = Vec::new();

		for thing in &mut things {
			thing.pose = drag_pose(&self.posed, thing.pose, &mut pose_of, &mut posed);
		}

		Self {
			stage: self.stage,
			thing_generations: vec![1; things.len()],
			solid_generations: vec![1; kept.len()],
			link_generations: vec![1; links.len()],
			pose_generations: vec![1; posed.len()],
			posed,
			things,
			solids: kept,
			links,
			// never: the bytes name handles that are not in here. @ref the doc
			// above, and the same rule `instantiate` follows.
			arena: None,
		}
	}
}

/// A length as an index, or [`NO_INDEX`] if it will not fit in one.
fn count(len: usize) -> u32 { u32::try_from(len).unwrap_or(NO_INDEX) }

/// The body at the far end of a joint from this one, if there is one.
///
/// Nothing when the joint does not name this body at all, and nothing when the
/// far end is the world: a wall is not a member of anything.
///
/// @param link - the joint
/// @param here - the body being walked from
fn across(link: &Link, here: u32) -> Option<u32> {
	match (link.first, link.second) {
		| (first, second) if first == here && second != NO_INDEX => Some(second),
		| (first, second) if second == here && first != NO_INDEX => Some(first),
		| _ => None,
	}
}

/// Brings a body's entity into a piece, or finds where it already went.
///
/// Two bodies naming one entity is legal - it is what a prop with a second
/// collider would be - so this is idempotent per entity rather than per body.
///
/// @param things - the entities of the description being cut from
/// @param at - which of them the body drives, or [`NO_INDEX`]
/// @param moved - old index to new, written
/// @param kept - the piece's own entities, appended to
/// @return where the entity now is, or [`NO_INDEX`] if the body drives nothing
fn drag_along(things: &[Thing], at: u32, moved: &mut [u32], kept: &mut Vec<Thing>) -> u32 {
	let Ok(at) = usize::try_from(at) else {
		return NO_INDEX;
	};

	let Some(thing) = things.get(at) else {
		return NO_INDEX;
	};

	if moved[at] == NO_INDEX {
		moved[at] = count(kept.len());
		let mut copy = thing.clone();
		copy.slot = count(kept.len());
		copy.generation = 1;
		kept.push(copy);
	}

	moved[at]
}

/// The same for the pose an entity that came along is moved by.
///
/// A copy of [`drag_along`] rather than one function generic over the two:
/// what the records have in common is two words, and a trait to say so would
/// be more machinery than the second copy is.
fn drag_pose(posed: &[Posed], at: u32, moved: &mut [u32], kept: &mut Vec<Posed>) -> u32 {
	let Ok(at) = usize::try_from(at) else {
		return NO_INDEX;
	};

	let Some(pose) = posed.get(at) else {
		return NO_INDEX;
	};

	if moved[at] == NO_INDEX {
		moved[at] = count(kept.len());
		let mut copy = pose.clone();
		copy.slot = count(kept.len());
		copy.generation = 1;
		kept.push(copy);
	}

	moved[at]
}

/// What an old index became, or [`NO_INDEX`] for the world and for anything
/// left behind.
fn at_new(table: &[u32], index: u32) -> u32 {
	usize::try_from(index)
		.ok()
		.and_then(|it| table.get(it).copied())
		.unwrap_or(NO_INDEX)
}

/// Writes down everything about a world that a world is made of.
///
/// @param world - what to describe
/// @return the description, with the arena in it
#[must_use]
pub fn capture(world: &World) -> SceneData {
	// before the entities, because a thing points at a pose by index and the
	// indices only exist once the list does.
	let posed = poses(world);
	let mut pose_of = vec![NO_INDEX; world.poses.slots()];

	for (index, pose) in posed.iter().enumerate() {
		let (Ok(index), Ok(slot)) = (u32::try_from(index), usize::try_from(pose.slot)) else {
			continue;
		};

		if let Some(entry) = pose_of.get_mut(slot) {
			*entry = index;
		}
	}

	let things = things(world, &pose_of);
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
		posed,
		pose_generations: (0..world.poses.slots())
			.map(|slot| world.poses.generation(slot))
			.collect(),
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
fn things(world: &World, pose_of: &[u32]) -> Vec<Thing> {
	world
		.entities
		.iter()
		.map(|(id, transform, renderable)| Thing {
			pose: pose_of
				.get(renderable.pose.slot())
				.copied()
				.filter(|_| world.poses.alive(renderable.pose))
				.unwrap_or(NO_INDEX),
			name: world.entities.name(id).to_owned(),
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

/// Every living pose, with its skeleton named.
fn poses(world: &World) -> Vec<Posed> {
	world
		.poses
		.iter()
		.map(|(id, pose)| Posed {
			// nothing in the world names a pose, so a captured one has none.
			// The writer invents one where something refers to it.
			name: String::new(),
			slot: u32::try_from(id.slot()).unwrap_or(0),
			generation: id.generation(),
			skeleton: world
				.skeletons
				.get(pose.skeleton)
				.map_or_else(String::new, |entry| entry.name().to_owned()),
			locals: pose.locals.clone(),
		})
		.collect()
}

/// Every living body, with its entity turned into an index.
fn solids(world: &World, thing_of: &[u32]) -> Vec<Solid> {
	world
		.bodies
		.iter()
		.map(|(id, body)| Solid {
			name: world.bodies.name(id).to_owned(),
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
			weightless: body.weightless,
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
			name: world.joints.name(id).to_owned(),
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
			stiffness: joint.stiffness,
			damping: joint.damping,
			max_impulse: joint.max_impulse,
			max_torque: joint.max_torque,
			collide: joint.collide,
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

crate::registry_handle! {
	/// A handle to a scene the host has loaded.
	///
	/// Not generational, like every other resource handle and unlike the ones
	/// into the world's own tables: a scene is compiled from a file and lives
	/// as long as the process, and recompiling one rewrites the entry the
	/// handle already points at. @ref [`registry`](crate::abi::registry).
	SceneId
}

/// One loaded scene: what it is called, what is in it, and how many times it
/// has been recompiled.
pub type Scene = Entry<SceneData>;

/// Every scene the host has loaded, by name.
///
/// Filled from `assets/` like the meshes are, and read by a game that wants to
/// put one into the world - either replacing what is there with
/// [`restore`](crate::abi::scene::restore) or creating a copy of it beside
/// what is there with [`instantiate`].
#[derive(Debug)]
pub struct Scenes {
	entries: Registry<SceneData>,
}

impl Scenes {
	/// A table holding nothing but its null entry.
	#[must_use]
	pub fn new() -> Self {
		Self {
			entries: Registry::new(SceneData::default()),
		}
	}

	/// Looks a scene up by name.
	///
	/// @param name - the asset name, `scenes/props` for
	/// `assets/scenes/props.scene` @return its handle, or [`SceneId::NONE`] if
	/// nothing answers to that name
	#[must_use]
	pub fn find(&self, name: &str) -> SceneId { SceneId::new(self.entries.find(name)) }

	/// Registers a scene under a name, replacing whatever was there.
	///
	/// @param name - what the game will ask for
	/// @param data - the description
	/// @return the handle, the same one as last time if the name is known
	pub fn insert(&mut self, name: &str, data: SceneData) -> SceneId {
		SceneId::new(self.entries.insert(name, data))
	}

	/// One scene, by handle.
	#[must_use]
	pub fn get(&self, id: SceneId) -> Option<&Scene> { self.entries.entry(id.index()) }

	/// What is in a scene, by handle.
	///
	/// The usual way in, because what a game does with a scene is hand it to a
	/// loader. An unknown handle answers with a description of nothing, which
	/// creates nothing and replaces nothing.
	#[must_use]
	pub fn data(&self, id: SceneId) -> &SceneData { self.get(id).map_or(&EMPTY, Entry::value) }

	/// How many scenes there are, counting the null one.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Always `false`: the null entry always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// Every scene, in slot order, starting with the null one.
	pub fn iter(&self) -> impl Iterator<Item = &Scene> { self.entries.iter() }
}

impl Default for Scenes {
	fn default() -> Self { Self::new() }
}

/// What an unknown handle reads as.
static EMPTY: SceneData = SceneData {
	stage: Stage::DEFAULT,
	things: Vec::new(),
	solids: Vec::new(),
	links: Vec::new(),
	posed: Vec::new(),
	thing_generations: Vec::new(),
	solid_generations: Vec::new(),
	link_generations: Vec::new(),
	pose_generations: Vec::new(),
	arena: None,
};

/// What a restore put back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Restored {
	/// How many entities landed in a slot.
	pub things: usize,

	/// How many bodies did.
	pub solids: usize,

	/// How many joints did.
	pub links: usize,

	/// How many poses did.
	pub posed: usize,

	/// Whether the game's own arena was put back.
	pub arena: bool,
}

/// Puts a described world back exactly as it was.
///
/// Everything alive is replaced rather than added to, and every table is
/// rebuilt slot for slot and generation for generation - which is what makes
/// the handles a game kept in its arena resolve to the same entities
/// afterwards. That property is the whole reason this exists, and it is why
/// there is a second loader for the case where a copy rather than the original
/// is wanted.
///
/// **The caller has one obligation, and nothing here can discharge it: drop
/// whatever a solver derived.** Warm-start impulses, baked collision meshes
/// and the set of pairs that were touching all live outside [`World`], and a
/// step run against them after a restore is a step mixing this world with the
/// one that was here before. `colby_core` cannot reach the solver, so this is
/// the host's to do, immediately after this call.
///
/// The arena is refused rather than forced when its layout number and the
/// running game's disagree: half a loaded world - entities standing where they
/// were and a game holding handles to none of them - is worse than a load that
/// did not happen. A description carrying no arena at all is not a mismatch;
/// there is simply nothing to put.
///
/// @param world - the world to replace
/// @param scene - what to put in it
/// @return what landed
///
/// # Errors
///
/// If the description's arena was written under a different layout number than
/// the running game claimed. Nothing is changed in that case.
pub fn restore(world: &mut World, scene: &SceneData) -> Result<Restored> {
	agreed(world, scene)?;

	// nothing playing survives, and this is the one thing a restore has to do
	// about audio. A description carries no voices - a voice is a moment
	// rather than a thing - so the arena about to be put back holds handles
	// into a table nobody restored, and a sound still running is one nothing
	// can name. The same class of debt as `Simulation::forget`, and unlike
	// that one it can be paid here, because the table is in `World`.
	world.audio.stop_all();

	// before the entities, because a restored entity's renderable names a pose
	// handle and the handles only exist once the table has been rebuilt.
	let generations = slots(&scene.pose_generations, scene.posed.iter().map(Posed::key));
	let entries = poses_of(world, scene);
	let poses = world.poses.restore(&generations, &entries);

	let generations = slots(&scene.thing_generations, scene.things.iter().map(Thing::key));
	let entries = placed(world, scene, &poses);
	let things = world.entities.restore(&generations, &entries);

	let generations = slots(&scene.solid_generations, scene.solids.iter().map(Solid::key));
	let entries = solid_bodies(world, scene, &things);
	let solids = world.bodies.restore(&generations, &entries);

	let generations = slots(&scene.link_generations, scene.links.iter().map(Link::key));
	let entries = link_joints(scene, &solids);
	let links = world.joints.restore(&generations, &entries);

	// after the three restores rather than inside them: a table restore is
	// handed slots and plain records, and a handle is the only way to address
	// what it just put back. It is also the only way that cannot get the two
	// lists out of step, because a record whose slot the table refused comes
	// back as a null handle and names nothing.
	for (id, thing) in things.iter().zip(&scene.things) {
		world.entities.set_name(*id, &thing.name);
	}

	for (id, solid) in solids.iter().zip(&scene.solids) {
		world.bodies.set_name(*id, &solid.name);
	}

	for (id, link) in links.iter().zip(&scene.links) {
		world.joints.set_name(*id, &link.name);
	}

	if let Some(arena) = scene.arena.as_ref() {
		world.state.put_raw(&arena.bytes, arena.layout);
	}

	stage_world(world, scene.stage);

	Ok(Restored {
		things: things.iter().filter(|id| id.is_some()).count(),
		solids: solids.iter().filter(|id| id.is_some()).count(),
		links: links.iter().filter(|id| id.is_some()).count(),
		posed: poses.iter().filter(|id| id.is_some()).count(),
		arena: scene.arena.is_some(),
	})
}

/// Whether the description's arena is one this build can read.
///
/// @param world - the world whose game has claimed a layout number
/// @param scene - the description
///
/// # Errors
///
/// If both sides claim a layout and the two disagree.
fn agreed(world: &World, scene: &SceneData) -> Result<()> {
	let claimed = world.state.layout();
	let Some(arena) = scene.arena.as_ref() else {
		return Ok(());
	};

	// a zero on either side is "nobody has claimed this", which agrees with
	// everything: a world whose game has not run yet, or a description written
	// before one did.
	if arena.layout == 0 || claimed == 0 || arena.layout == claimed {
		return Ok(());
	}

	Err(err!(Asset(
		"the scene's game state is layout {} and this build claims {claimed}",
		arena.layout
	)))
}

/// The generation of every slot a table should end up with.
///
/// The written array is the whole picture, dead slots included; each living
/// record then overwrites its own slot, because the record is what the
/// generation is actually about and the array is a second copy of the same
/// fact. A record past the end of the array grows it, which is what lets a
/// description somebody wrote by hand carry no array at all.
///
/// @param written - the array the description carried, possibly empty
/// @param records - the slot and generation of each living record
fn slots<I: Iterator<Item = (u32, u32)>>(written: &[u32], records: I) -> Vec<u32> {
	let mut generations = written.to_vec();

	for (slot, generation) in records {
		let Ok(slot) = usize::try_from(slot) else {
			continue;
		};

		if slot >= generations.len() {
			generations.resize(slot.saturating_add(1), 0);
		}

		generations[slot] = generation;
	}

	generations
}

/// Every entity, with its named assets looked up.
fn placed(
	world: &World,
	scene: &SceneData,
	poses: &[PoseId],
) -> Vec<(usize, Transform, Renderable)> {
	scene
		.things
		.iter()
		.map(|thing| {
			let renderable = Renderable {
				mesh: world.meshes.find(&thing.mesh),
				material: material(world, &thing.material),
				color: thing.color,
				pose: at(poses, thing.pose).unwrap_or(PoseId::NONE),
			};

			(usize::try_from(thing.slot).unwrap_or(usize::MAX), thing.transform, renderable)
		})
		.collect()
}

/// Every pose, with its skeleton looked up and its past caught up.
///
/// A world put back has not been stepped, so there is nothing between its past
/// and its present to draw: they are the same, and saying so here is why the
/// description carries one set of bones rather than two.
fn poses_of(world: &World, scene: &SceneData) -> Vec<(usize, Pose)> {
	scene
		.posed
		.iter()
		.map(|posed| (usize::try_from(posed.slot).unwrap_or(usize::MAX), pose_of(world, posed)))
		.collect()
}

/// One pose, with its skeleton looked up and its bones filled in.
///
/// **A description with no bones in it is a pose at rest**, not a pose of no
/// bones. That is the whole of how an authored scene works: a source says
/// which skeleton and nothing else, because where a character's bones are is a
/// property of the moment rather than of the level, and what comes out stands
/// in the shape its model was drawn in.
fn pose_of(world: &World, posed: &Posed) -> Pose {
	let skeleton = world.skeletons.find(&posed.skeleton);

	if posed.locals.is_empty() {
		return Pose::resting(skeleton, world.skeletons.bones(skeleton));
	}

	Pose {
		skeleton,
		previous: posed.locals.clone(),
		locals: posed.locals.clone(),
	}
}

/// Every body, with its shape's mesh and its entity looked up.
fn solid_bodies(world: &World, scene: &SceneData, things: &[EntityId]) -> Vec<(usize, Body)> {
	scene
		.solids
		.iter()
		.map(|solid| {
			let mut body = Body::new(solid.kind, shape(world, &solid.shape), solid.transform);
			body.velocity = solid.velocity;
			body.angular = solid.angular;
			body.mass = solid.mass;
			body.restitution = solid.restitution;
			body.friction = solid.friction;
			body.sensor = solid.sensor;
			body.weightless = solid.weightless;
			body.sleeping = solid.sleeping;
			body.layers = solid.layers;
			body.entity = at(things, solid.thing).unwrap_or(EntityId::NONE);

			(usize::try_from(solid.slot).unwrap_or(usize::MAX), body)
		})
		.collect()
}

/// Every joint, with both of its bodies looked up.
fn link_joints(scene: &SceneData, solids: &[BodyId]) -> Vec<(usize, Joint)> {
	scene
		.links
		.iter()
		.map(|link| {
			let joint = Joint {
				kind: link.kind,
				first: at(solids, link.first).unwrap_or(BodyId::NONE),
				second: at(solids, link.second).unwrap_or(BodyId::NONE),
				first_anchor: link.first_anchor,
				second_anchor: link.second_anchor,
				axis: link.axis,
				length: link.length,
				rest: link.rest,
				stiffness: link.stiffness,
				damping: link.damping,
				max_impulse: link.max_impulse,
				max_torque: link.max_torque,
				collide: link.collide,
			};

			(usize::try_from(link.slot).unwrap_or(usize::MAX), joint)
		})
		.collect()
}

/// One handle out of the list a table restore handed back.
///
/// @param handles - what the restore returned, one per record
/// @param index - what a record wrote down, or [`NO_INDEX`]
fn at<T: Copy>(handles: &[T], index: u32) -> Option<T> {
	if index == NO_INDEX {
		return None;
	}

	handles.get(usize::try_from(index).ok()?).copied()
}

/// A material by name, falling back to the default rather than to nothing.
///
/// The two branches produce the same handle and differ only in whether
/// anything is said about it, which is the point: a name nothing answers to is
/// worth a line in the log, and an entity that simply never chose a material
/// is not. Without the first branch every unnamed entity in a hand-written
/// scene would warn, which is how a warning stops being read.
fn material(world: &World, name: &str) -> MaterialId {
	if name.is_empty() {
		return MaterialId::DEFAULT;
	}

	let found = world.materials.find(name);
	if !found.is_some() {
		warn!(name, "a scene names a material nothing answers to");

		return MaterialId::DEFAULT;
	}

	found
}

/// A shape with its named mesh looked up.
fn shape(world: &World, form: &Form) -> Shape {
	Shape {
		kind: form.kind,
		radius: form.radius,
		extents: form.extents,
		mesh: world.meshes.find(&form.mesh),
	}
}

/// Writes the world's own settings and declares the whole thing a cut.
///
/// A load is the largest discontinuity there is, so nothing is drawn
/// traveling from where it used to be: every transform's past was written to
/// match its present by the table restores, and the camera says so here.
fn stage_world(world: &mut World, stage: Stage) {
	world.camera = stage.camera;
	world.clear = stage.clear;
	world.light = stage.light;
	world.ambient = stage.ambient;
	world.gravity = stage.gravity;
	world.time = stage.time;
	world.steps = stage.steps;
	world.contacts = 0;

	world.snap_camera();
	world.entities.snap_all();
}

/// What a scene turned into when it was created beside something else.
///
/// A restore needs nothing like this, because the whole point of one is that
/// the handles are the handles they were. An instantiate is the opposite: it
/// creates, so every handle is new, and the file's own way of naming things -
/// an index into it, or a name somebody typed - is the only thing that can
/// still be pointed at afterwards. This is the translation, and it is what
/// [`instantiate`] hands back instead of writing into an arena it cannot read.
#[derive(Clone, Debug, Default)]
pub struct Remap {
	/// Each entity's name and what it became, in the file's order.
	things: Vec<(String, EntityId)>,

	/// The same for bodies.
	solids: Vec<(String, BodyId)>,

	/// And for joints.
	links: Vec<(String, JointId)>,

	/// And for poses, which have no names because nothing points at one by
	/// anything but its place in the file.
	poses: Vec<PoseId>,
}

impl Remap {
	/// What the entity at a place in the file became.
	///
	/// @param index - its place in [`SceneData::things`]
	/// @return the handle, or [`EntityId::NONE`] for an index that is not one
	/// or an entity the world had no room for
	#[must_use]
	pub fn entity(&self, index: u32) -> EntityId {
		handle(&self.things, index).unwrap_or(EntityId::NONE)
	}

	/// What the body at a place in the file became.
	#[must_use]
	pub fn body(&self, index: u32) -> BodyId {
		handle(&self.solids, index).unwrap_or(BodyId::NONE)
	}

	/// What the joint at a place in the file became.
	#[must_use]
	pub fn joint(&self, index: u32) -> JointId {
		handle(&self.links, index).unwrap_or(JointId::NONE)
	}

	/// What the pose at a place in the file became.
	///
	/// @param index - its place in [`SceneData::posed`]
	/// @return the handle, or [`PoseId::NONE`] for an index that is not one or
	/// a pose the world had no room for
	#[must_use]
	pub fn pose(&self, index: u32) -> PoseId { at(&self.poses, index).unwrap_or(PoseId::NONE) }

	/// What the entity somebody called this became.
	///
	/// The way an authored scene is read: a person writing one names the floor
	/// `floor` and the game asks for it by that. An empty name never matches,
	/// because a record naming nothing is not a record called "".
	///
	/// @param name - what the file calls it
	#[must_use]
	pub fn entity_named(&self, name: &str) -> EntityId {
		named(&self.things, name).unwrap_or(EntityId::NONE)
	}

	/// What the body somebody called this became.
	#[must_use]
	pub fn body_named(&self, name: &str) -> BodyId {
		named(&self.solids, name).unwrap_or(BodyId::NONE)
	}

	/// What the joint somebody called this became.
	#[must_use]
	pub fn joint_named(&self, name: &str) -> JointId {
		named(&self.links, name).unwrap_or(JointId::NONE)
	}

	/// Every entity that was created, in the file's order.
	pub fn entities(&self) -> impl Iterator<Item = EntityId> {
		self.things.iter().map(|(_, id)| *id)
	}

	/// Every body that was created.
	pub fn bodies(&self) -> impl Iterator<Item = BodyId> { self.solids.iter().map(|(_, id)| *id) }

	/// Every joint that was created.
	pub fn joints(&self) -> impl Iterator<Item = JointId> { self.links.iter().map(|(_, id)| *id) }
}

/// One handle out of a list of named ones, by its place in the file.
fn handle<T: Copy>(list: &[(String, T)], index: u32) -> Option<T> {
	if index == NO_INDEX {
		return None;
	}

	list.get(usize::try_from(index).ok()?)
		.map(|(_, id)| *id)
}

/// One handle out of a remap's list, by name.
fn named<T: Copy>(list: &[(String, T)], name: &str) -> Option<T> {
	if name.is_empty() {
		return None;
	}

	list.iter()
		.find(|(written, _)| written == name)
		.map(|(_, id)| *id)
}

/// Creates everything a scene describes, beside whatever is already there.
///
/// The other loader, and the one a dupe, a prefab and a level laid into a
/// running game all go through. Nothing existing is disturbed: the tables grow,
/// the slots and generations in the description are ignored because they belong
/// to a world this is not, and the world's own settings are left where they
/// are - pasting a prop is not a reason to move somebody's camera.
///
/// **The arena is not touched, and it cannot be.** A restore puts the game's
/// bytes back because the handles in them still resolve; here every handle is
/// new, and nothing outside the game knows where in those four thousand bytes
/// a handle even is. So this hands back a [`Remap`] instead, and the game
/// writes down whatever it cares about itself. That is the same rule model
/// placements follow: what a description becomes is a loop the game writes.
///
/// A body arrives awake whatever the description says. Sleeping is a claim
/// about how long something has been still, and a body that was created a
/// moment ago has not been still for any length of time - the solver would
/// have to be told the same thing separately, and a dupe pasted in mid-air
/// hanging there is the shape of getting it wrong.
///
/// @param world - the world to add to
/// @param scene - what to create
/// @param at - added to every position, so a copy lands beside its original
/// rather than inside it; [`Vec3::ZERO`] puts everything exactly where the
/// description says
/// @return what each record became
pub fn instantiate(world: &mut World, scene: &SceneData, at: Vec3) -> Remap {
	// before the entities, for the reason a restore does it first: an entity
	// names a pose by handle and the handles do not exist yet. Two things
	// naming one index come out sharing one pose, which is what a copy of a
	// character has to be.
	let poses: Vec<PoseId> = scene
		.posed
		.iter()
		.map(|posed| spawn_pose(world, posed))
		.collect();

	let things: Vec<(String, EntityId)> = scene
		.things
		.iter()
		.map(|thing| (thing.name.clone(), spawn_thing(world, thing, &poses, at)))
		.collect();

	let solids: Vec<(String, BodyId)> = scene
		.solids
		.iter()
		.map(|solid| (solid.name.clone(), spawn_solid(world, solid, &things, at)))
		.collect();

	let links: Vec<(String, JointId)> = scene
		.links
		.iter()
		.map(|link| (link.name.clone(), spawn_link(world, link, &solids, at)))
		.collect();

	Remap { things, solids, links, poses }
}

/// Creates one pose, standing where the description left its bones.
///
/// A copy of a posed character is posed the same way, not put back at rest: a
/// duplicate of something mid-stride should look like what was duplicated.
/// Its past is its present, because it has not moved yet.
fn spawn_pose(world: &mut World, posed: &Posed) -> PoseId {
	let pose = pose_of(world, posed);

	world.poses.spawn(pose)
}

/// Creates one entity.
fn spawn_thing(world: &mut World, thing: &Thing, poses: &[PoseId], at: Vec3) -> EntityId {
	let mut transform = thing.transform;
	transform.position += at;

	let id = world.entities.spawn_at(transform);
	if !id.is_some() {
		return id;
	}

	// the copy is called what the original was called. Two things with one
	// name is exactly what a dupe is, and the handle is what tells them apart.
	world.entities.set_name(id, &thing.name);
	// the shadowing is why this is not `at`: the offset a paste applies is
	// also called that, and it is the older of the two names here.
	let pose = usize::try_from(thing.pose)
		.ok()
		.and_then(|index| poses.get(index))
		.copied()
		.unwrap_or(PoseId::NONE);

	world.entities.set_renderable(id, Renderable {
		mesh: world.meshes.find(&thing.mesh),
		material: material(world, &thing.material),
		color: thing.color,
		pose,
	});

	id
}

/// Creates one body, pointing at whatever its entity became.
fn spawn_solid(
	world: &mut World,
	solid: &Solid,
	things: &[(String, EntityId)],
	at: Vec3,
) -> BodyId {
	let mut transform = solid.transform;
	transform.position += at;

	let mut body = Body::new(solid.kind, shape(world, &solid.shape), transform);
	body.velocity = solid.velocity;
	body.angular = solid.angular;
	body.mass = solid.mass;
	body.restitution = solid.restitution;
	body.friction = solid.friction;
	body.sensor = solid.sensor;
	body.weightless = solid.weightless;
	body.layers = solid.layers;
	body.entity = handle(things, solid.thing).unwrap_or(EntityId::NONE);

	let id = world.bodies.spawn(body);
	world.bodies.set_name(id, &solid.name);

	id
}

/// Creates one joint, if both the bodies it names are there.
///
/// A joint whose second body is nothing is a joint pinned to a point in the
/// world, which is a real thing to be. A joint whose second body *failed* is
/// not, and it would be indistinguishable from the first if it were created
/// anyway - so it is refused instead, which is loud in the only way this can
/// be: the joint is missing rather than holding the wrong thing.
fn spawn_link(world: &mut World, link: &Link, solids: &[(String, BodyId)], at: Vec3) -> JointId {
	let first = handle(solids, link.first);
	let second = handle(solids, link.second);

	if (link.first != NO_INDEX && first.is_none())
		|| (link.second != NO_INDEX && second.is_none())
	{
		return JointId::NONE;
	}

	// only a joint pinned to the world has an anchor in world space, and only
	// that one moves with the paste.
	let anchored = if link.second == NO_INDEX {
		link.second_anchor + at
	} else {
		link.second_anchor
	};

	let id = world.joints.spawn(Joint {
		kind: link.kind,
		first: first.unwrap_or(BodyId::NONE),
		second: second.unwrap_or(BodyId::NONE),
		first_anchor: link.first_anchor,
		second_anchor: anchored,
		axis: link.axis,
		length: link.length,
		// the description's own, not one worked out here: a weld holds the
		// angle it was *made* at, and the two bodies have just been created at
		// whatever angle they were written down at, so the answer is already in
		// the file. @ref `World::join`, which is the other case.
		rest: link.rest,
		stiffness: link.stiffness,
		damping: link.damping,
		max_impulse: link.max_impulse,
		max_torque: link.max_torque,
		collide: link.collide,
	});
	world.joints.set_name(id, &link.name);

	id
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::abi::{
		MAX_ENTITIES, Material, MeshData, mesh,
		skeleton::{Bone, SkeletonData, SkeletonId},
	};

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
	/// A world with two entities, a body under one of them and a rope holding
	/// it - one of each thing the description carries.
	fn peopled() -> World {
		let mut world = furnished();
		let mesh = world.meshes.find("meshes/crystal");
		let material = world.materials.find("brass");

		let first = world.entities.spawn_at(Transform::at(Vec3::X));
		world
			.entities
			.set_renderable(first, Renderable::of(mesh, material, Vec3::Y));

		let second = world
			.entities
			.spawn_at(Transform::at(Vec3::new(0.0, 4.0, 0.0)));
		let body = world.bodies.spawn(
			Body::dynamic(Shape::ball(0.5), Transform::at(Vec3::new(0.0, 4.0, 0.0)), 3.0)
				.driving(second)
				.layered(Layers::single(2)),
		);
		world.join(Joint::rope(body, BodyId::NONE, (Vec3::ZERO, Vec3::Y * 6.0), 2.0));
		world.camera.position = Vec3::new(4.0, 5.0, 6.0);
		world.time = 3.25;

		world
	}

	/// A skeleton of two bones, registered under a name.
	fn rigged(world: &mut World) -> SkeletonId {
		world
			.skeletons
			.insert("models/hero/rig", SkeletonData {
				bones: vec![
					Bone {
						name: "hips".to_owned(),
						..Bone::default()
					},
					Bone {
						name: "head".to_owned(),
						parent: 0,
						rest: Transform::at(Vec3::Y),
						..Bone::default()
					},
				],
			})
	}

	/// A world with one character drawn as two entities sharing one pose.
	fn character() -> World {
		let mut world = furnished();
		let skeleton = rigged(&mut world);
		let mesh = world.meshes.find("meshes/crystal");
		let pose = world
			.poses
			.spawn(Pose::resting(skeleton, world.skeletons.bones(skeleton)));

		world
			.poses
			.get_mut(pose)
			.expect("the pose is there")
			.set(1, Transform::at(Vec3::new(0.0, 2.0, 0.0)));

		for name in ["body", "eyes"] {
			let id = world.entities.spawn_at(Transform::IDENTITY);

			world.entities.set_name(id, name);
			world
				.entities
				.set_renderable(id, Renderable::new(mesh, Vec3::ONE).posed(pose));
		}

		world
	}

	#[test]
	fn a_posed_world_put_back_captures_the_same_way_it_did_before() {
		let world = character();
		let scene = capture(&world);

		assert_eq!(scene.posed.len(), 1, "one pose, however many entities wear it");
		assert_eq!(scene.posed[0].skeleton, "models/hero/rig", "named rather than handled");
		assert_eq!(scene.posed[0].locals.len(), 2, "with both its bones written down");

		let mut empty = furnished();
		rigged(&mut empty);

		let report = restore(&mut empty, &scene).expect("the layouts agree");

		assert_eq!(report.posed, 1, "the pose landed");
		assert_eq!(capture(&empty), scene, "and describing it again describes what was read");
	}

	#[test]
	fn two_entities_that_shared_a_pose_still_share_one_afterwards() {
		let scene = capture(&character());
		let mut world = furnished();

		rigged(&mut world);

		let remap = instantiate(&mut world, &scene, Vec3::ZERO);
		let body = remap.entity(0);
		let eyes = remap.entity(1);
		let of = |id| {
			world
				.entities
				.renderable(id)
				.map(|renderable| renderable.pose)
		};

		assert_eq!(of(body), of(eyes), "one pose, not one each");
		assert!(of(body).is_some_and(PoseId::is_some), "and a real one");
		assert_eq!(world.poses.len(), 1, "so the table holds one");
		assert_eq!(remap.pose(0), of(body).expect("it is there"), "which the remap names");
	}

	#[test]
	fn a_pasted_character_stands_the_way_the_one_it_was_copied_from_stood() {
		let scene = capture(&character());
		let mut world = furnished();

		rigged(&mut world);
		instantiate(&mut world, &scene, Vec3::ZERO);

		let (_, pose) = world.poses.iter().next().expect("one pose");

		assert_eq!(
			pose.locals[1].position,
			Vec3::new(0.0, 2.0, 0.0),
			"mid-stride rather than back at rest"
		);
		assert_eq!(pose.previous, pose.locals, "and it has not moved yet, so it does not smear");
	}

	#[test]
	fn a_description_that_names_a_skeleton_and_no_bones_comes_back_resting() {
		let mut world = furnished();

		rigged(&mut world);

		// what a source produces: the skeleton and nothing else. It is not a
		// pose of no bones, it is a character standing as it was drawn.
		let scene = SceneData {
			posed: vec![Posed {
				name: "hero".to_owned(),
				slot: 0,
				generation: 1,
				skeleton: "models/hero/rig".to_owned(),
				locals: Vec::new(),
			}],
			pose_generations: vec![1],
			..SceneData::default()
		};

		restore(&mut world, &scene).expect("nothing disagrees");

		let (_, pose) = world.poses.iter().next().expect("one pose");

		assert_eq!(pose.locals.len(), 2, "filled from the skeleton it named");
		assert_eq!(pose.locals[1].position, Vec3::Y, "at the rest the skeleton has");
	}

	#[test]
	fn a_pose_whose_skeleton_did_not_load_is_a_character_standing_still() {
		let scene = capture(&character());
		// the same description into a world that never saw the skeleton.
		let mut world = furnished();

		restore(&mut world, &scene).expect("nothing disagrees");

		let (id, pose) = world
			.poses
			.iter()
			.next()
			.expect("the pose is still there");

		assert_eq!(pose.skeleton, SkeletonId::NONE, "naming nothing");
		assert_eq!(world.render_skinning(id, &mut Vec::new()), 0, "and moving nothing");
	}

	#[test]
	fn cutting_a_character_out_takes_its_pose_with_it() {
		let mut world = character();
		let standing = world
			.entities
			.iter()
			.next()
			.map(|(id, ..)| id)
			.expect("the body entity");
		let body = world
			.bodies
			.spawn(Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0).driving(standing));
		let scene = capture(&world);
		let solid = scene
			.solids
			.iter()
			.position(|it| it.slot == u32::try_from(body.slot()).unwrap_or(0))
			.expect("the body is in the description");
		let piece = scene.subset(&[u32::try_from(solid).unwrap_or(0)]);

		assert_eq!(piece.posed.len(), 1, "the pose came along");
		assert_eq!(piece.posed[0].slot, 0, "renumbered like everything else");
		assert_eq!(piece.things[0].pose, 0, "and the entity still points at it");
		assert_eq!(
			piece.posed[0].locals.len(),
			2,
			"with the bones it was standing in, not with none"
		);
	}

	#[test]
	fn a_world_put_back_captures_the_same_way_it_did_before() {
		let world = peopled();
		let scene = capture(&world);

		let mut empty = furnished();
		let report = restore(&mut empty, &scene).expect("the layouts agree");

		assert_eq!((report.things, report.solids, report.links), (2, 1, 1), "all of it landed");
		assert_eq!(
			capture(&empty),
			scene,
			"and describing it again describes exactly what was read"
		);
	}

	#[test]
	fn a_handle_kept_across_a_restore_still_names_the_same_entity() {
		let world = peopled();
		let held = world
			.entities
			.iter()
			.map(|(id, ..)| id)
			.last()
			.expect("two entities");
		let scene = capture(&world);

		let mut empty = furnished();
		restore(&mut empty, &scene).expect("the layouts agree");

		assert!(empty.entities.alive(held), "the handle resolves in the restored world");
		assert_eq!(
			empty
				.entities
				.transform(held)
				.map(|placed| placed.position),
			Some(Vec3::new(0.0, 4.0, 0.0)),
			"and it is the entity it was, not whoever took the slot"
		);
	}

	#[test]
	fn a_handle_that_was_already_stale_is_still_stale_afterwards() {
		let mut world = World::new();
		let doomed = world.entities.spawn();
		world.entities.spawn();
		world.entities.despawn(doomed);

		let scene = capture(&world);
		let mut empty = World::new();
		restore(&mut empty, &scene).expect("no arena to disagree about");

		assert!(!empty.entities.alive(doomed), "a dead slot comes back dead");

		// and taking the slot again does not resurrect it: the generation the
		// dead slot carried is what the description wrote down.
		let taken = empty.entities.spawn();

		assert_ne!(taken, doomed, "the new occupant is not the old handle");
		assert!(!empty.entities.alive(doomed), "which the old handle can still tell");
	}

	#[test]
	fn the_arena_comes_back_with_the_handles_a_game_kept_in_it() {
		#[repr(C)]
		#[derive(Clone, Copy, crate::bytemuck::Pod, crate::bytemuck::Zeroable)]
		struct Held {
			player: EntityId,
			pad: [u32; 2],
		}

		let mut world = World::new();
		let player = world
			.entities
			.spawn_at(Transform::at(Vec3::new(7.0, 0.0, 0.0)));
		world.state.get::<Held>(3).0.player = player;

		let scene = capture(&world);
		let mut empty = World::new();
		let report = restore(&mut empty, &scene).expect("nobody has claimed the arena");

		assert!(report.arena, "the arena is part of what a restore puts back");

		let (held, fresh) = empty.state.get::<Held>(3);

		assert!(!fresh, "the layout number came back with the bytes");
		assert_eq!(
			empty
				.entities
				.transform(held.player)
				.map(|placed| placed.position),
			Some(Vec3::new(7.0, 0.0, 0.0)),
			"and the handle inside it points at the entity it always did"
		);
	}

	#[test]
	fn a_scene_written_under_another_layout_is_refused_rather_than_forced() {
		#[repr(C)]
		#[derive(Clone, Copy, crate::bytemuck::Pod, crate::bytemuck::Zeroable)]
		struct Held {
			count: u32,
			pad: u32,
		}

		let mut world = World::new();
		world.entities.spawn();
		world.state.get::<Held>(4).0.count = 1;
		let scene = capture(&world);

		let mut newer = World::new();
		newer.state.get::<Held>(9).0.count = 2;
		let refused = restore(&mut newer, &scene);

		assert!(refused.is_err(), "the two builds do not agree about the arena");
		assert!(
			newer.entities.is_empty(),
			"and nothing was loaded, because half a world is worse than none"
		);
		assert_eq!(newer.state.get::<Held>(9).0.count, 2, "the running game keeps its own bytes");
	}

	#[test]
	fn an_unclaimed_arena_on_either_side_is_not_a_disagreement() {
		#[repr(C)]
		#[derive(Clone, Copy, crate::bytemuck::Pod, crate::bytemuck::Zeroable)]
		struct Held {
			count: u32,
			pad: u32,
		}

		let mut world = World::new();
		world.state.get::<Held>(4).0.count = 1;
		let claimed = capture(&world);

		let mut fresh = World::new();

		assert!(
			restore(&mut fresh, &claimed).is_ok(),
			"a world whose game has not claimed the arena takes whatever it is given"
		);

		let unclaimed = capture(&World::new());
		let mut running = World::new();
		running.state.get::<Held>(4).0.count = 5;

		assert!(
			restore(&mut running, &unclaimed).is_ok(),
			"and a description written before any game ran disagrees with nobody"
		);
	}

	#[test]
	fn a_restore_replaces_the_world_rather_than_adding_to_it() {
		let scene = capture(&peopled());
		let mut crowded = furnished();
		for _ in 0..5 {
			crowded.entities.spawn();
		}
		crowded
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY));

		restore(&mut crowded, &scene).expect("the layouts agree");

		assert_eq!(crowded.entities.len(), 2, "what was here is gone");
		assert_eq!(crowded.bodies.len(), 1, "in every table");
	}

	#[test]
	fn assets_are_found_by_name_however_this_world_happens_to_be_ordered() {
		let source = peopled();
		let scene = capture(&source);

		// the same assets behind one that the other world never had, so the
		// crystal lands on a different handle than the description was written
		// from. That is the whole reason a name is what gets written down.
		let mut other = World::new();
		other
			.meshes
			.insert("meshes/decoy", MeshData::default());
		other
			.meshes
			.insert("meshes/crystal", MeshData::default());
		other
			.materials
			.insert("brass", Material::colored(Vec3::ONE));

		assert_ne!(
			other.meshes.find("meshes/crystal"),
			source.meshes.find("meshes/crystal"),
			"the two worlds really do disagree about the handle"
		);

		restore(&mut other, &scene).expect("no arena to disagree about");

		let drawn: Vec<String> = other
			.entities
			.iter()
			.filter_map(|(_, _, renderable)| other.meshes.get(renderable.mesh))
			.map(|entry| entry.name().to_owned())
			.collect();

		assert!(
			drawn.contains(&"meshes/crystal".to_owned()),
			"and the entity draws the mesh it named all the same, got {drawn:?}"
		);
	}

	#[test]
	fn a_name_this_world_does_not_have_falls_back_rather_than_failing() {
		let mut scene = capture(&peopled());
		scene.things[0].mesh = "meshes/gone".to_owned();
		scene.things[0].material = "vanished".to_owned();

		let mut bare = World::new();
		restore(&mut bare, &scene).expect("no arena to disagree about");

		let (_, _, renderable) = bare
			.entities
			.iter()
			.next()
			.expect("it still exists");

		assert_eq!(renderable.mesh, crate::abi::MeshId::NONE, "the missing mesh draws nothing");
		assert_eq!(
			renderable.material,
			MaterialId::DEFAULT,
			"and the missing material is the default one"
		);
	}

	#[test]
	fn a_body_and_a_joint_come_back_holding_the_restored_handles() {
		let scene = capture(&peopled());
		let mut empty = furnished();
		restore(&mut empty, &scene).expect("the layouts agree");

		let (body_id, body) = empty.bodies.iter().next().expect("one body");

		assert!(empty.entities.alive(body.entity), "the body drives a living entity");
		assert_eq!(
			empty
				.entities
				.transform(body.entity)
				.map(|placed| placed.position),
			Some(Vec3::new(0.0, 4.0, 0.0)),
			"and it is the one it drove before"
		);
		assert_eq!(body.layers, Layers::single(2), "its layers come with it");

		let (_, joint) = empty.joints.iter().next().expect("one joint");

		assert_eq!(joint.first, body_id, "the rope holds the restored body");
		assert_eq!(joint.second, BodyId::NONE, "and a point in the world, as it did");
	}

	#[test]
	fn whether_a_joint_lets_its_bodies_collide_is_written_down_and_put_back() {
		let mut world = furnished();
		let first =
			world
				.bodies
				.spawn(Body::dynamic(Shape::UNIT, Transform::at(Vec3::ZERO), 1.0));
		let second = world
			.bodies
			.spawn(Body::dynamic(Shape::UNIT, Transform::at(Vec3::X), 1.0));

		world.join(Joint::weld(first, second, (Vec3::ZERO, Vec3::ZERO)));
		world.join(Joint::weld(first, second, (Vec3::ZERO, Vec3::ZERO)).touching());

		let scene = capture(&world);

		assert_eq!(
			scene
				.links
				.iter()
				.map(|link| link.collide)
				.collect::<Vec<bool>>(),
			vec![false, true],
			"a capture reads it off each joint rather than assuming"
		);

		let mut empty = furnished();
		restore(&mut empty, &scene).expect("the layouts agree");

		assert_eq!(
			empty
				.joints
				.iter()
				.map(|(_, joint)| joint.collide)
				.collect::<Vec<bool>>(),
			vec![false, true],
			"and a restore puts both back the way they were"
		);
	}

	/// A description of `count` bodies, each driving an entity of its own.
	///
	/// No world anywhere: what is being tested is arithmetic over plain data,
	/// and building it by hand is what says so.
	fn parts(count: u32) -> SceneData {
		let mut scene = SceneData::default();

		for index in 0..count {
			scene.things.push(Thing {
				name: format!("thing {index}"),
				slot: index,
				generation: 1,
				transform: Transform::at(Vec3::X * f32::from(u16::try_from(index).unwrap_or(0))),
				..Thing::default()
			});
			scene.solids.push(Solid {
				name: format!("part {index}"),
				slot: index,
				generation: 1,
				kind: BodyKind::Dynamic,
				thing: index,
				..Solid::default()
			});
		}

		let count = usize::try_from(count).unwrap_or(0);
		scene.thing_generations = vec![1; count];
		scene.solid_generations = vec![1; count];

		scene
	}

	/// A joint between two of them, or between one and the world.
	fn holding(first: u32, second: u32) -> Link {
		Link {
			name: format!("held {first} to {second}"),
			kind: JointKind::Weld,
			first,
			second,
			..Link::default()
		}
	}

	#[test]
	fn a_body_nothing_holds_is_a_piece_on_its_own() {
		let scene = parts(3);

		assert_eq!(scene.connected(1), vec![1], "itself and nothing else");
	}

	#[test]
	fn a_chain_of_joints_is_one_piece_from_either_end() {
		let mut scene = parts(4);
		scene.links.push(holding(0, 1));
		scene.links.push(holding(1, 2));
		scene.link_generations = vec![1; 2];

		assert_eq!(scene.connected(0), vec![0, 1, 2], "from the near end");
		assert_eq!(scene.connected(2), vec![0, 1, 2], "and from the far one");
		assert_eq!(scene.connected(3), vec![3], "while the loose one stays loose");
	}

	#[test]
	fn a_joint_pinned_to_the_world_holds_nothing_to_anything() {
		// two props bolted to the same wall are two pieces, not one. The wall
		// is not a body and cannot be a member of anything.
		let mut scene = parts(2);
		scene.links.push(holding(0, NO_INDEX));
		scene.links.push(holding(1, NO_INDEX));
		scene.link_generations = vec![1; 2];

		assert_eq!(scene.connected(0), vec![0], "each is its own piece");
		assert_eq!(scene.connected(1), vec![1], "however many are bolted beside it");
	}

	#[test]
	fn the_far_end_of_a_joint_is_a_body_or_it_is_nothing() {
		// `across` on its own, because in `connected` the bound check catches
		// the world index too - `NO_INDEX` is the largest a `u32` gets - and
		// two guards that cover each other are two guards neither of which is
		// proved. This one is about what a joint *means*; the bound below is
		// about a description somebody wrote wrong.
		assert_eq!(across(&holding(0, 1), 0), Some(1), "from the near end");
		assert_eq!(across(&holding(0, 1), 1), Some(0), "and from the far one");
		assert_eq!(across(&holding(0, 1), 2), None, "a joint that does not name it at all");
		assert_eq!(across(&holding(0, NO_INDEX), 0), None, "and the world is not a body");
		assert_eq!(across(&holding(NO_INDEX, 1), 1), None, "whichever end it is written on");
	}

	#[test]
	fn a_joint_naming_a_body_that_is_not_there_reaches_nothing() {
		// the other half of the pair above: a description somebody wrote by
		// hand, or one whose bodies were cut away under it.
		let mut scene = parts(2);
		scene.links.push(holding(0, 7));
		scene.link_generations = vec![1; 1];

		assert_eq!(scene.connected(0), vec![0], "an index past the end is not a member");
	}

	#[test]
	fn two_bodies_driving_one_entity_bring_one_entity() {
		// legal, and what a prop with a second collider is. The copy has to be
		// made once and pointed at twice, or the piece has two of everything it
		// draws.
		let mut scene = parts(2);
		scene.solids[1].thing = 0;

		let piece = scene.subset(&[0, 1]);

		assert_eq!(piece.solids.len(), 2, "both bodies");
		assert_eq!(piece.things.len(), 1, "and one entity between them");
		assert_eq!(
			piece
				.solids
				.iter()
				.map(|it| it.thing)
				.collect::<Vec<_>>(),
			vec![0, 0],
			"which both of them drive"
		);
	}

	#[test]
	fn a_seed_that_is_not_an_index_names_no_piece_at_all() {
		let scene = parts(2);

		assert!(scene.connected(9).is_empty(), "past the end");
		assert!(SceneData::default().connected(0).is_empty(), "and in an empty description");
	}

	#[test]
	fn a_piece_cut_out_carries_what_it_drives_and_what_holds_it() {
		let mut scene = parts(4);
		scene.links.push(holding(1, 2));
		// and one pinned to the world, which joins nothing to anything but is
		// still a property of the body it holds
		scene.links.push(holding(1, NO_INDEX));
		scene.link_generations = vec![1; 2];

		let piece = scene.subset(&scene.connected(1));

		assert_eq!(piece.solids.len(), 2, "the two that are held together");
		assert_eq!(piece.things.len(), 2, "and the entity each of them drives");
		assert_eq!(
			piece.solids[0].name, "part 1",
			"named as they were, because a name is not an index"
		);
		assert_eq!(
			piece.links.len(),
			2,
			"the joint inside the piece and the one pinning it to the world"
		);
		assert!(
			piece
				.links
				.iter()
				.all(|link| link.first != NO_INDEX && link.first < 2),
			"and both are renumbered into the piece's own list"
		);
	}

	#[test]
	fn a_joint_whose_far_end_was_left_behind_is_left_behind_with_it() {
		let mut scene = parts(3);
		scene.links.push(holding(0, 2));
		scene.link_generations = vec![1; 1];

		// asked for a body whose joint reaches something not in the list. Half
		// a joint holds nothing, and a paste would refuse it anyway.
		let piece = scene.subset(&[0]);

		assert_eq!(piece.solids.len(), 1, "the one that was asked for");
		assert!(piece.links.is_empty(), "and no joint reaching out of it");
	}

	#[test]
	fn a_piece_is_renumbered_into_a_description_of_its_own() {
		let scene = parts(5);
		let piece = scene.subset(&[3, 1]);

		assert_eq!(piece.solids.len(), 2, "two of the five");
		assert_eq!(
			piece
				.solids
				.iter()
				.map(|it| it.slot)
				.collect::<Vec<_>>(),
			vec![0, 1],
			"in slots of their own, in the order they were asked for"
		);
		assert_eq!(
			piece
				.solids
				.iter()
				.map(|it| it.thing)
				.collect::<Vec<_>>(),
			vec![0, 1],
			"each pointing at its own entity's new place"
		);
		assert_eq!(
			piece
				.things
				.iter()
				.map(|it| it.name.clone())
				.collect::<Vec<_>>(),
			vec!["thing 3".to_owned(), "thing 1".to_owned()],
			"and the entities are the ones those bodies drove"
		);
		assert_eq!(piece.solid_generations, vec![1, 1], "with a generation each");
	}

	#[test]
	fn a_piece_keeps_the_world_it_came_out_of_and_not_the_game_that_was_playing() {
		let mut scene = parts(2);
		scene.stage.gravity = Vec3::new(0.0, -3.0, 0.0);
		scene.arena = Some(Arena { layout: 7, bytes: vec![9; STATE_BYTES] });

		let piece = scene.subset(&[0]);

		assert_eq!(
			piece.stage.gravity,
			Vec3::new(0.0, -3.0, 0.0),
			"a piece settled under one gravity was settled under it"
		);
		assert!(
			piece.arena.is_none(),
			"and the game's own bytes name handles that are not in here"
		);
	}

	#[test]
	fn an_index_asked_for_twice_or_out_of_range_is_ignored() {
		let scene = parts(3);

		assert_eq!(scene.subset(&[1, 1, 1]).solids.len(), 1, "named three times, kept once");
		assert_eq!(scene.subset(&[0, 44]).solids.len(), 1, "and one of the two exists");
		assert!(scene.subset(&[]).is_empty(), "nothing asked for is nothing cut out");
	}

	#[test]
	fn a_piece_pasted_back_is_the_piece_and_nothing_else() {
		// the whole point, and the only test here with a world in it: what
		// comes out of `subset` is a `SceneData` like any other, so the loader
		// that already exists puts it back with no new code anywhere.
		let mut world = peopled();
		let whole = capture(&world);
		let held = whole
			.solids
			.iter()
			.position(|solid| solid.thing != NO_INDEX)
			.expect("a body driving something");
		let piece = whole.subset(&whole.connected(u32::try_from(held).unwrap_or(0)));

		let before = world.bodies.len();
		let put = instantiate(&mut world, &piece, Vec3::Y * 10.0);

		assert_eq!(
			world.bodies.len(),
			before + piece.solids.len(),
			"exactly the piece arrived beside what was already there"
		);
		assert!(put.body(0).is_some(), "with handles of its own");
	}

	#[test]
	fn a_body_gravity_does_not_reach_is_still_out_of_its_reach_afterwards() {
		// the field crosses this boundary three times - once on the way out and
		// once in each loader - and the file's own round trip covers none of
		// them: that one starts and ends at a `SceneData`.
		let mut world = World::new();
		let entity = world.entities.spawn();
		let second = world.entities.spawn();
		let floating = world.attach_body(entity, BodyKind::Dynamic, Shape::UNIT);
		let falling = world.attach_body(second, BodyKind::Dynamic, Shape::UNIT);

		if let Some(body) = world.bodies.get_mut(floating) {
			body.weightless = true;
		}

		let scene = capture(&world);
		assert!(scene.solids[0].weightless, "the description says which one floats");
		assert!(!scene.solids[1].weightless, "and which one does not");

		let mut put_back = World::new();
		restore(&mut put_back, &scene).expect("no arena to disagree about");

		assert!(
			put_back
				.bodies
				.get(floating)
				.is_some_and(|body| body.weightless),
			"a restore hands the same body back still weightless"
		);
		assert!(
			put_back
				.bodies
				.get(falling)
				.is_some_and(|body| !body.weightless),
			"and the other one still with its weight"
		);

		let mut beside = World::new();
		let put = instantiate(&mut beside, &scene, Vec3::ZERO);

		assert!(
			beside
				.bodies
				.get(put.body(0))
				.is_some_and(|body| body.weightless),
			"and so does an instantiate, which shares none of the same handles"
		);
	}

	#[test]
	fn the_free_list_comes_back_in_slot_order_rather_than_in_the_order_things_died() {
		let mut world = World::new();
		for _ in 0..4 {
			world.entities.spawn();
		}

		let holes: Vec<_> = world.entities.iter().map(|(id, ..)| id).collect();
		world.entities.despawn(holes[2]);
		world.entities.despawn(holes[0]);

		let scene = capture(&world);
		let mut empty = World::new();
		restore(&mut empty, &scene).expect("no arena to disagree about");

		// the free list is a stack, so the *last* slot pushed is the first one
		// taken - which after a restore is the highest empty one rather than
		// whichever died most recently. The number matters less than that two
		// hosts reading one description derive the same one.
		assert_eq!(empty.entities.spawn().slot(), 2, "the higher hole is filled first");
		assert_eq!(empty.entities.spawn().slot(), 0, "and then the lower one");
	}

	#[test]
	fn nothing_is_drawn_traveling_out_of_the_world_that_was_here_before() {
		let scene = capture(&peopled());
		let mut elsewhere = furnished();
		elsewhere
			.entities
			.spawn_at(Transform::at(Vec3::splat(-40.0)));

		restore(&mut elsewhere, &scene).expect("the layouts agree");

		// zero is the start of a step, where the renderer draws each entity's
		// *past*. A restore that wrote only the present would draw everything
		// at the origin for one frame and then have it snap into place.
		elsewhere.set_interpolation(0.0);

		let mut seen = 0;
		for (id, placed, _) in elsewhere.entities.iter() {
			let drawn = elsewhere
				.render_transform(id)
				.expect("it is alive");

			assert!(
				drawn.position.abs_diff_eq(placed.position, 1.0e-6),
				"a restored entity is drawn where it is at every point of a step, got {} for 				 {}",
				drawn.position,
				placed.position
			);

			seen += 1;
		}

		assert_eq!(seen, 2, "and both of them were looked at");
	}

	#[test]
	fn a_slot_past_what_the_table_can_hold_is_dropped_rather_than_panicking() {
		let mut scene = capture(&peopled());
		scene.things[0].slot = u32::try_from(MAX_ENTITIES).unwrap() + 4;
		scene.thing_generations = vec![1; 2];

		let mut empty = furnished();
		let report = restore(&mut empty, &scene).expect("the layouts agree");

		assert_eq!(report.things, 1, "one of the two would not fit");
		assert_eq!(empty.entities.len(), 1, "and the table holds exactly what landed");
	}
	#[test]
	fn a_material_nobody_named_is_the_default_one() {
		// the hand-authored case: a capture always writes a name, because the
		// default material is a real entry called "default", but a scene
		// somebody typed may leave it out.
		let mut scene = SceneData::default();
		scene.things.push(Thing {
			slot: 0,
			generation: 1,
			transform: Transform::IDENTITY,
			color: Vec3::ONE,
			..Thing::default()
		});

		let mut world = World::new();
		restore(&mut world, &scene).expect("no arena at all");

		let (_, _, renderable) = world.entities.iter().next().expect("it landed");

		assert_eq!(
			renderable.material,
			MaterialId::DEFAULT,
			"an unnamed material is the default one and not the null one"
		);
	}

	#[test]
	fn a_restore_empties_the_queues_that_describe_the_world_it_replaced() {
		let scene = capture(&peopled());
		let mut busy = furnished();
		let one =
			busy.bodies
				.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY));
		let two = busy
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY).sensing());
		busy.bodies.touched(crate::abi::Touch {
			first: one,
			second: two,
			kind: crate::abi::TouchKind::Began,
			point: Vec3::ZERO,
			normal: Vec3::Y,
		});
		busy.bodies
			.overlapped(crate::abi::Overlap { sensor: two, body: one });

		restore(&mut busy, &scene).expect("the layouts agree");

		assert!(busy.bodies.touches().is_empty(), "a landing in another world is not news here");
		assert!(
			busy.bodies.overlaps().is_empty(),
			"and neither is what was inside a trigger that no longer exists"
		);
	}

	#[test]
	fn a_restore_stops_whatever_was_playing() {
		// a description carries no voices, so the arena it puts back holds
		// handles into a table nobody restored. Leaving a sound running would
		// be a noise from the world before the load that nothing left in the
		// world after it can name or stop.
		let scene = capture(&peopled());
		let mut busy = furnished();
		let one = busy
			.audio
			.play(crate::abi::Voice::flat(crate::abi::SoundId::NONE).looping());
		let two = busy
			.audio
			.play(crate::abi::Voice::flat(crate::abi::SoundId::NONE));

		assert_eq!(busy.audio.len(), 2, "two things were playing");

		restore(&mut busy, &scene).expect("the layouts agree");

		assert!(busy.audio.is_empty(), "and now nothing is");
		assert!(!busy.audio.alive(one), "not the loop");
		assert!(!busy.audio.alive(two), "and not the one-shot either");
	}

	#[test]
	fn an_instantiate_leaves_what_was_playing_alone() {
		// the other half of the rule, and the reason it is not symmetric: a
		// paste puts a prop down beside what is already there, and the arena
		// it does not touch still holds every handle it held a moment ago.
		let scene = capture(&peopled());
		let mut busy = furnished();
		let id = busy
			.audio
			.play(crate::abi::Voice::flat(crate::abi::SoundId::NONE).looping());

		instantiate(&mut busy, &scene, Vec3::X * 3.0);

		assert!(busy.audio.alive(id), "the ambience is still the same ambience");
	}

	#[test]
	fn two_records_claiming_one_slot_do_not_both_get_it() {
		let mut scene = SceneData {
			thing_generations: vec![1],
			..SceneData::default()
		};
		for _ in 0..2 {
			scene.things.push(Thing {
				slot: 0,
				generation: 1,
				transform: Transform::IDENTITY,
				..Thing::default()
			});
		}

		let mut world = World::new();
		let report = restore(&mut world, &scene).expect("no arena at all");

		assert_eq!(report.things, 1, "the first one keeps the slot");
		assert_eq!(
			world.entities.len(),
			1,
			"because the alternative is one handle handed out twice"
		);
	}

	#[test]
	fn a_restore_forgets_the_holes_the_world_it_replaced_had() {
		let mut scene = SceneData {
			thing_generations: vec![1, 1, 1],
			..SceneData::default()
		};
		for slot in 0..3 {
			scene.things.push(Thing {
				slot,
				generation: 1,
				transform: Transform::IDENTITY,
				..Thing::default()
			});
		}

		// a world whose own table is full of holes, so a free list that
		// survived the restore would hand out a slot something already stands
		// in.
		let mut holed = World::new();
		let doomed = [
			holed.entities.spawn(),
			holed.entities.spawn(),
			holed.entities.spawn(),
			holed.entities.spawn(),
		];
		for id in doomed {
			holed.entities.despawn(id);
		}

		restore(&mut holed, &scene).expect("no arena at all");

		assert_eq!(holed.entities.len(), 3, "three entities landed");

		let next = holed.entities.spawn();

		assert_eq!(next.slot(), 3, "and the next slot is past them rather than under one");
		assert_eq!(holed.entities.len(), 4, "so nothing was overwritten");
	}

	#[test]
	fn a_record_that_carries_no_generation_at_all_still_lands_usable() {
		// the hand-written case again: nobody typing a scene writes a
		// generation, and zero is what a handle to nothing carries, so a slot
		// left holding it would hand out a handle that refers to nothing.
		let mut scene = SceneData::default();
		scene.things.push(Thing {
			slot: 0,
			generation: 0,
			transform: Transform::at(Vec3::Z),
			..Thing::default()
		});

		let mut world = World::new();
		restore(&mut world, &scene).expect("no arena at all");

		let (id, ..) = world.entities.iter().next().expect("it landed");

		assert!(id.is_some(), "the handle refers to something");
		assert!(world.entities.alive(id), "and the table agrees that it does");
	}

	#[test]
	fn a_written_generation_is_what_the_slot_comes_back_on() {
		// a description with no generation array at all - which is what an
		// authored scene is - so the records are the only place the answer can
		// come from.
		let mut scene = SceneData::default();
		scene.things.push(Thing {
			slot: 0,
			generation: 7,
			transform: Transform::IDENTITY,
			..Thing::default()
		});

		let mut world = World::new();
		restore(&mut world, &scene).expect("no arena at all");

		assert_eq!(
			world.entities.generation(0),
			7,
			"the slot is on the occupant the description named, not the first one"
		);

		let (id, ..) = world.entities.iter().next().expect("it landed");

		assert_eq!(id.generation(), 7, "and the handle handed out says so");
	}

	#[test]
	fn a_mesh_shaped_body_finds_its_geometry_by_name_too() {
		let mut source = furnished();
		let crystal = source.meshes.find("meshes/crystal");
		source.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::mesh(crystal),
			Transform::IDENTITY,
		));
		let scene = capture(&source);

		// the same mesh behind one this world has and that one did not, so the
		// handle differs.
		let mut other = World::new();
		other
			.meshes
			.insert("meshes/decoy", MeshData::default());
		other
			.meshes
			.insert("meshes/crystal", MeshData::default());

		assert_ne!(other.meshes.find("meshes/crystal"), crystal, "the handles really differ");

		restore(&mut other, &scene).expect("no arena to disagree about");

		let (_, body) = other.bodies.iter().next().expect("one body");

		assert_eq!(body.shape.kind, ShapeKind::Mesh, "it is still a mesh shape");
		assert_eq!(
			body.shape.mesh,
			other.meshes.find("meshes/crystal"),
			"and it collides against the geometry it named"
		);
	}
	#[test]
	fn instantiating_adds_to_a_world_rather_than_replacing_it() {
		let scene = capture(&peopled());
		let mut busy = furnished();
		let standing = busy
			.entities
			.spawn_at(Transform::at(Vec3::splat(-5.0)));

		let put = instantiate(&mut busy, &scene, Vec3::ZERO);

		assert_eq!(busy.entities.len(), 3, "one that was here and two that arrived");
		assert!(busy.entities.alive(standing), "what was here is untouched");
		assert_eq!(put.entities().count(), 2, "and the remap says what arrived");
	}

	#[test]
	fn every_handle_a_copy_gets_is_a_new_one() {
		// into the world it came from, which is the only place the question
		// means anything: a handle is an index into one world's tables, so two
		// worlds that both started empty hand out the same numbers and
		// comparing across them proves nothing at all.
		let mut world = peopled();
		let scene = capture(&world);
		let held: Vec<EntityId> = world.entities.iter().map(|(id, ..)| id).collect();

		let first = instantiate(&mut world, &scene, Vec3::X * 20.0);
		let again = instantiate(&mut world, &scene, Vec3::X * 40.0);

		for id in held {
			assert!(
				!first.entities().any(|it| it == id),
				"a copy is not the thing it was copied from"
			);
		}

		let overlap = first
			.entities()
			.any(|one| again.entities().any(|two| one == two));

		assert!(!overlap, "and two copies are not each other");
		assert_eq!(world.entities.len(), 6, "the originals and both copies");
	}

	#[test]
	fn a_copy_lands_where_it_was_asked_to() {
		let scene = capture(&peopled());
		let mut world = furnished();
		let put = instantiate(&mut world, &scene, Vec3::new(0.0, 10.0, 0.0));

		let moved = put.entity(0);
		let placed = world
			.entities
			.transform(moved)
			.expect("it landed")
			.position;

		assert_eq!(
			placed,
			scene.things[0].transform.position + Vec3::new(0.0, 10.0, 0.0),
			"every position is offset by what the paste asked for"
		);

		let body = put.body(0);
		let solid = world.bodies.get(body).expect("it landed");

		assert_eq!(
			solid.transform.position,
			scene.solids[0].transform.position + Vec3::new(0.0, 10.0, 0.0),
			"the bodies too, or a copy would be standing beside its own collision"
		);
	}

	#[test]
	fn a_body_points_at_the_entity_that_arrived_with_it() {
		let scene = capture(&peopled());
		let mut world = furnished();
		// something already in the table, so the indices in the file are not
		// the slots the copies land in and a lookup that used them would be
		// pointing at whoever was here.
		world.entities.spawn();

		let put = instantiate(&mut world, &scene, Vec3::ZERO);
		let body = world
			.bodies
			.get(put.body(0))
			.expect("the body landed");

		assert_eq!(
			body.entity,
			put.entity(scene.solids[0].thing),
			"the body drives the copy of the entity it drove"
		);
		assert!(world.entities.alive(body.entity), "which is a living one");
	}

	#[test]
	fn a_joint_holds_the_copies_rather_than_the_originals() {
		let mut world = peopled();
		let scene = capture(&world);
		let original = world
			.bodies
			.iter()
			.map(|(id, _)| id)
			.next()
			.expect("one body");

		let put = instantiate(&mut world, &scene, Vec3::X * 20.0);
		let joint = world
			.joints
			.get(put.joint(0))
			.expect("the joint landed");

		assert_eq!(joint.first, put.body(0), "it holds the copy");
		assert_ne!(joint.first, original, "and not the body it was copied from");
		assert_eq!(world.joints.len(), 2, "the original rope is still there beside it");
	}

	#[test]
	fn a_joint_pinned_to_the_world_takes_its_anchor_with_it() {
		let scene = capture(&peopled());
		let mut world = furnished();
		let put = instantiate(&mut world, &scene, Vec3::new(3.0, 0.0, 0.0));
		let joint = world
			.joints
			.get(put.joint(0))
			.expect("the joint landed");

		assert_eq!(joint.second, BodyId::NONE, "it is pinned to a point rather than a body");
		assert_eq!(
			joint.second_anchor,
			scene.links[0].second_anchor + Vec3::new(3.0, 0.0, 0.0),
			"and the point moved with the copy, or the rope would reach back to the original"
		);
	}

	#[test]
	fn an_anchor_on_a_body_is_left_alone() {
		// a local anchor is in its body's own space, so moving the copy moves
		// it already. Adding the offset again would put it a paste away from
		// the thing it is attached to.
		let scene = SceneData {
			solids: vec![lone_solid(), lone_solid()],
			links: vec![Link {
				kind: JointKind::Weld,
				first: 0,
				second: 1,
				first_anchor: Vec3::Y,
				second_anchor: -Vec3::Y,
				rest: Quat::IDENTITY,
				..Link::default()
			}],
			..SceneData::default()
		};

		let mut world = World::new();
		let put = instantiate(&mut world, &scene, Vec3::splat(50.0));
		let joint = world
			.joints
			.get(put.joint(0))
			.expect("the joint landed");

		assert_eq!(joint.first_anchor, Vec3::Y, "the first anchor is untouched");
		assert_eq!(joint.second_anchor, -Vec3::Y, "and so is the second, being a body's own");
	}

	#[test]
	fn a_joint_whose_body_did_not_arrive_is_not_created_at_all() {
		// a second body of nothing means "pinned to a point in the world", so a
		// joint that lost its second body and was created anyway would be
		// indistinguishable from one that never had it.
		let scene = SceneData {
			solids: vec![lone_solid()],
			links: vec![Link {
				kind: JointKind::Rope,
				first: 0,
				second: 7,
				length: 1.0,
				..Link::default()
			}],
			..SceneData::default()
		};

		let mut world = World::new();
		let put = instantiate(&mut world, &scene, Vec3::ZERO);

		assert!(!put.joint(0).is_some(), "the joint was refused");
		assert!(world.joints.is_empty(), "and nothing was created");
	}

	#[test]
	fn a_copy_arrives_awake_whatever_the_description_says() {
		let scene = SceneData {
			solids: vec![Solid { sleeping: true, ..lone_solid() }],
			..SceneData::default()
		};

		let mut world = World::new();
		let put = instantiate(&mut world, &scene, Vec3::ZERO);
		let body = world.bodies.get(put.body(0)).expect("it landed");

		assert!(
			!body.sleeping,
			"a body created a moment ago has not been still for any length of time"
		);
	}

	#[test]
	fn instantiating_leaves_the_arena_and_the_settings_alone() {
		#[repr(C)]
		#[derive(Clone, Copy, crate::bytemuck::Pod, crate::bytemuck::Zeroable)]
		struct Held {
			count: u32,
			pad: u32,
		}

		let mut source = peopled();
		source.state.get::<Held>(5).0.count = 9;
		let scene = capture(&source);

		let mut running = furnished();
		running.state.get::<Held>(5).0.count = 1;
		running.camera.position = Vec3::splat(-3.0);
		running.gravity = Vec3::ZERO;

		instantiate(&mut running, &scene, Vec3::ZERO);

		assert_eq!(running.state.get::<Held>(5).0.count, 1, "the game's own bytes are its own");
		assert_eq!(running.camera.position, Vec3::splat(-3.0), "pasting moves nobody's camera");
		assert_eq!(running.gravity, Vec3::ZERO, "and changes nothing about the world");
	}

	#[test]
	fn a_remap_answers_by_name_as_well_as_by_place() {
		let scene = SceneData {
			things: vec![
				Thing {
					name: "floor".to_owned(),
					..Thing::default()
				},
				Thing { name: String::new(), ..Thing::default() },
			],
			..SceneData::default()
		};

		let mut world = World::new();
		let put = instantiate(&mut world, &scene, Vec3::ZERO);

		assert_eq!(put.entity_named("floor"), put.entity(0), "a name finds what it named");
		assert!(!put.entity_named("nothing").is_some(), "and a name nobody used finds nothing");
		assert!(
			!put.entity_named("").is_some(),
			"the empty name is not a name, it is the absence of one"
		);
	}

	/// A body with nothing attached, for a description written by hand.
	fn lone_solid() -> Solid {
		Solid {
			kind: BodyKind::Dynamic,
			shape: Form {
				kind: ShapeKind::Box,
				extents: Vec3::splat(0.5),
				..Form::default()
			},
			mass: 1.0,
			layers: Layers::DEFAULT,
			thing: NO_INDEX,
			..Solid::default()
		}
	}

	#[test]
	fn what_things_are_called_is_written_down_and_put_back() {
		let mut world = peopled();
		let entity = world
			.entities
			.iter()
			.next()
			.map(|(id, ..)| id)
			.unwrap_or_default();
		let body = world
			.bodies
			.iter()
			.next()
			.map(|(id, _)| id)
			.unwrap_or_default();
		let joint = world
			.joints
			.iter()
			.next()
			.map(|(id, _)| id)
			.unwrap_or_default();

		world.entities.set_name(entity, "crystal");
		world.bodies.set_name(body, "ball");
		world.joints.set_name(joint, "rope");

		let scene = capture(&world);

		assert!(
			scene
				.things
				.iter()
				.any(|thing| thing.name == "crystal"),
			"a capture writes the name the world was holding"
		);
		assert!(
			scene
				.solids
				.iter()
				.any(|solid| solid.name == "ball"),
			"for a body too"
		);
		assert!(scene.links.iter().any(|link| link.name == "rope"), "and for a joint");

		// somebody else's world entirely, so the names cannot survive by
		// having been left behind in the tables.
		let mut other = furnished();
		restore(&mut other, &scene).expect("an unclaimed arena agrees with anything");

		assert_eq!(other.entities.name(entity), "crystal", "and a restore puts them all back");
		assert_eq!(other.bodies.name(body), "ball", "on the same handles they were on");
		assert_eq!(other.joints.name(joint), "rope", "in all three tables");
	}

	#[test]
	fn a_restore_forgets_a_name_the_world_it_replaced_was_holding() {
		let mut world = World::new();
		let doomed = world.entities.spawn_at(Transform::IDENTITY);
		world.entities.set_name(doomed, "leftover");

		let scene = SceneData {
			things: vec![Thing {
				name: String::new(),
				slot: doomed.slot().try_into().unwrap_or(0),
				generation: doomed.generation(),
				..Thing::default()
			}],
			..SceneData::default()
		};

		restore(&mut world, &scene).expect("nothing to disagree about");

		assert!(world.entities.alive(doomed), "the slot came back on the same generation");
		assert_eq!(
			world.entities.name(doomed),
			"",
			"and a description that calls it nothing is what it is now called"
		);
	}

	#[test]
	fn a_copy_is_called_what_the_original_was_called() {
		let mut world = peopled();
		let entity = world
			.entities
			.iter()
			.next()
			.map(|(id, ..)| id)
			.unwrap_or_default();
		let body = world
			.bodies
			.iter()
			.next()
			.map(|(id, _)| id)
			.unwrap_or_default();
		let joint = world
			.joints
			.iter()
			.next()
			.map(|(id, _)| id)
			.unwrap_or_default();

		world.entities.set_name(entity, "crystal");
		world.bodies.set_name(body, "ball");
		world.joints.set_name(joint, "rope");

		let scene = capture(&world);
		// back into the world it came from, which is the only place a handle
		// means anything: two empty worlds would both hand out slot zero and
		// the comparison below would pass with this completely broken.
		let put = instantiate(&mut world, &scene, Vec3::ZERO);

		let copied = put.entity_named("crystal");
		let copied_body = put.body_named("ball");
		let copied_joint = put.joint_named("rope");

		assert!(copied.is_some(), "the copy answers to the name the original had");
		assert_ne!(copied, entity, "and is not the original");
		assert_eq!(world.entities.name(copied), "crystal", "and carries the name in the world");

		assert!(copied_body.is_some(), "a body is copied by name as well");
		assert_ne!(copied_body, body, "and is its own body");
		assert_eq!(world.bodies.name(copied_body), "ball");

		assert!(copied_joint.is_some(), "and a joint");
		assert_ne!(copied_joint, joint, "which is its own joint");
		assert_eq!(world.joints.name(copied_joint), "rope");
	}
}
