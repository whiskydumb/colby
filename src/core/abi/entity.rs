//! Entities, addressed by handle.
//!
//! The storage lives in the host and is reached only through [`EntityId`], a
//! generational index. Nothing hands out a reference to an entity: the game
//! asks the world for a component by handle, uses it, and asks again next
//! frame. A handle to a despawned entity fails that lookup instead of aliasing
//! whatever took its slot.
//!
//! Capacity is fixed. That is not only about keeping `#[repr(C)]` honest - the
//! gameplay crate here is code that is *expected* to be wrong sometimes, and a
//! reload that spawns a ring of entities every time should run out of slots
//! rather than out of memory.
//!
//! Every slot carries two transforms: where the entity is, and where it was at
//! the previous simulation step. The second one is the host's - written by
//! [`Entities::advance`], read by the renderer through
//! [`interpolated`](Entities::interpolated), and invisible to the game, which
//! goes on writing one transform per step and knowing nothing about the rate
//! the picture is drawn at.
//!
//! An entity may hang off another - @ref [`Entities::parent`] - and its
//! transform is then its own place inside that parent rather than in the
//! world. The world's answer is [`Entities::placed`], worked out by walking up
//! the chain when it is asked for rather than kept in a second array, and the
//! renderer's is [`blended`](Entities::blended), which does the same between
//! two steps.
//!
//! An entity may be hidden, @ref [`Entities::set_hidden`], and everything
//! hanging off it is hidden with it. Whether a thing is drawn is worked out by
//! the same walk up the chain, [`Entities::shown`], and only a picture asks:
//! a hidden entity is still stepped, collided with and heard.
//!
//! An entity may refuse decals, @ref [`Entities::set_takes_decals`]: whatever
//! a decal throws lands on every surface inside its box except those of an
//! entity that said so. Unlike being hidden this is the entity's own word and
//! nothing hanging off it inherits it, because what it is about is a surface,
//! and a child's surface is its own.

use super::{
	decal::Decal,
	field::{Field, Value, field},
	light::Light,
	material::MaterialId,
	mesh::MeshId,
	names::Names,
	particles::Emitter,
	pose::PoseId,
	record::{Declared, Noted, Record, Records, Refused},
	strew::Mask,
	terrain::Terrain,
};
use crate::{
	Result,
	bytemuck::{Pod, Zeroable},
	glam::{Mat4, Quat, Vec3},
};

/// How many entities can exist at once.
///
/// Raising this is one constant, and a restart: it changes the layout of
/// `colby_core`.
pub const MAX_ENTITIES: usize = 1024;

/// A handle to an entity.
///
/// The generation makes a stale handle detectable: reusing a slot bumps it, so
/// a handle kept across a despawn no longer matches. Zero is never a live
/// generation, which is what makes a zeroed handle mean [`EntityId::NONE`] -
/// and what makes a freshly zeroed game-state arena hold nothing but nulls.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct EntityId {
	index: u32,
	generation: u32,
}

impl EntityId {
	/// A handle that refers to nothing, and always will.
	pub const NONE: Self = Self { index: 0, generation: 0 };

	/// Whether this handle could refer to anything at all.
	///
	/// A `true` here does not mean the entity is alive - only
	/// [`Entities::alive`] answers that.
	#[must_use]
	pub const fn is_some(self) -> bool { self.generation != 0 }

	/// The slot this addresses, whatever lives there now.
	///
	/// Paired with [`generation`](Self::generation), and the pair is what
	/// writing a handle down and reading it back needs: a scene records both
	/// and hands them to [`Entities::restore`], which puts every entity in the
	/// slot it was in.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "u32 to usize is lossless on every target this builds for, and try_from is not 		          available in a const fn"
	)]
	pub const fn slot(self) -> usize { self.index as usize }

	/// The handle for one slot, whatever is in it.
	///
	/// **Public, and it is not a way to be handed something.** This looks like
	/// the minting the table deliberately keeps to itself, and it is not one:
	/// [`EntityId`] is [`Pod`], so any crate in the workspace can already turn
	/// eight bytes into one of these and always could. What this adds is a
	/// *spelling* for it, which is what anything carrying a handle outside Rust
	/// needs - a script holds one as a number and has to be able to hand it
	/// back. What it does not add is a way to reach anything, because a handle
	/// resolves only where the table agrees somebody is there. @ref
	/// [`Entities::alive`]. The same argument
	/// [`PeerId::from_bits`](super::net::PeerId::from_bits) is written down
	/// with.
	///
	/// @param index - the slot
	/// @param generation - which occupant of it
	#[must_use]
	pub const fn at(index: u32, generation: u32) -> Self { Self { index, generation } }

	/// Which occupant of that slot this handle names.
	#[must_use]
	pub const fn generation(self) -> u32 { self.generation }
}

impl Default for EntityId {
	fn default() -> Self { Self::NONE }
}

/// Where an entity is, how it is turned, and how big it is.
///
/// Three dimensional from the start even though nothing drew in three
/// dimensions until recently, because the alternative was rewriting every call
/// site the day something did.
///
/// @note: not `Pod`, and not `#[repr(C)]`. glam's `Quat` is sixteen-byte
/// aligned under SSE2, so this struct has padding - which is fine, because it
/// never crosses as raw bytes. It is Rust data reached through `colby_core`,
/// which host and module share.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
	/// Position: in the world for an entity standing on its own, inside the
	/// parent for one that hangs off another. @ref [`Entities::placed`].
	pub position: Vec3,

	/// Rotation.
	pub rotation: Quat,

	/// Scale along each axis.
	pub scale: Vec3,
}

impl Transform {
	/// Its fields, for an inspector, a reader and a writer. @ref
	/// [`field`](super::field).
	pub const FIELDS: &[Field<Self>] = &[
		field!(
			Vec3,
			"position",
			position,
			"where it is: in the world on its own, inside its parent when it hangs off one"
		),
		field!(Quat, "rotation", rotation, "which way it is turned"),
		field!(Vec3, "scale", scale, "how big it is along each axis"),
	];
	/// At the origin, unrotated, unscaled.
	pub const IDENTITY: Self = Self {
		position: Vec3::ZERO,
		rotation: Quat::IDENTITY,
		scale: Vec3::ONE,
	};

	/// A transform at a position, with everything else left alone.
	#[must_use]
	pub const fn at(position: Vec3) -> Self { Self { position, ..Self::IDENTITY } }

	/// The transform a model matrix stands for.
	///
	/// The other way round from [`matrix`](Self::matrix), and the pair is only
	/// exact for a matrix that really is a translation, a rotation and a
	/// scale. **A chain of those is not always one of them**: a non-uniform
	/// scale with a rotation under it shears, and a shear has no rotation to
	/// recover, so what comes back is the nearest thing that is not a shear.
	/// Every skeleton in this project scales uniformly or not at all.
	///
	/// @param matrix - what to read apart
	#[must_use]
	pub fn from_matrix(matrix: Mat4) -> Self {
		let (scale, rotation, position) = matrix.to_scale_rotation_translation();

		Self { position, rotation, scale }
	}

	/// The model matrix this transform stands for.
	#[must_use]
	pub fn matrix(&self) -> Mat4 {
		Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
	}

	/// Sets every axis of the scale at once.
	pub fn set_scale(&mut self, scale: f32) { self.scale = Vec3::splat(scale); }

	/// This transform with another applied inside it: where a child stands
	/// when this is its parent.
	///
	/// Position, rotation and scale composed each on their own, which is exact
	/// whenever this transform's scale is the same along every axis and the
	/// nearest thing that is not a shear otherwise: a turned child under a
	/// parent stretched along one axis would need a shear, and a translation,
	/// a rotation and a scale cannot hold one. That is the bargain every
	/// engine with a transform of this shape makes, and the one
	/// [`from_matrix`](Self::from_matrix) already makes here.
	///
	/// @param child - a transform relative to this one
	#[must_use]
	pub fn then(&self, child: Self) -> Self {
		Self {
			position: self.position + self.rotation * (self.scale * child.position),
			rotation: self.rotation * child.rotation,
			scale: self.scale * child.scale,
		}
	}

	/// The transform that, applied inside `parent`, lands at `world`.
	///
	/// The inverse of [`then`](Self::then): `parent.then(Transform::local_of(
	/// world, parent))` is `world` to within rounding, wherever `parent` scales
	/// by something other than nothing. An axis a parent scales by nothing is
	/// one nothing under it can be placed along, and along it the child's own
	/// numbers are kept rather than divided by it.
	///
	/// @param world - where the child is to be, in the world
	/// @param parent - what it hangs off, in the world
	#[must_use]
	pub fn local_of(world: Self, parent: Self) -> Self {
		let undo = parent.rotation.inverse();
		let scale = Vec3::select(parent.scale.cmpeq(Vec3::ZERO), Vec3::ONE, parent.scale);

		Self {
			position: (undo * (world.position - parent.position)) / scale,
			rotation: undo * world.rotation,
			scale: world.scale / scale,
		}
	}

	/// This transform part of the way towards another one.
	///
	/// Rotation is a slerp rather than a lerp. It costs an `acos` and a couple
	/// of sines per entity per frame, and it is the difference between a
	/// turning object turning evenly and one that hurries through the middle
	/// of every step.
	///
	/// Both ends are exact rather than merely close, which matters more than
	/// it sounds: glam's `slerp` finishes with a `normalize`, so the value it
	/// hands back at `t == 1.0` is `other` to within an ulp rather than
	/// `other`. Invisible on screen, and quite visible to a test that compares
	/// two renders.
	///
	/// @param other - the transform at the far end
	/// @param t - zero for this one, one for the other
	#[must_use]
	pub fn lerp(self, other: Self, t: f32) -> Self {
		if t <= 0.0 || self == other {
			return self;
		}

		if t >= 1.0 {
			return other;
		}

		Self {
			position: self.position.lerp(other.position, t),
			rotation: self.rotation.slerp(other.rotation, t),
			scale: self.scale.lerp(other.scale, t),
		}
	}
}

impl Default for Transform {
	fn default() -> Self { Self::IDENTITY }
}

/// What an entity looks like: a shape, what it is made of, and a tint.
///
/// The `color` survived the arrival of materials on purpose. A material
/// describes a *substance* and is shared between everything made of it; the
/// tint is this one entity's, and it is what makes a ring of identically
/// materialled cubes come out in eight colors. They multiply.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Renderable {
	/// The shape to draw.
	pub mesh: MeshId,

	/// What it is made of.
	pub material: MaterialId,

	/// This entity's own tint, linear RGB, each channel in `0.0 ..= 1.0`.
	pub color: Vec3,

	/// The bones that move it, or [`PoseId::NONE`] for a shape nothing bends.
	///
	/// Almost every entity in a world holds nothing here: a crate and a wall
	/// are drawn from their own transform and one mesh. A character names a
	/// pose, and the mesh it names then has to carry a skin block - the two
	/// halves of the same claim, and a mesh with one and no pose is drawn in
	/// the shape it was modeled in.
	///
	/// Two entities may name the same pose, and one model of two materials
	/// does exactly that. @ref [`pose`](super::pose).
	pub pose: PoseId,
}

impl Renderable {
	/// Its fields, for an inspector, a reader and a writer. @ref
	/// [`field`](super::field).
	pub const FIELDS: &[Field<Self>] = &[
		field!(Mesh, "mesh", mesh, "the shape to draw"),
		field!(Material, "material", material, "what it is made of"),
		field!(Color, "color", color, "its own tint, multiplied into the material"),
		field!(Pose, "pose", pose, "the bones that move it, or none"),
	];
	/// Draws nothing.
	pub const NOTHING: Self = Self {
		mesh: MeshId::NONE,
		material: MaterialId::DEFAULT,
		color: Vec3::ONE,
		pose: PoseId::NONE,
	};

	/// A shape in a color, made of the default material.
	#[must_use]
	pub const fn new(mesh: MeshId, color: Vec3) -> Self {
		Self {
			mesh,
			material: MaterialId::DEFAULT,
			color,
			pose: PoseId::NONE,
		}
	}

	/// The same, made of something in particular.
	#[must_use]
	pub const fn of(mesh: MeshId, material: MaterialId, color: Vec3) -> Self {
		Self {
			mesh,
			material,
			color,
			pose: PoseId::NONE,
		}
	}

	/// The same, moved by a pose.
	///
	/// @param pose - the bones that bend it
	#[must_use]
	pub const fn posed(self, pose: PoseId) -> Self { Self { pose, ..self } }
}

impl Default for Renderable {
	fn default() -> Self { Self::NOTHING }
}

/// How the renderer treats an entity beyond what it looks like.
///
/// **The engine's own record**, which every entity carries: declared the way a
/// game declares its fields, @ref [`record`](super::record), so the inspector,
/// a scene source, a save and a piece of the world on the wire reach it through
/// the one path a game's fields take. A flag like this on [`Renderable`] would
/// have been a table row and a file record and a key and a line in three
/// loaders; here it is a row.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct Drawing {
	/// Whether it is drawn into the pass before the scene ahead of the test for
	/// what is hidden, whatever its size: nought for no, anything else for yes.
	///
	/// For a thing that is one of many small pieces of something that hides a
	/// great deal - a brick in a wall, a plank in a fence. A thing too small on
	/// screen is otherwise drawn after the test, so what only small things hide
	/// is drawn; one that covers is in the depth the test reads.
	pub covers: u32,
}

impl Drawing {
	/// What every entity starts as: drawn however its size says.
	pub const NONE: Self = Self { covers: 0 };

	/// Whether it is drawn ahead of the test whatever its size.
	#[must_use]
	pub const fn covers(self) -> bool { self.covers != 0 }
}

impl Default for Drawing {
	fn default() -> Self { Self::NONE }
}

/// [`Drawing`] as the record every world declares.
pub const DRAWING: Record<Drawing> = Record {
	name: "drawing",
	help: "how the renderer treats the entity beyond what it looks like",
	rows: &[crate::row!(
		Bool,
		Drawing,
		covers,
		"drawn into the pass before the scene ahead of the cover test, whatever its size"
	)],
	default: Drawing::NONE,
};

/// How the editor treats an entity beyond where it is and what it looks like.
///
/// **The engine's second record**, for [`Drawing`]'s reason: a word any entity
/// may carry that the inspector, a scene source, a save, a copy and a piece of
/// the world on the wire already reach through the one path a game's fields
/// take. Nothing that runs a game reads it, and a build with no editor in it
/// carries it all the same, so a world written down by one keeps what the other
/// wrote.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct Editing {
	/// Whether a click in the picture on anything hanging off it selects it
	/// instead: nought for no, anything else for yes.
	///
	/// What makes an entity a group rather than only a parent. The outermost
	/// group around what was clicked is what a click selects, so a group inside
	/// a group, and anything inside one, is reached from the hierarchy or with
	/// alt held.
	pub group: u32,
}

impl Editing {
	/// What every entity starts as: selected by a click on itself alone.
	pub const NONE: Self = Self { group: 0 };

	/// Whether a click on anything hanging off it selects it.
	#[must_use]
	pub const fn group(self) -> bool { self.group != 0 }
}

impl Default for Editing {
	fn default() -> Self { Self::NONE }
}

/// [`Editing`] as the record every world declares.
pub const EDITING: Record<Editing> = Record {
	name: "editing",
	help: "how the editor treats the entity",
	rows: &[crate::row!(
		Bool,
		Editing,
		group,
		"a click in the picture on anything hanging off it selects it instead"
	)],
	default: Editing::NONE,
};

/// Where an entity's baked light is, and whether a bake reaches it at all.
///
/// **The engine's third record**, for [`Drawing`]'s reason: a place and a flag
/// every entity may carry, which a scene source, a save, a copy and a piece of
/// the world on the wire already reach through the path a game's fields take.
/// The world's own [`lightmap`](super::World::lightmap) is one picture for
/// every still thing, and this is where on it one thing's light is.
///
/// **A bake writes the place and never the flag**; a person or a game writes
/// the flag. The place is whole texels on the lightmap, and a width of nought
/// is a thing no bake reached - which is what every entity starts as.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct Baking {
	/// Whether a bake leaves it out: nought for no, anything else for yes.
	///
	/// Out altogether, as a thing that moves is: it is not lit by the bake,
	/// it throws no shadow into it and sends no light on. For a thing a game
	/// will move without a body the solver knows about, which a bake cannot
	/// tell from one that stands still.
	pub skip: u32,

	/// Where its light starts on the lightmap, in texels from the left.
	pub left: i32,

	/// And in texels from the top.
	pub top: i32,

	/// How many texels across its light takes, or nought for none.
	pub width: i32,

	/// And how many down.
	pub height: i32,
}

impl Baking {
	/// What every entity starts as: reached by a bake, and not baked yet.
	pub const NONE: Self = Self {
		skip: 0,
		left: 0,
		top: 0,
		width: 0,
		height: 0,
	};

	/// Whether a bake leaves it out.
	#[must_use]
	pub const fn skip(self) -> bool { self.skip != 0 }

	/// Whether a bake gave it a place on the lightmap.
	#[must_use]
	pub const fn is_baked(self) -> bool { self.width > 0 && self.height > 0 }
}

impl Default for Baking {
	fn default() -> Self { Self::NONE }
}

/// [`Baking`] as the record every world declares.
pub const BAKING: Record<Baking> = Record {
	name: "baking",
	help: "where the entity's baked light is on the world's lightmap, and whether a bake leaves \
	       it out",
	rows: &[
		crate::row!(
			Bool,
			Baking,
			skip,
			"left out of every bake: neither lit by it nor lighting anything in it"
		),
		crate::row!(
			Int,
			Baking,
			left,
			"where its light starts on the lightmap, in texels from the left; a bake writes it"
		),
		crate::row!(
			Int,
			Baking,
			top,
			"where its light starts on the lightmap, in texels from the top; a bake writes it"
		),
		crate::row!(
			Int,
			Baking,
			width,
			"how many texels across its light takes, nought for none; a bake writes it"
		),
		crate::row!(
			Int,
			Baking,
			height,
			"how many texels down its light takes; a bake writes it"
		),
	],
	default: Baking::NONE,
};

/// The host's entity table.
///
/// Component storage is hard-coded to one array of [`Transform`] because there
/// is exactly one component so far. When there is a second reason to, this
/// becomes something that deserves the name.
///
/// Growth is bounded rather than fixed: the arrays start empty and stop at
/// [`MAX_ENTITIES`]. Bounded because the gameplay crate is code that is
/// *expected* to be wrong sometimes, and a reload that spawns a ring every time
/// should run out of slots rather than out of memory.
pub struct Entities {
	transforms: Vec<Transform>,
	/// Where everything was at the previous step. The same slots as
	/// `transforms`, and the same length; the renderer draws between the two.
	previous: Vec<Transform>,
	renderables: Vec<Renderable>,
	/// What each slot shines, or [`Light::NONE`] for a slot that is not a
	/// lamp. The same slots again, and almost every one of them holds the
	/// nothing: a light is rare where a transform is universal, and this is
	/// an array anyway for the reason the others are - a slot is addressed
	/// by index, and a side table would be a second lookup on the one path
	/// the renderer walks every frame. @ref [`light`](super::light).
	lights: Vec<Light>,
	/// What each slot throws off, or [`Emitter::NONE`] for a slot that throws
	/// nothing. The same slots again, and a light's argument word for word: an
	/// emitter is rare where a transform is universal, and the step walks
	/// every slot looking for one. What it throws is *not* here - the cloud is
	/// [`World::sparks`](super::World::sparks), one pool for the world, and
	/// this is only the description of it. @ref
	/// [`particles`](super::particles).
	emitters: Vec<Emitter>,
	/// What ground each slot is, or [`Terrain::NONE`] for a slot that is not
	/// ground. The same slots again, and the light's argument a third time: a
	/// terrain is rarer than an emitter and the array is still what the sync
	/// walks. **What it built is not here** - the mesh is an entry in
	/// [`Meshes`](super::Meshes) under a name and the body is a row in
	/// [`Bodies`](super::Bodies), so a terrain that has been built is
	/// indistinguishable from a hill somebody modeled. @ref
	/// [`terrain`](super::terrain).
	terrains: Vec<Terrain>,
	/// What each slot paints, or [`Decal::NONE`] for a slot that paints
	/// nothing. The same slots again, and the light's argument a fourth time:
	/// a decal is rare where a transform is universal, and the renderer walks
	/// every slot looking for one. **Where it paints is not here**: the box
	/// is the slot's own transform and the picture its own material. @ref
	/// [`decal`](super::decal).
	decals: Vec<Decal>,
	/// What share of a strewing's copies may stand where over each slot's
	/// ground, or nothing for a slot nobody has painted. The same slots again,
	/// and the one array here that is not a plain record: a mask is a grid of
	/// cells and the only thing an entity carries that a person draws rather
	/// than types. It is here, beside the light and the ground, because it is
	/// something an entity *has* - it is written down with the entity, cleared
	/// with its slot, and carried by a copy of it. @ref
	/// [`Mask`](super::strew::Mask).
	masks: Vec<Option<Mask>>,
	/// What each slot hangs off, or [`EntityId::NONE`] for a thing standing
	/// on its own. The same slots again. A handle rather than a slot number,
	/// so that a parent which died and whose slot something else took is a
	/// stale handle resolving to nobody, rather than a new parent nobody
	/// asked for. @ref [`Entities::parent`].
	parents: Vec<EntityId>,
	/// Whether each slot is hidden, which hides everything hanging off it as
	/// well. The same slots again, and each slot's own word only: whether a
	/// thing is drawn is a walk up its parents, @ref [`Entities::shown`], and
	/// never a second copy kept here.
	hidden: Vec<bool>,
	/// Whether each slot refuses decals, so that nothing a decal throws lands
	/// on it. The same slots again, and each slot's own word: unlike being
	/// hidden it is not handed down to what hangs off the slot. @ref
	/// [`Entities::takes_decals`].
	undecaled: Vec<bool>,
	/// What each slot is called, or the empty string. The same slots again,
	/// and the one array here that is not read by anything the engine does -
	/// it exists for whoever has to point at a particular entity in words.
	names: Names,
	/// Every declared record, one copy a slot. The same slots again: a
	/// record is something an entity carries, like its light, and a slot is
	/// handed out holding every record's default. @ref
	/// [`record`](super::record).
	records: Records,
	generations: Vec<u32>,
	alive: Vec<bool>,
	free: Vec<u32>,
	/// Slots whose past is rewritten to their present at the end of this step,
	/// so that they are not drawn traveling across whatever just happened to
	/// them. @ref [`Entities::snap`].
	pending: Vec<usize>,
	/// Whether every slot is.
	pending_all: bool,
	/// How many entities are alive right now.
	live: usize,
}

impl Entities {
	/// An empty table.
	#[must_use]
	pub const fn new() -> Self {
		Self {
			transforms: Vec::new(),
			previous: Vec::new(),
			renderables: Vec::new(),
			lights: Vec::new(),
			emitters: Vec::new(),
			terrains: Vec::new(),
			decals: Vec::new(),
			masks: Vec::new(),
			parents: Vec::new(),
			hidden: Vec::new(),
			undecaled: Vec::new(),
			names: Names::new(),
			records: Records::new(),
			generations: Vec::new(),
			alive: Vec::new(),
			free: Vec::new(),
			pending: Vec::new(),
			pending_all: false,
			live: 0,
		}
	}

	/// Creates an entity at the origin.
	///
	/// @return its handle, or [`EntityId::NONE`] if the table is full
	pub fn spawn(&mut self) -> EntityId { self.spawn_at(Transform::IDENTITY) }

	/// Creates an entity with a transform.
	///
	/// @param transform - where it starts
	/// @return its handle, or [`EntityId::NONE`] if the table is full
	pub fn spawn_at(&mut self, transform: Transform) -> EntityId {
		let Some(slot) = self.take_slot() else {
			return EntityId::NONE;
		};

		let Ok(index) = u32::try_from(slot) else {
			return EntityId::NONE;
		};

		self.generations[slot] = self.generations[slot].saturating_add(1);
		self.alive[slot] = true;
		self.transforms[slot] = transform;
		// both halves, and again at the end of the step. An entity that did
		// not exist a step ago has no past to be drawn arriving from, and the
		// slot's previous occupant is certainly not it; the pending entry
		// covers the usual shape of `spawn()` followed by a transform written
		// later in the same step.
		self.previous[slot] = transform;
		self.pending.push(slot);
		self.renderables[slot] = Renderable::NOTHING;
		self.lights[slot] = Light::NONE;
		self.emitters[slot] = Emitter::NONE;
		self.terrains[slot] = Terrain::NONE;
		// and it paints nothing, cleared here and not where the slot is given
		// back, by the rule the hidden word below keeps: each way a slot is
		// handed out clears it once, and nothing reads a dead slot.
		self.decals[slot] = Decal::NONE;
		// and nothing painted over it, by the name's rule above: a slot handed
		// out carries no stroke the last occupant's brush made.
		self.masks[slot] = None;
		// and it hangs off nothing, whatever the previous occupant did.
		self.parents[slot] = EntityId::NONE;
		// and it is shown, whatever the previous occupant was. Cleared where a
		// slot is handed out rather than where one is given back, the rule a
		// name follows below: nothing reads a dead slot's word.
		self.hidden[slot] = false;
		// and it takes decals, by the same rule and for the same reason.
		self.undecaled[slot] = false;
		// whatever the previous occupant of this slot was called is not what
		// this is called. This is the only place a name is cleared, and it is
		// here rather than at the despawn because a slot reaches the free list
		// three ways and leaves it one. @ref `abi::names`.
		self.names.set(slot, "");
		// and it carries every record at its default, with nothing waiting,
		// for the name's reason.
		self.records.clear(slot);
		self.live += 1;

		EntityId {
			index,
			generation: self.generations[slot],
		}
	}

	/// Destroys an entity.
	///
	/// @param id - the handle to destroy
	/// @return `true` if it was alive, `false` if the handle was stale
	pub fn despawn(&mut self, id: EntityId) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.alive[slot] = false;
		self.transforms[slot] = Transform::IDENTITY;
		self.previous[slot] = Transform::IDENTITY;
		self.renderables[slot] = Renderable::NOTHING;
		self.lights[slot] = Light::NONE;
		self.emitters[slot] = Emitter::NONE;
		self.terrains[slot] = Terrain::NONE;
		self.parents[slot] = EntityId::NONE;
		self.free.push(id.index);
		self.live -= 1;

		true
	}

	/// Destroys everything.
	pub fn clear(&mut self) {
		for slot in 0..self.alive.len() {
			if !self.alive[slot] {
				continue;
			}

			self.alive[slot] = false;
			self.transforms[slot] = Transform::IDENTITY;
			self.previous[slot] = Transform::IDENTITY;
			self.renderables[slot] = Renderable::NOTHING;
			self.lights[slot] = Light::NONE;
			self.emitters[slot] = Emitter::NONE;
			self.terrains[slot] = Terrain::NONE;
			self.parents[slot] = EntityId::NONE;
			if let Ok(index) = u32::try_from(slot) {
				self.free.push(index);
			}
		}

		self.live = 0;
	}

	/// Whether a handle refers to a living entity.
	#[must_use]
	pub fn alive(&self, id: EntityId) -> bool { self.slot(id).is_some() }

	/// How many entities are alive.
	#[must_use]
	pub const fn len(&self) -> usize { self.live }

	/// Whether there are no entities at all.
	#[must_use]
	pub const fn is_empty(&self) -> bool { self.live == 0 }

	/// How many more entities can be created.
	#[must_use]
	pub fn capacity_left(&self) -> usize { MAX_ENTITIES - self.alive.len() + self.free.len() }

	/// An entity's transform.
	#[must_use]
	pub fn transform(&self, id: EntityId) -> Option<&Transform> {
		self.slot(id).map(|slot| &self.transforms[slot])
	}

	/// An entity's transform, to change.
	pub fn transform_mut(&mut self, id: EntityId) -> Option<&mut Transform> {
		self.slot(id)
			.map(|slot| &mut self.transforms[slot])
	}

	/// What an entity looks like.
	#[must_use]
	pub fn renderable(&self, id: EntityId) -> Option<&Renderable> {
		self.slot(id).map(|slot| &self.renderables[slot])
	}

	/// What an entity looks like, to change.
	pub fn renderable_mut(&mut self, id: EntityId) -> Option<&mut Renderable> {
		self.slot(id)
			.map(|slot| &mut self.renderables[slot])
	}

	/// Gives an entity a shape and a color in one go.
	///
	/// @return `true` if the handle resolved
	pub fn set_renderable(&mut self, id: EntityId, renderable: Renderable) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.renderables[slot] = renderable;

		true
	}

	/// What an entity shines, or [`Light::NONE`] for one that shines nothing.
	///
	/// Every living slot answers this, and almost every answer is the nothing:
	/// carrying a light is the rare case, and a caller that walks the table
	/// asks [`Light::is_lit`] rather than this. @ref [`light`](super::light)
	/// for why the position and the direction are not in the answer.
	#[must_use]
	pub fn light(&self, id: EntityId) -> Option<&Light> {
		self.slot(id).map(|slot| &self.lights[slot])
	}

	/// What an entity shines, to change.
	pub fn light_mut(&mut self, id: EntityId) -> Option<&mut Light> {
		self.slot(id).map(|slot| &mut self.lights[slot])
	}

	/// Makes an entity a lamp, or stops it being one.
	///
	/// @param id - which entity
	/// @param light - what it shines; [`Light::NONE`] puts it out
	/// @return `true` if the handle resolved
	pub fn set_light(&mut self, id: EntityId, light: Light) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.lights[slot] = light;

		true
	}

	/// What an entity throws off, or [`Emitter::NONE`] for one that throws
	/// nothing.
	///
	/// Every living slot answers this and almost every answer is the nothing,
	/// exactly as [`light`](Self::light) is. A caller walking the table asks
	/// [`Emitter::throws`] rather than this. @ref
	/// [`particles`](super::particles) for why the position and the direction
	/// are not in the answer, and where what it has thrown lives.
	#[must_use]
	pub fn emitter(&self, id: EntityId) -> Option<&Emitter> {
		self.slot(id).map(|slot| &self.emitters[slot])
	}

	/// What an entity throws off, to change.
	pub fn emitter_mut(&mut self, id: EntityId) -> Option<&mut Emitter> {
		self.slot(id).map(|slot| &mut self.emitters[slot])
	}

	/// Makes an entity throw particles, or stops it.
	///
	/// **The cloud it has already thrown is not touched**, which is the answer
	/// with a reason: a fire that is turned off should burn out rather than
	/// vanish, and the step sweeps what is left as each particle's own life
	/// runs down. What does clear the cloud at once is the entity dying, and
	/// that is [`Sparks::forget`](super::particles::Sparks::forget) from the
	/// step rather than anything here - this table has no reach into the pool.
	///
	/// @param id - which entity
	/// @param emitter - what it throws; [`Emitter::NONE`] stops it
	/// @return `true` if the handle resolved
	pub fn set_emitter(&mut self, id: EntityId, emitter: Emitter) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.emitters[slot] = emitter;

		true
	}

	/// What ground an entity is, or [`Terrain::NONE`] for one that is not
	/// ground.
	///
	/// Every living slot answers this and almost every answer is the nothing,
	/// exactly as [`emitter`](Self::emitter) is. @ref
	/// [`terrain`](super::terrain) for why the mesh it describes is not in the
	/// answer.
	#[must_use]
	pub fn terrain(&self, id: EntityId) -> Option<&Terrain> {
		self.slot(id).map(|slot| &self.terrains[slot])
	}

	/// What ground an entity is, to change.
	pub fn terrain_mut(&mut self, id: EntityId) -> Option<&mut Terrain> {
		self.slot(id).map(|slot| &mut self.terrains[slot])
	}

	/// Makes an entity ground, or stops it being ground.
	///
	/// **What it had built is not touched here.** The mesh in the registry and
	/// the body in the table both belong to whoever builds them, and that is
	/// `colby_runtime::terrain` on the next step: it notices that the record
	/// no longer matches what it built and unmakes it. This table has no reach
	/// into either, exactly as it has none into the particle pool.
	///
	/// @param id - which entity
	/// @param terrain - what ground it is; [`Terrain::NONE`] for none
	/// @return `true` if the handle resolved
	pub fn set_terrain(&mut self, id: EntityId, terrain: Terrain) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.terrains[slot] = terrain;

		true
	}

	/// What an entity paints, or [`Decal::NONE`] for one that paints nothing.
	///
	/// Every living slot answers this and almost every answer is the nothing,
	/// exactly as [`light`](Self::light) is. A caller walking the table asks
	/// [`Decal::paints`] rather than this. @ref [`decal`](super::decal) for why
	/// the box and the picture are not in the answer.
	#[must_use]
	pub fn decal(&self, id: EntityId) -> Option<&Decal> {
		self.slot(id).map(|slot| &self.decals[slot])
	}

	/// What an entity paints, to change.
	pub fn decal_mut(&mut self, id: EntityId) -> Option<&mut Decal> {
		self.slot(id).map(|slot| &mut self.decals[slot])
	}

	/// Makes an entity a decal, or stops it being one.
	///
	/// @param id - which entity
	/// @param decal - what it paints; [`Decal::NONE`] paints nothing
	/// @return `true` if the handle resolved
	pub fn set_decal(&mut self, id: EntityId, decal: Decal) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.decals[slot] = decal;

		true
	}

	/// What share of a strewing's copies may stand where, or nothing for an
	/// entity nobody has painted.
	///
	/// It means something only on an entity that strews, and it is kept on
	/// every entity all the same, for the terrain record's reason: what a slot
	/// carries does not depend on what else the slot carries. @ref
	/// [`Mask`](super::strew::Mask).
	#[must_use]
	pub fn mask(&self, id: EntityId) -> Option<&Mask> {
		self.slot(id)
			.and_then(|slot| self.masks[slot].as_ref())
	}

	/// What it has painted, to paint on.
	pub fn mask_mut(&mut self, id: EntityId) -> Option<&mut Mask> {
		self.slot(id)
			.and_then(|slot| self.masks[slot].as_mut())
	}

	/// Puts a mask on an entity, or takes the one it had away.
	///
	/// @param id - which entity
	/// @param mask - what it has painted; nothing takes the mask away, and a
	/// strewing with no mask lays its whole field
	/// @return `true` if the handle resolved
	pub fn set_mask(&mut self, id: EntityId, mask: Option<Mask>) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.masks[slot] = mask;

		true
	}

	/// What an entity is called, or the empty string.
	///
	/// Never an identifier - that is the handle, and it is unique where this
	/// is not. @ref [`names`](crate::abi::names) for why the world holds this
	/// at all.
	#[must_use]
	pub fn name(&self, id: EntityId) -> &str {
		self.slot(id)
			.map_or("", |slot| self.names.at(slot))
	}

	/// Names an entity, cutting anything past
	/// [`MAX_NAME`](crate::abi::MAX_NAME).
	///
	/// @param id - what to name
	/// @param name - what to call it; empty clears the name
	/// @return `true` if the handle resolved
	pub fn set_name(&mut self, id: EntityId, name: &str) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.names.set(slot, name);

		true
	}

	/// What an entity hangs off, or [`EntityId::NONE`] for one standing on its
	/// own.
	///
	/// A parent that has been despawned is no parent: the handle kept here
	/// stops resolving the moment its slot is freed, so a child is not moved by
	/// whatever takes the slot afterwards and reads as a root from then on,
	/// without anything having to walk the table when its parent goes.
	#[must_use]
	pub fn parent(&self, id: EntityId) -> EntityId {
		let Some(slot) = self.slot(id) else {
			return EntityId::NONE;
		};

		let parent = self.parents[slot];

		if self.slot(parent).is_some() {
			parent
		} else {
			EntityId::NONE
		}
	}

	/// Hangs an entity off another, or takes it down.
	///
	/// The entity's own transform is left exactly as it is and starts meaning
	/// something else: its place inside the parent rather than in the world.
	/// A caller that wants the thing to stay where it stands writes
	/// [`set_placed`](Self::set_placed) afterwards with the place it had.
	///
	/// @param id - the entity
	/// @param parent - what to hang it off, or [`EntityId::NONE`] to stand it
	/// on its own
	/// @return `true` if it was done; `false` for a stale handle, a parent
	/// that is not alive, an entity hanging off itself, or a loop - an entity
	/// cannot hang off something that hangs off it
	pub fn set_parent(&mut self, id: EntityId, parent: EntityId) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		if !parent.is_some() {
			self.parents[slot] = EntityId::NONE;

			return true;
		}

		if self.slot(parent).is_none() || parent == id {
			return false;
		}

		// walking up from the parent: a walk that reaches this entity means
		// hanging off that parent would close a loop, and a loop is a chain
		// nothing can resolve. Bounded by the table, which no chain exceeds.
		let mut above = parent;
		for _ in 0..MAX_ENTITIES {
			above = self.parent(above);

			if !above.is_some() {
				break;
			}

			if above == id {
				return false;
			}
		}

		self.parents[slot] = parent;

		true
	}

	/// Where an entity is in the world, with everything it hangs off applied.
	///
	/// The same as [`transform`](Self::transform) for an entity standing on
	/// its own, and every parent's transform applied in turn for one that is
	/// not. Worked out when asked rather than kept: a second copy of every
	/// transform is stale for whoever reads it between the write and the pass
	/// that would refresh it, and a chain here is a few multiplies.
	///
	/// @param id - the entity
	/// @return where it is, or `None` if the handle is stale
	#[must_use]
	pub fn placed(&self, id: EntityId) -> Option<Transform> {
		let slot = self.slot(id)?;
		let mut placed = self.transforms[slot];
		let mut above = self.parent(id);

		// bounded like the check in `set_parent`, and for the same reason: a
		// loop cannot be made, and a walk that could not end anyway must.
		for _ in 0..MAX_ENTITIES {
			let Some(slot) = self.slot(above) else {
				break;
			};

			placed = self.transforms[slot].then(placed);
			above = self.parent(above);
		}

		Some(placed)
	}

	/// Puts an entity somewhere in the world, whatever it hangs off.
	///
	/// Writes the entity's own transform such that [`placed`](Self::placed)
	/// answers with `world`: the transform itself for an entity standing on
	/// its own, and the place inside the parent for one that is not.
	///
	/// @param id - the entity
	/// @param world - where it is to be, in the world
	/// @return `true` if the handle resolved
	pub fn set_placed(&mut self, id: EntityId, world: Transform) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		let local = match self.placed(self.parent(id)) {
			| Some(parent) => Transform::local_of(world, parent),
			| None => world,
		};

		self.transforms[slot] = local;

		true
	}

	/// Whether an entity is hidden itself.
	///
	/// Its own word and nothing else: a child of something hidden answers
	/// `false` here and is not drawn all the same. [`shown`](Self::shown) is
	/// the question a picture asks.
	///
	/// @param id - the entity
	/// @return its own word, or `false` for a stale handle
	#[must_use]
	pub fn hidden(&self, id: EntityId) -> bool {
		self.slot(id)
			.is_some_and(|slot| self.hidden[slot])
	}

	/// Hides an entity and everything hanging off it, or shows it again.
	///
	/// **Hidden is not switched off.** A hidden entity is stepped like any
	/// other: its bodies collide and move it, its sounds play and its emitter
	/// goes on throwing. What it loses is being drawn (its mesh, its shadow,
	/// its lamp and its cloud) and being clicked on in the editor. Nothing but
	/// a picture asks, which is also why nothing a simulation produces can
	/// depend on it.
	///
	/// @param id - the entity
	/// @param hidden - whether it is to be hidden
	/// @return `true` if the handle resolved
	pub fn set_hidden(&mut self, id: EntityId, hidden: bool) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.hidden[slot] = hidden;

		true
	}

	/// Whether an entity is to be drawn: alive, not hidden, and hanging off
	/// nothing that is.
	///
	/// Worked out when asked rather than kept, for the reason
	/// [`placed`](Self::placed) is: a second copy of the answer is stale for
	/// whoever reads it between a write and the pass that would refresh it, a
	/// child hung under something hidden being the obvious case, and the walk
	/// is the chain a drawn transform walks anyway. A hidden ancestor always
	/// wins; nothing under one can ask to be drawn regardless.
	///
	/// @param id - the entity
	/// @return `false` for a stale handle, a hidden entity, or one under a
	/// hidden one
	#[must_use]
	pub fn shown(&self, id: EntityId) -> bool {
		if self.slot(id).is_none() {
			return false;
		}

		let mut at = id;

		// bounded like the walk in `placed`, and for its reason: a loop cannot
		// be made, and a walk that could not end anyway must.
		for _ in 0..MAX_ENTITIES {
			if !at.is_some() {
				return true;
			}

			if self.hidden(at) {
				return false;
			}

			at = self.parent(at);
		}

		// @note: not reachable, a chain being shorter than the table, and no
		// test can see it. It errs towards drawing, the side on which a
		// picture can be seen to be wrong.
		true
	}

	/// Whether decals paint an entity's surfaces.
	///
	/// Its own word and nothing else: unlike [`shown`](Self::shown) this is not
	/// a walk up the parents, because what it says is about a surface and a
	/// child's surface is its own. The renderer asks it once for every entity
	/// it draws.
	///
	/// @param id - the entity
	/// @return `true` unless the entity said otherwise, and `false` for a stale
	/// handle, which has no surface to paint
	#[must_use]
	pub fn takes_decals(&self, id: EntityId) -> bool {
		self.slot(id)
			.is_some_and(|slot| !self.undecaled[slot])
	}

	/// Says whether decals may paint an entity's surfaces.
	///
	/// For something that moves through a world full of them: a character
	/// walking through a puddle's box would otherwise have the puddle painted
	/// up its legs. What it changes is only what is drawn, the way hiding
	/// does, and nothing a step produces can depend on it.
	///
	/// @param id - the entity
	/// @param takes - whether decals paint it
	/// @return `true` if the handle resolved
	pub fn set_takes_decals(&mut self, id: EntityId, takes: bool) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.undecaled[slot] = !takes;

		true
	}

	/// Declares a record every entity carries, or declares it again.
	///
	/// What a game calls from `init`, once for each record it keeps on an
	/// entity. @ref [`Records::declare`] for what a second declaration does,
	/// and [`record`](super::record) for what a record is.
	///
	/// @param record - what to declare
	/// @return what it took of what was already in the world by name
	///
	/// # Errors
	///
	/// When the record cannot be held, or another has its name.
	pub fn declare<T: Pod>(&mut self, record: &Record<T>) -> Result<Declared> {
		self.records.declare(record)
	}

	/// One entity's copy of a record, as the struct it was declared over.
	///
	/// @param record - which record
	/// @param id - the entity
	/// @return nothing for a stale handle, a record nobody declared, or a
	/// struct of another size than the one declared
	#[must_use]
	pub fn record<T: Pod>(&self, record: &Record<T>, id: EntityId) -> Option<&T> {
		self.records.view(record, self.slot(id)?)
	}

	/// One entity's copy of a record, to change.
	pub fn record_mut<T: Pod>(&mut self, record: &Record<T>, id: EntityId) -> Option<&mut T> {
		let slot = self.slot(id)?;

		self.records.view_mut(record, slot)
	}

	/// Every slot's copy of a record, in slot order, dead slots included.
	///
	/// For a pass that already walks the table and holds each entity's slot:
	/// one lookup of the record for the whole walk rather than one an entity.
	/// A caller with a handle and not a walk asks [`record`](Self::record).
	#[must_use]
	pub fn column<T: Pod>(&self, record: &Record<T>) -> Option<&[T]> {
		self.records.column(record)
	}

	/// Every declared record: what an inspector and a writer walk.
	#[must_use]
	pub const fn records(&self) -> &Records { &self.records }

	/// Every declared record, to declare into and for the host to attribute,
	/// mark and sweep across a reload.
	pub const fn records_mut(&mut self) -> &mut Records { &mut self.records }

	/// One field of one entity's record, whatever it holds.
	///
	/// @param id - the entity
	/// @param table - the record's place in [`Records::tables`]
	/// @param column - the field's place in its record's columns
	#[must_use]
	pub fn field(&self, id: EntityId, table: usize, column: usize) -> Option<Value> {
		self.records.field(table, column, self.slot(id)?)
	}

	/// Writes one field of one entity's record.
	///
	/// @return whether it was written: `false` for a stale handle, or a value
	/// the field does not hold
	pub fn set_field(
		&mut self,
		id: EntityId,
		table: usize,
		column: usize,
		value: &Value,
	) -> bool {
		self.slot(id)
			.is_some_and(|slot| self.records.set_field(table, column, slot, value))
	}

	/// What one entity's records hold that is worth writing down, by name.
	///
	/// @ref [`record`](super::record) for why a value leaves a world by name.
	#[must_use]
	pub fn noted(&self, id: EntityId) -> Vec<Noted> {
		self.slot(id)
			.map_or_else(Vec::new, |slot| self.records.noted(slot))
	}

	/// Puts written values into one entity's records.
	///
	/// @return what no declared record could take, for the caller to report
	/// once for a whole load
	pub fn note(&mut self, id: EntityId, noted: &[Noted]) -> Vec<Refused> {
		let Some(slot) = self.slot(id) else {
			return Vec::new();
		};

		self.records.note(slot, noted)
	}

	/// What waits in one entity for a record nobody has declared.
	#[must_use]
	pub fn waiting(&self, id: EntityId) -> &[Noted] {
		self.slot(id)
			.map_or(&[], |slot| self.records.waiting(slot))
	}

	/// Moves the present into the past, ready for another step.
	///
	/// The host calls this before every simulation step, and once more after a
	/// game module is swapped in: a reload is a discontinuity by definition,
	/// and nothing should be drawn sliding out of the pose the previous build
	/// left behind.
	pub fn advance(&mut self) {
		// `clone_from` rather than `copy_from_slice`, which panics on a length
		// mismatch. This runs in the host, outside the `catch_unwind` that
		// contains the game, so that failure would take the process with it.
		self.previous.clone_from(&self.transforms);
		self.pending.clear();
		self.pending_all = false;
	}

	/// Applies everything that asked not to be interpolated this step.
	///
	/// The host calls this after the game's `update`. Deferring is what makes
	/// [`snap`](Self::snap) independent of where in the step it was called -
	/// a teleport followed by a step's worth of ordinary movement leaves the
	/// movement interpolated and the teleport not, whichever order the two
	/// were written in.
	pub fn settle(&mut self) {
		if self.pending_all {
			self.previous.clone_from(&self.transforms);
			self.pending.clear();
			self.pending_all = false;

			return;
		}

		let previous = &mut self.previous;
		for &slot in &self.pending {
			let (Some(was), Some(is)) = (previous.get_mut(slot), self.transforms.get(slot))
			else {
				continue;
			};

			*was = *is;
		}

		self.pending.clear();
	}

	/// Declares that an entity's transform changed discontinuously.
	///
	/// A teleport, a wrap-around, a level swap: anything the entity did not
	/// travel to. Without this the renderer draws the journey, because a
	/// journey is exactly what two transforms a step apart look like.
	///
	/// @param id - the entity that jumped
	/// @return `true` if the handle resolved
	pub fn snap(&mut self, id: EntityId) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.pending.push(slot);

		true
	}

	/// The same for the whole table, for when a scene cuts rather than moves.
	pub fn snap_all(&mut self) { self.pending_all = true; }

	/// Where an entity should be drawn part of the way through a step.
	///
	/// @param id - the entity to place
	/// @param t - how far past the previous step this frame sits, `0.0 ..= 1.0`
	/// @return the blended transform, or `None` if the handle is stale
	#[must_use]
	pub fn interpolated(&self, id: EntityId, t: f32) -> Option<Transform> {
		let slot = self.slot(id)?;
		let current = self.transforms[slot];
		// a missing past is not worth losing the entity over: drawing it where
		// it is beats the renderer skipping it and the thing vanishing with
		// nothing said.
		let previous = self
			.previous
			.get(slot)
			.copied()
			.unwrap_or(current);

		Some(previous.lerp(current, t.clamp(0.0, 1.0)))
	}

	/// Where an entity should be drawn part of the way through a step, with
	/// everything it hangs off applied.
	///
	/// [`interpolated`](Self::interpolated) at every level of the chain and
	/// the levels composed, so a child of something turning is drawn turning
	/// with it between two steps rather than snapping to wherever the step
	/// left its parent.
	///
	/// @param id - the entity to place
	/// @param t - how far past the previous step this frame sits, `0.0 ..= 1.0`
	/// @return the blended transform, or `None` if the handle is stale
	#[must_use]
	pub fn blended(&self, id: EntityId, t: f32) -> Option<Transform> {
		let mut placed = self.interpolated(id, t)?;
		let mut above = self.parent(id);

		for _ in 0..MAX_ENTITIES {
			let Some(parent) = self.interpolated(above, t) else {
				break;
			};

			placed = parent.then(placed);
			above = self.parent(above);
		}

		Some(placed)
	}

	/// Every living entity, with everything it has.
	///
	/// Yields in slot order, which is stable until something is despawned.
	pub fn iter(&self) -> impl Iterator<Item = (EntityId, &Transform, &Renderable)> {
		self.alive
			.iter()
			.enumerate()
			.filter(|(_, alive)| **alive)
			.filter_map(|(slot, _)| {
				let id = EntityId {
					index: u32::try_from(slot).ok()?,
					generation: self.generations[slot],
				};

				Some((id, &self.transforms[slot], &self.renderables[slot]))
			})
	}

	/// Rebuilds the whole table from a description, slot for slot.
	///
	/// The other half of writing a world down: a handle only means anything if
	/// it lands back in the slot and the generation it came from, so this
	/// replaces the table rather than adding to it. Everything alive before is
	/// gone, and every handle to it is stale afterwards unless the description
	/// happens to name the same slot and generation - which is exactly what a
	/// saved world does and exactly what a different one does not.
	///
	/// The free list is *not* part of the description and is rebuilt here from
	/// whichever slots nothing occupies, in ascending order - so the next
	/// spawn takes the highest empty slot rather than whichever died most
	/// recently, the list being a stack. The order things died in is not
	/// recoverable and is not worth writing down: the only thing that depends
	/// on it is which slot the next spawn takes, and what matters there is
	/// that two hosts reading one description derive the same answer.
	///
	/// Both halves of every transform are set to the same value, so nothing is
	/// drawn traveling out of the world that was here before.
	///
	/// @param generations - the generation of every slot, dead ones included;
	/// its length is how many slots the table ends up with, capped at
	/// [`MAX_ENTITIES`]
	/// @param entries - `(slot, transform, renderable)` for each living entity
	/// @return one handle per entry, in order, [`EntityId::NONE`] for any whose
	/// slot the table could not hold
	pub fn restore(
		&mut self,
		generations: &[u32],
		entries: &[(usize, Transform, Renderable)],
	) -> Vec<EntityId> {
		let slots = generations.len().min(MAX_ENTITIES);

		self.transforms.clear();
		self.transforms.resize(slots, Transform::IDENTITY);
		self.previous.clear();
		self.previous.resize(slots, Transform::IDENTITY);
		self.renderables.clear();
		self.renderables
			.resize(slots, Renderable::NOTHING);
		self.lights.clear();
		self.lights.resize(slots, Light::NONE);
		self.emitters.clear();
		self.emitters.resize(slots, Emitter::NONE);
		self.terrains.clear();
		self.terrains.resize(slots, Terrain::NONE);
		self.parents.clear();
		self.parents.resize(slots, EntityId::NONE);
		self.hidden.clear();
		self.hidden.resize(slots, false);
		self.decals.clear();
		self.decals.resize(slots, Decal::NONE);
		self.masks.clear();
		self.masks.resize(slots, None);
		self.undecaled.clear();
		self.undecaled.resize(slots, false);
		self.names.reset(slots);
		self.records.reset(slots);
		self.generations.clear();
		self.generations
			.extend_from_slice(&generations[..slots]);
		self.alive.clear();
		self.alive.resize(slots, false);
		self.free.clear();
		self.pending.clear();
		self.pending_all = false;
		self.live = 0;

		let mut handles = Vec::with_capacity(entries.len());
		for &(slot, transform, renderable) in entries {
			handles.push(self.put(slot, transform, renderable));
		}

		for slot in 0..slots {
			if !self.alive[slot]
				&& let Ok(index) = u32::try_from(slot)
			{
				self.free.push(index);
			}
		}

		handles
	}

	/// Puts one entity into a named slot of a table that is already in use.
	///
	/// The entity half of [`Bodies::graft`](crate::abi::Bodies::graft), and
	/// the whole of the argument is there: a restore empties the table and
	/// this does not, because what is being described is one thing that
	/// appeared rather than a world that was replaced.
	///
	/// @param slot - which array index the far end has it in
	/// @param generation - which occupant of that slot it is
	/// @param transform - where it is
	/// @param renderable - what it looks like
	/// @return the handle, or [`EntityId::NONE`] when the slot is past the
	/// ceiling or is already occupied here
	pub fn graft(
		&mut self,
		slot: usize,
		generation: u32,
		transform: Transform,
		renderable: Renderable,
	) -> EntityId {
		if slot >= MAX_ENTITIES {
			return EntityId::NONE;
		}

		while self.alive.len() <= slot {
			self.transforms.push(Transform::IDENTITY);
			self.previous.push(Transform::IDENTITY);
			self.renderables.push(Renderable::NOTHING);
			self.lights.push(Light::NONE);
			self.emitters.push(Emitter::NONE);
			self.terrains.push(Terrain::NONE);
			self.decals.push(Decal::NONE);
			self.masks.push(None);
			self.parents.push(EntityId::NONE);
			self.hidden.push(false);
			self.undecaled.push(false);
			self.names.push();
			self.records.push();
			self.generations.push(0);
			self.alive.push(false);

			let grown = self.alive.len() - 1;

			if grown != slot
				&& let Ok(index) = u32::try_from(grown)
			{
				self.free.push(index);
			}
		}

		if self.alive.get(slot).copied().unwrap_or(true) {
			return EntityId::NONE;
		}

		if let Ok(index) = u32::try_from(slot) {
			self.free.retain(|free| *free != index);
		}

		self.generations[slot] = generation;
		self.put(slot, transform, renderable)
	}

	/// Puts one entity back into a slot a [`restore`](Self::restore) has just
	/// sized the table for.
	///
	/// A generation of zero is lifted to one: zero is what
	/// [`EntityId::NONE`] holds, so a slot carrying it would hand out a handle
	/// that refers to nothing.
	fn put(&mut self, slot: usize, transform: Transform, renderable: Renderable) -> EntityId {
		let (Ok(index), Some(alive)) = (u32::try_from(slot), self.alive.get_mut(slot)) else {
			return EntityId::NONE;
		};

		if *alive {
			// two entries claiming one slot. The first one keeps it, because
			// the alternative is a handle handed out twice.
			return EntityId::NONE;
		}

		*alive = true;
		self.transforms[slot] = transform;
		self.previous[slot] = transform;
		self.renderables[slot] = renderable;
		// and it shines nothing until whoever put it back says otherwise, for
		// the reason the parent below is left alone: a restore hands the table
		// slots and plain records, and a light is set by handle afterwards.
		self.lights[slot] = Light::NONE;
		self.emitters[slot] = Emitter::NONE;
		self.terrains[slot] = Terrain::NONE;
		self.decals[slot] = Decal::NONE;
		// and nothing painted over it, by the name's rule above: a slot handed
		// out carries no stroke the last occupant's brush made.
		self.masks[slot] = None;
		// off nothing until whoever put it back says otherwise, which a
		// restore does once every record has landed. @ref `scene::restore`.
		self.parents[slot] = EntityId::NONE;
		// and shown until whoever put it back says otherwise, for the light's
		// reason: a table handed plain records has this set by handle after.
		self.hidden[slot] = false;
		self.undecaled[slot] = false;
		// and every record at its default, for the light's reason: a record is
		// written by handle afterwards, by whoever put the entity back.
		self.records.clear(slot);
		self.generations[slot] = self.generations[slot].max(1);
		self.live += 1;

		EntityId {
			index,
			generation: self.generations[slot],
		}
	}

	/// How many slots the table has ever handed out.
	///
	/// Not the same as [`len`](Self::len), which counts what is alive. This is
	/// the length of the arrays, and it is what a scene has to write down: a
	/// dead slot still carries the generation that makes a handle to it stale.
	#[must_use]
	pub fn slots(&self) -> usize { self.alive.len() }

	/// Who is in one slot, if anybody.
	///
	/// The way back from a slot number to a handle, for the reason
	/// [`Bodies::at`](crate::abi::Bodies::at) has one: a description from
	/// somewhere else names a thing by its slot, and a reader that finds that
	/// slot occupied has no other way to say which thing is in it.
	///
	/// @param slot - an index below [`slots`](Self::slots)
	/// @return the occupant, or [`EntityId::NONE`] for a slot nobody is in
	#[must_use]
	pub fn at(&self, slot: usize) -> EntityId {
		let (Ok(index), Some(true)) = (u32::try_from(slot), self.alive.get(slot).copied()) else {
			return EntityId::NONE;
		};

		EntityId { index, generation: self.generation(slot) }
	}

	/// Which occupant of a slot the table is on.
	///
	/// @param slot - the array index, not a handle
	/// @return the generation, or zero for a slot that has never been used
	#[must_use]
	pub fn generation(&self, slot: usize) -> u32 {
		self.generations.get(slot).copied().unwrap_or(0)
	}

	/// The array slot a handle refers to, if it is still the one it was given.
	fn slot(&self, id: EntityId) -> Option<usize> {
		let slot = usize::try_from(id.index).ok()?;

		(id.generation != 0
			&& self.alive.get(slot).copied().unwrap_or(false)
			&& self.generations[slot] == id.generation)
			.then_some(slot)
	}

	/// Reserves a slot, reusing a freed one before growing.
	fn take_slot(&mut self) -> Option<usize> {
		if let Some(index) = self.free.pop() {
			return usize::try_from(index).ok();
		}

		if self.alive.len() >= MAX_ENTITIES {
			return None;
		}

		self.transforms.push(Transform::IDENTITY);
		self.previous.push(Transform::IDENTITY);
		self.renderables.push(Renderable::NOTHING);
		self.lights.push(Light::NONE);
		self.emitters.push(Emitter::NONE);
		self.terrains.push(Terrain::NONE);
		self.decals.push(Decal::NONE);
		self.masks.push(None);
		self.parents.push(EntityId::NONE);
		self.hidden.push(false);
		self.undecaled.push(false);
		self.names.push();
		self.records.push();
		self.generations.push(0);
		self.alive.push(false);

		Some(self.alive.len() - 1)
	}
}

impl Default for Entities {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use std::f32::consts::{FRAC_PI_2, FRAC_PI_6, TAU};

	use super::*;

	#[test]
	fn who_is_in_a_slot_is_the_handle_that_slot_hands_back() {
		// the way back from a slot number to a handle, for the reason
		// `Bodies::at` has one: a description from another machine names a
		// thing by its slot and nothing out there can mint a handle.
		let mut entities = Entities::new();
		let first = entities.spawn();

		assert_eq!(entities.at(first.slot()), first);
		assert!(entities.despawn(first));

		// a second occupant, so a generation is not the constant one.
		let second = entities.spawn();

		assert_eq!(second.generation(), 2);
		assert_eq!(entities.at(second.slot()), second, "the one there now");
		assert_ne!(entities.at(second.slot()), first, "not the one that was");
		assert!(entities.despawn(second));
		assert_eq!(entities.at(second.slot()), EntityId::NONE, "an empty slot holds nobody");
		assert_eq!(entities.at(9999), EntityId::NONE, "nor has the table got that one");
	}

	#[test]
	fn a_spawned_entity_is_alive_and_a_despawned_one_is_not() {
		let mut entities = Entities::new();
		let id = entities.spawn();

		assert!(entities.alive(id), "spawn hands back a live handle");
		assert_eq!(entities.len(), 1, "and the count follows");
		assert!(entities.despawn(id), "despawn reports that it did something");
		assert!(!entities.alive(id), "and the handle stops resolving");
		assert!(entities.is_empty(), "and the count follows");
	}

	#[test]
	fn a_stale_handle_does_not_reach_the_entity_that_took_its_slot() {
		let mut entities = Entities::new();
		let old = entities.spawn();
		entities.despawn(old);
		let new = entities.spawn();

		assert_eq!(
			entities.capacity_left(),
			MAX_ENTITIES - 1,
			"the slot was reused rather than a new one taken"
		);
		assert!(!entities.alive(old), "the old handle is stale");
		assert!(entities.alive(new), "the new one is not");
		assert_eq!(entities.transform_mut(old), None, "and it reaches no component");
	}

	#[test]
	fn despawning_twice_reports_the_second_time_as_a_miss() {
		let mut entities = Entities::new();
		let id = entities.spawn();

		assert!(entities.despawn(id), "the first despawn does the work");
		assert!(!entities.despawn(id), "the second finds nothing to do");
	}

	#[test]
	fn a_null_handle_reaches_nothing() {
		let mut entities = Entities::new();
		entities.spawn();

		assert!(!EntityId::NONE.is_some(), "the null handle knows it is null");
		assert!(!entities.alive(EntityId::NONE), "and resolves to nothing");
		assert_eq!(entities.transform(EntityId::NONE), None, "even against slot zero");
	}

	#[test]
	fn the_table_runs_out_rather_than_growing() {
		let mut entities = Entities::new();
		for _ in 0..MAX_ENTITIES {
			assert!(entities.spawn().is_some(), "up to capacity, every spawn works");
		}

		assert_eq!(entities.spawn(), EntityId::NONE, "past it, none do");
		assert_eq!(entities.len(), MAX_ENTITIES, "and nothing was overwritten");
	}

	#[test]
	fn iteration_sees_the_living_only() {
		let mut entities = Entities::new();
		let first = entities.spawn();
		let second = entities.spawn();
		let third = entities.spawn();
		entities.despawn(second);

		let seen: Vec<EntityId> = entities.iter().map(|(id, ..)| id).collect();

		assert_eq!(seen, vec![first, third], "the hole in the middle is skipped");
	}

	#[test]
	fn clear_frees_every_slot_for_reuse() {
		let mut entities = Entities::new();
		let id = entities.spawn();
		entities.spawn();
		entities.clear();

		assert!(entities.is_empty(), "nothing is left alive");
		assert!(!entities.alive(id), "and old handles are stale");
		assert_eq!(entities.iter().count(), 0, "and iteration finds nothing");
		assert_eq!(entities.capacity_left(), MAX_ENTITIES, "the slots came back");
	}

	#[test]
	fn a_transform_between_two_steps_is_the_midpoint() {
		let mut entities = Entities::new();
		let id = entities.spawn_at(Transform::at(Vec3::ZERO));
		entities.advance();

		if let Some(transform) = entities.transform_mut(id) {
			transform.position = Vec3::new(10.0, 0.0, 0.0);
		}

		let seen = entities
			.interpolated(id, 0.5)
			.expect("the handle is live");

		assert!(
			seen.position
				.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1.0e-5),
			"halfway through the step is halfway along the move, got {}",
			seen.position
		);
	}

	#[test]
	fn the_end_of_a_step_is_exactly_where_the_game_put_it() {
		let mut entities = Entities::new();
		let id = entities.spawn();
		entities.advance();

		let placed = Transform {
			position: Vec3::new(0.1, 0.2, 0.3),
			rotation: Quat::from_rotation_y(0.7),
			scale: Vec3::splat(1.3),
		};
		if let Some(transform) = entities.transform_mut(id) {
			*transform = placed;
		}

		let seen = entities
			.interpolated(id, 1.0)
			.expect("the handle is live");

		assert_eq!(
			seen.position.to_array().map(f32::to_bits),
			placed.position.to_array().map(f32::to_bits),
			"a frame at the end of a step has to match one that never interpolated at all, or \
			 every pixel test starts drifting"
		);
		assert_eq!(
			seen.rotation.to_array().map(f32::to_bits),
			placed.rotation.to_array().map(f32::to_bits),
			"and rotation especially: slerp ends with a normalize, so this is only true because \
			 of the fast path"
		);
	}

	#[test]
	fn a_rotation_between_two_steps_turns_at_an_even_rate() {
		let quarter = Transform::IDENTITY.lerp(
			Transform {
				rotation: Quat::from_rotation_y(TAU / 3.0),
				..Transform::IDENTITY
			},
			0.25,
		);

		// a quarter of the way through a hundred-and-twenty degree turn is
		// thirty degrees. A plain lerp of the quaternion, normalized, gives
		// 27.8 - close enough to look right in a screenshot and wrong enough
		// to see on something spinning fast.
		let turned = Quat::IDENTITY.angle_between(quarter.rotation);

		assert!(
			(turned - FRAC_PI_6).abs() < 0.01,
			"the arc is walked at a constant rate, not chorded: got {turned} radians"
		);
	}

	#[test]
	fn a_scale_between_two_steps_is_interpolated_like_everything_else() {
		let grown = Transform::IDENTITY.lerp(
			Transform {
				scale: Vec3::splat(3.0),
				..Transform::IDENTITY
			},
			0.5,
		);

		assert!(
			grown.scale.abs_diff_eq(Vec3::splat(2.0), 1.0e-5),
			"a thing that doubles over a step is drawn part-grown, got {}",
			grown.scale
		);
	}

	#[test]
	fn a_snapped_entity_is_drawn_where_it_landed() {
		let mut entities = Entities::new();
		let id = entities.spawn_at(Transform::at(Vec3::ZERO));
		entities.advance();

		if let Some(transform) = entities.transform_mut(id) {
			transform.position = Vec3::new(50.0, 0.0, 0.0);
		}
		entities.snap(id);
		// a step's worth of ordinary movement after the jump, and written
		// after the snap on purpose: the whole point of settling at the end is
		// that the order does not matter.
		if let Some(transform) = entities.transform_mut(id) {
			transform.position.x += 1.0;
		}
		entities.settle();

		let seen = entities
			.interpolated(id, 0.25)
			.expect("the handle is live");

		assert!(
			seen.position
				.abs_diff_eq(Vec3::new(51.0, 0.0, 0.0), 1.0e-5),
			"the fifty units are not a journey; the one unit that was is worth less than making \
			 that true, got {}",
			seen.position
		);
	}

	#[test]
	fn an_entity_spawned_mid_step_does_not_arrive_from_the_origin() {
		let mut entities = Entities::new();
		entities.advance();

		// the shape everyone writes: spawn, then put it somewhere.
		let id = entities.spawn();
		if let Some(transform) = entities.transform_mut(id) {
			transform.position = Vec3::new(0.0, 100.0, 0.0);
		}
		entities.settle();

		let seen = entities
			.interpolated(id, 0.5)
			.expect("the handle is live");

		assert!(
			seen.position
				.abs_diff_eq(Vec3::new(0.0, 100.0, 0.0), 1.0e-5),
			"a thing that did not exist a step ago has nowhere to fly in from, got {}",
			seen.position
		);
	}

	#[test]
	fn a_reused_slot_does_not_inherit_the_dead_entity_past() {
		let mut entities = Entities::new();
		let old = entities.spawn_at(Transform::at(Vec3::new(-99.0, 0.0, 0.0)));
		entities.advance();
		entities.despawn(old);

		let new = entities.spawn_at(Transform::at(Vec3::new(3.0, 0.0, 0.0)));
		entities.settle();

		let seen = entities
			.interpolated(new, 0.5)
			.expect("the handle is live");

		assert!(
			seen.position
				.abs_diff_eq(Vec3::new(3.0, 0.0, 0.0), 1.0e-5),
			"the slot came back, its history did not, got {}",
			seen.position
		);
	}

	#[test]
	fn advancing_forgets_what_settling_was_going_to_do() {
		let mut entities = Entities::new();
		let id = entities.spawn_at(Transform::at(Vec3::ZERO));
		entities.snap(id);
		// a new step, so a new pair of poses to interpolate between. A snap
		// aimed at the old pair has nothing left to say about this one.
		entities.advance();

		if let Some(transform) = entities.transform_mut(id) {
			transform.position = Vec3::new(4.0, 0.0, 0.0);
		}
		entities.settle();

		let seen = entities
			.interpolated(id, 0.5)
			.expect("the handle is live");

		assert!(
			seen.position
				.abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1.0e-5),
			"an old snap must not go on suppressing this step's movement, got {}",
			seen.position
		);
	}

	#[test]
	fn snapping_the_table_covers_everything_in_it() {
		let mut entities = Entities::new();
		let first = entities.spawn();
		let second = entities.spawn();
		entities.advance();

		for id in [first, second] {
			if let Some(transform) = entities.transform_mut(id) {
				transform.position = Vec3::new(7.0, 0.0, 0.0);
			}
		}

		entities.snap_all();
		entities.settle();

		for id in [first, second] {
			let seen = entities
				.interpolated(id, 0.0)
				.expect("the handle is live");

			assert!(
				seen.position
					.abs_diff_eq(Vec3::new(7.0, 0.0, 0.0), 1.0e-5),
				"a scene that cut leaves nothing still traveling, got {}",
				seen.position
			);
		}
	}

	#[test]
	fn a_stale_handle_snaps_nothing_and_says_so() {
		let mut entities = Entities::new();
		let id = entities.spawn();
		entities.despawn(id);

		assert!(!entities.snap(id), "a miss is a false, the way set_renderable is");
		assert_eq!(entities.interpolated(id, 0.5), None, "and it places nothing");
	}

	#[test]
	fn every_component_array_stays_the_same_length() {
		let mut entities = Entities::new();
		let mut ids = Vec::new();
		for _ in 0..8 {
			ids.push(entities.spawn());
		}

		entities.despawn(ids[3]);
		entities.spawn();
		entities.clear();
		entities.spawn();

		// @note: deliberately no `advance()` before this. `clone_from` would
		// paper over a `take_slot` that forgot one of the arrays, and the
		// failure that hides is an index panic in the host, outside the
		// `catch_unwind` that contains the game.
		let length = entities.transforms.len();

		assert_eq!(entities.previous.len(), length, "the two transform arrays are one array");
		assert_eq!(entities.renderables.len(), length, "and the rest of the table agrees");
		assert_eq!(entities.alive.len(), length, "and the rest of the table agrees");
		assert_eq!(entities.generations.len(), length, "and the rest of the table agrees");
		assert_eq!(entities.names.slots(), length, "and the rest of the table agrees");
		assert_eq!(entities.parents.len(), length, "and the rest of the table agrees");
		assert_eq!(entities.hidden.len(), length, "and the rest of the table agrees");
		assert_eq!(entities.decals.len(), length, "and the rest of the table agrees");
		assert_eq!(entities.undecaled.len(), length, "and the rest of the table agrees");
	}

	/// A parent that is turned, scaled and moved, so that every part of a
	/// composition has something to get wrong.
	fn awkward() -> Transform {
		Transform {
			position: Vec3::new(3.0, -1.0, 2.0),
			rotation: Quat::from_rotation_y(FRAC_PI_2),
			scale: Vec3::splat(2.0),
		}
	}

	#[test]
	fn a_child_stands_where_its_parent_puts_it() {
		let parent = awkward();
		let placed = parent.then(Transform::at(Vec3::X));

		// a step of one along the parent's x, which its quarter turn about y
		// points down negative z, at twice the size, from where it stands
		assert!(
			placed
				.position
				.abs_diff_eq(Vec3::new(3.0, -1.0, 0.0), 1.0e-5),
			"got {}",
			placed.position
		);
		assert!(placed.scale.abs_diff_eq(Vec3::splat(2.0), 1.0e-6), "the scale multiplies");
		assert!(turns(placed.rotation, parent.rotation, 1.0e-6), "and the turn is the parent's");
	}

	#[test]
	fn local_of_undoes_then_under_an_awkward_parent() {
		let parent = awkward();
		let world = Transform {
			position: Vec3::new(-4.0, 5.0, 1.5),
			rotation: Quat::from_rotation_x(FRAC_PI_6),
			scale: Vec3::new(1.0, 3.0, 0.5),
		};
		let local = Transform::local_of(world, parent);
		let back = parent.then(local);

		assert!(back.position.abs_diff_eq(world.position, 1.0e-4), "got {}", back.position);
		assert!(turns(back.rotation, world.rotation, 1.0e-5), "the turn comes back");
		assert!(back.scale.abs_diff_eq(world.scale, 1.0e-5), "got {}", back.scale);
		assert!(
			!local.position.abs_diff_eq(world.position, 1.0e-3),
			"and the local really is another number, or this proves nothing"
		);
	}

	#[test]
	fn a_child_is_placed_through_its_parent_and_a_root_where_it_says() {
		let mut entities = Entities::new();
		let parent = entities.spawn_at(awkward());
		let child = entities.spawn_at(Transform::at(Vec3::X));

		assert!(entities.set_parent(child, parent), "it hangs");
		assert_eq!(entities.parent(child), parent);
		assert_eq!(entities.parent(parent), EntityId::NONE, "and the parent is a root");

		let placed = entities.placed(child).expect("alive");

		assert!(
			placed
				.position
				.abs_diff_eq(Vec3::new(3.0, -1.0, 0.0), 1.0e-5),
			"got {placed:?}"
		);
		assert_eq!(
			entities.transform(child).copied(),
			Some(Transform::at(Vec3::X)),
			"while its own transform is untouched and now means inside the parent"
		);
		assert_eq!(entities.placed(parent), Some(awkward()), "a root is placed where it says");
	}

	#[test]
	fn set_placed_writes_the_local_that_lands_there() {
		let mut entities = Entities::new();
		let parent = entities.spawn_at(awkward());
		let child = entities.spawn();
		assert!(entities.set_parent(child, parent));

		let wanted = Transform::at(Vec3::splat(7.0));

		assert!(entities.set_placed(child, wanted));

		let placed = entities.placed(child).expect("alive");

		assert!(
			placed
				.position
				.abs_diff_eq(wanted.position, 1.0e-4),
			"got {placed:?}"
		);
		assert!(
			!entities
				.transform(child)
				.expect("alive")
				.position
				.abs_diff_eq(wanted.position, 1.0e-3),
			"by way of a local that is not the world position"
		);

		let lone = entities.spawn();

		assert!(entities.set_placed(lone, wanted));
		assert_eq!(entities.transform(lone).copied(), Some(wanted), "a root takes it as it is");
	}

	#[test]
	fn a_parent_that_is_itself_a_descendant_or_dead_is_refused() {
		let mut entities = Entities::new();
		let a = entities.spawn();
		let b = entities.spawn();
		let c = entities.spawn();

		assert!(entities.set_parent(b, a));
		assert!(entities.set_parent(c, b));

		assert!(!entities.set_parent(a, a), "itself");
		assert!(!entities.set_parent(a, c), "a loop three long");
		assert!(!entities.set_parent(a, b), "a loop two long");
		assert_eq!(entities.parent(a), EntityId::NONE, "and nothing was written");

		let gone = entities.spawn();
		assert!(entities.despawn(gone));

		assert!(!entities.set_parent(a, gone), "a dead parent");
		assert!(!entities.set_parent(gone, a), "or a dead child");
		assert!(entities.set_parent(c, EntityId::NONE), "taking down is always allowed");
		assert_eq!(entities.parent(c), EntityId::NONE);
	}

	#[test]
	fn a_parent_that_died_is_no_parent_and_its_slot_reused_is_not_one_either() {
		let mut entities = Entities::new();
		let parent = entities.spawn_at(Transform::at(Vec3::Y));
		let child = entities.spawn_at(Transform::at(Vec3::X));
		assert!(entities.set_parent(child, parent));
		assert!(entities.despawn(parent));

		assert_eq!(entities.parent(child), EntityId::NONE, "gone");
		assert_eq!(
			entities.placed(child),
			Some(Transform::at(Vec3::X)),
			"and the child stands on its own"
		);

		// the slot comes back with another occupant, which is not the parent
		let stranger = entities.spawn_at(Transform::at(Vec3::Z * 9.0));

		assert_eq!(
			stranger.slot(),
			parent.slot(),
			"the fixture reuses the slot, or it proves nothing"
		);
		assert_eq!(
			entities.parent(child),
			EntityId::NONE,
			"a stranger in the slot is nobody's parent"
		);
		assert_eq!(
			entities.parent(stranger),
			EntityId::NONE,
			"and a reused slot hangs off nothing"
		);
	}

	#[test]
	fn hiding_something_hides_everything_under_it_and_nothing_beside_it() {
		// three deep and two wide. Hiding the root is the case a walk that
		// looked at the parent alone gets wrong, the grandchild's own parent
		// not being hidden; hiding the middle is the case a walk that looked at
		// the root alone gets wrong.
		let mut entities = Entities::new();
		let root = entities.spawn();
		let middle = entities.spawn();
		let beside = entities.spawn();
		let under = entities.spawn();
		assert!(entities.set_parent(middle, root));
		assert!(entities.set_parent(beside, root));
		assert!(entities.set_parent(under, middle));

		assert!(entities.set_hidden(root, true), "the handle resolves");
		assert!(!entities.shown(under), "the grandchild of something hidden is hidden");
		assert!(
			!entities.shown(middle) && !entities.shown(beside),
			"and so is every child of it"
		);
		assert!(!entities.hidden(under), "by the root's word, not by one of its own");

		assert!(entities.set_hidden(root, false));
		assert!(entities.set_hidden(middle, true));
		assert!(
			entities.shown(root) && entities.shown(beside),
			"above it and beside it are drawn"
		);
		assert!(
			!entities.shown(middle) && !entities.shown(under),
			"the middle and what hangs off it are not"
		);
	}

	#[test]
	fn hanging_something_under_a_hidden_thing_hides_it_and_taking_it_down_shows_it() {
		// the case a kept answer goes stale on: nothing about the child changes,
		// only what it hangs off
		let mut entities = Entities::new();
		let hidden = entities.spawn();
		let child = entities.spawn();
		assert!(entities.set_hidden(hidden, true));

		assert!(entities.shown(child), "standing on its own it is drawn");
		assert!(entities.set_parent(child, hidden));
		assert!(!entities.shown(child), "hung under something hidden it is not");
		assert!(entities.set_parent(child, EntityId::NONE));
		assert!(entities.shown(child), "and taken down it is again");
	}

	#[test]
	fn a_hidden_parent_that_died_hides_nothing_and_a_stale_handle_is_not_shown() {
		let mut entities = Entities::new();
		let parent = entities.spawn();
		let child = entities.spawn();
		assert!(entities.set_parent(child, parent));
		assert!(entities.set_hidden(parent, true));
		assert!(entities.despawn(parent));

		assert!(entities.shown(child), "a dead parent is no parent, hidden or not");
		assert!(!entities.shown(parent), "and a stale handle is drawn by nobody");
		assert!(!entities.hidden(parent), "nor says it is hidden");
		assert!(!entities.set_hidden(parent, true), "nor can be hidden");
	}

	#[test]
	fn a_slot_handed_out_again_is_shown_however_it_is_handed_out() {
		// the ways a slot comes back into use, each of which has to clear what
		// the last occupant said, and each tried on a slot that was hidden
		let mut entities = Entities::new();
		let old = entities.spawn();
		assert!(entities.set_hidden(old, true));
		assert!(entities.despawn(old));

		let spawned = entities.spawn();
		assert_eq!(
			spawned.slot(),
			old.slot(),
			"the fixture reuses the slot, or it proves nothing"
		);
		assert!(entities.shown(spawned), "spawned into it");

		assert!(entities.set_hidden(spawned, true));
		assert!(entities.despawn(spawned));
		let grafted = entities.graft(old.slot(), 9, Transform::IDENTITY, Renderable::NOTHING);
		assert!(grafted.is_some(), "the slot was free to graft into");
		assert!(entities.shown(grafted), "grafted into it");

		assert!(entities.set_hidden(grafted, true));
		let put = entities.restore(&[4], &[(0, Transform::IDENTITY, Renderable::NOTHING)]);
		assert!(entities.shown(put[0]), "restored into it");
	}

	#[test]
	fn a_graft_that_grows_the_table_grows_every_array_by_one_a_slot() {
		// the path the length test above does not take: a piece arriving into a
		// slot past the end of a table that has never been that long
		let mut entities = Entities::new();
		let grafted = entities.graft(5, 3, Transform::IDENTITY, Renderable::NOTHING);

		assert!(grafted.is_some(), "the slot was free to graft into");
		assert_eq!(entities.alive.len(), 6, "slots nought to five");
		assert_eq!(entities.terrains.len(), 6, "and the ground agrees");
		assert_eq!(entities.decals.len(), 6, "and so do the decals");
		assert_eq!(entities.undecaled.len(), 6, "and whether each takes one");
		assert_eq!(entities.hidden.len(), 6, "and the rest of the table");
	}

	#[test]
	fn a_decal_belongs_to_its_entity_and_a_stale_handle_has_none() {
		let mut entities = Entities::new();
		let wall = entities.spawn();
		let puddle = entities.spawn();
		let painted = Decal { order: 4, ..Decal::BOX };

		assert!(entities.set_decal(puddle, painted), "the handle resolves");
		assert_eq!(entities.decal(puddle).copied(), Some(painted), "and hands it back");
		assert_eq!(entities.decal(wall).copied(), Some(Decal::NONE), "and nobody else has it");

		assert!(entities.despawn(puddle));
		assert!(entities.decal(puddle).is_none(), "a stale handle has none");
		assert!(!entities.set_decal(puddle, painted), "and cannot be given one");
	}

	#[test]
	fn taking_decals_is_a_word_of_the_entity_itself_and_its_children_keep_theirs() {
		let mut entities = Entities::new();
		let parent = entities.spawn();
		let child = entities.spawn();
		assert!(entities.set_parent(child, parent));

		assert!(entities.takes_decals(parent), "an entity takes decals until it says not");
		assert!(entities.set_takes_decals(parent, false), "the handle resolves");
		assert!(!entities.takes_decals(parent), "and then it takes none");
		assert!(entities.takes_decals(child), "while what hangs off it keeps its own word");

		assert!(entities.despawn(child));
		assert!(!entities.takes_decals(child), "and a stale handle has no surface to paint");
		assert!(!entities.set_takes_decals(child, true), "nor can be told to");
	}

	#[test]
	fn a_slot_handed_out_again_paints_nothing_and_takes_decals_however_it_is_handed_out() {
		// the hidden word's test, for the two words this card added: each way a
		// slot comes back into use, tried on a slot that was a decal refusing
		// decals, which nothing clears on the way out
		let mut entities = Entities::new();
		let old = entities.spawn();
		let mark = |entities: &mut Entities, id: EntityId| {
			assert!(entities.set_decal(id, Decal::BOX));
			assert!(entities.set_takes_decals(id, false));
		};

		mark(&mut entities, old);
		assert!(entities.despawn(old));

		let spawned = entities.spawn();
		assert_eq!(
			spawned.slot(),
			old.slot(),
			"the fixture reuses the slot, or it proves nothing"
		);
		assert_eq!(entities.decal(spawned).copied(), Some(Decal::NONE), "spawned into it");
		assert!(entities.takes_decals(spawned), "and takes decals");

		mark(&mut entities, spawned);
		assert!(entities.despawn(spawned));
		let grafted = entities.graft(old.slot(), 9, Transform::IDENTITY, Renderable::NOTHING);
		assert!(grafted.is_some(), "the slot was free to graft into");
		assert_eq!(entities.decal(grafted).copied(), Some(Decal::NONE), "grafted into it");
		assert!(entities.takes_decals(grafted), "and takes decals");

		mark(&mut entities, grafted);
		let put = entities.restore(&[4], &[(0, Transform::IDENTITY, Renderable::NOTHING)]);
		assert_eq!(entities.decal(put[0]).copied(), Some(Decal::NONE), "restored into it");
		assert!(entities.takes_decals(put[0]), "and takes decals");
	}

	#[test]
	fn a_slot_handed_out_again_carries_every_record_at_its_default_however_it_is_handed_out() {
		// the hidden word's test a third time, for what a record adds: each way a
		// slot comes back into use, tried on a slot whose record was written and
		// which had something waiting for a record nobody declared
		let mut entities = Entities::new();
		entities
			.declare(&DRAWING)
			.expect("drawing is a record a world holds");

		let waits = [Noted {
			record: "door".to_owned(),
			field: "open".to_owned(),
			value: super::super::record::Spelled::Truth(true),
		}];
		let mark = |entities: &mut Entities, id: EntityId| {
			if let Some(drawing) = entities.record_mut(&DRAWING, id) {
				drawing.covers = 1;
			}

			assert!(entities.note(id, &waits).is_empty(), "a door nobody declared waits");
			assert_eq!(entities.noted(id).len(), 2, "and both are there to write down");
		};
		let fresh = |entities: &Entities, id: EntityId, how: &str| {
			assert_eq!(entities.record(&DRAWING, id), Some(&Drawing::NONE), "{how}: the default");
			assert!(entities.waiting(id).is_empty(), "{how}: and nothing waits");
		};

		let old = entities.spawn();
		mark(&mut entities, old);
		assert!(entities.despawn(old));
		assert!(entities.record(&DRAWING, old).is_none(), "a stale handle reaches no record");
		assert!(entities.noted(old).is_empty(), "and has nothing to write down");

		let spawned = entities.spawn();
		assert_eq!(
			spawned.slot(),
			old.slot(),
			"the fixture reuses the slot, or it proves nothing"
		);
		fresh(&entities, spawned, "spawned into it");

		mark(&mut entities, spawned);
		assert!(entities.despawn(spawned));
		let grafted = entities.graft(old.slot(), 9, Transform::IDENTITY, Renderable::NOTHING);
		assert!(grafted.is_some(), "the slot was free to graft into");
		fresh(&entities, grafted, "grafted into it");

		mark(&mut entities, grafted);
		let put = entities.restore(&[4], &[(0, Transform::IDENTITY, Renderable::NOTHING)]);
		fresh(&entities, put[0], "restored into it");

		let far = entities.graft(6, 2, Transform::IDENTITY, Renderable::NOTHING);
		assert_eq!(
			entities.column(&DRAWING).map(<[Drawing]>::len),
			Some(7),
			"a graft past the end grows the record with every other array"
		);
		fresh(&entities, far, "grafted past the end");
		assert!(
			!entities.set_field(old, 0, 0, &Value::Bool(true)),
			"and a stale handle writes no field"
		);
		assert!(entities.set_field(far, 0, 0, &Value::Bool(true)), "where a live one does");
		assert_eq!(entities.field(far, 0, 0), Some(Value::Bool(true)), "and reads it back");
		assert!(
			entities
				.record(&DRAWING, far)
				.is_some_and(|it| it.covers()),
			"as the struct"
		);
	}

	#[test]
	fn a_child_is_drawn_between_where_its_parent_was_and_is() {
		let mut entities = Entities::new();
		let parent = entities.spawn_at(Transform::IDENTITY);
		let child = entities.spawn_at(Transform::at(Vec3::X));
		assert!(entities.set_parent(child, parent));
		entities.advance();

		if let Some(at) = entities.transform_mut(parent) {
			at.position = Vec3::new(0.0, 10.0, 0.0);
		}

		let halfway = entities.blended(child, 0.5).expect("alive");

		assert!(
			halfway
				.position
				.abs_diff_eq(Vec3::new(1.0, 5.0, 0.0), 1.0e-5),
			"got {halfway:?}"
		);
		assert_eq!(
			entities.interpolated(child, 0.5),
			Some(Transform::at(Vec3::X)),
			"while its own transform did not move"
		);
	}

	#[test]
	fn a_transform_turns_into_the_matrix_it_describes() {
		let transform = Transform {
			position: Vec3::new(1.0, 2.0, 3.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::splat(2.0),
		};

		let moved = transform
			.matrix()
			.transform_point3(Vec3::new(1.0, 0.0, 0.0));

		assert!(
			moved.abs_diff_eq(Vec3::new(3.0, 2.0, 3.0), 1.0e-5),
			"scale is applied before the translation, not after"
		);
	}

	#[test]
	fn a_spawned_entity_draws_nothing_until_told_otherwise() {
		let mut entities = Entities::new();
		let id = entities.spawn();

		assert_eq!(
			entities.renderable(id),
			Some(&Renderable::NOTHING),
			"spawning is not the same as appearing"
		);
		assert!(!MeshId::NONE.is_some(), "and the null mesh knows it is null");

		entities.set_renderable(id, Renderable::new(MeshId::CUBE, Vec3::X));

		assert_eq!(
			entities.renderable(id).map(|it| it.mesh),
			Some(MeshId::CUBE),
			"and a shape can be given afterwards"
		);
	}

	#[test]
	fn an_entity_is_unnamed_until_it_is_named() {
		let mut entities = Entities::new();
		let id = entities.spawn();

		assert_eq!(entities.name(id), "", "spawning is not the same as being called something");
		assert!(entities.set_name(id, "crate"), "naming reports that it did something");
		assert_eq!(entities.name(id), "crate");
	}

	#[test]
	fn a_reused_slot_is_not_called_what_was_there_before() {
		let mut entities = Entities::new();
		let old = entities.spawn();
		entities.set_name(old, "crate");
		entities.despawn(old);

		let new = entities.spawn();

		assert_eq!(entities.name(old), "", "the stale handle reaches nothing at all");
		assert_eq!(
			entities.name(new),
			"",
			"and whoever took the slot did not inherit the name with it"
		);
	}

	#[test]
	fn naming_a_stale_handle_does_nothing_and_says_so() {
		let mut entities = Entities::new();
		let id = entities.spawn();
		entities.despawn(id);

		assert!(!entities.set_name(id, "ghost"), "a stale handle names nothing");
		assert_eq!(entities.name(id), "");
	}

	#[test]
	fn a_slot_freed_by_a_clear_is_unnamed_when_it_comes_back() {
		let mut entities = Entities::new();
		let first = entities.spawn();
		let second = entities.spawn();
		entities.set_name(first, "crate");
		entities.set_name(second, "floor");

		// the other way a slot reaches the free list. A despawn is covered by
		// the test above; nothing clears a name in either path, so what is
		// being checked is that handing the slot out again does.
		entities.clear();

		let taken = entities.spawn();
		assert_eq!(entities.name(taken), "", "a slot handed out after a clear is unnamed");
	}

	#[test]
	fn a_restore_leaves_every_slot_unnamed_for_the_caller_to_write() {
		let mut entities = Entities::new();
		let id = entities.spawn();
		entities.set_name(id, "crate");

		let put = entities.restore(&[4, 1], &[(0, Transform::IDENTITY, Renderable::NOTHING)]);

		assert_eq!(
			entities.name(put[0]),
			"",
			"a restore sizes the array and writes no names; the names come after"
		);
		assert!(entities.set_name(put[0], "barrel"), "and the slot is there to be written");
		assert_eq!(entities.name(put[0]), "barrel");
	}

	/// Whether two rotations do the same thing to the three axes.
	fn turns(one: Quat, other: Quat, within: f32) -> bool {
		[Vec3::X, Vec3::Y, Vec3::Z]
			.into_iter()
			.all(|axis| (one * axis).abs_diff_eq(other * axis, within))
	}

	#[test]
	fn a_transform_read_out_of_its_own_matrix_is_the_transform_again() {
		let there = Transform {
			position: Vec3::new(1.5, -2.0, 0.25),
			rotation: Quat::from_rotation_y(0.9) * Quat::from_rotation_x(-0.4),
			scale: Vec3::splat(2.0),
		};
		let back = Transform::from_matrix(there.matrix());

		assert!(back.position.abs_diff_eq(there.position, 1e-5), "the place comes back");
		// what the rotation *does*, rather than `angle_between`. Near zero that
		// function is a square root of the error, so two quaternions one f32
		// step apart come out about seven ten-thousandths of a radian apart
		// and any tolerance tight enough to mean something fails.
		assert!(
			turns(back.rotation, there.rotation, 1e-5),
			"and the turn, got {} against {}",
			back.rotation,
			there.rotation
		);
		assert!(back.scale.abs_diff_eq(there.scale, 1e-5), "and the size");
	}

	#[test]
	fn a_matrix_of_two_transforms_reads_apart_as_the_pair_composed() {
		// what a ragdoll actually does with this: a body's place in the world is
		// two matrices multiplied and then read back out as one transform.
		let outer = Transform {
			position: Vec3::new(0.0, 3.0, 0.0),
			rotation: Quat::from_rotation_z(FRAC_PI_2),
			scale: Vec3::ONE,
		};
		let inner = Transform::at(Vec3::X);
		let both = Transform::from_matrix(outer.matrix() * inner.matrix());

		assert!(
			both.position
				.abs_diff_eq(Vec3::new(0.0, 4.0, 0.0), 1e-5),
			"the inner offset is turned by the outer rotation, got {}",
			both.position
		);
		assert!(
			turns(both.rotation, outer.rotation, 1e-5),
			"and the turn is the outer one, the inner having none"
		);
	}
}
