//! Bodies, and the two queries a game asks the world about them.
//!
//! Three decisions are written into the shape of this module, and each was
//! taken against a plausible alternative.
//!
//! **The body table is plain data in the host, exactly like the entity
//! table.** Not a handle into somebody else's world. The invariant the whole
//! engine stands on is that every byte the game can see lives in
//! [`World`](crate::abi::World) and survives a module swap untouched, and a
//! physics library that owns the authoritative transform breaks it - the editor
//! cannot inspect what it cannot reach, and a sandbox that wants to save a
//! scene cannot serialize it. The way back from that is mirroring, and it is
//! paid for in the obvious way: every entity's position, rotation and velocity
//! written into the library's own world every tick and read back out again,
//! which is how a pile of props ends up never properly sleeping.
//!
//! **The solver is not in this crate, and neither is anything that needs an
//! acceleration structure.** `colby_core` must not depend on a physics library,
//! and it does not depend on `colby_physics` either. What crosses is
//! [`Physics`], a `#[repr(C)]` table of two function pointers the host fills in
//! at startup, pointing across the boundary the other way from the table a
//! game module exports. Two, because a query is the only part that needs more
//! than the body table already says: a broadphase, and a baked collision mesh.
//! Spawning a body, moving it, or making it kinematic are writes into plain
//! data and need no pointer at all, which is why this table does not grow.
//!
//! **The table survives a reload because of which way the pointers point.**
//! They address `colby.exe`, which is never unloaded, so a module swap does not
//! touch them; the host installs them once and never again. Contrast
//! [`cvar`](crate::abi::cvar), where a registered command points *into* the
//! module's image and has to be forgotten before `FreeLibrary`. The direction
//! is the whole of the difference, and it is worth knowing which one is being
//! looked at before reasoning about either.
//!
//! A [`World`](crate::abi::World) nobody wired up holds [`Physics::STUB`],
//! whose queries report a clean miss. That is the same discipline as
//! [`MeshId::NONE`] and `PanelId::NONE`: a call through something that never
//! resolved changes nothing rather than having to be checked at every call
//! site.

use core::ffi::c_void;

use super::{entity::EntityId, mesh::MeshId, names::Names};
use crate::{
	abi::Transform,
	glam::{Mat3, Quat, Vec3},
};

/// How many bodies can exist at once.
///
/// Bounded for the reason the entity table is bounded: the gameplay crate is
/// code that is *expected* to be wrong sometimes, and a reload that spawns a
/// prop every step should run out of slots rather than out of memory.
pub const MAX_BODIES: usize = 1024;

/// How many bodies one query can be told to pretend are not there.
///
/// A fixed array rather than a pointer and a count: this ABI has no raw
/// pointers in it anywhere except [`Physics`] itself, and that is a property
/// worth more than the generality. Eight is what a sandbox needs: the prop in
/// the hand, the thing it is welded to, and the player holding both.
pub const MAX_IGNORED: usize = 8;

/// Which layers a thing is on, and which of them it interacts with.
///
/// Two bitmasks rather than one number, because "what am I" and "what do I
/// care about" are different questions and a single layer number can only
/// answer the first. A prop is on the prop layer and interacts with
/// everything; a trigger looking for players is on the trigger layer and
/// interacts with the player layer alone.
///
/// **The rule is symmetric: both sides have to agree.** A meets B only when
/// A's layer is in B's mask *and* B's layer is in A's mask. The alternative -
/// either side being enough - produces a pair that one body pushes and the
/// other passes through, which is a contradiction rather than a filter.
///
/// Bit `n` is layer `n`, and [`bit`](Self::bit) is how to name one without
/// writing a shift that is undefined past thirty-one.
#[repr(C)]
#[derive(
	Clone, Copy, Debug, PartialEq, Eq, Hash, crate::bytemuck::Pod, crate::bytemuck::Zeroable,
)]
pub struct Layers {
	/// The layers this is on. Usually one bit.
	pub layer: u32,

	/// The layers it interacts with.
	pub mask: u32,
}

impl Layers {
	/// On every layer, interacting with every layer.
	///
	/// What a query is unless it says otherwise: a trace has no layer of its
	/// own to be filtered by, so it claims all of them and nobody's mask
	/// excludes it.
	pub const ALL: Self = Self { layer: u32::MAX, mask: u32::MAX };
	/// On layer zero, interacting with every layer.
	///
	/// What a body is unless it says otherwise, and what every body in a world
	/// that has never heard of layers is. Note this is deliberately *not*
	/// [`ALL`](Self::ALL): a body on every layer at once cannot be filtered out
	/// by narrowing a mask, which is the one thing masks are for.
	pub const DEFAULT: Self = Self { layer: 1, mask: u32::MAX };
	/// On nothing, interacting with nothing.
	///
	/// What a zeroed [`Layers`] reads as, which is why the zero value is inert
	/// rather than universal: a handle read out of a freshly zeroed arena
	/// should do nothing, not everything.
	pub const NONE: Self = Self { layer: 0, mask: 0 };

	/// The bit that stands for one layer.
	///
	/// Wraps rather than overflowing, because a shift past thirty-one is
	/// undefined and a constant somebody typed is not worth a panic.
	///
	/// @param index - which layer, `0 ..= 31`
	#[must_use]
	pub const fn bit(index: u32) -> u32 { 1_u32 << (index % u32::BITS) }

	/// On one layer, interacting with every layer.
	///
	/// @param index - which layer, `0 ..= 31`
	#[must_use]
	pub const fn single(index: u32) -> Self { Self { layer: Self::bit(index), mask: u32::MAX } }

	/// On some layers, interacting with some layers.
	///
	/// @param layer - the bits it is on
	/// @param mask - the bits it interacts with
	#[must_use]
	pub const fn new(layer: u32, mask: u32) -> Self { Self { layer, mask } }

	/// The same layers, interacting with something else.
	///
	/// @param mask - the bits to interact with
	#[must_use]
	pub const fn interacting(mut self, mask: u32) -> Self {
		self.mask = mask;

		self
	}

	/// Whether these two have anything to say to each other.
	///
	/// @param other - the layers on the far side
	#[must_use]
	pub const fn meets(self, other: Self) -> bool {
		self.layer & other.mask != 0 && other.layer & self.mask != 0
	}
}

impl Default for Layers {
	fn default() -> Self { Self::DEFAULT }
}

/// Which of the three shapes a body is.
///
/// Three of them, because a box, a ball and a triangle soup cover everything a
/// prop can be, and a fourth is a solver problem rather than a modeling one.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ShapeKind {
	/// A box, `extents` half-wide on each axis.
	#[default]
	Box,

	/// A ball of `radius`.
	Sphere,

	/// The triangles of `mesh`.
	Mesh,
}

/// What a body is shaped like.
///
/// One struct covering all three kinds, with the fields the kind does not use
/// left alone. A shape crosses the boundary as plain data, and a tagged union
/// that has to be read through an accessor buys nothing over four words that
/// are sometimes zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
	/// Which of the three this is.
	pub kind: ShapeKind,

	/// The radius of a [`ShapeKind::Sphere`].
	pub radius: f32,

	/// The half-extents of a [`ShapeKind::Box`].
	pub extents: Vec3,

	/// The geometry of a [`ShapeKind::Mesh`].
	///
	/// Read once, when the body is first seen by the solver, and baked into a
	/// collision mesh of its own from then on. Recompiling the `.obj` therefore
	/// changes the picture and **not** the collision until the body is created
	/// again - deliberate, because a collision mesh is a physics resource with
	/// its own preparation, not a second view onto a vertex buffer.
	pub mesh: MeshId,
}

impl Shape {
	/// A unit cube, one wide on every axis.
	pub const UNIT: Self = Self::cuboid(Vec3::splat(0.5));

	/// A ball.
	///
	/// @param radius - how far the surface is from the middle
	#[must_use]
	pub const fn ball(radius: f32) -> Self {
		Self {
			kind: ShapeKind::Sphere,
			radius,
			extents: Vec3::ZERO,
			mesh: MeshId::NONE,
		}
	}

	/// A box.
	///
	/// @param extents - **half**-extents, so a unit cube is `Vec3::splat(0.5)`
	#[must_use]
	pub const fn cuboid(extents: Vec3) -> Self {
		Self {
			kind: ShapeKind::Box,
			radius: 0.0,
			extents,
			mesh: MeshId::NONE,
		}
	}

	/// The triangles of a mesh, exactly as they are.
	///
	/// @param mesh - what to collide against
	#[must_use]
	pub const fn mesh(mesh: MeshId) -> Self {
		Self {
			kind: ShapeKind::Mesh,
			radius: 0.0,
			extents: Vec3::ZERO,
			mesh,
		}
	}

	/// A box around a mesh, from the bounds the mesh reports.
	///
	/// The usual way to give a prop collision: the geometry is convex enough
	/// that its box is what a person would have typed, and a box is the shape
	/// every query handles exactly.
	///
	/// @param min - the low corner of the mesh's bounds
	/// @param max - the high corner
	/// @return the box, and where its middle sits relative to the mesh's origin
	#[must_use]
	pub fn around(min: Vec3, max: Vec3) -> (Self, Vec3) {
		let extents = ((max - min) * 0.5).max(Vec3::ZERO);

		(Self::cuboid(extents), (min + max) * 0.5)
	}

	/// The half-extents of the smallest axis-aligned box holding this shape,
	/// before any rotation.
	///
	/// A mesh reports nothing, because its size is not in this struct; the
	/// solver knows it and this does not.
	#[must_use]
	pub fn local_extents(&self) -> Vec3 {
		match self.kind {
			| ShapeKind::Box => self.extents,
			| ShapeKind::Sphere => Vec3::splat(self.radius),
			| ShapeKind::Mesh => Vec3::ZERO,
		}
	}
}

impl Default for Shape {
	fn default() -> Self { Self::UNIT }
}

/// What the solver is allowed to do with a body.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BodyKind {
	/// Never moves. The world, and everything bolted to it.
	#[default]
	Static,

	/// Moves only when something writes its transform. Pushes, is not pushed.
	Kinematic,

	/// Moved by the solver.
	Dynamic,
}

/// A handle to a body.
///
/// Generational, unlike a resource handle and like [`EntityId`]. The registries
/// never free a slot, because an asset lives as long as the process does; a
/// body is destroyed and its slot reused, so a handle kept across that has to
/// be detectable. A sandbox holding a prop that somebody else deleted must fail
/// its lookup rather than pick up whoever moved in.
///
/// `Pod` for the reason [`EntityId`] is: a game keeps its handles in the arena,
/// and a zeroed arena has to read back as [`BodyId::NONE`] rather than as
/// something that could resolve. Zero is never a live generation, which is what
/// makes that true.
#[repr(C)]
#[derive(
	Clone,
	Copy,
	Debug,
	Default,
	PartialEq,
	Eq,
	Hash,
	crate::bytemuck::Pod,
	crate::bytemuck::Zeroable,
)]
pub struct BodyId {
	index: u32,
	generation: u32,
}

impl BodyId {
	/// A handle that refers to nothing, and always will.
	pub const NONE: Self = Self { index: 0, generation: 0 };

	/// Whether this handle could refer to anything at all.
	///
	/// A `true` here does not mean the body exists - only [`Bodies::alive`]
	/// answers that.
	#[must_use]
	pub const fn is_some(self) -> bool { self.generation != 0 }

	/// The slot this addresses, whatever lives there now.
	///
	/// The solver's, for keying its own per-body tables - a baked collision
	/// mesh, later a contact cache. Paired with
	/// [`generation`](Self::generation), which is how it notices the slot
	/// changed hands.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "u32 to usize is lossless on every target this builds for, and try_from is 		          not available in a const fn"
	)]
	pub const fn slot(self) -> usize { self.index as usize }

	/// Which occupant of that slot this handle names.
	#[must_use]
	pub const fn generation(self) -> u32 { self.generation }
}

/// One body: a shape, where it is, how it is moving, and what its surface is
/// like.
///
/// Everything a solver needs and nothing it does not: the shape, the
/// transform, the velocities, the mass and the surface. All of it is here
/// because this table is the authority, and a solver that had to fetch a
/// velocity from somewhere else would be mirroring again.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
	/// What the solver may do with it.
	pub kind: BodyKind,

	/// What it is shaped like.
	pub shape: Shape,

	/// Where it is.
	///
	/// Which way this and the entity's transform are copied depends on `kind`,
	/// and the rule is the one the word "kinematic" already means:
	///
	/// - **[`BodyKind::Dynamic`]** - the solver owns the transform, and the
	///   entity is written *from* here at the end of every step.
	/// - **[`BodyKind::Kinematic`] and [`BodyKind::Static`]** - gameplay owns
	///   the transform, and this is written *from* the entity at the start of
	///   every step, so a collider bolted to something the game moves follows
	///   it without being told.
	///
	/// One consequence worth knowing: a game that moves an entity in `update`
	/// and traces against it in the *same* `update` traces against where it was
	/// a step ago, because the pull happens at the top of the next step. A game
	/// that cares writes here as well, and this is authoritative immediately.
	/// A body with no entity is always its own authority.
	///
	/// A jump rather than a slide wants
	/// [`teleport_body`](crate::abi::World::teleport_body), which snaps the
	/// entity so nothing is drawn crossing the gap.
	pub transform: Transform,

	/// How fast it is moving, in units a second. World space.
	///
	/// Written by the solver for a [`BodyKind::Dynamic`] body and read for
	/// every other kind, so a moving platform pushes what stands on it. A game
	/// may write it whenever it likes: that is what throwing something is.
	pub velocity: Vec3,

	/// How fast it is turning, in radians a second, about an axis that is the
	/// vector's direction. World space.
	pub angular: Vec3,

	/// How heavy it is, in whatever unit the scene is consistent about.
	///
	/// Ignored unless the body is [`BodyKind::Dynamic`], where it must be
	/// positive: a dynamic body of zero mass is a division waiting to happen,
	/// and the solver treats one as [`Body::MASS`] rather than as immovable.
	/// Immovable is what [`BodyKind::Static`] is for.
	///
	/// The inertia tensor is *not* here. It follows from this and the shape,
	/// the solver works it out per step, and a cached one is a value that goes
	/// stale the moment somebody changes either.
	pub mass: f32,

	/// How much of an impact comes back, from zero to one.
	pub restitution: f32,

	/// How hard it is to slide along.
	pub friction: f32,

	/// Whether this body notices what it overlaps instead of pushing it.
	///
	/// A sensor takes part in the narrow phase and in nothing after it. The
	/// pairs it makes are reported through [`Bodies::touches`] and
	/// [`Bodies::overlaps`] and are never handed to the solver, so a trigger
	/// volume knows what is inside it and moves nothing. Traces skip it too,
	/// because a pick ray stopping at an invisible box is the same bug seen
	/// from the other side.
	///
	/// Orthogonal to [`kind`](Self::kind) on purpose. Who owns the transform
	/// and whether it pushes are two questions, and a sensor is worth having
	/// as all three kinds: bolted to the world, carried by a door, or thrown.
	/// A [`BodyKind::Dynamic`] one does fall forever, because nothing is left
	/// to stop it.
	pub sensor: bool,

	/// Which layers it is on, and which it interacts with.
	///
	/// Read by everything that puts two bodies together - the narrow phase, the
	/// sweep that stops a fast body passing through a thin one, and both
	/// traces. A body whose layers do not meet another's is not merely unpushed
	/// by it: the pair is never formed, so there is no touch event and no
	/// overlap either, which is what a trigger volume that only notices players
	/// needs.
	///
	/// [`Layers::DEFAULT`] is layer zero interacting with everything, so a body
	/// that says nothing behaves exactly as it did before layers existed.
	pub layers: Layers,

	/// Whether the solver has stopped integrating this body.
	///
	/// Set by the solver when a dynamic body has been slow enough for long
	/// enough, and cleared the moment anything touches it - a force, a
	/// teleport, or another body arriving. Public because it is the question a
	/// sandbox asks constantly ("is this pile settled"), and because a game
	/// that writes `false` here is how you wake something up.
	pub sleeping: bool,

	/// The entity this body drives, or [`EntityId::NONE`].
	///
	/// Optional on purpose: a trigger volume, a clip brush and a query-only
	/// collider are all bodies with nothing to draw.
	pub entity: EntityId,
}

impl Body {
	/// How much a body grips unless it says otherwise.
	pub const FRICTION: f32 = 0.5;
	/// How heavy a body is unless it says otherwise.
	pub const MASS: f32 = 1.0;
	/// How bouncy a body is unless it says otherwise.
	pub const RESTITUTION: f32 = 0.2;

	/// A body of a shape, at a transform, driving nothing.
	///
	/// @param kind - what the solver may do with it
	/// @param shape - what it is shaped like
	/// @param transform - where it is
	#[must_use]
	pub const fn new(kind: BodyKind, shape: Shape, transform: Transform) -> Self {
		Self {
			kind,
			shape,
			transform,
			velocity: Vec3::ZERO,
			angular: Vec3::ZERO,
			mass: Self::MASS,
			restitution: Self::RESTITUTION,
			friction: Self::FRICTION,
			sensor: false,
			layers: Layers::DEFAULT,
			sleeping: false,
			entity: EntityId::NONE,
		}
	}

	/// A body the solver moves.
	///
	/// @param shape - what it is shaped like
	/// @param transform - where it starts
	/// @param mass - how heavy it is
	#[must_use]
	pub const fn dynamic(shape: Shape, transform: Transform, mass: f32) -> Self {
		let mut body = Self::new(BodyKind::Dynamic, shape, transform);
		body.mass = mass;

		body
	}

	/// The same body, thrown.
	///
	/// @param velocity - how fast, in units a second
	/// @param angular - how fast it turns, in radians a second
	#[must_use]
	pub const fn moving(mut self, velocity: Vec3, angular: Vec3) -> Self {
		self.velocity = velocity;
		self.angular = angular;

		self
	}

	/// Whether the solver integrates this body at all.
	///
	/// A mesh is never movable however it is declared. A triangle soup has no
	/// inside, so it has no mass distribution and no inertia tensor, and Jolt
	/// refuses the same combination for the same reason. Saying so here rather
	/// than warning about it in the solver means every other piece of code can
	/// ask one question instead of two.
	#[must_use]
	pub const fn movable(&self) -> bool {
		matches!(self.kind, BodyKind::Dynamic) && !matches!(self.shape.kind, ShapeKind::Mesh)
	}

	/// One over the mass, or zero for anything the solver does not move.
	///
	/// The form every impulse in the solver actually wants, and the reason a
	/// static body needs no special case anywhere: it is a body of infinite
	/// mass, and infinite mass is zero here.
	#[must_use]
	pub fn inverse_mass(&self) -> f32 {
		if !self.movable() {
			return 0.0;
		}

		let mass = if self.mass > 0.0 { self.mass } else { Self::MASS };

		1.0 / mass
	}

	/// The same body, driving an entity.
	///
	/// @param entity - what to write this body's transform into each step
	#[must_use]
	pub const fn driving(mut self, entity: EntityId) -> Self {
		self.entity = entity;

		self
	}

	/// The same body, with a surface.
	///
	/// @param restitution - how much of an impact comes back
	/// @param friction - how hard it is to slide along
	#[must_use]
	pub const fn surfaced(mut self, restitution: f32, friction: f32) -> Self {
		self.restitution = restitution;
		self.friction = friction;

		self
	}

	/// The same body, noticing rather than pushing.
	///
	/// @ref [`sensor`](Self::sensor) for what that costs and what it buys.
	#[must_use]
	pub const fn sensing(mut self) -> Self {
		self.sensor = true;

		self
	}

	/// The same body, on other layers.
	///
	/// Chainable, like the rest of these.
	///
	/// @param layers - which layers it is on and which it interacts with
	#[must_use]
	pub const fn layered(mut self, layers: Layers) -> Self {
		self.layers = layers;

		self
	}

	/// Whether this body pushes what it overlaps.
	///
	/// The question every piece of the solver asks, written once so that the
	/// negation is not spelled out at each of them. @ref
	/// [`movable`](Self::movable), which is the other half of the same shape.
	#[must_use]
	pub const fn solid(&self) -> bool { !self.sensor }

	/// The smallest axis-aligned box in world space that holds this body.
	///
	/// A rotated box is bounded by the box around its rotated corners, which is
	/// larger than the shape - that is what an axis-aligned bound is, and every
	/// query that uses this treats it as a filter rather than as an answer.
	///
	/// @return `(min, max)`, or `None` for a mesh, whose size this struct does
	/// not know
	#[must_use]
	pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
		if self.shape.kind == ShapeKind::Mesh {
			return None;
		}

		let extents = self.shape.local_extents() * self.transform.scale.abs();
		let spread = rotated_extents(extents, self.transform.rotation);

		Some((self.transform.position - spread, self.transform.position + spread))
	}
}

impl Default for Body {
	fn default() -> Self { Self::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY) }
}

/// The half-extents of the axis-aligned box holding a rotated box.
///
/// The absolute value of the rotation matrix applied to the extents, which is
/// the standard trick and is exact.
///
/// @param extents - the box's half-extents before rotation
/// @param rotation - how it is turned
fn rotated_extents(extents: Vec3, rotation: Quat) -> Vec3 {
	let matrix = Mat3::from_quat(rotation);

	Vec3::new(
		matrix.x_axis.abs().dot(extents.abs()),
		matrix.y_axis.abs().dot(extents.abs()),
		matrix.z_axis.abs().dot(extents.abs()),
	)
}

/// How many bodies a step will report starting or stopping touching.
///
/// Bounded like everything else here. A step that produces more than this many
/// is a step something has gone wrong in, and dropping the rest is better than
/// a queue that grows until the process does.
pub const MAX_TOUCHES: usize = 256;

/// What happened between two bodies.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TouchKind {
	/// They were not touching last step and are now.
	#[default]
	Began,

	/// They were touching last step and are not now.
	Ended,
}

/// One thing that happened between two bodies during a step.
///
/// A queue the step drains, exactly like the interface's events and for the
/// same two reasons: a callback would be a function pointer with the game
/// module's lifetime, and it would run gameplay code from inside the solver
/// rather than from `update`. @ref [`ui`](crate::abi::ui).
///
/// What this is *for* is the half of physics that is not about pushing things
/// apart: a trigger volume, a pressure plate, a sound when two props hit. Those
/// want to know the moment something changed, which is the one thing a table
/// read every step cannot tell you.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Touch {
	/// One of the two bodies. Always the lower slot of the pair, so that a
	/// game comparing against a handle it owns need only look at both fields
	/// rather than at both orders.
	pub first: BodyId,

	/// The other.
	pub second: BodyId,

	/// Whether they have just started or just stopped.
	pub kind: TouchKind,

	/// Where they met, in world space. Meaningless for [`TouchKind::Ended`].
	pub point: Vec3,

	/// Which way the second was pushed. Meaningless for [`TouchKind::Ended`].
	pub normal: Vec3,
}

impl Touch {
	/// Whether this names a body, in either position.
	///
	/// @param body - the handle to look for
	#[must_use]
	pub fn names(&self, body: BodyId) -> bool { self.first == body || self.second == body }

	/// The other body of the pair, given one of them.
	///
	/// @param body - the one that is known
	/// @return the other, or [`BodyId::NONE`] if this touch does not name it
	#[must_use]
	pub fn other(&self, body: BodyId) -> BodyId {
		if self.first == body {
			return self.second;
		}

		if self.second == body {
			return self.first;
		}

		BodyId::NONE
	}
}

/// How many sensor overlaps a step will report.
///
/// Bounded like the touch queue beside it, and for the same reason. What is
/// dropped past this is counted by nothing, because a step with more than two
/// hundred and fifty-six things inside its triggers has a problem a longer
/// list is not going to fix.
pub const MAX_OVERLAPS: usize = 256;

/// One body inside one sensor, for the whole of a step.
///
/// The state a trigger volume wants, as against the edges
/// [`Touch`] carries. Both exist because neither can be had from the other:
/// edges cannot be recovered from a list read every step, and the list cannot
/// be rebuilt from edges without the game keeping notes - notes that go wrong
/// in exactly the place it is hardest to notice, which is a prop despawned
/// while it was inside. Here the list is rebuilt from what is actually
/// overlapping, every step, so nothing can drift.
///
/// Exactly one of the two is a sensor: two sensors are never tested against
/// each other, because two things that push nothing have nothing to say about
/// meeting.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Overlap {
	/// The sensor.
	pub sensor: BodyId,

	/// What is inside it.
	pub body: BodyId,
}

/// The host's body table.
///
/// The same storage discipline as [`Entities`](super::entity::Entities), minus
/// the second transform: a body has no past, because what gets interpolated is
/// the entity it drives and the host already does that.
pub struct Bodies {
	bodies: Vec<Body>,
	/// What each slot is called, or the empty string. @ref
	/// [`names`](crate::abi::names).
	names: Names,
	generations: Vec<u32>,
	alive: Vec<bool>,
	free: Vec<u32>,
	live: usize,
	/// What started and stopped touching this step. Filled by the solver,
	/// drained by the game, cleared beside the input edges.
	touches: Vec<Touch>,
	/// What is inside a sensor right now. Filled by the solver, read by the
	/// game, and rebuilt rather than cleared - @ref [`Overlap`].
	overlaps: Vec<Overlap>,
}

impl Bodies {
	/// An empty table.
	#[must_use]
	pub const fn new() -> Self {
		Self {
			bodies: Vec::new(),
			names: Names::new(),
			generations: Vec::new(),
			alive: Vec::new(),
			free: Vec::new(),
			live: 0,
			touches: Vec::new(),
			overlaps: Vec::new(),
		}
	}

	/// What started and stopped touching during this step.
	#[must_use]
	pub fn touches(&self) -> &[Touch] { &self.touches }

	/// Notes that two bodies started or stopped touching.
	///
	/// The solver's. Silently drops anything past [`MAX_TOUCHES`], because a
	/// step that busy has a problem the queue is not going to fix.
	///
	/// @param touch - what happened
	pub fn touched(&mut self, touch: Touch) {
		if self.touches.len() < MAX_TOUCHES {
			self.touches.push(touch);
		}
	}

	/// Forgets the touches of the step that just ended.
	///
	/// The host's, called beside [`Input::end_step`](super::Input::end_step)
	/// and [`Ui::end_step`](super::Ui::end_step) and for the same reason: two
	/// steps in one frame must not see the same event twice.
	pub fn end_step(&mut self) { self.touches.clear(); }

	/// Every body inside a sensor, as of the top of this step.
	///
	/// Raw, so a handle here may name a body destroyed since the step began.
	/// [`inside`](Self::inside) is the form that cannot.
	#[must_use]
	pub fn overlaps(&self) -> &[Overlap] { &self.overlaps }

	/// What is inside one sensor, skipping anything that has since died.
	///
	/// @param sensor - the volume to look in
	pub fn inside(&self, sensor: BodyId) -> impl Iterator<Item = BodyId> {
		self.overlaps
			.iter()
			.filter(move |overlap| overlap.sensor == sensor)
			.map(|overlap| overlap.body)
			.filter(|&body| self.alive(body))
	}

	/// Notes that a body is inside a sensor.
	///
	/// The solver's. Silently drops anything past [`MAX_OVERLAPS`].
	///
	/// @param overlap - the sensor and what is in it
	pub fn overlapped(&mut self, overlap: Overlap) {
		if self.overlaps.len() < MAX_OVERLAPS {
			self.overlaps.push(overlap);
		}
	}

	/// Forgets every overlap, so the solver can say what is true now.
	///
	/// The solver's, called at the top of a step rather than at the bottom
	/// beside [`end_step`](Self::end_step). That is the difference between a
	/// state and an edge: an edge is consumed once and must not be seen twice,
	/// while a state has to survive until something knows better. Clearing this
	/// where the touches are cleared would leave a game reading an empty list
	/// on every frame that is not a step.
	pub fn forget_overlaps(&mut self) { self.overlaps.clear(); }

	/// Creates a body.
	///
	/// @param body - what to create
	/// @return its handle, or [`BodyId::NONE`] if the table is full
	pub fn spawn(&mut self, body: Body) -> BodyId {
		let Some(slot) = self.take_slot() else {
			return BodyId::NONE;
		};

		let Ok(index) = u32::try_from(slot) else {
			return BodyId::NONE;
		};

		self.generations[slot] = self.generations[slot].saturating_add(1);
		self.alive[slot] = true;
		self.bodies[slot] = body;
		// the one place a name is cleared. @ref `abi::names`.
		self.names.set(slot, "");
		self.live += 1;

		BodyId {
			index,
			generation: self.generations[slot],
		}
	}

	/// Destroys a body.
	///
	/// @param id - the handle to destroy
	/// @return `true` if it existed, `false` if the handle was stale
	pub fn despawn(&mut self, id: BodyId) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.alive[slot] = false;
		self.bodies[slot] = Body::default();
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
			self.bodies[slot] = Body::default();
			if let Ok(index) = u32::try_from(slot) {
				self.free.push(index);
			}
		}

		self.live = 0;
	}

	/// Whether a handle refers to a living body.
	#[must_use]
	pub fn alive(&self, id: BodyId) -> bool { self.slot(id).is_some() }

	/// How many bodies exist.
	#[must_use]
	pub const fn len(&self) -> usize { self.live }

	/// Whether there are none at all.
	#[must_use]
	pub const fn is_empty(&self) -> bool { self.live == 0 }

	/// How many more bodies can be created.
	#[must_use]
	pub fn capacity_left(&self) -> usize { MAX_BODIES - self.alive.len() + self.free.len() }

	/// A body.
	#[must_use]
	pub fn get(&self, id: BodyId) -> Option<&Body> {
		self.slot(id).map(|slot| &self.bodies[slot])
	}

	/// A body, to change.
	///
	/// Writing `transform` here is a *move*, and @ref [`Body::transform`] for
	/// which way it is then copied. A jump wants
	/// [`teleport_body`](crate::abi::World::teleport_body) instead, which snaps
	/// the entity as well.
	pub fn get_mut(&mut self, id: BodyId) -> Option<&mut Body> {
		self.slot(id).map(|slot| &mut self.bodies[slot])
	}

	/// What a body is called, or the empty string.
	///
	/// A body is named separately from the entity it drives, because a scene
	/// source lets a joint say which body it holds and the answer has to be
	/// writable even for a body no entity stands on. @ref
	/// [`names`](crate::abi::names).
	#[must_use]
	pub fn name(&self, id: BodyId) -> &str {
		self.slot(id)
			.map_or("", |slot| self.names.at(slot))
	}

	/// Names a body, cutting anything past
	/// [`MAX_NAME`](crate::abi::MAX_NAME).
	///
	/// @param id - what to name
	/// @param name - what to call it; empty clears the name
	/// @return `true` if the handle resolved
	pub fn set_name(&mut self, id: BodyId, name: &str) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.names.set(slot, name);

		true
	}

	/// Every living body, with its handle.
	pub fn iter(&self) -> impl Iterator<Item = (BodyId, &Body)> {
		self.bodies
			.iter()
			.enumerate()
			.filter(|&(slot, _)| self.alive[slot])
			.filter_map(|(slot, body)| {
				let index = u32::try_from(slot).ok()?;

				Some((
					BodyId {
						index,
						generation: self.generations[slot],
					},
					body,
				))
			})
	}

	/// Rebuilds the whole table from a description, slot for slot.
	///
	/// The same contract
	/// [`Entities::restore`](super::entity::Entities::restore) has, and for
	/// the same reason: a body handle a game kept in its arena is only worth
	/// restoring if it lands back on the same body. The touch and
	/// overlap queues are emptied with it, because both describe a world that
	/// is no longer here.
	///
	/// @param generations - the generation of every slot, dead ones included;
	/// its length is how many slots the table ends up with, capped at
	/// [`MAX_BODIES`]
	/// @param entries - `(slot, body)` for each living body
	/// @return one handle per entry, in order, [`BodyId::NONE`] for any whose
	/// slot the table could not hold
	pub fn restore(&mut self, generations: &[u32], entries: &[(usize, Body)]) -> Vec<BodyId> {
		let slots = generations.len().min(MAX_BODIES);

		self.bodies.clear();
		self.bodies.resize(slots, Body::default());
		self.names.reset(slots);
		self.generations.clear();
		self.generations
			.extend_from_slice(&generations[..slots]);
		self.alive.clear();
		self.alive.resize(slots, false);
		self.free.clear();
		self.touches.clear();
		self.overlaps.clear();
		self.live = 0;

		let mut handles = Vec::with_capacity(entries.len());
		for (slot, body) in entries {
			handles.push(self.put(*slot, *body));
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

	/// Puts one body back into a slot a [`restore`](Self::restore) has just
	/// sized the table for.
	fn put(&mut self, slot: usize, body: Body) -> BodyId {
		let (Ok(index), Some(alive)) = (u32::try_from(slot), self.alive.get_mut(slot)) else {
			return BodyId::NONE;
		};

		if *alive {
			return BodyId::NONE;
		}

		*alive = true;
		self.bodies[slot] = body;
		self.generations[slot] = self.generations[slot].max(1);
		self.live += 1;

		BodyId {
			index,
			generation: self.generations[slot],
		}
	}

	/// The generation living in a slot, whether or not it is occupied.
	///
	/// The solver's, for noticing that the body it cached something for is not
	/// the body in that slot any more.
	///
	/// @param slot - an index below [`Bodies::slots`](Self::slots)
	#[must_use]
	pub fn generation(&self, slot: usize) -> u32 {
		self.generations.get(slot).copied().unwrap_or(0)
	}

	/// How many slots the table has ever handed out.
	#[must_use]
	pub fn slots(&self) -> usize { self.bodies.len() }

	/// The slot a handle addresses, if it is still the body it was.
	fn slot(&self, id: BodyId) -> Option<usize> {
		if !id.is_some() {
			return None;
		}

		let slot = usize::try_from(id.index).ok()?;

		(self.alive.get(slot) == Some(&true)
			&& self.generations.get(slot) == Some(&id.generation))
		.then_some(slot)
	}

	/// A free slot, reused or newly grown.
	fn take_slot(&mut self) -> Option<usize> {
		if let Some(index) = self.free.pop() {
			return usize::try_from(index).ok();
		}

		if self.bodies.len() >= MAX_BODIES {
			return None;
		}

		self.bodies.push(Body::default());
		self.names.push();
		self.generations.push(0);
		self.alive.push(false);

		Some(self.bodies.len() - 1)
	}
}

impl Default for Bodies {
	fn default() -> Self { Self::new() }
}

/// What to ask the world about.
///
/// A start, an end, an optional box to sweep along it, and the handles to
/// pretend are not there - @ref [`MAX_IGNORED`] for why that last one is a
/// fixed array.
///
/// @note: `#[repr(C)]` and `Copy`, but **not** `Pod`: glam is built without
/// bytemuck, so nothing holding a `Vec3` can be. It does not need to be - the
/// arena is the only thing that requires `Pod`, and this never goes near it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceInfo {
	/// Where the trace begins.
	pub start: Vec3,

	/// Where it would end if it hit nothing.
	pub end: Vec3,

	/// Whether this sweeps a box rather than a ray.
	pub is_box: bool,

	/// The half-extents of that box.
	pub extents: Vec3,

	/// Which layers this trace is on, and which it can hit.
	///
	/// [`Layers::ALL`] unless it says otherwise, which is on every layer and
	/// interacting with every layer - so a trace that never mentions this
	/// reaches exactly what it always did. A trace narrowed to one layer skips
	/// every body outside it, and a body whose own mask excludes the trace's
	/// layer skips the trace, because the rule is the symmetric one @ref
	/// [`Layers::meets`].
	pub layers: Layers,

	/// Bodies to pretend are not there.
	ignore: [BodyId; MAX_IGNORED],

	/// How many of `ignore` are set.
	ignored: u32,
}

impl TraceInfo {
	/// A ray between two points.
	///
	/// @param start - where it begins
	/// @param end - where it stops
	#[must_use]
	pub const fn ray(start: Vec3, end: Vec3) -> Self {
		Self {
			start,
			end,
			is_box: false,
			extents: Vec3::ZERO,
			layers: Layers::ALL,
			ignore: [BodyId::NONE; MAX_IGNORED],
			ignored: 0,
		}
	}

	/// A ray from a point, in a direction, for a distance.
	///
	/// @param origin - where it begins
	/// @param direction - which way it goes; normalized here, so it need not be
	/// @param distance - how far it reaches
	#[must_use]
	pub fn along(origin: Vec3, direction: Vec3, distance: f32) -> Self {
		Self::ray(origin, origin + direction.normalize_or_zero() * distance)
	}

	/// A box swept between two points.
	///
	/// The box is axis-aligned and stays that way for the whole sweep.
	///
	/// @param start - where its middle begins
	/// @param end - where its middle stops
	/// @param extents - its half-extents
	#[must_use]
	pub const fn swept(start: Vec3, end: Vec3, extents: Vec3) -> Self {
		Self {
			start,
			end,
			is_box: true,
			extents,
			layers: Layers::ALL,
			ignore: [BodyId::NONE; MAX_IGNORED],
			ignored: 0,
		}
	}

	/// The same trace, on other layers.
	///
	/// Chainable, like [`ignoring`](Self::ignoring), and the two filters are
	/// different tools: an ignore list names the handful of bodies this one
	/// call must not see, and layers say what kind of thing the trace is about
	/// at all.
	///
	/// @param layers - which layers it is on and which it can hit
	#[must_use]
	pub const fn layered(mut self, layers: Layers) -> Self {
		self.layers = layers;

		self
	}

	/// The same trace, blind to one more body.
	///
	/// Chainable. Past [`MAX_IGNORED`] the extra handles are dropped, because
	/// the alternative - a trace that quietly hits the thing it was told to
	/// ignore - is the same bug with a longer fuse and no place to report it.
	///
	/// @param body - what to pretend is not there
	#[must_use]
	pub fn ignoring(mut self, body: BodyId) -> Self {
		let count = usize::try_from(self.ignored).unwrap_or(MAX_IGNORED);

		if count < MAX_IGNORED {
			self.ignore[count] = body;
			self.ignored = self.ignored.saturating_add(1);
		}

		self
	}

	/// The bodies this trace is blind to.
	#[must_use]
	pub fn ignored(&self) -> &[BodyId] {
		let count = usize::try_from(self.ignored).unwrap_or(0);

		&self.ignore[..count.min(MAX_IGNORED)]
	}

	/// Whether this trace is blind to a body.
	///
	/// @param body - the handle to check
	#[must_use]
	pub fn ignores(&self, body: BodyId) -> bool { self.ignored().contains(&body) }

	/// How far the trace reaches.
	#[must_use]
	pub fn distance(&self) -> f32 { self.start.distance(self.end) }
}

/// What the world said.
///
/// Everything a caller needs about what was hit, the body handle included:
/// reporting only the entity would lose every body that drives none.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceResult {
	/// Whether anything was in the way.
	pub hit: bool,

	/// Where the trace began.
	pub start: Vec3,

	/// Where it stopped: the point of contact, or `info.end` on a miss.
	pub end: Vec3,

	/// How far along the trace stopped, from zero to one. One on a miss.
	pub fraction: f32,

	/// The surface normal at the contact, pointing back along the trace.
	///
	/// [`Vec3::ZERO`] on a miss, rather than a sentinel that is itself a
	/// direction and can therefore be mistaken for a real one.
	pub normal: Vec3,

	/// Whether the trace began inside something.
	pub started_solid: bool,

	/// Whether it ended inside something.
	pub ended_solid: bool,

	/// What it hit.
	pub body: BodyId,

	/// The entity that body drives, or [`EntityId::NONE`].
	pub entity: EntityId,
}

impl TraceResult {
	/// A trace that hit nothing, having gone the whole way.
	///
	/// @param start - where it began
	/// @param end - where it stopped
	#[must_use]
	pub const fn miss(start: Vec3, end: Vec3) -> Self {
		Self {
			hit: false,
			start,
			end,
			fraction: 1.0,
			normal: Vec3::ZERO,
			started_solid: false,
			ended_solid: false,
			body: BodyId::NONE,
			entity: EntityId::NONE,
		}
	}
}

/// One entry point of [`Physics`].
///
/// # Safety
///
/// `context` must be the pointer the host installed alongside this function,
/// `bodies` must point at a live [`Bodies`] nobody is mutating for the duration
/// of the call, and `info` at a live [`TraceInfo`]. All three are guaranteed by
/// [`trace_ray`](crate::abi::World::trace_ray) and
/// [`trace_box`](crate::abi::World::trace_box), which are the only callers that
/// should exist.
pub type TraceFn = unsafe extern "C-unwind" fn(
	context: *mut c_void,
	bodies: *const Bodies,
	info: *const TraceInfo,
) -> TraceResult;

/// Reports a clean miss. What [`Physics::STUB`] is made of.
///
/// # Safety
///
/// `info` must point at a live [`TraceInfo`].
unsafe extern "C-unwind" fn nothing(
	_context: *mut c_void,
	_bodies: *const Bodies,
	info: *const TraceInfo,
) -> TraceResult {
	// SAFETY: the caller guarantees a live TraceInfo, which is what every
	// caller of a TraceFn has to guarantee anyway.
	let info = unsafe { &*info };

	TraceResult::miss(info.start, info.end)
}

/// The queries the host answers, as a table the game calls through.
///
/// Installed once, by the host, at startup - @ref the module docs for why that
/// is enough to survive every reload that follows.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Physics {
	/// Whatever the host needs to answer a query. Opaque here on purpose.
	context: *mut c_void,

	/// Traces a ray.
	ray: TraceFn,

	/// Sweeps a box.
	shape: TraceFn,
}

impl Physics {
	/// A table that reports a miss for everything.
	///
	/// What a [`World`](crate::abi::World) holds until a host installs a real
	/// one, so that a unit test, an offscreen capture or a dedicated server
	/// without a solver answers "nothing there" instead of dereferencing a
	/// null.
	pub const STUB: Self = Self {
		context: core::ptr::null_mut(),
		ray: nothing,
		shape: nothing,
	};

	/// A table over a host's solver.
	///
	/// @param context - handed back to both functions on every call; it must
	/// outlive every [`World`](crate::abi::World) this is installed into, and
	/// must not move @param ray - answers a ray trace
	/// @param shape - answers a swept-box trace
	#[must_use]
	pub const fn new(context: *mut c_void, ray: TraceFn, shape: TraceFn) -> Self {
		Self { context, ray, shape }
	}

	/// Traces a ray.
	#[must_use]
	#[expect(
		ffi_unwind_calls,
		reason = "the table is deliberately C-unwind, for the reason GameApi is: host and 		          module share one panic runtime under -Cprefer-dynamic, so a panic inside a 		          query reaches the host's catch_unwind instead of aborting the process"
	)]
	pub fn cast_ray(&self, bodies: &Bodies, info: &TraceInfo) -> TraceResult {
		// SAFETY: `context` is what the host installed beside `ray` and is
		// alive for as long as this table is; `bodies` and `info` are live
		// references borrowed for the duration of the call.
		unsafe { (self.ray)(self.context, bodies, info) }
	}

	/// Sweeps a box.
	#[must_use]
	#[expect(ffi_unwind_calls, reason = "as `cast_ray`")]
	pub fn cast_shape(&self, bodies: &Bodies, info: &TraceInfo) -> TraceResult {
		// SAFETY: as `cast_ray`.
		unsafe { (self.shape)(self.context, bodies, info) }
	}
}

impl Default for Physics {
	fn default() -> Self { Self::STUB }
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::abi::World;

	#[test]
	fn a_stale_body_handle_does_not_pick_up_its_successor() {
		let mut bodies = Bodies::new();
		let first = bodies.spawn(Body::default());

		assert!(bodies.despawn(first), "it was there");

		let second = bodies.spawn(Body::default());

		assert_eq!(second.index, first.index, "the slot really is reused");
		assert!(bodies.get(first).is_none(), "and the old handle no longer resolves");
		assert!(bodies.get(second).is_some(), "while the new one does");
	}

	#[test]
	fn a_none_handle_resolves_to_nothing_even_in_an_empty_table() {
		let bodies = Bodies::new();

		assert!(!bodies.alive(BodyId::NONE), "zero is never a live generation");
	}

	#[test]
	fn the_table_stops_rather_than_growing_without_end() {
		let mut bodies = Bodies::new();

		for _ in 0..MAX_BODIES {
			assert!(bodies.spawn(Body::default()).is_some(), "up to the bound they all land");
		}

		assert_eq!(bodies.spawn(Body::default()), BodyId::NONE, "and past it none do");
		assert_eq!(bodies.len(), MAX_BODIES, "with nothing lost on the way");
	}

	#[test]
	fn iterating_yields_the_living_and_their_handles() {
		let mut bodies = Bodies::new();
		let first =
			bodies.spawn(Body::new(BodyKind::Static, Shape::ball(1.0), Transform::IDENTITY));
		let second = bodies.spawn(Body::default());
		bodies.despawn(first);

		let seen: Vec<BodyId> = bodies.iter().map(|(id, _)| id).collect();

		assert_eq!(
			seen,
			vec![second],
			"the despawned one is gone and the handle is the live one"
		);
	}

	#[test]
	fn a_rotated_box_is_bounded_by_something_larger_than_itself() {
		let mut body = Body::new(
			BodyKind::Static,
			Shape::cuboid(Vec3::new(1.0, 0.1, 1.0)),
			Transform::IDENTITY,
		);
		let (low, high) = body.bounds().expect("a box has bounds");

		assert!(high.abs_diff_eq(Vec3::new(1.0, 0.1, 1.0), 1.0e-5), "unrotated it is itself");
		assert!(low.abs_diff_eq(-high, 1.0e-5), "and symmetric about the origin");

		body.transform.rotation = Quat::from_rotation_z(core::f32::consts::FRAC_PI_2);
		let (_, turned) = body.bounds().expect("still a box");

		assert!(
			turned.x < 0.2 && turned.y > 0.9,
			"a quarter turn about z swaps the wide axis for the thin one, got {turned}"
		);
	}

	#[test]
	fn a_mesh_body_reports_no_bounds_because_this_struct_does_not_know_them() {
		let body = Body::new(BodyKind::Static, Shape::mesh(MeshId::CUBE), Transform::IDENTITY);

		assert!(body.bounds().is_none(), "the solver knows the size, and this does not");
	}

	#[test]
	fn a_box_fitted_around_bounds_is_half_their_size_and_centered_on_them() {
		let (shape, center) = Shape::around(Vec3::new(-1.0, 0.0, -2.0), Vec3::new(3.0, 4.0, 2.0));

		assert!(
			shape
				.extents
				.abs_diff_eq(Vec3::new(2.0, 2.0, 2.0), 1.0e-5),
			"half the span"
		);
		assert!(center.abs_diff_eq(Vec3::new(1.0, 2.0, 0.0), 1.0e-5), "and the middle of it");
	}

	#[test]
	fn an_ignore_list_fills_up_rather_than_overflowing() {
		let mut info = TraceInfo::ray(Vec3::ZERO, Vec3::X);

		for index in 1..=u32::try_from(MAX_IGNORED + 4).expect("a small count") {
			info = info.ignoring(BodyId { index, generation: 1 });
		}

		assert_eq!(info.ignored().len(), MAX_IGNORED, "it stops at the bound");
		assert!(
			info.ignores(BodyId { index: 1, generation: 1 }),
			"keeping the ones asked for first"
		);
	}

	#[test]
	fn a_trace_along_a_direction_reaches_exactly_that_far() {
		let info = TraceInfo::along(Vec3::ZERO, Vec3::new(0.0, 0.0, -3.0), 10.0);

		assert!((info.distance() - 10.0).abs() < 1.0e-4, "the direction is normalized for us");
		assert!(
			info.end
				.abs_diff_eq(Vec3::new(0.0, 0.0, -10.0), 1.0e-4),
			"and points the right way"
		);
	}

	#[test]
	fn a_world_nobody_wired_up_reports_a_clean_miss() {
		let world = World::new();
		let result = world.trace_ray(&TraceInfo::ray(Vec3::ZERO, Vec3::splat(100.0)));

		assert!(!result.hit, "the stub answers rather than crashing");
		assert!((result.fraction - 1.0).abs() < f32::EPSILON, "having gone the whole way");
		assert_eq!(result.body, BodyId::NONE, "and hit nothing");
	}

	#[test]
	fn the_trace_types_are_the_size_they_look() {
		assert_eq!(
			size_of::<Physics>(),
			size_of::<usize>() * 3,
			"a context and two function pointers, and nothing hiding in it"
		);
		assert_eq!(size_of::<BodyId>(), size_of::<u64>(), "two words, like an EntityId");
	}

	#[test]
	fn a_body_pushes_what_it_meets_unless_it_is_told_not_to() {
		let solid = Body::default();
		let sensor = Body::default().sensing();

		assert!(solid.solid(), "a body is solid unless it says otherwise");
		assert!(!sensor.solid(), "and a sensor says otherwise");
		assert!(!solid.sensor, "which is one flag and not a kind");
	}

	#[test]
	fn a_sensor_is_still_whatever_kind_it_was() {
		let body = Body::dynamic(Shape::UNIT, Transform::IDENTITY, 1.0).sensing();

		assert!(body.movable(), "the solver still integrates it, so it falls");
		assert!(!body.solid(), "and it still pushes nothing on the way down");
		assert_eq!(body.kind, BodyKind::Dynamic, "the two questions are separate");
	}

	#[test]
	fn what_is_inside_a_sensor_skips_what_has_died_since() {
		let mut bodies = Bodies::new();
		let sensor = bodies.spawn(Body::default().sensing());
		let standing = bodies.spawn(Body::default());
		let leaving = bodies.spawn(Body::default());

		bodies.overlapped(Overlap { sensor, body: standing });
		bodies.overlapped(Overlap { sensor, body: leaving });
		bodies.despawn(leaving);

		let found: Vec<BodyId> = bodies.inside(sensor).collect();

		assert_eq!(found, vec![standing], "the dead handle is not reported");
		assert_eq!(bodies.overlaps().len(), 2, "while the raw list still holds both");
	}

	#[test]
	fn inside_answers_about_the_sensor_it_was_asked_about() {
		let mut bodies = Bodies::new();
		let first = bodies.spawn(Body::default().sensing());
		let second = bodies.spawn(Body::default().sensing());
		let prop = bodies.spawn(Body::default());

		bodies.overlapped(Overlap { sensor: first, body: prop });

		assert_eq!(bodies.inside(first).count(), 1, "the one it is in");
		assert_eq!(bodies.inside(second).count(), 0, "and not the one beside it");
	}

	#[test]
	fn the_overlap_list_stops_rather_than_growing_without_end() {
		let mut bodies = Bodies::new();
		let sensor = bodies.spawn(Body::default().sensing());

		for _ in 0..MAX_OVERLAPS + 8 {
			bodies.overlapped(Overlap { sensor, body: BodyId::NONE });
		}

		assert_eq!(bodies.overlaps().len(), MAX_OVERLAPS, "it stops at the bound");
	}

	#[test]
	fn overlaps_outlive_the_step_edges_beside_them() {
		let mut bodies = Bodies::new();
		let sensor = bodies.spawn(Body::default().sensing());
		let prop = bodies.spawn(Body::default());

		bodies.overlapped(Overlap { sensor, body: prop });
		bodies.touched(Touch {
			first: sensor,
			second: prop,
			kind: TouchKind::Began,
			point: Vec3::ZERO,
			normal: Vec3::Y,
		});
		bodies.end_step();

		assert!(bodies.touches().is_empty(), "an edge is consumed once");
		assert_eq!(bodies.overlaps().len(), 1, "and a state is not");

		bodies.forget_overlaps();

		assert!(bodies.overlaps().is_empty(), "until the solver says what is true now");
	}
	#[test]
	fn a_layer_bit_wraps_rather_than_shifting_off_the_end() {
		assert_eq!(Layers::bit(0), 1, "layer zero is the low bit");
		assert_eq!(Layers::bit(31), 1 << 31, "and thirty-one is the high one");
		assert_eq!(Layers::bit(32), 1, "past that it wraps rather than being undefined");
	}

	#[test]
	fn two_bodies_that_say_nothing_about_layers_still_meet() {
		assert!(
			Layers::DEFAULT.meets(Layers::DEFAULT),
			"or every world written before layers existed would come apart"
		);
		assert!(Layers::ALL.meets(Layers::DEFAULT), "and a trace reaches them all");
	}

	#[test]
	fn narrowing_one_side_alone_is_enough_to_separate_a_pair() {
		let prop = Layers::single(0);
		let ghost = Layers::single(1).interacting(Layers::bit(1));

		assert!(!ghost.meets(prop), "the one that narrowed does not want the other");
		assert!(
			!prop.meets(ghost),
			"and the one that did not is refused all the same, because the rule is symmetric"
		);
	}

	#[test]
	fn layers_that_overlap_in_both_directions_meet() {
		let player = Layers::new(Layers::bit(0), Layers::bit(1) | Layers::bit(2));
		let wall = Layers::new(Layers::bit(1), Layers::bit(0));

		assert!(player.meets(wall), "each is in the other's mask");
		assert!(wall.meets(player), "in both orders");
	}

	#[test]
	fn a_zeroed_layers_is_inert_rather_than_universal() {
		assert!(!Layers::NONE.meets(Layers::ALL), "it is on no layer anything can see");
		assert!(!Layers::ALL.meets(Layers::NONE), "in either order");
		assert!(!Layers::NONE.meets(Layers::NONE), "and not even with itself");
	}

	#[test]
	fn a_body_is_on_layer_zero_and_a_trace_is_on_all_of_them() {
		assert_eq!(
			Body::default().layers,
			Layers::DEFAULT,
			"a body that says nothing behaves as it did before layers existed"
		);
		assert_eq!(
			TraceInfo::ray(Vec3::ZERO, Vec3::X).layers,
			Layers::ALL,
			"and so does a trace"
		);
		assert_eq!(
			TraceInfo::swept(Vec3::ZERO, Vec3::X, Vec3::ONE).layers,
			Layers::ALL,
			"of either kind"
		);
	}

	#[test]
	fn a_body_carries_a_name_and_a_reused_slot_does_not_inherit_it() {
		let mut bodies = Bodies::new();
		let old = bodies.spawn(Body::default());

		assert_eq!(bodies.name(old), "", "a body starts unnamed");
		assert!(bodies.set_name(old, "floor"), "and can be told what it is");
		assert_eq!(bodies.name(old), "floor");

		bodies.despawn(old);
		let new = bodies.spawn(Body::default());

		assert_eq!(bodies.name(old), "", "the stale handle reaches nothing");
		assert_eq!(bodies.name(new), "", "and the slot came back unnamed");
		assert!(!bodies.set_name(old, "ghost"), "naming a stale handle does nothing");
	}

	#[test]
	fn every_body_array_stays_the_same_length() {
		let mut bodies = Bodies::new();
		let mut ids = Vec::new();
		for _ in 0..6 {
			ids.push(bodies.spawn(Body::default()));
		}

		bodies.despawn(ids[2]);
		bodies.spawn(Body::default());
		bodies.clear();
		bodies.spawn(Body::default());

		let length = bodies.bodies.len();

		assert_eq!(bodies.alive.len(), length, "the table is one table");
		assert_eq!(bodies.generations.len(), length, "and every array in it agrees");
		assert_eq!(bodies.names.slots(), length, "and every array in it agrees");
	}
}
