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

use super::{material::MaterialId, mesh::MeshId};
use crate::{
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
	/// Position in world space.
	pub position: Vec3,

	/// Rotation.
	pub rotation: Quat,

	/// Scale along each axis.
	pub scale: Vec3,
}

impl Transform {
	/// At the origin, unrotated, unscaled.
	pub const IDENTITY: Self = Self {
		position: Vec3::ZERO,
		rotation: Quat::IDENTITY,
		scale: Vec3::ONE,
	};

	/// A transform at a position, with everything else left alone.
	#[must_use]
	pub const fn at(position: Vec3) -> Self { Self { position, ..Self::IDENTITY } }

	/// The model matrix this transform stands for.
	#[must_use]
	pub fn matrix(&self) -> Mat4 {
		Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
	}

	/// Sets every axis of the scale at once.
	pub fn set_scale(&mut self, scale: f32) { self.scale = Vec3::splat(scale); }

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
}

impl Renderable {
	/// Draws nothing.
	pub const NOTHING: Self = Self {
		mesh: MeshId::NONE,
		material: MaterialId::DEFAULT,
		color: Vec3::ONE,
	};

	/// A shape in a color, made of the default material.
	#[must_use]
	pub const fn new(mesh: MeshId, color: Vec3) -> Self {
		Self {
			mesh,
			material: MaterialId::DEFAULT,
			color,
		}
	}

	/// The same, made of something in particular.
	#[must_use]
	pub const fn of(mesh: MeshId, material: MaterialId, color: Vec3) -> Self {
		Self { mesh, material, color }
	}
}

impl Default for Renderable {
	fn default() -> Self { Self::NOTHING }
}

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
	use std::f32::consts::{FRAC_PI_6, TAU};

	use super::*;

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
}
