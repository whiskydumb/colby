//! Joints: the ways two bodies can be told to stay near each other.
//!
//! Three of them, and they are the three a sandbox actually builds with - a
//! rope that stops two things drifting apart, a weld that stops them moving at
//! all, and a hinge that leaves one axis free. Everything a physics gun does is
//! made of those.
//!
//! Shaped exactly like [`Bodies`](super::physics::Bodies), for the same
//! reasons: plain data in the host, a bounded table, a generational handle,
//! and no knowledge of a solver anywhere in it. A joint that named a solver's
//! internals would be a joint a saved scene could not write.
//!
//! **A joint holds its anchors in each body's own space.** Not in the world:
//! the whole point of an anchor is that it stays put on the thing it is
//! attached to, and a world-space anchor would have to be rewritten every step
//! by whoever owns the body. Converting is one matrix each, and the solver does
//! it.

use super::physics::BodyId;
use crate::{
	bytemuck::{Pod, Zeroable},
	glam::{Quat, Vec3},
};

/// How many joints can exist at once.
///
/// Bounded for the reason the body table is bounded: a reload that welds
/// something every step should run out of slots rather than out of memory.
pub const MAX_JOINTS: usize = 1024;

/// What a joint does to the two bodies it holds.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum JointKind {
	/// Stops them getting further apart than [`Joint::length`], and does
	/// nothing at all when they are closer.
	///
	/// One constraint, and an *inequality*: a rope pulls and never pushes,
	/// which is the whole difference between a rope and a rod.
	#[default]
	Rope,

	/// Holds them exactly where they were relative to each other.
	///
	/// Six constraints - three that keep the anchors together and three that
	/// keep the orientations from drifting apart.
	Weld,

	/// Holds the anchors together and lets them turn about one axis.
	///
	/// Five constraints: the three of a weld's position, and two of its three
	/// angular ones. The third is [`Joint::axis`] and is left free.
	Axis,
}

/// A handle to a joint.
///
/// Generational, like [`BodyId`] and for the same reason: a joint is broken and
/// its slot reused, and a tool holding a handle across that must fail its
/// lookup rather than pick up whoever moved in.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct JointId {
	index: u32,
	generation: u32,
}

impl JointId {
	/// A handle that refers to nothing, and always will.
	pub const NONE: Self = Self { index: 0, generation: 0 };

	/// Which occupant of that slot this handle names.
	#[must_use]
	pub const fn generation(self) -> u32 { self.generation }

	/// Whether this handle could refer to anything at all.
	#[must_use]
	pub const fn is_some(self) -> bool { self.generation != 0 }

	/// The slot this addresses, whatever lives there now.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "u32 to usize is lossless on every target this builds for, and try_from is not \
		          available in a const fn"
	)]
	pub const fn slot(self) -> usize { self.index as usize }
}

/// One joint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Joint {
	/// Which of the three it is.
	pub kind: JointKind,

	/// One of the two bodies it holds.
	pub first: BodyId,

	/// The other.
	///
	/// May be [`BodyId::NONE`], which pins the first body to a point in the
	/// world rather than to anything: an anchor in the ceiling is a joint whose
	/// second body is nothing at all, and its second anchor is then read as a
	/// world position rather than a local one.
	pub second: BodyId,

	/// Where it is attached on the first body, in that body's own space.
	pub first_anchor: Vec3,

	/// Where it is attached on the second, in *its* own space - or in the
	/// world, if there is no second body.
	pub second_anchor: Vec3,

	/// The axis a [`JointKind::Axis`] turns about, in the first body's space.
	///
	/// Normalized by the solver, so it need not be.
	pub axis: Vec3,

	/// How far apart a [`JointKind::Rope`] lets them get.
	pub length: f32,

	/// How the two bodies were turned relative to each other when the joint
	/// was made.
	///
	/// A weld holds them at *this*, not at nothing: welding two props that were
	/// already at an angle should keep the angle. Filled by
	/// [`World::join`](crate::abi::World::join), which is the only thing that
	/// can see both bodies' transforms; a game that spawns a joint through
	/// [`Joints::spawn`] directly is answerable for it itself.
	pub rest: Quat,

	/// How much of the pull is given up each step, from zero to one.
	///
	/// Zero is a rigid joint. Anything above it is a joint that sags under
	/// load, which is what a rope made of rope does and what a weld made of
	/// metal does not. Above about a half nothing holds together at all.
	pub give: f32,
}

impl Joint {
	/// How much a joint gives unless it says otherwise.
	pub const GIVE: f32 = 0.0;

	/// A hinge between two bodies.
	///
	/// @param first - one body
	/// @param second - the other
	/// @param anchors - where it attaches on each, in that body's own space
	/// @param axis - what it turns about, in the first body's space
	#[must_use]
	pub const fn axis(first: BodyId, second: BodyId, anchors: (Vec3, Vec3), axis: Vec3) -> Self {
		let mut joint = Self::new(JointKind::Axis, first, second, anchors);
		joint.axis = axis;

		joint
	}

	/// A joint of a kind, between two bodies.
	///
	/// @param kind - which of the three
	/// @param first - one body
	/// @param second - the other, or [`BodyId::NONE`] to pin to the world
	/// @param anchors - where it attaches on each, in that body's own space
	#[must_use]
	pub const fn new(
		kind: JointKind,
		first: BodyId,
		second: BodyId,
		anchors: (Vec3, Vec3),
	) -> Self {
		Self {
			kind,
			first,
			second,
			first_anchor: anchors.0,
			second_anchor: anchors.1,
			axis: Vec3::Y,
			rest: Quat::IDENTITY,
			length: 0.0,
			give: Self::GIVE,
		}
	}

	/// A rope between two bodies.
	///
	/// @param first - one body
	/// @param second - the other
	/// @param anchors - where it attaches on each, in that body's own space
	/// @param length - how far apart it lets them get
	#[must_use]
	pub const fn rope(first: BodyId, second: BodyId, anchors: (Vec3, Vec3), length: f32) -> Self {
		let mut joint = Self::new(JointKind::Rope, first, second, anchors);
		joint.length = length;

		joint
	}

	/// The same joint, softened.
	///
	/// @param give - how much of the pull is given up each step
	#[must_use]
	pub const fn soft(mut self, give: f32) -> Self {
		self.give = give;

		self
	}

	/// A weld between two bodies.
	///
	/// @param first - one body
	/// @param second - the other
	/// @param anchors - where it attaches on each, in that body's own space
	#[must_use]
	pub const fn weld(first: BodyId, second: BodyId, anchors: (Vec3, Vec3)) -> Self {
		Self::new(JointKind::Weld, first, second, anchors)
	}

	/// How many scalar constraints this joint is.
	///
	/// Not used by anything but a statistics line and the doc comments above;
	/// it is here because "how many constraints is a weld" is the question
	/// somebody reading the solver asks first.
	#[must_use]
	pub const fn constraints(&self) -> usize {
		match self.kind {
			| JointKind::Rope => 1,
			| JointKind::Weld => 6,
			| JointKind::Axis => 5,
		}
	}
}

impl Default for Joint {
	fn default() -> Self {
		Self::new(JointKind::Rope, BodyId::NONE, BodyId::NONE, (Vec3::ZERO, Vec3::ZERO))
	}
}

/// The host's joint table.
///
/// The same storage discipline as [`Bodies`](super::physics::Bodies).
pub struct Joints {
	joints: Vec<Joint>,
	generations: Vec<u32>,
	alive: Vec<bool>,
	free: Vec<u32>,
	live: usize,
}

impl Joints {
	/// An empty table.
	#[must_use]
	pub const fn new() -> Self {
		Self {
			joints: Vec::new(),
			generations: Vec::new(),
			alive: Vec::new(),
			free: Vec::new(),
			live: 0,
		}
	}

	/// Whether a handle refers to a living joint.
	#[must_use]
	pub fn alive(&self, id: JointId) -> bool { self.slot(id).is_some() }

	/// Rebuilds the whole table from a description, slot for slot.
	///
	/// The same contract the entity and body tables have. Nothing checks that
	/// the bodies a restored joint names exist: the solver already skips a
	/// joint whose handles do not resolve, and a description that names a body
	/// it did not also describe is a broken description rather than a case to
	/// paper over.
	///
	/// @param generations - the generation of every slot, dead ones included;
	/// its length is how many slots the table ends up with, capped at
	/// [`MAX_JOINTS`]
	/// @param entries - `(slot, joint)` for each living joint
	/// @return one handle per entry, in order, [`JointId::NONE`] for any whose
	/// slot the table could not hold
	pub fn restore(&mut self, generations: &[u32], entries: &[(usize, Joint)]) -> Vec<JointId> {
		let slots = generations.len().min(MAX_JOINTS);

		self.joints.clear();
		self.joints.resize(slots, Joint::default());
		self.generations.clear();
		self.generations
			.extend_from_slice(&generations[..slots]);
		self.alive.clear();
		self.alive.resize(slots, false);
		self.free.clear();
		self.live = 0;

		let mut handles = Vec::with_capacity(entries.len());
		for (slot, joint) in entries {
			handles.push(self.put(*slot, *joint));
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

	/// Puts one joint back into a slot a [`restore`](Self::restore) has just
	/// sized the table for.
	fn put(&mut self, slot: usize, joint: Joint) -> JointId {
		let (Ok(index), Some(alive)) = (u32::try_from(slot), self.alive.get_mut(slot)) else {
			return JointId::NONE;
		};

		if *alive {
			return JointId::NONE;
		}

		*alive = true;
		self.joints[slot] = joint;
		self.generations[slot] = self.generations[slot].max(1);
		self.live += 1;

		JointId {
			index,
			generation: self.generations[slot],
		}
	}

	/// How many slots the table has ever handed out.
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

	/// Destroys everything.
	pub fn clear(&mut self) {
		for slot in 0..self.alive.len() {
			if !self.alive[slot] {
				continue;
			}

			self.alive[slot] = false;
			self.joints[slot] = Joint::default();
			if let Ok(index) = u32::try_from(slot) {
				self.free.push(index);
			}
		}

		self.live = 0;
	}

	/// Breaks a joint.
	///
	/// @param id - the handle to break
	/// @return `true` if it existed, `false` if the handle was stale
	pub fn despawn(&mut self, id: JointId) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.alive[slot] = false;
		self.joints[slot] = Joint::default();
		self.free.push(id.index);
		self.live -= 1;

		true
	}

	/// Breaks every joint holding a body.
	///
	/// What a game calls when it deletes a prop: a joint to a body that is gone
	/// holds nothing, and the solver would go on visiting it every step.
	///
	/// @param body - the body being removed
	/// @return how many joints were broken
	pub fn forget(&mut self, body: BodyId) -> usize {
		let doomed: Vec<JointId> = self
			.iter()
			.filter(|(_, joint)| joint.first == body || joint.second == body)
			.map(|(id, _)| id)
			.collect();
		let broken = doomed.len();

		for id in doomed {
			self.despawn(id);
		}

		broken
	}

	/// A joint.
	#[must_use]
	pub fn get(&self, id: JointId) -> Option<&Joint> {
		self.slot(id).map(|slot| &self.joints[slot])
	}

	/// A joint, to change.
	pub fn get_mut(&mut self, id: JointId) -> Option<&mut Joint> {
		self.slot(id).map(|slot| &mut self.joints[slot])
	}

	/// Whether there are none at all.
	#[must_use]
	pub const fn is_empty(&self) -> bool { self.live == 0 }

	/// Every living joint, with its handle.
	pub fn iter(&self) -> impl Iterator<Item = (JointId, &Joint)> {
		self.joints
			.iter()
			.enumerate()
			.filter(|&(slot, _)| self.alive[slot])
			.filter_map(|(slot, joint)| {
				let index = u32::try_from(slot).ok()?;

				Some((
					JointId {
						index,
						generation: self.generations[slot],
					},
					joint,
				))
			})
	}

	/// How many joints exist.
	#[must_use]
	pub const fn len(&self) -> usize { self.live }

	/// Creates a joint.
	///
	/// @param joint - what to create
	/// @return its handle, or [`JointId::NONE`] if the table is full
	pub fn spawn(&mut self, joint: Joint) -> JointId {
		let Some(slot) = self.take_slot() else {
			return JointId::NONE;
		};

		let Ok(index) = u32::try_from(slot) else {
			return JointId::NONE;
		};

		self.generations[slot] = self.generations[slot].saturating_add(1);
		self.alive[slot] = true;
		self.joints[slot] = joint;
		self.live += 1;

		JointId {
			index,
			generation: self.generations[slot],
		}
	}

	/// The slot a handle addresses, if it is still the joint it was.
	fn slot(&self, id: JointId) -> Option<usize> {
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

		if self.joints.len() >= MAX_JOINTS {
			return None;
		}

		self.joints.push(Joint::default());
		self.generations.push(0);
		self.alive.push(false);

		Some(self.joints.len() - 1)
	}
}

impl Default for Joints {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::*;

	fn body(index: u32) -> BodyId {
		let mut bodies = super::super::physics::Bodies::new();

		for _ in 0..=index {
			bodies.spawn(super::super::physics::Body::default());
		}

		bodies
			.iter()
			.nth(usize::try_from(index).unwrap_or(0))
			.expect("just spawned")
			.0
	}

	#[test]
	fn a_stale_joint_handle_does_not_pick_up_its_successor() {
		let mut joints = Joints::new();
		let first = joints.spawn(Joint::default());

		assert!(joints.despawn(first), "it was there");

		let second = joints.spawn(Joint::default());

		assert_eq!(second.slot(), first.slot(), "the slot really is reused");
		assert!(joints.get(first).is_none(), "and the old handle no longer resolves");
		assert!(joints.get(second).is_some(), "while the new one does");
	}

	#[test]
	fn breaking_everything_that_holds_a_body_leaves_the_rest_alone() {
		let mut joints = Joints::new();
		let (one, other, spare) = (body(0), body(1), body(2));

		joints.spawn(Joint::weld(one, other, (Vec3::ZERO, Vec3::ZERO)));
		joints.spawn(Joint::rope(other, one, (Vec3::ZERO, Vec3::ZERO), 1.0));
		let untouched = joints.spawn(Joint::weld(other, spare, (Vec3::ZERO, Vec3::ZERO)));

		assert_eq!(joints.forget(one), 2, "both of the ones naming it");
		assert_eq!(joints.len(), 1, "and only those");
		assert!(joints.get(untouched).is_some(), "the third is still there");
	}

	#[test]
	fn the_table_stops_rather_than_growing_without_end() {
		let mut joints = Joints::new();

		for _ in 0..MAX_JOINTS {
			assert!(joints.spawn(Joint::default()).is_some(), "up to the bound they all land");
		}

		assert_eq!(joints.spawn(Joint::default()), JointId::NONE, "and past it none do");
	}

	#[test]
	fn each_kind_knows_how_many_constraints_it_is() {
		let anchors = (Vec3::ZERO, Vec3::ZERO);
		let none = BodyId::NONE;

		assert_eq!(Joint::rope(none, none, anchors, 1.0).constraints(), 1, "a rope is a length");
		assert_eq!(Joint::weld(none, none, anchors).constraints(), 6, "a weld is everything");
		assert_eq!(
			Joint::axis(none, none, anchors, Vec3::Y).constraints(),
			5,
			"a hinge is everything but the one it turns about"
		);
	}

	#[test]
	fn a_zeroed_handle_refers_to_nothing() {
		let joints = Joints::new();

		assert!(!joints.alive(JointId::NONE), "zero is never a live generation");
		assert!(joints.is_empty(), "and an empty table is empty");
	}
}
