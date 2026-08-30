//! Joints: the ways two bodies can be told to stay near each other.
//!
//! Four of them. Three are what a sandbox builds with - a rope that stops two
//! things drifting apart, a weld that stops them moving at all, and a hinge
//! that leaves one axis free - and everything a physics gun does is made of
//! those. The fourth holds two anchors together and lets the bodies turn any
//! way they like, which is what a limb of a ragdoll hangs off.
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

use super::{names::Names, physics::BodyId};
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
///
/// **Declaration order is the order a scene file writes them in**, so a new
/// kind goes on the end rather than in the middle of the list it belongs to.
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

	/// Holds the anchors together and lets them turn any way at all.
	///
	/// Three constraints: a weld's position and none of its angle. It is the
	/// joint a ragdoll's limbs hang off, and it is deliberately the *simplest*
	/// one that can be: with nothing stopping a limb turning, a ragdoll made
	/// of these crumples - an elbow folds the wrong way, a head lies on a
	/// chest. Every engine checked says the same thing about its own version
	/// of this, and every one of them ships it as the default all the same,
	/// because the answer is a limit on the angle rather than a different
	/// joint. Those limits are not here yet.
	Ball,
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

	/// How stiff the spring holding it together is, in hertz.
	///
	/// **Zero is rigid**, and that is a flag rather than a limit: a spring of
	/// no frequency would be no spring at all, and what zero selects instead is
	/// the hard constraint this table has always solved, bit for bit. It is the
	/// convention the field uses, and it is what makes every joint written
	/// before this field existed behave exactly as it did.
	///
	/// Anything above zero is a *soft* constraint: the joint is allowed to be
	/// out, and pulls itself back at this rate. That is what a physics gun
	/// carrying a prop is, and what a weld that gives under load is. The number
	/// is a frequency rather than a spring constant so that it does not have to
	/// be retuned for every mass: a stiffness of ten holds a feather and an
	/// anvil equally well.
	///
	/// **The step rate is the ceiling on what this can mean.** A spring is
	/// integrated once a step, so a frequency approaching half the step rate is
	/// already rigid as far as the solver can tell, and numbers taken from an
	/// engine running a different rate do not transfer.
	pub stiffness: f32,

	/// How quickly the spring stops ringing, as a ratio.
	///
	/// One is critical: the joint arrives and stops. Below it the prop
	/// overshoots and comes back; above it the joint is sluggish and never
	/// quite arrives. Read only when [`stiffness`](Self::stiffness) is
	/// something other than zero, because a rigid constraint has nothing to
	/// damp.
	pub damping: f32,

	/// The most force it may spend on holding the anchors together, over one
	/// whole step, as an impulse.
	///
	/// **Zero is no ceiling at all.** That is a sentinel and it is worth the
	/// one it costs: the honest value would be an infinity, a `.scene` is JSON
	/// and JSON has no spelling for one, and a joint permitted to spend nothing
	/// is a joint that does not hold - which is what deleting it is for. So the
	/// unusable end of the range carries the meaning.
	///
	/// It is a cap on *effort*, not a breaking strain: a joint that reaches it
	/// stops pulling harder and goes on existing. Something too heavy to lift
	/// is therefore something that stays where it is, rather than something
	/// that tears the gun out of your hands.
	///
	/// Spent across the whole step rather than per pass, because the number of
	/// passes is a console variable about *quality* and a ceiling that moved
	/// when it did would not be a ceiling.
	pub max_impulse: f32,

	/// The same, for the half that holds the two orientations together.
	///
	/// Two numbers rather than one because the units differ: one is an impulse
	/// and the other an angular impulse, and no single number is both.
	pub max_torque: f32,
}

impl Joint {
	/// How quickly a spring settles unless it says otherwise.
	pub const DAMPING: f32 = 1.0;
	/// How much a joint may spend unless it says otherwise, which is all of it.
	pub const NO_CEILING: f32 = 0.0;
	/// How stiff a joint is unless it says otherwise, which is perfectly.
	pub const RIGID: f32 = 0.0;

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

	/// A ball and socket between two bodies.
	///
	/// @param first - one body
	/// @param second - the other
	/// @param anchors - where it attaches on each, in that body's own space
	#[must_use]
	pub const fn ball(first: BodyId, second: BodyId, anchors: (Vec3, Vec3)) -> Self {
		Self::new(JointKind::Ball, first, second, anchors)
	}

	/// A joint of a kind, between two bodies.
	///
	/// @param kind - which of the four
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
			stiffness: Self::RIGID,
			damping: Self::DAMPING,
			max_impulse: Self::NO_CEILING,
			max_torque: Self::NO_CEILING,
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

	/// The same joint, held by a spring rather than rigidly.
	///
	/// @param stiffness - how quickly it pulls itself back, in hertz
	/// @param damping - how quickly the spring stops ringing, one being
	/// critical
	#[must_use]
	pub const fn sprung(mut self, stiffness: f32, damping: f32) -> Self {
		self.stiffness = stiffness;
		self.damping = damping;

		self
	}

	/// The same joint, with a limit on what it may spend in one step.
	///
	/// @param max_impulse - the most it may pull with, or zero for no ceiling
	/// @param max_torque - the most it may turn with, or zero for no ceiling
	#[must_use]
	pub const fn capped(mut self, max_impulse: f32, max_torque: f32) -> Self {
		self.max_impulse = max_impulse;
		self.max_torque = max_torque;

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
			| JointKind::Ball => 3,
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
	/// What each slot is called, or the empty string. @ref
	/// [`names`](crate::abi::names).
	names: Names,
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
			names: Names::new(),
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
		self.names.reset(slots);
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

	/// What a joint is called, or the empty string. @ref
	/// [`names`](crate::abi::names).
	#[must_use]
	pub fn name(&self, id: JointId) -> &str {
		self.slot(id)
			.map_or("", |slot| self.names.at(slot))
	}

	/// Names a joint, cutting anything past
	/// [`MAX_NAME`](crate::abi::MAX_NAME).
	///
	/// @param id - what to name
	/// @param name - what to call it; empty clears the name
	/// @return `true` if the handle resolved
	pub fn set_name(&mut self, id: JointId, name: &str) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.names.set(slot, name);

		true
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
		// the one place a name is cleared. @ref `abi::names`.
		self.names.set(slot, "");
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
		self.names.push();
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
		assert_eq!(
			Joint::ball(none, none, anchors).constraints(),
			3,
			"and a ball is a weld with the angle taken out"
		);
	}

	#[test]
	fn a_ball_is_a_weld_with_nothing_said_about_the_angle() {
		let anchors = (Vec3::X, Vec3::NEG_X);
		let ball = Joint::ball(BodyId::NONE, BodyId::NONE, anchors);
		let weld = Joint::weld(BodyId::NONE, BodyId::NONE, anchors);

		assert_eq!(ball.kind, JointKind::Ball, "it is its own kind");
		assert_eq!(ball.first_anchor, weld.first_anchor, "with a weld's anchors");
		assert_eq!(ball.second_anchor, weld.second_anchor);
		assert_eq!(
			Joint { kind: JointKind::Weld, ..ball },
			weld,
			"and nothing else about it differs"
		);
	}

	#[test]
	fn a_zeroed_handle_refers_to_nothing() {
		let joints = Joints::new();

		assert!(!joints.alive(JointId::NONE), "zero is never a live generation");
		assert!(joints.is_empty(), "and an empty table is empty");
	}

	#[test]
	fn a_joint_carries_a_name_and_a_reused_slot_does_not_inherit_it() {
		let mut joints = Joints::new();
		let anchors = (Vec3::ZERO, Vec3::Y);
		let old = joints.spawn(Joint::rope(BodyId::NONE, BodyId::NONE, anchors, 1.0));

		assert_eq!(joints.name(old), "", "a joint starts unnamed");
		assert!(joints.set_name(old, "rope"), "and can be told what it is");
		assert_eq!(joints.name(old), "rope");

		joints.despawn(old);
		let new = joints.spawn(Joint::rope(BodyId::NONE, BodyId::NONE, anchors, 1.0));

		assert_eq!(joints.name(old), "", "the stale handle reaches nothing");
		assert_eq!(joints.name(new), "", "and the slot came back unnamed");
		assert!(!joints.set_name(old, "ghost"), "naming a stale handle does nothing");
	}

	#[test]
	fn every_joint_array_stays_the_same_length() {
		let mut joints = Joints::new();
		let anchors = (Vec3::ZERO, Vec3::Y);
		let rope = || Joint::rope(BodyId::NONE, BodyId::NONE, anchors, 1.0);
		let mut ids = Vec::new();
		for _ in 0..5 {
			ids.push(joints.spawn(rope()));
		}

		joints.despawn(ids[1]);
		joints.spawn(rope());
		joints.clear();
		joints.spawn(rope());

		let length = joints.joints.len();

		assert_eq!(joints.alive.len(), length, "the table is one table");
		assert_eq!(joints.generations.len(), length, "and every array in it agrees");
		assert_eq!(joints.names.slots(), length, "and every array in it agrees");
	}
}
