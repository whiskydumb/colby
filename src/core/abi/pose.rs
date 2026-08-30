//! Poses: where a skeleton's bones are right now, and the matrices that come
//! out of them.
//!
//! A [`SkeletonData`](super::skeleton::SkeletonData) says what bones there are
//! and where they sit with nothing moving them. A pose says where they are this
//! step. The two are apart because one is an asset shared by everything that
//! wears it and the other is one character's own state, and because a skeleton
//! is read from a file while a pose is written sixty times a second.
//!
//! **A pose is a table like the bodies and the joints, not a field on an
//! entity.** Two entities really do share one: a model of two materials becomes
//! two entities, both moved by the same bones, and a pose kept on an entity
//! would be two poses drifting apart. So
//! [`Renderable::pose`](super::Renderable) is a handle into this table,
//! generational for the same reason a body's is - a pose is destroyed and its
//! slot reused, and a game holding a stale handle must miss rather than find
//! whoever moved in.
//!
//! **It carries its past, exactly as the entity table does.** The simulation
//! writes bones once a step and the picture is drawn many times between two
//! steps, so what the renderer wants is somewhere between the two. Same three
//! calls, same meanings: [`Poses::advance`] moves the present into the past
//! before a step, [`Poses::snap`] says a bone jumped rather than traveled, and
//! [`Poses::settle`] applies the snaps at the end of the step. Without it a
//! character animated at sixty steps a second is drawn in sixty distinct
//! attitudes however fast the frames come.
//!
//! **Resolving is two forward passes over one buffer and nothing else.**
//! [`Poses::skinning`] writes each bone's model-space matrix, which needs only
//! its parent's - already done, because a skeleton is sorted parents first -
//! and then multiplies each by its inverse bind in place. No recursion, no
//! scratch buffer, and the result is exactly what the vertex stage multiplies a
//! vertex by.

use super::{
	entity::Transform,
	skeleton::{Bone, NO_PARENT, SkeletonId},
};
use crate::{
	bytemuck::{Pod, Zeroable},
	glam::Mat4,
};

/// The most poses that may exist at once.
///
/// A pose is two transforms per bone, so a two-hundred-bone one is sixteen
/// kilobytes and this many of them is a few megabytes. Bounded rather than
/// unbounded for the reason every other table here is: gameplay code is code
/// that is expected to be wrong sometimes, and a reload that poses a crowd
/// every time should run out of slots rather than out of memory.
pub const MAX_POSES: usize = 256;

/// A handle to a pose.
///
/// Generational, like [`BodyId`](super::physics::BodyId) and
/// [`JointId`](super::joint::JointId), and unlike an asset handle: a pose is
/// destroyed when its character is, and the slot is handed to somebody else.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct PoseId {
	index: u32,
	generation: u32,
}

impl PoseId {
	/// A handle that refers to nothing, and always will.
	pub const NONE: Self = Self { index: 0, generation: 0 };

	/// Which occupant of that slot this handle names.
	#[must_use]
	pub const fn generation(self) -> u32 { self.generation }

	/// Whether it could refer to anything at all.
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

/// Where one skeleton's bones are.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pose {
	/// Which skeleton these are the bones of.
	pub skeleton: SkeletonId,

	/// Each bone's transform relative to its parent, this step.
	///
	/// Local rather than model space, because that is what an animation writes
	/// and what a blend produces; the model-space walk happens once, in
	/// [`Poses::skinning`], and its answer is not kept.
	pub locals: Vec<Transform>,

	/// The same at the previous step, which the picture is drawn between.
	///
	/// Written by [`Poses::advance`] and by nothing else, except the snaps
	/// [`Poses::settle`] applies.
	pub previous: Vec<Transform>,
}

impl Pose {
	/// A pose of a skeleton, with every bone left where the skeleton put it.
	///
	/// @param skeleton - the handle this poses
	/// @param bones - that skeleton's bones, for their rests
	#[must_use]
	pub fn resting(skeleton: SkeletonId, bones: &[Bone]) -> Self {
		let locals: Vec<Transform> = bones.iter().map(|bone| bone.rest).collect();

		Self {
			skeleton,
			previous: locals.clone(),
			locals,
		}
	}

	/// How many bones it holds.
	#[must_use]
	pub fn len(&self) -> usize { self.locals.len() }

	/// Whether it holds none.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.locals.is_empty() }

	/// Moves one bone, doing nothing if there is no such bone.
	///
	/// @param bone - its index in the skeleton
	/// @param local - where to put it, relative to its parent
	/// @return `true` if the bone was there
	pub fn set(&mut self, bone: u16, local: Transform) -> bool {
		let Some(slot) = self.locals.get_mut(usize::from(bone)) else {
			return false;
		};

		*slot = local;

		true
	}

	/// Puts every bone back where its skeleton has it.
	///
	/// What a pose is created holding, and what a game calls when it wants to
	/// start over. Also the one call that fixes a pose whose skeleton was
	/// reloaded with a different number of bones in it.
	///
	/// @param bones - the skeleton's bones
	pub fn rest(&mut self, bones: &[Bone]) {
		self.locals.clear();
		self.locals
			.extend(bones.iter().map(|bone| bone.rest));
	}
}

/// Writes every living pose's past to match its present.
///
/// What both [`Poses::advance`] and a whole-table snap do, and the only place
/// that loop is written.
fn catch_up(poses: &mut [Pose], alive: &[bool]) {
	for (pose, alive) in poses.iter_mut().zip(alive) {
		if *alive {
			pose.previous.clone_from(&pose.locals);
		}
	}
}

/// Every pose in the world.
///
/// Shaped exactly like [`Joints`](super::joint::Joints) and
/// [`Bodies`](super::physics::Bodies): a bounded, generational table of plain
/// data that knows nothing about who animates it.
#[derive(Clone, Debug, Default)]
pub struct Poses {
	poses: Vec<Pose>,
	generations: Vec<u32>,
	alive: Vec<bool>,
	free: Vec<u32>,
	live: usize,
	/// Slots whose past is to be caught up with their present at the end of
	/// this step. @ref [`Self::snap`].
	pending: Vec<usize>,
	pending_all: bool,
}

impl Poses {
	/// An empty table.
	#[must_use]
	pub const fn new() -> Self {
		Self {
			poses: Vec::new(),
			generations: Vec::new(),
			alive: Vec::new(),
			free: Vec::new(),
			live: 0,
			pending: Vec::new(),
			pending_all: false,
		}
	}

	/// Creates a pose, resting.
	///
	/// @param pose - what to create; [`Pose::resting`] is the usual argument
	/// @return its handle, or [`PoseId::NONE`] if the table is full
	pub fn spawn(&mut self, pose: Pose) -> PoseId {
		let Some(slot) = self.take_slot() else {
			return PoseId::NONE;
		};

		let Ok(index) = u32::try_from(slot) else {
			return PoseId::NONE;
		};

		self.generations[slot] = self.generations[slot].saturating_add(1);
		self.alive[slot] = true;
		self.poses[slot] = pose;
		self.live += 1;

		PoseId {
			index,
			generation: self.generations[slot],
		}
	}

	/// Destroys a pose.
	///
	/// @param id - the handle to destroy
	/// @return `true` if it existed, `false` if the handle was stale
	pub fn despawn(&mut self, id: PoseId) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.alive[slot] = false;
		self.poses[slot] = Pose::default();
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
			self.poses[slot] = Pose::default();

			if let Ok(index) = u32::try_from(slot) {
				self.free.push(index);
			}
		}

		self.live = 0;
		self.pending.clear();
		self.pending_all = false;
	}

	/// A pose.
	#[must_use]
	pub fn get(&self, id: PoseId) -> Option<&Pose> { self.slot(id).map(|slot| &self.poses[slot]) }

	/// A pose, to move.
	pub fn get_mut(&mut self, id: PoseId) -> Option<&mut Pose> {
		self.slot(id).map(|slot| &mut self.poses[slot])
	}

	/// Whether a handle refers to a living pose.
	#[must_use]
	pub fn alive(&self, id: PoseId) -> bool { self.slot(id).is_some() }

	/// How many poses exist.
	#[must_use]
	pub const fn len(&self) -> usize { self.live }

	/// Whether there are none at all.
	#[must_use]
	pub const fn is_empty(&self) -> bool { self.live == 0 }

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

	/// Every living pose, with its handle.
	pub fn iter(&self) -> impl Iterator<Item = (PoseId, &Pose)> {
		self.poses
			.iter()
			.enumerate()
			.filter(|&(slot, _)| self.alive[slot])
			.filter_map(|(slot, pose)| {
				let index = u32::try_from(slot).ok()?;

				Some((
					PoseId {
						index,
						generation: self.generations[slot],
					},
					pose,
				))
			})
	}

	/// Moves every pose's present into its past, ready for another step.
	///
	/// The host calls this before the game's `update`, beside
	/// [`Entities::advance`](super::entity::Entities::advance) and for exactly
	/// the same reason.
	pub fn advance(&mut self) {
		catch_up(&mut self.poses, &self.alive);
		self.pending.clear();
		self.pending_all = false;
	}

	/// Applies everything that asked not to be interpolated this step.
	///
	/// Deferred rather than done at the moment it is asked for, so that a snap
	/// is independent of where in the step it was called - the same rule the
	/// entity table follows, and the reason neither of them needs a flag that
	/// can be left set by mistake.
	pub fn settle(&mut self) {
		if self.pending_all {
			catch_up(&mut self.poses, &self.alive);
			self.pending.clear();
			self.pending_all = false;

			return;
		}

		for &slot in &self.pending {
			if self.alive.get(slot) == Some(&true)
				&& let Some(pose) = self.poses.get_mut(slot)
			{
				pose.previous.clone_from(&pose.locals);
			}
		}

		self.pending.clear();
	}

	/// Declares that a pose changed discontinuously.
	///
	/// A clip that restarted, a ragdoll taking over, a character teleported
	/// into a different attitude: anything the bones did not travel to.
	/// Without this the renderer draws the journey.
	///
	/// @param id - the pose that jumped
	/// @return `true` if the handle resolved
	pub fn snap(&mut self, id: PoseId) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.pending.push(slot);

		true
	}

	/// The same for every pose, for when a scene is replaced rather than moved.
	pub fn snap_all(&mut self) { self.pending_all = true; }

	/// Rebuilds the whole table from a description, slot for slot.
	///
	/// The same contract the entity, body and joint tables have. Nothing here
	/// checks that a restored pose's skeleton exists or has that many bones:
	/// [`skinning`](Self::skinning) walks the bones it is handed and falls
	/// back per bone, so a description naming a skeleton that did not load
	/// draws a character standing still rather than nothing at all.
	///
	/// @param generations - the generation of every slot, dead ones included;
	/// its length is how many slots the table ends up with, capped at
	/// [`MAX_POSES`]
	/// @param entries - `(slot, pose)` for each living pose
	/// @return one handle per entry, in order, [`PoseId::NONE`] for any whose
	/// slot the table could not hold
	pub fn restore(&mut self, generations: &[u32], entries: &[(usize, Pose)]) -> Vec<PoseId> {
		let slots = generations.len().min(MAX_POSES);

		self.poses.clear();
		self.poses.resize(slots, Pose::default());
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

		for (slot, pose) in entries {
			handles.push(self.put(*slot, pose.clone()));
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

	/// The matrices one pose hands the vertex stage, appended to a buffer.
	///
	/// One per bone, `model_of(bone) * inverse_bind`, which is the identity
	/// for every bone of a pose left resting - so an unanimated character
	/// stands in the shape it was modeled in rather than folded up at the
	/// origin.
	///
	/// Two forward passes over the same memory. The first needs each bone's
	/// parent to be done already, which it is, because a skeleton is sorted
	/// parents first and a file whose bones are not is refused. The second
	/// multiplies each in place, which cannot be folded into the first: a
	/// bone's children want its model matrix, not its skinning one.
	///
	/// A pose holding fewer bones than the skeleton it names - a skeleton
	/// reloaded wider, a description restored against another build - falls
	/// back to that bone's rest rather than dropping the character.
	///
	/// @param id - the pose to resolve
	/// @param bones - its skeleton's bones, parents first
	/// @param t - how far past the previous step this frame sits, `0.0 ..= 1.0`
	/// @param out - the buffer to append to
	/// @return how many matrices were appended
	pub fn skinning(&self, id: PoseId, bones: &[Bone], t: f32, out: &mut Vec<Mat4>) -> usize {
		let Some(pose) = self.get(id) else {
			return 0;
		};

		let base = out.len();
		let blend = t.clamp(0.0, 1.0);

		for (index, bone) in bones.iter().enumerate() {
			let local = match (pose.locals.get(index), pose.previous.get(index)) {
				| (Some(is), Some(was)) => was.lerp(*is, blend),
				// a pose narrower than its skeleton, and a bone with no past
				// is drawn where it is rather than not at all.
				| (Some(is), None) => *is,
				| (None, _) => bone.rest,
			}
			.matrix();

			let model = if bone.parent == NO_PARENT {
				local
			} else {
				out.get(base + usize::from(bone.parent))
					.copied()
					.unwrap_or(Mat4::IDENTITY)
					* local
			};

			out.push(model);
		}

		for (matrix, bone) in out
			.get_mut(base..)
			.unwrap_or_default()
			.iter_mut()
			.zip(bones)
		{
			*matrix *= bone.inverse_bind;
		}

		bones.len()
	}

	/// Puts one pose back into a slot a [`restore`](Self::restore) has just
	/// sized the table for.
	fn put(&mut self, slot: usize, pose: Pose) -> PoseId {
		let (Ok(index), Some(alive)) = (u32::try_from(slot), self.alive.get_mut(slot)) else {
			return PoseId::NONE;
		};

		if *alive {
			return PoseId::NONE;
		}

		*alive = true;
		self.poses[slot] = pose;
		self.generations[slot] = self.generations[slot].max(1);
		self.live += 1;

		PoseId {
			index,
			generation: self.generations[slot],
		}
	}

	/// The slot a handle addresses, if it is still the pose it was.
	fn slot(&self, id: PoseId) -> Option<usize> {
		if !id.is_some() {
			return None;
		}

		let slot = usize::try_from(id.index).ok()?;

		(self.alive.get(slot) == Some(&true)
			&& self.generations.get(slot) == Some(&id.generation))
		.then_some(slot)
	}

	/// A free slot, taken from the list or appended.
	fn take_slot(&mut self) -> Option<usize> {
		if let Some(index) = self.free.pop() {
			return usize::try_from(index).ok();
		}

		if self.alive.len() >= MAX_POSES {
			return None;
		}

		self.poses.push(Pose::default());
		self.generations.push(0);
		self.alive.push(false);

		Some(self.alive.len() - 1)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		abi::skeleton::SkeletonData,
		glam::{Quat, Vec3},
	};

	/// A shoulder, an elbow a unit along `x` from it, a wrist a unit past that,
	/// with the inverse binds worked out from the rests the way an importer
	/// does.
	fn arm() -> SkeletonData {
		let mut data = SkeletonData {
			bones: vec![
				Bone {
					name: "shoulder".to_owned(),
					..Bone::default()
				},
				Bone {
					name: "elbow".to_owned(),
					parent: 0,
					rest: Transform::at(Vec3::X),
					..Bone::default()
				},
				Bone {
					name: "wrist".to_owned(),
					parent: 1,
					rest: Transform::at(Vec3::X),
					..Bone::default()
				},
			],
		};
		let mut model: Vec<Mat4> = Vec::with_capacity(data.len());

		for bone in &data.bones {
			let local = bone.rest.matrix();
			let world = if bone.parent == NO_PARENT {
				local
			} else {
				model[usize::from(bone.parent)] * local
			};

			model.push(world);
		}

		for (bone, world) in data.bones.iter_mut().zip(&model) {
			bone.inverse_bind = world.inverse();
		}

		data
	}

	/// A table holding one resting pose of that arm.
	fn posed() -> (Poses, PoseId, SkeletonData) {
		let arm = arm();
		let mut poses = Poses::new();
		let id = poses.spawn(Pose::resting(SkeletonId::new(1), &arm.bones));

		(poses, id, arm)
	}

	/// The matrices one pose resolves to.
	fn matrices(poses: &Poses, id: PoseId, arm: &SkeletonData, t: f32) -> Vec<Mat4> {
		let mut out = Vec::new();
		let written = poses.skinning(id, &arm.bones, t, &mut out);

		assert_eq!(written, out.len(), "it says how many it wrote");

		out
	}

	#[test]
	fn a_resting_pose_leaves_every_vertex_exactly_where_it_was_drawn() {
		let (poses, id, arm) = posed();

		for (index, matrix) in matrices(&poses, id, &arm, 1.0).iter().enumerate() {
			assert!(
				matrix.abs_diff_eq(Mat4::IDENTITY, 1.0e-5),
				"bone {index} moves nothing, got {matrix:?}"
			);
		}
	}

	#[test]
	fn turning_a_bone_carries_everything_below_it() {
		let (mut poses, id, arm) = posed();
		let turn = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);

		poses
			.get_mut(id)
			.expect("the pose is there")
			.set(0, Transform { rotation: turn, ..Transform::IDENTITY });
		poses.snap(id);
		poses.settle();

		// the wrist sat two units along x; a quarter turn about z about the
		// shoulder puts it two units along y, and the matrix that does that to
		// a vertex is the one the bone hands the vertex stage.
		let wrist = matrices(&poses, id, &arm, 1.0)[2];
		let moved = wrist.transform_point3(Vec3::new(2.0, 0.0, 0.0));

		assert!(
			moved.abs_diff_eq(Vec3::new(0.0, 2.0, 0.0), 1.0e-5),
			"a vertex at the wrist follows the shoulder, got {moved}"
		);
	}

	#[test]
	fn the_picture_is_drawn_between_the_two_steps_it_sits_in() {
		let (mut poses, id, arm) = posed();

		poses.advance();
		poses
			.get_mut(id)
			.expect("the pose is there")
			.set(1, Transform::at(Vec3::new(3.0, 0.0, 0.0)));

		let elbow = |t| matrices(&poses, id, &arm, t)[1].transform_point3(Vec3::X);

		assert!(
			elbow(0.0).abs_diff_eq(Vec3::X, 1.0e-5),
			"at the previous step it had not moved, got {}",
			elbow(0.0)
		);
		assert!(
			elbow(1.0).abs_diff_eq(Vec3::new(3.0, 0.0, 0.0), 1.0e-5),
			"at this one it has, got {}",
			elbow(1.0)
		);
		assert!(
			elbow(0.5).abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1.0e-5),
			"and halfway between, halfway, got {}",
			elbow(0.5)
		);
	}

	#[test]
	fn the_past_a_frame_blends_from_is_the_last_step_and_not_the_first() {
		let (mut poses, id, arm) = posed();
		let elbow = |poses: &Poses, t| matrices(poses, id, &arm, t)[1].transform_point3(Vec3::X);

		// one step: the elbow goes out to two.
		poses.advance();
		poses
			.get_mut(id)
			.expect("the pose is there")
			.set(1, Transform::at(Vec3::new(2.0, 0.0, 0.0)));

		// another: out to four. What the frames after it blend from is two,
		// not the one it started at - which is the whole of what moving the
		// present into the past before each step is for, and is invisible
		// until a pose has been through more than one.
		poses.advance();
		poses
			.get_mut(id)
			.expect("the pose is there")
			.set(1, Transform::at(Vec3::new(4.0, 0.0, 0.0)));

		assert!(
			elbow(&poses, 0.0).abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1.0e-5),
			"at the previous step it was at two, got {}",
			elbow(&poses, 0.0)
		);
		assert!(
			elbow(&poses, 0.5).abs_diff_eq(Vec3::new(3.0, 0.0, 0.0), 1.0e-5),
			"so halfway through is three rather than two and a half, got {}",
			elbow(&poses, 0.5)
		);
	}

	#[test]
	fn a_snapped_pose_is_not_drawn_traveling() {
		let (mut poses, id, arm) = posed();

		poses.advance();
		poses
			.get_mut(id)
			.expect("the pose is there")
			.set(1, Transform::at(Vec3::new(3.0, 0.0, 0.0)));
		poses.snap(id);
		poses.settle();

		let elbow = matrices(&poses, id, &arm, 0.0)[1].transform_point3(Vec3::X);

		assert!(
			elbow.abs_diff_eq(Vec3::new(3.0, 0.0, 0.0), 1.0e-5),
			"the past was caught up with the present, so every blend is the present: {elbow}"
		);
	}

	#[test]
	fn snapping_everything_is_the_same_for_every_pose_in_the_table() {
		let arm = arm();
		let mut poses = Poses::new();
		let first = poses.spawn(Pose::resting(SkeletonId::new(1), &arm.bones));
		let second = poses.spawn(Pose::resting(SkeletonId::new(1), &arm.bones));

		poses.advance();

		for id in [first, second] {
			poses
				.get_mut(id)
				.expect("both are there")
				.set(1, Transform::at(Vec3::new(3.0, 0.0, 0.0)));
		}

		poses.snap_all();
		poses.settle();

		for id in [first, second] {
			let elbow = matrices(&poses, id, &arm, 0.0)[1].transform_point3(Vec3::X);

			assert!(elbow.abs_diff_eq(Vec3::new(3.0, 0.0, 0.0), 1.0e-5), "got {elbow}");
		}
	}

	#[test]
	fn a_snap_holds_until_the_end_of_the_step_whenever_it_was_asked_for() {
		let (mut poses, id, arm) = posed();

		poses.advance();
		poses.snap(id);
		// asked for before the write it is about, which is the case a flag on
		// the pose would get wrong.
		poses
			.get_mut(id)
			.expect("the pose is there")
			.set(1, Transform::at(Vec3::new(3.0, 0.0, 0.0)));
		poses.settle();

		let elbow = matrices(&poses, id, &arm, 0.0)[1].transform_point3(Vec3::X);

		assert!(elbow.abs_diff_eq(Vec3::new(3.0, 0.0, 0.0), 1.0e-5), "got {elbow}");
	}

	#[test]
	fn two_poses_appended_to_one_buffer_do_not_read_each_others_bones() {
		let arm = arm();
		let mut poses = Poses::new();
		let turn = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
		let first = poses.spawn(Pose::resting(SkeletonId::new(1), &arm.bones));
		let second = poses.spawn(Pose::resting(SkeletonId::new(1), &arm.bones));

		// only the second one is bent, so a parent looked up by its place in
		// the whole buffer rather than in this pose's own run would carry the
		// bend into the first one - which is what a frame with two characters
		// in it does every time.
		poses
			.get_mut(second)
			.expect("the pose is there")
			.set(0, Transform { rotation: turn, ..Transform::IDENTITY });
		poses.snap_all();
		poses.settle();

		let mut out = Vec::new();
		let wrote = poses.skinning(first, &arm.bones, 1.0, &mut out);

		assert_eq!((wrote, out.len()), (3, 3), "the first pose appended to an empty buffer");

		let wrote = poses.skinning(second, &arm.bones, 1.0, &mut out);

		assert_eq!((wrote, out.len()), (3, 6), "and the second appended after it");

		for (index, matrix) in out.iter().take(3).enumerate() {
			assert!(
				matrix.abs_diff_eq(Mat4::IDENTITY, 1.0e-5),
				"bone {index} of the pose nobody bent moves nothing, got {matrix:?}"
			);
		}

		let wrist = out[5].transform_point3(Vec3::new(2.0, 0.0, 0.0));

		assert!(
			wrist.abs_diff_eq(Vec3::new(0.0, 2.0, 0.0), 1.0e-5),
			"and the one that was bent is, got {wrist}"
		);
	}

	#[test]
	fn a_pose_with_fewer_bones_than_its_skeleton_falls_back_to_the_rest() {
		let (mut poses, id, arm) = posed();

		poses
			.get_mut(id)
			.expect("the pose is there")
			.locals
			.truncate(1);

		let out = matrices(&poses, id, &arm, 1.0);

		assert_eq!(out.len(), 3, "one matrix per bone of the skeleton, not of the pose");

		for (index, matrix) in out.iter().enumerate() {
			assert!(
				matrix.abs_diff_eq(Mat4::IDENTITY, 1.0e-5),
				"bone {index} stands as the skeleton has it, got {matrix:?}"
			);
		}
	}

	#[test]
	fn a_handle_to_a_pose_that_is_gone_resolves_to_nothing() {
		let (mut poses, id, arm) = posed();

		assert!(poses.despawn(id), "it was there");
		assert!(!poses.alive(id), "and now it is not");
		assert!(poses.get(id).is_none());
		assert_eq!(poses.skinning(id, &arm.bones, 1.0, &mut Vec::new()), 0);
		assert!(!poses.despawn(id), "and it cannot be destroyed twice");
	}

	#[test]
	fn a_reused_slot_does_not_answer_to_the_handle_that_had_it() {
		let arm = arm();
		let mut poses = Poses::new();
		let first = poses.spawn(Pose::resting(SkeletonId::new(1), &arm.bones));

		poses.despawn(first);

		let second = poses.spawn(Pose::resting(SkeletonId::new(2), &arm.bones));

		assert_eq!(first.slot(), second.slot(), "the slot came back around");
		assert_ne!(first, second, "and the handle did not");
		assert!(poses.get(first).is_none(), "so the old one finds nothing");
		assert_eq!(
			poses.get(second).map(|pose| pose.skeleton),
			Some(SkeletonId::new(2)),
			"and the new one finds itself"
		);
	}

	#[test]
	fn the_table_runs_out_of_slots_rather_than_out_of_memory() {
		let arm = arm();
		let mut poses = Poses::new();

		for _ in 0..MAX_POSES {
			assert!(
				poses
					.spawn(Pose::resting(SkeletonId::new(1), &arm.bones))
					.is_some()
			);
		}

		assert_eq!(poses.len(), MAX_POSES);
		assert_eq!(
			poses.spawn(Pose::resting(SkeletonId::new(1), &arm.bones)),
			PoseId::NONE,
			"and the one past the end is refused rather than allocated"
		);
	}

	#[test]
	fn a_restore_puts_every_slot_and_every_generation_back() {
		let arm = arm();
		let mut poses = Poses::new();
		let mut bent = Pose::resting(SkeletonId::new(3), &arm.bones);

		bent.set(1, Transform::at(Vec3::new(9.0, 0.0, 0.0)));

		let handles = poses.restore(&[0, 4, 7], &[(2, bent.clone())]);

		assert_eq!(poses.slots(), 3, "three slots, whatever lives in them");
		assert_eq!(poses.len(), 1, "one of them occupied");
		assert_eq!(poses.generation(1), 4, "a dead slot keeps the generation it had");
		assert_eq!(handles.len(), 1);
		assert_eq!(poses.get(handles[0]), Some(&bent), "and the pose came back whole");
		assert_eq!(handles[0].generation(), 7, "under the generation it was saved with");
	}

	#[test]
	fn a_restored_table_hands_out_its_empty_slots_highest_first() {
		let arm = arm();
		let mut poses = Poses::new();

		poses.restore(&[0, 0, 5], &[(2, Pose::resting(SkeletonId::new(1), &arm.bones))]);

		let next = poses.spawn(Pose::default());

		assert_eq!(next.slot(), 1, "the free list is a stack, so the highest empty one");
	}

	#[test]
	fn resting_a_pose_puts_every_bone_back_where_the_skeleton_has_it() {
		let (mut poses, id, arm) = posed();
		let pose = poses.get_mut(id).expect("the pose is there");

		pose.set(2, Transform::at(Vec3::new(9.0, 9.0, 9.0)));
		pose.rest(&arm.bones);

		for (index, matrix) in matrices(&poses, id, &arm, 1.0).iter().enumerate() {
			assert!(
				matrix.abs_diff_eq(Mat4::IDENTITY, 1.0e-5),
				"bone {index} is back, got {matrix:?}"
			);
		}
	}

	#[test]
	fn moving_a_bone_that_is_not_there_changes_nothing_and_says_so() {
		let (mut poses, id, _) = posed();
		let pose = poses.get_mut(id).expect("the pose is there");

		assert!(pose.set(2, Transform::IDENTITY), "the wrist is a bone");
		assert!(!pose.set(40, Transform::IDENTITY), "and the fortieth is not");
		assert_eq!(pose.len(), 3, "nothing was appended by asking");
	}

	#[test]
	fn clearing_the_table_leaves_every_slot_free_and_nothing_alive() {
		let arm = arm();
		let mut poses = Poses::new();
		let id = poses.spawn(Pose::resting(SkeletonId::new(1), &arm.bones));

		poses.clear();

		assert!(poses.is_empty());
		assert!(!poses.alive(id), "the handle is stale");
		assert_eq!(poses.slots(), 1, "and the slot is still there to be handed out again");
	}
}
