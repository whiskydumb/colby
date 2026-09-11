//! Reaching: two bones turned so that the bone below them lands on a point.
//!
//! A pose worked out by [`evaluate`](super::anim::evaluate) puts a foot where
//! its clip put it, over ground the clip never saw. A stair, a slope or a
//! ledge is somewhere the foot should be and is not, and the two bones above
//! it are all it takes to put it there: the thigh swings, the knee bends, and
//! the foot comes down on the point. That is [`reach`].
//!
//! **It is a function over a finished pose, not a node in the blend tree**,
//! and for three reasons. The tree mixes whole poses in their bones' own terms
//! and never works out where a bone is, while a chain can only be bent where
//! it is. A chain solved underneath a blend would be mixed with one that was
//! not, and the foot would then land on neither. And the point comes from the
//! world, a ray fired under the foot, which a game fires and a tree does not.
//!
//! **The answer is worked out, not searched for.** Two bones of known length
//! and the distance between their ends make a triangle, and where its corner
//! goes is one line of arithmetic. Nothing iterates and nothing converges, so
//! the same pose and the same point give the same bend on every machine, which
//! is what lets a prediction replay one.
//!
//! **Only turns are written.** The root and the middle are turned so that the
//! middle and the end land where the triangle says, and the end is turned back
//! so that it faces the way it faced before: a foot planted flat stays as flat
//! as its clip made it rather than tilting with the shin. No bone's position
//! or scale is touched, so every bone keeps its length exactly.
//!
//! **The bend is a point or it is the pose's own.** Two bones reaching a point
//! can bend any way around the line from the root to it, and something has to
//! say which. A point to bend towards says so outright. With none, the chain
//! bends the way the pose already bends it, which for a leg a walk has already
//! bent is the answer wanted; and a chain the pose holds dead straight bends
//! somewhere fixed, the same way every time.
//!
//! **A point out of reach is reached towards.** The chain straightens along
//! the line to it, or folds as far as it folds when the point is nearer than
//! the two bones can come, and [`reach`] says how far short it stopped. A game
//! that lowers a pelvis so that a foot can get to the ground wants that
//! number.
//!
//! **Where the point is, is the game's.** Which bone is a foot, where the
//! ground is under it, how far to lower the body so that the lower foot can
//! get there, and how much of the answer to take are each a policy, and a
//! policy belongs to the game. It is the arrangement the controller and the
//! root motion already have: the engine answers the geometry and hands it
//! back.

use super::{
	entity::Transform,
	skeleton::{Bone, NO_PARENT},
};
use crate::glam::{Mat4, Quat, Vec3};

/// The shortest a bone may be and still be half of a chain.
///
/// In the pose's own units. A bone of no length points nowhere and has nothing
/// to turn, and a chain with one in it is two points pretending to be three.
const SHORTEST: f32 = 1.0e-6;

/// How far off the line from the root a point has to be to say which way to
/// bend, as a share of its distance from the root.
///
/// A pole on the line names no side, and one a hair's breadth off it names a
/// side by rounding, so both are taken as no pole at all rather than as a bend
/// that flips from one step to the next.
const ASIDE: f32 = 1.0e-4;

/// The nearest to its root a chain of two equal bones folds, as a share of
/// the chain's whole length.
///
/// Two equal bones fold all the way back onto the root, where the line to the
/// target points nowhere and the arithmetic divides by its length. A ten
/// thousandth of the chain is the nearest that is still a triangle.
const NEAREST: f32 = 1.0e-4;

/// One chain to bend, and where to.
///
/// Plain data a game builds when it wants a limb somewhere, the way it builds
/// a [`Tree`](super::anim::Tree) every step. Which space the two points are in
/// is the caller's: [`reach`] takes the pose's own and
/// [`World::reach`](super::World::reach) the world's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reach {
	/// The bone that is to land on the target: a foot, a hand.
	///
	/// Its parent is the chain's middle, a knee or an elbow, and that bone's
	/// parent is the chain's root, a hip or a shoulder. One index rather than
	/// three, because a two-bone chain is a bone and the two above it, and a
	/// chain named three times over is a chain that can be named wrong. Found
	/// once with [`SkeletonData::find`](super::skeleton::SkeletonData::find),
	/// as [`Tree::motion`](super::anim::Tree::motion) is.
	pub end: u16,

	/// Where the end is to land.
	pub target: Vec3,

	/// A point the middle bends towards, or `None` to bend it the way the pose
	/// already does.
	///
	/// Only which side of the line from the root to the target it lies on is
	/// read, so a point well out in front of a knee says "forwards" as plainly
	/// as one on the knee's own plane. A point on that line, or one that is not
	/// a number, says nothing and is taken as `None`.
	pub pole: Option<Vec3>,

	/// How far from the pose to the answer, `0.0 ..= 1.0`.
	///
	/// Each of the three bones is turned that share of the way from the turn
	/// the pose gave it to the one the answer wants, which is the mix a blend
	/// makes. Nought leaves the pose exactly as it was, and a number that is
	/// not one counts as nought.
	pub weight: f32,
}

impl Reach {
	/// A chain bent the whole way, the way it already bends.
	///
	/// @param end - the bone that is to land on the target
	/// @param target - where it is to land
	#[must_use]
	pub const fn new(end: u16, target: Vec3) -> Self {
		Self { end, target, pole: None, weight: 1.0 }
	}

	/// The same chain, bent towards a point.
	///
	/// @param pole - a point on the side the middle is to bend to
	#[must_use]
	pub const fn bending(mut self, pole: Vec3) -> Self {
		self.pole = Some(pole);

		self
	}

	/// The same chain, bent part of the way.
	///
	/// @param weight - how far from the pose to the answer, `0.0 ..= 1.0`
	#[must_use]
	pub const fn weighted(mut self, weight: f32) -> Self {
		self.weight = weight;

		self
	}
}

/// Bends a two-bone chain of a pose so that its end lands on a point.
///
/// Everything here is in the pose's own space, the one its bones are placed
/// in before anything carries the character anywhere: the target, the pole
/// and the distance handed back. [`World::reach`](super::World::reach) is the
/// same in the world's terms.
///
/// The three bones are read where the pose has them now, so whatever put them
/// there (a clip, a blend, a ragdoll) is what is bent from, and the lengths are
/// the distances between them as they stand. Their turns are the only thing
/// written: the root's and the middle's so that the chain lands where it was
/// bent to, and the end's so that it faces the way it did. A bone above the
/// chain or hanging below its end is not touched, and one below the end rides
/// along with it. Exact for a skeleton that scales evenly or not at all, which
/// is the bargain every transform here makes.
///
/// @param bones - the skeleton, parents before children
/// @param locals - the pose to bend, one transform a bone; three rotations of
/// it are written and nothing else
/// @param wanted - the chain, and where it is to reach
/// @param scratch - a buffer the caller keeps between calls, so that bending a
/// crowd allocates once
/// @return how far short of the target the end stops, nought when it gets
/// there; `None`, with nothing written, when the end has not two bones above
/// it in both the skeleton and the pose, when one of them has no length, or
/// when the target is not a place
pub fn reach(
	bones: &[Bone],
	locals: &mut [Transform],
	wanted: &Reach,
	scratch: &mut Vec<Mat4>,
) -> Option<f32> {
	let chain = Chain::of(bones, locals, wanted.end)?;

	if !wanted.target.is_finite() {
		return None;
	}

	lay(bones, locals, chain.end, scratch);

	let at = |bone: usize| {
		scratch
			.get(bone)
			.map(|model| model.w_axis.truncate())
	};
	let was = [at(chain.root)?, at(chain.middle)?, at(chain.end)?];
	let bent = bend(was, wanted.target, wanted.pole)?;
	// nought, less than nought and a weight that is not a number all fail this
	// test, and it is the test rather than any slerp that leaves such a pose
	// exactly as it was.
	if wanted.weight > 0.0 {
		let above = chain
			.above
			.and_then(|bone| scratch.get(bone))
			.map_or(Quat::IDENTITY, |model| model.to_scale_rotation_translation().1);
		let turns = turned(above, locals, chain, was, &bent)?;

		for (bone, turn) in [chain.root, chain.middle, chain.end]
			.into_iter()
			.zip(turns)
		{
			if let Some(local) = locals.get_mut(bone) {
				local.rotation = toward(local.rotation, turn, wanted.weight);
			}
		}
	}

	Some(bent.left)
}

/// A turn part of the way to another.
///
/// Both ends are exact rather than close, the rule
/// [`Transform::lerp`](super::entity::Transform::lerp) keeps. A slerp at its
/// far end hands back the answer to within a rounding, and the answer with
/// every sign turned over when the two turns lie in opposite hemispheres, so
/// the whole weight, and any weight past it, is the answer itself.
fn toward(from: Quat, to: Quat, weight: f32) -> Quat {
	if weight >= 1.0 { to } else { from.slerp(to, weight) }
}

/// The three bones of a chain, by index, and the bone its root hangs off.
#[derive(Clone, Copy, Debug)]
struct Chain {
	/// What the root hangs off, if anything.
	above: Option<usize>,

	/// The bone that swings.
	root: usize,

	/// The bone that bends.
	middle: usize,

	/// The bone that lands.
	end: usize,
}

impl Chain {
	/// A bone and the two above it, if it has two above it and the pose holds
	/// all three.
	fn of(bones: &[Bone], locals: &[Transform], end: u16) -> Option<Self> {
		let end = usize::from(end);
		let middle = parent_of(bones, end)?;
		let root = parent_of(bones, middle)?;
		let chain = Self {
			above: parent_of(bones, root),
			root,
			middle,
			end,
		};

		[root, middle, end]
			.iter()
			.all(|&bone| bone < locals.len())
			.then_some(chain)
	}
}

/// Which bone one hangs off, if it is a bone and hangs off one.
fn parent_of(bones: &[Bone], bone: usize) -> Option<usize> {
	let parent = bones.get(bone)?.parent;

	(parent != NO_PARENT).then_some(usize::from(parent))
}

/// Where each bone of a pose stands in the pose's own space, as far as one of
/// them.
///
/// The walk [`Poses::model`](super::pose::Poses::model) makes, over a pose's
/// present alone, and stopping at the end of the chain: nothing past it can
/// move the three bones that are asked about.
fn lay(bones: &[Bone], locals: &[Transform], last: usize, out: &mut Vec<Mat4>) {
	out.clear();

	for (index, bone) in bones
		.iter()
		.enumerate()
		.take(last.saturating_add(1))
	{
		let local = locals.get(index).copied().unwrap_or(bone.rest);
		let above = parent_of(bones, index)
			.and_then(|parent| out.get(parent))
			.copied()
			.unwrap_or(Mat4::IDENTITY);

		out.push(above * local.matrix());
	}
}

/// Where a chain's middle and end go, and how far short of its target it
/// stops.
#[derive(Clone, Copy, Debug)]
struct Bent {
	/// Where the middle goes.
	middle: Vec3,

	/// Where the end goes.
	end: Vec3,

	/// How far the end stops from the target, nought when it gets there.
	left: f32,
}

/// Works out where a chain's middle and end go.
///
/// The corner of a triangle whose sides are the two bones and the distance
/// from the root to the target, that distance held to what the bones can span:
/// no further than the two laid end to end, no nearer than the one folded back
/// over the other.
///
/// @param was - where the root, the middle and the end stand now
/// @param target - where the end is to go
/// @param pole - a point on the side to bend to, if one is given
/// @return where they go, or `None` for a bone of no length
fn bend(was: [Vec3; 3], target: Vec3, pole: Option<Vec3>) -> Option<Bent> {
	let [root, middle, end] = was;
	let upper = root.distance(middle);
	let lower = middle.distance(end);
	let usable = |length: f32| length.is_finite() && length > SHORTEST;

	if !usable(upper) || !usable(lower) {
		return None;
	}

	let whole = upper + lower;
	let spread = (upper - lower).abs();
	let toward = target - root;
	let distance = toward.length();
	// a target on the root itself points nowhere, so the chain keeps the line
	// its end is already on.
	let along = toward
		.try_normalize()
		.or_else(|| (end - root).try_normalize())
		.unwrap_or(Vec3::Y);
	let span = distance.clamp(spread.max(whole * NEAREST), whole);
	let side = pole
		.and_then(|pole| aside(pole - root, along))
		.or_else(|| aside(middle - root, along))
		.unwrap_or_else(|| along.any_orthonormal_vector());
	// how far along the line the middle's foot stands, and how far off the line
	// the middle is. The height is the product of the triangle's four sums
	// rather than the root of a difference of squares, so that it is nought
	// exactly where the chain lies straight or folded instead of the root of
	// whatever the difference rounds to.
	let foot = 0.5 * ((upper - lower) * (upper + lower) / span + span);
	let height = ((whole + span) * (span - spread) * (span + spread) * (whole - span))
		.max(0.0)
		.sqrt()
		/ (2.0 * span);

	Some(Bent {
		middle: root + along * foot + side * height,
		end: root + along * span,
		left: (distance - span).abs(),
	})
}

/// Which way off the line through the root an offset from the root lies, if
/// it lies far enough off it to say.
fn aside(offset: Vec3, along: Vec3) -> Option<Vec3> {
	let off = offset.reject_from_normalized(along);

	(off.length() > offset.length() * ASIDE).then(|| off.normalize())
}

/// The three bones' new turns, each against its own parent.
///
/// The root's and the middle's are the least turns that carry the chain from
/// where it stood to where it was bent to, the swing first and the fold after
/// it; the end's is whatever faces it the way it faced before, against the
/// middle it now hangs off.
fn turned(
	above: Quat,
	locals: &[Transform],
	chain: Chain,
	was: [Vec3; 3],
	bent: &Bent,
) -> Option<[Quat; 3]> {
	let [root, middle, end] = was;
	let facing = |bone: usize| locals.get(bone).map(|local| local.rotation);
	let root_faced = above * facing(chain.root)?;
	let middle_faced = root_faced * facing(chain.middle)?;
	let end_faced = middle_faced * facing(chain.end)?;
	let swing = arc(middle - root, bent.middle - root);
	let fold = arc(swing * (end - middle), bent.end - bent.middle);
	let root_now = swing * root_faced;
	let middle_now = fold * (swing * middle_faced);

	Some([
		(above.inverse() * root_now).normalize(),
		(root_now.inverse() * middle_now).normalize(),
		(middle_now.inverse() * end_faced).normalize(),
	])
}

/// The least turn that carries one direction onto another.
///
/// Worked out here rather than asked of the math library, whose own version
/// answers "not at all" for every turn under about seven ten-thousandths of a
/// radian, which on a leg is a foot left most of a millimeter short of a step
/// it was told to stand on. The quaternion made of the cross and one plus the
/// dot is exact for every angle but a half turn; there the axis is anyone's,
/// and a fixed one is taken.
fn arc(from: Vec3, to: Vec3) -> Quat {
	let (from, to) = (from.normalize_or(Vec3::Y), to.normalize_or(Vec3::Y));
	let axis = from.cross(to);
	let half = Quat::from_xyzw(axis.x, axis.y, axis.z, 1.0 + from.dot(to));

	if half.length_squared() > f32::EPSILON * f32::EPSILON {
		half.normalize()
	} else {
		Quat::from_axis_angle(from.any_orthonormal_vector(), core::f32::consts::PI)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		abi::{World, pose::Pose, skeleton::SkeletonData},
		glam::EulerRot,
	};

	// the bones of `leg`, by index.
	const HIP: u16 = 0;
	const THIGH: u16 = 1;
	const CALF: u16 = 2;
	const FOOT: u16 = 3;
	const TOE: u16 = 4;

	/// Directions to reach in, none of them along an axis or along the leg.
	const WAYS: [Vec3; 6] = [
		Vec3::new(0.3, -1.0, 0.2),
		Vec3::new(-0.5, -0.8, 0.4),
		Vec3::new(0.7, -0.2, -0.6),
		Vec3::new(0.1, -0.9, -0.3),
		Vec3::new(-0.2, 0.4, 0.9),
		Vec3::new(0.9, 0.1, 0.1),
	];

	/// A leg hanging off a hip that stands turned and tilted, a toe past its
	/// ankle.
	///
	/// Nothing about it is regular, and on purpose. The thigh is longer than
	/// the calf, so a mistake that swaps the two shows; the rest is already
	/// bent, in no plane the axes make, so a bend worked out against the wrong
	/// plane shows; and the hip is turned three ways at once, so a turn written
	/// in the wrong space shows.
	fn leg() -> Vec<Bone> {
		let bone = |name: &str, parent: u16, position: Vec3, rotation: Quat| Bone {
			name: name.to_owned(),
			parent,
			rest: Transform { position, rotation, scale: Vec3::ONE },
			..Bone::default()
		};

		vec![
			bone(
				"hip",
				NO_PARENT,
				Vec3::new(0.3, 1.1, -0.2),
				Quat::from_euler(EulerRot::YXZ, 0.5, 0.2, -0.1),
			),
			bone("thigh", HIP, Vec3::new(0.12, -0.05, 0.02), Quat::from_rotation_z(0.15)),
			bone("calf", THIGH, Vec3::new(0.02, -0.5, 0.03), Quat::from_rotation_x(-0.3)),
			bone("foot", CALF, Vec3::new(-0.01, -0.35, 0.0), Quat::from_rotation_x(0.25)),
			bone("toe", FOOT, Vec3::new(0.0, -0.04, 0.12), Quat::from_rotation_y(0.4)),
		]
	}

	/// Every bone where the skeleton rests it.
	fn rested(bones: &[Bone]) -> Vec<Transform> { bones.iter().map(|bone| bone.rest).collect() }

	/// Where every bone is, walked down from the top the plain way.
	///
	/// Written out here rather than borrowed from the module, so that a mistake
	/// in the walk the solver makes is not also the measure it is held to.
	fn places(bones: &[Bone], locals: &[Transform]) -> Vec<Mat4> {
		let mut models: Vec<Mat4> = Vec::with_capacity(bones.len());

		for (bone, local) in bones.iter().zip(locals) {
			let above = if bone.parent == NO_PARENT {
				Mat4::IDENTITY
			} else {
				models[usize::from(bone.parent)]
			};

			models.push(above * local.matrix());
		}

		models
	}

	/// One bone's place out of [`places`].
	fn spot(models: &[Mat4], bone: u16) -> Vec3 { models[usize::from(bone)].w_axis.truncate() }

	/// Where the chain of [`leg`] starts, and how long its two bones are.
	fn measured(bones: &[Bone]) -> (Vec3, f32, f32) {
		let models = places(bones, &rested(bones));
		let (hip, knee, ankle) = (spot(&models, THIGH), spot(&models, CALF), spot(&models, FOOT));

		(hip, hip.distance(knee), knee.distance(ankle))
	}

	/// Bends a copy of a pose, and hands back the copy and what was said.
	fn solve(
		bones: &[Bone],
		locals: &[Transform],
		wanted: &Reach,
	) -> (Vec<Transform>, Option<f32>) {
		let mut out = locals.to_vec();
		let left = reach(bones, &mut out, wanted, &mut Vec::new());

		(out, left)
	}

	#[test]
	fn a_target_within_reach_is_where_the_end_lands() {
		let bones = leg();
		let (hip, upper, lower) = measured(&bones);
		let spread = (upper - lower).abs();

		for way in WAYS {
			// between the chain folded and the chain straight, and at neither.
			for share in [0.02_f32, 0.3, 0.55, 0.8, 0.98] {
				let target =
					hip + way.normalize() * share.mul_add(2.0 * upper.min(lower), spread);
				let (out, left) = solve(&bones, &rested(&bones), &Reach::new(FOOT, target));
				let landed = spot(&places(&bones, &out), FOOT);

				assert_eq!(left, Some(0.0), "a point the chain can span leaves nothing to spare");
				assert!(
					landed.distance(target) < 2.0e-6,
					"the foot landed {} from {target}",
					landed.distance(target)
				);
			}
		}
	}

	#[test]
	fn a_reach_of_a_tenth_of_a_millimeter_is_reached_as_closely_as_a_long_one() {
		// the turns this asks for are well under a thousandth of a radian, which
		// is where a least turn built to give up near nothing turns nothing.
		let bones = leg();
		let rest = rested(&bones);
		let target = spot(&places(&bones, &rest), FOOT) + Vec3::new(1.0e-4, -6.0e-5, 3.0e-5);
		let (out, _) = solve(&bones, &rest, &Reach::new(FOOT, target));
		let landed = spot(&places(&bones, &out), FOOT);

		assert!(
			landed.distance(target) < 2.0e-6,
			"the foot landed {} from a point a tenth of a millimeter off",
			landed.distance(target)
		);
	}

	#[test]
	fn every_bone_keeps_its_length_and_nothing_but_three_turns_is_written() {
		let bones = leg();
		let rest = rested(&bones);
		let (hip, upper, lower) = measured(&bones);
		let far = hip + Vec3::new(0.4, -0.9, 0.3).normalize() * 2.0;
		let near = hip + Vec3::new(-0.2, -0.3, 0.5).normalize() * 0.05;
		let within = hip + Vec3::new(0.5, -0.6, -0.2).normalize() * 0.6;

		for target in [far, near, within] {
			let (out, _) = solve(&bones, &rest, &Reach::new(FOOT, target).bending(hip + Vec3::Z));
			let models = places(&bones, &out);
			let knee = spot(&models, CALF);

			assert!(
				(spot(&models, THIGH).distance(knee) - upper).abs() < 1.0e-6,
				"reaching for {target} the thigh stays {upper} long"
			);
			assert!(
				(knee.distance(spot(&models, FOOT)) - lower).abs() < 1.0e-6,
				"and the calf {lower}"
			);

			for (index, (was, now)) in rest.iter().zip(&out).enumerate() {
				assert_eq!(
					(was.position, was.scale),
					(now.position, now.scale),
					"bone {index} neither moved inside its parent nor grew"
				);
			}

			assert_eq!(
				out[usize::from(HIP)],
				rest[usize::from(HIP)],
				"the hip is not the chain's"
			);
			assert_eq!(out[usize::from(TOE)], rest[usize::from(TOE)], "and nor is the toe");
		}
	}

	#[test]
	fn the_middle_bends_in_the_plane_of_the_pole_and_on_its_side() {
		let bones = leg();
		let rest = rested(&bones);
		let (hip, ..) = measured(&bones);
		let target = hip + Vec3::new(0.2, -0.55, 0.15);
		let line = (target - hip).normalize();
		let pole = hip + Vec3::new(0.3, 0.1, 0.6);
		let (out, _) = solve(&bones, &rest, &Reach::new(FOOT, target).bending(pole));
		let knee = spot(&places(&bones, &out), CALF) - hip;
		let normal = line.cross(pole - hip).normalize();
		let aside = (pole - hip).reject_from_normalized(line);

		assert!(
			knee.dot(normal).abs() < 1.0e-6,
			"the knee is {} out of the plane of the hip, the target and the pole",
			knee.dot(normal)
		);
		assert!(
			knee.reject_from_normalized(line).dot(aside) > 0.0,
			"and on the pole's side of the line"
		);

		// the pole mirrored through the line puts the knee in the mirror.
		let mirrored = hip + (pole - hip).project_onto_normalized(line) - aside;
		let (other, _) = solve(&bones, &rest, &Reach::new(FOOT, target).bending(mirrored));
		let flipped = spot(&places(&bones, &other), CALF) - hip;
		let expected = knee.project_onto_normalized(line) - knee.reject_from_normalized(line);

		assert!(
			flipped.distance(expected) < 2.0e-6,
			"the other side puts the knee at {flipped} where the mirror of {knee} is {expected}"
		);
	}

	#[test]
	fn with_a_pole_the_answer_is_the_same_wherever_the_chain_started() {
		let bones = leg();
		let rest = rested(&bones);
		let mut elsewhere = rest.clone();

		elsewhere[usize::from(THIGH)].rotation = Quat::from_euler(EulerRot::XYZ, 0.7, -0.3, 0.2);
		elsewhere[usize::from(CALF)].rotation = Quat::from_rotation_x(-1.1);

		let (hip, ..) = measured(&bones);
		let wanted = Reach::new(FOOT, hip + Vec3::new(0.3, -0.45, -0.1))
			.bending(hip + Vec3::new(0.1, 0.2, 0.9));
		let one = places(&bones, &solve(&bones, &rest, &wanted).0);
		let two = places(&bones, &solve(&bones, &elsewhere, &wanted).0);

		for bone in [CALF, FOOT] {
			assert!(
				spot(&one, bone).distance(spot(&two, bone)) < 2.0e-6,
				"bone {bone} ended up in one place from both starts"
			);
		}

		assert!(
			spot(&places(&bones, &rest), CALF).distance(spot(&places(&bones, &elsewhere), CALF))
				> 0.1,
			"though the two knees started a long way apart"
		);
	}

	#[test]
	fn a_target_out_of_reach_straightens_the_chain_along_the_line_to_it() {
		let bones = leg();
		let (hip, upper, lower) = measured(&bones);
		let line = Vec3::new(0.4, -0.9, 0.3).normalize();
		let wanted = Reach::new(FOOT, hip + line * 2.0).bending(hip + Vec3::Z);
		let (out, left) = solve(&bones, &rested(&bones), &wanted);
		let models = places(&bones, &out);

		assert!(
			spot(&models, CALF).distance(hip + line * upper) < 2.0e-6,
			"the knee lies on the line, a thigh out from the hip"
		);
		assert!(
			spot(&models, FOOT).distance(hip + line * (upper + lower)) < 2.0e-6,
			"and the foot as far along it as the leg is long"
		);
		assert!(
			left.is_some_and(|left| (left - (2.0 - upper - lower)).abs() < 1.0e-6),
			"which is {left:?} short, and it says so"
		);
	}

	#[test]
	fn a_target_nearer_than_the_chain_can_fold_folds_it_as_far_as_it_goes() {
		let bones = leg();
		let (hip, upper, lower) = measured(&bones);
		let line = Vec3::new(-0.2, -0.3, 0.5).normalize();
		let (out, left) = solve(&bones, &rested(&bones), &Reach::new(FOOT, hip + line * 0.05));
		let models = places(&bones, &out);

		// the thigh is the longer, so the calf folds back along it and the foot
		// stops the difference out from the hip.
		assert!(
			spot(&models, CALF).distance(hip + line * upper) < 2.0e-6,
			"the knee is a thigh out along the line"
		);
		assert!(
			spot(&models, FOOT).distance(hip + line * (upper - lower)) < 2.0e-6,
			"and the foot folded back to the difference of the two"
		);
		assert!(
			left.is_some_and(|left| (left - (upper - lower - 0.05)).abs() < 1.0e-6),
			"which is {left:?} further out than the point"
		);
	}

	#[test]
	fn nought_weight_leaves_the_pose_exactly_as_it_was() {
		// every direction the reachable test uses, at distances inside the chain
		// and past it, because a slerp at nought hands its start back to the bit
		// nearly always, and only the test in front of it makes that always.
		let bones = leg();
		let rest = rested(&bones);
		let (hip, upper, lower) = measured(&bones);

		for way in WAYS {
			for share in [0.02_f32, 0.3, 0.55, 0.8, 0.98, 1.6] {
				let target = hip + way.normalize() * (share * (upper + lower));
				let (out, _) = solve(&bones, &rest, &Reach::new(FOOT, target).weighted(0.0));

				assert_eq!(out, rest, "a weight of nought reaching for {target} writes nothing");
			}
		}

		let far = hip + Vec3::new(0.4, -0.9, 0.3).normalize() * 2.0;

		for weight in [0.0, -1.0, f32::NAN] {
			let (out, left) = solve(&bones, &rest, &Reach::new(FOOT, far).weighted(weight));

			assert_eq!(out, rest, "a weight of {weight} writes nothing at all");
			assert!(
				left.is_some_and(|left| (left - (2.0 - upper - lower)).abs() < 1.0e-6),
				"and still says how far short the answer stops: {left:?}"
			);
		}
	}

	#[test]
	fn a_weight_between_turns_each_bone_that_share_of_the_way() {
		let bones = leg();
		let rest = rested(&bones);
		let (hip, ..) = measured(&bones);
		let wanted = Reach::new(FOOT, hip + Vec3::new(0.25, -0.5, 0.2)).bending(hip + Vec3::X);
		let (whole, _) = solve(&bones, &rest, &wanted);
		let (part, _) = solve(&bones, &rest, &wanted.weighted(0.3));
		let (past, _) = solve(&bones, &rest, &wanted.weighted(2.5));

		assert_eq!(past, whole, "a weight past one is the whole weight and no further");

		for bone in [THIGH, CALF, FOOT].map(usize::from) {
			assert_eq!(
				part[bone].rotation,
				rest[bone]
					.rotation
					.slerp(whole[bone].rotation, 0.3),
				"bone {bone} turned three tenths of the way"
			);
			assert_ne!(part[bone].rotation, whole[bone].rotation, "and not the whole way");
		}

		for bone in [HIP, TOE].map(usize::from) {
			assert_eq!(part[bone], rest[bone], "bone {bone} is not the chain's to turn");
		}
	}

	#[test]
	fn the_end_keeps_the_attitude_the_pose_gave_it_and_carries_what_hangs_off_it() {
		let bones = leg();
		let rest = rested(&bones);
		let before = places(&bones, &rest);
		let (hip, ..) = measured(&bones);
		let (out, _) =
			solve(&bones, &rest, &Reach::new(FOOT, hip + Vec3::new(-0.15, -0.45, 0.3)));
		let after = places(&bones, &out);
		let foot = usize::from(FOOT);
		let axes = |model: &Mat4| [model.x_axis, model.y_axis, model.z_axis];

		for (axis, (was, now)) in axes(&before[foot])
			.into_iter()
			.zip(axes(&after[foot]))
			.enumerate()
		{
			assert!(
				was.abs_diff_eq(now, 2.0e-6),
				"the foot's axis {axis} turned from {was} to {now}"
			);
		}

		let toe = |models: &[Mat4]| spot(models, TOE) - spot(models, FOOT);

		assert!(
			toe(&before).abs_diff_eq(toe(&after), 2.0e-6),
			"the toe is where it was against the foot"
		);
		assert!(
			spot(&after, FOOT).distance(spot(&before, FOOT)) > 0.05,
			"though the foot itself went somewhere else"
		);
	}

	#[test]
	fn with_no_pole_the_chain_bends_the_way_the_pose_already_bends() {
		let bones = leg();
		let rest = rested(&bones);
		let models = places(&bones, &rest);
		let (hip, knee_was) = (spot(&models, THIGH), spot(&models, CALF));
		let target = spot(&models, FOOT) + Vec3::new(0.06, 0.12, -0.04);
		let (out, _) = solve(&bones, &rest, &Reach::new(FOOT, target));
		let knee = spot(&places(&bones, &out), CALF) - hip;
		let line = (target - hip).normalize();
		let normal = line.cross(knee_was - hip).normalize();

		assert!(
			knee.dot(normal).abs() < 2.0e-6,
			"the knee stays in the plane the pose bent it in, and is {} out of it",
			knee.dot(normal)
		);
		assert!(
			knee.reject_from_normalized(line)
				.dot((knee_was - hip).reject_from_normalized(line))
				> 0.0,
			"and on the same side of it"
		);
	}

	#[test]
	fn a_pole_that_names_no_side_is_no_pole_at_all() {
		let bones = leg();
		let rest = rested(&bones);
		let (hip, ..) = measured(&bones);
		let target = hip + Vec3::new(0.2, -0.5, 0.1);
		let (unpoled, _) = solve(&bones, &rest, &Reach::new(FOOT, target));

		for pole in [hip + (target - hip) * 3.0, hip, Vec3::NAN] {
			let (out, _) = solve(&bones, &rest, &Reach::new(FOOT, target).bending(pole));

			assert_eq!(out, unpoled, "a pole at {pole} bends the chain the pose's own way");
		}
	}

	#[test]
	fn a_straight_chain_with_no_pole_bends_somewhere_and_the_same_way_every_time() {
		let straight = upright();
		let rest = rested(&straight);
		// straight under the hip and nearer than the leg is long, so neither the
		// pose nor a pole says which way the knee goes.
		let target = Vec3::new(0.0, 0.4, 0.0);
		let (once, _) = solve(&straight, &rest, &Reach::new(FOOT, target));
		let (again, _) = solve(&straight, &rest, &Reach::new(FOOT, target));
		let models = places(&straight, &once);
		let knee = spot(&models, CALF);

		assert_eq!(once, again, "the same pose and the same point bend the same way");
		assert!(spot(&models, FOOT).distance(target) < 2.0e-6, "the foot got there");
		assert!(knee.x.hypot(knee.z) > 0.1, "by a knee that went off the line, to {knee}");
	}

	#[test]
	fn a_chain_that_is_not_one_reaches_nothing_and_writes_nothing() {
		let bones = leg();
		let rest = rested(&bones);
		let (hip, ..) = measured(&bones);
		let target = hip + Vec3::new(0.2, -0.5, 0.1);

		for end in [HIP, THIGH, 99] {
			let (out, left) = solve(&bones, &rest, &Reach::new(end, target));

			assert_eq!((left, &out), (None, &rest), "bone {end} has not two bones above it");
		}

		for nowhere in [Vec3::NAN, Vec3::INFINITY] {
			let (out, left) = solve(&bones, &rest, &Reach::new(FOOT, nowhere));

			assert_eq!((left, &out), (None, &rest), "{nowhere} is not a place to reach for");
		}

		let mut narrow = rest[..usize::from(FOOT)].to_vec();

		assert_eq!(
			reach(&bones, &mut narrow, &Reach::new(FOOT, target), &mut Vec::new()),
			None,
			"a pose that stops short of the foot has no foot to bend"
		);
		assert_eq!(
			reach(&bones, &mut narrow, &Reach::new(FOOT, target).weighted(0.0), &mut Vec::new()),
			None,
			"at any weight, since it is not a chain before it is a question of how much"
		);

		let mut stub = leg();

		stub[usize::from(CALF)].rest.position = Vec3::ZERO;

		let stub_rest = rested(&stub);

		assert_eq!(
			solve(&stub, &stub_rest, &Reach::new(FOOT, target)),
			(stub_rest.clone(), None),
			"a thigh of no length is not half a chain"
		);
	}

	#[test]
	fn a_world_takes_the_target_where_the_character_stands() {
		let bones = leg();
		let mut world = World::new();
		let rig = world
			.skeletons
			.insert("leg", SkeletonData { bones: bones.clone() });
		let pose = world
			.poses
			.spawn(Pose::resting(rig, world.skeletons.bones(rig)));
		let at = Transform {
			position: Vec3::new(4.0, -1.0, 2.5),
			rotation: Quat::from_rotation_y(2.1) * Quat::from_rotation_x(0.2),
			scale: Vec3::splat(1.5),
		};
		let (hip, upper, lower) = measured(&bones);
		let line = Vec3::new(0.3, -0.8, 0.4).normalize();
		let target = at.matrix().transform_point3(hip + line * 0.6);

		assert_eq!(
			world.reach(pose, at, &Reach::new(FOOT, target)),
			Some(0.0),
			"a point in the world the leg can span, carried in by where it stands"
		);

		let posed = world.poses.get(pose).expect("the pose is there");
		let landed = at
			.matrix()
			.transform_point3(spot(&places(&bones, &posed.locals), FOOT));

		assert!(
			landed.distance(target) < 1.0e-5,
			"the foot landed {} from it",
			landed.distance(target)
		);

		// out of reach, the shortfall is in the pose's own units: the ones a
		// pelvis is lowered in, not the world's, which here are half again as
		// large.
		let far = at.matrix().transform_point3(hip + line * 2.0);
		let left = world.reach(pose, at, &Reach::new(FOOT, far));

		assert!(
			left.is_some_and(|left| (left - (2.0 - upper - lower)).abs() < 1.0e-5),
			"{left:?} short, in the pose's units"
		);
		assert!(world.poses.despawn(pose), "the pose was there");
		assert_eq!(
			world.reach(pose, at, &Reach::new(FOOT, target)),
			None,
			"and a pose nobody holds any more is bent by nobody"
		);
	}

	/// A leg standing dead straight under its hip, every bone along `y`.
	fn upright() -> Vec<Bone> {
		let bone = |name: &str, parent: u16, along: f32| Bone {
			name: name.to_owned(),
			parent,
			rest: Transform::at(Vec3::new(0.0, along, 0.0)),
			..Bone::default()
		};

		vec![
			bone("hip", NO_PARENT, 1.0),
			bone("thigh", HIP, 0.0),
			bone("calf", THIGH, -0.5),
			bone("foot", CALF, -0.35),
		]
	}

	/// The three turns a whole weight writes, worked out through the module's
	/// own steps one at a time.
	fn by_hand(bones: &[Bone], rest: &[Transform], wanted: &Reach) -> [Quat; 3] {
		let chain = Chain::of(bones, rest, wanted.end).expect("the end has two bones above it");
		let mut scratch = Vec::new();

		lay(bones, rest, chain.end, &mut scratch);

		let was =
			[chain.root, chain.middle, chain.end].map(|bone| scratch[bone].w_axis.truncate());
		let bent = bend(was, wanted.target, wanted.pole).expect("the chain bends");
		let above = chain
			.above
			.map_or(Quat::IDENTITY, |bone| scratch[bone].to_scale_rotation_translation().1);

		turned(above, rest, chain, was, &bent).expect("and turns")
	}

	#[test]
	fn the_whole_weight_writes_the_answer_itself_rather_than_a_slerp_of_it() {
		// held to the bit rather than to a rounding. The half turn is the case a
		// slerp to the answer gets visibly wrong: the thigh's new turn lies in the
		// other hemisphere from its old one, and a slerp hands back the same turn
		// with every sign turned over.
		let (bones, straight) = (leg(), upright());
		let (hip, ..) = measured(&bones);
		let cases = [
			(bones, Reach::new(FOOT, hip + Vec3::new(0.25, -0.5, 0.2)).bending(hip + Vec3::X)),
			(straight, Reach::new(FOOT, Vec3::new(0.0, 3.0, 0.0))),
		];

		for (bones, wanted) in cases {
			let rest = rested(&bones);
			let (out, _) = solve(&bones, &rest, &wanted);

			for (bone, turn) in [THIGH, CALF, FOOT]
				.map(usize::from)
				.into_iter()
				.zip(by_hand(&bones, &rest, &wanted))
			{
				assert_eq!(out[bone].rotation, turn, "bone {bone} holds the answer to the bit");
			}
		}
	}

	#[test]
	fn a_calf_longer_than_its_thigh_folds_the_knee_back_past_the_hip() {
		let mut bones = leg();

		bones[usize::from(CALF)].rest.position = Vec3::new(0.02, -0.3, 0.03);
		bones[usize::from(FOOT)].rest.position = Vec3::new(-0.01, -0.5, 0.0);

		let (hip, upper, lower) = measured(&bones);
		let line = Vec3::new(-0.2, -0.3, 0.5).normalize();
		let (out, left) = solve(&bones, &rested(&bones), &Reach::new(FOOT, hip + line * 0.05));
		let models = places(&bones, &out);

		// folded as far as it goes, the calf reaches back through the hip and the
		// foot stops the difference of the two out in front of it.
		assert!(lower > upper, "the calf is the longer, {lower} against {upper}");
		assert!(
			spot(&models, CALF).distance(hip - line * upper) < 2.0e-6,
			"the knee is a thigh out behind the hip"
		);
		assert!(
			spot(&models, FOOT).distance(hip + line * (lower - upper)) < 2.0e-6,
			"and the foot the difference out in front of it"
		);
		assert!(
			left.is_some_and(|left| (left - (lower - upper - 0.05)).abs() < 1.0e-6),
			"which is {left:?} further out than the point"
		);
	}

	#[test]
	fn two_equal_bones_reaching_for_their_own_root_fold_without_a_number_that_is_not_one() {
		let mut bones = leg();

		bones[usize::from(FOOT)].rest.position = bones[usize::from(CALF)].rest.position;

		let (hip, upper, lower) = measured(&bones);
		let (out, left) = solve(&bones, &rested(&bones), &Reach::new(FOOT, hip));
		let models = places(&bones, &out);

		assert!(
			(upper - lower).abs() < 1.0e-6,
			"the two bones are one length, {upper} and {lower}"
		);
		assert!(
			out.iter().all(|local| local.rotation.is_finite()),
			"every turn written is a number"
		);
		assert!(
			spot(&models, FOOT).distance(hip) <= (upper + lower) * NEAREST * 1.01,
			"and the foot folds back to within a hair of the hip, to {}",
			spot(&models, FOOT).distance(hip)
		);
		assert!(
			left.is_some_and(|left| (upper + lower).mul_add(-NEAREST, left).abs() < 1.0e-6),
			"which is {left:?} from it, and it says so"
		);
	}

	#[test]
	fn a_chain_sent_straight_back_the_way_it_hangs_turns_half_round_and_gets_there() {
		// the thigh has to swing exactly half a turn, the one angle a least turn
		// cannot be built from its cross and its dot.
		let straight = upright();
		let (out, left) =
			solve(&straight, &rested(&straight), &Reach::new(FOOT, Vec3::new(0.0, 3.0, 0.0)));
		let foot = spot(&places(&straight, &out), FOOT);

		assert!(
			out.iter().all(|local| local.rotation.is_finite()),
			"every turn written is a number"
		);
		assert!(
			foot.distance(Vec3::new(0.0, 1.85, 0.0)) < 2.0e-6,
			"the leg stands straight up from the hip, with its foot at {foot}"
		);
		assert!(
			left.is_some_and(|left| (left - 1.15).abs() < 1.0e-6),
			"{left:?} short of a point two above the hip"
		);
	}

	#[test]
	fn a_world_takes_the_pole_where_the_character_stands_as_well() {
		let bones = leg();
		let mut world = World::new();
		let rig = world
			.skeletons
			.insert("leg", SkeletonData { bones: bones.clone() });
		let pose = world
			.poses
			.spawn(Pose::resting(rig, world.skeletons.bones(rig)));
		let at = Transform {
			position: Vec3::new(-3.0, 2.0, 1.0),
			rotation: Quat::from_rotation_y(-1.3) * Quat::from_rotation_z(0.25),
			scale: Vec3::splat(0.8),
		};
		let (hip, ..) = measured(&bones);
		let out = at.matrix();
		let target = out.transform_point3(hip + Vec3::new(0.2, -0.55, 0.15));
		let pole = out.transform_point3(hip + Vec3::new(0.3, 0.1, 0.6));

		assert_eq!(
			world.reach(pose, at, &Reach::new(FOOT, target).bending(pole)),
			Some(0.0),
			"a point the leg can span, bent towards another, both in the world"
		);

		let models = places(
			&bones,
			&world
				.poses
				.get(pose)
				.expect("the pose is there")
				.locals,
		);
		let root = out.transform_point3(spot(&models, THIGH));
		let knee = out.transform_point3(spot(&models, CALF)) - root;
		let line = (target - root).normalize();
		let normal = line.cross(pole - root).normalize();

		assert!(
			knee.dot(normal).abs() < 1.0e-5,
			"the knee is {} out of the plane the world's three points make",
			knee.dot(normal)
		);
		assert!(
			knee.reject_from_normalized(line)
				.dot((pole - root).reject_from_normalized(line))
				> 0.0,
			"and on the side of it the pole is"
		);
	}
}
