//! Ragdolls: which bones get a body, what shape it is, and where the joints
//! between them sit.
//!
//! A ragdoll is a handful of bodies and the joints holding them, laid over a
//! skeleton so that the bodies can be made to carry the bones instead of the
//! other way round. Nothing here creates one: this module works out the
//! *layout*, and spawning is a loop the game writes, exactly as it is for a
//! model's placements. What the layout is worked out from is the skeleton at
//! rest and a list of segments a game names, and both halves of that are worth
//! the paragraph they cost.
//!
//! **A game names its limbs, and it names them with text.** The obvious
//! alternative is a rule over the skeleton - a bone longer than some minimum
//! gets a body - and it is what one of the large engines offers, with a hand
//! edit afterwards. Measured against the rig in this tree it produces
//! nonsense: the root's first child sits on top of it, so the segment between
//! them is nothing; there is a utility bone pointing backwards out of the hips
//! that would become a limb; the thighs hang off a spine bone rather than off
//! the pelvis; and the toes and the nubs at the ends are shorter than any
//! threshold that leaves the hands alone. Naming thirteen segments is what the
//! other large engine's wizard asks for, it is thirteen lines in the game, and
//! it cannot be wrong by accident.
//!
//! **A part is a segment rather than a bone.** A bone is a point and a
//! rotation; what a body needs is a length, and the only length a bone has is
//! the distance to something further down. So a part names two bones - where
//! it starts and where it ends - and the second one need not be a part itself.
//! A foot ends at the toe and the toe carries nothing.
//!
//! **[`Part::in_bone`] is the whole bridge, and it works both ways.** It is
//! the transform from the bone's own space into the body's, so
//!
//! ```text
//!   body in the world = at * model(bone) * in_bone
//!   model(bone)       = at^-1 * body in the world * in_bone^-1
//! ```
//!
//! which is what lets an animation carry the bodies while the character is on
//! its feet, and the bodies carry the bones once it is not. The rotation half
//! of it is what makes the box a limb rather than a crate: the body's own `x`
//! runs down the segment, so its half-extents are a half-length and two
//! thicknesses.
//!
//! **A part's parent is the nearest bone above it that is also a part.** Not
//! its bone's parent: a head hangs off a chest through a neck nobody gave a
//! body to, and the bones in between simply ride along. Parts come out sorted
//! by bone, so a parent is always written before its children - the same rule
//! the skeleton's bones and the blend tree's nodes follow, bought the same way
//! and paying for the same single forward pass.

use super::{
	entity::Transform,
	physics::{Body, BodyId, Shape},
	skeleton::{Bone, NO_PARENT, rests},
};
use crate::glam::{Mat4, Quat, Vec3};

/// How many bodies one ragdoll may be made of.
///
/// Well above what anything real wants: the wizard everybody copies asks for
/// thirteen, and a film rig with fingers would be under forty.
pub const MAX_PARTS: usize = 64;

/// A part index that refers to nothing, which is what a root's parent is.
pub const NO_PART: u16 = u16::MAX;

/// The shortest segment that becomes a body.
///
/// Two bones in the same place are not a limb, and a box of no length has no
/// direction to be turned along either.
const SHORTEST: f32 = 1e-4;

/// One limb a game asks for, as it names it.
///
/// Text rather than indices, for the reason a clip's tracks are text: two
/// exports of one biped share their names and share none of their numbering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment<'a> {
	/// The bone the limb starts at, and the bone its body follows.
	pub from: &'a str,

	/// The bone it reaches to, which says how long it is.
	///
	/// Need not be a limb itself, and need not be a direct child: a pelvis
	/// reaching to the bone above the chest is a pelvis two joints long.
	pub to: &'a str,

	/// How thick it is, as a half-extent.
	///
	/// **Zero means work it out from the length**, at
	/// [`Build::girth`](Build::girth) of it. That sentinel is not there to save
	/// typing: a share of the length is right for arms and wrong for legs,
	/// because two thighs derived from a ratio are wider than the distance
	/// between the hips and spend the simulation shoving each other apart.
	pub girth: f32,
}

impl<'a> Segment<'a> {
	/// A limb between two bones, as thick as its length says.
	///
	/// @param from - the bone it starts at
	/// @param to - the bone it reaches
	#[must_use]
	pub const fn new(from: &'a str, to: &'a str) -> Self { Self { from, to, girth: 0.0 } }

	/// A limb between two bones, of a thickness the caller knows better.
	///
	/// @param from - the bone it starts at
	/// @param to - the bone it reaches
	/// @param girth - its half-extent across the segment
	#[must_use]
	pub const fn thick(from: &'a str, to: &'a str, girth: f32) -> Self {
		Self { from, to, girth }
	}
}

/// The numbers that are about the whole ragdoll rather than one limb of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Build {
	/// How heavy the whole thing is, shared out among the parts by volume.
	///
	/// Sharing rather than one mass each is what keeps a head from weighing as
	/// much as a thigh, and a total is the number a person actually knows.
	/// **Zero or less gives every part [`Body::MASS`]**, which is what a
	/// ragdoll nobody has weighed should be rather than nothing at all.
	pub mass: f32,

	/// How thick a limb is as a share of its own length, where the segment
	/// does not say. @ref [`Segment::girth`].
	pub girth: f32,
}

impl Build {
	/// A ragdoll of the weight and proportions of nobody in particular.
	pub const DEFAULT: Self = Self { mass: 0.0, girth: 0.3 };
}

impl Default for Build {
	fn default() -> Self { Self::DEFAULT }
}

/// One limb: a body's shape, where it sits on its bone, and what holds it up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Part {
	/// Which bone this follows, and which bone it writes when the bodies are
	/// driving.
	pub bone: u16,

	/// Which part it hangs off, as an index into [`Ragdoll::parts`], or
	/// [`NO_PART`].
	///
	/// Always less than this part's own index, so one forward pass resolves
	/// the lot.
	pub parent: u16,

	/// The body a game spawned for it, or [`BodyId::NONE`] before it has.
	///
	/// Written by whoever spawns, because a plan knows what a limb should be
	/// and nothing about what exists.
	pub body: BodyId,

	/// What that body is shaped like: a box, half a segment long on `x`.
	pub shape: Shape,

	/// The bone's own space into the body's. @ref the module docs.
	pub in_bone: Transform,

	/// Where the joint attaches on this body, in the body's own space.
	///
	/// The bone's origin, which is the far end of the parent's segment. It
	/// works out to exactly half a length back along `x`, and it is stored
	/// rather than recomputed because a game passes it to
	/// [`Joint::ball`](super::joint::Joint::ball) and should not have to know
	/// that.
	pub anchor: Vec3,

	/// Where the same joint attaches on the parent's body, in *its* own space.
	///
	/// Meaningless when [`parent`](Self::parent) is [`NO_PART`].
	pub parent_anchor: Vec3,

	/// How heavy this limb is.
	pub mass: f32,
}

impl Part {
	/// A part of nothing, which is what an unfilled slot holds.
	pub const NOTHING: Self = Self {
		bone: 0,
		parent: NO_PART,
		body: BodyId::NONE,
		shape: Shape::UNIT,
		in_bone: Transform::IDENTITY,
		anchor: Vec3::ZERO,
		parent_anchor: Vec3::ZERO,
		mass: Body::MASS,
	};

	/// How long the segment this stands for is.
	#[must_use]
	pub fn length(&self) -> f32 { self.shape.extents.x * 2.0 }
}

impl Default for Part {
	fn default() -> Self { Self::NOTHING }
}

/// A whole ragdoll's layout.
///
/// Plain data the game builds every step and throws away, exactly as it does a
/// blend tree, and for the same reason: what a game can keep between steps is
/// its arena, and an arena is bytes. What it keeps there is the handles.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ragdoll {
	/// Every limb, parents before children.
	pub parts: Vec<Part>,

	/// How many segments were asked for and could not be made.
	dropped: u16,
}

impl Ragdoll {
	/// How many segments were asked for and could not be made into a limb.
	///
	/// A name no bone answers to, a segment of no length, or one part past
	/// [`MAX_PARTS`]. Counted rather than refused, because a ragdoll missing an
	/// arm is still a ragdoll and a caller that cares can say so once.
	#[must_use]
	pub const fn dropped(&self) -> u16 { self.dropped }

	/// Whether there are no limbs at all.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.parts.is_empty() }

	/// How many limbs there are.
	#[must_use]
	pub fn len(&self) -> usize { self.parts.len() }

	/// Which part follows a bone, if any does.
	///
	/// A scan, like every other lookup by name or index in this ABI, and for
	/// the same reason: the lists are short and the alternative is a second
	/// array to keep in step.
	///
	/// @param bone - the bone's index in its skeleton
	/// @return the part's index in [`parts`](Self::parts)
	#[must_use]
	pub fn part_of(&self, bone: u16) -> Option<u16> {
		self.parts
			.iter()
			.position(|part| part.bone == bone)
			.and_then(|index| u16::try_from(index).ok())
	}
}

/// Works out a ragdoll's layout from a skeleton at rest.
///
/// Pure: it reads the bones and the segments and writes nothing. Every number
/// in the answer is in the *model's* own space, which is where a pose lives, so
/// putting one in the world is one more matrix and that matrix is the caller's.
///
/// @param bones - the skeleton, parents before children
/// @param segments - the limbs wanted, named
/// @param build - the whole-ragdoll numbers
/// @return the layout, with every part's body still [`BodyId::NONE`]
#[must_use]
pub fn plan(bones: &[Bone], segments: &[Segment<'_>], build: Build) -> Ragdoll {
	let mut at = Vec::new();
	rests(bones, &mut at);

	let mut ragdoll = Ragdoll::default();

	for segment in segments {
		if ragdoll.parts.len() == MAX_PARTS {
			ragdoll.dropped = ragdoll.dropped.saturating_add(1);

			continue;
		}

		match limb(&at, bones, segment, build) {
			| Some(part) => ragdoll.parts.push(part),
			| None => ragdoll.dropped = ragdoll.dropped.saturating_add(1),
		}
	}

	// by bone, so a parent part is written before its children: a part's bone
	// is above its children's bones and a skeleton is already sorted that way.
	ragdoll.parts.sort_by_key(|part| part.bone);
	hang(&mut ragdoll, &at, bones);
	weigh(&mut ragdoll, build.mass);

	ragdoll
}

/// One limb, or nothing if the segment does not describe one.
fn limb(at: &[Mat4], bones: &[Bone], segment: &Segment<'_>, build: Build) -> Option<Part> {
	let from = named(bones, segment.from)?;
	let to = named(bones, segment.to)?;
	let along = at
		.get(usize::from(from))?
		.inverse()
		.transform_point3(at.get(usize::from(to))?.w_axis.truncate());
	let length = along.length();

	if length < SHORTEST {
		return None;
	}

	let girth = if segment.girth > 0.0 {
		segment.girth
	} else {
		length * build.girth.max(0.0)
	}
	.max(SHORTEST);
	let half = length / 2.0;

	Some(Part {
		bone: from,
		shape: Shape::cuboid(Vec3::new(half, girth, girth)),
		in_bone: Transform {
			position: along / 2.0,
			rotation: Quat::from_rotation_arc(Vec3::X, along / length),
			scale: Vec3::ONE,
		},
		// the bone's own origin, seen from a body whose x runs down the
		// segment and whose middle is halfway along it.
		anchor: Vec3::new(-half, 0.0, 0.0),
		..Part::NOTHING
	})
}

/// Fills in what each part hangs off and where the joint meets its parent.
fn hang(ragdoll: &mut Ragdoll, at: &[Mat4], bones: &[Bone]) {
	for index in 0..ragdoll.parts.len() {
		let part = ragdoll.parts[index];
		let Some(parent) = above(ragdoll, bones, part.bone) else {
			continue;
		};
		let held = ragdoll.parts[usize::from(parent)];
		let (Some(mine), Some(theirs)) =
			(at.get(usize::from(part.bone)), at.get(usize::from(held.bone)))
		else {
			continue;
		};

		ragdoll.parts[index].parent = parent;
		ragdoll.parts[index].parent_anchor = (*theirs * held.in_bone.matrix())
			.inverse()
			.transform_point3(mine.w_axis.truncate());
	}
}

/// The nearest part above a bone, walking up the skeleton.
fn above(ragdoll: &Ragdoll, bones: &[Bone], bone: u16) -> Option<u16> {
	let mut walking = bones.get(usize::from(bone))?.parent;

	while walking != NO_PARENT {
		if let Some(part) = ragdoll.part_of(walking) {
			return Some(part);
		}

		walking = bones.get(usize::from(walking))?.parent;
	}

	None
}

/// Shares a total mass out among the parts by how big each one is.
fn weigh(ragdoll: &mut Ragdoll, total: f32) {
	if total <= 0.0 {
		return;
	}

	let volume = |part: &Part| {
		let it = part.shape.extents;

		8.0 * it.x * it.y * it.z
	};
	let whole: f32 = ragdoll.parts.iter().map(volume).sum();

	if whole <= 0.0 {
		return;
	}

	for part in &mut ragdoll.parts {
		part.mass = (total * volume(part) / whole).max(SHORTEST);
	}
}

/// A bone by name, refusing the empty one.
fn named(bones: &[Bone], name: &str) -> Option<u16> {
	if name.is_empty() {
		return None;
	}

	bones
		.iter()
		.position(|bone| bone.name == name)
		.and_then(|index| u16::try_from(index).ok())
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A skeleton with the two things a ragdoll's arithmetic has to survive on
	/// a real one.
	///
	/// A leg down `x` - a hip, a thigh a unit long, a shin two units long and a
	/// foot bone at the end - because offsets along one axis are checkable by
	/// eye, and it is what the rig in this tree does. Then a spine up `y` with
	/// a neck in the middle of it, because a head hangs off a chest through a
	/// bone nobody gives a body to and that walk is the part a leg cannot
	/// exercise. And one bone with no name at all, which an importer really
	/// does produce.
	fn leg() -> Vec<Bone> {
		let bone = |name: &str, parent: u16, at: Vec3| Bone {
			name: name.to_owned(),
			parent,
			inverse_bind: Mat4::IDENTITY,
			rest: Transform::at(at),
		};

		vec![
			bone("hip", NO_PARENT, Vec3::new(0.0, 4.0, 0.0)),
			bone("thigh", 0, Vec3::ZERO),
			bone("shin", 1, Vec3::new(1.0, 0.0, 0.0)),
			bone("foot", 2, Vec3::new(2.0, 0.0, 0.0)),
			bone("chest", 0, Vec3::new(0.0, 1.0, 0.0)),
			bone("neck", 4, Vec3::new(0.0, 1.0, 0.0)),
			bone("head", 5, Vec3::new(0.0, 0.5, 0.0)),
			bone("crown", 6, Vec3::new(0.0, 0.5, 0.0)),
			bone("", 0, Vec3::new(0.0, 0.0, 1.0)),
		]
	}

	/// The plan a `leg` gives for a thigh and a shin.
	fn planned() -> Ragdoll {
		plan(
			&leg(),
			&[Segment::new("thigh", "shin"), Segment::new("shin", "foot")],
			Build::DEFAULT,
		)
	}

	#[test]
	fn a_part_is_half_its_segment_long_and_sits_halfway_down_it() {
		let ragdoll = planned();
		let thigh = ragdoll.parts[0];

		assert_eq!(ragdoll.len(), 2, "one part a segment");
		assert!(
			(thigh.shape.extents.x - 0.5).abs() < 1e-5,
			"a segment of one unit is a box half a unit long, got {}",
			thigh.shape.extents.x
		);
		assert!(
			thigh
				.in_bone
				.position
				.abs_diff_eq(Vec3::new(0.5, 0.0, 0.0), 1e-5),
			"whose middle is halfway along it, got {}",
			thigh.in_bone.position
		);
		assert!((thigh.length() - 1.0).abs() < 1e-5, "and the length reads back off the shape");
	}

	#[test]
	fn a_parts_own_x_runs_down_its_segment() {
		// the whole reason `in_bone` carries a rotation: a box is axis-aligned
		// in its own space, so unless one of its axes is put along the segment
		// it is a crate around the limb rather than the limb.
		let bones = vec![
			Bone {
				name: "root".to_owned(),
				parent: NO_PARENT,
				inverse_bind: Mat4::IDENTITY,
				rest: Transform::IDENTITY,
			},
			Bone {
				name: "up".to_owned(),
				parent: 0,
				inverse_bind: Mat4::IDENTITY,
				rest: Transform::at(Vec3::new(0.0, 3.0, 0.0)),
			},
		];
		let ragdoll = plan(&bones, &[Segment::new("root", "up")], Build::DEFAULT);
		let part = ragdoll.parts[0];

		assert!(
			(part.in_bone.rotation * Vec3::X).abs_diff_eq(Vec3::Y, 1e-5),
			"the body's own x points the way the segment goes, got {}",
			part.in_bone.rotation * Vec3::X
		);
		assert!(
			(part.shape.extents.x - 1.5).abs() < 1e-5,
			"and the half-length is on x whichever way the segment points"
		);
	}

	#[test]
	fn the_two_anchors_of_a_joint_are_the_same_point_in_the_world() {
		// what this really checks is that a game handing both to a ball joint
		// is handing it a joint that is already satisfied, so nothing yanks on
		// the step a ragdoll is switched on.
		let mut at = Vec::new();
		rests(&leg(), &mut at);

		let ragdoll = planned();
		let shin = ragdoll.parts[1];
		let thigh = ragdoll.parts[usize::from(shin.parent)];

		let mine =
			(at[usize::from(shin.bone)] * shin.in_bone.matrix()).transform_point3(shin.anchor);
		let theirs = (at[usize::from(thigh.bone)] * thigh.in_bone.matrix())
			.transform_point3(shin.parent_anchor);

		assert!(
			mine.abs_diff_eq(theirs, 1e-5),
			"the two anchors meet, and they did not: {mine} against {theirs}"
		);
		assert!(
			mine.abs_diff_eq(at[usize::from(shin.bone)].w_axis.truncate(), 1e-5),
			"and where they meet is the bone the part starts at"
		);
	}

	#[test]
	fn a_part_hangs_off_the_nearest_part_above_it_rather_than_off_its_own_bone() {
		// the shin's bone hangs off the thigh's, which is a part; the thigh's
		// hangs off the hip, which is not, so the thigh is a root.
		let ragdoll = planned();

		assert_eq!(ragdoll.parts[0].parent, NO_PART, "nothing above the thigh is a part");
		assert_eq!(ragdoll.parts[1].parent, 0, "and the shin hangs off the thigh");
	}

	#[test]
	fn a_part_skips_the_bones_between_it_and_the_part_above() {
		let ragdoll = plan(
			&leg(),
			&[Segment::new("hip", "thigh"), Segment::new("shin", "foot")],
			Build::DEFAULT,
		);

		assert_eq!(ragdoll.len(), 1, "the hip is two bones in one place and is no limb");
		assert_eq!(ragdoll.dropped(), 1, "and it is counted rather than lost");
		assert_eq!(
			ragdoll.parts[0].parent, NO_PART,
			"so the shin has nothing above it, the thigh never having been asked for"
		);
	}

	#[test]
	fn parts_come_out_with_every_parent_before_its_children() {
		// asked for in the wrong order on purpose. The rule is what makes the
		// forward pass that drives a pose possible at all.
		let ragdoll = plan(
			&leg(),
			&[Segment::new("shin", "foot"), Segment::new("thigh", "shin")],
			Build::DEFAULT,
		);

		assert!(
			ragdoll
				.parts
				.iter()
				.enumerate()
				.all(|(index, part)| part.parent == NO_PART || usize::from(part.parent) < index),
			"a parent is always earlier in the list than its child"
		);
		assert_eq!(ragdoll.parts[0].bone, 1, "which here means the thigh came out first");
	}

	#[test]
	fn a_part_hangs_off_one_several_bones_above_it() {
		// a head on a chest, through a neck nobody gave a body to. A walk that
		// stopped at the bone's own parent would find the neck, see it is no
		// part, and call the head a root.
		let bones = leg();
		let ragdoll = plan(
			&bones,
			&[Segment::new("chest", "neck"), Segment::new("head", "crown")],
			Build::DEFAULT,
		);

		assert_eq!(ragdoll.len(), 2, "both were made");
		assert_eq!(ragdoll.parts[0].bone, 4, "the chest is the earlier bone");
		assert_eq!(ragdoll.parts[1].parent, 0, "and the head hangs off it past the neck");

		let mut at = Vec::new();
		rests(&bones, &mut at);

		let (head, chest) = (ragdoll.parts[1], ragdoll.parts[0]);
		let mine =
			(at[usize::from(head.bone)] * head.in_bone.matrix()).transform_point3(head.anchor);
		let theirs = (at[usize::from(chest.bone)] * chest.in_bone.matrix())
			.transform_point3(head.parent_anchor);

		assert!(
			mine.abs_diff_eq(theirs, 1e-5),
			"and the joint between them still meets, across the bone in between"
		);
	}

	#[test]
	fn a_segment_naming_a_bone_nothing_answers_to_is_counted_rather_than_guessed() {
		let ragdoll = plan(
			&leg(),
			&[
				Segment::new("thigh", "shin"),
				Segment::new("thigh", "elbow"),
				Segment::new("wing", "shin"),
				// there really is a bone with no name in this skeleton, and
				// asking for nothing must not find it.
				Segment::new("", "shin"),
				Segment::new("thigh", ""),
			],
			Build::DEFAULT,
		);

		assert_eq!(ragdoll.len(), 1, "only the one that names two real bones is made");
		assert_eq!(ragdoll.dropped(), 4, "and the other four are counted");
	}

	#[test]
	fn a_thickness_a_segment_gives_beats_the_one_a_ratio_would() {
		let ragdoll = plan(
			&leg(),
			&[Segment::new("thigh", "shin"), Segment::thick("shin", "foot", 0.05)],
			Build { mass: 0.0, girth: 0.5 },
		);

		assert!(
			(ragdoll.parts[0].shape.extents.y - 0.5).abs() < 1e-5,
			"a segment of one unit at half is half a unit thick"
		);
		assert!(
			(ragdoll.parts[1].shape.extents.y - 0.05).abs() < 1e-5,
			"and one that says its own thickness is that thick whatever the ratio"
		);
	}

	#[test]
	fn a_total_mass_is_shared_out_by_how_big_each_part_is() {
		let ragdoll =
			plan(&leg(), &[Segment::new("thigh", "shin"), Segment::new("shin", "foot")], Build {
				mass: 9.0,
				girth: 0.25,
			});
		let carried: f32 = ragdoll.parts.iter().map(|part| part.mass).sum();

		assert!((carried - 9.0).abs() < 1e-4, "the whole of it is shared, got {carried}");
		assert!(
			ragdoll.parts[1].mass > ragdoll.parts[0].mass * 4.0,
			"and the shin, twice as long and twice as thick, is much the heavier: {} against {}",
			ragdoll.parts[1].mass,
			ragdoll.parts[0].mass
		);
	}

	#[test]
	fn a_ragdoll_nobody_weighed_has_a_part_of_the_ordinary_weight() {
		let ragdoll = planned();

		assert!(
			ragdoll
				.parts
				.iter()
				.all(|part| (part.mass - Body::MASS).abs() < f32::EPSILON),
			"zero is a sentinel rather than a weightless ragdoll"
		);
	}

	#[test]
	fn more_segments_than_a_ragdoll_holds_are_counted_rather_than_kept() {
		let bones = leg();
		let asked = vec![Segment::new("thigh", "shin"); MAX_PARTS + 3];
		let ragdoll = plan(&bones, &asked, Build::DEFAULT);

		assert_eq!(ragdoll.len(), MAX_PARTS, "it stops at the ceiling");
		assert_eq!(ragdoll.dropped(), 3, "and says how many it would not take");
	}

	#[test]
	fn a_part_can_be_found_by_the_bone_it_follows() {
		let ragdoll = planned();

		assert_eq!(ragdoll.part_of(1), Some(0), "the thigh's bone is the first part");
		assert_eq!(ragdoll.part_of(2), Some(1), "and the shin's the second");
		assert_eq!(ragdoll.part_of(0), None, "while the hip is nobody's");
	}

	#[test]
	fn a_plan_of_no_segments_is_a_ragdoll_of_no_parts_rather_than_a_panic() {
		let ragdoll = plan(&leg(), &[], Build::DEFAULT);

		assert!(ragdoll.is_empty(), "nothing asked for is nothing made");
		assert_eq!(ragdoll.dropped(), 0, "and nothing refused either");
		assert_eq!(plan(&[], &[Segment::new("thigh", "shin")], Build::DEFAULT).dropped(), 1);
	}
}
