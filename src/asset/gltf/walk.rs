//! A real walk, imported and played with its root motion on and off.
//!
//! The fixture is a seven-bone walker made with **Blender 5.2.1**, headless,
//! by a script that keeps each stance foot still on the ground while the pelvis
//! travels a unit a lap at an uneven pace, every bone keyed at every sixtieth
//! of a second, LINEAR. It goes through the importer every model goes through.
//!
//! The numbers it is held to came from a second reader, written against the
//! glTF specification rather than against this crate, which read the same
//! file and said:
//!
//! - the pelvis travels exactly one unit along `+z` a lap and never turns;
//! - carried by the pelvis's travel, a stance foot keeps its place to within
//!   `3e-7` at thirty and sixty steps a second, where every step lands on a
//!   key, and to within `1.7e-4` at a hundred and forty-four, where the file's
//!   own straight lines between keys stand in for the curve the legs were
//!   solved on;
//! - walked at an even pace instead, with the pelvis's travel thrown away, a
//!   whole stance slides `0.05967`;
//! - and played in place on a walker that stays put, the right foot jumps back
//!   a whole lap at the seam while it is still on the ground.
//!
//! The last two are the negative controls, and they are why the first two mean
//! anything: the same fixture measured the same way does slide, by the amount
//! the other reader predicted, as soon as the travel is not what moves it.
//!
//! **The legs were bent by a two-bone solve the script worked out for
//! itself**, every knee forwards, so the same file holds a second answer to
//! what [`reach`] works out. A leg held crooked and bent back to its own ankle,
//! the knee sent in front of the hip, puts the knee where that other solve did
//! at every key of the lap, to within `4.8e-7`; sent behind the hip, the same
//! knee lands well behind it, which is the control.

use std::path::Path;

use colby_core::{
	abi::{
		ClipId, NO_BONE, Node, SkeletonId, Transform, Tree, World,
		character::{Motion, move_and_slide},
		ik::{Reach, reach},
		pose::{Pose, PoseId},
	},
	glam::{EulerRot, Mat4, Quat, Vec3},
};

use super::{Gltf, import};

/// The walker and its one clip, as an exporter wrote them.
const WALK: &[u8] = include_bytes!("fixtures/walk.glb");

/// How far the pelvis travels in a lap, along `+z`.
const LAP: f32 = 1.0;

/// How far a whole stance slides when the walker goes at an even pace of a lap
/// a lap and its pelvis is pinned, as the other reader worked it out.
const PACED_SLIDE: f32 = 0.059_67;

/// How a run moves its walker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carried {
	/// By its pelvis's own travel: root motion.
	ByTravel,

	/// Pelvis pinned, and walked at an even pace of a lap a lap instead, which
	/// is what a game did before there was any travel to hand it.
	AtPace,

	/// Not at all, and the clip played in place with nothing pinned.
	Not,
}

/// The walker loaded into a world, the way a game would have it.
struct Walker {
	world: World,
	pose: PoseId,
	clip: ClipId,
	skeleton: SkeletonId,
	pelvis: u16,
	feet: [usize; 2],
}

/// Imports the fixture and stands the walker up in a world of its own.
fn walker() -> Walker {
	let file = Gltf::read(WALK, Path::new("walk.glb"), Path::new("")).expect("the fixture reads");
	let model = import(&file).expect("and imports");
	let rig = model
		.skins
		.first()
		.expect("one skin")
		.data
		.clone();
	let walk = model
		.clips
		.iter()
		.find(|clip| clip.name == "walk")
		.expect("a clip called walk")
		.data
		.clone();
	let pelvis = rig.find("pelvis").expect("a pelvis");
	let feet = [rig.find("foot_l"), rig.find("foot_r")]
		.map(|foot| usize::from(foot.expect("both feet")));
	let mut world = World::new();
	let skeleton = world.skeletons.insert("walker", rig);
	let clip = world.clips.insert("walk", walk);
	let pose = world
		.poses
		.spawn(Pose::resting(skeleton, world.skeletons.bones(skeleton)));

	Walker {
		world,
		pose,
		clip,
		skeleton,
		pelvis,
		feet,
	}
}

/// Where the walker ended up, and where its feet were at every step.
struct Run {
	/// Its own transform once the run is over.
	standing: Transform,

	/// Each step's number, and each foot's place in the world after it.
	steps: Vec<(u16, [Vec3; 2])>,
}

/// Moves the walker from where it stands to where it is carried, through the
/// controller a game moves a character with, turned the way it is carried.
fn walked(world: &World, standing: Transform, there: Transform, dt: f32) -> Transform {
	let velocity = (there.position - standing.position) / dt;
	let moved =
		move_and_slide(world, &Motion::new(standing.position, velocity, Vec3::splat(0.25), dt));

	Transform {
		position: moved.position,
		rotation: there.rotation,
		scale: standing.scale,
	}
}

/// Plays the walk for so many laps at so many steps a second.
fn run(carried: Carried, rate: u16, laps: u16) -> Run {
	let Walker {
		mut world,
		pose,
		clip,
		skeleton,
		pelvis,
		feet,
	} = walker();
	let dt = 1.0 / f32::from(rate);
	let mut standing = Transform::IDENTITY;
	let mut matrices: Vec<Mat4> = Vec::new();
	let mut steps = Vec::new();

	for step in 1..=rate * laps {
		let from = f32::from(step - 1) / f32::from(rate);
		let time = f32::from(step) / f32::from(rate);
		let mut tree = Tree::new();

		tree.push(Node::Clip { clip, time, from, looping: true });
		tree.motion = if carried == Carried::Not { NO_BONE } else { pelvis };

		let travel = world.travel(pose, &tree);

		assert!(world.animate(pose, &tree), "the tree works out");

		let there = match carried {
			| Carried::ByTravel => travel.applied(standing),
			| Carried::AtPace => Transform {
				position: standing.position + standing.rotation * Vec3::new(0.0, 0.0, LAP * dt),
				..standing
			},
			| Carried::Not => standing,
		};

		standing = walked(&world, standing, there, dt);
		matrices.clear();
		world
			.poses
			.model(pose, world.skeletons.bones(skeleton), 1.0, &mut matrices);

		let place = |foot: usize| {
			(standing.matrix() * matrices[foot])
				.w_axis
				.truncate()
		};

		steps.push((step, [place(feet[0]), place(feet[1])]));
	}

	Run { standing, steps }
}

/// One time one foot was on the ground: which foot, and the lap it came down
/// in.
type Stance = (usize, i32);

/// Which stance a moment of the lap belongs to, if its foot is down.
///
/// The windows the walk was made with, each a hundredth of a lap inside its
/// ends so that a foot lifting or landing on the boundary is not counted. The
/// right foot comes down in one lap and lifts in the next.
fn stance_of(foot: usize, phase: f32, lap: i32) -> Option<Stance> {
	match foot {
		| 0 => (0.01..=0.59).contains(&phase).then_some((0, lap)),
		| _ if phase >= 0.51 => Some((1, lap)),
		| _ => (phase <= 0.09).then_some((1, lap - 1)),
	}
}

/// Files one place under the stance it belongs to.
fn file(groups: &mut Vec<(Stance, Vec<Vec3>)>, stance: Option<Stance>, place: Vec3) {
	let Some(stance) = stance else {
		return;
	};

	if let Some((_, places)) = groups.iter_mut().find(|(key, _)| *key == stance) {
		places.push(place);

		return;
	}

	groups.push((stance, vec![place]));
}

/// Every stance of a run, as the places its foot stood at step by step.
fn stances(walked: &Run, rate: u16) -> Vec<Vec<Vec3>> {
	let mut groups: Vec<(Stance, Vec<Vec3>)> = Vec::new();

	for (step, feet) in &walked.steps {
		let lap = i32::from(step / rate);
		let phase = f32::from(step % rate) / f32::from(rate);

		for (foot, place) in feet.iter().enumerate() {
			file(&mut groups, stance_of(foot, phase, lap), *place);
		}
	}

	groups
		.into_iter()
		.map(|(_, places)| places)
		.collect()
}

/// How far apart the farthest two places of a stance are, along the worst axis.
fn spread(places: &[Vec3]) -> f32 {
	let (low, high) = places
		.iter()
		.fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(low, high), place| {
			(low.min(*place), high.max(*place))
		});

	(high - low).max_element()
}

/// The largest step a foot took from one moment to the next while it was down.
fn jump(places: &[Vec3]) -> f32 {
	places
		.windows(2)
		.map(|pair| pair[1].distance(pair[0]))
		.fold(0.0, f32::max)
}

#[test]
fn a_real_walk_carried_by_its_pelvis_goes_a_lap_a_lap_at_every_rate() {
	for rate in [30_u16, 60, 144] {
		let walked = run(Carried::ByTravel, rate, 3);
		let at = walked.standing.position;

		assert!(
			at.abs_diff_eq(Vec3::new(0.0, 0.0, 3.0 * LAP), 1.0e-4),
			"at {rate} a second three laps are three units along +z, and it went to {at}"
		);
		assert!(
			walked
				.standing
				.rotation
				.abs_diff_eq(Quat::IDENTITY, 1.0e-5),
			"and a pelvis that never turns turns nobody"
		);
	}
}

#[test]
fn a_real_walk_carried_by_its_pelvis_keeps_every_stance_foot_where_it_came_down() {
	// on the keys the file is the walk it was solved as; between them, at a
	// hundred and forty-four, its straight lines are what is being played.
	for (rate, allowed) in [(30_u16, 1.0e-5), (60, 1.0e-5), (144, 2.0e-4)] {
		let walked = run(Carried::ByTravel, rate, 3);
		let all = stances(&walked, rate);
		let worst = all
			.iter()
			.map(|places| spread(places))
			.fold(0.0, f32::max);

		assert!(
			all.len() >= 6,
			"at {rate} a second, {} stances is too few to mean much",
			all.len()
		);
		assert!(
			worst <= allowed,
			"at {rate} a second a foot on the ground moved {worst}, and may move {allowed}"
		);
	}
}

#[test]
fn walked_at_an_even_pace_instead_the_same_stance_feet_slide() {
	let walked = run(Carried::AtPace, 60, 3);
	let longest = stances(&walked, 60)
		.iter()
		.map(|places| spread(places))
		.fold(0.0, f32::max);

	assert!(
		(longest - PACED_SLIDE).abs() < 2.0e-4,
		"a stance slides {longest}, where the reader that is not this crate said {PACED_SLIDE}"
	);
	assert!(
		walked
			.standing
			.position
			.abs_diff_eq(Vec3::new(0.0, 0.0, 3.0 * LAP), 1.0e-4),
		"though the walker went just as far: {}",
		walked.standing.position
	);
}

#[test]
fn played_in_place_on_a_walker_that_stays_put_the_walk_goes_nowhere_and_jumps_at_the_seam() {
	let walked = run(Carried::Not, 60, 3);
	let largest = stances(&walked, 60)
		.iter()
		.map(|places| jump(places))
		.fold(0.0, f32::max);

	assert!(
		walked
			.standing
			.position
			.abs_diff_eq(Vec3::ZERO, 1.0e-6),
		"nothing carried it anywhere"
	);
	assert!(
		(largest - LAP).abs() < 1.0e-3,
		"and the right foot, still on the ground, went back a whole lap at the seam: {largest}"
	);
}

/// How far a knee bent back may land from where the walk had it, and a foot
/// from the ankle it was bent to.
///
/// What the file's own rounding allows: the knees the script solved for went
/// out through its exporter as single-precision turns and came back in
/// through this crate's importer. Measured, the worst of the lap's hundred and
/// twenty-two legs put a knee `4.8e-7` from the script's and a foot `4.0e-7`
/// from its ankle; this is four times that, and still a hundred thousand times
/// less than a knee bent the wrong way misses by.
const BENT_WITHIN: f32 = 2.0e-6;

/// Where every bone of the walker's pose is now, in the walker's own space.
fn laid(world: &World, pose: PoseId, skeleton: SkeletonId, into: &mut Vec<Mat4>) -> Vec<Vec3> {
	into.clear();
	world
		.poses
		.model(pose, world.skeletons.bones(skeleton), 1.0, into);

	into.iter()
		.map(|model| model.w_axis.truncate())
		.collect()
}

/// One leg at one frame: where the walk had the knee, where the leg held
/// crooked had it, and where bending the crooked leg back put it.
struct Knee {
	/// Where the walk had it.
	walked: Vec3,

	/// Where the leg held crooked had it.
	crooked: Vec3,

	/// Where bending the crooked leg back to the walk's ankle put it.
	bent: Vec3,

	/// How far the foot landed from that ankle.
	missed: f32,
}

/// Plays one frame of the walk, holds one leg crooked, and bends it back to
/// the ankle the walk had, with the knee sent to one side of the hip.
///
/// Crooked rather than straight, because a straight leg is not always
/// somewhere else: a foot trailing behind the hip bends the knee forwards until
/// the thigh hangs very nearly plumb, which is where a straight leg has its
/// knee too. Turned out to the side and twisted, the thigh and the calf put
/// the knee out of the walk's plane at every moment of the lap.
///
/// @param foot - which foot's leg
/// @param frame - which sixtieth of the lap
/// @param toward - which way from the hip the knee is sent, in the walker's
/// own space
fn rebend(walker: &mut Walker, foot: usize, frame: u16, toward: Vec3) -> Knee {
	let Walker { world, pose, clip, skeleton, .. } = walker;
	let bones = world.skeletons.bones(*skeleton).to_vec();
	let calf = usize::from(bones[foot].parent);
	let thigh = usize::from(bones[calf].parent);
	let mut matrices = Vec::new();

	assert!(world.play(*pose, *clip, f32::from(frame) / 60.0, true), "the pose is there");

	let walked = laid(world, *pose, *skeleton, &mut matrices);
	let posed = world
		.poses
		.get_mut(*pose)
		.expect("the pose is there");

	posed.locals[thigh].rotation = Quat::from_euler(EulerRot::XYZ, 0.9, 0.2, 0.7);
	posed.locals[calf].rotation = Quat::from_euler(EulerRot::XYZ, -0.8, 0.5, 0.1);

	let crooked = laid(world, *pose, *skeleton, &mut matrices);
	let end = u16::try_from(foot).expect("a bone index fits");
	let wanted = Reach::new(end, walked[foot]).bending(walked[thigh] + toward);
	let posed = world
		.poses
		.get_mut(*pose)
		.expect("the pose is there");

	assert_eq!(
		reach(&bones, &mut posed.locals, &wanted, &mut matrices),
		Some(0.0),
		"an ankle the walk put somewhere is within its own leg's reach"
	);

	let bent = laid(world, *pose, *skeleton, &mut matrices);

	Knee {
		walked: walked[calf],
		crooked: crooked[calf],
		bent: bent[calf],
		missed: bent[foot].distance(walked[foot]),
	}
}

#[test]
fn a_leg_held_crooked_and_bent_back_to_its_ankle_puts_the_knee_where_the_walk_had_it() {
	// the walk was made by a two-bone solve the script worked out for itself,
	// every knee bent forwards: so a leg held crooked and bent back to the
	// ankle the file has, the knee sent in front of the hip, has to come back to
	// where that other solve put it, at every one of the lap's sixty-one keys.
	let mut walker = walker();
	let feet = walker.feet;
	let mut worst = (0.0_f32, 0.0_f32);

	for frame in 0..=60 {
		for foot in feet {
			let knee = rebend(&mut walker, foot, frame, Vec3::Z);

			assert!(
				knee.crooked.distance(knee.walked) > 0.05,
				"frame {frame}: the leg held crooked had its knee somewhere else to start from"
			);

			worst = (worst.0.max(knee.bent.distance(knee.walked)), worst.1.max(knee.missed));
		}
	}

	assert!(
		worst.0 < BENT_WITHIN && worst.1 < BENT_WITHIN,
		"a knee came back {} from where the walk had it, and a foot landed {} from its ankle",
		worst.0,
		worst.1
	);
}

#[test]
fn sent_behind_the_hip_the_same_knee_bends_backwards_by_the_whole_bend() {
	// the negative control: the same legs and the same ankles, the knee sent
	// the other way, and it has to land well behind where the walk had it.
	let mut walker = walker();
	let feet = walker.feet;

	for frame in [0_u16, 15, 30, 45] {
		for foot in feet {
			let knee = rebend(&mut walker, foot, frame, Vec3::NEG_Z);

			assert!(
				knee.missed < BENT_WITHIN,
				"the foot still gets to its ankle, {} off",
				knee.missed
			);
			assert!(
				knee.bent.z < knee.walked.z - 0.05,
				"frame {frame}: a knee sent backwards went to {} where the walk had it at {}",
				knee.bent,
				knee.walked
			);
		}
	}
}
