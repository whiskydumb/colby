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

use std::path::Path;

use colby_core::{
	abi::{
		ClipId, NO_BONE, Node, SkeletonId, Transform, Tree, World,
		character::{Motion, move_and_slide},
		pose::{Pose, PoseId},
	},
	glam::{Mat4, Quat, Vec3},
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
