//! Keeping the navmesh in step with the world it describes.
//!
//! A [`Navmesh`] is derived, exactly as a [`Terrain`]'s geometry is, and this
//! is the module that derives it: once a step, **outside the edit-mode guard**,
//! and only when the static world it was baked from has actually moved. That
//! is the same arrangement [`terrain`](crate::terrain) keeps and for the same
//! argument - a person who has just dragged a hill into place has to see where
//! things can walk change while the world is stopped.
//!
//! **What decides "has moved" is a fingerprint rather than a comparison.** A
//! terrain's record is eight words and comparing it is a memcmp; the static
//! collision world is however many bodies there are, each with a transform and
//! a shape, and some of them naming a mesh whose contents can be replaced
//! underneath a handle that did not change. Folding all of that into one
//! number is one pass over the bodies a step - the same order of work the
//! terrain's own walk over the entities is - and it catches the case a handle
//! cannot: a recompiled `.obj`, or a terrain rebuilt at a new height, both of
//! which bump the registry's revision without moving anything.
//!
//! A collision costs a bake that should have happened and did not, over
//! sixty-four bits and a dozen floats. A rebuild that fires when nothing moved
//! costs the whole bake every step, which is why the fingerprint is checked
//! rather than a flag being set by everything that might have written.
//!
//! [`Navmesh`]: colby_core::abi::Navmesh
//! [`Terrain`]: colby_core::abi::Terrain

use std::time::Instant;

use colby_core::{
	abi::{BodyKind, NavSettings, Navmesh, PathKind, World, navmesh},
	glam::Vec3,
	info, trace, warn,
};
use colby_physics::Simulation;

/// The odd number an FNV-style fold multiplies by.
const PRIME: u64 = 0x0000_0100_0000_01B3;

/// What it starts from.
const SEED: u64 = 0xCBF2_9CE4_8422_2325;

/// What has been baked, so that nothing is baked twice.
///
/// The shape [`Ground`](crate::terrain::Ground) keeps, with a fingerprint
/// where it has a record: one thing remembered so that the usual case is a
/// comparison rather than the work.
#[derive(Clone, Debug, Default)]
pub(crate) struct Paths {
	/// The static world the navmesh was baked from, folded into one number.
	from: Option<u64>,

	/// How many cells anything may stand on.
	///
	/// A number about the project rather than about a frame, which is what
	/// `--profile` prints beside its means. A navmesh has no per-frame cost of
	/// its own - it is baked once and is then a lookup table - so a `cpu nav`
	/// row would read nil on every frame the table ever sees. How long a bake
	/// took is logged where it is spent instead, exactly as the terrain's is.
	cells: u64,

	/// The corners of the path `nav.show` last asked for.
	///
	/// Kept rather than allocated, because it is asked for again every step:
	/// what `nav.show` holds is *two places*, not one answer, so the path is
	/// found afresh against whatever the world now is. A wall dropped across
	/// it makes the drawn line go round on the next step, which is what makes
	/// it worth watching at all.
	line: Vec<Vec3>,

	/// What `nav.show` was holding when it was last understood.
	///
	/// Only so that a line nobody can read is complained about once instead of
	/// sixty times a second - a silent refusal is the failure mode this whole
	/// engine keeps being bitten by.
	said: String,
}

impl Paths {
	/// Nothing baked.
	#[must_use]
	pub(crate) fn new() -> Self { Self::default() }

	/// How many cells of the world are standable.
	pub(crate) const fn cells(&self) -> u64 { self.cells }
}

/// Brings the world's navmesh in line with its static bodies.
///
/// **The solver is asked to bake its collision meshes first**, and that is the
/// whole reason it is a parameter here. A bake asks the world where the ground
/// is by tracing rays at it; a trace against a mesh body answers out of
/// [`Simulation::prepare`]'s tables; and those were filled only by a step. A
/// navmesh built before the first step therefore came back with every ray
/// missing the landscape, which is indistinguishable from a world with no
/// ground in it - and since the fingerprint then said the world had not moved,
/// it never tried again. Only on a rebake, which is the step that is paying for
/// one anyway.
///
/// @param world - the host state: the bodies to bake from and the navmesh to
/// bake into
/// @param paths - what was baked last time
/// @param settings - how big the thing that walks is
/// @param simulation - the solver, if this process has one
pub(crate) fn sync(
	world: &mut World,
	paths: &mut Paths,
	settings: NavSettings,
	simulation: Option<&mut Simulation>,
) {
	let now = fingerprint(world, settings);

	if paths.from == Some(now) {
		return;
	}

	if let Some(simulation) = simulation {
		simulation.prepare(world);
	}

	let began = Instant::now();
	let cells = world.bake_nav(settings);
	let grid = world.nav();

	// logged rather than given a profiler row, for the terrain's reason: this
	// is spent once and a row reading nil on every frame but the first would
	// be noise where this is an answer.
	trace!(
		cells = grid.len(),
		walkable = cells,
		across = grid.across(),
		deep = grid.deep(),
		cell = grid.cell(),
		took_us = began.elapsed().as_micros(),
		"navmesh baked"
	);

	paths.from = Some(now);
	paths.cells = u64::try_from(cells).unwrap_or(0);
}

/// The whole static collision world, folded into one number.
///
/// **Every static body and the mesh revision of the ones that are meshes.**
/// The revision is what catches a terrain rebuilt at a new height or an `.obj`
/// recompiled on disk: both replace the geometry under a handle that did not
/// change, and a navmesh baked from the old shape would be a path through the
/// middle of the new one.
///
/// @param world - the world to fold
/// @param settings - the settings, because a bake at a different cell size is
/// a different bake
fn fingerprint(world: &World, settings: NavSettings) -> u64 {
	let mut seed = SEED;

	for number in [settings.cell, settings.radius, settings.step, settings.slope] {
		seed = mix(seed, number.to_bits());
	}

	for (id, body) in world.bodies.iter() {
		if body.kind != BodyKind::Static {
			continue;
		}

		seed = mix(seed, u32::try_from(id.slot()).unwrap_or(u32::MAX));
		seed = mix(seed, id.generation());
		seed = shaped(seed, body);
		seed = placed(seed, body);

		if let Some(mesh) = world.meshes.get(body.shape.mesh) {
			seed = mix(seed, mesh.revision());
		}
	}

	seed
}

/// Folds one body's shape in.
fn shaped(seed: u64, body: &colby_core::abi::Body) -> u64 {
	let shape = body.shape;
	let mut seed = mix(seed, shape.kind.index());

	seed = mix(seed, shape.radius.to_bits());
	seed = mix(seed, shape.mesh.index());

	for part in shape.extents.to_array() {
		seed = mix(seed, part.to_bits());
	}

	seed
}

/// Folds one body's placing in.
fn placed(seed: u64, body: &colby_core::abi::Body) -> u64 {
	let at = body.transform;
	let mut seed = seed;

	for part in at
		.position
		.to_array()
		.into_iter()
		.chain(at.scale.to_array())
		.chain(at.rotation.to_array())
	{
		seed = mix(seed, part.to_bits());
	}

	seed
}

/// One number folded into a running one.
///
/// FNV-1a over thirty-two bits at a time. Not a cryptographic hash and not
/// meant to be: what it has to do is come out different when the world is
/// different, and a fold this shape over a few dozen floats does.
fn mix(seed: u64, value: u32) -> u64 { (seed ^ u64::from(value)).wrapping_mul(PRIME) }

/// The settings the console is holding.
///
/// Read every step rather than kept, so that turning one of the four variables
/// takes effect on the next step without anything having to notice - which is
/// the whole of how the fingerprint above knows a bake is stale.
///
/// @param world - the world whose table the five were registered into
pub(crate) fn asked(world: &World) -> NavSettings {
	let read = |name: &str, fallback: f32| world.cvars.float(name).unwrap_or(fallback);

	NavSettings {
		cell: read(CELL, navmesh::CELL),
		radius: read(RADIUS, navmesh::RADIUS),
		step: read(STEP, NavSettings::DEFAULT.step),
		slope: read(SLOPE, NavSettings::DEFAULT.slope),
	}
	.sane()
}

/// The variable that says how wide one cell of the navmesh is.
pub(crate) const CELL: &str = "nav.cell";

/// The variable that says how wide the thing that walks is.
pub(crate) const RADIUS: &str = "nav.radius";

/// The variable that says how tall a lip it climbs.
pub(crate) const STEP: &str = "nav.step";

/// The variable that says the steepest ground it stands on.
pub(crate) const SLOPE: &str = "nav.slope";

/// The variable that draws the walkable set.
pub(crate) const DRAW: &str = "nav.draw";

/// The variable naming two places to draw the path between.
///
/// **A variable rather than a command**, and that is not a style choice:
/// `--shot` has no console at all (`console.rs:16`), so a command is a thing
/// no reproducible picture can ever contain, and the whole point of drawing a
/// path is to have a picture of one. It also comes out better than a command
/// would: what it holds is two *places*, so the path is found again every step
/// and a wall dropped across it is visible on the next one.
pub(crate) const SHOW: &str = "nav.show";

/// What color a standable cell is drawn in.
const WALKABLE: Vec3 = Vec3::new(0.2, 0.9, 0.4);

/// What a complete path is drawn in.
const FOUND: Vec3 = Vec3::new(0.95, 0.75, 0.15);

/// What a path that only got part of the way is drawn in.
///
/// A different color rather than a different shape, because the thing a
/// picture has to answer about a partial path is "did it get there", and a
/// line that stops has no way of saying whether it stopped because it arrived.
const SHORT: Vec3 = Vec3::new(0.95, 0.3, 0.2);

/// How far a drawn path floats above the ground, in world units.
///
/// A line lying exactly on the surface it was baked from fights the depth
/// buffer along its whole length, which reads as a dotted line rather than as
/// a path.
const LIFT: f32 = 0.25;

/// Draws the walkable set, cell by cell.
///
/// One cross a cell rather than a quad, because a quad's four lines are shared
/// with its neighbors and the debug table has no way to say so: a hundred by a
/// hundred grid would be forty thousand lines where this is twenty thousand,
/// and a cross reads better at a distance anyway.
///
/// The points are gathered before anything is drawn, because the grid lives on
/// the world and so does the pen. A `Vec` of them rather than a clone of the
/// grid: the grid is up to two megabytes and the open cells of a world worth
/// looking at are a fraction of it.
///
/// @param world - the world to draw into
fn cells(world: &mut World) {
	if !world.cvars.bool(DRAW).unwrap_or(false) {
		return;
	}

	let grid = world.nav();
	let arm = grid.cell() * 0.3;
	let mut open = Vec::new();

	for z in 0..grid.deep() {
		for x in 0..grid.across() {
			if let Some(at) = grid.point(x, z).filter(|_| grid.is_open(x, z)) {
				open.push(at);
			}
		}
	}

	let mut pen = world.debug.pen();

	for at in open {
		pen.line(at - Vec3::X * arm, at + Vec3::X * arm, WALKABLE);
		pen.line(at - Vec3::Z * arm, at + Vec3::Z * arm, WALKABLE);
	}
}

/// Draws whatever the two tools were asked for.
///
/// @param world - the world to draw into
/// @param paths - where the path being drawn is kept between steps
pub(crate) fn draw(world: &mut World, paths: &mut Paths) {
	cells(world);
	line(world, paths);
}

/// Finds and draws the path `nav.show` names, if it names one.
///
/// @param world - the world to ask and to draw into
/// @param paths - where the corners are kept between steps
fn line(world: &mut World, paths: &mut Paths) {
	let asked = world
		.cvars
		.text(SHOW)
		.unwrap_or_default()
		.to_owned();

	if asked.trim().is_empty() {
		paths.line.clear();
		paths.said.clear();

		return;
	}

	let Some((from, to)) = places(&asked) else {
		// once rather than every step, which is the difference between a
		// complaint and a wall of them.
		if paths.said != asked {
			paths.said.clone_from(&asked);
			warn!(asked, "{SHOW} wants four numbers: <x> <z> <x> <z>");
		}

		paths.line.clear();

		return;
	};

	let mut corners = std::mem::take(&mut paths.line);
	let kind = world.find_path(from, to, &mut corners);
	let color = match kind {
		| PathKind::Full => FOUND,
		| PathKind::Partial => SHORT,
		| PathKind::None => Vec3::ZERO,
	};

	if paths.said != asked {
		paths.said.clone_from(&asked);

		// the length in the plane rather than the number of corners: a funnel
		// turns a hundred cells into three, so a count says nothing about how
		// far anything is walking.
		let far: f32 = corners
			.windows(2)
			.map(|leg| leg[0].distance(leg[1]))
			.sum();

		info!(kind = ?kind, corners = corners.len(), length = far, "a path");
	}

	if kind.is_some() {
		show(world, &corners, color);
	}

	paths.line = corners;
}

/// Draws a path as a line with a mark at every corner.
///
/// **Each leg is drawn along the ground rather than as the chord between its
/// two ends**, and that is honesty rather than decoration: a corner of a
/// pulled path is a cell's *middle*, and the straight line between two of them
/// over rolling ground goes through every hill in between. Drawn as a chord it
/// disappears into the landscape and reads as a broken path; drawn along the
/// surface it is what somebody walking it would trace.
///
/// @param world - the world to draw into
/// @param path - the corners, in order
/// @param color - what to draw it in
fn show(world: &mut World, path: &[Vec3], color: Vec3) {
	let lift = Vec3::Y * LIFT;
	let step = world.nav().cell();
	let along = legs(world.nav(), path, step);
	let mut pen = world.debug.pen();

	for corner in path {
		pen.point(*corner + lift, LIFT * 4.0, color);
	}

	for leg in along.windows(2) {
		pen.line(leg[0] + lift, leg[1] + lift, color);
	}
}

/// Every corner of a path with the ground between them sampled in.
///
/// @param grid - the navmesh, for the heights
/// @param path - the corners, in order
/// @param step - how far apart the samples are; the caller passes the cell
/// size, because a sample finer than a cell says nothing the grid knows
fn legs(grid: &Navmesh, path: &[Vec3], step: f32) -> Vec<Vec3> {
	let mut along = Vec::new();

	for leg in path.windows(2) {
		let (from, to) = (leg[0], leg[1]);
		let taken = samples(from.distance(to), step);

		for sample in 0..taken {
			along.push(on_ground(grid, from.lerp(to, f32::from(sample) / f32::from(taken))));
		}
	}

	if let Some(&last) = path.last() {
		along.push(last);
	}

	along
}

/// How many samples one leg gets.
///
/// A whole number worked out once rather than a float walked up to a limit,
/// which is the same walk without a rounding question at the end of it. Capped
/// well above what a leg of [`REACH`](colby_core::abi::navmesh) cells needs,
/// because this is a debug line and not a budget.
///
/// @param far - how long the leg is
/// @param step - how far apart the samples should be
fn samples(far: f32, step: f32) -> u16 {
	let count = (far / step.max(f32::EPSILON)).ceil();

	if count.is_finite() { whole(count) } else { 1 }
}

/// A float as a count, clamped to something a `u16` holds.
fn whole(count: f32) -> u16 {
	let held = count.clamp(1.0, 4096.0);

	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "clamped to a span a u16 holds exactly, and the fraction is meant to go"
	)]
	let taken = held as u16;

	taken
}

/// A point put onto the surface of the cell it falls in.
fn on_ground(grid: &Navmesh, at: Vec3) -> Vec3 {
	grid.nearest(at)
		.and_then(|(x, z)| grid.point(x, z))
		.map_or(at, |on| Vec3::new(at.x, on.y, at.z))
}

/// The two places a line of four numbers names.
///
/// The height is not among them, and deliberately: the two ends of a path over
/// a grid are places on the ground, and which cell a point falls in is decided
/// by two of its three numbers anyway. @ref
/// [`Navmesh::nearest`](colby_core::abi::Navmesh::nearest).
///
/// @param said - what the variable holds
/// @return where from and where to, or `None` for anything else
fn places(said: &str) -> Option<(Vec3, Vec3)> {
	let mut numbers = said.split_whitespace().map(str::parse::<f32>);
	let mut next = || {
		numbers
			.next()?
			.ok()
			.filter(|it: &f32| it.is_finite())
	};
	let (x, z, to_x, to_z) = (next()?, next()?, next()?, next()?);

	numbers
		.next()
		.is_none()
		.then(|| (Vec3::new(x, 0.0, z), Vec3::new(to_x, 0.0, to_z)))
}

#[cfg(test)]
mod tests {
	use colby_core::abi::{Body, Shape, Transform, Value};

	use super::*;

	/// Settings the fixtures below share: a cell a fifth of the floor's width
	/// and an agent narrow enough that a doorway two units wide is one.
	const WALKER: NavSettings = NavSettings {
		cell: 0.5,
		radius: 0.4,
		..NavSettings::DEFAULT
	};

	/// A world with a real solver behind its traces, and the box holding it.
	///
	/// Boxed, because [`Simulation::table`] hands out a pointer to itself and
	/// the world it is installed into keeps it: both have to stop moving.
	fn wired() -> (Box<World>, Box<Simulation>) {
		let simulation = Box::new(Simulation::new());
		let mut world = Box::new(World::new());
		world.install_physics(simulation.table());

		(world, simulation)
	}

	/// Puts a static box in a world.
	///
	/// @param world - where to put it
	/// @param at - where its middle is
	/// @param extents - its half-extents
	fn block(world: &mut World, at: Vec3, extents: Vec3) {
		world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::cuboid(extents),
			Transform::at(at),
		));
	}

	/// A floor twenty units square, one unit thick, with its top at nil.
	fn floor(world: &mut World) {
		block(world, Vec3::new(0.0, -0.5, 0.0), Vec3::new(10.0, 0.5, 10.0));
	}

	#[test]
	fn a_world_with_nothing_static_in_it_bakes_nothing() {
		let (mut world, _held) = wired();

		assert_eq!(world.bake_nav(WALKER), 0, "no ground, nowhere to stand");
		assert!(world.nav().is_empty());
	}

	#[test]
	fn a_floor_bakes_into_somewhere_to_walk() {
		let (mut world, _held) = wired();
		floor(&mut world);

		let walkable = world.bake_nav(WALKER);
		let grid = world.nav();

		assert_eq!((grid.across(), grid.deep()), (40, 40), "twenty units at half a cell");
		assert!(walkable > 1000, "most of a bare floor is standable: {walkable}");

		// and the surface it found is the top of the box rather than its
		// middle or its underside, which is the one thing a downward ray can
		// get wrong without anything noticing.
		let middle = grid.point(20, 20).expect("in the grid");

		assert!(middle.y.abs() < 1.0e-3, "the floor is at nil: {middle:?}");
	}

	#[test]
	fn a_wall_across_the_floor_is_walked_round() {
		let (mut world, _held) = wired();
		floor(&mut world);
		// a wall two units tall along `x = 0`, stopping four units short of
		// the far edge, which leaves a doorway.
		block(&mut world, Vec3::new(0.0, 1.0, -2.0), Vec3::new(0.4, 1.0, 8.0));
		world.bake_nav(WALKER);

		let mut path = Vec::new();
		let kind =
			world.find_path(Vec3::new(-6.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0), &mut path);

		assert_eq!(kind, PathKind::Full, "the doorway is a way through");
		assert!(
			path.iter().any(|corner| corner.z > 5.0),
			"and the path goes through it rather than through the wall: {path:?}"
		);

		// no corner is inside the wall, which is what the whole thing is for.
		for corner in &path {
			assert!(corner.x.abs() > 0.4 || corner.z > 5.0, "{corner:?} is inside the wall");
		}
	}

	#[test]
	fn a_wall_with_no_door_leaves_the_far_side_unreachable() {
		let (mut world, _held) = wired();
		floor(&mut world);
		block(&mut world, Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.4, 1.0, 10.0));
		world.bake_nav(WALKER);

		let mut path = Vec::new();
		let kind =
			world.find_path(Vec3::new(-6.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0), &mut path);

		assert_eq!(kind, PathKind::Partial, "there is no way round");
		assert!(
			path.iter().all(|corner| corner.x < 0.0),
			"and the answer stays on the side it started: {path:?}"
		);
	}

	#[test]
	fn a_column_offers_its_topmost_surface_and_only_that_one() {
		// **the limit, written down as a test rather than as a comment.** A
		// slab a meter over the left half of the floor: what a bake finds
		// there is the top of the slab, and the floor beneath it - which is
		// perfectly flat and perfectly walkable in every other sense - is not
		// in the grid at all. Recast keeps every span a column has and this
		// keeps one, and the day that changes this test is what says so.
		let (mut world, _held) = wired();
		floor(&mut world);
		block(&mut world, Vec3::new(-5.0, 1.0, 0.0), Vec3::new(4.0, 0.2, 10.0));
		world.bake_nav(WALKER);

		let grid = world.nav();
		let over = grid
			.nearest(Vec3::new(-5.0, 0.0, 0.0))
			.expect("standable");
		let point = grid.point(over.0, over.1).expect("in the grid");

		assert!((point.y - 1.2).abs() < 1.0e-3, "{point:?} is not the top of the slab");
		assert!(
			point.x < -1.0,
			"and it is the cell asked about rather than one beside the slab: {point:?}"
		);
	}

	#[test]
	fn a_bake_needs_a_solver_and_says_so_by_finding_nothing() {
		// no `install_physics`, which is a dedicated server, an offscreen
		// capture and every unit test that never asked for one. The bounds are
		// still real, so this is the case where a wrong answer would look
		// exactly like a right one.
		let mut world = World::new();
		floor(&mut world);

		assert_eq!(world.bake_nav(WALKER), 0, "every trace missed");
		assert!(!world.nav().is_empty(), "and the grid is still the right size");
	}

	#[test]
	fn nothing_is_baked_twice_for_a_world_that_did_not_move() {
		let (mut world, mut held) = wired();
		let mut paths = Paths::new();
		floor(&mut world);

		sync(&mut world, &mut paths, WALKER, Some(&mut held));

		let first = paths.from;

		assert!(first.is_some(), "the first step baked");
		assert!(paths.cells() > 0);

		sync(&mut world, &mut paths, WALKER, Some(&mut held));

		assert_eq!(paths.from, first, "and the second step did not");
	}

	#[test]
	fn a_static_body_that_moved_is_a_bake_and_a_dynamic_one_is_not() {
		let (mut world, mut held) = wired();
		let mut paths = Paths::new();
		floor(&mut world);

		let rolling = world.bodies.spawn(Body::dynamic(
			Shape::ball(0.5),
			Transform::at(Vec3::new(0.0, 4.0, 0.0)),
			1.0,
		));
		let standing =
			world
				.bodies
				.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::at(Vec3::Y)));

		sync(&mut world, &mut paths, WALKER, Some(&mut held));

		let settled = paths.from;

		// a crate falling through the air changes nothing about where the
		// world lets anybody stand, and re-baking every step it moves would be
		// the whole cost of this for none of the answer.
		if let Some(body) = world.bodies.get_mut(rolling) {
			body.transform.position = Vec3::new(3.0, 2.0, 1.0);
		}

		sync(&mut world, &mut paths, WALKER, Some(&mut held));

		assert_eq!(paths.from, settled, "a dynamic body is not the world");

		if let Some(body) = world.bodies.get_mut(standing) {
			body.transform.position = Vec3::new(-3.0, 1.0, 2.0);
		}

		sync(&mut world, &mut paths, WALKER, Some(&mut held));

		assert_ne!(paths.from, settled, "and a static one is");
	}

	#[test]
	fn a_setting_that_changed_is_a_bake() {
		let (mut world, mut held) = wired();
		let mut paths = Paths::new();
		floor(&mut world);

		sync(&mut world, &mut paths, WALKER, Some(&mut held));

		let coarse = paths.from;
		let cells = paths.cells();

		sync(&mut world, &mut paths, NavSettings { cell: 1.0, ..WALKER }, Some(&mut held));

		assert_ne!(paths.from, coarse, "a different cell is a different bake");
		assert!(paths.cells() < cells, "and a coarser one has fewer of them");
	}

	#[test]
	fn geometry_replaced_under_a_handle_is_a_bake() {
		// the case a handle cannot answer and the registry's revision can: a
		// terrain rebuilt at a new height, or an `.obj` recompiled on disk,
		// both leave every body and every id exactly where it was.
		let (mut world, mut held) = wired();
		let mut paths = Paths::new();
		let mesh = world
			.meshes
			.insert("nav.fixture", colby_core::abi::mesh::quad());

		world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::mesh(mesh), Transform::IDENTITY));
		sync(&mut world, &mut paths, WALKER, Some(&mut held));

		let before = paths.from;

		world
			.meshes
			.insert("nav.fixture", colby_core::abi::mesh::cube());
		sync(&mut world, &mut paths, WALKER, Some(&mut held));

		assert_ne!(paths.from, before, "the revision moved even though nothing else did");
	}

	#[test]
	fn the_four_variables_are_what_a_bake_is_told() {
		let mut world = World::new();

		world.cvars.var(CELL, Value::Float(0.25), "");
		world.cvars.var(RADIUS, Value::Float(0.75), "");
		world.cvars.var(STEP, Value::Float(0.6), "");
		world.cvars.var(SLOPE, Value::Float(0.5), "");

		let settings = asked(&world);

		assert!((settings.cell - 0.25).abs() < f32::EPSILON);
		assert!((settings.radius - 0.75).abs() < f32::EPSILON);
		assert!((settings.step - 0.6).abs() < f32::EPSILON);
		assert!((settings.slope - 0.5).abs() < f32::EPSILON);

		// and a table with none of them registered is the defaults rather than
		// a bake of nothing, which is every process that never opened a
		// console.
		let bare = asked(&World::new());

		assert_eq!(bare, NavSettings::DEFAULT.sane());
	}
}
