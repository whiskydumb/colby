//! The host's solver, and the two queries it answers for the game.
//!
//! What this crate owns is deliberately small, and what it does *not* own is
//! the point. The authoritative state of every body - its shape, where it is,
//! what its surface is like - lives in `World::bodies`, plain data in
//! `colby_core` that the editor can read and a saved scene could write. This
//! crate owns only what is derived from that and can be thrown away: baked
//! collision meshes today, a broadphase and a contact cache later.
//!
//! That is the opposite of what an external physics library forces. Take one
//! and every entity's position, rotation and velocity has to be pushed *into*
//! it every tick and read back out, because the authority is outside the solver
//! and the solver insists on being the authority. Nothing here mirrors
//! anything.
//!
//! **The game reaches this through two function pointers and nothing else.**
//! [`Simulation::table`] hands the host a `#[repr(C)]`
//! [`Physics`](colby_core::abi::Physics) to install into the world, and the
//! game calls `world.trace_ray(..)` and `world.trace_box(..)` through it. Two,
//! because a query is the only part of physics that needs to know more than the
//! body table already says. Everything else a game does to a body - create it,
//! move it, make it kinematic - is a write into plain data, which is why this
//! table is the whole of the boundary and does not grow.
//!
//! The pointers address the executable, which is never unloaded, so a module
//! swap does not disturb them and the host installs them once.
//!
//! **What is here and what is not.** Ray and swept-box queries against boxes,
//! balls and triangle meshes; the transform relationship between a body and the
//! entity it is attached to. No dynamics: nothing falls, nothing collides,
//! `restitution` and `friction` are declared and unread. That is the next
//! piece, and it lands behind this boundary without moving it.

// @note: crate-wide opt-in to the workspace `unsafe-code = "deny"`. The unsafe
// here is confined to the three functions at the bottom of this file, which
// turn the raw arguments of a `TraceFn` back into references. It is the
// unavoidable half of a `#[repr(C)]` boundary and it is the only half: the
// geometry in `query` never sees a pointer.
#![allow(unsafe_code)]

use core::ffi::c_void;
use std::collections::HashSet;

use colby_core::{
	abi::{
		Bodies, BodyId, Entry, MeshData, Overlap, Physics, Shape, ShapeKind, Touch, TouchKind,
		TraceInfo, TraceResult, World,
	},
	glam::Vec3,
	trace,
};

mod contact;
mod convex;
pub mod debug;
mod query;
mod solve;

pub use self::solve::{MAX_PASSES, VELOCITY_PASSES};
use self::{contact::Manifold, query::Collider, solve::Solver};

/// The console variable that says how hard the solver tries.
///
/// Read every step rather than at startup, so that turning it while watching a
/// pile is a thing a person can do. @ref
/// [`VELOCITY_PASSES`](solve::VELOCITY_PASSES) for what the number means.
pub const PASSES: &str = "phys.passes";

/// How far a body may move in one step, as a share of its own smallest half,
/// before it is swept rather than simply moved.
///
/// Below one a body always overlaps its own previous position, so nothing can
/// pass through anything and the sweep is wasted work. A little under one buys
/// a margin for the shapes that are not boxes.
const SWEEP_AT: f32 = 0.8;

/// How far short of what it hit a swept body is put.
///
/// Exactly on the surface is a contact of zero depth, which the narrow phase
/// discards as too shallow to matter - so the body would be left touching
/// nothing and would sail on through next step.
const SWEEP_SKIN: f32 = 0.01;

/// The shortest step the solver will divide by.
///
/// A step of zero is a world nobody paced, and the overlap recovery term is one
/// over it.
const MINIMUM_STEP: f32 = 1.0e-4;

/// The host's physics state.
///
/// Boxed by whoever owns it and never moved afterwards: [`table`](Self::table)
/// hands out its address, and a [`World`] holds that address for the rest of
/// the process.
#[derive(Debug, Default)]
pub struct Simulation {
	/// A baked collision mesh per body slot, for the bodies that have one.
	colliders: Vec<Option<Collider>>,

	/// Which occupant of each slot the entry beside it was baked for.
	generations: Vec<u32>,

	/// Scratch for [`sync`](Self::sync): which slots are occupied right now.
	///
	/// A field rather than a local so that a step does not allocate.
	live: Vec<bool>,

	/// Where every body was at the top of this step.
	///
	/// The sweep's: what a body passed *through* is the segment between where
	/// it was and where the solver has just put it. @ref
	/// [`sweep`](Self::sweep).
	was: Vec<Vec3>,

	/// Which pairs were touching at the end of the previous step.
	///
	/// The difference between this and the pairs touching now is exactly the
	/// queue of things that began and ended, which is what a trigger volume
	/// wants and what reading a table every step cannot tell it.
	previous: HashSet<(BodyId, BodyId)>,

	/// What the narrow phase found this step, for the solver to separate.
	///
	/// Kept between steps for its allocation only; it is cleared and refilled
	/// every time. There is no contact cache yet - @ref the crate docs.
	manifolds: Vec<Manifold>,

	/// What a sensor overlapped this step.
	///
	/// The same list kept apart, because the solver must never see one: a
	/// sensor produces an event and never an impulse. @ref
	/// [`contact::find`].
	sensed: Vec<Manifold>,

	/// The sequential-impulse solver and its per-body scratch.
	solver: Solver,
}

impl Simulation {
	/// A simulation with nothing in it.
	#[must_use]
	pub fn new() -> Self {
		Self {
			colliders: Vec::new(),
			generations: Vec::new(),
			live: Vec::new(),
			was: Vec::new(),
			previous: HashSet::new(),
			manifolds: Vec::new(),
			sensed: Vec::new(),
			solver: Solver::new(),
		}
	}

	/// The table a [`World`] is given so the game can ask questions.
	///
	/// # Panics
	///
	/// Never. It is the *caller* that has an obligation: this hands out the
	/// address of `self`, so the value must be behind a `Box` (or otherwise
	/// pinned) and must outlive every world it is installed into. Moving it
	/// afterwards leaves a world holding a stale pointer.
	#[must_use]
	pub fn table(&self) -> Physics {
		let context: *mut Self = core::ptr::from_ref(self).cast_mut();

		Physics::new(context.cast::<c_void>(), trace_ray, trace_box)
	}

	/// Advances the simulation by one step.
	///
	/// Called from inside the fixed step, before the game's `update`, which is
	/// the order that lets gameplay read a post-solve world: what the game sees
	/// in `update` is what the frame after it will draw.
	///
	/// Nothing here writes a second transform or an interpolation factor. A
	/// body writes its entity's transform inside the step like any other
	/// gameplay code would, and the host's existing `advance`/`settle` pair
	/// blends it for the renderer.
	///
	/// @param world - the host state; `bodies` is read, `entities` written
	pub fn step(&mut self, world: &mut World) {
		self.sync(world);
		Self::pull(world);

		// taken out and put back so that the narrow phase can borrow the
		// simulation for its collision meshes while filling the list that lives
		// on it. The allocation survives the round trip, which is the only
		// reason the list is a field at all.
		let mut manifolds = core::mem::take(&mut self.manifolds);
		let mut sensed = core::mem::take(&mut self.sensed);
		contact::find(&world.bodies, self, &mut manifolds, &mut sensed);
		self.report(world, &manifolds, &sensed);
		self.remember_where(&world.bodies);

		let dt = world.dt.max(MINIMUM_STEP);
		let passes = world
			.cvars
			.float(PASSES)
			.map_or(VELOCITY_PASSES, asked_for);

		self.solver
			.run(&mut world.bodies, &world.joints, &manifolds, world.gravity, dt, passes);
		self.manifolds = manifolds;
		self.sensed = sensed;
		world.contacts =
			u32::try_from(self.manifolds.len() + self.sensed.len()).unwrap_or(u32::MAX);

		self.sweep(world);
		Self::push(world);

		// last, so that what is drawn is the world the *next* frame will show
		// rather than the one the last frame did. Costs one console lookup per
		// variable when it is off, which is what off should cost.
		debug::draw(world, self);
	}

	/// How many pairs the narrow phase found on the last step.
	///
	/// The number a statistics panel wants, and the one to look at first when a
	/// pile behaves oddly.
	#[must_use]
	pub fn contacts(&self) -> usize { self.manifolds.len() }

	/// What the narrow phase found on the last step.
	///
	/// The debug drawing's, which is the only thing that wants the points
	/// themselves rather than how many there are.
	pub(crate) fn manifolds(&self) -> &[Manifold] { &self.manifolds }

	/// Queues what started and stopped touching since the last step, and
	/// rebuilds the list of what is inside a sensor.
	///
	/// Edges and state, from the same pass over the same pairs. Neither can be
	/// had from the other: an edge cannot be recovered from a list read every
	/// step, and a list kept by adding and removing edges goes wrong the first
	/// time something is destroyed while it is inside.
	///
	/// @param world - the body table, whose queue and overlap list are written
	/// @param manifolds - what the narrow phase found for the solver
	/// @param sensed - what it found for the sensors
	fn report(&mut self, world: &mut World, manifolds: &[Manifold], sensed: &[Manifold]) {
		let mut now: HashSet<(BodyId, BodyId)> = HashSet::new();

		world.bodies.forget_overlaps();

		for manifold in manifolds.iter().chain(sensed) {
			if manifold.count == 0 {
				continue;
			}

			let pair = ordered(manifold.first, manifold.second);

			// a mesh makes one manifold per triangle, so everything below this
			// happens once for a pair rather than once for a manifold.
			if !now.insert(pair) {
				continue;
			}

			if let Some(overlap) = overlapping(&world.bodies, pair) {
				world.bodies.overlapped(overlap);
			}

			if self.previous.contains(&pair) {
				continue;
			}

			let point = manifold.points()[0];

			world.bodies.touched(Touch {
				first: pair.0,
				second: pair.1,
				kind: TouchKind::Began,
				point: point.position,
				normal: manifold.normal,
			});
		}

		for &pair in &self.previous {
			if now.contains(&pair) {
				continue;
			}

			let asleep = |id: BodyId| {
				world
					.bodies
					.get(id)
					.is_some_and(|body| body.sleeping)
			};
			let alive = (world.bodies.alive(pair.0), world.bodies.alive(pair.1));

			// a pair that has gone to sleep is still touching; the narrow phase
			// simply stopped asking. Reporting that as having parted would make
			// every settled pile announce its own collapse.
			if alive.0 && alive.1 && asleep(pair.0) && asleep(pair.1) {
				now.insert(pair);

				continue;
			}

			// both gone, and there is nobody left to tell. One gone is the case
			// that matters, and it is reported: the survivor is often a trigger
			// volume whose contents were deleted under it, and a count it never
			// hears about is wrong for the rest of the process. The dead handle
			// goes out as it was, because that is what the game will recognize.
			if !alive.0 && !alive.1 {
				continue;
			}

			world.bodies.touched(Touch {
				first: pair.0,
				second: pair.1,
				kind: TouchKind::Ended,
				point: Vec3::ZERO,
				normal: Vec3::ZERO,
			});
		}

		self.previous = now;
	}

	/// Notes where everything is, before the solver moves it.
	fn remember_where(&mut self, bodies: &Bodies) {
		self.was.clear();
		self.was.resize(bodies.slots(), Vec3::ZERO);

		for (id, body) in bodies.iter() {
			self.was[id.slot()] = body.transform.position;
		}
	}

	/// Pulls back anything that moved far enough in one step to pass through
	/// something.
	///
	/// A step is a sixtieth of a second and a body is moved in one go, so a
	/// prop traveling faster than its own thickness per step can be on one
	/// side of a wall at the start and the other side at the end, with no
	/// moment in between for the narrow phase to notice. The cure is to trace
	/// the line it took and put it back at the first thing on it.
	///
	/// Only bodies that actually moved that far are swept, which in a normal
	/// scene is none of them: the test is against the body's *own* size, so a
	/// pebble is swept at a speed a crate is not.
	///
	/// @param world - the bodies to check and correct
	fn sweep(&self, world: &mut World) {
		let mut pulled: Vec<(BodyId, Vec3)> = Vec::new();

		for (id, body) in world.bodies.iter() {
			// a sensor is meant to pass through things, so pulling it back out
			// of the first one it crossed would be the bug this exists to stop,
			// applied to the one body that wants it.
			if !body.movable() || body.sleeping || !body.solid() {
				continue;
			}

			let Some(&was) = self.was.get(id.slot()) else {
				continue;
			};

			let extents = span(&body.shape, body.transform.scale);
			let reach = extents.min_element().max(SWEEP_SKIN);
			let moved = body.transform.position - was;

			if moved.length() < reach * SWEEP_AT {
				continue;
			}

			let info = TraceInfo::swept(was, body.transform.position, extents).ignoring(id);
			let result = query::swept(&world.bodies, self, &info);

			if !result.hit || result.started_solid {
				continue;
			}

			pulled.push((id, result.end + result.normal * SWEEP_SKIN));
		}

		for (id, place) in pulled {
			if let Some(body) = world.bodies.get_mut(id) {
				body.transform.position = place;
			}
		}
	}

	/// The baked collision mesh of a body, if it has one.
	///
	/// @param id - the body
	pub(crate) fn collider(&self, id: BodyId) -> Option<&Collider> {
		let slot = id.slot();

		if self.generations.get(slot) != Some(&id.generation()) {
			return None;
		}

		self.colliders.get(slot)?.as_ref()
	}

	/// Bakes what is new, drops what is gone.
	///
	/// @param world - the host state, for the body table and the mesh registry
	fn sync(&mut self, world: &World) {
		let slots = world.bodies.slots();

		self.colliders.resize_with(slots, || None);
		self.generations.resize(slots, 0);
		self.live.clear();
		self.live.resize(slots, false);

		for (id, body) in world.bodies.iter() {
			let slot = id.slot();
			let generation = id.generation();

			self.live[slot] = true;

			if self.generations[slot] != generation {
				// the slot changed hands. Whatever was baked for it belongs to
				// a body that no longer exists.
				self.colliders[slot] = None;
				self.generations[slot] = generation;
			}

			if body.shape.kind != ShapeKind::Mesh {
				self.colliders[slot] = None;

				continue;
			}

			if self.colliders[slot].is_some() {
				continue;
			}

			let collider = bake(
				world
					.meshes
					.get(body.shape.mesh)
					.map(Entry::value),
			);

			trace!(slot, generation, triangles = collider.count(), "collision mesh baked");
			self.colliders[slot] = Some(collider);
		}

		for slot in 0..slots {
			if !self.live[slot] {
				self.colliders[slot] = None;
			}
		}
	}

	/// Copies the entity's transform into every body gameplay owns.
	///
	/// Static and kinematic bodies follow the thing they are bolted to; that is
	/// what "kinematic" means, and it is what makes a collider on an entity the
	/// game animates track it without being told. @ref
	/// [`Body::transform`](colby_core::abi::Body::transform).
	///
	/// @param world - the host state
	fn pull(world: &mut World) {
		let entities = &world.entities;

		for id in handles(&world.bodies) {
			let Some(body) = world.bodies.get_mut(id) else {
				continue;
			};

			if body.movable() {
				continue;
			}

			if let Some(&transform) = entities.transform(body.entity) {
				body.transform = transform;
			}
		}
	}

	/// Copies every solver-owned body's transform back into its entity.
	///
	/// @param world - the host state
	fn push(world: &mut World) {
		let bodies = &world.bodies;
		let entities = &mut world.entities;

		for (_, body) in bodies.iter() {
			if !body.movable() {
				continue;
			}

			if let Some(slot) = entities.transform_mut(body.entity) {
				*slot = body.transform;
			}
		}
	}
}

/// Turns a console variable's value into a pass count.
///
/// Anything that is not a sensible count - a negative, a fraction, something
/// enormous - becomes the default rather than an error: this is a knob, and a
/// knob that refuses a value in the middle of a frame has nowhere to say so.
///
/// @param asked - what the variable holds
fn asked_for(asked: f32) -> usize {
	let rounded = asked.round();

	if !rounded.is_finite() || rounded < 1.0 {
		return VELOCITY_PASSES;
	}

	// counted up to rather than cast down from. A float that says "twenty" is
	// only ever compared against integers here, so there is no truncation to
	// argue about and no lint to silence.
	(1..=MAX_PASSES)
		.take_while(|&count| f32::from(u16::try_from(count).unwrap_or(u16::MAX)) <= rounded)
		.last()
		.unwrap_or(VELOCITY_PASSES)
}

/// Which of a pair is the sensor, and what is inside it.
///
/// `None` unless one of them is one, which is every ordinary contact.
///
/// @param bodies - the table, to look both handles up in
/// @param pair - the two handles, in slot order
fn overlapping(bodies: &Bodies, pair: (BodyId, BodyId)) -> Option<Overlap> {
	let first = bodies.get(pair.0)?;

	if !first.solid() {
		return Some(Overlap { sensor: pair.0, body: pair.1 });
	}

	if !bodies.get(pair.1)?.solid() {
		return Some(Overlap { sensor: pair.1, body: pair.0 });
	}

	None
}

/// A pair of handles in a fixed order, so that one pair is one key.
fn ordered(first: BodyId, second: BodyId) -> (BodyId, BodyId) {
	if first.slot() <= second.slot() {
		(first, second)
	} else {
		(second, first)
	}
}

/// A shape's half-extents in world space.
///
/// A mesh has none this can know, and a mesh is never swept: it is never
/// dynamic. @ref [`Body::movable`](colby_core::abi::Body::movable).
fn span(shape: &Shape, scale: Vec3) -> Vec3 { shape.local_extents().abs() * scale.abs() }

/// Every live body handle, collected so the table can then be written through.
///
/// @param bodies - the table to walk
fn handles(bodies: &Bodies) -> Vec<BodyId> { bodies.iter().map(|(id, _)| id).collect() }

/// Turns a mesh into triangles in its own space.
///
/// @param mesh - the registry entry, or `None` if the handle resolved to
/// nothing
fn bake(mesh: Option<&MeshData>) -> Collider {
	let Some(mesh) = mesh else {
		return Collider::new(Vec::new());
	};

	let mut triangles = Vec::with_capacity(mesh.triangles());

	for corners in mesh.indices.chunks_exact(3) {
		let mut corner = [Vec3::ZERO; 3];
		let mut whole = true;

		for (slot, &index) in corner.iter_mut().zip(corners) {
			let Some(vertex) = usize::try_from(index)
				.ok()
				.and_then(|it| mesh.vertices.get(it))
			else {
				whole = false;

				break;
			};

			*slot = Vec3::from_array(vertex.position);
		}

		if whole {
			triangles.push(corner);
		}
	}

	Collider::new(triangles)
}

/// Answers a ray trace. One half of [`Simulation::table`].
///
/// # Safety
///
/// `context` must be the address of a live [`Simulation`], and `bodies` and
/// `info` must point at live values nobody is mutating for the duration of the
/// call. [`World::trace_ray`] guarantees all three.
unsafe extern "C-unwind" fn trace_ray(
	context: *mut c_void,
	bodies: *const Bodies,
	info: *const TraceInfo,
) -> TraceResult {
	// SAFETY: the caller's obligation, above. All three are taken as shared
	// references: a query writes nothing, so nothing here can alias mutably.
	let (simulation, bodies, info) = unsafe { borrow(context, bodies, info) };

	query::ray(bodies, simulation, info)
}

/// Answers a swept-box trace. The other half.
///
/// # Safety
///
/// As [`trace_ray`].
unsafe extern "C-unwind" fn trace_box(
	context: *mut c_void,
	bodies: *const Bodies,
	info: *const TraceInfo,
) -> TraceResult {
	// SAFETY: as `trace_ray`.
	let (simulation, bodies, info) = unsafe { borrow(context, bodies, info) };

	query::swept(bodies, simulation, info)
}

/// Turns the three raw arguments of a [`TraceFn`](colby_core::abi::TraceFn)
/// into references.
///
/// One place rather than two, so that the reasoning about them is written down
/// once.
///
/// # Safety
///
/// `context` must be the address of a live [`Simulation`]; `bodies` and `info`
/// must be live and not mutably borrowed elsewhere for the lifetime of the
/// returned references.
unsafe fn borrow<'a>(
	context: *mut c_void,
	bodies: *const Bodies,
	info: *const TraceInfo,
) -> (&'a Simulation, &'a Bodies, &'a TraceInfo) {
	// SAFETY: the caller's obligation, and all three are shared references: a
	// query writes nothing, so nothing taken here can alias mutably.
	let simulation = unsafe { &*context.cast::<Simulation>() };
	// SAFETY: as above.
	let bodies = unsafe { &*bodies };
	// SAFETY: as above.
	let info = unsafe { &*info };

	(simulation, bodies, info)
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{Body, BodyKind, EntityId, Joint, MeshId, Motion, Moved, Transform, character},
		glam::{Quat, Vec2},
		time::STEP_SECONDS,
	};

	use super::*;

	/// Runs the simulation for a while.
	///
	/// @param world - the world to advance
	/// @param simulation - the solver
	/// @param steps - how many, at the world's own step length
	fn settle(world: &mut World, simulation: &mut Simulation, steps: usize) {
		for _ in 0..steps {
			simulation.step(world);
			// as `step::run` does, so that a test sees one step's events rather
			// than every step's.
			world.bodies.end_step();
		}
	}

	/// A wide, thick static slab with its surface at y = 0.
	fn ground(world: &mut World) -> BodyId {
		world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::cuboid(Vec3::new(20.0, 0.5, 20.0)),
			Transform::at(Vec3::new(0.0, -0.5, 0.0)),
		))
	}

	/// Where a body is now.
	fn placed(world: &World, id: BodyId) -> Vec3 {
		world
			.bodies
			.get(id)
			.expect("the body is alive")
			.transform
			.position
	}

	/// A world with the simulation installed, and the box that keeps the
	/// simulation still.
	fn wired() -> (Box<World>, Box<Simulation>) {
		let simulation = Box::new(Simulation::new());
		let mut world = Box::new(World::new());
		world.install_physics(simulation.table());

		(world, simulation)
	}

	#[test]
	fn a_rope_stops_a_falling_body_at_its_length() {
		let (mut world, mut simulation) = wired();
		let hook = Vec3::new(0.0, 5.0, 0.0);
		let hanging = world.bodies.spawn(Body::dynamic(
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 4.6, 0.0)),
			1.0,
		));

		world.join(Joint::rope(hanging, BodyId::NONE, (Vec3::ZERO, hook), 2.0));
		settle(&mut world, &mut simulation, 600);

		let resting = placed(&world, hanging);

		assert!(
			(hook.y - resting.y - 2.0).abs() < 0.05,
			"it hangs two below the hook, got {}",
			resting.y
		);
		assert!(
			Vec2::new(resting.x, resting.z).length() < 0.05,
			"and straight below it rather than swinging forever, got {resting}"
		);
	}

	#[test]
	fn a_rope_does_nothing_at_all_while_it_is_slack() {
		// against a second world with no rope in it rather than against a
		// formula: "falls freely" here means a dozen steps of semi-implicit
		// Euler with damping, which is not the number out of the textbook and
		// is exactly the number the other world produces.
		let fallen = |roped: bool| {
			let (mut world, mut simulation) = wired();
			let hanging = world.bodies.spawn(Body::dynamic(
				Shape::UNIT,
				Transform::at(Vec3::new(0.0, 4.9, 0.0)),
				1.0,
			));

			if roped {
				world.join(Joint::rope(
					hanging,
					BodyId::NONE,
					(Vec3::ZERO, Vec3::new(0.0, 5.0, 0.0)),
					4.0,
				));
			}

			settle(&mut world, &mut simulation, 12);

			placed(&world, hanging).y
		};

		let free = fallen(false);
		let slack = fallen(true);

		assert!(free < 4.89, "it did fall, to {free}");
		assert!(
			(slack - free).abs() < 1.0e-5,
			"a slack rope is not there at all: {slack} against {free}"
		);
	}

	#[test]
	fn breaking_a_rope_lets_go() {
		let (mut world, mut simulation) = wired();
		let hook = Vec3::new(0.0, 5.0, 0.0);
		let hanging = world.bodies.spawn(Body::dynamic(
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 4.6, 0.0)),
			1.0,
		));
		let rope = world.join(Joint::rope(hanging, BodyId::NONE, (Vec3::ZERO, hook), 2.0));

		settle(&mut world, &mut simulation, 300);
		let held = placed(&world, hanging).y;

		assert!(world.joints.despawn(rope), "the handle resolved");
		settle(&mut world, &mut simulation, 60);

		assert!(
			placed(&world, hanging).y < held - 0.3,
			"and then it fell, from {held} to {}",
			placed(&world, hanging).y
		);
	}

	#[test]
	fn a_weld_carries_a_body_that_has_nothing_under_it() {
		let (mut world, mut simulation) = wired();
		let post = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 4.0, 0.0)),
		));
		let held = world.bodies.spawn(Body::dynamic(
			Shape::UNIT,
			Transform::at(Vec3::new(1.0, 4.0, 0.0)),
			1.0,
		));

		world.join(Joint::weld(
			held,
			post,
			(Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0)),
		));
		settle(&mut world, &mut simulation, 600);

		let after = *world.bodies.get(held).expect("alive");

		assert!(
			after
				.transform
				.position
				.abs_diff_eq(Vec3::new(1.0, 4.0, 0.0), 0.05),
			"a weld holds it exactly where it was, got {}",
			after.transform.position
		);
		assert!(
			after
				.transform
				.rotation
				.angle_between(Quat::IDENTITY)
				< 0.05,
			"and holds the angle too rather than letting it sag"
		);
	}

	#[test]
	fn a_weld_keeps_whatever_angle_it_was_made_at() {
		let (mut world, mut simulation) = wired();
		let post = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 4.0, 0.0)),
		));

		// placed so that the two anchors already coincide: a weld made across a
		// gap is a weld that yanks, and what is being asked here is about the
		// *angle*.
		let turn = Quat::from_rotation_z(0.6);
		let local = Vec3::new(-0.6, 0.0, 0.0);
		let meeting = Vec3::new(0.6, 4.0, 0.0);

		let mut tilted = Transform::at(meeting - turn * local);
		tilted.rotation = turn;
		let held = world
			.bodies
			.spawn(Body::dynamic(Shape::UNIT, tilted, 1.0));

		world.join(Joint::weld(held, post, (local, Vec3::new(0.6, 0.0, 0.0))));
		settle(&mut world, &mut simulation, 600);

		let sag = |world: &World| {
			world
				.bodies
				.get(held)
				.expect("alive")
				.transform
				.rotation
				.angle_between(turn)
		};
		let early = sag(&world);
		settle(&mut world, &mut simulation, 600);
		let late = sag(&world);

		assert!(
			(late - early).abs() < 0.01,
			"the sag has to be *steady*, or the weld is slowly letting go: {early} then {late}"
		);

		let turned = world
			.bodies
			.get(held)
			.expect("alive")
			.transform
			.rotation;

		assert!(
			turned.angle_between(turn) < 0.1,
			"welding two things at an angle keeps the angle rather than squaring them up: out 			 by {}",
			turned.angle_between(turn)
		);
	}

	#[test]
	fn a_hinge_turns_about_its_axis_and_holds_every_other_way() {
		let spun = |axis: Vec3| {
			let (mut world, mut simulation) = wired();
			world.gravity = Vec3::ZERO;
			let swinging = world.bodies.spawn(
				Body::dynamic(Shape::UNIT, Transform::at(Vec3::new(1.0, 0.0, 0.0)), 1.0)
					.moving(Vec3::ZERO, axis * 3.0),
			);

			world.join(Joint::axis(
				swinging,
				BodyId::NONE,
				(Vec3::new(-1.0, 0.0, 0.0), Vec3::ZERO),
				Vec3::Y,
			));
			settle(&mut world, &mut simulation, 90);

			world
				.bodies
				.get(swinging)
				.expect("alive")
				.transform
				.rotation
				.angle_between(Quat::IDENTITY)
		};

		assert!(spun(Vec3::Y) > 0.5, "about its own axis it turns freely, got {}", spun(Vec3::Y));
		assert!(spun(Vec3::X) < 0.1, "and about anything else it is held, got {}", spun(Vec3::X));
	}

	#[test]
	fn something_fast_does_not_pass_through_a_wall() {
		let (mut world, mut simulation) = wired();
		world.gravity = Vec3::ZERO;

		// a wall a tenth of a unit thick, and a bullet a tenth of a unit wide
		// moving forty units a second - two thirds of a unit per step,
		// which without a sweep puts it clean through with nothing in between.
		world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::cuboid(Vec3::new(4.0, 4.0, 0.05)),
			Transform::at(Vec3::ZERO),
		));
		let bullet = world.bodies.spawn(
			Body::dynamic(
				Shape::cuboid(Vec3::splat(0.05)),
				Transform::at(Vec3::new(0.0, 0.0, 5.0)),
				1.0,
			)
			.moving(Vec3::new(0.0, 0.0, -40.0), Vec3::ZERO),
		);

		settle(&mut world, &mut simulation, 60);

		assert!(
			placed(&world, bullet).z > 0.0,
			"it should be stopped on the near side of the wall, got {}",
			placed(&world, bullet).z
		);
	}

	#[test]
	fn landing_and_leaving_are_both_reported() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);
		let ball = world.bodies.spawn(Body::dynamic(
			Shape::ball(0.5),
			Transform::at(Vec3::new(0.0, 2.0, 0.0)),
			1.0,
		));

		let mut began = 0;
		let mut ended = 0;

		for _ in 0..300 {
			simulation.step(&mut world);

			for touch in world
				.bodies
				.touches()
				.iter()
				.filter(|touch| touch.names(ball))
			{
				match touch.kind {
					| TouchKind::Began => began += 1,
					| TouchKind::Ended => ended += 1,
				}
			}

			world.bodies.end_step();
		}

		assert!(began >= 1, "it landed at least once");
		assert!(
			began <= 4,
			"and a ball with almost no bounce should not have landed {began} times"
		);
		assert_eq!(began - ended, 1, "and it is still down: one more landing than leaving");
	}

	#[test]
	fn a_settled_pile_does_not_announce_its_own_collapse() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);
		let lower = world.bodies.spawn(Body::dynamic(
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 0.5, 0.0)),
			1.0,
		));
		world.bodies.spawn(Body::dynamic(
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 1.5, 0.0)),
			1.0,
		));

		// long enough that the pile is certainly asleep.
		settle(&mut world, &mut simulation, 400);

		let mut ended = 0;
		for _ in 0..120 {
			simulation.step(&mut world);
			ended += world
				.bodies
				.touches()
				.iter()
				.filter(|touch| touch.kind == TouchKind::Ended)
				.count();
			world.bodies.end_step();
		}

		assert!(world.bodies.get(lower).expect("alive").sleeping, "the pile really did settle");
		assert_eq!(ended, 0, "and going quiet is not the same as coming apart");
	}

	#[test]
	fn a_prop_falls_through_a_sensor_and_says_so() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);
		let volume = world.bodies.spawn(
			Body::new(
				BodyKind::Static,
				Shape::cuboid(Vec3::new(1.0, 0.5, 1.0)),
				Transform::at(Vec3::new(0.0, 2.0, 0.0)),
			)
			.sensing(),
		);
		let prop = world.bodies.spawn(Body::dynamic(
			Shape::ball(0.25),
			Transform::at(Vec3::new(0.0, 4.0, 0.0)),
			1.0,
		));

		let mut began = 0;
		let mut ended = 0;

		for _ in 0..240 {
			simulation.step(&mut world);

			for touch in world
				.bodies
				.touches()
				.iter()
				.filter(|touch| touch.names(volume))
			{
				match touch.kind {
					| TouchKind::Began => began += 1,
					| TouchKind::Ended => ended += 1,
				}
			}

			world.bodies.end_step();
		}

		assert_eq!((began, ended), (1, 1), "it went in once and came out once");
		assert!(
			placed(&world, prop).y < 0.4,
			"and it is on the floor rather than resting on the trigger, at {}",
			placed(&world, prop).y
		);
		assert!(
			world.bodies.overlaps().is_empty(),
			"and the overlap list is what is true now rather than what ever was"
		);
	}

	#[test]
	fn a_sensor_wrapped_around_a_pile_leaves_it_exactly_where_it_was() {
		let settled = |sensor: bool| {
			let (mut world, mut simulation) = wired();
			ground(&mut world);
			let lower = world.bodies.spawn(Body::dynamic(
				Shape::UNIT,
				Transform::at(Vec3::new(0.0, 0.5, 0.0)),
				1.0,
			));
			world.bodies.spawn(Body::dynamic(
				Shape::UNIT,
				Transform::at(Vec3::new(0.0, 1.5, 0.0)),
				1.0,
			));

			if sensor {
				world.bodies.spawn(
					Body::new(
						BodyKind::Static,
						Shape::cuboid(Vec3::splat(3.0)),
						Transform::at(Vec3::new(0.0, 1.0, 0.0)),
					)
					.sensing(),
				);
			}

			settle(&mut world, &mut simulation, 200);

			placed(&world, lower)
		};

		assert!(
			settled(true).abs_diff_eq(settled(false), 0.0),
			"a trigger wrapped around a pile is not a thing the pile can feel, and the 			 \
			 tolerance here is zero on purpose"
		);
	}

	#[test]
	fn a_sensor_is_not_there_as_far_as_a_trace_is_concerned() {
		let (mut world, _simulation) = wired();
		let volume = world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY).sensing());
		let aim = TraceInfo::ray(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -5.0));
		let sweep = TraceInfo::swept(
			Vec3::new(0.0, 0.0, 5.0),
			Vec3::new(0.0, 0.0, -5.0),
			Vec3::splat(0.1),
		);

		assert!(!world.trace_ray(&aim).hit, "a ray goes straight through it");
		assert!(!world.trace_box(&sweep).hit, "and so does a swept box");

		world
			.bodies
			.get_mut(volume)
			.expect("alive")
			.sensor = false;

		assert!(world.trace_ray(&aim).hit, "the same box, made solid, stops the ray");
		assert!(world.trace_box(&sweep).hit, "and the sweep");
	}

	#[test]
	fn a_static_body_carried_through_a_static_sensor_is_noticed() {
		let (mut world, mut simulation) = wired();
		let volume = world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY).sensing());
		let carried = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::UNIT,
			Transform::at(Vec3::new(4.0, 0.0, 0.0)),
		));

		simulation.step(&mut world);
		world.bodies.end_step();

		assert_eq!(world.bodies.inside(volume).count(), 0, "nothing is in it yet");

		world
			.bodies
			.get_mut(carried)
			.expect("alive")
			.transform
			.position = Vec3::ZERO;
		simulation.step(&mut world);

		assert_eq!(
			world.bodies.inside(volume).next(),
			Some(carried),
			"two bodies the solver never moves still meet when a game moves one"
		);
		assert!(
			world
				.bodies
				.touches()
				.iter()
				.any(|touch| touch.kind == TouchKind::Began && touch.names(volume)),
			"and the arrival is an edge as well as a state"
		);
	}

	#[test]
	fn a_trigger_hears_about_a_prop_deleted_inside_it() {
		let (mut world, mut simulation) = wired();
		let volume = world.bodies.spawn(
			Body::new(BodyKind::Static, Shape::cuboid(Vec3::splat(1.0)), Transform::IDENTITY)
				.sensing(),
		);
		let prop = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::ball(0.25),
			Transform::IDENTITY,
		));

		simulation.step(&mut world);

		assert_eq!(world.bodies.inside(volume).next(), Some(prop), "it is in there");

		world.bodies.end_step();
		world.bodies.despawn(prop);
		simulation.step(&mut world);

		assert!(
			world
				.bodies
				.touches()
				.iter()
				.any(|touch| touch.kind == TouchKind::Ended && touch.names(volume)),
			"a prop deleted inside a trigger still leaves it"
		);
		assert_eq!(world.bodies.inside(volume).count(), 0, "and the list agrees");
	}

	#[test]
	fn two_sensors_have_nothing_to_say_about_each_other() {
		let (mut world, mut simulation) = wired();
		let first = world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY).sensing());
		let second = world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY).sensing());

		simulation.step(&mut world);

		assert!(world.bodies.touches().is_empty(), "neither pushes, so neither notices");
		assert_eq!(world.bodies.inside(first).count(), 0, "and neither holds the other");
		assert_eq!(world.bodies.inside(second).count(), 0, "in either order");
	}

	#[test]
	fn a_ball_dropped_on_the_floor_comes_to_rest_on_it() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);
		let ball = world.bodies.spawn(Body::dynamic(
			Shape::ball(0.5),
			Transform::at(Vec3::new(0.0, 3.0, 0.0)),
			1.0,
		));

		settle(&mut world, &mut simulation, 240);
		let resting = placed(&world, ball);

		assert!(
			(resting.y - 0.5).abs() < 0.02,
			"a ball of radius a half rests with its middle a half up, got {}",
			resting.y
		);
		assert!(
			Vec2::new(resting.x, resting.z).length() < 0.02,
			"and it does not wander sideways, got {resting}"
		);
	}

	#[test]
	fn what_has_come_to_rest_goes_to_sleep() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);
		let ball = world.bodies.spawn(Body::dynamic(
			Shape::ball(0.5),
			Transform::at(Vec3::new(0.0, 1.5, 0.0)),
			1.0,
		));

		settle(&mut world, &mut simulation, 240);

		assert!(
			world.bodies.get(ball).expect("alive").sleeping,
			"or every settled prop goes on costing a solve forever"
		);
	}

	#[test]
	fn a_box_dropped_flat_lands_flat() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);
		let brick = world.bodies.spawn(Body::dynamic(
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 2.0, 0.0)),
			1.0,
		));

		settle(&mut world, &mut simulation, 240);
		let body = *world.bodies.get(brick).expect("alive");

		assert!(
			(body.transform.position.y - 0.5).abs() < 0.02,
			"resting on its face, got {}",
			body.transform.position.y
		);
		assert!(
			body.transform
				.rotation
				.angle_between(Quat::IDENTITY)
				< 0.05,
			"and still square to the floor rather than tipped: four contact points are what \
			 stop a box rocking, and one would not"
		);
	}

	#[test]
	fn a_stack_of_three_boxes_stays_a_stack() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);

		let mut stack = Vec::new();
		for level in 0..3_u8 {
			let height = f32::from(level).mul_add(1.02, 0.5);

			stack.push(world.bodies.spawn(Body::dynamic(
				Shape::UNIT,
				Transform::at(Vec3::new(0.0, height, 0.0)),
				1.0,
			)));
		}

		settle(&mut world, &mut simulation, 360);

		for (level, &id) in stack.iter().enumerate() {
			let wanted = 0.5 + f32::from(u8::try_from(level).unwrap_or(0));
			let found = placed(&world, id);

			assert!(
				(found.y - wanted).abs() < 0.05,
				"level {level} should sit at {wanted}, got {}",
				found.y
			);
			assert!(
				Vec2::new(found.x, found.z).length() < 0.08,
				"and should not have slid out from under the one above, got {found}"
			);
		}

		assert!(
			stack
				.iter()
				.all(|&id| world.bodies.get(id).expect("alive").sleeping),
			"and the whole pile should be asleep by now"
		);
	}

	#[test]
	fn a_pile_settles_even_beside_something_that_is_already_asleep() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);

		// spawned first, so its contact with the floor is the *earlier*
		// manifold and the pile's are the later ones. Once it is asleep the
		// solver skips its pair - a wall against a sleeper needs no impulse -
		// and everything after it shifts by one. That is the shape of the bug
		// this is here for: the contact impulses each pile member started the
		// step with were the ones belonging to its neighbor, which reads as a
		// stack that will not settle and creeps a tenth of a unit a second.
		let quiet = world.bodies.spawn(Body::dynamic(
			Shape::ball(0.5),
			Transform::at(Vec3::new(6.0, 0.5, 0.0)),
			1.0,
		));

		let mut stack = Vec::new();
		for level in 0..3_u8 {
			let mut transform =
				Transform::at(Vec3::new(0.0, f32::from(level).mul_add(0.9, 0.5), 0.0));
			transform.set_scale(0.62);
			stack.push(
				world
					.bodies
					.spawn(Body::dynamic(Shape::UNIT, transform, 1.0)),
			);
		}

		settle(&mut world, &mut simulation, 600);

		assert!(
			world.bodies.get(quiet).expect("alive").sleeping,
			"the ball beside the pile settled long ago"
		);

		for (level, &id) in stack.iter().enumerate() {
			let body = *world.bodies.get(id).expect("alive");
			let drift = Vec2::new(body.transform.position.x, body.transform.position.z).length();

			assert!(
				body.sleeping,
				"level {level} never stopped moving: speed {}, spin {}",
				body.velocity.length(),
				body.angular.length()
			);
			assert!(drift < 0.08, "and level {level} crept {drift} sideways on the way");
		}
	}

	#[test]
	fn a_taller_stack_of_smaller_boxes_stays_a_stack() {
		const LEVELS: u8 = 8;

		let (mut world, mut simulation) = wired();
		ground(&mut world);

		// five, and smaller than a unit, because both of those make the pile
		// harder: five is a longer chain for the impulses to travel, and a
		// smaller box turns more readily for the same contact error.
		let mut stack = Vec::new();
		for level in 0..LEVELS {
			let mut transform =
				Transform::at(Vec3::new(0.0, f32::from(level).mul_add(0.62, 0.31), 0.0));
			transform.set_scale(0.62);
			stack.push(
				world
					.bodies
					.spawn(Body::dynamic(Shape::UNIT, transform, 1.0)),
			);
		}

		settle(&mut world, &mut simulation, 900);

		for (level, &id) in stack.iter().enumerate() {
			let body = *world.bodies.get(id).expect("alive");
			let wanted = f32::from(u8::try_from(level).unwrap_or(0)).mul_add(0.62, 0.31);
			let found = body.transform.position;

			assert!(
				(found.y - wanted).abs() < 0.06,
				"level {level} should sit at {wanted}, got {}",
				found.y
			);
			assert!(
				Vec2::new(found.x, found.z).length() < 0.15,
				"and should still be over the one below it, got {found}"
			);
			assert!(body.sleeping, "and the pile should have gone quiet by now");
		}
	}

	#[test]
	fn a_tall_pile_settles_on_four_passes() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);

		// four, against the ten the solver ships with, and the pile is the
		// tallest it can carry. Before a manifold's contacts were answered
		// together rather than in turn, eight boxes needed twenty passes and
		// were a coin toss even then. This is the test that says so: the
		// eight-box test above runs at whatever the default is, and this one
		// pins how little it now takes.
		world.cvars.var(
			PASSES,
			colby_core::abi::Value::Float(4.0),
			"how hard the solver tries, for this test only",
		);

		let mut stack = Vec::new();
		for level in 0..8_u8 {
			let mut transform =
				Transform::at(Vec3::new(0.0, f32::from(level).mul_add(0.62, 0.31), 0.0));
			transform.set_scale(0.62);
			stack.push(
				world
					.bodies
					.spawn(Body::dynamic(Shape::UNIT, transform, 1.0)),
			);
		}

		settle(&mut world, &mut simulation, 900);

		for (level, &id) in stack.iter().enumerate() {
			let found = placed(&world, id);
			let wanted = f32::from(u8::try_from(level).unwrap_or(0)).mul_add(0.62, 0.31);

			assert!(
				(found.y - wanted).abs() < 0.06,
				"level {level} should sit at {wanted}, got {}",
				found.y
			);
			assert!(
				Vec2::new(found.x, found.z).length() < 0.15,
				"and should still be over the one below it, got {found}"
			);
		}
	}

	#[test]
	fn each_triangle_of_a_mesh_is_a_contact_of_its_own() {
		let (mut world, mut simulation) = wired();
		// the built-in quad is two triangles, and a box in the middle of it
		// straddles both. Both manifolds are between the same pair of bodies,
		// so anything keeping notes on a pair has to tell them apart by
		// something other than the pair.
		world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::mesh(MeshId::QUAD), Transform {
				position: Vec3::ZERO,
				rotation: Quat::IDENTITY,
				scale: Vec3::new(4.0, 1.0, 4.0),
			}));
		let brick = world.bodies.spawn(Body::dynamic(
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 0.9, 0.0)),
			1.0,
		));

		settle(&mut world, &mut simulation, 300);

		let resting = placed(&world, brick);

		assert!(
			(resting.y - 0.5).abs() < 0.03,
			"a unit box on a quad rests with its middle a half up, got {}",
			resting.y
		);
		assert!(
			Vec2::new(resting.x, resting.z).length() < 0.05,
			"and it does not slide off the seam between the two triangles, got {resting}"
		);
	}

	#[test]
	fn nothing_falls_through_a_collision_mesh() {
		let (mut world, mut simulation) = wired();
		world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::mesh(MeshId::QUAD), Transform {
				position: Vec3::ZERO,
				rotation: Quat::IDENTITY,
				scale: Vec3::new(20.0, 1.0, 20.0),
			}));
		let ball = world.bodies.spawn(Body::dynamic(
			Shape::ball(0.5),
			Transform::at(Vec3::new(0.0, 3.0, 0.0)),
			1.0,
		));

		settle(&mut world, &mut simulation, 240);

		assert!(
			(placed(&world, ball).y - 0.5).abs() < 0.03,
			"the quad's own triangles hold it up, got {}",
			placed(&world, ball).y
		);
	}

	#[test]
	fn a_static_body_ignores_gravity_however_long_it_is_left() {
		let (mut world, mut simulation) = wired();
		let post = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 6.0, 0.0)),
		));

		settle(&mut world, &mut simulation, 120);

		assert!(
			(placed(&world, post).y - 6.0).abs() < 1.0e-4,
			"static means static, got {}",
			placed(&world, post).y
		);
	}

	#[test]
	fn gravity_is_the_worlds_and_a_game_can_turn_it_off() {
		let (mut world, mut simulation) = wired();
		world.gravity = Vec3::ZERO;
		let floating = world.bodies.spawn(Body::dynamic(
			Shape::ball(0.5),
			Transform::at(Vec3::new(0.0, 6.0, 0.0)),
			1.0,
		));

		settle(&mut world, &mut simulation, 120);

		assert!(
			(placed(&world, floating).y - 6.0).abs() < 1.0e-3,
			"nothing pulls on it, got {}",
			placed(&world, floating).y
		);
	}

	#[test]
	fn a_bouncy_ball_comes_back_up_and_a_dead_one_does_not() {
		let peak = |restitution: f32| {
			let (mut world, mut simulation) = wired();
			ground(&mut world);
			let mut ball =
				Body::dynamic(Shape::ball(0.5), Transform::at(Vec3::new(0.0, 4.0, 0.0)), 1.0);
			ball.restitution = restitution;
			let ball = world.bodies.spawn(ball);

			let mut landed = false;
			let mut after = 0.0_f32;

			for _ in 0..180 {
				simulation.step(&mut world);
				let height = placed(&world, ball).y;

				landed |= height < 0.7;
				after = after.max(height * f32::from(u8::from(landed)));
			}

			after
		};

		let bouncy = peak(0.8);
		let dead = peak(0.0);

		assert!(bouncy > 1.5, "most of the drop comes back, got {bouncy}");
		assert!(dead < 0.7, "and none of it does when nothing is given back, got {dead}");
	}

	#[test]
	fn something_landing_on_a_sleeping_pile_wakes_it() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);
		let lower = world.bodies.spawn(Body::dynamic(
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 0.5, 0.0)),
			1.0,
		));

		settle(&mut world, &mut simulation, 180);
		assert!(world.bodies.get(lower).expect("alive").sleeping, "it settled");

		world.bodies.spawn(Body::dynamic(
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 4.0, 0.0)),
			1.0,
		));
		settle(&mut world, &mut simulation, 60);

		assert!(
			!world.bodies.get(lower).expect("alive").sleeping,
			"and then something landed on it"
		);
	}

	#[test]
	fn a_dynamic_body_drives_the_entity_it_is_attached_to_as_it_falls() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);
		let entity = world
			.entities
			.spawn_at(Transform::at(Vec3::new(0.0, 4.0, 0.0)));
		world.attach_body(entity, BodyKind::Dynamic, Shape::UNIT);

		settle(&mut world, &mut simulation, 240);

		let drawn = world
			.entities
			.transform(entity)
			.expect("alive")
			.position;

		assert!(
			(drawn.y - 0.5).abs() < 0.03,
			"what is drawn followed the body all the way down, got {drawn}"
		);
	}

	#[test]
	fn a_ray_down_the_z_axis_hits_the_near_face_of_a_box() {
		let (mut world, mut simulation) = wired();
		world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::at(Vec3::ZERO)));
		simulation.step(&mut world);

		let result =
			world.trace_ray(&TraceInfo::ray(Vec3::new(0.0, 0.0, 4.0), Vec3::new(0.0, 0.0, -4.0)));

		assert!(result.hit, "a unit cube at the origin is in the way");
		assert!(
			(result.fraction - 0.4375).abs() < 1.0e-4,
			"3.5 of the 8 units, got {}",
			result.fraction
		);
		assert!(
			result
				.end
				.abs_diff_eq(Vec3::new(0.0, 0.0, 0.5), 1.0e-4),
			"stopping on the face, got {}",
			result.end
		);
		assert!(
			result.normal.abs_diff_eq(Vec3::Z, 1.0e-4),
			"with the normal pointing back at where it came from, got {}",
			result.normal
		);
	}

	#[test]
	fn a_ray_that_misses_reports_the_whole_distance() {
		let (mut world, mut simulation) = wired();
		world.bodies.spawn(Body::default());
		simulation.step(&mut world);

		let result =
			world.trace_ray(&TraceInfo::ray(Vec3::new(4.0, 4.0, 4.0), Vec3::new(4.0, 4.0, -4.0)));

		assert!(!result.hit, "it goes past the corner");
		assert!((result.fraction - 1.0).abs() < f32::EPSILON, "all the way");
		assert_eq!(result.body, BodyId::NONE, "and hit nothing");
	}

	#[test]
	fn a_ray_misses_a_ball_by_a_hair_and_hits_it_by_the_same() {
		let (mut world, mut simulation) = wired();
		world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::ball(1.0),
			Transform::at(Vec3::ZERO),
		));
		simulation.step(&mut world);

		let outside = world
			.trace_ray(&TraceInfo::ray(Vec3::new(1.001, 0.0, 4.0), Vec3::new(1.001, 0.0, -4.0)));
		let inside = world
			.trace_ray(&TraceInfo::ray(Vec3::new(0.999, 0.0, 4.0), Vec3::new(0.999, 0.0, -4.0)));

		assert!(!outside.hit, "a thousandth outside the radius is outside");
		assert!(inside.hit, "and a thousandth inside is inside");
	}

	#[test]
	fn a_ray_that_starts_inside_says_so() {
		let (mut world, mut simulation) = wired();
		world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY));
		simulation.step(&mut world);

		let result = world.trace_ray(&TraceInfo::ray(Vec3::ZERO, Vec3::new(0.0, 0.0, -4.0)));

		assert!(result.started_solid, "the origin is inside a unit cube at the origin");
		assert!(!result.ended_solid, "and four units away is not");
	}

	#[test]
	fn a_ray_ignores_what_it_was_told_to() {
		let (mut world, mut simulation) = wired();
		let near = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 0.0, 1.0)),
		));
		let far = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 0.0, -1.0)),
		));
		simulation.step(&mut world);

		let trace = TraceInfo::ray(Vec3::new(0.0, 0.0, 6.0), Vec3::new(0.0, 0.0, -6.0));

		assert_eq!(world.trace_ray(&trace).body, near, "the near one is in front");
		assert_eq!(
			world.trace_ray(&trace.ignoring(near)).body,
			far,
			"and behind it is the far one"
		);
		assert!(
			!world
				.trace_ray(&trace.ignoring(near).ignoring(far))
				.hit,
			"with nothing behind that"
		);
	}

	#[test]
	fn a_rotated_box_is_hit_where_its_corner_now_is() {
		let (mut world, mut simulation) = wired();
		let mut transform = Transform::at(Vec3::ZERO);
		transform.rotation = Quat::from_rotation_y(core::f32::consts::FRAC_PI_4);
		world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::cuboid(Vec3::splat(0.5)),
			transform,
		));
		simulation.step(&mut world);

		let result =
			world.trace_ray(&TraceInfo::ray(Vec3::new(0.0, 0.0, 4.0), Vec3::new(0.0, 0.0, -4.0)));
		let corner = 0.5 * core::f32::consts::SQRT_2;

		assert!(result.hit, "the turned cube is still in the way");
		assert!(
			(result.end.z - corner).abs() < 1.0e-3,
			"and its corner now leads, at {corner} rather than 0.5, got {}",
			result.end.z
		);
	}

	#[test]
	fn a_scaled_box_is_hit_at_its_scaled_size() {
		let (mut world, mut simulation) = wired();
		let mut transform = Transform::at(Vec3::ZERO);
		transform.scale = Vec3::new(1.0, 1.0, 4.0);
		world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::UNIT, transform));
		simulation.step(&mut world);

		let result =
			world.trace_ray(&TraceInfo::ray(Vec3::new(0.0, 0.0, 6.0), Vec3::new(0.0, 0.0, -6.0)));

		assert!(
			(result.end.z - 2.0).abs() < 1.0e-4,
			"a unit box scaled four along z reaches z = 2, got {}",
			result.end.z
		);
	}

	#[test]
	fn a_mesh_body_is_traced_against_its_triangles() {
		let (mut world, mut simulation) = wired();
		world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::mesh(MeshId::CUBE),
			Transform::at(Vec3::ZERO),
		));
		simulation.step(&mut world);

		let result =
			world.trace_ray(&TraceInfo::ray(Vec3::new(0.0, 0.0, 4.0), Vec3::new(0.0, 0.0, -4.0)));

		assert!(result.hit, "the built-in cube's own triangles are the collision");
		assert!(
			result
				.end
				.abs_diff_eq(Vec3::new(0.0, 0.0, 0.5), 1.0e-4),
			"and they are a unit cube, got {}",
			result.end
		);
		assert!(
			result.normal.abs_diff_eq(Vec3::Z, 1.0e-4),
			"with a triangle normal turned to face the ray, got {}",
			result.normal
		);
	}

	#[test]
	fn a_swept_box_stops_a_half_extent_short_of_what_a_ray_would_reach() {
		let (mut world, mut simulation) = wired();
		world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY));
		simulation.step(&mut world);

		let sweep = world.trace_box(&TraceInfo::swept(
			Vec3::new(0.0, 0.0, 4.0),
			Vec3::new(0.0, 0.0, -4.0),
			Vec3::splat(0.25),
		));

		assert!(sweep.hit, "the box runs into the cube");
		assert!(
			(sweep.end.z - 0.75).abs() < 1.0e-4,
			"its middle stops a quarter short of the face, got {}",
			sweep.end.z
		);
	}

	/// A unit cube turned an eighth of a turn about y.
	///
	/// The shape every sweep test below is aimed past: its bounds are a seventh
	/// wider than it is, and everything in that margin is a place the old
	/// bounds-only sweep reported contact and there was none.
	fn turned() -> Transform {
		let mut transform = Transform::IDENTITY;
		transform.rotation = Quat::from_rotation_y(core::f32::consts::FRAC_PI_4);

		transform
	}

	#[test]
	fn a_swept_box_stops_at_a_turned_box_and_not_at_its_bounds() {
		let (mut world, mut simulation) = wired();
		world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::UNIT, turned()));
		simulation.step(&mut world);

		let sweep = world.trace_box(&TraceInfo::swept(
			Vec3::new(0.68, 0.0, 4.0),
			Vec3::new(0.68, 0.0, -4.0),
			Vec3::splat(0.05),
		));

		// the turned cube is a diamond in xz whose near edge is x + z = 0.7071,
		// and the cast box reaches it with its low corner at x = 0.63. So the
		// face it stops on is z = 0.0771 and its middle is a half-extent short
		// of that. The bounds would have said 0.757, which is where the corner
		// of a box that is not there would be.
		assert!(sweep.hit, "it does run into it");
		assert!(
			(sweep.end.z - 0.127).abs() < 0.01,
			"it stops where the face is and not where the bounds are, got {}",
			sweep.end.z
		);
	}

	#[test]
	fn a_cube_answers_a_sweep_the_same_as_its_own_triangles() {
		let swept = |shape: Shape| {
			let (mut world, mut simulation) = wired();
			world
				.bodies
				.spawn(Body::new(BodyKind::Static, shape, turned()));
			simulation.step(&mut world);

			world.trace_box(&TraceInfo::swept(
				Vec3::new(0.68, 0.0, 4.0),
				Vec3::new(0.68, 0.0, -4.0),
				Vec3::splat(0.05),
			))
		};

		let solid = swept(Shape::UNIT);
		let soup = swept(Shape::mesh(MeshId::CUBE));

		assert!(solid.hit && soup.hit, "both are in the way");
		assert!(
			(solid.fraction - soup.fraction).abs() < 0.01,
			"the same cube declared two ways stops the same sweep in the same place, got {} \
			 against {}",
			solid.fraction,
			soup.fraction
		);
	}

	#[test]
	fn a_swept_box_misses_the_corner_of_a_balls_bounds() {
		let (mut world, mut simulation) = wired();
		world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::ball(0.5), Transform::IDENTITY));
		simulation.step(&mut world);

		// a hair inside the ball's bounding box on two axes at once, which puts
		// the nearest corner of the cast box 0.69 from the middle of a ball of
		// half that radius. A box is not a ball, and this is the whole of the
		// difference.
		let sweep = world.trace_box(&TraceInfo::swept(
			Vec3::new(0.51, 0.51, 4.0),
			Vec3::new(0.51, 0.51, -4.0),
			Vec3::splat(0.02),
		));

		assert!(!sweep.hit, "it goes past the corner, stopping at {}", sweep.end);
		assert!((sweep.fraction - 1.0).abs() < f32::EPSILON, "having gone the whole way");
	}

	/// Half-extents of the box every controller test moves.
	///
	/// Half a unit tall and a quarter wide, which is a person shape without
	/// being a person: the point is that it is a box and the arithmetic knows
	/// it.
	const WALKER: Vec3 = Vec3::new(0.25, 0.5, 0.25);

	/// One step of moving a box, with gravity applied the way a game would.
	///
	/// @param world - the bodies to move through
	/// @param place - where it is and how fast, updated in place
	/// @param step - how tall a lip it may climb
	fn walk(world: &World, place: &mut (Vec3, Vec3), step: f32) -> Moved {
		place.1.y = 9.81_f32.mul_add(-STEP_SECONDS, place.1.y);

		let motion = Motion::new(place.0, place.1, WALKER, STEP_SECONDS).stepping(step);
		let moved = character::move_and_slide(world, &motion);

		*place = (moved.position, moved.velocity);

		moved
	}

	#[test]
	fn a_box_walking_into_a_wall_slides_along_it() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);
		world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::cuboid(Vec3::new(0.1, 2.0, 5.0)),
			Transform::at(Vec3::new(2.0, 1.0, 0.0)),
		));
		simulation.step(&mut world);

		let motion = Motion::new(
			Vec3::new(1.5, 0.5, 0.0),
			Vec3::new(30.0, 0.0, 30.0),
			WALKER,
			STEP_SECONDS,
		);
		let moved = character::move_and_slide(&world, &motion);

		// the wall's face is at x = 1.9 and the box is a quarter wide, so it can
		// get to 1.65 and no further. What it was going to spend crossing the
		// wall it spends going along it instead, which is the whole of sliding:
		// the half unit of z happens in full.
		assert_eq!(moved.slides, 1, "it met one surface");
		assert!(
			(moved.position.x - 1.65).abs() < 0.02,
			"and stopped against it, at {}",
			moved.position.x
		);
		assert!(
			(moved.position.z - 0.5).abs() < 0.02,
			"having gone the whole way along it, got {}",
			moved.position.z
		);
		assert!(
			moved.velocity.x.abs() < 1.0e-3,
			"the speed into the wall is gone, not stored up, got {}",
			moved.velocity.x
		);
		assert!((moved.velocity.z - 30.0).abs() < 1.0e-3, "and the speed along it is untouched");
	}

	#[test]
	fn a_box_standing_on_the_floor_does_not_drift() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);
		simulation.step(&mut world);

		let mut place = (Vec3::new(0.0, 0.5, 0.0), Vec3::ZERO);
		let mut lowest = f32::INFINITY;
		let mut highest = f32::NEG_INFINITY;

		for _ in 0..180 {
			let moved = walk(&world, &mut place, character::STEP);

			assert!(moved.grounded, "it is on the floor and stays on it");
			lowest = lowest.min(moved.position.y);
			highest = highest.max(moved.position.y);
		}

		// three seconds of standing still. A skin gap that were added rather than
		// converged to would be four centimeters of climb by now.
		assert!(
			highest - lowest < 0.01,
			"it neither sinks nor creeps upwards: {lowest} to {highest}"
		);
	}

	#[test]
	fn a_box_climbs_a_lip_no_taller_than_its_step() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);
		world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::cuboid(Vec3::new(0.5, 0.1, 2.0)),
			Transform::at(Vec3::new(2.0, 0.1, 0.0)),
		));
		simulation.step(&mut world);

		let mut place = (Vec3::new(1.0, 0.5, 0.0), Vec3::ZERO);
		let mut climbed = false;
		let mut highest = f32::NEG_INFINITY;

		// far enough to go up one side of the lip and off the other, which is
		// two answers for the price of one: it climbed, and it came down again
		// rather than walking through the air where the lip used to be.
		for _ in 0..40 {
			place.1.x = 5.0;
			let moved = walk(&world, &mut place, character::STEP);
			climbed |= moved.stepped;
			highest = highest.max(moved.position.y);
		}

		assert!(climbed, "it stepped up at some point");
		assert!(
			(highest - 0.7).abs() < 0.02,
			"getting as high as the top of the lip and no higher, reached {highest}"
		);
		assert!(place.0.x > 2.5, "and carried on past the far edge, at {}", place.0.x);
		assert!((place.0.y - 0.5).abs() < 0.02, "back down on the floor, at {}", place.0.y);
	}

	#[test]
	fn a_box_stops_at_a_lip_taller_than_its_step() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);
		world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::cuboid(Vec3::new(0.5, 0.3, 2.0)),
			Transform::at(Vec3::new(2.0, 0.3, 0.0)),
		));
		simulation.step(&mut world);

		let mut place = (Vec3::new(1.0, 0.5, 0.0), Vec3::ZERO);

		for _ in 0..40 {
			place.1.x = 5.0;
			let moved = walk(&world, &mut place, character::STEP);

			assert!(!moved.stepped, "a lip that tall is a wall");
		}

		assert!(
			(place.0.x - 1.25).abs() < 0.05,
			"it is up against the face and no further, at {}",
			place.0.x
		);
		assert!((place.0.y - 0.5).abs() < 0.02, "and still on the floor, at {}", place.0.y);
	}

	#[test]
	fn a_box_does_not_step_onto_something_too_steep_to_stand_on() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);

		// a wedge low enough to step over - its highest point is a little over
		// three tenths against a step of thirty-five hundredths - and turned so
		// that the face a box would come down on is sixty degrees from flat.
		// Short enough to climb and too steep to stand on is exactly the pair
		// the check inside the climb exists for.
		let mut wedge = Transform::at(Vec3::new(2.0, 0.1, 0.0));
		wedge.rotation = Quat::from_rotation_z(core::f32::consts::FRAC_PI_3);
		world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::cuboid(Vec3::new(0.15, 0.15, 2.0)),
			wedge,
		));
		simulation.step(&mut world);

		let mut place = (Vec3::new(1.0, 0.5, 0.0), Vec3::ZERO);

		for _ in 0..40 {
			place.1.x = 5.0;
			let moved = walk(&world, &mut place, character::STEP);

			assert!(!moved.stepped, "a step is for standing on, and that face is not");
		}

		assert!(
			place.0.x < 1.7,
			"so it is stopped against the wedge rather than up on it, at {}",
			place.0.x
		);
		assert!((place.0.y - 0.5).abs() < 0.02, "and still on the floor, at {}", place.0.y);
	}

	#[test]
	fn a_box_in_mid_air_does_not_climb_the_wall_it_runs_into() {
		let (mut world, mut simulation) = wired();
		// no floor at all, so the box is falling for the whole run.
		world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::cuboid(Vec3::new(0.5, 0.1, 2.0)),
			Transform::at(Vec3::new(2.0, 0.1, 0.0)),
		));
		simulation.step(&mut world);

		let mut place = (Vec3::new(1.0, 0.5, 0.0), Vec3::ZERO);

		for _ in 0..40 {
			place.1.x = 5.0;
			let moved = walk(&world, &mut place, character::STEP);

			assert!(
				!moved.stepped,
				"a lip climbed in mid-air is a box walking up the outside of a wall"
			);
		}

		assert!(
			place.0.y < 0.0,
			"it went past the lip rather than onto it, ending at {}",
			place.0.y
		);
	}

	#[test]
	fn a_box_knows_what_it_is_standing_on_and_when_it_is_not() {
		let (mut world, mut simulation) = wired();
		let floor = ground(&mut world);
		simulation.step(&mut world);

		let standing = character::move_and_slide(
			&world,
			&Motion::new(Vec3::new(0.0, 0.5, 0.0), Vec3::ZERO, WALKER, STEP_SECONDS),
		);
		let falling = character::move_and_slide(
			&world,
			&Motion::new(Vec3::new(0.0, 4.0, 0.0), Vec3::ZERO, WALKER, STEP_SECONDS),
		);

		assert!(standing.grounded, "there is a floor under it");
		assert_eq!(standing.ground_body, floor, "and it is the one that was made");
		assert!(
			standing
				.ground_normal
				.abs_diff_eq(Vec3::Y, 1.0e-3),
			"lying flat, got {}",
			standing.ground_normal
		);

		assert!(!falling.grounded, "four units up there is nothing");
		assert_eq!(falling.ground_body, BodyId::NONE, "and nothing to name");
	}

	#[test]
	fn a_limit_no_surface_can_meet_leaves_a_box_in_the_air() {
		let (mut world, mut simulation) = wired();
		ground(&mut world);
		simulation.step(&mut world);

		let strict = character::move_and_slide(
			&world,
			&Motion::new(Vec3::new(0.0, 0.5, 0.0), Vec3::ZERO, WALKER, STEP_SECONDS)
				.standing(1.01),
		);

		assert!(
			!strict.grounded,
			"a limit past straight up is a limit nothing meets, and the floor is right there"
		);
	}

	#[test]
	fn a_slope_is_ground_only_when_it_is_flat_enough() {
		let landed = |degrees: f32| {
			let (mut world, mut simulation) = wired();
			let tilt = degrees.to_radians();
			let mut ramp = Transform::IDENTITY;
			ramp.rotation = Quat::from_rotation_z(tilt);

			world.bodies.spawn(Body::new(
				BodyKind::Static,
				Shape::cuboid(Vec3::splat(4.0)),
				ramp,
			));
			simulation.step(&mut world);

			// straight above the middle of the ramp's top face, which is not above
			// the middle of the ramp: the face slides sideways as the ramp turns,
			// and dropping over the origin instead means a steep enough ramp is
			// missed entirely rather than stood on. Only far enough up to land
			// gently, too, because a fast landing slides far enough to reach the
			// end face, whose normal is flat enough to stand on and would be
			// answering a different question.
			let face = Vec3::new(-4.0 * tilt.sin(), 4.0 * tilt.cos(), 0.0);
			let mut place = (face + Vec3::Y * (WALKER.y + 0.6), Vec3::ZERO);
			let mut ever = false;

			for _ in 0..40 {
				ever |= walk(&world, &mut place, character::STEP).grounded;
			}

			ever
		};

		assert!(landed(30.0), "half the limit is something to stand on");
		assert!(!landed(60.0), "and half again past it is not");
	}

	#[test]
	fn a_static_body_follows_the_entity_it_is_bolted_to() {
		let (mut world, mut simulation) = wired();
		let entity = world
			.entities
			.spawn_at(Transform::at(Vec3::new(0.0, 0.0, -3.0)));
		world.attach_body(entity, BodyKind::Static, Shape::UNIT);
		simulation.step(&mut world);

		if let Some(transform) = world.entities.transform_mut(entity) {
			transform.position = Vec3::new(0.0, 0.0, 3.0);
		}
		simulation.step(&mut world);

		let result =
			world.trace_ray(&TraceInfo::ray(Vec3::new(0.0, 0.0, 8.0), Vec3::new(0.0, 0.0, -8.0)));

		assert!(result.hit, "the collider went where the entity went");
		assert!(
			(result.end.z - 3.5).abs() < 1.0e-3,
			"to the near face of where it is now, got {}",
			result.end.z
		);
	}

	#[test]
	fn a_dynamic_body_writes_the_entity_rather_than_reading_it() {
		let (mut world, mut simulation) = wired();
		// nothing pulling on it, so the only thing that can have moved the
		// entity is the body, which is the whole of what this asks.
		world.gravity = Vec3::ZERO;
		let entity = world.entities.spawn_at(Transform::IDENTITY);
		let body = world.attach_body(entity, BodyKind::Dynamic, Shape::UNIT);

		if let Some(body) = world.bodies.get_mut(body) {
			body.transform.position = Vec3::new(0.0, 7.0, 0.0);
		}
		simulation.step(&mut world);

		assert!(
			world
				.entities
				.transform(entity)
				.expect("the entity is alive")
				.position
				.abs_diff_eq(Vec3::new(0.0, 7.0, 0.0), 1.0e-5),
			"the solver owns a dynamic body's transform"
		);
	}

	#[test]
	fn a_teleport_moves_both_and_leaves_nothing_to_interpolate() {
		let (mut world, mut simulation) = wired();
		let entity = world.entities.spawn_at(Transform::IDENTITY);
		let body = world.attach_body(entity, BodyKind::Dynamic, Shape::UNIT);
		simulation.step(&mut world);
		world.advance();

		assert!(
			world.teleport_body(body, Transform::at(Vec3::new(0.0, 20.0, 0.0))),
			"the handle resolves"
		);
		world.settle();
		world.set_interpolation(0.5);

		assert!(
			world
				.render_transform(entity)
				.expect("the entity is alive")
				.position
				.abs_diff_eq(Vec3::new(0.0, 20.0, 0.0), 1.0e-4),
			"halfway through the step it is already there, rather than ten units up"
		);
	}

	#[test]
	fn a_despawned_body_stops_being_hit() {
		let (mut world, mut simulation) = wired();
		let body =
			world
				.bodies
				.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY));
		simulation.step(&mut world);

		let trace = TraceInfo::ray(Vec3::new(0.0, 0.0, 4.0), Vec3::new(0.0, 0.0, -4.0));
		assert!(world.trace_ray(&trace).hit, "it is there to begin with");

		world.bodies.despawn(body);
		simulation.step(&mut world);

		assert!(!world.trace_ray(&trace).hit, "and gone afterwards");
	}

	#[test]
	fn a_reused_slot_does_not_keep_the_collision_mesh_of_its_predecessor() {
		let (mut world, mut simulation) = wired();
		let first = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::mesh(MeshId::CUBE),
			Transform::IDENTITY,
		));
		simulation.step(&mut world);
		assert!(simulation.collider(first).is_some(), "the cube was baked");

		world.bodies.despawn(first);
		let second = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::ball(1.0),
			Transform::IDENTITY,
		));
		simulation.step(&mut world);

		assert_eq!(first.slot(), second.slot(), "the slot really is reused");
		assert!(simulation.collider(first).is_none(), "and the stale handle finds nothing");
		assert!(simulation.collider(second).is_none(), "nor does a ball have triangles");
	}

	#[test]
	fn a_body_driving_nothing_is_still_traced_against() {
		let (mut world, mut simulation) = wired();
		let body = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::ball(1.0),
			Transform::at(Vec3::new(0.0, 0.0, -2.0)),
		));
		simulation.step(&mut world);

		let result =
			world.trace_ray(&TraceInfo::ray(Vec3::new(0.0, 0.0, 4.0), Vec3::new(0.0, 0.0, -4.0)));

		assert_eq!(result.body, body, "a trigger volume has no entity and is still there");
		assert_eq!(result.entity, EntityId::NONE, "and reports none");
	}
}
