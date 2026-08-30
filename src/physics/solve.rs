//! Sequential impulses: gravity in, contacts resolved, transforms out.
//!
//! The scheme is Catto's and the names are his. Each contact point becomes a
//! row with its own effective mass, and the rows are visited in turn, over and
//! over, each one correcting the velocity the previous ones left behind. Eight
//! passes is enough for a pile; one is enough for a ball.
//!
//! Two choices are worth knowing before reading the arithmetic.
//!
//! **Penetration is removed by a second set of velocities, not by the first.**
//! Pushing bodies apart by adding to their real velocity works and then hands
//! them the energy it took to do it, so a stack of boxes settles by climbing
//! slowly out of itself and a deep overlap fires like a spring. The split
//! impulse gives every body a *pseudo* velocity used only to move it, thrown
//! away at the end of the step. Overlap goes away and nothing gains speed.
//!
//! **A body the solver does not move is a body of zero inverse mass**, and so
//! is a sleeping one. That is the whole of the special-casing: no branch
//! anywhere asks whether the thing on the other side of a contact is a wall,
//! because a wall is a row that divides by infinity.

use std::{collections::HashMap, f32::consts::TAU};

use colby_core::{
	abi::{Bodies, Body, BodyId, BodyKind, Joint, JointKind, Joints},
	glam::{Mat3, Quat, Vec3},
};

use crate::{
	contact::{Contact, Manifold, inverse_inertia},
	convex::MAX_CONTACTS,
};

/// How many times the velocities are corrected each step, unless the console
/// says otherwise.
///
/// The one number in this file that a person may reasonably want to turn. A
/// pile needs roughly as many passes as it is boxes tall, squared over
/// something: one box is happy with two, six wants about twenty, and above
/// that the count rises faster than it is worth paying. @ref
/// `colby-known-gaps` for what happens when there are not enough.
pub const VELOCITY_PASSES: usize = 10;

/// The most the console will ask for.
///
/// Not a safety limit so much as a typo limit: the passes are the whole cost of
/// the solver, and a spare zero would be a frame that takes a second.
pub const MAX_PASSES: usize = 64;

/// How many times the positions are.
///
/// Fewer, because the pseudo velocities converge faster: they have no gravity
/// fighting them and no restitution to overshoot.
const POSITION_PASSES: usize = 5;

/// How much overlap is left alone.
///
/// Chasing the last thousandth of a unit is what makes a settled pile hum.
const SLOP: f32 = 0.005;

/// What fraction of the remaining overlap one step removes.
const RECOVERY: f32 = 0.25;

/// Below this approach speed a contact does not bounce.
///
/// Without it a ball at rest on the floor keeps being told to leave at a
/// hundredth of the speed it arrived, forever.
const BOUNCE_FLOOR: f32 = 1.0;

/// How much speed a body loses to nothing each second.
const LINEAR_DAMPING: f32 = 0.06;

/// How much spin it loses.
const ANGULAR_DAMPING: f32 = 0.12;

/// Slower than this counts as still.
const SLEEP_SPEED: f32 = 0.06;

/// Turning slower than this does too.
const SLEEP_SPIN: f32 = 0.2;

/// How long a body has to be still before the solver stops integrating it.
const SLEEP_AFTER: f32 = 0.45;

/// Divisors below this are zero.
const EPSILON: f32 = 1.0e-8;

/// What fraction of a joint's positional error one step takes out.
///
/// Joints correct their drift with a *bias* folded into the velocity solve
/// rather than with the second set of velocities the contacts use. They can
/// afford to: a joint is not redundant with anything, so the energy a bias
/// hands it has nowhere to accumulate, and the split-impulse machinery would be
/// three more arrays for a problem that does not arise.
const MEND: f32 = 0.2;

/// How far a joint may be out before it is worth mending at all.
const JOINT_SLOP: f32 = 1.0e-3;

/// What a joint's spring works out to, for one step at one rate.
///
/// The soft-constraint formulation, and the whole of it is two scalars over the
/// arithmetic that was already here. Writing `K` for the effective inverse mass
/// a constraint presents, `h` for the step, `w` for the spring's frequency in
/// radians and `z` for its damping ratio, the usual derivation gives a bias
/// factor and a softness `y`:
///
/// ```text
/// q = h*w * (2z + h*w)      y = K / q      bias = h*w / (2z + h*w)
/// ```
///
/// and the impulse is `(K + y)^-1 * -(Cdot + bias/h * C + y * spent)`. Because
/// `y` is `K/q` rather than a free number, `K + y` is `K * (q+1)/q`, its
/// inverse is the *rigid* effective mass times `q/(q+1)`, and the whole soft
/// solve collapses into the rigid one scaled by one number with a second number
/// times the running total taken off it. No matrix in this file changes, and
/// the rigid path is the case `push = 1, relax = 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Spring {
	/// What fraction of the error is asked back as a velocity each step.
	bias: f32,

	/// What fraction of the impulse a rigid joint would want this one takes.
	push: f32,
}

impl Spring {
	/// The constraint this table solved before springs existed.
	///
	/// Not the limit of the formula above as the frequency rises - that limit
	/// has a bias of one - but the Baumgarte factor that was measured and
	/// tuned. Keeping it is what makes a joint written before this field
	/// existed behave bit for bit as it did.
	const RIGID: Self = Self { bias: MEND, push: 1.0 };

	/// What a joint's numbers work out to at this step length.
	///
	/// @param joint - the joint, for its stiffness and damping
	/// @param dt - how long a step is, in seconds
	fn of(joint: &Joint, dt: f32) -> Self {
		if joint.stiffness <= 0.0 {
			return Self::RIGID;
		}

		let omega = TAU * joint.stiffness;
		let zeta = joint.damping.max(0.0);
		let reach = 2.0_f32.mul_add(zeta, dt * omega);
		let quality = dt * omega * reach;

		if quality <= EPSILON || reach <= EPSILON {
			return Self::RIGID;
		}

		Self {
			bias: dt * omega / reach,
			push: quality / (quality + 1.0),
		}
	}

	/// How much of what has already been spent is handed back each pass.
	const fn relax(self) -> f32 { 1.0 - self.push }
}

/// What one joint has spent so far this step.
///
/// Cleared at the top of every step rather than carried across one, because
/// joints are not warm started - @ref `colby-known-gaps`. Two vectors because
/// the two halves of a joint have different units and cannot share a ceiling.
#[derive(Clone, Copy, Debug, Default)]
struct Spent {
	/// The impulse spent holding the anchors together.
	linear: Vec3,

	/// The angular impulse spent holding the orientations together.
	angular: Vec3,
}

/// Holds a running total to a ceiling and says what may still be applied.
///
/// The clamp is on the *total over the step* and not on one pass's share,
/// because the pass count is a console variable about quality and a ceiling
/// that rose with it would not be a ceiling. A pass that would take the total
/// past the limit therefore applies the part that fits and nothing more, and
/// every pass after it in the same step applies nothing at all.
///
/// @param total - what has been spent, written
/// @param wanted - what it would be if this pass had its way
/// @param ceiling - the most it may reach, or zero for no ceiling
/// @return the impulse that may actually be applied now
fn hold(total: &mut Vec3, wanted: Vec3, ceiling: f32) -> Vec3 {
	let held = if ceiling > 0.0 && wanted.length() > ceiling {
		wanted.normalize_or_zero() * ceiling
	} else {
		wanted
	};
	let applied = held - *total;
	*total = held;

	applied
}

/// How many contact points one pair is remembered by.
const REMEMBERED: usize = 4;

/// One contact point, with everything the passes need already worked out.
#[derive(Clone, Copy, Debug)]
struct Row {
	/// Which feature of the pair produced this point.
	feature: u32,

	/// Which manifold this came from.
	///
	/// Carried rather than inferred, and that is not tidiness. Rows are built
	/// by walking the manifolds and *skipping* some of them - a pair that is a
	/// wall against a sleeper needs no impulse - so the two lists are different
	/// lengths and matching them up by position files every impulse after the
	/// first skip under the wrong pair. What that looks like from outside is a
	/// stack that will not settle and creeps sideways about a tenth of a unit a
	/// second, which is a long way from where the mistake is.
	manifold: usize,

	/// The slot of the body the normal points away from.
	first: usize,

	/// The slot of the body it points at.
	second: usize,

	/// Which way to push, from the first at the second.
	normal: Vec3,

	/// Two directions across it.
	tangents: [Vec3; 2],

	/// From the first body's middle to the contact.
	from_first: Vec3,

	/// From the second body's middle to the contact.
	from_second: Vec3,

	/// One over the effective mass along the normal.
	normal_mass: f32,

	/// The same across it.
	tangent_mass: [f32; 2],

	/// How fast the two should be leaving each other when this is done.
	bounce: f32,

	/// How fast the position pass should push them apart.
	recovery: f32,

	/// How much of a push has been applied so far.
	normal_impulse: f32,

	/// How much of a drag.
	tangent_impulse: [f32; 2],

	/// How much of a push the position pass has applied.
	push_impulse: f32,

	/// How hard this pair is to slide.
	friction: f32,
}

/// Every body's state while the passes run.
///
/// Indexed by body slot, so a row addresses a body with one number and no
/// lookup. Kept on the [`Simulation`](crate::Simulation) between steps rather
/// than allocated per step.
#[derive(Debug, Default)]
pub(crate) struct Solver {
	/// One over each body's mass, or zero for anything unmoved.
	inverse_mass: Vec<f32>,

	/// Each body's world-space inverse inertia tensor.
	inverse_inertia: Vec<Mat3>,

	/// Where each body's middle is.
	center: Vec<Vec3>,

	/// How fast each is moving.
	velocity: Vec<Vec3>,

	/// How fast each is turning.
	angular: Vec<Vec3>,

	/// Which way each is facing.
	///
	/// The joints' - a body's anchor is written in the body's own space, and
	/// getting it into the world is this and the center.
	rotation: Vec<Quat>,

	/// The velocity that only moves a body out of an overlap.
	pseudo: Vec<Vec3>,

	/// How long each body has been still.
	resting: Vec<f32>,

	/// Which bodies are touching anything at all this step.
	touching: Vec<bool>,

	/// Which island each body belongs to, as a union-find parent.
	///
	/// Sleeping is a property of an island rather than of a body, and finding
	/// that out cost an afternoon: a box at the bottom of a pile goes still
	/// first, falls asleep alone, becomes a wall under everything above it, and
	/// is woken again by the very next step because something awake is leaning
	/// on it. That cycle never converges, and on the way round it kicks the
	/// pile hard enough to be visible. A stack sleeps all at once or not at
	/// all.
	parent: Vec<usize>,

	/// The rows, rebuilt each step.
	rows: Vec<Row>,

	/// Where each manifold's rows start and stop.
	///
	/// The rows of one pair are solved *together* rather than in turn, which is
	/// the whole reason they have to be findable as a group. @ref
	/// [`together`](Self::together).
	groups: Vec<(usize, usize)>,

	/// What each touching pair pushed with last step, keyed by the pair and
	/// which piece of a many-piece shape it was against.
	///
	/// The single most valuable thing in this file. Without it every step
	/// starts from no contact force at all and spends its passes rediscovering
	/// the weight of the pile above - and because the rows are solved in order,
	/// what it rediscovers is slightly wrong in the *same* direction every
	/// step, which a stack shows as a slow sideways crawl rather than as
	/// obvious jitter. Seeded with last step's answer, the passes start
	/// converged and the crawl is gone.
	cache: HashMap<(BodyId, BodyId, u32), [Remembered; REMEMBERED]>,

	/// What each joint has spent this step, indexed by joint slot.
	///
	/// Two things read it and it exists for both: a ceiling has to know what
	/// has already gone, and a soft constraint hands part of what it has
	/// already spent back every pass. Doing either without the other would be
	/// the same array twice.
	spent: Vec<Spent>,
}

/// What one contact point pushed with, kept for the next step.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Remembered {
	/// Which feature it was. @ref [`Touch::id`](crate::convex::Touch::id).
	id: u32,

	/// How hard it pushed.
	normal_impulse: f32,

	/// How hard it dragged.
	tangent_impulse: [f32; 2],

	/// Whether this slot holds anything at all.
	real: bool,
}

impl Solver {
	/// A solver that has never run.
	pub(crate) fn new() -> Self {
		Self {
			inverse_mass: Vec::new(),
			inverse_inertia: Vec::new(),
			center: Vec::new(),
			velocity: Vec::new(),
			angular: Vec::new(),
			rotation: Vec::new(),
			pseudo: Vec::new(),
			resting: Vec::new(),
			touching: Vec::new(),
			parent: Vec::new(),
			rows: Vec::new(),
			groups: Vec::new(),
			cache: HashMap::new(),
			spent: Vec::new(),
		}
	}

	/// Drops what this solver knows that outlives a step.
	///
	/// Exactly two things do. `resting` is how long each body has been still,
	/// and it is what decides when an island sleeps - a pile put back into a
	/// solver that had been watching another one go still would fall asleep
	/// the moment it arrived. `cache` is last step's impulses, and a step
	/// seeded from a world that is no longer here starts out pushing for
	/// reasons of its own.
	///
	/// Every other field is cleared and refilled by the step that reads it, so
	/// there is deliberately nothing about them here: a line that cannot be
	/// wrong is a line nobody can check.
	///
	/// @note: of the two, only `resting` has a test behind it. Keeping the
	/// impulse cache across a restore was tried at seven capture points
	/// against ten run lengths and changed nothing measurable, because it
	/// holds one step's worth and the next step overwrites it. It is cleared
	/// anyway, because it is *read* before it is rebuilt and being seeded from
	/// a world that is gone is wrong whether or not this scene shows it.
	pub(crate) fn forget(&mut self) {
		self.resting.clear();
		self.cache.clear();
	}

	/// Advances every dynamic body by one step.
	///
	/// @param bodies - the table, read and written
	/// @param manifolds - what the narrow phase found
	/// @param gravity - what everything accelerates by
	/// @param dt - how long a step is, in seconds
	/// @param passes - how many times to correct the velocities, clamped into
	/// `1 ..= MAX_PASSES`
	pub(crate) fn run(
		&mut self,
		bodies: &mut Bodies,
		joints: &Joints,
		manifolds: &[Manifold],
		gravity: Vec3,
		dt: f32,
		passes: usize,
	) {
		self.load(bodies);
		self.wake(bodies, manifolds);
		self.rouse(bodies, joints);
		self.accelerate(bodies, gravity, dt);
		self.rows(bodies, manifolds, dt);
		self.warm();

		// nothing spent yet. Cleared here rather than carried, because joints
		// are not warm started and because a ceiling is a ceiling *per step*.
		self.spent.clear();
		self.spent
			.resize(joints.slots(), Spent::default());

		for _ in 0..passes.clamp(1, MAX_PASSES) {
			// joints first: they are the stiffer constraint, and a contact that
			// has to argue with one converges faster than the other way round.
			for (id, joint) in joints.iter() {
				self.joint(id.slot(), joint, dt);
			}

			self.push();
		}

		for _ in 0..POSITION_PASSES {
			self.separate();
		}

		self.remember(manifolds);
		self.rest(bodies, manifolds, dt);
		self.island(bodies, manifolds);
		self.integrate(bodies, dt);
	}

	/// Counts how long each body has been going nowhere.
	///
	/// Touching something is part of going nowhere. A body with no contact at
	/// all this step is in the air, and a body that falls asleep in the air is
	/// a wall hanging over whatever was under it - which is not merely wrong
	/// but *unstable*, because the next step it touches something again, is
	/// woken, and the pile it was part of restarts with no accumulated contact
	/// impulses at all. A tall stack loses a contact for a single step now and
	/// then as it settles, and that was enough to knock over anything above
	/// four boxes.
	fn rest(&mut self, bodies: &Bodies, manifolds: &[Manifold], dt: f32) {
		self.touching.clear();
		self.touching.resize(bodies.slots(), false);

		for manifold in manifolds {
			touch(&mut self.touching, manifold.first.slot());
			touch(&mut self.touching, manifold.second.slot());
		}

		for (id, _) in bodies.iter() {
			let slot = id.slot();

			if self.inverse_mass[slot] <= 0.0 {
				continue;
			}

			let still = self.velocity[slot].length() < SLEEP_SPEED
				&& self.angular[slot].length() < SLEEP_SPIN
				&& self.touching[slot];

			self.resting[slot] = if still { self.resting[slot] + dt } else { 0.0 };
		}
	}

	/// Groups bodies that are leaning on each other, and sleeps the ones that
	/// have all gone still.
	///
	/// Only movable bodies join an island. A static floor does not, or the
	/// first pile to touch it would be in the same island as every other pile
	/// in the level and none of them would ever sleep.
	fn island(&mut self, bodies: &mut Bodies, manifolds: &[Manifold]) {
		let slots = bodies.slots();
		self.parent.clear();
		self.parent.extend(0..slots);

		for manifold in manifolds {
			let (first, second) = (manifold.first.slot(), manifold.second.slot());

			if self.awake(first) && self.awake(second) {
				self.join(first, second);
			}
		}

		// an island sleeps only if every one of its members is ready to, so one
		// body still moving keeps the whole pile awake.
		let mut ready = vec![true; slots];
		for slot in 0..slots {
			if !self.awake(slot) {
				continue;
			}

			if self.resting[slot] < SLEEP_AFTER {
				let root = self.root(slot);
				ready[root] = false;
			}
		}

		let handles: Vec<BodyId> = bodies.iter().map(|(id, _)| id).collect();
		for id in handles {
			let slot = id.slot();

			if !self.awake(slot) || !ready[self.root(slot)] {
				continue;
			}

			self.velocity[slot] = Vec3::ZERO;
			self.angular[slot] = Vec3::ZERO;
			self.inverse_mass[slot] = 0.0;

			if let Some(body) = bodies.get_mut(id) {
				body.velocity = Vec3::ZERO;
				body.angular = Vec3::ZERO;
				body.sleeping = true;
			}
		}
	}

	/// Whether the solver is moving the body in a slot this step.
	fn awake(&self, slot: usize) -> bool {
		self.inverse_mass
			.get(slot)
			.is_some_and(|&mass| mass > 0.0)
	}

	/// The island a slot belongs to, flattening the path on the way.
	fn root(&mut self, slot: usize) -> usize {
		let mut current = slot;

		while self.parent[current] != current {
			self.parent[current] = self.parent[self.parent[current]];
			current = self.parent[current];
		}

		current
	}

	/// Puts two slots in the same island.
	fn join(&mut self, first: usize, second: usize) {
		let (one, other) = (self.root(first), self.root(second));

		if one != other {
			self.parent[one] = other;
		}
	}

	/// Applies last step's contact impulses before the first pass runs.
	///
	/// Seeding the accumulators is only half of it: the impulses have to
	/// actually be applied, or the first pass sees velocities that never felt
	/// them and immediately cancels the seed.
	fn warm(&mut self) {
		for index in 0..self.rows.len() {
			let row = self.rows[index];
			let impulse = row.normal * row.normal_impulse
				+ row.tangents[0] * row.tangent_impulse[0]
				+ row.tangents[1] * row.tangent_impulse[1];

			self.apply(row.first, row.second, row.from_first, row.from_second, impulse);
		}
	}

	/// Files this step's impulses away for the next one.
	///
	/// Driven from the rows and their own record of where they came from, never
	/// from the two lists side by side. @ref [`Row::manifold`].
	fn remember(&mut self, manifolds: &[Manifold]) {
		self.cache.clear();

		for row in &self.rows {
			let Some(manifold) = manifolds.get(row.manifold) else {
				continue;
			};

			let kept = self
				.cache
				.entry((manifold.first, manifold.second, manifold.shard))
				.or_insert_with(|| [Remembered::default(); REMEMBERED]);

			let Some(slot) = kept.iter_mut().find(|it| !it.real) else {
				continue;
			};

			*slot = Remembered {
				id: row.feature,
				normal_impulse: row.normal_impulse,
				tangent_impulse: row.tangent_impulse,
				real: true,
			};
		}
	}

	/// Wakes both ends of a joint if either end is moving.
	///
	/// Without this a prop hung from a rope goes to sleep, the rope goes with
	/// it, and whatever it is tied to can then move without dragging it - which
	/// looks exactly like the rope having quietly detached.
	fn rouse(&mut self, bodies: &mut Bodies, joints: &Joints) {
		let mut woken: Vec<BodyId> = Vec::new();

		for (_, joint) in joints.iter() {
			let stirring = [joint.first, joint.second]
				.into_iter()
				.any(|id| bodies.get(id).is_some_and(disturbs));

			if !stirring {
				continue;
			}

			woken.extend(
				[joint.first, joint.second]
					.into_iter()
					.filter(|&id| bodies.get(id).is_some_and(|body| body.sleeping)),
			);
		}

		for id in woken {
			self.revive(bodies, id);
		}
	}

	/// Corrects one joint.
	///
	/// Every kind is made of the same two pieces: a *point* constraint that
	/// holds two anchors together, and an *angular* one that holds two
	/// orientations together. A weld is both. A hinge is both, with the angular
	/// one relaxed about the axis it turns on. A rope is neither - it is one
	/// scalar along the line between the anchors, and unlike everything else in
	/// this file it may only ever *pull*.
	///
	/// @param slot - which joint this is, for the ceiling's running total
	/// @param joint - the joint
	/// @param dt - how long a step is, for turning an error into a speed
	fn joint(&mut self, slot: usize, joint: &Joint, dt: f32) {
		let world = self.center.len() - 1;
		let (first, second) = (joint.first.slot(), joint.second.slot());

		if first >= world {
			return;
		}

		// a joint with nothing on the far end is pinned to the world, and the
		// world is the slot that weighs infinity.
		let anchored = !joint.second.is_some() || second >= world;
		let second = if anchored { world } else { second };

		let one = self.center[first] + self.rotation[first] * joint.first_anchor;
		let other = if anchored {
			// the world slot is put at the anchor, so the lever arm on that
			// side is nothing and the pivot is the anchor itself.
			self.center[world] = joint.second_anchor;
			self.rotation[world] = Quat::IDENTITY;
			self.velocity[world] = Vec3::ZERO;
			self.angular[world] = Vec3::ZERO;

			joint.second_anchor
		} else {
			self.center[second] + self.rotation[second] * joint.second_anchor
		};

		if slot >= self.spent.len() {
			return;
		}

		let spring = Spring::of(joint, dt);

		match joint.kind {
			| JointKind::Rope =>
				self.rope(slot, joint, (first, second), (one, other), dt, spring),
			| JointKind::Weld | JointKind::Axis => {
				self.pinned(slot, joint, (first, second), (one, other), dt, spring);
				self.aligned(slot, joint, (first, second), dt, spring);
			},
		}
	}

	/// The one scalar constraint of a rope.
	fn rope(
		&mut self,
		slot: usize,
		joint: &Joint,
		slots: (usize, usize),
		anchors: (Vec3, Vec3),
		dt: f32,
		spring: Spring,
	) {
		let (first, second) = slots;
		let between = anchors.1 - anchors.0;
		let stretch = between.length() - joint.length.max(0.0);

		// slack, so the rope is not there at all. This is the whole of what
		// makes it a rope rather than a rod, and it is one `if`.
		if stretch <= 0.0 {
			return;
		}

		let along = between.normalize_or(Vec3::Y);
		let (from_first, from_second) = self.arms(slots, anchors);
		let closing = self
			.relative(first, second, from_first, from_second)
			.dot(along);
		let mass = self.effective(first, second, from_first, from_second, along);

		let mend = spring.bias * (stretch - JOINT_SLOP).max(0.0) / dt;
		let rigid = -(closing + mend) * mass;
		let already = self.spent[slot].linear;
		let wanted = already + along * (rigid * spring.push) - already * spring.relax();

		let applied = hold(&mut self.spent[slot].linear, wanted, joint.max_impulse);
		self.apply(first, second, from_first, from_second, applied);
	}

	/// The three constraints that hold two anchors in the same place.
	fn pinned(
		&mut self,
		slot: usize,
		joint: &Joint,
		slots: (usize, usize),
		anchors: (Vec3, Vec3),
		dt: f32,
		spring: Spring,
	) {
		let (first, second) = slots;
		let (from_first, from_second) = self.arms(slots, anchors);
		let drift = anchors.1 - anchors.0;
		let closing = self.relative(first, second, from_first, from_second);

		let mut mass = Mat3::from_diagonal(Vec3::splat(
			self.inverse_mass[first] + self.inverse_mass[second],
		));
		mass -= skew(from_first) * self.inverse_inertia[first] * skew(from_first);
		mass -= skew(from_second) * self.inverse_inertia[second] * skew(from_second);

		if mass.determinant().abs() < EPSILON {
			return;
		}

		let mend = drift * (spring.bias / dt);
		let rigid = mass.inverse() * -(closing + mend);
		let already = self.spent[slot].linear;
		let wanted = already + rigid * spring.push - already * spring.relax();

		let applied = hold(&mut self.spent[slot].linear, wanted, joint.max_impulse);
		self.apply(first, second, from_first, from_second, applied);
	}

	/// The angular constraints that stop two bodies turning away from each
	/// other.
	///
	/// The error is read out of the relative rotation as a small-angle vector,
	/// which is the usual approximation and is exact enough for a joint that is
	/// never allowed to get far out. A hinge then throws away the component
	/// along its own axis, which is the one direction it is *for*.
	fn aligned(
		&mut self,
		slot: usize,
		joint: &Joint,
		slots: (usize, usize),
		dt: f32,
		spring: Spring,
	) {
		let (first, second) = slots;
		let relative = self.rotation[second] * self.rotation[first].inverse();
		let drift = relative * joint.rest.inverse();
		let mut error = Vec3::new(drift.x, drift.y, drift.z) * 2.0;

		if drift.w < 0.0 {
			error = -error;
		}

		let mut closing = self.angular[second] - self.angular[first];
		let mut mass = self.inverse_inertia[first] + self.inverse_inertia[second];

		if joint.kind == JointKind::Axis {
			// free about the hinge, held about everything else.
			let axis = (self.rotation[first] * joint.axis).normalize_or(Vec3::Y);

			error -= axis * error.dot(axis);
			closing -= axis * closing.dot(axis);
			mass += Mat3::from_diagonal(Vec3::splat(EPSILON));
		}

		if mass.determinant().abs() < EPSILON {
			return;
		}

		let mend = error * (spring.bias / dt);
		let mut rigid = mass.inverse() * -(closing + mend);

		if joint.kind == JointKind::Axis {
			let axis = (self.rotation[first] * joint.axis).normalize_or(Vec3::Y);
			rigid -= axis * rigid.dot(axis);
		}

		let already = self.spent[slot].angular;
		let wanted = already + rigid * spring.push - already * spring.relax();
		let impulse = hold(&mut self.spent[slot].angular, wanted, joint.max_torque);

		self.angular[first] -= self.inverse_inertia[first] * impulse;
		self.angular[second] += self.inverse_inertia[second] * impulse;
	}

	/// The two lever arms of a joint, from each body's middle to its anchor.
	fn arms(&self, slots: (usize, usize), anchors: (Vec3, Vec3)) -> (Vec3, Vec3) {
		(anchors.0 - self.center[slots.0], anchors.1 - self.center[slots.1])
	}

	/// Puts a body back to work.
	fn revive(&mut self, bodies: &mut Bodies, id: BodyId) {
		let slot = id.slot();

		if let Some(body) = bodies.get_mut(id) {
			body.sleeping = false;
			self.inverse_mass[slot] = body.inverse_mass();
			self.inverse_inertia[slot] = inverse_inertia(body);
		}

		if let Some(resting) = self.resting.get_mut(slot) {
			*resting = 0.0;
		}
	}

	/// Copies the table into the per-slot arrays.
	fn load(&mut self, bodies: &Bodies) {
		// one past the last body, and it is the world: a slot of infinite mass
		// that never moves. A joint pinned to a point rather than to a body
		// uses it, which is what lets every constraint below be written once
		// instead of twice with an `if` in the middle.
		let slots = bodies.slots() + 1;

		self.inverse_mass.clear();
		self.inverse_mass.resize(slots, 0.0);
		self.inverse_inertia.clear();
		self.inverse_inertia.resize(slots, Mat3::ZERO);
		self.center.clear();
		self.center.resize(slots, Vec3::ZERO);
		self.velocity.clear();
		self.velocity.resize(slots, Vec3::ZERO);
		self.angular.clear();
		self.angular.resize(slots, Vec3::ZERO);
		self.rotation.clear();
		self.rotation.resize(slots, Quat::IDENTITY);
		self.pseudo.clear();
		self.pseudo.resize(slots, Vec3::ZERO);
		self.resting.resize(slots, 0.0);

		for (id, body) in bodies.iter() {
			let slot = id.slot();

			self.center[slot] = body.transform.position;
			self.velocity[slot] = body.velocity;
			self.angular[slot] = body.angular;
			self.rotation[slot] = body.transform.rotation;

			// a sleeping body is a wall until something wakes it, and this one
			// line is the whole of what sleeping costs the solver.
			if body.movable() && !body.sleeping {
				self.inverse_mass[slot] = body.inverse_mass();
				self.inverse_inertia[slot] = inverse_inertia(body);
			}
		}
	}

	/// Wakes anything a moving body is leaning on.
	fn wake(&mut self, bodies: &mut Bodies, manifolds: &[Manifold]) {
		let mut woken: Vec<BodyId> = Vec::new();

		for manifold in manifolds {
			disturbed(bodies, manifold, &mut woken);
		}

		for id in woken {
			self.revive(bodies, id);
		}
	}

	/// Applies gravity and damping to everything awake.
	fn accelerate(&mut self, bodies: &Bodies, gravity: Vec3, dt: f32) {
		for (id, _) in bodies.iter() {
			let slot = id.slot();

			if self.inverse_mass[slot] <= 0.0 {
				continue;
			}

			self.velocity[slot] += gravity * dt;
			self.velocity[slot] *= 1.0 / dt.mul_add(LINEAR_DAMPING, 1.0);
			self.angular[slot] *= 1.0 / dt.mul_add(ANGULAR_DAMPING, 1.0);
		}
	}

	/// Turns the manifolds into rows.
	fn rows(&mut self, bodies: &Bodies, manifolds: &[Manifold], dt: f32) {
		self.rows.clear();
		self.groups.clear();

		for (index, manifold) in manifolds.iter().enumerate() {
			let (first, second) = (manifold.first.slot(), manifold.second.slot());

			if self
				.inverse_mass
				.get(first)
				.copied()
				.unwrap_or(0.0)
				<= 0.0 && self
				.inverse_mass
				.get(second)
				.copied()
				.unwrap_or(0.0)
				<= 0.0
			{
				continue;
			}

			if bodies.get(manifold.first).is_none() || bodies.get(manifold.second).is_none() {
				continue;
			}

			let start = self.rows.len();
			self.append(index, manifold, (first, second), dt);

			if self.rows.len() > start {
				self.groups.push((start, self.rows.len()));
			}
		}
	}

	/// Turns one manifold's points into rows, seeded from last step.
	///
	/// @param index - which manifold this is, which each row carries
	/// @param manifold - the pair
	/// @param slots - the two bodies' slots
	/// @param dt - how long a step is
	fn append(&mut self, index: usize, manifold: &Manifold, slots: (usize, usize), dt: f32) {
		let remembered = self
			.cache
			.get(&(manifold.first, manifold.second, manifold.shard))
			.copied();

		for point in manifold.points() {
			let mut row = self.row(index, manifold, slots, point, dt);
			if let Some(previous) = remembered.as_ref() {
				seed(&mut row, previous);
			}

			self.rows.push(row);
		}
	}

	/// Works out one row's effective masses and targets.
	fn row(
		&self,
		manifold_index: usize,
		manifold: &Manifold,
		slots: (usize, usize),
		point: &Contact,
		dt: f32,
	) -> Row {
		let (first, second) = slots;
		let normal = manifold.normal;
		let from_first = point.position - self.center[first];
		let from_second = point.position - self.center[second];
		let tangents = across(normal);

		let approach = self
			.relative(first, second, from_first, from_second)
			.dot(normal);

		// a contact that is merely resting must not be told to bounce, or a
		// settled pile shivers forever at the speed gravity added last step.
		let bounce = if approach < -BOUNCE_FLOOR {
			-manifold.restitution * approach
		} else {
			0.0
		};

		Row {
			feature: point.id,
			manifold: manifold_index,
			first,
			second,
			normal,
			tangents,
			from_first,
			from_second,
			normal_mass: self.effective(first, second, from_first, from_second, normal),
			tangent_mass: [
				self.effective(first, second, from_first, from_second, tangents[0]),
				self.effective(first, second, from_first, from_second, tangents[1]),
			],
			bounce,
			recovery: RECOVERY * (point.depth - SLOP).max(0.0) / dt,
			normal_impulse: 0.0,
			tangent_impulse: [0.0; 2],
			push_impulse: 0.0,
			friction: manifold.friction,
		}
	}

	/// How fast the second body is moving away from the first at a contact.
	fn relative(&self, first: usize, second: usize, from_first: Vec3, from_second: Vec3) -> Vec3 {
		let at_first = self.velocity[first] + self.angular[first].cross(from_first);
		let at_second = self.velocity[second] + self.angular[second].cross(from_second);

		at_second - at_first
	}

	/// The same, for the velocities that only remove overlap.
	///
	/// No lever arms, because there is no pseudo *angular* velocity to take
	/// one. @ref [`separate`](Self::separate) for why turning a body to clear
	/// an overlap is the thing that knocks a pile over.
	fn relative_pseudo(&self, first: usize, second: usize) -> Vec3 {
		self.pseudo[second] - self.pseudo[first]
	}

	/// One over the mass a contact presents along a direction.
	fn effective(
		&self,
		first: usize,
		second: usize,
		from_first: Vec3,
		from_second: Vec3,
		direction: Vec3,
	) -> f32 {
		let turn_first = from_first.cross(direction);
		let turn_second = from_second.cross(direction);

		let total = self.inverse_mass[first]
			+ self.inverse_mass[second]
			+ direction.dot((self.inverse_inertia[first] * turn_first).cross(from_first))
			+ direction.dot((self.inverse_inertia[second] * turn_second).cross(from_second));

		if total > EPSILON { 1.0 / total } else { 0.0 }
	}

	/// One velocity pass: separate along the normal, then drag across it.
	///
	/// Point by point, in the order the manifolds were found, which for a stack
	/// spawned from the ground up is the order the load travels. Revisiting
	/// each manifold's own points several times before moving on was tried and
	/// measured on 2026-08-27: it does not help, and at four inner passes it is
	/// worse. @ref `colby-known-gaps`.
	fn push(&mut self) {
		for index in 0..self.groups.len() {
			let (start, end) = self.groups[index];

			self.together(start, end);

			for row in start..end {
				self.rub(row);
			}
		}
	}

	/// Corrects one manifold's contacts, all of them from the same velocities.
	///
	/// **This is the difference between a pile of five and a pile of eight**,
	/// and it is one loop split into two.
	///
	/// A box resting flat touches at four corners, and those four are
	/// redundant: any three of them already say everything the fourth can.
	/// Solved one after another, each corner sees the velocity the previous
	/// ones left behind, so the load they settle into depends on the order they
	/// were visited in - and that bias is the same every step, which is exactly
	/// the recipe for a lean that grows. Reading all four from the same
	/// velocity and applying them together has no order in it at all, so a
	/// symmetric box is answered symmetrically.
	///
	/// It is Jacobi inside a manifold and Gauss-Seidel between them, which is
	/// unusual only in that the textbook order is the other way round. Measured
	/// on 2026-08-27: a pile of eight settles at every pass count from six to
	/// thirty-two, where the one-at-a-time version managed five and was erratic
	/// above it.
	///
	/// @param start - the manifold's first row
	/// @param end - one past its last
	fn together(&mut self, start: usize, end: usize) {
		let mut applied = [0.0_f32; MAX_CONTACTS];

		for (slot, index) in (start..end).enumerate().take(MAX_CONTACTS) {
			let row = self.rows[index];
			let closing = self
				.relative(row.first, row.second, row.from_first, row.from_second)
				.dot(row.normal);

			// the accumulated impulse is what gets clamped, not this step's
			// share of it. Clamping the share instead is what makes a stack
			// sink: a row that needs to pull back a previous overshoot cannot.
			let wanted = -(closing - row.bounce) * row.normal_mass;
			let total = (row.normal_impulse + wanted).max(0.0);

			applied[slot] = total - row.normal_impulse;
			self.rows[index].normal_impulse = total;
		}

		// and only now, once every one of them has been read.
		for (slot, index) in (start..end).enumerate().take(MAX_CONTACTS) {
			let row = self.rows[index];

			self.apply(
				row.first,
				row.second,
				row.from_first,
				row.from_second,
				row.normal * applied[slot],
			);
		}
	}

	/// The friction half of a velocity pass.
	fn rub(&mut self, index: usize) {
		let row = self.rows[index];
		let limit = row.friction * row.normal_impulse;

		for axis in 0..2 {
			let across = row.tangents[axis];
			let sliding = self
				.relative(row.first, row.second, row.from_first, row.from_second)
				.dot(across);

			let wanted = -sliding * row.tangent_mass[axis];
			let total = (row.tangent_impulse[axis] + wanted).clamp(-limit, limit);
			let applied = total - row.tangent_impulse[axis];
			self.rows[index].tangent_impulse[axis] = total;

			self.apply(row.first, row.second, row.from_first, row.from_second, across * applied);
		}
	}

	/// One position pass, run entirely on the pseudo velocities.
	fn separate(&mut self) {
		for index in 0..self.rows.len() {
			let row = self.rows[index];
			let approach = self
				.relative_pseudo(row.first, row.second)
				.dot(row.normal);

			let wanted = -(approach - row.recovery) * row.normal_mass;
			let total = (row.push_impulse + wanted).max(0.0);
			let applied = total - row.push_impulse;
			self.rows[index].push_impulse = total;

			let impulse = row.normal * applied;
			let (first, second) = (row.first, row.second);

			// linear only, and this is the single most load-bearing line in the
			// file. Overlap at a corner is nearly always cheaper to remove by
			// turning a body than by moving it, so a position pass that is
			// allowed to turn things will turn them - and a box in a pile that
			// has been turned a hundredth of a radian to clear a contact
			// presents a worse contact next step, which it is then turned again
			// to clear. The tilt grows exponentially and the pile falls over
			// after about two seconds, with the heights looking perfectly
			// healthy the whole way down. Correcting position without
			// correcting orientation is slower to clear a deep overlap under a
			// tilted body and is the difference between a stack and a mess.
			self.pseudo[first] -= impulse * self.inverse_mass[first];
			self.pseudo[second] += impulse * self.inverse_mass[second];
		}
	}

	/// Applies an impulse to both ends of a contact.
	fn apply(
		&mut self,
		first: usize,
		second: usize,
		from_first: Vec3,
		from_second: Vec3,
		impulse: Vec3,
	) {
		self.velocity[first] -= impulse * self.inverse_mass[first];
		self.velocity[second] += impulse * self.inverse_mass[second];
		self.angular[first] -= self.inverse_inertia[first] * from_first.cross(impulse);
		self.angular[second] += self.inverse_inertia[second] * from_second.cross(impulse);
	}

	/// Moves everything and writes it back into the table.
	fn integrate(&self, bodies: &mut Bodies, dt: f32) {
		let handles: Vec<BodyId> = bodies.iter().map(|(id, _)| id).collect();

		for id in handles {
			let slot = id.slot();

			if self.inverse_mass[slot] <= 0.0 {
				continue;
			}

			let (velocity, angular) = (self.velocity[slot], self.angular[slot]);

			let Some(body) = bodies.get_mut(id) else {
				continue;
			};

			body.velocity = velocity;
			body.angular = angular;

			// the pseudo velocities move the body and are then gone. They are
			// deliberately not added to `body.velocity`: the whole point of
			// them is that climbing out of an overlap is not a thing a body
			// gets to keep.
			let motion = velocity + self.pseudo[slot];

			body.transform.position += motion * dt;
			body.transform.rotation =
				(Quat::from_scaled_axis(angular * dt) * body.transform.rotation).normalize();
		}
	}
}

/// Copies last step's impulses onto the row for the same contact point.
///
/// Matched by identifier and not by position, which is what makes this a
/// *contact* cache rather than a *pair* cache. Position was the first attempt
/// and it fails in the one case that matters: a box that has turned slightly
/// has moved every corner further than any tolerance worth having, drops its
/// whole warm start, and is then more likely to turn again.
///
/// @param row - the row to seed
/// @param previous - what the same pair pushed with last step
fn seed(row: &mut Row, previous: &[Remembered; REMEMBERED]) {
	let Some(found) = previous
		.iter()
		.find(|it| it.real && it.id == row.feature)
	else {
		return;
	};

	row.normal_impulse = found.normal_impulse;
	row.tangent_impulse = found.tangent_impulse;
}

/// Notes any body of a pair that something moving is leaning on.
///
/// @param bodies - the table
/// @param manifold - the pair
/// @param into - where to append whatever should wake
fn disturbed(bodies: &Bodies, manifold: &Manifold, into: &mut Vec<BodyId>) {
	for (sleeper, other) in [(manifold.first, manifold.second), (manifold.second, manifold.first)]
	{
		let asleep = bodies
			.get(sleeper)
			.is_some_and(|body| body.sleeping);

		if asleep && bodies.get(other).is_some_and(disturbs) {
			into.push(sleeper);
		}
	}
}

/// Marks a slot as touching something, if there is such a slot.
fn touch(touching: &mut [bool], slot: usize) {
	if let Some(slot) = touching.get_mut(slot) {
		*slot = true;
	}
}

/// Whether a body can disturb something asleep.
///
/// A static floor cannot, however awake it looks - otherwise nothing resting on
/// one would ever stay asleep.
fn disturbs(body: &Body) -> bool {
	if body.movable() {
		return !body.sleeping;
	}

	matches!(body.kind, BodyKind::Kinematic)
		&& (body.velocity.length_squared() > EPSILON || body.angular.length_squared() > EPSILON)
}

/// The matrix that turns a vector into its cross product with `arm`.
///
/// `skew(a) * b` is `a.cross(b)`, which is what makes an angular term of an
/// effective mass writable as a matrix product rather than as three separate
/// cross products.
fn skew(arm: Vec3) -> Mat3 {
	Mat3::from_cols(
		Vec3::new(0.0, arm.z, -arm.y),
		Vec3::new(-arm.z, 0.0, arm.x),
		Vec3::new(arm.y, -arm.x, 0.0),
	)
}

/// Two unit directions across a normal.
///
/// Any two will do as long as they are perpendicular to it and to each other;
/// friction is isotropic, so which two is arbitrary and only stability cares
/// that the choice does not flip about.
fn across(normal: Vec3) -> [Vec3; 2] {
	let helper = if normal.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
	let first = normal.cross(helper).normalize_or(Vec3::X);

	[first, normal.cross(first).normalize_or(Vec3::Z)]
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_two_tangents_and_the_normal_are_a_frame() {
		for normal in [Vec3::Y, Vec3::X, Vec3::new(0.3, -0.5, 0.81).normalize()] {
			let [first, second] = across(normal);

			assert!(first.dot(normal).abs() < 1.0e-5, "the first is across it");
			assert!(second.dot(normal).abs() < 1.0e-5, "so is the second");
			assert!(first.dot(second).abs() < 1.0e-5, "and they are across each other");
		}
	}

	#[test]
	fn a_static_floor_does_not_keep_waking_what_rests_on_it() {
		let floor = Body::new(
			BodyKind::Static,
			colby_core::abi::Shape::UNIT,
			colby_core::abi::Transform::IDENTITY,
		);

		assert!(!disturbs(&floor), "or nothing would ever settle on one");
	}

	#[test]
	fn a_moving_platform_wakes_what_rides_it() {
		let mut platform = Body::new(
			BodyKind::Kinematic,
			colby_core::abi::Shape::UNIT,
			colby_core::abi::Transform::IDENTITY,
		);

		assert!(!disturbs(&platform), "a platform standing still is a floor");

		platform.velocity = Vec3::X;
		assert!(disturbs(&platform), "and one that is moving is not");
	}

	/// A joint of a stiffness, with everything else left alone.
	fn sprung(stiffness: f32, damping: f32) -> Joint {
		Joint::weld(BodyId::NONE, BodyId::NONE, (Vec3::ZERO, Vec3::ZERO))
			.sprung(stiffness, damping)
	}

	#[test]
	fn a_joint_with_no_stiffness_is_the_constraint_this_table_always_solved() {
		let rigid = Spring::of(&sprung(Joint::RIGID, 1.0), 1.0 / 60.0);

		assert_eq!(rigid, Spring::RIGID, "zero selects the hard constraint, not a limp spring");
		assert!((rigid.push - 1.0).abs() < f32::EPSILON, "which takes all of what it wants");
		assert!(rigid.relax().abs() < f32::EPSILON, "and hands none of it back");
	}

	#[test]
	fn a_spring_takes_part_of_what_a_rigid_joint_would_and_hands_the_rest_back() {
		let spring = Spring::of(&sprung(10.0, 1.0), 1.0 / 60.0);

		assert!(spring.push > 0.0 && spring.push < 1.0, "part of it, got {}", spring.push);
		assert!(
			spring.bias > 0.0 && spring.bias < 1.0,
			"and part of the error, got {}",
			spring.bias
		);
		assert!(
			(spring.push + spring.relax() - 1.0).abs() < 1.0e-6,
			"what it does not take is exactly what it gives back"
		);
	}

	#[test]
	fn a_stiffer_spring_is_closer_to_rigid() {
		let dt = 1.0 / 60.0;
		let soft = Spring::of(&sprung(2.0, 1.0), dt);
		let firm = Spring::of(&sprung(20.0, 1.0), dt);
		let hard = Spring::of(&sprung(2000.0, 1.0), dt);

		assert!(soft.push < firm.push, "a stiffer spring keeps more of its impulse");
		assert!(firm.push < hard.push, "and more again");
		assert!(soft.bias < firm.bias, "and asks for more of the error back");
		assert!(hard.push > 0.99, "and a very stiff one is rigid in all but name");
	}

	#[test]
	fn a_faster_step_makes_the_same_spring_softer() {
		let slow = Spring::of(&sprung(10.0, 1.0), 1.0 / 30.0);
		let fast = Spring::of(&sprung(10.0, 1.0), 1.0 / 240.0);

		// the same number of hertz is a smaller fraction of a shorter step, so
		// numbers taken from an engine running another rate do not transfer.
		assert!(fast.push < slow.push, "the shorter the step, the less rigid a spring is");
	}

	#[test]
	fn a_ceiling_of_zero_is_no_ceiling() {
		let mut total = Vec3::ZERO;
		let applied = hold(&mut total, Vec3::new(0.0, 900.0, 0.0), 0.0);

		assert_eq!(applied, Vec3::new(0.0, 900.0, 0.0), "all of it goes through");
		assert_eq!(total, Vec3::new(0.0, 900.0, 0.0), "and all of it is counted");
	}

	#[test]
	fn a_ceiling_holds_the_total_over_the_step_and_not_one_pass_of_it() {
		let mut total = Vec3::ZERO;
		let mut spent = Vec3::ZERO;

		// ten passes each asking for three quarters of the ceiling. Per pass
		// they would spend seven and a half of it; over the step they spend
		// exactly one.
		for _ in 0..10 {
			let wanted = total + Vec3::Y * 0.75;
			spent += hold(&mut total, wanted, 1.0);
		}

		assert!((total.y - 1.0).abs() < 1.0e-6, "the total stops at the ceiling, got {total}");
		assert!(
			(spent.y - 1.0).abs() < 1.0e-6,
			"and what was actually applied adds up to the same, got {spent}"
		);
	}

	#[test]
	fn a_pass_past_the_ceiling_applies_nothing_at_all() {
		let mut total = Vec3::new(0.0, 1.0, 0.0);
		let applied = hold(&mut total, Vec3::new(0.0, 4.0, 0.0), 1.0);

		assert_eq!(applied, Vec3::ZERO, "there is nothing left to spend");
		assert_eq!(total, Vec3::new(0.0, 1.0, 0.0), "and the total did not move");
	}

	#[test]
	fn a_ceiling_keeps_the_direction_the_constraint_asked_for() {
		let mut total = Vec3::ZERO;
		let wanted = Vec3::new(3.0, 4.0, 0.0);
		let applied = hold(&mut total, wanted, 1.0);

		assert!((applied.length() - 1.0).abs() < 1.0e-6, "shortened to the ceiling");
		assert!(
			applied
				.normalize()
				.abs_diff_eq(wanted.normalize(), 1.0e-6),
			"and pointing where it was asked to"
		);
	}
}
