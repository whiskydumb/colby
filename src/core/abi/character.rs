//! Collide and slide: moving a box through a world without going through it.
//!
//! The half of a character controller that is geometry rather than taste. What
//! is here is the part every game wants the same way and cannot afford to
//! disagree about - clip the motion against what it runs into, slide along it,
//! climb a lip that is short enough, and answer whether there is ground
//! underneath. What is *not* here is everything a game is: how fast, how
//! quickly it gets there, what a jump is worth, whether crouching is a thing,
//! where the camera goes. Those are numbers with opinions in them, and they
//! belong in the game.
//!
//! Three decisions are written into the shape of this module.
//!
//! **It is a function over [`World::trace_box`], not an entry in the physics
//! table.** The table is two function pointers because a query is the only
//! thing that needs the host's broadphase and its baked collision meshes -
//! @ref [`physics`](crate::abi::physics). A move is four queries and some
//! arithmetic, so it needs nothing the game cannot already reach, and putting
//! it here rather than behind a third pointer keeps that argument intact. It
//! also means the host, the game module and anything else linking
//! `colby_core` all run the same code, which is the property a networked
//! prediction eventually stands on: a client and a server that move a player
//! by different arithmetic disagree, and no amount of state fixes that.
//!
//! **The box is axis-aligned and there is no capsule.** That is what
//! [`World::trace_box`] sweeps, and it is why [`Motion::step`] exists: a
//! capsule climbs a small lip because of its shape, and a box has to be told
//! it may. The explicit version is the one a level designer can reason about -
//! a step is a number in units rather than a consequence of a radius.
//!
//! **Nothing here integrates anything.** Gravity is not applied, friction is
//! not applied, no key is read. A game adds to [`Motion::velocity`] and calls
//! this with what it wants to happen; what comes back is what actually
//! happened. Wiring it to a body is the game's too - the usual shape is a
//! [`BodyKind::Kinematic`](crate::abi::BodyKind::Kinematic) body written from
//! the result, so that props rest on the character and triggers notice it.

use super::{
	entity::Transform,
	physics::{BodyId, Layers, MAX_IGNORED, TraceInfo, TraceResult},
};
use crate::{abi::World, glam::Vec3};

/// How far a moving box is kept off whatever it lands on.
///
/// A box placed exactly on a surface starts the next sweep already touching
/// it, and a sweep that starts touching has nothing to report but the place it
/// began. A millimeter of daylight is what keeps every move after the first
/// one a real question.
const SKIN: f32 = 1.0e-3;

/// How many surfaces one move may slide along before it gives up.
///
/// Four, because the case that needs the fourth is a box wedged into a corner
/// of three planes and looking for a fourth to leave by. Past that the honest
/// answer is that it is stuck, and stopping is better than the alternative,
/// which is finding a way out through a wall.
const MAX_SLIDES: usize = 4;

/// How far below itself a box looks for ground, in units.
///
/// And how far *above* itself the looking starts, so that a support which
/// has risen into the box since the last step is still a surface to come to
/// rest on rather than a thing the box is inside of. Also how far it is
/// pulled down onto ground it finds. Walking off the top of
/// a stair without this leaves the box airborne for a step and arriving at the
/// next one falling, which reads as a bounce down every staircase.
const PROBE: f32 = 0.06;

/// How tall a lip a box climbs unless it is told otherwise.
pub const STEP: f32 = 0.35;

/// The steepest ground a box stands on unless it is told otherwise.
///
/// The cosine of the angle from straight up, so a larger number is a flatter
/// limit. This one is forty-five degrees, which is where the cosine happens to
/// be a constant with a name.
pub const GROUND: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// What to move, and what it is allowed to do on the way.
///
/// `#[repr(C)]` and `Copy` like the trace types beside it, and **not** `Pod`
/// for the same reason: glam is built without bytemuck, so nothing holding a
/// `Vec3` can be. The arena is the only thing that needs `Pod`, and a game
/// keeps its handles there rather than its motion.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motion {
	/// Where the box's middle is now.
	pub position: Vec3,

	/// How fast it is going, in units a second. World space.
	///
	/// Everything the game decided: input, gravity, a jump, whatever else. This
	/// is read and never written; what comes back has it clipped against
	/// everything the move ran into.
	pub velocity: Vec3,

	/// The box's half-extents.
	pub extents: Vec3,

	/// How long this move lasts, in seconds. Usually `World::dt`.
	pub dt: f32,

	/// How tall a lip the box may climb without leaving the ground.
	///
	/// Applied only when the box is already standing on something: a lip
	/// climbed in mid-air is a box walking up the outside of a wall.
	pub step: f32,

	/// The steepest surface the box will call ground.
	///
	/// The cosine of the angle from straight up. A surface flatter than this is
	/// something to stand on; anything steeper is a wall to slide along, even
	/// if it is underneath.
	pub ground: f32,

	/// Which layers the box is on, and which it collides with.
	///
	/// Handed straight to every sweep this move makes, so a character that
	/// walks through a layer says so once here rather than at each of them.
	/// [`Layers::ALL`] unless it says otherwise.
	pub layers: Layers,

	/// Bodies the move is blind to. Its own, at least.
	ignore: [BodyId; MAX_IGNORED],

	/// How many of `ignore` are set.
	ignored: u32,
}

impl Motion {
	/// A move, with the usual limits.
	///
	/// @param position - where the box's middle is
	/// @param velocity - how fast it is going, in units a second
	/// @param extents - its half-extents
	/// @param dt - how long the move lasts
	#[must_use]
	pub const fn new(position: Vec3, velocity: Vec3, extents: Vec3, dt: f32) -> Self {
		Self {
			position,
			velocity,
			extents,
			dt,
			step: STEP,
			ground: GROUND,
			layers: Layers::ALL,
			ignore: [BodyId::NONE; MAX_IGNORED],
			ignored: 0,
		}
	}

	/// The same move, climbing a different lip.
	///
	/// @param height - how tall a lip, in units; zero refuses every one
	#[must_use]
	pub const fn stepping(mut self, height: f32) -> Self {
		self.step = height;

		self
	}

	/// The same move, standing on a different limit.
	///
	/// @param cosine - the cosine of the steepest angle from straight up
	#[must_use]
	pub const fn standing(mut self, cosine: f32) -> Self {
		self.ground = cosine;

		self
	}

	/// The same move, on other layers.
	///
	/// @param layers - which layers the box is on and which it collides with
	#[must_use]
	pub const fn layered(mut self, layers: Layers) -> Self {
		self.layers = layers;

		self
	}

	/// The same move, blind to one more body.
	///
	/// Chainable, and it behaves exactly as
	/// [`TraceInfo::ignoring`](crate::abi::TraceInfo::ignoring) does, because
	/// it is the same list going to the same queries: past [`MAX_IGNORED`] the
	/// extra handles are dropped.
	///
	/// @param body - what to pretend is not there; the character's own, first
	#[must_use]
	pub fn ignoring(mut self, body: BodyId) -> Self {
		let count = usize::try_from(self.ignored).unwrap_or(MAX_IGNORED);

		if count < MAX_IGNORED {
			self.ignore[count] = body;
			self.ignored = self.ignored.saturating_add(1);
		}

		self
	}

	/// The bodies this move is blind to.
	#[must_use]
	pub fn ignored(&self) -> &[BodyId] {
		let count = usize::try_from(self.ignored).unwrap_or(0);

		&self.ignore[..count.min(MAX_IGNORED)]
	}
}

/// What actually happened.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moved {
	/// Where the box's middle ended up.
	pub position: Vec3,

	/// How fast it is going now.
	///
	/// What it was, with the component into every surface it slid along taken
	/// out. A game writes this back, which is what stops a character walking
	/// into a wall from building up speed it spends the moment the wall ends.
	pub velocity: Vec3,

	/// Whether there is something under it flat enough to stand on.
	pub grounded: bool,

	/// The normal of that ground, or [`Vec3::ZERO`].
	pub ground_normal: Vec3,

	/// What it is standing on, or [`BodyId::NONE`].
	pub ground_body: BodyId,

	/// How many surfaces the move slid along.
	///
	/// Zero is a move that met nothing. [`MAX_SLIDES`] means it ran out of
	/// tries, which is the shape of being stuck.
	pub slides: u32,

	/// Whether it climbed a lip on the way.
	pub stepped: bool,
}

impl Moved {
	/// A move that went nowhere and hit nothing.
	///
	/// @param motion - what was asked for
	#[must_use]
	pub const fn none(motion: &Motion) -> Self {
		Self {
			position: motion.position,
			velocity: motion.velocity,
			grounded: false,
			ground_normal: Vec3::ZERO,
			ground_body: BodyId::NONE,
			slides: 0,
			stepped: false,
		}
	}

	/// The transform a box at this position would have.
	///
	/// The usual next thing a game does with one of these, and it is here so
	/// that "where the character is" has one spelling.
	#[must_use]
	pub fn transform(&self) -> Transform { Transform::at(self.position) }
}

/// Moves a box through the world, sliding along whatever is in the way.
///
/// The whole of it is four kinds of query. The ordinary move slides along up
/// to [`MAX_SLIDES`] surfaces. If that ran into something and the box was
/// standing on ground, the same move is tried again from a lip's height up and
/// kept if it got further. Then a short probe downwards answers what the box is
/// standing on now and pulls it onto it.
///
/// @param world - the bodies to move through
/// @param motion - the box, where it is, and what it is allowed to do
/// @return where it ended up and what it is on
#[must_use]
pub fn move_and_slide(world: &World, motion: &Motion) -> Moved {
	let below = probe(world, motion, motion.position);
	let (flat, velocity, slides) = slide(world, motion, motion.position, motion.velocity);

	// a lip is only climbed from the ground. In mid-air the same three traces
	// are a box walking up the outside of a wall.
	let climbed = (slides > 0 && below.grounded)
		.then(|| climb(world, motion, flat))
		.flatten();

	let stepped = climbed.is_some();
	let position = climbed.unwrap_or(flat);
	let ground = probe(world, motion, position);

	Moved {
		position: ground.position,
		velocity,
		grounded: ground.grounded,
		ground_normal: ground.normal,
		ground_body: ground.body,
		slides,
		stepped,
	}
}

/// What a downward probe found.
#[derive(Clone, Copy, Debug)]
struct Ground {
	/// Where the box ends up once it is pulled onto what it found.
	position: Vec3,

	/// Whether that is something it can stand on.
	grounded: bool,

	/// The surface normal there.
	normal: Vec3,

	/// What the surface belongs to.
	body: BodyId,
}

/// Slides a box along everything it runs into.
///
/// @param world - the bodies to move through
/// @param motion - for the extents, the step length and the ignore list
/// @param from - where to start
/// @param velocity - how fast, in units a second
/// @return where it ended, the velocity with every surface taken out of it,
/// and how many it slid along
fn slide(world: &World, motion: &Motion, from: Vec3, velocity: Vec3) -> (Vec3, Vec3, u32) {
	let mut position = from;
	let mut velocity = velocity;
	let mut remaining = velocity * motion.dt;
	let mut slides = 0;

	for _ in 0..MAX_SLIDES {
		if remaining.length_squared() <= f32::EPSILON {
			break;
		}

		let result = cast(world, motion, position, position + remaining);
		position = result.end;

		if !result.hit {
			return (position, velocity, slides);
		}

		// off the surface by a hair, so the next sweep is a question rather
		// than an answer about where it already is.
		position += result.normal * SKIN;
		remaining = clip(remaining * (1.0 - result.fraction), result.normal);
		velocity = clip(velocity, result.normal);
		slides += 1;
	}

	(position, velocity, slides)
}

/// Tries the same move again from a lip's height up.
///
/// Three sweeps, which is the classic answer and still the cheapest one: up by
/// the step, the move again from there, then back down. Kept only if it got
/// further along the ground than the ordinary move did, so a box that fails to
/// climb is left exactly where sliding put it.
///
/// @param world - the bodies to move through
/// @param motion - the move that was asked for
/// @param flat - where the ordinary slide ended
/// @return where the climb ended, or `None` if it was not worth keeping
fn climb(world: &World, motion: &Motion, flat: Vec3) -> Option<Vec3> {
	let lift = Vec3::Y * motion.step;

	if motion.step <= 0.0 {
		return None;
	}

	let up = cast(world, motion, motion.position, motion.position + lift);
	let raised = up.end - Vec3::Y * SKIN;
	let (across, ..) = slide(world, motion, raised, motion.velocity);

	// down by what was climbed and then some, so that the box lands on the lip
	// rather than hanging over it.
	let down = cast(world, motion, across, across - lift - Vec3::Y * PROBE);

	if !down.hit || down.normal.y < motion.ground {
		// nothing under it, or nothing it could stand on. A climb that ends in
		// mid-air is worse than not climbing.
		return None;
	}

	let landed = down.end + Vec3::Y * SKIN;
	let gained = (landed - flat) * Vec3::new(1.0, 0.0, 1.0);

	(gained.length_squared() > SKIN * SKIN).then_some(landed)
}

/// Looks for ground under a box, and pulls it down onto what it finds.
///
/// @param world - the bodies to look at
/// @param motion - for the extents and the ignore list
/// @param position - where the box's middle is
/// @return what is underneath and where the box sits on it
fn probe(world: &World, motion: &Motion, position: Vec3) -> Ground {
	// from a probe's height above where the box is, not from where it is. A
	// support that rose into the box between one step and the next leaves it
	// overlapping, and a sweep that begins inside something has nothing to
	// report but the place it began - so the box would either be lifted by a
	// blind skin every step until it climbed away, or left where it was until
	// the platform swallowed it. Starting clear of the surface asks the
	// question that has a real answer: where does the box come to rest.
	let above = position + Vec3::Y * PROBE;
	let result = cast(world, motion, above, position - Vec3::Y * PROBE);

	if !result.hit || result.normal.y < motion.ground {
		return Ground {
			position,
			grounded: false,
			normal: Vec3::ZERO,
			body: BodyId::NONE,
		};
	}

	Ground {
		// left where it is if even a probe's height up is inside something,
		// which is a box buried rather than a box standing: pushing out of
		// things is the slide's, and it has already had its turn.
		position: if result.started_solid {
			position
		} else {
			result.end + Vec3::Y * SKIN
		},
		grounded: true,
		normal: result.normal,
		body: result.body,
	}
}

/// One sweep of the box, blind to what the move is blind to.
///
/// @param world - the bodies to sweep against
/// @param motion - for the extents and the ignore list
/// @param from - where the box's middle starts
/// @param to - where it would end
fn cast(world: &World, motion: &Motion, from: Vec3, to: Vec3) -> TraceResult {
	let mut info = TraceInfo::swept(from, to, motion.extents).layered(motion.layers);

	for &body in motion.ignored() {
		info = info.ignoring(body);
	}

	world.trace_box(&info)
}

/// A vector with everything pointing into a surface taken out of it.
///
/// @param vector - what to clip
/// @param normal - the surface's normal, pointing out of it
fn clip(vector: Vec3, normal: Vec3) -> Vec3 {
	let into = vector.dot(normal);

	if into >= 0.0 {
		// already going away from it. Clipping here would push the vector out
		// of a surface it is leaving, which is how a box climbs a wall.
		return vector;
	}

	vector - normal * into
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A move of a small box, one step long.
	fn moving(velocity: Vec3) -> Motion {
		Motion::new(Vec3::ZERO, velocity, Vec3::splat(0.25), 1.0 / 60.0)
	}

	#[test]
	fn a_world_with_no_solver_lets_a_box_go_where_it_asked() {
		let world = World::new();
		let motion = moving(Vec3::new(3.0, 0.0, 0.0));
		let moved = move_and_slide(&world, &motion);

		assert!(
			moved
				.position
				.abs_diff_eq(Vec3::new(0.05, 0.0, 0.0), 1.0e-4),
			"the stub reports a clean miss, so the whole step happens, got {}",
			moved.position
		);
		assert!(!moved.grounded, "and there is nothing under it either");
		assert_eq!(moved.slides, 0, "having met nothing on the way");
	}

	#[test]
	fn clipping_takes_out_what_points_into_a_surface_and_no_more() {
		let along = clip(Vec3::new(1.0, 0.0, -1.0), Vec3::Z);

		assert!(
			along.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1.0e-5),
			"the part into the wall goes and the part along it stays, got {along}"
		);
	}

	#[test]
	fn clipping_leaves_alone_what_is_already_leaving() {
		let away = Vec3::new(1.0, 0.0, 2.0);

		assert!(
			clip(away, Vec3::Z).abs_diff_eq(away, 1.0e-5),
			"a surface being left behind does not get to push"
		);
	}

	#[test]
	fn a_move_that_asks_for_nothing_reports_where_it_already_was() {
		let world = World::new();
		let motion = moving(Vec3::ZERO);
		let moved = move_and_slide(&world, &motion);

		assert!(moved.position.abs_diff_eq(Vec3::ZERO, 1.0e-6), "it went nowhere");
		assert_eq!(moved.slides, 0, "and slid along nothing");
	}

	#[test]
	fn an_ignore_list_fills_up_rather_than_overflowing() {
		let mut motion = moving(Vec3::X);

		for _ in 0..MAX_IGNORED + 3 {
			motion = motion.ignoring(BodyId::NONE);
		}

		assert_eq!(motion.ignored().len(), MAX_IGNORED, "it stops at the bound");
	}

	#[test]
	fn the_limits_are_the_ones_asked_for() {
		let motion = moving(Vec3::X).stepping(0.5).standing(0.9);

		assert!((motion.step - 0.5).abs() < f32::EPSILON, "the lip it will climb");
		assert!((motion.ground - 0.9).abs() < f32::EPSILON, "and the slope it will stand on");
	}
}
