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
//!
//! **And a fourth, which is why the three above were worth being strict
//! about.** [`replay`] runs a *run* of commands rather than one move, and
//! [`Drift`] is what stops a correction being a jump. Between them they are
//! the whole of what a client needs to move itself and be told it was wrong,
//! and they are here rather than in the networking crate because they are
//! arithmetic over a world - the same arithmetic the machine that decides runs,
//! out of the same file. Two machines that move a player by different code
//! disagree, and no amount of state fixes that.

use super::{
	entity::Transform,
	net::Command,
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

/// The most steps one command may be said to cover.
///
/// A command carries the step it was made on, and the difference between two
/// of them is how long the later one lasts - so a command claiming a step a
/// long way past the one before it is a command asking to be moved a long way
/// in one go. **That number is the sender's**, and on the machine that decides
/// the sender is somebody else, so it is a number to bound rather than
/// believe: without this a client could ask for one command covering an hour
/// and arrive on the far side of the map.
///
/// Twelve steps is a fifth of a second, which is generous for the case it
/// exists for - a client whose own frame hitched and made no command for a
/// moment. Past it the move is cut short rather than refused, because refusing
/// is the one answer that lets a hitch cost a player their input.
///
/// **This bounds one command and nothing bounds a run**, which is worth saying
/// because the two look like the same guard and are not. A run is whatever the
/// ring hands over, so at a depth of thirty-two and a ceiling of twelve, one
/// call can cover three hundred and eighty-four steps - six and a half seconds
/// of movement in one step of the machine that decides. A client whose
/// commands really did pile up that far behind is a client that has to be let
/// catch up, so bounding the *run* would break the case this exists for.
/// Bounding how far a player may travel per step of the world is a different
/// question with a different answer - it is the one lag compensation asks, and
/// it is deferred with the rest of that.
pub const MAX_CATCH_UP: u16 = 12;

/// How long a wrong guess takes to stop showing, in seconds.
///
/// A tenth, which is the default the system this is modeled on ships and is
/// about as long as a correction can be smeared before the smear itself is
/// what looks wrong.
pub const DECAY: f32 = 0.1;

/// Runs a run of commands over a world, threading the state between them.
///
/// The whole of what re-running a move costs, and it is deliberately *not* the
/// whole of a move: what each command means - how fast, how high a jump is
/// worth, what gravity does, whether crouching is a thing - is the game's, and
/// arrives as a closure. What is here is the part both machines have to agree
/// about to the bit: how long each command lasts, and that the result of one
/// is where the next starts from.
///
/// **It is the same function on both machines and that is the point.** A host
/// walking the commands one client sent, and that client re-running the ones
/// the host has not confirmed, are the same loop over the same list from a
/// different starting place. The day they are two loops is the day they
/// disagree about something nobody can find.
///
/// **The first command's length is measured from `since`**, which is the step
/// the starting state is *as of*. On a host that is the step of the last
/// command it ran for this peer; on a client it is the step of the last one the
/// host confirmed. A gap there is a client that made no command for a while,
/// and it is honored up to [`MAX_CATCH_UP`] and cut short past it.
///
/// **There is no previous move for the first command of a run, and `wish` is
/// told so rather than handed one.** That is the whole reason the argument is
/// an [`Option`]: a fabricated [`Moved::none`] would say `grounded: false`,
/// which is indistinguishable from a character that really is in the air, and
/// the two machines start their runs in different places - a host runs the one
/// or two commands that just arrived, a client re-runs a whole round trip's
/// worth. Handing both a made-up airborne predecessor is how a jump comes out
/// differently on the two machines forever.
///
/// What carries *across* a run is the game's to keep, and that is not a
/// shortcoming: the state a character carries between moves is exactly the
/// state a snapshot has to replicate anyway, so it belongs in the block the
/// world keeps per player. The system this is modeled on threads the same fact
/// through the same place.
///
/// @param world - what to move through
/// @param start - the state to begin from, whose `dt` is one step
/// @param since - the step that state is as of
/// @param commands - what to run, oldest first
/// @param wish - given a command, what came of the one before it *within this
/// run* or nothing for the first of them, and the motion to fill in, sets the
/// velocity that command asks for
/// @return where the last of them ended up, or a move that went nowhere for an
/// empty run
pub fn replay<Wish>(
	world: &World,
	start: &Motion,
	since: u64,
	commands: &[Command],
	mut wish: Wish,
) -> Moved
where
	Wish: FnMut(&Command, Option<&Moved>, &mut Motion),
{
	let mut motion = *start;
	let mut before: Option<Moved> = None;
	let mut previous = since;

	for command in commands {
		// **a step is a number the sender picks and nothing upstream checks
		// it.** The ring these came out of orders and deduplicates by
		// `Command::number` alone and never reads the step at all, so two
		// commands may claim the same moment and one may claim to happen
		// before the one in front of it. A length of nought is not an answer -
		// that is a frame of somebody's input dropped in silence - so the
		// floor is one step.
		let span = u16::try_from(command.step.saturating_sub(previous))
			.unwrap_or(MAX_CATCH_UP)
			.clamp(1, MAX_CATCH_UP);

		motion.dt = start.dt * f32::from(span);

		// the speed the last move ended at rather than the one it was asked
		// for, so a `wish` that adds to what is there adds to what happened.
		// Overwritten by almost every `wish`; what it stops is the *raw* value
		// of the previous ask being carried into the next move by a game that
		// only sometimes sets it.
		if let Some(was) = before.as_ref() {
			motion.velocity = was.velocity;
		}

		wish(command, before.as_ref(), &mut motion);

		let moved = move_and_slide(world, &motion);

		motion.position = moved.position;
		before = Some(moved);

		// **never backwards**, which is the guard the step field being the
		// sender's makes necessary. Letting it go back would not cost the
		// command that did it - that one is floored at a step - it would cost
		// the *next* one, whose gap is then measured from the low mark and can
		// be a full ceiling's worth. Alternating a low step with a high one
		// would then average six steps a command instead of one, which is the
		// ceiling defeated by the thing it was written for.
		previous = previous.max(command.step);
	}

	before.unwrap_or_else(|| Moved::none(start))
}

/// The difference between what was drawn and what turned out to be true,
/// fading.
///
/// **A correction is a fact and a jump is a decision**, and this is the one
/// that keeps the second from following the first. A client guesses where it
/// is, the machine that decides says otherwise, and the honest thing to do
/// with the difference is not to teleport a player who did nothing wrong: the
/// prediction is replaced at once, and what is *drawn* keeps the old place and
/// slides to the new one over [`DECAY`].
///
/// ```text
///   drift.correct(drawn_last_frame, predicted_now);   // when a correction lands
///   ..
///   let show = predicted + drift.offset();            // draw, then let go
///   drift.advance(world.dt);
/// ```
///
/// The order of those last two is the whole of the bit-exact claim below: read
/// the offset and *then* advance, and the frame a correction lands on draws
/// exactly where the frame before it did. Advance first and that frame has
/// already given up a step's worth, which is a small jump rather than none.
///
/// **A correction is smeared however big it is**, which is right for the
/// ordinary case and wrong for one: a host that teleports a player has not
/// mispredicted anything, and sliding them across the map over a tenth of a
/// second is worse than putting them there. Nothing on the wire says which
/// a change was - the same gap the proxy path has - so the game is the only
/// thing that can know, and skipping the `correct` is how it says so.
///
/// Two things fall out of taking the *drawn* position rather than the previous
/// prediction. The drawn position already carries whatever was left of an
/// earlier correction, so a correction arriving on top of one still fading
/// needs no special case - it absorbs the remainder by construction. And the
/// picture is continuous across the moment of correction to the bit, which is
/// the property worth having: a player sees a drift, never a step.
///
/// `Pod`, and the fields are an array and a float rather than a `Vec3` and a
/// `Duration`, because this is a thing a game keeps in its own arena and the
/// arena is bytes. @ref [`state`](crate::abi::state).
#[repr(C)]
#[derive(
	Clone, Copy, Debug, Default, PartialEq, crate::bytemuck::Pod, crate::bytemuck::Zeroable,
)]
pub struct Drift {
	/// The whole of the error, as it stood when it was noticed.
	error: [f32; 3],

	/// How much of [`DECAY`] is left, in seconds.
	left: f32,
}

// sixteen bytes with no padding, and the number is here so that a field added
// to a thing a game keeps in a fixed arena is a thing somebody decided.
const _: () = assert!(size_of::<Drift>() == 16, "a drift is three floats and a clock");

impl Drift {
	/// Nothing to correct.
	pub const NONE: Self = Self { error: [0.0; 3], left: 0.0 };

	/// Notes that what was drawn and what is now believed are not the same
	/// place.
	///
	/// @param drawn - where the player was shown last frame, offset and all
	/// @param predicted - where the replay now says they are
	pub fn correct(&mut self, drawn: Vec3, predicted: Vec3) {
		self.error = (drawn - predicted).to_array();
		self.left = DECAY;
	}

	/// Lets some of it go.
	///
	/// @param dt - how long has passed, in seconds
	pub fn advance(&mut self, dt: f32) { self.left = (self.left - dt).max(0.0); }

	/// How far from the prediction to draw, right now.
	///
	/// [`Vec3::ZERO`] once it has faded, and faded or not it costs a divide, a
	/// multiply and a compare - which is why nothing here has to be turned off
	/// when there is no correction to show.
	#[must_use]
	pub fn offset(&self) -> Vec3 {
		// guarded rather than trusted: `DECAY` is a constant today and a
		// console variable the first time somebody wants to see the smear, and
		// a division by nought here would put a `NaN` into a transform, which
		// is the kind of value that spreads.
		if DECAY <= 0.0 {
			return Vec3::ZERO;
		}

		Vec3::from_array(self.error) * (self.left / DECAY)
	}

	/// Whether a correction is still fading.
	///
	/// @note: about the *clock* rather than about the picture, and the two part
	/// company for a correction of nothing at all - a
	/// [`correct`](Self::correct) whose two places are the same leaves this
	/// true for a tenth of a second with a zero offset. Left that way rather
	/// than comparing an error against nought, because the question worth
	/// answering cheaply is "is a correction in progress", and a game that
	/// wants to know whether anything is visible has [`offset`](Self::offset).
	#[must_use]
	pub fn showing(&self) -> bool { self.left > 0.0 }
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
	use super::{super::physics::Physics, *};

	/// A move of a small box, one step long.
	fn moving(velocity: Vec3) -> Motion {
		Motion::new(Vec3::ZERO, velocity, Vec3::splat(0.25), 1.0 / 60.0)
	}

	/// A command, with four numbers that are four different numbers.
	///
	/// The step is said rather than derived from the number, because the two
	/// are different fields and the whole of what `replay` does with them is
	/// take their difference.
	fn asked(number: u32, step: u64) -> Command {
		Command {
			step,
			number,
			buttons: 1 << (number % 5),
			yaw: 0.25,
			pitch: -0.75,
		}
	}

	/// What one command's move looked like from inside the wish: how long it
	/// lasted, and the speed the command before it came to.
	type Seen = (f32, Option<Vec3>);

	/// Where the wall in [`walled`] stands, along x.
	const WALL: f32 = 6.0;

	/// A trace that reports a wall across x, so a move is really clipped.
	///
	/// Every other test in this file runs against [`Physics::STUB`], which
	/// reports a clean miss - and a move that hits nothing comes back with the
	/// velocity it was asked for, so a world with no solver in it cannot tell
	/// "carry what the last move came to" from "carry what the last move was
	/// asked for". They are the same number until something clips one.
	///
	/// # Safety
	///
	/// As [`TraceFn`]: both pointers are live for the duration of the call.
	unsafe extern "C-unwind" fn wall(
		_context: *mut core::ffi::c_void,
		_bodies: *const crate::abi::Bodies,
		info: *const TraceInfo,
	) -> TraceResult {
		// SAFETY: the caller hands over a live `TraceInfo` for the call, which
		// is the whole of this function's contract.
		let info = unsafe { &*info };
		let (from, to) = (info.start, info.end);
		let face = WALL - info.extents.x;
		let miss = TraceResult {
			hit: false,
			start: from,
			end: to,
			fraction: 1.0,
			normal: Vec3::ZERO,
			started_solid: false,
			ended_solid: false,
			body: BodyId::NONE,
			entity: crate::abi::EntityId::NONE,
		};

		if from.x >= face || to.x <= face {
			return miss;
		}

		let along = to.x - from.x;
		let fraction = ((face - from.x) / along).clamp(0.0, 1.0);

		TraceResult {
			hit: true,
			end: from + (to - from) * fraction,
			fraction,
			// pointing back out of it, which is what `clip` takes the motion
			// into the surface out of.
			normal: -Vec3::X,
			..miss
		}
	}

	/// A world with that wall in it.
	fn walled() -> World {
		let mut world = World::new();

		world.install_physics(Physics::new(core::ptr::null_mut(), wall, wall));
		world
	}

	/// Where every replay in these tests begins.
	///
	/// Deliberately not the origin and not on an axis, so a run that started
	/// where it was told and one that started at nought are different answers
	/// rather than the same one.
	const FROM: Vec3 = Vec3::new(4.0, -2.0, 7.0);

	/// A replay whose wish is a fixed velocity, and a note of what it saw.
	///
	/// Returns where it ended and, for each command, the length of the move and
	/// the previous result it was handed - which is the only way to see that
	/// one command's result is where the next starts from, and that the first
	/// of a run is handed nothing at all.
	fn ran(commands: &[Command], since: u64, velocity: Vec3) -> (Moved, Vec<Seen>) {
		let world = World::new();
		let start = Motion::new(FROM, Vec3::ZERO, Vec3::splat(0.25), 1.0 / 60.0);
		let mut seen = Vec::new();
		let moved = replay(&world, &start, since, commands, |_command, before, motion| {
			seen.push((motion.dt, before.map(|was| was.velocity)));
			motion.velocity = velocity;
		});

		(moved, seen)
	}

	#[test]
	fn a_run_of_commands_ends_where_the_same_moves_one_at_a_time_would() {
		// three commands a step apart, at a speed whose product with a step is
		// not a round number - so a move counted twice or missed once is a
		// different answer rather than the same one.
		let commands = [asked(4, 100), asked(5, 101), asked(6, 102)];
		let (moved, seen) = ran(&commands, 99, Vec3::new(3.0, 0.0, -1.5));

		assert_eq!(seen.len(), 3, "one move per command");

		for (dt, _) in &seen {
			assert!((dt - 1.0 / 60.0).abs() < 1.0e-7, "each lasts one step, got {dt}");
		}

		// three steps at three a second along x, and half that back along z,
		// from where the caller said rather than from the origin.
		assert!(
			moved
				.position
				.abs_diff_eq(FROM + Vec3::new(3.0 * 3.0 / 60.0, 0.0, -1.5 * 3.0 / 60.0), 1.0e-5),
			"got {}",
			moved.position
		);
	}

	#[test]
	fn the_result_of_one_command_is_where_the_next_starts_from() {
		let commands = [asked(1, 10), asked(2, 11)];
		let (_, seen) = ran(&commands, 9, Vec3::new(6.0, 0.0, 0.0));

		assert_eq!(
			seen[0].1, None,
			"the first of a run is handed nothing, rather than a made-up move that reads as \
			 airborne"
		);
		assert!(
			seen[1]
				.1
				.expect("the second has one")
				.abs_diff_eq(Vec3::new(6.0, 0.0, 0.0), 1.0e-5),
			"and it is what the first came to, got {:?}",
			seen[1].1
		);
	}

	#[test]
	fn a_command_a_way_past_the_one_before_it_lasts_that_much_longer() {
		// four steps of nothing and then a command: the client made none while
		// its own frame was busy, and the move it did make covers the gap.
		let commands = [asked(1, 205)];
		let (moved, seen) = ran(&commands, 201, Vec3::new(1.0, 0.0, 0.0));

		assert!((seen[0].0 - 4.0 / 60.0).abs() < 1.0e-7, "four steps, got {}", seen[0].0);
		assert!(
			moved
				.position
				.abs_diff_eq(FROM + Vec3::new(4.0 / 60.0, 0.0, 0.0), 1.0e-5),
			"and it went four steps' worth, got {}",
			moved.position
		);
	}

	/// A run whose commands are gaps of *different* sizes, which is the only
	/// shape that can tell a length from a length that compounds.
	#[test]
	fn each_command_of_a_run_is_measured_from_the_one_before_rather_than_the_start() {
		// two steps, then three, then one: no two the same, none of them one
		// until the last, and their sum is not any of them.
		let commands = [asked(1, 502), asked(2, 505), asked(3, 506)];
		let (moved, seen) = ran(&commands, 500, Vec3::new(1.0, 0.0, 0.0));
		let spans: Vec<f32> = seen.iter().map(|(dt, _)| dt * 60.0).collect();

		assert!(
			spans
				.iter()
				.zip([2.0, 3.0, 1.0])
				.all(|(got, want)| (got - want).abs() < 1.0e-5),
			"two then three then one, got {spans:?}"
		);
		// six steps altogether at one a second. A length that multiplied into
		// the one before it would give two, then six, then six.
		assert!(
			moved
				.position
				.abs_diff_eq(FROM + Vec3::new(6.0 / 60.0, 0.0, 0.0), 1.0e-5),
			"got {}",
			moved.position
		);
	}

	/// A step that goes backwards must not lend the command after it a run-up.
	#[test]
	fn a_step_that_goes_backwards_does_not_stretch_the_command_after_it() {
		// what a sender picks freely: a high step, then a low one, then high
		// again. The ring these came out of orders by number and never looks
		// at the step at all, so all three are taken.
		let commands = [asked(1, 1000), asked(2, 0), asked(3, 1001)];
		let (moved, seen) = ran(&commands, 999, Vec3::new(1.0, 0.0, 0.0));
		let spans: Vec<f32> = seen.iter().map(|(dt, _)| dt * 60.0).collect();

		assert!(
			spans
				.iter()
				.zip([1.0, 1.0, 1.0])
				.all(|(got, want)| (got - want).abs() < 1.0e-5),
			"a step each and no run-up, got {spans:?}"
		);
		assert!(
			moved
				.position
				.abs_diff_eq(FROM + Vec3::new(3.0 / 60.0, 0.0, 0.0), 1.0e-5),
			"three steps in three commands, got {}",
			moved.position
		);
	}

	#[test]
	fn a_command_claiming_an_hour_is_cut_short_rather_than_believed() {
		// the numbers themselves, because every assertion below spells them
		// symbolically and would move with them. A ceiling that let a command
		// cover a second, or a smear that lasted one, is not a failure any of
		// them could see.
		assert_eq!(MAX_CATCH_UP, 12, "a fifth of a second at sixty steps a second");
		assert!((DECAY - 0.1).abs() < f32::EPSILON, "and a tenth of one to stop showing");

		// the far side of the map in one move, which is what a sender who
		// picks this number freely would ask for.
		let commands = [asked(1, u64::MAX)];
		let (moved, seen) = ran(&commands, 0, Vec3::new(1.0, 0.0, 0.0));

		assert!(
			(seen[0].0 - f32::from(MAX_CATCH_UP) / 60.0).abs() < 1.0e-7,
			"cut to the ceiling, got {}",
			seen[0].0
		);
		assert!(
			moved.position.x - FROM.x < 1.0,
			"so it went nowhere near, got {}",
			moved.position.x - FROM.x
		);

		// a gap that fits in the number the span is read into, so what cuts it
		// is the ceiling rather than the conversion in front of it. A hundred
		// steps is what a client whose frame stopped for a second and a half
		// would really ask for.
		let big = [asked(1, 100)];
		let (_, seen) = ran(&big, 0, Vec3::X);

		assert!(
			(seen[0].0 - f32::from(MAX_CATCH_UP) / 60.0).abs() < 1.0e-7,
			"cut by the ceiling and not by the conversion, got {}",
			seen[0].0
		);

		// and the boundary itself is not cut: a gap of exactly the ceiling is
		// a length the sender really may have, and the one either side of it
		// says the comparison is the right way round.
		for gap in [u64::from(MAX_CATCH_UP) - 1, u64::from(MAX_CATCH_UP)] {
			let inside = [asked(1, gap)];
			let (_, seen) = ran(&inside, 0, Vec3::X);
			let want = f32::from(u16::try_from(gap).expect("small")) / 60.0;

			assert!((seen[0].0 - want).abs() < 1.0e-7, "gap {gap}, got {}", seen[0].0);
		}
	}

	/// Everything the caller said about the box reaches every move of a run.
	#[test]
	fn the_limits_a_caller_set_are_the_limits_every_command_moves_under() {
		let world = World::new();
		let mine = BodyId::NONE;
		let start = Motion::new(FROM, Vec3::ZERO, Vec3::splat(0.25), 1.0 / 60.0)
			.stepping(0.5)
			.standing(0.9)
			.layered(Layers::single(3))
			.ignoring(mine);
		let commands = [asked(1, 40), asked(2, 41)];
		let mut seen = 0;

		replay(&world, &start, 9, &commands, |_command, _before, motion| {
			seen += 1;

			assert!((motion.step - 0.5).abs() < f32::EPSILON, "the lip it may climb");
			assert!((motion.ground - 0.9).abs() < f32::EPSILON, "the slope it may stand on");
			assert_eq!(motion.layers, Layers::single(3), "the layers it is on");
			assert_eq!(motion.ignored(), &[mine], "and what it is blind to");
			assert!(
				motion
					.extents
					.abs_diff_eq(Vec3::splat(0.25), 1.0e-6),
				"and how big it is"
			);
		});

		assert_eq!(seen, 2, "on every command rather than the first");
	}

	#[test]
	fn a_command_that_did_not_move_the_clock_on_still_lasts_a_step() {
		// two commands claiming one step, and one claiming a step behind the
		// state it starts from. Neither is a move of nothing: a length of
		// nought is a frame of somebody's input dropped in silence.
		let commands = [asked(1, 50), asked(2, 50), asked(3, 20)];
		let (_, seen) = ran(&commands, 50, Vec3::X);

		for (index, (dt, _)) in seen.iter().enumerate() {
			assert!((dt - 1.0 / 60.0).abs() < 1.0e-7, "command {index} lasts a step, got {dt}");
		}
	}

	/// The speed carried into a command is what the one before it *came to*,
	/// not what it was asked for.
	///
	/// The two are the same number until something clips one, which is why
	/// this is the only test in the file with a wall in it.
	#[test]
	fn a_command_is_handed_the_speed_the_last_one_came_to() {
		let world = walled();
		// a step short of the wall, moving at it fast enough to reach it in
		// one step and keep going.
		let start = Motion::new(
			Vec3::new(WALL - 1.0, 0.0, 0.0),
			Vec3::ZERO,
			Vec3::splat(0.25),
			1.0 / 60.0,
		);
		let commands = [asked(1, 10), asked(2, 11)];
		let mut carried = Vec::new();
		let asking = Vec3::new(120.0, 0.0, 3.0);

		let moved = replay(&world, &start, 9, &commands, |_command, before, motion| {
			carried.push(motion.velocity);

			// **added to rather than set**, which is the shape that can tell
			// the two apart: a game that adds gravity or friction to what is
			// already there is adding to whatever this carried in.
			if before.is_none() {
				motion.velocity = asking;
			} else {
				motion.velocity += Vec3::new(0.0, 0.0, 1.0);
			}
		});

		assert!(
			carried[0].abs_diff_eq(Vec3::ZERO, 1.0e-6),
			"the first is handed the state it was given, got {}",
			carried[0]
		);
		assert!(
			(carried[1].x).abs() < 1.0e-5,
			"and the second is handed a speed with the wall taken out of it rather than the \
			 hundred and twenty that was asked for, got {}",
			carried[1]
		);
		assert!(
			(carried[1].z - 3.0).abs() < 1.0e-5,
			"with everything the wall did not stop still on it, got {}",
			carried[1]
		);
		assert!(
			moved.position.x < WALL,
			"and the box is on this side of the wall, got {}",
			moved.position.x
		);
	}

	#[test]
	fn a_run_of_nothing_is_a_move_that_went_nowhere() {
		let world = World::new();
		let start = Motion::new(Vec3::new(4.0, 5.0, 6.0), Vec3::X, Vec3::splat(0.25), 1.0 / 60.0);
		let mut called = 0;
		let moved = replay(&world, &start, 7, &[], |_, _, _| called += 1);

		assert_eq!(called, 0, "nothing was asked of the game");
		assert_eq!(moved.position, start.position, "and the state is where it was");
		assert_eq!(moved.velocity, start.velocity);
		assert!(!moved.grounded);
	}

	#[test]
	fn an_error_fades_rather_than_being_shown_all_at_once() {
		let mut drift = Drift::NONE;

		assert!(!drift.showing());
		assert_eq!(drift.offset(), Vec3::ZERO);

		// drawn a way from where the replay now says it is, with all three
		// components different and none of them nought - a fixture whose z
		// cancels cannot see a third of the arithmetic.
		let (drawn, predicted) = (Vec3::new(2.0, -1.0, 0.5), Vec3::new(1.0, 1.0, -3.5));

		drift.correct(drawn, predicted);
		assert!(drift.showing());
		assert!(
			drift
				.offset()
				.abs_diff_eq(Vec3::new(1.0, -2.0, 4.0), 1.0e-6),
			"the whole of it, at first, got {}",
			drift.offset()
		);
		assert!(
			(predicted + drift.offset()).abs_diff_eq(drawn, 1.0e-6),
			"so the picture does not move at the moment it is corrected"
		);

		// a quarter of the way through, and a quarter is not a half or a
		// whole: a decay that ignored the clock would answer the same.
		drift.advance(DECAY / 4.0);
		assert!(
			drift
				.offset()
				.abs_diff_eq(Vec3::new(0.75, -1.5, 3.0), 1.0e-5),
			"got {}",
			drift.offset()
		);

		drift.advance(DECAY);
		assert!(!drift.showing(), "and it is gone rather than overshooting");
		assert_eq!(drift.offset(), Vec3::ZERO);

		drift.advance(DECAY * 10.0);
		assert_eq!(drift.offset(), Vec3::ZERO, "and stays gone");
	}

	#[test]
	fn a_correction_on_top_of_one_still_fading_absorbs_what_was_left() {
		let mut drift = Drift::NONE;

		drift.correct(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO);
		drift.advance(DECAY / 2.0);

		// what is on screen right now, which is not the prediction and not the
		// old drawn position either.
		let showing = Vec3::ZERO + drift.offset();

		assert!(showing.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1.0e-5), "got {showing}");

		// and now a second correction, against a prediction that has moved on.
		let predicted = Vec3::new(1.0, 0.0, 0.0);

		drift.correct(showing, predicted);
		assert!(
			(predicted + drift.offset()).abs_diff_eq(showing, 1.0e-5),
			"the picture is still continuous, with no case for it in the code"
		);
		assert!(
			drift
				.offset()
				.abs_diff_eq(Vec3::new(4.0, 0.0, 0.0), 1.0e-5),
			"and the error is the whole gap rather than the new half of it, got {}",
			drift.offset()
		);
	}

	#[test]
	fn a_drift_is_something_a_game_can_keep_in_its_own_arena() {
		let mut drift = Drift::NONE;

		drift.correct(Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO);

		let bytes = bytemuck::bytes_of(&drift).to_vec();
		let back: Drift = bytemuck::pod_read_unaligned(&bytes);

		assert_eq!(bytes.len(), 16);
		assert_eq!(back, drift, "so it survives a round trip through the arena");
		// and what came back is the error itself rather than whatever the
		// round trip happened to agree with: a comparison of a copy against
		// its own source holds for any content at all.
		assert!(
			back.offset()
				.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1.0e-6),
			"got {}",
			back.offset()
		);

		let zeroed: Drift = bytemuck::pod_read_unaligned(&[0_u8; 16]);

		assert_eq!(zeroed, Drift::NONE, "and a fresh arena reads as nothing to correct");
		assert!(!zeroed.showing());
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
