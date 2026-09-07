//! What a fluid does to what is in it: a push up, and two drags.
//!
//! **Where this runs is the whole design.** It is a pass inside
//! [`Simulation::step`](crate::Simulation::step), after the narrow phase has
//! said what overlaps what and *before* the solver integrates. That is the one
//! window where a body's force accumulator can be written and spent in the
//! same step: the step order is `simulation.step` then `game.update`, so
//! anything gameplay pushes is felt one step later, and anything written here
//! is felt now. @ref [`Bodies::apply_force_at`] and `colby-forces`.
//!
//! **Three numbers come out of a shape and a surface**: how much of it is
//! under, where the middle of that is, and how big it is across the flow. The
//! first two are all the buoyancy needs, because a push applied at the middle
//! of the submerged part *is* the righting torque - a tilted crate has more of
//! itself under on one side, the push lands off its center of mass, and it
//! rolls level. Nobody writes a righting term; it falls out of using
//! [`Bodies::apply_force_at`] instead of [`Bodies::apply_force`]. Both
//! references do exactly this: Jolt applies its buoyancy impulse at the center
//! of buoyancy (`Body.cpp:284`), and Unreal hangs several pontoons off a hull
//! and pushes each at its own place (`BuoyancyComponent.cpp:326`).
//!
//! **How much is under, per shape.** A ball has a closed form and it is Jolt's
//! (`SphereShape.cpp:166-205`); a box is cut into [`CELLS`] corner cells and
//! each is filled by its own depth, which is *exact* for a box that is level
//! at any depth and an approximation for one that is tilted. A triangle soup
//! floats in neither engine - `MeshShape::GetSubmergedVolume` is
//! `JPH_ASSERT(false, "Not supported")` (`MeshShape.h:148`) - and here it
//! cannot arise at all, because [`Body::movable`] already refuses a dynamic
//! mesh for want of an inertia tensor.
//!
//! Jolt's own convex path is worth knowing before reaching for it: it does not
//! integrate the hull either, it integrates the hull's *bounding box*
//! (`ConvexShape.cpp:383-389`, eight corners and six faces). So of the shapes
//! that can float here, a box is the case Jolt is exact for and a ball is the
//! case it has a formula for, and there is no third.

use colby_core::{
	abi::{Bodies, Body, BodyId, ShapeKind, Water},
	glam::Vec3,
};

use crate::contact::inverse_inertia;

/// How many cells a box is cut into to find how much of it is under.
///
/// Two per axis, so eight, each the box's own octant. The number is not a
/// tuning knob so much as the smallest one that works: with a single cell a
/// box could not tilt, because one push at one place has no lever, and a
/// tilted box has to right itself. Eight gives every corner its own depth,
/// which is what Unreal buys with a hand-placed pontoon per corner
/// (`BuoyancyTypes.h:17-38`) - the difference being that these are derived
/// from the shape rather than authored, so a crate nobody thought about
/// floats.
///
/// Raising it costs a linear amount and buys accuracy only for a *tilted*
/// box: level, the ramp over a cell's own height is not an approximation at
/// all, because a box's cross-section does not change with depth.
pub const CELLS: usize = 8;

/// The smallest span, volume or speed worth doing arithmetic about.
const TINY: f32 = 1.0e-6;

/// What a shape displaces, and where the push for it lands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Displaced {
	/// How much of the shape is under the surface, in cubic units.
	pub volume: f32,

	/// The whole shape's volume, for the drags to take a fraction of.
	pub total: f32,

	/// The middle of the submerged part, in world space.
	///
	/// The center of buoyancy. Equal to the body's own middle when it is
	/// wholly under, which is why a fully submerged thing does not right
	/// itself - correctly, because a uniform body under water has no
	/// righting moment either.
	pub center: Vec3,
}

impl Displaced {
	/// Nothing under the surface at all.
	const DRY: Self = Self {
		volume: 0.0,
		total: 0.0,
		center: Vec3::ZERO,
	};

	/// Whether any of it is under.
	#[must_use]
	pub fn any(&self) -> bool { self.volume > TINY }

	/// How much of the shape is under, from nothing to all of it.
	#[must_use]
	pub fn fraction(&self) -> f32 {
		if self.total <= TINY {
			0.0
		} else {
			(self.volume / self.total).clamp(0.0, 1.0)
		}
	}
}

/// How much of a body is under a level surface, and where the middle of it is.
///
/// @param body - the body, for its shape and where it stands
/// @param surface - the height of the fluid's top, in world units
/// @return what it displaces
#[must_use]
pub fn displaced(body: &Body, surface: f32) -> Displaced {
	match body.shape.kind {
		| ShapeKind::Sphere => ball(body, surface),
		| ShapeKind::Box => cuboid(body, surface),
		// a triangle soup has no inside, which is the same reason it has no
		// inertia tensor and is never dynamic. @ref `Body::movable`.
		| ShapeKind::Mesh => Displaced::DRY,
	}
}

/// A ball's submerged volume and center of buoyancy.
///
/// The spherical cap, and both formulas are the ones Jolt uses
/// (`SphereShape.cpp:189-194`). Unreal's pontoon reaches for the same integral
/// (`BuoyancyComponent.cpp:344`), which is the sense in which the two
/// references disagree less than they look to: what they differ about is whose
/// volume, not whether to compute one.
fn ball(body: &Body, surface: f32) -> Displaced {
	let radius = body.shape.radius.abs() * body.transform.scale.abs().max_element();
	let center = body.transform.position;
	let total = (4.0 / 3.0) * core::f32::consts::PI * radius * radius * radius;

	if radius <= TINY {
		return Displaced::DRY;
	}

	// how far the middle is under, so that positive is wet
	let under = surface - center.y;

	if under <= -radius {
		return Displaced { volume: 0.0, total, center: Vec3::ZERO };
	}

	if under >= radius {
		return Displaced { volume: total, total, center };
	}

	// the height of the cap, measured up from the bottom of the ball
	let cap = radius + under;
	let volume = (core::f32::consts::PI / 3.0) * cap * cap * radius.mul_add(3.0, -cap);
	// and the cap's own middle, below the ball's
	let down = 0.75 * 2.0_f32.mul_add(radius, -cap).powi(2) / radius.mul_add(3.0, -cap).max(TINY);

	Displaced {
		volume,
		total,
		center: center - Vec3::Y * down,
	}
}

/// A box's submerged volume and center of buoyancy, by its own cells.
///
/// Each cell carries an eighth of the volume and is filled by the depth of its
/// own vertical span. **Exact for a level box at any depth**, because the
/// fraction of a cell under a level surface really is linear in that depth
/// when the cross-section does not change; an approximation once it tilts,
/// where the true answer is a clipped polyhedron. @ref [`CELLS`].
fn cuboid(body: &Body, surface: f32) -> Displaced {
	let extents = body.shape.local_extents().abs() * body.transform.scale.abs();
	let total = 8.0 * extents.x * extents.y * extents.z;

	if total <= TINY {
		return Displaced::DRY;
	}

	let rotation = body.transform.rotation;
	let half = extents * 0.5;
	// how tall one cell stands after the rotation: the rotated cell's own
	// bounding half-height, which is what the surface cuts across. The three
	// numbers are how much of each of the body's own axes points up.
	let upright =
		Vec3::new((rotation * Vec3::X).y, (rotation * Vec3::Y).y, (rotation * Vec3::Z).y).abs();
	let tall = upright.dot(half);
	let share = total / 8.0;

	let mut volume = 0.0_f32;
	let mut weighted = Vec3::ZERO;

	for corner in 0..CELLS {
		let sign = Vec3::new(
			if corner & 1 == 0 { -1.0 } else { 1.0 },
			if corner & 2 == 0 { -1.0 } else { 1.0 },
			if corner & 4 == 0 { -1.0 } else { 1.0 },
		);
		let middle = body.transform.position + rotation * (sign * half);
		let bottom = middle.y - tall;
		let deep = if tall <= TINY {
			// a flat box seen edge-on: the cell is a plane, so it is under or
			// it is not, and there is no ramp between
			f32::from(u8::from(surface >= middle.y)) * 2.0 * tall
		} else {
			(surface - bottom).clamp(0.0, 2.0 * tall)
		};

		if deep <= TINY {
			continue;
		}

		let part = share * (deep / (2.0 * tall).max(TINY));
		volume += part;
		weighted += part * Vec3::new(middle.x, deep.mul_add(0.5, bottom), middle.z);
	}

	if volume <= TINY {
		return Displaced { volume: 0.0, total, center: Vec3::ZERO };
	}

	Displaced { volume, total, center: weighted / volume }
}

/// The area a body presents to a flow, in square units.
///
/// The local bounding box's cross-section along the direction it is moving,
/// which is Jolt's (`Body.cpp:238-241`): a plank edge-on to the current has
/// less of itself in the way than the same plank face-on, and the whole of
/// what makes that true is this projection.
///
/// @param body - the body, for its shape and which way it is turned
/// @param through - the direction it is moving through the fluid, normalized
fn area(body: &Body, through: Vec3) -> f32 {
	let extents = body.shape.local_extents().abs() * body.transform.scale.abs();
	let size = extents * 2.0;
	let local = body
		.transform
		.rotation
		.inverse()
		.mul_vec3(through)
		.abs();

	local.dot(Vec3::new(size.y * size.z, size.z * size.x, size.x * size.y))
}

/// What one fluid does to one body this step.
///
/// Everything is a force or a torque rather than an impulse, because colby has
/// an accumulator the solver clears and Jolt does not - it writes straight
/// into the velocity (`AddLinearVelocityStep`), which is why its own buoyancy
/// is an impulse. Unreal accumulates forces, as here.
///
/// @param bodies - the table, written
/// @param id - which body is in the fluid
/// @param water - what it is in
/// @param surface - the height of the fluid's top
/// @param gravity - what the world pulls by
/// @param dt - how long a step is, for the drag's overshoot guard
/// @return whether anything was pushed
pub fn float(
	bodies: &mut Bodies,
	id: BodyId,
	water: &Water,
	surface: f32,
	gravity: Vec3,
	dt: f32,
) -> bool {
	let Some(body) = bodies.get(id).copied() else {
		return false;
	};

	// a sleeper is left alone rather than woken, which is Jolt's sample
	// exactly (`WaterShapeTest.cpp:114` applies buoyancy only to a body that
	// is active). The alternative is a crate that has settled at its
	// waterline being pushed every step forever, because at equilibrium the
	// push is not zero - it is exactly cancelling the weight - so it would
	// never be allowed to sleep again.
	if !body.movable() || body.sleeping {
		return false;
	}

	let under = displaced(&body, surface);

	if !under.any() {
		return false;
	}

	let mass = body.mass.max(TINY);
	// **weightless bodies do not float**, and that is not an oversight: a
	// buoyant force is the pressure gradient, and the pressure gradient is
	// gravity. Jolt says the same by folding its `GetGravityFactor()` into
	// the impulse (`Body.cpp:221`). The drags below still apply, because a
	// drag is what the fluid does rather than what the ground does.
	let push = if body.weightless {
		Vec3::ZERO
	} else {
		-gravity * water.density * under.volume
	};

	// the speed of the center of buoyancy rather than of the middle, so that
	// a spinning body's wet side is dragged and its dry side is not
	let arm = under.center - body.transform.position;
	let moving = water.against(body.velocity + body.angular.cross(arm));
	let speed = moving.length();
	let wet = under.fraction();
	let across = area(&body, Vec3::Y);

	// what stops a bob, and the quadratic drag below cannot do it: a drag in
	// the square of the speed takes almost nothing off a small oscillation,
	// so a crate left to it rings for minutes. First order and vertical only,
	// which is Unreal's `BuoyancyDamp` (`BuoyancyComponent.cpp:357-360`).
	// Clamped against what would reverse the fall, for the reason the drag is
	// clamped: this term is stiff by design and a stiff term at sixty hertz
	// can take off more than there was.
	let bob = -Vec3::Y
		* (water.damp * water.density * across * wet * moving.y)
			.clamp(-moving.y.abs() * mass / dt.max(TINY), moving.y.abs() * mass / dt.max(TINY));

	let drag = if speed > TINY {
		// quadratic, which is Jolt's deliberate step away from the article it
		// follows (`Body.cpp:232-236`), scaled by how much of the body is
		// actually in the fluid - Jolt scales only its angular drag that way
		// and this scales both, because a crate with a toe in the water being
		// dragged as hard as a sunk one is the visible half of the difference.
		let force =
			0.5 * water.density * water.linear_drag * area(&body, moving / speed) * wet * speed;
		// and the guard that cannot be skipped: a quadratic drag at sixty
		// hertz can take more speed off than there was, which reverses the
		// body. Jolt clamps the same thing in the same place
		// (`Body.cpp:255-258`). Against the speed *through the fluid* rather
		// than against the ground speed, because bringing something to a stop
		// in a current is not what a drag can do.
		-moving * (force * speed).min(speed * mass / dt.max(TINY)) / speed.max(TINY)
	} else {
		Vec3::ZERO
	};

	let spin = body.angular;
	let turn = if spin.length_squared() > TINY {
		// eq 2.5.15 of the article both engines cite, as Jolt writes it
		// (`Body.cpp:272`): the average width squared, the mass, and the
		// submerged fraction. Unreal's is the same shape with the width and
		// the mass folded into one coefficient (`BuoyancyComponent.cpp:784`).
		let extents = body.shape.local_extents().abs() * body.transform.scale.abs();
		let width = (extents.x + extents.y + extents.z) * 2.0 / 3.0;
		let torque = -spin * water.angular_drag * wet * width * width * mass;
		// the same overshoot guard, and here it can be exact rather than
		// estimated because the tensor is a call away
		let change = inverse_inertia(&body) * torque * dt;

		if change.length_squared() > spin.length_squared() {
			torque * (spin.length() / change.length().max(TINY))
		} else {
			torque
		}
	} else {
		Vec3::ZERO
	};

	// the push, the bob and the drag land together at the center of buoyancy,
	// which is what turns all three into a righting torque as well as a shove
	let pushed = bodies.apply_force_at(id, push + bob + drag, under.center);

	if turn != Vec3::ZERO {
		bodies.apply_torque(id, turn);
	}

	pushed
}

#[cfg(test)]
mod tests {
	use colby_core::abi::{Shape, Transform};

	use super::*;

	/// A body of a shape, standing at a height.
	fn at(shape: Shape, height: f32) -> Body {
		Body::dynamic(shape, Transform::at(Vec3::new(0.0, height, 0.0)), 1.0)
	}

	#[test]
	fn a_level_box_displaces_exactly_what_is_under_the_line() {
		// the unit cube this engine hands out, spanning -0.5 to 0.5
		let crate_ = at(Shape::UNIT, 0.0);

		for (surface, wanted, middle) in [
			(-0.5, 0.0, 0.0),
			(-0.25, 0.25, -0.375),
			(0.0, 0.5, -0.25),
			(0.25, 0.75, -0.125),
			(0.5, 1.0, 0.0),
		] {
			let under = displaced(&crate_, surface);

			assert!(
				(under.volume - wanted).abs() < 1.0e-5,
				"a surface at {surface} covers {wanted} of it, got {}",
				under.volume
			);

			if wanted > 0.0 {
				assert!(
					(under.center.y - middle).abs() < 1.0e-5,
					"and the middle of that part is at {middle}, got {}",
					under.center.y
				);
			}
		}
	}

	#[test]
	fn a_box_wholly_under_is_pushed_through_its_own_middle() {
		let crate_ = at(Shape::UNIT, 0.0);
		let under = displaced(&crate_, 10.0);

		assert!((under.volume - 1.0).abs() < 1.0e-5, "all of it");
		assert!(
			under
				.center
				.abs_diff_eq(crate_.transform.position, 1.0e-5),
			"through the middle, which is why a sunk crate does not right itself - and nothing \
			 uniform under water does"
		);
	}

	#[test]
	fn a_ball_half_under_displaces_half_a_ball() {
		let ball = at(Shape::ball(2.0), 0.0);
		let whole = (4.0 / 3.0) * core::f32::consts::PI * 8.0;
		let under = displaced(&ball, 0.0);

		assert!(
			(under.volume - whole / 2.0).abs() < 1.0e-4,
			"half a ball of radius two, got {}",
			under.volume
		);
		assert!(
			(under.center.y + 0.75).abs() < 1.0e-4,
			"and a hemisphere's middle is three eighths of the radius down, got {}",
			under.center.y
		);
		assert!(
			(displaced(&ball, 2.0).volume - whole).abs() < 1.0e-4,
			"and at the top of it, the whole ball"
		);
		assert!(!displaced(&ball, -2.0).any(), "and under the bottom of it, none");
	}

	#[test]
	fn a_tilted_box_has_more_of_itself_under_on_the_low_side() {
		// the test the whole design is for: a push at the middle of the wet
		// part is off the center of mass, which is the righting torque
		let mut crate_ = at(Shape::UNIT, 0.0);
		crate_.transform.rotation =
			colby_core::glam::Quat::from_rotation_z(core::f32::consts::FRAC_PI_4);

		let under = displaced(&crate_, 0.0);

		assert!(under.any(), "half of a tilted crate is still under a surface at its middle");
		assert!(
			under.center.y < -1.0e-3,
			"and the middle of the wet part is below the crate's own, got {}",
			under.center.y
		);
	}

	#[test]
	fn a_flat_plate_drags_more_face_on_than_edge_on() {
		let plate = at(Shape::cuboid(Vec3::new(1.0, 0.05, 1.0)), 0.0);

		assert!(
			area(&plate, Vec3::Y) > area(&plate, Vec3::X) * 4.0,
			"the wide way through the water is the slow way, got {} against {}",
			area(&plate, Vec3::Y),
			area(&plate, Vec3::X)
		);
	}

	/// A table with a body in it, floated for a while.
	///
	/// @param mass - how heavy the body is
	/// @param steps - how many times to push and integrate by hand
	/// @return where it ended up
	fn settled(mass: f32, steps: usize) -> f32 {
		let dt = 1.0 / 60.0;
		let gravity = Vec3::new(0.0, -9.81, 0.0);
		let mut bodies = Bodies::default();
		let id = bodies.spawn(Body::dynamic(
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 2.0, 0.0)),
			mass,
		));

		for _ in 0..steps {
			float(&mut bodies, id, &Water::pool(), 0.0, gravity, dt);

			let Some(body) = bodies.get_mut(id) else { break };
			let push = body.force;
			body.velocity += (gravity + push / mass) * dt;
			// the same damping the solver applies, so this stands in for it
			body.velocity *= 1.0 / dt.mul_add(0.06, 1.0);
			body.transform.position += body.velocity * dt;
			body.force = Vec3::ZERO;
			body.torque = Vec3::ZERO;
		}

		bodies
			.get(id)
			.map_or(f32::NAN, |body| body.transform.position.y)
	}

	#[test]
	fn a_crate_settles_at_a_waterline_its_mass_decides_and_a_heavy_one_sits_lower() {
		// the default fluid is twice the density of the default body, so half
		// of that body stands out: its middle ends up level with the surface.
		// @ref `colby_core::abi::water::DENSITY`.
		let light = settled(1.0, 150);
		let heavy = settled(1.5, 150);

		assert!(
			light.abs() < 0.05,
			"a crate of mass one floats with its middle at the surface, got {light}"
		);
		assert!(
			heavy < light - 0.1,
			"and one half again as heavy sits lower, got {heavy} against {light}"
		);
		assert!(heavy > -0.5, "but not so low it has sunk, got {heavy}");
	}

	#[test]
	fn a_sleeper_is_left_asleep_rather_than_pushed_awake() {
		let mut bodies = Bodies::default();
		let id = bodies.spawn(Body::dynamic(Shape::UNIT, Transform::IDENTITY, 1.0));

		if let Some(body) = bodies.get_mut(id) {
			body.sleeping = true;
		}

		assert!(
			!float(&mut bodies, id, &Water::pool(), 10.0, Vec3::NEG_Y * 9.81, 1.0 / 60.0),
			"a crate that has settled at its waterline stays settled"
		);
		assert!(
			bodies.get(id).is_some_and(|body| body.sleeping),
			"and nothing woke it, which is what would keep it awake forever"
		);
	}

	#[test]
	fn a_weightless_body_is_dragged_but_not_lifted() {
		let mut bodies = Bodies::default();
		let id = bodies.spawn(Body::dynamic(Shape::UNIT, Transform::IDENTITY, 1.0));

		if let Some(body) = bodies.get_mut(id) {
			body.weightless = true;
			body.velocity = Vec3::new(4.0, 0.0, 0.0);
		}

		float(&mut bodies, id, &Water::pool(), 10.0, Vec3::NEG_Y * 9.81, 1.0 / 60.0);

		let force = bodies
			.get(id)
			.map_or(Vec3::ZERO, |body| body.force);

		assert!(force.y.abs() < 1.0e-4, "nothing lifts it: buoyancy is gravity, got {force}");
		assert!(force.x < -1.0e-3, "but the fluid still holds it back, got {force}");
	}

	#[test]
	fn a_drag_never_takes_off_more_speed_than_there_was() {
		// the guard both references write out, at a speed and a drag chosen to
		// blow up without it
		let dt = 1.0 / 60.0;
		let mut bodies = Bodies::default();
		let id = bodies.spawn(Body::dynamic(Shape::UNIT, Transform::IDENTITY, 1.0));
		let thick = Water { linear_drag: 400.0, ..Water::pool() };

		if let Some(body) = bodies.get_mut(id) {
			body.velocity = Vec3::new(60.0, 0.0, 0.0);
			body.weightless = true;
		}

		float(&mut bodies, id, &thick, 10.0, Vec3::NEG_Y * 9.81, dt);

		let (force, velocity) = bodies
			.get(id)
			.map_or((Vec3::ZERO, Vec3::ZERO), |body| (body.force, body.velocity));
		let after = velocity + force * dt;

		assert!(
			after.x >= -1.0e-3,
			"a drag that reversed what it slowed would be a body thrown backwards, got {after}"
		);
	}

	#[test]
	fn a_current_carries_what_is_floating_in_it() {
		let dt = 1.0 / 60.0;
		let mut bodies = Bodies::default();
		let id = bodies.spawn(Body::dynamic(Shape::UNIT, Transform::IDENTITY, 1.0));
		let river = Water {
			flow: Vec3::new(3.0, 0.0, 0.0),
			..Water::pool()
		};

		if let Some(body) = bodies.get_mut(id) {
			body.weightless = true;
		}

		float(&mut bodies, id, &river, 10.0, Vec3::NEG_Y * 9.81, dt);

		let force = bodies
			.get(id)
			.map_or(Vec3::ZERO, |body| body.force);

		assert!(
			force.x > 1.0e-3,
			"something still in a running river is pushed downstream, got {force}"
		);
	}
}
