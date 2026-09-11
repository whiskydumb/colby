//! Throwing particles and moving them, once per simulation step.
//!
//! **Inside the step rather than the frame, and that is the whole design.**
//! Every engine read for this ticks its particles once a rendered frame -
//! Fyrox in `update`, Godot in its internal process notification, s&box in
//! `Step( timeDelta )`, Wicked in `UpdateCPU`. colby cannot: `--shot` is
//! ninety fixed steps and one frame, `--record` is ninety steps and no frame
//! at all, and `--link` is six hundred steps and a digest. A cloud ticked by
//! the frame would be one particle old in a screenshot and invisible to the
//! other two, so the three tools that exist to review a change could not
//! review this one. It goes where the solver goes.
//!
//! **CPU rather than a compute pass, and the field agrees more than it looks.**
//! Niagara's own default is `ENiagaraSimTarget::CPUSim` with the GPU an
//! opt-in per emitter (`NiagaraEmitter.h:336`); Fyrox and s&box have no GPU
//! path at all; Godot ships the two as separate node types; only Wicked is
//! compute-only. The colby-specific half of the argument is the paragraph
//! above - a compute pass reads nothing back, so nothing deterministic can see
//! it - and `PERF-1`'s measurement is the other half: `gpu scene` is already
//! the largest row of the frame, and this is the one kind of work that can be
//! put somewhere else.
//!
//! **The randomness is seeded per step and per slot**, never carried. That is
//! what makes two machines throw the same plume and what makes a world that
//! was loaded from a file at step four hundred identical to one that ran
//! there: there is no generator state to save, restore or get wrong. Fyrox
//! keeps a generator with a `reset()` on the node for the same property, and
//! seeding from the step number gets it for nothing.

use colby_core::{
	abi::{Emitter, EmitterKind, Spark, World},
	glam::{Quat, Vec3},
	random::Random,
};

/// The odd number the step, the slot and the generation are folded with.
///
/// The same constant `colby_net`'s own seeding uses, and it is here for the
/// same reason: three small numbers, two of which are usually nought, would
/// otherwise seed three streams that start out looking alike.
const MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Throws this step's particles and moves the ones already in the air.
///
/// The order is sweep, move, throw, and it is not arbitrary. Sweeping first
/// means the cap an emitter is measured against is what is really alive rather
/// than what was alive before the dead were counted. Moving before throwing
/// means a particle born this step is drawn where it was born rather than one
/// step downwind of it, which is what the sub-step offset below is for
/// instead.
///
/// @param world - the world to step; its `dt` is how long the step is
pub(crate) fn step(world: &mut World) {
	let dt = world.dt;

	if dt <= 0.0 {
		return;
	}

	sweep(world);
	drift(world, dt);
	throw(world, dt);
}

/// Drops every particle that has run out of life or whose emitter has stopped.
///
/// **An emitter turned off keeps its cloud**, which is why the test here is
/// whether the owner is still alive and still an emitter of some kind rather
/// than whether it `throws()`: a fire that was switched off should burn out
/// over the next second rather than vanish mid-air. What does take a cloud at
/// once is the entity dying, and that is the same test - a dead handle
/// resolves to nobody.
fn sweep(world: &mut World) {
	let entities = &world.entities;

	world.sparks.sweep(|owner| {
		entities
			.emitter(owner)
			.is_some_and(|emitter| emitter.kind.throws())
	});
}

/// Moves every live particle by one step.
///
/// Semi-implicit Euler - the acceleration lands on the velocity and the new
/// velocity moves the particle - which is what the solver next door does and
/// what every reference here does. The drag is applied as a share kept per
/// *second* raised to the step, rather than multiplied per step, so that a
/// world at another `sim.rate` blows its smoke the same way. Fyrox's own
/// integration folds `dt` into the velocity instead and is therefore only
/// correct at one step length; that is a bug to avoid rather than a shape to
/// copy.
fn drift(world: &mut World, dt: f32) {
	let gravity = world.gravity;
	let entities = &world.entities;
	let sparks = &mut world.sparks;

	for spark in sparks.iter_mut() {
		let Some(emitter) = entities.emitter(spark.owner) else {
			continue;
		};

		spark.velocity += gravity * emitter.gravity * dt;
		spark.velocity *= (1.0 - emitter.drag.clamp(0.0, 1.0)).powf(dt);
		spark.position += spark.velocity * dt;
		spark.age += dt;
	}
}

/// Throws whatever this step's emitters are owed.
fn throw(world: &mut World, dt: f32) {
	// one walk of the pool rather than one per emitter: what each emitter's
	// cap is measured against is how many of its own are alive, and asking
	// that per emitter would be the pool times the emitters. The array is over
	// the entity table's slots, which is what a particle's owner indexes.
	let mut alive = vec![0_u32; world.entities.slots()];

	for spark in world.sparks.iter() {
		if let Some(count) = alive.get_mut(spark.owner.slot()) {
			*count = count.saturating_add(1);
		}
	}

	let steps = world.steps;
	let mut throwing = Vec::new();

	for (id, ..) in world.entities.iter() {
		let Some(emitter) = world.entities.emitter(id).copied() else {
			continue;
		};

		if !emitter.throws() {
			continue;
		}

		let Some(at) = world.entities.placed(id) else {
			continue;
		};

		throwing.push((id, emitter, at));
	}

	for (id, emitter, at) in throwing {
		let slot = id.slot();
		let owed = world.sparks.owed(slot) + emitter.owed(dt);
		let room = emitter
			.cap
			.saturating_sub(alive.get(slot).copied().unwrap_or(0));
		// the whole part is what is thrown and the remainder is carried, which
		// is the whole of why an emitter at half a particle a second throws
		// anything at all rather than rounding to nothing every step forever.
		let wanted = owed.floor().max(0.0);
		let mut left = owed - wanted;

		// a run of the step's own seed, so the same step on another machine
		// throws the same particles. The generation is in it as well as the
		// slot: a slot reused by something else must not inherit the stream
		// its previous occupant was on.
		let mut random = Random::new(
			steps
				.wrapping_mul(MIX)
				.wrapping_add(u64::from(u32::try_from(slot).unwrap_or(u32::MAX)))
				.wrapping_add(u64::from(id.generation()).wrapping_mul(MIX)),
		);

		let count = whole(wanted).min(room);

		for number in 0..count {
			let (direction, speed, life) = shape(&emitter, at.rotation, &mut random);
			let velocity = direction * speed;
			// **where in the step it was thrown, not where the step began.**
			// A rate of two hundred a second at sixty steps is three particles
			// a step, and putting all three at the entity's position draws a
			// plume as a stack of shells rather than as a stream. The offset
			// is a share of one step's travel, so the same three land spread
			// along the path they would have taken.
			let through = f32::from(u16::try_from(number).unwrap_or(0))
				/ f32::from(u16::try_from(count.max(1)).unwrap_or(1));

			if !world.sparks.push(Spark {
				position: at.position + velocity * through * dt,
				velocity,
				age: through * dt,
				life,
				owner: id,
			}) {
				// the world is full. What is owed is kept rather than thrown
				// away, so an emitter that lost this step's particles to a
				// crowded world throws them the moment there is room.
				left += f32::from(u16::try_from(count - number).unwrap_or(u16::MAX));

				break;
			}
		}

		// anything the cap refused is owed no longer: an emitter at its own
		// ceiling is not behind, it is full, and carrying the debt would make
		// it burst the moment one particle died.
		world.sparks.set_owed(slot, left);
	}
}

/// Which way, how fast and for how long one particle leaves.
///
/// @param emitter - what is throwing it
/// @param rotation - how the entity is turned, which is what aims a cone
/// @param random - the step's own stream
/// @return the unit direction, the speed and the life
fn shape(emitter: &Emitter, rotation: Quat, random: &mut Random) -> (Vec3, f32, f32) {
	let around = unit(random) * std::f32::consts::TAU;
	// the cosine of the angle away from the axis. A point's is anywhere from
	// minus one to one, which is the standard uniform-on-a-sphere trick; a
	// cone's is between the cosine of its half-angle and one, which is the
	// same trick restricted to a cap and is uniform over the cap's *area*
	// rather than over its angle - an angle-uniform cone piles particles up
	// the middle.
	let along = match emitter.kind {
		| EmitterKind::Cone => {
			let narrowest = emitter
				.spread
				.clamp(0.0, std::f32::consts::PI)
				.cos();

			(1.0 - narrowest).mul_add(unit(random), narrowest)
		},
		| _ => 2.0_f32.mul_add(-unit(random), 1.0),
	};
	let sideways = (1.0 - along * along).max(0.0).sqrt();
	// built about -z, which is the forward a camera, a listener and a spot
	// light already agree on, and then turned the way the entity is turned.
	let direction =
		rotation * Vec3::new(sideways * around.cos(), sideways * around.sin(), -along);
	let speed = emitter.speed * spread(emitter.speed_spread, random);
	let life = emitter.life * spread(emitter.life_spread, random);

	(
		direction.normalize_or(Vec3::NEG_Y),
		speed,
		// never nought, or a particle would be born dead and divide its own
		// age by nothing
		life.max(f32::EPSILON),
	)
}

/// A multiplier around one, spread by a share.
///
/// At a share of nought this is exactly one and the emitter is a metronome; at
/// one it is anywhere from nothing to one, which is what makes a plume look
/// like a plume. Never above one: a spread that could make a particle *faster*
/// than the number somebody typed would make the typed number a floor rather
/// than the answer.
///
/// @param share - nought through one; anything outside is clamped
/// @param random - the stream to draw from
fn spread(share: f32, random: &mut Random) -> f32 {
	share.clamp(0.0, 1.0).mul_add(-unit(random), 1.0)
}

/// How wide the space a draw is reduced into is, as a float.
///
/// Two to the thirty-second is not exactly representable as an `f32`, so this
/// is the nearest one above it - which is what makes a share strictly below
/// one rather than occasionally equal to it.
const SPACE: f32 = 4_294_967_300.0;

/// A draw from nought up to one.
fn unit(random: &mut Random) -> f32 {
	// the top thirty-two bits, which is the well-mixed end of this register -
	// @ref `Random::below` for why the low end is not used for anything.
	#[expect(
		clippy::as_conversions,
		clippy::cast_precision_loss,
		reason = "a share of a known space; the loss below 2^24 is smaller than anything a \
		          particle can show"
	)]
	let share = (random.draw() >> 32) as f32 / SPACE;

	share
}

/// A count as a `u32`, saturating.
///
/// @param count - how many, as the float an emitter's debt is kept in
fn whole(count: f32) -> u32 {
	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "clamped to the range of a u32 first, so the cast cannot lose or go negative"
	)]
	let whole = count.clamp(0.0, f32::from(u16::MAX)) as u32;

	whole
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{EntityId, SparkBlend, Transform},
		time::Rate,
	};

	use super::*;

	/// A world with one emitter in it, at the origin.
	fn thrower(emitter: Emitter) -> (World, EntityId) {
		let mut world = World::new();

		world.dt = Rate::DEFAULT.seconds();

		let id = world.entities.spawn();

		assert!(world.entities.set_emitter(id, emitter), "the handle resolves");

		(world, id)
	}

	/// Runs a world for a while.
	fn run(world: &mut World, steps: u32) {
		for _ in 0..steps {
			world.steps = world.steps.saturating_add(1);
			step(world);
		}
	}

	#[test]
	fn an_emitter_fills_up_to_its_cap_and_stops() {
		let (mut world, id) = thrower(Emitter {
			cap: 8,
			life: 100.0,
			life_spread: 0.0,
			..Emitter::point(600.0, 100.0)
		});

		run(&mut world, 60);

		assert_eq!(world.sparks.len(), 8, "six hundred a second is stopped by a cap of eight");
		assert_eq!(world.sparks.count(id), 8, "and all of them are its own");
	}

	#[test]
	fn a_particle_dies_when_its_life_runs_out() {
		let (mut world, id) = thrower(Emitter {
			life_spread: 0.0,
			..Emitter::point(60.0, 0.25)
		});

		run(&mut world, 30);

		let held = world.sparks.len();

		assert!(held > 0, "half a second in, something is in the air");

		// the emitter stops, and a quarter of a second later everything it
		// threw has run out.
		assert!(world.entities.set_emitter(id, Emitter::NONE), "and it can be turned off");

		run(&mut world, 30);

		assert_eq!(world.sparks.len(), 0, "and half a second later the cloud is gone");
	}

	#[test]
	fn a_cloud_outlives_the_emitter_being_turned_down_but_not_the_entity_dying() {
		let (mut world, id) = thrower(Emitter {
			life_spread: 0.0,
			..Emitter::point(60.0, 10.0)
		});

		run(&mut world, 30);

		let held = world.sparks.len();

		assert!(held > 0, "something is in the air");

		// turned down to nothing a second, which is not the same as turned off
		assert!(
			world
				.entities
				.set_emitter(id, Emitter { rate: 0.0, ..Emitter::point(0.0, 10.0) }),
			"the handle resolves"
		);
		run(&mut world, 30);

		assert_eq!(world.sparks.len(), held, "a rate of nothing burns out rather than vanishing");

		assert!(world.entities.despawn(id), "and then the entity goes");
		run(&mut world, 1);

		assert_eq!(world.sparks.len(), 0, "which takes the cloud with it");
	}

	#[test]
	fn a_hidden_emitter_throws_exactly_what_a_shown_one_throws() {
		// hidden is a question only a picture asks, and this is the step's
		// half of that rule: a fire that was hidden has to be burning when it
		// is shown again, and burning the way it would have been
		let ran = |hidden: bool| {
			let (mut world, id) = thrower(Emitter {
				speed_spread: 0.6,
				life_spread: 0.5,
				..Emitter::cone(120.0, 2.0, 0.6)
			});

			assert!(world.entities.set_hidden(id, hidden), "the handle resolves");
			run(&mut world, 60);

			world
		};
		let shown = ran(false);
		let hidden = ran(true);

		assert!(shown.sparks.len() > 10, "something was thrown");
		assert_eq!(hidden.sparks.len(), shown.sparks.len(), "as much hidden as shown");
		assert!(
			hidden
				.sparks
				.iter()
				.zip(shown.sparks.iter())
				.all(|(one, other)| one.position.to_array().map(f32::to_bits)
					== other.position.to_array().map(f32::to_bits)),
			"and the same particles, to the bit"
		);
	}

	#[test]
	fn the_same_steps_throw_the_same_cloud_twice() {
		let make = || {
			let (mut world, _) = thrower(Emitter {
				speed_spread: 0.6,
				life_spread: 0.5,
				..Emitter::cone(120.0, 2.0, 0.6)
			});

			run(&mut world, 90);

			world
		};

		let one = make();
		let two = make();

		assert!(one.sparks.len() > 40, "there is a cloud to compare");
		assert_eq!(one.sparks.len(), two.sparks.len(), "and both runs threw the same number");

		for (left, right) in one.sparks.iter().zip(two.sparks.iter()) {
			assert_eq!(left, right, "and every particle of it is in the same place");
		}
	}

	#[test]
	fn two_emitters_in_one_step_do_not_throw_the_same_particles() {
		let (mut world, _) = thrower(Emitter::point(120.0, 2.0));
		let second = world.entities.spawn();

		assert!(
			world
				.entities
				.set_emitter(second, Emitter::point(120.0, 2.0)),
			"a second thrower at the same place"
		);

		run(&mut world, 30);

		let mut first = Vec::new();
		let mut other = Vec::new();

		for spark in world.sparks.iter() {
			if spark.owner == second {
				other.push(spark.velocity);
			} else {
				first.push(spark.velocity);
			}
		}

		assert!(!first.is_empty() && !other.is_empty(), "both threw something");
		assert!(
			first != other,
			"and the slot is in the seed, so two emitters at one place are not one emitter \
			 drawn twice"
		);
	}

	#[test]
	fn a_cone_throws_into_its_cone_and_a_point_does_not() {
		let narrow = 0.2_f32;
		let (mut world, id) = thrower(Emitter::cone(600.0, 4.0, narrow));

		run(&mut world, 10);

		assert!(world.sparks.len() > 20, "there is a cloud");

		for spark in world.sparks.iter() {
			let along = spark
				.velocity
				.normalize_or_zero()
				.dot(Vec3::NEG_Z);

			assert!(
				along >= narrow.cos() - 1e-4,
				"every particle is inside the mouth: {along} against {}",
				narrow.cos()
			);
		}

		assert!(
			world
				.entities
				.set_emitter(id, Emitter::point(600.0, 4.0)),
			"and the same emitter as a point"
		);
		world.sparks.clear();
		run(&mut world, 10);

		let widest = world
			.sparks
			.iter()
			.map(|spark| {
				spark
					.velocity
					.normalize_or_zero()
					.dot(Vec3::NEG_Z)
			})
			.fold(1.0_f32, f32::min);

		assert!(widest < 0.0, "throws behind itself as well: {widest}");
	}

	#[test]
	fn a_cone_is_aimed_by_the_entity_being_turned() {
		let (mut world, id) = thrower(Emitter::cone(600.0, 4.0, 0.2));

		// turned a half turn about y, so its own -z now points at the world's
		// +z.
		assert!(
			world.entities.set_placed(id, Transform {
				rotation: Quat::from_rotation_y(std::f32::consts::PI),
				..Transform::IDENTITY
			}),
			"the handle resolves"
		);
		run(&mut world, 10);

		assert!(world.sparks.len() > 20, "there is a cloud");

		for spark in world.sparks.iter() {
			assert!(
				spark.velocity.normalize_or_zero().z > 0.9,
				"the cone follows the entity's own forward rather than the world's"
			);
		}
	}

	#[test]
	fn gravity_pulls_a_particle_the_way_the_world_does() {
		let (mut world, _) = thrower(Emitter {
			gravity: 1.0,
			drag: 0.0,
			speed: 0.0,
			speed_spread: 0.0,
			..Emitter::point(60.0, 10.0)
		});

		world.gravity = Vec3::new(0.0, -10.0, 0.0);
		run(&mut world, 60);

		let lowest = world
			.sparks
			.iter()
			.map(|spark| spark.position.y)
			.fold(0.0_f32, f32::min);

		assert!(lowest < -1.0, "a second of ten a second squared has taken it down: {lowest}");
	}

	#[test]
	fn drag_takes_a_particle_s_speed_off_and_a_share_of_nought_does_not() {
		let fast = |drag| {
			let (mut world, _) = thrower(Emitter {
				drag,
				gravity: 0.0,
				speed_spread: 0.0,
				life_spread: 0.0,
				speed: 10.0,
				..Emitter::point(2.0, 10.0)
			});

			run(&mut world, 60);

			// the *slowest*, which is the oldest: the newest particle is
			// always at the speed it was thrown at whatever the drag is, so a
			// maximum here would measure nothing at all.
			world
				.sparks
				.iter()
				.map(|spark| spark.velocity.length())
				.fold(f32::MAX, f32::min)
		};

		let free = fast(0.0);
		let dragged = fast(0.9);

		assert!((free - 10.0).abs() < 1e-3, "no drag leaves the speed alone: {free}");
		assert!(dragged < free * 0.5, "and nine tenths a second takes most of it: {dragged}");
	}

	#[test]
	fn particles_born_in_one_step_are_spread_along_it_rather_than_stacked() {
		let (mut world, _) = thrower(Emitter {
			speed_spread: 0.0,
			life_spread: 0.0,
			gravity: 0.0,
			drag: 0.0,
			speed: 10.0,
			..Emitter::point(600.0, 10.0)
		});

		run(&mut world, 1);

		assert!(world.sparks.len() > 4, "several were thrown in one step");

		let ages: Vec<f32> = world
			.sparks
			.iter()
			.map(|spark| spark.age)
			.collect();
		let oldest = ages.iter().copied().fold(0.0_f32, f32::max);
		let youngest = ages.iter().copied().fold(f32::MAX, f32::min);

		assert!(
			oldest > youngest,
			"they are spread through the step rather than stamped at one instant"
		);
		assert!(oldest < world.dt, "and none of them is older than the step that made it");
	}

	#[test]
	fn a_world_that_is_not_stepping_throws_nothing() {
		let (mut world, _) = thrower(Emitter::point(600.0, 10.0));

		world.dt = 0.0;
		run(&mut world, 10);

		assert_eq!(world.sparks.len(), 0, "a step of no length is not a step");
	}

	#[test]
	fn what_an_emitter_is_owed_is_kept_across_steps() {
		// a fifth of a particle a step: nothing is thrown for four steps and
		// then one is, which is the whole point of keeping the remainder.
		let (mut world, id) = thrower(Emitter {
			life_spread: 0.0,
			..Emitter::point(12.0, 10.0)
		});

		run(&mut world, 4);

		assert_eq!(world.sparks.len(), 0, "four steps at a fifth each is not one particle");
		assert!(world.sparks.owed(id.slot()) > 0.5, "but most of one is owed");

		run(&mut world, 2);

		assert_eq!(world.sparks.len(), 1, "and the fifth step throws it");
	}

	#[test]
	fn an_emitter_at_its_cap_is_not_owed_what_it_could_not_throw() {
		let (mut world, id) = thrower(Emitter {
			cap: 2,
			life: 100.0,
			life_spread: 0.0,
			..Emitter::point(600.0, 100.0)
		});

		run(&mut world, 30);

		assert_eq!(world.sparks.len(), 2, "the cap holds");
		assert!(
			world.sparks.owed(id.slot()) < 1.0,
			"and the hundreds it could not throw are not owed: {}",
			world.sparks.owed(id.slot())
		);
	}

	#[test]
	fn the_blend_is_carried_and_read_by_nobody_here() {
		// the simulation is blind to how a particle is drawn, which is the
		// line between this module and the renderer. The assertion is that the
		// cloud is the same either way.
		let cloud = |blend| {
			let (mut world, _) = thrower(Emitter { blend, ..Emitter::point(120.0, 2.0) });

			run(&mut world, 30);

			world
				.sparks
				.iter()
				.map(|spark| spark.position)
				.collect::<Vec<_>>()
		};

		assert_eq!(
			cloud(SparkBlend::Additive),
			cloud(SparkBlend::Alpha),
			"how a cloud reaches the picture is not something the step knows about"
		);
	}
}
