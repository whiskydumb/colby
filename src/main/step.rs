//! One simulation step, wherever it is driven from.
//!
//! The window paces these off a [`Clock`](colby_core::time::Clock); `--shot`
//! runs a fixed number of them as fast as the machine will go. What is shared
//! between the two is this body and not the clock, which is what makes a
//! screenshot deterministic by construction rather than by care: there is one
//! definition of what a step does, and nothing in it depends on how much real
//! time has gone by.

use colby_core::{
	abi::{Input, World},
	time::STEP_SECONDS,
};
use colby_physics::Simulation;
use colby_script::Scripts;
use colby_ui::Interface;

use crate::game::Game;

/// Everything a step drives that outlives it.
///
/// One struct rather than four arguments: both call sites hold all four beside
/// each other and hand over the same four every time, and a signature that
/// grows every time a subsystem appears is a signature nobody reads. What is
/// *not* in here is the per-step half - the world, the input, the moment and
/// the mode - because those are what the step is actually about.
pub(crate) struct Parts<'a> {
	/// The loaded module, if there is one.
	pub(crate) game: Option<&'a mut Game>,

	/// The game's own interface, laid out and hit-tested here.
	pub(crate) interface: &'a mut Interface,

	/// The documents' own interface logic, if Lua came up.
	pub(crate) scripts: Option<&'a mut Scripts>,

	/// The physics, advanced here and queried by the game.
	pub(crate) simulation: &'a mut Simulation,
}

/// Advances the world by exactly one simulation step.
///
/// @param world - the host state the game reads and writes
/// @param parts - the subsystems this step drives
/// @param input - everything that has arrived since the previous step; the
/// edges in it are consumed here
/// @param time - the simulated time this step ends at, in seconds
/// @param editing - whether the world is being edited rather than played, in
/// which case neither the solver nor the game runs. @ref `crate::mode`
pub(crate) fn run(
	world: &mut World,
	parts: Parts<'_>,
	input: &mut Input,
	time: f32,
	editing: bool,
) {
	let Parts { game, interface, scripts, simulation } = parts;

	// the present becomes the past before the game touches anything. What the
	// renderer draws is somewhere between the two, and this is the one moment
	// in the step when they are the same thing.
	world.advance();

	world.time = time;
	world.dt = STEP_SECONDS;
	world.steps = world.steps.saturating_add(1);
	// written here rather than once a frame, so that whatever reads it inside
	// a step reads what this step is actually doing.
	world.editing = editing;

	// the debug geometry is swept here, at the *top* of the step, and not at the
	// bottom beside the input edges. Several frames are drawn between two steps
	// and every one of them draws this table, so clearing it at the end of a
	// step would erase exactly what those frames exist to show. @ref
	// `colby_core::abi::debug`.
	world.debug.begin_step(time);

	// and the playheads, at the top for a related reason: a sound the game
	// starts during this step should have been audible for none of it when the
	// step ends, and one started during the previous step for exactly one.
	// Outside the `editing` guard below, like the debug sweep and for the same
	// argument - time goes on while a world is being edited, and ambience that
	// stopped the moment somebody pressed F5 would be a bug rather than a
	// feature. @ref `colby_core::abi::audio`.
	world.audio.advance(&world.sounds, STEP_SECONDS);

	// hand the accumulated input over, then clear the parts that describe one
	// step only. A second step in the same frame therefore sees what is held
	// and none of the edges, which is what keeps one click from being four.
	world.input = *input;
	input.end_step();

	// the interface is tested against the pointer *before* the game runs, so
	// that a click is something `update` finds already waiting rather than
	// something that happens to it halfway through. This is the whole reason
	// hit-testing is in the step and not in the frame. @ref `colby_ui`.
	interface.update(world);

	// and the documents' own logic immediately after the events it just queued,
	// so that the whole interface half of a step happens together and the game
	// sees a world a script has already finished writing to. Inside the step
	// rather than the frame for the same reason the hit test is, plus one this
	// crate cares about more: `--shot` runs steps, so a screenshot shows what
	// the window shows. @ref `colby_script`.
	if let Some(scripts) = scripts {
		scripts.update(world);
	}

	// and the physics before the game too, and for a related reason: what
	// `update` reads is then the world as it now stands, and a trace it fires
	// answers about the state the next frame will draw rather than the one the
	// previous frame did. The cost is that a force applied here lands on the next
	// step, sixteen milliseconds later, which nothing can feel.
	//
	// Both of these are the whole of what edit mode turns off. Everything
	// above and below runs either way: the input edges still have to be
	// drained, the interface still has to be laid out and hit-tested, and the
	// debug table still has to be swept, or a world being edited would be one
	// nothing responded in. What stops is the two things that *move* it, which
	// is what makes a person the only writer while it is stopped.
	if !editing {
		simulation.step(world);

		if let Some(game) = game {
			game.update(world);
		}
	}

	// and last, whatever this step asked not to be interpolated: a teleport, a
	// wrap-around, an entity that did not exist a moment ago.
	world.settle();

	// the interface's events go the same way the input edges just did: a click
	// happened once, and a second step in the same frame must not see it again.
	world.ui.end_step();

	// and the physics events, for the same reason and beside them: a second
	// step in the same frame must not be told twice that two things met.
	world.bodies.end_step();
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{Body, Key, Shape, SoundData, Transform, Voice, debug},
		glam::Vec3,
	};

	use super::*;

	/// A world with one thing in it that falls if anything steps it.
	fn falling() -> (Box<World>, Box<Simulation>) {
		let simulation = Box::new(Simulation::new());
		let mut world = Box::<World>::default();
		world.install_physics(simulation.table());

		let entity = world
			.entities
			.spawn_at(Transform::at(Vec3::Y * 10.0));
		world.bodies.spawn(
			Body::dynamic(Shape::ball(0.5), Transform::at(Vec3::Y * 10.0), 1.0).driving(entity),
		);

		(world, simulation)
	}

	/// Runs a few steps in one mode.
	///
	/// The time each step ends at is computed from the count the world already
	/// holds, so two calls in a row do not run the clock backwards - which
	/// would make the debug table's sweep drop nothing.
	fn steps(
		world: &mut World,
		simulation: &mut Simulation,
		input: &mut Input,
		count: u32,
		editing: bool,
	) {
		let mut interface = Interface::new();

		for _ in 0..count {
			let ended = u32::try_from(world.steps.saturating_add(1)).unwrap_or(u32::MAX);
			let time = (colby_core::time::STEP * ended).as_secs_f32();

			let parts = Parts {
				game: None,
				interface: &mut interface,
				scripts: None,
				simulation,
			};

			run(world, parts, input, time, editing);
		}
	}

	#[test]
	fn nothing_falls_while_the_world_is_being_edited() {
		let (mut world, mut simulation) = falling();
		let mut input = Input::default();
		let body = world
			.bodies
			.iter()
			.next()
			.map(|(id, _)| id)
			.unwrap_or_default();
		let started = world
			.bodies
			.get(body)
			.map_or(0.0, |it| it.transform.position.y);

		steps(&mut world, &mut simulation, &mut input, 20, true);
		let edited = world
			.bodies
			.get(body)
			.map_or(0.0, |it| it.transform.position.y);

		assert!(
			(edited - started).abs() < f32::EPSILON,
			"the solver did not run, so nothing fell: {started} to {edited}"
		);

		steps(&mut world, &mut simulation, &mut input, 20, false);
		let played = world
			.bodies
			.get(body)
			.map_or(0.0, |it| it.transform.position.y);

		assert!(played < edited - 0.1, "and it falls again once it is played: {played}");
	}

	#[test]
	fn the_world_knows_which_mode_the_step_ran_in() {
		let (mut world, mut simulation) = falling();
		let mut input = Input::default();

		steps(&mut world, &mut simulation, &mut input, 1, true);
		assert!(world.editing, "a step told it was editing says so");

		steps(&mut world, &mut simulation, &mut input, 1, false);
		assert!(!world.editing, "and one told otherwise says that");
	}

	#[test]
	fn everything_but_the_simulation_still_runs_while_editing() {
		let (mut world, mut simulation) = falling();
		let mut input = Input::default();

		// a step's worth of debug geometry, and an input edge. Both are swept
		// by the step body rather than by the solver, so both are the check
		// that edit mode is not the same thing as no step at all - which is
		// what `sim.pause` is.
		world.debug.line(Vec3::ZERO, Vec3::X, debug::RED);
		input.set_key(Key::W, true);

		let was = world.steps;
		steps(&mut world, &mut simulation, &mut input, 1, true);

		assert_eq!(world.steps, was + 1, "the step ran and counted itself");
		assert!(world.time > 0.0, "and stamped the world with when it ended");
		assert!(world.debug.is_empty(), "and swept the geometry the previous one left");
		assert!(!input.pressed(Key::W), "and drained the input edges");
		assert!(input.held(Key::W), "without forgetting what is held");
	}

	#[test]
	fn the_step_carries_the_playheads_in_either_mode() {
		// audio goes with the debug sweep rather than with the solver: time
		// goes on while a world is being edited, and ambience that stopped the
		// moment somebody pressed F5 would be a bug. The test is in both modes
		// for exactly that reason.
		for editing in [false, true] {
			let (mut world, mut simulation) = falling();
			let mut input = Input::default();
			let sound = world.sounds.insert("sounds/test", SoundData {
				samples: vec![0; 1000],
				rate: 1000,
				channels: 1,
			});
			let voice = world.audio.play(Voice::flat(sound));

			steps(&mut world, &mut simulation, &mut input, 6, editing);

			let head = world
				.audio
				.get(voice)
				.map_or(0.0, |playing| playing.head);

			assert!(
				6.0_f32.mul_add(-STEP_SECONDS, head).abs() < 1e-6,
				"editing={editing}: six steps in, the playhead is six steps along: {head}"
			);

			steps(&mut world, &mut simulation, &mut input, 60, editing);

			assert!(
				!world.audio.alive(voice),
				"editing={editing}: and a second of sound is over after a second of steps"
			);
		}
	}
}
