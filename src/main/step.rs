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

/// Advances the world by exactly one simulation step.
///
/// @param world - the host state the game reads and writes
/// @param game - the loaded module, if there is one
/// @param interface - the game's own interface, laid out and hit-tested here
/// @param scripts - the documents' own interface logic, if Lua came up
/// @param simulation - the physics, advanced here and queried by the game
/// @param input - everything that has arrived since the previous step; the
/// edges in it are consumed here
/// @param time - the simulated time this step ends at, in seconds
pub(crate) fn run(
	world: &mut World,
	game: Option<&mut Game>,
	interface: &mut Interface,
	scripts: Option<&mut Scripts>,
	simulation: &mut Simulation,
	input: &mut Input,
	time: f32,
) {
	// the present becomes the past before the game touches anything. What the
	// renderer draws is somewhere between the two, and this is the one moment
	// in the step when they are the same thing.
	world.advance();

	world.time = time;
	world.dt = STEP_SECONDS;
	world.steps = world.steps.saturating_add(1);

	// the debug geometry is swept here, at the *top* of the step, and not at the
	// bottom beside the input edges. Several frames are drawn between two steps
	// and every one of them draws this table, so clearing it at the end of a
	// step would erase exactly what those frames exist to show. @ref
	// `colby_core::abi::debug`.
	world.debug.begin_step(time);

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
	simulation.step(world);

	if let Some(game) = game {
		game.update(world);
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
