//! Whether the world is being played or edited, and what happens in between.
//!
//! Two modes and one variable. **Playing** is what the process has always
//! done: the game's `update` runs, the solver steps, and everything moves.
//! **Editing** stops both of those and nothing else, so the world holds still
//! and whoever is looking at it is the only thing writing to it.
//!
//! **Why the mode exists at all, rather than `sim.pause`.** Pausing sets the
//! clock's speed to zero, so no step runs - and a step is what sweeps the
//! debug table, lays out the interface and drains the input edges. A world
//! being edited wants all of those and wants only the *simulation* stopped.
//! The two compose: a paused world in edit mode is a world where even time has
//! stopped.
//!
//! **What happens at the boundary, and it is the whole point.** Starting to
//! play writes the world down; stopping puts it back. That is
//! [`capture`](scene::capture) and [`restore`](scene::restore), which the
//! scene format already had, and it is why editing a world is not the same as
//! damaging it: what a game does to a world while it runs is undone when it
//! stops. The one obligation that comes with a restore comes with this too -
//! the solver is told to forget everything it derived, because a step run
//! against a stale contact cache is a step mixing two worlds.
//!
//! **The first stop is the exception, and it is deliberate.** The process
//! starts playing, so the first time somebody presses F5 there is no captured
//! world to go back to. What happens then is nothing: the world being played
//! becomes the world being edited. There is nothing better to do, and it is
//! also what somebody who started the engine and then wanted to move a crate
//! actually meant.
//!
//! **The mode does not know about the editor.** It decides whether gameplay
//! runs, which is the runner's business; a build with no editor in it still
//! has the variable and it still works, the same way `sim.pause` does. The one
//! thing here that *is* the editor's is [`toggle`], which exists because
//! something has to answer a key, and only a build with tools in it binds
//! one.

use colby_core::{
	abi::{
		World,
		scene::{self, SceneData},
	},
	error, info,
};
use colby_physics::Simulation;

/// The variable that decides whether the game is running.
pub(crate) const EDIT: &str = "sim.edit";

/// What the variable says right now.
///
/// A world whose console never registered it - a test, a screenshot - is being
/// played, which is what everything did before there was a mode at all.
///
/// @param world - where the variable lives
pub(crate) fn wanted(world: &World) -> bool { world.cvars.bool(EDIT).unwrap_or(false) }

/// Asks for the other mode.
///
/// Writes the variable rather than the state: a key, a button and a typed line
/// then reach the mode by one path, and the frame loop acts on the edge. That
/// is the same arrangement the editor's own visibility has, and it is what
/// makes `sim.edit 1` and F5 the same thing rather than two.
///
/// Only built with the editor, because only that build binds a key to it. A
/// shipping build keeps the variable and nothing in it presses this.
///
/// @param world - where the variable lives
#[cfg(feature = "editor")]
pub(crate) fn toggle(world: &mut World) {
	let editing = wanted(world);

	world
		.cvars
		.set(EDIT, if editing { "false" } else { "true" });
}

/// Which mode the world is in, and the world play started from.
#[derive(Debug, Default)]
pub(crate) struct Mode {
	/// What the variable said last time it was looked at.
	///
	/// Compared rather than watched, the same way the host's gravity is: a
	/// variable is a value somebody may write from anywhere, and an edge is
	/// the only thing worth acting on.
	editing: bool,

	/// The world play started from, waiting to be put back.
	///
	/// `None` while the world is being edited, and also during the very first
	/// stretch of play, which nobody asked to start. @ref the module note
	/// about the first stop.
	before: Option<Box<SceneData>>,
}

impl Mode {
	/// A world that is being played, as every process starts out.
	pub(crate) const fn new() -> Self { Self { editing: false, before: None } }

	/// How far the frame being drawn sits past the last simulated state.
	///
	/// One while editing, whatever the clock says. Nothing is being simulated,
	/// so there is nothing to smooth - and that is what lets a panel write a
	/// transform once a *frame* and have it drawn exactly where it was put,
	/// instead of blended back towards wherever the last step left it. A drag
	/// that used to be rubbery is straight, and it took no new mechanism: the
	/// blend was only ever there to hide a rate difference that edit mode does
	/// not have.
	///
	/// @param paced - what the clock says, for a world that is being played
	pub(crate) const fn interpolation(&self, paced: f32) -> f32 {
		if self.editing { 1.0 } else { paced }
	}

	/// Acts on the variable having moved since the last frame.
	///
	/// The frame loop's, called once a frame and **outside a simulation
	/// step**: stopping replaces every table in the world, and doing that
	/// halfway through a step would leave the rest of the step running against
	/// a world its first half never saw. The same rule, and the same reason, as
	/// a scene load from the console.
	///
	/// @param world - the world to write down or put back
	/// @param simulation - the solver, told to forget what it derived
	/// @param wanted - what the variable says now
	/// @return whether the mode changed
	pub(crate) fn follow(
		&mut self,
		world: &mut World,
		simulation: &mut Simulation,
		wanted: bool,
	) -> bool {
		if wanted == self.editing {
			return false;
		}

		self.editing = wanted;

		if wanted {
			self.stop(world, simulation);
		} else {
			self.start(world);
		}

		true
	}

	/// Writes the world down, because play is starting.
	fn start(&mut self, world: &World) {
		// boxed: a description of a full world is tens of kilobytes, and this
		// sits inside the one struct the whole process is held in.
		self.before = Some(Box::new(scene::capture(world)));

		info!(
			entities = world.entities.len(),
			bodies = world.bodies.len(),
			"playing; the world was written down and comes back when this stops"
		);
	}

	/// Puts the world back, because play is stopping.
	///
	/// Nothing to put back is not a failure: the process began mid-play and
	/// there was never a moment to write down. The world stays where it is and
	/// becomes the one being edited.
	fn stop(&mut self, world: &mut World, simulation: &mut Simulation) {
		let Some(before) = self.before.take() else {
			info!(
				entities = world.entities.len(),
				"editing; nothing was written down to go back to, so this world is the one"
			);

			return;
		};

		match scene::restore(world, &before) {
			| Ok(put) => {
				// immediately, and it is this caller's obligation rather than
				// something `restore` can discharge: baked collision meshes,
				// the pairs that were touching and the impulse cache all live
				// outside the world. @ref `Simulation::forget`.
				simulation.forget();

				info!(
					entities = put.things,
					bodies = put.solids,
					joints = put.links,
					state = put.arena,
					"editing; the world play started from is back"
				);
			},
			| Err(failure) => {
				// the one way this fails is a game whose state changed shape
				// while it was being played - a hot-reload that bumped the
				// layout number. Half a world is worse than none, so the
				// played world is kept and the capture is dropped rather than
				// held for a stop that can never take it.
				error!(
					%failure,
					"editing; the world play started from cannot be put back, so this one stays"
				);
			},
		}
	}
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{Body, Shape, Transform},
		glam::Vec3,
	};

	use super::*;

	/// A world with one thing in it that would move if anything stepped it.
	fn falling() -> World {
		let mut world = World::new();
		let entity = world
			.entities
			.spawn_at(Transform::at(Vec3::Y * 10.0));
		world.entities.set_name(entity, "crate");
		world.bodies.spawn(
			Body::dynamic(Shape::ball(0.5), Transform::at(Vec3::Y * 10.0), 1.0).driving(entity),
		);

		world
	}

	// as the function it covers: a shipping build has no key to press.
	#[cfg(feature = "editor")]
	#[test]
	fn the_key_and_the_typed_line_reach_the_mode_the_same_way() {
		let mut world = World::new();
		world
			.cvars
			.var(EDIT, colby_core::abi::Value::Bool(false), "");

		assert!(!wanted(&world), "a world starts out being played");

		toggle(&mut world);
		assert!(wanted(&world), "and the toggle writes the variable rather than the state");

		world.cvars.set(EDIT, "false");
		assert!(!wanted(&world), "which is the same thing a typed line writes");
	}

	#[test]
	fn a_world_being_edited_is_drawn_where_it_stands_rather_than_blended() {
		let mut world = World::new();
		let mut simulation = Simulation::new();
		let mut mode = Mode::new();

		assert!(
			(mode.interpolation(0.25) - 0.25).abs() < f32::EPSILON,
			"a world being played is blended by whatever the clock says"
		);

		mode.follow(&mut world, &mut simulation, true);

		assert!(
			(mode.interpolation(0.25) - 1.0).abs() < f32::EPSILON,
			"and one being edited is drawn as it stands"
		);
	}

	#[test]
	fn a_world_whose_console_never_registered_the_variable_is_played() {
		assert!(!wanted(&World::new()), "a test and a screenshot both play");
	}

	#[test]
	fn a_world_is_written_down_when_play_starts_and_put_back_when_it_stops() {
		let mut world = falling();
		let mut simulation = Simulation::new();
		let mut mode = Mode::new();

		// into edit first, which is the transition that has nothing to undo.
		assert!(mode.follow(&mut world, &mut simulation, true), "the mode moved");
		assert!(mode.editing, "and it is the one that was asked for");

		let entity = world
			.entities
			.iter()
			.next()
			.map(|(id, ..)| id)
			.unwrap_or_default();
		if let Some(transform) = world.entities.transform_mut(entity) {
			transform.position = Vec3::new(3.0, 4.0, 5.0);
		}

		mode.follow(&mut world, &mut simulation, false);
		if let Some(transform) = world.entities.transform_mut(entity) {
			transform.position = Vec3::ZERO;
		}

		mode.follow(&mut world, &mut simulation, true);

		assert_eq!(
			world
				.entities
				.transform(entity)
				.map(|it| it.position),
			Some(Vec3::new(3.0, 4.0, 5.0)),
			"stopping put back the world play started from, not the one play left"
		);
		assert_eq!(world.entities.name(entity), "crate", "names and all");
	}

	#[test]
	fn the_first_stop_keeps_the_world_it_was_playing() {
		let mut world = falling();
		let mut simulation = Simulation::new();
		let mut mode = Mode::new();

		let entity = world
			.entities
			.iter()
			.next()
			.map(|(id, ..)| id)
			.unwrap_or_default();
		if let Some(transform) = world.entities.transform_mut(entity) {
			transform.position = Vec3::new(1.0, 2.0, 3.0);
		}

		mode.follow(&mut world, &mut simulation, true);

		assert_eq!(
			world
				.entities
				.transform(entity)
				.map(|it| it.position),
			Some(Vec3::new(1.0, 2.0, 3.0)),
			"there was nothing written down, so nothing was undone"
		);
	}

	#[test]
	fn a_mode_that_did_not_move_does_nothing_at_all() {
		let mut world = falling();
		let mut simulation = Simulation::new();
		let mut mode = Mode::new();

		assert!(!mode.follow(&mut world, &mut simulation, false), "it was already playing");
		assert!(mode.before.is_none(), "and nothing was written down for a stop to spend");

		mode.follow(&mut world, &mut simulation, true);
		assert!(!mode.follow(&mut world, &mut simulation, true), "nor the other way");
	}

	#[test]
	fn the_capture_is_spent_rather_than_kept() {
		let mut world = falling();
		let mut simulation = Simulation::new();
		let mut mode = Mode::new();

		mode.follow(&mut world, &mut simulation, true);
		mode.follow(&mut world, &mut simulation, false);
		assert!(mode.before.is_some(), "play wrote the world down");

		mode.follow(&mut world, &mut simulation, true);
		assert!(
			mode.before.is_none(),
			"and stopping spent it, so a second stop has nothing stale to put back"
		);
	}

	#[test]
	fn a_world_whose_game_changed_shape_is_left_alone_rather_than_half_loaded() {
		let mut world = falling();
		let mut simulation = Simulation::new();
		let mut mode = Mode::new();

		// a game claims the arena, so the capture carries a layout number.
		world.state.put_raw(&[7; 8], 3);
		mode.follow(&mut world, &mut simulation, true);
		mode.follow(&mut world, &mut simulation, false);

		// and the build the world is stopped under claims another one, which
		// is what a hot-reload that bumped the layout looks like from here.
		world.state.put_raw(&[9; 8], 4);
		let entity = world
			.entities
			.iter()
			.next()
			.map(|(id, ..)| id)
			.unwrap_or_default();
		if let Some(transform) = world.entities.transform_mut(entity) {
			transform.position = Vec3::X;
		}

		mode.follow(&mut world, &mut simulation, true);

		assert_eq!(
			world
				.entities
				.transform(entity)
				.map(|it| it.position),
			Some(Vec3::X),
			"the played world is kept rather than half replaced"
		);
		assert!(mode.before.is_none(), "and the capture nobody can spend is dropped");
	}
}
