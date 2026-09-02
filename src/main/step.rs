//! One simulation step, wherever it is driven from.
//!
//! The window paces these off a [`Clock`](colby_core::time::Clock); `--shot`
//! runs a fixed number of them as fast as the machine will go. What is shared
//! between the two is this body and not the clock, which is what makes a
//! screenshot deterministic by construction rather than by care: there is one
//! definition of what a step does, and nothing in it depends on how much real
//! time has gone by.

use std::time::Duration;

use colby_audio::Device;
use colby_core::{
	abi::{Input, World},
	time::STEP_SECONDS,
};
use colby_physics::Simulation;
use colby_script::Scripts;
use colby_ui::Interface;

use crate::{game::Game, net::Net};

/// Everything a step drives that outlives it.
///
/// One struct rather than four arguments: both call sites hold all four beside
/// each other and hand over the same four every time, and a signature that
/// grows every time a subsystem appears is a signature nobody reads. What is
/// *not* in here is the per-step half - the world, the input, the moment and
/// the mode - because those are what the step is actually about.
pub(crate) struct Wired<'a> {
	/// The endpoint.
	pub(crate) net: &'a mut Net,

	/// How long this process has been running.
	pub(crate) now: Duration,
}

pub(crate) struct Parts<'a> {
	/// The loaded module, if there is one.
	pub(crate) game: Option<&'a mut Game>,

	/// The game's own interface, laid out and hit-tested here.
	pub(crate) interface: &'a mut Interface,

	/// The documents' own interface logic, if Lua came up.
	pub(crate) scripts: Option<&'a mut Scripts>,

	/// The physics, advanced here and queried by the game.
	pub(crate) simulation: &'a mut Simulation,

	/// The endpoint, and the moment this process is at on its own clock.
	///
	/// The two travel together because neither means anything without the
	/// other: a world that arrived is placed against the clock it arrived on,
	/// and that clock is the wire's rather than the simulation's. `--shot` and
	/// `--record` pass `None` and are untouched by any of it, which is the
	/// property nothing in this step may disturb.
	pub(crate) wire: Option<Wired<'a>>,

	/// The output device, if one opened.
	///
	/// Written to at the very bottom of the step and read from nowhere: what
	/// the mixer needs is a copy of the voice table as the step left it, and
	/// the step is the only place that knows when that is. `--shot` has none
	/// of this and is unchanged by it, which is the property that keeps a
	/// screenshot reproducible.
	pub(crate) audio: Option<&'a mut Device>,
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
	let Parts {
		game,
		interface,
		scripts,
		simulation,
		audio,
		wire,
	} = parts;

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
		// what the far end said, before the solver reads it: a body this end
		// does not own is driven rather than simulated, and what drives it is
		// the entity the solver is about to copy into it.
		//
		// Inside the edit-mode guard with the solver and the game, and for the
		// same reason: what stops while a world is being edited is everything
		// that *moves* it, and a wire is a third thing that moves it.
		if let Some(wire) = wire {
			wire.net.arrive(
				world,
				wire.now
					.saturating_sub(crate::net::behind(&world.cvars)),
			);
		}

		simulation.step(world);

		if let Some(game) = game {
			game.update(world);
		}

		// and every command that was in front of this step is done with,
		// whether or not the game looked at one.
		//
		// **This is `Input::end_step`'s rule and not a new one.** The host
		// hands the edges over and clears them afterwards, so that one click
		// is one click even for a game that never asked; a command is offered
		// for exactly one step for the same reason. A game that ignored it
		// does not get it again, and one that ran it cannot run it twice - and
		// neither promise is something every game has to remember to keep,
		// which is the whole argument for the mark living in the table.
		//
		// Inside the edit-mode guard with the solver and the game, because a
		// command nothing had a chance to act on is not a command anybody has
		// finished with.
		//
		// Only where this process decides. A client's own ring is settled by
		// what the host confirms, and a client marking its own commands done
		// would be a client acknowledging itself.
		if world.peer.is_host() {
			for (peer, _) in world.players.iter() {
				let newest = world.commands.newest(peer);

				world.commands.settle(peer, newest);
			}
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

	// last of all, and after `settle` rather than before it: what crosses to
	// the mixer is where every voice stands at the end of this step, gains and
	// all, and a snapshot taken any earlier would describe a world the game
	// had not finished writing. @ref `colby_audio::snapshot`.
	if let Some(audio) = audio {
		audio.publish(world);
	}
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{Body, BodyId, Command, EntityId, Key, Shape, SoundData, Transform, Voice, debug},
		glam::Vec3,
	};

	use super::*;

	/// A client with a body driving an entity, and two snapshots already taken
	/// a tenth of a second apart at nought and ten along an axis.
	fn wired_client() -> (Net, Box<World>, Box<Simulation>, EntityId, BodyId) {
		use std::{
			cell::RefCell,
			net::{IpAddr, Ipv4Addr, SocketAddr},
			rc::Rc,
		};

		use colby_net::{NOTHING, Snapshot, Solid};

		use crate::net::{Loopback, Wire};

		let at = |port| SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
		let shared = Rc::new(RefCell::new(Wire::default()));
		let mut net = Net::over(Box::new(Loopback::at(at(2), &shared)), false, 2);
		let simulation = Box::new(Simulation::new());
		let mut world = Box::<World>::default();

		world.install_physics(simulation.table());
		crate::net::install(&mut world.cvars);
		crate::net::joined(&mut world);
		net.introduce(at(1));

		let entity = world.entities.spawn();
		let body = world
			.bodies
			.spawn(Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0).driving(entity));

		for (number, along, when) in [(1_u32, 0.0_f32, 0_u64), (2, 10.0, 100)] {
			let mut table = vec![None; body.slot() + 1];
			let solid = Solid {
				position: [along, 0.0, 0.0],
				rotation: [0.0, 0.0, 0.0, 1.0],
				scale: [1.0, 1.0, 1.0],
				// the kind a host simulating it would send. `Solid::default`
				// is a static body and would be passed through as one.
				kind: 2,
				..Solid::default()
			};
			let mut out = Vec::new();

			table[body.slot()] = Some((body.generation(), solid));
			Snapshot::write(number, NOTHING, NOTHING, &[], &table, &mut out).expect("fits");
			assert!(net.absorbed(&out, Duration::from_millis(when)));
		}

		(net, world, simulation, entity, body)
	}

	/// Runs one step of a client at a moment on the wire's clock.
	fn stepped_at(net: &mut Net, world: &mut World, simulation: &mut Simulation, now: u64) {
		let parts = Parts {
			game: None,
			interface: &mut Interface::new(),
			scripts: None,
			simulation,
			audio: None,
			wire: Some(Wired { net, now: Duration::from_millis(now) }),
		};

		run(world, parts, &mut Input::default(), 0.0, false);
	}

	/// Where an entity is, and where it is drawn coming from.
	fn drawn(world: &World, entity: EntityId) -> (Option<f32>, Option<f32>) {
		(
			world
				.entities
				.transform(entity)
				.map(|it| it.position.x),
			world
				.entities
				.interpolated(entity, 0.0)
				.map(|it| it.position.x),
		)
	}

	/// A step with nothing in it but a world, for the questions that are about
	/// the step body rather than about the wire.
	fn plain(world: &mut World, simulation: &mut Simulation, editing: bool) {
		let parts = Parts {
			game: None,
			interface: &mut Interface::new(),
			scripts: None,
			simulation,
			audio: None,
			wire: None,
		};

		run(world, parts, &mut Input::default(), 0.0, editing);
	}

	/// A command is offered for one step, whoever does or does not act on it.
	///
	/// The same rule the input edges have, and it is the engine's rather than
	/// every game's: nothing here runs a command, and the mark still moves.
	#[test]
	fn a_step_is_done_with_every_command_that_was_in_front_of_it() {
		let mut simulation = Box::new(Simulation::new());
		let mut world = Box::<World>::default();

		world.install_physics(simulation.table());

		let peer = world.players.admit();

		// numbers that are neither one nor consecutive with the step count, so
		// a mark taken from the wrong number is visible.
		for number in [7_u32, 8, 12] {
			assert!(world.commands.push(peer, asked(number)));
		}

		assert_eq!(world.commands.unsettled(peer).len(), 3, "all three are waiting");

		plain(&mut world, &mut simulation, false);

		assert_eq!(world.commands.settled(peer), 12, "the newest that was there");
		assert!(world.commands.unsettled(peer).is_empty(), "so none is offered twice");

		// and one that arrives afterwards is offered once, on the next step.
		assert!(world.commands.push(peer, asked(13)));
		assert_eq!(world.commands.unsettled(peer), &[asked(13)]);

		plain(&mut world, &mut simulation, false);
		assert_eq!(world.commands.settled(peer), 13);
	}

	#[test]
	fn a_world_being_edited_is_not_done_with_anybodys_commands() {
		let mut simulation = Box::new(Simulation::new());
		let mut world = Box::<World>::default();

		world.install_physics(simulation.table());

		let peer = world.players.admit();

		assert!(world.commands.push(peer, asked(4)));

		// nothing that moves a world runs while it is being edited, and a
		// command nobody had a chance to act on is not one anybody is done
		// with.
		plain(&mut world, &mut simulation, true);
		assert_eq!(world.commands.settled(peer), 0, "still waiting");
		assert_eq!(world.commands.unsettled(peer), &[asked(4)]);

		plain(&mut world, &mut simulation, false);
		assert_eq!(world.commands.settled(peer), 4, "and offered the moment play resumes");
	}

	#[test]
	fn a_client_does_not_mark_its_own_commands_done() {
		let mut simulation = Box::new(Simulation::new());
		let mut world = Box::<World>::default();

		world.install_physics(simulation.table());

		let peer = world.players.admit();

		assert!(world.commands.push(peer, asked(4)));

		// this end is not the one that decides, so what it may be holding is
		// somebody else's to be done with. A client that settled its own ring
		// would be acknowledging itself and would stop asking for a move the
		// host never saw.
		crate::net::joined(&mut world);
		plain(&mut world, &mut simulation, false);

		assert_eq!(world.commands.settled(peer), 0, "nobody here has run it");
		assert_eq!(world.commands.unsettled(peer), &[asked(4)]);
	}

	/// One command, with four numbers that are four different numbers.
	fn asked(number: u32) -> Command {
		Command {
			step: u64::from(number) * 5 + 900,
			number,
			buttons: 1 << (number % 6),
			yaw: f32::from(u16::try_from(number % 11).unwrap_or(0)) * 0.75,
			pitch: -1.25,
		}
	}

	#[test]
	fn a_step_draws_the_world_a_client_was_told_about_a_delay_behind() {
		// the only test over the wiring rather than over the pieces, and the
		// only thing that can see the delay being applied at all: `Net::arrive`
		// is handed a moment, so a step that handed it the wrong one would be
		// invisible everywhere else.
		//
		// The default delay is a tenth, so a step at a hundred and fifty
		// milliseconds draws the moment at fifty - half way between the two
		// snapshots rather than on either. Landing *on* one is the trap: at
		// two hundred a step that forgot the delay would answer with snapshot
		// two as well, because a moment past the newest is the newest.
		let (mut net, mut world, mut simulation, entity, body) = wired_client();

		stepped_at(&mut net, &mut world, &mut simulation, 150);

		assert_eq!(
			world.bodies.get(body).map(|it| it.kind),
			Some(colby_core::abi::BodyKind::Kinematic),
			"the body the far end owns stopped being simulated"
		);
		assert_eq!(
			drawn(&world, entity).0,
			Some(5.0),
			"and the world drawn is the one a tenth of a second behind, not the newest"
		);
	}

	#[test]
	fn the_first_step_a_body_is_driven_is_a_cut_and_the_next_one_is_not() {
		// the first time, it comes from wherever this end's own solver had
		// left it - the origin here - so blending towards the truth would draw
		// it sliding in from there. `interpolated` at nought is the *previous*
		// pose, and after a cut the previous pose is the new one.
		let (mut net, mut world, mut simulation, entity, _) = wired_client();

		stepped_at(&mut net, &mut world, &mut simulation, 150);

		assert_eq!(drawn(&world, entity), (Some(5.0), Some(5.0)), "cut to, this first time");

		// and again, with the body already driven: now it moves, so it is
		// interpolated like anything else. Two hundred milliseconds rather
		// than some number in between, because a moment that lands exactly on
		// a snapshot is the only one whose answer is a float somebody can
		// write down.
		stepped_at(&mut net, &mut world, &mut simulation, 200);

		assert_eq!(
			drawn(&world, entity),
			(Some(10.0), Some(5.0)),
			"and then drawn coming from where it was, rather than cut to"
		);
	}

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
				audio: None,
				wire: None,
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
