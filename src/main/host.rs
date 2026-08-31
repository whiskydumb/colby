//! Serving a world to whoever turns up, with nothing on screen.
//!
//! `colby --host` is a *mode*, not a build. The same executable, the same game
//! module, the same step body - it simply opens a socket instead of a window
//! and never asks for an adapter. That was a decision rather than a default: an
//! engine whose dedicated server is a second build is an engine with two things
//! to keep working, and the thing that would justify one is a link-time saving
//! nobody has measured yet.
//!
//! It shares its loop with the window path in every way that matters and in one
//! way that does not: there is no vertical blank to pace against, so a pass
//! that ran no step sleeps for a moment rather than spinning a core at nothing.
//!
//! **No output device either.** A machine serving a world has no reason to make
//! a noise, and one that did would be making it into somebody's empty room.
//! Everything else about the step is what a window would have run.

use std::{
	thread,
	time::{Duration, Instant},
};

use colby_core::{
	Result,
	abi::{Input, World},
	glam::Vec2,
	info,
	time::{Clock, STEP},
};
use colby_physics::Simulation;
use colby_script::Scripts;
use colby_ui::Interface;

use crate::{assets::Assets, console::Console, game::Game, net::Net, step};

/// The flag that asks for a host.
const FLAG: &str = "--host";

/// How long a pass that ran no step waits before looking again.
///
/// Short enough that a step is never late by anything a person could measure,
/// and long enough that an idle host is not a core at full tilt.
const IDLE: Duration = Duration::from_millis(1);

/// The viewport a host lays its documents out against.
///
/// There is no window, and a document laid out against a different size would
/// put its boxes somewhere else - so a host uses the same numbers a screenshot
/// does, for the same reason. Nothing draws any of it.
const VIEWPORT: Vec2 = Vec2::new(1280.0, 720.0);

/// Reads the command line for a host.
///
/// Accepts `--host` on its own, `--host port` and `--host=port`.
///
/// @return which port to listen on, if a host was asked for
#[must_use]
pub(crate) fn requested() -> Option<u16> {
	let arguments: Vec<String> = std::env::args().skip(1).collect();

	parse(&arguments)
}

/// The same, over arguments already collected.
fn parse(arguments: &[String]) -> Option<u16> {
	for (index, argument) in arguments.iter().enumerate() {
		let port = if let Some(rest) = argument.strip_prefix(&format!("{FLAG}=")) {
			rest.parse::<u16>().ok()
		} else if argument == FLAG {
			arguments
				.get(index + 1)
				.and_then(|next| next.parse::<u16>().ok())
		} else {
			continue;
		};

		return Some(port.unwrap_or(crate::net::DEFAULT_PORT));
	}

	None
}

/// Brings a world up, opens a socket, and serves until somebody says stop.
///
/// @param port - what to listen on
pub(crate) fn serve(port: u16) -> Result {
	// boxed and installed before anything else touches the world, for the
	// reason the window and the screenshot box it: the world keeps this
	// address.
	let mut simulation = Box::new(Simulation::new());
	let mut world = Box::<World>::default();
	world.install_physics(simulation.table());

	let mut assets = Assets::new(&crate::workspace());
	assets.sync(&mut world);

	world.ui.set_viewport(VIEWPORT, 1.0);
	world.aspect = VIEWPORT.x / VIEWPORT.y;

	// the host's own variables before the module, so they belong to the engine
	// and survive a reload, exactly as the window path does it.
	crate::console::install(&mut world);

	let mut game = Game::open(&mut world)?;
	let console = Console::open(&mut world);
	let mut net = Net::host(port, crate::net::seed(&world.cvars))?;
	let mut scripts = Scripts::new()?;
	let mut interface = Interface::new();
	let mut input = Input::default();
	let mut clock = Clock::new();
	let started = Instant::now();

	info!(%port, address = %net.address(), "serving");

	while !world.quit {
		clock.tick();
		assets.poll(&mut world);
		console.poll(&mut world);
		crate::saves::serve(&mut world, &mut simulation);
		crate::net::serve(Some(&mut net));
		net.set(crate::net::conditions(&world.cvars));

		// off the wire before any step, so that what a step is about to run
		// against is everything that had arrived when it started rather than
		// whatever turned up halfway through. @ref `crate::net`.
		net.receive(started.elapsed());

		let mut ran = false;

		while let Some(time) = clock.step() {
			step::run(
				&mut world,
				step::Parts {
					game: Some(&mut game),
					interface: &mut interface,
					scripts: Some(&mut scripts),
					simulation: simulation.as_mut(),
					// a machine serving a world has nothing to make a noise
					// into. @ref the module comment.
					audio: None,
				},
				&mut input,
				time,
				false,
			);

			// and out once a step rather than once a pass, because a message
			// has to mean "this is where things stand at this moment".
			net.send(started.elapsed());
			ran = true;
		}

		if !ran {
			thread::sleep(IDLE);
		}
	}

	game.close(&mut world);
	console.close(&world);

	info!(
		steps = world.steps,
		peers = net.peers(),
		sent = net.sent(),
		delivered = net.delivered(),
		seconds = started.elapsed().as_secs_f32(),
		"colby stopped serving"
	);

	Ok(())
}

/// How long one step is, for whoever is reading the loop above.
const _: () = assert!(STEP.as_nanos() > 0, "a step has to take some time");

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_flag_is_read_on_its_own_and_with_a_port() {
		assert_eq!(parse(&["--host".to_owned()]), Some(crate::net::DEFAULT_PORT));
		assert_eq!(parse(&["--host".to_owned(), "9999".to_owned()]), Some(9999));
		assert_eq!(parse(&["--host=9999".to_owned()]), Some(9999));
		assert_eq!(parse(&["--connect".to_owned()]), None, "and not somebody else's");
		assert_eq!(parse(&[]), None);
	}

	#[test]
	fn a_word_after_the_flag_that_is_not_a_port_is_not_eaten() {
		assert_eq!(
			parse(&["--host".to_owned(), "--shot".to_owned()]),
			Some(crate::net::DEFAULT_PORT),
			"the next word is only a port when it is one"
		);
		assert_eq!(
			parse(&["--host".to_owned(), "99999".to_owned()]),
			Some(crate::net::DEFAULT_PORT),
			"and a number that is not a port is not one"
		);
	}
}
