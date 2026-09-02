//! Being on a wire with nothing on screen, at either end of it.
//!
//! `colby --host` and `colby --join` are *modes*, not builds. The same
//! executable, the same game module, the same step body - they simply open a
//! socket instead of a window and never ask for an adapter. That was a decision
//! rather than a default: an engine whose dedicated server is a second build is
//! an engine with two things to keep working, and the thing that would justify
//! one is a link-time saving nobody has measured yet.
//!
//! **`--join` exists so that two colbys can be run by a script.** A window
//! takes an adapter, a surface and the focus, so a client that only ran in one
//! could not be started beside a host and driven while somebody was at the
//! machine - and everything a client does differently from a host was
//! therefore being written blind. This is the same loop with the endpoint
//! pointed the other way: it asks the host for things instead of answering,
//! and it is told what the world is instead of saying. What it is *not* is a
//! second client: `--connect` opens a window and runs this same step through
//! `crate::app`, and the day the two disagree is a bug in one of them.
//!
//! It shares its loop with the window path in every way that matters and in one
//! way that does not: there is no vertical blank to pace against, so a pass
//! that ran no step sleeps for a moment rather than spinning a core at nothing.
//!
//! **No output device either.** A machine serving a world has no reason to make
//! a noise, and one that did would be making it into somebody's empty room.
//! Everything else about the step is what a window would have run.

use std::{
	net::SocketAddr,
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
use colby_net::{EVERY, Slot};
use colby_physics::Simulation;
use colby_script::Scripts;
use colby_ui::Interface;

use crate::{assets::Assets, console::Console, game::Game, net::Net, step};

/// The flag that asks for a host.
const FLAG: &str = "--host";

/// The flag that asks for a client with nothing on screen.
const JOIN: &str = "--join";

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

/// Reads the command line for a client with no window.
///
/// Accepts `--join address` and `--join=address`, where an address is whatever
/// the standard library reads as one - the same shapes `--connect` takes,
/// because it is the same question asked by a mode that draws nothing.
///
/// @return where the host is, if a windowless client was asked for
#[must_use]
pub(crate) fn joining() -> Option<SocketAddr> {
	let arguments: Vec<String> = std::env::args().skip(1).collect();

	address(&arguments)
}

/// The same, over arguments already collected.
///
/// The reading itself is `crate::net`'s, because `--connect` asks the identical
/// question and an address is an address. @ref `crate::net::after`.
fn address(arguments: &[String]) -> Option<SocketAddr> { crate::net::after(arguments, JOIN) }

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

/// Which end of a wire a windowless process is.
///
/// A two-variant enum rather than a boolean and an address, because the two
/// carry different things and the pair of them can disagree: a port means
/// nothing to a client and an address means nothing to a host.
#[derive(Clone, Copy, Debug)]
enum End {
	/// Listens on a port, decides everything, tells everybody.
	Serving(u16),

	/// Goes looking for a host, decides nothing, asks.
	Asking(SocketAddr),
}

/// Brings a world up, opens a socket, and serves until somebody says stop.
///
/// @param port - what to listen on
pub(crate) fn serve(port: u16) -> Result { run(End::Serving(port)) }

/// The same, at the other end: a client with nothing on screen.
///
/// @param address - where the host is
pub(crate) fn join(address: SocketAddr) -> Result { run(End::Asking(address)) }

/// The loop both ends share.
///
/// **One body and not two**, which is the whole reason this is worth writing
/// down: a host and a client differ in four lines - which way the socket is
/// opened, whether this process stops being the authority, whether the step is
/// handed the wire, and whether what goes out is a world or a request - and
/// every other thing they do is the same thing. Two loops would drift, and the
/// drift would be exactly the sort of difference nobody notices until two
/// machines disagree about where somebody is standing.
///
/// @param end - which end of the wire this process is
fn run(end: End) -> Result {
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

	// **the socket before the module**, and a client says what it is before
	// the module's `init` reads it. That ordering is the window path's, not a
	// choice made here: `crate::app` opens its endpoint and calls
	// `crate::net::joined` before it opens the game, and this mode exists to
	// behave the way a window does rather than better than one.
	let seed = crate::net::seed(&world.cvars);
	let mut net = match end {
		| End::Serving(port) => Net::host(port, seed)?,
		| End::Asking(address) => Net::connect(address, seed)?,
	};
	let hosting = matches!(end, End::Serving(_));

	if !hosting {
		crate::net::joined(&mut world);
	}

	let mut game = Game::open(&mut world)?;
	let console = Console::open(&mut world);
	let mut scripts = Scripts::new()?;
	let mut interface = Interface::new();
	let mut input = Input::default();
	let mut clock = Clock::new();
	let started = Instant::now();
	// taken down once a step and sent, rather than allocated per snapshot.
	let mut records: Vec<Slot> = Vec::new();

	// where the wire's own clock stands for the step about to run. A client
	// draws the world a delay behind what it was told, and the delay is
	// measured against this rather than against the wall, so that a pass which
	// runs four catch-up steps places four moments rather than one. @ref
	// `crate::app`, which keeps the same number for the same reason.
	let mut moment = Duration::ZERO;

	if hosting {
		info!(address = %net.address(), "serving");
	} else {
		info!(address = %net.address(), "asking");
	}

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
		// and then the world hears about it: whoever turned up is given a
		// name, and what they asked for is filed under it. Between the drain
		// and the step, so that a peer's very first message is one the step
		// after it can act on. @ref `Net::seat`.
		net.seat(&mut world);
		// and whatever they typed, if it is theirs to type. @ref
		// `crate::net::allowed`.
		crate::net::obey(&mut world, &net);

		let mut ran = false;

		while let Some(time) = clock.step() {
			step::run(
				&mut world,
				step::Parts {
					game: Some(&mut game),
					interface: &mut interface,
					scripts: Some(&mut scripts),
					simulation: simulation.as_mut(),
					// a machine on a wire with nothing on screen has nothing
					// to make a noise into. @ref the module comment.
					audio: None,
					// nothing tells a host what its world looks like, so at
					// that end the endpoint goes nowhere near the step - and
					// `Net::arrive` refuses a host anyway, so this is which
					// answer is given rather than whether one is. At the other
					// end it is the whole point.
					wire: (!hosting).then_some(step::Wired { net: &mut net, now: moment }),
				},
				&mut input,
				time,
				false,
			);

			// and out once a step rather than once a pass, because a message
			// has to mean "this is where things stand at this moment".
			//
			// The world itself goes with one message in every
			// [`EVERY`](colby_net::EVERY): a step is what makes a stack of
			// boxes stand up and a snapshot is what a far end can be told, and
			// they are not the same rate. The messages in between still go,
			// carrying what this end holds and whatever the reliable ring is
			// still owed.
			// @note: `World::steps` is public and a game module writes it, so
			// the cadence is on a clock the game can move. Nothing does today;
			// it is worth knowing before something does, because a game that
			// set it back would stop describing the world for a while and
			// nothing anywhere would say so.
			let describing = hosting && world.steps.is_multiple_of(u64::from(EVERY));

			if describing {
				crate::net::records(&world.bodies, &mut records);
			}

			// a host asks nobody for anything and says so; a client asks for
			// everything it has not been answered about, which is the window
			// rather than the newest of them. @ref `Net::ask`.
			if hosting {
				net.ask(&[]);
			} else {
				net.ask(world.commands.unsettled(world.peer));
			}

			net.send(started.elapsed(), describing.then_some(records.as_slice()));
			// and the wire's own clock moves on by exactly one step, after the
			// message that was about the step just run. @ref `crate::app`,
			// which advances it in the same place.
			moment = moment.saturating_add(STEP);
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
	fn an_address_is_read_after_the_other_flag_and_attached_to_it() {
		let both = address(&["--join".to_owned(), "127.0.0.1:9999".to_owned()]);
		assert_eq!(both.map(|found| found.port()), Some(9999));
		assert_eq!(
			address(&["--join=127.0.0.1:1234".to_owned()]).map(|found| found.port()),
			Some(1234)
		);
		// **an address after the other flag, not a port.** A payload nothing
		// could read as an address would let this pass whether the flag is
		// looked at or not, which is a test that cannot fail for the reason it
		// is about.
		assert_eq!(
			address(&["--connect".to_owned(), "127.0.0.1:9999".to_owned()]),
			None,
			"and not somebody else's"
		);
		assert_eq!(address(&[]), None);
	}

	#[test]
	fn a_flag_with_nothing_usable_after_it_asks_for_nothing() {
		assert_eq!(address(&["--join".to_owned()]), None, "a flag on its own is not an address");
		assert_eq!(
			address(&["--join".to_owned(), "not-an-address".to_owned()]),
			None,
			"and neither is a word"
		);
		assert_eq!(
			address(&["--join".to_owned(), "127.0.0.1".to_owned()]),
			None,
			"a host with no port is not one either, because a wire needs both"
		);
	}

	#[test]
	fn the_two_flags_do_not_read_each_other() {
		let line = ["--join".to_owned(), "127.0.0.1:9999".to_owned()];
		assert_eq!(parse(&line), None, "asking to join is not asking to host");

		let line = ["--connect".to_owned(), "127.0.0.1:27015".to_owned()];
		assert_eq!(address(&line), None, "and opening a window on one is not either");
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
