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
	path::Path,
	thread,
	time::{Duration, Instant},
};

use colby_core::{
	Result,
	abi::{Input, World, console},
	glam::Vec2,
	info,
	time::Clock,
};
use colby_net::Slot;
use colby_physics::Simulation;
use colby_script::Vm;
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
/// @param arguments - the command line, without the program's own name
/// @return which port to listen on, if a host was asked for
#[must_use]
pub(crate) fn requested(arguments: &[String]) -> Option<u16> { port_after(arguments, FLAG) }

/// Reads the command line for a client with no window.
///
/// Accepts `--join address` and `--join=address`, where an address is whatever
/// the standard library reads as one - the same shapes `--connect` takes,
/// because it is the same question asked by a mode that draws nothing.
///
/// @param arguments - the command line, without the program's own name
/// @return where the host is, if a windowless client was asked for
#[must_use]
pub(crate) fn joining(arguments: &[String]) -> Option<SocketAddr> { address(arguments) }

/// The same, over arguments already collected.
///
/// The reading itself is `crate::net`'s, because `--connect` asks the identical
/// question and an address is an address. @ref `crate::net::after`.
fn address(arguments: &[String]) -> Option<SocketAddr> { crate::net::after(arguments, JOIN) }

/// The same, over arguments already collected.
/// The port after a flag, whichever flag is doing the asking.
///
/// **Written once and called twice**, the same way an address is: two modes
/// want a port off the command line - a windowless end and a window that
/// serves - and a second copy of these rules would differ the first time
/// somebody fixed one of them. @ref `crate::net::serving`.
///
/// @param arguments - the command line, without the program's own name
/// @param flag - which flag to look for
/// @return the port after it, or the default when it stands alone
pub(crate) fn port_after(arguments: &[String], flag: &str) -> Option<u16> {
	for (index, argument) in arguments.iter().enumerate() {
		let port = if let Some(rest) = argument.strip_prefix(&format!("{flag}=")) {
			rest.parse::<u16>().ok()
		} else if argument == flag {
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
/// @param workspace - the checkout the runner was built from
pub(crate) fn serve(port: u16, workspace: &Path) -> Result { run(End::Serving(port), workspace) }

/// The same, at the other end: a client with nothing on screen.
///
/// @param address - where the host is
/// @param workspace - the checkout the runner was built from
pub(crate) fn join(address: SocketAddr, workspace: &Path) -> Result {
	run(End::Asking(address), workspace)
}

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
/// @param workspace - the checkout the runner was built from
fn run(end: End, workspace: &Path) -> Result {
	// boxed and installed before anything else touches the world, for the
	// reason the window and the screenshot box it: the world keeps this
	// address.
	let mut simulation = Box::new(Simulation::new());
	let mut world = Box::<World>::default();
	world.install_physics(simulation.table());

	let mut assets = Assets::new(workspace);
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
	let console = Console::open(&mut world, workspace);
	let mut scripts = Vm::new(console::defer)?;
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
		crate::console::serve(&mut world, workspace);
		// the tick rate, read where a window reads it. A serving end is the
		// authority and runs what it was told; an asking one takes the rate
		// every host runs, because nothing on the wire says what this host
		// chose. @ref `crate::app::paced`, which is where both halves of that
		// are checked.
		//
		// @note: no test reaches this line, and a mutation that made a serving
		// end follow instead of lead is not caught by anything. The whole of
		// this loop is like that - a socket, a module and a console stand
		// between it and any harness - and that is `NET-7` rather than
		// something this line can fix on its own.
		clock.set_rate(crate::app::paced(&world.cvars, !hosting));
		crate::saves::serve(&mut world, &mut simulation, workspace);
		crate::net::serve(&mut world, Some(&mut net));
		// everything the wire wants once a frame, in the one place it is
		// written. @ref `crate::net::hear`, which a window calls too.
		crate::net::hear(&mut net, &mut world, started.elapsed());

		let mut ran = false;
		// once for the pass, so two steps of one iteration are the same length
		// as each other whatever a console said between them.
		let rate = clock.rate();

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
				rate,
				time,
				false,
			);

			// everything the wire wants once a step, in the one place it is
			// written. @ref `crate::net::tell`, which a window calls too.
			crate::net::tell(&mut net, &world, &mut records, rate.hz(), started.elapsed());
			moment = moment.saturating_add(rate.step());
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
const _: () = assert!(
	colby_core::time::Rate::DEFAULT.step().as_nanos() > 0,
	"a step has to take some time"
);

#[cfg(test)]
mod tests {
	use super::*;

	/// The port `--host` asks for, over arguments already collected.
	fn parse(arguments: &[String]) -> Option<u16> { port_after(arguments, FLAG) }

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
