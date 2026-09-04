//! Being on a wire with nothing on screen, at either end of it.
//!
//! `colby --host` and `colby --join` are *runs*, not builds. The same
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
//! It shares the runtime with the window path and differs from it in one way
//! that does not matter: there is no vertical blank to pace against, so a pass
//! that ran no step sleeps for a moment rather than spinning a core at
//! nothing.
//!
//! **No output device either.** A machine serving a world has no reason to make
//! a noise, and one that did would be making it into somebody's empty room.
//! Everything else about the step is what a window would have run.

use std::{path::Path, thread, time::Duration};

use colby_core::{Result, abi::Input, info, time::Clock};

use crate::{Front, Runtime};

/// How long a pass that ran no step waits before looking again.
///
/// Short enough that a step is never late by anything a person could measure,
/// and long enough that an idle host is not a core at full tilt.
const IDLE: Duration = Duration::from_millis(1);

/// The loop both ends share.
///
/// **One body and not two**, which is the whole reason this is worth writing
/// down: a host and a client differ in three lines - which way the socket is
/// opened, whether this process stops being the authority, and whether what
/// goes out is a world or a request - and every other thing they do is the
/// same thing. Two loops would drift, and the drift would be exactly the sort
/// of difference nobody notices until two machines disagree about where
/// somebody is standing. The first two of those three are the runtime's, so
/// what is left here is the pace.
///
/// @param front - which end of the wire this process is: [`Front::Host`] or
/// [`Front::Join`], and nothing else opens no window
/// @param workspace - the checkout the runner was built from
pub(crate) fn run(front: Front, workspace: &Path) -> Result {
	let mut runtime = Runtime::open(front, workspace)?;
	let following = runtime.following();
	let mut input = Input::default();
	let mut clock = Clock::new();

	// where the wire's own clock stands for the step about to run. A client
	// draws the world a delay behind what it was told, and the delay is
	// measured against this rather than against the wall, so that a pass which
	// runs four catch-up steps places four moments rather than one. @ref
	// `crate::app`, which keeps the same number for the same reason.
	let mut moment = Duration::ZERO;

	if let Some(net) = runtime.net.as_ref() {
		if following {
			info!(address = %net.address(), "asking");
		} else {
			info!(address = %net.address(), "serving");
		}
	}

	while !runtime.world.quit {
		clock.tick();
		runtime.poll();

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
		clock.set_rate(crate::app::paced(&runtime.world.cvars, following));

		let mut ran = false;
		// once for the pass, so two steps of one iteration are the same length
		// as each other whatever a console said between them.
		let rate = clock.rate();

		while let Some(time) = clock.step() {
			// never editing: a windowless end has no key to press for it, and
			// a world nobody is looking at is a world being played.
			moment = runtime.step(&mut input, rate, time, false, moment);
			ran = true;
		}

		if !ran {
			thread::sleep(IDLE);
		}
	}

	runtime.close();

	if let Some(net) = runtime.net.as_ref() {
		info!(
			steps = runtime.world.steps,
			peers = net.peers(),
			sent = net.sent(),
			delivered = net.delivered(),
			seconds = runtime.now().as_secs_f32(),
			"colby stopped serving"
		);
	}

	Ok(())
}

/// How long one step is, for whoever is reading the loop above.
const _: () = assert!(
	colby_core::time::Rate::DEFAULT.step().as_nanos() > 0,
	"a step has to take some time"
);
