//! Two endpoints in one process, over a wire that lies, with one number at the
//! end of it.
//!
//! `colby --link` is the third of the family, after `--shot` and `--record`,
//! and it exists for the same reason both of those do: a change to something
//! nobody can look at is a change nobody can review. A screenshot is how a
//! renderer change is seen from the other end of a shell; a recording is how a
//! mixer change is; this is how a networking change is.
//!
//! **It is deliberately before anything is sent.** The plan puts this in the
//! first pass rather than the last, because everything after it - a snapshot, a
//! delta, a ring per client, a command that crosses - is a change whose only
//! honest question is "what did the far end end up with", and that question
//! needs a run somebody can repeat.
//!
//! No window, no socket, no operating system and no clock. Two endpoints, a
//! wire between them that loses and delays and duplicates on a seed, a fixed
//! number of steps, and a hash of everything that got through. Same build, same
//! number, on any machine.
//!
//! The conditions are fixed here rather than read from the console, exactly as
//! a screenshot's ninety steps are fixed: a tool whose answer depends on what
//! somebody last typed is not a tool.

use std::{
	cell::RefCell,
	net::{IpAddr, Ipv4Addr, SocketAddr},
	rc::Rc,
	time::Duration,
};

use colby_core::{Result, info, time::STEP, warn};
use colby_net::Conditions;

use crate::net::{Loopback, Net, Tally, Wire};

/// The flag that asks for a two-endpoint run.
const FLAG: &str = "--link";

/// How many steps to run when nobody says.
///
/// Ten seconds at the fixed step, which is long enough for a burst of losses to
/// happen several times over and for the round-trip estimate to settle.
const DEFAULT_STEPS: u32 = 600;

/// The most steps one run may be.
const MAX_STEPS: u32 = 100_000;

/// How often the host says something that has to arrive.
const HOST_EVERY: u32 = 60;

/// How often the client does.
///
/// A different number from the host's on purpose, so the two are not in step
/// with each other and a run covers both orders of arrival.
const CLIENT_EVERY: u32 = 90;

/// How many steps of quiet the run ends with.
///
/// Two seconds in which nothing new is said and everything already in flight
/// gets there. Without it the last command queued would be counted as owed and
/// have no chance at all to arrive - which is a wrong answer about the one
/// thing this mode exists to check.
const SETTLE: u32 = 120;

/// How bad the wire is.
///
/// Lively rather than plausible: a tenth of everything lost, in runs three
/// times longer than chance alone would make them, on a link that also delays,
/// shakes and occasionally says a thing twice. A gentler wire would be a run in
/// which nothing interesting ever happened.
const WIRE: Conditions = Conditions {
	lag: Duration::from_millis(30),
	jitter: Duration::from_millis(10),
	loss: 0.1,
	burst: 3.0,
	duplicate: 0.02,
};

/// What the two endpoints answer to.
///
/// Nothing binds these and nothing has to be able to reach them: the wire
/// between the two is a pair of inboxes and an address is only how they tell
/// each other apart.
const HOST_AT: u16 = 27015;

/// The other one.
const CLIENT_AT: u16 = 40000;

/// Reads the command line for a two-endpoint run.
///
/// Accepts `--link` on its own, `--link steps` and `--link=steps`.
///
/// @return how many steps to run, if a run was asked for
#[must_use]
pub(crate) fn requested() -> Option<u32> {
	let arguments: Vec<String> = std::env::args().skip(1).collect();

	parse(&arguments)
}

/// The same, over arguments already collected.
///
/// Split out so that it can be tested, which the environment cannot be.
fn parse(arguments: &[String]) -> Option<u32> {
	for (index, argument) in arguments.iter().enumerate() {
		let steps = if let Some(rest) = argument.strip_prefix(&format!("{FLAG}=")) {
			rest.parse::<u32>().ok()
		} else if argument == FLAG {
			arguments
				.get(index + 1)
				.and_then(|next| next.parse::<u32>().ok())
		} else {
			continue;
		};

		return Some(steps.unwrap_or(DEFAULT_STEPS).clamp(1, MAX_STEPS));
	}

	None
}

/// What one run came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Outcome {
	/// Every command that arrived anywhere, folded into one number.
	pub(crate) digest: u64,

	/// How many commands each side was given to deliver.
	pub(crate) said: u32,

	/// How many arrived.
	pub(crate) heard: u32,

	/// Messages the host put on the wire, and whole ones it took off.
	pub(crate) host: (u32, u32, u32),

	/// The same for the client.
	pub(crate) client: (u32, u32, u32),

	/// What each side's channel made of it: sent, acknowledged, lost.
	pub(crate) tally: (Tally, Tally),

	/// How many datagrams were addressed to nobody at all.
	pub(crate) nowhere: u32,

	/// The round trip each side settled on, in whole microseconds.
	pub(crate) rtt: (u128, u128),
}

/// Runs two endpoints against each other and reports what happened.
///
/// @param steps - how many simulation steps to run for
pub(crate) fn run(steps: u32) -> Result {
	let outcome = exchange(steps, WIRE);

	info!(
		steps,
		lag = ?WIRE.lag,
		jitter = ?WIRE.jitter,
		loss = WIRE.loss,
		burst = WIRE.burst,
		duplicate = WIRE.duplicate,
		"a wire that lies"
	);
	info!(
		sent = outcome.host.0,
		delivered = outcome.host.1,
		ignored = outcome.host.2,
		acknowledged = outcome.tally.0.1,
		lost = outcome.tally.0.2,
		rtt_us = outcome.rtt.0,
		"the host"
	);
	info!(
		sent = outcome.client.0,
		delivered = outcome.client.1,
		ignored = outcome.client.2,
		acknowledged = outcome.tally.1.1,
		lost = outcome.tally.1.2,
		rtt_us = outcome.rtt.1,
		"the client"
	);
	info!(
		said = outcome.said,
		heard = outcome.heard,
		nowhere = outcome.nowhere,
		digest = format!("{:016x}", outcome.digest),
		"everything that had to arrive, arrived"
	);

	if outcome.said != outcome.heard {
		warn!(
			said = outcome.said,
			heard = outcome.heard,
			"something that had to arrive did not, which is the one thing this cannot do"
		);
	}

	Ok(())
}

/// The run itself, with no logging in it.
///
/// Split from the reporting so that a test can drive it, which is the same
/// split every other mode in this crate has.
///
/// @param steps - how many simulation steps
/// @param wire - how bad the link in front of each endpoint is
pub(crate) fn exchange(steps: u32, wire: Conditions) -> Outcome {
	let shared = Rc::new(RefCell::new(Wire::default()));
	let (one, two) = (somewhere(HOST_AT), somewhere(CLIENT_AT));
	let mut host = Net::over(Box::new(Loopback::at(one, &shared)), true, 1);
	let mut client = Net::over(Box::new(Loopback::at(two, &shared)), false, 2);

	// the client knows who it came for; the host learns who turned up.
	client.set(wire);
	host.set(wire);

	let mut digest = 0xCBF2_9CE4_8422_2325_u64;
	let mut said = 0_u32;
	let mut heard = 0_u32;

	// a client that has never sent anything is a client the host has never
	// heard of, so the first thing that happens is the client saying hello.
	connect(&mut client, one);

	for step in 1..=steps.saturating_add(SETTLE) {
		let now = STEP * step;
		let talking = step <= steps;

		if talking && step.is_multiple_of(HOST_EVERY) {
			said += tell(&mut host, &format!("host.tick {step}"));
		}

		if talking && step.is_multiple_of(CLIENT_EVERY) {
			said += tell(&mut client, &format!("client.tick {step}"));
		}

		host.send(now);
		client.send(now);
		host.receive(now);
		client.receive(now);

		for (side, net) in [(0_u8, &host), (1, &client)] {
			for arrived in net.said() {
				heard += 1;
				digest = fold(digest, step, side, &arrived.text);
			}
		}
	}

	Outcome {
		digest,
		said,
		heard,
		host: (host.sent(), host.delivered(), host.ignored()),
		client: (client.sent(), client.delivered(), client.ignored()),
		tally: (host.tally(0), client.tally(0)),
		nowhere: shared.borrow().nowhere(),
		rtt: (host.rtt(0).as_micros(), client.rtt(0).as_micros()),
	}
}

/// Points a client at a host it has not met.
fn connect(client: &mut Net, host: SocketAddr) {
	// `Net::connect` binds a socket, which is exactly what this mode has none
	// of - so the peer is added the way that call would have added it.
	client.introduce(host);
}

/// Queues a line on one endpoint, and says whether it took it.
fn tell(net: &mut Net, text: &str) -> u32 {
	let peers = u32::try_from(net.peers()).unwrap_or(0);

	if let Err(error) = net.say(text) {
		warn!(%text, %error, "the ring would not take a command");

		return 0;
	}

	peers
}

/// An address on the machine this is running on.
fn somewhere(port: u16) -> SocketAddr { SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port) }

/// One command folded into the number that stands for a whole run.
fn fold(hash: u64, step: u32, side: u8, text: &str) -> u64 {
	let mut folded = hash;

	for byte in step
		.to_le_bytes()
		.iter()
		.chain(std::iter::once(&side))
		.chain(text.as_bytes())
	{
		folded ^= u64::from(*byte);
		folded = folded.wrapping_mul(0x0000_0100_0000_01B3);
	}

	folded
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_flag_is_read_on_its_own_and_with_a_count() {
		assert_eq!(parse(&["--link".to_owned()]), Some(DEFAULT_STEPS));
		assert_eq!(parse(&["--link".to_owned(), "120".to_owned()]), Some(120));
		assert_eq!(parse(&["--link=120".to_owned()]), Some(120));
		assert_eq!(parse(&["--shot".to_owned()]), None, "and not somebody else's flag");
		assert_eq!(parse(&[]), None);
	}

	#[test]
	fn a_count_of_nil_or_a_silly_one_is_clamped_rather_than_taken() {
		assert_eq!(parse(&["--link=0".to_owned()]), Some(1));
		assert_eq!(parse(&["--link=99999999".to_owned()]), Some(MAX_STEPS));
	}

	#[test]
	fn a_flag_that_is_not_a_count_after_the_word_is_not_eaten() {
		assert_eq!(
			parse(&["--link".to_owned(), "--shot".to_owned()]),
			Some(DEFAULT_STEPS),
			"the next word is only a count when it is one"
		);
	}

	#[test]
	fn everything_that_had_to_arrive_arrives_over_a_wire_that_loses_a_tenth() {
		// the one thing this whole subsystem promises: a command that was
		// queued gets there, whatever the wire does to the datagrams carrying
		// it. Nine hundred steps is fifteen commands over a link losing a tenth
		// of everything in runs of three.
		let outcome = exchange(900, WIRE);

		assert!(outcome.said > 10, "the run has to have said something: {}", outcome.said);
		assert_eq!(outcome.heard, outcome.said, "and every one of them arrived");
	}

	#[test]
	fn a_run_is_the_same_run_twice() {
		// what the whole mode is for. Nothing in it reads a clock, an
		// environment or a hashed container, so this holds on another machine
		// as well - and the number below is what says so out loud.
		let one = exchange(600, WIRE);
		let two = exchange(600, WIRE);

		assert_eq!(one, two);
	}

	#[test]
	fn a_run_is_the_run_it_has_always_been() {
		// the number this whole mode exists to produce, written down. Two runs
		// of one build agreeing says only that the build agrees with itself;
		// what the claim is about is another machine and another commit, and
		// the only thing that pins that is a literal.
		//
		// It moves when anything about what crosses moves - the draw order, the
		// conditions, the schedule, the format of a block. That is the point:
		// every one of those invalidates a recorded run, and this is what says
		// so out loud rather than leaving it to be noticed.
		assert_eq!(exchange(600, WIRE).digest, 0xFEA2_9F7A_1B23_1D5A);
	}

	#[test]
	fn a_command_the_ring_refused_is_not_counted_as_one_that_was_said() {
		// long enough that both rings really fill: each side says something
		// every sixty or ninety steps and a ring holds sixty-four, so past
		// about four thousand steps of total silence each starts refusing. A
		// refused command must not be counted, or the one number this mode
		// promises - what was said got there - would be counted against a
		// larger number than was ever really said.
		let dead = Conditions { loss: 1.0, ..WIRE };
		let outcome = exchange(6500, dead);

		assert_eq!(outcome.said, 128, "sixty-four each way, and the rest refused");
		assert_eq!(outcome.heard, 0, "and over a wire like that, none of them arrived");
	}

	#[test]
	fn a_wire_that_loses_everything_reports_nothing_as_having_arrived() {
		// the one thing the count in this mode may not do is report as
		// delivered something that never got there, and a wire that loses all
		// of it is where that would show.
		let dead = Conditions { loss: 1.0, ..WIRE };
		let outcome = exchange(600, dead);

		assert!(outcome.said > 0, "the client still queued some");
		assert_eq!(outcome.heard, 0, "and not one of them got there");
		assert_eq!(outcome.host.1, 0, "nothing was delivered in either direction");
		assert_eq!(outcome.client.1, 0);
	}

	#[test]
	fn a_run_over_a_perfect_wire_loses_nothing_and_is_a_different_run() {
		let lying = exchange(600, WIRE);
		let clean = exchange(600, Conditions::PERFECT);

		assert_eq!(clean.heard, clean.said);
		assert_eq!(clean.tally.0.2, 0, "a perfect wire loses no messages either");
		assert!(lying.tally.0.2 > 0, "and a lying one does: {:?}", lying.tally.0);
		assert_ne!(lying.digest, clean.digest, "so the two runs are not the same run");
	}
}
