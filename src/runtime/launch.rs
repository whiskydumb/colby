//! What the command line asked for, read once and in one place.
//!
//! Six runs, eight flags, one pass. Every flag's shape is written down here
//! and nowhere else: `--flag`, `--flag value` and `--flag=value`, with a value
//! taken from the next word only when it is one - a port that parses, a count
//! that parses, a path that does not start with a dash - so that `--host
//! --shot` is a host and a picture rather than a host on a port called
//! `--shot`. An address is taken whatever the next word is, short of another
//! flag, because a word that is not an address is refused by name a moment
//! later and the two flags that want one have nothing else to do with it.
//!
//! **When several runs are named, one is taken and the log says which.** A
//! person who typed `--shot` and `--record` meant one of them, and refusing to
//! start would be worse than picking. The order is the one the runs are
//! dispatched in: a two-endpoint run before everything, because it loads no
//! module; then a picture, a sound, a host, a windowless client; and a window
//! when nothing else was asked for - which itself serves when told to, talks
//! when told to, and is on its own otherwise. Serving wins over talking for
//! the same reason.

use std::{
	net::{SocketAddr, ToSocketAddrs},
	path::PathBuf,
};

use colby_core::warn;

use crate::{
	link,
	net::{DEFAULT_PORT, Standing},
	record, shot,
};

/// `--link [steps]`: two endpoints in one process, and a hash at the end.
const LINK: &str = "--link";

/// `--shot [path]`: one frame to a file.
const SHOT: &str = "--shot";

/// `--record [path [steps]]`: what a run sounds like, to a file.
const RECORD: &str = "--record";

/// `--host [port]`: a windowless authority.
const HOST: &str = "--host";

/// `--join <address>`: a windowless client.
const JOIN: &str = "--join";

/// `--listen [port]`: a window that serves as well as playing.
///
/// A flag of its own rather than a window-shaped `--host`, because the two are
/// genuinely different runs and one of them has to open no window at all: a
/// dedicated end is what a machine nobody is at runs, and what everything
/// checking this engine drives.
const LISTEN: &str = "--listen";

/// `--connect <address>`: a window that talks to a host.
const CONNECT: &str = "--connect";

/// `--project <dir>`: the project to run, whichever run it is.
///
/// Not a run of its own: a picture, a sound, a host and a window all run
/// *some* project, and without this flag it is the one in the working
/// directory. @ref `crate::run`, which looks there.
const PROJECT: &str = "--project";

/// What the command line asked for: a run, and the project to run it in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
	/// Which kind of run.
	pub run: Run,

	/// The project directory `--project` named, if it did.
	pub project: Option<PathBuf>,
}

impl Launch {
	/// Reads the command line.
	///
	/// @param arguments - the command line, without the program's own name
	/// @return what was asked for; a window on its own, in the project of the
	/// working directory, when nothing was
	#[must_use]
	pub fn parse(arguments: &[String]) -> Self {
		let flags = Flags::read(arguments);

		Self {
			project: flags.project.clone(),
			run: flags.decide(),
		}
	}
}

/// What kind of run was asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Run {
	/// Two endpoints against each other over a wire that lies, for this many
	/// steps.
	Link(u32),

	/// One frame, to this file.
	Shot(PathBuf),

	/// A run's sound, to this file, for this many steps.
	Record {
		/// Where to write the file.
		path: PathBuf,

		/// How many simulation steps to run.
		steps: u32,
	},

	/// A windowless authority on this port.
	Host(u16),

	/// A windowless client of the host at this address.
	Join(SocketAddr),

	/// A window, standing where it was told to on a wire.
	Window(Standing),
}

/// Every flag that was named, before one run is chosen among them.
#[derive(Debug, Default, PartialEq, Eq)]
struct Flags {
	link: Option<u32>,
	shot: Option<PathBuf>,
	record: Option<(PathBuf, u32)>,
	host: Option<u16>,
	join: Option<SocketAddr>,
	listen: Option<u16>,
	connect: Option<SocketAddr>,
	project: Option<PathBuf>,
}

impl Flags {
	/// One pass over the words.
	fn read(arguments: &[String]) -> Self {
		let mut flags = Self::default();
		let mut words = Words { arguments, at: 0 };

		while let Some((flag, inline)) = words.flag() {
			match flag {
				| LINK =>
					flags.link = Some(
						words
							.count(inline, link::DEFAULT_STEPS)
							.clamp(1, link::MAX_STEPS),
					),
				| SHOT =>
					flags.shot = Some(
						words
							.path(inline)
							.unwrap_or_else(|| PathBuf::from(shot::DEFAULT_PATH)),
					),
				| RECORD => flags.record = Some(words.recording(inline)),
				| HOST => flags.host = Some(words.port(inline)),
				| JOIN => flags.join = words.address(inline, JOIN),
				| LISTEN => flags.listen = Some(words.port(inline)),
				| CONNECT => flags.connect = words.address(inline, CONNECT),
				| PROJECT =>
					flags.project = words.path(inline).or_else(|| {
						warn!("{PROJECT} needs a directory after it");

						None
					}),
				| other => warn!(word = other, "not a flag this engine knows; left alone"),
			}
		}

		flags
	}

	/// The one run, out of everything that was named.
	fn decide(self) -> Run {
		let Self {
			link,
			shot,
			record,
			host,
			join,
			listen,
			connect,
			..
		} = self;
		let named =
			[link.is_some(), shot.is_some(), record.is_some(), host.is_some(), join.is_some()]
				.into_iter()
				.filter(|named| *named)
				.count();

		// the precedence, as a table: the first run named in dispatch order.
		let chosen = match (link, shot, record, host, join, listen, connect) {
			| (Some(steps), ..) => Run::Link(steps),
			| (None, Some(path), ..) => Run::Shot(path),
			| (None, None, Some((path, steps)), ..) => Run::Record { path, steps },
			| (None, None, None, Some(port), ..) => Run::Host(port),
			| (None, None, None, None, Some(address), ..) => Run::Join(address),
			| (None, None, None, None, None, Some(port), _) =>
				Run::Window(Standing::Serving(port)),
			| (None, None, None, None, None, None, Some(address)) =>
				Run::Window(Standing::Talking(address)),
			| (None, None, None, None, None, None, None) => Run::Window(Standing::Alone),
		};

		if named > 1 {
			warn!(?chosen, "several runs were asked for; taking the first in dispatch order");
		}

		chosen
	}
}

/// A cursor over the words, with every reading a value after a flag can have.
struct Words<'a> {
	arguments: &'a [String],
	at: usize,
}

impl<'a> Words<'a> {
	/// The next word as a flag, and its value when it was written
	/// `--flag=value`.
	fn flag(&mut self) -> Option<(&'a str, Option<&'a str>)> {
		let word = self.arguments.get(self.at)?;
		self.at += 1;

		Some(match word.split_once('=') {
			| Some((flag, value)) if flag.starts_with("--") => (flag, Some(value)),
			| _ => (word.as_str(), None),
		})
	}

	/// The word after the flag, if there is one.
	fn peek(&self) -> Option<&'a str> { self.arguments.get(self.at).map(String::as_str) }

	/// A count: inline, or the next word when it parses as one, or the default.
	///
	/// The next word is only a count when it is one: `--link --shot` is a
	/// two-endpoint run of the usual length and a picture, not a run of
	/// `--shot` steps.
	fn count(&mut self, inline: Option<&str>, default: u32) -> u32 {
		if let Some(inline) = inline {
			return inline.parse().unwrap_or(default);
		}

		match self.peek().and_then(|word| word.parse().ok()) {
			| Some(count) => {
				self.at += 1;

				count
			},
			| None => default,
		}
	}

	/// A port: as a count, with the usual port as the default.
	fn port(&mut self, inline: Option<&str>) -> u16 {
		if let Some(inline) = inline {
			return inline.parse().unwrap_or(DEFAULT_PORT);
		}

		match self.peek().and_then(|word| word.parse().ok()) {
			| Some(port) => {
				self.at += 1;

				port
			},
			| None => DEFAULT_PORT,
		}
	}

	/// A path: inline, or the next word when it does not look like a flag.
	fn path(&mut self, inline: Option<&str>) -> Option<PathBuf> {
		if let Some(inline) = inline {
			return Some(PathBuf::from(inline));
		}

		let next = self
			.peek()
			.filter(|word| !word.starts_with('-'))?;
		self.at += 1;

		Some(PathBuf::from(next))
	}

	/// A recording: a path and then a count, both optional and in that order.
	///
	/// The count is only read after a path: `--record 300` is a recording into
	/// a file called `300`, because a word after the flag is a path first.
	fn recording(&mut self, inline: Option<&str>) -> (PathBuf, u32) {
		let Some(path) = self.path(inline) else {
			return (PathBuf::from(record::DEFAULT_PATH), record::DEFAULT_STEPS);
		};

		let steps = self
			.count(None, record::DEFAULT_STEPS)
			.clamp(1, record::MAX_STEPS);

		(path, steps)
	}

	/// An address: inline, or the next word short of another flag, read the
	/// way the standard library reads one.
	///
	/// A word that is not an address is refused by name rather than left for
	/// somebody else, because nothing else on this command line wants a word.
	///
	/// @param inline - the value after `=`, if the flag had one
	/// @param flag - which flag is asking, for the message
	fn address(&mut self, inline: Option<&str>, flag: &str) -> Option<SocketAddr> {
		let text = match inline {
			| Some(inline) => inline.to_owned(),
			| None => match self.peek().filter(|word| !word.starts_with("--")) {
				| Some(word) => {
					self.at += 1;

					word.to_owned()
				},
				| None => {
					warn!("{flag} needs an address to connect to");

					return None;
				},
			},
		};

		match text.to_socket_addrs() {
			| Ok(mut found) => found.next().or_else(|| {
				warn!(%text, "that address is nowhere");

				None
			}),
			| Err(error) => {
				warn!(%text, %error, "that is not an address");

				None
			},
		}
	}
}

#[cfg(test)]
mod tests {
	use std::net::{IpAddr, Ipv4Addr};

	use super::*;

	/// The command line, as words.
	fn words(line: &[&str]) -> Vec<String> {
		line.iter()
			.map(|word| (*word).to_owned())
			.collect()
	}

	/// An address nobody has to be able to reach.
	fn somewhere(port: u16) -> SocketAddr {
		SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
	}

	/// What the command line came to, as a run.
	fn parse(line: &[&str]) -> Run { Launch::parse(&words(line)).run }

	/// Every flag on it, before one run is chosen.
	fn read(line: &[&str]) -> Flags { Flags::read(&words(line)) }

	/// A recording of this path and this many steps.
	fn recording(path: &str, steps: u32) -> Run {
		Run::Record { path: PathBuf::from(path), steps }
	}

	#[test]
	fn nothing_at_all_is_a_window_on_its_own() {
		// the ordinary case, and the one a change that opened a socket by
		// accident would break in silence.
		assert_eq!(parse(&[]), Run::Window(Standing::Alone));
	}

	#[test]
	fn a_two_endpoint_run_is_read_on_its_own_and_with_a_count() {
		assert_eq!(parse(&["--link"]), Run::Link(link::DEFAULT_STEPS));
		assert_eq!(parse(&["--link", "120"]), Run::Link(120));
		assert_eq!(parse(&["--link=120"]), Run::Link(120));
		assert_eq!(parse(&["--link=0"]), Run::Link(1), "a count of nil is clamped");
		assert_eq!(parse(&["--link=99999999"]), Run::Link(link::MAX_STEPS), "and a silly one");
		assert_eq!(read(&["--shot"]).link, None, "and not somebody else's flag");
	}

	#[test]
	fn a_flag_that_is_not_a_count_after_the_word_is_not_eaten() {
		let flags = read(&["--link", "--shot"]);

		assert_eq!(
			flags.link,
			Some(link::DEFAULT_STEPS),
			"the next word is only a count when it is one"
		);
		assert_eq!(
			flags.shot,
			Some(PathBuf::from(shot::DEFAULT_PATH)),
			"so the flag after it is read"
		);
	}

	#[test]
	fn a_picture_is_read_on_its_own_and_with_a_path() {
		assert_eq!(parse(&["--shot"]), Run::Shot(PathBuf::from(shot::DEFAULT_PATH)));
		assert_eq!(parse(&["--shot", "out.png"]), Run::Shot(PathBuf::from("out.png")));
		assert_eq!(parse(&["--shot=out.png"]), Run::Shot(PathBuf::from("out.png")));

		let flags = read(&["--shot", "--record"]);

		assert_eq!(flags.shot, Some(PathBuf::from(shot::DEFAULT_PATH)), "a flag is not a path");
		assert!(flags.record.is_some(), "and is read as itself");
	}

	#[test]
	fn a_recording_is_read_on_its_own_and_with_a_path() {
		assert_eq!(
			parse(&["--record"]),
			recording(record::DEFAULT_PATH, record::DEFAULT_STEPS),
			"the flag on its own writes where it says it will"
		);

		for line in [&["--record", "out.wav"][..], &["--record=out.wav"][..]] {
			assert_eq!(parse(line), recording("out.wav", record::DEFAULT_STEPS), "{line:?}");
		}
	}

	#[test]
	fn a_number_after_a_recordings_path_is_a_step_count() {
		for line in [&["--record", "out.wav", "300"][..], &["--record=out.wav", "300"][..]] {
			assert_eq!(parse(line), recording("out.wav", 300), "{line:?}");
		}

		assert_eq!(
			parse(&["--record", "out.wav", "--other"]),
			recording("out.wav", record::DEFAULT_STEPS),
			"a word that is not a number leaves the default alone"
		);
		assert_eq!(
			parse(&["--record", "out.wav", "9999999"]),
			recording("out.wav", record::MAX_STEPS),
			"the encoder would refuse the whole thing after doing all the work"
		);
		assert_eq!(parse(&["--record", "out.wav", "0"]), recording("out.wav", 1));
	}

	#[test]
	fn a_flag_where_a_recordings_path_would_be_is_not_a_path() {
		// `--record --shot out.png` asks for a recording with no path and a
		// picture, not for a recording called `--shot`.
		let flags = read(&["--record", "--shot", "out.png"]);

		assert_eq!(
			flags.record,
			Some((PathBuf::from(record::DEFAULT_PATH), record::DEFAULT_STEPS))
		);
		assert_eq!(flags.shot, Some(PathBuf::from("out.png")));
	}

	#[test]
	fn a_host_is_read_on_its_own_and_with_a_port() {
		assert_eq!(parse(&["--host"]), Run::Host(DEFAULT_PORT));
		assert_eq!(parse(&["--host", "9999"]), Run::Host(9999));
		assert_eq!(parse(&["--host=9999"]), Run::Host(9999));
		assert_eq!(read(&["--connect"]).host, None, "and not somebody else's");
	}

	#[test]
	fn a_word_after_a_host_that_is_not_a_port_is_not_eaten() {
		let flags = read(&["--host", "--shot"]);

		assert_eq!(flags.host, Some(DEFAULT_PORT), "the next word is only a port when it is one");
		assert!(flags.shot.is_some(), "so the flag after it is read");
		assert_eq!(
			parse(&["--host", "99999"]),
			Run::Host(DEFAULT_PORT),
			"and a number that is not a port is not one"
		);
	}

	#[test]
	fn a_windowless_client_is_read_after_its_flag_and_needs_an_address() {
		assert_eq!(parse(&["--join", "127.0.0.1:9999"]), Run::Join(somewhere(9999)));
		assert_eq!(parse(&["--join=127.0.0.1:1234"]), Run::Join(somewhere(1234)));
		// **an address after the flag, not a port.** A payload nothing could
		// read as an address would let this pass whether the flag is looked
		// at or not, which is a test that cannot fail for the reason it is
		// about.
		assert_eq!(read(&["--connect", "127.0.0.1:9999"]).join, None, "and not somebody else's");
	}

	#[test]
	fn a_flag_with_nothing_usable_after_it_asks_for_nothing() {
		assert_eq!(
			parse(&["--join"]),
			Run::Window(Standing::Alone),
			"a flag on its own is not an address"
		);
		assert_eq!(
			parse(&["--join", "not-an-address"]),
			Run::Window(Standing::Alone),
			"and neither is a word"
		);
		assert_eq!(
			parse(&["--join", "127.0.0.1"]),
			Run::Window(Standing::Alone),
			"a host with no port is not one either, because a wire needs both"
		);
	}

	#[test]
	fn the_two_windowless_flags_do_not_read_each_other() {
		assert_eq!(
			read(&["--join", "127.0.0.1:9999"]).host,
			None,
			"asking to join is not asking to host"
		);
		assert_eq!(
			read(&["--connect", "127.0.0.1:27015"]).join,
			None,
			"and opening a window on one is not either"
		);
	}

	#[test]
	fn a_window_told_to_talk_is_read_and_a_word_that_is_not_an_address_is_not() {
		assert_eq!(
			parse(&["--connect", "127.0.0.1:27015"]),
			Run::Window(Standing::Talking(somewhere(27_015)))
		);
		assert_eq!(
			parse(&["--connect=127.0.0.1:1"]),
			Run::Window(Standing::Talking(somewhere(1)))
		);
		assert_eq!(parse(&["--connect"]), Run::Window(Standing::Alone), "with nothing after it");
		assert_eq!(
			parse(&["--connect", "not an address"]),
			Run::Window(Standing::Alone),
			"and with something that is not one"
		);
	}

	#[test]
	fn a_window_asked_to_listen_serves_on_the_port_it_was_given() {
		assert_eq!(
			parse(&["--listen"]),
			Run::Window(Standing::Serving(DEFAULT_PORT)),
			"on its own it is the usual port"
		);
		assert_eq!(parse(&["--listen", "9999"]), Run::Window(Standing::Serving(9999)));
		assert_eq!(parse(&["--listen=9999"]), Run::Window(Standing::Serving(9999)));
	}

	#[test]
	fn a_window_told_to_do_both_serves() {
		// a window cannot be both ends of a wire, and of the two claims the
		// stronger one is being the authority. Refusing to start over it would
		// be worse than picking and saying which was picked.
		assert_eq!(
			parse(&["--connect", "127.0.0.1:1", "--listen", "9999"]),
			Run::Window(Standing::Serving(9999))
		);
		assert_eq!(
			parse(&["--listen", "--connect", "127.0.0.1:1"]),
			Run::Window(Standing::Serving(DEFAULT_PORT)),
			"whichever order they were typed in"
		);
		assert_eq!(
			parse(&["--connect", "--listen", "9999"]),
			Run::Window(Standing::Serving(9999)),
			"and a flag is never taken as an address"
		);
	}

	#[test]
	fn a_project_is_named_beside_whichever_run_and_needs_a_directory() {
		let launch = Launch::parse(&words(&["--project", "C:/somewhere/demo", "--shot"]));

		assert_eq!(launch.project, Some(PathBuf::from("C:/somewhere/demo")));
		assert_eq!(launch.run, Run::Shot(PathBuf::from(shot::DEFAULT_PATH)), "and the run");
		assert_eq!(
			Launch::parse(&words(&["--project=demo"])).project,
			Some(PathBuf::from("demo")),
			"either way round"
		);
		assert_eq!(
			Launch::parse(&words(&["--project", "--host"])).project,
			None,
			"a flag is not a directory"
		);
		assert_eq!(Launch::parse(&[]).project, None, "and nothing named is nothing");
	}

	#[test]
	fn the_first_run_in_dispatch_order_wins_when_several_are_named() {
		assert_eq!(
			parse(&["--record", "x.wav", "--shot", "y.png"]),
			Run::Shot(PathBuf::from("y.png")),
			"a picture before a sound"
		);
		assert_eq!(
			parse(&["--join", "127.0.0.1:1", "--host"]),
			Run::Host(DEFAULT_PORT),
			"a host before a client"
		);
		assert_eq!(
			parse(&["--host", "--link"]),
			Run::Link(link::DEFAULT_STEPS),
			"and a two-endpoint run before everything"
		);
		assert_eq!(
			parse(&["--listen", "--shot"]),
			Run::Shot(PathBuf::from(shot::DEFAULT_PATH)),
			"a window is what is left when nothing else was asked for"
		);
	}
}
