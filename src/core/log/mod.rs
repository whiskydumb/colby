//! Process-wide tracing setup.
//!
//! @note: the subscriber is installed as tracing's *global* default rather than
//! being threaded around. Under `just hot` the game module links colby_core
//! dynamically, which means it shares this crate's copy of `tracing-core` and
//! therefore this dispatcher: `tracing::info!` inside the hot-reloaded module
//! lands in the host's log with no boundary function of its own. That property
//! is the reason colby_core is built as a dylib, and it breaks the moment the
//! module ends up with its own statically linked copy.

use std::{
	collections::VecDeque,
	fmt::{Debug, Write as _},
	io::{IsTerminal, stdout},
	sync::{
		Mutex, OnceLock, PoisonError,
		atomic::{AtomicU64, Ordering},
	},
};

use tracing::{
	Event, Level, Subscriber,
	field::{Field, Visit},
};
use tracing_subscriber::{EnvFilter, Layer, fmt, layer::Context, prelude::*};

use crate::Result;

/// The environment variable that overrides the default log filter.
pub const FILTER_ENV: &str = "COLBY_LOG";

/// The filter used when [`FILTER_ENV`] is unset or unparsable.
pub const DEFAULT_FILTER: &str = "info,colby=debug";

/// How many lines are kept for anything that wants to show them.
pub const KEPT_LINES: usize = 512;

/// One line of log, for something that is not a terminal.
///
/// The terminal gets the same events through `tracing-subscriber`'s own layer;
/// this is what the editor's console reads, and the reason a `tracing::info!`
/// written inside the *game module* turns up there too - host and module share
/// one dispatcher, @ref the note at the top of this file.
#[derive(Clone, Debug)]
pub struct Line {
	/// How loud it was.
	pub level: Level,

	/// Which module said it.
	pub target: String,

	/// What it said, with its fields after it.
	pub message: String,
}

/// The lines kept so far.
static LINES: OnceLock<Mutex<VecDeque<Line>>> = OnceLock::new();

/// How many lines have ever been logged.
///
/// Only ever grows, so anything showing the lines can tell in one atomic read
/// whether there is anything new to copy.
static LOGGED: AtomicU64 = AtomicU64::new(0);

/// How many lines have ever been logged.
#[must_use]
pub fn logged() -> u64 { LOGGED.load(Ordering::Relaxed) }

/// Copies the lines kept so far, replacing whatever was in `into`.
///
/// A copy rather than a lock the caller holds, on purpose: a console that held
/// the lock while it drew would deadlock the moment something it drew logged
/// anything - which is exactly what a console is for. @ref [`logged`] for how
/// to avoid making the copy every frame.
///
/// @param into - the buffer to fill, reused between calls
pub fn copy_lines(into: &mut Vec<Line>) {
	into.clear();

	let kept = LINES
		.get_or_init(|| Mutex::new(VecDeque::with_capacity(KEPT_LINES)))
		.lock()
		.unwrap_or_else(PoisonError::into_inner);

	into.extend(kept.iter().cloned());
}

/// Installs the global tracing subscriber.
///
/// Call this once, from the host, before anything else logs. A second call is
/// an error rather than a silent no-op, because the second subscriber would
/// swallow every line.
///
/// @return `Ok` when this process now has a subscriber installed
pub fn init() -> Result {
	let filter =
		EnvFilter::try_from_env(FILTER_ENV).unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));

	// @note: color only when someone is looking at it. tracing-subscriber turns
	// ansi on whenever the feature is compiled in, which fills a redirected log
	// with escape sequences and makes it a nuisance to grep.
	let layer = fmt::layer()
		.with_ansi(stdout().is_terminal())
		.with_target(true)
		.with_thread_names(false)
		.with_level(true);

	tracing_subscriber::registry()
		.with(filter)
		.with(layer)
		.with(Kept)
		.try_init()
		.map_err(|error| crate::err!("installing the tracing subscriber: {error}"))
}

/// Adds one line to what is kept, dropping the oldest if there is no room.
fn keep(line: Line) {
	let mut kept = LINES
		.get_or_init(|| Mutex::new(VecDeque::with_capacity(KEPT_LINES)))
		.lock()
		.unwrap_or_else(PoisonError::into_inner);

	if kept.len() >= KEPT_LINES {
		kept.pop_front();
	}

	kept.push_back(line);
	LOGGED.fetch_add(1, Ordering::Relaxed);
}

/// The layer that keeps them.
struct Kept;

impl<S: Subscriber> Layer<S> for Kept {
	fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
		let mut words = Words::default();
		event.record(&mut words);

		keep(Line {
			level: *event.metadata().level(),
			target: event.metadata().target().to_owned(),
			message: words.text,
		});
	}
}

/// An event's message and fields, flattened into one string.
///
/// The same shape the terminal shows: the message first, then `field=value` for
/// everything else, so that a line read in the editor and a line read in a
/// redirected log say the same thing.
#[derive(Debug, Default)]
struct Words {
	text: String,
}

impl Visit for Words {
	fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
		// a String stops growing only when the allocator gives up, and there is
		// nothing this could report that to.
		let _written = if field.name() == "message" {
			write!(self.text, "{value:?}")
		} else {
			write!(self.text, " {}={:?}", field.name(), value)
		};
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Held for the length of every test here.
	///
	/// The buffer is one per process on purpose - it is what the editor reads,
	/// and there is one editor - so two tests writing into it at once would
	/// each see the other's lines. This is the cost of testing a global, and it
	/// is cheaper than pretending the global is not one.
	static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

	#[test]
	fn a_line_is_kept_with_its_fields_after_its_message() {
		let _held = ONE_AT_A_TIME
			.lock()
			.unwrap_or_else(PoisonError::into_inner);
		let before = logged();

		// a scoped subscriber rather than the global one: `init` can only be
		// called once in a process, and a test that spent it would take the
		// logger away from every other test.
		let subscriber = tracing_subscriber::registry().with(Kept);
		tracing::subscriber::with_default(subscriber, || {
			tracing::info!(count = 3, "the ring turned");
		});

		let mut lines = Vec::new();
		copy_lines(&mut lines);
		let last = lines.last().expect("the event was kept");

		assert_eq!(logged(), before + 1, "one event, one line");
		assert_eq!(last.level, Level::INFO, "the level is kept as a level, not as text");
		assert_eq!(
			last.message, "the ring turned count=3",
			"the message first and the fields after it, the way the terminal shows them"
		);
		assert!(
			last.target.starts_with("colby_core"),
			"and whoever said it, got {}",
			last.target
		);
	}

	#[test]
	fn the_oldest_lines_fall_off_the_end() {
		let _held = ONE_AT_A_TIME
			.lock()
			.unwrap_or_else(PoisonError::into_inner);

		for index in 0..KEPT_LINES + 10 {
			keep(Line {
				level: Level::TRACE,
				target: "test".to_owned(),
				message: index.to_string(),
			});
		}

		let mut lines = Vec::new();
		copy_lines(&mut lines);

		assert_eq!(lines.len(), KEPT_LINES, "this is a window on the log, not the log");
		assert_eq!(
			lines.last().map(|line| line.message.as_str()),
			Some((KEPT_LINES + 9).to_string()).as_deref(),
			"and it is the newest lines that are in it"
		);
	}
}
