//! Reading a line and doing what it says.
//!
//! One entry point, [`run`], which is what a terminal, a config file and - when
//! there is one - a console widget all go through. It is in `colby_core` rather
//! than in the runner because it needs [`World`] and nothing else: the same
//! function serves the host, the editor and anything the game builds.
//!
//! The grammar is the usual console one:
//!
//! ```text
//! sim.speed                  a variable on its own prints its value
//! sim.speed 0.25             a variable with a word sets it
//! game.reset                 a command runs
//! echo "two  words"          quotes keep spaces
//! sim.pause 1; sim.step 4    a semicolon starts another statement
//! // and this is a comment
//! ```
//!
//! Newlines separate statements too, which is what makes a config file and a
//! typed line the same language and lets the archive be written as one.

use std::{fs, path::Path};

use super::{
	World,
	cvar::{Args, Entry, Value},
	net::{Aim, PeerId},
};
use crate::{error, info, warn};

/// A line a command put off, waiting for whoever can answer it.
///
/// A [`ConsoleFn`](super::ConsoleFn) is handed a world and nothing else, and
/// some of what a console asks for is not in one: putting a saved world back
/// needs the solver, `net.say` needs the socket, a program's command needs the
/// interpreter, and all three are the runner's. A command like that leaves its
/// line on [`World::asked`] and whoever owns the thing it needs takes the line
/// off again and does the work.
///
/// **Who registered the name decides who takes the line.** The table already
/// says whose every entry is (@ref [`Owner`](super::cvar::Owner)): the frame
/// loop takes the engine's, the interpreter takes a program's, and a game
/// module's are left where they are for the game's own `update` to read. That
/// is the rule the wire uses for what a peer may run, and it turned out to be
/// the same line.
///
/// [`peer`](Self::peer) and [`aim`](Self::aim) are what the line was run with,
/// the two fields the runner swaps around a call. A line that waited is taken
/// up with the same pair, so a line from a peer stays that peer's rather than
/// becoming the host's for having waited a frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Asked {
	/// The name it was called under.
	pub name: String,

	/// The words that followed.
	pub words: Vec<String>,

	/// Who asked.
	pub peer: PeerId,

	/// Where they were pointing, or [`Aim::NONE`].
	pub aim: Aim,
}

/// How many lines may wait on a world at once.
///
/// Small on purpose. What waits is what was typed, said by a peer or asked by
/// a program since the previous frame, which is a handful; a name registered
/// with [`defer`] that nothing answers for would otherwise grow the queue for
/// as long as the process ran. Past this a line is refused with a message
/// naming it, which is how such a name gets found.
pub const MAX_WAITING: usize = 256;

/// The one command every line that has to wait is registered with.
///
/// Writes down what was asked, and who asked, and returns; @ref [`Asked`] for
/// who takes the line up. Its address is inside `colby_core`, which every
/// module shares and nothing unloads, so a name registered with it from
/// anywhere - the runner, a program, a game module - has no lifetime problem.
///
/// # Safety
///
/// As any [`ConsoleFn`](super::ConsoleFn): both pointers are live for the
/// duration of the call.
pub unsafe extern "C-unwind" fn defer(world: *mut World, args: *const Args) {
	// SAFETY: the console hands over a live world for the duration of the call.
	let world = unsafe { &mut *world };
	// SAFETY: and a live argument list beside it.
	let args = unsafe { &*args };

	if world.asked.len() >= MAX_WAITING {
		warn!(
			name = args.name(),
			waiting = world.asked.len(),
			"refused: too many lines are waiting, so something registered to wait is never \
			 answered"
		);

		return;
	}

	world.asked.push(Asked {
		name: args.name().to_owned(),
		words: args.words().to_vec(),
		peer: world.peer,
		aim: world.aim,
	});
}

/// Runs every statement in a line, a paste, or a whole file.
///
/// Nothing here is fallible in a way a caller could act on: a typo is a message
/// to whoever typed it, not an error to propagate.
///
/// @param world - the state commands are handed, and where the variables live
/// @param input - one or more statements
pub fn run(world: &mut World, input: &str) {
	for words in statements(input) {
		dispatch(world, &words);
	}
}

/// Runs the statements in a file.
///
/// @param world - as [`run`]
/// @param path - the file to read
/// @return whether it was read; a missing file is the caller's business, since
/// an absent config is normal and an absent `exec` argument is a mistake
pub fn exec(world: &mut World, path: &Path) -> bool {
	let text = match fs::read_to_string(path) {
		| Ok(text) => text,
		| Err(error) => {
			error!(path = %path.display(), %error, "could not read the config");

			return false;
		},
	};

	info!(path = %path.display(), "running config");
	run(world, &text);

	true
}

/// Does what one statement says.
fn dispatch(world: &mut World, words: &[String]) {
	let Some(name) = words.first().map(String::as_str) else {
		return;
	};

	// the pointer is copied out before anything else happens: a command is
	// about to be handed the whole world, and nothing may be holding a piece of
	// it at the time.
	if let Some(call) = world.cvars.call_of(name) {
		let args = Args::new(name.to_owned(), words[1..].to_vec());

		invoke(world, call, &args);

		return;
	}

	let Some(kind) = world
		.cvars
		.get(name)
		.and_then(Entry::value)
		.map(Value::type_name)
	else {
		warn!(name, "not a command or a variable; `help` lists what is");

		return;
	};

	let Some(text) = words.get(1) else {
		show(world, name);

		return;
	};

	if world.cvars.set(name, text) {
		show(world, name);
	} else {
		warn!(name, value = text, "{name} takes {kind}");
	}
}

/// Calls one command with the world.
///
/// @param world - handed across as a pointer, exactly as a `GameFn` is
/// @param call - the command's function
/// @param args - the words that followed its name
#[expect(
	ffi_unwind_calls,
	reason = "console commands are deliberately C-unwind, as the game boundary is: a command \
	          written in the game module and panicking should be catchable by the host rather \
	          than aborting the process"
)]
fn invoke(world: &mut World, call: super::cvar::ConsoleFn, args: &Args) {
	let pointer: *mut World = world;

	// SAFETY: `world` is a live World borrowed exclusively for this call and
	// nothing holds a piece of it, and `args` outlives the call. A command
	// belonging to a game module is removed from the table before that module
	// is unloaded, @ref `Cvars::forget_module`, so the pointer is into an image
	// that is still mapped.
	unsafe {
		call(pointer, args);
	}
}

/// Prints what a variable holds.
fn show(world: &World, name: &str) {
	let Some(value) = world.cvars.get(name).and_then(Entry::value) else {
		return;
	};

	info!("{name} is {}", value.quoted());
}

/// Splits input into statements, and statements into words.
///
/// @param input - one or more statements, separated by `;` or newlines
/// @return the words of each non-empty statement, quotes and comments removed
#[must_use]
pub fn statements(input: &str) -> Vec<Vec<String>> {
	let mut split = Split::default();
	let mut characters = input.chars().peekable();

	while let Some(character) = characters.next() {
		match character {
			// a byte-order mark is not a character anyone typed. Windows puts
			// one at the top of a file saved as UTF-8 and at the head of a
			// redirected stream, and without this the first word of the first
			// line is a name nothing is registered under. The OBJ importer
			// learned the same lesson.
			| '\u{feff}' if !split.quoted => {},
			| '"' => split.quote(),
			| '/' if !split.quoted && characters.peek() == Some(&'/') => {
				// a comment runs to the end of its line, and the line ends the
				// statement it was part of.
				while characters.next_if(|next| *next != '\n').is_some() {}
			},
			| ';' | '\n' if !split.quoted => split.end_statement(),
			| _ if character.is_whitespace() && !split.quoted => split.end_word(),
			| _ => split.push(character),
		}
	}

	split.finish()
}

/// The tokenizer's running state.
///
/// A struct rather than five locals because the loop that drives it has to stay
/// shallow enough to read, and because "end this word" and "end this statement"
/// are the two things it actually does.
#[derive(Debug, Default)]
struct Split {
	statements: Vec<Vec<String>>,
	words: Vec<String>,
	word: String,
	/// Whether a word has started. What tells an empty `""` from no word at
	/// all.
	began: bool,
	/// Whether we are inside quotes, where separators are ordinary characters.
	quoted: bool,
}

impl Split {
	/// Adds one character to the word being built.
	fn push(&mut self, character: char) {
		self.word.push(character);
		self.began = true;
	}

	/// Opens or closes quotes.
	fn quote(&mut self) {
		self.quoted = !self.quoted;
		self.began = true;
	}

	/// Finishes the word being built, if there is one.
	fn end_word(&mut self) {
		if !self.began {
			return;
		}

		self.words.push(std::mem::take(&mut self.word));
		self.began = false;
	}

	/// Finishes the statement being built, if it has any words.
	fn end_statement(&mut self) {
		self.end_word();

		if self.words.is_empty() {
			return;
		}

		self.statements
			.push(std::mem::take(&mut self.words));
	}

	/// Everything, with the last statement closed.
	fn finish(mut self) -> Vec<Vec<String>> {
		self.end_statement();

		self.statements
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The words of the one statement in `input`.
	fn words(input: &str) -> Vec<String> {
		statements(input)
			.into_iter()
			.next()
			.unwrap_or_default()
	}

	/// A command that writes what it was given somewhere a test can see it.
	///
	/// # Safety
	///
	/// As any [`ConsoleFn`](super::cvar::ConsoleFn).
	unsafe extern "C-unwind" fn note(world: *mut World, args: *const Args) {
		// SAFETY: `run` hands over a live world for the duration of the call.
		let world = unsafe { &mut *world };
		// SAFETY: and a live argument list.
		let args = unsafe { &*args };

		world.quit = args.word(0) != Some("no");
		world.owed_steps = u32::try_from(args.len()).unwrap_or(0);
	}

	/// A world with one command and one variable in it.
	fn world() -> World {
		let mut world = World::new();
		world.cvars.command("note", note, "");
		world
			.cvars
			.var("test.speed", Value::Float(1.0), "");

		world
	}

	#[test]
	fn a_variable_on_its_own_line_is_set_by_the_next_word() {
		let mut world = world();
		run(&mut world, "test.speed 2.5");

		assert_eq!(world.cvars.float("test.speed"), Some(2.5), "the word after it is the value");

		run(&mut world, "test.speed hello");

		assert_eq!(
			world.cvars.float("test.speed"),
			Some(2.5),
			"and a word that is not a number changes nothing"
		);
	}

	#[test]
	fn a_command_is_called_with_the_words_that_followed_it() {
		let mut world = world();
		run(&mut world, "note one two three");

		assert!(world.quit, "the command ran");
		assert_eq!(world.owed_steps, 3, "with three words, and not its own name among them");

		run(&mut world, "note no");

		assert!(!world.quit, "and it can see what they were");
	}

	#[test]
	fn several_statements_run_in_the_order_they_were_written() {
		let mut world = world();
		run(&mut world, "test.speed 3; note; test.speed 4");

		assert!(world.quit, "the middle one ran");
		assert_eq!(world.cvars.float("test.speed"), Some(4.0), "and the last one won");
	}

	#[test]
	fn a_name_nobody_registered_changes_nothing() {
		let mut world = world();
		run(&mut world, "nonsense 3");

		assert!(!world.quit, "an unknown name is a message to whoever typed it");
		assert_eq!(world.cvars.len(), 2, "and does not quietly become a variable");
	}

	#[test]
	fn a_written_value_reads_back_as_the_same_value() {
		let mut world = world();
		world
			.cvars
			.var("test.text", Value::Text(String::new()), "");
		world.cvars.set("test.text", "two  words");

		// what the archive does: write the value out, read the file back in.
		let written = format!(
			"test.text {}",
			world
				.cvars
				.get("test.text")
				.and_then(Entry::value)
				.expect("it is a variable")
				.quoted()
		);
		world.cvars.set("test.text", "something else");
		run(&mut world, &written);

		assert_eq!(
			world.cvars.text("test.text"),
			Some("two  words"),
			"a config file is only a config file if it round-trips"
		);
	}

	#[test]
	fn a_line_splits_into_words() {
		assert_eq!(words("sim.speed 0.25"), ["sim.speed", "0.25"], "on whitespace");
		assert_eq!(words("  sim.speed   0.25  "), ["sim.speed", "0.25"], "however much of it");
	}

	#[test]
	fn a_semicolon_and_a_newline_both_start_another_statement() {
		let both = statements("sim.pause 1; sim.step 4\ngame.reset");

		assert_eq!(both.len(), 3, "two separators, three statements");
		assert_eq!(both[1], ["sim.step", "4"], "and the middle one is intact");
	}

	#[test]
	fn quotes_hold_a_word_together() {
		assert_eq!(words(r#"echo "two  words""#), ["echo", "two  words"], "spaces and all");
		assert_eq!(
			words(r#"echo "one; two""#),
			["echo", "one; two"],
			"a semicolon inside quotes is a semicolon, not a separator"
		);
		assert_eq!(words(r#"echo """#), ["echo", ""], "and an empty quote is an empty word");
	}

	#[test]
	fn a_comment_runs_to_the_end_of_its_line() {
		let lines = statements("sim.pause 1 // why not\ngame.reset");

		assert_eq!(lines[0], ["sim.pause", "1"], "the comment is dropped");
		assert_eq!(lines[1], ["game.reset"], "and the next line is not");
		assert!(statements("// nothing here").is_empty(), "a whole comment is no statement");
		assert_eq!(
			words(r#"echo "// not a comment""#),
			["echo", "// not a comment"],
			"quotes hold, here too"
		);
	}

	#[test]
	fn a_byte_order_mark_is_not_part_of_the_first_word() {
		assert_eq!(
			words("\u{feff}help sim"),
			["help", "sim"],
			"a file saved as UTF-8 on Windows starts with one, and so does a redirected stream"
		);
		assert_eq!(
			statements("sim.pause 1\n\u{feff}sim.step 2").len(),
			2,
			"and it can turn up at the head of any line, not only the first"
		);
	}

	#[test]
	fn nothing_is_not_a_statement() {
		assert!(statements("").is_empty(), "an empty line");
		assert!(statements("   \n\n  ;  ").is_empty(), "or one with nothing but separators");
	}

	/// A world with one command in it that waits rather than acting.
	fn waiting() -> World {
		let mut world = World::new();
		world
			.cvars
			.command("scene.later", defer, "leaves its line for the frame loop");

		world
	}

	#[test]
	fn a_line_put_off_is_on_the_world_with_who_said_it_and_where_they_pointed() {
		let mut world = waiting();
		// somebody in particular, pointing somewhere in particular: the two
		// fields a runner swaps around a call are what a line that waited has
		// to carry, or it is run later as whoever happens to be asking then.
		let somebody = PeerId::from_bits((3 << 32) | 2);
		let there = Aim {
			origin: [1.0, 2.0, 3.0],
			direction: [0.0, 0.0, -1.0],
		};

		world.peer = somebody;
		world.aim = there;
		run(&mut world, "scene.later one two");

		assert_eq!(world.asked, [Asked {
			name: "scene.later".to_owned(),
			words: vec!["one".to_owned(), "two".to_owned()],
			peer: somebody,
			aim: there,
		}]);
		assert!(!world.quit, "and nothing else happened");
	}

	#[test]
	fn lines_wait_in_the_order_they_were_asked() {
		let mut world = waiting();

		run(&mut world, "scene.later first; scene.later second");

		let names: Vec<&str> = world
			.asked
			.iter()
			.map(|asked| asked.words[0].as_str())
			.collect();

		assert_eq!(names, ["first", "second"], "a queue rather than a slot");
	}

	#[test]
	fn a_queue_nobody_drains_stops_taking_lines_rather_than_growing() {
		let mut world = waiting();

		for _ in 0..MAX_WAITING + 8 {
			run(&mut world, "scene.later");
		}

		assert_eq!(
			world.asked.len(),
			MAX_WAITING,
			"the ceiling holds, and the lines past it are refused rather than kept"
		);
	}
}
