//! Console variables and commands: one namespace the engine, the host and the
//! game all register into.
//!
//! A **variable** is a named value someone reads every frame - `sim.speed`,
//! `game.spin_rate`. A **command** is a named function pointer taking the world
//! and the words that followed it. They share one table because a console wants
//! one lookup and one listing, so a kind is a bit on an entry rather than a
//! second registry.
//!
//! Three rules come from the hot-reload boundary, and all three are the
//! difference between this working and this crashing:
//!
//! 1. **Names and help text are copied, never borrowed.** A `&'static str` from
//!    a game module points into an image the host is about to unload.
//! 2. **A module's commands are dropped before its library is.** The function
//!    pointer in a [`Kind::Command`] registered by the game is an address
//!    inside that library. @ref [`Cvars::forget_module`].
//! 3. **A module's variables survive the swap, and so does anything the user
//!    typed.** A reload that reset every value would make the console useless
//!    for the thing it is most useful for - turning a number while watching the
//!    result. A value nobody has touched follows the code's default, so
//!    changing a default and saving still works. @ref [`Cvars::var`].
//!
//! There is deliberately no callback on a variable. It would be a function
//! pointer with exactly the lifetime problem above, and a value the engine
//! reads when it needs it has no lifetime at all.

use super::World;

/// What a variable holds.
///
/// Four kinds, and no more. A variable keeps the type it was
/// registered with: setting it parses into that type or refuses, so a float
/// cannot quietly become a string because someone typed a word.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
	/// `true` or `false`. Also accepts `1`/`0`, `on`/`off`, `yes`/`no`.
	Bool(bool),

	/// A whole number.
	Int(i64),

	/// A number. `f32` because everything else in the engine is.
	Float(f32),

	/// Anything at all.
	Text(String),
}

impl Value {
	/// What to call this kind of value in a message.
	#[must_use]
	pub const fn type_name(&self) -> &'static str {
		match self {
			| Self::Bool(_) => "a boolean",
			| Self::Int(_) => "a whole number",
			| Self::Float(_) => "a number",
			| Self::Text(_) => "text",
		}
	}

	/// Whether two values are the same kind, whatever they hold.
	#[must_use]
	pub const fn same_kind(&self, other: &Self) -> bool {
		matches!(
			(self, other),
			(Self::Bool(_), Self::Bool(_))
				| (Self::Int(_), Self::Int(_))
				| (Self::Float(_), Self::Float(_))
				| (Self::Text(_), Self::Text(_))
		)
	}

	/// Replaces the value from text, keeping the kind.
	///
	/// @param text - what the user typed
	/// @return `false` if the text is not that kind of value, in which case
	/// nothing was written
	pub fn set_text(&mut self, text: &str) -> bool {
		match self {
			| Self::Bool(held) => match text {
				| "true" | "1" | "on" | "yes" => *held = true,
				| "false" | "0" | "off" | "no" => *held = false,
				| _ => return false,
			},
			| Self::Int(held) => match text.parse() {
				| Ok(parsed) => *held = parsed,
				| Err(_) => return false,
			},
			| Self::Float(held) => match text.parse::<f32>() {
				| Ok(parsed) if parsed.is_finite() => *held = parsed,
				| _ => return false,
			},
			| Self::Text(held) => text.clone_into(held),
		}

		true
	}

	/// The value as text, the way it would be typed back in.
	#[must_use]
	pub fn text(&self) -> String {
		match self {
			| Self::Bool(held) => held.to_string(),
			| Self::Int(held) => held.to_string(),
			| Self::Float(held) => held.to_string(),
			| Self::Text(held) => held.clone(),
		}
	}

	/// The same, quoted if it would not survive being read back as one word.
	#[must_use]
	pub fn quoted(&self) -> String {
		let text = self.text();

		if text.is_empty() || text.contains([' ', '\t', ';', '"']) {
			return format!("\"{}\"", text.replace('"', ""));
		}

		text
	}
}

/// The signature every console command has.
///
/// `C-unwind` for the same reason [`GameFn`](super::GameFn) is: a panic inside
/// a command written in the game module should be caught by the host and
/// reported, not abort the process.
///
/// # Safety
///
/// `world` must point to a live [`World`] that nothing else is touching for the
/// duration of the call, and `args` to a live [`Args`]. A command registered by
/// a module must be removed before that module is unloaded, @ref
/// [`Cvars::forget_module`] - calling one afterwards is a jump into freed
/// memory.
pub type ConsoleFn = unsafe extern "C-unwind" fn(world: *mut World, args: *const Args);

/// The words that followed a command's own name.
///
/// Reached by pointer across the boundary, like [`World`]: it is host Rust data
/// and both sides share one definition of it through `colby_core`.
#[derive(Clone, Debug, Default)]
pub struct Args {
	name: String,
	words: Vec<String>,
}

impl Args {
	/// One invocation: what was typed, and what followed it.
	///
	/// @param name - the command's own name, as it was typed
	/// @param words - the words after it, in order
	#[must_use]
	pub fn new(name: String, words: Vec<String>) -> Self { Self { name, words } }

	/// The name this command was called under.
	///
	/// **Almost nothing wants this, and one thing cannot work without it.** A
	/// [`ConsoleFn`] is a bare function pointer with no context, so a single
	/// function standing in for a whole *table* of commands, the way a
	/// script's are, has no other way to find out which of them was asked for.
	/// Everything registered one name at a time can ignore it.
	#[must_use]
	pub fn name(&self) -> &str { &self.name }

	/// How many there are.
	#[must_use]
	pub fn len(&self) -> usize { self.words.len() }

	/// Whether the command was invoked bare.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.words.is_empty() }

	/// One word.
	#[must_use]
	pub fn word(&self, index: usize) -> Option<&str> { self.words.get(index).map(String::as_str) }

	/// One word as a number.
	#[must_use]
	pub fn float(&self, index: usize) -> Option<f32> {
		self.word(index)?
			.parse()
			.ok()
			.filter(|it: &f32| it.is_finite())
	}

	/// One word as a whole number.
	#[must_use]
	pub fn int(&self, index: usize) -> Option<i64> { self.word(index)?.parse().ok() }

	/// Everything, joined back together with single spaces.
	#[must_use]
	pub fn rest(&self) -> String { self.words.join(" ") }
}

/// Who registered an entry, and therefore what happens to it on a reload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Owner {
	/// The host or the engine. Lives as long as the process.
	#[default]
	Engine,

	/// The game module. Its commands go when the module does.
	Module,

	/// A program the interpreter is running. Its commands go when it does.
	///
	/// Kept apart from [`Module`](Self::Module) because the two are dropped by
	/// different things at different moments: a module's commands go before its
	/// library is unloaded, all at once, while a program's go one program at a
	/// time as each is built again. The *pointer* is not the reason - a
	/// program's command is a function inside the host, which is never
	/// unloaded, so it is the one kind here with no lifetime problem at all.
	Script,
}

/// A variable with a value, or a command with a function.
#[derive(Clone, Debug)]
pub enum Kind {
	/// A value, and the value it was registered with.
	Var {
		/// What it holds now.
		value: Value,

		/// What the code says it should hold. @ref [`Cvars::var`].
		default: Value,
	},

	/// Something to call.
	Command(ConsoleFn),
}

/// One entry in the table.
#[derive(Clone, Debug)]
pub struct Entry {
	name: String,
	help: String,
	kind: Kind,
	owner: Owner,
	/// Whether this is written to the config file when the process stops.
	archived: bool,
	/// Whether anything has set this since it was registered. What keeps a
	/// reload from undoing a value the user typed.
	touched: bool,
	/// Whether the module that registered this has gone away without
	/// registering it again. @ref [`Cvars::sweep`].
	stale: bool,
}

impl Entry {
	/// What it is called.
	#[must_use]
	pub fn name(&self) -> &str { &self.name }

	/// One line describing it, for `help`.
	#[must_use]
	pub fn help(&self) -> &str { &self.help }

	/// A variable's value, or `None` for a command.
	#[must_use]
	pub const fn value(&self) -> Option<&Value> {
		match &self.kind {
			| Kind::Var { value, .. } => Some(value),
			| Kind::Command(_) => None,
		}
	}

	/// Whether this is something to call rather than something to read.
	#[must_use]
	pub const fn is_command(&self) -> bool { matches!(self.kind, Kind::Command(_)) }

	/// Whether this is written to the config file.
	#[must_use]
	pub const fn is_archived(&self) -> bool { self.archived }

	/// Who registered it.
	#[must_use]
	pub const fn owner(&self) -> Owner { self.owner }
}

/// Every console variable and command in the process.
///
/// Held by [`World`], because the game reaches everything through that one
/// pointer and a console the game cannot read would be a console for half the
/// program.
///
/// A `Vec` and a linear search rather than a map: there are tens of these, they
/// are looked up once a frame at most, and the order they were registered in is
/// the order `help` should list them.
#[derive(Clone, Debug, Default)]
pub struct Cvars {
	entries: Vec<Entry>,

	/// Who anything registered from now on belongs to.
	///
	/// A mode rather than an argument on every call, because the game should
	/// not have to say who it is, and because the host knows exactly when the
	/// module is the one talking: it sets this before the module's `init` and
	/// leaves it, so that a game registering something from `update` is
	/// attributed correctly too. Getting that wrong would leave a command
	/// pointing into an unloaded library.
	owner: Owner,
}

impl Cvars {
	/// An empty table.
	#[must_use]
	pub const fn new() -> Self {
		Self {
			entries: Vec::new(),
			owner: Owner::Engine,
		}
	}

	/// Registers a variable, or brings an existing one up to date.
	///
	/// Idempotent, because the game calls it from `init` and `init` runs again
	/// on every reload. What happens the second time is the whole design:
	///
	/// - a value **nobody has set** takes the new default, so editing a default
	///   in code and saving does what it looks like it does;
	/// - a value **someone set** is kept, so a reload does not undo the number
	///   you are in the middle of tuning;
	/// - a value whose **kind changed** takes the new default either way, since
	///   the old one is no longer a value of the right sort.
	///
	/// @param name - dotted, lowercase, `sim.speed`
	/// @param default - the value and, by its variant, the kind
	/// @param help - one line, shown by `help`
	pub fn var(&mut self, name: &str, default: Value, help: &str) {
		self.insert(name, Kind::Var { value: default.clone(), default }, help, false);
	}

	/// The same, and written to the config file when the process stops.
	pub fn saved(&mut self, name: &str, default: Value, help: &str) {
		self.insert(name, Kind::Var { value: default.clone(), default }, help, true);
	}

	/// Registers a command, or replaces an existing one.
	///
	/// Replacing matters on a reload: the new module's code is at a new
	/// address, and the old pointer is into an image that no longer exists.
	///
	/// @param name - what is typed to invoke it
	/// @param call - what to call
	/// @param help - one line, shown by `help`
	pub fn command(&mut self, name: &str, call: ConsoleFn, help: &str) {
		self.insert(name, Kind::Command(call), help, false);
	}

	/// Marks everything registered from here on as the game module's.
	///
	/// The host sets this around the module's `init` so that a game does not
	/// have to say who it is on every line. @ref [`Cvars::forget_module`].
	///
	/// @param owner - who to attribute new and re-registered entries to
	pub fn attribute(&mut self, owner: Owner) { self.owner = owner; }

	/// Everything, in the order it was registered.
	pub fn iter(&self) -> impl Iterator<Item = &Entry> { self.entries.iter() }

	/// How many entries there are.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Whether nothing has been registered.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// One entry by name.
	#[must_use]
	pub fn get(&self, name: &str) -> Option<&Entry> {
		self.entries
			.iter()
			.find(|entry| entry.name == name)
	}

	/// A command's function, if that name is one.
	///
	/// Copied out rather than borrowed on purpose: the caller is about to hand
	/// the whole world to it, and cannot be holding a piece of it at the time.
	#[must_use]
	pub fn call_of(&self, name: &str) -> Option<ConsoleFn> {
		match self.get(name)?.kind {
			| Kind::Command(call) => Some(call),
			| Kind::Var { .. } => None,
		}
	}

	/// A variable's value as a boolean, if it is one.
	#[must_use]
	pub fn bool(&self, name: &str) -> Option<bool> {
		match self.get(name)?.value()? {
			| Value::Bool(held) => Some(*held),
			| _ => None,
		}
	}

	/// A variable's value as a whole number, if it is one.
	#[must_use]
	pub fn int(&self, name: &str) -> Option<i64> {
		match self.get(name)?.value()? {
			| Value::Int(held) => Some(*held),
			| _ => None,
		}
	}

	/// A variable's value as a number, if it is one.
	#[must_use]
	pub fn float(&self, name: &str) -> Option<f32> {
		match self.get(name)?.value()? {
			| Value::Float(held) => Some(*held),
			| _ => None,
		}
	}

	/// A variable's value as text, if it is text.
	#[must_use]
	pub fn text(&self, name: &str) -> Option<&str> {
		match self.get(name)?.value()? {
			| Value::Text(held) => Some(held),
			| _ => None,
		}
	}

	/// Sets a variable from text, keeping its kind.
	///
	/// @param name - the variable to set
	/// @param text - the value, as typed
	/// @return `false` if there is no such variable, it is a command, or the
	/// text is not a value of the right kind - nothing is written in any of
	/// those cases
	pub fn set(&mut self, name: &str, text: &str) -> bool {
		let Some(entry) = self
			.entries
			.iter_mut()
			.find(|entry| entry.name == name)
		else {
			return false;
		};

		let Kind::Var { value, .. } = &mut entry.kind else {
			return false;
		};

		if !value.set_text(text) {
			return false;
		}

		entry.touched = true;

		true
	}

	/// Puts a variable back to the value the code registered it with.
	///
	/// @return `false` if there is no such variable
	pub fn reset(&mut self, name: &str) -> bool {
		let Some(entry) = self
			.entries
			.iter_mut()
			.find(|entry| entry.name == name)
		else {
			return false;
		};

		let Kind::Var { value, default } = &mut entry.kind else {
			return false;
		};

		*value = default.clone();
		entry.touched = false;

		true
	}

	/// Drops the game module's commands and marks its variables stale.
	///
	/// Called by the host **before** the library is unloaded, which is the only
	/// moment it can be: every [`Kind::Command`] the module registered is an
	/// address inside the image about to be freed.
	///
	/// The variables stay, because they are plain data and because a reload
	/// should not throw away a number someone is in the middle of turning. What
	/// the mark is for is the other case - a variable the new build no longer
	/// registers, which [`sweep`](Self::sweep) removes once the new `init` has
	/// had its say.
	pub fn forget_module(&mut self) {
		self.entries
			.retain(|entry| !(entry.owner == Owner::Module && entry.is_command()));

		for entry in &mut self.entries {
			if entry.owner == Owner::Module {
				entry.stale = true;
			}
		}
	}

	/// Drops one command a program published.
	///
	/// Called when the program that published it is built again or goes away,
	/// so that a name a program no longer registers stops answering. Only a
	/// script's own commands can go this way: a program may not remove
	/// `quit`, and a table where anything could remove anything is a table
	/// nobody can reason about.
	///
	/// @param name - what to drop
	/// @return whether there was one to drop
	pub fn forget_script(&mut self, name: &str) -> bool {
		let was = self.entries.len();

		self.entries
			.retain(|entry| !(entry.name == name && entry.owner == Owner::Script));

		self.entries.len() != was
	}

	/// Removes what the reloaded module did not register again.
	///
	/// Called by the host after the module's `init`. A variable that was there
	/// before the swap and is not now was renamed or deleted in the source, and
	/// keeping it would leave `help` describing a build nobody is running.
	pub fn sweep(&mut self) { self.entries.retain(|entry| !entry.stale); }

	/// The shared half of registering anything.
	fn insert(&mut self, name: &str, kind: Kind, help: &str, archived: bool) {
		let owner = self.owner;

		if let Some(entry) = self
			.entries
			.iter_mut()
			.find(|entry| entry.name == name)
		{
			entry.adopt(kind, help, owner, archived);

			return;
		}

		self.entries.push(Entry {
			// copied, never borrowed: a `&'static str` from a game module
			// points into an image the host will unload.
			name: name.to_owned(),
			help: help.to_owned(),
			kind,
			owner,
			archived,
			touched: false,
			stale: false,
		});
	}
}

impl Entry {
	/// Brings an existing entry up to date with a fresh registration.
	fn adopt(&mut self, kind: Kind, help: &str, owner: Owner, archived: bool) {
		help.clone_into(&mut self.help);
		self.owner = owner;
		self.archived = archived;
		self.stale = false;

		match (&mut self.kind, kind) {
			| (Kind::Var { value, default }, Kind::Var { default: fresh, .. }) => {
				// the value only follows the code when nobody has moved it, and
				// always when the kind itself changed under it.
				if !self.touched || !value.same_kind(&fresh) {
					*value = fresh.clone();
					self.touched = false;
				}

				*default = fresh;
			},
			| (held, fresh) => {
				*held = fresh;
				self.touched = false;
			},
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A command that does nothing, for the entries that need one.
	///
	/// # Safety
	///
	/// Touches neither pointer.
	unsafe extern "C-unwind" fn nothing(_world: *mut World, _args: *const Args) {}

	/// A second one, so that "the pointer was replaced" is a thing a test can
	/// see.
	///
	/// # Safety
	///
	/// As [`nothing`].
	unsafe extern "C-unwind" fn also_nothing(_world: *mut World, _args: *const Args) {}

	#[test]
	fn a_program_may_take_back_its_own_command_and_nobody_elses() {
		// the whole reason this is not a general `remove`: a table where
		// anything can drop anything is a table nobody can reason about, and
		// the thing being given the ability here is a program somebody wrote
		// in a text file.
		let mut cvars = Cvars::new();

		cvars.command("engine.stop", nothing, "the engine's own");
		cvars.attribute(Owner::Module);
		cvars.command("game.go", nothing, "the game's own");
		cvars.attribute(Owner::Script);
		cvars.command("mine.go", nothing, "a program's own");
		cvars.attribute(Owner::Engine);

		assert!(!cvars.forget_script("engine.stop"), "the engine's stays");
		assert!(!cvars.forget_script("game.go"), "and so does the game's");
		assert!(cvars.get("engine.stop").is_some());
		assert!(cvars.get("game.go").is_some());

		assert!(cvars.forget_script("mine.go"), "and a program's own goes");
		assert!(cvars.get("mine.go").is_none());

		assert!(!cvars.forget_script("mine.go"), "and going twice is not going");
	}

	#[test]
	fn a_command_knows_which_of_its_names_it_was_called_under() {
		// what one function standing in for a whole table of commands has to
		// have. Two names, one function, and the only thing telling them apart
		// is this.
		let mut cvars = Cvars::new();

		cvars.command("first", nothing, "one");
		cvars.command("second", nothing, "two");

		assert_eq!(Args::new("first".to_owned(), Vec::new()).name(), "first");
		assert_eq!(
			Args::new("second".to_owned(), vec!["a".to_owned()]).name(),
			"second",
			"and the words after it are not part of it"
		);
		assert_eq!(Args::new("second".to_owned(), vec!["a".to_owned()]).word(0), Some("a"));
	}

	#[test]
	fn a_registered_variable_reads_back_as_what_it_is() {
		let mut cvars = Cvars::new();
		cvars.var("a.number", Value::Float(2.5), "");
		cvars.var("a.count", Value::Int(7), "");
		cvars.var("a.flag", Value::Bool(true), "");
		cvars.var("a.word", Value::Text("hello".to_owned()), "");

		assert_eq!(cvars.float("a.number"), Some(2.5), "a number reads as a number");
		assert_eq!(cvars.int("a.count"), Some(7), "and a count as a count");
		assert_eq!(cvars.bool("a.flag"), Some(true), "and a flag as a flag");
		assert_eq!(cvars.text("a.word"), Some("hello"), "and text as text");
		assert_eq!(cvars.float("a.count"), None, "asking for the wrong kind is a miss");
		assert_eq!(cvars.float("a.missing"), None, "and so is asking for nothing");
	}

	#[test]
	fn setting_keeps_the_kind_a_variable_was_registered_with() {
		let mut cvars = Cvars::new();
		cvars.var("a.number", Value::Float(1.0), "");

		assert!(cvars.set("a.number", "2.5"), "a number takes a number");
		assert_eq!(cvars.float("a.number"), Some(2.5), "and holds it");
		assert!(!cvars.set("a.number", "quite fast"), "and refuses a word");
		assert_eq!(
			cvars.float("a.number"),
			Some(2.5),
			"leaving what was there rather than half-writing something"
		);
	}

	#[test]
	fn a_flag_takes_the_words_people_type() {
		let mut cvars = Cvars::new();
		cvars.var("a.flag", Value::Bool(false), "");

		for text in ["true", "1", "on", "yes"] {
			assert!(cvars.set("a.flag", text), "`{text}` is a word for true");
			assert_eq!(cvars.bool("a.flag"), Some(true), "`{text}` means true");
		}

		for text in ["false", "0", "off", "no"] {
			assert!(cvars.set("a.flag", text), "`{text}` is a word for false");
			assert_eq!(cvars.bool("a.flag"), Some(false), "`{text}` means false");
		}

		assert!(!cvars.set("a.flag", "maybe"), "and nothing else is either");
	}

	#[test]
	fn a_reload_keeps_what_was_typed_and_follows_what_was_not() {
		let mut cvars = Cvars::new();
		cvars.var("kept", Value::Float(1.0), "");
		cvars.var("followed", Value::Float(1.0), "");
		cvars.set("kept", "9.0");

		// the module comes back with different defaults for both.
		cvars.var("kept", Value::Float(2.0), "");
		cvars.var("followed", Value::Float(2.0), "");

		assert_eq!(
			cvars.float("kept"),
			Some(9.0),
			"a value someone set survives the reload, or the console is useless for the one \
			 thing it is best at"
		);
		assert_eq!(
			cvars.float("followed"),
			Some(2.0),
			"and a value nobody touched follows the code, or editing a default does nothing"
		);
	}

	#[test]
	fn a_variable_that_changed_kind_takes_the_new_default_regardless() {
		let mut cvars = Cvars::new();
		cvars.var("changed", Value::Float(1.0), "");
		cvars.set("changed", "9.0");

		cvars.var("changed", Value::Text("nine".to_owned()), "");

		assert_eq!(
			cvars.text("changed"),
			Some("nine"),
			"there is no honest way to keep a number as the text it never was"
		);
	}

	#[test]
	fn resetting_puts_a_variable_back_and_lets_the_code_lead_again() {
		let mut cvars = Cvars::new();
		cvars.var("a.number", Value::Float(1.0), "");
		cvars.set("a.number", "9.0");

		assert!(cvars.reset("a.number"), "there is something to reset");
		assert_eq!(cvars.float("a.number"), Some(1.0), "and it goes back");

		cvars.var("a.number", Value::Float(3.0), "");

		assert_eq!(
			cvars.float("a.number"),
			Some(3.0),
			"and it follows the code again afterwards, as though never touched"
		);
	}

	#[test]
	fn a_command_is_not_a_variable_and_the_other_way_round() {
		let mut cvars = Cvars::new();
		cvars.command("do.it", nothing, "");
		cvars.var("a.number", Value::Float(1.0), "");

		assert!(cvars.call_of("do.it").is_some(), "a command has something to call");
		assert!(cvars.call_of("a.number").is_none(), "a variable does not");
		assert!(!cvars.set("do.it", "3"), "and cannot be set");
		assert_eq!(cvars.get("do.it").and_then(Entry::value), None, "or read");
	}

	#[test]
	fn re_registering_a_command_replaces_the_pointer() {
		let mut cvars = Cvars::new();
		cvars.command("do.it", nothing, "");
		cvars.command("do.it", also_nothing, "");

		let call = cvars
			.call_of("do.it")
			.expect("it is still a command");
		let expected: ConsoleFn = also_nothing;

		assert!(
			std::ptr::fn_addr_eq(call, expected),
			"the new build's code is at a new address, and the old one is in an image that is \
			 about to stop existing"
		);
		assert_eq!(cvars.len(), 1, "and it replaced rather than joined");
	}

	#[test]
	fn unloading_takes_the_modules_commands_and_leaves_everything_else() {
		let mut cvars = Cvars::new();
		cvars.command("quit", nothing, "");
		cvars.var("sim.speed", Value::Float(1.0), "");

		cvars.attribute(Owner::Module);
		cvars.command("game.reset", nothing, "");
		cvars.var("game.spin", Value::Float(0.4), "");
		cvars.set("game.spin", "3.0");

		cvars.forget_module();

		assert!(
			cvars.call_of("game.reset").is_none(),
			"the module's command is gone before its library is, or typing its name is a jump \
			 into freed memory"
		);
		assert!(cvars.call_of("quit").is_some(), "the engine's is not");
		assert_eq!(
			cvars.float("game.spin"),
			Some(3.0),
			"and the module's variables stay, with what was typed into them"
		);
	}

	#[test]
	fn sweeping_drops_what_the_new_build_no_longer_registers() {
		let mut cvars = Cvars::new();
		cvars.var("sim.speed", Value::Float(1.0), "");

		cvars.attribute(Owner::Module);
		cvars.var("game.spin", Value::Float(0.4), "");
		cvars.var("game.renamed", Value::Float(1.0), "");

		// the module goes, and comes back registering only one of the two.
		cvars.forget_module();
		cvars.var("game.spin", Value::Float(0.4), "");
		cvars.sweep();

		assert!(cvars.get("game.spin").is_some(), "what came back stays");
		assert!(cvars.get("game.renamed").is_none(), "what did not was renamed or deleted");
		assert!(cvars.get("sim.speed").is_some(), "and the engine's own are never swept");
	}

	#[test]
	fn a_value_survives_being_written_down_and_read_back() {
		let mut spaced = Value::Text(String::new());
		spaced.set_text("two  words");

		assert_eq!(spaced.quoted(), "\"two  words\"", "text with spaces is quoted");
		assert_eq!(Value::Float(2.5).quoted(), "2.5", "and a number needs none of that");
		assert_eq!(Value::Bool(true).quoted(), "true", "nor a flag");
		assert_eq!(
			Value::Text(String::new()).quoted(),
			"\"\"",
			"and nothing at all is written as nothing, in quotes, or the line would lose it"
		);
	}
}
