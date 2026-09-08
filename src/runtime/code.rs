//! Opening a source in whatever editor the person at the machine uses.
//!
//! **There is no text editor in colby and this is why.** Of the six engines in
//! the clones, two wrote one and both paid ten thousand lines for it - Godot's
//! `text_edit.cpp` plus `code_edit.cpp` is 14,392, Defold's `editor/code/` is
//! 11,689 lines of Clojure - and four did not. Unreal has no text editor at
//! all and sends a file out through a plugin per IDE; s&box ships a launcher,
//! not an editor; Fyrox has a multiline text box for a shader and nothing
//! else; and **Wicked, whose gameplay language is Lua exactly as colby's is**,
//! lists `.lua` files in its content browser and hands the folder to the
//! operating system.
//!
//! The honest question was whether colby has anything an external editor
//! cannot do. The one candidate is a live error with a line to jump to, and
//! colby does have those - the interpreter names a chunk after its asset, so a
//! fault reads `scripts/thruster:3: ...`. But that is an argument for *jumping
//! to a line*, which `zed file:3` answers completely, rather than for a text
//! editor. So: no editor, and one command that opens a file.
//!
//! **The shape is the one three of the five converged on independently.** Two
//! settings - an executable and an argument template - with `{file}`, `{line}`,
//! `{col}` and `{project}` substituted into them, split into words *before* the
//! substitution so that a path with a space in it stays one word, and spawned
//! with no shell anywhere. That is Godot's `text_editor/external/exec_path` and
//! `exec_flags` (whose source carries the same note about spaces and quotes),
//! s&box's `CustomCodeEditor.ExecutablePath` and `OpenFileArgs`, and the shape
//! Unreal's `ISourceCodeAccessor::OpenFileAtLine` has one implementation of per
//! IDE.
//!
//! **Two variables rather than one, and that is not tidiness.** A single
//! variable would have to hold `"C:\Program Files\Zed\bin\Zed.exe" {file}`,
//! with the program quoted inside the value - and
//! [`Value::quoted`](colby_core::abi::cvar::Value::quoted) *strips* inner
//! quotes when it writes `settings.cfg`, so the setting would come back after a
//! restart as a program called `C:\Program`. Split in two, neither value ever
//! needs a quote inside it.
//!
//! **Nothing set is not nothing done**: an empty [`EDITOR`] hands the file to
//! the platform, which is `start` on Windows and `xdg-open` elsewhere -
//! Wicked's answer, and the one that works on a machine nobody has configured.
//! What it cannot do is carry a line number.

#[cfg(test)]
use std::path::PathBuf;
use std::{path::Path, process::Command};

use colby_asset::{Project, compile};
use colby_core::{
	Result,
	abi::{Asked, World},
	err, error, info,
};

/// `code.open <name> [line]` - opens an asset's source.
pub(crate) const OPEN: &str = "code.open";

/// `code.editor` - the program to open it with, or nothing for the platform's.
pub(crate) const EDITOR: &str = "code.editor";

/// `code.args` - what to pass it, with the placeholders substituted.
pub(crate) const ARGS: &str = "code.args";

/// What [`ARGS`] holds until somebody changes it.
///
/// Godot's own default for the same setting, and the least that can work: a
/// program handed nothing but the file opens the file.
pub(crate) const DEFAULT_ARGS: &str = "{file}";

/// The names this module answers for, as they wait on the world.
const NAMES: &[&str] = &[OPEN];

/// The line to ask for when nobody said one.
///
/// One rather than nought, which is what every editor in the field wants:
/// Unreal clamps with `LineNumber > 0 ? LineNumber : 1`, s&box defaults with
/// `line ?? 1`, Godot with `MAX(p_line + 1, 1)`. A column is not asked for
/// anywhere yet and is always this.
const FIRST: u32 = 1;

/// One file to open, as a console line asked for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Request {
	/// The asset name, `scripts/thruster`.
	name: String,

	/// Which line to put the caret on.
	line: u32,
}

impl Request {
	/// What one waiting line asks for, if it is this module's.
	///
	/// **The last word is a line when it is a number and there is more than
	/// one word.** An asset name is a path and a path may hold a space, which
	/// is why every other command in [`crate::saves`] joins all its words into
	/// one name; this one has a second argument to find, so the rule is that
	/// the *number* at the end is the line and everything in front of it is the
	/// name. A file genuinely called `scripts/take 12` is reachable as
	/// `code.open scripts/take 12 1`.
	///
	/// @param asked - a line the frame loop took off the world
	fn of(asked: &Asked) -> Option<Self> {
		if asked.name != OPEN {
			return None;
		}

		let (words, line) = match asked.words.split_last() {
			| Some((last, rest)) if !rest.is_empty() => match last.parse::<u32>() {
				| Ok(line) => (rest, line.max(FIRST)),
				| Err(_) => (asked.words.as_slice(), FIRST),
			},
			| _ => (asked.words.as_slice(), FIRST),
		};

		Some(Self { name: words.join(" "), line })
	}
}

/// Opens whatever a command asked for, if one did.
///
/// The frame loop's, beside [`crate::saves::serve`] and for the same reason: a
/// [`ConsoleFn`](colby_core::abi::ConsoleFn) is handed a world and nothing
/// else, and finding a file needs the project.
///
/// **One at a time, and the last one.** Two opens typed before the next frame
/// means the second is what was meant, exactly as a scene load does.
///
/// @param world - where the lines wait
/// @param project - whose asset tree
pub(crate) fn serve(world: &mut World, project: &Project) {
	let Some(request) = crate::console::take(world, NAMES)
		.pop()
		.as_ref()
		.and_then(Request::of)
	else {
		return;
	};

	let editor = world
		.cvars
		.text(EDITOR)
		.unwrap_or_default()
		.to_owned();
	let args = world
		.cvars
		.text(ARGS)
		.unwrap_or(DEFAULT_ARGS)
		.to_owned();

	if let Err(failure) = open(project, &request, &editor, &args) {
		error!(%failure, "the file could not be opened");
	}
}

/// Finds the source a name compiles from and starts something on it.
///
/// @param project - whose asset tree
/// @param request - the name and the line
/// @param editor - what [`EDITOR`] holds
/// @param args - what [`ARGS`] holds
///
/// # Errors
///
/// If no source under that name is in the tree, or if the program will not
/// start.
fn open(project: &Project, request: &Request, editor: &str, args: &str) -> Result {
	let Some(source) = compile::source_of(&project.assets(), &request.name) else {
		return Err(err!(Asset(
			"nothing under assets/ compiles to {}, so there is no file to open",
			request.name
		)));
	};

	if editor.trim().is_empty() {
		platform(&source)?;
		info!(path = %source.display(), name = request.name, "opened by the platform");

		return Ok(());
	}

	let words = arguments(args, &source, request.line, project.root());

	Command::new(editor)
		.args(&words)
		.spawn()
		.map_err(|error| err!(Err("{editor} could not be started: {error}")))?;

	info!(
		path = %source.display(),
		name = request.name,
		line = request.line,
		editor,
		"opened in the editor",
	);

	Ok(())
}

/// The template as a list of arguments, with the placeholders filled in.
///
/// **Split first, substitute second**, which is the whole of why nothing here
/// quotes anything: a path holding a space arrives *after* the words were
/// decided, so it cannot become two of them. Godot's implementation of this
/// same setting carries the same note, and it is the one thing about the
/// arrangement that is not obvious.
///
/// [`console::statements`](colby_core::abi::console::statements) does the
/// splitting, so a template obeys the quoting rule the rest of the console
/// does; a template holding several statements is flattened, because a
/// program is started once however many semicolons somebody typed.
///
/// **A template with no `{file}` in it gets the path appended**, which is
/// Godot's rule as well: the commonest mistake is naming the flags and
/// forgetting the file, and the result of not doing this is an editor that
/// opens on nothing.
///
/// @param template - what [`ARGS`] holds
/// @param file - the source to open
/// @param line - the line to put the caret on
/// @param project - the project's directory, for `{project}`
#[expect(
	clippy::literal_string_with_formatting_args,
	reason = "these are the placeholders of somebody else's template language, written \n	          the way Godot and s&box write theirs, and the lint cannot tell one brace \n	          from another"
)]
fn arguments(template: &str, file: &Path, line: u32, project: &Path) -> Vec<String> {
	let file = file.display().to_string();
	let project = project.display().to_string();
	let mut words: Vec<String> = colby_core::abi::console::statements(template)
		.into_iter()
		.flatten()
		.map(|word| {
			word.replace("{file}", &file)
				.replace("{line}", &line.to_string())
				.replace("{col}", &FIRST.to_string())
				.replace("{project}", &project)
		})
		.collect();

	if !template.contains("{file}") {
		words.push(file);
	}

	words
}

/// What `cmd` is handed to open a file with whatever it is associated with.
///
/// **Written out rather than assembled from arguments, and that is not an
/// optimization.** `cmd.exe` parses its own command line rather than taking the
/// arguments a caller separated, and a path holding `&` - which is a legal file
/// name - would be passed through by [`Command::arg`] unquoted, because it
/// holds no space, and then read by `cmd` as the end of the command. So the
/// quoting is written here, and it is safe to write: a double quote is the one
/// character Windows forbids in a file name, so nothing can close it.
///
/// The empty `""` after `start` is its title argument, which it takes when the
/// first quoted word would otherwise be read as one.
#[cfg(windows)]
fn platform_line(file: &Path) -> String { format!("/c start \"\" \"{}\"", file.display()) }

/// Hands a file to whatever this platform opens it with.
#[cfg(windows)]
fn platform(file: &Path) -> Result {
	use std::os::windows::process::CommandExt as _;

	Command::new("cmd")
		.raw_arg(platform_line(file))
		.spawn()
		.map_err(|error| err!(Err("{} could not be opened: {error}", file.display())))?;

	Ok(())
}

/// Hands a file to whatever this platform opens it with.
///
/// No shell and so no quoting: `xdg-open` is handed one argument.
#[cfg(not(windows))]
fn platform(file: &Path) -> Result {
	Command::new("xdg-open")
		.arg(file)
		.spawn()
		.map_err(|error| err!(Err("{} could not be opened: {error}", file.display())))?;

	Ok(())
}

/// Where a name's source is, without going through the world.
///
/// The tests' way in, and the only thing here that is not the console's - the
/// same shape [`crate::saves`] keeps its reader in.
#[cfg(test)]
fn source(project: &Project, name: &str) -> Option<PathBuf> {
	compile::source_of(&project.assets(), name)
}

#[cfg(test)]
mod tests {
	use std::{env, fs};

	use colby_core::abi::{Aim, PeerId, cvar::Value};

	use super::*;

	/// A program every machine has that copies a file, so that a run leaves
	/// something to look at.
	const fn copier() -> &'static str { if cfg!(windows) { "cmd" } else { "cp" } }

	/// What it wants before the two paths.
	const fn copy_args() -> &'static str { if cfg!(windows) { "/c copy /y" } else { "--" } }

	/// What a started child left behind, waiting for it to finish writing.
	///
	/// **Nothing here waits on the process**: an editor is started and left
	/// running, which is the whole point, so the only thing a test can do is
	/// watch for what the child leaves. And watching for the file to *exist* is
	/// not enough - `copy` creates it and then writes it, and a read between
	/// the two fails with "used by another process", which is a flake that
	/// would have turned up on somebody else's machine rather than here. So
	/// what is waited for is a read that succeeds and returns something.
	///
	/// A second is far longer than a copy takes and short enough that a failure
	/// is a failure rather than a hang.
	fn waited(path: &Path) -> Option<String> {
		for _ in 0..100 {
			match fs::read_to_string(path) {
				| Ok(text) if !text.trim().is_empty() => return Some(text),
				| _ => std::thread::sleep(std::time::Duration::from_millis(10)),
			}
		}

		None
	}

	/// A line as the console would have left it.
	fn asked(name: &str, words: &[&str]) -> Asked {
		Asked {
			name: name.to_owned(),
			words: words
				.iter()
				.map(|word| (*word).to_owned())
				.collect(),
			peer: PeerId::HOST,
			aim: Aim::NONE,
		}
	}

	/// A scratch project with one program in it.
	fn fresh(name: &str) -> Project {
		let root = env::temp_dir().join(format!("colby_code_{name}"));
		drop(fs::remove_dir_all(&root));
		fs::create_dir_all(root.join("assets").join("scripts")).expect("a tree to work in");
		fs::write(
			root.join("assets")
				.join("scripts")
				.join("hello.lua"),
			b"function tick(dt) end\n",
		)
		.expect("a program");

		Project::parse(
			&root,
			r#"{ "schema": 1, "engine": "0.1.0", "id": "coded", "name": "Coded" }"#,
		)
		.expect("a project")
	}

	#[test]
	fn a_line_with_only_a_name_opens_the_first_line() {
		let request = Request::of(&asked(OPEN, &["scripts/hello"])).expect("this one is ours");

		assert_eq!(request.name, "scripts/hello");
		assert_eq!(request.line, FIRST, "nobody said a line");
	}

	#[test]
	fn a_number_at_the_end_is_the_line_and_the_rest_is_the_name() {
		let request = Request::of(&asked(OPEN, &["scripts/hello", "12"])).expect("ours");

		assert_eq!(request.name, "scripts/hello");
		assert_eq!(request.line, 12);
	}

	#[test]
	fn a_name_that_is_only_a_number_is_a_name() {
		// one word, so there is nothing in front of the number for it to be
		// the line of, and an asset really called `12` is reachable.
		let request = Request::of(&asked(OPEN, &["12"])).expect("ours");

		assert_eq!(request.name, "12");
		assert_eq!(request.line, FIRST);
	}

	#[test]
	fn a_name_with_a_space_in_it_survives_when_the_last_word_is_not_a_number() {
		let request = Request::of(&asked(OPEN, &["scripts/take", "two"])).expect("ours");

		assert_eq!(request.name, "scripts/take two");
		assert_eq!(request.line, FIRST);
	}

	#[test]
	fn a_line_of_nought_is_the_first_line() {
		// every editor in the field clamps this, because a caret on line
		// nought is a caret nowhere.
		let request = Request::of(&asked(OPEN, &["scripts/hello", "0"])).expect("ours");

		assert_eq!(request.line, FIRST);
	}

	#[test]
	fn a_line_asked_under_another_name_is_not_this_modules() {
		assert!(Request::of(&asked("scene.write", &["scenes/yard"])).is_none());
	}

	#[test]
	fn the_placeholders_are_filled_in_and_a_path_with_a_space_stays_one_word() {
		let words = arguments(
			"{project} {file}:{line}:{col}",
			Path::new("C:/two words/hello.lua"),
			12,
			Path::new("C:/two words"),
		);

		assert_eq!(words.len(), 2, "two arguments, whatever the paths hold");
		assert_eq!(words[0], "C:/two words");
		assert_eq!(words[1], "C:/two words/hello.lua:12:1");
	}

	#[test]
	fn a_template_that_names_no_file_gets_the_path_after_it() {
		// the commonest way to write one of these wrong, and the one an editor
		// opening on nothing does not explain.
		let words = arguments("--wait", Path::new("/tmp/hello.lua"), 1, Path::new("/tmp"));

		assert_eq!(words, vec!["--wait".to_owned(), "/tmp/hello.lua".to_owned()]);
	}

	#[test]
	fn an_empty_template_is_the_path_alone() {
		let words = arguments("", Path::new("/tmp/hello.lua"), 1, Path::new("/tmp"));

		assert_eq!(words, vec!["/tmp/hello.lua".to_owned()]);
	}

	#[test]
	fn a_quoted_word_in_a_template_stays_one_word() {
		let words = arguments(
			"\"+call cursor({line}, {col})\" {file}",
			Path::new("/tmp/hello.lua"),
			7,
			Path::new("/tmp"),
		);

		assert_eq!(words, vec!["+call cursor(7, 1)".to_owned(), "/tmp/hello.lua".to_owned()]);
	}

	#[test]
	fn a_name_in_the_tree_is_found_and_one_that_climbs_out_is_not() {
		let project = fresh("finding");

		assert_eq!(
			source(&project, "scripts/hello"),
			Some(project.assets().join("scripts").join("hello.lua"))
		);
		assert_eq!(source(&project, "scripts/missing"), None);
		assert_eq!(
			source(&project, "../../../windows/system32/calc"),
			None,
			"a name may not leave the tree"
		);
		assert_eq!(source(&project, ""), None, "and no name is no file");
	}

	#[test]
	fn a_name_nothing_compiles_from_is_refused_by_name_rather_than_started() {
		let project = fresh("refusing");
		let request = Request {
			name: "scripts/nowhere".to_owned(),
			line: 1,
		};

		let error =
			open(&project, &request, "", DEFAULT_ARGS).expect_err("there is no such source");

		assert!(
			error.to_string().contains("scripts/nowhere"),
			"saying which name it was: {error}"
		);
	}

	#[test]
	fn both_variables_survive_being_written_to_a_config_and_read_back() {
		// **the whole reason there are two of these rather than one**, and the
		// fault it avoids is silent: `Value::quoted` wraps a value holding a
		// space in quotes and *strips* the quotes that were inside it, so a
		// single variable holding `"C:\Program Files\..\Zed.exe" {file}` would
		// come back after one restart as a program called `C:\Program`. Split
		// in two, neither value ever needs a quote inside it, and this is what
		// a project's `settings.cfg` does to them both.
		let program = r"C:\Program Files\Zed\bin\Zed.exe";
		let template = "{project} {file}:{line}:{col}";
		let mut world = World::new();

		crate::console::install(&mut world);
		world.cvars.set(EDITOR, program);
		world.cvars.set(ARGS, template);

		// what `Console` writes into the file, and what it runs back at the
		// table on the next start
		let written = format!(
			"{EDITOR} {}\n{ARGS} {}\n",
			world
				.cvars
				.get(EDITOR)
				.and_then(colby_core::abi::cvar::Entry::value)
				.expect("it is a variable")
				.quoted(),
			world
				.cvars
				.get(ARGS)
				.and_then(colby_core::abi::cvar::Entry::value)
				.expect("it is a variable")
				.quoted(),
		);
		let mut read = World::new();
		crate::console::install(&mut read);
		colby_core::abi::console::run(&mut read, &written);

		assert_eq!(read.cvars.text(EDITOR), Some(program), "the program came back whole");
		assert_eq!(read.cvars.text(ARGS), Some(template), "and so did the template");

		// and it still means what it meant: the program with a space in it is
		// one word, and so is each argument
		let words = arguments(
			read.cvars.text(ARGS).expect("it is text"),
			Path::new(r"C:\two words\hello.lua"),
			12,
			Path::new(r"C:\two words"),
		);

		assert_eq!(words, vec![
			r"C:\two words".to_owned(),
			r"C:\two words\hello.lua:12:1".to_owned(),
		]);
	}

	#[test]
	fn one_variable_holding_both_would_come_back_broken() {
		// **the negative control for the test above, and the measurement the
		// design rests on.** One variable holding the program and the template
		// together has to quote the program inside the value; `quoted` strips
		// those quotes on the way into the file, and what comes back is a
		// program called `C:\Program`. Written down here because the failure
		// is silent - it would look like an editor that stopped working after
		// a restart, with a config file that reads almost right.
		let together = r#""C:\Program Files\Zed\bin\Zed.exe" {file}"#;
		let mut world = World::new();

		world
			.cvars
			.var("code.together", Value::Text(String::new()), "");
		world.cvars.set("code.together", together);

		let written = format!(
			"code.together {}",
			world
				.cvars
				.get("code.together")
				.and_then(colby_core::abi::cvar::Entry::value)
				.expect("it is a variable")
				.quoted()
		);
		let mut read = World::new();
		read.cvars
			.var("code.together", Value::Text(String::new()), "");
		colby_core::abi::console::run(&mut read, &written);

		let back = read
			.cvars
			.text("code.together")
			.expect("it is text");

		assert_ne!(back, together, "the quotes inside it did not survive the file");
		assert_eq!(
			colby_core::abi::console::statements(back)
				.into_iter()
				.flatten()
				.next(),
			Some(r"C:\Program".to_owned()),
			"and the program is now the first word of a path"
		);
	}

	#[test]
	fn a_line_typed_at_the_console_starts_a_program_on_the_file() {
		// **the whole chain, with a real process at the end of it**: the
		// command registered, the line run, the frame loop taking it up, the
		// name resolved to a file and something started on it. The editor is a
		// copier, because a copier is a program every machine has and its
		// having run leaves a file to look at - what is being proved is that
		// the arguments arrived, not what an editor would do with them.
		let project = fresh("driven");
		let proof = project.root().join("proof.txt");
		let mut world = World::new();

		crate::console::install(&mut world);
		world.cvars.set(EDITOR, copier());
		world
			.cvars
			.set(ARGS, &format!("{} {{file}} {}", copy_args(), proof.display()));
		colby_core::abi::console::run(&mut world, "code.open scripts/hello");

		assert_eq!(world.asked.len(), 1, "the line waited rather than acting inside itself");

		serve(&mut world, &project);

		assert!(world.asked.is_empty(), "and the frame loop took it");

		let copied = waited(&proof)
			.unwrap_or_else(|| panic!("the copier ran on the file: {}", proof.display()));

		assert_eq!(copied.trim(), "function tick(dt) end", "and on the right one");
	}

	#[test]
	fn a_name_nothing_is_behind_starts_nothing_at_all() {
		// the negative control for the run above, with one word changed: the
		// same command, the same editor, the same arguments, a name with no
		// file behind it - and no proof appears. Without this the run above
		// would pass just as well against a copier that had been started by
		// something else.
		let project = fresh("driven_nothing");
		let proof = project.root().join("proof.txt");
		let mut world = World::new();

		crate::console::install(&mut world);
		world.cvars.set(EDITOR, copier());
		world
			.cvars
			.set(ARGS, &format!("{} {{file}} {}", copy_args(), proof.display()));
		colby_core::abi::console::run(&mut world, "code.open scripts/nowhere");

		serve(&mut world, &project);

		assert!(world.asked.is_empty(), "the line was still taken");
		assert!(waited(&proof).is_none(), "and nothing was started");
	}

	#[test]
	#[cfg(windows)]
	fn the_platform_line_quotes_a_path_that_cmd_would_otherwise_cut_in_half() {
		// **the whole reason this line is written out rather than passed as
		// arguments.** `&` is legal in a Windows file name and holds no space,
		// so `Command::arg` would hand it over bare and `cmd` would read it as
		// the end of the command - opening one half and trying to run the
		// other half as a program.
		//
		// **Nothing is started here, and that is deliberate.** The run itself
		// would open whatever this person associated the extension with, and a
		// test suite that opens an editor window is a suite nobody runs twice.
		// So the text is checked and the one line that spawns is not; the live
		// drive is where somebody watches it open.
		assert_eq!(
			platform_line(Path::new(r"C:\a&b\x.lua")),
			r#"/c start "" "C:\a&b\x.lua""#,
			"quoted whole, so cmd reads one argument rather than two commands"
		);
	}

	#[test]
	fn an_editor_that_is_not_there_is_a_failure_and_not_a_silence() {
		let project = fresh("missing_editor");
		let request = Request {
			name: "scripts/hello".to_owned(),
			line: 1,
		};

		let error = open(&project, &request, "colby-no-such-editor", DEFAULT_ARGS)
			.expect_err("nothing by that name is on the path");

		assert!(
			error.to_string().contains("colby-no-such-editor"),
			"naming what would not start: {error}"
		);
	}
}
