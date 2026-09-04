//! A project: a directory named by one file at its root.
//!
//! `colby.project` is the marker. Where it is, everything the engine reads and
//! writes for a project hangs off that directory: sources under `assets/`, what
//! they compile to under `.colby/assets/`, saved worlds under `saves/`, the
//! console's archive in `settings.cfg`, and the game crate under a directory
//! the file names. The file is strict JSON in the shape a `.scene` source is,
//! and a field this build does not know is refused rather than skipped, for the
//! reason [`level`] gives and one more: a project file is the one thing a newer
//! engine writes and an older one reads, and the older one has to say so.
//!
//! **The directory is the identity, and the id is its name.** No GUID: an id is
//! a short lowercase word, `[a-z0-9_]{2,32}`, which is the rule a package cloud
//! puts on a project ident and also what a directory name and a crate name both
//! accept without escaping. It names the game module (`<id>_game`) and the
//! mount point an engine builds that crate under. The engine version in the
//! file is the one that wrote it, and a different one at open is a warning: the
//! gate that matters is the ABI number a module reports when it is loaded.
//!
//! ```text
//! {
//!   "schema": 1,
//!   "engine": "0.1.0",
//!   "id": "colby",
//!   "name": "colby",
//!   "game": "src/game",
//!   "startup_scene": "scenes/construct"
//! }
//! ```

use std::{
	ffi::OsStr,
	fs,
	path::{Component, Path, PathBuf},
};

use colby_core::{Result, err, warn};

use crate::{
	compile,
	json::{self, Value},
	level,
};

/// The file that names a directory as a project.
pub const FILE: &str = "colby.project";

/// The shape of the file this build writes and reads.
pub const SCHEMA: u32 = 1;

/// Where saved worlds go, relative to the project root.
pub const SAVES_DIR: &str = "saves";

/// The console's archive, relative to the project root.
pub const SETTINGS_FILE: &str = "settings.cfg";

/// The directory under an engine checkout that projects are mounted in, so
/// that their game crates are built as members of the engine's workspace.
pub const MOUNTS_DIR: &str = "projects";

/// The directory a mounted project's game crate has to be in.
pub const GAME_DIR: &str = "game";

/// The fewest characters an id may have.
const ID_SHORTEST: usize = 2;

/// The most characters an id may have.
const ID_LONGEST: usize = 32;

/// The engine version this build reports, and writes into a new project.
pub const ENGINE: &str = env!("CARGO_PKG_VERSION");

/// Every field a project file may carry.
const FIELDS: &[&str] = &["schema", "engine", "id", "name", "game", "startup_scene"];

/// A project, as its file describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Project {
	root: PathBuf,
	id: String,
	name: String,
	engine: String,
	game: Option<PathBuf>,
	startup_scene: Option<String>,
}

impl Project {
	/// Opens the project in a directory.
	///
	/// @param dir - the directory holding [`FILE`]
	///
	/// # Errors
	///
	/// If there is no project file there, or the one there is not one this
	/// build reads. @ref [`parse`](Self::parse) for what is checked.
	pub fn open(dir: &Path) -> Result<Self> {
		let path = dir.join(FILE);
		let text = fs::read_to_string(&path).map_err(|error| {
			err!(Asset(
				"no project at {}: {error}; pass --project <dir>, or start from a directory \
				 holding {FILE}",
				dir.display()
			))
		})?;
		let project = Self::parse(dir, &text)?;

		if project.engine != ENGINE {
			warn!(
				project = project.id,
				written_by = project.engine,
				running = ENGINE,
				"this project was written by another version of the engine"
			);
		}

		if let Some(folder) = dir.file_name().and_then(OsStr::to_str)
			&& folder != project.id
		{
			warn!(
				folder,
				id = project.id,
				"the project's directory is not called what the project is; the mount point and \
				 the module take the id"
			);
		}

		Ok(project)
	}

	/// Reads a project file's text.
	///
	/// Split from [`open`](Self::open) so that the rules can be tested without
	/// a filesystem.
	///
	/// @param root - the directory the file is in; everything is relative to it
	/// @param text - the file
	///
	/// # Errors
	///
	/// If a field is missing, unknown or not what it has to be: a schema this
	/// build does not read, an id that is not a short lowercase word, a name
	/// that is empty, a game directory outside the project.
	pub fn parse(root: &Path, text: &str) -> Result<Self> {
		let value = json::parse(text)?;
		level::fields(&value, FIELDS, "a project")?;

		let schema = value
			.get("schema")
			.and_then(Value::as_u32)
			.ok_or_else(|| err!(Asset("a project needs a schema number")))?;

		if schema > SCHEMA {
			return Err(err!(Asset(
				"this project was written by a newer engine: schema {schema}, and this build \
				 reads {SCHEMA}"
			)));
		}

		if schema == 0 {
			return Err(err!(Asset("schema 0 is not one any engine has written")));
		}

		let engine = required(&value, "engine")?.to_owned();
		let id = required(&value, "id")?.to_owned();

		if !valid_id(&id) {
			return Err(err!(Asset(
				"{id:?} is not an id a project can have: two to thirty-two of a-z, 0-9 and _"
			)));
		}

		let name = required(&value, "name")?.to_owned();
		let game = optional(&value, "game")?
			.map(inside)
			.transpose()?;
		let startup_scene = optional(&value, "startup_scene")?.map(str::to_owned);

		Ok(Self {
			root: root.to_owned(),
			id,
			name,
			engine,
			game,
			startup_scene,
		})
	}

	/// The directory everything is relative to.
	#[must_use]
	pub fn root(&self) -> &Path { &self.root }

	/// The short lowercase word the project goes by.
	#[must_use]
	pub fn id(&self) -> &str { &self.id }

	/// What a person calls it.
	#[must_use]
	pub fn name(&self) -> &str { &self.name }

	/// The engine version that wrote the file.
	#[must_use]
	pub fn engine(&self) -> &str { &self.engine }

	/// Where the sources are.
	#[must_use]
	pub fn assets(&self) -> PathBuf { compile::source_root(&self.root) }

	/// Where the sources compile to, and what the engine loads from.
	#[must_use]
	pub fn output(&self) -> PathBuf { compile::output_root(&self.root) }

	/// Where saved worlds go.
	#[must_use]
	pub fn saves(&self) -> PathBuf { self.root.join(SAVES_DIR) }

	/// Where the console keeps its archived variables.
	#[must_use]
	pub fn settings(&self) -> PathBuf { self.root.join(SETTINGS_FILE) }

	/// Where the game crate is, if the project has one.
	#[must_use]
	pub fn game(&self) -> Option<PathBuf> {
		self.game
			.as_ref()
			.map(|relative| self.root.join(relative))
	}

	/// Where the game crate is, relative to the root, as the file said it.
	#[must_use]
	pub fn game_dir(&self) -> Option<&Path> { self.game.as_deref() }

	/// The name of the module the game crate builds: `<id>_game`.
	///
	/// Derived rather than read from the crate's manifest, so that two
	/// projects can never share a module image and nothing has to parse TOML.
	/// A crate called anything else is a module the engine cannot find, and
	/// the message says which file it looked for.
	#[must_use]
	pub fn module(&self) -> String { format!("{}_game", self.id) }

	/// The scene the world starts as, if the file names one.
	#[must_use]
	pub fn startup_scene(&self) -> Option<&str> { self.startup_scene.as_deref() }

	/// Where an engine mounts this project to build its game crate.
	///
	/// @param engine - the engine checkout
	#[must_use]
	pub fn mount(&self, engine: &Path) -> PathBuf { engine.join(MOUNTS_DIR).join(&self.id) }
}

/// Whether a word may be a project's id.
///
/// Two to thirty-two of `a-z`, `0-9` and `_`: what a directory, a crate and a
/// package cloud all accept without escaping.
#[must_use]
pub fn valid_id(id: &str) -> bool {
	(ID_SHORTEST..=ID_LONGEST).contains(&id.len())
		&& id
			.bytes()
			.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

/// A text field that has to be there and has to say something.
fn required<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
	optional(value, field)?.ok_or_else(|| err!(Asset("a project needs a {field}")))
}

/// A text field that may be absent, but may not be empty or something else.
fn optional<'a>(value: &'a Value, field: &str) -> Result<Option<&'a str>> {
	let Some(found) = value.get(field) else {
		return Ok(None);
	};

	match found.as_str() {
		| Some(text) if !text.is_empty() => Ok(Some(text)),
		| Some(_) => Err(err!(Asset("a project's {field} cannot be empty"))),
		| None => Err(err!(Asset("a project's {field} has to be text"))),
	}
}

/// A path that stays inside the project.
///
/// Relative, and without a step upwards: a game crate is part of the project
/// or it is not the project's game crate.
fn inside(text: &str) -> Result<PathBuf> {
	let path = Path::new(text);
	let escapes = path.components().any(|part| {
		matches!(part, Component::ParentDir | Component::RootDir | Component::Prefix(_))
	});

	if escapes {
		return Err(err!(Asset(
			"{text:?} is outside the project; a game directory is inside it"
		)));
	}

	Ok(path.to_owned())
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The file the checkout carries, more or less.
	const WHOLE: &str = r#"{
		"schema": 1,
		"engine": "0.1.0",
		"id": "colby",
		"name": "colby",
		"game": "src/game",
		"startup_scene": "scenes/construct"
	}"#;

	/// What an error says.
	fn refused(text: &str) -> String {
		Project::parse(Path::new("C:/somewhere/colby"), text)
			.expect_err("it is refused")
			.to_string()
	}

	#[test]
	fn a_project_file_is_read_with_everything_named() {
		let root = Path::new("C:/somewhere/colby");
		let project = Project::parse(root, WHOLE).expect("a whole file");

		assert_eq!(project.id(), "colby");
		assert_eq!(project.name(), "colby");
		assert_eq!(project.engine(), "0.1.0");
		assert_eq!(project.startup_scene(), Some("scenes/construct"));
		assert_eq!(project.root(), root);
		assert_eq!(project.assets(), root.join("assets"));
		assert_eq!(project.output(), root.join(".colby").join("assets"));
		assert_eq!(project.saves(), root.join("saves"));
		assert_eq!(project.settings(), root.join("settings.cfg"));
		assert_eq!(project.game(), Some(root.join("src").join("game")));
		assert_eq!(project.game_dir(), Some(Path::new("src/game")));
		assert_eq!(project.module(), "colby_game", "the module is the id and a suffix");
		assert_eq!(
			project.mount(Path::new("C:/engine")),
			Path::new("C:/engine")
				.join("projects")
				.join("colby")
		);
	}

	#[test]
	fn the_two_optional_fields_may_be_absent() {
		let project = Project::parse(
			Path::new("."),
			r#"{ "schema": 1, "engine": "0.1.0", "id": "blank", "name": "Blank" }"#,
		)
		.expect("the least a project can say");

		assert_eq!(project.game(), None, "a project with no game crate runs on Lua alone");
		assert_eq!(project.startup_scene(), None, "and starts with whatever init makes");
		assert_eq!(project.module(), "blank_game");
	}

	#[test]
	fn a_field_this_build_does_not_know_is_refused() {
		// the `.scene` rule, and the reason a newer engine's field does not
		// silently do nothing in an older one.
		let text = refused(
			r#"{ "schema": 1, "engine": "0.1.0", "id": "x2", "name": "x", "thumbs": ".t" }"#,
		);

		assert!(text.contains("thumbs"), "it says which: {text}");
	}

	#[test]
	fn a_schema_this_build_does_not_read_is_refused_and_says_so() {
		let newer = refused(r#"{ "schema": 2, "engine": "9.0.0", "id": "later", "name": "x" }"#);

		assert!(newer.contains("newer engine"), "got {newer}");
		assert!(newer.contains("schema 2"), "and which: {newer}");

		let none = refused(r#"{ "engine": "0.1.0", "id": "later", "name": "x" }"#);

		assert!(none.contains("schema"), "a file with no schema is not one: {none}");

		let nought = refused(r#"{ "schema": 0, "engine": "0.1.0", "id": "later", "name": "x" }"#);

		assert!(nought.contains("schema 0"), "got {nought}");
	}

	#[test]
	fn an_id_is_a_short_lowercase_word() {
		for good in ["colby", "a1", "with_under_score", "x".repeat(32).as_str()] {
			assert!(valid_id(good), "{good} is an id");
		}

		for bad in
			["", "x", "Colby", "with-dash", "with space", "x".repeat(33).as_str(), "юникод"]
		{
			assert!(!valid_id(bad), "{bad:?} is not");
		}

		let text =
			refused(r#"{ "schema": 1, "engine": "0.1.0", "id": "Not Valid", "name": "x" }"#);

		assert!(text.contains("Not Valid"), "the refusal names it: {text}");
	}

	#[test]
	fn a_name_and_an_engine_have_to_be_there_and_say_something() {
		assert!(
			refused(r#"{ "schema": 1, "engine": "0.1.0", "id": "ok" }"#).contains("name"),
			"no name"
		);
		assert!(
			refused(r#"{ "schema": 1, "engine": "0.1.0", "id": "ok", "name": "" }"#)
				.contains("cannot be empty"),
			"an empty one"
		);
		assert!(
			refused(r#"{ "schema": 1, "id": "ok", "name": "x" }"#).contains("engine"),
			"no engine"
		);
		assert!(
			refused(r#"{ "schema": 1, "engine": 1, "id": "ok", "name": "x" }"#)
				.contains("has to be text"),
			"a number where text goes"
		);
	}

	#[test]
	fn a_game_directory_stays_inside_the_project() {
		for outside in ["../elsewhere", "C:/absolute/game", "/rooted"] {
			let text = refused(&format!(
				r#"{{ "schema": 1, "engine": "0.1.0", "id": "ok", "name": "x", "game": "{outside}" }}"#
			));

			assert!(text.contains("outside the project"), "{outside}: {text}");
		}

		let inside = Project::parse(
			Path::new("."),
			r#"{ "schema": 1, "engine": "0.1.0", "id": "ok", "name": "x", "game": "game" }"#,
		)
		.expect("the usual place");

		assert_eq!(inside.game(), Some(Path::new(".").join("game")));
	}

	#[test]
	fn a_missing_file_says_where_it_looked_and_what_to_do() {
		let dir = std::env::temp_dir().join("colby_project_nothing_here");
		let text = Project::open(&dir)
			.expect_err("there is no project there")
			.to_string();

		assert!(text.contains("colby_project_nothing_here"), "where: {text}");
		assert!(text.contains("--project"), "and what to do: {text}");
	}

	#[test]
	fn a_file_on_disk_opens_whatever_its_directory_is_called() {
		// a project cloned under another name is still that project: the id
		// decides the module and the mount point, and the mismatch is a warning.
		let dir = std::env::temp_dir().join("colby_project_renamed_clone");
		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(&dir).expect("a directory to put it in");
		fs::write(dir.join(FILE), WHOLE).expect("the file");

		let project = Project::open(&dir).expect("it opens");

		assert_eq!(project.id(), "colby");
		assert_eq!(project.root(), dir);

		drop(fs::remove_dir_all(&dir));
	}
}
