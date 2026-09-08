//! The projects a person has, as one small file of their own.
//!
//! Paths and two flags, and nothing that is already on disk: a project's name
//! and id live in its `colby.project` and are read from there every time the
//! list is shown, so a project renamed by hand is renamed here the same
//! moment and nothing goes stale. What the file keeps is what the disk cannot
//! know - which projects this person cares about, which are pinned, and when
//! each was last opened. That is the shape the field settled on: a tiny list
//! re-validated against disk on every load, never a database.
//!
//! **Per user, not per checkout.** The file sits in the person's own config
//! directory rather than beside the engine, so two checkouts of the engine
//! show the same projects, and a project's own version is what the row says
//! about it. Strict JSON in the shape a project file is, an unknown field
//! refused for the reason [`colby_asset::project`] gives.
//!
//! ```text
//! {
//!   "schema": 1,
//!   "projects_dir": "C:/Users/somebody/Documents/colby projects",
//!   "sort": "recent",
//!   "projects": [
//!     { "path": "C:/Users/somebody/Documents/colby projects/yard", "pinned": true, "last_opened": 1788000000 }
//!   ]
//! }
//! ```

use std::{
	env,
	fmt::Write as _,
	fs,
	path::{Path, PathBuf},
	time::{SystemTime, UNIX_EPOCH},
};

use colby_asset::{
	Project,
	json::{self, Value},
	project,
};
use colby_core::{Result, err, utils::path::lexical};

/// The file's name, under the per-user directory.
pub const FILE: &str = "projects.json";

/// The shape of the file this build writes and reads.
pub const SCHEMA: u32 = 1;

/// The per-user directory everything of colby's that is not a project's goes
/// in, under the platform's config directory.
pub const DIRECTORY: &str = "colby";

/// The folder new projects are made in unless somebody says otherwise, under
/// the person's documents.
pub const PROJECTS_FOLDER: &str = "colby projects";

/// Every field the file may carry.
const FIELDS: &[&str] = &["schema", "projects_dir", "sort", "projects"];

/// Every field an entry may carry.
const ENTRY_FIELDS: &[&str] = &["path", "pinned", "last_opened"];

/// How the list is ordered on screen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sort {
	/// The most recently opened first.
	#[default]
	Recent,

	/// By name.
	Name,
}

impl Sort {
	/// Every order, in the order the menu lists them.
	pub const ALL: [Self; 2] = [Self::Recent, Self::Name];

	/// The word the file and the menu use.
	#[must_use]
	pub const fn word(self) -> &'static str {
		match self {
			| Self::Recent => "recent",
			| Self::Name => "name",
		}
	}

	/// The order a word names, if it names one.
	#[must_use]
	pub fn from_word(word: &str) -> Option<Self> {
		Self::ALL
			.into_iter()
			.find(|sort| sort.word() == word)
	}
}

/// One project on the list: where it is, and the two things the disk cannot
/// know about it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
	/// The directory holding its `colby.project`.
	pub path: PathBuf,

	/// Whether it is shown at the top.
	pub pinned: bool,

	/// When it was last opened from here, in seconds since the epoch; nought
	/// for never.
	pub last_opened: u64,
}

/// The list, as the file describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct List {
	/// Where the wizard makes a project unless told otherwise.
	pub projects_dir: PathBuf,

	/// How the list is ordered on screen.
	pub sort: Sort,

	entries: Vec<Entry>,
}

impl List {
	/// An empty list, making projects under a folder.
	///
	/// @param projects_dir - the folder new projects go in
	#[must_use]
	pub fn new(projects_dir: &Path) -> Self {
		Self {
			projects_dir: projects_dir.to_owned(),
			sort: Sort::default(),
			entries: Vec::new(),
		}
	}

	/// Reads the file's text.
	///
	/// @param text - the file
	/// @param fallback_dir - the folder new projects go in when the file
	/// names none
	///
	/// # Errors
	///
	/// If a field is unknown or not what it has to be, or the schema is one
	/// this build does not read.
	pub fn read(text: &str, fallback_dir: &Path) -> Result<Self> {
		let value = json::parse(text)?;
		known(&value, FIELDS, "the project list")?;

		let schema = value
			.get("schema")
			.and_then(Value::as_u32)
			.ok_or_else(|| err!(Asset("the project list needs a schema number")))?;

		if schema > SCHEMA {
			return Err(err!(Asset(
				"the project list was written by a newer engine: schema {schema}, and this \
				 build reads {SCHEMA}"
			)));
		}

		let projects_dir = match value.get("projects_dir") {
			| Some(dir) =>
				PathBuf::from(dir.as_str().ok_or_else(|| {
					err!(Asset("the project list's projects_dir has to be text"))
				})?),
			| None => fallback_dir.to_owned(),
		};

		let sort = match value.get("sort") {
			| Some(word) => {
				let word = word
					.as_str()
					.ok_or_else(|| err!(Asset("the project list's sort has to be a word")))?;

				Sort::from_word(word)
					.ok_or_else(|| err!(Asset("{word:?} is not an order the list knows")))?
			},
			| None => Sort::default(),
		};

		let mut entries = Vec::new();

		for item in value
			.get("projects")
			.map(Value::as_array)
			.unwrap_or_default()
		{
			known(item, ENTRY_FIELDS, "a project on the list")?;

			let path = item
				.get("path")
				.and_then(Value::as_str)
				.filter(|path| !path.is_empty())
				.ok_or_else(|| err!(Asset("a project on the list needs a path")))?;

			entries.push(Entry {
				path: PathBuf::from(path),
				pinned: item
					.get("pinned")
					.and_then(Value::as_bool)
					.unwrap_or(false),
				last_opened: item
					.get("last_opened")
					.and_then(Value::as_u64)
					.unwrap_or(0),
			});
		}

		Ok(Self { projects_dir, sort, entries })
	}

	/// The file's text.
	#[must_use]
	pub fn write(&self) -> String {
		let mut out = String::new();

		out.push_str("{\n");
		writeln!(out, "\t\"schema\": {SCHEMA},").expect("a string takes what is written");
		writeln!(out, "\t\"projects_dir\": {},", json::quoted(&spelled(&self.projects_dir)))
			.expect("a string takes what is written");
		writeln!(out, "\t\"sort\": {},", json::quoted(self.sort.word()))
			.expect("a string takes what is written");
		out.push_str("\t\"projects\": [");

		for (index, entry) in self.entries.iter().enumerate() {
			out.push_str(if index == 0 { "\n" } else { ",\n" });
			write!(
				out,
				"\t\t{{ \"path\": {}, \"pinned\": {}, \"last_opened\": {} }}",
				json::quoted(&spelled(&entry.path)),
				entry.pinned,
				entry.last_opened
			)
			.expect("a string takes what is written");
		}

		if !self.entries.is_empty() {
			out.push_str("\n\t");
		}

		out.push_str("]\n}\n");

		out
	}

	/// Reads the file, or starts an empty list when there is none yet.
	///
	/// @param file - where the list is kept
	/// @param fallback_dir - the folder new projects go in when the file
	/// names none, or does not exist
	///
	/// # Errors
	///
	/// If the file is there and cannot be read as a list. A file that is not
	/// there is not an error: it is what the first run finds.
	pub fn load(file: &Path, fallback_dir: &Path) -> Result<Self> {
		if !file.is_file() {
			return Ok(Self::new(fallback_dir));
		}

		let text = fs::read_to_string(file)
			.map_err(|error| err!(Asset("{} cannot be read: {error}", file.display())))?;

		Self::read(&text, fallback_dir)
			.map_err(|error| err!(Asset("{}: {error}", file.display())))
	}

	/// Writes the file, making its directory if it has to.
	///
	/// @param file - where the list is kept
	pub fn save(&self, file: &Path) -> Result {
		if let Some(dir) = file.parent() {
			fs::create_dir_all(dir)?;
		}

		fs::write(file, self.write())
			.map_err(|error| err!(Asset("{} cannot be written: {error}", file.display())))
	}

	/// Every project on the list, in the order the file has them.
	#[must_use]
	pub fn entries(&self) -> &[Entry] { &self.entries }

	/// Puts a project on the list, unless it is there already.
	///
	/// @param path - the directory holding its `colby.project`
	/// @return whether it was added; the same directory twice is one entry,
	/// however the second one was spelled
	pub fn add(&mut self, path: &Path) -> bool {
		if self.find(path).is_some() {
			return false;
		}

		self.entries.push(Entry {
			path: path.to_owned(),
			pinned: false,
			last_opened: 0,
		});

		true
	}

	/// Takes a project off the list. The project itself is untouched.
	///
	/// @param path - which one
	/// @return whether it was there
	pub fn remove(&mut self, path: &Path) -> bool {
		let Some(index) = self.find(path) else {
			return false;
		};

		self.entries.remove(index);

		true
	}

	/// Pins a project to the top, or lets it go.
	///
	/// @param path - which one
	/// @param pinned - whether it is pinned now
	pub fn pin(&mut self, path: &Path, pinned: bool) {
		if let Some(index) = self.find(path) {
			self.entries[index].pinned = pinned;
		}
	}

	/// Notes that a project was opened just now.
	///
	/// @param path - which one
	/// @param now - the moment, in seconds since the epoch
	pub fn opened(&mut self, path: &Path, now: u64) {
		if let Some(index) = self.find(path) {
			self.entries[index].last_opened = now;
		}
	}

	/// Every project as it is to be shown: read from disk and put in order.
	///
	/// Pinned ones first, then by [`Sort`]; a project whose file cannot be
	/// read is kept in its place and says so, because a drive that is not
	/// plugged in is not a reason to forget what was on it.
	#[must_use]
	pub fn shown(&self) -> Vec<Shown> {
		let mut shown: Vec<Shown> = self.entries.iter().map(Shown::of).collect();

		order(&mut shown, self.sort);

		shown
	}

	/// Where a directory is on the list, if it is.
	fn find(&self, path: &Path) -> Option<usize> {
		self.entries
			.iter()
			.position(|entry| same_place(&entry.path, path))
	}
}

/// One project, read for showing.
#[derive(Clone, Debug)]
pub struct Shown {
	/// The entry it was read for.
	pub entry: Entry,

	/// The project its file describes, or why the file could not be read.
	pub project: std::result::Result<Project, String>,
}

impl Shown {
	/// Reads one entry's project file.
	fn of(entry: &Entry) -> Self {
		let file = entry.path.join(project::FILE);
		let project = if file.is_file() {
			fs::read_to_string(&file)
				.map_err(|error| error.to_string())
				.and_then(|text| {
					Project::parse(&entry.path, &text).map_err(|error| error.to_string())
				})
		} else {
			Err("not found".to_owned())
		};

		Self { entry: entry.clone(), project }
	}

	/// What to call it: the project's name, or the directory's when the file
	/// is not there to say.
	#[must_use]
	pub fn name(&self) -> String {
		self.project
			.as_ref()
			.map_or_else(|_| folder(&self.entry.path), |project| project.name().to_owned())
	}

	/// Its id, or nothing when the file is not there to say.
	#[must_use]
	pub fn id(&self) -> &str {
		self.project
			.as_ref()
			.map_or("", |project| project.id())
	}

	/// Whether a search word is somewhere in what the row shows.
	///
	/// @param filter - what was typed, in any case
	#[must_use]
	pub fn matches(&self, filter: &str) -> bool {
		let filter = filter.trim().to_lowercase();

		if filter.is_empty() {
			return true;
		}

		self.name().to_lowercase().contains(&filter)
			|| self.id().contains(&filter)
			|| spelled(&self.entry.path)
				.to_lowercase()
				.contains(&filter)
	}
}

/// Puts rows in the order they are shown: pinned first, then as asked, with
/// the name as the tie-breaker either way.
fn order(shown: &mut [Shown], sort: Sort) {
	shown.sort_by(|left, right| {
		let by_pin = right.entry.pinned.cmp(&left.entry.pinned);
		let by_name = || {
			left.name()
				.to_lowercase()
				.cmp(&right.name().to_lowercase())
				.then_with(|| left.entry.path.cmp(&right.entry.path))
		};

		by_pin.then_with(|| match sort {
			| Sort::Recent => right
				.entry
				.last_opened
				.cmp(&left.entry.last_opened)
				.then_with(by_name),
			| Sort::Name => by_name(),
		})
	});
}

/// Whether two paths name the same directory.
///
/// Through the filesystem when both resolve, so that a path typed with the
/// other slash or through a link is the same project; by spelling when one
/// does not, which is what a project on a drive that is not there comes to.
///
/// **Both are folded first**, and that is not a tidying step. `..` through a
/// directory that is not there is a path `fs::canonicalize` refuses on unix
/// and folds away on Windows, so without this a project added as
/// `projects/other/../yard` is one entry on one platform and a second copy of
/// an entry on the other. @ref [`lexical`].
fn same_place(left: &Path, right: &Path) -> bool {
	let (left, right) = (lexical(left), lexical(right));

	match (fs::canonicalize(&left), fs::canonicalize(&right)) {
		| (Ok(left), Ok(right)) => left == right,
		| _ => left == right,
	}
}

/// A path as text, with forward slashes: what the file holds and what a row
/// shows, the same on both platforms.
#[must_use]
pub fn spelled(path: &Path) -> String { path.to_string_lossy().replace('\\', "/") }

/// A directory's own name, for a project whose file is not there to say.
fn folder(path: &Path) -> String {
	path.file_name()
		.map_or_else(|| spelled(path), |name| name.to_string_lossy().into_owned())
}

/// Refuses a field this build does not know, in a flat object.
///
/// The rule a project file follows, for the same reason: a field somebody
/// guessed at must not do nothing while looking like it works.
fn known(value: &Value, fields: &[&str], what: &str) -> Result {
	for (name, _) in value.as_object() {
		if !fields.contains(&name.as_str()) {
			return Err(err!(Asset("{what} has no field called {name}")));
		}
	}

	Ok(())
}

/// This moment, in seconds since the epoch.
#[must_use]
pub fn now() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map_or(0, |since| since.as_secs())
}

/// How long ago a moment was, in words.
///
/// @param now - this moment, in seconds since the epoch
/// @param then - the moment asked about, nought for never
#[must_use]
pub fn ago(now: u64, then: u64) -> String {
	const MINUTE: u64 = 60;
	const HOUR: u64 = 60 * MINUTE;
	const DAY: u64 = 24 * HOUR;

	if then == 0 {
		return "never opened".to_owned();
	}

	let since = now.saturating_sub(then);

	if since < MINUTE {
		"just now".to_owned()
	} else if since < HOUR {
		format!("{} min ago", since / MINUTE)
	} else if since < DAY {
		format!("{} h ago", since / HOUR)
	} else if since < 2 * DAY {
		"yesterday".to_owned()
	} else {
		format!("{} days ago", since / DAY)
	}
}

/// Where the list is kept for this person, if the platform says where that is.
///
/// `%APPDATA%\colby\projects.json` on Windows; `$XDG_CONFIG_HOME/colby/` or
/// `~/.config/colby/` elsewhere. The one place in the workspace that reads
/// the environment for a path, because the file is the person's rather than a
/// project's or the engine's, and only the environment knows who the person
/// is.
#[must_use]
pub fn default_file() -> Option<PathBuf> {
	let config = if cfg!(windows) {
		env::var_os("APPDATA").map(PathBuf::from)
	} else {
		env::var_os("XDG_CONFIG_HOME")
			.map(PathBuf::from)
			.or_else(|| env::home_dir().map(|home| home.join(".config")))
	}?;

	Some(config.join(DIRECTORY).join(FILE))
}

/// The folder new projects go in unless somebody says otherwise.
///
/// The person's documents, which is where the field puts them, or their home
/// when there is no such folder, or the working directory when there is not
/// even a home.
#[must_use]
pub fn default_projects_dir() -> PathBuf {
	let Some(home) = env::home_dir() else {
		return PathBuf::from(PROJECTS_FOLDER);
	};

	let documents = home.join("Documents");

	if documents.is_dir() {
		documents.join(PROJECTS_FOLDER)
	} else {
		home.join(PROJECTS_FOLDER)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A directory nothing else is using.
	fn fresh(name: &str) -> PathBuf {
		let dir = env::temp_dir().join(format!("colby_list_{name}"));
		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(&dir).expect("a directory to work in");

		dir
	}

	/// A project on disk, so that a row has a file to read.
	fn project(root: &Path, id: &str, name: &str) -> PathBuf {
		let dir = root.join(id);
		fs::create_dir_all(&dir).expect("the project's directory");
		fs::write(
			dir.join(project::FILE),
			format!(r#"{{ "schema": 1, "engine": "0.1.0", "id": "{id}", "name": "{name}" }}"#),
		)
		.expect("the project file");

		dir
	}

	/// The names of rows, in order.
	fn names(shown: &[Shown]) -> Vec<String> { shown.iter().map(Shown::name).collect() }

	#[test]
	fn a_list_survives_the_text() {
		let mut list = List::new(Path::new("C:/somebody/Documents/colby projects"));
		list.sort = Sort::Name;
		list.add(Path::new("C:/somebody/Documents/colby projects/yard"));
		list.add(Path::new("D:/say \"hi\"/back\\slash"));
		list.pin(Path::new("C:/somebody/Documents/colby projects/yard"), true);
		list.opened(Path::new("C:/somebody/Documents/colby projects/yard"), 1_788_000_000);

		let text = list.write();
		let back = List::read(&text, Path::new("elsewhere")).expect("the text reads back");

		assert_eq!(back.projects_dir, list.projects_dir);
		assert_eq!(back.sort, Sort::Name);
		assert_eq!(back.entries(), &[
			Entry {
				path: PathBuf::from("C:/somebody/Documents/colby projects/yard"),
				pinned: true,
				last_opened: 1_788_000_000,
			},
			Entry {
				// a backslash typed on windows is a forward slash in the
				// file, so the same file reads the same on both platforms
				path: PathBuf::from("D:/say \"hi\"/back/slash"),
				pinned: false,
				last_opened: 0,
			},
		]);
		assert!(text.contains("\"schema\": 1"), "and says which shape it is: {text}");
	}

	#[test]
	fn an_empty_list_is_written_and_read_and_a_missing_file_is_one() {
		let list = List::new(Path::new("here"));
		let text = list.write();

		assert_eq!(List::read(&text, Path::new("x")).expect("reads"), list);
		assert!(text.contains("\"projects\": []"), "{text}");

		let missing = env::temp_dir()
			.join("colby_list_nowhere")
			.join(FILE);
		let loaded = List::load(&missing, Path::new("fallback")).expect("nothing there is empty");

		assert_eq!(loaded, List::new(Path::new("fallback")));
	}

	#[test]
	fn what_the_file_does_not_say_takes_the_defaults() {
		let list = List::read(r#"{ "schema": 1 }"#, Path::new("fallback")).expect("the least");

		assert_eq!(list.projects_dir, Path::new("fallback"));
		assert_eq!(list.sort, Sort::Recent);
		assert!(list.entries().is_empty());

		let list =
			List::read(r#"{ "schema": 1, "projects": [ { "path": "p" } ] }"#, Path::new("."))
				.expect("an entry with only a path");

		assert_eq!(list.entries()[0], Entry {
			path: PathBuf::from("p"),
			pinned: false,
			last_opened: 0
		});
	}

	#[test]
	fn a_field_this_build_does_not_know_is_refused_and_so_is_a_newer_schema() {
		let text = List::read(r#"{ "schema": 1, "thumbs": true }"#, Path::new("."))
			.expect_err("refused")
			.to_string();

		assert!(text.contains("thumbs"), "it says which: {text}");

		let text = List::read(
			r#"{ "schema": 1, "projects": [ { "path": "p", "org": "x" } ] }"#,
			Path::new("."),
		)
		.expect_err("refused")
		.to_string();

		assert!(text.contains("org"), "in an entry too: {text}");

		let text = List::read(r#"{ "schema": 2 }"#, Path::new("."))
			.expect_err("refused")
			.to_string();

		assert!(text.contains("newer engine"), "{text}");

		let text = List::read(r#"{ "schema": 1, "sort": "sideways" }"#, Path::new("."))
			.expect_err("refused")
			.to_string();

		assert!(text.contains("sideways"), "an order nobody knows: {text}");
	}

	#[test]
	fn a_project_that_is_not_there_is_one_entry_too() {
		// the spelling half of the rule below, and the half with teeth on both
		// platforms. A step up through a directory nobody made is a path
		// `fs::canonicalize` refuses outright on unix and folds away on
		// Windows before it ever reaches the disk, so a list that compares
		// what it was handed holds one project twice on one platform and once
		// on the other. Folding first is what makes the two agree.
		let root = fresh("missing");
		let gone = root.join("gone");
		let mut list = List::new(&root);

		assert!(list.add(&gone), "added, whether or not the directory is there");
		assert!(
			!list.add(&root.join("other").join("..").join("gone")),
			"and not a second time under another spelling"
		);
		assert_eq!(list.entries().len(), 1, "one project, one row");
	}

	#[test]
	fn the_same_directory_is_one_entry_however_it_is_spelled() {
		let root = fresh("same");
		let yard = project(&root, "yard", "The Yard");
		let mut list = List::new(&root);

		assert!(list.add(&yard), "added");
		assert!(!list.add(&yard), "not twice");
		assert!(!list.add(&root.join("yard").join(".")), "nor through a step that goes nowhere");
		assert!(!list.add(&root.join("other").join("..").join("yard")), "nor through a step up");
		assert_eq!(list.entries().len(), 1);

		assert!(list.remove(&root.join("yard").join(".")), "taken off by any spelling");
		assert!(list.entries().is_empty());
		assert!(!list.remove(&yard), "and not twice");
		assert!(yard.join(project::FILE).is_file(), "the project itself is untouched");
	}

	#[test]
	fn pinned_rows_come_first_and_the_rest_go_by_the_order_asked() {
		let root = fresh("order");
		let apple = project(&root, "apple", "Apple");
		let cherry = project(&root, "cherry", "Cherry");
		let banana = project(&root, "banana", "banana");
		let mut list = List::new(&root);
		list.add(&apple);
		list.add(&cherry);
		list.add(&banana);
		list.opened(&apple, 100);
		list.opened(&cherry, 300);
		list.opened(&banana, 200);

		assert_eq!(names(&list.shown()), ["Cherry", "banana", "Apple"], "most recent first");

		list.sort = Sort::Name;

		assert_eq!(
			names(&list.shown()),
			["Apple", "banana", "Cherry"],
			"by name, whatever the case"
		);

		list.pin(&apple, true);
		list.sort = Sort::Recent;

		assert_eq!(names(&list.shown()), ["Apple", "Cherry", "banana"], "pinned on top");
	}

	#[test]
	fn a_project_whose_file_is_gone_is_kept_and_says_so() {
		let root = fresh("gone");
		let yard = project(&root, "yard", "The Yard");
		let mut list = List::new(&root);
		list.add(&yard);
		list.add(&root.join("nowhere"));

		let shown = list.shown();
		let missing = shown
			.iter()
			.find(|row| row.entry.path.ends_with("nowhere"))
			.expect("still on the list");

		assert_eq!(missing.project.as_ref().err().map(String::as_str), Some("not found"));
		assert_eq!(missing.name(), "nowhere", "called by its directory");
		assert_eq!(missing.id(), "", "and has no id to show");

		let found = shown
			.iter()
			.find(|row| row.entry.path == yard)
			.expect("the other one");

		assert_eq!(found.name(), "The Yard");
		assert_eq!(found.id(), "yard");
	}

	#[test]
	fn a_search_word_is_looked_for_in_the_name_the_id_and_the_path() {
		let root = fresh("search");
		let yard = project(&root, "yard", "The Yard");
		let mut list = List::new(&root);
		list.add(&yard);

		let row = &list.shown()[0];

		assert!(row.matches(""), "nothing typed matches everything");
		assert!(row.matches("  "), "and so does nothing but spaces");
		assert!(row.matches("YARD"), "in any case");
		assert!(row.matches("the y"), "in the name");
		assert!(row.matches("colby_list_search"), "in the path");
		assert!(!row.matches("garden"), "and not something that is nowhere");
	}

	#[test]
	fn how_long_ago_is_said_in_the_nearest_unit() {
		let now = 10_000_000;

		assert_eq!(ago(now, 0), "never opened");
		assert_eq!(ago(now, now), "just now");
		assert_eq!(ago(now, now - 59), "just now");
		assert_eq!(ago(now, now - 60), "1 min ago");
		assert_eq!(ago(now, now - 3599), "59 min ago");
		assert_eq!(ago(now, now - 3600), "1 h ago");
		assert_eq!(ago(now, now - 5 * 3600), "5 h ago");
		assert_eq!(ago(now, now - 86_400), "yesterday");
		assert_eq!(ago(now, now - 3 * 86_400), "3 days ago");
		assert_eq!(ago(now, now + 500), "just now", "a clock that went backwards is not a panic");
	}

	#[test]
	fn the_file_is_saved_where_it_is_told_and_its_directory_is_made() {
		let root = fresh("save");
		let file = root.join("deeper").join("still").join(FILE);
		let mut list = List::new(&root);
		list.add(Path::new("somewhere"));

		list.save(&file).expect("saved");

		assert_eq!(List::load(&file, Path::new(".")).expect("loaded"), list);
	}

	#[test]
	fn a_file_that_cannot_be_read_names_itself() {
		let root = fresh("broken");
		let file = root.join(FILE);
		fs::write(&file, "{ not json").expect("the file");

		let text = List::load(&file, Path::new("."))
			.expect_err("refused")
			.to_string();

		assert!(text.contains(FILE), "which file: {text}");
	}

	#[test]
	fn the_default_places_are_somewhere() {
		// what they are depends on the machine; that they are a path under
		// something that exists is what a first run needs.
		let file = default_file().expect("a config directory on a developer's machine");

		assert!(file.ends_with(Path::new(DIRECTORY).join(FILE)), "{}", file.display());
		assert!(default_projects_dir().ends_with(PROJECTS_FOLDER));
	}

	#[test]
	fn paths_are_spelled_with_forward_slashes() {
		assert_eq!(spelled(Path::new("C:\\a\\b")), "C:/a/b");
		assert_eq!(spelled(Path::new("/a/b")), "/a/b");
		assert_eq!(Sort::from_word("name"), Some(Sort::Name));
		assert_eq!(Sort::from_word("Name"), None, "the words are the file's, lowercase");
	}
}
