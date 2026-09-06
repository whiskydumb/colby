//! A template: a directory a new project is copied from.
//!
//! `templates/<name>/` under the engine checkout, marked by a `template.json`
//! saying what to call it in the wizard. Everything else in it is copied as
//! it is, with three words filled in on the way - `$id`, `$name` and
//! `$engine` - in file names and in text files alike, which is what the field
//! does and what keeps a template a directory a person can read rather than
//! code that writes one. A file whose extension is not a text format the
//! engine knows is copied byte for byte.
//!
//! A template's `colby.project` therefore says `"id": "$id"` and is not a
//! project until it is copied, which is why the directory lives under
//! `templates/` and not under `projects/`, where the workspace's member glob
//! would try to build its crate.

use std::{
	fs,
	path::{Path, PathBuf},
};

use colby_asset::json::{self, Value};
use colby_core::{Result, err, warn};

/// The directory under the engine checkout templates live in.
pub const DIRECTORY: &str = "templates";

/// The file that marks a directory as a template, and names it.
pub const FILE: &str = "template.json";

/// Every field the file may carry.
const FIELDS: &[&str] = &["title", "description", "order"];

/// The word a project's id is written as in a template.
pub const ID: &str = "$id";

/// The word a project's name is written as.
pub const NAME: &str = "$name";

/// The word the engine version is written as.
pub const ENGINE: &str = "$engine";

/// Extensions of the files the words are filled in inside of.
///
/// Everything the engine reads as text, plus what a person writes beside it;
/// anything else is copied as bytes, because a `$id` inside a png is a png.
const TEXT_EXTENSIONS: &[&str] = &[
	"project", "toml", "rs", "scene", "lua", "html", "css", "cfg", "json", "txt", "md",
];

/// Files that are text without having an extension to say so.
const TEXT_NAMES: &[&str] = &[".gitignore", ".gitattributes", ".editorconfig"];

/// A template, as its file describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Template {
	/// The directory to copy.
	pub dir: PathBuf,

	/// What the wizard calls it.
	pub title: String,

	/// A line under the title.
	pub description: String,

	/// Where it sits in the wizard's list; lower first.
	pub order: u64,
}

impl Template {
	/// Reads one template's file.
	///
	/// @param dir - the directory holding [`FILE`]
	///
	/// # Errors
	///
	/// If the file is not there, or carries a field this build does not know,
	/// or has no title.
	pub fn read(dir: &Path) -> Result<Self> {
		let path = dir.join(FILE);
		let text = fs::read_to_string(&path)
			.map_err(|error| err!(Asset("{} cannot be read: {error}", path.display())))?;
		let value = json::parse(&text)?;

		for (name, _) in value.as_object() {
			if !FIELDS.contains(&name.as_str()) {
				return Err(err!(Asset("a template has no field called {name}")));
			}
		}

		let title = value
			.get("title")
			.and_then(Value::as_str)
			.filter(|title| !title.is_empty())
			.ok_or_else(|| err!(Asset("a template needs a title")))?;

		Ok(Self {
			dir: dir.to_owned(),
			title: title.to_owned(),
			description: value
				.get("description")
				.and_then(Value::as_str)
				.unwrap_or_default()
				.to_owned(),
			order: value
				.get("order")
				.and_then(Value::as_u64)
				.unwrap_or(0),
		})
	}

	/// Every template under an engine checkout, in the order they are listed.
	///
	/// A directory without the file is not a template and is passed over; one
	/// whose file cannot be read is a warning and is passed over too, because
	/// a broken template must not keep the wizard from offering the rest.
	///
	/// @param engine - the engine checkout
	#[must_use]
	pub fn find(engine: &Path) -> Vec<Self> {
		let mut found = Vec::new();
		let Ok(entries) = fs::read_dir(engine.join(DIRECTORY)) else {
			return found;
		};

		for entry in entries.flatten() {
			let dir = entry.path();

			if !dir.join(FILE).is_file() {
				continue;
			}

			match Self::read(&dir) {
				| Ok(template) => found.push(template),
				| Err(error) =>
					warn!(template = %dir.display(), %error, "not a template; skipped"),
			}
		}

		found.sort_by(|left, right| {
			left.order
				.cmp(&right.order)
				.then_with(|| left.title.cmp(&right.title))
		});

		found
	}

	/// Copies the template into a new directory, with the three words filled
	/// in.
	///
	/// @param target - the directory to make; it must not exist yet
	/// @param id - what `$id` becomes
	/// @param name - what `$name` becomes
	/// @param engine - what `$engine` becomes
	///
	/// # Errors
	///
	/// If the target is already there, or anything cannot be read or written.
	pub fn apply(&self, target: &Path, id: &str, name: &str, engine: &str) -> Result {
		if target.exists() {
			return Err(err!(Asset("{} already exists", target.display())));
		}

		let words = Words { id, name, engine };

		copy_dir(&self.dir, target, &words, true)
	}
}

/// The three words, and what each becomes.
struct Words<'a> {
	id: &'a str,
	name: &'a str,
	engine: &'a str,
}

impl Words<'_> {
	/// Fills the words in.
	fn fill(&self, text: &str) -> String {
		text.replace(ID, self.id)
			.replace(NAME, self.name)
			.replace(ENGINE, self.engine)
	}
}

/// Copies a directory, filling the words in as it goes.
///
/// @param from - the directory to copy
/// @param to - where to; made if it is not there
/// @param words - what to fill in
/// @param root - whether this is the template's own directory, whose marker
/// file is not part of a project
fn copy_dir(from: &Path, to: &Path, words: &Words<'_>, root: bool) -> Result {
	fs::create_dir_all(to)?;

	for entry in fs::read_dir(from)? {
		let entry = entry?;
		let source = entry.path();
		let file_name = entry.file_name().to_string_lossy().into_owned();

		if root && file_name == FILE {
			continue;
		}

		let target = to.join(words.fill(&file_name));

		if source.is_dir() {
			copy_dir(&source, &target, words, false)?;
		} else if is_text(&source) {
			let text = fs::read_to_string(&source)?;
			fs::write(&target, words.fill(&text))?;
		} else {
			fs::copy(&source, &target)?;
		}
	}

	Ok(())
}

/// Whether the words are filled in inside a file.
fn is_text(path: &Path) -> bool {
	let by_extension = path
		.extension()
		.and_then(|extension| extension.to_str())
		.is_some_and(|extension| TEXT_EXTENSIONS.contains(&extension));
	let by_name = path
		.file_name()
		.and_then(|name| name.to_str())
		.is_some_and(|name| TEXT_NAMES.contains(&name));

	by_extension || by_name
}

#[cfg(test)]
mod tests {
	use std::env;

	use colby_asset::{Project, project};

	use super::*;

	/// A directory nothing else is using.
	fn fresh(name: &str) -> PathBuf {
		let dir = env::temp_dir().join(format!("colby_template_{name}"));
		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(&dir).expect("a directory to work in");

		dir
	}

	/// Every file under a directory, as paths relative to it.
	fn files_under(root: &Path) -> Vec<PathBuf> {
		let mut files = Vec::new();
		walk(root, &PathBuf::new(), &mut files);

		files
	}

	/// One directory of the walk, into the list.
	fn walk(root: &Path, relative: &Path, files: &mut Vec<PathBuf>) {
		for entry in fs::read_dir(root.join(relative)).expect("a directory") {
			let entry = entry.expect("an entry");
			let path = relative.join(entry.file_name());

			if entry.path().is_dir() {
				walk(root, &path, files);
			} else {
				files.push(path);
			}
		}
	}

	/// A template on disk with the three words in every place they can be.
	fn template(root: &Path, name: &str, order: u64) -> PathBuf {
		let dir = root.join(DIRECTORY).join(name);
		fs::create_dir_all(dir.join("game")).expect("the template's directories");
		fs::write(
			dir.join(FILE),
			format!(r#"{{ "title": "{name}", "description": "a {name}", "order": {order} }}"#),
		)
		.expect("the marker");
		fs::write(
			dir.join(project::FILE),
			r#"{ "schema": 1, "engine": "$engine", "id": "$id", "name": "$name", "game": "game" }"#,
		)
		.expect("the project file");
		fs::write(dir.join("game").join("Cargo.toml"), "[package]\nname = \"$id_game\"\n")
			.expect("the manifest");
		fs::write(dir.join("$id.txt"), "$name\n").expect("a file named by the word");
		fs::write(dir.join("picture.png"), b"$id\x00\xFF").expect("a binary");
		fs::write(dir.join(".gitignore"), "$id/\n").expect("a dotfile");

		dir
	}

	#[test]
	fn a_template_is_copied_with_the_words_filled_in_and_a_binary_left_alone() {
		let root = fresh("apply");
		let dir = template(&root, "plain", 0);
		let template = Template::read(&dir).expect("a template");
		let target = root.join("yard");

		template
			.apply(&target, "yard", "The Yard", "0.9.0")
			.expect("copied");

		let project = Project::open(&target).expect("the copy is a project");

		assert_eq!(project.id(), "yard");
		assert_eq!(project.name(), "The Yard");
		assert_eq!(project.engine(), "0.9.0");
		assert_eq!(
			fs::read_to_string(target.join("game").join("Cargo.toml")).expect("the manifest"),
			"[package]\nname = \"yard_game\"\n"
		);
		assert_eq!(
			fs::read_to_string(target.join("yard.txt")).expect("named by the word"),
			"The Yard\n"
		);
		assert_eq!(
			fs::read(target.join("picture.png")).expect("the binary"),
			b"$id\x00\xFF",
			"a picture is bytes, whatever it happens to contain"
		);
		assert_eq!(
			fs::read_to_string(target.join(".gitignore")).expect("the dotfile"),
			"yard/\n",
			"a dotfile is text without an extension to say so"
		);
		assert!(!target.join(FILE).exists(), "the marker is the template's, not the project's");
	}

	#[test]
	fn a_target_that_is_already_there_is_refused_untouched() {
		let root = fresh("taken");
		let dir = template(&root, "plain", 0);
		let template = Template::read(&dir).expect("a template");
		let target = root.join("taken");
		fs::create_dir_all(&target).expect("something in the way");
		fs::write(target.join("mine.txt"), "mine").expect("a file of somebody's");

		let text = template
			.apply(&target, "taken", "Taken", "0.1.0")
			.expect_err("refused")
			.to_string();

		assert!(text.contains("already exists"), "{text}");
		assert!(!target.join(project::FILE).exists(), "nothing was written");
		assert!(target.join("mine.txt").is_file(), "and nothing was taken");
	}

	#[test]
	fn templates_are_found_in_order_and_a_directory_without_the_file_is_not_one() {
		let root = fresh("find");
		template(&root, "second", 5);
		template(&root, "first", 1);
		template(&root, "also_first", 1);
		fs::create_dir_all(root.join(DIRECTORY).join("not_one")).expect("an odd directory");
		let broken = root.join(DIRECTORY).join("broken");
		fs::create_dir_all(&broken).expect("a broken one");
		fs::write(broken.join(FILE), "{ not json").expect("its file");

		let titles: Vec<String> = Template::find(&root)
			.into_iter()
			.map(|template| template.title)
			.collect();

		assert_eq!(titles, ["also_first", "first", "second"], "by order, then by title");
		assert!(Template::find(&root.join("nowhere")).is_empty(), "no directory, no templates");
	}

	#[test]
	fn a_file_this_build_does_not_know_and_a_template_with_no_title_are_refused() {
		let root = fresh("refused");
		let dir = root.join("odd");
		fs::create_dir_all(&dir).expect("a directory");
		fs::write(dir.join(FILE), r#"{ "title": "x", "icon": "star" }"#).expect("the file");

		let text = Template::read(&dir)
			.expect_err("refused")
			.to_string();

		assert!(text.contains("icon"), "it says which: {text}");

		fs::write(dir.join(FILE), r#"{ "description": "no title" }"#).expect("the file");

		let text = Template::read(&dir)
			.expect_err("refused")
			.to_string();

		assert!(text.contains("title"), "{text}");
	}

	#[test]
	fn the_blank_template_is_the_blank_fixture() {
		// the one project that always lives in the tree is what a new project
		// starts from, and the two are kept the same file for file by this
		// rather than by care. The fixture's runs leave other files beside it;
		// only what the template holds is compared.
		let engine = Path::new(env!("CARGO_MANIFEST_DIR"))
			.parent()
			.and_then(Path::parent)
			.expect("the crate sits two directories below the checkout");
		let template =
			Template::read(&engine.join(DIRECTORY).join("blank")).expect("the blank template");
		let fixture = engine.join(project::MOUNTS_DIR).join("blank");
		let target = fresh("blank").join("blank");

		template
			.apply(&target, "blank", "Blank", project::ENGINE)
			.expect("copied");

		let mut compared = 0;

		for relative in files_under(&target) {
			let copied = fs::read(target.join(&relative)).expect("the copy");
			let kept = fs::read(fixture.join(&relative))
				.unwrap_or_else(|_| panic!("the fixture has no {}", relative.display()));

			assert!(copied == kept, "{} differs between the two", relative.display());
			compared += 1;
		}

		assert!(compared >= 3, "the project file and the crate's two files at least: {compared}");
	}
}
