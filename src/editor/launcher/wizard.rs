//! Making a new project: the form's rules, and what pressing Create does.
//!
//! The form is the field's - a name, an id derived from it, a location, a
//! template and a version-control choice - and the rules are colby's own: the
//! id is [`colby_asset::project::valid_id`]'s word and it names the folder,
//! so a project's directory is called what the project is and the engine's
//! warning about the two disagreeing is never heard from here.
//!
//! **Version control is two files and, when asked, `git init`.** The two
//! files say what a run leaves under a project and what git must not turn
//! into text; `git init` is only offered when git is on the path, and it is
//! the one thing here that runs somebody else's program.
//!
//! Nothing in this module draws. What it does is checked by tests, and the
//! page that shows it is [`creator`](super::creator).

use std::{
	fs,
	path::{Path, PathBuf},
	process::Command,
};

use colby_asset::project;
use colby_core::{Result, err, info};

use super::template::Template;

/// What a project is called until somebody says otherwise.
const DEFAULT_NAME: &str = "My Project";

/// What a new project's `.gitignore` says.
///
/// Everything a run leaves under a project: the compiled assets, the console's
/// archive, saved worlds and screenshots. The same four lines the engine's own
/// file has for them.
pub const GITIGNORE: &str = "# what running colby leaves under a project
.colby/
settings.cfg
saves/
screenshots/
";

/// What a new project's `.gitattributes` says.
///
/// Line endings normalized, and the binary formats a project holds named as
/// such so that `text=auto` never finds a NUL in a picture and guesses wrong.
pub const GITATTRIBUTES: &str = "* text=auto eol=lf

*.png  binary
*.jpg  binary
*.jpeg binary
*.glb  binary
*.bin  binary
*.ttf  binary
*.wav  binary
";

/// The form, as it stands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wizard {
	/// What a person calls the project.
	pub name: String,

	/// The short lowercase word it goes by, and the folder's name.
	pub id: String,

	/// Whether the id was typed by hand, in which case it stops following
	/// the name.
	pub id_edited: bool,

	/// The folder the project's directory is made in.
	pub location: String,

	/// Whether that folder becomes the one new projects go in from now on.
	pub remember: bool,

	/// Whether to write `.gitignore` and `.gitattributes`.
	pub vcs_files: bool,

	/// Whether to run `git init`.
	pub git_init: bool,

	/// Which template, as an index into the wizard's list.
	pub template: usize,
}

impl Wizard {
	/// A form filled in the way a person is likeliest to want it.
	///
	/// @param location - the folder new projects go in
	/// @param git - whether git was found, which is what decides the default
	/// of `git init`
	#[must_use]
	pub fn new(location: &Path, git: bool) -> Self {
		let name = unused_name(location);

		Self {
			id: ident(&name),
			name,
			id_edited: false,
			location: super::list::spelled(location),
			remember: false,
			vcs_files: true,
			git_init: git,
			template: 0,
		}
	}

	/// Follows the name with the id, unless the id was typed by hand.
	pub fn named(&mut self) {
		if !self.id_edited {
			self.id = ident(&self.name);
		}
	}

	/// Where the project would be made: `<location>/<id>`.
	#[must_use]
	pub fn target(&self) -> PathBuf { Path::new(self.location.trim()).join(self.id.trim()) }

	/// Whether the form can be created as it stands, and why not when it
	/// cannot.
	///
	/// @return the directory that would be made, or the one sentence the form
	/// shows under itself
	pub fn check(&self) -> std::result::Result<PathBuf, String> {
		if self.name.trim().is_empty() {
			return Err("a project needs a name".to_owned());
		}

		if !project::valid_id(self.id.trim()) {
			return Err("an id is two to thirty-two of a-z, 0-9 and _".to_owned());
		}

		if self.location.trim().is_empty() {
			return Err("a project needs somewhere to go".to_owned());
		}

		let target = self.target();

		if target.exists() {
			return Err(format!("{} already exists", super::list::spelled(&target)));
		}

		Ok(target)
	}

	/// Makes the project: the template copied, the version-control files
	/// written, git started.
	///
	/// @param template - the template chosen
	/// @param engine - the engine version to write into the project file
	/// @return the project's directory
	///
	/// # Errors
	///
	/// If the form does not pass [`check`](Self::check), or anything on the
	/// way cannot be written.
	pub fn create(&self, template: &Template, engine: &str) -> Result<PathBuf> {
		let target = self
			.check()
			.map_err(|reason| err!(Asset("{reason}")))?;

		template.apply(&target, self.id.trim(), self.name.trim(), engine)?;

		if self.vcs_files {
			write_vcs_files(&target)?;
		}

		if self.git_init {
			git_init(&target)?;
		}

		info!(
			project = self.id.trim(),
			at = %target.display(),
			template = template.title,
			"project created"
		);

		Ok(target)
	}
}

/// The id a name suggests: lowercase, anything that is not a letter, a digit
/// or an underscore turned into one, runs of them folded, the ends trimmed.
///
/// A name in an alphabet that has no place in an id comes out empty, and the
/// form then waits for one typed by hand.
///
/// @param name - what a person typed
#[must_use]
pub fn ident(name: &str) -> String {
	let mut id = String::with_capacity(name.len());

	for letter in name.chars().flat_map(char::to_lowercase) {
		if letter.is_ascii_lowercase() || letter.is_ascii_digit() {
			id.push(letter);
		} else if !id.ends_with('_') {
			id.push('_');
		}
	}

	let trimmed = id.trim_matches('_');

	trimmed
		.chars()
		.take(project::ID_LONGEST)
		.collect()
}

/// A name whose folder is not taken yet: `My Project`, then `My Project 2`
/// and so on.
///
/// @param location - the folder projects go in
#[must_use]
pub fn unused_name(location: &Path) -> String {
	let mut name = DEFAULT_NAME.to_owned();
	let mut number = 2;

	while location.join(ident(&name)).exists() {
		name = format!("{DEFAULT_NAME} {number}");
		number += 1;
	}

	name
}

/// Writes the two files git wants beside a project.
///
/// A file that is already there is left alone: a template may have brought
/// its own, and a person's is theirs.
///
/// @param dir - the project's directory
pub fn write_vcs_files(dir: &Path) -> Result {
	for (name, text) in [(".gitignore", GITIGNORE), (".gitattributes", GITATTRIBUTES)] {
		let path = dir.join(name);

		if !path.exists() {
			fs::write(&path, text)?;
		}
	}

	Ok(())
}

/// The version of the git on the path, or nothing when there is none.
#[must_use]
pub fn git_version() -> Option<String> {
	let output = Command::new("git")
		.arg("--version")
		.output()
		.ok()?;

	if !output.status.success() {
		return None;
	}

	Some(
		String::from_utf8_lossy(&output.stdout)
			.trim()
			.to_owned(),
	)
}

/// Starts a repository in a directory.
///
/// @param dir - the project's directory
pub fn git_init(dir: &Path) -> Result {
	let status = Command::new("git")
		.args(["init", "--quiet"])
		.current_dir(dir)
		.status()
		.map_err(|error| err!(Asset("git could not be run: {error}")))?;

	if !status.success() {
		return Err(err!(Asset("git init failed in {}: {status}", dir.display())));
	}

	Ok(())
}

#[cfg(test)]
mod tests {
	use std::env;

	use colby_asset::Project;

	use super::{super::template, *};

	/// A directory nothing else is using.
	fn fresh(name: &str) -> PathBuf {
		let dir = env::temp_dir().join(format!("colby_wizard_{name}"));
		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(&dir).expect("a directory to work in");

		dir
	}

	/// The smallest template there is: a project file and nothing else.
	fn template(root: &Path) -> Template {
		let dir = root.join(template::DIRECTORY).join("tiny");
		fs::create_dir_all(&dir).expect("the template's directory");
		fs::write(dir.join(template::FILE), r#"{ "title": "Tiny" }"#).expect("the marker");
		fs::write(
			dir.join(project::FILE),
			r#"{ "schema": 1, "engine": "$engine", "id": "$id", "name": "$name" }"#,
		)
		.expect("the project file");

		Template::read(&dir).expect("a template")
	}

	#[test]
	fn an_id_is_the_name_in_lowercase_with_everything_else_an_underscore() {
		assert_eq!(ident("My Project"), "my_project");
		assert_eq!(ident("Garry's Project"), "garry_s_project");
		assert_eq!(ident("  spaced   out  "), "spaced_out", "runs folded, ends trimmed");
		assert_eq!(ident("Yard-2"), "yard_2");
		assert_eq!(ident("UPPER"), "upper");
		assert_eq!(ident("---"), "", "nothing usable is nothing");
		assert_eq!(
			ident("\u{414}\u{432}\u{43e}\u{440}"),
			"",
			"an alphabet with no place in an id is nothing"
		);
		assert_eq!(ident("x"), "x", "too short is the check's business, not this one's");
		assert_eq!(ident(&"a".repeat(40)).len(), project::ID_LONGEST, "cut to the longest");
	}

	#[test]
	fn the_id_follows_the_name_until_it_is_typed_by_hand() {
		let root = fresh("follow");
		let mut wizard = Wizard::new(&root, false);

		assert_eq!(wizard.name, "My Project");
		assert_eq!(wizard.id, "my_project");

		wizard.name = "The Yard".to_owned();
		wizard.named();

		assert_eq!(wizard.id, "the_yard", "followed");

		wizard.id = "yard".to_owned();
		wizard.id_edited = true;
		wizard.name = "The Yard Again".to_owned();
		wizard.named();

		assert_eq!(wizard.id, "yard", "and left alone once typed");
	}

	#[test]
	fn the_default_name_steps_past_a_folder_that_is_taken() {
		let root = fresh("taken_name");

		assert_eq!(unused_name(&root), "My Project");

		fs::create_dir_all(root.join("my_project")).expect("the folder taken");

		assert_eq!(unused_name(&root), "My Project 2");

		fs::create_dir_all(root.join("my_project_2")).expect("and that one");

		assert_eq!(unused_name(&root), "My Project 3");
	}

	#[test]
	fn the_check_says_what_stops_the_form() {
		let root = fresh("check");
		let mut wizard = Wizard::new(&root, false);

		assert_eq!(wizard.check().expect("fine as it starts"), root.join("my_project"));

		wizard.name = " ".to_owned();
		assert!(
			wizard
				.check()
				.expect_err("no name")
				.contains("name")
		);

		wizard.name = "Yard".to_owned();
		wizard.id = "Y".to_owned();
		assert!(
			wizard
				.check()
				.expect_err("a bad id")
				.contains("id")
		);

		wizard.id = "yard".to_owned();
		wizard.location = String::new();
		assert!(
			wizard
				.check()
				.expect_err("nowhere")
				.contains("somewhere")
		);

		wizard.location = super::super::list::spelled(&root);
		fs::create_dir_all(root.join("yard")).expect("in the way");
		assert!(
			wizard
				.check()
				.expect_err("taken")
				.contains("already exists")
		);
	}

	#[test]
	fn creating_writes_the_project_and_the_two_git_files_and_nothing_starts_git() {
		let root = fresh("create");
		let template = template(&root);
		let mut wizard = Wizard::new(&root, false);
		wizard.name = "The Yard".to_owned();
		wizard.named();

		let made = wizard
			.create(&template, "0.1.0")
			.expect("created");

		assert_eq!(made, root.join("the_yard"));

		let project = Project::open(&made).expect("a project");

		assert_eq!(project.name(), "The Yard");
		assert_eq!(project.id(), "the_yard");
		assert_eq!(project.engine(), "0.1.0");
		assert_eq!(fs::read_to_string(made.join(".gitignore")).expect("written"), GITIGNORE);
		assert_eq!(
			fs::read_to_string(made.join(".gitattributes")).expect("written"),
			GITATTRIBUTES
		);
		assert!(!made.join(".git").exists(), "git init was not asked for");
		assert!(
			wizard
				.create(&template, "0.1.0")
				.expect_err("not twice")
				.to_string()
				.contains("already exists")
		);
	}

	#[test]
	fn the_two_git_files_are_left_alone_when_the_template_brought_its_own() {
		let root = fresh("kept");
		let template = template(&root);
		fs::write(template.dir.join(".gitignore"), "mine\n").expect("the template's own");
		let wizard = Wizard::new(&root, false);

		let made = wizard
			.create(&template, "0.1.0")
			.expect("created");

		assert_eq!(fs::read_to_string(made.join(".gitignore")).expect("kept"), "mine\n");
		assert_eq!(
			fs::read_to_string(made.join(".gitattributes")).expect("written"),
			GITATTRIBUTES,
			"and the one it did not bring is written"
		);
	}

	#[test]
	fn git_is_found_on_a_developers_machine_and_starts_a_repository_when_asked() {
		// not skipped when git is missing: this workspace is built and gated
		// with git, so a machine without it is a broken machine rather than a
		// case to allow for.
		let version = git_version().expect("git is on the path");

		assert!(version.starts_with("git version"), "{version}");

		let root = fresh("git");
		let template = template(&root);
		let mut wizard = Wizard::new(&root, true);
		wizard.vcs_files = false;

		assert!(wizard.git_init, "on by default when git is there");

		let made = wizard
			.create(&template, "0.1.0")
			.expect("created");

		assert!(made.join(".git").is_dir(), "a repository");
		assert!(
			!made.join(".gitignore").exists(),
			"and no files, because they were not asked for"
		);
	}
}
