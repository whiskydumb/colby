//! What the editor remembers about a project between runs.
//!
//! One file, `.colby/editor.json`, beside the compiled tree and the
//! thumbnails: the scenes that were open, which of them was on screen, and how
//! wide the panels had been dragged. Derived, like everything else under
//! `.colby/` - losing it costs a window that opens on the project's own scene
//! with the panels at their starting widths, and nothing else.
//!
//! **It is not the project's settings.** `settings.cfg` holds console
//! variables, is a config script somebody can read and edit, and belongs to
//! the engine; this holds where a person left their windows and belongs to the
//! editor. The two are kept apart because a variable is worth reading and a
//! panel width is not.
//!
//! **A scene named here may be gone by the next run.** A tab whose scene no
//! longer compiles is dropped when the file is read rather than opened empty,
//! and a file that names nothing at all is a file that opens one tab on
//! whatever the project starts with.

use std::{fmt::Write as _, fs, path::PathBuf};

use colby_asset::{
	Project,
	json::{self, Value},
};
use colby_core::{Result, err, warn};

/// The file, under the project's derived tree.
pub(crate) const FILE: [&str; 2] = [".colby", "editor.json"];

/// The version of the file this build writes.
pub(crate) const SCHEMA: u32 = 1;

/// How wide the hierarchy starts out, in points, before anybody has dragged
/// it. The three of them are here rather than beside the panels because a
/// starting width is what a file with nothing in it means.
const HIERARCHY_WIDTH: u32 = 240;

/// How wide the inspector starts out.
const INSPECTOR_WIDTH: u32 = 320;

/// How tall the bottom panel starts out.
const BOTTOM_HEIGHT: u32 = 220;

/// Everything the editor keeps about one project.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Session {
	/// The scenes that were open, in the order they were drawn.
	pub(crate) tabs: Vec<String>,

	/// Which of them was on screen.
	pub(crate) current: usize,

	/// How wide the three panels had been dragged, in points.
	pub(crate) panels: Sizes,
}

/// How wide the panels were, in points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sizes {
	/// The hierarchy, on the left.
	pub(crate) left: u32,

	/// The inspector, on the right.
	pub(crate) right: u32,

	/// The console and its neighbors, along the bottom.
	pub(crate) bottom: u32,
}

impl Default for Sizes {
	fn default() -> Self {
		Self {
			left: HIERARCHY_WIDTH,
			right: INSPECTOR_WIDTH,
			bottom: BOTTOM_HEIGHT,
		}
	}
}

impl Session {
	/// Where the file is for this project.
	pub(crate) fn path(project: &Project) -> PathBuf {
		FILE.iter()
			.fold(project.root().to_owned(), |path, part| path.join(part))
	}

	/// Reads what the editor left, or nothing at all.
	///
	/// A file that is not there, cannot be read, or was written by a newer
	/// build is nothing rather than a failure: this is a convenience and a
	/// window that opens without it is a window that works. The reason is
	/// warned about once, because a file somebody edited by hand and got
	/// wrong should say so.
	///
	/// @param project - whose file
	pub(crate) fn open(project: &Project) -> Self {
		let path = Self::path(project);
		let Ok(text) = fs::read_to_string(&path) else {
			return Self::default();
		};

		match Self::read(&text) {
			| Ok(session) => session,
			| Err(failure) => {
				warn!(path = %path.display(), %failure, "the editor's file was not read");

				Self::default()
			},
		}
	}

	/// Writes it out, making the directory if it is not there.
	///
	/// @param project - whose file
	///
	/// # Errors
	///
	/// If the directory or the file cannot be written.
	pub(crate) fn save(&self, project: &Project) -> Result {
		let path = Self::path(project);

		if let Some(dir) = path.parent() {
			fs::create_dir_all(dir)
				.map_err(|failure| err!(Asset("could not make {}: {failure}", dir.display())))?;
		}

		fs::write(&path, self.write())
			.map_err(|failure| err!(Asset("could not write {}: {failure}", path.display())))
	}

	/// Reads the file's text.
	///
	/// @param text - the file
	///
	/// # Errors
	///
	/// If it is not JSON, has no schema, or was written by a newer build.
	pub(crate) fn read(text: &str) -> Result<Self> {
		let value = json::parse(text)?;

		let schema = value
			.get("schema")
			.and_then(Value::as_u32)
			.ok_or_else(|| err!(Asset("the editor's file needs a schema number")))?;

		if schema > SCHEMA {
			return Err(err!(Asset(
				"the editor's file was written by a newer engine: schema {schema}, and this \
				 build reads {SCHEMA}"
			)));
		}

		let tabs: Vec<String> = value
			.get("tabs")
			.map(Value::as_array)
			.map(|names| {
				names
					.iter()
					.filter_map(|name| name.as_str().map(str::to_owned))
					.collect()
			})
			.unwrap_or_default();

		let current = usize::try_from(
			value
				.get("current")
				.and_then(Value::as_u32)
				.unwrap_or(0),
		)
		.unwrap_or(0);

		Ok(Self {
			current: current.min(tabs.len().saturating_sub(1)),
			tabs,
			panels: sizes(value.get("panels")),
		})
	}

	/// The file's text.
	pub(crate) fn write(&self) -> String {
		let mut out = String::new();

		out.push_str("{\n");
		writeln!(out, "\t\"schema\": {SCHEMA},").expect("a string takes what is written");
		out.push_str("\t\"tabs\": [");

		for (index, name) in self.tabs.iter().enumerate() {
			out.push_str(if index == 0 { "\n" } else { ",\n" });
			write!(out, "\t\t{}", json::quoted(name)).expect("a string takes what is written");
		}

		if !self.tabs.is_empty() {
			out.push_str("\n\t");
		}

		out.push_str("],\n");
		writeln!(out, "\t\"current\": {},", self.current)
			.expect("a string takes what is written");
		writeln!(
			out,
			"\t\"panels\": {{ \"left\": {}, \"right\": {}, \"bottom\": {} }}",
			self.panels.left, self.panels.right, self.panels.bottom
		)
		.expect("a string takes what is written");
		out.push_str("}\n");

		out
	}
}

/// The three widths, each falling back to what the panel starts at.
fn sizes(value: Option<&Value>) -> Sizes {
	let fallback = Sizes::default();
	let Some(value) = value else {
		return fallback;
	};

	let read = |name: &str, fallback: u32| {
		value
			.get(name)
			.and_then(Value::as_u32)
			.filter(|size| *size > 0)
			.unwrap_or(fallback)
	};

	Sizes {
		left: read("left", fallback.left),
		right: read("right", fallback.right),
		bottom: read("bottom", fallback.bottom),
	}
}

#[cfg(test)]
mod tests {
	use std::env;

	use super::*;

	/// A project rooted in a scratch directory.
	fn project(name: &str) -> (Project, PathBuf) {
		let root = env::temp_dir().join(format!("colby_session_{name}"));
		drop(fs::remove_dir_all(&root));
		fs::create_dir_all(&root).expect("a directory to work in");
		let project = Project::parse(
			&root,
			r#"{ "schema": 1, "engine": "0.1.0", "id": "sessioned", "name": "Sessioned" }"#,
		)
		.expect("a project");

		(project, root)
	}

	#[test]
	fn what_is_written_is_what_is_read_back() {
		let session = Session {
			tabs: vec!["scenes/yard".to_owned(), "scenes/hangar".to_owned()],
			current: 1,
			panels: Sizes { left: 300, right: 260, bottom: 180 },
		};

		let read = Session::read(&session.write()).expect("read back");

		assert_eq!(read, session);
	}

	#[test]
	fn a_file_with_nothing_in_it_but_a_schema_is_the_starting_state() {
		let read = Session::read("{ \"schema\": 1 }").expect("read");

		assert_eq!(read, Session::default());
		assert!(read.tabs.is_empty());
		assert_eq!(read.panels, Sizes::default());
	}

	#[test]
	fn a_file_from_a_newer_engine_is_refused_rather_than_half_read() {
		let text = format!("{{ \"schema\": {} }}", SCHEMA + 1);

		let failure = Session::read(&text).expect_err("refused");

		assert!(format!("{failure}").contains("newer engine"), "got {failure}");
	}

	#[test]
	fn a_current_tab_past_the_end_is_pulled_back_to_one_that_is_there() {
		let read =
			Session::read("{ \"schema\": 1, \"tabs\": [\"scenes/yard\"], \"current\": 7 }")
				.expect("read");

		assert_eq!(read.current, 0);
	}

	#[test]
	fn a_width_of_zero_is_the_starting_width_rather_than_a_panel_nobody_can_see() {
		let read = Session::read(
			"{ \"schema\": 1, \"panels\": { \"left\": 0, \"right\": 260, \"bottom\": 0 } }",
		)
		.expect("read");

		assert_eq!(read.panels.left, Sizes::default().left);
		assert_eq!(read.panels.right, 260);
		assert_eq!(read.panels.bottom, Sizes::default().bottom);
	}

	#[test]
	fn a_project_with_no_file_yet_opens_at_the_starting_state_and_can_be_written() {
		let (project, root) = project("fresh");

		assert_eq!(Session::open(&project), Session::default(), "nothing to read");

		let session = Session {
			tabs: vec!["scenes/yard".to_owned()],
			current: 0,
			panels: Sizes::default(),
		};
		session.save(&project).expect("written");

		assert!(
			Session::path(&project)
				.components()
				.any(|part| part.as_os_str() == ".colby"),
			"under the derived tree"
		);
		assert_eq!(Session::open(&project), session, "and read back next time");
		assert!(
			root.join(".colby").join("editor.json").exists(),
			"beside the compiled tree rather than in the repository"
		);
	}

	#[test]
	fn a_file_nobody_can_parse_opens_at_the_starting_state_rather_than_failing() {
		let (project, _) = project("broken");
		let path = Session::path(&project);
		fs::create_dir_all(path.parent().expect("a parent")).expect("the directory");
		fs::write(&path, "not json at all").expect("the file");

		assert_eq!(Session::open(&project), Session::default());
	}
}
