//! What is in a project's `assets/`: every source, what it compiles into,
//! and whether it has.
//!
//! No egui and no GPU in it: a walk of the two trees and a comparison of
//! their times, which is what makes it a module with tests. The compiler's
//! own report is not asked for. The output tree is the index, as it is for
//! the asset loop, and a source newer than its output is a source whose last
//! compile did not finish, whatever the reason - the console has the reason,
//! because the loop warned when it happened.

use std::{
	fs,
	path::{Path, PathBuf},
	time::SystemTime,
};

use colby_asset::compile::{self, Kind};

/// Whether a source has been compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum State {
	/// Its output is there and no older than it.
	Compiled,

	/// Its output is there and older than it: the last compile of it did
	/// not finish, and the engine is running the one before.
	Stale,

	/// There is no output at all: never compiled, or never successfully.
	Uncompiled,
}

impl State {
	/// What to say beside a row, in the fewest words; nothing for the usual
	/// case.
	pub(crate) const fn word(self) -> &'static str {
		match self {
			| Self::Compiled => "",
			| Self::Stale => "stale",
			| Self::Uncompiled => "not compiled",
		}
	}
}

/// One source under the asset tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
	/// The name the engine registers it under: `meshes/crystal`.
	pub(crate) name: String,

	/// What it compiles into.
	pub(crate) kind: Kind,

	/// The source file.
	pub(crate) source: PathBuf,

	/// Where its output is, or would be.
	pub(crate) output: PathBuf,

	/// Whether the output is there and current.
	pub(crate) state: State,
}

/// What to call a kind in a row.
pub(crate) const fn word(kind: Kind) -> &'static str {
	match kind {
		| Kind::Mesh => "mesh",
		| Kind::Texture => "texture",
		| Kind::Font => "font",
		| Kind::Document => "document",
		| Kind::Model => "model",
		| Kind::Sound => "sound",
		| Kind::Scene => "scene",
		| Kind::Material => "material",
		| Kind::Skeleton => "skeleton",
		| Kind::Clip => "clip",
		| Kind::Script => "program",
	}
}

/// Whether a source is text somebody would open in an editor.
///
/// **Not every source is**, which is why this is a question rather than a
/// blanket yes: a `.png`, a `.wav`, a `.ttf` and a `.glb` are bytes, and
/// opening one in a code editor shows a screen of replacement characters. The
/// four that are text are the four the compiler parses as text - geometry,
/// documents, materials and programs.
///
/// A `.scene` is text too and is deliberately **not** here: double-clicking one
/// already means "open this scene", and a row cannot mean two things. It is
/// still reachable by name from the console.
///
/// A `.gltf` is text and a `.glb` is not, and they are one [`Kind`] - so the
/// kind says no, because the wrong answer for half of them is a screen of
/// binary.
pub(crate) const fn is_text(kind: Kind) -> bool {
	matches!(kind, Kind::Mesh | Kind::Document | Kind::Material | Kind::Script)
}

/// Every source under a project's asset tree, by name.
///
/// A tree that is not there, or cannot be walked, is an empty list rather
/// than a failure: a project that has not made an asset yet looks exactly
/// like that.
///
/// @param sources - the project's `assets/`
/// @param outputs - its compiled tree
pub(crate) fn scan(sources: &Path, outputs: &Path) -> Vec<Entry> {
	let Ok(found) = compile::sources(sources) else {
		return Vec::new();
	};

	let mut entries: Vec<Entry> = found
		.into_iter()
		.filter_map(|source| entry(sources, outputs, source))
		.collect();
	entries.sort_by(|left, right| left.name.cmp(&right.name));

	entries
}

/// One source as an entry, if the compiler has a name and an output for it.
fn entry(sources: &Path, outputs: &Path, source: PathBuf) -> Option<Entry> {
	let kind = Kind::of(&source)?;
	let name = compile::asset_name(sources, &source).ok()?;
	let output = compile::output_path(sources, outputs, &source).ok()?;
	let state = state(&source, &output);

	Some(Entry { name, kind, source, output, state })
}

/// Whether a source has been compiled, by the two files' times.
///
/// @param source - the source
/// @param output - where its output would be
pub(crate) fn state(source: &Path, output: &Path) -> State {
	let Some(made) = modified(output) else {
		return State::Uncompiled;
	};

	match modified(source) {
		| Some(edited) if edited > made => State::Stale,
		| _ => State::Compiled,
	}
}

/// When a file was last written, if it is there.
pub(crate) fn modified(path: &Path) -> Option<SystemTime> {
	fs::metadata(path).ok()?.modified().ok()
}

#[cfg(test)]
mod tests {
	use std::{env, fs::File, time::Duration};

	use super::*;

	/// A scratch project tree: an `assets/` and a `.colby/assets/`.
	fn fresh(name: &str) -> (PathBuf, PathBuf) {
		let root = env::temp_dir().join(format!("colby_catalog_{name}"));
		drop(fs::remove_dir_all(&root));
		let sources = root.join("assets");
		let outputs = root.join(".colby").join("assets");
		fs::create_dir_all(&sources).expect("a directory to work in");
		fs::create_dir_all(&outputs).expect("and one to compile into");

		(sources, outputs)
	}

	/// Writes a file, making its directory.
	fn put(path: &Path, bytes: &[u8]) {
		if let Some(dir) = path.parent() {
			fs::create_dir_all(dir).expect("the directory");
		}

		fs::write(path, bytes).expect("the file");
	}

	/// Moves a file's time by this much, either way.
	fn shift(path: &Path, by: Duration, back: bool) {
		let now = SystemTime::now();
		let then = if back { now - by } else { now + by };
		File::options()
			.write(true)
			.open(path)
			.expect("the file opens")
			.set_modified(then)
			.expect("and takes a time");
	}

	#[test]
	fn every_source_is_listed_by_name_with_its_kind_and_whether_it_compiled() {
		let (sources, outputs) = fresh("listing");
		put(&sources.join("meshes/crystal.obj"), b"v 0 0 0\n");
		put(&outputs.join("meshes/crystal.cmesh"), b"compiled");
		put(&sources.join("textures/wall.png"), b"not really a png");
		put(&sources.join("scenes/yard.scene"), b"{}");
		put(&outputs.join("scenes/yard.cscene"), b"old");
		// the scene's output was made a minute before the source was edited
		shift(&outputs.join("scenes/yard.cscene"), Duration::from_mins(1), true);
		// and something that is not an asset at all
		put(&sources.join("notes.txt"), b"nothing");

		let entries = scan(&sources, &outputs);

		let names: Vec<&str> = entries
			.iter()
			.map(|entry| entry.name.as_str())
			.collect();
		assert_eq!(names, vec!["meshes/crystal", "scenes/yard", "textures/wall"], "by name");
		assert_eq!(entries[0].kind, Kind::Mesh);
		assert_eq!(entries[0].state, State::Compiled, "the mesh has its output");
		assert_eq!(entries[1].kind, Kind::Scene);
		assert_eq!(entries[1].state, State::Stale, "the scene's output is older than it");
		assert_eq!(entries[2].kind, Kind::Texture);
		assert_eq!(entries[2].state, State::Uncompiled, "the picture has none");
		assert_eq!(entries[0].output, outputs.join("meshes").join("crystal.cmesh"));
	}

	#[test]
	fn a_tree_that_is_not_there_is_an_empty_list() {
		let (sources, outputs) = fresh("absent");
		fs::remove_dir_all(&sources).expect("taken away");

		assert!(scan(&sources, &outputs).is_empty());
	}

	#[test]
	fn the_text_kinds_are_the_ones_a_code_editor_can_show() {
		assert!(is_text(Kind::Script), "a program");
		assert!(is_text(Kind::Mesh), "an .obj is a list of numbers as text");
		assert!(is_text(Kind::Material));
		assert!(is_text(Kind::Document));
		assert!(!is_text(Kind::Texture), "and a picture is not");
		assert!(!is_text(Kind::Sound));
		assert!(!is_text(Kind::Font));
		assert!(
			!is_text(Kind::Model),
			"a .gltf is text and a .glb is not, and one kind cannot answer both"
		);
		assert!(!is_text(Kind::Scene), "a scene row already means open the scene");
	}

	#[test]
	fn the_words_are_the_short_ones_a_row_has_room_for() {
		assert_eq!(word(Kind::Mesh), "mesh");
		assert_eq!(word(Kind::Script), "program");
		assert_eq!(State::Compiled.word(), "", "the usual case says nothing");
		assert_eq!(State::Uncompiled.word(), "not compiled");
	}
}
