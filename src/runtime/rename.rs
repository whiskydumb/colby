//! Renaming an asset, with everything that stands beside it.
//!
//! **Why this is a command and not a file-manager gesture.** A source carries
//! up to two sidecars - its identity (`crystal.obj.id`) and, for geometry, the
//! settings it was imported with (`crystal.obj.model`) - and both are named
//! after the whole of the file they stand beside. Move the source with a file
//! manager and they are left behind: the import settings stop applying, and the
//! identity is *lost*, which means every scene that named the asset by identity
//! now names nothing and refuses to compile. The one gesture that is safe is
//! the one that moves all three, and that is what this is.
//!
//! It is the same rule the field arrived at. Godot's file system dock moves
//! `.import` and `.uid` beside the file it is renaming
//! (`editor/docks/filesystem_dock.cpp:1578-1590`, `6ef60dc`, 2026-09-02), and
//! s&box's asset list moves the compiled `_c` and `_d` files beside theirs
//! (`AssetList/Entries/AssetEntry.cs:174-189`, `eb3e746`, 2026-08-30). colby
//! moves no compiled file: the sweep deletes the one whose source is gone and
//! the next pass writes the new one, which is the same outcome with nothing to
//! keep in step.
//!
//! **The new name is one component and the directory does not change.** Godot
//! refuses `/`, `\` and `:` in a rename (`filesystem_dock.cpp:1889`) and s&box
//! splits the extension off and puts it back (`AssetList.ContextMenu.cs:510`),
//! and both are right for the same reason: moving a file between directories is
//! a different operation with different consequences - a document's stylesheet
//! links are relative to where it stands - and a rename that quietly did it
//! would be a rename that quietly broke them.
//!
//! **There is no undo**, and neither engine has one either: Godot's
//! `_try_move_item` calls `DirAccess::rename` with no undo action anywhere near
//! it, and s&box's `Rename` calls `FileInfo.MoveTo`. The editor's history is a
//! stack of *worlds*, and a rename happens outside every world there is; the
//! undo of a rename is a rename back, which is the same gesture.
//!
//! **A world that is already running follows it.** The compiler gives the new
//! name the identity the old one had, the asset loop hands that identity to the
//! table, and the table moves the entry rather than appending a second one - so
//! every handle every entity is holding goes on resolving. @ref
//! [`Registry::adopt`](colby_core::abi::registry::Registry::adopt).

use std::{fs, path::Path};

use colby_asset::{Project, compile, ident, import};
use colby_core::{
	Result,
	abi::{Asked, World},
	err, error, info,
};

/// `asset.rename <name> <to>` - renames a source and its sidecars.
pub(crate) const RENAME: &str = "asset.rename";

/// The names this module answers for, as they wait on the world.
const NAMES: &[&str] = &[RENAME];

/// One rename, as a console line asked for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Request {
	/// The asset name as it stands, `meshes/crystal`.
	pub(crate) name: String,

	/// What the last part of it is to become, `gem`.
	pub(crate) to: String,
}

impl Request {
	/// What one waiting line asks for, if it is this module's.
	///
	/// **The last word is the new name and everything before it is the old
	/// one.** An asset name is a path and a path may hold a space, so a command
	/// with two names in it has to choose which of them may; this chooses the
	/// old one, because that is the one a person did not type. A new name
	/// holding a space is refused rather than mangled - @ref [`renamed`].
	///
	/// @param asked - a line the frame loop took off the world
	fn of(asked: &Asked) -> Option<Self> {
		if asked.name != RENAME {
			return None;
		}

		let (to, rest) = asked.words.split_last()?;

		if rest.is_empty() {
			return None;
		}

		Some(Self { name: rest.join(" "), to: to.clone() })
	}
}

/// Renames whatever a command asked for, if one did.
///
/// The frame loop's, beside [`crate::code::serve`] and for its reason: a
/// [`ConsoleFn`](colby_core::abi::ConsoleFn) is handed a world and nothing
/// else, and finding a file needs the project.
///
/// @param world - where the lines wait
/// @param project - whose asset tree
pub(crate) fn serve(world: &mut World, project: &Project) {
	for asked in crate::console::take(world, NAMES) {
		let Some(request) = Request::of(&asked) else {
			error!(line = RENAME, "a rename takes the asset name and then the new one");

			continue;
		};

		match renamed(&project.assets(), &request) {
			| Ok(name) => info!(was = request.name, now = name, "asset renamed"),
			| Err(failure) => error!(%failure, "the asset could not be renamed"),
		}
	}
}

/// Moves a source and the sidecars beside it.
///
/// @param root - the source tree, `<project>/assets`
/// @param request - what to rename and to what
/// @return the asset name it is known by now
///
/// # Errors
///
/// If the new name is not one component, if nothing under that name is in the
/// tree, if something already stands where it would go, or if the move fails.
pub(crate) fn renamed(root: &Path, request: &Request) -> Result<String> {
	let to = request.to.trim();

	if to.is_empty() {
		return Err(err!(Asset("a rename needs a name to rename to")));
	}

	// a space is refused rather than allowed, because a console line is words
	// and a name with a space in one cannot be told from the space between two
	// of them. The rest are refused for Godot's reason: a rename that moved a
	// file to another directory would be a rename that broke every relative
	// link in it.
	if let Some(bad) = to
		.chars()
		.find(|it| matches!(it, '/' | '\\' | ':' | ' '))
	{
		return Err(err!(Asset(
			"{to:?} is not a name: a rename changes the last part of an asset's name and not \
			 where it stands, so {bad:?} has no place in one"
		)));
	}

	// Godot's other rule, and the same words for it: a name that begins with a
	// dot is a file the editor cannot see, and `.` and `..` are not names at
	// all (`filesystem_dock.cpp:1892`).
	if to.starts_with('.') {
		return Err(err!(Asset(
			"{to:?} is not a name: one that begins with a dot is a file nothing in the editor \
			 lists, so a rename does not make one"
		)));
	}

	let Some(source) = compile::source_of(root, &request.name) else {
		return Err(err!(Asset(
			"nothing under assets/ is called {}, so there is nothing to rename",
			request.name
		)));
	};

	let extension = source
		.extension()
		.map(|held| held.to_string_lossy().into_owned())
		.unwrap_or_default();
	let target = source.with_file_name(format!("{to}.{extension}"));

	if target == source {
		return Ok(request.name.clone());
	}

	// every file that would move, checked before any of them does: a rename
	// half done is an asset with its identity beside a file that is not there.
	let moves = [
		(source.clone(), target.clone()),
		(ident::beside(&source), ident::beside(&target)),
		(import::beside(&source), import::beside(&target)),
	];

	for (was, now) in &moves {
		if now.exists() {
			return Err(err!(Asset(
				"{} is already there; two files cannot share one asset name",
				now.display()
			)));
		}

		if was == &source && !was.is_file() {
			return Err(err!(Asset("{} is not a file", was.display())));
		}
	}

	for (was, now) in &moves {
		if !was.is_file() {
			continue;
		}

		fs::rename(was, now).map_err(|error| {
			err!(Asset("{} could not be moved to {}: {error}", was.display(), now.display()))
		})?;
	}

	compile::asset_name(root, &target)
}

#[cfg(test)]
mod tests {
	use std::{env, path::PathBuf};

	use colby_core::abi::{Aim, PeerId};

	use super::*;

	/// A scratch source tree with one mesh, its identity and its settings.
	fn tree(name: &str) -> PathBuf {
		let root = env::temp_dir()
			.join("colby-rename-tests")
			.join(name);

		drop(fs::remove_dir_all(&root));
		fs::create_dir_all(root.join("meshes")).expect("a tree to work in");
		fs::write(root.join("meshes").join("crystal.obj"), b"v 0 0 0\n").expect("the source");
		fs::write(root.join("meshes").join("crystal.obj.id"), b"id://abcdefghijklm\n")
			.expect("its identity");
		fs::write(root.join("meshes").join("crystal.obj.model"), b"{}").expect("its settings");

		root
	}

	/// What to rename, spelled out.
	fn asking(name: &str, to: &str) -> Request {
		Request { name: name.to_owned(), to: to.to_owned() }
	}

	#[test]
	fn a_rename_moves_the_source_and_both_sidecars_with_it() {
		let root = tree("whole");

		let now = renamed(&root, &asking("meshes/crystal", "gem")).expect("it renames");

		assert_eq!(now, "meshes/gem", "and says what it is called now");
		assert!(root.join("meshes/gem.obj").is_file(), "the source moved");
		assert!(root.join("meshes/gem.obj.id").is_file(), "the identity moved with it");
		assert!(root.join("meshes/gem.obj.model").is_file(), "and so did the settings");
		assert!(!root.join("meshes/crystal.obj").exists(), "and nothing was left behind");
		assert!(!root.join("meshes/crystal.obj.id").exists());
		assert!(!root.join("meshes/crystal.obj.model").exists());
		assert_eq!(
			fs::read_to_string(root.join("meshes/gem.obj.id")).expect("it reads"),
			"id://abcdefghijklm\n",
			"and the identity is the one it was"
		);
	}

	#[test]
	fn a_source_with_no_sidecars_renames_anyway() {
		let root = tree("bare");
		fs::remove_file(root.join("meshes/crystal.obj.id")).expect("taken away");
		fs::remove_file(root.join("meshes/crystal.obj.model")).expect("and the other");

		renamed(&root, &asking("meshes/crystal", "gem")).expect("it renames");

		assert!(root.join("meshes/gem.obj").is_file());
		assert!(!root.join("meshes/gem.obj.id").exists(), "and none was invented");
	}

	#[test]
	fn a_name_that_would_move_it_somewhere_else_is_refused() {
		let root = tree("elsewhere");

		for bad in ["props/gem", "..", "a:b", "two words", "  ", ""] {
			assert!(
				renamed(&root, &asking("meshes/crystal", bad)).is_err(),
				"{bad:?} is not a name a rename takes"
			);
		}

		assert!(root.join("meshes/crystal.obj").is_file(), "and nothing moved");
	}

	#[test]
	fn a_name_something_already_has_is_refused_before_anything_moves() {
		let root = tree("taken");
		fs::write(root.join("meshes/gem.obj"), b"v 1 1 1\n").expect("somebody is there");

		drop(renamed(&root, &asking("meshes/crystal", "gem")).unwrap_err());

		assert!(root.join("meshes/crystal.obj").is_file(), "the source stayed");
		assert!(root.join("meshes/crystal.obj.id").is_file(), "and so did its identity");
		assert_eq!(
			fs::read_to_string(root.join("meshes/gem.obj")).expect("it reads"),
			"v 1 1 1\n",
			"and the one that was there was not written over"
		);
	}

	#[test]
	fn a_sidecar_in_the_way_stops_the_whole_rename() {
		// the case the check before the loop exists for: the source could move
		// and its identity could not, which is an asset that has lost one.
		let root = tree("half");
		fs::write(root.join("meshes/gem.obj.id"), b"id://nnnnnnnnnnnnn\n").expect("in the way");

		drop(renamed(&root, &asking("meshes/crystal", "gem")).unwrap_err());

		assert!(root.join("meshes/crystal.obj").is_file(), "so nothing moved at all");
		assert!(root.join("meshes/crystal.obj.id").is_file());
	}

	#[test]
	fn a_name_nothing_is_called_is_refused() {
		let root = tree("nowhere");

		drop(renamed(&root, &asking("meshes/nothing", "gem")).unwrap_err());
	}

	#[test]
	fn renaming_something_to_what_it_is_called_does_nothing() {
		let root = tree("same");

		let now = renamed(&root, &asking("meshes/crystal", "crystal")).expect("it is fine");

		assert_eq!(now, "meshes/crystal");
		assert!(root.join("meshes/crystal.obj.id").is_file(), "and the sidecar is untouched");
	}

	#[test]
	fn a_line_is_the_new_name_at_the_end_and_the_old_one_in_front() {
		let held = |words: &[&str]| {
			Request::of(&Asked {
				name: RENAME.to_owned(),
				words: words.iter().map(|it| (*it).to_owned()).collect(),
				peer: PeerId::HOST,
				aim: Aim::NONE,
			})
		};

		assert_eq!(held(&["meshes/crystal", "gem"]), Some(asking("meshes/crystal", "gem")));
		assert_eq!(
			held(&["meshes/my crystal", "gem"]),
			Some(asking("meshes/my crystal", "gem")),
			"the old name is the one that may hold a space"
		);
		assert_eq!(held(&["meshes/crystal"]), None, "one word is not a rename");
		assert_eq!(held(&[]), None, "and neither is none");
	}
}
