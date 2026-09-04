//! Putting a project's game crate where the engine's workspace can build it.
//!
//! A game module has to be built in the engine's own cargo workspace, and the
//! reason is not taste. The module and `colby.exe` share one `colby_core.dll`,
//! and a crate's symbol names carry a hash of everything cargo resolved for it:
//! the features its dependencies ended up with, the profile, the versions. A
//! crate built on its own resolves those differently from a crate built beside
//! `colby_engine` and the rest, so a module built on its own links against a
//! `colby_core` the running process does not have. Measured, 2026-09-04: the
//! same crate built by manifest path into the same `target/` recompiled
//! `colby_core` and eight of its dependencies; built as a member, it compiled
//! itself and nothing else.
//!
//! So a project is **mounted**: `<engine>/projects/<id>` is made to point at
//! the project's directory, and the workspace's `members` glob
//! `projects/*/game` picks the crate up. On Windows that is a junction, which
//! needs no privilege; elsewhere a symbolic link. A project that already lives
//! under `projects/` - the place the launcher will create them - needs nothing,
//! and neither does the engine's own project, whose crate is a member on its
//! own.
//!
//! **A mount that points at nothing breaks every cargo command in the engine
//! checkout**, because the glob matches a directory whose manifest cannot be
//! read. That is why [`sweep`] runs before anything is built: a project that
//! was moved or deleted leaves a dead mount behind, and the engine takes it
//! away rather than failing on it.
//!
//! The cost, written down where it is paid: while a project from elsewhere is
//! mounted, the engine checkout's `Cargo.lock` carries one entry for its game
//! crate. @ref [[colby-pre-commit-audit]] in memory for the alternatives that
//! were priced.

use std::{
	fs, io,
	path::{Path, PathBuf},
};

use colby_asset::{
	Project,
	project::{GAME_DIR, MOUNTS_DIR},
};
use colby_core::{Result, err, info, warn};

/// Makes sure the engine's workspace can build the project's game crate.
///
/// @param project - the project, which has a game crate
/// @param engine - the engine checkout
/// @return where the crate is reached from, or nothing when it already is a
/// member: the engine's own project, or one that lives under `projects/`
///
/// # Errors
///
/// If the crate is not at `game/` - the glob knows one place - if
/// `projects/<id>` is already another directory, or if the link cannot be made.
pub(crate) fn mount(project: &Project, engine: &Path) -> Result<Option<PathBuf>> {
	let root = real(project.root())?;
	let engine = real(engine)?;

	if root.starts_with(&engine) {
		return Ok(None);
	}

	if project.game_dir() != Some(Path::new(GAME_DIR)) {
		return Err(err!(Module(
			"a project built by this engine keeps its game crate at {GAME_DIR}/, and {} keeps \
			 it at {}",
			project.id(),
			project
				.game_dir()
				.map_or_else(|| "nowhere".to_owned(), |dir| dir.display().to_string())
		)));
	}

	let mount = project.mount(&engine);

	if let Ok(existing) = real(&mount) {
		if existing == root {
			return Ok(Some(mount));
		}

		return Err(err!(Module(
			"{} is already {}, which is not this project",
			mount.display(),
			existing.display()
		)));
	}

	fs::create_dir_all(engine.join(MOUNTS_DIR))?;
	link(&mount, &root)?;
	info!(project = project.id(), mount = %mount.display(), "mounted for the engine's workspace");

	Ok(Some(mount))
}

/// Takes away every mount whose project has gone.
///
/// Called before anything is built, because a dead one fails every cargo
/// command in the checkout rather than only the build of the project it was.
///
/// @param engine - the engine checkout
pub(crate) fn sweep(engine: &Path) {
	let Ok(entries) = fs::read_dir(engine.join(MOUNTS_DIR)) else {
		return;
	};

	for entry in entries.flatten() {
		let path = entry.path();

		// a directory that is really here is a project that lives here; a link
		// is a mount, and a mount whose target cannot be read is dead.
		let Ok(link) = fs::symlink_metadata(&path) else {
			continue;
		};

		if !link.file_type().is_symlink() || fs::metadata(&path).is_ok() {
			continue;
		}

		match unlink(&path) {
			| Ok(()) =>
				warn!(mount = %path.display(), "a mount whose project is gone; taken away"),
			| Err(error) =>
				warn!(mount = %path.display(), %error, "a dead mount could not be taken away"),
		}
	}
}

/// A path as the filesystem has it, with every link followed.
fn real(path: &Path) -> Result<PathBuf> {
	fs::canonicalize(path)
		.map_err(|error| err!(Module("{} cannot be resolved: {error}", path.display())))
}

/// Points `at` at `target`, the way this platform points a directory somewhere.
#[cfg(windows)]
fn link(at: &Path, target: &Path) -> Result {
	// a junction rather than a symbolic link: the second wants a privilege an
	// ordinary account does not have, and the first is what every tool here
	// already follows. `mklink` is the one thing that makes one without a
	// binding of its own.
	let status = std::process::Command::new("cmd")
		.args(["/c", "mklink", "/J"])
		.arg(at)
		.arg(target)
		.stdout(std::process::Stdio::null())
		.status()
		.map_err(|error| err!(Module("mklink could not be run: {error}")))?;

	if !status.success() {
		return Err(err!(Module(
			"mklink refused to point {} at {}: {status}",
			at.display(),
			target.display()
		)));
	}

	Ok(())
}

/// Points `at` at `target`, the way this platform points a directory somewhere.
#[cfg(not(windows))]
fn link(at: &Path, target: &Path) -> Result {
	std::os::unix::fs::symlink(target, at)
		.map_err(|error| err!(Module("{} could not be linked: {error}", at.display())))
}

/// Takes a mount away without touching what it pointed at.
fn unlink(at: &Path) -> io::Result<()> { fs::remove_dir(at) }

#[cfg(test)]
mod tests {
	use std::env;

	use super::*;

	/// A directory nothing else is using.
	fn fresh(name: &str) -> PathBuf {
		let dir = env::temp_dir().join(format!("colby_mount_{name}"));
		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(&dir).expect("a directory to work in");

		dir
	}

	/// A project with a game crate directory, described by a file.
	fn project(root: &Path, id: &str, game: Option<&str>) -> Project {
		let game = game.map_or_else(String::new, |dir| format!(", \"game\": \"{dir}\""));
		let text = format!(
			r#"{{ "schema": 1, "engine": "0.1.0", "id": "{id}", "name": "{id}"{game} }}"#
		);

		if let Some(dir) = Project::parse(root, &text)
			.ok()
			.and_then(|it| it.game())
		{
			fs::create_dir_all(dir).expect("the crate directory");
		}

		Project::parse(root, &text).expect("a project")
	}

	#[test]
	fn a_project_elsewhere_is_mounted_and_a_second_mount_is_the_same_one() {
		let engine = fresh("engine");
		let elsewhere = fresh("elsewhere");
		let project = project(&elsewhere, "demo", Some("game"));

		let mounted = mount(&project, &engine)
			.expect("it mounts")
			.expect("and something was mounted");

		assert_eq!(
			mounted,
			engine
				.canonicalize()
				.expect("real")
				.join("projects")
				.join("demo")
		);
		assert!(
			fs::metadata(mounted.join("game")).is_ok(),
			"the crate is reached through the mount"
		);

		let again = mount(&project, &engine).expect("mounting twice");

		assert_eq!(again.as_deref(), Some(mounted.as_path()), "the same mount, kept");

		unlink(&mounted).expect("taken away");
		assert!(fs::metadata(elsewhere.join("game")).is_ok(), "and the project is untouched");
	}

	#[test]
	fn a_project_under_the_engine_needs_no_mount() {
		let engine = fresh("engine_own");
		let own = project(&engine, "own", Some("src/game"));

		assert_eq!(mount(&own, &engine).expect("nothing to do"), None, "the engine's own");

		let under = engine.join("projects").join("under");
		fs::create_dir_all(&under).expect("a project living where the launcher puts them");
		let under = project(&under, "under", Some("game"));

		assert_eq!(mount(&under, &engine).expect("nothing to do"), None, "already a member");
	}

	#[test]
	fn a_crate_anywhere_but_game_cannot_be_mounted() {
		let engine = fresh("engine_odd");
		let elsewhere = fresh("elsewhere_odd");
		let odd = project(&elsewhere, "odd", Some("src/game"));

		let text = mount(&odd, &engine)
			.expect_err("the glob knows one place")
			.to_string();

		assert!(text.contains("game/"), "it says where: {text}");
	}

	#[test]
	fn a_mount_taken_by_another_project_is_refused() {
		let engine = fresh("engine_taken");
		let first = fresh("first_taken");
		let second = fresh("second_taken");
		let one = project(&first, "same", Some("game"));
		let two = project(&second, "same", Some("game"));

		let kept = mount(&one, &engine)
			.expect("the first mounts")
			.expect("and is mounted");
		let text = mount(&two, &engine)
			.expect_err("the second is somebody else")
			.to_string();

		assert!(text.contains("already"), "got {text}");

		unlink(&kept).expect("taken away");
	}

	#[test]
	fn a_mount_whose_project_is_gone_is_swept_and_a_live_one_is_kept() {
		let engine = fresh("engine_sweep");
		let gone = fresh("gone_sweep");
		let alive = fresh("alive_sweep");

		mount(&project(&gone, "gone", Some("game")), &engine).expect("mounted");
		mount(&project(&alive, "alive", Some("game")), &engine).expect("mounted");
		fs::remove_dir_all(&gone).expect("the project goes away");

		sweep(&engine);

		let mounts = engine.join("projects");

		assert!(fs::symlink_metadata(mounts.join("gone")).is_err(), "the dead mount is gone");
		assert!(
			fs::metadata(mounts.join("alive").join("game")).is_ok(),
			"and the live one is not"
		);

		unlink(&mounts.join("alive")).expect("taken away");
	}
}
