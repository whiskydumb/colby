//! Watches the workspace and keeps the game module image fresh.
//!
//! Two independent signals, on purpose. A change under `src/game` starts a
//! build; a change to the built module image is what actually triggers a
//! reload. Keeping them separate means a rebuild started from another terminal,
//! or an editor with its own build-on-save, reloads just the same - the runner
//! is not the only thing allowed to produce the file.
//!
//! There is a third signal that stops rather than starts anything. A change to
//! any crate *below* the game - core, engine, the runner itself - cannot be
//! swapped into a running process: the executable and `colby_core.dll` are
//! mapped and Windows will not let the linker replace them. Some hosts name a
//! threshold layer for this; here the layer below the game is simply all of
//! it.

use std::{
	fs,
	path::{Path, PathBuf},
	process::{Child, Command},
	time::{Duration, Instant, SystemTime},
};

use colby_core::{Result, debug, error, info, mods::path, warn};

use crate::Build;

/// How often the filesystem is looked at. Frequent enough to feel immediate,
/// rare enough that a few dozen `stat` calls per second are irrelevant.
const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// File names worth reacting to.
///
/// @note: `wgsl` was here while `shader.wgsl` was only ever `include_str!`d,
/// because editing it really did mean a rebuild and a restart. It reads the
/// file at runtime now, so a shader edit is picked up by the renderer's own
/// watcher and leaving it here would print a restart warning that is no longer
/// true. @ref [`Shader`](colby_engine::Shader).
const WATCHED_EXTENSIONS: &[&str] = &["rs", "toml"];

/// Crate directories a change to forces a restart.
///
/// Everything the runner itself links, which is every crate in the workspace
/// but two: the game, which is the one being swapped, and `assetc`, a separate
/// executable nothing in this process depends on, so rebuilding it is free.
///
/// @note: the list held four names for a long time - `asset`, `core`, `engine`
/// and the runner - and an edit under the six others went unremarked: no
/// warning, no rebuild, and a process quietly running the solver it started
/// with. Every crate the runner links belongs here, or the warning lies.
const FIXED_CRATES: &[&str] = &[
	"asset", "audio", "core", "editor", "engine", "main", "net", "physics", "runtime", "script",
	"ui",
];

/// A set of directories and the newest modification time seen across them.
struct Layer {
	roots: Vec<PathBuf>,
	stamp: SystemTime,
}

impl Layer {
	/// Starts watching a set of directories from their current state.
	fn new(roots: Vec<PathBuf>, fallback: SystemTime) -> Self {
		let stamp = newest_mtime(&roots).unwrap_or(fallback);

		Self { roots, stamp }
	}

	/// Answers whether anything under these directories has been written since
	/// the last time this returned `true`.
	fn changed(&mut self) -> bool {
		let Some(stamp) = newest_mtime(&self.roots) else {
			return false;
		};

		if stamp <= self.stamp {
			return false;
		}

		self.stamp = stamp;

		true
	}
}

/// The rebuild-and-reload loop's view of the filesystem.
pub(crate) struct Watch {
	game: Layer,
	fixed: Layer,
	artifact: PathBuf,
	artifact_stamp: SystemTime,
	build: Option<Child>,
	next_poll: Instant,
	stale_host: bool,
	/// What the build script knew: the cargo, the profile and the flags a
	/// rebuild has to match, and the package it must leave alone.
	facts: Build,
}

impl Watch {
	/// Starts watching the workspace a module is built from.
	///
	/// @param module - the crate name, e.g. `colby_game`
	/// @param sources - the directory holding that crate's sources
	/// @param facts - what the build script knew, so that the rebuild matches
	/// the build that is running
	/// @return a watcher primed with the current state, so nothing fires until
	/// something is actually written
	pub(crate) fn new(module: &str, sources: PathBuf, facts: &Build) -> Result<Self> {
		let artifact = path::from_name(module)?;
		let artifact_stamp = path::mtime(&artifact)?;

		let fixed = FIXED_CRATES
			.iter()
			.map(|crate_dir| facts.engine.join("src").join(crate_dir))
			.collect();

		info!(sources = ?sources, artifact = ?artifact, "watching the game crate");

		Ok(Self {
			game: Layer::new(vec![sources], artifact_stamp),
			fixed: Layer::new(fixed, artifact_stamp),
			artifact,
			artifact_stamp,
			build: None,
			next_poll: Instant::now() + POLL_INTERVAL,
			stale_host: false,
			facts: facts.clone(),
		})
	}

	/// Looks at the filesystem, and starts a build if the game moved.
	///
	/// @return `true` when a newly built module image is waiting to be loaded
	pub(crate) fn poll(&mut self) -> bool {
		if Instant::now() < self.next_poll {
			return false;
		}

		self.next_poll = Instant::now() + POLL_INTERVAL;
		self.reap();

		if self.artifact_changed() {
			return true;
		}

		self.note_fixed_changes();
		self.start_build();

		false
	}

	/// Collects a finished build and reports its outcome.
	fn reap(&mut self) {
		let Some(build) = self.build.as_mut() else {
			return;
		};

		match build.try_wait() {
			| Ok(None) => {},
			| Ok(Some(status)) => {
				self.build = None;
				if status.success() {
					debug!("game rebuilt");
				} else {
					warn!(%status, "rebuilding the game failed; the running module is unchanged");
				}
			},
			| Err(error) => {
				self.build = None;
				error!(%error, "waiting on the build process failed");
			},
		}
	}

	/// Answers whether the module image on disk is newer than the loaded one.
	fn artifact_changed(&mut self) -> bool {
		let Ok(stamp) = path::mtime(&self.artifact) else {
			return false;
		};

		if stamp <= self.artifact_stamp {
			return false;
		}

		self.artifact_stamp = stamp;

		true
	}

	/// Notices edits below the game layer and says so, once.
	///
	/// Building after one of these is pointless: cargo would have to relink
	/// `colby.exe` or `colby_core.dll`, both of which this process has mapped,
	/// and Windows answers that with `Access is denied`. Saying so plainly
	/// beats letting a linker error scroll past.
	fn note_fixed_changes(&mut self) {
		if !self.fixed.changed() || self.stale_host {
			return;
		}

		self.stale_host = true;
		warn!("a crate below the game changed; restart colby to pick it up");
	}

	/// Starts a build if the game sources moved.
	fn start_build(&mut self) {
		if self.build.is_some() || !self.game.changed() || self.stale_host {
			return;
		}

		info!("game sources changed, rebuilding");

		// @note: the whole workspace, not `--package colby_game`. Restricting
		// the build changes the unit graph - wgpu drops out, and with it the
		// features it turns on in crates colby_core also uses - which makes
		// cargo consider colby_core dirty and try to relink the dll this
		// process is running from. Building what `just hot` builds keeps the
		// fingerprints identical, so only the game is recompiled.
		//
		// The runner itself is excluded all the same. It can never be replaced
		// while it is running, and cargo attempting it turns a working reload
		// into `Access is denied` on the console. Excluding a binary nothing
		// else depends on does not disturb feature resolution.
		//
		// The child inherits this build's exact rustflags, because a module
		// built without `-Cprefer-dynamic` would carry its own std and its own
		// colby_core and quietly break every property the host relies on. It
		// inherits stdio too: a compile error belongs in front of whoever just
		// saved the file.
		let spawned = Command::new(&self.facts.cargo)
			.args(["build", "--workspace", "--exclude", self.facts.package.as_str()])
			.args(["--profile", self.facts.profile.as_str()])
			.current_dir(&self.facts.engine)
			.env("CARGO_ENCODED_RUSTFLAGS", &self.facts.rustflags)
			.spawn();

		match spawned {
			| Ok(child) => self.build = Some(child),
			| Err(error) => error!(%error, "could not start cargo"),
		}
	}
}

/// The most recent modification time under a set of directories.
///
/// Only files worth reacting to are considered, @ref [`WATCHED_EXTENSIONS`]. A
/// directory that cannot be read is skipped rather than raised: this is a poll,
/// and the next one is 250ms away.
///
/// @param roots - the directories to walk
/// @return the newest mtime found, if any
fn newest_mtime(roots: &[PathBuf]) -> Option<SystemTime> {
	let mut newest: Option<SystemTime> = None;
	let mut pending: Vec<PathBuf> = roots.to_vec();

	while let Some(dir) = pending.pop() {
		let Ok(entries) = fs::read_dir(&dir) else {
			continue;
		};

		for entry in entries.flatten() {
			let path = entry.path();
			if path.is_dir() {
				pending.push(path);
				continue;
			}

			if !watched(&path) {
				continue;
			}

			let Ok(stamp) = path::mtime(&path) else {
				continue;
			};

			if newest.is_none_or(|previous| stamp > previous) {
				newest = Some(stamp);
			}
		}
	}

	newest
}

/// Answers whether a change to this file should be reacted to.
fn watched(path: &Path) -> bool {
	path.extension()
		.and_then(|extension| extension.to_str())
		.is_some_and(|extension| WATCHED_EXTENSIONS.contains(&extension))
}
