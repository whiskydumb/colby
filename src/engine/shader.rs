//! Shader source, and noticing when someone edits it.
//!
//! The source is compiled into the binary *and* read from disk when the file is
//! there. The baked copy is what a shipped build runs; the file is what a
//! developer edits, and the two are the same bytes at build time. Nothing has
//! to be shipped for this to work and nothing has to be present for it to run.
//!
//! Reloading a shader is not like reloading a game module: there is no image to
//! unload, no state to keep and nothing mapped that the linker wants. It is a
//! string, a `create_shader_module` and a `create_render_pipeline`. The only
//! part worth care is that a shader which does not compile must cost a log line
//! rather than the process - @ref
//! [`Scene::set_shader`](crate::Scene::set_shader) for where that is arranged.

use std::{
	env, fs,
	path::{Path, PathBuf},
	time::{Duration, Instant, SystemTime},
};

use colby_core::{debug, warn};

/// How often the file is looked at.
///
/// The same quarter second the module and asset watchers use.
const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Overrides the directory shaders are read from.
///
/// Without it the directory is `src/engine`, baked in at build time - which
/// exists on the machine the engine was built on and nowhere else, so a copy
/// running somewhere else quietly uses the built-in source.
const DIRECTORY_VAR: &str = "COLBY_SHADERS";

/// One shader's source, and the file it came from if there was one.
#[derive(Clone, Debug)]
pub struct Shader {
	/// Where to look. `None` when there is no directory to look in.
	path: Option<PathBuf>,
	source: String,
	/// The mtime of the file the current source was read from, if any.
	stamp: Option<SystemTime>,
	next_poll: Instant,
}

impl Shader {
	/// Reads a shader, preferring the file on disk over the baked copy.
	///
	/// @param name - the file name, e.g. `shader.wgsl`
	/// @param built_in - the same file's contents, `include_str!`d by the
	/// caller
	#[must_use]
	pub fn new(name: &str, built_in: &str) -> Self {
		Self::at(Path::new(env!("CARGO_MANIFEST_DIR")), name, built_in)
	}

	/// The same, for a shader that lives beside another crate's sources.
	///
	/// The engine's own directory is baked in by [`new`](Self::new), which is
	/// the wrong one for a shader belonging to the interface:
	/// `CARGO_MANIFEST_DIR` is resolved where it is written, not where it is
	/// called from. A crate that owns a shader passes its own.
	///
	/// @param fallback - where to look when `COLBY_SHADERS` is not set
	/// @param name - the file name, e.g. `ui.wgsl`
	/// @param built_in - the same file's contents, `include_str!`d by the
	/// caller
	#[must_use]
	pub fn at(fallback: &Path, name: &str, built_in: &str) -> Self {
		let mut shader = Self {
			path: directory(fallback).map(|dir| dir.join(name)),
			source: built_in.to_owned(),
			stamp: None,
			next_poll: Instant::now() + POLL_INTERVAL,
		};

		if shader.read() {
			debug!(path = ?shader.path, "shader read from disk; edits to it will reload");
		} else {
			debug!(name, "no shader file to read; using the copy built into the binary");
		}

		shader
	}

	/// The current source.
	#[must_use]
	pub fn source(&self) -> &str { &self.source }

	/// The file being watched, if there is one.
	#[must_use]
	pub fn path(&self) -> Option<&Path> { self.path.as_deref() }

	/// Re-reads the file if it has been written since the last read.
	///
	/// @return `true` when [`source`](Self::source) now holds something new
	pub fn changed(&mut self) -> bool {
		if Instant::now() < self.next_poll {
			return false;
		}

		self.next_poll = Instant::now() + POLL_INTERVAL;

		let Some(stamp) = self.path.as_deref().and_then(mtime) else {
			return false;
		};

		// `is_some_and` rather than a comparison against a default: a shader
		// that fell back to the built-in has no stamp, and a file appearing
		// later should be picked up rather than compared against nothing.
		if self
			.stamp
			.is_some_and(|previous| stamp <= previous)
		{
			return false;
		}

		self.read()
	}

	/// Reads the file into [`source`](Self::source).
	///
	/// @return `false` when there is no file, or it could not be read
	fn read(&mut self) -> bool {
		let Some(path) = self.path.as_ref() else {
			return false;
		};

		let Some(stamp) = mtime(path) else {
			return false;
		};

		match fs::read_to_string(path) {
			| Ok(source) => {
				self.source = source;
				self.stamp = Some(stamp);

				true
			},
			| Err(error) => {
				// mid-write, most likely. The stamp is left alone so the next
				// poll tries again.
				warn!(?path, %error, "could not read the shader; keeping the last good source");

				false
			},
		}
	}
}

/// The directory shaders are read from, if there is one.
fn directory(fallback: &Path) -> Option<PathBuf> {
	if let Some(directory) = env::var_os(DIRECTORY_VAR) {
		return Some(PathBuf::from(directory));
	}

	// baked in at build time by whichever crate owns the shader. Present on the
	// machine the engine was built on, absent everywhere else, which is exactly
	// the distinction wanted.
	fallback.is_dir().then(|| fallback.to_path_buf())
}

/// A file's modification time, if it can be read.
fn mtime(path: &Path) -> Option<SystemTime> { path.metadata().ok()?.modified().ok() }

#[cfg(test)]
mod tests {
	use super::*;

	/// Something that is unmistakably not the built-in source.
	const REPLACEMENT: &str = "// replaced\n";

	/// A directory nobody else is using, with a shader in it.
	fn fixture(name: &str, source: &str) -> PathBuf {
		let dir = env::temp_dir()
			.join("colby-shader-tests")
			.join(name);

		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(&dir).expect("the fixture is made");
		fs::write(dir.join("shader.wgsl"), source).expect("the shader is written");

		dir
	}

	/// A shader watching a fixture directory, without touching the environment.
	fn watching(dir: &Path, built_in: &str) -> Shader {
		let mut shader = Shader {
			path: Some(dir.join("shader.wgsl")),
			source: built_in.to_owned(),
			stamp: None,
			next_poll: Instant::now(),
		};
		shader.read();

		shader
	}

	#[test]
	fn a_file_on_disk_wins_over_the_baked_copy() {
		let dir = fixture("prefers-the-file", REPLACEMENT);

		assert_eq!(
			watching(&dir, "// built in\n").source(),
			REPLACEMENT,
			"the file is what a developer edits, so it is what is used"
		);
	}

	#[test]
	fn a_missing_file_falls_back_to_the_baked_copy() {
		let mut shader = Shader {
			path: Some(env::temp_dir().join("colby-no-such-shader.wgsl")),
			source: "// built in\n".to_owned(),
			stamp: None,
			next_poll: Instant::now(),
		};

		assert!(!shader.read(), "there is nothing to read");
		assert_eq!(shader.source(), "// built in\n", "so the binary's own copy stands");
		assert!(!shader.changed(), "and nothing keeps reporting a change");
	}

	#[test]
	fn a_shader_with_no_directory_at_all_is_the_baked_copy() {
		let mut shader = Shader {
			path: None,
			source: "// built in\n".to_owned(),
			stamp: None,
			next_poll: Instant::now(),
		};

		assert!(shader.path().is_none(), "nothing to watch");
		assert!(!shader.changed(), "so nothing ever changes");
		assert_eq!(shader.source(), "// built in\n", "and the source never moves");
	}

	#[test]
	fn writing_the_file_is_noticed_once() {
		let dir = fixture("notices-a-write", "// first\n");
		let mut shader = watching(&dir, "// built in\n");

		assert_eq!(shader.source(), "// first\n", "the first read took the file");
		assert!(!shader.changed(), "and nothing has happened since");

		// the poll is rate limited and the mtime rule is `newer than`; both
		// need a moment that a pair of back-to-back writes does not give.
		std::thread::sleep(POLL_INTERVAL + Duration::from_millis(30));
		fs::write(dir.join("shader.wgsl"), REPLACEMENT).expect("the shader is rewritten");

		assert!(shader.changed(), "the write is noticed");
		assert_eq!(shader.source(), REPLACEMENT, "and the new source is in hand");
		assert!(!shader.changed(), "the same write is not noticed twice");
	}

	#[test]
	fn a_file_that_appears_later_is_picked_up() {
		let dir = fixture("appears-later", "// first\n");
		let path = dir.join("shader.wgsl");
		fs::remove_file(&path).expect("the fixture starts with nothing");

		let mut shader = watching(&dir, "// built in\n");

		assert_eq!(shader.source(), "// built in\n", "nothing to read yet");

		std::thread::sleep(POLL_INTERVAL + Duration::from_millis(30));
		fs::write(&path, REPLACEMENT).expect("the shader is written");

		assert!(shader.changed(), "a shader with no stamp compares against nothing");
		assert_eq!(shader.source(), REPLACEMENT, "and takes the file");
	}
}
