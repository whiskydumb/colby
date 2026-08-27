//! Locating a module on disk, and staging a copy the loader may keep.

use std::{
	env::{current_exe, temp_dir},
	fs,
	path::{Path, PathBuf},
	process,
	thread::sleep,
	time::{Duration, SystemTime},
};

use libloading::library_filename;

use crate::{Err, Result, debug, err, trace};

/// How long to wait between two identical `stat`s before believing a freshly
/// linked file is finished.
const SETTLE_INTERVAL: Duration = Duration::from_millis(40);

/// How many times to re-check before giving up on a file that keeps changing.
const SETTLE_TRIES: u32 = 32;

/// Resolves a module name to the file the build wrote.
///
/// Modules sit beside the executable, which is where cargo puts every artifact
/// of a profile.
///
/// @param name - the crate name, without prefix or extension
/// @return the absolute path to `<exe dir>/<name>.dll`
pub fn from_name(name: &str) -> Result<PathBuf> {
	let exe = current_exe()?;
	let dir = exe
		.parent()
		.ok_or_else(|| err!("executable {exe:?} has no parent directory"))?;

	Ok(dir.join(library_filename(name)))
}

/// Recovers a module name from a path produced by [`from_name`].
///
/// @param path - a path to a module image
/// @return the crate name
pub fn to_name(path: &Path) -> Result<String> {
	path.file_stem()
		.and_then(|stem| stem.to_str())
		.map(ToOwned::to_owned)
		.ok_or_else(|| err!("module path {path:?} has no usable file stem"))
}

/// Reads a file's modification time.
///
/// @param path - the file to stat
/// @return its mtime
pub fn mtime(path: &Path) -> Result<SystemTime> { Ok(fs::metadata(path)?.modified()?) }

/// The directory staged module images live in, private to this process.
///
/// Keyed by process id so two engines running at once cannot fight over the
/// same file names.
#[must_use]
pub fn scratch_dir() -> PathBuf {
	temp_dir()
		.join("colby-mods")
		.join(process::id().to_string())
}

/// Copies a module image somewhere the build is free to overwrite the original.
///
/// Windows keeps a loaded image mapped and refuses writes to the file behind
/// it, so loading `target/hot/colby_game.dll` directly would make the next
/// `cargo build` fail with a locked-file error. Each generation gets its own
/// directory so the copy keeps its original file name, which is also what lets
/// a debugger find the `.pdb` staged next to it.
///
/// @param source - the file the build wrote
/// @param generation - a counter that increases with every reload
/// @return the path of the copy to hand to the loader
pub fn stage(source: &Path, generation: u64) -> Result<PathBuf> {
	settle(source)?;

	let dir = scratch_dir().join(generation.to_string());
	fs::create_dir_all(&dir)?;

	let name = source
		.file_name()
		.ok_or_else(|| err!("module path {source:?} has no file name"))?;

	let image = dir.join(name);
	fs::copy(source, &image)?;

	// best effort: a debugger looks for the pdb beside the image when the path
	// baked into the module no longer resolves.
	let symbols = source.with_extension("pdb");
	if symbols.is_file()
		&& let Some(name) = symbols.file_name()
	{
		drop(fs::copy(&symbols, dir.join(name)));
	}

	trace!(source = ?source, image = ?image, "staged module image");

	Ok(image)
}

/// Deletes a staged generation once its library has been unloaded.
///
/// Failure is not an error: the image may still be mapped for a moment after
/// `FreeLibrary` returns, and a leftover file in `%TEMP%` costs nothing.
///
/// @param image - a path previously returned by [`stage`]
pub fn unstage(image: &Path) {
	let Some(dir) = image.parent() else {
		return;
	};

	if let Err(error) = fs::remove_dir_all(dir) {
		debug!(dir = ?dir, %error, "could not remove staged module directory");
	}
}

/// Removes every staged image this process left behind.
///
/// Called at startup and at shutdown. A process that died without unloading its
/// modules leaves its whole scratch directory behind, and the next run with the
/// same pid would otherwise inherit it.
pub fn clear_scratch() {
	let dir = scratch_dir();
	if !dir.exists() {
		return;
	}

	if let Err(error) = fs::remove_dir_all(&dir) {
		debug!(dir = ?dir, %error, "could not clear the module scratch directory");
	}
}

/// Waits until a file stops changing.
///
/// The linker writes a module image in place rather than renaming a finished
/// one over it, so a rebuild is observable as a partially written file. Two
/// consecutive identical `(len, mtime)` readings are taken as done.
///
/// @param path - the file to watch
/// @return `Ok` once the file has held still for one interval
fn settle(path: &Path) -> Result {
	let read = |path: &Path| -> Result<(u64, SystemTime)> {
		let meta = fs::metadata(path)?;

		Ok((meta.len(), meta.modified()?))
	};

	let mut previous = read(path)?;
	for _ in 0..SETTLE_TRIES {
		sleep(SETTLE_INTERVAL);
		let current = read(path)?;
		if current == previous {
			return Ok(());
		}

		previous = current;
	}

	Err!("module image {path:?} is still being written after {SETTLE_TRIES} checks")
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_module_name_round_trips_through_its_path() {
		let path = from_name("colby_game").expect("the executable has a directory");

		assert_eq!(
			path.extension()
				.and_then(|extension| extension.to_str()),
			Some("dll"),
			"windows modules are dlls, with no lib prefix"
		);
		assert_eq!(
			to_name(&path).expect("the path has a stem"),
			"colby_game",
			"the name survives the trip to a path and back"
		);
	}

	#[test]
	fn the_scratch_directory_is_private_to_this_process() {
		let dir = scratch_dir();
		let leaf = dir
			.file_name()
			.and_then(|name| name.to_str())
			.expect("the scratch directory ends in the process id");

		assert_eq!(
			leaf.parse::<u32>().expect("the leaf is a number"),
			process::id(),
			"two engines at once must not share staged images"
		);
	}
}
