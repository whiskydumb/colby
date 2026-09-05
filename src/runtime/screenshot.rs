//! `screenshot [name]` - what the window shows, written to a png.
//!
//! The window's own command, and the one thing in the runtime that draws a
//! frame twice: a capture the window's size is made on the window's device,
//! the scene and the game's interface are drawn into it exactly as the frame
//! draws them, and the pixels are read back and written. The editor is not in
//! it, the way it is not in `--shot`: a screenshot is of the game, and a tool
//! that wants itself in a picture can ask for that when there is a key for it.
//!
//! **A second scene on the same device, deliberately.** A scene is a table of
//! pipelines for one target format and the resources it has uploaded so far,
//! and this one uploads what the world holds, draws once and is dropped with
//! the picture. That costs a moment per screenshot and nothing between them,
//! and it is the same shape a thumbnail of an asset or a preview of a material
//! will take - the device is what they share, @ref `colby_engine::gpu`.
//!
//! Every picture lands under the project's `screenshots/`, named as somebody
//! said or by the next free number - the shape the console's saves already
//! have, and the shape the field has had for a long time.

use std::{
	fs,
	path::{Path, PathBuf},
};

use colby_core::{Err, Result, abi::World, info};
use colby_engine::{Capture, Gpu, Overlay, wgpu::TextureFormat};
use colby_ui::Interface;

/// The command.
pub(crate) const COMMAND: &str = "screenshot";

/// The directory under the project every picture lands in.
pub(crate) const DIRECTORY: &str = "screenshots";

/// The extension every picture has.
const EXTENSION: &str = "png";

/// The largest number a picture is given before the directory is called full.
///
/// Four digits, which is what the field settled on: enough that nobody fills
/// it by hand, and few enough that a scan for the first free one is nothing.
const MOST: u32 = 9999;

/// Where a picture goes.
///
/// @param root - the project
/// @param name - what was typed after the command, if anything
/// @return `screenshots/<name>.png`, or the next free `screenshots/NNNN.png`
pub(crate) fn place(root: &Path, name: Option<&str>) -> Result<PathBuf> {
	let directory = root.join(DIRECTORY);

	match name {
		| Some(name) => Ok(directory
			.join(plain(name)?)
			.with_extension(EXTENSION)),
		| None => next_free(&directory, MOST),
	}
}

/// Draws the world as the window shows it, and writes the picture.
///
/// @param gpu - the window's device, which the capture is made on
/// @param format - the window's color format, which the interface's pipeline
/// was built for and the capture therefore has to have
/// @param size - the window's size in pixels, so that the interface lands
/// where it is on screen
/// @param world - the state the frame is drawing
/// @param interface - the game's interface, already laid out and prepared
/// for this frame on the same device
/// @param path - where to write, @ref [`place`]
pub(crate) fn take(
	gpu: &Gpu,
	format: TextureFormat,
	size: (u32, u32),
	world: &mut World,
	interface: &mut Interface,
	path: &Path,
) -> Result {
	let mut capture = Capture::in_format(gpu, format, size.0, size.1)?;
	let overlay: &mut dyn Overlay = interface;
	let image = capture.shoot_with(world, &mut [overlay])?;

	if let Some(directory) = path.parent() {
		fs::create_dir_all(directory)?;
	}

	image.write_png(path)?;

	info!(
		path = %path.display(),
		width = image.width,
		height = image.height,
		"screenshot written"
	);

	Ok(())
}

/// The first `NNNN.png` under a directory that does not exist yet.
///
/// The first gap rather than one past the last, which is what the field does
/// and what a person who deleted a bad one expects.
///
/// @param directory - where the pictures are
/// @param most - the largest number to try
/// @return the path, or a refusal when every number is taken
fn next_free(directory: &Path, most: u32) -> Result<PathBuf> {
	(1..=most)
		.map(|number| directory.join(format!("{number:04}.{EXTENSION}")))
		.find(|path| !path.exists())
		.map_or_else(
			|| Err!(Err("{DIRECTORY}/ holds {most} pictures already; name one, or move some")),
			Ok,
		)
}

/// A name a picture may be called, out of what somebody typed.
///
/// One flat directory, the rule the saves follow and for the same reason: a
/// console is a place people type quickly, and a separator or a parent in a
/// name is refused rather than quietly resolved. A dot goes with them, since
/// the extension is this file's to add.
///
/// @param name - what the command was given
/// @return the trimmed name, or a refusal
fn plain(name: &str) -> Result<&str> {
	let trimmed = name.trim();

	if trimmed.is_empty() {
		return Err!(Err("a screenshot's name cannot be empty; leave it out for a number"));
	}

	if trimmed.contains(['/', '\\', ':', '.']) {
		return Err!(Err("{trimmed} is not a name a screenshot can have: no directory, no dot"));
	}

	Ok(trimmed)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// An empty directory of this test's own.
	fn scratch(name: &str) -> PathBuf {
		let directory =
			std::env::temp_dir().join(format!("colby_screenshot_{name}_{}", std::process::id()));
		if directory.exists() {
			fs::remove_dir_all(&directory).expect("a stale scratch directory goes");
		}
		fs::create_dir_all(&directory).expect("a scratch directory");

		directory
	}

	#[test]
	fn a_name_lands_under_screenshots_as_a_png() {
		let root = scratch("named");

		assert_eq!(
			place(&root, Some(" hangar ")).expect("a plain name"),
			root.join(DIRECTORY).join("hangar.png"),
			"trimmed, in the directory, with the extension added"
		);
	}

	#[test]
	fn a_name_that_is_a_path_or_has_a_dot_is_refused() {
		let root = scratch("refused");

		for bad in ["../up", "a/b", "c:d", "shot.png", "", "  "] {
			assert!(place(&root, Some(bad)).is_err(), "{bad:?} is not a name");
		}
	}

	#[test]
	fn no_name_is_the_first_free_number() {
		let root = scratch("numbered");
		let directory = root.join(DIRECTORY);

		assert_eq!(
			place(&root, None).expect("a number"),
			directory.join("0001.png"),
			"a directory that does not exist yet starts at one"
		);

		fs::create_dir_all(&directory).expect("the directory");
		fs::write(directory.join("0001.png"), b"").expect("a picture");
		fs::write(directory.join("0003.png"), b"").expect("another");

		assert_eq!(
			place(&root, None).expect("a number"),
			directory.join("0002.png"),
			"the first gap, not one past the last"
		);
	}

	#[test]
	fn a_full_directory_is_refused_rather_than_overwritten() {
		let directory = scratch("full").join(DIRECTORY);
		fs::create_dir_all(&directory).expect("the directory");

		for number in 1..=3 {
			fs::write(directory.join(format!("{number:04}.png")), b"").expect("a picture");
		}

		assert!(next_free(&directory, 3).is_err(), "three of three are taken");
		assert_eq!(
			next_free(&directory, 4).expect("room for one more"),
			directory.join("0004.png"),
			"and a fourth number is the fourth picture"
		);
	}
}
