//! Opening a module image.

use std::path::Path;

use super::{Library, path};
use crate::{Err, Result};

/// Loads a module image by its own path.
///
/// The image is expected to be a staged copy, @ref [`path::stage`]. Its own
/// imports - `colby_core` and `std`, both shared libraries under `just hot` -
/// resolve against the images already mapped into the process, not against
/// the directory it was copied into, so a copy in the scratch directory links
/// to exactly the same code the host is running.
///
/// @param image - the file to map
/// @return the open library
pub fn from_path(image: &Path) -> Result<Library> {
	match map(image) {
		| Ok(library) => Ok(library),
		| Err(error) => {
			let name = path::to_name(image)?;

			Err!(Module("loading {name:?} from {image:?} failed: {error}"))
		},
	}
}

/// Maps an image into the process.
///
/// `LoadLibraryExW`, which resolves every import before it returns and runs
/// the image's initializers.
#[cfg(windows)]
fn map(image: &Path) -> Result<Library, libloading::Error> {
	// SAFETY: this runs the image's initializers, which is arbitrary code from
	// a file the build just produced; that is the point of the exercise.
	// Nothing here can make that call safe.
	unsafe { Library::new(image) }
}

/// Maps an image into the process.
///
/// `dlopen` with `RTLD_NOW`, so that a symbol the image cannot resolve fails
/// the load with its name rather than the first call to it - which is what
/// the Windows loader does without being asked - and `RTLD_LOCAL`, so that
/// nothing resolves against the module: it exports one entry point and the
/// rest crosses as a table. @ref `crate::abi`.
#[cfg(unix)]
fn map(image: &Path) -> Result<Library, libloading::Error> {
	use libloading::os::unix::{RTLD_LOCAL, RTLD_NOW};

	// SAFETY: as on Windows: the image's initializers run, and they are the
	// build's own code.
	unsafe { Library::open(Some(image), RTLD_NOW | RTLD_LOCAL) }
}
