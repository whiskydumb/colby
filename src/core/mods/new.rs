//! Opening a module image.

use std::path::Path;

use super::{Library, path};
use crate::{Err, Result};

/// Loads a module image by its own path.
///
/// The image is expected to be a staged copy, @ref [`path::stage`]. Its own
/// imports - `colby_core.dll`, `std-*.dll` - resolve against the modules
/// already mapped into the process, not against the directory it was copied
/// into, so a copy in `%TEMP%` links to exactly the same code the host is
/// running.
///
/// @param image - the file to map
/// @return the open library
pub fn from_path(image: &Path) -> Result<Library> {
	// SAFETY: this is LoadLibraryExW. It runs the image's initializers, which
	// is arbitrary code from a file the build just produced; that is the point
	// of the exercise. Nothing here can make that call safe.
	let library = unsafe { Library::new(image) };

	match library {
		| Ok(library) => Ok(library),
		| Err(error) => {
			let name = path::to_name(image)?;

			Err!(Module("loading {name:?} from {image:?} failed: {error}"))
		},
	}
}
