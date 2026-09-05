//! Checks that the process is actually laid out the way hot-reload needs.
//!
//! Hot-reload rests on one physical fact: the host and every module link
//! `colby_core` and `std` *dynamically*, so there is one allocator, one panic
//! runtime and one copy of every static in the process. Build the same source
//! without `-Cprefer-dynamic` and everything still compiles and starts -
//! `catch_unwind` at the boundary then aborts instead of catching, and the
//! unload canary reads a counter nobody increments. Both failures look like
//! something else entirely, so the host asks the loader up front.

use std::ffi::OsStr;

use libloading::library_filename;

use super::Library;
use crate::{Err, Result};

/// Answers whether this crate is loaded as a shared library.
///
/// @return `true` when an image named `colby_core.dll` - `libcolby_core.so`
/// on unix - is mapped into the process, which is only the case when the
/// executable imports it rather than linking it in
#[must_use]
pub fn core_is_shared() -> bool {
	let name = library_filename(env!("CARGO_PKG_NAME"));

	match already_loaded(&name) {
		| Ok(library) => {
			// the handle took a reference on the image; give it back.
			drop(library.close());

			true
		},
		| Err(_) => false,
	}
}

/// Refuses to continue if the process is not laid out for hot-reload.
///
/// @return `Ok` when [`core_is_shared`] holds
pub fn require_shared_core() -> Result {
	if core_is_shared() {
		return Ok(());
	}

	Err!(Module(
		"colby_core is linked statically, so the host and a game module would not share std, \
		 the allocator, the panic runtime or the unload canary. Build with `just hot`, which \
		 passes -Cprefer-dynamic, or turn the `hot_reload` feature off"
	))
}

/// A handle to an image the process already has, without mapping anything.
///
/// `GetModuleHandleExW` with no flags, which takes a reference the caller
/// gives back.
#[cfg(windows)]
fn already_loaded(name: &OsStr) -> Result<Library, libloading::Error> {
	Library::open_already_loaded(name)
}

/// A handle to an image the process already has, without mapping anything.
///
/// `dlopen` with `RTLD_NOLOAD` maps nothing and runs no initializer: it
/// answers only for a name the loader already holds, and takes a reference
/// the caller gives back. The name matches what the executable imports
/// because cargo writes a path package's shared library without the metadata
/// hash in its name, on every platform, so the import is `libcolby_core.so`
/// exactly.
#[cfg(target_os = "linux")]
fn already_loaded(name: &OsStr) -> Result<Library, libloading::Error> {
	use std::ffi::c_int;

	use libloading::os::unix::RTLD_NOW;

	/// glibc's and musl's value. Not in `libloading`, which exposes the four
	/// POSIX flags and no more.
	const RTLD_NOLOAD: c_int = 0x4;

	// SAFETY: with RTLD_NOLOAD the call maps no image and runs no code from
	// one; it is a lookup among the images already in the process.
	unsafe { Library::open(Some(name), RTLD_NOW | RTLD_NOLOAD) }
}

#[cfg(all(unix, not(target_os = "linux")))]
compile_error!(
	"linkage::already_loaded has no answer for this unix: the value of RTLD_NOLOAD differs \
	 between loaders, and the check that colby_core is shared cannot be skipped"
);
