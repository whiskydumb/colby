//! Checks that the process is actually laid out the way hot-reload needs.
//!
//! Hot-reload rests on one physical fact: the host and every module link
//! `colby_core` and `std` *dynamically*, so there is one allocator, one panic
//! runtime and one copy of every static in the process. Build the same source
//! without `-Cprefer-dynamic` and everything still compiles and starts -
//! `catch_unwind` at the boundary then aborts instead of catching, and the
//! unload canary reads a counter nobody increments. Both failures look like
//! something else entirely, so the host asks the loader up front.

use libloading::library_filename;

use super::Library;
use crate::{Err, Result};

/// Answers whether this crate is loaded as a shared library.
///
/// @return `true` when a module named `colby_core.dll` is mapped into the
/// process, which is only the case when the executable imports it rather than
/// linking it in
#[must_use]
pub fn core_is_shared() -> bool {
	let name = library_filename(env!("CARGO_PKG_NAME"));

	match Library::open_already_loaded(name) {
		| Ok(library) => {
			// GetModuleHandleExW with no flags takes a reference; give it back.
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
