//! Detects modules that fail to leave the process.
//!
//! Four atomic operations that
//! turn "something still holds a pointer into the unloaded library" from a
//! crash at an arbitrary later moment into a log line at the unload.
//!
//! The host decrements the counter just before unloading. The module's static
//! destructor increments it as the library tears down. If the count is not back
//! to zero afterwards, the destructor never ran, which on Windows means
//! `FreeLibrary` did not actually unmap the image - some reference is still
//! outstanding.
//!
//! @note: this only works because the counter is a static in colby_core and
//! both sides link colby_core *dynamically*. Build without `-Cprefer-dynamic`
//! and each side gets its own counter, at which point the check silently
//! reports every module as stuck.

use std::sync::atomic::{AtomicI32, Ordering};

const ORDERING: Ordering = Ordering::Relaxed;

static STATIC_DTORS: AtomicI32 = AtomicI32::new(0);

/// Arms the check ahead of an unload.
///
/// Called by [`Module::unload`](super::Module::unload) to say that static
/// destruction is expected. @ref [`check_and_reset`].
pub(crate) fn prepare() {
	let count = STATIC_DTORS.fetch_sub(1, ORDERING);
	debug_assert!(count <= 0, "STATIC_DTORS should not be greater than zero.");
}

/// Reports that a module's static destructor ran.
///
/// This belongs inside [`mod_dtor!`](crate::mod_dtor) and nowhere else.
#[inline(always)]
pub fn report() { let _count = STATIC_DTORS.fetch_add(1, ORDERING); }

/// Answers whether the armed unload completed, and re-arms for the next one.
///
/// Resetting rather than merely reading means one stuck module does not poison
/// the check for every module unloaded after it.
///
/// @return `true` when static destruction took place
pub(crate) fn check_and_reset() -> bool { STATIC_DTORS.swap(0, ORDERING) == 0 }
