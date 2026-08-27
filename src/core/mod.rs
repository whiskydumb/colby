//! Foundations shared by every colby crate.
//!
//! The crate owns the error type and its construction macros, the tracing
//! logger, the hot-reload module loader and the `#[repr(C)]` types that cross
//! the host/game boundary. Everything above it depends on this crate, and under
//! `just hot` everything above it links to this crate *dynamically*, so the
//! statics declared here exist exactly once in the process.

// @note: crate-wide opt-in to the workspace `unsafe-code = "deny"`. The unsafe
// in this crate is confined to `mods` (LoadLibrary/GetProcAddress and the
// module ctor/dtor hooks) and to the bytemuck derives in `abi`.
#![allow(unsafe_code)]

pub mod abi;
pub mod error;
pub mod log;
pub mod time;
pub mod utils;

#[cfg(feature = "hot_reload")]
pub mod mods;

#[cfg(not(feature = "hot_reload"))]
/// Stands in for the module loader when hot-reload is compiled out.
///
/// A game module keeps its `mod_ctor!` / `mod_dtor!` invocations either way;
/// without the feature they expand to nothing.
pub mod mods {
	/// Expands to nothing. @ref the real
	/// [`mod_ctor!`](https://docs.rs/colby_core) built with `hot_reload`.
	#[macro_export]
	macro_rules! mod_ctor {
		($($body:tt)*) => {};
	}

	/// Expands to nothing.
	#[macro_export]
	macro_rules! mod_dtor {
		($($body:tt)*) => {};
	}
}

/// Re-exported so that no other crate needs a dependency on it.
///
/// @note: a crate deriving `Pod` or `Zeroable` through this path has to say
/// `#[bytemuck(crate = "::colby_core::bytemuck")]`, because the derive emits
/// `::bytemuck` otherwise.
pub use ::bytemuck;
/// The math library, re-exported so that no other crate needs a dependency on
/// it and everything agrees on one `Vec3`.
///
/// @note: a decision, not an accident. `Mat4::perspective_rh` and friends are
/// the kind of code that is wrong in ways only a picture reveals, and glam's
/// have been looked at by rather more people than this engine ever will.
pub use ::glam;
pub use ::tracing::{self, debug, error, info, trace, warn};

pub use crate::error::{Error, Result};

/// The macros a module's load and unload hooks expand to.
///
/// @note: the *declarative* forms, not the `#[ctor]` / `#[dtor]` attributes.
/// Those attributes emit code referring to `::ctor` and `::dtor` by absolute
/// path, so re-exporting them would force every module crate to depend on both
/// as well. The declarative forms are `macro_rules!` and carry their own
/// `$crate`, so a module crate needs nothing but colby_core.
#[cfg(feature = "hot_reload")]
pub mod hooks {
	pub use ::ctor::declarative::ctor;
	pub use ::dtor::declarative::dtor;
}

/// Refers to this crate by name from inside its own macros.
///
/// Macros exported here expand to `$crate::...` paths. Downstream crates see
/// the real crate name; this alias makes the same expansion resolve when the
/// macro is used within `colby_core` itself.
pub use crate as colby_core;
