//! Constructor and destructor hooks for a hot-reloadable module.
//!
//! A module places `mod_ctor! {}` and `mod_dtor! {}` at its crate root. They
//! expand to functions the C runtime calls when the library is mapped and
//! unmapped, which is where the unload canary reports from.
//!
//! @note: the hooks can be written by hand as `.init_array` / `.fini_array`
//! sections, but those are ELF; the Windows equivalents are the `.CRT$XCU`
//! table and an `atexit` registration made from it, and getting both right
//! across CRT configurations is exactly the kind of thing not worth
//! hand-rolling. The `ctor` and `dtor` crates already do it.

/// Declares a module's load hook.
///
/// @param body - optional statements to run when the module is mapped
#[macro_export]
macro_rules! mod_ctor {
	( $($body:tt)* ) => {
		$crate::mod_init! {
			$crate::tracing::debug!("module loaded");
			$($body)*
		}
	};
}

/// Declares a module's unload hook, including the canary report.
///
/// @param body - optional statements to run when the module is unmapped
/// @ref [`canary::report`](crate::mods::canary::report)
#[macro_export]
macro_rules! mod_dtor {
	( $($body:tt)* ) => {
		$crate::mod_fini! {
			$crate::tracing::debug!("module unloading");
			$($body)*
			$crate::mods::canary::report();
		}
	};
}

/// Emits a function the CRT calls as the library is mapped.
#[macro_export]
macro_rules! mod_init {
	( $($body:tt)* ) => {
		$crate::hooks::ctor! {
			#[ctor(unsafe)]
			fn _mod_init() { $($body)* }
		}
	};
}

/// Emits a function the CRT calls as the library is unmapped.
#[macro_export]
macro_rules! mod_fini {
	( $($body:tt)* ) => {
		$crate::hooks::dtor! {
			#[dtor(unsafe)]
			fn _mod_fini() { $($body)* }
		}
	};
}
