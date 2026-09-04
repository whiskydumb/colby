//! Loading, unloading and re-loading native modules at runtime.
//!
//! The shape of the thing is the one every hot-reloading host arrives at - an
//! ordered stack of modules, an mtime check to decide what is stale, a canary
//! that proves a module really left the process. The mechanics are Windows's
//! own, because none of the unix answers port:
//!
//! | unix                                | colby (windows)                     |
//! |------------------------------------|-------------------------------------|
//! | `dlopen` with `RTLD_GLOBAL`         | `LoadLibraryW`; no global namespace |
//! | symbols shared between host and mod | one exported C entry point          |
//! | load the built file in place        | copy to `%TEMP%` and load the copy  |
//! | `.init_array` / `.fini_array`       | `ctor` / `dtor` (`.CRT$XCU`, atexit)|
//!
//! The copy is not optional: Windows keeps a loaded image mapped and denies
//! writes to the file, so the linker could not overwrite `blank_game.dll` while
//! the running process had it open. Everything else follows from there.

pub(crate) use libloading::os::windows::{Library, Symbol};

pub mod canary;
pub mod linkage;
pub mod macros;
pub mod module;
pub mod new;
pub mod path;

pub use self::module::Module;
