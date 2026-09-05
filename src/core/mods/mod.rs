//! Loading, unloading and re-loading native modules at runtime.
//!
//! The shape of the thing is the one every hot-reloading host arrives at - an
//! ordered stack of modules, an mtime check to decide what is stale, a canary
//! that proves a module really left the process. It was built against
//! Windows, whose loader is the stricter of the two, and the same shape ports
//! to unix with five calls renamed:
//!
//! | what               | windows              | unix                         |
//! |--------------------|----------------------|------------------------------|
//! | map an image       | `LoadLibraryExW`     | `dlopen`, now and local      |
//! | the entry point    | `GetProcAddress`     | `dlsym`                      |
//! | unmap              | `FreeLibrary`        | `dlclose`                    |
//! | load/unload hooks  | `.CRT$XCU`, `atexit` | `.init_array`, `.fini_array` |
//! | is core shared     | `GetModuleHandleExW` | `dlopen` with `RTLD_NOLOAD`  |
//!
//! Two things that are not calls carry over as well. **The image is copied
//! before it is mapped** - @ref [`path::stage`] - because Windows keeps a
//! mapped file locked, so the linker could not overwrite `blank_game.dll`
//! while the running process had it open; and because a unix loader hands
//! back the mapping it already has for a name it already knows, so a module
//! that failed to leave the process would be "reloaded" as its old self. A
//! fresh path per generation makes a reload honest on both. **Symbols are not
//! shared**: a module is mapped into a private namespace, which is all
//! Windows has, and exports one C entry point; everything else crosses as a
//! table of function pointers. @ref `crate::abi`.

#[cfg(unix)]
pub(crate) use libloading::os::unix::{Library, Symbol};
#[cfg(windows)]
pub(crate) use libloading::os::windows::{Library, Symbol};

#[cfg(not(any(unix, windows)))]
compile_error!("the module loader knows windows and unix, and this target is neither");

pub mod canary;
pub mod linkage;
pub mod macros;
pub mod module;
pub mod new;
pub mod path;

pub use self::module::Module;
