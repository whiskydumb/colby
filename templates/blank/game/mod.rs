//! The smallest game there is.
//!
//! Three entry points that do nothing but say so. It is what a new project
//! starts from, and it is what the loader, the `static_game` check and the
//! hot-reload check have to load when the engine's own tree is what is being
//! built, which carries no other game.
//!
//! Everything a project shows comes from its scenes and its programs, and a
//! new one has neither. A game that wants more than that adds it here.

// @note: the workspace denies unsafe code; the three entry points are the whole
// of what a module cannot avoid, and each is a pointer the host promised.
#![allow(unsafe_code)]

use colby_core::{
	abi::{ABI_VERSION, GameApi, World},
	info, mod_ctor, mod_dtor,
};

// @note: the two hooks every module declares, and neither is decoration. The
// second is where the unload canary is reported from: without it the host
// reads every swap of this module as a module that failed to leave the
// process, and says so on every reload. @ref `colby_core::mods::canary`.
mod_ctor! {}
mod_dtor! {}

/// The one symbol the host resolves.
///
/// The host calls this once per load and then only ever calls through the table
/// it returned. @ref [`GAME_API_SYMBOL`](colby_core::abi::GAME_API_SYMBOL) for
/// the name it looks up.
#[unsafe(no_mangle)]
pub extern "C" fn colby_game_api() -> GameApi {
	GameApi {
		abi_version: ABI_VERSION,
		init,
		update,
		shutdown,
	}
}

/// Runs once each time this module is swapped in.
///
/// # Safety
///
/// `world` must point to a live [`World`] owned by the host.
unsafe extern "C-unwind" fn init(world: *mut World) {
	// SAFETY: the host hands over its own world and touches nothing else while
	// this runs.
	let world = unsafe { &*world };

	info!(reloads = world.reloads, entities = world.entities.len(), "$id game init");
}

/// Runs once a simulation step, and has nothing to do.
///
/// # Safety
///
/// As [`init`].
unsafe extern "C-unwind" fn update(_world: *mut World) {}

/// Runs once before this module is swapped out.
///
/// # Safety
///
/// As [`init`].
unsafe extern "C-unwind" fn shutdown(world: *mut World) {
	// SAFETY: as init.
	let world = unsafe { &*world };

	info!(steps = world.steps, "$id game shutdown");
}
