//! The host's side of the game boundary.
//!
//! Everything that knows a module can vanish lives here: the ABI check on load,
//! the `catch_unwind` around every call, and the rule that a module which
//! panicked or failed to load is simply not called again until the next
//! successful reload. The host keeps running either way, because the state the
//! game works on is the host's.

use colby_core::{
	Err, Error, Result,
	abi::{ABI_VERSION, GameApi, GameFn, World, cvar::Owner},
	error, info,
};
#[cfg(feature = "hot_reload")]
use colby_core::{
	abi::{GAME_API_SYMBOL, GameApiEntry},
	mods::Module,
};

/// Tidies up after a module's `init`.
///
/// Two things, both of which can only be done once the new build has had its
/// say. The world is advanced, because a module swap is a discontinuity
/// whatever the game does about it - `init` may have rebuilt the scene, moved
/// the camera or spawned everything from scratch, and none of that is movement.
/// And the console table is swept, which drops the variables the *previous*
/// build registered and this one did not: renamed, or gone.
///
/// @param world - the state the module was just initialized against
fn settle(world: &mut World) {
	world.advance();
	world.cvars.sweep();
}

/// The loaded game, or the absence of one.
///
/// `api` is `None` whenever there is nothing safe to call: before the first
/// load, after a failed reload, and after a panic escaped the module. That last
/// case is why a bad edit costs a log line instead of the process.
pub(crate) struct Game {
	#[cfg(feature = "hot_reload")]
	module: Option<Module>,
	/// The crate the module is built from, which is also its file name:
	/// `<id>_game` for a project called `id`. @ref `Project::module`.
	#[cfg(feature = "hot_reload")]
	name: String,
	api: Option<GameApi>,
}

impl Game {
	/// Loads the game and runs its `init`.
	///
	/// @param world - the host-owned state handed to the module
	/// @param module - the crate the module is built from, `<id>_game`. A
	/// project with no game crate names none, and then there is nothing to
	/// load; a build with the game linked in has one game whatever a project
	/// says, and ignores it
	/// @return the loaded game, nothing when there is none to load, or the
	/// reason it could not be loaded
	pub(crate) fn open(world: &mut World, module: Option<&str>) -> Result<Option<Self>> {
		let Some(mut game) = Self::closed(module) else {
			return Ok(None);
		};

		game.swap_in(world)?;

		Ok(Some(game))
	}

	/// Runs the game's `update` for one simulation step.
	///
	/// @param world - the host-owned state, with this step's inputs written
	pub(crate) fn update(&mut self, world: &mut World) {
		let Some(api) = self.api else {
			return;
		};

		self.call("update", api.update, world);
	}

	/// Runs the game's `shutdown` and unloads it.
	///
	/// @param world - the host-owned state
	pub(crate) fn close(&mut self, world: &mut World) { self.swap_out(world); }

	/// Swaps a freshly built module in, keeping every byte of `world`.
	///
	/// The old module is shut down and unloaded before the new one is opened,
	/// so at no point are two builds of the same game mapped at once.
	///
	/// @param world - the host-owned state, which survives the swap untouched
	/// @return `Ok` when the new module is loaded and initialized
	#[cfg(feature = "hot_reload")]
	pub(crate) fn reload(&mut self, world: &mut World) -> Result {
		self.swap_out(world);
		world.reloads = world.reloads.saturating_add(1);

		self.swap_in(world)
	}

	/// An unloaded game, or nothing when there is no module to load.
	#[cfg(feature = "hot_reload")]
	fn closed(module: Option<&str>) -> Option<Self> {
		Some(Self {
			module: None,
			name: module?.to_owned(),
			api: None,
		})
	}

	/// An unloaded game: the one linked in, whatever a project says.
	#[cfg(not(feature = "hot_reload"))]
	fn closed(_module: Option<&str>) -> Option<Self> { Some(Self { api: None }) }

	/// Calls one boundary entry point, containing anything it throws.
	///
	/// @param entry - the name to report if the call fails
	/// @param call - the function pointer to invoke
	/// @param world - the pointer handed across the boundary
	#[expect(
		ffi_unwind_calls,
		reason = "the boundary is deliberately C-unwind: host and module share one panic \
		          runtime under -Cprefer-dynamic, so a gameplay panic is catchable here instead \
		          of aborting the process"
	)]
	fn call(&mut self, entry: &'static str, call: GameFn, world: &mut World) {
		let world: *mut World = world;

		// SAFETY: `world` points at a live World owned by the caller and
		// borrowed exclusively for this call, which is what GameFn requires.
		// The pointer itself is valid for as long as the module is loaded, and
		// `api` is cleared the moment it is not.
		let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
			call(world);
		}));

		if let Err(payload) = result {
			let error = Error::from_panic(&*payload);
			drop(payload);

			self.api = None;
			error!(entry, %error, "game module panicked; parked until the next reload");
		}
	}
}

#[cfg(feature = "hot_reload")]
impl Game {
	/// Loads the module, checks its ABI and runs `init`.
	fn swap_in(&mut self, world: &mut World) -> Result {
		let module = Module::from_name(&self.name)?;
		let api = Self::resolve(&module)?;

		self.module = Some(module);
		self.api = Some(api);

		// everything the module registers from here on is the module's, and
		// that includes anything it registers later from `update`. The mode
		// stays set for as long as a module is loaded, because a command
		// attributed to the engine by mistake would outlive the code it points
		// at. @ref `Cvars::forget_module`.
		world.cvars.attribute(Owner::Module);

		info!(module = self.name, reloads = world.reloads, "game module swapped in");
		self.call("init", api.init, world);
		settle(world);

		Ok(())
	}

	/// Runs `shutdown` if there is anything to shut down, then unloads.
	fn swap_out(&mut self, world: &mut World) {
		if let Some(api) = self.api.take() {
			self.call("shutdown", api.shutdown, world);
		}

		// before the unload, and this is the only moment it can be: every
		// console command the module registered is an address inside the image
		// about to be freed, and one left in the table is a jump into nothing
		// the next time somebody types its name.
		world.cvars.forget_module();

		// dropping the module unloads it and runs the canary check.
		drop(self.module.take());
	}

	/// Resolves the module's entry point and validates the ABI it reports.
	fn resolve(module: &Module) -> Result<GameApi> {
		let entry = module.get::<GameApiEntry>(GAME_API_SYMBOL)?;
		let entry = *entry;

		// SAFETY: the symbol is the module's own entry point, and its prototype
		// is fixed by the ABI both sides compile against. A module built
		// against a different one is caught immediately below, before any of
		// the pointers it returned are called.
		let api = unsafe { entry() };

		if api.abi_version != ABI_VERSION {
			let found = api.abi_version;
			let name = module.name()?;

			return Err!(Module(
				"{name} reports ABI version {found}, this host speaks {ABI_VERSION}; rebuild \
				 the workspace"
			));
		}

		Ok(api)
	}
}

#[cfg(not(feature = "hot_reload"))]
impl Game {
	/// Binds the game linked into this executable: the Blank fixture's.
	fn swap_in(&mut self, world: &mut World) -> Result {
		let api = blank_game::colby_game_api();

		if api.abi_version != ABI_VERSION {
			let found = api.abi_version;

			return Err!(Module(
				"the linked game reports ABI version {found}, this host speaks {ABI_VERSION}"
			));
		}

		self.api = Some(api);

		// as the hot-reload path: whatever the game registers is the game's.
		// Nothing is ever unloaded here, so this only decides what `help` says
		// about who owns what.
		world.cvars.attribute(Owner::Module);

		info!("game linked in statically");
		self.call("init", api.init, world);
		settle(world);

		Ok(())
	}

	/// Runs `shutdown`. There is nothing to unload.
	fn swap_out(&mut self, world: &mut World) {
		if let Some(api) = self.api.take() {
			self.call("shutdown", api.shutdown, world);
		}
	}
}
