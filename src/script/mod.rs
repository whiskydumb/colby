//! Lua for the game's interface, and for nothing else yet.
//!
//! A document can carry a program the way it carries a stylesheet:
//! `<script src="hud.lua">` beside `<link href="theme.css">`, resolved and
//! folded into the `.cdoc` by the compiler. This crate is what runs it.
//!
//! Four decisions are written into the shape of the module, and each was taken
//! against a plausible alternative.
//!
//! **The VM lives in the host, exactly like the physics.** `colby.exe` owns it,
//! `World` does not know it exists, and no pointer from it reaches into the
//! game module - so a hot-reload of the gameplay dylib does not disturb a
//! running script, and none of the lifetime care a console command needs
//! applies here. That is the whole reason a script VM is worth having on the
//! interface first: it is the one surface where it costs nothing.
//!
//! **A program belongs to a panel, and is thrown away when its document
//! reloads.** Nothing new was needed for that - a document is a registry entry,
//! and the entry's `revision` moves when the compiler rewrites it, which is the
//! same signal the renderer uses to decide whether to upload a mesh again. When
//! it moves, the environment goes with everything in it: locals, handlers, all
//! of it. What the script wrote into the panel - text, classes, style - stays,
//! because that lives in the host's binds and those already survive a document
//! reload. Content is replaced; what was laid over it is not.
//!
//! **A script runs inside the simulation step**, in `step::run`, right after
//! the interface has been laid out and hit-tested and before physics and the
//! game's `update`. Two things follow. `--shot` runs scripts, so a screenshot
//! shows what the window shows; and a screenshot stays *reproducible*, which is
//! why there is no clock in the environment, why `math.random` is seeded from a
//! constant here, and why a runaway program is stopped by counting
//! instructions rather than by watching a stopwatch. A budget in milliseconds
//! would make the same document behave differently on a slower machine.
//!
//! **What a script can reach is a table built by hand.** @ref [`api`], which is
//! where the argument for each name in it lives.

use std::{
	cell::{Cell, RefCell},
	rc::Rc,
};

use colby_core::{
	Result,
	abi::{EventKind, PanelId, World, ui::Event},
	err, info, trace, warn,
};
use mlua::{Function, HookTriggers, Lua, LuaOptions, StdLib, Table, VmState};

mod api;

#[cfg(test)]
mod tests;

/// How many Lua instructions one call may spend before it is stopped.
///
/// A handler does a handful of table writes; this is four orders of magnitude
/// more than that, and it is the difference between `while true do end` in a
/// file somebody is editing and a window that never draws again.
const BUDGET: u32 = 200_000;

/// How often the VM stops to check that budget.
///
/// Fine enough that a runaway loop is caught in well under a frame and coarse
/// enough that the check is not the cost of running a script at all.
const GRAIN: u32 = 1_000;

/// The most memory every script in the process may hold between them, in bytes.
const MEMORY: usize = 4 << 20;

/// What `math.random` is seeded from.
///
/// A constant, and not a decorative one: Lua 5.4 seeds a fresh state from the
/// clock and an address, which would make two runs of `--shot` draw different
/// pictures the moment a document rolled a die. What it does *not* cover is a
/// program that walks a table with `pairs`, whose order Lua hashes with a seed
/// of its own.
const SEED: i64 = 0x_C01B_1105;

/// The standard libraries a document's script is given.
///
/// `io`, `os`, `package` and `debug` are not among them, which is what makes an
/// interface script unable to read a file, ask the time or reach the C API.
/// `coroutine` is left out for a duller reason: a handler that yields would
/// leave the step waiting on something that has no way to be resumed.
fn libraries() -> StdLib { StdLib::TABLE | StdLib::STRING | StdLib::MATH }

/// The base functions an environment inherits.
///
/// Everything else base declares - `load`, `dofile`, `require`,
/// `collectgarbage`, `rawset` and the rest - is left out, and a script asking
/// for one gets `nil` rather than a refusal, which is what Lua does with any
/// name nobody declared.
const BASE: [&str; 10] = [
	"assert", "error", "ipairs", "math", "next", "pairs", "pcall", "select", "string", "table",
];

/// Two more, kept apart only because they are conversions rather than control.
const CONVERSIONS: [&str; 3] = ["tonumber", "tostring", "type"];

/// Every document's program, and the one interpreter they share.
pub struct Scripts {
	lua: Lua,
	tables: api::Tables,
	loaded: Vec<Loaded>,
	/// Instructions the call now running has spent. Shared with the VM's hook,
	/// which is the only thing that writes it.
	spent: Rc<Cell<u32>>,
}

/// One panel's program, and what it was built from.
struct Loaded {
	/// Which panel it belongs to. A panel shows one document for as long as it
	/// exists - @ref [`Ui::show`](colby_core::abi::Ui::show), which hands the
	/// same panel back rather than repointing one - so this is the whole key.
	panel: PanelId,
	/// The revision of the document entry this was built from. When the
	/// compiler rewrites the file this moves, and that is the whole reload
	/// mechanism.
	revision: u32,
	name: String,
	/// Its handlers, or `None` for a document with no script at all.
	handlers: Option<Table>,
}

/// A document whose program has to be built, or built again.
struct Pending {
	panel: PanelId,
	revision: u32,
	name: String,
	script: String,
}

/// One event waiting for a handler.
struct Waiting {
	panel: PanelId,
	node: String,
	kind: &'static str,
}

impl Scripts {
	/// Brings up the interpreter with nothing loaded into it.
	///
	/// @return the scripts, or why Lua could not be started
	pub fn new() -> Result<Self> {
		let lua = Lua::new_with(libraries(), LuaOptions::default())
			.map_err(|error| err!(Script("the interpreter could not be started: {error}")))?;

		lua.set_memory_limit(MEMORY)
			.map_err(|error| err!(Script("the memory limit was refused: {error}")))?;

		let spent = Rc::new(Cell::new(0_u32));
		let counted = Rc::clone(&spent);
		lua.set_hook(HookTriggers::new().every_nth_instruction(GRAIN), move |_, _| {
			let used = counted.get().saturating_add(GRAIN);
			counted.set(used);

			if used > BUDGET {
				return Err(mlua::Error::runtime(format!(
					"spent more than {BUDGET} instructions in one call and was stopped"
				)));
			}

			Ok(VmState::Continue)
		})
		.map_err(|error| err!(Script("the instruction budget was refused: {error}")))?;

		let tables = Self::tables(&lua).map_err(|error| {
			err!(Script("the script environment could not be built: {error}"))
		})?;

		Ok(Self { lua, tables, loaded: Vec::new(), spent })
	}

	/// Runs whatever the interface has for the scripts this step.
	///
	/// Loads or reloads any document whose entry has moved, then hands this
	/// step's events to the handlers that asked for them. Called by the host
	/// from inside the step. @ref the module docs for why it is there.
	///
	/// @param world - the interface to serve, and to write through
	pub fn update(&mut self, world: &mut World) {
		let pending = self.pending(world);

		if pending.is_empty() && world.ui.events().is_empty() {
			return;
		}

		self.serve(world, &pending);
	}

	/// The panels whose program is missing or out of date.
	fn pending(&self, world: &World) -> Vec<Pending> {
		let mut jobs = Vec::new();

		for (panel, shown) in world.ui.panels() {
			let document = shown.document();
			let Some(entry) = world.ui.document(document) else {
				continue;
			};

			let revision = entry.revision();
			let current = self
				.loaded
				.iter()
				.any(|loaded| loaded.panel == panel && loaded.revision == revision);

			if current {
				continue;
			}

			jobs.push(Pending {
				panel,
				revision,
				name: entry.name().to_owned(),
				script: entry.value().script.clone(),
			});
		}

		jobs
	}

	/// Opens one scope, loads what has changed and dispatches what happened.
	///
	/// One scope for the whole step rather than one per call: the functions a
	/// script reaches the engine through are made here and destroyed when this
	/// returns, and making them six times over for six events would be six
	/// times the work for the same answer.
	fn serve(&mut self, world: &mut World, pending: &[Pending]) {
		let Self { lua, tables, loaded, spent } = self;
		let events = Self::waiting(world);
		let world = RefCell::new(world);
		let running = RefCell::new(api::Running::default());

		let outcome = lua.scope(|scope| {
			api::fill(scope, tables, &world, &running)?;

			for job in pending {
				Self::reload(lua, tables, loaded, spent, &running, job);
			}

			for event in &events {
				Self::dispatch(loaded, spent, &running, event);
			}

			Ok(())
		});

		if let Err(error) = outcome {
			warn!(%error, "the interface scripts could not be given their api this step");
		}
	}

	/// This step's events, in the order they happened.
	fn waiting(world: &World) -> Vec<Waiting> {
		world
			.ui
			.events()
			.iter()
			.map(|event: &Event| Waiting {
				panel: event.panel,
				node: event.node.clone(),
				kind: kind_name(event.kind),
			})
			.collect()
	}

	/// Builds one document's program, replacing whatever it had.
	///
	/// A chunk that will not load leaves the previous program in place and says
	/// so, which is what the shader watcher does with a file naga refuses: an
	/// interface that goes on working while somebody fixes a typo is worth more
	/// than one that empties itself the moment a file is saved halfway through.
	fn reload(
		lua: &Lua,
		tables: &api::Tables,
		loaded: &mut Vec<Loaded>,
		spent: &Cell<u32>,
		running: &RefCell<api::Running>,
		job: &Pending,
	) {
		let built = Self::build(lua, tables, spent, running, job);
		let slot = loaded
			.iter()
			.position(|loaded| loaded.panel == job.panel);

		let handlers = match built {
			| Ok(handlers) => {
				info!(
					document = job.name,
					panel = job.panel.index(),
					handlers = handlers.is_some(),
					"script loaded"
				);

				handlers
			},
			| Err(error) => {
				warn!(document = job.name, %error, "the document's script was not loaded");

				// the revision is recorded anyway, so a file that does not
				// compile is reported once rather than sixty times a second.
				// The next edit moves it again and it is tried again.
				slot.and_then(|slot| loaded.get(slot))
					.and_then(|loaded| loaded.handlers.clone())
			},
		};

		let entry = Loaded {
			panel: job.panel,
			revision: job.revision,
			name: job.name.clone(),
			handlers,
		};

		match slot {
			| Some(slot) =>
				if let Some(existing) = loaded.get_mut(slot) {
					*existing = entry;
				},
			| None => loaded.push(entry),
		}
	}

	/// Runs one chunk in an environment of its own.
	///
	/// @return the table its handlers were filed in, or `None` for a document
	/// that carries no script
	fn build(
		lua: &Lua,
		tables: &api::Tables,
		spent: &Cell<u32>,
		running: &RefCell<api::Running>,
		job: &Pending,
	) -> mlua::Result<Option<Table>> {
		if job.script.trim().is_empty() {
			return Ok(None);
		}

		let environment = lua.create_table()?;
		let meta = lua.create_table()?;
		meta.set("__index", tables.globals.clone())?;
		environment.set_metatable(Some(meta))?;
		environment.set("ui", tables.ui.clone())?;
		environment.set("colby", tables.engine.clone())?;

		let handlers = lua.create_table()?;
		*running.borrow_mut() = api::Running {
			panel: job.panel,
			handlers: Some(handlers.clone()),
			name: job.name.clone(),
		};
		spent.set(0);

		// the name is prefixed so that Lua reports `ui/hud:3:` rather than
		// quoting the whole chunk back at whoever is reading the log.
		lua.load(job.script.as_str())
			.set_name(format!("@{}", job.name))
			.set_environment(environment)
			.exec()?;

		Ok(Some(handlers))
	}

	/// Hands one event to the handler that asked for it, if there is one.
	fn dispatch(
		loaded: &[Loaded],
		spent: &Cell<u32>,
		running: &RefCell<api::Running>,
		event: &Waiting,
	) {
		let Some(entry) = loaded
			.iter()
			.find(|loaded| loaded.panel == event.panel)
		else {
			return;
		};

		let Some(handlers) = entry.handlers.as_ref() else {
			return;
		};

		let found = handlers
			.get::<Option<Table>>(event.node.as_str())
			.and_then(|node| {
				node.map_or(Ok(None), |node| node.get::<Option<Function>>(event.kind))
			});

		let handler = match found {
			| Ok(Some(handler)) => handler,
			| Ok(None) => return,
			| Err(error) => {
				warn!(document = entry.name, %error, "the handler table could not be read");

				return;
			},
		};

		*running.borrow_mut() = api::Running {
			panel: event.panel,
			handlers: Some(handlers.clone()),
			name: entry.name.clone(),
		};
		spent.set(0);

		// the line a person debugging a document actually wants, and the only
		// place the chain from a click to a named handler is visible at once.
		// The same shape as the interface's own line about the box it hit.
		trace!(document = entry.name, node = event.node, kind = event.kind, "script handled");

		if let Err(error) = handler.call::<()>((event.node.as_str(), event.kind)) {
			warn!(
				document = entry.name,
				node = event.node,
				kind = event.kind,
				%error,
				"a handler failed"
			);
		}
	}

	/// Builds the three tables every script sees.
	fn tables(lua: &Lua) -> mlua::Result<api::Tables> {
		let standard = lua.globals();
		let globals = lua.create_table()?;

		for name in BASE.iter().chain(CONVERSIONS.iter()) {
			let value: mlua::Value = standard.get(*name)?;
			globals.set(*name, value)?;
		}

		// deterministic from the first call, which is what keeps two runs of
		// `--shot` the same picture.
		standard
			.get::<Table>("math")?
			.get::<Function>("randomseed")?
			.call::<()>(SEED)?;

		Ok(api::Tables {
			ui: lua.create_table()?,
			engine: lua.create_table()?,
			globals,
		})
	}
}

/// What a script calls one kind of event.
///
/// The names are the ones a person would write in a document, and they are the
/// only place the wire between [`EventKind`] and Lua is spelled out.
const fn kind_name(kind: EventKind) -> &'static str {
	match kind {
		| EventKind::Press => api::KINDS[0],
		| EventKind::Release => api::KINDS[1],
		| EventKind::Click => api::KINDS[2],
	}
}
