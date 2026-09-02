//! Lua for the game's interface, and for the world's own programs.
//!
//! A document names the program its logic is in - `<script src="ui/hud">`, the
//! way an image names a texture - and the program is an asset the compiler
//! turned a `.lua` into. This crate is what runs it.
//!
//! **A program under `scripts/` belongs to the world rather than to a panel**,
//! and nothing has to load it: the host walks the table for that prefix and
//! runs what it finds, the way the prop catalogue is a walk of the scene table
//! for `props/`. @ref [`WORLD_PREFIX`](colby_core::abi::WORLD_PREFIX). What
//! that program can reach is *not* what a panel's can - it has no panel, so it
//! has no `ui` at all, and a name nobody declared is `nil`, which is Lua's own
//! answer to being asked for something that is not there.
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
//! **A program is an asset, and a panel runs one by name.** A document carries
//! the name of its program - `ui/hud` - and the program itself is an entry in
//! [`World::scripts`](colby_core::abi::World::scripts) like a mesh or a
//! texture. What a panel's program is rebuilt by is therefore the *program's*
//! entry moving and not the document's: recompiling a stylesheet rewrites the
//! document and no longer restarts a program that did not change, and a program
//! two documents share is one entry rather than two copies of the text.
//!
//! The handle is watched beside the revision, and that pair is the whole key. A
//! fresh registry entry starts at revision zero exactly as the null one does,
//! so a document naming a program the compiler has not written yet resolves to
//! nothing - and the program arriving a moment later moves the *handle* while
//! moving no revision at all.
//!
//! **A program is thrown away when it is rebuilt.** The environment goes with
//! everything in it: locals, handlers, all of it. What the script wrote into
//! the panel - text, classes, style - stays, because that lives in the host's
//! binds and those already survive a document reload. Content is replaced; what
//! was laid over it is not.
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
	abi::{EventKind, PanelId, ScriptId, Scripts, World, ui::Event},
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

/// Every running program, and the one interpreter they share.
///
/// Named for the machine rather than for what is in it, because
/// [`Scripts`](colby_core::abi::Scripts) is now the *table* of programs and
/// this is the thing that runs them.
pub struct Vm {
	lua: Lua,
	tables: api::Tables,
	loaded: Vec<Loaded>,
	/// Instructions the call now running has spent. Shared with the VM's hook,
	/// which is the only thing that writes it.
	spent: Rc<Cell<u32>>,
}

/// What a running program belongs to, which is also how it is found again.
///
/// Two kinds and two keys, because the two are counted differently. Two panels
/// showing one document are **two programs** with separate locals, so the panel
/// is the key there; the world runs **one** of each program it finds, so the
/// asset is the key here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Home {
	/// A panel's, addressed by the panel. A panel shows one document for as
	/// long as it exists - @ref [`Ui::show`](colby_core::abi::Ui::show), which
	/// hands the same panel back rather than repointing one.
	Panel(PanelId),

	/// The world's own, addressed by the program itself.
	World(ScriptId),
}

impl Home {
	/// The panel this writes to, or [`PanelId::NONE`] for a program that has
	/// none.
	const fn panel(self) -> PanelId {
		match self {
			| Self::Panel(panel) => panel,
			| Self::World(_) => PanelId::NONE,
		}
	}
}

/// One running program, and what it was built from.
struct Loaded {
	/// Whose it is, and the whole key.
	home: Home,
	/// Which entry of the script table it was built from.
	///
	/// Kept beside the revision because a fresh entry starts at revision zero,
	/// exactly like the null one: a document naming a program that had not been
	/// compiled yet resolves to nothing, and the program arriving afterwards
	/// moves the *handle* without moving any revision at all. A document
	/// rewritten to name something else moves it too, which is why nothing
	/// here watches the document's own revision.
	program: ScriptId,
	/// The revision of that entry. Somebody editing the file moves this, and
	/// so does `script.reload`; nothing else does.
	program_revision: u32,
	name: String,
	/// Its handlers, or `None` for a program that registered none - which is
	/// every world program, since `ui.on` is not something one can reach.
	handlers: Option<Table>,
}

/// A program that has to be built, or built again.
struct Pending {
	home: Home,
	program: ScriptId,
	program_revision: u32,
	name: String,
	script: String,
}

/// One event waiting for a handler.
struct Waiting {
	panel: PanelId,
	node: String,
	kind: &'static str,
}

impl Vm {
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

	/// Every program that is missing or out of date, panels first.
	///
	/// Panels first because that is the order the step runs in anyway, and
	/// within each half it is slot order - the order the compiler walked the
	/// tree in, which is sorted. Nothing here depends on a hash.
	fn pending(&self, world: &World) -> Vec<Pending> {
		let mut jobs = Vec::new();

		for (panel, shown) in world.ui.panels() {
			let document = shown.document();
			let Some(entry) = world.ui.document(document) else {
				continue;
			};

			// the *document's* name, because every line about a panel's
			// program is read by somebody who has the document open. Which
			// file the program came from is one lookup away.
			self.wanted(
				world,
				Home::Panel(panel),
				&entry.value().program,
				entry.name(),
				&mut jobs,
			);
		}

		for entry in world.scripts.iter() {
			if !Scripts::is_world(entry.name()) {
				continue;
			}

			let name = entry.name().to_owned();
			let id = world.scripts.find(&name);

			self.wanted(world, Home::World(id), &name, &name, &mut jobs);
		}

		jobs
	}

	/// Adds one program to the list if what it was built from has moved.
	///
	/// @param home - whose program this is
	/// @param named - the name of the program to run, which for a panel is what
	/// its document says and for the world is the program's own
	/// @param about - the name to put in any line written about it
	fn wanted(
		&self,
		world: &World,
		home: Home,
		named: &str,
		about: &str,
		jobs: &mut Vec<Pending>,
	) {
		let program = world.scripts.find(named);
		let found = world.scripts.get(program);
		let program_revision = found.map_or(0, colby_core::abi::Entry::revision);

		let current = self.loaded.iter().any(|loaded| {
			loaded.home == home
				&& loaded.program == program
				&& loaded.program_revision == program_revision
		});

		if current {
			return;
		}

		jobs.push(Pending {
			home,
			program,
			program_revision,
			name: about.to_owned(),
			script: found
				.map(|entry| entry.value().source.clone())
				.unwrap_or_default(),
		});
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
			.position(|loaded| loaded.home == job.home);

		let handlers = match built {
			| Ok(handlers) => {
				// `program` rather than `document`, and it changed when the
				// world got programs of its own: half of what this line is
				// written about has no document at all. The panel reads as
				// nought for those, which is `PanelId::NONE`.
				info!(
					program = job.name,
					panel = job.home.panel().index(),
					handlers = handlers.is_some(),
					"script loaded"
				);

				handlers
			},
			| Err(error) => {
				warn!(program = job.name, %error, "the program was not loaded");

				// the revision is recorded anyway, so a file that does not
				// compile is reported once rather than sixty times a second.
				// The next edit moves it again and it is tried again.
				slot.and_then(|slot| loaded.get(slot))
					.and_then(|loaded| loaded.handlers.clone())
			},
		};

		let entry = Loaded {
			home: job.home,
			program: job.program,
			program_revision: job.program_revision,
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
	/// @return the table its handlers were filed in, or `None` for a program
	/// that has nowhere to file one - which is a document carrying no program
	/// at all, and every world program, since `ui.on` is not a name one can
	/// reach
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

		// a **projection** of each table rather than the table, because
		// `Table` is a reference and handing the same one to every program
		// makes `colby.command = nil` in one of them everybody's problem for
		// the rest of the step. Reading falls through to the real table, so a
		// program still sees the functions this step made; writing lands in
		// the projection and is that program's own, and writing `nil` puts the
		// key back to absent, which reads through again.
		//
		// This is the trick the one interpreter here built for untrusted code
		// uses for each script's globals. What it does *not* cover is the
		// standard libraries: `math`, `string` and `table` are one object
		// each, shared, and Lua 5.4 has no way to make a table read-only. @ref
		// `colby-known-gaps`.
		environment.set("colby", Self::projection(lua, &tables.engine)?)?;
		if matches!(job.home, Home::Panel(_)) {
			environment.set("ui", Self::projection(lua, &tables.ui)?)?;
		}

		// a world program is handed nowhere to file a handler, because there is
		// no `ui.on` in its environment to file one with. `None` rather than an
		// empty table so that the line written about it does not claim it has
		// handlers, and so that dispatch has one thing to check rather than
		// two.
		let handlers = match job.home {
			| Home::Panel(_) => Some(lua.create_table()?),
			| Home::World(_) => None,
		};

		*running.borrow_mut() = api::Running {
			panel: job.home.panel(),
			handlers: handlers.clone(),
			name: job.name.clone(),
		};
		spent.set(0);

		// the name is prefixed so that Lua reports `ui/hud:3:` rather than
		// quoting the whole chunk back at whoever is reading the log.
		lua.load(job.script.as_str())
			.set_name(format!("@{}", job.name))
			.set_environment(environment)
			.exec()?;

		Ok(handlers)
	}

	/// An empty table that reads through to another one.
	///
	/// @param behind - what a name not found in the projection resolves to
	fn projection(lua: &Lua, behind: &Table) -> mlua::Result<Table> {
		let front = lua.create_table()?;
		let meta = lua.create_table()?;

		meta.set("__index", behind.clone())?;
		front.set_metatable(Some(meta))?;

		Ok(front)
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
			.find(|loaded| loaded.home == Home::Panel(event.panel))
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
