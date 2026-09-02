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
//! **The two halves run at two different moments, and that is the whole reason
//! there are two entry points.** [`Vm::interface`] runs where the interface
//! does, before the physics and *outside* the edit-mode guard, because laying
//! out and answering a click is not moving the world. [`Vm::gameplay`] runs
//! where the game's `update` does, after it and *inside* the guard, because a
//! world program is gameplay written in Lua and edit mode stops gameplay. A
//! world program is therefore neither loaded nor ticked while somebody is
//! editing - which is the same promise the game module already has.
//!
//! **A world program is ticked, and a panel's is not.** `function tick(dt)` in
//! a program under `scripts/` is called once a step; the same function in a
//! document's program is a complaint at load, because a document that has to
//! run every step is gameplay that has been put in the wrong place. What the
//! tick is read from is the program's own environment, once a step, so a
//! program that writes `tick = nil` stops being ticked - including from inside
//! its own tick.
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
use mlua::{Function, HookTriggers, Lua, LuaOptions, StdLib, Table, Value, VmState};

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
	/// The table its chunk ran in, kept so that `tick` can be read out of it
	/// once a step.
	///
	/// Read every step rather than resolved once, which costs one lookup and
	/// buys a program the ability to stop itself: writing `tick = nil` from
	/// inside its own tick is a program that ran once and is done.
	environment: Option<Table>,
	/// Whether the chunk left a `tick` behind. Only the early return reads
	/// this - the call itself asks the environment - so it is a hint rather
	/// than an authority.
	ticks: bool,
	/// Whether it has been stopped until it is built again.
	///
	/// Set by a tick that spent its whole budget, and by nothing else. An
	/// ordinary error is cheap and is counted; a budget overrun costs a full
	/// [`BUDGET`] every step for as long as nobody looks, which is the one
	/// failure worth switching off. Editing the file or `script.reload` builds
	/// a fresh entry, and a fresh entry is not muted.
	muted: bool,
	/// How many times its tick has failed since it was built.
	faults: u32,
	/// Roughly how many instructions its last tick spent.
	///
	/// Rounded up to a multiple of [`GRAIN`], because that is how often the
	/// interpreter stops to count.
	instructions: u32,
}

/// A program that has to be built, or built again.
struct Pending {
	home: Home,
	program: ScriptId,
	program_revision: u32,
	name: String,
	script: String,
}

/// What running one chunk left behind.
struct Built {
	/// Where `ui.on` filed things, or `None` for a program that has no `ui`.
	handlers: Option<Table>,

	/// The table the chunk ran in.
	environment: Table,

	/// Whether it left a `tick` behind.
	ticks: bool,
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

	/// Runs whatever the interface has for its programs this step.
	///
	/// Loads or reloads any panel whose program has moved, then hands this
	/// step's events to the handlers that asked for them. Called by the host
	/// from inside the step, before the physics and outside the edit-mode
	/// guard. @ref the module docs for why it is there.
	///
	/// @param world - the interface to serve, and to write through
	pub fn interface(&mut self, world: &mut World) {
		let pending = self.pending_panels(world);

		if pending.is_empty() && world.ui.events().is_empty() {
			return;
		}

		self.serve(world, &pending, false);
	}

	/// Runs whatever the world has for its own programs this step.
	///
	/// Loads or reloads any program under `scripts/` that has moved, then ticks
	/// every one of them that asked to be. Called by the host from inside the
	/// step, after the game's `update` and inside the edit-mode guard.
	///
	/// @param world - the world to run against, and to write through
	pub fn gameplay(&mut self, world: &mut World) {
		let pending = self.pending_world(world);

		if pending.is_empty() && !self.anything_ticks() {
			return;
		}

		self.serve(world, &pending, true);
	}

	/// Whether any loaded program would be called if a step ticked now.
	///
	/// The early return above, and the reason a tree with no world programs in
	/// it pays nothing at all for this: no scope is opened and no api is built.
	fn anything_ticks(&self) -> bool {
		self.loaded
			.iter()
			.any(|loaded| loaded.ticks && !loaded.muted)
	}

	/// The panels whose program is missing or out of date.
	fn pending_panels(&self, world: &World) -> Vec<Pending> {
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

		jobs
	}

	/// The world's own programs that are missing or out of date.
	///
	/// Slot order, which is the order the compiler walked the tree in, which is
	/// sorted. Nothing here depends on a hash, which matters because a
	/// screenshot runs these.
	fn pending_world(&self, world: &World) -> Vec<Pending> {
		let mut jobs = Vec::new();

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
	fn serve(&mut self, world: &mut World, pending: &[Pending], ticking: bool) {
		let Self { lua, tables, loaded, spent } = self;
		let events = if ticking { Vec::new() } else { Self::waiting(world) };
		let dt = world.dt;
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

			if ticking {
				Self::tick_all(loaded, spent, &running, dt);
			}

			Ok(())
		});

		if let Err(error) = outcome {
			warn!(%error, "the scripts could not be given their api this step");
		}
	}

	/// Calls every world program that asked to be called.
	///
	/// In the order they are loaded in, which is the order the table was walked
	/// in, which is sorted by name. A program that fails is counted and told
	/// about once; one that spends its whole budget is switched off until it is
	/// built again, because that failure costs the budget every step and an
	/// ordinary one costs nothing.
	fn tick_all(
		loaded: &mut [Loaded],
		spent: &Cell<u32>,
		running: &RefCell<api::Running>,
		dt: f32,
	) {
		for entry in loaded.iter_mut() {
			if entry.muted || !entry.ticks {
				continue;
			}

			Self::tick_one(entry, spent, running, dt);
		}
	}

	/// Calls one program's tick and records what it cost.
	///
	/// Its own function because the two failures want three levels of block
	/// between them and that is the shape a lint refuses; it is also the half
	/// worth naming, since the loop above is only "who" and this is "what
	/// happens to them".
	fn tick_one(entry: &mut Loaded, spent: &Cell<u32>, running: &RefCell<api::Running>, dt: f32) {
		let Some(called) = Self::tick_of(entry) else {
			return;
		};

		*running.borrow_mut() = api::Running {
			panel: entry.home.panel(),
			handlers: entry.handlers.clone(),
			name: entry.name.clone(),
		};
		spent.set(0);

		let outcome = called.call::<()>(dt);
		entry.instructions = spent.get();

		let Err(error) = outcome else {
			return;
		};

		entry.faults = entry.faults.saturating_add(1);

		// **the budget is told apart from an ordinary error by the counter
		// rather than by the message.** The hook is the only thing that can
		// push the count past the ceiling, so a count above it means the hook
		// stopped this call - no string to match and nothing to keep in step
		// with the message.
		if entry.instructions > BUDGET {
			entry.muted = true;

			warn!(
				program = entry.name,
				instructions = entry.instructions,
				"spent its whole budget in one tick and was switched off; edit it or run \
				 `script.reload` to start it again"
			);
		} else if entry.faults == 1 {
			warn!(program = entry.name, %error, "a tick failed");
		}
	}

	/// The function a program wants called this step, if it still wants one.
	///
	/// Read out of the environment rather than resolved once at load, which is
	/// what lets a program stop itself by writing `tick = nil` - including from
	/// inside its own tick.
	fn tick_of(entry: &mut Loaded) -> Option<Function> {
		let environment = entry.environment.as_ref()?;

		match environment.get::<Value>("tick") {
			| Ok(Value::Function(tick)) => Some(tick),
			// a program that wrote `tick = nil`, from its own tick or
			// anywhere else, has said it is done.
			| Ok(_) => None,
			| Err(error) => {
				entry.faults = entry.faults.saturating_add(1);
				if entry.faults == 1 {
					warn!(program = entry.name, %error, "the tick could not be read");
				}

				None
			},
		}
	}

	/// Writes a line per program the interpreter is running.
	///
	/// What `script.status` asks for. Written from here rather than from the
	/// command itself because these numbers are the interpreter's, while a
	/// [`ConsoleFn`](colby_core::abi::ConsoleFn) is handed nothing but a
	/// world. So the command leaves a mark and the next step answers it, which
	/// is the shape a scene load already has.
	pub fn report(&self) {
		for entry in &self.loaded {
			info!(
				program = entry.name,
				world = matches!(entry.home, Home::World(_)),
				ticks = entry.ticks,
				muted = entry.muted,
				faults = entry.faults,
				instructions = entry.instructions,
				budget = BUDGET,
				"status"
			);
		}

		info!(programs = self.loaded.len(), "running");
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

		let kept = match built {
			| Ok(built) => {
				// `program` rather than `document`, and it changed when the
				// world got programs of its own: half of what this line is
				// written about has no document at all. The panel reads as
				// nought for those, which is `PanelId::NONE`.
				info!(
					program = job.name,
					panel = job.home.panel().index(),
					handlers = built
						.as_ref()
						.is_some_and(|built| built.handlers.is_some()),
					ticks = built.as_ref().is_some_and(|built| built.ticks),
					"script loaded"
				);

				built
			},
			| Err(error) => {
				warn!(program = job.name, %error, "the program was not loaded");

				// the revision is recorded anyway, so a file that does not
				// compile is reported once rather than sixty times a second.
				// The next edit moves it again and it is tried again. What the
				// last build left running is left running, which is what the
				// shader watcher does with a file the compiler refuses.
				slot.and_then(|slot| loaded.get(slot))
					.and_then(|loaded| {
						Some(Built {
							handlers: loaded.handlers.clone(),
							environment: loaded.environment.clone()?,
							ticks: loaded.ticks,
						})
					})
			},
		};

		let entry = Loaded {
			home: job.home,
			program: job.program,
			program_revision: job.program_revision,
			name: job.name.clone(),
			handlers: kept
				.as_ref()
				.and_then(|built| built.handlers.clone()),
			environment: kept
				.as_ref()
				.map(|built| built.environment.clone()),
			ticks: kept.as_ref().is_some_and(|built| built.ticks),
			// a fresh entry is never muted, which is what makes editing the
			// file the way back from a budget somebody blew.
			muted: false,
			faults: 0,
			instructions: 0,
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
	/// @return what the chunk left behind, or `None` for a program with nothing
	/// in it at all
	fn build(
		lua: &Lua,
		tables: &api::Tables,
		spent: &Cell<u32>,
		running: &RefCell<api::Running>,
		job: &Pending,
	) -> mlua::Result<Option<Built>> {
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
			.set_environment(environment.clone())
			.exec()?;

		// a panel's program is interface logic and is answered by events; one
		// that wants a step is gameplay in the wrong place, and saying so at
		// load is cheaper than a `tick` that is never called and never
		// explained.
		let ticks = matches!(environment.get::<Value>("tick"), Ok(Value::Function(_)));
		if ticks && matches!(job.home, Home::Panel(_)) {
			warn!(
				program = job.name,
				"a document's program is not ticked; a program that has to run every step \
				 belongs under `{}`",
				colby_core::abi::WORLD_PREFIX
			);
		}

		Ok(Some(Built {
			handlers,
			environment,
			ticks: ticks && matches!(job.home, Home::World(_)),
		}))
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
			let value: Value = standard.get(*name)?;
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
