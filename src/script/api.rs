//! What a program can reach, and deliberately nothing more.
//!
//! The environment a script runs in is **built here by hand** rather than being
//! the standard one with the dangerous parts removed. That is the whole
//! enforcement of "this is interface logic, not gameplay": there is no entity
//! table, no body, no camera and no clock anywhere in the process's Lua, so a
//! script cannot ask for one and no check has to refuse it. `io`, `os`,
//! `package` and `debug` are never opened at all - @ref
//! [`Vm`](crate::Vm) for which libraries are - and what is left
//! cannot reach the filesystem or the wall clock, which is also what keeps
//! `--shot` reproducible.
//!
//! Every function here is **scoped**: it is created when a step needs it and
//! destroyed when that step is over. That is mlua's mechanism for handing a
//! callback a `&mut` to something the VM does not own, and it is what lets a
//! handler write into the host's [`World`] with no `unsafe` anywhere and
//! without Lua holding a copy of anything. The cost is one rule for whoever
//! writes the script: a function kept in a local between calls - `local set =
//! ui.set_text` - has expired by the next step, and calling it is a Lua error
//! rather than anything worse.

use std::cell::RefCell;

use colby_asset::css;
use colby_core::{
	abi::{World, console, ui::PanelId},
	info, warn,
};
use mlua::{Function, Result, Scope, Table, Value, Variadic};

/// The event names a script may ask for, which are the ones a document can
/// produce.
pub(crate) const KINDS: [&str; 3] = ["press", "release", "click"];

/// Which program the VM is inside at this instant.
///
/// The api functions are made once per step and shared by every program, so
/// "which panel does `set_text` write to" cannot be captured when they are
/// built. This is the answer, written by the VM immediately before it hands
/// control to Lua and read by the functions when they are called.
#[derive(Default)]
pub(crate) struct Running {
	/// The panel whose program is running, or `NONE` for a world program.
	pub(crate) panel: PanelId,

	/// Where `ui.on` files what it is given.
	pub(crate) handlers: Option<Table>,

	/// The program's own name, for anything that is logged.
	pub(crate) name: String,
}

/// The tables a script sees, whose fields this module fills.
///
/// They are made once and kept, because a script holds a reference to them from
/// the moment its chunk runs; only what is *in* them lasts a single step.
pub(crate) struct Tables {
	/// `ui` - the interface a document can write to.
	pub(crate) ui: Table,

	/// `colby` - the one way out to the rest of the engine.
	pub(crate) engine: Table,

	/// What every environment inherits: `string`, `table`, `math`, `print` and
	/// a handful of base functions.
	pub(crate) globals: Table,
}

/// Gives the tables their functions for the length of one step.
///
/// @param scope - the mlua scope the functions live and die with
/// @param tables - what a script sees
/// @param world - the host state the functions write into
/// @param running - which program is being served, written by the caller
pub(crate) fn fill<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
	running: &'env RefCell<Running>,
) -> Result<()>
where
	'world: 'env,
{
	let on = scope.create_function(
		move |lua, (node, kind, handler): (String, String, Function)| {
			if !KINDS.contains(&kind.as_str()) {
				return Err(mlua::Error::runtime(format!(
					"`{kind}` is not an event; colby has {}",
					KINDS.join(", ")
				)));
			}

			let running = running.borrow();
			let Some(handlers) = running.handlers.as_ref() else {
				return Err(mlua::Error::runtime(
					"ui.on is only callable from a document's script",
				));
			};

			// one table per node rather than one key per pair, so that dispatch is
			// two lookups whatever a document registers and so that a node's
			// handlers can be replaced without touching anybody else's.
			let node: Table = match handlers.get(node.as_str())? {
				| Some(known) => known,
				| None => {
					let fresh = lua.create_table()?;
					handlers.set(node.as_str(), fresh.clone())?;

					fresh
				},
			};

			node.set(kind, handler)
		},
	)?;

	// the one thing a script can read back, and the field is why: a handler on
	// a search box that cannot see what was typed into it is a handler that
	// can do nothing. Everything else here still writes and never reads - what
	// a script may see is its own panel and nothing beyond it.
	let text_of = scope.create_function(move |_, node: String| {
		let panel = running.borrow().panel;
		let found = world.borrow().ui.text(panel, &node).to_owned();

		Ok(found)
	})?;

	let set_text = scope.create_function(move |_, (node, text): (String, String)| {
		let panel = running.borrow().panel;
		world
			.borrow_mut()
			.ui
			.set_text(panel, &node, &text);

		Ok(())
	})?;

	let set_classes = scope.create_function(move |_, (node, classes): (String, String)| {
		let panel = running.borrow().panel;
		world
			.borrow_mut()
			.ui
			.set_classes(panel, &node, &classes);

		Ok(())
	})?;

	let set_style = scope.create_function(move |_, (node, text): (String, String)| {
		// the same parser the `style` attribute goes through, so that
		// `width: 37%` written from a script means what the identical words
		// mean in the document. A second way to say it would be a second thing
		// to keep in step.
		let mut warnings = Vec::new();
		let style = css::declarations(&text, &mut warnings);
		let panel = running.borrow().panel;

		if let Some(existing) = world.borrow_mut().ui.style_mut(panel, &node) {
			existing.merge(&style);
		}

		for warning in warnings {
			warn!(node, "{warning}");
		}

		Ok(())
	})?;

	let command = scope.create_function(move |_, line: String| {
		console::run(&mut world.borrow_mut(), &line);

		Ok(())
	})?;

	let print = scope.create_function(move |_, values: Variadic<Value>| {
		let mut said = Vec::with_capacity(values.len());
		for value in values.iter() {
			said.push(value.to_string()?);
		}

		// `program` rather than `document`: half of what prints has no document,
		// and a field that is wrong for half its readers is worse than no field.
		info!(program = running.borrow().name, "{}", said.join("\t"));

		Ok(())
	})?;

	tables.ui.set("on", on)?;
	tables.ui.set("text", text_of)?;
	tables.ui.set("set_text", set_text)?;
	tables.ui.set("set_classes", set_classes)?;
	tables.ui.set("set_style", set_style)?;
	tables.engine.set("command", command)?;
	tables.globals.set("print", print)?;

	Ok(())
}
