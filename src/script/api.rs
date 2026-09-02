//! What a program can reach, and deliberately nothing more.
//!
//! The environment a program runs in is **built here by hand** rather than
//! being the standard one with the dangerous parts removed. That is the whole
//! enforcement of the split, and it needs no check anywhere: a panel's program
//! is given `ui` and no world, a world program is given the world and no `ui`,
//! and a name nobody declared is `nil`, which is Lua's own answer to being
//! asked for something that is not there. `io`, `os`, `package` and `debug` are
//! never opened at all - @ref [`Vm`](crate::Vm) for which libraries are - and
//! what is left cannot reach the filesystem or the wall clock, which is also
//! what keeps `--shot` reproducible.
//!
//! **A handle is a tagged number** rather than an object - @ref
//! [`handle`](crate::handle) for the four reasons.
//!
//! Three rules the world half is built on, and each one was taken against
//! something plausible.
//!
//! - **A position is three numbers, in and out.** Not a table and not a
//!   userdata vector, because either allocates once per read and a program that
//!   walks every body each step reads hundreds of them. `local x, y, z =
//!   entity.position(h)` costs nothing, and a program that wants arithmetic
//!   with operators on it can write those nine lines in Lua itself.
//! - **A rotation is three angles**, in radians, yaw then pitch then roll, and
//!   what is stored is the quaternion they make. A quaternion is four numbers
//!   nobody writes by hand; angles are what a person means and what this
//!   engine's own camera already is.
//! - **Making and destroying is refused where this machine is not the
//!   authority.** Nothing on the wire says a thing appeared - a snapshot is
//!   matched by slot - so a client that spawned would put every record after it
//!   on the wrong entity, invisibly. It is an error rather than a silent
//!   nothing, and [`colby.is_host`] is how a program asks first. That is the
//!   same shape every engine here uses, spelled `if SERVER then` in the oldest
//!   of them. The refusal goes away when a message that says a thing appeared
//!   exists; it is `NET-1` on the audit list.
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
	abi::{MaterialId, Renderable, World, console, ui::PanelId},
	glam::{EulerRot, Quat, Vec3},
	info, warn,
};
use mlua::{Function, Result, Scope, Table, Value, Variadic};

use crate::handle::{Handle, Kind};

/// Three numbers, or three nothings.
///
/// A tuple of options rather than an option of a tuple, and the difference is
/// what it costs: this pushes three values straight onto the stack, so
/// `local x, y, z = entity.position(h)` allocates nothing at all, while
/// anything returning one value has to build a table or a vector for it. A
/// handle to something that has gone gives `nil, nil, nil`, so the first name
/// on the left is `nil` and `if x then` reads the way it should.
type Triple = (Option<f32>, Option<f32>, Option<f32>);

/// What a handle to something that is not there answers with.
const NOTHING: Triple = (None, None, None);

/// One vector, as three values.
const fn triple(x: f32, y: f32, z: f32) -> Triple { (Some(x), Some(y), Some(z)) }

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

	/// `entity` - what a world program names things in the entity table with.
	pub(crate) entities: Table,

	/// `body` - the same over the solver's table.
	pub(crate) bodies: Table,

	/// What every environment inherits: `string`, `table`, `math`, `print` and
	/// a handful of base functions.
	pub(crate) globals: Table,
}

/// Gives every table its functions for the length of one step.
///
/// Split one function per table below, because one that filled all four would
/// be a hundred and sixty lines of closures and the only thing shared between
/// them is the three lifetimes.
///
/// @param scope - the mlua scope the functions live and die with
/// @param tables - what a program sees
/// @param world - the host state the functions read and write
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
	interface(scope, tables, world, running)?;
	engine(scope, tables, world, running)?;
	entities(scope, tables, world)?;
	placing(scope, tables, world)?;
	drawing(scope, tables, world)?;
	bodies(scope, tables, world)?;

	Ok(())
}

/// `ui` - what a panel's program writes through, and the one thing it reads.
fn interface<'scope, 'env, 'world>(
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

	// the one thing a panel's program can read back, and the field is why: a
	// handler on a search box that cannot see what was typed into it is a
	// handler that can do nothing. Everything else here still writes and never
	// reads - what such a program may see is its own panel and nothing beyond
	// it.
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
		// `width: 37%` written from a program means what the identical words
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

	tables.ui.set("on", on)?;
	tables.ui.set("text", text_of)?;
	tables.ui.set("set_text", set_text)?;
	tables.ui.set("set_classes", set_classes)?;
	tables.ui.set("set_style", set_style)?;

	Ok(())
}

/// `colby` - what every program has, whichever half it belongs to.
fn engine<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
	running: &'env RefCell<Running>,
) -> Result<()>
where
	'world: 'env,
{
	let command = scope.create_function(move |_, line: String| {
		console::run(&mut world.borrow_mut(), &line);

		Ok(())
	})?;

	let describe = scope.create_function(move |_, bits: i64| {
		Ok(Handle::from_bits(bits).map(|handle| handle.to_string()))
	})?;

	// what a program asks before it makes or destroys anything. The answer is
	// `true` for a process on its own, which is every screenshot, every
	// recording and every window nobody has connected to - so a program that
	// never asks still works everywhere except where it would have been wrong.
	let is_host = scope.create_function(move |_, ()| Ok(world.borrow().peer.is_host()))?;

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

	tables.engine.set("command", command)?;
	tables.engine.set("describe", describe)?;
	tables.engine.set("is_host", is_host)?;
	tables.globals.set("print", print)?;

	Ok(())
}

/// `entity` - naming things in the entity table.
fn entities<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
) -> Result<()>
where
	'world: 'env,
{
	let find = scope.create_function(move |_, name: String| {
		let world = world.borrow();
		let found = world
			.entities
			.iter()
			.find(|(id, ..)| world.entities.name(*id) == name)
			.map(|(id, ..)| Handle::of_entity(id).to_bits());

		Ok(found)
	})?;

	let valid = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Entity)? else {
			return Ok(false);
		};

		Ok(world.borrow().entities.alive(handle.entity()))
	})?;

	let name_of = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Entity)? else {
			return Ok(None);
		};

		let world = world.borrow();
		let id = handle.entity();

		Ok(world
			.entities
			.alive(id)
			.then(|| world.entities.name(id).to_owned()))
	})?;

	// an array rather than an iterator, and in slot order, which is the order
	// the table hands them over. A program walking a Lua hash instead would
	// draw in whatever order the interpreter's string seed decided, and two
	// runs of `--shot` would be two pictures. @ref `colby-known-gaps`.
	let all = scope.create_function(move |lua, ()| {
		let world = world.borrow();
		let handles: Vec<i64> = world
			.entities
			.iter()
			.map(|(id, ..)| Handle::of_entity(id).to_bits())
			.collect();

		lua.create_sequence_from(handles)
	})?;

	let spawn = scope.create_function(move |_, ()| {
		let mut world = world.borrow_mut();
		authorized(&world, "spawn an entity")?;

		Ok(Handle::of_entity(world.entities.spawn()).to_bits())
	})?;

	let despawn = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Entity)? else {
			return Ok(false);
		};

		let mut world = world.borrow_mut();
		authorized(&world, "despawn an entity")?;

		Ok(world.entities.despawn(handle.entity()))
	})?;

	let set_name = scope.create_function(move |_, (bits, name): (Option<i64>, String)| {
		let Some(handle) = taken(bits, Kind::Entity)? else {
			return Ok(false);
		};

		Ok(world
			.borrow_mut()
			.entities
			.set_name(handle.entity(), &name))
	})?;

	tables.entities.set("find", find)?;
	tables.entities.set("valid", valid)?;
	tables.entities.set("name", name_of)?;
	tables.entities.set("all", all)?;
	tables.entities.set("spawn", spawn)?;
	tables.entities.set("despawn", despawn)?;
	tables.entities.set("set_name", set_name)?;

	Ok(())
}

/// `entity` again - where a thing is, and how it is turned and sized.
///
/// Its own function because one that filled the whole table would be a
/// hundred and eighty lines of closures; the split is by what the three
/// halves are about rather than by size - what a thing *is*, where it *is*,
/// and how it *looks*.
fn placing<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
) -> Result<()>
where
	'world: 'env,
{
	let position = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Entity)? else {
			return Ok(NOTHING);
		};

		Ok(world
			.borrow()
			.entities
			.transform(handle.entity())
			.map_or(NOTHING, |at| triple(at.position.x, at.position.y, at.position.z)))
	})?;

	let set_position =
		scope.create_function(move |_, (bits, x, y, z): (Option<i64>, f32, f32, f32)| {
			let Some(handle) = taken(bits, Kind::Entity)? else {
				return Ok(false);
			};

			let mut world = world.borrow_mut();
			let Some(at) = world.entities.transform_mut(handle.entity()) else {
				return Ok(false);
			};

			at.position = Vec3::new(x, y, z);

			Ok(true)
		})?;

	let angles = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Entity)? else {
			return Ok(NOTHING);
		};

		Ok(world
			.borrow()
			.entities
			.transform(handle.entity())
			.map_or(NOTHING, |at| {
				let (yaw, pitch, roll) = at.rotation.to_euler(EulerRot::YXZ);

				triple(yaw, pitch, roll)
			}))
	})?;

	let set_angles = scope.create_function(
		move |_, (bits, yaw, pitch, roll): (Option<i64>, f32, f32, f32)| {
			let Some(handle) = taken(bits, Kind::Entity)? else {
				return Ok(false);
			};

			let mut world = world.borrow_mut();
			let Some(at) = world.entities.transform_mut(handle.entity()) else {
				return Ok(false);
			};

			at.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);

			Ok(true)
		},
	)?;

	let scale = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Entity)? else {
			return Ok(NOTHING);
		};

		Ok(world
			.borrow()
			.entities
			.transform(handle.entity())
			.map_or(NOTHING, |at| triple(at.scale.x, at.scale.y, at.scale.z)))
	})?;

	let set_scale =
		scope.create_function(move |_, (bits, x, y, z): (Option<i64>, f32, f32, f32)| {
			let Some(handle) = taken(bits, Kind::Entity)? else {
				return Ok(false);
			};

			let mut world = world.borrow_mut();
			let Some(at) = world.entities.transform_mut(handle.entity()) else {
				return Ok(false);
			};

			at.scale = Vec3::new(x, y, z);

			Ok(true)
		})?;

	// the one thing the host cannot work out for itself: a jump. Everything
	// else is interpolated between where a thing was and where it is, and a
	// teleport drawn that way is a thing sliding across the map. @ref
	// `colby_core::abi::entity`.
	let snap = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Entity)? else {
			return Ok(false);
		};

		Ok(world.borrow_mut().entities.snap(handle.entity()))
	})?;

	tables.entities.set("position", position)?;
	tables
		.entities
		.set("set_position", set_position)?;
	tables.entities.set("angles", angles)?;
	tables.entities.set("set_angles", set_angles)?;
	tables.entities.set("scale", scale)?;
	tables.entities.set("set_scale", set_scale)?;
	tables.entities.set("snap", snap)?;
	Ok(())
}

/// `entity` a third time - what it is drawn as.
///
/// A mesh and a material are **named**, not handed over: a name is what a
/// program can write down and a handle into a registry is what the host
/// resolved it to. A name nothing answers to draws nothing rather than
/// failing, which is the rule the whole asset side already follows.
fn drawing<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
) -> Result<()>
where
	'world: 'env,
{
	let draw = scope.create_function(
		move |_, (bits, mesh, material): (Option<i64>, String, Option<String>)| {
			let Some(handle) = taken(bits, Kind::Entity)? else {
				return Ok(false);
			};

			let mut world = world.borrow_mut();
			let mesh = world.meshes.find(&mesh);
			let material =
				material.map_or(MaterialId::DEFAULT, |named| world.materials.find(&named));
			let id = handle.entity();
			let color = world
				.entities
				.renderable(id)
				.map_or(Vec3::ONE, |drawn| drawn.color);

			Ok(world.entities.set_renderable(id, Renderable {
				mesh,
				material,
				color,
				..Renderable::NOTHING
			}))
		},
	)?;

	let set_color =
		scope.create_function(move |_, (bits, r, g, b): (Option<i64>, f32, f32, f32)| {
			let Some(handle) = taken(bits, Kind::Entity)? else {
				return Ok(false);
			};

			let mut world = world.borrow_mut();
			let Some(drawn) = world.entities.renderable_mut(handle.entity()) else {
				return Ok(false);
			};

			drawn.color = Vec3::new(r, g, b);

			Ok(true)
		})?;

	tables.entities.set("draw", draw)?;
	tables.entities.set("set_color", set_color)?;
	Ok(())
}

/// Refuses to make or destroy anything where this machine is not the
/// authority.
///
/// **Nothing on the wire says a thing appeared.** A snapshot is a per-slot
/// record matched by slot, so a client that spawned would move every slot after
/// it and start driving the wrong things - a failure that is silent, total, and
/// looks like the network being wrong rather than the program. So the table
/// this end builds has to be the table the host builds, and the only way to
/// promise that is to let nobody but the host build one.
///
/// A process on its own is [`PeerId::HOST`](colby_core::abi::PeerId::HOST), so
/// a screenshot, a recording and a window nobody has joined are all unaffected;
/// only a client refuses. The way to write a program that runs at both ends is
/// to ask [`colby.is_host`] first.
///
/// @param world - the one being asked
/// @param doing - what was being attempted, for the message
fn authorized(world: &World, doing: &str) -> Result<()> {
	if world.peer.is_host() {
		return Ok(());
	}

	Err(mlua::Error::runtime(format!(
		"this end may not {doing}: nothing on the wire says a thing appeared, so only the \
		 authority may build the table. Ask `colby.is_host()` first"
	)))
}

/// `body` - the same over the solver's table.
fn bodies<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
) -> Result<()>
where
	'world: 'env,
{
	let find = scope.create_function(move |_, name: String| {
		let world = world.borrow();
		let found = world
			.bodies
			.iter()
			.find(|(id, _)| world.bodies.name(*id) == name)
			.map(|(id, _)| Handle::of_body(id).to_bits());

		Ok(found)
	})?;

	let valid = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(false);
		};

		Ok(world.borrow().bodies.alive(handle.body()))
	})?;

	let name_of = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(None);
		};

		let world = world.borrow();
		let id = handle.body();

		Ok(world
			.bodies
			.alive(id)
			.then(|| world.bodies.name(id).to_owned()))
	})?;

	let all = scope.create_function(move |lua, ()| {
		let world = world.borrow();
		let handles: Vec<i64> = world
			.bodies
			.iter()
			.map(|(id, _)| Handle::of_body(id).to_bits())
			.collect();

		lua.create_sequence_from(handles)
	})?;

	tables.bodies.set("find", find)?;
	tables.bodies.set("valid", valid)?;
	tables.bodies.set("name", name_of)?;
	tables.bodies.set("all", all)?;

	Ok(())
}

/// Reads one argument as a handle into the table that is asking.
///
/// **A handle of the wrong kind is an error naming both**, which is the whole
/// point of the tag: the two tables here are dense and both start at slot
/// nought generation one, so an entity handle handed to `body.name` would
/// otherwise be a confident answer about a different thing. A number that is
/// not a handle at all is refused the same way.
///
/// `nil` is neither, and is not an error: it is what `find` answers with when
/// nothing has that name, and `body.valid(body.find("nope"))` is a chain
/// somebody will write on the first day.
///
/// @param bits - what the program passed
/// @param wanted - the table doing the asking
/// @return the handle, or `None` for `nil`
fn taken(bits: Option<i64>, wanted: Kind) -> Result<Option<Handle>> {
	let Some(bits) = bits else {
		return Ok(None);
	};

	let Some(handle) = Handle::from_bits(bits) else {
		return Err(mlua::Error::runtime(format!(
			"{bits} is a number rather than a handle; {}.find gives one",
			wanted.table()
		)));
	};

	if handle.kind() != wanted {
		return Err(mlua::Error::runtime(format!(
			"that is {}, and this is the {} table",
			handle,
			wanted.table()
		)));
	}

	Ok(Some(handle))
}
