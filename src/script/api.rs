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
//! - **A spawn takes a table and everything else takes arguments.** A
//!   constructor with seven optional parts is unreadable positionally and would
//!   break every caller the day an eighth appeared; a read is called hundreds
//!   of times a step and must not allocate. Spawning is rare, so it is the one
//!   place a table is the right shape and the only place one is taken.
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
	abi::{
		Body, BodyKind, ConsoleFn, Key, Layers, MaterialId, Renderable, Shape, TraceInfo,
		Transform, Voice, World, console, cvar::Owner, ui::PanelId,
	},
	glam::{EulerRot, Quat, Vec3},
	info, warn,
};
use mlua::{Function, Result, Scope, Table, Value, Variadic};

use crate::handle::{Handle, Kind};

/// Two points and a color, which is what almost everything drawn over the
/// world is asked with.
type Segment = (f32, f32, f32, f32, f32, f32, f32, f32, f32);

/// What a swept box is asked with: from, to, half-extents, and one body to
/// pretend is not there.
///
/// A named type because ten arguments in a closure's signature is what the
/// lint asks to be given a name, and because the order is worth writing down
/// once rather than reading out of a tuple.
type Swept = (f32, f32, f32, f32, f32, f32, f32, f32, f32, Option<i64>);

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

/// One command that was typed and belongs to a program.
///
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

	/// Where `colby.publish` files what it is given, and what a command that
	/// was typed is looked up in.
	pub(crate) published: Option<Table>,
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

	/// `input` - this machine's keyboard and mouse.
	pub(crate) input: Table,

	/// `sound` - what a program makes audible.
	pub(crate) sounds: Table,

	/// `draw` - lines and words over the world, for one step.
	pub(crate) marks: Table,

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
/// @param publish - the runner's stand-in for every command a program
/// publishes, which is the one thing here this crate cannot write for itself
pub(crate) fn fill<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
	running: &'env RefCell<Running>,
	publish: ConsoleFn,
) -> Result<()>
where
	'world: 'env,
{
	interface(scope, tables, world, running)?;
	engine(scope, tables, world, running, publish)?;
	entities(scope, tables, world)?;
	placing(scope, tables, world)?;
	drawing(scope, tables, world)?;
	bodies(scope, tables, world)?;
	moving(scope, tables, world)?;
	pushing(scope, tables, world)?;
	resting(scope, tables, world)?;
	touching(scope, tables, world)?;
	asking(scope, tables, world)?;
	typing(scope, tables, world)?;
	sounding(scope, tables, world)?;
	marking(scope, tables, world)?;

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
	publish: ConsoleFn,
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

	// **a program publishing a command is the same discipline the sandbox is
	// already written in**: every action is a named function with a console
	// command in front of it and an optional target name. That is what makes
	// gameplay drivable from a script with no mouse in it, and it is the layer
	// the wire uses for remote calls - so a program that publishes one is
	// reachable by everything that already reaches the rest.
	let publish =
		scope.create_function(move |_, (name, help, handler): (String, String, Function)| {
			let running = running.borrow();
			let Some(filed) = running.published.as_ref() else {
				return Err(mlua::Error::runtime(
					"only a program under `scripts/` can publish a command: a document's \
					 program is interface logic, and its life is its panel's",
				));
			};

			let mut world = world.borrow_mut();
			let taken = world
				.cvars
				.get(&name)
				.is_some_and(|entry| entry.owner() != Owner::Script);

			if taken {
				return Err(mlua::Error::runtime(format!(
					"`{name}` is already the engine's or the game's; pick a name of your own"
				)));
			}

			// attributed to the interpreter rather than to whatever the host
			// last set, which while a module is loaded is the module - and a
			// command dropped when the gameplay dylib reloads is exactly what
			// this must not be.
			let was = Owner::Module;
			world.cvars.attribute(Owner::Script);
			world.cvars.command(&name, publish, &help);
			world.cvars.attribute(was);

			filed.set(name, handler)
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

	tables.engine.set("command", command)?;
	tables.engine.set("describe", describe)?;
	tables.engine.set("is_host", is_host)?;
	tables.engine.set("publish", publish)?;
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

	let set_name = scope.create_function(move |_, (bits, name): (Option<i64>, String)| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(false);
		};

		Ok(world
			.borrow_mut()
			.bodies
			.set_name(handle.body(), &name))
	})?;

	let spawn = scope.create_function(move |_, options: Table| {
		let mut world = world.borrow_mut();
		authorized(&world, "spawn a body")?;

		let body = described(&options)?;

		Ok(Handle::of_body(world.bodies.spawn(body)).to_bits())
	})?;

	let despawn = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(false);
		};

		let mut world = world.borrow_mut();
		authorized(&world, "despawn a body")?;

		Ok(world.bodies.despawn(handle.body()))
	})?;

	// the entity a body drives, which is the pair a program spends its whole
	// time going between: the body is what the solver moves and the entity is
	// what is drawn.
	let entity_of = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(None);
		};

		Ok(world
			.borrow()
			.bodies
			.get(handle.body())
			.map(|body| Handle::of_entity(body.entity).to_bits()))
	})?;

	// and the way back, which is a walk rather than a field: a body names its
	// entity and an entity names nothing. The table is bounded at a thousand,
	// so this is a scan a program may do once and keep the answer to.
	let of_entity = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Entity)? else {
			return Ok(None);
		};

		let world = world.borrow();
		let wanted = handle.entity();

		Ok(world
			.bodies
			.iter()
			.find(|(_, body)| body.entity == wanted)
			.map(|(id, _)| Handle::of_body(id).to_bits()))
	})?;

	tables.bodies.set("find", find)?;
	tables.bodies.set("valid", valid)?;
	tables.bodies.set("name", name_of)?;
	tables.bodies.set("all", all)?;
	tables.bodies.set("set_name", set_name)?;
	tables.bodies.set("spawn", spawn)?;
	tables.bodies.set("despawn", despawn)?;
	tables.bodies.set("entity", entity_of)?;
	tables.bodies.set("of_entity", of_entity)?;

	Ok(())
}

/// `body` again - where it is, how it is moving, and what moves it.
fn moving<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
) -> Result<()>
where
	'world: 'env,
{
	let position = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(NOTHING);
		};

		Ok(world
			.borrow()
			.bodies
			.get(handle.body())
			.map_or(NOTHING, |body| {
				let at = body.transform.position;

				triple(at.x, at.y, at.z)
			}))
	})?;

	// **a teleport rather than a write.** Putting a body somewhere has to move
	// the entity it drives as well and has to say the move was a jump, or the
	// thing is drawn sliding across the map from where it was. There is one
	// call in the engine that does all three, and this is it.
	let teleport =
		scope.create_function(move |_, (bits, x, y, z): (Option<i64>, f32, f32, f32)| {
			let Some(handle) = taken(bits, Kind::Body)? else {
				return Ok(false);
			};

			let mut world = world.borrow_mut();
			let id = handle.body();
			let Some(at) = world.bodies.get(id).map(|body| body.transform) else {
				return Ok(false);
			};

			world.teleport_body(id, Transform { position: Vec3::new(x, y, z), ..at });

			Ok(true)
		})?;

	let velocity = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(NOTHING);
		};

		Ok(world
			.borrow()
			.bodies
			.get(handle.body())
			.map_or(NOTHING, |body| triple(body.velocity.x, body.velocity.y, body.velocity.z)))
	})?;

	// there is no way to apply a *force* to a body anywhere in this engine, so
	// what a program pushing something writes is the speed itself. That is an
	// impulse rather than a force and the difference is real: it ignores the
	// mass. @ref `colby-direction` on what water is waiting for.
	let set_velocity =
		scope.create_function(move |_, (bits, x, y, z): (Option<i64>, f32, f32, f32)| {
			let Some(handle) = taken(bits, Kind::Body)? else {
				return Ok(false);
			};

			let mut world = world.borrow_mut();
			let Some(body) = world.bodies.get_mut(handle.body()) else {
				return Ok(false);
			};

			body.velocity = Vec3::new(x, y, z);
			// anything a program pushes has to be awake to be pushed, and a
			// pile that has settled is asleep. Writing a speed and leaving it
			// asleep is a push nothing acts on.
			body.sleeping = false;

			Ok(true)
		})?;

	let angular = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(NOTHING);
		};

		Ok(world
			.borrow()
			.bodies
			.get(handle.body())
			.map_or(NOTHING, |body| triple(body.angular.x, body.angular.y, body.angular.z)))
	})?;

	let set_angular =
		scope.create_function(move |_, (bits, x, y, z): (Option<i64>, f32, f32, f32)| {
			let Some(handle) = taken(bits, Kind::Body)? else {
				return Ok(false);
			};

			let mut world = world.borrow_mut();
			let Some(body) = world.bodies.get_mut(handle.body()) else {
				return Ok(false);
			};

			body.angular = Vec3::new(x, y, z);
			body.sleeping = false;

			Ok(true)
		})?;

	tables.bodies.set("position", position)?;
	tables.bodies.set("teleport", teleport)?;
	tables.bodies.set("velocity", velocity)?;
	tables.bodies.set("set_velocity", set_velocity)?;
	tables.bodies.set("angular", angular)?;
	tables.bodies.set("set_angular", set_angular)?;

	Ok(())
}

/// `body` a fourth time - what is pushing on it.
///
/// Its own function for the reason the three above are: the split in this file
/// is by subject rather than by size, and a force is a different subject from a
/// position or a speed. It is also the one of the four that a program cannot
/// write itself out of a velocity, because what a force does depends on a mass
/// the table knows and the program does not.
///
/// @param scope - the scope every callback is built in
/// @param tables - the tables to fill
/// @param world - the world the callbacks reach through
fn pushing<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
) -> Result<()>
where
	'world: 'env,
{
	// the two that push rather than *set*. What separates them from
	// `set_velocity` above is the mass: a speed written straight in is the same
	// for a feather and for a crate, and a force divided by what a thing weighs
	// is not. That difference is the whole reason these exist.
	let push =
		scope.create_function(move |_, (bits, x, y, z): (Option<i64>, f32, f32, f32)| {
			let Some(handle) = taken(bits, Kind::Body)? else {
				return Ok(false);
			};

			Ok(world
				.borrow_mut()
				.bodies
				.apply_force(handle.body(), Vec3::new(x, y, z)))
		})?;

	let spin =
		scope.create_function(move |_, (bits, x, y, z): (Option<i64>, f32, f32, f32)| {
			let Some(handle) = taken(bits, Kind::Body)? else {
				return Ok(false);
			};

			Ok(world
				.borrow_mut()
				.bodies
				.apply_torque(handle.body(), Vec3::new(x, y, z)))
		})?;

	tables.bodies.set("push", push)?;
	tables.bodies.set("spin", spin)?;

	Ok(())
}

/// `body` a third time - whether it is being simulated at all.
///
/// Its own function because one that filled the whole table would be a
/// hundred and fifty lines of closures; the split is by subject rather than
/// by size - where a thing is and how fast it is going, against whether
/// anything is moving it and which of them it is one of.
fn resting<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
) -> Result<()>
where
	'world: 'env,
{
	// freezing is `BodyKind::Kinematic` and there is no flag anywhere: in this
	// engine kinematic means gameplay owns the transform and the solver leaves
	// it alone, so nothing writes a frozen thing's position and everything
	// piles on it as though it were the map. @ref `colby-sandbox`.
	let frozen = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(false);
		};

		Ok(world
			.borrow()
			.bodies
			.get(handle.body())
			.is_some_and(|body| body.kind == BodyKind::Kinematic))
	})?;

	let freeze = scope.create_function(move |_, (bits, still): (Option<i64>, bool)| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(false);
		};

		let mut world = world.borrow_mut();
		let Some(body) = world.bodies.get_mut(handle.body()) else {
			return Ok(false);
		};

		// a mesh is never dynamic - a triangle soup has no inside, so no mass
		// distribution and no inertia tensor - so thawing one would be asking
		// the solver for something it cannot answer.
		if !still && !matches!(body.shape.kind, colby_core::abi::ShapeKind::Mesh) {
			body.kind = BodyKind::Dynamic;
			body.sleeping = false;
		} else if still {
			body.kind = BodyKind::Kinematic;
		}

		Ok(true)
	})?;

	let sleeping = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(false);
		};

		Ok(world
			.borrow()
			.bodies
			.get(handle.body())
			.is_some_and(|body| body.sleeping))
	})?;

	let wake = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(false);
		};

		let mut world = world.borrow_mut();
		let Some(body) = world.bodies.get_mut(handle.body()) else {
			return Ok(false);
		};

		body.sleeping = false;

		Ok(true)
	})?;

	// a layer is an identity rather than a filter, which is the rule the whole
	// sandbox is built on: nothing keeps a list of the props, and "which ones"
	// is answered by walking the table and asking. @ref `colby-sandbox`.
	let on_layer = scope.create_function(move |_, (bits, index): (Option<i64>, u32)| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(false);
		};

		Ok(world
			.borrow()
			.bodies
			.get(handle.body())
			.is_some_and(|body| body.layers.layer & Layers::bit(index) != 0))
	})?;

	let set_layer = scope.create_function(move |_, (bits, index): (Option<i64>, u32)| {
		let Some(handle) = taken(bits, Kind::Body)? else {
			return Ok(false);
		};

		let mut world = world.borrow_mut();
		let Some(body) = world.bodies.get_mut(handle.body()) else {
			return Ok(false);
		};

		body.layers.layer = Layers::bit(index);

		Ok(true)
	})?;

	tables.bodies.set("frozen", frozen)?;
	tables.bodies.set("freeze", freeze)?;
	tables.bodies.set("sleeping", sleeping)?;
	tables.bodies.set("wake", wake)?;
	tables.bodies.set("on_layer", on_layer)?;
	tables.bodies.set("set_layer", set_layer)?;
	Ok(())
}

/// `body.touches` - what met what during the step that just ran.
///
/// A queue the step drains rather than a callback, which is the arrangement
/// every event in this engine has and for the same three reasons: no function
/// pointer with a module's lifetime, no gameplay running from inside the
/// solver, and nothing that would make a screenshot stop being reproducible.
/// A touch is offered for exactly one step.
fn touching<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
) -> Result<()>
where
	'world: 'env,
{
	// a table per touch, which is the one shape that reads: a touch has five
	// parts and almost every program wants two of them by name. There are a
	// handful of these a step rather than hundreds, which is what makes the
	// allocation affordable here and not in `position`.
	let touches = scope.create_function(move |lua, ()| {
		let world = world.borrow();
		let out = lua.create_table()?;

		for (at, touch) in world.bodies.touches().iter().enumerate() {
			let one = lua.create_table()?;

			one.set("first", Handle::of_body(touch.first).to_bits())?;
			one.set("second", Handle::of_body(touch.second).to_bits())?;
			one.set("began", touch.kind == colby_core::abi::TouchKind::Began)?;
			one.set("x", touch.point.x)?;
			one.set("y", touch.point.y)?;
			one.set("z", touch.point.z)?;
			one.set("nx", touch.normal.x)?;
			one.set("ny", touch.normal.y)?;
			one.set("nz", touch.normal.z)?;

			out.set(at + 1, one)?;
		}

		Ok(out)
	})?;

	tables.bodies.set("touches", touches)?;

	Ok(())
}

/// `colby.trace_*` and the three axes - what a program asks the world.
///
/// A trace hands back a **table**, or nothing at all on a miss. The rule is the
/// one `touches` follows: a handful of these happen a step rather than
/// hundreds, and what comes back has six named parts that almost nobody wants
/// positionally. A miss is `nil` rather than a table saying so, because
/// `if colby.trace_ray(..) then` is how it reads and because the caller already
/// has the end point - it passed one in.
///
/// **No layer filter yet.** What a trace can be told is one body to pretend is
/// not there, which is the case everybody meets first: a trace from a thing
/// hits the thing. Seeing *through* a whole class of things wants the mask, and
/// that is not here.
fn asking<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
) -> Result<()>
where
	'world: 'env,
{
	let ray = scope.create_function(
		move |lua,
		      (x1, y1, z1, x2, y2, z2, ignore): (f32, f32, f32, f32, f32, f32, Option<i64>)| {
			let world = world.borrow();
			let info =
				ignoring(TraceInfo::ray(Vec3::new(x1, y1, z1), Vec3::new(x2, y2, z2)), ignore)?;

			reported(lua, &world.trace_ray(&info))
		},
	)?;

	let sweep = scope.create_function(
		move |lua, (x1, y1, z1, x2, y2, z2, hx, hy, hz, ignore): Swept| {
			let world = world.borrow();
			let info = ignoring(
				TraceInfo::swept(
					Vec3::new(x1, y1, z1),
					Vec3::new(x2, y2, z2),
					Vec3::new(hx, hy, hz),
				),
				ignore,
			)?;

			reported(lua, &world.trace_box(&info))
		},
	)?;

	// simulated seconds and steps, not the wall clock. There is deliberately
	// no way to ask what time it is: a program that read a real clock would
	// make two runs of `--shot` two pictures, which is the property the whole
	// interpreter is arranged around.
	let time = scope.create_function(move |_, ()| Ok(world.borrow().time))?;
	let steps = scope.create_function(move |_, ()| Ok(world.borrow().steps))?;
	let editing = scope.create_function(move |_, ()| Ok(world.borrow().editing))?;

	tables.engine.set("trace_ray", ray)?;
	tables.engine.set("trace_box", sweep)?;
	tables.engine.set("time", time)?;
	tables.engine.set("steps", steps)?;
	tables.engine.set("editing", editing)?;

	// the three axes a thing is turned along, which is what gameplay actually
	// wants out of a rotation: "which way is this facing" rather than "what
	// four numbers is it". Exact, and three multiplies rather than the trig a
	// program would otherwise write against the angles.
	for (name, axis) in [("forward", Vec3::NEG_Z), ("up", Vec3::Y), ("right", Vec3::X)] {
		let along = scope.create_function(move |_, bits: Option<i64>| {
			let Some(handle) = taken(bits, Kind::Entity)? else {
				return Ok(NOTHING);
			};

			Ok(world
				.borrow()
				.entities
				.transform(handle.entity())
				.map_or(NOTHING, |at| {
					let way = at.rotation * axis;

					triple(way.x, way.y, way.z)
				}))
		})?;

		tables.entities.set(name, along)?;
	}

	Ok(())
}

/// Adds the one body a trace is told to pretend is not there.
fn ignoring(info: TraceInfo, ignore: Option<i64>) -> Result<TraceInfo> {
	let Some(handle) = taken(ignore, Kind::Body)? else {
		return Ok(info);
	};

	Ok(info.ignoring(handle.body()))
}

/// What a trace hands back, or nothing at all.
fn reported(lua: &mlua::Lua, result: &colby_core::abi::TraceResult) -> Result<Option<Table>> {
	if !result.hit {
		return Ok(None);
	}

	let out = lua.create_table()?;

	out.set("body", Handle::of_body(result.body).to_bits())?;
	out.set("entity", Handle::of_entity(result.entity).to_bits())?;
	out.set("x", result.end.x)?;
	out.set("y", result.end.y)?;
	out.set("z", result.end.z)?;
	out.set("nx", result.normal.x)?;
	out.set("ny", result.normal.y)?;
	out.set("nz", result.normal.z)?;
	out.set("fraction", result.fraction)?;
	out.set("inside", result.started_solid)?;

	Ok(Some(out))
}

/// `input` - this machine's keyboard and mouse, and nobody else's.
///
/// **What is here crosses to nothing.** A key held on one machine means nothing
/// on another, and what a host is told about somebody else's hands is a
/// `Command` rather than this - so a program that moves a player with these
/// works on a machine playing alone and on a host, and does nothing at all on a
/// client, where the host decides. @ref `colby-networking`.
fn typing<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
) -> Result<()>
where
	'world: 'env,
{
	// a key is named rather than numbered, and the names are the engine's own
	// - one spelling in the process, so a program and a config file mean the
	// same thing by `w`. @ref `colby_core::abi::Key::named`.
	for (name, ask) in [("held", 0_u8), ("pressed", 1), ("released", 2)] {
		let asked = scope.create_function(move |_, key: String| {
			let Some(key) = Key::named(&key) else {
				return Err(mlua::Error::runtime(format!("`{key}` is not a key colby knows")));
			};

			let world = world.borrow();

			Ok(match ask {
				| 0 => world.input.held(key),
				| 1 => world.input.pressed(key),
				| _ => world.input.released(key),
			})
		})?;

		tables.input.set(name, asked)?;
	}

	for (name, ask) in [("mouse_held", 0_u8), ("mouse_pressed", 1), ("mouse_released", 2)] {
		let asked = scope.create_function(move |_, which: String| {
			let button = match which.as_str() {
				| "left" => colby_core::abi::Button::Left,
				| "right" => colby_core::abi::Button::Right,
				| "middle" => colby_core::abi::Button::Middle,
				| other => {
					return Err(mlua::Error::runtime(format!(
						"`{other}` is not a button; colby has left, right and middle"
					)));
				},
			};

			let world = world.borrow();

			Ok(match ask {
				| 0 => world.input.button_held(button),
				| 1 => world.input.button_pressed(button),
				| _ => world.input.button_released(button),
			})
		})?;

		tables.input.set(name, asked)?;
	}

	let cursor = scope.create_function(move |_, ()| {
		let world = world.borrow();

		Ok((world.input.cursor[0], world.input.cursor[1]))
	})?;

	let wheel = scope.create_function(move |_, ()| Ok(world.borrow().input.wheel))?;

	tables.input.set("cursor", cursor)?;
	tables.input.set("wheel", wheel)?;

	Ok(())
}

/// `sound` - what a program makes audible.
///
/// **A voice is a handle the step carries, not the device.** What decides that
/// a sound has ended is the simulation's own playhead, moved by `dt` at the top
/// of every step, so a one-second sound lasts the same number of steps on every
/// machine and a screenshot stays reproducible. Whatever is filling a driver's
/// buffer is downstream of that and is never allowed to decide anything. @ref
/// `colby_core::abi::audio`.
///
/// The handle is generational, which is what makes a program that starts a
/// footstep and forgets the handle safe: four seconds later that handle must
/// not turn down somebody else's music.
fn sounding<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
) -> Result<()>
where
	'world: 'env,
{
	// a name nothing answers to plays silence rather than failing, which is
	// the rule the whole asset side follows: a sound that has not been
	// compiled yet is a world that is quiet rather than a world that stops.
	let play = scope.create_function(
		move |_, (name, x, y, z, looping): (String, f32, f32, f32, Option<bool>)| {
			let mut world = world.borrow_mut();
			let sound = world.sounds.find(&name);
			let mut voice = Voice::at(sound, Vec3::new(x, y, z));

			if looping.unwrap_or(false) {
				voice = voice.looping();
			}

			Ok(Handle::of_voice(world.audio.play(voice)).to_bits())
		},
	)?;

	let play_flat =
		scope.create_function(move |_, (name, looping): (String, Option<bool>)| {
			let mut world = world.borrow_mut();
			let sound = world.sounds.find(&name);
			let mut voice = Voice::flat(sound);

			if looping.unwrap_or(false) {
				voice = voice.looping();
			}

			Ok(Handle::of_voice(world.audio.play(voice)).to_bits())
		})?;

	let stop = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Voice)? else {
			return Ok(false);
		};

		Ok(world.borrow_mut().audio.stop(handle.voice()))
	})?;

	let playing = scope.create_function(move |_, bits: Option<i64>| {
		let Some(handle) = taken(bits, Kind::Voice)? else {
			return Ok(false);
		};

		Ok(world.borrow().audio.alive(handle.voice()))
	})?;

	// what a looping voice following something is: the handle stays and the
	// place is written every step. The sandbox's own hum does exactly this.
	let move_to =
		scope.create_function(move |_, (bits, x, y, z): (Option<i64>, f32, f32, f32)| {
			let Some(handle) = taken(bits, Kind::Voice)? else {
				return Ok(false);
			};

			let mut world = world.borrow_mut();
			let Some(voice) = world.audio.get_mut(handle.voice()) else {
				return Ok(false);
			};

			voice.at = Vec3::new(x, y, z);
			voice.positioned = true;

			Ok(true)
		})?;

	let volume = scope.create_function(move |_, (bits, loudness): (Option<i64>, f32)| {
		let Some(handle) = taken(bits, Kind::Voice)? else {
			return Ok(false);
		};

		let mut world = world.borrow_mut();
		let Some(voice) = world.audio.get_mut(handle.voice()) else {
			return Ok(false);
		};

		voice.volume = loudness;

		Ok(true)
	})?;

	tables.sounds.set("play", play)?;
	tables.sounds.set("play_flat", play_flat)?;
	tables.sounds.set("stop", stop)?;
	tables.sounds.set("playing", playing)?;
	tables.sounds.set("move", move_to)?;
	tables.sounds.set("volume", volume)?;

	Ok(())
}

/// `draw` - lines and words over the world, for one step.
///
/// **Everything here expires the moment it was submitted**, which is not a
/// countdown: the table is swept at the *top* of the next step rather than the
/// bottom of this one, because several frames are drawn between two steps and
/// every one of them draws this. So a program that wants a line to stay draws
/// it every step, which is also what makes a line that stops appearing say
/// something.
///
/// Refused work is counted rather than lost - the table is bounded like every
/// other - and the count is on the statistics panel.
fn marking<'scope, 'env, 'world>(
	scope: &'scope Scope<'scope, 'env>,
	tables: &Tables,
	world: &'env RefCell<&'world mut World>,
) -> Result<()>
where
	'world: 'env,
{
	let line = scope.create_function(move |_, (x1, y1, z1, x2, y2, z2, r, g, b): Segment| {
		world.borrow_mut().debug.line(
			Vec3::new(x1, y1, z1),
			Vec3::new(x2, y2, z2),
			Vec3::new(r, g, b),
		);

		Ok(())
	})?;

	let arrow = scope.create_function(move |_, (x1, y1, z1, x2, y2, z2, r, g, b): Segment| {
		world.borrow_mut().debug.arrow(
			Vec3::new(x1, y1, z1),
			Vec3::new(x2, y2, z2),
			Vec3::new(r, g, b),
		);

		Ok(())
	})?;

	let cuboid = scope.create_function(move |_, (x, y, z, hx, hy, hz, r, g, b): Segment| {
		world.borrow_mut().debug.cuboid(
			Vec3::new(x, y, z),
			Vec3::new(hx, hy, hz),
			Quat::IDENTITY,
			Vec3::new(r, g, b),
		);

		Ok(())
	})?;

	let ball = scope.create_function(
		move |_, (x, y, z, radius, r, g, b): (f32, f32, f32, f32, f32, f32, f32)| {
			world
				.borrow_mut()
				.debug
				.ball(Vec3::new(x, y, z), radius, Vec3::new(r, g, b));

			Ok(())
		},
	)?;

	// the one kind that is not a segment, because words are the one thing that
	// cannot be drawn out of lines.
	let label = scope.create_function(
		move |_, (x, y, z, text, r, g, b): (f32, f32, f32, String, f32, f32, f32)| {
			world
				.borrow_mut()
				.debug
				.label(Vec3::new(x, y, z), &text, Vec3::new(r, g, b));

			Ok(())
		},
	)?;

	tables.marks.set("line", line)?;
	tables.marks.set("arrow", arrow)?;
	tables.marks.set("box", cuboid)?;
	tables.marks.set("ball", ball)?;
	tables.marks.set("label", label)?;

	Ok(())
}

/// Reads one spawn table into a body.
///
/// The one place a table crosses, because a constructor with this many optional
/// parts is unreadable positionally and would break every caller the day
/// another appeared. What is required is the shape; everything else has a
/// default that is the ordinary answer.
///
/// @param options - what the program wrote
fn described(options: &Table) -> Result<Body> {
	let shape: String = options.get("shape").unwrap_or_default();
	let mass: f32 = options.get("mass").unwrap_or(1.0);

	let shape = match shape.as_str() {
		| "box" => Shape::cuboid(Vec3::new(
			options.get("hx").unwrap_or(0.5),
			options.get("hy").unwrap_or(0.5),
			options.get("hz").unwrap_or(0.5),
		)),
		| "ball" => Shape::ball(options.get("radius").unwrap_or(0.5)),
		| other => {
			return Err(mlua::Error::runtime(format!(
				"`{other}` is not a shape; colby has box and ball"
			)));
		},
	};

	let at = Transform::at(Vec3::new(
		options.get("x").unwrap_or(0.0),
		options.get("y").unwrap_or(0.0),
		options.get("z").unwrap_or(0.0),
	));

	let kind: String = options.get("kind").unwrap_or_default();
	let mut body = match kind.as_str() {
		| "" | "dynamic" => Body::dynamic(shape, at, mass),
		| "kinematic" => Body::new(BodyKind::Kinematic, shape, at),
		| "static" => Body::new(BodyKind::Static, shape, at),
		| other => {
			return Err(mlua::Error::runtime(format!(
				"`{other}` is not a kind; colby has dynamic, kinematic and static"
			)));
		},
	};

	if let Some(bits) = options.get::<Option<i64>>("entity")? {
		let Some(handle) = taken(Some(bits), Kind::Entity)? else {
			return Err(mlua::Error::runtime("`entity` is not an entity"));
		};

		body = body.driving(handle.entity());
	}

	if let Some(index) = options.get::<Option<u32>>("layer")? {
		body = body.layered(Layers::single(index));
	}

	Ok(body.surfaced(
		options.get("restitution").unwrap_or(0.2),
		options.get("friction").unwrap_or(0.6),
	))
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
