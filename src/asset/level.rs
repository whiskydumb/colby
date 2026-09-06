//! Reading and writing a scene somebody wrote: `.scene`, which is JSON.
//!
//! The authored half of the scene format. A `.cscene` is what the engine reads
//! and what it writes for itself; this is where one comes from when nobody has
//! run the engine yet - a level, a prefab, a room laid out in a text editor -
//! and it compiles into exactly the same file a save is.
//!
//! Both directions live here on purpose. The one property that matters about a
//! format with a reader and a writer is that they agree, and the only way to
//! keep them agreeing is a test that runs one into the other - which needs
//! both in front of it.
//!
//! **A plain field is read and written through the record's own table.** A
//! body's mass, a joint's stiffness, an entity's position: each is a row of a
//! [`Field`] table, and the key in the text is the field's name - `position`,
//! `rotation`, `max_impulse`, `shape.radius` as a `radius` inside a `shape`.
//! Nothing here lists what a body's fields are; a field added to the table is
//! a key here the same day, read and written and refused when misspelled. What
//! is still read and written by hand is what a table cannot spell: a
//! *reference*, which the text names and the file numbers, and the three
//! things whose spelling is older than the tables - a body's layers as a
//! number and a list, a joint's two anchors as one list, and `position`
//! standing for the rest of a body's own place.
//!
//! **A source names things and the file numbers them.** A body says
//! `"entity": "crate"` and a joint says `"first": "crate"`, because a person
//! writing a scene knows what things are called and does not know what order
//! they will end up in. The compiled file holds indices instead, and turning
//! one into the other is the whole job of this module. That is also why a name
//! written twice is refused: two things called `crate` make `"entity":
//! "crate"` a question with two answers.
//!
//! **An unknown field is an error rather than a warning.** This is a format
//! colby owns, so there is no third-party file to be lenient about, and the
//! alternative is `"whereabouts": [1, 2, 3]` doing nothing at all while
//! looking exactly like it works - a field somebody guessed at, or the one
//! they meant with a letter out of place. A warning would say so once and
//! scroll away. The same goes for a value of the wrong kind: a mass that is a
//! word is refused, not read as the default. The text has no version, so a
//! file written with the keys the format had before the tables - `at`,
//! `turn`, `max impulse` - is refused naming the key, with the new word beside
//! it.
//!
//! ```text
//! {
//!   "stage":    { "camera": { "position": [0, 2, 5], "target": [0, 0, 0] } },
//!   "entities": [ { "name": "crate", "position": [0, 4, 0], "mesh": "cube" },
//!                 { "name": "lid", "parent": "crate", "position": [0, 0.6, 0], "mesh": "cube" } ],
//!   "bodies":   [ { "entity": "crate", "kind": "dynamic",
//!                   "shape": { "kind": "box", "extents": [0.5, 0.5, 0.5] } } ],
//!   "joints":   [ { "kind": "rope", "first": "crate", "length": 3.0,
//!                   "anchors": [[0, 0, 0], [0, 7, 0]] } ],
//!   "poses":    [ { "name": "hero", "skeleton": "models/hero/rig" } ]
//! }
//! ```
//!
//! **An entity hangs off another by name.** `"parent": "crate"` on the lid
//! above makes its `position`, `rotation` and `scale` its place inside the
//! crate rather than in the world, and it goes wherever the crate goes. A
//! parent nothing answers to, an entity hanging off itself, and a loop are
//! each an error naming the entity; the order the two are written in does not
//! matter.
//!
//! **A pose is a record of its own and an entity names it**, rather than an
//! entity naming a skeleton directly. The two are not the same claim: a model
//! of two materials is two entities moved by one set of bones, and two
//! entities that each named `models/hero/rig` would be two characters walking
//! in step by accident. Naming the pose says which of those is meant.
//!
//! **A source never writes where the bones are.** A pose here is a skeleton
//! and nothing else, and what comes out of it stands in the shape the model
//! was drawn in; whatever an animation does to it afterwards belongs to a
//! save, not to a level. Writing thirty-five transforms per character into a
//! file a person edits would bury the two lines they came to change.

use colby_core::{
	Result,
	abi::{
		Body, BodyId, BodyKind, Camera, EntityId, Field, Joint, JointKind, Layers, Light, MeshId,
		Renderable, Shape, Transform,
		field::{self, Kind},
		scene::{Link, NO_INDEX, Posed, SceneData, Solid, Stage, Thing},
	},
	err,
	glam::{Quat, Vec3},
};

use crate::{
	bytes::count,
	json::{self, Value},
};

/// The extension a scene source is written with.
pub const EXTENSION: &str = "scene";

/// The body a source describes when it says nothing about one.
///
/// A static unit box - and a sphere's radius of half a unit beside it, which
/// a live body's default does not carry. It used to be that a missing `shape`
/// gave a radius of nothing and an empty one gave half a unit, and nothing
/// could see the difference until the writer had to tell the two apart; the
/// reader was made to agree with itself, and this is where it agrees.
const BLANK_BODY: Body = {
	let mut body = Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY);
	body.shape.radius = 0.5;

	body
};

/// The joint a source describes when it says nothing about one: a rope of
/// length one, holding nothing, which is not quite [`Joint::default`] either.
const BLANK_JOINT: Joint = {
	let mut joint =
		Joint::new(JointKind::Rope, BodyId::NONE, BodyId::NONE, (Vec3::ZERO, Vec3::ZERO));
	joint.length = 1.0;

	joint
};

/// The two layer fields, which the text spells by hand as `layer` and
/// `collides`: a number and a list of numbers rather than two bit masks,
/// because that is what a person means.
const LAYER_FIELDS: &[&str] = &["layers.layer", "layers.mask"];

/// The two anchors, which the text spells by hand as one `anchors` list.
const ANCHOR_FIELDS: &[&str] = &["first_anchor", "second_anchor"];

/// The keys the format had before a plain field's key became its field name,
/// so that a file written with one is refused with the new word beside it.
const RENAMED: &[(&str, &str)] = &[
	("at", "position"),
	("turn", "rotation"),
	("max impulse", "max_impulse"),
	("max torque", "max_torque"),
	("fov", "fov_y"),
];

/// Reads a scene out of the text somebody wrote.
///
/// @param text - the whole `.scene` file
/// @return the description, ready to be written as a `.cscene`
///
/// # Errors
///
/// If the text is not JSON, holds a field this build does not know or a value
/// of the wrong kind, names something nothing answers to, or names one thing
/// twice.
pub fn import(text: &str) -> Result<SceneData> {
	let root = json::parse(text)?;
	fields(&root, &["stage", "entities", "bodies", "joints", "poses"], "the scene")?;

	let posed = poses(root.get("poses"))?;
	let things = entities(root.get("entities"), &posed)?;
	let solids = bodies(root.get("bodies"), &things)?;
	let links = joints(root.get("joints"), &solids)?;

	Ok(SceneData {
		stage: stage(root.get("stage"))?,
		thing_generations: vec![1; things.len()],
		solid_generations: vec![1; solids.len()],
		link_generations: vec![1; links.len()],
		pose_generations: vec![1; posed.len()],
		things,
		solids,
		links,
		posed,
		// nothing an author could write. A scene laid out by hand is a world
		// before any game has run in it, and an arena is what a game remembers
		// about one that has - which is as true of a peer's block as of the
		// world's, so neither is here.
		arena: None,
		player_arenas: Vec::new(),
		peer_generations: Vec::new(),
	})
}

/// Refuses a field this build does not know, in a flat object.
///
/// Shared with the project file, which follows the same rule for the same
/// reason. @ref `crate::project`. A record with a table behind it goes
/// through [`check`] instead, which knows the table.
///
/// @param value - the object to check, or nothing
/// @param known - every field that belongs
/// @param what - what to call the object in the message
pub(crate) fn fields(value: &Value, known: &[&str], what: &str) -> Result<()> {
	for (name, _) in value.as_object() {
		if !known.contains(&name.as_str()) {
			return Err(unknown(what, name));
		}
	}

	Ok(())
}

/// The world's own settings, or the ones a world starts with.
fn stage(value: Option<&Value>) -> Result<Stage> {
	let Some(value) = value else {
		return Ok(Stage::DEFAULT);
	};

	check(value, &[names(Stage::FIELDS, &[])], &["camera"], "a stage")?;

	let mut stage = Stage::DEFAULT;
	read(&mut stage, value, Stage::FIELDS, "a stage")?;

	if let Some(lens) = value.get("camera") {
		check(lens, &[names(Camera::FIELDS, &[])], &[], "a camera")?;
		read(&mut stage.camera, lens, Camera::FIELDS, "a camera")?;
	}

	Ok(stage)
}

/// Every pose the source declares.
///
/// Bones are deliberately not readable here - @ref the module's own note. A
/// pose out of a source is a skeleton and a name, and what stands on it stands
/// in the shape its model was drawn in.
fn poses(value: Option<&Value>) -> Result<Vec<Posed>> {
	let mut posed: Vec<Posed> = Vec::new();

	for (index, entry) in listed(value).iter().enumerate() {
		fields(entry, &["name", "skeleton"], "a pose")?;

		let name = text(entry.get("name"));
		once(posed.iter().any(|it| it.name == name), &name, "pose")?;

		posed.push(Posed {
			name,
			slot: count(index, "a scene's records")?,
			generation: 1,
			skeleton: text(entry.get("skeleton")),
			locals: Vec::new(),
		});
	}

	Ok(posed)
}

/// Every entity the source stands, with the pose that moves it looked up.
///
/// Its place comes through the transform's table and its tint through the
/// renderable's; the mesh, the material and the pose are references, and are
/// read by name.
fn entities(value: Option<&Value>, posed: &[Posed]) -> Result<Vec<Thing>> {
	let mut things: Vec<Thing> = Vec::new();

	for (index, entry) in listed(value).iter().enumerate() {
		check(
			entry,
			&[names(Transform::FIELDS, &[]), names(Renderable::FIELDS, &[])],
			&["name", "parent", "light"],
			"an entity",
		)?;

		let name = text(entry.get("name"));
		once(things.iter().any(|it| it.name == name), &name, "entity")?;

		let moved = text(entry.get("pose"));
		let pose = if moved.is_empty() {
			NO_INDEX
		} else {
			count(
				posed
					.iter()
					.position(|it| it.name == moved)
					.ok_or_else(|| {
						err!(Asset("an entity is moved by {moved}, and no pose is that"))
					})?,
				"a scene's records",
			)?
		};

		let mut transform = Transform::IDENTITY;
		read(&mut transform, entry, Transform::FIELDS, "an entity")?;

		let mut look = Renderable::NOTHING;
		read(&mut look, entry, Renderable::FIELDS, "an entity")?;

		// nested rather than folded in beside the other two tables, because a
		// light has a `color` and so does a renderable, and one flat namespace
		// would have to rename one of them. The stage's camera is nested for
		// the same reason and by the same two calls.
		let mut lamp = Light::NONE;
		if let Some(shining) = entry.get("light") {
			check(shining, &[names(Light::FIELDS, &[])], &[], "a light")?;
			read(&mut lamp, shining, Light::FIELDS, "a light")?;
		}

		things.push(Thing {
			name,
			slot: count(index, "a scene's records")?,
			generation: 1,
			transform,
			mesh: text(entry.get("mesh")),
			material: text(entry.get("material")),
			color: look.color,
			light: lamp,
			pose,
			parent: NO_INDEX,
		});
	}

	hang(&mut things, value)?;

	Ok(things)
}

/// Resolves what every entity hangs off, once every name is known.
///
/// A second pass on purpose: a child may be written above the thing it hangs
/// off, and a person is not going to sort a file by depth to please a reader.
/// A loop is caught here too, because one record at a time cannot see one.
///
/// @param things - the entities, read
/// @param value - the list they were read from
fn hang(things: &mut [Thing], value: Option<&Value>) -> Result {
	for (index, entry) in listed(value).iter().enumerate() {
		let named = text(entry.get("parent"));

		if named.is_empty() {
			continue;
		}

		let parent = things
			.iter()
			.position(|it| it.name == named)
			.ok_or_else(|| err!(Asset("an entity hangs off {named}, and no entity is that")))?;

		if parent == index {
			return Err(err!(Asset("{named} hangs off itself")));
		}

		things[index].parent = count(parent, "a scene's records")?;
	}

	// a chain of parents longer than the list has come back to something,
	// which is a loop, and a loop is a chain nothing can resolve. Counting is
	// the whole check: comparing against the record the climb started from
	// would catch only a loop that runs through that record, and the count
	// catches one that does not.
	for thing in &*things {
		let mut above = thing.parent;
		let mut climbed = 0;

		while above != NO_INDEX {
			if climbed > things.len() {
				return Err(err!(Asset("{} hangs off a chain that never ends", thing.name)));
			}

			above = things
				.get(usize::try_from(above).unwrap_or(usize::MAX))
				.map_or(NO_INDEX, |it| it.parent);
			climbed += 1;
		}
	}

	Ok(())
}

/// Every body, with the entity it drives looked up by name.
///
/// The plain fields come through the body's table, over [`BLANK_BODY`]; the
/// entity and the mesh of a mesh shape are references and are read by name,
/// and the layers are read as the number and the list a person writes.
fn bodies(value: Option<&Value>, things: &[Thing]) -> Result<Vec<Solid>> {
	let mut solids: Vec<Solid> = Vec::new();

	for (index, entry) in listed(value).iter().enumerate() {
		check(
			entry,
			&[names(Transform::FIELDS, &[]), names(Body::FIELDS, LAYER_FIELDS)],
			&["name", "entity", "layer", "collides"],
			"a body",
		)?;

		let name = text(entry.get("name"));
		once(solids.iter().any(|it| it.name == name), &name, "body")?;

		let driven = text(entry.get("entity"));
		let thing = if driven.is_empty() {
			NO_INDEX
		} else {
			count(
				things
					.iter()
					.position(|it| it.name == driven)
					.ok_or_else(|| {
						err!(Asset("a body drives {driven}, and no entity is that"))
					})?,
				"a scene's records",
			)?
		};

		let mut body = BLANK_BODY;
		read(&mut body, entry, Body::FIELDS, "a body")?;
		body.layers = layers(entry)?;
		body.transform = standing(entry, things, thing)?;

		let mesh = text(
			entry
				.get("shape")
				.and_then(|shape| shape.get("mesh")),
		);
		let mut solid = Solid::of(&body, mesh, thing);
		solid.name = name;
		solid.slot = count(index, "a scene's records")?;
		solid.generation = 1;

		solids.push(solid);
	}

	Ok(solids)
}

/// Where a body stands: where it says, or where its entity does.
///
/// A body with an entity and no `position` of its own stands where the entity
/// does, which is what a person writing one means by leaving it out. It is
/// the presence of `position` that decides, and a rotation or a scale written
/// without one is refused rather than quietly read against nothing.
///
/// @param entry - the body's object
/// @param things - the entities, read
/// @param thing - which of them it drives, or [`NO_INDEX`]
fn standing(entry: &Value, things: &[Thing], thing: u32) -> Result<Transform> {
	if entry.get("position").is_some() {
		let mut own = Transform::IDENTITY;
		read(&mut own, entry, Transform::FIELDS, "a body")?;

		return Ok(own);
	}

	if entry.get("rotation").is_some() || entry.get("scale").is_some() {
		return Err(err!(Asset(
			"a body turned or sized on its own has to say where it is: there is no position"
		)));
	}

	Ok(things
		.get(usize::try_from(thing).unwrap_or(usize::MAX))
		.map_or(Transform::IDENTITY, |it| it.transform))
}

/// Every joint, with both of its bodies looked up by name.
///
/// The plain fields come through the joint's table, over [`BLANK_JOINT`]; the
/// two bodies are references read by name, and the anchors are one list.
fn joints(value: Option<&Value>, solids: &[Solid]) -> Result<Vec<Link>> {
	let mut links: Vec<Link> = Vec::new();

	for (index, entry) in listed(value).iter().enumerate() {
		check(
			entry,
			&[names(Joint::FIELDS, ANCHOR_FIELDS)],
			&["name", "first", "second", "anchors"],
			"a joint",
		)?;

		let name = text(entry.get("name"));
		once(links.iter().any(|it| it.name == name), &name, "joint")?;

		let mut joint = BLANK_JOINT;
		read(&mut joint, entry, Joint::FIELDS, "a joint")?;

		let anchors = entry.get("anchors").cloned().unwrap_or_default();
		let anchors = anchors.as_array();
		joint.first_anchor = triple(anchors.first(), Vec3::ZERO, "a joint", "anchors")?;
		joint.second_anchor = triple(anchors.get(1), Vec3::ZERO, "a joint", "anchors")?;

		let mut link = Link::of(
			&joint,
			held(entry.get("first"), solids)?,
			held(entry.get("second"), solids)?,
		);
		link.name = name;
		link.slot = count(index, "a scene's records")?;
		link.generation = 1;

		links.push(link);
	}

	Ok(links)
}

/// Which body a joint names, or nothing.
fn held(value: Option<&Value>, solids: &[Solid]) -> Result<u32> {
	let name = text(value);
	if name.is_empty() {
		return Ok(NO_INDEX);
	}

	count(
		solids
			.iter()
			.position(|it| it.name == name)
			.ok_or_else(|| err!(Asset("a joint holds {name}, and no body is that")))?,
		"a scene's records",
	)
}

/// Which layers a body is on and which it interacts with.
///
/// Layers are numbered here rather than written as a mask: `"layer": 2` and
/// `"collides": [0, 2]` are what a person means, and the shifting is this
/// module's job. Leaving both out is layer zero against everything, which is
/// what a body that has never heard of layers is.
fn layers(value: &Value) -> Result<Layers> {
	let on = value.get("layer").map_or(Ok(0), |it| {
		it.as_u32()
			.ok_or_else(|| err!(Asset("a layer is a number from zero to thirty-one")))
	})?;

	let Some(collides) = value.get("collides") else {
		return Ok(Layers::single(on));
	};

	let mut mask = 0;
	for entry in collides.as_array() {
		let with = entry
			.as_u32()
			.ok_or_else(|| err!(Asset("a layer is a number from zero to thirty-one")))?;

		mask |= Layers::bit(with);
	}

	Ok(Layers::new(Layers::bit(on), mask))
}

// -------------------------------------------------------------------------
// a record's plain fields, through its table
// -------------------------------------------------------------------------

/// The keys a table gives a record, dotted for a field inside another.
///
/// @param table - the record's table
/// @param skipped - the plain fields the text spells by hand under other
/// keys, which are therefore not keys
fn names<T>(table: &[Field<T>], skipped: &[&str]) -> Vec<&'static str> {
	table
		.iter()
		.map(|field| field.name)
		.filter(|name| !skipped.contains(name))
		.collect()
}

/// Refuses a field this build does not know, anywhere in a record.
///
/// A key is known when a table names it, when it is read by hand, or when it
/// is the head of a dotted name - `shape` for `shape.radius` - in which case
/// what stands under it is an object and its keys are checked against the
/// tails.
///
/// @param value - the record's object
/// @param tables - the keys each of the record's tables gives it
/// @param hand - the keys read by hand beside the tables
/// @param what - what to call the record in a message
fn check(value: &Value, tables: &[Vec<&'static str>], hand: &[&str], what: &str) -> Result<()> {
	for (key, held) in value.as_object() {
		let key = key.as_str();

		if hand.contains(&key) || tables.iter().any(|names| names.contains(&key)) {
			continue;
		}

		let tails: Vec<&str> = tables
			.iter()
			.flat_map(|names| names.iter())
			.filter_map(|name| name.strip_prefix(key)?.strip_prefix('.'))
			.collect();

		if tails.is_empty() {
			return Err(unknown(what, key));
		}

		if !matches!(held, Value::Object(_)) {
			return Err(err!(Asset(
				"{what}'s {key} is a record of its own, with {}",
				tails.join(", ")
			)));
		}

		for (inner, _) in held.as_object() {
			if !tails.contains(&inner.as_str()) {
				return Err(err!(Asset("{what} has no field called {key}.{inner}")));
			}
		}
	}

	Ok(())
}

/// The error for a key nothing knows, with the new word beside an old one.
fn unknown(what: &str, key: &str) -> colby_core::Error {
	match RENAMED.iter().find(|(old, _)| *old == key) {
		| Some((_, now)) => err!(Asset("{what} has no field called {key}; it is {now} now")),
		| None => err!(Asset("{what} has no field called {key}")),
	}
}

/// Reads every plain field of a record's table out of its object.
///
/// A field the object does not mention keeps what the record holds, which is
/// what leaving one out means. A reference has no spelling and is read by
/// hand beside this. A field the text spells under other keys needs no skip
/// here: [`check`] has already refused its own name as a key, so the object
/// cannot hold one.
///
/// @param record - what to read into, holding the defaults
/// @param entry - the record's object
/// @param table - the record's table
/// @param what - what to call the record in a message
fn read<T>(record: &mut T, entry: &Value, table: &[Field<T>], what: &str) -> Result<()> {
	for field in table {
		if field.kind.is_reference() {
			continue;
		}

		let Some(written) = at_path(entry, field.name) else {
			continue;
		};

		let value = parsed(written, field.kind, &field.get(record), what, field.name)?;

		if !field.set(record, value) {
			return Err(err!(Asset("{what} would not take its {}", field.name)));
		}
	}

	Ok(())
}

/// A value nested by a dotted name: `shape.radius` is `radius` inside `shape`.
fn at_path<'a>(entry: &'a Value, path: &str) -> Option<&'a Value> {
	path.split('.')
		.try_fold(entry, |value, part| value.get(part))
}

/// One written value read as the kind a field holds.
///
/// @param written - what the text says
/// @param kind - what the field holds
/// @param current - what the record holds now, which a shorter list of
/// numbers keeps in the axes it did not mention
/// @param what - what to call the record in a message
/// @param name - what to call the field
fn parsed(
	written: &Value,
	kind: Kind,
	current: &field::Value,
	what: &str,
	name: &str,
) -> Result<field::Value> {
	let wrong = || err!(Asset("{what}'s {name} should be {}", kind.name()));

	Ok(match kind {
		| Kind::Bool => field::Value::Bool(written.as_bool().ok_or_else(wrong)?),
		| Kind::Int => field::Value::Int(whole(written).ok_or_else(wrong)?),
		| Kind::Float => field::Value::Float(written.as_f32().ok_or_else(wrong)?),
		| Kind::Text => field::Value::Text(written.as_str().ok_or_else(wrong)?.to_owned()),
		| Kind::Vec3 => {
			let field::Value::Vec3(held) = *current else {
				return Err(wrong());
			};

			field::Value::Vec3(triple(Some(written), held, what, name)?)
		},
		| Kind::Color => {
			let field::Value::Color(held) = *current else {
				return Err(wrong());
			};

			field::Value::Color(triple(Some(written), held, what, name)?)
		},
		| Kind::Quat => field::Value::Quat(turn(written, what, name)?),
		| Kind::Word(words) => {
			let word = written.as_str().ok_or_else(wrong)?;
			let index = kind.word(word).ok_or_else(|| {
				err!(Asset(
					"{what} has no {name} called {word}; it is one of {}",
					words.join(", ")
				))
			})?;

			field::Value::Word(index)
		},
		| Kind::Entity | Kind::Body | Kind::Joint | Kind::Pose | Kind::Mesh | Kind::Material =>
			return Err(wrong()),
	})
}

/// A whole number, of either sign, or nothing for a number that is not one.
fn whole(value: &Value) -> Option<i64> {
	let number = value.as_f64()?;

	if number < 0.0 {
		return Value::Number(-number)
			.as_u64()
			.and_then(|it| i64::try_from(it).ok())
			.map(|it| -it);
	}

	value
		.as_u64()
		.and_then(|it| i64::try_from(it).ok())
}

/// Three numbers, or nothing at all.
///
/// A shorter list keeps the default in the axes it did not mention, which is
/// what `"color": [1, 0]` most likely meant and is in any case better than
/// silently reading the missing one as zero. A longer one, or anything that
/// is not a list of numbers, is refused.
///
/// @param value - the list, or nothing
/// @param default - what the axes not written hold
/// @param what - what to call the record in a message
/// @param name - what to call the field
fn triple(value: Option<&Value>, default: Vec3, what: &str, name: &str) -> Result<Vec3> {
	let Some(value) = value else {
		return Ok(default);
	};

	let wrong = || err!(Asset("{what}'s {name} should be three numbers"));

	let Value::Array(parts) = value else {
		return Err(wrong());
	};

	if parts.len() > 3 {
		return Err(wrong());
	}

	let mut out = default.to_array();
	for (slot, part) in out.iter_mut().zip(parts) {
		*slot = part.as_f32().ok_or_else(wrong)?;
	}

	Ok(Vec3::from_array(out))
}

/// A rotation, as the four numbers of a unit quaternion.
fn turn(value: &Value, what: &str, name: &str) -> Result<Quat> {
	let Value::Array(parts) = value else {
		return Err(err!(Asset("{what}'s {name} should be a rotation, four numbers xyzw")));
	};

	if parts.len() != 4 || parts.iter().any(|part| part.as_f32().is_none()) {
		return Err(err!(Asset("{what}'s {name} should be a rotation, four numbers xyzw")));
	}

	let rotation = Quat::from_xyzw(
		number(parts.first(), 0.0),
		number(parts.get(1), 0.0),
		number(parts.get(2), 0.0),
		number(parts.get(3), 1.0),
	);

	if !rotation.is_normalized() {
		return Err(err!(Asset("{what}'s {name} has to be a unit quaternion, xyzw")));
	}

	Ok(rotation)
}

/// Refuses a name something else already has.
///
/// The empty name is not a name and any number of records may leave it out; it
/// is what a record nothing refers to holds.
///
/// @param taken - whether anything read so far answers to it
/// @param name - the name in question
/// @param what - what kind of record it is, for the message
fn once(taken: bool, name: &str, what: &str) -> Result<()> {
	if !name.is_empty() && taken {
		return Err(err!(Asset("two {what} records are both called {name}")));
	}

	Ok(())
}

/// The entries of a list, or none.
fn listed(value: Option<&Value>) -> &[Value] { value.map_or(&[], Value::as_array) }

/// A string field, or the empty one.
fn text(value: Option<&Value>) -> String {
	value
		.and_then(Value::as_str)
		.unwrap_or_default()
		.to_owned()
}

/// A number, or a default.
fn number(value: Option<&Value>, default: f32) -> f32 {
	value.and_then(Value::as_f32).unwrap_or(default)
}

// -------------------------------------------------------------------------
// writing one out
// -------------------------------------------------------------------------

/// Writes a scene as the text somebody could have written.
///
/// The other half of [`import`], and the one the editor needs: what a person
/// laid out with a pointer has to end up somewhere a person can read, diff and
/// merge. It writes the *source*, not the compiled file - the compiler turns
/// this into a `.cscene` exactly as it does one typed by hand, so there is one
/// path from a scene to the engine rather than two.
///
/// **A field equal to its default is left out.** That is not tidiness: a file
/// where every record spells out every number is a file where a real change is
/// one line in forty, and the diff is the thing this format exists for. It is
/// also what the one other engine with a text scene format does, for the same
/// stated reason.
///
/// **Records are one to a line.** Moving one prop is then one changed line
/// rather than a block, which is what makes two people editing one scene
/// possible at all.
///
/// **A name is invented only where one is needed.** A body names the entity it
/// drives and a joint names the bodies it holds, so those have to be called
/// something; everything else keeps whatever it was called, including nothing.
/// A name two records share is a question with two answers, which [`import`]
/// refuses - so the second of them is given a number.
///
/// @param scene - what to write
/// @return the text of a `.scene`
///
/// # Errors
///
/// If the description holds a number JSON has no spelling for - an infinity or
/// a nan, which is what a world that has blown up is full of.
pub fn export(scene: &SceneData) -> Result<String> {
	let thing_names = named(
		&scene.things,
		|thing| thing.name.as_str(),
		|index| {
			scene.solids.iter().any(|it| it.thing == index)
				|| scene.things.iter().any(|it| it.parent == index)
		},
		"entity",
	);
	let solid_names = named(
		&scene.solids,
		|solid| solid.name.as_str(),
		|index| {
			scene
				.links
				.iter()
				.any(|it| it.first == index || it.second == index)
		},
		"body",
	);
	let link_names = named(&scene.links, |link| link.name.as_str(), |_| false, "joint");
	let pose_names = named(
		&scene.posed,
		|posed| posed.name.as_str(),
		|index| scene.things.iter().any(|it| it.pose == index),
		"pose",
	);

	let things = scene
		.things
		.iter()
		.zip(&thing_names)
		.map(|(thing, name)| thing_of(thing, name, &thing_names, &pose_names))
		.collect::<Result<Vec<String>>>()?;
	let solids = scene
		.solids
		.iter()
		.zip(&solid_names)
		.map(|(solid, name)| solid_of(scene, solid, name, &thing_names))
		.collect::<Result<Vec<String>>>()?;
	let links = scene
		.links
		.iter()
		.zip(&link_names)
		.map(|(link, name)| link_of(link, name, &solid_names))
		.collect::<Result<Vec<String>>>()?;
	let poses: Vec<String> = scene
		.posed
		.iter()
		.zip(&pose_names)
		.map(|(posed, name)| pose_of(posed, name))
		.collect();

	// gathered rather than appended, so that the comma between two of them is
	// written by whatever knows there is a next one. A trailing one is the
	// single thing JSON refuses that is easy to write by accident.
	let parts: Vec<String> = [
		stage_of(&scene.stage)?,
		block("entities", &things),
		block("bodies", &solids),
		block("joints", &links),
		block("poses", &poses),
	]
	.into_iter()
	.flatten()
	.collect();

	Ok(format!("{{\n{}\n}}\n", parts.join(",\n\n")))
}

/// One name per record: unique, and empty for a record nothing has to name.
///
/// @param records - what to name
/// @param wanted - what each is already called
/// @param needed - whether anything refers to the record at that index
/// @param fallback - what to call one that has to have a name and has none
fn named<T>(
	records: &[T],
	wanted: impl Fn(&T) -> &str,
	needed: impl Fn(u32) -> bool,
	fallback: &str,
) -> Vec<String> {
	let mut taken: Vec<String> = Vec::with_capacity(records.len());

	for (index, record) in records.iter().enumerate() {
		let asked = wanted(record);
		let referred = u32::try_from(index).is_ok_and(&needed);

		if asked.is_empty() && !referred {
			taken.push(String::new());

			continue;
		}

		let base = if asked.is_empty() {
			format!("{fallback} {index}")
		} else {
			asked.to_owned()
		};

		// the first one keeps the name and the rest are numbered, which is
		// what a duplicate is: a copy of something, and the original was here
		// first.
		let mut name = base.clone();
		let mut number = 1_u32;
		while taken.contains(&name) {
			number = number.saturating_add(1);
			name = format!("{base} {number}");
		}

		taken.push(name);
	}

	taken
}

/// The world's own settings, or nothing if they are the ones a world starts
/// with.
fn stage_of(stage: &Stage) -> Result<Option<String>> {
	let mut rows = Rows::default();

	put_all(
		&mut rows,
		&stage.camera,
		&Camera::DEFAULT,
		Camera::FIELDS,
		&Writing {
			prefix: "camera.",
			what: "the stage",
			skipped: &[],
		},
		|_| None,
	)?;
	put_all(
		&mut rows,
		stage,
		&Stage::DEFAULT,
		Stage::FIELDS,
		&Writing {
			prefix: "",
			what: "the stage",
			skipped: &[],
		},
		|_| None,
	)?;

	if rows.is_empty() {
		return Ok(None);
	}

	Ok(Some(format!("\t\"stage\": {}", rows.text())))
}

/// One entity, with what it hangs off named rather than numbered.
fn thing_of(thing: &Thing, name: &str, things: &[String], poses: &[String]) -> Result<String> {
	let mut rows = Rows::default();

	rows.put_text("name", name);
	rows.put_text("parent", &at_index(things, thing.parent));
	put_place(&mut rows, &thing.transform, &Transform::IDENTITY, "an entity")?;

	let look = Renderable {
		color: thing.color,
		..Renderable::NOTHING
	};
	put_all(
		&mut rows,
		&look,
		&Renderable::NOTHING,
		Renderable::FIELDS,
		&Writing {
			prefix: "",
			what: "an entity",
			skipped: &[],
		},
		|field| match field {
			| "mesh" => named_row("mesh", &thing.mesh),
			| "material" => named_row("material", &thing.material),
			| "pose" => named_row("pose", &at_index(poses, thing.pose)),
			| _ => None,
		},
	)?;
	// under its own key, and nothing at all for an entity that is not a lamp:
	// every field matches the default, so `put_all` writes no row and the
	// object never appears. @ref `Rows::text`, which gathers a dotted name
	// into a nested object.
	put_all(
		&mut rows,
		&thing.light,
		&Light::NONE,
		Light::FIELDS,
		&Writing {
			prefix: "light.",
			what: "an entity's light",
			skipped: &[],
		},
		|_| None,
	)?;

	Ok(rows.text())
}

/// One pose: which skeleton it wears, and nothing about where its bones are.
///
/// Where they are is left out on purpose - @ref the module's own note. A
/// source is a level rather than a moment, and thirty-five transforms a
/// character would bury everything a person came to the file to change.
fn pose_of(posed: &Posed, name: &str) -> String {
	let mut rows = Rows::default();

	rows.put_text("name", name);
	rows.put_text("skeleton", &posed.skeleton);

	rows.text()
}

/// One body, with the entity it drives named rather than numbered.
fn solid_of(scene: &SceneData, solid: &Solid, name: &str, things: &[String]) -> Result<String> {
	let mut rows = Rows::default();

	rows.put_text("name", name);
	rows.put_text("entity", &at_index(things, solid.thing));

	// a body with an entity and no place of its own stands where the entity
	// does, which is what leaving it out means on the way in. So the three are
	// written only when they differ from that - and then `position` is
	// written even if it is nothing, because it is its presence that decides.
	let standing = scene
		.things
		.get(usize::try_from(solid.thing).unwrap_or(usize::MAX))
		.map_or(Transform::IDENTITY, |thing| thing.transform);
	put_place(&mut rows, &solid.transform, &standing, "a body")?;

	let body = solid.body(MeshId::NONE, EntityId::NONE);
	put_all(
		&mut rows,
		&body,
		&BLANK_BODY,
		Body::FIELDS,
		&Writing {
			prefix: "",
			what: "a body",
			skipped: LAYER_FIELDS,
		},
		|field| match field {
			| "shape.mesh" => named_row("shape.mesh", &solid.shape.mesh),
			| "layers.layer" => layer_row(solid.layers),
			| "layers.mask" => collides_row(solid.layers),
			| _ => None,
		},
	)?;

	Ok(rows.text())
}

/// One joint, with both bodies named rather than numbered.
fn link_of(link: &Link, name: &str, solids: &[String]) -> Result<String> {
	if !link.first_anchor.is_finite() || !link.second_anchor.is_finite() {
		return Err(err!(Asset("a joint holds a number JSON cannot write")));
	}

	let mut rows = Rows::default();

	rows.put_text("name", name);

	let joint = link.joint(BodyId::NONE, BodyId::NONE);
	put_all(
		&mut rows,
		&joint,
		&BLANK_JOINT,
		Joint::FIELDS,
		&Writing {
			prefix: "",
			what: "a joint",
			skipped: ANCHOR_FIELDS,
		},
		|field| match field {
			| "first" => named_row("first", &at_index(solids, link.first)),
			| "second" => named_row("second", &at_index(solids, link.second)),
			| "first_anchor" =>
				(link.first_anchor != Vec3::ZERO || link.second_anchor != Vec3::ZERO).then(|| {
					(
						"anchors",
						format!(
							"[{}, {}]",
							as_vector(link.first_anchor),
							as_vector(link.second_anchor)
						),
					)
				}),
			| _ => None,
		},
	)?;

	Ok(rows.text())
}

/// Which layer a body is on, back as the number a person writes, or nothing
/// for the layer a body is on anyway.
///
/// @note: the source says one layer per body, so a body somehow on several is
/// written on its lowest and the rest are lost. Nothing in the engine makes
/// one - `Layers::single` and the default both set exactly one bit - and the
/// alternative is a format where `"layer"` is sometimes a list.
fn layer_row(layers: Layers) -> Option<(&'static str, String)> {
	if layers.layer == Layers::DEFAULT.layer {
		return None;
	}

	let on = layers.layer.trailing_zeros().min(u32::BITS - 1);

	Some(("layer", on.to_string()))
}

/// Which layers a body interacts with, as the list a person writes, or
/// nothing for a body that meets everything.
fn collides_row(layers: Layers) -> Option<(&'static str, String)> {
	if layers.mask == u32::MAX {
		return None;
	}

	let with: Vec<String> = (0..u32::BITS)
		.filter(|index| layers.mask & Layers::bit(*index) != 0)
		.map(|index| index.to_string())
		.collect();

	Some(("collides", format!("[{}]", with.join(", "))))
}

/// A row naming something, or nothing when there is nothing to name.
fn named_row(name: &'static str, value: &str) -> Option<(&'static str, String)> {
	if value.is_empty() {
		return None;
	}

	Some((name, as_text(value)))
}

/// Where something is, if it is anywhere other than where it would be anyway.
///
/// The three go together: `position` is written even when it is nothing,
/// because [`import`] reads a body's rotation and scale only in its presence,
/// so writing one of them without it would quietly lose the other two.
fn put_place(
	rows: &mut Rows,
	transform: &Transform,
	otherwise: &Transform,
	what: &str,
) -> Result<()> {
	if transform == otherwise {
		return Ok(());
	}

	if !transform.position.is_finite() {
		return Err(err!(Asset("{what} holds a number JSON cannot write")));
	}

	rows.put("position", as_vector(transform.position));
	put_all(
		rows,
		transform,
		&Transform::IDENTITY,
		Transform::FIELDS,
		&Writing { prefix: "", what, skipped: &["position"] },
		|_| None,
	)
}

/// How one table is written: under what prefix, called what, and which of
/// its plain fields are spelled by hand under other keys instead.
struct Writing<'a> {
	/// What goes in front of every key, `camera.` for a record inside another.
	prefix: &'a str,

	/// What to call the record in a message.
	what: &'a str,

	/// The plain fields the text spells by hand, which the table never
	/// writes; the same list the reader skips, so the two agree.
	skipped: &'a [&'a str],
}

/// Writes every plain field of a record's table that differs from what the
/// reader would produce from nothing.
///
/// A reference has no spelling, and neither does a field spelled by hand
/// under other keys; for those the caller is asked, at the field's place in
/// the table, so that a row it writes lands where the field would have.
///
/// @param rows - where the rows go
/// @param record - what to write
/// @param otherwise - what the reader produces from nothing, which is left out
/// @param table - the record's table
/// @param writing - the prefix, the name and the hand-written fields
/// @param hand - a row for a field the table cannot spell, or nothing
fn put_all<T, F>(
	rows: &mut Rows,
	record: &T,
	otherwise: &T,
	table: &[Field<T>],
	writing: &Writing<'_>,
	hand: F,
) -> Result<()>
where
	F: Fn(&'static str) -> Option<(&'static str, String)>,
{
	for field in table {
		if let Some((name, spelled)) = hand(field.name) {
			rows.put(&format!("{}{name}", writing.prefix), spelled);

			continue;
		}

		if field.kind.is_reference() || writing.skipped.contains(&field.name) {
			continue;
		}

		let value = field.get(record);
		if value == field.get(otherwise) {
			continue;
		}

		if !value.is_finite() {
			return Err(err!(Asset("{} holds a number JSON cannot write", writing.what)));
		}

		if let Some(spelled) = spelling(field.kind, &value) {
			rows.put(&format!("{}{}", writing.prefix, field.name), spelled);
		}
	}

	Ok(())
}

/// One value, spelled the way the reader reads it back, or nothing for a
/// reference, which has no spelling.
fn spelling(kind: Kind, value: &field::Value) -> Option<String> {
	Some(match value {
		| field::Value::Bool(held) => held.to_string(),
		| field::Value::Int(held) => held.to_string(),
		| field::Value::Float(held) => as_number(*held),
		| field::Value::Text(held) => as_text(held),
		| field::Value::Vec3(held) | field::Value::Color(held) => as_vector(*held),
		| field::Value::Quat(held) => as_turn(*held),
		| field::Value::Word(index) => {
			let word = usize::try_from(*index)
				.ok()
				.and_then(|index| kind.words().get(index))?;

			as_text(word)
		},
		| field::Value::Entity(_)
		| field::Value::Body(_)
		| field::Value::Joint(_)
		| field::Value::Pose(_)
		| field::Value::Mesh(_)
		| field::Value::Material(_) => return None,
	})
}

/// The rows of one record on their way out, in the order they were put.
///
/// A dotted name is a field inside another: at writing time every row whose
/// name shares a head is gathered into one object, at the place the first of
/// them was put, so that `shape.kind` and `shape.radius` come out as one
/// `shape`.
#[derive(Default)]
struct Rows {
	rows: Vec<(String, String)>,
}

impl Rows {
	/// Puts one row, already spelled.
	fn put(&mut self, name: &str, spelled: String) { self.rows.push((name.to_owned(), spelled)); }

	/// Puts a text row, unless the text is empty.
	fn put_text(&mut self, name: &str, value: &str) {
		if value.is_empty() {
			return;
		}

		self.put(name, as_text(value));
	}

	/// Whether nothing was put.
	fn is_empty(&self) -> bool { self.rows.is_empty() }

	/// The record as one line of JSON.
	fn text(&self) -> String {
		let mut written: Vec<String> = Vec::new();
		let mut gathered: Vec<&str> = Vec::new();

		for (name, spelled) in &self.rows {
			let Some((head, _)) = name.split_once('.') else {
				written.push(format!("\"{name}\": {spelled}"));

				continue;
			};

			if gathered.contains(&head) {
				continue;
			}

			gathered.push(head);

			let inner: Vec<String> = self
				.rows
				.iter()
				.filter_map(|(other, spelled)| {
					let (found, tail) = other.split_once('.')?;

					(found == head).then(|| format!("\"{tail}\": {spelled}"))
				})
				.collect();

			written.push(format!("\"{head}\": {{ {} }}", inner.join(", ")));
		}

		if written.is_empty() {
			return "{}".to_owned();
		}

		format!("{{ {} }}", written.join(", "))
	}
}

/// One name out of a list, by the index a record wrote down.
fn at_index(names: &[String], index: u32) -> String {
	if index == NO_INDEX {
		return String::new();
	}

	names
		.get(usize::try_from(index).unwrap_or(usize::MAX))
		.cloned()
		.unwrap_or_default()
}

/// A list of records under a name, one to a line, or nothing if there are none.
fn block(name: &str, records: &[String]) -> Option<String> {
	if records.is_empty() {
		return None;
	}

	let mut out = String::new();
	out.push_str("\t\"");
	out.push_str(name);
	out.push_str("\": [\n");

	for (index, record) in records.iter().enumerate() {
		out.push_str("\t\t");
		out.push_str(record);

		if index + 1 < records.len() {
			out.push(',');
		}

		out.push('\n');
	}

	out.push_str("\t]");

	Some(out)
}

/// One number, in the shortest spelling that reads back as itself.
fn as_number(value: f32) -> String {
	// `{}` on a float is the shortest text that parses back to the same bits,
	// which is exactly the property a format meant to be read and written by
	// two different things needs. A whole number comes out without a point,
	// which JSON is happy with.
	format!("{value}")
}

/// Three numbers.
fn as_vector(value: Vec3) -> String {
	format!("[{}, {}, {}]", as_number(value.x), as_number(value.y), as_number(value.z))
}

/// Four numbers, in the order the reader wants them.
fn as_turn(value: Quat) -> String {
	format!(
		"[{}, {}, {}, {}]",
		as_number(value.x),
		as_number(value.y),
		as_number(value.z),
		as_number(value.w)
	)
}

/// A string, with the four things JSON will not take in one spelled out.
fn as_text(value: &str) -> String { json::quoted(value) }

#[cfg(test)]
mod tests {
	use colby_core::abi::{LightKind, ShapeKind, scene::Form};

	use super::*;

	/// A source with one of everything in it.
	const SOURCE: &str = r#"{
		"stage": {
			"camera": { "position": [0, 6, 12], "target": [0, 1, 0], "fov_y": 1.2 },
			"gravity": [0, -20, 0]
		},
		"entities": [
			{ "name": "crate", "position": [1, 4, -2], "scale": [2, 2, 2],
			  "mesh": "cube", "material": "plastic", "color": [0.8, 0.2, 0.1] },
			{ "name": "hook", "position": [0, 8, 0] }
		],
		"bodies": [
			{ "name": "crate", "entity": "crate", "kind": "dynamic",
			  "shape": { "kind": "box", "extents": [1, 1, 1] },
			  "mass": 4.0, "friction": 0.7, "layer": 2, "collides": [0, 2] },
			{ "name": "ground", "kind": "static", "position": [0, -0.5, 0],
			  "shape": { "kind": "mesh", "mesh": "meshes/floor" } }
		],
		"joints": [
			{ "name": "rope", "kind": "rope", "first": "crate",
			  "anchors": [[0, 1, 0], [0, 8, 0]], "length": 3.5,
			  "stiffness": 8.0, "damping": 0.6, "max_impulse": 45.0, "max_torque": 12.5 }
		]
	}"#;

	/// A source with a character in it, drawn as two entities on one pose.
	const RIGGED: &str = r#"{
		"poses": [ { "name": "hero", "skeleton": "models/hero/rig" } ],
		"entities": [
			{ "name": "body", "mesh": "models/hero/body", "pose": "hero" },
			{ "name": "eyes", "mesh": "models/hero/eyes", "pose": "hero" },
			{ "name": "crate", "mesh": "cube" }
		]
	}"#;

	#[test]
	fn a_source_pose_is_read_and_the_entities_that_wear_it_point_at_it() {
		let scene = import(RIGGED).expect("it is a scene");

		assert_eq!(scene.posed.len(), 1, "one pose");
		assert_eq!(scene.posed[0].skeleton, "models/hero/rig");
		assert!(
			scene.posed[0].locals.is_empty(),
			"and no bones, which is what makes it rest when it is loaded"
		);
		assert_eq!(scene.things[0].pose, 0, "the body wears it");
		assert_eq!(scene.things[1].pose, 0, "so does the eyes, and it is the same one");
		assert_eq!(scene.things[2].pose, NO_INDEX, "the crate wears nothing");
	}

	#[test]
	fn a_pose_survives_being_written_back_out_and_read_again() {
		let once = import(RIGGED).expect("it is a scene");
		let text = export(&once).expect("it writes");
		let twice = import(&text).expect("and reads back");

		assert_eq!(twice, once, "exactly, which is the only test a writer and a reader share");
		assert!(text.contains("\"poses\""), "and the block is really in the text: {text}");
	}

	#[test]
	fn a_pose_nothing_wears_and_nothing_names_is_still_written() {
		let orphan = r#"{ "poses": [ { "skeleton": "models/hero/rig" } ] }"#;
		let scene = import(orphan).expect("it is a scene");
		let text = export(&scene).expect("it writes");

		assert!(text.contains("models/hero/rig"), "got {text}");
		assert_eq!(import(&text).expect("it reads back"), scene);
	}

	#[test]
	fn a_captured_pose_is_given_a_name_only_because_something_wears_it() {
		// what a capture produces: no name at all, because nothing in a world
		// names a pose. The writer has to invent one, or the entity wearing it
		// has nothing to point at.
		let scene = SceneData {
			posed: vec![Posed {
				skeleton: "models/hero/rig".to_owned(),
				generation: 1,
				..Posed::default()
			}],
			things: vec![Thing {
				generation: 1,
				pose: 0,
				..Thing::default()
			}],
			pose_generations: vec![1],
			thing_generations: vec![1],
			..SceneData::default()
		};
		let text = export(&scene).expect("it writes");

		assert!(text.contains("\"pose 0\""), "a name was invented: {text}");
		assert_eq!(import(&text).expect("it reads back").things[0].pose, 0);
	}

	#[test]
	fn an_entity_worn_by_a_pose_nobody_declared_is_an_error_naming_it() {
		let refused = import(r#"{ "entities": [ { "name": "a", "pose": "ghost" } ] }"#)
			.expect_err("there is no such pose")
			.to_string();

		assert!(refused.contains("ghost"), "it says which: {refused}");
		assert!(refused.contains("no pose is that"), "got {refused}");
	}

	#[test]
	fn two_poses_called_the_same_are_an_error() {
		let twice = r#"{ "poses": [ { "name": "one" }, { "name": "one" } ] }"#;
		let refused = import(twice)
			.expect_err("two poses cannot be called the same")
			.to_string();

		assert!(refused.contains("one"), "got {refused}");
	}

	#[test]
	fn a_field_a_pose_does_not_have_is_an_error_naming_it() {
		let refused = import(r#"{ "poses": [ { "bones": [] } ] }"#)
			.expect_err("a source does not write bones")
			.to_string();

		assert!(refused.contains("bones"), "it names the field: {refused}");
	}

	/// A source where one entity hangs off another that is written below it.
	const HUNG: &str = r#"{
		"entities": [
			{ "name": "wheel", "parent": "car", "position": [1, 0, 0] },
			{ "name": "car", "position": [2, 0, 0], "scale": [2, 2, 2] }
		]
	}"#;

	#[test]
	fn an_entity_hangs_off_the_one_it_names_whichever_is_written_first() {
		let scene = import(HUNG).expect("it is a scene");

		assert_eq!(scene.things[0].parent, 1, "the wheel hangs off the car, by its place");
		assert_eq!(scene.things[1].parent, NO_INDEX, "the car hangs off nothing");
		assert_eq!(scene.things[0].transform.position, Vec3::X, "and its place is its own");
	}

	#[test]
	fn a_parent_survives_being_written_back_out_and_read_again() {
		let scene = import(HUNG).expect("it is a scene");
		let text = export(&scene).expect("it writes");

		assert!(text.contains("\"parent\": \"car\""), "written by name: {text}");
		assert_eq!(import(&text).expect("it reads back"), scene);
	}

	#[test]
	fn a_parent_is_named_even_when_nothing_else_needed_a_name() {
		let scene = SceneData {
			things: vec![Thing { generation: 1, ..Thing::default() }, Thing {
				generation: 1,
				parent: 0,
				..Thing::default()
			}],
			thing_generations: vec![1, 1],
			..SceneData::default()
		};
		let text = export(&scene).expect("it writes");

		assert!(text.contains("\"parent\": \"entity 0\""), "the parent was given a name: {text}");
		assert_eq!(import(&text).expect("it reads back").things[1].parent, 0);
	}

	#[test]
	fn a_parent_nothing_answers_to_and_a_loop_are_errors_naming_the_entity() {
		let nobody = r#"{ "entities": [ { "name": "wheel", "parent": "cart" } ] }"#;
		let error = import(nobody).expect_err("no cart").to_string();

		assert!(error.contains("cart"), "it says which: {error}");

		let itself = r#"{ "entities": [ { "name": "wheel", "parent": "wheel" } ] }"#;
		let error = import(itself).expect_err("itself").to_string();

		assert!(error.contains("wheel") && error.contains("itself"), "got {error}");

		let ring = r#"{ "entities": [
			{ "name": "a", "parent": "b" }, { "name": "b", "parent": "c" }, { "name": "c", "parent": "a" }
		] }"#;
		let error = import(ring).expect_err("a loop").to_string();

		assert!(error.contains("a hangs off a chain that never ends"), "got {error}");

		// and a loop that does not run through the record the climb started
		// from is a loop all the same
		let beside = r#"{ "entities": [
			{ "name": "a", "parent": "b" }, { "name": "b", "parent": "c" }, { "name": "c", "parent": "b" }
		] }"#;
		let error = import(beside)
			.expect_err("a loop beside it")
			.to_string();

		assert!(error.contains("hangs off a chain that never ends"), "got {error}");
	}

	#[test]
	fn a_source_reads_into_a_description_of_itself() {
		let scene = import(SOURCE).expect("it is a scene");

		assert_eq!(scene.things.len(), 2, "two entities");
		assert_eq!(scene.solids.len(), 2, "two bodies");
		assert_eq!(scene.links.len(), 1, "and a rope");

		assert_eq!(scene.stage.camera.position, Vec3::new(0.0, 6.0, 12.0), "the camera is read");
		assert_eq!(scene.stage.gravity, Vec3::new(0.0, -20.0, 0.0), "and so is the gravity");
		assert_eq!(
			scene.stage.ambient,
			Stage::DEFAULT.ambient,
			"and what the file left out is what a world starts with"
		);
	}

	#[test]
	fn a_name_becomes_an_index_into_the_file() {
		let scene = import(SOURCE).expect("it is a scene");

		assert_eq!(scene.solids[0].thing, 0, "the crate body drives the crate entity");
		assert_eq!(scene.things[0].name, "crate", "which is the first one written");
		assert_eq!(scene.solids[1].thing, NO_INDEX, "and the ground drives nothing");
		assert_eq!(scene.links[0].first, 0, "the rope holds the crate body");
		assert_eq!(scene.links[0].second, NO_INDEX, "and a point in the world");
	}

	#[test]
	fn a_body_with_no_place_of_its_own_stands_where_its_entity_does() {
		let scene = import(SOURCE).expect("it is a scene");

		assert_eq!(
			scene.solids[0].transform.position,
			Vec3::new(1.0, 4.0, -2.0),
			"the crate body is where the crate is, without saying so twice"
		);
		assert_eq!(
			scene.solids[1].transform.position,
			Vec3::new(0.0, -0.5, 0.0),
			"and one that does say where it is, is there"
		);
	}

	#[test]
	fn layers_are_written_as_numbers_and_read_as_bits() {
		let scene = import(SOURCE).expect("it is a scene");

		assert_eq!(
			scene.solids[0].layers,
			Layers::new(Layers::bit(2), Layers::bit(0) | Layers::bit(2)),
			"a layer index and a list of them become two masks"
		);
		assert_eq!(
			scene.solids[1].layers,
			Layers::single(0),
			"and a body that says nothing is on layer zero against everything"
		);
	}

	#[test]
	fn every_record_lands_in_its_own_slot_with_a_generation() {
		let scene = import(SOURCE).expect("it is a scene");

		assert_eq!(scene.things[1].slot, 1, "the second entity is in slot one");
		assert_eq!(scene.things[1].generation, 1, "on the first generation of it");
		assert_eq!(scene.thing_generations, vec![1, 1], "and the table is that big");
		assert_eq!(scene.solid_generations, vec![1, 1], "in every table");
		assert!(scene.arena.is_none(), "with no arena, there being no game yet");
	}

	#[test]
	fn what_a_source_leaves_out_is_what_it_meant() {
		let scene = import(r#"{ "entities": [ {} ], "bodies": [ {} ], "joints": [ {} ] }"#)
			.expect("a scene");

		assert_eq!(scene.things[0].transform, Transform::IDENTITY, "an entity is at the origin");
		assert_eq!(scene.things[0].color, Vec3::ONE, "and untinted");
		assert_eq!(scene.solids[0].kind, BodyKind::Static, "a body is static");
		assert_eq!(scene.solids[0].shape.kind, ShapeKind::Box, "and a unit box");
		assert_eq!(scene.solids[0].shape.extents, Vec3::splat(0.5), "half a unit each way");
		assert!(
			(scene.solids[0].shape.radius - 0.5).abs() < f32::EPSILON,
			"and half a unit across, for the sphere it would be if it were one"
		);
		assert_eq!(scene.links[0].kind, JointKind::Rope, "a joint is a rope");
		assert!((scene.links[0].length - 1.0).abs() < f32::EPSILON, "one unit long");
		assert_eq!(scene.links[0].axis, Vec3::Y, "turning about up if it were a hinge");
		assert!((scene.links[0].damping - 1.0).abs() < f32::EPSILON, "critically damped");
	}

	#[test]
	fn a_shorter_list_keeps_the_default_in_the_axes_it_did_not_mention() {
		let scene = import(r#"{ "entities": [ { "scale": [3] } ] }"#).expect("a scene");

		assert_eq!(
			scene.things[0].transform.scale,
			Vec3::new(3.0, 1.0, 1.0),
			"two written and one left alone, rather than two written and one zeroed"
		);
	}

	#[test]
	fn a_field_this_build_does_not_know_is_an_error_naming_it() {
		let refused = |text: &str| {
			import(text)
				.expect_err("it should not read")
				.to_string()
		};

		assert!(
			refused(r#"{ "entities": [ { "whereabouts": [1, 2, 3] } ] }"#)
				.contains("whereabouts"),
			"a field nobody knows is named rather than ignored"
		);
		assert!(refused(r#"{ "things": [] }"#).contains("things"), "at the top level too");
		assert!(
			refused(r#"{ "stage": { "lens": {} } }"#).contains("lens"),
			"and inside the stage"
		);
		assert!(
			refused(r#"{ "bodies": [ { "shape": { "size": 2 } } ] }"#).contains("size"),
			"and inside a shape"
		);
	}

	#[test]
	fn a_name_nothing_answers_to_is_an_error_naming_it() {
		let refused = |text: &str| {
			import(text)
				.expect_err("it should not read")
				.to_string()
		};

		assert!(
			refused(r#"{ "bodies": [ { "entity": "ghost" } ] }"#).contains("ghost"),
			"a body driving an entity nobody wrote"
		);
		assert!(
			refused(r#"{ "joints": [ { "first": "ghost" } ] }"#).contains("ghost"),
			"and a joint holding a body nobody wrote"
		);
	}

	#[test]
	fn one_name_used_twice_is_an_error() {
		let twice = r#"{ "entities": [ { "name": "one" }, { "name": "one" } ] }"#;
		let refused = import(twice)
			.expect_err("two things cannot be called the same")
			.to_string();

		assert!(refused.contains("one"), "the name is in the message: {refused}");

		let unnamed = r#"{ "entities": [ {}, {} ] }"#;

		assert!(
			import(unnamed).is_ok(),
			"but any number of records may leave the name out, which is not a name"
		);
	}

	#[test]
	fn a_word_that_is_not_a_kind_is_an_error() {
		let refused = |text: &str| {
			import(text)
				.expect_err("it should not read")
				.to_string()
		};

		assert!(refused(r#"{ "bodies": [ { "kind": "floaty" } ] }"#).contains("floaty"));
		assert!(refused(r#"{ "bodies": [ { "shape": { "kind": "blob" } } ] }"#).contains("blob"));
		assert!(refused(r#"{ "joints": [ { "kind": "spring" } ] }"#).contains("spring"));
	}

	#[test]
	fn a_turn_that_is_not_a_unit_quaternion_is_an_error() {
		let refused = import(r#"{ "entities": [ { "rotation": [1, 1, 1, 1] } ] }"#)
			.expect_err("that is not a rotation")
			.to_string();

		assert!(refused.contains("unit"), "and it says what one is: {refused}");

		let half = std::f32::consts::FRAC_1_SQRT_2;
		let text = format!(r#"{{ "entities": [ {{ "rotation": [0, {half}, 0, {half}] }} ] }}"#);
		let scene = import(&text).expect("a quarter turn is a rotation");

		assert!(scene.things[0].transform.rotation.is_normalized(), "and it survives");
	}

	#[test]
	fn a_joint_says_nothing_about_collision_unless_it_wants_the_unusual_answer() {
		let text = r#"{
			"entities": [ {} ],
			"bodies": [
				{ "name": "a", "kind": "dynamic" },
				{ "name": "b", "kind": "dynamic" }
			],
			"joints": [
				{ "kind": "weld", "first": "a", "second": "b" },
				{ "kind": "weld", "first": "a", "second": "b", "collide": true }
			]
		}"#;
		let scene = import(text).expect("a scene");

		assert!(!scene.links[0].collide, "a joint that says nothing holds them apart");
		assert!(scene.links[1].collide, "and one that asks for the other answer gets it");

		let written = export(&scene).expect("it can be written");

		assert_eq!(
			written.matches("\"collide\"").count(),
			1,
			"only the unusual one is written down, and it appears once: {written}"
		);
		assert_eq!(
			import(&written)
				.expect("what was written reads back")
				.links,
			scene.links,
			"and both come back the way they went in"
		);
	}

	#[test]
	fn every_kind_of_body_shape_and_joint_can_be_written() {
		let text = r#"{
			"entities": [ {} ],
			"bodies": [
				{ "name": "a", "kind": "static", "shape": { "kind": "box" } },
				{ "name": "b", "kind": "kinematic", "shape": { "kind": "sphere", "radius": 2 } },
				{ "name": "c", "kind": "dynamic", "shape": { "kind": "mesh", "mesh": "m" },
				  "sensor": true }
			],
			"joints": [
				{ "kind": "rope", "first": "a" },
				{ "kind": "weld", "first": "a", "second": "b" },
				{ "kind": "axis", "first": "b", "second": "c", "axis": [1, 0, 0] },
				{ "kind": "ball", "first": "a", "second": "c" }
			]
		}"#;
		let scene = import(text).expect("a scene");

		let kinds: Vec<BodyKind> = scene.solids.iter().map(|it| it.kind).collect();
		let shapes: Vec<ShapeKind> = scene
			.solids
			.iter()
			.map(|it| it.shape.kind)
			.collect();
		let joints: Vec<JointKind> = scene.links.iter().map(|it| it.kind).collect();

		assert_eq!(kinds, vec![BodyKind::Static, BodyKind::Kinematic, BodyKind::Dynamic]);
		assert_eq!(shapes, vec![ShapeKind::Box, ShapeKind::Sphere, ShapeKind::Mesh]);
		assert_eq!(joints, vec![
			JointKind::Rope,
			JointKind::Weld,
			JointKind::Axis,
			JointKind::Ball
		]);
		assert_eq!(
			import(&export(&scene).expect("it can be written"))
				.expect("what was written reads back")
				.links,
			scene.links,
			"and every one of those words survives being written out again"
		);
		assert!(scene.solids[2].sensor, "and a sensor says so");
		assert!((scene.solids[1].shape.radius - 2.0).abs() < f32::EPSILON, "with its radius");
	}

	#[test]
	fn text_that_is_not_json_at_all_is_an_error() {
		assert!(import("this is not a scene").is_err(), "and it does not panic");
		assert!(import("").is_err(), "nor does an empty file");
	}

	// ------------------------------------------------------------------
	// writing one out
	// ------------------------------------------------------------------

	/// Reads a source, writes it, and reads what was written.
	fn round(text: &str) -> (SceneData, SceneData) {
		let first = import(text).expect("it is a scene");
		let written = export(&first).expect("every number in it can be written");
		let again = import(&written).unwrap_or_else(|failure| {
			panic!("what was written did not read back: {failure}\n{written}")
		});

		(first, again)
	}

	#[test]
	fn a_scene_written_out_reads_back_as_itself() {
		let (first, again) = round(SOURCE);

		assert_eq!(again, first, "everything a source can say survives the round trip");
	}

	#[test]
	fn what_a_record_would_have_said_anyway_is_left_out() {
		let scene = import(SOURCE).expect("it is a scene");
		let written = export(&scene).expect("it can be written");

		assert!(
			!written.contains("\"kind\": \"static\""),
			"a static body is what a body is: {written}"
		);
		assert!(!written.contains("\"sensor\""), "and one is not a sensor unless it says so");
		assert!(
			!written.contains("\"ambient\""),
			"and the stage says nothing about what it did not change"
		);
		assert!(written.contains("\"gravity\""), "but does about what it did");
		assert!(!written.contains("\"restitution\""), "nor a body about a bounce it never set");

		// the two that read back the same either way, and are the whole point
		// all the same: a file where every record spells out an empty material
		// and lists all thirty-two layers is a file nobody reads a diff of.
		assert!(
			!written.contains("\"\""),
			"nothing is written as being called nothing: {written}"
		);
		assert!(
			!written.contains("\"collides\": [0, 1, 2"),
			"and a body that meets everything does not list everything: {written}"
		);
	}

	#[test]
	fn a_body_standing_where_its_entity_does_still_does_not_say_so_twice() {
		let (first, again) = round(SOURCE);

		assert_eq!(
			again.solids[0].transform, first.solids[0].transform,
			"it comes back in the same place"
		);

		let written = export(&first).expect("it can be written");
		let lines: Vec<&str> = written.lines().collect();
		let body = lines
			.iter()
			.find(|line| line.contains("\"entity\": \"crate\""))
			.expect("the crate's body is in there");

		assert!(
			!body.contains("\"position\""),
			"and it stands where the crate does without a place of its own: {body}"
		);
	}

	#[test]
	fn a_body_somewhere_other_than_its_entity_says_where() {
		let mut scene = import(SOURCE).expect("it is a scene");
		scene.solids[0].transform.position = Vec3::new(9.0, 9.0, 9.0);
		scene.solids[0].transform.scale = Vec3::splat(3.0);

		let written = export(&scene).expect("it can be written");
		let again = import(&written).expect("it reads back");

		assert_eq!(
			again.solids[0].transform.position,
			Vec3::new(9.0, 9.0, 9.0),
			"the place it is really in"
		);
		assert_eq!(
			again.solids[0].transform.scale,
			Vec3::splat(3.0),
			"and the size, which is only read at all because the place was written"
		);
	}

	#[test]
	fn a_turn_is_written_beside_the_place_that_makes_it_readable() {
		// a rotation and no offset at all: `at` has to be written even though
		// it is nothing, because the reader looks at `turn` only when it is
		// there.
		let mut scene = import(SOURCE).expect("it is a scene");
		scene.things[1].transform.position = Vec3::ZERO;
		scene.things[1].transform.rotation = Quat::from_rotation_y(0.5);

		let (_, again) = round(&export(&scene).expect("it can be written"));

		assert!(
			again.things[1]
				.transform
				.rotation
				.abs_diff_eq(Quat::from_rotation_y(0.5), 1.0e-6),
			"the turn came back: {}",
			again.things[1].transform.rotation
		);
	}

	#[test]
	fn two_records_sharing_one_name_come_out_told_apart() {
		// what a duplicate in the world looks like: a copy of a prop keeps the
		// name it was copied from, and a source may not say one name twice.
		let mut scene = import(SOURCE).expect("it is a scene");
		let copy = scene.things[0].clone();
		scene.things.push(copy);
		let mut body = scene.solids[0].clone();
		body.thing = 2;
		scene.solids.push(body);

		let written = export(&scene).expect("it can be written");
		let again = import(&written).expect("what was written reads back");

		assert_eq!(again.things[0].name, "crate", "the first keeps the name");
		assert_eq!(again.things[2].name, "crate 2", "and the copy is numbered");
		assert_eq!(again.solids[0].thing, 0, "and the two bodies drive the two of them");
		assert_eq!(again.solids[2].thing, 2);
	}

	#[test]
	fn something_that_has_to_be_named_is_given_a_name() {
		let mut scene = import(SOURCE).expect("it is a scene");
		scene.things[0].name.clear();
		scene.solids[0].name.clear();

		let written = export(&scene).expect("it can be written");
		let again = import(&written).expect("what was written reads back");

		assert!(!again.things[0].name.is_empty(), "the entity a body drives has a name");
		assert!(!again.solids[0].name.is_empty(), "and so does the body a joint holds");
		assert_eq!(again.solids[0].thing, 0, "and the body still drives it");
		assert_eq!(again.links[0].first, 0, "and the joint still holds the body");
	}

	#[test]
	fn something_nothing_refers_to_keeps_its_silence() {
		let mut scene = import(SOURCE).expect("it is a scene");
		scene.things[1].name.clear();
		scene.links[0].name.clear();

		let written = export(&scene).expect("it can be written");
		let again = import(&written).expect("what was written reads back");

		assert!(
			again.things[1].name.is_empty(),
			"nothing drives the hook, so nothing had to call it anything"
		);
		assert!(again.links[0].name.is_empty(), "and nothing at all refers to a joint");
	}

	#[test]
	fn a_shape_is_written_only_when_it_is_not_the_one_a_body_has_anyway() {
		let mut scene = import(SOURCE).expect("it is a scene");
		scene.solids[0].shape = Form {
			kind: ShapeKind::Box,
			radius: 0.5,
			extents: Vec3::splat(0.5),
			mesh: String::new(),
		};

		let written = export(&scene).expect("it can be written");
		let again = import(&written).expect("what was written reads back");

		assert_eq!(
			again.solids[0].shape, scene.solids[0].shape,
			"a body with no shape field reads back as the shape that has none"
		);

		let written = export(&scene).expect("it can be written");
		assert!(
			!written.contains("\"shape\": {}"),
			"and there is no such thing as an empty shape to have to write: {written}"
		);

		// one field away from the default is a shape again, and only that
		// field is written.
		scene.solids[0].shape.radius = 2.0;
		scene.solids[0].shape.kind = ShapeKind::Sphere;
		let written = export(&scene).expect("it can be written");
		let (_, back) = round(&written);

		assert!(
			(back.solids[0].shape.radius - 2.0).abs() < 1.0e-6,
			"the radius came back: {}",
			back.solids[0].shape.radius
		);
		let crate_line = written
			.lines()
			.find(|line| line.contains("\"entity\": \"crate\""))
			.expect("the crate's body is in there");

		assert_eq!(
			crate_line.matches("\"shape\"").count(),
			1,
			"two fields inside a shape are one shape, written once: {crate_line}"
		);
	}

	#[test]
	fn layers_go_back_out_as_the_numbers_they_came_in_as() {
		let (first, again) = round(SOURCE);

		assert_eq!(again.solids[0].layers, first.solids[0].layers, "a layer and a mask");
		assert_eq!(
			again.solids[1].layers,
			Layers::single(0),
			"and a body that never mentioned them still has not"
		);

		let written = export(&first).expect("it can be written");
		assert!(written.contains("\"layer\": 2"), "written as an index: {written}");
		assert!(written.contains("\"collides\": [0, 2]"), "and a list of them");
		assert_eq!(
			written.matches("\"layer\"").count(),
			1,
			"and only for the body that is not on layer zero: {written}"
		);
		assert_eq!(
			written.matches("\"collides\"").count(),
			1,
			"and only for the body that does not meet everything: {written}"
		);
	}

	#[test]
	fn a_body_gravity_does_not_reach_says_so_and_says_it_back() {
		// its own source rather than an extra record in the shared one, which
		// three tests count the bodies of.
		let source = r#"{
			"bodies": [
				{ "name": "balloon", "kind": "dynamic", "position": [2, 3, 0], "weightless": true },
				{ "name": "brick", "kind": "dynamic", "position": [0, 3, 0] }
			]
		}"#;

		let (first, again) = round(source);

		assert!(first.solids[0].weightless, "the one that said so");
		assert!(!first.solids[1].weightless, "and the one that did not");
		assert_eq!(again.solids[0].weightless, first.solids[0].weightless, "and it survives");
		assert_eq!(again.solids[1].weightless, first.solids[1].weightless, "both ways");

		let written = export(&first).expect("it can be written");
		assert_eq!(
			written.matches("\"weightless\": true").count(),
			1,
			"written once, on the body it is true of, and left out of the other: {written}"
		);
	}

	#[test]
	fn a_name_with_something_in_it_json_reserves_survives() {
		let mut scene = import(SOURCE).expect("it is a scene");
		scene.things[0].name = "a \"quoted\" \\ name\twith a tab".to_owned();

		let (_, again) = round(&export(&scene).expect("it can be written"));

		assert_eq!(
			again.things[0].name, scene.things[0].name,
			"every one of them came back as itself"
		);
	}

	#[test]
	fn a_number_json_has_no_spelling_for_is_refused() {
		let mut scene = import(SOURCE).expect("it is a scene");
		scene.things[0].transform.position.y = f32::NAN;

		let failure = export(&scene).expect_err("a nan is not a number JSON has");
		assert!(
			format!("{failure}").contains("entity"),
			"and the message says which kind of record: {failure}"
		);

		let mut blown = import(SOURCE).expect("it is a scene");
		blown.solids[0].velocity.x = f32::INFINITY;
		assert!(export(&blown).is_err(), "and an infinity is refused too");

		// a joint's own numbers, which is where an infinity is most tempting:
		// "no ceiling" is arithmetically an infinity and is written as a zero
		// for exactly this reason.
		let mut uncapped = import(SOURCE).expect("it is a scene");
		uncapped.links[0].max_impulse = f32::INFINITY;
		let refused = export(&uncapped).expect_err("a joint cannot carry one either");
		assert!(
			format!("{refused}").contains("joint"),
			"and the message says which kind of record: {refused}"
		);
	}

	#[test]
	fn the_same_scene_is_written_the_same_way_twice() {
		let scene = import(SOURCE).expect("it is a scene");

		assert_eq!(
			export(&scene).expect("it can be written"),
			export(&scene).expect("it can be written"),
			"a format meant to be diffed has to be the same bytes for the same world"
		);
	}

	#[test]
	fn a_world_with_nothing_in_it_is_still_a_scene() {
		let empty = SceneData::default();
		let written = export(&empty).expect("it can be written");
		let again = import(&written).expect("and read back");

		assert!(again.is_empty(), "nothing in, nothing out: {written}");
		assert_eq!(again.stage, Stage::DEFAULT, "and the settings a world starts with");
	}

	#[test]
	fn what_a_source_cannot_say_is_not_pretended() {
		// a capture carries things the text has no words for. What comes back
		// is what a source would have produced, which is the honest answer
		// rather than a silent half.
		let mut scene = import(SOURCE).expect("it is a scene");
		scene.stage.time = 12.5;
		scene.stage.steps = 750;
		scene.things[0].slot = 40;
		scene.things[0].generation = 9;

		let (_, again) = round(&export(&scene).expect("it can be written"));

		assert!((again.stage.time).abs() < 1.0e-6, "a source cannot say what time it is");
		assert_eq!(again.stage.steps, 0, "nor how many steps have run");
		assert_eq!(again.things[0].slot, 0, "and a record's slot is its place in the file");
		assert_eq!(again.things[0].generation, 1, "on its first occupant");
	}

	#[test]
	fn a_sleeping_body_and_a_joints_rest_are_written_now_that_the_table_spells_them() {
		// two things the text could not say before the tables: a body the
		// solver had stopped, and the angle a weld was made at. A settled pile
		// written out used to come back awake and a welded pair square.
		let mut scene = import(SOURCE).expect("it is a scene");
		scene.solids[0].sleeping = true;
		scene.links[0].rest = Quat::from_rotation_y(0.5);

		let written = export(&scene).expect("it can be written");
		let (_, again) = round(&written);

		assert!(written.contains("\"sleeping\": true"), "written, once: {written}");
		assert!(written.contains("\"rest\""), "and the rest rotation: {written}");
		assert!(again.solids[0].sleeping, "the body comes back asleep");
		assert!(
			again.links[0]
				.rest
				.abs_diff_eq(Quat::from_rotation_y(0.5), 1.0e-6),
			"and the weld at its angle"
		);
	}

	#[test]
	fn a_file_written_before_the_tables_is_refused_naming_the_key_and_the_new_word() {
		// the text has no version, so an old key is an unknown key; what it
		// gets that a misspelling does not is the word to replace it with.
		let refused = |text: &str| {
			import(text)
				.expect_err("an old key does not read")
				.to_string()
		};

		let old = refused(r#"{ "entities": [ { "at": [1, 2, 3] } ] }"#);
		assert!(old.contains("at") && old.contains("position"), "got {old}");

		let old = refused(r#"{ "bodies": [ { "turn": [0, 0, 0, 1] } ] }"#);
		assert!(old.contains("turn") && old.contains("rotation"), "got {old}");

		let old = refused(r#"{ "joints": [ { "max impulse": 4 } ] }"#);
		assert!(old.contains("max impulse") && old.contains("max_impulse"), "got {old}");

		let old = refused(r#"{ "stage": { "camera": { "fov": 1.2 } } }"#);
		assert!(old.contains("fov") && old.contains("fov_y"), "got {old}");

		let plain = refused(r#"{ "entities": [ { "whereabouts": [1, 2, 3] } ] }"#);
		assert!(!plain.contains(" now"), "and a plain unknown gets no such hint: {plain}");
	}

	#[test]
	fn a_value_of_the_wrong_kind_is_refused_naming_the_field() {
		// the rule the unknown-field rule already stands on: a mass that is a
		// word doing nothing while looking like it works is the same bug.
		let refused = |text: &str| {
			import(text)
				.expect_err("a value of the wrong kind does not read")
				.to_string()
		};

		let heavy = refused(r#"{ "bodies": [ { "mass": "heavy" } ] }"#);
		assert!(heavy.contains("mass") && heavy.contains("number"), "got {heavy}");

		let yes = refused(r#"{ "bodies": [ { "sensor": "yes" } ] }"#);
		assert!(yes.contains("sensor"), "got {yes}");

		let flat = refused(r#"{ "entities": [ { "position": 3 } ] }"#);
		assert!(flat.contains("position") && flat.contains("three numbers"), "got {flat}");

		let long = refused(r#"{ "entities": [ { "scale": [1, 2, 3, 4] } ] }"#);
		assert!(long.contains("scale"), "four numbers are not three: {long}");

		let word = refused(r#"{ "entities": [ { "color": ["red", 0, 0] } ] }"#);
		assert!(word.contains("color"), "and a word is not a number: {word}");

		let bare = refused(r#"{ "bodies": [ { "shape": 3 } ] }"#);
		assert!(bare.contains("shape"), "a shape is a record of its own: {bare}");

		let floaty = refused(r#"{ "bodies": [ { "kind": 2 } ] }"#);
		assert!(floaty.contains("kind"), "and a kind is a word, not a number: {floaty}");
	}

	#[test]
	fn a_body_turned_or_sized_without_a_position_is_refused_rather_than_read_against_nothing() {
		let refused = import(r#"{ "bodies": [ { "scale": [2, 2, 2] } ] }"#)
			.expect_err("a scale with no position")
			.to_string();

		assert!(refused.contains("position"), "it says what is missing: {refused}");
		assert!(
			import(r#"{ "bodies": [ { "position": [0, 0, 0], "scale": [2, 2, 2] } ] }"#).is_ok(),
			"and with one it reads"
		);
	}

	#[test]
	fn a_whole_number_is_read_as_one_and_a_fraction_is_not() {
		let parse = |written: Value| {
			parsed(&written, Kind::Int, &field::Value::Int(0), "a test", "count")
		};

		assert_eq!(parse(Value::Number(6.0)).ok(), Some(field::Value::Int(6)));
		assert_eq!(
			parse(Value::Number(-3.0)).ok(),
			Some(field::Value::Int(-3)),
			"of either sign"
		);
		assert!(parse(Value::Number(2.5)).is_err(), "a fraction is not a whole number");
		assert!(parse(Value::String("six".to_owned())).is_err(), "nor is a word");
	}

	/// A value of every plain kind that is not what a fresh record holds.
	fn sample(kind: Kind) -> Option<field::Value> {
		Some(match kind {
			| Kind::Bool => field::Value::Bool(true),
			| Kind::Int => field::Value::Int(6),
			| Kind::Float => field::Value::Float(2.5),
			| Kind::Text => field::Value::Text("hello".to_owned()),
			| Kind::Vec3 => field::Value::Vec3(Vec3::new(1.0, 2.0, 3.0)),
			| Kind::Quat => field::Value::Quat(Quat::from_rotation_y(0.5)),
			| Kind::Color => field::Value::Color(Vec3::new(0.2, 0.4, 0.6)),
			| Kind::Word(words) => field::Value::Word(u32::try_from(words.len()).ok()? - 1),
			| Kind::Entity
			| Kind::Body
			| Kind::Joint
			| Kind::Pose
			| Kind::Mesh
			| Kind::Material => return None,
		})
	}

	/// Sets every plain field the text spells to something other than its
	/// default.
	fn fill<T>(record: &mut T, table: &[Field<T>], skipped: &[&str]) {
		for field in table {
			if skipped.contains(&field.name) {
				continue;
			}

			if let Some(value) = sample(field.kind) {
				assert!(field.set(record, value), "{} takes its own kind", field.name);
			}
		}
	}

	#[test]
	fn a_light_is_read_out_of_its_own_object_and_written_back_into_one() {
		let scene = import(
			r#"{ "entities": [
				{ "name": "lamp", "color": [1, 0, 0], "light": {
					"kind": "spot", "color": [0, 0.5, 1], "intensity": 3,
					"range": 12, "inner": 0.2, "outer": 0.5 } },
				{ "name": "crate", "mesh": "cube" }
			] }"#,
		)
		.expect("it is a scene");

		let lamp = scene.things[0].light;

		assert_eq!(lamp.kind, LightKind::Spot, "the word is the kind");
		assert_eq!(lamp.color, Vec3::new(0.0, 0.5, 1.0), "and the light has its own color");
		assert_eq!(
			scene.things[0].color,
			Vec3::new(1.0, 0.0, 0.0),
			"which is not the entity's tint, the one thing a flat namespace could not keep apart"
		);
		assert!((lamp.intensity - 3.0).abs() < 1.0e-6 && (lamp.range - 12.0).abs() < 1.0e-6);
		assert!((lamp.inner - 0.2).abs() < 1.0e-6 && (lamp.outer - 0.5).abs() < 1.0e-6);
		assert_eq!(
			scene.things[1].light,
			Light::NONE,
			"and an entity without one shines nothing"
		);

		let text = export(&scene).expect("it writes back");

		assert!(text.contains("\"light\": {"), "the lamp is written under its own key");
		assert_eq!(
			text.matches("\"light\": {").count(),
			1,
			"and the crate, whose light is the default, gets no key at all"
		);
		assert_eq!(import(&text).expect("it reads back"), scene, "and the text is the scene");
	}

	#[test]
	fn a_light_field_nobody_declared_is_refused_by_name() {
		let refused = import(r#"{ "entities": [ { "name": "a", "light": { "watts": 60 } } ] }"#)
			.expect_err("a light has no watts");

		assert!(
			format!("{refused}").contains("watts"),
			"and the message says which word it was: {refused}"
		);
	}

	#[test]
	fn every_plain_field_of_every_table_survives_the_text() {
		// the test the tables make possible: nothing here names a field, so a
		// field added to a table tomorrow is read and written by this today,
		// or this fails. Every plain field of every record is set to something
		// other than its default, written, and read back.
		let mut body = BLANK_BODY;
		fill(&mut body, Body::FIELDS, LAYER_FIELDS);
		fill(&mut body.transform, Transform::FIELDS, &[]);
		body.layers = Layers::new(Layers::bit(3), Layers::bit(3) | Layers::bit(5));

		let mut joint = BLANK_JOINT;
		fill(&mut joint, Joint::FIELDS, ANCHOR_FIELDS);
		joint.first_anchor = Vec3::X;
		joint.second_anchor = Vec3::Y;

		let mut transform = Transform::IDENTITY;
		fill(&mut transform, Transform::FIELDS, &[]);
		let mut look = Renderable::NOTHING;
		fill(&mut look, Renderable::FIELDS, &[]);
		let mut lamp = Light::NONE;
		fill(&mut lamp, Light::FIELDS, &[]);

		let mut stage = Stage::DEFAULT;
		fill(&mut stage, Stage::FIELDS, &[]);
		fill(&mut stage.camera, Camera::FIELDS, &[]);

		let scene = SceneData {
			stage,
			things: vec![Thing {
				name: "crate".to_owned(),
				generation: 1,
				transform,
				mesh: "cube".to_owned(),
				material: "plastic".to_owned(),
				color: look.color,
				light: lamp,
				..Thing::default()
			}],
			solids: vec![Solid {
				name: "crate".to_owned(),
				generation: 1,
				..Solid::of(&body, "meshes/rock".to_owned(), 0)
			}],
			links: vec![Link {
				name: "tie".to_owned(),
				generation: 1,
				..Link::of(&joint, 0, NO_INDEX)
			}],
			thing_generations: vec![1],
			solid_generations: vec![1],
			link_generations: vec![1],
			..SceneData::default()
		};

		let written = export(&scene).expect("it can be written");
		let again = import(&written).unwrap_or_else(|failure| {
			panic!("what was written did not read back: {failure}\n{written}")
		});

		assert_eq!(again.stage, scene.stage, "the stage and its camera: {written}");
		assert_eq!(again.things, scene.things, "the entity: {written}");
		assert_eq!(again.solids, scene.solids, "the body: {written}");
		assert_eq!(again.links, scene.links, "and the joint: {written}");

		// and every one of those keys is the field's own name
		for field in Body::FIELDS {
			if field.kind.is_reference() || LAYER_FIELDS.contains(&field.name) {
				continue;
			}

			let key = field
				.name
				.rsplit('.')
				.next()
				.unwrap_or(field.name);
			assert!(
				written.contains(&format!("\"{key}\"")),
				"{} is in the text: {written}",
				field.name
			);
		}
	}
}
