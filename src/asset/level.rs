//! Reading a scene somebody wrote: `.scene`, which is JSON.
//!
//! The authored half of the scene format. A `.cscene` is what the engine reads
//! and what it writes for itself; this is where one comes from when nobody has
//! run the engine yet - a level, a prefab, a room laid out in a text editor -
//! and it compiles into exactly the same file a save is.
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
//! scroll away.
//!
//! ```text
//! {
//!   "stage":    { "camera": { "position": [0, 2, 5], "target": [0, 0, 0] } },
//!   "entities": [ { "name": "crate", "at": [0, 4, 0], "mesh": "cube" } ],
//!   "bodies":   [ { "entity": "crate", "kind": "dynamic",
//!                   "shape": { "kind": "box", "extents": [0.5, 0.5, 0.5] } } ],
//!   "joints":   [ { "kind": "rope", "first": "crate", "length": 3.0,
//!                   "anchors": [[0, 0, 0], [0, 7, 0]] } ]
//! }
//! ```

use colby_core::{
	Result,
	abi::{
		BodyKind, Camera, JointKind, Layers, ShapeKind, Transform,
		scene::{Form, Link, NO_INDEX, SceneData, Solid, Stage, Thing},
	},
	err,
	glam::{Quat, Vec3},
};

use crate::json::{self, Value};

/// The extension a scene source is written with.
pub const EXTENSION: &str = "scene";

/// Reads a scene out of the text somebody wrote.
///
/// @param text - the whole `.scene` file
/// @return the description, ready to be written as a `.cscene`
///
/// # Errors
///
/// If the text is not JSON, holds a field this build does not know, names
/// something nothing answers to, or names one thing twice.
pub fn import(text: &str) -> Result<SceneData> {
	let root = json::parse(text)?;
	fields(&root, &["stage", "entities", "bodies", "joints"], "the scene")?;

	let things = entities(root.get("entities"))?;
	let solids = bodies(root.get("bodies"), &things)?;
	let links = joints(root.get("joints"), &solids)?;

	Ok(SceneData {
		stage: stage(root.get("stage"))?,
		thing_generations: vec![1; things.len()],
		solid_generations: vec![1; solids.len()],
		link_generations: vec![1; links.len()],
		things,
		solids,
		links,
		// nothing an author could write. A scene laid out by hand is a world
		// before any game has run in it, and the arena is what a game
		// remembers about one that has.
		arena: None,
	})
}

/// Refuses a field this build does not know.
///
/// @param value - the object to check, or nothing
/// @param known - every field that belongs
/// @param what - what to call the object in the message
fn fields(value: &Value, known: &[&str], what: &str) -> Result<()> {
	for (name, _) in value.as_object() {
		if !known.contains(&name.as_str()) {
			return Err(err!(Asset("{what} has no field called {name}")));
		}
	}

	Ok(())
}

/// The world's own settings, or the ones a world starts with.
fn stage(value: Option<&Value>) -> Result<Stage> {
	let Some(value) = value else {
		return Ok(Stage::DEFAULT);
	};

	fields(value, &["camera", "clear", "light", "ambient", "gravity"], "a stage")?;

	Ok(Stage {
		camera: camera(value.get("camera"))?,
		clear: vector(value.get("clear"), Stage::DEFAULT.clear),
		light: vector(value.get("light"), Stage::DEFAULT.light),
		ambient: vector(value.get("ambient"), Stage::DEFAULT.ambient),
		gravity: vector(value.get("gravity"), Stage::DEFAULT.gravity),
		time: 0.0,
		steps: 0,
	})
}

/// Where the camera looks from.
fn camera(value: Option<&Value>) -> Result<Camera> {
	let Some(value) = value else {
		return Ok(Camera::DEFAULT);
	};

	fields(value, &["position", "target", "up", "fov", "near", "far"], "a camera")?;

	Ok(Camera {
		position: vector(value.get("position"), Camera::DEFAULT.position),
		target: vector(value.get("target"), Camera::DEFAULT.target),
		up: vector(value.get("up"), Camera::DEFAULT.up),
		fov_y: number(value.get("fov"), Camera::DEFAULT.fov_y),
		near: number(value.get("near"), Camera::DEFAULT.near),
		far: number(value.get("far"), Camera::DEFAULT.far),
	})
}

/// Every entity the source stands.
fn entities(value: Option<&Value>) -> Result<Vec<Thing>> {
	let mut things: Vec<Thing> = Vec::new();

	for (index, entry) in listed(value).iter().enumerate() {
		fields(
			entry,
			&["name", "at", "turn", "scale", "mesh", "material", "color"],
			"an entity",
		)?;

		let name = text(entry.get("name"));
		once(things.iter().any(|it| it.name == name), &name, "entity")?;

		things.push(Thing {
			name,
			slot: count(index)?,
			generation: 1,
			transform: transform(entry)?,
			mesh: text(entry.get("mesh")),
			material: text(entry.get("material")),
			color: vector(entry.get("color"), Vec3::ONE),
		});
	}

	Ok(things)
}

/// Every body, with the entity it drives looked up by name.
fn bodies(value: Option<&Value>, things: &[Thing]) -> Result<Vec<Solid>> {
	let mut solids: Vec<Solid> = Vec::new();

	for (index, entry) in listed(value).iter().enumerate() {
		fields(
			entry,
			&[
				"name",
				"entity",
				"kind",
				"shape",
				"at",
				"turn",
				"scale",
				"velocity",
				"angular",
				"mass",
				"restitution",
				"friction",
				"sensor",
				"layer",
				"collides",
			],
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
			)?
		};

		// a body with an entity and no place of its own stands where the
		// entity does, which is what a person writing one means by leaving it
		// out.
		let placed = things
			.get(usize::try_from(thing).unwrap_or(usize::MAX))
			.map_or(Transform::IDENTITY, |it| it.transform);

		solids.push(Solid {
			name,
			slot: count(index)?,
			generation: 1,
			kind: body_kind(entry.get("kind"))?,
			shape: shape(entry.get("shape"))?,
			transform: if entry.get("at").is_some() {
				transform(entry)?
			} else {
				placed
			},
			velocity: vector(entry.get("velocity"), Vec3::ZERO),
			angular: vector(entry.get("angular"), Vec3::ZERO),
			mass: number(entry.get("mass"), 1.0),
			restitution: number(entry.get("restitution"), 0.2),
			friction: number(entry.get("friction"), 0.5),
			sensor: flag(entry.get("sensor")),
			sleeping: false,
			layers: layers(entry)?,
			thing,
		});
	}

	Ok(solids)
}

/// Every joint, with both of its bodies looked up by name.
fn joints(value: Option<&Value>, solids: &[Solid]) -> Result<Vec<Link>> {
	let mut links: Vec<Link> = Vec::new();

	for (index, entry) in listed(value).iter().enumerate() {
		fields(
			entry,
			&["name", "kind", "first", "second", "anchors", "axis", "length", "give"],
			"a joint",
		)?;

		let name = text(entry.get("name"));
		once(links.iter().any(|it| it.name == name), &name, "joint")?;

		let anchors = entry.get("anchors").cloned().unwrap_or_default();
		let anchors = anchors.as_array();

		links.push(Link {
			name,
			slot: count(index)?,
			generation: 1,
			kind: joint_kind(entry.get("kind"))?,
			first: held(entry.get("first"), solids)?,
			second: held(entry.get("second"), solids)?,
			first_anchor: vector(anchors.first(), Vec3::ZERO),
			second_anchor: vector(anchors.get(1), Vec3::ZERO),
			axis: vector(entry.get("axis"), Vec3::Y),
			length: number(entry.get("length"), 1.0),
			// a weld written by hand holds the angle the two bodies are
			// written at, which is no relative rotation at all: they are
			// exactly where the file put them.
			rest: Quat::IDENTITY,
			give: number(entry.get("give"), 0.0),
		});
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
	)
}

/// A transform out of `at`, `turn` and `scale`.
fn transform(value: &Value) -> Result<Transform> {
	let turn = value.get("turn").cloned().unwrap_or_default();
	let turn = turn.as_array();
	let rotation = if turn.is_empty() {
		Quat::IDENTITY
	} else {
		Quat::from_xyzw(
			number(turn.first(), 0.0),
			number(turn.get(1), 0.0),
			number(turn.get(2), 0.0),
			number(turn.get(3), 1.0),
		)
	};

	if !rotation.is_normalized() {
		return Err(err!(Asset("a turn has to be a unit quaternion, xyzw")));
	}

	Ok(Transform {
		position: vector(value.get("at"), Vec3::ZERO),
		rotation,
		scale: vector(value.get("scale"), Vec3::ONE),
	})
}

/// What a body is shaped like.
fn shape(value: Option<&Value>) -> Result<Form> {
	let Some(value) = value else {
		return Ok(Form {
			kind: ShapeKind::Box,
			radius: 0.0,
			extents: Vec3::splat(0.5),
			mesh: String::new(),
		});
	};

	fields(value, &["kind", "radius", "extents", "mesh"], "a shape")?;

	let kind = match text(value.get("kind")).as_str() {
		| "" | "box" => ShapeKind::Box,
		| "sphere" => ShapeKind::Sphere,
		| "mesh" => ShapeKind::Mesh,
		| other => return Err(err!(Asset("{other} is not a shape a body can be"))),
	};

	Ok(Form {
		kind,
		radius: number(value.get("radius"), 0.5),
		extents: vector(value.get("extents"), Vec3::splat(0.5)),
		mesh: text(value.get("mesh")),
	})
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

/// What the solver may do with a body.
fn body_kind(value: Option<&Value>) -> Result<BodyKind> {
	match text(value).as_str() {
		| "" | "static" => Ok(BodyKind::Static),
		| "kinematic" => Ok(BodyKind::Kinematic),
		| "dynamic" => Ok(BodyKind::Dynamic),
		| other => Err(err!(Asset("{other} is not a kind of body"))),
	}
}

/// Which of the three joints one is.
fn joint_kind(value: Option<&Value>) -> Result<JointKind> {
	match text(value).as_str() {
		| "" | "rope" => Ok(JointKind::Rope),
		| "weld" => Ok(JointKind::Weld),
		| "axis" => Ok(JointKind::Axis),
		| other => Err(err!(Asset("{other} is not a kind of joint"))),
	}
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

/// A number field, or a default.
fn number(value: Option<&Value>, default: f32) -> f32 {
	value.and_then(Value::as_f32).unwrap_or(default)
}

/// A flag field, or false.
fn flag(value: Option<&Value>) -> bool { matches!(value, Some(Value::Bool(true))) }

/// A three-number field, or a default.
///
/// A shorter list keeps the default in the axes it did not mention, which is
/// what `"color": [1, 0]` most likely meant and is in any case better than
/// silently reading the missing one as zero.
fn vector(value: Option<&Value>, default: Vec3) -> Vec3 {
	let Some(value) = value else {
		return default;
	};

	let parts = value.as_array();

	Vec3::new(
		number(parts.first(), default.x),
		number(parts.get(1), default.y),
		number(parts.get(2), default.z),
	)
}

/// An index that has to fit in a record.
fn count(value: usize) -> Result<u32> {
	u32::try_from(value)
		.map_err(|_| err!(Asset("a scene of {value} is more than one file holds")))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A source with one of everything in it.
	const SOURCE: &str = r#"{
		"stage": {
			"camera": { "position": [0, 6, 12], "target": [0, 1, 0], "fov": 1.2 },
			"gravity": [0, -20, 0]
		},
		"entities": [
			{ "name": "crate", "at": [1, 4, -2], "scale": [2, 2, 2],
			  "mesh": "cube", "material": "plastic", "color": [0.8, 0.2, 0.1] },
			{ "name": "hook", "at": [0, 8, 0] }
		],
		"bodies": [
			{ "name": "crate", "entity": "crate", "kind": "dynamic",
			  "shape": { "kind": "box", "extents": [1, 1, 1] },
			  "mass": 4.0, "friction": 0.7, "layer": 2, "collides": [0, 2] },
			{ "name": "ground", "kind": "static", "at": [0, -0.5, 0],
			  "shape": { "kind": "mesh", "mesh": "meshes/floor" } }
		],
		"joints": [
			{ "name": "rope", "kind": "rope", "first": "crate",
			  "anchors": [[0, 1, 0], [0, 8, 0]], "length": 3.5, "give": 0.05 }
		]
	}"#;

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
		let scene = import(r#"{ "entities": [ {} ], "bodies": [ {} ] }"#).expect("a scene");

		assert_eq!(scene.things[0].transform, Transform::IDENTITY, "an entity is at the origin");
		assert_eq!(scene.things[0].color, Vec3::ONE, "and untinted");
		assert_eq!(scene.solids[0].kind, BodyKind::Static, "a body is static");
		assert_eq!(scene.solids[0].shape.kind, ShapeKind::Box, "and a unit box");
		assert_eq!(scene.solids[0].shape.extents, Vec3::splat(0.5), "half a unit each way");
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
		let refused = import(r#"{ "entities": [ { "turn": [1, 1, 1, 1] } ] }"#)
			.expect_err("that is not a rotation")
			.to_string();

		assert!(refused.contains("unit"), "and it says what one is: {refused}");

		let half = std::f32::consts::FRAC_1_SQRT_2;
		let text = format!(r#"{{ "entities": [ {{ "turn": [0, {half}, 0, {half}] }} ] }}"#);
		let scene = import(&text).expect("a quarter turn is a rotation");

		assert!(scene.things[0].transform.rotation.is_normalized(), "and it survives");
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
				{ "kind": "axis", "first": "b", "second": "c", "axis": [1, 0, 0] }
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
		assert_eq!(joints, vec![JointKind::Rope, JointKind::Weld, JointKind::Axis]);
		assert!(scene.solids[2].sensor, "and a sensor says so");
		assert!((scene.solids[1].shape.radius - 2.0).abs() < f32::EPSILON, "with its radius");
	}

	#[test]
	fn text_that_is_not_json_at_all_is_an_error() {
		assert!(import("this is not a scene").is_err(), "and it does not panic");
		assert!(import("").is_err(), "nor does an empty file");
	}
}
