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
		BodyKind, Camera, Joint, JointKind, Layers, ShapeKind, Transform,
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
			&[
				"name",
				"kind",
				"first",
				"second",
				"anchors",
				"axis",
				"length",
				"stiffness",
				"damping",
				"max impulse",
				"max torque",
			],
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
			stiffness: number(entry.get("stiffness"), Joint::RIGID),
			damping: number(entry.get("damping"), Joint::DAMPING),
			max_impulse: number(entry.get("max impulse"), Joint::NO_CEILING),
			max_torque: number(entry.get("max torque"), Joint::NO_CEILING),
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
	// the same half unit an empty `shape` gives, and not nothing. A box does
	// not use a radius, so the two were invisibly different for as long as
	// nothing wrote a scene back out; a writer that has to tell them apart has
	// to write `"shape": {}` for the second, which is a thing nobody should
	// have to read. @ref `export`.
	let Some(value) = value else {
		return Ok(Form {
			kind: ShapeKind::Box,
			radius: 0.5,
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
	spellable(scene)?;

	let thing_names = named(
		&scene.things,
		|thing| thing.name.as_str(),
		|index| scene.solids.iter().any(|it| it.thing == index),
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

	let things: Vec<String> = scene
		.things
		.iter()
		.zip(&thing_names)
		.map(|(thing, name)| thing_of(thing, name))
		.collect();
	let solids: Vec<String> = scene
		.solids
		.iter()
		.zip(&solid_names)
		.map(|(solid, name)| solid_of(scene, solid, name, &thing_names))
		.collect();
	let links: Vec<String> = scene
		.links
		.iter()
		.zip(&link_names)
		.map(|(link, name)| link_of(link, name, &solid_names))
		.collect();

	// gathered rather than appended, so that the comma between two of them is
	// written by whatever knows there is a next one. A trailing one is the
	// single thing JSON refuses that is easy to write by accident.
	let parts: Vec<String> = [
		stage_of(&scene.stage),
		block("entities", &things),
		block("bodies", &solids),
		block("joints", &links),
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
fn stage_of(stage: &Stage) -> Option<String> {
	let camera = &stage.camera;
	let start = &Camera::DEFAULT;
	let mut inner: Vec<(&str, String)> = Vec::new();

	let mut lens: Vec<(&str, String)> = Vec::new();
	put_vector(&mut lens, "position", camera.position, start.position);
	put_vector(&mut lens, "target", camera.target, start.target);
	put_vector(&mut lens, "up", camera.up, start.up);
	put_number(&mut lens, "fov", camera.fov_y, start.fov_y);
	put_number(&mut lens, "near", camera.near, start.near);
	put_number(&mut lens, "far", camera.far, start.far);

	if !lens.is_empty() {
		inner.push(("camera", object(&lens)));
	}

	put_vector(&mut inner, "clear", stage.clear, Stage::DEFAULT.clear);
	put_vector(&mut inner, "light", stage.light, Stage::DEFAULT.light);
	put_vector(&mut inner, "ambient", stage.ambient, Stage::DEFAULT.ambient);
	put_vector(&mut inner, "gravity", stage.gravity, Stage::DEFAULT.gravity);

	if inner.is_empty() {
		return None;
	}

	Some(format!("\t\"stage\": {}", object(&inner)))
}

/// One entity.
fn thing_of(thing: &Thing, name: &str) -> String {
	let mut fields: Vec<(&str, String)> = Vec::new();

	put_text(&mut fields, "name", name);
	put_place(&mut fields, &thing.transform, &Transform::IDENTITY);
	put_text(&mut fields, "mesh", &thing.mesh);
	put_text(&mut fields, "material", &thing.material);
	put_vector(&mut fields, "color", thing.color, Vec3::ONE);

	object(&fields)
}

/// One body, with the entity it drives named rather than numbered.
fn solid_of(scene: &SceneData, solid: &Solid, name: &str, things: &[String]) -> String {
	let mut fields: Vec<(&str, String)> = Vec::new();

	put_text(&mut fields, "name", name);

	let driven = at_index(things, solid.thing);
	put_text(&mut fields, "entity", &driven);

	if solid.kind != BodyKind::Static {
		fields.push(("kind", as_text(kind_word(solid.kind))));
	}

	if let Some(form) = form_of(&solid.shape) {
		fields.push(("shape", form));
	}

	// a body with an entity and no place of its own stands where the entity
	// does, which is what leaving it out means on the way in. So the three are
	// written only when they differ from that - and then `at` is written even
	// if it is nothing, because it is its presence that decides.
	let standing = scene
		.things
		.get(usize::try_from(solid.thing).unwrap_or(usize::MAX))
		.map_or(Transform::IDENTITY, |thing| thing.transform);
	put_place(&mut fields, &solid.transform, &standing);

	put_vector(&mut fields, "velocity", solid.velocity, Vec3::ZERO);
	put_vector(&mut fields, "angular", solid.angular, Vec3::ZERO);
	put_number(&mut fields, "mass", solid.mass, 1.0);
	put_number(&mut fields, "restitution", solid.restitution, 0.2);
	put_number(&mut fields, "friction", solid.friction, 0.5);

	if solid.sensor {
		fields.push(("sensor", "true".to_owned()));
	}

	put_layers(&mut fields, solid.layers);

	object(&fields)
}

/// One joint, with both bodies named rather than numbered.
fn link_of(link: &Link, name: &str, solids: &[String]) -> String {
	let mut fields: Vec<(&str, String)> = Vec::new();

	put_text(&mut fields, "name", name);

	if link.kind != JointKind::Rope {
		fields.push(("kind", as_text(joint_word(link.kind))));
	}

	put_text(&mut fields, "first", &at_index(solids, link.first));
	put_text(&mut fields, "second", &at_index(solids, link.second));

	if link.first_anchor != Vec3::ZERO || link.second_anchor != Vec3::ZERO {
		fields.push((
			"anchors",
			format!("[{}, {}]", as_vector(link.first_anchor), as_vector(link.second_anchor)),
		));
	}

	put_vector(&mut fields, "axis", link.axis, Vec3::Y);
	put_number(&mut fields, "length", link.length, 1.0);
	put_number(&mut fields, "stiffness", link.stiffness, Joint::RIGID);
	put_number(&mut fields, "damping", link.damping, Joint::DAMPING);
	put_number(&mut fields, "max impulse", link.max_impulse, Joint::NO_CEILING);
	put_number(&mut fields, "max torque", link.max_torque, Joint::NO_CEILING);

	object(&fields)
}

/// What a body is shaped like, or nothing if it is the shape a body has
/// without a `shape` field at all.
///
/// The two used to differ - a missing `shape` gave a radius of nothing and an
/// empty one half a unit - and a writer had to tell them apart by writing
/// `"shape": {}`, which is a thing nobody should have to read in a file meant
/// to be read. The reader was made to agree with itself instead. That
/// asymmetry had been there since the format existed and nothing could see it
/// until something had to write one back out.
fn form_of(form: &Form) -> Option<String> {
	let absent = form.kind == ShapeKind::Box
		&& form.extents == Vec3::splat(0.5)
		&& form.mesh.is_empty()
		&& (form.radius - 0.5).abs() < f32::EPSILON;

	if absent {
		return None;
	}

	let mut fields: Vec<(&str, String)> = Vec::new();

	if form.kind != ShapeKind::Box {
		fields.push(("kind", as_text(shape_word(form.kind))));
	}

	put_number(&mut fields, "radius", form.radius, 0.5);
	put_vector(&mut fields, "extents", form.extents, Vec3::splat(0.5));
	put_text(&mut fields, "mesh", &form.mesh);

	Some(object(&fields))
}

/// Which layers a body is on, back as the numbers a person writes.
///
/// @note: the source says one layer per body, so a body somehow on several is
/// written on its lowest and the rest are lost. Nothing in the engine makes
/// one - `Layers::single` and the default both set exactly one bit - and the
/// alternative is a format where `"layer"` is sometimes a list.
fn put_layers(fields: &mut Vec<(&'static str, String)>, layers: Layers) {
	let on = layers.layer.trailing_zeros().min(u32::BITS - 1);

	if layers.layer != Layers::DEFAULT.layer {
		fields.push(("layer", on.to_string()));
	}

	if layers.mask == u32::MAX {
		return;
	}

	let with: Vec<String> = (0..u32::BITS)
		.filter(|index| layers.mask & Layers::bit(*index) != 0)
		.map(|index| index.to_string())
		.collect();

	fields.push(("collides", format!("[{}]", with.join(", "))));
}

/// Where something is, if it is anywhere other than where it would be anyway.
///
/// The three go together: [`import`] reads `turn` and `scale` only in the
/// presence of `at`, so writing one of them without it would quietly lose the
/// other two.
fn put_place(
	fields: &mut Vec<(&'static str, String)>,
	transform: &Transform,
	otherwise: &Transform,
) {
	if transform == otherwise {
		return;
	}

	fields.push(("at", as_vector(transform.position)));

	if transform.rotation != Quat::IDENTITY {
		fields.push(("turn", as_turn(transform.rotation)));
	}

	if transform.scale != Vec3::ONE {
		fields.push(("scale", as_vector(transform.scale)));
	}
}

/// A text field, unless it is empty.
fn put_text(fields: &mut Vec<(&'static str, String)>, name: &'static str, value: &str) {
	if value.is_empty() {
		return;
	}

	fields.push((name, as_text(value)));
}

/// A number, unless it is the one it would be anyway.
#[expect(
	clippy::float_cmp,
	reason = "the question is whether the reader would produce this exact number from nothing 	          at all, which is an exact comparison and not an approximate one: a tolerance 	          here would throw away a small deliberate value"
)]
fn put_number(
	fields: &mut Vec<(&'static str, String)>,
	name: &'static str,
	value: f32,
	otherwise: f32,
) {
	if value == otherwise {
		return;
	}

	fields.push((name, as_number(value)));
}

/// Three numbers, unless they are the ones they would be anyway.
fn put_vector(
	fields: &mut Vec<(&'static str, String)>,
	name: &'static str,
	value: Vec3,
	otherwise: Vec3,
) {
	if value == otherwise {
		return;
	}

	fields.push((name, as_vector(value)));
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

/// A one-line object out of fields that are already written.
fn object(fields: &[(&str, String)]) -> String {
	let inner: Vec<String> = fields
		.iter()
		.map(|(name, value)| format!("\"{name}\": {value}"))
		.collect();

	if inner.is_empty() {
		return "{}".to_owned();
	}

	format!("{{ {} }}", inner.join(", "))
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
fn as_text(value: &str) -> String {
	let mut out = String::with_capacity(value.len() + 2);
	out.push('"');

	for letter in value.chars() {
		match letter {
			| '"' => out.push_str("\\\""),
			| '\\' => out.push_str("\\\\"),
			| '\n' => out.push_str("\\n"),
			| '\t' => out.push_str("\\t"),
			| '\r' => out.push_str("\\r"),
			// anything below a space has no spelling of its own and has to go
			// as a code point. Above it, JSON takes the character as it is.
			| _ if u32::from(letter) < 0x20 => escaped(&mut out, letter),
			| _ => out.push(letter),
		}
	}

	out.push('"');

	out
}

/// One character below a space, as the four hex digits JSON spells it with.
fn escaped(out: &mut String, letter: char) {
	const DIGITS: [char; 16] =
		['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'];

	// everything this is reached for is below 0x20, so the first two digits are
	// always zero and the last two are one byte's worth.
	let code = usize::try_from(u32::from(letter)).unwrap_or(0);

	out.push_str("\\u00");
	out.push(DIGITS[(code >> 4) & 0xF]);
	out.push(DIGITS[code & 0xF]);
}

/// The word a kind of body is written as.
const fn kind_word(kind: BodyKind) -> &'static str {
	match kind {
		| BodyKind::Static => "static",
		| BodyKind::Kinematic => "kinematic",
		| BodyKind::Dynamic => "dynamic",
	}
}

/// The word a kind of joint is written as.
const fn joint_word(kind: JointKind) -> &'static str {
	match kind {
		| JointKind::Rope => "rope",
		| JointKind::Weld => "weld",
		| JointKind::Axis => "axis",
	}
}

/// The word a kind of shape is written as.
const fn shape_word(kind: ShapeKind) -> &'static str {
	match kind {
		| ShapeKind::Box => "box",
		| ShapeKind::Sphere => "sphere",
		| ShapeKind::Mesh => "mesh",
	}
}

/// Refuses a description holding a number JSON has no spelling for.
///
/// An infinity or a nan is what a world that has blown up is full of, and
/// writing one produces a file that will not read back. Better to say so than
/// to write `inf` and have the compiler refuse it later with no idea where it
/// came from.
fn spellable(scene: &SceneData) -> Result<()> {
	let stage = &scene.stage;
	let camera = &stage.camera;

	let settled = camera.position.is_finite()
		&& camera.target.is_finite()
		&& camera.up.is_finite()
		&& [camera.fov_y, camera.near, camera.far]
			.iter()
			.all(|it| it.is_finite())
		&& stage.clear.is_finite()
		&& stage.light.is_finite()
		&& stage.ambient.is_finite()
		&& stage.gravity.is_finite();

	if !settled {
		return Err(err!(Asset("the stage holds a number JSON cannot write")));
	}

	for thing in &scene.things {
		if !finite(&thing.transform) || !thing.color.is_finite() {
			return Err(err!(Asset("an entity holds a number JSON cannot write")));
		}
	}

	for solid in &scene.solids {
		let good = finite(&solid.transform)
			&& solid.velocity.is_finite()
			&& solid.angular.is_finite()
			&& solid.shape.extents.is_finite()
			&& [solid.mass, solid.restitution, solid.friction, solid.shape.radius]
				.iter()
				.all(|it| it.is_finite());

		if !good {
			return Err(err!(Asset("a body holds a number JSON cannot write")));
		}
	}

	for link in &scene.links {
		let good = link.first_anchor.is_finite()
			&& link.second_anchor.is_finite()
			&& link.axis.is_finite()
			&& [link.length, link.stiffness, link.damping, link.max_impulse, link.max_torque]
				.iter()
				.all(|it| it.is_finite());

		if !good {
			return Err(err!(Asset("a joint holds a number JSON cannot write")));
		}
	}

	Ok(())
}

/// Whether every number in a transform is one that can be written.
fn finite(transform: &Transform) -> bool {
	transform.position.is_finite()
		&& transform.rotation.is_finite()
		&& transform.scale.is_finite()
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
			  "anchors": [[0, 1, 0], [0, 8, 0]], "length": 3.5,
			  "stiffness": 8.0, "damping": 0.6, "max impulse": 45.0, "max torque": 12.5 }
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
			!body.contains("\"at\""),
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
		let (_, back) = round(&export(&scene).expect("it can be written"));

		assert!(
			(back.solids[0].shape.radius - 2.0).abs() < 1.0e-6,
			"the radius came back: {}",
			back.solids[0].shape.radius
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
		// a capture carries four things the text has no words for. What comes
		// back is what a source would have produced, which is the honest
		// answer rather than a silent half.
		let mut scene = import(SOURCE).expect("it is a scene");
		scene.solids[0].sleeping = true;
		scene.stage.time = 12.5;
		scene.stage.steps = 750;
		scene.things[0].slot = 40;
		scene.things[0].generation = 9;

		let (_, again) = round(&export(&scene).expect("it can be written"));

		assert!(!again.solids[0].sleeping, "a source cannot say a body is asleep");
		assert!((again.stage.time).abs() < 1.0e-6, "nor what time it is");
		assert_eq!(again.stage.steps, 0, "nor how many steps have run");
		assert_eq!(again.things[0].slot, 0, "and a record's slot is its place in the file");
		assert_eq!(again.things[0].generation, 1, "on its first occupant");
	}
}
