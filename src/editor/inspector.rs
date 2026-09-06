//! One thing in the world at a time, in detail.
//!
//! **The inspector is one function over a table.** Every record with a
//! [`Field`] table is shown by [`inspect`]: a row per plain field, a widget
//! the field's kind decides, and a write only when a number actually moved.
//! Nothing here knows what a body's fields are; a field added to the table is
//! a row here the same day, which is the whole reason the table exists. What
//! is still drawn by hand is a relationship - what a joint holds, what an
//! entity hangs off - because a handle is worth showing by name and a table
//! knows nothing about names.
//!
//! **Every write is written down first**, @ref [`History::begin`]: a row is
//! edited on a copy of the record, and the copy goes back into the world only
//! after the world as it stood has been captured, so that a number dragged
//! in a field is one step back however many frames the drag took.
//!
//! Everything that can be tested lives in [`select`](crate::select); what is
//! here is the drawing, and it is checked by looking at it - except that an
//! inspector nobody touches writes nothing, which a headless frame can check.

use colby_core::{
	abi::{
		Body, BodyId, EntityId, Field, Joint, JointId, Light, Renderable, Sky, Transform, World,
		field::{Kind, Value},
		scene::{self, Stage},
	},
	glam::{EulerRot, Quat, Vec3},
};
use egui::{ComboBox, DragValue, Grid, ScrollArea, Ui};

use crate::{
	history::History,
	select::{self, Pick, Selection},
};

/// How far a drag of one pixel moves three numbers.
const MOVE_SPEED: f32 = 0.02;

/// How far a drag of one pixel turns something, in degrees.
const TURN_SPEED: f32 = 0.5;

/// How far a drag of one pixel moves a number on its own.
const STEP_SPEED: f32 = 0.01;

/// Draws the inspector into a panel.
///
/// @param ui - the panel
/// @param world - the tables to show, and to edit
/// @param selection - what is selected; the primary is shown
/// @param history - where a write is written down, so that it can be undone
/// @param rename - whether the name field is to take the keyboard this
/// frame, because somebody pressed the key for it
pub(crate) fn show(
	ui: &mut Ui,
	world: &mut World,
	selection: &Selection,
	history: &mut History,
	rename: bool,
) {
	ScrollArea::vertical()
		.auto_shrink([false, false])
		.show(ui, |ui| {
			if selection.len() > 1 {
				ui.label(format!(
					"{} selected: the last picked is shown, and a drag moves all of them",
					selection.len()
				));
				ui.separator();
			}

			detail(ui, world, selection.at(), history, rename);
		});
}

/// The selected thing, in detail.
fn detail(ui: &mut Ui, world: &mut World, pick: Pick, history: &mut History, rename: bool) {
	match pick {
		| Pick::Nothing => {
			ui.label("nothing selected, so this is the world itself");
			ui.separator();
			settings(ui, world, history);
		},
		| Pick::Entity(id) => {
			naming(ui, world, pick, history, rename);
			hanging(ui, world, id);
			placing(ui, world, pick, history);
			look(ui, world, id, history);
			lamp(ui, world, id, history);
		},
		| Pick::Body(id) => {
			naming(ui, world, pick, history, rename);
			ui.label(world.bodies.get(id).map_or_else(
				|| "gone".to_owned(),
				|body| format!("a {}", select::body_words(body)),
			));
			placing(ui, world, pick, history);
			solid(ui, world, id, history);
		},
		| Pick::Joint(id) => {
			naming(ui, world, pick, history, rename);
			tie(ui, world, id, history);
		},
	}

	ui.separator();
	// the honest answer to "why did my drag not stick", which used to be a
	// workaround written on the panel and is now a mode. @ref
	// `colby_core::abi::World::editing`.
	ui.label(if world.editing {
		"editing, so these are yours"
	} else {
		"playing, so the game may write these back every step. F5 to edit"
	});
}

/// The name field.
///
/// @param rename - whether it takes the keyboard this frame: F2's whole job
fn naming(ui: &mut Ui, world: &mut World, pick: Pick, history: &mut History, rename: bool) {
	let mut name = pick.name(world).to_owned();

	ui.horizontal(|ui| {
		ui.label("name");

		let response = ui.text_edit_singleline(&mut name);

		if rename {
			response.request_focus();
		}

		if response.changed() {
			history.begin("rename", world);
			select::rename(world, pick, &name);
		}
	});
}

/// What an entity hangs off, which is read rather than written here.
///
/// Hanging one entity off another is the hierarchy's job - a row dragged
/// onto another - and the fact is shown here so that the numbers under it
/// read right: a child's place is inside its parent.
fn hanging(ui: &mut Ui, world: &World, id: EntityId) {
	let parent = world.entities.parent(id);

	if !parent.is_some() {
		return;
	}

	ui.horizontal(|ui| {
		ui.label("inside");
		ui.monospace(select::entity_label(world, parent));
	});
}

/// Position, rotation and scale, for the things that have them.
///
/// In the thing's own terms - inside its parent, for an entity that hangs off
/// one - because that is what a person expects to type; the gizmo in the
/// viewport works in the world. @ref `select::local`.
fn placing(ui: &mut Ui, world: &mut World, pick: Pick, history: &mut History) {
	let Some(transform) = select::local(world, pick) else {
		return;
	};

	let mut edited = transform;

	if inspect(ui, "transform", &mut edited, Transform::FIELDS) {
		history.begin("place", world);
		select::place_local(world, pick, edited);
	}
}

/// What an entity looks like: the plain half of its renderable, which is the
/// tint. The mesh, the material and the pose are handles, and the tree names
/// the mesh in the row above.
fn look(ui: &mut Ui, world: &mut World, id: EntityId, history: &mut History) {
	let Some(mut renderable) = world.entities.renderable(id).copied() else {
		return;
	};

	if inspect(ui, "renderable", &mut renderable, Renderable::FIELDS) {
		history.begin("tint", world);
		world.entities.set_renderable(id, renderable);
	}
}

/// The world's own settings, shown when nothing in it is.
///
/// **Where an environment lives when there is no node to hang it on.** Godot
/// puts one on a `WorldEnvironment` and Unreal on a volume; colby has neither,
/// and the honest place for "the clear color, the sun, the ambient, the
/// gravity and the sky" is the panel that is otherwise empty. It is also the
/// first thing "nothing selected" has ever said that is worth reading.
///
/// The camera and the clock are in the record and are deliberately not
/// written back. @ref `colby_core::abi::scene::set_settings`.
fn settings(ui: &mut Ui, world: &mut World, history: &mut History) {
	let mut stage = scene::settings(world);
	let mut moved = inspect(ui, "world", &mut stage, Stage::FIELDS);

	moved |= inspect(ui, "sky", &mut stage.sky, Sky::FIELDS);

	if moved {
		history.begin("world", world);
		scene::set_settings(world, stage);
	}
}

/// What an entity shines, if anything.
///
/// Always drawn, whatever the entity is: turning a crate into a lamp is
/// picking a word in a drop-down, and a section that appeared only for
/// entities that were already lights would leave nowhere to do it. Every
/// field is plain, so the whole of it is the table. @ref
/// `colby_core::abi::light`.
fn lamp(ui: &mut Ui, world: &mut World, id: EntityId, history: &mut History) {
	let Some(mut light) = world.entities.light(id).copied() else {
		return;
	};

	if inspect(ui, "light", &mut light, Light::FIELDS) {
		history.begin("light", world);
		world.entities.set_light(id, light);
	}
}

/// Everything the solver reads about a body, to edit.
///
/// Its place is the row above, through the transform's own table, and the
/// entity it drives is the branch it hangs under in the tree.
fn solid(ui: &mut Ui, world: &mut World, id: BodyId, history: &mut History) {
	let Some(mut body) = world.bodies.get(id).copied() else {
		return;
	};

	if inspect(ui, "body", &mut body, Body::FIELDS) {
		history.begin("body", world);

		if let Some(held) = world.bodies.get_mut(id) {
			*held = body;
		}
	}
}

/// What a joint holds, by name, and then everything else about it.
///
/// Its two bodies are handles, which the table describes and cannot name, so
/// the two rows that name them are drawn here; the anchors are in each body's
/// own space, and are numbers all the same.
fn tie(ui: &mut Ui, world: &mut World, id: JointId, history: &mut History) {
	let Some(mut joint) = world.joints.get(id).copied() else {
		return;
	};

	Grid::new("held").num_columns(2).show(ui, |ui| {
		ui.label("first");
		ui.monospace(select::body_label(world, joint.first));
		ui.end_row();

		ui.label("second");
		ui.monospace(if joint.second.is_some() {
			select::body_label(world, joint.second)
		} else {
			"a point in the world".to_owned()
		});
		ui.end_row();
	});

	if inspect(ui, "joint", &mut joint, Joint::FIELDS) {
		history.begin("joint", world);

		if let Some(held) = world.joints.get_mut(id) {
			*held = joint;
		}
	}
}

/// One inspector over any record with a table: a row per plain field, and a
/// widget the field's kind decides.
///
/// A reference is left out. The table says a body drives an entity, and a row
/// that could only show a slot number would say less than the tree already
/// does by nesting one under the other; where a relationship is worth a row,
/// the caller draws it by name, @ref [`tie`].
///
/// **A write is guarded by the numbers having actually changed**, field by
/// field, and that matters more than it looks for a rotation: two different
/// triples of angles can name one rotation, so converting out and straight
/// back in every frame would walk a rotation somewhere it was never dragged.
///
/// @param ui - where to draw
/// @param salt - what tells this grid from another in the same panel
/// @param record - what to show and edit
/// @param fields - its table
/// @return whether any field was written
fn inspect<T>(ui: &mut Ui, salt: &str, record: &mut T, fields: &[Field<T>]) -> bool {
	let mut edited = false;

	Grid::new(salt).num_columns(2).show(ui, |ui| {
		for field in fields {
			if field.kind.is_reference() {
				continue;
			}

			ui.label(field.name).on_hover_text(field.help);

			let held = field.get(record);
			let mut value = held.clone();
			widget(ui, field, &mut value);

			if value != held && field.set(record, value) {
				edited = true;
			}

			ui.end_row();
		}
	});

	edited
}

/// The widget one value is edited with, by its kind.
///
/// @param ui - where to draw
/// @param field - whose value it is, for the words a word may be and for an
/// id no other widget in the panel has
/// @param value - what to draw and edit in place
fn widget<T>(ui: &mut Ui, field: &Field<T>, value: &mut Value) {
	match value {
		| Value::Bool(held) => {
			ui.checkbox(held, "");
		},
		| Value::Int(held) => {
			ui.add(DragValue::new(held));
		},
		| Value::Float(held) => {
			ui.add(DragValue::new(held).speed(STEP_SPEED));
		},
		| Value::Text(held) => {
			ui.text_edit_singleline(held);
		},
		| Value::Vec3(held) => vector(ui, held, MOVE_SPEED),
		| Value::Quat(held) => turn(ui, held),
		| Value::Color(held) => color(ui, held),
		| Value::Word(held) => words(ui, field.name, field.kind, held),
		// never reached: a reference is skipped before a widget is asked for,
		// @ref `inspect`. A slot number is what there would be to show.
		| Value::Entity(_)
		| Value::Body(_)
		| Value::Joint(_)
		| Value::Pose(_)
		| Value::Mesh(_)
		| Value::Material(_) => {
			ui.monospace("a reference");
		},
	}
}

/// Three numbers on one row.
fn vector(ui: &mut Ui, value: &mut Vec3, speed: f32) {
	ui.horizontal(|ui| {
		ui.add(
			DragValue::new(&mut value.x)
				.speed(speed)
				.prefix("x "),
		);
		ui.add(
			DragValue::new(&mut value.y)
				.speed(speed)
				.prefix("y "),
		);
		ui.add(
			DragValue::new(&mut value.z)
				.speed(speed)
				.prefix("z "),
		);
	});
}

/// A rotation as three angles in degrees.
///
/// A quaternion is not a thing anyone types, so this shows the yaw, pitch and
/// roll it stands for. **The write is guarded by the numbers having actually
/// changed**, and that matters more than it looks: two different triples can
/// name one rotation, so converting out and straight back in every frame would
/// walk a rotation somewhere it was never dragged.
fn turn(ui: &mut Ui, rotation: &mut Quat) {
	let (yaw, pitch, roll) = rotation.to_euler(EulerRot::YXZ);
	let held = Vec3::new(pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees());
	let mut edited = held;

	vector(ui, &mut edited, TURN_SPEED);

	if edited != held {
		*rotation = Quat::from_euler(
			EulerRot::YXZ,
			edited.y.to_radians(),
			edited.x.to_radians(),
			edited.z.to_radians(),
		);
	}
}

/// A color, as a swatch that opens a picker.
///
/// Guarded by the widget's own word rather than by comparing numbers: the
/// picker keeps its color as hue, saturation and value and writes the three
/// channels back from that every frame, which is a round trip through a
/// different number of bits, and comparing would see a change where nobody
/// made one.
fn color(ui: &mut Ui, value: &mut Vec3) {
	let mut rgb = value.to_array();

	if ui.color_edit_button_rgb(&mut rgb).changed() {
		*value = Vec3::from_array(rgb);
	}
}

/// One of a few words, as a drop-down over the field's own list.
fn words(ui: &mut Ui, salt: &str, kind: Kind, held: &mut u32) {
	let list = kind.words();
	let shown = usize::try_from(*held)
		.ok()
		.and_then(|index| list.get(index))
		.copied()
		.unwrap_or("?");

	ComboBox::from_id_salt(salt)
		.selected_text(shown)
		.show_ui(ui, |ui| {
			for (index, word) in list.iter().enumerate() {
				if let Ok(index) = u32::try_from(index) {
					ui.selectable_value(held, index, *word);
				}
			}
		});
}

#[cfg(test)]
mod tests {
	use colby_core::abi::{MeshId, Shape};
	use egui::{Context, RawInput};

	use super::*;

	/// Runs one frame of an inspector over a record with nobody touching it,
	/// and hands back what the frame reported.
	fn untouched<T: Clone + PartialEq + core::fmt::Debug>(
		record: &T,
		fields: &[Field<T>],
	) -> bool {
		let context = Context::default();
		let mut edited = record.clone();
		let mut written = false;

		// nothing paints this frame, and epaint asserts that a texture delta is
		// applied rather than dropped - the right rule for a painter and the
		// wrong one for a test - so the delta is cleared on purpose, the way
		// the editor does on its way out.
		let mut output = context.run_ui(RawInput::default(), |ui| {
			written = inspect(ui, "test", &mut edited, fields);
		});
		output.textures_delta.clear();

		assert_eq!(edited, *record, "nothing was touched, so nothing may have moved");

		written
	}

	#[test]
	fn an_inspector_nobody_touches_writes_nothing() {
		// the trap the rotation row guards against: a rotation shown as three
		// angles and read straight back is not always the same rotation, and
		// an inspector that wrote it back every frame would walk it. A
		// rotation that is none of the easy ones, so the round trip is real.
		let turned = Transform {
			position: Vec3::new(1.5, -2.0, 0.25),
			rotation: Quat::from_euler(EulerRot::YXZ, 1.2, -0.4, 2.9),
			scale: Vec3::new(1.0, 2.0, 0.5),
		};

		assert!(!untouched(&turned, Transform::FIELDS), "a transform stays put");

		let body = Body::dynamic(Shape::ball(0.7), turned, 2.5)
			.moving(Vec3::X, Vec3::Y)
			.surfaced(0.3, 0.9);

		assert!(!untouched(&body, Body::FIELDS), "and so does a body");

		let mut joint = Joint::weld(BodyId::at(1, 1), BodyId::at(2, 1), (Vec3::X, Vec3::Z))
			.sprung(4.0, 0.7)
			.capped(12.0, 3.0);
		joint.rest = turned.rotation;

		assert!(!untouched(&joint, Joint::FIELDS), "and a joint, rest rotation and all");
		assert!(
			!untouched(
				&Renderable::new(MeshId::CUBE, Vec3::new(0.2, 0.7, 0.9)),
				Renderable::FIELDS
			),
			"and a tint through the picker"
		);
		assert!(
			!untouched(&Light::spot(Vec3::new(1.0, 0.9, 0.7), 3.0, 8.0, 0.2, 0.5), Light::FIELDS),
			"and a lamp, its word and its two angles included"
		);
		assert!(
			!untouched(
				&Sky::gradient(
					Vec3::new(0.1, 0.2, 0.5),
					Vec3::new(0.6, 0.7, 0.8),
					Vec3::new(0.1, 0.1, 0.1)
				),
				Sky::FIELDS
			),
			"and a sky"
		);
	}

	#[test]
	fn the_world_itself_is_what_the_panel_shows_when_nothing_in_it_is() {
		let context = Context::default();
		let mut world = World::new();
		world.sky = Sky::day();
		world.clear = Vec3::new(0.3, 0.4, 0.5);
		let mut history = History::default();
		let was = scene::settings(&world);

		let mut output = context.run_ui(RawInput::default(), |ui| {
			detail(ui, &mut world, Pick::Nothing, &mut history, false);
		});
		output.textures_delta.clear();

		assert_eq!(
			scene::settings(&world),
			was,
			"a frame nobody touched writes nothing back to the world"
		);
		assert_eq!(history.undoable(), None, "and nothing is written down either");
	}

	#[test]
	fn the_settings_a_panel_writes_back_are_the_ones_its_table_names() {
		// the pairing `set_settings` is documented with: everything in
		// `Stage::FIELDS` has to survive the trip, and the camera and the
		// clock have to be left exactly where they were.
		let mut world = World::new();
		world.camera.position = Vec3::new(9.0, 9.0, 9.0);
		world.time = 42.0;
		world.steps = 700;

		let mut stage = scene::settings(&world);
		stage.clear = Vec3::new(0.1, 0.2, 0.3);
		stage.sky = Sky::day();
		stage.light = Vec3::new(1.0, -2.0, 3.0);
		stage.ambient = Vec3::splat(0.4);
		stage.gravity = Vec3::new(0.0, -3.0, 0.0);
		stage.camera.position = Vec3::ZERO;
		stage.time = 0.0;

		scene::set_settings(&mut world, stage);

		assert_eq!(world.clear, stage.clear, "the clear color went in");
		assert_eq!(world.sky, stage.sky, "and the sky");
		assert_eq!(world.light, stage.light, "and the sun");
		assert_eq!(world.ambient, stage.ambient, "and the ambient");
		assert_eq!(world.gravity, stage.gravity, "and the gravity");
		assert_eq!(
			world.camera.position,
			Vec3::new(9.0, 9.0, 9.0),
			"and the camera was left where whoever is flying it put it"
		);
		assert!((world.time - 42.0).abs() < 1.0e-6, "and the clock was not moved");
		assert_eq!(world.steps, 700, "in either half");
	}
}
