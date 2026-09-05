//! The world as a tree, and one thing in it at a time in detail.
//!
//! **What the tree groups by is what the tables say.** An entity hangs off
//! another by [`parent`](colby_core::abi::Entities::parent) since 1d, and the
//! tree does not draw that yet: what it draws is the older relationship, a
//! body naming the entity it drives, so an entity's bodies hang under it, the
//! bodies driving nothing stand on their own, and the joints are a group of
//! their own at the bottom. The hierarchy panel that draws the parents is the
//! editor application's, and it will be a second kind of nesting rather than a
//! replacement for this.
//!
//! **What a thing is called comes from the world, not from here.** A panel that
//! kept its own names would lose them the moment a scene was loaded. Something
//! nobody has named is shown by what it is made of, in angle brackets -
//! `<cube>` for an entity drawing that mesh, `<dynamic sphere 4>` for a body -
//! so that a name and a description can never be mistaken for each other.
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
//! Everything that can be tested lives in [`select`](crate::select); what is
//! here is the drawing, and it is checked by looking at it - except that an
//! inspector nobody touches writes nothing, which a headless frame can check.

use colby_core::{
	abi::{
		Body, BodyId, EntityId, Field, Joint, JointId, Renderable, Transform, World, console,
		field::{Kind, Value},
	},
	glam::{EulerRot, Quat, Vec3},
};
use egui::{ComboBox, Context, DragValue, Grid, ScrollArea, Window};

use crate::{
	gizmo::Tool,
	select::{self, Pick, Selection},
};

/// How tall the tree is allowed to get before it scrolls.
const TREE_HEIGHT: f32 = 240.0;

/// How far a drag of one pixel moves three numbers.
const MOVE_SPEED: f32 = 0.02;

/// How far a drag of one pixel turns something, in degrees.
const TURN_SPEED: f32 = 0.5;

/// How far a drag of one pixel moves a number on its own.
const STEP_SPEED: f32 = 0.01;

/// The scene window's own state.
#[derive(Debug, Default)]
pub(crate) struct Tree {
	/// The living entity handles, refilled every frame.
	///
	/// Refilled rather than held: the tables are the host's, and anything in
	/// them can go away between one frame and the next.
	entities: Vec<EntityId>,

	/// The living body handles, refilled every frame.
	bodies: Vec<BodyId>,

	/// The living joint handles, refilled every frame.
	joints: Vec<JointId>,

	/// What the world would be called if it were written out now.
	///
	/// Kept here rather than read from anywhere, because there is nowhere to
	/// read it from: a world does not know which file it came from. Naming the
	/// file is part of writing it, exactly as it is at a console.
	filed: String,
}

impl Tree {
	/// Draws the scene window.
	///
	/// @param context - egui, mid-frame
	/// @param world - the tables to show, and to edit
	/// @param selection - what is selected, which a row here may change
	/// @param tool - what the gizmo out in the world is doing, so that the
	/// three keys that switch it are written down somewhere
	pub(crate) fn show(
		&mut self,
		context: &Context,
		world: &mut World,
		selection: &mut Selection,
		tool: Tool,
	) {
		self.gather(world);

		Window::new("scene")
			.default_pos([820.0, 12.0])
			.default_width(320.0)
			.show(context, |ui| {
				ui.label(format!(
					"{} entities, {} bodies, {} joints",
					self.entities.len(),
					self.bodies.len(),
					self.joints.len()
				));

				ScrollArea::vertical()
					.max_height(TREE_HEIGHT)
					.auto_shrink([false, true])
					.show(ui, |ui| self.branches(ui, world, selection));

				ui.separator();
				detail(ui, world, selection.at());
				ui.separator();
				ui.label(format!("gizmo: {} - w move, e turn, r size", tool.word()));
				self.filing(ui, world);
			});
	}

	/// The row that writes the world back out as something a person can read.
	///
	/// Through the console rather than by calling into the runner: the editor
	/// is a crate that draws panels and the file is the runner's business, and
	/// a console line is the one way across that already exists and already
	/// works from a script, a config file and a document's own program.
	fn filing(&mut self, ui: &mut egui::Ui, world: &mut World) {
		if self.filed.is_empty() {
			self.filed.push_str("edited");
		}

		ui.horizontal(|ui| {
			ui.label("write to");
			ui.add(
				egui::TextEdit::singleline(&mut self.filed)
					.desired_width(120.0)
					.hint_text("name"),
			);

			if ui.button("assets/scenes").clicked() {
				console::run(world, &format!("scene.write {}", self.filed));
			}
		});
	}

	/// Refills the three lists of what is alive.
	fn gather(&mut self, world: &World) {
		self.entities.clear();
		self.entities
			.extend(world.entities.iter().map(|(id, ..)| id));

		self.bodies.clear();
		self.bodies
			.extend(world.bodies.iter().map(|(id, _)| id));

		self.joints.clear();
		self.joints
			.extend(world.joints.iter().map(|(id, _)| id));
	}

	/// Every entity with its bodies under it, then the rest.
	fn branches(&self, ui: &mut egui::Ui, world: &World, selection: &mut Selection) {
		for index in 0..self.entities.len() {
			let Some(entity) = self.entities.get(index).copied() else {
				continue;
			};

			row(ui, world, selection, Pick::Entity(entity), &entity_label(world, entity));
			self.under(ui, world, selection, entity);
		}

		self.loose(ui, world, selection);
		self.ties(ui, world, selection);
	}

	/// The bodies driving one entity, indented under it.
	fn under(
		&self,
		ui: &mut egui::Ui,
		world: &World,
		selection: &mut Selection,
		entity: EntityId,
	) {
		let driving: Vec<BodyId> = self
			.bodies
			.iter()
			.copied()
			.filter(|&body| drives(world, body) == Some(entity))
			.collect();

		if driving.is_empty() {
			return;
		}

		ui.indent(entity.slot(), |ui| {
			for body in driving {
				row(ui, world, selection, Pick::Body(body), &body_label(world, body));
			}
		});
	}

	/// The bodies nothing living stands on.
	///
	/// The floor is usually one of these: a shape with no entity behind it,
	/// because there is nothing to draw. So is a body whose entity was
	/// despawned without it, which is a bug worth being able to see.
	fn loose(&self, ui: &mut egui::Ui, world: &World, selection: &mut Selection) {
		let alone: Vec<BodyId> = self
			.bodies
			.iter()
			.copied()
			.filter(|&body| drives(world, body).is_none())
			.collect();

		if alone.is_empty() {
			return;
		}

		ui.separator();
		ui.label("bodies on their own");

		for body in alone {
			row(ui, world, selection, Pick::Body(body), &body_label(world, body));
		}
	}

	/// The joints, which hold bodies rather than standing anywhere.
	fn ties(&self, ui: &mut egui::Ui, world: &World, selection: &mut Selection) {
		if self.joints.is_empty() {
			return;
		}

		ui.separator();
		ui.label("joints");

		for index in 0..self.joints.len() {
			let Some(joint) = self.joints.get(index).copied() else {
				continue;
			};

			row(ui, world, selection, Pick::Joint(joint), &joint_label(world, joint));
		}
	}
}

/// One selectable line.
fn row(ui: &mut egui::Ui, world: &World, selection: &mut Selection, pick: Pick, label: &str) {
	if ui
		.selectable_label(selection.is(pick), label)
		.clicked()
	{
		// through the world rather than by assignment, so that what is
		// selected is remembered by name as well as by handle, and can be
		// found again when the world is replaced.
		selection.set(world, pick);
	}
}

/// The selected thing, in detail.
fn detail(ui: &mut egui::Ui, world: &mut World, pick: Pick) {
	match pick {
		| Pick::Nothing => {
			ui.label("nothing selected");
		},
		| Pick::Entity(id) => {
			naming(ui, world, pick);
			hanging(ui, world, id);
			placing(ui, world, pick);
			look(ui, world, id);
		},
		| Pick::Body(id) => {
			naming(ui, world, pick);
			ui.label(
				world
					.bodies
					.get(id)
					.map_or_else(|| "gone".to_owned(), |body| format!("a {}", body_words(body))),
			);
			placing(ui, world, pick);
			solid(ui, world, id);
		},
		| Pick::Joint(id) => {
			naming(ui, world, pick);
			tie(ui, world, id);
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

/// The entity a body drives, if it is still there.
fn drives(world: &World, body: BodyId) -> Option<EntityId> {
	let entity = world.bodies.get(body)?.entity;

	world.entities.alive(entity).then_some(entity)
}

/// What to call an entity in the tree.
fn entity_label(world: &World, id: EntityId) -> String {
	let name = world.entities.name(id);
	if !name.is_empty() {
		return name.to_owned();
	}

	let mesh = world
		.entities
		.renderable(id)
		.map(|renderable| renderable.mesh)
		.and_then(|mesh| world.meshes.get(mesh))
		.map_or("", |entry| entry.name());

	if mesh.is_empty() {
		format!("<entity {}>", id.slot())
	} else {
		format!("<{mesh}>")
	}
}

/// What to call a body in the tree.
fn body_label(world: &World, id: BodyId) -> String {
	let name = world.bodies.name(id);
	if !name.is_empty() {
		return name.to_owned();
	}

	world.bodies.get(id).map_or_else(
		|| format!("<body {}>", id.slot()),
		|body| format!("<{} {}>", body_words(body), id.slot()),
	)
}

/// What to call a joint in the tree.
fn joint_label(world: &World, id: JointId) -> String {
	let name = world.joints.name(id);
	if !name.is_empty() {
		return name.to_owned();
	}

	world.joints.get(id).map_or_else(
		|| format!("<joint {}>", id.slot()),
		|joint| format!("<{} {}>", joint.kind.word(), id.slot()),
	)
}

/// What a body is, in the fewest words that say it.
///
/// The words the format writes a body with, so that a row in this tree and a
/// line in a scene file agree; the tree used to have a vocabulary of its own,
/// and the moment the two had to be the same word was the moment the words
/// went on the kinds themselves.
fn body_words(body: &Body) -> String {
	let kind = body.kind.word();
	let shape = body.shape.kind.word();

	if body.sensor {
		// the first thing anybody wants to know about one, because a sensor is
		// the body that is there and does not push.
		format!("{kind} {shape} sensor")
	} else {
		format!("{kind} {shape}")
	}
}

/// The name field.
fn naming(ui: &mut egui::Ui, world: &mut World, pick: Pick) {
	let mut name = pick.name(world).to_owned();

	ui.horizontal(|ui| {
		ui.label("name");

		if ui.text_edit_singleline(&mut name).changed() {
			select::rename(world, pick, &name);
		}
	});
}

/// What an entity hangs off, which is read rather than written here.
///
/// Hanging one entity off another is the hierarchy panel's job, and that
/// panel is not built yet; until it is, the fact is shown so that the numbers
/// under it read right.
fn hanging(ui: &mut egui::Ui, world: &World, id: EntityId) {
	let parent = world.entities.parent(id);

	if !parent.is_some() {
		return;
	}

	ui.horizontal(|ui| {
		ui.label("inside");
		ui.monospace(entity_label(world, parent));
	});
}

/// Position, rotation and scale, for the things that have them.
///
/// In the thing's own terms - inside its parent, for an entity that hangs off
/// one - because that is what a person expects to type; the gizmo in the
/// viewport works in the world. @ref `select::local`.
fn placing(ui: &mut egui::Ui, world: &mut World, pick: Pick) {
	let Some(transform) = select::local(world, pick) else {
		return;
	};

	let mut edited = transform;

	if inspect(ui, "transform", &mut edited, Transform::FIELDS) {
		select::place_local(world, pick, edited);
	}
}

/// What an entity looks like: the plain half of its renderable, which is the
/// tint. The mesh, the material and the pose are handles, and the tree names
/// the mesh in the row above.
fn look(ui: &mut egui::Ui, world: &mut World, id: EntityId) {
	let Some(renderable) = world.entities.renderable_mut(id) else {
		return;
	};

	inspect(ui, "renderable", renderable, Renderable::FIELDS);
}

/// Everything the solver reads about a body, to edit.
///
/// Its place is the row above, through the transform's own table, and the
/// entity it drives is the branch it hangs under in the tree.
fn solid(ui: &mut egui::Ui, world: &mut World, id: BodyId) {
	let Some(body) = world.bodies.get_mut(id) else {
		return;
	};

	inspect(ui, "body", body, Body::FIELDS);
}

/// What a joint holds, by name, and then everything else about it.
///
/// Its two bodies are handles, which the table describes and cannot name, so
/// the two rows that name them are drawn here; the anchors are in each body's
/// own space, and are numbers all the same.
fn tie(ui: &mut egui::Ui, world: &mut World, id: JointId) {
	let Some(joint) = world.joints.get(id).copied() else {
		return;
	};

	Grid::new("held").num_columns(2).show(ui, |ui| {
		ui.label("first");
		ui.monospace(body_label(world, joint.first));
		ui.end_row();

		ui.label("second");
		ui.monospace(if joint.second.is_some() {
			body_label(world, joint.second)
		} else {
			"a point in the world".to_owned()
		});
		ui.end_row();
	});

	if let Some(held) = world.joints.get_mut(id) {
		inspect(ui, "joint", held, Joint::FIELDS);
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
/// @param salt - what tells this grid from another in the same window
/// @param record - what to show and edit
/// @param fields - its table
/// @return whether any field was written
fn inspect<T>(ui: &mut egui::Ui, salt: &str, record: &mut T, fields: &[Field<T>]) -> bool {
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
/// id no other widget in the window has
/// @param value - what to draw and edit in place
fn widget<T>(ui: &mut egui::Ui, field: &Field<T>, value: &mut Value) {
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
fn vector(ui: &mut egui::Ui, value: &mut Vec3, speed: f32) {
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
fn turn(ui: &mut egui::Ui, rotation: &mut Quat) {
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
fn color(ui: &mut egui::Ui, value: &mut Vec3) {
	let mut rgb = value.to_array();

	if ui.color_edit_button_rgb(&mut rgb).changed() {
		*value = Vec3::from_array(rgb);
	}
}

/// One of a few words, as a drop-down over the field's own list.
fn words(ui: &mut egui::Ui, salt: &str, kind: Kind, held: &mut u32) {
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
	use egui::RawInput;

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
	}

	#[test]
	fn a_body_is_described_in_the_words_a_file_writes_it_with() {
		let body = Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0);

		assert_eq!(body_words(&body), "dynamic sphere");
		assert_eq!(body_words(&body.sensing()), "dynamic sphere sensor");
		assert_eq!(body_words(&Body::default()), "static box");
	}
}
