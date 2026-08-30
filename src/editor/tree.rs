//! The world as a tree, and one thing in it at a time in detail.
//!
//! **There is no hierarchy in colby, so this is not one.** An entity has no
//! parent - the model importer flattened node trees on purpose, and nothing
//! composes transforms. What the tree groups by instead is what the tables
//! already say: a body names the entity it drives, and a joint names two
//! bodies. So an entity's bodies hang under it, the bodies driving nothing
//! stand on their own, and the joints are a group of their own at the bottom.
//! That is a real relationship rather than an invented one, and when parenting
//! arrives it will be a second kind of nesting rather than a replacement for
//! this.
//!
//! **What a thing is called comes from the world, not from here.** A panel that
//! kept its own names would lose them the moment a scene was loaded. Something
//! nobody has named is shown by what it is made of, in angle brackets -
//! `<cube>` for an entity drawing that mesh, `<dynamic ball 4>` for a body - so
//! that a name and a description can never be mistaken for each other.
//!
//! Everything that can be tested lives in [`select`](crate::select); what is
//! here is the drawing, and it is checked by looking at it.

use colby_core::{
	abi::{Body, BodyId, BodyKind, EntityId, JointId, JointKind, ShapeKind, World},
	glam::{EulerRot, Quat, Vec3},
};
use egui::{Context, DragValue, Grid, ScrollArea, Window};

use crate::{
	gizmo::Tool,
	select::{self, Pick, Selection},
};

/// How tall the tree is allowed to get before it scrolls.
const TREE_HEIGHT: f32 = 240.0;

/// How far a drag of one pixel moves a position.
const MOVE_SPEED: f32 = 0.02;

/// How far a drag of one pixel turns something, in degrees.
const TURN_SPEED: f32 = 0.5;

/// How far a drag of one pixel scales something.
const SCALE_SPEED: f32 = 0.01;

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
			placing(ui, world, pick);
			tint(ui, world, id);
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
		|joint| format!("<{} {}>", joint_word(joint.kind), id.slot()),
	)
}

/// What a body is, in the fewest words that say it.
///
/// The editor's own vocabulary rather than the scene format's: these are read
/// in a list, and the format's words are written to a file. When something
/// needs both to be the same word, that is the moment to put them somewhere
/// they can be shared.
fn body_words(body: &Body) -> String {
	let kind = match body.kind {
		| BodyKind::Static => "static",
		| BodyKind::Kinematic => "kinematic",
		| BodyKind::Dynamic => "dynamic",
	};
	let shape = match body.shape.kind {
		| ShapeKind::Box => "box",
		| ShapeKind::Sphere => "ball",
		| ShapeKind::Mesh => "mesh",
	};

	if body.sensor {
		// the first thing anybody wants to know about one, because a sensor is
		// the body that is there and does not push.
		format!("{kind} {shape} sensor")
	} else {
		format!("{kind} {shape}")
	}
}

/// What a joint does, in one word.
const fn joint_word(kind: JointKind) -> &'static str {
	match kind {
		| JointKind::Rope => "rope",
		| JointKind::Weld => "weld",
		| JointKind::Axis => "hinge",
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

/// Position, rotation and scale, for the things that have them.
fn placing(ui: &mut egui::Ui, world: &mut World, pick: Pick) {
	let Some(transform) = select::transform(world, pick) else {
		return;
	};

	let mut edited = transform;

	Grid::new("transform")
		.num_columns(2)
		.show(ui, |ui| {
			ui.label("position");
			vector(ui, &mut edited.position, MOVE_SPEED);
			ui.end_row();

			ui.label("rotation");
			turn(ui, &mut edited.rotation);
			ui.end_row();

			ui.label("scale");
			vector(ui, &mut edited.scale, SCALE_SPEED);
			ui.end_row();
		});

	if edited != transform {
		select::place(world, pick, edited);
	}
}

/// What the solver is doing with a body, which is read rather than written.
fn solid(ui: &mut egui::Ui, world: &World, id: BodyId) {
	let Some(body) = world.bodies.get(id) else {
		return;
	};

	Grid::new("body").num_columns(2).show(ui, |ui| {
		ui.label("mass");
		ui.monospace(format!("{:.2}", body.mass));
		ui.end_row();

		ui.label("speed");
		ui.monospace(format!("{:.2}", body.velocity.length()));
		ui.end_row();

		ui.label("asleep");
		ui.monospace(if body.sleeping { "yes" } else { "no" });
		ui.end_row();
	});
}

/// What a joint holds, which is read rather than written.
///
/// Its anchors are in each body's own space, so there is nothing to type here
/// that would mean anything without something drawn in the world to type it
/// against.
fn tie(ui: &mut egui::Ui, world: &World, id: JointId) {
	let Some(joint) = world.joints.get(id) else {
		return;
	};

	Grid::new("joint").num_columns(2).show(ui, |ui| {
		ui.label("kind");
		ui.monospace(joint_word(joint.kind));
		ui.end_row();

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

		if joint.kind == JointKind::Rope {
			ui.label("length");
			ui.monospace(format!("{:.2}", joint.length));
			ui.end_row();
		}
	});
}

/// The selected entity's own color.
fn tint(ui: &mut egui::Ui, world: &mut World, id: EntityId) {
	let Some(renderable) = world.entities.renderable(id).copied() else {
		return;
	};

	let mut color = renderable.color.to_array();

	ui.horizontal(|ui| {
		ui.label("tint");

		if ui.color_edit_button_rgb(&mut color).changed()
			&& let Some(held) = world.entities.renderable_mut(id)
		{
			held.color = Vec3::from_array(color);
		}
	});
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
