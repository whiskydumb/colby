//! The entity table, and one entity at a time in detail.
//!
//! The interesting part is not the list, it is what happens when a number is
//! dragged: the editor writes the transform and then calls
//! `Entities::snap`, because an entity moved by a person is a teleport and not
//! a journey. Without that the thing being dragged would smear across the gap
//! every frame, which is the same rule gameplay follows and the reason it is a
//! rule rather than a special case.

use colby_core::{
	abi::{EntityId, World},
	glam::Vec3,
};
use egui::{Context, DragValue, Grid, ScrollArea, Window};

/// How tall the list is allowed to get before it scrolls.
const LIST_HEIGHT: f32 = 160.0;

/// The entity window's own state.
#[derive(Debug, Default)]
pub(crate) struct Inspector {
	/// Which entity the detail half is showing.
	selected: EntityId,

	/// The living handles, refilled every frame.
	///
	/// Kept here rather than allocated per frame, and refilled rather than
	/// held: the table is the host's and an entity can go away between frames.
	listed: Vec<EntityId>,
}

impl Inspector {
	/// Draws the entity window.
	///
	/// @param context - egui, mid-frame
	/// @param world - the table to show, and to edit
	pub(crate) fn show(&mut self, context: &Context, world: &mut World) {
		self.listed.clear();
		self.listed
			.extend(world.entities.iter().map(|(id, ..)| id));

		Window::new("entities")
			.default_pos([820.0, 12.0])
			.default_width(300.0)
			.show(context, |ui| {
				ui.label(format!("{} alive", self.listed.len()));

				ScrollArea::vertical()
					.max_height(LIST_HEIGHT)
					.auto_shrink([false, true])
					.show(ui, |ui| self.list(ui, world));

				ui.separator();
				self.detail(ui, world);
			});
	}

	/// Every living entity, one line each.
	fn list(&mut self, ui: &mut egui::Ui, world: &World) {
		for index in 0..self.listed.len() {
			self.row(ui, world, index);
		}
	}

	/// One line of the list.
	fn row(&mut self, ui: &mut egui::Ui, world: &World, index: usize) {
		let Some(id) = self.listed.get(index).copied() else {
			return;
		};

		let mesh = world
			.entities
			.renderable(id)
			.map(|renderable| renderable.mesh)
			.and_then(|mesh| world.meshes.get(mesh))
			.map_or("nothing", |entry| entry.name());

		if ui
			.selectable_label(self.selected == id, format!("{index}  {mesh}"))
			.clicked()
		{
			self.selected = id;
		}
	}

	/// The selected entity, in detail.
	fn detail(&self, ui: &mut egui::Ui, world: &mut World) {
		let id = self.selected;

		let Some(transform) = world.entities.transform(id).copied() else {
			ui.label("nothing selected");

			return;
		};

		let mut edited = transform;

		Grid::new("transform")
			.num_columns(2)
			.show(ui, |ui| {
				ui.label("position");
				vector(ui, &mut edited.position, 0.02);
				ui.end_row();

				ui.label("scale");
				vector(ui, &mut edited.scale, 0.01);
				ui.end_row();
			});

		if edited != transform {
			if let Some(held) = world.entities.transform_mut(id) {
				*held = edited;
			}

			// dragged, not traveled: without this the entity is drawn sliding
			// from where it was to where the mouse put it, every frame the
			// mouse moves. @ref `Entities::snap`.
			world.entities.snap(id);
		}

		Self::tint(ui, world, id);

		ui.separator();
		ui.label("`sim.pause 1` first, or the game writes these back every step");
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
