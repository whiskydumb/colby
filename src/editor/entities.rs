//! The entity table, and one entity at a time in detail.
//!
//! The interesting part is not the list, it is what happens when a number is
//! dragged. This panel writes a transform once a *frame*, and the renderer
//! draws somewhere between the last two *steps* - so the two disagree, and the
//! answer depends on which mode the world is in.
//!
//! - **Editing**: nothing is being simulated, so the host pins the blend to the
//!   present and what is written is what is drawn. The panel does nothing about
//!   it.
//! - **Playing**: the blend is real, so a written transform is drawn sliding
//!   out of wherever the last step left it. `Entities::snap` is what says the
//!   entity was moved rather than that it traveled - the same call gameplay
//!   makes for a teleport, for the same reason.
//!
//! So the snap below is for the second case, and the first case needed no
//! mechanism at all. @ref `colby_core::abi::World::editing`.

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

			// dragged, not traveled. Only play mode blends at all, so this is
			// only about that one - and it costs nothing to say it in both.
			// @ref the module docs.
			world.entities.snap(id);
		}

		Self::tint(ui, world, id);

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
