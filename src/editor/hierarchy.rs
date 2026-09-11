//! The world as a tree, nested the way the tables say.
//!
//! An entity under what it hangs off, the bodies driving it under the entity,
//! the bodies driving nothing on their own, and the joints last, because a
//! joint holds bodies rather than standing anywhere. Two kinds of nesting in
//! one tree, then: the older one, a body naming the entity it drives, and the
//! newer one, an entity naming its parent.
//!
//! **A row is dragged onto another to hang it there, and onto the floor of
//! the panel to stand it on its own.** What is hung stays exactly where it is
//! in the world - only what it is measured from changes - which is what every
//! editor checked does on a drop and what a person dragging a wheel under a
//! car means. @ref [`select::hang`].
//!
//! **What a thing is called comes from the world, not from here.** A panel
//! that kept its own names would lose them the moment a scene was loaded.
//! Something nobody has named is shown by what it is made of, in angle
//! brackets - `<cube>` for an entity drawing that mesh, `<dynamic sphere 4>`
//! for a body - so that a name and a description can never be mistaken for
//! each other. @ref [`select::entity_label`].
//!
//! **A click selects, a ctrl-click adds or takes out**, and a right click
//! selects and opens the two things done to a selection: duplicate and
//! delete, which the keys do as well.
//!
//! Nothing here changes the world: a press is handed back as a [`Change`] and
//! applied by the caller, so the panel can be drawn in a test without a
//! window and the world is written in one place.

use colby_core::abi::{BodyId, EntityId, JointId, World};
use egui::{
	Align2, Button, DragAndDrop, Id, LayerId, Order, Response, RichText, ScrollArea, Sense,
	Stroke, StrokeKind, TextStyle, Ui, vec2,
};

use crate::{
	Change,
	select::{self, Pick, Selection},
};

/// How tall the floor of the panel is at least, so that there is always
/// somewhere to drop a row that should stand on its own.
const FLOOR: f32 = 48.0;

/// The panel's own state: the handles it draws, refilled every frame.
///
/// Refilled rather than held: the tables are the host's, and anything in them
/// can go away between one frame and the next.
#[derive(Debug, Default)]
pub(crate) struct Hierarchy {
	/// The living entity handles.
	entities: Vec<EntityId>,

	/// The living body handles.
	bodies: Vec<BodyId>,

	/// The living joint handles.
	joints: Vec<JointId>,

	/// The entities hanging off nothing, in slot order.
	roots: Vec<EntityId>,

	/// The children of each entity, by the parent's slot.
	///
	/// Sized by the table rather than by what is alive, so that a slot is an
	/// index into it directly.
	children: Vec<Vec<EntityId>>,
}

impl Hierarchy {
	/// Draws the tree into a panel.
	///
	/// @param ui - the panel
	/// @param world - the tables to show
	/// @param selection - what is selected, which a row here may change
	/// @param changes - where a press is written down
	pub(crate) fn show(
		&mut self,
		ui: &mut Ui,
		world: &World,
		selection: &Selection,
		changes: &mut Vec<Change>,
	) {
		self.gather(world);

		ui.label(format!(
			"{} entities, {} bodies, {} joints",
			self.entities.len(),
			self.bodies.len(),
			self.joints.len()
		));
		ui.separator();

		ScrollArea::vertical()
			.auto_shrink([false, false])
			.show(ui, |ui| {
				self.branches(ui, world, selection, changes);
				floor(ui, changes);
			});
	}

	/// Refills the lists of what is alive, and which entity hangs off which.
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

		self.roots.clear();
		self.children.iter_mut().for_each(Vec::clear);
		self.children
			.resize_with(world.entities.slots(), Vec::new);

		for index in 0..self.entities.len() {
			let id = self.entities[index];

			self.file(world, id);
		}
	}

	/// Puts one entity under its parent, or among the roots.
	fn file(&mut self, world: &World, id: EntityId) {
		let parent = world.entities.parent(id);

		match self.children.get_mut(parent.slot()) {
			| Some(under) if parent.is_some() => under.push(id),
			| _ => self.roots.push(id),
		}
	}

	/// Every root with everything under it, then the rest.
	fn branches(
		&self,
		ui: &mut Ui,
		world: &World,
		selection: &Selection,
		changes: &mut Vec<Change>,
	) {
		for &root in &self.roots {
			self.branch(ui, world, selection, changes, root);
		}

		self.loose(ui, world, selection, changes);
		self.ties(ui, world, selection, changes);
	}

	/// One entity, and everything under it: its bodies, then its children
	/// with theirs.
	///
	/// Recursive, and bounded by the table: a loop cannot be made, @ref
	/// `Entities::set_parent`, so the walk ends where the chain does.
	fn branch(
		&self,
		ui: &mut Ui,
		world: &World,
		selection: &Selection,
		changes: &mut Vec<Change>,
		id: EntityId,
	) {
		entity_row(ui, world, selection, changes, id);

		let driving = self.driving(world, id);
		let children = self
			.children
			.get(id.slot())
			.map_or(&[][..], Vec::as_slice);

		if driving.is_empty() && children.is_empty() {
			return;
		}

		ui.indent(id.slot(), |ui| {
			for body in driving {
				row(ui, selection, changes, Pick::Body(body), &select::body_label(world, body));
			}

			for &child in children {
				self.branch(ui, world, selection, changes, child);
			}
		});
	}

	/// The bodies driving one entity.
	fn driving(&self, world: &World, entity: EntityId) -> Vec<BodyId> {
		self.bodies
			.iter()
			.copied()
			.filter(|&body| select::drives(world, body) == Some(entity))
			.collect()
	}

	/// The bodies nothing living stands on.
	///
	/// The floor is usually one of these: a shape with no entity behind it,
	/// because there is nothing to draw. So is a body whose entity was
	/// despawned without it, which is a bug worth being able to see.
	fn loose(
		&self,
		ui: &mut Ui,
		world: &World,
		selection: &Selection,
		changes: &mut Vec<Change>,
	) {
		let alone: Vec<BodyId> = self
			.bodies
			.iter()
			.copied()
			.filter(|&body| select::drives(world, body).is_none())
			.collect();

		if alone.is_empty() {
			return;
		}

		ui.separator();
		ui.label("bodies on their own");

		for body in alone {
			row(ui, selection, changes, Pick::Body(body), &select::body_label(world, body));
		}
	}

	/// The joints, which hold bodies rather than standing anywhere.
	fn ties(&self, ui: &mut Ui, world: &World, selection: &Selection, changes: &mut Vec<Change>) {
		if self.joints.is_empty() {
			return;
		}

		ui.separator();
		ui.label("joints");

		for &joint in &self.joints {
			row(ui, selection, changes, Pick::Joint(joint), &select::joint_label(world, joint));
		}
	}
}

/// One entity's line: a row that can be dragged onto another, and dropped
/// on.
fn entity_row(
	ui: &mut Ui,
	world: &World,
	selection: &Selection,
	changes: &mut Vec<Change>,
	id: EntityId,
) {
	let pick = Pick::Entity(id);
	let label = select::entity_label(world, id);
	// dim for a thing that is not drawn, whichever of the two hides it
	let text = if world.entities.shown(id) {
		RichText::new(label.as_str())
	} else {
		RichText::new(label.as_str()).weak()
	};

	let response = ui
		.horizontal(|ui| {
			eye(ui, world, changes, id);

			// one widget that senses both a click and a drag, rather than a
			// drag source wrapped around a clickable row: egui hands a click
			// to the topmost widget that senses one, and a wrapper that senses
			// only drags laid over a row that senses only clicks is a row
			// nobody can click.
			ui.add(Button::selectable(selection.is(pick), text).sense(Sense::click_and_drag()))
		})
		.inner;

	// the payload is the handle: what is hung is looked up again when it
	// lands, so a row that went away mid-drag hangs nothing.
	response.dnd_set_drag_payload(id);

	acted(ui, &response, selection, changes, pick);

	if response.dragged() {
		ghost(ui, &label);
	}

	// what is held over this row, if it is another row: a row cannot be
	// hung off itself, and saying so with a highlight would promise a drop
	// that refuses.
	let held = response
		.dnd_hover_payload::<EntityId>()
		.filter(|held| **held != id);

	if held.is_some() {
		ui.painter().rect_stroke(
			response.rect,
			2.0,
			Stroke::new(1.5, ui.visuals().selection.stroke.color),
			StrokeKind::Outside,
		);
	}

	if let Some(child) = response
		.dnd_release_payload::<EntityId>()
		.filter(|held| **held != id)
	{
		changes.push(Change::Hang { child: *child, parent: id });
	}
}

/// What the eye in front of a row is drawn with.
///
/// The eye emoji, which egui's default fonts carry, written as an escape so
/// that the source stays ASCII.
const EYE: &str = "\u{1f441}";

/// The eye in front of an entity's row, and what pressing it asks for.
///
/// Plain for a thing that is drawn, dim for one that is hidden, and struck
/// through as well where the thing is hidden by its own word rather than by
/// something it hangs off. Pressing it flips the entity's own word and nothing
/// else: a child under something hidden is shown by showing what it hangs off,
/// and a child hidden by its own word stays hidden when its parent is shown.
///
/// @param ui - where to draw it
/// @param world - whose words are read
/// @param changes - where a press is written down
/// @param id - the entity
fn eye(ui: &mut Ui, world: &World, changes: &mut Vec<Change>, id: EntityId) {
	let own = world.entities.hidden(id);
	let glyph = match (own, world.entities.shown(id)) {
		| (true, _) => RichText::new(EYE).weak().strikethrough(),
		| (false, false) => RichText::new(EYE).weak(),
		| (false, true) => RichText::new(EYE),
	};
	let hint = if own {
		"show it, and what hangs off it"
	} else {
		"hide it, and what hangs off it"
	};

	if ui
		.add(Button::new(glyph).frame(false))
		.on_hover_text(hint)
		.clicked()
	{
		changes.push(Change::Hide { entity: id, hidden: !own });
	}
}

/// The name of the row being dragged, beside the pointer, so that what is
/// being dragged is seen to move.
fn ghost(ui: &Ui, label: &str) {
	let Some(pointer) = ui.ctx().pointer_interact_pos() else {
		return;
	};

	ui.ctx()
		.layer_painter(LayerId::new(Order::Tooltip, Id::new("hierarchy drag")))
		.text(
			pointer + vec2(12.0, 12.0),
			Align2::LEFT_TOP,
			label,
			TextStyle::Body.resolve(ui.style()),
			ui.visuals().strong_text_color(),
		);
}

/// One selectable line that nothing is dropped on.
fn row(ui: &mut Ui, selection: &Selection, changes: &mut Vec<Change>, pick: Pick, label: &str) {
	let response = ui.selectable_label(selection.is(pick), label);

	acted(ui, &response, selection, changes, pick);
}

/// What a press on any row comes to: a click selects, a ctrl-click adds or
/// takes out, a right click selects what was not selected and opens the
/// menu over the selection.
fn acted(
	ui: &Ui,
	response: &Response,
	selection: &Selection,
	changes: &mut Vec<Change>,
	pick: Pick,
) {
	if response.clicked() {
		changes.push(if ui.input(|input| input.modifiers.command) {
			Change::Toggle(pick)
		} else {
			Change::Select(pick)
		});
	}

	if response.secondary_clicked() && !selection.is(pick) {
		changes.push(Change::Select(pick));
	}

	response.context_menu(|ui| menu(ui, changes));
}

/// The two things done to a selection, for the people who do not know the
/// keys yet.
fn menu(ui: &mut Ui, changes: &mut Vec<Change>) {
	if ui.button("duplicate  ctrl+d").clicked() {
		changes.push(Change::Duplicate);
		ui.close();
	}

	if ui.button("delete  del").clicked() {
		changes.push(Change::Delete);
		ui.close();
	}
}

/// The rest of the panel: where a row is dropped to stand on its own.
///
/// At least [`FLOOR`] tall, so that a panel full of rows still has one.
fn floor(ui: &mut Ui, changes: &mut Vec<Change>) {
	let mut rect = ui.available_rect_before_wrap();
	rect.set_height(rect.height().max(FLOOR));

	let response = ui.allocate_rect(rect, Sense::hover());

	if DragAndDrop::has_payload_of_type::<EntityId>(ui.ctx()) {
		ui.painter().text(
			rect.center(),
			Align2::CENTER_CENTER,
			"drop here to stand it on its own",
			TextStyle::Small.resolve(ui.style()),
			ui.visuals().weak_text_color(),
		);
	}

	if let Some(child) = response.dnd_release_payload::<EntityId>() {
		changes.push(Change::Hang { child: *child, parent: EntityId::NONE });
	}
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{Body, Shape, Transform},
		glam::Vec3,
	};
	use egui::{Context, Modifiers, PointerButton, Pos2, RawInput, Rect};

	use super::*;

	/// A car with a wheel hanging off it, a body under the car, a floor on
	/// its own and a rope: one of everything the tree nests.
	fn yard() -> World {
		let mut world = World::new();

		let car = world.entities.spawn_at(Transform::at(Vec3::X));
		world.entities.set_name(car, "car");
		let wheel = world.entities.spawn_at(Transform::at(Vec3::Y));
		world.entities.set_name(wheel, "wheel");
		assert!(world.entities.set_parent(wheel, car));

		world
			.bodies
			.spawn(Body::dynamic(Shape::ball(0.5), Transform::at(Vec3::X), 1.0).driving(car));
		world.bodies.spawn(Body::new(
			colby_core::abi::BodyKind::Static,
			Shape::UNIT,
			Transform::IDENTITY,
		));

		world
	}

	#[test]
	fn a_child_is_filed_under_its_parent_and_a_root_among_the_roots() {
		let world = yard();
		let mut hierarchy = Hierarchy::default();

		hierarchy.gather(&world);

		assert_eq!(hierarchy.roots.len(), 1, "the car stands on its own");
		let car = hierarchy.roots[0];
		assert_eq!(world.entities.name(car), "car");
		assert_eq!(hierarchy.children[car.slot()].len(), 1, "and the wheel hangs off it");
		assert_eq!(
			world
				.entities
				.name(hierarchy.children[car.slot()][0]),
			"wheel"
		);
		assert_eq!(hierarchy.bodies.len(), 2);
		assert_eq!(hierarchy.driving(&world, car).len(), 1, "one body drives the car");
	}

	#[test]
	fn a_child_whose_parent_went_away_is_a_root_again() {
		let mut world = yard();
		let car = world
			.entities
			.iter()
			.map(|(id, ..)| id)
			.find(|&id| world.entities.name(id) == "car")
			.expect("the car is there");
		world.entities.despawn(car);

		let mut hierarchy = Hierarchy::default();
		hierarchy.gather(&world);

		assert_eq!(hierarchy.roots.len(), 1, "the wheel is on its own now");
		assert_eq!(world.entities.name(hierarchy.roots[0]), "wheel");
	}

	#[test]
	fn the_tree_draws_without_a_window_and_presses_nothing() {
		let world = yard();
		let mut hierarchy = Hierarchy::default();
		let selection = Selection::default();
		let mut changes = Vec::new();
		let context = Context::default();

		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 600.0))),
				..Default::default()
			},
			|ui| hierarchy.show(ui, &world, &selection, &mut changes),
		);
		// nothing paints this frame, and epaint asserts that a texture delta
		// is applied rather than dropped; cleared on purpose, the way the
		// shell does on its way out.
		output.textures_delta.clear();

		assert!(changes.is_empty(), "nobody pressed anything");
		assert_eq!(hierarchy.entities.len(), 2, "and the frame gathered the world");
	}

	/// The entity of the yard called something.
	fn named(world: &World, name: &str) -> EntityId {
		world
			.entities
			.iter()
			.map(|(id, ..)| id)
			.find(|&id| world.entities.name(id) == name)
			.expect("the yard has one")
	}

	/// One press and one release, which is a click.
	fn clicked(at: Pos2) -> Vec<egui::Event> {
		let mut events = vec![egui::Event::PointerMoved(at)];

		for pressed in [true, false] {
			events.push(egui::Event::PointerButton {
				pos: at,
				button: PointerButton::Primary,
				pressed,
				modifiers: Modifiers::NONE,
			});
		}

		events
	}

	/// One eye drawn on its own, and what pressing it asked for.
	///
	/// Twice over one context, the way a browser row is tested: egui answers
	/// where a widget is only after it has drawn, so one frame finds it and
	/// the next presses there. Once a frame whatever egui asks, because a
	/// context may run the closure twice and an eye drawn twice would answer
	/// a click twice.
	fn eyed(
		context: &Context,
		world: &World,
		id: EntityId,
		events: Vec<egui::Event>,
	) -> (Vec<Change>, Rect) {
		let mut changes = Vec::new();
		let mut drawn = Rect::NOTHING;
		let mut once = false;

		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(200.0, 40.0))),
				events,
				..Default::default()
			},
			|ui| {
				if !once {
					once = true;
					eye(ui, world, &mut changes, id);
					drawn = ui.min_rect();
				}
			},
		);
		output.textures_delta.clear();

		(changes, drawn)
	}

	#[test]
	fn the_eye_hides_a_thing_that_is_drawn_and_shows_one_that_is_hidden() {
		let mut world = yard();
		let car = named(&world, "car");
		let context = Context::default();
		let (_, drawn) = eyed(&context, &world, car, Vec::new());

		let (changes, _) = eyed(&context, &world, car, clicked(drawn.center()));

		assert_eq!(changes, vec![Change::Hide { entity: car, hidden: true }]);

		assert!(world.entities.set_hidden(car, true));

		let (changes, _) = eyed(&context, &world, car, clicked(drawn.center()));

		assert_eq!(
			changes,
			vec![Change::Hide { entity: car, hidden: false }],
			"and pressed again it shows it"
		);
	}

	#[test]
	fn the_eye_of_a_thing_its_parent_hides_sets_its_own_word() {
		// the wheel is not drawn because the car is hidden, and its own word
		// says it is not hidden: its eye flips that word and leaves the car's
		// alone, so what it asks for is to be hidden itself
		let mut world = yard();
		let car = named(&world, "car");
		let wheel = named(&world, "wheel");
		assert!(world.entities.set_hidden(car, true));
		let context = Context::default();
		let (_, drawn) = eyed(&context, &world, wheel, Vec::new());

		let (changes, _) = eyed(&context, &world, wheel, clicked(drawn.center()));

		assert_eq!(changes, vec![Change::Hide { entity: wheel, hidden: true }]);
	}
}
