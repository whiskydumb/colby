//! More than one scene open at once, and which of them the world is.
//!
//! **A tab is a named scene source**, the same name the asset browser lists
//! and the same name `scene.write` writes to. Nothing else is a tab: a save
//! file is not, and neither is a world nobody has named, which is why the one
//! a window opens with carries the name the project said to start from.
//!
//! **One world, several descriptions.** The tab on screen *is* the live
//! [`World`]; every other tab is a [`SceneData`] sitting in this table. Moving
//! between them is [`scene::capture`] of the one being left and
//! [`scene::restore`] of the one being arrived at - the same pair a stopped
//! play uses, and it carries the same obligation, which is why nothing here
//! restores anything: a restore leaves the solver holding caches of a world
//! that is gone, and the solver is out of reach from this crate. So a switch
//! hands back the description to put back and the frame loop does it, exactly
//! as an undo does. @ref [`Frame`](crate::Frame).
//!
//! **A history belongs to a tab**, not to the editor. Undoing in one scene
//! must not reach into another, and a stack that spans two worlds would hand
//! back a description of the wrong one.

use colby_core::abi::{
	World,
	scene::{self, SceneData},
};
use egui::Ui;

use crate::{Change, history::History};

/// One scene open in the editor.
pub(crate) struct Tab {
	/// The asset name of the scene source: `scenes/yard`.
	pub(crate) name: String,

	/// What was done to this scene, and how to undo it.
	pub(crate) history: History,

	/// The world, for a tab that is not the one on screen.
	///
	/// `None` for the current tab, whose world is the live one: a second copy
	/// of a world somebody is editing is a second world to keep in step, and
	/// there is no honest way to do that.
	held: Option<Box<SceneData>>,
}

impl Tab {
	/// A tab with nothing done to it yet.
	fn new(name: &str, held: Option<Box<SceneData>>) -> Self {
		Self {
			name: name.to_owned(),
			history: History::default(),
			held,
		}
	}
}

/// Every scene open, and which one the world is.
pub(crate) struct Tabs {
	/// In the order they were opened, which is the order they are drawn.
	open: Vec<Tab>,

	/// Which of them the live world is. Always a tab that exists.
	current: usize,
}

impl Tabs {
	/// One tab, named, holding whatever world the window came up with.
	///
	/// @param name - the scene the world was built from
	pub(crate) fn new(name: &str) -> Self {
		Self {
			open: vec![Tab::new(name, None)],
			current: 0,
		}
	}

	/// The tab the world is.
	pub(crate) fn current(&self) -> &Tab {
		self.open
			.get(self.current)
			.unwrap_or_else(|| &self.open[0])
	}

	/// What was done to the scene on screen.
	pub(crate) fn history(&mut self) -> &mut History {
		let current = self
			.current
			.min(self.open.len().saturating_sub(1));

		&mut self.open[current].history
	}

	/// Every tab's name, in the order they are drawn.
	pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
		self.open.iter().map(|tab| tab.name.as_str())
	}

	/// Which one the world is.
	pub(crate) const fn at(&self) -> usize { self.current }

	/// How many are open.
	pub(crate) fn len(&self) -> usize { self.open.len() }

	/// Renames the tab the world is, because it was written somewhere else.
	///
	/// A `scene.write` under a new name is a save-as, and what is on screen
	/// afterwards is the scene that was just written rather than the one it
	/// came from.
	///
	/// @param name - what it is called now
	pub(crate) fn rename(&mut self, name: &str) {
		if let Some(tab) = self.open.get_mut(self.current) {
			name.clone_into(&mut tab.name);
		}
	}

	/// Renames whichever tab is open on an asset name, if one is.
	///
	/// Unlike [`rename`](Self::rename), which is about the tab on screen: an
	/// asset renamed in the browser may be a scene open in a tab that is not
	/// the current one, and a tab claiming to be a source that is no longer
	/// there is a tab whose next write makes a second file.
	///
	/// @param was - the scene source's asset name before
	/// @param now - what it is called now
	pub(crate) fn rename_named(&mut self, was: &str, now: &str) {
		for tab in &mut self.open {
			if tab.name == was {
				now.clone_into(&mut tab.name);
			}
		}
	}

	/// Moves to another tab, if it is not the one already on screen.
	///
	/// @param world - the world being left, which is written down
	/// @param to - which tab to move to
	/// @return the world to put back, for the frame loop to restore
	pub(crate) fn switch(&mut self, world: &World, to: usize) -> Option<Box<SceneData>> {
		if to == self.current || to >= self.open.len() {
			return None;
		}

		self.open.get_mut(self.current)?.held = Some(Box::new(scene::capture(world)));
		self.current = to;

		self.open.get_mut(to)?.held.take()
	}

	/// Opens a scene, or moves to it if it is already open.
	///
	/// @param world - the world being left, written down if a move happens
	/// @param name - the scene source's asset name
	/// @param data - the compiled scene, for a tab that is not open yet
	/// @return the world to put back
	pub(crate) fn open(
		&mut self,
		world: &World,
		name: &str,
		data: &SceneData,
	) -> Option<Box<SceneData>> {
		if let Some(index) = self.open.iter().position(|tab| tab.name == name) {
			return self.switch(world, index);
		}

		self.open.get_mut(self.current)?.held = Some(Box::new(scene::capture(world)));
		self.open
			.push(Tab::new(name, Some(Box::new(data.clone()))));
		self.current = self.open.len() - 1;

		self.open.get_mut(self.current)?.held.take()
	}

	/// Closes one, unless it is the last one open.
	///
	/// The last tab stays: a window with no scene in it has nothing to draw
	/// and nothing to write, and there is no state that means it.
	///
	/// The world being closed is not written down: closing a tab is throwing
	/// away what is in it, and a description kept for a tab nobody can reach
	/// is a description nobody can reach either.
	///
	/// @param which - the tab to close
	/// @return the world to put back, when closing the current tab moved to
	/// another
	pub(crate) fn close(&mut self, which: usize) -> Option<Box<SceneData>> {
		if self.open.len() < 2 || which >= self.open.len() {
			return None;
		}

		let closing = which == self.current;
		self.open.remove(which);

		if !closing {
			// the one on screen stays on screen; its place in the row moved
			// if the tab that went was to its left
			if which < self.current {
				self.current -= 1;
			}

			return None;
		}

		// the neighbor to the left, or the first one when there is none
		self.current = which.saturating_sub(1).min(self.open.len() - 1);

		self.open.get_mut(self.current)?.held.take()
	}
}

/// The row of open scenes, one button each, with a cross on the one that has
/// the pointer over it.
///
/// The cross appears on hover rather than sitting on every tab, because a row
/// of crosses reads as a row of things to close rather than a row of scenes;
/// the last tab never shows one, because it will not close.
///
/// @param ui - the strip
/// @param tabs - what is open
/// @param changes - where a click goes
pub(crate) fn strip(ui: &mut Ui, tabs: &Tabs, changes: &mut Vec<Change>) {
	ui.horizontal(|ui| {
		for (which, name) in tabs.names().enumerate() {
			row(ui, which, name, tabs, changes);
		}
	});
}

/// One scene in the row.
fn row(ui: &mut Ui, which: usize, name: &str, tabs: &Tabs, changes: &mut Vec<Change>) {
	let response = ui.selectable_label(which == tabs.at(), name);

	if response.clicked() {
		changes.push(Change::Show { which });
	}

	if tabs.len() > 1 && response.hovered() && ui.small_button("x").clicked() {
		changes.push(Change::Shut { which });
	}
}

#[cfg(test)]
mod tests {
	use colby_core::{abi::Transform, glam::Vec3};

	use super::*;

	/// A world with one thing at a given height, so that a restored world can
	/// be told from another by looking at it.
	fn world_at(height: f32) -> World {
		let mut world = World::new();
		world.editing = true;
		let thing = world
			.entities
			.spawn_at(Transform::at(Vec3::Y * height));
		world.entities.set_name(thing, "marker");

		world
	}

	/// How high the one thing in a description is.
	fn height(data: &SceneData) -> f32 { data.things[0].transform.position.y }

	#[test]
	fn a_window_opens_with_one_tab_named_after_the_scene_it_came_up_with() {
		let tabs = Tabs::new("scenes/yard");

		assert_eq!(tabs.len(), 1);
		assert_eq!(tabs.at(), 0);
		assert_eq!(tabs.current().name, "scenes/yard");
		assert_eq!(tabs.names().collect::<Vec<_>>(), vec!["scenes/yard"]);
	}

	#[test]
	fn opening_a_second_scene_writes_the_first_down_and_hands_back_the_second() {
		let world = world_at(1.0);
		let mut tabs = Tabs::new("scenes/yard");
		let other = scene::capture(&world_at(9.0));

		let put = tabs
			.open(&world, "scenes/hangar", &other)
			.expect("the scene to put back");

		assert_eq!(tabs.len(), 2);
		assert_eq!(tabs.at(), 1);
		assert_eq!(tabs.current().name, "scenes/hangar");
		assert!((height(&put) - 9.0).abs() < 1.0e-5, "the scene that was opened");
	}

	#[test]
	fn coming_back_to_a_tab_puts_back_the_world_it_was_left_in() {
		let mut world = world_at(1.0);
		let mut tabs = Tabs::new("scenes/yard");
		let other = scene::capture(&world_at(9.0));
		drop(tabs.open(&world, "scenes/hangar", &other));

		// the second scene is the world now, and something moves in it
		world = world_at(9.0);
		let thing = world
			.entities
			.iter()
			.next()
			.map(|(id, ..)| id)
			.expect("the marker");
		if let Some(transform) = world.entities.transform_mut(thing) {
			transform.position = Vec3::Y * 4.0;
		}

		let back = tabs
			.switch(&world, 0)
			.expect("the first scene comes back");

		assert!((height(&back) - 1.0).abs() < 1.0e-5, "the yard as it was left");
		assert_eq!(tabs.at(), 0);

		let again = tabs
			.switch(&world_at(1.0), 1)
			.expect("and the second as it was left");

		assert!((height(&again) - 4.0).abs() < 1.0e-5, "with the thing where it was moved to");
	}

	#[test]
	fn opening_a_scene_that_is_already_open_moves_to_it_rather_than_opening_it_twice() {
		let world = world_at(1.0);
		let mut tabs = Tabs::new("scenes/yard");
		let other = scene::capture(&world_at(9.0));
		drop(tabs.open(&world, "scenes/hangar", &other));

		let back = tabs.open(&world_at(9.0), "scenes/yard", &other);

		assert_eq!(tabs.len(), 2, "no third tab");
		assert_eq!(tabs.at(), 0);
		assert!(back.is_some(), "and the world it was left in comes back");
	}

	#[test]
	fn a_history_belongs_to_its_tab() {
		let world = world_at(1.0);
		let mut tabs = Tabs::new("scenes/yard");
		tabs.history().begin("move", &world);
		let moved = world_at(5.0);
		tabs.history().settle(&moved);
		tabs.history().settle(&moved);
		assert_eq!(tabs.history().undoable(), Some("move"));

		let other = scene::capture(&world_at(9.0));
		drop(tabs.open(&moved, "scenes/hangar", &other));

		assert_eq!(tabs.history().undoable(), None, "a fresh scene has nothing to undo");

		drop(tabs.switch(&world_at(9.0), 0));

		assert_eq!(tabs.history().undoable(), Some("move"), "and the first still has its own");
	}

	#[test]
	fn the_last_tab_does_not_close() {
		let mut tabs = Tabs::new("scenes/yard");

		assert!(tabs.close(0).is_none());
		assert_eq!(tabs.len(), 1, "a window with no scene in it has nothing to draw");
	}

	#[test]
	fn closing_the_tab_on_screen_moves_to_its_neighbor() {
		let world = world_at(1.0);
		let mut tabs = Tabs::new("scenes/yard");
		let other = scene::capture(&world_at(9.0));
		drop(tabs.open(&world, "scenes/hangar", &other));

		let back = tabs
			.close(1)
			.expect("the neighbor's world comes back");

		assert_eq!(tabs.len(), 1);
		assert_eq!(tabs.current().name, "scenes/yard");
		assert!((height(&back) - 1.0).abs() < 1.0e-5);
	}

	#[test]
	fn closing_another_tab_leaves_the_one_on_screen_where_it_is() {
		let world = world_at(1.0);
		let mut tabs = Tabs::new("scenes/yard");
		let other = scene::capture(&world_at(9.0));
		drop(tabs.open(&world, "scenes/hangar", &other));

		assert!(tabs.close(0).is_none(), "nothing to put back");
		assert_eq!(tabs.current().name, "scenes/hangar", "still the one that was on screen");
		assert_eq!(tabs.at(), 0, "at its new place in the row");
	}

	#[test]
	fn writing_under_a_new_name_renames_the_tab() {
		let mut tabs = Tabs::new("scenes/yard");

		tabs.rename("scenes/yard_two");

		assert_eq!(tabs.current().name, "scenes/yard_two");
	}
}
