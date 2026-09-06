//! The strip along the top of the editor: which mode the world is in and the
//! button that changes it, the two buttons that take a step back and forward
//! again, which of its three things the gizmo is doing, and the row that
//! writes the world back out as text.
//!
//! Nothing here changes anything: a press is handed back as a [`Change`] and
//! applied by the caller, which is what lets the strip be drawn in a test
//! without a window and keeps the world written in one place.

use colby_core::abi::World;
use egui::{Align, Button, Layout, RichText, TextEdit, Ui};

use crate::{Change, KEEP, gizmo::Tool};

/// The strip's own state.
#[derive(Debug, Default)]
pub(crate) struct Bar {
	/// What the world would be called if it were written out now.
	///
	/// Kept here rather than read from anywhere, because there is nowhere to
	/// read it from: a world does not know which file it came from. Naming the
	/// file is part of writing it, exactly as it is at a console.
	filed: String,

	/// Whether somebody has typed in that field since the last write.
	///
	/// What keeps a name they are halfway through typing from being replaced
	/// by the tab's own the moment they move to another scene.
	typed: bool,
}

impl Bar {
	/// Draws the strip.
	///
	/// @param ui - the panel
	/// @param world - for which mode it is in
	/// @param tool - what the gizmo is doing
	/// @param steps - what an undo would undo and what a redo would do again,
	/// for the two buttons
	/// @param changes - where a press is written down
	pub(crate) fn show(
		&mut self,
		ui: &mut Ui,
		world: &World,
		tool: Tool,
		steps: Steps<'_>,
		scene: &str,
		changes: &mut Vec<Change>,
	) {
		// the field follows the tab on screen until somebody types in it, and
		// then it is theirs: what they typed is a save-as they have not
		// pressed yet, and a switch that overwrote it would throw it away
		if !self.typed {
			scene.clone_into(&mut self.filed);
		}

		ui.add_space(4.0);
		strip(ui, world, tool, steps, &mut self.filed, &mut self.typed, changes);
		ui.add_space(4.0);
	}
}

/// What the two step buttons say, borrowed for a frame.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Steps<'a> {
	/// What an undo would undo, or nothing to undo.
	pub(crate) undo: Option<&'a str>,

	/// What a redo would do again, or nothing to redo.
	pub(crate) redo: Option<&'a str>,
}

/// The one row, left to right.
fn strip(
	ui: &mut Ui,
	world: &World,
	tool: Tool,
	steps: Steps<'_>,
	filed: &mut String,
	typed: &mut bool,
	changes: &mut Vec<Change>,
) {
	ui.horizontal(|ui| {
		mode(ui, world, changes);
		ui.separator();
		stepping(ui, steps, changes);
		ui.separator();
		tools(ui, tool, changes);
		ui.separator();
		filing(ui, filed, typed, changes);
		hint(ui);
	});
}

/// Play or stop, and which of the two the world is in.
fn mode(ui: &mut Ui, world: &World, changes: &mut Vec<Change>) {
	if world.editing {
		if ui.button("play").on_hover_text("F5").clicked() {
			changes.push(Change::Edit(false));
		}

		ui.label("editing");
	} else {
		if ui.button("stop").on_hover_text("F5").clicked() {
			changes.push(Change::Edit(true));
		}

		// which of the two a stop is about to do, read fresh: it is a
		// variable and it can be turned while the game is running
		ui.label(if world.cvars.bool(KEEP).unwrap_or(false) {
			"playing; stopping keeps this world"
		} else {
			"playing; stopping puts the world back"
		});
	}
}

/// A step back and a step forward, each saying what it would do.
///
/// Greyed rather than hidden when there is nothing to do, so that the strip
/// keeps its shape.
fn stepping(ui: &mut Ui, steps: Steps<'_>, changes: &mut Vec<Change>) {
	let undo = steps
		.undo
		.map_or_else(|| "undo".to_owned(), |label| format!("undo {label}"));
	if ui
		.add_enabled(steps.undo.is_some(), Button::new(undo))
		.on_hover_text("ctrl+z")
		.clicked()
	{
		changes.push(Change::Undo);
	}

	let redo = steps
		.redo
		.map_or_else(|| "redo".to_owned(), |label| format!("redo {label}"));
	if ui
		.add_enabled(steps.redo.is_some(), Button::new(redo))
		.on_hover_text("ctrl+y")
		.clicked()
	{
		changes.push(Change::Redo);
	}
}

/// The three things the gizmo does, with the key that picks each.
fn tools(ui: &mut Ui, tool: Tool, changes: &mut Vec<Change>) {
	for (candidate, key) in [(Tool::Move, "w"), (Tool::Turn, "e"), (Tool::Size, "r")] {
		if ui
			.selectable_label(tool == candidate, format!("{} ({key})", candidate.word()))
			.clicked()
		{
			changes.push(Change::Tool(candidate));
		}
	}
}

/// The row that writes the world back out as something a person can read.
///
/// Through the console rather than by calling into the runner: the editor is
/// a crate that draws panels and the file is the runner's business, and a
/// console line is the one way across that already exists and already works
/// from a script, a config file and a document's own program.
fn filing(ui: &mut Ui, filed: &mut String, typed: &mut bool, changes: &mut Vec<Change>) {
	ui.label("write to");

	if ui
		.add(
			TextEdit::singleline(filed)
				.desired_width(160.0)
				.hint_text("name"),
		)
		.changed()
	{
		*typed = true;
	}

	if ui.button("assets/scenes").clicked() && !filed.trim().is_empty() {
		changes.push(Change::Write(filed.trim().to_owned()));
		// written, so the field goes back to following the tab - which is
		// about to be the name that was just written to
		*typed = false;
	}
}

/// The two keys, at the far end.
fn hint(ui: &mut Ui) {
	ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
		// a step in from the edge: the strip's layout runs to the window's
		// edge rather than to the panel's margin, and a sentence that ends
		// exactly at the edge reads as cut off.
		ui.add_space(8.0);
		ui.label(RichText::new("F1 hides the editor, F5 plays and stops").weak());
	});
}
