//! Every console variable there is, as something to turn rather than to type.
//!
//! The table is the model and this panel invents nothing: it keeps a search
//! box and no other state. A row reads an entry and writes through
//! [`Cvars::set`](colby_core::abi::cvar::Cvars::set), the same call a typed
//! line makes, so a number dragged here and a number typed at the prompt land
//! by one path and are kept by one rule - what was registered as saved is
//! written to the config when the engine stops, and everything else lasts
//! this run.
//!
//! Commands are not here. There is nothing to turn on a command, and `help`
//! at the prompt is the list of them.
//!
//! Grouped by what the name says before its first dot, because that is the
//! only grouping the table has: nothing declares a section, and a module or a
//! program that registers `mine.speed` gets a `mine` group without asking
//! anyone.

use colby_core::abi::{World, cvar::Value};
use egui::{CollapsingHeader, DragValue, Grid, RichText, ScrollArea, TextEdit, Ui};

/// The group a name with no dot in it goes under.
const LOOSE: &str = "general";

/// How wide the box for a text value is, in points.
const TEXT: f32 = 180.0;

/// How fast a number moves under the pointer, per pixel.
const SPEED: f64 = 0.05;

/// One variable, copied out of the table for the frame that draws it.
///
/// Copied rather than borrowed because drawing a row writes to the table it
/// would have been borrowed from. Tens of small values a frame is nothing,
/// and it is what makes @ref `groups` a plain function over a world.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Row {
	/// What it is registered under.
	pub(crate) name: String,

	/// The line describing it.
	pub(crate) help: String,

	/// What it holds now.
	pub(crate) value: Value,

	/// What the code registered it with.
	pub(crate) default: Value,

	/// Whether it is written to the config file.
	pub(crate) archived: bool,
}

/// The settings panel's own state.
#[derive(Debug, Default)]
pub(crate) struct Settings {
	/// What is in the search box.
	search: String,
}

impl Settings {
	/// Draws the settings into a panel.
	///
	/// @param ui - the panel
	/// @param world - whose table, and what the widgets write to
	pub(crate) fn show(&mut self, ui: &mut Ui, world: &mut World) {
		let groups = groups(world, &self.search);
		let count: usize = groups.iter().map(|(_, rows)| rows.len()).sum();

		ui.horizontal(|ui| {
			ui.label(format!("{count} settings"));
			ui.add(
				TextEdit::singleline(&mut self.search)
					.desired_width(160.0)
					.hint_text("search"),
			);
			ui.label(RichText::new("saved ones are kept between runs").weak());
		});
		ui.separator();

		ScrollArea::vertical()
			.auto_shrink([false, false])
			.show(ui, |ui| list(ui, world, &groups));
	}
}

/// Every group, one after another.
fn list(ui: &mut Ui, world: &mut World, groups: &[(String, Vec<Row>)]) {
	for (name, rows) in groups {
		group(ui, world, name, rows);
	}
}

/// One group, open until somebody shuts it.
fn group(ui: &mut Ui, world: &mut World, name: &str, rows: &[Row]) {
	CollapsingHeader::new(name)
		.default_open(true)
		.show(ui, |ui| grid(ui, world, name, rows));
}

/// The rows of one group, in columns that line up.
fn grid(ui: &mut Ui, world: &mut World, name: &str, rows: &[Row]) {
	Grid::new(format!("settings {name}"))
		.num_columns(4)
		.striped(true)
		.show(ui, |ui| {
			for row in rows {
				line(ui, world, row);
			}
		});
}

/// One variable: its name, something to turn it with, a way back to the
/// default, and whether it outlives the run.
fn line(ui: &mut Ui, world: &mut World, row: &Row) {
	ui.label(&row.name).on_hover_text(&row.help);

	if let Some(text) = widget(ui, row) {
		// the same call the console makes, and the same refusal. What a
		// refusal would mean here is a number the widget wandered out of
		// range - `set_text` will not take one that is not finite - and the
		// answer to it is the one the panel gives anyway: the next frame
		// reads the table again and the widget snaps back to what is in it.
		world.cvars.set(&row.name, &text);
	}

	if row.value == row.default {
		// the column is kept even when there is nothing in it, so that the
		// last column does not walk left and right as values are turned
		ui.label("");
	} else if ui
		.small_button("reset")
		.on_hover_text(format!("back to {}", row.default.quoted()))
		.clicked()
	{
		world.cvars.reset(&row.name);
	}

	ui.label(if row.archived {
		RichText::new("saved").weak()
	} else {
		RichText::new("")
	})
	.on_hover_text(if row.archived {
		"written to the config when the engine stops"
	} else {
		"for this run only"
	});
	ui.end_row();
}

/// Something to turn this value with, and the text to write if it moved.
///
/// A copy is turned and the text of it handed back, rather than the table
/// being written through: the widget wants a `&mut` to the value and the
/// write wants the name, and going through the text is what the console does
/// anyway. Every kind's text round-trips exactly, floats included.
///
/// @param ui - the row
/// @param row - the variable
/// @return what to set it to, or `None` if nobody moved it this frame
fn widget(ui: &mut Ui, row: &Row) -> Option<String> {
	let mut value = row.value.clone();

	let moved = match &mut value {
		| Value::Bool(held) => ui.checkbox(held, "").changed(),
		| Value::Int(held) => ui.add(DragValue::new(held)).changed(),
		| Value::Float(held) => ui
			.add(DragValue::new(held).speed(SPEED))
			.changed(),
		| Value::Text(held) => ui
			.add(TextEdit::singleline(held).desired_width(TEXT))
			.changed(),
	};

	moved.then(|| value.text())
}

/// The group a name belongs to: what it says before its first dot.
///
/// @param name - the variable's name
pub(crate) fn group_of(name: &str) -> &str {
	match name.split_once('.') {
		| Some((prefix, _)) if !prefix.is_empty() => prefix,
		| _ => LOOSE,
	}
}

/// Every variable in the table, by group, whose name contains the filter.
///
/// Groups come out in name order, because a list that moves when a module
/// registers something is a list nobody can point at; the rows inside a group
/// stay in the order the table holds them, which is the order they were
/// registered and the order `help` lists them.
///
/// @param world - whose table
/// @param filter - what a name has to contain, case ignored; empty for all
pub(crate) fn groups(world: &World, filter: &str) -> Vec<(String, Vec<Row>)> {
	let filter = filter.trim().to_lowercase();
	let mut groups: Vec<(String, Vec<Row>)> = Vec::new();

	for entry in world.cvars.iter() {
		let (Some(value), Some(default)) = (entry.value(), entry.default_value()) else {
			// a command: nothing to turn
			continue;
		};

		if !filter.is_empty() && !entry.name().to_lowercase().contains(&filter) {
			continue;
		}

		let row = Row {
			name: entry.name().to_owned(),
			help: entry.help().to_owned(),
			value: value.clone(),
			default: default.clone(),
			archived: entry.is_archived(),
		};

		match groups
			.iter_mut()
			.find(|(name, _)| name == group_of(&row.name))
		{
			| Some((_, rows)) => rows.push(row),
			| None => groups.push((group_of(&row.name).to_owned(), vec![row])),
		}
	}

	groups.sort_by(|left, right| left.0.cmp(&right.0));

	groups
}

#[cfg(test)]
mod tests {
	use egui::{Context, Pos2, RawInput, Rect, vec2};

	use super::*;

	/// A world with a few variables and a command in it.
	fn stocked() -> World {
		let mut world = World::new();
		world
			.cvars
			.var("r.shadows", Value::Bool(true), "draw shadows");
		world
			.cvars
			.saved("snd.volume", Value::Float(0.8), "how loud everything is");
		world
			.cvars
			.var("r.shadow_distance", Value::Float(60.0), "how far shadows reach");
		world
			.cvars
			.var("loose", Value::Int(3), "a name with no dot in it");
		world
			.cvars
			.command("quit", colby_core::abi::console::defer, "stop");

		world
	}

	#[test]
	fn a_name_is_grouped_by_what_it_says_before_its_first_dot() {
		assert_eq!(group_of("r.shadows"), "r");
		assert_eq!(group_of("snd.volume"), "snd");
		assert_eq!(group_of("a.b.c"), "a", "the first dot, not the last");
		assert_eq!(group_of("loose"), LOOSE, "a name with no dot is not a group of one");
		assert_eq!(group_of(".hidden"), LOOSE, "and neither is a name that starts with one");
	}

	#[test]
	fn every_variable_is_listed_under_its_group_and_no_command_is() {
		let world = stocked();

		let groups = groups(&world, "");

		let names: Vec<&str> = groups
			.iter()
			.map(|(name, _)| name.as_str())
			.collect();
		assert_eq!(names, vec![LOOSE, "r", "snd"], "groups in name order");
		let shading: Vec<&str> = groups[1]
			.1
			.iter()
			.map(|row| row.name.as_str())
			.collect();
		assert_eq!(
			shading,
			vec!["r.shadows", "r.shadow_distance"],
			"and rows in the order they were registered"
		);
		assert!(
			groups
				.iter()
				.flat_map(|(_, rows)| rows)
				.all(|row| row.name != "quit"),
			"a command has nothing to turn"
		);
	}

	#[test]
	fn a_row_carries_what_it_holds_what_it_started_as_and_whether_it_is_kept() {
		let mut world = stocked();
		assert!(world.cvars.set("snd.volume", "0.25"), "turned down");

		let groups = groups(&world, "volume");

		assert_eq!(groups.len(), 1, "only the one group has a match");
		let row = &groups[0].1[0];
		assert_eq!(row.name, "snd.volume");
		assert_eq!(row.value, Value::Float(0.25), "what it holds");
		assert_eq!(row.default, Value::Float(0.8), "and what it would go back to");
		assert!(row.archived, "registered as saved");
		assert_eq!(row.help, "how loud everything is");
	}

	#[test]
	fn the_search_matches_a_name_anywhere_in_it_whatever_its_case() {
		let world = stocked();

		assert_eq!(groups(&world, "SHADOW").len(), 1, "case is ignored");
		assert_eq!(groups(&world, "shadow")[0].1.len(), 2, "and it matches inside a name");
		assert!(groups(&world, "nothing here").is_empty());
	}

	#[test]
	fn the_panel_draws_without_a_window_and_writes_nothing_by_itself() {
		let mut world = stocked();
		let mut settings = Settings::default();
		let context = Context::default();

		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(700.0, 400.0))),
				..Default::default()
			},
			|ui| settings.show(ui, &mut world),
		);
		output.textures_delta.clear();

		assert_eq!(world.cvars.bool("r.shadows"), Some(true), "drawing turns nothing");
		assert_eq!(world.cvars.float("snd.volume"), Some(0.8));
	}
}
