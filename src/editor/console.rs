//! The console, with somewhere to read it.
//!
//! The system underneath was built a step earlier and reads a terminal; this is
//! the same system with a window in front of it. The first thing it adds is
//! what a terminal already had and an engine did not: **the lines that came
//! back**. `colby_core::log` keeps the last few hundred, including the ones
//! logged inside the game module, because host and module share one
//! dispatcher.
//!
//! The other two are what a terminal has by way of its shell rather than by
//! itself. A **filter** on the level and the target, which hides lines for the
//! frame it is drawing and throws none of them away, so that turning it off
//! brings everything back. And **completion** from the table itself, on Tab:
//! the table is the only list of names there is, and a name half-remembered is
//! the usual reason for typing `help` and reading all of them.

use std::fmt::Write as _;

use colby_core::{
	abi::{World, console, cvar::Entry},
	log::{self, Line},
	tracing::Level,
};
use egui::{Align, Color32, Id, Key, Layout, Modifiers, RichText, ScrollArea, TextEdit};

/// How many lines the input remembers.
const RECALLED: usize = 64;

/// How many names a Tab press lists before it stops naming them.
const NAMED: usize = 12;

/// Every level, loudest first: the order they are offered in, and the order
/// @ref `rank` counts them in.
const LEVELS: [Level; 5] = [Level::ERROR, Level::WARN, Level::INFO, Level::DEBUG, Level::TRACE];

/// The console window's own state.
#[derive(Debug)]
pub(crate) struct Console {
	/// A copy of the log, refreshed only when there is something new in it.
	lines: Vec<Line>,

	/// How many lines had been logged when that copy was taken.
	logged: u64,

	/// What is being typed.
	input: String,

	/// What has been typed before, oldest first.
	history: Vec<String>,

	/// How far back through it the up arrow has walked.
	recalled: Option<usize>,

	/// The quietest level shown. Everything louder than it is shown too, so
	/// this is a floor and not a choice of one level.
	level: Level,

	/// What a line's target has to contain to be shown, case ignored.
	target: String,

	/// The names the last Tab press could not choose between.
	suggested: Vec<String>,
}

impl Default for Console {
	fn default() -> Self {
		Self {
			lines: Vec::new(),
			logged: 0,
			input: String::new(),
			history: Vec::new(),
			recalled: None,
			// everything, because a console that hides a line by default is a
			// console that lies about what happened
			level: Level::TRACE,
			target: String::new(),
			suggested: Vec::new(),
		}
	}
}

impl Console {
	/// Draws the console into a panel.
	///
	/// @param ui - the panel
	/// @param world - what commands act on
	pub(crate) fn show(&mut self, ui: &mut egui::Ui, world: &mut World) {
		self.refresh();
		self.body(ui, world);
	}

	/// The scrollback and the prompt.
	///
	/// Laid out from the bottom, which is what puts the prompt on the floor of
	/// the window and gives the scrollback whatever is left. Measuring the
	/// space and handing it to the scroll area instead makes the two argue: the
	/// area asks for everything available, the window grows to give it, and the
	/// console ends up as tall as the screen.
	fn body(&mut self, ui: &mut egui::Ui, world: &mut World) {
		self.filters(ui);
		ui.separator();

		ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
			self.prompt(ui, world);
			self.suggestions(ui);
			ui.separator();

			ScrollArea::vertical()
				.stick_to_bottom(true)
				.auto_shrink([false, false])
				.show(ui, |ui| self.scrollback(ui));
		});
	}

	/// What is shown and what is not: a floor for the level, and a word the
	/// target has to contain.
	///
	/// Nothing is thrown away by either of them. The lines are all still
	/// there, @ref `refresh` takes the whole log, and turning a filter off
	/// brings them back; what a filter hides this frame is hidden this frame
	/// only.
	fn filters(&mut self, ui: &mut egui::Ui) {
		ui.horizontal(|ui| {
			ui.label("show");

			for level in LEVELS {
				self.level = floor(ui, self.level, level);
			}

			ui.separator();
			ui.add(
				TextEdit::singleline(&mut self.target)
					.desired_width(140.0)
					.hint_text("from"),
			)
			.on_hover_text("only lines whose target contains this");

			let hidden = self.hidden();

			if hidden > 0 {
				ui.label(RichText::new(format!("{hidden} hidden")).weak());
			}
		});
	}

	/// How many of the lines in hand the filters are keeping back.
	fn hidden(&self) -> usize {
		self.lines
			.iter()
			.filter(|line| !shown(line, self.level, &self.target))
			.count()
	}

	/// The names the last Tab press found, when it found more than one.
	fn suggestions(&self, ui: &mut egui::Ui) {
		if self.suggested.is_empty() {
			return;
		}

		let listed: Vec<&str> = self
			.suggested
			.iter()
			.take(NAMED)
			.map(String::as_str)
			.collect();
		let more = self.suggested.len().saturating_sub(listed.len());
		let mut text = listed.join("  ");

		if more > 0 {
			write!(text, "  and {more} more").expect("a string takes what is written to it");
		}

		ui.label(RichText::new(text).monospace().weak());
	}

	/// The lines, back the right way up.
	///
	/// The window is laid out bottom-up; without turning it around again inside
	/// the scroll area the oldest line would be at the bottom.
	fn scrollback(&self, ui: &mut egui::Ui) {
		ui.with_layout(Layout::top_down(Align::LEFT), |ui| self.written(ui));
	}

	/// Everything logged so far, oldest at the top.
	fn written(&self, ui: &mut egui::Ui) {
		for line in &self.lines {
			if !shown(line, self.level, &self.target) {
				continue;
			}

			ui.label(
				RichText::new(format!("{} {}", line.target, line.message))
					.monospace()
					.color(color(line.level)),
			);
		}
	}

	/// The line being typed, and what happens when it is.
	fn prompt(&mut self, ui: &mut egui::Ui, world: &mut World) {
		// the field's own id, so that whether it has the keyboard is known
		// before it is drawn rather than after: a Tab has to be taken out of
		// the queue ahead of the field, or the field puts a tab character in
		// the line, and ahead of egui's own focus, which `lock_focus` holds
		// where it is
		let id = Id::new("console prompt");
		let completing = ui.memory(|memory| memory.has_focus(id))
			&& ui.input_mut(|input| input.consume_key(Modifiers::NONE, Key::Tab));

		let response = ui.add(
			TextEdit::singleline(&mut self.input)
				.id(id)
				.lock_focus(true)
				.desired_width(f32::INFINITY)
				.hint_text("try `help`, or Tab to finish a name")
				.font(egui::TextStyle::Monospace),
		);

		if completing {
			self.complete(world);
		} else if response.changed() {
			// what was suggested was suggested about what was typed then
			self.suggested.clear();
		}

		if response.has_focus() {
			self.recall(ui);
		}

		if !response.lost_focus() || !ui.input(|input| input.key_pressed(Key::Enter)) {
			return;
		}

		let line = std::mem::take(&mut self.input);
		self.suggested.clear();

		if !line.trim().is_empty() {
			self.remember(line.clone());
			console::run(world, &line);
		}

		// the cursor stays where it was, because typing one command is almost
		// always followed by typing another.
		response.request_focus();
	}

	/// Walks back and forward through what has been typed.
	fn recall(&mut self, ui: &egui::Ui) {
		let (back, forward) = ui
			.input(|input| (input.key_pressed(Key::ArrowUp), input.key_pressed(Key::ArrowDown)));

		if back {
			let next = match self.recalled {
				| Some(0) | None if self.history.is_empty() => return,
				| Some(index) => index.saturating_sub(1),
				| None => self.history.len() - 1,
			};

			self.recalled = Some(next);
			self.input = self.history[next].clone();
		} else if forward {
			let Some(index) = self.recalled else {
				return;
			};

			if index + 1 < self.history.len() {
				self.recalled = Some(index + 1);
				self.input = self.history[index + 1].clone();
			} else {
				self.recalled = None;
				self.input.clear();
			}
		}
	}

	/// Finishes the name being typed, as far as the table can.
	///
	/// The first word only. What follows a name is its value, and the table
	/// has nothing to say about that; a line with a space in it is already
	/// past the part this knows.
	///
	/// One match is written out with a space after it. Several are written as
	/// far as they agree - which is what makes two Tabs useful, since the
	/// second one has more to work with - and named under the prompt. None
	/// leaves the line alone: a name nobody registered is worth seeing.
	fn complete(&mut self, world: &World) {
		if self.input.contains(char::is_whitespace) {
			return;
		}

		let found = completions(world, &self.input);

		match found.len() {
			| 0 => self.suggested.clear(),
			| 1 => {
				self.input.clone_from(&found[0]);
				self.input.push(' ');
				self.suggested.clear();
			},
			| _ => {
				self.input = common(&found);
				self.suggested = found;
			},
		}
	}

	/// Keeps a line for the up arrow, without keeping it twice in a row.
	fn remember(&mut self, line: String) {
		self.recalled = None;

		if self.history.last() == Some(&line) {
			return;
		}

		if self.history.len() >= RECALLED {
			self.history.remove(0);
		}

		self.history.push(line);
	}

	/// Takes a fresh copy of the log, if anything has been logged since the
	/// last one.
	fn refresh(&mut self) {
		let logged = log::logged();

		if logged == self.logged {
			return;
		}

		self.logged = logged;
		log::copy_lines(&mut self.lines);
	}
}

/// One level's button: the floor as it stands after it has been offered.
///
/// @param ui - the filter row
/// @param floor - the level shown down to now
/// @param level - the one this button sets
/// @return what the floor is after this button has had its say
fn floor(ui: &mut egui::Ui, floor: Level, level: Level) -> Level {
	if ui
		.selectable_label(floor == level, word(level))
		.on_hover_text("and everything louder")
		.clicked()
	{
		return level;
	}

	floor
}

/// Whether a line passes the filters as they stand.
///
/// @param line - the line
/// @param level - the quietest level shown
/// @param target - what the target has to contain, or empty for all of them
fn shown(line: &Line, level: Level, target: &str) -> bool {
	if rank(line.level) > rank(level) {
		return false;
	}

	let target = target.trim().to_lowercase();

	target.is_empty() || line.target.to_lowercase().contains(&target)
}

/// How loud a level is, counting from the loudest.
///
/// Written out rather than taken from the ordering `tracing` gives a `Level`,
/// so that a change there cannot quietly turn the filter upside down and hide
/// exactly the lines somebody is looking for.
const fn rank(level: Level) -> u8 {
	match level {
		| Level::ERROR => 0,
		| Level::WARN => 1,
		| Level::INFO => 2,
		| Level::DEBUG => 3,
		| Level::TRACE => 4,
	}
}

/// What to call a level in a button.
const fn word(level: Level) -> &'static str {
	match level {
		| Level::ERROR => "errors",
		| Level::WARN => "warnings",
		| Level::INFO => "info",
		| Level::DEBUG => "debug",
		| Level::TRACE => "everything",
	}
}

/// Every name in the table that begins with what has been typed.
///
/// Commands as well as variables: both are typed at the same prompt, and a
/// completion that knew only half the table would be worse than none.
///
/// @param world - whose table
/// @param typed - the word so far; nothing at all completes to nothing,
/// because the whole table is not a suggestion
fn completions(world: &World, typed: &str) -> Vec<String> {
	if typed.is_empty() {
		return Vec::new();
	}

	world
		.cvars
		.iter()
		.map(Entry::name)
		.filter(|name| name.starts_with(typed))
		.map(str::to_owned)
		.collect()
}

/// The longest text every one of them begins with.
fn common(names: &[String]) -> String {
	let Some(first) = names.first() else {
		return String::new();
	};

	let end = names[1..]
		.iter()
		.fold(first.chars().count(), |end, name| end.min(shared(first, name)));

	first.chars().take(end).collect()
}

/// How many characters two names share from the start.
fn shared(left: &str, right: &str) -> usize {
	left.chars()
		.zip(right.chars())
		.take_while(|(left, right)| left == right)
		.count()
}

/// What each level looks like.
///
/// Chosen against egui's dark theme rather than picked for prettiness: an error
/// has to be findable in a wall of grey.
fn color(level: Level) -> Color32 {
	match level {
		| Level::ERROR => Color32::from_rgb(240, 105, 105),
		| Level::WARN => Color32::from_rgb(240, 190, 100),
		| Level::INFO => Color32::from_rgb(210, 210, 215),
		| Level::DEBUG => Color32::from_rgb(140, 165, 200),
		| Level::TRACE => Color32::from_rgb(130, 130, 135),
	}
}

#[cfg(test)]
mod tests {
	use colby_core::abi::cvar::Value;
	use egui::{Context, Pos2, RawInput, Rect, vec2};

	use super::*;

	/// One logged line.
	fn said(level: Level, target: &str, message: &str) -> Line {
		Line {
			level,
			target: target.to_owned(),
			message: message.to_owned(),
		}
	}

	/// A world with a handful of names that share their beginnings.
	fn named() -> World {
		let mut world = World::new();
		world
			.cvars
			.var("r.shadows", Value::Bool(true), "draw shadows");
		world
			.cvars
			.var("r.shadow_distance", Value::Float(60.0), "how far");
		world
			.cvars
			.var("r.backend", Value::Text("auto".to_owned()), "which one");
		world
			.cvars
			.command("screenshot", console::defer, "take one");

		world
	}

	/// One headless frame of the console in a panel of this size, and where
	/// the prompt landed in it.
	fn framed(console: &mut Console, world: &mut World, size: egui::Vec2) -> (Rect, Rect) {
		let context = Context::default();
		let mut built = false;
		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, size)),
				..Default::default()
			},
			|ui| {
				if !built {
					built = true;
					console.show(ui, world);
				}
			},
		);
		output.textures_delta.clear();

		let prompt = context
			.read_response(Id::new("console prompt"))
			.expect("the prompt was drawn")
			.rect;

		(Rect::from_min_size(Pos2::ZERO, size), prompt)
	}

	#[test]
	fn the_prompt_stays_on_the_floor_of_the_panel_however_short_it_is() {
		let mut world = World::new();

		// the bottom panel is about this, and the filter row above the
		// scrollback has to come out of the scrollback's share rather than
		// out of the prompt's
		for height in [190.0, 120.0, 90.0] {
			let mut console = Console::default();
			let (panel, prompt) = framed(&mut console, &mut world, vec2(720.0, height));

			assert!(
				prompt.max.y <= panel.max.y,
				"a panel {height} tall put the prompt at {prompt:?}, below its floor {panel:?}"
			);
			assert!(prompt.min.y >= panel.min.y, "and not above its ceiling either");
		}
	}

	#[test]
	fn a_level_is_a_floor_and_everything_louder_than_it_is_shown_too() {
		let warning = said(Level::WARN, "colby_engine", "a warning");
		let trace = said(Level::TRACE, "colby_engine", "a trace");

		assert!(shown(&warning, Level::WARN, ""), "the level chosen is shown");
		assert!(shown(&warning, Level::TRACE, ""), "and so is it under a quieter floor");
		assert!(!shown(&trace, Level::WARN, ""), "a quieter line is not");
		assert!(shown(&trace, Level::TRACE, ""));
	}

	#[test]
	fn the_levels_are_counted_from_the_loudest_whatever_tracing_thinks() {
		assert_eq!(rank(Level::ERROR), 0);
		assert!(rank(Level::ERROR) < rank(Level::WARN));
		assert!(rank(Level::WARN) < rank(Level::INFO));
		assert!(rank(Level::INFO) < rank(Level::DEBUG));
		assert!(rank(Level::DEBUG) < rank(Level::TRACE));
	}

	#[test]
	fn a_target_filter_matches_part_of_a_target_whatever_its_case() {
		let line = said(Level::INFO, "colby_editor", "something");

		assert!(shown(&line, Level::TRACE, "editor"), "part of it is enough");
		assert!(shown(&line, Level::TRACE, "EDITOR"), "case is ignored");
		assert!(shown(&line, Level::TRACE, "  "), "and so is a box with nothing in it");
		assert!(!shown(&line, Level::TRACE, "physics"));
	}

	#[test]
	fn a_name_is_completed_from_the_whole_table_commands_and_all() {
		let world = named();

		assert_eq!(completions(&world, "screen"), vec!["screenshot"], "a command counts");
		assert_eq!(
			completions(&world, "r.shadow"),
			vec!["r.shadows", "r.shadow_distance"],
			"in the order the table holds them"
		);
		assert!(completions(&world, "").is_empty(), "the whole table is not a suggestion");
		assert!(completions(&world, "nothing").is_empty());
	}

	#[test]
	fn several_matches_are_written_out_as_far_as_they_agree_and_then_named() {
		let mut console = Console::default();
		let world = named();
		console.input = "r.".to_owned();

		console.complete(&world);

		assert_eq!(console.input, "r.", "three names agree on no more than that");
		assert_eq!(console.suggested.len(), 3, "and all three are named");

		console.input = "r.s".to_owned();
		console.complete(&world);

		assert_eq!(console.input, "r.shadow", "two agree on this much");
		assert_eq!(console.suggested.len(), 2);
	}

	#[test]
	fn one_match_is_written_out_whole_with_a_space_after_it() {
		let mut console = Console::default();
		let world = named();
		console.input = "r.b".to_owned();

		console.complete(&world);

		assert_eq!(console.input, "r.backend ", "ready for its value");
		assert!(console.suggested.is_empty(), "nothing left to choose between");
	}

	#[test]
	fn a_line_that_is_already_past_its_name_is_left_alone() {
		let mut console = Console::default();
		let world = named();
		console.input = "r.backend vulk".to_owned();

		console.complete(&world);

		assert_eq!(console.input, "r.backend vulk", "the table knows nothing about a value");
	}

	#[test]
	fn a_name_nobody_registered_is_left_where_it_was_to_be_seen() {
		let mut console = Console::default();
		let world = named();
		console.input = "r.shadowz".to_owned();

		console.complete(&world);

		assert_eq!(console.input, "r.shadowz");
		assert!(console.suggested.is_empty());
	}

	#[test]
	fn what_they_all_begin_with_stops_on_a_character_boundary() {
		let names = vec!["snd.volume".to_owned(), "snd.music".to_owned()];

		assert_eq!(common(&names), "snd.");
		assert_eq!(common(&["alone".to_owned()]), "alone", "one name is all of it");
		assert_eq!(common(&[]), "", "and none is nothing");
		// two names that share their first character but not its bytes
		let wide = vec!["\u{e9}clair".to_owned(), "\u{e9}pee".to_owned()];
		assert_eq!(common(&wide), "\u{e9}", "a whole character, not half of one");
	}
}
