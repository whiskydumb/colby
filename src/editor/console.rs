//! The console, with somewhere to read it.
//!
//! The system underneath was built a step earlier and reads a terminal; this is
//! the same system with a window in front of it. The only thing it adds is what
//! a terminal already had and an engine did not: **the lines that came back**.
//! `colby_core::log` keeps the last few hundred, including the ones logged
//! inside the game module, because host and module share one dispatcher.

use colby_core::{
	abi::{World, console},
	log::{self, Line},
	tracing::Level,
};
use egui::{Align, Color32, Context, Key, Layout, RichText, ScrollArea, TextEdit, Window};

/// How many lines the input remembers.
const RECALLED: usize = 64;

/// The console window's own state.
#[derive(Debug, Default)]
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
}

impl Console {
	/// Draws the console window.
	///
	/// @param context - egui, mid-frame
	/// @param world - what commands act on
	pub(crate) fn show(&mut self, context: &Context, world: &mut World) {
		self.refresh();

		Window::new("console")
			.default_pos([12.0, 300.0])
			.default_size([660.0, 300.0])
			.show(context, |ui| self.body(ui, world));
	}

	/// The scrollback and the prompt.
	///
	/// Laid out from the bottom, which is what puts the prompt on the floor of
	/// the window and gives the scrollback whatever is left. Measuring the
	/// space and handing it to the scroll area instead makes the two argue: the
	/// area asks for everything available, the window grows to give it, and the
	/// console ends up as tall as the screen.
	fn body(&mut self, ui: &mut egui::Ui, world: &mut World) {
		ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
			self.prompt(ui, world);
			ui.separator();

			ScrollArea::vertical()
				.stick_to_bottom(true)
				.auto_shrink([false, false])
				.show(ui, |ui| self.scrollback(ui));
		});
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
			ui.label(
				RichText::new(format!("{} {}", line.target, line.message))
					.monospace()
					.color(color(line.level)),
			);
		}
	}

	/// The line being typed, and what happens when it is.
	fn prompt(&mut self, ui: &mut egui::Ui, world: &mut World) {
		let response = ui.add(
			TextEdit::singleline(&mut self.input)
				.desired_width(f32::INFINITY)
				.hint_text("try `help`")
				.font(egui::TextStyle::Monospace),
		);

		if response.has_focus() {
			self.recall(ui);
		}

		if !response.lost_focus() || !ui.input(|input| input.key_pressed(Key::Enter)) {
			return;
		}

		let line = std::mem::take(&mut self.input);
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
