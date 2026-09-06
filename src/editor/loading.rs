//! The loading screen: what the world is doing while it comes up.
//!
//! The editor's first frames. A project opened from the launcher compiles its
//! assets and builds its game crate before there is anything to show, and
//! those are seconds; a window that shows nothing for seconds looks like a
//! window that died. So the window comes up first, and every stage of the
//! stand-up is a line on it - done, running, still to come - fed by the same
//! stages the runtime runs, which is the shape the field settled on: the
//! splash belongs to the editor's project load, and its text comes from the
//! step mechanism the load already has.
//!
//! When a stage fails the screen stays, with the failure in red and the one
//! thing to do about it, because a window that closes on its own leaves a
//! person who opened it from a list with nothing to read.

use colby_engine::Overlay;
use egui::{CentralPanel, RichText, Ui};
use wgpu::{Device, Queue, TextureFormat, TextureView};
use winit::{event::WindowEvent, window::Window};

use crate::shell::Shell;

/// Where one stage stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
	/// Finished.
	Done,

	/// Running now.
	Current,

	/// Still to come.
	Pending,
}

/// One stage, as the screen shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step<'a> {
	/// What the stage does, as a few words.
	pub title: &'a str,

	/// Where it stands.
	pub state: State,
}

/// The loading screen: egui against a window, and the page over it.
pub struct Loading {
	shell: Shell,
	name: String,
}

impl Loading {
	/// Brings the screen up against the window and the device it draws with.
	///
	/// @param window - the window events come from
	/// @param device - the device the frames belong to
	/// @param format - the color format the surface was configured with
	/// @param name - what the project is called
	#[must_use]
	pub fn new(window: &Window, device: &Device, format: TextureFormat, name: &str) -> Self {
		Self {
			shell: Shell::new(window, device, format),
			name: name.to_owned(),
		}
	}

	/// Offers one window event to the screen.
	///
	/// @param window - the window the event came from
	/// @param event - the event
	/// @return whether egui took it
	pub fn on_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
		self.shell.on_event(window, event)
	}

	/// Builds this frame's page.
	///
	/// @param window - the window, for input and for the cursor
	/// @param steps - every stage, in order, with where each stands
	/// @param failure - why the world stopped coming up, if it did
	pub fn run(&mut self, window: &Window, steps: &[Step<'_>], failure: Option<&str>) {
		let name = self.name.as_str();

		self.shell
			.run(window, |ui| page(ui, name, steps, failure));
	}
}

impl Overlay for Loading {
	fn draw(
		&mut self,
		device: &Device,
		queue: &Queue,
		target: &TextureView,
		width: u32,
		height: u32,
	) {
		self.shell
			.draw(device, queue, target, width, height);
	}
}

/// The page: the project's name, a line per stage, and the failure if there
/// is one.
///
/// @param ui - the whole window, egui's root layout for this frame
/// @param name - what the project is called
/// @param steps - every stage, in order, with where each stands
/// @param failure - why the world stopped coming up, if it did
pub(crate) fn page(ui: &mut Ui, name: &str, steps: &[Step<'_>], failure: Option<&str>) {
	CentralPanel::default().show(ui, |ui| {
		ui.add_space(ui.available_height() * 0.3);
		ui.vertical_centered(|ui| {
			ui.heading(name);
			ui.add_space(12.0);

			for step in steps {
				ui.label(line(step));
			}

			if let Some(why) = failure {
				ui.add_space(12.0);
				ui.label(RichText::new(why).color(ui.visuals().error_fg_color));
				ui.label(RichText::new("close the window").weak());
			}
		});
	});
}

/// One stage as one line: a mark, and the title in the weight its state has.
fn line(step: &Step<'_>) -> RichText {
	match step.state {
		| State::Done => RichText::new(format!("+  {}", step.title)).weak(),
		| State::Current => RichText::new(format!(">  {} ...", step.title)).strong(),
		| State::Pending => RichText::new(format!("   {}", step.title)).weak(),
	}
}

#[cfg(test)]
mod tests {
	use egui::{Context, RawInput};

	use super::*;

	#[test]
	fn the_page_draws_every_state_and_a_failure_without_a_window() {
		let steps = [
			Step {
				title: "compiling the assets",
				state: State::Done,
			},
			Step {
				title: "building the game crate",
				state: State::Current,
			},
			Step {
				title: "loading the game module",
				state: State::Pending,
			},
		];
		let context = Context::default();

		let output = context.run_ui(RawInput::default(), |ui| {
			page(ui, "Yard", &steps, Some("building the game crate failed"));
		});

		// epaint refuses to drop a delta nobody applied, and nothing here
		// paints; cleared on purpose, the way the shell does on its way out.
		let mut textures = output.textures_delta;
		textures.clear();

		assert_eq!(line(&steps[0]).text(), "+  compiling the assets");
		assert_eq!(line(&steps[1]).text(), ">  building the game crate ...");
		assert_eq!(line(&steps[2]).text(), "   loading the game module");
	}
}
