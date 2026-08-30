//! What the loop is doing, in numbers.
//!
//! Everything here is read from the clock and the world; the panel keeps no
//! state of its own, which is why it is a function rather than a type.

use colby_core::{abi::World, time::Clock};
use egui::{Context, Grid, Window};

/// Draws the statistics window.
///
/// @param context - egui, mid-frame
/// @param world - the state to count
/// @param clock - the pacing to report
/// @param frames - how many frames have been drawn
pub(crate) fn show(context: &Context, world: &World, clock: &Clock, frames: u64) {
	Window::new("statistics")
		.default_pos([12.0, 12.0])
		.resizable(false)
		.show(context, |ui| {
			// egui's own smoothed frame time rather than a mean kept here: it
			// is measured over the same frames this is drawn in, and one number
			// nobody has to maintain is worth more than a better one.
			let seconds = context.input(|input| input.stable_dt);

			ui.label(format!(
				"{:.0} fps, {:.1} ms a frame",
				1.0 / seconds.max(1.0e-6),
				seconds * 1000.0
			));
			// which mode, said first and said plainly: everything else in this
			// window is a number that means something different depending on
			// it, starting with the two that stop moving.
			ui.label(if world.editing {
				"editing. F5 plays, and stopping puts this world back"
			} else {
				"playing. F5 stops, and comes back to the world it started from"
			});
			ui.separator();

			Grid::new("numbers")
				.num_columns(2)
				.show(ui, |ui| {
					row(ui, "frames drawn", &frames.to_string());
					row(ui, "steps simulated", &world.steps.to_string());
					row(ui, "simulated time", &format!("{:.1} s", world.time));
					row(ui, "into this step", &format!("{:.0}%", world.interpolation() * 100.0));
					row(ui, "speed", &format!("{:.2}x", clock.speed()));
					row(ui, "stalls", &clock.stalls().to_string());
					row(ui, "entities", &world.entities.len().to_string());
					row(
						ui,
						"bodies",
						&format!(
							"{} ({} asleep)",
							world.bodies.len(),
							world
								.bodies
								.iter()
								.filter(|(_, body)| body.sleeping)
								.count()
						),
					);
					row(ui, "contacts", &world.contacts.to_string());
					// the dropped count is the only place the debug table's
					// bound is visible at all: past it, lines stop being taken
					// and the picture is quietly short of whatever was asked
					// for last.
					row(ui, "debug lines", &match world.debug.dropped() {
						| 0 => world.debug.lines().len().to_string(),
						| lost => format!("{} ({lost} dropped)", world.debug.lines().len()),
					});
					row(ui, "reloads", &world.reloads.to_string());
				});

			if clock.speed() <= f32::EPSILON {
				ui.separator();
				ui.label("paused. `sim.step 1` advances one step");
			}
		});
}

/// One name and one value.
fn row(ui: &mut egui::Ui, name: &str, value: &str) {
	ui.label(name);
	ui.monospace(value);
	ui.end_row();
}
