//! What the loop is doing, in numbers.
//!
//! Everything here is read from the clock and the world; the panel keeps no
//! state of its own, which is why it is a function rather than a type.

use colby_core::{abi::World, time::Clock};
use egui::{Grid, ScrollArea, Ui};

use crate::KEEP;

/// Draws the statistics into a panel.
///
/// @param ui - the panel
/// @param world - the state to count
/// @param clock - the pacing to report
/// @param frames - how many frames have been drawn
pub(crate) fn show(ui: &mut Ui, world: &World, clock: &Clock, frames: u64) {
	ScrollArea::vertical()
		.auto_shrink([false, false])
		.show(ui, |ui| body(ui, world, clock, frames));
}

/// The numbers.
fn body(ui: &mut Ui, world: &World, clock: &Clock, frames: u64) {
	// egui's own smoothed frame time rather than a mean kept here: it is
	// measured over the same frames this is drawn in, and one number nobody
	// has to maintain is worth more than a better one.
	let seconds = ui.input(|input| input.stable_dt);

	ui.label(format!(
		"{:.0} fps, {:.1} ms a frame",
		1.0 / seconds.max(1.0e-6),
		seconds * 1000.0
	));
	// which mode, said first and said plainly: everything else here is a
	// number that means something different depending on it, starting with
	// the two that stop moving.
	// what a stop does is a variable away from being the opposite, so the
	// line says which of the two it is rather than promising one of them
	let keeping = world.cvars.bool(KEEP).unwrap_or(false);
	ui.label(match (world.editing, keeping) {
		| (true, _) => "editing. F5 plays",
		| (false, false) => "playing. F5 stops, and comes back to the world it started from",
		| (false, true) => "playing. F5 stops and keeps this world, because sim.keep is on",
	});
	ui.separator();

	Grid::new("numbers")
		.num_columns(2)
		.show(ui, |ui| numbers(ui, world, clock, frames));

	if clock.speed() <= f32::EPSILON {
		ui.separator();
		ui.label("paused. `sim.step 1` advances one step");
	}
}

/// One row per number.
fn numbers(ui: &mut Ui, world: &World, clock: &Clock, frames: u64) {
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
	// the refused count is the only place the voice table's bound is visible
	// at all: past it a sound is not played and nothing else says so, which
	// is a silence with no cause anybody could find.
	row(ui, "voices", &bounded(world.audio.len(), world.audio.dropped(), "refused"));
	// and the same for the debug table, which is where the shape came from:
	// past its bound, lines stop being taken and the picture is quietly short
	// of whatever was asked for last.
	row(
		ui,
		"debug lines",
		&bounded(world.debug.lines().len(), world.debug.dropped(), "dropped"),
	);
	row(ui, "reloads", &world.reloads.to_string());
}

/// How many there are, and how many there would have been.
///
/// The two bounded tables in the world report themselves this way, and the
/// second one is the reason this is a function: a panel is the only place
/// either bound is visible at all, and two rows spelling out the same shape
/// invites the third one to spell it differently.
///
/// @param held - how many the table is holding
/// @param refused - how many it turned away since it was last asked
/// @param what - the word for what happened to those, in the past tense
/// @return the count on its own while nothing was refused, which is almost
/// always, and the count with the loss beside it when something was
fn bounded(held: usize, refused: u32, what: &str) -> String {
	match refused {
		| 0 => held.to_string(),
		| lost => format!("{held} ({lost} {what})"),
	}
}

/// One name and one value.
fn row(ui: &mut Ui, name: &str, value: &str) {
	ui.label(name);
	ui.monospace(value);
	ui.end_row();
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_table_that_refused_nothing_reports_only_what_it_holds() {
		assert_eq!(bounded(0, 0, "refused"), "0");
		assert_eq!(bounded(7, 0, "dropped"), "7");
	}

	#[test]
	fn a_table_that_refused_something_says_so_and_says_what() {
		// the whole reason either number is on screen: a bound that is being
		// hit and is invisible is a bug that looks like the engine being
		// wrong about something else.
		assert_eq!(bounded(64, 6, "refused"), "64 (6 refused)");
		assert_eq!(bounded(65_536, 12, "dropped"), "65536 (12 dropped)");
	}
}
