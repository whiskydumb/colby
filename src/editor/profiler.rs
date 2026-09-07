//! What a frame costs, one row per part of it.
//!
//! **The view, and nothing else.** Every number here was measured by the
//! runner and averaged before it arrived - @ref `Host::profile` - because both
//! halves of a frame's cost live where this crate cannot reach: the hardware
//! spans are on the scene's own apparatus and the simulation's are on the
//! solver. What is here is a grid.
//!
//! **The tab is the gate.** Measuring a frame costs a query set, two buffers
//! and a readback, so nobody pays for it unless this pane is up. That is what
//! every engine read for `PERF-1` does - a bool in Godot and Wicked, a plugin
//! in bevy, a stat command in Unreal - and it is why this pane's being on
//! screen is handed back to the runner rather than kept here.
//!
//! **The numbers arrive late and that is not a fault.** A frame's hardware
//! spans are read two or three frames after the frame, because a window that
//! waited on the queue would be measuring a stall. Averaged over twenty of
//! them, which is Wicked's window, the lateness is invisible; the pane says so
//! anyway, because a person comparing this against `just profile` should know
//! why the two differ in the third decimal.

use egui::{Grid, RichText, ScrollArea, Ui};

use crate::{Part, Profile};

/// Draws the profiler into a panel.
///
/// @param ui - the panel
/// @param profile - what the runner measured, already averaged
pub(crate) fn show(ui: &mut Ui, profile: Profile<'_>) {
	ScrollArea::vertical()
		.auto_shrink([false, false])
		.show(ui, |ui| body(ui, profile));
}

/// The rows.
fn body(ui: &mut Ui, profile: Profile<'_>) {
	if profile.parts.is_empty() {
		ui.label("nothing has been measured yet; the first numbers are a few frames away");

		return;
	}

	ui.horizontal(|ui| {
		match profile.passes {
			| Some(passes) => ui.label(format!("{passes} render passes")),
			| None => ui.label("no frame has been read back yet"),
		}
		.on_hover_text(
			"the one number here that does not move between two runs of the same scene, which \
			 is what makes it the part worth comparing",
		);

		if !profile.hardware {
			ui.label(
				RichText::new("no timestamp queries on this adapter, so the gpu rows are empty")
					.weak(),
			);
		}
	});
	ui.separator();

	Grid::new("parts")
		.num_columns(3)
		.striped(true)
		.show(ui, |ui| {
			ui.label(RichText::new("part").strong());
			ui.label(RichText::new("mean").strong());
			ui.label(RichText::new("worst").strong());
			ui.end_row();

			for part in profile.parts {
				row(ui, *part);
			}
		});

	ui.separator();
	ui.label(
		RichText::new(
			"the mean of the last twenty frames it was measured in. The gpu rows are read back \
			 two or three frames late, because a window that waited on the queue would be \
			 timing a stall",
		)
		.weak(),
	);
}

/// One part of the frame.
fn row(ui: &mut Ui, part: Part) {
	ui.monospace(part.name);

	// **a part that never ran is not a part that was free.** A glow chain in a
	// project that does not bloom has no number, and printing nought for it
	// would read as the cheapest thing in the frame.
	match (part.mean, part.worst) {
		| (Some(mean), Some(worst)) => {
			ui.monospace(micros(mean));
			ui.monospace(micros(worst));
		},
		| _ => {
			ui.label(RichText::new("-").weak());
			ui.label(RichText::new("never ran").weak());
		},
	}

	ui.end_row();
}

/// A length of time in microseconds, which is the scale every row of this is
/// at: a whole frame at sixty is sixteen thousand of them.
fn micros(took: std::time::Duration) -> String { format!("{} us", took.as_micros()) }

#[cfg(test)]
mod tests {
	use std::time::Duration;

	use egui::{Context, RawInput};

	use super::*;

	/// One headless frame of the pane.
	///
	/// The pane borrows what it draws and writes nothing anywhere, so what a
	/// frame of it can be asked is whether it draws at all - which is worth
	/// asking, because egui panics on a grid whose rows disagree about how
	/// many columns they have, and a row that took an early return would.
	fn drawn(profile: Profile<'_>) {
		let context = Context::default();
		let mut output = context.run_ui(RawInput::default(), |ui| show(ui, profile));

		output.textures_delta.clear();
	}

	#[test]
	fn a_pane_with_nothing_measured_yet_draws_and_says_so() { drawn(Profile::default()); }

	#[test]
	fn a_pane_draws_a_part_that_ran_beside_one_that_never_did() {
		// the same grid, one row with two numbers in it and one with two
		// words: egui counts columns per row and a mismatch is a panic
		let held = [
			Part {
				name: "gpu scene",
				mean: Some(Duration::from_micros(420)),
				worst: Some(Duration::from_micros(900)),
			},
			Part {
				name: "gpu glow",
				mean: None,
				worst: None,
			},
		];

		drawn(Profile {
			parts: &held,
			passes: Some(15),
			hardware: true,
		});
		drawn(Profile {
			parts: &held,
			passes: None,
			hardware: false,
		});
	}

	#[test]
	fn a_length_is_printed_in_microseconds_and_nought_is_a_measurement() {
		assert_eq!(micros(Duration::from_micros(420)), "420 us");
		assert_eq!(
			micros(Duration::ZERO),
			"0 us",
			"a part measured at nothing is not the same as a part that never ran, which the row \
			 above prints as two words instead"
		);
	}
}
