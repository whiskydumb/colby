//! Rendering one frame to a file instead of to a window.
//!
//! `colby --shot picture.png` loads the game module exactly as the window path
//! does, runs it for a fixed number of simulation steps, and writes what it
//! sees. No window is opened, so this works over a remote shell and in a build.
//! It is also the only way anyone reviewing a change to the renderer can see
//! the result without being at the machine.
//!
//! It shares the step body with the window path and nothing else. In
//! particular it does not share the clock: the whole point of a screenshot is
//! that the same build produces the same picture, and a clock would make the
//! number of steps depend on how busy the machine was.

use std::path::{Path, PathBuf};

use colby_core::{
	Err, Result,
	abi::{Input, World},
	info,
	time::STEP,
};
use colby_engine::{Capture, Image, Overlay};
use colby_physics::Simulation;
use colby_script::Vm;
use colby_ui::Interface;

use crate::{assets::Assets, game::Game, step};

/// The flag that asks for a screenshot.
const FLAG: &str = "--shot";

/// Where the picture goes when the flag is given without a path.
const DEFAULT_PATH: &str = "colby.png";

/// How big the picture is.
const SIZE: (u32, u32) = (1280, 720);

/// How many simulation steps to run before taking the picture.
///
/// A second and a half at the fixed step, which is long enough for a game that
/// animates to have something to show.
const STEPS: u32 = 90;

/// Reads the command line for a screenshot request.
///
/// Accepts `--shot` on its own, `--shot path` and `--shot=path`.
///
/// @return where to write the picture, if one was asked for
#[must_use]
pub(crate) fn requested() -> Option<PathBuf> {
	let mut arguments = std::env::args().skip(1);

	while let Some(argument) = arguments.next() {
		if let Some(path) = argument.strip_prefix(&format!("{FLAG}=")) {
			return Some(PathBuf::from(path));
		}

		if argument == FLAG {
			return Some(
				arguments
					.next()
					.map_or_else(|| PathBuf::from(DEFAULT_PATH), PathBuf::from),
			);
		}
	}

	None
}

/// Runs the game for a moment and writes what the camera sees.
///
/// @param path - where to write the picture
/// @return `Ok` once the file is on disk
pub(crate) fn take(path: &Path) -> Result {
	let Some(mut capture) = Capture::new(SIZE.0, SIZE.1)? else {
		return Err!(Graphics("no usable adapter, so there is nothing to render with"));
	};

	// boxed and installed before anything else touches the world, for the reason
	// the window path boxes it: the world keeps this address.
	let mut simulation = Box::new(Simulation::new());
	let mut world = Box::<World>::default();
	world.install_physics(simulation.table());

	// the same order the window path uses: assets first, so the game's `init`
	// can resolve a mesh by name. A screenshot that skipped this would quietly
	// be a picture of a different scene than the one on screen.
	Assets::new(&crate::workspace()).sync(&mut world);

	let mut game = Game::open(&mut world)?;
	let mut input = Input::default();

	// the same size the picture is, at a scale of one: a screenshot taken on a
	// display with another scale would otherwise lay the interface out
	// differently from one taken here, and the whole point of `--shot` is that
	// the same build makes the same picture.
	// a screenshot runs the documents' scripts, because a screenshot is meant to
	// show what the window shows. Nothing in the environment can read a clock or
	// a file, so ninety steps of it are the same ninety steps on every machine.
	// @ref `colby_script`.
	let mut scripts = Vm::new()?;

	let mut interface = Interface::new();
	let viewport = colby_core::glam::Vec2::new(
		f32::from(u16::try_from(SIZE.0).unwrap_or(u16::MAX)),
		f32::from(u16::try_from(SIZE.1).unwrap_or(u16::MAX)),
	);
	world.ui.set_viewport(viewport, 1.0);

	// beside the viewport and before the steps, exactly as the window path
	// does it. `shoot_with` sets this as well, and too late to be the only
	// place: it sets it on the way into the scene pass, which is after the
	// overlays have been prepared - and a debug label is projected during
	// prepare, so one would be placed through whatever aspect the world was
	// still holding. That is a stretch of exactly the window's own ratio, and
	// it is invisible at the middle of the screen, which is where the only
	// label this ever had happened to sit.
	world.aspect = viewport.x / viewport.y;
	interface.attach(capture.device(), Capture::format())?;

	for number in 1..=STEPS {
		// the simulated time this step ends at, computed rather than
		// accumulated: ninety steps is exactly ninety steps however the
		// arithmetic rounds.
		let time = (STEP * number).as_secs_f32();

		step::run(
			&mut world,
			step::Parts {
				game: Some(&mut game),
				interface: &mut interface,
				scripts: Some(&mut scripts),
				simulation: simulation.as_mut(),
				// no device, deliberately: a screenshot must be the same on a
				// machine with speakers and one without, and opening one would
				// put a driver's cadence inside a path that has to be
				// reproducible.
				audio: None,
				// no wire, so a screenshot is what it always was.
				wire: None,
			},
			&mut input,
			time,
			// a screenshot always plays. There is no console in this path to
			// ask for anything else, and a picture of a world nobody stepped
			// is not what `--shot` is for.
			false,
		);
	}

	// the end of the last step exactly, rather than part of the way towards a
	// step that was never run. Without this a screenshot would depend on
	// whatever the world happened to be holding.
	world.set_interpolation(1.0);

	interface.run(&world);
	interface.prepare(capture.device(), capture.queue(), &world);

	let overlay: &mut dyn Overlay = &mut interface;
	let image: Image = capture.shoot_with(&mut world, &mut [overlay])?;
	image.write_png(path)?;
	game.close(&mut world);

	info!(
		path = %path.display(),
		width = image.width,
		height = image.height,
		steps = STEPS,
		"screenshot written"
	);

	Ok(())
}
