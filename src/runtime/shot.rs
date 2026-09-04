//! Rendering one frame to a file instead of to a window.
//!
//! `colby --shot picture.png` brings the world up exactly as the window path
//! does, runs it for a fixed number of simulation steps, and writes what it
//! sees. No window is opened, so this works over a remote shell and in a build.
//! It is also the only way anyone reviewing a change to the renderer can see
//! the result without being at the machine.
//!
//! It shares the runtime with the window path and nothing else. In particular
//! it does not share the clock: the whole point of a screenshot is that the
//! same build produces the same picture, and a clock would make the number of
//! steps depend on how busy the machine was.

use std::{path::Path, time::Duration};

use colby_core::{
	Err, Result,
	abi::Input,
	info,
	time::{Rate, STEP},
};
use colby_engine::{Capture, Image, Overlay};

use crate::{Front, Project, Runtime};

/// Where the picture goes when the flag is given without a path.
pub(crate) const DEFAULT_PATH: &str = "colby.png";

/// How big the picture is.
const SIZE: (u32, u32) = (1280, 720);

/// How many simulation steps to run before taking the picture.
///
/// A second and a half at the fixed step, which is long enough for a game that
/// animates to have something to show.
const STEPS: u32 = 90;

/// Runs the game for a moment and writes what the camera sees.
///
/// @param path - where to write the picture
/// @param project - the project to picture
/// @return `Ok` once the file is on disk
pub(crate) fn take(path: &Path, project: &Project) -> Result {
	// the adapter first, before anything is brought up: a machine with nothing
	// to render on has no business loading a module to find that out.
	let Some(mut capture) = Capture::new(SIZE.0, SIZE.1)? else {
		return Err!(Graphics("no usable adapter, so there is nothing to render with"));
	};

	// no console, so no config file and none of the host's variables, and no
	// device: a screenshot must be the same on a machine with speakers and one
	// without, and opening one would put a driver's cadence inside a path that
	// has to be reproducible. The runtime lays the interface out against the
	// picture's own size at a scale of one, for the same reason - a screenshot
	// taken on a display with another scale would otherwise lay the interface
	// out differently from one taken here. @ref `Front::Fixed`.
	let mut runtime = Runtime::open(Front::Fixed, project)?;
	let mut input = Input::default();

	runtime
		.interface
		.attach(capture.device(), Capture::format())?;

	for number in 1..=STEPS {
		// the simulated time this step ends at, computed rather than
		// accumulated: ninety steps is exactly ninety steps however the
		// arithmetic rounds. The wire's clock is nothing here, because there is
		// no wire; a screenshot always plays, because there is no console in
		// this path to ask for anything else and a picture of a world nobody
		// stepped is not what `--shot` is for.
		let time = (STEP * number).as_secs_f32();

		runtime.step(&mut input, Rate::DEFAULT, time, false, Duration::ZERO);
	}

	// the end of the last step exactly, rather than part of the way towards a
	// step that was never run. Without this a screenshot would depend on
	// whatever the world happened to be holding.
	runtime.world.set_interpolation(1.0);

	runtime.interface.run(&runtime.world);
	runtime
		.interface
		.prepare(capture.device(), capture.queue(), &runtime.world);

	let overlay: &mut dyn Overlay = &mut runtime.interface;
	let image: Image = capture.shoot_with(&mut runtime.world, &mut [overlay])?;
	image.write_png(path)?;
	runtime.close();

	info!(
		path = %path.display(),
		width = image.width,
		height = image.height,
		steps = STEPS,
		"screenshot written"
	);

	Ok(())
}
