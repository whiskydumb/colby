//! The runner.
//!
//! It owns every byte of mutable state in the process, loads the game module,
//! and swaps it when a new build appears. Nothing it holds lives inside the
//! module, which is the entire reason a swap is allowed to be this boring.

// @note: crate-wide opt-in to the workspace `unsafe-code = "deny"`. Every unsafe
// block in this crate is a call across the module boundary or the resolution of
// a symbol from it; there is nothing else here that could be one.
#![allow(unsafe_code)]

#[cfg(not(any(feature = "hot_reload", feature = "static_game")))]
compile_error!("colby needs either the `hot_reload` or the `static_game` feature");

mod app;
mod assets;
mod console;
mod game;
mod input;
mod mode;
mod record;
mod saves;
mod shot;
mod step;
#[cfg(feature = "hot_reload")]
mod watch;

use std::{env, path::PathBuf, process::ExitCode};

use colby_core::{Result, err, error, log};
use colby_engine::winit::event_loop::{ControlFlow, EventLoop};

use crate::app::App;

/// The workspace directory this executable was built from.
///
/// Baked in at build time and overridable at runtime, which is what makes it
/// possible to run the executable from somewhere else and still point it at a
/// checkout. Everything that reads a tree of files - the game module's watcher,
/// the asset compiler - resolves against this.
#[must_use]
pub(crate) fn workspace() -> PathBuf {
	env::var("COLBY_WORKSPACE")
		.map(PathBuf::from)
		.unwrap_or_else(|_| PathBuf::from(env!("COLBY_WORKSPACE")))
}

fn main() -> ExitCode {
	match run() {
		| Ok(()) => ExitCode::SUCCESS,
		| Err(error) => {
			error!(%error, "colby stopped");

			ExitCode::FAILURE
		},
	}
}

/// Brings the process up, runs the event loop, and takes it back down in order.
fn run() -> Result {
	log::init()?;
	prepare()?;

	// a screenshot never opens a window, so it takes its own way out.
	if let Some(path) = shot::requested() {
		let result = shot::take(&path);
		finish();

		return result;
	}

	// and neither does a recording, which additionally opens no device: what
	// it writes has to be the same file on a machine with speakers and one
	// without.
	if let Some(request) = record::requested() {
		let result = record::take(&request);
		finish();

		return result;
	}

	let event_loop =
		EventLoop::new().map_err(|error| err!(Graphics("creating the event loop: {error}")))?;

	// @note: `Poll` rather than `Wait`. The frame is paced by the surface's
	// vsync, not by the event loop, and a game keeps simulating whether or not
	// the window has anything to say.
	event_loop.set_control_flow(ControlFlow::Poll);

	let mut app = App::new();
	let result = event_loop
		.run_app(&mut app)
		.map_err(|error| err!(Graphics("running the event loop: {error}")));

	finish();
	result?;

	app.into_result()
}

/// Verifies the process is laid out for hot-reload, and clears `%TEMP%`.
#[cfg(feature = "hot_reload")]
fn prepare() -> Result {
	colby_core::mods::linkage::require_shared_core()?;
	colby_core::mods::path::clear_scratch();

	Ok(())
}

#[cfg(not(feature = "hot_reload"))]
fn prepare() -> Result { Ok(()) }

/// Removes this run's staged module images.
#[cfg(feature = "hot_reload")]
fn finish() { colby_core::mods::path::clear_scratch(); }

#[cfg(not(feature = "hot_reload"))]
fn finish() {}
