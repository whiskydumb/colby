//! The runtime: everything that runs a world, as a library.
//!
//! It owns every byte of mutable state in the process, loads the game module,
//! and swaps it when a new build appears. Nothing it holds lives inside the
//! module, which is the entire reason a swap is allowed to be this boring.
//!
//! **A library rather than a binary, and the difference is what it does not
//! know.** Nothing in here reads the command line or a path baked in at build
//! time: the executable that links this hands over the arguments and the few
//! facts only a build script can know - @ref [`Build`] - and everything below
//! takes them as values. That is what lets a second executable link the same
//! runtime, and what keeps a test from depending on where the checkout happens
//! to be. The two asset-tree overrides in [`assets`] are the one thing still
//! read from the environment, and they go when a project replaces the
//! checkout as the thing being run.
//!
//! Six modes, and four of them open no window. `--shot` writes a picture,
//! `--record` a sound, `--link` runs two endpoints against each other and
//! prints a hash, `--host` and `--join` are the two windowless ends of a wire,
//! and everything else is a window. @ref [`run`], which is where the six are
//! told apart.

// @note: crate-wide opt-in to the workspace `unsafe-code = "deny"`. Every unsafe
// block in this crate is a call across the module boundary or the resolution of
// a symbol from it; there is nothing else here that could be one.
#![allow(unsafe_code)]

#[cfg(not(any(feature = "hot_reload", feature = "static_game")))]
compile_error!("colby_runtime needs either the `hot_reload` or the `static_game` feature");

mod app;
mod assets;
mod console;
mod game;
mod host;
mod input;
mod link;
mod mode;
mod net;
mod record;
mod saves;
mod shot;
mod step;
#[cfg(feature = "hot_reload")]
mod watch;

use std::path::PathBuf;

use colby_core::{Result, err, log};
use colby_engine::winit::event_loop::{ControlFlow, EventLoop};

use crate::app::App;

/// The few facts a running process cannot work out for itself.
///
/// All of them are properties of the *build* rather than of the run, and the
/// only thing that knows them is the build script of whichever executable
/// links this runtime: which profile it was built into, which `RUSTFLAGS`
/// produced it, which cargo built it, which package it is, and where the
/// workspace was. The runtime needs them for exactly one thing - rebuilding
/// the game module so that it matches the build that is running - and takes
/// them as a value so that nothing in here is baked in.
#[derive(Clone, Debug)]
pub struct Build {
	/// The workspace directory the executable was built from.
	///
	/// Everything that reads a tree of files - the game module's watcher, the
	/// asset compiler, the console archive, the saves - resolves against this.
	pub workspace: PathBuf,

	/// The cargo that built the executable, as a path or a name on `PATH`.
	pub cargo: String,

	/// The profile directory the executable was built into: `hot`, `dev`.
	pub profile: String,

	/// The `RUSTFLAGS` the executable was built with, in cargo's encoded form.
	///
	/// A module built with different flags would carry its own std and its own
	/// `colby_core`, which quietly breaks every property hot-reload relies on.
	pub rustflags: String,

	/// The package the executable is, so that a rebuild excludes it.
	///
	/// A running executable can never be replaced while it is running, and
	/// cargo attempting it turns a working reload into `Access is denied`.
	pub package: String,
}

/// Brings the process up, runs whichever mode was asked for, and takes it back
/// down in order.
///
/// @param arguments - the command line, without the program's own name
/// @param build - what the build script of the executable knew
pub fn run(arguments: &[String], build: Build) -> Result {
	log::init()?;

	// before anything else, including the check that this process is laid out
	// for hot-reload: a two-endpoint run loads no module, opens no window and
	// opens no socket. The wire between the two is a pair of inboxes, so the
	// answer is the same on a machine with a network and one without, and in
	// any build. @ref `crate::link`.
	if let Some(steps) = link::requested(arguments) {
		return link::run(steps);
	}

	prepare()?;

	// a screenshot never opens a window, so it takes its own way out.
	if let Some(path) = shot::requested(arguments) {
		let result = shot::take(&path, &build.workspace);
		finish();

		return result;
	}

	// and neither does a recording, which additionally opens no device: what
	// it writes has to be the same file on a machine with speakers and one
	// without.
	if let Some(request) = record::requested(arguments) {
		let result = record::take(&request, &build.workspace);
		finish();

		return result;
	}

	// and a host opens a socket instead of a window, which is a mode rather
	// than a build: the same executable, the same module, the same step.
	if let Some(port) = host::requested(arguments) {
		let result = host::serve(port, &build.workspace);
		finish();

		return result;
	}

	// and so does a client that was asked for without one, which is the same
	// loop with the endpoint pointed the other way. It is here rather than in
	// the window path because it never makes a window at all. @ref
	// `crate::host`.
	if let Some(address) = host::joining(arguments) {
		let result = host::join(address, &build.workspace);
		finish();

		return result;
	}

	let event_loop =
		EventLoop::new().map_err(|error| err!(Graphics("creating the event loop: {error}")))?;

	// @note: `Poll` rather than `Wait`. The frame is paced by the surface's
	// vsync, not by the event loop, and a game keeps simulating whether or not
	// the window has anything to say.
	event_loop.set_control_flow(ControlFlow::Poll);

	let mut app = App::new(build, net::standing(arguments));
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
