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
//! to be. Everything on disk that is not the engine's own is a project's -
//! @ref [`Project`] - named by `--project` or found in the working directory,
//! and nothing in here reads the environment at all.
//!
//! **One state, one stand-up, one parser.** [`Runtime`] is the state a world
//! needs to run and [`Runtime::open`] the one order it is brought up in, with
//! a [`Front`] saying what kind of process this is; [`Launch`] is the command
//! line read once. Six runs, and four of them open no window: `--shot` writes
//! a picture, `--record` a sound, `--link` runs two endpoints against each
//! other and prints a hash, `--host` and `--join` are the two windowless ends
//! of a wire, and everything else is a window. @ref [`run`], which is where
//! the six are told apart.

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
mod launch;
mod link;
mod mode;
mod net;
mod record;
mod runtime;
mod saves;
mod shot;
mod step;
#[cfg(feature = "hot_reload")]
mod watch;

use std::path::{Path, PathBuf};

pub use colby_asset::Project;
use colby_core::{Result, log};

pub use crate::{
	launch::{Launch, Run},
	net::Standing,
	runtime::{Front, Runtime, VIEWPORT},
};

/// The few facts a running process cannot work out for itself.
///
/// All of them are properties of the *build* rather than of the run, and the
/// only thing that knows them is the build script of whichever executable
/// links this runtime: which profile it was built into, which `RUSTFLAGS`
/// produced it, which cargo built it, which package it is, and where the
/// engine checkout was. The runtime needs them for exactly one thing -
/// rebuilding the game module so that it matches the build that is running -
/// and takes them as a value so that nothing in here is baked in.
#[derive(Clone, Debug)]
pub struct Build {
	/// The engine checkout the executable was built from.
	///
	/// What a rebuild of a game module runs in, and what the watcher looks at
	/// for edits beneath the game that need a restart. Nothing a project owns
	/// resolves against this: assets, saves and config are the project's.
	pub engine: PathBuf,

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

/// Brings the process up, runs whichever run was asked for, and takes it back
/// down in order.
///
/// @param arguments - the command line, without the program's own name
/// @param build - what the build script of the executable knew
/// @param here - the working directory, which is where the project is looked
/// for when `--project` names none
pub fn run(arguments: &[String], build: Build, here: &Path) -> Result {
	log::init()?;

	let Launch { run, project } = Launch::parse(arguments);

	// a two-endpoint run before anything else, including the check that this
	// process is laid out for hot-reload and the project itself: it loads no
	// module, opens no window, opens no socket and reads no file. The wire
	// between the two is a pair of inboxes, so the answer is the same on a
	// machine with a network and one without, and in any build. @ref
	// `crate::link`.
	if let Run::Link(steps) = run {
		return link::run(steps);
	}

	// the project before anything that touches a file, because every file
	// hangs off it: the one named, or the one in the working directory - the
	// rule a project manager's `--path` follows, and the branch a launcher
	// takes when neither is there.
	let project = Project::open(project.as_deref().unwrap_or(here))?;

	prepare()?;

	let result = match run {
		// answered above, before `prepare`; here so that the match is
		// exhaustive rather than wildcarded.
		| Run::Link(_) => Ok(()),
		// a screenshot never opens a window, so it takes its own way out; and
		// neither does a recording, which additionally opens no device, because
		// what it writes has to be the same file on a machine with speakers and
		// one without.
		| Run::Shot(path) => shot::take(&path, &project),
		| Run::Record { path, steps } => record::take(&path, steps, &project),
		// a socket instead of a window, which is a run rather than a build: the
		// same executable, the same module, the same step. @ref `crate::host`.
		| Run::Host(port) => host::run(Front::Host(port), &project),
		| Run::Join(address) => host::run(Front::Join(address), &project),
		| Run::Window(standing) => app::run(build, standing, &project),
	};

	finish();

	result
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
