//! Writing a world to a file and reading one back, from the console.
//!
//! The first thing colby writes for itself. Every other file in the tree is
//! *compiled* - a source goes in, a `.cmesh` or a `.ctex` comes out, and the
//! output is derived and lives under `target/`. A save is neither: there is no
//! source, nothing derives it, and `cargo clean` must not take it. So it lives
//! beside `cvars.cfg` in the workspace, in `saves/`, for the same reason that
//! one does.
//!
//! **A load cannot happen inside the command that asked for it, and that is
//! the whole shape of this module.** Putting a world back replaces every table
//! in [`World`] and touches nothing a solver derived - @ref
//! [`Simulation::forget`](colby_physics::Simulation::forget) - and a console
//! command is handed a `&mut World` and nothing else, on purpose: a command is
//! a function pointer that a *game module* may also register, so the signature
//! cannot mention anything the host owns privately. A command therefore leaves
//! a note here and the frame loop reads it, which is the same shape `sim.step`
//! already has with `World::owed_steps` and differs only in where the note is
//! kept: a scene path is the host's business rather than the world's, so it
//! does not become a field on `World`.

use std::{
	fs,
	path::PathBuf,
	sync::{Mutex, MutexGuard},
};

#[cfg(test)]
use colby_asset::AlignedBytes;
use colby_asset::scene as file;
use colby_core::{
	Result,
	abi::{World, scene},
	err, error, info, warn,
};
use colby_physics::Simulation;

/// The directory saves live in, under the workspace.
pub(crate) const DIRECTORY: &str = "saves";

/// What a console command has asked the host to do, waiting for a frame.
///
/// One at a time: typing two loads before the next frame means the second one
/// is what was meant. A queue would only make the first one happen too.
static ASKED: Mutex<Option<Request>> = Mutex::new(None);

/// One thing to do with a scene.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Request {
	/// Write the world out under a name.
	Save(String),

	/// Read a world back in from one.
	Load(String),
}

/// Leaves a note for the frame loop.
///
/// @param request - what to do
pub(crate) fn ask(request: Request) { *lock() = Some(request); }

/// Does whatever a command asked for, if one did.
///
/// The frame loop's, called once a frame and outside a simulation step: a load
/// replaces the world, and doing that halfway through a step would leave the
/// second half of it running against a world the first half never saw.
///
/// @param world - the world to write out or replace
/// @param simulation - the solver, whose derived state a load drops
pub(crate) fn serve(world: &mut World, simulation: &mut Simulation) {
	let Some(request) = lock().take() else {
		return;
	};

	let outcome = match &request {
		| Request::Save(name) => save(world, name),
		| Request::Load(name) => load(world, simulation, name),
	};

	if let Err(failure) = outcome {
		error!(%failure, "the scene could not be dealt with");
	}
}

/// Writes the world out.
///
/// @param world - what to write
/// @param name - what to call it, without an extension
///
/// # Errors
///
/// If the description will not fit in one file, or the directory or the file
/// cannot be written.
fn save(world: &World, name: &str) -> Result {
	let path = path(name)?;
	let bytes = file::encode(&scene::capture(world))?;

	if let Some(directory) = path.parent() {
		fs::create_dir_all(directory)?;
	}

	fs::write(&path, &bytes)?;
	info!(
		path = %path.display(),
		bytes = bytes.len(),
		entities = world.entities.len(),
		bodies = world.bodies.len(),
		"scene saved"
	);

	Ok(())
}

/// Reads a world back in.
///
/// @param world - the world to replace
/// @param simulation - the solver, told to forget what it derived
/// @param name - the save to read, without an extension
///
/// # Errors
///
/// If the file cannot be read, is not a scene this build reads, or was written
/// by a build of the game whose state has a different shape.
fn load(world: &mut World, simulation: &mut Simulation, name: &str) -> Result {
	let path = path(name)?;
	let read = file::SceneFile::open(&path)?;
	let put = scene::restore(world, &read.to_scene_data())?;

	// immediately, and it is the caller's obligation rather than something
	// `restore` can do for itself: `colby_core` does not depend on the solver
	// and the query table is deliberately two functions. @ref
	// `colby_physics::Simulation::forget`.
	simulation.forget();

	info!(
		path = %path.display(),
		entities = put.things,
		bodies = put.solids,
		joints = put.links,
		state = put.arena,
		"scene loaded"
	);

	Ok(())
}

/// Where a save by that name is.
///
/// One flat directory and no subdirectories: a name is a file name, so a
/// separator or a parent in it is refused rather than quietly resolved. A
/// console is a place people type quickly, and `scene.save ../../something` is
/// not a thing to find out about afterwards.
///
/// A dot is refused with them, which is the rule the compiled formats already
/// follow for the same reason: the last dot in a path here is the extension,
/// so `scene.save version.2` would land in `version.cscene` and lose half of
/// what was typed without saying so.
///
/// @param name - what a command was given
///
/// # Errors
///
/// If the name is empty or is anything other than a plain file name.
fn path(name: &str) -> Result<PathBuf> {
	let trimmed = name.trim();

	if trimmed.is_empty() {
		return Err(err!(Asset("a save needs a name")));
	}

	if trimmed.contains(['/', '\\', ':', '.']) {
		return Err(err!(Asset("{trimmed} is not a name a save can have")));
	}

	Ok(crate::workspace()
		.join(DIRECTORY)
		.join(trimmed)
		.with_extension(file::EXTENSION))
}

/// The pending note, whichever thread is asking.
///
/// A poisoned lock is nothing to stop for: what it guards is one optional
/// request, and the worst a torn one could be is a save that does not happen.
fn lock() -> MutexGuard<'static, Option<Request>> {
	ASKED.lock().unwrap_or_else(|poisoned| {
		warn!("the scene request lock was poisoned; carrying on with what is in it");

		poisoned.into_inner()
	})
}

/// Reads what is on disk, without going through the world.
///
/// The tests' way in, and the only thing here that is not the console's.
#[cfg(test)]
fn read(name: &str) -> Result<colby_core::abi::SceneData> {
	let bytes = AlignedBytes::read(&path(name)?)?;

	Ok(file::SceneFile::from_bytes(bytes)?.to_scene_data())
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{Body, Shape, Transform},
		glam::Vec3,
	};

	use super::*;

	/// A world with something in it, and a simulation wired to it.
	fn peopled() -> (Box<World>, Box<Simulation>) {
		let simulation = Box::new(Simulation::new());
		let mut world = Box::new(World::new());
		world.install_physics(simulation.table());

		let entity = world
			.entities
			.spawn_at(Transform::at(Vec3::new(3.0, 4.0, 5.0)));
		world
			.bodies
			.spawn(Body::dynamic(Shape::UNIT, Transform::at(Vec3::Y), 2.0).driving(entity));

		(world, simulation)
	}

	#[test]
	fn a_name_that_is_not_a_file_name_is_refused() {
		for name in ["", "   ", "..", "../escape", "sub/one", "c:\\here", "version.2"] {
			assert!(path(name).is_err(), "{name} is not a name a save can have");
		}

		assert!(path("quicksave").is_ok(), "and a plain one is");
		assert!(path(" quicksave ").is_ok(), "with the spaces around it taken off");
	}

	#[test]
	fn a_name_lands_in_the_saves_directory_with_the_scene_extension() {
		let path = path("quicksave").expect("a plain name");

		assert_eq!(
			path.extension().and_then(std::ffi::OsStr::to_str),
			Some(file::EXTENSION),
			"the extension is the format's"
		);
		assert_eq!(
			path.parent().and_then(std::path::Path::file_name),
			Some(std::ffi::OsStr::new(DIRECTORY)),
			"and it is under the saves directory"
		);
	}

	#[test]
	fn asking_twice_before_a_frame_leaves_the_second_one() {
		ask(Request::Save("first".to_owned()));
		ask(Request::Load("second".to_owned()));

		let held = lock().take();

		assert_eq!(
			held,
			Some(Request::Load("second".to_owned())),
			"the last thing typed is the one that was meant"
		);
	}

	#[test]
	fn a_world_written_out_reads_back_as_itself() {
		let (world, _simulation) = peopled();
		let name = "colby_test_round_trip";

		save(&world, name).expect("it is written");
		let read = read(name).expect("and read back");

		assert_eq!(read, scene::capture(&world), "the file is the world");

		drop(fs::remove_file(path(name).expect("a plain name")));
	}

	#[test]
	fn a_save_nobody_wrote_is_an_error_rather_than_an_empty_world() {
		let (mut world, mut simulation) = peopled();
		let before = scene::capture(&world);

		let failure = load(&mut world, &mut simulation, "colby_test_no_such_save")
			.expect_err("there is no such file");

		assert!(
			failure
				.to_string()
				.contains("colby_test_no_such_save"),
			"the message names it: {failure}"
		);
		assert_eq!(scene::capture(&world), before, "and the world is untouched");
	}
}
