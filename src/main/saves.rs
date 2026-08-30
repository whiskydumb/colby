//! Writing a world down and reading one back, from the console.
//!
//! **Two files, and the difference between them is the whole of this module.**
//!
//! A **save** is `saves/<name>.cscene`: the world exactly as it stood, arena
//! and generations and all, meant to be put back. Every other file in the tree
//! is *compiled* - a source goes in, a `.cmesh` or a `.ctex` comes out, and the
//! output is derived and lives under `target/`. A save is neither: there is no
//! source, nothing derives it, and `cargo clean` must not take it. So it lives
//! beside `cvars.cfg` in the workspace, for the same reason that one does.
//!
//! A **scene source** is `assets/scenes/<name>.scene`: the world as text
//! somebody can read, diff and edit, which the compiler then turns into an
//! ordinary asset. It is the opposite kind of file - it belongs in the
//! repository, it is the input rather than the output, and writing one is how
//! the editor hands its work back to a person rather than to the engine.
//!
//! Writing a source closes the loop the whole scene format was built for: the
//! editor writes it, the asset watcher notices, the compiler produces the
//! `.cscene` beside every other asset, and the world it describes is in
//! `World::scenes` a quarter of a second later - with no reload of anything.
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
	path::{Path, PathBuf},
	sync::{Mutex, MutexGuard},
};

#[cfg(test)]
use colby_asset::AlignedBytes;
use colby_asset::{level, scene as file};
use colby_core::{
	Result,
	abi::{World, scene},
	err, error, info, warn,
};
use colby_physics::Simulation;

/// The directory saves live in, under the workspace.
pub(crate) const DIRECTORY: &str = "saves";

/// The directory scene sources live in, under the asset tree.
///
/// Not a choice this module makes: it is where the compiler already looks, and
/// a source written anywhere else would be a file nothing compiles.
pub(crate) const SOURCES: &str = "scenes";

/// The directory props live in, under the same tree.
///
/// Also not a choice: the spawn menu finds a prop by walking the scene registry
/// for everything named `props/`, and the compiler names an asset by its own
/// path. So a contraption written here is offered by the menu on the next run
/// with nothing told about it.
pub(crate) const PROPS: &str = "props";

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

	/// Write the world out as a source somebody can edit.
	Write(String),

	/// Write one *registered scene* out as a prop somebody can spawn.
	///
	/// The odd one out, and deliberately so: the other three are about the
	/// world, and this one is about a piece of it that the game has already
	/// cut out and put in the registry. The host has no idea which piece,
	/// because what is under somebody's crosshair is the game's business and
	/// lives in its arena. What the two share is a name.
	Prop(String),
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
		| Request::Write(name) => write(world, name),
		| Request::Prop(name) => prop(world, name),
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

/// Writes the world out as a source somebody can edit.
///
/// The other direction of the editor's work: everything a person moved with a
/// pointer, back as text they can read and a version control system can merge.
/// What lands on disk is an *input* - the asset watcher picks it up on its next
/// poll, the compiler turns it into a `.cscene`, and the world it describes is
/// in the registry a moment later.
///
/// It overwrites without asking, which is what saving means everywhere else;
/// the log says whether a file was made or replaced, and a source lives in the
/// repository, which is where the previous one still is.
///
/// @param world - what to write
/// @param name - what to call it, without an extension
///
/// # Errors
///
/// If the name is not a plain file name, if the world holds a number JSON
/// cannot write, or if the directory or the file cannot be written.
fn write(world: &World, name: &str) -> Result {
	let path = source(name)?;
	let existed = path.exists();
	let bytes = written(world, &path)?;

	info!(
		path = %path.display(),
		bytes,
		entities = world.entities.len(),
		bodies = world.bodies.len(),
		replaced = existed,
		"scene written as a source"
	);

	Ok(())
}

/// Writes one registered scene out as a prop.
///
/// **A saved contraption is a prop and there is no second format for one.** The
/// game cuts a connected piece out of the world, registers it under
/// `props/<name>`, and asks for this; what lands on disk is an ordinary
/// `.scene` in the directory the spawn menu already walks, so the next run
/// offers it beside the ones written by hand. Nothing new reads it and nothing
/// new writes it.
///
/// @param world - the registry to take the scene from
/// @param name - what it is called, without its prefix or its extension
///
/// # Errors
///
/// If the name is not a plain file name, if nothing is registered under it, if
/// the piece holds a number JSON cannot write, or if the file cannot be
/// written.
fn prop(world: &World, name: &str) -> Result {
	let path = crate::assets::source_root()
		.join(PROPS)
		.join(plain(name)?)
		.with_extension(level::EXTENSION);
	let registered = format!("{PROPS}/{name}");
	let id = world.scenes.find(&registered);

	if !id.is_some() {
		return Err(err!(Asset("nothing is registered as {registered}")));
	}

	let existed = path.exists();
	let piece = world.scenes.data(id);
	let text = level::export(piece)?;

	if let Some(directory) = path.parent() {
		fs::create_dir_all(directory)?;
	}

	fs::write(&path, text.as_bytes())?;
	info!(
		path = %path.display(),
		bytes = text.len(),
		entities = piece.things.len(),
		bodies = piece.solids.len(),
		joints = piece.links.len(),
		replaced = existed,
		"prop written"
	);

	Ok(())
}

/// The same, given the file rather than the name.
///
/// Split out for the reason the asset loop's own `at` is: a function that takes
/// a path can be run against a temporary directory, and one that works a path
/// out of the environment cannot be run twice at once.
///
/// @note: this is also the only one of the two a test may call. The named form
/// writes into the asset tree, which is the repository - and a test that calls
/// it to check that a bad name is *refused* writes a file into it the moment
/// somebody mutates the check it is testing. Found exactly that way.
///
/// @param world - what to write
/// @param path - where to put it
/// @return how many bytes were written
///
/// # Errors
///
/// If the world holds a number JSON cannot write, or the directory or the file
/// cannot be written.
fn written(world: &World, path: &Path) -> Result<usize> {
	let text = level::export(&scene::capture(world))?;

	if let Some(directory) = path.parent() {
		fs::create_dir_all(directory)?;
	}

	fs::write(path, text.as_bytes())?;

	Ok(text.len())
}

/// Where a scene source by that name is.
///
/// Under the asset tree rather than the workspace, and under whichever tree
/// `COLBY_ASSETS` names if it names one - so a source written here is a source
/// the watcher is watching, on every machine and in every test.
///
/// @param name - what a command was given
///
/// # Errors
///
/// As [`path`], and for the same reasons.
fn source(name: &str) -> Result<PathBuf> {
	Ok(crate::assets::source_root()
		.join(SOURCES)
		.join(plain(name)?)
		.with_extension(level::EXTENSION))
}

/// Where a save by that name is.
///
/// @param name - what a command was given
///
/// # Errors
///
/// As [`plain`].
fn path(name: &str) -> Result<PathBuf> {
	Ok(crate::workspace()
		.join(DIRECTORY)
		.join(plain(name)?)
		.with_extension(file::EXTENSION))
}

/// A name a file may actually be called, out of what somebody typed.
///
/// One flat directory and no subdirectories, whichever of the two a name is
/// for: a name is a file name, so a separator or a parent in it is refused
/// rather than quietly resolved. A console is a place people type quickly, and
/// `scene.save ../../something` is not a thing to find out about afterwards.
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
fn plain(name: &str) -> Result<&str> {
	let trimmed = name.trim();

	if trimmed.is_empty() {
		return Err(err!(Asset("a scene needs a name")));
	}

	if trimmed.contains(['/', '\\', ':', '.']) {
		return Err(err!(Asset("{trimmed} is not a name a scene can have")));
	}

	Ok(trimmed)
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
	use std::env;

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
			assert!(source(name).is_err(), "nor one a source can, which is one rule");
		}

		assert!(path("quicksave").is_ok(), "and a plain one is");
		assert!(path(" quicksave ").is_ok(), "with the spaces around it taken off");
		assert!(source(" quicksave ").is_ok(), "either side of the same rule");
	}

	#[test]
	fn a_source_lands_where_the_compiler_is_already_looking() {
		let path = source("edited").expect("a plain name");

		assert_eq!(
			path.extension().and_then(std::ffi::OsStr::to_str),
			Some(level::EXTENSION),
			"the extension a source has"
		);
		assert_eq!(
			path.parent().and_then(Path::file_name),
			Some(std::ffi::OsStr::new(SOURCES)),
			"in the directory scenes are compiled from"
		);
		assert!(
			path.starts_with(crate::assets::source_root()),
			"under the tree the watcher is watching: {}",
			path.display()
		);
	}

	#[test]
	fn a_world_written_as_a_source_is_the_world() {
		let (world, _simulation) = peopled();
		let inside = env::temp_dir()
			.join("colby_test_written")
			.join("scenes");
		let path = inside.join("edited.scene");

		// a previous run of this left the directory behind, and a test that
		// only passes on a machine that has run it before is not a test of
		// anything. What is under test includes making the directory.
		drop(fs::remove_dir_all(&inside));

		let bytes = written(&world, &path).expect("it is written");
		let text = fs::read_to_string(&path).expect("and read back");

		assert_eq!(bytes, text.len(), "what was reported is what landed");

		let read = level::import(&text).expect("what was written is a scene source");

		assert_eq!(read.things.len(), world.entities.len(), "every entity is in it");
		assert_eq!(read.solids.len(), world.bodies.len(), "and every body");
		assert_eq!(
			read.things[0].transform.position,
			Vec3::new(3.0, 4.0, 5.0),
			"where the world had it"
		);
		assert_eq!(read.solids[0].thing, 0, "and the body still drives it");

		drop(fs::remove_dir_all(&inside));
	}

	#[test]
	fn a_source_written_twice_replaces_the_first_rather_than_adding_to_it() {
		let (world, _simulation) = peopled();
		let inside = env::temp_dir()
			.join("colby_test_replaced")
			.join("scenes");
		let path = inside.join("edited.scene");
		drop(fs::remove_dir_all(&inside));

		assert!(!path.exists(), "there is nothing there to start with");
		written(&world, &path).expect("it is written");
		assert!(path.exists(), "and now there is");

		let once = fs::read_to_string(&path).expect("read back");
		written(&world, &path).expect("and written over");
		let twice = fs::read_to_string(&path).expect("read back again");

		assert_eq!(once, twice, "the same world writes the same file");
		assert!(level::import(&twice).is_ok(), "and what is there is still one scene");

		drop(fs::remove_dir_all(&inside));
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
			path.parent().and_then(Path::file_name),
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
