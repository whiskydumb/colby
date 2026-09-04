//! Writing a world down and reading one back, from the console.
//!
//! **Two files, and the difference between them is the whole of this module.**
//!
//! A **save** is `saves/<name>.cscene`: the world exactly as it stood, arena
//! and generations and all, meant to be put back. Every other file in the tree
//! is *compiled* - a source goes in, a `.cmesh` or a `.ctex` comes out, and the
//! output is derived and lives under `.colby/`. A save is neither: there is no
//! source, nothing derives it, and `cargo clean` must not take it. So it lives
//! beside `settings.cfg` in the project, for the same reason that one does.
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
//! its line on `World::asked` and the frame loop reads it - the same shape
//! `sim.step` already has with `World::owed_steps`, and the same field every
//! other command that needs something the host owns leaves its line in. @ref
//! [`Asked`] for who takes which line.

use std::{
	fs,
	path::{Path, PathBuf},
};

#[cfg(test)]
use colby_asset::AlignedBytes;
use colby_asset::{Project, level, scene as file};
use colby_core::{
	Result,
	abi::{Asked, World, scene},
	err, error, info,
};
use colby_physics::Simulation;

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

/// `scene.save <name>` - writes the world out.
pub(crate) const SAVE: &str = "scene.save";

/// `scene.load <name>` - puts a saved world back.
pub(crate) const LOAD: &str = "scene.load";

/// `scene.write <name>` - writes the world out as a source somebody can edit.
///
/// The other end of the editor's work, and the one that goes back into the
/// repository rather than beside it.
pub(crate) const WRITE: &str = "scene.write";

/// `scene.prop <name>` - writes one registered scene out as a prop.
///
/// The half of saving a contraption that needs a filesystem. The other half is
/// the game's: it cuts the piece out and registers it, because what is under
/// somebody's crosshair is not something the host can see. @ref
/// [`Request::Prop`].
pub(crate) const PROP: &str = "scene.prop";

/// The four names this module answers for, as they wait on the world.
const NAMES: &[&str] = &[SAVE, LOAD, WRITE, PROP];

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

impl Request {
	/// What one waiting line asks for, if it is one of this module's.
	///
	/// @param asked - a line the frame loop took off the world
	/// @return the request, or `None` for a name that is not a scene's
	fn of(asked: &Asked) -> Option<Self> {
		let name = asked.words.join(" ");

		match asked.name.as_str() {
			| SAVE => Some(Self::Save(name)),
			| LOAD => Some(Self::Load(name)),
			| WRITE => Some(Self::Write(name)),
			| PROP => Some(Self::Prop(name)),
			| _ => None,
		}
	}
}

/// Does whatever a command asked for, if one did.
///
/// The frame loop's, called once a frame and outside a simulation step: a load
/// replaces the world, and doing that halfway through a step would leave the
/// second half of it running against a world the first half never saw.
///
/// **One at a time, and the last one.** Typing two loads before the next frame
/// means the second one is what was meant, and running both would only make
/// the first one happen too.
///
/// @param world - the world to write out or replace
/// @param simulation - the solver, whose derived state a load drops
/// @param project - whose saves and asset tree these are
pub(crate) fn serve(world: &mut World, simulation: &mut Simulation, project: &Project) {
	let Some(request) = crate::console::take(world, NAMES)
		.pop()
		.as_ref()
		.and_then(Request::of)
	else {
		return;
	};

	let outcome = match &request {
		| Request::Save(name) => save(world, project, name),
		| Request::Load(name) => load(world, simulation, project, name),
		| Request::Write(name) => write(world, project, name),
		| Request::Prop(name) => prop(world, project, name),
	};

	if let Err(failure) = outcome {
		error!(%failure, "the scene could not be dealt with");
	}
}

/// Writes the world out.
///
/// @param world - what to write
/// @param project - whose saves
/// @param name - what to call it, without an extension
///
/// # Errors
///
/// If the description will not fit in one file, or the directory or the file
/// cannot be written.
fn save(world: &World, project: &Project, name: &str) -> Result {
	let path = path(&project.saves(), name)?;
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
/// @param project - whose saves
/// @param name - the save to read, without an extension
///
/// # Errors
///
/// If the file cannot be read, is not a scene this build reads, or was written
/// by a build of the game whose state has a different shape.
fn load(world: &mut World, simulation: &mut Simulation, project: &Project, name: &str) -> Result {
	let path = path(&project.saves(), name)?;
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
/// @param project - whose asset tree
/// @param name - what to call it, without an extension
///
/// # Errors
///
/// If the name is not a plain file name, if the world holds a number JSON
/// cannot write, or if the directory or the file cannot be written.
fn write(world: &World, project: &Project, name: &str) -> Result {
	let path = source(&project.assets(), name)?;
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
/// @param project - whose asset tree
/// @param name - what it is called, without its prefix or its extension
///
/// # Errors
///
/// If the name is not a plain file name, if nothing is registered under it, if
/// the piece holds a number JSON cannot write, or if the file cannot be
/// written.
fn prop(world: &World, project: &Project, name: &str) -> Result {
	let path = project
		.assets()
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
/// @param assets - the asset tree
/// @param name - what a command was given
///
/// # Errors
///
/// As [`path`], and for the same reasons.
fn source(assets: &Path, name: &str) -> Result<PathBuf> {
	Ok(assets
		.join(SOURCES)
		.join(plain(name)?)
		.with_extension(level::EXTENSION))
}

/// Where a save by that name is.
///
/// @param saves - the directory saves live in
/// @param name - what a command was given
///
/// # Errors
///
/// As [`plain`].
fn path(saves: &Path, name: &str) -> Result<PathBuf> {
	Ok(saves
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

/// Reads what is on disk, without going through the world.
///
/// The tests' way in, and the only thing here that is not the console's.
#[cfg(test)]
fn read(project: &Project, name: &str) -> Result<colby_core::abi::SceneData> {
	let bytes = AlignedBytes::read(&path(&project.saves(), name)?)?;

	Ok(file::SceneFile::from_bytes(bytes)?.to_scene_data())
}

#[cfg(test)]
mod tests {
	use std::env;

	use colby_asset::project::SAVES_DIR;
	use colby_core::{
		abi::{Body, Shape, Transform},
		glam::Vec3,
	};

	use super::*;

	/// A workspace nothing else is using, so that a test writes nowhere
	/// near the checkout.
	fn workspace(name: &str) -> PathBuf {
		let inside = env::temp_dir().join(format!("colby_test_{name}"));
		drop(fs::remove_dir_all(&inside));

		inside
	}

	/// A project under that workspace, with nothing in it yet.
	fn project(name: &str) -> Project {
		Project::parse(
			&workspace(name),
			r#"{ "schema": 1, "engine": "0.1.0", "id": "testing", "name": "a test" }"#,
		)
		.expect("a project")
	}

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
		let root = project("names");

		for name in ["", "   ", "..", "../escape", "sub/one", "c:\\here", "version.2"] {
			assert!(path(&root.saves(), name).is_err(), "{name} is not a name a save can have");
			assert!(
				source(&root.assets(), name).is_err(),
				"nor one a source can, which is one rule"
			);
		}

		assert!(path(&root.saves(), "quicksave").is_ok(), "and a plain one is");
		assert!(
			path(&root.saves(), " quicksave ").is_ok(),
			"with the spaces around it taken off"
		);
		assert!(source(&root.assets(), " quicksave ").is_ok(), "either side of the same rule");
	}

	#[test]
	fn a_source_lands_where_the_compiler_is_already_looking() {
		let root = project("looking");
		let path = source(&root.assets(), "edited").expect("a plain name");

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
			path.starts_with(root.assets()),
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
		let root = project("landing");
		let path = path(&root.saves(), "quicksave").expect("a plain name");

		assert_eq!(
			path.extension().and_then(std::ffi::OsStr::to_str),
			Some(file::EXTENSION),
			"the extension is the format's"
		);
		assert_eq!(
			path.parent().and_then(Path::file_name),
			Some(std::ffi::OsStr::new(SAVES_DIR)),
			"and it is under the saves directory"
		);
		assert!(path.starts_with(root.root()), "of the project it was given, not the checkout");
	}

	/// A world with the four scene commands registered the way the host does.
	fn console() -> World {
		let mut world = World::new();

		for name in NAMES {
			world
				.cvars
				.command(name, colby_core::abi::console::defer, "");
		}

		world
	}

	#[test]
	fn a_scene_line_waits_on_the_world_and_reads_as_what_it_asked_for() {
		let mut world = console();

		colby_core::abi::console::run(&mut world, "scene.load quicksave");

		let taken = crate::console::take(&mut world, NAMES);

		assert_eq!(taken.len(), 1, "one line waited");
		assert_eq!(
			Request::of(&taken[0]),
			Some(Request::Load("quicksave".to_owned())),
			"and it is the load that was typed"
		);
		assert!(world.asked.is_empty(), "taken, not copied");
		assert_eq!(
			Request::of(&Asked {
				name: "echo".to_owned(),
				..taken[0].clone()
			}),
			None,
			"a name that is not a scene's is not one of these"
		);
	}

	#[test]
	fn asking_twice_before_a_frame_leaves_the_second_one() {
		let mut world = console();

		colby_core::abi::console::run(&mut world, "scene.save first; scene.load second");

		let held = crate::console::take(&mut world, NAMES)
			.pop()
			.as_ref()
			.and_then(Request::of);

		assert_eq!(
			held,
			Some(Request::Load("second".to_owned())),
			"the last thing typed is the one that was meant"
		);
	}

	#[test]
	fn a_line_that_is_not_a_scenes_is_left_where_it_waits() {
		let mut world = console();
		world
			.cvars
			.command("net.later", colby_core::abi::console::defer, "");

		colby_core::abi::console::run(&mut world, "net.later; scene.save one");

		let (mut world, mut simulation) = {
			let (mut peopled, simulation) = peopled();
			std::mem::swap(&mut peopled.asked, &mut world.asked);

			(peopled, simulation)
		};
		let root = project("left");

		serve(&mut world, &mut simulation, &root);

		assert_eq!(world.asked.len(), 1, "the other subsystem's line is still there");
		assert_eq!(world.asked[0].name, "net.later");
		assert!(
			path(&root.saves(), "one")
				.expect("a plain name")
				.is_file(),
			"and the scene's own line was served"
		);

		drop(fs::remove_dir_all(root.root()));
	}

	#[test]
	fn a_world_written_out_reads_back_as_itself() {
		let (world, _simulation) = peopled();
		let root = project("round_trip");
		let name = "quicksave";

		save(&world, &root, name).expect("it is written");
		let read = read(&root, name).expect("and read back");

		assert_eq!(read, scene::capture(&world), "the file is the world");

		drop(fs::remove_dir_all(root.root()));
	}

	#[test]
	fn a_save_nobody_wrote_is_an_error_rather_than_an_empty_world() {
		let (mut world, mut simulation) = peopled();
		let before = scene::capture(&world);

		let root = project("no_such");
		let failure = load(&mut world, &mut simulation, &root, "nothing_here")
			.expect_err("there is no such file");

		assert!(failure.to_string().contains("nothing_here"), "the message names it: {failure}");
		assert_eq!(scene::capture(&world), before, "and the world is untouched");
	}
}
