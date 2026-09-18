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
use colby_asset::{Project, import, level, material, obj, scene as file};
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
/// Also not a choice: a game that keeps a catalogue of props finds them by
/// walking the scene registry for everything named `props/`, and the compiler
/// names an asset by its own path. So a contraption written here is in that
/// catalogue on the next run with nothing told about it.
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

/// `material.write <name>` - writes one registered material back as a source.
///
/// **The odd one out beside [`PROP`]**, and for the same reason: it is about
/// something in a registry rather than about the world. What makes it this
/// module's is the only thing all five share - a name, and a file to put under
/// it.
pub(crate) const MATERIAL: &str = "material.write";

/// `model.write <name>` - starts an import sidecar beside a model's source.
///
/// The sixth, and the one that writes the *least*: a sidecar that says nothing
/// changes nothing, and the whole of what this does is put a file with the
/// right name in the right place so that somebody has one to edit. What it
/// refuses is overwriting one that is already there, because everything worth
/// keeping in one of these was typed by hand. @ref `colby_asset::import`.
pub(crate) const MODEL: &str = "model.write";

/// `blocks.write <name>` - writes a bake's meshes out as sources.
///
/// The seventh, and the only one that writes **geometry**. Every other source
/// in this project was typed by a person; a bake makes one, and it has to land
/// somewhere the compiler will pick it up - which rules out the derived tree,
/// whose pruning deletes an output with no source within a quarter of a second.
///
/// **The mesh crosses through the registry rather than through the line.** The
/// editor builds the geometry and registers it under `maps/<name>/<material>`;
/// this walks the registry for that prefix and writes each one as an `.obj`.
/// So the console carries a name, which is what it is for, and the data crosses
/// on a surface both sides already share. @ref `colby_editor::bake`.
pub(crate) const BLOCKS: &str = "blocks.write";

/// The seven names this module answers for, as they wait on the world.
const NAMES: &[&str] = &[SAVE, LOAD, WRITE, PROP, MATERIAL, MODEL, BLOCKS];

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

	/// Write one *registered material* back as the source it came from.
	///
	/// A material is the one asset the editor can change in place: the
	/// inspector writes the registry, and this is what puts the registry back
	/// on disk. The name is the asset's own, `materials/brass`, so where it
	/// goes is decided by what it is called and nothing has to be asked.
	Material(String),

	/// Start an import sidecar beside a model's source.
	///
	/// The second of the two that are about an asset rather than about the
	/// world, and unlike the first it takes nothing *from* the world: the
	/// numbers in a sidecar are not in a registry anywhere, because what they
	/// describe is how the file was read rather than what came out of it. So
	/// this writes a file that says nothing, and the editing happens in the
	/// file. The name is the model's own, `models/lamp`.
	Model(String),

	/// Write a bake's meshes out as `.obj` sources.
	///
	/// The third about an asset rather than about the world, and the only one
	/// that writes geometry. What it takes from the world is the mesh registry,
	/// which is where the editor left what it built.
	Blocks(String),
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
			| MATERIAL => Some(Self::Material(name)),
			| MODEL => Some(Self::Model(name)),
			| BLOCKS => Some(Self::Blocks(name)),
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
		| Request::Material(name) => material(world, project, name),
		| Request::Model(name) => model(project, name),
		| Request::Blocks(name) => blocks(world, project, name),
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
/// `.scene` in the directory a game's catalogue already walks, so the next run
/// finds it beside the ones written by hand. Nothing new reads it and nothing
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
pub(crate) fn written(world: &World, path: &Path) -> Result<usize> {
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
pub(crate) fn source(assets: &Path, name: &str) -> Result<PathBuf> {
	Ok(assets
		.join(SOURCES)
		.join(plain(name)?)
		.with_extension(level::EXTENSION))
}

/// Writes one registered material back as the `.material` it came from.
///
/// **The asset's own name is the path**, which is what makes this need nothing
/// asked: `materials/brass` is `assets/materials/brass.material`, and a
/// material that came out of a model - `models/lamp/brass` - would land under
/// `assets/models/`, where the compiler would then find a source beside a
/// model that also declares it. So one that is not under `materials/` is
/// refused rather than written somewhere surprising.
///
/// @param world - the registry to take it from
/// @param project - whose asset tree
/// @param name - the asset name
///
/// # Errors
///
/// If nothing answers to the name, if the name is not a material source's, or
/// if the file cannot be written.
fn material(world: &World, project: &Project, name: &str) -> Result {
	// through the handle rather than straight to `get`, because a registry's
	// null entry answers every handle it does not have: `find` on a name
	// nobody registered is `NONE`, and `NONE` reads back as the default
	// material rather than as nothing.
	let found = world.materials.find(name);
	let Some(held) = found
		.is_some()
		.then(|| world.materials.get(found))
		.flatten()
		.copied()
	else {
		return Err(err!(Asset("nothing in this world is called {name}")));
	};

	// the prefix is stripped and the rest checked as a plain name: a material
	// out of a model is called `models/lamp/brass`, and writing that under
	// `assets/models/` would put a source beside a model that also declares
	// it - two producers of one asset, quietly.
	let Some(stem) = name.strip_prefix(MATERIALS) else {
		return Err(err!(Asset(
			"{name} is not under {MATERIALS}; a material a model declared is written by \
			 rebuilding the model"
		)));
	};

	let path = project
		.assets()
		.join(MATERIALS.trim_end_matches('/'))
		.join(plain(stem)?)
		.with_extension(material::SOURCE_EXTENSION);
	let described =
		colby_asset::model::Material::described(name, &held, |id| pictured(world, id));
	let text = material::export(&described)?;

	if let Some(directory) = path.parent() {
		fs::create_dir_all(directory)?;
	}

	fs::write(&path, text.as_bytes())?;
	info!(path = %path.display(), name, "material written as a source");

	Ok(())
}

/// Starts an import sidecar beside a model's source.
///
/// **It writes an empty one, and that is the whole design.** A sidecar's three
/// answers are not in any registry - they are how the file was *read*, not what
/// came out of it - so there is nothing here to take from the world and put on
/// disk the way a material's numbers are taken. What a person needs is a file
/// with the right name in the right place, which is the one thing that is
/// awkward to get right by hand: the name is the source's whole name plus
/// `.model`, and getting it wrong produces a file the compiler never looks at.
///
/// **One that is already there is refused**, because everything worth keeping
/// in one of these was typed by somebody.
///
/// @param project - whose asset tree
/// @param name - the model's asset name, `models/lamp`
///
/// # Errors
///
/// If no source under that name is in the tree, if a sidecar is already beside
/// it, or if the file cannot be written.
fn model(project: &Project, name: &str) -> Result {
	let Some(source) = import::source_of(&project.assets(), name) else {
		return Err(err!(Asset(
			"nothing under assets/ compiles to {name}, so there is nothing to stand beside"
		)));
	};
	let path = import::beside(&source);

	if path.exists() {
		return Err(err!(Asset(
			"{} is already there, and what is in one of these was typed by somebody",
			path.display()
		)));
	}

	let text = import::export(&import::Import::NONE)?;

	fs::write(&path, text.as_bytes())?;
	info!(
		path = %path.display(),
		name,
		source = %source.display(),
		"an import sidecar started; it changes nothing until it is edited"
	);

	Ok(())
}

/// Writes every mesh of a bake out as an `.obj` the compiler will pick up.
///
/// **The registry is the seam.** The editor built the geometry and put it in
/// `World::meshes` under `maps/<name>/<material>`; this finds them by that
/// prefix and writes each one under `assets/`, where the compiler turns it back
/// into a `.cmesh`. Nothing about a mesh crosses the console line - the line
/// carries the name, and both sides already share the registry.
///
/// The path is the asset name, so `maps/room/brass` is
/// `assets/maps/room/brass.obj` and nothing has to be asked. A name that is not
/// under `maps/` is refused for the reason a material outside `materials/` is:
/// writing one anywhere else would put a source beside something that already
/// produces that name.
///
/// @param world - the registry to take the geometry from
/// @param project - whose asset tree
/// @param name - the bake's own name, `room`
///
/// # Errors
///
/// If nothing was baked under that name, or if a file cannot be written.
fn blocks(world: &World, project: &Project, name: &str) -> Result {
	let under = format!("{MAPS}{}/", plain(name)?);
	let baked: Vec<(String, colby_core::abi::mesh::MeshData)> = world
		.meshes
		.iter()
		.filter(|entry| entry.name().starts_with(&under))
		.map(|entry| (entry.name().to_owned(), entry.value().clone()))
		.collect();

	if baked.is_empty() {
		return Err(err!(Asset(
			"nothing is registered under {under}; bake some blocks before writing them"
		)));
	}

	let mut written = 0;

	for (asset, data) in &baked {
		let path = asset
			.split('/')
			.fold(project.assets(), |path, part| path.join(part))
			.with_extension(obj::EXTENSION);

		if let Some(directory) = path.parent() {
			fs::create_dir_all(directory)?;
		}

		fs::write(&path, obj::export(data).as_bytes())?;
		written += 1;

		info!(
			path = %path.display(),
			name = asset,
			vertices = data.vertices.len(),
			triangles = data.triangles(),
			"a baked mesh written as a source"
		);
	}

	info!(name, written, "blocks written");

	Ok(())
}

/// The directory a bake's meshes live in, under the asset tree.
pub(crate) const MAPS: &str = "maps/";

/// What to write down for one of a material's pictures.
///
/// **The built-in stand-in is written as nothing**, and that is the whole
/// point of this being a function rather than a lookup: a material with no
/// normal map holds [`TextureId::FLAT_NORMAL`], because the ABI rewrites
/// `NONE` into it so that the shader needs no branch - so a live record never
/// says "no normal map", it says the name of a texture nobody put in the
/// tree. Writing that back put `"normal": "flat_normal"` into a file somebody
/// had written by hand and left empty, which is how this was found.
///
/// @param world - the registry to name the handle in
/// @param id - the handle
/// @return its asset name, or empty for none and for the stand-in
fn pictured(world: &World, id: colby_core::abi::TextureId) -> String {
	use colby_core::abi::TextureId;

	if !id.is_some() || id == TextureId::FLAT_NORMAL {
		return String::new();
	}

	world
		.textures
		.get(id)
		.map_or_else(String::new, |entry| entry.name().to_owned())
}

/// The directory a material somebody wrote lives in, under the source tree.
///
/// Not a choice: the compiler names an asset by its own path, so a file at
/// `assets/materials/brass.material` is `materials/brass` and there is nowhere
/// else it could be while keeping that name.
pub(crate) const MATERIALS: &str = "materials/";

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
pub(crate) fn plain(name: &str) -> Result<&str> {
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
	fn a_material_is_written_back_as_the_source_it_came_from() {
		let root = project("materials");
		let mut world = World::new();
		let brass = colby_core::abi::Material {
			base_color: Vec3::new(0.85, 0.62, 0.22),
			metallic: 1.0,
			roughness: 0.2,
			..colby_core::abi::Material::DEFAULT
		};
		world.materials.insert("materials/brass", brass);

		material(&world, &root, "materials/brass").expect("it writes");

		let path = root
			.assets()
			.join("materials")
			.join("brass.material");
		let text = fs::read_to_string(&path).expect("the file is there");
		let read = material::import(&text).expect("and reads back");

		assert_eq!(
			read.surface.base_color, brass.base_color,
			"the color survives the round trip"
		);
		assert!((read.surface.metallic - 1.0).abs() < 1.0e-6, "and the metal");
		assert!((read.surface.roughness - 0.2).abs() < 1.0e-6, "and the roughness");
	}

	#[test]
	fn a_material_with_no_normal_map_writes_no_normal_map() {
		// the ABI rewrites `NONE` into `flat_normal` so the shader needs no
		// branch, and a writer that took the live record at its word put the
		// stand-in's name into a file somebody had left empty.
		let root = project("flat_normals");
		let mut world = World::new();
		world
			.materials
			.insert("materials/plain", colby_core::abi::Material::DEFAULT);

		material(&world, &root, "materials/plain").expect("it writes");

		let text = fs::read_to_string(
			root.assets()
				.join("materials")
				.join("plain.material"),
		)
		.expect("the file is there");

		assert!(!text.contains("flat_normal"), "no stand-in is named: {text}");
		assert!(!text.contains("normal"), "and no normal map row at all: {text}");
	}

	#[test]
	fn a_material_a_model_declared_is_not_written_over_the_model() {
		// `models/lamp/brass` would land under `assets/models/`, where the
		// compiler would then find a source beside a model that declares the
		// same name - two producers of one asset, quietly.
		let root = project("model_materials");
		let mut world = World::new();
		world
			.materials
			.insert("models/lamp/brass", colby_core::abi::Material::DEFAULT);

		let refused =
			material(&world, &root, "models/lamp/brass").expect_err("it is not a source's name");

		assert!(
			format!("{refused}").contains("materials/"),
			"and the message says where one lives: {refused}"
		);
	}

	#[test]
	fn a_material_nobody_registered_is_refused_by_name() {
		let refused = material(&World::new(), &project("missing_material"), "materials/nowhere")
			.expect_err("nothing answers to it");

		assert!(
			format!("{refused}").contains("materials/nowhere"),
			"and the message says which: {refused}"
		);
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
	fn a_hidden_parent_saved_and_loaded_is_hidden_again_and_its_child_with_it() {
		// through the real commands and a real file: a save that forgot the
		// word, or a load that forgot to put it back, would show the car again
		let project = project("hidden_round_trip");
		let mut simulation = Box::new(Simulation::new());
		let mut world = Box::new(console());

		world.install_physics(simulation.table());

		let car = world.entities.spawn();
		let wheel = world.entities.spawn();
		assert!(world.entities.set_parent(wheel, car));
		assert!(world.entities.set_hidden(car, true));

		colby_core::abi::console::run(&mut world, "scene.save hidden");
		serve(&mut world, &mut simulation, &project);

		assert!(world.entities.set_hidden(car, false), "shown again before the load");

		colby_core::abi::console::run(&mut world, "scene.load hidden");
		serve(&mut world, &mut simulation, &project);

		assert!(world.entities.hidden(car), "the car came back hidden");
		assert!(!world.entities.shown(wheel), "and the wheel with it, by the car's word");
	}

	#[test]
	fn a_world_saved_with_its_ground_built_comes_back_with_one_terrain_and_one_body() {
		// **the whole file path, which no unit test in `crate::terrain` reaches
		// and no oracle here can see.** A `--shot` run serves console requests
		// once, in its single frame, so a save and a load typed in the same run
		// collapse into whichever was last - the live probe that was meant to
		// drive this proved nothing, and this is what replaced it.
		//
		// What has to hold: the geometry is *not* in the file, the record is,
		// and the sync after the load finds the body the file brought rather
		// than making a second.
		let project = project("terrain_round_trip");
		let mut simulation = Box::new(Simulation::new());
		let mut world = Box::new(console());
		let mut ground = crate::terrain::Ground::new();

		world.install_physics(simulation.table());

		let hill = world.entities.spawn_at(Transform::IDENTITY);
		let terrain = colby_core::abi::Terrain {
			size: 24.0,
			height: 5.0,
			side: 33,
			..colby_core::abi::Terrain::of(19)
		};

		world.entities.set_terrain(hill, terrain);
		world.entities.set_name(hill, "ground");
		crate::terrain::sync(&mut world, &mut ground);

		assert_eq!(world.bodies.len(), 1, "the ground stands before it is saved");

		let mesh = world.meshes.find("terrain.0");
		let before = world
			.meshes
			.get(mesh)
			.map(|entry| entry.value().triangles());

		colby_core::abi::console::run(&mut world, "scene.save hills");
		serve(&mut world, &mut simulation, &project);

		let file = project
			.saves()
			.join("hills")
			.with_extension(colby_asset::scene::EXTENSION);
		let bytes = fs::metadata(&file)
			.expect("a save was written")
			.len();

		// the geometry alone would be about seventy-five kilobytes: a thousand
		// and eighty-nine vertices at forty-eight bytes each, and two thousand
		// and forty-eight triangles at twelve. What is in the file is the
		// record, and everything else in it is the fixed blocks a save has
		// whatever is standing in it.
		assert!(
			bytes < 32 * 1024,
			"a save of a terrain of {} triangles is {bytes} bytes: the geometry got into the \
			 file",
			terrain.triangles()
		);

		colby_core::abi::console::run(&mut world, "scene.load hills");
		serve(&mut world, &mut simulation, &project);

		assert_eq!(
			world.entities.terrain(hill).map(|it| it.kind),
			Some(colby_core::abi::TerrainKind::Noise),
			"the record came back"
		);

		crate::terrain::sync(&mut world, &mut ground);

		assert_eq!(world.bodies.len(), 1, "one body afterwards, not two");
		assert_eq!(
			world
				.meshes
				.get(world.meshes.find("terrain.0"))
				.map(|entry| entry.value().triangles()),
			before,
			"and the same geometry under the same name"
		);
		assert_eq!(
			world.entities.renderable(hill).map(|it| it.mesh),
			Some(mesh),
			"drawn with it, too"
		);
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

	#[test]
	fn starting_a_sidecar_puts_an_empty_one_beside_the_source() {
		let project = project("model-write");
		let models = project.assets().join("models");

		fs::create_dir_all(&models).expect("the tree is made");
		fs::write(models.join("lamp.glb"), b"not really a model, and nothing here reads one")
			.expect("a source");

		model(&project, "models/lamp").expect("it writes");

		let path = models.join("lamp.glb.model");

		assert!(path.is_file(), "beside the source, named after the whole of it");

		let text = fs::read_to_string(&path).expect("and it reads");
		let read = import::import(&text).expect("as a sidecar");

		assert!(
			read.is_silent(),
			"saying nothing, so having started one changes nothing: {text}"
		);
	}

	#[test]
	fn a_sidecar_that_is_already_there_is_not_written_over() {
		// everything worth keeping in one of these was typed by somebody
		let project = project("model-write-twice");
		let models = project.assets().join("models");

		fs::create_dir_all(&models).expect("the tree is made");
		fs::write(models.join("lamp.gltf"), b"a source").expect("a source");

		let path = models.join("lamp.gltf.model");
		let mine = r#"{ "scale": [0.01, 0.01, 0.01] }"#;

		fs::write(&path, mine).expect("and one somebody wrote");

		let refused = model(&project, "models/lamp").expect_err("it refuses");

		assert!(format!("{refused}").contains("already there"), "saying why: {refused}");
		assert_eq!(
			fs::read_to_string(&path).expect("it is still there"),
			mine,
			"and what was typed is untouched"
		);
	}

	#[test]
	fn a_name_no_source_answers_to_starts_nothing() {
		let project = project("model-write-nothing");

		fs::create_dir_all(project.assets().join("models")).expect("the tree is made");

		let refused = model(&project, "models/ghost").expect_err("there is no such source");

		assert!(
			format!("{refused}").contains("nothing to stand beside"),
			"saying why: {refused}"
		);
	}

	#[test]
	fn a_sidecar_is_started_beside_whichever_of_the_three_extensions_is_there() {
		// the panel has a name and the tree has the file; which extension it
		// wears is a question about the tree
		for extension in ["gltf", "glb", "obj"] {
			let project = project(&format!("model-write-{extension}"));
			let models = project.assets().join("models");

			fs::create_dir_all(&models).expect("the tree is made");
			fs::write(models.join(format!("lamp.{extension}")), b"a source").expect("a source");

			model(&project, "models/lamp").expect("it writes");

			assert!(
				models
					.join(format!("lamp.{extension}.model"))
					.is_file(),
				"a {extension} gets one named after itself"
			);
		}
	}

	#[test]
	fn every_name_this_module_waits_on_is_a_command_somebody_can_type() {
		// the gap a live run found and these tests did not: the four before it
		// called `model` straight, so nothing here noticed that `model.write`
		// was in `NAMES`, parsed by `Request::of`, served by `serve` - and
		// registered nowhere, so the console answered "not a command".
		let mut world = World::new();

		crate::console::install(&mut world);

		for name in NAMES {
			assert!(
				world.cvars.get(name).is_some(),
				"{name} is registered, or nobody can ask for it"
			);
		}
	}

	#[test]
	fn a_bake_is_written_out_as_one_obj_per_mesh() {
		let project = project("blocks-write");
		let mut world = World::new();

		world
			.meshes
			.insert("maps/room/default", colby_core::abi::mesh::cube());
		world
			.meshes
			.insert("maps/room/brass", colby_core::abi::mesh::quad());
		// and one that is not this bake's, to prove the prefix is doing work
		world
			.meshes
			.insert("maps/hall/default", colby_core::abi::mesh::cube());

		blocks(&world, &project, "room").expect("it writes");

		let under = project.assets().join("maps").join("room");

		assert!(under.join("default.obj").is_file(), "one file per mesh");
		assert!(under.join("brass.obj").is_file(), "named by its material");
		assert!(
			!project
				.assets()
				.join("maps")
				.join("hall")
				.join("default.obj")
				.is_file(),
			"and another bake's meshes are left alone"
		);

		let text = fs::read_to_string(under.join("default.obj")).expect("it reads");
		let read = obj::import(&text).expect("and it is geometry");

		assert_eq!(
			read.triangles(),
			colby_core::abi::mesh::cube().triangles(),
			"the same geometry the registry held"
		);
	}

	#[test]
	fn writing_a_bake_nobody_made_says_so() {
		let project = project("blocks-write-nothing");
		let world = World::new();

		let refused = blocks(&world, &project, "room").expect_err("nothing is registered");

		assert!(
			format!("{refused}").contains("bake some blocks"),
			"saying what to do about it: {refused}"
		);
	}

	#[test]
	fn a_bake_name_that_would_escape_the_asset_tree_is_refused() {
		let project = project("blocks-write-escape");
		let mut world = World::new();

		world
			.meshes
			.insert("maps/room/default", colby_core::abi::mesh::cube());

		for name in ["../room", "a/b", ""] {
			assert!(
				blocks(&world, &project, name).is_err(),
				"{name:?} is not a name a file may be written under"
			);
		}
	}
}
