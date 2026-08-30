//! The host's asset loop: compile what changed, load what moved.
//!
//! The same shape as the game module's watcher in [`watch`](crate::watch), and
//! for the same reason. Two independent signals: a change under `assets/`
//! recompiles, and a change to a compiled `.cmesh` is what actually reloads a
//! mesh. Keeping them separate means `just assets` run from another terminal
//! reloads exactly as an edit in an editor does - the runner is not the only
//! thing allowed to produce the file.
//!
//! The difference from the module watcher is that there is no subprocess and no
//! restart threshold. Compiling a mesh is a parse and a write, so it happens in
//! this process, on the frame that notices; and geometry is data rather than
//! code, so nothing has to be unloaded before the new version goes in. A
//! reloaded asset keeps its handle - the registry entry is rewritten, not
//! replaced - which is why neither the game nor the renderer has to be told
//! that anything happened. Meshes and textures go through the same loop and
//! differ only in which registry they land in.
//!
//! Failures never stop anything. A source halfway through being saved fails to
//! parse, gets a warning, and is tried again on the next poll; the mesh already
//! on screen stays until a good version replaces it.

use std::{
	path::{Path, PathBuf},
	time::{Duration, Instant, SystemTime},
};

use colby_asset::{
	MeshFile, TextureFile, anim::ClipFile, compile, compile::Kind, document::DocumentFile,
	font::FontFile, model::ModelFile, scene::SceneFile, skeleton::SkeletonFile,
};
use colby_core::{
	abi::{
		ClipData, DocumentData, FontData, Material, MaterialId, MeshData, MeshId, ModelData,
		Placement, SceneData, SkeletonData, SkeletonId, TextureData, TextureId, World,
	},
	debug, info, warn,
};

/// How often the asset trees are looked at.
///
/// The same quarter second the module watcher uses. A compile pass over a few
/// dozen files is a few dozen `stat` calls, which is not worth pacing
/// differently.
const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Overrides the source tree. Mostly for running the executable from
/// elsewhere.
const SOURCE_VAR: &str = "COLBY_ASSETS";

/// Overrides the compiled tree. The one to set for a build that ships its
/// assets beside the executable instead of under `target`.
const OUTPUT_VAR: &str = "COLBY_ASSETS_OUT";

/// The tree sources are read from.
///
/// A function rather than a value worked out in one place, because two things
/// now need the answer: this loop, which watches it, and the editor, which
/// writes a scene source into it. A source written under a different root from
/// the one being watched is a file nothing would ever compile.
pub(crate) fn source_root() -> PathBuf {
	std::env::var_os(SOURCE_VAR)
		.map_or_else(|| compile::source_root(&crate::workspace()), PathBuf::from)
}

/// One compiled asset the world is currently holding.
struct Loaded {
	name: String,
	path: PathBuf,
	kind: Kind,
	stamp: SystemTime,
}

/// The compile-and-load loop's view of the two asset trees.
pub(crate) struct Assets {
	source: PathBuf,
	output: PathBuf,
	next_poll: Instant,
	loaded: Vec<Loaded>,
}

impl Assets {
	/// Points the loop at a workspace's asset trees.
	///
	/// Neither directory has to exist. A workspace with no `assets/` compiles
	/// nothing and loads nothing, which is what a project that has not made an
	/// asset yet looks like.
	///
	/// @param workspace - the checkout the runner was built from
	pub(crate) fn new(workspace: &Path) -> Self {
		let source = source_root();
		let output = std::env::var_os(OUTPUT_VAR)
			.map_or_else(|| compile::output_root(workspace), PathBuf::from);

		info!(source = ?source, output = ?output, "watching the asset tree");

		Self::at(source, output)
	}

	/// Points the loop at two directories directly.
	///
	/// @param source - the tree to compile from
	/// @param output - the tree to compile into and load from
	pub(crate) fn at(source: PathBuf, output: PathBuf) -> Self {
		Self {
			source,
			output,
			next_poll: Instant::now() + POLL_INTERVAL,
			loaded: Vec::new(),
		}
	}

	/// Compiles what is stale and loads what moved, right now.
	///
	/// Called once before the game's `init`, so a game asking for a mesh by
	/// name on its first frame finds it.
	///
	/// @param world - the host state whose registry is filled
	pub(crate) fn sync(&mut self, world: &mut World) {
		self.build();
		self.load(world);
	}

	/// The same, but at most once every [`POLL_INTERVAL`].
	///
	/// @param world - the host state whose registry is kept current
	pub(crate) fn poll(&mut self, world: &mut World) {
		if Instant::now() < self.next_poll {
			return;
		}

		self.next_poll = Instant::now() + POLL_INTERVAL;
		self.sync(world);
	}

	/// Runs the compiler over the source tree.
	///
	/// One source that will not compile is a warning naming the file and the
	/// line; the rest are compiled anyway, and the bad one is retried the next
	/// time it is written.
	fn build(&self) {
		if !self.source.is_dir() {
			return;
		}

		let report = match compile::compile_dir(&self.source, &self.output, false) {
			| Ok(report) => report,
			| Err(error) => {
				warn!(%error, "compiling the asset tree failed");

				return;
			},
		};

		for failure in &report.failed {
			warn!(error = %failure.error, "an asset did not compile; keeping the last good one");
		}

		if report.is_quiet() {
			return;
		}

		for compiled in &report.compiled {
			debug!(
				name = compiled.name,
				bytes = compiled.bytes,
				produced = ?compiled.produced,
				"asset compiled"
			);
		}

		for removed in &report.removed {
			debug!(path = ?removed, "compiled asset removed, its source is gone");
		}
	}

	/// Brings the world's mesh registry level with the compiled tree.
	fn load(&mut self, world: &mut World) {
		let Ok(present) = compile::outputs(&self.output) else {
			return;
		};

		for path in &present {
			self.load_one(world, path);
		}

		self.forget_missing(world, &present);

		// a clip's tracks are matched to a skeleton's bones by name, and this
		// pass is the one moment either of them can be rewritten. Without it,
		// a rig edited to gain a bone leaves every clip playing on it one bone
		// out - and silently, because every index still resolves to something.
		world.clips.relink(&world.skeletons);
	}

	/// Loads one compiled asset if it is new or has moved since it was last
	/// read.
	fn load_one(&mut self, world: &mut World, path: &Path) {
		let Ok(stamp) = mtime(path) else {
			return;
		};

		let known = self
			.loaded
			.iter()
			.position(|loaded| loaded.path == *path);

		if known.is_some_and(|index| self.loaded[index].stamp >= stamp) {
			return;
		}

		let (Ok(name), Some(kind)) =
			(compile::asset_name(&self.output, path), Kind::of_output(path))
		else {
			return;
		};

		// recorded whether or not the read works. A file that cannot be read
		// will not become readable on its own, and re-reading it four times a
		// second would fill the log; the next write moves the stamp and it is
		// tried again.
		match known {
			| Some(index) => self.loaded[index].stamp = stamp,
			| None => self.loaded.push(Loaded {
				name: name.clone(),
				path: path.to_path_buf(),
				kind,
				stamp,
			}),
		}

		match kind {
			| Kind::Mesh => load_mesh(world, path, &name),
			| Kind::Texture => load_texture(world, path, &name),
			| Kind::Font => load_font(world, path, &name),
			| Kind::Document => load_document(world, path, &name),
			| Kind::Model => load_model(world, path, &name),
			| Kind::Scene => load_scene(world, path, &name),
			| Kind::Skeleton => load_skeleton(world, path, &name),
			| Kind::Clip => load_clip(world, path, &name),
		}
	}

	/// Empties the registry entries whose file has been deleted.
	///
	/// The entry itself stays - a handle the game is holding has to keep
	/// resolving - but it stops drawing, which is the honest picture of an
	/// asset that no longer exists.
	fn forget_missing(&mut self, world: &mut World, present: &[PathBuf]) {
		let mut gone = Vec::new();
		self.loaded.retain(|loaded| {
			if present.contains(&loaded.path) {
				return true;
			}

			gone.push((loaded.name.clone(), loaded.kind));

			false
		});

		for (name, kind) in gone {
			match kind {
				| Kind::Mesh => drop(world.meshes.insert(&name, MeshData::default())),
				| Kind::Texture => drop(world.textures.insert(&name, TextureData::white())),
				| Kind::Font => drop(world.fonts.insert(&name, FontData::empty())),
				| Kind::Document => drop(world.ui.insert(&name, DocumentData::empty())),
				| Kind::Model => drop(world.models.insert(&name, ModelData::default())),
				| Kind::Scene => drop(world.scenes.insert(&name, SceneData::default())),
				| Kind::Skeleton => drop(
					world
						.skeletons
						.insert(&name, SkeletonData::default()),
				),
				| Kind::Clip => drop(world.clips.insert(&name, ClipData::default())),
			}

			info!(name, ?kind, "asset unloaded; its file is gone");
		}
	}
}

/// Reads one `.cmesh` into the world's mesh registry.
fn load_mesh(world: &mut World, path: &Path, name: &str) {
	let file = match MeshFile::open(path) {
		| Ok(file) => file,
		| Err(error) => {
			warn!(%error, "the mesh on disk could not be read");

			return;
		},
	};

	let data = file.to_mesh_data();
	let existing = world.meshes.find(name);
	if existing.is_some()
		&& world
			.meshes
			.get(existing)
			.is_some_and(|mesh| *mesh.value() == data)
	{
		// the file moved but its contents did not: an editor rewriting what it
		// did not change, or a filesystem whose timestamps are coarse enough
		// that every poll looks like an edit. Registering it anyway would bump
		// the revision and make the renderer re-upload for nothing, four times
		// a second.
		return;
	}

	let (min, max) = file.bounds();
	let (triangles, vertices) = (data.triangles(), data.vertices.len());
	let id = world.meshes.insert(name, data);

	info!(
		name,
		slot = id.index(),
		triangles,
		vertices,
		bounds = format!("{min} .. {max}"),
		"mesh loaded"
	);
}

/// Reads one `.ctex` into the world's texture registry.
fn load_texture(world: &mut World, path: &Path, name: &str) {
	let file = match TextureFile::open(path) {
		| Ok(file) => file,
		| Err(error) => {
			warn!(%error, "the texture on disk could not be read");

			return;
		},
	};

	let data = file.to_texture_data();
	if !data.is_consistent() {
		warn!(name, "the texture's levels do not add up; keeping the last good one");

		return;
	}

	let existing = world.textures.find(name);
	if existing.is_some()
		&& world
			.textures
			.get(existing)
			.is_some_and(|texture| *texture.value() == data)
	{
		// as above: the same pixels do not need re-uploading.
		return;
	}

	let (width, height, levels) = (data.width, data.height, data.levels.len());
	let id = world.textures.insert(name, data);

	info!(name, slot = id.index(), width, height, levels, "texture loaded");
}

/// Reads one `.cfont` into the world's font registry.
fn load_font(world: &mut World, path: &Path, name: &str) {
	let file = match FontFile::open(path) {
		| Ok(file) => file,
		| Err(error) => {
			warn!(%error, "the font on disk could not be read");

			return;
		},
	};

	let data = file.to_font_data();

	let existing = world.fonts.find(name);
	if existing.is_some()
		&& world
			.fonts
			.get(existing)
			.is_some_and(|font| *font.value() == data)
	{
		// as the other two: an atlas that has not changed does not need
		// uploading again, and the revision is what the interface watches.
		return;
	}

	let (glyphs, width, height) = (data.glyphs.len(), data.atlas_width, data.atlas_height);
	let id = world.fonts.insert(name, data);

	info!(name, slot = id.index(), glyphs, width, height, "font loaded");
}

/// Reads one `.cdoc` into the world's document table.
///
/// Parsed here rather than by the compiler, because a `.cdoc` is text: the
/// compiler's job was to fold in the stylesheets and to have complained about
/// anything unreadable already. @ref `colby_asset::document`.
fn load_document(world: &mut World, path: &Path, name: &str) {
	let data = match DocumentFile::open(path).and_then(|file| file.to_document_data()) {
		| Ok(data) => data,
		| Err(error) => {
			warn!(%error, "the document on disk could not be read");

			return;
		},
	};

	let existing = world.ui.find(name);
	if existing.is_some()
		&& world
			.ui
			.document(existing)
			.is_some_and(|document| *document.value() == data)
	{
		// as the other three: an unchanged document does not need laying out
		// again, and the revision is what the interface watches.
		return;
	}

	let (nodes, rules) = (data.nodes.len(), data.rules.len());
	let id = world.ui.insert(name, data);

	info!(name, slot = id.index(), nodes, rules, "document loaded");
}

/// Reads one `.cscene` into the world's scene table.
///
/// The one loader that resolves nothing. Every other asset arrives with its
/// references turned into handles here, because a mesh is drawn and a texture
/// is sampled and both need one now. A scene is neither: it is a description
/// somebody hands to a loader later, and what its names resolve to depends on
/// the world it is being put into rather than on this one. So the description
/// is stored as it was written, names and all.
fn load_scene(world: &mut World, path: &Path, name: &str) {
	let data = match SceneFile::open(path) {
		| Ok(file) => file.to_scene_data(),
		| Err(error) => {
			warn!(%error, "the scene on disk could not be read");

			return;
		},
	};

	let existing = world.scenes.find(name);
	if existing.is_some()
		&& world
			.scenes
			.get(existing)
			.is_some_and(|scene| *scene.value() == data)
	{
		// as the other loaders: an unchanged scene is not worth a revision,
		// and a game watching one for changes should not see one.
		return;
	}

	let (entities, bodies, joints) = (data.things.len(), data.solids.len(), data.links.len());
	let id = world.scenes.insert(name, data);

	info!(name, slot = id.index(), entities, bodies, joints, "scene loaded");
}

/// Reads one `.cmodel` into the world's model table, and its materials into
/// the material table.
///
/// **A placement's handles are resolved once and never again, so a name that
/// is not loaded yet is *reserved* rather than left as nothing.** An empty
/// entry claims the slot and the real asset overwrites it later; the handle
/// the placement already carries is right either way, because a registry entry
/// keeps its slot for the life of the process.
///
/// As it happens the walk usually reaches a model's meshes first - a path
/// sorts by component, and `models/lamp` comes before `models/lamp.cmodel`.
/// That is luck rather than a rule, and it is not the kind worth depending on:
/// a mesh that arrives after its model would otherwise leave the model
/// standing on nothing forever, because nothing re-resolves a placement.
///
/// A model's materials land in `World::materials` beside the game's own. They
/// are named inside the model's own path, so nothing a game declares can
/// collide with one.
fn load_model(world: &mut World, path: &Path, name: &str) {
	let data = match ModelFile::open(path) {
		| Ok(file) => file.to_model_data(),
		| Err(error) => {
			warn!(%error, "the model on disk could not be read");

			return;
		},
	};

	for material in &data.materials {
		let albedo = reserve_texture(world, &material.albedo);
		let normal = reserve_texture(world, &material.normal);

		// `bumped` is what turns a material with no map into one holding the
		// flat one, which is the ABI's own rewrite and not this loader's.
		world.materials.insert(&material.name, Material {
			base_color: material.base_color,
			wrap: material.wrap,
			..Material::textured(albedo)
				.bumped(normal)
				.finished(material.metallic, material.roughness)
		});
	}

	let placements = data
		.placements
		.iter()
		.map(|placement| Placement {
			name: placement.name.clone(),
			mesh: reserve_mesh(world, &placement.mesh),
			material: if placement.material.is_empty() {
				MaterialId::DEFAULT
			} else {
				world.materials.find(&placement.material)
			},
			skeleton: reserve_skeleton(world, &placement.skeleton),
			transform: placement.transform,
		})
		.collect();
	let loaded = ModelData { placements };
	let existing = world.models.find(name);

	if existing.is_some()
		&& world
			.models
			.get(existing)
			.is_some_and(|model| *model.value() == loaded)
	{
		// as the other four: a model that stands what it already stood does not
		// need registering again, and its revision is what anything watching it
		// would otherwise see move for nothing.
		return;
	}

	let standing = loaded.placements.len();
	let materials = data.materials.len();
	let id = world.models.insert(name, loaded);

	info!(name, slot = id.index(), standing, materials, "model loaded");
}

/// Reads one `.canim` into the world's clip registry.
fn load_clip(world: &mut World, path: &Path, name: &str) {
	let data = match ClipFile::open(path) {
		| Ok(file) => file.to_clip_data(),
		| Err(error) => {
			warn!(%error, "the clip on disk could not be read");

			return;
		},
	};
	let existing = world.clips.find(name);

	if existing.is_some()
		&& world
			.clips
			.get(existing)
			.is_some_and(|clip| *clip.value() == data)
	{
		// as every other loader: a clip whose keys did not move does not need
		// registering again, and its revision is what the bindings watching it
		// would otherwise see move for nothing.
		return;
	}

	let tracks = data.len();
	let keys = data.keys();
	let seconds = data.duration();
	let id = world.clips.insert(name, data);

	info!(name, slot = id.index(), tracks, keys, seconds, "clip loaded");
}

/// Reads one `.cskel` into the world's skeleton registry.
fn load_skeleton(world: &mut World, path: &Path, name: &str) {
	let data = match SkeletonFile::open(path) {
		| Ok(file) => file.to_skeleton_data(),
		| Err(error) => {
			warn!(%error, "the skeleton on disk could not be read");

			return;
		},
	};
	let existing = world.skeletons.find(name);

	if existing.is_some()
		&& world
			.skeletons
			.get(existing)
			.is_some_and(|skeleton| *skeleton.value() == data)
	{
		// as every other loader: a skeleton whose bones did not move does not
		// need registering again, and its revision is what anything watching
		// it would otherwise see move for nothing.
		return;
	}

	let bones = data.len();
	let id = world.skeletons.insert(name, data);

	info!(name, slot = id.index(), bones, "skeleton loaded");
}

/// The handle a name will resolve to, claiming the slot if nothing has yet.
///
/// The same reservation a mesh gets, for the same reason: a model may be
/// walked before the skeleton it names, and an entry claimed now is the one
/// the real file overwrites later. Without it a model read first would stand
/// on nothing for the life of the process.
fn reserve_skeleton(world: &mut World, name: &str) -> SkeletonId {
	if name.is_empty() {
		return SkeletonId::NONE;
	}

	let found = world.skeletons.find(name);

	if found.is_some() {
		return found;
	}

	world
		.skeletons
		.insert(name, SkeletonData::default())
}

/// The handle of a mesh a model names, claiming the slot when it is not loaded.
fn reserve_mesh(world: &mut World, name: &str) -> MeshId {
	if name.is_empty() {
		return MeshId::NONE;
	}

	let found = world.meshes.find(name);

	if found.is_some() {
		return found;
	}

	world.meshes.insert(name, MeshData::default())
}

/// The same for a picture. An empty name is what a material with none wrote.
fn reserve_texture(world: &mut World, name: &str) -> TextureId {
	if name.is_empty() {
		return TextureId::NONE;
	}

	let found = world.textures.find(name);

	if found.is_some() {
		return found;
	}

	world.textures.insert(name, TextureData::white())
}

/// A file's modification time.
fn mtime(path: &Path) -> std::io::Result<SystemTime> { path.metadata()?.modified() }

#[cfg(test)]
mod tests {
	use std::{fs, thread::sleep};

	use colby_core::abi::{Clip, Mesh, Model, Texture};

	use super::*;

	/// A two-by-two truecolor PNG: red, green over blue, white.
	const RGB_QUAD: [u8; 77] = [
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
		0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x08, 0x02, 0x00, 0x00, 0x00, 0xFD,
		0xD4, 0x9A, 0x73, 0x00, 0x00, 0x00, 0x14, 0x49, 0x44, 0x41, 0x54, 0x78, 0xDA, 0x63, 0xF8,
		0xCF, 0xC0, 0xC0, 0x00, 0xC2, 0x0C, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x00, 0x1F, 0xEE, 0x05,
		0xFB, 0x60, 0x6C, 0x70, 0xF2, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42,
		0x60, 0x82,
	];

	/// The same size in grayscale: black, white over white, black.
	const GREY_QUAD: [u8; 71] = [
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
		0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x08, 0x00, 0x00, 0x00, 0x00, 0x57,
		0xDD, 0x52, 0xF8, 0x00, 0x00, 0x00, 0x0E, 0x49, 0x44, 0x41, 0x54, 0x78, 0xDA, 0x63, 0x60,
		0xF8, 0xCF, 0xF0, 0x9F, 0x01, 0x00, 0x06, 0x00, 0x01, 0xFF, 0x92, 0x99, 0xB2, 0xEC, 0x00,
		0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
	];

	/// Writes bytes into a fixture tree.
	fn put_bytes(source: &Path, relative: &str, bytes: &[u8]) {
		let path = source.join(relative);
		if let Some(parent) = path.parent() {
			fs::create_dir_all(parent).expect("the directory is made");
		}

		fs::write(&path, bytes).expect("the source is written");
	}

	/// A triangle, as OBJ. `size` moves one corner, so two versions of it are
	/// the same shape at different scales.
	fn triangle(size: f32) -> String { format!("v 0 0 0\nv {size} 0 0\nv 0 {size} 0\nf 1 2 3\n") }

	/// A quad, as OBJ: a different triangle count, so a reload is visible in
	/// one number.
	const QUAD: &str = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";

	/// Two empty directories nobody else is using.
	fn trees(name: &str) -> (PathBuf, PathBuf) {
		let root = std::env::temp_dir()
			.join("colby-asset-loop-tests")
			.join(name);

		drop(fs::remove_dir_all(&root));
		fs::create_dir_all(root.join("src")).expect("the fixture is made");

		(root.join("src"), root.join("out"))
	}

	/// Writes a source into a fixture tree.
	fn put(source: &Path, relative: &str, text: &str) {
		let path = source.join(relative);
		if let Some(parent) = path.parent() {
			fs::create_dir_all(parent).expect("the directory is made");
		}

		fs::write(&path, text).expect("the source is written");
	}

	#[test]
	fn a_source_is_compiled_and_registered_under_its_path() {
		let (source, output) = trees("first-load");
		put(&source, "meshes/thing.obj", &triangle(1.0));

		let mut world = World::new();
		Assets::at(source, output).sync(&mut world);

		let id = world.meshes.find("meshes/thing");

		assert!(id.is_some(), "the mesh reached the registry");
		assert_ne!(id, MeshId::CUBE, "in a slot of its own, after the primitives");
		assert_eq!(
			world
				.meshes
				.get(id)
				.map(|mesh| mesh.value().triangles()),
			Some(1),
			"with the geometry the source described"
		);
	}

	#[test]
	fn editing_a_source_reloads_the_mesh_without_moving_its_handle() {
		let (source, output) = trees("reload");
		put(&source, "meshes/thing.obj", &triangle(1.0));

		let mut world = World::new();
		let mut assets = Assets::at(source.clone(), output);
		assets.sync(&mut world);

		let before = world.meshes.find("meshes/thing");

		assert_eq!(
			world.meshes.get(before).map(Mesh::revision),
			Some(0),
			"loaded once, replaced never"
		);

		// the mtime rule is `newer than`, and the two writes are microseconds
		// apart on a filesystem with finer resolution than that.
		sleep(Duration::from_millis(20));
		put(&source, "meshes/thing.obj", QUAD);
		assets.sync(&mut world);

		let after = world.meshes.find("meshes/thing");

		assert_eq!(after, before, "the handle survives, so nothing holding it has to be told");
		assert_eq!(
			world
				.meshes
				.get(after)
				.map(|mesh| mesh.value().triangles()),
			Some(2),
			"and the geometry behind it is the edited one"
		);
		assert_eq!(
			world.meshes.get(after).map(Mesh::revision),
			Some(1),
			"which is what the renderer compares against to know it must re-upload"
		);
	}

	#[test]
	fn a_source_that_has_not_changed_is_not_reloaded() {
		let (source, output) = trees("quiet");
		put(&source, "meshes/thing.obj", &triangle(1.0));

		let mut world = World::new();
		let mut assets = Assets::at(source, output);
		assets.sync(&mut world);
		assets.sync(&mut world);
		assets.sync(&mut world);

		let id = world.meshes.find("meshes/thing");

		assert_eq!(
			world.meshes.get(id).map(Mesh::revision),
			Some(0),
			"three passes over an untouched tree upload nothing"
		);
	}

	#[test]
	fn rewriting_a_source_without_changing_it_uploads_nothing() {
		let (source, output) = trees("rewritten");
		put(&source, "meshes/thing.obj", &triangle(1.0));

		let mut world = World::new();
		let mut assets = Assets::at(source.clone(), output);
		assets.sync(&mut world);

		let id = world.meshes.find("meshes/thing");

		// the same bytes, a newer timestamp. Everything downstream of the
		// timestamp fires; the registry is the thing that has to notice.
		sleep(Duration::from_millis(20));
		put(&source, "meshes/thing.obj", &triangle(1.0));
		assets.sync(&mut world);

		assert_eq!(
			world.meshes.get(id).map(Mesh::revision),
			Some(0),
			"a save that changed nothing must not make the renderer re-upload"
		);
	}

	#[test]
	fn a_source_that_stops_compiling_leaves_the_last_good_mesh_alone() {
		let (source, output) = trees("broken");
		put(&source, "meshes/thing.obj", &triangle(1.0));

		let mut world = World::new();
		let mut assets = Assets::at(source.clone(), output);
		assets.sync(&mut world);

		let id = world.meshes.find("meshes/thing");

		sleep(Duration::from_millis(20));
		put(&source, "meshes/thing.obj", "v 0 0 0\nf 1 7 9\n");
		assets.sync(&mut world);

		assert_eq!(
			world
				.meshes
				.get(id)
				.map(|mesh| mesh.value().triangles()),
			Some(1),
			"a half-saved or broken edit does not empty the screen"
		);
		assert_eq!(world.meshes.get(id).map(Mesh::revision), Some(0), "and nothing was replaced");

		// and a good edit after a bad one still lands.
		sleep(Duration::from_millis(20));
		put(&source, "meshes/thing.obj", QUAD);
		assets.sync(&mut world);

		assert_eq!(
			world
				.meshes
				.get(id)
				.map(|mesh| mesh.value().triangles()),
			Some(2),
			"the next good version replaces it"
		);
	}

	#[test]
	fn deleting_a_source_empties_the_mesh_it_registered() {
		let (source, output) = trees("deleted");
		put(&source, "meshes/thing.obj", &triangle(1.0));

		let mut world = World::new();
		let mut assets = Assets::at(source.clone(), output);
		assets.sync(&mut world);

		let id = world.meshes.find("meshes/thing");

		fs::remove_file(source.join("meshes").join("thing.obj")).expect("the source is deleted");
		assets.sync(&mut world);

		assert_eq!(world.meshes.find("meshes/thing"), id, "the handle still resolves");
		assert_eq!(
			world
				.meshes
				.get(id)
				.map(|mesh| mesh.value().is_empty()),
			Some(true),
			"but it draws nothing, which is what an asset that is gone looks like"
		);
	}

	#[test]
	fn a_png_is_compiled_and_registered_as_a_texture() {
		let (source, output) = trees("first-texture");
		put_bytes(&source, "textures/quad.png", &RGB_QUAD);

		let mut world = World::new();
		Assets::at(source, output).sync(&mut world);

		let id = world.textures.find("textures/quad");

		assert!(id.is_some(), "the texture reached the registry");
		assert_ne!(id, TextureId::NONE, "in a slot of its own, after the white one");
		assert_eq!(
			world
				.textures
				.get(id)
				.map(|texture| (texture.value().width, texture.value().levels.len())),
			Some((2, 2)),
			"two by two, with the level below it"
		);
	}

	#[test]
	fn editing_an_image_reloads_the_texture_without_moving_its_handle() {
		let (source, output) = trees("reload-texture");
		put_bytes(&source, "textures/quad.png", &RGB_QUAD);

		let mut world = World::new();
		let mut assets = Assets::at(source.clone(), output);
		assets.sync(&mut world);

		let before = world.textures.find("textures/quad");

		assert_eq!(
			world.textures.get(before).map(Texture::revision),
			Some(0),
			"loaded once, replaced never"
		);

		sleep(Duration::from_millis(20));
		put_bytes(&source, "textures/quad.png", &GREY_QUAD);
		assets.sync(&mut world);

		let after = world.textures.find("textures/quad");

		assert_eq!(after, before, "the handle survives, so materials need not be told");
		assert_eq!(
			world.textures.get(after).map(Texture::revision),
			Some(1),
			"and the revision is what the renderer compares"
		);
		assert_eq!(
			world
				.textures
				.get(after)
				.map(|texture| texture.value().levels[0][0]),
			Some(0x00),
			"and the pixels really are the grey image, whose first texel is black"
		);
	}

	#[test]
	fn meshes_and_textures_go_through_the_same_loop() {
		let (source, output) = trees("both-kinds");
		put(&source, "meshes/thing.obj", &triangle(1.0));
		put_bytes(&source, "textures/quad.png", &RGB_QUAD);

		let mut world = World::new();
		Assets::at(source, output).sync(&mut world);

		assert!(world.meshes.find("meshes/thing").is_some(), "the mesh landed");
		assert!(world.textures.find("textures/quad").is_some(), "and so did the texture");
	}

	#[test]
	fn deleting_an_image_leaves_its_texture_white_rather_than_broken() {
		let (source, output) = trees("deleted-texture");
		put_bytes(&source, "textures/quad.png", &RGB_QUAD);

		let mut world = World::new();
		let mut assets = Assets::at(source.clone(), output);
		assets.sync(&mut world);

		let id = world.textures.find("textures/quad");

		fs::remove_file(source.join("textures").join("quad.png")).expect("it is deleted");
		assets.sync(&mut world);

		assert_eq!(world.textures.find("textures/quad"), id, "the handle still resolves");
		assert_eq!(
			world
				.textures
				.get(id)
				.map(|texture| texture.value().levels[0].clone()),
			Some(vec![0xFF, 0xFF, 0xFF, 0xFF]),
			"and samples as white, so a material using it keeps its own color"
		);
	}

	#[test]
	fn a_workspace_with_no_assets_directory_is_not_a_problem() {
		let (source, output) = trees("empty");
		drop(fs::remove_dir_all(&source));

		let mut world = World::new();
		Assets::at(source, output).sync(&mut world);

		assert_eq!(world.meshes.len(), 4, "the built-in primitives and nothing else");
		assert_eq!(
			world.textures.len(),
			3,
			"and the null texture, the white one and the flat normal map"
		);
	}

	/// A rig of two bones with one animation over it, and no geometry at all.
	///
	/// Written as text with its buffer inside it so that there is one more
	/// binary fixture than there needs to be, which is none. The buffer is
	/// three key times and then three turns.
	const MOVES: &str = r#"{
		"asset": { "version": "2.0" },
		"buffers": [{ "byteLength": 60, "uri": "data:application/octet-stream;base64,AAAAAAAAgD8AAABAAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAA8wQ1P/MENT8AAAAAAAAAAAAAAAAAAIA/" }],
		"bufferViews": [
			{ "buffer": 0, "byteOffset": 0, "byteLength": 12 },
			{ "buffer": 0, "byteOffset": 12, "byteLength": 48 }
		],
		"accessors": [
			{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "SCALAR" },
			{ "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC4" }
		],
		"nodes": [
			{ "name": "hips", "children": [1] },
			{ "name": "spine" }
		],
		"skins": [{ "name": "rig", "joints": [0, 1] }],
		"animations": [{
			"name": "walk",
			"samplers": [{ "input": 0, "output": 1, "interpolation": "LINEAR" }],
			"channels": [{ "sampler": 0, "target": { "node": 1, "path": "rotation" } }]
		}]
	}"#;

	/// The world after a fixture tree holding one animated rig is compiled.
	fn with_moves(name: &str) -> (World, Assets) {
		let (source, output) = trees(name);

		put(&source, "models/moves.gltf", MOVES);

		let mut world = World::new();
		let mut assets = Assets::at(source, output);

		assets.sync(&mut world);

		(world, assets)
	}

	#[test]
	fn a_clip_beside_a_model_reaches_the_registry_with_its_keys() {
		let (world, _) = with_moves("clip-load");
		let id = world.clips.find("models/moves/walk");

		assert!(id.is_some(), "the clip reached the registry under the model's own name");

		let clip = world.clips.data(id);

		assert_eq!(clip.len(), 1, "one channel, one track");
		assert_eq!(clip.tracks[0].bone, "spine", "naming the bone the skin named");
		assert_eq!(clip.tracks[0].keys(), 3, "three keys");
		assert!((clip.duration() - 2.0).abs() < 1.0e-6, "running two seconds");
	}

	#[test]
	fn a_clip_binds_to_the_skeleton_that_was_written_beside_it() {
		let (mut world, _) = with_moves("clip-bind");
		let clip = world.clips.find("models/moves/walk");
		let skeleton = world.skeletons.find("models/moves/rig");

		assert!(skeleton.is_some(), "the rig reached the registry too");

		world.clips.bind(clip, skeleton, &world.skeletons);

		assert_eq!(
			world.clips.bones(clip, skeleton),
			&[1],
			"the track lands on the spine, which is the second bone of the rig"
		);
	}

	#[test]
	fn a_binding_survives_another_pass_over_a_tree_that_did_not_change() {
		// the relink at the end of every pass is what keeps a binding honest
		// when a rig is recompiled, and what it must not do is throw one away
		// because nothing happened.
		let (mut world, mut assets) = with_moves("clip-relink");
		let clip = world.clips.find("models/moves/walk");
		let skeleton = world.skeletons.find("models/moves/rig");

		world.clips.bind(clip, skeleton, &world.skeletons);
		assets.sync(&mut world);

		assert_eq!(world.clips.bindings(), 1, "the binding is still the one binding");
		assert_eq!(world.clips.bones(clip, skeleton), &[1], "and still points at the spine");
	}

	#[test]
	fn deleting_a_clip_leaves_its_handle_resolving_to_an_animation_of_nothing() {
		let (source, output) = trees("clip-delete");

		put(&source, "models/moves.gltf", MOVES);

		let mut world = World::new();
		let mut assets = Assets::at(source.clone(), output);

		assets.sync(&mut world);

		let id = world.clips.find("models/moves/walk");

		assert!(!world.clips.data(id).is_empty(), "there is a clip there to begin with");

		fs::remove_file(source.join("models").join("moves.gltf")).expect("it is deleted");
		assets.sync(&mut world);

		assert_eq!(world.clips.find("models/moves/walk"), id, "the handle still resolves");
		assert!(
			world.clips.data(id).is_empty(),
			"and plays nothing, which is the honest picture of a clip that is gone"
		);
	}

	/// The same rig with a bone above it, so every bone below is renumbered.
	const MOVES_TALLER: &str = r#"{
		"asset": { "version": "2.0" },
		"buffers": [{ "byteLength": 60, "uri": "data:application/octet-stream;base64,AAAAAAAAgD8AAABAAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAA8wQ1P/MENT8AAAAAAAAAAAAAAAAAAIA/" }],
		"bufferViews": [
			{ "buffer": 0, "byteOffset": 0, "byteLength": 12 },
			{ "buffer": 0, "byteOffset": 12, "byteLength": 48 }
		],
		"accessors": [
			{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "SCALAR" },
			{ "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC4" }
		],
		"nodes": [
			{ "name": "hips", "children": [1] },
			{ "name": "spine" },
			{ "name": "root", "children": [0] }
		],
		"skins": [{ "name": "rig", "joints": [0, 1, 2] }],
		"animations": [{
			"name": "walk",
			"samplers": [{ "input": 0, "output": 1, "interpolation": "LINEAR" }],
			"channels": [{ "sampler": 0, "target": { "node": 1, "path": "rotation" } }]
		}]
	}"#;

	#[test]
	fn a_rig_that_gained_a_bone_moves_every_binding_on_it_by_the_next_pass() {
		// the whole reason the pass relinks. Nothing tells a binding that the
		// rig under it was recompiled, and every index it holds still resolves
		// to a bone - just not the one whose name the track carries.
		let (source, output) = trees("relink-grown");

		put(&source, "models/moves.gltf", MOVES);

		let mut world = World::new();
		let mut assets = Assets::at(source.clone(), output);

		assets.sync(&mut world);

		let clip = world.clips.find("models/moves/walk");
		let skeleton = world.skeletons.find("models/moves/rig");

		world.clips.bind(clip, skeleton, &world.skeletons);

		assert_eq!(world.clips.bones(clip, skeleton), &[1], "the spine, second of two");

		// the mtime rule is `newer than`, and the two writes are microseconds
		// apart on a filesystem with finer resolution than that.
		sleep(Duration::from_millis(20));
		put(&source, "models/moves.gltf", MOVES_TALLER);
		assets.sync(&mut world);

		assert_eq!(
			world.skeletons.bones(skeleton).len(),
			3,
			"the rig was recompiled into the entry the handle already pointed at"
		);
		assert_eq!(
			world.clips.bones(clip, skeleton),
			&[2],
			"and the binding moved with the name, rather than staying on the bone that index \
			 now means"
		);
	}

	#[test]
	fn rewriting_a_clip_source_without_changing_it_does_not_move_its_revision() {
		// what the revision is for: a binding is worked out again when it
		// moves, so moving it for a file whose keys are identical is work
		// nobody asked for on every save of an unrelated part of the model.
		let (source, output) = trees("clip-quiet");

		put(&source, "models/moves.gltf", MOVES);

		let mut world = World::new();
		let mut assets = Assets::at(source.clone(), output);

		assets.sync(&mut world);

		let id = world.clips.find("models/moves/walk");

		assert_eq!(world.clips.get(id).map(Clip::revision), Some(0), "loaded once");

		sleep(Duration::from_millis(20));
		put(&source, "models/moves.gltf", MOVES);
		assets.sync(&mut world);

		assert_eq!(
			world.clips.get(id).map(Clip::revision),
			Some(0),
			"and read again without being registered again, because nothing in it changed"
		);
	}

	/// A scene an exporter wrote, with its pictures inside it.
	const MODEL: &[u8] = include_bytes!("../asset/gltf/fixtures/model.glb");

	/// The world after a fixture tree holding one model has been compiled and
	/// loaded.
	fn with_model(name: &str) -> (World, PathBuf) {
		let (source, output) = trees(name);

		put_bytes(&source, "models/lamp.glb", MODEL);

		let mut world = World::new();

		Assets::at(source.clone(), output).sync(&mut world);

		(world, source)
	}

	#[test]
	fn a_model_reaches_the_table_with_everything_it_stands() {
		let (world, _) = with_model("model-load");
		let id = world.models.find("models/lamp");

		assert!(id.is_some(), "the model reached the registry");
		assert_eq!(world.models.placements(id).len(), 5, "five pieces stand somewhere");

		let named: Vec<&str> = world
			.models
			.placements(id)
			.iter()
			.map(|placement| placement.name.as_str())
			.collect();

		assert_eq!(named, vec!["column", "arm", "arm_mirror", "panel_0", "panel_1"]);
	}

	#[test]
	fn every_handle_a_placement_holds_answers_with_the_asset_it_named() {
		// the claim the reserving exists for: a model is walked before anything
		// under its own directory, so a naive loader would resolve every one of
		// these to nothing.
		let (world, _) = with_model("model-handles");
		let id = world.models.find("models/lamp");

		for placement in world.models.placements(id) {
			assert!(placement.mesh.is_some(), "{} stands on nothing", placement.name);
			assert!(
				world
					.meshes
					.get(placement.mesh)
					.is_some_and(|mesh| mesh.value().triangles() > 0),
				"{} stands on a mesh that draws nothing",
				placement.name
			);
			assert!(placement.material.is_some(), "{} is made of nothing", placement.name);
			assert_ne!(
				placement.material,
				MaterialId::DEFAULT,
				"{} fell back to the default rather than what the file said",
				placement.name
			);
		}
	}

	#[test]
	fn a_model_loaded_before_its_meshes_still_ends_up_standing_on_them() {
		// what the reserving buys, and the only way to see it: nothing
		// re-resolves a placement, so a mesh that arrived after its model would
		// otherwise leave it standing on nothing for the life of the process.
		let (source, output) = trees("model-order");

		put_bytes(&source, "models/lamp.glb", MODEL);

		let mut world = World::new();

		Assets::at(source, output.clone()).sync(&mut world);

		// a world that has seen the model and none of its meshes
		let mut fresh = World::new();

		load_model(&mut fresh, &output.join("models").join("lamp.cmodel"), "models/lamp");

		let id = fresh.models.find("models/lamp");
		let column = fresh
			.models
			.placements(id)
			.iter()
			.find(|placement| placement.name == "column")
			.expect("the column stands somewhere")
			.mesh;

		assert!(column.is_some(), "a slot was claimed for a mesh nobody has loaded");
		assert_eq!(
			fresh
				.meshes
				.get(column)
				.map(|mesh| mesh.value().triangles()),
			Some(0),
			"and it draws nothing until one arrives"
		);

		load_mesh(
			&mut fresh,
			&output
				.join("models")
				.join("lamp")
				.join("column.cmesh"),
			"models/lamp/column",
		);

		assert!(
			fresh
				.meshes
				.get(column)
				.is_some_and(|mesh| mesh.value().triangles() > 0),
			"and the handle the model handed out answers with it"
		);

		// the same for a picture, which a material holds the same way a
		// placement holds a mesh.
		let albedo = fresh
			.materials
			.get(fresh.materials.find("models/lamp/stone"))
			.expect("the material is there")
			.albedo;

		assert!(albedo.is_some(), "a slot was claimed for a picture nobody has loaded");
		assert_eq!(
			fresh
				.textures
				.get(albedo)
				.map(|texture| texture.value().width),
			Some(1),
			"and it is one white texel until one arrives"
		);

		load_texture(
			&mut fresh,
			&output
				.join("models")
				.join("lamp")
				.join("tiles.ctex"),
			"models/lamp/tiles",
		);

		assert_eq!(
			fresh
				.textures
				.get(albedo)
				.map(|texture| texture.value().width),
			Some(32),
			"and the handle the material handed out answers with it"
		);
	}

	#[test]
	fn a_models_materials_land_beside_the_games_own() {
		let (world, _) = with_model("model-materials");
		let stone = world.materials.find("models/lamp/stone");

		assert!(stone.is_some(), "named inside the model's own path");

		let value = *world
			.materials
			.get(stone)
			.expect("the material is there");
		let albedo = world.textures.find("models/lamp/tiles");

		assert_eq!(value.albedo, albedo, "wearing the picture the model carried");
		assert_eq!(
			world
				.textures
				.get(albedo)
				.map(|texture| texture.value().width),
			Some(32),
			"and the picture is the one that was in the file, not the slot it reserved"
		);
		assert_ne!(value.normal, TextureId::FLAT_NORMAL, "and a real map");
	}

	#[test]
	fn a_model_that_changes_moves_its_own_revision_and_nothing_elses() {
		let (source, output) = trees("model-reload");

		put_bytes(&source, "models/lamp.glb", MODEL);

		let mut world = World::new();
		let mut assets = Assets::at(source, output);

		assets.sync(&mut world);

		let id = world.models.find("models/lamp");
		let before = world
			.models
			.get(id)
			.map(Model::revision)
			.unwrap_or_default();

		sleep(POLL_INTERVAL + Duration::from_millis(30));
		assets.sync(&mut world);

		assert_eq!(
			world.models.get(id).map(Model::revision),
			Some(before),
			"a model that stands what it stood is not registered again"
		);
	}

	#[test]
	fn rewriting_a_model_with_the_same_bytes_does_not_move_its_revision() {
		// an editor saving a file it did not change, which is the only way to
		// reach the comparison inside the loader: the modification time moves
		// and the contents do not.
		let (source, output) = trees("model-rewrite");

		put_bytes(&source, "models/lamp.glb", MODEL);

		let mut world = World::new();
		let mut assets = Assets::at(source.clone(), output);

		assets.sync(&mut world);

		let id = world.models.find("models/lamp");
		let before = world
			.models
			.get(id)
			.map(Model::revision)
			.unwrap_or_default();

		sleep(POLL_INTERVAL + Duration::from_millis(30));
		put_bytes(&source, "models/lamp.glb", MODEL);
		assets.sync(&mut world);

		assert_eq!(
			world.models.get(id).map(Model::revision),
			Some(before),
			"the same model does not need registering again"
		);
		assert_eq!(world.models.placements(id).len(), 5, "and it still stands what it stood");
	}

	#[test]
	fn deleting_a_model_leaves_it_standing_nothing() {
		let (source, output) = trees("model-delete");

		put_bytes(&source, "models/lamp.glb", MODEL);

		let mut world = World::new();
		let mut assets = Assets::at(source.clone(), output);

		assets.sync(&mut world);

		let id = world.models.find("models/lamp");

		assert!(!world.models.placements(id).is_empty());

		fs::remove_file(source.join("models").join("lamp.glb")).expect("the model is deleted");
		sleep(POLL_INTERVAL + Duration::from_millis(30));
		assets.sync(&mut world);

		assert_eq!(world.models.find("models/lamp"), id, "the handle still resolves");
		assert!(world.models.placements(id).is_empty(), "and it stands nothing");
	}
	#[test]
	fn a_scene_source_reaches_the_table_with_everything_in_it() {
		let (source, output) = trees("scene");
		put(
			&source,
			"scenes/room.scene",
			r#"{
				"entities": [ { "name": "crate", "at": [0, 4, 0], "mesh": "cube" } ],
				"bodies": [ { "entity": "crate", "kind": "dynamic" } ]
			}"#,
		);

		let mut world = World::new();
		let mut assets = Assets::at(source, output);
		assets.sync(&mut world);

		let id = world.scenes.find("scenes/room");

		assert!(id.is_some(), "the scene is in the table");

		let data = world.scenes.data(id);

		assert_eq!(data.things.len(), 1, "with its entity");
		assert_eq!(data.solids.len(), 1, "and its body");
		assert_eq!(
			data.things[0].mesh, "cube",
			"and the names are still names: what they resolve to is the business of whichever 			 world it is put into"
		);
	}

	#[test]
	fn editing_a_scene_reloads_it_without_moving_its_handle() {
		let (source, output) = trees("scene-edit");
		put(&source, "scenes/room.scene", r#"{ "entities": [ {} ] }"#);

		let mut world = World::new();
		let mut assets = Assets::at(source.clone(), output);
		assets.sync(&mut world);

		let id = world.scenes.find("scenes/room");
		let first = world
			.scenes
			.get(id)
			.expect("it is there")
			.revision();

		sleep(POLL_INTERVAL + Duration::from_millis(40));
		put(&source, "scenes/room.scene", r#"{ "entities": [ {}, {} ] }"#);
		assets.poll(&mut world);

		assert_eq!(world.scenes.find("scenes/room"), id, "the handle is the one it was");
		assert_eq!(world.scenes.data(id).things.len(), 2, "and it holds what the file now says");
		assert_ne!(
			world
				.scenes
				.get(id)
				.expect("still there")
				.revision(),
			first,
			"with a revision that moved, which is what anything watching it reads"
		);
	}

	#[test]
	fn deleting_a_scene_leaves_it_empty_rather_than_broken() {
		let (source, output) = trees("scene-gone");
		put(&source, "scenes/room.scene", r#"{ "entities": [ {} ] }"#);

		let mut world = World::new();
		let mut assets = Assets::at(source.clone(), output);
		assets.sync(&mut world);

		let id = world.scenes.find("scenes/room");

		assert_eq!(world.scenes.data(id).things.len(), 1, "it was there");

		fs::remove_file(source.join("scenes/room.scene")).expect("and is now gone");
		assets.sync(&mut world);

		assert!(
			world.scenes.data(id).things.is_empty(),
			"the handle still resolves and describes nothing, which creates nothing"
		);
	}
}
