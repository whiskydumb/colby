//! The compiler: a source tree in, a tree of `.cmesh` out.
//!
//! Two directories, mirrored. `assets/` holds what a person edits and
//! `.colby/assets/` holds what the engine loads, at the same relative paths
//! with the extension swapped. A mesh's **name** is that relative path without
//! its extension and with forward slashes - `assets/meshes/crystal.obj`
//! compiles to `.colby/assets/meshes/crystal.cmesh` and registers as
//! `meshes/crystal`. Names are paths rather than bare stems so that two
//! directories may hold a `wall.obj` without one quietly winning.
//!
//! Compilation is incremental on modification time, the same rule the game
//! module's watcher uses: an output newer than its input is left alone. That is
//! what makes it cheap enough for the runner to call four times a second.
//!
//! Modification time is not the only thing that makes an output stale. An
//! output written by a different [`FORMAT_VERSION`](crate::FORMAT_VERSION) is
//! stale however new it is, because the engine would refuse to read it - so
//! bumping the format rebuilds the tree by itself instead of leaving a message
//! telling someone to pass `--force`.
//!
//! An output whose source has been deleted is removed, so the two trees cannot
//! drift into disagreeing about what exists. Nothing else in the output tree is
//! touched.
//!
//! **One thing here is decided by a file's name rather than by its contents**,
//! and it is the only such rule in the project: a `.png` whose stem ends in
//! [`NORMAL_SUFFIX`] is compiled as numbers rather than as a color - @ref
//! [`texel_of`]. A PNG has no field that could say so, and the alternative is a
//! manifest beside every texture, which is a parser and a second staleness
//! input for a question that has one answer per file and never changes. When
//! materials become assets of their own the declaration moves there and this
//! rule goes.
//!
//! This is a library first. `just assets` runs it through `colby_assetc`, and
//! the runner calls [`compile_dir`] in-process on a timer - the same code, so
//! the two cannot disagree about what a file compiles to.

use std::{
	fs,
	path::{Component, Path, PathBuf},
	time::SystemTime,
};

use colby_core::{
	Error, Result,
	abi::{
		Transform,
		mesh::{self, MeshData},
		texture::Texel,
	},
	err,
	glam::Vec3,
};

use crate::{
	anim, document, font, format, gltf, html, import, jpeg, level, lua, material, model, obj,
	png, scene, script, skeleton, sound, texture, ttf, wav,
};

/// The directory under a workspace that holds editable sources.
pub const SOURCE_DIR: &str = "assets";

/// Where compiled assets go, relative to a project.
///
/// Under `.colby/` because they are derived and a project is not a cargo
/// workspace: what the engine makes for itself lives in a directory the engine
/// owns and version control ignores, and nothing should ever be tempted to
/// edit one. `just clean` takes it along with `target/`.
pub const OUTPUT_DIR: [&str; 2] = [".colby", "assets"];

/// What a `.png` has to be called to be compiled as numbers rather than color.
///
/// Matched against the stem, so `tiles_normal.png` is one and `normal_map.png`
/// is not.
pub const NORMAL_SUFFIX: &str = "_normal";

/// The source extensions the compiler knows.
pub const SOURCE_EXTENSIONS: &[&str] = &[
	obj::EXTENSION,
	png::EXTENSION,
	jpeg::EXTENSION,
	jpeg::LONG_EXTENSION,
	ttf::EXTENSION,
	html::EXTENSION,
	gltf::EXTENSION,
	gltf::BINARY_EXTENSION,
	level::EXTENSION,
	wav::EXTENSION,
	lua::EXTENSION,
	material::SOURCE_EXTENSION,
];

/// The extensions it writes.
pub const OUTPUT_EXTENSIONS: &[&str] = &[
	format::EXTENSION,
	texture::EXTENSION,
	font::EXTENSION,
	document::EXTENSION,
	model::EXTENSION,
	scene::EXTENSION,
	skeleton::EXTENSION,
	anim::EXTENSION,
	sound::EXTENSION,
	script::EXTENSION,
	material::EXTENSION,
];

/// Which of colby's formats a source compiles into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
	/// A `.obj` becomes a `.cmesh`.
	Mesh,

	/// A `.png` becomes a `.ctex`.
	Texture,

	/// A `.ttf` becomes a `.cfont`.
	Font,

	/// A `.html` becomes a `.cdoc`, with everything it links to folded in.
	Document,

	/// A `.gltf` or a `.glb` becomes a `.cmodel`, with its meshes and its
	/// pictures written beside it as assets of their own.
	///
	/// The only kind that writes more than one file. What a model *is* here
	/// is what its meshes are made of and where each piece stands; the
	/// geometry and the pixels are `.cmesh` and `.ctex` files under a
	/// directory named after it, loaded by exactly the code that loads a
	/// mesh anybody made by hand.
	Model,

	/// A `.wav` becomes a `.csnd`.
	///
	/// The samples are widened to sixteen bits and otherwise left alone:
	/// nothing is resampled, because the rate a machine's device runs at is
	/// not knowable here, and nothing is compressed, because the thing that
	/// forces a codec is music and there is none yet.
	Sound,

	/// A `.scene` becomes a `.cscene`, the same file a save is.
	///
	/// The only output the engine also writes for itself, and the reason the
	/// format is one rather than two: a level and a saved game are the same
	/// list of things standing in the same places.
	Scene,

	/// A `.material` becomes a `.cmat`.
	///
	/// The one kind whose source and output hold the same nine numbers: what
	/// compiling buys here is not a cheaper parse but a *uniform* one, so the
	/// browser's compiled-or-not, the staleness sweep and
	/// [`of_output`](Self::of_output) all work on a material without learning
	/// that it is small.
	Material,

	/// A `.cskel`, which **no source compiles into**.
	///
	/// The only kind with no source of its own: a skeleton comes out of a
	/// model beside its meshes, and never from a file somebody wrote. It is a
	/// kind all the same, because a `.cskel` left behind by a format bump has
	/// to make the model that wrote it stale - @ref [`beside_is_stale`] - and
	/// that check works by asking an output which format it is in.
	Skeleton,

	/// A `.canim`, which **no source compiles into** either.
	///
	/// The second kind with no source of its own, and for the same reason a
	/// skeleton has none: a clip comes out of a model beside its meshes. @ref
	/// [`Self::Skeleton`].
	Clip,

	/// A `.lua` becomes a `.clua`.
	///
	/// The one kind whose compiler does almost nothing, and it is a kind
	/// anyway: a program has to be walked, named, versioned and pruned like
	/// anything else, and the alternative - reading `.lua` files off the source
	/// tree at run time - is a second asset path with none of that. @ref
	/// [`script`](crate::script) for why the output is text rather than
	/// bytecode.
	Script,
}

impl Kind {
	/// What the compiler makes of a file, by its extension.
	#[must_use]
	pub fn of(path: &Path) -> Option<Self> {
		let extension = path
			.extension()
			.and_then(|extension| extension.to_str())
			.map(str::to_ascii_lowercase)?;

		match extension.as_str() {
			| obj::EXTENSION => Some(Self::Mesh),
			| png::EXTENSION | jpeg::EXTENSION | jpeg::LONG_EXTENSION => Some(Self::Texture),
			| ttf::EXTENSION => Some(Self::Font),
			| wav::EXTENSION => Some(Self::Sound),
			| html::EXTENSION => Some(Self::Document),
			| gltf::EXTENSION | gltf::BINARY_EXTENSION => Some(Self::Model),
			| level::EXTENSION => Some(Self::Scene),
			| material::SOURCE_EXTENSION => Some(Self::Material),
			| lua::EXTENSION => Some(Self::Script),
			| _ => None,
		}
	}

	/// The extension its output is written with.
	#[must_use]
	pub const fn extension(self) -> &'static str {
		match self {
			| Self::Mesh => format::EXTENSION,
			| Self::Texture => texture::EXTENSION,
			| Self::Font => font::EXTENSION,
			| Self::Sound => sound::EXTENSION,
			| Self::Document => document::EXTENSION,
			| Self::Model => model::EXTENSION,
			| Self::Scene => scene::EXTENSION,
			| Self::Material => material::EXTENSION,
			| Self::Skeleton => skeleton::EXTENSION,
			| Self::Clip => anim::EXTENSION,
			| Self::Script => script::EXTENSION,
		}
	}

	/// Which of colby's formats an *output* is in, by its extension.
	#[must_use]
	pub fn of_output(path: &Path) -> Option<Self> {
		let extension = path
			.extension()
			.and_then(|extension| extension.to_str())
			.map(str::to_ascii_lowercase)?;

		match extension.as_str() {
			| format::EXTENSION => Some(Self::Mesh),
			| texture::EXTENSION => Some(Self::Texture),
			| font::EXTENSION => Some(Self::Font),
			| sound::EXTENSION => Some(Self::Sound),
			| document::EXTENSION => Some(Self::Document),
			| model::EXTENSION => Some(Self::Model),
			| scene::EXTENSION => Some(Self::Scene),
			| material::EXTENSION => Some(Self::Material),
			| skeleton::EXTENSION => Some(Self::Skeleton),
			| anim::EXTENSION => Some(Self::Clip),
			| script::EXTENSION => Some(Self::Script),
			| _ => None,
		}
	}

	/// The format version the file at this path claims, if it is one at all.
	#[must_use]
	pub fn version_of(self, path: &Path) -> Option<u32> {
		match self {
			| Self::Mesh => format::version_of(path),
			| Self::Texture => texture::version_of(path),
			| Self::Font => font::version_of(path),
			| Self::Sound => sound::version_of(path),
			| Self::Document => document::version_of(path),
			| Self::Model => model::version_of(path),
			| Self::Scene => scene::version_of(path),
			| Self::Material => material::version_of(path),
			| Self::Skeleton => skeleton::version_of(path),
			| Self::Clip => anim::version_of(path),
			| Self::Script => script::version_of(path),
		}
	}

	/// The format version this build writes.
	#[must_use]
	pub const fn version(self) -> u32 {
		match self {
			| Self::Mesh => format::FORMAT_VERSION,
			| Self::Texture => texture::FORMAT_VERSION,
			| Self::Font => font::FORMAT_VERSION,
			| Self::Sound => sound::FORMAT_VERSION,
			| Self::Document => document::FORMAT_VERSION,
			| Self::Model => model::FORMAT_VERSION,
			| Self::Scene => scene::FORMAT_VERSION,
			| Self::Material => material::FORMAT_VERSION,
			| Self::Skeleton => skeleton::FORMAT_VERSION,
			| Self::Clip => anim::FORMAT_VERSION,
			| Self::Script => script::FORMAT_VERSION,
		}
	}
}

/// What one compiled asset turned out to be.
///
/// A sum rather than a struct with unused fields: a texture has no triangles
/// and a mesh has no mip levels, and a report that says `0` for both is a
/// report nobody can read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Produced {
	/// Geometry.
	Mesh {
		/// How many vertices the output holds.
		vertices: usize,

		/// How many triangles it draws.
		triangles: usize,

		/// The low and high corners of its bounding box.
		bounds: (Vec3, Vec3),
	},

	/// Pixels.
	Texture {
		/// The largest level's width.
		width: u32,

		/// Its height.
		height: u32,

		/// How many mip levels were built.
		levels: usize,
	},

	/// Boxes.
	Document {
		/// How many nodes the tree holds.
		nodes: usize,

		/// How many style rules apply to them.
		rules: usize,
	},

	/// A surface: what it is made of, and the pictures it wears.
	Material {
		/// How many of its two pictures it names.
		textures: usize,
	},

	/// A world: what stands where, what collides and what is held together.
	Scene {
		/// How many entities stand in it.
		entities: usize,

		/// How many bodies.
		bodies: usize,

		/// How many joints.
		joints: usize,
	},

	/// A whole model: what it is made of and where each piece stands.
	Model {
		/// How many meshes were written beside it.
		meshes: usize,

		/// How many skeletons were, for the meshes bones move.
		skeletons: usize,

		/// How many clips were, for the skeletons to be moved by.
		clips: usize,

		/// How many pictures had to be taken out of it.
		textures: usize,

		/// How many materials it declares.
		materials: usize,

		/// How many pieces of it stand somewhere.
		placements: usize,
	},

	/// Samples.
	Sound {
		/// How many frames long the recording is.
		frames: usize,

		/// How many of them go by in a second.
		rate: u32,

		/// One for mono, two for stereo.
		channels: u16,
	},

	/// Letters.
	Font {
		/// How many glyphs were baked.
		glyphs: usize,

		/// The atlas width.
		width: u32,

		/// The atlas height.
		height: u32,
	},

	/// A program.
	Script {
		/// How many lines of it there are.
		///
		/// The only thing worth counting about a program that has not been
		/// parsed, and the compiler deliberately does not parse one: the
		/// interpreter lives in the host and the compiler does not link it.
		lines: usize,
	},
}

/// One asset the compiler wrote.
#[derive(Clone, Debug)]
pub struct Compiled {
	/// The name the engine will register it under, e.g. `meshes/crystal`.
	pub name: String,

	/// Where it was written.
	pub output: PathBuf,

	/// How big the file is.
	pub bytes: usize,

	/// What is in it.
	pub produced: Produced,

	/// What the compiler read but did not understand.
	///
	/// A misspelled property or an attribute nothing implements. Not a failure,
	/// since the rest of the file compiled, but the one moment anybody is going
	/// to be told about it, so the runner logs these when it loads the result.
	pub warnings: Vec<String>,
}

/// One mesh the compiler could not write.
#[derive(Debug)]
pub struct Failure {
	/// The source that could not be read.
	pub source: PathBuf,

	/// Why.
	pub error: Error,
}

/// What one run of the compiler did.
#[derive(Debug, Default)]
pub struct Report {
	/// Every mesh written this run.
	pub compiled: Vec<Compiled>,

	/// How many sources were already up to date.
	pub unchanged: usize,

	/// Outputs deleted because their source is gone.
	pub removed: Vec<PathBuf>,

	/// Sources that could not be compiled. One bad file does not stop the run.
	pub failed: Vec<Failure>,
}

impl Report {
	/// Whether anything at all changed on disk.
	#[must_use]
	pub fn is_quiet(&self) -> bool { self.compiled.is_empty() && self.removed.is_empty() }
}

/// Where a project keeps its editable assets.
#[must_use]
pub fn source_root(workspace: &Path) -> PathBuf { workspace.join(SOURCE_DIR) }

/// Where a project keeps its compiled assets.
#[must_use]
pub fn output_root(workspace: &Path) -> PathBuf {
	OUTPUT_DIR
		.iter()
		.fold(workspace.to_path_buf(), |path, part| path.join(part))
}

/// Where a file a source names is, if it stays inside the source tree.
///
/// The one rule for a stylesheet a document links, a buffer or a picture a
/// model names, and anything else a source reaches for beside itself: a file
/// from outside the tree is one the staleness check cannot see, so it is not
/// followed. Resolved lexically - `.` and `..` folded without touching the
/// filesystem, because the file may not exist yet - and compared against the
/// tree resolved the same way, because `Path::starts_with` compares components
/// and would let `assets/ui/../../secrets.css` through as starting with
/// `assets`.
///
/// @note: a step up over nothing is kept rather than dropped. A tree named
/// `../project/assets` has to keep its shape on both sides of the comparison;
/// folding that step away on one side is what once made every stylesheet of a
/// project opened by a relative path fall silently out of its documents.
///
/// @param path - the file as the source named it, joined onto the source's own
/// directory
/// @param root - the source tree, which a reference may not leave
/// @return the resolved path, or nothing when it climbs out of the tree
#[must_use]
pub(crate) fn within(path: &Path, root: &Path) -> Option<PathBuf> {
	let path = lexical(path);

	path.starts_with(lexical(root)).then_some(path)
}

/// Resolves `.` and `..` without touching the filesystem.
///
/// @ref [`within`] for why a step up over nothing stays a step up.
fn lexical(path: &Path) -> PathBuf {
	let mut out = PathBuf::new();

	for part in path.components() {
		match part {
			| Component::CurDir => {},
			| Component::ParentDir => step_up(&mut out),
			| other => out.push(other),
		}
	}

	out
}

/// A step up: over a name it takes the name away, over nothing or over another
/// step up it stays a step up.
fn step_up(path: &mut PathBuf) {
	if matches!(path.components().next_back(), Some(Component::Normal(_))) {
		path.pop();
	} else {
		path.push("..");
	}
}

/// The name a source compiles to.
///
/// @param root - the source tree the path is inside
/// @param source - the file, inside that tree
/// @return the relative path without its extension, with forward slashes
pub fn asset_name(root: &Path, source: &Path) -> Result<String> {
	let relative = source
		.strip_prefix(root)
		.map_err(|_| err!(Asset("{} is not inside {}", source.display(), root.display())))?;

	let stem = relative.with_extension("");
	let name: Vec<String> = stem
		.components()
		.map(|part| part.as_os_str().to_string_lossy().into_owned())
		.collect();

	if name.is_empty() {
		return Err(err!(Asset("{} has no name to register under", source.display())));
	}

	Ok(name.join("/"))
}

/// What an image's channels are taken to mean.
///
/// The whole of the naming rule, in one place so that there is one place to
/// look for it and one place to change it. A stem ending in [`NORMAL_SUFFIX`]
/// is numbers; everything else is a color.
///
/// @param source - the `.png` being compiled
/// @return the layout to store and to build the mip chain with
#[must_use]
pub fn texel_of(source: &Path) -> Texel {
	let named = source
		.file_stem()
		.and_then(|stem| stem.to_str())
		.is_some_and(|stem| stem.ends_with(NORMAL_SUFFIX));

	if named { Texel::Rgba8Unorm } else { Texel::Rgba8Srgb }
}

/// Where a source compiles to.
///
/// @param root - the source tree
/// @param out - the output tree
/// @param source - the file, inside the source tree
pub fn output_path(root: &Path, out: &Path, source: &Path) -> Result<PathBuf> {
	let relative = source
		.strip_prefix(root)
		.map_err(|_| err!(Asset("{} is not inside {}", source.display(), root.display())))?;

	let kind = Kind::of(source)
		.ok_or_else(|| err!(Asset("{} is not a format the compiler knows", source.display())))?;

	Ok(out
		.join(relative)
		.with_extension(kind.extension()))
}

/// Compiles one source into one output, whether or not it needed it.
///
/// @param source - the file to read
/// @param output - the file to write; its directory is created
/// @param root - the source tree, which a document's `<link>` may not leave
/// @return what the output turned out to be
pub fn compile_file(source: &Path, output: &Path, root: &Path) -> Result<Compiled> {
	let kind = Kind::of(source)
		.ok_or_else(|| err!(Asset("{} is not a format the compiler knows", source.display())))?;

	let mut warnings = Vec::new();
	let (bytes, produced) = match kind {
		| Kind::Mesh => compile_mesh(source)?,
		| Kind::Texture => {
			let data = if has_extension(source, &[png::EXTENSION]) {
				png::import_file(source, texel_of(source))?
			} else {
				jpeg::import_file(source, texel_of(source))?
			};
			let bytes = texture::encode(&data)
				.map_err(|error| err!(Asset("{}: {error}", source.display())))?;
			let produced = Produced::Texture {
				width: data.width,
				height: data.height,
				levels: data.levels.len(),
			};

			(bytes, produced)
		},
		| Kind::Font => {
			let data = ttf::import(&fs::read(source)?)
				.map_err(|error| err!(Asset("{}: {error}", source.display())))?;
			let bytes = font::encode(&data)
				.map_err(|error| err!(Asset("{}: {error}", source.display())))?;
			let produced = Produced::Font {
				glyphs: data.glyphs.len(),
				width: data.atlas_width,
				height: data.atlas_height,
			};

			(bytes, produced)
		},
		| Kind::Sound => {
			let data = wav::import_file(source)?;
			let bytes = sound::encode(&data)
				.map_err(|error| err!(Asset("{}: {error}", source.display())))?;
			let produced = Produced::Sound {
				frames: data.frames(),
				rate: data.rate,
				channels: data.channels,
			};

			(bytes, produced)
		},
		| Kind::Document => {
			let text = document::merge(source, root)?;
			// parsed here and thrown away: the output is the text, and this is
			// the pass that turns a misspelled property into one line in the
			// log at compile time instead of a box that is silently the wrong
			// size for as long as nobody looks closely.
			let parsed = html::parse(&text, &[])
				.map_err(|error| err!(Asset("{}: {error}", source.display())))?;

			warnings = parsed.warnings;
			let produced = Produced::Document {
				nodes: parsed.document.nodes.len(),
				rules: parsed.document.rules.len(),
			};

			(document::encode(&text), produced)
		},
		| Kind::Script => compile_script(source)?,
		| Kind::Material => compile_material(source)?,
		| Kind::Scene => compile_scene(source)?,
		| Kind::Model => {
			let (bytes, produced, said) = compile_model(source, output, root)?;

			warnings = said;

			(bytes, produced)
		},
		// unreachable through `Kind::of`, which is the only way a source
		// becomes a kind, and named rather than caught by a wildcard so that
		// the next kind added has to say what it compiles from.
		| Kind::Skeleton | Kind::Clip =>
			return Err(err!(Asset(
				"{}: a skeleton and a clip are only ever written beside a model, never compiled \
				 from a source of their own",
				source.display()
			))),
	};

	if let Some(parent) = output.parent() {
		fs::create_dir_all(parent)?;
	}

	fs::write(output, &bytes)?;

	Ok(Compiled {
		name: String::new(),
		output: output.to_path_buf(),
		bytes: bytes.len(),
		produced,
		warnings,
	})
}

/// Reads a `.obj` and writes the `.cmesh` it describes.
///
/// Its own function since a sidecar joined it: the arm above is a match over
/// eleven kinds and this one now has a second input.
///
/// **A sidecar beside an OBJ may only move it.** A `.cmesh` is one mesh with
/// no names and no materials in it, so the other two answers have nothing to
/// act on and are refused - @ref [`import::check_mesh_only`]. The transform is
/// worth having on its own: the format carries no unit at all, and Unreal runs
/// a `.obj` through the same import pipeline as everything else and hands it
/// the same three offsets.
///
/// @param source - the `.obj`, in the source tree
/// @return the file to write and what to report about it
fn compile_mesh(source: &Path) -> Result<(Vec<u8>, Produced)> {
	let mut data = obj::import_file(source)?;
	let guide = import::read_beside(source)?;

	if let Some(sidecar) = &guide {
		import::check_mesh_only(sidecar)
			.map_err(|error| err!(Asset("{}: {error}", import::beside(source).display())))?;
		moved(&mut data, sidecar);
	}

	let bytes = match guide {
		| Some(_) => format::encode_guided(&data),
		| None => format::encode(&data),
	}
	.map_err(|error| err!(Asset("{}: {error}", source.display())))?;
	let produced = Produced::Mesh {
		vertices: data.vertices.len(),
		triangles: data.triangles(),
		bounds: data.bounds(),
	};

	Ok((bytes, produced))
}

/// Puts a mesh's own vertices through an import transform.
///
/// Positions by the matrix and normals by its rotation, then the tangents
/// again from scratch: a tangent is built out of the positions, the texture
/// coordinates and the normal, and rebuilding it is both shorter and more
/// nearly right than turning the old one. The winding is left alone because
/// [`import`] refuses a transform that mirrors, which is the only thing that
/// could have turned a triangle inside out.
///
/// @param data - the geometry, edited in place
/// @param sidecar - what the file beside the source said
fn moved(data: &mut MeshData, sidecar: &import::Import) {
	if sidecar.transform == Transform::IDENTITY {
		return;
	}

	let matrix = sidecar.matrix();
	let turn = sidecar.transform.rotation;

	for vertex in &mut data.vertices {
		let position = matrix.transform_point3(Vec3::from_array(vertex.position));
		let normal = (turn * Vec3::from_array(vertex.normal)).normalize_or_zero();

		vertex.position = position.to_array();
		vertex.normal = normal.to_array();
	}

	mesh::tangents(data);
}

/// Compiles one model, and everything it is made of, into a directory of its
/// own.
///
/// The only compile that writes more than one file. A model's meshes and its
/// extracted pictures are assets in their own right - the loader that reads
/// them is the one that reads a mesh anybody made by hand - so they go beside
/// the `.cmodel` under a directory named after it, and the `.cmodel` names
/// them the way the registry will.
///
/// **The directory is emptied first.** Everything in it was derived from this
/// one source, so a mesh the model no longer has should not survive an edit
/// that removed it. That is also what lets the pruning above keep the whole
/// directory without knowing what is supposed to be in it.
///
/// @param source - the `.gltf` or `.glb`
/// @param output - where the `.cmodel` goes; its stem names the directory
/// @param root - the source tree, which nothing it names may leave
/// @return the file to write, what is in it, and what could not be used
fn compile_model(source: &Path, output: &Path, root: &Path) -> Result<Written> {
	let file = gltf::Gltf::open(source, root)?;
	let imported = gltf::import(&file)?;
	let stem = asset_name(root, source)?;
	let directory = output.with_extension("");
	let guide = import::read_beside(source)?;
	let sidecar = guide.clone().unwrap_or(import::Import::NONE);
	let mut warnings = imported.warnings.clone();

	drop(fs::remove_dir_all(&directory));
	fs::create_dir_all(&directory)?;

	let placements = placed(&imported, &sidecar, &stem, &mut warnings);
	let standing: Vec<&str> = placements
		.iter()
		.map(|placement| placement.mesh.as_str())
		.collect();

	// only what something still stands. A piece the sidecar skipped costs its
	// `.cmesh` as well as its row, which is most of what skipping is for: a
	// collision proxy nobody draws is geometry nobody should be shipping.
	for piece in &imported.meshes {
		let name = format!("{stem}/{}", piece.name);

		if !standing.contains(&name.as_str()) {
			continue;
		}

		let bytes = format::encode(&piece.data)
			.map_err(|error| err!(Asset("{}: {error}", source.display())))?;

		fs::write(beside(&directory, &piece.name, format::EXTENSION), bytes)?;
	}

	for picture in &imported.textures {
		let bytes = texture::encode(&picture.data)
			.map_err(|error| err!(Asset("{}: {error}", source.display())))?;

		fs::write(beside(&directory, &picture.name, texture::EXTENSION), bytes)?;
	}

	// a skin with no bones in it could not be read, and nothing names it, so
	// there is nothing to write down.
	for rig in imported
		.skins
		.iter()
		.filter(|rig| !rig.data.is_empty())
	{
		let bytes = skeleton::encode(&rig.data)
			.map_err(|error| err!(Asset("{}: {error}", source.display())))?;

		fs::write(beside(&directory, &rig.name, skeleton::EXTENSION), bytes)?;
	}

	// and the same for a clip that moves nothing colby knows a name for.
	for moves in imported
		.clips
		.iter()
		.filter(|moves| !moves.data.is_empty())
	{
		let bytes = anim::encode(&moves.data)
			.map_err(|error| err!(Asset("{}: {error}", source.display())))?;

		fs::write(beside(&directory, &moves.name, anim::EXTENSION), bytes)?;
	}

	// only the surfaces something still wears. A material the sidecar sent
	// somewhere else is not written here, because writing it would register a
	// row under the model's own name that nothing in the file names.
	let worn: Vec<&str> = placements
		.iter()
		.map(|placement| placement.material.as_str())
		.collect();
	let data = model::ModelData {
		guided: guide.is_some(),
		materials: imported
			.materials
			.iter()
			.map(|surface| model::Material {
				name: format!("{stem}/{}", surface.name),
				albedo: picture_name(surface.albedo.as_ref(), &stem, &imported, root),
				normal: picture_name(surface.normal.as_ref(), &stem, &imported, root),
				base_color: surface.base_color,
				metallic: surface.metallic,
				roughness: surface.roughness,
				// one, and the exchange format is why: a texture transform
				// lives in `KHR_texture_transform`, which nothing here reads.
				uv_scale: colby_core::glam::Vec2::ONE,
				wrap: surface.wrap,
				blend: surface.blend,
				opacity: surface.opacity,
			})
			.filter(|surface| worn.contains(&surface.name.as_str()))
			.collect(),
		placements,
	};
	let produced = Produced::Model {
		meshes: data
			.placements
			.iter()
			.map(|placement| placement.mesh.as_str())
			.collect::<std::collections::BTreeSet<_>>()
			.len(),
		skeletons: imported
			.skins
			.iter()
			.filter(|rig| !rig.data.is_empty())
			.count(),
		clips: imported
			.clips
			.iter()
			.filter(|moves| !moves.data.is_empty())
			.count(),
		textures: imported.textures.len(),
		materials: data.materials.len(),
		placements: data.placements.len(),
	};
	let bytes =
		model::encode(&data).map_err(|error| err!(Asset("{}: {error}", source.display())))?;

	Ok((bytes, produced, warnings))
}

/// Every piece of a model that survives its sidecar, where the sidecar puts it.
///
/// Three of the sidecar's answers land here: what to leave out, where the rest
/// of it goes, and what each surface is really made of.
///
/// **A skinned piece is not moved.** The exchange format says a skinned mesh
/// node's own transform is ignored, because every vertex of it is placed by its
/// bones instead, so colby writes the identity there - and an import transform
/// composed onto the identity is still not what draws it. Said out loud rather
/// than left to be discovered.
///
/// @param imported - what the file turned out to hold
/// @param sidecar - what the file beside it said
/// @param stem - the model's own asset name, which its pieces are named inside
/// @param warnings - where a word about a shear or a skin goes
fn placed(
	imported: &gltf::Model,
	sidecar: &import::Import,
	stem: &str,
	warnings: &mut Vec<String>,
) -> Vec<model::Placement> {
	imported
		.placements
		.iter()
		.filter(|placement| sidecar.takes(&placement.name))
		.filter_map(|placement| {
			let piece = imported.meshes.get(placement.mesh)?;
			let (transform, drift) = sidecar.applied(placement.transform);

			if import::is_sheared(drift) {
				warnings.push(format!(
					"{} is sheared by the import transform, by {drift}, and colby places it 					 square",
					placement.name
				));
			}

			if placement.skeleton.is_some() && !sidecar.is_silent() {
				warnings.push(format!(
					"{} is moved by bones, so the import transform does not place it",
					placement.name
				));
			}

			Some(model::Placement {
				name: placement.name.clone(),
				mesh: format!("{stem}/{}", piece.name),
				material: piece
					.material
					.and_then(|index| imported.materials.get(index))
					.map(|surface| {
						sidecar
							.material(&surface.name)
							.map_or_else(|| format!("{stem}/{}", surface.name), str::to_owned)
					})
					.unwrap_or_default(),
				skeleton: placement
					.skeleton
					.and_then(|index| imported.skins.get(index))
					.map(|rig| format!("{stem}/{}", rig.name))
					.unwrap_or_default(),
				transform,
			})
		})
		.collect()
}

/// What one compile hands back: the file, what is in it, and what it dropped.
type Written = (Vec<u8>, Produced, Vec<String>);

/// Where one of a model's own assets is written.
fn beside(directory: &Path, name: &str, extension: &str) -> PathBuf {
	directory.join(format!("{name}.{extension}"))
}

/// The asset name of one of a material's pictures.
///
/// A picture taken out of the model is registered under the model's own name;
/// one that is a file of its own is registered under whatever the compiler
/// calls that file, because it is on the same walk and will be compiled by it.
fn picture_name(
	picture: Option<&gltf::Picture>,
	stem: &str,
	imported: &gltf::Model,
	root: &Path,
) -> String {
	match picture {
		| Some(gltf::Picture::Inside(index)) => imported
			.textures
			.get(*index)
			.map(|taken| format!("{stem}/{}", taken.name))
			.unwrap_or_default(),
		| Some(gltf::Picture::Beside(path)) => asset_name(root, path).unwrap_or_default(),
		| None => String::new(),
	}
}

/// Compiles a whole source tree.
///
/// One source that cannot be read lands in [`Report::failed`] and the rest are
/// compiled anyway. That is not leniency - a compiler the runner calls on a
/// timer will regularly see a file halfway through being saved, and stopping
/// the run would mean a good edit to one mesh is held up by a bad edit to
/// another.
///
/// @param root - the source tree
/// @param out - the output tree
/// @param force - recompile even when the output is newer
/// @return what the run did
pub fn compile_dir(root: &Path, out: &Path, force: bool) -> Result<Report> {
	if !root.is_dir() {
		return Err(err!(Asset("{} is not a directory", root.display())));
	}

	let mut report = Report::default();
	let mut wanted = Vec::new();

	for source in sources(root)? {
		let output = output_path(root, out, &source)?;

		// two pictures under one name would quietly overwrite each other, and a
		// second extension for one kind is what made that reachable.
		if wanted.contains(&output) {
			report.failed.push(Failure {
				error: err!(Asset(
					"another source already compiles to {}; two files cannot share one asset \
					 name",
					output.display()
				)),
				source,
			});

			continue;
		}

		wanted.push(output.clone());

		if !force && !is_stale(&source, &output, root) {
			report.unchanged += 1;
			continue;
		}

		match compile_file(&source, &output, root) {
			| Ok(mut compiled) => {
				compiled.name = asset_name(root, &source)?;
				report.compiled.push(compiled);
			},
			| Err(error) => report.failed.push(Failure { source, error }),
		}
	}

	prune(out, &wanted, &mut report);

	Ok(report)
}

/// Every source file under a tree, sorted.
pub fn sources(root: &Path) -> Result<Vec<PathBuf>> { walk(root, SOURCE_EXTENSIONS) }

/// Every compiled asset under a tree, sorted.
///
/// This is what the runner loads from, and it walks the output tree rather than
/// trusting a compile report: an asset built by `just assets` in another
/// terminal should be picked up just the same.
pub fn outputs(root: &Path) -> Result<Vec<PathBuf>> { walk(root, OUTPUT_EXTENSIONS) }

/// Every file under a tree with one of these extensions, sorted.
///
/// Sorted so that a run reports its work in an order a person can follow, and
/// so that two runs over the same tree agree about it.
fn walk(root: &Path, extensions: &[&str]) -> Result<Vec<PathBuf>> {
	let mut found = Vec::new();
	let mut pending = vec![root.to_path_buf()];

	while let Some(dir) = pending.pop() {
		for entry in fs::read_dir(&dir)? {
			let path = entry?.path();
			if path.is_dir() {
				pending.push(path);
				continue;
			}

			if has_extension(&path, extensions) {
				found.push(path);
			}
		}
	}

	found.sort();

	Ok(found)
}

/// Whether a path ends in one of these extensions, ignoring case.
fn has_extension(path: &Path, extensions: &[&str]) -> bool {
	path.extension()
		.and_then(|extension| extension.to_str())
		.map(str::to_ascii_lowercase)
		.is_some_and(|extension| extensions.contains(&extension.as_str()))
}

/// Whether an output needs rebuilding.
///
/// A missing output, or one no newer than its source, is stale. A source whose
/// time cannot be read is treated as stale too: recompiling costs a millisecond
/// and skipping could mean never noticing an edit.
///
/// An output the engine could not read is stale whatever its timestamp says.
/// That check reads twelve bytes per up-to-date output per pass, which is more
/// than the `stat` it follows and still nothing next to being told to run the
/// compiler by hand after every format change.
fn is_stale(source: &Path, output: &Path, root: &Path) -> bool {
	let Ok(built) = mtime(output) else {
		return true;
	};

	let Some(kind) = Kind::of(source) else {
		return false;
	};

	if kind.version_of(output) != Some(kind.version()) {
		return true;
	}

	if kind == Kind::Model && beside_is_stale(output) {
		return true;
	}

	if guide_changed(source, output, kind) {
		return true;
	}

	if mtime(source)
		.ok()
		.is_none_or(|edited| edited >= built)
	{
		return true;
	}

	// and the files this one was built out of. A document is compiled with its
	// stylesheets and its scripts folded into it, so editing a sheet three
	// documents share has to rebuild all three - otherwise the picture only
	// changes for whichever of them somebody happens to touch next.
	extra_inputs(source, kind, root)
		.iter()
		.any(|path| {
			mtime(path)
				.ok()
				.is_none_or(|edited| edited >= built)
		})
}

/// Whether anything a model wrote beside its own file is in an old format.
///
/// A model is the one source that writes more than one output, and the check
/// above is handed only the first of them. Its meshes and its pictures are
/// ordinary assets in a directory named after it, so a bump to `.cmesh` or
/// `.ctex` leaves them behind - and being left behind here is not a stale
/// picture, it is a file the loader refuses outright and a model standing on
/// nothing.
///
/// @param output - the `.cmodel` the model was compiled into
/// @return whether it has to be built again for its neighbors' sake
fn beside_is_stale(output: &Path) -> bool {
	// no directory at all is a model with neither geometry nor pictures of its
	// own, which is unusual and is not a reason to rebuild it.
	let Ok(entries) = fs::read_dir(output.with_extension("")) else {
		return false;
	};

	entries.flatten().any(|entry| {
		let path = entry.path();

		Kind::of_output(&path).is_some_and(|kind| kind.version_of(&path) != Some(kind.version()))
	})
}

/// Everything besides the source itself that an output was built out of.
///
/// A compiler's associated files, and colby has three cases of them. A document
/// links stylesheets and scripts, and both are folded into the `.cdoc`. A
/// model links buffers and pictures, and although those are not folded in,
/// what is written out of the model is built from them. And either of the two
/// geometry kinds may have a sidecar beside it saying how to read the file -
/// @ref [`import`].
///
/// **A sidecar is listed only when it is there.** A path that does not exist
/// has no time to read, and a caller reads that as stale - so naming one that
/// is usually absent would rebuild every model on every pass, four times a
/// second. What happens when one is *deleted* is a different question and is
/// answered by a mark in the output; @ref [`is_stale`].
fn extra_inputs(source: &Path, kind: Kind, root: &Path) -> Vec<PathBuf> {
	let mut found = match kind {
		| Kind::Document => {
			let Ok(text) = fs::read_to_string(source) else {
				return Vec::new();
			};

			// the stylesheets and nothing else. A program a document names is
			// an asset in its own right, so it goes stale on its own and the
			// document naming it does not move when it does. That is most of
			// what making it one bought.
			document::stylesheets(source, &text, root)
		},
		| Kind::Model => gltf::linked(source, root),
		| _ => Vec::new(),
	};

	if matches!(kind, Kind::Mesh | Kind::Model) {
		let sidecar = import::beside(source);

		if sidecar.is_file() {
			found.push(sidecar);
		}
	}

	found
}

/// Whether an output was built through a sidecar that is not there any more.
///
/// The one thing the source tree cannot be asked. Deleting `lamp.gltf.model`
/// moves nothing the sweep looks at - the `.gltf` did not change and the output
/// is still newer than it - so without this the model would go on standing at
/// the scale of a file nobody can find. The output says how it was built and
/// this compares that against what is beside the source now.
///
/// @param source - the `.obj`, `.gltf` or `.glb`
/// @param output - what it compiled to
/// @param kind - which of the two it is
fn guide_changed(source: &Path, output: &Path, kind: Kind) -> bool {
	let was = match kind {
		| Kind::Mesh => format::flags_of(output).is_some_and(|flags| flags & format::GUIDED != 0),
		| Kind::Model => model::flags_of(output).is_some_and(|flags| flags & model::GUIDED != 0),
		| _ => return false,
	};

	was && !import::beside(source).is_file()
}

/// Turns one `.lua` into the bytes of a `.clua`.
///
/// Its own function only because [`compile_file`] is one arm per kind and the
/// line count is a lint; it is also the arm that reads best out of line,
/// because the whole of what compiling a program comes to is reading it.
///
/// @param source - the `.lua`, in the source tree
/// @return the file to write and what to report about it
fn compile_script(source: &Path) -> Result<(Vec<u8>, Produced)> {
	let text = lua::import(&fs::read_to_string(source)?);
	let produced = Produced::Script { lines: text.lines().count() };

	Ok((script::encode(&text), produced))
}

/// Reads a `.scene` and writes the `.cscene` it describes.
///
/// Its own function for [`compile_material`]'s reason, and it went out of the
/// match on the same day: eleven kinds is more than a match with bodies in it
/// can hold.
///
/// @param source - the `.scene`, in the source tree
/// @return the file to write and what to report about it
fn compile_scene(source: &Path) -> Result<(Vec<u8>, Produced)> {
	let data = level::import(&fs::read_to_string(source)?)
		.map_err(|error| err!(Asset("{}: {error}", source.display())))?;
	let bytes =
		scene::encode(&data).map_err(|error| err!(Asset("{}: {error}", source.display())))?;
	let produced = Produced::Scene {
		entities: data.things.len(),
		bodies: data.solids.len(),
		joints: data.links.len(),
	};

	Ok((bytes, produced))
}

/// Reads a `.material` and writes the `.cmat` it describes.
///
/// Its own function for the reason [`compile_script`] is: the arm above is a
/// match over eleven kinds, and a body of more than two lines in any of them
/// is what pushes the whole of it past what one function may be.
///
/// @param source - the `.material`, in the source tree
/// @return the file to write and what to report about it
fn compile_material(source: &Path) -> Result<(Vec<u8>, Produced)> {
	let coat = material::import(&fs::read_to_string(source)?)
		.map_err(|error| err!(Asset("{}: {error}", source.display())))?;
	let named = usize::from(!coat.albedo.is_empty()) + usize::from(!coat.normal.is_empty());

	Ok((material::encode(&coat), Produced::Material { textures: named }))
}

/// A file's modification time.
fn mtime(path: &Path) -> Result<SystemTime> { Ok(fs::metadata(path)?.modified()?) }

/// Deletes outputs whose source is gone.
///
/// Only colby's own compiled formats are considered, and only ones the run did
/// not want, so nothing else that happens to live in the output tree is at
/// risk.
fn prune(out: &Path, wanted: &[PathBuf], report: &mut Report) {
	let Ok(existing) = outputs(out) else {
		return;
	};

	// everything under a model's own directory was derived from the model, and
	// the model rewrites the whole directory when it is compiled. So the rule
	// here is about the model rather than about its parts, and nothing has to
	// read a `.cmodel` to find out what is supposed to be beside it.
	let models: Vec<PathBuf> = wanted
		.iter()
		.filter(|path| Kind::of_output(path) == Some(Kind::Model))
		.map(|path| path.with_extension(""))
		.collect();

	for path in existing {
		if wanted.contains(&path)
			|| models
				.iter()
				.any(|inside| path.starts_with(inside))
		{
			continue;
		}

		if fs::remove_file(&path).is_ok() {
			report.removed.push(path);
		}
	}
}

#[cfg(test)]
mod tests {
	use std::{thread::sleep, time::Duration};

	use super::*;
	use crate::{document::DocumentFile, format::MeshFile, texture::TextureFile};

	/// Every kind the compiler knows about.
	///
	/// Written out rather than derived, and kept honest by
	/// [`the_list_of_kinds_holds_every_one_there_is`], which indexes into it
	/// through an exhaustive match: a variant added to [`Kind`] has to be given
	/// an index, and the only index left over is one this array does not have.
	const EVERY_KIND: [Kind; 11] = [
		Kind::Mesh,
		Kind::Texture,
		Kind::Font,
		Kind::Sound,
		Kind::Document,
		Kind::Model,
		Kind::Scene,
		Kind::Material,
		Kind::Skeleton,
		Kind::Clip,
		Kind::Script,
	];

	/// Where a kind sits in [`EVERY_KIND`].
	const fn index_of(kind: Kind) -> usize {
		match kind {
			| Kind::Mesh => 0,
			| Kind::Texture => 1,
			| Kind::Font => 2,
			| Kind::Sound => 3,
			| Kind::Document => 4,
			| Kind::Model => 5,
			| Kind::Scene => 6,
			| Kind::Material => 7,
			| Kind::Skeleton => 8,
			| Kind::Clip => 9,
			| Kind::Script => 10,
		}
	}

	/// A path with this extension on it, for asking the two classifiers.
	fn named(extension: &str) -> PathBuf { PathBuf::from("thing").with_extension(extension) }

	#[test]
	fn the_list_of_kinds_holds_every_one_there_is() {
		// walked by index rather than by value, which is the direction with
		// teeth: the other way round, a list holding one kind twice and
		// another not at all agrees with itself and every test that walks it
		// quietly skips the missing one.
		for (index, kind) in EVERY_KIND.iter().enumerate() {
			assert_eq!(
				index_of(*kind),
				index,
				"{kind:?} sits at {index} and the match puts it at {}, so the list has a 				 \
				 duplicate and a gap",
				index_of(*kind)
			);
		}
	}

	#[test]
	fn every_kind_writes_an_extension_the_walker_looks_for() {
		// the bug this catches, which happened: a kind was added to `Kind::of`
		// and to the compiler and left out of the two extension lists, so the
		// walker never handed the compiler one of its sources and never found
		// one of its outputs. Everything compiled, every other test passed,
		// and the asset was invisible.
		for kind in EVERY_KIND {
			assert!(
				OUTPUT_EXTENSIONS.contains(&kind.extension()),
				"{kind:?} writes .{} and OUTPUT_EXTENSIONS does not list it, so nothing it 				 \
				 wrote is ever loaded",
				kind.extension()
			);
			assert_eq!(
				Kind::of_output(&named(kind.extension())),
				Some(kind),
				"and .{} has to lead back to it",
				kind.extension()
			);
		}
	}

	#[test]
	fn every_kind_with_a_source_has_one_the_walker_looks_for() {
		for kind in EVERY_KIND {
			let reachable = SOURCE_EXTENSIONS
				.iter()
				.any(|extension| Kind::of(&named(extension)) == Some(kind));
			// the two kinds with no source of their own are named rather than
			// caught by a wildcard, so that a third one has to be thought
			// about here.
			let expected = !matches!(kind, Kind::Skeleton | Kind::Clip);

			assert_eq!(
				reachable, expected,
				"{kind:?}: a kind with a source of its own has to be reachable through 				 \
				 SOURCE_EXTENSIONS, or nothing ever hands the compiler one"
			);
		}
	}

	#[test]
	fn every_source_extension_the_walker_collects_compiles_into_something() {
		for extension in SOURCE_EXTENSIONS {
			assert!(
				Kind::of(&named(extension)).is_some(),
				".{extension} is walked for and then not recognized, so every file with one 				 fails to compile"
			);
		}
	}

	/// A cube as OBJ, written the way a tool would.
	const CUBE_OBJ: &str = "\
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
";

	/// A directory nobody else is using, removed and recreated.
	fn workspace(name: &str) -> PathBuf {
		let dir = std::env::temp_dir()
			.join("colby-compile-tests")
			.join(name);

		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(dir.join(SOURCE_DIR).join("meshes")).expect("the fixture is made");

		dir
	}

	/// Writes a source into a fixture workspace.
	fn put(workspace: &Path, relative: &str, text: &str) -> PathBuf {
		let path = source_root(workspace).join(relative);
		if let Some(parent) = path.parent() {
			fs::create_dir_all(parent).expect("the directory is made");
		}

		fs::write(&path, text).expect("the source is written");

		path
	}

	/// Compiles a fixture workspace.
	fn run(workspace: &Path, force: bool) -> Report {
		compile_dir(&source_root(workspace), &output_root(workspace), force)
			.expect("the tree compiles")
	}

	/// A scene source with an entity, a body under it and a rope holding it.
	const SCENE: &str = r#"{
		"entities": [ { "name": "crate", "position": [0, 4, 0], "mesh": "cube" } ],
		"bodies": [ { "name": "crate", "entity": "crate", "kind": "dynamic" } ],
		"joints": [ { "first": "crate", "anchors": [[0, 0, 0], [0, 8, 0]], "length": 3 } ]
	}"#;

	#[test]
	fn a_scene_source_compiles_into_a_scene_file() {
		let workspace = workspace("scene");
		put(&workspace, "scenes/room.scene", SCENE);

		let report = run(&workspace, false);
		let compiled = report
			.compiled
			.iter()
			.find(|it| it.name == "scenes/room")
			.expect("it compiled");

		assert!(
			matches!(compiled.produced, Produced::Scene { entities: 1, bodies: 1, joints: 1 }),
			"the report says what is in it, got {:?}",
			compiled.produced
		);

		let file = scene::SceneFile::open(&compiled.output).expect("and it reads back");
		let data = file.to_scene_data();

		assert_eq!(data.things[0].mesh, "cube", "the entity names its mesh");
		assert_eq!(data.solids[0].thing, 0, "the body points at it by index");
		assert_eq!(data.links[0].first, 0, "and so does the rope");
	}

	#[test]
	fn a_scene_that_will_not_read_is_reported_rather_than_written() {
		let workspace = workspace("scene-broken");
		put(
			&workspace,
			"scenes/broken.scene",
			r#"{ "entities": [ { "whereabouts": [1] } ] }"#,
		);

		let report = run(&workspace, false);
		let failed = report.failed.first().expect("it failed");

		assert!(
			failed.source.ends_with("broken.scene")
				&& failed.error.to_string().contains("whereabouts"),
			"the file and the field are both named: {}",
			failed.error
		);
		assert!(
			!output_root(&workspace)
				.join("scenes/broken.cscene")
				.exists(),
			"and nothing was written"
		);
	}

	#[test]
	fn a_scene_written_by_another_version_is_recompiled() {
		let workspace = workspace("scene-stale");
		put(&workspace, "scenes/room.scene", SCENE);
		run(&workspace, false);

		let output = output_root(&workspace).join("scenes/room.cscene");
		let mut bytes = fs::read(&output).expect("it was written");
		bytes[8] = 99;
		fs::write(&output, &bytes).expect("and is now from the future");

		let report = run(&workspace, false);

		assert!(
			report
				.compiled
				.iter()
				.any(|it| it.name == "scenes/room"),
			"a version this build does not read is stale however new the file is"
		);
	}

	#[test]
	fn a_name_is_the_relative_path_without_its_extension() {
		let root = Path::new("C:/w/assets");

		assert_eq!(
			asset_name(root, &root.join("meshes").join("crystal.obj")).expect("it is inside"),
			"meshes/crystal",
			"forward slashes, whatever the platform separator is"
		);
		assert_eq!(
			asset_name(root, &root.join("floor.obj")).expect("it is inside"),
			"floor",
			"a file at the top of the tree is just its stem"
		);
		assert!(
			asset_name(root, Path::new("C:/elsewhere/thing.obj")).is_err(),
			"and a file outside the tree has no name here"
		);
	}

	#[test]
	fn a_reference_stays_inside_a_tree_named_by_a_climbing_path() {
		// the fault this guards against: folding `..` by popping drops a step
		// up over nothing, so a tree named `../project/assets` came out as
		// `project/assets` and nothing inside it was inside it any more -
		// which was every stylesheet of a project opened by a relative path
		// falling silently out of its documents.
		let root = Path::new("../project/assets");

		assert_eq!(
			within(&root.join("ui").join("theme.css"), root),
			Some(root.join("ui").join("theme.css")),
			"a file beside a document is inside the tree the document is in"
		);
		assert_eq!(
			within(
				&root
					.join("ui")
					.join("..")
					.join("..")
					.join("secrets.css"),
				root
			),
			None,
			"and one that climbs out of it is still outside, however the tree was named"
		);
		assert_eq!(
			within(Path::new("../../project/assets/x.css"), root),
			None,
			"two steps up are not one"
		);
		assert_eq!(
			within(&root.join("ui").join(".").join("theme.css"), root),
			Some(root.join("ui").join("theme.css")),
			"and a step nowhere is folded away"
		);
	}

	#[test]
	fn an_output_mirrors_its_source_with_the_extension_swapped() {
		let root = Path::new("C:/w/assets");
		let out = Path::new("C:/w/target/assets");
		let output = output_path(root, out, &root.join("meshes").join("crystal.obj"))
			.expect("it is inside");

		assert_eq!(output, out.join("meshes").join("crystal.cmesh"), "same shape, new leaf");
	}

	#[test]
	fn the_default_roots_sit_where_the_readme_says() {
		let workspace = Path::new("C:/w");

		assert_eq!(source_root(workspace), workspace.join("assets"), "sources are edited");
		assert_eq!(
			output_root(workspace),
			workspace.join(".colby").join("assets"),
			"and outputs are derived, so they live in the engine's own directory"
		);
	}

	#[test]
	fn a_source_compiles_and_reads_back_as_the_mesh_it_describes() {
		let workspace = workspace("round-trip");
		put(&workspace, "meshes/box.obj", CUBE_OBJ);

		let report = run(&workspace, false);

		assert_eq!(report.compiled.len(), 1, "one source, one output");
		assert_eq!(report.failed.len(), 0, "and nothing went wrong");

		let compiled = &report.compiled[0];

		assert_eq!(compiled.name, "meshes/box", "registered under its path");
		assert_eq!(
			compiled.produced,
			Produced::Mesh {
				vertices: 24,
				triangles: 12,
				bounds: (Vec3::splat(-0.5), Vec3::splat(0.5)),
			},
			"six quads is twelve triangles, four vertices each because the shading is flat"
		);

		let file = MeshFile::open(&compiled.output).expect("the output is a mesh");

		assert_eq!(file.header().index_count, 36, "and holds what the report claimed");
		assert_eq!(file.vertices().len(), 24, "with every vertex");
		assert!(
			file.bounds()
				.0
				.abs_diff_eq(Vec3::splat(-0.5), 1.0e-6),
			"and the box it measured"
		);
	}

	/// The two-by-two truecolor PNG from `png::tests`, so that a texture can go
	/// through the compiler here without a second fixture to keep in step.
	const RGB_QUAD: [u8; 77] = [
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
		0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x08, 0x02, 0x00, 0x00, 0x00, 0xFD,
		0xD4, 0x9A, 0x73, 0x00, 0x00, 0x00, 0x14, 0x49, 0x44, 0x41, 0x54, 0x78, 0xDA, 0x63, 0xF8,
		0xCF, 0xC0, 0xC0, 0x00, 0xC2, 0x0C, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x00, 0x1F, 0xEE, 0x05,
		0xFB, 0x60, 0x6C, 0x70, 0xF2, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42,
		0x60, 0x82,
	];

	#[test]
	fn what_an_image_holds_is_decided_by_the_end_of_its_name() {
		let workspace = workspace("normals");
		let textures = source_root(&workspace).join("textures");
		fs::create_dir_all(&textures).expect("the directory");
		fs::write(textures.join("wall.png"), RGB_QUAD).expect("the color is written");
		fs::write(textures.join("wall_normal.png"), RGB_QUAD).expect("the normal is written");

		let report = run(&workspace, false);

		assert!(report.failed.is_empty(), "both compile");

		let texel_of_output = |name: &str| {
			let compiled = report
				.compiled
				.iter()
				.find(|compiled| compiled.name == name)
				.expect("the texture is in the report");

			TextureFile::open(&compiled.output)
				.expect("it reads back")
				.header()
				.texel
		};

		assert_eq!(
			texel_of_output("textures/wall"),
			Texel::Rgba8Srgb.code(),
			"an ordinary name is a color"
		);
		assert_eq!(
			texel_of_output("textures/wall_normal"),
			Texel::Rgba8Unorm.code(),
			"and one ending in the suffix is numbers"
		);
	}

	#[test]
	fn a_png_compiles_to_a_texture_beside_the_meshes() {
		let workspace = workspace("textures");
		put(&workspace, "meshes/box.obj", CUBE_OBJ);
		fs::create_dir_all(source_root(&workspace).join("textures")).expect("the directory");
		fs::write(source_root(&workspace).join("textures/quad.png"), RGB_QUAD)
			.expect("the image is written");

		let report = run(&workspace, false);

		assert_eq!(report.compiled.len(), 2, "both kinds compile in one pass");
		assert!(report.failed.is_empty(), "and neither failed");

		let image = report
			.compiled
			.iter()
			.find(|compiled| compiled.name == "textures/quad")
			.expect("the texture is in the report");

		assert_eq!(
			image.produced,
			Produced::Texture { width: 2, height: 2, levels: 2 },
			"two by two, with the level below it"
		);
		assert_eq!(
			image.output.extension().and_then(|e| e.to_str()),
			Some("ctex"),
			"written with its own extension, not the mesh one"
		);

		let file = TextureFile::open(&image.output).expect("the output is a texture");

		assert_eq!(file.header().levels, 2, "and holds what the report claimed");
	}

	#[test]
	fn a_second_run_does_nothing_and_a_touched_source_is_rebuilt() {
		let workspace = workspace("incremental");
		put(&workspace, "meshes/box.obj", CUBE_OBJ);

		assert_eq!(run(&workspace, false).compiled.len(), 1, "the first run compiles");

		let second = run(&workspace, false);

		assert!(second.compiled.is_empty(), "the second finds nothing to do");
		assert_eq!(second.unchanged, 1, "and says why");
		assert!(second.is_quiet(), "so nothing on disk moved");

		// mtime resolution is fine on NTFS, but the two writes are microseconds
		// apart and the rule is `newer than`, so give the clock a moment.
		sleep(Duration::from_millis(20));
		put(&workspace, "meshes/box.obj", CUBE_OBJ);

		let third = run(&workspace, false);

		assert_eq!(third.compiled.len(), 1, "an edited source is compiled again");
		assert_eq!(third.unchanged, 0, "and nothing is skipped");
	}

	#[test]
	fn editing_a_program_rebuilds_the_program_and_leaves_the_document_alone() {
		// the point of a program being an asset, and the assertion with teeth
		// in it is the second one: a document that still listed the program as
		// something it was built out of would be rebuilt here as well, which is
		// what used to happen and what made a shared program expensive.
		let workspace = workspace("script-staleness");
		put(&workspace, "ui/hud.lua", "local a = 1\n");
		put(&workspace, "ui/hud.html", "<script src=\"ui/hud\"></script>\n<div></div>\n");

		assert_eq!(run(&workspace, false).compiled.len(), 2, "a document and a program");
		assert_eq!(run(&workspace, false).unchanged, 2, "and the second run has nothing to do");

		// as above: the rule is `newer than`, and two writes in the same
		// microsecond are not.
		sleep(Duration::from_millis(20));
		put(&workspace, "ui/hud.lua", "local a = 2\n");

		let report = run(&workspace, false);

		assert_eq!(report.compiled.len(), 1, "the program is rebuilt");
		assert_eq!(report.unchanged, 1, "and the document is not");
		assert!(
			report.compiled[0].name == "ui/hud"
				&& Kind::of_output(&report.compiled[0].output) == Some(Kind::Script),
			"and it is the program that was rebuilt rather than the document, which shares its \
			 name: {:?}",
			report.compiled[0].output
		);

		let output = output_root(&workspace)
			.join("ui")
			.join("hud.clua");
		let compiled = script::ScriptFile::open(&output).expect("it reads");

		assert!(compiled.source().contains("local a = 2"), "with the new text in it");
	}

	#[test]
	fn a_document_and_the_program_it_names_are_two_assets_of_one_name() {
		// they collide in every way but the extension, which is what makes the
		// pair worth a test: the compiler names an asset by its own path, so
		// `ui/hud.html` and `ui/hud.lua` are both `ui/hud` and land in two
		// different tables.
		let workspace = workspace("script-beside-document");
		put(&workspace, "ui/hud.lua", "ui.on(\"a\", \"click\", function() end)\n");
		put(
			&workspace,
			"ui/hud.html",
			"<script src=\"ui/hud\"></script>\n<div id=\"a\"></div>\n",
		);

		let report = run(&workspace, false);
		let names: Vec<&str> = report
			.compiled
			.iter()
			.map(|one| one.name.as_str())
			.collect();

		assert_eq!(names, ["ui/hud", "ui/hud"], "one name, two assets");

		let document = DocumentFile::open(
			&output_root(&workspace)
				.join("ui")
				.join("hud.cdoc"),
		)
		.expect("it reads")
		.to_document_data()
		.expect("and parses");

		assert_eq!(document.program, "ui/hud", "and the document names the program");
	}

	#[test]
	fn an_output_from_another_format_version_is_rebuilt_without_being_asked() {
		let workspace = workspace("stale-format");
		put(&workspace, "meshes/box.obj", CUBE_OBJ);
		run(&workspace, false);

		let output = output_root(&workspace)
			.join("meshes")
			.join("box.cmesh");

		// the version word sits at bytes 8..12, right after the magic.
		let mut bytes = fs::read(&output).expect("the output is there");
		bytes[8..12].copy_from_slice(&(format::FORMAT_VERSION + 1).to_le_bytes());
		fs::write(&output, &bytes).expect("and is rewritten as a newer format");

		let report = run(&workspace, false);

		assert_eq!(
			report.compiled.len(),
			1,
			"an output this build cannot read is stale however new it is"
		);
		assert_eq!(
			MeshFile::open(&output)
				.expect("and reads back")
				.header()
				.version,
			format::FORMAT_VERSION,
			"as this build's version"
		);
	}

	#[test]
	fn force_rebuilds_what_is_already_current() {
		let workspace = workspace("force");
		put(&workspace, "meshes/box.obj", CUBE_OBJ);
		run(&workspace, false);

		let forced = run(&workspace, true);

		assert_eq!(forced.compiled.len(), 1, "force ignores the timestamps");
		assert_eq!(forced.unchanged, 0, "and skips nothing");
	}

	#[test]
	fn one_bad_source_does_not_stop_the_others() {
		let workspace = workspace("partial-failure");
		put(&workspace, "meshes/good.obj", CUBE_OBJ);
		put(&workspace, "meshes/bad.obj", "v 0 0 0\nf 1 9 9\n");

		let report = run(&workspace, false);

		assert_eq!(report.compiled.len(), 1, "the good one is written");
		assert_eq!(report.failed.len(), 1, "and the bad one is reported");
		assert!(
			report.failed[0]
				.error
				.to_string()
				.contains("bad.obj:2:"),
			"named the file and the line to look at: {}",
			report.failed[0].error
		);
		assert!(
			!output_root(&workspace)
				.join("meshes")
				.join("bad.cmesh")
				.exists(),
			"and nothing half-written is left behind"
		);
	}

	#[test]
	fn an_output_whose_source_is_gone_is_removed() {
		let workspace = workspace("prune");
		let source = put(&workspace, "meshes/box.obj", CUBE_OBJ);
		run(&workspace, false);

		let output = output_root(&workspace)
			.join("meshes")
			.join("box.cmesh");

		assert!(output.exists(), "the output is there to start with");

		fs::remove_file(&source).expect("the source is deleted");
		let report = run(&workspace, false);

		assert_eq!(report.removed, vec![output.clone()], "and the output goes with it");
		assert!(!output.exists(), "really goes");
		assert!(!report.is_quiet(), "which counts as the run having done something");
	}

	#[test]
	fn a_file_the_compiler_has_no_importer_for_is_left_alone() {
		let workspace = workspace("unknown-extension");
		put(&workspace, "meshes/notes.txt", "this is not geometry");

		let report = run(&workspace, false);

		assert!(report.compiled.is_empty(), "nothing to compile");
		assert!(report.failed.is_empty(), "and nothing to complain about");
	}

	#[test]
	fn a_source_tree_that_is_not_there_is_an_error_rather_than_an_empty_run() {
		let workspace = workspace("missing-root");
		let root = workspace.join("nowhere");
		let error = compile_dir(&root, &output_root(&workspace), false)
			.expect_err("a typo in the path should not look like an empty project");

		assert!(error.to_string().contains("nowhere"), "{error}");
	}

	#[test]
	fn a_sidecar_beside_an_obj_moves_its_vertices() {
		let workspace = workspace("mesh-guided");

		put(&workspace, "meshes/box.obj", CUBE_OBJ);
		run(&workspace, false);

		let output = output_root(&workspace)
			.join("meshes")
			.join("box.cmesh");
		let before = MeshFile::open(&output)
			.expect("it reads")
			.to_mesh_data();

		fs::write(
			source_root(&workspace)
				.join("meshes")
				.join("box.obj.model"),
			r#"{ "position": [0, 10, 0], "scale": [2, 2, 2] }"#,
		)
		.expect("the sidecar is written");

		let report = run(&workspace, false);

		assert_eq!(report.failed.len(), 0, "{:?}", report.failed);

		let after = MeshFile::open(&output)
			.expect("it reads")
			.to_mesh_data();

		assert_eq!(after.vertices.len(), before.vertices.len(), "the same geometry");

		let (low, high) = after.bounds();
		let (was_low, was_high) = before.bounds();

		assert!(
			low.abs_diff_eq(was_low * 2.0 + Vec3::Y * 10.0, 1e-4)
				&& high.abs_diff_eq(was_high * 2.0 + Vec3::Y * 10.0, 1e-4),
			"twice the size and lifted by ten: {low:?}..{high:?} against \
			 {was_low:?}..{was_high:?}"
		);
		assert!(
			after
				.vertices
				.iter()
				.all(|vertex| (Vec3::from_array(vertex.normal).length() - 1.0).abs() < 1e-3),
			"and every normal is still a unit vector"
		);

		drop(fs::remove_dir_all(&workspace));
	}

	#[test]
	fn a_sidecar_beside_an_obj_that_says_more_than_it_can_mean_is_refused() {
		let workspace = workspace("mesh-guided-too-much");

		put(&workspace, "meshes/box.obj", CUBE_OBJ);
		fs::write(
			source_root(&workspace)
				.join("meshes")
				.join("box.obj.model"),
			r#"{ "skip": ["nothing"] }"#,
		)
		.expect("the sidecar is written");

		let report = run(&workspace, false);

		assert_eq!(report.failed.len(), 1, "refused rather than half-obeyed");
		assert!(
			report.failed[0]
				.error
				.to_string()
				.contains("one mesh with no names"),
			"saying why: {}",
			report.failed[0].error
		);

		drop(fs::remove_dir_all(&workspace));
	}

	#[test]
	fn deleting_a_sidecar_rebuilds_the_mesh_it_stood_beside() {
		let workspace = workspace("mesh-guide-deleted");

		put(&workspace, "meshes/box.obj", CUBE_OBJ);

		let sidecar = source_root(&workspace)
			.join("meshes")
			.join("box.obj.model");

		fs::write(&sidecar, r#"{ "scale": [8, 8, 8] }"#).expect("the sidecar is written");
		run(&workspace, false);

		let output = output_root(&workspace)
			.join("meshes")
			.join("box.cmesh");

		assert_eq!(
			format::flags_of(&output),
			Some(format::GUIDED),
			"the mesh says it was built through one"
		);

		fs::remove_file(&sidecar).expect("and then somebody deletes it");

		assert_eq!(run(&workspace, false).compiled.len(), 1, "which is a rebuild");
		assert_eq!(format::flags_of(&output), Some(0), "and the mark is gone with the file");

		let (low, high) = MeshFile::open(&output)
			.expect("it reads")
			.to_mesh_data()
			.bounds();

		assert!(
			low.abs_diff_eq(Vec3::splat(-0.5), 1e-4) && high.abs_diff_eq(Vec3::splat(0.5), 1e-4),
			"the cube is a unit cube again: {low:?}..{high:?}"
		);
		assert_eq!(run(&workspace, false).compiled.len(), 0, "and settles there");

		drop(fs::remove_dir_all(&workspace));
	}
}

#[cfg(test)]
mod model_tests {
	use std::{thread::sleep, time::Duration};

	use super::{
		super::{format::MeshFile, model::ModelFile, texture::TextureFile},
		*,
	};

	/// A scene an exporter wrote, with its pictures inside it.
	const PACKED: &[u8] = include_bytes!("gltf/fixtures/model.glb");

	/// The same scene written as a document, and the three files it names.
	const LOOSE: &[u8] = include_bytes!("gltf/fixtures/model.gltf");

	/// Its buffer.
	const BUFFER: &[u8] = include_bytes!("gltf/fixtures/model.bin");

	/// Its color picture.
	const COLOR: &[u8] = include_bytes!("gltf/fixtures/tiles.png");

	/// Its normal map.
	const BUMP: &[u8] = include_bytes!("gltf/fixtures/tiles_normal.png");

	/// The same checker as a jpeg, which the specification allows a model to
	/// name just as readily.
	const PHOTO: &[u8] = include_bytes!("gltf/fixtures/tiles.jpg");

	/// A directory nobody else is using, with a source tree in it.
	fn workspace(name: &str) -> PathBuf {
		let dir = std::env::temp_dir()
			.join("colby-model-compile-tests")
			.join(name);

		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(source_root(&dir).join("models")).expect("the fixture is made");

		dir
	}

	/// Writes a source file into a fixture workspace.
	fn put(workspace: &Path, relative: &str, bytes: &[u8]) -> PathBuf {
		let path = source_root(workspace).join(relative);

		if let Some(parent) = path.parent() {
			fs::create_dir_all(parent).expect("the directory is made");
		}

		fs::write(&path, bytes).expect("the source is written");

		path
	}

	/// Compiles a fixture workspace.
	fn run(workspace: &Path, force: bool) -> Report {
		compile_dir(&source_root(workspace), &output_root(workspace), force)
			.expect("the tree compiles")
	}

	/// Where an asset name lands in an output tree.
	fn at(out: &Path, name: &str, extension: &str) -> PathBuf {
		name.split('/')
			.fold(out.to_path_buf(), |path, part| path.join(part))
			.with_extension(extension)
	}

	/// The model a fixture workspace compiled.
	fn compiled(dir: &Path) -> model::ModelData {
		ModelFile::open(
			&output_root(dir)
				.join("models")
				.join("lamp.cmodel"),
		)
		.expect("the model reads")
		.to_model_data()
	}

	#[test]
	fn a_model_is_rebuilt_when_a_file_it_wrote_beside_itself_is_left_behind() {
		let dir = workspace("stale-beside");

		put(&dir, "models/lamp.glb", PACKED);
		run(&dir, false);

		// only the .cmodel carries the model format's own version, so a bump
		// to .cmesh or .ctex is invisible from it. Push one of its meshes back
		// a version, which is what every model in a tree looks like the day
		// the mesh format moves.
		let mesh = fs::read_dir(output_root(&dir).join("models").join("lamp"))
			.expect("the directory is there")
			.flatten()
			.map(|entry| entry.path())
			.find(|path| Kind::of_output(path) == Some(Kind::Mesh))
			.expect("the model wrote at least one mesh");
		let mut bytes = fs::read(&mesh).expect("the mesh is there");
		bytes[8..12].copy_from_slice(&(format::FORMAT_VERSION - 1).to_le_bytes());
		fs::write(&mesh, &bytes).expect("and is rewritten as an older format");

		let report = run(&dir, false);

		assert_eq!(
			report.compiled.len(),
			1,
			"the model is stale because something it wrote is, and nothing else says so"
		);
		assert_eq!(
			MeshFile::open(&mesh)
				.expect("and reads back")
				.header()
				.version,
			format::FORMAT_VERSION,
			"as this build's version"
		);
	}

	#[test]
	fn a_model_writes_its_meshes_and_its_pictures_beside_itself() {
		let dir = workspace("packed");

		put(&dir, "models/lamp.glb", PACKED);

		let report = run(&dir, false);
		let out = output_root(&dir);

		assert!(report.failed.is_empty(), "{:?}", report.failed);
		assert_eq!(report.compiled.len(), 1, "one source, one entry in the report");
		assert!(out.join("models").join("lamp.cmodel").is_file());

		let mut written: Vec<String> = fs::read_dir(out.join("models").join("lamp"))
			.expect("the directory is there")
			.filter_map(|entry| {
				Some(
					entry
						.ok()?
						.file_name()
						.to_string_lossy()
						.into_owned(),
				)
			})
			.collect();

		written.sort();

		assert_eq!(written, vec![
			"arm.cmesh",
			"arm_mirrored.cmesh",
			"column.cmesh",
			"panel_0.cmesh",
			"panel_1.cmesh",
			"tiles.ctex",
			"tiles_normal.ctex",
		]);

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn everything_a_model_names_is_on_disk_under_exactly_that_name() {
		// the whole contract between this format and this compiler: a `.cmodel`
		// names its meshes and its pictures the way the registry will, so the
		// loader has nothing to do but look them up.
		let dir = workspace("names");

		put(&dir, "models/lamp.glb", PACKED);
		run(&dir, false);

		let out = output_root(&dir);
		let data = compiled(&dir);

		assert!(!data.placements.is_empty(), "and there is something to check");

		for placement in &data.placements {
			assert!(
				MeshFile::open(&at(&out, &placement.mesh, format::EXTENSION)).is_ok(),
				"{} names {} and there is no mesh there",
				placement.name,
				placement.mesh
			);
		}

		let pictures = data
			.materials
			.iter()
			.flat_map(|material| {
				[(&material.name, &material.albedo), (&material.name, &material.normal)]
			})
			.filter(|(_, named)| !named.is_empty());

		for (owner, named) in pictures {
			assert!(
				TextureFile::open(&at(&out, named, texture::EXTENSION)).is_ok(),
				"{owner} names {named} and there is no picture there"
			);
		}

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn a_material_keeps_what_it_was_made_of_and_what_stands_on_it() {
		let dir = workspace("materials");

		put(&dir, "models/lamp.glb", PACKED);
		run(&dir, false);

		let data = compiled(&dir);
		let names: Vec<&str> = data
			.materials
			.iter()
			.map(|material| material.name.as_str())
			.collect();

		assert_eq!(names, vec!["models/lamp/brass", "models/lamp/stone"]);
		assert_eq!(data.materials[1].albedo, "models/lamp/tiles");
		assert_eq!(data.materials[1].normal, "models/lamp/tiles_normal");

		let column = data
			.placements
			.iter()
			.find(|placement| placement.name == "column")
			.expect("the column stands somewhere");

		assert_eq!(column.mesh, "models/lamp/column");
		assert_eq!(column.material, "models/lamp/stone");

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn what_a_document_says_about_its_alpha_reaches_the_compiled_model() {
		let dir = workspace("alpha");

		// the shipped fixture with one material asked to blend, which is the
		// one link in the chain from a file on disk to a record in a `.cmodel`
		// that nothing else here walks: the importer is tested against
		// documents it builds itself, and the format against records it builds
		// itself, and this is the seam between the two.
		let written = std::str::from_utf8(LOOSE).expect("the fixture is text");
		let asked =
			written.replace("\"name\":\"brass\",", "\"name\":\"brass\",\"alphaMode\":\"BLEND\",");

		assert!(
			asked.contains("\"alphaMode\":\"BLEND\""),
			"the line this test edits is still in the fixture"
		);
		assert!(
			!written.contains("alphaMode"),
			"and the fixture did not already say it, which would make this test say nothing"
		);

		put(&dir, "models/lamp.gltf", asked.as_bytes());
		put(&dir, "models/model.bin", BUFFER);
		put(&dir, "models/tiles.png", COLOR);
		put(&dir, "models/tiles_normal.png", BUMP);

		let report = run(&dir, false);

		assert!(report.failed.is_empty(), "{:?}", report.failed);

		let data = compiled(&dir);

		assert_eq!(data.materials[0].name, "models/lamp/brass", "the one that was asked");
		assert_eq!(
			data.materials[0].blend,
			colby_core::abi::material::Blend::Alpha,
			"and it blends on the far side"
		);
		assert_eq!(
			data.materials[1].blend,
			colby_core::abi::material::Blend::Opaque,
			"while the one beside it, which said nothing, is solid"
		);

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn a_picture_beside_the_document_is_named_as_the_asset_it_already_is() {
		let dir = workspace("loose");

		put(&dir, "models/lamp.gltf", LOOSE);
		put(&dir, "models/model.bin", BUFFER);
		put(&dir, "models/tiles.png", COLOR);
		put(&dir, "models/tiles_normal.png", BUMP);

		let report = run(&dir, false);
		let out = output_root(&dir);

		assert!(report.failed.is_empty(), "{:?}", report.failed);
		assert_eq!(
			compiled(&dir).materials[1].albedo,
			"models/tiles",
			"the loose file's own name, not one under the model"
		);
		assert!(
			!out.join("models")
				.join("lamp")
				.join("tiles.ctex")
				.is_file(),
			"nothing was taken out of the document"
		);
		assert!(
			out.join("models").join("tiles.ctex").is_file(),
			"because the compiler found it on the same walk"
		);

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn editing_a_file_a_model_names_rebuilds_the_model() {
		let dir = workspace("linked");

		put(&dir, "models/lamp.gltf", LOOSE);
		put(&dir, "models/model.bin", BUFFER);
		put(&dir, "models/tiles.png", COLOR);

		let bump = put(&dir, "models/tiles_normal.png", BUMP);

		assert_eq!(run(&dir, false).compiled.len(), 3, "a model and its two pictures");
		assert!(run(&dir, false).is_quiet(), "and nothing is stale a moment later");

		sleep(Duration::from_millis(20));
		fs::write(&bump, BUMP).expect("the picture is touched");

		let report = run(&dir, false);
		let mut names: Vec<String> = report
			.compiled
			.iter()
			.map(|compiled| compiled.name.clone())
			.collect();

		names.sort();

		assert_eq!(
			names,
			vec!["models/lamp".to_owned(), "models/tiles_normal".to_owned()],
			"the picture and the model that names it"
		);

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn a_model_that_loses_a_piece_does_not_leave_it_behind() {
		let dir = workspace("shrink");

		put(&dir, "models/lamp.glb", PACKED);
		run(&dir, false);

		let stale = output_root(&dir)
			.join("models")
			.join("lamp")
			.join("ghost.cmesh");

		fs::write(&stale, [0_u8; 4]).expect("something else is left in there");
		run(&dir, true);

		assert!(!stale.is_file(), "the directory belongs to the model and is rewritten whole");

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn a_models_own_files_are_not_swept_away_as_orphans() {
		let dir = workspace("keep");

		put(&dir, "models/lamp.glb", PACKED);
		run(&dir, false);

		let report = run(&dir, false);

		assert!(report.removed.is_empty(), "nothing was pruned: {:?}", report.removed);
		assert!(
			output_root(&dir)
				.join("models")
				.join("lamp")
				.join("column.cmesh")
				.is_file()
		);

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn deleting_a_model_takes_everything_that_came_out_of_it() {
		let dir = workspace("delete");
		let source = put(&dir, "models/lamp.glb", PACKED);

		run(&dir, false);
		fs::remove_file(&source).expect("the model is deleted");

		let report = run(&dir, false);

		assert_eq!(report.removed.len(), 8, "the model and the seven files under it");
		assert!(
			!output_root(&dir)
				.join("models")
				.join("lamp")
				.join("column.cmesh")
				.is_file()
		);
		assert!(
			!output_root(&dir)
				.join("models")
				.join("lamp.cmodel")
				.is_file()
		);

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn a_jpeg_compiles_into_a_texture_like_any_other_picture() {
		let dir = workspace("photo");

		put(&dir, "textures/wall.jpg", PHOTO);

		let report = run(&dir, false);

		assert!(report.failed.is_empty(), "{:?}", report.failed);
		assert_eq!(report.compiled[0].name, "textures/wall");

		let file = TextureFile::open(
			&output_root(&dir)
				.join("textures")
				.join("wall.ctex"),
		)
		.expect("the texture reads back");

		assert_eq!(file.header().width, 32);
		assert!(file.header().levels > 1, "and it arrives with a chain");

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn two_pictures_under_one_name_are_refused_rather_than_one_overwriting_the_other() {
		// a second extension for one kind is what made this reachable: a `.png`
		// and a `.jpg` beside each other compile to the same `.ctex`, and which
		// of them won would depend on the order of a directory listing.
		let dir = workspace("collide");

		put(&dir, "textures/wall.png", COLOR);
		put(&dir, "textures/wall.jpg", PHOTO);

		let report = run(&dir, false);

		assert_eq!(report.compiled.len(), 1, "one of them compiled");
		assert_eq!(report.failed.len(), 1, "and the other said why it did not");
		assert!(
			report.failed[0]
				.error
				.to_string()
				.contains("cannot share one asset name"),
			"got {}",
			report.failed[0].error
		);

		drop(fs::remove_dir_all(&dir));
	}

	// ---------------------------------------------------------------------
	// the import sidecar
	// ---------------------------------------------------------------------

	#[test]
	fn a_piece_bones_move_is_not_moved_by_the_import_and_says_so() {
		// the fixture has no skin in it, so this is built by hand: a test
		// that could only be written against a file colby does not have is a
		// test that does not get written. @ref `colby-verification-loop`.
		let imported = gltf::Model {
			meshes: vec![gltf::Piece {
				name: "torso".to_owned(),
				..gltf::Piece::default()
			}],
			placements: vec![
				gltf::Placement {
					name: "torso".to_owned(),
					mesh: 0,
					skeleton: Some(0),
					// what the importer writes for a skinned node, because
					// the exchange format says its own transform is ignored
					transform: Transform::IDENTITY,
				},
				gltf::Placement {
					name: "hat".to_owned(),
					mesh: 0,
					skeleton: None,
					transform: Transform::at(Vec3::Y),
				},
			],
			skins: vec![gltf::Skin {
				name: "rig".to_owned(),
				..gltf::Skin::default()
			}],
			..gltf::Model::default()
		};
		let sidecar = import::import(r#"{ "scale": [2, 2, 2] }"#).expect("a sidecar");
		let mut warnings = Vec::new();
		let placements = placed(&imported, &sidecar, "models/citizen", &mut warnings);

		assert_eq!(placements.len(), 2, "both pieces are still written");
		assert!(
			placements[1]
				.transform
				.position
				.abs_diff_eq(Vec3::Y * 2.0, 1e-4),
			"the one nothing moves is moved by the import"
		);
		assert!(
			warnings
				.iter()
				.any(|said| said.contains("torso") && said.contains("bones")),
			"and the one bones move is said out loud rather than left to be found: {warnings:?}"
		);
		assert!(
			!warnings.iter().any(|said| said.contains("hat")),
			"while the other is not complained about: {warnings:?}"
		);

		let mut quiet = Vec::new();

		drop(placed(&imported, &import::Import::NONE, "models/citizen", &mut quiet));
		assert!(quiet.is_empty(), "and a model with no sidecar is told nothing: {quiet:?}");
	}

	#[test]
	fn a_placement_sheared_by_the_import_is_said_out_loud() {
		let imported = gltf::Model {
			meshes: vec![gltf::Piece::default()],
			placements: vec![gltf::Placement {
				name: "arm".to_owned(),
				mesh: 0,
				skeleton: None,
				transform: Transform {
					rotation: colby_core::glam::Quat::from_rotation_z(
						std::f32::consts::FRAC_PI_4,
					),
					..Transform::IDENTITY
				},
			}],
			..gltf::Model::default()
		};
		let sidecar = import::import(r#"{ "scale": [1, 4, 1] }"#).expect("an uneven one");
		let mut warnings = Vec::new();

		drop(placed(&imported, &sidecar, "models/lamp", &mut warnings));

		assert!(
			warnings
				.iter()
				.any(|said| said.contains("arm") && said.contains("sheared")),
			"the same word the flattening uses, for the same arithmetic: {warnings:?}"
		);
	}

	/// Writes a sidecar beside a source already in the tree.
	fn guide(workspace: &Path, relative: &str, text: &str) -> PathBuf {
		let source = source_root(workspace).join(relative);
		let path = import::beside(&source);

		fs::write(&path, text).expect("the sidecar is written");

		path
	}

	/// Where one of the model's own pieces was written, whether or not it was.
	fn piece(workspace: &Path, name: &str) -> PathBuf {
		output_root(workspace)
			.join("models")
			.join("lamp")
			.join(format!("{name}.{}", format::EXTENSION))
	}

	#[test]
	fn a_model_with_no_sidecar_beside_it_compiles_as_it_always_did() {
		// the negative control the rest of these stand on: if this moves, the
		// comparisons below are measuring the wrong thing
		let dir = workspace("guide-absent");

		put(&dir, "models/lamp.glb", PACKED);

		let report = run(&dir, false);

		assert_eq!(report.failed.len(), 0, "having none is not a failure: {:?}", report.failed);

		let data = compiled(&dir);

		assert!(!data.guided, "and the file says nobody guided it");
		assert_eq!(data.placements.len(), 5, "every piece the file stands");
		assert_eq!(data.materials.len(), 2, "and both of its surfaces");
		assert_eq!(
			data.placements[0].transform.scale,
			Vec3::new(0.5, 1.5, 0.5),
			"at the scale the file wrote, which is not the identity - so every test below 			 \
			 that compares against this one is comparing against a real number"
		);

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn a_sidecar_moves_every_piece_of_the_model() {
		// against the same tree compiled without one, because the fixture's
		// own placements are neither at the origin nor unscaled: a test that
		// asserted absolute numbers would be asserting the fixture
		let dir = workspace("guide-transform");

		put(&dir, "models/lamp.glb", PACKED);
		run(&dir, false);

		let before = compiled(&dir);

		guide(&dir, "models/lamp.glb", r#"{ "position": [0, 10, 0], "scale": [2, 2, 2] }"#);

		let report = run(&dir, false);

		assert_eq!(report.failed.len(), 0, "{:?}", report.failed);

		let after = compiled(&dir);

		assert!(after.guided, "the file records that it was built through one");
		assert!(!before.guided, "and the one before it did not");
		assert_eq!(after.placements.len(), before.placements.len(), "the same pieces, moved");

		for (was, now) in before.placements.iter().zip(&after.placements) {
			assert!(
				now.transform
					.scale
					.abs_diff_eq(was.transform.scale * 2.0, 1e-4),
				"{} is twice the size it was: {:?} against {:?}",
				now.name,
				now.transform.scale,
				was.transform.scale
			);
			assert!(
				now.transform
					.position
					.abs_diff_eq(was.transform.position * 2.0 + Vec3::Y * 10.0, 1e-4),
				"{} is scaled about the origin and then lifted: {:?} against {:?}",
				now.name,
				now.transform.position,
				was.transform.position
			);
		}

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn a_skipped_node_costs_its_row_and_its_mesh() {
		let dir = workspace("guide-skip");

		put(&dir, "models/lamp.glb", PACKED);
		run(&dir, false);

		let before = compiled(&dir);
		let column = before
			.placements
			.iter()
			.find(|placement| placement.name == "column")
			.expect("the fixture stands a column")
			.mesh
			.clone();

		assert!(piece(&dir, "column").is_file(), "whose geometry was written");

		guide(&dir, "models/lamp.glb", r#"{ "skip": ["column"] }"#);
		run(&dir, false);

		let after = compiled(&dir);

		assert_eq!(after.placements.len(), before.placements.len() - 1, "the row is gone");
		assert!(
			!after
				.placements
				.iter()
				.any(|placement| placement.mesh == column),
			"and nothing stands its mesh"
		);
		assert!(
			!piece(&dir, "column").is_file(),
			"so the .cmesh is not written either, which is most of what skipping is for"
		);
		// a mesh of several primitives is numbered, so the panel is two
		assert!(piece(&dir, "panel_0").is_file(), "and the rest of the model still is");
		assert!(piece(&dir, "arm_mirrored").is_file(), "the mirrored copy among it");

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn a_remapped_material_is_worn_by_name_and_the_models_own_is_not_written() {
		let dir = workspace("guide-material");

		put(&dir, "models/lamp.glb", PACKED);
		guide(&dir, "models/lamp.glb", r#"{ "materials": { "brass": "materials/brass" } }"#);
		run(&dir, false);

		let data = compiled(&dir);

		assert!(
			data.placements
				.iter()
				.any(|placement| placement.material == "materials/brass"),
			"something wears the one somebody wrote: {:?}",
			data.placements
				.iter()
				.map(|placement| placement.material.as_str())
				.collect::<Vec<_>>()
		);
		assert!(
			!data
				.materials
				.iter()
				.any(|surface| surface.name == "models/lamp/brass"),
			"and the model's own is not registered, because nothing in it names one"
		);
		assert!(
			data.materials
				.iter()
				.any(|surface| surface.name == "models/lamp/stone"),
			"while the surface nobody remapped is still the model's own"
		);

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn editing_a_sidecar_rebuilds_the_model_it_stands_beside() {
		let dir = workspace("guide-edited");

		put(&dir, "models/lamp.glb", PACKED);
		guide(&dir, "models/lamp.glb", r#"{ "scale": [2, 2, 2] }"#);
		run(&dir, false);

		assert_eq!(run(&dir, false).compiled.len(), 0, "and settles");

		sleep(Duration::from_millis(1100));
		guide(&dir, "models/lamp.glb", r#"{ "scale": [4, 4, 4] }"#);

		let report = run(&dir, false);

		assert_eq!(report.compiled.len(), 1, "the sidecar is an input like any other");
		assert!(
			compiled(&dir).placements[0]
				.transform
				.scale
				.abs_diff_eq(Vec3::new(0.5, 1.5, 0.5) * 4.0, 1e-4),
			"and the new number is what the model stands at"
		);

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn deleting_a_sidecar_rebuilds_the_model_it_stood_beside() {
		// the one thing the source tree cannot be asked: the .glb did not
		// move and the output is still newer than it, so only the mark in the
		// output knows the difference
		let dir = workspace("guide-deleted");

		put(&dir, "models/lamp.glb", PACKED);

		let sidecar = guide(&dir, "models/lamp.glb", r#"{ "scale": [8, 8, 8] }"#);

		run(&dir, false);
		assert!(compiled(&dir).guided, "built through one");

		fs::remove_file(&sidecar).expect("and then somebody deletes it");

		let report = run(&dir, false);

		assert_eq!(report.compiled.len(), 1, "which is a rebuild, not a silence");

		let data = compiled(&dir);

		assert!(!data.guided, "and the mark is gone with the file");
		assert!(
			data.placements[0]
				.transform
				.scale
				.abs_diff_eq(Vec3::new(0.5, 1.5, 0.5), 1e-4),
			"so the model stands at the size the .glb says again"
		);
		assert_eq!(run(&dir, false).compiled.len(), 0, "and settles there");

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn a_sidecar_the_compiler_cannot_read_is_one_failure_and_not_a_stopped_run() {
		let dir = workspace("guide-broken");

		put(&dir, "models/lamp.glb", PACKED);
		guide(&dir, "models/lamp.glb", r#"{ "lods": 4 }"#);

		let report = run(&dir, false);

		assert_eq!(report.compiled.len(), 0, "the model did not compile");
		assert_eq!(report.failed.len(), 1, "and said why");
		assert!(
			report.failed[0]
				.error
				.to_string()
				.contains("lods"),
			"naming the word: {}",
			report.failed[0].error
		);

		drop(fs::remove_dir_all(&dir));
	}
}
