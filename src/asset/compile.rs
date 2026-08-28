//! The compiler: a source tree in, a tree of `.cmesh` out.
//!
//! Two directories, mirrored. `assets/` holds what a person edits and
//! `target/assets/` holds what the engine loads, at the same relative paths
//! with the extension swapped. A mesh's **name** is that relative path without
//! its extension and with forward slashes - `assets/meshes/crystal.obj`
//! compiles to `target/assets/meshes/crystal.cmesh` and registers as
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
	path::{Path, PathBuf},
	time::SystemTime,
};

use colby_core::{Error, Result, abi::texture::Texel, err, glam::Vec3};

use crate::{document, font, format, html, obj, png, texture, ttf};

/// The directory under a workspace that holds editable sources.
pub const SOURCE_DIR: &str = "assets";

/// Where compiled assets go, relative to a workspace.
///
/// Under `target/` because they are derived: `cargo clean` should take them,
/// and nothing should ever be tempted to edit one.
pub const OUTPUT_DIR: [&str; 2] = ["target", "assets"];

/// What a `.png` has to be called to be compiled as numbers rather than color.
///
/// Matched against the stem, so `tiles_normal.png` is one and `normal_map.png`
/// is not.
pub const NORMAL_SUFFIX: &str = "_normal";

/// The source extensions the compiler knows.
pub const SOURCE_EXTENSIONS: &[&str] =
	&[obj::EXTENSION, png::EXTENSION, ttf::EXTENSION, html::EXTENSION];

/// The extensions it writes.
pub const OUTPUT_EXTENSIONS: &[&str] =
	&[format::EXTENSION, texture::EXTENSION, font::EXTENSION, document::EXTENSION];

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
			| png::EXTENSION => Some(Self::Texture),
			| ttf::EXTENSION => Some(Self::Font),
			| html::EXTENSION => Some(Self::Document),
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
			| Self::Document => document::EXTENSION,
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
			| document::EXTENSION => Some(Self::Document),
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
			| Self::Document => document::version_of(path),
		}
	}

	/// The format version this build writes.
	#[must_use]
	pub const fn version(self) -> u32 {
		match self {
			| Self::Mesh => format::FORMAT_VERSION,
			| Self::Texture => texture::FORMAT_VERSION,
			| Self::Font => font::FORMAT_VERSION,
			| Self::Document => document::FORMAT_VERSION,
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

	/// Letters.
	Font {
		/// How many glyphs were baked.
		glyphs: usize,

		/// The atlas width.
		width: u32,

		/// The atlas height.
		height: u32,
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

/// Where a workspace keeps its editable assets.
#[must_use]
pub fn source_root(workspace: &Path) -> PathBuf { workspace.join(SOURCE_DIR) }

/// Where a workspace keeps its compiled assets.
#[must_use]
pub fn output_root(workspace: &Path) -> PathBuf {
	OUTPUT_DIR
		.iter()
		.fold(workspace.to_path_buf(), |path, part| path.join(part))
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
		| Kind::Mesh => {
			let data = obj::import_file(source)?;
			let bytes = format::encode(&data)
				.map_err(|error| err!(Asset("{}: {error}", source.display())))?;
			let produced = Produced::Mesh {
				vertices: data.vertices.len(),
				triangles: data.triangles(),
				bounds: data.bounds(),
			};

			(bytes, produced)
		},
		| Kind::Texture => {
			let data = png::import_file(source, texel_of(source))?;
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

/// Everything besides the source itself that an output was built out of.
///
/// A compiler's associated files, and colby has one case of them with two
/// halves: the stylesheets a document links to and the scripts it links to.
/// Both are folded into the `.cdoc`, so both make it stale.
fn extra_inputs(source: &Path, kind: Kind, root: &Path) -> Vec<PathBuf> {
	if kind != Kind::Document {
		return Vec::new();
	}

	let Ok(text) = fs::read_to_string(source) else {
		return Vec::new();
	};

	let mut inputs = document::stylesheets(source, &text, root);
	inputs.extend(document::scripts(source, &text, root));

	inputs
}

/// A file's modification time.
fn mtime(path: &Path) -> Result<SystemTime> { Ok(fs::metadata(path)?.modified()?) }

/// Deletes outputs whose source is gone.
///
/// Only `.cmesh` files are considered, and only ones the run did not want, so
/// nothing else that happens to live in the output tree is at risk.
fn prune(out: &Path, wanted: &[PathBuf], report: &mut Report) {
	let Ok(existing) = outputs(out) else {
		return;
	};

	for path in existing {
		if wanted.contains(&path) {
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
	use crate::{document::DocumentFile, font::FontFile, format::MeshFile, texture::TextureFile};

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
			workspace.join("target").join("assets"),
			"and outputs are derived, so they live under target"
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
	fn editing_a_script_a_document_links_to_rebuilds_the_document() {
		let workspace = workspace("script-staleness");
		put(&workspace, "ui/hud.lua", "local a = 1\n");
		put(&workspace, "ui/hud.html", "<script src=\"hud.lua\"></script>\n<div></div>\n");

		assert_eq!(run(&workspace, false).compiled.len(), 1, "the first run compiles it");
		assert_eq!(run(&workspace, false).unchanged, 1, "and the second has nothing to do");

		// as above: the rule is `newer than`, and two writes in the same
		// microsecond are not.
		sleep(Duration::from_millis(20));
		put(&workspace, "ui/hud.lua", "local a = 2\n");

		let report = run(&workspace, false);

		assert_eq!(
			report.compiled.len(),
			1,
			"the program is folded into the document, so editing it makes the document stale \
			 exactly as editing a stylesheet does"
		);

		let output = output_root(&workspace)
			.join("ui")
			.join("hud.cdoc");
		let compiled = DocumentFile::open(&output)
			.expect("it reads")
			.to_document_data()
			.expect("and parses");

		assert!(compiled.script.contains("local a = 2"), "with the new text in it");
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
	fn the_assets_that_ship_with_the_workspace_all_compile() {
		let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
			.ancestors()
			.nth(2)
			.expect("the crate sits two directories below the workspace root");

		let root = source_root(repository);
		if !root.is_dir() {
			eprintln!("no {} directory; skipping", root.display());
			return;
		}

		let out = workspace("shipped").join("out");
		let report = compile_dir(&root, &out, true).expect("the shipped tree compiles");

		assert!(report.failed.is_empty(), "every shipped asset compiles: {:?}", report.failed);
		assert!(!report.compiled.is_empty(), "and there is at least one of them");

		for compiled in &report.compiled {
			match Kind::of_output(&compiled.output) {
				| Some(Kind::Mesh) => {
					let file = MeshFile::open(&compiled.output).expect("the mesh reads back");

					assert!(
						file.header().index_count > 0,
						"{} compiled to a mesh that draws nothing",
						compiled.name
					);
				},
				| Some(Kind::Texture) => {
					let file =
						TextureFile::open(&compiled.output).expect("the texture reads back");

					assert!(
						file.header().levels > 0,
						"{} compiled to a texture with no levels",
						compiled.name
					);
				},
				| Some(Kind::Font) => {
					let file = FontFile::open(&compiled.output).expect("the font reads back");

					assert!(
						file.header().glyph_count > 0,
						"{} compiled to a font with no glyphs in it",
						compiled.name
					);
					assert!(
						file.glyphs()
							.iter()
							.any(|glyph| glyph.codepoint == u32::from('Ж')),
						"{} compiled without Cyrillic, which is half the text anyone here will 						 type",
						compiled.name
					);
				},
				| Some(Kind::Document) => {
					let file =
						DocumentFile::open(&compiled.output).expect("the document reads back");
					let data = file.to_document_data().expect("and parses back");

					assert!(
						data.nodes.len() > 1,
						"{} compiled to a document with nothing in it",
						compiled.name
					);
					assert!(
						compiled.warnings.is_empty(),
						"{} compiled with complaints: {:?}",
						compiled.name,
						compiled.warnings
					);
				},
				| None => panic!("{} compiled to something unrecognized", compiled.name),
			}
		}
	}
}
