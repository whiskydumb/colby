//! `colby-assetc` - the asset compiler as a command.
//!
//! The runner compiles assets in-process while it runs, so this exists for the
//! two cases that are not that: building a tree without starting the engine,
//! and looking at what a source actually compiled to. Both call the same
//! [`colby_asset::compile_dir`] the runner does, so there is no second answer
//! to what a file turns into.
//!
//! ```text
//! colby-assetc [--project DIR] [--assets DIR] [--out DIR] [--force] [--quiet]
//! ```
//!
//! The trees are a project's: `assets/` and `.colby/assets/` under the
//! directory `--project` names, or under the working directory when it names
//! none, which is where `just` runs it from. Either tree can be pointed
//! somewhere else by hand, and with both pointed there is no project to read.
//! A source that fails to compile is reported and the rest are compiled
//! anyway; the exit status is non-zero if anything failed.

use std::{path::PathBuf, process::ExitCode};

use colby_asset::{
	Project,
	compile::{self, Produced, Report},
};
use colby_core::{Result, err, glam::Vec3};

/// What the command line asked for.
#[derive(Debug, Default)]
struct Options {
	project: Option<PathBuf>,
	assets: Option<PathBuf>,
	out: Option<PathBuf>,
	force: bool,
	quiet: bool,
}

/// What to print when asked, and when the arguments make no sense.
const USAGE: &str = "\
colby-assetc - compile a colby asset tree

usage:
    colby-assetc [options]

options:
    -p, --project DIR  the project whose trees these are   [default: the working directory]
    -a, --assets DIR   the source tree to read             [default: the project's assets/]
    -o, --out DIR      the tree to write                   [default: the project's \
                     .colby/assets/]
    -f, --force        recompile even what is current
    -q, --quiet        print only failures
    -h, --help         print this
";

fn main() -> ExitCode {
	let options = match Options::parse() {
		| Ok(Some(options)) => options,
		| Ok(None) => {
			print!("{USAGE}");

			return ExitCode::SUCCESS;
		},
		| Err(error) => {
			eprintln!("colby-assetc: {error}\n\n{USAGE}");

			return ExitCode::FAILURE;
		},
	};

	let (assets, out) = match options.trees() {
		| Ok(trees) => trees,
		| Err(error) => {
			eprintln!("colby-assetc: {error}");

			return ExitCode::FAILURE;
		},
	};

	match compile::compile_dir(&assets, &out, options.force) {
		| Ok(report) => {
			let failed = report.failed.len();
			print(&options, &report);

			if failed == 0 {
				ExitCode::SUCCESS
			} else {
				ExitCode::FAILURE
			}
		},
		| Err(error) => {
			eprintln!("colby-assetc: {error}");

			ExitCode::FAILURE
		},
	}
}

impl Options {
	/// Reads the command line.
	///
	/// @return the options, or `None` when help was asked for
	fn parse() -> Result<Option<Self>> {
		let mut options = Self::default();

		let arguments: Vec<String> = std::env::args().skip(1).collect();
		let mut index = 0;
		while let Some(argument) = arguments.get(index) {
			index += 1;

			match argument.as_str() {
				| "-h" | "--help" => return Ok(None),
				| "-f" | "--force" => options.force = true,
				| "-q" | "--quiet" => options.quiet = true,
				| "-p" | "--project" => {
					options.project = Some(PathBuf::from(value(&arguments, index, argument)?));
					index += 1;
				},
				| "-a" | "--assets" => {
					options.assets = Some(PathBuf::from(value(&arguments, index, argument)?));
					index += 1;
				},
				| "-o" | "--out" => {
					options.out = Some(PathBuf::from(value(&arguments, index, argument)?));
					index += 1;
				},
				| other => return Err(err!("unknown argument {other:?}")),
			}
		}

		Ok(Some(options))
	}

	/// The two trees to compile between.
	///
	/// With both named by hand no project is read at all. Otherwise the project
	/// fills in what was not named: the one `--project` points at, or the one
	/// in the working directory.
	///
	/// @return the source tree and the output tree
	fn trees(&self) -> Result<(PathBuf, PathBuf)> {
		if let (Some(assets), Some(out)) = (&self.assets, &self.out) {
			return Ok((assets.clone(), out.clone()));
		}

		let here = PathBuf::from(".");
		let project = Project::open(self.project.as_deref().unwrap_or(&here))?;

		Ok((
			self.assets
				.clone()
				.unwrap_or_else(|| project.assets()),
			self.out
				.clone()
				.unwrap_or_else(|| project.output()),
		))
	}
}

/// Reads the value that follows a flag.
fn value(arguments: &[String], index: usize, flag: &str) -> Result<String> {
	arguments
		.get(index)
		.cloned()
		.ok_or_else(|| err!("{flag} needs a directory after it"))
}

/// Says what the run did.
fn print(options: &Options, report: &Report) {
	if !options.quiet {
		for compiled in &report.compiled {
			println!(
				"{:<24} {:>8} bytes   {}",
				compiled.name,
				compiled.bytes,
				describe(compiled.produced)
			);
		}

		for removed in &report.removed {
			println!("{:<24} removed, its source is gone", removed.display());
		}
	}

	for failure in &report.failed {
		eprintln!("{}", failure.error);
	}

	if options.quiet {
		return;
	}

	println!(
		"{} compiled, {} unchanged, {} removed, {} failed",
		report.compiled.len(),
		report.unchanged,
		report.removed.len(),
		report.failed.len()
	);
}

/// One line about what an asset turned out to be.
fn describe(produced: Produced) -> String {
	match produced {
		| Produced::Mesh { vertices, triangles, bounds } => {
			let corner =
				|corner: Vec3| format!("({:.2}, {:.2}, {:.2})", corner.x, corner.y, corner.z);

			format!(
				"{triangles:>6} tris {vertices:>6} verts   {} .. {}",
				corner(bounds.0),
				corner(bounds.1)
			)
		},
		| Produced::Texture { width, height, levels } =>
			format!("{width:>6} x {height:<6} {levels} mip levels"),
		| Produced::Sound { frames, rate, channels } => format!(
			"{frames:>6} frames at {rate} a second, {}",
			if channels == 1 { "mono" } else { "stereo" }
		),
		| Produced::Font { glyphs, width, height } =>
			format!("{glyphs:>6} glyphs in a {width} x {height} atlas"),
		| Produced::Document { nodes, rules } => format!("{nodes:>6} boxes {rules:>6} rules"),
		| Produced::Model {
			meshes,
			skeletons,
			clips,
			textures,
			materials,
			placements,
		} => format!(
			"{meshes:>6} meshes {skeletons} skeletons {clips} clips {textures} pictures \
			 {materials} materials {placements} standing"
		),
		| Produced::Scene { entities, bodies, joints } =>
			format!("{entities:>6} entities {bodies} bodies {joints} joints"),
		| Produced::Script { lines } => format!("{lines:>6} lines"),
	}
}
