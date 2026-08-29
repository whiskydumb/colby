//! `colby-assetc` - the asset compiler as a command.
//!
//! The runner compiles assets in-process while it runs, so this exists for the
//! two cases that are not that: building a tree without starting the engine,
//! and looking at what a source actually compiled to. Both call the same
//! [`colby_asset::compile_dir`] the runner does, so there is no second answer
//! to what a file turns into.
//!
//! ```text
//! colby-assetc [--assets DIR] [--out DIR] [--force] [--quiet]
//! ```
//!
//! Paths default to `assets` and `target/assets` under the working directory,
//! which is where `just` runs it from. A source that fails to compile is
//! reported and the rest are compiled anyway; the exit status is non-zero if
//! anything failed.

use std::{
	path::{Path, PathBuf},
	process::ExitCode,
};

use colby_asset::compile::{self, Produced, Report};
use colby_core::{Result, err, glam::Vec3};

/// What the command line asked for.
#[derive(Debug)]
struct Options {
	assets: PathBuf,
	out: PathBuf,
	force: bool,
	quiet: bool,
}

/// What to print when asked, and when the arguments make no sense.
const USAGE: &str = "\
colby-assetc - compile a colby asset tree

usage:
    colby-assetc [options]

options:
    -a, --assets DIR   the source tree to read       [default: assets]
    -o, --out DIR      the tree to write             [default: target/assets]
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

	match compile::compile_dir(&options.assets, &options.out, options.force) {
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
		let mut options = Self {
			assets: PathBuf::from(compile::SOURCE_DIR),
			out: compile::output_root(Path::new("")),
			force: false,
			quiet: false,
		};

		let arguments: Vec<String> = std::env::args().skip(1).collect();
		let mut index = 0;
		while let Some(argument) = arguments.get(index) {
			index += 1;

			match argument.as_str() {
				| "-h" | "--help" => return Ok(None),
				| "-f" | "--force" => options.force = true,
				| "-q" | "--quiet" => options.quiet = true,
				| "-a" | "--assets" => {
					options.assets = PathBuf::from(value(&arguments, index, argument)?);
					index += 1;
				},
				| "-o" | "--out" => {
					options.out = PathBuf::from(value(&arguments, index, argument)?);
					index += 1;
				},
				| other => return Err(err!("unknown argument {other:?}")),
			}
		}

		Ok(Some(options))
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
		| Produced::Font { glyphs, width, height } =>
			format!("{glyphs:>6} glyphs in a {width} x {height} atlas"),
		| Produced::Document { nodes, rules } => format!("{nodes:>6} boxes {rules:>6} rules"),
		| Produced::Model { meshes, textures, materials, placements } => format!(
			"{meshes:>6} meshes {textures} pictures {materials} materials {placements} standing"
		),
	}
}
