//! The runner's executable.
//!
//! Everything that runs a world is `colby_runtime`; this is the few dozen lines
//! that turn a process into a call to it. It reads the command line, hands over
//! the facts its build script baked in - @ref [`Build`] - and reports how the
//! run ended. Nothing else lives here, so that a second executable can link
//! the same runtime and differ from this one only in what it hands over.

use std::{env, path::PathBuf, process::ExitCode};

use colby_core::error;
use colby_runtime::Build;

fn main() -> ExitCode {
	let arguments: Vec<String> = env::args().skip(1).collect();

	match colby_runtime::run(&arguments, build()) {
		| Ok(()) => ExitCode::SUCCESS,
		| Err(error) => {
			error!(%error, "colby stopped");

			ExitCode::FAILURE
		},
	}
}

/// What the build script knew, as the runtime wants it.
///
/// The workspace is baked in at build time and overridable at runtime, which is
/// what makes it possible to run the executable from somewhere else and still
/// point it at a checkout. The rest is what a rebuild of the game module has
/// to match, and only a build script can know it. @ref `build.rs`.
fn build() -> Build {
	// bound before they are copied, because `env!` expands to a literal and an
	// empty literal turned into a `String` trips a lint in exactly the build
	// whose flags are empty.
	let (cargo, profile, rustflags, package): (&str, &str, &str, &str) = (
		env!("COLBY_CARGO"),
		env!("COLBY_PROFILE"),
		env!("COLBY_ENCODED_RUSTFLAGS"),
		env!("CARGO_PKG_NAME"),
	);

	Build {
		workspace: env::var("COLBY_WORKSPACE")
			.map(PathBuf::from)
			.unwrap_or_else(|_| PathBuf::from(env!("COLBY_WORKSPACE"))),
		cargo: cargo.to_owned(),
		profile: profile.to_owned(),
		rustflags: rustflags.to_owned(),
		package: package.to_owned(),
	}
}
