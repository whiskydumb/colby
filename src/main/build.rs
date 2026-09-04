//! Build-time plumbing the runner needs at runtime.
//!
//! Two jobs. It hands the runner the few facts it cannot work out for itself -
//! which profile it was built into, which `RUSTFLAGS` produced it, where the
//! engine checkout is - so that the rebuild it launches on a source change
//! matches the build it is running. And under `-Cprefer-dynamic` it copies the
//! toolchain's `std-*.dll` next to the executable, because with a dynamically
//! linked std the loader needs that file before `main` exists and cargo does
//! not place it anywhere.

use std::{
	env, fs,
	path::{Path, PathBuf},
};

fn main() {
	println!("cargo::rerun-if-changed=build.rs");
	println!("cargo::rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

	// @note: `RUSTFLAGS` is not visible to a build script; cargo forwards the
	// ``-separated `CARGO_ENCODED_RUSTFLAGS` instead. Keeping it encoded is
	// what lets the runner hand the exact same flags to the build it spawns,
	// including any that contain a space.
	let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
	let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
	let engine = manifest
		.parent()
		.and_then(Path::parent)
		.expect("the package sits two directories below the workspace root");

	let profile = profile_dir();

	println!("cargo::rustc-env=COLBY_ENCODED_RUSTFLAGS={rustflags}");
	println!("cargo::rustc-env=COLBY_ENGINE={}", engine.display());
	println!(
		"cargo::rustc-env=COLBY_PROFILE={}",
		profile
			.file_name()
			.map(|name| name.to_string_lossy().into_owned())
			.unwrap_or_else(|| "dev".to_owned())
	);
	println!(
		"cargo::rustc-env=COLBY_CARGO={}",
		env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned())
	);

	if rustflags.contains("prefer-dynamic") {
		place_std_dylib(&profile);
	}
}

/// The directory cargo is writing this profile's artifacts into.
///
/// There is no environment variable for it: `PROFILE` only ever says `debug` or
/// `release`, which is wrong for a custom profile like `hot`. `OUT_DIR` is
/// `<target>/<profile>/build/<pkg>-<hash>/out`, so the profile directory is
/// three levels up.
fn profile_dir() -> PathBuf {
	let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set for build scripts"));

	out.ancestors()
		.nth(3)
		.expect("OUT_DIR sits three directories below the profile directory")
		.to_path_buf()
}

/// Copies the toolchain's dynamic std beside the executable.
///
/// With `-Cprefer-dynamic` every artifact imports `std-<hash>.dll`, and Windows
/// resolves that at load time from the executable's own directory. Running
/// through `cargo run` happens to work because cargo puts the toolchain's
/// library directory on `PATH` for the child; running `target/hot/colby.exe`
/// directly does not, and that is how the runner is meant to be started.
///
/// @param profile - the directory the executable is being written into
fn place_std_dylib(profile: &Path) {
	let Some(source) = find_std_dylib() else {
		println!(
			"cargo::warning=could not find std-*.dll in the toolchain; a hot build will not \
			 start"
		);
		return;
	};

	let Some(name) = source.file_name() else {
		return;
	};

	let target = profile.join(name);
	if target.is_file() {
		return;
	}

	if let Err(error) = fs::copy(&source, &target) {
		println!("cargo::warning=copying {} failed: {error}", source.display());
	}
}

/// Locates `std-<hash>.dll` inside the active toolchain.
fn find_std_dylib() -> Option<PathBuf> {
	let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
	let output = std::process::Command::new(rustc)
		.arg("--print")
		.arg("sysroot")
		.output()
		.ok()?;

	let sysroot = PathBuf::from(String::from_utf8(output.stdout).ok()?.trim());

	fs::read_dir(sysroot.join("bin"))
		.ok()?
		.filter_map(Result::ok)
		.map(|entry| entry.path())
		.find(|path| {
			let name = path
				.file_name()
				.unwrap_or_default()
				.to_string_lossy();

			name.starts_with("std-") && name.ends_with(".dll")
		})
}
