//! Build-time plumbing the runner needs at runtime.
//!
//! Two jobs. It hands the runner the few facts it cannot work out for itself -
//! which profile it was built into, which `RUSTFLAGS` produced it, where the
//! engine checkout is - so that the rebuild it launches on a source change
//! matches the build it is running. And under `-Cprefer-dynamic` it puts the
//! toolchain's shared `std` where the executable will find it, because with a
//! dynamically linked std the loader needs that file before `main` exists and
//! cargo does not place it anywhere: a copy beside the executable, which is
//! where Windows looks, and on Linux an rpath saying to look there, because
//! its loader does not on its own.

use std::{
	env, fs,
	path::{Path, PathBuf},
	process::Command,
};

fn main() {
	println!("cargo::rerun-if-changed=build.rs");
	println!("cargo::rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

	// @note: `RUSTFLAGS` is not visible to a build script; cargo forwards the
	// ``-separated `CARGO_ENCODED_RUSTFLAGS` instead. Keeping it encoded is
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
		point_loader_beside_executable();
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
/// With `-Cprefer-dynamic` every artifact imports `std-<hash>.dll` -
/// `libstd-<hash>.so` on Linux - and the loader has to find it before `main`
/// runs. Running through `cargo run` happens to work because cargo puts the
/// toolchain's library directory on the search path for the child; running
/// `target/hot/colby` directly does not, and that is how the runner is meant
/// to be started.
///
/// @param profile - the directory the executable is being written into
fn place_std_dylib(profile: &Path) {
	let Some(source) = find_std_dylib() else {
		println!(
			"cargo::warning=could not find the toolchain's shared std; a hot build will not \
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

/// Tells the Linux loader to look beside the executable.
///
/// Windows searches the executable's own directory for an import as a matter
/// of course, and that is where cargo puts `colby_core.dll` and where
/// [`place_std_dylib`] puts std. A Linux loader searches only what the
/// executable's rpath, `LD_LIBRARY_PATH` and the system say, so the rpath has
/// to say `$ORIGIN`, which is that same directory. `libcolby_core.so` is there
/// because cargo hard-links a workspace member's shared library up from
/// `deps/`, and under the name the import says, because a path package's
/// shared library carries no metadata hash in its name. The linker gets the
/// string as it is; `$ORIGIN` is expanded by the loader, not by a shell.
fn point_loader_beside_executable() {
	if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
		println!("cargo::rustc-link-arg-bins=-Wl,-rpath,$ORIGIN");
	}
}

/// Locates the shared std inside the active toolchain.
///
/// `rustc --print target-libdir` names the directory that holds it on every
/// platform, `lib/rustlib/<target>/lib` under the sysroot; only the spelling
/// of the file differs, @ref [`dylib_affixes`].
fn find_std_dylib() -> Option<PathBuf> {
	let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
	let target = env::var("TARGET").ok()?;
	let output = Command::new(rustc)
		.args(["--print", "target-libdir", "--target", &target])
		.output()
		.ok()?;

	let libdir = PathBuf::from(String::from_utf8(output.stdout).ok()?.trim());
	let (prefix, suffix) = dylib_affixes();
	let head = format!("{prefix}std-");

	fs::read_dir(libdir)
		.ok()?
		.filter_map(Result::ok)
		.map(|entry| entry.path())
		.find(|path| {
			let name = path
				.file_name()
				.unwrap_or_default()
				.to_string_lossy();

			name.starts_with(&head) && name.ends_with(suffix)
		})
}

/// How the target names a shared library.
///
/// Read from the target rather than from this build script's own platform,
/// which is the host's.
fn dylib_affixes() -> (&'static str, &'static str) {
	match env::var("CARGO_CFG_TARGET_OS").as_deref() {
		| Ok("windows") => ("", ".dll"),
		| Ok("macos" | "ios") => ("lib", ".dylib"),
		| _ => ("lib", ".so"),
	}
}
