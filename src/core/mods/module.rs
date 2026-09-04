//! One loaded module image and its lifetime.

use std::{
	path::{Path, PathBuf},
	sync::atomic::{AtomicU64, Ordering},
	time::SystemTime,
};

use super::{Library, Symbol, canary, new, path};
use crate::{Result, debug, error};

/// Distinguishes staged copies of the same module across reloads.
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// A native module the host loaded and can unload again.
///
/// The struct owns three things that have to move together: the open library,
/// the staged copy on disk it was loaded from, and the modification time of the
/// *original* at the moment it was staged. That last one is what
/// [`changed`](Self::changed) compares against, so a reload is decided by the
/// file the build writes, not by the copy nobody else touches.
pub struct Module {
	handle: Option<Library>,
	source: PathBuf,
	image: PathBuf,
	staged: SystemTime,
}

impl Module {
	/// Loads the module a crate name resolves to.
	///
	/// @param name - the crate name, e.g. `blank_game`
	/// @return the loaded module
	pub fn from_name(name: &str) -> Result<Self> { Self::from_path(&path::from_name(name)?) }

	/// Loads the module at a path, staging a copy first.
	///
	/// @param source - the file the build wrote
	/// @return the loaded module
	pub fn from_path(source: &Path) -> Result<Self> {
		let generation = GENERATION.fetch_add(1, Ordering::Relaxed);
		let staged = path::mtime(source)?;
		let image = path::stage(source, generation)?;
		let handle = new::from_path(&image)?;

		debug!(source = ?source, image = ?image, generation, "module loaded");

		Ok(Self {
			handle: Some(handle),
			source: source.to_path_buf(),
			image,
			staged,
		})
	}

	/// Resolves an exported symbol.
	///
	/// The returned pointer is only valid until this module is unloaded, and
	/// nothing in the type system says so.
	///
	/// @param symbol - a NUL-terminated exported name
	/// @return the bound symbol
	pub fn get<Prototype>(&self, symbol: &[u8]) -> Result<Symbol<Prototype>> {
		let handle = self
			.handle
			.as_ref()
			.ok_or_else(|| crate::err!(Module("{:?} is already unloaded", self.source)))?;

		// SAFETY: this is GetProcAddress. The caller states the prototype and
		// nothing checks it; a mismatch is undefined behavior at the call, not
		// here. Callers go through the ABI version check for that reason.
		let bound = unsafe { handle.get::<Prototype>(symbol) };

		bound.map_err(|error| {
			let name = String::from_utf8_lossy(symbol);

			crate::err!(Module("{:?} has no symbol {name:?}: {error}", self.source))
		})
	}

	/// Answers whether the build has replaced the file this module came from.
	///
	/// @return `true` when the original is newer than the staged copy
	pub fn changed(&self) -> Result<bool> { Ok(path::mtime(&self.source)? > self.staged) }

	/// Unloads the module and verifies that it really left the process.
	///
	/// @ref [`canary`](super::canary) for what "really left" means and why a
	/// failure here is a log line rather than an error.
	pub fn unload(&mut self) {
		canary::prepare();
		self.close();

		if canary::check_and_reset() {
			path::unstage(&self.image);
		} else {
			let source = &self.source;
			error!(?source, "module is stuck and failed to unload");
		}
	}

	/// The name of the crate this module was built from.
	pub fn name(&self) -> Result<String> { path::to_name(&self.source) }

	/// The file the build wrote, which is what [`changed`](Self::changed)
	/// watches.
	#[must_use]
	pub fn source(&self) -> &Path { &self.source }

	/// The staged copy that is actually mapped into the process.
	#[must_use]
	pub fn image(&self) -> &Path { &self.image }

	/// Releases the library handle without checking the canary.
	fn close(&mut self) {
		if let Some(handle) = self.handle.take()
			&& let Err(error) = handle.close()
		{
			let source = &self.source;
			error!(?source, %error, "FreeLibrary failed");
		}
	}
}

impl Drop for Module {
	fn drop(&mut self) {
		if self.handle.is_some() {
			self.unload();
		}
	}
}
