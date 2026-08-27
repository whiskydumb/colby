//! The shared error type and its construction macros.
//!
//! One `Error` covers the whole workspace. Variants exist to be *matched on*,
//! not to be exhaustive about their cause, so anything without a caller that
//! would branch on it lands in [`Error::Err`] with a formatted message.

mod err;

use std::any::Any;

pub use self::err::{error_chain, panic_message};

/// The result type used throughout colby.
///
/// Both parameters default, so `Result` alone means `Result<(), Error>` and
/// `Result<T>` means `Result<T, Error>`.
pub type Result<T = (), E = Error> = std::result::Result<T, E>;

/// Every failure colby reports.
///
/// Construct these through [`err!`](crate::err) and [`Err!`](crate::Err) rather
/// than by hand: those macros log at the callsite and build the value in one
/// expression.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
	/// A failure with no caller that would branch on its kind.
	#[error("{0}")]
	Err(String),

	/// A failure loading, resolving or unloading a hot-reload module.
	#[error("module: {0}")]
	Module(String),

	/// A failure bringing up or driving the renderer.
	#[error("graphics: {0}")]
	Graphics(String),

	/// A failure importing, compiling or reading an asset.
	///
	/// Always carries enough to act on: which file, and which line of it where
	/// the source format has lines.
	#[error("asset: {0}")]
	Asset(String),

	/// A failure loading or running a document's interface script.
	///
	/// Carries what the VM said, which for a script is a file, a line and a
	/// sentence - the same shape the asset errors have and for the same reason:
	/// somebody is editing the file it names.
	#[error("script: {0}")]
	Script(String),

	/// A panic that unwound out of a hot-reload module and was caught at the
	/// boundary.
	///
	/// The payload is not carried along: it was allocated by the module and
	/// keeping it alive would pin the library the host is about to unload.
	#[error("PANIC! {0}")]
	Panic(String),

	/// An input or output failure, with the original `io::Error` preserved.
	#[error("I/O: {0}")]
	Io(#[from] std::io::Error),

	/// A system clock reading earlier than the reference time it was compared
	/// against.
	#[error(transparent)]
	SystemTime(#[from] std::time::SystemTimeError),
}

impl Error {
	/// Converts a caught panic payload into [`Error::Panic`].
	///
	/// The payload is inspected for the usual `&str` and `String` shapes and
	/// then dropped, leaving only an owned message. @ref
	/// [`panic_message`](self::panic_message).
	#[must_use]
	pub fn from_panic(payload: &(dyn Any + Send)) -> Self { Self::Panic(panic_message(payload)) }
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Err, err};

	#[test]
	fn a_bare_message_becomes_the_catch_all_variant() {
		let error = err!("no window");

		assert!(matches!(error, Error::Err(ref text) if text == "no window"), "{error}");
	}

	#[test]
	fn inline_captures_are_formatted() {
		let name = "colby_game";
		let error = err!("{name} has no api symbol");

		assert_eq!(error.to_string(), "colby_game has no api symbol", "captured by name");
	}

	#[test]
	fn a_variant_can_be_scoped_and_formatted_at_once() {
		let version = 7_u32;
		let error = err!(Module("abi version {version}"));

		assert_eq!(
			error.to_string(),
			"module: abi version 7",
			"the variant prefixes its Display"
		);
	}

	#[test]
	fn a_level_logs_and_still_carries_the_message() {
		let error = err!(Graphics(warn!("adapter {id} vanished", id = 3)));

		assert_eq!(error.to_string(), "graphics: adapter 3 vanished", "same text in both places");
	}

	#[test]
	fn the_result_form_wraps_the_value_form() {
		let result: Result = Err!(Module("gone"));
		let error = result.expect_err("Err! always constructs the error variant");

		assert_eq!(error.to_string(), "module: gone", "Err! is err! in a Result");
	}

	#[test]
	fn a_panic_payload_survives_as_text() {
		let payload = std::panic::catch_unwind(|| panic!("gameplay exploded"))
			.expect_err("the closure panics");
		let error = Error::from_panic(&*payload);

		assert_eq!(
			error.to_string(),
			"PANIC! gameplay exploded",
			"the message is kept, not the box"
		);
	}
}
