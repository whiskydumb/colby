//! Error construction macros.
//!
//! The point of the pattern is
//! that a failure is logged where it happens and turned into an `Error`
//! carrying the same text, in one expression, instead of the usual
//! `error!("..."); return Err(Error::Err("...".to_owned()))` pair that drifts
//! apart over time.
//!
//! `Err!` is `err!` wrapped in [`Result::Err`], so `err!` is the one to reach
//! for anywhere a bare `Error` is wanted.
//!
//! ```ignore
//! err!("no window");                       // Error::Err
//! err!("no window: {reason}");             // inline captures work
//! err!(Module("{name} has no api symbol"));// scoped to a variant
//! err!(Module(error!("{name} is stuck"))); // logs at ERROR, then constructs
//! return Err!(Graphics(warn!("adapter {id} vanished")));
//! ```
//!
//! @note: the same trick can be played by reaching into tracing's
//! `#[doc(hidden)]` callsite machinery, which is what it takes to record
//! structured fields into the message buffer. This one is built on the public
//! `tracing` macros instead: the same
//! log-then-construct behavior and the same callsite metadata, without pinning
//! colby to tracing internals. Structured fields inside `err!` are the thing
//! given up, and can be added back the day something needs them.

use std::{any::Any, error::Error as StdError, iter::successors};

/// Constructs an error result through [`err!`](crate::err).
///
/// Every input form accepted by `err!` is accepted here; the value is simply
/// wrapped in [`Result::Err`].
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! Err {
	($($args:tt)*) => {
		Err($crate::err!($($args)*))
	};
}

/// Constructs an [`Error`](crate::Error) from formatted or scoped input.
///
/// A form containing a tracing level logs the formatted message at that level
/// before returning the error built from the same text.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! err {
	($variant:ident($level:ident!($($args:tt)+))) => {{
		let message = ::std::format!($($args)+);
		$crate::err_log!($level, "{message}");
		$crate::error::Error::$variant(message)
	}};

	($variant:ident($($args:ident),+)) => {
		$crate::error::Error::$variant($($args),+)
	};

	($variant:ident($($args:tt)+)) => {
		$crate::error::Error::$variant(::std::format!($($args)+))
	};

	($level:ident!($($args:tt)+)) => {{
		let message = ::std::format!($($args)+);
		$crate::err_log!($level, "{message}");
		$crate::error::Error::Err(message)
	}};

	($($args:tt)+) => {
		$crate::error::Error::Err(::std::format!($($args)+))
	};
}

/// Emits an already formatted error message at the requested tracing level.
///
/// Only the levels that make sense for a failure are accepted; anything else is
/// a compile error rather than a silently dropped log line.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! err_log {
	(error, $($args:tt)+) => { $crate::tracing::error!($($args)+) };
	(warn, $($args:tt)+) => { $crate::tracing::warn!($($args)+) };
	(info, $($args:tt)+) => { $crate::tracing::info!($($args)+) };
	(debug, $($args:tt)+) => { $crate::tracing::debug!($($args)+) };
	(trace, $($args:tt)+) => { $crate::tracing::trace!($($args)+) };
}

/// Flattens an error's `source()` chain into one `; caused by: ` string.
///
/// Wrapped transport and driver errors often show a useless outer message while
/// the real cause sits two links down. Logging the whole chain at the failure
/// site makes those self-diagnosing.
///
/// @param error - the head of the chain
/// @return the chain joined into a single line
#[must_use]
pub fn error_chain(error: &dyn StdError) -> String {
	successors(Some(error), |&error| error.source())
		.map(ToString::to_string)
		.collect::<Vec<_>>()
		.join("; caused by: ")
}

/// Extracts a readable message from a caught panic payload.
///
/// `panic!` payloads are `&'static str` for literal messages and `String` for
/// formatted ones. Anything else only reports its presence.
///
/// @param payload - the value returned by a failed `catch_unwind`
/// @return an owned message, never borrowed from the payload
#[must_use]
pub fn panic_message(payload: &(dyn Any + Send)) -> String {
	if let Some(message) = payload.downcast_ref::<&'static str>() {
		return (*message).to_owned();
	}

	if let Some(message) = payload.downcast_ref::<String>() {
		return message.clone();
	}

	"unknown panic payload".to_owned()
}
