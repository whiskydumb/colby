//! Small helpers with no better home.

pub mod defer;

/// Replaces the value behind a mutable reference and returns the old one.
///
/// @param state - the slot to overwrite
/// @param source - the value to move into it
/// @return the value that was there before
/// @ref [`scope_restore!`](crate::scope_restore) uses this to remember what to
/// put back.
#[inline]
pub fn exchange<T>(state: &mut T, source: T) -> T { std::mem::replace(state, source) }
