//! The arena the game keeps its own state in.
//!
//! This exists so that adding a value to gameplay stops meaning a rebuild of
//! `colby_core` and a restart. The host owns a fixed run of bytes and never
//! looks inside; the game declares a `#[repr(C)]` struct, stamps it with a
//! layout number, and reads it back every frame.
//!
//! ```ignore
//! #[repr(C)]
//! #[derive(Clone, Copy, colby_core::bytemuck::Pod, colby_core::bytemuck::Zeroable)]
//! struct State { spin: f32, lives: u32 }
//!
//! const LAYOUT: u64 = 1;
//!
//! let (state, fresh) = world.state.get::<State>(LAYOUT);
//! ```
//!
//! Change the struct, bump `LAYOUT`, save: the arena zeroes itself, `fresh`
//! comes back `true`, and the game starts over from a known state - all without
//! the process restarting.
//!
//! Forgetting to bump it is not unsound. `T: Pod` means every bit pattern is a
//! valid `T`, so the worst case is reading yesterday's bytes as today's fields:
//! wrong numbers, not undefined behavior.

use crate::bytemuck::{self, Pod};

/// How many bytes of state the game gets.
///
/// Fixed, because a growable arena would have to live behind a pointer the
/// game could outlive. Raising this is one constant, and a restart.
pub const STATE_BYTES: usize = 4096;

/// The alignment the arena guarantees.
///
/// Enough for anything with `f64`, `u64` or a 16-byte vector in it.
pub const STATE_ALIGN: usize = 16;

/// A fixed run of bytes the game interprets however it likes.
#[repr(C, align(16))]
pub struct GameState {
	bytes: [u8; STATE_BYTES],

	/// The layout number the bytes were last written under.
	///
	/// Zero means nothing has claimed the arena yet, which is why a layout of
	/// zero is refused.
	layout: u64,
}

impl GameState {
	/// An arena nobody has claimed.
	#[must_use]
	pub const fn new() -> Self { Self { bytes: [0; STATE_BYTES], layout: 0 } }

	/// Reads the arena as `T`, resetting it if the layout has moved.
	///
	/// @param layout - the game's own version number for `T`, never zero
	/// @return the state, and whether it was just zeroed
	///
	/// # Panics
	///
	/// If `layout` is zero, which is reserved for "unclaimed".
	pub fn get<T: Pod>(&mut self, layout: u64) -> (&mut T, bool) {
		const {
			assert!(size_of::<T>() <= STATE_BYTES, "game state is larger than the arena");
			assert!(
				align_of::<T>() <= STATE_ALIGN,
				"game state needs more alignment than the arena has"
			);
		}

		assert!(layout != 0, "a layout number of zero is reserved for an unclaimed arena");

		let fresh = self.layout != layout;
		if fresh {
			self.bytes = [0; STATE_BYTES];
			self.layout = layout;
		}

		(bytemuck::from_bytes_mut(&mut self.bytes[..size_of::<T>()]), fresh)
	}

	/// The layout number currently stamped on the arena.
	#[must_use]
	pub const fn layout(&self) -> u64 { self.layout }

	/// Forgets everything, so the next [`get`](Self::get) reports itself fresh.
	pub fn reset(&mut self) {
		self.bytes = [0; STATE_BYTES];
		self.layout = 0;
	}
}

impl Default for GameState {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::bytemuck::Zeroable;

	#[repr(C)]
	#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
	struct Example {
		count: u32,
		flag: u32,
	}

	#[repr(C)]
	#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
	struct Wider {
		count: u32,
		flag: u32,
		extra: u64,
	}

	#[test]
	fn the_first_claim_hands_back_zeroes_and_says_so() {
		let mut state = GameState::new();
		let (value, fresh) = state.get::<Example>(1);

		assert!(fresh, "nobody had claimed the arena");
		assert_eq!(*value, Example { count: 0, flag: 0 }, "and it starts zeroed");
	}

	#[test]
	fn writes_survive_the_next_read() {
		let mut state = GameState::new();
		state.get::<Example>(1).0.count = 7;

		let (value, fresh) = state.get::<Example>(1);

		assert!(!fresh, "the layout has not moved");
		assert_eq!(value.count, 7, "so the bytes are still the game's");
	}

	#[test]
	fn bumping_the_layout_starts_over() {
		let mut state = GameState::new();
		state.get::<Example>(1).0.count = 7;

		let (value, fresh) = state.get::<Wider>(2);

		assert!(fresh, "a new layout number resets the arena");
		assert_eq!(value.count, 0, "and everything reads as zero");
		assert_eq!(state.layout(), 2, "the arena remembers who owns it now");
	}

	#[test]
	fn a_reset_arena_reports_itself_fresh_again() {
		let mut state = GameState::new();
		state.get::<Example>(1).0.count = 7;
		state.reset();

		let (value, fresh) = state.get::<Example>(1);

		assert!(fresh, "reset unclaims the arena");
		assert_eq!(value.count, 0, "and zeroes it");
	}

	#[test]
	#[should_panic(expected = "reserved for an unclaimed arena")]
	fn a_layout_of_zero_is_refused() {
		let mut state = GameState::new();
		let (value, _) = state.get::<Example>(0);

		assert_eq!(value.count, 0, "unreachable: the call above panics");
	}
}
