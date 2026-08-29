//! Keyboard and mouse state, as the game sees it.
//!
//! The host fills this in from window events and hands it over once per
//! simulation step; the game only reads. Keys are colby's own enum rather than
//! winit's, so the ABI does not move when winit's does, and so the set stays
//! small enough to live in a couple of words.
//!
//! Held state persists between steps. Everything else - the pressed and
//! released edges, the cursor delta, the wheel - describes the one step it is
//! handed to, and is cleared by [`Input::end_step`] afterwards.
//!
//! **Steps, not frames**, and the difference is load-bearing now that the two
//! rates are different. Events accumulate across every rendered frame that runs
//! no step, so a click in a frame the simulation skipped is not lost; and a
//! frame that runs four catch-up steps hands the edges to the first of them
//! only, so one click is one click. The deltas - the cursor and the wheel - go
//! to that first step whole rather than being divided, because they are
//! quantities rather than rates: nothing multiplies them by `dt`, and a mouse
//! that moved forty pixels moved forty pixels however many steps the frame
//! happened to run.

/// How many bytes of typed text one step can carry.
///
/// Sixty-four, which is a sixtieth of a second of typing at any speed a person
/// reaches and about three of the longest thing an input method hands over at
/// once. What does not fit is dropped whole rather than cut, because half of a
/// character is not a character.
pub const TYPED_BYTES: usize = 64;

/// How many `u64` words the key bitsets need.
///
/// @ref [`Key::COUNT`], which must not exceed `KEY_WORDS * 64`.
pub const KEY_WORDS: usize = 2;

/// Every key colby reports.
///
/// Deliberately a curated set, not a transcription of a physical keyboard. A
/// key that turns out to be needed is one line here and one line in the host's
/// mapping table; a key nobody uses costs a bit nobody reads.
///
/// Left and right modifiers are folded together: gameplay wants to know that
/// shift is down, not which shift.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
	A = 0,
	B,
	C,
	D,
	E,
	F,
	G,
	H,
	I,
	J,
	K,
	L,
	M,
	N,
	O,
	P,
	Q,
	R,
	S,
	T,
	U,
	V,
	W,
	X,
	Y,
	Z,

	Digit0,
	Digit1,
	Digit2,
	Digit3,
	Digit4,
	Digit5,
	Digit6,
	Digit7,
	Digit8,
	Digit9,

	Left,
	Right,
	Up,
	Down,

	Space,
	Enter,
	Escape,
	Tab,
	Backspace,
	Delete,
	Home,
	End,

	Shift,
	Control,
	Alt,
	Super,

	F1,
	F2,
	F3,
	F4,
	F5,
	F6,
	F7,
	F8,
	F9,
	F10,
	F11,
	F12,

	Minus,
	Equal,
	BracketLeft,
	BracketRight,
	Semicolon,
	Quote,
	Comma,
	Period,
	Slash,
	Backslash,
	Backquote,
}

impl Key {
	/// How many keys are defined.
	pub const COUNT: usize = 75;

	/// This key's bit position in the [`Input`] bitsets.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "the discriminant of a fieldless enum is the bit index, and there is no From \
		          impl that says so"
	)]
	pub const fn index(self) -> usize { self as usize }
}

/// Every mouse button colby reports.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Button {
	Left = 0,
	Right,
	Middle,
	Back,
	Forward,
}

impl Button {
	/// How many buttons are defined.
	pub const COUNT: usize = 5;

	/// This button's bit in the [`Input`] masks.
	#[must_use]
	#[expect(clippy::as_conversions, reason = "as Key::index")]
	pub const fn mask(self) -> u32 { 1 << (self as u32) }
}

/// One step's worth of input.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Input {
	/// The surface size in physical pixels.
	pub viewport: [f32; 2],

	/// The cursor in physical pixels, origin top left.
	pub cursor: [f32; 2],

	/// The cursor in clip space: x right, y up, both in `-1.0 ..= 1.0`.
	///
	/// This is the one to compare against geometry, since the game draws in
	/// clip space. It is *not* aspect-corrected, for the same reason.
	pub cursor_clip: [f32; 2],

	/// How far the cursor moved since the previous step, in physical pixels.
	///
	/// A quantity rather than a rate: a frame that runs several catch-up steps
	/// hands the whole of it to the first of them rather than dividing it, and
	/// a frame that runs none hands it to whichever step comes next.
	pub cursor_delta: [f32; 2],

	/// Scroll since the previous step, in lines. Positive is away from the
	/// user.
	pub wheel: f32,

	/// Buttons currently down. @ref [`Button::mask`].
	pub buttons: u32,

	/// Buttons that went down during this step.
	pub buttons_pressed: u32,

	/// Buttons that came up during this step.
	pub buttons_released: u32,

	/// Keys currently down.
	pub keys: [u64; KEY_WORDS],

	/// Keys that went down during this step.
	pub keys_pressed: [u64; KEY_WORDS],

	/// Keys that came up during this step.
	pub keys_released: [u64; KEY_WORDS],

	/// Whether the window has keyboard focus.
	pub focused: bool,

	/// What was typed this step, as UTF-8. Read through
	/// [`typed`](Self::typed).
	///
	/// A fixed array rather than a pointer and a length, for the reason
	/// [`TraceInfo`](crate::abi::TraceInfo)'s ignore list is one: this ABI has
	/// no raw pointers in it outside the query table, and that is worth more
	/// than being able to type a paragraph in one step.
	///
	/// **Text, not keys.** A control character never lands here - Enter, Tab,
	/// Backspace and the arrows are in the key table, where a shortcut can ask
	/// about them without also being typed into a field. What is here is what a
	/// person meant to write, with their own layout, their own modifiers and
	/// their own input method already applied, which is the one thing a bitset
	/// of physical keys cannot say.
	typed: [u8; TYPED_BYTES],

	/// How many of `typed` are used.
	typed_len: u32,
}

impl Input {
	/// Whether a key is currently down.
	#[must_use]
	pub const fn held(&self, key: Key) -> bool { test(&self.keys, key) }

	/// Whether a key went down during this step.
	#[must_use]
	pub const fn pressed(&self, key: Key) -> bool { test(&self.keys_pressed, key) }

	/// Whether a key came up during this step.
	#[must_use]
	pub const fn released(&self, key: Key) -> bool { test(&self.keys_released, key) }

	/// Two keys read as one axis.
	///
	/// @param negative - the key that reads as `-1.0`
	/// @param positive - the key that reads as `1.0`
	/// @return `-1.0`, `0.0` or `1.0`; both keys down cancel out
	#[must_use]
	pub fn axis(&self, negative: Key, positive: Key) -> f32 {
		let mut axis = 0.0;
		if self.held(positive) {
			axis += 1.0;
		}

		if self.held(negative) {
			axis -= 1.0;
		}

		axis
	}

	/// Whether a mouse button is currently down.
	#[must_use]
	pub const fn button_held(&self, button: Button) -> bool { self.buttons & button.mask() != 0 }

	/// Whether a mouse button went down during this step.
	#[must_use]
	pub const fn button_pressed(&self, button: Button) -> bool {
		self.buttons_pressed & button.mask() != 0
	}

	/// Whether a mouse button came up during this step.
	#[must_use]
	pub const fn button_released(&self, button: Button) -> bool {
		self.buttons_released & button.mask() != 0
	}

	/// Records a key going down or coming up. Host only.
	pub fn set_key(&mut self, key: Key, down: bool) {
		let index = key.index();
		let (word, mask) = (word_of(index), mask_of(index));
		let was_down = self.keys[word] & mask != 0;

		if down {
			if !was_down {
				self.keys_pressed[word] |= mask;
			}

			self.keys[word] |= mask;
		} else {
			if was_down {
				self.keys_released[word] |= mask;
			}

			self.keys[word] &= !mask;
		}
	}

	/// Records a mouse button going down or coming up. Host only.
	pub fn set_button(&mut self, button: Button, down: bool) {
		let mask = button.mask();
		let was_down = self.buttons & mask != 0;

		if down {
			if !was_down {
				self.buttons_pressed |= mask;
			}

			self.buttons |= mask;
		} else {
			if was_down {
				self.buttons_released |= mask;
			}

			self.buttons &= !mask;
		}
	}

	/// Records the surface size. Host only.
	///
	/// @param width - physical pixels
	/// @param height - physical pixels
	pub fn set_viewport(&mut self, width: f64, height: f64) {
		self.viewport = [pixels(width), pixels(height)];
		self.place_cursor();
	}

	/// Records a cursor position in physical pixels. Host only.
	pub fn set_cursor(&mut self, x: f64, y: f64) {
		let moved = [pixels(x), pixels(y)];
		self.cursor_delta = [
			self.cursor_delta[0] + moved[0] - self.cursor[0],
			self.cursor_delta[1] + moved[1] - self.cursor[1],
		];

		self.cursor = moved;
		self.place_cursor();
	}

	/// Adds to this step's scroll total. Host only.
	pub fn add_wheel(&mut self, lines: f32) { self.wheel += lines; }

	/// Records a change of keyboard focus. Host only.
	///
	/// Losing focus releases everything: a key held across the change never
	/// reports its key-up to this window, and a key stuck down is worse than a
	/// key released a moment early.
	pub fn set_focus(&mut self, focused: bool) {
		self.focused = focused;
		if focused {
			return;
		}

		for (released, held) in self
			.keys_released
			.iter_mut()
			.zip(self.keys.iter_mut())
		{
			*released |= *held;
			*held = 0;
		}

		self.buttons_released |= self.buttons;
		self.buttons = 0;
	}

	/// Clears everything that describes one step, keeping what is held. Host
	/// only, after the game has read it.
	///
	/// Called from inside the step loop rather than once per frame, which is
	/// what makes an edge arrive exactly once. @ref the module docs.
	pub fn end_step(&mut self) {
		self.keys_pressed = [0; KEY_WORDS];
		self.keys_released = [0; KEY_WORDS];
		self.buttons_pressed = 0;
		self.buttons_released = 0;
		self.cursor_delta = [0.0, 0.0];
		self.wheel = 0.0;
		self.typed_len = 0;
	}

	/// What was typed this step.
	///
	/// Empty on almost every step, which is what makes reading it cheap enough
	/// to do unconditionally.
	#[must_use]
	pub fn typed(&self) -> &str {
		let len = usize::try_from(self.typed_len)
			.unwrap_or(0)
			.min(TYPED_BYTES);

		// valid by construction: only whole strings go in, and only when they
		// fit. The fallback is not reachable and is here so that a corrupted
		// arena reads as nothing typed rather than as a panic.
		core::str::from_utf8(&self.typed[..len]).unwrap_or("")
	}

	/// Adds what a person typed.
	///
	/// The host's, once per key event that carries text. Control characters are
	/// dropped, and so is anything that does not fit whole - @ref
	/// [`typed`](Self::typed) for both rules.
	///
	/// @param text - what the window reported, in the layout the person is
	/// using
	pub fn type_text(&mut self, text: &str) {
		for character in text.chars().filter(|it| !it.is_control()) {
			let used = usize::try_from(self.typed_len)
				.unwrap_or(TYPED_BYTES)
				.min(TYPED_BYTES);
			let wanted = character.len_utf8();

			if used + wanted > TYPED_BYTES {
				return;
			}

			character.encode_utf8(&mut self.typed[used..used + wanted]);
			self.typed_len = self
				.typed_len
				.saturating_add(u32::try_from(wanted).unwrap_or(0));
		}
	}

	/// Recomputes the clip-space cursor from the pixel cursor and the viewport.
	fn place_cursor(&mut self) {
		let [width, height] = self.viewport;
		if width <= 0.0 || height <= 0.0 {
			return;
		}

		self.cursor_clip =
			[2.0 * self.cursor[0] / width - 1.0, 1.0 - 2.0 * self.cursor[1] / height];
	}
}

impl Default for Input {
	fn default() -> Self {
		Self {
			viewport: [1.0, 1.0],
			cursor: [0.0, 0.0],
			cursor_clip: [0.0, 0.0],
			cursor_delta: [0.0, 0.0],
			wheel: 0.0,
			buttons: 0,
			buttons_pressed: 0,
			buttons_released: 0,
			keys: [0; KEY_WORDS],
			keys_pressed: [0; KEY_WORDS],
			keys_released: [0; KEY_WORDS],
			focused: true,
			typed: [0; TYPED_BYTES],
			typed_len: 0,
		}
	}
}

/// Which word of a bitset a bit index lands in.
const fn word_of(index: usize) -> usize { index / 64 }

/// The bit within its word that a bit index selects.
const fn mask_of(index: usize) -> u64 { 1 << (index % 64) }

/// Reads one key out of a bitset.
const fn test(bits: &[u64; KEY_WORDS], key: Key) -> bool {
	let index = key.index();

	bits[word_of(index)] & mask_of(index) != 0
}

/// Narrows a physical pixel measurement to `f32`.
///
/// Window sizes and cursor positions are nowhere near the point where `f32`
/// stops representing integers exactly, and everything downstream is `f32`.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "see above; a surface would have to be sixteen million pixels wide to matter"
)]
fn pixels(value: f64) -> f32 { value as f32 }

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn what_was_typed_comes_back_as_it_went_in() {
		let mut input = Input::default();

		assert_eq!(input.typed(), "", "nothing has been typed yet");

		input.type_text("ab");
		input.type_text("c");

		assert_eq!(input.typed(), "abc", "and one step's worth is one string");
	}

	#[test]
	fn a_control_character_is_a_key_rather_than_something_typed() {
		let mut input = Input::default();

		// what a window hands over for Enter, Tab and Backspace. They are in
		// the key table, and a field that took them as text would end up with
		// a tab inside it every time somebody moved on.
		input.type_text("a\r\n\tb\u{8}");

		assert_eq!(input.typed(), "ab", "only what a person meant to write");
	}

	#[test]
	fn typing_more_than_fits_drops_it_whole_rather_than_cutting_it() {
		let mut input = Input::default();
		let long = "x".repeat(TYPED_BYTES - 1);

		input.type_text(&long);
		// two bytes wide, and one byte of room left.
		input.type_text("\u{444}");
		input.type_text("y");

		assert_eq!(input.typed().len(), TYPED_BYTES, "the last one that fits still goes in");
		assert!(
			input.typed().ends_with('y'),
			"and the one that did not fit left nothing behind, got {:?}",
			input.typed()
		);
	}

	#[test]
	fn a_step_forgets_what_the_last_one_typed() {
		let mut input = Input::default();

		input.type_text("hello");
		input.end_step();

		assert_eq!(input.typed(), "", "typing is an edge, like a key press");
	}

	#[test]
	fn every_key_fits_in_the_bitset() {
		assert!(Key::COUNT <= KEY_WORDS * 64, "KEY_WORDS has to grow with the key set");
		assert_eq!(
			Key::Backquote.index() + 1,
			Key::COUNT,
			"Key::COUNT is out of step with the last variant"
		);
		assert!(
			Button::Forward.mask() < 1 << Button::COUNT,
			"Button::COUNT is out of step with the last variant"
		);
	}

	#[test]
	fn a_key_press_is_an_edge_and_a_level() {
		let mut input = Input::default();
		input.set_key(Key::W, true);

		assert!(input.pressed(Key::W), "the step it goes down");
		assert!(input.held(Key::W), "and every step after");

		input.end_step();

		assert!(!input.pressed(Key::W), "the edge lasts one step");
		assert!(input.held(Key::W), "the level does not");
	}

	#[test]
	fn a_repeated_press_is_not_a_second_edge() {
		let mut input = Input::default();
		input.set_key(Key::W, true);
		input.end_step();
		input.set_key(Key::W, true);

		assert!(!input.pressed(Key::W), "key repeat must not read as a new press");
	}

	#[test]
	fn keys_in_the_second_word_work_too() {
		let mut input = Input::default();
		input.set_key(Key::Backquote, true);

		assert!(input.held(Key::Backquote), "index 72 lands in the second word");
		assert!(!input.held(Key::A), "and does not spill into the first");
	}

	#[test]
	fn opposed_keys_cancel() {
		let mut input = Input::default();
		input.set_key(Key::D, true);

		assert!((input.axis(Key::A, Key::D) - 1.0).abs() < f32::EPSILON, "positive alone");

		input.set_key(Key::A, true);

		assert!(input.axis(Key::A, Key::D).abs() < f32::EPSILON, "both cancel out");
	}

	#[test]
	fn losing_focus_releases_what_was_held() {
		let mut input = Input::default();
		input.set_key(Key::Shift, true);
		input.set_button(Button::Left, true);
		input.end_step();

		input.set_focus(false);

		assert!(!input.held(Key::Shift), "nothing stays down through a focus loss");
		assert!(input.released(Key::Shift), "and it reads as a release, not a disappearance");
		assert!(!input.button_held(Button::Left), "buttons too");
		assert!(input.button_released(Button::Left), "buttons too");
	}

	#[test]
	fn a_frame_that_runs_no_step_keeps_the_edges_for_the_step_that_follows() {
		let mut input = Input::default();
		input.set_key(Key::Space, true);
		input.set_key(Key::Space, false);

		// no `end_step` here: this is a rendered frame the simulation had no
		// whole step for, and a tap inside one must not vanish.
		assert!(input.pressed(Key::Space), "the press survives to the next step");
		assert!(input.released(Key::Space), "and so does the release");
		assert!(!input.held(Key::Space), "even though nothing is down by then");
	}

	#[test]
	fn a_catch_up_frame_hands_the_edges_to_one_step_only() {
		let mut input = Input::default();
		input.set_button(Button::Left, true);
		input.add_wheel(3.0);

		let first = input;
		input.end_step();
		let second = input;

		assert!(first.button_pressed(Button::Left), "the first step of the frame sees the click");
		assert!(
			!second.button_pressed(Button::Left),
			"the second does not, or one click builds two houses"
		);
		assert!(second.button_held(Button::Left), "though it is still down");
		assert!(second.wheel.abs() < f32::EPSILON, "and the scroll is spent, not repeated");
	}

	#[test]
	fn the_cursor_maps_into_clip_space() {
		let mut input = Input::default();
		input.set_viewport(800.0, 600.0);
		input.set_cursor(400.0, 300.0);

		let [x, y] = input.cursor_clip;

		assert!(x.abs() < f32::EPSILON && y.abs() < f32::EPSILON, "the middle is the origin");

		input.set_cursor(800.0, 0.0);
		let [x, y] = input.cursor_clip;

		assert!(
			(x - 1.0).abs() < f32::EPSILON && (y - 1.0).abs() < f32::EPSILON,
			"top right is +1, +1: y is flipped from window coordinates"
		);
	}
}
