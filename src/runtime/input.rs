//! Turning winit events into [`Input`].
//!
//! The mapping lives on the host side on purpose: `colby_core` knows nothing
//! about winit, so a winit upgrade is a change to this table and nothing else.
//! Keys colby does not name are dropped here rather than being invented
//! downstream.

use colby_core::{
	abi::{Button, Input, Key},
	trace,
};
use colby_engine::winit::{
	event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
	keyboard::{KeyCode, PhysicalKey},
};

/// How many lines of scroll one screen pixel counts for.
///
/// Trackpads and precision wheels report pixels; the rest report lines. This
/// is the usual approximation of one notch.
const PIXELS_PER_LINE: f32 = 40.0;

/// Every winit key colby has a name for.
///
/// Left and right modifiers both map to the single folded key, which is what
/// [`Key`] promises.
const KEYS: &[(KeyCode, Key)] = &[
	(KeyCode::KeyA, Key::A),
	(KeyCode::KeyB, Key::B),
	(KeyCode::KeyC, Key::C),
	(KeyCode::KeyD, Key::D),
	(KeyCode::KeyE, Key::E),
	(KeyCode::KeyF, Key::F),
	(KeyCode::KeyG, Key::G),
	(KeyCode::KeyH, Key::H),
	(KeyCode::KeyI, Key::I),
	(KeyCode::KeyJ, Key::J),
	(KeyCode::KeyK, Key::K),
	(KeyCode::KeyL, Key::L),
	(KeyCode::KeyM, Key::M),
	(KeyCode::KeyN, Key::N),
	(KeyCode::KeyO, Key::O),
	(KeyCode::KeyP, Key::P),
	(KeyCode::KeyQ, Key::Q),
	(KeyCode::KeyR, Key::R),
	(KeyCode::KeyS, Key::S),
	(KeyCode::KeyT, Key::T),
	(KeyCode::KeyU, Key::U),
	(KeyCode::KeyV, Key::V),
	(KeyCode::KeyW, Key::W),
	(KeyCode::KeyX, Key::X),
	(KeyCode::KeyY, Key::Y),
	(KeyCode::KeyZ, Key::Z),
	(KeyCode::Digit0, Key::Digit0),
	(KeyCode::Digit1, Key::Digit1),
	(KeyCode::Digit2, Key::Digit2),
	(KeyCode::Digit3, Key::Digit3),
	(KeyCode::Digit4, Key::Digit4),
	(KeyCode::Digit5, Key::Digit5),
	(KeyCode::Digit6, Key::Digit6),
	(KeyCode::Digit7, Key::Digit7),
	(KeyCode::Digit8, Key::Digit8),
	(KeyCode::Digit9, Key::Digit9),
	(KeyCode::ArrowLeft, Key::Left),
	(KeyCode::ArrowRight, Key::Right),
	(KeyCode::ArrowUp, Key::Up),
	(KeyCode::ArrowDown, Key::Down),
	(KeyCode::Space, Key::Space),
	(KeyCode::Enter, Key::Enter),
	(KeyCode::NumpadEnter, Key::Enter),
	(KeyCode::Escape, Key::Escape),
	(KeyCode::Tab, Key::Tab),
	(KeyCode::Backspace, Key::Backspace),
	(KeyCode::Delete, Key::Delete),
	(KeyCode::Home, Key::Home),
	(KeyCode::End, Key::End),
	(KeyCode::ShiftLeft, Key::Shift),
	(KeyCode::ShiftRight, Key::Shift),
	(KeyCode::ControlLeft, Key::Control),
	(KeyCode::ControlRight, Key::Control),
	(KeyCode::AltLeft, Key::Alt),
	(KeyCode::AltRight, Key::Alt),
	(KeyCode::SuperLeft, Key::Super),
	(KeyCode::SuperRight, Key::Super),
	(KeyCode::F1, Key::F1),
	(KeyCode::F2, Key::F2),
	(KeyCode::F3, Key::F3),
	(KeyCode::F4, Key::F4),
	(KeyCode::F5, Key::F5),
	(KeyCode::F6, Key::F6),
	(KeyCode::F7, Key::F7),
	(KeyCode::F8, Key::F8),
	(KeyCode::F9, Key::F9),
	(KeyCode::F10, Key::F10),
	(KeyCode::F11, Key::F11),
	(KeyCode::F12, Key::F12),
	(KeyCode::Minus, Key::Minus),
	(KeyCode::Equal, Key::Equal),
	(KeyCode::BracketLeft, Key::BracketLeft),
	(KeyCode::BracketRight, Key::BracketRight),
	(KeyCode::Semicolon, Key::Semicolon),
	(KeyCode::Quote, Key::Quote),
	(KeyCode::Comma, Key::Comma),
	(KeyCode::Period, Key::Period),
	(KeyCode::Slash, Key::Slash),
	(KeyCode::Backslash, Key::Backslash),
	(KeyCode::Backquote, Key::Backquote),
];

/// Folds one window event into the accumulating input state.
///
/// Events colby does not care about are ignored, so this can be called with
/// every event the window produces.
///
/// @param input - the state carried between frames
/// @param event - the event to fold in
pub(crate) fn apply(input: &mut Input, event: &WindowEvent) {
	match *event {
		| WindowEvent::KeyboardInput {
			event: KeyEvent { physical_key, state, ref text, .. },
			..
		} => {
			// what the window says was typed, which is the only thing that knows
			// the layout, the modifiers and whatever input method is in front of
			// them. Only on the way down: a key coming back up types nothing.
			if state == ElementState::Pressed
				&& let Some(typed) = text.as_deref()
			{
				input.type_text(typed);
			}

			// @note: `repeat` is deliberately not filtered. A held key already
			// reads as held, and set_key only raises the pressed edge on a
			// transition, so a repeat is a no-op either way.
			let down = state == ElementState::Pressed;

			// @note: a key colby has no name for is dropped, but not silently.
			// "that key does nothing" is otherwise indistinguishable from a
			// mapping bug, and the answer is one line in KEYS either way.
			match physical_key {
				| PhysicalKey::Code(code) => match key_of(code) {
					| Some(key) => {
						trace!(?key, down, "key");
						input.set_key(key, down);
					},
					| None => trace!(?code, down, "unnamed key, dropped"),
				},
				| PhysicalKey::Unidentified(code) =>
					trace!(?code, down, "unidentified key, dropped"),
			}
		},
		| WindowEvent::MouseInput { button, state, .. } =>
			if let Some(button) = button_of(button) {
				let down = state == ElementState::Pressed;

				// @note: the cursor is logged here rather than on every move,
				// which would be a line per frame. A click is the moment its
				// position matters, and this covers the whole path from the
				// window's pixels to the clip-space value the game reads.
				trace!(?button, down, cursor = ?input.cursor_clip, "button");
				input.set_button(button, down);
			},
		| WindowEvent::CursorMoved { position, .. } => input.set_cursor(position.x, position.y),
		| WindowEvent::MouseWheel { delta, .. } => {
			let lines = lines_of(delta);
			trace!(lines, "wheel");
			input.add_wheel(lines);
		},
		| WindowEvent::Focused(focused) => input.set_focus(focused),
		| WindowEvent::Resized(size) =>
			input.set_viewport(f64::from(size.width), f64::from(size.height)),
		| _ => {},
	}
}

/// The colby key a winit key code stands for, if colby names it.
fn key_of(code: KeyCode) -> Option<Key> {
	KEYS.iter()
		.find(|(winit, _)| *winit == code)
		.map(|&(_, key)| key)
}

/// The colby button a winit button stands for, if colby names it.
fn button_of(button: MouseButton) -> Option<Button> {
	match button {
		| MouseButton::Left => Some(Button::Left),
		| MouseButton::Right => Some(Button::Right),
		| MouseButton::Middle => Some(Button::Middle),
		| MouseButton::Back => Some(Button::Back),
		| MouseButton::Forward => Some(Button::Forward),
		| MouseButton::Other(_) => None,
	}
}

/// One scroll event in lines, whichever unit it arrived in.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "a scroll delta in pixels is a screen distance; f32 covers it several thousand \
	          times over"
)]
fn lines_of(delta: MouseScrollDelta) -> f32 {
	match delta {
		| MouseScrollDelta::LineDelta(_, lines) => lines,
		| MouseScrollDelta::PixelDelta(position) => position.y as f32 / PIXELS_PER_LINE,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_table_covers_every_key_colby_names() {
		let mut seen = [false; Key::COUNT];
		for &(_, key) in KEYS {
			seen[key.index()] = true;
		}

		let missing: Vec<usize> = seen
			.iter()
			.enumerate()
			.filter_map(|(index, seen)| (!seen).then_some(index))
			.collect();

		assert!(missing.is_empty(), "keys with no winit code mapped to them: {missing:?}");
	}

	#[test]
	fn both_modifiers_fold_onto_one_key() {
		assert_eq!(key_of(KeyCode::ShiftLeft), Some(Key::Shift), "left folds");
		assert_eq!(key_of(KeyCode::ShiftRight), Some(Key::Shift), "and right folds too");
	}

	#[test]
	fn an_unnamed_key_is_dropped_rather_than_guessed() {
		assert_eq!(key_of(KeyCode::F24), None, "colby does not name F24");
	}
}
