//! Something drawn on top of a frame, after the scene and before it is shown.
//!
//! The seam exists because the renderer owns the surface: it acquires the
//! texture, records the scene into it and presents it, so anything else that
//! wants to draw into that same frame has to be handed the middle of that
//! sequence. One method, no generics, and deliberately no knowledge of what is
//! drawing - `colby_engine` does not depend on egui, and the day the game's own
//! HTML/CSS interface arrives it will come through here as well.
//!
//! An overlay records and submits its own commands. It is handed a target the
//! scene has already been drawn into, so it loads rather than clears.

use wgpu::{Device, Queue, TextureView};

/// Draws over a frame that already has a scene in it.
pub trait Overlay {
	/// Records and submits whatever goes on top.
	///
	/// @param device - the device the target belongs to
	/// @param queue - where to submit; the scene's work is already on it
	/// @param target - the frame, already drawn into and not to be cleared
	/// @param width - the target's width in pixels
	/// @param height - the target's height in pixels
	fn draw(
		&mut self,
		device: &Device,
		queue: &Queue,
		target: &TextureView,
		width: u32,
		height: u32,
	);
}
