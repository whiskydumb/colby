//! Words anchored at a point in the world, drawn flat on the screen.
//!
//! The other half of the debug renderer. Segments are world-space geometry and
//! belong to the scene; words are glyphs out of a distance field, and every
//! piece of machinery for those already exists here - the atlas upload, the
//! batching, the shader. A second copy of all of it in the renderer, for the
//! sake of a dozen strings a frame, would be the wrong trade twice over.
//!
//! So a label is projected through the camera into layout pixels and appended
//! to the interface's own list, after the documents. Three consequences, all of
//! them the behavior a label wants anyway:
//!
//! - it is **screen-aligned and a constant size**, so a readout stays readable
//!   from across the map;
//! - it is **never occluded**, because the interface draws over the scene;
//! - it is **not clipped**, because nothing here clips - a label whose anchor
//!   is off screen is culled by its projection rather than cut in half.

use colby_core::{
	abi::{FontData, FontId, World},
	glam::{Mat4, Vec2, Vec3},
};

use crate::draw::{self, DrawList};

/// The console variable that says how big a world label is drawn.
pub const TEXT_SIZE: &str = "debug.text_size";

/// How big one is when nobody has said, in layout pixels.
pub const DEFAULT_TEXT_SIZE: f32 = 14.0;

/// The range the variable is held inside.
///
/// The lower bound is where a distance field stops resolving into letters; the
/// upper is where one label fills the window, which is a typo rather than a
/// request.
const TEXT_SIZE_RANGE: (f32, f32) = (6.0, 96.0);

/// How far outside the screen an anchor may sit and still be drawn.
///
/// In normalized device coordinates, where the screen is `-1.0 ..= 1.0`. A
/// little over, because a label is centered on its anchor and half of it can be
/// on screen while the point it names is not.
const MARGIN: f32 = 1.5;

/// Adds every label the world is holding to this frame's list.
///
/// Called after the documents, so a label draws over the interface as well as
/// over the scene. That is the right way round: it is a tool, and a tool that
/// can hide behind what it is describing is not one.
///
/// @param world - the labels, the camera, the fonts and the viewport
/// @param list - the interface's list, appended to
pub(crate) fn build(world: &World, list: &mut DrawList) {
	if world.debug.labels().is_empty() {
		return;
	}

	let viewport = world.ui.viewport();
	if viewport.x < 1.0 || viewport.y < 1.0 {
		return;
	}

	let Some(id) = font(world) else {
		return;
	};

	let data = world.fonts.data(id);
	let size = world
		.cvars
		.float(TEXT_SIZE)
		.unwrap_or(DEFAULT_TEXT_SIZE)
		.clamp(TEXT_SIZE_RANGE.0, TEXT_SIZE_RANGE.1);

	// the camera the frame is drawn through, not the one the last step left:
	// a label pinned to a moving body would otherwise swim against the body it
	// is pinned to, by exactly the interpolation.
	let view_projection = world
		.render_camera()
		.view_projection(world.aspect);

	for label in world.debug.labels() {
		let Some(at) = project(view_projection, label.at, viewport) else {
			continue;
		};

		// centered over the anchor and sitting above it, so that whatever is
		// being named stays visible under its own name.
		let measured = data.measure(&label.text, size, None);

		draw::glyphs(list, &label.text, &draw::Run {
			font: data,
			id,
			size,
			origin: [measured.x.mul_add(-0.5, at.x), at.y - measured.y],
			wrap: None,
			color: label.color.extend(1.0),
			// never, on purpose: a label is engine chrome anchored in the world,
			// and the box it would be cut by belongs to somebody else's document.
			clip: (crate::layout::UNCLIPPED, 0.0),
		});
	}
}

/// Where a point in the world lands on the screen, in layout pixels.
///
/// @return `None` if it is behind the camera, past the far plane, or far enough
/// off screen that no part of a label there could be seen
fn project(view_projection: Mat4, at: Vec3, viewport: Vec2) -> Option<Vec2> {
	let clip = view_projection * at.extend(1.0);

	// a point on the plane through the eye divides by zero, and the result is
	// not a wrong answer but a pair of nans - which pass every comparison below
	// and reach the vertex buffer. Everything actually *behind* the eye is
	// rejected by the depth test two lines down, because a negative w puts it
	// past the far plane; this guard is only about the degenerate case, and it
	// is the one a comparison cannot catch afterwards.
	if clip.w <= f32::EPSILON {
		return None;
	}

	let ndc = clip.truncate() / clip.w;
	if ndc.z > 1.0 || ndc.x.abs() > MARGIN || ndc.y.abs() > MARGIN {
		return None;
	}

	Some(Vec2::new(
		ndc.x.mul_add(0.5, 0.5) * viewport.x,
		// flipped: clip space has y up and the layout has it down.
		ndc.y.mul_add(-0.5, 0.5) * viewport.y,
	))
}

/// The font a label is drawn in: the first one the world has that has glyphs.
///
/// Not a name and not a variable. A label is engine chrome rather than part of
/// anybody's interface, so it has no stylesheet to ask, and "whatever font is
/// loaded" is right on every machine that has one.
fn font(world: &World) -> Option<FontId> {
	// from one, because slot zero of every registry is the null entry.
	(1..world.fonts.len())
		.filter_map(|slot| u32::try_from(slot).ok())
		.map(FontId::new)
		.find(|id| drawable(world.fonts.data(*id)))
}

/// Whether a font has anything to draw with.
fn drawable(font: &FontData) -> bool {
	!font.is_empty() && font.atlas_width > 0 && font.atlas_height > 0
}

#[cfg(test)]
mod tests {
	use colby_core::abi::{Glyph, debug::WHITE};

	use super::*;

	/// A font whose glyphs are eight texels square and advance by ten.
	fn baked() -> FontData {
		FontData {
			pixel_size: 10.0,
			ascent: 8.0,
			descent: 2.0,
			line_height: 12.0,
			spread: 4.0,
			atlas_width: 16,
			atlas_height: 16,
			atlas: vec![0; 256],
			glyphs: (0x20..0x7F_u32)
				.map(|codepoint| Glyph {
					codepoint,
					advance: 10.0,
					bearing_x: 0.0,
					bearing_y: 8.0,
					atlas_x: 0,
					atlas_y: 0,
					atlas_width: if codepoint == 0x20 { 0 } else { 8 },
					atlas_height: if codepoint == 0x20 { 0 } else { 8 },
				})
				.collect(),
		}
	}

	/// A world looking down the z axis at the origin, with a font loaded.
	fn looking() -> World {
		let mut world = World::new();

		world.fonts.insert("fonts/test", baked());
		world
			.ui
			.set_viewport(Vec2::new(800.0, 600.0), 1.0);
		world.aspect = 800.0 / 600.0;
		world.camera.position = Vec3::new(0.0, 0.0, 10.0);
		world.camera.target = Vec3::ZERO;

		world
	}

	#[test]
	fn a_label_at_the_middle_of_the_world_lands_in_the_middle_of_the_screen() {
		let world = looking();
		let placed = project(
			world
				.render_camera()
				.view_projection(world.aspect),
			Vec3::ZERO,
			Vec2::new(800.0, 600.0),
		)
		.expect("the origin is in front of a camera looking at it");

		assert!(
			placed.abs_diff_eq(Vec2::new(400.0, 300.0), 0.5),
			"the point the camera is aimed at is the middle of the picture, got {placed}"
		);
	}

	#[test]
	fn a_label_behind_the_camera_is_dropped_rather_than_mirrored() {
		let world = looking();
		let matrix = world
			.render_camera()
			.view_projection(world.aspect);

		// just behind the eye, which sits at ten, rather than far behind it.
		// A point far behind is culled by the depth check as well, so it says
		// nothing about the guard that is load-bearing here: dividing by a
		// negative w mirrors a point onto a screen it is not on.
		let near = project(matrix, Vec3::new(0.0, 0.0, 10.5), Vec2::new(800.0, 600.0));
		assert!(near.is_none(), "a point half a unit behind the eye is not on screen");

		let far = project(matrix, Vec3::new(0.0, 0.0, 40.0), Vec2::new(800.0, 600.0));
		assert!(far.is_none(), "nor is one thirty units behind it");
	}

	#[test]
	fn a_label_on_the_eye_itself_is_dropped_rather_than_becoming_a_nan() {
		let world = looking();
		let placed = project(
			world
				.render_camera()
				.view_projection(world.aspect),
			world.camera.position,
			Vec2::new(800.0, 600.0),
		);

		assert!(
			placed.is_none_or(Vec2::is_finite),
			"dividing by a w of zero is a nan, and a nan is less than every bound this checks 			 against, so it would reach the vertex buffer: got {placed:?}"
		);
		assert!(placed.is_none(), "and there is nowhere on screen for it to be");
	}

	#[test]
	fn a_label_far_off_to_the_side_is_dropped() {
		let world = looking();
		let placed = project(
			world
				.render_camera()
				.view_projection(world.aspect),
			Vec3::new(400.0, 0.0, 0.0),
			Vec2::new(800.0, 600.0),
		);

		assert!(placed.is_none(), "no part of a label out there could be on screen");
	}

	#[test]
	fn the_screen_has_y_down_and_the_world_has_it_up() {
		let world = looking();
		let matrix = world
			.render_camera()
			.view_projection(world.aspect);

		let above = project(matrix, Vec3::Y, Vec2::new(800.0, 600.0)).expect("in view");
		let below = project(matrix, Vec3::NEG_Y, Vec2::new(800.0, 600.0)).expect("in view");

		assert!(
			above.y < below.y,
			"something higher in the world is nearer the top of the picture: {above} against \
			 {below}"
		);
	}

	#[test]
	fn a_label_becomes_glyphs_centered_over_where_it_is_anchored() {
		let mut world = looking();
		world.debug.label(Vec3::ZERO, "ab", WHITE);

		let mut list = DrawList::default();
		build(&world, &mut list);

		assert_eq!(list.vertices.len(), 8, "two glyphs of four corners");

		let left = list
			.vertices
			.iter()
			.fold(f32::MAX, |least, vertex| least.min(vertex.position[0]));
		assert!(
			(left - (400.0 - 14.0)).abs() < 2.0,
			"two glyphs at fourteen pixels are twenty-eight wide, so the run starts half that \
			 left of the middle: got {left}"
		);
	}

	#[test]
	fn a_label_sits_above_the_point_it_names() {
		let mut world = looking();
		world.debug.label(Vec3::ZERO, "x", WHITE);

		let mut list = DrawList::default();
		build(&world, &mut list);

		let bottom = list
			.vertices
			.iter()
			.fold(f32::MIN, |most, vertex| most.max(vertex.position[1]));
		assert!(
			bottom <= 300.5,
			"the words are clear of the anchor, or they hide the thing they are naming: got \
			 {bottom}"
		);
	}

	#[test]
	fn a_world_with_no_font_loaded_draws_no_labels_rather_than_failing() {
		let mut world = World::new();
		world
			.ui
			.set_viewport(Vec2::new(800.0, 600.0), 1.0);
		world
			.debug
			.label(Vec3::ZERO, "nothing to draw it with", WHITE);

		let mut list = DrawList::default();
		build(&world, &mut list);

		assert!(list.is_empty(), "the null font has no glyphs and is not a fallback");
	}

	#[test]
	fn the_size_variable_is_held_inside_a_range() {
		let mut world = looking();
		world
			.cvars
			.var(TEXT_SIZE, colby_core::abi::Value::Float(100_000.0), "");
		world.debug.label(Vec3::ZERO, "x", WHITE);

		let mut list = DrawList::default();
		build(&world, &mut list);

		let width = list
			.vertices
			.iter()
			.fold(0.0_f32, |most, vertex| most.max(vertex.half_size[0]));
		assert!(
			width < TEXT_SIZE_RANGE.1,
			"a typo in the console should not fill the window with one letter: got {width}"
		);
	}
}
