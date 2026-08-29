//! Rendering a document offscreen and reading the pixels back.
//!
//! The other three modules can be checked without a GPU, and are. What cannot
//! is the shader: a rounded corner, an antialiased glyph and an image are all
//! decisions taken in `ui.wgsl`, and the only honest way to ask whether they
//! were taken correctly is to draw one and look at what came out. These tests
//! do the looking; @ref `colby-verification-loop` for the other half, which is
//! a person opening the png.
//!
//! Every test here skips itself on a machine with no adapter, the same way the
//! renderer's own pixel tests do: a build with no GPU should report the tests
//! that could not run rather than failing them.

#[cfg(test)]
mod tests {
	use colby_asset::html;
	use colby_core::{
		abi::{FontData, Glyph, World},
		glam::{Vec2, Vec3},
	};
	use colby_engine::{Capture, Image, Overlay, capture::distance};

	use crate::Interface;

	/// How big every test picture is.
	const SIZE: (u32, u32) = (256, 256);

	/// A font of solid square glyphs, so that "is there ink here" has an answer
	/// that does not depend on a typeface.
	///
	/// The field is fully inside everywhere, which draws a filled block of the
	/// glyph's size - exactly what a distance field of a square should do.
	fn blocks() -> FontData {
		let glyphs = (0x20..0x7F_u32)
			.map(|codepoint| Glyph {
				codepoint,
				advance: 10.0,
				bearing_x: 0.0,
				bearing_y: 10.0,
				atlas_x: 0,
				atlas_y: 0,
				atlas_width: if codepoint == 0x20 { 0 } else { 8 },
				atlas_height: if codepoint == 0x20 { 0 } else { 8 },
			})
			.collect();

		FontData {
			pixel_size: 10.0,
			ascent: 10.0,
			descent: 2.0,
			line_height: 12.0,
			spread: 4.0,
			atlas_width: 8,
			atlas_height: 8,
			// 255 is as far inside the outline as the field goes.
			atlas: vec![255; 64],
			glyphs,
		}
	}

	/// Draws one document over an empty scene and hands back the picture.
	///
	/// @param html - the document, `<style>` and all
	/// @return the pixels, or `None` when this machine has no GPU
	fn shoot(source: &str) -> Option<Image> { shoot_scrolled(source, "", 0.0) }

	/// The same, with one box scrolled first.
	///
	/// @param source - the document
	/// @param node - the `id` to scroll, or empty to scroll nothing
	/// @param offset - how far down, in layout pixels; clamped by the layout
	/// @return the pixels, or `None` when this machine has no GPU
	fn shoot_scrolled(source: &str, node: &str, offset: f32) -> Option<Image> {
		let mut capture = Capture::new(SIZE.0, SIZE.1).expect("the adapter query works")?;

		let mut world = Box::new(World::new());
		world.clear = Vec3::ZERO;
		world.fonts.insert("fonts/blocks", blocks());
		world.ui.set_viewport(
			Vec2::new(
				f32::from(u16::try_from(SIZE.0).unwrap_or(0)),
				f32::from(u16::try_from(SIZE.1).unwrap_or(0)),
			),
			1.0,
		);

		let parsed = html::parse(source, &[]).expect("the document reads");
		world.ui.insert("ui/test", parsed.document);
		let panel = world.ui.show("ui/test");
		world.ui.set_scroll(panel, node, offset);

		let mut interface = Interface::new();
		interface
			.attach(capture.device(), Capture::format())
			.expect("the interface pipeline builds");
		interface.run(&world);
		interface.prepare(capture.device(), capture.queue(), &world);

		let overlay: &mut dyn Overlay = &mut interface;

		Some(
			capture
				.shoot_with(&mut world, &mut [overlay])
				.expect("the frame renders"),
		)
	}

	/// Draws one label anchored at the origin over an empty scene.
	///
	/// No document at all, which is the point: a label is engine chrome and has
	/// to reach the screen whether or not the game is showing an interface.
	///
	/// @return the pixels, or `None` when this machine has no GPU
	fn shoot_label() -> Option<Image> {
		let mut capture = Capture::new(SIZE.0, SIZE.1).expect("the adapter query works")?;

		let mut world = Box::new(World::new());
		world.clear = Vec3::ZERO;
		world.fonts.insert("fonts/blocks", blocks());
		world.ui.set_viewport(
			Vec2::new(
				f32::from(u16::try_from(SIZE.0).unwrap_or(0)),
				f32::from(u16::try_from(SIZE.1).unwrap_or(0)),
			),
			1.0,
		);

		world.camera.position = Vec3::new(0.0, 0.0, 5.0);
		world.camera.target = Vec3::ZERO;
		world
			.debug
			.label(Vec3::ZERO, "ab", colby_core::abi::debug::WHITE);

		let mut interface = Interface::new();
		interface
			.attach(capture.device(), Capture::format())
			.expect("the interface pipeline builds");
		interface.run(&world);
		interface.prepare(capture.device(), capture.queue(), &world);

		let overlay: &mut dyn Overlay = &mut interface;

		Some(
			capture
				.shoot_with(&mut world, &mut [overlay])
				.expect("the frame renders"),
		)
	}

	/// Whether anything was drawn inside a rectangle of the picture.
	fn ink(image: &Image, left: u32, top: u32, right: u32, bottom: u32) -> bool {
		(top..bottom).any(|y| (left..right).any(|x| image.pixel(x, y)[0] > 60))
	}

	#[test]
	fn a_label_anchored_in_the_world_is_drawn_over_the_scene() {
		let Some(image) = shoot_label() else {
			return;
		};

		let middle = SIZE.0 / 2;

		assert!(
			ink(&image, middle - 20, middle - 24, middle + 20, middle),
			"the camera is aimed at the anchor and the words sit just above it"
		);
		assert!(
			!ink(&image, 0, 0, 40, 40),
			"and nothing was drawn in the corner, so this is the label rather than a full \
			 screen of something"
		);
		assert!(
			!ink(&image, middle - 20, middle + 8, middle + 20, middle + 40),
			"nor below the anchor, which is where the thing being named would be"
		);
	}

	/// A small box holding one much larger red one.
	///
	/// @param overflow - what the outer box does with what does not fit
	/// @param radius - how round the outer box is
	fn nested(overflow: &str, radius: &str) -> String {
		format!(
			"<style>body {{ padding: 0; }} #p {{ position: absolute; left: 40px; top: 40px; \
			 width: 80px; height: 80px; overflow: {overflow}; border-radius: {radius}; }} #c {{ \
			 position: absolute; left: 0; top: 0; width: 200px; height: 200px; background: \
			 #ff0000; }}</style><div id=\"p\"><div id=\"c\"></div></div>"
		)
	}

	/// A hundred by eighty of window over two hundred of nothing in particular.
	///
	/// The contents are transparent on purpose: the only thing with any ink in
	/// it is the bar, so a test can ask about the bar without picking it out of
	/// what it is drawn over.
	const SCROLLING: &str = "<style>body { padding: 0; } #list { position: absolute; left: 0; \
	                         top: 0; width: 100px; height: 80px; overflow: scroll; } #inner { \
	                         width: 100px; height: 200px; }</style><div id=\"list\"><div \
	                         id=\"inner\"></div></div>";

	#[test]
	fn a_box_that_can_scroll_says_so_with_a_bar() {
		let Some(top) = shoot_scrolled(SCROLLING, "list", 0.0) else {
			return;
		};
		let Some(bottom) = shoot_scrolled(SCROLLING, "list", 4000.0) else {
			return;
		};

		// eighty of two hundred is a bar of thirty-two, at the right edge,
		// traveling forty-eight from top to bottom.
		assert!(ink(&top, 93, 4, 99, 28), "unscrolled, the bar is at the top");
		assert!(!ink(&top, 93, 52, 99, 76), "and not at the bottom");

		assert!(ink(&bottom, 93, 52, 99, 76), "scrolled to the end, it is at the bottom");
		assert!(!ink(&bottom, 93, 4, 99, 28), "and no longer at the top");
	}

	#[test]
	fn a_box_with_room_for_everything_draws_no_bar_at_all() {
		let Some(image) = shoot(&SCROLLING.replace("height: 200px", "height: 20px")) else {
			return;
		};

		assert!(
			!ink(&image, 93, 0, 99, 80),
			"nothing overflows, so there is nothing to say and no bar saying it"
		);
	}

	#[test]
	fn a_box_cuts_off_what_does_not_fit_inside_it() {
		let Some(image) = shoot(&nested("hidden", "0")) else {
			return;
		};

		// the outer box is 80 wide at (40, 40) and the inner one is 200, so
		// everything past 120 is the part that has to be gone.
		assert!(ink(&image, 50, 50, 110, 110), "what fits is drawn");
		assert!(
			!ink(&image, 130, 130, 230, 230),
			"and what does not is not, diagonally past the corner"
		);
		assert!(
			!ink(&image, 130, 50, 230, 110),
			"nor along the rows the box does occupy, which is the case a test of the corner \
			 alone would pass without clipping either axis"
		);
	}

	#[test]
	fn the_same_box_left_visible_lets_all_of_it_out() {
		let Some(image) = shoot(&nested("visible", "0")) else {
			return;
		};

		assert!(
			ink(&image, 130, 130, 230, 230),
			"nothing was asked for, so nothing is cut off - which is what says the test above \
			 measures clipping rather than a layout that never overflowed"
		);
	}

	#[test]
	fn a_clip_is_every_clipping_box_above_it_and_not_the_nearest_one() {
		// the middle box is *wider* than the one holding it, so a clip that
		// took only the nearest one would let the red out to two hundred.
		let Some(image) = shoot(
			"<style>body { padding: 0; } #p { position: absolute; left: 40px; top: 40px; \n			 \
			 width: 80px; height: 80px; overflow: hidden; } #m { position: absolute; left: \n			 \
			 0; top: 0; width: 200px; height: 200px; overflow: hidden; } #c { position: \n			 \
			 absolute; left: 0; top: 0; width: 400px; height: 400px; background: #ff0000; \n			 \
			 }</style><div id=\"p\"><div id=\"m\"><div id=\"c\"></div></div></div>",
		) else {
			return;
		};

		assert!(ink(&image, 50, 50, 110, 110), "the outermost box still shows what fits");
		assert!(
			!ink(&image, 130, 130, 230, 230),
			"and the one it holds does not get to overrule it by being bigger"
		);
	}

	#[test]
	fn a_round_box_cuts_on_its_curve_and_not_on_its_corner() {
		let Some(image) = shoot(&nested("hidden", "40px")) else {
			return;
		};

		// eighty wide and forty round is a circle at (80, 80).
		assert!(ink(&image, 70, 70, 90, 90), "the middle of the circle is filled");
		assert!(
			!ink(&image, 41, 41, 50, 50),
			"and its corner is empty, which a scissor rectangle could not have managed"
		);
		assert!(
			ink(&image, 70, 42, 90, 50),
			"while the top of the curve, between the corners, is still inside"
		);
	}

	#[test]
	fn a_box_is_painted_where_the_layout_put_it_and_nowhere_else() {
		let Some(image) = shoot(
			"<style>body { padding: 0; } #a { position: absolute; left: 40px; top: 40px; width: \
			 60px; height: 60px; background: #ff0000; }</style><div id=\"a\"></div>",
		) else {
			return;
		};

		assert!(
			distance(image.pixel(70, 70), [255, 0, 0, 255]) < 8,
			"the middle of the box is the color it was painted, got {:?}",
			image.pixel(70, 70)
		);
		assert!(
			image.pixel(20, 20)[0] < 8,
			"and outside it there is nothing but the scene's clear color, got {:?}",
			image.pixel(20, 20)
		);
		assert!(
			image.pixel(150, 150)[0] < 8,
			"on both sides of it, got {:?}",
			image.pixel(150, 150)
		);
	}

	#[test]
	fn a_rounded_corner_really_is_rounded() {
		let Some(image) = shoot(
			"<style>body { padding: 0; } #a { position: absolute; left: 0; top: 0; width: \
			 100px; height: 100px; border-radius: 30px; background: #ff0000; }</style><div \
			 id=\"a\"></div>",
		) else {
			return;
		};

		assert!(
			image.pixel(2, 2)[0] < 40,
			"the very corner is outside a thirty-pixel radius, got {:?}",
			image.pixel(2, 2)
		);
		assert!(
			distance(image.pixel(50, 50), [255, 0, 0, 255]) < 8,
			"and the middle is not, got {:?}",
			image.pixel(50, 50)
		);
		assert!(
			distance(image.pixel(50, 2), [255, 0, 0, 255]) < 8,
			"nor is the middle of the top edge, which is what makes this a rounded rectangle \
			 rather than a smaller one: got {:?}",
			image.pixel(50, 2)
		);
	}

	#[test]
	fn a_transparent_box_leaves_the_scene_showing_through() {
		let Some(image) = shoot(
			"<style>body { padding: 0; } #a { position: absolute; left: 0; top: 0; width: \
			 100px; height: 100px; background: rgba(255, 0, 0, 0.5); }</style><div \
			 id=\"a\"></div>",
		) else {
			return;
		};

		let red = image.pixel(50, 50)[0];

		assert!(
			red > 100 && red < 220,
			"half of an opaque red over black is neither of them, got {red}"
		);
	}

	#[test]
	fn text_puts_ink_on_the_screen_where_its_box_is() {
		let Some(image) = shoot(
			"<style>body { padding: 0; font-family: \"fonts/blocks\"; font-size: 10px; color: \
			 #00ff00; } #a { position: absolute; left: 20px; top: 20px; }</style><div \
			 id=\"a\">HH</div>",
		) else {
			return;
		};

		// the glyphs are eight-texel blocks on a ten-pixel advance, drawn from
		// the box's top-left corner down to its baseline.
		let inside = image.pixel(24, 24);

		assert!(
			inside[1] > 128,
			"there are letters in the box, and they are the color the stylesheet asked for: got \
			 {inside:?}"
		);
		assert!(
			image.pixel(200, 200)[1] < 8,
			"and none of them anywhere else, got {:?}",
			image.pixel(200, 200)
		);
	}

	#[test]
	fn a_hidden_document_draws_nothing_at_all() {
		let Some(image) = shoot(
			"<style>#a { display: none; width: 200px; height: 200px; background: #ff0000; \
			 }</style><div id=\"a\"></div>",
		) else {
			return;
		};

		for (x, y) in [(10, 10), (128, 128), (200, 200)] {
			assert!(
				image.pixel(x, y)[0] < 8,
				"a box that is not displayed is not drawn: {:?} at {x}, {y}",
				image.pixel(x, y)
			);
		}
	}

	#[test]
	fn a_box_inside_another_is_drawn_over_it() {
		let Some(image) = shoot(
			"<style>body { padding: 0; } #outer { position: absolute; left: 0; top: 0; width: \
			 200px; height: 200px; background: #ff0000; padding: 50px; } #inner { width: 100px; \
			 height: 100px; background: #0000ff; }</style><div id=\"outer\"><div \
			 id=\"inner\"></div></div>",
		) else {
			return;
		};

		assert!(
			distance(image.pixel(100, 100), [0, 0, 255, 255]) < 8,
			"the child is on top in the middle, got {:?}",
			image.pixel(100, 100)
		);
		assert!(
			distance(image.pixel(20, 20), [255, 0, 0, 255]) < 8,
			"and the parent's padding still shows around it, got {:?}",
			image.pixel(20, 20)
		);
	}
}
