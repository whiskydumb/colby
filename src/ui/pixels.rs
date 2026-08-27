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
	fn shoot(source: &str) -> Option<Image> {
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
		world.ui.show("ui/test");

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
