//! Reading an outline font and baking it into an atlas.
//!
//! The importer half of fonts, and it runs only in the compiler - the same
//! arrangement [`png`](crate::png) has with textures. What comes out is
//! [`FontData`]: a table of metrics and one channel of signed distance field,
//! which is what the interface measures with and what the GPU samples. Nothing
//! at runtime reads a `.ttf`, and nothing at runtime rasterizes a glyph.
//!
//! Baking once at [`PIXEL_SIZE`] and scaling is the whole point of the distance
//! field. A stylesheet asks for `font-size: 13px` in one place and `40px` in
//! another; a coverage atlas would have to hold both, and this holds neither -
//! it holds where the edges are, and the shader antialiases against that at
//! whatever size the words land on screen. The usual way to get one is to
//! shell out to a tool that bakes it; this is in-process and one channel rather
//! than three.
//!
//! ```text
//! assets/fonts/hack.ttf  --[ttf::import]-->  FontData
//!                        --[font::encode]-->  .colby/assets/fonts/hack.cfont
//! ```

use ab_glyph_rasterizer::{Rasterizer, point};
use colby_core::{
	Result,
	abi::font::{FontData, Glyph},
	err,
};
use ttf_parser::{Face, GlyphId, OutlineBuilder};

use crate::sdf;

/// The extension the importer reads.
pub const EXTENSION: &str = "ttf";

/// The em size every glyph is rasterized at, in pixels.
///
/// Big enough that the distance field has room to describe a curve, small
/// enough that three hundred glyphs fit in an atlas of a megabyte. Text drawn
/// much larger than this loses its sharpest corners - the field rounds them -
/// which is the accepted cost of one atlas for every size.
pub const PIXEL_SIZE: f32 = 48.0;

/// How far either side of an outline the distance field reaches, in pixels.
///
/// Also how much room every glyph is given in the atlas beyond its own outline,
/// because a field has to have somewhere to fall off. Wider is smoother under
/// magnification and costs atlas space on every glyph at once.
pub const SPREAD: f32 = 6.0;

/// How many samples across one pixel the rasterizer takes.
///
/// The distance transform works on a hard inside/outside mask, so its answer is
/// quantized to whatever grid the mask is on. Rasterizing four times over and
/// averaging the result divides that quantization by four, which is the
/// difference between an edge that looks straight and one that looks like
/// stairs at large sizes.
pub const SUPERSAMPLE: u32 = 4;

/// How many empty texels sit between two glyphs in the atlas.
///
/// One is enough: a bilinear sample never reaches more than half a texel past
/// the edge it was asked for, and the field already carries [`SPREAD`] texels
/// of its own margin.
pub const ATLAS_GAP: u32 = 1;

/// The widest atlas the packer will try.
pub const MAX_ATLAS: u32 = 4096;

/// Which characters are baked.
///
/// Latin, Latin-1, Cyrillic, and the punctuation an interface actually reaches
/// for. A range the font does not cover costs nothing - the glyph is simply not
/// there, and text that asks for it leaves a gap. @ref
/// [`FontData::MISSING_ADVANCE`].
pub const CHARSET: &[(u32, u32)] = &[
	// printable ASCII
	(0x20, 0x7E),
	// Latin-1 supplement, without the control block
	(0xA0, 0xFF),
	// Cyrillic, the basic block
	(0x400, 0x45F),
	// dashes, quotation marks, the bullet and the ellipsis
	(0x2010, 0x2027),
	// arrows
	(0x2190, 0x2193),
];

/// Reads an outline font and bakes it.
///
/// @param bytes - the whole `.ttf`
/// @return the metrics and the atlas, or why the font could not be used
pub fn import(bytes: &[u8]) -> Result<FontData> {
	let face =
		Face::parse(bytes, 0).map_err(|error| err!(Asset("not a usable font: {error}")))?;

	let units = f32::from(face.units_per_em());
	if units <= 0.0 {
		return Err(err!(Asset("the font declares no em size, so nothing can be scaled to it")));
	}

	let scale = PIXEL_SIZE / units;
	let ascent = f32::from(face.ascender()) * scale;
	let descent = -f32::from(face.descender()) * scale;
	let line_height = (f32::from(face.ascender()) - f32::from(face.descender())
		+ f32::from(face.line_gap()))
		* scale;

	let baked = bake_glyphs(&face, scale);
	if baked.is_empty() {
		return Err(err!(Asset("the font has none of the characters colby bakes")));
	}

	let (atlas_width, atlas_height, placed) = pack(&baked)?;
	let mut atlas = vec![0_u8; count_of(atlas_width * atlas_height)];
	let mut glyphs = Vec::with_capacity(baked.len());

	for (baked, (x, y)) in baked.iter().zip(placed.iter().copied()) {
		blit(&mut atlas, atlas_width, baked, x, y);

		glyphs.push(Glyph {
			codepoint: baked.codepoint,
			advance: baked.advance,
			bearing_x: baked.bearing_x,
			bearing_y: baked.bearing_y,
			atlas_x: narrow(x),
			atlas_y: narrow(y),
			atlas_width: narrow(baked.width),
			atlas_height: narrow(baked.height),
		});
	}

	Ok(FontData {
		pixel_size: PIXEL_SIZE,
		ascent,
		descent,
		line_height,
		spread: SPREAD,
		atlas_width,
		atlas_height,
		atlas,
		glyphs,
	})
}

/// Where every glyph went in the atlas, in the order they were baked.
type Placements = Vec<(u32, u32)>;

/// One glyph after rasterization and before packing.
struct Baked {
	codepoint: u32,
	advance: f32,
	bearing_x: f32,
	bearing_y: f32,
	width: u32,
	height: u32,
	/// The distance field, one byte per texel, `width` by `height`.
	field: Vec<u8>,
}

/// Rasterizes every character of [`CHARSET`] the font has.
///
/// In codepoint order, which is the order [`FontData::glyph`] binary searches.
fn bake_glyphs(face: &Face<'_>, scale: f32) -> Vec<Baked> {
	let mut baked = Vec::new();

	for &(first, last) in CHARSET {
		for codepoint in first..=last {
			let Some(character) = char::from_u32(codepoint) else {
				continue;
			};

			let Some(id) = face.glyph_index(character) else {
				continue;
			};

			let advance = face
				.glyph_hor_advance(id)
				.map_or(0.0, |advance| f32::from(advance) * scale);

			baked.push(bake_one(face, id, codepoint, advance, scale));
		}
	}

	baked
}

/// Rasterizes one glyph into a distance field.
fn bake_one(face: &Face<'_>, id: GlyphId, codepoint: u32, advance: f32, scale: f32) -> Baked {
	let blank = Baked {
		codepoint,
		advance,
		bearing_x: 0.0,
		bearing_y: 0.0,
		width: 0,
		height: 0,
		field: Vec::new(),
	};

	// measured first, with a builder that draws nothing: the bounding box comes
	// back from `outline_glyph`, and the rasterizer has to be sized before it
	// can be drawn into. A glyph with no outline at all - a space - is a real
	// glyph with a real advance and nothing to put in the atlas.
	let mut extents = Extents::default();
	let Some(_) = face.outline_glyph(id, &mut extents) else {
		return blank;
	};

	let Some(box_of) = extents.finish() else {
		return blank;
	};

	let left = box_of[0].mul_add(scale, -SPREAD).floor();
	let top = box_of[3].mul_add(scale, SPREAD).ceil();
	let width = count(box_of[2].mul_add(scale, SPREAD).ceil() - left);
	let height = count(top - box_of[1].mul_add(scale, -SPREAD).floor());

	if width == 0 || height == 0 {
		return blank;
	}

	let field = rasterize(face, id, scale, left, top, width, height);

	Baked {
		codepoint,
		advance,
		bearing_x: left,
		bearing_y: top,
		width,
		height,
		field,
	}
}

/// Draws one glyph supersampled, then reduces it to a field of distances.
fn rasterize(
	face: &Face<'_>,
	id: GlyphId,
	scale: f32,
	left: f32,
	top: f32,
	width: u32,
	height: u32,
) -> Vec<u8> {
	let high_width = count_of(width * SUPERSAMPLE);
	let high_height = count_of(height * SUPERSAMPLE);
	let sample = scalar(SUPERSAMPLE);

	let mut outline = Outline {
		rasterizer: Rasterizer::new(high_width, high_height),
		scale: scale * sample,
		left: left * sample,
		top: top * sample,
		start: point(0.0, 0.0),
		pen: point(0.0, 0.0),
	};

	face.outline_glyph(id, &mut outline);
	outline.close();

	let mut inside = vec![false; high_width * high_height];
	outline
		.rasterizer
		.for_each_pixel(|index, coverage| {
			if let Some(slot) = inside.get_mut(index) {
				*slot = coverage >= 0.5;
			}
		});

	let field = sdf::signed_distance(&inside, high_width, high_height);

	reduce(&field, high_width, width, height)
}

/// Averages each supersampled block down to one texel of the atlas.
///
/// The average is taken over distances rather than over coverage, which is why
/// this is not the same thing as a blur: the result is still a distance, and
/// still means the same thing at every point.
fn reduce(field: &[f32], high_width: usize, width: u32, height: u32) -> Vec<u8> {
	let block = count_of(SUPERSAMPLE);
	let samples = scalar(SUPERSAMPLE * SUPERSAMPLE);
	let mut out = Vec::with_capacity(count_of(width * height));

	for y in 0..count_of(height) {
		for x in 0..count_of(width) {
			// back into output pixels: the field was measured on the
			// supersampled grid, where everything is that many times further
			// apart than it is in the atlas.
			let distance = total_over(field, high_width, x * block, y * block)
				/ samples / scalar(SUPERSAMPLE);

			out.push(encode_distance(distance));
		}
	}

	out
}

/// Adds up one supersampled block's distances.
fn total_over(field: &[f32], high_width: usize, left: usize, top: usize) -> f32 {
	let block = count_of(SUPERSAMPLE);
	let mut total = 0.0_f32;

	for row in 0..block {
		let start = (top + row) * high_width + left;

		for column in 0..block {
			total += field
				.get(start + column)
				.copied()
				.unwrap_or(-SPREAD);
		}
	}

	total
}

/// A distance in pixels as the byte the atlas stores.
///
/// `128` sits on the outline, `255` is [`SPREAD`] pixels inside it and `0` is
/// the same distance outside. @ref [`FontData::spread`].
fn encode_distance(distance: f32) -> u8 {
	let normalized = (distance / SPREAD)
		.mul_add(0.5, 0.5)
		.clamp(0.0, 1.0);

	byte(normalized * 255.0)
}

/// Copies one glyph's field into the atlas at a position.
fn blit(atlas: &mut [u8], atlas_width: u32, baked: &Baked, x: u32, y: u32) {
	let stride = count_of(atlas_width);

	for row in 0..count_of(baked.height) {
		let from = row * count_of(baked.width);
		let to = (count_of(y) + row) * stride + count_of(x);

		let Some(source) = baked
			.field
			.get(from..from + count_of(baked.width))
		else {
			continue;
		};

		let Some(target) = atlas.get_mut(to..to + count_of(baked.width)) else {
			continue;
		};

		target.copy_from_slice(source);
	}
}

/// Finds an atlas size every glyph fits in, and where each one goes.
///
/// Shelf packing: rows as tall as the tallest glyph put in them, filled left to
/// right. It wastes a little height against a real bin packer and is thirty
/// lines instead of two hundred; for glyphs, which are all about the same
/// height, the waste is single digits.
///
/// @param baked - the glyphs, in the order they will be written
/// @return the atlas width and height, and one position per glyph
fn pack(baked: &[Baked]) -> Result<(u32, u32, Placements)> {
	let mut width = 256;

	while width <= MAX_ATLAS {
		if let Some((height, placed)) = try_pack(baked, width)
			&& height <= width
		{
			return Ok((width, height.max(1), placed));
		}

		width *= 2;
	}

	Err(err!(Asset(
		"the font's {} glyphs will not fit in a {MAX_ATLAS}x{MAX_ATLAS} atlas",
		baked.len()
	)))
}

/// Lays every glyph out on shelves of a given width.
///
/// @return the height used and where each glyph went, or `None` if one glyph is
/// wider than the whole atlas
fn try_pack(baked: &[Baked], width: u32) -> Option<(u32, Placements)> {
	let mut placed = Vec::with_capacity(baked.len());
	let mut pen_x = 0;
	let mut shelf_y = 0;
	let mut shelf_height = 0;

	for glyph in baked {
		if glyph.width > width {
			return None;
		}

		if pen_x + glyph.width > width {
			shelf_y += shelf_height + ATLAS_GAP;
			shelf_height = 0;
			pen_x = 0;
		}

		placed.push((pen_x, shelf_y));
		pen_x += glyph.width + ATLAS_GAP;
		shelf_height = shelf_height.max(glyph.height);
	}

	Some((shelf_y + shelf_height, placed))
}

/// Measures a glyph's outline without drawing it.
///
/// `outline_glyph` reports a bounding box, and for a composite glyph it is the
/// one the font declares rather than the one its contours actually reach. This
/// takes the contours' word for it, which is the box the rasterizer has to be
/// sized for.
#[derive(Default)]
struct Extents {
	low: Option<[f32; 4]>,
}

impl Extents {
	/// The box the contours reached: min x, min y, max x, max y.
	fn finish(&self) -> Option<[f32; 4]> { self.low }

	fn touch(&mut self, x: f32, y: f32) {
		match &mut self.low {
			| Some(box_of) => {
				box_of[0] = box_of[0].min(x);
				box_of[1] = box_of[1].min(y);
				box_of[2] = box_of[2].max(x);
				box_of[3] = box_of[3].max(y);
			},
			| None => self.low = Some([x, y, x, y]),
		}
	}
}

impl OutlineBuilder for Extents {
	fn move_to(&mut self, x: f32, y: f32) { self.touch(x, y); }

	fn line_to(&mut self, x: f32, y: f32) { self.touch(x, y); }

	fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
		// the control point too: a curve never reaches it, but the box that
		// contains the control points contains the curve, and a box that is a
		// little too big only costs empty texels.
		self.touch(x1, y1);
		self.touch(x, y);
	}

	fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
		self.touch(x1, y1);
		self.touch(x2, y2);
		self.touch(x, y);
	}

	fn close(&mut self) {}
}

/// Feeds a glyph's contours to the rasterizer, in atlas pixels.
///
/// Font outlines have y pointing up from the baseline and the rasterizer wants
/// it pointing down from the top of the image, so every point is flipped on the
/// way through. Getting this wrong draws every letter upside down, which is at
/// least easy to see.
struct Outline {
	rasterizer: Rasterizer,
	scale: f32,
	left: f32,
	top: f32,
	start: ab_glyph_rasterizer::Point,
	pen: ab_glyph_rasterizer::Point,
}

impl Outline {
	fn at(&self, x: f32, y: f32) -> ab_glyph_rasterizer::Point {
		point(x.mul_add(self.scale, -self.left), y.mul_add(-self.scale, self.top))
	}
}

impl OutlineBuilder for Outline {
	fn move_to(&mut self, x: f32, y: f32) {
		// an open contour is closed before another one starts, or the
		// rasterizer's winding never comes back to zero and the glyph fills
		// its whole bounding box.
		self.close();
		self.pen = self.at(x, y);
		self.start = self.pen;
	}

	fn line_to(&mut self, x: f32, y: f32) {
		let next = self.at(x, y);
		self.rasterizer.draw_line(self.pen, next);
		self.pen = next;
	}

	fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
		let control = self.at(x1, y1);
		let next = self.at(x, y);
		self.rasterizer.draw_quad(self.pen, control, next);
		self.pen = next;
	}

	fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
		let first = self.at(x1, y1);
		let second = self.at(x2, y2);
		let next = self.at(x, y);
		self.rasterizer
			.draw_cubic(self.pen, first, second, next);
		self.pen = next;
	}

	fn close(&mut self) {
		if self.pen != self.start {
			self.rasterizer.draw_line(self.pen, self.start);
		}

		self.pen = self.start;
	}
}

/// A count as a number, for the arithmetic that mixes the two.
#[expect(
	clippy::as_conversions,
	clippy::cast_precision_loss,
	reason = "atlas sizes are thousands at most, four thousand times short of where f32 stops \
	          counting whole numbers"
)]
fn scalar(value: u32) -> f32 { value as f32 }

/// A number of pixels as a count, rounded up.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "clamped to a range an atlas can hold before the conversion, and a saturating cast \
	          is what a f32 to u32 conversion already is"
)]
fn count(value: f32) -> u32 { value.max(0.0).ceil().min(scalar(MAX_ATLAS)) as u32 }

/// The same, as a length.
fn count_of(value: u32) -> usize { usize::try_from(value).unwrap_or(0) }

/// A texel value from a number in `0.0 ..= 255.0`.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "the caller clamps into the byte range first, so there is nothing to truncate"
)]
fn byte(value: f32) -> u8 { value.clamp(0.0, 255.0) as u8 }

/// An atlas coordinate as it is stored.
fn narrow(value: u32) -> u16 { u16::try_from(value).unwrap_or(u16::MAX) }

#[cfg(test)]
mod tests {
	use super::*;

	/// The font every test here bakes.
	///
	/// A fixture beside this crate rather than anything a project ships: these
	/// tests are about the baking, not about a typeface, and the engine's own
	/// tree carries no assets. Hack, under the MIT license beside it.
	const fn face() -> &'static [u8] { include_bytes!("fixtures/hack.ttf") }

	#[test]
	fn a_file_that_is_not_a_font_is_refused_with_a_sentence() {
		let error = import(b"this is not a font at all").expect_err("it is not one");

		assert!(
			error.to_string().contains("font"),
			"the message should say what kind of file it wanted: {error}"
		);
	}

	#[test]
	fn a_baked_font_has_metrics_a_line_can_be_laid_out_with() {
		let bytes = face();

		let font = import(bytes).expect("the font beside this crate bakes");

		assert!(font.ascent > 0.0, "letters reach above the baseline");
		assert!(font.descent > 0.0, "and descenders below it");
		assert!(
			font.line_height >= font.ascent + font.descent,
			"and a line is at least as tall as the two together: {} against {}",
			font.line_height,
			font.ascent + font.descent
		);
	}

	#[test]
	fn every_glyph_lands_inside_the_atlas_it_was_packed_into() {
		let bytes = face();

		let font = import(bytes).expect("the font beside this crate bakes");

		for glyph in &font.glyphs {
			let right = u32::from(glyph.atlas_x) + u32::from(glyph.atlas_width);
			let bottom = u32::from(glyph.atlas_y) + u32::from(glyph.atlas_height);

			assert!(
				right <= font.atlas_width && bottom <= font.atlas_height,
				"glyph {:#06X} runs to {right}x{bottom} in an atlas of {}x{}",
				glyph.codepoint,
				font.atlas_width,
				font.atlas_height
			);
		}

		assert_eq!(
			font.atlas.len(),
			count_of(font.atlas_width * font.atlas_height),
			"and the atlas is exactly as big as it says it is"
		);
	}

	#[test]
	fn glyphs_come_out_in_codepoint_order_so_a_lookup_is_a_search() {
		let bytes = face();

		let font = import(bytes).expect("the font beside this crate bakes");

		assert!(
			font.glyphs
				.windows(2)
				.all(|pair| pair[0].codepoint < pair[1].codepoint),
			"FontData::glyph binary searches this, so an unsorted table finds nothing"
		);
	}

	#[test]
	fn the_letters_a_person_types_are_all_there() {
		let bytes = face();

		let font = import(bytes).expect("the font beside this crate bakes");

		for character in "Hg0 ,.:xY".chars() {
			assert!(
				font.glyph(u32::from(character)).is_some(),
				"the font was baked without {character:?}"
			);
		}

		assert!(
			font.glyph(u32::from('Ж')).is_some(),
			"and without Cyrillic, which is half the text anyone here will type"
		);
	}

	#[test]
	fn a_space_has_an_advance_and_no_picture() {
		let bytes = face();

		let font = import(bytes).expect("the font beside this crate bakes");
		let space = font
			.glyph(u32::from(' '))
			.expect("every font has a space");

		assert!(space.advance > 0.0, "a space moves the pen");
		assert!(!space.is_drawn(), "and puts nothing in the atlas");
	}

	#[test]
	fn a_letter_is_solid_in_the_middle_and_empty_outside_it() {
		let bytes = face();

		let font = import(bytes).expect("the font beside this crate bakes");
		let glyph = font
			.glyph(u32::from('H'))
			.expect("every font has an H");

		let at = |x: u32, y: u32| {
			let index = count_of((u32::from(glyph.atlas_y) + y) * font.atlas_width)
				+ count_of(u32::from(glyph.atlas_x) + x);

			font.atlas.get(index).copied().unwrap_or(0)
		};

		// the very corner of the cell is the padding the spread reserved, and
		// is as far outside the letter as the field goes.
		assert_eq!(at(0, 0), 0, "the corner of the cell is outside the letter");

		// the middle of an H is its crossbar.
		let middle = at(u32::from(glyph.atlas_width) / 2, u32::from(glyph.atlas_height) / 2);
		assert!(
			middle > 128,
			"the middle of an H is inside it, so its distance is positive: got {middle}"
		);
	}

	#[test]
	fn the_baked_size_is_what_the_metrics_are_in() {
		let bytes = face();

		let font = import(bytes).expect("the font beside this crate bakes");

		assert!(
			(font.pixel_size - PIXEL_SIZE).abs() < f32::EPSILON,
			"a run at another size divides by this, so it has to be the size it was baked at"
		);
		assert!(
			(font.scale(PIXEL_SIZE * 2.0) - 2.0).abs() < 1.0e-5,
			"and twice the size is twice the scale"
		);
	}
}
