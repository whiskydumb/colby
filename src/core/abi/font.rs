//! Fonts: baked glyph metrics, a distance-field atlas, and the host's registry
//! of them.
//!
//! Here rather than in the engine for the same reason meshes and textures are:
//! [`FontId`] crosses the boundary, and a handle means nothing away from the
//! table it indexes. @ref [`registry`](super::registry) for the shape all four
//! tables share.
//!
//! **Nothing in this module reads a `.ttf`.** A font arrives already
//! rasterized: the asset compiler turns an outline font into a signed distance
//! field and a table of metrics, and what reaches here is the result. That is
//! the same bargain textures made - the decoder lands offline, and neither the
//! engine nor the runner links a font library.
//!
//! The other half of why this is in `colby_core` rather than in the interface
//! crate: **measuring text is an input to layout, not an output of it.** A text
//! node's width and height *are* the measured text, so the measurement has to
//! be reachable from wherever the layout runs - and it has to be the same
//! measurement the drawing uses, or the words land outside the box that was
//! reserved for them. [`FontData::run`] is that one implementation: the
//! measurement is what you get when nothing is drawn.

use super::registry::{Entry, Registry};
use crate::{
	bytemuck::{Pod, Zeroable},
	glam::Vec2,
	registry_handle,
};

/// The name the always-present empty font is registered under.
pub const EMPTY_NAME: &str = "";

/// What one character costs and where its picture is.
///
/// `#[repr(C)]` and [`Pod`] because this is also the on-disk record: a `.cfont`
/// holds an array of exactly these, and reading one is a cast rather than a
/// decode. @ref `colby_asset::font`.
///
/// Every measurement is in pixels at [`FontData::pixel_size`], so a run at
/// another size multiplies by [`FontData::scale`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Glyph {
	/// The character this draws.
	pub codepoint: u32,

	/// How far the pen moves after drawing it.
	pub advance: f32,

	/// How far right of the pen the picture starts.
	pub bearing_x: f32,

	/// How far above the baseline the picture's top edge sits.
	///
	/// Positive up, so the top-left corner of the quad is
	/// `(pen + bearing_x, baseline - bearing_y)` once both are scaled.
	pub bearing_y: f32,

	/// Where in the atlas the picture starts, in texels.
	pub atlas_x: u16,

	/// The same, vertically.
	pub atlas_y: u16,

	/// How wide the picture is in the atlas, in texels.
	///
	/// Zero for a character that has no picture at all - a space, or one the
	/// font does not draw. Such a glyph still has an
	/// [`advance`](Self::advance).
	pub atlas_width: u16,

	/// How tall it is.
	pub atlas_height: u16,
}

impl Glyph {
	/// Whether there is anything in the atlas to draw.
	#[must_use]
	pub const fn is_drawn(&self) -> bool { self.atlas_width > 0 && self.atlas_height > 0 }
}

registry_handle! {
	/// Which font in [`Fonts`].
	///
	/// Not generational, like every other resource handle here: a font resolved
	/// by name in `init` stays valid for the life of the process, and
	/// recompiling the source rewrites the entry the handle already points at.
	/// @ref [`registry`](super::registry).
	FontId
}

/// One entry of the font registry.
pub type Font = Entry<FontData>;

/// A baked font: the metrics, and one channel of distance field.
///
/// The atlas is a single byte per texel. `128` is exactly on the outline, below
/// is outside and above is inside; [`spread`](Self::spread) says how many
/// pixels the whole `0 ..= 255` range covers, which is what lets a shader turn
/// the byte back into a distance and antialias against it at any size.
#[derive(Clone, Debug, PartialEq)]
pub struct FontData {
	/// The em size the glyphs were rasterized at, in pixels.
	pub pixel_size: f32,

	/// How far above the baseline the tallest letters reach, in pixels.
	pub ascent: f32,

	/// How far below it the descenders drop. Positive.
	pub descent: f32,

	/// Baseline to baseline, in pixels.
	pub line_height: f32,

	/// How many pixels the full range of a distance byte covers.
	pub spread: f32,

	/// The atlas width, in texels.
	pub atlas_width: u32,

	/// The atlas height, in texels.
	pub atlas_height: u32,

	/// One byte per texel, row major, top row first.
	pub atlas: Vec<u8>,

	/// Every glyph, **sorted by codepoint** so a lookup is a binary search.
	pub glyphs: Vec<Glyph>,
}

/// The empty font as a constant, so [`Fonts::data`] has something to borrow.
///
/// Unreachable in practice - slot zero of the table is always there - but
/// returning a reference means having one, and a static is cheaper than making
/// every caller handle a case that cannot happen.
static EMPTY: FontData = FontData {
	pixel_size: 1.0,
	ascent: 0.8,
	descent: 0.2,
	line_height: 1.2,
	spread: 1.0,
	atlas_width: 0,
	atlas_height: 0,
	atlas: Vec::new(),
	glyphs: Vec::new(),
};

impl FontData {
	/// What the character a font has no glyph for is drawn as.
	///
	/// Nothing, and the pen still moves - a missing character should leave a
	/// hole the width of a space rather than pull the rest of the line left,
	/// which is much harder to notice.
	pub const MISSING_ADVANCE: f32 = 0.5;

	/// A font with no glyphs, for the registry's null slot.
	///
	/// Its metrics are a plausible line rather than zeroes: text in a font that
	/// failed to load should leave a gap of about the right size, so that the
	/// rest of the interface stays where it belongs and the missing words are
	/// the only thing wrong with the picture.
	#[must_use]
	pub fn empty() -> Self { EMPTY.clone() }

	/// Whether this font can draw anything at all.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.glyphs.is_empty() }

	/// How much to multiply a baked measurement by to draw at `size`.
	///
	/// @param size - the em size wanted, in pixels
	#[must_use]
	pub fn scale(&self, size: f32) -> f32 {
		if self.pixel_size > 0.0 {
			size / self.pixel_size
		} else {
			0.0
		}
	}

	/// Baseline to baseline at `size`.
	#[must_use]
	pub fn line_height_at(&self, size: f32) -> f32 { self.line_height * self.scale(size) }

	/// How far below the top of a line its baseline sits, at `size`.
	#[must_use]
	pub fn ascent_at(&self, size: f32) -> f32 { self.ascent * self.scale(size) }

	/// One glyph, by the character it draws.
	///
	/// @param codepoint - the character
	/// @return its glyph, or `None` when the font was not baked with it
	#[must_use]
	pub fn glyph(&self, codepoint: u32) -> Option<&Glyph> {
		self.glyphs
			.binary_search_by_key(&codepoint, |glyph| glyph.codepoint)
			.ok()
			.and_then(|index| self.glyphs.get(index))
	}

	/// How wide a run of text is, and how tall.
	///
	/// The measurement half of [`run`](Self::run), which is the same code with
	/// nothing drawn. @ref [`run`](Self::run) for what `wrap` does.
	///
	/// @param text - the string to measure
	/// @param size - the em size to measure at, in pixels
	/// @param wrap - the width to break lines at, or `None` for one long line
	/// @return the width of the longest line and the height of all of them
	#[must_use]
	pub fn measure(&self, text: &str, size: f32, wrap: Option<f32>) -> Vec2 {
		self.run(text, size, wrap, |_, _, _| {})
	}

	/// Walks a run of text, handing out every glyph and where it goes.
	///
	/// The one place text is laid out. Measuring calls this and ignores the
	/// glyphs; drawing calls it and keeps them. Two implementations of this
	/// would be two implementations that disagree, and the disagreement would
	/// look like a word sticking out of a box that was measured for it.
	///
	/// Positions are relative to the top-left of the run, and each is the
	/// glyph's own top-left corner - already scaled, already offset by its
	/// bearings. Glyphs with nothing to draw are not handed out.
	///
	/// Line breaking is greedy and breaks on spaces: a word longer than the
	/// wrap width overhangs rather than being cut in half. `\n` always breaks,
	/// wrap or no wrap, and `\r` is dropped so a file with Windows line endings
	/// does not draw a box every line.
	///
	/// @param text - the string to lay out
	/// @param size - the em size, in pixels
	/// @param wrap - the width to break lines at, or `None` for one long line
	/// @param sink - called with each glyph, its x and its y
	/// @return the width of the longest line and the height of all of them
	pub fn run<F>(&self, text: &str, size: f32, wrap: Option<f32>, mut sink: F) -> Vec2
	where
		F: FnMut(&Glyph, f32, f32),
	{
		let scale = self.scale(size);
		let line_height = self.line_height * scale;
		let ascent = self.ascent * scale;

		let mut widest = 0.0_f32;
		let mut lines = 0_u32;
		let mut pen = 0.0_f32;
		let mut line = 0.0_f32;

		// how wide the current line is up to its last drawn glyph. The pen is
		// not that number: it also holds whatever spaces have been stepped over
		// since, and a line measured with its trailing spaces in it is wider
		// than the words on it - which at a wrap point is the difference
		// between a run that fits its box and one that reports itself too wide
		// for the box it was just broken into.
		let mut ink = 0.0_f32;

		// where the current word started, so a break can rewind to it. Both are
		// `None` between words, which is also what makes a run of spaces at a
		// wrap point collapse instead of pushing the next word off the edge.
		let mut word_start: Option<usize> = None;
		let mut word_pen = 0.0_f32;

		let mut index = 0;
		let characters: Vec<char> = text.chars().collect();

		while let Some(&character) = characters.get(index) {
			index += 1;

			if character == '\r' {
				continue;
			}

			if character == '\n' {
				widest = widest.max(ink);
				lines += 1;
				line += line_height;
				pen = 0.0;
				ink = 0.0;
				word_start = None;

				continue;
			}

			let advance = self.advance_of(character, scale);

			if character == ' ' || character == '\t' {
				pen += advance;
				word_start = None;

				continue;
			}

			if word_start.is_none() {
				word_start = Some(index - 1);
				word_pen = pen;
			}

			// the break decision is taken before the glyph is placed, and it
			// rewinds to the start of the word rather than splitting it. A word
			// that is wider than the whole line has nowhere to go, so it is left
			// where it is: overhanging is ugly, and a word cut in half is worse.
			if let Some(limit) = wrap
				&& pen + advance > limit
				&& let Some(start) = word_start
				&& word_pen > 0.0
			{
				widest = widest.max(ink);
				lines += 1;
				line += line_height;
				pen = 0.0;
				ink = 0.0;
				index = start;
				word_start = None;

				continue;
			}

			if let Some(glyph) = self.glyph(u32::from(character))
				&& glyph.is_drawn()
			{
				sink(
					glyph,
					glyph.bearing_x.mul_add(scale, pen),
					glyph.bearing_y.mul_add(-scale, line + ascent),
				);
			}

			pen += advance;
			ink = pen;
		}

		widest = widest.max(ink);
		lines += 1;

		Vec2::new(widest, line_height * lines_as_f32(lines))
	}

	/// How far the pen moves for one character, at `scale`.
	fn advance_of(&self, character: char, scale: f32) -> f32 {
		self.glyph(u32::from(character))
			.map_or(self.pixel_size * Self::MISSING_ADVANCE, |glyph| glyph.advance)
			* scale
	}
}

impl Default for FontData {
	fn default() -> Self { Self::empty() }
}

/// Every font the interface can draw with, addressed by [`FontId`].
///
/// Slot zero is [`FontId::NONE`] and is [`FontData::empty`], so a document that
/// names a font nobody compiled still reserves the right amount of room for its
/// words instead of collapsing.
#[derive(Clone, Debug)]
pub struct Fonts {
	entries: Registry<FontData>,
}

impl Fonts {
	/// A registry holding nothing but the empty font.
	#[must_use]
	pub fn new() -> Self {
		Self {
			entries: Registry::new(FontData::empty()),
		}
	}

	/// Looks a font up by name.
	///
	/// @return its handle, or [`FontId::NONE`] if nothing answers to it
	#[must_use]
	pub fn find(&self, name: &str) -> FontId { FontId::new(self.entries.find(name)) }

	/// Registers a baked font under a name, replacing whatever was there.
	///
	/// @return the handle, the same one as last time if the name is known
	pub fn insert(&mut self, name: &str, data: FontData) -> FontId {
		FontId::new(self.entries.insert(name, data))
	}

	/// One font, by handle.
	#[must_use]
	pub fn get(&self, id: FontId) -> Option<&Font> { self.entries.entry(id.index()) }

	/// One font's data, by handle, falling back to the empty one.
	///
	/// What layout and drawing both call: neither has anything useful to do
	/// about a handle that does not resolve, and both would rather measure a
	/// hole than branch.
	#[must_use]
	pub fn data(&self, id: FontId) -> &FontData {
		self.entries
			.entry(id.index())
			.or_else(|| self.entries.entry(0))
			.map_or(&EMPTY, Entry::value)
	}

	/// How many fonts there are, counting the null one.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Always `false`: slot zero always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// Every font, in slot order.
	pub fn iter(&self) -> impl Iterator<Item = &Font> { self.entries.iter() }
}

impl Default for Fonts {
	fn default() -> Self { Self::new() }
}

/// Lines counted as a number rather than cast.
///
/// A run with four billion lines in it is not a thing that happens, and the
/// cast lint is not worth an `expect` here.
fn lines_as_f32(lines: u32) -> f32 {
	u16::try_from(lines).map_or_else(|_| f32::from(u16::MAX), f32::from)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A font whose every glyph is one pixel wide and advances by ten.
	///
	/// Built by hand rather than baked, so the arithmetic under test is the
	/// only thing that can be wrong.
	fn ten_wide(characters: &str) -> FontData {
		let mut glyphs: Vec<Glyph> = characters
			.chars()
			.map(|character| Glyph {
				codepoint: u32::from(character),
				advance: 10.0,
				bearing_x: 1.0,
				bearing_y: 8.0,
				atlas_x: 0,
				atlas_y: 0,
				atlas_width: if character == ' ' { 0 } else { 8 },
				atlas_height: if character == ' ' { 0 } else { 8 },
			})
			.collect();

		glyphs.sort_by_key(|glyph| glyph.codepoint);

		FontData {
			pixel_size: 10.0,
			ascent: 8.0,
			descent: 2.0,
			line_height: 12.0,
			spread: 4.0,
			atlas_width: 16,
			atlas_height: 16,
			atlas: vec![0; 256],
			glyphs,
		}
	}

	/// Whether a run measures the width it should, within a rounding error.
	fn wide(font: &FontData, text: &str, size: f32, expected: f32) -> bool {
		(font.measure(text, size, None).x - expected).abs() < 1.0e-5
	}

	#[test]
	fn an_empty_font_still_takes_up_a_line() {
		let font = FontData::empty();

		assert!(font.is_empty(), "there is nothing in it to draw");
		assert!(
			font.measure("anything", 16.0, None).y > 0.0,
			"text in a font that failed to load should leave a gap of about the right size, not \
			 collapse the box it is in"
		);
		assert!(font.measure("", 16.0, None).x < 1.0e-5, "and nothing is nothing wide");
	}

	#[test]
	fn a_glyph_is_found_by_the_character_it_draws() {
		let font = ten_wide("abc");

		assert!(
			font.glyph(u32::from('b'))
				.is_some_and(|glyph| (glyph.advance - 10.0).abs() < 1.0e-5),
			"it is there"
		);
		assert!(font.glyph(u32::from('z')).is_none(), "and one that was never baked is not");
	}

	#[test]
	fn a_run_is_as_wide_as_its_advances() {
		let font = ten_wide("abc");

		assert!(
			wide(&font, "abc", 10.0, 30.0),
			"three glyphs of ten at the size they were baked at"
		);
		assert!(
			wide(&font, "abc", 20.0, 60.0),
			"and twice that at twice the size, because a run scales rather than re-measuring"
		);
	}

	#[test]
	fn a_character_the_font_never_baked_still_moves_the_pen() {
		let font = ten_wide("abc");

		assert!(
			wide(&font, "azc", 10.0, 25.0),
			"two glyphs of ten and half a one for the character with no picture: a missing \
			 letter should leave a hole rather than pull the rest of the line left"
		);
	}

	#[test]
	fn a_newline_starts_another_line_whether_or_not_anything_wraps() {
		let font = ten_wide("abc");
		let size = font.measure("ab\nc", 10.0, None);

		assert!(
			size.abs_diff_eq(Vec2::new(20.0, 24.0), 1.0e-5),
			"the longest line is the two-glyph one and there are two of twelve: got {size}"
		);
	}

	#[test]
	fn a_carriage_return_draws_nothing_and_measures_nothing() {
		let font = ten_wide("abc");

		assert_eq!(
			font.measure("ab\r\nc", 10.0, None),
			font.measure("ab\nc", 10.0, None),
			"a file saved with Windows line endings must not draw a box every line"
		);
	}

	#[test]
	fn wrapping_breaks_between_words_rather_than_inside_one() {
		let font = ten_wide("abc ");
		// three words of two glyphs, twenty wide each, in a box wide enough for
		// two of them and the space between.
		let size = font.measure("ab ab ab", 10.0, Some(50.0));

		assert!((size.y - 24.0).abs() < 1.0e-5, "so it takes two lines: got {}", size.y);
		assert!(size.x <= 50.0, "and neither of them is wider than the box: got {}", size.x);
	}

	#[test]
	fn a_word_wider_than_the_box_overhangs_rather_than_being_cut_up() {
		let font = ten_wide("abc");
		let size = font.measure("abcabc", 10.0, Some(20.0));

		assert!(
			size.abs_diff_eq(Vec2::new(60.0, 12.0), 1.0e-5),
			"one line, because there is nowhere to break it, and it sticks out - which is at 			 least legible: got {size}"
		);
	}

	#[test]
	fn drawing_and_measuring_agree_because_they_are_one_function() {
		let font = ten_wide("abc ");
		let mut right = 0.0_f32;
		let mut bottom = 0.0_f32;

		let measured = font.run("ab ab ab", 10.0, Some(50.0), |glyph, x, y| {
			right = right.max(x + f32::from(glyph.atlas_width));
			bottom = bottom.max(y + f32::from(glyph.atlas_height));
		});

		assert!(
			right <= measured.x + 1.0,
			"no glyph may be drawn past the width that was measured for it: {right} against {}",
			measured.x
		);
		assert!(bottom <= measured.y, "nor below the height: {bottom} against {}", measured.y);
	}

	#[test]
	fn a_glyph_with_no_picture_is_never_handed_out_to_be_drawn() {
		let font = ten_wide("abc ");
		let mut drawn = 0_u32;

		font.run("a b", 10.0, None, |_, _, _| drawn += 1);

		assert_eq!(drawn, 2, "the space has an advance and nothing in the atlas");
	}

	#[test]
	fn the_null_slot_of_the_table_is_a_font_that_draws_nothing() {
		let fonts = Fonts::new();

		assert_eq!(fonts.find("nothing"), FontId::NONE, "no name resolves in an empty table");
		assert!(
			fonts.data(FontId::NONE).is_empty(),
			"and slot zero answers with a font rather than with nothing at all"
		);
	}

	#[test]
	fn a_registered_font_keeps_its_slot_when_it_is_recompiled() {
		let mut fonts = Fonts::new();
		let first = fonts.insert("fonts/hack", ten_wide("abc"));
		let again = fonts.insert("fonts/hack", ten_wide("abcd"));

		assert_eq!(first, again, "the same name is the same handle, so a game never re-resolves");
		assert_eq!(fonts.len(), 2, "and nothing was appended the second time");
		assert_eq!(
			fonts.get(first).map(Entry::revision),
			Some(1),
			"which is how the interface finds out its atlas has to be uploaded again"
		);
	}
}
