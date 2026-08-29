//! Turning laid-out boxes into triangles.
//!
//! Nothing here touches a GPU, which is the point: the list this builds is
//! ordinary data, so what a document draws can be checked by a test rather than
//! only by looking at a picture. @ref [`paint`](crate::paint) for the half that
//! does touch one.
//!
//! Every quad carries its own shape, so the whole interface is one draw call
//! per bound texture. A rectangle can join whatever batch is open - it ignores
//! the texture entirely - and only a glyph run in another font or an image
//! starts a new one.

use colby_core::{
	abi::{FontData, FontId, TextureId, World, ui::DocumentData},
	bytemuck::{Pod, Zeroable},
	glam::Vec4,
};

use crate::layout::{Placed, background, foreground};

/// One corner of one quad.
///
/// Fat, at seventy-six bytes. An interface is thousands of these rather than
/// millions, and the alternative - a uniform per box, or a pipeline per kind -
/// costs a draw call per box instead of a few floats per corner. The clip
/// rectangle is here for the same reason and is the case that makes the trade
/// pay: cutting with a scissor rectangle instead would split a document into
/// one draw call per clipping box on top of the one per texture, and it could
/// not follow a corner radius at all.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Vertex {
	/// Where it is, in layout pixels with the origin at the top left.
	pub position: [f32; 2],

	/// Where it is relative to the middle of its own box.
	pub local: [f32; 2],

	/// Half the box's size.
	pub half_size: [f32; 2],

	/// Where to sample, in `0.0 ..= 1.0`.
	pub uv: [f32; 2],

	/// What to paint it, linear with straight alpha.
	pub color: [f32; 4],

	/// The corner radius, or for a glyph how many layout pixels the whole
	/// range of a distance byte covers.
	pub radius: f32,

	/// Which of the three kinds this is. @ref `ui.wgsl`.
	pub kind: f32,

	/// What is left of it after every clipping ancestor: left, top, right,
	/// bottom, in layout pixels. @ref
	/// [`UNCLIPPED`](crate::layout::UNCLIPPED).
	pub clip: [f32; 4],

	/// How round the corners of that rectangle are.
	pub clip_radius: f32,
}

/// How wide the caret is, in layout pixels.
///
/// A constant rather than a share of the font size: a caret is a thing the eye
/// finds rather than a letter, and one that got thicker with the text would be
/// a smudge at a heading's size.
const CARET_WIDTH: f32 = 1.5;

/// How wide the bar down the side of a scrolling box is, in layout pixels.
const BAR_WIDTH: f32 = 5.0;

/// How far in from the edge it sits.
const BAR_INSET: f32 = 2.0;

/// The shortest it is ever drawn, so that a very long list still has something
/// to look at rather than a dot.
const BAR_MIN: f32 = 18.0;

/// How much of the box's own text color it is painted in.
///
/// The text color rather than a property of its own, which is one fewer thing
/// for a stylesheet to have to say and reads correctly on both a light panel
/// and a dark one. A `scrollbar-color` is what replaces this the day somebody
/// wants two different ones.
const BAR_ALPHA: f32 = 0.45;

/// A rounded rectangle, painted flat.
pub const KIND_RECT: f32 = 0.0;

/// A glyph, read out of a font's distance field.
pub const KIND_GLYPH: f32 = 1.0;

/// An image, read out of a texture.
pub const KIND_IMAGE: f32 = 2.0;

/// What a batch samples from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Binding {
	/// Nothing in particular. Rectangles do not sample.
	#[default]
	Blank,

	/// A font's atlas.
	Font(FontId),

	/// A texture from the world's table.
	Image(TextureId),
}

/// A run of triangles that share a texture.
#[derive(Clone, Copy, Debug)]
pub struct Batch {
	/// What it samples.
	pub binding: Binding,

	/// The first index of the run.
	pub first: u32,

	/// How many indices are in it.
	pub count: u32,
}

/// Everything the interface draws this frame.
#[derive(Clone, Debug, Default)]
pub struct DrawList {
	/// Every corner.
	pub vertices: Vec<Vertex>,

	/// Three per triangle.
	pub indices: Vec<u32>,

	/// One per run of triangles that share a texture.
	pub batches: Vec<Batch>,
}

impl DrawList {
	/// Whether there is nothing to draw.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.indices.is_empty() }

	/// Empties it, keeping the memory.
	pub fn clear(&mut self) {
		self.vertices.clear();
		self.indices.clear();
		self.batches.clear();
	}

	/// Adds one quad to the open batch.
	fn quad(
		&mut self,
		rect: [f32; 4],
		uv: [f32; 4],
		color: Vec4,
		radius: f32,
		kind: f32,
		clip: ([f32; 4], f32),
	) {
		let (x, y, width, height) = (rect[0], rect[1], rect[2], rect[3]);
		if width <= 0.0 || height <= 0.0 || color.w <= 0.0 {
			return;
		}

		let half = [width / 2.0, height / 2.0];
		let base = u32::try_from(self.vertices.len()).unwrap_or(0);
		let corners = [
			([x, y], [-half[0], -half[1]], [uv[0], uv[1]]),
			([x + width, y], [half[0], -half[1]], [uv[2], uv[1]]),
			([x + width, y + height], [half[0], half[1]], [uv[2], uv[3]]),
			([x, y + height], [-half[0], half[1]], [uv[0], uv[3]]),
		];

		for (position, local, at) in corners {
			self.vertices.push(Vertex {
				position,
				local,
				half_size: half,
				uv: at,
				color: color.to_array(),
				radius,
				kind,
				clip: clip.0,
				clip_radius: clip.1,
			});
		}

		self.indices
			.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
	}

	/// Makes sure the open batch samples what the next quad wants.
	///
	/// A rectangle asks for [`Binding::Blank`], which any batch satisfies: it
	/// does not sample, so it can go in with whatever is already open.
	fn want(&mut self, binding: Binding) {
		let open = self.batches.last().map(|batch| batch.binding);

		match open {
			| Some(current) if current == binding || binding == Binding::Blank => {},
			// a batch that has only had rectangles in it so far has not sampled
			// anything, so it can be told what to sample instead of being closed
			// and followed by a second one. Without this, every box with a
			// background costs a draw call ahead of the words inside it.
			| Some(Binding::Blank) =>
				if let Some(batch) = self.batches.last_mut() {
					batch.binding = binding;
				},
			| _ => self.batches.push(Batch {
				binding,
				first: u32::try_from(self.indices.len()).unwrap_or(0),
				count: 0,
			}),
		}
	}

	/// Closes the open batch over everything written since it opened.
	fn close(&mut self) {
		let end = u32::try_from(self.indices.len()).unwrap_or(0);

		if let Some(batch) = self.batches.last_mut() {
			batch.count = end.saturating_sub(batch.first);
		}
	}
}

/// Builds the triangles for a laid-out interface.
///
/// @param world - the documents, fonts and textures the boxes refer to
/// @param placed - what [`layout::run`](crate::layout::run) produced
/// @param list - cleared and filled
pub fn build(world: &World, placed: &[Placed], list: &mut DrawList) {
	list.clear();

	for box_of in placed {
		let Some(document) = world
			.ui
			.panel(box_of.panel)
			.and_then(|panel| world.ui.document(panel.document()))
			.map(colby_core::abi::registry::Entry::value)
		else {
			continue;
		};

		let Some(node) = document.node(box_of.node) else {
			continue;
		};

		list.want(Binding::Blank);
		let radius = DocumentData::radius(&box_of.style, box_of.rect[2].min(box_of.rect[3]));
		list.quad(
			box_of.rect,
			[0.0, 0.0, 1.0, 1.0],
			background(box_of).0,
			radius,
			KIND_RECT,
			clip_of(box_of),
		);
		list.close();

		if node.is_image() && !box_of.image.is_empty() {
			image(world, box_of, radius, list);
		}

		// a field draws its own words, so nothing has positioned them inside its
		// padding the way the layout positions a child. Every other box's words
		// are a child and are already where they belong.
		let inset = if node.is_input() {
			padding_of(box_of)
		} else {
			[0.0, 0.0]
		};

		if node.has_words() && !box_of.text.is_empty() {
			text(world, box_of, inset, list);
		}

		if box_of.focused {
			caret(world, box_of, inset, list);
		}

		if box_of.scrollable > 0.0 {
			bar(box_of, list);
		}
	}

	list.close();
}

/// The left and top padding of a box, in layout pixels.
///
/// A percentage resolves against the box's own width, which is what CSS says
/// for padding on both axes.
fn padding_of(box_of: &Placed) -> [f32; 2] {
	let resolve = |length: Option<colby_core::abi::ui::Length>| {
		length
			.and_then(|it| it.resolve(box_of.rect[2]))
			.unwrap_or(0.0)
	};

	[resolve(box_of.style.padding.left), resolve(box_of.style.padding.top)]
}

/// Adds the upright line that says where typing will land.
///
/// Measured through the same function that laid the words out, over the part of
/// the value before the caret - which is why the caret is a byte offset rather
/// than a character count: it is an index into the string, and every other
/// answer would need converting before it could be used.
fn caret(world: &World, box_of: &Placed, inset: [f32; 2], list: &mut DrawList) {
	if !box_of.font.is_some() {
		return;
	}

	let font = world.fonts.data(box_of.font);
	let at = usize::try_from(box_of.caret)
		.unwrap_or(0)
		.min(box_of.text.len());
	let Some(before) = box_of.text.get(..at) else {
		// the offset landed inside a character, which nothing here should be
		// able to produce. Drawing no caret is better than drawing a wrong one.
		return;
	};

	let along = font.measure(before, box_of.font_size, None).x;

	list.want(Binding::Blank);
	list.quad(
		[
			box_of.rect[0] + inset[0] + along,
			box_of.rect[1] + inset[1],
			CARET_WIDTH,
			box_of.font_size.max(CARET_WIDTH),
		],
		[0.0, 0.0, 1.0, 1.0],
		foreground(box_of).0,
		0.0,
		KIND_RECT,
		clip_of(box_of),
	);
	list.close();
}

/// Adds the bar down the side of a box that can be scrolled.
///
/// Drawn over the contents rather than beside them, which is why the layout
/// reserves no room for it: a box that got narrower the moment something
/// overflowed it would reflow its own contents and could oscillate.
///
/// Its length is the share of the contents that is on screen and its position
/// is how far down that share is, which is the whole of what a bar says.
fn bar(box_of: &Placed, list: &mut DrawList) {
	let height = box_of.rect[3];
	let content = height + box_of.scrollable;

	if content <= 0.0 || height <= 0.0 {
		return;
	}

	let thumb = (height * height / content)
		.max(BAR_MIN)
		.min(height);
	let travel = (height - thumb).max(0.0);
	let along = travel * (box_of.scroll / box_of.scrollable).clamp(0.0, 1.0);

	let mut color = foreground(box_of).0;
	color.w *= BAR_ALPHA;

	list.want(Binding::Blank);
	list.quad(
		[
			box_of.rect[0] + box_of.rect[2] - BAR_WIDTH - BAR_INSET,
			box_of.rect[1] + along,
			BAR_WIDTH,
			thumb,
		],
		[0.0, 0.0, 1.0, 1.0],
		color,
		BAR_WIDTH / 2.0,
		KIND_RECT,
		clip_of(box_of),
	);
	list.close();
}

/// Adds an image over its whole box.
fn image(world: &World, box_of: &Placed, radius: f32, list: &mut DrawList) {
	let id = world.textures.find(&box_of.image);
	if !id.is_some() {
		return;
	}

	list.want(Binding::Image(id));
	list.quad(
		box_of.rect,
		[0.0, 0.0, 1.0, 1.0],
		foreground(box_of).0,
		radius,
		KIND_IMAGE,
		clip_of(box_of),
	);
	list.close();
}

/// Adds the glyphs of one box's run of text.
fn text(world: &World, box_of: &Placed, inset: [f32; 2], list: &mut DrawList) {
	if !box_of.font.is_some() {
		return;
	}

	// wrapped at a hair over the box, not exactly at it: the box was sized by
	// this same measurement and then rounded to whole pixels, and breaking a
	// line half a pixel earlier than it was measured to break would put the
	// last word somewhere the layout did not reserve room for.
	let wrap = Some(box_of.rect[2] + 1.0);

	glyphs(list, &box_of.text, &Run {
		font: world.fonts.data(box_of.font),
		id: box_of.font,
		size: box_of.font_size,
		origin: [box_of.rect[0] + inset[0], box_of.rect[1] + inset[1]],
		wrap,
		color: foreground(box_of).0,
		clip: clip_of(box_of),
	});
}

/// The clip rectangle a box is drawn under, as the pair a quad wants.
fn clip_of(box_of: &Placed) -> ([f32; 4], f32) { (box_of.clip, box_of.clip_radius) }

/// Where a run of text goes and what it is drawn with.
///
/// A struct rather than six arguments, and it is the same six every caller has:
/// what to lay out against, where to put it and what color to make it.
pub(crate) struct Run<'a> {
	/// The metrics and the atlas to lay out against.
	pub(crate) font: &'a FontData,

	/// The handle the batch samples.
	pub(crate) id: FontId,

	/// The size to draw at, in layout pixels.
	pub(crate) size: f32,

	/// The top left of the run, in layout pixels.
	pub(crate) origin: [f32; 2],

	/// How wide to let a line get before breaking it.
	pub(crate) wrap: Option<f32>,

	/// Linear with straight alpha.
	pub(crate) color: Vec4,

	/// What is left of the run after clipping, and how round that is. @ref
	/// [`UNCLIPPED`](crate::layout::UNCLIPPED) for a run that is not clipped at
	/// all, which is what a label anchored in the world uses.
	pub(crate) clip: ([f32; 4], f32),
}

/// Adds one glyph quad per drawn character of a run of text.
///
/// Split out from [`text`] because a document is not the only thing with words
/// in it: a label anchored in the world is the same glyphs at a point somebody
/// else worked out. @ref [`world_text`](crate::world_text).
///
/// @param list - what to append to; a batch is opened and closed around the run
/// @param text - the words
/// @param run - where they go and what they are drawn with
pub(crate) fn glyphs(list: &mut DrawList, text: &str, run: &Run<'_>) {
	let font = run.font;
	if font.atlas_width == 0 || font.atlas_height == 0 {
		return;
	}

	let scale = font.scale(run.size);
	let atlas = (texels(font.atlas_width), texels(font.atlas_height));
	// how many layout pixels the whole `0 ..= 255` range of a distance byte
	// covers at this size: the shader turns a sample back into a distance with
	// it, and antialiases against that.
	let range = font.spread * 2.0 * scale;

	list.want(Binding::Font(run.id));

	font.run(text, run.size, run.wrap, |glyph, x, y| {
		let width = f32::from(glyph.atlas_width) * scale;
		let height = f32::from(glyph.atlas_height) * scale;

		list.quad(
			[run.origin[0] + x, run.origin[1] + y, width, height],
			[
				f32::from(glyph.atlas_x) / atlas.0,
				f32::from(glyph.atlas_y) / atlas.1,
				f32::from(glyph.atlas_x + glyph.atlas_width) / atlas.0,
				f32::from(glyph.atlas_y + glyph.atlas_height) / atlas.1,
			],
			run.color,
			range,
			KIND_GLYPH,
			run.clip,
		);
	});

	list.close();
}

/// An atlas dimension as a number, never zero.
fn texels(value: u32) -> f32 { f32::from(u16::try_from(value).unwrap_or(u16::MAX)).max(1.0) }

#[cfg(test)]
mod tests {
	use colby_core::abi::{
		Glyph,
		ui::{Node, PanelId, document::ROOT, style::Color},
	};

	use super::*;

	/// A font whose glyphs are eight texels square and advance by ten.
	fn font() -> FontData {
		FontData {
			pixel_size: 10.0,
			ascent: 8.0,
			descent: 2.0,
			line_height: 12.0,
			spread: 4.0,
			atlas_width: 16,
			atlas_height: 16,
			atlas: vec![0; 256],
			glyphs: vec![Glyph {
				codepoint: u32::from('a'),
				advance: 10.0,
				bearing_x: 0.0,
				bearing_y: 8.0,
				atlas_x: 0,
				atlas_y: 0,
				atlas_width: 8,
				atlas_height: 8,
			}],
		}
	}

	/// A world with one panel showing a document of one text node.
	fn world() -> (World, PanelId) {
		let mut world = World::new();
		world.fonts.insert("fonts/test", font());

		let mut data = DocumentData::empty();
		data.nodes.push(Node {
			tag: Node::TEXT.to_owned(),
			text: "aa".to_owned(),
			parent: ROOT,
			first_child: colby_core::abi::ui::document::NONE,
			next_sibling: colby_core::abi::ui::document::NONE,
			..Node::default()
		});
		if let Some(root) = data.nodes.first_mut() {
			root.first_child = 1;
		}

		world.ui.insert("ui/one", data);
		let panel = world.ui.show("ui/one");

		(world, panel)
	}

	/// A box, placed by hand rather than laid out.
	fn placed(panel: PanelId, node: u32, rect: [f32; 4]) -> Placed {
		Placed {
			panel,
			node,
			rect,
			clip: crate::layout::UNCLIPPED,
			clip_radius: 0.0,
			scroll: 0.0,
			scrollable: 0.0,
			focused: false,
			caret: 0,
			style: colby_core::abi::ui::Style::root(),
			font: FontId::NONE,
			font_size: 10.0,
			opacity: 1.0,
			text: String::new(),
			image: String::new(),
		}
	}

	#[test]
	fn a_box_with_a_background_becomes_one_quad() {
		let (world, panel) = world();
		let mut item = placed(panel, ROOT, [0.0, 0.0, 100.0, 50.0]);
		item.style.background = Some(Color::WHITE);

		let mut list = DrawList::default();
		build(&world, &[item], &mut list);

		assert_eq!(list.vertices.len(), 4, "four corners");
		assert_eq!(list.indices.len(), 6, "two triangles");
		assert!(
			list.vertices
				.iter()
				.all(|vertex| (vertex.kind - KIND_RECT).abs() < f32::EPSILON),
			"and all of them a rectangle"
		);
	}

	#[test]
	fn a_box_painted_with_nothing_draws_nothing() {
		let (world, panel) = world();
		let item = placed(panel, ROOT, [0.0, 0.0, 100.0, 50.0]);

		let mut list = DrawList::default();
		build(&world, &[item], &mut list);

		assert!(list.is_empty(), "a transparent background is not a draw call");
	}

	#[test]
	fn a_box_with_no_size_draws_nothing() {
		let (world, panel) = world();
		let mut item = placed(panel, ROOT, [0.0, 0.0, 0.0, 50.0]);
		item.style.background = Some(Color::WHITE);

		let mut list = DrawList::default();
		build(&world, &[item], &mut list);

		assert!(list.is_empty(), "nothing zero pixels wide is on screen");
	}

	#[test]
	fn every_drawn_glyph_of_a_run_becomes_a_quad() {
		let (world, panel) = world();
		let mut item = placed(panel, 1, [10.0, 20.0, 100.0, 12.0]);
		item.font = world.fonts.find("fonts/test");
		item.text = "aa".to_owned();

		let mut list = DrawList::default();
		build(&world, &[item], &mut list);

		assert_eq!(list.vertices.len(), 8, "two glyphs of four corners");
		assert!(
			list.vertices
				.iter()
				.all(|vertex| (vertex.kind - KIND_GLYPH).abs() < f32::EPSILON),
			"and the shader is told they are glyphs"
		);
	}

	#[test]
	fn a_glyph_lands_where_the_box_it_belongs_to_is() {
		let (world, panel) = world();
		let mut item = placed(panel, 1, [10.0, 20.0, 100.0, 12.0]);
		item.font = world.fonts.find("fonts/test");
		item.text = "a".to_owned();

		let mut list = DrawList::default();
		build(&world, &[item], &mut list);

		let first = list.vertices.first().expect("there is a glyph");
		assert!(
			(first.position[0] - 10.0).abs() < 1.0e-4,
			"the run starts at the left edge of its box, got {}",
			first.position[0]
		);
		assert!(
			first.position[1] >= 20.0,
			"and below its top edge, on the baseline the font asked for: got {}",
			first.position[1]
		);
	}

	#[test]
	fn glyphs_of_one_font_are_one_batch_and_the_boxes_join_it() {
		let (world, panel) = world();
		let mut back = placed(panel, ROOT, [0.0, 0.0, 200.0, 100.0]);
		back.style.background = Some(Color::WHITE);
		let mut words = placed(panel, 1, [0.0, 0.0, 100.0, 12.0]);
		words.font = world.fonts.find("fonts/test");
		words.text = "a".to_owned();

		let mut list = DrawList::default();
		build(&world, &[back, words], &mut list);

		assert_eq!(
			list.batches.len(),
			1,
			"a rectangle does not sample anything, so it costs no draw call of its own"
		);
		assert_eq!(
			list.batches[0].count,
			u32::try_from(list.indices.len()).unwrap_or(0),
			"and the batch covers everything written"
		);
	}

	#[test]
	fn the_glyph_quads_sample_the_part_of_the_atlas_they_were_baked_into() {
		let (world, panel) = world();
		let mut item = placed(panel, 1, [0.0, 0.0, 100.0, 12.0]);
		item.font = world.fonts.find("fonts/test");
		item.text = "a".to_owned();

		let mut list = DrawList::default();
		build(&world, &[item], &mut list);

		let last = list.vertices.get(2).expect("the far corner");
		assert!(
			(last.uv[0] - 0.5).abs() < 1.0e-4 && (last.uv[1] - 0.5).abs() < 1.0e-4,
			"eight texels of a sixteen-texel atlas is half of it, got {:?}",
			last.uv
		);
	}

	#[test]
	fn opacity_reaches_the_color_rather_than_being_dropped() {
		let (world, panel) = world();
		let mut item = placed(panel, ROOT, [0.0, 0.0, 10.0, 10.0]);
		item.style.background = Some(Color::WHITE);
		item.opacity = 0.25;

		let mut list = DrawList::default();
		build(&world, &[item], &mut list);

		let vertex = list.vertices.first().expect("there is a quad");
		assert!(
			(vertex.color[3] - 0.25).abs() < 1.0e-4,
			"a box at a quarter opacity is drawn at a quarter, got {}",
			vertex.color[3]
		);
	}
}
