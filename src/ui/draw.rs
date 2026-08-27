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
	abi::{FontId, TextureId, World, ui::DocumentData},
	bytemuck::{Pod, Zeroable},
	glam::Vec4,
};

use crate::layout::{Placed, background, foreground};

/// One corner of one quad.
///
/// Fat, at fifty-six bytes. An interface is thousands of these rather than
/// millions, and the alternative - a uniform per box, or a pipeline per kind -
/// costs a draw call per box instead of a few floats per corner.
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
}

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
	fn quad(&mut self, rect: [f32; 4], uv: [f32; 4], color: Vec4, radius: f32, kind: f32) {
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
		list.quad(box_of.rect, [0.0, 0.0, 1.0, 1.0], background(box_of).0, radius, KIND_RECT);
		list.close();

		if node.is_image() && !box_of.image.is_empty() {
			image(world, box_of, radius, list);
		}

		if node.is_text() && !box_of.text.is_empty() {
			text(world, box_of, list);
		}
	}

	list.close();
}

/// Adds an image over its whole box.
fn image(world: &World, box_of: &Placed, radius: f32, list: &mut DrawList) {
	let id = world.textures.find(&box_of.image);
	if !id.is_some() {
		return;
	}

	list.want(Binding::Image(id));
	list.quad(box_of.rect, [0.0, 0.0, 1.0, 1.0], foreground(box_of).0, radius, KIND_IMAGE);
	list.close();
}

/// Adds one glyph quad per drawn character of a run of text.
fn text(world: &World, box_of: &Placed, list: &mut DrawList) {
	if !box_of.font.is_some() {
		return;
	}

	let font = world.fonts.data(box_of.font);
	if font.atlas_width == 0 || font.atlas_height == 0 {
		return;
	}

	let scale = font.scale(box_of.font_size);
	let atlas = (texels(font.atlas_width), texels(font.atlas_height));
	// how many layout pixels the whole `0 ..= 255` range of a distance byte
	// covers at this size: the shader turns a sample back into a distance with
	// it, and antialiases against that.
	let range = font.spread * 2.0 * scale;
	let color = foreground(box_of).0;

	list.want(Binding::Font(box_of.font));

	// wrapped at a hair over the box, not exactly at it: the box was sized by
	// this same measurement and then rounded to whole pixels, and breaking a
	// line half a pixel earlier than it was measured to break would put the
	// last word somewhere the layout did not reserve room for.
	let wrap = Some(box_of.rect[2] + 1.0);

	font.run(&box_of.text, box_of.font_size, wrap, |glyph, x, y| {
		let width = f32::from(glyph.atlas_width) * scale;
		let height = f32::from(glyph.atlas_height) * scale;

		list.quad(
			[box_of.rect[0] + x, box_of.rect[1] + y, width, height],
			[
				f32::from(glyph.atlas_x) / atlas.0,
				f32::from(glyph.atlas_y) / atlas.1,
				f32::from(glyph.atlas_x + glyph.atlas_width) / atlas.0,
				f32::from(glyph.atlas_y + glyph.atlas_height) / atlas.1,
			],
			color,
			range,
			KIND_GLYPH,
		);
	});

	list.close();
}

/// An atlas dimension as a number, never zero.
fn texels(value: u32) -> f32 { f32::from(u16::try_from(value).unwrap_or(u16::MAX)).max(1.0) }

#[cfg(test)]
mod tests {
	use colby_core::abi::{
		FontData, Glyph,
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
