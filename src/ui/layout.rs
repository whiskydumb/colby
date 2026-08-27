//! Turning a document into a list of boxes with positions on them.
//!
//! taffy does the flexbox; everything here is the glue either side of it. That
//! is what keeps this module a couple of hundred lines rather than a thousand:
//! writing a flexbox implementation is a project, and using one is an
//! afternoon.
//!
//! The one interesting part is that **text is measured from inside the
//! layout**. A text node's width and height *are* its words, so taffy calls
//! back into [`FontData::measure`](colby_core::abi::FontData::measure) while it
//! is deciding how wide the box around them should be, and the answer depends
//! on how much room it was offered - which is how wrapping works at all.
//!
//! The tree is rebuilt from scratch every time this runs, twice a frame: once
//! in the step, to hit-test the pointer against what is on screen, and once
//! before the draw, because the window may have been resized since. An
//! interface is a few dozen boxes and taffy is fast; when that stops being true
//! the answer is to keep the tree and mark it dirty, and nothing above this
//! module would notice.

use colby_core::{
	abi::{
		FontId, World,
		ui::{
			DocumentData, Node, PanelId,
			document::{NONE, ROOT},
			style::{Align, Color, Direction, Justify, Length, Position, Style, Wrap},
		},
	},
	glam::Vec2,
};
use taffy::{
	AvailableSpace, Dimension, LengthPercentage, LengthPercentageAuto, NodeId, Rect, Size,
	TaffyTree, compute_leaf_layout,
	prelude::{FlexDirection, FlexWrap, JustifyContent},
	style::{
		AlignItems, Display as TaffyDisplay, Position as TaffyPosition, Style as TaffyStyle,
	},
};

/// One box, laid out, in layout pixels with the origin at the top left.
#[derive(Clone, Debug)]
pub struct Placed {
	/// Which panel it belongs to.
	pub panel: PanelId,

	/// Which node of that panel's document it is.
	pub node: u32,

	/// Where it is: x, y, width, height.
	pub rect: [f32; 4],

	/// Everything the cascade decided about it.
	pub style: Style,

	/// The font its text is in, already resolved.
	pub font: FontId,

	/// How big that text is, in layout pixels.
	pub font_size: f32,

	/// How opaque it is, its ancestors' opacity included.
	pub opacity: f32,

	/// The words it draws, if it is a run of text.
	pub text: String,

	/// The texture it draws, if it is an image.
	pub image: String,
}

impl Placed {
	/// Whether a point in layout pixels is inside this box.
	#[must_use]
	pub fn contains(&self, point: Vec2) -> bool {
		point.x >= self.rect[0]
			&& point.y >= self.rect[1]
			&& point.x < self.rect[0] + self.rect[2]
			&& point.y < self.rect[1] + self.rect[3]
	}
}

/// What a text leaf needs to measure itself.
struct Measure {
	text: String,
	font: FontId,
	size: f32,
}

/// One node on its way into taffy.
struct Building {
	document_node: u32,
	taffy: NodeId,
	parent: usize,
	style: Style,
	font: FontId,
	font_size: f32,
	opacity: f32,
	text: String,
	image: String,
}

/// Lays out every panel that is on screen, in the order they are drawn.
///
/// @param world - the documents, the fonts and the viewport
/// @param into - cleared and filled with one entry per visible box
pub fn run(world: &World, into: &mut Vec<Placed>) {
	into.clear();

	let viewport = world.ui.viewport();

	for (id, panel) in world.ui.panels() {
		if !panel.is_shown() {
			continue;
		}

		let Some(document) = world
			.ui
			.document(panel.document())
			.map(colby_core::abi::registry::Entry::value)
		else {
			continue;
		};

		place(world, id, document, viewport, into);
	}
}

/// Lays out one panel.
fn place(
	world: &World,
	id: PanelId,
	document: &DocumentData,
	viewport: Vec2,
	into: &mut Vec<Placed>,
) {
	let Some(panel) = world.ui.panel(id) else {
		return;
	};

	let mut tree: TaffyTree<Measure> = TaffyTree::new();
	let mut building: Vec<Building> = Vec::new();

	// the root is a box the size of the window, so that percentages at the top
	// of a document mean a share of the screen.
	let root = Style {
		width: Some(Length::Px(viewport.x)),
		height: Some(Length::Px(viewport.y)),
		..Style::root()
	};

	if !build(world, panel, document, ROOT, &root, usize::MAX, &mut tree, &mut building) {
		return;
	}

	let Some(top) = building.first().map(|it| it.taffy) else {
		return;
	};

	// children are attached after every node exists, so the tree can be built
	// top down: taffy's `new_with_children` wants them the other way round, and
	// building bottom up would put parents after their children in `building`,
	// which is the order the absolute positions are accumulated in.
	for node in building.iter().skip(1) {
		if let Some(parent) = building.get(node.parent) {
			drop(tree.add_child(parent.taffy, node.taffy));
		}
	}

	let available = Size {
		width: AvailableSpace::Definite(viewport.x),
		height: AvailableSpace::Definite(viewport.y),
	};

	let measured =
		tree.compute_layout_with_measure(top, available, |inputs, _, context, style| {
			compute_leaf_layout(
				inputs,
				style,
				|_, _| 0.0,
				|known, available| measure(world, context, known, available),
			)
		});

	if measured.is_err() {
		return;
	}

	// absolute positions, accumulated in creation order. A node's parent is
	// always earlier in the list than it is, so one pass is enough.
	let mut origins: Vec<Vec2> = Vec::with_capacity(building.len());

	for node in &building {
		let Ok(layout) = tree.layout(node.taffy) else {
			origins.push(Vec2::ZERO);

			continue;
		};

		let parent = origins
			.get(node.parent)
			.copied()
			.unwrap_or(Vec2::ZERO);
		let origin = parent + Vec2::new(layout.location.x, layout.location.y);
		origins.push(origin);

		into.push(Placed {
			panel: id,
			node: node.document_node,
			rect: [origin.x, origin.y, layout.size.width, layout.size.height],
			style: node.style.clone(),
			font: node.font,
			font_size: node.font_size,
			opacity: node.opacity,
			text: node.text.clone(),
			image: node.image.clone(),
		});
	}
}

/// Adds one node and everything under it.
///
/// @return whether the node was added at all; a hidden box adds nothing
#[expect(
	clippy::too_many_arguments,
	reason = "every one of these is threaded down the recursion, and a struct holding them \
	          would be the same arguments behind one name"
)]
fn build(
	world: &World,
	panel: &colby_core::abi::ui::Panel,
	document: &DocumentData,
	index: u32,
	inherited: &Style,
	parent: usize,
	tree: &mut TaffyTree<Measure>,
	building: &mut Vec<Building>,
) -> bool {
	let Some(node) = document.node(index) else {
		return false;
	};

	let hovered = is_hovered(document, panel.hovered(), index);
	let mut style = document.computed(index, inherited, panel.classes(node), hovered);

	if let Some(bind) = panel.bind(&node.id) {
		style.merge(&bind.style);
	}

	if !style.is_shown() {
		return false;
	}

	let parent_size = inherited
		.font_size
		.and_then(|length| length.resolve(0.0))
		.unwrap_or(colby_core::abi::ui::style::DEFAULT_FONT_SIZE);
	let font_size = DocumentData::font_size(&style, parent_size);
	let font = style
		.font_family
		.as_deref()
		.map_or(FontId::NONE, |name| world.fonts.find(name));

	let opacity = style.opacity.unwrap_or(1.0).clamp(0.0, 1.0)
		* inherited.opacity.unwrap_or(1.0).clamp(0.0, 1.0);

	// a run of text is bound to by the nearest box with an `id`, because that is
	// the box a person wrote the identifier on: `set_text(hud, "score", ..)` is
	// meant to replace what is inside `<div id="score">`, and the words under it
	// are a child nobody named.
	let text = if node.is_text() {
		panel
			.bind(named(document, index).unwrap_or_default())
			.and_then(|bind| bind.text.clone())
			.unwrap_or_else(|| node.text.clone())
	} else {
		panel.text(node).to_owned()
	};

	let mut taffy_style = convert(&style);
	if node.is_text() {
		// a run of text is exactly its words and never stretches to fill the box
		// around it. `align-items: stretch` is CSS's default and the right one
		// for boxes - a sidebar should fill the height it is given - but a line
		// of text stretched to six hundred pixels is six hundred pixels of box
		// to hit-test against twelve pixels of letters.
		taffy_style.align_self = Some(AlignItems::FLEX_START);
	}

	let taffy = if node.is_text() {
		tree.new_leaf_with_context(taffy_style, Measure {
			text: text.clone(),
			font,
			size: font_size,
		})
	} else {
		tree.new_leaf(taffy_style)
	};

	let Ok(taffy) = taffy else {
		return false;
	};

	let self_index = building.len();
	building.push(Building {
		document_node: index,
		taffy,
		parent: if parent == usize::MAX { 0 } else { parent },
		font,
		font_size,
		opacity,
		text: if node.is_text() { text } else { String::new() },
		image: node.source.clone(),
		style: style.clone(),
	});

	// what children start from: the text properties, at this node's own size,
	// so that a percentage font size is a share of the parent's rather than of
	// the document's.
	let mut down = style.inherited();
	down.font_size = Some(Length::Px(font_size));
	down.opacity = Some(opacity);

	for child in document.children(index) {
		build(world, panel, document, child, &down, self_index, tree, building);
	}

	true
}

/// Measures one run of text, given how much room it was offered.
fn measure(
	world: &World,
	context: Option<&mut Measure>,
	known: Size<Option<f32>>,
	available: Size<AvailableSpace>,
) -> Size<f32> {
	let Some(context) = context else {
		return Size::ZERO;
	};

	// the width it is allowed to take before it has to break. A definite
	// offer is a real limit; `MinContent` asks how narrow it can be, which for
	// a run of words is the longest word, and taffy's own answer to that is to
	// offer zero.
	let wrap = match (known.width, available.width) {
		| (Some(width), _) | (None, AvailableSpace::Definite(width)) => Some(width),
		| (None, AvailableSpace::MinContent) => Some(0.0),
		| (None, AvailableSpace::MaxContent) => None,
	};

	let size = world
		.fonts
		.data(context.font)
		.measure(&context.text, context.size, wrap);

	Size {
		width: known.width.unwrap_or(size.x),
		height: known.height.unwrap_or(size.y),
	}
}

/// Whether a node is under the pointer, or has the node that is inside it.
///
/// The second half is what CSS means by `:hover`: hovering a label inside a
/// button hovers the button too, which is what makes a button light up when the
/// pointer is over its text rather than over its padding.
fn is_hovered(document: &DocumentData, hovered: u32, index: u32) -> bool {
	if hovered == NONE {
		return false;
	}

	let mut walk = hovered;
	let mut guard = 0;

	while walk != NONE && guard < document.nodes.len() {
		if walk == index {
			return true;
		}

		walk = document
			.node(walk)
			.map_or(NONE, |node| node.parent);
		guard += 1;
	}

	false
}

/// Turns one computed style into the taffy one.
fn convert(style: &Style) -> TaffyStyle {
	TaffyStyle {
		display: TaffyDisplay::Flex,
		position: match style.position {
			| Some(Position::Absolute) => TaffyPosition::Absolute,
			| _ => TaffyPosition::Relative,
		},
		inset: inset_rect(&style.inset),
		size: Size {
			width: dimension(style.width),
			height: dimension(style.height),
		},
		min_size: Size {
			width: auto_length(style.min_width),
			height: auto_length(style.min_height),
		},
		max_size: Size {
			width: auto_length(style.max_width),
			height: auto_length(style.max_height),
		},
		margin: margin_rect(&style.margin),
		padding: rect(&style.padding),
		gap: Size {
			width: length(style.gap),
			height: length(style.gap),
		},
		flex_direction: match style.direction {
			| Some(Direction::RowReverse) => FlexDirection::RowReverse,
			| Some(Direction::Column) => FlexDirection::Column,
			| Some(Direction::ColumnReverse) => FlexDirection::ColumnReverse,
			| _ => FlexDirection::Row,
		},
		flex_wrap: match style.wrap {
			| Some(Wrap::Wrap) => FlexWrap::Wrap,
			| _ => FlexWrap::NoWrap,
		},
		flex_grow: style.grow.unwrap_or(0.0),
		flex_shrink: style.shrink.unwrap_or(1.0),
		flex_basis: dimension(style.basis),
		align_items: Some(match style.align {
			| Some(Align::Start) => AlignItems::FLEX_START,
			| Some(Align::End) => AlignItems::FLEX_END,
			| Some(Align::Center) => AlignItems::CENTER,
			| _ => AlignItems::STRETCH,
		}),
		justify_content: Some(match style.justify {
			| Some(Justify::End) => JustifyContent::FLEX_END,
			| Some(Justify::Center) => JustifyContent::CENTER,
			| Some(Justify::SpaceBetween) => JustifyContent::SPACE_BETWEEN,
			| Some(Justify::SpaceAround) => JustifyContent::SPACE_AROUND,
			| Some(Justify::SpaceEvenly) => JustifyContent::SPACE_EVENLY,
			| _ => JustifyContent::FLEX_START,
		}),
		..TaffyStyle::default()
	}
}

/// A length that may be `auto`, as a taffy size.
fn dimension(value: Option<Length>) -> Dimension {
	match value {
		| Some(Length::Px(pixels)) => Dimension::length(pixels),
		| Some(Length::Percent(share)) => Dimension::percent(share / 100.0),
		| _ => Dimension::auto(),
	}
}

/// The same, where taffy wants its other type.
fn auto_length(value: Option<Length>) -> LengthPercentageAuto {
	match value {
		| Some(Length::Px(pixels)) => LengthPercentageAuto::length(pixels),
		| Some(Length::Percent(share)) => LengthPercentageAuto::percent(share / 100.0),
		| _ => LengthPercentageAuto::auto(),
	}
}

/// A length that may not be `auto`.
fn length(value: Option<Length>) -> LengthPercentage {
	match value {
		| Some(Length::Percent(share)) => LengthPercentage::percent(share / 100.0),
		| Some(Length::Px(pixels)) => LengthPercentage::length(pixels),
		| _ => LengthPercentage::length(0.0),
	}
}

/// Four sides, none of which may be `auto`.
fn rect(edges: &colby_core::abi::ui::Edges) -> Rect<LengthPercentage> {
	Rect {
		left: length(edges.left),
		right: length(edges.right),
		top: length(edges.top),
		bottom: length(edges.bottom),
	}
}

/// Where an absolutely positioned box sits, where unset means `auto`.
fn inset_rect(edges: &colby_core::abi::ui::Edges) -> Rect<LengthPercentageAuto> {
	Rect {
		left: auto_length(edges.left),
		right: auto_length(edges.right),
		top: auto_length(edges.top),
		bottom: auto_length(edges.bottom),
	}
}

/// Margins, where unset means **zero** rather than `auto`.
///
/// The distinction is not pedantry: `margin: auto` is how CSS centers a flex
/// item, so handing taffy an `auto` for every side nobody mentioned centers
/// every box in the document inside its parent. That is one word of difference
/// and a layout that looks nothing like the one that was written.
fn margin_rect(edges: &colby_core::abi::ui::Edges) -> Rect<LengthPercentageAuto> {
	let side = |value: Option<Length>| match value {
		| Some(Length::Px(pixels)) => LengthPercentageAuto::length(pixels),
		| Some(Length::Percent(share)) => LengthPercentageAuto::percent(share / 100.0),
		| Some(Length::Auto) => LengthPercentageAuto::auto(),
		| None => LengthPercentageAuto::length(0.0),
	};

	Rect {
		left: side(edges.left),
		right: side(edges.right),
		top: side(edges.top),
		bottom: side(edges.bottom),
	}
}

/// What a box is painted, its ancestors' opacity applied.
#[must_use]
pub fn background(placed: &Placed) -> Color {
	placed
		.style
		.background
		.unwrap_or(Color::NONE)
		.faded(placed.opacity)
}

/// What a box's text is painted, the same way.
#[must_use]
pub fn foreground(placed: &Placed) -> Color {
	placed
		.style
		.color
		.unwrap_or(Color::WHITE)
		.faded(placed.opacity)
}

/// The node an event about `index` should name.
///
/// The nearest box with an `id`, starting at the one that was hit. Clicking the
/// words on a button is clicking the button, and the words are not something a
/// document had any reason to name.
#[must_use]
pub fn named(document: &DocumentData, index: u32) -> Option<&str> {
	let mut walk = index;
	let mut guard = 0;

	while walk != NONE && guard < document.nodes.len() {
		let node: &Node = document.node(walk)?;
		if !node.id.is_empty() {
			return Some(&node.id);
		}

		walk = node.parent;
		guard += 1;
	}

	None
}
