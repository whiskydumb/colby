//! The properties a stylesheet can set, and how two of them combine.
//!
//! Data only: nothing here reads text. The parser lives in `colby_asset::css`
//! with the other importers, for the same reason the OBJ reader does - a format
//! a person edits is the compiler's business, and what crosses into the engine
//! is the result. What is here is the shape both sides agree on, plus the
//! cascade, which is the one piece of behavior a stylesheet has that is not
//! parsing.
//!
//! Every property is an [`Option`], and that is load-bearing rather than
//! tidiness: "not set" and "set to zero" are different answers, and the whole
//! cascade is "later rules overwrite the properties they mention and leave the
//! rest alone".

use crate::glam::Vec4;

/// How long something is.
///
/// Three of the units CSS has. `em`, `vw` and the rest are not here because
/// nothing has needed them; each is a line in the parser and an arm in the
/// conversion to taffy on the day one does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Length {
	/// A number of pixels.
	Px(f32),

	/// A share of the same measurement on the parent, `0.0 ..= 100.0`.
	Percent(f32),

	/// Whatever the layout works out.
	Auto,
}

impl Length {
	/// Zero pixels, which is what a bare `0` in a stylesheet means.
	pub const ZERO: Self = Self::Px(0.0);

	/// The length in pixels, given what a percentage is a percentage of.
	///
	/// @param whole - what `100%` would be
	/// @return the length, or `None` for [`Auto`](Self::Auto)
	#[must_use]
	pub fn resolve(self, whole: f32) -> Option<f32> {
		match self {
			| Self::Px(pixels) => Some(pixels),
			| Self::Percent(share) => Some(share / 100.0 * whole),
			| Self::Auto => None,
		}
	}
}

/// A color, in linear space with straight alpha.
///
/// Linear rather than sRGB because that is what the surface takes: the
/// swapchain is an sRGB format, so the GPU encodes on the way out and a value
/// handed to it already encoded comes back twice as pale as it was written.
/// The conversion happens once, in [`from_srgb`](Self::from_srgb), where the
/// stylesheet's `#rrggbb` is turned into numbers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Color(pub Vec4);

impl Color {
	/// Nothing at all: the color of a background nobody set.
	pub const NONE: Self = Self(Vec4::ZERO);
	/// Opaque white.
	pub const WHITE: Self = Self(Vec4::ONE);

	/// A color from the bytes a stylesheet writes.
	///
	/// @param red - the sRGB red byte
	/// @param green - the sRGB green byte
	/// @param blue - the sRGB blue byte
	/// @param alpha - opacity, `0 ..= 255`, which is *not* encoded
	#[must_use]
	pub fn from_srgb(red: u8, green: u8, blue: u8, alpha: u8) -> Self {
		Self(Vec4::new(linear(red), linear(green), linear(blue), f32::from(alpha) / 255.0))
	}

	/// Whether this would draw nothing.
	#[must_use]
	pub fn is_invisible(self) -> bool { self.0.w <= 0.0 }

	/// The same color at another opacity.
	#[must_use]
	pub fn faded(self, by: f32) -> Self { Self(self.0 * Vec4::new(1.0, 1.0, 1.0, by)) }
}

/// Whether a box lays its children out or is not drawn at all.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Display {
	/// A flex container. The only layout mode there is.
	#[default]
	Flex,

	/// Not laid out, not drawn, not hit-tested.
	None,
}

/// Whether a box is placed by the flow or against its parent's edges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Position {
	/// Placed by the flow, and `left`/`top` nudge it from there.
	#[default]
	Relative,

	/// Taken out of the flow and placed against the parent's padding box.
	Absolute,
}

/// Which way a container stacks its children.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Direction {
	/// Left to right.
	#[default]
	Row,

	/// Right to left.
	RowReverse,

	/// Top to bottom.
	Column,

	/// Bottom to top.
	ColumnReverse,
}

/// Whether children may start a second line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Wrap {
	/// One line, however tight it gets.
	#[default]
	NoWrap,

	/// As many lines as it takes.
	Wrap,
}

/// How the spare room along the main axis is handed out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Justify {
	/// All of it after the last child.
	#[default]
	Start,

	/// All of it before the first.
	End,

	/// Half either side.
	Center,

	/// Between the children, none at the ends.
	SpaceBetween,

	/// Between the children and a half share at the ends.
	SpaceAround,

	/// Equal shares everywhere, ends included.
	SpaceEvenly,
}

/// How children sit across the other axis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Align {
	/// Against the near edge.
	Start,

	/// Against the far edge.
	End,

	/// In the middle.
	Center,

	/// Filling it.
	#[default]
	Stretch,
}

/// What a box does with what does not fit inside it.
///
/// One property rather than CSS's `overflow-x` and `overflow-y`, and clipping
/// on one axis clips on both. Two axes would want two clip rectangles that are
/// not a rectangle between them, and the case that wants only one - a row of
/// words that may run wide and must not run tall - is better served by wrapping
/// than by letting it escape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Overflow {
	/// It draws outside its box, as far as it likes.
	#[default]
	Visible,

	/// It is cut off at the edge of the box.
	Hidden,

	/// It is cut off, and the box can be scrolled to see the rest.
	Scroll,
}

impl Overflow {
	/// Whether this cuts anything off.
	///
	/// [`Scroll`](Self::Scroll) clips exactly as [`Hidden`](Self::Hidden) does;
	/// what it adds is a way to move what is behind the cut, and nothing about
	/// the cut itself.
	#[must_use]
	pub const fn clips(self) -> bool { !matches!(self, Self::Visible) }
}

/// Four numbers around a box.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Edges {
	/// The left one.
	pub left: Option<Length>,

	/// The right one.
	pub right: Option<Length>,

	/// The top one.
	pub top: Option<Length>,

	/// The bottom one.
	pub bottom: Option<Length>,
}

impl Edges {
	/// The same length on all four sides.
	#[must_use]
	pub const fn all(length: Length) -> Self {
		Self {
			left: Some(length),
			right: Some(length),
			top: Some(length),
			bottom: Some(length),
		}
	}

	/// Takes every side the other one sets.
	pub fn merge(&mut self, other: Self) {
		self.left = other.left.or(self.left);
		self.right = other.right.or(self.right);
		self.top = other.top.or(self.top);
		self.bottom = other.bottom.or(self.bottom);
	}

	/// Whether nothing at all is set.
	#[must_use]
	pub fn is_unset(&self) -> bool {
		self.left.is_none() && self.right.is_none() && self.top.is_none() && self.bottom.is_none()
	}
}

/// Everything a rule, an attribute or a game can say about a box.
///
/// Not `Copy`, because of the one string in it. Cloned once per node per
/// layout, which for an interface of a few dozen boxes is not a number worth
/// designing around.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Style {
	/// Whether the box exists at all.
	pub display: Option<Display>,

	/// Whether it is in the flow.
	pub position: Option<Position>,

	/// Which way its children stack.
	pub direction: Option<Direction>,

	/// Whether they may start a second line.
	pub wrap: Option<Wrap>,

	/// What happens to spare room along the main axis.
	pub justify: Option<Justify>,

	/// Where children sit across the other one.
	pub align: Option<Align>,

	/// What it does with what does not fit inside it.
	pub overflow: Option<Overflow>,

	/// This box's share of its parent's spare room.
	pub grow: Option<f32>,

	/// Its share of the shortfall when there is not enough.
	pub shrink: Option<f32>,

	/// Its size along the main axis before any of that.
	pub basis: Option<Length>,

	/// The gap between children, both ways.
	pub gap: Option<Length>,

	/// A fixed width.
	pub width: Option<Length>,

	/// A fixed height.
	pub height: Option<Length>,

	/// The narrowest it may be.
	pub min_width: Option<Length>,

	/// The shortest it may be.
	pub min_height: Option<Length>,

	/// The widest it may be.
	pub max_width: Option<Length>,

	/// The tallest it may be.
	pub max_height: Option<Length>,

	/// Room outside the box.
	pub margin: Edges,

	/// Room inside it, between its edge and its children.
	pub padding: Edges,

	/// Where an absolutely positioned box sits, or how far a relative one is
	/// nudged.
	pub inset: Edges,

	/// What is painted behind its children.
	pub background: Option<Color>,

	/// What its text is painted in. Inherited.
	pub color: Option<Color>,

	/// How big its text is. Inherited.
	pub font_size: Option<Length>,

	/// Which font, by the name the asset registered under. Inherited.
	pub font_family: Option<String>,

	/// How round the corners of its background are.
	pub radius: Option<Length>,

	/// How opaque the whole box is, children included.
	pub opacity: Option<f32>,
}

impl Style {
	/// The values everything starts from, before a stylesheet says anything.
	///
	/// A box with nothing set is a transparent flex row that takes the size of
	/// what is in it, and that is deliberately the same set of defaults CSS
	/// has - apart from `flex-direction`, where CSS also says row.
	#[must_use]
	pub fn root() -> Self {
		Self {
			display: Some(Display::Flex),
			position: Some(Position::Relative),
			direction: Some(Direction::Row),
			wrap: Some(Wrap::NoWrap),
			justify: Some(Justify::Start),
			align: Some(Align::Stretch),
			overflow: Some(Overflow::Visible),
			grow: Some(0.0),
			shrink: Some(1.0),
			basis: Some(Length::Auto),
			gap: Some(Length::ZERO),
			width: Some(Length::Auto),
			height: Some(Length::Auto),
			min_width: None,
			min_height: None,
			max_width: None,
			max_height: None,
			margin: Edges::all(Length::ZERO),
			padding: Edges::all(Length::ZERO),
			inset: Edges::default(),
			background: Some(Color::NONE),
			color: Some(Color::WHITE),
			font_size: Some(Length::Px(DEFAULT_FONT_SIZE)),
			font_family: None,
			radius: Some(Length::ZERO),
			opacity: Some(1.0),
		}
	}

	/// Takes every property the other one sets and keeps the rest.
	///
	/// The whole of the cascade. Rules are applied in order of how specific
	/// they are, then in the order they were written, and the `style`
	/// attribute last of all.
	pub fn merge(&mut self, other: &Self) {
		self.display = other.display.or(self.display);
		self.position = other.position.or(self.position);
		self.direction = other.direction.or(self.direction);
		self.wrap = other.wrap.or(self.wrap);
		self.justify = other.justify.or(self.justify);
		self.align = other.align.or(self.align);
		self.overflow = other.overflow.or(self.overflow);
		self.grow = other.grow.or(self.grow);
		self.shrink = other.shrink.or(self.shrink);
		self.basis = other.basis.or(self.basis);
		self.gap = other.gap.or(self.gap);
		self.width = other.width.or(self.width);
		self.height = other.height.or(self.height);
		self.min_width = other.min_width.or(self.min_width);
		self.min_height = other.min_height.or(self.min_height);
		self.max_width = other.max_width.or(self.max_width);
		self.max_height = other.max_height.or(self.max_height);
		self.margin.merge(other.margin);
		self.padding.merge(other.padding);
		self.inset.merge(other.inset);
		self.background = other.background.or(self.background);
		self.color = other.color.or(self.color);
		self.font_size = other.font_size.or(self.font_size);
		self.radius = other.radius.or(self.radius);
		self.opacity = other.opacity.or(self.opacity);

		if other.font_family.is_some() {
			self.font_family.clone_from(&other.font_family);
		}
	}

	/// The properties a child takes from its parent, and nothing else.
	///
	/// Three of them, which is the short answer to "how much of CSS
	/// inheritance is this". Text properties inherit because a paragraph
	/// inside a panel should be the panel's color without being told; box
	/// properties do not, because a child that inherited its parent's width
	/// would be unusable.
	#[must_use]
	pub fn inherited(&self) -> Self {
		Self {
			color: self.color,
			font_size: self.font_size,
			font_family: self.font_family.clone(),
			..Self::default()
		}
	}

	/// Whether this box is drawn and laid out at all.
	#[must_use]
	pub fn is_shown(&self) -> bool { self.display != Some(Display::None) }
}

/// How big text is when nothing says otherwise, in pixels.
pub const DEFAULT_FONT_SIZE: f32 = 16.0;

/// One channel of sRGB as a linear number.
///
/// The exact curve rather than a gamma of 2.2: the two differ most in the dark
/// end, which is where an interface puts its shadows and its panel backgrounds.
fn linear(channel: u8) -> f32 {
	let value = f32::from(channel) / 255.0;

	if value <= 0.040_45 {
		value / 12.92
	} else {
		((value + 0.055) / 1.055).powf(2.4)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_percentage_is_a_share_of_what_it_is_measured_against() {
		assert_eq!(Length::Px(12.0).resolve(400.0), Some(12.0), "pixels are pixels");
		assert_eq!(
			Length::Percent(25.0).resolve(400.0),
			Some(100.0),
			"a quarter of four hundred"
		);
		assert_eq!(Length::Auto.resolve(400.0), None, "and auto is the layout's business");
	}

	#[test]
	fn a_stylesheet_color_is_stored_the_way_the_surface_wants_it() {
		let white = Color::from_srgb(255, 255, 255, 255);
		let black = Color::from_srgb(0, 0, 0, 255);
		let grey = Color::from_srgb(128, 128, 128, 255);

		assert!(white.0.abs_diff_eq(Vec4::ONE, 1.0e-4), "white stays white");
		assert!(
			black
				.0
				.abs_diff_eq(Vec4::new(0.0, 0.0, 0.0, 1.0), 1.0e-4),
			"and black, black"
		);
		assert!(
			grey.0.x > 0.2 && grey.0.x < 0.23,
			"a mid grey is about a fifth of the way up in linear light, not half: got {}",
			grey.0.x
		);
	}

	#[test]
	fn alpha_is_not_encoded_along_with_the_color() {
		let half = Color::from_srgb(255, 255, 255, 128);

		assert!(
			(half.0.w - 128.0 / 255.0).abs() < 1.0e-4,
			"opacity is a fraction, not a brightness, and putting it through the curve would \
			 make everything translucent darker than it was asked to be"
		);
	}

	#[test]
	fn merging_takes_what_is_set_and_leaves_what_is_not() {
		let mut base = Style {
			width: Some(Length::Px(100.0)),
			color: Some(Color::WHITE),
			..Style::default()
		};

		base.merge(&Style {
			width: Some(Length::Px(200.0)),
			height: Some(Length::Px(50.0)),
			..Style::default()
		});

		assert_eq!(base.width, Some(Length::Px(200.0)), "the later rule wins where it speaks");
		assert_eq!(base.height, Some(Length::Px(50.0)), "and adds what was not there");
		assert_eq!(base.color, Some(Color::WHITE), "and says nothing about the rest");
	}

	#[test]
	fn only_the_text_properties_are_inherited() {
		let parent = Style {
			width: Some(Length::Px(100.0)),
			color: Some(Color::from_srgb(255, 0, 0, 255)),
			font_size: Some(Length::Px(20.0)),
			font_family: Some("fonts/hack".to_owned()),
			padding: Edges::all(Length::Px(8.0)),
			..Style::default()
		};

		let child = parent.inherited();

		assert_eq!(child.color, parent.color, "text color comes down");
		assert_eq!(child.font_size, parent.font_size, "so does its size");
		assert_eq!(child.font_family, parent.font_family, "and which font it is in");
		assert_eq!(child.width, None, "a child that inherited a width would be unusable");
		assert!(child.padding.is_unset(), "and one that inherited padding, nested");
	}

	#[test]
	fn edges_merge_a_side_at_a_time() {
		let mut edges = Edges::all(Length::Px(4.0));
		edges.merge(Edges {
			left: Some(Length::Px(12.0)),
			..Edges::default()
		});

		assert_eq!(edges.left, Some(Length::Px(12.0)), "the side that was named");
		assert_eq!(edges.right, Some(Length::Px(4.0)), "and not the three that were not");
	}

	#[test]
	fn the_root_style_sets_every_property_it_can() {
		let root = Style::root();

		assert!(root.display.is_some(), "so that nothing downstream has to guess a default");
		assert!(root.font_size.is_some(), "text has a size before a stylesheet is read");
		assert!(root.is_shown(), "and the root of a document is on screen");
		assert_eq!(root.font_family, None, "the one thing there is no sensible default for");
	}
}
