//! A parsed interface document: its boxes, its rules, and the cascade.
//!
//! The tree is a flat array with links rather than boxes inside boxes, for the
//! same reason the entity table is: everything that walks it wants an index it
//! can keep, and an index is what crosses back out to the layout and the
//! renderer without borrowing the document while they work.
//!
//! Nothing here parses anything. `colby_asset::html` and `colby_asset::css`
//! build one of these, the compiler writes it into a `.cdoc`, and the runner
//! registers the result - the same path a mesh takes from `.obj` to
//! [`Meshes`](crate::abi::Meshes).

use super::style::{Length, Style};

/// The link value meaning there is no such node.
pub const NONE: u32 = u32::MAX;

/// The index of a document's root, which every document has.
pub const ROOT: u32 = 0;

/// One box, or one run of text.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Node {
	/// The element name, lowercase. Empty for a run of text.
	pub tag: String,

	/// The `id` attribute, or empty. What a game addresses this node by.
	pub id: String,

	/// The `class` attribute as written, space separated.
	pub classes: String,

	/// The text this node draws. Only ever set on a text node.
	pub text: String,

	/// For an image, the name of the texture to draw.
	pub source: String,

	/// The `style` attribute, already parsed.
	pub inline: Style,

	/// The node this hangs off, or [`NONE`] for the root.
	pub parent: u32,

	/// Its first child, or [`NONE`].
	pub first_child: u32,

	/// The next node with the same parent, or [`NONE`].
	pub next_sibling: u32,
}

impl Node {
	/// The tag an image has.
	pub const IMAGE: &'static str = "img";
	/// The tag a field somebody can type into has.
	pub const INPUT: &'static str = "input";
	/// The tag a run of text has.
	pub const TEXT: &'static str = "";

	/// Whether this node draws words rather than a box.
	#[must_use]
	pub fn is_text(&self) -> bool { self.tag == Self::TEXT }

	/// Whether this node draws a picture.
	#[must_use]
	pub fn is_image(&self) -> bool { self.tag == Self::IMAGE }

	/// Whether this node is a field somebody can type into.
	///
	/// Unlike every other box, its words are its own rather than a child's:
	/// what it holds is a value rather than content, so there is nothing for a
	/// run of text under it to be. That is also what makes
	/// [`Ui::set_text`](super::Ui::set_text) fill it in and
	/// [`Ui::text`](super::Ui::text) read it back, with no rule of its own.
	#[must_use]
	pub fn is_input(&self) -> bool { self.tag == Self::INPUT }

	/// Whether this node draws words of its own.
	#[must_use]
	pub fn has_words(&self) -> bool { self.is_text() || self.is_input() }
}

/// What a rule matches: a tag, a class, an identifier, and whether the pointer
/// has to be over the box.
///
/// One of each at most, and no combinators. `div.panel#hud:hover` is as
/// complicated as a selector gets, which is enough to style an interface and
/// short enough to match without building a matching engine. An empty field
/// matches anything.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Selector {
	/// The element name, or empty for any.
	pub tag: String,

	/// One class name, or empty for any.
	pub class: String,

	/// One identifier, or empty for any.
	pub id: String,

	/// Whether this only applies while the pointer is over the box.
	pub hover: bool,
}

impl Selector {
	/// How strongly this claims a node, the way CSS counts it.
	///
	/// An identifier beats any number of classes, a class beats any number of
	/// tags, and `:hover` counts as a class - which is what makes
	/// `.button:hover` win over `.button` however they are ordered.
	#[must_use]
	pub fn specificity(&self) -> u32 {
		let mut score = 0;

		if !self.id.is_empty() {
			score += 100;
		}

		if !self.class.is_empty() {
			score += 10;
		}

		if self.hover {
			score += 10;
		}

		if !self.tag.is_empty() {
			score += 1;
		}

		score
	}

	/// Whether this selector claims a node.
	///
	/// @param node - the node in question
	/// @param classes - its class list, which the game may have replaced
	/// @param hovered - whether the pointer is over it
	#[must_use]
	pub fn matches(&self, node: &Node, classes: &str, hovered: bool) -> bool {
		if self.hover && !hovered {
			return false;
		}

		if !self.tag.is_empty() && self.tag != node.tag {
			return false;
		}

		if !self.id.is_empty() && self.id != node.id {
			return false;
		}

		if !self.class.is_empty()
			&& !classes
				.split_whitespace()
				.any(|class| class == self.class)
		{
			return false;
		}

		true
	}
}

/// One `selector { property: value; }` from a stylesheet.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rule {
	/// What it claims.
	pub selector: Selector,

	/// What it sets.
	pub style: Style,
}

/// A whole interface document: the boxes and the rules over them.
///
/// One file, one document, one registry entry. A `<link>` to a stylesheet is
/// resolved by the compiler and folded in here, so nothing at runtime has to
/// find a second file or decide what to do when it is missing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DocumentData {
	/// Every node, the root first.
	pub nodes: Vec<Node>,

	/// Every rule, in the order they were written.
	pub rules: Vec<Rule>,

	/// The name of the program this document's interface logic is in, or
	/// empty.
	///
	/// A **name**, not the text: `<script src="hud.lua">` is resolved by the
	/// compiler into the name that same file is registered under - `ui/hud` -
	/// and the program itself lives in
	/// [`World::scripts`](crate::abi::World::scripts) like any other asset.
	/// That is what a `<link>` does *not* do, and the difference is deliberate:
	/// a stylesheet is folded in because rules of equal weight have to be
	/// applied in the order they were written, while a program is a thing on
	/// its own that several documents may share and that reloads without any
	/// of them being rebuilt.
	///
	/// One name rather than a list, because one panel runs one program.
	/// Nothing in `colby_core` runs it - the host does, and drops everything
	/// the program made when either revision moves. @ref `colby_script`.
	pub program: String,
}

impl DocumentData {
	/// A document holding one empty root and no rules.
	#[must_use]
	pub fn empty() -> Self {
		Self {
			nodes: vec![Node {
				tag: "body".to_owned(),
				parent: NONE,
				first_child: NONE,
				next_sibling: NONE,
				..Node::default()
			}],
			rules: Vec::new(),
			program: String::new(),
		}
	}

	/// Whether there is anything to draw.
	#[must_use]
	pub fn is_empty(&self) -> bool {
		self.nodes
			.get(1)
			.is_none_or(|_| self.nodes.len() <= 1)
	}

	/// One node, by index.
	#[must_use]
	pub fn node(&self, index: u32) -> Option<&Node> {
		self.nodes.get(usize::try_from(index).ok()?)
	}

	/// The index of the node with an `id` attribute, if there is one.
	///
	/// A linear scan. A document is tens of nodes, a game looks a handful of
	/// them up per step, and a map would be a second thing to keep in step with
	/// the tree every time it is rebuilt.
	///
	/// @param id - the `id` attribute to look for
	#[must_use]
	pub fn find(&self, id: &str) -> Option<u32> {
		if id.is_empty() {
			return None;
		}

		self.nodes
			.iter()
			.position(|node| node.id == id)
			.and_then(|index| u32::try_from(index).ok())
	}

	/// Every child of a node, in order.
	#[must_use]
	pub fn children(&self, index: u32) -> Children<'_> {
		Children {
			document: self,
			next: self
				.node(index)
				.map_or(NONE, |node| node.first_child),
		}
	}

	/// Works out one node's style.
	///
	/// Rules first, most specific last, then the node's own `style` attribute.
	/// What the caller adds after that is whatever the game has bound.
	///
	/// @param index - the node
	/// @param inherited - the text properties handed down by its parent
	/// @param classes - its class list, which the game may have replaced
	/// @param hovered - whether the pointer is over it
	/// @return the whole computed style
	#[must_use]
	pub fn computed(&self, index: u32, inherited: &Style, classes: &str, hovered: bool) -> Style {
		let mut style = inherited.clone();

		let Some(node) = self.node(index) else {
			return style;
		};

		// gathered rather than sorted in place: rules are kept in source order
		// because that is the tie-break, and a stable sort by specificity is
		// what turns the two into one ordering.
		let mut claiming: Vec<(u32, &Rule)> = self
			.rules
			.iter()
			.filter(|rule| rule.selector.matches(node, classes, hovered))
			.map(|rule| (rule.selector.specificity(), rule))
			.collect();

		claiming.sort_by_key(|(specificity, _)| *specificity);

		for (_, rule) in claiming {
			style.merge(&rule.style);
		}

		style.merge(&node.inline);

		style
	}

	/// The font size a node's text is drawn at, in pixels.
	///
	/// A percentage is a share of the parent's size, which is the only place a
	/// percentage means something other than a share of a length.
	///
	/// @param style - the node's computed style
	/// @param parent - the size the parent's text is at
	#[must_use]
	pub fn font_size(style: &Style, parent: f32) -> f32 {
		style
			.font_size
			.and_then(|length| length.resolve(parent))
			.filter(|size| size.is_finite() && *size > 0.0)
			.unwrap_or(parent)
	}

	/// How round a node's corners are, in pixels.
	///
	/// @param style - the node's computed style
	/// @param box_size - the shorter side of the box, which a percentage is of
	#[must_use]
	pub fn radius(style: &Style, box_size: f32) -> f32 {
		style
			.radius
			.unwrap_or(Length::ZERO)
			.resolve(box_size)
			.unwrap_or(0.0)
			.clamp(0.0, box_size / 2.0)
	}
}

/// Walks a node's children.
pub struct Children<'a> {
	document: &'a DocumentData,
	next: u32,
}

impl Iterator for Children<'_> {
	type Item = u32;

	fn next(&mut self) -> Option<u32> {
		if self.next == NONE {
			return None;
		}

		let index = self.next;
		self.next = self
			.document
			.node(index)
			.map_or(NONE, |node| node.next_sibling);

		Some(index)
	}
}

#[cfg(test)]
mod tests {
	use super::{
		super::style::{Color, Display},
		*,
	};

	/// A root with two children, the second of which holds a run of text.
	fn document() -> DocumentData {
		let mut data = DocumentData::empty();

		data.nodes.push(Node {
			tag: "div".to_owned(),
			id: "left".to_owned(),
			classes: "panel".to_owned(),
			parent: ROOT,
			first_child: NONE,
			next_sibling: 2,
			..Node::default()
		});
		data.nodes.push(Node {
			tag: "div".to_owned(),
			id: "right".to_owned(),
			classes: "panel wide".to_owned(),
			parent: ROOT,
			first_child: 3,
			next_sibling: NONE,
			..Node::default()
		});
		data.nodes.push(Node {
			tag: Node::TEXT.to_owned(),
			text: "hello".to_owned(),
			parent: 2,
			first_child: NONE,
			next_sibling: NONE,
			..Node::default()
		});

		if let Some(root) = data.nodes.first_mut() {
			root.first_child = 1;
		}

		data
	}

	/// A rule that sets one width.
	fn rule(selector: Selector, width: f32) -> Rule {
		Rule {
			selector,
			style: Style {
				width: Some(Length::Px(width)),
				..Style::default()
			},
		}
	}

	#[test]
	fn a_node_is_found_by_the_identifier_a_game_would_use() {
		let data = document();

		assert_eq!(data.find("right"), Some(2), "the one that was asked for");
		assert_eq!(data.find("missing"), None, "and nothing for one that is not there");
		assert_eq!(data.find(""), None, "the empty identifier is not an identifier");
	}

	#[test]
	fn children_come_back_in_the_order_they_were_written() {
		let data = document();

		assert_eq!(data.children(ROOT).collect::<Vec<_>>(), vec![1, 2], "both, in order");
		assert_eq!(data.children(2).collect::<Vec<_>>(), vec![3], "and the text under the right");
		assert_eq!(data.children(1).count(), 0, "the left one has none");
	}

	#[test]
	fn a_selector_with_nothing_in_it_claims_everything() {
		let data = document();
		let node = data.node(1).expect("it is there");

		assert!(
			Selector::default().matches(node, &node.classes, false),
			"which is what makes a bare `*`-shaped rule the base of a stylesheet"
		);
	}

	#[test]
	fn every_part_of_a_selector_has_to_agree() {
		let data = document();
		let node = data.node(2).expect("it is there");
		let claims = |selector: Selector| selector.matches(node, &node.classes, false);

		assert!(
			claims(Selector {
				tag: "div".to_owned(),
				..Selector::default()
			}),
			"by tag"
		);
		assert!(
			claims(Selector {
				class: "wide".to_owned(),
				..Selector::default()
			}),
			"by class"
		);
		assert!(
			claims(Selector {
				id: "right".to_owned(),
				..Selector::default()
			}),
			"by id"
		);
		assert!(
			!claims(Selector {
				tag: "span".to_owned(),
				class: "wide".to_owned(),
				..Selector::default()
			}),
			"and one part disagreeing is the whole selector disagreeing"
		);
	}

	#[test]
	fn a_hover_rule_waits_for_the_pointer() {
		let data = document();
		let node = data.node(1).expect("it is there");
		let selector = Selector {
			class: "panel".to_owned(),
			hover: true,
			..Selector::default()
		};

		assert!(!selector.matches(node, &node.classes, false), "not while the pointer is away");
		assert!(selector.matches(node, &node.classes, true), "and yes once it arrives");
	}

	#[test]
	fn the_more_specific_rule_wins_however_they_were_ordered() {
		let mut data = document();
		data.rules.push(rule(
			Selector {
				id: "left".to_owned(),
				..Selector::default()
			},
			300.0,
		));
		data.rules.push(rule(
			Selector {
				class: "panel".to_owned(),
				..Selector::default()
			},
			100.0,
		));

		let style = data.computed(1, &Style::default(), "panel", false);

		assert_eq!(
			style.width,
			Some(Length::Px(300.0)),
			"the identifier beats the class even though the class came second"
		);
	}

	#[test]
	fn two_rules_of_the_same_weight_go_in_the_order_they_were_written() {
		let mut data = document();
		data.rules.push(rule(
			Selector {
				class: "panel".to_owned(),
				..Selector::default()
			},
			100.0,
		));
		data.rules.push(rule(
			Selector {
				class: "wide".to_owned(),
				..Selector::default()
			},
			400.0,
		));

		let style = data.computed(2, &Style::default(), "panel wide", false);

		assert_eq!(style.width, Some(Length::Px(400.0)), "the later of two equals");
	}

	#[test]
	fn the_style_attribute_beats_every_rule() {
		let mut data = document();
		data.rules.push(rule(
			Selector {
				id: "left".to_owned(),
				..Selector::default()
			},
			300.0,
		));

		if let Some(node) = data.nodes.get_mut(1) {
			node.inline.width = Some(Length::Px(7.0));
		}

		let style = data.computed(1, &Style::default(), "panel", false);

		assert_eq!(style.width, Some(Length::Px(7.0)), "written on the box itself, so it wins");
	}

	#[test]
	fn a_class_the_game_replaced_is_the_one_that_is_matched() {
		let mut data = document();
		data.rules.push(rule(
			Selector {
				class: "low".to_owned(),
				..Selector::default()
			},
			42.0,
		));

		let style = data.computed(1, &Style::default(), "panel low", false);

		assert_eq!(
			style.width,
			Some(Length::Px(42.0)),
			"swapping a class is how a game changes how something looks, so the bound list has \
			 to be what the cascade reads"
		);
	}

	#[test]
	fn inherited_properties_arrive_before_any_rule_is_applied() {
		let data = document();
		let inherited = Style {
			color: Some(Color::WHITE),
			..Style::default()
		};

		let style = data.computed(3, &inherited, "", false);

		assert_eq!(style.color, Some(Color::WHITE), "the text is its parent's color");
	}

	#[test]
	fn a_font_size_in_percent_is_a_share_of_the_parent_text() {
		let half = Style {
			font_size: Some(Length::Percent(50.0)),
			..Style::default()
		};

		assert!((DocumentData::font_size(&half, 20.0) - 10.0).abs() < 1.0e-5, "half of twenty");
		assert!(
			(DocumentData::font_size(&Style::default(), 20.0) - 20.0).abs() < 1.0e-5,
			"and saying nothing is the parent's size"
		);
	}

	#[test]
	fn a_corner_radius_never_exceeds_half_the_box() {
		let round = Style {
			radius: Some(Length::Px(400.0)),
			..Style::default()
		};

		assert!(
			(DocumentData::radius(&round, 40.0) - 20.0).abs() < 1.0e-5,
			"a radius past half the box turns the shape inside out in the shader"
		);
	}

	#[test]
	fn a_hidden_box_says_so_through_its_style() {
		let hidden = Style {
			display: Some(Display::None),
			..Style::default()
		};

		assert!(!hidden.is_shown(), "and the layout skips it entirely");
	}
}
