//! The game's interface: documents from the asset tree, and the state a game
//! puts into them.
//!
//! Three decisions are written into the shape of this module, and each was
//! taken against a plausible alternative.
//!
//! **A document is an asset, not code.** `assets/ui/hud.html` compiles like
//! `assets/meshes/crystal.obj` does, lands in a registry beside the meshes and
//! the textures, and reloads the same way - edit the file and the interface
//! changes in a running window, with no module swap and no restart. The two
//! alternatives were building the tree from `init` every reload, or keeping it
//! in the game's arena; neither of them gives that, and it is the whole reason
//! this engine has an asset pipeline at all.
//!
//! **A game addresses a node by its `id`, not by a handle.** Recompiling a
//! document rebuilds its nodes, so an index into them is a handle that goes
//! stale exactly when a person edits the file they are looking at - the one
//! moment it must not. A name survives that, and `set_text(hud, "score", ..)`
//! is what the code would have said anyway.
//!
//! **Events are a queue the game drains, not callbacks it registers.** A
//! callback would be a function pointer with the module's lifetime, which is a
//! problem already solved once for console commands and would have to be solved
//! again here. The deciding argument is the other one: hit-testing follows the
//! pointer, so it happens once a *frame*, and a callback fired from it would
//! run gameplay code at the frame rate rather than the step rate. The host
//! tests and queues; the step drains. @ref [`Ui::end_step`].

use crate::{
	abi::registry::{Entry, Registry},
	glam::Vec2,
	registry_handle,
};

pub mod document;
pub mod style;

pub use self::{
	document::{DocumentData, Node, Rule, Selector},
	style::{Align, Color, Direction, Display, Edges, Justify, Length, Position, Style, Wrap},
};

registry_handle! {
	/// Which document in [`Documents`].
	///
	/// Not generational, like every other resource handle: recompiling a
	/// document rewrites the entry the handle already points at, and the
	/// interface notices because the entry's revision moved.
	DocumentId
}

registry_handle! {
	/// One document a game has put on screen, with whatever it has bound into
	/// it.
	///
	/// Slot zero is [`PanelId::NONE`] and is a panel showing nothing, so that
	/// writing through a handle that never resolved changes nothing rather than
	/// having to be checked at every call site.
	PanelId
}

/// One entry of the document registry.
pub type Document = Entry<DocumentData>;

/// What happened to a node.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EventKind {
	/// A button went down over it.
	#[default]
	Press,

	/// A button came up over it.
	Release,

	/// A button went down and came up over the same node.
	Click,
}

/// One thing that happened in the interface during a step.
///
/// The node is named rather than indexed, for the same reason a game names one:
/// a document that was recompiled between the press and the release has new
/// indices and the same names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
	/// Which panel it happened in.
	pub panel: PanelId,

	/// The `id` of the node it happened to. Empty for a node with no `id`,
	/// which is a node no game asked about.
	pub node: String,

	/// What happened.
	pub kind: EventKind,
}

/// What a game has written into one node.
///
/// Everything here is an override of what the document says: an unset field
/// means "whatever is in the file". That is what lets a document be reloaded
/// under a running game without throwing away the score it was displaying.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bind {
	/// The `id` of the node this applies to.
	pub node: String,

	/// The words to draw instead of the ones in the file.
	pub text: Option<String>,

	/// The class list to match instead of the one in the file.
	pub classes: Option<String>,

	/// Properties to apply after everything else, the way a `style` attribute
	/// would. Where a health bar's width comes from.
	pub style: Style,
}

/// How far one box has been scrolled.
///
/// Keyed by `id` like a [`Bind`] and kept beside them rather than among them,
/// for two reasons. Every field of a `Bind` is an `Option` meaning "the game
/// said nothing about this", and a scroll of zero is a value rather than an
/// absence. And a bind is what the *game* wrote while this is what a *person*
/// did, which is a difference worth being able to see in a debugger.
///
/// Keyed by `id` rather than by node index because it has to survive the
/// document being recompiled under a running process - which is the one thing
/// this engine exists to do, and the moment somebody is most likely to be
/// looking at a scrolled list. A box with no `id` cannot keep a scroll
/// position, exactly as it cannot be given text.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scroll {
	/// The `id` of the node this is about.
	pub node: String,

	/// How far down its contents have been moved, in layout pixels.
	///
	/// Never negative, and never past the end of what is inside: the layout
	/// measures both every step and clamps this to what it found.
	pub offset: f32,
}

/// One document on screen.
#[derive(Clone, Debug)]
pub struct Panel {
	document: DocumentId,
	shown: bool,
	binds: Vec<Bind>,
	/// How far each scrollable node has been scrolled. Host-written.
	scrolls: Vec<Scroll>,
	/// The node the pointer is over, or [`document::NONE`]. Host-written, once
	/// a frame, from the layout.
	hovered: u32,
	/// The node a button went down on, or [`document::NONE`]. Host-written.
	pressed: u32,
	/// The `id` of the field the keyboard is going to, or empty. Host-written.
	///
	/// By `id` rather than by index, for the reason a scroll offset is: a
	/// document recompiled under a running process renumbers its boxes, and
	/// somebody halfway through typing is exactly who should not lose their
	/// place for it.
	focused: String,
	/// Where the caret sits in the focused field, as a byte offset into its
	/// value. Host-written, and meaningless when nothing is focused.
	caret: u32,
}

impl Panel {
	/// Which document it shows.
	#[must_use]
	pub const fn document(&self) -> DocumentId { self.document }

	/// Whether it is on screen.
	#[must_use]
	pub const fn is_shown(&self) -> bool { self.shown }

	/// The node the pointer is over, or [`document::NONE`].
	#[must_use]
	pub const fn hovered(&self) -> u32 { self.hovered }

	/// The node a button is being held on, or [`document::NONE`].
	#[must_use]
	pub const fn pressed(&self) -> u32 { self.pressed }

	/// What the game has bound to a node, if anything.
	#[must_use]
	pub fn bind(&self, node: &str) -> Option<&Bind> {
		if node.is_empty() {
			return None;
		}

		self.binds.iter().find(|bind| bind.node == node)
	}

	/// The class list the cascade should match against for a node.
	///
	/// The game's, if it has replaced it, and the document's otherwise.
	#[must_use]
	pub fn classes<'a>(&'a self, node: &'a Node) -> &'a str {
		self.bind(&node.id)
			.and_then(|bind| bind.classes.as_deref())
			.unwrap_or(&node.classes)
	}

	/// The words a node should draw.
	#[must_use]
	pub fn text<'a>(&'a self, node: &'a Node) -> &'a str {
		self.bind(&node.id)
			.and_then(|bind| bind.text.as_deref())
			.unwrap_or(&node.text)
	}

	/// The `id` of the field the keyboard is going to, or empty.
	#[must_use]
	pub fn focused(&self) -> &str { &self.focused }

	/// Where the caret sits in that field, as a byte offset.
	#[must_use]
	pub const fn caret(&self) -> u32 { self.caret }

	/// How far a node has been scrolled, or zero if it never has been.
	///
	/// @param node - the `id` to look for
	#[must_use]
	pub fn scroll(&self, node: &str) -> f32 {
		if node.is_empty() {
			return 0.0;
		}

		self.scrolls
			.iter()
			.find(|scroll| scroll.node == node)
			.map_or(0.0, |scroll| scroll.offset)
	}

	/// Moves a node's contents, making an entry if there is none.
	///
	/// The host's. Nothing here knows how far is too far - that is the
	/// layout's, which measures what is inside the box every step.
	///
	/// @param node - the `id` to move; an empty one is ignored, because there
	/// would be nothing to find the entry by again
	/// @param offset - how far down, in layout pixels
	pub fn set_scroll(&mut self, node: &str, offset: f32) {
		if node.is_empty() {
			return;
		}

		if let Some(scroll) = self
			.scrolls
			.iter_mut()
			.find(|scroll| scroll.node == node)
		{
			scroll.offset = offset;

			return;
		}

		self.scrolls.push(Scroll {
			// copied rather than borrowed, for the reason a bind's name is.
			node: node.to_owned(),
			offset,
		});
	}

	/// What the game has bound to a node, making an entry if there is none.
	fn bind_mut(&mut self, node: &str) -> &mut Bind {
		let known = self
			.binds
			.iter()
			.position(|bind| bind.node == node);

		let index = known.unwrap_or_else(|| {
			self.binds.push(Bind {
				// copied, never borrowed: this string may have come from the
				// game module, and the host outlives it. Same rule the console
				// table follows. @ref [`cvar`](crate::abi::cvar).
				node: node.to_owned(),
				..Bind::default()
			});

			self.binds.len() - 1
		});

		self.binds
			.get_mut(index)
			.unwrap_or_else(|| unreachable!("the entry was just found or just pushed"))
	}
}

/// Every interface document in the process, and what is on screen.
///
/// Held by [`World`](crate::abi::World), like the other tables.
#[derive(Clone, Debug)]
pub struct Ui {
	documents: Registry<DocumentData>,
	panels: Vec<Panel>,
	events: Vec<Event>,
	pointer: Vec2,
	viewport: Vec2,
	scale: f32,
	over: bool,
}

impl Ui {
	/// An interface with nothing in it.
	#[must_use]
	pub fn new() -> Self {
		Self {
			documents: Registry::new(DocumentData::empty()),
			panels: vec![Panel {
				document: DocumentId::NONE,
				shown: false,
				binds: Vec::new(),
				scrolls: Vec::new(),
				hovered: document::NONE,
				pressed: document::NONE,
				focused: String::new(),
				caret: 0,
			}],
			events: Vec::new(),
			pointer: Vec2::new(-1.0, -1.0),
			viewport: Vec2::new(1.0, 1.0),
			scale: 1.0,
			over: false,
		}
	}

	//
	// what a game calls
	//

	/// Puts a document on screen, or finds the one that already is.
	///
	/// Idempotent by name, like registering a material: `init` runs again on
	/// every reload, and a game that showed its interface there should find the
	/// same panel with the same text still in it rather than a second copy.
	///
	/// @param name - the document's asset name, e.g. `ui/hud`
	/// @return its panel, or [`PanelId::NONE`] if no such document is loaded
	pub fn show(&mut self, name: &str) -> PanelId {
		let document = DocumentId::new(self.documents.find(name));
		if !document.is_some() {
			return PanelId::NONE;
		}

		if let Some(index) = self
			.panels
			.iter()
			.position(|panel| panel.document == document)
		{
			if let Some(panel) = self.panels.get_mut(index) {
				panel.shown = true;
			}

			return PanelId::new(u32::try_from(index).unwrap_or(0));
		}

		self.panels.push(Panel {
			document,
			shown: true,
			binds: Vec::new(),
			scrolls: Vec::new(),
			hovered: document::NONE,
			pressed: document::NONE,
			focused: String::new(),
			caret: 0,
		});

		PanelId::new(u32::try_from(self.panels.len() - 1).unwrap_or(0))
	}

	/// Takes a panel off screen, keeping everything bound into it.
	pub fn hide(&mut self, panel: PanelId) {
		if let Some(panel) = self.panel_mut(panel) {
			panel.shown = false;
			panel.hovered = document::NONE;
			panel.pressed = document::NONE;
		}
	}

	/// Whether a panel is on screen.
	#[must_use]
	pub fn is_shown(&self, panel: PanelId) -> bool {
		self.panel(panel).is_some_and(Panel::is_shown)
	}

	/// Replaces the words a node draws.
	///
	/// @param panel - the panel it is in
	/// @param node - the node's `id`
	/// @param text - what to draw
	pub fn set_text(&mut self, panel: PanelId, node: &str, text: &str) {
		if node.is_empty() {
			return;
		}

		let Some(panel) = self.panel_mut(panel) else {
			return;
		};

		let bind = panel.bind_mut(node);
		match &mut bind.text {
			// written in place, because this is called every step with a
			// string that is usually the same length as the last one.
			| Some(held) => text.clone_into(held),
			| None => bind.text = Some(text.to_owned()),
		}
	}

	/// Replaces the class list a node is matched by.
	///
	/// How a game changes the way something looks without knowing what it looks
	/// like: the stylesheet decides what `.low` means, and the game decides
	/// when the bar is low.
	pub fn set_classes(&mut self, panel: PanelId, node: &str, classes: &str) {
		if node.is_empty() {
			return;
		}

		let Some(panel) = self.panel_mut(panel) else {
			return;
		};

		let bind = panel.bind_mut(node);
		match &mut bind.classes {
			| Some(held) => classes.clone_into(held),
			| None => bind.classes = Some(classes.to_owned()),
		}
	}

	/// Properties to put on a node, after everything the document says.
	///
	/// The continuous half of the interface: a class cannot be a bar that is
	/// thirty-seven percent full, and this can.
	///
	/// @return the override, or `None` for a panel that does not resolve
	pub fn style_mut(&mut self, panel: PanelId, node: &str) -> Option<&mut Style> {
		if node.is_empty() {
			return None;
		}

		Some(&mut self.panel_mut(panel)?.bind_mut(node).style)
	}

	/// Whether the pointer is over something the interface painted.
	///
	/// What a game checks before acting on a click of its own: a button press
	/// that landed on a panel is not a button press that should also swing the
	/// camera. The same rule the host applies between the editor and the game,
	/// one layer down.
	///
	/// A box that paints nothing does not count, which is what keeps the
	/// document's root - a transparent box the size of the window - from
	/// swallowing every click in the game.
	#[must_use]
	pub const fn pointer_over(&self) -> bool { self.over }

	/// Everything that happened in the interface this step.
	#[must_use]
	pub fn events(&self) -> &[Event] { &self.events }

	/// Whether a node was clicked this step.
	///
	/// The form a game actually wants: one `if` per button, and no matching on
	/// an event type nobody else uses.
	#[must_use]
	pub fn clicked(&self, panel: PanelId, node: &str) -> bool {
		self.events.iter().any(|event| {
			event.panel == panel && event.kind == EventKind::Click && event.node == node
		})
	}

	//
	// what the host calls
	//

	/// The document table, to look a name up in.
	#[must_use]
	pub fn find(&self, name: &str) -> DocumentId { DocumentId::new(self.documents.find(name)) }

	/// Registers a parsed document under a name, replacing whatever was there.
	///
	/// @return the handle, the same one as last time if the name is known
	pub fn insert(&mut self, name: &str, data: DocumentData) -> DocumentId {
		DocumentId::new(self.documents.insert(name, data))
	}

	/// One document, by handle.
	#[must_use]
	pub fn document(&self, id: DocumentId) -> Option<&Document> {
		self.documents.entry(id.index())
	}

	/// How many documents are loaded, counting the empty one.
	#[must_use]
	pub fn len(&self) -> usize { self.documents.len() }

	/// Always `false`: slot zero always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.documents.is_empty() }

	/// Every panel and its handle, in the order they were first shown.
	///
	/// Which is also the order they are drawn in: later panels are on top.
	pub fn panels(&self) -> impl Iterator<Item = (PanelId, &Panel)> {
		self.panels
			.iter()
			.enumerate()
			.skip(1)
			.map(|(index, panel)| (PanelId::new(u32::try_from(index).unwrap_or(0)), panel))
	}

	/// One panel, by handle.
	#[must_use]
	pub fn panel(&self, panel: PanelId) -> Option<&Panel> {
		if !panel.is_some() {
			return None;
		}

		self.panels.get(panel.slot())
	}

	/// Records which node the pointer is over.
	///
	/// The host's to call, from the layout, before the step runs.
	pub fn set_hovered(&mut self, panel: PanelId, node: u32) {
		if let Some(panel) = self.panel_mut(panel) {
			panel.hovered = node;
		}
	}

	/// How far a node of a panel has been scrolled.
	///
	/// @param panel - which document
	/// @param node - the `id` of the box
	#[must_use]
	pub fn scroll(&self, panel: PanelId, node: &str) -> f32 {
		self.panel(panel)
			.map_or(0.0, |panel| panel.scroll(node))
	}

	/// Moves a node's contents.
	///
	/// @param panel - which document
	/// @param node - the `id` of the box
	/// @param offset - how far down, in layout pixels
	pub fn set_scroll(&mut self, panel: PanelId, node: &str, offset: f32) {
		if let Some(panel) = self.panel_mut(panel) {
			panel.set_scroll(node, offset);
		}
	}

	/// The `id` of the field the keyboard is going to in a panel.
	#[must_use]
	pub fn focused(&self, panel: PanelId) -> &str { self.panel(panel).map_or("", Panel::focused) }

	/// Where the caret sits in that field, as a byte offset into its value.
	#[must_use]
	pub fn caret(&self, panel: PanelId) -> u32 { self.panel(panel).map_or(0, Panel::caret) }

	/// Sends the keyboard to a field, or to nothing.
	///
	/// The host's, and a game's when it wants a search box ready to type in
	/// without a click first.
	///
	/// @param panel - which document
	/// @param node - the `id` of the field, or empty to focus nothing
	/// @param caret - where to put the caret, as a byte offset into the value
	pub fn set_focus(&mut self, panel: PanelId, node: &str, caret: u32) {
		if let Some(panel) = self.panel_mut(panel) {
			// copied rather than borrowed, for the reason a bind's name is.
			node.clone_into(&mut panel.focused);
			panel.caret = caret;
		}
	}

	/// What a node says now.
	///
	/// What the game last wrote, or what somebody has typed into it, or failing
	/// both what the document says - which is one question with one answer, and
	/// the same one a browser gives for an input's `value`. The document is the
	/// default underneath rather than a separate thing to ask about.
	///
	/// @param panel - which document
	/// @param node - the `id` of the box
	#[must_use]
	pub fn text(&self, panel: PanelId, node: &str) -> &str {
		let Some(found) = self.panel(panel) else {
			return "";
		};

		if let Some(bound) = found
			.bind(node)
			.and_then(|bind| bind.text.as_deref())
		{
			return bound;
		}

		self.document(found.document())
			.map(Entry::value)
			.and_then(|document| {
				let index = document.find(node)?;

				document.node(index)
			})
			.map_or("", |node| node.text.as_str())
	}

	/// Records which node a button is being held on.
	pub fn set_pressed(&mut self, panel: PanelId, node: u32) {
		if let Some(panel) = self.panel_mut(panel) {
			panel.pressed = node;
		}
	}

	/// Records whether the pointer is over anything the interface painted.
	///
	/// The host's to call, from the hit test, once a step.
	pub const fn set_pointer_over(&mut self, over: bool) { self.over = over; }

	/// Adds something that happened, for the step to find.
	pub fn push_event(&mut self, event: Event) { self.events.push(event); }

	/// Drops the events of the step that has just finished.
	///
	/// Called by the host at the end of a step, exactly like the edges in
	/// [`Input`](crate::abi::Input): a click is a thing that happened once, and
	/// a second step in the same frame must not see it again.
	pub fn end_step(&mut self) { self.events.clear(); }

	/// Where the pointer is, in layout pixels.
	#[must_use]
	pub const fn pointer(&self) -> Vec2 { self.pointer }

	/// The area documents are laid out in, in layout pixels.
	#[must_use]
	pub const fn viewport(&self) -> Vec2 { self.viewport }

	/// How many physical pixels one layout pixel is.
	///
	/// One on a normal display and one and a half on a scaled one, so that an
	/// interface written in pixels is the same size on both. Screenshots are
	/// taken at one, which is what keeps them comparable across machines.
	#[must_use]
	pub const fn scale(&self) -> f32 { self.scale }

	/// Tells the interface how big the window is and how dense its pixels are.
	///
	/// @param physical - the surface size in physical pixels
	/// @param scale - physical pixels per layout pixel
	pub fn set_viewport(&mut self, physical: Vec2, scale: f32) {
		self.scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
		self.viewport = (physical / self.scale).max(Vec2::ONE);
	}

	/// Tells the interface where the pointer is.
	///
	/// @param physical - the cursor in physical pixels, origin top left
	pub fn set_pointer(&mut self, physical: Vec2) { self.pointer = physical / self.scale; }

	/// One panel, by handle, to write to.
	fn panel_mut(&mut self, panel: PanelId) -> Option<&mut Panel> {
		if !panel.is_some() {
			return None;
		}

		self.panels.get_mut(panel.slot())
	}
}

impl Default for Ui {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::*;

	/// An interface with one document registered under a name.
	fn loaded() -> Ui {
		let mut ui = Ui::new();
		ui.insert("ui/hud", DocumentData::empty());

		ui
	}

	#[test]
	fn a_panel_for_a_document_nobody_loaded_resolves_to_nothing() {
		let mut ui = Ui::new();

		assert_eq!(ui.show("ui/missing"), PanelId::NONE, "there is no such document");
		assert!(!ui.is_shown(PanelId::NONE), "and the null panel is never on screen");
	}

	#[test]
	fn showing_the_same_document_twice_is_the_same_panel() {
		let mut ui = loaded();
		let first = ui.show("ui/hud");
		let second = ui.show("ui/hud");

		assert!(first.is_some(), "it resolved");
		assert_eq!(first, second, "init runs again on every reload, and must not stack panels");
		assert_eq!(ui.panels().count(), 1, "so there is still only the one");
	}

	#[test]
	fn hiding_a_panel_keeps_what_was_bound_into_it() {
		let mut ui = loaded();
		let panel = ui.show("ui/hud");
		ui.set_text(panel, "score", "1200");
		ui.hide(panel);

		assert!(!ui.is_shown(panel), "it is off screen");

		let shown = ui.show("ui/hud");
		assert_eq!(
			ui.panel(shown)
				.and_then(|it| it.bind("score"))
				.and_then(|bind| bind.text.as_deref()),
			Some("1200"),
			"and comes back saying what it said before"
		);
	}

	#[test]
	fn writing_the_same_node_twice_replaces_rather_than_stacks() {
		let mut ui = loaded();
		let panel = ui.show("ui/hud");

		ui.set_text(panel, "score", "1");
		ui.set_text(panel, "score", "2");

		let binds = ui.panel(panel).map_or(0, |it| it.binds.len());

		assert_eq!(binds, 1, "this is called every step; one entry per node, not one per call");
		assert_eq!(
			ui.panel(panel)
				.and_then(|it| it.bind("score"))
				.and_then(|bind| bind.text.as_deref()),
			Some("2"),
			"and it holds the latest"
		);
	}

	#[test]
	fn a_node_with_no_identifier_cannot_be_bound_to() {
		let mut ui = loaded();
		let panel = ui.show("ui/hud");

		ui.set_text(panel, "", "nowhere");

		assert_eq!(
			ui.panel(panel).map_or(0, |it| it.binds.len()),
			0,
			"an empty name would collect every unnamed box in the document into one entry"
		);
		assert!(ui.style_mut(panel, "").is_none(), "and the same for a style override");
	}

	#[test]
	fn a_bound_class_list_is_what_the_cascade_should_read() {
		let mut ui = loaded();
		let panel = ui.show("ui/hud");
		let node = Node {
			id: "bar".to_owned(),
			classes: "bar".to_owned(),
			..Node::default()
		};

		assert_eq!(
			ui.panel(panel).map(|it| it.classes(&node)),
			Some("bar"),
			"the document's, until the game says otherwise"
		);

		ui.set_classes(panel, "bar", "bar low");

		assert_eq!(
			ui.panel(panel).map(|it| it.classes(&node)),
			Some("bar low"),
			"and the game's afterwards"
		);
	}

	#[test]
	fn a_style_override_is_where_a_continuous_value_goes() {
		let mut ui = loaded();
		let panel = ui.show("ui/hud");

		if let Some(style) = ui.style_mut(panel, "bar") {
			style.width = Some(Length::Percent(37.0));
		}

		assert_eq!(
			ui.panel(panel)
				.and_then(|it| it.bind("bar"))
				.map(|bind| bind.style.width),
			Some(Some(Length::Percent(37.0))),
			"a class cannot be thirty-seven percent, and this can"
		);
	}

	#[test]
	fn an_event_lives_exactly_one_step() {
		let mut ui = loaded();
		let panel = ui.show("ui/hud");

		ui.push_event(Event {
			panel,
			node: "start".to_owned(),
			kind: EventKind::Click,
		});

		assert!(ui.clicked(panel, "start"), "the step it happened in sees it");

		ui.end_step();

		assert!(
			!ui.clicked(panel, "start"),
			"and a second step in the same frame must not see one click as two"
		);
	}

	#[test]
	fn a_click_belongs_to_the_panel_it_happened_in() {
		let mut ui = loaded();
		ui.insert("ui/menu", DocumentData::empty());
		let hud = ui.show("ui/hud");
		let menu = ui.show("ui/menu");

		ui.push_event(Event {
			panel: menu,
			node: "start".to_owned(),
			kind: EventKind::Click,
		});

		assert!(ui.clicked(menu, "start"), "in the panel it happened in");
		assert!(!ui.clicked(hud, "start"), "and not in one that happens to have the same names");
	}

	#[test]
	fn the_layout_viewport_is_the_window_divided_by_the_scale() {
		let mut ui = Ui::new();
		ui.set_viewport(Vec2::new(2560.0, 1440.0), 2.0);

		assert!(
			ui.viewport()
				.abs_diff_eq(Vec2::new(1280.0, 720.0), 1.0e-4),
			"an interface written in pixels should be the same size on a scaled display"
		);

		ui.set_pointer(Vec2::new(1280.0, 720.0));
		assert!(
			ui.pointer()
				.abs_diff_eq(Vec2::new(640.0, 360.0), 1.0e-4),
			"and the pointer has to arrive in the same space the boxes are in"
		);
	}

	#[test]
	fn a_scale_of_nothing_is_refused_rather_than_dividing_by_it() {
		let mut ui = Ui::new();
		ui.set_viewport(Vec2::new(1280.0, 720.0), 0.0);

		assert!((ui.scale() - 1.0).abs() < f32::EPSILON, "zero is not a scale");
		assert!(ui.viewport().x > 0.0, "and the viewport survived it");
	}

	#[test]
	fn panels_come_back_in_the_order_they_were_shown() {
		let mut ui = loaded();
		ui.insert("ui/menu", DocumentData::empty());
		let hud = ui.show("ui/hud");
		let menu = ui.show("ui/menu");

		assert_eq!(
			ui.panels().map(|(id, _)| id).collect::<Vec<_>>(),
			vec![hud, menu],
			"which is also the order they are drawn in, so the menu is over the hud"
		);
	}
}
