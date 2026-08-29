//! The game's interface: HTML and CSS over flexbox, drawn over the scene.
//!
//! Engine-side, not gameplay-side - editing this is a restart, the same as
//! editing the renderer. It is its own crate so that taffy stays out of
//! `colby_engine` and so that the interface can be laid out and hit-tested with
//! no GPU at all, which is what makes most of it testable.
//!
//! **This is not the editor.** `colby_editor` is egui, it is a tool, and a game
//! never looks at it; this is what a game draws its own interface with. The two
//! share exactly one thing, the [`Overlay`] seam the renderer calls between the
//! scene and the present, and nothing else.
//!
//! The work is split in three, and the split is the reason any of it can be
//! checked without a window:
//!
//! - [`layout`] turns a document into boxes with positions on them, calling
//!   into taffy for the flexbox and into the font metrics to measure text.
//! - [`draw`] turns those boxes into triangles. Plain data, no GPU.
//! - [`paint`] puts the triangles on screen.
//!
//! **Hit-testing happens in the simulation step, not in the frame.** That is
//! the one decision here worth arguing about, and it goes this way because the
//! alternative - a callback fired while the pointer is being tested - would run
//! gameplay code at whatever rate the frames happen to be arriving at. What the
//! host does instead is test, queue an event, and let the step drain it. @ref
//! [`colby_core::abi::ui`].

use colby_core::{
	Result,
	abi::{
		Button, Input, Key, World,
		ui::{
			Event, EventKind, PanelId,
			document::{NONE, Node},
		},
	},
	glam::Vec2,
	trace,
};
use colby_engine::{
	Overlay,
	wgpu::{Device, Queue, TextureFormat, TextureView},
};

pub mod draw;
pub mod layout;
pub mod paint;
pub mod world_text;

#[cfg(test)]
mod pixels;

pub use self::{
	draw::{Binding, DrawList, Vertex},
	layout::Placed,
	paint::Painter,
};

/// The nearest character boundary at or before a byte offset.
///
/// @param value - the string the offset is into
/// @param at - the offset, which may be past the end or inside a character
fn boundary(value: &str, at: usize) -> usize {
	let mut at = at.min(value.len());

	// zero is always a boundary, so this stops.
	while !value.is_char_boundary(at) {
		at -= 1;
	}

	at
}

/// One step's typing and editing, applied to a value and a caret.
///
/// A free function because it is all of the interesting part and none of it
/// needs a world: a string, a place in it, and what the keyboard did.
///
/// @param input - this step's keys and text
/// @param value - the field's contents, edited in place
/// @param caret - a byte offset into it, moved in place
fn edit(input: &Input, value: &mut String, caret: &mut usize) {
	// snapped first, because the value can have been replaced under the caret:
	// a game calling `set_text` on the field somebody is typing in leaves an
	// offset that means nothing, and half of a character is not a place.
	*caret = boundary(value, *caret);

	let typed = input.typed();
	if !typed.is_empty() {
		value.insert_str(*caret, typed);
		*caret += typed.len();
	}

	if input.pressed(Key::Backspace)
		&& let Some(before) = value
			.get(..*caret)
			.and_then(|it| it.chars().next_back())
	{
		*caret -= before.len_utf8();
		value.remove(*caret);
	}

	if input.pressed(Key::Delete)
		&& value
			.get(*caret..)
			.is_some_and(|it| it.chars().next().is_some())
	{
		value.remove(*caret);
	}

	if input.pressed(Key::Left)
		&& let Some(before) = value
			.get(..*caret)
			.and_then(|it| it.chars().next_back())
	{
		*caret -= before.len_utf8();
	}

	if input.pressed(Key::Right)
		&& let Some(after) = value
			.get(*caret..)
			.and_then(|it| it.chars().next())
	{
		*caret += after.len_utf8();
	}

	if input.pressed(Key::Home) {
		*caret = 0;
	}

	if input.pressed(Key::End) {
		*caret = value.len();
	}
}

/// How far one line of the wheel scrolls, in layout pixels.
///
/// A constant rather than something a stylesheet says, because it is a
/// property of the wheel rather than of the box: two lists side by side
/// should move by the same amount for the same flick.
const SCROLL_LINE: f32 = 40.0;

/// The whole interface: what is on screen, and what it takes to draw it.
#[derive(Default)]
pub struct Interface {
	placed: Vec<Placed>,
	list: DrawList,
	painter: Option<Painter>,
	viewport: [f32; 2],
}

impl Interface {
	/// An interface with nothing laid out and no GPU behind it.
	///
	/// Useful on its own: layout, hit-testing and the draw list all work
	/// without a device, which is what lets a test check what a document turns
	/// into instead of only what it looks like.
	#[must_use]
	pub fn new() -> Self { Self::default() }

	/// Gives it a pipeline to draw with.
	///
	/// @param device - the device the frames belong to
	/// @param format - the color format the target was configured with
	pub fn attach(&mut self, device: &Device, format: TextureFormat) -> Result<()> {
		self.painter = Some(Painter::new(device, format)?);

		Ok(())
	}

	/// Whether there is a pipeline behind it.
	#[must_use]
	pub const fn is_attached(&self) -> bool { self.painter.is_some() }

	/// Lays the interface out, tests the pointer against it and queues events.
	///
	/// Called by the host **inside the step**, before the game's `update`, so
	/// that a click is something the game finds already waiting for it rather
	/// than something that happens to it halfway through a frame.
	///
	/// @param world - the documents to lay out and the events to queue into
	pub fn update(&mut self, world: &mut World) {
		layout::run(world, &mut self.placed);

		let pointer = world.ui.pointer();
		let hit = self
			.placed
			.iter()
			.rev()
			.find(|placed| paints(placed) && placed.contains(pointer))
			.map(|placed| (placed.panel, placed.node));

		// what the game asks before it acts on the same click. A document's root
		// is a transparent box the size of the window, so "is the pointer over a
		// document" would be yes everywhere; "is it over something the document
		// painted" is the question that has a useful answer.
		world.ui.set_pointer_over(hit.is_some());

		let panels: Vec<PanelId> = world.ui.panels().map(|(id, _)| id).collect();
		for panel in &panels {
			let hovered = match hit {
				| Some((over, node)) if over == *panel => node,
				| _ => NONE,
			};

			world.ui.set_hovered(*panel, hovered);
		}

		Self::wheel(world, &self.placed);
		Self::focus(world, &self.placed, hit);
		Self::keyboard(world, &self.placed);
		Self::buttons(world, hit);
	}

	/// Sends the keyboard wherever the last press put it.
	///
	/// A press on a field focuses it and puts the caret at the end; a press on
	/// anything else takes the focus away. Tab moves to the next field in the
	/// document and shift-tab to the previous, wrapping at both ends.
	///
	/// @param world - the input to read and the panel to write
	/// @param placed - this step's layout, for which boxes are fields
	/// @param hit - what the pointer is over, if anything
	fn focus(world: &mut World, placed: &[Placed], hit: Option<(PanelId, u32)>) {
		let input = world.input;

		if input.button_pressed(Button::Left) {
			let onto = hit
				.filter(|&(panel, node)| Self::is_field(world, panel, node))
				.and_then(|(panel, node)| {
					Self::name_of(world, panel, node).map(|name| (panel, name))
				});

			for panel in Self::panels(world) {
				let named = match &onto {
					| Some((over, name)) if *over == panel => name.clone(),
					| _ => String::new(),
				};
				let caret = u32::try_from(world.ui.text(panel, &named).len()).unwrap_or(0);

				world.ui.set_focus(panel, &named, caret);
			}
		}

		if !input.pressed(Key::Tab) {
			return;
		}

		let back = input.held(Key::Shift);

		for panel in Self::panels(world) {
			let fields: Vec<String> = placed
				.iter()
				.filter(|it| it.panel == panel)
				.filter(|it| Self::is_field(world, panel, it.node))
				.filter_map(|it| Self::name_of(world, panel, it.node))
				.collect();

			if fields.is_empty() {
				continue;
			}

			let here = fields
				.iter()
				.position(|name| name == world.ui.focused(panel));
			let next = match (here, back) {
				| (Some(0) | None, true) => fields.len() - 1,
				| (Some(index), true) => index - 1,
				| (Some(index), false) => (index + 1) % fields.len(),
				| (None, false) => 0,
			};
			let Some(name) = fields.get(next).cloned() else {
				continue;
			};
			let caret = u32::try_from(world.ui.text(panel, &name).len()).unwrap_or(0);

			world.ui.set_focus(panel, &name, caret);
		}
	}

	/// Applies this step's typing and editing keys to whatever has the focus.
	///
	/// Text goes in at the caret and the editing keys move or delete around it.
	/// Both halves come from [`Input`](colby_core::abi::Input) and they are
	/// deliberately separate there: a control character never arrives as text,
	/// so nothing here has to tell a tab from a Tab.
	///
	/// @param world - the input to read, and the value and caret to write
	/// @param placed - this step's layout, for the panels that are on screen
	fn keyboard(world: &mut World, placed: &[Placed]) {
		for panel in Self::panels(world) {
			let name = world.ui.focused(panel).to_owned();
			if name.is_empty()
				|| !placed
					.iter()
					.any(|it| it.panel == panel && it.focused)
			{
				continue;
			}

			let mut value = world.ui.text(panel, &name).to_owned();
			let mut caret = usize::try_from(world.ui.caret(panel))
				.unwrap_or(0)
				.min(value.len());

			edit(&world.input, &mut value, &mut caret);

			world.ui.set_text(panel, &name, &value);
			world
				.ui
				.set_focus(panel, &name, u32::try_from(caret).unwrap_or(0));
		}
	}

	/// Whether a node is a field the keyboard can go to.
	fn is_field(world: &World, panel: PanelId, node: u32) -> bool {
		world
			.ui
			.panel(panel)
			.and_then(|panel| world.ui.document(panel.document()))
			.and_then(|document| document.value().node(node).map(Node::is_input))
			.unwrap_or(false)
	}

	/// Every panel, as handles that outlive the borrow.
	fn panels(world: &World) -> Vec<PanelId> { world.ui.panels().map(|(id, _)| id).collect() }

	/// Turns this step's wheel into a scroll on whatever is under the pointer.
	///
	/// The innermost box that can scroll and has the pointer in it, which is
	/// the last one in draw order that answers both - a child is always drawn
	/// after the box holding it. A box that paints nothing still takes the
	/// wheel, unlike a click, because the thing being scrolled is usually a
	/// transparent container around what a person can actually see.
	///
	/// @param world - the input to read and the panel to write
	/// @param placed - this step's layout, for what can scroll and how far
	fn wheel(world: &mut World, placed: &[Placed]) {
		let lines = world.input.wheel;
		if lines.abs() < f32::EPSILON {
			return;
		}

		let pointer = world.ui.pointer();
		let Some(box_of) = placed
			.iter()
			.rev()
			.find(|placed| placed.scrollable > 0.0 && placed.contains(pointer))
		else {
			return;
		};

		let Some(name) = Self::name_of(world, box_of.panel, box_of.node) else {
			// a box with no `id` has nowhere to keep an offset, so it cannot
			// scroll however much its contents overflow. @ref
			// [`Scroll`](colby_core::abi::ui::Scroll).
			return;
		};

		// away from the person moves the contents down, which is a smaller
		// offset: the offset is how far the contents have been pulled *up*.
		let moved = lines.mul_add(-SCROLL_LINE, world.ui.scroll(box_of.panel, &name));

		world
			.ui
			.set_scroll(box_of.panel, &name, moved.clamp(0.0, box_of.scrollable));
	}

	/// Turns this step's button edges into events.
	fn buttons(world: &mut World, hit: Option<(PanelId, u32)>) {
		let input = world.input;

		if input.button_pressed(Button::Left)
			&& let Some((panel, node)) = hit
		{
			world.ui.set_pressed(panel, node);

			if let Some(name) = Self::name_of(world, panel, node) {
				Self::queue(world, Event {
					panel,
					node: name,
					kind: EventKind::Press,
				});
			}
		}

		if !input.button_released(Button::Left) {
			return;
		}

		// the release is reported wherever it landed, and a click only when it
		// landed on the same named box the press did. That is what a button is:
		// pressing one and sliding off it is not a click, and every interface a
		// person has ever used agrees.
		let released = hit.and_then(|(panel, node)| {
			Self::name_of(world, panel, node).map(|name| (panel, name))
		});

		let panels: Vec<(PanelId, u32)> = world
			.ui
			.panels()
			.map(|(id, panel)| (id, panel.pressed()))
			.collect();

		if let Some((panel, name)) = released {
			Self::queue(world, Event {
				panel,
				node: name.clone(),
				kind: EventKind::Release,
			});

			let pressed_here = panels
				.iter()
				.find(|(id, _)| *id == panel)
				.map(|(_, node)| *node)
				.unwrap_or(NONE);

			if pressed_here != NONE
				&& Self::name_of(world, panel, pressed_here).as_deref() == Some(name.as_str())
			{
				Self::queue(world, Event {
					panel,
					node: name,
					kind: EventKind::Click,
				});
			}
		}

		for (panel, _) in panels {
			world.ui.set_pressed(panel, NONE);
		}
	}

	/// Queues one event, and says at trace level what it landed on.
	///
	/// The line is the answer to the question a person debugging a document
	/// actually asks - "did that click hit what I think it hit" - and it is the
	/// only place the whole chain from the window's cursor to a named box is
	/// visible at once. The same shape as the input table's line about a key it
	/// has no name for.
	fn queue(world: &mut World, event: Event) {
		trace!(node = %event.node, kind = ?event.kind, "interface event");

		world.ui.push_event(event);
	}

	/// The `id` an event about a node should name.
	fn name_of(world: &World, panel: PanelId, node: u32) -> Option<String> {
		let document = world
			.ui
			.panel(panel)
			.and_then(|panel| world.ui.document(panel.document()))?;

		layout::named(document.value(), node).map(str::to_owned)
	}

	/// Lays the interface out again and builds this frame's triangles.
	///
	/// Called after the steps and before the draw, because the window may have
	/// been resized since the last step and a document that is a share of the
	/// screen should be the share the screen is now.
	///
	/// @param world - what to draw
	pub fn run(&mut self, world: &World) {
		layout::run(world, &mut self.placed);
		draw::build(world, &self.placed, &mut self.list);

		// after the documents, so that a label drawn to explain something is not
		// behind the interface that is in the way. @ref [`world_text`].
		world_text::build(world, &mut self.list);

		self.viewport = world.ui.viewport().to_array();
	}

	/// Puts everything this frame needs on the GPU.
	///
	/// Separate from [`Overlay::draw`] because that one is handed a frame and
	/// nothing else, and uploading a font atlas needs the world it came out of.
	///
	/// @param device - the device the frames belong to
	/// @param queue - where to write
	/// @param world - where the fonts and textures live
	pub fn prepare(&mut self, device: &Device, queue: &Queue, world: &World) {
		let Some(painter) = self.painter.as_mut() else {
			return;
		};

		painter.reload_shader(device);
		painter.upload(device, queue, world, &self.list);
		painter.write(device, queue, &self.list, self.viewport);
	}

	/// Everything laid out this frame, parents before children.
	#[must_use]
	pub fn placed(&self) -> &[Placed] { &self.placed }

	/// This frame's triangles.
	#[must_use]
	pub const fn list(&self) -> &DrawList { &self.list }

	/// The box under a point, innermost first, or `None`.
	///
	/// @param point - in layout pixels
	#[must_use]
	pub fn at(&self, point: Vec2) -> Option<&Placed> {
		self.placed
			.iter()
			.rev()
			.find(|placed| placed.contains(point))
	}
}

/// Whether a box puts anything on screen, and can therefore be pointed at.
///
/// A box with no background, no picture and no words is scaffolding: it holds
/// other boxes apart and a click should go through it to whatever is behind.
fn paints(placed: &Placed) -> bool {
	if !placed.text.is_empty() || !placed.image.is_empty() {
		return true;
	}

	placed
		.style
		.background
		.is_some_and(|color| !color.is_invisible())
		&& placed.opacity > 0.0
}

impl Overlay for Interface {
	fn draw(
		&mut self,
		device: &Device,
		queue: &Queue,
		target: &TextureView,
		_width: u32,
		_height: u32,
	) {
		if let Some(painter) = self.painter.as_ref() {
			painter.render(device, queue, target, &self.list);
		}
	}
}

#[cfg(test)]
mod tests {
	use colby_core::abi::{
		FontData, Glyph,
		ui::{Length, style::Color},
	};

	use super::*;

	/// A font whose glyphs are all ten wide and one line is twelve tall.
	fn font() -> FontData {
		let glyphs = (0x20..0x7F_u32)
			.map(|codepoint| Glyph {
				codepoint,
				advance: 10.0,
				bearing_x: 0.0,
				bearing_y: 8.0,
				atlas_x: 0,
				atlas_y: 0,
				atlas_width: if codepoint == 0x20 { 0 } else { 8 },
				atlas_height: if codepoint == 0x20 { 0 } else { 8 },
			})
			.collect();

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

	/// One step's input with the wheel turned.
	///
	/// A statement rather than `..Default::default()`: `Input` keeps the
	/// typed-text buffer private, and the struct-update spelling needs every
	/// field to be visible.
	fn wheeled(lines: f32) -> Input {
		let mut input = Input::default();
		input.wheel = lines;

		input
	}

	/// A world showing one document, laid out in a window of a given size.
	fn showing(html: &str, size: Vec2) -> (World, PanelId) {
		let mut world = World::new();
		world.fonts.insert("fonts/test", font());
		world.ui.set_viewport(size, 1.0);

		let parsed = colby_asset::html::parse(html, &[]).expect("the document reads");
		world.ui.insert("ui/test", parsed.document);
		let panel = world.ui.show("ui/test");

		(world, panel)
	}

	#[test]
	fn a_box_with_a_size_lands_where_the_stylesheet_put_it() {
		let (world, _) = showing(
			"<style>#a { width: 100px; height: 40px; background: red; }</style><div \
			 id=\"a\"></div>",
			Vec2::new(800.0, 600.0),
		);

		let mut interface = Interface::new();
		interface.run(&world);

		let placed = interface
			.placed()
			.iter()
			.find(|placed| placed.node == 1)
			.expect("the box was laid out");

		assert!(
			(placed.rect[2] - 100.0).abs() < 0.5 && (placed.rect[3] - 40.0).abs() < 0.5,
			"a hundred by forty, got {:?}",
			placed.rect
		);
	}

	#[test]
	fn flexbox_really_is_doing_the_work() {
		let (world, _) = showing(
			"<style>body { justify-content: center; align-items: center; }#a { width: 100px; \
			 height: 40px; background: red; }</style><div id=\"a\"></div>",
			Vec2::new(800.0, 600.0),
		);

		let mut interface = Interface::new();
		interface.run(&world);

		let placed = interface
			.placed()
			.iter()
			.find(|placed| placed.node == 1)
			.expect("the box was laid out");

		assert!(
			(placed.rect[0] - 350.0).abs() < 1.0 && (placed.rect[1] - 280.0).abs() < 1.0,
			"centered in eight hundred by six hundred, got {:?}",
			placed.rect
		);
	}

	#[test]
	fn a_text_box_is_exactly_as_big_as_its_words() {
		let (world, _) = showing(
			"<style>body { font-family: \"fonts/test\"; font-size: 10px; }</style><div \
			 id=\"a\">abc</div>",
			Vec2::new(800.0, 600.0),
		);

		let mut interface = Interface::new();
		interface.run(&world);

		let text = interface
			.placed()
			.iter()
			.find(|placed| !placed.text.is_empty())
			.expect("the run of text was laid out");

		assert!(
			(text.rect[2] - 30.0).abs() < 0.5,
			"three glyphs of ten, measured from inside the layout: got {:?}",
			text.rect
		);
		assert!((text.rect[3] - 12.0).abs() < 0.5, "and one line of twelve");
	}

	#[test]
	fn a_hidden_box_is_not_laid_out_at_all() {
		let (world, _) = showing(
			"<style>#a { display: none; }</style><div id=\"a\"><div id=\"b\"></div></div>",
			Vec2::new(800.0, 600.0),
		);

		let mut interface = Interface::new();
		interface.run(&world);

		assert_eq!(
			interface.placed().len(),
			1,
			"the root, and neither the hidden box nor anything inside it"
		);
	}

	#[test]
	fn the_pointer_finds_the_innermost_box_under_it() {
		let (mut world, panel) = showing(
			"<style>#outer { width: 200px; height: 200px; padding: 50px; background: red; \
			 }#inner { width: 100px; height: 100px; background: blue; }</style><div \
			 id=\"outer\"><div id=\"inner\"></div></div>",
			Vec2::new(800.0, 600.0),
		);

		world.ui.set_pointer(Vec2::new(100.0, 100.0));

		let mut interface = Interface::new();
		interface.update(&mut world);

		let hovered = world
			.ui
			.panel(panel)
			.map_or(NONE, colby_core::abi::ui::Panel::hovered);

		assert_eq!(hovered, 2, "the inner box, not the one it is inside");
	}

	/// A short box with something far too tall inside it.
	///
	/// Eighty pixels of window on two hundred of contents, so there are a
	/// hundred and twenty to scroll.
	const TALL: &str = "<style>body { padding: 0; } #list { position: absolute; left: 0; top: \
	                    0; width: 100px; height: 80px; overflow: scroll; } #inner { width: \
	                    100px; height: 200px; background: red; }</style><div id=\"list\"><div \
	                    id=\"inner\"></div></div>";

	/// The laid-out box with an `id`, out of a fresh layout.
	fn box_of(world: &World, id: &str) -> Placed {
		let mut placed = Vec::new();
		layout::run(world, &mut placed);

		let document = world
			.ui
			.panel(world.ui.panels().next().expect("a panel").0)
			.and_then(|panel| world.ui.document(panel.document()))
			.expect("a document");

		placed
			.into_iter()
			.find(|placed| {
				document
					.value()
					.node(placed.node)
					.is_some_and(|node| node.id == id)
			})
			.expect("the box is in the layout")
	}

	#[test]
	fn a_box_that_scrolls_measures_what_it_cannot_show() {
		let (world, _) = showing(TALL, Vec2::new(800.0, 600.0));

		assert!(
			(box_of(&world, "list").scrollable - 120.0).abs() < 0.5,
			"two hundred of contents in eighty of box, got {}",
			box_of(&world, "list").scrollable
		);
	}

	#[test]
	fn a_box_that_only_hides_measures_nothing_to_scroll() {
		let (world, _) = showing(
			&TALL.replace("overflow: scroll", "overflow: hidden"),
			Vec2::new(800.0, 600.0),
		);

		assert!(
			box_of(&world, "list").scrollable <= 0.0,
			"cutting something off is not the same as offering to move it"
		);
	}

	#[test]
	fn a_scroll_moves_what_is_inside_and_leaves_the_box_alone() {
		let (mut world, panel) = showing(TALL, Vec2::new(800.0, 600.0));

		let still = box_of(&world, "inner").rect[1];
		world.ui.set_scroll(panel, "list", 40.0);
		let moved = box_of(&world, "inner").rect[1];

		assert!(
			(still - moved - 40.0).abs() < 0.5,
			"the contents went up by exactly what was asked, {still} to {moved}"
		);
		assert!(
			(box_of(&world, "list").rect[1] - 0.0).abs() < 0.5,
			"and the box holding them did not move at all"
		);
	}

	#[test]
	fn a_scroll_stops_at_the_end_of_what_is_inside() {
		let (mut world, panel) = showing(TALL, Vec2::new(800.0, 600.0));

		world.ui.set_scroll(panel, "list", 4000.0);

		assert!(
			(box_of(&world, "list").scroll - 120.0).abs() < 0.5,
			"asking for four thousand gets the hundred and twenty there are, got {}",
			box_of(&world, "list").scroll
		);
	}

	#[test]
	fn the_wheel_scrolls_whatever_is_under_the_pointer() {
		let (mut world, panel) = showing(TALL, Vec2::new(800.0, 600.0));
		let mut interface = Interface::new();
		// away from the person, which pulls the contents up.
		world.input = wheeled(-2.0);

		world.ui.set_pointer(Vec2::new(400.0, 300.0));
		interface.update(&mut world);

		assert!(
			world.ui.scroll(panel, "list") <= 0.0,
			"the pointer is nowhere near the list, so the list did not move"
		);

		world.ui.set_pointer(Vec2::new(50.0, 40.0));
		interface.update(&mut world);

		assert!(
			2.0_f32
				.mul_add(-SCROLL_LINE, world.ui.scroll(panel, "list"))
				.abs() < 0.5,
			"and over it, two lines of wheel are two lines of scroll, got {}",
			world.ui.scroll(panel, "list")
		);
	}

	#[test]
	fn a_scroll_container_shrinks_to_its_share_rather_than_to_its_contents() {
		// a flex item's automatic minimum size is normally the size of what is
		// inside it, which is what makes a row refuse to be narrower than its
		// widest child. A box that holds its own contents is exempt, and this
		// is the one place that rule shows: without it the list is five hundred
		// wide inside a row of a hundred.
		let (world, _) = showing(
			"<style>body { padding: 0; } #row { position: absolute; left: 0; top: 0; width: \
			 100px; height: 40px; } #list { flex-grow: 1; height: 40px; overflow: scroll; } \
			 #inner { width: 500px; height: 20px; }</style><div id=\"row\"><div \
			 id=\"list\"><div id=\"inner\"></div></div></div>",
			Vec2::new(800.0, 600.0),
		);

		assert!(
			(box_of(&world, "list").rect[2] - 100.0).abs() < 0.5,
			"the list takes the row's width, got {}",
			box_of(&world, "list").rect[2]
		);
	}

	#[test]
	fn the_wheel_finds_the_innermost_thing_that_can_scroll() {
		// both of these can scroll and the pointer is in both. The one that
		// should move is the one nearest the pointer, which is the last of the
		// two in draw order: a child is always drawn after the box holding it.
		let (mut world, panel) = showing(
			"<style>body { padding: 0; } #outer { position: absolute; left: 0; top: 0; width: \
			 200px; height: 100px; overflow: scroll; } #inner { width: 200px; height: 80px; \
			 overflow: scroll; } #deep { width: 200px; height: 400px; } #filler { width: 200px; \
			 height: 200px; }</style><div id=\"outer\"><div id=\"inner\"><div \
			 id=\"deep\"></div></div><div id=\"filler\"></div></div>",
			Vec2::new(800.0, 600.0),
		);

		assert!(box_of(&world, "outer").scrollable > 0.0, "the outer one can move");
		assert!(box_of(&world, "inner").scrollable > 0.0, "and so can the inner one");

		let mut interface = Interface::new();
		world.input = wheeled(-1.0);

		// well inside the inner box: the row splits its two hundred between the
		// list and the filler, so the list ends at a hundred and a pointer there
		// is already in its neighbor.
		world.ui.set_pointer(Vec2::new(50.0, 40.0));
		interface.update(&mut world);

		assert!(
			(world.ui.scroll(panel, "inner") - SCROLL_LINE).abs() < 0.5,
			"the inner one moved, got {}",
			world.ui.scroll(panel, "inner")
		);
		assert!(
			world.ui.scroll(panel, "outer") <= 0.0,
			"and the one holding it stayed where it was, got {}",
			world.ui.scroll(panel, "outer")
		);
	}

	/// One step's input with something typed into it.
	///
	/// A statement rather than `..Default::default()`, for the reason
	/// [`wheeled`] is one.
	fn typing(text: &str) -> Input {
		let mut input = Input::default();
		input.type_text(text);

		input
	}

	/// Two fields side by side, both named.
	const FIELDS: &str = "<style>body { padding: 0; } #one { position: absolute; left: 0; top: \
	                      0; width: 100px; height: 20px; } #two { position: absolute; left: 0; \
	                      top: 40px; width: 100px; height: 20px; }</style><input id=\"one\" \
	                      value=\"ab\"><input id=\"two\">";

	#[test]
	fn a_field_says_what_the_document_said_until_somebody_changes_it() {
		let (mut world, panel) = showing(FIELDS, Vec2::new(800.0, 600.0));

		assert_eq!(world.ui.text(panel, "one"), "ab", "the value attribute is the default");

		world.ui.set_text(panel, "one", "cd");

		assert_eq!(world.ui.text(panel, "one"), "cd", "and the game writing it wins");
	}

	#[test]
	fn pressing_a_field_sends_the_keyboard_to_it_and_pressing_away_takes_it_back() {
		let (mut world, panel) = showing(FIELDS, Vec2::new(800.0, 600.0));
		let mut interface = Interface::new();
		let mut input = Input::default();

		world.ui.set_pointer(Vec2::new(50.0, 10.0));
		input.set_button(Button::Left, true);
		world.input = input;
		interface.update(&mut world);

		assert_eq!(world.ui.focused(panel), "one", "the press landed on the first field");
		assert_eq!(world.ui.caret(panel), 2, "with the caret after what was already in it");

		// let go, then press again somewhere else: the focus follows a press
		// rather than a held button, so a second one has to be a second edge.
		input.end_step();
		input.set_button(Button::Left, false);
		world.input = input;
		interface.update(&mut world);

		input.end_step();
		input.set_button(Button::Left, true);
		world.ui.set_pointer(Vec2::new(400.0, 400.0));
		world.input = input;
		interface.update(&mut world);

		assert_eq!(world.ui.focused(panel), "", "and a press on nothing takes it away again");
	}

	#[test]
	fn tab_walks_the_fields_and_comes_back_round() {
		let (mut world, panel) = showing(FIELDS, Vec2::new(800.0, 600.0));
		let mut interface = Interface::new();
		let mut input = Input::default();

		input.set_key(Key::Tab, true);
		world.input = input;
		interface.update(&mut world);

		assert_eq!(world.ui.focused(panel), "one", "nothing was focused, so it starts at one");

		// released and pressed again, because Tab moving the focus is an edge.
		input.end_step();
		input.set_key(Key::Tab, false);
		world.input = input;
		interface.update(&mut world);

		input.end_step();
		input.set_key(Key::Tab, true);
		world.input = input;
		interface.update(&mut world);

		assert_eq!(world.ui.focused(panel), "two", "and then the next one");
	}

	#[test]
	fn what_is_typed_lands_in_the_focused_field_at_the_caret() {
		let (mut world, panel) = showing(FIELDS, Vec2::new(800.0, 600.0));
		let mut interface = Interface::new();

		world.ui.set_focus(panel, "one", 1);
		world.input = typing("XY");
		interface.update(&mut world);

		assert_eq!(world.ui.text(panel, "one"), "aXYb", "typed between the two letters");
		assert_eq!(world.ui.caret(panel), 3, "and the caret came with it");
	}

	#[test]
	fn nothing_is_typed_into_a_panel_with_nothing_focused() {
		let (mut world, panel) = showing(FIELDS, Vec2::new(800.0, 600.0));
		let mut interface = Interface::new();

		world.input = typing("XY");
		interface.update(&mut world);

		assert_eq!(world.ui.text(panel, "one"), "ab", "the keyboard had nowhere to go");
	}

	#[test]
	fn a_field_that_is_not_on_screen_takes_nothing() {
		// what a document recompiled without the field somebody was typing in
		// leaves behind: a focus naming a box that no longer exists. Writing
		// into it would bind text to a node nothing draws, and quietly keep
		// doing it for as long as the keyboard was pointed there.
		let (mut world, panel) = showing(FIELDS, Vec2::new(800.0, 600.0));
		let mut interface = Interface::new();

		world.ui.set_focus(panel, "ghost", 0);
		world.input = typing("XY");
		interface.update(&mut world);

		assert_eq!(world.ui.text(panel, "ghost"), "", "nothing was written anywhere");
	}

	#[test]
	fn backspace_takes_a_whole_character_and_not_a_byte() {
		let (mut world, panel) = showing(FIELDS, Vec2::new(800.0, 600.0));
		let mut interface = Interface::new();
		let mut input = Input::default();

		world.ui.set_focus(panel, "one", 0);
		// two bytes each, so a field that counted bytes would leave half of one
		// behind and stop being a string at all.
		world.input = typing("\u{444}\u{44b}");
		interface.update(&mut world);

		assert_eq!(world.ui.text(panel, "one"), "\u{444}\u{44b}ab", "both went in");

		input.set_key(Key::Backspace, true);
		world.input = input;
		interface.update(&mut world);

		assert_eq!(world.ui.text(panel, "one"), "\u{444}ab", "and one whole one came out");
		assert_eq!(world.ui.caret(panel), 2, "leaving the caret after the other");
	}

	#[test]
	fn the_caret_walks_by_characters_and_stops_at_both_ends() {
		let mut value = "a\u{444}b".to_owned();
		let mut caret = 0;
		let mut input = Input::default();

		input.set_key(Key::Left, true);
		edit(&input, &mut value, &mut caret);
		assert_eq!(caret, 0, "there is nowhere to the left of the start");

		input = Input::default();
		input.set_key(Key::Right, true);
		edit(&input, &mut value, &mut caret);
		assert_eq!(caret, 1, "one ASCII letter is one byte");
		edit(&input, &mut value, &mut caret);
		assert_eq!(caret, 3, "and one Cyrillic one is two");

		input = Input::default();
		input.set_key(Key::End, true);
		edit(&input, &mut value, &mut caret);
		assert_eq!(caret, 4, "the end is the end");
		edit(&input, &mut value, &mut caret);
		assert_eq!(caret, 4, "and stays there");
	}

	#[test]
	fn a_caret_left_inside_a_character_is_moved_off_it() {
		// what a game writing over the field somebody is typing in leaves
		// behind. Editing from there has to be impossible rather than merely
		// unlikely: `String::remove` panics on an offset like this one.
		let mut value = "\u{444}b".to_owned();
		let mut caret = 1;
		let mut input = Input::default();

		input.set_key(Key::Delete, true);
		edit(&input, &mut value, &mut caret);

		assert_eq!(value, "b", "the whole character went, from the boundary before it");
		assert_eq!(caret, 0, "and the caret is somewhere that means something");
	}

	#[test]
	fn a_press_and_a_release_on_the_same_box_are_a_click() {
		let (mut world, panel) = showing(
			"<style>#go { width: 100px; height: 40px; background: red; }</style><div \
			 id=\"go\">press</div>",
			Vec2::new(800.0, 600.0),
		);

		world.ui.set_pointer(Vec2::new(50.0, 20.0));

		let mut interface = Interface::new();
		let mut input = Input::default();

		input.set_button(Button::Left, true);
		world.input = input;
		interface.update(&mut world);

		assert!(!world.ui.clicked(panel, "go"), "a press on its own is not a click");
		world.ui.end_step();

		input.end_step();
		input.set_button(Button::Left, false);
		world.input = input;
		interface.update(&mut world);

		assert!(world.ui.clicked(panel, "go"), "and the release that follows it is");
	}

	#[test]
	fn a_press_that_slides_off_the_box_is_not_a_click() {
		let (mut world, panel) = showing(
			"<style>#go { width: 100px; height: 40px; background: red; }</style><div \
			 id=\"go\"></div>",
			Vec2::new(800.0, 600.0),
		);

		let mut interface = Interface::new();
		let mut input = Input::default();

		world.ui.set_pointer(Vec2::new(50.0, 20.0));
		input.set_button(Button::Left, true);
		world.input = input;
		interface.update(&mut world);
		world.ui.end_step();

		// off the box, then let go.
		world.ui.set_pointer(Vec2::new(400.0, 400.0));
		input.end_step();
		input.set_button(Button::Left, false);
		world.input = input;
		interface.update(&mut world);

		assert!(!world.ui.clicked(panel, "go"), "letting go somewhere else is not a click");
	}

	#[test]
	fn clicking_the_words_on_a_button_clicks_the_button() {
		let (mut world, panel) = showing(
			"<style>body { font-family: \"fonts/test\"; }#go { width: 200px; height: 40px; \
			 background: red; }</style><div id=\"go\">press me</div>",
			Vec2::new(800.0, 600.0),
		);

		world.ui.set_pointer(Vec2::new(20.0, 5.0));

		let mut interface = Interface::new();
		let mut input = Input::default();

		input.set_button(Button::Left, true);
		world.input = input;
		interface.update(&mut world);
		world.ui.end_step();

		input.end_step();
		input.set_button(Button::Left, false);
		world.input = input;
		interface.update(&mut world);

		assert!(
			world.ui.clicked(panel, "go"),
			"the run of text has no identifier of its own, so the event is the button's"
		);
	}

	#[test]
	fn a_hover_rule_applies_while_the_pointer_is_over_the_box() {
		let (mut world, _) = showing(
			"<style>#a { width: 100px; height: 40px; background: red; }#a:hover { background: \
			 blue; }</style><div id=\"a\"></div>",
			Vec2::new(800.0, 600.0),
		);

		let mut interface = Interface::new();

		world.ui.set_pointer(Vec2::new(-1.0, -1.0));
		interface.update(&mut world);
		interface.run(&world);
		let away = interface.placed()[1].style.background;

		world.ui.set_pointer(Vec2::new(50.0, 20.0));
		interface.update(&mut world);
		interface.run(&world);
		let over = interface.placed()[1].style.background;

		assert_ne!(away, over, "the same box is painted differently under the pointer");
		assert_eq!(over, Some(Color::from_srgb(0, 0, 255, 255)), "and it is the hover rule's");
	}

	#[test]
	fn what_a_game_binds_beats_what_the_document_says() {
		let (mut world, panel) = showing(
			"<style>#bar { height: 10px; background: red; }</style><div id=\"bar\"></div>",
			Vec2::new(800.0, 600.0),
		);

		if let Some(style) = world.ui.style_mut(panel, "bar") {
			style.width = Some(Length::Px(37.0));
		}

		let mut interface = Interface::new();
		interface.run(&world);

		assert!(
			(interface.placed()[1].rect[2] - 37.0).abs() < 0.5,
			"a bar's width is a number the game writes, not a class it swaps: got {:?}",
			interface.placed()[1].rect
		);
	}

	#[test]
	fn text_a_game_bound_is_the_text_that_is_drawn() {
		let (mut world, panel) = showing(
			"<style>body { font-family: \"fonts/test\"; font-size: 10px; }</style><div \
			 id=\"score\">0</div>",
			Vec2::new(800.0, 600.0),
		);

		world.ui.set_text(panel, "score", "1200");

		let mut interface = Interface::new();
		interface.run(&world);

		let text = interface
			.placed()
			.iter()
			.find(|placed| !placed.text.is_empty())
			.expect("there is a run of text");

		assert_eq!(text.text, "1200", "what the game wrote");
		assert!(
			(text.rect[2] - 40.0).abs() < 0.5,
			"and it was measured after being written, not before: got {:?}",
			text.rect
		);
	}

	#[test]
	fn a_document_nobody_showed_is_not_drawn() {
		let (mut world, panel) = showing(
			"<style>#a { width: 10px; height: 10px; background: red; }</style><div \
			 id=\"a\"></div>",
			Vec2::new(800.0, 600.0),
		);

		world.ui.hide(panel);

		let mut interface = Interface::new();
		interface.run(&world);

		assert!(interface.placed().is_empty(), "nothing hidden is laid out");
		assert!(interface.list().is_empty(), "nor drawn");
	}

	#[test]
	fn an_interface_with_no_gpu_behind_it_still_lays_out_and_draws_nothing() {
		let (world, _) = showing("<div id=\"a\"></div>", Vec2::new(800.0, 600.0));

		let mut interface = Interface::new();
		interface.run(&world);

		assert!(!interface.is_attached(), "no pipeline");
		assert!(!interface.placed().is_empty(), "and the layout happened anyway");
	}
}
