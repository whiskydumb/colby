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
		Button, World,
		ui::{Event, EventKind, PanelId, document::NONE},
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

		Self::buttons(world, hit);
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
		FontData, Glyph, Input,
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
