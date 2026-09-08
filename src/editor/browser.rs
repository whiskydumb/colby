//! The asset browser: every source under `assets/`, with what it is, whether
//! it has compiled, and a picture of it where there is one to draw.
//!
//! A row is dragged into the picture to put the thing in the world - a scene
//! laid down where it was dropped, a mesh as an entity standing there, a
//! model as an entity with a child per piece; @ref `select::drop`. A texture
//! or a sound is not a thing that stands anywhere, and dropping one says so.
//!
//! The tree is walked again once a second, which is how a file just written
//! by the editor itself or by somebody else shows up; the asset loop polls
//! four times a second and this need not keep up with it.
//!
//! Nothing here changes the world: what is dragged is carried as a payload,
//! and the frame that sees it dropped over the picture hands back a
//! [`Change`](crate::Change).

use std::{
	path::PathBuf,
	time::{Duration, Instant},
};

use colby_asset::{Project, compile::Kind};
use colby_engine::Gpu;
use egui::{
	Align2, Button, Id, LayerId, Order, RichText, ScrollArea, Sense, TextEdit, TextStyle, Ui,
	vec2,
};

use crate::{
	Change,
	catalog::{self, Entry, State},
	thumbs::Thumbs,
};

/// How often the tree is walked again.
const RESCAN: Duration = Duration::from_secs(1);

/// How wide a picture is in a row, in points.
const THUMB: f32 = 28.0;

/// What is being dragged out of the browser: one asset, by name and kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Dropped {
	/// The asset name, `meshes/crystal`.
	pub(crate) name: String,

	/// What it is.
	pub(crate) kind: Kind,
}

/// The browser's own state.
#[derive(Default)]
pub(crate) struct Browser {
	/// Every source, as last walked.
	entries: Vec<Entry>,

	/// When they were last walked.
	scanned: Option<Instant>,

	/// What is in the search box.
	search: String,

	/// The pictures, once a project is known.
	thumbs: Option<Thumbs>,

	/// Which project the pictures and the entries are of.
	root: Option<PathBuf>,
}

impl Browser {
	/// Draws the browser into a panel.
	///
	/// @param ui - the panel
	/// @param project - whose tree, if the window has one
	/// @param gpu - the device to draw a mesh's picture with, if there is one
	pub(crate) fn show(
		&mut self,
		ui: &mut Ui,
		project: Option<&Project>,
		gpu: Option<&Gpu>,
		changes: &mut Vec<Change>,
	) {
		let Some(project) = project else {
			ui.label("no project");

			return;
		};

		self.refresh(project);

		ui.horizontal(|ui| {
			ui.label(format!("{} assets under assets/", self.entries.len()));
			ui.add(
				TextEdit::singleline(&mut self.search)
					.desired_width(160.0)
					.hint_text("search"),
			);
			ui.label(
				RichText::new(
					"drag one into the picture, or double-click to open a scene or edit a 					 \
					 source",
				)
				.weak(),
			);
		});
		ui.separator();

		// one picture made per frame at most, @ref `Thumbs::get`
		let mut made = false;

		ScrollArea::vertical()
			.auto_shrink([false, false])
			.show(ui, |ui| self.rows(ui, gpu, &mut made, changes));
	}

	/// Walks the tree again when it is time to, and starts over when the
	/// project changed.
	fn refresh(&mut self, project: &Project) {
		if self.root.as_deref() != Some(project.root()) {
			self.root = Some(project.root().to_owned());
			self.thumbs = Some(Thumbs::new(project.thumbs(), project.output()));
			self.entries.clear();
			self.scanned = None;
		}

		if self
			.scanned
			.is_none_or(|then| then.elapsed() >= RESCAN)
		{
			self.entries = catalog::scan(&project.assets(), &project.output());
			self.scanned = Some(Instant::now());
		}
	}

	/// Every row that matches the search.
	fn rows(
		&mut self,
		ui: &mut Ui,
		gpu: Option<&Gpu>,
		made: &mut bool,
		changes: &mut Vec<Change>,
	) {
		let filter = self.search.trim().to_lowercase();

		for entry in &self.entries {
			if !filter.is_empty() && !entry.name.to_lowercase().contains(&filter) {
				continue;
			}

			let thumb = self
				.thumbs
				.as_mut()
				.and_then(|thumbs| thumbs.get(ui.ctx(), gpu, entry, made));

			row(ui, entry, thumb, changes);
		}
	}
}

/// One asset: its picture or the room for one, a row that can be dragged,
/// and a word about its state when there is something to say.
fn row(ui: &mut Ui, entry: &Entry, thumb: Option<egui::TextureId>, changes: &mut Vec<Change>) {
	ui.horizontal(|ui| {
		match thumb {
			| Some(id) => {
				ui.image((id, vec2(THUMB, THUMB)));
			},
			| None => {
				ui.add_space(THUMB + ui.spacing().item_spacing.x);
			},
		}

		let response = ui.add(
			Button::selectable(false, format!("{}  {}", catalog::word(entry.kind), entry.name))
				.sense(Sense::click_and_drag()),
		);
		response.dnd_set_drag_payload(Dropped {
			name: entry.name.clone(),
			kind: entry.kind,
		});

		if response.dragged() {
			ghost(ui, &entry.name);
		}

		// a scene is the one kind of asset there is somewhere to go to, so it
		// is the one kind a double-click opens *here*; every other kind that
		// is text goes out to an editor, which is the same gesture Godot's
		// file system dock uses for the same thing. What is left - a picture,
		// a sound, a font, a model - is dragged into the picture and nothing
		// else.
		if response.double_clicked() {
			if entry.kind == Kind::Scene {
				changes.push(Change::Open { name: entry.name.clone() });
			} else if catalog::is_text(entry.kind) {
				changes.push(Change::Code { name: entry.name.clone() });
			}
		}

		// the two kinds the inspector has a panel for: a material, whose field
		// table it can draw and whose numbers it can change, and a model,
		// which it can only read out - what pieces came out of the file, and
		// what a sidecar had to do with it.
		if matches!(entry.kind, Kind::Material | Kind::Model) && response.clicked() {
			changes.push(Change::Inspect { name: entry.name.clone() });
		}

		if entry.state != State::Compiled {
			ui.label(RichText::new(entry.state.word()).weak());
		}
	});
}

/// The name of the row being dragged, beside the pointer.
fn ghost(ui: &Ui, name: &str) {
	let Some(pointer) = ui.ctx().pointer_interact_pos() else {
		return;
	};

	ui.ctx()
		.layer_painter(LayerId::new(Order::Tooltip, Id::new("browser drag")))
		.text(
			pointer + vec2(12.0, 12.0),
			Align2::LEFT_TOP,
			name,
			TextStyle::Body.resolve(ui.style()),
			ui.visuals().strong_text_color(),
		);
}

#[cfg(test)]
mod tests {
	use std::{env, fs};

	use egui::{Context, Modifiers, PointerButton, Pos2, RawInput, Rect};

	use super::*;

	/// One row drawn on its own, and what pressing it asked for.
	///
	/// [`row`] rather than [`Browser::show`], because what is being tested is
	/// what a gesture on a row *means* and a row is where that is decided; the
	/// walk, the search and the pictures are the browser's and have their own
	/// test above. The rectangle is what the row drew, which is where the
	/// second call clicks.
	///
	/// @param context - reused between the two calls, so that egui remembers
	/// where the widget was and how long ago the last click was
	/// @param entry - the row
	/// @param events - what happened this frame
	/// @return what the row asked for, and where it drew itself
	fn pressed(
		context: &Context,
		entry: &Entry,
		events: Vec<egui::Event>,
	) -> (Vec<Change>, Rect) {
		let mut changes = Vec::new();
		let mut drawn = Rect::NOTHING;
		let mut once = false;

		// once whatever egui asks: a context may run the closure twice in one
		// call, and a row drawn twice would answer a click twice.
		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(600.0, 60.0))),
				events,
				..Default::default()
			},
			|ui| {
				if !once {
					once = true;
					row(ui, entry, None, &mut changes);
					drawn = ui.min_rect();
				}
			},
		);
		output.textures_delta.clear();

		(changes, drawn)
	}

	/// A row for a source that is not on disk anywhere.
	fn entry(name: &str, kind: Kind) -> Entry {
		Entry {
			name: name.to_owned(),
			kind,
			source: PathBuf::from(name),
			output: PathBuf::from(name),
			state: State::Compiled,
		}
	}

	/// Two presses and two releases in one frame, which is a double click.
	fn twice(at: Pos2) -> Vec<egui::Event> {
		let mut events = vec![egui::Event::PointerMoved(at)];

		for _ in 0..2 {
			for pressed in [true, false] {
				events.push(egui::Event::PointerButton {
					pos: at,
					button: PointerButton::Primary,
					pressed,
					modifiers: Modifiers::NONE,
				});
			}
		}

		events
	}

	#[test]
	fn the_browser_draws_a_project_without_a_window_and_lists_its_sources() {
		let root = env::temp_dir().join("colby_browser_draws");
		drop(fs::remove_dir_all(&root));
		fs::create_dir_all(root.join("assets").join("meshes")).expect("a tree to work in");
		fs::write(
			root.join("assets")
				.join("meshes")
				.join("crystal.obj"),
			b"v 0 0 0\n",
		)
		.expect("a source");
		let project = Project::parse(
			&root,
			r#"{ "schema": 1, "engine": "0.1.0", "id": "browsed", "name": "Browsed" }"#,
		)
		.expect("a project");
		let mut browser = Browser::default();
		let context = Context::default();

		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(600.0, 400.0))),
				..Default::default()
			},
			|ui| browser.show(ui, Some(&project), None, &mut Vec::new()),
		);
		output.textures_delta.clear();

		assert_eq!(browser.entries.len(), 1, "the tree was walked");
		assert_eq!(browser.entries[0].name, "meshes/crystal");
		assert_eq!(browser.entries[0].state, State::Uncompiled, "nothing compiled it");
		assert_eq!(browser.root.as_deref(), Some(root.as_path()));
	}

	#[test]
	fn double_clicking_a_program_asks_for_it_to_be_opened_in_an_editor() {
		let context = Context::default();
		let row = entry("scripts/thruster", Kind::Script);
		// one frame to find out where the row landed, and a second to press
		// there: egui answers where a widget is only after it has drawn
		let (_, drawn) = pressed(&context, &row, Vec::new());

		let (changes, _) = pressed(&context, &row, twice(drawn.center()));

		assert_eq!(
			changes,
			vec![Change::Code { name: "scripts/thruster".to_owned() }],
			"and that is the only thing it asks for"
		);
	}

	#[test]
	fn double_clicking_a_scene_still_opens_the_scene_rather_than_its_text() {
		// the exception, and the reason `catalog::is_text` says no to a scene:
		// this gesture already meant something on this row.
		let context = Context::default();
		let row = entry("scenes/yard", Kind::Scene);
		let (_, drawn) = pressed(&context, &row, Vec::new());

		let (changes, _) = pressed(&context, &row, twice(drawn.center()));

		assert_eq!(changes, vec![Change::Open { name: "scenes/yard".to_owned() }]);
	}

	#[test]
	fn double_clicking_a_material_inspects_it_and_opens_it() {
		// the one row where two gestures overlap, and the answer is that both
		// happen: egui counts a double click as a click as well, so the
		// material's source opens *and* it lands in the inspector.
		//
		// **Recorded rather than prevented, because preventing it is not
		// available.** With a real mouse the first click of the pair lands a
		// frame before the second and has already asked for the inspector by
		// the time anything knows a second is coming; suppressing the click on
		// the frame the double lands would change this test and nothing a
		// person does. Reading a material's numbers while its text opens is
		// what whoever pressed it wanted either way.
		let context = Context::default();
		let row = entry("materials/brass", Kind::Material);
		let (_, drawn) = pressed(&context, &row, Vec::new());

		let (changes, _) = pressed(&context, &row, twice(drawn.center()));

		assert_eq!(changes, vec![
			Change::Code { name: "materials/brass".to_owned() },
			Change::Inspect { name: "materials/brass".to_owned() },
		]);
	}

	#[test]
	fn double_clicking_a_picture_asks_for_nothing() {
		let context = Context::default();
		let row = entry("textures/wall", Kind::Texture);
		let (_, drawn) = pressed(&context, &row, Vec::new());

		let (changes, _) = pressed(&context, &row, twice(drawn.center()));

		assert!(changes.is_empty(), "a .png in a code editor is a screen of nothing");
	}

	#[test]
	fn one_click_on_a_program_asks_for_nothing() {
		// the negative control for the two above: the same row, the same
		// place, one press instead of two, and nothing is asked for - so what
		// those tests saw was the second click and not merely a click.
		let context = Context::default();
		let row = entry("scripts/thruster", Kind::Script);
		let (_, drawn) = pressed(&context, &row, Vec::new());
		let at = drawn.center();

		let (changes, _) = pressed(&context, &row, vec![
			egui::Event::PointerMoved(at),
			egui::Event::PointerButton {
				pos: at,
				button: PointerButton::Primary,
				pressed: true,
				modifiers: Modifiers::NONE,
			},
			egui::Event::PointerButton {
				pos: at,
				button: PointerButton::Primary,
				pressed: false,
				modifiers: Modifiers::NONE,
			},
		]);

		assert!(changes.is_empty());
	}

	#[test]
	fn no_project_is_a_line_and_no_walk() {
		let mut browser = Browser::default();
		let context = Context::default();

		let mut output = context
			.run_ui(RawInput::default(), |ui| browser.show(ui, None, None, &mut Vec::new()));
		output.textures_delta.clear();

		assert!(browser.entries.is_empty());
		assert!(browser.scanned.is_none());
	}
}
