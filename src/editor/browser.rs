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
//! A right click on a row opens a menu with the two things that are about the
//! *asset* rather than about the world: renaming it, and copying its identity.
//! Neither is done here - the editor holds no path, and the runner is the half
//! that owns the project - so both go out as a [`Change`](crate::Change) and
//! come back as a console line, exactly as opening a source does.
//!
//! Nothing here changes the world: what is dragged is carried as a payload,
//! and the frame that sees it dropped over the picture hands back a
//! [`Change`](crate::Change).

use std::{
	path::PathBuf,
	time::{Duration, Instant},
};

use colby_asset::{Project, compile::Kind, ident::Ids};
use colby_core::{abi::ident, info};
use colby_engine::Gpu;
use egui::{
	Align2, Button, Id, Key, LayerId, Order, RichText, ScrollArea, Sense, TextEdit, TextStyle,
	Ui, vec2,
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

	/// Every identity in the tree, as last read.
	///
	/// Read off the output tree with the walk rather than out of the world's
	/// tables: a row is a *source file*, and a source that has never compiled
	/// is in no table and still has an identity.
	ids: Ids,

	/// Which row has the keyboard, by asset name.
	renaming: Option<String>,

	/// What is in that row's field.
	typed: String,
}

/// The one row that is being renamed, and what is in it.
///
/// Handed to [`row`] rather than read off the browser, because a row is drawn
/// by a free function so that a test can press one without a project.
struct Naming<'a> {
	/// Which asset the field is open on.
	on: &'a mut Option<String>,

	/// What has been typed into it.
	typed: &'a mut String,
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
				RichText::new("drag one into the picture, or double-click to open or edit one")
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
			self.ids = Ids::read(&project.output());
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
			let mut naming = Naming {
				on: &mut self.renaming,
				typed: &mut self.typed,
			};

			row(
				ui,
				entry,
				thumb,
				self.ids.id(&entry.name).unwrap_or_default(),
				&mut naming,
				changes,
			);
		}
	}
}

/// One asset: its picture or the room for one, a row that can be dragged,
/// and a word about its state when there is something to say.
///
/// @param id - the asset's identity, for the hover and the menu
/// @param naming - which row has the keyboard, and what is in it
fn row(
	ui: &mut Ui,
	entry: &Entry,
	thumb: Option<egui::TextureId>,
	id: ident::Id,
	naming: &mut Naming<'_>,
	changes: &mut Vec<Change>,
) {
	ui.horizontal(|ui| {
		match thumb {
			| Some(held) => {
				ui.image((held, vec2(THUMB, THUMB)));
			},
			| None => {
				ui.add_space(THUMB + ui.spacing().item_spacing.x);
			},
		}

		if naming.on.as_deref() == Some(entry.name.as_str()) {
			naming_row(ui, entry, naming, changes);

			return;
		}

		let response = ui
			.add(
				Button::selectable(
					false,
					format!("{}  {}", catalog::word(entry.kind), entry.name),
				)
				.sense(Sense::click_and_drag()),
			)
			.on_hover_text(match id.is_none() {
				| true => format!(
					"{}
no identity yet; it gets one when it compiles",
					entry.name
				),
				| false => format!(
					"{}
{id}",
					entry.name
				),
			});
		response.dnd_set_drag_payload(Dropped {
			name: entry.name.clone(),
			kind: entry.kind,
		});
		response.context_menu(|ui| menu(ui, entry, id, naming));

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

/// The two things a right click offers, both about the asset and not the world.
///
/// @param id - the asset's identity, or nothing when it has none yet
/// @param naming - which row has the keyboard, to open the field on this one
fn menu(ui: &mut Ui, entry: &Entry, id: ident::Id, naming: &mut Naming<'_>) {
	if ui.button("rename").clicked() {
		// the last part of the name, which is the only part a rename changes:
		// where an asset stands is a different question with different
		// consequences. @ref `colby_runtime`'s `rename` module.
		last(&entry.name).clone_into(naming.typed);
		*naming.on = Some(entry.name.clone());
		ui.close();
	}

	if ui
		.add_enabled(!id.is_none(), Button::new("copy identity"))
		.on_hover_text("the thirteen letters a source names this asset by, whatever it is called")
		.clicked()
	{
		ui.ctx().copy_text(id.to_string());
		info!(name = entry.name, %id, "the identity is in the clipboard");
		ui.close();
	}
}

/// The row while it is being renamed: a field with the last part of the name.
///
/// Enter asks for the rename, escape puts it away, and so does pressing
/// anywhere else - which is what losing the keyboard means and is the one
/// gesture nobody has to be taught.
fn naming_row(ui: &mut Ui, entry: &Entry, naming: &mut Naming<'_>, changes: &mut Vec<Change>) {
	let id = Id::new(("browser rename", &entry.name));
	let response = ui.add(
		TextEdit::singleline(naming.typed)
			.id(id)
			.desired_width(160.0),
	);

	// the frame it opens on and no other. Asking again on the frame it *loses*
	// the keyboard would put the focus straight back, and losing the keyboard
	// is the whole of how this row knows enter was pressed.
	if !response.has_focus() && !response.lost_focus() {
		response.request_focus();
	}

	ui.label(RichText::new(catalog::word(entry.kind)).weak());

	if ui.input(|input| input.key_pressed(Key::Escape)) {
		*naming.on = None;

		return;
	}

	if response.lost_focus() {
		let typed = naming.typed.trim().to_owned();

		*naming.on = None;

		// the same name is not a rename, and neither is an empty field: both
		// are what pressing enter on a field nobody edited means.
		if ui.input(|input| input.key_pressed(Key::Enter))
			&& !typed.is_empty()
			&& typed != last(&entry.name)
		{
			changes.push(Change::RenameAsset { name: entry.name.clone(), to: typed });
		}
	}
}

/// The last part of an asset name, which is the part a rename changes.
fn last(name: &str) -> &str { name.rsplit('/').next().unwrap_or(name) }

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
		let mut on = None;

		typing(context, entry, events, &mut on, &mut String::new())
	}

	/// The same, with the field's state held by the caller across frames.
	///
	/// @param on - which row has the keyboard, which a test opens by hand
	/// @param typed - what is in the field
	fn typing(
		context: &Context,
		entry: &Entry,
		events: Vec<egui::Event>,
		on: &mut Option<String>,
		typed: &mut String,
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
					row(
						ui,
						entry,
						None,
						ident::Id::NONE,
						&mut Naming { on, typed },
						&mut changes,
					);
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

	/// What a key press and its release look like to egui.
	fn key(which: Key) -> Vec<egui::Event> {
		[true, false]
			.into_iter()
			.map(|pressed| egui::Event::Key {
				key: which,
				physical_key: None,
				pressed,
				repeat: false,
				modifiers: Modifiers::NONE,
			})
			.collect()
	}

	#[test]
	fn a_row_being_renamed_asks_for_the_rename_when_enter_is_pressed() {
		let context = Context::default();
		let row = entry("meshes/crystal", Kind::Mesh);
		let mut on = Some("meshes/crystal".to_owned());
		let mut typed = "gem".to_owned();

		// one frame for the field to take the keyboard, and a second for the
		// key: a widget that was not focused cannot lose focus.
		let (waiting, _) = typing(&context, &row, Vec::new(), &mut on, &mut typed);
		let (changes, _) = typing(&context, &row, key(Key::Enter), &mut on, &mut typed);

		assert!(waiting.is_empty(), "a field that is merely open asks for nothing");
		assert_eq!(changes, vec![Change::RenameAsset {
			name: "meshes/crystal".to_owned(),
			to: "gem".to_owned(),
		}]);
		assert_eq!(on, None, "and the field is put away");
	}

	#[test]
	fn escape_puts_the_field_away_and_renames_nothing() {
		let context = Context::default();
		let row = entry("meshes/crystal", Kind::Mesh);
		let mut on = Some("meshes/crystal".to_owned());
		let mut typed = "gem".to_owned();

		typing(&context, &row, Vec::new(), &mut on, &mut typed);
		let (changes, _) = typing(&context, &row, key(Key::Escape), &mut on, &mut typed);

		assert!(changes.is_empty(), "nothing was asked for");
		assert_eq!(on, None, "and the field is gone");
	}

	#[test]
	fn a_field_nobody_edited_is_not_a_rename() {
		// the two ways to press enter on a field that says nothing new, and
		// neither is a rename: the compiler would refuse the first as a name
		// something already has, and the second as no name at all.
		let context = Context::default();
		let row = entry("meshes/crystal", Kind::Mesh);

		for said in ["crystal", "   "] {
			let mut on = Some("meshes/crystal".to_owned());
			let mut typed = said.to_owned();

			typing(&context, &row, Vec::new(), &mut on, &mut typed);
			let (changes, _) = typing(&context, &row, key(Key::Enter), &mut on, &mut typed);

			assert!(changes.is_empty(), "{said:?} is not a rename");
			assert_eq!(on, None, "and the field is put away either way");
		}
	}

	#[test]
	fn a_field_that_loses_the_keyboard_without_enter_renames_nothing() {
		// pressing anywhere else puts the field away and asks for nothing,
		// which is the one gesture nobody has to be taught - and the thing
		// that tells this row's enter from any other way of losing the
		// keyboard.
		let context = Context::default();
		let row = entry("meshes/crystal", Kind::Mesh);
		let mut on = Some("meshes/crystal".to_owned());
		let mut typed = "gem".to_owned();

		typing(&context, &row, Vec::new(), &mut on, &mut typed);
		// a press on nothing, which is what clicking away from a field is
		let away = Pos2::new(560.0, 50.0);
		let (changes, _) = typing(
			&context,
			&row,
			vec![
				egui::Event::PointerMoved(away),
				egui::Event::PointerButton {
					pos: away,
					button: PointerButton::Primary,
					pressed: true,
					modifiers: Modifiers::NONE,
				},
				egui::Event::PointerButton {
					pos: away,
					button: PointerButton::Primary,
					pressed: false,
					modifiers: Modifiers::NONE,
				},
			],
			&mut on,
			&mut typed,
		);

		assert!(changes.is_empty(), "a field left alone is not a rename");
		assert_eq!(on, None, "and it is put away all the same");
	}

	#[test]
	fn the_part_of_a_name_a_rename_changes_is_the_last_one() {
		assert_eq!(last("meshes/crystal"), "crystal");
		assert_eq!(last("meshes/props/small/crystal"), "crystal");
		assert_eq!(last("crystal"), "crystal", "a name with no directory is all last part");
		assert_eq!(last(""), "");
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
