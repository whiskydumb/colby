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
				RichText::new("drag one into the picture, or open a scene by double-clicking")
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
		// is the one kind a double-click opens. The other kinds are dragged
		// into the picture and nothing else.
		if entry.kind == Kind::Scene && response.double_clicked() {
			changes.push(Change::Open { name: entry.name.clone() });
		}

		// and a material is the one kind there is something to *edit*: it has
		// a field table, so the inspector can draw it, and it is the only
		// asset that does.
		if entry.kind == Kind::Material && response.clicked() {
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

	use egui::{Context, Pos2, RawInput, Rect};

	use super::*;

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
