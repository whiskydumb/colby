//! The list page: every project, a search box, a sort, and the way to a new
//! one.
//!
//! The shape is the field's: a row of controls along the top - the order, a
//! search box, a small button for a project that is already on disk and a
//! large one for a new one - and the projects under it, pinned ones first.
//! A row is what its `colby.project` says it is, read a moment ago; one whose
//! file is not there says so and stays, with the one thing that can be done
//! about it.
//!
//! Nothing here changes anything: a press is handed back as a [`Change`] and
//! applied by the caller, which is what lets the page be drawn in a test
//! without a window and lets the state be changed in one place.

use std::path::{Path, PathBuf};

use colby_asset::project::ENGINE;
use egui::{
	Align, Button, CentralPanel, ComboBox, Frame, Key, Layout, Panel, RichText, ScrollArea,
	Sense, TextEdit, Ui,
};

use super::{
	Change,
	list::{self, List, Shown, Sort},
};

/// The page's own state: what is typed into it.
#[derive(Debug, Default)]
pub(crate) struct Home {
	/// What is in the search box.
	search: String,

	/// What is in the box a project on disk is added from.
	add_path: String,

	/// Whether that box is open.
	adding: bool,
}

/// What the page shows, borrowed for a frame.
pub(crate) struct View<'a> {
	/// The list, for its order and its folder.
	pub(crate) list: &'a List,

	/// The projects as last read, in order.
	pub(crate) shown: &'a [Shown],

	/// This moment, for how long ago each was opened.
	pub(crate) now: u64,

	/// The engine checkout, for the footer.
	pub(crate) engine: &'a Path,

	/// The last thing worth saying under the list.
	pub(crate) notice: &'a str,

	/// Why the list file could not be read, if it could not.
	pub(crate) broken: Option<&'a str>,
}

impl Home {
	/// Draws the page.
	///
	/// @param ui - the whole window, egui's root layout for this frame
	/// @param view - what to show
	/// @return everything that was pressed, in order
	pub(crate) fn show(&mut self, ui: &mut Ui, view: &View<'_>) -> Vec<Change> {
		let mut changes = Vec::new();

		Panel::top("launcher_top").show(ui, |ui| {
			ui.add_space(6.0);
			self.controls(ui, view, &mut changes);

			if self.adding {
				ui.add_space(4.0);
				self.adder(ui, &mut changes);
			}

			ui.add_space(6.0);
		});

		Panel::bottom("launcher_foot").show(ui, |ui| {
			ui.add_space(4.0);

			if let Some(why) = view.broken {
				ui.label(
					RichText::new(format!(
						"the project list cannot be read and is left alone: {why}"
					))
					.color(ui.visuals().error_fg_color),
				);
			} else if !view.notice.is_empty() {
				ui.label(view.notice);
			}

			ui.label(
				RichText::new(format!(
					"colby {ENGINE}  .  engine {}  .  projects in {}",
					list::spelled(view.engine),
					list::spelled(&view.list.projects_dir)
				))
				.weak()
				.small(),
			);
			ui.add_space(4.0);
		});

		CentralPanel::default().show(ui, |ui| {
			ScrollArea::vertical()
				.auto_shrink([false, false])
				.show(ui, |ui| self.rows(ui, view, &mut changes));
		});

		for path in dropped(ui) {
			changes.push(Change::Add(path));
		}

		changes
	}

	/// The row of controls along the top.
	fn controls(&mut self, ui: &mut Ui, view: &View<'_>, changes: &mut Vec<Change>) {
		ui.horizontal(|ui| {
			if let Some(sort) = sort_box(ui, view.list.sort) {
				changes.push(Change::Sort(sort));
			}

			ui.add(
				TextEdit::singleline(&mut self.search)
					.hint_text("search")
					.desired_width(ui.available_width() - 190.0),
			);

			if ui
				.add(Button::new("add"))
				.on_hover_text("put a project that is already on disk on the list")
				.clicked()
			{
				self.adding = !self.adding;
			}

			if ui
				.add(Button::new(RichText::new("New project").strong()))
				.clicked()
			{
				changes.push(Change::New);
			}
		});
	}

	/// The box a project on disk is added from.
	fn adder(&mut self, ui: &mut Ui, changes: &mut Vec<Change>) {
		ui.horizontal(|ui| {
			ui.label("directory");

			let typed = ui.add(
				TextEdit::singleline(&mut self.add_path)
					.hint_text("a directory holding colby.project, or drop one on this window")
					.desired_width(ui.available_width() - 60.0),
			);
			let entered = typed.lost_focus() && ui.input(|input| input.key_pressed(Key::Enter));

			if (ui.button("add").clicked() || entered) && !self.add_path.trim().is_empty() {
				changes.push(Change::Add(PathBuf::from(self.add_path.trim())));
				self.add_path.clear();
				self.adding = false;
			}
		});
	}

	/// The projects, pinned ones first.
	fn rows(&self, ui: &mut Ui, view: &View<'_>, changes: &mut Vec<Change>) {
		let listed: Vec<&Shown> = view
			.shown
			.iter()
			.filter(|row| row.matches(&self.search))
			.collect();

		if listed.is_empty() {
			ui.add_space(24.0);
			ui.vertical_centered(|ui| {
				ui.label(RichText::new(empty_words(view, &self.search)).weak());
			});

			return;
		}

		let pinned: Vec<&Shown> = listed
			.iter()
			.copied()
			.filter(|row| row.entry.pinned)
			.collect();
		let rest: Vec<&Shown> = listed
			.iter()
			.copied()
			.filter(|row| !row.entry.pinned)
			.collect();

		if !pinned.is_empty() {
			section(ui, "pinned");

			for row in pinned {
				row_of(ui, row, view.now, changes);
			}
		}

		if !rest.is_empty() {
			section(
				ui,
				if view.shown.iter().any(|row| row.entry.pinned) {
					"projects"
				} else {
					""
				},
			);

			for row in rest {
				row_of(ui, row, view.now, changes);
			}
		}
	}
}

/// The order, as a drop-down.
///
/// @return the order chosen, when it is not the one shown
fn sort_box(ui: &mut Ui, current: Sort) -> Option<Sort> {
	let mut sort = current;

	ComboBox::from_id_salt("launcher_sort")
		.selected_text(sort.word())
		.width(80.0)
		.show_ui(ui, |ui| {
			for choice in Sort::ALL {
				ui.selectable_value(&mut sort, choice, choice.word());
			}
		});

	(sort != current).then_some(sort)
}

/// One project.
fn row_of(ui: &mut Ui, row: &Shown, now: u64, changes: &mut Vec<Change>) {
	let path = row.entry.path.clone();
	let found = row.project.is_ok();
	let pin = if row.entry.pinned { "unpin" } else { "pin" };

	let inner = Frame::group(ui.style())
		.inner_margin(8.0)
		.show(ui, |ui| {
			ui.set_width(ui.available_width());

			ui.horizontal(|ui| {
				titles(ui, row, found);
				buttons(ui, row, found, now, changes);
			});
		});

	let response = inner.response.interact(Sense::click());

	if response.double_clicked() && found {
		changes.push(Change::Open(path.clone()));
	}

	response.context_menu(|ui| {
		if found && ui.button("open").clicked() {
			changes.push(Change::Open(path.clone()));
			ui.close();
		}

		if ui.button(pin).clicked() {
			changes.push(Change::Pin(path.clone(), !row.entry.pinned));
			ui.close();
		}

		if ui.button("remove from the list").clicked() {
			changes.push(Change::Remove(path.clone()));
			ui.close();
		}
	});
}

/// The left half of a row: the name and the id, and the path under them.
fn titles(ui: &mut Ui, row: &Shown, found: bool) {
	ui.vertical(|ui| {
		ui.horizontal(|ui| {
			let name = RichText::new(row.name()).strong();
			ui.label(if found { name } else { name.weak() });

			if found {
				ui.label(RichText::new(row.id()).weak());
			} else {
				let why = row
					.project
					.as_ref()
					.err()
					.map_or("", String::as_str);

				ui.label(RichText::new(why).color(ui.visuals().error_fg_color));
			}
		});
		ui.label(
			RichText::new(list::spelled(&row.entry.path))
				.weak()
				.small(),
		);
	});
}

/// The right half of a row: when it was opened, and what can be done with it.
fn buttons(ui: &mut Ui, row: &Shown, found: bool, now: u64, changes: &mut Vec<Change>) {
	let path = &row.entry.path;
	let pin = if row.entry.pinned { "unpin" } else { "pin" };

	ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
		let main = ui.button(if found { "open" } else { "remove" });

		if main.clicked() && found {
			changes.push(Change::Open(path.clone()));
		} else if main.clicked() {
			changes.push(Change::Remove(path.clone()));
		}

		if ui.button(pin).clicked() {
			changes.push(Change::Pin(path.clone(), !row.entry.pinned));
		}

		ui.label(
			RichText::new(list::ago(now, row.entry.last_opened))
				.weak()
				.small(),
		);
	});
}

/// A heading between two groups of rows.
fn section(ui: &mut Ui, title: &str) {
	if !title.is_empty() {
		ui.add_space(6.0);
		ui.label(RichText::new(title).weak().small());
	}

	ui.add_space(2.0);
}

/// What an empty list says, which depends on why it is empty.
fn empty_words(view: &View<'_>, search: &str) -> String {
	if view.shown.is_empty() {
		"no projects yet: make one, or drop a directory holding colby.project on this window"
			.to_owned()
	} else {
		format!("nothing on the list matches {:?}", search.trim())
	}
}

/// Every path dropped on the window this frame.
///
/// A project file dropped is taken as its directory by whoever adds it, the
/// way a typed path is.
fn dropped(ui: &Ui) -> Vec<PathBuf> {
	ui.input(|input| {
		input
			.raw
			.dropped_files
			.iter()
			.map(|file| file.path().to_path_buf())
			.collect()
	})
}
