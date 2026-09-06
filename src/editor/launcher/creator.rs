//! The form page: a template on the left, the setup on the right, Create
//! under it.
//!
//! The same page as the list, replaced rather than a window over it: the
//! launcher is one small window and a form that fits in it wants all of it.
//! What the form can say about itself it says as it is typed - the id
//! following the name, the full path of what would be made, the one sentence
//! that stops Create - so that the button is never pressed to find out.

use egui::{Align, Button, CentralPanel, Checkbox, Grid, Layout, Panel, RichText, TextEdit, Ui};

use super::{Change, list, template::Template, wizard::Wizard};

/// Draws the page.
///
/// @param ui - the whole window, egui's root layout for this frame
/// @param wizard - the form, edited in place
/// @param templates - what can be made
/// @param git - the git on the path, if there is one
/// @param notice - why the last Create did not make anything, if it did not
/// @return what was pressed, if anything
pub(crate) fn show(
	ui: &mut Ui,
	wizard: &mut Wizard,
	templates: &[Template],
	git: Option<&str>,
	notice: &str,
) -> Option<Change> {
	let mut change = None;

	Panel::top("creator_top").show(ui, |ui| {
		ui.add_space(6.0);
		ui.horizontal(|ui| {
			if ui.button("back").clicked() {
				change = Some(Change::Back);
			}

			ui.add_space(8.0);
			ui.heading("New project");
		});
		ui.add_space(6.0);
	});

	Panel::left("creator_templates")
		.resizable(false)
		.default_size(200.0)
		.show(ui, |ui| {
			ui.add_space(6.0);
			ui.label(RichText::new("templates").weak().small());
			ui.add_space(4.0);

			if templates.is_empty() {
				ui.label(RichText::new("none found under the engine's templates/").weak());
			}

			for (index, template) in templates.iter().enumerate() {
				let chosen = wizard.template == index;

				if ui
					.selectable_label(chosen, RichText::new(&template.title).strong())
					.clicked()
				{
					wizard.template = index;
				}

				ui.label(
					RichText::new(&template.description)
						.weak()
						.small(),
				);
				ui.add_space(6.0);
			}
		});

	CentralPanel::default().show(ui, |ui| {
		ui.add_space(6.0);
		ui.label(RichText::new("project setup").weak().small());
		ui.add_space(4.0);

		setup(ui, wizard);

		ui.add_space(10.0);
		ui.label(RichText::new("other").weak().small());
		ui.add_space(4.0);
		ui.checkbox(&mut wizard.vcs_files, "write .gitignore and .gitattributes");

		match git {
			| Some(version) => {
				ui.checkbox(&mut wizard.git_init, format!("git init  ({version})"));
			},
			| None => {
				wizard.git_init = false;
				ui.add_enabled(false, Checkbox::new(&mut wizard.git_init, "git init"))
					.on_disabled_hover_text("git is not on the path");
			},
		}

		if footer(ui, wizard, !templates.is_empty(), notice) {
			change = Some(Change::Create);
		}
	});

	change
}

/// The Create button, what would be made or why it cannot be, and the last
/// refusal; along the bottom.
///
/// @return whether Create was pressed
fn footer(ui: &mut Ui, wizard: &Wizard, templates: bool, notice: &str) -> bool {
	ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
		ui.add_space(4.0);

		let check = wizard.check();
		let pressed = ui
			.horizontal(|ui| {
				let ready = check.is_ok() && templates;
				let pressed = ui
					.add_enabled(ready, Button::new(RichText::new("Create").strong()))
					.clicked();

				verdict(ui, &check);

				pressed
			})
			.inner;

		if !notice.is_empty() {
			ui.label(RichText::new(notice).color(ui.visuals().error_fg_color));
		}

		pressed
	})
	.inner
}

/// What would be made, or the one sentence that stops it.
fn verdict(ui: &mut Ui, check: &Result<std::path::PathBuf, String>) {
	match check {
		| Ok(target) => {
			ui.label(RichText::new(list::spelled(target)).weak());
		},
		| Err(why) => {
			ui.label(RichText::new(why).color(ui.visuals().error_fg_color));
		},
	}
}

/// The name, the id and the location.
fn setup(ui: &mut Ui, wizard: &mut Wizard) {
	Grid::new("creator_setup")
		.num_columns(2)
		.spacing([8.0, 6.0])
		.show(ui, |ui| {
			ui.label("name");
			if ui
				.add(TextEdit::singleline(&mut wizard.name).desired_width(f32::INFINITY))
				.changed()
			{
				wizard.named();
			}
			ui.end_row();

			ui.label("id");
			if ui
				.add(
					TextEdit::singleline(&mut wizard.id)
						.desired_width(f32::INFINITY)
						.hint_text("a-z, 0-9 and _; the folder's name"),
				)
				.changed()
			{
				wizard.id_edited = true;
			}
			ui.end_row();

			ui.label("location");
			ui.add(TextEdit::singleline(&mut wizard.location).desired_width(f32::INFINITY));
			ui.end_row();

			ui.label("");
			ui.checkbox(&mut wizard.remember, "remember as the projects folder");
			ui.end_row();
		});
}
