//! The launcher: every project this person has, and a new one from a template.
//!
//! What `colby` opens when it is started with no project named and none in
//! the working directory. Two pages in one window, the way the field does it:
//! the list, with a search box and a sort and a way to add a project that is
//! already on disk; and the form a new project is made with. Opening a project
//! is not this window becoming the editor - it is the same executable started
//! again with `--project`, and this window closing, which is what keeps the
//! launcher a screen over nothing rather than a second way to bring a world up.
//!
//! **What is checked by running it lives in three modules with no egui in
//! them**: [`list`] is the file and its order, [`template`] is what a new
//! project is copied from, [`wizard`] is the form's rules and what Create
//! does. The two pages, [`home`] and [`creator`], draw those and hand back
//! what was pressed as a [`Change`], and [`Pages`] applies it - so that a
//! click is a value a test can feed in without a window.

use std::{
	path::{Path, PathBuf},
	time::{Duration, Instant},
};

use colby_asset::project::{self, ENGINE};
use colby_core::{error, info, warn};
use colby_engine::Overlay;
use egui::Ui;
use wgpu::{Device, Queue, TextureFormat, TextureView};
use winit::{event::WindowEvent, window::Window};

use crate::shell::Shell;

mod creator;
mod home;
pub mod list;
pub mod template;
pub mod wizard;

use self::{
	list::{List, Shown, Sort},
	template::Template,
	wizard::Wizard,
};

/// How often the projects on the list are read from disk again while nothing
/// on the list changed: often enough that a drive plugged back in is noticed,
/// rarely enough that the list is not what the disk spends its time on.
const REFRESH_INTERVAL: Duration = Duration::from_secs(2);

/// What the launcher asks the process that holds it to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
	/// Open this project in a process of its own, and close.
	Open(PathBuf),
}

/// Something that was pressed on a page.
///
/// A value rather than a call, so that a page is a function from state to
/// intent and the state is changed in one place, [`Pages::apply`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Change {
	/// Open a project.
	Open(PathBuf),

	/// Pin a project to the top, or let it go.
	Pin(PathBuf, bool),

	/// Take a project off the list.
	Remove(PathBuf),

	/// Put a directory on the list, if it is a project.
	Add(PathBuf),

	/// Order the list another way.
	Sort(Sort),

	/// Go to the form.
	New,

	/// Come back from the form with nothing made.
	Back,

	/// Make what the form says.
	Create,
}

/// Which page is up.
#[derive(Debug)]
enum Page {
	/// The list.
	Home,

	/// The form.
	Creator(Wizard),
}

/// The launcher's state, and the two pages over it.
///
/// Everything but the drawing: tests build one of these against a scratch
/// directory and feed it changes.
pub(crate) struct Pages {
	/// The engine checkout, where the templates are.
	engine: PathBuf,

	/// Where the list is kept, if the platform said where that is.
	file: Option<PathBuf>,

	/// The list itself.
	list: List,

	/// Why the list file could not be read, in which case the list is empty
	/// and is never written over: a file this build cannot read is somebody's
	/// to look at, not something to replace.
	broken: Option<String>,

	/// Every template under the engine.
	templates: Vec<Template>,

	/// The git on the path, if there is one.
	git: Option<String>,

	page: Page,
	home: home::Home,

	/// The projects as last read from disk, in order.
	shown: Vec<Shown>,

	/// When they were last read.
	refreshed: Option<Instant>,

	/// The last thing worth saying under whichever page is up.
	notice: String,
}

impl Pages {
	/// Reads the list and finds the templates.
	///
	/// @param engine - the engine checkout
	/// @param file - where the list is kept, or nothing when the platform has
	/// no place for it, in which case it lives for the run
	pub(crate) fn new(engine: &Path, file: Option<PathBuf>) -> Self {
		let fallback = list::default_projects_dir();
		let (list, broken) = match file
			.as_deref()
			.map(|file| List::load(file, &fallback))
		{
			| Some(Ok(list)) => (list, None),
			| Some(Err(error)) => {
				error!(%error, "the project list cannot be read; showing none, and not writing over it");

				(List::new(&fallback), Some(error.to_string()))
			},
			| None => {
				warn!("no config directory; the project list lives for this run only");

				(List::new(&fallback), None)
			},
		};

		let templates = Template::find(engine);

		if templates.is_empty() {
			warn!(
				under = %engine.join(template::DIRECTORY).display(),
				"no templates; the form can list nothing to make"
			);
		}

		Self {
			engine: engine.to_owned(),
			file,
			list,
			broken,
			templates,
			git: wizard::git_version(),
			page: Page::Home,
			home: home::Home::default(),
			shown: Vec::new(),
			refreshed: None,
			notice: String::new(),
		}
	}

	/// Builds whichever page is up, and does what was pressed on it.
	///
	/// @param ui - the whole window, egui's root layout for this frame
	/// @return what the process holding the launcher is asked to do, if
	/// anything
	pub(crate) fn show(&mut self, ui: &mut Ui) -> Option<Action> {
		self.refresh(false);

		let changes = match &mut self.page {
			| Page::Home => {
				let view = home::View {
					list: &self.list,
					shown: &self.shown,
					now: list::now(),
					engine: &self.engine,
					notice: &self.notice,
					broken: self.broken.as_deref(),
				};

				self.home.show(ui, &view)
			},
			| Page::Creator(wizard) =>
				creator::show(ui, wizard, &self.templates, self.git.as_deref(), &self.notice)
					.into_iter()
					.collect(),
		};

		let mut action = None;

		for change in changes {
			if let Some(asked) = self.apply(change) {
				action = Some(asked);
			}
		}

		action
	}

	/// Does what was pressed.
	///
	/// @param change - what
	/// @return what the process is asked to do, if this was that kind of press
	pub(crate) fn apply(&mut self, change: Change) -> Option<Action> {
		match change {
			| Change::Open(path) => {
				self.list.opened(&path, list::now());
				self.save();

				return Some(Action::Open(path));
			},
			| Change::Pin(path, pinned) => {
				self.list.pin(&path, pinned);
				self.save();
			},
			| Change::Remove(path) => {
				if self.list.remove(&path) {
					self.notice = format!(
						"{} taken off the list; nothing on disk was touched",
						list::spelled(&path)
					);
				}

				self.save();
			},
			| Change::Add(path) => self.add(&path),
			| Change::Sort(sort) => {
				self.list.sort = sort;
				self.save();
			},
			| Change::New => {
				self.notice.clear();
				self.page =
					Page::Creator(Wizard::new(&self.list.projects_dir, self.git.is_some()));
			},
			| Change::Back => {
				self.notice.clear();
				self.page = Page::Home;
			},
			| Change::Create => return self.create(),
		}

		self.refresh(true);

		None
	}

	/// Puts a directory on the list, if it holds a project.
	///
	/// A path to the project file itself is taken as its directory, because
	/// that is what a person who typed the file's path meant.
	fn add(&mut self, path: &Path) {
		let dir = if path.is_file()
			&& path
				.file_name()
				.is_some_and(|name| name == project::FILE)
		{
			path.parent()
				.map_or_else(|| path.to_owned(), Path::to_owned)
		} else {
			path.to_owned()
		};

		if !dir.join(project::FILE).is_file() {
			self.notice = format!("{} holds no {}", list::spelled(&dir), project::FILE);

			return;
		}

		self.notice = if self.list.add(&dir) {
			self.save();

			format!("{} is on the list", list::spelled(&dir))
		} else {
			format!("{} was on the list already", list::spelled(&dir))
		};
	}

	/// Makes what the form says, and opens it.
	fn create(&mut self) -> Option<Action> {
		let Page::Creator(wizard) = &self.page else {
			return None;
		};
		let Some(template) = self.templates.get(wizard.template) else {
			"there is no template to make a project from".clone_into(&mut self.notice);

			return None;
		};

		match wizard.create(template, ENGINE) {
			| Ok(dir) => {
				if wizard.remember {
					self.list.projects_dir = PathBuf::from(wizard.location.trim());
				}

				self.list.add(&dir);
				self.list.opened(&dir, list::now());
				self.save();
				self.notice.clear();
				self.page = Page::Home;
				self.refresh(true);

				Some(Action::Open(dir))
			},
			| Err(error) => {
				error!(%error, "the project was not made");
				self.notice = error.to_string();

				None
			},
		}
	}

	/// Writes the list, where there is somewhere to write it.
	fn save(&mut self) {
		if self.broken.is_some() {
			return;
		}

		let Some(file) = self.file.as_deref() else {
			return;
		};

		if let Err(error) = self.list.save(file) {
			error!(%error, "the project list was not saved");
			self.notice = error.to_string();
		}
	}

	/// Reads every project on the list from disk again.
	///
	/// @param force - whether to read now whatever the clock says, because the
	/// list itself changed
	fn refresh(&mut self, force: bool) {
		let due = self
			.refreshed
			.is_none_or(|last| last.elapsed() >= REFRESH_INTERVAL);

		if !force && !due {
			return;
		}

		self.shown = self.list.shown();
		self.refreshed = Some(Instant::now());
	}

	/// Every project as last read, in order.
	#[cfg(test)]
	fn shown(&self) -> &[Shown] { &self.shown }

	/// The list as it stands.
	#[cfg(test)]
	fn list(&self) -> &List { &self.list }
}

/// The launcher: egui against a window, and the pages over it.
pub struct Launcher {
	shell: Shell,
	pages: Pages,
}

impl Launcher {
	/// Brings the launcher up against the window and the device it draws with.
	///
	/// @param window - the window events come from
	/// @param device - the device the frames belong to
	/// @param format - the color format the surface was configured with
	/// @param engine - the engine checkout, where the templates are
	#[must_use]
	pub fn new(window: &Window, device: &Device, format: TextureFormat, engine: &Path) -> Self {
		let pages = Pages::new(engine, list::default_file());

		info!(
			projects = pages.list.entries().len(),
			templates = pages.templates.len(),
			git = pages.git.is_some(),
			"the launcher is up"
		);

		Self {
			shell: Shell::new(window, device, format),
			pages,
		}
	}

	/// Offers one window event to the launcher.
	///
	/// @param window - the window the event came from
	/// @param event - the event
	/// @return whether egui took it
	pub fn on_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
		self.shell.on_event(window, event)
	}

	/// Builds this frame's page.
	///
	/// @param window - the window, for input and for the cursor
	/// @return what the process holding the launcher is asked to do, if
	/// anything was pressed that asks for something
	pub fn run(&mut self, window: &Window) -> Option<Action> {
		let Self { shell, pages } = self;
		let mut action = None;

		shell.run(window, |ui| action = pages.show(ui));

		action
	}
}

impl Overlay for Launcher {
	fn draw(
		&mut self,
		device: &Device,
		queue: &Queue,
		target: &TextureView,
		width: u32,
		height: u32,
	) {
		self.shell
			.draw(device, queue, target, width, height);
	}
}

#[cfg(test)]
mod tests {
	use std::{env, fs};

	use egui::{Context, RawInput};

	use super::*;

	/// A directory nothing else is using.
	fn fresh(name: &str) -> PathBuf {
		let dir = env::temp_dir().join(format!("colby_launcher_{name}"));
		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(&dir).expect("a directory to work in");

		dir
	}

	/// An engine checkout with one template in it, and a project beside it.
	fn engine(root: &Path) -> (PathBuf, PathBuf) {
		let engine = root.join("engine");
		let template = engine.join(template::DIRECTORY).join("tiny");
		fs::create_dir_all(&template).expect("the template's directory");
		fs::write(template.join(template::FILE), r#"{ "title": "Tiny" }"#).expect("the marker");
		fs::write(
			template.join(project::FILE),
			r#"{ "schema": 1, "engine": "$engine", "id": "$id", "name": "$name" }"#,
		)
		.expect("the project file");

		let yard = root.join("yard");
		fs::create_dir_all(&yard).expect("a project's directory");
		fs::write(
			yard.join(project::FILE),
			r#"{ "schema": 1, "engine": "0.1.0", "id": "yard", "name": "The Yard" }"#,
		)
		.expect("its file");

		(engine, yard)
	}

	/// One frame of whichever page is up, with nobody touching it.
	fn frame(pages: &mut Pages) -> Option<Action> {
		let context = Context::default();
		let mut action = None;
		let output = context.run_ui(RawInput::default(), |ui| action = pages.show(ui));

		// epaint refuses to drop a delta nobody applied, and nothing here
		// paints; cleared on purpose, the way the shell does on its way out.
		let mut textures = output.textures_delta;
		textures.clear();

		action
	}

	#[test]
	fn a_project_added_is_on_the_list_and_in_the_file_and_opening_it_is_the_action() {
		let root = fresh("add");
		let (engine, yard) = engine(&root);
		let file = root.join("config").join(list::FILE);
		let mut pages = Pages::new(&engine, Some(file.clone()));

		assert!(pages.list().entries().is_empty(), "nothing yet");

		pages.apply(Change::Add(yard.clone()));

		assert_eq!(pages.list().entries().len(), 1);
		assert_eq!(pages.shown()[0].name(), "The Yard", "read from its file");
		assert!(file.is_file(), "and written");

		pages.apply(Change::Add(yard.join(project::FILE)));

		assert_eq!(pages.list().entries().len(), 1, "the file's path is the same project");
		assert!(pages.notice.contains("already"), "{}", pages.notice);

		pages.apply(Change::Add(root.join("nowhere")));

		assert_eq!(pages.list().entries().len(), 1, "a directory with no project is not added");
		assert!(pages.notice.contains(project::FILE), "{}", pages.notice);

		assert_eq!(pages.apply(Change::Open(yard.clone())), Some(Action::Open(yard.clone())));
		assert!(pages.list().entries()[0].last_opened > 0, "and noted as opened");

		pages.apply(Change::Pin(yard.clone(), true));
		assert!(pages.list().entries()[0].pinned);

		pages.apply(Change::Sort(Sort::Name));
		assert_eq!(pages.list().sort, Sort::Name);

		let again = Pages::new(&engine, Some(file));

		assert_eq!(again.list(), pages.list(), "everything came back from the file");

		pages.apply(Change::Remove(yard.clone()));
		assert!(pages.list().entries().is_empty());
		assert!(yard.join(project::FILE).is_file(), "the project itself is untouched");
	}

	#[test]
	fn the_form_makes_a_project_puts_it_on_the_list_and_opens_it() {
		let root = fresh("create");
		let (engine, _) = engine(&root);
		let projects = root.join("projects");
		let file = root.join("config").join(list::FILE);
		let mut pages = Pages::new(&engine, Some(file));
		pages.list.projects_dir.clone_from(&projects);

		assert!(matches!(pages.page, Page::Home));
		assert_eq!(pages.apply(Change::New), None);

		let Page::Creator(wizard) = &mut pages.page else {
			panic!("the form is up");
		};

		wizard.name = "The Yard".to_owned();
		wizard.named();
		wizard.git_init = false;

		let made = projects.join("the_yard");

		assert_eq!(pages.apply(Change::Create), Some(Action::Open(made.clone())));
		assert!(made.join(project::FILE).is_file(), "made");
		assert!(made.join(".gitignore").is_file(), "with its git files");
		assert!(matches!(pages.page, Page::Home), "and back on the list");
		assert_eq!(pages.shown()[0].name(), "The Yard");
		assert!(pages.shown()[0].entry.last_opened > 0);

		assert_eq!(pages.apply(Change::New), None);
		let Page::Creator(wizard) = &mut pages.page else {
			panic!("the form is up again");
		};
		wizard.name = "The Yard".to_owned();
		wizard.named();
		wizard.git_init = false;

		assert_eq!(pages.apply(Change::Create), None, "the same again is refused");
		assert!(pages.notice.contains("already exists"), "{}", pages.notice);
		assert!(matches!(pages.page, Page::Creator(_)), "and the form stays up to be fixed");

		assert_eq!(pages.apply(Change::Back), None);
		assert!(matches!(pages.page, Page::Home));
	}

	#[test]
	fn remembering_the_location_moves_where_the_next_project_goes() {
		let root = fresh("remember");
		let (engine, _) = engine(&root);
		let elsewhere = root.join("elsewhere");
		let mut pages = Pages::new(&engine, None);
		pages.apply(Change::New);

		let Page::Creator(wizard) = &mut pages.page else {
			panic!("the form is up");
		};
		wizard.location = list::spelled(&elsewhere);
		wizard.remember = true;
		wizard.git_init = false;
		// the id the form settled on, rather than `my_project`: the default
		// name steps past folders taken in the *first* location, which is the
		// person's real projects folder, and a project made there for real
		// once turned this test red.
		let id = wizard.id.clone();

		let made = pages.apply(Change::Create);

		assert_eq!(made, Some(Action::Open(elsewhere.join(&id))));
		assert_eq!(pages.list().projects_dir, elsewhere);
	}

	#[test]
	fn a_list_file_this_build_cannot_read_is_shown_as_such_and_never_written_over() {
		let root = fresh("broken");
		let (engine, yard) = engine(&root);
		let file = root.join("config").join(list::FILE);
		fs::create_dir_all(file.parent().expect("a parent")).expect("the directory");
		fs::write(&file, r#"{ "schema": 1, "thumbs": true }"#).expect("a file from later");

		let mut pages = Pages::new(&engine, Some(file.clone()));

		assert!(
			pages
				.broken
				.as_deref()
				.is_some_and(|why| why.contains("thumbs"))
		);

		pages.apply(Change::Add(yard));

		assert_eq!(
			fs::read_to_string(&file).expect("still there"),
			r#"{ "schema": 1, "thumbs": true }"#,
			"left exactly as it was"
		);
	}

	#[test]
	fn both_pages_draw_a_frame_nobody_touches_and_ask_for_nothing() {
		let root = fresh("frames");
		let (engine, yard) = engine(&root);
		let mut pages = Pages::new(&engine, None);
		pages.apply(Change::Add(yard));
		pages.apply(Change::Add(root.join("gone")));
		pages.list.add(&root.join("gone"));

		assert_eq!(frame(&mut pages), None, "the list");

		pages.apply(Change::New);

		assert_eq!(frame(&mut pages), None, "the form");
		assert!(matches!(pages.page, Page::Creator(_)), "still up");
	}
}
