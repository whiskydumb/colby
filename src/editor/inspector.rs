//! One thing in the world at a time, in detail.
//!
//! **The inspector is one function over a table.** Every record with a
//! [`Field`] table is shown by [`inspect`]: a row per plain field, a widget
//! the field's kind decides, and a write only when a number actually moved.
//! Nothing here knows what a body's fields are; a field added to the table is
//! a row here the same day, which is the whole reason the table exists. What
//! is still drawn by hand is a relationship - what a joint holds, what an
//! entity hangs off - because a handle is worth showing by name and a table
//! knows nothing about names.
//!
//! **An entity's records are the same rows over a table the engine does not
//! know the type of**, @ref [`records`]: the engine's own `drawing` and a
//! game's `door` are drawn by one loop, from the fields their declarations
//! named, and written back field by field through the entity's handle.
//!
//! **Every write is written down first**, @ref [`History::begin`]: a row is
//! edited on a copy of the record, and the copy goes back into the world only
//! after the world as it stood has been captured, so that a number dragged
//! in a field is one step back however many frames the drag took.
//!
//! Everything that can be tested lives in [`select`](crate::select); what is
//! here is the drawing, and it is checked by looking at it - except that an
//! inspector nobody touches writes nothing, which a headless frame can check.

use colby_asset::Project;
use colby_core::{
	abi::{
		Body, BodyId, Decal, Emitter, EntityId, Field, Joint, JointId, Light, Material,
		MaterialId, ModelId, Post, Renderable, Sky, Terrain, TextureId, Transform, World,
		field::Value,
		scene::{self, Stage},
	},
	glam::{EulerRot, Quat, Vec2, Vec3},
};
use egui::{ComboBox, DragValue, Grid, Label, RichText, ScrollArea, Ui};

use crate::{
	history::History,
	select::{self, Pick, Selection},
};

/// How far a drag of one pixel moves three numbers.
const MOVE_SPEED: f32 = 0.02;

/// How far a drag of one pixel turns something, in degrees.
const TURN_SPEED: f32 = 0.5;

/// How far a drag of one pixel moves a number on its own.
const STEP_SPEED: f32 = 0.01;

/// Draws the inspector into a panel.
///
/// @param ui - the panel
/// @param world - the tables to show, and to edit
/// @param selection - what is selected; the primary is shown
/// @param history - where a write is written down, so that it can be undone
/// @param rename - whether the name field is to take the keyboard this
/// frame, because somebody pressed the key for it
pub(crate) fn show(
	ui: &mut Ui,
	world: &mut World,
	selection: &Selection,
	history: &mut History,
	rename: bool,
	project: Option<&Project>,
) {
	ScrollArea::vertical()
		.auto_shrink([false, false])
		.show(ui, |ui| {
			if selection.len() > 1 {
				ui.label(format!(
					"{} selected: the last picked is shown, and a drag moves all of them",
					selection.len()
				));
				ui.separator();
			}

			detail(ui, world, selection.at(), history, rename, project);
		});
}

/// The selected thing, in detail.
fn detail(
	ui: &mut Ui,
	world: &mut World,
	pick: Pick,
	history: &mut History,
	rename: bool,
	project: Option<&Project>,
) {
	match pick {
		| Pick::Nothing => {
			ui.label("nothing selected, so this is the world itself");
			ui.separator();
			settings(ui, world, history);
		},
		| Pick::Entity(id) => {
			naming(ui, world, pick, history, rename);
			hanging(ui, world, id);
			placing(ui, world, pick, history);
			look(ui, world, id, history);
			lamp(ui, world, id, history);
			thrower(ui, world, id, history);
			land(ui, world, id, history);
			paint(ui, world, id, history);
			records(ui, world, id, history);
		},
		| Pick::Body(id) => {
			naming(ui, world, pick, history, rename);
			ui.label(world.bodies.get(id).map_or_else(
				|| "gone".to_owned(),
				|body| format!("a {}", select::body_words(body)),
			));
			placing(ui, world, pick, history);
			solid(ui, world, id, history);
		},
		| Pick::Joint(id) => {
			naming(ui, world, pick, history, rename);
			tie(ui, world, id, history);
		},
		| Pick::Material(id) => coat(ui, world, id),
		| Pick::Model(id) => made_of(ui, world, id, project),
	}

	ui.separator();
	// the honest answer to "why did my drag not stick", which used to be a
	// workaround written on the panel and is now a mode. @ref
	// `colby_core::abi::World::editing`.
	ui.label(if world.editing {
		"editing, so these are yours"
	} else {
		"playing, so the game may write these back every step. F5 to edit"
	});
}

/// The name field.
///
/// @param rename - whether it takes the keyboard this frame: F2's whole job
fn naming(ui: &mut Ui, world: &mut World, pick: Pick, history: &mut History, rename: bool) {
	let mut name = pick.name(world).to_owned();

	ui.horizontal(|ui| {
		ui.label("name");

		let response = ui.text_edit_singleline(&mut name);

		if rename {
			response.request_focus();
		}

		if response.changed() {
			history.begin("rename", world);
			select::rename(world, pick, &name);
		}
	});
}

/// What an entity hangs off, which is read rather than written here.
///
/// Hanging one entity off another is the hierarchy's job - a row dragged
/// onto another - and the fact is shown here so that the numbers under it
/// read right: a child's place is inside its parent.
fn hanging(ui: &mut Ui, world: &World, id: EntityId) {
	let parent = world.entities.parent(id);

	if !parent.is_some() {
		return;
	}

	ui.horizontal(|ui| {
		ui.label("inside");
		ui.monospace(select::entity_label(world, parent));
	});
}

/// Position, rotation and scale, for the things that have them.
///
/// In the thing's own terms - inside its parent, for an entity that hangs off
/// one - because that is what a person expects to type; the gizmo in the
/// viewport works in the world. @ref `select::local`.
fn placing(ui: &mut Ui, world: &mut World, pick: Pick, history: &mut History) {
	let Some(transform) = select::local(world, pick) else {
		return;
	};

	let mut edited = transform;

	if inspect(ui, "transform", &mut edited, Transform::FIELDS) {
		history.begin("place", world);
		select::place_local(world, pick, edited);
	}
}

/// What an entity looks like: the plain half of its renderable, which is the
/// tint. The mesh, the material and the pose are handles, and the tree names
/// the mesh in the row above.
fn look(ui: &mut Ui, world: &mut World, id: EntityId, history: &mut History) {
	let Some(mut renderable) = world.entities.renderable(id).copied() else {
		return;
	};

	if inspect(ui, "renderable", &mut renderable, Renderable::FIELDS) {
		history.begin("tint", world);
		world.entities.set_renderable(id, renderable);
	}
}

/// The world's own settings, shown when nothing in it is.
///
/// **Where an environment lives when there is no node to hang it on.** Godot
/// puts one on a `WorldEnvironment` and Unreal on a volume; colby has neither,
/// and the honest place for "the clear color, the sun, the ambient, the
/// gravity and the sky" is the panel that is otherwise empty. It is also the
/// first thing "nothing selected" has ever said that is worth reading.
///
/// The camera and the clock are in the record and are deliberately not
/// written back. @ref `colby_core::abi::scene::set_settings`.
fn settings(ui: &mut Ui, world: &mut World, history: &mut History) {
	let mut stage = scene::settings(world);
	let mut moved = inspect(ui, "world", &mut stage, Stage::FIELDS);

	moved |= inspect(ui, "sky", &mut stage.sky, Sky::FIELDS);
	moved |= inspect(ui, "post", &mut stage.post, Post::FIELDS);

	if moved {
		history.begin("world", world);
		scene::set_settings(world, stage);
	}
}

/// What an entity shines, if anything.
///
/// Always drawn, whatever the entity is: turning a crate into a lamp is
/// picking a word in a drop-down, and a section that appeared only for
/// entities that were already lights would leave nowhere to do it. Every
/// field is plain, so the whole of it is the table. @ref
/// `colby_core::abi::light`.
fn lamp(ui: &mut Ui, world: &mut World, id: EntityId, history: &mut History) {
	let Some(mut light) = world.entities.light(id).copied() else {
		return;
	};

	if inspect(ui, "light", &mut light, Light::FIELDS) {
		history.begin("light", world);
		world.entities.set_light(id, light);
	}
}

/// What an entity throws off, if anything.
///
/// Always drawn, for the light's reason and word for word: turning a crate
/// into a fire is picking a word in a drop-down, and a section that appeared
/// only for entities that were already emitters would leave nowhere to do it.
/// @ref `colby_core::abi::particles`.
fn thrower(ui: &mut Ui, world: &mut World, id: EntityId, history: &mut History) {
	let Some(mut emitter) = world.entities.emitter(id).copied() else {
		return;
	};

	if inspect(ui, "emitter", &mut emitter, Emitter::FIELDS) {
		history.begin("emitter", world);
		world.entities.set_emitter(id, emitter);
	}
}

/// What ground an entity is, if any.
///
/// Always drawn, for the light's reason a third time. **What a change here
/// costs is not a field write**: the runtime notices the new record on the next
/// step and rebuilds the mesh, so dragging `side` through five hundred rebuilds
/// half a million triangles once per drag event. That is what the ceiling in
/// `colby_core::abi::terrain` is for, and it is the reason this section has no
/// live preview of its own - the world *is* the preview.
fn land(ui: &mut Ui, world: &mut World, id: EntityId, history: &mut History) {
	let Some(mut terrain) = world.entities.terrain(id).copied() else {
		return;
	};

	if inspect(ui, "terrain", &mut terrain, Terrain::FIELDS) {
		history.begin("terrain", world);
		world.entities.set_terrain(id, terrain);
	}
}

/// What an entity paints, if anything, and whether it is painted itself.
///
/// Always drawn, for the light's reason: turning a block into a decal is
/// picking a word in a drop-down. The word against decals sits under it,
/// because it is the other half of the same question and nothing else on the
/// panel is about paint. @ref `colby_core::abi::decal`.
fn paint(ui: &mut Ui, world: &mut World, id: EntityId, history: &mut History) {
	if let Some(mut decal) = world.entities.decal(id).copied()
		&& inspect(ui, "decal", &mut decal, Decal::FIELDS)
	{
		history.begin("decal", world);
		world.entities.set_decal(id, decal);
	}

	let mut takes = world.entities.takes_decals(id);

	if ui
		.checkbox(&mut takes, "decals paint it")
		.changed()
	{
		history.begin("decals paint it", world);
		world.entities.set_takes_decals(id, takes);
	}
}

/// What an entity's records hold: a grid for each declared record, and a
/// line for whatever waits for a record nobody has declared.
///
/// **Every record, whatever the entity is**, for the light's reason: every
/// entity carries every record, and a section that appeared only for an entity
/// that was already a door would leave nowhere to make one. Nothing here knows
/// what a record's fields are; the engine's and a game's are drawn alike, from
/// the declaration the host copied. @ref `colby_core::abi::record`.
///
/// What waits is shown and not edited: it has no kind until the record that
/// takes it is declared, and it is kept so that a world written down while its
/// game is not loaded loses nothing.
fn records(ui: &mut Ui, world: &mut World, id: EntityId, history: &mut History) {
	for table in 0..world.entities.records().tables().len() {
		record(ui, world, id, table, history);
	}

	let mut waiting: Vec<(&str, usize)> = Vec::new();

	for one in world.entities.waiting(id) {
		match waiting
			.iter_mut()
			.find(|(record, _)| *record == one.record)
		{
			| Some((_, many)) => *many += 1,
			| None => waiting.push((&one.record, 1)),
		}
	}

	for (record, many) in waiting {
		ui.label(format!(
			"{record}: {} kept for a record nobody has declared",
			counted(many, "value")
		));
	}
}

/// One declared record of one entity: its name, and a row a field.
///
/// Its columns are copied out first, because the grid writes the world a row
/// at a time and a borrow of the table would hold the world for the whole of
/// it.
///
/// @param table - the record's place in the declared records
fn record(ui: &mut Ui, world: &mut World, id: EntityId, table: usize, history: &mut History) {
	let Some((name, help, columns)) = world
		.entities
		.records()
		.tables()
		.get(table)
		.map(|held| (held.name().to_owned(), held.help().to_owned(), held.columns().to_vec()))
	else {
		return;
	};

	ui.label(RichText::new(&name).strong())
		.on_hover_text(help);

	Grid::new(format!("record {name}"))
		.num_columns(2)
		.show(ui, |ui| {
			for (index, column) in columns.iter().enumerate() {
				let Some(held) = world.entities.field(id, table, index) else {
					continue;
				};
				let words: Vec<&str> = column
					.kind()
					.words()
					.iter()
					.map(String::as_str)
					.collect();
				let mut value = held.clone();

				ui.label(column.name())
					.on_hover_text(column.help());
				widget(ui, &format!("{name}.{}", column.name()), &words, &mut value);

				if value != held {
					history.begin("record", world);
					world.entities.set_field(id, table, index, &value);
				}

				ui.end_row();
			}
		});
}

/// Everything the solver reads about a body, to edit.
///
/// Its place is the row above, through the transform's own table, and the
/// entity it drives is the branch it hangs under in the tree.
fn solid(ui: &mut Ui, world: &mut World, id: BodyId, history: &mut History) {
	let Some(mut body) = world.bodies.get(id).copied() else {
		return;
	};

	if inspect(ui, "body", &mut body, Body::FIELDS) {
		history.begin("body", world);

		if let Some(held) = world.bodies.get_mut(id) {
			*held = body;
		}
	}
}

/// What a joint holds, by name, and then everything else about it.
///
/// Its two bodies are handles, which the table describes and cannot name, so
/// the two rows that name them are drawn here; the anchors are in each body's
/// own space, and are numbers all the same.
fn tie(ui: &mut Ui, world: &mut World, id: JointId, history: &mut History) {
	let Some(mut joint) = world.joints.get(id).copied() else {
		return;
	};

	Grid::new("held").num_columns(2).show(ui, |ui| {
		ui.label("first");
		ui.monospace(select::body_label(world, joint.first));
		ui.end_row();

		ui.label("second");
		ui.monospace(if joint.second.is_some() {
			select::body_label(world, joint.second)
		} else {
			"a point in the world".to_owned()
		});
		ui.end_row();
	});

	if inspect(ui, "joint", &mut joint, Joint::FIELDS) {
		history.begin("joint", world);

		if let Some(held) = world.joints.get_mut(id) {
			*held = joint;
		}
	}
}

/// A material: its two pictures by name, then everything else by table.
///
/// **No history.** Undo here is a snapshot of the *world*, and a material is a
/// file the world happens to have loaded - a step back that put the old
/// numbers into the registry and left the file holding the new ones would be
/// a world disagreeing with its own assets. What takes an edit back is the
/// same thing that takes any file edit back, and the button below is what puts
/// one on disk in the first place.
///
/// @param ui - where to draw
/// @param world - the registry to read and write
/// @param id - which material
fn coat(ui: &mut Ui, world: &mut World, id: MaterialId) {
	let Some(mut material) = world.materials.get(id).copied() else {
		return;
	};
	let name = world.materials.name(id).to_owned();

	ui.monospace(&name);
	ui.separator();

	let mut edited = false;
	let mut albedo = world
		.textures
		.get(material.albedo)
		.map_or_else(String::new, |entry| entry.name().to_owned());
	let mut normal = world
		.textures
		.get(material.normal)
		.map_or_else(String::new, |entry| entry.name().to_owned());

	Grid::new("pictures")
		.num_columns(2)
		.show(ui, |ui| {
			ui.label("albedo").on_hover_text(PICTURE);
			edited |= named(ui, world, &mut albedo, &mut material.albedo);
			ui.end_row();

			ui.label("normal").on_hover_text(PICTURE);
			edited |= named(ui, world, &mut normal, &mut material.normal);
			ui.end_row();
		});

	edited |= inspect(ui, "material", &mut material, Material::FIELDS);

	if edited && let Some(held) = world.materials.get_mut(id) {
		*held = material;
	}

	ui.separator();

	// through the console, because writing a file is the runner's business and
	// a console line is the one way across that already exists. The same
	// bargain the bar's `scene.write` takes. @ref `colby_runtime::saves`.
	if ui
		.button("write to assets/")
		.on_hover_text("puts these numbers back in the .material this came from")
		.clicked()
		&& !name.is_empty()
	{
		colby_core::abi::console::run(world, &format!("material.write {name}"));
	}
}

/// A model: what came out of the file, and what a sidecar had to do with it.
///
/// **The one panel that reads rather than writes**, and that is what a model
/// is: a `.cmodel` is derived, every number in it was decided by the exporter
/// or by the sidecar, and a row edited here would be a row the next compile
/// throws away. What can be changed is the file beside the source, and the
/// button offers to start one.
///
/// @param ui - the panel
/// @param world - the registries, read only
/// @param id - which model
/// @param project - whose asset tree, for the sidecar beside the source
fn made_of(ui: &mut Ui, world: &mut World, id: ModelId, project: Option<&Project>) {
	let name = world.models.name(id).to_owned();

	ui.monospace(&name);
	ui.separator();

	// copied out because the button below needs the world to run a console
	// line, and a borrowed slice of it would still be alive by then. The
	// model's own name comes off what it names: five rows saying
	// `models/lamp/` under a panel titled `models/lamp` is five copies of one
	// word, and what is left - `column`, `brass` - is what tells them apart. A
	// name from *outside* the model keeps the whole of itself, which is
	// exactly the one worth reading in full.
	let inside = format!("{name}/");
	let pieces: Vec<(String, String, String)> = world
		.models
		.placements(id)
		.iter()
		.map(|placement| {
			let mesh = world
				.meshes
				.get(placement.mesh)
				.map_or("", |entry| entry.name());

			(
				placement.name.clone(),
				within(mesh, &inside),
				within(world.materials.name(placement.material), &inside),
			)
		})
		.collect();

	ui.label(format!("{} standing", counted(pieces.len(), "piece")));

	// **both ways, and the horizontal one is not a nicety.** A `Grid` asks for
	// the width its widest row wants, and a panel that grants it is a panel
	// that grew - which is what picking a model in the browser did: the
	// inspector widened and took the width out of the picture. Inside a
	// scrolling area the grid asks for nothing.
	ScrollArea::both()
		.max_height(PIECES)
		.auto_shrink([false, false])
		.show(ui, |ui| {
			Grid::new("pieces")
				.num_columns(3)
				.striped(true)
				.show(ui, |ui| {
					for (piece, mesh, material) in &pieces {
						ui.label(piece);
						ui.monospace(mesh);
						ui.monospace(material);
						ui.end_row();
					}
				});
		});

	ui.separator();
	guiding(ui, world, &name, project);
}

/// How tall the list of pieces may get before it scrolls, in points.
const PIECES: f32 = 220.0;

/// An asset name with the model's own prefix taken off, when it has one.
///
/// @param name - the asset name
/// @param inside - the model's name and a slash
fn within(name: &str, inside: &str) -> String {
	name.strip_prefix(inside)
		.unwrap_or(name)
		.to_owned()
}

/// The sidecar beside the model's source, and what it says.
///
/// Read off disk each frame it is looked at rather than kept: it is a few
/// hundred bytes, it is only read while a model is the selection, and the
/// alternative is a cache that goes stale the moment somebody edits the file
/// in the editor they actually edit it in. The browser rescans a whole tree
/// once a second for the same reason.
///
/// @param ui - the panel
/// @param world - the world a console line is run against
/// @param name - the model's asset name
/// @param project - whose asset tree
fn guiding(ui: &mut Ui, world: &mut World, name: &str, project: Option<&Project>) {
	ui.label("import");

	let Some(project) = project else {
		ui.weak("no project, so there is no source to stand beside");

		return;
	};
	let Some(source) = colby_asset::import::source_of(&project.assets(), name) else {
		ui.weak("no source under that name in assets/");

		return;
	};
	let sidecar = colby_asset::import::beside(&source);
	let shown = sidecar
		.strip_prefix(project.root())
		.unwrap_or(&sidecar)
		.display()
		.to_string();

	match std::fs::read_to_string(&sidecar) {
		| Ok(text) => {
			ui.monospace(shown);
			// wrapped, and that is not a nicety either: `ui.code` measures a
			// long line as the width it wants, a panel that grants it is a
			// panel that grew, and a sidecar's longest line is a remap - the
			// one thing worth reading in full and the one thing that is long.
			ui.add(Label::new(RichText::new(text.trim()).monospace().code()).wrap());
		},
		| Err(_) => {
			ui.weak("none: the file compiles as the exporter wrote it");

			if ui
				.button("start one")
				.on_hover_text(format!(
					"writes {shown}, which changes nothing until it is edited"
				))
				.clicked()
			{
				colby_core::abi::console::run(world, &format!("model.write {name}"));
			}
		},
	}
}

/// A count with its noun, pluralized the one way English mostly is.
fn counted(many: usize, noun: &str) -> String {
	if many == 1 {
		format!("1 {noun}")
	} else {
		format!("{many} {noun}s")
	}
}

/// What both picture rows say when hovered.
const PICTURE: &str = "the asset name of a compiled texture, or empty for none";

/// One texture named by hand, because a handle has no spelling.
///
/// The row is a field of text and the handle follows it: a name the registry
/// answers to is taken, and one it does not is left in the field for the
/// person to finish typing. **Not refused and not cleared** - a half-typed
/// name is not an error, and clearing the handle on every keystroke would
/// take the picture off the ball while somebody spells its name.
///
/// @param ui - where to draw
/// @param world - the registry to look a name up in
/// @param typed - the text field's contents, kept across frames by the caller
/// @param handle - the material's own field, written when the name resolves
/// @return whether the handle moved
fn named(ui: &mut Ui, world: &World, typed: &mut String, handle: &mut TextureId) -> bool {
	ui.text_edit_singleline(typed).changed() && resolve(world, typed, handle)
}

/// The rule behind that row, with no widget in it.
///
/// Empty clears the handle; a name the registry answers to takes it; anything
/// else leaves both alone, which is what a name somebody is halfway through
/// typing is.
///
/// @param world - the registry to look a name up in
/// @param typed - what is in the field
/// @param handle - the material's own field, written when the name resolves
/// @return whether the handle moved
fn resolve(world: &World, typed: &str, handle: &mut TextureId) -> bool {
	if typed.trim().is_empty() {
		let moved = handle.is_some();
		*handle = TextureId::NONE;

		return moved;
	}

	let found = world.textures.find(typed.trim());
	if !found.is_some() || found == *handle {
		return false;
	}

	*handle = found;

	true
}

/// One inspector over any record with a table: a row per plain field, and a
/// widget the field's kind decides.
///
/// A reference is left out. The table says a body drives an entity, and a row
/// that could only show a slot number would say less than the tree already
/// does by nesting one under the other; where a relationship is worth a row,
/// the caller draws it by name, @ref [`tie`].
///
/// **A write is guarded by the numbers having actually changed**, field by
/// field, and that matters more than it looks for a rotation: two different
/// triples of angles can name one rotation, so converting out and straight
/// back in every frame would walk a rotation somewhere it was never dragged.
///
/// @param ui - where to draw
/// @param salt - what tells this grid from another in the same panel
/// @param record - what to show and edit
/// @param fields - its table
/// @return whether any field was written
fn inspect<T>(ui: &mut Ui, salt: &str, record: &mut T, fields: &[Field<T>]) -> bool {
	let mut edited = false;

	Grid::new(salt).num_columns(2).show(ui, |ui| {
		for field in fields {
			if field.kind.is_reference() {
				continue;
			}

			ui.label(field.name).on_hover_text(field.help);

			let held = field.get(record);
			let mut value = held.clone();
			widget(ui, field.name, field.kind.words(), &mut value);

			if value != held && field.set(record, value) {
				edited = true;
			}

			ui.end_row();
		}
	});

	edited
}

/// The widget one value is edited with, by its kind.
///
/// @param ui - where to draw
/// @param salt - an id no other widget in the panel has: the field's name
/// @param words - the words a word may be, and none for any other value
/// @param value - what to draw and edit in place
fn widget(ui: &mut Ui, salt: &str, words: &[&str], value: &mut Value) {
	match value {
		| Value::Bool(held) => {
			ui.checkbox(held, "");
		},
		| Value::Int(held) => {
			ui.add(DragValue::new(held));
		},
		| Value::Float(held) => {
			ui.add(DragValue::new(held).speed(STEP_SPEED));
		},
		| Value::Text(held) => {
			ui.text_edit_singleline(held);
		},
		| Value::Vec2(held) => couple(ui, held),
		| Value::Vec3(held) => vector(ui, held, MOVE_SPEED),
		| Value::Quat(held) => turn(ui, held),
		| Value::Color(held) => color(ui, held),
		| Value::Word(held) => pick(ui, salt, words, held),
		// never reached: a reference is skipped before a widget is asked for,
		// @ref `inspect`. A slot number is what there would be to show.
		| Value::Entity(_)
		| Value::Body(_)
		| Value::Joint(_)
		| Value::Pose(_)
		| Value::Mesh(_)
		| Value::Material(_)
		| Value::Texture(_) => {
			ui.monospace("a reference");
		},
	}
}

/// Two numbers on one row.
///
/// Its own speed rather than [`MOVE_SPEED`]: the only pair in the tables is a
/// texture's repeat, and a tenth of a tile a pixel is a drag nobody can aim.
fn couple(ui: &mut Ui, value: &mut Vec2) {
	ui.horizontal(|ui| {
		ui.add(
			DragValue::new(&mut value.x)
				.speed(STEP_SPEED)
				.prefix("u "),
		);
		ui.add(
			DragValue::new(&mut value.y)
				.speed(STEP_SPEED)
				.prefix("v "),
		);
	});
}

/// Three numbers on one row.
fn vector(ui: &mut Ui, value: &mut Vec3, speed: f32) {
	ui.horizontal(|ui| {
		ui.add(
			DragValue::new(&mut value.x)
				.speed(speed)
				.prefix("x "),
		);
		ui.add(
			DragValue::new(&mut value.y)
				.speed(speed)
				.prefix("y "),
		);
		ui.add(
			DragValue::new(&mut value.z)
				.speed(speed)
				.prefix("z "),
		);
	});
}

/// A rotation as three angles in degrees.
///
/// A quaternion is not a thing anyone types, so this shows the yaw, pitch and
/// roll it stands for. **The write is guarded by the numbers having actually
/// changed**, and that matters more than it looks: two different triples can
/// name one rotation, so converting out and straight back in every frame would
/// walk a rotation somewhere it was never dragged.
fn turn(ui: &mut Ui, rotation: &mut Quat) {
	let (yaw, pitch, roll) = rotation.to_euler(EulerRot::YXZ);
	let held = Vec3::new(pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees());
	let mut edited = held;

	vector(ui, &mut edited, TURN_SPEED);

	if edited != held {
		*rotation = Quat::from_euler(
			EulerRot::YXZ,
			edited.y.to_radians(),
			edited.x.to_radians(),
			edited.z.to_radians(),
		);
	}
}

/// A color, as a swatch that opens a picker.
///
/// Guarded by the widget's own word rather than by comparing numbers: the
/// picker keeps its color as hue, saturation and value and writes the three
/// channels back from that every frame, which is a round trip through a
/// different number of bits, and comparing would see a change where nobody
/// made one.
fn color(ui: &mut Ui, value: &mut Vec3) {
	let mut rgb = value.to_array();

	if ui.color_edit_button_rgb(&mut rgb).changed() {
		*value = Vec3::from_array(rgb);
	}
}

/// One of a few words, as a drop-down over the field's own list.
fn pick(ui: &mut Ui, salt: &str, list: &[&str], held: &mut u32) {
	let shown = usize::try_from(*held)
		.ok()
		.and_then(|index| list.get(index))
		.copied()
		.unwrap_or("?");

	ComboBox::from_id_salt(salt)
		.selected_text(shown)
		.show_ui(ui, |ui| {
			for (index, word) in list.iter().enumerate() {
				if let Ok(index) = u32::try_from(index) {
					ui.selectable_value(held, index, *word);
				}
			}
		});
}

#[cfg(test)]
mod tests {
	use colby_core::abi::{MeshId, Shape};
	use egui::{Context, Pos2, RawInput, Rect, vec2};

	use super::*;

	/// Runs one frame of an inspector over a record with nobody touching it,
	/// and hands back what the frame reported.
	fn untouched<T: Clone + PartialEq + core::fmt::Debug>(
		record: &T,
		fields: &[Field<T>],
	) -> bool {
		let context = Context::default();
		let mut edited = record.clone();
		let mut written = false;

		// nothing paints this frame, and epaint asserts that a texture delta is
		// applied rather than dropped - the right rule for a painter and the
		// wrong one for a test - so the delta is cleared on purpose, the way
		// the editor does on its way out.
		let mut output = context.run_ui(RawInput::default(), |ui| {
			written = inspect(ui, "test", &mut edited, fields);
		});
		output.textures_delta.clear();

		assert_eq!(edited, *record, "nothing was touched, so nothing may have moved");

		written
	}

	/// Every piece of text a frame painted, for a test that asks what a panel
	/// shows rather than what it wrote.
	fn painted(shapes: &[egui::epaint::ClippedShape]) -> Vec<String> {
		let mut texts = Vec::new();
		let mut open: Vec<&egui::Shape> = shapes
			.iter()
			.map(|clipped| &clipped.shape)
			.collect();

		while let Some(shape) = open.pop() {
			if let egui::Shape::Text(text) = shape {
				texts.push(text.galley.text().to_owned());
			} else if let egui::Shape::Vec(inner) = shape {
				open.extend(inner);
			}
		}

		texts
	}

	/// A world holding one material under a name, and its handle.
	fn coated() -> (World, MaterialId) {
		let mut world = World::new();
		let id = world
			.materials
			.insert("materials/brass", Material {
				base_color: Vec3::new(0.85, 0.62, 0.22),
				metallic: 1.0,
				..Material::DEFAULT
			});

		(world, id)
	}

	#[test]
	fn a_material_is_drawn_from_its_own_table_and_left_as_it_was() {
		let (mut world, id) = coated();
		let context = Context::default();
		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(320.0, 600.0))),
				..Default::default()
			},
			|ui| coat(ui, &mut world, id),
		);
		output.textures_delta.clear();

		assert_eq!(
			world.materials.get(id).map(|held| held.metallic),
			Some(1.0),
			"a frame nobody touched leaves the registry as it was"
		);
	}

	#[test]
	fn a_picture_named_is_taken_and_a_half_typed_one_is_left_alone() {
		// the rule, without the widget: a name somebody is halfway through
		// typing is not an error and must not take the picture off the ball.
		let (mut world, _) = coated();
		world
			.textures
			.insert("textures/brass", colby_core::abi::TextureData::white());

		let mut handle = TextureId::NONE;

		assert!(!resolve(&world, "textures/bra", &mut handle), "a half-typed name is nothing");
		assert!(!handle.is_some(), "and the handle is left where it was");
		assert!(resolve(&world, "textures/brass", &mut handle), "a whole one is taken");
		assert!(handle.is_some(), "and the handle is what it resolved to");
		assert!(
			!resolve(&world, "textures/brass", &mut handle),
			"the same name again is no move"
		);
		assert!(resolve(&world, "  ", &mut handle), "and nothing at all clears it");
		assert!(!handle.is_some(), "which is how a picture is taken off");
	}

	#[test]
	fn an_inspector_nobody_touches_writes_nothing() {
		// the trap the rotation row guards against: a rotation shown as three
		// angles and read straight back is not always the same rotation, and
		// an inspector that wrote it back every frame would walk it. A
		// rotation that is none of the easy ones, so the round trip is real.
		let turned = Transform {
			position: Vec3::new(1.5, -2.0, 0.25),
			rotation: Quat::from_euler(EulerRot::YXZ, 1.2, -0.4, 2.9),
			scale: Vec3::new(1.0, 2.0, 0.5),
		};

		assert!(!untouched(&turned, Transform::FIELDS), "a transform stays put");

		let body = Body::dynamic(Shape::ball(0.7), turned, 2.5)
			.moving(Vec3::X, Vec3::Y)
			.surfaced(0.3, 0.9);

		assert!(!untouched(&body, Body::FIELDS), "and so does a body");

		let mut joint = Joint::weld(BodyId::at(1, 1), BodyId::at(2, 1), (Vec3::X, Vec3::Z))
			.sprung(4.0, 0.7)
			.capped(12.0, 3.0);
		joint.rest = turned.rotation;

		assert!(!untouched(&joint, Joint::FIELDS), "and a joint, rest rotation and all");
		assert!(
			!untouched(
				&Renderable::new(MeshId::CUBE, Vec3::new(0.2, 0.7, 0.9)),
				Renderable::FIELDS
			),
			"and a tint through the picker"
		);
		assert!(
			!untouched(&Light::spot(Vec3::new(1.0, 0.9, 0.7), 3.0, 8.0, 0.2, 0.5), Light::FIELDS),
			"and a lamp, its word and its two angles included"
		);
		assert!(
			!untouched(&Decal { fade: 0.25, order: -2, ..Decal::BOX }, Decal::FIELDS),
			"and a decal, its word and its whole number included"
		);
		assert!(
			!untouched(
				&Sky::gradient(
					Vec3::new(0.1, 0.2, 0.5),
					Vec3::new(0.6, 0.7, 0.8),
					Vec3::new(0.1, 0.1, 0.1)
				),
				Sky::FIELDS
			),
			"and a sky"
		);
		assert!(
			!untouched(&Post::DEFAULT, Post::FIELDS),
			"and the post-processing, its word, its checkbox and its ten numbers included"
		);
	}

	/// A game's record with a field of every kind a widget has to draw.
	#[repr(C)]
	#[derive(Clone, Copy, colby_core::bytemuck::Pod, colby_core::bytemuck::Zeroable)]
	#[bytemuck(crate = "::colby_core::bytemuck")]
	struct Door {
		open: u32,
		speed: f32,
		turns: i32,
		hinge: [f32; 3],
		tint: [f32; 3],
		lean: [f32; 4],
		mark: [f32; 2],
		style: u32,
	}

	/// The door as a game declares it.
	const DOOR: colby_core::abi::Record<Door> = colby_core::abi::Record {
		name: "door",
		help: "a thing that swings",
		rows: &[
			colby_core::row!(Bool, Door, open, "whether it stands open"),
			colby_core::row!(Float, Door, speed, "how fast"),
			colby_core::row!(Int, Door, turns, "how many times"),
			colby_core::row!(Vec3, Door, hinge, "what it swings about"),
			colby_core::row!(Color, Door, tint, "its paint"),
			colby_core::row!(Quat, Door, lean, "how it hangs"),
			colby_core::row!(Vec2, Door, mark, "where its handle is"),
			colby_core::row!(Word(&["swing", "slide"]), Door, style, "how it opens"),
		],
		default: Door {
			open: 0,
			speed: 1.0,
			turns: 0,
			hinge: [0.0, 1.0, 0.0],
			tint: [1.0, 1.0, 1.0],
			lean: [0.0, 0.0, 0.0, 1.0],
			mark: [0.5, 0.5],
			style: 0,
		},
	};

	#[test]
	fn an_entity_s_records_are_drawn_and_a_frame_nobody_touched_writes_nothing() {
		// the rotation's trap again, for the rows a record draws: every kind a
		// widget has, a rotation that is none of the easy ones, and a value
		// waiting for a record nobody declared beside them. The world is being
		// edited, so a write would open a step
		let context = Context::default();
		let mut world = World::new();
		let mut history = History::default();
		let id = world.entities.spawn();

		world.editing = true;
		world
			.entities
			.declare(&DOOR)
			.expect("a door is a record a world holds");

		if let Some(door) = world.entities.record_mut(&DOOR, id) {
			door.open = 1;
			door.speed = 0.37;
			door.turns = -3;
			door.tint = [0.2, 0.7, 0.9];
			door.lean = Quat::from_euler(EulerRot::YXZ, 1.2, -0.4, 2.9).to_array();
			door.style = 1;
		}

		if let Some(drawing) = world
			.entities
			.record_mut(&colby_core::abi::DRAWING, id)
		{
			drawing.covers = 1;
		}

		world.entities.note(id, &[colby_core::abi::Noted {
			record: "gate".to_owned(),
			field: "open".to_owned(),
			value: colby_core::abi::Spelled::Truth(true),
		}]);

		let before = world.entities.noted(id);
		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(320.0, 4000.0))),
				..Default::default()
			},
			|ui| detail(ui, &mut world, Pick::Entity(id), &mut history, false, None),
		);
		output.textures_delta.clear();

		let texts = painted(&output.shapes);

		for shown in ["drawing", "door", "gate: 1 value kept for a record nobody has declared"] {
			assert!(
				texts.iter().any(|text| text == shown),
				"the panel shows {shown:?}, got {texts:?}"
			);
		}

		assert_eq!(before.len(), 8, "the fixture holds a value in each record and one waiting");
		assert_eq!(world.entities.noted(id), before, "a frame nobody touched moves no record");

		// and opens nothing, so the next change is a step of its own rather than
		// the tail of one the panel never stopped writing
		history.settle(&world);
		history.begin("move", &world);

		if let Some(transform) = world.entities.transform_mut(id) {
			transform.position.x += 1.0;
		}

		history.settle(&world);
		history.settle(&world);

		assert_eq!(history.undoable(), Some("move"), "and writes nothing down");
	}

	#[test]
	fn the_world_itself_is_what_the_panel_shows_when_nothing_in_it_is() {
		let context = Context::default();
		let mut world = World::new();
		world.sky = Sky::day();
		world.clear = Vec3::new(0.3, 0.4, 0.5);
		let mut history = History::default();
		let was = scene::settings(&world);

		let mut output = context.run_ui(RawInput::default(), |ui| {
			detail(ui, &mut world, Pick::Nothing, &mut history, false, None);
		});
		output.textures_delta.clear();

		assert_eq!(
			scene::settings(&world),
			was,
			"a frame nobody touched writes nothing back to the world"
		);
		assert_eq!(history.undoable(), None, "and nothing is written down either");
	}

	#[test]
	fn the_settings_a_panel_writes_back_are_the_ones_its_table_names() {
		// the pairing `set_settings` is documented with: everything in
		// `Stage::FIELDS` has to survive the trip, and the camera and the
		// clock have to be left exactly where they were.
		let mut world = World::new();
		world.camera.position = Vec3::new(9.0, 9.0, 9.0);
		world.time = 42.0;
		world.steps = 700;

		let mut stage = scene::settings(&world);
		stage.clear = Vec3::new(0.1, 0.2, 0.3);
		stage.sky = Sky::day();
		stage.light = Vec3::new(1.0, -2.0, 3.0);
		stage.ambient = Vec3::splat(0.4);
		stage.gravity = Vec3::new(0.0, -3.0, 0.0);
		stage.camera.position = Vec3::ZERO;
		stage.time = 0.0;

		scene::set_settings(&mut world, stage);

		assert_eq!(world.clear, stage.clear, "the clear color went in");
		assert_eq!(world.sky, stage.sky, "and the sky");
		assert_eq!(world.light, stage.light, "and the sun");
		assert_eq!(world.ambient, stage.ambient, "and the ambient");
		assert_eq!(world.gravity, stage.gravity, "and the gravity");
		assert_eq!(
			world.camera.position,
			Vec3::new(9.0, 9.0, 9.0),
			"and the camera was left where whoever is flying it put it"
		);
		assert!((world.time - 42.0).abs() < 1.0e-6, "and the clock was not moved");
		assert_eq!(world.steps, 700, "in either half");
	}

	#[test]
	fn a_model_is_drawn_as_what_stands_in_it_and_nothing_is_written_back() {
		// the one panel that only reads. A `.cmodel` is derived, so a row
		// edited here would be a row the next compile throws away - and a
		// frame that quietly wrote one would be the worst kind of that.
		let mut world = World::new();
		let mesh = world
			.meshes
			.insert("models/lamp/shade", colby_core::abi::mesh::MeshData::default());
		let material = world
			.materials
			.insert("materials/brass", Material::DEFAULT);
		let id = world
			.models
			.insert("models/lamp", colby_core::abi::model::ModelData {
				placements: vec![colby_core::abi::model::Placement {
					name: "shade".to_owned(),
					mesh,
					material,
					skeleton: colby_core::abi::SkeletonId::NONE,
					transform: Transform::IDENTITY,
				}],
			});
		let was = world.models.placements(id).to_vec();
		let context = Context::default();

		let mut output = context.run_ui(RawInput::default(), |ui| {
			made_of(ui, &mut world, id, None);
		});
		output.textures_delta.clear();

		assert_eq!(
			world.models.placements(id),
			was,
			"a frame nobody touched leaves the model as it stood"
		);
		assert_eq!(world.models.name(id), "models/lamp", "and it is still called that");
	}

	#[test]
	fn a_models_own_name_comes_off_what_it_names_and_a_strangers_does_not() {
		assert_eq!(within("models/lamp/column", "models/lamp/"), "column");
		assert_eq!(
			within("materials/brass", "models/lamp/"),
			"materials/brass",
			"a material from outside the model keeps the whole of itself, which is the one 			 \
			 worth reading in full"
		);
		assert_eq!(within("", "models/lamp/"), "", "and a handle to nothing stays nothing");
	}

	#[test]
	fn a_count_reads_as_english() {
		assert_eq!(counted(0, "piece"), "0 pieces");
		assert_eq!(counted(1, "piece"), "1 piece");
		assert_eq!(counted(5, "piece"), "5 pieces");
	}
}
