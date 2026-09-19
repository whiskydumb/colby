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
//! **What is shown is the last thing picked, and what is changed is written to
//! everything picked of its kind**: the field that moved and nothing else, part
//! by part for a vector, in the same step back. @ref [`select::Edit`] for what
//! a changed field becomes on another thing. So a wall of bricks picked whole
//! is marked as covering by one tick of one box.
//!
//! Everything that can be tested lives in [`select`](crate::select); what is
//! here is the drawing, and it is checked by looking at it - except that an
//! inspector nobody touches writes nothing, which a headless frame can check.

use colby_asset::Project;
use colby_core::{
	abi::{
		Body, BodyId, Decal, Emitter, EntityId, Field, Joint, JointId, Light, Material,
		MaterialId, MeshId, ModelId, Post, Renderable, Sky, Terrain, TextureId, Transform, World,
		field::Value,
		scene::{self, Stage},
	},
	glam::{EulerRot, Quat, Vec2, Vec3},
};
use egui::{ComboBox, DragValue, Grid, Id, Label, RichText, ScrollArea, TextEdit, Ui};

use crate::{
	history::History,
	select::{self, Edit, Pick, Selection},
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
					"{} selected: the last picked is shown, a drag moves all of them, and a \
					 field changed here is written to all of them of its kind",
					selection.len()
				));
				ui.separator();
			}

			let pick = selection.at();
			let others: Vec<Pick> = selection
				.others()
				.into_iter()
				.filter(|other| core::mem::discriminant(other) == core::mem::discriminant(&pick))
				.collect();

			detail(ui, world, pick, &others, history, rename, project);
		});
}

/// The selected thing, in detail.
///
/// @param others - everything else selected of the same kind, which a changed
/// field is written to as well
fn detail(
	ui: &mut Ui,
	world: &mut World,
	pick: Pick,
	others: &[Pick],
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
			let entities: Vec<EntityId> = others
				.iter()
				.filter_map(|other| match *other {
					| Pick::Entity(other) => Some(other),
					| Pick::Nothing
					| Pick::Body(_)
					| Pick::Joint(_)
					| Pick::Material(_)
					| Pick::Model(_) => None,
				})
				.collect();

			naming(ui, world, pick, history, rename);
			hanging(ui, world, id);
			placing(ui, world, pick, others, history);
			look(ui, world, id, &entities, history);
			lamp(ui, world, id, &entities, history);
			thrower(ui, world, id, &entities, history);
			land(ui, world, id, &entities, history);
			paint(ui, world, id, &entities, history);
			records(ui, world, id, &entities, history);
		},
		| Pick::Body(id) => {
			naming(ui, world, pick, history, rename);
			ui.label(world.bodies.get(id).map_or_else(
				|| "gone".to_owned(),
				|body| format!("a {}", select::body_words(body)),
			));
			placing(ui, world, pick, others, history);
			solid(ui, world, id, others, history);
		},
		| Pick::Joint(id) => {
			naming(ui, world, pick, history, rename);
			tie(ui, world, id, others, history);
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
fn placing(ui: &mut Ui, world: &mut World, pick: Pick, others: &[Pick], history: &mut History) {
	let Some(transform) = select::local(world, pick) else {
		return;
	};

	let mut edited = transform;
	let edits = inspect(ui, world, "transform", &mut edited, Transform::FIELDS);

	if !edits.is_empty() {
		history.begin("place", world);
		select::place_local(world, pick, edited);
		select::spread_places(world, others, &edits);
	}
}

/// What an entity looks like: the plain half of its renderable, which is the
/// tint. The mesh, the material and the pose are handles, and the tree names
/// the mesh in the row above.
fn look(
	ui: &mut Ui,
	world: &mut World,
	id: EntityId,
	others: &[EntityId],
	history: &mut History,
) {
	let Some(mut renderable) = world.entities.renderable(id).copied() else {
		return;
	};

	let edits = inspect(ui, world, "renderable", &mut renderable, Renderable::FIELDS);

	if !edits.is_empty() {
		history.begin("tint", world);
		world.entities.set_renderable(id, renderable);
		// @note: a mutation passing the others nothing here survives the suite. The
		// one field this table edits is a color, whose picker a headless frame does
		// not press, and what the call does is the lamp's, the ground's, the
		// emitter's and the decal's, which the suite presses.
		select::spread_into(
			world,
			others,
			Renderable::FIELDS,
			&edits,
			|world, id| world.entities.renderable(id).copied(),
			|world, id, renderable| world.entities.set_renderable(id, renderable),
		);
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
	let mut moved = !inspect(ui, world, "world", &mut stage, Stage::FIELDS).is_empty();

	moved |= !inspect(ui, world, "sky", &mut stage.sky, Sky::FIELDS).is_empty();
	moved |= !inspect(ui, world, "post", &mut stage.post, Post::FIELDS).is_empty();

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
fn lamp(
	ui: &mut Ui,
	world: &mut World,
	id: EntityId,
	others: &[EntityId],
	history: &mut History,
) {
	let Some(mut light) = world.entities.light(id).copied() else {
		return;
	};

	let edits = inspect(ui, world, "light", &mut light, Light::FIELDS);

	if !edits.is_empty() {
		history.begin("light", world);
		world.entities.set_light(id, light);
		select::spread_into(
			world,
			others,
			Light::FIELDS,
			&edits,
			|world, id| world.entities.light(id).copied(),
			|world, id, light| world.entities.set_light(id, light),
		);
	}
}

/// What an entity throws off, if anything.
///
/// Always drawn, for the light's reason and word for word: turning a crate
/// into a fire is picking a word in a drop-down, and a section that appeared
/// only for entities that were already emitters would leave nowhere to do it.
/// @ref `colby_core::abi::particles`.
fn thrower(
	ui: &mut Ui,
	world: &mut World,
	id: EntityId,
	others: &[EntityId],
	history: &mut History,
) {
	let Some(mut emitter) = world.entities.emitter(id).copied() else {
		return;
	};

	let edits = inspect(ui, world, "emitter", &mut emitter, Emitter::FIELDS);

	if !edits.is_empty() {
		history.begin("emitter", world);
		world.entities.set_emitter(id, emitter);
		select::spread_into(
			world,
			others,
			Emitter::FIELDS,
			&edits,
			|world, id| world.entities.emitter(id).copied(),
			|world, id, emitter| world.entities.set_emitter(id, emitter),
		);
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
fn land(
	ui: &mut Ui,
	world: &mut World,
	id: EntityId,
	others: &[EntityId],
	history: &mut History,
) {
	let Some(mut terrain) = world.entities.terrain(id).copied() else {
		return;
	};

	let edits = inspect(ui, world, "terrain", &mut terrain, Terrain::FIELDS);

	if !edits.is_empty() {
		history.begin("terrain", world);
		world.entities.set_terrain(id, terrain);
		select::spread_into(
			world,
			others,
			Terrain::FIELDS,
			&edits,
			|world, id| world.entities.terrain(id).copied(),
			|world, id, terrain| world.entities.set_terrain(id, terrain),
		);
	}
}

/// What an entity paints, if anything, and whether it is painted itself.
///
/// Always drawn, for the light's reason: turning a block into a decal is
/// picking a word in a drop-down. The word against decals sits under it,
/// because it is the other half of the same question and nothing else on the
/// panel is about paint. @ref `colby_core::abi::decal`.
fn paint(
	ui: &mut Ui,
	world: &mut World,
	id: EntityId,
	others: &[EntityId],
	history: &mut History,
) {
	if let Some(mut decal) = world.entities.decal(id).copied() {
		let edits = inspect(ui, world, "decal", &mut decal, Decal::FIELDS);

		if !edits.is_empty() {
			history.begin("decal", world);
			world.entities.set_decal(id, decal);
			select::spread_into(
				world,
				others,
				Decal::FIELDS,
				&edits,
				|world, id| world.entities.decal(id).copied(),
				|world, id, decal| world.entities.set_decal(id, decal),
			);
		}
	}

	let mut takes = world.entities.takes_decals(id);

	if ui
		.checkbox(&mut takes, "decals paint it")
		.changed()
	{
		history.begin("decals paint it", world);

		for &each in std::iter::once(&id).chain(others) {
			world.entities.set_takes_decals(each, takes);
		}
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
fn records(
	ui: &mut Ui,
	world: &mut World,
	id: EntityId,
	others: &[EntityId],
	history: &mut History,
) {
	for table in 0..world.entities.records().tables().len() {
		record(ui, world, id, others, table, history);
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
/// @param others - every other entity selected, which a changed field is
/// written to as well
/// @param table - the record's place in the declared records
fn record(
	ui: &mut Ui,
	world: &mut World,
	id: EntityId,
	others: &[EntityId],
	table: usize,
	history: &mut History,
) {
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

	// the grid and every widget in it share one salt, and it carries the word
	// `record` so that a game's record called `light` cannot collide with the
	// engine's own light table, which salts with its bare name
	let salt = format!("record {name}");

	Grid::new(&salt).num_columns(2).show(ui, |ui| {
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
			widget(ui, &format!("{salt}.{}", column.name()), &words, &mut value);

			if value != held {
				history.begin("record", world);
				world.entities.set_field(id, table, index, &value);
				select::spread_record(world, others, table, &Edit { index, held, value });
			}

			ui.end_row();
		}
	});
}

/// Everything the solver reads about a body, to edit.
///
/// Its place is the row above, through the transform's own table, and the
/// entity it drives is the branch it hangs under in the tree.
fn solid(ui: &mut Ui, world: &mut World, id: BodyId, others: &[Pick], history: &mut History) {
	let Some(mut body) = world.bodies.get(id).copied() else {
		return;
	};

	let edits = inspect(ui, world, "body", &mut body, Body::FIELDS);

	if edits.is_empty() {
		return;
	}

	history.begin("body", world);

	if let Some(held) = world.bodies.get_mut(id) {
		*held = body;
	}

	for other in others {
		if let Pick::Body(other) = *other
			&& let Some(held) = world.bodies.get_mut(other)
		{
			select::spread(held, Body::FIELDS, &edits);
		}
	}
}

/// What a joint holds, by name, and then everything else about it.
///
/// Its two bodies are handles, which the table describes and cannot name, so
/// the two rows that name them are drawn here; the anchors are in each body's
/// own space, and are numbers all the same.
fn tie(ui: &mut Ui, world: &mut World, id: JointId, others: &[Pick], history: &mut History) {
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

	let edits = inspect(ui, world, "joint", &mut joint, Joint::FIELDS);

	if edits.is_empty() {
		return;
	}

	history.begin("joint", world);

	if let Some(held) = world.joints.get_mut(id) {
		*held = joint;
	}

	for other in others {
		if let Pick::Joint(other) = *other
			&& let Some(held) = world.joints.get_mut(other)
		{
			select::spread(held, Joint::FIELDS, &edits);
		}
	}
}

/// A material, by its own table: the two pictures by name and the rest by kind.
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

	// the two pictures are rows of the table like everything else now: they
	// are `Kind::Texture` fields, and a reference row is what a reference
	// field draws as. The hand-written pair they replace kept their text in a
	// local rebuilt from the handle every frame, so neither could be typed
	// into - only pasted whole. @ref `reference`.
	let edited = !inspect(ui, world, "material", &mut material, Material::FIELDS).is_empty();

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
		.map(|placement| piece_row(world, placement, &inside))
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

/// One piece's row: its name, what it is made of, and what it wears.
///
/// A lamp has no mesh to name and wears nothing, so its row says what it
/// shines where a piece of geometry says what it is made of.
///
/// @param world - the registries the handles are named in
/// @param placement - the piece
/// @param inside - the model's name and a slash, which comes off its own names
fn piece_row(
	world: &World,
	placement: &colby_core::abi::model::Placement,
	inside: &str,
) -> (String, String, String) {
	if !placement.mesh.is_some() && placement.light.kind.is_lit() {
		return (
			placement.name.clone(),
			format!("{} lamp", placement.light.kind.word()),
			String::new(),
		);
	}

	let mesh = world
		.meshes
		.get(placement.mesh)
		.map_or("", |entry| entry.name());

	(
		placement.name.clone(),
		within(mesh, inside),
		within(world.materials.name(placement.material), inside),
	)
}

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

/// What a reference row says when hovered, after the field's own help.
const BY_NAME: &str = "by asset name, or empty for none";

/// The name of the thing a handle points at, or nothing when it points at
/// nothing.
///
/// @param world - the registries to read
/// @param value - the handle
/// @return its name, empty for a handle at nothing or a kind with no names
fn spelled(world: &World, value: &Value) -> String {
	match *value {
		| Value::Mesh(id) => world
			.meshes
			.get(id)
			.map_or_else(String::new, |entry| entry.name().to_owned()),
		| Value::Material(id) => world.materials.name(id).to_owned(),
		| Value::Texture(id) => world
			.textures
			.get(id)
			.map_or_else(String::new, |entry| entry.name().to_owned()),
		| _ => String::new(),
	}
}

/// Writes a handle from a name, if the name is one.
///
/// Empty clears the handle; a name a registry answers to takes it; anything
/// else leaves it alone, which is what a name somebody is halfway through
/// typing is. **Not refused and not cleared** - a half-typed name is not an
/// error, and clearing the handle on every keystroke would take the picture
/// off the ball while somebody spells its name.
///
/// @param world - the registries to look a name up in
/// @param typed - what is in the field
/// @param value - the handle, written when the name resolves
/// @return whether the handle moved
fn resolve(world: &World, typed: &str, value: &mut Value) -> bool {
	let wanted = typed.trim();

	let found = match *value {
		| Value::Mesh(_) if wanted.is_empty() => Value::Mesh(MeshId::NONE),
		| Value::Mesh(_) => Value::Mesh(world.meshes.find(wanted)),
		| Value::Material(_) if wanted.is_empty() => Value::Material(MaterialId::DEFAULT),
		| Value::Material(_) => Value::Material(world.materials.find(wanted)),
		| Value::Texture(_) if wanted.is_empty() => Value::Texture(TextureId::NONE),
		| Value::Texture(_) => Value::Texture(world.textures.find(wanted)),
		| _ => return false,
	};

	// a name nothing answers to reads back as the null handle, which is what
	// `Registry::find` says for anything it does not know. Clearing was asked
	// for by an empty field and is handled above, so here it means "not yet".
	if !wanted.is_empty() && spelled(world, &found).is_empty() {
		return false;
	}

	if found == *value {
		return false;
	}

	*value = found;

	true
}

/// A handle shown and edited as the name of the thing it points at.
///
/// **The typed text lives in egui's own store between frames**, keyed by the
/// row's id, and that is the whole reason this is not four lines. A name
/// somebody is halfway through typing resolves to no handle, so there is
/// nowhere in the world to keep it; a string rebuilt from the handle every
/// frame - which is what the two picture rows used to do - puts the old name
/// back on every keystroke, and egui keeps a cursor for a text field but never
/// its text. So the field could only ever be changed by pasting a whole name
/// in one go.
///
/// **The store is read only while the row has the keyboard**, and the question
/// is asked before the widget is drawn rather than after it: a row that has
/// just lost the keyboard has to show what the handle really says in the same
/// frame, so a name left half-typed does not linger and a handle changed from
/// somewhere else - an undo, another thing selected - turns up at once.
///
/// @param ui - where to draw
/// @param world - the registries to look a name up in
/// @param salt - the row's id, unique in the panel: the table and the field
/// @param value - the handle, written when the name resolves
/// @return whether the handle moved
fn reference(ui: &mut Ui, world: &World, salt: &str, value: &mut Value) -> bool {
	let id = Id::new(salt);
	let held = ui.memory(|memory| memory.has_focus(id));
	let mut typed = ui
		.data(|data| data.get_temp::<String>(id))
		.filter(|_| held)
		.unwrap_or_else(|| spelled(world, value));

	let response = ui.add(TextEdit::singleline(&mut typed).id(id));
	let moved = response.changed() && resolve(world, &typed, value);

	ui.data_mut(|data| data.insert_temp(id, typed));

	moved
}

/// One inspector over any record with a table: a row per field, and a widget
/// the field's kind decides.
///
/// **A handle into one of the asset registries is a row too**, drawn as the
/// name of the thing it points at: an entity's mesh and its material, a body's
/// collision mesh, a thrower's picture, a world's cubemap. Before this every
/// reference was skipped, so what an entity was made of could only be changed
/// by writing the `.scene` by hand - there was no control anywhere in the
/// editor for it. A handle into the *world* is still skipped: an entity, a
/// body or a joint is picked in the picture, a slot number is all a row could
/// show, and the tree already says more by nesting one under the other. A pose
/// is skipped for a different reason - poses are made at runtime and have no
/// names to type.
///
/// **A write is guarded by the numbers having actually changed**, field by
/// field, and that matters more than it looks for a rotation: two different
/// triples of angles can name one rotation, so converting out and straight
/// back in every frame would walk a rotation somewhere it was never dragged.
///
/// @param ui - where to draw
/// @param world - the registries a reference row reads a name out of
/// @param salt - what tells this grid from another in the same panel, and what
/// every widget in it is salted with
/// @param record - what to show and edit
/// @param fields - its table
/// @return every field that was written, with what it held before, for
/// whatever else is selected
fn inspect<T>(
	ui: &mut Ui,
	world: &World,
	salt: &str,
	record: &mut T,
	fields: &[Field<T>],
) -> Vec<Edit> {
	let mut edits = Vec::new();

	Grid::new(salt).num_columns(2).show(ui, |ui| {
		for (index, field) in fields.iter().enumerate() {
			let held = field.get(record);
			let named = matches!(held, Value::Mesh(_) | Value::Material(_) | Value::Texture(_));

			if field.kind.is_reference() && !named {
				continue;
			}

			let row = format!("{salt}.{}", field.name);
			let mut value = held.clone();

			if named {
				ui.label(field.name)
					.on_hover_text(format!("{}, {BY_NAME}", field.help));
				reference(ui, world, &row, &mut value);
			} else {
				ui.label(field.name).on_hover_text(field.help);
				widget(ui, &row, field.kind.words(), &mut value);
			}

			if value != held && field.set(record, value.clone()) {
				edits.push(Edit { index, held, value });
			}

			ui.end_row();
		}
	});

	edits
}

/// The widget one value is edited with, by its kind.
///
/// @param ui - where to draw
/// @param salt - an id no other widget in the panel has: the table's name and
/// the field's
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
	use colby_core::abi::Shape;
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
		let world = World::new();
		let mut output = context.run_ui(RawInput::default(), |ui| {
			written = !inspect(ui, &world, "test", &mut edited, fields).is_empty();
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

		let mut value = Value::Texture(TextureId::NONE);
		let handle = |value: &Value| match *value {
			| Value::Texture(id) => id,
			| _ => panic!("the kind does not change under it: {value:?}"),
		};

		assert!(!resolve(&world, "textures/bra", &mut value), "a half-typed name is nothing");
		assert!(!handle(&value).is_some(), "and the handle is left where it was");
		assert!(resolve(&world, "textures/brass", &mut value), "a whole one is taken");
		assert!(handle(&value).is_some(), "and the handle is what it resolved to");
		assert!(!resolve(&world, "textures/brass", &mut value), "the same name again is no move");
		assert!(resolve(&world, "  ", &mut value), "and nothing at all clears it");
		assert!(!handle(&value).is_some(), "which is how a picture is taken off");
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

		// and something else selected beside it, which a frame nobody touched must
		// leave alone as well
		let other = world.entities.spawn();
		let others = [Pick::Entity(other)];
		let theirs = world.entities.noted(other);

		let before = world.entities.noted(id);
		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(320.0, 4000.0))),
				..Default::default()
			},
			|ui| detail(ui, &mut world, Pick::Entity(id), &others, &mut history, false, None),
		);
		output.textures_delta.clear();

		let texts = painted(&output.shapes);

		for shown in [
			"drawing",
			"editing",
			"door",
			"gate: 1 value kept for a record nobody has declared",
		] {
			assert!(
				texts.iter().any(|text| text == shown),
				"the panel shows {shown:?}, got {texts:?}"
			);
		}

		assert_eq!(before.len(), 8, "the fixture holds a value in each record and one waiting");
		assert_eq!(world.entities.noted(id), before, "a frame nobody touched moves no record");
		assert_eq!(world.entities.noted(other), theirs, "and none of what is selected beside it");

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
			detail(ui, &mut world, Pick::Nothing, &[], &mut history, false, None);
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
				placements: vec![
					colby_core::abi::model::Placement {
						name: "shade".to_owned(),
						mesh,
						material,
						skeleton: colby_core::abi::SkeletonId::NONE,
						transform: Transform::IDENTITY,
						light: Light::NONE,
					},
					colby_core::abi::model::Placement {
						name: "bulb".to_owned(),
						light: Light::point(Vec3::ONE, 2.0, 5.0),
						..colby_core::abi::model::Placement::default()
					},
				],
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
	fn a_lamps_row_says_what_it_shines_where_a_pieces_says_what_it_is_made_of() {
		let mut world = World::new();
		let mesh = world
			.meshes
			.insert("models/lamp/shade", colby_core::abi::mesh::MeshData::default());
		let shade = colby_core::abi::model::Placement {
			name: "shade".to_owned(),
			mesh,
			..colby_core::abi::model::Placement::default()
		};
		let bulb = colby_core::abi::model::Placement {
			name: "bulb".to_owned(),
			light: Light::spot(Vec3::ONE, 2.0, 5.0, 0.2, 0.5),
			..colby_core::abi::model::Placement::default()
		};

		assert_eq!(
			piece_row(&world, &bulb, "models/lamp/"),
			("bulb".to_owned(), "spot lamp".to_owned(), String::new()),
			"a lamp's kind in the mesh's column, and nothing worn"
		);
		assert_eq!(
			piece_row(&world, &shade, "models/lamp/").1,
			"shade",
			"while a piece of geometry names its mesh, as it always did"
		);
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

	/// Every shape a frame painted, with what a list of shapes holds taken out
	/// of it.
	fn flat(shapes: &[egui::epaint::ClippedShape]) -> Vec<egui::Shape> {
		let mut found = Vec::new();
		let mut open: Vec<egui::Shape> = shapes
			.iter()
			.map(|clipped| clipped.shape.clone())
			.collect();

		while let Some(shape) = open.pop() {
			if let egui::Shape::Vec(inner) = shape {
				open.extend(inner);
			} else {
				found.push(shape);
			}
		}

		found
	}

	/// Where to click to press a checkbox a frame painted: the middle of its
	/// own words when it has some, and otherwise of the first box painted on
	/// the row of the label in front of it, past the label's end.
	fn checkbox(shapes: &[egui::epaint::ClippedShape], label: &str) -> Option<Pos2> {
		let flat = flat(shapes);
		// the one nearest the top of the panel, as it is read: two sections may
		// name a field alike, and what egui hands back is not in the order it is
		// laid out
		let words = flat
			.iter()
			.filter_map(|shape| match shape {
				| egui::Shape::Text(text) if text.galley.text() == label =>
					Some(text.visual_bounding_rect()),
				| _ => None,
			})
			.min_by(|one, two| one.min.y.total_cmp(&two.min.y))?;

		if label.contains(' ') {
			return Some(words.center());
		}

		flat.iter()
			.filter_map(|shape| match shape {
				| egui::Shape::Rect(boxed)
					if boxed.rect.min.x >= words.max.x
						&& (boxed.rect.center().y - words.center().y).abs() < words.height() =>
					Some(boxed.rect),
				| _ => None,
			})
			.min_by(|one, two| one.min.x.total_cmp(&two.min.x))
			.map(|boxed| boxed.center())
	}

	/// Frames of a panel over one context: the first to find the widget beside
	/// a label, the second to press it, and one more for each list of what is
	/// to follow the press. Drawn once a frame whatever egui asks, because a
	/// checkbox drawn twice answers a click twice.
	fn driven(label: &str, then: &[&[egui::Event]], draw: &mut dyn FnMut(&mut Ui)) {
		let context = Context::default();
		let shapes = painted_frame(&context, Vec::new(), draw);
		let at = checkbox(&shapes, label).expect("the label and its widget were painted");
		let mut events = vec![egui::Event::PointerMoved(at)];

		for down in [true, false] {
			events.push(egui::Event::PointerButton {
				pos: at,
				button: egui::PointerButton::Primary,
				pressed: down,
				modifiers: egui::Modifiers::NONE,
			});
		}

		drop(painted_frame(&context, events, draw));

		for events in then {
			drop(painted_frame(&context, events.to_vec(), draw));
		}
	}

	/// One frame of a panel over a context, and what it painted.
	///
	/// A frame with nothing in it is drawn every time egui asks, so that what
	/// it hands back is the pass that laid everything out; a frame with a
	/// press in it is drawn once, so that the press is answered once.
	fn painted_frame(
		context: &Context,
		events: Vec<egui::Event>,
		draw: &mut dyn FnMut(&mut Ui),
	) -> Vec<egui::epaint::ClippedShape> {
		let pressing = !events.is_empty();
		let mut once = false;
		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(360.0, 6000.0))),
				events,
				..Default::default()
			},
			|ui| {
				if !pressing || !once {
					once = true;
					draw(ui);
				}
			},
		);
		output.textures_delta.clear();

		output.shapes
	}

	/// The checkbox beside a label in the inspector over what is picked,
	/// pressed.
	fn pressed(
		world: &mut World,
		pick: Pick,
		others: &[Pick],
		history: &mut History,
		label: &str,
	) {
		driven(label, &[], &mut |ui| detail(ui, world, pick, others, history, false, None));
	}

	/// What a key pressed on a number that has taken the keyboard does: one
	/// step of its speed up.
	const UP: egui::Event = egui::Event::Key {
		key: egui::Key::ArrowUp,
		physical_key: None,
		pressed: true,
		repeat: false,
		modifiers: egui::Modifiers::NONE,
	};

	/// The number beside a label in the inspector over what is picked, clicked
	/// into and stepped up once.
	fn stepped(
		world: &mut World,
		pick: Pick,
		others: &[Pick],
		history: &mut History,
		label: &str,
	) {
		driven(label, &[&[UP]], &mut |ui| detail(ui, world, pick, others, history, false, None));
	}

	/// The number beside a label in the inspector over what is picked, clicked
	/// into and typed over, then left: a whole number, which a step up cannot
	/// move, and a frame after the typing, which is when egui takes what was
	/// typed.
	fn typed(
		world: &mut World,
		pick: Pick,
		others: &[Pick],
		history: &mut History,
		label: &str,
		number: &str,
	) {
		let enter = egui::Event::Key {
			key: egui::Key::Enter,
			physical_key: None,
			pressed: true,
			repeat: false,
			modifiers: egui::Modifiers::NONE,
		};

		driven(label, &[&[egui::Event::Text(number.to_owned()), enter], &[]], &mut |ui| {
			detail(ui, world, pick, others, history, false, None);
		});
	}

	#[test]
	fn a_number_stepped_on_what_is_shown_is_written_to_every_entity_picked() {
		let mut world = World::new();
		world.editing = true;
		let mut history = History::default();
		let [shown, other, alone] = [(); 3].map(|()| world.entities.spawn());
		let picked = [Pick::Entity(other)];

		if let Some(at) = world.entities.transform_mut(other) {
			at.position = Vec3::new(5.0, 6.0, 7.0);
		}

		stepped(&mut world, Pick::Entity(shown), &picked, &mut history, "position");
		let across = world
			.entities
			.transform(shown)
			.map_or(0.0, |it| it.position.x);
		assert!(across > 0.0, "the number moved on the entity shown: {across}");
		assert_eq!(
			world
				.entities
				.transform(other)
				.map(|it| it.position),
			Some(Vec3::new(across, 6.0, 7.0)),
			"and across on the entity picked with it, which keeps its own height and depth"
		);
		assert_eq!(
			world
				.entities
				.transform(alone)
				.map(|it| it.position),
			Some(Vec3::ZERO)
		);

		stepped(&mut world, Pick::Entity(shown), &picked, &mut history, "rate");
		let rate = |world: &World, id| {
			world
				.entities
				.emitter(id)
				.map_or(0.0, |it| it.rate)
		};
		assert!(rate(&world, shown) > Emitter::NONE.rate, "an emitter's rate, stepped");
		assert!(
			(rate(&world, other) - rate(&world, shown)).abs() < f32::EPSILON
				&& (rate(&world, alone) - Emitter::NONE.rate).abs() < f32::EPSILON,
			"on the emitter picked with it and not on the other"
		);

		// order typed rather than fade stepped: the fade's row is under the note egui
		// draws over a second widget of one id in a debug build, which takes the
		// click, and a whole number steps a quarter at a time and rounds back
		typed(&mut world, Pick::Entity(shown), &picked, &mut history, "order", "3");
		let order = |world: &World, id| world.entities.decal(id).map_or(0, |it| it.order);
		assert_eq!(
			[order(&world, shown), order(&world, other), order(&world, alone)],
			[3, 3, Decal::NONE.order],
			"and a decal's order, typed"
		);
	}

	#[test]
	fn a_place_changed_on_an_entity_is_not_written_to_a_body_picked_beside_it() {
		// the inspector shows the entity and a changed field goes to what is picked
		// of its kind: a body's place is the world's and would be moved by an
		// entity's own numbers
		let mut world = World::new();
		world.editing = true;
		let mut history = History::default();
		let crate_ = world.entities.spawn();
		let stone = world.bodies.spawn(Body::new(
			colby_core::abi::BodyKind::Static,
			Shape::UNIT,
			Transform::at(Vec3::new(5.0, 0.0, 0.0)),
		));
		let mut selection = Selection::default();
		selection.set(&world, Pick::Body(stone));
		selection.toggle(&world, Pick::Entity(crate_));

		driven("position", &[&[UP]], &mut |ui| {
			show(ui, &mut world, &selection, &mut history, false, None);
		});

		assert!(
			world
				.entities
				.transform(crate_)
				.is_some_and(|it| it.position.x > 0.0),
			"the entity shown moved"
		);
		assert_eq!(
			world
				.bodies
				.get(stone)
				.map(|it| it.transform.position),
			Some(Vec3::new(5.0, 0.0, 0.0)),
			"and the body picked beside it did not"
		);
	}

	#[test]
	fn a_record_field_ticked_on_what_is_shown_is_written_to_every_entity_picked_as_one_step() {
		let mut world = World::new();
		world.editing = true;
		let mut history = History::default();
		let [shown, left, right, alone] = [(); 4].map(|()| world.entities.spawn());
		let covering = |world: &World, id| {
			world
				.entities
				.record(&colby_core::abi::DRAWING, id)
				.is_some_and(|it| it.covers())
		};

		pressed(
			&mut world,
			Pick::Entity(shown),
			&[Pick::Entity(left), Pick::Entity(right)],
			&mut history,
			"covers",
		);

		assert!(covering(&world, shown), "the brick shown covers");
		assert!(
			covering(&world, left) && covering(&world, right),
			"and so does every brick picked with it"
		);
		assert!(!covering(&world, alone), "and nothing that was not picked");

		// the frame that wrote, and the first one nobody wrote in, which closes it
		history.settle(&world);
		history.settle(&world);
		assert_eq!(history.undoable(), Some("record"), "as one step back");
	}

	#[test]
	fn a_table_s_flag_ticked_on_what_is_shown_is_written_to_every_thing_of_its_kind_picked() {
		let mut world = World::new();
		world.editing = true;
		let mut history = History::default();
		let [shown, other, alone] = [(); 3].map(|()| world.entities.spawn());
		let picked = [Pick::Entity(other)];

		pressed(&mut world, Pick::Entity(shown), &picked, &mut history, "shadow");
		let shadowed = |world: &World, id| {
			world
				.entities
				.light(id)
				.is_some_and(|it| it.shadow)
		};
		assert_eq!(
			[shadowed(&world, shown), shadowed(&world, other), shadowed(&world, alone)],
			[false, false, true],
			"a lamp's shadow, on the lamp shown and the lamp picked with it"
		);

		pressed(&mut world, Pick::Entity(shown), &picked, &mut history, "solid");
		let solid = |world: &World, id| {
			world
				.entities
				.terrain(id)
				.is_some_and(|it| it.solid)
		};
		assert_eq!(
			[solid(&world, shown), solid(&world, other), solid(&world, alone)],
			[!Terrain::NONE.solid, !Terrain::NONE.solid, Terrain::NONE.solid],
			"ground's, the same"
		);

		pressed(&mut world, Pick::Entity(shown), &picked, &mut history, "decals paint it");
		assert_eq!(
			[
				world.entities.takes_decals(shown),
				world.entities.takes_decals(other),
				world.entities.takes_decals(alone)
			],
			[false, false, true],
			"and whether decals paint it"
		);

		let [one, two, three] = [(); 3].map(|()| {
			world.bodies.spawn(Body::new(
				colby_core::abi::BodyKind::Static,
				Shape::UNIT,
				Transform::IDENTITY,
			))
		});

		pressed(&mut world, Pick::Body(one), &[Pick::Body(two)], &mut history, "sensor");
		let sensing = |world: &World, id| world.bodies.get(id).is_some_and(|it| it.sensor);
		assert_eq!(
			[sensing(&world, one), sensing(&world, two), sensing(&world, three)],
			[true, true, false],
			"a body's, on the bodies picked"
		);

		let [first, second, third] = [(); 3].map(|()| {
			world
				.joints
				.spawn(Joint::weld(one, two, (Vec3::ZERO, Vec3::X)))
		});

		pressed(&mut world, Pick::Joint(first), &[Pick::Joint(second)], &mut history, "collide");
		let colliding = |world: &World, id| world.joints.get(id).is_some_and(|it| it.collide);
		assert_eq!(
			[colliding(&world, first), colliding(&world, second), colliding(&world, third)],
			[!Joint::weld(one, two, (Vec3::ZERO, Vec3::X)).collide; 2]
				.into_iter()
				.chain([Joint::weld(one, two, (Vec3::ZERO, Vec3::X)).collide])
				.collect::<Vec<bool>>()[..],
			"and a joint's"
		);
	}

	/// Two tables on one panel, each with a word field called `kind`.
	///
	/// Salted by the field's name alone both drop-downs are one id, and egui
	/// covers the pair with its clash overlay - which is not only a wrong list
	/// but a swallowed click, in every build with debug assertions. `kind` is a
	/// word field of five tables here and `blend` of two.
	#[test]
	fn two_tables_with_a_field_of_one_name_do_not_share_a_widget_id() {
		let context = Context::default();
		context.options_mut(|options| options.warn_on_id_clash = true);

		let world = World::new();
		let (mut light, mut decal) = (Light::NONE, Decal::NONE);
		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(320.0, 600.0))),
				..Default::default()
			},
			|ui| {
				inspect(ui, &world, "light", &mut light, Light::FIELDS);
				inspect(ui, &world, "decal", &mut decal, Decal::FIELDS);
			},
		);
		output.textures_delta.clear();

		let drawn = painted(&output.shapes);
		let complaints: Vec<&str> = drawn
			.iter()
			.filter(|text| text.contains("use of widget ID"))
			.map(String::as_str)
			.collect();

		assert!(complaints.is_empty(), "egui says two widgets share an id: {complaints:?}");
		assert!(
			drawn.iter().any(|text| text == "kind"),
			"and both tables really drew the row: {drawn:?}"
		);
	}

	/// Types a name into a reference row a character at a time, a frame each,
	/// and leaves the handle wherever the typing took it.
	///
	/// The row's id is [`Id::new`] of its salt, so a test can take the keyboard
	/// for it without knowing where on screen it landed. A character is an
	/// [`egui::Event::Text`], which is what a keyboard sends; the first frame
	/// takes the focus and selects what is there, because a row already showing
	/// a name is a row a person selects before retyping.
	///
	/// @param world - the registries the name is looked up in
	/// @param salt - the row, as `inspect` spells it
	/// @param value - the handle, edited in place
	/// @param text - what to type, one character a frame
	/// @return the context it typed into, whose store is where the row's text
	/// lives between frames
	fn typing(world: &World, salt: &str, value: &mut Value, text: &str) -> Context {
		let context = Context::default();
		let id = Id::new(salt);
		let screen = Rect::from_min_size(Pos2::ZERO, vec2(320.0, 600.0));
		let frame = |events: Vec<egui::Event>, value: &mut Value| {
			let mut output = context.run_ui(
				RawInput {
					screen_rect: Some(screen),
					events,
					..Default::default()
				},
				|ui| {
					ui.memory_mut(|memory| memory.request_focus(id));
					reference(ui, world, salt, value);
				},
			);
			output.textures_delta.clear();
		};

		frame(Vec::new(), value);
		frame(
			vec![egui::Event::Key {
				key: egui::Key::A,
				physical_key: None,
				pressed: true,
				repeat: false,
				modifiers: egui::Modifiers::COMMAND,
			}],
			value,
		);

		for letter in text.chars() {
			frame(vec![egui::Event::Text(letter.to_string())], value);
		}

		context.clone()
	}

	/// A name typed one character at a time reaches the handle.
	///
	/// **The row this replaces could not do this**, and nothing said so: it
	/// kept the field's text in a local rebuilt from the handle every frame,
	/// and egui keeps a cursor for a text field but never its text, so every
	/// keystroke that did not happen to complete a whole asset name was put
	/// back before the next frame drew. Only a paste of the finished name ever
	/// worked.
	#[test]
	fn a_name_typed_a_letter_at_a_time_reaches_the_handle() {
		let (mut world, _) = coated();
		world
			.textures
			.insert("rust", colby_core::abi::TextureData::white());

		let mut value = Value::Texture(TextureId::NONE);
		drop(typing(&world, "material.albedo", &mut value, "rust"));

		assert_eq!(
			value,
			Value::Texture(world.textures.find("rust")),
			"four keystrokes over four frames spell a name the registry knows"
		);
	}

	/// The row driven through the real panel, on the thing it is all for: what
	/// an entity is made of, changed by typing a name, and written to every
	/// entity picked with it.
	///
	/// Before this there was no control anywhere in the editor that could do
	/// it: `inspect` skipped every reference, the asset browser's drop made a
	/// new entity rather than re-pointing one, and a scene's meshes and
	/// materials could only be assigned by writing the `.scene` by hand.
	#[test]
	fn an_entity_s_mesh_typed_in_the_panel_reaches_every_entity_picked() {
		let mut world = World::new();
		world.editing = true;
		let mut history = History::default();
		let cube = world
			.meshes
			.insert("cube", colby_core::abi::MeshData::default());
		let [shown, other, alone] = [(); 3].map(|()| world.entities.spawn());

		let select_all = [egui::Event::Key {
			key: egui::Key::A,
			physical_key: None,
			pressed: true,
			repeat: false,
			modifiers: egui::Modifiers::COMMAND,
		}];
		let letters: Vec<Vec<egui::Event>> = "cube"
			.chars()
			.map(|letter| vec![egui::Event::Text(letter.to_string())])
			.collect();
		let mut then: Vec<&[egui::Event]> = vec![&select_all];
		then.extend(letters.iter().map(Vec::as_slice));

		driven("mesh", &then, &mut |ui| {
			detail(
				ui,
				&mut world,
				Pick::Entity(shown),
				&[Pick::Entity(other)],
				&mut history,
				false,
				None,
			);
		});

		let mesh = |world: &World, id| {
			world
				.entities
				.renderable(id)
				.map(|look| look.mesh)
		};

		assert_eq!(mesh(&world, shown), Some(cube), "the entity shown is made of it");
		assert_eq!(mesh(&world, other), Some(cube), "and so is the one picked with it");
		assert_eq!(
			mesh(&world, alone),
			Some(MeshId::NONE),
			"and the one nobody picked is left alone"
		);
	}

	/// A name left half-typed does not linger once the row loses the keyboard.
	///
	/// The store is what lets a name be typed at all, and it is also what would
	/// keep `cub` in the field forever if nothing put it back: the handle never
	/// moved, so the row would be showing something the world does not say.
	#[test]
	fn a_half_typed_name_is_put_back_when_the_row_loses_the_keyboard() {
		let mut world = World::new();
		world
			.meshes
			.insert("cube", colby_core::abi::MeshData::default());

		let mut value = Value::Mesh(world.meshes.find("cube"));
		let context = typing(&world, "renderable.mesh", &mut value, "cub");

		assert_eq!(
			value,
			Value::Mesh(world.meshes.find("cube")),
			"a name that is not one leaves the handle where it was"
		);

		// one more frame on the same context, because that is where the row's
		// text is kept, with the keyboard given up as a click elsewhere gives it
		context.memory_mut(|memory| memory.surrender_focus(Id::new("renderable.mesh")));

		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(320.0, 600.0))),
				..Default::default()
			},
			|ui| {
				reference(ui, &world, "renderable.mesh", &mut value);
			},
		);
		output.textures_delta.clear();

		let drawn = painted(&output.shapes);

		assert!(
			drawn.iter().any(|text| text == "cube"),
			"the row is back to what the handle really says: {drawn:?}"
		);
		assert!(
			!drawn.iter().any(|text| text == "cub"),
			"and the half-typed name is gone: {drawn:?}"
		);
	}

	/// A reference row is drawn where a reference field is, and only where the
	/// registries can spell one.
	#[test]
	fn a_reference_into_the_world_is_still_left_out_of_the_table() {
		let world = World::new();
		let mut edited = Body::dynamic(Shape::ball(1.0), Transform::IDENTITY, 1.0);
		let context = Context::default();

		let mut output = context.run_ui(
			RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(320.0, 600.0))),
				..Default::default()
			},
			|ui| drop(inspect(ui, &world, "body", &mut edited, Body::FIELDS)),
		);
		output.textures_delta.clear();

		let drawn = painted(&output.shapes);

		assert!(
			drawn.iter().any(|text| text == "shape.mesh"),
			"a mesh is an asset and has a row: {drawn:?}"
		);
		assert!(
			!drawn.iter().any(|text| text == "entity"),
			"the entity it drives is picked in the picture, not typed: {drawn:?}"
		);
	}
}
