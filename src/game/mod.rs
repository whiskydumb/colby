//! Gameplay code, and the only crate in the workspace that gets swapped while
//! the process keeps running.
//!
//! Two rules hold everything else up:
//!
//! 1. **No state lives here.** No statics, no caches, no lazily built tables.
//!    Everything the game remembers lives in the host: entities in
//!    `World::entities`, and the game's own values in `World::state`, the arena
//!    [`State`] below claims. That is what makes a reload a pointer swap with
//!    nothing to migrate.
//! 2. **Nothing panics across the boundary on purpose.** A panic that does
//!    escape is caught by the host and the module is parked until the next
//!    reload, but that is a safety net, not a design.
//!
//! Everything below the entry points is fair game to edit while colby is
//! running. Save the file and the window changes.

#![allow(unsafe_code)]

use colby_core::{
	abi::{
		ABI_VERSION, Args, Body, BodyId, BodyKind, Button, EntityId, Entry, GameApi, JointId,
		Key, Layers, Material, MaterialId, MeshId, Motion, PanelId, Renderable, SceneData,
		SceneId, Shape, Solid, TouchKind, TraceInfo, Transform, World, character, debug, scene,
	},
	bytemuck::{Pod, Zeroable},
	glam::{Vec2, Vec3},
	info, mod_ctor, mod_dtor, trace, warn,
};

mod_ctor! {}
mod_dtor! {}

/// The map everything else stands on.
///
/// `assets/scenes/construct.scene`: a yard, a parapet round it, a hangar with a
/// ramp and a stair up to a mezzanine, a few platforms, and a hole in one
/// corner to lose things down. Boxes rather than one mesh, and the reason is
/// the solver: a mesh collider is a linear scan over every triangle behind one
/// bounds rejection, and the bounds of a whole map pass every trace, so every
/// ray would walk thousands of triangles. Thirty boxes cost thirty exact tests.
///
/// Nothing in this module knows what is in it. Editing the file moves the world
/// under a running process, exactly as the props do.
const MAP_SCENE: &str = "scenes/construct";

/// How many times the floor's image repeats across it.
///
/// The cube's own coordinates run zero to one whatever it is scaled to, so a
/// fifty unit slab shows one tile stretched across fifty units without this.
/// Nineteen over fifty by forty-four is about two and a half units a tile.
///
/// It is square because `uv_scale` is one number a side and a cube's six faces
/// do not agree about which world axis each side is: matching one face exactly
/// stretches the rest. Every other surface in the map is a flat color, where
/// the question does not arise. @ref `colby-sandbox-brief` for what the
/// expensive answer would be.
const MAP_TILES: f32 = 19.0;

/// Where the player stands when the scene is put back.
///
/// In the yard, south of the hangar, with the props beside it and the camera
/// far enough back to see both. Half a unit up because the box is a unit tall
/// and the floor's surface is zero.
const PLAYER_START: Vec3 = Vec3::new(2.0, 0.5, -6.0);

/// Half-extents of the player's box.
///
/// Person-shaped rather than a person: a unit tall and under half a unit
/// across. There is no capsule to be, @ref
/// [`character`](colby_core::abi::character).
const PLAYER_EXTENTS: Vec3 = Vec3::new(0.22, 0.5, 0.22);

/// How fast the player walks, in units a second.
const PLAYER_SPEED: f32 = 4.5;

/// How hard the player jumps, in units a second.
///
/// About two thirds of a unit of height under the world's own gravity, which
/// is enough to clear a prop and not enough to reach the crystal.
const PLAYER_JUMP: f32 = 4.2;

/// How far above the player's middle the camera looks.
const EYE_LIFT: f32 = 0.4;

/// How much one line of scroll changes the camera's distance.
const ZOOM_STEP: f32 = 0.6;

/// How far the camera is allowed to sit from its target.
const DISTANCE_RANGE: (f32, f32) = (2.0, 40.0);

/// How many radians of orbit one pixel of drag is worth.
const DRAG_RATE: f32 = 0.006;

/// Where the camera starts: yaw, pitch, distance.
///
/// Chosen against the map rather than picked: the orbit puts the camera at
/// `target + (sin(yaw), .., cos(yaw)) * distance`, so a yaw whose cosine is
/// positive stands it *north* of the player, which for this map is inside the
/// hangar's west wall. The first screenshot taken of the map was a view from
/// inside that wall, and the readout said `hangar wall west @ 0.0` because the
/// pick ray started solid. There is no camera collision here and nothing plans
/// to add one; what there is instead is a starting pose that does not need it.
const START_ORBIT: (f32, f32, f32) = (3.6, 0.35, 9.0);

/// The color of the box the player is.
const PLAYER_COLOR: Vec3 = Vec3::new(0.30, 0.85, 0.55);

/// The image the floor is made of.
const FLOOR_TEXTURE: &str = "textures/tiles";

/// The normal map over it, which is what makes the grout look sunk rather than
/// painted on.
///
/// Compiled as numbers rather than as a color because its name ends in
/// `_normal` - the whole of the rule is in `colby_asset::compile`.
const FLOOR_NORMALS: &str = "textures/tiles_normal";

/// The model the scene stands in the far corner.
///
/// A whole file's worth of geometry, materials and pictures reached by one
/// name - which is what step five of the roadmap was for. What the game does
/// with it is the loop below and nothing else.
const LAMP_MODEL: &str = "models/lamp";

/// Where the model stands, and how big.
///
/// A placement is in the *model's* own space, so putting one in a scene is
/// the game's arithmetic: this one only moves and scales, and a model that
/// had to be turned as well would compose the rotations here.
const LAMP_AT: Vec3 = Vec3::new(-6.0, 0.0, -1.0);

/// How much of the model's own size it is drawn at.
const LAMP_SCALE: f32 = 1.6;

/// How many of a model's pieces the demo has room for.
///
/// The arena is a fixed layout, so a game reserves slots rather than growing
/// with whatever the artist exported. A model with more pieces than this
/// stands the first few and says so.
const LAMP_PIECES: usize = 8;

/// The interface document the game puts on screen.
///
/// An asset, like the crystal: `assets/ui/hud.html` compiles to this name, and
/// editing that file changes the window without rebuilding this module. The
/// nodes below are addressed by their `id` attribute for the same reason a mesh
/// is addressed by name - recompiling the document renumbers its boxes, and a
/// handle into them would go stale exactly when somebody is editing it.
/// The document the spawn menu is written in.
const MENU: &str = "ui/spawn";

/// How many rows that document has.
///
/// The document's, not the game's: a document is what the file says and nothing
/// at run time can add a box to it, so a menu with more to offer than this
/// shows the first twelve of it. Growing a list from data is a mechanism this
/// engine does not have yet.
const ROWS: usize = 12;

/// What the menu offers: a label, whether it is a ball, and how big.
///
/// Two shapes at six sizes rather than a catalogue, because the list is here to
/// exercise the interface rather than to be a sandbox's inventory. Typing
/// `ball` cuts it to six and `0.9` to two, which is what a search box is for.
const SPAWNABLE: [(&str, bool, f32); ROWS] = [
	("cube 0.30", false, 0.30),
	("cube 0.45", false, 0.45),
	("cube 0.60", false, 0.60),
	("cube 0.75", false, 0.75),
	("cube 0.90", false, 0.90),
	("cube 1.05", false, 1.05),
	("ball 0.30", true, 0.30),
	("ball 0.45", true, 0.45),
	("ball 0.60", true, 0.60),
	("ball 0.75", true, 0.75),
	("ball 0.90", true, 0.90),
	("ball 1.05", true, 1.05),
];

/// How many things the menu will drop before it starts again at the oldest.
///
/// Bounded like every table in the engine, and small because the point is a
/// menu rather than a pile.
const SPAWNED: usize = 8;

/// How far in front of the player a dropped thing appears, in units.
const DROP_REACH: f32 = 1.6;

/// How far above the player's middle.
const DROP_LIFT: f32 = 0.9;

const HUD: &str = "ui/hud";

/// How many loose props the scene drops on the floor.
const PROPS: usize = 5;

/// The scene the props are written in.
///
/// Everything about them - where they are let go from, how big they are, what
/// they are made of, which one hangs and how long its rope is - lives in
/// `assets/scenes/props.scene` rather than in this file. Editing it moves them
/// in the running window with no module reload, which is the whole reason this
/// engine compiles assets on a timer.
///
/// Three boxes over each other and two balls beside them: the boxes show that a
/// stack settles and stays settled, and the balls show that it is a solver
/// rather than a set of rules about boxes.
const PROPS_SCENE: &str = "scenes/props";

/// Where the rope is tied when there is no scene to say.
const HOOK: Vec3 = Vec3::new(5.20, 4.00, -5.40);

/// How big the cross marking the rope's hook is, in world units.
const HOOK_SIZE: f32 = 0.08;

/// The layer the map is on.
///
/// Layers are numbers here rather than names in a manifest, and a game naming
/// its own is the whole of the mechanism: the engine holds two bitmasks per
/// body and has no opinion about what any bit means.
///
/// This is also how the map is *found*. A sandbox's scenery and its props are
/// open-ended, so neither belongs in the arena as an array of handles; walking
/// the body table and asking which layer each one is on answers "what is the
/// map" without anything having to remember. It is what `Layers` is for, and it
/// is why the pit's own volume is on this layer too rather than on one of its
/// own.
const LAYER_WORLD: u32 = 0;

/// The layer everything the solver throws about is on.
const LAYER_PROP: u32 = 1;

/// The layer the player's box is on.
const LAYER_PLAYER: u32 = 2;

/// What a piece of the map is: on the world layer, in the way of everything.
///
/// The same thing [`Layers::DEFAULT`] is, which is why nothing in
/// `construct.scene` writes a layer at all.
const WORLD_LAYERS: Layers = Layers::single(LAYER_WORLD);

/// What a prop is.
const PROP_LAYERS: Layers = Layers::single(LAYER_PROP);

/// What the player is, and what it walks into.
const PLAYER_LAYERS: Layers = Layers::new(
	Layers::bit(LAYER_PLAYER),
	Layers::bit(LAYER_WORLD) | Layers::bit(LAYER_PROP) | Layers::bit(LAYER_PLAYER),
);

/// What the map calls the volume at the bottom of its hole.
///
/// A sensor on the world layer whose mask is props and the player, so it
/// notices those two and nothing else: the floor it hangs under and the
/// parapet beside it are on the world layer as well, and the pair is refused
/// before the narrow phase ever sees it. @ref [`swallow`].
const PIT_BODY: &str = "pit";

/// How far above whatever the pick ray found its label is written.
const LABEL_LIFT: f32 = 0.55;

/// How heavy a prop is.
const PROP_MASS: f32 = 1.0;

/// How much of an impact a prop gives back.
const PROP_BOUNCE: f32 = 0.22;

/// How hard a prop is to slide.
const PROP_GRIP: f32 = 0.62;

/// How far to one side of the original a copy is stood, in units.
///
/// A little more than a prop is wide, so a copy lands clear of what it was
/// copied from rather than inside it - two bodies created overlapping is the
/// one thing a solver has no good answer to.
const DUPE_SIDEWAYS: f32 = 1.15;

/// How far above it, so a copy of something resting on the floor is not
/// created half inside the floor.
const DUPE_LIFT: f32 = 0.35;

/// How far the pick ray reaches, in world units.
///
/// Longer than the camera ever gets from the scene, so that "nothing there" is
/// really nothing rather than the ray running out.
const REACH: f32 = 60.0;

/// How much brighter whatever is under the cursor is drawn.
const HIGHLIGHT: f32 = 1.45;

/// The game's own state, kept in the host's arena.
///
/// Add a field, bump [`STATE_LAYOUT`], save: the arena zeroes itself and the
/// game starts over, with `colby_core` untouched and the process still running.
/// That is the whole reason this is an arena rather than fields on `World`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct State {
	/// Where the camera sits, as an orbit around its target.
	yaw: f32,

	/// Radians above the horizon.
	pitch: f32,

	/// How far from the target.
	distance: f32,

	/// The panel the interface is shown in.
	hud: PanelId,

	/// The map's scene, so that editing the file lays it out again.
	///
	/// Two words for thirty entities and thirty-one bodies, because none of
	/// them is here: the map is *found* by walking the body table for
	/// [`LAYER_WORLD`], which is what keeps an open-ended thing out of a fixed
	/// arena. @ref [`lay_map`].
	map_scene: SceneId,

	/// Which revision of it is standing.
	map_revision: u32,

	/// The volume at the bottom of the hole, looked up by name once.
	pit: BodyId,

	/// How many props the hole has swallowed.
	///
	/// A running count rather than what is inside it now: a thing falling past
	/// a volume is inside it for about a step, so the momentary number is zero
	/// whenever anybody looks.
	swallowed: u32,

	/// The loose props the solver owns.
	props: [EntityId; PROPS],

	/// Their bodies.
	prop_bodies: [BodyId; PROPS],

	/// The scene they were laid out from.
	props_scene: SceneId,

	/// Which revision of it they are, so that editing the file lays them out
	/// again and nothing else does.
	props_revision: u32,

	/// The rope one of them hangs from.
	rope: JointId,

	/// The box the player walks around as.
	player: EntityId,

	/// Its body, which is kinematic: the game moves it and the solver pushes
	/// props out of its way rather than the other way round.
	player_body: BodyId,

	/// Whether it was standing on something at the end of the last step. Not a
	/// `bool` because [`Pod`] wants every bit pattern to be a valid value.
	player_grounded: u32,

	/// The panel the spawn menu is shown in.
	menu: PanelId,

	/// What the menu has dropped, oldest first.
	dropped: [EntityId; SPAWNED],

	/// Their bodies.
	dropped_bodies: [BodyId; SPAWNED],

	/// How many of those slots have ever been used, so the next one is
	/// `dropped_next % SPAWNED` and the oldest is the one it lands on.
	dropped_next: u32,

	/// How many times a prop has started touching something.
	///
	/// Counted from the queue the solver fills, which is the only way to know
	/// that something *happened* rather than that something is the case.
	landings: u32,

	/// What the pick ray found, for the readout and the highlight.
	picked: EntityId,

	/// The body it found, which is not always something with an entity.
	picked_body: BodyId,

	/// How far away it was, in units.
	picked_distance: f32,

	/// One entity per piece of the model standing in the yard.
	///
	/// Spawned once and never moved, so it is the one thing in this scene the
	/// editor can drag and have it stay put.
	lamp: [EntityId; LAMP_PIECES],
}

/// The version of [`State`]'s layout. Bump it whenever the struct changes.
///
/// Forgetting to is not unsound - `State` is `Pod`, so every bit pattern is a
/// valid `State` - but the values will be yesterday's bytes read through
/// today's fields.
const STATE_LAYOUT: u64 = 14;

/// The module's single exported symbol.
///
/// The host resolves this by name, calls it once per load, and then only ever
/// calls through the returned table. @ref
/// [`GAME_API_SYMBOL`](colby_core::abi::GAME_API_SYMBOL) for the name the host
/// looks up, and keep the two in step.
#[unsafe(no_mangle)]
pub extern "C" fn colby_game_api() -> GameApi {
	GameApi {
		abi_version: ABI_VERSION,
		init,
		update,
		shutdown,
	}
}

/// Runs once each time this module is swapped in.
///
/// # Safety
///
/// `world` must point to a live [`World`] owned by the host.
unsafe extern "C-unwind" fn init(world: *mut World) {
	// SAFETY: the host guarantees a live, exclusively borrowed World for the
	// duration of the call; see GameFn.
	let world = unsafe { &mut *world };

	world.light = Vec3::new(-0.5, -1.0, -0.35);
	world.ambient = Vec3::splat(0.22);
	world.clear = Vec3::new(0.04, 0.05, 0.07);
	world.camera.fov_y = 0.9;

	// @note: a reload finds the arena and the entities exactly as the previous
	// build left them, so there is nothing to rebuild on the way back in. The
	// scene is built only when the arena reports itself fresh, which happens on
	// the first load and whenever STATE_LAYOUT moves.
	let (state, fresh) = world.state.get::<State>(STATE_LAYOUT);
	if fresh {
		(state.yaw, state.pitch, state.distance) = START_ORBIT;

		// the arena was reset, so the handles it held are gone. Anything the
		// old build spawned would be orphaned; clear the table and start over.
		world.entities.clear();
		// and the bodies with them: a body naming an entity that no longer
		// exists drives nothing, and would still be traced against.
		world.bodies.clear();
		for slot in &mut state.lamp {
			*slot = world.entities.spawn();
		}
		// spawned where it stands and at the size it is, because nothing writes
		// the player's transform afterwards except the controller, which only
		// ever moves it. `place` deliberately leaves it alone.
		let mut stance = Transform::at(PLAYER_START);
		stance.scale = PLAYER_EXTENTS * 2.0;
		state.player = world.entities.spawn_at(stance);

		world.camera.target = PLAYER_START + Vec3::Y * EYE_LIFT;
	}

	// registered on every load, like the materials below: registering is
	// idempotent, a value somebody typed in the console survives it, and an
	// untouched one follows whatever the constant now says. The command has to
	// be registered again whatever happens, because the host drops it before
	// unloading this library - its address is in here. @ref
	// [`cvar`](colby_core::abi::cvar).
	world.cvars.command(
		"game.reset",
		reset,
		"put the scene and the camera back where they started",
	);
	world
		.cvars
		.command("game.cleanup", clear, "despawn every prop, leaving the map alone");

	// on every load, not only a fresh one: the mesh a name resolves to is the
	// host's business, and an asset that appeared since the last swap should be
	// picked up by this one. The materials the map names are declared in here,
	// so this has to run before anything is laid out.
	dress(world);
	collide(world);

	if fresh {
		// only on a fresh arena: a reload should leave the map and the pile
		// exactly where the last build left them, which is the whole point of
		// the state living in the host.
		lay_map(world);
		lay_props(world);
	}

	// showing a document is idempotent by name, so a reload finds the same
	// panel with the same text still in it rather than stacking a second copy.
	let hud = world.ui.show(HUD);
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.hud = hud;

	info!(
		reloads = world.reloads,
		entities = world.entities.len(),
		bodies = world.bodies.len(),
		fresh,
		"game init"
	);
}

/// Runs once per simulation step.
///
/// # Safety
///
/// `world` must point to a live [`World`] owned by the host.
unsafe extern "C-unwind" fn update(world: *mut World) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };
	let input = world.input;

	interface(world);

	// a click that landed on the interface is not also a click on the world.
	// The host applies the same rule between the editor and the game; this is
	// it one layer down, and it is why dragging across a panel does not swing
	// the camera with it.
	let over_interface = world.ui.pointer_over();

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);

	// hold the right button and drag to swing the camera around its target.
	if input.button_held(Button::Right) && !over_interface {
		state.yaw = input.cursor_delta[0].mul_add(-DRAG_RATE, state.yaw);
		state.pitch = input.cursor_delta[1].mul_add(DRAG_RATE, state.pitch);
	}

	state.distance = input
		.wheel
		.mul_add(-ZOOM_STEP, state.distance)
		.clamp(DISTANCE_RANGE.0, DISTANCE_RANGE.1);

	let (yaw, pitch, distance) = (state.yaw, state.pitch, state.distance);

	walk(world, yaw);

	// the camera watches the player rather than panning on its own. The same
	// keys cannot mean two things, and the reason to have a controller at all
	// is that a sandbox is somewhere you are rather than something you look at.
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let standing = state.player;

	if let Some(&transform) = world.entities.transform(standing) {
		world.camera.target = transform.position + Vec3::Y * EYE_LIFT;
	}

	world.camera.orbit(yaw, pitch, distance);

	relay_map(world);
	relay_props(world);
	rope_line(world);
	swallow(world);
	pick(world);
	duplicate(world, yaw);
	light_up(world);
	label_pick(world);
	menu(world);
	count_landings(world);
}

/// Copies whatever the cursor is on and stands the copy beside it.
///
/// The sandbox's duplicator, at the size one prop makes it. What it
/// demonstrates is that a scene is a *value*: the whole world is described,
/// two records are kept out of it, and those two are created again. Nothing
/// here transcribes a body field by field, which is the difference between a
/// dupe and a second spawn function that has to be kept in step with the first
/// one forever.
///
/// The copy goes into the same ring the spawn menu drops into, so a room does
/// not fill up with copies and the panel's count is the feedback.
///
/// @param world - the picked body, and the tables the copy is added to
/// @param yaw - which way the camera is looking, so the copy appears to one
/// side of the original rather than behind it
fn duplicate(world: &mut World, yaw: f32) {
	if !world.input.pressed(Key::F) {
		return;
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (entity, body, next) = (state.picked, state.picked_body, state.dropped_next);

	// only something the solver owns. Copying the floor, or the crystal the
	// ring turns around, is a request nobody meant to make - and the copy
	// would be a static body standing in mid-air.
	if !world.bodies.get(body).is_some_and(Body::movable) || !world.entities.alive(entity) {
		return;
	}

	let Some(scene) = cut_out(world, body) else {
		return;
	};

	// the oldest of the ring goes, exactly as a spawn does, and before the copy
	// is made rather than after: the slot the copy lands in is then the one
	// that has just been freed, which is what keeps the ring a ring.
	let slot = usize::try_from(next).unwrap_or(0) % SPAWNED;
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (old_entity, old_body) = (state.dropped[slot], state.dropped_bodies[slot]);
	world.bodies.despawn(old_body);
	world.entities.despawn(old_entity);

	let right = Vec3::new(yaw.cos(), 0.0, -yaw.sin());
	let put = scene::instantiate(world, &scene, right * DUPE_SIDEWAYS + Vec3::Y * DUPE_LIFT);
	let (copied, copied_body) = (put.entity(0), put.body(0));

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.dropped[slot] = copied;
	state.dropped_bodies[slot] = copied_body;
	state.dropped_next = next.saturating_add(1);

	trace!(of = body.slot(), into = copied_body.slot(), "duplicated");
}

/// Describes the world and keeps only one body and what it drives.
///
/// The two records are lifted out whole, so everything a body carries - its
/// shape, its surface, its layers, the mesh its collision is baked from - comes
/// along without this function naming any of it. The indices are the only thing
/// that has to be rewritten, because they addressed a description with
/// two dozen things in it and now address one with two.
///
/// @param world - the world to describe
/// @param body - the body to keep
/// @return a scene of one prop, or nothing if the body drives no entity
fn cut_out(world: &World, body: BodyId) -> Option<SceneData> {
	let whole = scene::capture(world);
	let slot = u32::try_from(body.slot()).ok()?;
	let solid = whole
		.solids
		.iter()
		.find(|solid| solid.slot == slot)?;
	let thing = whole
		.things
		.get(usize::try_from(solid.thing).ok()?)?;

	Some(SceneData {
		things: vec![thing.clone()],
		solids: vec![Solid {
			// the first and only entity of the new description, whatever
			// number it had in the old one.
			thing: 0,
			..solid.clone()
		}],
		..SceneData::default()
	})
}

/// Shows the spawn menu, filters it by what is in the search box, and drops
/// whatever was clicked.
///
/// The whole of the interface's fourth step as a game sees it: a list longer
/// than the box holding it, a field the keyboard goes to, and a class put on
/// the rows the search has ruled out. Nothing here knows about clipping or
/// scrolling - both are the stylesheet's - which is the property that makes
/// them worth having in the engine rather than in the game.
///
/// @param world - the panel to fill and the bodies to add to
fn menu(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let mut panel = state.menu;

	// re-resolved when it is nothing, for the reason the hud is.
	if !panel.is_some() {
		panel = world.ui.show(MENU);
		let (state, _) = world.state.get::<State>(STATE_LAYOUT);
		state.menu = panel;
	}

	if !panel.is_some() {
		return;
	}

	let wanted = world.ui.text(panel, "search").to_lowercase();
	let mut shown = 0;
	let mut clicked = None;

	for (index, &(label, ..)) in SPAWNABLE.iter().enumerate() {
		let name = format!("item{index}");
		let matches = wanted.is_empty() || label.contains(&wanted);

		if matches {
			shown += 1;
		}

		if matches && world.ui.clicked(panel, &name) {
			clicked = Some(index);
		}

		world.ui.set_text(panel, &name, label);
		// the class the stylesheet turns into `display: none`, which takes the
		// row out of the layout rather than merely hiding it - so the list is
		// exactly as tall as what is left and the bar says the truth.
		world
			.ui
			.set_classes(panel, &name, if matches { "entry" } else { "entry gone" });
	}

	if let Some(index) = clicked {
		drop_one(world, index);
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let count = state
		.dropped_next
		.min(u32::try_from(SPAWNED).unwrap_or(0));

	world
		.ui
		.set_text(panel, "shown", &format!("{shown}/{ROWS}"));
	world
		.ui
		.set_text(panel, "dropped", &format!("{count} out"));
}

/// Drops one of the menu's entries in front of the player.
///
/// The oldest is reused once [`SPAWNED`] of them exist, entity and body
/// together: a body whose entity was despawned is still traced against, so the
/// two lifetimes are kept in step by hand. @ref `colby-known-gaps`.
///
/// @param world - the entities and bodies to add to
/// @param index - which row of [`SPAWNABLE`] was clicked
fn drop_one(world: &mut World, index: usize) {
	let Some(&(_, ball, size)) = SPAWNABLE.get(index) else {
		return;
	};

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (player, yaw, next) = (state.player, state.yaw, state.dropped_next);
	let slot = usize::try_from(next).unwrap_or(0) % SPAWNED;
	let (old_entity, old_body) = (state.dropped[slot], state.dropped_bodies[slot]);

	let Some(&standing) = world.entities.transform(player) else {
		return;
	};

	world.bodies.despawn(old_body);
	world.entities.despawn(old_entity);

	let ahead = Vec3::new(-yaw.sin(), 0.0, -yaw.cos());
	let mut place = Transform::at(standing.position + ahead * DROP_REACH + Vec3::Y * DROP_LIFT);
	place.set_scale(size);

	let entity = world.entities.spawn_at(place);
	let mesh = if ball { MeshId::SPHERE } else { MeshId::CUBE };
	let material = world.materials.find("plastic");

	world
		.entities
		.set_renderable(entity, Renderable::of(mesh, material, prop_color(index % PROPS, ball)));

	let shape = if ball { Shape::ball(0.5) } else { Shape::UNIT };
	let body = world.attach_body(entity, BodyKind::Dynamic, shape);

	if let Some(solid) = world.bodies.get_mut(body) {
		solid.mass = PROP_MASS;
		solid.restitution = PROP_BOUNCE;
		solid.friction = PROP_GRIP;
		solid.layers = PROP_LAYERS;
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.dropped[slot] = entity;
	state.dropped_bodies[slot] = body;
	state.dropped_next = next.saturating_add(1);
}

/// Walks the player one step and writes where it ended up.
///
/// The whole of what gameplay owns about a character: which way the keys point,
/// how fast that is, what a jump is worth, and how gravity gets into the
/// velocity. Everything geometric - what it runs into, what it slides along,
/// what it may climb, whether there is ground - is
/// [`character::move_and_slide`], and this never learns any of it.
///
/// @param world - read for input and bodies, written for the player's place
/// @param yaw - which way the camera is looking, so that "forward" is forward
fn walk(world: &mut World, yaw: f32) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (player, body) = (state.player, state.player_body);
	let grounded = state.player_grounded != 0;

	let Some(&placed) = world.entities.transform(player) else {
		return;
	};

	let (input, dt) = (world.input, world.dt);

	// two sets of movement keys, summed and clamped, so holding both does not
	// walk twice as fast. They point where the camera is looking rather than
	// along the world axes, so that "left" means left on screen.
	let sideways =
		(input.axis(Key::A, Key::D) + input.axis(Key::Left, Key::Right)).clamp(-1.0, 1.0);
	let forwards = (input.axis(Key::S, Key::W) + input.axis(Key::Down, Key::Up)).clamp(-1.0, 1.0);
	let right = Vec3::new(yaw.cos(), 0.0, -yaw.sin());
	let ahead = Vec3::new(-yaw.sin(), 0.0, -yaw.cos());
	let wish = (right * sideways + ahead * forwards).normalize_or_zero() * PLAYER_SPEED;

	let falling = world
		.bodies
		.get(body)
		.map_or(0.0, |body| body.velocity.y);
	let mut velocity = Vec3::new(wish.x, falling, wish.z);

	// standing on something is what makes a jump possible and is also what
	// stops the fall accumulating: without the second line a player who has
	// stood still for a minute steps off a lip at terminal velocity.
	if grounded {
		velocity.y = if input.pressed(Key::Space) { PLAYER_JUMP } else { 0.0 };
	}

	velocity.y = world.gravity.y.mul_add(dt, velocity.y);

	let motion = Motion::new(placed.position, velocity, PLAYER_EXTENTS, dt)
		.ignoring(body)
		.layered(PLAYER_LAYERS);
	let moved = character::move_and_slide(world, &motion);

	// the entity first, because the body is kinematic and is written from it at
	// the top of the next step - and then the body as well, so that a trace
	// taken later in this same update sees where the player is rather than
	// where it was. @ref [`Body::transform`](colby_core::abi::Body::transform).
	if let Some(transform) = world.entities.transform_mut(player) {
		transform.position = moved.position;
	}

	if let Some(solid) = world.bodies.get_mut(body) {
		solid.transform.position = moved.position;
		solid.velocity = moved.velocity;
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.player_grounded = u32::from(moved.grounded);
}

/// Removes whatever has fallen into the hole, and puts the player back.
///
/// What the sensor at the bottom of the map is *for*, rather than a volume that
/// exists to prove sensors work. A prop that leaves the map would otherwise
/// fall forever, keeping a body awake and a slot taken for the life of the
/// process; a player that leaves it would never come back.
///
/// The two are told apart by which layer the body is on, which is the same
/// question `lay_map` asks and the same one the pit's own mask already answered
/// once: the volume never even forms a pair with the map around it.
///
/// @param world - the overlap list to read, and the tables to remove from
fn swallow(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (pit, player) = (state.pit, state.player_body);

	if !pit.is_some() {
		return;
	}

	// collected first: removing a body writes the table the overlap list is
	// read out of.
	let fallen: Vec<(BodyId, EntityId)> = world
		.bodies
		.inside(pit)
		.filter(|&id| id != player)
		.filter_map(|id| world.bodies.get(id).map(|body| (id, body.entity)))
		.collect();

	let caught = world.bodies.inside(pit).any(|id| id == player);

	for (body, entity) in &fallen {
		// @ref [`sweep_props`] for why the joints go first.
		world.joints.forget(*body);
		world.bodies.despawn(*body);
		world.entities.despawn(*entity);
	}

	if caught {
		// the only witness there is: the player is put back where it started,
		// so a picture taken afterwards looks exactly like one taken before the
		// fall. @ref the pre-commit audit list, which this line belongs on.
		trace!("the player fell out of the map");
		put_player_back(world);
	}

	if fallen.is_empty() {
		return;
	}

	let gone = u32::try_from(fallen.len()).unwrap_or(0);
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.swallowed = state.swallowed.saturating_add(gone);

	trace!(gone, total = state.swallowed, "the hole swallowed something");
}

/// Stands the player back where it starts, with nothing left of the fall.
///
/// Through `teleport_body` rather than by writing the entity: the two are
/// copied into each other once a step, so moving one of them alone is a body
/// and a picture in different places until the next step, and the picture would
/// be drawn crossing the whole map to get there. @ref
/// [`teleport_body`](colby_core::abi::World::teleport_body).
///
/// @param world - the player to move
fn put_player_back(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let body = state.player_body;

	let mut stance = Transform::at(PLAYER_START);
	stance.scale = PLAYER_EXTENTS * 2.0;
	world.teleport_body(body, stance);

	// the speed of a fall that is over. `walk` reads it back out of the body
	// every step, so leaving it would put the player on the floor already
	// moving at whatever it reached on the way down.
	if let Some(solid) = world.bodies.get_mut(body) {
		solid.velocity = Vec3::ZERO;
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.player_grounded = 0;
}

/// Writes what the pick ray found above the thing it found.
///
/// The same sentence the panel already shows, put where the answer is rather
/// than in a corner. It exists to drive the other half of the debug renderer -
/// a label is the only thing in it that is not a segment - and because a
/// readout beside the crosshair is what a person actually wants when the
/// question is "which one of these is it".
///
/// @param world - the pick's result, and the table the label goes in
fn label_pick(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	if !state.picked.is_some() {
		return;
	}

	// copied out of the arena, because `named` reads the tables the arena is
	// borrowed from.
	let state = *state;
	let (picked, text) = (state.picked, named(world, &state));
	let Some(&transform) = world.entities.transform(picked) else {
		return;
	};

	world
		.debug
		.label(transform.position + Vec3::Y * LABEL_LIFT, &text, debug::WHITE);
}

/// Adds up what started touching this step.
///
/// The whole of what an event queue is for: a table read every step says what
/// *is*, and a landing is something that *happened*. Drained here and cleared
/// by the host beside the input edges, so a step that runs twice in one frame
/// does not count a landing twice.
fn count_landings(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let props = state.prop_bodies;

	let landed = world
		.bodies
		.touches()
		.iter()
		.filter(|touch| touch.kind == TouchKind::Began)
		.filter(|touch| props.iter().any(|&id| touch.names(id)))
		.count();

	if landed == 0 {
		return;
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.landings = state
		.landings
		.saturating_add(u32::try_from(landed).unwrap_or(0));
}

/// Traces from the camera through the pointer and remembers what is there.
///
/// The whole of the physics boundary as gameplay sees it: build a
/// [`TraceInfo`], hand it to the world, read a
/// [`TraceResult`](colby_core::abi::TraceResult) back. Nothing here knows what
/// is behind those two functions, which is the property the boundary exists
/// for.
///
/// @param world - the camera to look from and the bodies to look at
fn pick(world: &mut World) {
	let viewport = world.ui.viewport();
	let pointer = world.ui.pointer();

	// a pointer that has never been over the window sits outside it, and
	// `--shot` never moves one at all. Falling back to the middle of the screen
	// makes this a crosshair, which is what puts it in a screenshot.
	let aim = if pointer.cmpge(Vec2::ZERO).all() && pointer.cmple(viewport).all() {
		pointer
	} else {
		viewport * 0.5
	};

	let camera = world.camera;
	let direction = camera.pixel_direction(aim, viewport);
	let result = world.trace_ray(&TraceInfo::along(camera.position, direction, REACH));

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let moved = state.picked != result.entity || state.picked_body != result.body;
	state.picked = result.entity;
	state.picked_body = result.body;
	state.picked_distance = if result.hit { result.fraction * REACH } else { 0.0 };
	let state = *state;

	if moved {
		// one line per change rather than one per step, so a live run can be
		// read. This is the only way to see the query answering in a window,
		// and in particular the only way to see it still answering after a
		// module reload - the table is the host's, and a swap must not disturb
		// it. @ref the pre-commit audit list.
		trace!(hit = %named(world, &state), fraction = result.fraction, "pick");
	}
}

/// Brightens whatever the pick ray found and puts everything else back.
///
/// Written every step rather than toggled, for the reason the interface's text
/// is: there is then nothing to remember between steps and nothing a reload can
/// leave behind lit.
///
/// @param world - the entities to color
fn light_up(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let picked = state.picked;
	let (props, scene) = (state.props, state.props_scene);

	// the props' own colors are the scene's rather than a formula's, so they
	// are read back out of it here. A highlight multiplies whatever the file
	// wrote.
	let mut base = [Vec3::ONE; PROPS];
	for (slot, thing) in base
		.iter_mut()
		.zip(&world.scenes.data(scene).things)
	{
		*slot = thing.color;
	}

	let lit = |id: EntityId, color: Vec3| {
		if id == picked && id.is_some() {
			color * HIGHLIGHT
		} else {
			color
		}
	};

	for (index, id) in props.into_iter().enumerate() {
		if let Some(renderable) = world.entities.renderable_mut(id) {
			renderable.color = lit(id, base[index]);
		}
	}
}

/// Fills the interface in and reads what was clicked in it.
///
/// Called every step, which is what the binding API is shaped for: writing the
/// same text twice replaces it rather than stacking a second copy, so there is
/// nothing to remember between steps and nothing to tear down on a reload.
///
/// @param world - the interface to write into and the events to read
fn interface(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let mut hud = state.hud;

	// re-resolved when it is nothing, for the same reason a mesh handle is:
	// the document may have been compiled after this module was loaded.
	if !hud.is_some() {
		hud = world.ui.show(HUD);
		let (state, _) = world.state.get::<State>(STATE_LAYOUT);
		state.hud = hud;
	}

	if !hud.is_some() {
		return;
	}

	// neither button is read here any more, and that is the point of step
	// thirteen: `assets/ui/hud.lua` handles both of them. What is left in Rust
	// is everything below, which needs the world - a script can see the
	// interface and nothing else.
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let state = *state;
	let hit = named(world, &state);
	let props = state.props;
	let grounded = state.player_grounded != 0;
	let (time, entities) = (world.time, world.entities.len());
	let landings = state.landings;
	let joints = world.joints.len();
	let asleep = props
		.iter()
		.filter(|&&id| id.is_some())
		.filter(|&&id| resting(world, id))
		.count();
	let contacts = world.contacts;
	let footing = if grounded { "on the ground" } else { "in the air" };

	// counted rather than remembered, which is the same rule the map follows:
	// what a prop is is a body on a layer, so how many there are is a walk.
	let loose = world
		.bodies
		.iter()
		.filter(|(_, body)| on_layer(body, PROP_LAYERS))
		.count();
	let swallowed = state.swallowed;

	world
		.ui
		.set_text(hud, "time", &format!("{time:.1} s"));
	world
		.ui
		.set_text(hud, "entities", &format!("{entities}"));
	world.ui.set_text(hud, "hit", &hit);
	world
		.ui
		.set_text(hud, "physics", &format!("{asleep}/{PROPS} asleep, {contacts} hits"));
	world
		.ui
		.set_text(hud, "joints", &format!("{joints} held, {landings} landings"));
	world
		.ui
		.set_text(hud, "props", &format!("{loose} loose, {swallowed} lost"));
	world.ui.set_text(hud, "player", footing);
}

/// Puts the scene and the camera back where they started.
///
/// What space does, and what `game.reset` does.
///
/// @param world - the state to put back
fn recenter(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	(state.yaw, state.pitch, state.distance) = START_ORBIT;
	state.swallowed = 0;

	// every prop, whatever laid it out: the scene's five, whatever the menu
	// dropped and whatever was duplicated are one set as far as this is
	// concerned, because they are one layer. @ref [`sweep_props`].
	sweep_props(world);
	put_player_back(world);
	lay_props(world);

	let (yaw, pitch, distance) = START_ORBIT;
	world.camera.target = PLAYER_START + Vec3::Y * EYE_LIFT;
	world.camera.orbit(yaw, pitch, distance);

	// a cut rather than a move: the player's place, the camera's target and its
	// orbit all jumped at once, and the host would otherwise draw every one of
	// them traveling there over the next frame. This is the whole of what a
	// fixed timestep asks of gameplay code, and it is said once here rather
	// than beside each of the writes above, because it takes effect at the end
	// of the step whatever order it was written in.
	world.entities.snap_all();
	world.snap_camera();
}

/// `game.reset` - the console's way of pressing space.
///
/// # Safety
///
/// `world` must point to a live [`World`] owned by the host, and `args` to a
/// live argument list. The host removes this command from the table before
/// unloading the module it lives in.
unsafe extern "C-unwind" fn reset(world: *mut World, _args: *const Args) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };

	recenter(world);
}

/// `game.cleanup` - what the interface's own second button asks for.
///
/// A console command rather than something the interface does, because the
/// console is the one surface a document's script can reach: the button in
/// `hud.lua` says this and nothing else. Typing `game.cleanup` is the other way
/// to ask, and it is also how a script with no mouse in it drives the same
/// thing.
///
/// # Safety
///
/// `world` must point to a live [`World`] owned by the host, and `args` to a
/// live argument list. The host removes this command from the table before
/// unloading the module it lives in.
unsafe extern "C-unwind" fn clear(world: *mut World, _args: *const Args) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };

	sweep_props(world);
}

/// Removes every prop, leaving the map and the player alone.
///
/// The same walk [`lay_map`] does, one layer along: what a prop *is* is a body
/// on [`LAYER_PROP`], so nothing has to have been remembered for this to find
/// all of them. That is the property the arena is being kept clear for, and it
/// is why this can remove things the scene laid out, things the menu dropped
/// and copies made by the duplicator without knowing which is which.
///
/// The handles the arena does still hold are cleared afterwards, because a
/// handle to a despawned body would otherwise be handed to the solver by
/// whatever reads it next.
///
/// @param world - the tables to remove from
fn sweep_props(world: &mut World) {
	// collected first: despawning writes the table being walked.
	let loose: Vec<(BodyId, EntityId)> = world
		.bodies
		.iter()
		.filter(|(_, body)| on_layer(body, PROP_LAYERS))
		.map(|(id, body)| (id, body.entity))
		.collect();

	for (body, entity) in &loose {
		// before the body, not after: a joint naming a slot that has been
		// handed out again holds whoever moved into it, and the solver visits
		// it every step either way.
		world.joints.forget(*body);
		world.bodies.despawn(*body);
		world.entities.despawn(*entity);
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.props = [EntityId::NONE; PROPS];
	state.prop_bodies = [BodyId::NONE; PROPS];
	state.dropped = [EntityId::NONE; SPAWNED];
	state.dropped_bodies = [BodyId::NONE; SPAWNED];
	state.dropped_next = 0;
	state.rope = JointId::NONE;
	state.picked = EntityId::NONE;
	state.picked_body = BodyId::NONE;

	info!(gone = loose.len(), "props swept");
}

/// Runs once before this module is swapped out.
///
/// # Safety
///
/// `world` must point to a live [`World`] owned by the host.
unsafe extern "C-unwind" fn shutdown(world: *mut World) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };

	info!(steps = world.steps, entities = world.entities.len(), "game shutdown");
}

/// Stands the model in the corner of the scene.
///
/// The whole of what a game does with a model, and it is deliberately a loop
/// rather than a call into the engine: a placement is geometry, a material and
/// a transform in the model's own space, and what to *make* of one - an entity,
/// a body, a prop somebody can pick up - is the game's to decide. This one
/// makes entities and nothing else.
///
/// Read on every load rather than only a fresh one, like everything else in
/// [`dress`]: a model recompiled since the last swap should be picked up by
/// this one, and the handles it hands out are the same either way.
fn stand_model(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let slots = state.lamp;
	let id = world.models.find(LAMP_MODEL);
	let standing = world.models.placements(id).len();

	// copied out because placing them writes the entity table, which is not
	// something that can be done while the model is borrowed.
	let pieces: Vec<(MeshId, MaterialId, Transform)> = world
		.models
		.placements(id)
		.iter()
		.take(LAMP_PIECES)
		.map(|placement| (placement.mesh, placement.material, placement.transform))
		.collect();

	if standing > LAMP_PIECES {
		warn!(
			model = LAMP_MODEL,
			standing,
			room = LAMP_PIECES,
			"the model has more pieces than the scene reserved room for"
		);
	}

	for (slot, (mesh, material, transform)) in slots.into_iter().zip(&pieces) {
		let mut stance = *transform;
		stance.position = LAMP_AT + transform.position * LAMP_SCALE;
		stance.scale *= LAMP_SCALE;

		if let Some(placed) = world.entities.transform_mut(slot) {
			*placed = stance;
		}

		world.entities.snap(slot);
		world
			.entities
			.set_renderable(slot, Renderable::of(*mesh, *material, Vec3::ONE));
	}

	// a model that shrank, or one that failed to load at all, leaves slots
	// behind. An entity drawing nothing is the honest picture of that.
	for slot in slots.into_iter().skip(pieces.len()) {
		world
			.entities
			.set_renderable(slot, Renderable::NOTHING);
	}
}

/// Declares every material the scenes can name, and dresses the player.
///
/// **Every material a `.scene` names has to be registered before that scene is
/// laid out**, because a description carries the *name* and the loader resolves
/// it once on the way in. That is why this runs ahead of [`lay_map`] and
/// [`lay_props`] rather than beside them.
///
/// Cheap enough to run on every load rather than only on a fresh arena, which
/// is what lets a surface pick up a texture the host has since compiled.
///
/// The map's own surfaces are five, one per kind of thing rather than one per
/// slab, and only the floor is textured. The rest are flat colors until there
/// is a pack to point them at, which is a change to these five lines and to
/// nothing else: what each piece of the map is made of is written in
/// `construct.scene` as a name.
fn dress(world: &mut World) {
	// asked for by name. The registry answers with a handle the game keeps for
	// this load only - resolving again next time costs nothing and means an
	// asset that arrived late is still found.
	let tiles = world.textures.find(FLOOR_TEXTURE);
	let tile_normals = world.textures.find(FLOOR_NORMALS);

	// materials are the game's own table, so they are declared here rather than
	// imported from anywhere. Registering by name is idempotent: a reload finds
	// the same handles and overwrites the same entries.
	world.materials.insert(
		"construct/floor",
		Material::textured(tiles)
			.bumped(tile_normals)
			.finished(0.0, 0.75)
			.tiled(MAP_TILES),
	);
	world
		.materials
		.insert("construct/wall", Material::DEFAULT.finished(0.0, 0.85));
	// a polished dielectric rather than a metal, which is the same call the
	// demo's crystal made and for the same reason: there is no environment map,
	// so a real metal is a dark shape with one bright edge on it. @ref
	// `colby-known-gaps`.
	world
		.materials
		.insert("construct/metal", Material::DEFAULT.finished(0.0, 0.45));
	world
		.materials
		.insert("construct/trim", Material::DEFAULT.finished(0.0, 0.6));
	world
		.materials
		.insert("construct/platform", Material::DEFAULT.finished(0.0, 0.7));

	let plastic = world
		.materials
		.insert("plastic", Material::DEFAULT.finished(0.0, 0.5));
	// the props in `scenes/props` name this one, which is the other half of
	// the rule at the top of this function. @ref [`lay_props`].
	world
		.materials
		.insert("metal", Material::DEFAULT.finished(1.0, 0.25));

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let player = state.player;

	// the props are not here, and neither is the map: what either is made of is
	// written in its own scene.
	world
		.entities
		.set_renderable(player, Renderable::of(MeshId::CUBE, plastic, PLAYER_COLOR));

	stand_model(world);
}

/// A color for the `index`th thing the spawn menu drops.
///
/// Deliberately not [`hue`]: the ring walks the whole circle and what the menu
/// drops should read as a separate set of things rather than as more of it.
/// The five props the scene lays out carry their own colors and do not come
/// through here.
///
/// @param index - which of a run of them this is
/// @param ball - whether it is round, which is worth telling apart at a glance
fn prop_color(index: usize, ball: bool) -> Vec3 {
	let shade = turn(index, PROPS).mul_add(0.4, 0.55);

	if ball {
		return Vec3::new(0.35, shade, 0.85);
	}

	Vec3::new(shade, 0.45, 0.35)
}

/// Lays the props out from their scene, throwing away whatever was there.
///
/// Everything about them is in the file, so putting them back where they were
/// let go from is not a teleport with the velocity cleared: it is reading the
/// file again. That is also what makes the reset button and editing the file
/// the same operation.
///
/// @param world - the tables to clear and fill
fn lay_props(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.landings = 0;
	let (props, bodies, rope) = (state.props, state.prop_bodies, state.rope);

	world.joints.despawn(rope);
	for (entity, body) in props.into_iter().zip(bodies) {
		world.joints.forget(body);
		world.bodies.despawn(body);
		world.entities.despawn(entity);
	}

	let scene = world.scenes.find(PROPS_SCENE);
	let revision = world.scenes.get(scene).map_or(0, Entry::revision);
	// cloned first: the table is read through `world` and the instantiate writes
	// through it, and a description is a few hundred bytes.
	let data = world.scenes.data(scene).clone();
	let put = scene::instantiate(world, &data, Vec3::ZERO);

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.props_scene = scene;
	state.props_revision = revision;
	state.rope = put.joint(0);
	for (slot, id) in state.props.iter_mut().zip(put.entities()) {
		*slot = id;
	}
	for (slot, id) in state.prop_bodies.iter_mut().zip(put.bodies()) {
		*slot = id;
	}

	info!(
		scene = PROPS_SCENE,
		revision,
		entities = put.entities().count(),
		"props laid out"
	);
}

/// Lays them out again when their file has changed.
///
/// The whole of what a game has to do to hot-reload a scene: the host
/// recompiles the file, the registry entry is rewritten under the same name
/// and its revision moves, and this notices. Nothing is watched, nothing is
/// subscribed to, and a reload of the *module* does not do it - which is the
/// point, because the pile should stay where it was left.
///
/// @param world - the scene to read and the tables to fill
fn relay_props(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (scene, was) = (state.props_scene, state.props_revision);

	// re-resolved when it is nothing, for the reason the panels are: a file
	// that did not exist when the game started is a file somebody may be about
	// to write, and a name resolves to nothing until it does.
	if !scene.is_some() {
		if world.scenes.find(PROPS_SCENE).is_some() {
			lay_props(world);
		}

		return;
	}

	if world.scenes.get(scene).map_or(0, Entry::revision) == was {
		return;
	}

	lay_props(world);
}

/// Whether a body is one of the things some layers name.
///
/// The whole of how this game finds anything open-ended: a piece of the map is
/// a body on [`WORLD_LAYERS`] and a prop is one on [`PROP_LAYERS`], so the set
/// of either is a walk of the body table rather than an array of handles in a
/// four-thousand-byte arena. That is what lets a map grow to two hundred boxes
/// and a yard fill up with props without a constant anywhere having to say how
/// many there may be.
///
/// Only the layer is compared and not the mask: what a body *is* and what it is
/// interested in are different questions, and only the first is an identity.
///
/// @param body - the body to ask about
/// @param layers - the layers that name a kind of thing
fn on_layer(body: &Body, layers: Layers) -> bool { body.layers.layer == layers.layer }

/// Puts a body on some layers, once it exists.
///
/// `attach_body` hands back a handle rather than a body, so this is the second
/// half of every one of these calls. A handle that no longer resolves is
/// nothing to complain about here: the caller has just made it.
///
/// @param world - the body table
/// @param body - what to move
/// @param layers - which layers it is on and which it interacts with
fn layer(world: &mut World, body: BodyId, layers: Layers) {
	if let Some(solid) = world.bodies.get_mut(body) {
		solid.layers = layers;
	}
}

/// Gives the player the box the solver knows it by.
///
/// The map's own bodies are not here: they are written in `construct.scene`
/// beside the entities that draw them, which is what a scene format is for.
/// What is left is the one body no file could describe, because it is made at
/// the same moment as the entity it drives and lives exactly as long as this
/// module's idea of a player.
///
/// Rebuilt on every load rather than only on a fresh arena, which costs four
/// dozen bytes and means a change to its size or its layers takes effect on the
/// next save rather than on the next restart.
fn collide(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let player = state.player;
	world.bodies.despawn(state.player_body);

	// kinematic rather than dynamic: the controller decides where the player
	// goes, and the solver's job is only to push props out from under it. A
	// dynamic one would be a box the player argues with.
	let player_body =
		world.attach_body(player, BodyKind::Kinematic, Shape::cuboid(Vec3::splat(0.5)));
	layer(world, player_body, PLAYER_LAYERS);

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.player_body = player_body;
	// what the ray found is per-load and not worth carrying across one: the
	// bodies it named are the ones just despawned. Clearing it also means the
	// first trace of every load says so in the log, which is what makes a live
	// reload readable.
	state.picked = EntityId::NONE;

	info!(bodies = world.bodies.len(), "the player has a body");
}

/// Draws the rope one of the props hangs from.
///
/// The whole of what used to put a scene where it belongs. Everything else this
/// function did was the demo turning a ring of cubes, and the map that replaced
/// it does not move: a `.scene` says where its thirty boxes are, and nothing
/// writes them again afterwards.
///
/// Read out of the *joint* rather than out of a constant, so that moving the
/// hook in the file moves the line drawn to it, and out of the *body* rather
/// than the entity, so the far end is where the solver put it rather than a
/// step behind.
///
/// @param world - the joint to read and the table the segments go in
fn rope_line(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (held, hook) = world
		.joints
		.get(state.rope)
		.map_or((BodyId::NONE, HOOK), |joint| (joint.first, joint.second_anchor));
	let end = world
		.bodies
		.get(held)
		.map_or(hook, |body| body.transform.position);

	world.debug.line(hook, end, debug::YELLOW);
	world.debug.point(hook, HOOK_SIZE, debug::YELLOW);
}

/// Lays the map out from its scene, throwing away whatever was there.
///
/// The map is thirty boxes and a volume, and **not one of their handles is kept
/// anywhere**: what is standing is found by walking the body table for
/// [`LAYER_WORLD`], which is what makes a map that grows cost the arena nothing
/// at all. The two words this does keep are the scene and its revision, which
/// is only how it knows the file has changed.
///
/// @param world - the tables to clear and fill
fn lay_map(world: &mut World) {
	// collected first: despawning writes the table being walked. The entity is
	// taken along because a body whose entity outlives it is a thing drawn with
	// nothing behind it, and the reverse is a thing traced against with nothing
	// in front.
	let standing: Vec<(BodyId, EntityId)> = world
		.bodies
		.iter()
		.filter(|(_, body)| on_layer(body, WORLD_LAYERS))
		.map(|(id, body)| (id, body.entity))
		.collect();

	for (body, entity) in &standing {
		world.bodies.despawn(*body);
		world.entities.despawn(*entity);
	}

	let scene = world.scenes.find(MAP_SCENE);
	let revision = world.scenes.get(scene).map_or(0, Entry::revision);
	// cloned first: the table is read through `world` and the instantiate
	// writes through it.
	let data = world.scenes.data(scene).clone();
	let put = scene::instantiate(world, &data, Vec3::ZERO);
	let pit = put.body_named(PIT_BODY);

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.map_scene = scene;
	state.map_revision = revision;
	state.pit = pit;

	if !pit.is_some() {
		warn!(
			scene = MAP_SCENE,
			body = PIT_BODY,
			"the map has no volume to catch what falls off it"
		);
	}

	info!(
		scene = MAP_SCENE,
		revision,
		entities = put.entities().count(),
		replaced = standing.len(),
		"map laid out"
	);
}

/// Lays it out again when its file has changed.
///
/// The same six lines the props have, for the same reason and against the same
/// mechanism: the host recompiles the file, the registry entry is rewritten
/// under the same name and its revision moves, and this notices. Editing
/// `construct.scene` therefore rebuilds the map in the running window with no
/// module reload at all.
///
/// @param world - the scene to read and the tables to fill
fn relay_map(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (scene, was) = (state.map_scene, state.map_revision);

	// re-resolved when it is nothing, for the reason the props' is: a file that
	// did not exist when the game started is one somebody may be about to
	// write.
	if !scene.is_some() {
		if world.scenes.find(MAP_SCENE).is_some() {
			lay_map(world);
		}

		return;
	}

	if world.scenes.get(scene).map_or(0, Entry::revision) == was {
		return;
	}

	lay_map(world);
}

/// Whether the body driving an entity has gone to sleep.
///
/// Asked through the entity because that is the handle this scene keeps for a
/// prop; the body is found by walking the table, which for five props is
/// cheaper than another array in the arena.
fn resting(world: &World, entity: EntityId) -> bool {
	world
		.bodies
		.iter()
		.any(|(_, body)| body.entity == entity && body.sleeping)
}

/// What the pick ray found, as a line of text.
///
/// **Read out of the world rather than compared against the arena.** Every
/// piece of the map is called something in `construct.scene`, `instantiate`
/// copies those names onto the bodies it creates, and asking the table is
/// therefore the whole of this - which is what stops a readout being a list of
/// special cases that has to grow by one line every time the map does.
///
/// By body rather than by entity, because a body is what the ray answers with
/// and not every body draws anything: the volume at the bottom of the hole is
/// nameable and invisible.
///
/// @param world - the tables to ask
/// @param state - the arena, for what the ray found
fn named(world: &World, state: &State) -> String {
	if !state.picked_body.is_some() {
		return "nothing".to_owned();
	}

	let distance = state.picked_distance;

	if state.picked == state.player {
		return format!("player @ {distance:.1}");
	}

	// the body first, then the entity it drives: a scene names both, and a body
	// spawned by this module rather than read out of a file has only the
	// second. Neither is a fact the arena could have held.
	let name = match world.bodies.name(state.picked_body) {
		| "" => world.entities.name(state.picked),
		| named => named,
	};

	if name.is_empty() {
		return format!("something @ {distance:.1}");
	}

	format!("{name} @ {distance:.1}")
}

/// `index` out of `count`, as a fraction of a full turn.
fn turn(index: usize, count: usize) -> f32 {
	let index = u16::try_from(index).unwrap_or(0);
	let count = u16::try_from(count).unwrap_or(1).max(1);

	f32::from(index) / f32::from(count)
}
