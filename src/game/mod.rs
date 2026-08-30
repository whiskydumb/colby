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
		ABI_VERSION, Args, Body, BodyId, BodyKind, Button, EntityId, Entry, GameApi, Joint,
		JointId, Key, Layers, Material, MaterialId, MeshId, Motion, PanelId, Renderable,
		SceneData, SceneId, Shape, ShapeKind, Stage, TouchKind, TraceInfo, Transform, World,
		character, console, debug, scene,
	},
	bytemuck::{Pod, Zeroable},
	glam::{Quat, Vec2, Vec3},
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
/// The document's, not the game's. @ref [`menu`].
const MENU_ROWS: usize = 16;

/// Where a prop comes from.
///
/// A prefix in the scene registry rather than a directory anything opens: the
/// compiler has already walked `assets/` and named everything under it by its
/// own path, so `assets/props/crate.scene` is the scene `props/crate` and the
/// catalogue is a filter over names. @ref [`catalogue`].
const PROPS_DIR: &str = "props/";

/// How many props the yard will hold before it stops taking more.
///
/// Not the body table's limit, which is a thousand and change: this is the
/// number past which the broadphase - a linear scan over every pair - stops
/// being free. What is refused says so in the log rather than quietly not
/// happening.
const MAX_PROPS: usize = 96;

/// How far in front of the player a dropped thing appears, in units.
const DROP_REACH: f32 = 1.6;

/// How far above the player's middle.
const DROP_LIFT: f32 = 0.9;

const HUD: &str = "ui/hud";

/// A slot number no table will ever hand out.
///
/// Used to make a comparison against a slot that does not exist fail, rather
/// than to name anything.
const NO_SLOT: u32 = u32::MAX;

/// How many things the scene lays out.
///
/// The arena mirrors the set the *file* holds, so a sixth record in it needs a
/// sixth slot here. That is the one place `props.scene` is not simply data: a
/// prop added there and not here is laid out and then not remembered, which is
/// invisible until something wants to put it back.
const PROPS: usize = 6;

/// The scene the props are written in.
///
/// Everything about them - where they are let go from, how big they are, what
/// they are made of, which one hangs and how long its rope is - lives in
/// `assets/scenes/props.scene` rather than in this file. Editing it moves them
/// in the running window with no module reload, which is the whole reason this
/// engine compiles assets on a timer.
///
/// Three boxes over each other and two balls beside them: the boxes show that a
/// stack settles and stays settled, and one ball shows that it is a solver
/// rather than a set of rules about boxes. The other ball is `weightless`, so
/// it hangs where it was let go with nothing holding it up and is still shoved
/// by anything that touches it - which is the difference between a body gravity
/// does not reach and one the solver does not move.
///
/// The sixth is a door on the hangar, hung on a hinge with a vertical axis. It
/// is the only thing in the demo that shows what a `JointKind::Axis` is *for*:
/// five directions held and one left free, so it swings and cannot be pushed
/// off its hinge. Shoving it with the physics gun is the whole demonstration.
const PROPS_SCENE: &str = "scenes/props";

/// How big the cross marking a joint's far anchor is, in world units.
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

/// How close and how far the gun will hold something, in units.
///
/// The near end is far enough that a carried prop is not inside the player's
/// own box; the far end is most of the yard.
const HOLD_RANGE: (f32, f32) = (2.0, 24.0);

/// How much one line of scroll moves what is being held.
const HOLD_STEP: f32 = 0.8;

/// How stiff the joint that carries a prop is, in hertz.
///
/// Measured on a grid rather than taken from anywhere: @ref the doc on
/// [`carry`]. The numbers the field publishes are for a solver running a
/// different step, and a frequency near half of sixty is rigid here whatever it
/// says elsewhere.
const HOLD_STIFFNESS: f32 = 12.0;

/// How quickly that spring stops ringing.
///
/// Above one on purpose: a prop that overshoots the point it is being carried
/// to and comes back is a prop that feels like it is on elastic.
const HOLD_DAMPING: f32 = 1.6;

/// How many times a prop's own weight the gun may spend in one step.
///
/// The ceiling is scaled by the held body's mass rather than being a number of
/// newtons, which is what stops the gun feeling sluggish on a light prop and
/// explosive on a heavy one. What it buys is not a weight limit - a physics gun
/// is supposed to lift anything it can hold - but a bound on what one step may
/// do, so a prop shoved into a wall pushes rather than launching.
const HOLD_STRENGTH: f32 = 26.0;

/// The same, for the half that keeps the prop facing the way it was picked up.
const HOLD_TWIST: f32 = 6.0;

/// Where the beam is drawn from, above the player's middle and in front of it.
///
/// The forward part is not decoration. The player is a box a unit tall centered
/// on its own middle, so a muzzle only *lifted* is a muzzle inside the thing
/// that draws it, and the beam comes out of the player's chest and is hidden by
/// it. Found by looking at a shot: the line was there and almost none of it
/// was.
const MUZZLE_AT: (f32, f32) = (0.3, 0.45);

/// How big the cross marking where the beam is attached is.
const MUZZLE_MARK: f32 = 0.09;

/// The five things the toolgun does, in the order the number keys pick them.
///
/// Chosen so that between them they drive every joint the engine has and every
/// verb the sandbox already grew: a weld and a hinge are the two joint kinds
/// nothing had ever made, a rope is the one the demo made by hand, and the last
/// two are the console's `game.remove` and `game.freeze` with a crosshair in
/// front of them.
const TOOLS: [&str; 5] = ["weld", "hinge", "rope", "remover", "freeze"];

/// Which of them is which.
const WELD: u32 = 0;
/// @ref [`TOOLS`].
const HINGE: u32 = 1;
/// @ref [`TOOLS`].
const ROPE: u32 = 2;
/// @ref [`TOOLS`].
const REMOVER: u32 = 3;
/// @ref [`TOOLS`].
const FREEZE: u32 = 4;

/// Which gun is in the hands: the one that carries, or the one with modes.
const PHYSGUN: u32 = 0;
/// @ref [`PHYSGUN`].
const TOOLGUN: u32 = 1;

/// How stiff a weld the toolgun makes is, in hertz.
///
/// Soft rather than rigid, and it is the second thing the spring was added for.
/// A rigid weld takes a fifth of its error out every step whatever that costs,
/// and the sandbox's ordinary case is welding two props that are *not*
/// touching, so a rigid one yanks them together hard enough to knock over
/// whatever they were standing on.
///
/// **Small, because stiffness passes rigid on the way up rather than
/// approaching it.** The rigid path corrects by a tuned Baumgarte fifth; a
/// spring's own bias is derived and climbs towards one, so it overtakes the
/// rigid path at about seven hertz. Measured across a unit of gap: rigid yanks
/// at twelve units a second and arrives in ten steps, this yanks at five and a
/// half and arrives in seventeen, and twenty hertz yanks at *more* than rigid.
const WELD_STIFFNESS: f32 = 3.5;

/// How quickly that weld stops ringing.
const WELD_DAMPING: f32 = 1.4;

/// What the first half of a two-click tool is marked with.
const PENDING_COLOR: Vec3 = Vec3::new(1.0, 0.45, 0.15);

/// How big that mark is.
const PENDING_MARK: f32 = 0.12;

/// What a joint is drawn in.
const JOINT_COLOR: Vec3 = Vec3::new(1.0, 0.9, 0.1);

/// What whatever the crosshair is on is outlined in.
///
/// Drawn rather than tinted, for the reason a frozen prop is: what a prop was
/// colored before depends on which file it came out of, and an outline needs to
/// remember nothing. It also works for the map, which a tint could not - the
/// map's entities are laid out once and never written again.
const PICKED_COLOR: Vec3 = Vec3::new(1.0, 0.85, 0.35);

/// What a frozen prop is outlined in.
///
/// Drawn rather than tinted, and the reason is bookkeeping. A tint would have
/// to remember what the prop was colored before it was frozen, and where that
/// color comes from differs by how the prop arrived: the scene's five carry
/// their own and a duplicate carries whatever it was copied from. An outline
/// needs none of that, restores itself by not being drawn, and works for a prop
/// that arrived by a route nobody has written yet.
const FROZEN_COLOR: Vec3 = Vec3::new(0.55, 0.85, 1.0);

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

	/// Where on it the ray landed, in the world.
	///
	/// Three floats rather than a `Vec3` because the arena is [`Pod`] and glam
	/// is built without bytemuck, so nothing holding one can be.
	picked_at: [f32; 3],

	/// Which way the surface faced there, which is a hinge's axis.
	picked_normal: [f32; 3],

	/// The joint carrying whatever the gun is holding, or nothing.
	///
	/// The gun *is* this joint: there is no second mechanism, no controller and
	/// no per-step teleport. Letting go is despawning it.
	hold: JointId,

	/// The body it is holding.
	held: BodyId,

	/// How far in front of the eye it is being carried.
	hold_distance: f32,

	/// Where the beam is attached, in the held body's own space.
	hold_anchor: [f32; 3],

	/// How the prop was turned when it was picked up, xyzw.
	hold_rest: [f32; 4],

	/// Which way the camera was looking then.
	///
	/// A carried prop turns with the player and not with the pitch, which is
	/// what makes carrying one feel like carrying rather than like aiming.
	hold_yaw: f32,

	/// Which gun is in the hands. @ref [`PHYSGUN`].
	gun: u32,

	/// Which of [`TOOLS`] the toolgun is set to.
	tool: u32,

	/// The body a two-click tool is waiting on, or nothing.
	tool_first: BodyId,

	/// Where on it the first click landed, in the world.
	tool_at: [f32; 3],

	/// Which way the surface faced there, for a hinge's axis.
	tool_normal: [f32; 3],

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
const STATE_LAYOUT: u64 = 16;

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
		// said rather than relied on: a fresh arena is zeroed, and zero happens
		// to be both guns' and the first tool's number.
		state.gun = PHYSGUN;
		state.tool = WELD;

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
	// every action the gun performs is a named function over the world with a
	// console command in front of it. That is the discipline replication will
	// want, and until there is any it is what lets a script with no mouse in it
	// drive the whole weapon.
	world.cvars.command(
		"game.grab",
		take,
		"take hold of what the crosshair is on, or of a named body",
	);
	world
		.cvars
		.command("game.release", drop_held, "let go of whatever is held");
	world.cvars.command(
		"game.freeze",
		hold_still,
		"stop a prop where it stands, by name or under the crosshair",
	);
	world.cvars.command(
		"game.unfreeze",
		let_go,
		"let one go again, by name or under the crosshair",
	);
	world
		.cvars
		.command("game.thaw", thaw_all, "let every frozen prop go at once");
	world.cvars.command(
		"game.spawn",
		put_one,
		"drop a prop in front of the player, by the name the menu shows",
	);
	world.cvars.command(
		"game.tool",
		pick_tool,
		"put the toolgun in hand and set it, or list the modes",
	);
	world.cvars.command(
		"game.apply",
		use_it,
		"use the toolgun on a named body, or on whatever the crosshair is on",
	);
	world.cvars.command(
		"game.save",
		keep_one,
		"keep what is held or aimed at as a prop called <name>, joints and all",
	);

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

	// the wheel moves what is held, or the camera when nothing is. One knob,
	// two meanings, and which one is obvious from what is in front of you.
	if state.hold.is_some() {
		state.hold_distance = input
			.wheel
			.mul_add(HOLD_STEP, state.hold_distance)
			.clamp(HOLD_RANGE.0, HOLD_RANGE.1);
	} else {
		state.distance = input
			.wheel
			.mul_add(-ZOOM_STEP, state.distance)
			.clamp(DISTANCE_RANGE.0, DISTANCE_RANGE.1);
	}

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
	draw_joints(world);
	swallow(world);
	pick(world);
	physgun(world, yaw);
	freezer(world);
	toolgun(world);
	outline_frozen(world);
	outline_picked(world);
	duplicate(world, yaw);
	label_pick(world);
	menu(world);
	count_landings(world);
}

/// Copies the contraption the cursor is on and stands the copy beside it.
///
/// The sandbox's duplicator. What it demonstrates is that a scene is a *value*:
/// the whole world is described, a connected piece of it is kept, and that
/// piece is created again. Nothing here transcribes a body field by field,
/// which is the difference between a dupe and a second spawn function that has
/// to be kept in step with the first one forever - and it is why a copy of a
/// welded bridge arrives welded, with no code anywhere knowing what a bridge
/// is.
///
/// @param world - the picked body, and the tables the copy is added to
/// @param yaw - which way the camera is looking, so the copy appears to one
/// side of the original rather than behind it
fn duplicate(world: &mut World, yaw: f32) {
	if !world.input.pressed(Key::F) {
		return;
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (entity, body) = (state.picked, state.picked_body);

	// only something the solver owns. Copying the floor, or the crystal the
	// ring turns around, is a request nobody meant to make - and the copy
	// would be a static body standing in mid-air.
	if !world.bodies.get(body).is_some_and(Body::movable) || !world.entities.alive(entity) {
		return;
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let held = state.hold;

	let Some(scene) = cut_out(world, body, held) else {
		return;
	};

	let right = Vec3::new(yaw.cos(), 0.0, -yaw.sin());
	let put = scene::instantiate(world, &scene, right * DUPE_SIDEWAYS + Vec3::Y * DUPE_LIFT);

	trace!(of = body.slot(), into = put.body(0).slot(), "duplicated");
}

/// Describes the world and cuts out the whole contraption one body is part of.
///
/// Two lines of arithmetic and two calls, because both the walking and the
/// cutting are the engine's: what a description *is* is its business even
/// though what one *becomes* is the game's, and the editor wants the identical
/// pair to export a selection as a prefab. What is left here is finding which
/// entry of the description a handle is, which is the one question a game can
/// answer and a description cannot.
///
/// **A piece rather than a prop, and that is the whole of what a duplicator
/// is.** Point at one plank of a bridge and what comes back is the bridge:
/// every prop the joints reach and every joint between them, renumbered into a
/// description of its own. A prop nothing is welded to is a piece of one, which
/// is why this replaced cutting a single body rather than being added beside
/// it.
///
/// **The gun's own grip is not part of the thing it is holding**, and finding
/// that out cost a saved contraption with a physics gun welded into it. The
/// joint a carry makes is a weld pinned to a point in the world, which is
/// exactly what nailing a prop to a wall is, so nothing about the joint itself
/// could tell them apart. What can is the game, which knows which joint it
/// made, so it takes it out of the description before the cutting starts.
///
/// @param world - the world to describe
/// @param body - a body of the piece to keep
/// @param ignoring - a joint to leave out, or [`JointId::NONE`]
/// @return the piece, or nothing if the handle is in no description
fn cut_out(world: &World, body: BodyId, ignoring: JointId) -> Option<SceneData> {
	let mut whole = scene::capture(world);
	let slot = u32::try_from(body.slot()).ok()?;

	if ignoring.is_some() {
		let held = u32::try_from(ignoring.slot()).unwrap_or(NO_SLOT);
		whole.links.retain(|link| link.slot != held);
	}

	let at = whole
		.solids
		.iter()
		.position(|solid| solid.slot == slot)?;
	let seed = u32::try_from(at).ok()?;

	Some(whole.subset(&whole.connected(seed)))
}

/// Keeps the piece being held, or the one under the crosshair, as a prop.
///
/// **A saved contraption is a prop, and there is no second format for one.**
/// The piece is registered as `props/<name>`, which is the only thing the spawn
/// menu ever looks at, so it is spawnable the moment it is saved - before
/// anything reaches a disk at all. Writing it down is then a separate thing
/// that makes it survive a restart, and the host does it, because a game module
/// has no business reaching into the asset tree.
///
/// The way it asks is the console, which is the surface a document's own script
/// already reaches gameplay through. @ref `colby-abi-rules`.
///
/// @param world - the world to cut from and the registry to add to
/// @param name - what to call it, without a prefix or an extension
/// @return whether one was kept
fn keep_build(world: &mut World, name: &str) -> bool {
	if name.is_empty() {
		warn!("a build needs a name to be saved under");

		return false;
	}

	// what is held, or what is aimed at. The same rule the freeze key follows,
	// and it is the useful one: a contraption you are carrying is a contraption
	// you have just decided you want.
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let body = if state.held.is_some() {
		state.held
	} else {
		state.picked_body
	};

	let held = state.hold;

	let Some(mut piece) = cut_out(world, body, held) else {
		warn!(name, "nothing held or under the crosshair to keep");

		return false;
	};

	// a saved contraption is a *template* rather than a snapshot. What was
	// caught mid-drift would otherwise be spawned mid-drift every time, and the
	// world it was cut from is nobody's business once it is a prop.
	for solid in &mut piece.solids {
		solid.velocity = Vec3::ZERO;
		solid.angular = Vec3::ZERO;
		solid.sleeping = false;
	}
	piece.stage = Stage::DEFAULT;
	center(&mut piece);

	let registered = format!("{PROPS_DIR}{name}");
	world.scenes.insert(&registered, piece.clone());
	info!(
		name = registered,
		bodies = piece.solids.len(),
		joints = piece.links.len(),
		"kept, and it is a prop now"
	);

	// the host's half: a file under `assets/props/`, so that it is still a prop
	// next time. Asked for by name through the console rather than done here,
	// because the filesystem and the asset tree are the runner's.
	console::run(world, &format!("scene.prop {name}"));

	true
}

/// Every prop the tree holds, in a fixed order.
///
/// **There is no prop table and no prop format.** The compiler turns every
/// `.scene` under `assets/` into an asset named by its own path, so a file at
/// `assets/props/crate.scene` is a scene called `props/crate`, and finding the
/// catalogue is walking the registry for that prefix. Adding a prop is adding a
/// file; nothing in this module is told and nothing has to be rebuilt.
///
/// Sorted, because a registry is in whatever order the walk found things and a
/// menu that reshuffles itself between runs is a menu nobody can learn.
///
/// @param world - the scene registry to walk
/// @return the names without their prefix, sorted
fn catalogue(world: &World) -> Vec<String> {
	let mut found: Vec<String> = world
		.scenes
		.iter()
		.filter_map(|scene| scene.name().strip_prefix(PROPS_DIR))
		.map(str::to_owned)
		.collect();
	found.sort();

	found
}

/// Shows the spawn menu, filters it by what is in the search box, and spawns
/// whatever was clicked.
///
/// The whole of the interface's fourth step as a game sees it: a list longer
/// than the box holding it, a field the keyboard goes to, and a class put on
/// the rows the search has ruled out. Nothing here knows about clipping or
/// scrolling - both are the stylesheet's - which is the property that makes
/// them worth having in the engine rather than in the game.
///
/// **A document is what the file says**, so the menu can only address the rows
/// `spawn.html` was written with. A tree holding more props than that shows the
/// first [`MENU_ROWS`] of them and the rest are reached by typing into the
/// search box. Growing a list from data is a mechanism this engine does not
/// have, and it is the interface's own step rather than the sandbox's.
///
/// @param world - the panel to fill and the tables to add to
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

	let offered = catalogue(world);
	let wanted = world.ui.text(panel, "search").to_lowercase();
	let shown: Vec<&String> = offered
		.iter()
		.filter(|name| wanted.is_empty() || name.contains(&wanted))
		.collect();
	let mut clicked = None;

	for row in 0..MENU_ROWS {
		let id = format!("item{row}");
		let label = shown.get(row);

		if label.is_some_and(|_| world.ui.clicked(panel, &id)) {
			clicked = label.map(|name| (*name).clone());
		}

		match label {
			| Some(name) => {
				world.ui.set_text(panel, &id, name);
				world.ui.set_classes(panel, &id, "entry");
			},
			// the class the stylesheet turns into `display: none`, which takes
			// the row out of the layout rather than merely hiding it - so the
			// list is exactly as tall as what is left and the bar says the
			// truth.
			| None => world.ui.set_classes(panel, &id, "entry gone"),
		}
	}

	if let Some(name) = clicked {
		spawn_prop(world, &name);
	}

	let loose = props(world).len();
	let (fits, total) = (shown.len().min(MENU_ROWS), offered.len());

	world
		.ui
		.set_text(panel, "shown", &format!("{fits}/{total}"));
	world
		.ui
		.set_text(panel, "dropped", &format!("{loose} out"));
}

/// Puts one of them in front of the player.
///
/// Nothing is remembered about it. The prop is a description the loader turns
/// into an entity and a body with fresh handles, and from then on it is found
/// the way every prop is found - by the layer it is on. That is what lets the
/// yard fill up without a fixed array anywhere deciding how full it may get.
///
/// @param world - the tables to add to
/// @param name - which prop, without its prefix
/// @return whether one was made
fn spawn_prop(world: &mut World, name: &str) -> bool {
	let loose = props(world).len();
	if loose >= MAX_PROPS {
		warn!(loose, room = MAX_PROPS, name, "the yard is full");

		return false;
	}

	let scene = world.scenes.find(&format!("{PROPS_DIR}{name}"));
	if !scene.is_some() {
		warn!(name, "nothing in assets/props is called that");

		return false;
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (player, yaw) = (state.player, state.yaw);

	let Some(&standing) = world.entities.transform(player) else {
		return false;
	};

	let ahead = Vec3::new(-yaw.sin(), 0.0, -yaw.cos());
	let at = standing.position + ahead * DROP_REACH + Vec3::Y * DROP_LIFT;
	// cloned first: the table is read through `world` and the instantiate
	// writes through it.
	let data = world.scenes.data(scene).clone();
	let put = scene::instantiate(world, &data, at);

	trace!(name, body = put.body(0).slot(), "spawned");

	true
}

/// Moves a description so that its bodies stand around the origin.
///
/// A piece cut out of a world is written down in that world's coordinates, and
/// a prop is spawned by [`scene::instantiate`] offsetting everything by where
/// it is being put - so a contraption saved eight units from the middle of the
/// yard would arrive eight units from wherever it was asked for. Every prop
/// written by hand stands at the origin for the same reason.
///
/// The middle is the mean of the bodies rather than the seed the piece was
/// walked from, because `connected` hands its answer back in index order and
/// which body that puts first is not something a person chose.
///
/// Only a joint pinned to the *world* moves with them: every other anchor is in
/// a body's own space and moves with the body. That is the same rule
/// `instantiate` follows in the other direction.
///
/// @note: this is arguably `SceneData`'s rather than the game's - an editor
/// exporting a selection as a prefab wants the identical arithmetic. It is
/// here because the cutting commit had already landed. @ref the audit list.
///
/// @param piece - the description to move, written
fn center(piece: &mut SceneData) {
	let count = u16::try_from(piece.solids.len()).unwrap_or(0);
	if count == 0 {
		return;
	}

	let total: Vec3 = piece
		.solids
		.iter()
		.map(|solid| solid.transform.position)
		.sum();
	let middle = total / f32::from(count);

	for thing in &mut piece.things {
		thing.transform.position -= middle;
	}

	for solid in &mut piece.solids {
		solid.transform.position -= middle;
	}

	for link in &mut piece.links {
		if link.second == scene::NO_INDEX {
			link.second_anchor -= middle;
		}
	}
}

/// Every prop in the world, however it got there.
///
/// The scene's own, the menu's, a duplicate and one somebody welded to a wall
/// are all the same set as far as anything asking a question about props is
/// concerned, and what makes them one set is the layer rather than an array.
///
/// @param world - the bodies to walk
fn props(world: &World) -> Vec<BodyId> {
	world
		.bodies
		.iter()
		.filter(|(_, body)| on_layer(body, PROP_LAYERS))
		.map(|(id, _)| id)
		.collect()
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
	let loose = props(world);

	let landed = world
		.bodies
		.touches()
		.iter()
		.filter(|touch| touch.kind == TouchKind::Began)
		.filter(|touch| loose.iter().any(|&id| touch.names(id)))
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
	// where, not only how far: the physics gun attaches its beam at the point
	// the ray landed on, which is what makes carrying a long prop by its end
	// behave like carrying it by its end.
	state.picked_at = result.end.to_array();
	state.picked_normal = result.normal.to_array();
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

/// The physics gun: grab, carry, let go.
///
/// **A physics gun is a joint.** Not a spring bolted to the side of the solver,
/// not a teleport, and not a controller that writes a velocity every step: a
/// `Weld` pinned to a point in the world, made on the press and destroyed on
/// the release, whose far anchor is rewritten each step to sit in front of the
/// eye. Everything that makes it feel like a gun rather than like a magnet is
/// two numbers on that joint - a spring and a ceiling on what it may spend.
///
/// The left button, because the camera already has the right one. The wheel
/// moves what is held rather than the camera, and only while something is held.
///
/// @param world - the tables to read and the joint to make
/// @param yaw - which way the camera is looking
fn physgun(world: &mut World, yaw: f32) {
	let over_interface = world.ui.pointer_over();
	let input = world.input;
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let carrying = state.gun == PHYSGUN;

	if carrying && input.button_pressed(Button::Left) && !over_interface {
		grab(world, yaw, "");
	}

	if carrying && input.button_released(Button::Left) {
		release(world);
	}

	carry(world, yaw);
}

/// Takes hold of whatever the crosshair is on.
///
/// Reads what [`pick`] already found rather than tracing again, which is not
/// only cheaper: the pick ray is *unfiltered*, so a prop behind a wall is not
/// what it found and cannot be grabbed through one. A trace narrowed to the
/// prop layer would reach straight through the map, which is the wrong kind of
/// gun.
///
/// @param world - the picked body, and the joint table
/// @param yaw - which way the camera is looking, remembered for the carry
/// @param named - a body to take instead of the one under the crosshair
fn grab(world: &mut World, yaw: f32, named: &str) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);

	if state.hold.is_some() {
		return;
	}

	let (body, distance, at) = (state.picked_body, state.picked_distance, state.picked_at);
	let at = Vec3::from_array(at);

	// naming one takes it by its middle from wherever the player is standing,
	// which is what a console has instead of a crosshair. The pointer is the
	// ordinary way in and this is the one a script can drive.
	let (body, at, distance) = if named.is_empty() {
		(body, at, distance)
	} else {
		let Some((found, solid)) = world
			.bodies
			.iter()
			.find(|(id, _)| world.bodies.name(*id) == named)
		else {
			warn!(name = named, "nothing in the world is called that");

			return;
		};
		let middle = solid.transform.position;
		let eye = world.camera.position;

		(found, middle, (middle - eye).length())
	};

	let Some(solid) = world.bodies.get(body) else {
		return;
	};

	// taking hold of a frozen prop lets it go first, which is what the field
	// does and is the only thing that could be meant: the alternative is a gun
	// that silently refuses half the props in the yard.
	if frozen(solid) {
		unfreeze(world, body);
	}

	let Some(solid) = world.bodies.get(body) else {
		return;
	};

	// only something the solver owns, and only something on the prop layer.
	// The map is neither, which is what stops the gun picking up the hangar.
	if !solid.movable() || !on_layer(solid, PROP_LAYERS) {
		return;
	}

	let placed = solid.transform;
	let mass = solid.mass.max(Body::MASS);
	// the anchor is rotated into the body's own space and *not* scaled, because
	// that is what the solver does with it on the way back out.
	let anchor = placed.rotation.inverse() * (at - placed.position);
	let rest = placed.rotation;

	// a sleeping body is a body of no inverse mass, which is to say a wall, and
	// a joint that pulled on one would pull against infinity. Nothing else
	// wakes it: the solver rouses the ends of a joint only when one of them is
	// already moving.
	if let Some(solid) = world.bodies.get_mut(body) {
		solid.sleeping = false;
	}

	let gravity = world.gravity.length().max(1.0);
	let dt = world.dt.max(f32::EPSILON);
	let weight = mass * gravity * dt;

	let joint = world.join(
		Joint::weld(body, BodyId::NONE, (anchor, at))
			.sprung(HOLD_STIFFNESS, HOLD_DAMPING)
			.capped(weight * HOLD_STRENGTH, weight * HOLD_TWIST),
	);

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.hold = joint;
	state.held = body;
	state.hold_distance = distance.clamp(HOLD_RANGE.0, HOLD_RANGE.1);
	state.hold_anchor = anchor.to_array();
	state.hold_rest = rest.to_array();
	state.hold_yaw = yaw;

	trace!(body = body.slot(), distance = state.hold_distance, "grabbed");
}

/// Lets go of whatever is held.
///
/// The prop keeps whatever speed the joint had given it, which is what throwing
/// something is: there is no separate throw, and there does not need to be.
///
/// @param world - the joint table
fn release(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (joint, body) = (state.hold, state.held);

	if !joint.is_some() {
		return;
	}

	state.hold = JointId::NONE;
	state.held = BodyId::NONE;
	world.joints.despawn(joint);

	trace!(body = body.slot(), "let go");
}

/// Moves the point a held prop is being pulled towards, and draws the beam.
///
/// Three things happen here and each is one line of the reason the gun is a
/// joint at all. The far anchor is rewritten, so the prop follows the eye. The
/// `rest` is rewritten from the *yaw alone*, so the prop turns with the player
/// and does not roll when the player looks up. And the body is kept awake,
/// because a prop carried steadily is a prop that has been still for a while
/// and the solver would otherwise put it to sleep in mid-air.
///
/// **The two numbers on the joint were measured rather than chosen.** @ref
/// `colby-known-gaps` for the grid.
///
/// @param world - the joint to move and the table the beam goes in
/// @param yaw - which way the camera is looking now
fn carry(world: &mut World, yaw: f32) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (joint, body, distance) = (state.hold, state.held, state.hold_distance);
	let (rest, held_yaw) = (state.hold_rest, state.hold_yaw);
	let (player, anchor) = (state.player, state.hold_anchor);

	if !joint.is_some() {
		return;
	}

	// the prop was taken out from under the gun by something else - the hole,
	// a cleanup, a scene reloading. Nothing to carry and a joint to drop.
	if world.bodies.get(body).is_none() {
		release(world);

		return;
	}

	let camera = world.camera;
	let viewport = world.ui.viewport();
	let pointer = world.ui.pointer();
	let aim = if pointer.cmpge(Vec2::ZERO).all() && pointer.cmple(viewport).all() {
		pointer
	} else {
		viewport * 0.5
	};
	let target = camera.position + camera.pixel_direction(aim, viewport) * distance;
	let turned = Quat::from_rotation_y(yaw - held_yaw) * Quat::from_array(rest);

	if let Some(link) = world.joints.get_mut(joint) {
		link.second_anchor = target;
		// a world-pinned weld reads `rest` against an identity rotation on the
		// far side, so the orientation it holds the body at is the inverse of
		// what is written here. @ref `Joint::rest`.
		link.rest = turned.inverse();
	}

	let Some(solid) = world.bodies.get_mut(body) else {
		return;
	};

	solid.sleeping = false;
	let placed = solid.transform;
	let grip = placed.position + placed.rotation * Vec3::from_array(anchor);

	// a line rather than a beam, because there is no transparency in the scene
	// pass and a solid quad pretending to be one would look worse than a line
	// that is honestly a line. @ref `colby-direction`, step fourteen.
	let ahead = Vec3::new(-yaw.sin(), 0.0, -yaw.cos());
	let muzzle = world
		.entities
		.transform(player)
		.map_or(camera.position, |it| it.position + Vec3::Y * MUZZLE_AT.1 + ahead * MUZZLE_AT.0);

	world.debug.line(muzzle, grip, debug::CYAN);
	world.debug.point(grip, MUZZLE_MARK, debug::CYAN);
}

/// The toolgun: five modes on one weapon.
///
/// **Q swaps the two guns and the number keys pick a mode**, and picking one
/// puts the toolgun in your hands, because reaching for a tool is what saying
/// which tool means. The left button then means one thing whichever gun is out:
/// use what you are holding.
///
/// @param world - the tables to read and change
fn toolgun(world: &mut World) {
	let input = world.input;
	let over_interface = world.ui.pointer_over();

	if input.pressed(Key::Q) {
		// putting a gun away drops what it was holding, which is the only thing
		// it could mean. Without this the prop stays on the end of a joint
		// nothing is driving, and the next click of the *other* gun lets go of
		// it as a side effect.
		release(world);
		let (state, _) = world.state.get::<State>(STATE_LAYOUT);
		state.gun = if state.gun == PHYSGUN { TOOLGUN } else { PHYSGUN };
		forget_first(world);
	}

	for (index, key) in [Key::Digit1, Key::Digit2, Key::Digit3, Key::Digit4, Key::Digit5]
		.into_iter()
		.enumerate()
	{
		if !input.pressed(key) {
			continue;
		}

		release(world);
		let (state, _) = world.state.get::<State>(STATE_LAYOUT);
		state.gun = TOOLGUN;
		state.tool = u32::try_from(index).unwrap_or(0);
		forget_first(world);
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	if state.gun != TOOLGUN {
		return;
	}

	if input.button_pressed(Button::Left) && !over_interface {
		let (body, at, normal) = (state.picked_body, state.picked_at, state.picked_normal);
		use_tool(world, body, Vec3::from_array(at), Vec3::from_array(normal));
	}

	mark_pending(world);
}

/// Does whatever the gun is set to, to whatever was named.
///
/// The one entry point, so that a click and a console line cannot come to mean
/// different things. Everything below it is a named function over the world
/// with no input in it at all, which is the shape replication will want and is
/// what lets a script drive the whole weapon.
///
/// @param world - the tables to change
/// @param body - what was hit
/// @param at - where on it, in the world
/// @param normal - which way the surface faced there
fn use_tool(world: &mut World, body: BodyId, at: Vec3, normal: Vec3) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);

	match state.tool {
		| REMOVER => {
			remove_prop(world, body);
		},
		| FREEZE => {
			freeze(world, body);
		},
		| _ => link(world, body, at, normal),
	}
}

/// The half of the toolgun that takes two clicks.
///
/// The first names one end and is remembered; the second makes the joint and
/// forgets it. A second click on the map rather than on a prop pins the joint
/// to a point in the world instead, which is how a prop is nailed to a wall -
/// and it costs nothing, because a joint whose second body is nothing has meant
/// exactly that since joints existed.
///
/// @param world - the tables to change
/// @param body - what was hit
/// @param at - where on it, in the world
/// @param normal - which way the surface faced there
fn link(world: &mut World, body: BodyId, at: Vec3, normal: Vec3) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (first, tool) = (state.tool_first, state.tool);

	if !first.is_some() {
		// the first end has to be something the solver owns: a joint between
		// two pieces of the map holds nothing that could move.
		if !world.bodies.get(body).is_some_and(Body::movable) {
			return;
		}

		let (state, _) = world.state.get::<State>(STATE_LAYOUT);
		state.tool_first = body;
		state.tool_at = at.to_array();
		state.tool_normal = normal.to_array();

		trace!(
			tool = TOOLS[usize::try_from(tool).unwrap_or(0)],
			body = body.slot(),
			"first end"
		);

		return;
	}

	if body == first {
		forget_first(world);

		return;
	}

	let started = Vec3::from_array(state.tool_at);
	let facing = Vec3::from_array(state.tool_normal);
	let Some(one) = world.bodies.get(first).copied() else {
		forget_first(world);

		return;
	};

	// a second end the solver does not own is the world, and the anchor on that
	// side is then a point rather than a place on something.
	let solid = world.bodies.get(body).copied();
	let movable = solid.is_some_and(|it| it.movable());
	let second = if movable { body } else { BodyId::NONE };
	let far = match solid.filter(|_| movable) {
		| Some(it) => anchor_on(&it, at),
		| None => at,
	};

	let joint = match tool {
		| HINGE => Joint::axis(
			first,
			second,
			(anchor_on(&one, started), far),
			// the face that was clicked. A hinge whose axis is the surface
			// normal is the one a person means by pointing at a face, and it is
			// what makes a door out of a plank and a wall.
			one.transform.rotation.inverse() * facing,
		),
		| ROPE =>
			Joint::rope(first, second, (anchor_on(&one, started), far), (at - started).length()),
		| _ => Joint::weld(first, second, (anchor_on(&one, started), far))
			.sprung(WELD_STIFFNESS, WELD_DAMPING),
	};

	let made = world.join(joint);
	let name = TOOLS[usize::try_from(tool).unwrap_or(0)];
	trace!(tool = name, joint = made.slot(), first = first.slot(), "joined");

	forget_first(world);
}

/// Where a point on a body is, in that body's own space.
///
/// Rotated and *not* scaled, because that is what the solver does with an
/// anchor on the way back out. Getting it wrong is a joint that holds somewhere
/// other than where it was made.
///
/// @param solid - the body
/// @param at - a point on it, in the world
fn anchor_on(solid: &Body, at: Vec3) -> Vec3 {
	solid.transform.rotation.inverse() * (at - solid.transform.position)
}

/// Drops the end a two-click tool was waiting on.
///
/// @param world - the arena
fn forget_first(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.tool_first = BodyId::NONE;
}

/// Draws the end a two-click tool is waiting on, and the line it would make.
///
/// @param world - the arena and the table the segments go in
fn mark_pending(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (first, at, picked) = (state.tool_first, state.tool_at, state.picked_at);

	if !first.is_some() {
		return;
	}

	let (at, picked) = (Vec3::from_array(at), Vec3::from_array(picked));
	world.debug.point(at, PENDING_MARK, PENDING_COLOR);
	world.debug.line(at, picked, PENDING_COLOR);
}

/// Takes one prop out of the world, joints and all.
///
/// @param world - the tables to remove from
/// @param body - the prop to remove
/// @return whether one went
fn remove_prop(world: &mut World, body: BodyId) -> bool {
	let Some(solid) = world.bodies.get(body) else {
		return false;
	};

	if !on_layer(solid, PROP_LAYERS) {
		return false;
	}

	let entity = solid.entity;
	// before the body, not after: a joint naming a slot that has been handed
	// out again holds whoever moved into it.
	world.joints.forget(body);
	world.bodies.despawn(body);
	world.entities.despawn(entity);

	trace!(body = body.slot(), "removed");

	true
}

/// Whether a prop has been frozen.
///
/// **Frozen is a body kind and not a flag.** There is no field anywhere saying
/// so: a frozen prop is a [`BodyKind::Kinematic`] one, which in this engine
/// means gameplay owns its transform and the solver leaves it alone. Nothing
/// writes a frozen prop's transform, so it stays exactly where it was, and
/// everything else piles on it as though it were the map. That is what freezing
/// *is*, and it is the first consumer `Kinematic` has ever had that is not the
/// player.
///
/// The layer is part of the question because the player's own box is kinematic
/// too, and it is not frozen.
///
/// @param body - the body to ask about
fn frozen(body: &Body) -> bool {
	matches!(body.kind, BodyKind::Kinematic) && on_layer(body, PROP_LAYERS)
}

/// Freezes and unfreezes, on one key.
///
/// Holding something and pressing it freezes that and lets go, which is the one
/// gesture worth having: you carry a prop into place and nail it there without
/// a second thought about which button. With nothing held it toggles whatever
/// the crosshair is on.
///
/// @param world - the bodies to change
fn freezer(world: &mut World) {
	if !world.input.pressed(Key::R) {
		return;
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (held, picked) = (state.held, state.picked_body);

	if state.hold.is_some() {
		freeze(world, held);

		return;
	}

	if world.bodies.get(picked).is_some_and(frozen) {
		unfreeze(world, picked);
	} else {
		freeze(world, picked);
	}
}

/// Stops a prop where it stands.
///
/// The speed goes with the motion: a body that came back to life carrying the
/// velocity it had when it was frozen would leap the moment it was thawed, and
/// the whole point of freezing something is that you have decided where it
/// belongs.
///
/// @param world - the body table
/// @param body - what to freeze
/// @return whether anything was frozen
fn freeze(world: &mut World, body: BodyId) -> bool {
	let Some(solid) = world.bodies.get(body) else {
		return false;
	};

	if !solid.movable() || !on_layer(solid, PROP_LAYERS) {
		return false;
	}

	// letting go is part of freezing rather than something the key does before
	// it, so that the console and the key mean the same thing. A gun still
	// holding a kinematic body is a gun pulling on a wall: the joint spends its
	// whole ceiling every step and nothing moves.
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	if state.held == body {
		release(world);
	}

	let Some(solid) = world.bodies.get_mut(body) else {
		return false;
	};

	solid.kind = BodyKind::Kinematic;
	solid.velocity = Vec3::ZERO;
	solid.angular = Vec3::ZERO;
	solid.sleeping = false;

	trace!(body = body.slot(), "frozen");

	true
}

/// Lets one go again.
///
/// Awake rather than asleep, because a prop that has just been handed back to
/// the solver has not been still for any length of time - it has been *held*
/// still, which is a different thing and is exactly the case the sleeping rule
/// is not allowed to confuse.
///
/// @param world - the body table
/// @param body - what to release
/// @return whether anything was thawed
fn unfreeze(world: &mut World, body: BodyId) -> bool {
	let Some(solid) = world.bodies.get_mut(body) else {
		return false;
	};

	if !frozen(solid) {
		return false;
	}

	solid.kind = BodyKind::Dynamic;
	solid.sleeping = false;

	trace!(body = body.slot(), "thawed");

	true
}

/// Lets every frozen prop go at once.
///
/// The same walk the map and the sweep do, one question along: what is frozen
/// is a body on the prop layer of a certain kind, so the set of them is found
/// rather than remembered.
///
/// @param world - the bodies to walk
/// @return how many were let go
fn thaw(world: &mut World) -> usize {
	let stuck: Vec<BodyId> = world
		.bodies
		.iter()
		.filter(|(_, body)| frozen(body))
		.map(|(id, _)| id)
		.collect();

	for body in &stuck {
		unfreeze(world, *body);
	}

	stuck.len()
}

/// Outlines every frozen prop.
///
/// A prop is a box or a ball and never a mesh, because a mesh body is never
/// dynamic and so could never have been frozen. Two arms and no third.
///
/// @param world - the bodies to walk and the table the segments go in
fn outline_frozen(world: &mut World) {
	let stuck: Vec<BodyId> = world
		.bodies
		.iter()
		.filter(|(_, body)| frozen(body))
		.map(|(id, _)| id)
		.collect();

	for body in stuck {
		outline(world, body, FROZEN_COLOR);
	}
}

/// Draws one body's shape as segments.
///
/// The one place either outline is drawn, so that a frozen prop and the thing
/// under the crosshair cannot come to disagree about what a shape looks like.
/// A mesh draws nothing: the only mesh bodies here are the map's, which is
/// boxes, and a triangle soup outlined is a wall of noise rather than an
/// answer.
///
/// @param world - the body to read and the table the segments go in
/// @param body - what to draw
/// @param color - what to draw it in
fn outline(world: &mut World, body: BodyId, color: Vec3) {
	let Some(solid) = world.bodies.get(body) else {
		return;
	};

	let (shape, placed) = (solid.shape, solid.transform);

	match shape.kind {
		| ShapeKind::Box => world.debug.cuboid(
			placed.position,
			shape.extents.abs() * placed.scale.abs(),
			placed.rotation,
			color,
		),
		| ShapeKind::Sphere => world.debug.ball(
			placed.position,
			shape.radius.abs() * placed.scale.abs().max_element(),
			color,
		),
		| ShapeKind::Mesh => {},
	}
}

/// Outlines whatever the pick ray found.
///
/// It used to brighten it instead, by writing the entity's color every step and
/// restoring it from whatever laid the thing out. That worked while there were
/// five props from one file and stopped the moment a prop could come from any
/// file in a directory, be duplicated, or be a piece of the map - because then
/// "put it back the way it was" needs a table of what every thing's color was,
/// which is a table nobody wants to keep. An outline remembers nothing.
///
/// @param world - the picked body and the table the segments go in
fn outline_picked(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let picked = state.picked_body;

	outline(world, picked, PICKED_COLOR);
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
	let grounded = state.player_grounded != 0;
	let (time, entities) = (world.time, world.entities.len());
	let landings = state.landings;
	let joints = world.joints.len();
	let contacts = world.contacts;
	let footing = if grounded { "on the ground" } else { "in the air" };
	let in_hand = if state.gun == TOOLGUN {
		let tool = TOOLS
			.get(usize::try_from(state.tool).unwrap_or(0))
			.copied()
			.unwrap_or(TOOLS[0]);
		let waiting = if state.tool_first.is_some() { ", one end" } else { "" };

		format!("toolgun: {tool}{waiting}")
	} else {
		"physgun".to_owned()
	};

	// counted rather than remembered, which is the same rule the map follows:
	// what a prop is is a body on a layer, so how many there are is a walk.
	let loose = props(world);
	let asleep = loose
		.iter()
		.filter(|&&id| {
			world
				.bodies
				.get(id)
				.is_some_and(|body| body.sleeping)
		})
		.count();
	let stuck = world
		.bodies
		.iter()
		.filter(|(_, body)| frozen(body))
		.count();
	let swallowed = state.swallowed;

	world
		.ui
		.set_text(hud, "time", &format!("{time:.1} s"));
	world
		.ui
		.set_text(hud, "entities", &format!("{entities}"));
	world.ui.set_text(hud, "hit", &hit);
	world.ui.set_text(
		hud,
		"physics",
		&format!("{asleep}/{} asleep, {contacts} hits", loose.len()),
	);
	world
		.ui
		.set_text(hud, "joints", &format!("{joints} held, {landings} landings"));
	world.ui.set_text(
		hud,
		"props",
		&format!("{} loose, {stuck} frozen, {swallowed} lost", loose.len()),
	);
	world.ui.set_text(hud, "player", footing);
	world.ui.set_text(hud, "gun", &in_hand);
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

/// `game.grab` - the console's way of pressing the left button.
///
/// # Safety
///
/// `world` must point to a live [`World`] owned by the host, and `args` to a
/// live argument list. The host removes this command from the table before
/// unloading the module it lives in.
unsafe extern "C-unwind" fn take(world: *mut World, args: *const Args) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };
	// SAFETY: the host guarantees a live argument list for the call.
	let named = unsafe { &*args }.rest();

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let yaw = state.yaw;

	grab(world, yaw, &named);
}

/// `game.release` - and of letting go of it.
///
/// # Safety
///
/// As [`take`].
unsafe extern "C-unwind" fn drop_held(world: *mut World, _args: *const Args) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };

	release(world);
}

/// The body a console command is about: the one named, or the one aimed at.
///
/// @param world - the bodies to search
/// @param named - a name, or empty for the crosshair
fn asked_for(world: &mut World, named: &str) -> BodyId {
	if named.is_empty() {
		let (state, _) = world.state.get::<State>(STATE_LAYOUT);

		return state.picked_body;
	}

	world
		.bodies
		.iter()
		.find(|(id, _)| world.bodies.name(*id) == named)
		.map_or(BodyId::NONE, |(id, _)| id)
}

/// `game.freeze` - the console's way of pressing the key.
///
/// # Safety
///
/// As [`take`].
unsafe extern "C-unwind" fn hold_still(world: *mut World, args: *const Args) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };
	// SAFETY: the host guarantees a live argument list for the call.
	let named = unsafe { &*args }.rest();
	let body = asked_for(world, &named);

	if !freeze(world, body) {
		warn!(name = named, "nothing there to freeze");
	}
}

/// `game.unfreeze` - and of pressing it again.
///
/// # Safety
///
/// As [`take`].
unsafe extern "C-unwind" fn let_go(world: *mut World, args: *const Args) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };
	// SAFETY: the host guarantees a live argument list for the call.
	let named = unsafe { &*args }.rest();
	let body = asked_for(world, &named);

	if !unfreeze(world, body) {
		warn!(name = named, "nothing there to thaw");
	}
}

/// `game.thaw` - every one of them at once.
///
/// # Safety
///
/// As [`take`].
unsafe extern "C-unwind" fn thaw_all(world: *mut World, _args: *const Args) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };

	info!(gone = thaw(world), "everything is moving again");
}

/// `game.spawn` - the console's way of clicking a row of the menu.
///
/// # Safety
///
/// As [`take`].
unsafe extern "C-unwind" fn put_one(world: *mut World, args: *const Args) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };
	// SAFETY: the host guarantees a live argument list for the call.
	let named = unsafe { &*args }.rest();

	if named.is_empty() {
		info!(offered = ?catalogue(world), "what there is");

		return;
	}

	spawn_prop(world, &named);
}

/// `game.tool` - the console's way of pressing a number key.
///
/// # Safety
///
/// As [`take`].
unsafe extern "C-unwind" fn pick_tool(world: *mut World, args: *const Args) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };
	// SAFETY: the host guarantees a live argument list for the call.
	let named = unsafe { &*args }.rest();

	let Some(index) = TOOLS.iter().position(|it| *it == named) else {
		info!(modes = ?TOOLS, "the toolgun's modes");

		return;
	};

	release(world);
	forget_first(world);
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.gun = TOOLGUN;
	state.tool = u32::try_from(index).unwrap_or(0);

	info!(tool = named, "in hand");
}

/// `game.apply` - and of clicking with it.
///
/// A named body is taken at its middle and by the way it is facing up, because
/// a console has no crosshair to have clicked a face with. That is the one
/// place this and the button differ, and it matters only to a hinge.
///
/// # Safety
///
/// As [`take`].
unsafe extern "C-unwind" fn use_it(world: *mut World, args: *const Args) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };
	// SAFETY: the host guarantees a live argument list for the call.
	let named = unsafe { &*args }.rest();
	let body = asked_for(world, &named);

	let Some(solid) = world.bodies.get(body).copied() else {
		warn!(name = named, "nothing there to use it on");

		return;
	};

	let (at, normal) = if named.is_empty() {
		let (state, _) = world.state.get::<State>(STATE_LAYOUT);

		(Vec3::from_array(state.picked_at), Vec3::from_array(state.picked_normal))
	} else {
		(solid.transform.position, Vec3::Y)
	};

	use_tool(world, body, at, normal);
}

/// `game.save <name>` - keeps a contraption as a prop.
///
/// Not `scene.save`, which is the host's and writes the *world* into `saves/`.
/// This one writes a piece of it into the asset tree, where a prop lives.
///
/// # Safety
///
/// As [`take`].
unsafe extern "C-unwind" fn keep_one(world: *mut World, args: *const Args) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };
	// SAFETY: the host guarantees a live argument list for the call.
	let named = unsafe { &*args }.rest();

	keep_build(world, &named);
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

	for (body, _) in &loose {
		remove_prop(world, *body);
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.props = [EntityId::NONE; PROPS];
	state.prop_bodies = [BodyId::NONE; PROPS];
	state.rope = JointId::NONE;
	state.picked = EntityId::NONE;
	state.picked_body = BodyId::NONE;
	// the gun was holding one of them. `carry` notices a body that has gone and
	// lets go by itself, so this is not load-bearing; it is the difference
	// between a handle that is stale for a step and one that never is.
	state.hold = JointId::NONE;
	state.held = BodyId::NONE;
	state.tool_first = BodyId::NONE;

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

/// Draws every joint in the world as a line between its two anchors.
///
/// It used to draw exactly one rope out of a handle in the arena, because there
/// was exactly one joint and the game had made it. The toolgun makes as many as
/// somebody clicks, so what is drawn is the table rather than a handle - and a
/// weld, a hinge and a rope all become visible for nothing.
///
/// The anchors come from the *bodies* rather than from the entities, so the
/// ends are where the solver put them rather than a step behind. A joint whose
/// second body is nothing is pinned to a point, and its second anchor is that
/// point rather than a place on something.
///
/// @param world - the joints to read and the table the segments go in
fn draw_joints(world: &mut World) {
	let held: Vec<(Vec3, Vec3)> = world
		.joints
		.iter()
		.filter_map(|(_, joint)| {
			let one = world.bodies.get(joint.first)?.transform;
			let far = world
				.bodies
				.get(joint.second)
				.map_or(joint.second_anchor, |other| {
					other.transform.position + other.transform.rotation * joint.second_anchor
				});

			Some((one.position + one.rotation * joint.first_anchor, far))
		})
		.collect();

	for (from, to) in held {
		world.debug.line(from, to, JOINT_COLOR);
		world.debug.point(to, HOOK_SIZE, JOINT_COLOR);
	}
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
