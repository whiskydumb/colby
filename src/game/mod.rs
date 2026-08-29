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

use std::f32::consts::TAU;

use colby_core::{
	abi::{
		ABI_VERSION, Args, Body, BodyId, BodyKind, Button, EntityId, GameApi, Joint, JointId,
		Key, Layers, Length, Material, MaterialId, MeshId, Motion, PanelId, Renderable, Shape,
		TouchKind, TraceInfo, Transform, Value, World, character, debug,
	},
	bytemuck::{Pod, Zeroable},
	glam::{Quat, Vec2, Vec3},
	info, mod_ctor, mod_dtor, trace, warn,
};

mod_ctor! {}
mod_dtor! {}

/// How many cubes stand in the ring.
const RING: usize = 8;

/// How far the ring's cubes stand from the middle.
const RING_RADIUS: f32 = 2.6;

/// How fast the ring turns, in radians per second.
///
/// The default only: `init` registers it as a console variable, `update` reads
/// that variable, and `game.spin_rate 3` in the console turns the ring faster
/// without a rebuild. A value typed there survives a reload, and this constant
/// is what it goes back to.
const SPIN_RATE: f32 = 0.4;

/// The console variable [`SPIN_RATE`] is the default of.
const SPIN_RATE_CVAR: &str = "game.spin_rate";

/// How high the ring's cubes bob, and how fast.
const BOB: (f32, f32) = (0.35, 1.6);

/// How wide the floor is.
const FLOOR_SIZE: f32 = 14.0;

/// How high the floor's surface sits.
const FLOOR_Y: f32 = -0.5;

/// How big the shape in the middle is drawn.
const CENTER_SCALE: f32 = 0.9;

/// Where the player stands when the scene is put back.
const PLAYER_START: Vec3 = Vec3::new(1.4, 0.4, 1.4);

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
const START_ORBIT: (f32, f32, f32) = (0.6, 0.5, 9.0);

/// What the floor is tinted, on top of its texture.
///
/// Close to white: the image already carries the color, and a tint is for
/// nudging it rather than replacing it.
const FLOOR_COLOR: Vec3 = Vec3::new(0.82, 0.84, 0.88);

/// The color of the box the player is.
const PLAYER_COLOR: Vec3 = Vec3::new(0.30, 0.85, 0.55);

/// What the shape in the middle is colored.
const CENTER_COLOR: Vec3 = Vec3::new(0.95, 0.76, 0.20);

/// The asset the middle of the scene is made of.
///
/// A name, resolved against the host's registry every time this module is
/// swapped in. `assets/meshes/crystal.obj` compiles to this name; edit that
/// file and the shape changes under a running process, because the handle the
/// name resolves to stays the same and only the geometry behind it moves.
const CENTER_MESH: &str = "meshes/crystal";

/// What the middle falls back to when the asset is not there.
///
/// Not an error: a checkout with no `assets/` directory, or one where the
/// crystal failed to compile, should still put something on screen rather than
/// a hole where the scene was.
const CENTER_FALLBACK: MeshId = MeshId::CUBE;

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
const LAMP_AT: Vec3 = Vec3::new(1.3, 0.0, 4.6);

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

/// Where each prop is let go from, how big it is, and whether it is a ball.
///
/// Three boxes over each other and two balls beside them: the boxes show that a
/// stack settles and stays settled, and the balls show that it is a solver
/// rather than a set of rules about boxes. They are let go from a height rather
/// than placed at rest so that pressing space is worth doing.
const PROP_DROPS: [(Vec3, f32, bool); PROPS] = [
	(Vec3::new(4.20, 0.20, -0.90), 0.62, false),
	(Vec3::new(4.20, 1.05, -0.90), 0.62, false),
	(Vec3::new(4.20, 1.90, -0.90), 0.62, false),
	(Vec3::new(5.60, 2.60, 0.60), 0.70, true),
	(Vec3::new(5.40, 2.30, -1.50), 0.52, true),
];

/// Which prop hangs from a rope.
const SWINGING: usize = 3;

/// Where that rope is tied, in the world.
const HOOK: Vec3 = Vec3::new(4.60, 3.20, 0.60);

/// How long it is.
const ROPE: f32 = 1.4;

/// How big the cross marking the rope's hook is, in world units.
const HOOK_SIZE: f32 = 0.08;

/// The layer the floor, the crystal and the ring are on.
///
/// Layers are numbers here rather than names in a manifest, and a game naming
/// its own is the whole of the mechanism: the engine holds two bitmasks per
/// body and has no opinion about what any bit means.
const LAYER_WORLD: u32 = 0;

/// The layer everything the solver throws about is on.
const LAYER_PROP: u32 = 1;

/// The layer the player's box is on.
const LAYER_PLAYER: u32 = 2;

/// The layer the trigger volume is on.
const LAYER_TRIGGER: u32 = 3;

/// What the scenery is: on the world layer, in the way of everything.
const WORLD_LAYERS: Layers = Layers::single(LAYER_WORLD);

/// What a prop is.
const PROP_LAYERS: Layers = Layers::single(LAYER_PROP);

/// What the player is, and what it walks into.
///
/// Everything except the trigger, which it would pass through anyway - a
/// sensor pushes nothing - but saying so here is what makes the trigger's own
/// mask readable as a decision rather than as a coincidence.
const PLAYER_LAYERS: Layers = Layers::new(
	Layers::bit(LAYER_PLAYER),
	Layers::bit(LAYER_WORLD) | Layers::bit(LAYER_PROP) | Layers::bit(LAYER_PLAYER),
);

/// What the trigger is: interested in props and in nothing else.
const TRIGGER_LAYERS: Layers = Layers::new(Layers::bit(LAYER_TRIGGER), Layers::bit(LAYER_PROP));

/// Where the trigger volume sits, in the world.
///
/// Over the corner the props are dropped into rather than somewhere tidy, so
/// that a screenshot taken at any moment has something in it. A volume nothing
/// ever enters demonstrates nothing.
const TRIGGER_CENTER: Vec3 = Vec3::new(4.80, 0.45, -0.35);

/// Its half-extents.
///
/// Sized to the corner the props are dropped into rather than around what it
/// must not touch: it is on [`TRIGGER_LAYERS`], which is interested in props
/// alone, so the floor slab it now reaches through and the ring it now reaches
/// into are not in its list. It still ends below the ball that hangs, which is
/// what makes the count four out of five rather than five out of five - and
/// that is a fact about where the ball is rather than about the geometry
/// underneath.
const TRIGGER_EXTENTS: Vec3 = Vec3::new(1.75, 1.05, 1.55);

/// How far above whatever the pick ray found its label is written.
const LABEL_LIFT: f32 = 0.55;

/// How heavy a prop is.
const PROP_MASS: f32 = 1.0;

/// How much of an impact a prop gives back.
const PROP_BOUNCE: f32 = 0.22;

/// How hard a prop is to slide.
const PROP_GRIP: f32 = 0.62;

/// How far the pick ray reaches, in world units.
///
/// Longer than the camera ever gets from the scene, so that "nothing there" is
/// really nothing rather than the ray running out.
const REACH: f32 = 60.0;

/// How much brighter whatever is under the cursor is drawn.
const HIGHLIGHT: f32 = 1.45;

/// How thick the floor's collider is.
///
/// The floor is drawn as a quad, which has no thickness at all, and a surface
/// a trace can pass through edge-on is not a floor. Half a tenth of a unit is
/// enough to be solid and too little to see.
///
/// The slab is hung *below* the quad rather than centered on it, which is why
/// the floor is the one body in this scene with no entity behind it: a shape
/// has no offset of its own, so the only way to put a surface where a person
/// drew one is to move the body, and a body that follows an entity cannot be
/// moved. Getting this wrong is props resting five centimeters above the
/// floor - small enough to miss and quite visible once seen.
const FLOOR_THICKNESS: f32 = 0.05;

/// How many times that image repeats across the floor.
///
/// The quad's own coordinates run zero to one whatever it is scaled to, so this
/// is what decides how big a tile looks. Six across fourteen units is about
/// two units a tile.
const FLOOR_TILES: f32 = 6.0;

/// The game's own state, kept in the host's arena.
///
/// Add a field, bump [`STATE_LAYOUT`], save: the arena zeroes itself and the
/// game starts over, with `colby_core` untouched and the process still running.
/// That is the whole reason this is an arena rather than fields on `World`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct State {
	/// Radians the ring has turned so far.
	spin: f32,

	/// Where the camera sits, as an orbit around its target.
	yaw: f32,

	/// Radians above the horizon.
	pitch: f32,

	/// How far from the target.
	distance: f32,

	/// The floor, so the game moves exactly the entities it made.
	floor: EntityId,

	/// The cube in the middle.
	center: EntityId,

	/// The cubes standing around it.
	ring: [EntityId; RING],

	/// The panel the interface is shown in.
	hud: PanelId,

	/// The slab under the floor.
	floor_body: BodyId,

	/// The shape in the middle, collided against as its own triangles.
	center_body: BodyId,

	/// One box per cube of the ring.
	ring_bodies: [BodyId; RING],

	/// The loose props the solver owns.
	props: [EntityId; PROPS],

	/// Their bodies.
	prop_bodies: [BodyId; PROPS],

	/// The rope one of them hangs from.
	rope: JointId,

	/// The volume that notices what is in it and pushes nothing.
	trigger: BodyId,

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

	/// One entity per piece of the model in the corner.
	///
	/// Spawned once and never moved, so it is the one thing in this scene the
	/// editor can drag and have it stay put.
	lamp: [EntityId; LAMP_PIECES],

	/// Whether the `hold` button is holding the ring still. Not a `bool`
	/// because [`Pod`] wants every bit pattern to be a valid value.
	holding: u32,
}

/// The version of [`State`]'s layout. Bump it whenever the struct changes.
///
/// Forgetting to is not unsound - `State` is `Pod`, so every bit pattern is a
/// valid `State` - but the values will be yesterday's bytes read through
/// today's fields.
const STATE_LAYOUT: u64 = 12;

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
		state.spin = 0.0;
		(state.yaw, state.pitch, state.distance) = START_ORBIT;

		// the arena was reset, so the handles it held are gone. Anything the
		// old build spawned would be orphaned; clear the table and start over.
		world.entities.clear();
		// and the bodies with them: a body naming an entity that no longer
		// exists drives nothing, and would still be traced against.
		world.bodies.clear();
		state.floor = world.entities.spawn();
		state.center = world.entities.spawn();
		for slot in &mut state.ring {
			*slot = world.entities.spawn();
		}
		for slot in &mut state.props {
			*slot = world.entities.spawn();
		}
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
	world.cvars.saved(
		SPIN_RATE_CVAR,
		Value::Float(SPIN_RATE),
		"how fast the ring turns, in radians a second",
	);
	world.cvars.command(
		"game.reset",
		reset,
		"put the scene and the camera back where they started",
	);
	world
		.cvars
		.command("game.hold", hold, "stop the ring turning, or let it turn again");

	// on every load, not only a fresh one: the mesh a name resolves to is the
	// host's business, and an asset that appeared since the last swap should be
	// picked up by this one.
	dress(world);
	place(world);
	if fresh {
		// only on a fresh arena: a reload should leave the pile exactly where
		// the last build left it, which is the whole point of the state living
		// in the host.
		drop_props(world);
	}
	collide(world);

	// showing a document is idempotent by name, so a reload finds the same
	// panel with the same text still in it rather than stacking a second copy.
	let hud = world.ui.show(HUD);
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.hud = hud;

	info!(
		reloads = world.reloads,
		ring = RING,
		entities = world.entities.len(),
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
	let dt = world.dt;

	// the ring's speed comes from the console variable, falling back to the
	// constant that is its default. This is the only line that changes when a
	// number becomes something a person can turn while watching it.
	let spin_rate = world
		.cvars
		.float(SPIN_RATE_CVAR)
		.unwrap_or(SPIN_RATE);

	interface(world, spin_rate);

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

	// the left button holds the ring still, and so does the interface's own
	// button - one behavior, two ways to ask for it.
	let held = state.holding != 0 || (input.button_held(Button::Left) && !over_interface);
	if !held {
		state.spin = dt.mul_add(spin_rate, state.spin);
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

	place(world);
	pick(world);
	light_up(world);
	label_pick(world);
	trigger(world);
	menu(world);
	count_landings(world);
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
		.set_renderable(entity, Renderable::of(mesh, material, prop_color(index % PROPS)));

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

/// Outlines the trigger volume and writes what is in it above it.
///
/// Drawn by the game rather than left to `phys.draw_shapes`, for the reason
/// the rope is drawn: a screenshot has no console to turn anything on with,
/// and a trigger nobody can see is indistinguishable from one that is not
/// working. Magenta because that is the color the solver's own drawing gives a
/// sensor, and two names for one thing is how a picture stops meaning
/// anything.
///
/// @param world - the overlap list to read and the table the lines go in
fn trigger(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let volume = state.trigger;

	if !volume.is_some() {
		return;
	}

	let inside = world.bodies.inside(volume).count();
	let lifted = TRIGGER_CENTER + Vec3::new(0.0, TRIGGER_EXTENTS.y + LABEL_LIFT, 0.0);
	let text = if inside == 0 {
		"trigger: empty".to_owned()
	} else {
		format!("trigger: {inside}")
	};

	world
		.debug
		.cuboid(TRIGGER_CENTER, TRIGGER_EXTENTS, Quat::IDENTITY, debug::MAGENTA);
	world.debug.label(lifted, &text, debug::MAGENTA);
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

	let (picked, text) = (state.picked, named(state));
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

	if moved {
		// one line per change rather than one per step, so a live run can be
		// read. This is the only way to see the query answering in a window,
		// and in particular the only way to see it still answering after a
		// module reload - the table is the host's, and a swap must not disturb
		// it. @ref the pre-commit audit list.
		trace!(hit = %named(state), fraction = result.fraction, "pick");
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
	let (center, ring, picked) = (state.center, state.ring, state.picked);
	let props = state.props;

	let lit = |id: EntityId, color: Vec3| {
		if id == picked && id.is_some() {
			color * HIGHLIGHT
		} else {
			color
		}
	};

	if let Some(renderable) = world.entities.renderable_mut(center) {
		renderable.color = lit(center, Vec3::ONE);
	}

	for (index, id) in ring.into_iter().enumerate() {
		if let Some(renderable) = world.entities.renderable_mut(id) {
			renderable.color = lit(id, hue(index));
		}
	}

	for (index, id) in props.into_iter().enumerate() {
		if let Some(renderable) = world.entities.renderable_mut(id) {
			renderable.color = lit(id, prop_color(index));
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
/// @param spin_rate - what the ring is turning at, for the readout
fn interface(world: &mut World, spin_rate: f32) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let mut hud = state.hud;

	// re-resolved when it is nothing, for the same reason the crystal's mesh
	// is: the document may have been compiled after this module was loaded.
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
	let spin = state.spin;
	let hit = named(state);
	let props = state.props;
	let volume = state.trigger;
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
	let inside = world.bodies.inside(volume).count();
	let footing = if grounded { "on the ground" } else { "in the air" };

	world
		.ui
		.set_text(hud, "time", &format!("{time:.1} s"));
	world
		.ui
		.set_text(hud, "entities", &format!("{entities}"));
	world
		.ui
		.set_text(hud, "rate", &format!("{spin_rate:.2}"));
	world.ui.set_text(hud, "hit", &hit);
	world
		.ui
		.set_text(hud, "physics", &format!("{asleep}/{PROPS} asleep, {contacts} hits"));
	world
		.ui
		.set_text(hud, "joints", &format!("{joints} held, {landings} landings"));
	world
		.ui
		.set_text(hud, "trigger", &format!("{inside} inside"));
	world.ui.set_text(hud, "player", footing);

	// the one thing a class cannot say: how full the bar is. A stylesheet
	// decides what the bar looks like and the game decides how long it is.
	let turn = (spin / TAU).fract() * 100.0;
	if let Some(style) = world.ui.style_mut(hud, "fill") {
		style.width = Some(Length::Percent(turn.max(1.0)));
	}
}

/// Puts the scene and the camera back where they started.
///
/// What space does, and what `game.reset` does.
///
/// @param world - the state to put back
fn recenter(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.spin = 0.0;
	(state.yaw, state.pitch, state.distance) = START_ORBIT;

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (player, body) = (state.player, state.player_body);
	let (dropped, dropped_bodies) = (state.dropped, state.dropped_bodies);
	state.player_grounded = 0;
	state.dropped = [EntityId::NONE; SPAWNED];
	state.dropped_bodies = [BodyId::NONE; SPAWNED];
	state.dropped_next = 0;

	for (entity, solid) in dropped.into_iter().zip(dropped_bodies) {
		world.bodies.despawn(solid);
		world.entities.despawn(entity);
	}

	if let Some(transform) = world.entities.transform_mut(player) {
		transform.position = PLAYER_START;
		transform.scale = PLAYER_EXTENTS * 2.0;
	}

	if let Some(solid) = world.bodies.get_mut(body) {
		solid.transform.position = PLAYER_START;
		solid.velocity = Vec3::ZERO;
	}

	let (yaw, pitch, distance) = START_ORBIT;
	world.camera.target = PLAYER_START + Vec3::Y * EYE_LIFT;
	world.camera.orbit(yaw, pitch, distance);

	place(world);
	drop_props(world);

	// a cut rather than a move: the ring's angle, the camera's target and its
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

/// `game.hold` - what the interface's own button asks for.
///
/// A console command rather than a flag the interface writes, because the
/// console is the one surface a document's script can reach: the button in
/// `hud.lua` keeps its own idea of whether it is pressed and says this when it
/// changes. Typing `game.hold` has always been the other way to ask.
///
/// # Safety
///
/// `world` must point to a live [`World`] owned by the host, and `args` to a
/// live argument list. The host removes this command from the table before
/// unloading the module it lives in.
unsafe extern "C-unwind" fn hold(world: *mut World, _args: *const Args) {
	// SAFETY: as init.
	let world = unsafe { &mut *world };

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.holding ^= 1;
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

/// Gives every entity of the scene its shape and color.
///
/// Cheap enough to run on every load rather than only on a fresh arena, which
/// is what lets the middle pick up an asset the host has since compiled.
fn dress(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (floor, center, ring) = (state.floor, state.center, state.ring);

	// asked for by name. The registry answers with a handle the game keeps for
	// this load only - resolving again next time costs nothing and means an
	// asset that arrived late is still found.
	let crystal = world.meshes.find(CENTER_MESH);
	let crystal = if crystal.is_some() { crystal } else { CENTER_FALLBACK };
	let tiles = world.textures.find(FLOOR_TEXTURE);
	let tile_normals = world.textures.find(FLOOR_NORMALS);

	// materials are the game's own table, so they are declared here rather than
	// imported from anywhere. Registering by name is idempotent: a reload finds
	// the same handles and overwrites the same entries.
	let stone = world.materials.insert(
		"stone",
		Material::textured(tiles)
			.bumped(tile_normals)
			.finished(0.0, 0.75)
			.tiled(FLOOR_TILES),
	);
	// a polished dielectric rather than a metal: that is what a crystal is, and
	// a metal with nothing to reflect is a dark shape with one bright edge.
	let quartz = world
		.materials
		.insert("quartz", Material::colored(CENTER_COLOR).finished(0.0, 0.12));
	// the ring alternates between the two ends of the model, so both are on
	// screen at once and the difference is visible rather than described.
	let plastic = world
		.materials
		.insert("plastic", Material::DEFAULT.finished(0.0, 0.5));
	let metal = world
		.materials
		.insert("metal", Material::DEFAULT.finished(1.0, 0.25));

	world
		.entities
		.set_renderable(floor, Renderable::of(MeshId::QUAD, stone, FLOOR_COLOR));
	world
		.entities
		.set_renderable(center, Renderable::of(crystal, quartz, Vec3::ONE));

	for (index, id) in ring.into_iter().enumerate() {
		let material = if index % 2 == 0 { plastic } else { metal };

		world
			.entities
			.set_renderable(id, Renderable::of(MeshId::CUBE, material, hue(index)));
	}

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (props, player) = (state.props, state.player);

	world
		.entities
		.set_renderable(player, Renderable::of(MeshId::CUBE, plastic, PLAYER_COLOR));

	for (index, id) in props.into_iter().enumerate() {
		let ball = PROP_DROPS[index].2;
		let mesh = if ball { MeshId::SPHERE } else { MeshId::CUBE };
		// not `quartz`: that material carries the crystal's own color, and a
		// material's color multiplies the entity's tint rather than being
		// replaced by it, so a blue ball made of quartz comes out olive.
		let material = if ball { metal } else { plastic };

		world
			.entities
			.set_renderable(id, Renderable::of(mesh, material, prop_color(index)));
	}

	stand_model(world);
}

/// A color for the `index`th prop.
///
/// Deliberately not [`hue`]: the ring walks the whole circle and the props
/// should read as a separate set of things rather than as more of it.
fn prop_color(index: usize) -> Vec3 {
	let shade = turn(index, PROPS).mul_add(0.4, 0.55);

	if PROP_DROPS[index].2 {
		return Vec3::new(0.35, shade, 0.85);
	}

	Vec3::new(shade, 0.45, 0.35)
}

/// Puts every prop back where it is let go from, at rest.
///
/// The transform is written through the *body* as well as through the entity,
/// and the velocity is cleared: a prop that is teleported without having its
/// velocity taken away arrives at the top of its drop still traveling at
/// whatever speed it hit the floor with.
fn drop_props(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.landings = 0;
	let (props, bodies) = (state.props, state.prop_bodies);

	for (index, id) in props.into_iter().enumerate() {
		let (position, size, _) = PROP_DROPS[index];
		let mut transform = Transform::at(position);
		transform.set_scale(size);

		if let Some(slot) = world.entities.transform_mut(id) {
			*slot = transform;
		}

		world.entities.snap(id);

		if let Some(body) = world.bodies.get_mut(bodies[index]) {
			body.transform = transform;
			body.velocity = Vec3::ZERO;
			body.angular = Vec3::ZERO;
			body.sleeping = false;
		}
	}
}

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

/// Gives every entity of the scene a body to be traced against.
///
/// Rebuilt on every load rather than only on a fresh arena, and that is not
/// laziness: a collision mesh is baked when its body is created and does not
/// follow the `.obj` afterwards, so a body that outlived a reload would be
/// collided against the geometry the *previous* build compiled. Bodies are four
/// dozen bytes; correctness here is cheaper than cleverness.
///
/// Everything is [`BodyKind::Static`], so the bodies follow the entities the
/// game moves rather than the other way round. @ref
/// [`Body::transform`](colby_core::abi::Body::transform).
fn collide(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (center, ring, props) = (state.center, state.ring, state.props);
	let player = state.player;
	let old = [state.floor_body, state.center_body, state.trigger, state.player_body];

	for id in old
		.into_iter()
		.chain(state.ring_bodies)
		.chain(state.prop_bodies)
	{
		world.bodies.despawn(id);
	}

	// the middle collides against its own triangles, which is the one place
	// this scene exercises a mesh shape - and it is the shape a prop wants,
	// because a crystal's box is a poor answer to "did the cursor touch it".
	let crystal = world
		.entities
		.renderable(center)
		.map_or(CENTER_FALLBACK, |renderable| renderable.mesh);

	// hung under the quad rather than attached to it, @ref [`FLOOR_THICKNESS`].
	let mut slab = Transform::at(Vec3::new(0.0, FLOOR_Y - FLOOR_THICKNESS, 0.0));
	slab.scale = Vec3::new(FLOOR_SIZE, 1.0, FLOOR_SIZE);
	let floor_body = world.bodies.spawn(
		Body::new(BodyKind::Static, Shape::cuboid(Vec3::new(0.5, FLOOR_THICKNESS, 0.5)), slab)
			.layered(WORLD_LAYERS),
	);
	let center_body = world.attach_body(center, BodyKind::Static, Shape::mesh(crystal));
	layer(world, center_body, WORLD_LAYERS);

	// a volume that notices and pushes nothing. Static and driving no entity:
	// there is nothing to draw it as and nothing to move it, and the whole of
	// what it is for is the list of what is inside it.
	let trigger = world.bodies.spawn(
		Body::new(
			BodyKind::Static,
			Shape::cuboid(TRIGGER_EXTENTS),
			Transform::at(TRIGGER_CENTER),
		)
		.sensing()
		.layered(TRIGGER_LAYERS),
	);

	let mut ring_bodies = [BodyId::NONE; RING];
	for (slot, id) in ring_bodies.iter_mut().zip(ring) {
		*slot = world.attach_body(id, BodyKind::Static, Shape::UNIT);
		layer(world, *slot, WORLD_LAYERS);
	}

	// the props are the only bodies in the scene the *solver* owns. Everything
	// above follows an entity the game animates; these five do the opposite,
	// and the entity follows them.
	let mut prop_bodies = [BodyId::NONE; PROPS];
	for (index, slot) in prop_bodies.iter_mut().enumerate() {
		let shape = if PROP_DROPS[index].2 {
			Shape::ball(0.5)
		} else {
			Shape::UNIT
		};

		*slot = world.attach_body(props[index], BodyKind::Dynamic, shape);

		if let Some(body) = world.bodies.get_mut(*slot) {
			body.mass = PROP_MASS;
			body.restitution = PROP_BOUNCE;
			body.friction = PROP_GRIP;
			body.layers = PROP_LAYERS;
		}
	}

	// one of the balls hangs rather than lies. A rope is the cheapest thing that
	// shows a joint working: it is obvious from across the room whether it is
	// holding, and obvious the moment it is not.
	world.joints.clear();
	let rope =
		world.join(Joint::rope(prop_bodies[SWINGING], BodyId::NONE, (Vec3::ZERO, HOOK), ROPE));

	// kinematic rather than dynamic: the controller decides where the player
	// goes, and the solver's job is only to push props out from under it. A
	// dynamic one would be a box the player argues with.
	let player_body =
		world.attach_body(player, BodyKind::Kinematic, Shape::cuboid(Vec3::splat(0.5)));
	layer(world, player_body, PLAYER_LAYERS);

	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	state.player_body = player_body;
	state.floor_body = floor_body;
	state.center_body = center_body;
	state.ring_bodies = ring_bodies;
	state.prop_bodies = prop_bodies;
	state.rope = rope;
	state.trigger = trigger;
	// what the ray found is per-load and not worth carrying across one: the
	// bodies it named are the ones just despawned. Clearing it also means the
	// first trace of every load says so in the log, which is what makes a live
	// reload readable.
	state.picked = EntityId::NONE;

	info!(bodies = world.bodies.len(), "scene given collision");
}

/// Puts every entity of the scene where this frame says it should be.
///
/// A handle that no longer resolves is skipped rather than treated as an error:
/// the entity table belongs to the host, and something else is allowed to have
/// removed one.
fn place(world: &mut World) {
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let (spin, floor, center, ring) = (state.spin, state.floor, state.center, state.ring);

	if let Some(transform) = world.entities.transform_mut(floor) {
		*transform = Transform {
			position: Vec3::new(0.0, FLOOR_Y, 0.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(FLOOR_SIZE, 1.0, FLOOR_SIZE),
		};
	}

	// stood on the floor by its own bounding box rather than by a number typed
	// here. Make the crystal taller in assets/meshes/crystal.obj and it rises to
	// match instead of sinking through the floor - which is the difference
	// between an asset the engine knows the shape of and one it merely draws.
	let rest = world
		.entities
		.renderable(center)
		.map(|renderable| renderable.mesh)
		.and_then(|mesh| world.meshes.get(mesh))
		.map_or(0.0, |mesh| -mesh.value().bounds().0.y * CENTER_SCALE);

	if let Some(transform) = world.entities.transform_mut(center) {
		transform.position = Vec3::new(0.0, FLOOR_Y + rest, 0.0);
		transform.rotation = Quat::from_rotation_y(-spin * 1.5);
		transform.set_scale(CENTER_SCALE);
	}

	// the rope, drawn between its hook and whatever it is holding. Read from the
	// *body* rather than the entity so that it is exactly where the solver put
	// it rather than a step behind. This used to be a very thin cube with an
	// entity of its own, because there was nothing else to draw a line with.
	let (state, _) = world.state.get::<State>(STATE_LAYOUT);
	let held = state.prop_bodies[SWINGING];
	let end = world
		.bodies
		.get(held)
		.map_or(HOOK, |body| body.transform.position);

	world.debug.line(HOOK, end, debug::YELLOW);
	world.debug.point(HOOK, HOOK_SIZE, debug::YELLOW);

	for (index, id) in ring.into_iter().enumerate() {
		let Some(transform) = world.entities.transform_mut(id) else {
			continue;
		};

		let angle = TAU.mul_add(turn(index, RING), spin);
		let bob = spin.mul_add(BOB.1, angle).sin() * BOB.0;

		transform.position = Vec3::new(RING_RADIUS * angle.cos(), bob, RING_RADIUS * angle.sin());
		transform.rotation = Quat::from_rotation_y(angle) * Quat::from_rotation_x(spin);
		transform.set_scale(0.6);
	}
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
/// @param state - the arena, for the handles to compare against
fn named(state: &State) -> String {
	if !state.picked_body.is_some() {
		return "nothing".to_owned();
	}

	let distance = state.picked_distance;

	// by body rather than by entity, because the floor has no entity: its slab
	// hangs below the quad and so cannot follow one. @ref [`FLOOR_THICKNESS`].
	if state.picked_body == state.floor_body {
		return format!("floor @ {distance:.1}");
	}

	if state.picked == state.center {
		return format!("crystal @ {distance:.1}");
	}

	if state.picked == state.player {
		return format!("player @ {distance:.1}");
	}

	if let Some(index) = state
		.ring
		.iter()
		.position(|&id| id == state.picked)
	{
		return format!("cube {index} @ {distance:.1}");
	}

	state
		.props
		.iter()
		.position(|&id| id == state.picked)
		.map_or_else(
			|| format!("something @ {distance:.1}"),
			|index| format!("prop {index} @ {distance:.1}"),
		)
}

/// A color for the `index`th cube of the ring, walked around the hue circle.
fn hue(index: usize) -> Vec3 {
	let angle = TAU * turn(index, RING);
	let channel = |offset: f32| (angle + offset).sin().mul_add(0.4, 0.55);

	Vec3::new(channel(0.0), channel(TAU / 3.0), channel(TAU * 2.0 / 3.0))
}

/// `index` out of `count`, as a fraction of a full turn.
fn turn(index: usize, count: usize) -> f32 {
	let index = u16::try_from(index).unwrap_or(0);
	let count = u16::try_from(count).unwrap_or(1).max(1);

	f32::from(index) / f32::from(count)
}
