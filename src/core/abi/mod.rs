//! The host/game boundary.
//!
//! The game module exports one C symbol returning a [`GameApi`] - a plain
//! `#[repr(C)]` struct of function pointers - and the host calls through it.
//! There is deliberately no symbol linkage between host and module: the unix
//! answer is `RTLD_GLOBAL`, and that has no Windows equivalent.
//!
//! The other half of the contract is that **the module holds no state**. Every
//! byte the game reads or writes lives in [`World`], which the host owns and
//! keeps across reloads. A swap is then a function-pointer replacement with
//! nothing to migrate.
//!
//! @note: `World` itself is not `#[repr(C)]`, and does not need to be. The
//! boundary that has to have a fixed layout is the *call* - `GameApi` and the
//! plain-data types reachable from `World` - and those all do. `World` is host
//! Rust data reached through one pointer, and there is exactly one definition
//! of it in the process because host and module share `colby_core.dll`. The
//! runner refuses to start if that is not true, @ref
//! [`mods::linkage`](crate::mods::linkage), and [`ABI_VERSION`] catches a
//! module built against a different definition.

use crate::glam::{Mat4, Quat, Vec3};

pub mod anim;
pub mod audio;
pub mod camera;
pub mod character;
pub mod console;
pub mod cvar;
pub mod debug;
pub mod entity;
pub mod font;
pub mod input;
pub mod joint;
pub mod material;
pub mod mesh;
pub mod model;
pub mod names;
pub mod net;
pub mod physics;
pub mod pose;
pub mod ragdoll;
pub mod registry;
pub mod scene;
pub mod script;
pub mod skeleton;
pub mod state;
pub mod texture;
pub mod ui;

pub use self::{
	anim::{
		Channel, Clip, ClipData, ClipId, Clips, Interpolation, MAX_KEYS, MAX_NODES, MAX_TRACKS,
		NO_BONE, Node, Track, Tree,
	},
	audio::{
		Category, Listener, MAX_VOICES, Mix, Sound, SoundData, SoundId, Sounds, Voice, VoiceId,
		Voices,
	},
	camera::Camera,
	character::{DECAY, Drift, MAX_CATCH_UP, Motion, Moved, replay},
	cvar::{Args, ConsoleFn, Cvars, Value},
	debug::{Debug, Label, Line, Pen},
	entity::{Entities, EntityId, MAX_ENTITIES, Renderable, Transform},
	font::{Font, FontData, FontId, Fonts, Glyph},
	input::{Button, Input, Key},
	joint::{Joint, JointId, JointKind, Joints, MAX_JOINTS},
	material::{Material, MaterialId, Materials},
	mesh::{BONES_PER_VERTEX, Mesh, MeshData, MeshId, MeshVertex, Meshes, SkinVertex},
	model::{Model, ModelData, ModelId, Models, Placement},
	names::MAX_NAME,
	net::{Command, Commands, PeerId, Role},
	physics::{
		Bodies, Body, BodyId, BodyKind, Layers, MAX_BODIES, MAX_OVERLAPS, MAX_TOUCHES, Overlap,
		Physics, Shape, ShapeKind, Touch, TouchKind, TraceFn, TraceInfo, TraceResult,
	},
	pose::{MAX_POSES, Pose, PoseId, Poses},
	ragdoll::{Build, MAX_PARTS, NO_PART, Part, Ragdoll, Segment},
	registry::{Entry, Registry},
	scene::{
		Arena, Form, Grafted, Link, Posed, Remap, Restored, Scene, SceneData, SceneId, Scenes,
		Solid, Stage, Thing,
	},
	script::{Script, ScriptData, ScriptId, Scripts, WORLD_PREFIX},
	skeleton::{
		Bone, MAX_BONES, NO_PARENT, Skeleton, SkeletonData, SkeletonId, Skeletons, rests,
	},
	state::{GameState, Players},
	texture::{Texel, Texture, TextureData, TextureId, Textures},
	ui::{DocumentData, DocumentId, Event, EventKind, Length, PanelId, Ui},
};

/// The revision of the types and signatures in this module.
///
/// The host refuses a module reporting a different value. Bump it whenever a
/// signature or a layout below changes; forgetting to is a crash rather than an
/// error message.
pub const ABI_VERSION: u32 = 50;

/// The C symbol every game module exports, NUL-terminated for `GetProcAddress`.
pub const GAME_API_SYMBOL: &[u8] = b"colby_game_api\0";

/// Every entry point in [`GameApi`] has this signature.
///
/// The single argument is the host's [`World`]. Nothing else crosses, because
/// nothing else needs to: the module reads its inputs and writes its outputs
/// through that one pointer.
///
/// # Safety
///
/// `world` must point to a live, initialized [`World`] that no one else is
/// touching for the duration of the call.
pub type GameFn = unsafe extern "C-unwind" fn(world: *mut World);

/// The function-pointer table a game module exports.
///
/// `C-unwind` rather than `C` is deliberate: a panic inside gameplay code
/// should reach the host's `catch_unwind` and be reported, not abort the
/// process. That only works because host and module share one panic runtime,
/// which is what `-Cprefer-dynamic` buys.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GameApi {
	/// The [`ABI_VERSION`] the module was compiled against.
	pub abi_version: u32,

	/// Called once each time the module is swapped in, before the first
	/// `update`.
	pub init: GameFn,

	/// Called once per simulation step, at a constant rate whatever the frame
	/// rate is doing. @ref [`World::dt`].
	pub update: GameFn,

	/// Called once before the module is swapped out or the host exits.
	pub shutdown: GameFn,
}

/// The signature of the exported [`GAME_API_SYMBOL`].
///
/// # Safety
///
/// Only valid when resolved from a module built against this `ABI_VERSION`.
pub type GameApiEntry = unsafe extern "C" fn() -> GameApi;

/// All the state the game can see, owned by the host.
///
/// Host-written fields are inputs to the game; game-written fields are what the
/// engine reads back out. Because this lives in the host it survives a module
/// swap untouched, which is the whole reason the game can be replaced mid-frame
/// without migrating anything.
///
/// Big enough that the host keeps it behind a `Box`.
pub struct World {
	/// Seconds the simulation has run for. Host-written.
	///
	/// Simulated seconds, not wall-clock ones: it advances by exactly
	/// [`dt`](Self::dt) per step and stops advancing while the process is
	/// stalled, so after a stall it is permanently behind the wall clock by
	/// whatever the host dropped. That is the point - it is the clock the
	/// simulation is on, and it is the same for a given number of steps on
	/// every machine.
	pub time: f32,

	/// How long one simulation step is, in seconds. Host-written.
	///
	/// A constant, and the reason gameplay can integrate against it without
	/// caring what the frame rate is doing. @ref
	/// [`STEP_SECONDS`](crate::time::STEP_SECONDS).
	pub dt: f32,

	/// Simulation steps since the host started. Host-written.
	///
	/// Steps, not rendered frames: at 144 Hz there are more frames than these,
	/// and during a stall there are fewer.
	pub steps: u64,

	/// How many times the game module has been swapped in. Host-written, and
	/// the game's cue that it is looking at state an older build left behind.
	pub reloads: u32,

	/// Whether the world is being edited rather than played. Host-written.
	///
	/// While this is set the host does not call the game's `update` at all and
	/// does not step the solver, so nothing here moves except what a person
	/// moves. It is exposed anyway, for two reasons. A game that draws
	/// something from outside `update` - a console command, a script handler -
	/// can tell which it is. And every engine with an editor publishes the
	/// same fact under some name, because gameplay code that runs at authoring
	/// time eventually needs to know that it is.
	///
	/// The step still runs: input, the interface, its scripts and the debug
	/// table's sweep all happen, so time goes on and what is drawn over the
	/// world stays fresh.
	pub editing: bool,

	/// Which endpoint of a conversation this process is.
	///
	/// [`PeerId::HOST`] from [`World::new`](Self::new) onwards, which is not a
	/// placeholder: a process playing on its own simulates its own world, so it
	/// really is its own authority and every body in it reads back as
	/// [`Role::Authority`]. A window that has gone looking for somebody else's
	/// world says so once, and is then **nobody until the host names it** -
	/// the runner writes what the host said, refusing any name at the host's
	/// own slot. @ref `crate::net::joined` and `Net::seat` in the runner.
	///
	/// The value a client is meant to hold before it has been told who it is
	/// is [`PeerId::NONE`], which owns nothing and decides nothing. That is
	/// the deliberate part: the unknown answer is the powerless one, and the
	/// default is not an unknown but a statement.
	///
	/// What a game does with this is ask a body what part it plays:
	/// `body.role(world.peer)`. @ref [`net`](crate::abi::net).
	///
	/// Not written down by a save. Which endpoint a process is, is a fact
	/// about the process rather than about the world - a world captured on a
	/// host and restored on a client would otherwise claim to be the
	/// authority. Same reason [`editing`](Self::editing) is not written down.
	pub peer: PeerId,

	/// The window's width divided by its height. Host-written; the renderer
	/// uses it so that a shape does not stretch with the window.
	pub aspect: f32,

	/// Keyboard and mouse for this step. Host-written.
	///
	/// **This machine's own, and it crosses to nobody.** The surface size, the
	/// cursor in pixels, what was typed - none of it means anything on another
	/// machine, and a window that is not this one has its own. What does cross
	/// is the small part of it a host has to be told, and that is a
	/// [`Command`] rather than this. @ref [`commands`](Self::commands).
	pub input: Input,

	/// What every peer recently asked for, one ring each.
	///
	/// Beside [`input`](Self::input) and the other end of the same wire a
	/// snapshot travels: what is written here is what a person wanted, and
	/// what a snapshot carries is what the machine that simulates decided
	/// about it. A host reads everybody's ring and runs them; a client keeps
	/// its own, sends the last few in every datagram, and re-runs whatever the
	/// host has not confirmed.
	///
	/// The game fills its own end of it - one command a step, from whatever is
	/// under that machine's hands - and the runner carries what is unsettled in
	/// every message and files what arrives under the peer it came from.
	///
	/// Not written down by a save, for the reason an owner is not: what
	/// somebody was asking for a moment ago is a moment rather than a world.
	pub commands: Commands,

	/// Where the renderer looks from. Game-written.
	pub camera: Camera,

	/// The window clear color, linear RGB. Game-written.
	pub clear: Vec3,

	/// The direction the light travels, in world space. Game-written.
	///
	/// Not normalized on the way in; the shader does that.
	pub light: Vec3,

	/// How lit a surface facing away from the light still is. Game-written.
	pub ambient: Vec3,

	/// What every dynamic body accelerates by, in units a second squared.
	///
	/// Game-written, like `light` and `ambient` and unlike everything else the
	/// solver reads: it is a property of the world rather than of a body, so it
	/// sits here rather than needing a call. The host's `phys.gravity` console
	/// variable writes it too.
	pub gravity: Vec3,

	/// Whether something has asked the process to stop. Game-written, or
	/// written by the `quit` command; the host reads it once a frame.
	pub quit: bool,

	/// How many pairs of bodies were touching at the end of the last step.
	///
	/// Host-written, like `steps`, and for the same kind of reason: it is the
	/// engine reporting on itself so that a panel can be a view onto something
	/// that exists rather than a reason to grow a mechanism.
	pub contacts: u32,

	/// Simulation steps the host owes over and above real time.
	///
	/// The console's single-step button: `sim.step 4` while paused puts four
	/// here, and the host runs them and clears it. Nothing else should write
	/// it, because time that did not pass is a thing to ask for deliberately.
	pub owed_steps: u32,

	/// Every entity in the world. Host-owned storage, reached by handle.
	pub entities: Entities,

	/// Every joint holding two bodies together, reached by handle.
	///
	/// Host-owned plain data beside the bodies, and for the same reasons. @ref
	/// [`joint`](crate::abi::joint).
	pub joints: Joints,

	/// Every physics body, reached by handle.
	///
	/// Host-owned plain data, like the entities and unlike a resource
	/// registry: a body is created and destroyed, so its handle is
	/// generational. The solver reads and writes this table and owns none of
	/// it, which is what keeps the authoritative transform somewhere the
	/// editor can see and a saved scene can reach. @ref
	/// [`physics`](crate::abi::physics).
	pub bodies: Bodies,

	/// Every mesh the renderer can draw, reached by handle.
	///
	/// Host-owned: the runner compiles what is under `assets/` and registers
	/// each result here by name, so `meshes.find("meshes/crystal")` is how a
	/// game gets hold of geometry it did not generate itself. Registering is
	/// open to the game too, for geometry it builds at runtime.
	pub meshes: Meshes,

	/// Every texture the renderer can sample, reached by handle.
	///
	/// Filled the same way as `meshes`, from the same tree.
	pub textures: Textures,

	/// Every font the interface can draw with, reached by handle.
	///
	/// Filled the same way as `meshes` and from the same tree: the compiler
	/// bakes a `.ttf` into metrics and a distance field, and nothing at runtime
	/// links a font library. @ref [`font`](crate::abi::font).
	pub fonts: Fonts,

	/// Every sound the mixer can play, reached by handle.
	///
	/// Filled the same way as `meshes` and from the same tree: the compiler
	/// turns whatever a recorder wrote into interleaved samples, and nothing
	/// at runtime links an audio library to open a file. @ref
	/// [`audio`](crate::abi::audio).
	pub sounds: Sounds,

	/// Every sound being played, reached by handle.
	///
	/// Host-owned plain data like the bodies, and advanced by the step rather
	/// than by whatever is turning it into samples: a voice ends after the
	/// same number of steps on every machine. @ref
	/// [`audio`](crate::abi::audio).
	pub audio: Voices,

	/// Where the world is heard from. Game-written, like the camera.
	///
	/// Deliberately not derived from the camera: a game whose camera is an
	/// orbit control would otherwise hear the world from wherever somebody was
	/// looking. @ref [`Listener::at_camera`] for the one line that makes it
	/// follow.
	pub listener: Listener,

	/// How loud each category of sound is. Game-written, and written by the
	/// host's console variables too - the same arrangement `gravity` has.
	pub mix: Mix,

	/// Every scene the host has loaded, reached by handle.
	///
	/// Filled the same way as `meshes`, from the same tree: an
	/// `assets/scenes/props.scene` compiles into a `.cscene` and lands here
	/// under `scenes/props`. What a game does with one is hand it to a loader
	/// - @ref [`scene`](crate::abi::scene) for the two of them.
	pub scenes: Scenes,

	/// Every model the host has loaded, reached by handle.
	///
	/// Filled the same way as `meshes`, from the same tree, and holding none
	/// of the geometry: a model is a list of what stands where, and what
	/// stands there is a mesh in the table above this one. @ref
	/// [`model`](crate::abi::model).
	pub models: Models,

	/// Every animation clip the host has loaded, reached by handle.
	///
	/// Filled the same way as `meshes`, from the same tree, and holding no
	/// skeleton: a clip names the bones it moves with text, so the same walk
	/// plays on every rig whose bones answer to the same names. What turns
	/// that text into indices is a table this registry keeps beside its
	/// entries. @ref [`anim`](crate::abi::anim).
	pub clips: Clips,

	/// Every skeleton the host has loaded, reached by handle.
	///
	/// Filled the same way as `meshes`, from the same tree, and holding no
	/// geometry either: a skeleton is the bones a skinned mesh is moved by,
	/// and the mesh naming them is in the table above this one. @ref
	/// [`skeleton`](crate::abi::skeleton).
	pub skeletons: Skeletons,

	/// Where every posed skeleton's bones are, reached by handle.
	///
	/// Not an asset: a skeleton is read from a file and shared, a pose is one
	/// character's own state and is written every step. A
	/// [`Renderable`](Self::entities) names one, and two entities may name the
	/// same - a model of two materials is two entities moved by one set of
	/// bones. @ref [`pose`](crate::abi::pose).
	pub poses: Poses,

	/// Every material, reached by handle.
	///
	/// Unlike the other two this is mostly the *game's* table: a material is a
	/// handful of numbers and a texture handle, so a game builds them inline
	/// rather than importing them from anywhere.
	pub materials: Materials,

	/// Every console variable and command in the process.
	///
	/// Host-owned like the other tables, and open to the game the same way
	/// materials are: registering a variable from `init` is how gameplay gets a
	/// number someone can turn without a rebuild. @ref
	/// [`cvar`](crate::abi::cvar) for what a reload does to them.
	pub cvars: Cvars,

	/// Every interface document, and what the game has put on screen.
	///
	/// Filled from the asset tree like the meshes are, and written by the game
	/// the way the material table is. @ref [`ui`](crate::abi::ui).
	pub ui: Ui,

	/// Every program the host has been handed, reached by handle.
	///
	/// Filled from the asset tree like the meshes are: an `assets/ui/hud.lua`
	/// compiles into a `.clua` and lands here under `ui/hud`, and a document
	/// naming it carries that name rather than the text. Nothing in
	/// `colby_core` runs one - the interpreter is the host's, which is what
	/// keeps a program alive across a reload of the game module. @ref
	/// [`script`](crate::abi::script).
	pub scripts: Scripts,

	/// Lines, shapes and words to draw over the world.
	///
	/// Host-owned plain data anyone with the world may write: the solver
	/// outlines its own bodies, gameplay marks up whatever it is working on.
	/// Swept at the top of each step rather than the bottom, because several
	/// frames are drawn between two steps. @ref [`debug`](crate::abi::debug).
	pub debug: Debug,

	/// The world's own state, which everybody in it shares.
	///
	/// The host allocates it and never looks inside. What goes here is what
	/// the world *is*: the map, the props, the score. @ref
	/// [`state`](crate::abi::state) for the three arenas and the rule for
	/// deciding which one a value belongs in.
	pub state: GameState,

	/// One block of state per peer, and the peers themselves.
	///
	/// What is true of one person rather than of the world: what they are
	/// holding, which tool they picked. A peer reads its own through
	/// [`Players::get`](crate::abi::state::Players::get), and the host reads
	/// everybody's, because the host is the only thing that simulates.
	///
	/// Also the table that mints a [`PeerId`]: slot zero is
	/// [`PeerId::HOST`]'s from the moment the world exists, and
	/// [`admit`](crate::abi::state::Players::admit) hands out the rest.
	pub players: Players,

	/// The state of this screen, which crosses to nobody and is never saved.
	///
	/// A camera, a panel handle, the voice something is humming with. The one
	/// arena a snapshot must never overwrite, because two people looking at
	/// one world are still looking from two places - and the one a save leaves
	/// alone, because where somebody was looking is not part of the world.
	pub local: GameState,

	/// The two queries the host answers about [`bodies`](Self::bodies).
	///
	/// Private, like the interpolation fields and for a stronger reason: it is
	/// the one thing on `World` the game must never write, because the
	/// pointers in it are the host's. Reached through
	/// [`trace_ray`](Self::trace_ray) and [`trace_box`](Self::trace_box), and
	/// installed once by [`install_physics`](Self::install_physics).
	physics: Physics,

	/// The camera as it stood at the end of the previous step.
	///
	/// Private, unlike everything above: it is neither an input to the game nor
	/// an output from it, only the other end of the blend the renderer asks for
	/// through [`render_camera`](Self::render_camera).
	camera_previous: Camera,

	/// Whether the game has said the camera cut rather than moved.
	camera_snap: bool,

	/// How far the frame being drawn sits past the last simulated state.
	interpolation: f32,

	/// One pose per node, while a blend tree is being worked out.
	///
	/// Private and kept between calls for the reason every other scratch in
	/// this engine is: a game animating a crowd should allocate once rather
	/// than once a character a step. Nothing outside
	/// [`animate`](Self::animate) ever looks at it, and what it holds between
	/// two calls means nothing.
	blending: Vec<Transform>,

	/// One matrix per bone, while a ragdoll is being read either way.
	///
	/// The same arrangement [`blending`](Self::blending) is, for the same
	/// reason: a game's own memory is plain bytes and cannot hold a `Vec`, so
	/// the scratch a ragdoll needs lives here rather than being asked for.
	rigging: Vec<Mat4>,
}

impl World {
	/// A world with nothing in it.
	#[must_use]
	pub fn new() -> Self {
		Self {
			time: 0.0,
			dt: crate::time::STEP_SECONDS,
			steps: 0,
			reloads: 0,
			aspect: 1.0,
			editing: false,
			// its own authority, until somebody says otherwise. @ref the
			// field, and note this is deliberately not the zero value.
			peer: PeerId::HOST,
			input: Input::default(),
			commands: Commands::new(),
			camera: Camera::DEFAULT,
			clear: Vec3::ZERO,
			light: Vec3::new(-0.4, -1.0, -0.3),
			ambient: Vec3::splat(0.25),
			gravity: Vec3::new(0.0, -9.81, 0.0),
			quit: false,
			owed_steps: 0,
			contacts: 0,
			entities: Entities::new(),
			bodies: Bodies::new(),
			joints: Joints::new(),
			meshes: Meshes::new(),
			textures: Textures::new(),
			fonts: Fonts::new(),
			sounds: Sounds::new(),
			audio: Voices::new(),
			listener: Listener::DEFAULT,
			mix: Mix::FULL,
			models: Models::new(),
			clips: Clips::new(),
			skeletons: Skeletons::new(),
			poses: Poses::new(),
			scenes: Scenes::new(),
			materials: Materials::new(),
			cvars: Cvars::new(),
			ui: Ui::new(),
			scripts: Scripts::new(),
			debug: Debug::new(),
			state: GameState::new(),
			players: Players::new(),
			local: GameState::new(),
			physics: Physics::STUB,
			blending: Vec::new(),
			rigging: Vec::new(),
			camera_previous: Camera::DEFAULT,
			camera_snap: false,
			// one, so that a world nobody paces - a test, a screenshot - draws
			// the state it was handed rather than a blend of it with whatever
			// came before. The host overwrites this every frame.
			interpolation: 1.0,
		}
	}

	/// Moves the present into the past, ready for another step.
	///
	/// The host calls this before every simulation step, and once more after a
	/// game module is swapped in - a reload is a discontinuity whatever the
	/// game does about it, so nothing is drawn arriving from the pose the
	/// previous build left behind.
	pub fn advance(&mut self) {
		self.entities.advance();
		self.poses.advance();
		self.camera_previous = self.camera;
		self.camera_snap = false;
	}

	/// Applies everything the step just finished asked not to be interpolated.
	///
	/// The host calls this after the game's `update`. @ref
	/// [`Entities::settle`].
	pub fn settle(&mut self) {
		self.entities.settle();
		self.poses.settle();

		// guarded, unlike `advance`. Settling every step unconditionally would
		// leave `camera_previous` equal to `camera` at every render, which
		// looks like nothing at all and is in fact the camera never being
		// interpolated again.
		if self.camera_snap {
			self.camera_previous = self.camera;
			self.camera_snap = false;
		}
	}

	/// Declares that the camera cut rather than moved.
	///
	/// The camera is interpolated like everything else, so a game that jumps
	/// it, on a reset or a cut to another view, has to say so, or the whole
	/// picture slides into the new pose over the next frame. Takes effect at
	/// the end of the step, so it can be called before or after the jump it
	/// describes.
	pub fn snap_camera(&mut self) { self.camera_snap = true; }

	/// Sets how far the frame being drawn sits past the last simulated state.
	///
	/// The host's to call, once per frame, before it renders.
	///
	/// @param t - the fraction of a step, clamped into `0.0 ..= 1.0`
	pub fn set_interpolation(&mut self, t: f32) { self.interpolation = t.clamp(0.0, 1.0); }

	/// How far the frame being drawn sits past the last simulated state.
	#[must_use]
	pub const fn interpolation(&self) -> f32 { self.interpolation }

	/// The camera this frame should be drawn through.
	#[must_use]
	pub fn render_camera(&self) -> Camera {
		self.camera_previous
			.lerp(self.camera, self.interpolation)
	}

	/// Where an entity should be drawn this frame.
	///
	/// @param id - the entity to place
	/// @return the blended transform, or `None` if the handle is stale
	#[must_use]
	pub fn render_transform(&self, id: EntityId) -> Option<Transform> {
		self.entities.interpolated(id, self.interpolation)
	}

	/// The matrices a pose hands the vertex stage this frame, appended.
	///
	/// The same idea [`render_transform`](Self::render_transform) is, for the
	/// same reason: bones move once a step and are drawn many times between
	/// two, so what a frame wants is somewhere between the pose's past and its
	/// present. The skeleton is looked up here rather than by the caller,
	/// because a pose already names it and two places knowing the pairing is
	/// one too many.
	///
	/// @param id - the pose to resolve
	/// @param out - the buffer to append to
	/// @return how many matrices were appended
	pub fn render_skinning(&self, id: PoseId, out: &mut Vec<Mat4>) -> usize {
		let Some(pose) = self.poses.get(id) else {
			return 0;
		};

		self.poses
			.skinning(id, self.skeletons.bones(pose.skeleton), self.interpolation, out)
	}

	/// Plays one clip into a pose, over the skeleton the pose names.
	///
	/// The whole of what animating a character costs a game while there is one
	/// clip to play: a handle, a moment on the game's own clock, and whether
	/// it starts again at the end. Where the moment comes from is the game's -
	/// a phase it advances by `dt` and keeps in its own memory - because that
	/// is the one part of this that is gameplay.
	///
	/// Three things happen, in this order, and the order is the whole of the
	/// contract. The clip's tracks are matched to the skeleton's bones by
	/// name, once, and kept - @ref [`Clips::bind`]. Every bone is put back
	/// where the skeleton rests it, so a bone the clip says nothing about
	/// stands rather than keeping last step's attitude. Then the clip is
	/// written over that.
	///
	/// Resting first is what makes a played pose *complete*, and a complete
	/// pose is what makes it possible to blend two of them later. The cost is
	/// that this cannot be called twice to lay one clip over another; laying
	/// one over another is a blend, and a blend is not two writes.
	///
	/// @param pose - the pose to write
	/// @param clip - what to play into it
	/// @param time - seconds on the game's own clock
	/// @param looping - whether the clip starts again rather than holding its
	/// last key
	/// @return `false` if the pose handle is stale, in which case nothing was
	/// written
	pub fn play(&mut self, pose: PoseId, clip: ClipId, time: f32, looping: bool) -> bool {
		let Some(skeleton) = self.poses.get(pose).map(|posed| posed.skeleton) else {
			return false;
		};
		let Self { poses, clips, skeletons, .. } = self;

		clips.bind(clip, skeleton, skeletons);

		let Some(posed) = poses.get_mut(pose) else {
			return false;
		};

		posed.rest(skeletons.bones(skeleton));
		clips
			.data(clip)
			.sample(time, looping, clips.bones(clip, skeleton), &mut posed.locals);

		true
	}

	/// Works a blend tree out into a pose.
	///
	/// What [`play`](Self::play) is for one clip, for as many as a game cares
	/// to mix: it binds every clip the tree names to the pose's skeleton, then
	/// works the tree out in one forward pass. The scratch it needs lives here
	/// rather than being asked for, because a game's own memory is plain bytes
	/// and cannot hold one.
	///
	/// Every leaf starts from the skeleton at rest, exactly as `play` does, so
	/// every pose inside the tree is a whole one and blending two of them
	/// means what it looks like it means.
	///
	/// @param pose - the pose to write
	/// @param tree - the blend to work out
	/// @return `false` if the pose handle is stale or the tree could not be
	/// worked out; in the second case the pose is left at rest
	pub fn animate(&mut self, pose: PoseId, tree: &Tree) -> bool {
		let Some(skeleton) = self.poses.get(pose).map(|posed| posed.skeleton) else {
			return false;
		};

		for node in &tree.nodes {
			if let Node::Clip { clip, .. } = *node {
				self.clips.bind(clip, skeleton, &self.skeletons);
			}
		}

		let Self { poses, clips, skeletons, blending, .. } = self;
		let Some(posed) = poses.get_mut(pose) else {
			return false;
		};

		anim::evaluate(
			tree,
			clips,
			skeleton,
			skeletons.bones(skeleton),
			blending,
			&mut posed.locals,
		)
	}

	/// Puts a ragdoll's bodies where the pose says its bones are.
	///
	/// The direction that runs while a character is on its feet: an animation
	/// writes the pose, this carries the bodies along with it, and the bodies
	/// are [`BodyKind::Kinematic`] so that the solver leaves them alone and
	/// they still shove whatever they walk into. It is also what makes
	/// switching a ragdoll on cost nothing: the bodies are already exactly
	/// where the character is, so there is no seam to hide and nothing to
	/// snap.
	///
	/// A part whose body handle does not resolve is skipped, which is what a
	/// plan that has not been spawned yet looks like.
	///
	/// @param pose - the pose to read
	/// @param at - where the character stands, the transform its entities wear
	/// @param ragdoll - the layout, its bodies filled in
	/// @return `false` if the pose handle is stale, in which case nothing moved
	pub fn pull_ragdoll(&mut self, pose: PoseId, at: Transform, ragdoll: &Ragdoll) -> bool {
		let Self { poses, skeletons, bodies, rigging, .. } = self;
		let Some(bones) = poses
			.get(pose)
			.map(|posed| skeletons.bones(posed.skeleton))
		else {
			return false;
		};

		rigging.clear();

		// the present rather than anything interpolated: this is inside the
		// step, and what the frames between two steps are drawn at is the
		// renderer's business and nobody else's.
		poses.model(pose, bones, 1.0, rigging);

		let stance = at.matrix();

		for part in &ragdoll.parts {
			let (Some(model), Some(body)) =
				(rigging.get(usize::from(part.bone)), bodies.get_mut(part.body))
			else {
				continue;
			};

			body.transform = Transform::from_matrix(stance * *model * part.in_bone.matrix());
		}

		true
	}

	/// Writes a pose from where a ragdoll's bodies have ended up.
	///
	/// The other direction, and the one that makes a character fall over. Two
	/// passes over the bones and no recursion, which the skeleton's own
	/// ordering buys:
	///
	/// - the first works out where every bone is. A bone a part follows takes
	///   its place from that part's body; every other bone is carried by its
	///   parent, exactly as it was. That second half is what keeps a neck
	///   between a chest and a head, and what lets a hand ride on the forearm
	///   that has a body while having none itself.
	/// - the second writes back the local transform of the bones that have
	///   bodies, each against wherever its parent ended up.
	///
	/// **The character's own transform is not written and does not move.** A
	/// ragdoll that rolls away leaves the entity where it stood and carries
	/// everything in the bones, which is what the two engines that do this
	/// both do. The cost is named in the notes: after a fall, an entity's
	/// position no longer says where the character is.
	///
	/// Switching *off* is the discontinuity, not switching on: the bones jump
	/// from wherever they fell to wherever the animation puts them, so a game
	/// that stops a ragdoll owes [`Poses::snap`].
	///
	/// @param pose - the pose to write
	/// @param at - where the character stands, the transform its entities wear
	/// @param ragdoll - the layout, its bodies filled in
	/// @return `false` if the pose handle is stale, in which case nothing was
	/// written
	pub fn push_ragdoll(&mut self, pose: PoseId, at: Transform, ragdoll: &Ragdoll) -> bool {
		let Self { poses, skeletons, bodies, rigging, .. } = self;
		let Some(bones) = poses
			.get(pose)
			.map(|posed| skeletons.bones(posed.skeleton))
		else {
			return false;
		};
		let Some(posed) = poses.get(pose) else {
			return false;
		};

		let stance = at.matrix().inverse();

		rigging.clear();
		rigging.reserve(bones.len());

		for (index, bone) in bones.iter().enumerate() {
			let carried = u16::try_from(index)
				.ok()
				.and_then(|it| ragdoll.part_of(it))
				.and_then(|part| ragdoll.parts.get(usize::from(part)))
				.and_then(|part| Some((part, bodies.get(part.body)?)));

			let model = match carried {
				| Some((part, body)) =>
					stance * body.transform.matrix() * part.in_bone.matrix().inverse(),
				// carried by whatever is above it, exactly as the pose had it.
				| None =>
					over(bone, rigging)
						* posed
							.locals
							.get(index)
							.copied()
							.unwrap_or(bone.rest)
							.matrix(),
			};

			rigging.push(model);
		}

		let Some(posed) = poses.get_mut(pose) else {
			return false;
		};

		for part in &ragdoll.parts {
			let index = usize::from(part.bone);
			let (Some(bone), Some(model)) = (bones.get(index), rigging.get(index)) else {
				continue;
			};

			if !bodies.alive(part.body) {
				continue;
			}

			posed.set(part.bone, Transform::from_matrix(over(bone, rigging).inverse() * *model));
		}

		true
	}

	/// Hands the world the queries a solver can answer.
	///
	/// The host's to call, once, at startup. The pointers inside address the
	/// executable rather than the game module, so nothing about a reload
	/// disturbs them and there is no second call to make. @ref
	/// [`physics`](crate::abi::physics).
	///
	/// @param physics - the table to install
	pub const fn install_physics(&mut self, physics: Physics) { self.physics = physics; }

	/// Traces a ray through the world.
	///
	/// @param info - where from, where to, and what to be blind to
	/// @return what was in the way, or a miss
	#[must_use]
	pub fn trace_ray(&self, info: &TraceInfo) -> TraceResult {
		self.physics.cast_ray(&self.bodies, info)
	}

	/// Sweeps an axis-aligned box through the world.
	///
	/// @param info - the sweep, whose `extents` are the box's half-extents
	/// @return what was in the way, or a miss
	#[must_use]
	pub fn trace_box(&self, info: &TraceInfo) -> TraceResult {
		self.physics.cast_shape(&self.bodies, info)
	}

	/// Creates a joint, remembering how the two bodies are turned right now.
	///
	/// The only way a weld or a hinge should be made: `rest` is the relative
	/// rotation the joint holds them at, and this is the one place that can see
	/// both bodies to work it out. @ref
	/// [`Joint::rest`](crate::abi::Joint::rest).
	///
	/// @param joint - what to create; its `rest` is overwritten
	/// @return the joint's handle, or [`JointId::NONE`] if the table is full
	pub fn join(&mut self, mut joint: Joint) -> JointId {
		let turned = |id| {
			self.bodies
				.get(id)
				.map_or(Quat::IDENTITY, |body| body.transform.rotation)
		};

		joint.rest = turned(joint.second) * turned(joint.first).inverse();

		self.joints.spawn(joint)
	}

	/// Creates a body that drives an entity, starting where the entity is.
	///
	/// The usual way to give something on screen a presence in the world. A
	/// body with nothing to draw is [`Bodies::spawn`] instead.
	///
	/// @param entity - what the body drives
	/// @param kind - what the solver may do with it
	/// @param shape - what it is shaped like
	/// @return the body's handle, or [`BodyId::NONE`] if the table is full or
	/// the entity handle was stale
	pub fn attach_body(&mut self, entity: EntityId, kind: BodyKind, shape: Shape) -> BodyId {
		let Some(&transform) = self.entities.transform(entity) else {
			return BodyId::NONE;
		};

		self.bodies
			.spawn(Body::new(kind, shape, transform).driving(entity))
	}

	/// Hits a body once, rather than pushing on it for a step.
	///
	/// **On `World` rather than on `Bodies`, and that is the whole reason it is
	/// here**: an impulse is a force times the length of a step, and the length
	/// of a step is [`dt`](Self::dt) - which the body table cannot see. Reading
	/// it off the world rather than off a constant is also what keeps this
	/// right the day the rate stops being one.
	///
	/// The conversion is exact rather than approximate. A velocity gains the
	/// force divided by the mass and multiplied by the step, so a force of one
	/// impulse per step gains the impulse divided by the mass, and the step
	/// length cancels.
	///
	/// One accumulator therefore serves both, which is what the missing inertia
	/// tensor forces: nothing outside the solver can turn an angular impulse
	/// into a spin, because the tensor is worked out per step from the mass and
	/// the shape and is deliberately never stored.
	///
	/// The one thing this is not is an impulse applied *after* the step's
	/// damping. This one is damped by the step it lands in.
	///
	/// @param id - which body
	/// @param impulse - newton-seconds, in world space
	/// @param at - where it lands, in world space
	/// @return whether it landed anywhere
	pub fn apply_impulse(&mut self, id: BodyId, impulse: Vec3, at: Vec3) -> bool {
		if self.dt <= 0.0 {
			return false;
		}

		self.bodies
			.apply_force_at(id, impulse / self.dt, at)
	}

	/// Moves a body and declares that it cut rather than traveled.
	///
	/// The entity is written here rather than at the next step, and snapped, so
	/// that nothing is drawn sliding across the gap. Ordinary movement is a
	/// write through [`Bodies::get_mut`] and wants none of this.
	///
	/// @param id - the body to move
	/// @param transform - where it now is
	/// @return `true` if the handle resolved
	pub fn teleport_body(&mut self, id: BodyId, transform: Transform) -> bool {
		let Some(body) = self.bodies.get_mut(id) else {
			return false;
		};

		body.transform = transform;
		let entity = body.entity;

		if let Some(slot) = self.entities.transform_mut(entity) {
			*slot = transform;
			self.entities.snap(entity);
		}

		true
	}
}

impl Default for World {
	fn default() -> Self { Self::new() }
}

/// Where a bone's parent is, out of a buffer of model matrices.
///
/// The identity for a root, and the identity again for a parent the buffer
/// does not reach - which is the same answer a bone standing on its own
/// deserves and is why the two cases are not told apart.
///
/// @note: the first of those two cannot be caught by a mutation and is kept
/// anyway, which is the fourth line in this ABI to be in that position and the
/// same reason each time: [`NO_PARENT`] is 65535 and a skeleton holds at most
/// [`MAX_BONES`], so the lookup below would miss and answer the same. It is
/// the only line that says what the sentinel means.
///
/// @param bone - the bone whose parent is wanted
/// @param matrices - one per bone, parents already written
fn over(bone: &Bone, matrices: &[Mat4]) -> Mat4 {
	if bone.parent == NO_PARENT {
		return Mat4::IDENTITY;
	}

	matrices
		.get(usize::from(bone.parent))
		.copied()
		.unwrap_or(Mat4::IDENTITY)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A world holding a two-limbed skeleton, a pose of it, and a ragdoll with
	/// a body spawned for each part.
	///
	/// The skeleton is a leg down `x` with a foot bone at the end that no part
	/// follows, which is what makes the "a bone with no body rides the one
	/// above it" question askable at all.
	fn strung() -> (Box<World>, PoseId, Ragdoll) {
		let bone = |name: &str, parent: u16, at: Vec3| Bone {
			name: name.to_owned(),
			parent,
			inverse_bind: Mat4::IDENTITY,
			rest: Transform::at(at),
		};
		let bones = vec![
			bone("hip", NO_PARENT, Vec3::new(0.0, 4.0, 0.0)),
			bone("thigh", 0, Vec3::ZERO),
			bone("shin", 1, Vec3::new(1.0, 0.0, 0.0)),
			bone("foot", 2, Vec3::new(2.0, 0.0, 0.0)),
		];

		let mut world = Box::new(World::new());
		let skeleton = world
			.skeletons
			.insert("models/hero/rig", SkeletonData { bones: bones.clone() });
		let pose = world
			.poses
			.spawn(Pose::resting(skeleton, world.skeletons.bones(skeleton)));

		let mut ragdoll = ragdoll::plan(
			&bones,
			&[Segment::new("thigh", "shin"), Segment::new("shin", "foot")],
			Build::DEFAULT,
		);

		for part in &mut ragdoll.parts {
			part.body = world.bodies.spawn(Body::new(
				BodyKind::Kinematic,
				part.shape,
				Transform::IDENTITY,
			));
		}

		(world, pose, ragdoll)
	}

	/// Where every bone of a pose is, in the model's own space.
	fn bones_at(world: &World, pose: PoseId) -> Vec<Vec3> {
		let skeleton = world
			.poses
			.get(pose)
			.expect("the pose is there")
			.skeleton;
		let mut out = Vec::new();

		world
			.poses
			.model(pose, world.skeletons.bones(skeleton), 1.0, &mut out);

		out.iter()
			.map(|matrix| matrix.w_axis.truncate())
			.collect()
	}

	#[test]
	fn a_bodys_place_is_the_pose_and_the_stance_together() {
		let (mut world, pose, ragdoll) = strung();
		let at = Transform {
			position: Vec3::new(10.0, 0.0, -3.0),
			rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
			scale: Vec3::ONE,
		};

		assert!(world.pull_ragdoll(pose, at, &ragdoll), "the pose is there");

		// the thigh bone rests at (0, 4, 0) and its segment runs a unit along
		// x, so the body's middle sits at (0.5, 4, 0) in the model. A quarter
		// turn about y sends model x to world -z, and the stance is ten along
		// x and three back.
		let thigh = world
			.bodies
			.get(ragdoll.parts[0].body)
			.expect("alive")
			.transform;

		assert!(
			thigh
				.position
				.abs_diff_eq(Vec3::new(10.0, 4.0, -3.5), 1e-4),
			"the body is carried by both the pose and the stance, got {}",
			thigh.position
		);
	}

	#[test]
	fn pulling_then_pushing_leaves_a_pose_exactly_where_it_was() {
		// the round trip is the whole contract: `in_bone` and its inverse have
		// to agree, and the two passes of the push have to undo what the pull
		// composed. Started from a pose that is *not* the rest, so an
		// implementation that quietly wrote rests would fail.
		let (mut world, pose, ragdoll) = strung();
		let at = Transform {
			position: Vec3::new(-2.0, 1.0, 0.5),
			rotation: Quat::from_rotation_z(0.7),
			scale: Vec3::ONE,
		};
		let bend = Transform {
			rotation: Quat::from_rotation_z(0.5),
			..Transform::IDENTITY
		};

		world
			.poses
			.get_mut(pose)
			.expect("the pose is there")
			.set(2, bend);

		let before = bones_at(&world, pose);

		assert!(world.pull_ragdoll(pose, at, &ragdoll));
		assert!(world.push_ragdoll(pose, at, &ragdoll));

		for (was, is) in before.iter().zip(bones_at(&world, pose)) {
			assert!(was.abs_diff_eq(is, 1e-4), "a bone moved: {was} became {is}");
		}
	}

	#[test]
	fn a_pull_follows_the_pose_it_is_handed_rather_than_the_one_before_it() {
		// what a game does every step while a character is on its feet: the
		// animation moves the bones and the bodies have to arrive at the new
		// places rather than at last step's.
		let (mut world, pose, ragdoll) = strung();
		let at = Transform::IDENTITY;

		world.pull_ragdoll(pose, at, &ragdoll);

		let was = world
			.bodies
			.get(ragdoll.parts[1].body)
			.expect("alive")
			.transform
			.position;

		world
			.poses
			.get_mut(pose)
			.expect("the pose is there")
			.set(1, Transform {
				rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
				..Transform::IDENTITY
			});
		world.pull_ragdoll(pose, at, &ragdoll);

		let is = world
			.bodies
			.get(ragdoll.parts[1].body)
			.expect("alive")
			.transform
			.position;

		assert!(
			was.abs_diff_eq(Vec3::new(2.0, 4.0, 0.0), 1e-4),
			"the shin's body starts a unit past the thigh, got {was}"
		);
		assert!(
			is.abs_diff_eq(Vec3::new(0.0, 6.0, 0.0), 1e-4),
			"and a quarter turn at the thigh swings it onto y, got {is}"
		);
	}

	#[test]
	fn a_body_shoved_by_hand_takes_its_bone_with_it() {
		let (mut world, pose, ragdoll) = strung();
		let at = Transform::IDENTITY;

		world.pull_ragdoll(pose, at, &ragdoll);

		let shin = ragdoll.parts[1];
		let moved = world
			.bodies
			.get(shin.body)
			.expect("alive")
			.transform
			.position + Vec3::new(0.0, 3.0, 0.0);

		world
			.bodies
			.get_mut(shin.body)
			.expect("alive")
			.transform
			.position = moved;

		assert!(world.push_ragdoll(pose, at, &ragdoll));

		// where the bone ends up is where the body says its own far end is:
		// the anchor is half a length back along the body's x.
		let places = bones_at(&world, pose);
		let wanted = world
			.bodies
			.get(shin.body)
			.expect("alive")
			.transform
			.matrix()
			.transform_point3(shin.anchor);

		assert!(
			places[2].abs_diff_eq(wanted, 1e-4),
			"the shin's bone follows its body, got {} for {}",
			places[2],
			wanted
		);
	}

	#[test]
	fn a_bone_no_part_follows_rides_the_one_above_it() {
		// the foot bone has no body of its own. It must end up where the shin
		// carried it, which is the whole reason the first pass walks every
		// bone rather than only the parts.
		let (mut world, pose, ragdoll) = strung();
		let at = Transform::IDENTITY;

		world.pull_ragdoll(pose, at, &ragdoll);

		let shin = ragdoll.parts[1];

		world
			.bodies
			.get_mut(shin.body)
			.expect("alive")
			.transform
			.position += Vec3::new(0.0, 3.0, 0.0);
		world.push_ragdoll(pose, at, &ragdoll);

		let places = bones_at(&world, pose);

		assert!(
			(places[3] - places[2]).abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1e-4),
			"the foot stays two units past the shin, got {}",
			places[3] - places[2]
		);
		assert!(
			places[3].y > 2.5,
			"and it went up with it rather than staying at the rest, got {}",
			places[3]
		);
	}

	#[test]
	fn a_ragdoll_never_writes_the_transform_a_character_stands_at() {
		let (mut world, pose, ragdoll) = strung();
		let entity = world.entities.spawn_at(Transform::at(Vec3::Y));
		let at = Transform::at(Vec3::new(4.0, 0.0, 0.0));

		world.pull_ragdoll(pose, at, &ragdoll);
		world
			.bodies
			.get_mut(ragdoll.parts[0].body)
			.expect("alive")
			.transform
			.position += Vec3::new(0.0, 9.0, 0.0);
		world.push_ragdoll(pose, at, &ragdoll);

		assert_eq!(
			world
				.entities
				.transform(entity)
				.map(|it| it.position),
			Some(Vec3::Y),
			"the entity stands where it stood, whatever the bodies did"
		);
	}

	#[test]
	fn a_part_with_no_body_is_stepped_over_rather_than_guessed_at() {
		let (mut world, pose, ragdoll) = strung();
		let mut unspawned = ragdoll.clone();
		unspawned.parts[0].body = BodyId::NONE;

		let bend = Transform {
			rotation: Quat::from_rotation_z(0.3),
			..Transform::IDENTITY
		};

		world
			.poses
			.get_mut(pose)
			.expect("the pose is there")
			.set(unspawned.parts[0].bone, bend);

		let before = bones_at(&world, pose);

		assert!(world.pull_ragdoll(pose, Transform::IDENTITY, &unspawned));
		assert!(
			world
				.bodies
				.get(ragdoll.parts[0].body)
				.expect("alive")
				.transform
				.position
				.abs_diff_eq(Vec3::ZERO, 1e-5),
			"the body nobody claimed was left where it was spawned"
		);

		assert!(world.push_ragdoll(pose, Transform::IDENTITY, &unspawned));

		for (was, is) in before.iter().zip(bones_at(&world, pose)) {
			assert!(was.abs_diff_eq(is, 1e-4), "and its bone was left where it was as well");
		}

		// exactly, not nearly. A part with no body still has a bone, and that
		// bone's place would be recovered through a matrix and read back apart
		// again on every step of a long fall - so "close enough" here is a
		// bone that walks away from where it was put.
		assert_eq!(
			world
				.poses
				.get(pose)
				.expect("the pose is there")
				.locals[usize::from(unspawned.parts[0].bone)],
			bend,
			"a bone no living body follows is not written at all"
		);
	}

	#[test]
	fn a_stale_pose_handle_moves_nothing_either_way() {
		let (mut world, pose, ragdoll) = strung();

		assert!(world.poses.despawn(pose), "it was there");
		assert!(!world.pull_ragdoll(pose, Transform::IDENTITY, &ragdoll), "nothing to read");
		assert!(!world.push_ragdoll(pose, Transform::IDENTITY, &ragdoll), "nothing to write");
		assert!(
			world
				.bodies
				.get(ragdoll.parts[0].body)
				.expect("alive")
				.transform
				.position
				.abs_diff_eq(Vec3::ZERO, 1e-5),
			"and no body moved"
		);
	}

	#[test]
	fn game_api_is_a_table_of_pointers() {
		assert_eq!(
			size_of::<GameApi>(),
			size_of::<u64>() * 4,
			"a version word and three function pointers, padded to eight bytes"
		);
	}

	#[test]
	fn a_default_world_starts_opaque_black_and_empty() {
		let world = World::new();

		assert_eq!(world.clear, Vec3::ZERO, "nothing on screen until the game says so");
		assert!(world.aspect > 0.0, "aspect is a divisor and must never start at zero");
		assert!(world.light.length() > 0.0, "a zero light direction has no normal to take");
		assert!(world.entities.is_empty(), "and nothing exists yet");
		assert_eq!(world.state.layout(), 0, "nor has the game claimed its arena");
	}

	#[test]
	fn a_default_world_can_already_draw_the_built_in_shapes() {
		let world = World::new();

		assert_eq!(
			world.meshes.find(mesh::CUBE_NAME),
			MeshId::CUBE,
			"the primitives are seeded before anything asks for them"
		);
		assert_eq!(world.meshes.find(mesh::QUAD_NAME), MeshId::QUAD, "both of them");
		assert_eq!(
			world.meshes.find("meshes/crystal"),
			MeshId::NONE,
			"and nothing from disk until the host loads it"
		);
	}

	#[test]
	fn a_world_nobody_paces_draws_the_state_it_was_handed() {
		let mut world = World::new();
		let id = world.entities.spawn_at(Transform::at(Vec3::X));

		assert!(
			(world.interpolation() - 1.0).abs() < f32::EPSILON,
			"a fresh world is at the end of its step"
		);

		world.advance();
		if let Some(transform) = world.entities.transform_mut(id) {
			transform.position = Vec3::new(5.0, 0.0, 0.0);
		}

		assert_eq!(
			world.render_transform(id).map(|it| it.position),
			Some(Vec3::new(5.0, 0.0, 0.0)),
			"so a test or a screenshot sees exactly what it wrote"
		);
	}

	#[test]
	fn interpolation_is_clamped_however_it_is_set() {
		let mut world = World::new();

		world.set_interpolation(4.0);
		assert!(
			(world.interpolation() - 1.0).abs() < f32::EPSILON,
			"past the end of the step is the end of it"
		);

		world.set_interpolation(-1.0);
		assert!(world.interpolation().abs() < f32::EPSILON, "and before the start is the start");
	}

	#[test]
	fn the_render_camera_sits_between_the_two_steps() {
		let mut world = World::new();
		world.camera.position = Vec3::new(0.0, 0.0, 10.0);
		world.advance();
		world.camera.position = Vec3::new(0.0, 0.0, 20.0);
		world.set_interpolation(0.5);

		assert!(
			world
				.render_camera()
				.position
				.abs_diff_eq(Vec3::new(0.0, 0.0, 15.0), 1.0e-5),
			"the camera is interpolated too, or everything on screen judders when it moves"
		);
	}

	#[test]
	fn a_camera_that_merely_moved_is_still_interpolated_after_the_step() {
		let mut world = World::new();
		world.camera.position = Vec3::new(0.0, 0.0, 10.0);
		world.advance();

		world.camera.position = Vec3::new(0.0, 0.0, 20.0);
		// the ordinary case: the step ends, nothing was declared a cut.
		world.settle();
		world.set_interpolation(0.5);

		assert!(
			world
				.render_camera()
				.position
				.abs_diff_eq(Vec3::new(0.0, 0.0, 15.0), 1.0e-5),
			"settling must not snap a camera that did not ask to be snapped, or the camera \
			 stops interpolating entirely and judders against a scene that does not: got {}",
			world.render_camera().position
		);
	}

	#[test]
	fn a_camera_that_cut_does_not_slide_into_place() {
		let mut world = World::new();
		world.camera.position = Vec3::new(0.0, 0.0, 10.0);
		world.advance();

		world.camera.position = Vec3::new(0.0, 0.0, 90.0);
		world.snap_camera();
		// ordinary movement after the cut, in the same step: the cut still
		// holds, because it is applied at the end rather than where it is
		// written.
		world.camera.position.z += 1.0;
		world.settle();
		world.set_interpolation(0.5);

		assert!(
			world
				.render_camera()
				.position
				.abs_diff_eq(Vec3::new(0.0, 0.0, 91.0), 1.0e-5),
			"a cut camera is drawn where it landed, got {}",
			world.render_camera().position
		);
	}

	#[test]
	fn a_default_world_has_no_input_held() {
		let world = World::new();

		assert_eq!(world.input.keys, [0; input::KEY_WORDS], "nothing is down before a frame");
		assert_eq!(world.input.buttons, 0, "nothing is down before a frame");
	}

	#[test]
	fn a_world_on_its_own_is_its_own_authority() {
		let mut world = World::new();

		assert_eq!(world.peer, PeerId::HOST, "and it is not the zero value");
		assert!(world.peer.is_host());

		let body = world.bodies.spawn(Body::default());
		let claimed = world
			.bodies
			.spawn(Body { owner: PeerId::HOST, ..Body::default() });

		for id in [body, claimed] {
			assert_eq!(
				world
					.bodies
					.get(id)
					.map(|body| body.role(world.peer)),
				Some(Role::Authority),
				"nothing in an unnetworked world is anybody else's to decide"
			);
		}
	}

	#[test]
	fn a_world_that_is_a_client_decides_only_what_it_owns() {
		let mut world = World::new();
		let mine = PeerId::at(1, 1);

		world.peer = mine;

		let owned = world
			.bodies
			.spawn(Body { owner: mine, ..Body::default() });
		let theirs = world.bodies.spawn(Body {
			owner: PeerId::at(2, 1),
			..Body::default()
		});
		let nobodys = world.bodies.spawn(Body::default());

		let role = |id| {
			world
				.bodies
				.get(id)
				.map(|body| body.role(world.peer))
		};

		assert_eq!(role(owned), Some(Role::AutonomousProxy));
		assert_eq!(role(theirs), Some(Role::SimulatedProxy), "somebody else's is watched");
		assert_eq!(
			role(nobodys),
			Some(Role::SimulatedProxy),
			"and the map is the host's business, not everybody's"
		);
	}

	/// The two tables agree about who somebody is, which is the whole of what
	/// makes them one mechanism rather than two.
	#[test]
	fn a_command_is_filed_under_the_peer_the_world_minted() {
		let mut world = World::new();
		let peer = world.players.admit();
		let asked = Command {
			step: world.steps,
			number: 1,
			buttons: 0b10,
			yaw: 0.5,
			pitch: -0.25,
		};

		assert!(world.commands.kept(peer).is_empty(), "a world starts asking for nothing");
		assert!(world.commands.push(peer, asked));
		assert_eq!(world.commands.kept(peer), &[asked]);
		assert_eq!(world.commands.unsettled(peer), &[asked]);

		// and the identity that came out of one table is the one the other is
		// keyed by, generation and all: a peer let go by `players` is answered
		// with nothing here as soon as its slot is handed on.
		assert!(world.players.forget(peer));

		let next = world.players.admit();

		assert_eq!(next.slot(), peer.slot(), "the same slot, a different occupant");
		assert!(world.commands.push(next, asked), "and its ring starts over");
		assert!(
			world.commands.kept(peer).is_empty(),
			"so what the peer before asked for is not run for the one after"
		);
	}

	#[test]
	fn the_exported_symbol_is_nul_terminated() {
		assert_eq!(
			GAME_API_SYMBOL.last(),
			Some(&0),
			"GetProcAddress takes a C string, not a Rust slice"
		);
	}
}
