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
pub mod physics;
pub mod pose;
pub mod registry;
pub mod scene;
pub mod skeleton;
pub mod state;
pub mod texture;
pub mod ui;

pub use self::{
	anim::{
		Channel, Clip, ClipData, ClipId, Clips, Interpolation, MAX_KEYS, MAX_NODES, MAX_TRACKS,
		NO_BONE, Node, Track, Tree,
	},
	camera::Camera,
	character::{Motion, Moved},
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
	physics::{
		Bodies, Body, BodyId, BodyKind, Layers, MAX_BODIES, MAX_OVERLAPS, MAX_TOUCHES, Overlap,
		Physics, Shape, ShapeKind, Touch, TouchKind, TraceFn, TraceInfo, TraceResult,
	},
	pose::{MAX_POSES, Pose, PoseId, Poses},
	registry::{Entry, Registry},
	scene::{
		Arena, Form, Link, Posed, Remap, Restored, Scene, SceneData, SceneId, Scenes, Solid,
		Stage, Thing,
	},
	skeleton::{Bone, MAX_BONES, NO_PARENT, Skeleton, SkeletonData, SkeletonId, Skeletons},
	state::GameState,
	texture::{Texel, Texture, TextureData, TextureId, Textures},
	ui::{DocumentData, DocumentId, Event, EventKind, Length, PanelId, Ui},
};

/// The revision of the types and signatures in this module.
///
/// The host refuses a module reporting a different value. Bump it whenever a
/// signature or a layout below changes; forgetting to is a crash rather than an
/// error message.
pub const ABI_VERSION: u32 = 33;

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

	/// The window's width divided by its height. Host-written; the renderer
	/// uses it so that a shape does not stretch with the window.
	pub aspect: f32,

	/// Keyboard and mouse for this step. Host-written.
	pub input: Input,

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

	/// Lines, shapes and words to draw over the world.
	///
	/// Host-owned plain data anyone with the world may write: the solver
	/// outlines its own bodies, gameplay marks up whatever it is working on.
	/// Swept at the top of each step rather than the bottom, because several
	/// frames are drawn between two steps. @ref [`debug`](crate::abi::debug).
	pub debug: Debug,

	/// The game's own state. The host allocates it and never looks inside.
	pub state: GameState,

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
			input: Input::default(),
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
			models: Models::new(),
			clips: Clips::new(),
			skeletons: Skeletons::new(),
			poses: Poses::new(),
			scenes: Scenes::new(),
			materials: Materials::new(),
			cvars: Cvars::new(),
			ui: Ui::new(),
			debug: Debug::new(),
			state: GameState::new(),
			physics: Physics::STUB,
			blending: Vec::new(),
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

#[cfg(test)]
mod tests {
	use super::*;

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
	fn the_exported_symbol_is_nul_terminated() {
		assert_eq!(
			GAME_API_SYMBOL.last(),
			Some(&0),
			"GetProcAddress takes a C string, not a Rust slice"
		);
	}
}
