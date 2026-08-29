//! colby's runtime scene format: `.cscene`.
//!
//! A world written down. Unlike every other format here it is not only
//! *compiled* into: it is also the first thing the engine writes for itself,
//! because a save is a world at a moment and there is nothing to compile it
//! from. One format serves both, which is deliberate - a saved game and an
//! authored level are the same list of things standing in the same places, and
//! two formats would be two readers, two versions and one of them always
//! behind.
//!
//! ```text
//!    0  SceneHeader                      128 bytes
//!  128  Setting                          112 bytes, one of them
//!    .  [Stood; stood_count]              72 bytes each
//!    .  [Bulk;  bulk_count]              132 bytes each
//!    .  [Tie;   tie_count]                84 bytes each
//!    .  [u32; stood_slots + bulk_slots + tie_slots]
//!    .  the game's own arena, if there is one
//!    .  the string blob, NUL-separated UTF-8
//! ```
//!
//! Every record block is `#[repr(C)]` and cast in place out of an
//! [`AlignedBytes`](crate::AlignedBytes), the way a `.cmesh` and a `.cmodel`
//! are. Names cannot be, so every name in a record is an offset into one blob
//! of NUL-terminated text at the end, offset zero being the empty string -
//! exactly the arrangement the model format uses and for the same reason.
//!
//! **What a record names and what it points at are different kinds of thing.**
//! An asset is named: `meshes/crystal`, resolved through a registry, because
//! where an asset landed this run says nothing about where it lands the next
//! one. A body's entity and a joint's bodies are *indices into this file*,
//! because the thing they point at is written down here beside them. Both of
//! those go through the same records whichever way the file is being loaded -
//! @ref [`scene`](colby_core::abi::scene) for the two loaders and why there
//! are two.
//!
//! **The generations block is what makes a handle survive a save.** One `u32`
//! per slot each table ever handed out, dead slots included, so a handle that
//! had already gone stale is still stale after a load. The free list is not
//! written: it is derived from whichever slots nothing occupies.

use std::path::Path;

use colby_core::{
	Result,
	abi::{
		BodyKind, Camera, JointKind, Layers, ShapeKind, Transform,
		scene::{Arena, Form, Link, SceneData, Solid, Stage, Thing},
		state::STATE_BYTES,
	},
	bytemuck::{self, Pod, Zeroable},
	err,
	glam::{Quat, Vec3},
};

use crate::bytes::{AlignedBytes, fits, span};

/// The eight bytes every `.cscene` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYSCN";

/// The revision of everything in this module.
///
/// Bump it whenever the header or any block changes shape. A file carrying a
/// different number is refused with a message rather than read as if it
/// agreed.
pub const FORMAT_VERSION: u32 = 1;

/// The extension a compiled or saved scene is written with.
pub const EXTENSION: &str = "cscene";

/// How big [`SceneHeader`] is, and where the first block starts.
pub const HEADER_BYTES: usize = 128;

/// The bit in [`SceneHeader::flags`] that says the file carries a game's arena.
///
/// A flag rather than a zero length, because an arena of zero bytes stamped
/// with a layout number is a thing a game can legitimately have and "there is
/// no arena at all" is not the same statement.
pub const FLAG_ARENA: u32 = 1;

/// Every flag this build knows about.
///
/// A file setting anything outside this is refused rather than read with the
/// unknown part ignored: the bit is there because some later version needed it
/// to be understood.
pub const KNOWN_FLAGS: u32 = FLAG_ARENA;

/// The largest string blob the reader will accept, in bytes.
pub const MAX_NAMES: usize = 1 << 20;

/// The fixed head of a `.cscene`.
///
/// Offsets are stored rather than implied, so a later version can insert a
/// block without moving the ones after it. The arena's layout number lives
/// here as two halves rather than in the block, which is what keeps every
/// block in the file four-byte business and out of any argument about where an
/// eight-byte value may start.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct SceneHeader {
	/// [`MAGIC`]. Anything else is not one of these files.
	pub magic: [u8; 8],

	/// [`FORMAT_VERSION`] at the time the file was written.
	pub version: u32,

	/// Which optional blocks are here. @ref [`KNOWN_FLAGS`].
	pub flags: u32,

	/// Bytes in the settings record. Must be `size_of::<Setting>()`.
	pub setting_stride: u32,

	/// Bytes per entity record. Must be `size_of::<Stood>()`.
	pub stood_stride: u32,

	/// Bytes per body record. Must be `size_of::<Bulk>()`.
	pub bulk_stride: u32,

	/// Bytes per joint record. Must be `size_of::<Tie>()`.
	pub tie_stride: u32,

	/// Where the settings record starts, in bytes from the start of the file.
	pub setting_offset: u32,

	/// Where the entity block starts.
	pub stood_offset: u32,

	/// Where the body block starts.
	pub bulk_offset: u32,

	/// Where the joint block starts.
	pub tie_offset: u32,

	/// How many entities were alive.
	pub stood_count: u32,

	/// How many bodies were.
	pub bulk_count: u32,

	/// How many joints were.
	pub tie_count: u32,

	/// How many slots the entity table had ever handed out.
	pub stood_slots: u32,

	/// The same for the body table.
	pub bulk_slots: u32,

	/// The same for the joint table.
	pub tie_slots: u32,

	/// Where the generations start: the three tables' arrays, back to back, in
	/// the order the three counts above are in.
	pub generations_offset: u32,

	/// The low half of the arena's layout number.
	pub arena_layout_low: u32,

	/// The high half of it.
	pub arena_layout_high: u32,

	/// Where the arena's bytes start.
	pub arena_offset: u32,

	/// How many of them there are.
	pub arena_length: u32,

	/// Where the string blob starts.
	pub names_offset: u32,

	/// How long the string blob is.
	pub names_length: u32,

	/// Spare, so the header is a round hundred and twenty-eight bytes and
	/// every block after it inherits the buffer's alignment.
	pub reserved: [u32; 7],
}

// the blocks after the header inherit the buffer's alignment only because the
// header is a multiple of it, and a field added without shrinking the spare
// would move all of them without anybody noticing until a cast failed.
const _: () = assert!(
	size_of::<SceneHeader>() == HEADER_BYTES,
	"the header has to stay a hundred and twenty-eight bytes"
);

/// The world's own settings: where it looks from, what lights it, how hard it
/// pulls, and what time it is.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Setting {
	/// Simulation steps so far. First, because it is the only eight-byte
	/// field and putting it anywhere else would pad the record.
	pub steps: u64,

	/// Where the camera is.
	pub camera_position: [f32; 3],

	/// What it looks at.
	pub camera_target: [f32; 3],

	/// Which way is up.
	pub camera_up: [f32; 3],

	/// Vertical field of view, in radians.
	pub fov_y: f32,

	/// The near plane.
	pub near: f32,

	/// The far plane.
	pub far: f32,

	/// The clear color, linear RGB.
	pub clear: [f32; 3],

	/// The direction the light travels.
	pub light: [f32; 3],

	/// How lit a surface facing away from it still is.
	pub ambient: [f32; 3],

	/// What every dynamic body accelerates by.
	pub gravity: [f32; 3],

	/// Simulated seconds so far.
	pub time: f32,

	/// Spare, and the reason this record has no padding in it.
	pub reserved: u32,
}

/// One entity standing somewhere, looking like something.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Stood {
	/// Offset into the blob of what it is called, or zero.
	pub name: u32,

	/// The slot it occupied.
	pub slot: u32,

	/// Which occupant of that slot it was.
	pub generation: u32,

	/// Offset of its mesh's asset name, or zero for nothing to draw.
	pub mesh: u32,

	/// Offset of its material's asset name, or zero for the default one.
	pub material: u32,

	/// Where it is.
	pub position: [f32; 3],

	/// How it is turned, xyzw.
	pub rotation: [f32; 4],

	/// How big it is along each axis.
	pub scale: [f32; 3],

	/// Its own tint, linear RGB.
	pub color: [f32; 3],
}

/// The bit in [`Bulk::flags`] that says a body notices rather than pushes.
pub const BULK_SENSOR: u32 = 1;

/// The bit that says the solver had stopped integrating it.
pub const BULK_SLEEPING: u32 = 2;

/// One body: its shape, where it is, how it moves and what it is made of.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Bulk {
	/// Offset into the blob of what it is called, or zero.
	pub name: u32,

	/// The slot it occupied.
	pub slot: u32,

	/// Which occupant of that slot it was.
	pub generation: u32,

	/// What the solver may do with it, as [`BodyKind`] in declaration order.
	pub kind: u32,

	/// Which of the three shapes it is, as [`ShapeKind`] in declaration order.
	pub shape_kind: u32,

	/// Offset of a mesh shape's asset name, or zero.
	pub shape_mesh: u32,

	/// Which entry of the entity block it drives, or [`u32::MAX`].
	pub thing: u32,

	/// [`BULK_SENSOR`] and [`BULK_SLEEPING`].
	pub flags: u32,

	/// The layers it is on.
	pub layer: u32,

	/// The layers it interacts with.
	pub mask: u32,

	/// The radius of a ball.
	pub radius: f32,

	/// How heavy it is.
	pub mass: f32,

	/// How much of an impact comes back.
	pub restitution: f32,

	/// How hard it is to slide along.
	pub friction: f32,

	/// The half-extents of a box.
	pub extents: [f32; 3],

	/// Where it is.
	pub position: [f32; 3],

	/// How it is turned, xyzw.
	pub rotation: [f32; 4],

	/// How big it is along each axis.
	pub scale: [f32; 3],

	/// How fast it is moving.
	pub velocity: [f32; 3],

	/// How fast it is turning.
	pub angular: [f32; 3],
}

/// One joint holding two bodies, or one body and a point in the world.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Tie {
	/// Offset into the blob of what it is called, or zero.
	pub name: u32,

	/// The slot it occupied.
	pub slot: u32,

	/// Which occupant of that slot it was.
	pub generation: u32,

	/// Which of the three it is, as [`JointKind`] in declaration order.
	pub kind: u32,

	/// Which entry of the body block it holds, or [`u32::MAX`].
	pub first: u32,

	/// The other, or [`u32::MAX`] for a point in the world.
	pub second: u32,

	/// How far apart a rope lets them get.
	pub length: f32,

	/// How much of the pull is given up each step.
	pub give: f32,

	/// Where it attaches on the first body, in that body's own space.
	pub first_anchor: [f32; 3],

	/// Where it attaches on the second, or in the world.
	pub second_anchor: [f32; 3],

	/// The axis a hinge turns about.
	pub axis: [f32; 3],

	/// The relative rotation it was made at, xyzw.
	pub rest: [f32; 4],
}

/// A `.cscene` held in memory, checked, and ready to be read in place.
#[derive(Clone, Debug)]
pub struct SceneFile {
	bytes: AlignedBytes,
	header: SceneHeader,
}

impl SceneFile {
	/// Reads and checks a scene.
	///
	/// @param path - the `.cscene` to read
	/// @return the file, or why it could not be used
	pub fn open(path: &Path) -> Result<Self> {
		let bytes = AlignedBytes::read(path)?;
		let header = check(bytes.as_slice())
			.map_err(|reason| err!(Asset("{}: {reason}", path.display())))?;

		Ok(Self { bytes, header })
	}

	/// Checks bytes that are already in memory.
	///
	/// @param bytes - the whole file
	/// @return the file, or why it could not be used
	pub fn from_bytes(bytes: AlignedBytes) -> Result<Self> {
		let header = check(bytes.as_slice()).map_err(|reason| err!(Asset("{reason}")))?;

		Ok(Self { bytes, header })
	}

	/// The header, as it was read.
	#[must_use]
	pub const fn header(&self) -> &SceneHeader { &self.header }

	/// The settings record.
	#[must_use]
	pub fn setting(&self) -> Setting {
		self.block::<Setting>(self.header.setting_offset, 1)
			.first()
			.copied()
			.unwrap_or(EMPTY_SETTING)
	}

	/// The entity block, borrowed out of the buffer.
	#[must_use]
	pub fn stood(&self) -> &[Stood] {
		self.block(self.header.stood_offset, self.header.stood_count)
	}

	/// The body block.
	#[must_use]
	pub fn bulk(&self) -> &[Bulk] { self.block(self.header.bulk_offset, self.header.bulk_count) }

	/// The joint block.
	#[must_use]
	pub fn tie(&self) -> &[Tie] { self.block(self.header.tie_offset, self.header.tie_count) }

	/// The three generation arrays, back to back.
	#[must_use]
	pub fn generations(&self) -> &[u32] {
		let total = self
			.header
			.stood_slots
			.saturating_add(self.header.bulk_slots)
			.saturating_add(self.header.tie_slots);

		self.block(self.header.generations_offset, total)
	}

	/// The game's own arena, if the file carries one.
	#[must_use]
	pub fn arena(&self) -> Option<Arena> {
		if self.header.flags & FLAG_ARENA == 0 {
			return None;
		}

		let layout = u64::from(self.header.arena_layout_low)
			| (u64::from(self.header.arena_layout_high) << u32::BITS);

		Some(Arena {
			layout,
			bytes: self
				.block::<u8>(self.header.arena_offset, self.header.arena_length)
				.to_vec(),
		})
	}

	/// One name out of the blob.
	///
	/// @param offset - what a record stored
	/// @return the text up to its terminator, or nothing when the offset is
	/// not one this file wrote
	#[must_use]
	pub fn name(&self, offset: u32) -> &str {
		let (Ok(start), Ok(base), Ok(length)) = (
			usize::try_from(offset),
			usize::try_from(self.header.names_offset),
			usize::try_from(self.header.names_length),
		) else {
			return "";
		};

		let blob = self
			.bytes
			.as_slice()
			.get(base..base.saturating_add(length))
			.unwrap_or_default();
		let rest = blob.get(start..).unwrap_or_default();
		let end = rest
			.iter()
			.position(|byte| *byte == 0)
			.unwrap_or(rest.len());

		std::str::from_utf8(rest.get(..end).unwrap_or_default()).unwrap_or("")
	}

	/// Copies the whole file into the description the two loaders read.
	#[must_use]
	pub fn to_scene_data(&self) -> SceneData {
		let generations = self.generations();
		let stood = usize::try_from(self.header.stood_slots).unwrap_or(0);
		let bulk = usize::try_from(self.header.bulk_slots).unwrap_or(0);
		let tie = usize::try_from(self.header.tie_slots).unwrap_or(0);

		SceneData {
			stage: stage_of(self.setting()),
			things: self
				.stood()
				.iter()
				.map(|it| self.thing(it))
				.collect(),
			solids: self
				.bulk()
				.iter()
				.map(|it| self.solid(it))
				.collect(),
			links: self
				.tie()
				.iter()
				.map(|it| self.link(it))
				.collect(),
			thing_generations: generations
				.get(..stood)
				.unwrap_or_default()
				.to_vec(),
			solid_generations: generations
				.get(stood..stood.saturating_add(bulk))
				.unwrap_or_default()
				.to_vec(),
			link_generations: generations
				.get(stood.saturating_add(bulk)..stood.saturating_add(bulk).saturating_add(tie))
				.unwrap_or_default()
				.to_vec(),
			arena: self.arena(),
		}
	}

	/// One entity record, with its names read out.
	fn thing(&self, stood: &Stood) -> Thing {
		Thing {
			name: self.name(stood.name).to_owned(),
			slot: stood.slot,
			generation: stood.generation,
			transform: transform_of(stood.position, stood.rotation, stood.scale),
			mesh: self.name(stood.mesh).to_owned(),
			material: self.name(stood.material).to_owned(),
			color: Vec3::from_array(stood.color),
		}
	}

	/// One body record.
	fn solid(&self, bulk: &Bulk) -> Solid {
		Solid {
			name: self.name(bulk.name).to_owned(),
			slot: bulk.slot,
			generation: bulk.generation,
			kind: body_kind(bulk.kind),
			shape: Form {
				kind: shape_kind(bulk.shape_kind),
				radius: bulk.radius,
				extents: Vec3::from_array(bulk.extents),
				mesh: self.name(bulk.shape_mesh).to_owned(),
			},
			transform: transform_of(bulk.position, bulk.rotation, bulk.scale),
			velocity: Vec3::from_array(bulk.velocity),
			angular: Vec3::from_array(bulk.angular),
			mass: bulk.mass,
			restitution: bulk.restitution,
			friction: bulk.friction,
			sensor: bulk.flags & BULK_SENSOR != 0,
			sleeping: bulk.flags & BULK_SLEEPING != 0,
			layers: Layers::new(bulk.layer, bulk.mask),
			thing: bulk.thing,
		}
	}

	/// One joint record.
	fn link(&self, tie: &Tie) -> Link {
		Link {
			name: self.name(tie.name).to_owned(),
			slot: tie.slot,
			generation: tie.generation,
			kind: joint_kind(tie.kind),
			first: tie.first,
			second: tie.second,
			first_anchor: Vec3::from_array(tie.first_anchor),
			second_anchor: Vec3::from_array(tie.second_anchor),
			axis: Vec3::from_array(tie.axis),
			length: tie.length,
			rest: Quat::from_array(tie.rest),
			give: tie.give,
		}
	}

	/// A block, borrowed out of the buffer.
	fn block<T: Pod>(&self, offset: u32, count: u32) -> &[T] {
		let Some(range) = span::<T>(offset, count) else {
			return &[];
		};

		self.bytes
			.as_slice()
			.get(range)
			.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
			.unwrap_or(&[])
	}
}

/// What a settings record reads as when there is not one.
const EMPTY_SETTING: Setting = Setting {
	steps: 0,
	camera_position: [0.0; 3],
	camera_target: [0.0; 3],
	camera_up: [0.0, 1.0, 0.0],
	fov_y: 1.0,
	near: 0.1,
	far: 200.0,
	clear: [0.0; 3],
	light: [0.0, -1.0, 0.0],
	ambient: [0.0; 3],
	gravity: [0.0; 3],
	time: 0.0,
	reserved: 0,
};

/// Writes a world out as a `.cscene`.
///
/// @param data - the description to write
/// @return the whole file, ready to put on disk
pub fn encode(data: &SceneData) -> Result<Vec<u8>> {
	let mut names = Names::default();
	let stood: Vec<Stood> = data
		.things
		.iter()
		.map(|thing| Stood {
			name: names.put(&thing.name),
			slot: thing.slot,
			generation: thing.generation,
			mesh: names.put(&thing.mesh),
			material: names.put(&thing.material),
			position: thing.transform.position.to_array(),
			rotation: thing.transform.rotation.to_array(),
			scale: thing.transform.scale.to_array(),
			color: thing.color.to_array(),
		})
		.collect();
	let bulk: Vec<Bulk> = data
		.solids
		.iter()
		.map(|solid| bulk_of(solid, &mut names))
		.collect();
	let tie: Vec<Tie> = data
		.links
		.iter()
		.map(|link| tie_of(link, &mut names))
		.collect();

	let mut generations = data.thing_generations.clone();
	generations.extend_from_slice(&data.solid_generations);
	generations.extend_from_slice(&data.link_generations);

	let places = Places::of(&stood, &bulk, &tie, &generations, data.arena.as_ref());
	let header = head(data, &places, (&stood, &bulk, &tie), names.blob.len())?;

	let mut out = Vec::with_capacity(places.names + names.blob.len());
	out.extend_from_slice(bytemuck::bytes_of(&header));
	out.extend_from_slice(bytemuck::bytes_of(&setting_of(data.stage)));
	out.extend_from_slice(bytemuck::cast_slice(&stood));
	out.extend_from_slice(bytemuck::cast_slice(&bulk));
	out.extend_from_slice(bytemuck::cast_slice(&tie));
	out.extend_from_slice(bytemuck::cast_slice(&generations));
	if let Some(arena) = data.arena.as_ref() {
		out.extend_from_slice(&arena.bytes);
	}
	out.extend_from_slice(&names.blob);

	Ok(out)
}

/// Where each block lands, worked out once so the header and the writing
/// cannot disagree.
struct Places {
	setting: usize,
	stood: usize,
	bulk: usize,
	tie: usize,
	generations: usize,
	arena: usize,
	names: usize,
}

impl Places {
	/// Adds the blocks up in the order they are written.
	fn of(
		stood: &[Stood],
		bulk: &[Bulk],
		tie: &[Tie],
		generations: &[u32],
		arena: Option<&Arena>,
	) -> Self {
		let setting = HEADER_BYTES;
		let stood_at = setting + size_of::<Setting>();
		let bulk_at = stood_at + size_of_val(stood);
		let tie_at = bulk_at + size_of_val(bulk);
		let generations_at = tie_at + size_of_val(tie);
		let arena_at = generations_at + size_of_val(generations);
		let names_at = arena_at + arena.map_or(0, |it| it.bytes.len());

		Self {
			setting,
			stood: stood_at,
			bulk: bulk_at,
			tie: tie_at,
			generations: generations_at,
			arena: arena_at,
			names: names_at,
		}
	}
}

/// The header, filled from what has already been laid out.
fn head(
	data: &SceneData,
	places: &Places,
	blocks: (&[Stood], &[Bulk], &[Tie]),
	names: usize,
) -> Result<SceneHeader> {
	let (stood, bulk, tie) = blocks;
	let layout = data.arena.as_ref().map_or(0, |it| it.layout);

	Ok(SceneHeader {
		magic: MAGIC,
		version: FORMAT_VERSION,
		flags: if data.arena.is_some() { FLAG_ARENA } else { 0 },
		setting_stride: width::<Setting>()?,
		stood_stride: width::<Stood>()?,
		bulk_stride: width::<Bulk>()?,
		tie_stride: width::<Tie>()?,
		setting_offset: count(places.setting)?,
		stood_offset: count(places.stood)?,
		bulk_offset: count(places.bulk)?,
		tie_offset: count(places.tie)?,
		stood_count: count(stood.len())?,
		bulk_count: count(bulk.len())?,
		tie_count: count(tie.len())?,
		stood_slots: count(data.thing_generations.len())?,
		bulk_slots: count(data.solid_generations.len())?,
		tie_slots: count(data.link_generations.len())?,
		generations_offset: count(places.generations)?,
		arena_layout_low: u32::try_from(layout & u64::from(u32::MAX)).unwrap_or(0),
		arena_layout_high: u32::try_from(layout >> u32::BITS).unwrap_or(0),
		arena_offset: count(places.arena)?,
		arena_length: count(data.arena.as_ref().map_or(0, |it| it.bytes.len()))?,
		names_offset: count(places.names)?,
		names_length: count(names)?,
		reserved: [0; 7],
	})
}

/// One body, as the file holds it.
fn bulk_of(solid: &Solid, names: &mut Names) -> Bulk {
	let mut flags = 0;
	if solid.sensor {
		flags |= BULK_SENSOR;
	}
	if solid.sleeping {
		flags |= BULK_SLEEPING;
	}

	Bulk {
		name: names.put(&solid.name),
		slot: solid.slot,
		generation: solid.generation,
		kind: body_code(solid.kind),
		shape_kind: shape_code(solid.shape.kind),
		shape_mesh: names.put(&solid.shape.mesh),
		thing: solid.thing,
		flags,
		layer: solid.layers.layer,
		mask: solid.layers.mask,
		radius: solid.shape.radius,
		mass: solid.mass,
		restitution: solid.restitution,
		friction: solid.friction,
		extents: solid.shape.extents.to_array(),
		position: solid.transform.position.to_array(),
		rotation: solid.transform.rotation.to_array(),
		scale: solid.transform.scale.to_array(),
		velocity: solid.velocity.to_array(),
		angular: solid.angular.to_array(),
	}
}

/// One joint, as the file holds it.
fn tie_of(link: &Link, names: &mut Names) -> Tie {
	Tie {
		name: names.put(&link.name),
		slot: link.slot,
		generation: link.generation,
		kind: joint_code(link.kind),
		first: link.first,
		second: link.second,
		length: link.length,
		give: link.give,
		first_anchor: link.first_anchor.to_array(),
		second_anchor: link.second_anchor.to_array(),
		axis: link.axis.to_array(),
		rest: link.rest.to_array(),
	}
}

/// The settings record, from the description.
fn setting_of(stage: Stage) -> Setting {
	Setting {
		steps: stage.steps,
		camera_position: stage.camera.position.to_array(),
		camera_target: stage.camera.target.to_array(),
		camera_up: stage.camera.up.to_array(),
		fov_y: stage.camera.fov_y,
		near: stage.camera.near,
		far: stage.camera.far,
		clear: stage.clear.to_array(),
		light: stage.light.to_array(),
		ambient: stage.ambient.to_array(),
		gravity: stage.gravity.to_array(),
		time: stage.time,
		reserved: 0,
	}
}

/// The description's settings, from the record.
fn stage_of(setting: Setting) -> Stage {
	Stage {
		camera: Camera {
			position: Vec3::from_array(setting.camera_position),
			target: Vec3::from_array(setting.camera_target),
			up: Vec3::from_array(setting.camera_up),
			fov_y: setting.fov_y,
			near: setting.near,
			far: setting.far,
		},
		clear: Vec3::from_array(setting.clear),
		light: Vec3::from_array(setting.light),
		ambient: Vec3::from_array(setting.ambient),
		gravity: Vec3::from_array(setting.gravity),
		time: setting.time,
		steps: setting.steps,
	}
}

/// A transform out of three arrays.
fn transform_of(position: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Transform {
	Transform {
		position: Vec3::from_array(position),
		rotation: Quat::from_array(rotation),
		scale: Vec3::from_array(scale),
	}
}

/// What kind of body a code stands for.
///
/// An unknown one is static, which is the safe way to be wrong: a body that
/// does not move cannot fall through the world or shove anything.
const fn body_kind(code: u32) -> BodyKind {
	match code {
		| 1 => BodyKind::Kinematic,
		| 2 => BodyKind::Dynamic,
		| _ => BodyKind::Static,
	}
}

/// The code for a kind of body.
const fn body_code(kind: BodyKind) -> u32 {
	match kind {
		| BodyKind::Static => 0,
		| BodyKind::Kinematic => 1,
		| BodyKind::Dynamic => 2,
	}
}

/// What kind of shape a code stands for.
const fn shape_kind(code: u32) -> ShapeKind {
	match code {
		| 1 => ShapeKind::Sphere,
		| 2 => ShapeKind::Mesh,
		| _ => ShapeKind::Box,
	}
}

/// The code for a kind of shape.
const fn shape_code(kind: ShapeKind) -> u32 {
	match kind {
		| ShapeKind::Box => 0,
		| ShapeKind::Sphere => 1,
		| ShapeKind::Mesh => 2,
	}
}

/// What kind of joint a code stands for.
const fn joint_kind(code: u32) -> JointKind {
	match code {
		| 1 => JointKind::Weld,
		| 2 => JointKind::Axis,
		| _ => JointKind::Rope,
	}
}

/// The code for a kind of joint.
const fn joint_code(kind: JointKind) -> u32 {
	match kind {
		| JointKind::Rope => 0,
		| JointKind::Weld => 1,
		| JointKind::Axis => 2,
	}
}

/// The version a `.cscene` claims, without reading the rest of it.
///
/// @param path - the file to look at
/// @return its version, or nothing when it is not one of these at all
#[must_use]
pub fn version_of(path: &Path) -> Option<u32> {
	let mut head = [0_u8; 12];
	let mut file = std::fs::File::open(path).ok()?;
	std::io::Read::read_exact(&mut file, &mut head).ok()?;

	if head.get(..MAGIC.len()) != Some(&MAGIC[..]) {
		return None;
	}

	let version: [u8; 4] = head.get(8..12)?.try_into().ok()?;

	Some(u32::from_le_bytes(version))
}

/// The blob being built, and where each name already in it starts.
#[derive(Default)]
struct Names {
	blob: Vec<u8>,
	written: Vec<(String, u32)>,
}

impl Names {
	/// Puts a name in, or finds the one already there.
	///
	/// The empty name is offset zero and is written once, at the head, because
	/// that is what a record naming nothing stores and a reader has to find a
	/// terminator there.
	fn put(&mut self, name: &str) -> u32 {
		if self.blob.is_empty() {
			self.blob.push(0);
		}

		if name.is_empty() {
			return 0;
		}

		if let Some((_, already)) = self
			.written
			.iter()
			.find(|(written, _)| written == name)
		{
			return *already;
		}

		let at = u32::try_from(self.blob.len()).unwrap_or(0);

		self.blob.extend_from_slice(name.as_bytes());
		self.blob.push(0);
		self.written.push((name.to_owned(), at));

		at
	}
}

/// A count that has to fit in the header.
fn count(value: usize) -> Result<u32> {
	u32::try_from(value)
		.map_err(|_| err!(Asset("a scene of {value} is more than one file holds")))
}

/// The width of a record, as the header stores it.
fn width<T>() -> Result<u32> { count(size_of::<T>()) }

/// Every way a `.cscene` can be wrong, checked once.
fn check(bytes: &[u8]) -> std::result::Result<SceneHeader, String> {
	let head = bytes.get(..HEADER_BYTES).ok_or_else(|| {
		format!("a scene is at least {HEADER_BYTES} bytes and this is {}", bytes.len())
	})?;
	let header: SceneHeader = *bytemuck::try_from_bytes(head)
		.map_err(|error| format!("the header could not be read: {error}"))?;

	if header.magic != MAGIC {
		return Err("this is not a colby scene".to_owned());
	}

	if header.version != FORMAT_VERSION {
		return Err(format!(
			"this scene is version {} and this build reads version {FORMAT_VERSION}",
			header.version
		));
	}

	if header.flags & !KNOWN_FLAGS != 0 {
		return Err(format!(
			"this scene uses feature {:#x}, which this build does not",
			header.flags & !KNOWN_FLAGS
		));
	}

	strides(&header)?;
	blocks(bytes, &header)?;

	Ok(header)
}

/// Whether every record is the size this build reads.
fn strides(header: &SceneHeader) -> std::result::Result<(), String> {
	let widths = [
		(header.setting_stride, size_of::<Setting>(), "settings"),
		(header.stood_stride, size_of::<Stood>(), "entities"),
		(header.bulk_stride, size_of::<Bulk>(), "bodies"),
		(header.tie_stride, size_of::<Tie>(), "joints"),
	];

	for (written, expected, what) in widths {
		if usize::try_from(written) != Ok(expected) {
			return Err(format!(
				"this scene's {what} are {written} bytes each and this build reads {expected}"
			));
		}
	}

	Ok(())
}

/// Whether every block is inside the file it claims to be in.
fn blocks(bytes: &[u8], header: &SceneHeader) -> std::result::Result<(), String> {
	if usize::try_from(header.names_length).unwrap_or(usize::MAX) > MAX_NAMES {
		return Err("this scene's names are longer than any real one's".to_owned());
	}

	if usize::try_from(header.arena_length).unwrap_or(usize::MAX) > STATE_BYTES {
		return Err(format!(
			"this scene's game state is {} bytes and the arena is {STATE_BYTES}",
			header.arena_length
		));
	}

	let total = header
		.stood_slots
		.checked_add(header.bulk_slots)
		.and_then(|it| it.checked_add(header.tie_slots))
		.ok_or_else(|| "this scene claims more slots than a count holds".to_owned())?;

	fits::<Setting>(bytes, HEADER_BYTES, (header.setting_offset, 1), "settings")?;
	fits::<Stood>(bytes, HEADER_BYTES, (header.stood_offset, header.stood_count), "entities")?;
	fits::<Bulk>(bytes, HEADER_BYTES, (header.bulk_offset, header.bulk_count), "bodies")?;
	fits::<Tie>(bytes, HEADER_BYTES, (header.tie_offset, header.tie_count), "joints")?;
	fits::<u32>(bytes, HEADER_BYTES, (header.generations_offset, total), "generations")?;
	fits::<u8>(bytes, HEADER_BYTES, (header.names_offset, header.names_length), "names")?;

	if header.flags & FLAG_ARENA != 0 {
		fits::<u8>(
			bytes,
			HEADER_BYTES,
			(header.arena_offset, header.arena_length),
			"game state",
		)?;
	}

	Ok(())
}

#[cfg(test)]
mod tests {
	use colby_core::abi::{
		Body, BodyId, Joint, Material, MeshData, Renderable, Shape, World, scene,
	};

	use super::*;

	/// Half of a quarter turn, in the two places a unit quaternion holds it.
	const TURN: f32 = std::f32::consts::FRAC_1_SQRT_2;

	/// The entities the sample stands.
	fn sample_things() -> Vec<Thing> {
		vec![
			Thing {
				name: "floor".to_owned(),
				slot: 0,
				generation: 1,
				transform: Transform::at(Vec3::NEG_Y),
				mesh: "meshes/crystal".to_owned(),
				material: "brass".to_owned(),
				color: Vec3::new(0.8, 0.7, 0.6),
			},
			Thing {
				name: String::new(),
				slot: 2,
				generation: 3,
				transform: Transform {
					position: Vec3::new(9.0, 8.0, 7.0),
					rotation: Quat::from_xyzw(0.0, TURN, 0.0, TURN),
					scale: Vec3::new(2.0, 3.0, 4.0),
				},
				mesh: "meshes/crystal".to_owned(),
				material: String::new(),
				color: Vec3::ONE,
			},
		]
	}

	/// The bodies it carries.
	fn sample_solids() -> Vec<Solid> {
		vec![
			Solid {
				name: "slab".to_owned(),
				slot: 1,
				generation: 2,
				kind: BodyKind::Kinematic,
				shape: Form {
					kind: ShapeKind::Mesh,
					radius: 0.0,
					extents: Vec3::ZERO,
					mesh: "meshes/crystal".to_owned(),
				},
				transform: Transform::at(Vec3::X),
				velocity: Vec3::new(1.0, 2.0, 3.0),
				angular: Vec3::new(0.5, 0.0, -0.5),
				mass: 7.0,
				restitution: 0.3,
				friction: 0.6,
				sensor: true,
				sleeping: false,
				layers: Layers::new(4, 12),
				thing: 1,
			},
			Solid {
				name: String::new(),
				slot: 4,
				generation: 1,
				kind: BodyKind::Dynamic,
				shape: Form {
					kind: ShapeKind::Sphere,
					radius: 0.75,
					extents: Vec3::ZERO,
					mesh: String::new(),
				},
				transform: Transform::IDENTITY,
				velocity: Vec3::ZERO,
				angular: Vec3::ZERO,
				mass: 1.0,
				restitution: 0.2,
				friction: 0.5,
				sensor: false,
				sleeping: true,
				layers: Layers::DEFAULT,
				thing: scene::NO_INDEX,
			},
		]
	}

	/// A description with one of everything a record can hold.
	fn sample() -> SceneData {
		SceneData {
			things: sample_things(),
			solids: sample_solids(),
			stage: Stage {
				camera: Camera {
					position: Vec3::new(1.0, 2.0, 3.0),
					target: Vec3::new(4.0, 5.0, 6.0),
					up: Vec3::Y,
					fov_y: 1.1,
					near: 0.2,
					far: 300.0,
				},
				clear: Vec3::new(0.1, 0.2, 0.3),
				light: Vec3::new(-0.4, -1.0, -0.3),
				ambient: Vec3::splat(0.25),
				gravity: Vec3::new(0.0, -9.81, 0.0),
				time: 12.5,
				steps: 5_000_000_000,
			},
			links: vec![Link {
				name: "rope".to_owned(),
				slot: 0,
				generation: 1,
				kind: JointKind::Axis,
				first: 0,
				second: scene::NO_INDEX,
				first_anchor: Vec3::Y,
				second_anchor: Vec3::new(0.0, 6.0, 0.0),
				axis: Vec3::X,
				length: 2.5,
				rest: Quat::from_xyzw(TURN, 0.0, 0.0, TURN),
				give: 0.1,
			}],
			thing_generations: vec![1, 0, 3],
			solid_generations: vec![0, 2, 0, 0, 1],
			link_generations: vec![1],
			// past what a u32 holds on purpose: the layout number is written as
			// two halves and only a number needing both proves the second one.
			arena: Some(Arena {
				layout: 0x0003_0000_0000_000C,
				bytes: vec![7; STATE_BYTES],
			}),
		}
	}

	/// Encodes and reads back, checking nothing complained on the way.
	fn round_trip(data: &SceneData) -> SceneData {
		let bytes = encode(data).expect("it fits in one file");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect("what this module wrote, this module reads");

		file.to_scene_data()
	}

	#[test]
	fn everything_written_comes_back() {
		let data = sample();

		assert_eq!(round_trip(&data), data, "a scene through the file is the scene");
	}

	#[test]
	fn a_world_survives_the_whole_way_round() {
		let mut world = World::new();
		world
			.meshes
			.insert("meshes/crystal", MeshData::default());
		world
			.materials
			.insert("brass", Material::colored(Vec3::ONE));

		let entity = world
			.entities
			.spawn_at(Transform::at(Vec3::new(2.0, 3.0, 4.0)));
		world.entities.set_renderable(
			entity,
			Renderable::of(
				world.meshes.find("meshes/crystal"),
				world.materials.find("brass"),
				Vec3::X,
			),
		);
		let body = world
			.bodies
			.spawn(Body::dynamic(Shape::ball(0.4), Transform::at(Vec3::Y), 2.0).driving(entity));
		world.join(Joint::rope(body, BodyId::NONE, (Vec3::ZERO, Vec3::Y * 5.0), 1.5));

		let written = scene::capture(&world);
		let read = round_trip(&written);

		let mut empty = World::new();
		empty
			.meshes
			.insert("meshes/crystal", MeshData::default());
		empty
			.materials
			.insert("brass", Material::colored(Vec3::ONE));
		scene::restore(&mut empty, &read).expect("the layouts agree");

		assert_eq!(
			scene::capture(&empty),
			written,
			"a world through a file and back describes itself the same way"
		);
	}

	#[test]
	fn a_name_two_records_share_is_written_once() {
		let data = sample();
		let bytes = encode(&data).expect("it fits");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("readable");
		let blob = file.header().names_length;

		let mut once = data;
		once.things[1].mesh = String::new();
		let shorter = encode(&once).expect("it fits");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&shorter)).expect("readable");

		assert_eq!(
			blob,
			file.header().names_length,
			"dropping the second use of a name changes nothing, because it was one copy"
		);
	}

	#[test]
	fn a_record_naming_nothing_reads_back_as_nothing() {
		let file =
			SceneFile::from_bytes(AlignedBytes::from_slice(&encode(&sample()).expect("it fits")))
				.expect("readable");

		assert_eq!(file.stood()[1].name, 0, "an empty name is offset zero");
		assert!(file.name(0).is_empty(), "and offset zero reads as nothing");
	}

	#[test]
	fn every_kind_of_body_shape_and_joint_survives() {
		let kinds = [BodyKind::Static, BodyKind::Kinematic, BodyKind::Dynamic];
		let shapes = [ShapeKind::Box, ShapeKind::Sphere, ShapeKind::Mesh];
		let joints = [JointKind::Rope, JointKind::Weld, JointKind::Axis];

		for (index, kind) in kinds.into_iter().enumerate() {
			let mut data = sample();
			data.solids[0].kind = kind;
			data.solids[0].shape.kind = shapes[index];
			data.links[0].kind = joints[index];

			let back = round_trip(&data);

			assert_eq!(back.solids[0].kind, kind, "the body kind survives");
			assert_eq!(back.solids[0].shape.kind, shapes[index], "and the shape kind");
			assert_eq!(back.links[0].kind, joints[index], "and the joint kind");
		}
	}

	#[test]
	fn a_scene_with_no_arena_says_so_rather_than_carrying_an_empty_one() {
		let mut data = sample();
		data.arena = None;

		let bytes = encode(&data).expect("it fits");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("readable");

		assert_eq!(file.header().flags & FLAG_ARENA, 0, "the flag is clear");
		assert!(file.arena().is_none(), "and there is nothing to read");
		assert_eq!(round_trip(&data), data, "which is what comes back");
	}

	#[test]
	fn an_arena_of_no_bytes_is_not_the_same_as_no_arena() {
		let mut data = sample();
		data.arena = Some(Arena { layout: 9, bytes: Vec::new() });

		let back = round_trip(&data);
		let arena = back.arena.expect("there is one");

		assert_eq!(arena.layout, 9, "stamped with the number it was written under");
		assert!(arena.bytes.is_empty(), "and holding nothing, which is a different claim");
	}

	#[test]
	fn the_three_generation_arrays_come_back_apart() {
		let back = round_trip(&sample());

		assert_eq!(back.thing_generations, vec![1, 0, 3], "the entity slots");
		assert_eq!(back.solid_generations, vec![0, 2, 0, 0, 1], "the body slots");
		assert_eq!(back.link_generations, vec![1], "and the joint slots");
	}

	#[test]
	fn the_version_can_be_read_without_the_rest_of_the_file() {
		let directory = std::env::temp_dir().join("colby_scene_version");
		std::fs::create_dir_all(&directory).expect("a temporary directory");
		let path = directory.join("one.cscene");
		std::fs::write(&path, encode(&sample()).expect("it fits")).expect("written");

		assert_eq!(version_of(&path), Some(FORMAT_VERSION), "a scene reports its version");

		std::fs::write(&path, b"not a scene at all").expect("written");

		assert_eq!(version_of(&path), None, "and something else reports nothing");

		std::fs::remove_dir_all(&directory).ok();
	}

	#[test]
	fn a_file_that_is_not_one_of_these_is_refused_with_a_reason() {
		let refused = |bytes: Vec<u8>| {
			SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
				.expect_err("it should not read")
				.to_string()
		};

		assert!(refused(vec![0; 8]).contains("at least"), "a file too short to hold a header");

		let mut wrong = encode(&sample()).expect("it fits");
		wrong[0] = b'X';

		assert!(refused(wrong).contains("not a colby scene"), "a file with the wrong magic");

		let mut old = encode(&sample()).expect("it fits");
		old[8] = 99;

		assert!(refused(old).contains("version"), "a file from another version");

		let mut strange = encode(&sample()).expect("it fits");
		strange[12] = 0x80;

		assert!(refused(strange).contains("feature"), "a file using a flag this build lacks");
	}

	#[test]
	fn a_block_that_runs_off_the_end_is_refused() {
		let mut bytes = encode(&sample()).expect("it fits");
		let truncated = bytes.len() - 40;
		bytes.truncate(truncated);

		let refused = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("it should not read")
			.to_string();

		assert!(
			refused.contains("the file is"),
			"a block past the end is named rather than read, got {refused}"
		);
	}

	#[test]
	fn a_record_of_the_wrong_width_is_refused_rather_than_misread() {
		let mut bytes = encode(&sample()).expect("it fits");
		// the body stride, which is the fifth u32 in the header.
		bytes[20..24].copy_from_slice(&64_u32.to_le_bytes());

		let refused = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("it should not read")
			.to_string();

		assert!(
			refused.contains("bytes each"),
			"the width is checked before anything is cast, got {refused}"
		);
	}

	#[test]
	fn an_arena_larger_than_the_arena_is_refused() {
		let mut bytes = encode(&sample()).expect("it fits");
		let length = u32::try_from(STATE_BYTES + 1).expect("small");
		// arena_length is the twenty-second u32 in the header.
		let at = 8 + 20 * 4;
		bytes[at..at + 4].copy_from_slice(&length.to_le_bytes());

		let refused = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("it should not read")
			.to_string();

		assert!(
			refused.contains("game state"),
			"a game state bigger than the arena is refused, got {refused}"
		);
	}
	#[test]
	fn each_block_is_checked_against_the_file_on_its_own() {
		// the offsets, in the order the header holds them, with the name the
		// reader uses for each.
		let blocks = [
			(32_usize, "settings"),
			(36, "entities"),
			(40, "bodies"),
			(44, "joints"),
			(72, "generations"),
			(84, "game state"),
			(92, "names"),
		];

		for (at, what) in blocks {
			let mut bytes = encode(&sample()).expect("it fits");
			bytes[at..at + 4].copy_from_slice(&0x4000_0000_u32.to_le_bytes());

			let refused = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
				.expect_err("a block past the end should not read")
				.to_string();

			assert!(
				refused.contains(what),
				"the {what} are named when they are the ones off the end, got {refused}"
			);
		}
	}
}
