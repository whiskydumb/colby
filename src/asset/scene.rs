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
//!    0  SceneHeader                      160 bytes
//!  160  Setting                          112 bytes, one of them
//!    .  [Stood; stood_count]              76 bytes each
//!    .  [Bulk;  bulk_count]              132 bytes each
//!    .  [Tie;   tie_count]               100 bytes each
//!    .  [Bent;  bent_count]                24 bytes each
//!    .  [Local; locals_count]              40 bytes each
//!    .  [u32; stood_slots + bulk_slots + tie_slots + bent_slots + kept_slots]
//!    .  [Kept; kept_count]                20 bytes each
//!    .  every peer's arena, back to back
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
		net::MAX_PEERS,
		scene::{Arena, Form, Link, Posed, SceneData, Solid, Stage, Thing},
		state::STATE_BYTES,
	},
	bytemuck::{self, Pod, Zeroable},
	err,
	glam::{Quat, Vec3},
};

use crate::bytes::{AlignedBytes, Names, count, fits, span, width};

/// The eight bytes every `.cscene` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYSCN";

/// The revision of everything in this module.
///
/// Bump it whenever the header or any block changes shape. A file carrying a
/// different number is refused with a message rather than read as if it
/// agreed.
pub const FORMAT_VERSION: u32 = 5;

/// The extension a compiled or saved scene is written with.
pub const EXTENSION: &str = "cscene";

/// How big [`SceneHeader`] is, and where the first block starts.
pub const HEADER_BYTES: usize = 160;

/// The bit in [`SceneHeader::flags`] that says the file carries a game's arena.
///
/// A flag rather than a zero length, because an arena of zero bytes stamped
/// with a layout number is a thing a game can legitimately have and "there is
/// no arena at all" is not the same statement.
pub const FLAG_ARENA: u32 = 1;

/// The bit that says the file carries a block of arena per peer.
///
/// A separate flag from [`FLAG_ARENA`] for the reason that one exists at all:
/// a world where every peer's block is empty is a different statement from a
/// world written before peers had blocks, and only the second may be read as
/// "leave the table alone".
pub const FLAG_PLAYERS: u32 = 2;

/// Every flag this build knows about.
///
/// A file setting anything outside this is refused rather than read with the
/// unknown part ignored: the bit is there because some later version needed it
/// to be understood.
pub const KNOWN_FLAGS: u32 = FLAG_ARENA | FLAG_PLAYERS;

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

	/// Bytes per pose record. Must be `size_of::<Bent>()`.
	pub bent_stride: u32,

	/// Where the settings record starts, in bytes from the start of the file.
	pub setting_offset: u32,

	/// Where the entity block starts.
	pub stood_offset: u32,

	/// Where the body block starts.
	pub bulk_offset: u32,

	/// Where the joint block starts.
	pub tie_offset: u32,

	/// Where the pose block starts.
	pub bent_offset: u32,

	/// Where the block of bones every pose points into starts.
	pub locals_offset: u32,

	/// How many entities were alive.
	pub stood_count: u32,

	/// How many bodies were.
	pub bulk_count: u32,

	/// How many joints were.
	pub tie_count: u32,

	/// How many poses were.
	pub bent_count: u32,

	/// How many bones there are altogether, over every pose.
	pub locals_count: u32,

	/// How many slots the entity table had ever handed out.
	pub stood_slots: u32,

	/// The same for the body table.
	pub bulk_slots: u32,

	/// The same for the joint table.
	pub tie_slots: u32,

	/// The same for the pose table.
	pub bent_slots: u32,

	/// Where the generations start: the five tables' arrays, back to back, in
	/// the order the five counts above are in - entities, bodies, joints,
	/// poses, peers.
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

	/// How wide one [`Kept`] record is.
	pub kept_stride: u32,

	/// Where the [`Kept`] records start.
	pub kept_offset: u32,

	/// How many of them there are: one per peer that was here.
	pub kept_count: u32,

	/// How many peer slots the table had, for the generations array.
	pub kept_slots: u32,

	/// Where the peers' arena bytes start, all of them back to back.
	pub kept_bytes_offset: u32,

	/// How many of those bytes there are.
	pub kept_bytes_length: u32,

	/// Spare, so the header is a round hundred and sixty bytes and every block
	/// after it inherits the buffer's alignment.
	pub reserved: [u32; 3],
}

// the blocks after the header inherit the buffer's alignment only because the
// header is a multiple of it, and a field added without shrinking the spare
// would move all of them without anybody noticing until a cast failed. The
// same reasoning is why every block that has to be aligned is laid out before
// the first one whose length a game chooses. @ref `Places::of`.
const _: () = assert!(
	size_of::<SceneHeader>() == HEADER_BYTES,
	"the header has to stay a hundred and sixty bytes"
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

	/// Which pose moves it, as an index into the pose block, or
	/// [`NO_INDEX`](colby_core::abi::scene::NO_INDEX).
	pub pose: u32,
}

/// One posed skeleton, as the file holds it.
///
/// Its bones are not here: they vary in number and a record may not, so they
/// are a run in one block at the end, exactly as a name is a run in the string
/// blob. Two poses of one skeleton are two runs; two characters standing the
/// same way is a coincidence rather than a fact worth writing down once.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Bent {
	/// Offset into the blob of what it is called, or zero.
	pub name: u32,

	/// The slot it occupied.
	pub slot: u32,

	/// Which occupant of that slot it was.
	pub generation: u32,

	/// Offset into the blob of its skeleton's asset name, or zero.
	pub skeleton: u32,

	/// Where its bones start, as an index into the block of them.
	pub first: u32,

	/// How many bones it has.
	pub count: u32,
}

/// One peer's arena, as the file holds it.
///
/// Its bytes are not here, for the reason a pose's bones are not in a
/// [`Bent`]: they vary in length and a record may not, so they are a run in
/// one block, named by an offset and a length. The layout number is split into
/// halves like the world arena's and for the same reason: it keeps this record
/// four-byte business, so where it lands never depends on an eight-byte
/// value's alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Kept {
	/// Which peer slot this block belonged to.
	pub slot: u32,

	/// The low half of the layout number stamped on it.
	pub layout_low: u32,

	/// The high half of it.
	pub layout_high: u32,

	/// Where its bytes start, as an index into the block of them.
	pub first: u32,

	/// How many bytes it has.
	pub count: u32,
}

/// One bone of one pose, relative to its parent.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Local {
	/// Where it is.
	pub position: [f32; 3],

	/// How it is turned, xyzw.
	pub rotation: [f32; 4],

	/// How big it is along each axis.
	pub scale: [f32; 3],
}

/// The bit in [`Bulk::flags`] that says a body notices rather than pushes.
pub const BULK_SENSOR: u32 = 1;

/// The bit that says the solver had stopped integrating it.
pub const BULK_SLEEPING: u32 = 2;

/// The bit that says gravity does not reach it.
///
/// **This bit did not move [`FORMAT_VERSION`], and it is the one kind of field
/// that does not have to.** The rule everywhere else is that a new field bumps
/// the version, because a new field changes the size of a fixed record and
/// therefore every offset after it. A bit in a word that already exists changes
/// neither, and a file written before this bit had a meaning has it clear,
/// which reads as the body that falls. So an asset compiled by the previous
/// build is still correct and is not rebuilt, and a save taken by it still
/// loads.
pub const BULK_WEIGHTLESS: u32 = 4;

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

	/// [`BULK_SENSOR`], [`BULK_SLEEPING`] and [`BULK_WEIGHTLESS`].
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

/// The bit in [`Tie::flags`] that says the two bodies it holds still collide.
///
/// The bit means the *unusual* answer, so a record whose flags are zero is the
/// joint every engine hands out by default. @ref
/// [`Joint::collide`](colby_core::abi::Joint::collide).
pub const TIE_COLLIDE: u32 = 1;

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

	/// Which of the four it is, as [`JointKind`] in declaration order.
	pub kind: u32,

	/// [`TIE_COLLIDE`], and room for whatever comes after it.
	pub flags: u32,

	/// Which entry of the body block it holds, or [`u32::MAX`].
	pub first: u32,

	/// The other, or [`u32::MAX`] for a point in the world.
	pub second: u32,

	/// How far apart a rope lets them get.
	pub length: f32,

	/// How stiff the spring holding it together is, in hertz. Zero is rigid.
	pub stiffness: f32,

	/// How quickly that spring stops ringing, as a ratio.
	pub damping: f32,

	/// The most it may pull with over one step, or zero for no ceiling.
	pub max_impulse: f32,

	/// The most it may turn with, or zero for no ceiling.
	pub max_torque: f32,

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

	/// The pose block.
	#[must_use]
	pub fn bent(&self) -> &[Bent] { self.block(self.header.bent_offset, self.header.bent_count) }

	/// Every pose's bones, back to back; a record says where its own start.
	#[must_use]
	pub fn locals(&self) -> &[Local] {
		self.block(self.header.locals_offset, self.header.locals_count)
	}

	/// The five generation arrays, back to back.
	#[must_use]
	pub fn generations(&self) -> &[u32] {
		let total = self
			.header
			.stood_slots
			.saturating_add(self.header.bulk_slots)
			.saturating_add(self.header.tie_slots)
			.saturating_add(self.header.bent_slots)
			.saturating_add(self.header.kept_slots);

		self.block(self.header.generations_offset, total)
	}

	/// The peers' arena records, if the file carries any.
	#[must_use]
	pub fn kept(&self) -> &[Kept] {
		if self.header.flags & FLAG_PLAYERS == 0 {
			return &[];
		}

		self.block(self.header.kept_offset, self.header.kept_count)
	}

	/// One peer's arena, out of the run block.
	///
	/// @param kept - the record naming it
	#[must_use]
	pub fn block_of(&self, kept: &Kept) -> Arena {
		let layout = u64::from(kept.layout_low) | (u64::from(kept.layout_high) << u32::BITS);
		// checked rather than saturating: a saturated offset is a number that
		// still reads *somewhere*, and somewhere is worse than nowhere.
		let Some(start) = self
			.header
			.kept_bytes_offset
			.checked_add(kept.first)
		else {
			return Arena { layout, bytes: Vec::new() };
		};

		Arena {
			layout,
			bytes: self.block::<u8>(start, kept.count).to_vec(),
		}
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
		let bent = usize::try_from(self.header.bent_slots).unwrap_or(0);
		let after_ties = stood.saturating_add(bulk).saturating_add(tie);
		let after_poses = after_ties.saturating_add(bent);
		let kept = usize::try_from(self.header.kept_slots).unwrap_or(0);

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
				.get(stood.saturating_add(bulk)..after_ties)
				.unwrap_or_default()
				.to_vec(),
			posed: self
				.bent()
				.iter()
				.map(|it| self.pose(it))
				.collect(),
			player_arenas: self
				.kept()
				.iter()
				.map(|it| (it.slot, self.block_of(it)))
				.collect(),
			peer_generations: generations
				.get(after_poses..after_poses.saturating_add(kept))
				.unwrap_or_default()
				.to_vec(),
			pose_generations: generations
				.get(after_ties..after_poses)
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
			pose: stood.pose,
		}
	}

	/// One pose record, with its run of bones read out.
	///
	/// A run that reaches past the block is read as far as it goes rather than
	/// refused: the check that sized the block has already run, so what is
	/// left is a record disagreeing with it, and a character with half its
	/// bones is a better answer than a load that did not happen.
	fn pose(&self, bent: &Bent) -> Posed {
		let first = usize::try_from(bent.first).unwrap_or(usize::MAX);
		let count = usize::try_from(bent.count).unwrap_or(0);
		let run = self
			.locals()
			.get(first..first.saturating_add(count))
			.unwrap_or_default();

		Posed {
			name: self.name(bent.name).to_owned(),
			slot: bent.slot,
			generation: bent.generation,
			skeleton: self.name(bent.skeleton).to_owned(),
			locals: run
				.iter()
				.map(|local| transform_of(local.position, local.rotation, local.scale))
				.collect(),
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
			weightless: bulk.flags & BULK_WEIGHTLESS != 0,
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
			collide: tie.flags & TIE_COLLIDE != 0,
			first: tie.first,
			second: tie.second,
			first_anchor: Vec3::from_array(tie.first_anchor),
			second_anchor: Vec3::from_array(tie.second_anchor),
			axis: Vec3::from_array(tie.axis),
			length: tie.length,
			rest: Quat::from_array(tie.rest),
			stiffness: tie.stiffness,
			damping: tie.damping,
			max_impulse: tie.max_impulse,
			max_torque: tie.max_torque,
		}
	}

	/// A block, borrowed out of the buffer.
	fn block<T: Pod>(&self, offset: u32, count: u32) -> &[T] {
		block_of(self.bytes.as_slice(), offset, count)
	}
}

/// One record block, cast out of bytes that have already been checked.
///
/// A free function rather than a method because [`check`] reads the records
/// too, and two castings of the same block that could disagree is the sort of
/// thing that is only found by a file nobody has.
///
/// @param bytes - the whole file
/// @param offset - where the block starts, as the header stores it
/// @param count - how many records, as the header stores it
/// @return the records, or nothing at all if they cannot be read
fn block_of<T: Pod>(bytes: &[u8], offset: u32, count: u32) -> &[T] {
	let Some(range) = span::<T>(offset, count) else {
		return &[];
	};

	bytes
		.get(range)
		.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
		.unwrap_or(&[])
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
			pose: thing.pose,
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

	let mut locals: Vec<Local> = Vec::new();
	let bent: Vec<Bent> = data
		.posed
		.iter()
		.map(|posed| bent_of(posed, &mut names, &mut locals))
		.collect();

	let mut generations = data.thing_generations.clone();
	generations.extend_from_slice(&data.solid_generations);
	generations.extend_from_slice(&data.link_generations);
	generations.extend_from_slice(&data.pose_generations);
	generations.extend_from_slice(&data.peer_generations);

	let mut kept_bytes = Vec::new();
	let kept: Vec<Kept> = data
		.player_arenas
		.iter()
		.map(|(slot, arena)| kept_of(*slot, arena, &mut kept_bytes))
		.collect();

	let blocks = Blocks {
		stood: &stood,
		bulk: &bulk,
		tie: &tie,
		bent: &bent,
		locals: &locals,
		kept: &kept,
		kept_bytes: kept_bytes.len(),
	};
	let places = Places::of(&blocks, &generations, data.arena.as_ref());
	let header = head(data, &places, &blocks, names.blob().len())?;

	let mut out = Vec::with_capacity(places.names + names.blob().len());
	out.extend_from_slice(bytemuck::bytes_of(&header));
	out.extend_from_slice(bytemuck::bytes_of(&setting_of(data.stage)));
	out.extend_from_slice(bytemuck::cast_slice(&stood));
	out.extend_from_slice(bytemuck::cast_slice(&bulk));
	out.extend_from_slice(bytemuck::cast_slice(&tie));
	out.extend_from_slice(bytemuck::cast_slice(&bent));
	out.extend_from_slice(bytemuck::cast_slice(&locals));
	out.extend_from_slice(bytemuck::cast_slice(&generations));
	out.extend_from_slice(bytemuck::cast_slice(&kept));
	out.extend_from_slice(&kept_bytes);
	if let Some(arena) = data.arena.as_ref() {
		out.extend_from_slice(&arena.bytes);
	}
	out.extend_from_slice(names.blob());

	Ok(out)
}

/// Where each block lands, worked out once so the header and the writing
/// cannot disagree.
struct Places {
	setting: usize,
	stood: usize,
	bulk: usize,
	tie: usize,
	bent: usize,
	locals: usize,
	generations: usize,
	arena: usize,
	kept: usize,
	kept_bytes: usize,
	names: usize,
}

impl Places {
	/// Adds the blocks up in the order they are written.
	fn of(blocks: &Blocks<'_>, generations: &[u32], arena: Option<&Arena>) -> Self {
		let Blocks {
			stood,
			bulk,
			tie,
			bent,
			locals,
			kept,
			kept_bytes,
		} = *blocks;
		let setting = HEADER_BYTES;
		let stood_at = setting + size_of::<Setting>();
		let bulk_at = stood_at + size_of_val(stood);
		let tie_at = bulk_at + size_of_val(bulk);
		let bent_at = tie_at + size_of_val(tie);
		let locals_at = bent_at + size_of_val(bent);
		let generations_at = locals_at + size_of_val(locals);
		// the records come before the arena and not after it. They are
		// four-byte things and an arena is a run of bytes of any length a game
		// likes, so putting them after one would make whether this file is
		// readable depend on how long somebody's game state happened to be.
		// Everything after the records is bytes, which needs no alignment at
		// all.
		let kept_at = generations_at + size_of_val(generations);
		let kept_bytes_at = kept_at + size_of_val(kept);
		let arena_at = kept_bytes_at + kept_bytes;
		let names_at = arena_at + arena.map_or(0, |it| it.bytes.len());

		Self {
			setting,
			stood: stood_at,
			bulk: bulk_at,
			tie: tie_at,
			bent: bent_at,
			locals: locals_at,
			generations: generations_at,
			arena: arena_at,
			kept: kept_at,
			kept_bytes: kept_bytes_at,
			names: names_at,
		}
	}
}

/// Every record block, handed to the header filler as one argument.
struct Blocks<'a> {
	stood: &'a [Stood],
	bulk: &'a [Bulk],
	tie: &'a [Tie],
	bent: &'a [Bent],
	locals: &'a [Local],
	kept: &'a [Kept],
	kept_bytes: usize,
}

/// The header, filled from what has already been laid out.
fn head(
	data: &SceneData,
	places: &Places,
	blocks: &Blocks<'_>,
	names: usize,
) -> Result<SceneHeader> {
	let Blocks {
		stood,
		bulk,
		tie,
		bent,
		locals,
		kept,
		kept_bytes,
	} = *blocks;
	let layout = data.arena.as_ref().map_or(0, |it| it.layout);
	let arena_here = if data.arena.is_some() { FLAG_ARENA } else { 0 };
	// from both lists rather than one of them. The records are written from
	// `player_arenas` and the generations from `peer_generations`, so a flag
	// taken from either alone can announce "no peers here" over a file that
	// carries them - and the reader would then skip records nothing had
	// validated.
	let carrying_peers = !data.player_arenas.is_empty() || !data.peer_generations.is_empty();
	let peers_here = if carrying_peers { FLAG_PLAYERS } else { 0 };

	Ok(SceneHeader {
		magic: MAGIC,
		version: FORMAT_VERSION,
		flags: arena_here | peers_here,
		setting_stride: width::<Setting>("a scene's records")?,
		stood_stride: width::<Stood>("a scene's records")?,
		bulk_stride: width::<Bulk>("a scene's records")?,
		tie_stride: width::<Tie>("a scene's records")?,
		bent_stride: width::<Bent>("a scene's records")?,
		setting_offset: count(places.setting, "a scene's records")?,
		stood_offset: count(places.stood, "a scene's records")?,
		bulk_offset: count(places.bulk, "a scene's records")?,
		tie_offset: count(places.tie, "a scene's records")?,
		bent_offset: count(places.bent, "a scene's records")?,
		locals_offset: count(places.locals, "a scene's records")?,
		stood_count: count(stood.len(), "a scene's records")?,
		bulk_count: count(bulk.len(), "a scene's records")?,
		tie_count: count(tie.len(), "a scene's records")?,
		bent_count: count(bent.len(), "a scene's records")?,
		locals_count: count(locals.len(), "a scene's records")?,
		stood_slots: count(data.thing_generations.len(), "a scene's records")?,
		bulk_slots: count(data.solid_generations.len(), "a scene's records")?,
		tie_slots: count(data.link_generations.len(), "a scene's records")?,
		bent_slots: count(data.pose_generations.len(), "a scene's records")?,
		generations_offset: count(places.generations, "a scene's records")?,
		arena_layout_low: u32::try_from(layout & u64::from(u32::MAX)).unwrap_or(0),
		arena_layout_high: u32::try_from(layout >> u32::BITS).unwrap_or(0),
		arena_offset: count(places.arena, "a scene's records")?,
		arena_length: count(
			data.arena.as_ref().map_or(0, |it| it.bytes.len()),
			"a scene's records",
		)?,
		names_offset: count(places.names, "a scene's records")?,
		names_length: count(names, "a scene's records")?,
		kept_stride: width::<Kept>("a scene's records")?,
		kept_offset: count(places.kept, "a scene's records")?,
		kept_count: count(kept.len(), "a scene's records")?,
		kept_slots: count(data.peer_generations.len(), "a scene's records")?,
		kept_bytes_offset: count(places.kept_bytes, "a scene's records")?,
		kept_bytes_length: count(kept_bytes, "a scene's records")?,
		reserved: [0; 3],
	})
}

/// One peer's arena, with its bytes appended to the run block.
///
/// @param slot - which peer slot it belonged to
/// @param arena - its bytes and the number stamped on them
/// @param bytes - the run block every peer's bytes go into
fn kept_of(slot: u32, arena: &Arena, bytes: &mut Vec<u8>) -> Kept {
	let first = u32::try_from(bytes.len()).unwrap_or(0);

	bytes.extend_from_slice(&arena.bytes);

	Kept {
		slot,
		layout_low: u32::try_from(arena.layout & u64::from(u32::MAX)).unwrap_or(0),
		layout_high: u32::try_from(arena.layout >> u32::BITS).unwrap_or(0),
		first,
		count: u32::try_from(arena.bytes.len()).unwrap_or(0),
	}
}

/// One pose, as the file holds it, with its bones appended to the run block.
fn bent_of(posed: &Posed, names: &mut Names, locals: &mut Vec<Local>) -> Bent {
	let first = u32::try_from(locals.len()).unwrap_or(0);

	locals.extend(posed.locals.iter().map(|local| Local {
		position: local.position.to_array(),
		rotation: local.rotation.to_array(),
		scale: local.scale.to_array(),
	}));

	Bent {
		name: names.put(&posed.name),
		slot: posed.slot,
		generation: posed.generation,
		skeleton: names.put(&posed.skeleton),
		first,
		count: u32::try_from(posed.locals.len()).unwrap_or(0),
	}
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
	if solid.weightless {
		flags |= BULK_WEIGHTLESS;
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
		flags: if link.collide { TIE_COLLIDE } else { 0 },
		first: link.first,
		second: link.second,
		length: link.length,
		stiffness: link.stiffness,
		damping: link.damping,
		max_impulse: link.max_impulse,
		max_torque: link.max_torque,
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
/// The last arm is unreachable: [`codes`] has refused every code that is
/// not one of these before anything gets here.
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
///
/// The last arm is unreachable: [`codes`] has refused every code that is
/// not one of these before anything gets here.
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
///
/// The last arm is unreachable: [`codes`] has refused every code that is
/// not one of these before anything gets here.
const fn joint_kind(code: u32) -> JointKind {
	match code {
		| 1 => JointKind::Weld,
		| 2 => JointKind::Axis,
		| 3 => JointKind::Ball,
		| _ => JointKind::Rope,
	}
}

/// The code for a kind of joint.
const fn joint_code(kind: JointKind) -> u32 {
	match kind {
		| JointKind::Rope => 0,
		| JointKind::Weld => 1,
		| JointKind::Axis => 2,
		| JointKind::Ball => 3,
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

	// @note: the advice differs by where the file came from and this cannot
	// tell. A compiled scene is recompiled from its source the moment the
	// version moves, so a person never sees this. A *save* has no source, so
	// this is the end of it - which is the one way this format differs from
	// every other one here, and the message says so rather than offering
	// advice only half its callers can take.
	if header.version != FORMAT_VERSION {
		return Err(format!(
			"this scene is version {} and this build reads version {FORMAT_VERSION}; a compiled \
			 one is rebuilt from its source, a save is not",
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
	codes(bytes, &header)?;

	Ok(header)
}

/// Whether every code in a record is one this build knows.
///
/// **A code is refused and a flag bit is not**, and the difference is what
/// each one means. A bit says a record has a property, so a build that does
/// not know the bit reads a record without that property - which is a smaller
/// answer than the file's rather than a wrong one, and is the whole reason
/// [`BULK_WEIGHTLESS`] could be added without moving [`FORMAT_VERSION`]. A
/// code says *which* record this is, so a build that does not know it has
/// nothing smaller to fall back to: it would read a hinge as a rope, or a
/// ball as a rope, and go on as though the file had said so.
///
/// The cost is that a file written by a later build with one more kind in it
/// is refused by this one, and that is the point.
///
/// Runs after [`blocks`], which is what makes the casts here safe.
///
/// @param bytes - the whole file
/// @param header - its header, already checked
fn codes(bytes: &[u8], header: &SceneHeader) -> std::result::Result<(), String> {
	let bulks: &[Bulk] = block_of(bytes, header.bulk_offset, header.bulk_count);
	let ties: &[Tie] = block_of(bytes, header.tie_offset, header.tie_count);

	for bulk in bulks {
		if bulk.kind > 2 {
			return Err(format!(
				"a body in this scene is kind {}, which this build does not have",
				bulk.kind
			));
		}

		if bulk.shape_kind > 2 {
			return Err(format!(
				"a body in this scene is shaped {}, which this build does not have",
				bulk.shape_kind
			));
		}
	}

	for tie in ties {
		if tie.kind > 3 {
			return Err(format!(
				"a joint in this scene is kind {}, which this build does not have",
				tie.kind
			));
		}
	}

	Ok(())
}

/// Whether every record is the size this build reads.
fn strides(header: &SceneHeader) -> std::result::Result<(), String> {
	let widths = [
		(header.setting_stride, size_of::<Setting>(), "settings"),
		(header.stood_stride, size_of::<Stood>(), "entities"),
		(header.bulk_stride, size_of::<Bulk>(), "bodies"),
		(header.tie_stride, size_of::<Tie>(), "joints"),
		(header.bent_stride, size_of::<Bent>(), "poses"),
		(header.kept_stride, size_of::<Kept>(), "peers"),
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

	if usize::try_from(header.kept_slots).unwrap_or(usize::MAX) > MAX_PEERS {
		return Err(format!(
			"this scene claims {} peer slots and a world holds {MAX_PEERS}",
			header.kept_slots
		));
	}

	// and the *records* are bounded too, which is the check that was missing
	// beside the other three. One record is twenty bytes and names four
	// thousand, so an unbounded count is two hundred times the file in memory
	// before anything downstream gets to throw it away.
	if usize::try_from(header.kept_count).unwrap_or(usize::MAX) > MAX_PEERS {
		return Err(format!(
			"this scene carries {} peer blocks and a world holds {MAX_PEERS}",
			header.kept_count
		));
	}

	// every peer's block is an arena, so the whole run is bounded by what one
	// arena is times how many there can be. A file claiming more than that is
	// asking for a reservation rather than describing a world.
	let room = MAX_PEERS.saturating_mul(STATE_BYTES);
	if usize::try_from(header.kept_bytes_length).unwrap_or(usize::MAX) > room {
		return Err(format!(
			"this scene's peers hold {} bytes between them and there is room for {room}",
			header.kept_bytes_length
		));
	}

	let total = header
		.stood_slots
		.checked_add(header.bulk_slots)
		.and_then(|it| it.checked_add(header.tie_slots))
		.and_then(|it| it.checked_add(header.bent_slots))
		.and_then(|it| it.checked_add(header.kept_slots))
		.ok_or_else(|| "this scene claims more slots than a count holds".to_owned())?;

	fits::<Setting>(bytes, HEADER_BYTES, (header.setting_offset, 1), "settings")?;
	fits::<Stood>(bytes, HEADER_BYTES, (header.stood_offset, header.stood_count), "entities")?;
	fits::<Bulk>(bytes, HEADER_BYTES, (header.bulk_offset, header.bulk_count), "bodies")?;
	fits::<Tie>(bytes, HEADER_BYTES, (header.tie_offset, header.tie_count), "joints")?;
	fits::<Bent>(bytes, HEADER_BYTES, (header.bent_offset, header.bent_count), "poses")?;
	fits::<Local>(bytes, HEADER_BYTES, (header.locals_offset, header.locals_count), "bones")?;
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

	// a file that does not say it carries peers may not describe any either.
	// Without this a single word of `kept_slots` on a file with the bit clear
	// reaches into the block after the generations, comes back as a peer
	// table, and empties every block in the world it is loaded into.
	if header.flags & FLAG_PLAYERS == 0
		&& (header.kept_slots != 0 || header.kept_count != 0 || header.kept_bytes_length != 0)
	{
		return Err("this scene describes peers it says it does not carry".to_owned());
	}

	if header.flags & FLAG_PLAYERS != 0 {
		fits::<Kept>(bytes, HEADER_BYTES, (header.kept_offset, header.kept_count), "peers")?;
		fits::<u8>(
			bytes,
			HEADER_BYTES,
			(header.kept_bytes_offset, header.kept_bytes_length),
			"peer state",
		)?;

		// and each record's own run has to be inside the block the two fields
		// above just proved is inside the file. Without this a record naming a
		// run past the end reads as an empty arena rather than as a refusal,
		// which is a world quietly missing what somebody was holding.
		//
		// @note: read a record at a time rather than cast as a slice. This
		// takes a plain `&[u8]` and a cast wants the alignment the buffer has
		// and the argument does not promise.
		let records = span::<Kept>(header.kept_offset, header.kept_count)
			.and_then(|range| bytes.get(range))
			.unwrap_or_default();

		for chunk in records.chunks_exact(size_of::<Kept>()) {
			let one: Kept = bytemuck::pod_read_unaligned(chunk);
			let end = one
				.first
				.checked_add(one.count)
				.ok_or_else(|| "a peer's state runs past what a count holds".to_owned())?;

			if end > header.kept_bytes_length {
				return Err(format!(
					"a peer's state ends at {end} and the block is {} bytes",
					header.kept_bytes_length
				));
			}

			if usize::try_from(one.count).unwrap_or(usize::MAX) > STATE_BYTES {
				return Err(format!(
					"a peer's state is {} bytes and an arena is {STATE_BYTES}",
					one.count
				));
			}

			// refused here rather than dropped four layers down, where a slot
			// nobody has is silently no peer at all.
			if usize::try_from(one.slot).unwrap_or(usize::MAX) >= MAX_PEERS {
				return Err(format!(
					"a peer's state is for slot {} and a world holds {MAX_PEERS}",
					one.slot
				));
			}
		}
	}

	Ok(())
}

#[cfg(test)]
mod tests {
	use std::mem::offset_of;

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
				pose: 0,
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
				pose: scene::NO_INDEX,
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
				weightless: false,
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
				// the two flags set on different records, so a round trip that
				// dropped either would show it
				sensor: false,
				weightless: true,
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
			posed: vec![Posed {
				name: "hero".to_owned(),
				slot: 1,
				generation: 4,
				skeleton: "models/hero/rig".to_owned(),
				// two bones, neither of them at rest, so a round trip that
				// dropped the run block or read it at the wrong offset comes
				// back wrong rather than plausible
				locals: vec![Transform::at(Vec3::new(1.0, 2.0, 3.0)), Transform {
					position: Vec3::NEG_X,
					rotation: Quat::from_xyzw(TURN, 0.0, 0.0, TURN),
					scale: Vec3::splat(2.0),
				}],
			}],
			pose_generations: vec![0, 4],
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
				// four numbers a reader would not produce from nothing, so the
				// round trip has to carry every one of them
				stiffness: 12.5,
				damping: 0.4,
				max_impulse: 90.0,
				max_torque: 35.5,
				collide: true,
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
			// two peers in non-adjacent slots, of different lengths and
			// different layouts: a reader that took the run block at one
			// stride, or read the slots in order, comes back wrong rather than
			// plausible.
			player_arenas: vec![
				(0, Arena {
					layout: 0x0000_0005_0000_0001,
					bytes: vec![3; 16],
				}),
				(4, Arena { layout: 9, bytes: vec![8; STATE_BYTES] }),
			],
			peer_generations: vec![u32::MAX, 0, 0, 0, 2, 0, 0, 0, 0],
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
	fn whether_a_joint_lets_its_bodies_collide_survives_both_ways() {
		for collide in [false, true] {
			let mut data = sample();
			data.links[0].collide = collide;

			assert_eq!(
				round_trip(&data).links[0].collide,
				collide,
				"a joint that says {collide} comes back saying it"
			);
		}
	}

	#[test]
	fn a_joint_that_holds_its_bodies_apart_is_what_a_record_of_no_flags_is() {
		let mut data = sample();
		data.links[0].collide = false;

		let bytes = encode(&data).expect("it fits");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("readable");

		assert_eq!(
			file.tie()[0].flags,
			0,
			"the bit means the unusual answer, so the usual one writes nothing"
		);
	}

	/// The bytes of a scene with one field of one record overwritten.
	///
	/// @param field - how far into the record it is
	/// @param code - what to put there
	fn tie_kind_of(field: usize, code: u32) -> Vec<u8> {
		let mut bytes = encode(&sample()).expect("it fits");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("readable");
		let at = usize::try_from(file.header().tie_offset).expect("an offset") + field;

		bytes[at..at + 4].copy_from_slice(&code.to_le_bytes());

		bytes
	}

	#[test]
	fn a_kind_of_joint_this_build_does_not_have_is_refused_rather_than_read_as_a_rope() {
		let bytes = tie_kind_of(core::mem::offset_of!(Tie, kind), 9);
		let reason = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("a kind nothing answers to is not readable");

		assert!(
			format!("{reason}").contains('9'),
			"and the message says which one, got {reason}"
		);
	}

	#[test]
	fn a_flag_on_a_joint_this_build_does_not_know_is_ignored_rather_than_refused() {
		// the opposite of the rule above, and deliberately so: a bit says a
		// record has a property, and a build that does not know the bit reads a
		// record without it. That is what let the weightless bit be added
		// without moving the version, and refusing here would take it away.
		let bytes = tie_kind_of(core::mem::offset_of!(Tie, flags), 1 << 20);

		assert!(
			SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).is_ok(),
			"a bit from the future is a property this build does not have"
		);
	}

	#[test]
	fn a_kind_of_body_or_shape_this_build_does_not_have_is_refused_too() {
		for field in [core::mem::offset_of!(Bulk, kind), core::mem::offset_of!(Bulk, shape_kind)]
		{
			let mut bytes = encode(&sample()).expect("it fits");
			let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("readable");
			let at = usize::try_from(file.header().bulk_offset).expect("an offset") + field;

			bytes[at..at + 4].copy_from_slice(&7_u32.to_le_bytes());

			assert!(
				SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).is_err(),
				"a body code nothing answers to is refused rather than read as the first kind"
			);
		}
	}

	#[test]
	fn every_kind_of_body_shape_and_joint_survives() {
		let kinds = [BodyKind::Static, BodyKind::Kinematic, BodyKind::Dynamic];
		let shapes = [ShapeKind::Box, ShapeKind::Sphere, ShapeKind::Mesh];
		let joints = [JointKind::Rope, JointKind::Weld, JointKind::Axis, JointKind::Ball];

		for (index, kind) in kinds.into_iter().enumerate() {
			let mut data = sample();
			data.solids[0].kind = kind;
			data.solids[0].shape.kind = shapes[index];

			let back = round_trip(&data);

			assert_eq!(back.solids[0].kind, kind, "the body kind survives");
			assert_eq!(back.solids[0].shape.kind, shapes[index], "and the shape kind");
		}

		// a loop of its own, because there is one more kind of joint than
		// there are kinds of body and walking the two together would quietly
		// stop testing the last one.
		for kind in joints {
			let mut data = sample();
			data.links[0].kind = kind;

			assert_eq!(round_trip(&data).links[0].kind, kind, "the joint kind survives");
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
	fn every_peers_block_comes_back_at_its_own_slot_and_its_own_length() {
		let back = round_trip(&sample());
		let mine = sample();

		assert_eq!(back.player_arenas, mine.player_arenas, "slots, layouts and bytes");
		assert_eq!(back.peer_generations, mine.peer_generations, "and every slot's generation");

		// the two the sample was built to separate: a reader taking the run
		// block at one stride, or reading the records in order and assuming
		// slot equals position, gets past the assertion above only by luck.
		let lengths: Vec<usize> = back
			.player_arenas
			.iter()
			.map(|(_, arena)| arena.bytes.len())
			.collect();

		assert_eq!(lengths, vec![16, STATE_BYTES], "two different lengths, in order");
		assert_eq!(
			back.player_arenas[1].0, 4,
			"and the second record is at slot four rather than at slot one"
		);
	}

	#[test]
	fn a_scene_with_no_peers_in_it_says_so_rather_than_carrying_an_empty_block() {
		let bare = SceneData { player_arenas: Vec::new(), ..sample() };
		let bytes = encode(&SceneData { peer_generations: Vec::new(), ..bare }).expect("it fits");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("it reads");

		assert_eq!(file.header().flags & FLAG_PLAYERS, 0, "the flag is clear");
		assert!(file.kept().is_empty(), "and there is nothing to read");
		assert!(
			file.to_scene_data().peer_generations.is_empty(),
			"which is what an older description looks like, and is left alone on a restore"
		);
	}

	#[test]
	fn a_file_claiming_more_peers_than_a_world_holds_is_refused() {
		let bytes = encode(&sample()).expect("it fits");
		let mut broken = bytes;
		let at = offset_of!(SceneHeader, kept_slots);

		broken[at..at + 4].copy_from_slice(
			&u32::try_from(MAX_PEERS + 1)
				.unwrap_or(0)
				.to_le_bytes(),
		);

		let refused = SceneFile::from_bytes(AlignedBytes::from_slice(&broken))
			.expect_err("a world has nowhere to put them")
			.to_string();

		assert!(refused.contains("peer slots"), "and it says what it refused, got {refused}");
	}

	/// The header fields a hostile file can move on their own, each patched by
	/// itself on a file that is otherwise entirely valid.
	fn patched(field: usize, value: u32) -> String {
		let mut bytes = encode(&sample()).expect("it fits");

		bytes[field..field + 4].copy_from_slice(&value.to_le_bytes());

		SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("it should not read")
			.to_string()
	}

	#[test]
	fn a_file_carrying_more_peer_blocks_than_a_world_holds_is_refused() {
		let count = u32::try_from(MAX_PEERS + 1).unwrap_or(u32::MAX);
		let refused = patched(offset_of!(SceneHeader, kept_count), count);

		// twenty bytes of record naming four thousand of arena is two hundred
		// times the file in memory, so this is the one of the four bounds that
		// costs something to leave out.
		assert!(refused.contains("peer blocks"), "got {refused}");
	}

	#[test]
	fn a_file_that_says_it_has_no_peers_may_not_describe_any() {
		let bare = SceneData {
			player_arenas: Vec::new(),
			peer_generations: Vec::new(),
			..sample()
		};
		let mut bytes = encode(&bare).expect("it fits");
		let at = offset_of!(SceneHeader, kept_slots);

		assert_eq!(
			SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
				.expect("it reads as written")
				.header()
				.flags & FLAG_PLAYERS,
			0,
			"the flag is clear to begin with"
		);

		// one word. Without the check this reads a generation out of the block
		// after the generations, comes back as a peer table, and empties every
		// block in the world it is loaded into.
		bytes[at..at + 4].copy_from_slice(&1_u32.to_le_bytes());

		let refused = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect_err("it says it has none")
			.to_string();

		assert!(refused.contains("says it does not carry"), "got {refused}");
	}

	#[test]
	fn a_peers_block_for_a_slot_no_world_has_is_refused() {
		let bytes = encode(&sample()).expect("it fits");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("it reads");
		let at = usize::try_from(file.header().kept_offset).unwrap_or(0) + offset_of!(Kept, slot);

		drop(file);

		let mut broken = bytes;
		let slot = u32::try_from(MAX_PEERS).unwrap_or(u32::MAX);

		broken[at..at + 4].copy_from_slice(&slot.to_le_bytes());

		let refused = SceneFile::from_bytes(AlignedBytes::from_slice(&broken))
			.expect_err("no world has that slot")
			.to_string();

		assert!(refused.contains("for slot"), "got {refused}");
	}

	#[test]
	fn a_peer_record_of_the_wrong_width_is_refused_like_every_other_record() {
		let refused = patched(offset_of!(SceneHeader, kept_stride), 4);

		assert!(refused.contains("peers are 4 bytes each"), "got {refused}");
	}

	#[test]
	fn a_short_arena_does_not_push_the_peer_records_off_a_boundary() {
		// the records are laid out before the arena precisely so that this
		// cannot happen. A five-byte arena is legal - `put_raw` takes a short
		// slice - and used to leave `kept_offset` one byte off four.
		let odd = SceneData {
			arena: Some(Arena { layout: 3, bytes: vec![1; 5] }),
			..sample()
		};
		let back = round_trip(&odd);

		assert_eq!(back.player_arenas, odd.player_arenas, "the peers still read back");
		assert_eq!(
			back.arena.map(|it| it.bytes.len()),
			Some(5),
			"and so does an arena of a length nothing rounds"
		);
	}

	#[test]
	fn a_peers_block_that_runs_past_the_end_of_the_run_is_refused() {
		let bytes = encode(&sample()).expect("it fits");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("it reads");
		let at =
			usize::try_from(file.header().kept_offset).unwrap_or(0) + offset_of!(Kept, count);
		let length = file.header().kept_bytes_length;

		drop(file);

		let mut broken = bytes;

		// one byte more than the whole run holds, and deliberately not a number
		// that overflows: `u32::MAX` would be caught by the addition instead,
		// and the range check itself would go untested.
		broken[at..at + 4].copy_from_slice(&length.saturating_add(1).to_le_bytes());

		let refused = SceneFile::from_bytes(AlignedBytes::from_slice(&broken))
			.expect_err("a record cannot name a run that is not there")
			.to_string();

		assert!(refused.contains("block is"), "and it says so, got {refused}");
	}

	#[test]
	fn a_peers_block_longer_than_an_arena_is_refused() {
		let bytes = encode(&sample()).expect("it fits");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("it reads");
		let at =
			usize::try_from(file.header().kept_offset).unwrap_or(0) + offset_of!(Kept, count);
		let run = usize::try_from(file.header().kept_bytes_length).unwrap_or(0);

		drop(file);

		// a record inside the run block, and still longer than one arena is.
		// Two peers of four thousand bytes each leaves room for a record that
		// fits in the run and could never fit in a `GameState`.
		assert!(run > STATE_BYTES, "the sample has more than one arena's worth in it");

		let mut broken = bytes;
		let claimed = u32::try_from(STATE_BYTES + 1).unwrap_or(u32::MAX);

		broken[at..at + 4].copy_from_slice(&claimed.to_le_bytes());

		let refused = SceneFile::from_bytes(AlignedBytes::from_slice(&broken))
			.expect_err("no arena is that big")
			.to_string();

		assert!(refused.contains("an arena is"), "and it says so, got {refused}");
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
		let at = offset_of!(SceneHeader, bulk_stride);
		bytes[at..at + 4].copy_from_slice(&64_u32.to_le_bytes());

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
		let at = offset_of!(SceneHeader, arena_length);
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
		// where each offset sits, asked of the header rather than counted by
		// hand: a field added in the middle used to make every number here
		// address the wrong word, and the test then failed for a reason that
		// had nothing to do with what it is about.
		let blocks = [
			(offset_of!(SceneHeader, setting_offset), "settings"),
			(offset_of!(SceneHeader, stood_offset), "entities"),
			(offset_of!(SceneHeader, bulk_offset), "bodies"),
			(offset_of!(SceneHeader, tie_offset), "joints"),
			(offset_of!(SceneHeader, bent_offset), "poses"),
			(offset_of!(SceneHeader, locals_offset), "bones"),
			(offset_of!(SceneHeader, generations_offset), "generations"),
			(offset_of!(SceneHeader, arena_offset), "game state"),
			(offset_of!(SceneHeader, names_offset), "names"),
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
