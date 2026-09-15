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
//!    0  SceneHeader                      224 bytes
//!  224  Setting                          232 bytes, one of them
//!    .  [Stood; stood_count]              84 bytes each
//!    .  [Lit;   lit_count]                40 bytes each
//!    .  [Shed;  shed_count]               88 bytes each
//!    .  [Sod;   sod_count]                44 bytes each
//!    .  [Daub;  daub_count]               16 bytes each
//!    .  [Jot;   jot_count]                32 bytes each
//!    .  [Bulk;  bulk_count]              132 bytes each
//!    .  [Wet;   wet_count]                36 bytes each
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
		BodyKind, Camera, Decal, DecalKind, Emitter, EmitterKind, JointKind, Layers, Light,
		LightKind, Noted, Post, ShapeKind, Sky, SkyKind, SparkBlend, Spelled, Terrain,
		TerrainKind, TextureId, ToneMap, Transform, Water, WaterKind,
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
///
/// Sixteen since a light says whether it throws a shadow: one word on the light
/// record, which had no spare to take it. The word before it is a *kind*, read
/// as an index into a list of words and refused when it is not one of them, so
/// the trick that costs nothing - @ref [`BULK_WEIGHTLESS`], a bit in a word
/// that already exists - was not available: a build that did not know the bit
/// would refuse the file rather than ignore it. So the record grew, and what
/// grew is a flags word, which the *next* bit will be free to join.
///
/// Seventeen since the world says how hazy its air is: one word on the settings
/// record, whose spare the sky's environment had spent. The record is
/// eight-aligned by the `steps` at its top, so the word cost itself and a new
/// spare. @ref [`Setting::spare`].
///
/// Eighteen since an entity carries records: a block of [`Jot`]s, one a value
/// that differs from its record's default, each naming the record and the
/// field. The header had no spare words left and grew by four.
pub const FORMAT_VERSION: u32 = 18;

/// The extension a compiled or saved scene is written with.
pub const EXTENSION: &str = "cscene";

/// How big [`SceneHeader`] is, and where the first block starts.
pub const HEADER_BYTES: usize = 224;

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

	/// Bytes per light record. Must be `size_of::<Lit>()`.
	pub lit_stride: u32,

	/// Where the light block starts.
	pub lit_offset: u32,

	/// How many entities carried a light.
	///
	/// One record per lamp rather than a wider entity record: a light is the
	/// rare thing an entity has, and version 6 wrote none at all.
	pub lit_count: u32,

	/// Bytes per water record. Must be `size_of::<Wet>()`.
	pub wet_stride: u32,

	/// Where the water block starts.
	pub wet_offset: u32,

	/// How many bodies held a fluid.
	///
	/// One record per pool rather than a wider body record, for the light
	/// block's reason: water is the rare thing a body has, and every version
	/// before this one wrote none at all.
	pub wet_count: u32,

	/// Bytes per emitter record. Must be `size_of::<Shed>()`.
	pub shed_stride: u32,

	/// Where the emitter block starts.
	pub shed_offset: u32,

	/// How many entities carried an emitter.
	///
	/// One record per emitter rather than a wider entity record, for the
	/// light block's reason and with the same arithmetic behind it: throwing
	/// particles is the rare thing an entity does, and every version before
	/// this one wrote none at all.
	pub shed_count: u32,

	/// Bytes per terrain record. Must be `size_of::<Sod>()`.
	pub sod_stride: u32,

	/// Where the terrain block starts.
	pub sod_offset: u32,

	/// How many entities were ground.
	///
	/// One record per terrain rather than a wider entity record, for the
	/// light block's reason and with far more of it behind it: being ground
	/// is the rarest thing an entity does - a world has one or none - and
	/// every version before this one wrote none at all.
	pub sod_count: u32,

	/// Bytes per decal record. Must be `size_of::<Daub>()`.
	///
	/// This and the two after it are the three words the header kept spare,
	/// which is exactly what a block costs, so the header is still two hundred
	/// and eight bytes and a multiple of sixteen with nothing to spare. The
	/// next block added grows it by four words and keeps one of them spare,
	/// the way the water block did.
	pub daub_stride: u32,

	/// Where the decal block starts.
	pub daub_offset: u32,

	/// How many entities painted something.
	///
	/// One record per decal rather than a wider entity record, for the light
	/// block's reason: painting is the rare thing an entity does, and every
	/// version before this one wrote none at all.
	pub daub_count: u32,

	/// Bytes per record value. Must be `size_of::<Jot>()`.
	///
	/// This and the two after it are a block's three words, and the fourth is
	/// [`spare`](Self::spare): the decal block took the last spare words, so
	/// this one grew the header by four and keeps one over.
	pub jot_stride: u32,

	/// Where the record values start.
	pub jot_offset: u32,

	/// How many record values there are, over every entity.
	///
	/// One a value that differs from its record's default, rather than a record
	/// per entity: a world of a thousand crates at every default writes none.
	pub jot_count: u32,

	/// Nought. The first word the next block takes.
	pub spare: u32,
}

// the light block took the header's last three spare words, the water block
// grew it by four and left one over, the emitter block took that one and three
// more, the terrain block took the two those left and two more, the decal block
// took the three that left, and the record block grew it by four again and
// kept one - two hundred and twenty-four bytes with one word to spare.
//
// the blocks after the header inherit the buffer's alignment only because the
// header is a multiple of it, and a field added without shrinking the spare
// would move all of them without anybody noticing until a cast failed. The
// same reasoning is why every block that has to be aligned is laid out before
// the first one whose length a game chooses. @ref `Places::of`.
const _: () = assert!(
	size_of::<SceneHeader>() == HEADER_BYTES,
	"the header has to stay two hundred and twenty-four bytes"
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

	/// Which sky is drawn, as [`SkyKind`](colby_core::abi::SkyKind) in
	/// declaration order.
	pub sky_kind: u32,

	/// The color straight up, linear RGB.
	pub sky_zenith: [f32; 3],

	/// The color at eye level.
	pub sky_horizon: [f32; 3],

	/// The color straight down.
	pub sky_ground: [f32; 3],

	/// Which curve squeezes the picture, as
	/// [`ToneMap`](colby_core::abi::ToneMap) in declaration order.
	pub tonemap: u32,

	/// The value the reinhard curve maps to white.
	pub white: f32,

	/// Whether the exposure is measured, as nought or one.
	///
	/// A word rather than a byte, because every field of this record is four
	/// bytes and a bool that is one byte would put padding in it.
	pub auto_exposure: u32,

	/// The multiplier used when it is not.
	pub exposure: f32,

	/// Stops added to a measured one.
	pub exposure_bias: f32,

	/// The smallest a measured exposure may be.
	pub exposure_min: f32,

	/// The largest it may be.
	pub exposure_max: f32,

	/// How fast the eye adapts, per second.
	pub exposure_rate: f32,

	/// How much of the bright pass is added back.
	pub bloom: f32,

	/// How bright a pixel has to be to bloom.
	pub bloom_threshold: f32,

	/// The color a distant surface fades towards, linear RGB.
	pub fog: [f32; 3],

	/// How quickly it fades, per unit of distance.
	pub fog_density: f32,

	/// How much light the air catches around the sun.
	///
	/// **This field took the record's spare word and did not move
	/// [`FORMAT_VERSION`], which is the second kind of field that does not have
	/// to.** The first is a bit in a word that already exists, @ref
	/// [`BULK_WEIGHTLESS`]; this is the word itself. The spare was there
	/// because the eight-byte `steps` at the top makes the record
	/// eight-aligned, so its length has to be a multiple of eight and the
	/// fields before it add up to four short - and a file written before this
	/// word meant anything holds nought there, which reads as the world with
	/// no shafts in it. So an asset compiled by the previous build is still
	/// correct and is not rebuilt, and a save taken by it still loads.
	///
	/// The next field to arrive pays for two: itself and a new spare.
	pub shafts: f32,

	/// How far away the lens is focused; nought is a lens that holds
	/// everything sharp.
	pub focus: f32,

	/// How far off that a surface is blurred all the way.
	pub focus_range: f32,

	/// How wide the blur gets there, as a radius in pixels.
	pub blur: f32,

	/// Offset into the blob of the sky's environment name, or zero for none.
	///
	/// **This is the spare the lens paid for, spent, and it moved no
	/// [`FORMAT_VERSION`].** The record is the length it was, so a build that
	/// does not know this word reads every field before it correctly and reads
	/// this one as nought - which is the empty string, because offset nought is
	/// where the blob's empty name lives. What such a build does with the rest
	/// of a cubemap sky is refuse the *kind*: `sky_kind` is read as an index
	/// into a list of words and an index past the end of it reads as no sky at
	/// all, @ref [`stage_of`]. So an older build opens a newer file and shows a
	/// world with the clear color behind it, which is the graceful answer
	/// rather than the wrong one.
	///
	/// When it was spent there was no spare after it, and the next field to
	/// arrive paid four bytes of itself, four of a new spare, and the version:
	/// that was [`haze`](Self::haze).
	pub sky_cubemap: u32,

	/// How much of the light crossing a unit of air the air scatters.
	pub haze: f32,

	/// Nothing, and kept that way on purpose: the word the next field to arrive
	/// takes without moving [`FORMAT_VERSION`].
	///
	/// The record is eight-aligned by the `steps` at the top, so its length has
	/// to be a multiple of eight, and [`haze`](Self::haze) left it four short.
	/// A file written before anything means this word holds nought there, so
	/// whatever takes it next has to read nought as "the world does not say",
	/// which is what the two fields that took a spare before did.
	pub spare: u32,
}

// a record with padding in it is not `Pod`, so this would already have failed
// to compile - but it would have failed pointing at the derive rather than at
// the reason, and the reason is worth having written down where the fields are.
const _: () = assert!(
	size_of::<Setting>().is_multiple_of(align_of::<Setting>()),
	"a settings record has to be a whole number of its own alignment"
);

/// The bit in [`Stood::flags`] that says the entity is hidden, and with it
/// everything that hangs off it.
///
/// The bit means the unusual answer, the rule [`TIE_COLLIDE`] follows, so a
/// record whose flags are zero is an entity that is drawn. A bit this build
/// does not know is read as a property the record does not have, @ref
/// `codes`, which is what lets the next one be added without moving
/// [`FORMAT_VERSION`].
pub const STOOD_HIDDEN: u32 = 1;

/// The bit in [`Stood::flags`] that says decals leave the entity alone.
///
/// The second bit of the word, and the first to arrive the way
/// [`STOOD_HIDDEN`] said the next one would: as a bit carrying the unusual
/// answer, so a record of no flags is an entity decals paint. Unlike the
/// first it is not handed down to what hangs off the entity.
pub const STOOD_UNDECALED: u32 = 2;

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

	/// Which entity it hangs off, as an index into this block, or
	/// [`NO_INDEX`](colby_core::abi::scene::NO_INDEX) for one standing on its
	/// own. Its position, rotation and scale are then its place inside that
	/// entity.
	pub parent: u32,

	/// [`STOOD_HIDDEN`] and [`STOOD_UNDECALED`], and room for whatever comes
	/// after them.
	pub flags: u32,
}

/// One entity's terrain, as the file holds it.
///
/// Written only for an entity that is ground, so the ordinary world carries
/// none of these at all. **What is not here is the geometry**, and that is the
/// point of the whole card: the mesh a terrain builds is a function of these
/// nine numbers, so writing it down would be writing down a derivation - and a
/// megabyte of it. `colby_runtime::terrain` builds it back on the first step
/// after a load, which is also the step the body's mesh handle is corrected on.
///
/// The body itself **is** written, like any other body, and that is not a
/// contradiction: it is a thing in the world with a name, a layer mask and a
/// friction somebody may have changed, and the only field of it the terrain
/// owns is which mesh it collides against.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Sod {
	/// Which entry of the entity block this belongs to.
	pub thing: u32,

	/// What shape of ground it is, as
	/// [`TerrainKind`](colby_core::abi::TerrainKind) in declaration order.
	///
	/// A record is only written for ground, so nothing here should be the
	/// `none` word - but a reader that finds one takes it, for the reason
	/// [`Lit::kind`] gives.
	pub kind: u32,

	/// How wide the ground is on a side, in world units.
	pub size: f32,

	/// How far it rises and falls from end to end.
	pub height: f32,

	/// How many vertices there are on each side.
	pub side: u32,

	/// What the noise is seeded with.
	pub seed: u32,

	/// How many hills fit across it.
	pub frequency: f32,

	/// How many octaves are summed.
	pub octaves: u32,

	/// How loud each octave is against the last.
	pub roughness: f32,

	/// How many units of ground one texture repeat covers.
	pub tiling: f32,

	/// Whether anything can stand on it, as one or nought.
	///
	/// A word rather than a byte, because every other field of every other
	/// record here is four bytes wide and a `bool` in a `repr(C)` struct is a
	/// hole waiting for a padding byte to be read as data. Anything but nought
	/// is read as `true`, which is the rule a flag bit already keeps.
	pub solid: u32,
}

/// One entity's decal, as the file holds it.
///
/// Written only for an entity that paints, keyed by its place in the entity
/// block the way a [`Lit`] is. No blob and no picture: what a decal throws is
/// the entity's own material, already named in its [`Stood`], and its box is
/// the entity's own transform, already written there too. Which is why the
/// whole record is four words.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Daub {
	/// Which entry of the entity block this belongs to.
	pub thing: u32,

	/// What shape it paints into, as
	/// [`DecalKind`](colby_core::abi::DecalKind) in declaration order.
	///
	/// A record is only written for a decal that paints, so nothing here
	/// should be the `none` word - but a reader that finds one takes it, for
	/// the reason [`Lit::kind`] gives.
	pub kind: u32,

	/// How much it fades on a surface turned away from it.
	pub fade: f32,

	/// Which of two decals painting one surface is on top: the higher.
	pub order: i32,
}

/// The [`Jot::spelling`] of a flag: nought or one in the first word.
pub const JOT_TRUTH: u32 = 0;

/// The spelling of one number: a double, its low half in the first word and
/// its high half in the second.
pub const JOT_NUMBER: u32 = 1;

/// The spelling of a word: its offset into the blob in the first word.
///
/// Two, three and four numbers are spelled as their count, each an `f32` in a
/// word of its own, which is why this is five.
pub const JOT_WORD: u32 = 5;

/// One value of one record an entity carries, as the file holds it.
///
/// Keyed by the entity's place in the entity block the way a [`Lit`] is, and
/// written only for a value that differs from its record's default. **By name
/// rather than by place**: the record and the field are offsets into the blob,
/// and the value is a spelling with no kind, because the records a file is read
/// into are whatever the running game declares - a later build, or none yet.
/// @ref [`record`](colby_core::abi::record).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Jot {
	/// Which entry of the entity block this belongs to.
	pub thing: u32,

	/// Offset into the blob of the record's name.
	pub record: u32,

	/// Offset into the blob of the field's name.
	pub field: u32,

	/// How [`value`](Self::value) is spelled: [`JOT_TRUTH`], [`JOT_NUMBER`],
	/// two to four for as many numbers, or [`JOT_WORD`].
	///
	/// A spelling this build does not know reads as no value, for the reason a
	/// light of an unknown kind reads as no light.
	pub spelling: u32,

	/// The value, in the words its spelling says.
	pub value: [u32; 4],
}

/// One entity's light, as the file holds it.
///
/// Keyed by the entity the way a [`Bulk`] is, rather than folded into
/// [`Stood`]: nearly every entity in a world carries no light, and a wider
/// entity record would spend forty bytes on each of them to say so. Both
/// shapes bump [`FORMAT_VERSION`] once and only once; this one costs a world
/// of a thousand crates and two lamps eighty bytes instead of forty
/// thousand.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Lit {
	/// Which entry of the entity block this belongs to.
	pub thing: u32,

	/// Which shape it throws, as
	/// [`LightKind`](colby_core::abi::LightKind) in declaration order.
	///
	/// A record is only written for a light that is one of the lit kinds, so
	/// nothing here should be the `none` word - but a reader that finds one
	/// takes it, because a light of no kind is exactly the absence a missing
	/// record already means and refusing the file over it would be refusing a
	/// world for describing nothing twice.
	pub kind: u32,

	/// Its color, linear RGB.
	pub color: [f32; 3],

	/// How bright, as a multiplier on the color.
	pub intensity: f32,

	/// How far its contribution reaches.
	pub range: f32,

	/// The half-angle of a cone's bright middle, in radians.
	pub inner: f32,

	/// The half-angle of a cone's edge, in radians.
	pub outer: f32,

	/// What is unusual about it, as bits. @ref [`LIT_UNSHADOWED`].
	pub flags: u32,
}

/// The bit in [`Lit::flags`] that says nothing it lights throws a shadow.
///
/// **The unusual answer, the way [`STOOD_UNDECALED`] is**, because a lamp casts
/// by default: a record of no flags is a lamp that throws a shadow, which is
/// what a light placed by hand means. A bit this build does not know is read as
/// a property the record does not have, which is what lets the next one arrive
/// without moving [`FORMAT_VERSION`] again - and this word is here so that
/// there is a next one to arrive into.
pub const LIT_UNSHADOWED: u32 = 1;

/// One body's fluid, as the file holds it.
///
/// Written only for a body whose water is one of the filled kinds, so the
/// ordinary world carries none of these at all - which is the whole reason it
/// is a block of its own rather than six more words on every [`Bulk`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Wet {
	/// Which entry of the body block this belongs to.
	pub body: u32,

	/// What fills it, as [`WaterKind`](colby_core::abi::WaterKind) in
	/// declaration order.
	///
	/// A record is only written for water that is one of the filled kinds, so
	/// nothing here should be the `none` word - but a reader that finds one
	/// takes it, for the reason [`Lit::kind`] gives.
	pub kind: u32,

	/// How heavy the fluid is, in mass per cubic unit.
	pub density: f32,

	/// How hard it stops something bobbing.
	pub damp: f32,

	/// How hard it resists being moved through.
	pub linear_drag: f32,

	/// How hard it resists being turned in.
	pub angular_drag: f32,

	/// Which way it runs, in units a second.
	pub flow: [f32; 3],
}

/// One entity's emitter, as the file holds it.
///
/// Written only for an entity whose emitter is one of the throwing kinds, so
/// the ordinary world carries none of these at all - which is the whole reason
/// it is a block of its own rather than eighteen more words on every
/// [`Stood`].
///
/// **The picture is a name and not a handle**, which is the one way this
/// differs from [`Lit`]: a light holds only numbers and an emitter points at a
/// texture, so the offset into the string blob is here beside the numbers,
/// exactly as [`Stood::mesh`] and [`Stood::material`] are.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Shed {
	/// Which entry of the entity block this belongs to.
	pub thing: u32,

	/// What shape it throws into, as
	/// [`EmitterKind`](colby_core::abi::EmitterKind) in declaration order.
	///
	/// A record is only written for an emitter of one of the throwing kinds,
	/// so nothing here should be the `none` word - but a reader that finds one
	/// takes it, for the reason [`Lit::kind`] gives.
	pub kind: u32,

	/// How the cloud reaches the picture, as
	/// [`SparkBlend`](colby_core::abi::SparkBlend) in declaration order.
	pub blend: u32,

	/// Offset into the string blob of the picture's asset name, or zero.
	pub texture: u32,

	/// How many particles a second it throws.
	pub rate: f32,

	/// The most it may have alive at once.
	pub cap: u32,

	/// How long a particle lives, in seconds.
	pub life: f32,

	/// How much of that is thrown away at random.
	pub life_spread: f32,

	/// How fast a particle leaves, in units a second.
	pub speed: f32,

	/// How much of that is thrown away at random.
	pub speed_spread: f32,

	/// The half-angle of a cone's mouth, in radians.
	pub spread: f32,

	/// How wide a particle is when it is thrown.
	pub size: f32,

	/// How wide it is when it dies.
	pub size_end: f32,

	/// The color it is thrown with, linear RGB.
	pub color: [f32; 3],

	/// The color it dies with, linear RGB.
	pub color_end: [f32; 3],

	/// How opaque a particle is at its brightest.
	pub opacity: f32,

	/// How much of the world's gravity a particle feels.
	pub gravity: f32,

	/// How much of its speed it loses a second, as a share.
	pub drag: f32,
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

	/// The light block, one record per entity that carries one.
	#[must_use]
	pub fn lit(&self) -> &[Lit] { self.block(self.header.lit_offset, self.header.lit_count) }

	/// Every emitter record.
	#[must_use]
	pub fn shed(&self) -> &[Shed] { self.block(self.header.shed_offset, self.header.shed_count) }

	/// Every terrain record.
	#[must_use]
	pub fn sod(&self) -> &[Sod] { self.block(self.header.sod_offset, self.header.sod_count) }

	/// Every decal record.
	#[must_use]
	pub fn daub(&self) -> &[Daub] { self.block(self.header.daub_offset, self.header.daub_count) }

	/// Every record value, over every entity.
	#[must_use]
	pub fn jot(&self) -> &[Jot] { self.block(self.header.jot_offset, self.header.jot_count) }

	/// The body block.
	#[must_use]
	pub fn bulk(&self) -> &[Bulk] { self.block(self.header.bulk_offset, self.header.bulk_count) }

	/// The water block, one record per body that holds a fluid.
	#[must_use]
	pub fn wet(&self) -> &[Wet] { self.block(self.header.wet_offset, self.header.wet_count) }

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
		let things = self.things();

		// the bodies first and the water into them afterwards, for the reason
		// the lights go onto the entities afterwards and by the same rule: a
		// record naming a place the body block does not have is dropped rather
		// than refused.
		let mut solids: Vec<Solid> = self
			.bulk()
			.iter()
			.map(|it| self.solid(it))
			.collect();
		for record in self.wet() {
			if let Some(solid) = usize::try_from(record.body)
				.ok()
				.and_then(|index| solids.get_mut(index))
			{
				solid.water = water_of(record);
			}
		}

		SceneData {
			stage: stage_of(self.setting()),
			sky_cubemap: self.name(self.setting().sky_cubemap).to_owned(),
			things,
			solids,
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

	/// Every entity record, with what the four blocks keyed by an entity put
	/// onto it.
	///
	/// Lifted out of [`to_scene_data`](Self::to_scene_data), which counts its
	/// lines. The entities first and the lights onto them afterwards, because a
	/// light record names its entity by place in the block above. A record
	/// naming a place that is not there is dropped rather than refused: the
	/// block's length was checked against the file, so what is left is a record
	/// disagreeing with the entity block, and a world missing one lamp is a
	/// better answer than a load that did not happen. The same argument a
	/// pose's run of bones is read with.
	fn things(&self) -> Vec<Thing> {
		let mut things: Vec<Thing> = self
			.stood()
			.iter()
			.map(|it| self.thing(it))
			.collect();
		for record in self.lit() {
			if let Some(thing) = usize::try_from(record.thing)
				.ok()
				.and_then(|index| things.get_mut(index))
			{
				thing.light = light_of(record);
			}
		}

		// and the emitters the same way, and after the lights rather than
		// before them for no reason but that the block is written in that
		// order and a reader that walks the file in the file's own order is
		// one fewer thing to hold in your head.
		for record in self.shed() {
			if let Some(thing) = usize::try_from(record.thing)
				.ok()
				.and_then(|index| things.get_mut(index))
			{
				thing.emitter = emitter_of(record);
				self.name(record.texture)
					.clone_into(&mut thing.emitter_texture);
			}
		}

		// and the ground after the two, in the file's own order again. It
		// names no asset at all, so there is nothing beside it to look up.
		for record in self.sod() {
			if let Some(thing) = usize::try_from(record.thing)
				.ok()
				.and_then(|index| things.get_mut(index))
			{
				thing.terrain = terrain_of(record);
			}
		}

		// and the decals, in the file's own order a fourth time, naming nothing
		// either: what a decal throws is the entity's own material.
		for record in self.daub() {
			if let Some(thing) = usize::try_from(record.thing)
				.ok()
				.and_then(|index| things.get_mut(index))
			{
				thing.decal = decal_of(record);
			}
		}

		// and what the entities' records hold, in the file's own order a fifth
		// time and in the order each entity's values were written
		for record in self.jot() {
			let noted = self.noted(record);

			if let Some((thing, noted)) = usize::try_from(record.thing)
				.ok()
				.and_then(|index| things.get_mut(index))
				.zip(noted)
			{
				thing.records.push(noted);
			}
		}

		things
	}

	/// One record value, with its names read out.
	///
	/// @return nothing for a spelling this build does not know, or a value that
	/// names no record or no field, which no declared record could take
	fn noted(&self, jot: &Jot) -> Option<Noted> {
		let [first, second, ..] = jot.value;
		let numbers = |count: usize| {
			jot.value
				.iter()
				.take(count)
				.map(|word| f32::from_bits(*word))
				.collect()
		};

		let value = match jot.spelling {
			| JOT_TRUTH => Spelled::Truth(first != 0),
			| JOT_NUMBER => Spelled::Number(f64::from_bits(
				u64::from(first) | (u64::from(second) << u32::BITS),
			)),
			| 2..=4 => Spelled::Numbers(numbers(usize::try_from(jot.spelling).ok()?)),
			| JOT_WORD => Spelled::Word(self.name(first).to_owned()),
			| _ => return None,
		};
		let record = self.name(jot.record);
		let field = self.name(jot.field);

		(!record.is_empty() && !field.is_empty()).then(|| Noted {
			record: record.to_owned(),
			field: field.to_owned(),
			value,
		})
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
			// put on afterwards by the caller, out of the light block.
			light: Light::NONE,
			// and the same, out of the emitter block.
			emitter: Emitter::NONE,
			emitter_texture: String::new(),
			// and the same again, out of the terrain block.
			terrain: Terrain::NONE,
			// and the same a fourth time, out of the decal block.
			decal: Decal::NONE,
			pose: stood.pose,
			parent: stood.parent,
			hidden: stood.flags & STOOD_HIDDEN != 0,
			takes_decals: stood.flags & STOOD_UNDECALED == 0,
			// and whatever its records hold, out of the record block.
			records: Vec::new(),
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
			// filled from the water block afterwards, the way a light is.
			water: Water::NONE,
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
	sky_kind: 0,
	sky_zenith: [0.0; 3],
	sky_horizon: [0.0; 3],
	sky_ground: [0.0; 3],
	tonemap: 0,
	white: 1.0,
	auto_exposure: 0,
	exposure: 1.0,
	exposure_bias: 0.0,
	exposure_min: 0.0,
	exposure_max: 1.0,
	exposure_rate: 0.0,
	bloom: 0.0,
	bloom_threshold: 1.0,
	fog: [0.0; 3],
	fog_density: 0.0,
	shafts: 0.0,
	focus: 0.0,
	focus_range: 0.0,
	blur: 0.0,
	sky_cubemap: 0,
	haze: 0.0,
	spare: 0,
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
		.map(|thing| stood_of(thing, &mut names))
		.collect();
	let carried = carried_of(data, &mut names)?;
	let jot = jots_of(data, &mut names)?;
	let bulk: Vec<Bulk> = data
		.solids
		.iter()
		.map(|solid| bulk_of(solid, &mut names))
		.collect();
	// one record per pool, indexed by the body's place in the block above, for
	// the reason a light's index is the entity's place: a piece grafted
	// somewhere else keeps its bodies' order and not their slots.
	let wet: Vec<Wet> = data
		.solids
		.iter()
		.enumerate()
		.filter(|(_, solid)| solid.water.kind.is_wet())
		.map(|(index, solid)| wet_of(index, solid.water))
		.collect::<Result<Vec<_>>>()?;
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

	// last of the records to be made, because the blob has to be finished
	// before its length goes into the header and this puts one name in it
	let setting = setting_of(data.stage, &data.sky_cubemap, &mut names);

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
		lit: &carried.lit,
		shed: &carried.shed,
		sod: &carried.sod,
		daub: &carried.daub,
		jot: &jot,
		bulk: &bulk,
		wet: &wet,
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
	out.extend_from_slice(bytemuck::bytes_of(&setting));
	out.extend_from_slice(bytemuck::cast_slice(&stood));
	out.extend_from_slice(bytemuck::cast_slice(&carried.lit));
	out.extend_from_slice(bytemuck::cast_slice(&carried.shed));
	out.extend_from_slice(bytemuck::cast_slice(&carried.sod));
	out.extend_from_slice(bytemuck::cast_slice(&carried.daub));
	out.extend_from_slice(bytemuck::cast_slice(&jot));
	out.extend_from_slice(bytemuck::cast_slice(&bulk));
	out.extend_from_slice(bytemuck::cast_slice(&wet));
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

/// The four blocks keyed by an entity: what it shines, what it throws, what
/// ground it is and what it paints.
struct Carried {
	lit: Vec<Lit>,
	shed: Vec<Shed>,
	sod: Vec<Sod>,
	daub: Vec<Daub>,
}

/// The four blocks keyed by an entity, built.
///
/// Lifted out of [`encode`], which counts its lines. One record per entity
/// that has the thing in question, and the index is the entity's place in the
/// entity block rather than its slot: a piece grafted somewhere else keeps its
/// entities' order and not their slots, and a record has to follow the entity
/// it is on. The same choice a body's `thing` makes.
///
/// @param data - the description being written
/// @param names - the blob an emitter's picture goes into, after every
/// entity's own names, which is the order the blob has always been written in
fn carried_of(data: &SceneData, names: &mut Names) -> Result<Carried> {
	let keyed = || data.things.iter().enumerate();

	Ok(Carried {
		lit: keyed()
			.filter(|(_, thing)| thing.light.kind.is_lit())
			.map(|(index, thing)| lit_of(index, thing.light))
			.collect::<Result<Vec<_>>>()?,
		shed: keyed()
			.filter(|(_, thing)| thing.emitter.kind.throws())
			.map(|(index, thing)| shed_of(index, thing, names))
			.collect::<Result<Vec<_>>>()?,
		sod: keyed()
			.filter(|(_, thing)| thing.terrain.is_ground())
			.map(|(index, thing)| sod_of(index, thing))
			.collect::<Result<Vec<_>>>()?,
		daub: keyed()
			.filter(|(_, thing)| thing.decal.paints())
			.map(|(index, thing)| daub_of(index, thing.decal))
			.collect::<Result<Vec<_>>>()?,
	})
}

/// Every record value in a description, one a value, keyed by the entity's
/// place in the entity block.
///
/// Lifted out of [`encode`], which counts its lines. The names go into the
/// blob after every entity's own and every emitter's picture, which is the
/// order the blob has always been written in.
///
/// @param data - the description being written
/// @param names - the blob the record's, the field's and a word's names go into
fn jots_of(data: &SceneData, names: &mut Names) -> Result<Vec<Jot>> {
	let mut jots = Vec::new();

	for (index, thing) in data.things.iter().enumerate() {
		for noted in &thing.records {
			jots.push(jot_of(index, noted, names)?);
		}
	}

	Ok(jots)
}

/// One record value, as the file holds it.
///
/// @param index - the entity's place in the entity block
/// @param noted - the value, by name
/// @param names - the blob its names go into
fn jot_of(index: usize, noted: &Noted, names: &mut Names) -> Result<Jot> {
	let mut value = [0; 4];
	let spelling = match &noted.value {
		| Spelled::Truth(truth) => {
			value[0] = u32::from(*truth);

			JOT_TRUTH
		},
		| Spelled::Number(number) => {
			let bits = number.to_bits();

			value[0] = u32::try_from(bits & u64::from(u32::MAX)).unwrap_or(0);
			value[1] = u32::try_from(bits >> u32::BITS).unwrap_or(0);

			JOT_NUMBER
		},
		| Spelled::Numbers(numbers) => {
			if !(2..=4).contains(&numbers.len()) {
				return Err(err!(Asset(
					"{}.{} holds {} numbers, and a record's value is two to four",
					noted.record,
					noted.field,
					numbers.len()
				)));
			}

			for (word, number) in value.iter_mut().zip(numbers) {
				*word = number.to_bits();
			}

			count(numbers.len(), "a record's numbers")?
		},
		| Spelled::Word(word) => {
			value[0] = names.put(word);

			JOT_WORD
		},
	};

	Ok(Jot {
		thing: count(index, "a scene's records")?,
		record: names.put(&noted.record),
		field: names.put(&noted.field),
		spelling,
		value,
	})
}

/// Where each block lands, worked out once so the header and the writing
/// cannot disagree.
struct Places {
	setting: usize,
	stood: usize,
	lit: usize,
	shed: usize,
	sod: usize,
	daub: usize,
	jot: usize,
	bulk: usize,
	wet: usize,
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
			lit,
			shed,
			sod,
			daub,
			jot,
			bulk,
			wet,
			tie,
			bent,
			locals,
			kept,
			kept_bytes,
		} = *blocks;
		let setting = HEADER_BYTES;
		let stood_at = setting + size_of::<Setting>();
		// beside the entities and before the bodies, because that is the order
		// the world reads in. Every offset is stored, so where a block lands is
		// a matter of what is legible rather than of what a reader can find.
		let lit_at = stood_at + size_of_val(stood);
		// and the emitters beside the lights, for the same argument: both hang
		// off an entity and both are read once the entity table is back.
		let shed_at = lit_at + size_of_val(lit);
		// and the ground beside both, for the same argument a third time.
		let sod_at = shed_at + size_of_val(shed);
		// and the decals beside all three, for the same argument a fourth time.
		let daub_at = sod_at + size_of_val(sod);
		// and the records' values beside the decals, a fifth time.
		let jot_at = daub_at + size_of_val(daub);
		let bulk_at = jot_at + size_of_val(jot);
		let wet_at = bulk_at + size_of_val(bulk);
		let tie_at = wet_at + size_of_val(wet);
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
			lit: lit_at,
			shed: shed_at,
			sod: sod_at,
			daub: daub_at,
			jot: jot_at,
			bulk: bulk_at,
			wet: wet_at,
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
	lit: &'a [Lit],
	shed: &'a [Shed],
	sod: &'a [Sod],
	daub: &'a [Daub],
	jot: &'a [Jot],
	bulk: &'a [Bulk],
	wet: &'a [Wet],
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
		lit,
		shed,
		sod,
		daub,
		jot,
		bulk,
		wet,
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
		lit_stride: width::<Lit>("a scene's records")?,
		lit_offset: count(places.lit, "a scene's records")?,
		lit_count: count(lit.len(), "a scene's records")?,
		wet_stride: width::<Wet>("a scene's records")?,
		wet_offset: count(places.wet, "a scene's records")?,
		wet_count: count(wet.len(), "a scene's records")?,
		shed_stride: width::<Shed>("a scene's records")?,
		shed_offset: count(places.shed, "a scene's records")?,
		shed_count: count(shed.len(), "a scene's records")?,
		sod_stride: width::<Sod>("a scene's records")?,
		sod_offset: count(places.sod, "a scene's records")?,
		sod_count: count(sod.len(), "a scene's records")?,
		daub_stride: width::<Daub>("a scene's records")?,
		daub_offset: count(places.daub, "a scene's records")?,
		daub_count: count(daub.len(), "a scene's records")?,
		jot_stride: width::<Jot>("a scene's records")?,
		jot_offset: count(places.jot, "a scene's records")?,
		jot_count: count(jot.len(), "a scene's records")?,
		spare: 0,
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

/// One body's fluid, as the file holds it.
fn wet_of(index: usize, water: Water) -> Result<Wet> {
	Ok(Wet {
		body: count(index, "a scene's records")?,
		kind: water.kind.index(),
		density: water.density,
		damp: water.damp,
		linear_drag: water.linear_drag,
		angular_drag: water.angular_drag,
		flow: water.flow.to_array(),
	})
}

/// One light, as the file holds it.
fn lit_of(index: usize, light: Light) -> Result<Lit> {
	Ok(Lit {
		thing: count(index, "a scene's records")?,
		kind: light.kind.index(),
		color: light.color.to_array(),
		intensity: light.intensity,
		range: light.range,
		inner: light.inner,
		outer: light.outer,
		flags: if light.shadow { 0 } else { LIT_UNSHADOWED },
	})
}

/// One body, with its names put in the blob.
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

/// The settings record, from the description, with its one name put in the
/// blob.
fn setting_of(stage: Stage, cubemap: &str, names: &mut Names) -> Setting {
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
		sky_kind: stage.sky.kind.index(),
		sky_zenith: stage.sky.zenith.to_array(),
		sky_horizon: stage.sky.horizon.to_array(),
		sky_ground: stage.sky.ground.to_array(),
		tonemap: stage.post.tonemap.index(),
		white: stage.post.white,
		auto_exposure: u32::from(stage.post.auto_exposure),
		exposure: stage.post.exposure,
		exposure_bias: stage.post.exposure_bias,
		exposure_min: stage.post.exposure_min,
		exposure_max: stage.post.exposure_max,
		exposure_rate: stage.post.exposure_rate,
		bloom: stage.post.bloom,
		bloom_threshold: stage.post.bloom_threshold,
		fog: stage.post.fog.to_array(),
		fog_density: stage.post.fog_density,
		shafts: stage.post.shafts,
		focus: stage.camera.focus,
		focus_range: stage.camera.focus_range,
		blur: stage.camera.blur,
		sky_cubemap: names.put(cubemap),
		haze: stage.post.haze,
		spare: 0,
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
			focus: setting.focus,
			focus_range: setting.focus_range,
			blur: setting.blur,
		},
		clear: Vec3::from_array(setting.clear),
		// a kind this build does not know reads as no sky, for the reason a
		// light of an unknown kind reads as no light. @ref `light_of`.
		sky: Sky {
			kind: SkyKind::at(setting.sky_kind).unwrap_or(SkyKind::None),
			zenith: Vec3::from_array(setting.sky_zenith),
			horizon: Vec3::from_array(setting.sky_horizon),
			ground: Vec3::from_array(setting.sky_ground),
			// named rather than numbered, so the handle is resolved by whoever
			// has a registry to resolve it against. @ref
			// `SceneData::sky_cubemap`.
			cubemap: TextureId::NONE,
		},
		// a curve this build does not know reads as none, the same way an
		// unknown light or sky kind does. @ref `light_of`.
		post: Post {
			tonemap: ToneMap::at(setting.tonemap).unwrap_or(ToneMap::None),
			white: setting.white,
			auto_exposure: setting.auto_exposure != 0,
			exposure: setting.exposure,
			exposure_bias: setting.exposure_bias,
			exposure_min: setting.exposure_min,
			exposure_max: setting.exposure_max,
			exposure_rate: setting.exposure_rate,
			bloom: setting.bloom,
			bloom_threshold: setting.bloom_threshold,
			fog: Vec3::from_array(setting.fog),
			fog_density: setting.fog_density,
			shafts: setting.shafts,
			haze: setting.haze,
		},
		light: Vec3::from_array(setting.light),
		ambient: Vec3::from_array(setting.ambient),
		gravity: Vec3::from_array(setting.gravity),
		time: setting.time,
		steps: setting.steps,
	}
}

/// One light record, as the world holds it.
///
/// A kind the list does not have reads as no light at all rather than as a
/// refusal, for the reason a `.cscene` refuses a *code* elsewhere and ignores
/// a flag bit: this build knowing fewer kinds than the writer did is the one
/// thing a version number already caught, and what is left is a file the
/// version agrees with carrying a number nothing in it means.
fn water_of(record: &Wet) -> Water {
	Water {
		kind: WaterKind::at(record.kind).unwrap_or(WaterKind::None),
		density: record.density,
		damp: record.damp,
		linear_drag: record.linear_drag,
		angular_drag: record.angular_drag,
		flow: Vec3::from_array(record.flow),
	}
}

/// One light, as the world holds it.
/// One entity's emitter, as a record, with its picture put in the blob.
///
/// @param index - the entity's place in the entity block
/// @param thing - the description it came off
/// @param names - the blob the picture's name is appended to
fn shed_of(index: usize, thing: &Thing, names: &mut Names) -> Result<Shed> {
	let emitter = thing.emitter;

	Ok(Shed {
		thing: count(index, "a scene's records")?,
		kind: emitter.kind.index(),
		blend: emitter.blend.index(),
		texture: names.put(&thing.emitter_texture),
		rate: emitter.rate,
		cap: emitter.cap,
		life: emitter.life,
		life_spread: emitter.life_spread,
		speed: emitter.speed,
		speed_spread: emitter.speed_spread,
		spread: emitter.spread,
		size: emitter.size,
		size_end: emitter.size_end,
		color: emitter.color.to_array(),
		color_end: emitter.color_end.to_array(),
		opacity: emitter.opacity,
		gravity: emitter.gravity,
		drag: emitter.drag,
	})
}

/// One entity, as a record, with its three names put in the blob.
///
/// Lifted out of [`encode`] rather than left inline, because that function
/// counts its lines and this is the longest of the six things it builds.
///
/// @param thing - the description it came off
/// @param names - the blob its name, mesh and material are appended to
fn stood_of(thing: &Thing, names: &mut Names) -> Stood {
	Stood {
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
		parent: thing.parent,
		flags: flags_of(thing),
	}
}

/// The word of flags one entity's record carries.
///
/// Each bit is the unusual answer, so an entity that is drawn and takes decals
/// writes nought.
///
/// @param thing - the description it came off
fn flags_of(thing: &Thing) -> u32 {
	let hidden = if thing.hidden { STOOD_HIDDEN } else { 0 };
	let undecaled = if thing.takes_decals { 0 } else { STOOD_UNDECALED };

	hidden | undecaled
}

/// One entity's terrain, as a record.
///
/// No blob at all, which is the one thing here that differs from
/// [`shed_of`]: a terrain names no asset. What it is made of is the entity's
/// own material, already written into its [`Stood`].
///
/// @param index - the entity's place in the entity block
/// @param thing - the description it came off
fn sod_of(index: usize, thing: &Thing) -> Result<Sod> {
	let terrain = thing.terrain;

	Ok(Sod {
		thing: count(index, "a scene's records")?,
		kind: terrain.kind.index(),
		size: terrain.size,
		height: terrain.height,
		side: terrain.side,
		seed: terrain.seed,
		frequency: terrain.frequency,
		octaves: terrain.octaves,
		roughness: terrain.roughness,
		tiling: terrain.tiling,
		solid: u32::from(terrain.solid),
	})
}

/// One terrain record, as a description holds it.
fn terrain_of(record: &Sod) -> Terrain {
	Terrain {
		kind: TerrainKind::at(record.kind).unwrap_or(TerrainKind::None),
		size: record.size,
		height: record.height,
		side: record.side,
		seed: record.seed,
		frequency: record.frequency,
		octaves: record.octaves,
		roughness: record.roughness,
		tiling: record.tiling,
		// anything but nought, for a flag bit's reason: a byte written by a
		// later version that means something more is still, today, a terrain
		// somebody can stand on.
		solid: record.solid != 0,
	}
}

/// One entity's decal, as a record.
///
/// No blob at all, for the terrain's reason: a decal names no asset.
///
/// @param index - the entity's place in the entity block
/// @param decal - what it paints
fn daub_of(index: usize, decal: Decal) -> Result<Daub> {
	Ok(Daub {
		thing: count(index, "a scene's records")?,
		kind: decal.kind.index(),
		fade: decal.fade,
		order: decal.order,
	})
}

/// One decal record, as a description holds it.
fn decal_of(record: &Daub) -> Decal {
	Decal {
		kind: DecalKind::at(record.kind).unwrap_or(DecalKind::None),
		fade: record.fade,
		order: record.order,
	}
}

/// One emitter record, as a description holds it.
///
/// The picture is left to the caller, which is the one thing here that needs
/// the blob.
fn emitter_of(record: &Shed) -> Emitter {
	Emitter {
		kind: EmitterKind::at(record.kind).unwrap_or(EmitterKind::None),
		blend: SparkBlend::at(record.blend).unwrap_or_default(),
		// a description never handles an asset; the name goes on beside it.
		texture: TextureId::NONE,
		rate: record.rate,
		cap: record.cap,
		life: record.life,
		life_spread: record.life_spread,
		speed: record.speed,
		speed_spread: record.speed_spread,
		spread: record.spread,
		size: record.size,
		size_end: record.size_end,
		color: Vec3::from_array(record.color),
		color_end: Vec3::from_array(record.color_end),
		opacity: record.opacity,
		gravity: record.gravity,
		drag: record.drag,
	}
}

fn light_of(record: &Lit) -> Light {
	Light {
		kind: LightKind::at(record.kind).unwrap_or(LightKind::None),
		color: Vec3::from_array(record.color),
		intensity: record.intensity,
		range: record.range,
		inner: record.inner,
		outer: record.outer,
		shadow: (record.flags & LIT_UNSHADOWED) == 0,
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
		(header.lit_stride, size_of::<Lit>(), "lights"),
		(header.shed_stride, size_of::<Shed>(), "emitters"),
		(header.sod_stride, size_of::<Sod>(), "terrains"),
		(header.daub_stride, size_of::<Daub>(), "decals"),
		(header.jot_stride, size_of::<Jot>(), "record values"),
		(header.bulk_stride, size_of::<Bulk>(), "bodies"),
		(header.wet_stride, size_of::<Wet>(), "waters"),
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
	fits::<Lit>(bytes, HEADER_BYTES, (header.lit_offset, header.lit_count), "lights")?;
	fits::<Shed>(bytes, HEADER_BYTES, (header.shed_offset, header.shed_count), "emitters")?;
	fits::<Sod>(bytes, HEADER_BYTES, (header.sod_offset, header.sod_count), "terrains")?;
	fits::<Daub>(bytes, HEADER_BYTES, (header.daub_offset, header.daub_count), "decals")?;
	fits::<Jot>(bytes, HEADER_BYTES, (header.jot_offset, header.jot_count), "record values")?;
	fits::<Bulk>(bytes, HEADER_BYTES, (header.bulk_offset, header.bulk_count), "bodies")?;
	fits::<Wet>(bytes, HEADER_BYTES, (header.wet_offset, header.wet_count), "waters")?;
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
				light: Light::point(Vec3::new(1.0, 0.9, 0.7), 2.5, 12.0),
				emitter: Emitter::NONE,
				emitter_texture: String::new(),
				terrain: Terrain::NONE,
				pose: 0,
				parent: scene::NO_INDEX,
				// this one hidden and the second not, so a writer that put one
				// answer on every record comes back unequal to the fixture
				hidden: true,
				// and refusing decals, so its word of flags carries both bits
				takes_decals: false,
				decal: Decal::NONE,
				records: Vec::new(),
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
				// on the second one on purpose: its slot is 2 and its place in
				// the block is 1, so a writer keying a light by slot rather
				// than by index puts this record on nobody.
				light: Light::spot(Vec3::new(0.2, 0.4, 1.0), 4.0, 30.0, 0.3, 0.6),
				// on the second one for the light's reason, and with every
				// number moved off its default so a round trip that dropped a
				// field is a failing test rather than a coincidence
				emitter: Emitter {
					blend: SparkBlend::Alpha,
					rate: 40.5,
					cap: 96,
					life: 2.25,
					life_spread: 0.4,
					speed: 3.5,
					speed_spread: 0.6,
					spread: 0.35,
					size: 0.15,
					size_end: 0.85,
					color: Vec3::new(1.0, 0.5, 0.25),
					color_end: Vec3::new(0.1, 0.1, 0.4),
					opacity: 0.75,
					gravity: 0.5,
					drag: 0.2,
					..Emitter::cone(40.5, 2.25, 0.35)
				},
				emitter_texture: "textures/smoke".to_owned(),
				// on the second one for the light's reason, and with every
				// number moved off its default for the emitter's
				terrain: Terrain {
					size: 96.5,
					height: 12.25,
					side: 65,
					seed: 7,
					frequency: 5.5,
					octaves: 3,
					roughness: 0.35,
					tiling: 4.5,
					solid: false,
					..Terrain::hills()
				},
				pose: scene::NO_INDEX,
				// hanging off the first, so the field carries something a
				// round trip could lose
				parent: 0,
				hidden: false,
				takes_decals: true,
				// on the second one for the light's reason, with both numbers off
				// their defaults and the order below nought
				decal: Decal { fade: 0.25, order: -7, ..Decal::BOX },
				// on the second one for the light's reason, one of every spelling
				// and two records, with a whole number no single-precision number
				// holds, so a value read at the wrong width comes back wrong
				records: sample_records(),
			},
		]
	}

	/// One value of every spelling, over two records.
	fn sample_records() -> Vec<Noted> {
		let noted = |record: &str, field: &str, value: Spelled| Noted {
			record: record.to_owned(),
			field: field.to_owned(),
			value,
		};

		vec![
			noted("drawing", "covers", Spelled::Truth(true)),
			noted("door", "count", Spelled::Number(16_777_217.0)),
			noted("door", "mark", Spelled::Numbers(vec![0.5, -0.25])),
			noted("door", "hinge", Spelled::Numbers(vec![1.0, 2.0, 3.0])),
			noted("door", "lean", Spelled::Numbers(vec![0.0, 0.0, 0.0, 1.0])),
			noted("door", "style", Spelled::Word("slide".to_owned())),
			noted("door", "open", Spelled::Truth(false)),
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
				water: Water::NONE,
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
				// on the second one for the reason the light is on the second
				// entity: its slot is 4 and its place in the block is 1, so a
				// writer keying water by slot rather than by index puts this
				// record on nobody. Every number is off its default, so a
				// round trip that dropped any of them would show it.
				water: Water {
					kind: WaterKind::Volume,
					density: 3.5,
					damp: 6.25,
					linear_drag: 0.44,
					angular_drag: 0.11,
					flow: Vec3::new(0.0, 0.0, -2.0),
				},
				thing: scene::NO_INDEX,
			},
		]
	}

	/// A description with one of everything a record can hold.
	fn sample() -> SceneData {
		SceneData {
			things: sample_things(),
			solids: sample_solids(),
			sky_cubemap: String::new(),
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
					focus: 7.5,
					focus_range: 22.0,
					blur: 11.0,
				},
				clear: Vec3::new(0.1, 0.2, 0.3),
				// drawn, and three colors none of which is a default, so the
				// round trip carries something it could lose
				sky: Sky::gradient(
					Vec3::new(0.05, 0.15, 0.45),
					Vec3::new(0.60, 0.70, 0.85),
					Vec3::new(0.08, 0.07, 0.06),
				),
				// every number moved off its default, so a field the writer
				// forgot would come back wrong rather than come back right by
				// accident
				post: Post {
					tonemap: ToneMap::Reinhard,
					white: 6.5,
					auto_exposure: false,
					exposure: 1.25,
					exposure_bias: -0.75,
					exposure_min: 0.02,
					exposure_max: 12.0,
					exposure_rate: 3.5,
					bloom: 0.4,
					bloom_threshold: 1.6,
					fog: Vec3::new(0.31, 0.42, 0.53),
					fog_density: 0.02,
					shafts: 0.7,
					haze: 0.035,
				},
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
	fn whether_an_entity_is_hidden_survives_both_ways() {
		for hidden in [false, true] {
			let mut data = sample();
			data.things[1].hidden = hidden;

			assert_eq!(
				round_trip(&data).things[1].hidden,
				hidden,
				"an entity that says {hidden} comes back saying it"
			);
		}
	}

	#[test]
	fn an_entity_that_is_drawn_is_what_a_record_of_no_flags_is() {
		let bytes = encode(&sample()).expect("it fits");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("readable");

		assert_eq!(
			file.stood()[0].flags,
			STOOD_HIDDEN | STOOD_UNDECALED,
			"the hidden one that refuses decals sets both bits"
		);
		assert_eq!(file.stood()[1].flags, 0, "and the one that is drawn writes nothing");
	}

	/// The bytes of the sample with its first entity's flags overwritten.
	fn stood_flags_of(word: u32) -> Vec<u8> {
		let mut bytes = encode(&sample()).expect("it fits");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("readable");
		let at = usize::try_from(file.header().stood_offset).expect("an offset")
			+ core::mem::offset_of!(Stood, flags);

		bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());

		bytes
	}

	#[test]
	fn a_flag_this_build_does_not_know_is_a_property_the_entity_does_not_have() {
		// the flag rule rather than the code rule, @ref `codes`: a bit from a
		// later build is read, not refused, and it does not leak into the one
		// this build knows
		let hidden = |word: u32| {
			SceneFile::from_bytes(AlignedBytes::from_slice(&stood_flags_of(word)))
				.expect("a flag nothing answers to is read rather than refused")
				.to_scene_data()
				.things[0]
				.hidden
		};

		assert!(!hidden(1 << 20), "a bit from a later build hides nothing");
		assert!(hidden((1 << 20) | STOOD_HIDDEN), "and covers nothing the known bit says");
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

	/// The sample written out, with one word of its first light record
	/// overwritten.
	///
	/// @param at - the record's own field offset, in bytes
	/// @param word - what to put there
	fn light_word_changed(at: usize, word: u32) -> SceneData {
		let data = sample();
		let mut bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);
		let first = usize::try_from(header.lit_offset).expect("it is an offset") + at;

		assert_eq!(header.lit_count, 2, "the sample stands two lamps");
		bytes[first..first + 4].copy_from_slice(&word.to_le_bytes());

		SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect("a changed word is not a broken file")
			.to_scene_data()
	}

	#[test]
	fn a_sky_of_a_kind_this_build_does_not_know_reads_as_no_sky() {
		let data = sample();
		let mut bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);
		let at = usize::try_from(header.setting_offset).expect("it is an offset")
			+ offset_of!(Setting, sky_kind);

		bytes[at..at + 4].copy_from_slice(&9_u32.to_le_bytes());

		let read = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect("a changed word is not a broken file")
			.to_scene_data();

		assert_eq!(
			read.stage.sky.kind,
			SkyKind::None,
			"a word off the end of the list is nothing rather than a refusal"
		);
		assert_eq!(
			read.stage.sky.zenith, data.stage.sky.zenith,
			"and the colors beside it are still read"
		);
	}

	#[test]
	fn the_settings_record_ends_in_a_spare_word_again_after_the_haze() {
		// the claim the version bump was paid for, collected: the name that
		// spent the last spare is where it was, the haze is the word after it,
		// and the record ends in a new spare holding nought. @ref
		// [`Setting::spare`].
		let data = SceneData {
			sky_cubemap: "skies/dusk".to_owned(),
			stage: Stage {
				sky: Sky::environment(TextureId::NONE),
				..sample().stage
			},
			..sample()
		};
		let bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);
		let start = usize::try_from(header.setting_offset).expect("it is an offset");
		let spare = start + offset_of!(Setting, spare);

		assert_eq!(
			offset_of!(Setting, haze),
			offset_of!(Setting, sky_cubemap) + 4,
			"the haze is the word after the name"
		);
		assert_eq!(
			spare + 4,
			start + size_of::<Setting>(),
			"and the spare is the last word of the record"
		);
		assert_eq!(size_of::<Setting>(), 232, "which is two words longer than it was");
		assert_eq!(
			u32::from_le_bytes(
				<[u8; 4]>::try_from(&bytes[spare..spare + 4]).expect("four bytes of a word")
			),
			0,
			"and a written file holds nought there, which is what makes the word free"
		);

		let read = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect("a scene naming a sky")
			.to_scene_data();

		assert_eq!(read.sky_cubemap, "skies/dusk", "the name came back");
		assert_eq!(read.stage.sky.kind, SkyKind::Cubemap, "and so did the word beside it");
		assert!(
			(read.stage.post.haze - data.stage.post.haze).abs() < 1.0e-9,
			"and so did the haze"
		);
		assert!(
			(read.stage.camera.blur - data.stage.camera.blur).abs() < 1.0e-9,
			"and the field before the name is still where it was"
		);
	}

	#[test]
	fn a_scene_that_names_no_sky_puts_nought_in_the_word_an_older_build_ignores() {
		// the other half of the claim: a file from before this word meant
		// anything holds nought there, and nought is the empty name.
		let bytes = encode(&sample()).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);
		let at = usize::try_from(header.setting_offset).expect("it is an offset")
			+ offset_of!(Setting, sky_cubemap);
		let word = u32::from_le_bytes(
			<[u8; 4]>::try_from(&bytes[at..at + 4]).expect("four bytes of a word"),
		);

		assert_eq!(word, 0, "which is where the blob's empty name lives");
		assert_eq!(
			SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
				.expect("a scene naming no sky")
				.to_scene_data()
				.sky_cubemap,
			"",
			"and it reads back as no name"
		);
	}

	#[test]
	fn a_light_naming_an_entity_that_is_not_there_is_dropped() {
		let read = light_word_changed(offset_of!(Lit, thing), 99);

		assert_eq!(read.things[0].light, Light::NONE, "the record went nowhere");
		assert_eq!(
			read.things[1].light,
			sample_things()[1].light,
			"and the one beside it landed as it always did"
		);
	}

	#[test]
	fn a_light_of_a_kind_this_build_does_not_know_reads_as_no_light() {
		let read = light_word_changed(offset_of!(Lit, kind), 9);

		assert_eq!(
			read.things[0].light.kind,
			LightKind::None,
			"a word off the end of the list is nothing rather than a refusal"
		);
		assert!(
			(read.things[0].light.range - sample_things()[0].light.range).abs() < 1.0e-6,
			"and the numbers beside it are still read"
		);
	}

	#[test]
	fn a_world_of_a_thousand_crates_and_no_lamp_writes_no_light_block() {
		let mut data = sample();
		for thing in &mut data.things {
			thing.light = Light::NONE;
		}

		let bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);

		assert_eq!(header.lit_count, 0, "nothing shines, so nothing is written down");
		assert_eq!(round_trip(&data), data, "and it comes back the same way");
	}

	/// The sample written out, with one word of its first emitter record
	/// overwritten.
	///
	/// @param at - the record's own field offset, in bytes
	/// @param word - what to put there
	fn shed_word_changed(at: usize, word: u32) -> SceneData {
		let data = sample();
		let mut bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);
		let first = usize::try_from(header.shed_offset).expect("it is an offset") + at;

		assert_eq!(header.shed_count, 1, "the sample carries one emitter");
		bytes[first..first + 4].copy_from_slice(&word.to_le_bytes());

		SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect("a changed word is not a broken file")
			.to_scene_data()
	}

	#[test]
	fn an_emitter_is_written_against_its_place_in_the_entity_block_and_not_its_slot() {
		// the sample's emitter is on the second entity, whose slot is 2 and
		// whose place in the block is 1. A writer keying by slot puts the
		// record on nobody; a reader keying by slot reads it onto nobody. The
		// same trap the light block has.
		let data = sample();
		let bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("readable");

		assert_eq!(header.shed_count, 1, "one entity throws anything");
		assert_eq!(file.shed()[0].thing, 1, "and it is the second entry, not slot two");
		assert_eq!(
			round_trip(&data).things[1].emitter,
			data.things[1].emitter,
			"and every number of it comes back"
		);
		assert_eq!(
			round_trip(&data).things[1].emitter_texture,
			"textures/smoke",
			"picture and all"
		);
	}

	/// The sample written out, with one word of its first terrain record
	/// overwritten.
	///
	/// @param at - the record's own field offset, in bytes
	/// @param word - what to put there
	fn sod_word_changed(at: usize, word: u32) -> SceneData {
		let data = sample();
		let mut bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);
		let first = usize::try_from(header.sod_offset).expect("it is an offset") + at;

		assert_eq!(header.sod_count, 1, "the sample carries one terrain");
		bytes[first..first + 4].copy_from_slice(&word.to_le_bytes());

		SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect("a changed word is not a broken file")
			.to_scene_data()
	}

	/// The sample written out, with one word of its first decal record
	/// overwritten.
	///
	/// @param at - the record's own field offset, in bytes
	/// @param word - what to put there
	fn daub_word_changed(at: usize, word: u32) -> SceneData {
		let data = sample();
		let mut bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);
		let first = usize::try_from(header.daub_offset).expect("it is an offset") + at;

		assert_eq!(header.daub_count, 1, "the sample carries one decal");
		bytes[first..first + 4].copy_from_slice(&word.to_le_bytes());

		SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect("a changed word is not a broken file")
			.to_scene_data()
	}

	#[test]
	fn a_decal_is_written_against_its_place_in_the_entity_block_and_not_its_slot() {
		// the trap every block keyed by an entity has: the sample's decal is on
		// the second entity, whose slot is two and whose place in the block is
		// one
		let data = sample();
		let bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("readable");

		assert_eq!(header.daub_count, 1, "one entity paints");
		assert_eq!(file.daub()[0].thing, 1, "and it is the second entry, not slot two");
		assert_eq!(
			round_trip(&data).things[1].decal,
			data.things[1].decal,
			"and every number of it comes back"
		);
	}

	#[test]
	fn a_world_that_paints_nothing_writes_no_decal_block_at_all() {
		let mut data = sample();
		for thing in &mut data.things {
			thing.decal = Decal::NONE;
		}

		let bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);

		assert_eq!(header.daub_count, 0, "nothing paints, so nothing is written down");
		assert_eq!(round_trip(&data), data, "and it comes back the same way");
	}

	#[test]
	fn a_decal_naming_an_entity_that_is_not_there_is_dropped() {
		let read = daub_word_changed(offset_of!(Daub, thing), 99);

		assert_eq!(read.things[1].decal, Decal::NONE, "the record went nowhere");
	}

	#[test]
	fn a_decal_of_a_kind_this_build_does_not_know_reads_as_no_decal() {
		let read = daub_word_changed(offset_of!(Daub, kind), 9);

		assert_eq!(
			read.things[1].decal.kind,
			DecalKind::None,
			"a word this build has no spelling for paints nothing"
		);
		assert_eq!(
			read.things[1].decal.order, -7,
			"and the numbers beside it are still read, the way a light's are"
		);
	}

	/// The sample written out, with one word of its first record value
	/// overwritten.
	///
	/// @param at - the record's own field offset, in bytes
	/// @param word - what to put there
	fn jot_word_changed(at: usize, word: u32) -> SceneData {
		let data = sample();
		let mut bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);
		let first = usize::try_from(header.jot_offset).expect("it is an offset") + at;

		assert_eq!(header.jot_count, 7, "the sample carries seven record values");
		bytes[first..first + 4].copy_from_slice(&word.to_le_bytes());

		SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect("a changed word is not a broken file")
			.to_scene_data()
	}

	#[test]
	fn a_record_value_is_written_against_its_place_in_the_entity_block_and_comes_back_exactly() {
		// the second entity again: its slot is two and its place in the block one
		let data = sample();
		let bytes = encode(&data).expect("it fits in one file");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("readable");

		assert!(file.jot().iter().all(|jot| jot.thing == 1), "every value is the second entry's");
		assert_eq!(
			file.jot()
				.iter()
				.map(|jot| jot.spelling)
				.collect::<Vec<_>>(),
			vec![JOT_TRUTH, JOT_NUMBER, 2, 3, 4, JOT_WORD, JOT_TRUTH],
			"one of every spelling"
		);
		assert_eq!(
			round_trip(&data).things[1].records,
			sample_records(),
			"and every value comes back, in its order, the whole number to the unit"
		);
	}

	#[test]
	fn a_world_whose_records_are_at_their_defaults_writes_no_record_values() {
		let mut data = sample();
		for thing in &mut data.things {
			thing.records.clear();
		}

		let bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);

		assert_eq!(header.jot_count, 0, "nothing to write down, so nothing is");
		assert_eq!(header.jot_stride, 32, "and the stride is said all the same");
		assert_eq!(round_trip(&data), data, "and it comes back the same way");
	}

	#[test]
	fn a_record_value_naming_an_entity_that_is_not_there_is_dropped() {
		let read = jot_word_changed(offset_of!(Jot, thing), 99);

		assert_eq!(read.things[1].records.len(), 6, "the first value went nowhere");
		assert_eq!(read.things[1].records[..], sample_records()[1..], "and the rest are there");
	}

	#[test]
	fn a_record_value_spelled_in_a_way_this_build_does_not_know_reads_as_nothing() {
		let read = jot_word_changed(offset_of!(Jot, spelling), 9);

		assert_eq!(read.things[1].records[..], sample_records()[1..], "only that value is lost");

		let unnamed = jot_word_changed(offset_of!(Jot, field), 0);

		assert_eq!(
			unnamed.things[1].records[..],
			sample_records()[1..],
			"and a value naming no field is no value either"
		);
	}

	#[test]
	fn a_record_value_of_numbers_no_record_holds_is_refused_rather_than_written() {
		// none and one as well as five: a count is the spelling, so none would be
		// written as a flag and one as half of a double
		for many in [0, 1, 5] {
			let mut data = sample();
			data.things[0].records = vec![Noted {
				record: "door".to_owned(),
				field: "far".to_owned(),
				value: Spelled::Numbers(vec![1.0; many]),
			}];

			let refused = encode(&data)
				.expect_err("that many numbers are not a value")
				.to_string();

			assert!(
				refused.contains("door.far") && refused.contains("two to four"),
				"{many} numbers, got {refused}"
			);
		}
	}

	#[test]
	fn whether_an_entity_takes_decals_survives_both_ways() {
		for takes in [false, true] {
			let mut data = sample();
			data.things[1].takes_decals = takes;

			assert_eq!(
				round_trip(&data).things[1].takes_decals,
				takes,
				"an entity that says {takes} comes back saying it"
			);
		}
	}

	#[test]
	fn the_word_against_decals_is_a_bit_of_its_own_and_neither_bit_leaks_into_the_other() {
		let read = |word: u32| {
			let first = SceneFile::from_bytes(AlignedBytes::from_slice(&stood_flags_of(word)))
				.expect("a flag nothing answers to is read rather than refused")
				.to_scene_data()
				.things
				.into_iter()
				.next()
				.expect("the sample stands two");

			(first.hidden, first.takes_decals)
		};

		assert_eq!(read(0), (false, true), "no flags: drawn, and painted by decals");
		assert_eq!(read(STOOD_HIDDEN), (true, true), "hidden says nothing about decals");
		assert_eq!(
			read(STOOD_UNDECALED),
			(false, false),
			"and the word against them hides nothing"
		);
		assert_eq!(read(1 << 20), (false, true), "and a bit from a later build is neither");
	}

	#[test]
	fn a_terrain_is_written_against_its_place_in_the_entity_block_and_not_its_slot() {
		// the same trap the light and the emitter blocks both have: the
		// sample's ground is on the second entity, whose slot is two and whose
		// place in the block is one.
		let data = sample();
		let bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("readable");

		assert_eq!(header.sod_count, 1, "one entity is ground");
		assert_eq!(file.sod()[0].thing, 1, "and it is the second entry, not slot two");
		assert_eq!(
			round_trip(&data).things[1].terrain,
			data.things[1].terrain,
			"and every number of it comes back"
		);
	}

	#[test]
	fn a_world_with_no_ground_writes_no_terrain_block_at_all() {
		let mut data = sample();
		for thing in &mut data.things {
			thing.terrain = Terrain::NONE;
		}

		let bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);

		assert_eq!(header.sod_count, 0, "nothing is ground, so nothing is written down");
		assert_eq!(round_trip(&data), data, "and it comes back the same way");
	}

	#[test]
	fn a_terrain_naming_an_entity_that_is_not_there_is_dropped() {
		let read = sod_word_changed(offset_of!(Sod, thing), 99);

		assert_eq!(read.things[1].terrain, Terrain::NONE, "the record went nowhere");
	}

	#[test]
	fn a_terrain_of_a_kind_this_build_does_not_know_reads_as_no_terrain() {
		let read = sod_word_changed(offset_of!(Sod, kind), 9);

		assert_eq!(
			read.things[1].terrain.kind,
			TerrainKind::None,
			"a word this build has no spelling for is no ground"
		);
		assert!(
			(read.things[1].terrain.height - 12.25).abs() < 1.0e-4,
			"and the numbers beside it are still read, the way a light's are"
		);
	}

	#[test]
	fn anything_but_nought_in_the_solid_word_is_ground_to_stand_on() {
		// a flag's rule rather than a code's: a later version writing something
		// more into this word still means a terrain somebody can stand on.
		let read = sod_word_changed(offset_of!(Sod, solid), 7);

		assert!(read.things[1].terrain.solid, "seven is not nought");
	}

	#[test]
	fn a_terrain_carries_none_of_its_geometry_into_the_file() {
		// the whole argument of the card, as a number: a default terrain is
		// thirty-two thousand triangles and the file it is written into is
		// nine kilobytes.
		let mut data = sample();
		data.things[1].terrain = Terrain::hills();

		let bytes = encode(&data).expect("it fits in one file");

		assert!(data.things[1].terrain.triangles() > 30_000, "the terrain really is a big one");
		assert!(bytes.len() < 64 * 1024, "and the file is {} bytes", bytes.len());
	}

	#[test]
	fn an_emitter_naming_an_entity_that_is_not_there_is_dropped() {
		let read = shed_word_changed(offset_of!(Shed, thing), 99);

		assert_eq!(read.things[1].emitter, Emitter::NONE, "the record went nowhere");
		assert!(read.things[1].emitter_texture.is_empty(), "and neither did its picture");
	}

	#[test]
	fn an_emitter_of_a_kind_this_build_does_not_know_reads_as_no_emitter() {
		let read = shed_word_changed(offset_of!(Shed, kind), 9);

		assert_eq!(
			read.things[1].emitter.kind,
			EmitterKind::None,
			"a word off the end of the list is nothing rather than a refusal"
		);
		assert!(
			(read.things[1].emitter.rate - sample_things()[1].emitter.rate).abs() < 1.0e-6,
			"and the numbers beside it are still read"
		);
	}

	#[test]
	fn a_world_that_throws_nothing_writes_no_emitter_block() {
		let mut data = sample();
		for thing in &mut data.things {
			thing.emitter = Emitter::NONE;
			thing.emitter_texture = String::new();
		}

		let bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);

		assert_eq!(header.shed_count, 0, "nothing throws, so nothing is written down");
		assert_eq!(round_trip(&data), data, "and it comes back the same way");
	}

	#[test]
	fn the_header_stays_a_multiple_of_sixteen_and_says_where_every_block_is() {
		// the whole reason the spare words exist. A header that is not a
		// multiple of sixteen moves every block after it by a byte or two and
		// nothing notices until a cast fails on a machine that cares.
		assert_eq!(HEADER_BYTES % 16, 0, "the blocks after it inherit its alignment");
		assert_eq!(size_of::<SceneHeader>(), HEADER_BYTES);

		let bytes = encode(&sample()).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);

		assert!(header.daub_offset > header.sod_offset, "the decals after the ground");
		assert!(header.jot_offset > header.daub_offset, "the record values after the decals");
		assert!(header.bulk_offset > header.jot_offset, "and the bodies after the record values");
		assert_eq!(header.spare, 0, "with a spare word holding nought");
		assert!(
			header.shed_offset > header.lit_offset,
			"the emitters are written after the lights"
		);
		assert!(header.bulk_offset > header.shed_offset, "and the bodies after the emitters");
	}

	#[test]
	fn a_pool_is_written_against_its_place_in_the_body_block_and_not_its_slot() {
		let data = sample();
		let bytes = encode(&data).expect("it fits in one file");
		let file = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("readable");

		assert_eq!(file.wet().len(), 1, "one of the two bodies holds a fluid");
		assert_eq!(
			file.wet()[0].body,
			1,
			"and it is named by its place in the block, not by its slot of four"
		);
		assert_eq!(
			data.solids[1].water,
			round_trip(&data).solids[1].water,
			"every number of it comes back"
		);
	}

	#[test]
	fn water_of_a_kind_this_build_does_not_know_reads_as_no_water() {
		let data = sample();
		let mut bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);
		let at =
			usize::try_from(header.wet_offset).expect("it is an offset") + offset_of!(Wet, kind);

		bytes[at..at + 4].copy_from_slice(&9_u32.to_le_bytes());

		let read = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect("a changed word is not a broken file")
			.to_scene_data();

		assert_eq!(
			read.solids[1].water.kind,
			WaterKind::None,
			"a word off the end of the list is nothing rather than a refusal"
		);
		assert!(
			(read.solids[1].water.density - sample_solids()[1].water.density).abs() < 1.0e-6,
			"and the numbers beside it are still read"
		);
	}

	#[test]
	fn a_pool_named_at_a_body_that_is_not_there_is_dropped_rather_than_refused() {
		let data = sample();
		let mut bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);
		let at =
			usize::try_from(header.wet_offset).expect("it is an offset") + offset_of!(Wet, body);

		bytes[at..at + 4].copy_from_slice(&99_u32.to_le_bytes());

		let read = SceneFile::from_bytes(AlignedBytes::from_slice(&bytes))
			.expect("a changed word is not a broken file")
			.to_scene_data();

		assert!(
			read.solids
				.iter()
				.all(|solid| !solid.water.is_wet()),
			"a world short one pool is a better answer than a load that did not happen"
		);
	}

	#[test]
	fn a_world_of_bodies_and_no_pool_writes_no_water_block() {
		let mut data = sample();
		for solid in &mut data.solids {
			solid.water = Water::NONE;
		}

		let bytes = encode(&data).expect("it fits in one file");
		let header: SceneHeader = *bytemuck::from_bytes(&bytes[..HEADER_BYTES]);

		assert_eq!(header.wet_count, 0, "nothing is wet, so nothing is written down");
		assert_eq!(round_trip(&data), data, "and it comes back the same way");
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
	fn a_record_value_of_the_wrong_width_is_refused_like_every_other_record() {
		let refused = patched(offset_of!(SceneHeader, jot_stride), 16);

		assert!(refused.contains("record values are 16 bytes each"), "got {refused}");
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
			(offset_of!(SceneHeader, lit_offset), "lights"),
			(offset_of!(SceneHeader, shed_offset), "emitters"),
			(offset_of!(SceneHeader, sod_offset), "terrains"),
			(offset_of!(SceneHeader, daub_offset), "decals"),
			(offset_of!(SceneHeader, jot_offset), "record values"),
			(offset_of!(SceneHeader, bulk_offset), "bodies"),
			(offset_of!(SceneHeader, wet_offset), "waters"),
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
