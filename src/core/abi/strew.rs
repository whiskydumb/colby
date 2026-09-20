//! Many copies of one mesh, laid over the ground an entity hangs off.
//!
//! **An entity whose [`Strewing`] record says it strews does not draw its mesh
//! where it stands.** Its mesh is laid down, copy after copy, over the
//! triangles of the entity it hangs off - a terrain, a floor, any mesh at all -
//! by the rule its record holds: how many copies to a square unit, how steep a
//! ground they stand on, how big, how turned, how sunk. Every copy is worked
//! out again from that rule whenever the rule, the ground or the mesh changes,
//! the way a terrain's geometry is worked out again from its description.
//!
//! **A rule and not a store, and nothing a copy is gets written down.** Not in
//! a save, not in a scene source, not on the wire: what is written is the
//! record, a few dozen numbers however many copies it lays. The copies are in
//! [`Strewn`], a table on the world that nothing saves, beside the mesh
//! registry a terrain's geometry lands in. Two reasons, one about the field and
//! one about this engine. Of the engines read for this, every one that has a
//! dense layer at all works it out on the fly and keeps no copy of it; what
//! they keep copy by copy is the deliberate layer - a tree here, a rock there -
//! and here that is what an entity is for. And the editor's step back is a copy
//! of the world, two a step and sixty-four steps deep, so a hundred thousand
//! copies written into the world would be in every one of them.
//!
//! **Over the ground's own triangles**, in the ground's own space, so the
//! copies move, turn and grow with it. A triangle is laid with as many copies
//! as the density times the area it covers seen from above, the part of one
//! left over settled by a draw; a triangle facing down or edge on covers none,
//! and one steeper than the rule's slope is passed over whole.
//!
//! **One stream a triangle**, seeded with the rule's seed and the triangle's
//! place in the mesh and nothing else. So a copy is where it is because of its
//! own triangle: laying the ground's other triangles differently moves none of
//! it. And **every number a copy needs is drawn, whether or not it is kept** -
//! where on the triangle, the turn, the lean, the size, the sinking, the shade,
//! its place in its patch and the draw a [`Mask`] is read against - so that
//! raising the density adds copies after the ones there were, and a band, a
//! size, a lean, a slope or a stroke of a brush takes away or reshapes the
//! copies it is about and moves no other.
//!
//! **Where a copy may stand is painted**, @ref [`Mask`]: a grid over the ground
//! seen from above, one cell holding what share of the copies over it stand.
//! A strewing nobody has painted has no mask and lays its whole field, and so
//! does one whose cells are all open - the same copies, byte for byte, which is
//! what the draw above is for.
//!
//! **The same copies on every machine.** From the rule to a copy there is
//! nothing but adding, multiplying, dividing and square roots, which every
//! machine answers alike: a turn is a point on the unit circle found by drawing
//! ([`Random::circle`]), never a sine. The rule's two angles, the slope and the
//! lean, are turned into cosines once, in double precision, and narrowed - the
//! arrangement the bake keeps for its own few angles. [`Laid::digest`] is what
//! says the result is the same.
//!
//! **In patches.** The copies are kept in runs, one run to each square of the
//! ground [`PATCH`] units across, each with a box around every copy in it: what
//! a renderer asks whether it can see, how far away it is and which level of
//! the mesh to draw it at, a few hundred times a frame instead of a hundred
//! thousand. Inside a run the copies are in an order drawn at random, so that
//! the first part of a run is an even thinning of all of it.

use core::ops::Range;

use super::{
	entity::{Entities, EntityId, Transform},
	mesh::{MeshData, MeshId, MeshVertex},
	record::Record,
};
use crate::{
	bytemuck::{self, Pod, Zeroable},
	glam::{Mat3, Quat, Vec2, Vec3},
	random::{Random, threshold},
};

/// How wide one patch of the ground is, in the ground's own units.
///
/// Eight, which puts a hundred and twenty-eight units of ground in two hundred
/// and fifty-six patches: few enough that asking each of them every question a
/// frame asks costs tens of microseconds, and small enough that a patch's
/// nearest copy and its furthest are drawn at about the same level of detail.
pub const PATCH: f32 = 8.0;

/// The most copies one strewing lays.
///
/// Two to the eighteenth, the probes' own ceiling: about eight megabytes of
/// copies here and thirty-two of what a renderer makes of them. A rule that
/// would lay more is laid at a lower density that fits, and says so. @ref
/// [`Laid::thinned`].
pub const MOST: usize = 262_144;

/// How many copies stand on a square unit of ground unless the record says
/// otherwise.
pub const DENSITY: f32 = 1.0;

/// The steepest ground a copy stands on unless the record says otherwise, in
/// degrees from level: any ground that faces up at all.
pub const SLOPE: f32 = 90.0;

/// How far above and below the ground's origin a copy may stand by default.
///
/// Further than any ground this engine builds reaches, so that the band says
/// nothing until somebody narrows it.
pub const BAND: f32 = 10_000.0;

/// What share of its reach, at the far end, a strewing thins out over unless
/// the record says otherwise.
pub const FADE: f32 = 0.25;

/// The furthest a copy leans at random, in degrees: lying flat.
pub const MAX_TILT: f32 = 90.0;

/// The most cells one [`Mask`] holds.
///
/// Sixteen thousand three hundred and eighty-four, which is sixteen kilobytes
/// of it. Three things are measured against this number and it is the smallest
/// of the three: a mask is written into every step back the editor keeps, it
/// travels inside the piece of a world that crosses the wire - where a whole
/// message is seventy-five kilobytes - and it is in the scene file. A ground
/// wider than the mask is fine grained is laid out with wider cells, @ref
/// [`Mask::over`].
pub const MOST_CELLS: usize = 1 << 14;

/// How wide one cell of a [`Mask`] is over a small ground, in the ground's own
/// units.
///
/// One, so that a mask over a ground a person can see the whole of is about as
/// fine as the brush that paints it. A wider ground doubles it until the cells
/// fit [`MOST_CELLS`].
pub const CELL: f32 = 1.0;

/// What a cell holds where nothing has been painted away: every copy the rule
/// would lay there stands.
pub const OPEN: u8 = u8::MAX;

/// How the copies are strewn, and whether they are at all.
///
/// **The engine's fourth record**, for [`Drawing`](super::Drawing)'s reason: a
/// rule every entity may carry, which an inspector, a scene source, a save, a
/// copy and a piece of the world on the wire already reach through the path a
/// game's fields take - and a new field on it moves no file's version.
///
/// What is strewn is the entity's own mesh in its own material, held the way
/// the entity's own turn and size hold it; where it is strewn is over the mesh
/// of the entity it hangs off. The entity's own place changes nothing about the
/// copies.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Strewing {
	/// Whether the entity's mesh is strewn over the ground it hangs off rather
	/// than drawn once where it stands: nought for no, anything else for yes.
	pub strews: u32,

	/// What the copies are laid out from: two rules alike but for this lay two
	/// fields that have nothing to do with each other.
	pub seed: i32,

	/// How many copies stand on a square unit of ground, seen from above.
	pub density: f32,

	/// The steepest ground a copy stands on, in degrees from level.
	pub slope: f32,

	/// The lowest and the highest ground a copy stands on, in the ground's own
	/// space.
	pub band: [f32; 2],

	/// The smallest and the largest a copy is, as a share of how the entity
	/// holds one.
	pub size: [f32; 2],

	/// The least and the most a copy is sunk into the ground, along its own up.
	pub sink: [f32; 2],

	/// Whether each copy is turned at random about its own up: nought for no.
	pub turns: u32,

	/// How far a copy leans with the ground under it: nought stands it upright,
	/// one along the face it stands on.
	pub align: f32,

	/// The most a copy leans at random, in degrees.
	pub tilt: f32,

	/// How much darker than the entity's own color a copy may be: nought all
	/// alike, one down to black.
	pub shade: f32,

	/// How far from the eye a copy is drawn, nought for however far.
	pub reach: f32,

	/// What share of its reach, at the far end, the copies thin out over.
	pub fade: f32,

	/// Whether the copies throw shadows: nought for no.
	pub shadows: u32,

	/// Whether anything collides with them: one body made of every copy's
	/// coarsest level. Nought for no.
	pub solid: u32,
}

impl Strewing {
	/// What every entity starts as: strewing nothing, and a rule that would lay
	/// a copy a square unit, upright, turned at random, anywhere the ground
	/// faces up.
	pub const NONE: Self = Self {
		strews: 0,
		seed: 0,
		density: DENSITY,
		slope: SLOPE,
		band: [-BAND, BAND],
		size: [1.0, 1.0],
		sink: [0.0, 0.0],
		turns: 1,
		align: 0.0,
		tilt: 0.0,
		shade: 0.0,
		reach: 0.0,
		fade: FADE,
		shadows: 1,
		solid: 0,
	};

	/// Whether the entity's mesh is strewn rather than drawn where it stands.
	#[must_use]
	pub const fn strews(self) -> bool { self.strews != 0 }

	/// Whether each copy is turned at random about its own up.
	#[must_use]
	pub const fn turns(self) -> bool { self.turns != 0 }

	/// Whether the copies throw shadows.
	#[must_use]
	pub const fn shadows(self) -> bool { self.shadows != 0 }

	/// Whether anything collides with the copies.
	#[must_use]
	pub const fn solid(self) -> bool { self.solid != 0 }

	/// The same rule with every number where laying out can use it.
	///
	/// A number that is not one becomes the default's; a count below nought
	/// becomes nought; an angle is held between nought and a right angle; a
	/// share between nought and one; and a pair given the wrong way round is
	/// read the right way round. A file and a game may write any of these, and
	/// only a panel clamps as it goes.
	#[must_use]
	pub fn sane(self) -> Self {
		let or = |value: f32, fallback: f32| if value.is_finite() { value } else { fallback };
		let pair = |[low, high]: [f32; 2], fallback: [f32; 2]| {
			let (low, high) = (or(low, fallback[0]), or(high, fallback[1]));

			if low <= high { [low, high] } else { [high, low] }
		};
		let [size_low, size_high] = pair(self.size, Self::NONE.size);

		Self {
			density: or(self.density, DENSITY).max(0.0),
			slope: or(self.slope, SLOPE).clamp(0.0, SLOPE),
			band: pair(self.band, Self::NONE.band),
			size: [size_low.max(0.0), size_high.max(0.0)],
			sink: pair(self.sink, Self::NONE.sink),
			align: or(self.align, 0.0).clamp(0.0, 1.0),
			tilt: or(self.tilt, 0.0).clamp(0.0, MAX_TILT),
			shade: or(self.shade, 0.0).clamp(0.0, 1.0),
			reach: or(self.reach, 0.0).max(0.0),
			fade: or(self.fade, FADE).clamp(0.0, 1.0),
			..self
		}
	}

	/// Whether two rules would lay the same copies, compared bit for bit.
	///
	/// Bits rather than values, so that a number that is not one is still the
	/// same as itself and a rule holding one is not laid again every step.
	#[must_use]
	pub fn same(&self, other: &Self) -> bool {
		bytemuck::bytes_of(self) == bytemuck::bytes_of(other)
	}
}

impl Default for Strewing {
	fn default() -> Self { Self::NONE }
}

/// [`Strewing`] as the record every world declares.
pub const STREWING: Record<Strewing> = Record {
	name: "strewing",
	help: "whether the entity's mesh is strewn over the ground it hangs off, and by what rule",
	rows: &[
		crate::row!(
			Bool,
			Strewing,
			strews,
			"its mesh strewn over the ground it hangs off rather than drawn where it stands"
		),
		crate::row!(Int, Strewing, seed, "what the copies are laid out from"),
		crate::row!(
			Float,
			Strewing,
			density,
			"how many copies stand on a square unit of ground, seen from above"
		),
		crate::row!(Float, Strewing, slope, "the steepest ground a copy stands on, in degrees"),
		crate::row!(
			Vec2,
			Strewing,
			band,
			"the lowest and highest ground a copy stands on, in the ground's own space"
		),
		crate::row!(
			Vec2,
			Strewing,
			size,
			"the smallest and largest a copy is, as a share of how the entity holds one"
		),
		crate::row!(
			Vec2,
			Strewing,
			sink,
			"the least and most a copy is sunk into the ground, along its own up"
		),
		crate::row!(Bool, Strewing, turns, "each copy turned at random about its own up"),
		crate::row!(
			Float,
			Strewing,
			align,
			"how far a copy leans with the ground: nought upright, one along its face"
		),
		crate::row!(Float, Strewing, tilt, "the most a copy leans at random, in degrees"),
		crate::row!(
			Float,
			Strewing,
			shade,
			"how much darker a copy may be: nought all alike, one down to black"
		),
		crate::row!(
			Float,
			Strewing,
			reach,
			"how far from the eye a copy is drawn, nought for any"
		),
		crate::row!(
			Float,
			Strewing,
			fade,
			"what share of the reach, at its far end, the copies thin out over"
		),
		crate::row!(Bool, Strewing, shadows, "the copies throw shadows"),
		crate::row!(
			Bool,
			Strewing,
			solid,
			"things collide with the copies: one body of every copy's coarsest level"
		),
	],
	default: Strewing::NONE,
};

/// Whether an entity strews its mesh rather than drawing it where it stands.
///
/// The one question the renderer, the bake and the editor's pointer all ask
/// before they treat an entity's mesh as something standing where the entity
/// does: for one that strews, nothing is there.
///
/// @param entities - the table
/// @param id - the entity
/// @return `false` for a stale handle, or a world that never declared the
/// record
#[must_use]
pub fn strews(entities: &Entities, id: EntityId) -> bool {
	entities
		.record(&STREWING, id)
		.is_some_and(|strewing| strewing.strews())
}

/// One copy, as it was laid: where it stands on the ground, how it is turned,
/// how big and how dark, all in the ground's own space.
///
/// Thirty-six bytes. The turn is the whole of it - the lean with the ground,
/// the lean at random, the turn about its up and the way the entity holds one -
/// so that a copy's model matrix is its place, this turn and its size times the
/// entity's own scale, and nothing else.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Piece {
	/// Where it stands.
	pub at: [f32; 3],

	/// How it is turned, `[x, y, z, w]`.
	pub turn: [f32; 4],

	/// How big it is, as a share of how the entity holds one.
	pub size: f32,

	/// What its color is multiplied by: one for as the entity is, less for a
	/// darker copy.
	pub shade: f32,
}

impl Piece {
	/// Where the copy stands and how, as a transform in the ground's space.
	///
	/// @param held - the entity's own scale, which every copy is held at
	#[must_use]
	pub fn transform(&self, held: Vec3) -> Transform {
		Transform {
			position: Vec3::from_array(self.at),
			rotation: Quat::from_array(self.turn),
			scale: held * self.size,
		}
	}

	/// The words a digest is made of, in the order they are laid out in.
	fn words(&self) -> impl Iterator<Item = u32> + use<> {
		let [east, up, south] = self.at.map(f32::to_bits);
		let [turn_x, turn_y, turn_z, turn_w] = self.turn.map(f32::to_bits);

		[
			east,
			up,
			south,
			turn_x,
			turn_y,
			turn_z,
			turn_w,
			self.size.to_bits(),
			self.shade.to_bits(),
		]
		.into_iter()
	}
}

/// A run of copies standing on one square of the ground, and the box they fill.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Patch {
	/// Where the run starts among the copies.
	pub first: u32,

	/// How many copies it holds.
	pub count: u32,

	/// The low corner of the box around every copy in it, in the ground's
	/// space.
	pub low: [f32; 3],

	/// And the high corner.
	pub high: [f32; 3],

	/// The largest any copy in it is drawn, on its largest axis: what a level
	/// of the mesh is chosen by.
	pub largest: f32,
}

impl Patch {
	/// Where its copies are among all of them.
	#[must_use]
	pub fn run(&self) -> Range<usize> {
		let first = usize::try_from(self.first).unwrap_or(usize::MAX);
		let count = usize::try_from(self.count).unwrap_or(0);

		first..first.saturating_add(count)
	}
}

/// Everything one rule laid over one ground.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Laid {
	/// Every copy, patch by patch, and inside a patch in the order drawn for
	/// them.
	pub pieces: Vec<Piece>,

	/// Every patch, in the order their copies are in.
	pub patches: Vec<Patch>,

	/// How many copies the rule drew before the band took any away: the
	/// density's own answer, which is what a count is checked against.
	pub drawn: u64,

	/// How many of those a [`Mask`] took away.
	///
	/// Its own count and not part of [`drawn`](Self::drawn), so that what the
	/// band left out is what is left over: a rule's copies are the ones it
	/// drew, less the ones the band left out, less these.
	pub masked: u64,

	/// Whether the rule would have laid more than [`MOST`] and was laid at a
	/// lower density that fits.
	pub thinned: bool,

	/// Sixty-four bits of every copy, in order: the same number on every
	/// machine the same rule is laid on.
	pub digest: u64,
}

/// What a layout was laid from: what, when it changes, lays it again.
#[derive(Clone, Copy, Debug)]
pub struct Key {
	/// The rule.
	pub rule: Strewing,

	/// The ground, and the revision of it that was read.
	pub ground: (MeshId, u32),

	/// What was strewn, and the revision of it that was read.
	pub mesh: (MeshId, u32),

	/// The entity's own transform inside the ground: its turn and scale are how
	/// every copy is held, and its place is where the body a solid strewing
	/// makes stands.
	pub local: Transform,

	/// The digest of the mask it was laid through, or nought for a layout laid
	/// with no mask at all.
	///
	/// The digest rather than a count of how many times one has been painted,
	/// and for the fault the renderer's own key was found to have: a world put
	/// back brings a mask whose history did not happen here, so the only thing
	/// that says two masks are the same mask is the cells. @ref
	/// [`Mask::digest`].
	pub mask: u64,
}

impl Key {
	/// Whether two keys would lay the same thing, compared bit for bit.
	///
	/// @param other - the other key
	#[must_use]
	pub fn same(&self, other: &Self) -> bool {
		let bits = |transform: &Transform| {
			let mut words = [0_u32; 10];

			for (word, value) in words.iter_mut().zip(
				transform
					.position
					.to_array()
					.into_iter()
					.chain(transform.rotation.to_array())
					.chain(transform.scale.to_array()),
			) {
				*word = value.to_bits();
			}

			words
		};

		self.rule.same(&other.rule)
			&& self.ground == other.ground
			&& self.mesh == other.mesh
			&& self.mask == other.mask
			&& bits(&self.local) == bits(&other.local)
	}
}

/// What one strewing entity laid, and what from.
#[derive(Clone, Debug)]
pub struct Layout {
	/// Who laid it: the handle, so a slot that changed hands is not taken for
	/// the entity that had it.
	pub entity: EntityId,

	/// What it was laid from.
	pub key: Key,

	/// What was laid.
	pub laid: Laid,

	/// The ground that a solid strewing's body collides with, or
	/// [`MeshId::NONE`] for one that is not solid.
	pub solid: MeshId,

	/// Which laying this is, counted across the whole table: what a renderer
	/// keeps beside what it made of it, and compares.
	pub revision: u32,
}

/// Every copy every strewing laid, by the strewing entity's slot.
///
/// **Host-written and not saved**, the arrangement the particle pool has: what
/// is here follows from the records and the ground, and a save that carried it
/// would be carrying a derivation. A game may read it. @ref
/// `colby_runtime::strew`, which keeps it in line with the records.
#[derive(Clone, Debug, Default)]
pub struct Strewn {
	/// One slot of the entity table each.
	layouts: Vec<Option<Layout>>,

	/// How many layings there have been.
	revision: u32,
}

impl Strewn {
	/// Nothing strewn.
	#[must_use]
	pub const fn new() -> Self { Self { layouts: Vec::new(), revision: 0 } }

	/// What the strewing entity in a slot laid, if it laid anything.
	///
	/// @param slot - the entity's place in the table
	#[must_use]
	pub fn get(&self, slot: usize) -> Option<&Layout> { self.layouts.get(slot)?.as_ref() }

	/// Puts what a strewing entity laid in its slot, replacing whatever was
	/// there, and counts a laying.
	///
	/// @param slot - the entity's place in the table
	/// @param layout - what it laid; its revision is written here
	pub fn put(&mut self, slot: usize, mut layout: Layout) {
		if slot >= self.layouts.len() {
			self.layouts
				.resize_with(slot.saturating_add(1), || None);
		}

		self.revision = self.revision.wrapping_add(1);
		layout.revision = self.revision;

		if let Some(held) = self.layouts.get_mut(slot) {
			*held = Some(layout);
		}
	}

	/// Takes what a slot laid away.
	///
	/// @param slot - the entity's place in the table
	/// @return what was there
	pub fn take(&mut self, slot: usize) -> Option<Layout> {
		let taken = self.layouts.get_mut(slot)?.take();

		if taken.is_some() {
			self.revision = self.revision.wrapping_add(1);
		}

		taken
	}

	/// Every slot that laid something, with what it laid, in slot order.
	pub fn iter(&self) -> impl Iterator<Item = (usize, &Layout)> {
		self.layouts
			.iter()
			.enumerate()
			.filter_map(|(slot, layout)| Some((slot, layout.as_ref()?)))
	}

	/// How many copies there are in the whole table.
	#[must_use]
	pub fn pieces(&self) -> usize {
		self.iter()
			.map(|(_, layout)| layout.laid.pieces.len())
			.sum()
	}

	/// How many layings there have been: a number that changes whenever
	/// anything in the table does.
	#[must_use]
	pub const fn revision(&self) -> u32 { self.revision }
}

/// What share of a rule's copies may stand where: the ground painted a cell at
/// a time.
///
/// **What a brush paints is the rule's input and never its output.** A cell
/// holds how much of what the rule would lay over it stands - [`OPEN`] for all
/// of it, nought for none, and anything between for that share - and no copy is
/// written down here or anywhere else. Painting therefore *thins* a field: the
/// draw a copy is kept or dropped by is drawn for every copy whether it is kept
/// or not, so a cell painted takes its own copies away and moves nobody else's.
/// @ref [`Laying::copy`].
///
/// **A grid over the ground seen from above, in the ground's own space**, and
/// not a picture on a second unwrap of the ground or a weight on its vertices.
/// The first reason is the strongest: a density here is already a count per
/// square unit of ground *seen from above*, so this is the projection the rule
/// is measured in and not a second one. The others are that a ground's mesh is
/// shared by whatever draws it and is built again whenever a terrain's numbers
/// change, which makes a weight on a vertex a weight on something that comes
/// and goes; and that a ground somebody modeled need have no second unwrap at
/// all.
///
/// **It outlives the ground it was painted over.** Heights that move, a mesh
/// built again, a ground scaled or turned: the cells say nothing about any of
/// them, and a copy reads the cell it stands over. Ground outside the grid
/// reads [`OPEN`], so a ground that has grown carries its whole field over the
/// new part rather than a bare one, and @ref [`Mask::fitted`] is what lays a
/// wider grid when somebody paints there.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
	/// The corner the cells start at, east and south, in the ground's space.
	from: [f32; 2],

	/// How wide one cell is.
	step: f32,

	/// How many cells there are east and south.
	counts: [u32; 2],

	/// One byte a cell, a row of east cells at a time.
	cells: Vec<u8>,

	/// Sixty-four bits of every cell and of the grid they stand on, never
	/// nought: what a layout is laid again when it changes.
	digest: u64,
}

impl Mask {
	/// A mask covering a ground, with nothing painted away yet.
	///
	/// The cells are [`CELL`] wide, doubled until there are no more than
	/// [`MOST_CELLS`] of them - so a ground somebody can see the whole of is
	/// painted about as finely as the brush that paints it, and a landscape is
	/// painted more coarsely rather than not at all.
	///
	/// @param bounds - the ground's box in its own space
	/// @return nothing for a box that is not a box
	#[must_use]
	pub fn over(bounds: (Vec3, Vec3)) -> Option<Self> {
		let (low, high) = bounds;

		if !low.is_finite() || !high.is_finite() {
			return None;
		}

		let from = [low.x.min(high.x), low.z.min(high.z)];
		let to = [low.x.max(high.x), low.z.max(high.z)];
		let mut step = CELL;

		// a doubling that cannot run away: sixty-four of them is every step a
		// float has room for, and a ground that wide has no copies on it
		for _ in 0..64 {
			let counts = [across(from[0], to[0], step)?, across(from[1], to[1], step)?];

			if cells_in(counts).is_some_and(|many| many <= MOST_CELLS) {
				let many = cells_in(counts)?;

				return Self::new(from, step, counts, vec![OPEN; many]);
			}

			step *= 2.0;
		}

		None
	}

	/// A mask read back from somewhere that wrote one down.
	///
	/// @param from - the corner the cells start at, east and south
	/// @param step - how wide one cell is
	/// @param counts - how many cells east and south
	/// @param cells - one byte a cell, a row of east cells at a time
	/// @return nothing for a grid that is not one, or for a number of cells
	/// that is not the grid's
	#[must_use]
	pub fn new(from: [f32; 2], step: f32, counts: [u32; 2], cells: Vec<u8>) -> Option<Self> {
		let many = cells_in(counts)?;

		if !from[0].is_finite()
			|| !from[1].is_finite()
			|| step.is_nan()
			|| step <= 0.0
			|| many > MOST_CELLS
			|| cells.len() != many
		{
			return None;
		}

		let mut mask = Self { from, step, counts, cells, digest: 0 };

		mask.restamp();

		Some(mask)
	}

	/// The corner the cells start at, east and south.
	#[must_use]
	pub const fn from(&self) -> [f32; 2] { self.from }

	/// How wide one cell is.
	#[must_use]
	pub const fn step(&self) -> f32 { self.step }

	/// How many cells there are east and south.
	#[must_use]
	pub const fn counts(&self) -> [u32; 2] { self.counts }

	/// Every cell, a row of east cells at a time.
	#[must_use]
	pub fn cells(&self) -> &[u8] { &self.cells }

	/// Sixty-four bits of the whole of it, and never nought.
	///
	/// Nought is what a key holds for a layout laid with no mask at all, so a
	/// mask may not answer it. @ref [`Key::mask`].
	#[must_use]
	pub const fn digest(&self) -> u64 { self.digest }

	/// What share of the field has been painted away, from nought for none of
	/// it to one for all.
	#[must_use]
	pub fn painted(&self) -> f32 {
		let open = self.cells.len().saturating_mul(usize::from(OPEN));
		let held: usize = self
			.cells
			.iter()
			.map(|cell| usize::from(*cell))
			.sum();

		if open == 0 { 0.0 } else { 1.0 - share_of(held, open) }
	}

	/// What share of a rule's copies stands at a place on the ground.
	///
	/// Read between the four cells around the place rather than out of the one
	/// it falls in, so that a grid coarser than the copies are dense reads as a
	/// slope and not as squares. A place outside the grid reads as [`OPEN`],
	/// and so does every place on a mask whose cells are all open - which is
	/// what makes a mask nobody has painted the same field, copy for copy, as
	/// no mask at all.
	///
	/// @param east - where it stands along the ground's first axis
	/// @param south - and along its third
	#[must_use]
	pub fn at(&self, east: f32, south: f32) -> f32 {
		let reach = self.reach();

		if east < self.from[0] || east > reach[0] || south < self.from[1] || south > reach[1] {
			return 1.0;
		}

		// the cells' own middles are half a cell in, which is where reading
		// between them starts from
		let (left, right, along) =
			cell_at((east - self.from[0]) / self.step - 0.5, self.counts[0]);
		let (near, far, across) =
			cell_at((south - self.from[1]) / self.step - 0.5, self.counts[1]);
		let wide = usize::try_from(self.counts[0]).unwrap_or(0);
		let corner = |x: usize, z: usize| {
			self.cells
				.get(z.saturating_mul(wide).saturating_add(x))
				.map_or(OPEN, |cell| *cell)
		};
		let row =
			|z: usize| between(f32::from(corner(left, z)), f32::from(corner(right, z)), along);

		between(row(near), row(far), across) / f32::from(OPEN)
	}

	/// Paints a round patch of it.
	///
	/// @param at - where the middle of the patch is, east and south, in the
	/// ground's space
	/// @param radius - how far it reaches
	/// @param strength - how hard, from minus one for putting a whole field
	/// back to one for taking one away
	/// @return whether any cell changed
	pub fn paint(&mut self, at: Vec2, radius: f32, strength: f32) -> bool {
		if !at.is_finite() || radius.is_nan() || radius <= 0.0 || !strength.is_finite() {
			return false;
		}

		let mut moved = false;
		let low = self.cell_of(Vec2::new(at.x - radius, at.y - radius));
		let high = self.cell_of(Vec2::new(at.x + radius, at.y + radius));

		for row in low.1..=high.1 {
			for column in low.0..=high.0 {
				moved |= self.dab(column, row, at, radius, strength);
			}
		}

		if moved {
			self.restamp();
		}

		moved
	}

	/// Paints one cell of a patch, and says whether it moved.
	///
	/// A call of its own rather than the body of two loops, because two loops
	/// and a question inside them is one level of nesting past what this
	/// workspace allows - and because what it decides, how far into the patch
	/// a cell's own middle is, is the whole of a stroke's softness.
	///
	/// @param column - which cell east
	/// @param row - which cell south
	/// @param at - the middle of the patch
	/// @param radius - how far it reaches
	/// @param strength - how hard
	fn dab(&mut self, column: usize, row: usize, at: Vec2, radius: f32, strength: f32) -> bool {
		let middle = Vec2::new(
			along(self.from[0], self.step, place_of(column) + 0.5),
			along(self.from[1], self.step, place_of(row) + 0.5),
		);
		// one at the middle of the patch and nought at its edge, which is the
		// softness a stroke's edge has
		let reach = (middle - at).length() / radius;

		if reach > 1.0 {
			return false;
		}

		let wide = usize::try_from(self.counts[0]).unwrap_or(0);
		let into = 1.0 - reach * reach;
		let Some(cell) = self
			.cells
			.get_mut(row.saturating_mul(wide).saturating_add(column))
		else {
			return false;
		};
		let painted = stepped(*cell, strength * into * f32::from(OPEN));
		let moved = painted != *cell;

		*cell = painted;

		moved
	}

	/// Puts every cell back to [`OPEN`].
	///
	/// @return whether any cell changed
	pub fn clear(&mut self) -> bool {
		let moved = self.cells.iter().any(|cell| *cell != OPEN);

		if moved {
			self.cells.fill(OPEN);
			self.restamp();
		}

		moved
	}

	/// The same mask over a wider grid, when a ground has grown past this one.
	///
	/// Cells are carried over by which cell of the new grid each one falls in,
	/// and ground the old grid never covered is [`OPEN`]. What this is for is
	/// the moment a brush is put to ground that was not there when the mask was
	/// made - a terrain widened, a ground swapped for a bigger one - and it is
	/// done there rather than every step, so that the grid is a fact about the
	/// mask rather than about whatever the ground is doing this frame.
	///
	/// @param bounds - the ground's box now, in its own space
	/// @return a wider mask, or nothing when this one already covers it
	#[must_use]
	pub fn fitted(&self, bounds: (Vec3, Vec3)) -> Option<Self> {
		let (low, high) = bounds;

		if !low.is_finite() || !high.is_finite() {
			return None;
		}

		let reach = self.reach();
		let wanted = (
			Vec3::new(low.x.min(high.x), 0.0, low.z.min(high.z)),
			Vec3::new(low.x.max(high.x), 0.0, low.z.max(high.z)),
		);

		if wanted.0.x >= self.from[0]
			&& wanted.0.z >= self.from[1]
			&& wanted.1.x <= reach[0]
			&& wanted.1.z <= reach[1]
		{
			return None;
		}

		let held =
			(Vec3::new(self.from[0], 0.0, self.from[1]), Vec3::new(reach[0], 0.0, reach[1]));
		let mut grown = Self::over((wanted.0.min(held.0), wanted.1.max(held.1)))?;
		let wide = usize::try_from(grown.counts[0]).unwrap_or(0);

		for (place, cell) in grown.cells.iter_mut().enumerate() {
			let (column, row) = (place % wide.max(1), place / wide.max(1));
			let middle = Vec2::new(
				along(grown.from[0], grown.step, place_of(column) + 0.5),
				along(grown.from[1], grown.step, place_of(row) + 0.5),
			);
			let taken = self.cell_of(middle);

			if middle.x >= self.from[0]
				&& middle.x <= reach[0]
				&& middle.y >= self.from[1]
				&& middle.y <= reach[1]
			{
				*cell = self
					.cells
					.get(
						taken
							.1
							.saturating_mul(usize::try_from(self.counts[0]).unwrap_or(0))
							.saturating_add(taken.0),
					)
					.map_or(OPEN, |cell| *cell);
			}
		}

		grown.restamp();

		Some(grown)
	}

	/// The far corner of the grid, east and south.
	fn reach(&self) -> [f32; 2] {
		[
			along(self.from[0], self.step, span_of(self.counts[0])),
			along(self.from[1], self.step, span_of(self.counts[1])),
		]
	}

	/// Which cell a place falls in, held inside the grid.
	fn cell_of(&self, at: Vec2) -> (usize, usize) {
		let held = |along: f32, from: f32, count: u32| -> usize {
			let last = usize::try_from(count.saturating_sub(1)).unwrap_or(0);

			floored((along - from) / self.step).min(last)
		};

		(
			held(at.x, self.from[0], self.counts[0]),
			held(at.y, self.from[1], self.counts[1]),
		)
	}

	/// Works the digest out again, after anything about the cells has changed.
	fn restamp(&mut self) {
		let grid = [
			self.from[0].to_bits(),
			self.from[1].to_bits(),
			self.step.to_bits(),
			self.counts[0],
			self.counts[1],
		];
		let stamped = fnv(grid
			.into_iter()
			.flat_map(u32::to_le_bytes)
			.chain(self.cells.iter().copied()));

		// nought is what a key says for a layout laid with no mask, so a mask
		// may not answer it. One collision in eighteen million million million
		// is not something to leave a hole for.
		self.digest = if stamped == 0 { 1 } else { stamped };
	}
}

/// How many cells a grid holds, or nothing for one too big to count.
fn cells_in(counts: [u32; 2]) -> Option<usize> {
	let wide = usize::try_from(counts[0]).ok()?;
	let deep = usize::try_from(counts[1]).ok()?;
	let many = wide.checked_mul(deep)?;

	(many > 0).then_some(many)
}

/// How many cells of a width cover a span, at least one.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "a ceiling held below what the ceiling on cells allows before it is cast"
)]
fn across(from: f32, to: f32, step: f32) -> Option<u32> {
	let cells = ((to - from) / step).ceil();

	if !cells.is_finite() || cells > 65_536.0 {
		return None;
	}

	Some((cells.max(1.0) as u32).max(1))
}

/// A count of cells as a number to measure with.
#[expect(
	clippy::as_conversions,
	clippy::cast_precision_loss,
	reason = "a count below the ceiling on cells, which a float holds exactly"
)]
fn span_of(count: u32) -> f32 { count as f32 }

/// A place in a row of cells as a number to measure with.
#[expect(
	clippy::as_conversions,
	clippy::cast_precision_loss,
	reason = "a place below the ceiling on cells, which a float holds exactly"
)]
fn place_of(place: usize) -> f32 { place as f32 }

/// The whole number a number floors to, at least nought.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "floored and held above nought and below what the type holds before it is cast"
)]
fn floored(along: f32) -> usize {
	let floored = along.floor();

	if floored.is_finite() {
		floored.clamp(0.0, 65_536.0) as usize
	} else {
		0
	}
}

/// A cell with some paint taken out of it, held inside what a byte holds.
///
/// The subtraction is here and not at the call site so that it is a difference
/// of two numbers rather than a product added to a sum, which the workspace's
/// lints push towards a fused multiply-add - and a mask decides which copies
/// stand, so it is one of the places that may not have one. @ref [`along`].
///
/// @param cell - what it holds now
/// @param by - how much to take out of it, of nought to [`OPEN`]
fn stepped(cell: u8, by: f32) -> u8 {
	let value = f32::from(cell) - by;

	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "rounded and held inside what a byte holds before it is cast"
	)]
	if value.is_finite() {
		value.clamp(0.0, f32::from(OPEN)).round() as u8
	} else {
		OPEN
	}
}

/// One whole number over another.
#[expect(
	clippy::as_conversions,
	clippy::cast_precision_loss,
	reason = "two sums of bytes, each below what a float counts exactly"
)]
fn share_of(held: usize, whole: usize) -> f32 { held as f32 / whole as f32 }

/// The two cells a place falls between, and how far it is from the first.
///
/// Held inside the grid at either end, so that a place in the outer half of an
/// outer cell reads that cell and not past it.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "held inside the grid before it is cast"
)]
fn cell_at(along: f32, count: u32) -> (usize, usize, f32) {
	if !along.is_finite() {
		return (0, 0, 0.0);
	}

	let last = span_of(count.saturating_sub(1));
	let floored = along.floor();
	let low = floored.clamp(0.0, last) as usize;
	let high = (floored + 1.0).clamp(0.0, last) as usize;

	(low, high, (along - floored).clamp(0.0, 1.0))
}

/// Lays a rule's copies out over a ground.
///
/// @param ground - the mesh the copies stand on, in its own space; its own
/// triangles and not any coarser level of them
/// @param rule - how they are laid
/// @param local - the strewing entity's own transform, whose turn and scale
/// are how every copy is held
/// @param bounds - the box of what is strewn, in its own space
/// @param mask - what share of the copies may stand where, or nothing for a
/// rule nobody has painted. **The ceiling on copies is about the rule and not
/// about the painting**: a rule that would lay more than [`MOST`] is laid at a
/// density that fits whether or not a mask would have taken those copies away,
/// so that what a person paints does not change how thickly the rest is laid.
/// @return every copy, in patches
#[must_use]
pub fn lay_out(
	ground: &MeshData,
	rule: &Strewing,
	local: &Transform,
	bounds: (Vec3, Vec3),
	mask: Option<&Mask>,
) -> Laid {
	let rule = rule.sane();
	let faces = faces(ground, rule.slope);
	let asked: f64 = faces
		.iter()
		.map(|face| f64::from(face.area) * f64::from(rule.density))
		.sum();
	let most = f64::from(u32::try_from(MOST).unwrap_or(u32::MAX));
	// a rule that would lay more than fits is laid at the density that fills
	// about ninety-nine hundredths of it, so the draws that settle each
	// triangle's last copy do not carry it past the ceiling; the ceiling below
	// stops it anyway, and this is what keeps that stop from ever falling on
	// the last triangles alone
	let thinned = asked > most;
	let density = if thinned {
		narrowed(f64::from(rule.density) * most * 0.99 / asked)
	} else {
		rule.density
	};
	let cosine = cosine_of(rule.tilt);
	let mut laying = Laying {
		rule,
		local: *local,
		cosine,
		mask,
		drawn: 0,
		masked: 0,
		found: Vec::new(),
	};

	for face in &faces {
		if laying.found.len() >= MOST {
			break;
		}

		laying.face(face, density);
	}

	let (pieces, patches) = patched(laying.found, local.scale, bounds);
	let digest = digest(pieces.iter().flat_map(Piece::words));

	Laid {
		pieces,
		patches,
		drawn: laying.drawn,
		masked: laying.masked,
		thinned,
		digest,
	}
}

/// The copies laid back into one mesh, for something to collide with: every
/// copy's coarsest level, where the copy stands, in the space of the entity
/// that strews them.
///
/// In the entity's space rather than the ground's because the body that holds
/// it stands where the entity does, which is what keeps it there when the
/// ground moves. Nothing is drawn with it, so it carries positions and nothing
/// the eye would need.
///
/// @param laid - what was laid
/// @param mesh - what was strewn, whose coarsest level is what collides
/// @param local - the strewing entity's own transform inside the ground
/// @return the mesh, or an empty one when nothing was laid or the entity is
/// flattened along an axis and has no inside to put the mesh in
#[must_use]
pub fn solid(laid: &Laid, mesh: &MeshData, local: &Transform) -> MeshData {
	let inverse = local.matrix().inverse();

	if laid.pieces.is_empty() || !inverse.is_finite() {
		return MeshData::default();
	}

	let indices: &[u32] = mesh
		.levels
		.last()
		.map_or(&mesh.indices, |level| &level.indices);
	let corners = mesh.vertices.len();
	let mut solid = MeshData {
		vertices: Vec::with_capacity(corners.saturating_mul(laid.pieces.len())),
		indices: Vec::with_capacity(indices.len().saturating_mul(laid.pieces.len())),
		..MeshData::default()
	};

	for piece in &laid.pieces {
		let Ok(first) = u32::try_from(solid.vertices.len()) else {
			break;
		};
		let placed = inverse * piece.transform(local.scale).matrix();

		solid
			.vertices
			.extend(mesh.vertices.iter().map(|vertex| {
				let at = placed.transform_point3(Vec3::from_array(vertex.position));

				MeshVertex::new(at, Vec3::Y, Vec2::ZERO)
			}));
		solid.indices.extend(
			indices
				.iter()
				.map(|index| first.saturating_add(*index)),
		);
	}

	solid
}

/// One triangle of the ground a copy may stand on.
struct Face {
	/// Where it is in the mesh, counted in triangles: what its stream is seeded
	/// with.
	index: u32,

	/// Its three corners.
	corners: [Vec3; 3],

	/// Which way it faces, of unit length.
	normal: Vec3,

	/// How much ground it covers seen from above.
	area: f32,
}

/// Every triangle of a ground that faces up and is no steeper than a slope.
///
/// @param ground - the mesh
/// @param slope - the steepest, in degrees from level
fn faces(ground: &MeshData, slope: f32) -> Vec<Face> {
	let level = cosine_of(slope);
	let mut found = Vec::with_capacity(ground.indices.len() / 3);

	for (index, corners) in ground.indices.chunks_exact(3).enumerate() {
		let (Ok(index), Some(corners)) = (u32::try_from(index), corners_of(ground, corners))
		else {
			continue;
		};
		let [a, b, c] = corners;
		let across = (b - a).cross(c - a);
		let length = across.length();

		// a triangle facing down or edge on covers no ground seen from above,
		// and one with no area at all has no way it faces
		//
		// @note: facing up is also what the slope below asks, whose cosine is
		// above nought for every slope `sane` lets through, so a mutation that
		// takes the first half of this out passes everything. It stays because
		// it is what the area further down is measured on.
		if !(across.y > 0.0 && length > 0.0) {
			continue;
		}

		let normal = across / length;

		if normal.y < level {
			continue;
		}

		found.push(Face {
			index,
			corners,
			normal,
			area: across.y * 0.5,
		});
	}

	found
}

/// A triangle's three corners, or nothing for one naming a corner the mesh
/// does not have.
fn corners_of(mesh: &MeshData, triangle: &[u32]) -> Option<[Vec3; 3]> {
	let corner = |slot: usize| -> Option<Vec3> {
		let index = usize::try_from(*triangle.get(slot)?).ok()?;

		Some(Vec3::from_array(mesh.vertices.get(index)?.position))
	};

	Some([corner(0)?, corner(1)?, corner(2)?])
}

/// What laying copies over one ground keeps as it goes.
struct Laying<'a> {
	/// The rule, made sane.
	rule: Strewing,

	/// The strewing entity's own transform: how every copy is held.
	local: Transform,

	/// The cosine of the rule's lean at random.
	cosine: f32,

	/// What share of the copies may stand where, or nothing for a rule nobody
	/// has painted.
	mask: Option<&'a Mask>,

	/// How many copies the rule has drawn.
	drawn: u64,

	/// How many of them the mask has taken away.
	masked: u64,

	/// What has been kept so far, each with its patch and its place in it.
	found: Vec<Found>,
}

/// What became of one copy the rule drew.
enum Drew {
	/// It stands.
	Kept(Found),

	/// The band left it out.
	Banded,

	/// The mask left it out.
	Masked,
}

/// One copy kept, before the copies are put in their patches.
struct Found {
	/// Which square of the ground it stands on.
	patch: (i32, i32),

	/// Its place among the others in that square.
	rank: u32,

	/// The copy.
	piece: Piece,
}

impl Laying<'_> {
	/// Lays one triangle's copies.
	///
	/// @param face - the triangle
	/// @param density - how many copies a square unit, after any thinning
	fn face(&mut self, face: &Face, density: f32) {
		let mut random = Random::new(stream(self.rule.seed, face.index));
		let expected = density * face.area;
		let whole = expected.floor();
		// always drawn, so that a triangle's copies are the same draws whatever
		// its last copy's chance came to
		let more = random.chance(threshold(expected - whole));
		let count = count_of(whole) + u64::from(more);

		for _ in 0..count {
			if self.found.len() >= MOST {
				return;
			}

			self.drawn = self.drawn.saturating_add(1);

			let drew = self.copy(face, &mut random);

			match drew {
				| Drew::Kept(found) => self.found.push(found),
				| Drew::Masked => self.masked = self.masked.saturating_add(1),
				| Drew::Banded => {},
			}
		}
	}

	/// Draws one copy, and keeps it if the rule's band does.
	///
	/// **Every number is drawn first and in one order**, whatever becomes of
	/// the copy: a copy the band leaves out uses the same draws a kept one
	/// would, so the copies after it in its triangle are where they would have
	/// been.
	///
	/// @param face - the triangle it stands on
	/// @param random - the triangle's stream
	/// @return the copy, or which of the two reasons left it out
	fn copy(&self, face: &Face, random: &mut Random) -> Drew {
		let (mut along, mut across) = (random.unit(), random.unit());
		let yaw = random.circle();
		let lean = random.unit();
		let lean_way = random.circle();
		let grown = random.unit();
		let sunk = random.unit();
		let dark = random.unit();
		let rank = u32::try_from(random.draw() >> 32).unwrap_or(0);
		// the draw the mask is read against, and it is drawn here whether the
		// strewing has ever been painted or not, so that painting one takes
		// copies away and moves none of the ones it leaves
		let painted = random.unit();

		// a point in the parallelogram the triangle is half of, folded back
		// into the triangle when it lands in the other half
		if along + across > 1.0 {
			along = 1.0 - along;
			across = 1.0 - across;
		}

		let [first, second, third] = face.corners;
		let point = first + (second - first) * along + (third - first) * across;

		if point.y < self.rule.band[0] || point.y > self.rule.band[1] {
			return Drew::Banded;
		}

		// and then what somebody painted, which is the second and last reason
		// a copy the rule drew does not stand. After the band rather than
		// before it because both are asked of a copy that has already been
		// drawn whole, and in this order because the band is a comparison and
		// this is four cells read out of a grid.
		//
		// **A cell reading open keeps every copy**: a share of one is above
		// every draw `unit` hands out, which is what makes a mask nobody has
		// painted the same field as no mask at all.
		if self
			.mask
			.is_some_and(|mask| painted >= mask.at(point.x, point.z))
		{
			return Drew::Masked;
		}

		let rule = &self.rule;
		let up = Vec3::Y * (1.0 - rule.align) + face.normal * rule.align;
		let aligned = Quat::from_rotation_arc(Vec3::Y, up.normalize_or(Vec3::Y));
		// a way uniformly over the cap of directions within the lean of up: the
		// height on the cap is what is even, and the square root of what is
		// left of the unit length is the way out from the middle
		let height = between(1.0, self.cosine, lean);
		let out = (1.0 - height * height).max(0.0).sqrt();
		let leaned = Quat::from_rotation_arc(
			Vec3::Y,
			Vec3::new(out * lean_way.x, height, out * lean_way.y).normalize_or(Vec3::Y),
		);
		// a point on the circle is a half-turn's cosine and sine, and an even
		// half-turn is an even whole one
		//
		// @note: so is one whose cosine is taken with either sign, and a
		// mutation that takes its magnitude is seen by the pinned layout alone:
		// the turns it lays are as even, and other ones
		let turned = if rule.turns() {
			Quat::from_xyzw(0.0, yaw.y, 0.0, yaw.x).normalize()
		} else {
			Quat::IDENTITY
		};
		let standing = aligned * leaned;
		let turn = standing * turned * self.local.rotation;
		let sink = between(rule.sink[0], rule.sink[1], sunk);
		let at = point - standing * Vec3::Y * sink;

		Drew::Kept(Found {
			patch: patch_of(at),
			rank,
			piece: Piece {
				at: at.to_array(),
				turn: turn.to_array(),
				size: between(rule.size[0], rule.size[1], grown),
				shade: between(1.0, 1.0 - rule.shade, dark),
			},
		})
	}
}

/// The stream one triangle's copies are drawn from.
///
/// The seed and the triangle's place run through a mixer that turns one bit
/// of difference into half the bits of the answer, because two streams of the
/// shift register seeded a step apart start out too much alike to lay
/// neighboring triangles independently.
///
/// @param seed - the rule's
/// @param index - the triangle's place in the ground
fn stream(seed: i32, index: u32) -> u64 {
	let seed = u64::from(seed.cast_unsigned());
	let mut mixed = ((seed << 32) | u64::from(index)).wrapping_add(0x9E37_79B9_7F4A_7C15);

	mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
	mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

	mixed ^ (mixed >> 31)
}

/// How many copies a whole number of them in a float is.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "a float has no fallible form of this; what is past the ceiling is stopped by it"
)]
fn count_of(whole: f32) -> u64 {
	if whole.is_finite() && whole > 0.0 {
		whole as u64
	} else {
		0
	}
}

/// Which square of the ground a place stands on.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "floored and held inside what the integer holds before it is cast"
)]
fn patch_of(at: Vec3) -> (i32, i32) {
	let square = |along: f32| {
		let floored = (along / PATCH).floor();

		if floored.is_finite() {
			floored.clamp(-1.0e9, 1.0e9) as i32
		} else {
			0
		}
	};

	(square(at.x), square(at.z))
}

/// Puts the copies in their patches: square by square, the rows of squares
/// along the ground's depth, and inside a square in the order drawn for them.
///
/// A sort that is stable, so two copies the draw put at one place in one square
/// keep the order they were laid in, which is the same on every machine.
///
/// @param found - every copy kept
/// @param held - the entity's own scale, every copy's box is carried at
/// @param bounds - the box of what is strewn, in its own space
fn patched(mut found: Vec<Found>, held: Vec3, bounds: (Vec3, Vec3)) -> (Vec<Piece>, Vec<Patch>) {
	found.sort_by_key(|one| (one.patch.1, one.patch.0, one.rank));

	let mut pieces = Vec::with_capacity(found.len());
	let mut patches: Vec<Patch> = Vec::new();
	let mut last = None;
	let (center, half) = ((bounds.0 + bounds.1) * 0.5, (bounds.1 - bounds.0) * 0.5);
	let widest = held.abs().max_element();

	for one in found {
		let Ok(at) = u32::try_from(pieces.len()) else {
			break;
		};
		let transform = one.piece.transform(held);
		// the box of a turned box: its middle carried, and each half width the
		// sum of what every half width of the box reaches along it
		let turned = Mat3::from_quat(transform.rotation) * Mat3::from_diagonal(transform.scale);
		let middle = transform.position + turned * center;
		let reach =
			Mat3::from_cols(turned.x_axis.abs(), turned.y_axis.abs(), turned.z_axis.abs())
				* half.abs();
		let (low, high) = (middle - reach, middle + reach);
		let largest = one.piece.size.abs() * widest;

		match patches.last_mut() {
			| Some(patch) if last == Some(one.patch) => {
				patch.count = patch.count.saturating_add(1);
				patch.low = Vec3::from_array(patch.low).min(low).to_array();
				patch.high = Vec3::from_array(patch.high).max(high).to_array();
				patch.largest = patch.largest.max(largest);
			},
			| _ => patches.push(Patch {
				first: at,
				count: 1,
				low: low.to_array(),
				high: high.to_array(),
				largest,
			}),
		}

		last = Some(one.patch);
		pieces.push(one.piece);
	}

	(pieces, patches)
}

/// How far along an axis a whole number of cells reaches from a corner.
///
/// Written as a product added to a sum and deliberately not fused, for
/// [`between`]'s reason and with as much riding on it: which cell a copy reads
/// is what decides whether the copy stands, so it has to be the same bits on
/// every machine.
#[expect(
	clippy::suboptimal_flops,
	reason = "a fused multiply and add is not the same bits on every machine"
)]
fn along(from: f32, step: f32, cells: f32) -> f32 { from + step * cells }

/// A number part of the way from one to another.
///
/// Written as a product added to a sum, and on purpose not fused into one
/// operation: a fused multiply and add is a library's where the processor has
/// none, and two libraries need not round it alike, where the standard pins a
/// product and a sum each to the bit. The copies have to be the same bytes on
/// every machine.
///
/// @param from - where it starts, at a share of nought
/// @param to - where it ends, at a share of one
/// @param share - how far along
#[expect(
	clippy::suboptimal_flops,
	reason = "a fused multiply and add is not the same bits on every machine"
)]
fn between(from: f32, to: f32, share: f32) -> f32 { from + (to - from) * share }

/// The cosine of an angle in degrees, worked out in double precision and
/// narrowed once.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "worked out in double precision and narrowed once, where it is kept"
)]
fn cosine_of(degrees: f32) -> f32 { f64::from(degrees).to_radians().cos() as f32 }

/// A double narrowed to a float.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "a density worked out in double precision and narrowed once"
)]
fn narrowed(value: f64) -> f32 { value as f32 }

/// Sixty-four bits of FNV over a run of words.
fn digest(words: impl Iterator<Item = u32>) -> u64 { fnv(words.flat_map(u32::to_le_bytes)) }

/// Sixty-four bits of FNV over a run of bytes.
fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
	bytes.fold(0xCBF2_9CE4_8422_2325, |held, byte| {
		(held ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
	})
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		abi::{mesh::cube, terrain::Terrain},
		glam::Mat4,
	};

	/// The matrix a copy is drawn at.
	fn matrix_of(piece: &Piece, held: Vec3) -> Mat4 { piece.transform(held).matrix() }

	/// Whether every one of some places is inside a patch's box.
	fn boxed(places: &[Vec3], patch: &Patch) -> bool {
		let (low, high) = (Vec3::from_array(patch.low), Vec3::from_array(patch.high));

		places
			.iter()
			.all(|at| at.cmpge(low - 1.0e-4).all() && at.cmple(high + 1.0e-4).all())
	}

	/// The eight corners of the unit cube, carried by a matrix.
	fn corners(matrix: Mat4) -> [Vec3; 8] {
		core::array::from_fn(|corner| {
			let half = |bit: usize| if corner & bit == 0 { -0.5 } else { 0.5 };

			matrix.transform_point3(Vec3::new(half(1), half(2), half(4)))
		})
	}

	/// A flat square of ground, a unit of it a quad, `side` units on a side,
	/// centered on the origin.
	fn floor(side: u32) -> MeshData {
		let mut ground = Terrain::hills();

		ground.size = f32::from(u16::try_from(side).unwrap_or(1));
		ground.side = side + 1;
		ground.height = 0.0;

		ground.build()
	}

	/// The hills the terrain draws by default.
	fn hills(seed: u32) -> MeshData { Terrain::of(seed).build() }

	/// A flat square of ground lifted into a slope of one along its depth: as
	/// high at each place as it is deep, so every face looks up and back at
	/// forty-five degrees and covers what the flat square did, seen from above.
	fn sloped(side: u32) -> MeshData {
		let mut ground = floor(side);

		for vertex in &mut ground.vertices {
			vertex.position[1] = vertex.position[2];
		}

		ground
	}

	/// A rule that strews, otherwise as given.
	fn rule(density: f32) -> Strewing { Strewing { strews: 1, density, ..Strewing::NONE } }

	/// The unit cube's box.
	const UNIT: (Vec3, Vec3) = (Vec3::splat(-0.5), Vec3::splat(0.5));

	#[test]
	fn a_flat_ground_carries_its_density_to_within_what_the_last_draws_settle() {
		let laid = lay_out(&floor(32), &rule(3.0), &Transform::IDENTITY, UNIT, None);

		// a thousand and twenty-four square units, three to each: every
		// triangle covers half a unit and is owed one and a half copies, so its
		// count is one or two and the whole has a spread of about sixteen
		let count = laid.pieces.len();

		assert!(count.abs_diff(3072) < 80, "{count} copies on 1024 square units at 3 a unit");
		assert_eq!(laid.drawn, u64::try_from(count).unwrap(), "nothing a band took away");
		assert!(!laid.thinned, "far below the ceiling");
	}

	#[test]
	fn every_copy_stands_on_the_ground_and_inside_it() {
		let ground = hills(1971);
		let laid = lay_out(&ground, &rule(2.0), &Transform::IDENTITY, UNIT, None);
		let terrain = Terrain::of(1971);

		assert!(laid.pieces.len() > 20_000, "{} copies on the default hills", laid.pieces.len());

		for piece in &laid.pieces {
			let [x, y, z] = piece.at;

			assert!(x.abs() <= 64.0 && z.abs() <= 64.0, "inside the ground: {x} {z}");
			// the surface the triangles make, which is the heights at the
			// corners of a cell and straight lines between: within the
			// smoothness the noise has across one cell of it
			let high = terrain.height_at(Vec2::new(x, z));

			assert!((y - high).abs() < 0.05, "on the ground at {x} {z}: {y} against {high}");
		}
	}

	#[test]
	fn the_same_rule_lays_the_same_copies_and_another_seed_others() {
		let ground = hills(7);
		let one = lay_out(&ground, &rule(1.0), &Transform::IDENTITY, UNIT, None);
		let two = lay_out(&ground, &rule(1.0), &Transform::IDENTITY, UNIT, None);
		let other = lay_out(
			&ground,
			&Strewing { seed: 1, ..rule(1.0) },
			&Transform::IDENTITY,
			UNIT,
			None,
		);

		assert_eq!(one, two, "one rule, one ground, one answer");
		assert_ne!(one.digest, other.digest, "a seed is a real knob");
		assert!(
			one.pieces.len().abs_diff(other.pieces.len()) < 400,
			"and another seed is the same density"
		);
	}

	#[test]
	fn the_layout_of_a_fixed_rule_is_the_layout_it_has_always_been() {
		// a known answer, written down: every copy of this rule on this ground,
		// folded into sixty-four bits. Two runs agreeing says nothing about
		// whether a draw moved; this does, and it is the same number on every
		// machine the gate runs on, which is the claim the module makes.
		//
		// If this fails, the laying changed. That is a decision to take on
		// purpose: every strewn field anybody made moves with it.
		let laid = lay_out(
			&hills(1971),
			&Strewing {
				seed: 42,
				slope: 20.0,
				band: [-2.0, 3.0],
				size: [0.5, 1.5],
				sink: [0.0, 0.25],
				align: 0.5,
				tilt: 12.0,
				shade: 0.3,
				..rule(0.5)
			},
			&Transform {
				rotation: Quat::from_rotation_x(-core::f32::consts::FRAC_PI_2),
				scale: Vec3::new(1.0, 2.0, 1.0),
				..Transform::IDENTITY
			},
			UNIT,
			None,
		);

		assert_eq!(laid.pieces.len(), 7815, "{}", laid.pieces.len());
		assert_eq!(laid.digest, 0x5AF8_8355_70F4_A0AA, "{:#018X}", laid.digest);
	}

	#[test]
	fn a_slope_leaves_out_what_is_steeper_and_moves_nothing_it_keeps() {
		let ground = hills(1971);
		let everywhere = lay_out(&ground, &rule(2.0), &Transform::IDENTITY, UNIT, None);
		let gentle = lay_out(
			&ground,
			&Strewing { slope: 8.0, ..rule(2.0) },
			&Transform::IDENTITY,
			UNIT,
			None,
		);
		let level = cosine_of(8.0);

		assert!(
			gentle.pieces.len() < everywhere.pieces.len() * 9 / 10,
			"the default hills have steeper ground than eight degrees: {} of {}",
			gentle.pieces.len(),
			everywhere.pieces.len()
		);

		// every copy kept by the gentler rule is one of the copies the steeper
		// one laid, bit for bit: a slope passes a triangle over and draws
		// nothing of it, and every other triangle's stream is its own
		let kept: std::collections::HashSet<[u32; 9]> = everywhere
			.pieces
			.iter()
			.map(|piece| {
				let mut words = [0; 9];
				words
					.iter_mut()
					.zip(piece.words())
					.for_each(|(slot, word)| *slot = word);
				words
			})
			.collect();

		for piece in &gentle.pieces {
			let mut words = [0; 9];
			words
				.iter_mut()
				.zip(piece.words())
				.for_each(|(slot, word)| *slot = word);

			assert!(kept.contains(&words), "a copy the gentler rule kept moved: {piece:?}");
		}

		// and no copy it kept stands on a triangle steeper than it asks
		let steep: Vec<[Vec3; 3]> = faces(&ground, 90.0)
			.into_iter()
			.filter(|face| face.normal.y < level)
			.map(|face| face.corners)
			.collect();

		assert!(!steep.is_empty(), "the check below has triangles to look inside");

		for piece in &gentle.pieces {
			let at = Vec3::from_array(piece.at);

			assert!(
				!steep
					.iter()
					.any(|&corners| deep_inside(at, corners)),
				"a copy at {at} stands on a triangle steeper than the slope"
			);
		}
	}

	/// Whether a point seen from above lies inside a triangle seen from above,
	/// and further from each of its edges than a rounding could put a point
	/// laid on the triangle beside it.
	fn deep_inside(point: Vec3, [a, b, c]: [Vec3; 3]) -> bool {
		let side = |from: Vec3, to: Vec3| {
			let edge = Vec2::new(to.x - from.x, to.z - from.z);

			edge.perp_dot(Vec2::new(point.x - from.x, point.z - from.z))
				/ edge.length().max(f32::EPSILON)
		};
		let (one, two, three) = (side(a, b), side(b, c), side(c, a));
		let margin = 1.0e-4;

		(one > margin && two > margin && three > margin)
			|| (one < -margin && two < -margin && three < -margin)
	}

	#[test]
	fn raising_the_density_adds_copies_after_the_ones_there_were() {
		let ground = floor(16);
		let sparse = lay_out(&ground, &rule(1.0), &Transform::IDENTITY, UNIT, None);
		let dense = lay_out(&ground, &rule(4.0), &Transform::IDENTITY, UNIT, None);

		let every: Vec<[f32; 3]> = dense
			.pieces
			.iter()
			.map(|piece| piece.at)
			.collect();

		for piece in &sparse.pieces {
			assert!(
				every.contains(&piece.at),
				"a copy at {:?} moved when more were asked",
				piece.at
			);
		}
	}

	#[test]
	fn a_band_keeps_only_the_ground_inside_it() {
		let ground = hills(1971);
		let everywhere = lay_out(&ground, &rule(2.0), &Transform::IDENTITY, UNIT, None);
		let banded = lay_out(
			&ground,
			&Strewing { band: [-1.0, 1.0], ..rule(2.0) },
			&Transform::IDENTITY,
			UNIT,
			None,
		);

		assert!(banded.pieces.len() < everywhere.pieces.len(), "some ground is outside it");
		assert!(!banded.pieces.is_empty(), "and some inside");
		assert_eq!(banded.drawn, everywhere.drawn, "a band draws as much and keeps less");

		for piece in &banded.pieces {
			assert!((-1.0..=1.0).contains(&piece.at[1]), "a copy at height {}", piece.at[1]);
		}
	}

	#[test]
	fn a_pair_given_the_wrong_way_round_is_read_the_right_way_round() {
		let sane = Strewing {
			band: [3.0, -3.0],
			size: [2.0, -1.0],
			slope: f32::NAN,
			density: -4.0,
			tilt: 400.0,
			..rule(1.0)
		}
		.sane();

		assert_eq!(sane.band.map(f32::to_bits), [-3.0_f32, 3.0].map(f32::to_bits), "upside down");
		assert_eq!(sane.size.map(f32::to_bits), [0.0_f32, 2.0].map(f32::to_bits), "below nought");
		assert!((sane.slope - SLOPE).abs() < f32::EPSILON, "not a number is the default");
		assert!(sane.density.abs() < f32::EPSILON, "fewer than none is none");
		assert!((sane.tilt - MAX_TILT).abs() < f32::EPSILON, "past lying flat is lying flat");

		let level = Strewing { slope: -20.0, ..rule(1.0) }.sane();

		assert!(level.slope.abs() < f32::EPSILON, "a slope below level is level ground only");
	}

	#[test]
	fn a_rule_that_lays_too_many_is_thinned_to_fit_and_says_so() {
		let laid = lay_out(&floor(64), &rule(100.0), &Transform::IDENTITY, UNIT, None);

		assert!(laid.thinned, "409,600 asked of a ceiling of {MOST}");
		assert!(laid.pieces.len() <= MOST, "{}", laid.pieces.len());
		assert!(laid.pieces.len() > MOST * 95 / 100, "and nearly filled: {}", laid.pieces.len());

		// thinned everywhere alike, and not stopped short at the ceiling with
		// the last of the ground left bare: the quarter of the floor laid last
		// holds a quarter of the copies
		let last = laid
			.pieces
			.iter()
			.filter(|piece| piece.at[2] > 16.0)
			.count();

		assert!(
			last.abs_diff(laid.pieces.len() / 4) < laid.pieces.len() / 100,
			"{last} of {} on the last quarter of the floor",
			laid.pieces.len()
		);
	}

	#[test]
	fn upright_copies_stand_upright_whatever_the_ground_does() {
		let laid = lay_out(
			&hills(3),
			&Strewing { turns: 0, ..rule(0.5) },
			&Transform::IDENTITY,
			UNIT,
			None,
		);

		for piece in &laid.pieces {
			assert_eq!(
				piece.turn.map(f32::to_bits),
				Quat::IDENTITY.to_array().map(f32::to_bits),
				"neither turned nor leaned"
			);
		}
	}

	#[test]
	fn a_copy_aligned_with_the_ground_stands_along_its_face() {
		// a ground whose every face looks the same way, so the way a copy
		// should stand is known without finding the triangle under it
		let ground = sloped(16);
		let face = Vec3::new(0.0, 1.0, -1.0).normalize();

		for (align, aimed) in [(1.0, face), (0.5, (Vec3::Y + face).normalize()), (0.0, Vec3::Y)] {
			let laid = lay_out(
				&ground,
				&Strewing { align, ..rule(1.0) },
				&Transform::IDENTITY,
				UNIT,
				None,
			);

			assert!(!laid.pieces.is_empty());

			// a copy turned about its own up keeps its up
			for piece in &laid.pieces {
				let up = Quat::from_array(piece.turn) * Vec3::Y;

				assert!(
					up.dot(aimed) > 1.0 - 1.0e-5,
					"aligned {align}, a copy standing along {up}"
				);
			}
		}
	}

	#[test]
	fn a_slope_carries_the_density_of_the_ground_seen_from_above_and_not_along_it() {
		// a floor lifted into a slope of one: its surface is the square root of
		// two times what it covers seen from above, and the count follows the
		// covering - two a unit over two hundred and fifty-six units
		let laid = lay_out(&sloped(16), &rule(2.0), &Transform::IDENTITY, UNIT, None);

		assert!(
			laid.pieces.len().abs_diff(512) < 60,
			"{} copies on 256 square units seen from above at 2 a unit",
			laid.pieces.len()
		);
	}

	#[test]
	fn a_copy_sunk_on_a_slope_is_sunk_along_its_own_up() {
		let ground = sloped(16);
		let face = Vec3::new(0.0, 1.0, -1.0).normalize();
		let laid = lay_out(
			&ground,
			&Strewing {
				align: 1.0,
				sink: [0.5, 0.5],
				..rule(1.0)
			},
			&Transform::IDENTITY,
			UNIT,
			None,
		);

		assert!(!laid.pieces.is_empty());

		// the ground is the plane through the origin with that face, so a copy
		// sunk half a unit along it stands half a unit behind the plane
		for piece in &laid.pieces {
			let behind = Vec3::from_array(piece.at).dot(face);

			assert!(
				(behind + 0.5).abs() < 1.0e-4,
				"a copy {behind} from the ground along its face"
			);
		}
	}

	#[test]
	fn a_lean_at_random_stays_inside_its_cone_and_uses_it() {
		let laid = lay_out(
			&floor(32),
			&Strewing { tilt: 30.0, ..rule(2.0) },
			&Transform::IDENTITY,
			UNIT,
			None,
		);
		let cone = cosine_of(30.0);
		let mut leanest: f32 = 1.0;

		for piece in &laid.pieces {
			let up = Quat::from_array(piece.turn) * Vec3::Y;

			assert!(up.y >= cone - 1.0e-5, "a copy leaning past thirty degrees: {up}");
			leanest = leanest.min(up.y);
		}

		assert!(leanest < cone + 0.01, "and some lean nearly all the way: {leanest}");
	}

	#[test]
	fn a_copy_is_sunk_along_its_own_up_and_sized_inside_its_range() {
		let laid = lay_out(
			&floor(16),
			&Strewing {
				sink: [0.5, 0.5],
				size: [0.25, 0.75],
				shade: 0.5,
				..rule(2.0)
			},
			&Transform::IDENTITY,
			UNIT,
			None,
		);

		for piece in &laid.pieces {
			assert!((piece.at[1] + 0.5).abs() < 1.0e-5, "half a unit into flat ground");
			assert!((0.25..=0.75).contains(&piece.size), "a size of {}", piece.size);
			assert!((0.5..=1.0).contains(&piece.shade), "a shade of {}", piece.shade);
		}

		// and the ranges are used, end to end
		let sizes = laid.pieces.iter().map(|piece| piece.size);
		let shades = laid.pieces.iter().map(|piece| piece.shade);

		assert!(sizes.clone().fold(1.0, f32::min) < 0.26, "the smallest near the least");
		assert!(sizes.fold(0.0, f32::max) > 0.74, "the largest near the most");
		assert!(shades.fold(1.0, f32::min) < 0.51, "the darkest near half");
	}

	#[test]
	fn the_first_part_of_a_patch_is_spread_over_the_whole_of_it() {
		// what a renderer thinning a patch by drawing the first part of its run
		// counts on: the leading quarter stands all over the square, not in the
		// first triangles laid
		let laid = lay_out(&floor(32), &rule(8.0), &Transform::IDENTITY, UNIT, None);

		for patch in &laid.patches {
			let run = &laid.pieces[patch.run()];
			let lead = &run[..run.len() / 4];
			let depth = |pieces: &[Piece]| {
				pieces
					.iter()
					.map(|piece| piece.at[2])
					.sum::<f32>() / f32::from(u16::try_from(pieces.len()).unwrap_or(u16::MAX))
			};

			assert!(run.len() > 400, "{} copies in a patch", run.len());
			assert!(
				(depth(lead) - depth(run)).abs() < 1.0,
				"the leading quarter stands {} deep and the run {}",
				depth(lead),
				depth(run)
			);
		}
	}

	#[test]
	fn a_copy_is_held_the_way_the_entity_holds_one() {
		let held = Quat::from_rotation_z(0.7);
		let laid = lay_out(
			&floor(8),
			&Strewing { turns: 0, ..rule(1.0) },
			&Transform {
				rotation: held,
				scale: Vec3::new(2.0, 3.0, 4.0),
				position: Vec3::new(50.0, 50.0, 50.0),
			},
			UNIT,
			None,
		);

		for piece in &laid.pieces {
			let turn = Quat::from_array(piece.turn);

			assert!(turn.dot(held).abs() > 1.0 - 1.0e-6, "held as the entity is turned");
			assert!(piece.at[0].abs() <= 4.0, "and the entity's own place moves nothing");
		}
	}

	#[test]
	fn patches_are_runs_of_one_square_each_with_a_box_around_every_copy() {
		let held = Vec3::new(1.0, 2.0, 1.0);
		let laid = lay_out(
			&hills(11),
			&Strewing {
				tilt: 40.0,
				size: [0.5, 2.0],
				..rule(1.0)
			},
			&Transform { scale: held, ..Transform::IDENTITY },
			UNIT,
			None,
		);
		let mut next = 0;

		for patch in &laid.patches {
			assert_eq!(patch.run().start, next, "runs follow one another");
			next = patch.run().end;

			let square = patch_of(Vec3::from_array(laid.pieces[patch.run().start].at));

			for piece in &laid.pieces[patch.run()] {
				assert_eq!(patch_of(Vec3::from_array(piece.at)), square, "one square a run");

				assert!(
					boxed(&corners(matrix_of(piece, held)), patch),
					"a corner of a copy outside its patch's box"
				);
				assert!(piece.size * 2.0 <= patch.largest + 1.0e-5, "the largest is the largest");
			}
		}

		assert_eq!(next, laid.pieces.len(), "and every copy is in one");

		let squares: std::collections::HashSet<(i32, i32)> = laid
			.patches
			.iter()
			.map(|patch| patch_of(Vec3::from_array(laid.pieces[patch.run().start].at)))
			.collect();

		assert_eq!(squares.len(), laid.patches.len(), "one run a square");
	}

	#[test]
	fn a_ground_with_no_faces_up_lays_nothing() {
		assert!(
			lay_out(&MeshData::default(), &rule(4.0), &Transform::IDENTITY, UNIT, None)
				.pieces
				.is_empty()
		);

		let mut upside = floor(4);
		for triangle in upside.indices.chunks_exact_mut(3) {
			triangle.swap(1, 2);
		}

		assert!(
			lay_out(&upside, &rule(4.0), &Transform::IDENTITY, UNIT, None)
				.pieces
				.is_empty(),
			"a floor facing down covers no ground seen from above"
		);
	}

	#[test]
	fn the_solid_is_made_of_the_coarsest_level_of_a_mesh_that_has_levels() {
		let mut mesh = cube();

		mesh.levels = vec![
			crate::abi::mesh::Level {
				indices: mesh.indices[..18].to_vec(),
				error: 0.1,
			},
			crate::abi::mesh::Level {
				indices: mesh.indices[..6].to_vec(),
				error: 0.3,
			},
		];

		let laid = lay_out(&floor(4), &rule(1.0), &Transform::IDENTITY, UNIT, None);
		let merged = solid(&laid, &mesh, &Transform::IDENTITY);

		assert!(!laid.pieces.is_empty());
		assert_eq!(
			merged.indices.len(),
			6 * laid.pieces.len(),
			"two triangles a copy, the last level's"
		);
	}

	#[test]
	fn the_solid_is_every_copy_where_it_stands_in_the_entity_space() {
		let mesh = cube();
		let local = Transform {
			position: Vec3::new(3.0, 1.0, -2.0),
			rotation: Quat::from_rotation_y(0.4),
			scale: Vec3::new(0.5, 2.0, 0.5),
		};
		let laid = lay_out(&floor(4), &rule(1.0), &local, UNIT, None);
		let merged = solid(&laid, &mesh, &local);

		assert_eq!(merged.vertices.len(), mesh.vertices.len() * laid.pieces.len());
		assert_eq!(merged.indices.len(), mesh.indices.len() * laid.pieces.len());

		// carried back out through the entity's own transform, each copy's
		// corners are where the copy stands them in the ground's space
		for (index, piece) in laid.pieces.iter().enumerate() {
			let placed = matrix_of(piece, local.scale);

			for (offset, vertex) in mesh.vertices.iter().enumerate() {
				let there = placed.transform_point3(Vec3::from_array(vertex.position));
				let here = local.matrix().transform_point3(Vec3::from_array(
					merged.vertices[index * mesh.vertices.len() + offset].position,
				));

				assert!((there - here).length() < 1.0e-4, "{there} against {here}");
			}
		}

		let flattened = Transform { scale: Vec3::new(1.0, 0.0, 1.0), ..local };

		assert!(
			solid(&laid, &mesh, &flattened)
				.vertices
				.is_empty(),
			"no inside to put it in"
		);
	}

	#[test]
	fn a_record_holds_every_row_it_declares_and_nothing_else() {
		let mut records = crate::abi::Records::new();

		records
			.declare(&STREWING)
			.expect("the record is one a world holds");

		assert_eq!(STREWING.rows.len(), 15, "a row a field");
		assert_eq!(size_of::<Strewing>(), 18 * 4, "eighteen words");
	}

	#[test]
	fn the_table_counts_a_laying_and_a_taking_away() {
		let mut strewn = Strewn::new();
		let key = Key {
			mask: 0,
			rule: Strewing::NONE,
			ground: (MeshId::NONE, 0),
			mesh: (MeshId::NONE, 0),
			local: Transform::IDENTITY,
		};

		strewn.put(3, Layout {
			entity: EntityId::NONE,
			key,
			laid: Laid::default(),
			solid: MeshId::NONE,
			revision: 0,
		});

		assert_eq!(strewn.revision(), 1, "one laying");
		assert_eq!(strewn.get(3).map(|layout| layout.revision), Some(1));
		assert!(strewn.get(2).is_none(), "the slots before it laid nothing");
		assert_eq!(strewn.iter().count(), 1);

		assert!(strewn.take(3).is_some());
		assert!(strewn.take(3).is_none(), "and nothing is left to take");
		assert_eq!(strewn.revision(), 2, "one laying and one taking away");
	}

	#[test]
	fn a_key_is_the_same_as_itself_even_holding_a_number_that_is_not_one() {
		let key = Key {
			mask: 0,
			rule: Strewing { density: f32::NAN, ..Strewing::NONE },
			ground: (MeshId::new(4), 2),
			mesh: (MeshId::new(5), 1),
			local: Transform::IDENTITY,
		};

		assert!(key.same(&key), "a rule with a number that is not one is not laid every step");
		assert!(
			!key.same(&Key { ground: (MeshId::new(4), 3), ..key }),
			"a ground built again is another ground"
		);
		assert!(
			!key.same(&Key { local: Transform::at(Vec3::X), ..key }),
			"an entity moved is another place for its body"
		);
	}

	#[test]
	fn streams_of_neighboring_triangles_start_nothing_alike() {
		let firsts: Vec<u64> = (0..64)
			.map(|index| Random::new(stream(0, index)).draw())
			.collect();

		for pair in firsts.windows(2) {
			let differing = (pair[0] ^ pair[1]).count_ones();

			assert!((12..=52).contains(&differing), "{differing} bits apart");
		}

		// and where a copy lands in one triangle says nothing about where it
		// lands in the next: the first draws of neighboring streams are not
		// correlated beyond what four thousand samples leave
		let firsts: Vec<f64> = (0..4096)
			.map(|index| f64::from(Random::new(stream(0, index)).unit()))
			.collect();
		let mean = firsts.iter().sum::<f64>() / 4096.0;
		let spread = firsts
			.iter()
			.map(|one| (one - mean).powi(2))
			.sum::<f64>();
		let together = firsts
			.windows(2)
			.map(|pair| (pair[0] - mean) * (pair[1] - mean))
			.sum::<f64>();

		assert!(
			(together / spread).abs() < 0.06,
			"neighbors correlated by {}",
			together / spread
		);
	}
	/// Whether two shares are the same number to the bit, which is what every
	/// answer a mask gives has to be: a cell reading open by a hair less than
	/// one would take away a copy whose own draw came out at that hair.
	fn same(held: f32, wanted: f32) -> bool { held.to_bits() == wanted.to_bits() }

	/// A count as the number a laying reports.
	fn counted(many: usize) -> u64 { u64::try_from(many).expect("a count of copies") }

	/// A place as the bits it is written down as, for an exact comparison the
	/// workspace's lints allow.
	fn bits(at: [f32; 3]) -> [u32; 3] { at.map(f32::to_bits) }

	/// A mask over a ground, every cell open.
	fn open(side: f32) -> Mask {
		Mask::over((Vec3::splat(-side * 0.5), Vec3::splat(side * 0.5))).expect("a box is a box")
	}

	#[test]
	fn a_mask_nobody_painted_lays_the_field_no_mask_lays_byte_for_byte() {
		let ground = hills(1971);
		let bare = lay_out(&ground, &rule(2.0), &Transform::IDENTITY, UNIT, None);
		let open = lay_out(&ground, &rule(2.0), &Transform::IDENTITY, UNIT, Some(&open(64.0)));

		assert_eq!(bare.digest, open.digest, "every copy of it, to the bit");
		assert_eq!(bare.pieces, open.pieces, "and the same copies in the same order");
		assert_eq!(open.masked, 0, "nothing was taken away");
		assert_eq!(bare.drawn, open.drawn, "and the rule drew what it would have drawn");
	}

	#[test]
	fn a_mask_painted_shut_lays_nothing_where_it_is_shut_and_moves_nothing_elsewhere() {
		let ground = hills(1971);
		let bare = lay_out(&ground, &rule(2.0), &Transform::IDENTITY, UNIT, None);
		let mut mask = open(64.0);

		// a patch of it shut, hard enough to reach nought at the middle
		for _ in 0..4 {
			mask.paint(Vec2::new(4.0, 4.0), 6.0, 1.0);
		}

		let laid = lay_out(&ground, &rule(2.0), &Transform::IDENTITY, UNIT, Some(&mask));

		assert!(laid.pieces.len() < bare.pieces.len(), "some of the field has gone");
		assert_eq!(laid.drawn, bare.drawn, "and the rule drew every one of them anyway");
		assert_eq!(
			laid.masked,
			counted(bare.pieces.len()) - counted(laid.pieces.len()),
			"what is missing is what the mask took"
		);

		for piece in &laid.pieces {
			let reach = Vec2::new(piece.at[0] - 4.0, piece.at[2] - 4.0).length();

			assert!(reach > 1.0, "a copy at {reach} from the middle of a stroke that shut it");
		}

		// and every copy that is still there is exactly where it was
		let held: Vec<&Piece> = bare
			.pieces
			.iter()
			.filter(|piece| {
				laid.pieces
					.iter()
					.any(|kept| bits(kept.at) == bits(piece.at))
			})
			.collect();

		assert_eq!(held.len(), laid.pieces.len(), "every copy kept was already laid there");

		for (kept, was) in laid.pieces.iter().zip(&held) {
			assert_eq!(&kept, was, "and it is the same copy, turn, size and shade");
		}
	}

	#[test]
	fn a_cell_half_open_keeps_about_half_of_what_stood_on_it() {
		let ground = floor(64);
		let mut mask = open(64.0);

		mask.cells.fill(OPEN / 2);
		mask.restamp();

		let bare = lay_out(&ground, &rule(4.0), &Transform::IDENTITY, UNIT, None);
		let half = lay_out(&ground, &rule(4.0), &Transform::IDENTITY, UNIT, Some(&mask));
		let share = share_of(half.pieces.len(), bare.pieces.len().max(1));

		assert!((share - 0.5).abs() < 0.03, "half a cell kept {share} of the field");
		assert_eq!(
			half.masked + counted(half.pieces.len()),
			bare.drawn,
			"and what is not standing was masked"
		);
	}

	#[test]
	fn painting_one_corner_of_a_ground_moves_no_copy_in_another() {
		let ground = hills(7);
		let bare = lay_out(&ground, &rule(1.0), &Transform::IDENTITY, UNIT, None);
		let mut mask = open(64.0);

		for _ in 0..6 {
			mask.paint(Vec2::new(-20.0, -20.0), 8.0, 1.0);
		}

		let laid = lay_out(&ground, &rule(1.0), &Transform::IDENTITY, UNIT, Some(&mask));
		let far = |piece: &Piece| piece.at[0] > 0.0 && piece.at[2] > 0.0;
		let before: Vec<&Piece> = bare.pieces.iter().filter(|it| far(it)).collect();
		let after: Vec<&Piece> = laid.pieces.iter().filter(|it| far(it)).collect();

		assert!(!before.is_empty(), "there is a far corner to compare");
		assert_eq!(before.len(), after.len(), "the far corner kept every copy");

		for (was, is) in before.iter().zip(&after) {
			assert_eq!(was, is, "and each is the copy it was, to the bit");
		}
	}

	#[test]
	fn a_place_outside_the_grid_is_open_and_a_shut_grid_lays_nothing() {
		let mut mask = open(8.0);

		assert!(same(mask.at(100.0, 0.0), 1.0), "east of the grid");
		assert!(same(mask.at(0.0, -100.0), 1.0), "and north of it");

		mask.cells.fill(0);
		mask.restamp();

		assert!(same(mask.at(0.0, 0.0), 0.0), "and nothing stands inside a shut one");
		assert_eq!(
			lay_out(&floor(8), &rule(8.0), &Transform::IDENTITY, UNIT, Some(&mask))
				.pieces
				.len(),
			0,
			"a rule laid through it lays nothing at all"
		);
	}

	#[test]
	fn a_mask_reads_between_its_cells_rather_than_out_of_the_one_underneath() {
		// two cells, one shut and one open, over four units: read at the two
		// middles it is nought and one, and halfway between them a half
		let mask = Mask::new([0.0, 0.0], 2.0, [2, 1], vec![0, OPEN]).expect("a grid of two");

		assert!(same(mask.at(1.0, 1.0), 0.0), "the middle of the shut cell");
		assert!(same(mask.at(3.0, 1.0), 1.0), "the middle of the open one");
		assert!((mask.at(2.0, 1.0) - 0.5).abs() < 1.0e-6, "and a half between them");
		assert!(same(mask.at(0.5, 1.0), 0.0), "outside the first middle it holds the first cell");
		assert!(same(mask.at(3.5, 1.0), 1.0), "and outside the last, the last");
	}

	#[test]
	fn a_mask_reads_its_two_axes_apart_and_slopes_along_both() {
		// four cells east by two south, each a different number, so that a
		// read with the two axes the wrong way round lands on another cell -
		// and a slope along each axis, so that reading out of one cell rather
		// than between four is another answer
		let cells = vec![0, 85, 170, 255, 255, 170, 85, 0];
		let mask = Mask::new([0.0, 0.0], 2.0, [4, 2], cells).expect("a grid of eight");
		let share = |cell: u8| f32::from(cell) / f32::from(OPEN);

		assert!(same(mask.at(1.0, 1.0), share(0)), "the first cell east, the first south");
		assert!(same(mask.at(7.0, 1.0), share(255)), "the last cell east, the first south");
		assert!(same(mask.at(1.0, 3.0), share(255)), "and the first east, the last south");
		assert!(same(mask.at(7.0, 3.0), share(0)), "and the last of both");

		// halfway between two cells is halfway between their two numbers,
		// along each axis and along both at once
		assert!(same(mask.at(2.0, 1.0), share(42) + 0.5 / f32::from(OPEN)), "halfway east");
		assert!(same(mask.at(1.0, 2.0), share(127) + 0.5 / f32::from(OPEN)), "halfway south");
		// the middle of the four in the middle: halfway along a row that runs
		// up and halfway along one that runs down, which is the middle of both
		assert!(same(mask.at(4.0, 2.0), 0.5), "the middle of the four in the middle");
	}

	#[test]
	fn a_grid_is_as_fine_as_it_can_be_and_never_holds_more_cells_than_one_holds() {
		let small = open(8.0);
		assert!(same(small.step(), CELL), "a small ground has cells of a unit");
		assert_eq!(small.counts(), [8, 8], "one a unit");

		let big = Mask::over((Vec3::splat(-2048.0), Vec3::splat(2048.0))).expect("a box");
		let cells = usize::try_from(big.counts()[0]).expect("a count of cells")
			* usize::try_from(big.counts()[1]).expect("a count of cells");

		assert!(cells <= MOST_CELLS, "{cells} cells, and a mask holds {MOST_CELLS}");
		assert!(big.step() > CELL, "a ground that wide is painted more coarsely");
		assert!(
			big.step() * f32::from(u8::try_from(big.counts()[0] / 64).unwrap_or(1)) > 0.0,
			"and the grid still reaches across it"
		);
	}

	#[test]
	fn a_grid_that_is_not_one_is_refused() {
		assert!(Mask::new([0.0, 0.0], 1.0, [2, 2], vec![OPEN; 3]).is_none(), "too few cells");
		assert!(Mask::new([0.0, 0.0], 1.0, [2, 2], vec![OPEN; 5]).is_none(), "too many");
		assert!(Mask::new([0.0, 0.0], 0.0, [2, 2], vec![OPEN; 4]).is_none(), "a step of nought");
		assert!(Mask::new([0.0, 0.0], -1.0, [2, 2], vec![OPEN; 4]).is_none(), "or below it");
		assert!(
			Mask::new([f32::NAN, 0.0], 1.0, [1, 1], vec![OPEN]).is_none(),
			"a corner nowhere"
		);
		assert!(Mask::new([0.0, 0.0], 1.0, [0, 4], vec![]).is_none(), "a grid of no cells");
		assert!(
			Mask::new([0.0, 0.0], 1.0, [1024, 1024], vec![OPEN; 1024 * 1024]).is_none(),
			"and one past the ceiling"
		);
	}

	#[test]
	fn a_digest_follows_every_cell_and_is_never_nought() {
		let mut mask = open(8.0);
		let was = mask.digest();

		assert_ne!(was, 0, "nought is what a key says for no mask at all");
		mask.cells[3] = 7;
		mask.restamp();

		assert_ne!(mask.digest(), was, "a cell changed is another mask");
		assert_ne!(mask.digest(), 0, "and it is still not nought");

		let moved = Mask::new([1.0, 0.0], 1.0, mask.counts(), mask.cells.clone())
			.expect("the same cells over another corner");

		assert_ne!(moved.digest(), mask.digest(), "the grid is part of it, not only the cells");
	}

	#[test]
	fn a_stroke_and_the_share_it_took_agree() {
		let mut mask = open(16.0);

		assert!(same(mask.painted(), 0.0), "nothing painted");
		assert!(mask.paint(Vec2::ZERO, 4.0, 1.0), "a stroke lands");
		assert!(mask.painted() > 0.0, "and some of the field has gone");
		assert!(mask.painted() < 1.0, "though not all of it");

		assert!(mask.clear(), "and it can be put back");
		assert!(same(mask.painted(), 0.0), "all of it");
		assert!(!mask.clear(), "a mask already open does not move");
		assert!(!mask.paint(Vec2::ZERO, 0.0, 1.0), "nor does a brush of no width");
		assert!(!mask.paint(Vec2::ZERO, 4.0, 0.0), "nor one that paints nothing");
	}

	#[test]
	fn a_wider_ground_grows_the_grid_and_keeps_what_was_painted() {
		let mut mask = open(16.0);

		for _ in 0..8 {
			mask.paint(Vec2::new(-4.0, -4.0), 3.0, 1.0);
		}

		assert!(same(mask.at(-4.0, -4.0), 0.0), "shut where it was painted");
		assert!(
			mask.fitted((Vec3::splat(-8.0), Vec3::splat(8.0)))
				.is_none(),
			"ground it already covers wants no new grid"
		);

		let grown = mask
			.fitted((Vec3::splat(-64.0), Vec3::splat(64.0)))
			.expect("ground past it does");

		assert!(grown.counts()[0] > mask.counts()[0], "the grid reaches further");
		assert!(same(grown.at(-4.0, -4.0), 0.0), "and what was painted is where it was");
		assert!(same(grown.at(60.0, 60.0), 1.0), "with the new ground open");
	}

	#[test]
	fn a_mask_is_read_in_the_grounds_own_space_whatever_holds_the_copies() {
		// the strewing's own turn and scale hold every copy, and they must not
		// reach the mask: a cell is a place on the ground, not on a copy.
		//
		// **The patch is painted somewhere whose two numbers differ**, and the
		// place across the diagonal from it is asked for copies: a laying that
		// read a cell with the ground's two axes the wrong way round would
		// clear the wrong corner, and every round patch at a place like (4, 4)
		// hides that exactly.
		let ground = floor(16);
		let mut mask = open(16.0);
		let (shut, mirrored) = (Vec2::new(6.0, -3.0), Vec2::new(-3.0, 6.0));

		for _ in 0..8 {
			mask.paint(shut, 3.0, 1.0);
		}

		let held = Transform {
			position: Vec3::new(50.0, 9.0, -3.0),
			rotation: Quat::from_rotation_y(1.0),
			scale: Vec3::new(2.0, 0.5, 3.0),
		};
		let laid = lay_out(&ground, &rule(4.0), &held, UNIT, Some(&mask));
		let near = |piece: &Piece, at: Vec2| {
			Vec2::new(piece.at[0] - at.x, piece.at[2] - at.y).length() < 1.0
		};

		assert!(!laid.pieces.is_empty(), "the rest of the ground is still laid");
		assert!(
			!laid.pieces.iter().any(|piece| near(piece, shut)),
			"nothing stands where the mask is shut"
		);
		assert!(
			laid.pieces
				.iter()
				.any(|piece| near(piece, mirrored)),
			"and the place across the diagonal from it is as full as ever"
		);
	}
}
