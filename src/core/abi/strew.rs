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
//! its place in its patch and a draw kept for a mask - so that raising the
//! density adds copies after the ones there were, and a band, a size, a lean or
//! a slope changed in a panel takes away or reshapes the copies it is about and
//! moves no other.
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

/// Lays a rule's copies out over a ground.
///
/// @param ground - the mesh the copies stand on, in its own space; its own
/// triangles and not any coarser level of them
/// @param rule - how they are laid
/// @param local - the strewing entity's own transform, whose turn and scale
/// are how every copy is held
/// @param bounds - the box of what is strewn, in its own space
/// @return every copy, in patches
#[must_use]
pub fn lay_out(
	ground: &MeshData,
	rule: &Strewing,
	local: &Transform,
	bounds: (Vec3, Vec3),
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
		drawn: 0,
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
struct Laying {
	/// The rule, made sane.
	rule: Strewing,

	/// The strewing entity's own transform: how every copy is held.
	local: Transform,

	/// The cosine of the rule's lean at random.
	cosine: f32,

	/// How many copies the rule has drawn.
	drawn: u64,

	/// What has been kept so far, each with its patch and its place in it.
	found: Vec<Found>,
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

impl Laying {
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

			if let Some(found) = self.copy(face, &mut random) {
				self.found.push(found);
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
	/// @return the copy, or nothing for one the band leaves out
	fn copy(&self, face: &Face, random: &mut Random) -> Option<Found> {
		let (mut along, mut across) = (random.unit(), random.unit());
		let yaw = random.circle();
		let lean = random.unit();
		let lean_way = random.circle();
		let grown = random.unit();
		let sunk = random.unit();
		let dark = random.unit();
		let rank = u32::try_from(random.draw() >> 32).unwrap_or(0);
		// kept for a mask the strewing may one day be painted with, so that
		// painting one moves nothing it does not paint away
		let _mask = random.unit();

		// a point in the parallelogram the triangle is half of, folded back
		// into the triangle when it lands in the other half
		if along + across > 1.0 {
			along = 1.0 - along;
			across = 1.0 - across;
		}

		let [first, second, third] = face.corners;
		let point = first + (second - first) * along + (third - first) * across;

		if point.y < self.rule.band[0] || point.y > self.rule.band[1] {
			return None;
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

		Some(Found {
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
fn digest(words: impl Iterator<Item = u32>) -> u64 {
	words.fold(0xCBF2_9CE4_8422_2325, |held, word| {
		word.to_le_bytes()
			.iter()
			.fold(held, |held, byte| (held ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01B3))
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
		let laid = lay_out(&floor(32), &rule(3.0), &Transform::IDENTITY, UNIT);

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
		let laid = lay_out(&ground, &rule(2.0), &Transform::IDENTITY, UNIT);
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
		let one = lay_out(&ground, &rule(1.0), &Transform::IDENTITY, UNIT);
		let two = lay_out(&ground, &rule(1.0), &Transform::IDENTITY, UNIT);
		let other =
			lay_out(&ground, &Strewing { seed: 1, ..rule(1.0) }, &Transform::IDENTITY, UNIT);

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
		);

		assert_eq!(laid.pieces.len(), 7815, "{}", laid.pieces.len());
		assert_eq!(laid.digest, 0x5AF8_8355_70F4_A0AA, "{:#018X}", laid.digest);
	}

	#[test]
	fn a_slope_leaves_out_what_is_steeper_and_moves_nothing_it_keeps() {
		let ground = hills(1971);
		let everywhere = lay_out(&ground, &rule(2.0), &Transform::IDENTITY, UNIT);
		let gentle =
			lay_out(&ground, &Strewing { slope: 8.0, ..rule(2.0) }, &Transform::IDENTITY, UNIT);
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
		let sparse = lay_out(&ground, &rule(1.0), &Transform::IDENTITY, UNIT);
		let dense = lay_out(&ground, &rule(4.0), &Transform::IDENTITY, UNIT);

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
		let everywhere = lay_out(&ground, &rule(2.0), &Transform::IDENTITY, UNIT);
		let banded = lay_out(
			&ground,
			&Strewing { band: [-1.0, 1.0], ..rule(2.0) },
			&Transform::IDENTITY,
			UNIT,
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
		let laid = lay_out(&floor(64), &rule(100.0), &Transform::IDENTITY, UNIT);

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
		let laid =
			lay_out(&hills(3), &Strewing { turns: 0, ..rule(0.5) }, &Transform::IDENTITY, UNIT);

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
			let laid =
				lay_out(&ground, &Strewing { align, ..rule(1.0) }, &Transform::IDENTITY, UNIT);

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
		let laid = lay_out(&sloped(16), &rule(2.0), &Transform::IDENTITY, UNIT);

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
		let laid = lay_out(&floor(32), &rule(8.0), &Transform::IDENTITY, UNIT);

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
			lay_out(&MeshData::default(), &rule(4.0), &Transform::IDENTITY, UNIT)
				.pieces
				.is_empty()
		);

		let mut upside = floor(4);
		for triangle in upside.indices.chunks_exact_mut(3) {
			triangle.swap(1, 2);
		}

		assert!(
			lay_out(&upside, &rule(4.0), &Transform::IDENTITY, UNIT)
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

		let laid = lay_out(&floor(4), &rule(1.0), &Transform::IDENTITY, UNIT);
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
		let laid = lay_out(&floor(4), &rule(1.0), &local, UNIT);
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
}
