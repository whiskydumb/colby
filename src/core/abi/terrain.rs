//! A shape of ground, as a description an entity carries.
//!
//! One record with a kind, sitting in a per-slot array on
//! [`Entities`](super::Entities), exactly as [`Light`](super::light) and
//! [`Emitter`](super::particles::Emitter) do - so a terrain hangs off a
//! parent, moves with a gizmo, is written down by a scene and comes back from
//! an undo without any of those knowing what a height field is. The four
//! engines read for this all make terrain a thing in the world rather than a
//! field on it: Unreal an `ALandscape` actor, s&box a `Terrain : Collider`
//! component, Fyrox a scene node, Wicked a struct that owns entities. **Two
//! of the four hold several in one scene on purpose** - `Terrain.cs:53` asks
//! whether any *other* terrain is still active - so "there is only ever one"
//! is not a fact about the field.
//!
//! **What the description says, and what it does not.** It says how big the
//! ground is, how high, how finely it is measured and what shape the land
//! takes. It does not say where the vertices are: the mesh is built from these
//! numbers by `colby_runtime::terrain` and registered under a name, exactly
//! the way an `.obj` would have been, and from that moment the renderer, the
//! solver, the shadow pass and the saves all treat it as ordinary geometry.
//! That is the whole of why this file has no rendering and no collision in it.
//!
//! **There are no chunks, and that is a decision rather than an omission.**
//! Every engine here chunks its terrain, and every one of them does it for
//! level of detail, streaming, or frustum culling: Fyrox walks a quadtree,
//! s&box builds clipmap rings around the camera, Wicked streams chunks in and
//! out by distance, Unreal streams components. colby has none of the three -
//! `colby_engine::Scene` draws every entity that has a mesh, in view or not -
//! so chunking here would turn one draw call into sixty-four and buy nothing
//! back. What chunking *would* have bought is collision, because a triangle
//! soup is tested one triangle at a time; that is bought instead by the grid
//! inside `colby_physics`'s collider, which speeds up every mesh in the world
//! rather than only this one.
//!
//! **The heights are generated rather than read.** All four engines store real
//! heights, and all four store them at sixteen bits or better - Unreal packs a
//! `uint16` across the red and green channels of its heightmap
//! (`LandscapeDataAccess.h:40-51`), s&box keeps a `ushort[]` in a `.terrain`
//! file, Wicked a `vector<uint16_t>` per chunk, Fyrox a whole `R32F` texture.
//! colby's [`Texel`](super::texture::Texel) is eight bits a channel and
//! nothing else, so a `.png` through the importer it already has would give
//! two hundred and fifty-six levels and visibly terrace anything with real
//! relief. A heights asset is a card of its own; until it is written, the
//! shape comes from noise, which is Wicked's own answer and the only one of
//! the four you can use without a sculpting brush.
//! [`TerrainKind`] is a word so that `heightmap` can join it later without
//! moving anything - the rule [`WaterKind`](super::water::WaterKind) and
//! [`SparkBlend`](super::particles::SparkBlend) already keep.

use super::{
	field::{Field, Kind, Value, field, word},
	mesh::{MeshData, MeshVertex, tangents},
};
use crate::glam::{Vec2, Vec3};

/// The fewest vertices a terrain may have on a side.
///
/// Two, which is a single quad: the smallest thing that is still a surface.
pub const MIN_SIDE: u32 = 2;

/// The most it may have.
///
/// Five hundred and thirteen, which is five hundred and twelve quads and
/// **five hundred and twenty-four thousand two hundred and eighty-eight
/// triangles**. The costs that ceiling is chosen against, all three of them
/// paid once when the mesh is built: about twelve and a half megabytes of
/// vertex buffer at forty-eight bytes each, about two megabytes of indices,
/// and about nineteen megabytes of the solver's own copy of the triangles.
///
/// It is a ceiling rather than a budget: the field's own resolutions are
/// smaller per chunk and unbounded in total - s&box starts at five hundred and
/// twelve across the whole terrain (`TerrainStorage.cs`, `SetResolution(512)`)
/// and Fyrox at two hundred and fifty-nine per chunk. What colby cannot do
/// yet is the *total*, because it has no streaming; so the honest limit is one
/// mesh's worth, and this is it.
pub const MAX_SIDE: u32 = 513;

/// How many vertices a side a terrain has unless it says otherwise.
///
/// A hundred and twenty-nine, which is a hundred and twenty-eight quads: at
/// the default [`SIZE`] that is one cell per world unit, and thirty-two
/// thousand seven hundred and sixty-eight triangles.
pub const SIDE: u32 = 129;

/// How wide a terrain is unless it says otherwise, in world units.
pub const SIZE: f32 = 128.0;

/// How far a terrain rises and falls unless it says otherwise.
///
/// Eight against a hundred and twenty-eight across, which is a rolling field
/// rather than mountains: a slope somebody can walk up, so that the first
/// thing anybody builds can be stood on rather than looked at.
pub const HEIGHT: f32 = 8.0;

/// How many hills fit across a terrain unless it says otherwise.
pub const FREQUENCY: f32 = 3.0;

/// How many octaves of noise are summed unless it says otherwise.
///
/// Four. Each one doubles the frequency and multiplies the amplitude by
/// [`ROUGHNESS`], so four covers hills, banks, bumps and grain.
pub const OCTAVES: u32 = 4;

/// The most octaves that may be asked for.
///
/// Eight, past which the eighth octave's wavelength is smaller than one cell
/// at the default resolution and the sum stops being a landscape and starts
/// being noise on top of one.
pub const MAX_OCTAVES: u32 = 8;

/// How much of an octave's amplitude the next one keeps, unless it says
/// otherwise.
pub const ROUGHNESS: f32 = 0.5;

/// How many world units of ground one repeat of the texture covers, unless it
/// says otherwise.
pub const TILING: f32 = 8.0;

/// What the lattice is hashed with.
///
/// The golden ratio's reciprocal in sixty-four bits, the same odd multiplier
/// `colby_runtime::sparks` seeds a step with, and the same one
/// [`Random`](crate::random::Random) wakes from a nil seed with. One constant
/// for every place in this workspace that has to turn a coordinate into a
/// number nobody can predict.
const MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// A second odd multiplier, for the other axis.
///
/// Two of them rather than one so that `(x, z)` and `(z, x)` are different
/// lattice points; with one multiplier they would hash to the same value and
/// every terrain would come out symmetric about its diagonal.
const MIX_Z: u64 = 0x9E37_79B9_7F4A_7C15_u64.rotate_left(31);

/// How many of the hash's bits become the sample.
///
/// Twenty-four, taken off the top: the low bits of a multiply are the ones the
/// multiply does not reach, which is the same reason
/// [`Random::below`](crate::random::Random::below) reads the top of a wide
/// product rather than a remainder.
const SAMPLE_BITS: u32 = 24;

/// What a full sample is worth, as a float.
const SAMPLE_SPAN: f32 = 16_777_216.0;

/// One row for a whole number that is not signed.
///
/// A [`Value::Int`] is signed and sixty-four bits wide and these are neither,
/// so the two conversions are written where they can refuse a number that will
/// not fit - the shape [`Emitter::FIELDS`](super::particles::Emitter::FIELDS)
/// gives its `cap` by hand, factored because a terrain has three of them.
///
/// @param $low - the smallest the field may hold
/// @param $high - the largest
macro_rules! count {
	($name:literal, $path:ident, $low:expr, $high:expr, $help:literal) => {
		Field {
			name: $name,
			help: $help,
			kind: Kind::Int,
			get: |terrain: &Terrain| Value::Int(i64::from(terrain.$path)),
			set: |terrain: &mut Terrain, value| match value {
				| Value::Int(held) => match u32::try_from(held) {
					| Ok(count) => {
						terrain.$path = count.clamp($low, $high);

						true
					},
					// a negative count is a refusal rather than a nought: "no
					// ground" is spelled with a kind, and somebody who typed
					// minus one meant something this cannot do.
					| Err(_) => false,
				},
				| _ => false,
			},
		}
	};
}

/// What shape of ground an entity carries.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TerrainKind {
	/// None at all: the entity is not ground.
	///
	/// A variant rather than an `Option`, for
	/// [`LightKind::None`](super::light::LightKind::None)'s reason: every slot
	/// of the table holds one of these, and a record that is written down
	/// needs a spelling for "no terrain" as much as for the other.
	#[default]
	None,

	/// Summed value noise, from the seed and the four numbers below it.
	Noise,
}

impl TerrainKind {
	/// The word each kind is written as, in declaration order.
	///
	/// The word that is coming is `heightmap`, and it is coming behind an
	/// asset that does not exist yet. @ref the module's own documentation.
	pub const WORDS: &[&str] = &["none", "noise"];

	/// The kind at a place in [`WORDS`](Self::WORDS), if there is one.
	///
	/// @param index - the place
	#[must_use]
	pub const fn at(index: u32) -> Option<Self> {
		match index {
			| 0 => Some(Self::None),
			| 1 => Some(Self::Noise),
			| _ => None,
		}
	}

	/// Where this kind is in [`WORDS`](Self::WORDS).
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "the discriminant is the place in the list, by declaration order"
	)]
	pub const fn index(self) -> u32 { self as u32 }

	/// Whether this kind is ground at all.
	#[must_use]
	pub const fn is_ground(self) -> bool { matches!(self, Self::Noise) }
}

/// The ground an entity is.
///
/// Sixteen fields including the kind, and not one of them is a handle: what
/// the ground is made of is the entity's own
/// [`Renderable::material`](super::Renderable::material), because a terrain is
/// drawn by exactly the path a crate is and had no reason to learn a second
/// way of naming a surface. That is the one place this differs from
/// [`Emitter`](super::particles::Emitter), which needed a picture of its own
/// because a billboard reads none of a material's fields.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Terrain {
	/// What shape of ground this is, or [`TerrainKind::None`].
	pub kind: TerrainKind,

	/// How wide the ground is on each side, in world units.
	///
	/// Square, and one number rather than two: every terrain in the four
	/// engines read is square by construction, and a rectangle is two entities
	/// side by side.
	pub size: f32,

	/// How far the ground rises and falls from end to end, in world units.
	///
	/// The whole span rather than an amplitude, so that halving it halves what
	/// somebody sees. The surface is centered on the entity's own height: the
	/// mean of the noise is nil, so a terrain at the origin has as much ground
	/// above it as below.
	pub height: f32,

	/// How many vertices there are on each side.
	///
	/// Held inside [`MIN_SIDE`] ..= [`MAX_SIDE`] twice over: by the field row,
	/// so that an inspector dragging through nil is clamped rather than
	/// refused, and again by [`vertices_across`](Self::vertices_across),
	/// because a file is not written through a field row and may say anything.
	pub side: u32,

	/// What the noise is seeded with.
	///
	/// Any number at all. Two terrains with the same seed and the same four
	/// numbers below are the same ground, in this process and in every other -
	/// which is what lets a screenshot of one be compared byte for byte.
	pub seed: u32,

	/// How many hills fit across [`size`](Self::size), at the first octave.
	pub frequency: f32,

	/// How many octaves are summed.
	///
	/// Each one is twice the frequency of the last and
	/// [`roughness`](Self::roughness) of its amplitude. Held between one and
	/// [`MAX_OCTAVES`] the two ways [`side`](Self::side) is.
	pub octaves: u32,

	/// How much of an octave's amplitude the next one keeps.
	///
	/// Nought leaves one smooth swell; a half is the usual landscape; one
	/// makes every octave as loud as the first, which is gravel.
	pub roughness: f32,

	/// How many world units of ground one repeat of the texture covers.
	///
	/// The terrain's own tiling rather than the material's, because the
	/// material is shared with everything else wearing it and the ground is
	/// the one surface whose size is a property of the world rather than of
	/// the mesh. Nil or less means one repeat over the whole terrain.
	pub tiling: f32,

	/// Whether anything can stand on it.
	///
	/// The body is made and unmade with this. s&box and Wicked both publish
	/// the same switch (`EnableCollision`, `FLAGS::PHYSICS`) and for the same
	/// reason: a terrain that is only a backdrop should not be paying for a
	/// collision mesh.
	pub solid: bool,
}

impl Terrain {
	/// Its fields, for an inspector, a reader and a writer. @ref
	/// [`field`](super::field).
	pub const FIELDS: &[Field<Self>] = &[
		word!(
			"kind",
			kind,
			TerrainKind::WORDS,
			TerrainKind::at,
			TerrainKind::index,
			"what shape of ground this is, or none"
		),
		field!(Float, "size", size, "how wide the ground is on a side, in world units"),
		field!(Float, "height", height, "how far it rises and falls from end to end"),
		count!("side", side, MIN_SIDE, MAX_SIDE, "how many vertices there are on each side"),
		count!("seed", seed, 0, u32::MAX, "what the noise is seeded with"),
		field!(Float, "frequency", frequency, "how many hills fit across it"),
		count!("octaves", octaves, 1, MAX_OCTAVES, "how many octaves of noise are summed"),
		field!(Float, "roughness", roughness, "how loud each octave is against the last"),
		field!(Float, "tiling", tiling, "how many units of ground one texture repeat covers"),
		field!(Bool, "solid", solid, "whether anything can stand on it"),
	];
	/// No ground at all.
	///
	/// The nine numbers are still a landscape somebody would recognize, so
	/// that turning `kind` from `none` to `noise` in an inspector makes ground
	/// rather than a flat sheet at the origin. The rule
	/// [`Water::NONE`](super::water::Water::NONE) keeps for its three.
	pub const NONE: Self = Self {
		kind: TerrainKind::None,
		size: SIZE,
		height: HEIGHT,
		side: SIDE,
		seed: 0,
		frequency: FREQUENCY,
		octaves: OCTAVES,
		roughness: ROUGHNESS,
		tiling: TILING,
		solid: true,
	};

	/// The defaults, made ground.
	#[must_use]
	pub const fn hills() -> Self { Self { kind: TerrainKind::Noise, ..Self::NONE } }

	/// Ground from a seed, otherwise the defaults.
	///
	/// @param seed - what to shape it with
	#[must_use]
	pub const fn of(seed: u32) -> Self { Self { seed, ..Self::hills() } }

	/// Whether this describes any ground at all.
	#[must_use]
	pub const fn is_ground(self) -> bool { self.kind.is_ground() }

	/// How many vertices a side this really builds.
	///
	/// @return [`side`](Self::side) held inside [`MIN_SIDE`] ..= [`MAX_SIDE`]
	#[must_use]
	pub const fn vertices_across(self) -> u32 {
		if self.side < MIN_SIDE {
			MIN_SIDE
		} else if self.side > MAX_SIDE {
			MAX_SIDE
		} else {
			self.side
		}
	}

	/// How many quads a side this really builds.
	#[must_use]
	pub const fn cells_across(self) -> u32 { self.vertices_across() - 1 }

	/// How many triangles the mesh will have.
	#[must_use]
	pub fn triangles(self) -> u64 {
		let cells = u64::from(self.cells_across());

		cells * cells * 2
	}

	/// How high the ground stands at a point of it, in the entity's own space.
	///
	/// Defined over the whole plane rather than only over the terrain, so that
	/// a caller asking about a point off the edge gets the landscape's own
	/// continuation rather than a refusal - which is what makes the normals at
	/// the border right and would make two terrains laid side by side meet.
	///
	/// @param at - where to ask, in the entity's own space; `y` is ignored
	/// @return how high the surface is there, nil for a terrain that is none
	#[must_use]
	pub fn height_at(self, at: Vec2) -> f32 {
		if !self.is_ground() {
			return 0.0;
		}

		let span = if self.size.abs() > f32::EPSILON {
			self.size
		} else {
			SIZE
		};
		let share = at / span;
		let octaves = self.octaves.clamp(1, MAX_OCTAVES);
		let mut frequency = self.frequency.max(0.0);
		let mut amplitude = 1.0_f32;
		let mut total = 0.0_f32;
		let mut loudest = 0.0_f32;

		for octave in 0..octaves {
			total = amplitude.mul_add(noise(share * frequency, self.seed, octave), total);
			loudest += amplitude;
			frequency *= 2.0;
			amplitude *= self.roughness.clamp(0.0, 1.0);
		}

		if loudest <= f32::EPSILON {
			return 0.0;
		}

		total / loudest * self.height * 0.5
	}

	/// Which way the surface faces at a point of it, in the entity's own
	/// space.
	///
	/// A central difference over the field rather than an average of the
	/// faces around a vertex, which is what Fyrox's own terrain shader does
	/// (`terrain.shader`, four `textureLoad`s and one cross) and is both
	/// cheaper and smoother: it is the analytic normal of the surface the
	/// heights describe, not of the triangles that approximate it.
	///
	/// @param at - where to ask, in the entity's own space
	/// @param step - how far apart the two samples on each axis are; the cell
	/// size is what the mesh builder passes
	#[must_use]
	pub fn normal_at(self, at: Vec2, step: f32) -> Vec3 {
		let step = if step.abs() > f32::EPSILON { step.abs() } else { 1.0 };
		let along_x =
			self.height_at(at + Vec2::new(step, 0.0)) - self.height_at(at - Vec2::new(step, 0.0));
		let along_z =
			self.height_at(at + Vec2::new(0.0, step)) - self.height_at(at - Vec2::new(0.0, step));

		Vec3::new(-along_x, 2.0 * step, -along_z).normalize_or(Vec3::Y)
	}

	/// The geometry this describes.
	///
	/// Centered on the entity's origin, so that a terrain placed somewhere is
	/// centered there and turning it turns it about its middle - which is what
	/// [`GpuMesh`]'s sort key and the solver's bounds both read as the middle
	/// anyway.
	///
	/// @return the mesh, or an empty one for a terrain that is none
	///
	/// [`GpuMesh`]: https://docs.rs/colby_engine
	#[must_use]
	pub fn build(self) -> MeshData {
		if !self.is_ground() {
			return MeshData::default();
		}

		let wide = self.vertices_across();
		let cells = self.cells_across();
		let span = self.size;
		let step = span / across(cells);
		let corner = -span * 0.5;
		let repeat = if self.tiling.abs() > f32::EPSILON {
			span / self.tiling
		} else {
			1.0
		};

		let corners = usize::try_from(wide).unwrap_or(0);
		let mut mesh = MeshData {
			vertices: Vec::with_capacity(corners * corners),
			indices: Vec::with_capacity(corners.saturating_sub(1).pow(2) * 6),
			skin: Vec::new(),
		};

		for row in 0..wide {
			let z = step.mul_add(across(row), corner);

			for column in 0..wide {
				let x = step.mul_add(across(column), corner);
				let flat = Vec2::new(x, z);
				let uv = Vec2::new(across(column) / across(cells), across(row) / across(cells))
					* repeat;

				mesh.vertices.push(MeshVertex::new(
					Vec3::new(x, self.height_at(flat), z),
					self.normal_at(flat, step),
					uv,
				));
			}
		}

		for row in 0..cells {
			for column in 0..cells {
				let low = row * wide + column;
				let high = low + wide;

				// counter-clockwise seen from above, which is the winding
				// `mesh::push_face` produces and the one the scene's pipelines
				// cull the back of.
				mesh.indices
					.extend_from_slice(&[low, high, low + 1, low + 1, high, high + 1]);
			}
		}

		tangents(&mut mesh);

		mesh
	}
}

impl Default for Terrain {
	fn default() -> Self { Self::NONE }
}

/// One octave of value noise at a point of the lattice.
///
/// Value noise rather than gradient noise, and the difference is one line: a
/// lattice point's value is a hash of its coordinates, where Perlin's would be
/// a hashed *direction* dotted with the offset. Wicked's terrain uses Perlin
/// (`wi::noise::Perlin`); at the amplitudes a landscape is summed at the two
/// are not tellable apart, and a hash needs no gradient table to get wrong.
///
/// The interpolation is the smoothstep `3t^2 - 2t^3`, which is what makes the
/// first derivative continuous across a cell boundary - without it every
/// lattice line would be a visible crease in the shading, because the normal
/// is the derivative.
///
/// @param at - where to sample, in lattice units
/// @param seed - the terrain's seed
/// @param octave - which octave this is, so that two octaves of the same
/// terrain are different noise rather than the same noise twice
/// @return somewhere in `-1.0 ..= 1.0`
fn noise(at: Vec2, seed: u32, octave: u32) -> f32 {
	let cell = at.floor();
	let inside = at - cell;
	let (x, z) = (cell.x, cell.y);
	let smooth = inside * inside * (Vec2::splat(3.0) - 2.0 * inside);

	let low = lerp(lattice(x, z, seed, octave), lattice(x + 1.0, z, seed, octave), smooth.x);
	let high = lerp(
		lattice(x, z + 1.0, seed, octave),
		lattice(x + 1.0, z + 1.0, seed, octave),
		smooth.x,
	);

	lerp(low, high, smooth.y)
}

/// The value at one lattice point.
///
/// @param x - the lattice column, as a whole number in a float
/// @param z - the lattice row
/// @param seed - the terrain's seed
/// @param octave - which octave is asking
/// @return somewhere in `-1.0 ..= 1.0`
fn lattice(x: f32, z: f32, seed: u32, octave: u32) -> f32 {
	let mut hash = (whole(x))
		.wrapping_mul(MIX)
		.wrapping_add((whole(z)).wrapping_mul(MIX_Z))
		.wrapping_add(u64::from(seed).wrapping_mul(MIX))
		.wrapping_add(u64::from(octave).wrapping_mul(MIX_Z));

	// the avalanche of a shift-register step, so that two lattice points one
	// apart do not differ in one bit of the answer.
	hash ^= hash >> 33;
	hash = hash.wrapping_mul(MIX);
	hash ^= hash >> 29;
	hash = hash.wrapping_mul(MIX_Z);
	hash ^= hash >> 32;

	let sample = hash >> (u64::BITS - SAMPLE_BITS);

	(sampled(sample) / SAMPLE_SPAN).mul_add(2.0, -1.0)
}

/// A whole number in a float, as bits a hash can chew.
///
/// Wrapping rather than saturating: a coordinate this far out is not a place
/// anybody stands, and what matters is that the same coordinate always gives
/// the same number.
///
/// @param value - a whole number, as [`f32::floor`] left it
fn whole(value: f32) -> u64 {
	let clamped = value.clamp(-1.0e9, 1.0e9);

	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		reason = "clamped to a span an i64 holds exactly, and the wrap is the point"
	)]
	let signed = clamped as i64;

	#[expect(
		clippy::as_conversions,
		clippy::cast_sign_loss,
		reason = "the bits are what is wanted, not the value"
	)]
	let bits = signed as u64;

	bits
}

/// One number a share of the way to another.
fn lerp(from: f32, to: f32, share: f32) -> f32 { (to - from).mul_add(share, from) }

/// A count of vertices as a float.
///
/// Through a `u16`, which is exactly what [`mesh::tangents`]'s own helper does
/// and for the same reason: this workspace refuses `as`, and [`MAX_SIDE`] is
/// five hundred and thirteen.
///
/// @param count - how many
fn across(count: u32) -> f32 { f32::from(u16::try_from(count).unwrap_or(u16::MAX)) }

/// A sample of [`SAMPLE_BITS`] bits as a float.
///
/// Split rather than converted, for [`across`]'s reason. The high byte and the
/// low sixteen bits are each exact in an `f32`, and so is their sum.
///
/// @param sample - twenty-four bits, as [`lattice`] shifted them down
fn sampled(sample: u64) -> f32 {
	let high = u8::try_from((sample >> 16) & 0xFF).unwrap_or(u8::MAX);
	let low = u16::try_from(sample & 0xFFFF).unwrap_or(u16::MAX);

	f32::from(high) * 65_536.0 + f32::from(low)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_terrain_that_is_none_builds_nothing() {
		let mesh = Terrain::NONE.build();

		assert!(mesh.is_empty(), "no ground is no geometry");
		assert!(mesh.vertices.is_empty(), "and no vertices either");
	}

	#[test]
	fn a_terrain_that_is_none_stands_at_nought_everywhere() {
		let flat = Terrain::NONE;

		assert!(flat.height_at(Vec2::new(3.0, -7.0)).abs() < f32::EPSILON);
		assert_eq!(flat.normal_at(Vec2::ZERO, 1.0), Vec3::Y, "and faces straight up");
	}

	#[test]
	fn the_mesh_has_a_vertex_per_lattice_point_and_two_triangles_per_cell() {
		let ground = Terrain { side: 5, ..Terrain::hills() };
		let mesh = ground.build();

		assert_eq!(mesh.vertices.len(), 25, "five by five");
		assert_eq!(mesh.triangles(), 32, "four by four cells, two triangles each");
		assert_eq!(u64::try_from(mesh.triangles()).unwrap_or(0), ground.triangles());
		assert!(mesh.indices_are_in_range(), "and every index addresses one of them");
	}

	#[test]
	fn the_same_seed_is_the_same_ground() {
		let one = Terrain::of(7).build();
		let other = Terrain::of(7).build();

		assert_eq!(one, other, "a seed is the whole of what shapes it");
	}

	#[test]
	fn a_different_seed_is_different_ground() {
		let one = Terrain::of(7).build();
		let other = Terrain::of(8).build();

		assert_ne!(one, other, "or the seed would not be a knob");
	}

	#[test]
	fn the_ground_is_not_symmetric_about_its_diagonal() {
		// the bug this catches: one multiplier in the lattice hash makes
		// `(x, z)` and `(z, x)` the same point, and every terrain comes out
		// mirrored about a line nobody drew.
		let ground = Terrain::of(3);
		let one = ground.height_at(Vec2::new(11.0, -29.0));
		let other = ground.height_at(Vec2::new(-29.0, 11.0));

		assert!((one - other).abs() > 1.0e-4, "{one} and {other} are the same height");
	}

	#[test]
	fn the_ground_stays_inside_the_height_it_was_given() {
		let ground = Terrain { height: 10.0, ..Terrain::of(11) };

		for step in 0..400 {
			let along = f32::from(i16::try_from(step).unwrap_or(0)) * 0.7 - 140.0;
			let high = ground.height_at(Vec2::new(along, along * 0.37));

			assert!(high.abs() <= 5.0 + 1.0e-3, "{high} is outside half of ten");
		}
	}

	#[test]
	fn the_surface_is_continuous_across_a_lattice_line() {
		// smoothstep rather than a straight lerp is what makes this hold for
		// the *normal* as well as for the height; without it the crease is
		// invisible in the geometry and obvious in the shading.
		let ground = Terrain::of(5);
		let before = ground.height_at(Vec2::new(-0.001, 0.3));
		let after = ground.height_at(Vec2::new(0.001, 0.3));

		assert!((before - after).abs() < 1.0e-3, "{before} and {after} across a cell edge");
	}

	#[test]
	fn a_flat_terrain_faces_straight_up_and_a_slope_does_not() {
		let flat = Terrain { height: 0.0, ..Terrain::hills() };

		assert_eq!(flat.normal_at(Vec2::ZERO, 1.0), Vec3::Y, "no relief, no slope");

		let hilly = Terrain::of(2);
		let somewhere = hilly.normal_at(Vec2::new(4.0, 9.0), 1.0);

		assert!(somewhere.y > 0.0, "the ground never faces downwards");
		assert!(
			somewhere.x.abs() + somewhere.z.abs() > 1.0e-3,
			"and a real landscape leans somewhere: {somewhere:?}"
		);
	}

	#[test]
	fn the_mesh_is_centered_on_the_entity() {
		let mesh = Terrain { size: 20.0, side: 3, ..Terrain::hills() }.build();
		let (low, high) = mesh.bounds();

		assert!((low.x + 10.0).abs() < 1.0e-4, "{low:?} starts ten to the left");
		assert!((high.x - 10.0).abs() < 1.0e-4, "{high:?} ends ten to the right");
		assert!((low.z + 10.0).abs() < 1.0e-4, "and the same across");
		assert!((high.z - 10.0).abs() < 1.0e-4);
	}

	#[test]
	fn the_resolution_is_held_inside_its_bounds() {
		let tiny = Terrain { side: 0, ..Terrain::hills() };
		let huge = Terrain { side: 100_000, ..Terrain::hills() };

		assert_eq!(tiny.vertices_across(), MIN_SIDE, "nil is still a quad");
		assert_eq!(huge.vertices_across(), MAX_SIDE, "and a typo is not a gigabyte");
		assert_eq!(tiny.build().vertices.len(), 4, "which is what gets built");
	}

	#[test]
	fn the_tiling_is_measured_in_world_units() {
		let mesh = Terrain {
			size: 64.0,
			tiling: 8.0,
			side: 3,
			..Terrain::hills()
		}
		.build();
		let far = mesh.vertices.last().expect("a mesh with corners");

		assert!((far.uv[0] - 8.0).abs() < 1.0e-4, "sixty-four units at eight is eight repeats");
	}

	#[test]
	fn the_words_and_the_kinds_agree() {
		assert_eq!(TerrainKind::WORDS.len(), 2, "two spellings, two kinds");
		for (index, word) in TerrainKind::WORDS.iter().enumerate() {
			let index = u32::try_from(index).expect("a short list");
			let kind = TerrainKind::at(index).expect("a word this list holds");

			assert_eq!(kind.index(), index, "{word} reads back where it was written");
		}
		assert_eq!(TerrainKind::at(2), None, "and nothing past the end");
	}

	#[test]
	fn every_field_reads_back_what_was_written() {
		for entry in Terrain::FIELDS {
			let mut record = Terrain::NONE;
			let value = match entry.kind {
				| Kind::Bool => Value::Bool(false),
				| Kind::Int => Value::Int(3),
				| Kind::Float => Value::Float(2.5),
				| Kind::Word(_) => Value::Word(1),
				| _ => panic!("{} is a kind a terrain does not have", entry.name),
			};

			assert!((entry.set)(&mut record, value.clone()), "{} takes its own kind", entry.name);
			assert_eq!((entry.get)(&record), value, "{} reads back", entry.name);
		}
	}

	#[test]
	fn none_is_the_default() {
		assert_eq!(Terrain::default(), Terrain::NONE, "an entity is not ground");
		assert!(!Terrain::NONE.is_ground());
		assert!(Terrain::hills().is_ground(), "and turning the word on makes it so");
	}

	#[test]
	fn turning_the_word_on_leaves_a_landscape_rather_than_a_sheet() {
		let mut record = Terrain::NONE;
		let kind = Terrain::FIELDS
			.iter()
			.find(|entry| entry.name == "kind")
			.expect("a terrain has a kind");

		assert!((kind.set)(&mut record, Value::Word(TerrainKind::Noise.index())));
		assert!(record.height.abs() > f32::EPSILON, "the numbers survived being none");
		assert!(!record.build().is_empty(), "so there is ground the moment it is asked for");
	}
}
