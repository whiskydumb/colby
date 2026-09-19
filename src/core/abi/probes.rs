//! What a bake kept of the light arriving at points of the air, for everything
//! it kept no picture of.
//!
//! **A grid of probes, each six numbers a color.** A probe stands at the middle
//! of a cell of a regular grid over the box every still thing fits in, and
//! keeps the light arriving at that point from each of six half-spaces - the
//! one about each way along each axis - averaged the way a surface facing that
//! way would weigh it: the cosine of each direction with the axis. That is the
//! unit [`World::ambient`](super::World::ambient) means, and the unit the
//! lightmap keeps, so a probe's face is what a surface facing along that axis
//! at that point would read off a lightmap if it had a place on one. A surface
//! facing any other way reads the three faces its normal leans towards, each
//! weighed by the square of the normal's part along that axis, which add to
//! one.
//!
//! **What reads them is what has no place on the lightmap**: a thing a body
//! moves, a thing bones bend, glass, a thing a bake was told to leave out, a
//! still thing whose mesh has no second set. The lightmap is the better answer
//! for a surface that stands still, and nothing that has one reads these.
//!
//! **Kept as a picture**, beside the lightmap, in the order [`Grid::texel`]
//! says: the six faces side by side, each as wide as the grid is along x, and
//! the grid's layers along y one under another, each as tall as the grid is
//! along z. A flat picture of one level is what the compiler already makes of
//! a bake's light, and what the renderer already reads one of.

use super::texture::TextureId;
use crate::glam::Vec3;

/// How many faces a probe keeps.
pub const FACES: usize = 6;

/// The way each face looks, in the order a probe keeps them: `+x -x +y -y +z
/// -z`, the order a cube's faces are kept in.
pub const AXES: [Vec3; FACES] =
	[Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];

/// Where a world's probes stand: a box cut into cells of one size, and a probe
/// at the middle of each.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
	/// The box's corner with the least of every axis. The first probe stands
	/// half a step in from it along each.
	pub from: Vec3,

	/// How wide a cell is along every axis, in world units: how far apart two
	/// probes beside each other stand.
	pub step: f32,

	/// How many cells, and so how many probes, along x, y and z.
	pub counts: [u32; 3],
}

impl Grid {
	/// No grid, which is what a world nobody baked has.
	pub const NONE: Self = Self {
		from: Vec3::ZERO,
		step: 0.0,
		counts: [0; 3],
	};

	/// Whether it holds a probe at all, with a step that means something.
	#[must_use]
	pub fn is_some(&self) -> bool {
		self.step.is_finite() && self.step > 0.0 && self.counts.iter().all(|count| *count > 0)
	}

	/// How many probes.
	#[must_use]
	pub fn len(&self) -> usize {
		self.counts
			.iter()
			.map(|count| usize::try_from(*count).unwrap_or(usize::MAX))
			.fold(1_usize, usize::saturating_mul)
	}

	/// Whether it holds no probe.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.len() == 0 || !self.is_some() }

	/// The box's corner with the most of every axis.
	#[must_use]
	pub fn to(&self) -> Vec3 {
		let [x, y, z] = self.counts.map(whole);

		self.from + Vec3::new(x, y, z) * self.step
	}

	/// Where one probe stands.
	///
	/// @param at - which, counted from nought along x, y and z
	#[must_use]
	pub fn at(&self, [x, y, z]: [u32; 3]) -> Vec3 {
		self.from + (Vec3::new(whole(x), whole(y), whole(z)) + 0.5) * self.step
	}

	/// Which probe a number counts to, in the order a bake works them out: x
	/// fastest, then z, then y - the order the picture's rows run in.
	///
	/// @param index - from nought, below [`len`](Self::len)
	#[must_use]
	pub fn place(&self, index: usize) -> [u32; 3] {
		let [across, _, deep] = self
			.counts
			.map(|count| usize::try_from(count.max(1)).unwrap_or(1));
		let x = index % across;
		let z = (index / across) % deep;
		let y = index / across / deep;

		[x, y, z].map(|part| u32::try_from(part).unwrap_or(u32::MAX))
	}

	/// The number [`place`](Self::place) counts to a probe with.
	///
	/// @param at - the probe, counted from nought along x, y and z
	#[must_use]
	pub fn index(&self, [x, y, z]: [u32; 3]) -> usize {
		let [across, _, deep] = self
			.counts
			.map(|count| usize::try_from(count).unwrap_or(0));
		let [x, y, z] = [x, y, z].map(|part| usize::try_from(part).unwrap_or(usize::MAX));

		y.saturating_mul(deep)
			.saturating_add(z)
			.saturating_mul(across)
			.saturating_add(x)
	}

	/// How many texels across and down the picture that keeps these probes is:
	/// six times as wide as the grid along x, and as tall as its layers along y
	/// times its depth along z.
	///
	/// @return the size, or nothing for a grid a picture cannot hold
	#[must_use]
	pub fn picture(&self) -> Option<[u32; 2]> {
		let [x, y, z] = self.counts;

		Some([x.checked_mul(6)?, y.checked_mul(z)?])
	}

	/// Which texel of that picture keeps one face of one probe.
	///
	/// @param at - the probe, counted from nought along x, y and z
	/// @param face - which face, in [`AXES`]' order
	/// @return how many texels from the left and from the top
	#[must_use]
	pub fn texel(&self, [x, y, z]: [u32; 3], face: u32) -> [u32; 2] {
		let [across, _, deep] = self.counts;

		[
			face.saturating_mul(across).saturating_add(x),
			y.saturating_mul(deep).saturating_add(z),
		]
	}
}

impl Default for Grid {
	fn default() -> Self { Self::NONE }
}

/// A world's probes: the picture their light is kept in, and where they stand.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Probes {
	/// The picture, or [`TextureId::NONE`] for a world with no probes.
	pub picture: TextureId,

	/// Where they stand.
	pub grid: Grid,
}

impl Probes {
	/// No probes, which is what a world nobody baked has.
	pub const NONE: Self = Self {
		picture: TextureId::NONE,
		grid: Grid::NONE,
	};

	/// Whether there are probes to read: a picture named and a grid with
	/// probes in it.
	#[must_use]
	pub fn is_some(&self) -> bool { self.picture.is_some() && self.grid.is_some() }
}

/// A count as a float: exact below two to the twenty-fourth, far above any grid
/// a picture can hold.
#[expect(
	clippy::as_conversions,
	clippy::cast_precision_loss,
	reason = "a count of cells, far below where a float stops holding whole numbers"
)]
const fn whole(count: u32) -> f32 { count as f32 }

#[cfg(test)]
mod tests {
	use super::*;

	fn grid() -> Grid {
		Grid {
			from: Vec3::new(-2.0, -1.0, 4.0),
			step: 0.5,
			counts: [3, 2, 4],
		}
	}

	#[test]
	fn a_probe_stands_at_the_middle_of_its_cell() {
		let grid = grid();

		assert_eq!(grid.at([0, 0, 0]), Vec3::new(-1.75, -0.75, 4.25), "half a step in");
		assert_eq!(grid.at([2, 1, 3]), Vec3::new(-0.75, -0.25, 5.75), "the last");
		assert_eq!(grid.to(), Vec3::new(-0.5, 0.0, 6.0), "the far corner");
		assert_eq!(grid.len(), 24, "three by two by four");
	}

	#[test]
	fn every_probe_is_counted_once_in_the_order_the_picture_runs() {
		let grid = grid();
		let places: Vec<[u32; 3]> = (0..grid.len())
			.map(|index| grid.place(index))
			.collect();

		assert_eq!(places[0], [0, 0, 0], "the first");
		assert_eq!(places[1], [1, 0, 0], "x fastest");
		assert_eq!(places[3], [0, 0, 1], "then z");
		assert_eq!(places[12], [0, 1, 0], "then y");

		let mut sorted = places.clone();

		sorted.sort_unstable();
		sorted.dedup();

		assert_eq!(sorted.len(), grid.len(), "none twice");

		for (index, place) in places.iter().enumerate() {
			assert_eq!(grid.index(*place), index, "{place:?} counts back to its number");
		}
	}

	#[test]
	fn every_face_of_every_probe_has_a_texel_of_its_own_inside_the_picture() {
		let grid = grid();
		let [width, height] = grid.picture().expect("a picture holds it");
		let mut seen = vec![false; usize::try_from(width * height).expect("small")];

		assert_eq!([width, height], [18, 8], "six faces of three across, two layers of four");

		for index in 0..grid.len() {
			let place = grid.place(index);

			for face in 0..6 {
				let [column, row] = grid.texel(place, face);

				assert!(column < width && row < height, "{place:?} face {face} is inside");

				let at = usize::try_from(row * width + column).expect("small");

				assert!(!seen[at], "{place:?} face {face} shares its texel");
				seen[at] = true;
			}
		}

		assert!(seen.iter().all(|taken| *taken), "and every texel keeps a face");
	}

	#[test]
	fn a_grid_with_no_probe_or_no_step_is_none() {
		assert!(!Grid::NONE.is_some(), "the default");
		assert!(Grid::NONE.is_empty());
		assert!(!Grid { counts: [3, 0, 1], ..grid() }.is_some(), "a count of nought");
		assert!(!Grid { step: 0.0, ..grid() }.is_some(), "a step of nought");
		assert!(!Grid { step: f32::NAN, ..grid() }.is_some(), "a step that is no number");
		assert!(grid().is_some());
		assert!(!Probes::NONE.is_some(), "no picture");
		assert!(
			!Probes { picture: TextureId::NONE, grid: grid() }.is_some(),
			"a grid with no picture"
		);
	}

	#[test]
	fn the_faces_are_kept_in_a_cubes_order() {
		assert_eq!(AXES, [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z]);
	}
}
