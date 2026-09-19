//! The light arriving at points of the air, for everything the lightmap keeps
//! no place for.
//!
//! ```text
//!   grid_of(&scene, density)                  where the probes stand
//!   probes(&scene, grid, &pattern, .., sent)  what each of them gathers
//! ```
//!
//! **A regular grid, a probe at the middle of each cell.** The box every still
//! triangle fits in, grown by a cell each way - a cell of air round everything,
//! so that a thing standing on the highest roof or against the outermost wall
//! is still inside it - cut into cubes of one size, as many a unit as the bake
//! is asked for. Nothing is refined where the light changes fastest, which is
//! what the field's larger bakers spend most of their probe code on; the price
//! is probes standing in empty air, which cost rays and nothing else. The
//! middles of cells rather than their corners, so that a floor level with the
//! box's bottom is not a plane probes stand on.
//!
//! **Each probe is six gathers**, one about each way along each axis, with the
//! pattern and the average a texel's gather uses: what a surface facing that
//! way at that point would gather, in the unit the lightmap keeps. What a ray
//! that lands on a surface brings back is whatever the caller says - a bake
//! reads the picture its last gather read, so a probe and the texel beside it
//! carry the light off the same number of surfaces.
//!
//! **A probe is inside something when more than three in ten of its rays meet
//! a back.** A probe out in the air meets none; one standing exactly on a
//! surface meets the back of it with half of its rays; one inside a wall with
//! all of them. Three in ten tells the second and the third from the first,
//! where the half a texel's own rule draws the line at would leave a probe on a
//! surface to a coin's toss. What such a probe gathered is thrown away, and it
//! takes the average of the probes beside it that hold, a ring at a time out
//! from those - a probe inside something is only ever read as the far corner of
//! a cell whose near corners hold.
//!
//! **Kept as a picture**, in the order [`Grid::texel`] says, beside the
//! lightmap. @ref [`probes`](colby_core::abi::probes).

use colby_core::{
	abi::probes::{AXES, FACES, Grid},
	glam::Vec3,
};

use crate::{
	atlas::MAX_SIDE,
	gather::{Gathered, Pattern, each, seed},
	scene::Scene,
	tree::Hit,
};

/// The most probes a bake keeps. Past it the grid is laid coarser until it
/// fits, as it is when its picture would be wider or taller than
/// [`MAX_SIDE`].
///
/// A quarter of a million: twelve megabytes on the device at eight bytes a
/// face, and a few seconds of rays at the pattern's default.
pub const MAX_PROBES: usize = 1 << 18;

/// How much coarser the grid is laid each time it does not fit.
const COARSER: f32 = 1.25;

/// The share of a probe's rays that may meet a back before the probe is taken
/// to be inside something: three in ten, as a numerator and a denominator.
const INSIDE: (u32, u32) = (3, 10);

/// The pass a probe's seeds are drawn for, which no gather over the texels is.
const PASS: u32 = u32::MAX;

/// What a bake worked out for its probes.
#[derive(Clone, Debug, PartialEq)]
pub struct Probed {
	/// Where they stand.
	pub grid: Grid,

	/// Each probe's six faces in [`AXES`]' order, the probes in the order
	/// [`Grid::place`] counts them.
	pub faces: Vec<[Vec3; FACES]>,

	/// How many were inside something, and took their light from the probes
	/// round them.
	pub buried: usize,

	/// How many rays they sent.
	pub rays: u64,
}

impl Probed {
	/// The picture the probes are kept in: every face of every probe at the
	/// texel [`Grid::texel`] gives it, row by row from the top.
	///
	/// @return how many texels across and down, and the texels; nothing for a
	/// grid no picture can hold
	#[must_use]
	pub fn picture(&self) -> Option<([u32; 2], Vec<Vec3>)> {
		let [width, height] = self.grid.picture()?;
		let across = usize::try_from(width).ok()?;
		let mut texels = vec![Vec3::ZERO; across.checked_mul(usize::try_from(height).ok()?)?];

		for (index, faces) in self.faces.iter().enumerate() {
			let place = self.grid.place(index);

			for (face, light) in (0_u32..).zip(faces) {
				let [column, row] = self.grid.texel(place, face);
				let at = usize::try_from(row)
					.ok()?
					.checked_mul(across)?
					.checked_add(usize::try_from(column).ok()?)?;

				*texels.get_mut(at)? = *light;
			}
		}

		Some(([width, height], texels))
	}
}

/// Where a scene's probes stand.
///
/// @param scene - what stands still
/// @param density - how many probes a unit along each axis; nought for none
/// @return the grid, laid coarser than asked if it has to be to fit; nothing
/// for no density, and for a scene with nothing still in it
#[must_use]
pub fn grid_of(scene: &Scene, density: f32) -> Option<Grid> {
	if density.is_nan() || density <= 0.0 {
		return None;
	}

	let (low, high) = bounds(scene)?;
	let mut step = density.recip();

	// bounded: each turn widens the cells by a quarter, and sixty-four of them
	// make a cell some sixteen million times as wide
	for _ in 0..64 {
		let grid = laid(low, high, step)?;

		if fits(&grid) {
			return Some(grid);
		}

		step *= COARSER;
	}

	None
}

/// The box every still triangle's corners fit in.
///
/// Worked out by comparison, axis by axis, rather than by a vector's `min` and
/// `max`: the answer has to be the same bits on every machine, and those answer
/// a number that is not a number by the order they were handed their operands.
fn bounds(scene: &Scene) -> Option<(Vec3, Vec3)> {
	let mut corners = scene
		.corners()
		.iter()
		.map(|corner| corner.position.to_array())
		.filter(|position| position.iter().all(|part| part.is_finite()));
	let first = corners.next()?;

	let (low, high) = corners.fold((first, first), |(mut low, mut high), position| {
		for axis in 0..3 {
			if position[axis] < low[axis] {
				low[axis] = position[axis];
			}

			if position[axis] > high[axis] {
				high[axis] = position[axis];
			}
		}

		(low, high)
	});

	Some((Vec3::from_array(low), Vec3::from_array(high)))
}

/// A grid of cells of one width over a box grown by a cell each way.
///
/// @return the grid, or nothing for a width or a box no count of cells spans
fn laid(low: Vec3, high: Vec3, step: f32) -> Option<Grid> {
	if !step.is_finite() || step <= 0.0 {
		return None;
	}

	let from = low - step;
	let reach = ((high + step) - from) / step;
	let counts = reach
		.to_array()
		.map(|cells| count_of(cells.ceil()).map(|count| count.max(1)));

	Some(Grid {
		from,
		step,
		counts: [counts[0]?, counts[1]?, counts[2]?],
	})
}

/// Whether a grid is small enough to keep: few enough probes, and a picture no
/// wider or taller than a lightmap may be.
fn fits(grid: &Grid) -> bool {
	grid.len() <= MAX_PROBES
		&& grid
			.picture()
			.is_some_and(|[width, height]| width <= MAX_SIDE && height <= MAX_SIDE)
}

/// A whole number of cells as a count, or nothing for one past what a count
/// holds.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "a whole number checked to lie between nought and what a count holds first"
)]
fn count_of(cells: f32) -> Option<u32> {
	(cells.is_finite() && (0.0..=16_777_216.0).contains(&cells)).then_some(cells as u32)
}

/// Works out every probe of a grid.
///
/// @param scene - what stands still
/// @param grid - where the probes stand
/// @param pattern - the directions each face sends its rays in
/// @param threads - how many threads at most; the answer does not depend on it
/// @param sent - what a surface a ray landed on sends back along it
/// @return the probes, or nothing when every one of them is inside something
pub fn probes<Sent: Fn(&Hit) -> Vec3 + Sync>(
	scene: &Scene,
	grid: Grid,
	pattern: &Pattern,
	threads: usize,
	sent: Sent,
) -> Option<Probed> {
	let gathered: Vec<[Gathered; FACES]> = each(grid.len(), threads, |index| {
		let at = grid.at(grid.place(index));

		core::array::from_fn(|face| {
			let point = index.saturating_mul(FACES).saturating_add(face);

			scene.gather(at, AXES[face], seed(point, PASS), pattern, |hit, _| sent(hit))
		})
	});
	let held: Vec<bool> = gathered
		.iter()
		.map(|faces| !inside(faces))
		.collect();

	if !held.contains(&true) {
		return None;
	}

	let mut faces: Vec<[Vec3; FACES]> = gathered
		.iter()
		.map(|faces| faces.map(|face| face.light))
		.collect();
	let buried = held.iter().filter(|holds| !**holds).count();

	fill(&grid, &mut faces, held);

	Some(Probed {
		grid,
		faces,
		buried,
		rays: u64::try_from(grid.len().saturating_mul(FACES))
			.unwrap_or(u64::MAX)
			.saturating_mul(u64::try_from(pattern.len()).unwrap_or(0)),
	})
}

/// Whether a probe is inside something: more than three in ten of the rays of
/// all six of its faces met a back.
fn inside(faces: &[Gathered; FACES]) -> bool {
	let (behind, rays) = faces
		.iter()
		.fold((0_u64, 0_u64), |(behind, rays), face| {
			(behind + u64::from(face.behind), rays + u64::from(face.rays))
		});
	let (share, of) = INSIDE;

	behind * u64::from(of) > rays * u64::from(share)
}

/// Gives every probe that does not hold the average of those beside it that
/// do, a ring at a time outwards from the ones that held from the start.
///
/// Beside is along an axis: the six a cell shares a face with. A ring is worked
/// out whole before any of it is written, so the order a ring is walked in
/// changes nothing.
fn fill(grid: &Grid, faces: &mut [[Vec3; FACES]], mut held: Vec<bool>) {
	let mut ring: Vec<usize> = (0..faces.len())
		.filter(|&at| !held[at] && beside(grid, at).any(|next| held[next]))
		.collect();

	while !ring.is_empty() {
		let filled: Vec<(usize, [Vec3; FACES])> = ring
			.iter()
			.filter_map(|&at| averaged(grid, faces, &held, at).map(|value| (at, value)))
			.collect();

		for &(at, value) in &filled {
			faces[at] = value;
			held[at] = true;
		}

		let mut next: Vec<usize> = filled
			.iter()
			.flat_map(|&(at, _)| beside(grid, at))
			.filter(|&at| !held[at])
			.collect();

		next.sort_unstable();
		next.dedup();
		ring = next;
	}
}

/// The average of the probes beside one that hold, face by face, added up in
/// double precision and narrowed once; nothing when none of them holds.
fn averaged(
	grid: &Grid,
	faces: &[[Vec3; FACES]],
	held: &[bool],
	at: usize,
) -> Option<[Vec3; FACES]> {
	let mut total = [[0.0_f64; 3]; FACES];
	let mut count = 0_u32;

	for next in beside(grid, at).filter(|&next| held[next]) {
		for (sum, light) in total.iter_mut().zip(&faces[next]) {
			for (channel, part) in sum.iter_mut().zip(light.to_array()) {
				*channel += f64::from(part);
			}
		}

		count += 1;
	}

	(count > 0).then(|| {
		total.map(|sum| Vec3::from_array(sum.map(|channel| narrowed(channel / f64::from(count)))))
	})
}

/// The probes a cell shares a face with, inside the grid.
fn beside(grid: &Grid, at: usize) -> impl Iterator<Item = usize> + '_ {
	let place = grid.place(at).map(i64::from);
	let counts = grid.counts.map(i64::from);

	[[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
		.into_iter()
		.filter_map(move |way: [i64; 3]| {
			let next = [place[0] + way[0], place[1] + way[1], place[2] + way[2]];
			let inside = (0..3).all(|axis| (0..counts[axis]).contains(&next[axis]));

			inside.then(|| grid.index(next.map(|part| u32::try_from(part).unwrap_or(0))))
		})
}

/// A double narrowed to a float, rounding to the nearest.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "an average worked out in double precision and narrowed once, where it is kept"
)]
const fn narrowed(value: f64) -> f32 { value as f32 }

#[cfg(test)]
mod tests {
	use colby_core::abi::{MeshId, Renderable, Transform, World, material::MaterialId};

	use super::*;
	use crate::gather::threads;

	/// A world of one quad of floor, so wide, and a block standing on it.
	fn yard(floor: f32) -> World {
		let mut world = World::new();

		for (mesh, position, scale) in [
			(MeshId::QUAD, Vec3::ZERO, Vec3::new(floor, 1.0, floor)),
			(MeshId::CUBE, Vec3::new(1.0, 1.5, -2.0), Vec3::new(2.0, 3.0, 1.0)),
		] {
			let thing =
				world
					.entities
					.spawn_at(Transform { position, scale, ..Transform::IDENTITY });

			world
				.entities
				.set_renderable(thing, Renderable::of(mesh, MaterialId::DEFAULT, Vec3::ONE));
		}

		world
	}

	fn gathered(behind: u32, rays: u32) -> Gathered {
		Gathered { light: Vec3::ONE, behind, rays }
	}

	#[test]
	fn the_grid_covers_every_still_triangle_with_a_cell_of_air_round_it() {
		let scene = Scene::of(&yard(8.0));
		let grid = grid_of(&scene, 1.0).expect("a yard has a grid");

		// the floor from -4 to 4, the block from 0 to 3 high: grown by one each
		// way, from -5 to 5 across and deep and from -1 to 4 up
		assert_eq!(
			grid.from.to_array().map(f32::to_bits),
			[-5.0_f32, -1.0, -5.0].map(f32::to_bits),
			"a cell below and beside"
		);
		assert_eq!(grid.step.to_bits(), 1.0_f32.to_bits(), "one a unit");
		assert_eq!(grid.counts, [10, 5, 10], "and a cell above");
		assert_eq!(
			grid.at([0, 0, 0]).to_array().map(f32::to_bits),
			[-4.5_f32, -0.5, -4.5].map(f32::to_bits),
			"at the middle of a cell"
		);

		let halves = grid_of(&scene, 2.0).expect("twice as dense");

		assert_eq!(halves.counts, [18, 8, 18], "twice as many cells, the margin half as wide");
		assert_eq!(
			halves.from.to_array().map(f32::to_bits),
			[-4.5_f32, -0.5, -4.5].map(f32::to_bits)
		);
	}

	#[test]
	fn no_density_and_no_still_triangle_is_no_grid() {
		let scene = Scene::of(&yard(8.0));

		for density in [0.0, -1.0, f32::NAN] {
			assert!(grid_of(&scene, density).is_none(), "{density} probes a unit is none");
		}

		assert!(grid_of(&Scene::of(&World::new()), 1.0).is_none(), "and nothing still is none");
	}

	#[test]
	fn a_grid_too_large_to_keep_is_laid_coarser_until_it_fits() {
		// a floor a kilometer wide at four probes a unit would be sixteen
		// million a layer
		let scene = Scene::of(&yard(1000.0));
		let grid = grid_of(&scene, 4.0).expect("a grid, coarser than asked");
		let [width, height] = grid.picture().expect("a picture holds it");

		assert!(grid.len() <= MAX_PROBES, "{} probes", grid.len());
		assert!(width <= MAX_SIDE && height <= MAX_SIDE, "a picture of {width}x{height}");

		// and each turn a quarter coarser than the last, exactly
		let mut step = 0.25_f32;
		let turns = (0..64).find(|_| {
			let reached = step.to_bits() == grid.step.to_bits();

			step *= COARSER;

			reached
		});

		assert!(turns.is_some_and(|turns| turns > 0), "a quarter of a unit widened in turns");

		let finer = laid(
			Vec3::new(-500.0, 0.0, -500.0),
			Vec3::new(500.0, 3.0, 500.0),
			grid.step / COARSER,
		)
		.expect("the turn before");

		assert!(!fits(&finer), "and the turn before it did not fit: {:?}", finer.counts);
	}

	#[test]
	fn a_grid_whose_picture_would_be_too_tall_is_laid_coarser_however_few_its_probes() {
		// a wall half a unit thick, a hundred high and a hundred long: at one
		// probe a unit three across, 102 layers of 102 rows - 31,212 probes, an
		// eighth of the most, in a picture 18 texels wide and 10,404 tall
		let mut world = World::new();
		let wall = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, 50.0, 0.0),
			scale: Vec3::new(0.5, 100.0, 100.0),
			..Transform::IDENTITY
		});

		world
			.entities
			.set_renderable(wall, Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, Vec3::ONE));

		let asked = laid(Vec3::new(-0.25, 0.0, -50.0), Vec3::new(0.25, 100.0, 50.0), 1.0)
			.expect("the grid asked for");

		assert_eq!(asked.counts, [3, 102, 102], "the grid asked for");
		assert!(asked.len() <= MAX_PROBES, "few enough probes");
		assert_eq!(asked.picture(), Some([18, 10_404]), "in a picture too tall");
		assert!(!fits(&asked), "so it does not fit");

		let grid = grid_of(&Scene::of(&world), 1.0).expect("a grid, coarser than asked");

		assert_eq!(grid.counts, [3, 82, 82], "a quarter coarser, and no more");
		assert_eq!(grid.step.to_bits(), 1.25_f32.to_bits(), "once");
	}

	#[test]
	fn a_probe_more_than_three_in_ten_of_whose_rays_meet_backs_is_inside_and_no_other() {
		let at = |behind: [u32; 6]| inside(&behind.map(|behind| gathered(behind, 100)));

		assert!(!at([0; 6]), "in the air");
		assert!(!at([30, 30, 30, 30, 30, 30]), "three in ten is not inside");
		assert!(at([30, 30, 30, 30, 30, 31]), "one more ray is");
		assert!(at([100, 50, 50, 50, 50, 0]), "a probe standing on a floor is");
		assert!(at([100; 6]), "and one inside a wall");
	}

	#[test]
	fn a_probe_that_does_not_hold_takes_the_average_of_those_beside_it_a_ring_at_a_time() {
		let grid = Grid {
			from: Vec3::ZERO,
			step: 1.0,
			counts: [4, 1, 2],
		};
		let light = |value: f32| [Vec3::splat(value); FACES];
		// a row of four along x, twice along z: the first column holds, then
		// the second row's last probe holds too
		let mut faces = vec![
			light(8.0),
			light(99.0),
			light(99.0),
			light(99.0),
			light(4.0),
			light(99.0),
			light(99.0),
			light(2.0),
		];
		let held = vec![true, false, false, false, true, false, false, true];

		fill(&grid, &mut faces, held);

		let first: Vec<f32> = faces.iter().map(|probe| probe[0].x).collect();

		// ring one: [1] from [0] (8), [3] from [7] (2), [5] from [4] (4) and [6]
		// from [7] (2); ring two: [2] from [1], [3] and [6] (8, 2, 2 -> 4)
		assert_eq!(first, [8.0, 8.0, 4.0, 2.0, 4.0, 4.0, 2.0, 2.0]);

		for probe in &faces {
			assert!(probe.iter().all(|face| *face == probe[0]), "every face alike");
		}
	}

	#[test]
	fn a_grid_where_every_probe_is_inside_something_is_no_probes() {
		let scene = Scene::of(&yard(8.0));
		let pattern = Pattern::new(16);
		// eight probes inside the block, which is closed on every side, so
		// every ray of every face meets the back of one of its walls
		let block = Grid {
			from: Vec3::new(0.5, 0.5, -2.2),
			step: 0.1,
			counts: [2, 2, 2],
		};

		assert!(probes(&scene, block, &pattern, threads(), |_| Vec3::ONE).is_none(), "none hold");

		let grid = grid_of(&scene, 1.0).expect("a grid");

		assert!(
			probes(&scene, grid, &pattern, threads(), |_| Vec3::ONE).is_some(),
			"where a yard's do"
		);
	}

	#[test]
	fn the_picture_keeps_each_face_of_each_probe_where_the_grid_says() {
		let grid = Grid {
			from: Vec3::ZERO,
			step: 1.0,
			counts: [2, 3, 2],
		};
		let faces: Vec<[Vec3; FACES]> = (0..grid.len())
			.map(|index| {
				core::array::from_fn(|face| {
					Vec3::new(
						f32::from(u8::try_from(index).expect("small")),
						f32::from(u8::try_from(face).expect("small")),
						7.0,
					)
				})
			})
			.collect();
		let probed = Probed { grid, faces, buried: 0, rays: 0 };
		let ([width, height], texels) = probed.picture().expect("a picture");

		assert_eq!([width, height], [12, 6], "six faces of two, three layers of two");

		for (index, probe) in probed.faces.iter().enumerate() {
			for (face, light) in (0_u32..).zip(probe) {
				let [column, row] = grid.texel(grid.place(index), face);

				assert_eq!(
					texels[usize::try_from(row * width + column).expect("small")],
					*light,
					"probe {index} face {face}"
				);
			}
		}
	}
}
