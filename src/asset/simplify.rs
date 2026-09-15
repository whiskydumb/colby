//! The coarser levels a mesh is compiled with.
//!
//! **Every compiled mesh gets them, as many as it thins out to.** A level is
//! fewer of the mesh's own triangles over the same vertices - @ref
//! [`Level`] - and what decides where one may be drawn is how far it stands
//! from the mesh, so what this works out is a chain of index lists and a
//! distance for each. A source's sidecar may ask for fewer, or for none.
//!
//! What a level is made of, rule by rule, each for a reason:
//!
//! - **Half the triangles of the level before it**, simplified from that level
//!   rather than from the mesh. Each step then starts from something already
//!   close to what it is asked for, and the chain costs about twice one
//!   simplification of the whole mesh however long it runs.
//! - **The normals count beside the positions**, weighed one to one. The
//!   distance that comes back then counts a turned normal the way it counts a
//!   moved point, so a level waits until the light across it would look the
//!   same as well as its outline; what that costs is a later switch.
//! - **The border does not move.** A model is one mesh per material, so a
//!   surface made of two materials is two meshes meeting along an open edge,
//!   and each simplified on its own would open a crack between them.
//! - **A part smaller than the distance may go**, and the distance counts it
//!   going. A level that has to keep every screw is a level that stops thinning
//!   out long before the screws are smaller than a pixel.
//! - **A mesh bones move is kept even**: long thin triangles fold badly when a
//!   joint bends them, and the simplifier trades a little of the distance for
//!   triangles that are more alike.
//! - **The distance never shrinks down the chain, and grows by at least half at
//!   every level**, so two levels are never chosen within a hair of each other.
//! - **The chain stops** when a step keeps three quarters of what it was given,
//!   when the distance passes the size of the mesh itself, or at
//!   [`MAX_LEVELS`].
//! - **The distance is written in the mesh's own units**, not as a share of its
//!   size, because a share of its size is not something a picture can turn into
//!   pixels without the size beside it.
//!
//! **A mesh whose every vertex is its own is barely thinned.** The simplifier
//! keeps an edge where two vertices share a place and differ in what they
//! carry, and a mesh shaded flat by giving each triangle its own corners is
//! nothing but such edges. Welding near-equal vertices first would free it, and
//! nothing here does yet.

use colby_core::{
	abi::{MAX_LEVELS, MeshData, MeshVertex, mesh::Level},
	bytemuck,
};
use meshopt::{SimplifyOptions, VertexDataAdapter};

/// The fewest indices a level is asked to keep: four triangles.
///
/// Below that a level is asked for nothing it could honestly be, and a chain
/// that got there has already thinned out as far as is worth drawing.
const SMALLEST: usize = 12;

/// How much a normal weighs against a position, for each of its three axes.
const NORMAL_WEIGHTS: [f32; 3] = [1.0; 3];

/// How much a step has to take away before the chain goes on: it stops when
/// what came back is this share of four of what went in, or more.
const KEPT_QUARTERS: usize = 3;

/// How much the distance grows at least, from one level to the next.
const GROWTH: f32 = 1.5;

/// The farthest a level may stand from the mesh, as a share of the mesh's size.
///
/// A level that far from what it stands for is a different shape, and past it
/// the simplifier's estimate is no longer about how the level looks.
const FARTHEST: f32 = 1.0;

/// The coarser levels of a mesh, the finest first.
///
/// @param data - the mesh, as it is about to be written
/// @param most - how many levels at most; nought for none
/// @return the levels, which may be fewer than asked for and may be none
#[must_use]
pub fn levels(data: &MeshData, most: usize) -> Vec<Level> {
	// @note: no mesh a test can afford thins out past seven on its own - a
	// closed ball ends its chain within seven levels however dense it
	// starts, four hundred thousand included - so a mutation that dropped this
	// passed every test. It stays because a denser mesh would otherwise come
	// back with more levels than the file will hold.
	let most = most.min(MAX_LEVELS);

	if most == 0
		|| data.indices.len() <= SMALLEST
		|| !data.indices_are_in_range()
		|| !data.indices.len().is_multiple_of(3)
	{
		return Vec::new();
	}

	let Ok(positions) =
		VertexDataAdapter::new(bytemuck::cast_slice(&data.vertices), size_of::<MeshVertex>(), 0)
	else {
		return Vec::new();
	};

	let asking = Asking::new(data, &positions);
	let size = meshopt::simplify_scale(&positions);
	let mut chain: Vec<Level> = Vec::new();
	let mut far = 0.0_f32;

	while chain.len() < most {
		let before = chain
			.last()
			.map_or(data.indices.as_slice(), |level| level.indices.as_slice());
		let Some((indices, step)) = asking.thinner(before) else {
			break;
		};
		let grown = (far * GROWTH).max(step);

		// a distance that is not a number stops the chain rather than joining it.
		//
		// @note: no mesh tried reaches the second half. Each step is already
		// held to that distance by the simplifier, so only the half-again
		// growth can pass it, and on every mesh tried the step after a level
		// that far out has pruned the mesh away and stopped the chain first; a
		// mutation that dropped it passed every test
		if grown.is_nan() || grown > FARTHEST {
			break;
		}

		far = grown;
		chain.push(Level { indices, error: far * size });
	}

	chain
}

/// Everything one mesh's simplifications share.
struct Asking<'a> {
	/// The vertices, as the simplifier reads their positions.
	positions: &'a VertexDataAdapter<'a>,

	/// Every vertex's normal, three floats each.
	normals: Vec<f32>,

	/// One word a vertex that none of them is locked.
	unlocked: Vec<bool>,

	/// How the simplifier is asked to go about it.
	options: SimplifyOptions,
}

impl<'a> Asking<'a> {
	/// Everything a mesh's chain is asked with.
	///
	/// @param data - the mesh
	/// @param positions - its vertices, as the simplifier reads them
	fn new(data: &MeshData, positions: &'a VertexDataAdapter<'a>) -> Self {
		Self {
			positions,
			normals: data
				.vertices
				.iter()
				.flat_map(|vertex| vertex.normal)
				.collect(),
			unlocked: vec![false; data.vertices.len()],
			options: if data.is_skinned() {
				SimplifyOptions::Sparse
					| SimplifyOptions::LockBorder
					| SimplifyOptions::Prune
					| SimplifyOptions::Regularize
			} else {
				SimplifyOptions::Sparse | SimplifyOptions::LockBorder | SimplifyOptions::Prune
			},
		}
	}

	/// One level thinner than the one given, or nothing when it thins no
	/// further.
	///
	/// @param before - the level to thin out, as indices over the mesh
	/// @return its indices and how far it stands from the mesh as a share of
	/// the mesh's size, or nothing when the step kept too much
	fn thinner(&self, before: &[u32]) -> Option<(Vec<u32>, f32)> {
		let target = (before.len() / 6 * 3).max(SMALLEST);

		if target >= before.len() {
			return None;
		}

		let (mut indices, mut step) = self.simplified(before, target, self.options);

		// a mesh made of nothing but small parts can be pruned away entirely;
		// asked again with its parts kept, it thins out like any other
		if indices.is_empty() && self.options.contains(SimplifyOptions::Prune) {
			(indices, step) =
				self.simplified(before, target, self.options - SimplifyOptions::Prune);
		}

		if indices.is_empty() || indices.len() * 4 >= before.len() * KEPT_QUARTERS {
			return None;
		}

		Some((indices, step))
	}

	/// One call to the simplifier.
	fn simplified(
		&self,
		before: &[u32],
		target: usize,
		options: SimplifyOptions,
	) -> (Vec<u32>, f32) {
		let mut step = 0.0_f32;
		let indices = meshopt::simplify_with_attributes_and_locks(
			before,
			self.positions,
			&self.normals,
			&NORMAL_WEIGHTS,
			size_of::<[f32; 3]>(),
			&self.unlocked,
			target,
			FARTHEST,
			options,
			Some(&mut step),
		);

		(indices, step)
	}
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			SkinVertex,
			mesh::{self, cube, quad, sphere},
		},
		glam::{Vec2, Vec3},
	};

	use super::*;

	/// A ball of bumps: a grid over a sphere, pushed in and out by a few waves,
	/// so there is something to thin out and something to lose doing it.
	///
	/// @param rings - bands of latitude
	/// @param segments - bands of longitude
	fn bumpy(rings: u32, segments: u32) -> MeshData { ball(rings, segments, 0.08) }

	/// A ball over a latitude and longitude grid.
	///
	/// @param rings - bands of latitude
	/// @param segments - bands of longitude
	/// @param bump - how far a few waves push its surface in and out
	fn ball(rings: u32, segments: u32, bump: f32) -> MeshData {
		let mut data = MeshData::default();

		for ring in 0..=rings {
			let down = f32::from(u16::try_from(ring).expect("small"))
				/ f32::from(u16::try_from(rings).expect("small"));
			let angle = down * core::f32::consts::PI;

			for segment in 0..=segments {
				let around = f32::from(u16::try_from(segment).expect("small"))
					/ f32::from(u16::try_from(segments).expect("small"));
				let turn = around * core::f32::consts::TAU;
				let way =
					Vec3::new(angle.sin() * turn.cos(), angle.cos(), angle.sin() * turn.sin());
				let pushed = (bump * (turn * 5.0).sin()).mul_add((angle * 4.0).cos(), 1.0);

				data.vertices
					.push(MeshVertex::new(way * pushed, way, Vec2::new(around, down)));
			}
		}

		let stride = segments + 1;

		for ring in 0..rings {
			for segment in 0..segments {
				let top = ring * stride + segment;
				let bottom = top + stride;

				data.indices
					.extend([top, top + 1, bottom, top + 1, bottom + 1, bottom]);
			}
		}

		mesh::tangents(&mut data);

		data
	}

	/// A flat grid of quads with an open edge all the way round.
	fn sheet(side: u32) -> MeshData {
		let mut data = MeshData::default();

		for row in 0..=side {
			for column in 0..=side {
				let (x, z) = (
					f32::from(u16::try_from(column).expect("small")),
					f32::from(u16::try_from(row).expect("small")),
				);
				// a gentle swell, so a collapse inside the sheet costs something
				let y = 0.05 * (x * 0.7).sin() * (z * 0.9).cos();

				data.vertices
					.push(MeshVertex::new(Vec3::new(x, y, z), Vec3::Y, Vec2::new(x, z)));
			}
		}

		let stride = side + 1;

		for row in 0..side {
			for column in 0..side {
				let at = row * stride + column;

				data.indices.extend([
					at,
					at + stride,
					at + 1,
					at + 1,
					at + stride,
					at + stride + 1,
				]);
			}
		}

		mesh::tangents(&mut data);

		data
	}

	/// The swelling sheet, with normals that lean every way across it.
	fn leaning() -> MeshData {
		let mut data = sheet(24);

		for vertex in &mut data.vertices {
			let [x, _, z] = vertex.position;

			vertex.normal = Vec3::new((x * 2.3).sin(), 1.0, (z * 1.7).cos())
				.normalize()
				.to_array();
		}

		data
	}

	/// Twenty-seven small bumpy balls standing apart in a block, as one mesh:
	/// parts small enough against the whole for pruning to take some away.
	fn scattered() -> MeshData {
		let mut data = MeshData::default();

		for place in 0..27_u16 {
			let small = bumpy(4, 6);
			let first = u32::try_from(data.vertices.len()).expect("a small mesh");
			let offset =
				Vec3::new(f32::from(place % 3), f32::from((place / 3) % 3), f32::from(place / 9))
					* 3.0;

			data.vertices
				.extend(small.vertices.iter().map(|vertex| MeshVertex {
					position: (Vec3::from_array(vertex.position) * 0.1 + offset).to_array(),
					..*vertex
				}));
			data.indices
				.extend(small.indices.iter().map(|index| index + first));
		}

		data
	}

	/// The size the simplifier measures a mesh by.
	fn size_of_mesh(data: &MeshData) -> f32 {
		meshopt::simplify_scale(
			&VertexDataAdapter::new(
				bytemuck::cast_slice(&data.vertices),
				size_of::<MeshVertex>(),
				0,
			)
			.expect("48-byte vertices"),
		)
	}

	#[test]
	fn a_dense_mesh_thins_out_level_by_level() {
		let data = bumpy(48, 64);
		let chain = levels(&data, MAX_LEVELS);

		assert!(
			chain.len() >= 4,
			"a ball of six thousand triangles thins out a long way: {}",
			chain.len()
		);
		assert!(chain.len() <= MAX_LEVELS, "and no further than a mesh may carry");

		let mut with = data;
		with.levels = chain;

		assert!(
			with.levels_are_in_range(),
			"every level is whole triangles over the ball's own vertices"
		);
		assert!(with.levels_thin_out(), "each thinner than the one before and no nearer");
	}

	#[test]
	fn every_level_keeps_at_most_three_quarters_and_about_half_of_the_one_before() {
		let meshes = [
			("ball", bumpy(48, 64)),
			("sheet", sheet(24)),
			("leaning", leaning()),
			("scattered", scattered()),
		];

		for (name, data) in meshes {
			let mut before = data.indices.len();

			for (at, level) in levels(&data, MAX_LEVELS).iter().enumerate() {
				assert!(
					level.indices.len() * 4 < before * 3,
					"the {name}'s level {at} keeps {} of {before}, more than three quarters",
					level.indices.len()
				);

				before = level.indices.len();
			}
		}

		// and where nothing is small enough to be pruned away, a level is what
		// it was asked for: half, and not a third
		let data = bumpy(48, 64);
		let mut before = data.indices.len();

		for (at, level) in levels(&data, MAX_LEVELS).iter().enumerate() {
			assert!(
				level.indices.len() * 3 >= before,
				"level {at} keeps {} of {before}",
				level.indices.len()
			);

			before = level.indices.len();
		}
	}

	#[test]
	fn the_distance_grows_by_at_least_half_and_is_written_in_the_meshes_own_units() {
		// the sheet because its second step, left to itself, comes back only a
		// third further than its first (measured: 0.00067 after 0.00050), so the
		// growth is what makes it half again
		for data in [sheet(24), bumpy(48, 64)] {
			for pair in levels(&data, MAX_LEVELS).windows(2) {
				// by the number rather than by the constant that says it, and to
				// within the rounding of which product was taken first
				assert!(
					pair[1].error >= pair[0].error * 1.5 * (1.0 - 1.0e-6),
					"{} after {} grew by less than half",
					pair[1].error,
					pair[0].error
				);
			}
		}

		let data = bumpy(48, 64);
		let chain = levels(&data, MAX_LEVELS);

		// the same ball twice as big stands twice as far from its levels, which
		// a distance written as a share of the size would not
		let mut big = data;

		for vertex in &mut big.vertices {
			vertex.position = (Vec3::from_array(vertex.position) * 2.0).to_array();
		}

		let doubled = levels(&big, MAX_LEVELS);

		assert_eq!(doubled.len(), chain.len(), "the same chain");

		for (small, large) in chain.iter().zip(&doubled) {
			assert_eq!(small.indices, large.indices, "of the same triangles");
			assert!(
				small.error.mul_add(-2.0, large.error).abs() <= small.error * 1.0e-3,
				"at twice the distance: {} against {}",
				large.error,
				small.error
			);
		}
	}

	#[test]
	fn the_levels_are_the_same_every_time() {
		let data = bumpy(32, 40);

		assert_eq!(
			levels(&data, MAX_LEVELS),
			levels(&data, MAX_LEVELS),
			"the same list, index for index"
		);
	}

	#[test]
	fn a_count_asked_for_is_the_most_there_are() {
		let data = bumpy(48, 64);

		assert!(levels(&data, 0).is_empty(), "none asked for, none made");
		assert_eq!(levels(&data, 2).len(), 2, "two asked for, two made");
		assert_eq!(
			levels(&data, 2),
			levels(&data, MAX_LEVELS)[..2].to_vec(),
			"and they are the first two of the long chain"
		);
		assert!(levels(&data, 100).len() <= MAX_LEVELS, "and past the limit is the limit");
	}

	#[test]
	fn no_level_stands_further_from_the_mesh_than_the_mesh_is_big() {
		let meshes = [
			("small ball", bumpy(12, 16)),
			("ball", bumpy(48, 64)),
			("leaning", leaning()),
			("scattered", scattered()),
			("sphere", sphere()),
		];

		for (name, data) in meshes {
			let size = size_of_mesh(&data);

			for (at, level) in levels(&data, MAX_LEVELS).iter().enumerate() {
				assert!(
					level.error <= size * (1.0 + 1.0e-6),
					"the {name}'s level {at} stands {} from a mesh {size} big",
					level.error
				);
			}
		}
	}

	#[test]
	fn a_step_that_pruning_empties_is_asked_again_with_its_parts_kept() {
		// measured: the sheet's fourth step, pruning allowed, takes the whole
		// sheet away, because by then the sheet is a part small enough to go
		let data = sheet(24);
		let chain = levels(&data, 3);
		let positions = VertexDataAdapter::new(
			bytemuck::cast_slice(&data.vertices),
			size_of::<MeshVertex>(),
			0,
		)
		.expect("48-byte vertices");
		let asking = Asking::new(&data, &positions);
		let before = &chain[2].indices;
		let target = (before.len() / 6 * 3).max(SMALLEST);
		let (pruned, _) = asking.simplified(before, target, asking.options);

		assert!(pruned.is_empty(), "pruning takes it all at this step: {}", pruned.len());

		let (kept, _) = asking
			.thinner(before)
			.expect("asked again with its parts kept, it thins out");

		assert!(
			!kept.is_empty() && kept.len() < before.len(),
			"a real level: {} of {}",
			kept.len(),
			before.len()
		);
	}

	#[test]
	fn an_open_edge_stays_where_it_was_at_every_level() {
		let side = 24;
		let data = sheet(side);
		let chain = levels(&data, MAX_LEVELS);

		assert!(!chain.is_empty(), "a swelling sheet of a thousand triangles thins out");

		let stride = side + 1;
		let on_edge = |index: u32| {
			let (row, column) = (index / stride, index % stride);

			row == 0 || row == side || column == 0 || column == side
		};

		for (at, level) in chain.iter().enumerate() {
			for index in (0..stride * stride).filter(|index| on_edge(*index)) {
				assert!(
					level.indices.contains(&index),
					"level {at} dropped the edge vertex {index}, which would open a crack \
					 against the next mesh along"
				);
			}
		}
	}

	#[test]
	fn a_mesh_with_nothing_to_take_away_has_no_levels() {
		assert!(levels(&cube(), MAX_LEVELS).is_empty(), "twelve flat triangles");
		assert!(levels(&quad(), MAX_LEVELS).is_empty(), "two");
		assert!(levels(&MeshData::default(), MAX_LEVELS).is_empty(), "none");
	}

	#[test]
	fn a_mesh_that_cannot_be_read_has_no_levels_rather_than_a_crash() {
		let mut past = bumpy(16, 16);
		past.indices[0] = 1_000_000;

		assert!(levels(&past, MAX_LEVELS).is_empty(), "an index past the vertices");

		let mut partial = bumpy(16, 16);
		partial.indices.pop();

		assert!(levels(&partial, MAX_LEVELS).is_empty(), "an index list of broken triangles");
	}

	#[test]
	fn a_skinned_mesh_keeps_its_weights_because_it_keeps_its_vertices() {
		let mut data = bumpy(32, 40);
		data.skin = data
			.vertices
			.iter()
			.map(|vertex| SkinVertex::rigid(u16::from(vertex.position[1] > 0.0)))
			.collect();

		let chain = levels(&data, MAX_LEVELS);

		assert!(!chain.is_empty(), "a skinned ball thins out too");

		let mut with = data.clone();
		with.levels = chain;

		assert!(with.levels_are_in_range(), "over the same vertices, so over the same weights");
		assert_eq!(with.skin, data.skin, "which nothing touched");
	}

	/// How unlike each other a list's triangles are in size: the spread of
	/// their areas over the mean area.
	fn unevenness(data: &MeshData, indices: &[u32]) -> f32 {
		let at = |index: u32| {
			Vec3::from_array(data.vertices[usize::try_from(index).expect("small")].position)
		};
		let areas: Vec<f32> = indices
			.chunks_exact(3)
			.map(|corners| {
				(at(corners[1]) - at(corners[0]))
					.cross(at(corners[2]) - at(corners[0]))
					.length() * 0.5
			})
			.collect();
		let count = f32::from(u16::try_from(areas.len()).expect("a small list"));
		let mean = areas.iter().sum::<f32>() / count;
		let spread = areas
			.iter()
			.map(|value| (value - mean) * (value - mean))
			.sum::<f32>()
			/ count;

		spread.sqrt() / mean
	}

	#[test]
	fn a_mesh_bones_move_is_thinned_into_more_even_triangles() {
		let still = bumpy(32, 40);
		let mut moved = still.clone();
		moved.skin = vec![SkinVertex::rigid(0); moved.vertices.len()];

		let (plain, even) = (levels(&still, 1), levels(&moved, 1));
		let (loose, kept) =
			(unevenness(&still, &plain[0].indices), unevenness(&moved, &even[0].indices));

		// measured: 0.87 without bones, 0.50 with them
		assert!(
			kept < loose * 0.8,
			"the skinned ball's first level is the more even: {kept} against {loose}"
		);
	}

	#[test]
	fn the_built_in_sphere_is_thinned_like_any_other_mesh_it_is_handed() {
		// the registry's own ball is never compiled, so it never gets levels;
		// this is the simplifier answering for it if something ever asks
		let chain = levels(&sphere(), MAX_LEVELS);
		let mut with = sphere();
		with.levels = chain;

		assert!(
			with.levels_are_in_range() && with.levels_thin_out(),
			"whatever it makes is sound"
		);
	}

	#[test]
	fn a_turned_normal_counts_in_how_far_a_level_stands() {
		// two sheets of the same positions: one whose normals all point up, and
		// one whose normals lean every way. Thinned by the same steps, the second
		// has to stand further from its levels, because its light would change
		let flat = sheet(24);
		let mut leaning = flat.clone();

		for vertex in &mut leaning.vertices {
			let [x, _, z] = vertex.position;
			let lean = Vec3::new((x * 2.3).sin(), 1.0, (z * 1.7).cos()).normalize();

			vertex.normal = lean.to_array();
		}

		let (Some(still), Some(turned)) = (levels(&flat, 1).pop(), levels(&leaning, 1).pop())
		else {
			panic!("both sheets thin out once");
		};

		// measured at a hundred times further: 1.26 against 0.012
		assert!(
			turned.error > still.error * 10.0,
			"the leaning sheet's level stands further: {} against {}",
			turned.error,
			still.error
		);
	}
}
