//! Where each still thing's light goes on one picture.
//!
//! ```text
//!   Atlas::of(&scene, texels)       a place on one picture for every piece
//!   atlas.place(piece)              where one piece's light is, if anywhere
//! ```
//!
//! **A thing's place is its mesh's sheet, made larger where the thing is
//! larger.** A mesh's second set was laid out on a sheet at
//! [`unwrap::TEXELS`] a unit of the mesh as it stands in its own file; a thing
//! that stands scaled up in the world takes that sheet larger by the one factor
//! that gives the texels its charts cover the area its surface has in the
//! world, at the density asked for. That is the rule the unwrap lays each chart
//! out by, applied a thing at a time and measured on the thing's own triangles
//! and its own second set - so a thing stretched further one way than another,
//! and a second set somebody made by hand, go through the same arithmetic as
//! everything else. A stretched thing's density is the same on average and
//! not on every face: its sheet was laid for the mesh, not for the stretch.
//!
//! **Never smaller than the sheet.** The gutters between the charts were laid
//! for the sheet, and below it they close: a density asked for below the
//! sheet's own leaves a thing at its sheet.
//!
//! **One picture, packed by the unwrap's own packer** with the unwrap's gutter
//! between any two places, so two charts of two things are as far apart as two
//! charts of one. What does not fit in [`MAX_SIDE`] a side is refused rather
//! than split across pictures: one binding is what the renderer reads, and a
//! world that needs more than that at the density asked for can ask for less.
//!
//! What gets no place: a thing drawn unlit, which reads no light; a mesh with
//! no second set, which has nowhere on it to read one from; and a sheet wider
//! than a picture can be. The first still gives off its light in a bake and the
//! other two still send on what reaches them.

use colby_core::{
	Result, err,
	glam::Vec2,
	unwrap::{self, GUTTER},
};

use crate::scene::{Piece, Scene};

/// The widest and tallest a lightmap is, in texels.
///
/// What every device the renderer asks for is guaranteed to hold, and what the
/// reader of the file it is kept in will read.
pub const MAX_SIDE: u32 = 8192;

/// How far in from the corner of its cell a place starts: half the gutter, so
/// that a place is one texel from the edge of the picture and two from the
/// next, the unwrap's own arrangement.
const MARGIN: u32 = GUTTER / 2;

/// How far past a whole number of texels a side may be worked out and still be
/// that number: a sixteenth of a texel.
///
/// A sheet laid out at exactly the density asked for measures at it only to
/// within a rounding of its second set, and a rounding above one would make
/// every such sheet a texel larger than it is.
const SLACK: f64 = 1.0 / 16.0;

/// Where one thing's light is on the picture, in whole texels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
	/// The first column.
	pub left: u32,

	/// The first row.
	pub top: u32,

	/// How many columns.
	pub width: u32,

	/// How many rows.
	pub height: u32,
}

impl Rect {
	/// Where a point of the second set falls on the picture, in texels from its
	/// top left corner.
	///
	/// @param uv2 - the point on the thing's own sheet, from nought to one
	#[must_use]
	pub fn place(self, uv2: Vec2) -> [f64; 2] {
		let across = f64::from(uv2.x) * f64::from(self.width);
		let down = f64::from(uv2.y) * f64::from(self.height);

		[f64::from(self.left) + across, f64::from(self.top) + down]
	}

	/// Whether a texel of the picture is inside it.
	#[must_use]
	pub const fn holds(self, column: u32, row: u32) -> bool {
		column >= self.left
			&& row >= self.top
			&& column - self.left < self.width
			&& row - self.top < self.height
	}
}

/// Why a still thing has no place on the picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placeless {
	/// It is drawn as its own color and reads no light.
	Unlit,

	/// Its mesh has no second set of coordinates.
	NoSheet,

	/// Its sheet is wider or taller than a picture can be.
	Oversized,
}

/// A place on one picture for every piece that has one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Atlas {
	width: u32,
	height: u32,
	places: Vec<Option<Rect>>,
	placeless: Vec<(u32, Placeless)>,
}

impl Atlas {
	/// A place for every still thing that can have one.
	///
	/// @param scene - what stands still
	/// @param texels - how many texels a unit of surface is asked for; a thing
	/// is never given fewer than its sheet was laid out with
	/// @return the places, or why they do not fit on one picture
	///
	/// # Errors
	///
	/// If the places do not fit in [`MAX_SIDE`] a side.
	pub fn of(scene: &Scene, texels: f32) -> Result<Self> {
		let mut places = vec![None; scene.pieces().len()];
		let mut placeless = Vec::new();
		let mut sized: Vec<(usize, [u32; 2])> = Vec::new();

		for (index, piece) in scene.pieces().iter().enumerate() {
			let why = if piece.look.unlit {
				Some(Placeless::Unlit)
			} else if piece.sheet[0] == 0 || piece.sheet[1] == 0 {
				Some(Placeless::NoSheet)
			} else {
				None
			};

			if let Some(why) = why {
				placeless.push((whole(index), why));

				continue;
			}

			let scale = scale_of(scene, piece, texels);
			let size = [side(piece.sheet[0], scale), side(piece.sheet[1], scale)];

			if size.iter().any(|&side| side > MAX_SIDE - GUTTER) {
				placeless.push((whole(index), Placeless::Oversized));

				continue;
			}

			sized.push((index, size));
		}

		let cells: Vec<[u32; 2]> = sized
			.iter()
			.map(|(_, size)| [size[0] + GUTTER, size[1] + GUTTER])
			.collect();
		let (corners, sheet) = unwrap::packed(&cells);

		if sheet[0] > MAX_SIDE || sheet[1] > MAX_SIDE {
			let asked: u64 = cells
				.iter()
				.map(|cell| u64::from(cell[0]) * u64::from(cell[1]))
				.sum();

			return Err(err!(Asset(
				"the light of {} still things at {texels} texels a unit takes {}x{} texels \
				 ({asked} in their places), past the {MAX_SIDE} a side a lightmap can be; ask \
				 for fewer texels a unit, or leave the largest out of the bake",
				cells.len(),
				sheet[0],
				sheet[1]
			)));
		}

		for ((index, size), corner) in sized.iter().zip(corners) {
			if let Some(place) = places.get_mut(*index) {
				*place = Some(Rect {
					left: corner[0] + MARGIN,
					top: corner[1] + MARGIN,
					width: size[0],
					height: size[1],
				});
			}
		}

		Ok(Self {
			width: sheet[0],
			height: sheet[1],
			places,
			placeless,
		})
	}

	/// How many texels across the picture is.
	#[must_use]
	pub const fn width(&self) -> u32 { self.width }

	/// How many texels down.
	#[must_use]
	pub const fn height(&self) -> u32 { self.height }

	/// Whether nothing has a place, which is a picture of nothing.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.places.iter().all(Option::is_none) }

	/// Where one piece's light is.
	///
	/// @param piece - as the scene numbers them
	/// @return its place, or nothing for a piece with none
	#[must_use]
	pub fn place(&self, piece: u32) -> Option<Rect> {
		usize::try_from(piece)
			.ok()
			.and_then(|piece| self.places.get(piece))
			.copied()
			.flatten()
	}

	/// Every piece's place, in the scene's order.
	#[must_use]
	pub fn places(&self) -> &[Option<Rect>] { &self.places }

	/// Every piece with no place, and why.
	#[must_use]
	pub fn placeless(&self) -> &[(u32, Placeless)] { &self.placeless }
}

/// The one factor a piece's sheet is made larger by: the one that puts its
/// charts' texels over its surface in the world at the density asked for, and
/// never below one.
///
/// @param scene - whose triangles
/// @param piece - which of them
/// @param texels - the density asked for, a unit of surface
fn scale_of(scene: &Scene, piece: &Piece, texels: f32) -> f64 {
	let (mut world, mut laid) = (0.0_f64, 0.0_f64);
	let sheet = [f64::from(piece.sheet[0]), f64::from(piece.sheet[1])];
	let first = usize::try_from(piece.first).unwrap_or(usize::MAX);
	let count = usize::try_from(piece.count).unwrap_or(0);

	for triangle in scene.triangles().iter().skip(first).take(count) {
		let corners = triangle.map(|index| {
			usize::try_from(index)
				.ok()
				.and_then(|index| scene.corners().get(index))
		});
		let [Some(one), Some(two), Some(three)] = corners else {
			continue;
		};

		world += length(cross(
			minus(one.position.to_array(), two.position.to_array()),
			minus(one.position.to_array(), three.position.to_array()),
		));

		let [along, across] = [two.uv2 - one.uv2, three.uv2 - one.uv2]
			.map(|edge| [f64::from(edge.x) * sheet[0], f64::from(edge.y) * sheet[1]]);
		let forward = along[0] * across[1];
		let back = along[1] * across[0];

		laid += (forward - back).abs();
	}

	let scale = f64::from(texels) * (world / laid).sqrt();

	// a sheet that covers nothing, or a surface of nothing, is left as it was
	if scale.is_nan() {
		1.0
	} else {
		scale.clamp(1.0, f64::from(MAX_SIDE))
	}
}

/// One side of a place: the sheet's side made larger by a factor, rounded up
/// to whole texels, and never smaller than the sheet's.
fn side(sheet: u32, scale: f64) -> u32 {
	let grown = f64::from(sheet) * scale;
	let wanted = grown - SLACK;

	whole_up(wanted).max(sheet)
}

/// A size in texels rounded up, held between nought and the widest picture.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "the value is held between nought and the widest picture and rounded up on the \
	          line above the cast, and try_from is not available for a float"
)]
fn whole_up(size: f64) -> u32 {
	let held = if size.is_nan() {
		0.0
	} else {
		size.clamp(0.0, f64::from(MAX_SIDE))
	};

	held.ceil() as u32
}

/// A piece's place in the scene's list, as the atlas keeps it.
fn whole(index: usize) -> u32 { u32::try_from(index).unwrap_or(u32::MAX) }

/// One point less another, widened.
fn minus(from: [f32; 3], less: [f32; 3]) -> [f64; 3] {
	[0, 1, 2].map(|axis| f64::from(from[axis]) - f64::from(less[axis]))
}

/// The cross product of two vectors.
fn cross(one: [f64; 3], other: [f64; 3]) -> [f64; 3] {
	let term = |first: usize, second: usize| {
		let forward = one[first] * other[second];
		let back = one[second] * other[first];

		forward - back
	};

	[term(1, 2), term(2, 0), term(0, 1)]
}

/// The length of a vector.
fn length(vector: [f64; 3]) -> f64 {
	let [x, y, z] = vector.map(|axis| axis * axis);

	(x + y + z).sqrt()
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			MeshId, Renderable, Transform, World,
			material::{Material, MaterialId},
			mesh::cube,
		},
		glam::Vec3,
	};

	use super::*;

	/// A world of cubes, each standing at its own scale.
	fn cubes(scales: &[Vec3]) -> World {
		let mut world = World::new();

		for (step, scale) in (0_u16..).zip(scales) {
			let at = world.entities.spawn_at(Transform {
				position: Vec3::new(f32::from(step) * 40.0, 0.0, 0.0),
				scale: *scale,
				..Transform::IDENTITY
			});

			world
				.entities
				.set_renderable(at, Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, Vec3::ONE));
		}

		world
	}

	#[test]
	fn a_thing_at_the_scale_its_sheet_was_laid_out_at_takes_its_sheet_exactly() {
		let atlas = Atlas::of(&Scene::of(&cubes(&[Vec3::ONE])), unwrap::TEXELS).expect("fits");
		let place = atlas.place(0).expect("the cube has a place");

		assert_eq!([place.width, place.height], cube().sheet, "the sheet, texel for texel");
	}

	#[test]
	fn a_thing_scaled_up_takes_its_sheet_larger_by_its_scale() {
		let atlas =
			Atlas::of(&Scene::of(&cubes(&[Vec3::splat(3.0)])), unwrap::TEXELS).expect("it fits");
		let place = atlas.place(0).expect("a place");
		let sheet = cube().sheet;

		assert_eq!(
			[place.width, place.height],
			[sheet[0] * 3, sheet[1] * 3],
			"three times as wide makes three times the texels a side"
		);
	}

	#[test]
	fn a_thing_scaled_down_and_a_density_below_the_sheet_s_keep_the_sheet() {
		let small =
			Atlas::of(&Scene::of(&cubes(&[Vec3::splat(0.25)])), unwrap::TEXELS).expect("fits");
		let sparse = Atlas::of(&Scene::of(&cubes(&[Vec3::ONE])), 1.0).expect("fits");
		let sheet = cube().sheet;

		for (atlas, what) in [(&small, "scaled down"), (&sparse, "asked for fewer texels")] {
			let place = atlas.place(0).expect("a place");

			assert_eq!([place.width, place.height], sheet, "{what}: the sheet is the floor");
		}
	}

	#[test]
	fn a_thing_stretched_one_way_takes_the_factor_that_keeps_its_area() {
		// a slab four by one by one: area 2 (4 + 1 + 4) = 18 against the cube's
		// 6, so the sheet is made larger by the square root of three
		let atlas = Atlas::of(&Scene::of(&cubes(&[Vec3::new(4.0, 1.0, 1.0)])), unwrap::TEXELS)
			.expect("fits");
		let place = atlas.place(0).expect("a place");
		let sheet = cube().sheet;
		let factor = 3.0_f64.sqrt();

		assert_eq!(
			[place.width, place.height],
			[sheet[0], sheet[1]].map(|side| {
				let grown = f64::from(side) * factor;

				whole_up(grown - SLACK)
			}),
			"the sheet times the square root of three"
		);
	}

	#[test]
	fn places_are_apart_by_the_gutter_and_inside_the_picture() {
		let scales = [Vec3::ONE, Vec3::splat(2.0), Vec3::new(3.0, 1.0, 0.5), Vec3::splat(0.5)];
		let atlas = Atlas::of(&Scene::of(&cubes(&scales)), unwrap::TEXELS).expect("fits");
		let places: Vec<Rect> = atlas.places().iter().flatten().copied().collect();

		assert_eq!(places.len(), 4, "every cube has a place");

		for (index, one) in places.iter().enumerate() {
			assert!(
				one.left >= MARGIN
					&& one.top >= MARGIN
					&& one.left + one.width + MARGIN <= atlas.width()
					&& one.top + one.height + MARGIN <= atlas.height(),
				"{one:?} is a texel inside the picture of {}x{}",
				atlas.width(),
				atlas.height()
			);

			for other in places.iter().skip(index + 1) {
				let apart_across = one.left + one.width + GUTTER <= other.left
					|| other.left + other.width + GUTTER <= one.left;
				let apart_down = one.top + one.height + GUTTER <= other.top
					|| other.top + other.height + GUTTER <= one.top;

				assert!(apart_across || apart_down, "{one:?} and {other:?} are two apart");
			}
		}
	}

	#[test]
	fn what_reads_no_light_or_has_nowhere_to_keep_it_has_no_place() {
		let mut world = cubes(&[Vec3::ONE]);
		let unlit = world
			.materials
			.insert("unlit", Material { unlit: true, ..Material::DEFAULT });
		let sheetless = world.meshes.insert("bare", {
			let mut mesh = cube();
			mesh.paint.clear();
			mesh.sheet = [0, 0];
			mesh
		});

		for (mesh, material) in [(MeshId::CUBE, unlit), (sheetless, MaterialId::DEFAULT)] {
			let at = world
				.entities
				.spawn_at(Transform::at(Vec3::new(0.0, 9.0, 0.0)));

			world
				.entities
				.set_renderable(at, Renderable::of(mesh, material, Vec3::ONE));
		}

		let atlas = Atlas::of(&Scene::of(&world), unwrap::TEXELS).expect("fits");

		assert!(atlas.place(0).is_some(), "the plain cube has one");
		assert_eq!(
			atlas.placeless(),
			&[(1, Placeless::Unlit), (2, Placeless::NoSheet)],
			"the unlit one and the one with no second set have none, and say why"
		);
	}

	#[test]
	fn more_than_one_picture_holds_is_refused_naming_the_ceiling() {
		// each fits a picture alone, and four of them do not fit one together
		let refused = Atlas::of(&Scene::of(&cubes(&[Vec3::splat(250.0); 4])), unwrap::TEXELS)
			.expect_err("four cubes two hundred and fifty wide at five texels a unit");
		let said = format!("{refused}");

		assert!(said.contains("8192"), "it names the ceiling: {said}");
	}
}
