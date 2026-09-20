//! The brush that paints where a strewing's copies may stand.
//!
//! **What it paints is the rule's input.** A stroke writes cells of a
//! [`Mask`], which is what the laying reads as one more reason a copy it drew
//! does not stand; no copy is written down by anything here, and taking a
//! stroke back puts every copy it removed exactly where it was. @ref
//! `colby_core::abi::strew` for what a cell means.
//!
//! **It goes by the ground's own triangles and not by what a click picks.**
//! [`aim::under`](crate::aim) answers *which* entity a ray reaches and it
//! answers it against each entity's mesh bounds, which is exact enough to
//! choose a thing out of a scene and useless for a point on a hillside: a box
//! around a landscape is met at the sky. So a stroke traces one mesh - the
//! ground the selected strewing hangs off - triangle by triangle, and takes the
//! nearest hit. One mesh and not the world, which is what keeps it affordable:
//! a landscape of thirty thousand triangles is a fraction of a millisecond, and
//! it is asked once a dab rather than once a frame.
//!
//! **A dab a distance rather than a dab a frame.** A stroke lands a dab where
//! it starts and then every quarter of the brush's own width it travels, so
//! what a stroke does depends on where the pointer went and not on how many
//! frames it took to get there. Painting at ten frames a second and at three
//! hundred leave the same mask - which is also what stops a field being laid
//! again every frame of a drag.
//!
//! The arithmetic is here and the pointer is in [`viewport`](crate::viewport),
//! deliberately: what is here has tests and no egui in it.

use colby_core::{
	abi::{EntityId, Mask, MeshData, Transform, World, strew},
	glam::{Mat4, Vec2, Vec3},
};

/// How wide the brush is to begin with, in the ground's own units.
pub(crate) const RADIUS: f32 = 4.0;

/// How hard it paints to begin with: a share of the whole of a cell per dab,
/// at the middle of the brush.
pub(crate) const STRENGTH: f32 = 0.35;

/// The widest and narrowest the brush may be.
pub(crate) const RANGE: (f32, f32) = (0.25, 256.0);

/// How far a stroke travels between dabs, as a share of the brush's width.
///
/// A quarter, so that a stroke drawn slowly and one drawn quickly leave about
/// the same paint down: fewer and they read as dots, more and a slow stroke
/// saturates whatever it passes over.
pub(crate) const SPACING: f32 = 0.25;

/// A stroke of the brush, from the moment the pointer went down.
///
/// Its whole state is where the last dab landed, because everything else a
/// stroke does is in the cells already. @ref [`Brush::dab`] for why that is
/// what a stroke has to remember.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Brush {
	/// Where the last dab landed, in the ground's space, or nothing before the
	/// first one.
	last: Option<Vec2>,
}

impl Brush {
	/// Whether a dab belongs here, and remembers it if it does.
	///
	/// @param at - where the stroke is now, east and south in the ground's
	/// space
	/// @param radius - how wide the brush is
	/// @return whether to paint
	pub(crate) fn dab(&mut self, at: Vec2, radius: f32) -> bool {
		let far = self
			.last
			.is_none_or(|last| (at - last).length() >= radius * SPACING);

		if far {
			self.last = Some(at);
		}

		far
	}

	/// Forgets where it was, so that the next dab lands wherever the stroke
	/// starts.
	pub(crate) fn lift(&mut self) { self.last = None; }
}

/// Which entity a strewing stands its copies on, and where the ray meets it.
///
/// @param world - the world
/// @param id - the strewing entity
/// @param from - where the ray starts, in world space
/// @param along - which way it goes; need not be of unit length
/// @return where on the ground it landed, **in the ground's own space**, or
/// nothing when the entity strews nothing, hangs off nothing, or the ray
/// misses. All three numbers, because a brush paints by two of them and is
/// drawn at the third.
pub(crate) fn ground_under(world: &World, id: EntityId, from: Vec3, along: Vec3) -> Option<Vec3> {
	let ground = world.entities.parent(id);
	let placed = world.entities.placed(ground)?;
	let mesh = world
		.entities
		.renderable(ground)
		.and_then(|renderable| world.meshes.get(renderable.mesh))?;

	if !strew::strews(&world.entities, id) {
		return None;
	}

	nearest(mesh.value(), &placed, from, along)
}

/// Where a strewing's ground stands in the world.
///
/// One call and not one a place: what is drawn from it is a ring of fifty
/// points, and looking the ground up for each of them would be fifty walks up
/// the same chain of parents.
///
/// @param world - the world
/// @param id - the strewing entity, whose parent is the ground
/// @return the matrix that carries a place in the ground's own space into the
/// world, or nothing for a strewing hanging off nothing
pub(crate) fn ground_of(world: &World, id: EntityId) -> Option<Mat4> {
	let matrix = world
		.entities
		.placed(world.entities.parent(id))?
		.matrix();

	matrix.is_finite().then_some(matrix)
}

/// A ring of places around a point, in the ground's own space.
///
/// In the ground's space and not the world's, because that is the space the
/// cells are in: a ground scaled or turned paints an oval on the world and the
/// ring has to be the same oval.
///
/// @param at - the middle, in the ground's space
/// @param radius - how wide, in the ground's units
/// @param steps - how many points
pub(crate) fn ring(at: Vec3, radius: f32, steps: usize) -> Vec<Vec3> {
	(0..=steps)
		.map(|step| {
			#[expect(
				clippy::as_conversions,
				clippy::cast_precision_loss,
				reason = "a small count of steps around a ring"
			)]
			let share = step as f32 / steps.max(1) as f32;
			let (sin, cos) = (std::f32::consts::TAU * share).sin_cos();

			at + Vec3::new(cos, 0.0, sin) * radius
		})
		.collect()
}

/// The nearest place a ray meets a mesh, in the mesh's own space.
///
/// @param mesh - the geometry, in its own space
/// @param placed - where it stands in the world
/// @param from - where the ray starts, in world space
/// @param along - which way it goes
fn nearest(mesh: &MeshData, placed: &Transform, from: Vec3, along: Vec3) -> Option<Vec3> {
	let matrix = placed.matrix();
	let inverse = matrix.inverse();

	if !inverse.is_finite() {
		return None;
	}

	let start = inverse.transform_point3(from);
	let direction = inverse.transform_vector3(along);
	let mut nearest = f32::INFINITY;
	let mut found = None;

	for triangle in mesh.indices.chunks_exact(3) {
		let Some(corners) = corners_of(mesh, triangle) else {
			continue;
		};
		let Some(distance) = meets(start, direction, corners) else {
			continue;
		};

		if distance < nearest {
			nearest = distance;
			found = Some(start + direction * distance);
		}
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

/// How far along a ray it meets a triangle, if it does.
///
/// The ray is written as the corner plus two of the triangle's edges plus a
/// distance along itself, and the three unknowns come out of one determinant
/// each - which is the same arrangement the slab test in
/// [`aim`](crate::aim) has, three planes instead of six and no box.
///
/// Both faces of a triangle count: a brush put to the underside of a ground
/// should paint it, and refusing a back face would make painting depend on
/// which way somebody modeled the ground.
///
/// @return the fraction along the ray at which it meets, or nothing when it
/// misses or the triangle has no area
fn meets(from: Vec3, along: Vec3, corners: [Vec3; 3]) -> Option<f32> {
	let [first, second, third] = corners;
	let (edge, other) = (second - first, third - first);
	let across = along.cross(other);
	let volume = edge.dot(across);

	// a ray in the triangle's own plane, or a triangle of no area at all
	if volume.abs() < 1.0e-12 {
		return None;
	}

	let inverse = 1.0 / volume;
	let out = from - first;
	let along_edge = out.dot(across) * inverse;

	if !(0.0..=1.0).contains(&along_edge) {
		return None;
	}

	let sideways = out.cross(edge);
	let along_other = along.dot(sideways) * inverse;

	if along_other < 0.0 || along_edge + along_other > 1.0 {
		return None;
	}

	let distance = other.dot(sideways) * inverse;

	(distance > 0.0).then_some(distance)
}

/// Paints one dab into a strewing's mask, making the mask if there is none and
/// widening it if the ground has grown past it.
///
/// @param world - the world
/// @param id - the strewing entity
/// @param at - where the dab lands, east and south in the ground's space
/// @param radius - how wide the brush is
/// @param strength - how hard, from minus one for putting a whole field back to
/// one for taking one away
/// @return whether any cell changed
pub(crate) fn paint(
	world: &mut World,
	id: EntityId,
	at: Vec2,
	radius: f32,
	strength: f32,
) -> bool {
	let Some(bounds) = ground_bounds(world, id) else {
		return false;
	};
	// made here and not when an entity starts strewing: a mask is a grid over
	// a ground, and until somebody paints there is no reason to have decided
	// what ground that was. Widened here for the same reason - the grid is a
	// fact about the mask, and a stroke is the moment it has to be true.
	let held = world.entities.mask(id).cloned();
	let fitted = match held {
		| Some(mask) => mask.fitted(bounds).or(Some(mask)),
		| None => Mask::over(bounds),
	};

	let Some(mut mask) = fitted else {
		return false;
	};
	let moved = mask.paint(at, radius, strength);

	// the mask is put back whether a cell moved or not: it may have been made
	// or widened by this very call, and a mask nobody has painted yet is still
	// a grid the next stroke writes into.
	world.entities.set_mask(id, Some(mask));

	moved
}

/// The box of the ground a strewing stands its copies on, in the ground's own
/// space.
fn ground_bounds(world: &World, id: EntityId) -> Option<(Vec3, Vec3)> {
	let ground = world.entities.parent(id);
	let mesh = world
		.entities
		.renderable(ground)
		.and_then(|renderable| world.meshes.get(renderable.mesh))?;
	let (low, high) = mesh.value().bounds();

	(low.cmplt(high).any()).then_some((low, high))
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{MeshVertex, Renderable, STREWING, Strewing, mesh},
		glam::Quat,
	};

	use super::*;

	/// A flat square of ground, `side` units across, centered on the origin.
	fn floor(side: f32) -> MeshData {
		let half = side * 0.5;
		let corner = |x: f32, z: f32| MeshVertex::new(Vec3::new(x, 0.0, z), Vec3::Y, Vec2::ZERO);

		MeshData {
			vertices: vec![
				corner(-half, -half),
				corner(half, -half),
				corner(half, half),
				corner(-half, half),
			],
			indices: vec![0, 2, 1, 0, 3, 2],
			..MeshData::default()
		}
	}

	/// A world with a ground and a strewing hung off it.
	fn meadow() -> (World, EntityId) {
		let mut world = World::new();
		world
			.entities
			.declare(&STREWING)
			.expect("a world with nothing declared takes the record");
		world.meshes.insert("meshes/cube", mesh::cube());
		let ground = world.meshes.insert("meshes/floor", floor(32.0));

		let stood = world.entities.spawn();
		world
			.entities
			.set_renderable(stood, Renderable::new(ground, Vec3::ONE));

		let strewing = world.entities.spawn();
		world
			.entities
			.set_renderable(strewing, Renderable::new(colby_core::abi::MeshId::CUBE, Vec3::ONE));
		world.entities.set_parent(strewing, stood);

		if let Some(rule) = world.entities.record_mut(&STREWING, strewing) {
			*rule = Strewing { strews: 1, ..Strewing::NONE };
		}

		(world, strewing)
	}

	#[test]
	fn a_ray_down_the_middle_lands_where_it_points() {
		let (world, strewing) = meadow();

		let landed = ground_under(&world, strewing, Vec3::new(3.0, 10.0, -5.0), Vec3::NEG_Y);
		assert_eq!(landed, Some(Vec3::new(3.0, 0.0, -5.0)), "straight down onto the floor");

		let missed = ground_under(&world, strewing, Vec3::new(3.0, 10.0, -5.0), Vec3::Y);
		assert_eq!(missed, None, "and a ray pointing away from it meets nothing");

		let outside = ground_under(&world, strewing, Vec3::new(99.0, 10.0, 0.0), Vec3::NEG_Y);
		assert_eq!(outside, None, "and one beside the floor meets nothing either");
	}

	#[test]
	fn a_ray_lands_in_the_grounds_own_space_whatever_the_ground_is_doing() {
		let (mut world, strewing) = meadow();
		let ground = world.entities.parent(strewing);

		// turned a quarter turn about up, moved and doubled: a point four units
		// east of the middle in the world is two units *south* of it on the
		// ground, which is a place no arithmetic in world space would give
		if let Some(transform) = world.entities.transform_mut(ground) {
			*transform = Transform {
				position: Vec3::new(10.0, 2.0, 0.0),
				rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
				scale: Vec3::splat(2.0),
			};
		}

		let landed = ground_under(&world, strewing, Vec3::new(14.0, 9.0, 0.0), Vec3::NEG_Y)
			.expect("it lands on the ground");

		assert!(
			landed.abs_diff_eq(Vec3::new(0.0, 0.0, 2.0), 1.0e-3),
			"four units east of the middle, in a ground turned and scaled: {landed:?}"
		);
	}

	#[test]
	fn a_ray_takes_the_nearest_of_two_triangles_it_goes_through() {
		let (mut world, strewing) = meadow();
		let ground = world.entities.parent(strewing);
		let mut stacked = floor(8.0);
		let over = floor(8.0);
		let first = u32::try_from(stacked.vertices.len()).expect("four corners");

		stacked
			.vertices
			.extend(over.vertices.iter().map(|vertex| {
				MeshVertex::new(
					Vec3::from_array(vertex.position) + Vec3::Y * 3.0,
					Vec3::Y,
					Vec2::ZERO,
				)
			}));
		stacked
			.indices
			.extend(over.indices.iter().map(|index| index + first));

		let held = world.meshes.insert("meshes/stacked", stacked);
		world
			.entities
			.set_renderable(ground, Renderable::new(held, Vec3::ONE));

		let landed = ground_under(&world, strewing, Vec3::new(1.0, 10.0, 1.0), Vec3::NEG_Y);
		assert_eq!(landed, Some(Vec3::new(1.0, 3.0, 1.0)), "the higher of the two is nearer");

		// from below, the nearer one is the lower
		let under = ground_under(&world, strewing, Vec3::new(1.0, -10.0, 1.0), Vec3::Y);
		assert_eq!(under, Some(Vec3::new(1.0, 0.0, 1.0)), "and from below, the lower one");
	}

	#[test]
	fn an_entity_that_strews_nothing_has_no_ground_to_paint() {
		let (mut world, strewing) = meadow();

		if let Some(rule) = world.entities.record_mut(&STREWING, strewing) {
			rule.strews = 0;
		}

		assert_eq!(
			ground_under(&world, strewing, Vec3::new(0.0, 10.0, 0.0), Vec3::NEG_Y),
			None,
			"nothing is strewn over it, so there is nothing to paint"
		);
	}

	#[test]
	fn a_stroke_dabs_by_the_distance_it_has_gone_and_not_by_the_frame() {
		let mut brush = Brush::default();

		assert!(brush.dab(Vec2::ZERO, 4.0), "the first dab lands where the stroke starts");
		assert!(!brush.dab(Vec2::new(0.1, 0.0), 4.0), "and a pointer that barely moved is one");
		assert!(!brush.dab(Vec2::new(0.9, 0.0), 4.0), "however many frames it takes");
		assert!(brush.dab(Vec2::new(1.0, 0.0), 4.0), "a quarter of the width apart is another");

		brush.lift();
		assert!(brush.dab(Vec2::new(1.0, 0.0), 4.0), "and a new stroke dabs where it starts");
	}

	#[test]
	fn a_stroke_makes_a_mask_over_the_ground_and_thins_the_field_under_it() {
		let (mut world, strewing) = meadow();

		assert!(world.entities.mask(strewing).is_none(), "nobody has painted it");
		assert!(paint(&mut world, strewing, Vec2::ZERO, 4.0, 1.0), "and a stroke lands");

		let mask = world
			.entities
			.mask(strewing)
			.expect("the stroke made one");

		assert!(mask.at(0.0, 0.0) < 1.0, "the middle of the stroke is painted away");
		assert_eq!(
			mask.at(15.0, 15.0).to_bits(),
			1.0_f32.to_bits(),
			"the far corner is untouched"
		);
		assert!(mask.painted() > 0.0, "and some share of the field has gone");
	}

	#[test]
	fn a_stroke_over_ground_the_mask_never_covered_widens_it() {
		let (mut world, strewing) = meadow();
		let ground = world.entities.parent(strewing);

		assert!(paint(&mut world, strewing, Vec2::ZERO, 4.0, 1.0), "painted over the small one");
		let held = world
			.entities
			.mask(strewing)
			.expect("the stroke made one")
			.clone();

		let wider = world.meshes.insert("meshes/wider", floor(128.0));
		world
			.entities
			.set_renderable(ground, Renderable::new(wider, Vec3::ONE));

		assert!(
			paint(&mut world, strewing, Vec2::splat(50.0), 4.0, 1.0),
			"and over the wide one"
		);
		let grown = world
			.entities
			.mask(strewing)
			.expect("it is still there");

		assert!(
			grown.counts()[0] > held.counts()[0] || grown.step() > held.step(),
			"the grid reaches the new ground: {:?} against {:?}",
			grown.counts(),
			held.counts()
		);
		assert!(grown.at(50.0, 50.0) < 1.0, "the second stroke is on it");
		assert!(grown.at(0.0, 0.0) < 1.0, "and the first one is still where it was");
	}

	#[test]
	fn painting_the_other_way_puts_a_field_back() {
		let (mut world, strewing) = meadow();

		for _ in 0..8 {
			paint(&mut world, strewing, Vec2::ZERO, 4.0, 1.0);
		}

		let shut = world
			.entities
			.mask(strewing)
			.expect("the strokes made one")
			.at(0.0, 0.0);
		assert_eq!(shut.to_bits(), 0.0_f32.to_bits(), "eight strokes take the field away");

		for _ in 0..8 {
			paint(&mut world, strewing, Vec2::ZERO, 4.0, -1.0);
		}

		let back = world
			.entities
			.mask(strewing)
			.expect("it is still there")
			.at(0.0, 0.0);
		assert_eq!(back.to_bits(), 1.0_f32.to_bits(), "and eight the other way put it back");
	}
}
