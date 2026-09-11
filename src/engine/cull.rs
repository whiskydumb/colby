//! What a pass can see, as six planes, and the questions a frame asks of them.
//!
//! Until this, [`Scene`](crate::Scene) drew every entity that had a mesh, in
//! view or not, and each of the four shadow cascades drew all of them again. A
//! frame now asks of everything it would draw whether any of it can land
//! inside the volume a pass sees - the view for the picture, a cascade's box
//! for that cascade's depth pass - and leaves out what cannot. Arithmetic with
//! no device in it, like [`shadow`](crate::shadow)'s fitting, so that the part
//! that can go wrong is a unit test rather than a hole in a picture.
//!
//! **The planes come from the matrix, not from the camera.** Whatever carries
//! the world into clip space is exactly what the hardware clips against:
//! `-w <= x <= w`, `-w <= y <= w` and, in wgpu's depth range, `0 <= z <= w`.
//! Each of the six is a plane in the world, read off the rows of the matrix, so
//! one function serves a perspective view and an orthographic cascade alike,
//! and a box this leaves out is one the hardware would have clipped whole.
//! That is the claim everything here rests on: **the picture with the test and
//! the picture without it are the same picture**, and all that differs is
//! what the frame spent getting there.
//!
//! **A box, not a sphere.** Every engine read tests a box, and the two that
//! test a sphere as well do it in front of the box, as a cheaper way of saying
//! no. A sphere on its own is loose for anything long: the ball around a
//! two-by-twenty pillar is seventeen units across and reaches the camera from
//! behind it. The box here is the mesh's own, carried by the entity's matrix
//! rather than squared up into a world-aligned one, so a turned beam is tested
//! as the beam it is. The ball around the box is kept for what those two keep
//! it for, which is speed: it settles most planes with one product where the
//! box takes four, and it only ever says what the box would have said. @ref
//! [`Frustum::holds`].
//!
//! **What bones move is bounded where the bones put it.** A skinned mesh's own
//! box is the shape it was modeled in, and a ragdoll carries everything in its
//! bones while its entity stays where the character fell from. So each bone
//! gets the box of the vertices it moves, and a frame carries each through
//! that bone's matrix. @ref [`bones`] and [`posed`].

use colby_core::{
	abi::{MeshData, skeleton::MAX_BONES},
	glam::{Mat4, Vec3, Vec4},
};

/// The console variable that turns the test off.
///
/// **On, and not saved.** Leaving out what a pass cannot see is not a setting
/// anybody tunes: with it on, the frame is the same picture for less. What the
/// switch is for is the other way round - drawing everything, which is how
/// what the test saves is measured and how a picture is shown not to depend on
/// it. One engine read has exactly this, on by default; the others have no
/// switch at all.
pub const ENABLED: &str = "r.cull";

/// How much further out than the arithmetic says a box has to be before it is
/// left out, as a share of the numbers the arithmetic was done with.
///
/// The hardware clips each vertex in single precision along a path of its own,
/// so a box this finds exactly on a plane may have a corner the hardware puts
/// a hair inside it. Leaving out only what is out by more than rounding can
/// explain keeps "left out" strictly inside "clipped whole"; the cost is the
/// odd box that grazes an edge being drawn, which the clipper then removes.
///
/// @note: no test can see this, and a mutation pass that set it to nought
/// passed every one. What it guards is a sliver thinner than rounding, and a
/// sliver that thin covers no sample of any pixel. It stays because it is the
/// one line that says the rule, which is that the test errs towards drawing.
const SLACK: f32 = 1.0e-4;

/// Six planes, each facing inwards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
	/// `[nx, ny, nz, d]` with the normal of unit length: a point `p` is on the
	/// inside of one when `n . p + d` is not below nought.
	planes: [Vec4; 6],
}

impl Frustum {
	/// The volume a matrix keeps.
	///
	/// @param matrix - world space into clip space: the camera's
	/// view-projection, or one cascade's
	/// @return its planes, left, right, bottom, top, near and far
	#[must_use]
	pub fn of(matrix: Mat4) -> Self {
		let (x, y, z, w) = (matrix.row(0), matrix.row(1), matrix.row(2), matrix.row(3));

		Self {
			// the near one is `z` on its own rather than `w + z`: wgpu's depth
			// runs from nought, not from minus one.
			planes: [w + x, w - x, w + y, w - y, z, w - z].map(inwards),
		}
	}

	/// Whether any of a box can be inside.
	///
	/// The box is given as it stands in the world. @ref [`Placed`]. Along a
	/// plane's normal it reaches as far as the lengths of its half-edges along
	/// that normal added together, and it is out only when its middle is
	/// further behind the plane than that.
	///
	/// **Most planes are settled before the box is asked.** A middle on the
	/// inside of a plane is not behind it whatever the box's shape, and one
	/// further behind than the ball around the box reaches is behind it
	/// whatever the shape too; only a middle between the two needs the three
	/// products the box's own reach costs. Asking the box every time was
	/// measured on a thousand entities against five volumes, and cost more on
	/// the processor than leaving things out saved the graphics card.
	///
	/// @param placed - the box, in the world
	/// @return `false` only when the whole box is behind one of the planes
	#[must_use]
	pub fn holds(&self, placed: &Placed) -> bool {
		self.planes.iter().all(|plane| {
			let normal = plane.truncate();
			let distance = normal.dot(placed.center) + plane.w;

			if distance >= 0.0 {
				return true;
			}

			if behind(distance, placed.round) {
				return false;
			}

			let reach = placed
				.edges
				.iter()
				.map(|edge| normal.dot(*edge).abs())
				.sum::<f32>();

			!behind(distance, reach)
		})
	}

	/// Whether any of a ball can be inside.
	///
	/// For what reaches the same distance every way, which is a lamp: nothing
	/// further from it than its range is lit by it at all.
	///
	/// @param center - the middle of the ball, in the world
	/// @param radius - how far it reaches
	/// @return `false` only when the whole ball is behind one of the planes
	#[must_use]
	pub fn holds_ball(&self, center: Vec3, radius: f32) -> bool {
		self.planes
			.iter()
			.all(|plane| !behind(plane.truncate().dot(center) + plane.w, radius.max(0.0)))
	}
}

/// Whether something reaching `reach` either way from a point `distance` in
/// front of a plane is wholly behind it.
///
/// Asked this way round so that a distance that is not a number answers no: a
/// box that has blown up is drawn, as the hardware would draw it, rather than
/// quietly left out.
fn behind(distance: f32, reach: f32) -> bool {
	let margin = SLACK * (1.0 + distance.abs() + reach);

	distance < -(reach + margin)
}

/// A plane scaled so that its normal is of unit length.
///
/// A matrix that is not a projection at all - the zeros the cascades hold while
/// shadows are off - has rows of nothing, and a row of nothing is not a plane.
/// It becomes one that keeps everything, which is the only answer a test that
/// cannot tell is allowed to give.
fn inwards(plane: Vec4) -> Vec4 {
	let length = plane.truncate().length();

	if length.is_finite() && length > f32::MIN_POSITIVE {
		plane / length
	} else {
		Vec4::W
	}
}

/// A box as it stands in the world, and the ball around it.
///
/// Worked out once per entity per frame and asked of five volumes, which is
/// why the ball travels with the box rather than being worked out per test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
	/// The middle, in the world.
	pub center: Vec3,

	/// The three half-edges, in the world: a model matrix's columns times a
	/// mesh's half-extents, which is a turned and stretched box exactly.
	pub edges: [Vec3; 3],

	/// How far from the middle any corner can be.
	pub round: f32,
}

impl Placed {
	/// A box from its middle and its three half-edges.
	///
	/// The ball's radius is the half-edges' lengths added together, which
	/// holds however the three lean on one another, and a hair more: the
	/// three products the box's reach is made of are each rounded, and a reach
	/// rounded past its ball would let the ball leave out a box the box itself
	/// would keep.
	///
	/// @param center - the middle, in the world
	/// @param edges - the three half-edges, in the world
	#[must_use]
	pub fn new(center: Vec3, edges: [Vec3; 3]) -> Self {
		let lengths = edges.map(Vec3::length);

		Self {
			center,
			edges,
			round: (lengths[0] + lengths[1] + lengths[2]) * 4.0_f32.mul_add(f32::EPSILON, 1.0),
		}
	}
}

/// A box: its middle, and how far it reaches along each axis.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bounds {
	/// The middle.
	pub center: Vec3,

	/// Half its size along each axis.
	pub half: Vec3,
}

impl Bounds {
	/// The box between two corners.
	///
	/// @param low - the corner nearest minus infinity on every axis
	/// @param high - the one nearest plus infinity
	#[must_use]
	pub fn between(low: Vec3, high: Vec3) -> Self {
		Self {
			center: (low + high) * 0.5,
			half: (high - low) * 0.5,
		}
	}

	/// The box around every vertex of a mesh, in the shape it was modeled in.
	#[must_use]
	pub fn of(data: &MeshData) -> Self {
		let (low, high) = data.bounds();

		Self::between(low, high)
	}

	/// This box carried by a matrix, as the box it becomes.
	///
	/// The middle carried as a point, and the three half-edges as the matrix's
	/// columns scaled by the half-extents - which is exact however the matrix
	/// turns or stretches, and is what [`Frustum::holds`] takes.
	///
	/// @param matrix - model space into the world
	/// @return the box, in the world
	#[must_use]
	pub fn carried(self, matrix: Mat4) -> Placed {
		Placed::new(matrix.transform_point3(self.center), [
			matrix.x_axis.truncate() * self.half.x,
			matrix.y_axis.truncate() * self.half.y,
			matrix.z_axis.truncate() * self.half.z,
		])
	}

	/// The upright box around this one carried by a matrix.
	///
	/// Not exact the way [`carried`](Self::carried) is: the upright box around
	/// a turned one is bigger than it. It is what putting several bones' boxes
	/// together needs, because the union of two turned boxes is not a box of
	/// any kind and the union of two upright ones is.
	///
	/// @param matrix - what carries it
	/// @return the smallest upright box holding the carried one
	#[must_use]
	pub fn around(self, matrix: Mat4) -> Self {
		Self {
			center: matrix.transform_point3(self.center),
			half: matrix.x_axis.truncate().abs() * self.half.x
				+ matrix.y_axis.truncate().abs() * self.half.y
				+ matrix.z_axis.truncate().abs() * self.half.z,
		}
	}
}

/// Each bone's box: every vertex it moves at all, in the shape the mesh was
/// modeled in.
///
/// Indexed by bone, and a bone no vertex names has no box. A vertex naming a
/// bone past [`MAX_BONES`] is filed under one slot past the widest skeleton
/// the loader lets through, which is where the shader's clamp sends it too:
/// past the end of a pose's run is the run's last matrix, and every run is
/// shorter than that slot. @ref [`posed`].
///
/// @param data - the mesh; one nothing moves has an empty skin and no boxes
/// @return one box, or none, per bone
#[must_use]
pub fn bones(data: &MeshData) -> Vec<Option<Bounds>> {
	let mut corners: Vec<Option<(Vec3, Vec3)>> = Vec::new();

	for (vertex, skin) in data.vertices.iter().zip(&data.skin) {
		let at = Vec3::from_array(vertex.position);

		for (bone, weight) in skin.bones.iter().zip(skin.weights) {
			if weight > 0 {
				widen(&mut corners, usize::from(*bone).min(MAX_BONES), at);
			}
		}
	}

	corners
		.into_iter()
		.map(|held| held.map(|(low, high)| Bounds::between(low, high)))
		.collect()
}

/// Grows one bone's corners to take in a point.
fn widen(corners: &mut Vec<Option<(Vec3, Vec3)>>, bone: usize, at: Vec3) {
	if corners.len() <= bone {
		corners.resize(bone + 1, None);
	}

	if let Some(slot) = corners.get_mut(bone) {
		*slot = Some(slot.map_or((at, at), |(low, high)| (low.min(at), high.max(at))));
	}
}

/// Where bones have put a mesh this frame, as one upright box in its model's
/// space.
///
/// Each bone's box through the matrix the shader moves that bone's vertices
/// with, and the boxes put together. A vertex several bones share lands where
/// the weighted sum of their matrices puts it, which is between where each of
/// them alone would, so the union still holds it. A bone past the end of the
/// run reads the run's last matrix, as `skinning` in `shader.wgsl` does.
///
/// @param bones - each bone's box, from [`bones`]
/// @param joints - this pose's matrices, one per bone of its skeleton
/// @return the box, or `None` when there is no run or nothing in the mesh is
/// moved by one, in which case the mesh stands as it was modeled
#[must_use]
pub fn posed(bones: &[Option<Bounds>], joints: &[Mat4]) -> Option<Bounds> {
	let last = joints.len().checked_sub(1)?;
	let mut corners: Option<(Vec3, Vec3)> = None;

	for (bone, held) in bones.iter().enumerate() {
		let (Some(held), Some(joint)) = (held, joints.get(bone.min(last))) else {
			continue;
		};

		let moved = held.around(*joint);
		let (low, high) = (moved.center - moved.half, moved.center + moved.half);

		corners =
			Some(corners.map_or((low, high), |(below, above)| (below.min(low), above.max(high))));
	}

	corners.map(|(low, high)| Bounds::between(low, high))
}

/// How much of the world one frame drew.
///
/// Counts rather than durations, and stable the way a pass count is: a time is
/// a fact about the afternoon, and how much of a project its camera leaves out
/// at a given step is a fact about the project.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Drawn {
	/// Entities with a mesh to draw that nothing hides, whether or not
	/// anything drew them.
	pub meshes: usize,

	/// How many of them the picture drew.
	pub seen: usize,

	/// How many times a cascade drew one, the four added together.
	///
	/// With the test off every solid entity is drawn into all four, which is
	/// four times what the picture's solid half holds.
	pub cast: usize,

	/// Entities with a mesh that are hidden, by their own word or by something
	/// they hang off, and so were put in no list at all.
	///
	/// Beside [`meshes`](Self::meshes) rather than inside it: the two add up to
	/// every entity with a mesh, and hiding a subtree moves this by the size of
	/// it wherever the camera is. @ref
	/// [`Entities::shown`](colby_core::abi::Entities::shown).
	pub hidden: usize,

	/// How many lamps the frame carried to the shader, after every rule that
	/// decides it: the room a frame has, whether a lamp's reach touches the
	/// view, and whether it is shown.
	pub lamps: usize,
}

#[cfg(test)]
mod tests {
	use core::f32::consts::{FRAC_PI_2, FRAC_PI_4};

	use colby_core::{
		abi::{Camera, MeshVertex, SkinVertex, Transform},
		glam::Quat,
	};

	use super::*;
	use crate::shadow;

	/// A camera at the origin looking down `-z`, square and a right angle
	/// across, so that the four sides of its view are the planes `x = z`,
	/// `x = -z`, `y = z` and `y = -z` and a test can say where one is.
	fn looking() -> Frustum {
		let camera = Camera {
			position: Vec3::ZERO,
			target: Vec3::NEG_Z,
			up: Vec3::Y,
			fov_y: FRAC_PI_2,
			near: 0.1,
			far: 100.0,
		};

		Frustum::of(camera.view_projection(1.0))
	}

	/// An upright cube of a half-size, standing at a point.
	fn cube(at: Vec3, half: f32) -> Placed {
		Bounds {
			center: Vec3::ZERO,
			half: Vec3::splat(half),
		}
		.carried(Mat4::from_translation(at))
	}

	/// Whether the view holds a cube.
	fn sees(at: Vec3, half: f32) -> bool { looking().holds(&cube(at, half)) }

	#[test]
	fn a_box_in_front_of_the_eye_is_held_and_one_behind_it_is_not() {
		assert!(sees(Vec3::new(0.0, 0.0, -10.0), 1.0), "straight ahead");
		assert!(!sees(Vec3::new(0.0, 0.0, 10.0), 1.0), "straight behind");
		assert!(sees(Vec3::ZERO, 1.0), "and one the eye is standing inside of is held too");
	}

	#[test]
	fn each_side_turns_away_a_box_wholly_beyond_it_and_keeps_one_across_it() {
		// ten units out the sides of a right-angled view are ten units either
		// way, so a cube of half a unit centered on a side straddles it and one
		// two units further out is past it by one and a half
		for way in [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y] {
			let across = way * 10.0 + Vec3::new(0.0, 0.0, -10.0);
			let past = way * 12.0 + Vec3::new(0.0, 0.0, -10.0);

			assert!(sees(across, 0.5), "a cube across the {way} side is held");
			assert!(!sees(past, 0.5), "and one beyond it is not");
		}

		// and the far end, a hundred units out
		assert!(sees(Vec3::new(0.0, 0.0, -100.0), 0.5), "across the far plane");
		assert!(!sees(Vec3::new(0.0, 0.0, -102.0), 0.5), "beyond it");
	}

	#[test]
	fn a_bar_lying_along_a_side_just_outside_it_is_left_out_and_one_turned_in_is_not() {
		// a bar ten long and a fifth thick, turned an eighth of a turn about y
		// so that its length runs along the right-hand side of the view, and
		// standing a half unit outside it. **The case an upright box gets
		// wrong**: squared up into the world its length would reach across
		// the side, and it would be drawn.
		let outside = Transform {
			position: Vec3::new(10.0, 0.0, -10.0) + Vec3::new(1.0, 0.0, 1.0).normalize() * 0.5,
			rotation: Quat::from_rotation_y(FRAC_PI_4),
			scale: Vec3::new(10.0, 0.2, 0.2),
		};
		let unit = Bounds {
			center: Vec3::ZERO,
			half: Vec3::splat(0.5),
		};

		let placed = unit.carried(outside.matrix());

		assert!(!looking().holds(&placed), "the bar alongside the side is out");

		// the same bar at the same place turned the other way, so its length
		// points in across the side: some of it is inside, so it is held
		let placed = unit.carried(
			Transform {
				rotation: Quat::from_rotation_y(-FRAC_PI_4),
				..outside
			}
			.matrix(),
		);

		assert!(looking().holds(&placed), "the bar pointing in is held");
	}

	/// The box test with no ball in front of it, the way it was first written.
	fn plainly(frustum: &Frustum, placed: &Placed) -> bool {
		frustum.planes.iter().all(|plane| {
			let normal = plane.truncate();
			let reach = placed
				.edges
				.iter()
				.map(|edge| normal.dot(*edge).abs())
				.sum::<f32>();

			!behind(normal.dot(placed.center) + plane.w, reach)
		})
	}

	/// A point somewhere in a slab of world around and ahead of the eye, the
	/// same one for the same step on every run.
	fn spot(step: u16) -> Vec3 {
		let at = f32::from(step);

		Vec3::new(
			(at * 0.618_034).fract().mul_add(80.0, -40.0),
			(at * 0.414_214).fract().mul_add(40.0, -20.0),
			(at * 0.732_051).fract().mul_add(140.0, -120.0),
		)
	}

	#[test]
	fn the_ball_in_front_of_the_box_never_answers_differently_from_the_box() {
		// four thousand boxes of three shapes in three attitudes, scattered
		// around and ahead of the eye: every answer the quick way has to be
		// the answer the long way, and the sweep has to have asked both ways
		// often enough for that to mean something
		let view = looking();
		let shapes = [Vec3::splat(0.5), Vec3::new(8.0, 0.2, 0.2), Vec3::new(3.0, 0.1, 3.0)];
		let turns = [
			Quat::IDENTITY,
			Quat::from_rotation_y(0.5),
			Quat::from_euler(colby_core::glam::EulerRot::XYZ, 0.7, -0.3, 1.1),
		];
		let (mut held, mut left) = (0, 0);

		for step in 0..4000_u16 {
			let place = spot(step);
			let placed = Bounds {
				center: Vec3::ZERO,
				half: shapes[usize::from(step % 3)],
			}
			.carried(Mat4::from_rotation_translation(turns[usize::from((step / 3) % 3)], place));
			let answer = view.holds(&placed);

			assert_eq!(answer, plainly(&view, &placed), "a box at {place}");

			if answer {
				held += 1;
			} else {
				left += 1;
			}
		}

		assert!(
			held > 400 && left > 400,
			"the sweep asked both ways: {held} held and {left} left out"
		);
	}

	#[test]
	fn a_ball_is_held_while_any_of_it_reaches_in() {
		let view = looking();

		// five units behind the eye, where the near plane is a tenth in front
		assert!(!view.holds_ball(Vec3::new(0.0, 0.0, 5.0), 4.0), "a ball four across");
		assert!(view.holds_ball(Vec3::new(0.0, 0.0, 5.0), 6.0), "and one six across reaches in");
		assert!(
			!view.holds_ball(Vec3::new(0.0, 0.0, 5.0), -8.0),
			"and a reach below nought is none rather than a ball turned inside out"
		);
	}

	#[test]
	fn a_matrix_that_is_no_projection_keeps_everything() {
		// what the cascades hold while shadows are off, and what a camera that
		// has blown up hands over: neither can say where anything is, and a
		// test that cannot say has to keep rather than drop
		for broken in [Mat4::ZERO, Mat4::NAN] {
			let placed = cube(Vec3::new(1.0e6, -1.0e6, 1.0e6), 1.0);

			assert!(Frustum::of(broken).holds(&placed), "{broken} drops nothing");
			assert!(Frustum::of(broken).holds_ball(placed.center, 1.0), "{broken} drops no ball");
		}

		assert!(
			looking().holds(&Placed::new(Vec3::ZERO, [Vec3::NAN; 3])),
			"and a box whose edges are not numbers is drawn rather than dropped"
		);
	}

	#[test]
	fn a_cascade_keeps_what_its_depth_pass_would_draw_and_drops_what_it_would_clip() {
		// the orthographic half of `Frustum::of`, against the matrices the
		// cascades really use: its six planes have to be the box the depth
		// pass clips to, which the corners of the box say independently
		let camera = Camera {
			position: Vec3::ZERO,
			target: Vec3::NEG_Z,
			..Camera::DEFAULT
		};
		let light = Vec3::new(-0.3, -1.0, -0.2).normalize();
		let cascades = shadow::fit(&camera, 16.0 / 9.0, light, shadow::DEFAULT_DISTANCE);
		let nearest = cascades.matrices[0];
		let box_of = Frustum::of(nearest);

		// a unit cube two units down the line of sight, inside the first slice
		let ahead = Vec3::new(0.0, 0.0, -2.0);
		// the same thirty units back along the light: between the light and
		// the slice, where a roof would be, and inside the fifty units the
		// cascade looks back for casters
		let above = ahead - light * 30.0;
		// and ninety back, past where the cascade looks at all
		let beyond = ahead - light * 90.0;

		for (at, kept) in [(ahead, true), (above, true), (beyond, false)] {
			let placed = cube(at, 0.5);
			let landed = nearest.project_point3(at);
			let clipped =
				landed.x.abs() > 1.0 || landed.y.abs() > 1.0 || !(0.0..=1.0).contains(&landed.z);

			assert_eq!(box_of.holds(&placed), kept, "a cube at {at}");
			assert_eq!(
				!clipped, kept,
				"and the matrix itself agrees about {at}, landing at {landed}"
			);
		}
	}

	/// A mesh of two small cubes of vertices, one at each end of a bar: the
	/// near one moved by bone nought, the far one by whichever bone is named.
	fn two_ended(far_bone: u16) -> MeshData {
		let mut data = MeshData::default();

		for (along, bone) in [(0.0_f32, 0), (2.0, far_bone)] {
			for corner in [Vec3::splat(-0.25), Vec3::splat(0.25)] {
				data.vertices.push(MeshVertex {
					position: (corner + Vec3::X * along).to_array(),
					..MeshVertex::default()
				});
				data.skin.push(SkinVertex::rigid(bone));
			}
		}

		data
	}

	#[test]
	fn each_bone_is_bounded_by_the_vertices_it_moves_and_no_others() {
		let boxes = bones(&two_ended(1));

		assert_eq!(boxes.len(), 2, "two bones named, two slots");
		assert_eq!(boxes[0], Some(Bounds::between(Vec3::splat(-0.25), Vec3::splat(0.25))));
		assert_eq!(
			boxes[1],
			Some(Bounds::between(Vec3::new(1.75, -0.25, -0.25), Vec3::new(2.25, 0.25, 0.25)))
		);
		assert!(bones(&MeshData::default()).is_empty(), "and a mesh nothing moves has none");
	}

	#[test]
	fn a_weight_of_nothing_is_no_claim_on_a_vertex() {
		let mut data = two_ended(1);

		// the far vertices also name bone two, with nothing behind the name
		for skin in &mut data.skin {
			skin.bones[1] = 2;
		}

		let boxes = bones(&data);

		assert!(boxes.get(2).is_none_or(Option::is_none), "bone two moves nothing: {boxes:?}");
	}

	#[test]
	fn a_bone_past_the_widest_skeleton_is_filed_where_the_clamp_sends_it() {
		let boxes = bones(&two_ended(u16::MAX));

		assert_eq!(boxes.len(), MAX_BONES + 1, "one slot past the widest, not sixty thousand");
		assert!(boxes[MAX_BONES].is_some(), "and the far end is in it");
	}

	#[test]
	fn a_bone_bent_away_carries_its_box_with_it() {
		let boxes = bones(&two_ended(1));
		let up = Mat4::from_translation(Vec3::Y * 10.0);
		let moved = posed(&boxes, &[Mat4::IDENTITY, up]).expect("both bones have a box");

		let lifted = Vec3::new(2.0, 10.0, 0.0);
		let (low, high) = (moved.center - moved.half, moved.center + moved.half);

		assert!(
			lifted.cmpge(low).all() && lifted.cmple(high).all(),
			"the far end, ten up, is inside {moved:?}"
		);
		assert!(
			Vec3::ZERO.cmpge(low).all() && Vec3::ZERO.cmple(high).all(),
			"and so is the near end, which did not move"
		);
		assert!(
			(high.y - 10.25).abs() < 1.0e-5,
			"and it reaches no higher than the far end does: {high}"
		);
	}

	#[test]
	fn a_bone_past_the_run_reads_the_last_matrix_as_the_shader_does() {
		// bone five, in a pose whose skeleton has two: the shader reads the
		// second matrix for it, so the box has to go where that one puts it
		let boxes = bones(&two_ended(5));
		let up = Mat4::from_translation(Vec3::Y * 10.0);
		let moved = posed(&boxes, &[Mat4::IDENTITY, up]).expect("there are boxes");

		assert!(
			(moved.center.y + moved.half.y - 10.25).abs() < 1.0e-5,
			"the far end went up with the last bone: {moved:?}"
		);
	}

	#[test]
	fn a_mesh_with_no_run_stands_as_it_was_modeled() {
		let boxes = bones(&two_ended(1));

		assert_eq!(posed(&boxes, &[]), None, "no matrices, no posed box");
		assert_eq!(posed(&[], &[Mat4::IDENTITY]), None, "and no boxes, none either");
	}

	#[test]
	fn a_turned_box_carried_upright_holds_every_corner_of_the_turned_one() {
		let held = Bounds {
			center: Vec3::new(1.0, 2.0, 3.0),
			half: Vec3::new(0.5, 1.0, 2.0),
		};
		let turn = Mat4::from_scale_rotation_translation(
			Vec3::new(2.0, 1.0, 0.5),
			Quat::from_euler(colby_core::glam::EulerRot::XYZ, 0.3, 0.7, -0.4),
			Vec3::new(-3.0, 1.0, 4.0),
		);
		let around = held.around(turn);
		let (low, high) = (around.center - around.half, around.center + around.half);

		for sign in 0..8_u8 {
			let pick = |bit: u8| if (sign & bit) == 0 { -1.0 } else { 1.0 };
			let corner = held.center + held.half * Vec3::new(pick(1), pick(2), pick(4));
			let landed = turn.transform_point3(corner);

			assert!(
				landed.cmpge(low - 1.0e-4).all() && landed.cmple(high + 1.0e-4).all(),
				"corner {corner} lands at {landed}, outside {around:?}"
			);
		}
	}
}
