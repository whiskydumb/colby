//! Which of a mesh's levels a frame draws for one thing.
//!
//! **The coarsest level that stands within a pixel of the mesh.** A compiled
//! mesh carries coarser ways to draw it, each with how far it may stand from
//! the mesh in the mesh's own units - @ref
//! [`Level`](colby_core::abi::mesh::Level). Seen from a distance, that far
//! covers a number of pixels, and while the number is no more than
//! [`THRESHOLD`] says, the level and the mesh are the same picture to within a
//! pixel. So the frame walks the levels finest first and draws the last one
//! that still fits.
//!
//! **Arithmetic with no device in it**, like [`cull`](crate::cull), and asked
//! once per thing per frame where the frame decides which of its lists the
//! thing is in: the level decided there is the one the picture, the pass before
//! it, the test for what is behind something nearer and every shadow map draw,
//! so none of them can disagree about which surface a thing has.
//!
//! **Three choices, each for a reason:**
//!
//! - **Pixels, not a distance.** A distance written beside a level would be a
//!   number somebody picked; how far a level stands from the mesh is a fact the
//!   compiler measured, and a pixel is a fact about the picture. The one number
//!   a person chooses is how many pixels a level may be off.
//! - **The nearest point of the thing's box**, not its middle and not the ball
//!   around it. A long wall seen end on is near at one end; its middle is far
//!   and the ball around it reaches half its length towards the eye either way.
//! - **Nothing is remembered between frames.** A thing at the edge of a level
//!   switches as soon as it crosses it, which is one pixel's worth of change,
//!   and a picture taken in one frame is the picture a window draws in that
//!   frame.
//!
//! **Its largest size counts**: a mesh stretched along one axis stands further
//! from its levels along that axis, and the level has to fit along all of them.

use colby_core::glam::{Mat4, Vec3};

use crate::cull::Placed;

/// The console variable that says how many pixels a level may be off.
///
/// **One, and not saved.** A setting a machine could trade for speed, but what
/// the variable is for first is the other way round: nought draws every mesh
/// whole, which is how a picture is shown to be the one a build from before
/// levels took and how what they save is measured.
pub const THRESHOLD: &str = "r.detail";

/// How many pixels a level may be off when nothing says otherwise.
pub const DEFAULT_THRESHOLD: f32 = 1.0;

/// What one frame asks every thing's level against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eye {
	/// Where the eye is.
	eye: Vec3,

	/// How many pixels one unit covers one unit away from the eye: the
	/// picture's height over twice the tangent of half its field of view.
	pixels: f32,

	/// How many pixels a level may be off; nought or less draws every mesh
	/// whole.
	threshold: f32,
}

impl Eye {
	/// What a frame drawn through a camera asks with.
	///
	/// @param eye - where the camera is
	/// @param projection - its projection alone, not the view; the second
	/// column's second row is one over the tangent of half the vertical field
	/// @param height - how many pixels tall the picture is, the rectangle it is
	/// drawn into when it is drawn into one
	/// @param threshold - how many pixels a level may be off
	#[must_use]
	pub fn new(eye: Vec3, projection: Mat4, height: u32, threshold: f32) -> Self {
		Self {
			eye,
			pixels: projection.y_axis.y
				* f32::from(u16::try_from(height).unwrap_or(u16::MAX))
				* 0.5,
			threshold,
		}
	}

	/// The level to draw one thing at: nought is the mesh itself.
	///
	/// Written as a product against a product rather than a division, so a
	/// thing at the eye - nought away, where every level would cover the whole
	/// picture - is drawn whole rather than divided by nothing.
	///
	/// @param placed - the thing's box, in the world
	/// @param size - the thing's largest scale along any axis
	/// @param errors - how far each coarser level stands from the mesh, finest
	/// first
	/// @return how many of the levels fit, which is the one to draw
	///
	/// @note a threshold that is not a number, and a mesh with no levels, get
	/// the answer the walk below would give them anyway - nothing fits either
	/// way. They are asked first so that a mesh drawn whole costs no distance,
	/// which is almost every thing in a world, and so the answer for a broken
	/// threshold does not rest on how a comparison with one reads.
	#[must_use]
	pub fn level(&self, placed: &Placed, size: f32, errors: &[f32]) -> usize {
		if self.threshold.is_nan() || self.threshold <= 0.0 || errors.is_empty() {
			return 0;
		}

		let away = self.threshold * distance(placed, self.eye);

		errors
			.iter()
			.take_while(|error| **error * size * self.pixels <= away)
			.count()
	}

	/// Whether one thing stands under a number of pixels across.
	///
	/// Asked of every solid thing in view by a frame that draws the pass before
	/// the scene around the test for what is behind something nearer, @ref
	/// [`cover::SIZE`](crate::cover::SIZE), with the measure a level is asked
	/// with: pixels, from the nearest point of the box.
	///
	/// **The ball around the box, not the box**: as wide as the box's longest
	/// diagonal however it is turned, so a thing called small is never wider
	/// than the number says. Written as a product against a product, like
	/// [`level`](Self::level), so a thing the eye stands inside is never small,
	/// and nor is anything against a size of nought or less or not a number.
	///
	/// **Most things are settled before the nearest point is asked**, the way
	/// most planes are in [`Frustum::holds`](crate::cull::Frustum::holds): the
	/// box's nearest point is no further than its middle, which the box holds,
	/// and no nearer than the ball around it reaches. Only a thing whose size
	/// in pixels lies between the two answers takes the three projections:
	/// taking them for all of a street's seven hundred things was measured at
	/// seventeen microseconds of its frame, and settling most of them first at
	/// four.
	///
	/// @note: the two short answers are what a test over twenty thousand boxes
	/// holds against the nearest point alone, and a mutation pass that took
	/// both out passed everything - they change what an answer costs, never
	/// what it is.
	///
	/// @param placed - the thing's box, in the world
	/// @param size - how many pixels across
	#[must_use]
	pub fn under(&self, placed: &Placed, size: f32) -> bool {
		let across = placed
			.edges
			.iter()
			.map(|edge| edge.length_squared())
			.sum::<f32>()
			.sqrt() * 2.0;
		let wide = across * self.pixels;
		let middle = placed.center.distance(self.eye);

		if wide >= size * middle {
			return false;
		}

		if wide < size * (middle - across * 0.5) {
			return true;
		}

		wide < size * distance(placed, self.eye)
	}
}

/// How far from a point the nearest point of a box is.
///
/// The box's three half-edges are square to one another, which a translation,
/// a rotation and a scale always leave them; the point is moved into the box's
/// own frame, held to the box along each edge, and moved back. Nought for a
/// point inside the box.
///
/// @param placed - the box, in the world
/// @param from - the point
#[must_use]
pub fn distance(placed: &Placed, from: Vec3) -> f32 {
	let off = from - placed.center;
	let nearest = placed
		.edges
		.iter()
		.fold(placed.center, |nearest, edge| {
			let length = edge.length_squared();

			if length > f32::MIN_POSITIVE {
				nearest + *edge * (off.dot(*edge) / length).clamp(-1.0, 1.0)
			} else {
				nearest
			}
		});

	from.distance(nearest)
}

#[cfg(test)]
mod tests {
	use core::f32::consts::FRAC_PI_4;

	use colby_core::{
		abi::{Camera, Transform},
		glam::Quat,
	};

	use super::*;
	use crate::cull::Bounds;

	/// A unit box turned and stretched, standing at a point.
	fn placed(transform: Transform) -> Placed {
		Bounds {
			center: Vec3::ZERO,
			half: Vec3::splat(0.5),
		}
		.carried(transform.matrix())
	}

	/// A camera of a vertical field of one radian, a picture 720 tall.
	fn looking(threshold: f32) -> Eye {
		let camera = Camera {
			position: Vec3::ZERO,
			target: Vec3::NEG_Z,
			fov_y: 1.0,
			..Camera::DEFAULT
		};

		Eye::new(camera.position, camera.projection(16.0 / 9.0), 720, threshold)
	}

	#[test]
	fn the_nearest_point_of_a_box_is_measured_to_along_each_of_its_own_edges() {
		let upright = placed(Transform::at(Vec3::new(0.0, 0.0, -10.0)));

		assert!((distance(&upright, Vec3::ZERO) - 9.5).abs() < 1.0e-5, "straight at a face");
		assert!(
			distance(&upright, Vec3::new(0.0, 0.0, -10.2)) < 1.0e-6,
			"and nothing from inside it"
		);

		// a corner: half a unit in on two axes, so the nearest point is the
		// corner itself
		let corner = distance(&upright, Vec3::new(3.5, 4.5, -10.0));

		assert!((corner - (9.0_f32 + 16.0).sqrt()).abs() < 1.0e-4, "to an edge: {corner}");

		// a long bar turned an eighth of a turn, its end towards the eye: the
		// ball around it would reach twice as far towards the eye as the bar
		let bar = placed(Transform {
			position: Vec3::new(0.0, 0.0, -20.0),
			rotation: Quat::from_rotation_y(FRAC_PI_4),
			scale: Vec3::new(20.0, 1.0, 1.0),
		});
		let along = distance(&bar, Vec3::ZERO);

		// worked out by hand: the near end is at (-7.07, 0, -12.93), and its
		// corner half a unit across towards the eye at (-6.72, 0, -12.58)
		assert!((along - 14.265).abs() < 0.01, "the near end of the bar: {along}");
		assert!(along < 20.0, "not its middle, twenty away");
		assert!(along > 20.0 - bar.round, "and not the ball around it, nine away: {along}");

		// a box as flat as a sheet of paper has an edge of no length, which is
		// not divided by
		let sheet = Bounds {
			center: Vec3::ZERO,
			half: Vec3::new(0.5, 0.0, 0.5),
		}
		.carried(Transform::at(Vec3::new(0.0, -2.0, -10.0)).matrix());
		let flat = distance(&sheet, Vec3::ZERO);

		assert!((flat - (4.0_f32 + 90.25).sqrt()).abs() < 1.0e-4, "to a sheet: {flat}");
	}

	#[test]
	fn a_level_is_drawn_while_it_stands_within_the_threshold_in_pixels() {
		let view = looking(1.0);
		let unit = placed(Transform::at(Vec3::new(0.0, 0.0, -100.5)));
		// one unit at a hundred units away covers 720 / (2 tan 0.5) / 100
		let per_unit = 720.0 / (2.0 * 0.5_f32.tan()) / 100.0;
		let errors = [0.4 / per_unit, 0.9 / per_unit, 1.9 / per_unit];

		assert_eq!(
			view.level(&unit, 1.0, &errors),
			2,
			"0.4 and 0.9 of a pixel fit, 1.9 does not"
		);
		assert_eq!(view.level(&unit, 2.0, &errors), 1, "twice the size, twice the pixels");
		assert_eq!(
			looking(2.0).level(&unit, 1.0, &errors),
			3,
			"and two pixels allowed, all three"
		);
		assert_eq!(view.level(&unit, 1.0, &[]), 0, "a mesh with no levels is the mesh");
	}

	#[test]
	fn a_level_exactly_the_threshold_off_is_drawn() {
		// numbers a float holds exactly: a face eight units away, 256 pixels a
		// unit one unit away and so 32 at eight, and a level a thirty-second of
		// a unit off - one pixel, which is no more than one
		let view = Eye {
			eye: Vec3::ZERO,
			pixels: 256.0,
			threshold: 1.0,
		};
		let eight_away = placed(Transform::at(Vec3::new(0.0, 0.0, -8.5)));

		assert_eq!(
			distance(&eight_away, Vec3::ZERO).to_bits(),
			8.0_f32.to_bits(),
			"the face is eight away to the bit"
		);
		assert_eq!(view.level(&eight_away, 1.0, &[1.0 / 32.0]), 1, "a pixel off is drawn");
		assert_eq!(
			view.level(&eight_away, 1.0, &[1.0 / 32.0 + 1.0e-6]),
			0,
			"and a hair past a pixel is not"
		);
	}

	#[test]
	fn a_threshold_of_nought_or_worse_draws_every_mesh_whole() {
		let far = placed(Transform::at(Vec3::new(0.0, 0.0, -1.0e6)));
		// the first no further from the mesh than nothing, which is what a
		// simplifier that took away only what was flat measures: nought pixels
		// is within a threshold of nought, and the mesh is still drawn whole
		let errors = [0.0, 2.0e-9];

		assert_eq!(
			looking(1.0).level(&far, 1.0, &errors),
			2,
			"a million units away, the coarsest"
		);

		for off in [0.0, -1.0, f32::NAN] {
			assert_eq!(
				looking(off).level(&far, 1.0, &errors),
				0,
				"and at {off}, the mesh itself"
			);
		}
	}

	#[test]
	fn a_thing_the_eye_stands_inside_is_drawn_whole() {
		let around = placed(Transform {
			scale: Vec3::splat(4.0),
			..Transform::IDENTITY
		});

		assert_eq!(looking(1.0).level(&around, 1.0, &[1.0e-12]), 0, "nought away is never far");
	}

	#[test]
	fn a_thing_under_the_size_is_small_and_one_exactly_that_size_across_is_not() {
		// numbers a float holds exactly: a rod a unit long - its ball a unit
		// across - eight units away, at 256 pixels a unit one unit away, is 32
		// pixels across, which is not under 32
		let view = Eye {
			eye: Vec3::ZERO,
			pixels: 256.0,
			threshold: 1.0,
		};
		let rod = Bounds {
			center: Vec3::ZERO,
			half: Vec3::new(0.5, 0.0, 0.0),
		}
		.carried(Transform::at(Vec3::new(0.0, 0.0, -8.0)).matrix());

		assert_eq!(
			distance(&rod, Vec3::ZERO).to_bits(),
			8.0_f32.to_bits(),
			"the rod is eight away to the bit"
		);
		assert!(!view.under(&rod, 32.0), "exactly 32 across is not under 32");
		assert!(view.under(&rod, 32.000_008), "and a hair more than 32 is");
		assert!(view.under(&rod, 64.0), "as is anything larger");

		// and a box whose middle is further than its nearest point and nearer
		// than its ball, so that only the nearest point settles it: its ball is
		// a unit and a quarter across, 320 pixels at 256 a unit, and its near
		// face is eight away - exactly 40 pixels
		let plate = placed(Transform {
			position: Vec3::new(0.0, 0.0, -8.5),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(0.75, 0.0, 1.0),
		});

		assert_eq!(
			distance(&plate, Vec3::ZERO).to_bits(),
			8.0_f32.to_bits(),
			"the plate's face is eight away to the bit"
		);
		assert!(!view.under(&plate, 40.0), "exactly 40 across is not under 40");
		assert!(view.under(&plate, 40.000_004), "and a hair more than 40 is");
	}

	#[test]
	fn a_ball_is_measured_along_all_three_edges_from_the_nearest_point_of_the_box() {
		let view = looking(1.0);
		let per_unit = 720.0 / (2.0 * 0.5_f32.tan());
		// a box two by one by one, turned an eighth of a turn and standing twenty
		// away: its ball is the root of six across, its longest edge two, and the
		// half-edges added together four
		let slab = placed(Transform {
			position: Vec3::new(0.0, 0.0, -20.0),
			rotation: Quat::from_rotation_y(FRAC_PI_4),
			scale: Vec3::new(2.0, 1.0, 1.0),
		});
		let away = distance(&slab, Vec3::ZERO);
		let across = 6.0_f32.sqrt() * per_unit / away;

		assert!(away < 19.5, "the nearest point is nearer than its middle's face: {away}");
		assert!(view.under(&slab, across * 1.001), "just over what it measures, it is under");
		assert!(!view.under(&slab, across * 0.999), "and just short of it, it is not");
		assert!(
			!view.under(&slab, 2.2 * per_unit / away),
			"a size its longest edge alone would be under, its ball is not"
		);
		assert!(
			view.under(&slab, 3.0 * per_unit / away),
			"and a size its half-edges added together would not be under, its ball is"
		);
	}

	#[test]
	fn settling_a_thing_by_its_middle_and_its_ball_first_gives_the_nearest_point_s_answer() {
		// thousands of boxes of every size turned every way, near and far,
		// against sizes from a hair to hundreds of pixels: the answer the two
		// short ways give is the one the nearest point alone gives
		let view = looking(1.0);
		let mut seed = 0x2545_F491_u32;
		let mut next = move || {
			seed ^= seed << 13;
			seed ^= seed >> 17;
			seed ^= seed << 5;

			f32::from(u16::try_from(seed >> 16).unwrap_or(0)) / 65_535.0
		};
		let (mut small, mut large) = (0, 0);

		for _ in 0..20_000 {
			let box_ = placed(Transform {
				position: Vec3::new(
					next().mul_add(200.0, -100.0),
					next().mul_add(200.0, -100.0),
					next().mul_add(200.0, -100.0),
				),
				rotation: Quat::from_euler(
					colby_core::glam::EulerRot::YXZ,
					next() * 6.3,
					next() * 6.3,
					next() * 6.3,
				),
				scale: Vec3::new(
					next().mul_add(20.0, 0.01),
					next().mul_add(20.0, 0.01),
					next().mul_add(20.0, 0.01),
				),
			});
			let size = next().mul_add(400.0, 0.5);
			let across = box_
				.edges
				.iter()
				.map(|edge| edge.length_squared())
				.sum::<f32>()
				.sqrt() * 2.0;
			let nearest = across * view.pixels < size * distance(&box_, view.eye);
			let answer = view.under(&box_, size);

			assert_eq!(answer, nearest, "{box_:?} against {size} pixels");

			if answer {
				small += 1;
			} else {
				large += 1;
			}
		}

		assert!(small > 2_000 && large > 2_000, "{small} small and {large} large");
	}

	#[test]
	fn nothing_is_small_against_a_size_of_nought_or_worse_or_to_an_eye_inside_it() {
		let far = placed(Transform::at(Vec3::new(0.0, 0.0, -1.0e6)));
		let around = placed(Transform {
			scale: Vec3::splat(4.0),
			..Transform::IDENTITY
		});

		assert!(looking(1.0).under(&far, 1.0), "a million units away, a unit is small");

		for size in [0.0, -1.0, f32::NAN] {
			assert!(!looking(1.0).under(&far, size), "and nothing is under {size}");
		}

		assert!(!looking(1.0).under(&around, 1.0e6), "nought away is never small");
	}

	#[test]
	fn an_error_that_is_not_a_number_stops_the_walk() {
		let far = placed(Transform::at(Vec3::new(0.0, 0.0, -1000.0)));

		assert_eq!(
			looking(1.0).level(&far, 1.0, &[1.0e-6, f32::NAN, 2.0e-6]),
			1,
			"what comes after a broken level is not reached"
		);
	}

	#[test]
	fn further_away_is_never_finer() {
		let errors = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064];
		let view = looking(1.0);
		let mut before = 0;

		for step in 1..400_u16 {
			let away = f32::from(step) * 0.5;
			let level =
				view.level(&placed(Transform::at(Vec3::new(0.0, 0.0, -away))), 1.0, &errors);

			assert!(level >= before, "{level} at {away} after {before}");

			before = level;
		}

		assert_eq!(before, errors.len(), "and far enough away, the coarsest");
	}
}

/// What a frame draws a thing at a level with, held against the same frame with
/// that level drawn as a mesh of its own. On the device, through a capture.
#[cfg(test)]
mod drawn {
	use colby_core::{
		abi::{
			EntityId, Level, Material, MeshData, MeshId, MeshVertex, Renderable, Transform,
			Value, World, material::Blend,
		},
		glam::Vec2,
	};

	use super::*;
	use crate::{Capture, Image, Viewport, cover, cull::Drawn, prepass, scene::MSAA};

	/// How big every capture here is.
	const SIZE: (u32, u32) = (320, 240);

	/// Where the eye stands and what it looks at, in every world here.
	const EYE: Vec3 = Vec3::new(0.0, 1.5, 6.0);

	/// How far each coarser level of the ball stands from it.
	///
	/// Picked against the arithmetic rather than measured: one unit a unit
	/// away covers `240 / (2 tan 0.5)` = 219.66 pixels of these captures, so
	/// the first level fits a pixel from 2.2 units away and the second from
	/// 6.6.
	const ERRORS: [f32; 2] = [0.01, 0.03];

	/// A capture on the binary's one device, or `None` with no GPU.
	///
	/// One for each world a test draws: a capture keeps what it uploaded by
	/// the registry's slot and revision, and two fresh worlds that each
	/// registered one mesh hold two different meshes under the same two.
	fn capture() -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, SIZE.0, SIZE.1) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// A ball of radius a half over a latitude and longitude grid: 24 rings of
	/// 32, 1536 triangles.
	fn ball() -> MeshData {
		let (rings, segments) = (24_u16, 32_u16);
		let mut data = MeshData::default();

		for ring in 0..=rings {
			let down = f32::from(ring) / f32::from(rings);
			let angle = down * core::f32::consts::PI;

			for segment in 0..=segments {
				let around = f32::from(segment) / f32::from(segments);
				let turn = around * core::f32::consts::TAU;
				let way =
					Vec3::new(angle.sin() * turn.cos(), angle.cos(), angle.sin() * turn.sin());

				data.vertices
					.push(MeshVertex::new(way * 0.5, way, Vec2::new(around, down)));
			}
		}

		let stride = u32::from(segments) + 1;

		data.indices = (0..u32::from(rings))
			.flat_map(|ring| (0..u32::from(segments)).map(move |segment| ring * stride + segment))
			.flat_map(|top| [top, top + 1, top + stride, top + 1, top + stride + 1, top + stride])
			.collect();
		colby_core::abi::mesh::tangents(&mut data);

		data
	}

	/// Every `step`th triangle of a list: a coarser level that looks nothing
	/// like the mesh, so a frame that drew the wrong one cannot pass for the
	/// right one.
	fn every(indices: &[u32], step: usize) -> Vec<u32> {
		indices
			.chunks_exact(3)
			.step_by(step)
			.flatten()
			.copied()
			.collect()
	}

	/// The ball with two coarser levels: every other triangle, then every
	/// fourth.
	fn leveled() -> MeshData {
		let mut data = ball();

		data.levels = vec![
			Level {
				indices: every(&data.indices, 2),
				error: ERRORS[0],
			},
			Level {
				indices: every(&data.indices, 4),
				error: ERRORS[1],
			},
		];

		data
	}

	/// The same ball with one of its levels as the mesh itself and no levels.
	///
	/// @param level - nought for the ball, one or two for a level of it
	fn alone(level: usize) -> MeshData {
		let data = leveled();
		let indices = match level {
			| 0 => data.indices.clone(),
			| _ => data.levels[level - 1].indices.clone(),
		};

		MeshData { indices, levels: Vec::new(), ..data }
	}

	/// A floor, the sun, and one ball of `data` standing with its middle at a
	/// point, looked at from [`EYE`].
	fn world(data: MeshData, at: Vec3) -> (World, EntityId) {
		let mut world = World::new();

		world.camera.position = EYE;
		world.camera.target = Vec3::new(0.0, 1.0, 0.0);
		world.light = Vec3::new(-0.4, -1.0, -0.3).normalize();

		let floor = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, -0.5, 0.0),
			scale: Vec3::new(40.0, 1.0, 40.0),
			..Transform::IDENTITY
		});

		world
			.entities
			.set_renderable(floor, Renderable::new(MeshId::CUBE, Vec3::splat(0.4)));

		let mesh = world.meshes.insert("test/ball", data);
		let ball = world.entities.spawn_at(Transform::at(at));

		world
			.entities
			.set_renderable(ball, Renderable::new(mesh, Vec3::new(0.8, 0.5, 0.3)));

		(world, ball)
	}

	/// Sets one of the variables a frame reads.
	fn said(world: &mut World, name: &str, value: &str) {
		world.cvars.var(name, Value::Float(0.0), "");
		world.cvars.set(name, value);
	}

	/// How many samples a frame is drawn with, and whether the test for what is
	/// behind something nearer runs.
	fn asked(world: &mut World, samples: &str, covering: &str) {
		said(world, MSAA, samples);
		world
			.cvars
			.var(cover::ENABLED, Value::Bool(true), "");
		world.cvars.set(cover::ENABLED, covering);
	}

	/// Makes a ball glass: blended, so it is drawn in the second of the
	/// picture's lists, after everything solid.
	fn glassed(world: &mut World, ball: EntityId) {
		let glass = world.materials.insert("test/glass", Material {
			blend: Blend::Alpha,
			opacity: 0.4,
			..Material::DEFAULT
		});

		if let Some((mesh, tint)) = world
			.entities
			.renderable(ball)
			.map(|renderable| (renderable.mesh, renderable.color))
		{
			world
				.entities
				.set_renderable(ball, Renderable::of(mesh, glass, tint));
		}
	}

	/// A frame drawn, and its counts.
	fn shot(capture: &mut Capture, world: &mut World) -> (Image, Drawn) {
		let image = capture.shoot(world).expect("the capture renders");

		capture.scene_mut().settle();

		(image, capture.scene_mut().drawn())
	}

	/// How many pixels of two pictures differ at all.
	fn apart(one: &Image, other: &Image) -> usize {
		one.pixels
			.chunks_exact(4)
			.zip(other.pixels.chunks_exact(4))
			.filter(|(a, b)| a != b)
			.count()
	}

	/// How far from [`EYE`] the nearest point of a ball of radius a half
	/// standing at `(0, 1, z)` is, by hand: the eye is inside the box's reach
	/// across and up, so it is the depth to the near face.
	fn by_hand(z: f32) -> f32 { EYE.z - z - 0.5 }

	/// The level the arithmetic says at a distance, by hand.
	fn level_at(away: f32, height: f32) -> usize {
		let pixels = height / (2.0 * 0.5_f32.tan());

		ERRORS
			.iter()
			.take_while(|error| **error * pixels <= away)
			.count()
	}

	#[test]
	fn a_thing_far_enough_away_is_the_picture_of_its_level_drawn_as_a_mesh_of_its_own() {
		let (Some(mut capture), Some(mut other)) = (capture(), capture()) else {
			return;
		};
		let at = Vec3::new(0.0, 1.0, -4.0);

		assert_eq!(level_at(by_hand(at.z), 240.0), 2, "the fixture stands where both fit");

		for samples in ["1", "4"] {
			for covering in ["true", "false"] {
				let (mut leveled_world, _) = world(leveled(), at);
				let (mut alone_world, _) = world(alone(2), at);

				asked(&mut leveled_world, samples, covering);
				asked(&mut alone_world, samples, covering);

				let (drawn_at_level, drawn) = shot(&mut capture, &mut leveled_world);
				let (drawn_alone, alone_drawn) = shot(&mut other, &mut alone_world);

				assert_eq!(drawn.lowered, 1, "the ball is drawn coarser than itself");
				assert_eq!(
					drawn.triangles,
					1536 / 4 + 12,
					"at its second level, with the floor's twelve beside it"
				);
				assert_eq!(
					(alone_drawn.lowered, alone_drawn.triangles),
					(0, 1536 / 4 + 12),
					"and the same number of triangles drawn whole as a mesh of its own"
				);
				assert_eq!(
					apart(&drawn_at_level, &drawn_alone),
					0,
					"at {samples} samples with the test for what is behind something nearer \
					 {covering}, the picture - shadows and all - is the level's own"
				);
			}
		}
	}

	#[test]
	fn a_glass_thing_far_enough_away_is_drawn_and_counted_at_its_level() {
		let (Some(mut capture), Some(mut other)) = (capture(), capture()) else {
			return;
		};
		let at = Vec3::new(0.0, 1.0, -4.0);
		let (mut leveled_world, leveled_ball) = world(leveled(), at);
		let (mut alone_world, alone_ball) = world(alone(2), at);

		glassed(&mut leveled_world, leveled_ball);
		glassed(&mut alone_world, alone_ball);

		let (drawn_at_level, drawn) = shot(&mut capture, &mut leveled_world);
		let (drawn_alone, _) = shot(&mut other, &mut alone_world);

		assert_eq!(
			(drawn.lowered, drawn.triangles),
			(1, 1536 / 4 + 12),
			"the glass ball is drawn at its second level, and its triangles are counted there"
		);
		assert_eq!(apart(&drawn_at_level, &drawn_alone), 0, "and its picture is the level's own");
	}

	#[test]
	fn at_a_threshold_of_nought_a_mesh_with_levels_is_the_mesh_drawn_whole() {
		let (Some(mut capture), Some(mut other)) = (capture(), capture()) else {
			return;
		};
		let at = Vec3::new(0.0, 1.0, -4.0);
		let (mut leveled_world, _) = world(leveled(), at);
		let (mut whole_world, _) = world(alone(0), at);

		said(&mut leveled_world, THRESHOLD, "0");

		let (off, drawn) = shot(&mut capture, &mut leveled_world);
		let (whole, _) = shot(&mut other, &mut whole_world);

		assert_eq!((drawn.lowered, drawn.triangles), (0, 1536 + 12), "nothing drawn coarser");
		assert_eq!(apart(&off, &whole), 0, "and the picture is the mesh's, to the bit");
	}

	#[test]
	fn a_thing_behind_something_nearer_is_counted_at_its_level() {
		let Some(mut capture) = capture() else {
			return;
		};
		let (mut world, _) = world(leveled(), Vec3::new(0.0, 1.0, -4.0));
		let wall = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, 1.0, 0.0),
			scale: Vec3::new(4.0, 3.0, 0.2),
			..Transform::IDENTITY
		});

		world
			.entities
			.set_renderable(wall, Renderable::new(MeshId::CUBE, Vec3::splat(0.6)));

		let (_, drawn) = shot(&mut capture, &mut world);

		assert_eq!(
			(drawn.lowered, drawn.covered, drawn.covered_triangles),
			(1, 1, 1536 / 4),
			"the ball behind the wall is left out at its second level, and counted at it"
		);
	}

	#[test]
	fn what_the_pass_before_the_scene_wrote_is_the_level_the_picture_drew() {
		let (Some(mut capture), Some(mut other), Some(mut third)) =
			(capture(), capture(), capture())
		else {
			return;
		};
		let at = Vec3::new(0.0, 1.0, -4.0);

		// its normal and its roughness, drawn instead of the picture
		for view in ["1", "2"] {
			let (mut leveled_world, _) = world(leveled(), at);
			let (mut alone_world, _) = world(alone(2), at);

			said(&mut leveled_world, prepass::VIEW, view);
			said(&mut alone_world, prepass::VIEW, view);

			let (seen_at_level, _) = shot(&mut capture, &mut leveled_world);
			let (seen_alone, _) = shot(&mut other, &mut alone_world);
			let (mut whole_world, _) = world(alone(0), at);

			said(&mut whole_world, prepass::VIEW, view);

			let (seen_whole, _) = shot(&mut third, &mut whole_world);

			assert_eq!(apart(&seen_at_level, &seen_alone), 0, "view {view} is the level's");
			assert!(
				apart(&seen_at_level, &seen_whole) > 0,
				"and not the whole ball's, which the view can tell apart"
			);
		}
	}

	#[test]
	fn every_command_the_test_draws_through_starts_where_its_level_s_run_does() {
		let Some(mut capture) = capture() else {
			return;
		};
		let (mut world, _) = world(leveled(), Vec3::new(0.0, 1.0, -4.0));

		shot(&mut capture, &mut world);

		let commands = capture
			.scene_mut()
			.cover_command_values()
			.expect("the test ran");
		let ball: Vec<&[u32]> = commands
			.chunks_exact(5)
			.filter(|command| command[0] != 36)
			.collect();

		assert_eq!(
			ball,
			[&[1536 / 4 * 3, 1, 1536 * 3 + 1536 / 2 * 3, 0, 0][..]],
			"the ball's one command draws its second level's run, which starts after the ball's \
			 own indices and its first level's"
		);
	}

	#[test]
	fn things_at_one_level_of_one_mesh_are_one_batch_whatever_order_they_were_made_in() {
		let Some(mut capture) = capture() else {
			return;
		};
		// made far, near, far: sorted by the order they were made in alone, the
		// two far ones would be two batches with the near one between them. The
		// near one to the left and both far ones to the right, so nothing is
		// behind anything
		let (mut world, first) = world(leveled(), Vec3::new(1.0, 1.0, -4.0));
		let (mesh, tint) = world
			.entities
			.renderable(first)
			.map(|renderable| (renderable.mesh, renderable.color))
			.expect("the ball draws");

		for at in [Vec3::new(-1.0, 1.0, 3.7), Vec3::new(2.2, 1.0, -4.0)] {
			let ball = world.entities.spawn_at(Transform::at(at));

			world
				.entities
				.set_renderable(ball, Renderable::new(mesh, tint));
		}

		let (_, drawn) = shot(&mut capture, &mut world);

		// the near one's box is 1.87 away, where a level would be 2.2 pixels
		// off; the far ones' 9.5 and 9.7, where the second is 6.6
		assert_eq!(
			(drawn.lowered, drawn.triangles),
			(2, 1536 + 1536 / 4 * 2 + 12),
			"the near one whole and both far ones at their second level"
		);

		let commands = capture
			.scene_mut()
			.cover_command_values()
			.expect("the test ran");
		let balls: Vec<&[u32]> = commands
			.chunks_exact(5)
			.filter(|command| command[0] != 36)
			.collect();

		assert_eq!(
			balls,
			[
				&[1536 * 3, 1, 0, 0, 0][..],
				&[1536 / 4 * 3, 2, 1536 * 3 + 1536 / 2 * 3, 0, 0][..]
			],
			"one command for the whole ball and one for both far ones"
		);
	}

	#[test]
	fn a_picture_drawn_into_a_rectangle_is_asked_in_the_rectangle_s_pixels() {
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut tall = Capture::new(gpu, SIZE.0, SIZE.1 * 2).expect("the capture builds");
		let at = Vec3::new(0.0, 1.0, 2.5);
		let away = by_hand(at.z);

		assert_eq!(level_at(away, 480.0), 0, "the whole of the tall target is too tall to lower");
		assert_eq!(level_at(away, 240.0), 1, "while half of it is not");

		let (mut whole, _) = world(leveled(), at);

		tall.shoot(&mut whole)
			.expect("the capture renders");
		tall.scene_mut().settle();

		assert_eq!(tall.scene_mut().drawn().lowered, 0, "in the whole target, the ball itself");

		let mut halved = Capture::new(gpu, SIZE.0, SIZE.1 * 2).expect("the capture builds");
		let (mut within, _) = world(leveled(), at);
		let half = Viewport {
			x: 0,
			y: 120,
			width: SIZE.0,
			height: SIZE.1,
		};

		halved
			.shoot_within(&mut within, half)
			.expect("the capture renders");
		halved.scene_mut().settle();

		let drawn = halved.scene_mut().drawn();

		assert_eq!(
			(drawn.lowered, drawn.triangles),
			(1, 1536 / 2 + 12),
			"in a rectangle half as tall, its first level"
		);
	}

	#[test]
	fn a_thing_stretched_along_one_axis_is_asked_at_its_largest_size() {
		let Some(mut capture) = capture() else {
			return;
		};
		let at = Vec3::new(0.0, 1.0, -4.0);
		let (mut world, ball) = world(leveled(), at);

		// three times as tall and no deeper, so its near face stands where the
		// round ball's does and only its size differs: 9.5 away, its levels are
		// three times as far off and only the first fits
		if let Some(transform) = world.entities.transform_mut(ball) {
			transform.scale = Vec3::new(1.0, 3.0, 1.0);
		}

		world.entities.snap(ball);
		world.settle();

		assert_eq!(level_at(by_hand(at.z), 240.0), 2, "the round ball is drawn at its second");

		let (_, drawn) = shot(&mut capture, &mut world);

		assert_eq!(
			(drawn.lowered, drawn.triangles),
			(1, 1536 / 2 + 12),
			"and the tall one at its first"
		);
	}

	#[test]
	fn a_camera_looking_down_asks_with_its_lens_and_not_with_where_it_looks() {
		let Some(mut capture) = capture() else {
			return;
		};
		let (mut world, _) = world(leveled(), Vec3::new(0.0, 1.0, 0.0));

		// two thirds of the way to straight down, where the matrix with the view
		// in it counts under half as many pixels a unit up the picture as the
		// lens alone does
		world.camera.position = Vec3::new(0.0, 5.5, 2.0);
		world.camera.target = Vec3::new(0.0, 1.0, 0.0);

		// the nearest point of the ball's box is its top front edge
		let away = (world.camera.position - Vec3::new(0.0, 1.5, 0.5)).length();

		assert_eq!(level_at(away, 240.0), 1, "{away} away, its first level");

		let (_, drawn) = shot(&mut capture, &mut world);

		assert_eq!((drawn.lowered, drawn.triangles), (1, 1536 / 2 + 12), "and drawn at it");
	}

	#[test]
	fn a_thing_walking_away_is_drawn_coarser_exactly_where_the_arithmetic_says() {
		let Some(mut capture) = capture() else {
			return;
		};
		let (mut world, ball) = world(leveled(), Vec3::new(0.0, 1.0, 4.0));
		let triangles = [1536, 1536 / 2, 1536 / 4];
		let mut seen = [false; 3];

		for step in 0..=120_u16 {
			let z = f32::from(step).mul_add(-0.25, 4.5);

			if let Some(transform) = world.entities.transform_mut(ball) {
				transform.position.z = z;
			}

			world.entities.snap(ball);
			world.settle();

			let (_, drawn) = shot(&mut capture, &mut world);
			let expected = level_at(by_hand(z), 240.0);

			seen[expected] = true;

			assert_eq!(
				drawn.triangles,
				triangles[expected] + 12,
				"at z {z}, {} away, the level the arithmetic says is {expected}",
				by_hand(z)
			);
		}

		assert_eq!(seen, [true; 3], "and the walk crossed from the ball to both of its levels");
	}
}
