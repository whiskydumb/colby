//! The sun and the lamps at one point, with a ray towards each for its shadow.
//!
//! **The shader's own arithmetic, written again.** A lamp falls off by an
//! inverse square with a window closing at its range, a cone by the square of
//! a line in the cosine off its axis, and a surface takes the cosine of the
//! angle the light arrives at - the same three factors, in the same form, that
//! light a surface in a frame, so that what a lit wall throws onto a floor in
//! a bake is what that wall looks like in the picture. Only the diffuse half is
//! worked out: a bounce is what a surface sends back in every direction, which
//! is its diffuse light, and a gleam is the renderer's to draw each frame.
//!
//! **A shadow is a ray, not a map.** The frame looks a lamp's shadow up in a
//! map of a few hundred texels; here a ray is traced to the light, which is
//! the answer the map approximates. A lamp told to throw no shadow throws none
//! here either, since what it lights each frame is lit through walls.
//!
//! The unit is the ambient color's, @ref the crate: a surface of diffuse color
//! `albedo` sends back `albedo` times what this answers.

use colby_core::glam::Vec3;

use crate::{
	scene::{BIAS, Lamp, Scene},
	tree::Ray,
};

impl Scene {
	/// What the sun and the lamps shine onto one point of a surface.
	///
	/// @param at - the point
	/// @param normal - which way the surface faces there, of unit length
	/// @return the light arriving, in the ambient color's unit
	#[must_use]
	pub fn direct(&self, at: Vec3, normal: Vec3) -> Vec3 {
		let start = at + normal * BIAS;
		let mut total = Vec3::ZERO;

		if let Some(sun) = self.sun() {
			let facing = normal.dot(sun);

			if facing > 0.0
				&& !self
					.tree()
					.blocked(&Ray::new(start, sun), f32::INFINITY)
			{
				total += Vec3::splat(facing);
			}
		}

		for lamp in self.lamps() {
			total += self.lamp_at(lamp, at, start, normal);
		}

		total
	}

	/// What one lamp shines onto one point.
	///
	/// @param lamp - the lamp
	/// @param at - the point
	/// @param start - where its shadow ray starts, just off the surface
	/// @param normal - which way the surface faces there
	fn lamp_at(&self, lamp: &Lamp, at: Vec3, start: Vec3, normal: Vec3) -> Vec3 {
		let towards = lamp.position - at;
		let distance_square = towards.length_squared();
		let range_square = lamp.range * lamp.range;

		// at or past the reach, or a distance that is not a number
		if distance_square.is_nan() || distance_square >= range_square {
			return Vec3::ZERO;
		}

		let way = towards * distance_square.max(1.0e-8).sqrt().recip();
		let off_axis = (-lamp.direction).dot(way) * lamp.scale;
		let cone = (off_axis + lamp.offset).clamp(0.0, 1.0);
		let facing = normal.dot(way).clamp(0.0, 1.0);

		if !(cone > 0.0 && facing > 0.0) {
			return Vec3::ZERO;
		}

		if lamp.shadow {
			let gap = lamp.position - start;
			let reach = gap.length();

			if self
				.tree()
				.blocked(&Ray::new(start, gap / reach.max(f32::MIN_POSITIVE)), reach)
			{
				return Vec3::ZERO;
			}
		}

		lamp.color * (falloff(distance_square, range_square) * cone * cone * facing)
	}
}

/// How much of a lamp survives the distance to a point.
///
/// The shader's `lamp_falloff`: an inverse square with a window closed
/// smoothly at the range, so a lamp ends where its reach says rather than
/// lighting the whole world by a millionth.
///
/// @param distance_square - the square of the distance to the lamp
/// @param range_square - the square of its range
#[must_use]
pub fn falloff(distance_square: f32, range_square: f32) -> f32 {
	let factor = distance_square / range_square.max(1.0e-4);
	let squared = factor * factor;
	let smoothed = (1.0 - squared).clamp(0.0, 1.0);

	smoothed * smoothed / distance_square.max(1.0e-4)
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			Body, BodyKind, EntityId, MeshId, Renderable, Shape, Transform, World, light::Light,
			material::MaterialId,
		},
		glam::Quat,
	};

	use super::*;

	/// A world with nothing in it and no sun.
	fn dark() -> World {
		let mut world = World::new();

		world.light = Vec3::ZERO;

		world
	}

	/// A floor forty units across at height nought.
	fn floor(world: &mut World) -> EntityId {
		let floor = world.entities.spawn_at(Transform {
			scale: Vec3::new(40.0, 1.0, 40.0),
			..Transform::IDENTITY
		});

		world
			.entities
			.set_renderable(floor, Renderable::of(MeshId::QUAD, MaterialId::DEFAULT, Vec3::ONE));

		floor
	}

	/// A lamp standing somewhere, pointing down.
	fn lamp(world: &mut World, at: Vec3, light: Light) -> EntityId {
		let lamp = world.entities.spawn_at(Transform {
			position: at,
			// -z turned to point straight down: a quarter turn about x, exactly
			rotation: Quat::from_xyzw(
				-std::f32::consts::FRAC_1_SQRT_2,
				0.0,
				0.0,
				std::f32::consts::FRAC_1_SQRT_2,
			),
			..Transform::IDENTITY
		});

		world.entities.set_light(lamp, light);

		lamp
	}

	/// The closed form for a point lamp over a floor: the intensity, the cosine
	/// of the angle it arrives at, and the falloff, in double precision.
	fn expected(height: f64, across: f64, intensity: f64, range: f64) -> f64 {
		let (up, along) = (height * height, across * across);
		let distance_square = up + along;
		let cosine = height / distance_square.sqrt();
		let factor = distance_square / (range * range);
		let squared = factor * factor;
		let window = (1.0 - squared).clamp(0.0, 1.0);

		intensity * cosine * window * window / distance_square
	}

	#[test]
	fn a_floor_under_a_point_lamp_is_lit_as_the_inverse_square_says() {
		let mut world = dark();
		let (height, intensity, range) = (2.0_f32, 3.0_f32, 12.0_f32);

		floor(&mut world);
		lamp(
			&mut world,
			Vec3::new(0.0, height, 0.0),
			Light::point(Vec3::ONE, intensity, range),
		);

		let scene = Scene::of(&world);

		for across in [0.0_f32, 0.5, 1.0, 2.0, 3.5, 6.0, 9.0, 11.5] {
			let lit = scene.direct(Vec3::new(across, 0.0, 0.0), Vec3::Y);
			let known = expected(
				f64::from(height),
				f64::from(across),
				f64::from(intensity),
				f64::from(range),
			);

			// two parts in a million, and a floor under it for the far edge where
			// the window has closed the light almost to nothing
			let allowed = known * 2.0e-6;

			assert!(
				(f64::from(lit.x) - known).abs() <= allowed + 1.0e-9,
				"{across} along: {} against {known}",
				lit.x
			);
			assert_eq!(lit.x.to_bits(), lit.y.to_bits(), "a white lamp is white");
		}

		assert_eq!(
			scene.direct(Vec3::new(12.5, 0.0, 0.0), Vec3::Y),
			Vec3::ZERO,
			"and past its range it lights nothing"
		);
	}

	#[test]
	fn a_cone_lights_its_middle_whole_its_edge_part_way_and_outside_nothing() {
		let mut world = dark();
		let (inner, outer) = (0.3_f32, 0.6_f32);

		floor(&mut world);
		lamp(
			&mut world,
			Vec3::new(0.0, 2.0, 0.0),
			Light::spot(Vec3::ONE, 1.0, 50.0, inner, outer),
		);

		let scene = Scene::of(&world);
		let at_angle = |angle: f64| {
			let across = 2.0 * angle.tan();
			let point = Vec3::new(f32_of(across), 0.0, 0.0);

			f64::from(scene.direct(point, Vec3::Y).x) / expected(2.0, across, 1.0, 50.0)
		};

		assert!(
			(at_angle(0.1) - 1.0).abs() < 1.0e-5,
			"inside the bright middle: {}",
			at_angle(0.1)
		);
		assert!(at_angle(0.62).abs() < 1.0e-9, "outside the edge: {}", at_angle(0.62));

		let between = 0.45_f64;
		let line = (between.cos() - f64::from(outer).cos())
			/ (f64::from(inner).cos() - f64::from(outer).cos());

		let squared = line * line;

		assert!(
			(at_angle(between) - squared).abs() < 1.0e-4,
			"between, the square of the line: {} against {}",
			at_angle(between),
			squared
		);
	}

	#[test]
	fn something_between_a_lamp_and_a_point_shadows_it_unless_the_lamp_is_told_not_to() {
		let mut world = dark();

		floor(&mut world);
		let lit = lamp(&mut world, Vec3::new(0.0, 3.0, 0.0), Light::point(Vec3::ONE, 1.0, 20.0));
		let block = world
			.entities
			.spawn_at(Transform::at(Vec3::new(0.0, 1.5, 0.0)));

		world
			.entities
			.set_renderable(block, Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, Vec3::ONE));

		let shaded = Scene::of(&world).direct(Vec3::ZERO, Vec3::Y);
		let beside = Scene::of(&world).direct(Vec3::new(3.0, 0.0, 0.0), Vec3::Y);

		assert_eq!(shaded, Vec3::ZERO, "under the block, in its shadow");
		assert!(beside.x > 0.0, "and lit beside it");

		if let Some(light) = world.entities.light_mut(lit) {
			light.shadow = false;
		}

		assert!(
			Scene::of(&world).direct(Vec3::ZERO, Vec3::Y).x > 0.0,
			"a lamp that throws no shadow lights through it"
		);
	}

	#[test]
	fn the_sun_lights_what_faces_it_and_not_what_a_roof_covers() {
		let mut world = dark();

		world.light = Vec3::new(0.0, -2.0, 0.0);
		floor(&mut world);

		let roof = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, 4.0, 0.0),
			scale: Vec3::new(4.0, 0.5, 4.0),
			..Transform::IDENTITY
		});

		world
			.entities
			.set_renderable(roof, Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, Vec3::ONE));

		let scene = Scene::of(&world);

		assert_eq!(
			scene.direct(Vec3::new(10.0, 0.0, 0.0), Vec3::Y),
			Vec3::ONE,
			"the open floor, head on"
		);
		assert_eq!(scene.direct(Vec3::ZERO, Vec3::Y), Vec3::ZERO, "under the roof");
		assert_eq!(
			scene.direct(Vec3::new(10.0, 0.0, 0.0), Vec3::NEG_Y),
			Vec3::ZERO,
			"facing away"
		);
		assert!(
			(scene
				.direct(Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.6, 0.8, 0.0))
				.x - 0.8)
				.abs() < 1.0e-6,
			"and a slope takes the cosine"
		);
	}

	#[test]
	fn a_surface_facing_away_from_a_lamp_takes_nothing_from_it() {
		let mut world = dark();

		lamp(&mut world, Vec3::new(0.0, 2.0, 0.0), Light::point(Vec3::ONE, 1.0, 20.0));

		let scene = Scene::of(&world);

		assert!(scene.direct(Vec3::ZERO, Vec3::Y).x > 0.0, "facing it, lit");
		assert_eq!(scene.direct(Vec3::ZERO, Vec3::NEG_Y), Vec3::ZERO, "facing away, nothing");
		assert_eq!(scene.direct(Vec3::ZERO, Vec3::X), Vec3::ZERO, "edge on, nothing");
	}

	#[test]
	fn a_ceiling_over_a_lamp_does_not_shadow_what_it_lights_below() {
		let mut world = dark();
		let ceiling = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, 4.0, 0.0),
			// a half turn about x, exactly: the quad faces down
			rotation: Quat::from_xyzw(1.0, 0.0, 0.0, 0.0),
			scale: Vec3::new(40.0, 1.0, 40.0),
		});

		world.entities.set_renderable(
			ceiling,
			Renderable::of(MeshId::QUAD, MaterialId::DEFAULT, Vec3::ONE),
		);
		floor(&mut world);
		lamp(&mut world, Vec3::new(0.0, 2.0, 0.0), Light::point(Vec3::ONE, 1.0, 20.0));

		let lit = Scene::of(&world).direct(Vec3::new(1.0, 0.0, 0.0), Vec3::Y);

		assert!(lit.x > 0.0, "a shadow ray stops at the lamp: {}", lit.x);
	}

	#[test]
	fn a_slope_the_sun_shines_on_does_not_shadow_itself() {
		let mut world = dark();
		let slope = world.entities.spawn_at(Transform {
			rotation: Quat::from_rotation_z(0.4),
			scale: Vec3::new(30.0, 1.0, 30.0),
			..Transform::IDENTITY
		});

		world.light = Vec3::new(0.2, -1.0, 0.3);
		world
			.entities
			.set_renderable(slope, Renderable::of(MeshId::QUAD, MaterialId::DEFAULT, Vec3::ONE));

		let scene = Scene::of(&world);
		let sun = scene.sun().expect("a sun");

		// points worked out on the slope the way a bake works them out, so
		// each carries the rounding a real one does
		for step in 1..40_u16 {
			let weight = f32::from(step) / 41.0;
			let (along, less) = (weight * 0.5, weight * 0.4);
			let surface = scene
				.surface_at(u32::from(step % 2), along, 0.5 - less)
				.expect("a point on the slope");
			let lit = scene.direct(surface.at, surface.normal);

			assert_eq!(lit, Vec3::splat(surface.normal.dot(sun)), "at {}", surface.at);
		}
	}

	#[test]
	fn a_lamp_a_moving_body_carries_or_that_is_hidden_is_not_baked() {
		let mut world = dark();
		let carried =
			lamp(&mut world, Vec3::new(0.0, 3.0, 0.0), Light::point(Vec3::ONE, 1.0, 20.0));

		lamp(&mut world, Vec3::new(5.0, 3.0, 0.0), Light::point(Vec3::ONE, 1.0, 20.0));

		let hidden =
			lamp(&mut world, Vec3::new(9.0, 3.0, 0.0), Light::point(Vec3::ONE, 1.0, 20.0));

		world.entities.set_hidden(hidden, true);
		world.bodies.spawn(
			Body::new(BodyKind::Dynamic, Shape::ball(0.2), Transform::IDENTITY).driving(carried),
		);

		let scene = Scene::of(&world);

		assert_eq!(scene.lamps().len(), 1, "the still, shown one only");
		assert_eq!(
			scene.lamps()[0].position,
			Vec3::new(5.0, 3.0, 0.0),
			"and it is the still one"
		);
	}

	#[test]
	fn the_falloff_is_the_inverse_square_near_the_lamp_and_nothing_at_its_range() {
		assert!(
			(falloff(1.0, 1.0e6) - 1.0).abs() < 1.0e-6,
			"one at one unit, far inside the range"
		);
		assert!((falloff(4.0, 1.0e6) - 0.25).abs() < 1.0e-6, "a quarter at two");
		assert_eq!(falloff(100.0, 100.0).to_bits(), 0.0_f32.to_bits(), "nothing at the range");
		assert_eq!(
			falloff(0.0, 100.0).to_bits(),
			1.0e4_f32.to_bits(),
			"and a point on the lamp is held, not infinite"
		);
	}

	/// A double narrowed to a float, for placing test points.
	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		reason = "a test point worked out in double precision"
	)]
	fn f32_of(value: f64) -> f32 { value as f32 }
}
