//! The camera the renderer looks through.
//!
//! A look-at camera: a position, a point of interest, and an up vector. That is
//! what an orbit control wants, and an orbit control is what there is. A camera
//! carrying its own rotation is the better shape for a first-person controller
//! and can be added when one exists - the renderer only ever asks for a matrix.

use crate::glam::{
	Mat4, Vec2, Vec3,
	camera::rh::{proj::directx::perspective, view::look_at_mat4},
};

/// The narrowest and widest vertical field of view that is accepted.
///
/// Not taste: outside this range the projection matrix stops being useful and
/// starts being a way to divide by something very close to zero.
pub const FOV_RANGE: (f32, f32) = (0.1, 3.0);

/// Where the renderer looks from, and at what.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
	/// Where the camera is, in world space.
	pub position: Vec3,

	/// What it is pointed at.
	pub target: Vec3,

	/// Which way is up. Must not be parallel to the line of sight.
	pub up: Vec3,

	/// Vertical field of view, in radians. @ref [`FOV_RANGE`].
	pub fov_y: f32,

	/// Nothing closer than this is drawn.
	pub near: f32,

	/// Nothing further than this is drawn.
	pub far: f32,
}

impl Camera {
	/// Three units back from the origin, looking at it.
	pub const DEFAULT: Self = Self {
		position: Vec3::new(0.0, 2.0, 5.0),
		target: Vec3::ZERO,
		up: Vec3::Y,
		fov_y: 1.0,
		near: 0.1,
		far: 200.0,
	};

	/// The view matrix: world space into camera space.
	#[must_use]
	pub fn view(&self) -> Mat4 { look_at_mat4(self.position, self.target, self.up) }

	/// The projection matrix: camera space into clip space.
	///
	/// Right-handed, with a zero-to-one depth range and y up in clip space.
	///
	/// @note: glam calls that combination `directx`, and it is the one wgpu
	/// uses. Its `vulkan` sibling has the same depth range but flips y, which
	/// would put the picture upside down.
	///
	/// @param aspect - the surface's width divided by its height
	#[must_use]
	pub fn projection(&self, aspect: f32) -> Mat4 {
		perspective(
			self.fov_y.clamp(FOV_RANGE.0, FOV_RANGE.1),
			aspect.max(0.001),
			self.near.max(0.001),
			self.far.max(self.near + 0.001),
		)
	}

	/// Both of the above, in the order the shader multiplies them.
	#[must_use]
	pub fn view_projection(&self, aspect: f32) -> Mat4 { self.projection(aspect) * self.view() }

	/// Which way the world lies through a point on the near plane.
	///
	/// Built from the camera's own basis rather than by inverting the
	/// view-projection: the basis is three cross products and is exact, and an
	/// inverse of a matrix that already clamped its own inputs is neither.
	///
	/// @param ndc - normalized device coordinates, `-1 ..= 1` on both axes,
	/// with x rightwards and **y upwards** - the sign a screen pixel has to be
	/// flipped into
	/// @param aspect - the surface's width divided by its height
	/// @return a unit vector in world space
	#[must_use]
	pub fn direction(&self, ndc: Vec2, aspect: f32) -> Vec3 {
		let forward = (self.target - self.position).normalize_or(Vec3::NEG_Z);
		let right = forward.cross(self.up).normalize_or(Vec3::X);
		let up = right.cross(forward);

		let half_height = (self.fov_y.clamp(FOV_RANGE.0, FOV_RANGE.1) * 0.5).tan();
		let half_width = half_height * aspect.max(0.001);

		(forward + right * (ndc.x * half_width) + up * (ndc.y * half_height))
			.normalize_or(forward)
	}

	/// The same, from a pixel in a window rather than from clip space.
	///
	/// @param pixel - where in the surface, measured from the **top** left, as
	/// every pointer on this platform is
	/// @param viewport - how wide and how tall that surface is
	/// @return a unit vector in world space
	#[must_use]
	pub fn pixel_direction(&self, pixel: Vec2, viewport: Vec2) -> Vec3 {
		let size = viewport.max(Vec2::ONE);
		let ndc = Vec2::new(
			(pixel.x / size.x).mul_add(2.0, -1.0),
			(pixel.y / size.y).mul_add(-2.0, 1.0),
		);

		self.direction(ndc, size.x / size.y)
	}

	/// Places the camera on a sphere around its target.
	///
	/// The usual orbit control: an angle around the up axis, an angle above the
	/// horizon, and a distance. Pitch is held just short of the poles, where
	/// the up vector and the line of sight line up and the view matrix stops
	/// being defined.
	///
	/// @param yaw - radians around the up axis
	/// @param pitch - radians above the horizon
	/// @param distance - how far from the target to sit
	pub fn orbit(&mut self, yaw: f32, pitch: f32, distance: f32) {
		let pitch = pitch.clamp(-1.55, 1.55);
		let flat = pitch.cos() * distance.max(0.01);

		self.position = self.target
			+ Vec3::new(flat * yaw.sin(), pitch.sin() * distance.max(0.01), flat * yaw.cos());
	}

	/// This camera part of the way towards another one.
	///
	/// Position, target and field of view blend. `up`, `near` and `far` are
	/// taken from the far end rather than blended: two `up` vectors that are
	/// not parallel pass through zero somewhere between them, and
	/// `look_at_mat4` of a zero up vector is not a matrix. None of the three
	/// is a thing that changes often enough to be worth that.
	///
	/// @note: an orbiting camera interpolated this way travels the chord
	/// rather than the arc, so its distance from the target dips by
	/// `d * (1 - cos(half the step's turn))` mid-step. Millimeters at a
	/// normal drag rate. The host cannot do better, because the yaw and pitch
	/// that produced the pose are the game's and it never sees them - which
	/// is the price of the game not knowing this happens at all.
	///
	/// @param other - the camera at the far end
	/// @param t - zero for this one, one for the other
	#[must_use]
	pub fn lerp(self, other: Self, t: f32) -> Self {
		if t <= 0.0 || self == other {
			return self;
		}

		if t >= 1.0 {
			return other;
		}

		Self {
			position: self.position.lerp(other.position, t),
			target: self.target.lerp(other.target, t),
			fov_y: (other.fov_y - self.fov_y).mul_add(t, self.fov_y),
			..other
		}
	}
}

impl Default for Camera {
	fn default() -> Self { Self::DEFAULT }
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_projection_puts_the_near_plane_at_zero_and_the_far_plane_at_one() {
		let camera = Camera::DEFAULT;
		let projection = camera.projection(16.0 / 9.0);

		let near = projection.project_point3(Vec3::new(0.0, 0.0, -camera.near));
		let far = projection.project_point3(Vec3::new(0.0, 0.0, -camera.far));

		assert!(near.z.abs() < 1.0e-3, "wgpu's depth range starts at zero, got {}", near.z);
		assert!((far.z - 1.0).abs() < 1.0e-3, "and ends at one, got {}", far.z);
	}

	#[test]
	fn the_view_puts_the_target_straight_ahead() {
		let camera = Camera::DEFAULT;
		let seen = camera.view().transform_point3(camera.target);

		assert!(
			seen.x.abs() < 1.0e-5 && seen.y.abs() < 1.0e-5,
			"the target lands on the center of the screen, got {seen}"
		);
		assert!(seen.z < 0.0, "and in front of the camera, which is -z in a right-handed view");
	}

	#[test]
	fn orbiting_keeps_the_requested_distance() {
		let mut camera = Camera::DEFAULT;
		camera.target = Vec3::new(1.0, 0.0, -2.0);
		camera.orbit(0.7, 0.3, 4.0);

		assert!(
			(camera.position.distance(camera.target) - 4.0).abs() < 1.0e-4,
			"orbit moves the camera, not the target"
		);
	}

	#[test]
	fn a_camera_halfway_between_two_steps_looks_from_between_them() {
		let mut was = Camera::DEFAULT;
		was.position = Vec3::new(0.0, 0.0, 10.0);
		was.target = Vec3::ZERO;
		was.fov_y = 0.8;

		// every part of a camera that a step can change, changed at once: where
		// it is, what it is looking at, and how wide.
		let mut is = was;
		is.position = Vec3::new(0.0, 0.0, 20.0);
		is.target = Vec3::new(4.0, 0.0, 0.0);
		is.fov_y = 1.0;

		let seen = was.lerp(is, 0.5);

		assert!(
			seen.position
				.abs_diff_eq(Vec3::new(0.0, 0.0, 15.0), 1.0e-5),
			"halfway through the step is halfway along the move, got {}",
			seen.position
		);
		assert!(
			seen.target.abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1.0e-5),
			"and halfway along the pan: a camera whose target jumps at the step rate judders 			 exactly as badly as one whose position does, got {}",
			seen.target
		);
		assert!(
			(seen.fov_y - 0.9).abs() < 1.0e-5,
			"and halfway through the zoom, got {}",
			seen.fov_y
		);
	}

	#[test]
	fn a_camera_at_the_end_of_a_step_is_the_one_the_game_wrote() {
		let was = Camera::DEFAULT;
		let mut is = was;
		is.position = Vec3::new(3.0, 1.0, 7.0);
		is.fov_y = 0.7;

		let seen = was.lerp(is, 1.0);

		assert_eq!(
			seen, is,
			"the far end is exact, not close: a render at one has to match a render with no \
			 interpolation at all"
		);
		assert_eq!(was.lerp(is, 0.0), was, "and so is the near end");
	}

	#[test]
	fn interpolating_a_camera_leaves_its_up_vector_alone() {
		let was = Camera::DEFAULT;
		let mut is = was;
		is.up = -Vec3::Y;

		let seen = was.lerp(is, 0.5);

		assert_eq!(
			seen.up, is.up,
			"lerping between opposite up vectors would pass through zero, and a zero up vector \
			 has no view matrix"
		);
	}

	#[test]
	fn orbiting_straight_up_does_not_reach_the_pole() {
		let mut camera = Camera::DEFAULT;
		camera.orbit(0.0, 10.0, 3.0);

		let up_component = (camera.position - camera.target)
			.normalize()
			.dot(Vec3::Y);

		assert!(
			up_component < 0.9999,
			"pitch is held short of straight up, where the view matrix falls apart"
		);
	}

	#[test]
	fn the_middle_of_the_screen_looks_at_what_the_camera_is_pointed_at() {
		let camera = Camera::DEFAULT;
		let ahead = (camera.target - camera.position).normalize();

		assert!(
			camera
				.direction(Vec2::ZERO, 1.6)
				.abs_diff_eq(ahead, 1.0e-5),
			"the center of clip space is the line of sight, whatever the aspect"
		);
	}

	#[test]
	fn the_edge_of_the_screen_is_half_the_field_of_view_away() {
		let mut camera = Camera::DEFAULT;
		camera.fov_y = 1.0;
		let ahead = (camera.target - camera.position).normalize();
		let edge = camera.direction(Vec2::new(0.0, 1.0), 1.0);
		let angle = ahead.dot(edge).clamp(-1.0, 1.0).acos();

		assert!(
			(angle - 0.5).abs() < 1.0e-4,
			"the top of the picture is fov_y / 2 above the middle, got {angle}"
		);
	}

	#[test]
	fn a_pixel_is_measured_from_the_top_and_clip_space_from_the_bottom() {
		let camera = Camera::DEFAULT;
		let viewport = Vec2::new(1280.0, 720.0);

		let top = camera.pixel_direction(Vec2::new(640.0, 0.0), viewport);
		let bottom = camera.pixel_direction(Vec2::new(640.0, 720.0), viewport);

		assert!(
			top.y > bottom.y,
			"pixel row zero is the top of the window and must look upwards: got {top} against 			 {bottom}"
		);
		assert!(
			camera
				.pixel_direction(viewport * 0.5, viewport)
				.abs_diff_eq(camera.direction(Vec2::ZERO, 16.0 / 9.0), 1.0e-5),
			"and the middle pixel is the middle of clip space"
		);
	}

	#[test]
	fn a_direction_survives_a_viewport_that_has_not_been_set_yet() {
		let camera = Camera::DEFAULT;
		let direction = camera.pixel_direction(Vec2::ZERO, Vec2::ZERO);

		assert!(
			direction.is_finite() && direction.length() > 0.5,
			"a zero viewport divides by nothing and still answers, got {direction}"
		);
	}
}
