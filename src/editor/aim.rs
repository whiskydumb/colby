//! Where the editor looks from, and what is under the pointer when it clicks.
//!
//! Both halves are arithmetic over a [`World`] and a [`Camera`] with no egui in
//! them, which is what makes them the tested half of a viewport that is
//! otherwise checked by looking at it.
//!
//! **Picking is a ray against what is drawn, not against what collides.** A
//! person clicks on a thing they can see, and in this engine those are two
//! different tables: a body may have no entity, an entity may have no body, and
//! the demo has one of each. So the ray is tested against every entity's mesh
//! bounds, in that entity's own space, and the nearest hit wins. What that
//! costs is a body nobody draws - a bare collider, a sensor volume - being
//! unclickable and reachable only from the tree, which is named in the gaps.
//!
//! **Bounds rather than triangles**, deliberately: it is exact enough to pick
//! the right thing out of a scene and it is a dozen lines. `World::trace_ray`
//! is the exact one and it answers about bodies, which is the question this is
//! not asking. Refining a bounds hit against the triangles behind it is a
//! later change that nothing above this function would notice.
//!
//! **The camera is kept as an orbit** - a target, two angles and a distance -
//! rather than as the position and target a [`Camera`] holds. Those two are
//! what the renderer wants and they are a lossy place to keep a control
//! scheme: dragging sideways is a change in one angle, and recovering that
//! angle from a position every frame accumulates error. So the orbit is the
//! truth here and the camera is written from it.

use colby_core::{
	abi::{Camera, Transform, World},
	glam::{Vec2, Vec3},
};

use crate::select::Pick;

/// How far the camera may sit from what it is looking at.
const RANGE: (f32, f32) = (0.05, 5000.0);

/// How far a drag of one point turns the camera, in radians.
const TURN: f32 = 0.008;

/// How far one notch of the wheel moves the camera, as a share of its distance.
///
/// A share rather than a length, so that a wheel notch covers the same part of
/// what is on screen however far out the view is.
const WHEEL: f32 = 0.12;

/// How far a drag of one point slides the target, per unit of distance.
const SLIDE: f32 = 0.0016;

/// Where the editor is looking from, as an orbit around a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Fly {
	/// What it is looking at.
	target: Vec3,

	/// Radians around the up axis.
	yaw: f32,

	/// Radians above the horizon.
	pitch: f32,

	/// How far from the target it sits.
	distance: f32,
}

impl Fly {
	/// Reads an orbit out of a camera that is already somewhere.
	///
	/// What this is for is the moment edit mode starts: the editor takes over
	/// a camera the game was driving, and it should carry on from where the
	/// person was already looking rather than cutting to somewhere else.
	///
	/// @param camera - the camera to read
	pub(crate) fn read(camera: &Camera) -> Self {
		let offset = camera.position - camera.target;
		let distance = offset.length().clamp(RANGE.0, RANGE.1);

		Self {
			target: camera.target,
			// the inverse of what `Camera::orbit` builds: x and z carry the
			// yaw with the pitch's cosine in both, so the arc tangent of the
			// pair is the yaw whatever the pitch was.
			yaw: offset.x.atan2(offset.z),
			pitch: (offset.y / distance).clamp(-1.0, 1.0).asin(),
			distance,
		}
	}

	/// Writes the orbit into a camera.
	///
	/// @param camera - the camera to move
	pub(crate) fn write(self, camera: &mut Camera) {
		camera.target = self.target;
		camera.orbit(self.yaw, self.pitch, self.distance);
	}

	/// Turns around the target.
	///
	/// @param drag - how far the pointer moved, in points
	pub(crate) fn turn(&mut self, drag: Vec2) {
		self.yaw = drag.x.mul_add(-TURN, self.yaw);
		// held just short of the poles, where the up vector and the line of
		// sight line up and there is no view matrix. `Camera::orbit` clamps
		// as well; this one is here so that the *stored* angle does not run
		// away while the clamped one stands still.
		self.pitch = drag
			.y
			.mul_add(TURN, self.pitch)
			.clamp(-1.55, 1.55);
	}

	/// Moves closer to the target, or further from it.
	///
	/// @param notches - wheel lines, positive towards the target
	pub(crate) fn dolly(&mut self, notches: f32) {
		// multiplied rather than added, so that every notch covers the same
		// share of the way in and the last one never arrives.
		self.distance =
			(self.distance * WHEEL.mul_add(-notches, 1.0).max(0.05)).clamp(RANGE.0, RANGE.1);
	}

	/// Slides the target across the view, taking the camera with it.
	///
	/// Scaled by the distance, so that a drag covers the same part of what is
	/// on screen whether the view is close in or far out.
	///
	/// @param drag - how far the pointer moved, in points
	pub(crate) fn slide(&mut self, drag: Vec2) {
		let (right, up) = self.across();
		let far = self.distance * SLIDE;

		self.target += right * (-drag.x * far) + up * (drag.y * far);
	}

	/// The two axes of the screen, in world space.
	fn across(self) -> (Vec3, Vec3) {
		let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
		let (sin_pitch, cos_pitch) = self.pitch.sin_cos();

		// straight out of `Camera::orbit`: the offset from the target to the
		// camera, so the way the camera faces is the other way.
		let forward = -Vec3::new(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw);
		let right = Vec3::new(cos_yaw, 0.0, -sin_yaw);

		(right, right.cross(forward))
	}
}

impl Default for Fly {
	fn default() -> Self { Self::read(&Camera::DEFAULT) }
}

/// The editor's camera across frames: the orbit, and the pose it last wrote.
///
/// The second half is what settles the argument over who owns the camera while
/// the world is being edited. The orbit is the editor's, but it is not the only
/// thing that ever writes `World::camera` - a scene loaded from the console
/// carries one, and so does the world that comes back when play stops. So the
/// pose this last wrote is kept and compared: if the world's camera is not that
/// pose any more, somebody else moved it and the orbit is read out of the new
/// one. **Whoever said something last wins, and standing still says nothing**,
/// which is the rule the material table, the console table and the host's
/// gravity all already follow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct View {
	/// Where the editor is looking from.
	orbit: Fly,

	/// The camera this last put in the world.
	wrote: Camera,
}

impl View {
	/// The view to use this frame.
	///
	/// @param held - what the last frame ended with, if anything
	/// @param camera - the world's camera as it now stands
	pub(crate) fn taken(held: Option<Self>, camera: &Camera) -> Self {
		match held {
			| Some(view) if view.wrote == *camera => view,
			| _ => Self { orbit: Fly::read(camera), wrote: *camera },
		}
	}

	/// The orbit, to drive.
	pub(crate) const fn orbit(&mut self) -> &mut Fly { &mut self.orbit }

	/// Puts the orbit in the world, and remembers exactly what was put there.
	///
	/// Remembered *after* the write rather than before it, because reading an
	/// orbit out of a camera and writing it back is not quite the identity in
	/// floating point - and a comparison against the pose that was meant
	/// rather than the pose that landed would report somebody else's edit
	/// every single frame.
	///
	/// @param camera - the world's camera
	pub(crate) fn put(&mut self, camera: &mut Camera) {
		self.orbit.write(camera);
		self.wrote = *camera;
	}
}

/// The nearest entity a ray goes through.
///
/// Every entity that draws something is tested, against its mesh's bounds in
/// its own space rather than in the world's - which is what makes a rotated
/// box pick as the box it looks like instead of as the larger box around it.
/// An entity drawing nothing is not tested at all: there is nothing on screen
/// to have been clicked.
///
/// @param world - what to look through
/// @param from - where the ray starts, in world space
/// @param along - which way it goes; need not be a unit vector
/// @return what it hit, or [`Pick::Nothing`]
pub(crate) fn under(world: &World, from: Vec3, along: Vec3) -> Pick {
	let mut nearest = f32::INFINITY;
	let mut found = Pick::Nothing;

	for (id, transform, renderable) in world.entities.iter() {
		let Some(mesh) = world.meshes.get(renderable.mesh) else {
			continue;
		};

		// a mesh with nothing in it reports a point at the origin, and a point
		// is not a thing a ray can be said to go through. **This is also what
		// skips an entity drawing nothing**, and it is the only thing that
		// needs to: the null mesh handle resolves to slot zero of the
		// registry, which holds exactly such an empty mesh. A second guard on
		// the handle was written first and taken back out - no mutation of it
		// could fail a test, because this one already covered every case it
		// did.
		let (min, max) = mesh.value().bounds();
		if min.cmpge(max).all() {
			continue;
		}

		let (start, direction) = local(transform, from, along);
		let Some(distance) = slab(start, direction, min, max) else {
			continue;
		};

		if distance < nearest {
			nearest = distance;
			found = Pick::Entity(id);
		}
	}

	found
}

/// A ray from the pointer into the world.
///
/// @param camera - the camera the picture was drawn through
/// @param at - where the pointer is, origin top left
/// @param viewport - how wide and tall that picture is, in the same units
/// @return where the ray starts and which way it goes
pub(crate) fn ray(camera: &Camera, at: Vec2, viewport: Vec2) -> (Vec3, Vec3) {
	(camera.position, camera.pixel_direction(at, viewport))
}

/// A ray put into an entity's own space.
///
/// A [`Transform`] is translate, rotate, scale in that order, so undoing it is
/// the same three the other way round. The direction is scaled too and is
/// therefore no longer a unit vector - which is correct and is why the slab
/// test below is written in terms of a fraction along the ray rather than a
/// length.
fn local(transform: &Transform, from: Vec3, along: Vec3) -> (Vec3, Vec3) {
	// a zero on any axis is an entity squashed flat, and dividing by it is an
	// infinity that turns the slab test into nonsense. The smallest number
	// that is not one gives the same answer everywhere it matters.
	let scale = Vec3::select(
		transform
			.scale
			.abs()
			.cmplt(Vec3::splat(f32::EPSILON)),
		Vec3::splat(f32::EPSILON),
		transform.scale,
	);
	let turn = transform.rotation.inverse();

	((turn * (from - transform.position)) / scale, (turn * along) / scale)
}

/// How far along a ray it enters a box, if it does.
///
/// The slab test, written so that a component of the direction being exactly
/// zero produces an infinity rather than a branch: a ray parallel to a face
/// then misses unless it started between that face's two planes, which is what
/// comparing infinities works out to.
///
/// @return the fraction along the ray at which it enters, zero if it started
/// inside, or nothing if it misses
fn slab(from: Vec3, along: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
	let inverse = Vec3::ONE / along;
	let first = (min - from) * inverse;
	let second = (max - from) * inverse;

	let entering = first.min(second).max_element();
	let leaving = first.max(second).min_element();

	// a ray that leaves before it enters missed, and one that leaves behind
	// where it started is pointing away from the box.
	if leaving < entering.max(0.0) {
		return None;
	}

	Some(entering.max(0.0))
}

#[cfg(test)]
mod tests {
	use std::f32::consts::FRAC_PI_4;

	use colby_core::{
		abi::{MeshData, MeshId, Renderable, mesh},
		glam::Quat,
	};

	use super::*;

	/// A world holding one cube mesh, ready for entities to draw it.
	fn cubed() -> World {
		let mut world = World::new();
		world.meshes.insert("meshes/cube", mesh::cube());

		world
	}

	/// Stands a cube somewhere.
	fn stood(world: &mut World, transform: Transform) -> Pick {
		let mesh = world.meshes.find("meshes/cube");
		let id = world.entities.spawn_at(transform);
		world
			.entities
			.set_renderable(id, Renderable::new(mesh, Vec3::ONE));

		Pick::Entity(id)
	}

	#[test]
	fn a_ray_through_something_finds_it() {
		let mut world = cubed();
		let cube = stood(&mut world, Transform::IDENTITY);

		assert_eq!(under(&world, Vec3::Z * 10.0, Vec3::NEG_Z), cube, "straight at it");
	}

	#[test]
	fn a_ray_beside_something_finds_nothing() {
		let mut world = cubed();
		stood(&mut world, Transform::IDENTITY);

		assert_eq!(
			under(&world, Vec3::new(9.0, 0.0, 10.0), Vec3::NEG_Z),
			Pick::Nothing,
			"past it on the right"
		);
		assert_eq!(
			under(&world, Vec3::Z * 10.0, Vec3::Z),
			Pick::Nothing,
			"and pointing the other way"
		);
	}

	#[test]
	fn the_nearer_of_two_is_the_one_that_is_found() {
		let mut world = cubed();
		let far = stood(&mut world, Transform::at(Vec3::ZERO));
		let near = stood(&mut world, Transform::at(Vec3::Z * 4.0));

		assert_ne!(near, far, "two of them");
		assert_eq!(
			under(&world, Vec3::Z * 10.0, Vec3::NEG_Z),
			near,
			"the one the ray reaches first"
		);
		assert_eq!(
			under(&world, Vec3::NEG_Z * 10.0, Vec3::Z),
			far,
			"and from the other side, the other one"
		);
	}

	#[test]
	fn something_turned_is_tested_as_the_shape_it_looks_like() {
		let mut world = cubed();
		let mut turned = Transform::at(Vec3::ZERO);
		turned.rotation = Quat::from_rotation_y(FRAC_PI_4);
		let cube = stood(&mut world, turned);

		// a unit cube turned an eighth of a turn is a diamond seen from above,
		// with its corners about 0.707 out and its edges on `x + z = 0.707`.
		// The box *around* that diamond is the one a world-space test would
		// use, and it reaches 0.707 on both axes at once - so a ray that runs
		// parallel to an edge of the diamond and just outside it passes
		// through that box's corner and through no part of the cube.
		assert_eq!(
			under(&world, Vec3::new(0.0, 0.0, 1.2), Vec3::new(1.0, 0.0, -1.0)),
			Pick::Nothing,
			"it goes through the corner of the box around it and misses the thing itself"
		);
		// and the other way round, which is what makes this about the turn
		// rather than about the box around it: a corner of the diamond reaches
		// out past where the *unturned* cube ends, so a ray down z at x = 0.6
		// hits the thing as it stands and would miss it if the turn were
		// ignored.
		assert_eq!(
			under(&world, Vec3::new(0.6, 0.0, 10.0), Vec3::NEG_Z),
			cube,
			"the corner is out there and the ray goes through it"
		);
		assert_ne!(
			under(&world, Vec3::new(0.2, 0.0, 10.0), Vec3::NEG_Z),
			Pick::Nothing,
			"and the middle of it is still where it always was"
		);
	}

	#[test]
	fn something_scaled_is_as_big_as_it_is_drawn() {
		let mut world = cubed();
		let mut big = Transform::at(Vec3::ZERO);
		big.set_scale(4.0);
		let cube = stood(&mut world, big);

		assert_eq!(
			under(&world, Vec3::new(1.5, 0.0, 10.0), Vec3::NEG_Z),
			cube,
			"a ray that would miss the mesh hits the thing drawn four times its size"
		);
	}

	#[test]
	fn a_ray_that_starts_inside_hits_at_no_distance_at_all() {
		let mut world = cubed();
		let cube = stood(&mut world, Transform::IDENTITY);

		assert_eq!(under(&world, Vec3::ZERO, Vec3::X), cube, "it is already in there");
	}

	#[test]
	fn something_drawing_nothing_is_not_there_to_be_clicked() {
		let mut world = cubed();
		let empty = world.entities.spawn_at(Transform::IDENTITY);

		assert!(world.entities.alive(empty), "it exists");
		assert!(!MeshId::NONE.is_some(), "and draws nothing");
		assert_eq!(
			under(&world, Vec3::Z * 10.0, Vec3::NEG_Z),
			Pick::Nothing,
			"so a ray through where it stands finds nothing"
		);
	}

	#[test]
	fn a_mesh_with_no_vertices_is_not_a_point_to_be_hit() {
		let mut world = cubed();
		world
			.meshes
			.insert("meshes/empty", MeshData::default());
		let mesh = world.meshes.find("meshes/empty");
		let id = world.entities.spawn_at(Transform::IDENTITY);
		world
			.entities
			.set_renderable(id, Renderable::new(mesh, Vec3::ONE));

		assert_eq!(
			under(&world, Vec3::Z * 10.0, Vec3::NEG_Z),
			Pick::Nothing,
			"an empty mesh reports a point at the origin, and that is not a shape"
		);
	}

	#[test]
	fn a_view_that_nobody_disturbed_is_the_one_from_last_frame() {
		let mut camera = Camera::DEFAULT;
		let mut view = View::taken(None, &camera);
		// far enough round that the angle is past a half turn. Reading it back
		// out of a camera would give the same *direction* wrapped into the
		// range an arc tangent hands back, which is a different number - and
		// telling those two apart is the whole reason the orbit is kept rather
		// than re-derived every frame.
		view.orbit().turn(Vec2::new(-600.0, 0.0));
		view.put(&mut camera);

		let again = View::taken(Some(view), &camera);

		assert_eq!(again, view, "the world's camera is exactly what this put there");
		assert!(
			view.orbit.yaw > std::f32::consts::PI,
			"and the angle it is holding is one no reading could recover: {}",
			view.orbit.yaw
		);
	}

	#[test]
	fn a_camera_somebody_else_moved_is_the_one_the_view_takes() {
		let mut camera = Camera::DEFAULT;
		let mut view = View::taken(None, &camera);
		view.put(&mut camera);

		// what a scene loaded from the console leaves behind, or the world
		// that comes back when play stops.
		camera.position = Vec3::new(20.0, 30.0, 40.0);
		camera.target = Vec3::new(1.0, 1.0, 1.0);

		let mut again = View::taken(Some(view), &camera);
		let mut written = camera;
		again.put(&mut written);

		assert_ne!(again, view, "it did not carry on from where it was");
		assert!(
			written
				.position
				.abs_diff_eq(camera.position, 1.0e-3),
			"it took the camera it found: {} against {}",
			written.position,
			camera.position
		);
	}

	#[test]
	fn a_ray_starts_at_the_camera_and_goes_where_the_pointer_is() {
		let mut camera = Camera::DEFAULT;
		camera.target = Vec3::ZERO;
		camera.position = Vec3::Z * 6.0;

		let viewport = Vec2::new(800.0, 600.0);
		let (from, middle) = ray(&camera, viewport * 0.5, viewport);

		assert_eq!(from, camera.position, "it starts where the camera is");
		assert!(
			middle.abs_diff_eq(Vec3::NEG_Z, 1.0e-3),
			"and the middle of the picture is straight ahead: {middle}"
		);

		let (_, corner) = ray(&camera, Vec2::new(viewport.x, 0.0), viewport);

		assert!(corner.x > 0.0, "the top right goes right");
		assert!(corner.y > 0.0, "and up, which is the axis a screen counts the other way");
	}

	#[test]
	fn an_orbit_read_out_of_a_camera_puts_it_back_where_it_was() {
		let mut camera = Camera::DEFAULT;
		camera.target = Vec3::new(1.0, 2.0, 3.0);
		camera.position = Vec3::new(5.0, 6.0, 7.0);

		let mut written = camera;
		Fly::read(&camera).write(&mut written);

		assert!(
			written
				.position
				.abs_diff_eq(camera.position, 1.0e-4),
			"{} against {}",
			written.position,
			camera.position
		);
		assert!(written.target.abs_diff_eq(camera.target, 1.0e-4));
	}

	#[test]
	fn turning_moves_the_camera_and_leaves_what_it_looks_at_alone() {
		let mut camera = Camera::DEFAULT;
		camera.target = Vec3::ZERO;
		camera.position = Vec3::Z * 9.0;

		let mut fly = Fly::read(&camera);
		fly.turn(Vec2::new(60.0, 0.0));
		fly.write(&mut camera);

		assert!(camera.target.abs_diff_eq(Vec3::ZERO, 1.0e-4), "the target stayed");
		assert!(!camera.position.abs_diff_eq(Vec3::Z * 9.0, 1.0e-3), "the camera did not");
		assert!(
			(camera.position.length() - 9.0).abs() < 1.0e-3,
			"and it stayed the same distance out: {}",
			camera.position.length()
		);
	}

	#[test]
	fn the_pitch_stops_short_of_the_poles() {
		let mut fly = Fly::read(&Camera::DEFAULT);

		for _ in 0..200 {
			fly.turn(Vec2::new(0.0, 100.0));
		}

		assert!(fly.pitch <= 1.55, "held below the top: {}", fly.pitch);

		for _ in 0..400 {
			fly.turn(Vec2::new(0.0, -100.0));
		}

		assert!(fly.pitch >= -1.55, "and above the bottom: {}", fly.pitch);
	}

	#[test]
	fn the_wheel_covers_the_same_share_of_the_way_in_every_time() {
		let mut fly = Fly::read(&Camera::DEFAULT);
		let started = fly.distance;

		fly.dolly(1.0);
		let once = fly.distance;
		fly.dolly(1.0);
		let twice = fly.distance;

		assert!(once < started, "a notch comes closer");
		assert!(twice < once, "and so does the next one");
		// the claim is not that it comes closer, it is that it covers the same
		// *share* every time - so the second notch moves it less than the
		// first, in units, and it never arrives. Subtracting a fixed length
		// instead would pass the two lines above and fail this one.
		assert!(
			(once - twice) < (started - once),
			"the second notch is the smaller step: {} against {}",
			once - twice,
			started - once
		);
		assert!(twice > 0.0, "without ever arriving");
	}

	#[test]
	fn the_wheel_cannot_be_wound_past_the_target_or_out_of_the_world() {
		let mut fly = Fly::read(&Camera::DEFAULT);

		for _ in 0..500 {
			fly.dolly(1.0);
		}
		assert!(fly.distance >= RANGE.0, "it stops short: {}", fly.distance);

		for _ in 0..500 {
			fly.dolly(-1.0);
		}
		assert!(fly.distance <= RANGE.1, "and it stops far out: {}", fly.distance);
	}

	#[test]
	fn sliding_takes_the_camera_along_with_what_it_looks_at() {
		let mut camera = Camera::DEFAULT;
		camera.target = Vec3::ZERO;
		camera.position = Vec3::Z * 9.0;

		let mut fly = Fly::read(&camera);
		fly.slide(Vec2::new(100.0, 0.0));
		fly.write(&mut camera);

		assert!(camera.target.x < -0.1, "the target went the other way: {}", camera.target);
		assert!(
			(camera.position - camera.target).abs_diff_eq(Vec3::Z * 9.0, 1.0e-3),
			"and the camera kept exactly its place behind it"
		);
	}
}
