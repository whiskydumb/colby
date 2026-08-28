//! What the solver draws about itself, when somebody asks.
//!
//! The three things this crate knows that nothing else can see: what shape a
//! body really has, where two of them touch and which way the push goes, and
//! where a joint is anchored. All three are the questions a physics bug turns
//! into - "is the collision the same as the picture", "is it touching where it
//! looks like it is", "is that rope tied where I said" - and none of them has
//! an answer in the window without lines.
//!
//! Three console variables, all off. Off because this is a tool rather than a
//! feature: the cost of drawing a full world's outlines is real, and a picture
//! covered in wireframe is worse than no picture for everything except the one
//! question it answers.
//!
//! Everything here writes into `world.debug` and nothing here knows what a
//! pipeline is. @ref [`colby_core::abi::debug`].

use core::mem;

use colby_core::{
	abi::{Body, BodyKind, JointKind, ShapeKind, World, debug},
	glam::Vec3,
};

use crate::{Simulation, contact::Manifold};

/// The console variable that outlines every body.
pub const SHAPES: &str = "phys.draw_shapes";

/// The console variable that marks every contact and its normal.
pub const CONTACTS: &str = "phys.draw_contacts";

/// The console variable that marks every joint and its two anchors.
pub const JOINTS: &str = "phys.draw_joints";

/// How long the arm of a contact's cross is, in world units.
const CONTACT_SIZE: f32 = 0.04;

/// How long a contact normal is drawn, in world units.
///
/// A constant rather than the penetration depth: a depth is a fraction of a
/// millimeter in a settled pile, so an arrow that long would be invisible
/// exactly when the pile is behaving. The depth is what the color says instead.
const NORMAL_LENGTH: f32 = 0.35;

/// How deep an overlap has to be to be drawn as a problem rather than a touch.
const DEEP: f32 = 0.02;

/// How long the arm of a joint anchor's cross is, in world units.
const ANCHOR_SIZE: f32 = 0.05;

/// Draws whatever the console has asked for.
///
/// Called at the end of a step, after the solver has run: what it draws is then
/// the world the next frame will show rather than the one the last frame did.
///
/// @param world - read for its bodies and joints, written for its lines
/// @param simulation - the baked collision meshes and the manifolds
pub(crate) fn draw(world: &mut World, simulation: &Simulation) {
	let asked = |name| world.cvars.bool(name).unwrap_or(false);
	let (shapes, contacts, joints) = (asked(SHAPES), asked(CONTACTS), asked(JOINTS));

	if !(shapes || contacts || joints) {
		return;
	}

	// taken out and put back, the way the narrow phase borrows the simulation
	// while filling a list that lives on it. The alternative is for each of
	// these to copy what it is about to draw out of the world first - a pen
	// borrows the world mutably and the bodies live in the same world - and
	// three allocations a step for a tool that is usually off is the wrong way
	// round. Measured: it was most of what turning this on cost.
	let mut table = mem::take(&mut world.debug);

	if shapes {
		draw_shapes(&mut table, world, simulation);
	}

	if contacts {
		draw_contacts(&mut table, simulation.manifolds());
	}

	if joints {
		draw_joints(&mut table, world);
	}

	world.debug = table;
}

/// Outlines every body in the color of what the solver may do with it.
fn draw_shapes(table: &mut debug::Debug, world: &World, simulation: &Simulation) {
	let mut pen = table.pen();

	for (id, body) in world.bodies.iter() {
		let color = shape_color(body);
		let transform = body.transform;

		match body.shape.kind {
			| ShapeKind::Box => pen.cuboid(
				transform.position,
				body.shape.extents.abs() * transform.scale.abs(),
				transform.rotation,
				color,
			),
			// the radius the *solver* uses, which under a non-uniform scale is
			// the sphere around the ellipsoid rather than the ellipsoid.
			// Drawing the ellipsoid would be drawing what the ray query
			// believes and not what the contacts do.
			| ShapeKind::Sphere => pen.ball(
				transform.position,
				body.shape.radius.abs() * transform.scale.abs().max_element(),
				color,
			),
			| ShapeKind::Mesh => {
				let Some(collider) = simulation.collider(id) else {
					continue;
				};

				let matrix = transform.matrix();
				for corners in collider.triangles() {
					let [first, second, third] =
						corners.map(|corner| matrix.transform_point3(corner));

					pen.line(first, second, color);
					pen.line(second, third, color);
					pen.line(third, first, color);
				}
			},
		}
	}
}

/// What color a body is outlined in.
///
/// The distinction worth seeing at a glance is not what a body *is* but what it
/// is *doing*: a dynamic body that has gone to sleep behaves like a wall until
/// something wakes it, and a pile that will not settle looks identical to one
/// that has until the colors say otherwise.
fn shape_color(body: &Body) -> Vec3 {
	match body.kind {
		| BodyKind::Static => debug::GRAY,
		| BodyKind::Kinematic => debug::CYAN,
		| BodyKind::Dynamic if body.sleeping => debug::BLUE,
		| BodyKind::Dynamic => debug::GREEN,
	}
}

/// Marks every contact the narrow phase found, and which way it pushes.
///
/// Drawn over the world rather than behind it, and that is the whole reason the
/// depth rule exists: a contact is by definition on the surface between two
/// bodies, so a depth-tested cross there is half inside the very body that
/// produced it.
fn draw_contacts(table: &mut debug::Debug, manifolds: &[Manifold]) {
	let mut pen = table.on_top();

	for manifold in manifolds {
		for contact in manifold.points() {
			// magenta for an overlap deep enough to be a symptom, red for the
			// shallow ones a settled pile is made of. The arrow is a constant
			// length because the depth it would otherwise be is a fraction of a
			// millimeter exactly when the pile is behaving.
			let color = if contact.depth > DEEP {
				debug::MAGENTA
			} else {
				debug::RED
			};

			pen.point(contact.position, CONTACT_SIZE, color);
			pen.arrow(
				contact.position,
				contact.position + manifold.normal * NORMAL_LENGTH,
				color,
			);
		}
	}
}

/// Marks every joint: a line between its anchors, a cross at each end.
///
/// The anchors are in each body's own space, so this is the one drawing that
/// shows what the solver is actually holding rather than where the two bodies
/// happen to be. A weld made across a gap looks exactly like a correct one from
/// the outside and nothing like it here.
fn draw_joints(table: &mut debug::Debug, world: &World) {
	let anchor = |id, local: Vec3| {
		world
			.bodies
			.get(id)
			.map_or(local, |body| body.transform.matrix().transform_point3(local))
	};

	let mut pen = table.on_top();

	for (_, joint) in world.joints.iter() {
		let color = match joint.kind {
			| JointKind::Rope => debug::YELLOW,
			| JointKind::Weld => debug::ORANGE,
			| JointKind::Axis => debug::CYAN,
		};

		// a joint whose second body is nothing is pinned to a point in the
		// world, and its second anchor is already a world position. That falls
		// out of the lookup failing rather than needing a branch.
		let (first, second) = (
			anchor(joint.first, joint.first_anchor),
			anchor(joint.second, joint.second_anchor),
		);

		pen.line(first, second, color);
		pen.point(first, ANCHOR_SIZE, color);
		pen.point(second, ANCHOR_SIZE, color);
	}
}

#[cfg(test)]
mod tests {
	use colby_core::abi::{
		BodyId, Joint, Shape, Transform, Value,
		debug::{self, MAX_LINES},
	};

	use super::*;

	/// A world with the named variables turned on.
	fn asking(names: &[&str]) -> World {
		let mut world = World::new();

		for name in names {
			world.cvars.var(name, Value::Bool(true), "");
		}

		world
	}

	#[test]
	fn nothing_is_drawn_until_a_variable_is_turned_on() {
		let mut world = World::new();
		world
			.bodies
			.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY));

		draw(&mut world, &Simulation::new());

		assert!(world.debug.is_empty(), "a tool that draws without being asked is not off");
	}

	#[test]
	fn a_box_body_is_outlined_where_it_stands() {
		let mut world = asking(&[SHAPES]);
		world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::cuboid(Vec3::splat(0.5)),
			Transform::at(Vec3::new(4.0, 0.0, 0.0)),
		));

		draw(&mut world, &Simulation::new());

		assert_eq!(world.debug.lines().len(), 12, "one box, twelve edges");
		for line in world.debug.lines() {
			assert!(
				(line.from.x - 4.0).abs() <= 0.5 + 1.0e-4,
				"every corner is half a unit from the middle, got {}",
				line.from.x
			);
		}
	}

	#[test]
	fn a_scaled_box_is_outlined_at_the_size_the_solver_collides() {
		let mut world = asking(&[SHAPES]);
		let mut transform = Transform::IDENTITY;
		transform.scale = Vec3::splat(3.0);

		world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::cuboid(Vec3::splat(0.5)),
			transform,
		));

		draw(&mut world, &Simulation::new());

		let widest = world
			.debug
			.lines()
			.iter()
			.fold(0.0_f32, |most, line| most.max(line.from.x.abs()));
		assert!(
			(widest - 1.5).abs() < 1.0e-4,
			"the scale multiplies the extents, exactly as the narrow phase does it: got {widest}"
		);
	}

	#[test]
	fn the_four_states_a_body_can_be_in_are_four_colors() {
		let awake = Body::dynamic(Shape::UNIT, Transform::IDENTITY, 1.0);
		let mut asleep = awake;
		asleep.sleeping = true;

		let stationary = Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY);
		let driven = Body::new(BodyKind::Kinematic, Shape::UNIT, Transform::IDENTITY);

		let colors = [
			shape_color(&awake),
			shape_color(&asleep),
			shape_color(&stationary),
			shape_color(&driven),
		];

		for (index, color) in colors.iter().enumerate() {
			for other in colors.iter().skip(index + 1) {
				assert!(
					!color.abs_diff_eq(*other, 1.0e-3),
					"a pile that has settled and one that has not must not look the same, nor a \
					 wall and a moving platform: {color} against {other}"
				);
			}
		}
	}

	#[test]
	fn a_contact_is_marked_over_the_world_rather_than_inside_it() {
		let mut world = asking(&[CONTACTS]);
		let mut simulation = Simulation::new();

		// two unit boxes overlapping by a tenth, which is a real manifold.
		let floor = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::UNIT,
			Transform::at(Vec3::ZERO),
		));
		let prop = world.bodies.spawn(Body::dynamic(
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 0.9, 0.0)),
			1.0,
		));
		assert!(floor.is_some() && prop.is_some(), "both bodies were taken");

		simulation.step(&mut world);
		draw(&mut world, &simulation);

		assert!(!world.debug.is_empty(), "two overlapping boxes touch and it is drawn");
		assert!(
			world.debug.lines().iter().all(|line| line.on_top),
			"a contact sits on the surface between two bodies, so a depth-tested cross there is \
			 half inside one of them"
		);
	}

	#[test]
	fn a_joint_is_drawn_between_the_anchors_and_not_between_the_bodies() {
		let mut world = asking(&[JOINTS]);
		let body = world.bodies.spawn(Body::dynamic(
			Shape::UNIT,
			Transform::at(Vec3::new(0.0, 2.0, 0.0)),
			1.0,
		));

		world.join(Joint {
			kind: JointKind::Rope,
			first: body,
			second: BodyId::NONE,
			// a corner of the body, not its middle.
			first_anchor: Vec3::new(0.5, 0.5, 0.0),
			second_anchor: Vec3::new(0.0, 5.0, 0.0),
			..Joint::default()
		});

		draw(&mut world, &Simulation::new());

		let span = world
			.debug
			.lines()
			.first()
			.expect("the joint was drawn");
		assert!(
			span.from
				.abs_diff_eq(Vec3::new(0.5, 2.5, 0.0), 1.0e-4),
			"the near end is the anchor in the body's own space, put into the world: got {}",
			span.from
		);
		assert!(
			span.to
				.abs_diff_eq(Vec3::new(0.0, 5.0, 0.0), 1.0e-4),
			"and the far end is a world position, because there is no second body: got {}",
			span.to
		);
	}

	#[test]
	fn the_three_kinds_of_joint_are_told_apart() {
		let mut colors = Vec::new();

		for kind in [JointKind::Rope, JointKind::Weld, JointKind::Axis] {
			let mut world = asking(&[JOINTS]);
			let body = world
				.bodies
				.spawn(Body::dynamic(Shape::UNIT, Transform::IDENTITY, 1.0));

			world.join(Joint {
				kind,
				first: body,
				second: BodyId::NONE,
				second_anchor: Vec3::Y,
				..Joint::default()
			});

			draw(&mut world, &Simulation::new());
			colors.push(
				world
					.debug
					.lines()
					.first()
					.map_or(debug::WHITE, |line| line.color),
			);
		}

		assert!(
			!colors[0].abs_diff_eq(colors[1], 1.0e-3)
				&& !colors[1].abs_diff_eq(colors[2], 1.0e-3),
			"a rope, a weld and a hinge are three different failures and read as three colors"
		);
	}

	#[test]
	fn a_full_world_of_outlines_stops_at_the_bound_rather_than_growing() {
		let mut world = asking(&[SHAPES]);

		// a ball is seventy-two lines, so a thousand of them is well past it.
		for _ in 0..1000 {
			world.bodies.spawn(Body::new(
				BodyKind::Static,
				Shape::ball(0.5),
				Transform::IDENTITY,
			));
		}

		draw(&mut world, &Simulation::new());

		assert_eq!(world.debug.lines().len(), MAX_LINES, "it stops where the table stops");
		assert!(world.debug.dropped() > 0, "and the count says the picture is short");
	}
}
