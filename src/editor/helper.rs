//! What the editor draws for a thing that has nothing to look at, and the
//! handles on the fields such a thing is made of.
//!
//! **A helper is code here rather than a word in a table**, which is what every
//! editor read for this does: the kinds are the engine's, they are few, and
//! what each one is worth drawing is a different shape for each. A lamp is a
//! sphere or a cone, a tie is a line between two anchors, a body nobody draws
//! is its own outline. A table could say "a sphere of this field" and could not
//! say "a cone of two angles down the thing's own -z, unless its kind says
//! otherwise".
//!
//! **A mark is drawn whether or not the thing is selected, and that is the
//! whole point.** A click cannot land on what is not on screen: before this, an
//! entity drawing nothing was reachable only from the hierarchy, because
//! [`aim::under`](crate::aim::under) tests mesh bounds and a lamp has none.
//! Every editor in the field draws an icon for a light at all times, and picks
//! it. What is *added* by selecting a thing is the rest of the picture - how
//! far it reaches, which way it throws - the way a lamp's reach has been drawn
//! since the light panel existed.
//!
//! **A mark wins a click against a mesh behind or in front of it.** The field
//! is split three ways here and colby's reason is its own: the marks are
//! painted over the picture with no depth at all, and a mesh is picked by its
//! *bounds*, so a ray that starts inside a bounding box - the camera standing
//! in a baked room, or under a hill - answers at no distance and would take
//! every click. What is drawn on top is what is clicked, which is also the rule
//! the gizmo's own arms already follow.
//!
//! **Everything here is a function of a [`World`] and a [`Camera`]**, like
//! [`gizmo`](crate::gizmo) and for the same reason: the drawing is egui's and
//! is checked by looking at it, and this half is checked by running it.

use colby_core::{
	abi::{
		Camera, Draw, Emitter, EntityId, JointId, JointKind, Light, LightKind, MAX_CONE, Shape,
		ShapeKind, Transform, World, field::Value,
	},
	glam::{Mat4, Quat, Vec2, Vec3},
};

use crate::{
	gizmo,
	select::{self, Edit, Pick},
};

/// Half the width of a mark on screen, in points.
pub(crate) const MARK: f32 = 8.0;

/// How near the pointer has to be to the middle of a mark, in points.
///
/// A little wider than the mark is drawn, the way an arm is one pixel of ink
/// and eight of reach. @ref [`gizmo::GRAB`].
pub(crate) const REACH: f32 = 10.0;

/// How near the pointer has to be to a handle, in points.
pub(crate) const HOLD: f32 = 9.0;

/// The shortest reach a handle may leave a lamp with, in world units.
///
/// A lamp whose range is nothing throws nothing, and a lamp that throws nothing
/// has no mark and no handle - so a drag that took the range to nought would
/// take away the thing the pointer was holding. The inspector can still type a
/// nought; a handle stops short of it.
pub(crate) const LEAST_RANGE: f32 = 0.05;

/// The narrowest a handle may make a cone, in radians.
///
/// A cone of no angle has a rim of no radius, and its handle would sit in the
/// middle of the mouth under the range handle: the dead handle the rotate ring
/// already refuses to draw.
pub(crate) const LEAST_CONE: f32 = 0.02;

/// How long a hinge's axis is drawn, in world units.
const AXIS_LENGTH: f32 = 0.5;

/// How long the arm of a cross at an anchor is, in world units.
const ANCHOR_SIZE: f32 = 0.06;

/// How many straight pieces a circle in the world is drawn with.
const CIRCLE_STEPS: usize = 24;

/// How long an arrow's barbs are, as a share of the shaft.
const BARB: f32 = 0.2;

/// How far out those barbs spread, as a share of their own length.
const BARB_SPREAD: f32 = 0.5;

/// What a mark stands for: a glyph, a color and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
	/// A lamp throwing light every way.
	Lamp,

	/// A lamp throwing a cone down its own -z.
	Spot,

	/// A thing throwing particles.
	Thrower,

	/// A thing painting a picture onto what is around it.
	Painter,
}

/// One thing with nothing to look at, marked where it stands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Mark {
	/// What a click on it selects.
	pub(crate) pick: Pick,

	/// Which glyph.
	pub(crate) kind: Kind,

	/// Where it is, in points from the picture's corner.
	pub(crate) at: Vec2,

	/// Which way the thing points on screen, for the kinds that point: a unit
	/// vector, or nothing when the two ends land in one place.
	pub(crate) aim: Option<Vec2>,
}

/// What an outline is drawn for, which is what it is drawn in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Drawn {
	/// A joint, between the two places it holds.
	Tie,

	/// A body nobody draws.
	Solid,

	/// A body nobody draws that is full of fluid.
	Fluid,
}

/// One thing with nothing to look at whose shape *is* the picture: lines in the
/// world rather than a glyph on the screen.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Outline {
	/// What a click on it selects.
	pub(crate) pick: Pick,

	/// What it is.
	pub(crate) drawn: Drawn,

	/// The lines, in world space.
	pub(crate) segments: Vec<(Vec3, Vec3)>,
}

/// Which field a handle writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Knob {
	/// How far a lamp reaches: `range`.
	Range,

	/// The half-angle of a cone's edge: `outer`.
	Outer,

	/// A field of a game's record that says it is drawn as a distance.
	Radius {
		/// The record's place in the declared records.
		table: usize,

		/// The field's place in its record.
		column: usize,
	},

	/// A field of a game's record that says it is a place in the thing.
	Point {
		/// The record's place in the declared records.
		table: usize,

		/// The field's place in its record.
		column: usize,
	},
}

impl Knob {
	/// What to call the step back it opens.
	///
	/// A record's own word is the inspector's, so that dragging a field in the
	/// picture and typing it in the panel read the same in the undo button.
	pub(crate) const fn word(self) -> &'static str {
		match self {
			| Self::Range => "reach",
			| Self::Outer => "cone",
			| Self::Radius { .. } | Self::Point { .. } => "record",
		}
	}
}

/// Where a drag on a handle is measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Along {
	/// A distance along a line: a reach, a radius, the rim of a cone.
	Line {
		/// Where the line starts.
		origin: Vec3,

		/// Which way it goes; a unit vector.
		way: Vec3,
	},

	/// A place on the plane through the handle facing the camera, which is what
	/// a point is dragged on: three arms for it would be the move gizmo again,
	/// on a thing that is not a transform.
	Plane {
		/// Which way that plane faces; a unit vector.
		normal: Vec3,
	},
}

/// What the field a handle writes holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Held {
	/// A number: a reach, a radius, an angle.
	Number(f32),

	/// A place, in the world: what a field holding one in the thing's own
	/// terms comes to where the thing now stands.
	Place(Vec3),
}

/// One handle: where it sits, and where a drag on it is measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Handle {
	/// Which field it writes.
	pub(crate) knob: Knob,

	/// Where it is, in the world.
	pub(crate) at: Vec3,

	/// The line or the plane a drag on it is measured on.
	pub(crate) along: Along,

	/// The lamp's reach when the handle was worked out, which an angle is
	/// measured against.
	pub(crate) range: f32,

	/// What the field held then.
	pub(crate) held: Held,
}

/// Every thing with nothing to look at, marked where it stands on screen.
///
/// An entity that draws something is not marked: it can be clicked where it is
/// drawn, and a glyph over it would be one more thing in the way. Nor is one
/// that is hidden, nor one that is nothing in particular - an empty entity or a
/// group is reached through what hangs off it or through its row, which is what
/// three of the five editors in the field do with an empty node.
///
/// @param world - what to look through
/// @param camera - the camera the picture was drawn with
/// @param viewport - how big that picture is, in points
pub(crate) fn marks(world: &World, camera: &Camera, viewport: Vec2) -> Vec<Mark> {
	let view = camera.view_projection(viewport.x.max(1.0) / viewport.y.max(1.0));
	let mut found = Vec::new();

	for (id, _, renderable) in world.entities.iter() {
		if !world.entities.shown(id) {
			continue;
		}

		// a mesh with nothing in it reports a point, and the null mesh is such
		// a mesh: `aim::under`'s rule for a thing that draws nothing
		let drawn = world
			.meshes
			.get(renderable.mesh)
			.map(|mesh| mesh.value().bounds())
			.is_some_and(|(min, max)| !min.cmpge(max).all());

		if drawn {
			continue;
		}

		let (Some(kind), Some(placed)) = (kind_of(world, id), world.entities.placed(id)) else {
			continue;
		};

		let Some(at) = gizmo::project(view, placed.position, viewport) else {
			continue;
		};

		let aim = if kind == Kind::Spot {
			let way = (placed.rotation * Vec3::NEG_Z).normalize_or(Vec3::NEG_Z);

			gizmo::project(view, placed.position + way, viewport)
				.and_then(|tip| (tip - at).try_normalize())
		} else {
			None
		};

		found.push(Mark { pick: Pick::Entity(id), kind, at, aim });
	}

	found
}

/// What an entity with no mesh is, if it is anything.
fn kind_of(world: &World, id: EntityId) -> Option<Kind> {
	if let Some(light) = world.entities.light(id).filter(|it| it.is_lit()) {
		return Some(if light.kind == LightKind::Spot {
			Kind::Spot
		} else {
			Kind::Lamp
		});
	}

	if world
		.entities
		.emitter(id)
		.is_some_and(|emitter| emitter.kind.throws())
	{
		return Some(Kind::Thrower);
	}

	world
		.entities
		.decal(id)
		.is_some_and(|decal| decal.paints())
		.then_some(Kind::Painter)
}

/// Every joint, and every body nobody draws, as lines in the world.
///
/// **A joint is drawn at all times**, as it is in all five editors read for
/// this, and it is the only thing on screen that says what the solver is
/// holding: the two anchors are in the bodies' own spaces, so a weld made
/// across a gap looks like a correct one everywhere else. **A body is drawn
/// when nothing draws where it stands** - a sensor, a pool with no entity, a
/// bare collider - because otherwise nothing at all says it is there. A mesh
/// shape is left out: its triangles came from an asset something else is
/// drawing, and a soup is not an outline.
///
/// @param world - what to look through
pub(crate) fn outlines(world: &World) -> Vec<Outline> {
	let mut found = Vec::new();

	for (id, _) in world.joints.iter() {
		found.push(Outline {
			pick: Pick::Joint(id),
			drawn: Drawn::Tie,
			segments: vec![anchors(world, id)],
		});
	}

	for (id, body) in world.bodies.iter() {
		let entity = world
			.entities
			.alive(body.entity)
			.then_some(body.entity);

		if let Some(entity) = entity
			&& (!world.entities.shown(entity) || draws(world, entity))
		{
			continue;
		}

		let mut segments = shape_of(body.transform, body.shape);

		if segments.is_empty() {
			continue;
		}

		let fluid = body.water.is_wet();

		if let (true, Some((low, high))) = (fluid, body.bounds()) {
			segments.extend(surface(low, high, body.surface().unwrap_or(high.y)));
		}

		found.push(Outline {
			pick: entity.map_or(Pick::Body(id), Pick::Entity),
			drawn: if fluid { Drawn::Fluid } else { Drawn::Solid },
			segments,
		});
	}

	found
}

/// Whether an entity draws anything at all.
fn draws(world: &World, id: EntityId) -> bool {
	world
		.entities
		.renderable(id)
		.and_then(|look| world.meshes.get(look.mesh))
		.map(|mesh| mesh.value().bounds())
		.is_some_and(|(min, max)| !min.cmpge(max).all())
}

/// A joint's two anchors, in the world.
///
/// An anchor is kept in its body's own space; one whose body is gone - a joint
/// pinned to a point in the world - is already a world position, which falls
/// out of the lookup failing rather than needing a branch.
fn anchors(world: &World, id: JointId) -> (Vec3, Vec3) {
	let Some(joint) = world.joints.get(id) else {
		return (Vec3::ZERO, Vec3::ZERO);
	};

	let at = |body, local: Vec3| {
		world
			.bodies
			.get(body)
			.map_or(local, |held| held.transform.matrix().transform_point3(local))
	};

	(at(joint.first, joint.first_anchor), at(joint.second, joint.second_anchor))
}

/// What a selected thing adds to what is drawn for it: a tie's anchors, and the
/// axis a hinge turns about.
///
/// @param world - what to look through
/// @param pick - what is selected
pub(crate) fn detail(world: &World, pick: Pick) -> Vec<(Vec3, Vec3)> {
	let Pick::Joint(id) = pick else {
		return Vec::new();
	};

	let Some(joint) = world.joints.get(id).copied() else {
		return Vec::new();
	};

	let (first, second) = anchors(world, id);
	let mut segments = Vec::new();

	for at in [first, second] {
		for way in [Vec3::X, Vec3::Y, Vec3::Z] {
			segments.push((at - way * ANCHOR_SIZE, at + way * ANCHOR_SIZE));
		}
	}

	if joint.kind == JointKind::Axis {
		let turn = world
			.bodies
			.get(joint.first)
			.map_or(Quat::IDENTITY, |body| body.transform.rotation);
		let way = (turn * joint.axis).normalize_or(Vec3::Y);

		segments.extend(arrow(first, first + way * AXIS_LENGTH));
	}

	segments
}

/// The lines of a body's shape, in the world; empty for a shape that has none.
fn shape_of(transform: Transform, shape: Shape) -> Vec<(Vec3, Vec3)> {
	match shape.kind {
		| ShapeKind::Box => {
			let half = shape.extents.abs() * transform.scale.abs();
			let matrix = Mat4::from_scale_rotation_translation(
				half * 2.0,
				transform.rotation,
				transform.position,
			);

			box_edges(matrix)
		},
		// the radius the *solver* uses, which under a lopsided scale is the
		// sphere around the ellipsoid rather than the ellipsoid
		| ShapeKind::Sphere => {
			let wide = shape.radius.abs() * transform.scale.abs().max_element();
			let mut segments = Vec::new();

			for normal in [Vec3::X, Vec3::Y, Vec3::Z] {
				segments.extend(circle(transform.position, normal, wide));
			}

			segments
		},
		| ShapeKind::Mesh => Vec::new(),
	}
}

/// The twelve edges of the box a matrix takes the unit cube to.
fn box_edges(matrix: Mat4) -> Vec<(Vec3, Vec3)> {
	const CORNERS: [Vec3; 8] = [
		Vec3::new(-0.5, -0.5, -0.5),
		Vec3::new(0.5, -0.5, -0.5),
		Vec3::new(0.5, 0.5, -0.5),
		Vec3::new(-0.5, 0.5, -0.5),
		Vec3::new(-0.5, -0.5, 0.5),
		Vec3::new(0.5, -0.5, 0.5),
		Vec3::new(0.5, 0.5, 0.5),
		Vec3::new(-0.5, 0.5, 0.5),
	];
	const EDGES: [(usize, usize); 12] = [
		(0, 1),
		(1, 2),
		(2, 3),
		(3, 0),
		(4, 5),
		(5, 6),
		(6, 7),
		(7, 4),
		(0, 4),
		(1, 5),
		(2, 6),
		(3, 7),
	];

	let at = CORNERS.map(|corner| matrix.transform_point3(corner));

	EDGES
		.into_iter()
		.map(|(from, to)| (at[from], at[to]))
		.collect()
}

/// A circle in the world, as straight pieces.
fn circle(center: Vec3, normal: Vec3, radius: f32) -> Vec<(Vec3, Vec3)> {
	let facing = normal.normalize_or(Vec3::Z);
	let first = facing.any_orthonormal_vector();
	let second = facing.cross(first);
	let mut points = Vec::with_capacity(CIRCLE_STEPS + 1);

	for step in 0..=CIRCLE_STEPS {
		let angle = std::f32::consts::TAU * share(step);
		let (sin, cos) = angle.sin_cos();

		points.push(center + (first * cos + second * sin) * radius);
	}

	points
		.windows(2)
		.filter_map(|pair| Some((*pair.first()?, *pair.get(1)?)))
		.collect()
}

/// How far round a circle one step is.
fn share(step: usize) -> f32 {
	let step = u16::try_from(step).unwrap_or(u16::MAX);
	let steps = u16::try_from(CIRCLE_STEPS).unwrap_or(u16::MAX);

	f32::from(step) / f32::from(steps.max(1))
}

/// The rectangle a fluid is filled to, across the box that holds it.
///
/// Level whatever the body is turned to, because a surface is: it is a property
/// of the body's *bounds* rather than of its shape, and so is on screen nowhere
/// else. @ref `colby_core::abi::Body::surface`.
fn surface(low: Vec3, high: Vec3, at: f32) -> Vec<(Vec3, Vec3)> {
	let corners = [
		Vec3::new(low.x, at, low.z),
		Vec3::new(high.x, at, low.z),
		Vec3::new(high.x, at, high.z),
		Vec3::new(low.x, at, high.z),
	];

	(0..4)
		.map(|index| (corners[index], corners[(index + 1) % 4]))
		.collect()
}

/// How far a particle an emitter throws gets before it dies, gravity aside.
///
/// The arithmetic the step does, solved: a particle leaves at `speed` and keeps
/// a share `1 - drag` of it every second, so how far it gets is that speed
/// summed over its life. No drag at all is `speed * life`; a drag of one keeps
/// nothing and goes nowhere. **Gravity is deliberately left out**: it bends the
/// path rather than shortening it, and a cone drawn around a falling plume
/// would be a cone nobody could aim.
///
/// @param emitter - what it says about how it throws
pub(crate) fn reach(emitter: &Emitter) -> f32 {
	let kept = 1.0 - emitter.drag.clamp(0.0, 1.0);
	let life = emitter.life.max(0.0);
	let speed = emitter.speed.max(0.0);

	if kept >= 1.0 {
		return speed * life;
	}

	if kept <= 0.0 {
		return 0.0;
	}

	speed * (1.0 - kept.powf(life)) / -kept.ln()
}

/// An arrow from one place to another: a shaft and four barbs.
///
/// @param from - where it starts
/// @param to - where it points
/// @return the lines, or none at all when the two places are one
pub(crate) fn arrow(from: Vec3, to: Vec3) -> Vec<(Vec3, Vec3)> {
	let along = to - from;
	let length = along.length();

	if length < f32::EPSILON {
		return Vec::new();
	}

	let way = along / length;
	let across = way.any_orthonormal_vector();
	let other = way.cross(across);
	let back = to - way * (length * BARB);
	let wide = length * BARB * BARB_SPREAD;
	let mut segments = vec![(from, to)];

	for corner in [across, -across, other, -other] {
		segments.push((to, back + corner * wide));
	}

	segments
}

/// Which mark the pointer is on, if it is on one: the nearest within reach.
///
/// @param marks - what [`marks`] returned
/// @param at - where the pointer is, from the picture's corner
pub(crate) fn nearest(marks: &[Mark], at: Vec2) -> Option<usize> {
	let mut nearest = REACH;
	let mut found = None;

	for (index, mark) in marks.iter().enumerate() {
		let away = mark.at.distance(at);

		if away < nearest {
			nearest = away;
			found = Some(index);
		}
	}

	found
}

/// Which outline the pointer is on, if it is on one.
///
/// Measured against the lines on screen at the width an arm is grabbed at, the
/// way a gizmo's own segments are.
///
/// @param outlines - what [`outlines`] returned
/// @param camera - the camera the picture was drawn with
/// @param viewport - how big that picture is, in points
/// @param at - where the pointer is, from the picture's corner
pub(crate) fn touched(
	outlines: &[Outline],
	camera: &Camera,
	viewport: Vec2,
	at: Vec2,
) -> Option<usize> {
	let view = camera.view_projection(viewport.x.max(1.0) / viewport.y.max(1.0));
	let mut nearest = gizmo::GRAB;
	let mut found = None;

	for (index, outline) in outlines.iter().enumerate() {
		for (from, to) in &outline.segments {
			let (Some(start), Some(end)) =
				(gizmo::project(view, *from, viewport), gizmo::project(view, *to, viewport))
			else {
				continue;
			};

			let away = gizmo::to_segment(at, start, end);

			if away < nearest {
				nearest = away;
				found = Some(index);
			}
		}
	}

	found
}

/// What is under the pointer, if a helper is.
///
/// Marks first, then the lines: a mark is a thing's whole picture and a line is
/// the edge of one, so the pointer resting on both means the mark.
///
/// @param world - what to look through
/// @param camera - the camera the picture was drawn with
/// @param viewport - how big that picture is, in points
/// @param at - where the pointer is, from the picture's corner
pub(crate) fn under(world: &World, camera: &Camera, viewport: Vec2, at: Vec2) -> Option<Pick> {
	let marked = marks(world, camera, viewport);

	if let Some(index) = nearest(&marked, at) {
		return marked.get(index).map(|mark| mark.pick);
	}

	let lines = outlines(world);

	touched(&lines, camera, viewport, at).and_then(|index| lines.get(index).map(|it| it.pick))
}

/// The handles a selected thing offers.
///
/// A lamp's reach and a cone's edge, and nothing else: a decal's box and a
/// body's shape are the transform's own scale, which the size tool already
/// stretches, and nothing in the field puts a handle on a particle or a joint
/// limit. A cone's *inner* angle is drawn and has no handle: its own is nought
/// by default, and a handle at nought would sit under the reach handle in the
/// middle of the mouth.
///
/// @param world - what to look through
/// @param camera - the camera the picture was drawn with; a reach is dragged
/// across the view, so where the camera stands decides where the handle is
/// @param pick - what is selected
pub(crate) fn handles(world: &World, camera: &Camera, pick: Pick) -> Vec<Handle> {
	let Pick::Entity(id) = pick else {
		return Vec::new();
	};

	let (Some(light), Some(placed)) = (
		world
			.entities
			.light(id)
			.copied()
			.filter(|it| it.is_lit()),
		world.entities.placed(id),
	) else {
		return Vec::new();
	};

	if light.kind == LightKind::Spot {
		let (_, outer) = light.cone();
		let way = (placed.rotation * Vec3::NEG_Z).normalize_or(Vec3::NEG_Z);
		let mouth = placed.position + way * light.range;
		let across = across(camera, way);

		return vec![
			Handle {
				knob: Knob::Range,
				at: mouth,
				along: Along::Line { origin: placed.position, way },
				range: light.range,
				held: Held::Number(light.range),
			},
			Handle {
				knob: Knob::Outer,
				at: mouth + across * (light.range * outer.tan()),
				along: Along::Line { origin: mouth, way: across },
				range: light.range,
				held: Held::Number(outer),
			},
		];
	}

	// across the view rather than down one of the lamp's own axes: a sphere has
	// no axis of its own, and an axis pointing at the eye is one a pixel of
	// pointer cannot slide along at all
	let way = right(camera);

	vec![Handle {
		knob: Knob::Range,
		at: placed.position + way * light.range,
		along: Along::Line { origin: placed.position, way },
		range: light.range,
		held: Held::Number(light.range),
	}]
}

/// The handles a selected thing's own records offer.
///
/// A field says how it is drawn where it is declared, @ref
/// [`Draw`](colby_core::abi::Draw), and nothing here knows what record it
/// belongs to: the engine's `drawing` and a game's `door` are walked by one
/// loop, exactly as the inspector walks them.
///
/// @param world - what to look through
/// @param camera - the camera the picture was drawn with
/// @param pick - what is selected
pub(crate) fn noted(world: &World, camera: &Camera, pick: Pick) -> Vec<Handle> {
	let Pick::Entity(id) = pick else {
		return Vec::new();
	};

	let Some(placed) = world.entities.placed(id) else {
		return Vec::new();
	};

	let mut found = Vec::new();

	for (table, held) in world
		.entities
		.records()
		.tables()
		.iter()
		.enumerate()
	{
		for (column, row) in held.columns().iter().enumerate() {
			let value = world.entities.field(id, table, column);

			match (row.draw(), value) {
				| (Draw::Radius, Some(Value::Float(radius))) => {
					let way = right(camera);

					found.push(Handle {
						knob: Knob::Radius { table, column },
						at: placed.position + way * radius,
						along: Along::Line { origin: placed.position, way },
						range: radius,
						held: Held::Number(radius),
					});
				},
				| (Draw::Point, Some(Value::Vec3(point))) => {
					let at = placed.matrix().transform_point3(point);

					found.push(Handle {
						knob: Knob::Point { table, column },
						at,
						along: Along::Plane { normal: facing(camera) },
						range: 0.0,
						held: Held::Place(at),
					});
				},
				| _ => {},
			}
		}
	}

	found
}

/// What a selected thing's drawn record fields add to the picture: the circles
/// of a radius, and a line out to a place.
///
/// @param world - what to look through
/// @param pick - what is selected
pub(crate) fn sketch(world: &World, pick: Pick) -> Vec<(Vec3, Vec3)> {
	let Pick::Entity(id) = pick else {
		return Vec::new();
	};

	let Some(placed) = world.entities.placed(id) else {
		return Vec::new();
	};

	let mut segments = Vec::new();

	for (table, held) in world
		.entities
		.records()
		.tables()
		.iter()
		.enumerate()
	{
		for (column, row) in held.columns().iter().enumerate() {
			match (row.draw(), world.entities.field(id, table, column)) {
				| (Draw::Radius, Some(Value::Float(radius))) if radius > 0.0 =>
					for normal in [Vec3::X, Vec3::Y, Vec3::Z] {
						segments.extend(circle(placed.position, normal, radius));
					},
				| (Draw::Point, Some(Value::Vec3(point))) =>
					segments.push((placed.position, placed.matrix().transform_point3(point))),
				| _ => {},
			}
		}
	}

	segments
}

/// Which way the camera faces, in the world.
fn facing(camera: &Camera) -> Vec3 { (camera.target - camera.position).normalize_or(Vec3::NEG_Z) }

/// Which way is right on screen, in the world.
fn right(camera: &Camera) -> Vec3 {
	let forward = (camera.target - camera.position).normalize_or(Vec3::NEG_Z);

	forward.cross(camera.up).normalize_or(Vec3::X)
}

/// Which way is right on screen, in the plane a cone's mouth lies in.
fn across(camera: &Camera, way: Vec3) -> Vec3 {
	let right = right(camera);
	let flat = right - way * right.dot(way);

	flat.try_normalize()
		.unwrap_or_else(|| way.any_orthonormal_vector())
}

/// Which handle the pointer is on, if it is on one.
///
/// @param handles - what [`handles`] returned
/// @param camera - the camera the picture was drawn with
/// @param viewport - how big that picture is, in points
/// @param at - where the pointer is, from the picture's corner
pub(crate) fn grabbed(
	handles: &[Handle],
	camera: &Camera,
	viewport: Vec2,
	at: Vec2,
) -> Option<usize> {
	let view = camera.view_projection(viewport.x.max(1.0) / viewport.y.max(1.0));
	let mut nearest = HOLD;
	let mut found = None;

	for (index, handle) in handles.iter().enumerate() {
		let Some(point) = gizmo::project(view, handle.at, viewport) else {
			continue;
		};

		let away = point.distance(at);

		if away < nearest {
			nearest = away;
			found = Some(index);
		}
	}

	found
}

/// What the pointer reads on one handle: how far along its line it is.
///
/// @param handle - which handle
/// @param camera - the camera the picture was drawn with
/// @param at - where the pointer is, from the picture's corner
/// @param viewport - how big the picture is, in points
pub(crate) fn read(handle: &Handle, camera: &Camera, at: Vec2, viewport: Vec2) -> Option<Vec3> {
	let (from, ray) = crate::aim::ray(camera, at, viewport);

	match handle.along {
		| Along::Line { origin, way } =>
			gizmo::along(origin, way, from, ray).map(|far| origin + way * far),
		| Along::Plane { normal } => {
			let facing = ray.dot(normal);

			// the pointer looking along the plane never meets it, which is the
			// ring's rule for a handle seen edge on
			if facing.abs() < 1.0e-4 {
				return None;
			}

			let reached = (handle.at - from).dot(normal) / facing;

			(reached > 0.0).then(|| from + ray * reached)
		},
	}
}

/// What a drag makes of the field a handle writes.
///
/// Measured from where the drag began rather than from the frame before, the
/// rule the gizmo's arms keep: a drag that crossed a slow frame lands where one
/// that did not lands.
///
/// @param handle - what was grabbed, holding what the field was then
/// @param moved - how far the pointer has gone since, in the world
/// @param step - the grid, or nothing for none; an angle is never snapped,
/// because a step in world units says nothing about an angle
pub(crate) fn dragged(handle: &Handle, moved: Vec3, step: Option<f32>) -> Held {
	let far = match handle.along {
		| Along::Line { way, .. } => moved.dot(way),
		| Along::Plane { .. } => 0.0,
	};

	match (handle.knob, handle.held) {
		| (Knob::Range | Knob::Radius { .. }, Held::Number(held)) =>
			Held::Number(on_grid(held + far, step).max(LEAST_RANGE)),
		| (Knob::Outer, Held::Number(held)) => {
			// the pointer reads a distance out from the axis, so the angle it
			// stands for is the one that rim belongs to
			let rim = handle.range.mul_add(held.tan(), far).max(0.0);

			Held::Number(
				rim.atan2(handle.range.max(f32::EPSILON))
					.clamp(LEAST_CONE, MAX_CONE),
			)
		},
		| (Knob::Point { .. }, Held::Place(held)) =>
			Held::Place(select::snapped(held + moved, step.unwrap_or(0.0))),
		| _ => handle.held,
	}
}

/// A number put on the grid, if there is one.
fn on_grid(value: f32, step: Option<f32>) -> f32 {
	match step {
		| Some(step) if step.is_finite() && step > 0.0 => (value / step).round() * step,
		| _ => value,
	}
}

/// Writes what a handle dragged, to the lamp it belongs to and to every other
/// lamp picked.
///
/// The inspector's rule for a changed field, through the same call: the field
/// that moved and only it, in the same step back. So a handle and the panel
/// agree about what a selection of several means. @ref
/// [`select::spread_into`].
///
/// @param world - the world to write
/// @param pick - the thing the handle hangs off
/// @param others - every other entity picked
/// @param handle - which handle
/// @param value - what the field now holds
/// @return whether anything was written
pub(crate) fn write(
	world: &mut World,
	pick: Pick,
	others: &[EntityId],
	handle: &Handle,
	value: Held,
) -> bool {
	let Pick::Entity(id) = pick else {
		return false;
	};

	match (handle.knob, value) {
		| (Knob::Range | Knob::Outer, Held::Number(number)) => {
			let Some(index) = column(handle.knob) else {
				return false;
			};

			shine(world, id, others, index, number)
		},
		| (Knob::Radius { table, column }, Held::Number(number)) =>
			note(world, id, others, (table, column), &Value::Float(number)),
		| (Knob::Point { table, column }, Held::Place(place)) => {
			// what it comes to in the thing's own terms, which is where the
			// field keeps it: a door carried across a room takes its numbers
			// with it
			let Some(placed) = world.entities.placed(id) else {
				return false;
			};

			let local = placed.matrix().inverse().transform_point3(place);

			note(world, id, others, (table, column), &Value::Vec3(local))
		},
		| _ => false,
	}
}

/// Writes one field of a lamp, and of every other lamp picked.
fn shine(world: &mut World, id: EntityId, others: &[EntityId], index: usize, value: f32) -> bool {
	let Some(mut light) = world.entities.light(id).copied() else {
		return false;
	};

	let Some(field) = Light::FIELDS.get(index) else {
		return false;
	};

	let held = field.get(&light);

	if !field.set(&mut light, Value::Float(value)) {
		return false;
	}

	let now = field.get(&light);

	if now == held {
		return false;
	}

	world.entities.set_light(id, light);
	select::spread_into(
		world,
		others,
		Light::FIELDS,
		&[Edit { index, held, value: now }],
		|world, id| world.entities.light(id).copied(),
		|world, id, light| world.entities.set_light(id, light),
	);

	true
}

/// Writes one field of one record, and the same field of every other entity
/// picked. @ref [`select::spread_record`].
fn note(
	world: &mut World,
	id: EntityId,
	others: &[EntityId],
	(table, column): (usize, usize),
	value: &Value,
) -> bool {
	let Some(held) = world.entities.field(id, table, column) else {
		return false;
	};

	if held == *value {
		return false;
	}

	if !world.entities.set_field(id, table, column, value) {
		return false;
	}

	select::spread_record(world, others, table, &Edit {
		index: column,
		held,
		value: value.clone(),
	});

	true
}

/// Where a knob's field is in a light's table.
fn column(knob: Knob) -> Option<usize> {
	let name = match knob {
		| Knob::Range => "range",
		| Knob::Outer => "outer",
		| Knob::Radius { .. } | Knob::Point { .. } => return None,
	};

	Light::FIELDS
		.iter()
		.position(|field| field.name == name)
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{Body, BodyKind, Decal, Joint, MeshId, Record, Renderable, Water, mesh},
		bytemuck::{Pod, Zeroable},
	};

	use super::*;

	/// A record a game might declare, with a field drawn each of the two ways.
	#[repr(C)]
	#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
	#[bytemuck(crate = "::colby_core::bytemuck")]
	struct Door {
		reach: f32,
		target: [f32; 3],
		speed: f32,
	}

	/// The door as the game spells it.
	const DOOR: Record<Door> = Record {
		name: "door",
		help: "a thing that opens",
		rows: &[
			colby_core::row!(Float, Door, reach, "how far off it notices somebody")
				.drawn(Draw::Radius),
			colby_core::row!(Vec3, Door, target, "where it swings to").drawn(Draw::Point),
			colby_core::row!(Float, Door, speed, "how fast it swings"),
		],
		default: Door {
			reach: 2.0,
			target: [1.0, 0.0, 0.0],
			speed: 1.0,
		},
	};

	/// The size of the picture the tests project into.
	const VIEW: Vec2 = Vec2::new(1280.0, 720.0);

	/// A camera nine back along z, looking at the origin.
	fn watching() -> Camera {
		let mut camera = Camera::DEFAULT;
		camera.target = Vec3::ZERO;
		camera.position = Vec3::Z * 9.0;

		camera
	}

	/// A world with the built-in cube in it.
	fn cubed() -> World {
		let mut world = World::new();
		world.meshes.insert("meshes/cube", mesh::cube());

		world
	}

	/// An entity standing somewhere, drawing nothing.
	fn stood(world: &mut World, at: Vec3) -> EntityId {
		world.entities.spawn_at(Transform::at(at))
	}

	/// The same, drawing the cube.
	fn drawn(world: &mut World, at: Vec3) -> EntityId {
		let id = stood(world, at);
		world
			.entities
			.set_renderable(id, Renderable::new(MeshId::CUBE, Vec3::ONE));

		id
	}

	#[test]
	fn only_a_thing_that_draws_nothing_and_is_something_is_marked() {
		let mut world = cubed();
		let lamp = stood(&mut world, Vec3::ZERO);
		world
			.entities
			.set_light(lamp, Light::point(Vec3::ONE, 1.0, 4.0));

		let empty = stood(&mut world, Vec3::X);
		let crate_ = drawn(&mut world, Vec3::NEG_X);
		world
			.entities
			.set_light(crate_, Light::point(Vec3::ONE, 1.0, 4.0));

		let found = marks(&world, &watching(), VIEW);

		assert_eq!(found.len(), 1, "the lamp with no mesh, and nothing else");
		assert_eq!(found[0].pick, Pick::Entity(lamp));
		assert_eq!(found[0].kind, Kind::Lamp);
		assert!(
			!found
				.iter()
				.any(|mark| mark.pick == Pick::Entity(empty)),
			"an entity that is nothing in particular is not marked"
		);
	}

	#[test]
	fn a_mark_says_which_of_the_four_kinds_it_is() {
		let mut world = cubed();
		let lamp = stood(&mut world, Vec3::ZERO);
		world
			.entities
			.set_light(lamp, Light::point(Vec3::ONE, 1.0, 4.0));

		let spot = stood(&mut world, Vec3::X);
		world
			.entities
			.set_light(spot, Light::spot(Vec3::ONE, 1.0, 4.0, 0.0, std::f32::consts::FRAC_PI_4));

		let thrower = stood(&mut world, Vec3::Y);
		world
			.entities
			.set_emitter(thrower, Emitter::point(8.0, 1.0));

		let painter = stood(&mut world, Vec3::NEG_Y);
		world.entities.set_decal(painter, Decal::BOX);

		let kinds: Vec<Kind> = marks(&world, &watching(), VIEW)
			.into_iter()
			.map(|mark| mark.kind)
			.collect();

		assert_eq!(kinds, vec![Kind::Lamp, Kind::Spot, Kind::Thrower, Kind::Painter]);
	}

	#[test]
	fn a_spot_says_which_way_it_points_on_screen() {
		let mut world = cubed();
		let spot = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			// a quarter turn about y takes the thing's own -z to -x, which is
			// left on a screen looked at down z
			rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
			scale: Vec3::ONE,
		});
		world
			.entities
			.set_light(spot, Light::spot(Vec3::ONE, 1.0, 4.0, 0.0, std::f32::consts::FRAC_PI_4));

		let found = marks(&world, &watching(), VIEW);
		let aim = found[0].aim.expect("a spot points somewhere");

		assert!(aim.x < -0.9, "it points left across the picture: {aim}");
	}

	#[test]
	fn a_hidden_thing_is_not_marked() {
		let mut world = cubed();
		let lamp = stood(&mut world, Vec3::ZERO);
		world
			.entities
			.set_light(lamp, Light::point(Vec3::ONE, 1.0, 4.0));
		assert!(world.entities.set_hidden(lamp, true));

		assert!(marks(&world, &watching(), VIEW).is_empty());
	}

	#[test]
	fn the_pointer_lands_on_the_nearest_mark_and_on_nothing_out_beyond_them() {
		let mut world = cubed();
		let lamp = stood(&mut world, Vec3::ZERO);
		world
			.entities
			.set_light(lamp, Light::point(Vec3::ONE, 1.0, 4.0));

		let camera = watching();
		let at = marks(&world, &camera, VIEW)[0].at;

		assert_eq!(under(&world, &camera, VIEW, at), Some(Pick::Entity(lamp)), "on it");
		assert_eq!(
			under(&world, &camera, VIEW, at + Vec2::new(REACH * 0.5, 0.0)),
			Some(Pick::Entity(lamp)),
			"within reach of it"
		);
		assert_eq!(
			under(&world, &camera, VIEW, at + Vec2::new(REACH * 3.0, 0.0)),
			None,
			"and not out beyond that"
		);
	}

	#[test]
	fn a_mark_is_found_where_a_mesh_around_the_camera_would_take_every_click() {
		let mut world = cubed();
		let lamp = stood(&mut world, Vec3::ZERO);
		world
			.entities
			.set_light(lamp, Light::point(Vec3::ONE, 1.0, 4.0));

		// the room the camera stands in: `aim::under` answers this at no
		// distance for every ray, which is why a mark is not ordered by depth
		let mut room = Transform::at(Vec3::ZERO);
		room.scale = Vec3::splat(40.0);
		let big = world.entities.spawn_at(room);
		world
			.entities
			.set_renderable(big, Renderable::new(MeshId::CUBE, Vec3::ONE));

		let camera = watching();
		let at = marks(&world, &camera, VIEW)[0].at;

		assert_eq!(under(&world, &camera, VIEW, at), Some(Pick::Entity(lamp)));
		assert_eq!(
			crate::aim::under(&world, camera.position, Vec3::NEG_Z),
			Pick::Entity(big),
			"and the mesh is what the ray finds, at no distance at all"
		);
	}

	#[test]
	fn a_joint_is_a_line_between_its_anchors_and_a_click_finds_it() {
		let mut world = cubed();
		let first = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::UNIT,
			Transform::at(Vec3::NEG_X * 2.0),
		));
		let second = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::UNIT,
			Transform::at(Vec3::X * 2.0),
		));
		let tie = world.joints.spawn(Joint::new(
			JointKind::Rope,
			first,
			second,
			(Vec3::ZERO, Vec3::ZERO),
		));

		let found = outlines(&world);
		let line = found
			.iter()
			.find(|outline| outline.pick == Pick::Joint(tie))
			.expect("the tie is drawn");

		assert_eq!(line.drawn, Drawn::Tie);
		assert_eq!(line.segments.len(), 1, "one line, anchor to anchor");
		assert!(
			line.segments[0]
				.0
				.abs_diff_eq(Vec3::NEG_X * 2.0, 1.0e-4)
		);
		assert!(
			line.segments[0]
				.1
				.abs_diff_eq(Vec3::X * 2.0, 1.0e-4)
		);

		let camera = watching();
		let view = camera.view_projection(VIEW.x / VIEW.y);
		let at = gizmo::project(view, Vec3::X, VIEW).expect("in front");

		assert_eq!(under(&world, &camera, VIEW, at), Some(Pick::Joint(tie)), "the line is hit");
	}

	#[test]
	fn a_body_is_outlined_when_nothing_draws_where_it_stands() {
		let mut world = cubed();
		let bare = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::UNIT,
			Transform::at(Vec3::ZERO),
		));
		let seen = drawn(&mut world, Vec3::X * 4.0);
		world.bodies.spawn(
			Body::new(BodyKind::Static, Shape::UNIT, Transform::at(Vec3::X * 4.0)).driving(seen),
		);

		let found = outlines(&world);

		assert_eq!(found.len(), 1, "the bare body, and not the one under a crate");
		assert_eq!(found[0].pick, Pick::Body(bare));
		assert_eq!(found[0].drawn, Drawn::Solid);
		assert_eq!(found[0].segments.len(), 12, "a box is twelve edges");
	}

	#[test]
	fn a_body_under_a_hidden_thing_is_not_outlined() {
		let mut world = cubed();
		let trigger = stood(&mut world, Vec3::ZERO);
		let body = Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY).driving(trigger);
		world.bodies.spawn(body);
		assert!(world.entities.set_hidden(trigger, true));

		assert!(
			outlines(&world).is_empty(),
			"what is hidden is not drawn, the way it is not picked"
		);
	}

	#[test]
	fn a_round_body_is_outlined_as_three_circles() {
		let mut world = cubed();
		let mut shape = Shape::UNIT;
		shape.kind = ShapeKind::Sphere;
		shape.radius = 2.0;
		world
			.bodies
			.spawn(Body::new(BodyKind::Static, shape, Transform::IDENTITY));

		let found = outlines(&world);

		assert_eq!(found[0].segments.len(), 3 * CIRCLE_STEPS, "three circles of straight pieces");
		let widest = found[0]
			.segments
			.iter()
			.map(|(from, _)| from.length())
			.fold(0.0_f32, f32::max);

		assert!((widest - 2.0).abs() < 1.0e-3, "as wide as the shape is: {widest}");
	}

	#[test]
	fn a_cone_handle_lies_across_the_axis_even_when_the_axis_points_across_the_view() {
		let mut world = cubed();
		let spot = stood(&mut world, Vec3::ZERO);
		let angle = std::f32::consts::FRAC_PI_4;
		// turned so that the cone points along the camera's own right, where
		// the right of the screen is a direction a rim cannot be measured along
		let mut turned = Transform::at(Vec3::ZERO);
		turned.rotation = Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2);
		assert!(world.entities.set_placed(spot, turned));
		world
			.entities
			.set_light(spot, Light::spot(Vec3::ONE, 1.0, 4.0, 0.0, angle));

		let found = handles(&world, &watching(), Pick::Entity(spot));
		let Along::Line { way, .. } = found[1].along else {
			panic!("a cone's edge is dragged along a line");
		};

		assert!(
			way.dot(Vec3::X).abs() < 1.0e-3,
			"the rim is measured across the cone's own axis, not along it: {way}"
		);
		assert!(found[1].at.distance(found[0].at) > 1.0, "so the handle is out on the rim");
	}

	#[test]
	fn a_place_the_pointer_cannot_reach_is_not_read() {
		let camera = watching();
		let handle = Handle {
			knob: Knob::Point { table: 0, column: 0 },
			// behind the camera, where a plane is only met by a ray going the
			// other way
			at: Vec3::Z * 40.0,
			along: Along::Plane { normal: Vec3::NEG_Z },
			range: 0.0,
			held: Held::Place(Vec3::Z * 40.0),
		};

		assert_eq!(read(&handle, &camera, VIEW * 0.5, VIEW), None, "it is behind the eye");
	}

	#[test]
	fn a_body_under_an_entity_that_draws_nothing_is_outlined_and_picks_the_entity() {
		let mut world = cubed();
		let trigger = stood(&mut world, Vec3::ZERO);
		let body = Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY).driving(trigger);
		world.bodies.spawn(body);

		let found = outlines(&world);

		assert_eq!(found.len(), 1);
		assert_eq!(found[0].pick, Pick::Entity(trigger), "the thing a person means");
	}

	#[test]
	fn a_pool_says_where_its_surface_is() {
		let mut world = cubed();
		let mut standing = Transform::at(Vec3::ZERO);
		standing.scale = Vec3::new(4.0, 2.0, 4.0);
		let mut body = Body::new(BodyKind::Static, Shape::UNIT, standing);
		body.water = Water::pool();
		world.bodies.spawn(body);

		let found = outlines(&world);

		assert_eq!(found[0].drawn, Drawn::Fluid);
		assert_eq!(found[0].segments.len(), 12 + 4, "the box and the waterline");
		let top = found[0].segments[12].0.y;
		assert!((top - 1.0).abs() < 1.0e-4, "the top of the box it is filled to: {top}");
	}

	#[test]
	fn a_mesh_shaped_body_is_not_outlined() {
		let mut world = cubed();
		let mut shape = Shape::UNIT;
		shape.kind = ShapeKind::Mesh;
		world
			.bodies
			.spawn(Body::new(BodyKind::Static, shape, Transform::IDENTITY));

		assert!(outlines(&world).is_empty(), "a soup is not an outline");
	}

	#[test]
	fn a_hinge_shows_its_anchors_and_the_axis_it_turns_about() {
		let mut world = cubed();
		let first =
			world
				.bodies
				.spawn(Body::new(BodyKind::Static, Shape::UNIT, Transform::IDENTITY));
		let second = world.bodies.spawn(Body::new(
			BodyKind::Static,
			Shape::UNIT,
			Transform::at(Vec3::X * 2.0),
		));
		let tie =
			world
				.joints
				.spawn(Joint::axis(first, second, (Vec3::ZERO, Vec3::ZERO), Vec3::Y));

		let segments = detail(&world, Pick::Joint(tie));

		assert_eq!(segments.len(), 3 + 3 + 5, "a cross at each anchor and an arrow for the axis");
		let axis = segments[6];
		assert!(
			(axis.1 - axis.0)
				.normalize()
				.abs_diff_eq(Vec3::Y, 1.0e-4),
			"the axis goes the way the joint says"
		);
		assert!(
			detail(&world, Pick::Entity(EntityId::NONE)).is_empty(),
			"and nothing else has one"
		);
	}

	#[test]
	fn a_lamp_has_one_handle_on_the_rim_across_the_view() {
		let mut world = cubed();
		let lamp = stood(&mut world, Vec3::ZERO);
		world
			.entities
			.set_light(lamp, Light::point(Vec3::ONE, 1.0, 4.0));

		let camera = watching();
		let found = handles(&world, &camera, Pick::Entity(lamp));

		assert_eq!(found.len(), 1);
		assert_eq!(found[0].knob, Knob::Range);
		assert!(
			found[0].at.abs_diff_eq(Vec3::X * 4.0, 1.0e-3),
			"four out to the right of the picture: {}",
			found[0].at
		);
	}

	#[test]
	fn a_spot_has_a_reach_handle_down_its_axis_and_a_cone_handle_on_the_rim() {
		let mut world = cubed();
		let spot = stood(&mut world, Vec3::ZERO);
		let angle = std::f32::consts::FRAC_PI_4;
		world
			.entities
			.set_light(spot, Light::spot(Vec3::ONE, 1.0, 5.0, 0.0, angle));

		let found = handles(&world, &watching(), Pick::Entity(spot));

		assert_eq!(found.len(), 2);
		assert!(
			found[0].at.abs_diff_eq(Vec3::NEG_Z * 5.0, 1.0e-3),
			"the mouth, down the thing's own -z: {}",
			found[0].at
		);
		assert_eq!(found[1].knob, Knob::Outer);
		assert!(
			found[1]
				.at
				.abs_diff_eq(Vec3::new(5.0 * angle.tan(), 0.0, -5.0), 1.0e-3),
			"and the rim, as wide as the angle makes it: {}",
			found[1].at
		);
	}

	#[test]
	fn a_lamp_that_throws_nothing_has_no_handle() {
		let mut world = cubed();
		let lamp = stood(&mut world, Vec3::ZERO);

		assert!(handles(&world, &watching(), Pick::Entity(lamp)).is_empty(), "not a lamp at all");

		world
			.entities
			.set_light(lamp, Light::point(Vec3::ONE, 1.0, 0.0));

		assert!(
			handles(&world, &watching(), Pick::Entity(lamp)).is_empty(),
			"and one that reaches nowhere throws nothing"
		);
	}

	/// A handle to drag along a line, with a field at `held`.
	fn holding(knob: Knob, held: f32, range: f32) -> Handle {
		Handle {
			knob,
			at: Vec3::ZERO,
			along: Along::Line { origin: Vec3::ZERO, way: Vec3::X },
			range,
			held: Held::Number(held),
		}
	}

	/// What a drag of so far along that line makes of the number.
	fn number(handle: &Handle, moved: f32, step: Option<f32>) -> f32 {
		match dragged(handle, Vec3::X * moved, step) {
			| Held::Number(value) => value,
			| Held::Place(place) =>
				panic!("a number was asked for and a place came back: {place}"),
		}
	}

	#[test]
	fn a_reach_grows_by_how_far_the_pointer_went_and_never_goes_to_nothing() {
		let handle = holding(Knob::Range, 4.0, 4.0);

		assert!((number(&handle, 2.0, None) - 6.0).abs() < 1.0e-4, "out two is six");
		assert!((number(&handle, -1.5, None) - 2.5).abs() < 1.0e-4, "and back one and a half");
		assert!(
			number(&handle, -400.0, None) >= LEAST_RANGE,
			"a drag through nought stops short of it, or the lamp it is holding would go out"
		);
	}

	#[test]
	fn a_reach_lands_on_the_grid_and_an_angle_never_does() {
		let reach = holding(Knob::Range, 4.0, 4.0);

		assert!(
			(number(&reach, 0.2, Some(0.5)) - 4.0).abs() < 1.0e-4,
			"a fifth out is still four on a half-unit grid"
		);
		assert!(
			(number(&reach, 0.4, Some(0.5)) - 4.5).abs() < 1.0e-4,
			"and two fifths is four and a half"
		);

		let cone = holding(Knob::Outer, std::f32::consts::FRAC_PI_4, 4.0);
		let snapped = number(&cone, 0.3, Some(0.5));
		let free = number(&cone, 0.3, None);

		assert!(
			(snapped - free).abs() < 1.0e-6,
			"an angle is a different unit and no grid in world units touches it"
		);
	}

	#[test]
	fn a_cone_opens_by_the_rim_the_pointer_dragged_and_stops_short_of_a_hemisphere() {
		let range = 4.0;
		let held = std::f32::consts::FRAC_PI_4;
		let handle = holding(Knob::Outer, held, range);

		let wider = number(&handle, 1.0, None);
		let expected = range.mul_add(held.tan(), 1.0).atan2(range);

		assert!((wider - expected).abs() < 1.0e-5, "the angle the new rim stands for");
		assert!(wider > held, "and it opened");
		assert!(number(&handle, 1.0e6, None) <= MAX_CONE, "a cone stops short of a hemisphere");
		assert!(number(&handle, -1.0e6, None) >= LEAST_CONE, "and short of nothing");
	}

	#[test]
	fn a_drag_is_written_to_the_lamp_and_to_every_other_lamp_picked() {
		let mut world = cubed();
		let first = stood(&mut world, Vec3::ZERO);
		let second = stood(&mut world, Vec3::X * 3.0);

		for id in [first, second] {
			world
				.entities
				.set_light(id, Light::point(Vec3::ONE, 1.0, 4.0));
		}

		let handle = holding(Knob::Range, 4.0, 4.0);

		assert!(write(&mut world, Pick::Entity(first), &[second], &handle, Held::Number(7.0)));

		for id in [first, second] {
			let range = world
				.entities
				.light(id)
				.map(|light| light.range)
				.unwrap_or_default();

			assert!((range - 7.0).abs() < 1.0e-4, "both reach seven: {range}");
		}
	}

	#[test]
	fn writing_what_the_field_already_holds_writes_nothing() {
		let mut world = cubed();
		let lamp = stood(&mut world, Vec3::ZERO);
		world
			.entities
			.set_light(lamp, Light::point(Vec3::ONE, 1.0, 4.0));

		let handle = holding(Knob::Range, 4.0, 4.0);

		assert!(
			!write(&mut world, Pick::Entity(lamp), &[], &handle, Held::Number(4.0)),
			"nothing moved"
		);
	}

	#[test]
	fn a_plume_reaches_as_far_as_a_particle_gets_in_its_life() {
		let mut emitter = Emitter::point(8.0, 2.0);
		emitter.speed = 3.0;
		emitter.drag = 0.0;

		assert!((reach(&emitter) - 6.0).abs() < 1.0e-4, "no drag is speed all the way");

		emitter.drag = 0.5;
		let dragged = reach(&emitter);

		assert!(dragged > 0.0 && dragged < 6.0, "a drag takes some of it off: {dragged}");

		emitter.drag = 1.0;
		assert!(reach(&emitter).abs() < 1.0e-6, "and one keeps nothing at all");

		emitter.drag = 0.0;
		emitter.speed = -4.0;
		assert!(reach(&emitter).abs() < 1.0e-6, "a speed nobody can throw at reaches nowhere");
	}

	#[test]
	fn an_arrow_is_a_shaft_and_four_barbs_and_nothing_between_one_place_and_itself() {
		let segments = arrow(Vec3::ZERO, Vec3::X * 2.0);

		assert_eq!(segments.len(), 5);
		assert_eq!(segments[0], (Vec3::ZERO, Vec3::X * 2.0), "the shaft first");

		for (from, to) in segments.iter().skip(1) {
			assert!(from.abs_diff_eq(Vec3::X * 2.0, 1.0e-5), "every barb starts at the point");
			assert!(to.x < 2.0, "and goes back down the shaft: {to}");
		}

		assert!(arrow(Vec3::Y, Vec3::Y).is_empty(), "and a place to itself is no arrow");
	}

	#[test]
	fn the_two_knobs_of_a_lamp_name_fields_of_its_table_and_a_record_s_do_not() {
		for knob in [Knob::Range, Knob::Outer] {
			assert!(column(knob).is_some(), "{} is in the light's table", knob.word());
		}

		assert_eq!(
			column(Knob::Radius { table: 0, column: 0 }),
			None,
			"a record's field is reached by where it sits rather than by a name in a table"
		);
	}

	/// A world with a door declared, and an entity standing somewhere with one.
	fn doored(at: Transform) -> (World, EntityId) {
		let mut world = cubed();
		world.entities.declare(&DOOR).expect("a door");
		let id = world.entities.spawn_at(at);

		(world, id)
	}

	#[test]
	fn a_record_that_says_a_radius_and_a_place_offers_a_handle_for_each() {
		let (world, id) = doored(Transform::at(Vec3::new(2.0, 0.0, 0.0)));
		let camera = watching();
		let found = noted(&world, &camera, Pick::Entity(id));

		assert_eq!(found.len(), 2, "the two fields the record says are drawn");
		assert!(
			found[0]
				.at
				.abs_diff_eq(Vec3::new(4.0, 0.0, 0.0), 1.0e-3),
			"the radius out across the view: {}",
			found[0].at
		);
		assert!(
			matches!(found[1].along, Along::Plane { .. }),
			"and a place is dragged on a plane facing the eye"
		);
		assert!(
			found[1]
				.at
				.abs_diff_eq(Vec3::new(3.0, 0.0, 0.0), 1.0e-3),
			"where the place is in the world: {}",
			found[1].at
		);
	}

	#[test]
	fn a_record_that_says_nothing_offers_no_handle() {
		let mut world = cubed();
		let id = world.entities.spawn_at(Transform::IDENTITY);

		// the engine's own records are declared by every world and say nothing
		// about being drawn
		assert!(noted(&world, &watching(), Pick::Entity(id)).is_empty());
		assert!(sketch(&world, Pick::Entity(id)).is_empty());
	}

	#[test]
	fn what_a_record_draws_is_three_circles_and_a_line_out_to_the_place() {
		let (world, id) = doored(Transform::at(Vec3::ZERO));
		let segments = sketch(&world, Pick::Entity(id));

		assert_eq!(segments.len(), 3 * CIRCLE_STEPS + 1, "three circles and the line");
		let line = segments[3 * CIRCLE_STEPS];
		assert!(line.0.abs_diff_eq(Vec3::ZERO, 1.0e-4), "the line starts at the thing");
		assert!(line.1.abs_diff_eq(Vec3::X, 1.0e-4), "and ends at the place: {}", line.1);
	}

	#[test]
	fn a_place_dragged_is_written_in_the_thing_s_own_terms_and_to_everything_picked() {
		let mut turned = Transform::at(Vec3::new(0.0, 0.0, 0.0));
		// a quarter turn about y sends the thing's own x down the world's -z
		turned.rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
		let (mut world, id) = doored(turned);
		let other = world.entities.spawn_at(Transform::IDENTITY);
		let handle = noted(&world, &watching(), Pick::Entity(id))
			.into_iter()
			.find(|handle| matches!(handle.knob, Knob::Point { .. }))
			.expect("a place is drawn");

		let moved = Vec3::new(0.0, 2.0, 0.0);
		let value = dragged(&handle, moved, None);

		assert!(write(&mut world, Pick::Entity(id), &[other], &handle, value));

		let Some(Value::Vec3(place)) = world.entities.field(id, 2, 1) else {
			panic!("the door's place is a field of three numbers");
		};

		assert!(
			place.abs_diff_eq(Vec3::new(1.0, 2.0, 0.0), 1.0e-3),
			"up two in the world is up two in a thing turned about y: {place}"
		);

		let Some(Value::Vec3(theirs)) = world.entities.field(other, 2, 1) else {
			panic!("every entity carries every record");
		};

		assert!(theirs.abs_diff_eq(place, 1.0e-4), "and everything picked took it: {theirs}");
	}

	#[test]
	fn a_place_dragged_lands_on_the_grid_the_way_a_move_does() {
		let (world, id) = doored(Transform::at(Vec3::ZERO));
		let handle = noted(&world, &watching(), Pick::Entity(id))
			.into_iter()
			.find(|handle| matches!(handle.knob, Knob::Point { .. }))
			.expect("a place is drawn");

		let Held::Place(place) = dragged(&handle, Vec3::new(0.0, 0.7, 0.0), Some(0.5)) else {
			panic!("a place comes back a place");
		};

		assert!(
			place.abs_diff_eq(Vec3::new(1.0, 0.5, 0.0), 1.0e-4),
			"a place lands on a line of the grid, as a move does: {place}"
		);
	}

	#[test]
	fn writing_a_record_field_what_it_already_holds_writes_nothing() {
		let (mut world, id) = doored(Transform::at(Vec3::ZERO));
		let other = world.entities.spawn_at(Transform::IDENTITY);
		let handle = noted(&world, &watching(), Pick::Entity(id))
			.into_iter()
			.find(|handle| matches!(handle.knob, Knob::Radius { .. }))
			.expect("a radius is drawn");

		assert!(
			!write(&mut world, Pick::Entity(id), &[other], &handle, Held::Number(2.0)),
			"nothing moved"
		);
		assert_eq!(
			world.entities.field(other, 2, 0),
			Some(Value::Float(2.0)),
			"so nothing was spread either"
		);
	}

	#[test]
	fn a_radius_dragged_grows_the_field_and_never_takes_it_to_nothing() {
		let (mut world, id) = doored(Transform::at(Vec3::ZERO));
		let handle = noted(&world, &watching(), Pick::Entity(id))
			.into_iter()
			.find(|handle| matches!(handle.knob, Knob::Radius { .. }))
			.expect("a radius is drawn");

		let value = dragged(&handle, Vec3::X * 3.0, None);

		assert!(write(&mut world, Pick::Entity(id), &[], &handle, value));
		assert_eq!(world.entities.field(id, 2, 0), Some(Value::Float(5.0)));

		let value = dragged(&handle, Vec3::X * -100.0, None);

		assert!(
			matches!(value, Held::Number(number) if number >= LEAST_RANGE),
			"a drag through nought stops short of it: {value:?}"
		);
	}
}
