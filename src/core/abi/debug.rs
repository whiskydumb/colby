//! Lines, shapes and words drawn over the world so that something invisible can
//! be looked at.
//!
//! A table on [`World`](super::World), plain data in the host, exactly like the
//! bodies and the interface: gameplay lives in a library that is swapped out
//! mid-frame, so anything it wants drawn has to be written somewhere that
//! survives the swap. Nothing here knows what a pipeline is - the renderer
//! reads this table the way it reads the entity table.
//!
//! **There is one primitive, and it is a segment.** A box is twelve of them, a
//! ball is three circles, a point is three crossing strokes, an arrow is a
//! shaft and four barbs. That keeps the whole of the drawing side to one
//! pipeline and one buffer, and it means a shape nobody thought of is a
//! function here rather than a change to the renderer. The one thing that
//! cannot be a segment is text, which is why [`Label`] is the second and last
//! kind.
//!
//! **A lifetime is an absolute moment, not a countdown.** Every entry carries
//! the simulated time it stops being drawn at, and [`Debug::begin_step`] drops
//! whatever is past. Transient geometry - the overwhelming majority, submitted
//! afresh every step - expires at exactly the moment it was submitted, so it
//! survives every frame drawn before the next step and is gone at the top of
//! it. That ordering is the whole reason the sweep is at the *start* of a step
//! rather than the end like the input edges: several frames are drawn between
//! two steps, and clearing at the end would erase what those frames exist to
//! show.

use crate::glam::{Quat, Vec3};

/// How many segments can be waiting to be drawn at once.
///
/// Bounded for the reason the body table is bounded: a game that draws in a
/// loop should lose lines and say so rather than run out of memory. One ball is
/// seventy-two of these, so this is about "every body in a full world,
/// outlined" with room over.
pub const MAX_LINES: usize = 65_536;

/// How many words can be waiting at once.
pub const MAX_LABELS: usize = 1024;

/// The longest a label may be, in characters.
///
/// Anything past this is cut. A label is a readout, and a readout that does not
/// fit on screen was not one.
pub const MAX_LABEL_CHARS: usize = 128;

/// How many segments a circle is drawn with.
///
/// Twenty-four is round enough to read as a circle at any size a debug view is
/// looked at, and three of them is a ball for seventy-two lines.
const CIRCLE_SEGMENTS: usize = 24;

/// How long an arrow's barbs are, as a share of the shaft.
const BARB: f32 = 0.15;

/// How far out an arrow's barbs spread, as a share of their own length.
const BARB_SPREAD: f32 = 0.5;

/// Linear white.
pub const WHITE: Vec3 = Vec3::new(1.0, 1.0, 1.0);

/// Linear red.
pub const RED: Vec3 = Vec3::new(1.0, 0.1, 0.1);

/// Linear green.
pub const GREEN: Vec3 = Vec3::new(0.1, 1.0, 0.1);

/// Linear blue.
pub const BLUE: Vec3 = Vec3::new(0.2, 0.4, 1.0);

/// Linear yellow.
pub const YELLOW: Vec3 = Vec3::new(1.0, 0.9, 0.1);

/// Linear cyan.
pub const CYAN: Vec3 = Vec3::new(0.1, 0.9, 1.0);

/// Linear magenta.
pub const MAGENTA: Vec3 = Vec3::new(1.0, 0.2, 0.8);

/// Linear orange.
pub const ORANGE: Vec3 = Vec3::new(1.0, 0.5, 0.05);

/// A middle gray.
pub const GRAY: Vec3 = Vec3::new(0.55, 0.55, 0.55);

/// One segment waiting to be drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
	/// One end, in world space.
	pub from: Vec3,

	/// The other.
	pub to: Vec3,

	/// What color to draw it, linear RGB like everything else the renderer
	/// takes.
	pub color: Vec3,

	/// The simulated time it stops being drawn at.
	///
	/// Compared against [`World::time`](super::World::time), so a value equal
	/// to the moment it was submitted means "this step only".
	pub expires: f32,

	/// Whether it draws through whatever is in front of it.
	///
	/// Off by default, because a line that ignores the world tells you nothing
	/// about where it is. On for the things that begin *inside* geometry - a
	/// contact normal starts at the surface it was found on and would otherwise
	/// be half hidden by the very body that produced it.
	pub on_top: bool,
}

/// Words drawn at a point in the world.
///
/// Not a segment, which is why it is a kind of its own. Drawn screen-aligned at
/// a constant size and never occluded: a label that shrank with distance or
/// disappeared behind a wall would fail at the one job it has.
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
	/// Where it is anchored, in world space.
	pub at: Vec3,

	/// What it says. At most [`MAX_LABEL_CHARS`] characters.
	pub text: String,

	/// What color to draw it, linear RGB.
	pub color: Vec3,

	/// The simulated time it stops being drawn at. @ref [`Line::expires`].
	pub expires: f32,
}

/// Everything waiting to be drawn over the world.
///
/// Reached as `world.debug`, and written by anyone who has the world: the
/// solver outlines its own bodies, the game marks up whatever it is debugging,
/// and the host clears the lot on request.
#[derive(Clone, Debug, Default)]
pub struct Debug {
	lines: Vec<Line>,
	labels: Vec<Label>,

	/// The simulated time the step being written now ends at.
	///
	/// Stamped onto everything submitted, so that a caller holding only
	/// `&mut Debug` can still say "for two seconds" without being handed the
	/// clock. Written by [`begin_step`](Self::begin_step).
	now: f32,

	/// How much this step asked for and did not get.
	dropped: u32,
}

impl Debug {
	/// A table with nothing in it.
	#[must_use]
	pub const fn new() -> Self {
		Self {
			lines: Vec::new(),
			labels: Vec::new(),
			now: 0.0,
			dropped: 0,
		}
	}

	/// Drops whatever has expired and stamps the moment the new step ends at.
	///
	/// The host's, called at the top of every step. @ref the module docs for
	/// why this is the top and not the bottom.
	///
	/// @param time - the simulated time this step ends at
	pub fn begin_step(&mut self, time: f32) {
		self.lines.retain(|line| line.expires > time);
		self.labels.retain(|label| label.expires > time);
		self.now = time;
		self.dropped = 0;
	}

	/// Throws everything away, lasting geometry included.
	pub fn clear(&mut self) {
		self.lines.clear();
		self.labels.clear();
		self.dropped = 0;
	}

	/// A pen that draws for this step only, behind whatever is in front.
	#[must_use]
	pub fn pen(&mut self) -> Pen<'_> {
		Pen {
			expires: self.now,
			on_top: false,
			debug: self,
		}
	}

	/// A pen whose marks stay for a while.
	///
	/// @param seconds - how long, from the end of this step. Negative is zero.
	#[must_use]
	pub fn lasting(&mut self, seconds: f32) -> Pen<'_> { self.pen().lasting(seconds) }

	/// A pen whose marks draw through whatever is in front of them.
	#[must_use]
	pub fn on_top(&mut self) -> Pen<'_> { self.pen().on_top() }

	/// Draws a segment for this step.
	pub fn line(&mut self, from: Vec3, to: Vec3, color: Vec3) {
		self.pen().line(from, to, color);
	}

	/// Draws an axis-aligned box for this step.
	pub fn bounds(&mut self, min: Vec3, max: Vec3, color: Vec3) {
		self.pen().bounds(min, max, color);
	}

	/// Draws a box that may be turned, for this step.
	pub fn cuboid(&mut self, center: Vec3, extents: Vec3, rotation: Quat, color: Vec3) {
		self.pen()
			.cuboid(center, extents, rotation, color);
	}

	/// Draws a ball as three circles, for this step.
	pub fn ball(&mut self, center: Vec3, radius: f32, color: Vec3) {
		self.pen().ball(center, radius, color);
	}

	/// Draws a circle about an axis, for this step.
	pub fn circle(&mut self, center: Vec3, axis: Vec3, radius: f32, color: Vec3) {
		self.pen().circle(center, axis, radius, color);
	}

	/// Draws a small three-armed cross, for this step.
	pub fn point(&mut self, at: Vec3, size: f32, color: Vec3) {
		self.pen().point(at, size, color);
	}

	/// Draws an arrow, for this step.
	pub fn arrow(&mut self, from: Vec3, to: Vec3, color: Vec3) {
		self.pen().arrow(from, to, color);
	}

	/// Draws three colored axes, for this step.
	pub fn axes(&mut self, at: Vec3, rotation: Quat, size: f32) {
		self.pen().axes(at, rotation, size);
	}

	/// Draws words at a point in the world, for this step.
	pub fn label(&mut self, at: Vec3, text: &str, color: Vec3) {
		self.pen().label(at, text, color);
	}

	/// Every segment waiting to be drawn.
	#[must_use]
	pub fn lines(&self) -> &[Line] { &self.lines }

	/// Every label waiting to be drawn.
	#[must_use]
	pub fn labels(&self) -> &[Label] { &self.labels }

	/// How many entries were asked for this step and refused.
	///
	/// Not zero means the picture is missing pieces, which is worth knowing
	/// before concluding that something is not being drawn at all.
	#[must_use]
	pub const fn dropped(&self) -> u32 { self.dropped }

	/// Whether there is nothing to draw.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.lines.is_empty() && self.labels.is_empty() }

	/// The simulated time entries submitted now are stamped against.
	#[must_use]
	pub const fn now(&self) -> f32 { self.now }

	/// Takes one segment, or counts it as lost.
	fn push_line(&mut self, line: Line) {
		if self.lines.len() >= MAX_LINES {
			self.dropped = self.dropped.saturating_add(1);

			return;
		}

		self.lines.push(line);
	}

	/// Takes one label, or counts it as lost.
	fn push_label(&mut self, label: Label) {
		if self.labels.len() >= MAX_LABELS {
			self.dropped = self.dropped.saturating_add(1);

			return;
		}

		self.labels.push(label);
	}
}

/// A [`Debug`] with a lifetime and a depth rule already chosen.
///
/// Every shape is written once, here, and the two spellings on [`Debug`] itself
/// are one-line forwarders through a default pen. That is what keeps
/// "for this step" - which is almost every call - free of an argument that
/// would be zero every time, without the geometry existing twice.
pub struct Pen<'a> {
	debug: &'a mut Debug,
	expires: f32,
	on_top: bool,
}

impl Pen<'_> {
	/// Makes these marks last.
	///
	/// @param seconds - how long, from the end of this step. Negative is zero.
	#[must_use]
	pub fn lasting(mut self, seconds: f32) -> Self {
		self.expires = self.debug.now + seconds.max(0.0);

		self
	}

	/// Makes these marks draw through whatever is in front of them.
	#[must_use]
	pub fn on_top(mut self) -> Self {
		self.on_top = true;

		self
	}

	/// Draws a segment.
	pub fn line(&mut self, from: Vec3, to: Vec3, color: Vec3) {
		self.debug.push_line(Line {
			from,
			to,
			color,
			expires: self.expires,
			on_top: self.on_top,
		});
	}

	/// Draws the twelve edges of an axis-aligned box.
	///
	/// @param min - the low corner
	/// @param max - the high corner
	pub fn bounds(&mut self, min: Vec3, max: Vec3, color: Vec3) {
		let center = (min + max) * 0.5;
		let extents = (max - min) * 0.5;

		self.cuboid(center, extents, Quat::IDENTITY, color);
	}

	/// Draws the twelve edges of a box that may be turned.
	///
	/// @param center - where the middle of it is, in world space
	/// @param extents - **half**-extents, so a unit cube is `Vec3::splat(0.5)`
	/// @param rotation - how it is turned
	pub fn cuboid(&mut self, center: Vec3, extents: Vec3, rotation: Quat, color: Vec3) {
		let corner = |x: f32, y: f32, z: f32| center + rotation * (extents * Vec3::new(x, y, z));

		// the eight corners in the order the edge table below indexes them: the
		// low face counter-clockwise, then the high face above it.
		let corners = [
			corner(-1.0, -1.0, -1.0),
			corner(1.0, -1.0, -1.0),
			corner(1.0, -1.0, 1.0),
			corner(-1.0, -1.0, 1.0),
			corner(-1.0, 1.0, -1.0),
			corner(1.0, 1.0, -1.0),
			corner(1.0, 1.0, 1.0),
			corner(-1.0, 1.0, 1.0),
		];

		for [first, second] in BOX_EDGES {
			self.line(corners[first], corners[second], color);
		}
	}

	/// Draws a ball as three circles, one about each axis.
	///
	/// Three rather than a wireframe sphere: three rings read as a ball from
	/// every angle and cost seventy-two lines, where a latitude-longitude cage
	/// costs hundreds and reads as a mess.
	pub fn ball(&mut self, center: Vec3, radius: f32, color: Vec3) {
		self.circle(center, Vec3::X, radius, color);
		self.circle(center, Vec3::Y, radius, color);
		self.circle(center, Vec3::Z, radius, color);
	}

	/// Draws a circle lying in the plane an axis is normal to.
	///
	/// @param center - the middle of it
	/// @param axis - what it lies across; need not be normalized
	/// @param radius - how far out it goes
	pub fn circle(&mut self, center: Vec3, axis: Vec3, radius: f32, color: Vec3) {
		if radius <= 0.0 {
			return;
		}

		let normal = axis.normalize_or(Vec3::Y);
		let across = perpendicular(normal);
		let right = across * radius;
		let up = normal.cross(across) * radius;

		let mut previous = center + right;
		for step in 1..=CIRCLE_SEGMENTS {
			let angle = turn(step);
			let point = center + right * angle.cos() + up * angle.sin();

			self.line(previous, point, color);
			previous = point;
		}
	}

	/// Draws a small cross, one stroke along each axis.
	///
	/// What a contact point or a joint anchor is marked with: a position with
	/// no size of its own still has to be visible from every direction.
	///
	/// @param size - how far each arm reaches from the middle
	pub fn point(&mut self, at: Vec3, size: f32, color: Vec3) {
		let arm = size.max(0.0);

		self.line(at - Vec3::X * arm, at + Vec3::X * arm, color);
		self.line(at - Vec3::Y * arm, at + Vec3::Y * arm, color);
		self.line(at - Vec3::Z * arm, at + Vec3::Z * arm, color);
	}

	/// Draws a shaft with a head on the far end.
	///
	/// The head is four barbs rather than a cone, for the reason everything
	/// here is segments: it costs four lines and reads correctly from any
	/// angle.
	pub fn arrow(&mut self, from: Vec3, to: Vec3, color: Vec3) {
		self.line(from, to, color);

		let along = to - from;
		let length = along.length();
		if length <= f32::EPSILON {
			return;
		}

		let direction = along / length;
		let back = direction * (length * BARB);
		let side = perpendicular(direction) * (length * BARB * BARB_SPREAD);
		let other = direction.cross(side);

		self.line(to, to - back + side, color);
		self.line(to, to - back - side, color);
		self.line(to, to - back + other, color);
		self.line(to, to - back - other, color);
	}

	/// Draws three axes in the usual colors: x red, y green, z blue.
	///
	/// @param size - how long each arm is
	pub fn axes(&mut self, at: Vec3, rotation: Quat, size: f32) {
		let arm = size.max(0.0);

		self.line(at, at + rotation * (Vec3::X * arm), RED);
		self.line(at, at + rotation * (Vec3::Y * arm), GREEN);
		self.line(at, at + rotation * (Vec3::Z * arm), BLUE);
	}

	/// Draws words anchored at a point in the world.
	///
	/// Cut to [`MAX_LABEL_CHARS`] characters, by characters rather than by
	/// bytes so that cutting cannot leave half of one.
	pub fn label(&mut self, at: Vec3, text: &str, color: Vec3) {
		if text.is_empty() {
			return;
		}

		self.debug.push_label(Label {
			at,
			text: text.chars().take(MAX_LABEL_CHARS).collect(),
			color,
			expires: self.expires,
		});
	}
}

/// The twelve edges of a box, as pairs of indices into its eight corners.
const BOX_EDGES: [[usize; 2]; 12] = [
	[0, 1],
	[1, 2],
	[2, 3],
	[3, 0],
	[4, 5],
	[5, 6],
	[6, 7],
	[7, 4],
	[0, 4],
	[1, 5],
	[2, 6],
	[3, 7],
];

/// Some unit vector at a right angle to this one.
///
/// Which one does not matter - it only has to be consistent and not degenerate,
/// which is what the choice of reference axis is about: crossing with the axis
/// a direction is most nearly parallel to is what produces a zero vector.
fn perpendicular(direction: Vec3) -> Vec3 {
	let reference = if direction.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };

	direction.cross(reference).normalize_or(Vec3::Y)
}

/// How far round a circle the `step`th segment ends.
#[expect(
	clippy::as_conversions,
	clippy::cast_precision_loss,
	reason = "the numerator is at most CIRCLE_SEGMENTS, which is two dozen"
)]
fn turn(step: usize) -> f32 { core::f32::consts::TAU * (step as f32) / (CIRCLE_SEGMENTS as f32) }

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_fresh_table_has_nothing_in_it() {
		let debug = Debug::new();

		assert!(debug.is_empty(), "nothing is drawn before anyone asks");
		assert_eq!(debug.dropped(), 0, "and nothing has been lost");
	}

	#[test]
	fn a_transient_line_survives_its_own_step_and_no_longer() {
		let mut debug = Debug::new();

		debug.begin_step(1.0);
		debug.line(Vec3::ZERO, Vec3::X, WHITE);
		assert_eq!(debug.lines().len(), 1, "submitted during the step");

		// every frame drawn between this step and the next sees it, which is
		// the reason the sweep is at the top of a step rather than the bottom.
		debug.begin_step(1.0 + 1.0 / 60.0);
		assert!(debug.lines().is_empty(), "and it is gone at the top of the next step");
	}

	#[test]
	fn a_lasting_line_outlives_the_step_that_drew_it() {
		let mut debug = Debug::new();

		debug.begin_step(1.0);
		debug.lasting(0.5).line(Vec3::ZERO, Vec3::X, RED);

		debug.begin_step(1.4);
		assert_eq!(debug.lines().len(), 1, "half a second has not gone by yet");

		debug.begin_step(1.6);
		assert!(debug.lines().is_empty(), "and now it has");
	}

	#[test]
	fn a_negative_lifetime_is_no_lifetime_rather_than_a_line_already_gone() {
		let mut debug = Debug::new();

		debug.begin_step(2.0);
		debug.lasting(-5.0).line(Vec3::ZERO, Vec3::X, RED);

		let expires = debug
			.lines()
			.first()
			.map_or(f32::NAN, |line| line.expires);
		assert!(
			(expires - 2.0).abs() < f32::EPSILON,
			"a lifetime nobody meant should behave like the ordinary case, got {expires}"
		);
	}

	#[test]
	fn a_pen_carries_its_depth_rule_onto_every_mark() {
		let mut debug = Debug::new();

		debug.on_top().point(Vec3::ZERO, 0.1, RED);

		assert_eq!(debug.lines().len(), 3, "one stroke along each axis");
		assert!(
			debug.lines().iter().all(|line| line.on_top),
			"and the whole shape draws through the world, not one third of it"
		);
	}

	#[test]
	fn the_two_pen_settings_compose() {
		let mut debug = Debug::new();

		debug.begin_step(3.0);
		debug
			.lasting(1.0)
			.on_top()
			.line(Vec3::ZERO, Vec3::Y, GREEN);

		let line = debug.lines().first().expect("the line was taken");
		assert!(line.on_top, "asking for both gives both");
		assert!((line.expires - 4.0).abs() < f32::EPSILON, "got {}", line.expires);
	}

	#[test]
	fn a_box_is_twelve_edges_and_a_ball_is_three_rings() {
		let mut debug = Debug::new();

		debug.bounds(Vec3::splat(-1.0), Vec3::splat(1.0), WHITE);
		assert_eq!(debug.lines().len(), 12, "a box has twelve edges and no more");

		debug.clear();
		debug.ball(Vec3::ZERO, 1.0, WHITE);
		assert_eq!(debug.lines().len(), CIRCLE_SEGMENTS * 3, "three closed circles");
	}

	#[test]
	fn every_corner_of_a_box_is_on_the_bounds_it_was_given() {
		let mut debug = Debug::new();
		let (min, max) = (Vec3::new(-1.0, 0.0, -2.0), Vec3::new(3.0, 4.0, 2.0));

		debug.bounds(min, max, WHITE);

		for line in debug.lines() {
			for end in [line.from, line.to] {
				assert!(
					end.abs_diff_eq(end.clamp(min, max), 1.0e-4),
					"every corner is on the bounds it was given, got {end}"
				);
			}
		}
	}

	#[test]
	fn a_turned_box_is_the_same_box_turned() {
		let mut debug = Debug::new();
		let quarter = Quat::from_rotation_y(core::f32::consts::FRAC_PI_2);

		debug.cuboid(Vec3::ZERO, Vec3::new(2.0, 1.0, 1.0), quarter, WHITE);

		let widest = debug
			.lines()
			.iter()
			.flat_map(|line| [line.from, line.to])
			.fold(0.0_f32, |most, end| most.max(end.z.abs()));

		assert!(
			(widest - 2.0).abs() < 1.0e-4,
			"a box two long in x, turned a quarter turn about y, is two long in z: got {widest}"
		);
	}

	#[test]
	fn every_point_of_a_circle_is_the_radius_from_the_middle() {
		let mut debug = Debug::new();
		let center = Vec3::new(1.0, 2.0, 3.0);

		debug.circle(center, Vec3::new(1.0, 1.0, 0.0), 2.0, WHITE);

		for line in debug.lines() {
			let distance = (line.from - center).length();
			assert!(
				(distance - 2.0).abs() < 1.0e-3,
				"a circle of radius two, got a point {distance} out"
			);
		}
	}

	#[test]
	fn a_circle_about_an_axis_lies_flat_across_it() {
		let mut debug = Debug::new();

		debug.circle(Vec3::ZERO, Vec3::Y, 1.0, WHITE);

		for line in debug.lines() {
			assert!(
				line.from.y.abs() < 1.0e-4,
				"a circle across y has no height to it, got {}",
				line.from.y
			);
		}
	}

	#[test]
	fn a_circle_of_no_size_is_not_drawn() {
		let mut debug = Debug::new();

		debug.circle(Vec3::ZERO, Vec3::Y, 0.0, WHITE);

		assert!(debug.is_empty(), "nothing with no radius is on screen");
	}

	#[test]
	fn an_arrow_points_at_where_it_was_aimed() {
		let mut debug = Debug::new();
		let (from, to) = (Vec3::ZERO, Vec3::new(0.0, 5.0, 0.0));

		debug.arrow(from, to, YELLOW);

		let shaft = debug
			.lines()
			.first()
			.expect("the shaft is the first line");
		assert!(shaft.to.abs_diff_eq(to, 1.0e-4), "the shaft ends at the target");
		assert_eq!(debug.lines().len(), 5, "a shaft and four barbs");

		for barb in debug.lines().iter().skip(1) {
			assert!(
				barb.to.y < to.y,
				"every barb points back along the shaft, got one at {}",
				barb.to.y
			);
		}
	}

	#[test]
	fn an_arrow_of_no_length_is_a_shaft_and_no_head() {
		let mut debug = Debug::new();

		debug.arrow(Vec3::ONE, Vec3::ONE, YELLOW);

		assert_eq!(
			debug.lines().len(),
			1,
			"there is no direction to put a head on, and normalizing one would be a nan"
		);
	}

	#[test]
	fn the_three_axes_are_drawn_in_the_three_colors() {
		let mut debug = Debug::new();

		debug.axes(Vec3::ZERO, Quat::IDENTITY, 1.0);

		let colors: Vec<Vec3> = debug
			.lines()
			.iter()
			.map(|line| line.color)
			.collect();
		assert!(
			colors.len() == 3
				&& colors[0].abs_diff_eq(RED, 1.0e-6)
				&& colors[1].abs_diff_eq(GREEN, 1.0e-6)
				&& colors[2].abs_diff_eq(BLUE, 1.0e-6),
			"x red, y green, z blue, in that order: got {colors:?}"
		);
	}

	#[test]
	fn a_label_is_cut_rather_than_left_to_fill_the_screen() {
		let mut debug = Debug::new();
		let long = "a".repeat(MAX_LABEL_CHARS * 3);

		debug.label(Vec3::ZERO, &long, WHITE);

		assert_eq!(
			debug
				.labels()
				.first()
				.map(|label| label.text.chars().count()),
			Some(MAX_LABEL_CHARS),
			"cut to the bound"
		);
	}

	#[test]
	fn a_label_is_cut_between_characters_and_not_inside_one() {
		let mut debug = Debug::new();
		let long = "\u{444}".repeat(MAX_LABEL_CHARS * 2);

		debug.label(Vec3::ZERO, &long, WHITE);

		let text = debug
			.labels()
			.first()
			.map_or("", |label| label.text.as_str());
		assert_eq!(text.chars().count(), MAX_LABEL_CHARS, "counted in characters");
		assert_eq!(
			text.len(),
			MAX_LABEL_CHARS * 2,
			"and each of those characters is two bytes, so none of them was split"
		);
	}

	#[test]
	fn an_empty_label_is_not_a_label() {
		let mut debug = Debug::new();

		debug.label(Vec3::ZERO, "", WHITE);

		assert!(debug.is_empty(), "there is nothing to read");
	}

	#[test]
	fn a_runaway_caller_loses_lines_and_the_count_says_so() {
		let mut debug = Debug::new();

		for _ in 0..=MAX_LINES {
			debug.line(Vec3::ZERO, Vec3::X, WHITE);
		}

		assert_eq!(debug.lines().len(), MAX_LINES, "it stops at the bound");
		assert_eq!(debug.dropped(), 1, "and says how much it stopped short by");
	}

	#[test]
	fn the_lost_count_is_about_this_step_rather_than_the_run() {
		let mut debug = Debug::new();

		for _ in 0..=MAX_LABELS {
			debug.label(Vec3::ZERO, "x", WHITE);
		}
		assert_eq!(debug.dropped(), 1, "one over");

		debug.begin_step(1.0);
		assert_eq!(debug.dropped(), 0, "and the next step starts over");
	}

	#[test]
	fn clearing_takes_lasting_geometry_with_it() {
		let mut debug = Debug::new();

		debug
			.lasting(600.0)
			.line(Vec3::ZERO, Vec3::X, RED);
		debug
			.lasting(600.0)
			.label(Vec3::ZERO, "ten minutes", RED);
		debug.clear();

		assert!(debug.is_empty(), "clear means clear, or it is not worth having");
	}
}
