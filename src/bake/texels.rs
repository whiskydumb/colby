//! Which point of which surface each texel of the lightmap stands for.
//!
//! ```text
//!   Texels::of(&scene, &atlas)      every texel a triangle touches, and the ring
//!   texels.samples()                a point on a surface for each touched one
//! ```
//!
//! **A texel a triangle touches stands for a point on that triangle**: its
//! middle, when its middle is inside one, and otherwise the point of the
//! triangles touching it nearest its middle. So a texel along the edge of a
//! chart, which the picture reads half of whenever it reads the edge, is lit
//! as the surface it is half on rather than as a copy of the texel beside it.
//! A middle on an edge two triangles share is inside exactly one of them, and
//! every corner is held at 256 steps a texel, so each of these decisions is
//! made in whole numbers and comes out the same on every machine.
//!
//! **A texel no triangle touches, but within a texel of one, belongs to that
//! triangle's chart.** A picture reading a point reads the four texels whose
//! middles are within one texel of it, which reaches a texel past what a chart
//! touches; those texels are filled later from their own chart's and no
//! other's, @ref [`fill`](crate::lightmap). Two charts the unwrap's gutter
//! apart never reach the same texel, so no chart's light is ever read as
//! another's.
//!
//! A chart, here, is triangles joined by edges they share on the second set:
//! both ends in the same place on the sheet, to the bit.

use colby_core::glam::{Vec2, Vec3};

use crate::{
	atlas::{Atlas, Rect},
	scene::{Corner, Scene},
};

/// How finely a corner is placed on the picture: 256 steps a texel.
const SUBTEXELS: i64 = 256;

/// What a texel of the picture has none of: no sample, no chart, no place.
pub const NONE: u32 = u32::MAX;

/// One texel a triangle touches, and the point of the surface it stands for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
	/// Which texel: its row times the picture's width, and its column.
	pub at: u32,

	/// The triangle the point is on, as the scene numbers them.
	pub triangle: u32,

	/// How much of the triangle's second corner the point is.
	pub along: f32,

	/// And how much of its third.
	pub across: f32,

	/// How far a step of one texel along the picture's rows, and one down its
	/// columns, reaches across the surface there, in the world.
	pub reach: [Vec3; 2],
}

/// What every texel of the picture stands for.
#[derive(Clone, Debug, PartialEq)]
pub struct Texels {
	width: u32,
	height: u32,
	samples: Vec<Sample>,
	/// Every texel: the sample it is, or [`NONE`].
	sampled: Vec<u32>,
	/// Every texel: the chart it belongs to, or [`NONE`].
	charts: Vec<u32>,
	/// Every texel: the piece whose place holds it, or [`NONE`].
	owners: Vec<u32>,
	/// How many texels a triangle reaches without touching.
	ring: usize,
}

/// What a triangle is to one texel, the best first.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Claim {
	/// Nothing within a texel of it.
	Nothing,

	/// Within a texel of the texel's middle, without touching the texel.
	Reaches(usize),

	/// Touching the texel, its nearest point this far from the middle, squared.
	Touches(usize, [f64; 2], f64),

	/// Holding the texel's middle.
	Holds(usize),
}

impl Claim {
	/// Whether this claim beats another on the same texel: a middle held beats
	/// a touch, a nearer touch beats a further one, and a touch beats a reach.
	/// On a tie the earlier triangle keeps the texel, which is the one that
	/// claimed it first.
	fn beats(self, other: Self) -> bool {
		match (self, other) {
			| (Self::Touches(_, _, near), Self::Touches(_, _, far)) => near < far,
			| (Self::Nothing, _)
			| (Self::Reaches(_), Self::Reaches(_) | Self::Touches(..) | Self::Holds(_))
			| (Self::Holds(_) | Self::Touches(..), Self::Holds(_)) => false,
			| (Self::Holds(_), _)
			| (Self::Touches(..), Self::Nothing | Self::Reaches(_))
			| (Self::Reaches(_), Self::Nothing) => true,
		}
	}

	/// The triangle making the claim, as the piece counts them.
	const fn triangle(self) -> Option<usize> {
		match self {
			| Self::Nothing => None,
			| Self::Reaches(triangle) | Self::Touches(triangle, ..) | Self::Holds(triangle) =>
				Some(triangle),
		}
	}
}

/// One triangle laid on the picture.
struct Laid {
	/// Its corners on the picture, in texels.
	places: [[f64; 2]; 3],

	/// The same held at 256 steps a texel, wound so that the inside is to the
	/// left of every edge.
	wound: [[i64; 2]; 3],

	/// How far one texel along the rows and one down the columns reach across
	/// the surface.
	reach: [Vec3; 2],
}

impl Texels {
	/// What every texel of a picture stands for.
	///
	/// @param scene - what stands still
	/// @param atlas - where each piece's light goes
	#[must_use]
	pub fn of(scene: &Scene, atlas: &Atlas) -> Self {
		let (width, height) = (atlas.width(), atlas.height());
		let count = usize::try_from(u64::from(width) * u64::from(height)).unwrap_or(0);
		let mut texels = Self {
			width,
			height,
			samples: Vec::new(),
			sampled: vec![NONE; count],
			charts: vec![NONE; count],
			owners: vec![NONE; count],
			ring: 0,
		};
		let mut next_chart = 0_u32;

		for (piece, place) in atlas.places().iter().enumerate() {
			let (Some(place), Some(held)) = (place, scene.pieces().get(piece)) else {
				continue;
			};
			let first = usize::try_from(held.first).unwrap_or(usize::MAX);
			let triangles: Vec<[u32; 3]> = scene
				.triangles()
				.iter()
				.skip(first)
				.take(usize::try_from(held.count).unwrap_or(0))
				.copied()
				.collect();
			let charts = charts_of(scene, &triangles);

			texels.lay(scene, &triangles, (*place, piece), (first, &charts, next_chart));
			next_chart = next_chart.saturating_add(
				charts
					.iter()
					.copied()
					.max()
					.map_or(0, |most| most + 1),
			);
		}

		texels
	}

	/// How many texels across the picture is.
	#[must_use]
	pub const fn width(&self) -> u32 { self.width }

	/// How many down.
	#[must_use]
	pub const fn height(&self) -> u32 { self.height }

	/// Every texel a triangle touches, in the order of the picture's rows
	/// within each piece's place, the places in the scene's order.
	#[must_use]
	pub fn samples(&self) -> &[Sample] { &self.samples }

	/// The sample a texel is, if it is one.
	///
	/// @param at - the texel, as a place in the picture
	#[must_use]
	pub fn sample_at(&self, at: usize) -> Option<u32> {
		self.sampled
			.get(at)
			.copied()
			.filter(|index| *index != NONE)
	}

	/// The chart a texel belongs to, or [`NONE`].
	#[must_use]
	pub fn chart(&self, at: usize) -> u32 { self.charts.get(at).copied().unwrap_or(NONE) }

	/// The piece whose place holds a texel, a texel around it included, or
	/// [`NONE`].
	#[must_use]
	pub fn owner(&self, at: usize) -> u32 { self.owners.get(at).copied().unwrap_or(NONE) }

	/// How many texels a triangle reaches without touching.
	#[must_use]
	pub const fn ring(&self) -> usize { self.ring }

	/// One piece's triangles laid on its place.
	///
	/// @param scene - whose corners
	/// @param triangles - the piece's triangles
	/// @param place - where on the picture, and which piece it is
	/// @param charts - where the piece's triangles start in the scene's list,
	/// each triangle's chart within the piece, and the first chart's number
	fn lay(
		&mut self,
		scene: &Scene,
		triangles: &[[u32; 3]],
		(place, piece): (Rect, usize),
		(first, charts, chart_base): (usize, &[u32], u32),
	) {
		// the place and a texel round it, which is the half of the gutter this
		// piece may fill: two places are two apart, so each takes one
		let window = Window {
			left: place.left.saturating_sub(1),
			top: place.top.saturating_sub(1),
			right: (place.left + place.width + 1).min(self.width),
			bottom: (place.top + place.height + 1).min(self.height),
		};
		let laids: Vec<Option<Laid>> = triangles
			.iter()
			.map(|triangle| laid(scene, triangle, place))
			.collect();
		let mut claims = vec![Claim::Nothing; window.texels()];

		for (index, laid) in laids.iter().enumerate() {
			if let Some(laid) = laid {
				window.claim(laid, index, &mut claims);
			}
		}

		let piece = u32::try_from(piece).unwrap_or(NONE);

		for (offset, claim) in claims.iter().enumerate() {
			let at = window.texel(offset, self.width);

			if let Some(owner) = self.owners.get_mut(at) {
				*owner = piece;
			}

			let Some(index) = claim.triangle() else {
				continue;
			};

			if let Some(held) = self.charts.get_mut(at) {
				*held = chart_base.saturating_add(charts.get(index).copied().unwrap_or(0));
			}

			let point = match claim {
				| Claim::Holds(_) => middle_of(at, self.width),
				| Claim::Touches(_, nearest, _) => *nearest,
				| Claim::Reaches(_) | Claim::Nothing => {
					self.ring += 1;

					continue;
				},
			};

			if let Some(Some(laid)) = laids.get(index) {
				self.sample(laid, first + index, at, point);
			}
		}
	}

	/// Makes a texel a sample of one point of one triangle.
	///
	/// @param laid - the triangle on the picture
	/// @param triangle - the triangle, as the scene numbers them
	/// @param at - the texel
	/// @param point - the point on the picture it stands for
	fn sample(&mut self, laid: &Laid, triangle: usize, at: usize, point: [f64; 2]) {
		let [along, across] = weights(laid.places, point);
		let (Ok(triangle), Ok(texel), Ok(sample)) =
			(u32::try_from(triangle), u32::try_from(at), u32::try_from(self.samples.len()))
		else {
			return;
		};

		self.samples.push(Sample {
			at: texel,
			triangle,
			along: narrowed(along),
			across: narrowed(across),
			reach: laid.reach,
		});

		if let Some(held) = self.sampled.get_mut(at) {
			*held = sample;
		}
	}
}

/// A piece's place and a texel round it, on the picture.
#[derive(Clone, Copy, Debug)]
struct Window {
	left: u32,
	top: u32,
	right: u32,
	bottom: u32,
}

impl Window {
	/// How many columns it has.
	fn across(self) -> usize {
		usize::try_from(self.right.saturating_sub(self.left)).unwrap_or(0)
	}

	/// How many texels it has.
	fn texels(self) -> usize {
		self.across() * usize::try_from(self.bottom.saturating_sub(self.top)).unwrap_or(0)
	}

	/// A texel of it, as a place in the picture.
	///
	/// @param offset - the texel, as a place in the window
	/// @param width - how wide the picture is
	fn texel(self, offset: usize, width: u32) -> usize {
		let across = self.across().max(1);
		let column = u64::from(self.left) + u64::try_from(offset % across).unwrap_or(0);
		let row = u64::from(self.top) + u64::try_from(offset / across).unwrap_or(0);

		usize::try_from(row * u64::from(width) + column).unwrap_or(usize::MAX)
	}

	/// What one triangle claims of every texel within a texel of it, kept where
	/// it beats what was there.
	///
	/// @param laid - the triangle on the picture
	/// @param index - its number within the piece
	/// @param claims - the window's claims so far
	fn claim(self, laid: &Laid, index: usize, claims: &mut [Claim]) {
		let (low, high) = bounds(&laid.wound);
		let columns =
			(low[0] - 1).max(i64::from(self.left))..=(high[0] + 1).min(i64::from(self.right) - 1);
		let rows =
			(low[1] - 1).max(i64::from(self.top))..=(high[1] + 1).min(i64::from(self.bottom) - 1);
		let across = self.across();

		for (column, row) in rows.flat_map(|row| columns.clone().map(move |column| (column, row)))
		{
			let (Ok(along), Ok(down)) = (
				usize::try_from(column - i64::from(self.left)),
				usize::try_from(row - i64::from(self.top)),
			) else {
				continue;
			};
			let claim = claim_of(laid, index, [column, row]);

			if let Some(held) = claims.get_mut(down * across + along)
				&& claim.beats(*held)
			{
				*held = claim;
			}
		}
	}
}

/// One triangle of a piece on the picture, or nothing for one that covers no
/// area of it.
fn laid(scene: &Scene, triangle: &[u32; 3], place: Rect) -> Option<Laid> {
	let [one, two, three] = triangle.map(|index| {
		usize::try_from(index)
			.ok()
			.and_then(|index| scene.corners().get(index))
	});
	let corners: [&Corner; 3] = [one?, two?, three?];
	let places = corners.map(|corner| place.place(corner.uv2));
	let held = places.map(|at| at.map(subtexel));
	let wound = match edge(held[0], held[1], held[2]) {
		| 0 => return None,
		| side if side > 0 => held,
		| _ => [held[0], held[2], held[1]],
	};

	Some(Laid {
		places,
		wound,
		reach: reach_of(places, corners.map(|corner| corner.position)),
	})
}

/// What one triangle is to one texel.
///
/// @param laid - the triangle on the picture
/// @param index - its number within the piece
/// @param texel - the texel's column and row
fn claim_of(laid: &Laid, index: usize, texel: [i64; 2]) -> Claim {
	let middle = texel.map(|at| at * SUBTEXELS + SUBTEXELS / 2);
	let [one, two, three] = laid.wound;

	if inside(one, two, middle) && inside(two, three, middle) && inside(three, one, middle) {
		return Claim::Holds(index);
	}

	let square = texel.map(|at| [at * SUBTEXELS, (at + 1) * SUBTEXELS]);

	if meets(&laid.wound, square, false) {
		let wanted = middle.map(|at| as_float(at) / as_float(SUBTEXELS));
		let corners = laid
			.wound
			.map(|corner| corner.map(|at| as_float(at) / as_float(SUBTEXELS)));
		let nearest = nearest_on(corners, wanted);
		let [x, y] = [nearest[0] - wanted[0], nearest[1] - wanted[1]];
		let (across, down) = (x * x, y * y);

		return Claim::Touches(index, nearest, across + down);
	}

	let reach =
		texel.map(|at| [at * SUBTEXELS - SUBTEXELS / 2, (at + 1) * SUBTEXELS + SUBTEXELS / 2]);

	if meets(&laid.wound, reach, true) {
		return Claim::Reaches(index);
	}

	Claim::Nothing
}

/// The least and the greatest texel a triangle's corners fall in.
fn bounds(wound: &[[i64; 2]; 3]) -> ([i64; 2], [i64; 2]) {
	let least = |axis: usize| {
		wound
			.iter()
			.map(|corner| corner[axis])
			.min()
			.unwrap_or(0)
	};
	let most = |axis: usize| {
		wound
			.iter()
			.map(|corner| corner[axis])
			.max()
			.unwrap_or(0)
	};

	(
		[least(0), least(1)].map(|at| at.div_euclid(SUBTEXELS)),
		[most(0), most(1)].map(|at| at.div_euclid(SUBTEXELS)),
	)
}

/// Whether a triangle wound the right way and a box meet.
///
/// @param wound - the triangle, its inside to the left of every edge
/// @param reach - the box, as its least and greatest place on each axis
/// @param open - whether the box's own edge is outside it, so that touching
/// it is not meeting it
fn meets(wound: &[[i64; 2]; 3], reach: [[i64; 2]; 2], open: bool) -> bool {
	let (low, high) = bounds_exact(wound);
	let apart = |least: i64, most: i64, from: i64, to: i64| {
		if open {
			least >= to || most <= from
		} else {
			least > to || most < from
		}
	};

	if apart(low[0], high[0], reach[0][0], reach[0][1])
		|| apart(low[1], high[1], reach[1][0], reach[1][1])
	{
		return false;
	}

	let corners = [
		[reach[0][0], reach[1][0]],
		[reach[0][1], reach[1][0]],
		[reach[0][0], reach[1][1]],
		[reach[0][1], reach[1][1]],
	];

	!(0..3).any(|side| {
		let (from, to) = (wound[side], wound[(side + 1) % 3]);

		corners.iter().all(|corner| {
			let facing = edge(from, to, *corner);

			if open { facing <= 0 } else { facing < 0 }
		})
	})
}

/// The least and greatest of a triangle's corners, axis by axis, at 256 steps
/// a texel.
fn bounds_exact(wound: &[[i64; 2]; 3]) -> ([i64; 2], [i64; 2]) {
	let least = |axis: usize| {
		wound
			.iter()
			.map(|corner| corner[axis])
			.min()
			.unwrap_or(0)
	};
	let most = |axis: usize| {
		wound
			.iter()
			.map(|corner| corner[axis])
			.max()
			.unwrap_or(0)
	};

	([least(0), least(1)], [most(0), most(1)])
}

/// Twice the area of the triangle from `from` to `to` to `point`: above nought
/// when the point is to the left of the edge.
fn edge(from: [i64; 2], to: [i64; 2], point: [i64; 2]) -> i128 {
	let along = i128::from(to[0] - from[0]) * i128::from(point[1] - from[1]);
	let across = i128::from(to[1] - from[1]) * i128::from(point[0] - from[0]);

	along - across
}

/// Whether a point is on the inside of one edge of a triangle wound the right
/// way: strictly inside, or on the edge when the edge is one that owns what is
/// on it. Of an edge and the same edge running the other way exactly one owns
/// it, so a middle on an edge two triangles share is inside one of them.
fn inside(from: [i64; 2], to: [i64; 2], point: [i64; 2]) -> bool {
	match edge(from, to, point) {
		| 0 => {
			let (along, across) = (to[0] - from[0], to[1] - from[1]);

			across > 0 || (across == 0 && along < 0)
		},
		| side => side > 0,
	}
}

/// The point of a triangle nearest a point, both on the picture.
///
/// The regions of a triangle's corners, edges and inside, each told apart by
/// dot products, which is additions and products and one quotient: the same
/// answer on every machine.
fn nearest_on(corners: [[f64; 2]; 3], point: [f64; 2]) -> [f64; 2] {
	let [one, two, three] = corners;
	let (side, other) = (minus2(two, one), minus2(three, one));
	let from_one = minus2(point, one);
	let (d1, d2) = (dot2(side, from_one), dot2(other, from_one));

	if d1 <= 0.0 && d2 <= 0.0 {
		return one;
	}

	let from_two = minus2(point, two);
	let (d3, d4) = (dot2(side, from_two), dot2(other, from_two));

	if d3 >= 0.0 && d4 <= d3 {
		return two;
	}

	let near_third = cross_terms(d1, d4, d3, d2);

	if near_third <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
		return toward(one, side, d1 / (d1 - d3));
	}

	let from_three = minus2(point, three);
	let (d5, d6) = (dot2(side, from_three), dot2(other, from_three));

	if d6 >= 0.0 && d5 <= d6 {
		return three;
	}

	let near_second = cross_terms(d5, d2, d1, d6);

	if near_second <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
		return toward(one, other, d2 / (d2 - d6));
	}

	let near_first = cross_terms(d3, d6, d5, d4);
	let (rising, falling) = (d4 - d3, d5 - d6);

	if near_first <= 0.0 && rising >= 0.0 && falling >= 0.0 {
		return toward(two, minus2(three, two), rising / (rising + falling));
	}

	// inside, which a texel whose middle is not can only be by a rounding
	point
}

/// `a b - c d`, with the two products apart.
fn cross_terms(a: f64, b: f64, c: f64, d: f64) -> f64 {
	let first = a * b;
	let second = c * d;

	first - second
}

/// A point moved part of the way along a vector.
fn toward(from: [f64; 2], way: [f64; 2], part: f64) -> [f64; 2] {
	let across = way[0] * part;
	let down = way[1] * part;

	[from[0] + across, from[1] + down]
}

/// How much of a triangle's second corner and of its third a point on the
/// picture is.
fn weights(corners: [[f64; 2]; 3], point: [f64; 2]) -> [f64; 2] {
	let [one, two, three] = corners;
	let (side, other, to) = (minus2(two, one), minus2(three, one), minus2(point, one));
	let whole = cross2(side, other);

	if whole == 0.0 {
		return [0.0; 2];
	}

	[cross2(to, other) / whole, cross2(side, to) / whole]
}

/// How far one texel along the picture's rows and one down its columns reach
/// across a triangle's surface.
fn reach_of(places: [[f64; 2]; 3], positions: [Vec3; 3]) -> [Vec3; 2] {
	let (side, other) = (minus2(places[1], places[0]), minus2(places[2], places[0]));
	let whole = cross2(side, other);

	if whole == 0.0 {
		return [Vec3::ZERO; 2];
	}

	let [first, second] = [positions[1] - positions[0], positions[2] - positions[0]]
		.map(|edge| edge.to_array().map(f64::from));
	let blend = |one: f64, two: f64| {
		[0, 1, 2].map(|axis| {
			let near = first[axis] * one;
			let far = second[axis] * two;

			narrowed((near + far) / whole)
		})
	};

	[
		Vec3::from_array(blend(other[1], -side[1])),
		Vec3::from_array(blend(-other[0], side[0])),
	]
}

/// Every triangle's chart, within the piece: triangles joined by edges whose
/// ends are in the same places on the second set, numbered by their first
/// triangle.
fn charts_of(scene: &Scene, triangles: &[[u32; 3]]) -> Vec<u32> {
	let key = |index: u32| {
		usize::try_from(index)
			.ok()
			.and_then(|index| scene.corners().get(index))
			.map_or([0; 2], |corner| folded(corner.uv2))
	};
	let mut edges: Vec<([[u32; 2]; 2], usize)> = Vec::with_capacity(triangles.len() * 3);

	for (index, triangle) in triangles.iter().enumerate() {
		for side in 0..3 {
			let (from, to) = (key(triangle[side]), key(triangle[(side + 1) % 3]));

			edges.push((if from <= to { [from, to] } else { [to, from] }, index));
		}
	}

	edges.sort_unstable();

	let mut parent: Vec<usize> = (0..triangles.len()).collect();

	for pair in edges.windows(2) {
		if pair[0].0 == pair[1].0 {
			join(&mut parent, pair[0].1, pair[1].1);
		}
	}

	let mut numbered = vec![NONE; triangles.len()];
	let mut next = 0_u32;

	(0..triangles.len())
		.map(|triangle| {
			let top = root(&mut parent, triangle);

			if numbered[top] == NONE {
				numbered[top] = next;
				next += 1;
			}

			numbered[top]
		})
		.collect()
}

/// A place on the second set as bits, minus nought folded into nought.
fn folded(uv2: Vec2) -> [u32; 2] { [uv2.x + 0.0, uv2.y + 0.0].map(f32::to_bits) }

/// The triangle a set is kept under, flattening the path on the way.
fn root(parent: &mut [usize], triangle: usize) -> usize {
	let mut top = triangle;

	while parent[top] != top {
		top = parent[top];
	}

	let mut at = triangle;

	while parent[at] != top {
		let next = parent[at];
		parent[at] = top;
		at = next;
	}

	top
}

/// Joins two triangles' sets under the smaller of their two roots.
fn join(parent: &mut [usize], one: usize, other: usize) {
	let (one, other) = (root(parent, one), root(parent, other));

	if one != other {
		let (keep, under) = if one < other { (one, other) } else { (other, one) };

		parent[under] = keep;
	}
}

/// A texel's middle on the picture, in texels.
fn middle_of(at: usize, width: u32) -> [f64; 2] {
	let width = usize::try_from(width).unwrap_or(1).max(1);
	let [column, row] = [at % width, at / width].map(|part| u32::try_from(part).unwrap_or(0));

	[f64::from(column) + 0.5, f64::from(row) + 0.5]
}

/// One flat point less another.
fn minus2(from: [f64; 2], less: [f64; 2]) -> [f64; 2] { [from[0] - less[0], from[1] - less[1]] }

/// The dot product of two flat vectors.
fn dot2(one: [f64; 2], other: [f64; 2]) -> f64 {
	let first = one[0] * other[0];
	let second = one[1] * other[1];

	first + second
}

/// The flat cross product of two vectors.
fn cross2(one: [f64; 2], other: [f64; 2]) -> f64 {
	cross_terms(one[0], other[1], one[1], other[0])
}

/// A place on the picture held at 256 steps a texel, rounded to the nearest.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "a place on a picture no wider than the widest, at 256 steps a texel: well inside \
	          an i64, and rounded on the line above the cast"
)]
fn subtexel(place: f64) -> i64 {
	let held = if place.is_nan() {
		0.0
	} else {
		place.clamp(-8192.0, 16_384.0)
	};

	(held * as_float(SUBTEXELS)).round() as i64
}

/// A whole number of steps as a float, which every one here is small enough
/// to be exactly.
#[expect(
	clippy::as_conversions,
	clippy::cast_precision_loss,
	reason = "a place at 256 steps a texel on a picture no wider than 8192: exact in a double"
)]
const fn as_float(steps: i64) -> f64 { steps as f64 }

/// A double narrowed to a float, rounding to the nearest.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "a weight or a length worked out in double precision and narrowed once, where it \
	          is kept"
)]
const fn narrowed(value: f64) -> f32 { value as f32 }

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{MeshId, Renderable, Transform, World, material::MaterialId},
		unwrap::TEXELS,
	};

	use super::*;

	/// A world of a stretched cube, a quad and a small cube, which between
	/// them have charts of every shape the built-in meshes make.
	fn world() -> World {
		let mut world = World::new();

		for (mesh, at, scale) in [
			(MeshId::CUBE, Vec3::ZERO, Vec3::new(1.3, 0.7, 2.1)),
			(MeshId::QUAD, Vec3::new(5.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.7)),
			(MeshId::CUBE, Vec3::new(-5.0, 0.0, 0.0), Vec3::splat(0.45)),
		] {
			let thing = world.entities.spawn_at(Transform {
				position: at,
				scale,
				..Transform::IDENTITY
			});

			world
				.entities
				.set_renderable(thing, Renderable::of(mesh, MaterialId::DEFAULT, Vec3::ONE));
		}

		world
	}

	/// The scene, the places and the texels of [`world`].
	fn laid_out() -> (Scene, Atlas, Texels) {
		let scene = Scene::of(&world());
		let atlas = Atlas::of(&scene, TEXELS).expect("three small things fit");
		let texels = Texels::of(&scene, &atlas);

		(scene, atlas, texels)
	}

	/// A triangle's corners on the picture, worked out here rather than by the
	/// module: its place, times its second set.
	fn on_picture(scene: &Scene, atlas: &Atlas, piece: u32, triangle: usize) -> [[f64; 2]; 3] {
		let place = atlas.place(piece).expect("a place");

		scene.triangles()[triangle].map(|index| {
			let corner = scene.corners()[usize::try_from(index).expect("small")];
			let across = f64::from(corner.uv2.x) * f64::from(place.width);
			let down = f64::from(corner.uv2.y) * f64::from(place.height);

			[f64::from(place.left) + across, f64::from(place.top) + down]
		})
	}

	/// Every triangle of every piece, with the piece it is in.
	fn triangles_of(scene: &Scene) -> Vec<(u32, usize)> {
		scene
			.pieces()
			.iter()
			.enumerate()
			.flat_map(|(piece, held)| {
				let first = usize::try_from(held.first).expect("small");
				let count = usize::try_from(held.count).expect("small");

				(first..first + count)
					.map(move |triangle| (u32::try_from(piece).expect("small"), triangle))
			})
			.collect()
	}

	/// Points all over a triangle: a grid of its two edges' fractions.
	fn points_on(corners: [[f64; 2]; 3]) -> Vec<[f64; 2]> {
		(0..=24_u32)
			.flat_map(|step| (0..=24 - step).map(move |other| (step, other)))
			.map(|(step, other)| {
				let (one, two) = (f64::from(step) / 24.0, f64::from(other) / 24.0);

				[0, 1].map(|axis| {
					let along = (corners[1][axis] - corners[0][axis]) * one;
					let across = (corners[2][axis] - corners[0][axis]) * two;

					corners[0][axis] + along + across
				})
			})
			.collect()
	}

	/// A whole number of texels, from a float that is one.
	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		reason = "a column or a row of a small picture, whole already"
	)]
	fn whole(place: f64) -> i64 { place as i64 }

	/// The place in the picture of a column and a row.
	fn at(texels: &Texels, column: i64, row: i64) -> usize {
		let width = i64::from(texels.width());

		usize::try_from(row * width + column).expect("inside the picture")
	}

	/// The four texels a picture read at a point reads: those whose middles are
	/// nearest it.
	fn read_at(texels: &Texels, point: [f64; 2]) -> [usize; 4] {
		let [column, row] = point.map(|place| whole((place - 0.5).floor()));

		[[0, 0], [1, 0], [0, 1], [1, 1]]
			.map(|[across, down]| at(texels, column + across, row + down))
	}

	#[test]
	fn every_texel_a_chart_is_read_through_belongs_to_that_chart() {
		// a picture read at a point reads the four texels whose middles are
		// nearest it: walk points all over every triangle and ask each of the
		// four whose it is
		let (scene, atlas, texels) = laid_out();
		let reads: Vec<(u32, usize, u32, [f64; 2])> = triangles_of(&scene)
			.into_iter()
			.flat_map(|(piece, triangle)| {
				let corners = on_picture(&scene, &atlas, piece, triangle);
				let middle = [0, 1]
					.map(|axis| (corners[0][axis] + corners[1][axis] + corners[2][axis]) / 3.0);
				let chart =
					texels.chart(at(&texels, whole(middle[0].floor()), whole(middle[1].floor())));

				points_on(corners)
					.into_iter()
					.map(move |point| (piece, triangle, chart, point))
			})
			.collect();

		for (piece, triangle, chart, point) in &reads {
			assert_ne!(*chart, NONE, "the texel under triangle {triangle}'s middle has a chart");

			for texel in read_at(&texels, *point) {
				assert_eq!(
					texels.chart(texel),
					*chart,
					"a point of triangle {triangle} at {point:?} reads texel {texel} of another \
					 chart"
				);
				assert_eq!(texels.owner(texel), *piece, "or of another thing's place");
			}
		}

		assert!(reads.len() > 5000, "every triangle walked: {}", reads.len());
	}

	/// Whether a texel's middle is well inside a triangle on the picture, where
	/// a rounding cannot decide it.
	fn well_inside(corners: [[f64; 2]; 3], column: i64, row: i64) -> bool {
		let [one, two, three] = corners;
		let (side, other) =
			([two[0] - one[0], two[1] - one[1]], [three[0] - one[0], three[1] - one[1]]);
		let to = [0, 1].map(|axis| {
			let middle = if axis == 0 { column } else { row };

			integral(middle) + 0.5 - one[axis]
		});
		let cross = |first: [f64; 2], second: [f64; 2]| {
			let forward = first[0] * second[1];
			let back = first[1] * second[0];

			forward - back
		};
		let (along, across) =
			(cross(to, other) / cross(side, other), cross(side, to) / cross(side, other));

		along > 0.01 && across > 0.01 && along + across < 0.99
	}

	/// A whole number as a float, which every column and row here is small
	/// enough to be exactly.
	#[expect(
		clippy::as_conversions,
		clippy::cast_precision_loss,
		reason = "a column or a row of a small picture"
	)]
	const fn integral(whole: i64) -> f64 { whole as f64 }

	#[test]
	fn a_texel_whose_middle_a_triangle_holds_is_a_sample_of_that_triangle() {
		let (scene, atlas, texels) = laid_out();
		let mut held = 0;

		for (piece, triangle) in triangles_of(&scene) {
			let corners = on_picture(&scene, &atlas, piece, triangle);
			let axis_of = |axis: usize| corners.map(|corner| corner[axis]);
			let low = [0, 1].map(|axis| {
				let values = axis_of(axis);

				whole(values[0].min(values[1]).min(values[2]).floor())
			});
			let high = [0, 1].map(|axis| {
				let values = axis_of(axis);

				whole(values[0].max(values[1]).max(values[2]).ceil())
			});
			let middles: Vec<(i64, i64)> = (low[1]..=high[1])
				.flat_map(|row| (low[0]..=high[0]).map(move |column| (column, row)))
				.filter(|&(column, row)| well_inside(corners, column, row))
				.collect();

			for (column, row) in middles {
				let texel = at(&texels, column, row);
				let sample = texels
					.sample_at(texel)
					.expect("a texel whose middle is held");
				let by = texels.samples()[usize::try_from(sample).expect("small")].triangle;

				assert_eq!(
					usize::try_from(by).expect("small"),
					triangle,
					"texel {texel} is a sample of the triangle holding its middle"
				);
				held += 1;
			}
		}

		assert!(held > 200, "the middles of every chart: {held}");
	}

	/// Where a sample's point is on the picture, from its triangle's corners
	/// there and its two weights.
	fn sampled_at(scene: &Scene, atlas: &Atlas, sample: &Sample) -> [f64; 2] {
		let triangle = usize::try_from(sample.triangle).expect("small");
		let piece = triangles_of(scene)
			.into_iter()
			.find(|(_, held)| *held == triangle)
			.map(|(piece, _)| piece)
			.expect("a triangle of a piece");
		let corners = on_picture(scene, atlas, piece, triangle);
		let (along, across) = (f64::from(sample.along), f64::from(sample.across));

		[0, 1].map(|axis| {
			let first = (corners[1][axis] - corners[0][axis]) * along;
			let second = (corners[2][axis] - corners[0][axis]) * across;

			corners[0][axis] + first + second
		})
	}

	/// The middle of a sample's texel on the picture.
	fn middle(texels: &Texels, sample: &Sample) -> [f64; 2] {
		let width = texels.width();

		[f64::from(sample.at % width) + 0.5, f64::from(sample.at / width) + 0.5]
	}

	#[test]
	fn every_texel_a_triangle_touches_is_a_sample() {
		// a dense walk over every triangle, and the texel each point is in
		let (scene, atlas, texels) = laid_out();

		for (piece, triangle) in triangles_of(&scene) {
			let corners = on_picture(&scene, &atlas, piece, triangle);

			for point in points_on(corners) {
				let texel = at(&texels, whole(point[0].floor()), whole(point[1].floor()));

				assert!(
					texels.sample_at(texel).is_some(),
					"texel {texel} holds a point of triangle {triangle} at {point:?} and is no \
					 sample"
				);
			}
		}
	}

	#[test]
	fn a_sample_whose_middle_is_held_stands_for_that_middle() {
		let (scene, atlas, texels) = laid_out();
		let mut held = 0;

		for sample in texels.samples() {
			let (point, middle) = (sampled_at(&scene, &atlas, sample), middle(&texels, sample));
			let off = (point[0] - middle[0])
				.abs()
				.max((point[1] - middle[1]).abs());

			// a middle inside its triangle is the point; a touch is the nearest
			// point, which is in the texel or a corner's reach of it
			assert!(off < 0.71, "texel {} stands {off} texels from its point", sample.at);
			held += usize::from(off < 1.0e-3);
		}

		assert!(held * 2 > texels.samples().len(), "most samples are their middles: {held}");
	}

	/// The nearest of some walked points inside a texel to its middle, or
	/// nothing when none is inside it.
	fn nearest_inside(walked: &[[f64; 2]], middle: [f64; 2]) -> Option<f64> {
		walked
			.iter()
			.filter(|at| (at[0] - middle[0]).abs() <= 0.5 && (at[1] - middle[1]).abs() <= 0.5)
			.map(|at| (at[0] - middle[0]).hypot(at[1] - middle[1]))
			.reduce(f64::min)
	}

	#[test]
	fn a_touched_texel_stands_for_the_nearest_point_of_any_triangle_touching_it() {
		// every triangle a walk finds a point of inside a touched texel touches
		// it, and none of those comes nearer its middle than the point it
		// stands for - a walked point is never nearer than the nearest one
		let (scene, atlas, texels) = laid_out();
		let mut compared = 0;

		for sample in texels.samples() {
			let (point, middle) = (sampled_at(&scene, &atlas, sample), middle(&texels, sample));
			let own = (point[0] - middle[0]).hypot(point[1] - middle[1]);

			if own < 1.0e-3 {
				continue;
			}

			let touching: Vec<(usize, f64)> = triangles_of(&scene)
				.into_iter()
				.filter_map(|(piece, triangle)| {
					let walked = points_on(on_picture(&scene, &atlas, piece, triangle));

					nearest_inside(&walked, middle).map(|other| (triangle, other))
				})
				.collect();

			for (triangle, other) in touching {
				// the point is worked out on the corners held at 256 steps a texel,
				// so it may be a step and a half further than the walk says
				assert!(
					own <= other + 0.006,
					"texel {} stands {own} from its middle where triangle {triangle} comes \
					 {other}",
					sample.at
				);
				compared += 1;
			}
		}

		assert!(compared > 50, "touched texels with a triangle to weigh: {compared}");
	}

	#[test]
	fn a_sample_stands_for_a_point_on_its_triangle_within_its_texel() {
		let (_, _, texels) = laid_out();
		let width = texels.width();

		for sample in texels.samples() {
			let (along, across) = (f64::from(sample.along), f64::from(sample.across));

			assert!(
				along >= -1.0e-3 && across >= -1.0e-3 && along + across <= 1.0 + 1.0e-3,
				"texel {} stands for a point of its triangle: {along}, {across}",
				sample.at
			);
			assert!(sample.at / width < texels.height(), "and is a texel of the picture");
		}
	}

	#[test]
	fn a_cube_is_six_charts_and_a_quad_one() {
		let (scene, atlas, texels) = laid_out();
		let charts_of = |piece: u32| {
			let place = atlas.place(piece).expect("a place");
			let mut seen: Vec<u32> = (place.top..place.top + place.height)
				.flat_map(|row| {
					(place.left..place.left + place.width).map(move |column| (column, row))
				})
				.map(|(column, row)| texels.chart(at(&texels, i64::from(column), i64::from(row))))
				.filter(|chart| *chart != NONE)
				.collect();

			seen.sort_unstable();
			seen.dedup();
			seen.len()
		};

		assert_eq!(scene.pieces().len(), 3, "three things");
		assert_eq!(charts_of(0), 6, "a face a chart");
		assert_eq!(charts_of(1), 1, "a quad is one");
		assert_eq!(charts_of(2), 6, "however small the cube");
	}
}
