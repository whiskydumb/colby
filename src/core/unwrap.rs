//! A mesh's second set of texture coordinates: where each point of its surface
//! keeps the light a bake worked out for it.
//!
//! ```text
//!   unwrap::second(&mut mesh, scale)    charted, cut, measured, packed
//!   mesh.sheet                          how many texels across and down
//!   mesh.paint[vertex].uv2              where the vertex is on that sheet
//! ```
//!
//! **Not an unwrap from scratch but a repacking of the one the mesh already
//! has.** Whoever made the mesh cut its surface into islands when they laid its
//! first set of coordinates out, and every picture it wears already reads them.
//! What a lightmap needs that a picture does not is that no two points of the
//! surface share a place, so an island that is repeated, mirrored onto another
//! or tiled is laid out again here on a sheet of its own, beside the others,
//! with room between them. Cutting a surface into islands is the part of an
//! unwrap that takes the most code, and this takes it from the mesh as it came.
//!
//! What a chart is, rule by rule:
//!
//! - **Two triangles are one chart where they share an edge in the first set as
//!   well as in space**: both ends of the edge in the same place with the same
//!   coordinates, the edge running the other way, and the two wound the same
//!   way in the first set. A mesh cut apart at an edge only because its normals
//!   change there is still one island - the place and the coordinates are
//!   compared, not the vertex - and a triangle wound against its neighbor is a
//!   fold, which is never joined.
//! - **A triangle the first set collapsed is laid by the way it faces**: of the
//!   six ways along the three axes, the one its normal leans furthest along,
//!   joined to its neighbors facing the same way and laid flat by dropping that
//!   axis. A mesh made with no coordinates at all is laid out whole this way.
//! - **A chart that still lands on itself is cut a triangle at a time.** A fold
//!   cannot be in one; what is left is a surface that winds all the way around
//!   and past where it began, found by drawing the chart at the sheet's density
//!   and meeting a texel it already covered.
//!
//! **A chart takes the area its surface has, at [`TEXELS`] a unit, in the
//! proportions its two axes carry.** The length a unit of each of its two
//! coordinates carries across the surface is measured on its own, so a face
//! stretched four to one in its first set comes out four to one; then both are
//! scaled together until the chart's area is the surface's. The surface is
//! measured at the scale the mesh stands at - a piece of a model at the scale
//! the model puts it - so that a mesh modeled in centimeters is not given a
//! hundred times the texels of the same thing modeled in meters.
//!
//! **Every chart is a rectangle with [`GUTTER`] texels between it and the
//! next**, and one between it and the edge of the sheet, so two sheets laid
//! edge to edge keep the same distance. A sample reads the texels whose middles
//! are within one of it, so two charts that far apart never read each other's.
//! Rectangles rather than the charts' own outlines: a coarser level of the mesh
//! is made of the chart's own vertices, so its triangles stay inside the
//! chart's rectangle, and a chart tucked into another's hollow would be under
//! them.
//!
//! **The same bytes on every machine.** Nothing here draws a random number or
//! walks a hash map; the arithmetic is additions, products, quotients and
//! square roots, which the floating-point standard pins to the bit, with no
//! product fused into a sum; and every decision after the measuring is made in
//! whole numbers.

use std::{cmp::Reverse, collections::VecDeque};

use crate::{
	abi::mesh::{MeshData, PaintVertex},
	glam::Vec3,
};

/// How many texels one unit of a surface is laid out at.
///
/// The density a bake is made at unless it asks for another. A sheet is never
/// drawn smaller than it was laid out, so this is also the least any mesh gets.
pub const TEXELS: f32 = 5.0;

/// How many texels at least stand between two charts on a sheet.
///
/// A sample reads the texels whose middles are within one texel of it, so two
/// charts two apart never read a texel the other needs.
pub const GUTTER: u32 = 2;

/// How far in from the corner of its cell a chart starts: half the gutter, so
/// that a chart is one texel from the edge of the sheet and two from the next.
const MARGIN: u32 = GUTTER / 2;

/// How nearly flat a triangle may lie in the first set before it counts as not
/// laid out there at all: the square of the sine of the angle between its two
/// edges there.
///
/// A millionth of a radian. Below it the length a unit of the coordinates
/// carries across the surface is a quotient of two numbers that are noise.
const COLLAPSED: f64 = 1.0e-12;

/// The widest a chart is laid out, in texels. A larger one is laid out coarser.
///
/// As wide as a picture on nearly every device is allowed to be.
const WIDEST: f64 = 16_384.0;

/// How finely a corner is placed when a chart is drawn to look for where it
/// lands on itself: 256 steps a texel.
const SUBTEXELS: i32 = 256;

/// The most texels a chart is drawn at when looking for where it lands on
/// itself. A larger chart is looked at coarser, which finds only a larger
/// overlap.
const LOOKED_AT: f64 = 4_194_304.0;

/// How many widths of sheet the packer tries before it keeps the smallest.
const WIDTHS: u64 = 16;

/// How a chart is laid flat before it is scaled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Flat {
	/// By its own coordinates in the first set.
	First,

	/// By its place, dropping the axis it faces along: nought is x, one y, two
	/// z.
	Facing(usize),
}

/// One edge of a triangle, as charts are found by: the family its triangle
/// joins, the corner it runs from and the one it runs to, and the triangle.
type Edge = (u8, [u32; 5], [u32; 5], usize);

/// A sheet one width gave: what it is ranked by, where each cell went, and how
/// big it is.
type Sheet = ((u64, u64, u64), Vec<[u32; 2]>, [u32; 2]);

/// Triangles laid out together.
#[derive(Clone, Debug)]
struct Chart {
	/// How its corners are laid flat.
	flat: Flat,

	/// Which triangles, in the order the mesh has them.
	triangles: Vec<usize>,
}

/// A chart, measured.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Measure {
	/// The least of each of its two flat coordinates.
	low: [f64; 2],

	/// How many texels a unit of each of the two carries.
	rate: [f64; 2],

	/// How many texels it spans the first way and the second.
	size: [f64; 2],
}

impl Measure {
	/// Whether it is laid on the sheet turned a quarter, so that it lies wide.
	fn upright(&self) -> bool { self.size[1] > self.size[0] }

	/// The cell it takes on the sheet, gutter and all, once it lies wide.
	fn cell(&self) -> [u32; 2] {
		let (wide, high) = if self.upright() {
			(self.size[1], self.size[0])
		} else {
			(self.size[0], self.size[1])
		};

		[texels(wide) + GUTTER, texels(high) + GUTTER]
	}

	/// Where a point of the chart falls inside its cell, from the cell's
	/// corner once the margin is taken off.
	///
	/// @param flat - the point's flat coordinates
	fn spot(&self, flat: [f64; 2]) -> [f64; 2] {
		let along = (flat[0] - self.low[0]) * self.rate[0];
		let across = (flat[1] - self.low[1]) * self.rate[1];

		// a quarter turn rather than a swap, which would be a mirror as well
		if self.upright() {
			[across, self.size[0] - along]
		} else {
			[along, across]
		}
	}
}

/// Gives a mesh a second set of coordinates to keep baked light in, unless it
/// has one.
///
/// **What is left as it was**: a mesh bones move, which is never baked; a mesh
/// that already has a sheet; one with coarser levels, whose indices would name
/// vertices the copies below took apart - the levels are made afterwards; and
/// one that is not whole triangles over its own vertices.
///
/// **A second set somebody made is kept**, and given a square sheet on which it
/// holds [`TEXELS`] a unit on average. It counts as one when it covers any
/// area.
///
/// Otherwise the mesh is laid out here, and three things change: every vertex
/// has a paint entry, white where nobody painted it; a vertex two charts share
/// is copied, with everything else it carries, so that each has its own; and
/// [`MeshData::sheet`] says how many texels across and down the second set was
/// laid out for. Nothing else moves, so the mesh draws as it did.
///
/// @param mesh - edited in place
/// @param scale - how the mesh is scaled where it stands, axis by axis; one for
/// a mesh whose own units are the world's
pub fn second(mesh: &mut MeshData, scale: Vec3) {
	if mesh.is_skinned()
		|| mesh.sheet != [0, 0]
		|| !mesh.levels.is_empty()
		|| mesh.indices.is_empty()
		|| !mesh.indices.len().is_multiple_of(3)
		|| !mesh.indices_are_in_range()
		|| !mesh.paint_fits()
		|| u32::try_from(mesh.vertices.len() + mesh.indices.len()).is_err()
	{
		return;
	}

	let scale = [scale.x, scale.y, scale.z].map(|axis| f64::from(axis.abs()));

	if let Some(side) = own(mesh, scale) {
		mesh.sheet = [side, side];

		return;
	}

	if mesh.paint.is_empty() {
		mesh.paint = vec![PaintVertex::PLAIN; mesh.vertices.len()];
	}

	let (charts, measures) = cut(mesh, charted(mesh, scale), scale);
	let owners = separated(mesh, &charts);
	let cells: Vec<[u32; 2]> = measures.iter().map(Measure::cell).collect();
	let (corners, sheet) = packed(&cells);

	for (vertex, owner) in owners.into_iter().enumerate() {
		let (Some(chart), Some(measure), Some(corner)) =
			(charts.get(owner), measures.get(owner), corners.get(owner))
		else {
			continue;
		};
		let spot = measure.spot(flattened(mesh, vertex, chart.flat, scale));
		let across = f64::from(corner[0] + MARGIN) + spot[0];
		let down = f64::from(corner[1] + MARGIN) + spot[1];

		if let Some(entry) = mesh.paint.get_mut(vertex) {
			entry.uv2 =
				[narrowed(across / f64::from(sheet[0])), narrowed(down / f64::from(sheet[1]))];
		}
	}

	mesh.sheet = sheet;
}

/// The side of the square sheet a second set somebody made is given, when the
/// mesh has one: as many texels as keep [`TEXELS`] a unit on average over its
/// whole surface.
///
/// @param mesh - the mesh
/// @param scale - how it is scaled where it stands
/// @return the side, or nothing for a mesh whose second set covers no area
fn own(mesh: &MeshData, scale: [f64; 3]) -> Option<u32> {
	if mesh.paint.is_empty() {
		return None;
	}

	let (mut laid, mut world) = (0.0_f64, 0.0_f64);

	for triangle in 0..mesh.triangles() {
		let corners = corners_of(mesh, triangle);
		let flat = corners.map(|vertex| {
			mesh.paint
				.get(vertex)
				.map_or([0.0; 2], |entry| entry.uv2.map(f64::from))
		});
		let place = corners.map(|vertex| placed(mesh, vertex, scale));

		laid += twice_area(flat).abs();
		world += length(cross(minus(place[1], place[0]), minus(place[2], place[0])));
	}

	(laid > 0.0 && laid.is_finite()).then(|| {
		let side = f64::from(TEXELS) * (world / laid).sqrt();

		texels(side).max(1)
	})
}

/// The mesh's triangles, gathered into charts.
///
/// Every edge of every triangle is written down with the family its triangle
/// joins and the two corners it runs between, and sorted; an edge then finds
/// the ones running the other way between the same corners by a search, and
/// their triangles are joined. Sorted rather than hashed, so that nothing about
/// the answer depends on the order a table was walked in.
///
/// @return the charts, in the order of each one's first triangle
fn charted(mesh: &MeshData, scale: [f64; 3]) -> Vec<Chart> {
	let count = mesh.triangles();
	let families: Vec<u8> = (0..count)
		.map(|triangle| family(mesh, corners_of(mesh, triangle), scale))
		.collect();
	let mut edges: Vec<Edge> = Vec::with_capacity(count * 3);

	for (triangle, &kin) in families.iter().enumerate() {
		let corners = corners_of(mesh, triangle);
		let first = kin < 2;

		for side in 0..3 {
			edges.push((
				kin,
				key(mesh, corners[side], first),
				key(mesh, corners[(side + 1) % 3], first),
				triangle,
			));
		}
	}

	edges.sort_unstable();

	let mut parent: Vec<usize> = (0..count).collect();

	for &(kin, from, to, triangle) in &edges {
		let start = edges.partition_point(|other| (other.0, other.1, other.2) < (kin, to, from));

		for other in edges
			.iter()
			.skip(start)
			.take_while(|other| (other.0, other.1, other.2) == (kin, to, from))
		{
			join(&mut parent, triangle, other.3);
		}
	}

	let mut chart_of = vec![usize::MAX; count];
	let mut charts: Vec<Chart> = Vec::new();

	for triangle in 0..count {
		let top = root(&mut parent, triangle);

		// the root is the smallest triangle of its set, so it was met first
		if top == triangle {
			chart_of[triangle] = charts.len();
			charts.push(Chart {
				flat: flat_of(families[triangle]),
				triangles: vec![triangle],
			});
		} else if let Some(chart) = charts.get_mut(chart_of[top]) {
			chart_of[triangle] = chart_of[top];
			chart.triangles.push(triangle);
		}
	}

	charts
}

/// Which charts a triangle may join: nought and one for one laid out in the
/// first set, wound one way and the other; two to seven for one the first set
/// collapsed, by the axis its normal leans furthest along and the side of it.
fn family(mesh: &MeshData, corners: [usize; 3], scale: [f64; 3]) -> u8 {
	let flat = corners.map(|vertex| first(mesh, vertex));
	let (along, across) = (minus2(flat[1], flat[0]), minus2(flat[2], flat[0]));
	let twice = twice_area(flat);
	// the square of the sine of the angle between the two edges, times the
	// square of both their lengths
	let open = twice * twice;
	let least = COLLAPSED * dot2(along, along) * dot2(across, across);

	if twice.is_finite() && open > least {
		return u8::from(twice < 0.0);
	}

	let place = corners.map(|vertex| placed(mesh, vertex, scale));
	let normal = cross(minus(place[1], place[0]), minus(place[2], place[0]));
	let axis = facing(normal);
	let behind = normal
		.get(axis)
		.is_some_and(|component| *component < 0.0);
	let side = u8::try_from(axis * 2).unwrap_or(0);

	2 + side + u8::from(behind)
}

/// The axis a normal leans furthest along, the first of any that tie.
fn facing(normal: [f64; 3]) -> usize {
	let [x, y, z] = normal.map(f64::abs);

	if x >= y && x >= z {
		0
	} else if y >= z {
		1
	} else {
		2
	}
}

/// How a chart of a family is laid flat.
fn flat_of(family: u8) -> Flat {
	if family < 2 {
		Flat::First
	} else {
		Flat::Facing(usize::from((family - 2) / 2))
	}
}

/// What an edge's corner is compared by: its place and, in the first set, its
/// coordinates, each as the bits of the number with both zeros one number.
fn key(mesh: &MeshData, vertex: usize, first: bool) -> [u32; 5] {
	let Some(corner) = mesh.vertices.get(vertex) else {
		return [0; 5];
	};
	let bits = |value: f32| if value == 0.0 { 0 } else { value.to_bits() };
	let place = corner.position.map(bits);
	let flat = if first { corner.uv.map(bits) } else { [0; 2] };

	[place[0], place[1], place[2], flat[0], flat[1]]
}

/// The set a triangle's set has joined, which is its smallest triangle.
fn root(parent: &mut [usize], triangle: usize) -> usize {
	let mut at = triangle;

	while parent[at] != at {
		let above = parent[parent[at]];

		parent[at] = above;
		at = above;
	}

	at
}

/// Joins two triangles' sets under the smaller of their roots.
fn join(parent: &mut [usize], one: usize, other: usize) {
	let (one, other) = (root(parent, one), root(parent, other));

	if one != other {
		parent[one.max(other)] = one.min(other);
	}
}

/// The charts, each measured, with every one that lands on itself cut a
/// triangle at a time.
fn cut(mesh: &MeshData, charts: Vec<Chart>, scale: [f64; 3]) -> (Vec<Chart>, Vec<Measure>) {
	let mut kept = Vec::with_capacity(charts.len());
	let mut measures = Vec::with_capacity(charts.len());

	for chart in charts {
		let measure = measured(mesh, &chart, scale);

		if !lands_on_itself(mesh, &chart, &measure, scale) {
			kept.push(chart);
			measures.push(measure);

			continue;
		}

		for &triangle in &chart.triangles {
			let alone = Chart {
				flat: chart.flat,
				triangles: vec![triangle],
			};

			measures.push(measured(mesh, &alone, scale));
			kept.push(alone);
		}
	}

	(kept, measures)
}

/// Measures a chart: where its flat coordinates start, how many texels a unit
/// of each carries, and how many texels it spans.
///
/// The length a unit of each coordinate carries across the surface is worked
/// out a triangle at a time and averaged over the chart by the area each
/// triangle has in flat coordinates; then both are scaled by one number so that
/// the chart's area is the surface's at [`TEXELS`] a unit. A chart too wide for
/// [`WIDEST`] is scaled down whole until it fits.
fn measured(mesh: &MeshData, chart: &Chart, scale: [f64; 3]) -> Measure {
	let mut low = [f64::INFINITY; 2];
	let mut high = [f64::NEG_INFINITY; 2];
	let (mut along, mut across, mut laid, mut world) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

	for &triangle in &chart.triangles {
		let corners = corners_of(mesh, triangle);
		let flat = corners.map(|vertex| flattened(mesh, vertex, chart.flat, scale));
		let place = corners.map(|vertex| placed(mesh, vertex, scale));

		for point in flat {
			low = [low[0].min(point[0]), low[1].min(point[1])];
			high = [high[0].max(point[0]), high[1].max(point[1])];
		}

		let (first, second) = (minus2(flat[1], flat[0]), minus2(flat[2], flat[0]));
		let (one, other) = (minus(place[1], place[0]), minus(place[2], place[0]));

		// how far the surface moves along each flat coordinate, times the
		// triangle's flat area, which is what the average is weighed by
		along += length(minus(times(one, second[1]), times(other, first[1])));
		across += length(minus(times(other, first[0]), times(one, second[0])));
		laid += twice_area(flat).abs();
		world += length(cross(one, other));
	}

	let (per_first, per_second) = (along / laid, across / laid);
	let exact = (world / (per_first * per_second * laid)).sqrt();
	let texels = f64::from(TEXELS);
	let mut rate = [per_first * exact * texels, per_second * exact * texels];

	if !(rate[0].is_finite() && rate[1].is_finite() && laid > 0.0) {
		rate = [0.0; 2];
	}

	let mut size = [(high[0] - low[0]) * rate[0], (high[1] - low[1]) * rate[1]];

	if !(size[0].is_finite() && size[1].is_finite()) {
		(rate, size) = ([0.0; 2], [0.0; 2]);
	}

	let widest = size[0].max(size[1]);

	if widest > WIDEST {
		let shrink = WIDEST / widest;

		rate = rate.map(|each| each * shrink);
		// the widest way held to the widest exactly, rather than to a share of
		// it that rounds a hair past
		size = if size[0] >= size[1] {
			[WIDEST, size[1] * shrink]
		} else {
			[size[0] * shrink, WIDEST]
		};
	}

	Measure { low, rate, size }
}

/// Whether two triangles of a chart cover the middle of one texel, drawn at
/// the sheet's density.
///
/// Each corner is placed to the nearest of 256 steps a texel and every test is
/// made in whole numbers, and a middle on an edge two triangles share belongs
/// to exactly one of them, so a surface that only meets itself along its own
/// edges covers no middle twice. A chart of one triangle cannot land on itself.
fn lands_on_itself(mesh: &MeshData, chart: &Chart, measure: &Measure, scale: [f64; 3]) -> bool {
	if chart.triangles.len() < 2 {
		return false;
	}

	let area = measure.size[0] * measure.size[1];
	let shrink = if area > LOOKED_AT {
		(LOOKED_AT / area).sqrt()
	} else {
		1.0
	};
	let columns = usize::try_from(texels(measure.size[0] * shrink) + 1).unwrap_or(1);
	let rows = usize::try_from(texels(measure.size[1] * shrink) + 1).unwrap_or(1);
	let mut covered = vec![0_u64; (columns * rows).div_ceil(64)];

	chart.triangles.iter().any(|&triangle| {
		let fixed = corners_of(mesh, triangle).map(|vertex| {
			let flat = flattened(mesh, vertex, chart.flat, scale);
			let along = (flat[0] - measure.low[0]) * measure.rate[0];
			let across = (flat[1] - measure.low[1]) * measure.rate[1];

			[subtexel(along * shrink), subtexel(across * shrink)]
		});

		drawn_twice(fixed, [columns, rows], &mut covered)
	})
}

/// Draws one triangle into a map of covered texels, and says whether it met a
/// middle already covered.
///
/// @param corners - its corners, at 256 steps a texel
/// @param extent - how many texels the map is across and down
/// @param covered - one bit a texel, row by row
fn drawn_twice(corners: [[i64; 2]; 3], extent: [usize; 2], covered: &mut [u64]) -> bool {
	let [first, mut second, mut third] = corners.map(|corner| corner.map(i128::from));
	let wound = edge(first, second, third);

	if wound == 0 {
		return false;
	}

	if wound < 0 {
		(second, third) = (third, second);
	}

	let whole = i128::from(SUBTEXELS);
	let half = whole / 2;
	// the texels whose middles are inside the box: the first middle not below
	// the least corner, to the last not above the greatest
	let span = |axis: usize| {
		let least = first[axis].min(second[axis]).min(third[axis]);
		let most = first[axis].max(second[axis]).max(third[axis]);
		let last = i128::try_from(extent[axis]).unwrap_or(0) - 1;

		(-(half - least).div_euclid(whole)).max(0)..=(most - half).div_euclid(whole).min(last)
	};

	for row in span(1) {
		for column in span(0) {
			let middle = [column * whole + half, row * whole + half];

			if !(inside(first, second, middle)
				&& inside(second, third, middle)
				&& inside(third, first, middle))
			{
				continue;
			}

			let (Ok(column), Ok(row)) = (usize::try_from(column), usize::try_from(row)) else {
				continue;
			};

			if claimed(covered, row * extent[0] + column) {
				return true;
			}
		}
	}

	false
}

/// Marks one texel of a map of covered texels, and says whether it was marked
/// already.
///
/// @param covered - one bit a texel
/// @param at - which texel
fn claimed(covered: &mut [u64], at: usize) -> bool {
	let bit = 1_u64 << (at % 64);
	let Some(word) = covered.get_mut(at / 64) else {
		return false;
	};
	let before = *word & bit != 0;

	*word |= bit;

	before
}

/// Twice the area of the triangle from `from` to `to` to `point`: above nought
/// when the point is to the left of the edge.
fn edge(from: [i128; 2], to: [i128; 2], point: [i128; 2]) -> i128 {
	let along = (to[0] - from[0]) * (point[1] - from[1]);
	let across = (to[1] - from[1]) * (point[0] - from[0]);

	along - across
}

/// Whether a point is on the inside of one edge of a triangle wound the right
/// way: strictly inside, or on the edge when the edge is one that owns what is
/// on it. Of an edge and the same edge running the other way exactly one owns
/// it, so a point on an edge two triangles share is inside one of them.
fn inside(from: [i128; 2], to: [i128; 2], point: [i128; 2]) -> bool {
	match edge(from, to, point) {
		| 0 => {
			let (along, across) = (to[0] - from[0], to[1] - from[1]);

			across > 0 || (across == 0 && along < 0)
		},
		| side => side > 0,
	}
}

/// Gives every chart vertices of its own: a vertex a chart reaches after
/// another already has is copied, and the chart's triangles are pointed at the
/// copy.
///
/// Charts are visited in order and each triangle's corners in order, so the
/// copies are appended the same way every time.
///
/// @return which chart each vertex, copies included, belongs to; `usize::MAX`
/// for a vertex no triangle uses
fn separated(mesh: &mut MeshData, charts: &[Chart]) -> Vec<usize> {
	let before = mesh.vertices.len();
	let mut owners = vec![usize::MAX; before];
	let mut copies = vec![(usize::MAX, 0_u32); before];

	for (chart, each) in charts.iter().enumerate() {
		for corner in each
			.triangles
			.iter()
			.flat_map(|triangle| triangle * 3..triangle * 3 + 3)
		{
			let Some(vertex) = mesh
				.indices
				.get(corner)
				.and_then(|index| usize::try_from(*index).ok())
			else {
				continue;
			};
			let owner = owners.get(vertex).copied();

			if owner == Some(usize::MAX) {
				owners[vertex] = chart;
			}

			if owner.is_none_or(|owner| owner == usize::MAX || owner == chart) {
				continue;
			}

			let copy = match copies.get(vertex).copied() {
				| Some((made, copy)) if made == chart => copy,
				| _ => {
					let copy = duplicate(mesh, vertex);

					owners.push(chart);
					copies[vertex] = (chart, copy);

					copy
				},
			};

			mesh.indices[corner] = copy;
		}
	}

	owners
}

/// Appends a copy of a vertex, with its skin and its paint when it has them.
///
/// @return the copy's index
fn duplicate(mesh: &mut MeshData, vertex: usize) -> u32 {
	let copy = u32::try_from(mesh.vertices.len()).unwrap_or(u32::MAX);

	if let Some(original) = mesh.vertices.get(vertex).copied() {
		mesh.vertices.push(original);
	}

	if let Some(pull) = mesh.skin.get(vertex).copied() {
		mesh.skin.push(pull);
	}

	if let Some(paint) = mesh.paint.get(vertex).copied() {
		mesh.paint.push(paint);
	}

	copy
}

/// Where each cell goes on the sheet, and how big the sheet is.
///
/// The tallest first, then the widest, then in the order they came, each put
/// as low as it goes and then as far left - onto a skyline, the height the
/// cells placed so far reach at every column. Several widths of sheet are
/// tried, from the widest cell or seven tenths of the square the cells would
/// make, whichever is wider, to sixteen tenths of it; the one whose sheet has
/// the fewest texels is kept, and of those the squarest, then the narrowest.
///
/// @param cells - each chart's cell, gutter and all
/// @return each chart's corner, in the order given, and the sheet's width and
/// height
fn packed(cells: &[[u32; 2]]) -> (Vec<[u32; 2]>, [u32; 2]) {
	let mut order: Vec<usize> = (0..cells.len()).collect();

	order.sort_by_key(|&at| (Reverse(cells[at][1]), Reverse(cells[at][0]), at));

	let area: u64 = cells
		.iter()
		.map(|cell| u64::from(cell[0]) * u64::from(cell[1]))
		.sum();
	let widest = cells
		.iter()
		.map(|cell| u64::from(cell[0]))
		.max()
		.unwrap_or(1)
		.max(1);
	let square = area.isqrt();
	let narrowest = widest.max(square * 7 / 10);
	let broadest = narrowest.max((square * 16).div_ceil(10));
	let tried: Vec<u64> = if broadest - narrowest < WIDTHS {
		(narrowest..=broadest).collect()
	} else {
		(0..WIDTHS)
			.map(|step| narrowest + (broadest - narrowest) * step / (WIDTHS - 1))
			.collect()
	};
	let mut best: Option<Sheet> = None;

	for width in tried {
		let (corners, height) = skyline(cells, &order, width);
		let rank = (width * height, width.max(height), width);

		if best
			.as_ref()
			.is_none_or(|(kept, ..)| rank < *kept)
		{
			best = Some((rank, corners, [clamped(width), clamped(height)]));
		}
	}

	best.map_or((Vec::new(), [0, 0]), |(_, corners, sheet)| (corners, sheet))
}

/// Puts every cell onto a sheet of one width.
///
/// @param cells - each chart's cell
/// @param order - which cell to put down first, second, and so on
/// @param width - the sheet's width in texels, at least the widest cell
/// @return each cell's corner in the order `cells` has them, and how high the
/// sheet reaches
fn skyline(cells: &[[u32; 2]], order: &[usize], width: u64) -> (Vec<[u32; 2]>, u64) {
	let columns = usize::try_from(width).unwrap_or(usize::MAX);
	let mut sky = vec![0_u64; columns];
	let mut corners = vec![[0_u32; 2]; cells.len()];

	for &at in order {
		let [wide, high] = cells[at];
		let wide = usize::try_from(wide)
			.unwrap_or(columns)
			.clamp(1, columns);
		let (left, floor) = lowest(&sky, wide);
		let top = floor + u64::from(high);

		for column in sky.iter_mut().skip(left).take(wide) {
			*column = top;
		}

		corners[at] = [clamped(u64::try_from(left).unwrap_or(0)), clamped(floor)];
	}

	(corners, sky.iter().copied().max().unwrap_or(0))
}

/// The leftmost of the lowest places a cell so many columns wide can rest on a
/// skyline: where it starts, and the height of the tallest column under it.
fn lowest(sky: &[u64], wide: usize) -> (usize, u64) {
	let mut window: VecDeque<usize> = VecDeque::new();
	let mut best: Option<(usize, u64)> = None;

	for (at, &height) in sky.iter().enumerate() {
		while window
			.back()
			.is_some_and(|&last| sky[last] <= height)
		{
			window.pop_back();
		}

		window.push_back(at);

		if window
			.front()
			.is_some_and(|&first| first + wide <= at)
		{
			window.pop_front();
		}

		if at + 1 < wide {
			continue;
		}

		let under = window.front().map_or(0, |&tallest| sky[tallest]);

		if best.is_none_or(|(_, kept)| under < kept) {
			best = Some((at + 1 - wide, under));
		}
	}

	best.unwrap_or((0, 0))
}

/// The three vertices of a triangle, as indices into the vertices.
fn corners_of(mesh: &MeshData, triangle: usize) -> [usize; 3] {
	[0, 1, 2].map(|corner| {
		mesh.indices
			.get(triangle * 3 + corner)
			.and_then(|index| usize::try_from(*index).ok())
			.unwrap_or(0)
	})
}

/// A vertex's coordinates in the first set.
fn first(mesh: &MeshData, vertex: usize) -> [f64; 2] {
	mesh.vertices
		.get(vertex)
		.map_or([0.0; 2], |corner| corner.uv.map(f64::from))
}

/// A vertex's place, at the scale the mesh stands at.
fn placed(mesh: &MeshData, vertex: usize, scale: [f64; 3]) -> [f64; 3] {
	mesh.vertices
		.get(vertex)
		.map_or([0.0; 3], |corner| {
			let place = corner.position.map(f64::from);

			[place[0] * scale[0], place[1] * scale[1], place[2] * scale[2]]
		})
}

/// A vertex laid flat the way its chart is.
fn flattened(mesh: &MeshData, vertex: usize, flat: Flat, scale: [f64; 3]) -> [f64; 2] {
	match flat {
		| Flat::First => first(mesh, vertex),
		| Flat::Facing(axis) => {
			let place = placed(mesh, vertex, scale);

			match axis {
				| 0 => [place[2], place[1]],
				| 1 => [place[0], place[2]],
				| _ => [place[0], place[1]],
			}
		},
	}
}

/// Twice the signed area of a flat triangle.
fn twice_area(corners: [[f64; 2]; 3]) -> f64 {
	let (along, across) = (minus2(corners[1], corners[0]), minus2(corners[2], corners[0]));
	let forward = along[0] * across[1];
	let back = across[0] * along[1];

	forward - back
}

/// One flat point less another.
fn minus2(from: [f64; 2], less: [f64; 2]) -> [f64; 2] { [from[0] - less[0], from[1] - less[1]] }

/// The dot product of two flat vectors.
fn dot2(one: [f64; 2], other: [f64; 2]) -> f64 {
	let first = one[0] * other[0];
	let second = one[1] * other[1];

	first + second
}

/// One point less another.
fn minus(from: [f64; 3], less: [f64; 3]) -> [f64; 3] {
	[from[0] - less[0], from[1] - less[1], from[2] - less[2]]
}

/// A vector times a number.
fn times(vector: [f64; 3], by: f64) -> [f64; 3] { vector.map(|axis| axis * by) }

/// The cross product of two vectors.
fn cross(one: [f64; 3], other: [f64; 3]) -> [f64; 3] {
	let term = |first: usize, second: usize| {
		let forward = one[first] * other[second];
		let back = one[second] * other[first];

		forward - back
	};

	[term(1, 2), term(2, 0), term(0, 1)]
}

/// The length of a vector.
fn length(vector: [f64; 3]) -> f64 {
	let [x, y, z] = vector.map(|axis| axis * axis);

	(x + y + z).sqrt()
}

/// A size in texels, rounded up to whole ones; nought for one that is not a
/// number or not above nought.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "the value is held between nought and the widest chart and rounded up on the line \
	          above the cast, and try_from is not available for a float"
)]
fn texels(size: f64) -> u32 {
	let held = if size.is_nan() { 0.0 } else { size.clamp(0.0, WIDEST) };

	held.ceil() as u32
}

/// A place at 256 steps a texel, rounded to the nearest.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "a place inside a chart no wider than the widest, at 256 steps a texel: well \
	          inside an i64, and rounded on the line above the cast"
)]
fn subtexel(place: f64) -> i64 {
	let held = if place.is_nan() {
		0.0
	} else {
		place.clamp(-WIDEST, 2.0 * WIDEST)
	};

	(held * f64::from(SUBTEXELS)).round() as i64
}

/// A double narrowed to a float, rounding to the nearest.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "a place on the sheet worked out in double precision and narrowed once, where it \
	          is kept"
)]
const fn narrowed(value: f64) -> f32 { value as f32 }

/// A count of texels as a sheet holds it, the largest one there is for one
/// that does not fit.
fn clamped(value: u64) -> u32 { u32::try_from(value).unwrap_or(u32::MAX) }

/// The side of the square sheet a stretch of ground so long is laid out on:
/// [`TEXELS`] a unit, rounded up to whole texels, and the gutter.
///
/// For a mesh whose second set is written by hand rather than laid out here.
///
/// @param length - how long each side of it is, in world units
#[must_use]
pub fn square(length: f32) -> u32 { texels(f64::from(length) * f64::from(TEXELS)) + GUTTER }

/// Where a point a share of the way across a sheet's content falls on the
/// sheet: a texel in from the edge, and the rest of the sheet but the gutter
/// spread over the share.
///
/// @param share - from nought at one edge of what is laid out to one at the
/// other
/// @param side - how many texels the sheet has that way, gutter and all
#[must_use]
pub fn onto(share: f32, side: u32) -> f32 {
	let content = f64::from(side.saturating_sub(GUTTER));
	let along = f64::from(share) * content;

	narrowed((f64::from(MARGIN) + along) / f64::from(side.max(1)))
}

#[cfg(test)]
mod tests {
	use std::collections::HashMap;

	use super::*;
	use crate::{
		abi::{
			mesh::{MeshVertex, SkinVertex, cube, quad, sphere},
			terrain::{Terrain, TerrainKind},
		},
		glam::{Vec2, Vec4},
	};

	/// Loose triangles from `(position, uv)` corners, three a triangle, each
	/// corner a vertex of its own, all facing up.
	fn soup(corners: &[(Vec3, Vec2)]) -> MeshData {
		MeshData {
			vertices: corners
				.iter()
				.map(|(position, uv)| MeshVertex::new(*position, Vec3::Y, *uv))
				.collect(),
			indices: (0..u32::try_from(corners.len()).expect("a small fixture")).collect(),
			..MeshData::default()
		}
	}

	/// A flat rectangle `across` by `deep` in the xz plane, laid out on the
	/// whole of the first set: two triangles over four vertices.
	fn slab(across: f32, deep: f32) -> MeshData {
		let corner = |x: f32, z: f32, u: f32, v: f32| {
			MeshVertex::new(Vec3::new(x * across, 0.0, z * deep), Vec3::Y, Vec2::new(u, v))
		};

		MeshData {
			vertices: vec![
				corner(0.0, 0.0, 0.0, 0.0),
				corner(1.0, 0.0, 1.0, 0.0),
				corner(1.0, 1.0, 1.0, 1.0),
				corner(0.0, 1.0, 0.0, 1.0),
			],
			indices: vec![0, 2, 1, 0, 3, 2],
			..MeshData::default()
		}
	}

	/// A fan around one vertex whose first set winds twice around the middle,
	/// climbing as it goes so that no two of its corners are in one place: a
	/// surface that lands on itself with no fold anywhere. Every number is a
	/// sum of halves, so the fixture is the same bits on every machine.
	fn spiral(turns: usize) -> MeshData {
		const AROUND: [(f32, f32); 8] = [
			(1.0, 0.0),
			(0.75, 0.75),
			(0.0, 1.0),
			(-0.75, 0.75),
			(-1.0, 0.0),
			(-0.75, -0.75),
			(0.0, -1.0),
			(0.75, -0.75),
		];
		let steps = AROUND.len() * turns;
		let mut mesh = MeshData::default();

		mesh.vertices
			.push(MeshVertex::new(Vec3::ZERO, Vec3::Y, Vec2::splat(0.5)));

		for step in 0..=steps {
			let (x, z) = AROUND[step % AROUND.len()];
			let climb = f32::from(u16::try_from(step).expect("a small fan")) * 0.125;

			mesh.vertices.push(MeshVertex::new(
				Vec3::new(x, climb, z),
				Vec3::Y,
				Vec2::new(x, z) * 0.25 + 0.5,
			));
		}

		for step in 0..steps {
			let at = u32::try_from(step).expect("a small fan") + 1;

			mesh.indices.extend([0, at, at + 1]);
		}

		mesh
	}

	/// The charts of a mesh found again from its second set alone: triangles
	/// that share a corner in the same place with the same second
	/// coordinates.
	fn charts_of(mesh: &MeshData) -> Vec<Vec<usize>> {
		let mut parent: Vec<usize> = (0..mesh.triangles()).collect();
		let mut seen: HashMap<([u32; 3], [u32; 2]), usize> = HashMap::new();

		for triangle in 0..mesh.triangles() {
			for vertex in corners_of(mesh, triangle) {
				let at = (
					mesh.vertices[vertex].position.map(f32::to_bits),
					mesh.paint[vertex].uv2.map(f32::to_bits),
				);
				// the first triangle met at a corner, which joins itself
				let first = *seen.entry(at).or_insert(triangle);

				join(&mut parent, triangle, first);
			}
		}

		let mut charts: HashMap<usize, Vec<usize>> = HashMap::new();

		for triangle in 0..mesh.triangles() {
			charts
				.entry(root(&mut parent, triangle))
				.or_default()
				.push(triangle);
		}

		let mut out: Vec<Vec<usize>> = charts.into_values().collect();

		out.sort();

		out
	}

	/// A corner's second coordinates in texels of its sheet.
	fn on_sheet(mesh: &MeshData, vertex: usize) -> [f64; 2] {
		let uv2 = mesh.paint[vertex].uv2;

		[
			f64::from(uv2[0]) * f64::from(mesh.sheet[0]),
			f64::from(uv2[1]) * f64::from(mesh.sheet[1]),
		]
	}

	/// Everything a mesh laid out here must be, checked from the outside.
	///
	/// @param mesh - laid out, with a scale of one
	/// @param label - what it is, for the messages
	fn assert_laid_out(mesh: &MeshData, label: &str) {
		assert!(mesh.sheet[0] > 0 && mesh.sheet[1] > 0, "{label}: a sheet, got {:?}", mesh.sheet);
		assert!(mesh.paint_fits() && mesh.is_painted(), "{label}: a paint entry a vertex");

		let sheet = mesh.sheet.map(f64::from);
		let charts = charts_of(mesh);
		let slack = 1.0e-3;
		let mut boxes = Vec::new();

		for chart in &charts {
			let mut low = [f64::INFINITY; 2];
			let mut high = [f64::NEG_INFINITY; 2];
			let (mut texels_laid, mut world) = (0.0, 0.0);

			for &triangle in chart {
				let corners = corners_of(mesh, triangle);
				let spots = corners.map(|vertex| on_sheet(mesh, vertex));
				let place = corners.map(|vertex| placed(mesh, vertex, [1.0; 3]));

				low = spots
					.iter()
					.fold(low, |low, spot| [low[0].min(spot[0]), low[1].min(spot[1])]);
				high = spots
					.iter()
					.fold(high, |high, spot| [high[0].max(spot[0]), high[1].max(spot[1])]);
				texels_laid += twice_area(spots).abs();
				world += length(cross(minus(place[1], place[0]), minus(place[2], place[0])));
			}

			assert!(
				low[0] >= f64::from(MARGIN) - slack
					&& low[1] >= f64::from(MARGIN) - slack
					&& high[0] <= sheet[0] - f64::from(MARGIN) + slack
					&& high[1] <= sheet[1] - f64::from(MARGIN) + slack,
				"{label}: a chart {low:?}..{high:?} a texel inside a sheet of {sheet:?}"
			);

			let wanted = world * f64::from(TEXELS) * f64::from(TEXELS);
			let share = wanted * 1.0e-5;

			assert!(
				(texels_laid - wanted).abs() <= share + 1.0e-3,
				"{label}: a chart takes the area its surface has at five texels a unit, {} \
				 texels against {}",
				texels_laid / 2.0,
				wanted / 2.0
			);

			boxes.push((low, high));
		}

		for (at, (low, high)) in boxes.iter().enumerate() {
			for (other_low, other_high) in boxes.iter().skip(at + 1) {
				let apart_across = (other_low[0] - high[0]).max(low[0] - other_high[0]);
				let apart_down = (other_low[1] - high[1]).max(low[1] - other_high[1]);

				assert!(
					apart_across.max(apart_down) >= f64::from(GUTTER) - slack,
					"{label}: two charts {low:?}..{high:?} and {other_low:?}..{other_high:?} \
					 are {} texels apart",
					apart_across.max(apart_down)
				);
			}
		}
	}

	/// The mesh laid out, at a scale of one.
	fn laid(mut mesh: MeshData) -> MeshData {
		second(&mut mesh, Vec3::ONE);

		mesh
	}

	/// A built-in mesh without the second set it is built with.
	fn bare(mut mesh: MeshData) -> MeshData {
		mesh.paint.clear();
		mesh.sheet = [0, 0];

		mesh
	}

	#[test]
	fn a_square_unit_is_one_chart_five_texels_across_with_one_to_spare_all_round() {
		let mesh = laid(bare(quad()));

		assert_eq!(mesh, quad(), "which is the built-in quad as it is built");
		assert_eq!(mesh.sheet, [7, 7], "five texels and the gutter");
		assert_eq!(mesh.vertices.len(), 4, "nothing copied");

		for (vertex, entry) in mesh.vertices.iter().zip(&mesh.paint) {
			let wanted = vertex.uv.map(|each| {
				let spread = 5.0 * f64::from(each);

				narrowed((1.0 + spread) / 7.0)
			});

			assert_eq!(
				entry.uv2.map(f32::to_bits),
				wanted.map(f32::to_bits),
				"the first set moved a texel in and spread over five: {:?}",
				entry.uv2
			);
			assert_eq!(entry.color, [PaintVertex::WHOLE; 4], "and painted white");
		}

		assert_laid_out(&mesh, "the quad");
	}

	#[test]
	fn a_cube_is_six_charts_none_within_two_texels_of_another() {
		let mesh = laid(bare(cube()));

		assert_eq!(mesh, cube(), "which is the built-in cube as it is built");
		assert_eq!(charts_of(&mesh).len(), 6, "a face a chart");
		assert_eq!(mesh.sheet, [14, 21], "two faces across and three down, no texel wasted");
		assert_eq!(mesh.vertices.len(), 24, "and nothing copied: no vertex is two faces'");
		assert_laid_out(&mesh, "the cube");
	}

	#[test]
	fn a_face_stretched_four_to_one_is_laid_four_to_one() {
		// a block's face the bake gave the whole of the first set whatever the
		// block's size: one number for both ways would have laid it square
		let mesh = laid(slab(4.0, 1.0));

		assert_eq!(mesh.sheet, [22, 7], "twenty texels by five, and the gutter");
		assert_laid_out(&mesh, "a four by one face");

		let (low, high) = mesh.vertices.iter().enumerate().fold(
			([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
			|(low, high), (vertex, _)| {
				let spot = on_sheet(&mesh, vertex);

				([low[0].min(spot[0]), low[1].min(spot[1])], [
					high[0].max(spot[0]),
					high[1].max(spot[1]),
				])
			},
		);

		assert!(
			(high[0] - low[0] - 20.0).abs() < 1.0e-4 && (high[1] - low[1] - 5.0).abs() < 1.0e-4,
			"four units long at five a unit, one deep: {low:?}..{high:?}"
		);

		for triangle in 0..mesh.triangles() {
			let corners = corners_of(&mesh, triangle);

			for (from, to) in [(0, 1), (1, 2), (2, 0)] {
				let (one, other) = (corners[from], corners[to]);
				let along =
					length(minus(placed(&mesh, other, [1.0; 3]), placed(&mesh, one, [1.0; 3])));
				let on = minus2(on_sheet(&mesh, other), on_sheet(&mesh, one));
				let texels = dot2(on, on).sqrt();

				assert!(
					(texels - along * 5.0).abs() < 1.0e-3,
					"every edge five texels a unit whichever way it runs: {along} units, \
					 {texels} texels"
				);
			}
		}
	}

	#[test]
	fn a_chart_taller_than_wide_is_turned_a_quarter_rather_than_mirrored() {
		let mesh = laid(slab(1.0, 4.0));

		assert_eq!(mesh.sheet, [22, 7], "laid wide like the other");
		assert_laid_out(&mesh, "a one by four face");

		let spots: Vec<[f64; 2]> = (0..mesh.vertices.len())
			.map(|vertex| on_sheet(&mesh, vertex))
			.collect();

		for triangle in 0..mesh.triangles() {
			let corners = corners_of(&mesh, triangle);
			let before = twice_area(corners.map(|vertex| first(&mesh, vertex)));
			let after = twice_area(corners.map(|vertex| spots[vertex]));

			assert!(
				before * after > 0.0,
				"a turn keeps the way a triangle is wound, a mirror would not: {before} and \
				 {after}"
			);
		}
	}

	#[test]
	fn the_scale_a_mesh_stands_at_is_the_scale_it_is_laid_at() {
		let mut mesh = bare(quad());

		second(&mut mesh, Vec3::new(2.0, 1.0, 3.0));

		assert_eq!(mesh.sheet, [17, 12], "ten texels by fifteen, laid wide, and the gutter");

		let mut shrunk = bare(quad());

		second(&mut shrunk, Vec3::new(-0.25, 1.0, -0.25));

		assert_eq!(
			shrunk.sheet,
			[4, 4],
			"a mirror counts as its size, and a quarter of a unit is a texel and a quarter, two 			 whole ones"
		);
	}

	#[test]
	fn islands_join_across_an_edge_where_only_the_normals_part() {
		// two faces of a box meeting at a right angle, split at the edge only
		// because each has its own normal: the first set runs straight on
		let wall = |x0: f32, x1: f32, u0: f32, u1: f32, up: bool| {
			let (a, b) = if up {
				(Vec3::new(x0, 0.0, 0.0), Vec3::new(x1, 0.0, 0.0))
			} else {
				(Vec3::new(x0, 0.0, 0.0), Vec3::new(x0, -(x1 - x0), 0.0))
			};
			let (c, d) = (a + Vec3::Z, b + Vec3::Z);

			[
				(a, Vec2::new(u0, 0.0)),
				(d, Vec2::new(u1, 1.0)),
				(b, Vec2::new(u1, 0.0)),
				(a, Vec2::new(u0, 0.0)),
				(c, Vec2::new(u0, 1.0)),
				(d, Vec2::new(u1, 1.0)),
			]
		};
		let mut corners = wall(0.0, 1.0, 0.0, 1.0, true).to_vec();

		corners.extend(wall(1.0, 2.0, 1.0, 2.0, false));

		let mesh = laid(soup(&corners));

		assert_eq!(charts_of(&mesh).len(), 1, "one island around the corner");
		assert_eq!(mesh.sheet, [12, 7], "two units long and one deep");
		assert_laid_out(&mesh, "a bent strip");
	}

	#[test]
	fn a_fold_in_the_first_set_is_two_charts_and_what_they_share_is_copied() {
		// two triangles sharing an edge by index, the second's far corner laid
		// on the same side of that edge as the first's: a fold
		let mut mesh = MeshData {
			vertices: vec![
				MeshVertex::new(Vec3::ZERO, Vec3::Y, Vec2::new(0.0, 0.0)),
				MeshVertex::new(Vec3::X, Vec3::Y, Vec2::new(1.0, 0.0)),
				MeshVertex::new(Vec3::Z, Vec3::Y, Vec2::new(0.0, 1.0)),
				MeshVertex::new(Vec3::new(1.0, 0.0, -1.0), Vec3::Y, Vec2::new(0.5, 1.0)),
			],
			indices: vec![0, 2, 1, 1, 3, 0],
			..MeshData::default()
		};

		mesh.paint = vec![PaintVertex::new(Vec4::new(0.25, 0.5, 0.75, 1.0), Vec2::ZERO); 4];

		let before = mesh.clone();

		second(&mut mesh, Vec3::ONE);

		assert_eq!(charts_of(&mesh).len(), 2, "never joined across a fold");
		assert_eq!(mesh.vertices.len(), 6, "the two corners on the fold are copied");

		for (copy, original) in [(4, 1), (5, 0)] {
			assert_eq!(mesh.vertices[copy], before.vertices[original], "{copy} is {original}");
			assert_eq!(
				mesh.paint[copy].color, before.paint[original].color,
				"and wears its paint"
			);
		}

		assert_eq!(
			&mesh.vertices[..4],
			before.vertices.as_slice(),
			"and nothing that was there moved"
		);
		assert_laid_out(&mesh, "a fold");
	}

	#[test]
	fn a_fold_at_the_end_of_a_strip_parts_the_fold_rather_than_cutting_the_strip_up() {
		// four squares in a row, their first set running on, and a triangle
		// hung off the far end whose far corner is laid back over the last
		// square: joined, the strip would land on itself and be cut into nine
		let mut mesh = MeshData::default();

		for step in 0..=4_u16 {
			let along = f32::from(step);

			mesh.vertices.extend([
				MeshVertex::new(Vec3::new(along, 0.0, 0.0), Vec3::Y, Vec2::new(along, 0.0)),
				MeshVertex::new(Vec3::new(along, 0.0, 1.0), Vec3::Y, Vec2::new(along, 1.0)),
			]);
		}

		for step in 0..4_u32 {
			let (near, far) = (step * 2, step * 2 + 2);

			mesh.indices
				.extend([near, near + 1, far + 1, near, far + 1, far]);
		}

		mesh.vertices.push(MeshVertex::new(
			Vec3::new(5.0, 0.0, 0.5),
			Vec3::Y,
			Vec2::new(3.5, 0.5),
		));
		mesh.indices.extend([8, 9, 10]);

		let mesh = laid(mesh);

		assert_eq!(charts_of(&mesh).len(), 2, "the strip and the fold");
		assert_laid_out(&mesh, "a strip with a fold at its end");
	}

	#[test]
	fn a_surface_that_winds_past_where_it_began_is_cut_a_triangle_at_a_time() {
		let once = laid(spiral(1));

		assert_eq!(charts_of(&once).len(), 1, "once around is one chart");
		assert_laid_out(&once, "once around");

		let twice = laid(spiral(2));

		assert_eq!(charts_of(&twice).len(), 16, "twice around lands on itself and is cut");
		assert_laid_out(&twice, "twice around");

		// the same surface wound the other way in the first set
		let mut mirrored = spiral(2);

		for vertex in &mut mirrored.vertices {
			vertex.uv[0] = 1.0 - vertex.uv[0];
		}

		let mirrored = laid(mirrored);

		assert_eq!(charts_of(&mirrored).len(), 16, "and wound the other way, the same");
		assert_laid_out(&mirrored, "twice around, mirrored");
	}

	#[test]
	fn an_island_laid_out_askew_keeps_the_area_of_its_surface() {
		// a square unit whose first set is a parallelogram: sheared, so that
		// measuring each way alone overstates its area
		let mut mesh = slab(1.0, 1.0);

		for vertex in &mut mesh.vertices {
			let lean = vertex.uv[1] * 0.75;

			vertex.uv[0] += lean;
		}

		let mesh = laid(mesh);

		assert_eq!(charts_of(&mesh).len(), 1, "one chart");
		assert_laid_out(&mesh, "a sheared square");
	}

	#[test]
	fn a_sheet_with_no_first_set_seen_from_both_sides_is_two_charts() {
		// a square nobody unwrapped, drawn from above and again from below over
		// the same four corners: the two sides face opposite ways
		let mut mesh = slab(1.0, 1.0);

		for vertex in &mut mesh.vertices {
			vertex.uv = [0.0; 2];
		}

		mesh.indices = vec![0, 2, 1, 0, 3, 2, 0, 1, 2, 0, 2, 3];

		let mesh = laid(mesh);

		assert_eq!(charts_of(&mesh).len(), 2, "a chart a side, rather than one cut apart");
		assert_laid_out(&mesh, "a sheet seen from both sides");
	}

	#[test]
	fn a_corner_at_minus_nought_is_the_corner_at_nought() {
		let mut mesh = slab(1.0, 1.0);
		let base = u32::try_from(mesh.vertices.len()).expect("a small fixture");

		// a second square beside the first along x, sharing its edge at x = 1,
		// whose corners on that edge say z is minus nought
		mesh.vertices
			.extend(slab(1.0, 1.0).vertices.iter().map(|vertex| {
				let mut moved = *vertex;

				moved.position[0] += 1.0;
				moved.uv[0] += 1.0;

				if moved.position[2] == 0.0 {
					moved.position[2] = -0.0;
				}

				moved
			}));
		mesh.indices.extend(
			slab(1.0, 1.0)
				.indices
				.iter()
				.map(|index| index + base),
		);

		let mesh = laid(mesh);

		assert_eq!(charts_of(&mesh).len(), 1, "one strip two units long");
		assert_eq!(mesh.sheet, [12, 7], "ten texels by five, and the gutter");
	}

	#[test]
	fn triangles_with_no_first_set_join_by_their_place_alone() {
		// a square whose coordinates lie on a line in each triangle, and differ
		// between the two where they meet
		let mut mesh = slab(1.0, 1.0);

		mesh.vertices[0].uv = [0.0, 0.0];
		mesh.vertices[1].uv = [0.5, 0.5];
		mesh.vertices[2].uv = [0.25, 0.25];
		mesh.vertices[3].uv = [0.75, 0.75];

		let mut apart = mesh.clone();

		// the second triangle's own corners at the shared edge, somewhere else
		// on the line
		apart
			.vertices
			.extend([mesh.vertices[0], mesh.vertices[2]].map(|vertex| {
				let mut moved = vertex;

				moved.uv = [vertex.uv[0] * 0.5, vertex.uv[1] * 0.5];

				moved
			}));
		apart.indices = vec![0, 2, 1, 4, 3, 5];

		let apart = laid(apart);

		assert_eq!(charts_of(&apart).len(), 1, "one chart, found by place");
		assert_eq!(apart.sheet, [7, 7], "the square unit it is");
	}

	#[test]
	fn a_chart_wider_than_a_picture_is_laid_out_coarser() {
		let mesh = laid(slab(5000.0, 1.0));

		assert_eq!(
			mesh.sheet[0], 16_386,
			"no wider than the widest picture, and the gutter: {:?}",
			mesh.sheet
		);
		assert_eq!(
			mesh.sheet[1], 6,
			"and coarser across as well, by the same share: five texels are three and a quarter"
		);
	}

	#[test]
	fn a_vertex_a_chart_shares_is_copied_once_however_many_of_its_triangles_use_it() {
		// a pyramid nobody unwrapped over five vertices, its four sides first
		// and its base last: the base's two triangles share two corners the
		// sides already have, so the base copies each of them once and both of
		// its triangles use the copies
		let (a, b, c, d) = (
			Vec3::new(-0.5, 0.0, -0.5),
			Vec3::new(0.5, 0.0, -0.5),
			Vec3::new(0.5, 0.0, 0.5),
			Vec3::new(-0.5, 0.0, 0.5),
		);
		let top = Vec3::new(0.0, 1.0, 0.0);
		let mesh = MeshData {
			vertices: [a, b, c, d, top]
				.map(|at| MeshVertex::new(at, Vec3::Y, Vec2::ZERO))
				.to_vec(),
			indices: vec![0, 4, 1, 1, 4, 2, 2, 4, 3, 3, 4, 0, 0, 1, 2, 0, 2, 3],
			..MeshData::default()
		};
		let mesh = laid(mesh);

		assert_eq!(charts_of(&mesh).len(), 5, "four sides and the base");
		// the top is four sides' and each corner of the base three charts'
		assert_eq!(mesh.vertices.len(), 5 + 3 + 4 * 2, "a copy for every chart after the first");
		assert_laid_out(&mesh, "a pyramid");
	}

	#[test]
	fn a_mesh_with_no_first_set_is_laid_by_the_way_its_faces_face() {
		let mut blank = bare(cube());

		for vertex in &mut blank.vertices {
			vertex.uv = [0.0; 2];
		}

		let mesh = laid(blank);

		assert_eq!(charts_of(&mesh).len(), 6, "a chart for each way a face faces");
		assert_eq!(mesh.sheet, [14, 21], "each a face of the cube at five texels a unit");
		assert_laid_out(&mesh, "a cube with no coordinates");
	}

	#[test]
	fn a_triangle_the_first_set_collapsed_is_laid_by_its_facing_beside_the_rest() {
		let mut mesh = slab(1.0, 1.0);

		// a third triangle whose coordinates lie on a line
		mesh.vertices.extend([
			MeshVertex::new(Vec3::new(2.0, 0.0, 0.0), Vec3::Y, Vec2::new(0.0, 0.0)),
			MeshVertex::new(Vec3::new(3.0, 0.0, 1.0), Vec3::Y, Vec2::new(0.5, 0.5)),
			MeshVertex::new(Vec3::new(3.0, 0.0, 0.0), Vec3::Y, Vec2::new(1.0, 1.0)),
		]);
		mesh.indices.extend([4, 5, 6]);

		let mesh = laid(mesh);

		assert_eq!(charts_of(&mesh).len(), 2, "the square and the one laid by its facing");
		assert_laid_out(&mesh, "a collapsed triangle beside a square");
	}

	#[test]
	fn a_second_set_of_its_own_is_kept_and_given_a_square_sheet_by_its_area() {
		let mut mesh = bare(quad());

		mesh.paint = mesh
			.vertices
			.iter()
			.map(|vertex| PaintVertex::new(Vec4::ONE, Vec2::from_array(vertex.uv) * 0.5))
			.collect();

		let before = mesh.clone();

		second(&mut mesh, Vec3::ONE);

		assert_eq!(mesh.paint, before.paint, "the second set somebody made is theirs");
		assert_eq!(mesh.vertices, before.vertices, "and nothing is copied");
		assert_eq!(
			mesh.sheet,
			[10, 10],
			"a quarter of a sheet holding a square unit at five texels a unit is ten across"
		);
	}

	#[test]
	fn a_second_set_that_covers_nothing_is_no_second_set() {
		let mut mesh = bare(quad());

		mesh.paint = vec![PaintVertex::new(Vec4::new(1.0, 0.0, 0.0, 1.0), Vec2::ZERO); 4];

		second(&mut mesh, Vec3::ONE);

		assert_eq!(mesh.sheet, [7, 7], "a color alone is laid out like no paint at all");
		assert!(
			mesh.paint
				.iter()
				.all(|entry| entry.color == [PaintVertex::WHOLE, 0, 0, PaintVertex::WHOLE]),
			"and keeps its color"
		);
	}

	#[test]
	fn what_is_not_to_be_laid_out_is_left_as_it_was() {
		let mut skinned = bare(quad());

		skinned.skin = vec![SkinVertex::rigid(0); 4];

		let mut leveled = bare(cube());

		leveled.levels = vec![crate::abi::mesh::Level {
			indices: leveled.indices[..12].to_vec(),
			error: 0.5,
		}];

		let mut sheeted = bare(quad());

		sheeted.sheet = [9, 9];

		let mut partial = bare(quad());

		partial.indices.pop();

		for (label, mesh) in [
			("a mesh bones move", skinned),
			("a mesh with levels", leveled),
			("a mesh with a sheet", sheeted),
			("a mesh of part of a triangle", partial),
			("a mesh of nothing", MeshData::default()),
		] {
			let mut after = mesh.clone();

			second(&mut after, Vec3::ONE);

			assert_eq!(after, mesh, "{label}");
		}
	}

	#[test]
	fn a_copy_carries_everything_its_vertex_carries() {
		let mut mesh = quad();

		mesh.skin = vec![SkinVertex::rigid(3); 4];
		mesh.paint = vec![PaintVertex::new(Vec4::new(0.5, 0.25, 1.0, 1.0), Vec2::ONE); 4];

		let copy = duplicate(&mut mesh, 2);

		assert_eq!(copy, 4, "appended");
		assert_eq!(mesh.vertices[4], mesh.vertices[2], "the vertex");
		assert_eq!(mesh.skin[4], mesh.skin[2], "its skin");
		assert_eq!(mesh.paint[4], mesh.paint[2], "and its paint");
		assert!(mesh.skin_fits() && mesh.paint_fits(), "every block as long as the vertices");
	}

	#[test]
	fn laid_out_twice_is_laid_out_once() {
		let mut mesh = spiral(2);
		let base = u32::try_from(mesh.vertices.len()).expect("a small fixture");

		mesh.vertices.extend(slab(4.0, 1.0).vertices);
		mesh.indices.extend(
			slab(4.0, 1.0)
				.indices
				.iter()
				.map(|index| index + base),
		);

		let (mut one, mut other) = (mesh.clone(), mesh);

		second(&mut one, Vec3::ONE);
		second(&mut other, Vec3::ONE);

		assert_eq!(one, other, "the same answer twice");

		let again = one.clone();

		second(&mut one, Vec3::ONE);

		assert_eq!(one, again, "and a mesh already laid out is left alone");
	}

	#[test]
	fn a_point_on_an_edge_two_triangles_share_belongs_to_one_of_them() {
		let (a, b) = ([0, 0], [256, 256]);

		assert_ne!(
			inside(a, b, [128, 128]),
			inside(b, a, [128, 128]),
			"one of the two ways along an edge owns it"
		);

		let (left, right) = ([0, 0], [256, 0]);

		assert_ne!(
			inside(left, right, [128, 0]),
			inside(right, left, [128, 0]),
			"and so of an edge that runs level, where the other half of the rule decides"
		);

		// a square of four texels split corner to corner: every middle once
		let (low, right, high, left) = ([0, 0], [512, 0], [512, 512], [0, 512]);
		let mut covered = vec![0_u64; 1];

		assert!(!drawn_twice([low, right, high], [2, 2], &mut covered), "the first half");
		assert!(!drawn_twice([low, high, left], [2, 2], &mut covered), "and the second");
		assert_eq!(covered[0].count_ones(), 4, "four middles, each covered once");
		assert!(
			drawn_twice([low, right, left], [2, 2], &mut covered),
			"and a third lands on them"
		);
	}

	#[test]
	fn the_packer_keeps_every_cell_apart_and_inside_its_sheet() {
		let cells: Vec<[u32; 2]> = (0..60_u32)
			.map(|at| [3 + (at * 7) % 13, 3 + (at * 5) % 9])
			.collect();
		let (corners, sheet) = packed(&cells);
		let area: u64 = cells
			.iter()
			.map(|cell| u64::from(cell[0]) * u64::from(cell[1]))
			.sum();

		assert!(u64::from(sheet[0]) * u64::from(sheet[1]) >= area, "{sheet:?} holds {area}");

		for (at, (corner, cell)) in corners.iter().zip(&cells).enumerate() {
			assert!(
				corner[0] + cell[0] <= sheet[0] && corner[1] + cell[1] <= sheet[1],
				"cell {at} at {corner:?} inside {sheet:?}"
			);

			for (other_corner, other_cell) in corners.iter().zip(&cells).skip(at + 1) {
				let apart = corner[0] + cell[0] <= other_corner[0]
					|| other_corner[0] + other_cell[0] <= corner[0]
					|| corner[1] + cell[1] <= other_corner[1]
					|| other_corner[1] + other_cell[1] <= corner[1];

				assert!(apart, "cell {at} at {corner:?} overlaps one at {other_corner:?}");
			}
		}
	}

	#[test]
	fn the_tallest_cell_goes_down_first_in_the_corner() {
		let (corners, _) = packed(&[[3, 3], [9, 9], [5, 4]]);

		assert_eq!(corners[1], [0, 0], "the tallest in the corner: {corners:?}");
	}

	#[test]
	fn the_lowest_place_is_the_leftmost_of_the_lowest() {
		assert_eq!(lowest(&[3, 1, 1, 2, 1, 1], 2), (1, 1), "the first two ones side by side");
		assert_eq!(
			lowest(&[3, 1, 2, 1, 1, 4], 3),
			(1, 2),
			"the tallest under it counts, and of two as low the leftmost"
		);
		assert_eq!(lowest(&[0, 0, 0], 3), (0, 0), "a cell as wide as the sheet");
	}

	#[test]
	fn a_stretch_of_ground_is_laid_on_a_square_a_texel_in_from_every_side() {
		assert_eq!(square(1.0), 7, "five texels and the gutter");
		assert_eq!(square(0.3), 4, "a texel and a half rounds up to two");
		assert_eq!(onto(0.0, 7).to_bits(), (1.0_f32 / 7.0).to_bits(), "a texel in");
		assert_eq!(onto(1.0, 7).to_bits(), (6.0_f32 / 7.0).to_bits(), "and a texel short");
	}

	/// FNV-1a over the words given.
	fn digest(numbers: impl Iterator<Item = u32>) -> u64 {
		numbers.fold(0xCBF2_9CE4_8422_2325, |held, number| {
			number
				.to_le_bytes()
				.iter()
				.fold(held, |held, byte| (held ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01B3))
		})
	}

	/// Every word of a mesh's second set: its sheet, its indices, and every
	/// vertex's second coordinates.
	fn words(mesh: &MeshData) -> impl Iterator<Item = u32> + '_ {
		mesh.sheet
			.into_iter()
			.chain(mesh.indices.iter().copied())
			.chain(
				mesh.paint
					.iter()
					.flat_map(|entry| entry.uv2.map(f32::to_bits)),
			)
	}

	#[test]
	fn the_second_set_is_the_same_bytes_on_every_machine() {
		let ground = Terrain {
			kind: TerrainKind::Noise,
			size: 12.0,
			side: 9,
			..Terrain::NONE
		}
		.build();
		let mut blank = bare(cube());

		for vertex in &mut blank.vertices {
			vertex.uv = [0.0; 2];
		}

		let laid_out = [
			cube(),
			quad(),
			laid(slab(4.0, 1.0)),
			laid(slab(1.0, 4.0)),
			laid(spiral(1)),
			laid(spiral(2)),
			laid(blank),
		];
		// the ball's place is not the same bits everywhere, its second set is
		let answer = digest(
			laid_out
				.iter()
				.chain([&ground])
				.flat_map(words)
				.chain(
					sphere()
						.paint
						.iter()
						.flat_map(|entry| entry.uv2.map(f32::to_bits)),
				)
				.chain(sphere().sheet),
		);

		// written down from the first run on one machine; another machine
		// answering anything else lays the same mesh out differently there
		assert_eq!(
			answer, 0x146B_A3F0_50AE_4E0B,
			"the digest of every second set: {answer:#018x}"
		);
	}
}
