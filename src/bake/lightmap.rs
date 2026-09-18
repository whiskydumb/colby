//! The world's still light, worked out into one picture.
//!
//! ```text
//!   bake(&scene, settings, threads)    the picture, and each thing's place on it
//! ```
//!
//! **What the picture holds is the light arriving at a surface that the frame
//! does not draw for itself**: the sky a surface under a roof still sees, what
//! glows, and what a lit wall throws onto the floor beside it - in the unit
//! [`World::ambient`](colby_core::abi::World::ambient) has always meant, so the
//! picture stands where that color stood. The sun and the lamps shining
//! straight onto a surface are not in it: the frame draws them, with their own
//! shadows, every frame.
//!
//! In order, each worked out on every texel before the next begins:
//!
//! 1. **Where each texel is**: the point of the surface it stands for, @ref
//!    [`texels`](crate::texels), pushed out past a back face met within a texel
//!    of it. A texel half under a wall stands for the floor beside the wall
//!    rather than the floor inside it.
//! 2. **The direct light**, the sun's and the lamps', at every texel - not kept
//!    in the picture, only handed on by what it lights.
//! 3. **Gathers, as many as [`Settings::bounces`]**, each the average of what
//!    the rays from a texel bring back: the sky where a ray leaves the world,
//!    and where it lands, what glows there and what the surface there sends on
//!    of the light the *last* gather found arriving at it, read off the picture
//!    at the point it landed - so every gather reads the one before and never
//!    itself, and the order the texels are worked in changes nothing. Three
//!    gathers carry a lamp's light off three surfaces and the sky's off two. A
//!    texel more than half of whose first gather's rays meet the back of
//!    something is inside it, and its light is left to its neighbors.
//! 4. **The picture filled** between gathers and at the end: every texel a
//!    picture reading a chart can reach takes its value from that chart's
//!    texels and no other's, and the rest of each thing's place from whatever
//!    is beside it, so a coarser level of a mesh drawn over its chart's gutter
//!    reads a color rather than black.

use std::time::{Duration, Instant};

use colby_core::{
	Result,
	abi::{EntityId, MeshId},
	err,
	glam::Vec3,
	unwrap,
};

use crate::{
	atlas::{Atlas, Placeless, Rect},
	gather::{Gathered, Pattern, each, seed},
	scene::{BIAS, Scene},
	texels::{NONE, Sample, Texels},
	tree::{Hit, Ray},
};

/// What a bake is asked for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
	/// How many texels a unit of surface is given, at the least; no thing is
	/// given fewer than its sheet was laid out with.
	pub texels: f32,

	/// How many gathers: a lamp's light is carried off that many surfaces and
	/// the sky's off one fewer.
	pub bounces: u32,

	/// How many rays a texel sends in each gather.
	pub rays: u32,
}

impl Settings {
	/// What a bake is asked for when nobody says otherwise: the density the
	/// sheets were laid out at, three gathers, and a hundred and twenty-eight
	/// rays a texel.
	pub const DEFAULT: Self = Self {
		texels: unwrap::TEXELS,
		bounces: 3,
		rays: 128,
	};

	/// The same, each held where a bake means something: a tenth of a texel a
	/// unit to sixty-four, one gather to sixteen, sixteen rays to 8192.
	#[must_use]
	pub fn sane(self) -> Self {
		let texels = if self.texels.is_nan() {
			Self::DEFAULT.texels
		} else {
			self.texels.clamp(0.1, 64.0)
		};

		Self {
			texels,
			bounces: self.bounces.clamp(1, 16),
			rays: self.rays.clamp(16, 8192),
		}
	}
}

impl Default for Settings {
	fn default() -> Self { Self::DEFAULT }
}

/// The picture, and where each thing's light is on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Baked {
	/// How many texels across.
	pub width: u32,

	/// How many down.
	pub height: u32,

	/// The light arriving at each texel, row by row from the top, in the
	/// ambient color's unit; black where nothing is.
	pub light: Vec<Vec3>,

	/// Every still thing that has a place, and the place.
	pub places: Vec<(EntityId, Rect)>,

	/// Every still thing that has none, the mesh it draws, and why.
	pub placeless: Vec<(EntityId, MeshId, Placeless)>,

	/// What was worked out, and how long each part took.
	pub report: Report,
}

/// What a bake did, for a person to read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
	/// How many still things.
	pub pieces: usize,

	/// How many triangles they are.
	pub triangles: usize,

	/// How many lamps stand still.
	pub lamps: usize,

	/// How many texels stand for a point of a surface.
	pub samples: usize,

	/// How many more a picture reads, filled from their own chart.
	pub ring: usize,

	/// How many samples were pushed out past a back face.
	pub pushed: usize,

	/// How many were found inside something and left to their neighbors.
	pub buried: usize,

	/// How many rays the gathers sent.
	pub rays: u64,

	/// Each part, and how long it took.
	pub parts: Vec<(String, Duration)>,
}

/// One texel's point of the surface, where its rays start from.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Spot {
	/// Where.
	at: Vec3,

	/// Which way the surface faces there.
	normal: Vec3,

	/// Whether it was pushed out past a back face.
	pushed: bool,
}

/// Works out a world's still light.
///
/// @param scene - what stands still
/// @param settings - what is asked for, held where it means something
/// @param threads - how many threads at most; the answer does not depend on it
/// @return the picture and every place on it, or why there is none
///
/// # Errors
///
/// If nothing still has a place, or the places do not fit on one picture.
pub fn bake(scene: &Scene, settings: Settings, threads: usize) -> Result<Baked> {
	let settings = settings.sane();
	let mut clock = Clock::default();
	let atlas = Atlas::of(scene, settings.texels)?;

	if atlas.is_empty() {
		return Err(err!(Asset(
			"nothing still has a place to keep light in: a bake needs a shown mesh no moving \
			 body drives, with a second set of coordinates, drawn lit"
		)));
	}

	let texels = Texels::of(scene, &atlas);
	clock.lap("the places and the texels");

	let samples = texels.samples();
	let spots = each(samples.len(), threads, |index| spot_of(scene, &samples[index]));
	clock.lap("where each texel is");

	let direct =
		each(spots.len(), threads, |index| scene.direct(spots[index].at, spots[index].normal));
	clock.lap("the sun and the lamps");

	let pattern = Pattern::new(settings.rays);
	let mut valid = vec![true; samples.len()];
	let mut arriving = picture(&texels, &direct, &valid, false);
	let mut light = Vec::new();
	let mut buried = 0;

	for pass in 0..settings.bounces {
		let read = Reading { scene, atlas: &atlas, picture: &arriving };
		let gathered =
			each(samples.len(), threads, |index| {
				if !valid[index] {
					return None;
				}

				let at = usize::try_from(samples[index].at).unwrap_or(usize::MAX);
				let spot = spots[index];

				Some(scene.gather(spot.at, spot.normal, seed(at, pass), &pattern, |hit, _| {
					read.sent(hit)
				}))
			});

		if pass == 0 {
			buried = bury(&mut valid, &gathered);
		}

		let found: Vec<Vec3> = gathered
			.iter()
			.map(|found| found.map_or(Vec3::ZERO, |found| found.light))
			.collect();

		light = picture(&texels, &found, &valid, pass + 1 == settings.bounces);

		if pass + 1 < settings.bounces {
			let both: Vec<Vec3> = direct
				.iter()
				.zip(&found)
				.map(|(direct, found)| *direct + *found)
				.collect();

			arriving = picture(&texels, &both, &valid, false);
		}

		clock.lap(&if pass == 0 {
			"the sky and the first bounce".to_owned()
		} else {
			format!("bounce {}", pass + 1)
		});
	}

	let (places, placeless) = places_of(scene, &atlas);
	let lit = valid.iter().filter(|keep| **keep).count();

	Ok(Baked {
		width: atlas.width(),
		height: atlas.height(),
		light,
		places,
		placeless,
		report: Report {
			pieces: scene.pieces().len(),
			triangles: scene.triangles().len(),
			lamps: scene.lamps().len(),
			samples: samples.len(),
			ring: texels.ring(),
			pushed: spots.iter().filter(|spot| spot.pushed).count(),
			buried,
			rays: u64::try_from(
				samples.len()
					+ lit * usize::try_from(settings.bounces.saturating_sub(1)).unwrap_or(0),
			)
			.unwrap_or(u64::MAX)
			.saturating_mul(u64::try_from(pattern.len()).unwrap_or(0)),
			parts: clock.parts,
		},
	})
}

/// Marks every sample more than half of whose first gather's rays met the
/// back of something as inside it.
///
/// @return how many were
fn bury(valid: &mut [bool], gathered: &[Option<Gathered>]) -> usize {
	let mut buried = 0;

	for (keep, found) in valid.iter_mut().zip(gathered) {
		let inside = found.is_some_and(|found| found.behind.saturating_mul(2) > found.rays);

		*keep &= !inside;
		buried += usize::from(inside);
	}

	buried
}

/// Every still thing that has a place and the place, and every one that has
/// none with the mesh it draws and why.
type Places = (Vec<(EntityId, Rect)>, Vec<(EntityId, MeshId, Placeless)>);

/// [`Places`], out of the scene and its atlas.
fn places_of(scene: &Scene, atlas: &Atlas) -> Places {
	let pieces = scene.pieces();
	let places = atlas
		.places()
		.iter()
		.zip(pieces)
		.filter_map(|(place, piece)| place.map(|place| (piece.entity, place)))
		.collect();
	let placeless = atlas
		.placeless()
		.iter()
		.filter_map(|&(index, why)| {
			let piece = pieces.get(usize::try_from(index).ok()?)?;

			Some((piece.entity, piece.mesh, why))
		})
		.collect();

	(places, placeless)
}

/// How long each part of a bake took.
#[derive(Debug)]
struct Clock {
	since: Instant,
	parts: Vec<(String, Duration)>,
}

impl Default for Clock {
	fn default() -> Self { Self { since: Instant::now(), parts: Vec::new() } }
}

impl Clock {
	/// Writes down the part that just finished, and starts the next.
	fn lap(&mut self, part: &str) {
		let now = Instant::now();

		self.parts
			.push((part.to_owned(), now - self.since));
		self.since = now;
	}
}

/// Where a texel's rays start from, and which way.
///
/// The point of the surface the texel stands for, pushed out past a back face
/// if one of four rays toward the texel's corners, each as long as the texel's
/// diagonal, meets one: the point is then inside something that covers part of
/// the texel, and the nearest such face is where the texel's surface comes out
/// from under it. The push is along the ray, so the point stays on the
/// texel's own surface.
fn spot_of(scene: &Scene, sample: &Sample) -> Spot {
	let Some(surface) = scene.surface_at(sample.triangle, sample.along, sample.across) else {
		return Spot {
			at: Vec3::ZERO,
			normal: Vec3::Y,
			pushed: false,
		};
	};
	let face = face_of(scene, sample.triangle).unwrap_or(surface.normal);
	let at = pulled_in(scene, sample.triangle, surface.at);
	let start = at + face * BIAS;
	let [along, down] = sample.reach;
	let mut nearest: Option<(f32, Vec3)> = None;

	for way in [along + down, along - down, down - along, -along - down] {
		let reach = way.length();

		if reach.is_nan() || reach <= 0.0 {
			continue;
		}

		let ray = Ray::new(start, way / reach);
		let Some(hit) = scene.tree().nearest(&ray, reach) else {
			continue;
		};

		if !hit.front && nearest.is_none_or(|(distance, _)| hit.distance < distance) {
			nearest = Some((hit.distance, ray.direction));
		}
	}

	nearest.map_or(
		Spot {
			at,
			normal: surface.normal,
			pushed: false,
		},
		|(distance, way)| Spot {
			at: at + way * (distance + BIAS),
			normal: surface.normal,
			pushed: true,
		},
	)
}

/// A point of a triangle moved a little way in from its edges: toward the
/// triangle's middle by [`BIAS`], or half the way there when the middle is
/// nearer than twice that.
///
/// A texel along a chart's edge stands for a point on the edge, and where two
/// surfaces meet in a corner - a floor and a wall - that point is on both. A
/// ray that starts on a surface meets it at a distance of nought, which is no
/// meeting, so half of what such a point gathers would go out through the wall
/// it stands on; a point a hair inside its own triangle is a hair off the
/// wall, and the wall is in the way of every ray that should find it.
fn pulled_in(scene: &Scene, triangle: u32, at: Vec3) -> Vec3 {
	let Some(corners) = corners_of(scene, triangle) else {
		return at;
	};
	let middle = (corners[0] + corners[1] + corners[2]) / 3.0;
	let toward = middle - at;
	let far = toward.length();

	if far.is_nan() || far <= 0.0 {
		return at;
	}

	at + toward * (BIAS.min(far * 0.5) / far)
}

/// Which way a triangle's flat face looks, from its winding.
fn face_of(scene: &Scene, triangle: u32) -> Option<Vec3> {
	let [one, two, three] = corners_of(scene, triangle)?;

	(two - one).cross(three - one).try_normalize()
}

/// Where a triangle's three corners are in the world.
fn corners_of(scene: &Scene, triangle: u32) -> Option<[Vec3; 3]> {
	let corners = scene
		.triangles()
		.get(usize::try_from(triangle).ok()?)?
		.map(|index| {
			usize::try_from(index)
				.ok()
				.and_then(|index| scene.corners().get(index))
				.map(|corner| corner.position)
		});

	Some([corners[0]?, corners[1]?, corners[2]?])
}

/// What a gather reads where its rays land: the scene, where each thing's
/// light is, and the picture the last gather left.
struct Reading<'a> {
	scene: &'a Scene,
	atlas: &'a Atlas,
	picture: &'a [Vec3],
}

impl Reading<'_> {
	/// What the surface a ray landed on sends back along it: what it gives
	/// off, and its share of the light arriving there.
	///
	/// The light arriving is read off the picture at the point the ray landed,
	/// between the four texels around it the way a picture is read; a thing
	/// with no place on the picture is lit where the ray landed by the sun and
	/// the lamps alone, traced there, since it keeps no light of its own.
	fn sent(&self, hit: &Hit) -> Vec3 {
		let Some(surface) = self.scene.surface(hit) else {
			return Vec3::ZERO;
		};

		if surface.diffuse == Vec3::ZERO {
			return surface.emission;
		}

		let arriving = self.atlas.place(surface.piece).map_or_else(
			|| self.scene.direct(surface.at, surface.normal),
			|place| {
				read(
					self.picture,
					self.atlas.width(),
					self.atlas.height(),
					place.place(surface.uv2),
				)
			},
		);

		surface.emission + surface.diffuse * arriving
	}
}

/// A picture read at a point between texels, the way a picture is read: the
/// four texels whose middles are nearest, blended by how near, held at the
/// picture's edge.
///
/// Blended one axis at a time as `a + (b - a) t`, so four texels of one value
/// read as that value to the bit.
///
/// @param picture - the texels, row by row
/// @param width - how many across
/// @param height - how many down
/// @param at - where, in texels from the top left corner
fn read(picture: &[Vec3], width: u32, height: u32, at: [f64; 2]) -> Vec3 {
	let [x, y] = at.map(|place| place - 0.5);
	let [column, row] = [x, y].map(f64::floor);
	let [right, down] = [narrowed(x - column), narrowed(y - row)];
	let texel = |across: f64, along: f64| {
		let column = held(across, width);
		let row = held(along, height);

		usize::try_from(u64::from(row) * u64::from(width) + u64::from(column))
			.ok()
			.and_then(|at| picture.get(at))
			.copied()
			.unwrap_or(Vec3::ZERO)
	};
	let lerp = |from: Vec3, to: Vec3, part: f32| from + (to - from) * part;
	let top = lerp(texel(column, row), texel(column + 1.0, row), right);
	let bottom = lerp(texel(column, row + 1.0), texel(column + 1.0, row + 1.0), right);

	lerp(top, bottom, down)
}

/// A column or a row as a texel of the picture, held at its edge.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "a whole number held between nought and the last texel on the line above the cast"
)]
fn held(place: f64, side: u32) -> u32 {
	let last = f64::from(side.saturating_sub(1));
	let place = if place.is_nan() { 0.0 } else { place.clamp(0.0, last) };

	place as u32
}

/// The picture of a value at every sample, filled.
///
/// Every sample that holds keeps its value; every other texel of a thing's
/// place and the texel round it takes the average of the four beside it that
/// already have one - of its own chart where it has one - or failing those,
/// of the four at its corners, a ring at a time outwards from what holds. A
/// texel of a chart none of whose texels hold, which is a chart wholly inside
/// something, takes what is beside it whatever it is.
///
/// A picture a gather reads is filled only as far as a read can reach, which
/// is the texels of a chart; the gutters between charts are filled once, in
/// the picture that is kept.
///
/// @param texels - what every texel stands for
/// @param values - one a sample
/// @param valid - which samples hold
/// @param whole - whether to fill every texel of every place, or only those
/// of a chart
fn picture(texels: &Texels, values: &[Vec3], valid: &[bool], whole: bool) -> Vec<Vec3> {
	let count =
		usize::try_from(u64::from(texels.width()) * u64::from(texels.height())).unwrap_or(0);
	let mut picture = vec![Vec3::ZERO; count];
	let mut known = vec![false; count];

	for ((sample, value), keep) in texels.samples().iter().zip(values).zip(valid) {
		let Ok(at) = usize::try_from(sample.at) else {
			continue;
		};

		if *keep && let (Some(texel), Some(mark)) = (picture.get_mut(at), known.get_mut(at)) {
			*texel = *value;
			*mark = true;
		}
	}

	let mut fill = Fill {
		texels,
		picture: &mut picture,
		known: &mut known,
		loose: false,
		whole,
	};

	fill.spread();

	if whole {
		fill.loose = true;
		fill.spread();
	}

	picture
}

/// A picture being filled outwards from the texels that hold.
struct Fill<'a> {
	texels: &'a Texels,
	picture: &'a mut [Vec3],
	known: &'a mut [bool],
	/// Whether a texel of a chart may take what is beside it whatever chart
	/// that is: once every chart has spread as far as it reaches.
	loose: bool,
	/// Whether every texel of every place is filled, or only those of a chart.
	whole: bool,
}

/// The four texels beside one, then the four at its corners.
const BESIDE: [[i64; 2]; 4] = [[0, -1], [0, 1], [-1, 0], [1, 0]];

/// And the four at its corners.
const CORNERS: [[i64; 2]; 4] = [[-1, -1], [1, -1], [-1, 1], [1, 1]];

impl Fill<'_> {
	/// Fills a ring at a time until nothing more can be.
	fn spread(&mut self) {
		let mut ring: Vec<usize> = (0..self.known.len())
			.filter(|&at| self.wants(at) && self.worked(at).is_some())
			.collect();

		while !ring.is_empty() {
			let filled: Vec<(usize, Vec3)> = ring
				.iter()
				.filter_map(|&at| self.worked(at).map(|value| (at, value)))
				.collect();

			for &(at, value) in &filled {
				self.picture[at] = value;
				self.known[at] = true;
			}

			let mut next: Vec<usize> = filled
				.iter()
				.flat_map(|&(at, _)| {
					self.around(at, &BESIDE)
						.chain(self.around(at, &CORNERS))
				})
				.filter(|&at| self.wants(at))
				.collect();

			next.sort_unstable();
			next.dedup();
			ring = next
				.into_iter()
				.filter(|&at| self.worked(at).is_some())
				.collect();
		}
	}

	/// Whether a texel is in some thing's place, or round it, has no value, and
	/// is one this fill fills.
	fn wants(&self, at: usize) -> bool {
		!self.known[at]
			&& self.texels.owner(at) != NONE
			&& (self.whole || self.texels.chart(at) != NONE)
	}

	/// The average of the texels beside one that it may take a value from, or
	/// of those at its corners if none beside it may.
	fn worked(&self, at: usize) -> Option<Vec3> {
		[BESIDE, CORNERS].iter().find_map(|ways| {
			let (total, count) = self
				.around(at, ways)
				.filter(|&from| self.gives(at, from))
				.fold(([0.0_f64; 3], 0_u32), |(total, count), from| {
					let [red, green, blue] = self.picture[from].to_array();

					(
						[
							total[0] + f64::from(red),
							total[1] + f64::from(green),
							total[2] + f64::from(blue),
						],
						count + 1,
					)
				});

			(count > 0)
				.then(|| Vec3::from_array(total.map(|sum| narrowed(sum / f64::from(count)))))
		})
	}

	/// Whether one texel may take its value from another: one that has a value,
	/// in the same thing's place, and of the same chart unless the fill is
	/// loose or the one taking has no chart.
	fn gives(&self, at: usize, from: usize) -> bool {
		let chart = self.texels.chart(at);

		self.known[from]
			&& self.texels.owner(from) == self.texels.owner(at)
			&& (self.loose || chart == NONE || self.texels.chart(from) == chart)
	}

	/// The texels a set of steps away from one, inside the picture.
	fn around<'s>(
		&'s self,
		at: usize,
		ways: &'s [[i64; 2]; 4],
	) -> impl Iterator<Item = usize> + 's {
		let width = i64::from(self.texels.width());
		let height = i64::from(self.texels.height());
		let (column, row) = (
			i64::try_from(at).unwrap_or(0) % width.max(1),
			i64::try_from(at).unwrap_or(0) / width.max(1),
		);

		ways.iter().filter_map(move |[across, down]| {
			let (x, y) = (column + across, row + down);

			((0..width).contains(&x) && (0..height).contains(&y))
				.then(|| usize::try_from(y * width + x).ok())
				.flatten()
		})
	}
}

/// A double narrowed to a float, rounding to the nearest.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "an average or a fraction worked out in double precision and narrowed once, where \
	          it is kept"
)]
const fn narrowed(value: f64) -> f32 { value as f32 }

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			BAKING, Renderable, Transform, World,
			light::Light,
			material::{Material, MaterialId},
			mesh::{MeshData, cube},
		},
		glam::Quat,
	};

	use super::*;
	use crate::gather::threads;

	/// The unit cube grown and turned inside out: a room seen from within.
	fn inside_out(half: f32) -> MeshData {
		let mut mesh = cube();

		for vertex in &mut mesh.vertices {
			vertex.position = (Vec3::from(vertex.position) * (half * 2.0)).to_array();
			vertex.normal = (-Vec3::from(vertex.normal)).to_array();
		}

		for triangle in mesh.indices.chunks_exact_mut(3) {
			triangle.swap(1, 2);
		}

		mesh
	}

	/// A world with no sun and one flat sky.
	fn open(sky: Vec3) -> World {
		let mut world = World::new();

		world.light = Vec3::ZERO;
		world.ambient = sky;

		world
	}

	/// A thing drawing a mesh with a material, standing somewhere.
	fn put(world: &mut World, mesh: MeshId, material: MaterialId, at: Transform) -> EntityId {
		let id = world.entities.spawn_at(at);

		world
			.entities
			.set_renderable(id, Renderable::of(mesh, material, Vec3::ONE));

		id
	}

	/// A room of glowing walls, all of one material, and nothing else.
	fn furnace(glow: Vec3, albedo: Vec3) -> World {
		let mut world = open(Vec3::splat(9.0));
		let room = world.meshes.insert("room", inside_out(2.0));
		let walls = world.materials.insert("walls", Material {
			base_color: albedo,
			emissive: glow,
			..Material::DEFAULT
		});

		put(&mut world, room, walls, Transform::IDENTITY);

		world
	}

	/// Every texel of the picture that is in a place or round it.
	fn placed(baked: &Baked) -> Vec<usize> {
		let width = usize::try_from(baked.width).expect("a small picture");

		(0..baked.light.len())
			.filter(|&at| {
				let (column, row) = (at % width, at / width);

				baked.places.iter().any(|(_, place)| {
					let [left, top, right, bottom] = [
						place.left.saturating_sub(1),
						place.top.saturating_sub(1),
						place.left + place.width + 1,
						place.top + place.height + 1,
					]
					.map(|side| usize::try_from(side).expect("small"));

					(left..right).contains(&column) && (top..bottom).contains(&row)
				})
			})
			.collect()
	}

	/// Whether a texel is inside one place.
	fn inside(baked: &Baked, place: Rect, at: usize) -> bool {
		let at = u32::try_from(at).expect("a small picture");

		place.holds(at % baked.width, at / baked.width)
	}

	#[test]
	fn a_closed_room_of_glowing_walls_bakes_the_series_its_walls_make_into_every_texel() {
		let (glow, albedo) = (Vec3::new(0.5, 0.25, 0.75), Vec3::new(0.6, 0.3, 0.9));
		let scene = Scene::of(&furnace(glow, albedo));
		let settings = Settings {
			bounces: 4,
			rays: 32,
			..Settings::DEFAULT
		};
		let baked = bake(&scene, settings, threads()).expect("a room bakes");
		// every ray meets a wall, every wall sends its glow and its share of
		// what the last gather found, and every gather reads one value
		// everywhere: so the series, to the bit, in every texel of the room
		let mut known = Vec3::ZERO;

		for _ in 0..settings.bounces {
			known = glow + albedo * known;
		}

		let texels = placed(&baked);

		assert!(texels.len() > 1000, "the room's six walls, four times the cube's sheet");

		for at in texels {
			assert_eq!(
				baked.light[at].to_array().map(f32::to_bits),
				known.to_array().map(f32::to_bits),
				"texel {at}: {} where the series is {known}",
				baked.light[at]
			);
		}

		assert_eq!(baked.report.buried, 0, "no ray from inside a room meets its outside");
		assert_eq!(baked.report.pushed, 0, "and no texel is under anything");
	}

	#[test]
	fn an_open_floor_under_a_flat_sky_bakes_the_sky_to_the_bit_and_not_the_sun() {
		let sky = Vec3::new(0.2, 0.35, 0.8);
		let mut world = open(sky);

		// a sun, which the picture never holds: nothing here is lit by it
		// twice, because nothing is above the floor to send it back
		world.light = Vec3::new(-0.3, -1.0, 0.2);
		put(&mut world, MeshId::QUAD, MaterialId::DEFAULT, Transform {
			scale: Vec3::new(12.0, 1.0, 12.0),
			..Transform::IDENTITY
		});

		let baked =
			bake(&Scene::of(&world), Settings::DEFAULT, threads()).expect("a floor bakes");

		for at in placed(&baked) {
			assert_eq!(
				baked.light[at].to_array().map(f32::to_bits),
				sky.to_array().map(f32::to_bits),
				"texel {at} sees nothing but the sky: {}",
				baked.light[at]
			);
		}
	}

	#[test]
	fn a_floor_beside_a_black_wall_sees_as_much_of_the_sky_as_the_wall_leaves() {
		let mut world = open(Vec3::ONE);
		let black = world.materials.insert("black", Material {
			base_color: Vec3::ZERO,
			..Material::DEFAULT
		});

		put(&mut world, MeshId::QUAD, MaterialId::DEFAULT, Transform {
			scale: Vec3::new(16.0, 1.0, 16.0),
			..Transform::IDENTITY
		});
		// a wall four high along the floor's middle, far longer than anything
		// is from it, so what a floor point sees of it is the long wall's
		put(&mut world, MeshId::CUBE, black, Transform {
			position: Vec3::new(0.0, 2.0, 0.0),
			scale: Vec3::new(0.2, 4.0, 400.0),
			..Transform::IDENTITY
		});

		let scene = Scene::of(&world);
		let baked = bake(&scene, Settings::DEFAULT, threads()).expect("it bakes");
		let floor = baked.places[0].1;
		let texels =
			Texels::of(&scene, &Atlas::of(&scene, Settings::DEFAULT.texels).expect("fits"));
		let mut checked = 0;
		// each texel is a hundred and twenty-eight rays, so it is the average
		// over a band of distances that has to agree with the view factor
		let mut sums = [(0.0_f64, 0.0_f64); 12];

		for sample in texels.samples() {
			let surface = scene
				.surface_at(sample.triangle, sample.along, sample.across)
				.expect("a point");
			let away = surface.at.x.abs() - 0.1;
			let at = usize::try_from(sample.at).expect("small");

			if !inside(&baked, floor, at)
				|| !(0.4..6.0).contains(&away)
				|| surface.at.z.abs() > 5.0
			{
				continue;
			}

			// a floor point this far from an endless wall this high sees it
			// over half of the difference between one and the cosine of the
			// angle to its top; the bands are a unit wide, split by the
			// texel's column so each band holds two halves that agree
			let seen = 0.5 * (1.0 - away / away.hypot(4.0));
			let light = baked.light[at].x;

			let off = light - (1.0 - seen);
			let bucket = usize::from(u8::try_from(sample.at % 2).unwrap_or(0))
				+ 2 * [1.0, 2.0, 3.0, 4.0, 5.0]
					.iter()
					.filter(|edge| away > **edge)
					.count();

			assert!(
				off.abs() < 0.08,
				"{away} from the wall: {light}, the sky left {}",
				1.0 - seen
			);
			sums[bucket].0 += f64::from(off);
			sums[bucket].1 += 1.0;
			checked += 1;
		}

		for (band, (sum, count)) in sums.iter().enumerate() {
			assert!(*count > 20.0, "band {band} has texels in it: {count}");
			assert!(
				(sum / count).abs() < 0.006,
				"band {band}: off by {} on average over {count} texels",
				sum / count
			);
		}

		assert!(checked > 2000, "a strip either side of the wall: {checked}");
	}

	#[test]
	fn a_floor_under_a_block_is_pushed_out_from_under_it_or_left_to_its_neighbors() {
		let mut world = open(Vec3::ONE);

		put(&mut world, MeshId::QUAD, MaterialId::DEFAULT, Transform {
			scale: Vec3::new(8.0, 1.0, 8.0),
			..Transform::IDENTITY
		});
		// a block standing on the floor, its edges across the floor's texels:
		// x from -0.92 to 1.18, z from -0.92 to 0.78
		put(&mut world, MeshId::CUBE, MaterialId::DEFAULT, Transform {
			position: Vec3::new(0.13, 0.5, -0.07),
			scale: Vec3::new(2.1, 1.0, 1.7),
			..Transform::IDENTITY
		});

		let scene = Scene::of(&world);
		let baked = bake(&scene, Settings::DEFAULT, threads()).expect("it bakes");

		assert!(baked.report.pushed > 0, "texels half under the block are pushed out");
		assert!(baked.report.buried > 0, "and those wholly under it are inside it");

		for at in placed(&baked) {
			assert!(
				baked.light[at].is_finite() && baked.light[at].min_element() > 0.0,
				"texel {at} has light, the buried ones from their neighbors: {}",
				baked.light[at]
			);
		}

		// where the floor's pushed points went: out past every side of the block,
		// each a hair outside it - so nothing they gather meets its inside
		let texels =
			Texels::of(&scene, &Atlas::of(&scene, Settings::DEFAULT.texels).expect("fits"));
		let pushed: Vec<Spot> = texels
			.samples()
			.iter()
			.map(|sample| spot_of(&scene, sample))
			.filter(|spot| spot.pushed && spot.at.y.abs() < 0.01)
			.collect();
		let pattern = Pattern::new(64);
		let sides = [
			pushed.iter().any(|spot| spot.at.x < -0.92),
			pushed.iter().any(|spot| spot.at.x > 1.18),
			pushed.iter().any(|spot| spot.at.z < -0.92),
			pushed.iter().any(|spot| spot.at.z > 0.78),
		];

		assert_eq!(sides, [true; 4], "the floor comes out from under every side of the block");

		for spot in &pushed {
			let found = scene.gather(spot.at, spot.normal, 7, &pattern, |_, _| Vec3::ZERO);

			assert_eq!(found.behind, 0, "a pushed point at {} is outside the block", spot.at);
		}
	}

	#[test]
	fn a_texel_more_than_half_of_whose_rays_meet_backs_is_inside_and_no_other() {
		let met = |behind: u32| Some(Gathered { light: Vec3::ONE, behind, rays: 128 });
		let mut valid = vec![true; 5];
		let buried = bury(&mut valid, &[met(0), met(40), met(64), met(65), None]);

		assert_eq!(buried, 1, "one texel is inside something");
		assert_eq!(
			valid,
			[true, true, true, false, true],
			"the one past half of its rays; half and fewer are not, and one not gathered stays"
		);
	}

	#[test]
	fn a_thing_its_record_leaves_out_is_neither_lit_nor_in_the_way() {
		let sky = Vec3::ONE;
		let mut world = open(sky);

		put(&mut world, MeshId::QUAD, MaterialId::DEFAULT, Transform {
			scale: Vec3::new(8.0, 1.0, 8.0),
			..Transform::IDENTITY
		});

		let lid = put(&mut world, MeshId::CUBE, MaterialId::DEFAULT, Transform {
			position: Vec3::new(0.0, 1.5, 0.0),
			scale: Vec3::new(10.0, 0.2, 10.0),
			..Transform::IDENTITY
		});

		world
			.entities
			.record_mut(&BAKING, lid)
			.expect("every entity carries it")
			.skip = 1;

		let baked = bake(&Scene::of(&world), Settings::DEFAULT, threads()).expect("it bakes");

		assert_eq!(baked.places.len(), 1, "the lid has no place");

		for at in placed(&baked) {
			assert_eq!(baked.light[at], sky, "and the floor sees the sky through it");
		}
	}

	#[test]
	fn a_lit_wall_sends_its_light_on_to_the_floor_beside_it() {
		let mut world = open(Vec3::ZERO);

		world.light = Vec3::new(-1.0, -1.0, 0.0);
		put(&mut world, MeshId::QUAD, MaterialId::DEFAULT, Transform {
			scale: Vec3::new(8.0, 1.0, 8.0),
			..Transform::IDENTITY
		});
		put(&mut world, MeshId::CUBE, MaterialId::DEFAULT, Transform {
			position: Vec3::new(-2.0, 2.0, 0.0),
			scale: Vec3::new(0.2, 4.0, 8.0),
			..Transform::IDENTITY
		});

		let scene = Scene::of(&world);
		let floor_light = |bounces: u32| {
			let baked = bake(&scene, Settings { bounces, ..Settings::DEFAULT }, threads())
				.expect("it bakes");
			let floor = baked.places[0].1;

			placed(&baked)
				.into_iter()
				.filter(|&at| inside(&baked, floor, at))
				.map(|at| f64::from(baked.light[at].x))
				.fold((0.0_f64, 0.0_f64), |(sum, most), light| (sum + light, most.max(light)))
		};
		let (once, brightest) = floor_light(1);
		let (thrice, _) = floor_light(3);

		// a sky of nothing and a sun: all the floor holds is what the sunlit
		// face of the wall sends back to it, and each gather more can only add
		// to what the last one carried
		assert!(brightest > 0.05, "the wall's face lights the floor by {brightest}");
		assert!(
			thrice >= once,
			"three gathers carry at least what one does: {thrice} against {once}"
		);
	}

	#[test]
	fn a_surface_that_sends_nothing_back_still_gives_off_its_glow() {
		// a lid turned to face down over a floor in the dark: once black and
		// glowing, once drawn unlit; all the middle of the floor holds is it
		for (lid, wanted) in [
			(
				Material {
					base_color: Vec3::ZERO,
					emissive: Vec3::new(0.2, 0.4, 0.8),
					..Material::DEFAULT
				},
				Vec3::new(0.2, 0.4, 0.8),
			),
			(
				Material {
					base_color: Vec3::new(0.8, 0.4, 0.2),
					unlit: true,
					..Material::DEFAULT
				},
				Vec3::new(0.8, 0.4, 0.2),
			),
		] {
			let mut world = open(Vec3::ZERO);
			let material = world.materials.insert("lid", lid);

			put(&mut world, MeshId::QUAD, MaterialId::DEFAULT, Transform {
				scale: Vec3::new(4.0, 1.0, 4.0),
				..Transform::IDENTITY
			});
			put(&mut world, MeshId::QUAD, material, Transform {
				position: Vec3::new(0.0, 0.25, 0.0),
				rotation: Quat::from_rotation_x(std::f32::consts::PI),
				scale: Vec3::new(3.0, 1.0, 3.0),
			});

			let baked = bake(&Scene::of(&world), Settings::DEFAULT, threads()).expect("it bakes");
			let floor = baked.places[0].1;
			let middle = [floor.left + floor.width / 2, floor.top + floor.height / 2];
			let light =
				baked.light[usize::try_from(middle[1] * baked.width + middle[0]).expect("small")];

			// a lid three wide a quarter up covers nearly all the middle sees
			assert!(
				(light - wanted * 0.95).max_element() > 0.0
					&& (light - wanted).max_element() <= 1.0e-4,
				"the middle of the floor sees the lid's {wanted} nearly all round: {light}"
			);
		}
	}

	#[test]
	fn a_lit_thing_with_no_place_still_sends_its_light_on() {
		let mut world = open(Vec3::ZERO);
		let sheetless = world.meshes.insert("bare", {
			let mut mesh = cube();

			mesh.paint.clear();
			mesh.sheet = [0, 0];

			mesh
		});

		world.light = Vec3::new(-1.0, -1.0, 0.0);
		put(&mut world, MeshId::QUAD, MaterialId::DEFAULT, Transform {
			scale: Vec3::new(8.0, 1.0, 8.0),
			..Transform::IDENTITY
		});
		put(&mut world, sheetless, MaterialId::DEFAULT, Transform {
			position: Vec3::new(-2.0, 2.0, 0.0),
			scale: Vec3::new(0.2, 4.0, 8.0),
			..Transform::IDENTITY
		});

		let baked = bake(&Scene::of(&world), Settings::DEFAULT, threads()).expect("it bakes");
		let floor = baked.places[0].1;
		let brightest = placed(&baked)
			.into_iter()
			.filter(|&at| inside(&baked, floor, at))
			.map(|at| baked.light[at].x)
			.fold(0.0_f32, f32::max);

		assert_eq!(baked.placeless.len(), 1, "the wall has no place");
		assert!(brightest > 0.05, "and it lights the floor all the same: {brightest}");
	}

	#[test]
	fn a_texel_is_filled_from_its_own_chart_and_its_own_thing_and_nothing_else() {
		// every sample holds the number of its chart: then every texel a chart
		// is read through holds that number, and every other texel of a place
		// holds the number of one of that place's charts
		let mut world = open(Vec3::ONE);

		for (at, scale) in
			[(0.0, Vec3::ONE), (3.0, Vec3::new(2.0, 0.5, 1.5)), (7.0, Vec3::splat(0.4))]
		{
			put(&mut world, MeshId::CUBE, MaterialId::DEFAULT, Transform {
				position: Vec3::new(at, 0.0, 0.0),
				scale,
				..Transform::IDENTITY
			});
		}

		let scene = Scene::of(&world);
		let texels =
			Texels::of(&scene, &Atlas::of(&scene, Settings::DEFAULT.texels).expect("fits"));
		let chart_of = |at: u32| texels.chart(usize::try_from(at).expect("small"));
		let values: Vec<Vec3> = texels
			.samples()
			.iter()
			.map(|sample| {
				Vec3::splat(f32::from(u16::try_from(chart_of(sample.at)).expect("few")))
			})
			.collect();
		// every sample of an even chart beside another chart's texel counts as
		// inside something, so the texels a fill has to reach sit against a
		// chart that is not theirs and that holds - one side of each border
		// only, or the other side would be as empty as they are
		let width = i64::from(texels.width());
		let bordering = |at: u32| {
			let (column, row) = (i64::from(at) % width, i64::from(at) / width);
			let own = chart_of(at);

			[[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
				.iter()
				.filter_map(|[across, down]| {
					usize::try_from((row + down) * width + column + across).ok()
				})
				.any(|near| texels.chart(near) != NONE && texels.chart(near) != own)
		};
		let valid: Vec<bool> = texels
			.samples()
			.iter()
			.map(|sample| chart_of(sample.at) % 2 == 1 || !bordering(sample.at))
			.collect();
		let kept: Vec<u32> = texels
			.samples()
			.iter()
			.zip(&valid)
			.filter(|(_, keep)| **keep)
			.map(|(sample, _)| chart_of(sample.at))
			.collect();

		assert!(valid.iter().any(|keep| !*keep), "some charts sit a texel from another");

		let filled = picture(&texels, &values, &valid, true);
		let span_of = |owner: u32| -> (f32, f32) {
			(0..filled.len())
				.filter(|&at| texels.owner(at) == owner && texels.chart(at) != NONE)
				.map(|at| f32::from(u16::try_from(texels.chart(at)).expect("few")))
				.fold((f32::MAX, f32::MIN), |(least, most), chart| {
					(least.min(chart), most.max(chart))
				})
		};

		for (at, value) in filled.iter().enumerate() {
			let owner = texels.owner(at);

			if owner == NONE {
				assert_eq!(*value, Vec3::ZERO, "texel {at} is in no place and holds nothing");
			} else if texels.chart(at) != NONE && kept.contains(&texels.chart(at)) {
				// a chart with nothing left that holds takes what is beside it,
				// which is the loose fill's to do and not this test's
				let chart = f32::from(u16::try_from(texels.chart(at)).expect("few"));

				assert_eq!(*value, Vec3::splat(chart), "texel {at} holds its own chart's light");
			} else if texels.chart(at) == NONE {
				// a gutter may mix the charts of its own place, and nothing else
				let (least, most) = span_of(owner);

				assert!(
					(least..=most).contains(&value.x),
					"texel {at} of place {owner} holds {value}, outside that place's charts"
				);
			}
		}
	}

	#[test]
	fn one_thread_and_many_bake_the_same_bytes_and_so_does_a_second_bake() {
		let room = Scene::of(&furnace(Vec3::new(0.3, 0.1, 0.2), Vec3::splat(0.5)));
		let mut world = open(Vec3::new(0.4, 0.5, 0.6));

		world.light = Vec3::new(-0.2, -1.0, 0.3);
		put(&mut world, MeshId::QUAD, MaterialId::DEFAULT, Transform {
			scale: Vec3::new(6.0, 1.0, 6.0),
			..Transform::IDENTITY
		});
		put(&mut world, MeshId::CUBE, MaterialId::DEFAULT, Transform {
			position: Vec3::new(0.4, 0.6, 0.2),
			scale: Vec3::new(1.3, 1.2, 0.7),
			..Transform::IDENTITY
		});

		for scene in [room, Scene::of(&world)] {
			let settings = Settings {
				rays: 24,
				bounces: 2,
				..Settings::DEFAULT
			};
			let alone = bake(&scene, settings, 1).expect("one thread");
			let many = bake(&scene, settings, threads().max(4)).expect("many");
			let again = bake(&scene, settings, threads().max(4)).expect("again");
			let bits = |baked: &Baked| -> Vec<u32> {
				baked
					.light
					.iter()
					.flat_map(|texel| texel.to_array().map(f32::to_bits))
					.collect()
			};

			assert_eq!(bits(&alone), bits(&many), "the number of threads changes nothing");
			assert_eq!(bits(&many), bits(&again), "and a second bake is the first");
			assert_eq!(alone.places, many.places, "nor where anything is");
		}
	}

	#[test]
	fn a_picture_is_read_between_texels_and_one_value_reads_as_itself() {
		let picture = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(3.0, 2.0, 1.0), Vec3::ZERO, Vec3::ONE];
		let flat = [Vec3::new(0.1, 0.7, 0.3); 4];

		assert_eq!(read(&picture, 2, 2, [0.5, 0.5]), picture[0], "a middle reads its texel");
		assert_eq!(read(&picture, 2, 2, [1.5, 0.5]), picture[1], "and so does the next");
		assert_eq!(
			read(&picture, 2, 2, [1.0, 0.5]),
			Vec3::new(2.0, 2.0, 2.0),
			"halfway between two is the middle of them"
		);
		assert_eq!(read(&picture, 2, 2, [-3.0, 9.0]), picture[2], "and past an edge is held");

		for at in [[0.3, 0.9], [1.2, 1.7], [0.77, 1.01]] {
			assert_eq!(
				read(&flat, 2, 2, at),
				flat[0],
				"four of one value are that value at {at:?}"
			);
		}
	}

	#[test]
	fn what_a_bake_is_asked_for_is_held_where_it_means_something() {
		let wild = Settings { texels: f32::NAN, bounces: 0, rays: 1 }.sane();
		let wide = Settings {
			texels: 1.0e9,
			bounces: 99,
			rays: 1_000_000,
		}
		.sane();

		assert_eq!(wild, Settings {
			texels: Settings::DEFAULT.texels,
			bounces: 1,
			rays: 16
		});
		assert_eq!(wide, Settings { texels: 64.0, bounces: 16, rays: 8192 });
	}

	#[test]
	fn a_world_with_nothing_still_in_it_is_refused_rather_than_baked_empty() {
		let mut world = open(Vec3::ONE);
		let lamp = world.entities.spawn();

		world
			.entities
			.set_light(lamp, Light::point(Vec3::ONE, 1.0, 5.0));

		assert!(
			bake(&Scene::of(&world), Settings::DEFAULT, 1).is_err(),
			"a lamp is not a surface"
		);
	}
}
