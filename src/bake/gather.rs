//! The light arriving at a point from every way, and many points at once.
//!
//! **A fixed pattern, turned for each point.** The directions a point sends its
//! rays in are one pattern worked out once - points spread evenly over a disk,
//! lifted onto the hemisphere above it - turned about the point's normal by an
//! angle its own seed picks. Spread evenly, a hundred directions answer about
//! as well as several hundred drawn at random; turned per point, neighbors do
//! not all miss the same thin pole in the same way, which would draw its miss
//! as a stripe across the surface where a different turn draws it as grain.
//!
//! **Lifting a disk onto a hemisphere is the cosine for free.** A point spread
//! evenly over the unit disk and lifted straight up onto the hemisphere lands
//! in each direction exactly as often as that direction's cosine with the
//! normal says - which is the weight the light arriving from it carries. So
//! the average of what the rays bring back is the answer, with no weights to
//! multiply by, and no sine or cosine anywhere: the lift is a square root and
//! the turn is a pair of numbers on the unit circle found by drawing a point
//! in a disk and dividing it by its length.
//!
//! **The average is added up in double precision**, and that is what makes a
//! closed room of glowing walls answer its own series to the bit: every ray
//! brings back the same number, and a hundred of one number added in double
//! precision and divided by a hundred is that number again.

use std::{
	num::NonZero,
	sync::atomic::{AtomicUsize, Ordering},
};

use colby_core::{
	glam::{Vec2, Vec3},
	random::Random,
};

use crate::{
	scene::{BIAS, Scene},
	tree::{Hit, Ray},
};

/// How many points one turn of a worker takes before it asks for more.
const CHUNK: usize = 64;

/// The directions every point sends its rays in, before its turn.
#[derive(Clone, Debug, PartialEq)]
pub struct Pattern {
	/// Places inside the unit disk, spread evenly: the first two bases of
	/// Halton's sequence, kept where they fall inside the circle.
	places: Vec<Vec2>,
}

/// What a point gathered.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Gathered {
	/// The light arriving, in the ambient color's unit: the average of what
	/// every ray brought back.
	pub light: Vec3,

	/// How many rays met the back of a surface.
	///
	/// A ray that meets a back is looking at the inside of something, which a
	/// point standing where it should cannot do: many of them say the point
	/// is buried, and whoever asked decides what that makes it. They bring
	/// back nothing.
	pub behind: u32,

	/// How many rays were sent.
	pub rays: u32,
}

impl Pattern {
	/// A pattern of so many directions.
	///
	/// @param rays - how many; at least one
	#[must_use]
	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		reason = "a place worked out in double precision and narrowed once, where it is kept"
	)]
	pub fn new(rays: u32) -> Self {
		let wanted = usize::try_from(rays.max(1)).unwrap_or(1);
		let mut places = Vec::with_capacity(wanted);
		let mut index = 1_u32;

		while places.len() < wanted {
			let (x, y) = (radical(index, 2), radical(index, 3));
			// from nought and one to minus one and one: doubled by adding, which
			// is exact, and moved down by one
			let (x, y) = (x + x - 1.0, y + y - 1.0);
			let (across, up) = (x * x, y * y);

			if across + up < 1.0 {
				places.push(Vec2::new(x as f32, y as f32));
			}

			index = index.saturating_add(1);
		}

		Self { places }
	}

	/// How many directions it holds.
	#[must_use]
	pub fn len(&self) -> usize { self.places.len() }

	/// Whether it holds none, which a pattern never does.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.places.is_empty() }
}

impl Scene {
	/// The light arriving at one point of a surface from every way above it.
	///
	/// A ray that leaves the world brings back the sky; one that meets the
	/// front of a surface brings back what `arriving` says that surface sends
	/// towards the point; one that meets a back brings back nothing and is
	/// counted.
	///
	/// @param at - the point
	/// @param normal - which way the surface faces there, of unit length
	/// @param seed - what turns this point's pattern; the same seed turns it
	/// the same way
	/// @param pattern - the directions
	/// @param arriving - what a surface a ray landed on sends back along it,
	/// handed the hit and the way the ray went
	#[must_use]
	pub fn gather<Arriving: Fn(&Hit, Vec3) -> Vec3>(
		&self,
		at: Vec3,
		normal: Vec3,
		seed: u64,
		pattern: &Pattern,
		arriving: Arriving,
	) -> Gathered {
		let start = at + normal * BIAS;
		let (first, second) = basis(normal);
		// a turn by an angle the generator picks, a point on the unit circle
		let turn = Random::new(seed).circle();
		let mut total = [0.0_f64; 3];
		let mut behind = 0_u32;

		for place in &pattern.places {
			// the place turned about the middle of the disk - `x' = c x - s y`,
			// `y' = s x + c y` - then lifted
			let turned = Vec2::new(
				Vec2::new(turn.x, -turn.y).dot(*place),
				Vec2::new(turn.y, turn.x).dot(*place),
			);
			let lift = (1.0 - turned.length_squared()).max(0.0).sqrt();
			let way = (first * turned.x + second * turned.y + normal * lift).normalize_or(normal);
			let ray = Ray::new(start, way);
			let light = match self.tree().nearest(&ray, f32::INFINITY) {
				| None => self.sky().toward(way),
				| Some(hit) if hit.front => arriving(&hit, way),
				| Some(_) => {
					behind = behind.saturating_add(1);

					Vec3::ZERO
				},
			};

			for (sum, channel) in total.iter_mut().zip(light.to_array()) {
				*sum += f64::from(channel);
			}
		}

		let rays = u32::try_from(pattern.len()).unwrap_or(u32::MAX);

		Gathered {
			light: Vec3::from_array(total.map(|sum| narrowed(sum / f64::from(rays.max(1))))),
			behind,
			rays,
		}
	}
}

/// Works something out for every index from nought up, on several threads,
/// and hands the answers back in index order.
///
/// Each index is worked out alone and put in its own place, so how many
/// threads there are and which of them took which index changes nothing that
/// comes back - which is what lets a bake be the same bytes on a machine of two
/// cores and one of thirty-two.
///
/// @param count - how many indices
/// @param threads - how many threads at most; one works on the caller's
/// @param work - what to work out for one index
#[must_use]
pub fn each<T: Send, Work: Fn(usize) -> T + Sync>(
	count: usize,
	threads: usize,
	work: Work,
) -> Vec<T> {
	let threads = threads.clamp(1, count.div_ceil(CHUNK).max(1));

	if threads == 1 {
		return (0..count).map(work).collect();
	}

	let next = AtomicUsize::new(0);
	let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|scope| {
		let workers: Vec<_> =
			std::iter::repeat_with(|| scope.spawn(|| taken(&next, count, &work)))
				.take(threads)
				.collect();

		workers
			.into_iter()
			.map(|worker| {
				worker
					.join()
					.unwrap_or_else(|panic| std::panic::resume_unwind(panic))
			})
			.collect()
	});

	let mut placed: Vec<Option<T>> = std::iter::repeat_with(|| None)
		.take(count)
		.collect();

	for (index, value) in parts.into_iter().flatten() {
		if let Some(slot) = placed.get_mut(index) {
			*slot = Some(value);
		}
	}

	placed.into_iter().flatten().collect()
}

/// What one worker does: take the next run of indices nobody has taken, work
/// each out, and come back for more until there are none.
///
/// @param next - the first index nobody has taken, shared by every worker
/// @param count - how many indices there are
/// @param work - what to work out for one
/// @return every index this worker took, with its answer
fn taken<T, Work: Fn(usize) -> T>(
	next: &AtomicUsize,
	count: usize,
	work: &Work,
) -> Vec<(usize, T)> {
	let mut done = Vec::new();

	loop {
		let start = next.fetch_add(CHUNK, Ordering::Relaxed);

		if start >= count {
			return done;
		}

		done.extend((start..(start + CHUNK).min(count)).map(|index| (index, work(index))));
	}
}

/// A double narrowed to a float, rounding to the nearest.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "an average worked out in double precision and narrowed once, where it is kept"
)]
const fn narrowed(value: f64) -> f32 { value as f32 }

/// How many threads this machine runs at once.
#[must_use]
pub fn threads() -> usize { std::thread::available_parallelism().map_or(1, NonZero::get) }

/// A seed for one point of one pass: the two numbers mixed so neighbors and
/// passes draw unrelated turns.
///
/// @param point - which point
/// @param pass - which pass over the points
#[must_use]
pub fn seed(point: usize, pass: u32) -> u64 {
	let point = u64::try_from(point).unwrap_or(u64::MAX);

	point
		.wrapping_add(1)
		.wrapping_mul(0x9E37_79B9_7F4A_7C15)
		^ u64::from(pass).wrapping_mul(0xD1B5_4A32_D192_ED03)
}

/// One index of Halton's sequence in one base: its digits in that base,
/// reversed behind the point.
fn radical(index: u32, base: u32) -> f64 {
	let mut left = index;
	let mut fraction = 1.0;
	let mut total = 0.0;
	let base_float = f64::from(base);

	while left > 0 {
		fraction /= base_float;
		let digit = fraction * f64::from(left % base);

		total += digit;
		left /= base;
	}

	total
}

/// Two axes across a normal, square with it and with each other.
///
/// The branchless construction the field has used since it was published,
/// with nothing but a copy of a sign in it; it is continuous everywhere except
/// across the one plane where the normal's third axis changes sign.
fn basis(normal: Vec3) -> (Vec3, Vec3) {
	let sign = 1.0_f32.copysign(normal.z);
	let lean = -(sign + normal.z).recip();
	let shared = normal.x * normal.y * lean;
	let first_x = sign * normal.x * normal.x * lean;
	let second_y = normal.y * normal.y * lean;

	(
		Vec3::new(1.0 + first_x, sign * shared, -sign * normal.x),
		Vec3::new(shared, sign + second_y, -normal.y),
	)
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			MeshId, Renderable, Transform, World,
			material::{Material, MaterialId},
			mesh::{MeshData, cube},
		},
		glam::Quat,
	};

	use super::*;

	/// A box of the given half size seen from inside: the unit cube, grown,
	/// with every triangle turned to face in.
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

	/// What a surface a ray landed on sends back when what arrives at it is
	/// the same everywhere: its glow, and its share of what arrives.
	fn bounced(scene: &Scene, hit: &Hit, arriving: Vec3) -> Vec3 {
		let surface = scene.surface(hit).expect("a surface");

		surface.emission + surface.diffuse * arriving
	}

	/// What a surface a ray landed on sends back when the sun and the lamps
	/// are all that light it.
	fn lit_once(scene: &Scene, hit: &Hit) -> Vec3 {
		let surface = scene.surface(hit).expect("a surface");

		bounced(scene, hit, scene.direct(surface.at, surface.normal))
	}

	/// A room of glowing walls, all of one material, and nothing else.
	fn furnace(glow: Vec3, albedo: Vec3) -> World {
		let mut world = World::new();
		let room = world.meshes.insert("room", inside_out(2.0));
		let walls = world.materials.insert("walls", Material {
			base_color: albedo,
			emissive: glow,
			..Material::DEFAULT
		});
		let at = world.entities.spawn();

		world.ambient = Vec3::new(9.0, 9.0, 9.0);
		world
			.entities
			.set_renderable(at, Renderable::of(room, walls, Vec3::ONE));

		world
	}

	#[test]
	fn a_closed_room_of_glowing_walls_gathers_the_series_its_walls_make() {
		let (glow, albedo) = (Vec3::new(0.5, 0.25, 0.75), Vec3::new(0.6, 0.3, 0.9));
		let scene = Scene::of(&furnace(glow, albedo));
		let pattern = Pattern::new(128);
		let points = [
			(Vec3::new(0.0, -2.0, 0.0), Vec3::Y),
			(Vec3::new(1.3, -2.0, -0.7), Vec3::Y),
			(Vec3::new(0.2, 0.4, -0.3), Vec3::new(0.48, 0.6, 0.64)),
			(Vec3::new(2.0, 1.9, 1.9), Vec3::NEG_X),
		];
		let mut known = Vec3::ZERO;

		// every ray meets a wall, every wall sends back its glow and its share
		// of what the last pass found - one number for every ray, so the
		// average is that number to the bit
		for pass in 0..5 {
			let before = known;
			let arriving = |hit: &Hit, _: Vec3| bounced(&scene, hit, before);

			known = glow + albedo * before;

			for (index, (at, normal)) in points.into_iter().enumerate() {
				let gathered = scene.gather(at, normal, seed(index, pass), &pattern, arriving);

				assert_eq!(gathered.light, known, "pass {pass}, point {index}");
				assert_eq!(gathered.behind, 0, "no ray finds the outside of the room");
				assert_eq!(gathered.rays, 128, "and every ray was sent");
			}
		}
	}

	#[test]
	fn an_open_floor_under_a_flat_sky_gathers_the_sky_to_the_bit() {
		let mut world = World::new();
		let floor = world.entities.spawn_at(Transform {
			scale: Vec3::new(30.0, 1.0, 30.0),
			..Transform::IDENTITY
		});

		world.ambient = Vec3::new(0.2, 0.35, 0.8);
		world
			.entities
			.set_renderable(floor, Renderable::of(MeshId::QUAD, MaterialId::DEFAULT, Vec3::ONE));

		let scene = Scene::of(&world);
		let pattern = Pattern::new(96);

		for (index, x) in [0.0_f32, 3.0, -7.5, 12.0].into_iter().enumerate() {
			let gathered = scene.gather(
				Vec3::new(x, 0.0, 1.0),
				Vec3::Y,
				seed(index, 0),
				&pattern,
				|_, _| panic!("nothing is above the floor to meet"),
			);

			assert_eq!(gathered.light, world.ambient, "at {x}");
		}
	}

	#[test]
	fn a_floor_beside_a_wall_sees_the_sky_the_wall_leaves_it() {
		let mut world = World::new();

		world.ambient = Vec3::ONE;

		for (position, scale) in [
			(Vec3::ZERO, Vec3::new(30.0, 1.0, 30.0)),
			(Vec3::new(0.0, 5.0, -1.0), Vec3::new(30.0, 10.0, 1.0)),
		] {
			let mesh = if scale.y > 1.0 { MeshId::CUBE } else { MeshId::QUAD };
			let thing =
				world
					.entities
					.spawn_at(Transform { position, scale, ..Transform::IDENTITY });

			world
				.entities
				.set_renderable(thing, Renderable::of(mesh, MaterialId::DEFAULT, Vec3::ONE));
		}

		let scene = Scene::of(&world);
		let near = scene.gather(Vec3::ZERO, Vec3::Y, 2, &Pattern::new(256), |_, _| Vec3::ZERO);
		// what the wall hides, worked out the long way: its face is thirty wide
		// and ten high, half a unit from the point, and a patch of it hides
		// the cosine at the floor times the cosine at the wall over pi times
		// the square of the distance
		let hidden = hidden_by_the_wall(30.0, 10.0, 0.5);

		assert!(
			(f64::from(near.light.x) - (1.0 - hidden)).abs() < 0.005,
			"beside the wall {} of the sky, where the wall leaves {}",
			near.light.x,
			1.0 - hidden
		);
		assert_eq!(near.behind, 0, "and nothing it sees is the inside of anything");
	}

	/// The share of a floor point's cosine-weighted hemisphere a wall in front
	/// of it hides, by adding up the wall a small square at a time.
	///
	/// @param width - how wide the wall is, centered on the point
	/// @param height - how high, from the floor
	/// @param distance - how far its face is from the point
	fn hidden_by_the_wall(width: f64, height: f64, distance: f64) -> f64 {
		let steps = 1500_u32;
		let (across, up) = (width / f64::from(steps), height / f64::from(steps));
		let mut total = 0.0;

		for column in 0..steps {
			let x = (f64::from(column) + 0.5) * across;
			let x = x - width / 2.0;

			for row in 0..steps {
				let y = (f64::from(row) + 0.5) * up;
				let (wide, high, deep) = (x * x, y * y, distance * distance);
				let square = wide + high + deep;
				let facing = y * distance;

				total += facing / (std::f64::consts::PI * square * square);
			}
		}

		total * across * up
	}

	#[test]
	fn a_slope_under_a_flat_sky_gathers_the_sky_to_the_bit() {
		let mut world = World::new();
		let slope = world.entities.spawn_at(Transform {
			rotation: Quat::from_rotation_z(0.4),
			scale: Vec3::new(30.0, 1.0, 30.0),
			..Transform::IDENTITY
		});

		world.ambient = Vec3::new(0.3, 0.5, 0.7);
		world
			.entities
			.set_renderable(slope, Renderable::of(MeshId::QUAD, MaterialId::DEFAULT, Vec3::ONE));

		let scene = Scene::of(&world);
		let pattern = Pattern::new(64);

		// points worked out on the slope the way a bake works them out, so
		// each carries the rounding a real one does, and a ray that started
		// on it rather than off it would meet it again
		for step in 1..40_u16 {
			let weight = f32::from(step) / 41.0;
			let (along, less) = (weight * 0.5, weight * 0.4);
			let surface = scene
				.surface_at(u32::from(step % 2), along, 0.5 - less)
				.expect("a point on the slope");
			let gathered = scene.gather(
				surface.at,
				surface.normal,
				seed(usize::from(step), 0),
				&pattern,
				|_, _| Vec3::ZERO,
			);

			assert_eq!(gathered.light, world.ambient, "at {}", surface.at);
		}
	}

	#[test]
	fn the_floor_of_a_well_sees_the_sky_its_opening_leaves_it() {
		let mut world = World::new();

		world.ambient = Vec3::ONE;

		// four walls two high round a floor two across, overlapping at the
		// corners so no ray finds a gap between them
		for (position, scale) in [
			(Vec3::new(0.0, 1.0, -1.25), Vec3::new(3.0, 2.0, 0.5)),
			(Vec3::new(0.0, 1.0, 1.25), Vec3::new(3.0, 2.0, 0.5)),
			(Vec3::new(-1.25, 1.0, 0.0), Vec3::new(0.5, 2.0, 3.0)),
			(Vec3::new(1.25, 1.0, 0.0), Vec3::new(0.5, 2.0, 3.0)),
		] {
			let wall =
				world
					.entities
					.spawn_at(Transform { position, scale, ..Transform::IDENTITY });

			world.entities.set_renderable(
				wall,
				Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, Vec3::ONE),
			);
		}

		let scene = Scene::of(&world);
		let seen = scene.gather(Vec3::ZERO, Vec3::Y, 9, &Pattern::new(1024), |_, _| Vec3::ZERO);
		let opening = seen_through_the_opening(2.0, 2.0);

		assert!(
			(f64::from(seen.light.x) - opening).abs() < 0.01,
			"{} of the sky through the opening, where the opening leaves {opening}",
			seen.light.x
		);
	}

	/// The share of a floor point's cosine-weighted hemisphere a square opening
	/// straight above it covers, by adding up the opening a small square at a
	/// time: the cosine at the floor times the cosine at the opening over pi
	/// times the square of the distance, both cosines the height over the
	/// distance.
	///
	/// @param side - how wide the opening is, centered over the point
	/// @param height - how far above the point it is
	fn seen_through_the_opening(side: f64, height: f64) -> f64 {
		let steps = 1000_u32;
		let step = side / f64::from(steps);
		let mut total = 0.0;

		for column in 0..steps {
			let x = (f64::from(column) + 0.5) * step;
			let x = x - side / 2.0;

			for row in 0..steps {
				let z = (f64::from(row) + 0.5) * step;
				let z = z - side / 2.0;
				let (wide, deep, high) = (x * x, z * z, height * height);
				let square = wide + deep + high;

				total += high / (std::f64::consts::PI * square * square);
			}
		}

		total * step * step
	}

	#[test]
	fn a_point_inside_a_block_sees_its_insides() {
		let mut world = World::new();
		let block = world.entities.spawn_at(Transform::at(Vec3::ZERO));

		world
			.entities
			.set_renderable(block, Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, Vec3::ONE));

		let scene = Scene::of(&world);
		let gathered = scene.gather(Vec3::ZERO, Vec3::Y, 3, &Pattern::new(64), |_, _| Vec3::ONE);

		assert_eq!(gathered.behind, 64, "every way out is the back of a face");
		assert_eq!(gathered.light, Vec3::ZERO, "and brings back nothing");
	}

	#[test]
	fn a_pattern_spreads_its_directions_by_the_cosine() {
		let pattern = Pattern::new(4096);
		// the cosine-weighted average of the height above the disk is two
		// thirds, and of its square is a half
		let (mut height, mut squared) = (0.0_f64, 0.0_f64);

		for place in &pattern.places {
			let lift = f64::from((1.0 - place.length_squared()).max(0.0).sqrt());

			height += lift;
			squared += lift * lift;
		}

		let count = f64::from(u32::try_from(pattern.len()).expect("a small pattern"));

		assert!(
			(height / count - 2.0 / 3.0).abs() < 2.0e-3,
			"the mean height {}",
			height / count
		);
		assert!((squared / count - 0.5).abs() < 2.0e-3, "the mean square {}", squared / count);
		assert!(
			pattern
				.places
				.iter()
				.all(|place| place.length_squared() < 1.0),
			"every place inside the disk"
		);
	}

	#[test]
	fn the_axes_across_a_normal_are_square_with_it_and_each_other() {
		let mut random = Random::new(5);

		for _ in 0..2000 {
			let normal = Vec3::new(random.signed(), random.signed(), random.signed())
				.normalize_or(Vec3::Y);
			let (first, second) = basis(normal);

			assert!(first.dot(normal).abs() < 1.0e-5, "the first across {normal}");
			assert!(second.dot(normal).abs() < 1.0e-5, "the second across {normal}");
			assert!(first.dot(second).abs() < 1.0e-5, "square with each other");
			assert!((first.length() - 1.0).abs() < 1.0e-5, "of unit length");
			assert!(
				first.cross(second).dot(normal) > 0.99,
				"and turning the same way as the normal"
			);
		}
	}

	#[test]
	fn every_point_comes_back_in_its_place_however_many_threads_work() {
		let squares = |threads| each(1000, threads, |index| index * index);
		let one = squares(1);

		assert_eq!(one.len(), 1000, "every index");
		assert!(
			one.iter()
				.enumerate()
				.all(|(index, value)| *value == index * index),
			"in order"
		);
		assert_eq!(squares(3), one, "three threads say the same");
		assert_eq!(squares(16), one, "and sixteen");
		assert!(each(0, 8, |index| index).is_empty(), "nothing to do is nothing done");
	}

	#[test]
	fn a_gather_is_the_same_bytes_on_one_thread_and_on_many_and_twice() {
		let mut world = furnace(Vec3::new(0.1, 0.2, 0.3), Vec3::new(0.5, 0.5, 0.5));
		let pillar = world.entities.spawn_at(Transform {
			position: Vec3::new(0.5, -1.0, 0.2),
			scale: Vec3::new(0.4, 2.0, 0.4),
			..Transform::IDENTITY
		});

		world
			.entities
			.set_renderable(pillar, Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, Vec3::ONE));

		let scene = Scene::of(&world);
		let pattern = Pattern::new(64);
		let run = |threads| {
			each(700, threads, |index| {
				let step = f32::from(u16::try_from(index).expect("a small count")) / 700.0;
				let at = Vec3::new(3.8, 0.0, 2.0) * step + Vec3::new(-1.9, -1.999, -1.0);

				scene
					.gather(at, Vec3::Y, seed(index, 7), &pattern, |hit, _| lit_once(&scene, hit))
			})
		};
		let bits = |gathered: &[Gathered]| {
			gathered
				.iter()
				.flat_map(|one| one.light.to_array().map(f32::to_bits))
				.collect::<Vec<_>>()
		};
		let alone = run(1);

		assert_eq!(bits(&run(8)), bits(&alone), "eight threads, the same bytes");
		assert_eq!(bits(&run(1)), bits(&alone), "and a second run, the same bytes");
		assert!(
			alone
				.iter()
				.any(|one| one.light != alone[0].light),
			"and the points differ, so it means something"
		);
	}
}
