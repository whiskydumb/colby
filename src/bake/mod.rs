//! Light worked out before a world is played: what reaches each still surface
//! from the sky and from everything else in the room.
//!
//! ```text
//!   Scene::of(&world)                     what stands still, as triangles
//!   scene.direct(at, normal)              the sun and the lamps at a point
//!   scene.gather(at, normal, seed, ..)    the light arriving from every way
//!   bake(&scene, settings, threads)       all of it, into one picture
//! ```
//!
//! **What the renderer draws every frame is not what this works out.** The sun
//! and the lamps are lit each frame with their own shadows, so a bake adds
//! nothing by repeating them; what no frame can afford is the light that
//! arrives by any other route - the share of the sky a surface under a roof
//! still sees, and what a lit wall throws onto the floor beside it. That is
//! what a gather adds up, and it takes the place of the one color, or the one
//! sky, a frame lights such a surface with today.
//!
//! **The unit is the ambient color's.** A surface whose diffuse color is
//! `albedo` sends back `albedo * value` of whatever this hands out, which is
//! what `World::ambient` has always meant and what the sun and a lamp mean in
//! the shader: a lamp of intensity one head on at one unit away is one. So a
//! value here can stand in for the ambient color in the shader without a
//! factor between them, and a world whose sky is one flat color gathers exactly
//! that color on a floor nothing overhangs.
//!
//! **The same bytes on every machine.** The rays are traced on the processor
//! against [`tree::Tree`], built here; a gather's directions come from a fixed
//! pattern turned per point by a seeded generator; every point is worked out
//! alone and written to its own place, so the number of threads changes
//! nothing; and the arithmetic a ray does is additions, products, quotients and
//! square roots, which the floating-point standard pins to the bit. A sine, a
//! power or a logarithm is not pinned, since two libraries may round one
//! differently, so those appear only in tables worked out once, in double
//! precision, and narrowed: the cone of a lamp, the turn of a material's
//! coordinates, the curve of an eight-bit color.
//!
//! What is here, and what each part is for:
//!
//! - [`scene`] - what a bake sees of a world: the triangles of every still
//!   thing, what each is made of, the lamps, the sun and the sky;
//! - [`tree`] - the hierarchy of boxes over those triangles, and a ray's
//!   nearest hit or whether anything is in its way;
//! - [`light`] - the sun and the lamps at one point, by the shader's own
//!   falloff and cone, with a ray towards each for its shadow;
//! - [`sky`] - what a ray that leaves the world brings back;
//! - [`picture`] - a texture read on the processor, for the color of a surface
//!   a ray lands on;
//! - [`gather`] - the pattern of directions, and the average of what they bring
//!   back, over many points at once;
//! - [`atlas`] - where each still thing's light goes on one picture;
//! - [`texels`] - which point of which surface each texel of it stands for;
//! - [`lightmap`] - the passes that work the picture out, and the filling of
//!   what no surface stands for.

pub mod atlas;
pub mod gather;
pub mod light;
pub mod lightmap;
pub mod picture;
pub mod scene;
pub mod sky;
pub mod texels;
pub mod tree;

pub use self::{
	atlas::{Atlas, MAX_SIDE, Placeless, Rect},
	gather::{Gathered, Pattern, each, threads},
	lightmap::{Baked, Report, Settings, bake},
	picture::Picture,
	scene::{Corner, Lamp, Look, Piece, Scene, Surface},
	sky::Sky,
	texels::{Sample, Texels},
	tree::{Hit, Ray, Tree},
};

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			MeshId, Renderable, Transform, World,
			light::Light,
			material::{Material, MaterialId},
			mesh::{MeshData, MeshVertex},
			sky::{Sky as Heavens, SkyKind},
			texture::{CUBE_FACES, Texel, TextureData},
		},
		glam::{Quat, Vec2, Vec3},
		utils::half::half,
	};

	use super::*;
	use crate::gather::seed;

	/// A sky of six faces, four texels a side, every texel a color of its own:
	/// a face read turned or mirrored reads other colors.
	fn cube() -> TextureData {
		let texels = (0..6_u16 * 16).flat_map(|index| {
			let (face, place) = (f32::from(index / 16), f32::from(index % 16));
			let color = Vec3::new(0.3, 0.0, 0.02) * face
				+ Vec3::new(0.0, 0.06, -0.04) * place
				+ Vec3::new(0.1, 0.05, 0.6);

			color
				.to_array()
				.into_iter()
				.chain([1.0])
				.flat_map(|channel| half(channel).to_le_bytes())
		});

		TextureData {
			width: 4,
			height: 4,
			faces: CUBE_FACES,
			texel: Texel::Rgba16Float,
			levels: vec![texels.collect()],
		}
	}

	/// A ball built with nothing but square roots: an octahedron whose faces
	/// are cut in four twice, every new corner pushed out onto the sphere.
	///
	/// The engine's own ball is built with a sine and a cosine, which two
	/// libraries round apart - measured with this test, which the built-in ball
	/// moved on another machine while every part of the bake agreed - so a
	/// digest that has to be the same on every machine cannot stand on it.
	fn ball() -> MeshData {
		let mut corners = vec![Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];
		let mut faces =
			vec![[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [
				0, 3, 5,
			]];

		for _ in 0..2 {
			faces = cut(&mut corners, &faces);
		}

		MeshData {
			vertices: corners
				.iter()
				.map(|corner| MeshVertex::new(*corner * 0.5, *corner, Vec2::ZERO))
				.collect(),
			indices: faces
				.iter()
				.flatten()
				.map(|&index| u32::try_from(index).expect("a small ball"))
				.collect(),
			..MeshData::default()
		}
	}

	/// Every face cut in four, the three new corners pushed out onto the
	/// sphere.
	fn cut(corners: &mut Vec<Vec3>, faces: &[[usize; 3]]) -> Vec<[usize; 3]> {
		let mut middle = |one: usize, two: usize| {
			corners.push(((corners[one] + corners[two]) * 0.5).normalize());
			corners.len() - 1
		};

		faces
			.iter()
			.flat_map(|&[a, b, c]| {
				let (ab, bc, ca) = (middle(a, b), middle(b, c), middle(c, a));

				[[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
			})
			.collect()
	}

	/// A world that reaches every part of a bake that could round differently
	/// on another machine: a color picture on the sRGB curve, turned by an
	/// angle, a cone, a sky of six faces, the sun and a curved surface.
	fn everything() -> World {
		let mut world = World::new();
		let picture = world.textures.insert("bricks", TextureData {
			width: 2,
			height: 2,
			faces: 1,
			texel: Texel::Rgba8Srgb,
			levels: vec![
				[[200, 60, 40, 255], [180, 170, 160, 255], [90, 90, 200, 255], [
					30, 200, 60, 255,
				]]
				.concat(),
				vec![130, 120, 110, 255],
			],
		});
		let turned = world.materials.insert("turned", Material {
			albedo: picture,
			uv_scale: Vec2::new(3.0, 2.0),
			uv_rotation: 0.7,
			uv_offset: Vec2::new(0.1, 0.3),
			..Material::DEFAULT
		});
		let sky = world.textures.insert("sky", cube());
		let ball = world.meshes.insert("ball", ball());

		world.sky = Heavens {
			kind: SkyKind::Cubemap,
			cubemap: sky,
			..Heavens::NONE
		};
		world.light = Vec3::new(-0.3, -1.0, 0.4);

		for (mesh, material, position, scale) in [
			(MeshId::QUAD, turned, Vec3::ZERO, Vec3::new(20.0, 1.0, 20.0)),
			(
				MeshId::CUBE,
				MaterialId::DEFAULT,
				Vec3::new(0.0, 2.0, -3.0),
				Vec3::new(8.0, 4.0, 0.5),
			),
			(ball, MaterialId::DEFAULT, Vec3::new(2.0, 1.0, 0.0), Vec3::ONE),
		] {
			let thing =
				world
					.entities
					.spawn_at(Transform { position, scale, ..Transform::IDENTITY });

			world
				.entities
				.set_renderable(thing, Renderable::of(mesh, material, Vec3::ONE));
		}

		let lamp = world.entities.spawn_at(Transform {
			position: Vec3::new(-2.0, 3.0, 1.0),
			rotation: Quat::from_xyzw(-0.5, 0.1, 0.0, 0.86),
			..Transform::IDENTITY
		});

		world
			.entities
			.set_light(lamp, Light::spot(Vec3::new(1.0, 0.8, 0.6), 6.0, 12.0, 0.3, 0.7));

		world
	}

	/// Sixty-four bits of FNV over a list of numbers' bits.
	fn digest(numbers: impl Iterator<Item = u32>) -> u64 {
		numbers.fold(0xCBF2_9CE4_8422_2325, |held, number| {
			number
				.to_le_bytes()
				.iter()
				.fold(held, |held, byte| (held ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01B3))
		})
	}

	#[test]
	fn a_whole_bake_answers_the_same_bytes_on_every_machine() {
		// everything the gather's digest reaches, now through the places, the
		// texels, the push, the passes and the fill: a lightmap of it and every
		// place on it, as bits. The ball of square roots has no second set, so
		// the path a thing with no place takes is in it too.
		let scene = Scene::of(&everything());
		let baked = bake(
			&scene,
			Settings {
				rays: 32,
				bounces: 2,
				..Settings::DEFAULT
			},
			threads(),
		)
		.expect("the world bakes");
		let answer =
			digest(
				baked
					.light
					.iter()
					.flat_map(|texel| texel.to_array().map(f32::to_bits))
					.chain(baked.places.iter().flat_map(|(_, place)| {
						[place.left, place.top, place.width, place.height]
					}))
					.chain([baked.width, baked.height]),
			);

		assert_eq!(baked.placeless.len(), 1, "the ball, which has no second set");
		// written down from the first run on one machine, as the gather's is
		assert_eq!(answer, 0x6772_B35B_5C31_F6F6, "the digest of the whole bake: {answer:#018x}");
	}

	#[test]
	fn a_bake_answers_the_same_numbers_on_every_machine() {
		let scene = Scene::of(&everything());
		let pattern = Pattern::new(64);
		let lit_once = |hit: &Hit, _: Vec3| {
			scene.surface(hit).map_or(Vec3::ZERO, |surface| {
				surface.emission + surface.diffuse * scene.direct(surface.at, surface.normal)
			})
		};
		let gathered = each(2000, threads(), |index| {
			let step = f32::from(u16::try_from(index).expect("a small count")) / 2000.0;
			let at = Vec3::new(17.0, 0.0, 13.0) * step + Vec3::new(-8.5, 0.0, -6.5);
			let direct = scene.direct(at, Vec3::Y);

			(scene.gather(at, Vec3::Y, seed(index, 0), &pattern, lit_once), direct)
		});
		let answer = digest(gathered.iter().flat_map(|(gathered, direct)| {
			[gathered.light, *direct]
				.into_iter()
				.flat_map(|light| light.to_array().map(f32::to_bits))
				.chain([gathered.behind])
		}));

		// written down from the first run on one machine; another machine
		// answering anything else is a bake that is not the same bytes there
		assert_eq!(
			answer, 0xE5C2_FF37_AFA7_873A,
			"the digest of two thousand points: {answer:#018x}"
		);
	}
}
