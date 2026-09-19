//! What a bake sees of a world: every still thing as triangles in the world,
//! what each is made of, the lamps, the sun and the sky.
//!
//! **Copied out once, and then the world is let go.** A bake runs on many
//! threads for as long as it takes, and the world it was asked about goes on
//! being stepped and edited meanwhile; so everything a ray can ask is taken
//! out of the world here, into plain lists in world space, and nothing below
//! reads the world again.
//!
//! **What stands still, and nothing else.** A bake is light worked out once for
//! things that will be where they are now when the light is read. So a thing
//! is baked when it is drawn - alive, shown, with a mesh drawn where it stands
//! rather than strewn, and not glass - and nothing that moves it: no body the
//! solver or a game moves drives it or anything it hangs off, no pose bends it,
//! and its [`Baking`] record does not say to leave it out. The same rule picks
//! the lamps, so that a lamp a character carries lights the room each frame and
//! throws nothing into the bake. Glass is left out because what passes through
//! it is not a surface's to keep, which is the same reason the pass before the
//! scene leaves it out. What a strewing lays over its ground is not in a bake
//! either: its copies are not entities.
//!
//! **What a surface is made of, to a ray, is the material the picture draws it
//! with**: its color times the thing's own tint times the paint on its
//! vertices times its color picture, read at the coordinates the shader reads
//! it at - scaled, turned and moved the same way - and how much of that is
//! metal, which sends nothing back diffusely. What it gives off is its glow.
//! A surface drawn unlit gives off what it shows and takes no light, because
//! that is what the picture of it does. Decals, a cutout's holes and the
//! normal map are not read: the first two are a gap recorded with this card,
//! and the third turns the shading and not the surface.

use std::collections::HashSet;

use colby_core::{
	abi::{
		BAKING, Baking, EntityId, Entry, MeshId, Transform, World,
		entity::MAX_ENTITIES,
		light::{Light, LightKind},
		material::{Blend, Material, Wrap},
		mesh::{MeshData, PaintVertex},
		physics::BodyKind,
		strew,
		texture::TextureId,
	},
	glam::{Vec2, Vec3, Vec4},
};

use crate::{
	picture::Picture,
	sky::Sky,
	tree::{Hit, Tree},
};

/// How far off a surface a ray starts, in world units.
///
/// A millimeter in a world measured in meters. A ray that started on the
/// surface itself would meet the triangle it left at a distance of a rounding
/// error about half the time; this is far above that error anywhere a world
/// of this engine's size reaches, and far below any gap a person builds.
pub const BIAS: f32 = 1.0e-3;

/// One corner of one triangle, in the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Corner {
	/// Where it is.
	pub position: Vec3,

	/// Which way the surface faces there, of unit length.
	pub normal: Vec3,

	/// Where a picture on the first set of coordinates is read, already
	/// scaled, turned and moved as the material says.
	pub uv: Vec2,

	/// Where a picture on the second set is read: nought where the mesh has
	/// none.
	pub uv2: Vec2,

	/// The color it was painted, linear; white where nobody painted it.
	pub paint: Vec3,
}

/// What one thing is made of, as far as light is concerned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
	/// Its material's color times the thing's own tint.
	pub tint: Vec3,

	/// How much of it is metal.
	pub metallic: f32,

	/// What it gives off, its color already times its strength.
	pub emissive: Vec3,

	/// Its color picture, as a place in the scene's list, or `None` for white.
	pub albedo: Option<u32>,

	/// Its metal-and-roughness picture, the same way.
	pub finish: Option<u32>,

	/// Its glow picture, the same way.
	pub glow: Option<u32>,

	/// Whether the glow picture is read from the second set of coordinates.
	pub glow_uv2: bool,

	/// Whether a place past a picture's edge wraps round or is held.
	pub wrap: Wrap,

	/// Whether it is drawn as its own color with no light on it.
	pub unlit: bool,
}

/// One still thing, as the triangles it became.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Piece {
	/// The entity it is.
	pub entity: EntityId,

	/// The mesh it draws.
	pub mesh: MeshId,

	/// What it is made of.
	pub look: Look,

	/// How many texels across and down its mesh's second set was laid out
	/// for, or nought and nought for a mesh with none.
	pub sheet: [u32; 2],

	/// Its first triangle in the scene's list.
	pub first: u32,

	/// How many triangles it has, one after another from there.
	pub count: u32,
}

/// One lamp, where it stands and what it throws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lamp {
	/// Where it is.
	pub position: Vec3,

	/// The way a cone points, of unit length.
	pub direction: Vec3,

	/// Its color times its intensity.
	pub color: Vec3,

	/// How far it reaches.
	pub range: f32,

	/// The cone's falloff as a line in the cosine of the angle off its axis:
	/// what the cosine is multiplied by. Nought for a point.
	pub scale: f32,

	/// And what is added: one for a point, which is lit everywhere.
	pub offset: f32,

	/// Whether what it lights throws a shadow.
	pub shadow: bool,
}

/// What a ray found where it landed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
	/// Where.
	pub at: Vec3,

	/// Which way the surface faces there, of unit length.
	pub normal: Vec3,

	/// How much of the light arriving there it sends back diffusely, per
	/// channel: its color, less what of it is metal.
	pub diffuse: Vec3,

	/// What it gives off whatever arrives.
	pub emission: Vec3,

	/// Where on the second set of coordinates it is.
	pub uv2: Vec2,

	/// Which piece it belongs to, as a place in the scene's list.
	pub piece: u32,
}

/// Everything a bake needs of a world.
#[derive(Clone, Debug)]
pub struct Scene {
	corners: Vec<Corner>,
	triangles: Vec<[u32; 3]>,
	owners: Vec<u32>,
	pieces: Vec<Piece>,
	pictures: Vec<Picture>,
	lamps: Vec<Lamp>,
	sun: Option<Vec3>,
	sky: Sky,
	tree: Tree,
}

/// What reading the world's things is building up.
#[derive(Default)]
struct Gathering {
	corners: Vec<Corner>,
	triangles: Vec<[u32; 3]>,
	owners: Vec<u32>,
	pieces: Vec<Piece>,
	pictures: Vec<Picture>,
	/// Which texture each place in `pictures` came from, and every texture
	/// that could not be read, as `None`, so each is decoded once.
	read: Vec<(TextureId, Option<u32>)>,
}

impl Scene {
	/// Everything a bake needs of a world, copied out of it.
	///
	/// @param world - the world to bake
	#[must_use]
	pub fn of(world: &World) -> Self {
		let moving = moving(world);
		let mut gathering = Gathering::default();

		for (id, _, renderable) in world.entities.iter() {
			// a thing that strews its mesh has nothing standing where it stands:
			// its copies are laid over its ground and no bake sees them
			if !world.entities.shown(id)
				|| renderable.pose.is_some()
				|| left_out(world, id)
				|| strew::strews(&world.entities, id)
				|| !still(world, id, &moving)
			{
				continue;
			}

			let Some(mesh) = world
				.meshes
				.get(renderable.mesh)
				.map(Entry::value)
			else {
				continue;
			};
			let material = world
				.materials
				.get(renderable.material)
				.copied()
				.unwrap_or(Material::DEFAULT);

			if mesh.indices.is_empty() || material.blend == Blend::Alpha {
				continue;
			}

			let Some(placed) = world.entities.placed(id) else {
				continue;
			};
			let look = gathering.look_of(world, &material, renderable.color);

			gathering.add(id, renderable.mesh, look, mesh, placed, &material);
		}

		let corners: Vec<[Vec3; 3]> = gathering
			.triangles
			.iter()
			.map(|triangle| {
				triangle.map(|index| {
					gathering
						.corners
						.get(usize::try_from(index).unwrap_or(usize::MAX))
						.map_or(Vec3::ZERO, |corner| corner.position)
				})
			})
			.collect();

		Self {
			tree: Tree::build(&corners),
			corners: gathering.corners,
			triangles: gathering.triangles,
			owners: gathering.owners,
			pieces: gathering.pieces,
			pictures: gathering.pictures,
			lamps: lamps(world, &moving),
			// the way towards the sun, which is the other way from the way its
			// light travels
			sun: (-world.light).try_normalize(),
			sky: Sky::of(world),
		}
	}

	/// Every still thing, in the order the world holds them.
	#[must_use]
	pub fn pieces(&self) -> &[Piece] { &self.pieces }

	/// Every corner of every triangle.
	#[must_use]
	pub fn corners(&self) -> &[Corner] { &self.corners }

	/// Every triangle, as three places in [`corners`](Self::corners), wound
	/// counter-clockwise seen from the side it faces.
	#[must_use]
	pub fn triangles(&self) -> &[[u32; 3]] { &self.triangles }

	/// Every lamp that stands still.
	#[must_use]
	pub fn lamps(&self) -> &[Lamp] { &self.lamps }

	/// The way towards the sun, or `None` for a world with none.
	#[must_use]
	pub const fn sun(&self) -> Option<Vec3> { self.sun }

	/// What a ray that leaves the world brings back.
	#[must_use]
	pub const fn sky(&self) -> &Sky { &self.sky }

	/// The hierarchy the rays are traced through.
	#[must_use]
	pub const fn tree(&self) -> &Tree { &self.tree }

	/// What a ray found where it landed.
	///
	/// @param hit - where it landed
	/// @return the surface there, or `None` for a hit the scene has no
	/// triangle for
	#[must_use]
	pub fn surface(&self, hit: &Hit) -> Option<Surface> {
		self.surface_at(hit.triangle, hit.along, hit.across)
	}

	/// The surface at one place on one triangle.
	///
	/// @param triangle - which, as the scene numbers them
	/// @param along - how much of its second corner the place is
	/// @param across - how much of its third
	#[must_use]
	pub fn surface_at(&self, triangle: u32, along: f32, across: f32) -> Option<Surface> {
		let slot = usize::try_from(triangle).ok()?;
		let piece = *self.owners.get(slot)?;
		let look = self
			.pieces
			.get(usize::try_from(piece).ok()?)?
			.look;
		let [a, b, c] = self.triangles.get(slot)?.map(|index| {
			self.corners
				.get(usize::try_from(index).unwrap_or(usize::MAX))
		});
		let (a, b, c) = (a?, b?, c?);
		let first = 1.0 - along - across;
		// three equal corners are that value to the bit, where a weighted sum
		// of them would be it to within a rounding: an unpainted mesh stays
		// white and a flat face keeps its one normal exactly
		let blend3 = |pick: fn(&Corner) -> Vec3| {
			let (one, two, three) = (pick(a), pick(b), pick(c));

			if one == two && two == three {
				return one;
			}

			one * first + two * along + three * across
		};
		let blend2 = |pick: fn(&Corner) -> Vec2| {
			let (one, two, three) = (pick(a), pick(b), pick(c));

			if one == two && two == three {
				return one;
			}

			one * first + two * along + three * across
		};

		let flat = (b.position - a.position)
			.cross(c.position - a.position)
			.normalize_or_zero();
		let uv = blend2(|corner| corner.uv);
		let uv2 = blend2(|corner| corner.uv2);
		let seen = |slot: Option<u32>, at: Vec2| {
			slot.and_then(|slot| self.pictures.get(usize::try_from(slot).ok()?))
				.map_or(Vec4::ONE, |picture| picture.at(at, look.wrap))
		};

		let color = look.tint * blend3(|corner| corner.paint) * seen(look.albedo, uv).truncate();
		let metallic = (look.metallic * seen(look.finish, uv).z).clamp(0.0, 1.0);
		let glowing = seen(look.glow, if look.glow_uv2 { uv2 } else { uv }).truncate();
		let (diffuse, emission) = if look.unlit {
			(Vec3::ZERO, color)
		} else {
			(color * (1.0 - metallic), look.emissive * glowing)
		};

		Some(Surface {
			at: blend3(|corner| corner.position),
			normal: blend3(|corner| corner.normal).normalize_or(flat),
			diffuse,
			emission,
			uv2,
			piece,
		})
	}
}

impl Gathering {
	/// What a material makes a thing out of, its pictures read on the way.
	fn look_of(&mut self, world: &World, material: &Material, tint: Vec3) -> Look {
		Look {
			tint: material.base_color * tint,
			metallic: material.metallic,
			emissive: material.emissive * material.emissive_strength,
			albedo: self.picture(world, material.albedo),
			finish: self.picture(world, material.finish),
			glow: self.picture(world, material.glow),
			glow_uv2: material.glow_uv2,
			wrap: material.wrap,
			unlit: material.unlit,
		}
	}

	/// A texture's place in the list of pictures, reading it the first time.
	///
	/// No texture, and one that cannot be read as a flat picture, are both
	/// white - which is what the renderer binds for either.
	fn picture(&mut self, world: &World, id: TextureId) -> Option<u32> {
		if !id.is_some() {
			return None;
		}

		if let Some((_, place)) = self.read.iter().find(|(read, _)| *read == id) {
			return *place;
		}

		let place = world
			.textures
			.get(id)
			.and_then(|entry| Picture::of(entry.value()))
			.and_then(|picture| {
				let place = u32::try_from(self.pictures.len()).ok()?;

				self.pictures.push(picture);

				Some(place)
			});

		self.read.push((id, place));

		place
	}

	/// One thing's triangles, in the world.
	fn add(
		&mut self,
		entity: EntityId,
		mesh_id: MeshId,
		look: Look,
		mesh: &MeshData,
		placed: Transform,
		material: &Material,
	) {
		let (Ok(base), Ok(piece), Ok(first)) = (
			u32::try_from(self.corners.len()),
			u32::try_from(self.pieces.len()),
			u32::try_from(self.triangles.len()),
		) else {
			return;
		};

		let matrix = placed.matrix();
		// what carries a normal under a translation, a rotation and a scale:
		// the rotation over the scale, which is the model matrix over the
		// square of the scale. An axis scaled to nothing keeps the normal it had.
		let over = Vec3::select(
			placed.scale.cmpeq(Vec3::ZERO),
			Vec3::ONE,
			(placed.scale * placed.scale).recip(),
		);
		// a mirror turns a triangle's winding inside out; turning it back keeps
		// the front of every triangle on the outside of what it is part of
		let mirrored = placed.scale.x * placed.scale.y * placed.scale.z < 0.0;
		let turn = turn_of(material.uv_rotation);

		for (index, vertex) in mesh.vertices.iter().enumerate() {
			let paint = mesh
				.paint
				.get(index)
				.copied()
				.unwrap_or(PaintVertex::PLAIN);
			let scaled = Vec2::from(vertex.uv) * material.uv_scale;

			self.corners.push(Corner {
				position: matrix.transform_point3(Vec3::from(vertex.position)),
				normal: matrix
					.transform_vector3(Vec3::from(vertex.normal) * over)
					.normalize_or_zero(),
				// scaled, turned and then moved, which is the shader's order
				uv: material.uv_offset + Vec2::new(turn.dot(scaled), turn.perp().dot(scaled)),
				uv2: Vec2::from(paint.uv2),
				paint: Vec3::new(
					painted(paint.color[0]),
					painted(paint.color[1]),
					painted(paint.color[2]),
				),
			});
		}

		let count = mesh.vertices.len();

		for corners in mesh.indices.chunks_exact(3) {
			let [a, b, c] = [corners[0], corners[1], corners[2]];

			if [a, b, c]
				.iter()
				.any(|&index| usize::try_from(index).map_or(true, |index| index >= count))
			{
				continue;
			}

			let (b, c) = if mirrored { (c, b) } else { (b, c) };

			self.triangles
				.push([base + a, base + b, base + c]);
			self.owners.push(piece);
		}

		let made = u32::try_from(self.triangles.len()).unwrap_or(first) - first;

		self.pieces.push(Piece {
			entity,
			mesh: mesh_id,
			look,
			sheet: mesh.sheet,
			first,
			count: made,
		});
	}
}

impl Lamp {
	/// One light in the world, as a bake reads it.
	///
	/// The cone's two numbers are the renderer's own: one over the width of
	/// the band between the two cosines, and the offset that puts its far edge
	/// at nought. The cosines are worked out in double precision and narrowed,
	/// which is the one place the lamps meet a function two libraries may round
	/// apart.
	///
	/// @param light - what it throws
	/// @param at - where it stands and which way it is turned, in the world
	#[must_use]
	pub fn of(light: Light, at: Transform) -> Self {
		let (scale, offset) = if light.kind == LightKind::Spot {
			let (inner, outer) = light.cone();
			let (inner, outer) = (cosine(inner), cosine(outer));
			let scale = (inner - outer).max(1.0e-4).recip();

			(scale, -outer * scale)
		} else {
			(0.0, 1.0)
		};

		Self {
			position: at.position,
			direction: (at.rotation * Vec3::NEG_Z).normalize_or(Vec3::NEG_Z),
			color: light.color * light.intensity,
			range: light.range,
			scale,
			offset,
			shadow: light.shadow,
		}
	}
}

/// Every entity a body the solver or a game moves drives.
fn moving(world: &World) -> HashSet<EntityId> {
	world
		.bodies
		.iter()
		.filter(|(_, body)| body.kind != BodyKind::Static && body.entity.is_some())
		.map(|(_, body)| body.entity)
		.collect()
}

/// Whether nothing moves an entity: no moving body drives it or anything it
/// hangs off.
fn still(world: &World, id: EntityId, moving: &HashSet<EntityId>) -> bool {
	let mut at = id;

	// bounded like every walk up the tree of entities, and for the same reason
	for _ in 0..MAX_ENTITIES {
		if !at.is_some() {
			return true;
		}

		if moving.contains(&at) {
			return false;
		}

		at = world.entities.parent(at);
	}

	true
}

/// Whether an entity's record says a bake leaves it out.
fn left_out(world: &World, id: EntityId) -> bool {
	world
		.entities
		.record(&BAKING, id)
		.is_some_and(|baking| Baking::skip(*baking))
}

/// Every lamp that is lit, shown and still.
fn lamps(world: &World, moving: &HashSet<EntityId>) -> Vec<Lamp> {
	world
		.entities
		.iter()
		.filter_map(|(id, ..)| {
			let light = world
				.entities
				.light(id)
				.copied()
				.filter(|light| light.is_lit())?;

			if !world.entities.shown(id) || left_out(world, id) || !still(world, id, moving) {
				return None;
			}

			Some(Lamp::of(light, world.entities.placed(id)?))
		})
		.collect()
}

/// A painted channel as a fraction.
fn painted(channel: u16) -> f32 { f32::from(channel) / f32::from(PaintVertex::WHOLE) }

/// How a material turns its coordinates, as the first row of the turn: the
/// cosine and the sine of its angle, so that `u' = cos u + sin v`. The second
/// row is this one's perpendicular, `(-sin, cos)`, which is the shader's
/// `v' = -sin u + cos v`.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "worked out in double precision and narrowed once, where it is stored"
)]
fn turn_of(angle: f32) -> Vec2 {
	let (sine, cosine) = f64::from(angle).sin_cos();

	Vec2::new(cosine as f32, sine as f32)
}

/// A cosine worked out in double precision and narrowed.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "worked out in double precision and narrowed once, where it is stored"
)]
fn cosine(angle: f32) -> f32 { f64::from(angle).cos() as f32 }

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			Body, Renderable, Shape,
			material::MaterialId,
			mesh::quad,
			texture::{Texel, TextureData},
		},
		glam::Quat,
	};

	use super::*;
	use crate::tree::Ray;

	/// An entity drawing a mesh with a material, standing somewhere.
	fn thing(world: &mut World, mesh: MeshId, material: MaterialId, at: Transform) -> EntityId {
		let id = world.entities.spawn_at(at);

		world
			.entities
			.set_renderable(id, Renderable::of(mesh, material, Vec3::ONE));

		id
	}

	/// The entities a scene baked, in its order.
	fn baked(scene: &Scene) -> Vec<EntityId> {
		scene
			.pieces()
			.iter()
			.map(|piece| piece.entity)
			.collect()
	}

	#[test]
	fn what_moves_or_is_not_drawn_is_not_baked_and_everything_else_is() {
		let mut world = World::new();
		let block = |x: f32| Transform::at(Vec3::new(x, 0.0, 0.0));
		let plain = thing(&mut world, MeshId::CUBE, MaterialId::DEFAULT, block(0.0));
		let pinned = thing(&mut world, MeshId::CUBE, MaterialId::DEFAULT, block(2.0));
		let falling = thing(&mut world, MeshId::CUBE, MaterialId::DEFAULT, block(4.0));
		let pushed = thing(&mut world, MeshId::CUBE, MaterialId::DEFAULT, block(6.0));
		let riding = thing(&mut world, MeshId::CUBE, MaterialId::DEFAULT, block(8.0));
		let hidden = thing(&mut world, MeshId::CUBE, MaterialId::DEFAULT, block(10.0));
		let glass = world
			.materials
			.insert("glass", Material { blend: Blend::Alpha, ..Material::DEFAULT });
		let pane = thing(&mut world, MeshId::CUBE, glass, block(12.0));
		let nothing = thing(&mut world, MeshId::NONE, MaterialId::DEFAULT, block(14.0));

		for (kind, entity) in [
			(BodyKind::Static, pinned),
			(BodyKind::Dynamic, falling),
			(BodyKind::Kinematic, pushed),
		] {
			world.bodies.spawn(
				Body::new(kind, Shape::cuboid(Vec3::splat(0.5)), Transform::IDENTITY)
					.driving(entity),
			);
		}

		world.entities.set_parent(riding, falling);
		world.entities.set_hidden(hidden, true);

		let scene = Scene::of(&world);

		assert_eq!(
			baked(&scene),
			vec![plain, pinned],
			"a plain block and one a still body holds"
		);
		assert!(!baked(&scene).contains(&riding), "not what hangs off a falling thing");
		assert!(!baked(&scene).contains(&pane), "not glass");
		assert!(!baked(&scene).contains(&nothing), "and not an entity with no mesh");
		assert_eq!(scene.triangles().len(), 24, "two cubes of twelve triangles");
		assert_eq!(scene.tree().len(), 24, "every one of them in the tree");
	}

	#[test]
	fn a_thing_that_strews_its_mesh_is_not_baked_where_it_stands() {
		let mut world = World::new();
		let ground = thing(&mut world, MeshId::CUBE, MaterialId::DEFAULT, Transform::IDENTITY);
		let strewing = thing(
			&mut world,
			MeshId::CUBE,
			MaterialId::DEFAULT,
			Transform::at(Vec3::new(3.0, 0.0, 0.0)),
		);

		world.entities.set_parent(strewing, ground);
		if let Some(rule) = world
			.entities
			.record_mut(&strew::STREWING, strewing)
		{
			rule.strews = 1;
		}

		assert_eq!(
			baked(&Scene::of(&world)),
			vec![ground],
			"the ground is baked, and nothing stands where the strewing does"
		);
	}

	#[test]
	fn what_its_record_leaves_out_is_neither_a_surface_nor_a_lamp_of_the_bake() {
		let mut world = World::new();
		let kept = thing(&mut world, MeshId::CUBE, MaterialId::DEFAULT, Transform::IDENTITY);
		let skipped = thing(
			&mut world,
			MeshId::CUBE,
			MaterialId::DEFAULT,
			Transform::at(Vec3::new(3.0, 0.0, 0.0)),
		);
		let lamp = world
			.entities
			.spawn_at(Transform::at(Vec3::new(0.0, 4.0, 0.0)));
		let other = world
			.entities
			.spawn_at(Transform::at(Vec3::new(0.0, 5.0, 0.0)));

		world
			.entities
			.set_light(lamp, Light::point(Vec3::ONE, 1.0, 9.0));
		world
			.entities
			.set_light(other, Light::point(Vec3::ONE, 2.0, 9.0));

		for id in [skipped, other] {
			world
				.entities
				.record_mut(&BAKING, id)
				.expect("every entity carries it")
				.skip = 1;
		}

		let scene = Scene::of(&world);

		assert_eq!(baked(&scene), vec![kept], "the cube its record leaves out is not baked");
		assert_eq!(scene.lamps().len(), 1, "and nor is the lamp");
		assert_eq!(scene.lamps()[0].color, Vec3::ONE, "the one left is the other one");
	}

	#[test]
	fn a_thing_stands_where_its_parents_put_it() {
		let mut world = World::new();
		let parent = world.entities.spawn_at(Transform {
			position: Vec3::new(10.0, 0.0, 0.0),
			scale: Vec3::splat(2.0),
			..Transform::IDENTITY
		});
		let child = thing(
			&mut world,
			MeshId::CUBE,
			MaterialId::DEFAULT,
			Transform::at(Vec3::new(0.0, 3.0, 0.0)),
		);

		world.entities.set_parent(child, parent);

		let scene = Scene::of(&world);
		let hit = scene
			.tree()
			.nearest(&Ray::new(Vec3::new(10.0, 20.0, 0.0), Vec3::NEG_Y), f32::INFINITY)
			.expect("the child is under the ray");

		// the child stands six up inside a parent that doubles it, so its top
		// face is at six plus one
		assert!((hit.distance - 13.0).abs() < 1.0e-5, "its top at seven: {}", hit.distance);
	}

	#[test]
	fn a_mirrored_thing_still_faces_out() {
		let mut world = World::new();

		thing(&mut world, MeshId::CUBE, MaterialId::DEFAULT, Transform {
			scale: Vec3::new(-1.0, 1.0, 1.0),
			..Transform::IDENTITY
		});

		let scene = Scene::of(&world);

		for way in [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::Z] {
			let hit = scene
				.tree()
				.nearest(&Ray::new(way * -5.0, way), f32::INFINITY)
				.expect("the block is in the way");

			assert!(hit.front, "from outside along {way} the front is met");
		}
	}

	#[test]
	fn a_hit_reads_the_color_the_picture_draws_there() {
		let mut world = World::new();
		// two by two: red and green over blue and white, as numbers so the test
		// reads them back unbent
		let picture = world.textures.insert("check", TextureData {
			width: 2,
			height: 2,
			faces: 1,
			texel: Texel::Rgba8Unorm,
			levels: vec![
				[[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 255, 255]]
					.concat(),
				vec![128, 128, 128, 255],
			],
		});
		let painted = world.materials.insert("painted", Material {
			base_color: Vec3::new(0.5, 1.0, 1.0),
			albedo: picture,
			metallic: 0.25,
			emissive: Vec3::new(2.0, 0.0, 0.0),
			wrap: Wrap::Clamp,
			..Material::DEFAULT
		});
		let floor = thing(&mut world, MeshId::QUAD, painted, Transform::IDENTITY);

		world.entities.set_renderable(
			floor,
			Renderable::of(MeshId::QUAD, painted, Vec3::new(1.0, 0.5, 1.0)),
		);

		let scene = Scene::of(&world);
		// the quad's corner at -x, -z carries (0, 0): straight down onto the
		// middle of the top left texel
		let hit = scene
			.tree()
			.nearest(&Ray::new(Vec3::new(-0.25, 1.0, -0.25), Vec3::NEG_Y), f32::INFINITY)
			.expect("the floor");
		let surface = scene.surface(&hit).expect("a surface");
		let red = Vec3::new(0.5, 0.0, 0.0);

		assert_eq!(surface.normal, Vec3::Y, "a flat face keeps its one normal exactly");
		assert!(
			(surface.diffuse - red * 0.75).abs().max_element() < 1.0e-6,
			"red, times the material's half, the tint's half, less the quarter that is metal: {}",
			surface.diffuse
		);
		assert_eq!(surface.emission, Vec3::new(2.0, 0.0, 0.0), "and it gives off its glow");
		assert!(
			(surface.at - Vec3::new(-0.25, 0.0, -0.25)).length() < 1.0e-6,
			"where the ray landed"
		);
	}

	#[test]
	fn a_stretched_ball_faces_the_way_its_stretched_surface_does() {
		let mut world = World::new();
		let stretch = Vec3::new(3.0, 1.0, 0.5);

		thing(&mut world, MeshId::SPHERE, MaterialId::DEFAULT, Transform {
			scale: stretch,
			..Transform::IDENTITY
		});

		let scene = Scene::of(&world);
		// the ball is half a unit across, so the stretched one is an
		// ellipsoid of these half-axes, whose normal at a point is the point
		// divided by the square of each
		let axes = stretch * 0.5;

		for corner in scene.corners() {
			let normal = (corner.position / (axes * axes)).normalize_or_zero();

			if normal == Vec3::ZERO {
				continue;
			}

			assert!(
				(corner.normal - normal).length() < 1.0e-5,
				"at {} the surface faces {normal}, not {}",
				corner.position,
				corner.normal
			);
		}
	}

	#[test]
	fn a_surface_nobody_painted_is_its_color_to_the_bit_wherever_a_ray_lands() {
		let mut world = World::new();
		let tinted = world.materials.insert("tinted", Material {
			base_color: Vec3::new(0.3, 0.6, 0.9),
			..Material::DEFAULT
		});

		thing(&mut world, MeshId::SPHERE, tinted, Transform::IDENTITY);

		let scene = Scene::of(&world);
		let count = u32::try_from(scene.triangles().len()).expect("a small ball");

		for step in 0..500_u32 {
			let (along, across) = (
				f32::from(u16::try_from(step % 97).expect("small")) / 197.0,
				f32::from(u16::try_from(step % 89).expect("small")) / 181.0,
			);
			let surface = scene
				.surface_at(step % count, along, across)
				.expect("a point on the ball");

			assert_eq!(surface.diffuse, Vec3::new(0.3, 0.6, 0.9), "at {along}, {across}");
		}
	}

	#[test]
	fn an_unlit_surface_gives_off_its_color_and_takes_no_light() {
		let mut world = World::new();
		let unlit = world.materials.insert("unlit", Material {
			base_color: Vec3::new(0.2, 0.4, 0.8),
			emissive: Vec3::new(5.0, 5.0, 5.0),
			unlit: true,
			..Material::DEFAULT
		});

		thing(&mut world, MeshId::QUAD, unlit, Transform::IDENTITY);

		let surface = Scene::of(&world)
			.surface_at(0, 0.25, 0.25)
			.expect("a point on the quad");

		assert_eq!(surface.diffuse, Vec3::ZERO, "no light is sent back");
		assert_eq!(surface.emission, Vec3::new(0.2, 0.4, 0.8), "what it shows, and not its glow");
	}

	#[test]
	fn the_coordinates_are_scaled_turned_and_moved_as_the_shader_reads_them() {
		let mut world = World::new();
		let turned = world.materials.insert("turned", Material {
			uv_scale: Vec2::new(2.0, 3.0),
			uv_rotation: std::f32::consts::FRAC_PI_2,
			uv_offset: Vec2::new(0.25, 0.5),
			..Material::DEFAULT
		});

		thing(&mut world, MeshId::QUAD, turned, Transform::IDENTITY);

		let scene = Scene::of(&world);
		let plain = quad();

		for (corner, vertex) in scene.corners().iter().zip(&plain.vertices) {
			let [u, v] = vertex.uv;
			// a quarter turn: `u' = cos u + sin v = 3 v`, `v' = -sin u + cos v = -2 u`
			let expected = Vec2::new(0.25, 0.5) + Vec2::new(3.0 * v, -2.0 * u);

			assert!(
				(corner.uv - expected).length() < 1.0e-6,
				"({u}, {v}) goes to {expected}, not {}",
				corner.uv
			);
		}
	}

	#[test]
	fn a_lamp_is_read_as_the_renderer_packs_it() {
		let turned = Transform {
			position: Vec3::new(1.0, 2.0, 3.0),
			rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
			..Transform::IDENTITY
		};
		let spot = Lamp::of(Light::spot(Vec3::new(1.0, 0.5, 0.25), 4.0, 9.0, 0.2, 0.5), turned);
		let point = Lamp::of(Light::point(Vec3::ONE, 2.0, 5.0), turned);

		assert_eq!(spot.position, turned.position, "where the entity stands");
		assert!(
			(spot.direction - Vec3::NEG_X).length() < 1.0e-6,
			"down its own -z, turned: {}",
			spot.direction
		);
		assert_eq!(spot.color, Vec3::new(4.0, 2.0, 1.0), "its color times its intensity");
		let (edge, middle) = (0.5_f32.cos() * spot.scale, 0.2_f32.cos() * spot.scale);

		assert!((edge + spot.offset).abs() < 1.0e-5, "the cone's line is nought at its edge");
		assert!((middle + spot.offset - 1.0).abs() < 1.0e-5, "and one at the edge of its middle");
		assert_eq!((point.scale, point.offset), (0.0, 1.0), "a point is lit everywhere");
	}
}
