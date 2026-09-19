//! The probes: what a bake kept of the light arriving at points of the air,
//! read where the sky's diffuse half was by everything the lightmap keeps no
//! place for.
//!
//! One flat picture the world names and the grid it was laid on - @ref
//! [`probes`](colby_core::abi::probes) for both - and nothing else: the picture
//! is read through the lightmap's sampler, the grid rides in the frame's
//! uniform, and there is no pass here, because the expensive half of a probe is
//! the bake. @ref `colby_bake::probes`.
//!
//! **In the frame's own group**, for the environment's reason, and **one
//! picture rather than a volume**: the bake's pictures compile to flat
//! textures, and a flat one is what the device already has uploaded. Six faces
//! side by side and the grid's layers one under another means a point is read
//! with two taps a face, one in each of the two layers it stands between, and a
//! blend by hand between them - where a volume would be one tap and a second
//! kind of texture for one reader.
//!
//! **Who reads them**: a thing with no place on the lightmap this frame, whose
//! material is lit, whose box reaches into the grid's - a thing a body moves, a
//! thing bones bend, glass, a thing a bake was told to leave out, a still thing
//! whose mesh has no second set. A thing wholly outside the grid is drawn as it
//! would be in a world nobody baked.
//!
//! **A thing that reads them is drawn by a pipeline of its own**, the
//! lightmap's reason: the sum that reads probes and the sum that does not are
//! two arrangements of the same terms, and a compiler handed both may round the
//! second as it rounds the first. So a frame that reads none - a world nobody
//! baked, a bake that kept none, either switch off - draws with the pipelines
//! and the arithmetic it drew with before there were probes. @ref
//! `crate::scene::Ambient`.

use colby_core::abi::probes::Grid;
use wgpu::{
	BindGroupLayoutEntry, BindingType, Device, Queue, ShaderStages, TextureSampleType,
	TextureView, TextureViewDimension,
};

use crate::{cull::Placed, lightmap};

/// Whether a world's probes are read at all.
///
/// A switch for the lightmap's reason: what it is for is measuring what reading
/// them costs, and taking the picture a build from before them would have
/// taken. `r.lightmap` switches them off with the lightmap, since both are what
/// a bake kept, and this one alone leaves the lightmap read.
pub const ENABLED: &str = "r.probes";

/// The probes the scene reads, and the grid they are read through.
pub(crate) struct Probes {
	/// A view of one texel of nothing, for the frames that read no probes.
	blank: TextureView,

	/// The view the bind group holds: the world's picture, or the blank one.
	bound: TextureView,

	/// Where the bound picture's probes stand, or [`Grid::NONE`] while the
	/// blank one is bound.
	grid: Grid,

	/// Which texture the bound view was made from, or nothing while the blank
	/// one is bound: the answer to whether this frame reads probes at all.
	source: Option<(u32, u32)>,
}

impl Probes {
	/// The blank picture, which is what a world with no probes gets.
	///
	/// @param device - the device to build against
	/// @param queue - the queue the texel of nothing is written through
	pub(crate) fn new(device: &Device, queue: &Queue) -> Self {
		let blank = lightmap::nothing(device, queue);

		Self {
			bound: blank.clone(),
			blank,
			grid: Grid::NONE,
			source: None,
		}
	}

	/// The view a bind group binds.
	pub(crate) const fn view(&self) -> &TextureView { &self.bound }

	/// Whether this frame reads probes.
	pub(crate) const fn reads(&self) -> bool { self.source.is_some() }

	/// Binds the picture of the probes the world names, if what is bound should
	/// change.
	///
	/// A picture whose size is not what its grid says is not read: the texel a
	/// face of a probe is looked up at would be some other probe's, or past the
	/// picture's edge.
	///
	/// @param picture - the probes' picture as the scene has it uploaded, or
	/// nothing for a world that names none
	/// @param grid - where they stand
	/// @param wanted - whether the switches let probes be read at all
	/// @return whether the bound view changed, which is what makes the group
	/// stale
	pub(crate) fn update(
		&mut self,
		picture: Option<lightmap::Picture<'_>>,
		grid: Grid,
		wanted: bool,
	) -> bool {
		let read = picture.filter(|found| wanted && grid.picture() == Some(found.size));
		let source = read.as_ref().map(|found| found.source);

		self.grid = if read.is_some() { grid } else { Grid::NONE };

		if source == self.source {
			return false;
		}

		self.bound = read.map_or_else(|| self.blank.clone(), |found| found.view.clone());
		self.source = source;

		true
	}

	/// Whether a thing whose box is this reaches into the grid's, in a frame
	/// that reads probes at all.
	///
	/// @param placed - the thing's box, in the world
	pub(crate) fn covers(&self, placed: &Placed) -> bool {
		if !self.reads() {
			return false;
		}

		let reach = placed.edges[0].abs() + placed.edges[1].abs() + placed.edges[2].abs();
		let (low, high) = (placed.center - reach, placed.center + reach);

		high.cmpge(self.grid.from).all() && low.cmple(self.grid.to()).all()
	}

	/// What the frame's uniform says of the grid: its corner and its step, and
	/// how many probes along each axis - nought in all eight words for a frame
	/// that reads none, which no pipeline that draws such a frame reads.
	pub(crate) fn words(&self) -> ([f32; 4], [u32; 4]) {
		let [x, y, z] = self.grid.counts;

		(self.grid.from.extend(self.grid.step).to_array(), [x, y, z, 0])
	}
}

/// The entry the probes' picture takes in the frame's group.
///
/// @param binding - which binding it takes
pub(crate) const fn layout_entry(binding: u32) -> BindGroupLayoutEntry {
	BindGroupLayoutEntry {
		binding,
		visibility: ShaderStages::FRAGMENT,
		ty: BindingType::Texture {
			sample_type: TextureSampleType::Float { filterable: true },
			view_dimension: TextureViewDimension::D2,
			multisampled: false,
		},
		count: None,
	}
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			EntityId, Light, Material, MeshId, Post, Renderable, Sky, Texel, TextureData,
			TextureId, ToneMap, Transform, World,
			cvar::Value,
			material::{Blend, MaterialId},
			probes::{AXES, Probes as Named},
			texture::CUBE_FACES,
		},
		glam::{Quat, Vec3},
		utils::half::half,
	};

	use super::*;
	use crate::{Capture, Image, scene::MSAA};

	/// How big every capture here is.
	const SIZE: (u32, u32) = (320, 240);

	/// A grid over [`floor_world`]'s floor, whose top at nought is the middle
	/// of its first layer: two units a cell, six along x and z and two layers.
	const OVER_THE_FLOOR: Grid = Grid {
		from: Vec3::new(-6.0, -1.0, -6.0),
		step: 2.0,
		counts: [6, 2, 6],
	};

	/// A capture on the binary's one device, or `None` with no GPU.
	fn capture() -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, SIZE.0, SIZE.1) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// The picture a grid of probes is kept in, each face of each probe what
	/// `light` says of the probe and the face.
	fn kept<F: Fn([u32; 3], u32) -> [f32; 3]>(grid: Grid, light: F) -> TextureData {
		let [width, height] = grid.picture().expect("a small grid");
		let mut texels = vec![[0.0_f32; 3]; usize::try_from(width * height).expect("small")];

		for index in 0..grid.len() {
			let place = grid.place(index);

			for face in 0..6 {
				let [column, row] = grid.texel(place, face);

				texels[usize::try_from(row * width + column).expect("small")] =
					light(place, face);
			}
		}

		TextureData {
			width,
			height,
			faces: 1,
			texel: Texel::Rgba16Float,
			levels: vec![
				texels
					.iter()
					.flat_map(|[red, green, blue]| [*red, *green, *blue, 1.0])
					.flat_map(|channel| half(channel).to_le_bytes())
					.collect(),
			],
		}
	}

	/// Names a picture of probes on a grid as the world's.
	fn named(world: &mut World, grid: Grid, picture: TextureData) -> TextureId {
		let id = world
			.textures
			.insert("lightmaps/probes/test", picture);

		world.probes = Named { picture: id, grid };

		id
	}

	/// A console variable declared with its default and set.
	fn asked(world: &mut World, name: &str, default: Value, value: &str) {
		world.cvars.var(name, default, "");
		world.cvars.set(name, value);
	}

	/// How many samples a pixel is drawn with.
	fn samples(world: &mut World, count: &str) { asked(world, MSAA, Value::Float(1.0), count); }

	/// One of the two switches, on or off.
	fn switched(world: &mut World, name: &str, on: &str) {
		asked(world, name, Value::Bool(true), on);
	}

	/// The picture's own look out of the way and a flat ambient color: no
	/// curve, an exposure of one and no sky.
	fn plainly(world: &mut World, ambient: f32) {
		world.post = Post {
			tonemap: ToneMap::None,
			auto_exposure: false,
			exposure: 1.0,
			..Post::DEFAULT
		};
		world.sky = Sky::NONE;
		world.clear = Vec3::ZERO;
		world.ambient = Vec3::splat(ambient);
	}

	/// A box of a material standing somewhere, in a color of its own.
	fn slab(
		world: &mut World,
		(position, scale): (Vec3, Vec3),
		material: MaterialId,
		tint: Vec3,
	) -> EntityId {
		let id = world.entities.spawn_at(Transform {
			position,
			rotation: Quat::IDENTITY,
			scale,
		});

		world
			.entities
			.set_renderable(id, Renderable::of(MeshId::CUBE, material, tint));

		id
	}

	/// A floor ten units a side whose top is at nought, seen from above and in
	/// front, lit by a sun at a slant and a flat ambient color. Nothing gave it
	/// a place on a lightmap, so in a world with probes over it, it reads them.
	fn floor_world(ambient: f32, tint: Vec3) -> World {
		let mut world = World::new();

		plainly(&mut world, ambient);
		world.light = Vec3::new(0.3, -1.0, -0.4);
		world.camera.position = Vec3::new(0.0, 3.0, 4.0);
		world.camera.target = Vec3::ZERO;
		slab(
			&mut world,
			(Vec3::new(0.0, -0.5, 0.0), Vec3::new(10.0, 1.0, 10.0)),
			MaterialId::DEFAULT,
			tint,
		);

		world
	}

	/// How many pixels of two pictures differ, and by how much at most in any
	/// one channel.
	fn apart(one: &Image, other: &Image) -> (usize, u8) {
		one.pixels
			.chunks_exact(4)
			.zip(other.pixels.chunks_exact(4))
			.filter(|(a, b)| a != b)
			.fold((0, 0), |(count, widest), (a, b)| {
				let far = a
					.iter()
					.zip(b)
					.map(|(x, y)| x.abs_diff(*y))
					.max()
					.unwrap_or(0);

				(count + 1, widest.max(far))
			})
	}

	/// A shot, which the tests here always expect to render.
	fn shot(capture: &mut Capture, world: &mut World) -> Image {
		capture.shoot(world).expect("the capture renders")
	}

	/// The same floor with the sun traveling straight up, so that nothing it
	/// faces is lit by it and what arrives from everywhere is all of its light:
	/// under the sun at a slant it is past white, where no relation can be
	/// read.
	fn unlit_floor(ambient: f32, tint: Vec3) -> World {
		let mut world = floor_world(ambient, tint);

		world.light = Vec3::Y;

		world
	}

	/// Whether no pixel of a picture is at the top of its range in any channel
	/// of color, where two lights would read as one.
	fn unsaturated(image: &Image) -> bool {
		image
			.pixels
			.chunks_exact(4)
			.all(|pixel| pixel[..3].iter().all(|channel| *channel < 250))
	}

	#[test]
	fn probes_are_bound_once_and_read_only_where_their_picture_is_the_size_their_grid_says() {
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut probes = Probes::new(gpu.device(), gpu.queue());
		let view = lightmap::nothing(gpu.device(), gpu.queue());
		let grid = Grid {
			from: Vec3::new(-1.0, 0.0, -2.0),
			step: 0.5,
			counts: [2, 3, 4],
		};
		let picture =
			|revision, size| lightmap::Picture { view: &view, source: (5, revision), size };
		let small =
			|center: Vec3| Placed::new(center, [Vec3::X * 0.1, Vec3::Y * 0.1, Vec3::Z * 0.1]);

		assert!(!probes.update(None, grid, true), "a world naming none keeps the blank bound");
		assert!(!probes.reads(), "and reads none");
		assert_eq!(probes.words(), ([0.0; 4], [0; 4]), "and says nothing of a grid");
		assert!(
			!probes.update(Some(picture(1, [12, 11])), grid, true),
			"a picture of another size than its grid's is not read"
		);
		assert!(!probes.reads());
		assert!(probes.update(Some(picture(1, [12, 12])), grid, true), "one of its size is");
		assert!(probes.reads());
		assert_eq!(
			probes.words(),
			([-1.0, 0.0, -2.0, 0.5], [2, 3, 4, 0]),
			"and the uniform says where its grid is"
		);
		assert!(
			!probes.update(Some(picture(1, [12, 12])), grid, true),
			"the same keeps the group"
		);
		assert!(probes.update(Some(picture(2, [12, 12])), grid, true), "a new revision does not");

		// the grid's box runs from (-1, 0, -2) to (0, 1.5, 0)
		assert!(probes.covers(&small(Vec3::new(-0.5, 0.5, -1.0))), "a thing inside");
		assert!(probes.covers(&small(Vec3::new(-1.05, 0.5, -1.0))), "a thing reaching in");
		assert!(!probes.covers(&small(Vec3::new(-1.2, 0.5, -1.0))), "a thing wholly outside");
		assert!(!probes.covers(&small(Vec3::new(0.5, 0.5, -1.0))), "and past the far corner");
		assert!(probes.update(Some(picture(2, [12, 12])), grid, false), "the switch off unbinds");
		assert!(!probes.reads(), "and reads none");
		assert!(!probes.covers(&small(Vec3::new(-0.5, 0.5, -1.0))), "so nothing reaches them");
	}

	#[test]
	fn a_frame_that_reads_no_probes_draws_what_a_world_with_none_draws() {
		// the old bytes, four ways: no probes named, the switch off, every bake's
		// light off, and a grid that stands somewhere else - a sun and a lamp as
		// well, so that a change to the sum that reached either would show. And
		// the control of the control: the probes read are another picture.
		let Some(mut capture) = capture() else {
			return;
		};

		for count in ["1", "4"] {
			let mut world = floor_world(0.35, Vec3::new(0.8, 0.6, 0.4));
			let lamp = world
				.entities
				.spawn_at(Transform::at(Vec3::new(-1.0, 0.8, 0.5)));

			world
				.entities
				.set_light(lamp, Light::point(Vec3::new(1.0, 0.8, 0.6), 2.0, 4.0));
			samples(&mut world, count);

			let none = shot(&mut capture, &mut world);

			named(&mut world, OVER_THE_FLOOR, kept(OVER_THE_FLOOR, |_, _| [0.9, 0.2, 0.1]));

			let read = shot(&mut capture, &mut world);

			assert_eq!(capture.scene_mut().probed_drawn(), 1, "the floor read them");

			switched(&mut world, ENABLED, "0");
			let off = shot(&mut capture, &mut world);
			switched(&mut world, ENABLED, "1");

			switched(&mut world, lightmap::ENABLED, "0");
			let dark = shot(&mut capture, &mut world);
			switched(&mut world, lightmap::ENABLED, "1");

			world.probes.grid = Grid {
				from: Vec3::new(20.0, 0.0, 20.0),
				..OVER_THE_FLOOR
			};
			let away = shot(&mut capture, &mut world);

			assert_eq!(
				capture.scene_mut().probed_drawn(),
				0,
				"a grid elsewhere is read by nothing"
			);
			assert!(
				none.pixels == off.pixels,
				"at {count}: the switch off draws what none draws"
			);
			assert!(none.pixels == dark.pixels, "and so does the lightmap's switch");
			assert!(none.pixels == away.pixels, "and a thing wholly outside the grid");
			assert!(apart(&none, &read).0 > 10_000, "while reading them is another picture");
		}
	}

	/// A cube of one level for a sky, every face its own color.
	fn sky_cube() -> TextureData {
		let faces = [
			[0.9, 0.3, 0.2],
			[0.2, 0.8, 0.3],
			[0.4, 0.5, 0.9],
			[0.3, 0.2, 0.1],
			[0.7, 0.7, 0.2],
			[0.2, 0.6, 0.7],
		];

		TextureData {
			width: 4,
			height: 4,
			faces: CUBE_FACES,
			texel: Texel::Rgba16Float,
			levels: vec![
				faces
					.iter()
					.flat_map(|color| std::iter::repeat_n(*color, 16))
					.flat_map(|[red, green, blue]| [red, green, blue, 1.0])
					.flat_map(|channel| half(channel).to_le_bytes())
					.collect(),
			],
		}
	}

	/// The floor lit by nothing but what arrives from everywhere - a sky of six
	/// colors or a flat color - under probes of nothing where asked.
	fn everywhere(
		capture: &mut Capture,
		tint: Vec3,
		(skied, count): (bool, &str),
		probed: bool,
	) -> Image {
		let mut world = floor_world(0.35, tint);

		// straight up, so nothing the floor faces is lit by it
		world.light = Vec3::Y;
		samples(&mut world, count);

		if skied {
			let cube = world.textures.insert("skies/test", sky_cube());

			world.sky = Sky::environment(cube);
		}

		if probed {
			named(&mut world, OVER_THE_FLOOR, kept(OVER_THE_FLOOR, |_, _| [0.0; 3]));
		}

		shot(capture, &mut world)
	}

	#[test]
	fn probes_of_nothing_leave_the_reflection_and_take_the_whole_diffuse_half() {
		// the composition in one relation, the lightmap's: a surface under probes
		// of nothing is the same surface painted black and read by none, to the
		// byte, under a flat ambient color and under a sky of six colors
		let Some(mut capture) = capture() else {
			return;
		};
		let tint = Vec3::new(0.8, 0.6, 0.4);

		for asked_for in [(false, "1"), (false, "4"), (true, "1"), (true, "4")] {
			let probed = everywhere(&mut capture, tint, asked_for, true);
			let black = everywhere(&mut capture, Vec3::ZERO, asked_for, false);
			let unprobed = everywhere(&mut capture, tint, asked_for, false);

			assert!(
				probed.pixels == black.pixels,
				"{asked_for:?}: probes of nothing are the surface painted black: {:?}",
				apart(&probed, &black)
			);
			assert!(apart(&probed, &unprobed).0 > 10_000, "and the painted one unread is not");
		}
	}

	#[test]
	fn probes_holding_the_ambient_color_light_as_the_ambient_did() {
		// a flat ambient of a half and probes of a half on every face are one
		// light, and the pictures differ by `a (d + s)` against `a d + a s` and by
		// the flat normal's four-thousandth, which are roundings
		let Some(mut capture) = capture() else {
			return;
		};

		for count in ["1", "4"] {
			let mut world = unlit_floor(0.5, Vec3::new(0.8, 0.6, 0.4));

			samples(&mut world, count);

			let unprobed = shot(&mut capture, &mut world);

			named(&mut world, OVER_THE_FLOOR, kept(OVER_THE_FLOOR, |_, _| [0.5; 3]));

			let probed = shot(&mut capture, &mut world);

			named(&mut world, OVER_THE_FLOOR, kept(OVER_THE_FLOOR, |_, _| [0.0; 3]));

			let nothing = shot(&mut capture, &mut world);
			let (moved, widest) = apart(&unprobed, &probed);

			assert!(unsaturated(&unprobed), "at {count}: nothing past white");
			assert!(widest <= 1, "at {count}: within a level, {moved} pixels up to {widest}");
			assert!(apart(&probed, &nothing).0 > 10_000, "and probes of nothing are not it");
		}
	}

	#[test]
	fn a_surface_reads_the_face_its_normal_leans_into_and_no_other() {
		// a floor looks up: probes holding a half on their face up and nothing on
		// the rest light it as an ambient of a half does, and probes holding a
		// half on the face down alone light it as probes of nothing do - the face
		// it turns away from is not read at all, where the four beside it are,
		// by the square of the flat normal's four-thousandth
		let Some(mut capture) = capture() else {
			return;
		};
		let face_of = |way: Vec3| {
			AXES.iter()
				.position(|axis| *axis == way)
				.and_then(|face| u32::try_from(face).ok())
				.expect("a face looks that way")
		};
		let (up, down) = (face_of(Vec3::Y), face_of(Vec3::NEG_Y));
		// one world, its probes named again for each shot: a second world's
		// picture would land in the same slot at the same revision as the first's,
		// which a scene that keeps what it uploaded takes for the same picture
		let mut world = unlit_floor(0.5, Vec3::new(0.8, 0.6, 0.4));
		let mut probed = |world: &mut World, holding: &dyn Fn(u32) -> [f32; 3]| {
			named(world, OVER_THE_FLOOR, kept(OVER_THE_FLOOR, |_, face| holding(face)));

			shot(&mut capture, world)
		};
		let only_up = probed(&mut world, &|face| if face == up { [0.5; 3] } else { [0.0; 3] });
		let down_only =
			probed(&mut world, &|face| if face == down { [0.5; 3] } else { [0.0; 3] });
		let nothing = probed(&mut world, &|_| [0.0; 3]);

		world.probes = Named::NONE;

		let ambient = shot(&mut capture, &mut world);
		let (moved, widest) = apart(&ambient, &only_up);

		assert!(unsaturated(&ambient), "nothing past white");
		assert!(widest <= 1, "the face up is the ambient within a level: {moved} up to {widest}");
		assert!(down_only.pixels == nothing.pixels, "and the face down is not read at all");
		assert!(apart(&only_up, &nothing).0 > 10_000, "which is another picture");
	}

	#[test]
	fn a_point_between_two_layers_reads_between_them() {
		// layers a unit and a half apart with the floor's top a quarter of the
		// way up: nothing below and one above reads as an ambient of a quarter
		let Some(mut capture) = capture() else {
			return;
		};
		let grid = Grid {
			from: Vec3::new(-6.0, -1.5, -6.0),
			..OVER_THE_FLOOR
		};

		for count in ["1", "4"] {
			let mut world = unlit_floor(0.25, Vec3::new(0.8, 0.6, 0.4));

			samples(&mut world, count);

			let ambient = shot(&mut capture, &mut world);

			named(
				&mut world,
				grid,
				kept(grid, |[_, layer, _], _| [f32::from(u8::try_from(layer).expect("two")); 3]),
			);

			let probed = shot(&mut capture, &mut world);

			named(&mut world, grid, kept(grid, |_, _| [1.0; 3]));

			let above = shot(&mut capture, &mut world);
			let (moved, widest) = apart(&ambient, &probed);

			assert!(unsaturated(&above), "at {count}: nothing past white");
			assert!(
				widest <= 1,
				"at {count}: a quarter of the way, {moved} pixels up to {widest}"
			);
			assert!(apart(&probed, &above).0 > 10_000, "and not the layer above");
		}
	}

	#[test]
	fn outside_the_grid_s_box_a_point_reads_the_sky() {
		// a grid over the floor's left half: the right half of the floor is lit
		// by the sky as though there were no probes, within the rounding of the
		// other sum, and the left by the probes, which hold red
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = unlit_floor(0.35, Vec3::new(0.8, 0.6, 0.4));
		let unprobed = shot(&mut capture, &mut world);
		let left = Grid { counts: [3, 2, 6], ..OVER_THE_FLOOR };

		named(&mut world, left, kept(left, |_, _| [1.0, 0.0, 0.0]));

		let probed = shot(&mut capture, &mut world);
		let side = |image: &Image, right: bool| -> Vec<[u8; 4]> {
			let middle = image.width / 2;
			let (from, to) = if right {
				(middle + 24, image.width)
			} else {
				(0, middle - 24)
			};

			(image.height / 2..image.height)
				.flat_map(|y| (from..to).map(move |x| (x, y)))
				.map(|(x, y)| image.pixel(x, y))
				.collect()
		};
		let widest = side(&unprobed, true)
			.iter()
			.zip(side(&probed, true))
			.filter_map(|(was, now)| {
				was.iter()
					.zip(now)
					.map(|(a, b)| a.abs_diff(b))
					.max()
			})
			.max()
			.unwrap_or(0);
		let redder = side(&unprobed, false)
			.iter()
			.zip(side(&probed, false))
			.filter(|(was, now)| now[0] > was[0].saturating_add(8))
			.count();

		assert!(unsaturated(&probed), "nothing past white");
		assert!(widest <= 1, "the right half reads the sky, within a level: {widest}");
		assert!(redder > 5_000, "and the left the probes: {redder} pixels redder");
	}

	#[test]
	fn glass_reads_the_probes_round_it() {
		// a pane in a grid of its own, over a floor outside it: probes of nothing
		// draw the pane as the pane painted black and read by none, and the floor
		// behind it as it was
		let Some(mut capture) = capture() else {
			return;
		};
		let pane_over = Grid {
			from: Vec3::new(-1.25, 0.75, -0.25),
			step: 0.5,
			counts: [5, 5, 1],
		};
		let with = |tint: Vec3, probed: bool, capture: &mut Capture| {
			let mut world = floor_world(0.35, Vec3::splat(0.5));

			world.light = Vec3::Y;

			let glass = world.materials.insert("test/glass", Material {
				blend: Blend::Alpha,
				opacity: 0.5,
				..Material::DEFAULT
			});

			slab(&mut world, (Vec3::new(0.0, 2.0, 0.0), Vec3::new(2.0, 2.0, 0.1)), glass, tint);

			if probed {
				named(&mut world, pane_over, kept(pane_over, |_, _| [0.0; 3]));
			}

			let image = shot(capture, &mut world);

			(image, capture.scene_mut().probed_drawn())
		};
		let (probed, drawn) = with(Vec3::new(0.3, 0.6, 0.9), true, &mut capture);
		let (black, _) = with(Vec3::ZERO, false, &mut capture);
		let (unprobed, _) = with(Vec3::new(0.3, 0.6, 0.9), false, &mut capture);

		assert_eq!(drawn, 1, "the pane read them and the floor did not");
		assert!(probed.pixels == black.pixels, "a pane under probes of nothing is a black one");
		assert!(apart(&probed, &unprobed).0 > 1_000, "and not the pane unread");
	}

	#[test]
	fn things_that_read_the_probes_and_things_that_do_not_are_drawn_apart() {
		// two slabs of one mesh and material, a grid over the left one only: the
		// left reads the probes through their pipeline and the right is drawn by
		// the plain one exactly as with no probes at all. One batch for both would
		// draw both one way or the other
		let Some(mut capture) = capture() else {
			return;
		};
		let tint = Vec3::new(0.8, 0.6, 0.4);
		let mut world = World::new();

		plainly(&mut world, 0.35);
		world.light = Vec3::Y;
		world.camera.position = Vec3::new(0.0, 4.0, 5.0);
		world.camera.target = Vec3::ZERO;

		for x in [-2.6, 2.6] {
			slab(
				&mut world,
				(Vec3::new(x, -0.5, 0.0), Vec3::new(5.0, 1.0, 8.0)),
				MaterialId::DEFAULT,
				tint,
			);
		}

		// the left slab runs from -5.1 to -0.1 along x and the right from 0.1 to
		// 5.1: a grid to nought holds the whole of the one and none of the other
		let one = Grid {
			from: Vec3::new(-5.5, -1.5, -4.5),
			step: 0.5,
			counts: [11, 6, 18],
		};
		let both = Grid { counts: [22, 6, 18], ..one };
		let neither = shot(&mut capture, &mut world);

		named(&mut world, one, kept(one, |_, _| [0.9, 0.2, 0.1]));

		let left = shot(&mut capture, &mut world);

		assert_eq!(capture.scene_mut().probed_drawn(), 1, "one slab reads them");

		named(&mut world, both, kept(both, |_, _| [0.9, 0.2, 0.1]));

		let each = shot(&mut capture, &mut world);

		assert_eq!(capture.scene_mut().probed_drawn(), 2, "and then both");

		// each slab's half of the picture, less a band down the middle where the
		// two meet and the share of the sky sees both
		let half = |image: &Image, right_half: bool| -> Vec<u8> {
			let middle = image.width / 2;
			let (from, to) = if right_half {
				(middle + 8, image.width)
			} else {
				(0, middle - 8)
			};

			(0..image.height)
				.flat_map(|y| (from..to).map(move |x| (x, y)))
				.flat_map(|(x, y)| image.pixel(x, y))
				.collect()
		};

		assert!(half(&left, true) == half(&neither, true), "the slab outside draws as before");
		assert!(half(&left, false) == half(&each, false), "the one inside as when both are");
		assert!(half(&left, false) != half(&neither, false), "and that is another picture");
	}

	/// Gives a thing the whole of a sixteen by eight lightmap as its place.
	fn placed_whole(world: &mut World, id: EntityId) {
		if let Some(baking) = world
			.entities
			.record_mut(&colby_core::abi::BAKING, id)
		{
			baking.width = 16;
			baking.height = 8;
		}
	}

	#[test]
	fn a_thing_with_a_place_on_the_lightmap_or_an_unlit_one_reads_no_probes() {
		// a floor a bake gave a place reads the lightmap, and one drawn as its own
		// color reads no light at all: probes over either change nothing
		let Some(mut capture) = capture() else {
			return;
		};

		for unlit in [false, true] {
			let mut world = floor_world(0.35, Vec3::new(0.8, 0.6, 0.4));
			let floor = world
				.entities
				.iter()
				.map(|(id, ..)| id)
				.next()
				.expect("the floor is the one entity");

			if unlit {
				let flat = world
					.materials
					.insert("test/flat", Material { unlit: true, ..Material::DEFAULT });

				world.entities.set_renderable(
					floor,
					Renderable::of(MeshId::CUBE, flat, Vec3::new(0.8, 0.6, 0.4)),
				);
			} else {
				let red: Vec<u8> = std::iter::repeat_n([0.9_f32, 0.2, 0.1, 1.0], 16 * 8)
					.flatten()
					.flat_map(|channel| half(channel).to_le_bytes())
					.collect();

				world.lightmap = world
					.textures
					.insert("lightmaps/test", TextureData {
						width: 16,
						height: 8,
						faces: 1,
						texel: Texel::Rgba16Float,
						levels: vec![red],
					});

				placed_whole(&mut world, floor);
			}

			let without = shot(&mut capture, &mut world);

			named(&mut world, OVER_THE_FLOOR, kept(OVER_THE_FLOOR, |_, _| [0.0, 2.0, 0.0]));

			let with = shot(&mut capture, &mut world);

			assert_eq!(
				capture.scene_mut().probed_drawn(),
				0,
				"unlit {unlit}: nothing reads them"
			);
			assert!(without.pixels == with.pixels, "unlit {unlit}: and nothing moved");
		}
	}

	#[test]
	fn a_turned_surface_reads_its_faces_by_the_squares_of_its_normal() {
		// a slab turned an eighth of a turn about z: its two broad faces look up
		// and along x at once and its two narrow ones down and along x, each
		// normal's squares a half and a half. Probes holding one along x either
		// way, nothing up or down and a half along z light every face it shows as
		// an ambient of a half does - where the normal's parts themselves, seven
		// tenths each, would light the broad faces at seven tenths
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();

		plainly(&mut world, 0.5);
		world.light = Vec3::Y;
		world.camera.position = Vec3::new(3.0, 3.0, 4.0);
		world.camera.target = Vec3::ZERO;

		let turned = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::from_rotation_z(-std::f32::consts::FRAC_PI_4),
			scale: Vec3::new(4.0, 0.2, 4.0),
		});

		world.entities.set_renderable(
			turned,
			Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, Vec3::new(0.8, 0.6, 0.4)),
		);

		let grid = Grid {
			from: Vec3::splat(-4.0),
			step: 2.0,
			counts: [4, 4, 4],
		};
		let ambient = shot(&mut capture, &mut world);

		named(
			&mut world,
			grid,
			kept(grid, |_, face| {
				[[1.0; 3], [1.0; 3], [0.0; 3], [0.0; 3], [0.5; 3], [0.5; 3]]
					[usize::try_from(face).expect("six")]
			}),
		);

		let probed = shot(&mut capture, &mut world);
		let (moved, widest) = apart(&ambient, &probed);

		assert!(unsaturated(&ambient), "nothing past white");
		assert!(
			apart(&ambient, &shot_of_nothing(&mut capture, &mut world, grid)).0 > 10_000,
			"a light"
		);
		assert!(widest <= 1, "a half of the face along +x: {moved} pixels up to {widest}");
	}

	/// The same world under probes of nothing, on the same grid.
	fn shot_of_nothing(capture: &mut Capture, world: &mut World, grid: Grid) -> Image {
		named(world, grid, kept(grid, |_, _| [0.0; 3]));

		shot(capture, world)
	}

	#[test]
	fn probes_that_differ_along_x_are_read_where_the_point_stands_between_them() {
		// probes the same either side of the floor's middle, seen from straight in
		// front of it: the picture is its own mirror image, which a read half a
		// probe along x would not be
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = unlit_floor(0.0, Vec3::new(0.8, 0.8, 0.8));

		world.camera.position = Vec3::new(0.0, 5.0, 3.0);
		// the share of the sky and the reflections are worked out on the screen,
		// and round the floor's edges they read a silhouette that is not its own
		// mirror image either
		asked(&mut world, crate::occlusion::STRENGTH, Value::Float(1.0), "0");
		asked(&mut world, crate::reflection::STRENGTH, Value::Float(1.0), "0");

		// probes at x = -5, -3, -1, 1, 3, 5: brighter the further from the middle
		named(
			&mut world,
			OVER_THE_FLOOR,
			kept(OVER_THE_FLOOR, |[x, _, _], _| {
				let from_middle =
					[0.9, 0.5, 0.15, 0.15, 0.5, 0.9][usize::try_from(x).expect("six")];

				[from_middle, from_middle * 0.5, 0.1]
			}),
		);

		let probed = shot(&mut capture, &mut world);
		let width = probed.width;
		// the floor's slanted edges are drawn by a rasterizer's rule for which pixel
		// an edge takes, which is not its own mirror image: a pixel on one edge and
		// its mirror on the other can be floor and the black behind it, and a pixel
		// of floor beside the black is shaded apart from its mirror whatever lights
		// it - measured under probes of one value everywhere. Pairs whose pixels and
		// both their neighbors along the row are floor, then, the picture's own
		// first and last columns having one neighbor only
		let floor = |x: u32, y: u32| {
			[x.saturating_sub(1), x, (x + 1).min(width - 1)]
				.iter()
				.all(|&at| {
					probed.pixel(at, y)[..3]
						.iter()
						.any(|channel| *channel > 10)
				})
		};
		let pairs: Vec<([u8; 4], [u8; 4])> = (0..probed.height)
			.flat_map(|y| (1..width / 2).map(move |x| (x, y)))
			.filter(|&(x, y)| floor(x, y) && floor(width - 1 - x, y))
			.map(|(x, y)| (probed.pixel(x, y), probed.pixel(width - 1 - x, y)))
			.collect();
		let widest = pairs
			.iter()
			.map(|(one, other)| {
				one.iter()
					.zip(other)
					.take(3)
					.map(|(a, b)| a.abs_diff(*b))
					.max()
					.unwrap_or(0)
			})
			.max()
			.unwrap_or(0);

		assert!(unsaturated(&probed), "nothing past white");
		assert!(pairs.len() > 30_000, "most of the picture is floor: {} pairs", pairs.len());
		assert!(widest <= 1, "the picture is its own mirror image within a level: {widest}");
	}

	#[test]
	fn a_layer_is_found_on_a_grid_wider_than_it_is_deep() {
		// three layers of six by four probes, the floor's top at the middle of the
		// second: every probe of that layer holds a half and the other two layers
		// other light, so a row of the picture taken by the wrong stride reads it
		let Some(mut capture) = capture() else {
			return;
		};
		let grid = Grid {
			from: Vec3::new(-6.0, -3.0, -4.0),
			step: 2.0,
			counts: [6, 3, 4],
		};

		for count in ["1", "4"] {
			let mut world = unlit_floor(0.5, Vec3::new(0.8, 0.6, 0.4));

			samples(&mut world, count);

			let ambient = shot(&mut capture, &mut world);

			named(
				&mut world,
				grid,
				kept(grid, |[_, layer, _], _| {
					[[0.1; 3], [0.5; 3], [2.0; 3]][usize::try_from(layer).expect("three")]
				}),
			);

			let probed = shot(&mut capture, &mut world);
			let (moved, widest) = apart(&ambient, &probed);

			assert!(unsaturated(&ambient), "at {count}: nothing past white");
			assert!(widest <= 1, "at {count}: the second layer, {moved} pixels up to {widest}");
		}
	}

	#[test]
	fn a_probed_cutout_keeps_its_holes_and_reads_the_probes_on_the_rest() {
		// the probed pipelines have a masked entry point of their own: a cutout a
		// bake left out keeps its holes as holes and is lit by the probes where it
		// is solid
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();

		plainly(&mut world, 0.35);
		world.light = Vec3::Y;
		world.camera.position = Vec3::new(0.0, 5.0, 0.01);
		world.camera.target = Vec3::ZERO;

		let solid = [0xE0, 0xE0, 0xE0, 0xFF];
		let hole = [0xE0, 0xE0, 0xE0, 0x11];
		let picture = world.textures.insert("test/holed", TextureData {
			width: 2,
			height: 2,
			faces: 1,
			texel: Texel::Rgba8Srgb,
			levels: vec![[solid, hole, hole, solid].concat()],
		});
		let cutout = world.materials.insert("test/cutout", Material {
			blend: Blend::Mask,
			..Material::textured(picture)
		});
		let quad = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(4.0, 1.0, 4.0),
		});

		world
			.entities
			.set_renderable(quad, Renderable::of(MeshId::QUAD, cutout, Vec3::ONE));

		let bare = shot(&mut capture, &mut world);

		named(&mut world, OVER_THE_FLOOR, kept(OVER_THE_FLOOR, |_, _| [4.0, 0.0, 0.0]));

		let probed = shot(&mut capture, &mut world);
		let pairs = || {
			bare.pixels
				.chunks_exact(4)
				.zip(probed.pixels.chunks_exact(4))
		};
		let holes = pairs()
			.filter(|(was, _)| was[..3] == [0, 0, 0])
			.count();
		let kept_holes = pairs()
			.filter(|(was, now)| was[..3] == [0, 0, 0] && was == now)
			.count();
		let lit = pairs()
			.filter(|(was, now)| was[..3] != [0, 0, 0] && now[0] > was[0])
			.count();

		assert_eq!(capture.scene_mut().probed_drawn(), 1, "the cutout reads them");
		assert!(
			holes > 1000 && kept_holes == holes,
			"every hole stays a hole: {kept_holes} of {holes}"
		);
		assert!(lit > 1000, "and the solid half reads the red: {lit} pixels");
	}
}
