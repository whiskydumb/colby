//! The lightmap: what a bake kept of the light arriving at every still thing,
//! read where the sky's diffuse half was.
//!
//! One flat picture the world names, one sampler, and a table of where each
//! thing's light is on the picture. Small on purpose - there is no pass here
//! and no pipeline, because the expensive half of a lightmap is the bake, and
//! that happens on the processor, offline. @ref `colby_bake`.
//!
//! **In the frame's own group**, for the environment's reason: the scene binds
//! four groups and four is the floor a device has to offer. The picture and its
//! sampler are read by the fragment stage and the table by the vertex stage,
//! and none of the three changes per draw.
//!
//! **A table, not a word on the instance.** Every vertex location a device has
//! to offer is taken, and the two words an instance has spare are sixty-four
//! bits: four sixteen-bit fractions of the picture, which at its widest is an
//! eighth of a texel out. So each thing's place goes into one storage buffer
//! for the whole frame, the joints' arrangement, and an instance carries the
//! index of its own - @ref [`Placement`](crate::scene::Placement). The buffer
//! is sized for every entity a world can hold, so no frame can ask for more
//! than it has.
//!
//! **A place is a scale and an offset**, `(width / W, height / H, left / W, top
//! / H)` for a picture `W` texels across and `H` down: the thing's own second
//! set, nought to one across its sheet, carried onto the picture by one
//! multiply and one add in the vertex stage. Worked out in doubles from the
//! record's whole texels and narrowed once.
//!
//! **A thing that reads it is drawn by a pipeline of its own**, and that is
//! what keeps every other picture the bytes it was. The sum that reads a
//! lightmap and the sum that does not are two arrangements of the same terms,
//! and a compiler handed both in one function may rewrite the second to share
//! work with the first - which was measured moving pictures of worlds nobody
//! baked. So the scene's pipelines that draw a thing with no place hand the
//! shader a constant nought for the baked light, the reading folds away when it
//! is compiled, and a frame hands places out only while it reads a lightmap at
//! all: a world nobody baked, and a baked one with the switch off, draw with
//! the pipelines and the arithmetic they drew with before this file existed.
//! @ref `crate::scene::Way`.
//!
//! **The off state is one texel of nothing**, built here, because a binding
//! cannot be left empty. Nothing reads it: no thing has a place in a frame that
//! binds it. Black rather than white, so that a read that should not have
//! happened shows as light gone missing.
//!
//! **One level, read at level nought.** A coarser level would average the
//! texels on both sides of the gap between two charts, and every chart would
//! bleed into the next; the level is named rather than worked out from a
//! pixel's footprint, which also asks nothing of the derivatives.

use colby_core::{
	Result,
	abi::{Baking, MAX_ENTITIES},
	bytemuck, err,
};
use wgpu::{
	AddressMode, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor,
	BufferUsages, Device, Extent3d, FilterMode, MipmapFilterMode, Origin3d, Queue, Sampler,
	SamplerBindingType, SamplerDescriptor, ShaderStages, TexelCopyBufferLayout,
	TexelCopyTextureInfo, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
	TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
};

/// Whether a world's lightmap is read at all.
///
/// A switch for the environment's reason: what it is for is measuring what
/// reading the picture costs, and taking the picture a build from before it
/// would have taken.
pub const ENABLED: &str = "r.lightmap";

/// The format the texel of nothing is in: the one a bake's picture compiles to.
const FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// One place as the vertex stage reads it: how much of the picture the sheet
/// spans across and down, then where on the picture it starts across and down,
/// all four as fractions of the picture.
type Place = [f32; 4];

/// The picture a frame found the world naming, as the scene has it uploaded.
pub(crate) struct Picture<'a> {
	/// The uploaded texture's view.
	pub(crate) view: &'a TextureView,

	/// Which registry slot it came from and at which revision, which is what
	/// makes a bound picture stale.
	pub(crate) source: (u32, u32),

	/// How many texels across and down it is.
	pub(crate) size: [u32; 2],
}

/// The picture the scene samples, the sampler it reads through, and this
/// frame's places on it.
pub(crate) struct Lightmap {
	/// A view of one texel of nothing, for the frames that read no lightmap.
	///
	/// The view alone: a view holds its texture, and nothing here ever writes
	/// the texel again.
	blank: TextureView,

	/// The view the bind group holds: the world's picture, or the blank one.
	bound: TextureView,

	/// Bilinear and clamped, at one level.
	sampler: Sampler,

	/// Every place this frame has handed out, on the device.
	places: Buffer,

	/// The same, as they are handed out, kept so it allocates once.
	written: Vec<Place>,

	/// How many texels across and down the bound picture is, or nought and
	/// nought while the blank one is bound, which is what hands no place out.
	size: [u32; 2],

	/// Which texture the bound view was made from, or nothing while the blank
	/// one is bound.
	///
	/// This is the answer to "does this frame read a lightmap at all": a world
	/// naming nothing, naming a cube, naming a picture not uploaded yet, or
	/// with its switch off all arrive here as nothing.
	source: Option<(u32, u32)>,
}

impl Lightmap {
	/// The blank picture, the sampler and the table, which is what a world
	/// nobody baked gets.
	///
	/// @param device - the device to build against
	/// @param queue - the queue the texel of nothing is written through
	/// @return the lightmap, or why the table could not be described
	pub(crate) fn new(device: &Device, queue: &Queue) -> Result<Self> {
		let size = u64::try_from(MAX_ENTITIES * size_of::<Place>())
			.map_err(|_| err!(Graphics("the lightmap's table is too large to describe")))?;
		let places = device.create_buffer(&BufferDescriptor {
			label: Some("lightmap places"),
			size,
			usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let blank = nothing(device, queue);

		Ok(Self {
			bound: blank.clone(),
			blank,
			sampler: sampler(device),
			places,
			written: Vec::with_capacity(MAX_ENTITIES),
			size: [0, 0],
			source: None,
		})
	}

	/// The view a bind group binds.
	pub(crate) const fn view(&self) -> &TextureView { &self.bound }

	/// The sampler it reads through.
	pub(crate) const fn sampler(&self) -> &Sampler { &self.sampler }

	/// The table of places a bind group binds.
	pub(crate) const fn places(&self) -> &Buffer { &self.places }

	/// Whether this frame reads a lightmap.
	#[cfg(test)]
	pub(crate) const fn reads(&self) -> bool { self.source.is_some() }

	/// Binds the picture the world names, if what is bound should change.
	///
	/// @param picture - the picture the world names as the scene has it
	/// uploaded, or nothing for a world that names none or names something that
	/// is not a flat picture
	/// @param wanted - whether the switch lets a lightmap be read at all
	/// @return whether the bound view changed, which is what makes the group
	/// stale
	pub(crate) fn update(&mut self, picture: Option<Picture<'_>>, wanted: bool) -> bool {
		let read = picture.filter(|_| wanted);
		let source = read.as_ref().map(|found| found.source);

		self.size = read.as_ref().map_or([0, 0], |found| found.size);

		if source == self.source {
			return false;
		}

		self.bound = read.map_or_else(|| self.blank.clone(), |found| found.view.clone());
		self.source = source;

		true
	}

	/// Forgets last frame's places.
	pub(crate) fn begin(&mut self) { self.written.clear(); }

	/// Hands out a place for one thing's light, if it has one.
	///
	/// @param baking - the thing's record
	/// @return which entry of the table its place is, or nothing for a thing a
	/// bake gave no place to and in a frame that reads no lightmap
	pub(crate) fn take(&mut self, baking: Baking) -> Option<u32> {
		let place = place_of(baking, self.size)?;

		// cannot happen: a frame stages each entity once and there are no more
		// entities than this. Refusing rather than trusting that is a line, and
		// what the alternative costs is a write past the end of the buffer.
		if self.written.len() >= MAX_ENTITIES {
			return None;
		}

		let index = u32::try_from(self.written.len()).ok()?;
		self.written.push(place);

		Some(index)
	}

	/// Writes the places this frame handed out.
	///
	/// @param queue - the queue to write through
	pub(crate) fn upload(&self, queue: &Queue) {
		if self.written.is_empty() {
			return;
		}

		queue.write_buffer(&self.places, 0, bytemuck::cast_slice(&self.written));
	}

	/// The places handed out so far this frame.
	#[cfg(test)]
	pub(crate) fn written(&self) -> &[Place] { &self.written }
}

/// One thing's place as the vertex stage reads it.
///
/// @param baking - the thing's record, whole texels on the picture
/// @param size - how many texels across and down the picture is
/// @return how much of the picture the sheet spans across and down and where
/// it starts across and down, as fractions; nothing for a thing with no place
/// and for a picture with no texels
pub(crate) fn place_of(baking: Baking, [width, height]: [u32; 2]) -> Option<Place> {
	if !baking.is_baked() || width == 0 || height == 0 {
		return None;
	}

	let across = f64::from(width);
	let down = f64::from(height);

	Some([
		narrowed(f64::from(baking.width) / across),
		narrowed(f64::from(baking.height) / down),
		narrowed(f64::from(baking.left) / across),
		narrowed(f64::from(baking.top) / down),
	])
}

/// A double narrowed to the float the device reads.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "the narrowing is the point, and every value here is a fraction of a picture no \
	          wider than a float holds exactly"
)]
const fn narrowed(value: f64) -> f32 { value as f32 }

/// The three entries a lightmap adds to the frame's group.
///
/// @param texture - which binding the picture takes
/// @param sampler - which binding its sampler takes
/// @param places - which binding the table of places takes
pub(crate) fn layout_entries(
	texture: u32,
	sampler: u32,
	places: u32,
) -> [BindGroupLayoutEntry; 3] {
	[
		BindGroupLayoutEntry {
			binding: texture,
			visibility: ShaderStages::FRAGMENT,
			ty: BindingType::Texture {
				sample_type: TextureSampleType::Float { filterable: true },
				view_dimension: TextureViewDimension::D2,
				multisampled: false,
			},
			count: None,
		},
		BindGroupLayoutEntry {
			binding: sampler,
			visibility: ShaderStages::FRAGMENT,
			ty: BindingType::Sampler(SamplerBindingType::Filtering),
			count: None,
		},
		BindGroupLayoutEntry {
			binding: places,
			// read where a vertex is carried onto the picture and nowhere else:
			// the fragment stage is handed the coordinate, not the table
			visibility: ShaderStages::VERTEX,
			ty: BindingType::Buffer {
				ty: BufferBindingType::Storage { read_only: true },
				has_dynamic_offset: false,
				min_binding_size: None,
			},
			count: None,
		},
	]
}

/// Bilinear, clamped, and nothing between levels, because there is one.
///
/// Clamped rather than repeating: a place on the edge of the picture would
/// otherwise read the opposite edge's texels into its border. And without
/// anisotropy, the decals' reason: a sample stretched along a grazing angle
/// reaches further than the gap between two places.
fn sampler(device: &Device) -> Sampler {
	device.create_sampler(&SamplerDescriptor {
		label: Some("lightmap"),
		address_mode_u: AddressMode::ClampToEdge,
		address_mode_v: AddressMode::ClampToEdge,
		address_mode_w: AddressMode::ClampToEdge,
		mag_filter: FilterMode::Linear,
		min_filter: FilterMode::Linear,
		mipmap_filter: MipmapFilterMode::Nearest,
		..SamplerDescriptor::default()
	})
}

/// A view of one texel of nothing.
///
/// Written rather than left to the device's own clearing, so that what it holds
/// is said here and not somewhere else.
pub(crate) fn nothing(device: &Device, queue: &Queue) -> TextureView {
	let one = Extent3d {
		width: 1,
		height: 1,
		depth_or_array_layers: 1,
	};
	let texture = device.create_texture(&TextureDescriptor {
		label: Some("no lightmap"),
		size: one,
		mip_level_count: 1,
		sample_count: 1,
		dimension: TextureDimension::D2,
		format: FORMAT,
		usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
		view_formats: &[],
	});

	queue.write_texture(
		TexelCopyTextureInfo {
			texture: &texture,
			mip_level: 0,
			origin: Origin3d::ZERO,
			aspect: TextureAspect::All,
		},
		&[0; 8],
		TexelCopyBufferLayout {
			offset: 0,
			bytes_per_row: Some(8),
			rows_per_image: Some(1),
		},
		one,
	);

	texture.create_view(&TextureViewDescriptor::default())
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			BAKING, EntityId, Light, Material, MeshData, MeshId, MeshVertex, PaintVertex, Post,
			Renderable, Sky, Texel, TextureData, TextureId, ToneMap, Transform, World,
			cvar::Value,
			material::{Blend, MaterialId},
			texture::CUBE_FACES,
		},
		glam::{Quat, Vec3},
		utils::half::half,
	};

	use super::*;
	use crate::{Capture, Image, occlusion, reflection, scene::MSAA};

	/// How big every capture here is.
	const SIZE: (u32, u32) = (320, 240);

	/// What a baked point's light is written over the sum with, as the shader
	/// spells it: the one piece of the shader a frame that reads no lightmap
	/// must draw the same bytes without.
	const OVERWRITE: &str = "    if (baked.a > 0.5) {\n        indirect = baked.rgb * \
	                         indirect_diffuse + stand_in * ambient_specular;\n    }\n";

	/// A capture on the binary's one device, or `None` with no GPU.
	fn capture() -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, SIZE.0, SIZE.1) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// The scene's own shader with one piece of it replaced, which has to be in
	/// it exactly once.
	fn variant(find: &str, replace: &str) -> String {
		let source = include_str!("shader.wgsl");

		assert_eq!(source.matches(find).count(), 1, "`{find}` is in the shader exactly once");

		source.replace(find, replace)
	}

	/// A picture in the format a bake's compiles to, each texel what `texel`
	/// says of its column and row.
	fn painted<F: Fn(u32, u32) -> [f32; 3]>(width: u32, height: u32, texel: F) -> TextureData {
		let level = (0..height)
			.flat_map(|row| (0..width).map(move |column| (column, row)))
			.flat_map(|(column, row)| {
				let [red, green, blue] = texel(column, row);

				[red, green, blue, 1.0]
			})
			.flat_map(|channel| half(channel).to_le_bytes())
			.collect();

		TextureData {
			width,
			height,
			faces: 1,
			texel: Texel::Rgba16Float,
			levels: vec![level],
		}
	}

	/// The same, one color everywhere, sixteen texels by eight.
	fn even(value: [f32; 3]) -> TextureData { painted(16, 8, |_, _| value) }

	/// Gives a thing a place on the lightmap, as a bake would.
	fn place(world: &mut World, id: EntityId, [left, top, width, height]: [i32; 4]) {
		let baking = world
			.entities
			.record_mut(&BAKING, id)
			.expect("every entity carries the record");

		baking.left = left;
		baking.top = top;
		baking.width = width;
		baking.height = height;
	}

	/// Names a picture as the world's lightmap.
	fn named(world: &mut World, picture: TextureData) -> TextureId {
		let id = world.textures.insert("lightmaps/test", picture);
		world.lightmap = id;

		id
	}

	/// A console variable declared with its default and set.
	fn asked(world: &mut World, name: &str, default: Value, value: &str) {
		world.cvars.var(name, default, "");
		world.cvars.set(name, value);
	}

	/// How many samples a pixel is drawn with.
	fn samples(world: &mut World, count: &str) { asked(world, MSAA, Value::Float(1.0), count); }

	/// Whether the lightmap is read.
	fn reading(world: &mut World, on: &str) { asked(world, ENABLED, Value::Bool(true), on); }

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

	/// A box of the default material standing somewhere, in a color of its own.
	fn slab(world: &mut World, position: Vec3, scale: Vec3, tint: Vec3) -> EntityId {
		let id = world.entities.spawn_at(Transform {
			position,
			rotation: Quat::IDENTITY,
			scale,
		});

		world
			.entities
			.set_renderable(id, Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, tint));

		id
	}

	/// A floor whose top is at nought, seen from above and in front, lit by a
	/// sun at a slant and a flat ambient color.
	fn floor_world(ambient: f32, tint: Vec3) -> (World, EntityId) {
		let mut world = World::new();

		plainly(&mut world, ambient);
		world.light = Vec3::new(0.3, -1.0, -0.4);
		world.camera.position = Vec3::new(0.0, 3.0, 4.0);
		world.camera.target = Vec3::ZERO;

		let floor = slab(&mut world, Vec3::new(0.0, -0.5, 0.0), Vec3::new(10.0, 1.0, 10.0), tint);

		(world, floor)
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

	/// A baked record with nothing but a place.
	const fn baked(left: i32, top: i32, width: i32, height: i32) -> Baking {
		Baking { skip: 0, left, top, width, height }
	}

	#[test]
	fn a_place_is_its_whole_texels_over_the_pictures_to_the_bit() {
		// a picture whose sides are powers of two holds every fraction here
		// exactly, so the four numbers are the bits and not approximately them
		let place = place_of(baked(3, 5, 7, 2), [16, 8]).expect("a baked thing has a place");

		assert_eq!(
			place.map(f32::to_bits),
			[0.4375_f32, 0.25, 0.1875, 0.625].map(f32::to_bits),
			"how much of the picture the sheet spans, then where it starts, across then down"
		);

		// and one whose sides are not, the rooms street's picture and a place on
		// it, within the float's own precision of the fraction in doubles
		let place = place_of(baked(258, 1276, 58, 87), [979, 1713]).expect("baked");
		let wanted = [58.0 / 979.0, 87.0 / 1713.0, 258.0 / 979.0, 1276.0 / 1713.0];

		for (got, want) in place.iter().zip(wanted) {
			let off = (f64::from(*got) - want).abs();

			assert!(off <= want * 6.0e-8, "{got} is the float nearest {want}: {off}");
		}
	}

	#[test]
	fn a_thing_no_bake_reached_and_a_picture_of_nothing_give_no_place() {
		assert_eq!(place_of(Baking::NONE, [16, 8]), None, "a record a bake never wrote");
		assert_eq!(place_of(baked(2, 2, 5, 0), [16, 8]), None, "a place of no height");
		assert_eq!(place_of(baked(2, 2, 5, 5), [0, 8]), None, "a picture of no width");
		assert_eq!(place_of(baked(2, 2, 5, 5), [16, 0]), None, "and of no height");
	}

	#[test]
	fn a_picture_is_bound_once_and_places_are_handed_out_until_every_entity_has_one() {
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut lightmap = Lightmap::new(gpu.device(), gpu.queue()).expect("it builds");
		let view = nothing(gpu.device(), gpu.queue());
		let picture = |revision| Picture {
			view: &view,
			source: (5, revision),
			size: [16, 8],
		};

		assert!(!lightmap.update(None, true), "a world naming nothing keeps the blank bound");
		assert!(!lightmap.reads(), "and reads nothing");
		assert_eq!(lightmap.take(baked(8, 0, 8, 8)), None, "nor hands a place out");
		assert!(lightmap.update(Some(picture(1)), true), "a picture named binds it");
		assert!(lightmap.reads(), "and is read");
		assert!(!lightmap.update(Some(picture(1)), true), "the same again keeps the group");
		assert!(lightmap.update(Some(picture(2)), true), "a new revision does not");
		assert!(lightmap.update(Some(picture(2)), false), "the switch off binds the blank");
		assert!(!lightmap.reads(), "and reads nothing");

		// and hands no place out, which is what draws every thing with the
		// pipelines that read no lightmap
		lightmap.begin();

		assert_eq!(lightmap.take(baked(8, 0, 8, 8)), None, "off, a baked thing has no entry");
		assert!(lightmap.update(Some(picture(2)), true), "on again binds the picture");
		assert_eq!(lightmap.take(baked(8, 0, 8, 8)), Some(0), "the first place is entry nought");
		assert_eq!(lightmap.take(Baking::NONE), None, "a thing with none takes no entry");
		assert_eq!(lightmap.take(baked(0, 4, 4, 4)), Some(1), "and the next is one");
		assert_eq!(
			lightmap.written(),
			&[[0.5, 1.0, 0.5, 0.0], [0.25, 0.5, 0.0, 0.5]],
			"each its texels over the picture's"
		);

		lightmap.begin();

		let handed = (0..MAX_ENTITIES)
			.filter_map(|_| lightmap.take(baked(0, 0, 1, 1)))
			.count();

		assert_eq!(handed, MAX_ENTITIES, "every entity a world can hold has an entry");
		assert_eq!(lightmap.take(baked(0, 0, 1, 1)), None, "and there is none past that");
	}

	#[test]
	fn a_frame_that_reads_no_lightmap_draws_what_a_shader_with_none_draws() {
		// the old bytes, four ways: the switch off, a world naming no picture, a
		// thing with no place, and the shader with the overwrite cut out, reading
		// everything. A sun and a lamp too, so that a change to the sum that
		// reached either would be in the picture as well. And the control of the
		// control: the same world read is another picture.
		let Some(mut capture) = capture() else {
			return;
		};
		let unwritten = variant(OVERWRITE, "");

		for count in ["1", "4"] {
			let (mut world, floor) = floor_world(0.35, Vec3::new(0.8, 0.6, 0.4));
			let lamp = world
				.entities
				.spawn_at(Transform::at(Vec3::new(-1.0, 0.8, 0.5)));

			world
				.entities
				.set_light(lamp, Light::point(Vec3::new(1.0, 0.8, 0.6), 2.0, 4.0));
			samples(&mut world, count);
			place(&mut world, floor, [0, 0, 16, 8]);

			let picture = named(&mut world, even([0.9, 0.2, 0.1]));
			capture
				.scene_mut()
				.set_shader(include_str!("shader.wgsl"))
				.expect("the scene's own shader builds");

			reading(&mut world, "0");
			let off = capture
				.shoot(&mut world)
				.expect("the capture renders");

			reading(&mut world, "1");
			let read = capture
				.shoot(&mut world)
				.expect("the capture renders");

			world.lightmap = TextureId::NONE;
			let unnamed = capture
				.shoot(&mut world)
				.expect("the capture renders");

			world.lightmap = picture;
			place(&mut world, floor, [0, 0, 0, 0]);
			let unplaced = capture
				.shoot(&mut world)
				.expect("the capture renders");

			place(&mut world, floor, [0, 0, 16, 8]);
			capture
				.scene_mut()
				.set_shader(&unwritten)
				.expect("the shader without the overwrite builds");
			let cut = capture
				.shoot(&mut world)
				.expect("the capture renders");

			assert!(off.pixels == unnamed.pixels, "at {count}: off draws what naming none draws");
			assert!(off.pixels == unplaced.pixels, "and what a thing with no place draws");
			assert!(
				off.pixels == cut.pixels,
				"and what a shader with no overwrite draws, reading everything"
			);
			assert!(apart(&off, &read).0 > 10_000, "while reading it is another picture");
		}

		capture
			.scene_mut()
			.set_shader(include_str!("shader.wgsl"))
			.expect("the scene's own shader builds");
	}

	/// A cube of one level for a sky, every face its own color, so that what a
	/// surface reflects depends on which way it looks.
	fn sky_cube() -> TextureData {
		let faces = [
			[0.9, 0.3, 0.2],
			[0.2, 0.8, 0.3],
			[0.4, 0.5, 0.9],
			[0.3, 0.2, 0.1],
			[0.7, 0.7, 0.2],
			[0.2, 0.6, 0.7],
		];
		let level: Vec<u8> = faces
			.iter()
			.flat_map(|color| std::iter::repeat_n(*color, 16))
			.flat_map(|[red, green, blue]| [red, green, blue, 1.0])
			.flat_map(|channel| half(channel).to_le_bytes())
			.collect();

		TextureData {
			width: 4,
			height: 4,
			faces: CUBE_FACES,
			texel: Texel::Rgba16Float,
			levels: vec![level],
		}
	}

	/// The floor lit by nothing but what arrives from everywhere, a sky of six
	/// colors or a flat color, and a lightmap of nothing where asked.
	fn everywhere(
		capture: &mut Capture,
		tint: Vec3,
		(skied, count): (bool, &str),
		bake: bool,
	) -> Image {
		let (mut world, floor) = floor_world(0.35, tint);

		world.light = Vec3::Y;
		samples(&mut world, count);

		if skied {
			let cube = world.textures.insert("skies/test", sky_cube());
			world.sky = Sky::environment(cube);
		}

		if bake {
			named(&mut world, even([0.0, 0.0, 0.0]));
			place(&mut world, floor, [0, 0, 16, 8]);
		}

		capture
			.shoot(&mut world)
			.expect("the capture renders")
	}

	#[test]
	fn a_lightmap_of_nothing_leaves_the_reflection_and_takes_the_whole_diffuse_half() {
		// the composition in one relation: a surface lit by a picture of nothing
		// is the same surface painted black and not baked at all, to the byte -
		// the diffuse half's light is the lightmap's and nothing else's, and the
		// reflection is the environment's whatever the lightmap holds. Under a
		// flat ambient color and under a sky of six colors, and with the sun
		// traveling straight up, because its light on a painted surface is not
		// its light on a black one.
		let Some(mut capture) = capture() else {
			return;
		};
		let tint = Vec3::new(0.8, 0.6, 0.4);

		for asked_for in [(false, "1"), (false, "4"), (true, "1"), (true, "4")] {
			let baked_shot = everywhere(&mut capture, tint, asked_for, true);
			let black = everywhere(&mut capture, Vec3::ZERO, asked_for, false);
			let unbaked = everywhere(&mut capture, tint, asked_for, false);

			assert!(
				baked_shot.pixels == black.pixels,
				"{asked_for:?}: a lightmap of nothing is the surface painted black: {:?}",
				apart(&baked_shot, &black)
			);
			assert!(
				apart(&baked_shot, &unbaked).0 > 10_000,
				"and the painted surface unbaked is another picture"
			);
		}
	}

	#[test]
	fn a_lightmap_holding_the_ambient_color_draws_what_the_ambient_did() {
		// a flat ambient of a half and a lightmap of a half - both exact in a
		// half float - are one light, and the pictures differ only by `a (d + s)`
		// against `a d + a s`, which is a rounding
		let Some(mut capture) = capture() else {
			return;
		};

		for count in ["1", "4"] {
			let (mut world, floor) = floor_world(0.5, Vec3::new(0.8, 0.6, 0.4));

			samples(&mut world, count);
			named(&mut world, even([0.5, 0.5, 0.5]));

			let unbaked = capture
				.shoot(&mut world)
				.expect("the capture renders");

			place(&mut world, floor, [0, 0, 16, 8]);

			let baked_shot = capture
				.shoot(&mut world)
				.expect("the capture renders");
			let (moved, widest) = apart(&unbaked, &baked_shot);

			assert!(widest <= 1, "at {count}: within a level, {moved} pixels up to {widest}");
		}
	}

	/// A square two units on a side facing `+z`, whose second set is its own
	/// corners: nought and nought at the top left as a camera on `+z` sees it.
	fn upright() -> MeshData {
		let corners = [
			([-1.0, 1.0, 0.0], [0.0, 0.0]),
			([1.0, 1.0, 0.0], [1.0, 0.0]),
			([1.0, -1.0, 0.0], [1.0, 1.0]),
			([-1.0, -1.0, 0.0], [0.0, 1.0]),
		];

		MeshData {
			vertices: corners
				.map(|(position, _)| MeshVertex {
					position,
					normal: [0.0, 0.0, 1.0],
					uv: [0.0, 0.0],
					tangent: [1.0, 0.0, 0.0, 1.0],
				})
				.to_vec(),
			indices: vec![0, 3, 2, 0, 2, 1],
			paint: corners
				.map(|(_, uv2)| PaintVertex { uv2, ..PaintVertex::PLAIN })
				.to_vec(),
			..MeshData::default()
		}
	}

	/// Where on the picture the square is: its first and last column and row.
	fn extent(image: &Image) -> [u32; 4] {
		let lit: Vec<(u32, u32)> = (0..image.height)
			.flat_map(|y| (0..image.width).map(move |x| (x, y)))
			.filter(|&(x, y)| {
				image
					.pixel(x, y)
					.iter()
					.take(3)
					.any(|&channel| channel > 40)
			})
			.collect();
		let across = || lit.iter().map(|&(x, _)| x);
		let down = || lit.iter().map(|&(_, y)| y);

		[
			across().min().unwrap_or(0),
			across().max().unwrap_or(0),
			down().min().unwrap_or(0),
			down().max().unwrap_or(0),
		]
	}

	/// Which of four colors, a quarter each, a texel of the upright square's
	/// place holds, and the fifth color everywhere else on the picture.
	fn quartered(column: u32, row: u32) -> [f32; 3] {
		match (column >= 8, column >= 12, row >= 4) {
			| (false, ..) => [1.0, 0.0, 1.0],
			| (true, false, false) => [1.0, 0.0, 0.0],
			| (true, true, false) => [0.0, 1.0, 0.0],
			| (true, false, true) => [0.0, 0.0, 1.0],
			| (true, true, true) => [1.0, 1.0, 1.0],
		}
	}

	#[test]
	fn a_place_is_read_the_right_way_up_and_where_its_record_says() {
		// the right half of a picture two places wide, in four colors a quarter
		// each, and the left half a fifth color no quarter of the square may
		// show: a place read from the wrong half, turned, mirrored or scaled
		// shows the wrong color in some quarter
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();

		plainly(&mut world, 0.0);
		world.light = Vec3::Y;
		world.camera.position = Vec3::new(0.0, 0.0, 3.0);
		world.camera.target = Vec3::ZERO;

		let mesh = world.meshes.insert("test/upright", upright());
		let square = world.entities.spawn_at(Transform::at(Vec3::ZERO));

		world
			.entities
			.set_renderable(square, Renderable::of(mesh, MaterialId::DEFAULT, Vec3::ONE));
		named(&mut world, painted(16, 8, quartered));
		place(&mut world, square, [8, 0, 8, 8]);

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let [left, right, top, bottom] = extent(&image);
		let across = |quarter: u32| left + (right - left) * quarter / 4;
		let down = |quarter: u32| top + (bottom - top) * quarter / 4;
		let quarters = [
			("top left", image.pixel(across(1), down(1)), [true, false, false]),
			("top right", image.pixel(across(3), down(1)), [false, true, false]),
			("bottom left", image.pixel(across(1), down(3)), [false, false, true]),
			("bottom right", image.pixel(across(3), down(3)), [true, true, true]),
		];

		assert!(right - left > 100 && bottom - top > 100, "the square fills the middle");

		for (name, pixel, lit) in quarters {
			let seen = [pixel[0] > 200, pixel[1] > 200, pixel[2] > 200];
			let dark = pixel
				.iter()
				.take(3)
				.zip(lit)
				.all(|(channel, on)| on || *channel < 30);

			assert!(seen == lit && dark, "the {name} quarter reads its own color: {pixel:?}");
		}
	}

	#[test]
	fn the_share_of_the_sky_takes_as_much_of_a_baked_light_as_it_took_of_the_sky() {
		// nothing lights this crease but what a bake kept: no sun, no ambient
		// color, no reflection. So where the share darkens the picture, it
		// darkened the baked light
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();

		plainly(&mut world, 0.0);
		world.light = Vec3::Y;
		world.camera.position = Vec3::new(0.0, 2.5, 1.0);
		world.camera.target = Vec3::new(0.0, 0.0, -2.0);

		let floor =
			slab(&mut world, Vec3::new(0.0, -0.5, 0.0), Vec3::new(40.0, 1.0, 40.0), Vec3::ONE);
		let wall =
			slab(&mut world, Vec3::new(0.0, 5.0, -2.5), Vec3::new(40.0, 10.0, 1.0), Vec3::ONE);

		named(&mut world, even([0.6, 0.6, 0.6]));
		place(&mut world, floor, [0, 0, 16, 8]);
		place(&mut world, wall, [0, 0, 16, 8]);
		samples(&mut world, "1");
		asked(
			&mut world,
			reflection::STRENGTH,
			Value::Float(reflection::DEFAULT_STRENGTH),
			"0",
		);
		asked(&mut world, occlusion::STRENGTH, Value::Float(occlusion::DEFAULT_STRENGTH), "0");

		let open = capture
			.shoot(&mut world)
			.expect("the capture renders");

		world.cvars.set(occlusion::STRENGTH, "1");

		let shared = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let (moved, _) = apart(&open, &shared);
		let lighter = open
			.pixels
			.chunks_exact(4)
			.zip(shared.pixels.chunks_exact(4))
			.filter(|(before, after)| {
				after
					.iter()
					.take(3)
					.zip(before.iter())
					.any(|(a, b)| a > b)
			})
			.count();

		assert!(moved > 1000, "the crease darkens: {moved} pixels");
		assert_eq!(lighter, 0, "and nothing is lighter");
	}

	#[test]
	fn glass_reads_no_lightmap_whatever_its_record_says() {
		// no bake lights glass, and a place is a number anybody may type: a pane
		// given one draws as the pane given none
		let Some(mut capture) = capture() else {
			return;
		};
		let (mut world, _) = floor_world(0.35, Vec3::new(0.5, 0.5, 0.5));
		let glass = world.materials.insert("test/glass", Material {
			blend: Blend::Alpha,
			opacity: 0.5,
			..Material::DEFAULT
		});
		let pane = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, 1.0, 0.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(2.0, 2.0, 0.1),
		});

		world
			.entities
			.set_renderable(pane, Renderable::of(MeshId::CUBE, glass, Vec3::new(0.3, 0.6, 0.9)));
		named(&mut world, even([4.0, 0.0, 0.0]));

		let unplaced = capture
			.shoot(&mut world)
			.expect("the capture renders");

		place(&mut world, pane, [0, 0, 16, 8]);

		let placed = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert!(unplaced.pixels == placed.pixels, "a pane given a place draws as one with none");
		assert_eq!(
			capture.scene_mut().places_handed(),
			0,
			"and takes no entry in the table, nor a batch of its own"
		);
	}

	#[test]
	fn things_of_one_mesh_and_material_are_drawn_apart_by_whether_they_read_the_lightmap() {
		// two slabs side by side, one given a place and one not, in one frame:
		// the baked one reads the lightmap through the baked pipeline and the
		// other is drawn by the plain one exactly as when neither has a place.
		// One batch for both would draw both one way or the other
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();

		plainly(&mut world, 0.35);
		world.light = Vec3::new(0.3, -1.0, -0.4);
		world.camera.position = Vec3::new(0.0, 4.0, 5.0);
		world.camera.target = Vec3::ZERO;

		let tint = Vec3::new(0.8, 0.6, 0.4);
		let left = slab(&mut world, Vec3::new(-2.6, -0.5, 0.0), Vec3::new(5.0, 1.0, 8.0), tint);
		let right = slab(&mut world, Vec3::new(2.6, -0.5, 0.0), Vec3::new(5.0, 1.0, 8.0), tint);

		named(&mut world, even([0.9, 0.2, 0.1]));

		let neither = capture
			.shoot(&mut world)
			.expect("the capture renders");

		place(&mut world, left, [0, 0, 16, 8]);

		let one = capture
			.shoot(&mut world)
			.expect("the capture renders");

		place(&mut world, right, [0, 0, 16, 8]);

		let both = capture
			.shoot(&mut world)
			.expect("the capture renders");
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

		assert!(
			half(&one, true) == half(&neither, true),
			"the slab with no place draws as before"
		);
		assert!(
			half(&one, false) == half(&both, false),
			"the one with a place draws as when both have"
		);
		assert!(half(&one, false) != half(&neither, false), "and that is another picture");
	}

	/// Two texels by two, solid on one diagonal and holes on the other.
	fn holed() -> TextureData {
		let solid = [0xE0, 0xE0, 0xE0, 0xFF];
		let hole = [0xE0, 0xE0, 0xE0, 0x11];

		TextureData {
			width: 2,
			height: 2,
			faces: 1,
			texel: Texel::Rgba8Srgb,
			levels: vec![[solid, hole, hole, solid].concat()],
		}
	}

	#[test]
	fn a_baked_cutout_keeps_its_holes_and_reads_its_light_on_the_rest() {
		// the baked pipelines have a masked entry point of their own, and the
		// plain masked one must stay plain: a cutout with no place in a world
		// that names a lightmap draws as in a world that names none, and one with
		// a place is lit by it where it is solid and leaves its holes as holes
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();

		plainly(&mut world, 0.35);
		world.light = Vec3::Y;
		world.camera.position = Vec3::new(0.0, 5.0, 0.01);
		world.camera.target = Vec3::ZERO;

		let picture = world.textures.insert("test/holed", holed());
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

		let bare = capture
			.shoot(&mut world)
			.expect("the capture renders");

		named(&mut world, even([4.0, 0.0, 0.0]));

		let unplaced = capture
			.shoot(&mut world)
			.expect("the capture renders");

		place(&mut world, quad, [0, 0, 16, 8]);

		let placed = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let pairs = || {
			bare.pixels
				.chunks_exact(4)
				.zip(placed.pixels.chunks_exact(4))
		};
		let holes = pairs()
			.filter(|(was, _)| was[..3] == [0, 0, 0])
			.count();
		let kept = pairs()
			.filter(|(was, now)| was[..3] == [0, 0, 0] && was == now)
			.count();
		let lit = pairs()
			.filter(|(was, now)| was[..3] != [0, 0, 0] && now[0] > was[0])
			.count();

		assert!(bare.pixels == unplaced.pixels, "a cutout with no place reads no lightmap");
		assert!(holes > 1000 && kept == holes, "every hole stays a hole: {kept} of {holes}");
		assert!(lit > 1000, "and the solid half reads the red it was baked: {lit} pixels");
	}

	#[test]
	fn a_cube_named_as_the_lightmap_is_not_read() {
		// six faces are not a picture a second set can land on: a world naming
		// one draws as a world naming none
		let Some(mut capture) = capture() else {
			return;
		};
		let (mut world, floor) = floor_world(0.35, Vec3::new(0.8, 0.6, 0.4));

		place(&mut world, floor, [0, 0, 16, 8]);
		world.lightmap = world
			.textures
			.insert("lightmaps/test", sky_cube());

		let cube = capture
			.shoot(&mut world)
			.expect("the capture renders");

		world.lightmap = TextureId::NONE;

		let none = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert!(cube.pixels == none.pixels, "a cube is not read as a lightmap");
	}

	#[test]
	fn a_bake_again_is_the_new_picture_read() {
		// a bake writes the same name again, which is a new revision of the same
		// slot: the group is made again and the new light is what is read
		let Some(mut capture) = capture() else {
			return;
		};
		let (mut world, floor) = floor_world(0.35, Vec3::ONE);

		// no sun, which on a white floor is past white whatever else arrives
		world.light = Vec3::Y;
		named(&mut world, even([0.2, 0.2, 0.2]));
		place(&mut world, floor, [0, 0, 16, 8]);

		let first = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let rebound = capture.scene_mut().rebinds();

		named(&mut world, even([0.8, 0.8, 0.8]));

		let second = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert!(capture.scene_mut().rebinds() > rebound, "the group is made again");
		assert!(apart(&first, &second).0 > 10_000, "and the floor reads the new light");
	}
}
