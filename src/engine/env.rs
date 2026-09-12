//! The environment: what a surface reflects when it is not reflecting a lamp.
//!
//! One cube texture and one sampler, uploaded from whatever the world's sky
//! names and bound beside the shadow atlas. Small on purpose - there is no pass
//! here and no pipeline, because the expensive half of an environment map is
//! the filtering and that happens in the asset compiler, offline. @ref
//! `colby_asset::cube`.
//!
//! **It shares the atlas's bind group, and that is a constraint rather than a
//! choice.** The scene binds four groups - the frame, the material, the shadow
//! atlas and the bones - and four is the floor every device is required to
//! offer, so there is no fifth to put this in. The atlas's group is the one it
//! joins, because a map and an environment are both things the whole frame
//! reads and neither changes per draw.
//!
//! **The off state is a white cube of one texel a face**, registered nowhere
//! and built here, because a binding cannot be left empty. Nothing reads it:
//! the shader branches on a word in the frame's own uniform and takes the line
//! it took before there were environments at all. That branch, rather than a
//! multiply by one, is what makes a world with no cubemap draw the same bytes
//! it drew before this file existed - `a * (b + c)` and `a * b + a * c` are not
//! the same float, and the negative control is checked to the byte.

use colby_core::{
	Result,
	abi::{TextureData, World, texture::CUBE_FACES},
	err,
};
use wgpu::{
	AddressMode, BindGroupLayoutEntry, BindingType, Device, Extent3d, FilterMode,
	MipmapFilterMode, Origin3d, Queue, Sampler, SamplerBindingType, SamplerDescriptor,
	ShaderStages, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect,
	TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
	TextureView, TextureViewDescriptor, TextureViewDimension,
};

/// Whether a world's environment is read at all.
///
/// A switch rather than a ceiling, because there is nothing to count: one cube
/// is one cube. What it is for is the same thing every other renderer switch
/// here is for - measuring what the feature costs, and taking the picture a
/// build from before it would have taken.
pub const ENABLED: &str = "r.environment";

/// The format an environment is held in on the device.
///
/// The same one the renderer's own target is in, and for the same reason: a sky
/// holds a sun, and a sun does not fit between nought and one. @ref
/// `colby_core::abi::Texel::Rgba16Float`.
const FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// The cube the scene samples, and the sampler it reads through.
pub(crate) struct Environment {
	/// The cube itself, held so that nothing else has to.
	///
	/// A view keeps its texture alive on its own, and holding the texture
	/// anyway is the difference between relying on that and not having to.
	cube: Texture,

	/// The view the bind group holds.
	view: TextureView,

	/// Trilinear and clamped, which is what walks the roughness chain.
	sampler: Sampler,

	/// How many levels the bound cube has, or nought for the white one.
	///
	/// This is the number the shader branches on, so it is also the answer to
	/// "is there an environment at all". A world whose sky names nothing, whose
	/// name nothing answers to, or whose switch is off all arrive here as
	/// nought.
	levels: u32,

	/// Which texture this was built from, so that a frame that names the same
	/// one does not rebuild the group.
	source: Option<(u32, u32)>,
}

impl Environment {
	/// The white cube, which is what a world with no sky gets.
	///
	/// @param device - the device to build against
	/// @param queue - the queue the texels are written through
	pub(crate) fn new(device: &Device, queue: &Queue) -> Result<Self> {
		let blank = TextureData::white_cube();
		let (cube, view) = upload(device, queue, &blank)
			.ok_or_else(|| err!(Graphics("the blank environment could not be built")))?;

		Ok(Self {
			cube,
			view,
			sampler: sampler(device),
			levels: 0,
			source: None,
		})
	}

	/// The view a bind group binds.
	pub(crate) const fn view(&self) -> &TextureView { &self.view }

	/// The sampler it reads through.
	pub(crate) const fn sampler(&self) -> &Sampler { &self.sampler }

	/// How many roughness levels the bound cube has, nought for none.
	pub(crate) const fn levels(&self) -> u32 { self.levels }

	/// Puts the world's environment on the device, if it changed.
	///
	/// @param device - the device to build against
	/// @param queue - the queue the texels are written through
	/// @param world - whose sky and texture registry are read
	/// @param wanted - whether the switch lets an environment be read at all
	/// @return whether the bound cube changed, which is what makes the group
	/// stale
	pub(crate) fn update(
		&mut self,
		device: &Device,
		queue: &Queue,
		world: &World,
		wanted: bool,
	) -> bool {
		let asked = if wanted && world.sky.lights() {
			world
				.textures
				.get(world.sky.cubemap)
				.map(|entry| (world.sky.cubemap.index(), entry))
		} else {
			None
		};

		let Some((slot, entry)) = asked else {
			return self.blank(device, queue);
		};

		// a cube and nothing else: a flat picture bound as an environment would
		// be six views of the same rectangle, which is a bug shaped like a
		// feature. A world naming one gets the white cube and the branch that
		// skips it.
		let data = entry.value();
		if !data.is_cube() || !data.is_consistent() {
			return self.blank(device, queue);
		}

		let named = Some((slot, entry.revision()));
		if named == self.source {
			return false;
		}

		let Some((cube, view)) = upload(device, queue, data) else {
			return self.blank(device, queue);
		};

		self.cube = cube;
		self.view = view;
		self.levels = u32::try_from(data.levels.len()).unwrap_or(1);
		self.source = named;

		true
	}

	/// Back to the white cube, if it is not already bound.
	fn blank(&mut self, device: &Device, queue: &Queue) -> bool {
		if self.source.is_none() {
			return false;
		}

		if let Some((cube, view)) = upload(device, queue, &TextureData::white_cube()) {
			self.cube = cube;
			self.view = view;
		}

		self.levels = 0;
		self.source = None;

		true
	}
}

/// Trilinear, clamped, and filtering between levels as well as inside them.
///
/// The level is what carries the roughness, so a surface whose roughness falls
/// between two of them has to read both: without that a ball with a smooth
/// gradient of roughness over it shows bands where the level steps.
fn sampler(device: &Device) -> Sampler {
	device.create_sampler(&SamplerDescriptor {
		label: Some("environment"),
		address_mode_u: AddressMode::ClampToEdge,
		address_mode_v: AddressMode::ClampToEdge,
		address_mode_w: AddressMode::ClampToEdge,
		mag_filter: FilterMode::Linear,
		min_filter: FilterMode::Linear,
		mipmap_filter: MipmapFilterMode::Linear,
		..SamplerDescriptor::default()
	})
}

/// Writes a cube's levels onto a new texture and hands back a cube view.
///
/// Six array layers with a cube view over them, which is how the API spells a
/// cube: the layers are the faces, in the order the file wrote them.
///
/// @param device - the device to build against
/// @param queue - the queue the texels are written through
/// @param data - the cube, already checked
/// @return the texture and a cube view of it, or nothing if the data is not a
/// cube at all
fn upload(device: &Device, queue: &Queue, data: &TextureData) -> Option<(Texture, TextureView)> {
	if !data.is_cube() || data.width == 0 || data.levels.is_empty() {
		return None;
	}

	let levels = u32::try_from(data.levels.len()).ok()?;
	let texture = device.create_texture(&TextureDescriptor {
		label: Some("environment"),
		size: Extent3d {
			width: data.width,
			height: data.height,
			depth_or_array_layers: CUBE_FACES,
		},
		mip_level_count: levels,
		sample_count: 1,
		dimension: TextureDimension::D2,
		format: FORMAT,
		usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
		view_formats: &[],
	});

	for level in 0..levels {
		write_level(queue, &texture, data, level);
	}

	let view = texture.create_view(&TextureViewDescriptor {
		label: Some("environment"),
		dimension: Some(TextureViewDimension::Cube),
		..TextureViewDescriptor::default()
	});

	Some((texture, view))
}

/// One level's six faces, in one write.
fn write_level(queue: &Queue, texture: &Texture, data: &TextureData, level: u32) {
	let Some(bytes) = usize::try_from(level)
		.ok()
		.and_then(|index| data.levels.get(index))
	else {
		return;
	};

	if bytes.len() != data.level_bytes(level) {
		return;
	}

	let (width, height) = data.level_size(level);
	let row = width * u32::try_from(data.texel.bytes()).unwrap_or(8);

	queue.write_texture(
		TexelCopyTextureInfo {
			texture,
			mip_level: level,
			origin: Origin3d::ZERO,
			aspect: TextureAspect::All,
		},
		bytes,
		TexelCopyBufferLayout {
			offset: 0,
			bytes_per_row: Some(row),
			// what makes one write do all six: the faces lie back to back in
			// the level, so the layer stride is exactly one face's bytes
			rows_per_image: Some(height),
		},
		Extent3d {
			width,
			height,
			depth_or_array_layers: CUBE_FACES,
		},
	);
}

/// The two entries an environment adds to the group it shares with the atlas.
///
/// @param texture - which binding the cube takes
/// @param sampler - which binding its sampler takes
pub(crate) fn layout_entries(texture: u32, sampler: u32) -> [BindGroupLayoutEntry; 2] {
	[
		BindGroupLayoutEntry {
			binding: texture,
			visibility: ShaderStages::FRAGMENT,
			// filterable, unlike the atlas beside it: a shadow map is compared
			// and an environment is read, so this one blends colors and that
			// one blends the answers to a comparison
			ty: BindingType::Texture {
				sample_type: TextureSampleType::Float { filterable: true },
				view_dimension: TextureViewDimension::Cube,
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
	]
}

#[cfg(test)]
mod tests {
	use colby_core::abi::{Sky, Texel, TextureId};

	use super::*;
	use crate::gpu::Gpu;

	/// The one device every test on it shares, or `None` on a machine with no
	/// GPU.
	///
	/// The shared one rather than a stub of its own, which is what the rest of
	/// this crate's tests use: opening a device per test is six more devices
	/// held at once, and the tests here upload real texels rather than only
	/// checking that a pipeline builds.
	fn device() -> Option<&'static Gpu> { crate::gpu::shared() }

	/// A cube of a side with a full chain, every texel the same two bytes.
	fn cube(side: u32) -> TextureData {
		let count = TextureData::full_chain(side, side);
		let levels = (0..count)
			.map(|level| {
				let across = usize::try_from((side >> level.min(31)).max(1)).unwrap_or(1);

				vec![0x3C; across * across * 6 * 8]
			})
			.collect();

		TextureData {
			width: side,
			height: side,
			faces: CUBE_FACES,
			texel: Texel::Rgba16Float,
			levels,
		}
	}

	#[test]
	fn a_fresh_environment_is_the_white_cube_and_says_it_has_no_levels() {
		let Some(gpu) = device() else {
			return;
		};
		let environment = Environment::new(gpu.device(), gpu.queue()).expect("it builds");

		assert_eq!(
			environment.levels(),
			0,
			"which is the word the shader branches on, so nought means the old line"
		);
	}

	#[test]
	fn a_world_naming_a_cube_binds_it_and_naming_it_again_does_not_rebuild() {
		let Some(gpu) = device() else {
			return;
		};
		let mut world = World::new();
		let id = world.textures.insert("skies/dusk", cube(8));
		world.sky = Sky::environment(id);

		let mut environment = Environment::new(gpu.device(), gpu.queue()).expect("it builds");

		assert!(
			environment.update(gpu.device(), gpu.queue(), &world, true),
			"the first frame binds it"
		);
		assert_eq!(environment.levels(), 4, "eight down to one is four levels");
		assert!(
			!environment.update(gpu.device(), gpu.queue(), &world, true),
			"and the second frame does not, which is what keeps the group alive"
		);
	}

	#[test]
	fn the_switch_and_a_world_with_no_sky_arrive_at_the_same_place() {
		let Some(gpu) = device() else {
			return;
		};
		let mut world = World::new();
		let id = world.textures.insert("skies/dusk", cube(4));
		world.sky = Sky::environment(id);

		let mut environment = Environment::new(gpu.device(), gpu.queue()).expect("it builds");
		environment.update(gpu.device(), gpu.queue(), &world, true);

		assert!(
			environment.update(gpu.device(), gpu.queue(), &world, false),
			"turning the switch off changes what is bound"
		);
		assert_eq!(environment.levels(), 0, "back to the white cube");

		environment.update(gpu.device(), gpu.queue(), &world, true);
		world.sky = Sky::NONE;

		assert!(
			environment.update(gpu.device(), gpu.queue(), &world, true),
			"and so does taking the sky away"
		);
		assert_eq!(environment.levels(), 0, "to the same place");
	}

	#[test]
	fn a_flat_picture_named_as_an_environment_is_refused_rather_than_bound() {
		let Some(gpu) = device() else {
			return;
		};
		let mut world = World::new();
		let id = world
			.textures
			.insert("textures/wall", TextureData::white());
		world.sky = Sky::environment(id);

		let mut environment = Environment::new(gpu.device(), gpu.queue()).expect("it builds");
		environment.update(gpu.device(), gpu.queue(), &world, true);

		assert_eq!(
			environment.levels(),
			0,
			"six views of one rectangle is a bug shaped like a feature, so it is not bound"
		);
	}

	#[test]
	fn a_sky_naming_a_texture_nothing_answers_to_lights_out_of_the_ambient() {
		let Some(gpu) = device() else {
			return;
		};
		let mut world = World::new();
		world.sky = Sky::environment(TextureId::new(77));

		let mut environment = Environment::new(gpu.device(), gpu.queue()).expect("it builds");
		environment.update(gpu.device(), gpu.queue(), &world, true);

		assert_eq!(environment.levels(), 0, "a handle nothing answers to binds nothing");
	}

	#[test]
	fn reloading_the_same_texture_rebinds_it() {
		let Some(gpu) = device() else {
			return;
		};
		let mut world = World::new();
		let id = world.textures.insert("skies/dusk", cube(4));
		world.sky = Sky::environment(id);

		let mut environment = Environment::new(gpu.device(), gpu.queue()).expect("it builds");
		environment.update(gpu.device(), gpu.queue(), &world, true);

		// the same handle, different pixels - which is what a hot reload of a
		// sky is, and what a slot number alone would miss
		world.textures.insert("skies/dusk", cube(8));

		assert!(
			environment.update(gpu.device(), gpu.queue(), &world, true),
			"the revision moved, so the cube is uploaded again"
		);
		assert_eq!(environment.levels(), 4, "at the new size");
	}
}
