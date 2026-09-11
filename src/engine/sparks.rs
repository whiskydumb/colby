//! The GPU half of the particle renderer: one instance buffer, two pipelines.
//!
//! Part of [`Scene`](crate::Scene) rather than an
//! [`Overlay`](crate::Overlay), for [`Lines`](crate::lines::Lines)'s reason:
//! a particle is world-space geometry seen through the scene's camera, and
//! whether it is hidden by a wall is not optional. So it draws inside the
//! scene's own pass, with the scene's depth attachment and the scene's
//! globals - which also means the camera cannot disagree between the two.
//!
//! **A particle is an instance, not four vertices.** Fyrox writes four
//! vertices per particle into a dynamic buffer and Wicked bakes them with a
//! compute shader; both are answers to problems this renderer does not have -
//! Fyrox's is a GL-era vertex layout and Wicked's is that it wants a
//! ray-tracing acceleration structure over its particles. Here the four
//! corners come from `vertex_index` and the particle from `instance_index`, so
//! a cloud costs thirty-two bytes each and one draw call per picture.
//!
//! **Two pipelines, and neither of them is [`Blend`].** A surface's blend mode
//! is a question about how a material's alpha is read, and the scene has six
//! pipelines for the answers; this is a different question - whether the cloud
//! lightens what is behind it or covers it - and the two answers are
//! [`SparkBlend`]'s. Neither writes depth, both test it, and nothing here
//! casts a shadow, which is what every engine read does with geometry that
//! writes no depth.
//!
//! [`Blend`]: colby_core::abi::material::Blend

use colby_core::{
	Result,
	abi::{EntityId, SparkBlend, TextureId, World},
	bytemuck::{self, Pod, Zeroable},
	err,
	glam::Vec3,
	warn,
};
use wgpu::{
	BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
	BindGroupLayoutEntry, BindingResource, BindingType, BlendComponent, BlendFactor,
	BlendOperation, BlendState, Buffer, BufferAddress, BufferDescriptor, BufferUsages,
	ColorTargetState, ColorWrites, CompareFunction, DepthBiasState, DepthStencilState, Device,
	ErrorFilter, Face, FragmentState, FrontFace, MultisampleState, PipelineCompilationOptions,
	PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, Queue, RenderPass,
	RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerBindingType,
	ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState, TextureFormat,
	TextureSampleType, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState,
	VertexStepMode,
};

use crate::scene::{DEPTH_FORMAT, GpuTexture};

/// How many particles a fresh buffer has room for.
///
/// It grows from here and never shrinks, like the debug renderer's and the
/// interface's.
const INITIAL_INSTANCES: u64 = 512;

/// One particle, as the vertex stage reads it.
///
/// Thirty-two bytes, and everything in it has already been worked out on the
/// CPU: the size is interpolated across the particle's life, the color is the
/// emitter's two colors mixed, and the alpha is the fade times the emitter's
/// own opacity. The shader does the billboard and the texture and nothing
/// else, which is what keeps this one pipeline rather than a family of them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub(crate) struct SparkInstance {
	position: [f32; 3],
	size: f32,
	color: [f32; 4],
}

/// A run of instances that share a picture and a way of reaching the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
	blend: SparkBlend,
	texture: usize,
	first: u32,
	count: u32,
}

/// Every live particle this frame, and what it takes to draw them.
pub(crate) struct Sparks {
	/// One per [`SparkBlend`], in [`SparkBlend::row`] order, **built the first
	/// frame that has a particle in it and not before**.
	///
	/// Lazily, because almost no world throws anything and two pipelines is
	/// two shader compiles on every device that ever draws at all - which in
	/// this workspace means each of the renderer's pixel tests, every one of
	/// which opens a device of its own. A world with no emitter in it now
	/// costs exactly a layout, a sampler and sixteen kilobytes of buffer for
	/// a feature it does not use.
	///
	/// @note: this is *not* what fixed the intermittent driver crash in the
	/// engine's own test suite. That was measured against a worktree at the
	/// commit before this one and crashes there at the same rate; it is
	/// older than particles. The lazy build is worth having on its own.
	pipelines: Option<[RenderPipeline; 2]>,

	/// What the pipelines above have to be built for, when they are.
	target: TextureFormat,

	/// And how many samples a pixel of it has.
	samples: u32,

	/// The layout one picture's group is built against.
	layout: BindGroupLayout,

	/// How a particle's picture is sampled.
	///
	/// Its own rather than one of the scene's two, and clamped rather than
	/// repeating: a particle's texture is a sprite on a square, and a
	/// repeating sampler bleeds the far edge of it into the near one wherever
	/// the interpolation reaches past the middle of the outermost texel.
	sampler: Sampler,

	/// One bind group per picture that has been drawn, by registry slot.
	///
	/// Grown on demand and rebuilt when the texture under it is re-uploaded,
	/// which is the same staleness rule every other uploaded thing here
	/// follows: the group holds a view of one particular texture, and
	/// re-uploading an image makes a new one this group knows nothing about.
	groups: Vec<Option<(BindGroup, u32)>>,

	instances: Buffer,
	capacity: u64,

	/// This frame's cloud, laid out: what is written to the GPU and what it is
	/// worked out in. @ref [`Cloud`].
	cloud: Cloud,
}

impl Sparks {
	/// The buffer, the sampler and the picture layout - and no pipeline.
	///
	/// **Nothing here compiles a shader**, which is the point: @ref
	/// [`pipelines`](Self::pipelines) for what that cost and how it was found.
	/// The pair arrives on the first frame that has a particle in it.
	///
	/// @param device - the device to build against
	/// @param format - the color format the fragment stage will write
	/// @param samples - how many the target has
	pub(crate) fn new(device: &Device, format: TextureFormat, samples: u32) -> Self {
		let layout = picture_layout(device);

		Self {
			pipelines: None,
			target: format,
			samples,
			layout,
			sampler: sampler(device),
			groups: Vec::new(),
			instances: buffer(device, INITIAL_INSTANCES),
			capacity: INITIAL_INSTANCES,
			cloud: Cloud::default(),
		}
	}

	/// Notes that the target has a new sample count.
	///
	/// **Throws the pipelines away rather than rebuilding them**, which is
	/// where this differs from
	/// [`Lines::set_samples`](crate::lines::Lines::set_samples): the pair is
	/// built on demand anyway, so the next frame with a particle in it builds
	/// the right one and a frame without one builds nothing at all. The
	/// buffer and the picture groups are kept - those are what a cloud costs
	/// to *allocate*, and a person turning anti-aliasing on should not pay for
	/// them again.
	///
	/// @param format - the color format the fragment stage writes
	/// @param samples - how many the target has
	pub(crate) fn set_samples(&mut self, format: TextureFormat, samples: u32) {
		self.target = format;
		self.samples = samples;
		self.pipelines = None;
	}

	/// Builds the two pipelines, if this is the first frame that needs them.
	///
	/// A complaint is said once and the frame draws no particles, which is the
	/// same answer the scene's own table takes when a sample count will not
	/// build: a picture missing its smoke beats no picture.
	///
	/// @param device - the device to build against
	/// @param globals - the layout of group nought
	fn ensure(&mut self, device: &Device, globals: &BindGroupLayout) {
		if self.pipelines.is_some() {
			return;
		}

		match pipelines(device, self.target, globals, &self.layout, self.samples) {
			| Ok(built) => self.pipelines = Some(built),
			| Err(complaint) => {
				warn!(%complaint, "no particles will be drawn");
			},
		}
	}

	/// Lays this frame's cloud out and writes it to the GPU.
	///
	/// @param device - the device the buffer belongs to
	/// @param queue - where to write
	/// @param world - whose `sparks` pool and emitters are read
	/// @param textures - the scene's uploaded textures, for the picture groups
	pub(crate) fn upload(
		&mut self,
		device: &Device,
		queue: &Queue,
		world: &World,
		textures: &[GpuTexture],
		globals: &BindGroupLayout,
	) {
		self.cloud.lay_out(world, textures.len());

		if self.cloud.sorted.is_empty() {
			return;
		}

		// the first cloud this device has seen is what pays for the pipelines,
		// and a device that never sees one never builds them.
		self.ensure(device, globals);

		let wanted = u64::try_from(self.cloud.sorted.len()).unwrap_or(0);
		if wanted > self.capacity {
			self.capacity = wanted.next_power_of_two();
			self.instances = buffer(device, self.capacity);
		}

		queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&self.cloud.sorted));

		// collected first, because binding takes `&mut self` and the runs are
		// on it. A handful of pictures, so the allocation is nothing and the
		// alternative is an index loop over a length that cannot change.
		let pictures: Vec<usize> = self
			.cloud
			.runs
			.iter()
			.map(|run| run.texture)
			.collect();

		for picture in pictures {
			self.bind(device, textures, picture);
		}
	}

	/// Makes sure one picture's bind group is there and is not stale.
	fn bind(&mut self, device: &Device, textures: &[GpuTexture], slot: usize) {
		let Some(texture) = textures.get(slot) else {
			return;
		};

		if self.groups.len() <= slot {
			self.groups
				.resize_with(slot.saturating_add(1), || None);
		}

		if let Some(Some((_, revision))) = self.groups.get(slot)
			&& *revision == texture.revision
		{
			return;
		}

		let group = device.create_bind_group(&BindGroupDescriptor {
			label: Some("particle picture"),
			layout: &self.layout,
			entries: &[
				BindGroupEntry {
					binding: 0,
					resource: BindingResource::TextureView(&texture.view),
				},
				BindGroupEntry {
					binding: 1,
					resource: BindingResource::Sampler(&self.sampler),
				},
			],
		});

		if let Some(held) = self.groups.get_mut(slot) {
			*held = Some((group, texture.revision));
		}
	}

	/// Records every run into a pass the scene has already drawn into.
	///
	/// Group 0 is set again rather than inherited, for the reason the debug
	/// renderer sets it again: a draw that depends on the caller's last
	/// statement is a draw that breaks when the caller is reordered.
	///
	/// @param pass - the scene's pass, with its depth attachment
	/// @param globals - the camera and the light
	pub(crate) fn draw(&self, pass: &mut RenderPass<'_>, globals: &BindGroup) {
		if self.cloud.runs.is_empty() {
			return;
		}

		pass.set_bind_group(0, globals, &[]);
		pass.set_vertex_buffer(0, self.instances.slice(..));

		let Some(pipelines) = self.pipelines.as_ref() else {
			return;
		};
		let mut bound = None;

		for run in &self.cloud.runs {
			let Some(Some((group, _))) = self.groups.get(run.texture) else {
				continue;
			};

			if bound != Some(run.blend) {
				pass.set_pipeline(&pipelines[run.blend.row()]);
				bound = Some(run.blend);
			}

			pass.set_bind_group(1, group, &[]);
			// four corners of a strip, one instance per particle
			pass.draw(0..4, run.first..run.first.saturating_add(run.count));
		}
	}

	/// How many particles this frame drew, for a report.
	pub(crate) fn drawn(&self) -> usize { self.cloud.sorted.len() }

	/// Whether the two pipelines are there.
	///
	/// `false` before the first cloud this device has seen, and `false` again
	/// after one whose shader would not compile - [`ensure`](Self::ensure)
	/// warns and draws nothing rather than stopping a running window over a
	/// shader somebody is editing, so the count above cannot tell the two
	/// apart: it is filled before the pipelines are asked for. @ref
	/// `crate::headless`, which is the only thing that asks, and which without
	/// this would pass over a `sparks.wgsl` that does not compile at all.
	///
	/// **Under `cfg(test)`, because that is the whole truth about it**: a
	/// running window has no use for the answer, and the alternative was one
	/// more line of shipped surface that only a test reads.
	#[cfg(test)]
	pub(crate) const fn built(&self) -> bool { self.pipelines.is_some() }
}

/// One number some of the way to another.
fn mix(from: f32, to: f32, through: f32) -> f32 { (to - from).mul_add(through, from) }

/// Three of them.
fn mix3(from: Vec3, to: Vec3, through: f32) -> Vec3 { from + (to - from) * through }

/// An instance buffer with room for this many particles.
fn buffer(device: &Device, instances: u64) -> Buffer {
	device.create_buffer(&BufferDescriptor {
		label: Some("particles"),
		size: instances.max(1) * stride(),
		usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
		mapped_at_creation: false,
	})
}

/// How a particle's picture is sampled.
fn sampler(device: &Device) -> Sampler {
	device.create_sampler(&wgpu::SamplerDescriptor {
		label: Some("particle picture"),
		address_mode_u: wgpu::AddressMode::ClampToEdge,
		address_mode_v: wgpu::AddressMode::ClampToEdge,
		address_mode_w: wgpu::AddressMode::ClampToEdge,
		mag_filter: wgpu::FilterMode::Linear,
		min_filter: wgpu::FilterMode::Linear,
		mipmap_filter: wgpu::MipmapFilterMode::Linear,
		..wgpu::SamplerDescriptor::default()
	})
}

/// The layout every picture's group is built against.
fn picture_layout(device: &Device) -> BindGroupLayout {
	device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("particle picture"),
		entries: &[
			BindGroupLayoutEntry {
				binding: 0,
				visibility: ShaderStages::FRAGMENT,
				ty: BindingType::Texture {
					sample_type: TextureSampleType::Float { filterable: true },
					view_dimension: wgpu::TextureViewDimension::D2,
					multisampled: false,
				},
				count: None,
			},
			BindGroupLayoutEntry {
				binding: 1,
				visibility: ShaderStages::FRAGMENT,
				ty: BindingType::Sampler(SamplerBindingType::Filtering),
				count: None,
			},
		],
	})
}

/// Both pipelines, or the first complaint wgpu had.
///
/// One error scope over both, because a pair half built is the thing this
/// module has no way to draw with. @ref [`Sparks::set_samples`].
fn pipelines(
	device: &Device,
	format: TextureFormat,
	globals: &BindGroupLayout,
	picture: &BindGroupLayout,
	samples: u32,
) -> Result<[RenderPipeline; 2]> {
	let scope = device.push_error_scope(ErrorFilter::Validation);
	let built = [
		build_pipeline(device, format, globals, picture, SparkBlend::Additive, samples),
		build_pipeline(device, format, globals, picture, SparkBlend::Alpha, samples),
	];

	if let Some(complaint) = pollster::block_on(scope.pop()) {
		return Err(err!(Graphics("the particle pipeline: {complaint}")));
	}

	Ok(built)
}

fn build_pipeline(
	device: &Device,
	format: TextureFormat,
	globals: &BindGroupLayout,
	picture: &BindGroupLayout,
	blend: SparkBlend,
	samples: u32,
) -> RenderPipeline {
	let shader = device.create_shader_module(ShaderModuleDescriptor {
		label: Some("particles"),
		source: ShaderSource::Wgsl(include_str!("sparks.wgsl").into()),
	});

	let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some("particles"),
		bind_group_layouts: &[Some(globals), Some(picture)],
		immediate_size: 0,
	});

	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some(match blend {
			| SparkBlend::Additive => "particles additive",
			| SparkBlend::Alpha => "particles alpha",
		}),
		layout: Some(&layout),
		vertex: VertexState {
			module: &shader,
			entry_point: Some("vertex_main"),
			compilation_options: PipelineCompilationOptions::default(),
			buffers: &[Some(VertexBufferLayout {
				array_stride: stride(),
				step_mode: VertexStepMode::Instance,
				attributes: &INSTANCE_ATTRIBUTES,
			})],
		},
		primitive: PrimitiveState {
			topology: PrimitiveTopology::TriangleStrip,
			strip_index_format: None,
			front_face: FrontFace::Ccw,
			// a square that turns to face the camera has no back to cull, and
			// working out which way round its corners came out under an
			// arbitrary camera roll is work for no picture. Fyrox's own
			// particle pass sets `cull_face: None` and says as much.
			cull_mode: None::<Face>,
			unclipped_depth: false,
			polygon_mode: PolygonMode::Fill,
			conservative: false,
		},
		depth_stencil: Some(DepthStencilState {
			format: DEPTH_FORMAT,
			// **tested and not written**, which is the whole of what makes a
			// cloud composite rather than punch holes in itself. It is the
			// same pair `Blend::Alpha` takes and it is why nothing here casts
			// a shadow either.
			depth_write_enabled: Some(false),
			depth_compare: Some(CompareFunction::Less),
			stencil: StencilState::default(),
			bias: DepthBiasState::default(),
		}),
		multisample: MultisampleState {
			count: samples,
			..MultisampleState::default()
		},
		fragment: Some(FragmentState {
			module: &shader,
			entry_point: Some("fragment_main"),
			compilation_options: PipelineCompilationOptions::default(),
			targets: &[Some(ColorTargetState {
				format,
				blend: Some(match blend {
					// what is behind it is kept whole and the particle is
					// added on top of it, scaled by its own alpha. Order
					// cannot matter, because addition does not care.
					| SparkBlend::Additive => BlendState {
						color: BlendComponent {
							src_factor: BlendFactor::SrcAlpha,
							dst_factor: BlendFactor::One,
							operation: BlendOperation::Add,
						},
						// the target's alpha is left alone: this renderer
						// composites onto an opaque float target and a cloud
						// that ate its alpha would be a cloud that punched a
						// hole in a picture nothing reads the alpha of.
						alpha: BlendComponent {
							src_factor: BlendFactor::Zero,
							dst_factor: BlendFactor::One,
							operation: BlendOperation::Add,
						},
					},
					| SparkBlend::Alpha => BlendState::ALPHA_BLENDING,
				}),
				write_mask: ColorWrites::ALL,
			})],
		}),
		multiview_mask: None,
		cache: None,
	})
}

/// What one [`SparkInstance`] hands the vertex stage.
const INSTANCE_ATTRIBUTES: [VertexAttribute; 3] = [
	VertexAttribute {
		format: VertexFormat::Float32x3,
		offset: 0,
		shader_location: 0,
	},
	VertexAttribute {
		format: VertexFormat::Float32,
		offset: 12,
		shader_location: 1,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 16,
		shader_location: 2,
	},
];

/// A frame's cloud, laid out as instances sorted into runs, with no device in
/// it.
///
/// The half of [`Sparks::upload`] that has arithmetic in it, apart from the
/// wgpu calls so that a test runs this very loop rather than a copy of it.
/// Every list is kept between frames, which is what lets a frame with a cloud
/// in it allocate nothing but the order it sorts.
#[derive(Default)]
struct Cloud {
	/// The instances this frame, sorted into runs.
	sorted: Vec<SparkInstance>,

	/// Which run of `sorted` is drawn with which pipeline and which picture.
	runs: Vec<Run>,

	/// The instances in the order the pool holds them, before the sort.
	unsorted: Vec<SparkInstance>,

	/// The sort key beside each unsorted instance.
	keys: Vec<(u32, u32)>,

	/// Whether each slot's entity is drawn this frame: nought for not asked
	/// yet, one for shown and two for hidden. @ref [`showing`].
	shown: Vec<u8>,
}

impl Cloud {
	/// Lays a world's cloud out.
	///
	/// @param world - whose pool and emitters are read
	/// @param textures - how many pictures the scene has uploaded
	fn lay_out(&mut self, world: &World, textures: usize) {
		self.sorted.clear();
		self.runs.clear();
		self.unsorted.clear();
		self.keys.clear();
		self.shown.clear();
		self.shown.resize(world.entities.slots(), 0);

		for spark in world.sparks.iter() {
			let Some(emitter) = world.entities.emitter(spark.owner) else {
				continue;
			};

			// a hidden entity's cloud is not drawn, and it goes on being
			// thrown: the step never asks. @ref `Entities::set_hidden`.
			if !showing(&mut self.shown, world, spark.owner) {
				continue;
			}

			let through = spark.through();
			// the picture's slot, clamped into the table the way a
			// renderable's material is: an emitter naming a texture that has
			// gone is drawn with the white one rather than not at all.
			let texture = if emitter.texture.slot() < textures {
				emitter.texture.slot()
			} else {
				TextureId::NONE.slot()
			};

			self.keys.push((
				// the pipeline first and the picture second, which is what
				// makes each run one `set_pipeline` and one `set_bind_group`.
				// Nothing below it: the sort *within* a cloud is deliberately
				// not done - @ref the module docs on `SparkBlend::Additive`.
				u32::try_from(emitter.blend.row()).unwrap_or(0),
				u32::try_from(texture).unwrap_or(0),
			));
			self.unsorted.push(SparkInstance {
				position: spark.position.to_array(),
				size: mix(emitter.size, emitter.size_end, through).max(0.0),
				color: mix3(emitter.color, emitter.color_end, through)
					.extend(spark.fade() * emitter.opacity.clamp(0.0, 1.0))
					.to_array(),
			});
		}

		// the two are sorted together, which is why the key is beside the
		// instance rather than in it: thirty-two bytes moved per swap instead
		// of forty, and the key never reaches the GPU.
		let mut order: Vec<u32> = (0..u32::try_from(self.unsorted.len()).unwrap_or(0)).collect();

		order.sort_unstable_by_key(|at| self.keys[usize::try_from(*at).unwrap_or(0)]);

		for at in &order {
			let at = usize::try_from(*at).unwrap_or(0);
			let (blend, texture) = self.keys[at];
			let blend = SparkBlend::at(blend).unwrap_or_default();
			let texture = usize::try_from(texture).unwrap_or(0);
			let first = u32::try_from(self.sorted.len()).unwrap_or(0);

			self.sorted.push(self.unsorted[at]);

			match self.runs.last_mut() {
				| Some(run) if run.blend == blend && run.texture == texture => run.count += 1,
				| _ => self
					.runs
					.push(Run { blend, texture, first, count: 1 }),
			}
		}
	}
}

/// Whether the entity that threw a particle is drawn, worked out once a frame
/// for each entity rather than once for each particle.
///
/// Remembered by slot, which is sound only for an owner the emitter lookup has
/// already found alive: for the length of a frame a slot has one occupant.
///
/// @param known - one word a slot, nought where nothing has been asked yet
/// @param world - whose table answers
/// @param owner - the entity that threw the particle
fn showing(known: &mut [u8], world: &World, owner: EntityId) -> bool {
	let Some(seen) = known.get_mut(owner.slot()) else {
		return world.entities.shown(owner);
	};

	if *seen == 0 {
		*seen = if world.entities.shown(owner) { 1 } else { 2 };
	}

	*seen == 1
}

/// The instance stride, asserted to match the attributes above.
const fn stride() -> BufferAddress {
	const {
		assert!(size_of::<SparkInstance>() == 32, "SparkInstance is no longer eight floats");
		assert!(align_of::<SparkInstance>() == 4, "SparkInstance gained padding");
	}

	32
}

#[cfg(test)]
mod tests {
	use colby_core::abi::{Emitter, EmitterKind, Spark, Textures};

	use super::*;

	/// Lays a world's cloud out the way [`Sparks::upload`] does, without a
	/// device: [`Cloud::lay_out`] is that half of the upload, so this runs the
	/// very loop a frame runs rather than a copy of it.
	fn laid_out(world: &World, textures: usize) -> (Vec<SparkInstance>, Vec<Run>) {
		let mut cloud = Cloud::default();

		cloud.lay_out(world, textures);

		(cloud.sorted, cloud.runs)
	}

	/// A world with one emitter and a cloud of `count` particles in it.
	fn clouded(emitter: Emitter, count: usize) -> (World, EntityId) {
		let mut world = World::new();
		let id = world.entities.spawn();

		assert!(world.entities.set_emitter(id, emitter), "the handle resolves");

		for number in 0..count {
			assert!(
				world.sparks.push(Spark {
					position: Vec3::new(f32::from(u8::try_from(number).unwrap_or(0)), 0.0, 0.0),
					velocity: Vec3::ZERO,
					age: 0.5,
					life: 1.0,
					owner: id,
				}),
				"there is room"
			);
		}

		(world, id)
	}

	#[test]
	fn a_cloud_of_one_picture_and_one_blend_is_one_run() {
		let (world, _) = clouded(Emitter::point(10.0, 1.0), 64);
		let (instances, runs) = laid_out(&world, Textures::new().len());

		assert_eq!(instances.len(), 64, "every particle became an instance");
		assert_eq!(runs.len(), 1, "and one picture at one blend is one draw call");
		assert_eq!(runs[0].count, 64, "carrying all of them");
		assert_eq!(runs[0].first, 0, "from the start of the buffer");
	}

	#[test]
	fn two_blends_are_two_runs_and_the_additive_one_comes_first() {
		let mut world = World::new();
		let table = Textures::new().len();

		for (number, blend) in [SparkBlend::Alpha, SparkBlend::Additive]
			.into_iter()
			.enumerate()
		{
			let id = world.entities.spawn();

			assert!(
				world
					.entities
					.set_emitter(id, Emitter { blend, ..Emitter::point(10.0, 1.0) }),
				"the handle resolves"
			);
			assert!(
				world.sparks.push(Spark {
					position: Vec3::new(f32::from(u8::try_from(number).unwrap_or(0)), 0.0, 0.0),
					velocity: Vec3::ZERO,
					age: 0.5,
					life: 1.0,
					owner: id,
				}),
				"one particle each"
			);
		}

		let (_, runs) = laid_out(&world, table);

		assert_eq!(runs.len(), 2, "two blends cannot share a pipeline");
		assert_eq!(
			runs[0].blend,
			SparkBlend::Additive,
			"and the rows decide the order, so the additive half is first however the emitters \
			 were made"
		);
	}

	#[test]
	fn a_picture_that_is_not_in_the_table_falls_back_to_white() {
		let (world, _) = clouded(
			Emitter {
				texture: TextureId::new(900),
				..Emitter::point(10.0, 1.0)
			},
			4,
		);
		let (_, runs) = laid_out(&world, Textures::new().len());

		assert_eq!(runs.len(), 1, "one run");
		assert_eq!(
			runs[0].texture,
			TextureId::NONE.slot(),
			"drawn with the white texel rather than not at all"
		);
	}

	#[test]
	fn size_and_color_are_read_off_the_emitter_at_the_particle_s_age() {
		let (world, _) = clouded(
			Emitter {
				size: 1.0,
				size_end: 3.0,
				color: Vec3::new(1.0, 0.0, 0.0),
				color_end: Vec3::new(0.0, 0.0, 1.0),
				opacity: 1.0,
				..Emitter::point(10.0, 1.0)
			},
			1,
		);
		let (instances, _) = laid_out(&world, Textures::new().len());

		// the particle is half way through its life
		assert!(
			(instances[0].size - 2.0).abs() < 1e-5,
			"the size is half way: {}",
			instances[0].size
		);
		assert!(
			(instances[0].color[0] - 0.5).abs() < 1e-5
				&& (instances[0].color[2] - 0.5).abs() < 1e-5,
			"and so is the color"
		);
		assert!(
			(instances[0].color[3] - 1.0).abs() < 1e-5,
			"and half way through, a particle is at its brightest"
		);
	}

	#[test]
	fn a_particle_whose_emitter_has_gone_is_not_drawn() {
		let (mut world, id) = clouded(Emitter::point(10.0, 1.0), 4);

		assert!(world.entities.despawn(id), "the emitter goes");

		let (instances, runs) = laid_out(&world, Textures::new().len());

		assert!(instances.is_empty(), "and the frame draws none of what it threw");
		assert!(runs.is_empty(), "with no run to draw them in");
	}

	#[test]
	fn an_emitter_at_nought_opacity_still_makes_instances_and_they_are_clear() {
		let (world, _) = clouded(
			Emitter {
				opacity: 0.0,
				..Emitter::point(10.0, 1.0)
			},
			2,
		);
		let (instances, _) = laid_out(&world, Textures::new().len());

		assert_eq!(instances.len(), 2, "the cloud is still there");
		assert!(instances[0].color[3].abs() < 1e-6, "and every particle of it is invisible");
	}

	#[test]
	fn a_cloud_whose_emitter_hangs_off_something_hidden_is_not_drawn_and_is_kept() {
		let (mut world, id) = clouded(Emitter::point(10.0, 1.0), 4);
		let torch = world.entities.spawn();
		assert!(world.entities.set_parent(id, torch), "it hangs");
		assert!(world.entities.set_hidden(torch, true), "the handle resolves");

		let (instances, runs) = laid_out(&world, Textures::new().len());

		assert!(instances.is_empty() && runs.is_empty(), "the frame draws none of it");
		assert_eq!(world.sparks.len(), 4, "while the pool still holds all four");

		assert!(world.entities.set_hidden(torch, false));

		let (instances, _) = laid_out(&world, Textures::new().len());

		assert_eq!(instances.len(), 4, "and shown again, the same four are drawn");
	}

	#[test]
	fn two_clouds_in_one_pool_are_each_drawn_or_not_by_their_own_emitter() {
		// what is asked is remembered a slot at a time, so particles of a hidden
		// emitter and a shown one taking turns in the pool are what catches an
		// answer remembered for the wrong slot, or for no slot at all
		let mut world = World::new();
		let shown = world.entities.spawn();
		let hidden = world.entities.spawn();

		for id in [shown, hidden] {
			assert!(
				world
					.entities
					.set_emitter(id, Emitter::point(10.0, 1.0))
			);
		}

		assert!(world.entities.set_hidden(hidden, true));

		for number in 0..6_u8 {
			assert!(
				world.sparks.push(Spark {
					position: Vec3::new(f32::from(number), 0.0, 0.0),
					velocity: Vec3::ZERO,
					age: 0.5,
					life: 1.0,
					owner: if number.is_multiple_of(2) { hidden } else { shown },
				}),
				"there is room"
			);
		}

		let (instances, _) = laid_out(&world, Textures::new().len());

		assert_eq!(instances.len(), 3, "the shown one's three and none of the other's");
		assert!(
			instances.iter().all(|it| [1.0_f32, 3.0, 5.0]
				.iter()
				.any(|x| (it.position[0] - x).abs() < 1e-6)),
			"and they are the shown one's"
		);
	}

	#[test]
	fn a_cloud_laid_out_again_asks_again_rather_than_remembering_the_last_frame() {
		// the upload keeps one cloud for the life of the scene, so what it
		// remembered about a slot has to be forgotten between two frames: a
		// hide between them has to reach the second
		let (mut world, id) = clouded(Emitter::point(10.0, 1.0), 4);
		let mut cloud = Cloud::default();

		cloud.lay_out(&world, Textures::new().len());

		assert_eq!(cloud.sorted.len(), 4, "drawn to begin with");
		assert!(world.entities.set_hidden(id, true));

		cloud.lay_out(&world, Textures::new().len());

		assert!(cloud.sorted.is_empty(), "and the next frame asks again");
	}

	#[test]
	fn a_world_with_no_cloud_draws_nothing() {
		let world = World::new();
		let (instances, runs) = laid_out(&world, Textures::new().len());

		assert!(instances.is_empty(), "no particles");
		assert!(runs.is_empty(), "and no run");
	}

	#[test]
	fn a_kind_of_none_still_draws_what_it_threw_before_it_was_turned_off() {
		let (mut world, id) = clouded(Emitter::point(10.0, 1.0), 4);

		assert!(
			world
				.entities
				.set_emitter(id, Emitter { kind: EmitterKind::None, ..Emitter::NONE }),
			"turned off"
		);

		let (instances, _) = laid_out(&world, Textures::new().len());

		assert_eq!(
			instances.len(),
			4,
			"the renderer draws the pool it is handed; what is in the pool is the step's \
			 business"
		);
	}
}
