//! The lens out of focus: what is not at the distance the camera focuses on.
//!
//! A pass after the scene, and the third reader of the depth buffer after the
//! view that draws it and the smear around the sun. @ref
//! [`depth`](crate::depth).
//!
//! **Three numbers, and none of them is a millimeter.** How far away the lens
//! is focused, how far off that a surface is blurred all the way, and how wide
//! that blur gets as a radius in pixels. A physical lens would want a focal
//! length and an aperture, and a focal length has to come from a sensor size -
//! which this camera does not have and would have to be invented for it, at
//! which point the blur changes when somebody zooms and nobody can predict it.
//! The radius in pixels is also the number that bounds what this costs, and in
//! a physical model there is no such number at all: the two engines here with
//! one carry a separate, admittedly unphysical, ceiling beside it.
//!
//! **Three passes, two of them at half the picture on each axis.** A separable
//! gaussian across and then down, at a radius each pixel takes from its own
//! distance, and one full-size pass blending the result back over the picture.
//! Half is the smallest reduction anything in the field takes for this, and the
//! reduction costs no pass of its own: one linear tap at a half-size texel's
//! center is exactly the average of the four pixels under it.
//!
//! **What this does not do is spill.** A near thing out of focus blurs its own
//! pixels and leaves the sharp background beside it sharp, where a real lens
//! would spread it over the edge. Every arrangement that spills wants either a
//! test on every tap at the widest radius in the picture, or a pre-sort into a
//! near half and a far half with weights of their own - several times three
//! passes. This is the cheaper of the two shapes the field ships, and the
//! engine nearest this one in language and API ships it as its default.

use colby_core::{
	Result,
	abi::{Camera, World},
	bytemuck::{self, Pod, Zeroable},
	err,
};
use wgpu::{
	AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
	BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType,
	BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer, BufferBindingType,
	BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, CommandEncoder, Device,
	ErrorFilter, Extent3d, FilterMode, FragmentState, LoadOp, MultisampleState, Operations,
	PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState, Queue,
	RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
	Sampler, SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource,
	ShaderStages, StoreOp, TextureDescriptor, TextureDimension, TextureSampleType, TextureUsages,
	TextureView, TextureViewDescriptor, TextureViewDimension, VertexState,
};

use crate::{
	depth::Depth,
	post::HDR_FORMAT,
	timing::{Ends, Pass, Timings},
};

/// How many pixels of the picture one pixel of the blur stands for, per axis.
///
/// Matched by `SCALE` in `focus.wgsl`. Two, which is what the two engines here
/// that reduce at all reduce by; the other two blur at full size and one of
/// those reduces only at its lower two quality settings.
const SCALE: u32 = 2;

/// The widest blur a world may ask for, as a radius in pixels.
///
/// Twice `TAPS` in `focus.wgsl`, because a tap there is a texel of the
/// half-size buffer and a texel there is two pixels of the picture. It is a
/// ceiling on the cost rather than a look, which is exactly what the two
/// engines here with a physical model call theirs while admitting it is not
/// physical.
const MAX_BLUR: f32 = 64.0;

/// How many taps the blur takes on each side of a pixel at [`MAX_BLUR`].
///
/// Matched by `TAPS` in `focus.wgsl`, where the loop is. Here, and only under
/// a test, because what a run of taps sums to is a closed form a picture can be
/// checked against - and the sum needs the count.
#[cfg(test)]
const TAPS: i32 = 32;

/// The smallest distance a blur may be complete over.
///
/// Clamped here rather than left to the shader so that the number in the
/// buffer is the number the picture was drawn with: a range of nothing is a
/// step from sharp to fully blurred at the plane in focus, which is what a
/// world that wrote a nought there asked for.
const MIN_RANGE: f32 = 1.0e-4;

/// The numbers the passes read, laid out the way `focus.wgsl` declares them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct Tuning {
	/// `[how far away the lens is focused, how far off that blur is complete,
	/// how wide it gets there in pixels, unused]`.
	lens: [f32; 4],

	/// `[the projection's two numbers, unused, unused]`.
	range: [f32; 4],

	/// `[one texel of the half-size buffer across, one down, unused,
	/// unused]`.
	texel: [f32; 4],
}

/// What one frame asks the blur for.
///
/// Worked out where the rest of the frame's numbers are, in
/// [`Scene::upload`](crate::Scene), for the reason the smear's are: it needs
/// the camera the frame is drawn from, and the passes are recorded later.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Asking {
	/// How far away the lens is focused.
	focus: f32,

	/// How far off that a surface is blurred all the way, never below
	/// [`MIN_RANGE`].
	range: f32,

	/// How wide the blur gets there, in pixels, never above [`MAX_BLUR`].
	blur: f32,

	/// The projection's `z_axis.z` and `w_axis.z`.
	lens: [f32; 2],
}

/// The two half-size buffers and the ways to read them.
struct Buffers {
	/// The picture reduced and blurred across.
	across: TextureView,
	across_read: BindGroup,

	/// The same, blurred down as well.
	down: TextureView,
	down_read: BindGroup,
}

/// Everything the three passes need, and the passes.
pub(crate) struct Focus {
	/// The size of the picture, which the buffers below are half of.
	size: (u32, u32),

	/// The two half-size buffers, built the first frame something asks.
	///
	/// **Kept once built rather than let go**, the way the smear's are and
	/// unlike the resolved depth: a focus pull crosses in and out of a world
	/// that wants no blur at all several times a second, and giving three
	/// megabytes back and taking them again on each crossing would be paying
	/// for the wrong thing. A world that never focuses never builds them.
	buffers: Option<Buffers>,

	/// How the picture is read, rebuilt when the picture is.
	picture: Option<BindGroup>,

	/// How the depth is read, and which rebuild of it that group is for. @ref
	/// [`Depth::epoch`](crate::depth::Depth::epoch).
	reading: Option<(u64, BindGroup)>,

	tuning: Buffer,
	numbers: BindGroup,
	sampler: Sampler,
	texture_layout: BindGroupLayout,
	depth_layout: BindGroupLayout,
	across: RenderPipeline,
	down: RenderPipeline,
	over: RenderPipeline,
	device: Device,
}

impl Focus {
	/// Builds the three pipelines and the block they read their numbers from.
	///
	/// No texture yet: a world whose camera focuses on nothing, which is every
	/// world until somebody says otherwise, never makes one.
	///
	/// @param device - the device to build against
	/// @param width - the picture's width in pixels
	/// @param height - its height
	pub(crate) fn new(device: &Device, width: u32, height: u32) -> Result<Self> {
		let (numbers_layout, texture_layout, depth_layout) = layouts(device);
		// clamped rather than repeating, as every pass over a picture here is:
		// a tap that ran off the edge and came back on the other side would
		// fold the left of the screen into the right, and a wide blur at the
		// edge of the picture marches straight off it.
		let sampler = device.create_sampler(&SamplerDescriptor {
			label: Some("focus"),
			address_mode_u: AddressMode::ClampToEdge,
			address_mode_v: AddressMode::ClampToEdge,
			address_mode_w: AddressMode::ClampToEdge,
			mag_filter: FilterMode::Linear,
			min_filter: FilterMode::Linear,
			..SamplerDescriptor::default()
		});
		let tuning = device.create_buffer(&BufferDescriptor {
			label: Some("focus tuning"),
			size: u64::try_from(size_of::<Tuning>())
				.map_err(|_| err!(Graphics("the lens tuning block does not fit a buffer")))?,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let numbers = device.create_bind_group(&BindGroupDescriptor {
			label: Some("focus numbers"),
			layout: &numbers_layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: tuning.as_entire_binding(),
			}],
		});
		let scope = device.push_error_scope(ErrorFilter::Validation);
		let module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("focus"),
			source: ShaderSource::Wgsl(include_str!("focus.wgsl").into()),
		});
		let across = screen_pipeline(
			device,
			&module,
			"focus across",
			"fragment_across",
			None,
			ColorWrites::ALL,
			&[Some(&numbers_layout), Some(&texture_layout), Some(&depth_layout)],
		);
		let down = screen_pipeline(
			device,
			&module,
			"focus down",
			"fragment_down",
			None,
			ColorWrites::ALL,
			&[Some(&numbers_layout), Some(&texture_layout)],
		);
		// the one pass here that blends rather than replaces, and the blend is
		// the whole trick: the source's alpha says how much of the blur
		// belongs at this pixel, so the picture underneath is mixed in by the
		// hardware and this pass never has to read it. **The alpha channel is
		// left alone** - nothing downstream reads it, and a pass that wrote it
		// would be writing a mix factor where a picture's alpha used to be.
		let over = screen_pipeline(
			device,
			&module,
			"focus over",
			"fragment_over",
			Some(BlendState {
				color: BlendComponent {
					src_factor: BlendFactor::SrcAlpha,
					dst_factor: BlendFactor::OneMinusSrcAlpha,
					operation: BlendOperation::Add,
				},
				alpha: BlendComponent::REPLACE,
			}),
			ColorWrites::COLOR,
			&[Some(&numbers_layout), Some(&texture_layout), Some(&depth_layout)],
		);

		if let Some(complaint) = pollster::block_on(scope.pop()) {
			return Err(err!(Graphics("the depth of field pipelines: {complaint}")));
		}

		Ok(Self {
			size: (width, height),
			buffers: None,
			picture: None,
			reading: None,
			tuning,
			numbers,
			sampler,
			texture_layout,
			depth_layout,
			across,
			down,
			over,
			device: device.clone(),
		})
	}

	/// Rebuilds what a new picture size changes.
	///
	/// The buffers are dropped rather than rebuilt: the next frame that asks
	/// makes them at the new size, and a frame that does not ask has saved
	/// itself the allocation.
	pub(crate) fn resize(&mut self, width: u32, height: u32) {
		if self.size == (width, height) {
			return;
		}

		self.size = (width, height);
		self.buffers = None;
		self.picture = None;
	}

	/// Records the three passes, or nothing at all.
	///
	/// @param encoder - the frame's, with the scene's pass, the depth's
	/// resolve and the smear already in it
	/// @param queue - where the tuning block is written
	/// @param asked - what this frame wants, or nothing
	/// @param picture - the target the world was drawn into, which is both
	/// what the blur is taken from and what it is put back over
	/// @param depth - the buffer the scene wrote, for the one sample a pixel
	/// it hands out and for which rebuild of that view this is
	/// @param timings - what the passes write their marks into
	pub(crate) fn render(
		&mut self,
		encoder: &mut CommandEncoder,
		queue: &Queue,
		asked: Option<Asking>,
		picture: &TextureView,
		depth: &Depth,
		timings: &Timings,
	) {
		let (Some(asked), Some(view), epoch) = (asked, depth.readable(), depth.epoch()) else {
			return;
		};
		let half = half_of(self.size);

		queue.write_buffer(&self.tuning, 0, bytemuck::bytes_of(&tuning_of(asked, half)));

		if self.buffers.is_none() {
			self.buffers = Some(build(&self.device, &self.sampler, &self.texture_layout, half));
		}

		if self.picture.is_none() {
			self.picture = Some(sampled(
				&self.device,
				&self.sampler,
				&self.texture_layout,
				picture,
				"focus picture",
			));
		}

		// kept rather than made each frame, for the reason the smear's is: a
		// group is only safe to keep while the view inside it is the view that
		// exists. @ref [`Depth::epoch`](crate::depth::Depth::epoch).
		if self
			.reading
			.as_ref()
			.is_none_or(|(held, _)| *held != epoch)
		{
			let group = self
				.device
				.create_bind_group(&BindGroupDescriptor {
					label: Some("focus depth"),
					layout: &self.depth_layout,
					entries: &[BindGroupEntry {
						binding: 0,
						resource: BindingResource::TextureView(view),
					}],
				});

			self.reading = Some((epoch, group));
		}

		let (Some(buffers), Some(source), Some((_, reading))) =
			(self.buffers.as_ref(), self.picture.as_ref(), self.reading.as_ref())
		else {
			return;
		};

		screen_pass(
			encoder,
			"focus across",
			&buffers.across,
			false,
			timings.writes(Pass::Focus, Ends::Open),
			|pass| {
				pass.set_pipeline(&self.across);
				pass.set_bind_group(0, &self.numbers, &[]);
				pass.set_bind_group(1, source, &[]);
				pass.set_bind_group(2, reading, &[]);
			},
		);
		screen_pass(
			encoder,
			"focus down",
			&buffers.down,
			false,
			timings.writes(Pass::Focus, Ends::Middle),
			|pass| {
				pass.set_pipeline(&self.down);
				pass.set_bind_group(0, &self.numbers, &[]);
				pass.set_bind_group(1, &buffers.across_read, &[]);
			},
		);
		// and back over the picture, which is loaded rather than cleared
		// because the world is already in it and this pass mixes with it
		screen_pass(
			encoder,
			"focus over",
			picture,
			true,
			timings.writes(Pass::Focus, Ends::Close),
			|pass| {
				pass.set_pipeline(&self.over);
				pass.set_bind_group(0, &self.numbers, &[]);
				pass.set_bind_group(1, &buffers.down_read, &[]);
				pass.set_bind_group(2, reading, &[]);
			},
		);
	}
}

/// What this frame asks the blur for, if anything.
///
/// Nothing when the camera focuses on nothing, when its blur is nothing wide,
/// or when any of the three numbers is not a number at all. The two that are
/// clamped are clamped here rather than in the shader, so that what a picture
/// was drawn with is what the buffer holds.
///
/// @param world - for the aspect the projection is built at
/// @param camera - the camera this frame is drawn from
#[must_use]
pub(crate) fn asking_of(world: &World, camera: &Camera) -> Option<Asking> {
	if !camera.is_focusing() {
		return None;
	}

	if !camera.focus.is_finite() || !camera.focus_range.is_finite() || !camera.blur.is_finite() {
		return None;
	}

	let lens = camera.projection(world.aspect);

	Some(Asking {
		focus: camera.focus,
		range: camera.focus_range.max(MIN_RANGE),
		blur: camera.blur.min(MAX_BLUR),
		lens: [lens.z_axis.z, lens.w_axis.z],
	})
}

/// How big the two buffers are: half the picture on each axis, and never
/// nothing.
fn half_of((width, height): (u32, u32)) -> (u32, u32) {
	((width / SCALE).max(1), (height / SCALE).max(1))
}

/// The tuning block for one frame.
fn tuning_of(asked: Asking, (width, height): (u32, u32)) -> Tuning {
	Tuning {
		lens: [asked.focus, asked.range, asked.blur, 0.0],
		range: [asked.lens[0], asked.lens[1], 0.0, 0.0],
		// one texel of the half-size buffer as a share of the picture, which
		// is the step both blurs take. Read off the size rather than off a
		// derivative, because the two passes read sources of two different
		// sizes and the step is the target's either way.
		texel: [1.0 / texels(width), 1.0 / texels(height), 0.0, 0.0],
	}
}

/// A count of texels as a number the shader can step by.
///
/// Through `u16` because this workspace allows no silent conversion, and a
/// picture wider than sixty-five thousand pixels is not a thing this runs on.
fn texels(count: u32) -> f32 { f32::from(u16::try_from(count.max(1)).unwrap_or(u16::MAX)) }

/// The two half-size buffers and the ways to read them.
fn build(
	device: &Device,
	sampler: &Sampler,
	layout: &BindGroupLayout,
	half: (u32, u32),
) -> Buffers {
	let across = half_size(device, half, "focus across");
	let down = half_size(device, half, "focus down");
	let across_read = sampled(device, sampler, layout, &across, "focus across");
	let down_read = sampled(device, sampler, layout, &down, "focus down");

	Buffers { across, across_read, down, down_read }
}

/// One buffer half the picture across, in the picture's own format.
fn half_size(device: &Device, (width, height): (u32, u32), label: &str) -> TextureView {
	device
		.create_texture(&TextureDescriptor {
			label: Some(label),
			size: Extent3d { width, height, depth_or_array_layers: 1 },
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: HDR_FORMAT,
			usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
			view_formats: &[],
		})
		.create_view(&TextureViewDescriptor::default())
}

/// A texture and the sampler beside it.
fn sampled(
	device: &Device,
	sampler: &Sampler,
	layout: &BindGroupLayout,
	view: &TextureView,
	label: &str,
) -> BindGroup {
	device.create_bind_group(&BindGroupDescriptor {
		label: Some(label),
		layout,
		entries: &[
			BindGroupEntry {
				binding: 0,
				resource: BindingResource::TextureView(view),
			},
			BindGroupEntry {
				binding: 1,
				resource: BindingResource::Sampler(sampler),
			},
		],
	})
}

/// The three ways a pass here is handed something: the numbers, a picture with
/// a sampler beside it, and the depth.
///
/// Its own rather than borrowed from either module next door, for the reason
/// theirs are private: a pipeline layout is what a pipeline requires, and these
/// three require a different list from any of those.
fn layouts(device: &Device) -> (BindGroupLayout, BindGroupLayout, BindGroupLayout) {
	let numbers = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("focus numbers"),
		entries: &[BindGroupLayoutEntry {
			binding: 0,
			visibility: ShaderStages::FRAGMENT,
			ty: BindingType::Buffer {
				ty: BufferBindingType::Uniform,
				has_dynamic_offset: false,
				min_binding_size: None,
			},
			count: None,
		}],
	});
	let texture = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("focus source"),
		entries: &[
			BindGroupLayoutEntry {
				binding: 0,
				visibility: ShaderStages::FRAGMENT,
				ty: BindingType::Texture {
					sample_type: TextureSampleType::Float { filterable: true },
					view_dimension: TextureViewDimension::D2,
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
	});
	let depth = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("focus depth"),
		entries: &[crate::depth::entry(0)],
	});

	(numbers, texture, depth)
}

/// One pass over one triangle covering its target.
///
/// @note: the two passes that do not go over the picture clear rather than
/// load, and every pixel of their targets is written - so nothing can observe
/// what the clear put there. It is the cheaper of the two loads all the same,
/// which is why it is the one chosen. @ref `B1-7`, which is the same note
/// about the depth resolve.
///
/// @param over - whether what is already in the target is kept
fn screen_pass<F>(
	encoder: &mut CommandEncoder,
	label: &str,
	view: &TextureView,
	over: bool,
	marks: Option<wgpu::RenderPassTimestampWrites<'_>>,
	setup: F,
) where
	F: FnOnce(&mut wgpu::RenderPass<'_>),
{
	let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
		label: Some(label),
		color_attachments: &[Some(RenderPassColorAttachment {
			view,
			depth_slice: None,
			resolve_target: None,
			ops: Operations {
				load: if over {
					LoadOp::Load
				} else {
					LoadOp::Clear(wgpu::Color::BLACK)
				},
				store: StoreOp::Store,
			},
		})],
		depth_stencil_attachment: None,
		timestamp_writes: marks,
		occlusion_query_set: None,
		multiview_mask: None,
	});

	setup(&mut pass);
	pass.draw(0..3, 0..1);
}

/// One pipeline over the full-screen triangle, with no vertex buffers and no
/// depth.
fn screen_pipeline(
	device: &Device,
	module: &wgpu::ShaderModule,
	label: &str,
	entry: &str,
	blend: Option<BlendState>,
	writes: ColorWrites,
	groups: &[Option<&BindGroupLayout>],
) -> RenderPipeline {
	let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some(label),
		bind_group_layouts: groups,
		immediate_size: 0,
	});

	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some(label),
		layout: Some(&layout),
		vertex: VertexState {
			module,
			entry_point: Some("vertex_screen"),
			compilation_options: PipelineCompilationOptions::default(),
			buffers: &[],
		},
		primitive: PrimitiveState::default(),
		depth_stencil: None,
		multisample: MultisampleState::default(),
		fragment: Some(FragmentState {
			module,
			entry_point: Some(entry),
			compilation_options: PipelineCompilationOptions::default(),
			targets: &[Some(ColorTargetState {
				format: HDR_FORMAT,
				blend,
				write_mask: writes,
			})],
		}),
		multiview_mask: None,
		cache: None,
	})
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{MeshId, Post, Renderable, Sky, ToneMap, Transform, Value},
		glam::{Quat, Vec3},
	};

	use super::*;
	use crate::{Capture, Image, capture::rgb, scene::MSAA};

	/// How big every picture here is.
	///
	/// Both a multiple of four, so that the middle of the picture is the edge
	/// of a texel of the half-size buffer as well and the closed form below
	/// has a seam to sit on.
	const SIZE: (u32, u32) = (320, 240);

	/// How far in front of the camera the wall of two colors stands.
	const WALL: f32 = 20.0;

	/// How wide the blur the pictures here ask for is, in pixels.
	///
	/// Sixteen: wide enough that its reach is many pixels and its shape can be
	/// read off several of them, and narrow enough that the eight taps it
	/// takes on each side stay well inside a picture this size.
	const WIDE: f32 = 16.0;

	/// How far off the plane in focus the blur here is complete.
	const RANGE: f32 = 4.0;

	/// How many taps the blur takes on each side of a pixel at [`WIDE`], and
	/// at a pixel less than that.
	///
	/// The shader works this out for itself as the radius in texels of the
	/// half-size buffer, rounded up and held under `TAPS`. [`WIDE`] is chosen
	/// so that it comes out a whole number and one pixel less comes out the
	/// same number rounded up - which is what lets the closed form below count
	/// in whole taps rather than rounding a float into a count, and what makes
	/// the second of its two radii the one that says which way the shader
	/// rounds. Pinned to what it is derived from by a test.
	const TAKEN: i32 = 8;

	/// How many bytes a prediction here may be out by.
	///
	/// Two, which is what a linear value read back out of an eight-bit sRGB
	/// picture, blurred, and written out through the same curve costs. The
	/// same tolerance the smear around the sun asked of its second answer.
	const NEARLY: i32 = 2;

	/// A capture on the binary's one device, or `None` with no GPU.
	fn capture() -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, SIZE.0, SIZE.1) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// A world with no curve and no eye, looking down `-z`.
	///
	/// What is measured here is a difference between two pictures of one
	/// scene, and an eye that adapted to either of them would move every pixel
	/// of the other. **The light travels `-z`, away from the camera**, so it
	/// comes from behind the camera and a face turned towards the camera is
	/// lit flat and bright - which is what gives the closed form below a step
	/// big enough to measure the shape of a kernel against.
	fn lit() -> World {
		let mut world = World::new();

		world.post = Post {
			tonemap: ToneMap::None,
			auto_exposure: false,
			exposure: 1.0,
			..Post::DEFAULT
		};
		world.ambient = Vec3::splat(0.05);
		world.sky = Sky::gradient(Vec3::splat(0.30), Vec3::splat(0.45), Vec3::splat(0.20));
		// the shape a capture will overwrite this with anyway, said here
		// because [`across`] is asked how wide the view is before any frame
		// has been drawn
		world.aspect = f32::from(u16::try_from(SIZE.0).unwrap_or(u16::MAX))
			/ f32::from(u16::try_from(SIZE.1).unwrap_or(u16::MAX));
		world.camera.position = Vec3::ZERO;
		world.camera.target = Vec3::NEG_Z;
		world.camera.focus_range = RANGE;
		world.camera.blur = WIDE;
		world.light = Vec3::NEG_Z;
		world.cvars.var(MSAA, Value::Float(1.0), "");

		world
	}

	/// Half of how wide the picture is in world units, this far along the
	/// view.
	fn across(world: &World, distance: f32) -> f32 {
		distance / world.camera.projection(world.aspect).x_axis.x
	}

	/// Half of how tall it is, which is not the same number.
	fn high(world: &World, distance: f32) -> f32 {
		distance / world.camera.projection(world.aspect).y_axis.y
	}

	/// A flat board standing across the view, from one place to another.
	///
	/// Its face lands exactly `distance` along the view, which is what makes
	/// the circle of confusion over it one number.
	///
	/// @param distance - how far along the view its face is
	/// @param from - where its left edge is, as a share of the half width
	/// @param to - where its right edge is
	fn board(world: &mut World, distance: f32, (from, to): (f32, f32), color: Vec3) {
		let half = across(world, distance);
		let (left, right) = (half * from, half * to);
		let id = world.entities.spawn_at(Transform {
			position: Vec3::new((left + right) * 0.5, 0.0, -distance - 0.5),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(right - left, distance * 4.0, 1.0),
		});

		world
			.entities
			.set_renderable(id, Renderable::new(MeshId::CUBE, color));
	}

	/// A wall of two colors filling the picture, its seam down the middle.
	fn wall(world: &mut World) {
		board(world, WALL, (-1.2, 0.0), rgb(0.12, 0.12, 0.12));
		board(world, WALL, (0.0, 1.2), rgb(0.65, 0.65, 0.65));
	}

	/// The same wall with as much between its two halves as a picture holds.
	///
	/// Its own rather than the one above, and the difference is what the two
	/// are for: a step this steep is what lets a closed form tell one kernel
	/// from another two texels out in its tail, and it is also what makes the
	/// last rounding of a pixel that should not have moved visible. The
	/// measurement wants the first and the identity wants the second.
	fn contrasted(world: &mut World) {
		board(world, WALL, (-1.2, 0.0), rgb(0.02, 0.02, 0.02));
		board(world, WALL, (0.0, 1.2), rgb(0.90, 0.90, 0.90));
	}

	/// The same wall with its seam across the middle instead of down it.
	///
	/// Placed about the height the camera looks at rather than about the
	/// ground, so the seam lands on the middle row.
	fn stacked(world: &mut World) {
		let tall = high(world, WALL);
		let eye = world.camera.position.y;

		for (color, from) in [(rgb(0.12, 0.12, 0.12), 0.5), (rgb(0.65, 0.65, 0.65), -0.5)] {
			let id = world.entities.spawn_at(Transform {
				position: Vec3::new(0.0, tall.mul_add(from, eye), -WALL - 0.5),
				rotation: Quat::IDENTITY,
				scale: Vec3::new(WALL * 4.0, tall, 1.0),
			});

			world
				.entities
				.set_renderable(id, Renderable::new(MeshId::CUBE, color));
		}
	}

	/// The picture this world draws with the lens focused somewhere.
	///
	/// @param focus - the distance in focus, or nought for no lens at all
	fn shot(capture: &mut Capture, world: &mut World, focus: f32) -> Image {
		world.camera.focus = focus;

		capture.shoot(world).expect("the capture renders")
	}

	/// One pixel of the middle row, in the red channel.
	fn level(image: &Image, column: u32) -> i32 { i32::from(image.pixel(column, SIZE.1 / 2)[0]) }

	/// A byte of an sRGB picture as the linear value behind it.
	fn linear(byte: i32) -> f64 {
		let level = f64::from(byte) / 255.0;

		if level <= 0.040_45 {
			level / 12.92
		} else {
			((level + 0.055) / 1.055).powf(2.4)
		}
	}

	/// A linear value as the byte an sRGB picture holds it as.
	fn encoded(value: f64) -> i32 {
		let level = value.clamp(0.0, 1.0);
		let curved = if level <= 0.003_130_8 {
			level * 12.92
		} else {
			1.055_f64.mul_add(level.powf(1.0 / 2.4), -0.055)
		};

		#[expect(
			clippy::as_conversions,
			clippy::cast_possible_truncation,
			reason = "a level in nought to one times 255, rounded, is a byte and nothing else"
		)]
		let byte = (curved * 255.0).round() as i32;

		byte
	}

	/// How much the picture changes across a column, at the middle row.
	///
	/// The sharpness of an edge, in bytes: a board with a hard silhouette
	/// against something of another color has a big one, and the same board
	/// out of focus has a small one.
	fn contrast(image: &Image, column: u32) -> i32 {
		(level(image, column + 1) - level(image, column.saturating_sub(1))).abs()
	}

	#[test]
	fn a_surface_at_the_plane_in_focus_comes_through_the_lens_all_but_untouched() {
		// **the negative control inside the tree**, and the whole of why the
		// last pass hands its mix to the blend rather than doing it itself: a
		// pixel on the plane in focus keeps the value the scene drew rather
		// than being copied through a half-size blur and back. The wall has a
		// hard seam in it, so a picture that had been through a blur at any
		// real strength would show it.
		//
		// **All but, and the shortfall is the depth buffer rather than the
		// blend.** Turning a stored depth back into a distance subtracts two
		// numbers a couple of thousandths apart, so a wall at exactly the
		// plane in focus comes back a ten-thousandth off it and its circle of
		// confusion is about a thousandth of a pixel rather than nothing. At a
		// hard seam that rounds about one channel in a hundred and forty the
		// other way, and never by more than one level.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();

		wall(&mut world);

		let sharp = shot(&mut capture, &mut world, 0.0);
		let quiet = capture.scene_mut().spans().passes();
		let focused = shot(&mut capture, &mut world, WALL);
		let asked = capture.scene_mut().spans().passes();
		let moved: Vec<i32> = focused
			.pixels
			.iter()
			.zip(sharp.pixels.iter())
			.map(|(now, then)| i32::from(*now) - i32::from(*then))
			.filter(|off| *off != 0)
			.collect();
		let worst = moved
			.iter()
			.map(|off| off.abs())
			.max()
			.unwrap_or(0);

		assert!(asked > quiet, "the lens did run: {asked} passes against {quiet}");
		assert!(
			worst <= 1,
			"nothing on the plane in focus moves by more than a level, got {worst}"
		);
		assert!(
			moved.len() * 100 < sharp.pixels.len(),
			"and fewer than one channel in a hundred moves at all, got {} of {}",
			moved.len(),
			sharp.pixels.len()
		);
	}

	#[test]
	fn the_blur_goes_down_as_well_as_across() {
		// the second of the two passes, which the first cannot stand in for: a
		// wall split top from bottom has a seam no blur across it can soften,
		// so the only thing that can soften it is the pass that blurs down.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();

		stacked(&mut world);

		let sharp = shot(&mut capture, &mut world, WALL);
		let blurred = shot(&mut capture, &mut world, RANGE.mul_add(-4.0, WALL));
		let middle = SIZE.0 / 2;
		let seam = SIZE.1 / 2;
		let step = |image: &Image, row: u32| {
			i32::from(image.pixel(middle, row + 1)[0])
				- i32::from(image.pixel(middle, row.saturating_sub(1))[0])
		};

		assert!(
			step(&sharp, seam).abs() > 40,
			"the two halves are far enough apart to see a blur between them, got {}",
			step(&sharp, seam)
		);
		assert!(
			step(&blurred, seam).abs() * 3 < step(&sharp, seam).abs(),
			"and blurring down takes most of that step away: {} against {}",
			step(&blurred, seam),
			step(&sharp, seam)
		);
	}

	#[test]
	fn the_blur_is_the_gaussian_its_two_constants_describe() {
		// **the closed form.** Every pixel of this wall is the same distance
		// away, so the circle of confusion over the whole picture is one
		// number and the blur is a plain separable gaussian of a known width.
		// What lands at the seam is then that kernel summed over a step -
		// worked out here from the two flat levels the sharp picture shows and
		// from nothing the shader says.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();

		contrasted(&mut world);

		let sharp = shot(&mut capture, &mut world, WALL);
		let (dark, bright) = (linear(level(&sharp, 40)), linear(level(&sharp, 280)));

		// **at two radii, and the second is the one that pins the tap count.**
		// Sixteen is eight whole texels of the half-size buffer, where a
		// kernel that rounded its reach down would take exactly as many taps
		// as one that rounded up; fifteen is seven and a half, where the two
		// differ by a tap.
		for wide in [WIDE, WIDE - 1.0] {
			world.camera.blur = wide;

			// nearer than the wall by more than the range, so the wall is
			// blurred all the way and the radius is what the camera asked for
			let blurred = shot(&mut capture, &mut world, RANGE.mul_add(-4.0, WALL));

			// out to the radius and past it, not only either side of the seam:
			// the middle of a blurred step is nearly the mean whatever the
			// kernel's shape is, and it is the tails that say how wide the
			// gaussian is and how far it reaches
			for column in (SIZE.0 / 2 - 20)..(SIZE.0 / 2 + 20) {
				let want = encoded(predicted(dark, bright, column, wide));
				let got = level(&blurred, column);

				assert!(
					(want - got).abs() <= NEARLY,
					"column {column} of a wall blurred by {wide} pixels: wanted {want}, got \
					 {got}"
				);
			}
		}

		world.camera.blur = WIDE;
	}

	/// What the three passes make of a step, at one column of the picture.
	///
	/// The first reduces and blurs across, so it works in texels of the
	/// half-size buffer, each standing for two columns; the second blurs down
	/// a picture that does not change down, which leaves it alone; and the
	/// last samples the half-size buffer at this column, which falls a quarter
	/// of a texel off a texel center and is therefore a blend of two of them.
	///
	/// @param dark - the linear level left of the seam
	/// @param bright - the linear level right of it
	/// @param column - which column of the picture to predict
	/// @param wide - the radius the camera asked for, in pixels
	fn predicted(dark: f64, bright: f64, column: u32, wide: f32) -> f64 {
		// what the shader works out for itself: the radius in texels of the
		// half-size buffer, two standard deviations wide, and the taps out to
		// it
		let radius = f64::from(wide) * 0.5;
		let sigma = radius * 0.5;
		let seam = f64::from(SIZE.0 / SCALE / 2);
		let step = |texel: f64| if texel < seam { dark } else { bright };
		let across = |texel: f64| {
			let mut total = step(texel);
			let mut weights = 1.0;

			for tap in 1..=TAKEN {
				let away = f64::from(tap);
				let weight = (-(away * away) / (2.0 * sigma * sigma)).exp();

				total = (step(texel + away) + step(texel - away)).mul_add(weight, total);
				weights = weight.mul_add(2.0, weights);
			}

			total / weights
		};
		// where this column lands in the half-size buffer, as a texel index
		// with a fraction: uv times the width, less the half texel a center
		// sits at
		let at = (f64::from(column) + 0.5) / f64::from(SCALE) - 0.5;
		let (left, part) = (at.floor(), at.fract());

		across(left).mul_add(1.0 - part, across(left + 1.0) * part)
	}

	#[test]
	fn the_focus_travels_and_the_sharpness_goes_with_it() {
		// three boards at three distances, each with the sky either side of
		// it, and the lens focused on each in turn. What is asked is that the
		// board in focus has the hardest edge of the three every time - which
		// is the whole of what this feature is for, said as a number.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();
		let away = [4.0_f32, 10.0, 30.0];
		// in hundredths of the way across the picture, so that where each
		// board's edge lands is a column and not a rounded float
		let sides = [(12_u32, 27_u32), (42, 57), (72, 87)];

		for (distance, (from, to)) in away.iter().zip(sides) {
			board(
				&mut world,
				*distance,
				(world_share(from), world_share(to)),
				rgb(0.02, 0.02, 0.02),
			);
		}

		// the left edge of each board, which is where the sky beside it and
		// the board itself meet
		let edges: Vec<u32> = sides
			.iter()
			.map(|(from, _)| column_of(*from))
			.collect();

		for (which, distance) in away.iter().enumerate() {
			let picture = shot(&mut capture, &mut world, *distance);
			let hardest: Vec<i32> = edges
				.iter()
				.map(|edge| contrast(&picture, *edge))
				.collect();
			let best = hardest
				.iter()
				.enumerate()
				.max_by_key(|(_, sharpness)| **sharpness)
				.map(|(at, _)| at);

			assert_eq!(
				best,
				Some(which),
				"focused at {distance}, the board at that distance has the hardest edge of the \
				 three, got {hardest:?}"
			);
		}
	}

	/// Which column a share of the way across the picture is.
	///
	/// In hundredths and by integers throughout, so that nothing here turns a
	/// float into an index.
	///
	/// @param share - hundredths of the way across, from the left
	fn column_of(share: u32) -> u32 { SIZE.0 * share / 100 }

	/// The same share in the coordinates [`board`] places things in, which run
	/// from minus one at the left of the picture to one at the right.
	fn world_share(share: u32) -> f32 {
		f32::from(u16::try_from(share).unwrap_or(0)) / 50.0 - 1.0
	}

	#[test]
	fn a_wider_blur_reaches_further() {
		// the radius is in pixels and says so, so twice the radius has to
		// smear a step about twice as far. Measured as the furthest column
		// from the seam that differs from the sharp picture by more than a
		// byte.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();

		wall(&mut world);

		let sharp = shot(&mut capture, &mut world, WALL);
		let mut reach = Vec::new();

		for blur in [WIDE * 0.5, WIDE] {
			world.camera.blur = blur;

			let blurred = shot(&mut capture, &mut world, RANGE.mul_add(-4.0, WALL));
			let far = (1..SIZE.0 / 4)
				.filter(|away| {
					let column = SIZE.0 / 2 + away;

					(level(&blurred, column) - level(&sharp, column)).abs() > 1
				})
				.max()
				.unwrap_or(0);

			reach.push(far);
		}

		let (narrow, wide) = (reach[0], reach[1]);

		assert!(narrow > 0, "a blur of {} pixels reaches somewhere", WIDE * 0.5);
		assert!(
			wide > narrow * 3 / 2,
			"twice the radius reaches half again as far at the very least: {wide} against \
			 {narrow}"
		);
		assert!(
			wide < narrow * 3,
			"and not three times as far, which would mean the radius is not what is being \
			 doubled: {wide} against {narrow}"
		);
	}

	#[test]
	fn what_wrote_no_depth_is_blurred_as_though_it_were_the_far_plane() {
		// **the known limit, pinned rather than hidden.** The sky writes no
		// depth, so a reader sees the clear, which is the far plane - and the
		// lens blurs it as the furthest thing there is. That is right for the
		// sky and wrong for the two other things that write no depth, glass
		// and particles, which is why this is a test of what happens rather
		// than a test that it is correct.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();
		let middle = SIZE.0 / 2;

		// a board covering the left half only, with the sky in the right half
		board(&mut world, WALL, (-1.2, 0.0), rgb(0.02, 0.02, 0.02));

		let sharp = shot(&mut capture, &mut world, 0.0);
		let horizon = world.camera.far;
		// the lens at the far plane itself, where the sky is: the sky comes
		// out of this untouched and the board, twenty units away, does not
		let distant = shot(&mut capture, &mut world, horizon);
		// and the lens on the board, where the sky is the thing out of focus
		let near = shot(&mut capture, &mut world, WALL);

		assert!(
			(level(&distant, middle + 40) - level(&sharp, middle + 40)).abs() <= 1,
			"focused at the far plane, a patch of sky is the picture: {} against {}",
			level(&distant, middle + 40),
			level(&sharp, middle + 40)
		);
		assert!(
			(level(&distant, middle - 2) - level(&sharp, middle - 2)).abs() > 8,
			"and the board at that moment is not: {} against {}",
			level(&distant, middle - 2),
			level(&sharp, middle - 2)
		);
		assert!(
			(level(&near, middle + 2) - level(&sharp, middle + 2)).abs() > 8,
			"focused on the board, it is the sky that smears at the seam: {} against {}",
			level(&near, middle + 2),
			level(&sharp, middle + 2)
		);
		assert!(
			contrast(&near, middle) < contrast(&sharp, middle),
			"which is the seam softening: {} against {}",
			contrast(&near, middle),
			contrast(&sharp, middle)
		);
	}

	#[test]
	fn the_lens_costs_three_passes_and_only_while_the_camera_focuses() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();

		wall(&mut world);

		let mut frame = |world: &mut World, samples: &str, focus: f32| {
			world.cvars.set(MSAA, samples);
			world.camera.focus = focus;
			capture.draw(world, &mut []);

			capture.scene_mut().spans().passes()
		};

		let quiet = frame(&mut world, "1", 0.0);
		let asked = frame(&mut world, "1", WALL);
		let again = frame(&mut world, "1", 0.0);

		assert_eq!(asked, quiet + 3, "two blurs at half size and one pass putting them back");
		assert_eq!(again, quiet, "and none of the three in a frame that does not ask");

		let many = frame(&mut world, "4", 0.0);
		let asked = frame(&mut world, "4", WALL);

		assert_eq!(
			asked,
			many + 4,
			"at four samples it is the three and the resolve that makes the depth readable"
		);
	}

	#[test]
	fn a_resized_picture_is_blurred_at_the_new_size() {
		// the two buffers are half the picture, so a picture that changed size
		// and buffers that did not would be read outside themselves - or,
		// worse, read inside themselves and blurred by the wrong distance.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();

		wall(&mut world);
		capture.draw(&mut world, &mut []);

		for size in [(SIZE.0 / 2, SIZE.1 / 2), SIZE] {
			for samples in ["1", "4"] {
				world.cvars.set(MSAA, samples);
				capture.scene_mut().resize(size.0, size.1);

				let sharp = shot(&mut capture, &mut world, WALL);
				let blurred = shot(&mut capture, &mut world, RANGE.mul_add(-4.0, WALL));
				// the picture read back is the capture's own size whatever the
				// scene was resized to - the composite stretches the one over
				// the other - so the seam is in the middle of that, not of this
				let seam = SIZE.0 / 2;

				assert!(
					contrast(&blurred, seam) < contrast(&sharp, seam),
					"a picture of {size:?} at {samples} samples still blurs its wall"
				);
			}
		}
	}

	#[test]
	fn a_frame_that_stopped_asking_for_the_depth_asks_again_and_reads_its_own() {
		// the bind group holding the depth is kept between frames and rebuilt
		// when the view under it is, and the view is let go the moment nothing
		// reads it. A group kept across that would be holding the depth of a
		// frame that is gone - so the world has to *move* between the two,
		// and by enough to change the circle of confusion rather than only the
		// picture. Four samples, because that is the count at which the
		// readable view is a second buffer rather than the scene's own.
		let (Some(mut capture), Some(mut fresh)) = (capture(), capture()) else {
			return;
		};
		let mut world = lit();
		let blurred = RANGE.mul_add(-4.0, WALL);

		wall(&mut world);
		world.cvars.set(MSAA, "4");

		let sharp = shot(&mut capture, &mut world, WALL);
		let first = shot(&mut capture, &mut world, blurred);
		// a frame nothing reads the depth in, which drops the resolved buffer
		let quiet = shot(&mut capture, &mut world, 0.0);

		// and now the camera walks up to the wall, which brings it from past
		// the end of the ramp to halfway along it: a stale depth would blur it
		// by the whole radius where this asks for half
		world.camera.position.z = RANGE.mul_add(0.5, blurred - WALL);
		world.camera.target = world.camera.position + Vec3::NEG_Z;

		let again = shot(&mut capture, &mut world, blurred);
		let never = shot(&mut fresh, &mut world, blurred);

		assert_eq!(
			again.pixels, never.pixels,
			"a lens that stopped and started again reads this frame depth, not the one it read 			 before it stopped"
		);
		assert!(
			contrast(&again, SIZE.0 / 2) > contrast(&first, SIZE.0 / 2),
			"and the wall it walked up to is less blurred than the one it stood back from: {} 			 against {}",
			contrast(&again, SIZE.0 / 2),
			contrast(&first, SIZE.0 / 2)
		);
		assert!(
			contrast(&first, SIZE.0 / 2) < contrast(&quiet, SIZE.0 / 2)
				&& contrast(&sharp, SIZE.0 / 2) == contrast(&quiet, SIZE.0 / 2),
			"and the frame between the two was the wall sharp, as sharp as with no lens at all"
		);
	}

	#[test]
	fn a_camera_that_focuses_on_nothing_is_asked_nothing() {
		let mut world = World::new();

		world.aspect = 16.0 / 9.0;
		world.camera.blur = WIDE;

		assert!(
			asking_of(&world, &world.render_camera()).is_none(),
			"a lens focused nowhere records no passes at all"
		);

		world.camera.focus = WALL;

		assert!(
			asking_of(&world, &world.render_camera()).is_some(),
			"and a distance turns it on"
		);

		for broken in [0.0, -1.0, f32::NAN, f32::INFINITY] {
			world.camera.blur = broken;

			assert!(
				asking_of(&world, &world.render_camera()).is_none(),
				"a radius of {broken} is no blur"
			);
		}

		world.camera.blur = WIDE;

		for broken in [f32::NAN, f32::INFINITY] {
			world.camera.focus = broken;

			assert!(
				asking_of(&world, &world.render_camera()).is_none(),
				"and a distance of {broken} is nowhere"
			);
		}
	}

	#[test]
	fn the_two_numbers_a_world_may_write_wrong_are_held_where_the_loop_can_take_them() {
		let mut world = World::new();

		world.aspect = 16.0 / 9.0;
		world.camera.focus = WALL;
		world.camera.blur = MAX_BLUR * 10.0;
		world.camera.focus_range = 0.0;

		let asked = asking_of(&world, &world.render_camera()).expect("a lens that is on");

		assert!(
			(asked.blur - MAX_BLUR).abs() < f32::EPSILON,
			"a radius wider than the loop can walk is the widest it can, got {}",
			asked.blur
		);
		assert!(
			asked.range >= MIN_RANGE,
			"and a range of nothing is a step rather than a division by it, got {}",
			asked.range
		);
	}

	#[test]
	fn the_shader_declares_the_entry_points_the_pipelines_ask_for() {
		// the pipelines name their entry points as strings, and a shader that
		// renamed one would fail at build time on a device and nowhere at all
		// on the stub. This is the cheap half of that check; the headless test
		// is the other.
		let source = include_str!("focus.wgsl");

		for entry in ["vertex_screen", "fragment_across", "fragment_down", "fragment_over"] {
			assert!(source.contains(&format!("fn {entry}(")), "the shader declares {entry}");
		}
	}

	#[test]
	fn the_three_numbers_both_languages_hold_agree() {
		// `SCALE` is what turns a pixel of the blur back into a pixel of the
		// depth buffer, and `TAPS` is the bound on a loop that `MAX_BLUR`
		// clamps a world down to. All three are written twice, once in each
		// language, and the two copies have to say the same thing.
		let source = include_str!("focus.wgsl");

		assert!(
			source.contains(&format!("const SCALE: i32 = {SCALE};")),
			"the shader reduces by what this module says it does"
		);
		assert!(
			source.contains(&format!("const TAPS: i32 = {TAPS};")),
			"and walks as far as this module thinks it does"
		);
		assert!(
			(MAX_BLUR - f32::from(u16::try_from(TAPS).unwrap_or(0)) * 2.0).abs() < f32::EPSILON,
			"and the widest blur a world may ask for is two pixels a tap, which is what a texel \
			 of a half-size buffer is worth"
		);
		assert!(
			f64::from(WIDE)
				.mul_add(0.5, -f64::from(TAKEN))
				.abs() < 1.0e-9
				&& TAKEN <= TAPS,
			"and the closed form counts the taps the shader would take at this radius"
		);
	}
}
