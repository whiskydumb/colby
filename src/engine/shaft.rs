//! The light the air catches around the sun.
//!
//! A pass after the scene, and the second reader of the depth buffer after the
//! view that draws it. @ref [`depth`](crate::depth).
//!
//! **What it smears is the picture, not a sun.** There is no disc in the sky to
//! blur - the sky is a gradient in three colors - so the light comes from what
//! the picture already holds where the depth says nothing near was drawn, which
//! in a world with a sky is the sky. That also settles the color: light through
//! a gap is the color of what is behind the gap, so this needs no tint of its
//! own. A world with no sky at all smears its clear color, which is the same
//! sentence.
//!
//! **Three passes, two of them at a quarter of the picture on each axis.** The
//! first cuts the mask out of the picture, the second drags it outwards from
//! where the sun is on the screen, and the third adds the result back into the
//! picture - before the eye measures anything and before the bloom gathers, so
//! a strong shaft stops the eye down and glows at its edges. A smear has no
//! detail in it by construction, which is what makes a sixteenth of the pixels
//! the right place to compute it.
//!
//! **Only the sun.** The mechanism is a blur away from one point on the screen,
//! and the point has to be in front of the camera for that to mean anything; a
//! lamp standing in the room is a volume rather than a smear, and a different
//! job. Both engines in the field with this shape refuse everything but their
//! directional light.

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

/// How many pixels of the picture one pixel of the mask stands for, per axis.
///
/// Matched by `SCALE` in `shaft.wgsl`, which turns a pixel of the mask back
/// into the pixel of the depth buffer at the middle of the block it covers.
/// The field is split: one engine here halves and one quarters, and quartering
/// is what the note left by the depth buffer's own card asked for.
const SCALE: u32 = 4;

/// How many taps the smear takes towards the sun.
///
/// Matched by `TAPS` in `shaft.wgsl`, which is where the loop is and where a
/// frame reads it. Here, and only under a test, because what the march sums to
/// is a closed form a picture can be checked against - and the sum needs the
/// count. @ref the test that does it.
#[cfg(test)]
const TAPS: i32 = 64;

/// How far along the way to the sun the march reaches.
const REACH: f32 = 0.65;

/// How much one tap of the march adds.
const WEIGHT: f32 = 0.25;

/// How much less the next tap adds than the one before it.
const DECAY: f32 = 0.945;

/// How far of the sun anything happens at all, in pictures high.
///
/// Without it the whole sky would be dragged over the whole picture and the
/// effect would be a wash rather than rays. Half a picture's height is what the
/// engine that writes this out longhand uses.
const WIDE: f32 = 0.5;

/// How squarely the sun has to be faced for the smear to be all the way on.
///
/// The cosine between the way the camera looks and the way the sun lies. At
/// nothing and below there is no smear at all, because a point behind the
/// camera has no place on the screen; between the two it eases in.
///
/// **Eased rather than faded over time**, which is where this parts company
/// with the field: an engine that keeps a fade factor and moves it by the
/// length of the frame makes a picture depend on how long the window has been
/// up, and one process here renders exactly one frame and writes it to a file.
/// The same curve without the history costs one line and is the same picture
/// every time.
const FACING: f32 = 0.25;

/// The numbers the passes read, laid out the way `shaft.wgsl` declares them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct Tuning {
	/// `[where the sun is across the picture, and down it, how far of that
	/// anything happens, the picture's width over its height]`.
	sun: [f32; 4],

	/// `[how far the march reaches, what one tap adds, how fast that falls,
	/// how much is put back]`.
	smear: [f32; 4],

	/// `[how far along the view light starts counting, the projection's two
	/// numbers, unused]`.
	range: [f32; 4],
}

/// What one frame asks the smear for.
///
/// Worked out where the rest of the frame's numbers are, in
/// [`Scene::upload`](crate::Scene), for the reason the depth view's are: it
/// needs the camera the frame is drawn from and the console has not been asked
/// anything by the time the passes are recorded.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Asking {
	/// Where the sun is, in the coordinates a tap is taken at.
	sun: [f32; 2],

	/// How much of the smear is put back, with the fade by how squarely the
	/// sun is faced already folded in.
	strength: f32,

	/// The picture's width over its height.
	aspect: f32,

	/// How far along the view a surface has to be before its light counts,
	/// which is the middle of the camera's own range.
	middle: f32,

	/// The projection's `z_axis.z` and `w_axis.z`.
	lens: [f32; 2],
}

/// The two quarter-sized buffers and the ways to read them.
struct Buffers {
	/// The picture cut down to what the air may catch.
	mask: TextureView,
	mask_read: BindGroup,

	/// That mask dragged outwards from the sun.
	smear: TextureView,
	smear_read: BindGroup,
}

/// Everything the three passes need, and the passes.
pub(crate) struct Shaft {
	/// The size of the picture, which the buffers below are a quarter of.
	size: (u32, u32),

	/// The two quarter buffers, built the first frame something asks.
	///
	/// **Kept once built rather than let go the way the resolved depth is.**
	/// The sun passing behind the camera turns the smear off for as long as
	/// somebody is looking the other way, and a world that gave a megabyte back
	/// every time a person turned round would be paying for the wrong thing. A
	/// world that never asks never builds them.
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
	mask: RenderPipeline,
	smear: RenderPipeline,
	apply: RenderPipeline,
	device: Device,
}

impl Shaft {
	/// Builds the three pipelines and the block they read their numbers from.
	///
	/// No texture yet: a world that asks for no shafts, which is every world
	/// until somebody says otherwise, never makes one.
	///
	/// @param device - the device to build against
	/// @param width - the picture's width in pixels
	/// @param height - its height
	pub(crate) fn new(device: &Device, width: u32, height: u32) -> Result<Self> {
		let (numbers_layout, texture_layout, depth_layout) = layouts(device);
		// clamped rather than repeating, as every pass over a picture here is:
		// a tap that ran off the edge and came back on the other side would
		// fold the left of the screen into the right, and this one marches
		// straight off the edge whenever the sun is past it.
		let sampler = device.create_sampler(&SamplerDescriptor {
			label: Some("shaft"),
			address_mode_u: AddressMode::ClampToEdge,
			address_mode_v: AddressMode::ClampToEdge,
			address_mode_w: AddressMode::ClampToEdge,
			mag_filter: FilterMode::Linear,
			min_filter: FilterMode::Linear,
			..SamplerDescriptor::default()
		});
		let tuning = device.create_buffer(&BufferDescriptor {
			label: Some("shaft tuning"),
			size: u64::try_from(size_of::<Tuning>())
				.map_err(|_| err!(Graphics("the shaft's tuning block does not fit a buffer")))?,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let numbers = device.create_bind_group(&BindGroupDescriptor {
			label: Some("shaft numbers"),
			layout: &numbers_layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: tuning.as_entire_binding(),
			}],
		});
		let scope = device.push_error_scope(ErrorFilter::Validation);
		let module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("shaft"),
			source: ShaderSource::Wgsl(include_str!("shaft.wgsl").into()),
		});
		let mask = screen_pipeline(device, &module, "shaft mask", "fragment_mask", None, &[
			Some(&numbers_layout),
			Some(&texture_layout),
			Some(&depth_layout),
		]);
		let smear = screen_pipeline(device, &module, "shaft smear", "fragment_smear", None, &[
			Some(&numbers_layout),
			Some(&texture_layout),
		]);
		// the one pass here that blends rather than replaces: the smear goes
		// over a picture that is already drawn, and what light does to a
		// picture is add to it.
		let apply = screen_pipeline(
			device,
			&module,
			"shaft apply",
			"fragment_apply",
			Some(BlendState {
				color: BlendComponent {
					src_factor: BlendFactor::One,
					dst_factor: BlendFactor::One,
					operation: BlendOperation::Add,
				},
				alpha: BlendComponent::REPLACE,
			}),
			&[Some(&numbers_layout), Some(&texture_layout)],
		);

		if let Some(complaint) = pollster::block_on(scope.pop()) {
			return Err(err!(Graphics("the light shaft pipelines: {complaint}")));
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
			mask,
			smear,
			apply,
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
	/// @param encoder - the frame's, with the scene's pass and the depth's
	/// resolve already in it
	/// @param queue - where the tuning block is written
	/// @param asked - what this frame wants, or nothing
	/// @param picture - the target the world was drawn into, which is both
	/// what the mask is cut from and what the smear is added back to
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
		let (Some(asked), Some(depth), epoch) = (asked, depth.readable(), depth.epoch()) else {
			return;
		};

		queue.write_buffer(&self.tuning, 0, bytemuck::bytes_of(&tuning_of(asked)));

		if self.buffers.is_none() {
			self.buffers =
				Some(build(&self.device, &self.sampler, &self.texture_layout, self.size));
		}

		if self.picture.is_none() {
			self.picture = Some(sampled(
				&self.device,
				&self.sampler,
				&self.texture_layout,
				picture,
				"shaft picture",
			));
		}

		// kept rather than made each frame, which is what the depth buffer's
		// own card left as a debt: this runs in every frame a world has shafts
		// in, and a group is only safe to keep while the view inside it is the
		// view that exists. @ref [`Depth::epoch`](crate::depth::Depth::epoch).
		if self
			.reading
			.as_ref()
			.is_none_or(|(held, _)| *held != epoch)
		{
			let group = self
				.device
				.create_bind_group(&BindGroupDescriptor {
					label: Some("shaft depth"),
					layout: &self.depth_layout,
					entries: &[BindGroupEntry {
						binding: 0,
						resource: BindingResource::TextureView(depth),
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
			"shaft mask",
			&buffers.mask,
			false,
			timings.writes(Pass::Shaft, Ends::Open),
			|pass| {
				pass.set_pipeline(&self.mask);
				pass.set_bind_group(0, &self.numbers, &[]);
				pass.set_bind_group(1, source, &[]);
				pass.set_bind_group(2, reading, &[]);
			},
		);
		screen_pass(
			encoder,
			"shaft smear",
			&buffers.smear,
			false,
			timings.writes(Pass::Shaft, Ends::Middle),
			|pass| {
				pass.set_pipeline(&self.smear);
				pass.set_bind_group(0, &self.numbers, &[]);
				pass.set_bind_group(1, &buffers.mask_read, &[]);
			},
		);
		// and back over the picture, which is loaded rather than cleared
		// because the world is already in it
		screen_pass(
			encoder,
			"shaft apply",
			picture,
			true,
			timings.writes(Pass::Shaft, Ends::Close),
			|pass| {
				pass.set_pipeline(&self.apply);
				pass.set_bind_group(0, &self.numbers, &[]);
				pass.set_bind_group(1, &buffers.smear_read, &[]);
			},
		);
	}
}

/// What this frame asks the smear for, if anything.
///
/// Nothing when the world wants no shafts, when the sun has no direction at
/// all, or when it lies behind the camera - and the last of those three is the
/// same number as the fade, which is worth saying out loud. The projection's
/// last row makes the `w` of a direction pushed through it the cosine between
/// that direction and the way the camera looks, so "is the sun in front" and
/// "how squarely is it faced" are one quantity read twice.
///
/// @param world - for the strength and the sun's direction
/// @param camera - the camera this frame is drawn from
#[must_use]
pub(crate) fn asking_of(world: &World, camera: &Camera) -> Option<Asking> {
	if !world.post.is_shafting() {
		return None;
	}

	let towards = (-world.light).try_normalize()?;
	let clip = camera.view_projection(world.aspect) * towards.extend(0.0);
	let facing = clip.w;

	if facing <= 0.0 {
		return None;
	}

	let across = clip.x / facing;
	let down = clip.y / facing;
	let sun = [across.mul_add(0.5, 0.5), down.mul_add(-0.5, 0.5)];

	if !sun[0].is_finite() || !sun[1].is_finite() {
		return None;
	}

	let lens = camera.projection(world.aspect);

	Some(Asking {
		sun,
		strength: world.post.shafts * eased(facing),
		aspect: world.aspect.max(0.001),
		// the middle of the camera's own range rather than a distance
		// somebody sets: the sky is at the far plane by construction, so the
		// far plane is the only distance this can be written against.
		middle: camera.far.max(camera.near + 0.001) * 0.5,
		lens: [lens.z_axis.z, lens.w_axis.z],
	})
}

/// How much of the smear a cosine of [`FACING`] or less is worth.
///
/// A smooth step rather than a line, so that the smear arriving and leaving has
/// no corner in it. Zero at and below nothing, one at and above [`FACING`].
fn eased(facing: f32) -> f32 {
	let t = (facing / FACING).clamp(0.0, 1.0);

	t * t * 2.0_f32.mul_add(-t, 3.0)
}

/// The tuning block for one frame.
fn tuning_of(asked: Asking) -> Tuning {
	Tuning {
		sun: [asked.sun[0], asked.sun[1], WIDE, asked.aspect],
		smear: [REACH, WEIGHT, DECAY, asked.strength],
		range: [asked.middle, asked.lens[0], asked.lens[1], 0.0],
	}
}

/// The two quarter buffers and the ways to read them.
fn build(
	device: &Device,
	sampler: &Sampler,
	layout: &BindGroupLayout,
	(width, height): (u32, u32),
) -> Buffers {
	let small = ((width / SCALE).max(1), (height / SCALE).max(1));
	let mask = quarter(device, small, "shaft mask");
	let smear = quarter(device, small, "shaft smear");
	let mask_read = sampled(device, sampler, layout, &mask, "shaft mask");
	let smear_read = sampled(device, sampler, layout, &smear, "shaft smear");

	Buffers { mask, mask_read, smear, smear_read }
}

/// One buffer a quarter of the picture across, in the picture's own format.
fn quarter(device: &Device, (width, height): (u32, u32), label: &str) -> TextureView {
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
/// Its own rather than borrowed from the module next door, and for the reason
/// that module's own are private: a pipeline layout is what a pipeline
/// requires, and these three pipelines require a different list from any of
/// that module's seven.
fn layouts(device: &Device) -> (BindGroupLayout, BindGroupLayout, BindGroupLayout) {
	let numbers = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("shaft numbers"),
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
		label: Some("shaft source"),
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
		label: Some("shaft depth"),
		entries: &[crate::depth::entry(0)],
	});

	(numbers, texture, depth)
}

/// One pass over one triangle covering its target.
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
				write_mask: ColorWrites::ALL,
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
	const SIZE: (u32, u32) = (320, 240);

	/// How far in front of the camera a thing that blocks the sun stands.
	const NEAR: f32 = 5.0;

	/// Where the pillar's two edges fall, in hundredths of the way across the
	/// picture.
	///
	/// Just to one side of the sun, which is in the middle, and wide enough
	/// that the march from the reading below crosses most of it - including
	/// the near end, which is where the weights are.
	const SHADE: (u32, u32) = (52, 60);

	/// The two places the shadow of that pillar is read: one with the pillar
	/// between it and the sun, one with nothing in the way, both the same
	/// distance from the sun and both sky.
	const READ: (u32, u32) = (62, 38);

	/// How much of the smear the pictures here ask for.
	///
	/// Half, which is where the field starts it. The sky below is dim on
	/// purpose to go with it: the smear adds about its own brightness again
	/// near the sun, and a picture that clamps is one two strengths cannot be
	/// told apart in.
	const SOME: f32 = 0.5;

	/// How much of the closed form below actually lands at the middle pixel.
	///
	/// **Measured, and the shortfall is interpolation rather than
	/// arithmetic.** The falloff away from the sun is at its greatest exactly
	/// where that test reads, and both the mask and the smear are sampled
	/// between texels on the way back to the picture - so what lands is the
	/// average of a peak rather than the peak itself. It is a property of
	/// where the sun is and how big the buffers are, and of nothing the march
	/// does, which is what makes it a constant here rather than a slack
	/// tolerance.
	const LANDED: f64 = 0.947;

	/// How far either side of [`LANDED`] the measurement may be.
	///
	/// Two percent, which is narrow enough to see the distrust of the
	/// picture's edges taken out - that one is worth six and a half - and wide
	/// enough for a float to round.
	const NEARLY: f64 = 0.02;

	/// A capture on the binary's one device, or `None` with no GPU.
	fn capture() -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, SIZE.0, SIZE.1) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// A world under a dim sky with the sun dead ahead.
	///
	/// No curve and no eye: what is measured here is a difference between two
	/// pictures, and an eye that adapted to the brighter of them would move
	/// every pixel of it.
	fn lit() -> World {
		let mut world = World::new();

		world.post = Post {
			tonemap: ToneMap::None,
			auto_exposure: false,
			exposure: 1.0,
			..Post::DEFAULT
		};
		world.ambient = Vec3::splat(0.2);
		world.sky = Sky::gradient(Vec3::splat(0.08), Vec3::splat(0.12), Vec3::splat(0.04));
		// the shape a capture will overwrite this with anyway, said here
		// because [`x_of`] is asked where a share of the picture is before
		// any frame has been drawn
		world.aspect = f32::from(u16::try_from(SIZE.0).unwrap_or(u16::MAX))
			/ f32::from(u16::try_from(SIZE.1).unwrap_or(u16::MAX));
		world.camera.position = Vec3::ZERO;
		world.camera.target = Vec3::NEG_Z;
		// traveling +z, so it comes from -z, which is where the camera looks
		world.light = Vec3::Z;
		world.cvars.var(MSAA, Value::Float(1.0), "");

		world
	}

	/// Which column a share of the way across the picture is.
	///
	/// In hundredths and by integers throughout, so that nothing here turns a
	/// float into an index.
	///
	/// @param share - hundredths of the way across, from the left
	fn column_of(share: u32) -> u32 { SIZE.0 * share / 100 }

	/// The same share as a fraction of the way across.
	fn share_of(share: u32) -> f32 { f32::from(u16::try_from(share).unwrap_or(0)) / 100.0 }

	/// Where a share of the way across the picture is in world space, on a
	/// plane [`NEAR`] in front of the camera.
	fn x_of(world: &World, share: u32) -> f32 {
		let across = world.camera.projection(world.aspect).x_axis.x;

		share_of(share).mul_add(2.0, -1.0) * NEAR / across
	}

	/// A box between the sun and one side of the picture, running off the top
	/// and the bottom of it.
	fn pillar(world: &mut World) {
		let (left, right) = (x_of(world, SHADE.0), x_of(world, SHADE.1));
		let id = world.entities.spawn_at(Transform {
			position: Vec3::new((left + right) * 0.5, 0.0, -NEAR - 0.5),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(right - left, 40.0, 1.0),
		});

		world
			.entities
			.set_renderable(id, Renderable::new(MeshId::CUBE, rgb(0.25, 0.25, 0.25)));
	}

	/// A wall covering the whole picture, nearer than the middle of the view.
	fn wall(world: &mut World) {
		let id = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, 0.0, -NEAR - 0.5),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(200.0, 200.0, 1.0),
		});

		world
			.entities
			.set_renderable(id, Renderable::new(MeshId::CUBE, rgb(0.25, 0.25, 0.25)));
	}

	/// The picture this world draws at a strength.
	fn shot(capture: &mut Capture, world: &mut World, shafts: f32) -> Image {
		world.post.shafts = shafts;

		capture.shoot(world).expect("the capture renders")
	}

	/// How much brighter one pixel of a picture is than the same pixel of
	/// another, in the red channel.
	fn added(then: &Image, now: &Image, column: u32) -> i32 {
		let row = SIZE.1 / 2;

		i32::from(now.pixel(column, row)[0]) - i32::from(then.pixel(column, row)[0])
	}

	#[test]
	fn the_sky_around_the_sun_brightens_and_less_where_something_stands_in_the_way() {
		// **the test that says the mask is the depth.** The two pixels read are
		// the same pixel before anything is smeared - the sky's color depends
		// on how far up a ray points and not on which side of the picture it
		// is - and they sit the same distance from the sun, so the falloff
		// gives them the same amount. The only thing that can tell them apart
		// is that the march from one of them crosses a pillar.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();

		pillar(&mut world);

		let dark = shot(&mut capture, &mut world, 0.0);
		let (shaded, clear) = (column_of(READ.0), column_of(READ.1));

		assert_eq!(
			dark.pixel(shaded, SIZE.1 / 2),
			dark.pixel(clear, SIZE.1 / 2),
			"the fixture is symmetric about the sun before anything is smeared"
		);

		let smeared = shot(&mut capture, &mut world, SOME);
		let behind = added(&dark, &smeared, shaded);
		let open = added(&dark, &smeared, clear);

		assert!(behind > 0 && open > 0, "both sides of the sun caught light: {behind} {open}");
		assert!(
			open > behind + 10,
			"and the side with the pillar in the way caught less: {behind} against {open}"
		);
	}

	#[test]
	fn what_the_air_catches_falls_off_away_from_the_sun() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();
		let dark = shot(&mut capture, &mut world, 0.0);
		let smeared = shot(&mut capture, &mut world, SOME);
		// on a picture with nothing in it, so that the shape measured is the
		// falloff and not an occluder
		let mut last = 0;

		for share in [10, 20, 30, 40] {
			let here = added(&dark, &smeared, column_of(share));

			assert!(
				here > last,
				"{share} hundredths across caught {here}, no more than the {last} further out"
			);

			last = here;
		}
	}

	#[test]
	fn a_picture_with_nothing_past_the_middle_of_the_view_catches_no_light() {
		// the wall is five units away and the camera's range is two hundred,
		// so every pixel is inside the half light may not come from. The sun
		// is where it was and the strength is what it was: what changed is
		// that the depth says there is nothing far away to catch, and the
		// answer is the same picture byte for byte.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();

		wall(&mut world);

		let dark = shot(&mut capture, &mut world, 0.0);
		let smeared = shot(&mut capture, &mut world, SOME);

		assert!(
			smeared.pixels == dark.pixels,
			"a picture with no distance in it catches nothing at all"
		);
	}

	#[test]
	fn a_strength_of_nothing_and_a_sun_behind_the_camera_are_both_the_plain_picture() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();

		pillar(&mut world);

		let dark = shot(&mut capture, &mut world, 0.0);

		for again in [0.0, -1.0] {
			let same = shot(&mut capture, &mut world, again);

			assert!(same.pixels == dark.pixels, "a strength of {again} draws the plain picture");
		}

		// the same world with the light traveling the other way, which puts the
		// sun behind the camera rather than in front of it. Its own pair of
		// pictures rather than a comparison with the two above: the sun lights
		// the pillar as well as smearing, so turning it round changes the
		// picture for a reason that has nothing to do with this.
		world.light = Vec3::NEG_Z;

		let away = shot(&mut capture, &mut world, 0.0);
		let behind = shot(&mut capture, &mut world, SOME);

		assert!(
			behind.pixels == away.pixels,
			"and a sun behind the camera has no place on the screen to smear from"
		);
	}

	#[test]
	fn the_smear_costs_three_passes_and_only_while_a_world_asks_for_it() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();

		pillar(&mut world);

		let mut frame = |world: &mut World, samples: &str, shafts: f32| {
			world.cvars.set(MSAA, samples);
			world.post.shafts = shafts;
			capture.draw(world, &mut []);

			capture.scene_mut().spans().passes()
		};

		let quiet = frame(&mut world, "1", 0.0);
		let asked = frame(&mut world, "1", SOME);
		let again = frame(&mut world, "1", 0.0);

		assert_eq!(asked, quiet + 3, "a mask, a smear and one pass putting it back");
		assert_eq!(again, quiet, "and none of the three in a frame that does not ask");

		let many = frame(&mut world, "4", 0.0);
		let asked = frame(&mut world, "4", SOME);

		assert_eq!(
			asked,
			many + 4,
			"at four samples it is the three and the resolve that makes the depth readable"
		);
	}

	#[test]
	fn a_resized_picture_is_smeared_at_the_new_size() {
		// the two buffers are a quarter of the picture, so a picture that
		// changed size and buffers that did not would be read outside
		// themselves - or, worse, read inside themselves and smeared by the
		// wrong distance. A capture's own target does not move with the
		// scene's, so what is asked here is that a frame still renders at
		// either size and still catches light at both.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();
		let middle = SIZE.0 / 2;

		pillar(&mut world);
		capture.draw(&mut world, &mut []);

		for size in [(SIZE.0 / 2, SIZE.1 / 2), SIZE] {
			capture.scene_mut().resize(size.0, size.1);

			let dark = shot(&mut capture, &mut world, 0.0);
			let smeared = shot(&mut capture, &mut world, SOME);

			assert!(
				added(&dark, &smeared, middle) > 0,
				"a picture of {size:?} catches light around the sun"
			);
		}
	}

	/// A world looking down -z, with the sun wherever the light comes from.
	///
	/// @param light - the direction the light *travels*, which is the
	/// direction the sun lies in turned round
	fn looking(light: Vec3, shafts: f32) -> World {
		let mut world = World::new();

		world.aspect = 16.0 / 9.0;
		world.camera.position = Vec3::ZERO;
		world.camera.target = Vec3::NEG_Z;
		world.light = light;
		world.post = Post { shafts, ..Post::DEFAULT };

		world
	}

	#[test]
	fn a_world_that_asks_for_nothing_is_asked_nothing() {
		let world = looking(Vec3::new(0.0, -0.2, 1.0), 0.0);

		assert!(
			asking_of(&world, &world.render_camera()).is_none(),
			"a strength of nothing records no passes at all"
		);
	}

	#[test]
	fn the_sun_dead_ahead_lands_in_the_middle_of_the_picture() {
		// light traveling away down -z comes *from* +z, which is behind a
		// camera looking down -z; so the sun dead ahead is light traveling +z
		let world = looking(Vec3::Z, 0.5);
		let asked = asking_of(&world, &world.render_camera()).expect("the sun is ahead");

		assert!(
			(asked.sun[0] - 0.5).abs() < 1.0e-5 && (asked.sun[1] - 0.5).abs() < 1.0e-5,
			"the sun is at {:?}, not the middle",
			asked.sun
		);
		assert!(
			(asked.strength - 0.5).abs() < 1.0e-6,
			"and squarely faced, so the strength is what was asked for: {}",
			asked.strength
		);
	}

	#[test]
	fn the_sun_above_the_camera_lands_above_the_middle_and_the_other_way_round() {
		for (light, above) in
			[(Vec3::new(0.0, -0.4, 1.0), true), (Vec3::new(0.0, 0.4, 1.0), false)]
		{
			let world = looking(light, 0.5);
			let asked = asking_of(&world, &world.render_camera()).expect("the sun is ahead");

			assert!(
				(asked.sun[1] < 0.5) == above,
				"light traveling {light:?} puts the sun at {:?}, and above is {above}",
				asked.sun
			);
			assert!(
				(asked.sun[0] - 0.5).abs() < 1.0e-5,
				"and it has not moved sideways: {:?}",
				asked.sun
			);
		}
	}

	#[test]
	fn a_sun_behind_the_camera_is_no_smear_and_one_at_the_side_is_a_faint_one() {
		let behind = looking(Vec3::NEG_Z, 0.5);

		assert!(
			asking_of(&behind, &behind.render_camera()).is_none(),
			"a point behind the camera has no place on the screen"
		);

		// a sun a degree off square to the view is as good as behind: the
		// cosine is near nothing, and the ease takes the strength with it
		let sideways = looking(Vec3::new(-1.0, 0.0, 0.02), 0.5);
		let asked = asking_of(&sideways, &sideways.render_camera())
			.expect("it is in front, if only barely");

		assert!(
			asked.strength > 0.0 && asked.strength < 0.01,
			"a sun that far round is nearly nothing: {}",
			asked.strength
		);

		let square = looking(Vec3::Z, 0.5);
		let full = asking_of(&square, &square.render_camera()).expect("the sun is ahead");

		assert!(full.strength > asked.strength, "and less than one dead ahead");
	}

	#[test]
	fn a_sun_with_no_direction_at_all_is_no_smear() {
		let world = looking(Vec3::ZERO, 0.5);

		assert!(
			asking_of(&world, &world.render_camera()).is_none(),
			"nothing to normalize is nothing to point at"
		);
	}

	#[test]
	fn the_ease_is_flat_at_both_ends_and_climbs_between_them() {
		assert!(eased(0.0).abs() < 1.0e-9, "nothing at square nothing");
		assert!((eased(FACING) - 1.0).abs() < 1.0e-6, "all of it at the threshold");
		assert!((eased(1.0) - 1.0).abs() < 1.0e-6, "and no more past it");

		let (low, mid, high) = (eased(FACING * 0.25), eased(FACING * 0.5), eased(FACING * 0.75));

		assert!(low < mid && mid < high, "it climbs: {low} {mid} {high}");
		assert!((mid - 0.5).abs() < 1.0e-6, "and is half way at half way");
		assert!(low < 0.25 - 0.05, "flat at the bottom: {low}");
		assert!(high > 0.75 + 0.05, "and at the top: {high}");
	}

	#[test]
	fn the_numbers_reach_the_block_in_the_order_the_shader_reads_them() {
		let world = looking(Vec3::Z, 0.75);
		let asked = asking_of(&world, &world.render_camera()).expect("the sun is ahead");
		let tuning = tuning_of(asked);

		for (at, (seen, meant)) in [
			(tuning.sun[2], WIDE),
			(tuning.sun[3], world.aspect),
			(tuning.smear[0], REACH),
			(tuning.smear[1], WEIGHT),
			(tuning.smear[2], DECAY),
			(tuning.smear[3], 0.75),
			(tuning.range[0], world.camera.far * 0.5),
			(
				tuning.range[1],
				world
					.render_camera()
					.projection(world.aspect)
					.z_axis
					.z,
			),
		]
		.into_iter()
		.enumerate()
		{
			assert!(
				(seen - meant).abs() < 1.0e-4,
				"the number at {at} reached the block as {seen}, not {meant}"
			);
		}
	}

	/// A linear value as an sRGB target holds it, and back.
	///
	/// The picture is written into a target that applies the curve, so a byte
	/// read out of it is not the light that was there. These two are the
	/// standard transfer function, written out here because the only other
	/// copy is in a shader.
	fn undone(byte: u8) -> f64 {
		let level = f64::from(byte) / 255.0;

		if level <= 0.04045 {
			level / 12.92
		} else {
			((level + 0.055) / 1.055).powf(2.4)
		}
	}

	/// How much light one pixel gained between two pictures, in the units the
	/// smear was added in.
	fn gained(then: &Image, now: &Image, at: (u32, u32)) -> f64 {
		undone(now.pixel(at.0, at.1)[0]) - undone(then.pixel(at.0, at.1)[0])
	}

	/// What one pixel of mask turns into once the march is summed.
	///
	/// The march from the sun's own pixel is degenerate - every step is the
	/// zero vector, so all [`TAPS`] taps read that same pixel - which makes
	/// the smear there the mask times one plus a geometric series, and that is
	/// a closed form a test can check a picture against.
	fn summed() -> f64 {
		let decay = f64::from(DECAY);
		let series = (1.0 - decay.powi(TAPS)) / (1.0 - decay);

		f64::from(WEIGHT).mul_add(series, 1.0)
	}

	/// The distrust of the picture's edges, at the middle of it.
	fn middling() -> f64 {
		// the same line the shader's `reaching` runs, at the middle of the
		// picture: a half across and a half down, and the fourth power of what
		// comes out
		let edge = (0.5 * 0.5 * 0.5 * 0.5_f64).mul_add(-8.0, 1.0);

		(edge * edge * edge).mul_add(-edge, 1.0)
	}

	#[test]
	fn the_light_put_back_at_the_sun_is_the_mask_there_times_what_the_march_sums_to() {
		// **the one pixel where the whole of the smear is arithmetic.** The
		// march away from the sun's own pixel steps nowhere, so every tap
		// reads that pixel and the sum is a geometric series; the mask there
		// is the sky the picture already holds, times the distrust of the
		// edges at the middle of the picture, times a falloff that is one
		// because the distance from the sun is nought. So what the last pass
		// adds is a number rather than a shape, and this is that number.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = lit();
		let middle = (SIZE.0 / 2, SIZE.1 / 2);
		let dark = shot(&mut capture, &mut world, 0.0);
		let smeared = shot(&mut capture, &mut world, SOME);
		let sky = undone(dark.pixel(middle.0, middle.1)[0]);
		let meant = sky * middling() * summed() * f64::from(SOME);
		let seen = gained(&dark, &smeared, middle);
		let (landed, wanted) = (seen / meant, LANDED);

		assert!(
			(landed - wanted).abs() < NEARLY,
			"the middle gained {seen:.5} of the {meant:.5} worked out, which is {landed:.4} of \
			 it rather than {wanted}"
		);
	}

	#[test]
	fn the_falloff_around_the_sun_leaves_the_far_side_of_a_mirrored_pair_in_shade() {
		// **what says the falloff away from the sun is doing anything.** The
		// distrust of the picture's edges depends on how far across the
		// picture a pixel is and not on which side, so two pixels mirrored
		// about the middle are distrusted exactly alike - and with the sun off
		// to one side they are two different distances from it. Nothing else
		// tells them apart.
		let Some(mut capture) = capture() else {
			return;
		};
		// the light turned so that the sun lands a quarter of the way across:
		// the projection scales the direction by one over the tangent of the
		// half angle over the shape of the picture, and 0.3642 of that is a
		// half of clip space
		let mut world = lit();

		world.light = Vec3::new(0.3642, 0.0, 1.0);

		let dark = shot(&mut capture, &mut world, 0.0);
		let smeared = shot(&mut capture, &mut world, SOME);
		let row = SIZE.1 / 2;
		let (near, far) = (column_of(40), column_of(60));

		assert_eq!(
			dark.pixel(near, row),
			dark.pixel(far, row),
			"the two are the same pixel before anything is smeared"
		);

		let close = gained(&dark, &smeared, (near, row));
		let away = gained(&dark, &smeared, (far, row));

		assert!(close > 0.0 && away >= 0.0, "both are light or nothing: {close} {away}");
		assert!(
			close > away * 3.0,
			"the one nearer the sun gained {close:.5} and the one further {away:.5}"
		);
	}

	#[test]
	fn a_frame_that_stopped_asking_and_asked_again_reads_the_depth_of_the_frame_it_is_in() {
		// **the bind group the smear keeps.** At four samples a pixel the
		// depth it reads is a buffer of its own, and that buffer is let go the
		// moment a frame stops asking and made again the next time one does -
		// so a group kept across those two frames holds a view of a texture
		// nothing has written since. What it would read is the picture before
		// last, which is what moving the pillar in between makes visible.
		let (Some(mut walked), Some(mut fresh)) = (capture(), capture()) else {
			return;
		};
		let mut world = lit();

		world.cvars.set(MSAA, "4");
		pillar(&mut world);

		// the same world with the pillar on the other side of the sun, which
		// is a different depth buffer and a different smear
		let mut moved = lit();

		moved.cvars.set(MSAA, "4");
		moved.light = world.light;

		let (left, right) = (x_of(&moved, 100 - SHADE.1), x_of(&moved, 100 - SHADE.0));
		let id = moved.entities.spawn_at(Transform {
			position: Vec3::new((left + right) * 0.5, 0.0, -NEAR - 0.5),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(right - left, 40.0, 1.0),
		});

		moved
			.entities
			.set_renderable(id, Renderable::new(MeshId::CUBE, rgb(0.25, 0.25, 0.25)));

		// a capture that never saw the other world, for the answer to be right
		let straight = shot(&mut fresh, &mut moved, SOME);
		// and one walked through asking, not asking, and asking again
		let asked = shot(&mut walked, &mut world, SOME);

		shot(&mut walked, &mut world, 0.0);

		let again = shot(&mut walked, &mut moved, SOME);

		assert!(
			asked.pixels != straight.pixels,
			"the two worlds are different pictures, or this proves nothing"
		);
		assert!(
			again.pixels == straight.pixels,
			"a frame that asked again read the depth of its own frame"
		);
	}
}
