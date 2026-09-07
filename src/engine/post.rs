//! The passes between the last surface drawn and the screen.
//!
//! The world is drawn into a sixteen-bit float target rather than into the
//! window, so that a value past one is a number instead of white; this module
//! is what squeezes that down. What it owns is that target, the ladder that
//! measures how bright the picture is, the one texel that remembers what the
//! eye had got used to, and the pipeline that writes the result out.
//!
//! **The eye belongs to a surface, not to a world.** Two views of one scene are
//! entitled to be exposed differently - a thumbnail of an asset and a window
//! looking at a lamp are not the same picture - so the state lives here, beside
//! the target it measures, and a fresh one starts with no history.
//!
//! **No compute shader anywhere.** The average is taken by rendering the
//! picture into a fixed hundred-and-twenty-eight square of log luminance and
//! then halving that to one texel with a linear sampler, which is eight tiny
//! passes and needs no feature beyond what a triangle needs. Fyrox measures the
//! same way and in about the same number of lines; a histogram in a compute
//! pass is what the big engines do, and is what this becomes the day the meter
//! is measured to matter.

use colby_core::{
	Result,
	abi::{Post, ToneMap},
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
	RenderPassColorAttachment, RenderPassDescriptor, RenderPassTimestampWrites, RenderPipeline,
	RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor,
	ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, TextureDescriptor,
	TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
	TextureViewDescriptor, TextureViewDimension, VertexState,
};

use crate::timing::{Ends, Pass, Timings};

/// The format the world is drawn into.
///
/// Sixteen-bit floats, which is what Godot's render buffers are
/// (`R16G16B16A16_SFLOAT`) and what everything else with a tonemap uses. Eleven
/// bits and ten would be half the bandwidth and has no alpha channel, which the
/// blended pass needs.
pub(crate) const HDR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// How many samples a pixel is drawn with when the console asks for none.
pub(crate) const NO_SAMPLES: u32 = 1;

/// How many it is drawn with when the console asks for any.
///
/// **Four and only four, and that is wgpu's number rather than a taste.**
/// `MULTISAMPLE_X4` is what the WebGPU specification guarantees for a format
/// at all (`wgpu-types/src/texture/format.rs:917`), and
/// [`HDR_FORMAT`] is guaranteed `MULTISAMPLE_X4 | MULTISAMPLE_RESOLVE`
/// (`:991`) while `Depth32Float` is guaranteed the first of the two (`:1000`),
/// which is exactly what a multisampled scene needs, because a depth buffer is
/// never resolved. Two and eight would each need
/// `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` asked for and then checked per
/// format per adapter, which is three branches and three sets of pipelines to
/// buy "slightly worse" and "slightly better".
pub(crate) const SAMPLES: u32 = 4;

/// The format the meter and the eye are kept in.
///
/// One channel of half float holding a base-two logarithm: half a stop of
/// precision at the ends of a range no scene reaches, and filterable without
/// asking the device for a feature.
const METER_FORMAT: TextureFormat = TextureFormat::R16Float;

/// How many rungs the bloom chain may have.
///
/// Six halvings from half a picture reaches a couple of hundredths of its
/// width, which is as wide as a glow ever needs to spread. More rungs cost
/// more passes and buy a halo nobody asked for.
const GLOW_RUNGS: usize = 6;

/// How small a bloom rung may get before another one is not worth a pass.
const GLOW_SMALLEST: u32 = 8;

/// How wide the first rung of the ladder is.
///
/// Fixed rather than a share of the window, so that the number of passes does
/// not depend on how big somebody's screen is: a hundred and twenty-eight
/// halves seven times to one, always.
const METER_SIZE: u32 = 128;

/// The numbers the passes read, laid out the way `post.wgsl` declares them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct Tuning {
	/// `[which curve, the reinhard white point, the set exposure, whether it
	/// is measured]`.
	curve: [f32; 4],

	/// `[the key the meter aims for, the smallest exposure, the largest, how
	/// far the eye moves this frame]`.
	meter: [f32; 4],

	/// `[how much is added back, the threshold, the knee under it, unused]`.
	bloom: [f32; 4],
}

/// One rung of the ladder: a target, and the way to read it.
struct Rung {
	view: TextureView,
	read: BindGroup,
}

/// The target the world is drawn into, and everything that reads it.
pub(crate) struct Chain {
	/// What every pass of the scene draws into, and what the composite reads.
	///
	/// When the scene is multisampled this is what it *resolves into* rather
	/// than what it draws into, and everything downstream is unchanged either
	/// way. @ref [`multi`](Self::multi).
	target: TextureView,

	/// The multisampled target the scene draws into, when it is multisampled.
	///
	/// `None` is one sample a pixel and no resolve. Nothing samples this and
	/// nothing may: it is written by the scene's one pass and resolved into
	/// [`target`](Self::target) by the same pass.
	multi: Option<TextureView>,

	/// How many samples a pixel of the scene is drawn with.
	samples: u32,

	/// The way the composite reads that target.
	target_read: BindGroup,

	/// The ladder, widest first, ending at one texel.
	rungs: Vec<Rung>,

	/// The bloom chain, widest first at half the target and halving down.
	///
	/// Empty when the target is too small to halve at all, which is a
	/// thumbnail rather than a window and wants no glow anyway.
	glow: Vec<Rung>,

	/// The way the composite reads the widest rung of it.
	glow_read: BindGroup,

	/// Where the eye is, and where it is about to be.
	///
	/// Two, because a pass may not read the texture it writes. Which of them
	/// holds the answer is [`eye`](Self::eye).
	eyes: [Rung; 2],

	/// Which of the two holds the answer.
	eye: usize,

	/// Whether anything has ever been measured into it.
	///
	/// The whole of why a picture taken by a process that renders exactly one
	/// frame is a picture of an open eye. @ref the module docs.
	adapted: bool,

	tuning: Buffer,
	numbers: BindGroup,
	sampler: Sampler,
	texture_layout: BindGroupLayout,
	luminance: RenderPipeline,
	halve: RenderPipeline,
	adapt: RenderPipeline,
	threshold: RenderPipeline,
	down: RenderPipeline,
	up: RenderPipeline,
	composite: RenderPipeline,
	size: (u32, u32),
}

impl Chain {
	/// Builds the target, the ladder, the eye and the four pipelines.
	///
	/// @param device - the device to build against
	/// @param format - the color format the composite writes, which is the
	/// window's or the capture's
	/// @param width - the target's width in pixels
	/// @param height - its height
	pub(crate) fn new(
		device: &Device,
		format: TextureFormat,
		width: u32,
		height: u32,
	) -> Result<Self> {
		let (numbers_layout, texture_layout, eye_layout) = layouts(device);

		// clamped rather than repeating: every one of these passes reads a
		// picture, and a tap that ran off the edge and came back on the other
		// side would fold the left of the screen into the right.
		let sampler = device.create_sampler(&SamplerDescriptor {
			label: Some("post"),
			address_mode_u: AddressMode::ClampToEdge,
			address_mode_v: AddressMode::ClampToEdge,
			address_mode_w: AddressMode::ClampToEdge,
			mag_filter: FilterMode::Linear,
			min_filter: FilterMode::Linear,
			..SamplerDescriptor::default()
		});

		let tuning = device.create_buffer(&BufferDescriptor {
			label: Some("post tuning"),
			size: u64::try_from(size_of::<Tuning>())
				.map_err(|_| err!(Graphics("the tuning block does not fit a buffer")))?,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let numbers = device.create_bind_group(&BindGroupDescriptor {
			label: Some("post numbers"),
			layout: &numbers_layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: tuning.as_entire_binding(),
			}],
		});

		let source = include_str!("post.wgsl");
		let scope = device.push_error_scope(ErrorFilter::Validation);
		let module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("post"),
			source: ShaderSource::Wgsl(source.into()),
		});

		let built =
			pipelines(device, &module, format, [&numbers_layout, &texture_layout, &eye_layout]);

		if let Some(complaint) = pollster::block_on(scope.pop()) {
			return Err(err!(Graphics("the post-processing pipelines: {complaint}")));
		}

		let eyes = [
			one_texel(device, &eye_layout, "post eye 0"),
			one_texel(device, &eye_layout, "post eye 1"),
		];
		let rungs = ladder(device, &sampler, &texture_layout);
		let (target, target_read) = colors(device, &sampler, &texture_layout, width, height);
		// none until somebody asks: a fresh surface is built before the world
		// it will draw exists, so it cannot know what the console says yet.
		// @ref `set_samples`, which the scene calls every frame.
		let multi = None;
		let glow = chain(device, &sampler, &texture_layout, width, height);
		let glow_read = widest(device, &sampler, &texture_layout, &glow, &target);

		Ok(Self {
			target,
			target_read,
			multi,
			samples: NO_SAMPLES,
			rungs,
			glow,
			glow_read,
			eyes,
			eye: 0,
			adapted: false,
			tuning,
			numbers,
			sampler,
			texture_layout,
			luminance: built.0,
			halve: built.1,
			adapt: built.2,
			threshold: built.3,
			down: built.4,
			up: built.5,
			composite: built.6,
			size: (width, height),
		})
	}

	/// What the scene's passes draw into.
	///
	/// The multisampled target when there is one, and the plain one when there
	/// is not. Everything downstream reads the *plain* one either way, which is
	/// the whole reason the resolve happens at the end of the scene's own pass
	/// rather than in a pass of its own: the chain, the composite, the overlay
	/// and the interface never learn that any of this happened.
	pub(crate) const fn target(&self) -> &TextureView {
		match &self.multi {
			| Some(view) => view,
			| None => &self.target,
		}
	}

	/// Where the scene's pass resolves to, or nothing when it is not
	/// multisampled.
	pub(crate) const fn resolve_into(&self) -> Option<&TextureView> {
		match &self.multi {
			| Some(_) => Some(&self.target),
			| None => None,
		}
	}

	/// Draws the scene with this many samples a pixel from now on.
	///
	/// Nothing but the one extra texture: the plain target it resolves into is
	/// the same texture the chain has always read, so a change here costs no
	/// bind group and no pass. **The pipelines are the caller's** - a pipeline
	/// records a sample count of its own and wgpu refuses a pass whose
	/// attachments disagree with it.
	///
	/// @param device - the device to build against
	/// @param samples - [`SAMPLES`] or [`NO_SAMPLES`]
	pub(crate) fn set_samples(&mut self, device: &Device, samples: u32) {
		if self.samples == samples {
			return;
		}

		self.samples = samples;
		self.multi = multisampled(device, samples, self.size.0, self.size.1);
	}

	/// Rebuilds the target for a new size.
	///
	/// The ladder is a fixed size and the eye is one texel, so neither moves;
	/// what the eye holds is deliberately kept, because a window being dragged
	/// wider is the same room.
	pub(crate) fn resize(&mut self, device: &Device, width: u32, height: u32) {
		if self.size == (width, height) {
			return;
		}

		let (target, read) = colors(device, &self.sampler, &self.texture_layout, width, height);

		self.target = target;
		self.target_read = read;
		self.multi = multisampled(device, self.samples, width, height);
		self.glow = chain(device, &self.sampler, &self.texture_layout, width, height);
		self.glow_read =
			widest(device, &self.sampler, &self.texture_layout, &self.glow, &self.target);
		self.size = (width, height);
	}

	/// Measures the picture, moves the eye, and writes the frame out.
	///
	/// @param encoder - the frame's encoder, with the scene already recorded
	/// @param queue - where the tuning block is written
	/// @param post - what the world asks for
	/// @param seconds - how long this frame was, for the eye
	/// @param out - the window's or the capture's own view
	/// @param timings - what to write this frame's marks into, which is
	/// nothing at all until somebody has asked to be told
	pub(crate) fn resolve(
		&mut self,
		encoder: &mut CommandEncoder,
		queue: &Queue,
		post: Post,
		seconds: f32,
		out: &TextureView,
		timings: &Timings,
	) {
		// the eye is measured first and read by the composite in the same
		// encoder, so the tuning block has to describe the frame that is about
		// to happen rather than the one that did.
		let moving = if self.adapted { post.adapt(seconds) } else { 1.0 };

		queue.write_buffer(&self.tuning, 0, bytemuck::bytes_of(&tuning_of(post, moving)));

		if post.auto_exposure {
			self.measure(encoder, timings);
		}

		if post.is_blooming() {
			self.gather(encoder, timings);
		}

		let composite = timings.writes(Pass::Composite, Ends::Both);
		let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
			label: Some("post composite"),
			color_attachments: &[Some(RenderPassColorAttachment {
				view: out,
				depth_slice: None,
				resolve_target: None,
				ops: Operations {
					// the whole target is written, so nothing is loaded and
					// nothing is cleared. What was in the window is gone
					// either way.
					load: LoadOp::Clear(wgpu::Color::BLACK),
					store: StoreOp::Store,
				},
			})],
			depth_stencil_attachment: None,
			timestamp_writes: composite,
			occlusion_query_set: None,
			multiview_mask: None,
		});

		pass.set_pipeline(&self.composite);
		pass.set_bind_group(0, &self.numbers, &[]);
		pass.set_bind_group(1, &self.target_read, &[]);
		pass.set_bind_group(2, &self.eyes[self.eye].read, &[]);
		pass.set_bind_group(3, &self.glow_read, &[]);
		pass.draw(0..3, 0..1);
	}

	/// The bloom chain: what is bright, spread wide.
	///
	/// Down the chain thresholding once and halving after, then back up
	/// adding each rung into the one under it. The widest rung is what the
	/// composite reads, and it holds every rung above it by the time this
	/// returns.
	fn gather(&self, encoder: &mut CommandEncoder, timings: &Timings) {
		// the span opens on the first pass down and closes on the last pass
		// up, so a chain of one rung - a target too small to halve twice - is
		// both ends of its own span rather than an opening nothing closes.
		let alone = self.glow.len() < 2;

		for (step, rung) in self.glow.iter().enumerate() {
			let (pipeline, source) = if step == 0 {
				(&self.threshold, &self.target_read)
			} else {
				(&self.down, &self.glow[step - 1].read)
			};
			let marks = match (step, alone) {
				| (0, true) => timings.writes(Pass::Glow, Ends::Both),
				| (0, false) => timings.writes(Pass::Glow, Ends::Open),
				| _ => timings.writes(Pass::Glow, Ends::Middle),
			};

			screen_pass(encoder, "post glow down", &rung.view, marks, |pass| {
				pass.set_pipeline(pipeline);
				pass.set_bind_group(0, &self.numbers, &[]);
				pass.set_bind_group(1, source, &[]);
			});
		}

		// and back, narrowest first, each into the wider one under it. The
		// pass loads rather than clears, because what is already in a rung is
		// its own half of the answer.
		for step in (1..self.glow.len()).rev() {
			let source = &self.glow[step].read;
			// the widest rung is written last, so that is where the span ends
			let marks =
				timings.writes(Pass::Glow, if step == 1 { Ends::Close } else { Ends::Middle });

			over_pass(encoder, "post glow up", &self.glow[step - 1].view, marks, |pass| {
				pass.set_pipeline(&self.up);
				pass.set_bind_group(1, source, &[]);
			});
		}
	}

	/// The ladder and the eye, for a frame that measures.
	fn measure(&mut self, encoder: &mut CommandEncoder, timings: &Timings) {
		for (step, rung) in self.rungs.iter().enumerate() {
			let (pipeline, source) = if step == 0 {
				(&self.luminance, &self.target_read)
			} else {
				(&self.halve, &self.rungs[step - 1].read)
			};
			let marks =
				timings.writes(Pass::Meter, if step == 0 { Ends::Open } else { Ends::Middle });

			screen_pass(encoder, "post meter", &rung.view, marks, |pass| {
				pass.set_pipeline(pipeline);
				pass.set_bind_group(1, source, &[]);
			});
		}

		let Some(bottom) = self.rungs.last() else {
			return;
		};

		let before = self.eye;
		let after = 1 - before;
		// the eye moving is the last thing the meter does, so the span ends
		// here rather than at the bottom of the ladder
		let marks = timings.writes(Pass::Meter, Ends::Close);

		screen_pass(encoder, "post adapt", &self.eyes[after].view, marks, |pass| {
			pass.set_pipeline(&self.adapt);
			pass.set_bind_group(0, &self.numbers, &[]);
			pass.set_bind_group(1, &bottom.read, &[]);
			pass.set_bind_group(2, &self.eyes[before].read, &[]);
		});

		self.eye = after;
		self.adapted = true;
	}
}

/// The seven pipelines, all against one shader module.
///
/// A tuple rather than a struct because the caller unpacks it into fields
/// straight away and a struct of seven pipelines would be named twice for
/// nothing. In the order the frame runs them.
fn pipelines(
	device: &Device,
	module: &wgpu::ShaderModule,
	format: TextureFormat,
	[numbers, texture, eye]: [&BindGroupLayout; 3],
) -> (
	RenderPipeline,
	RenderPipeline,
	RenderPipeline,
	RenderPipeline,
	RenderPipeline,
	RenderPipeline,
	RenderPipeline,
) {
	// each pipeline declares exactly the groups its entry point reads, and
	// the holes are why the list is of options: a pipeline layout is what a
	// pipeline *requires*, so declaring a group it ignores would mean the
	// pass could not draw until something irrelevant had been bound.
	let luminance =
		screen_pipeline(device, module, "post luminance", "fragment_luminance", METER_FORMAT, &[
			None,
			Some(texture),
		]);
	let halve = screen_pipeline(device, module, "post halve", "fragment_halve", METER_FORMAT, &[
		None,
		Some(texture),
	]);
	let adapt = screen_pipeline(device, module, "post adapt", "fragment_adapt", METER_FORMAT, &[
		Some(numbers),
		Some(texture),
		Some(eye),
	]);
	let threshold =
		screen_pipeline(device, module, "post threshold", "fragment_threshold", HDR_FORMAT, &[
			Some(numbers),
			Some(texture),
		]);
	let down = screen_pipeline(device, module, "post down", "fragment_down", HDR_FORMAT, &[
		None,
		Some(texture),
	]);
	// the one pass here that blends rather than replaces: a rung is written
	// on the way up over what it already held on the way down, so the
	// widest one ends up carrying everything above it.
	let up = adding_pipeline(device, module, "post up", "fragment_up", HDR_FORMAT, &[
		None,
		Some(texture),
	]);
	let composite =
		screen_pipeline(device, module, "post composite", "fragment_composite", format, &[
			Some(numbers),
			Some(texture),
			Some(eye),
			Some(texture),
		]);

	(luminance, halve, adapt, threshold, down, up, composite)
}

/// The three ways a pass here is handed something: the numbers, a texture
/// with a sampler beside it, and the one texel the eye is kept in.
///
/// The eye's has no sampler in it, because the two passes that read it read
/// one texel with `textureLoad` and a load needs none.
fn layouts(device: &Device) -> (BindGroupLayout, BindGroupLayout, BindGroupLayout) {
	let numbers = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("post numbers"),
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
		label: Some("post source"),
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
	let eye = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("post eye"),
		entries: &[BindGroupLayoutEntry {
			binding: 0,
			visibility: ShaderStages::FRAGMENT,
			ty: BindingType::Texture {
				sample_type: TextureSampleType::Float { filterable: true },
				view_dimension: TextureViewDimension::D2,
				multisampled: false,
			},
			count: None,
		}],
	});

	(numbers, texture, eye)
}

/// The tuning block for one frame.
fn tuning_of(post: Post, moving: f32) -> Tuning {
	Tuning {
		curve: [
			// as a float because the whole block is floats and a word for
			// the sake of one index would put padding in it.
			index_of(post.tonemap),
			post.white,
			post.exposure,
			if post.auto_exposure { 1.0 } else { 0.0 },
		],
		meter: [
			// the key with the bias in stops already folded in, so that
			// the shader's line is the divide and the clamp and nothing
			// else. @ref `Post::metered`, which is the same arithmetic and
			// is where it is tested.
			colby_core::abi::post::MIDDLE_GREY * post.exposure_bias.exp2(),
			post.exposure_min.min(post.exposure_max),
			post.exposure_max,
			moving,
		],
		bloom: [
			post.bloom.max(0.0),
			post.bloom_threshold.max(0.0),
			// half the threshold, which is the band the soft edge spans. A
			// number of its own would be a thirteenth field on a record that
			// already has twelve, for a knob nobody has asked to turn.
			post.bloom_threshold.max(0.0) * 0.5,
			0.0,
		],
	}
}

/// A curve's place in its word list, as a float.
fn index_of(curve: ToneMap) -> f32 {
	// through u16 rather than by an `as`: three words will not reach the place
	// where an integer stops being exact in a float, and saying so costs a
	// conversion the compiler removes.
	u16::try_from(curve.index()).map_or(0.0, f32::from)
}

/// The world's target and the way to read it.
/// The multisampled target the scene draws into, when it is multisampled.
///
/// No `TEXTURE_BINDING`: nothing samples it and nothing may - a multisampled
/// texture is read with a different binding type and a different shader, and
/// the only thing that ever touches this one is the resolve at the end of the
/// pass that wrote it.
///
/// @param samples - how many a pixel is drawn with; one is no target at all
/// @return the view, or `None` when a pixel is one sample
fn multisampled(device: &Device, samples: u32, width: u32, height: u32) -> Option<TextureView> {
	if samples <= 1 {
		return None;
	}

	let texture = device.create_texture(&TextureDescriptor {
		label: Some("hdr multisampled"),
		size: Extent3d {
			width: width.max(1),
			height: height.max(1),
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		sample_count: samples,
		dimension: TextureDimension::D2,
		format: HDR_FORMAT,
		usage: TextureUsages::RENDER_ATTACHMENT,
		view_formats: &[],
	});

	Some(texture.create_view(&TextureViewDescriptor::default()))
}

fn colors(
	device: &Device,
	sampler: &Sampler,
	layout: &BindGroupLayout,
	width: u32,
	height: u32,
) -> (TextureView, BindGroup) {
	let texture = device.create_texture(&TextureDescriptor {
		label: Some("hdr"),
		size: Extent3d {
			width: width.max(1),
			height: height.max(1),
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		sample_count: 1,
		dimension: TextureDimension::D2,
		format: HDR_FORMAT,
		usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
		view_formats: &[],
	});
	let view = texture.create_view(&TextureViewDescriptor::default());
	let read = sampled(device, sampler, layout, &view, "hdr");

	(view, read)
}

/// The bloom chain: half the target, halving until it is too small to be worth
/// another pass.
///
/// Six at most and none at all below sixteen pixels. The widest rung is half
/// the target rather than the whole of it, which halves the bandwidth of every
/// pass and costs nothing anybody can see: a glow is by definition the part of
/// a picture with no detail in it.
fn chain(
	device: &Device,
	sampler: &Sampler,
	layout: &BindGroupLayout,
	width: u32,
	height: u32,
) -> Vec<Rung> {
	let mut rungs = Vec::new();
	let (mut wide, mut tall) = (width / 2, height / 2);

	while rungs.len() < GLOW_RUNGS && wide.min(tall) >= GLOW_SMALLEST {
		let texture = device.create_texture(&TextureDescriptor {
			label: Some("post glow"),
			size: Extent3d {
				width: wide,
				height: tall,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: HDR_FORMAT,
			usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
			view_formats: &[],
		});
		let view = texture.create_view(&TextureViewDescriptor::default());
		let read = sampled(device, sampler, layout, &view, "post glow");

		rungs.push(Rung { view, read });
		wide /= 2;
		tall /= 2;
	}

	rungs
}

/// The way the composite reads the chain, or the target when there is none.
///
/// A pipeline requires every group its layout names, so the composite has to
/// be handed *something* at group three even in a frame with no bloom in it.
/// The target itself is the honest something: it is the right size and the
/// right format, and nothing samples it, because the shader skips the tap when
/// the intensity is nought.
fn widest(
	device: &Device,
	sampler: &Sampler,
	layout: &BindGroupLayout,
	glow: &[Rung],
	target: &TextureView,
) -> BindGroup {
	glow.first().map_or_else(
		|| sampled(device, sampler, layout, target, "post glow stand-in"),
		|rung| sampled(device, sampler, layout, &rung.view, "post glow"),
	)
}

/// Every rung, widest first, ending at one texel.
fn ladder(device: &Device, sampler: &Sampler, layout: &BindGroupLayout) -> Vec<Rung> {
	let mut rungs = Vec::new();
	let mut side = METER_SIZE;

	loop {
		let texture = device.create_texture(&TextureDescriptor {
			label: Some("post meter"),
			size: Extent3d {
				width: side,
				height: side,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: METER_FORMAT,
			usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
			view_formats: &[],
		});
		let view = texture.create_view(&TextureViewDescriptor::default());
		let read = sampled(device, sampler, layout, &view, "post meter");

		rungs.push(Rung { view, read });

		if side == 1 {
			return rungs;
		}

		side /= 2;
	}
}

/// One texel of half float, and the way to read it.
fn one_texel(device: &Device, layout: &BindGroupLayout, label: &str) -> Rung {
	let texture = device.create_texture(&TextureDescriptor {
		label: Some(label),
		size: Extent3d {
			width: 1,
			height: 1,
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		sample_count: 1,
		dimension: TextureDimension::D2,
		format: METER_FORMAT,
		usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
		view_formats: &[],
	});
	let view = texture.create_view(&TextureViewDescriptor::default());
	// the eye's layout is the texture alone, so the sampler is not in it; the
	// shader reads it with `textureLoad`, which needs none.
	let read = device.create_bind_group(&BindGroupDescriptor {
		label: Some(label),
		layout,
		entries: &[BindGroupEntry {
			binding: 0,
			resource: BindingResource::TextureView(&view),
		}],
	});

	Rung { view, read }
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

/// One pass over one triangle covering its target.
///
/// @param marks - which end of a timed span this pass carries, if any
fn screen_pass<F>(
	encoder: &mut CommandEncoder,
	label: &str,
	view: &TextureView,
	marks: Option<RenderPassTimestampWrites<'_>>,
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
				load: LoadOp::Clear(wgpu::Color::BLACK),
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

/// One pass over a target that keeps what is already in it.
///
/// The upsample's, and the only one here that loads: a rung on the way up is
/// written over its own half of the answer rather than instead of it.
fn over_pass<F>(
	encoder: &mut CommandEncoder,
	label: &str,
	view: &TextureView,
	marks: Option<RenderPassTimestampWrites<'_>>,
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
				load: LoadOp::Load,
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

/// The same as [`screen_pipeline`], with the fragment added to the target
/// rather than replacing it.
fn adding_pipeline(
	device: &Device,
	module: &wgpu::ShaderModule,
	label: &str,
	entry: &str,
	format: TextureFormat,
	groups: &[Option<&BindGroupLayout>],
) -> RenderPipeline {
	built(
		device,
		module,
		label,
		entry,
		format,
		groups,
		Some(BlendState {
			color: BlendComponent {
				src_factor: BlendFactor::One,
				dst_factor: BlendFactor::One,
				operation: BlendOperation::Add,
			},
			alpha: BlendComponent::REPLACE,
		}),
	)
}

/// One pipeline over the full-screen triangle, with no vertex buffers and no
/// depth.
fn screen_pipeline(
	device: &Device,
	module: &wgpu::ShaderModule,
	label: &str,
	entry: &str,
	format: TextureFormat,
	groups: &[Option<&BindGroupLayout>],
) -> RenderPipeline {
	built(device, module, label, entry, format, groups, Some(BlendState::REPLACE))
}

/// Everything both of them do, with the blend as the one difference.
fn built(
	device: &Device,
	module: &wgpu::ShaderModule,
	label: &str,
	entry: &str,
	format: TextureFormat,
	groups: &[Option<&BindGroupLayout>],
	blend: Option<BlendState>,
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
				format,
				blend,
				write_mask: ColorWrites::ALL,
			})],
		}),
		multiview_mask: None,
		cache: None,
	})
}
