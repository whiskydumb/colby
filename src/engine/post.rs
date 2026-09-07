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
	ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, Texture, TextureDescriptor,
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
/// not depend on how big somebody's screen is. **Sixty-four rather than a
/// hundred and twenty-eight since `PERF-2`**: the ladder reduces by
/// [`METER_STEP`] a rung now, so the number that matters is how many powers of
/// eight there are between this and one, and sixty-four is two of them.
///
/// It has to be a power of [`METER_STEP`] or the last rung is not the step's
/// own size and the eye's reduction stops being exact. A test says so.
const METER_SIZE: u32 = 64;

/// How much narrower each rung of the ladder is than the one before it.
///
/// **Eight, and the number was measured rather than chosen.** With halving,
/// the ladder was eight rungs and an eye - nine passes of one tap each - and it
/// cost about fifty-three microseconds on an RX 9060 XT at seven-twenty, of
/// which five to seven belonged to every pass whatever was in it. Sweeping
/// [`METER_SIZE`] showed the cost was very nearly linear in the *number of
/// passes* and barely moved with the work inside them, which is the shape of a
/// barrier rather than of arithmetic. So the fix is fewer passes with more in
/// each, and eight is what Godot reduces by for the same reason
/// (`luminance.cpp:91-93`).
///
/// Eight and not more because sixteen bilinear taps cover exactly an
/// eight-by-eight block: each tap is a two-by-two average and sixteen of them
/// tile sixty-four texels. Reducing by sixteen would need sixty-four taps to
/// stay exact, or would start guessing.
const METER_STEP: u32 = 8;

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

	/// The texture behind the view, kept only so that a test can copy the
	/// eye's one texel back and read what the meter decided. @ref
	/// [`Chain::measured`].
	#[cfg(test)]
	texture: Texture,
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

	/// The picture itself, behind [`target`](Self::target).
	///
	/// Kept only so that a test can write an exact picture into it and ask the
	/// meter what it makes of it. @ref [`Chain::paint`], and the test that uses
	/// it for why the property is not reachable through a rendered scene.
	#[cfg(test)]
	hdr: Texture,

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
	reduce: RenderPipeline,
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
		let colors = colors(device, &sampler, &texture_layout, width, height);
		let (target, target_read) = (colors.1, colors.2);
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
			#[cfg(test)]
			hdr: colors.0,
			adapted: false,
			tuning,
			numbers,
			sampler,
			texture_layout,
			luminance: built.0,
			reduce: built.1,
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

		let colors = colors(device, &self.sampler, &self.texture_layout, width, height);

		#[cfg(test)]
		{
			self.hdr = colors.0;
		};
		self.target = colors.1;
		self.target_read = colors.2;
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
				(&self.reduce, &self.rungs[step - 1].read)
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
		// here rather than at the bottom of the ladder. It is also the ladder's
		// last reduction - the rung it reads is `METER_STEP` texels across and
		// this target is one - which is what took the pass count from four to
		// three. @ref `fragment_adapt`.
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

	/// Writes an exact grey picture into the target, for a test.
	///
	/// **The only way to ask the meter about a picture nobody can render.**
	/// What the first pass does wrong is a resonance between where its taps
	/// land and how often the content repeats, and at seven-twenty it is sharp
	/// enough that a tenth of a pixel either side of the period hides it
	/// entirely - so a rendered wall would have to have its projected stripe
	/// land on five pixels exactly, through a perspective camera, a field of
	/// view and a resolve. Painting the picture instead removes the geometry,
	/// the camera, the samples and the curve, and leaves the pass under test.
	///
	/// @param queue - where the upload goes
	/// @param levels - one luminance per pixel, row by row, as long as the
	/// target is wide times high
	#[cfg(test)]
	fn paint(&self, queue: &Queue, levels: &[f32]) {
		let (width, height) = self.size;
		let mut bytes = Vec::with_capacity(levels.len() * 8);

		// grey, so that whatever weights `luminance` uses in the shader the
		// answer is the level itself: the three of them sum to one.
		for level in levels {
			let half = half_from(*level).to_le_bytes();

			for _ in 0..3 {
				bytes.extend_from_slice(&half);
			}

			bytes.extend_from_slice(&half_from(1.0).to_le_bytes());
		}

		queue.write_texture(
			wgpu::TexelCopyTextureInfo {
				texture: &self.hdr,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			&bytes,
			wgpu::TexelCopyBufferLayout {
				offset: 0,
				bytes_per_row: Some(width * 8),
				rows_per_image: Some(height),
			},
			Extent3d { width, height, depth_or_array_layers: 1 },
		);
	}

	/// What the eye holds, in stops.
	///
	/// The one texel is `mix(before, measured, ..)` of the *log* of the
	/// luminance, and on a chain that has never measured anything the blend is
	/// one - so a single frame's answer is the measurement itself and not a
	/// step towards it. @ref `fragment_adapt`.
	///
	/// @param device - to build the staging buffer on
	/// @param queue - to submit the copy and to poll
	/// @return the mean of the log base two of the picture's luminance
	#[cfg(test)]
	fn measured(&self, device: &Device, queue: &Queue) -> f32 {
		// one texel of two bytes, padded to the row alignment a texture copy
		// insists on
		let staging = device.create_buffer(&BufferDescriptor {
			label: Some("post eye readback"),
			size: 256,
			usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
			mapped_at_creation: false,
		});
		let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("post eye readback"),
		});

		encoder.copy_texture_to_buffer(
			wgpu::TexelCopyTextureInfo {
				texture: &self.eyes[self.eye].texture,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			wgpu::TexelCopyBufferInfo {
				buffer: &staging,
				layout: wgpu::TexelCopyBufferLayout {
					offset: 0,
					bytes_per_row: Some(256),
					rows_per_image: Some(1),
				},
			},
			Extent3d {
				width: 1,
				height: 1,
				depth_or_array_layers: 1,
			},
		);
		queue.submit([encoder.finish()]);

		let slice = staging.slice(..);
		slice.map_async(wgpu::MapMode::Read, |_| {});
		device
			.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
			.expect("the readback completes");

		let view = slice
			.get_mapped_range()
			.expect("the readback maps");
		let bits = u16::from_le_bytes([view[0], view[1]]);

		drop(view);
		staging.unmap();

		half_into(bits)
	}
}

/// A float as the sixteen bits [`HDR_FORMAT`] and [`METER_FORMAT`] hold.
///
/// Written out because `f16` is still unstable and a dependency for twenty
/// lines of bit arithmetic is not a trade this workspace makes. Only a test
/// uses it, and only on values a test chose.
#[cfg(test)]
fn half_from(value: f32) -> u16 {
	let bits = value.to_bits();
	let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
	let exponent = i32::try_from((bits >> 23) & 0xFF).unwrap_or(0) - 127;
	let mantissa = bits & 0x007F_FFFF;

	if exponent > 15 {
		// past what a half holds, which is infinity with a full exponent
		return sign | 0x7C00;
	}

	if exponent < -14 {
		// subnormal, or nought
		let shift = u32::try_from(-14 - exponent).unwrap_or(24) + 14;
		let scaled = if shift < 32 {
			(mantissa | 0x0080_0000) >> shift
		} else {
			0
		};

		return sign | u16::try_from(scaled).unwrap_or(0);
	}

	let biased = u16::try_from(exponent + 15).unwrap_or(0) << 10;

	sign | biased | u16::try_from(mantissa >> 13).unwrap_or(0)
}

/// The other way: sixteen bits back into a float.
#[cfg(test)]
fn half_into(bits: u16) -> f32 {
	let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
	let exponent = i32::from((bits >> 10) & 0x1F);
	let mantissa = f32::from(bits & 0x03FF);

	if exponent == 0 {
		return sign * mantissa * 2.0_f32.powi(-24);
	}

	if exponent == 31 {
		return sign * f32::INFINITY;
	}

	sign * (1.0 + mantissa / 1024.0) * 2.0_f32.powi(exponent - 15)
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
	let reduce =
		screen_pipeline(device, module, "post reduce", "fragment_reduce", METER_FORMAT, &[
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

	(luminance, reduce, adapt, threshold, down, up, composite)
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
) -> (Texture, TextureView, BindGroup) {
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
		// **the copy bit is a test's and only a test's.** A picture painted
		// from the outside is the only way to ask the meter about content
		// nobody can render exactly, and `cfg!` is what keeps the shipping
		// texture's usage the two flags a frame actually needs.
		usage: TextureUsages::RENDER_ATTACHMENT
			| TextureUsages::TEXTURE_BINDING
			| if cfg!(test) {
				TextureUsages::COPY_DST
			} else {
				TextureUsages::empty()
			},
		view_formats: &[],
	});
	let view = texture.create_view(&TextureViewDescriptor::default());
	let read = sampled(device, sampler, layout, &view, "hdr");

	(texture, view, read)
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

		rungs.push(Rung {
			view,
			read,
			#[cfg(test)]
			texture,
		});
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

/// Every rung, widest first, ending at [`METER_STEP`] texels across.
///
/// **It stops one reduction short of a single texel on purpose.** The eye's own
/// pass reduces as well as blending, so the rung it reads is the step's own
/// size and the ladder has no reason to produce a one-texel rung nothing would
/// sample. At [`METER_SIZE`] of sixty-four that is two rungs, sixty-four and
/// eight, and three passes counting the eye.
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

		rungs.push(Rung {
			view,
			read,
			#[cfg(test)]
			texture,
		});

		if side <= METER_STEP {
			return rungs;
		}

		side /= METER_STEP;
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
		// as `colors`: the copy bit is a test's, so that the one texel the eye
		// holds can be read back and compared with arithmetic.
		usage: TextureUsages::RENDER_ATTACHMENT
			| TextureUsages::TEXTURE_BINDING
			| if cfg!(test) {
				TextureUsages::COPY_SRC
			} else {
				TextureUsages::empty()
			},
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

	Rung {
		view,
		read,
		#[cfg(test)]
		texture,
	}
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

#[cfg(test)]
mod tests {
	use super::*;

	/// How many rungs [`ladder`] builds at the constants in hand, without a
	/// device to build them on.
	fn rungs() -> u32 {
		let mut side = METER_SIZE;
		let mut count = 1;

		while side > METER_STEP {
			side /= METER_STEP;
			count += 1;
		}

		count
	}

	/// Where the ladder stops.
	fn bottom() -> u32 {
		let mut side = METER_SIZE;

		while side > METER_STEP {
			side /= METER_STEP;
		}

		side
	}

	/// How far the meter may be from the truth, in stops.
	///
	/// **This number is the debt, and it has moved once already.** `PERF-8`
	/// said the meter's accuracy had no test and that one had been tried and
	/// thrown away. The test landed at nine tenths of a stop, which was what
	/// the meter then earned: every tap of the first pass was read wherever the
	/// jitter had put it, so it came back through the linear sampler as a blend
	/// of two texels, and the log of a blend of two brightnesses is above the
	/// blend of their logs - 0.83 stops of it on stripes two pixels wide.
	/// Snapping each tap to a texel took the worst of the eight pictures below
	/// to **0.18**, and this came with it.
	///
	/// A quarter of a stop is about nineteen percent and is still not tight.
	/// What is left is four samples standing for two hundred pixels, and a hash
	/// nobody has looked at the distribution of. It is tight enough to fail
	/// both of the samplings this one replaced: sixteen taps on a regular grid
	/// go **1.6 stops** out where the content resonates with them, and the
	/// unsnapped jitter 0.83 where it does not.
	const LIMIT: f64 = 0.25;

	/// The picture the meter is asked about, which is the size a frame is.
	///
	/// **Seven-twenty and not the 320 by 240 the pixel tests use**, because the
	/// thing being measured is a resonance between how far apart the first
	/// pass puts its taps and how often the content repeats, and both scale
	/// with the picture. One texel of a sixty-four square meter covers twenty
	/// pixels of this, and the taps of a regular grid would sit five apart at
	/// pixels 2, 7, 12 and 17 of every one - so a stripe repeating every five
	/// pixels lands all four on the same phase, and the meter reads the wall
	/// **1.6 stops too dark**, measured. At 320 by 240 the same taps sit one
	/// and a quarter pixels apart and no stripe anybody can draw beats with
	/// them, which is why the test that was thrown away saw nothing.
	const WIDTH: usize = 1280;

	/// And its height.
	const HEIGHT: usize = 720;

	/// A device, a chain at the size a frame is, and somewhere to composite to.
	///
	/// `None` where the machine has no adapter, which is how every rendered
	/// test in this workspace skips rather than fails.
	fn metering() -> Option<(crate::Gpu, Chain, TextureView)> {
		let gpu = match crate::Gpu::open(crate::gpu::backends(None), None) {
			| Ok(Some(gpu)) => gpu,
			| Ok(None) => return None,
			| Err(error) => panic!("opening the device failed: {error}"),
		};
		let format = TextureFormat::Rgba8UnormSrgb;
		let width = u32::try_from(WIDTH).unwrap_or(1280);
		let height = u32::try_from(HEIGHT).unwrap_or(720);
		let chain = Chain::new(gpu.device(), format, width, height).expect("the chain builds");
		let out = gpu
			.device()
			.create_texture(&TextureDescriptor {
				label: Some("post test target"),
				size: Extent3d { width, height, depth_or_array_layers: 1 },
				mip_level_count: 1,
				sample_count: 1,
				dimension: TextureDimension::D2,
				format,
				usage: TextureUsages::RENDER_ATTACHMENT,
				view_formats: &[],
			})
			.create_view(&TextureViewDescriptor::default());

		Some((gpu, chain, out))
	}

	/// Vertical stripes of two luminances, half and half, at a pixel period.
	///
	/// A period of nought is a flat picture of the first level, which is the
	/// control: a meter that cannot read a wall of one brightness is broken in
	/// a way no sampling argument explains.
	fn striped(period: usize, bright: f32, dim: f32) -> Vec<f32> {
		let lit = |x: usize| period == 0 || x % period < period / 2;
		let row: Vec<f32> = (0..WIDTH)
			.map(|x| if lit(x) { bright } else { dim })
			.collect();
		let mut levels = Vec::with_capacity(WIDTH * HEIGHT);

		for _ in 0..HEIGHT {
			levels.extend_from_slice(&row);
		}

		levels
	}

	/// The mean of the log of every pixel: what the meter is trying to find.
	///
	/// Arithmetic over the whole picture rather than a second sampling of it,
	/// which is the half `PERF-5` could not have: its ground truth was the same
	/// pass at a bigger meter, so it shared whatever the pass itself got wrong.
	fn truth(levels: &[f32]) -> f64 {
		let total: f64 = levels
			.iter()
			.map(|level| f64::from(level.max(colby_core::abi::post::DARKEST).log2()))
			.sum();
		let count = f64::from(u32::try_from(levels.len()).unwrap_or(1));

		total / count
	}

	/// How many frames a picture is metered for before the eye is read.
	///
	/// **One would do on a fresh chain and does not on a used one**, which is
	/// the trap in driving this from a test: the first frame blends the whole
	/// way because nothing has been measured yet, and every frame after that
	/// blends by the rate - eighty-six percent of the way at this step - so a
	/// second picture metered on the same chain is thirteen percent of the
	/// first one. Eight frames leave a ten-millionth of it. @ref
	/// `fragment_adapt` and [`Post::adapt`].
	const FRAMES: usize = 8;

	/// Paints a picture, meters it until the eye has settled, and reads it.
	fn eye_on(gpu: &crate::Gpu, chain: &mut Chain, out: &TextureView, levels: &[f32]) -> f32 {
		chain.paint(gpu.queue(), levels);

		for _ in 0..FRAMES {
			let mut encoder =
				gpu.device()
					.create_command_encoder(&wgpu::CommandEncoderDescriptor {
						label: Some("post test meter"),
					});

			chain.resolve(&mut encoder, gpu.queue(), Post::DEFAULT, 1.0, out, &Timings::new(0.0));
			gpu.queue().submit([encoder.finish()]);
		}

		chain.measured(gpu.device(), gpu.queue())
	}

	#[test]
	fn the_meter_reads_a_picture_within_a_known_number_of_stops_of_the_truth() {
		// **the test `PERF-8` said could not be built, and why it can be.** The
		// one that was thrown away drew a striped wall through the camera and
		// saw nothing, because the failure is a resonance a tenth of a pixel
		// wide and a wall in perspective has a projected period that is some
		// real number nobody solved for. So no wall: the picture is painted
		// into the target the scene would have drawn into, and every pixel of
		// it is a number this test chose. No camera, no field of view, no
		// samples a pixel and no curve - the pass under test and nothing else.
		//
		// Flat is the control: nothing about where a tap lands can matter when
		// every pixel is the same. **Five is the resonant period** at this
		// size - a sixty-four square meter over 1280 makes one texel twenty
		// pixels and four regular taps five apart, so every tap lands on the
		// same stripe. Two is the worst case for a tap read through the linear
		// sampler, which is what the jitter costs; eight is the worst of what
		// is left once that is cured. Three, seven and twenty are ordinary.
		let Some((gpu, mut chain, out)) = metering() else {
			eprintln!("no GPU adapter; skipping the meter test");

			return;
		};

		for period in [0, 2, 3, 4, 5, 7, 8, 20] {
			let levels = striped(period, 4.0, 0.25);
			let want = truth(&levels);
			let got = f64::from(eye_on(&gpu, &mut chain, &out, &levels));
			let off = (got - want).abs();

			assert!(
				off <= LIMIT,
				"stripes every {period} pixels metered {got} where the picture is {want}, 				 \
				 which is {off} stops out and the limit is {LIMIT}"
			);
		}
	}

	#[test]
	fn a_flat_picture_is_metered_exactly_whatever_it_is_worth() {
		// the control on its own, tighter than the rest by a long way: nothing
		// about where a tap lands can matter when every pixel is the same, so
		// what is left is the half floats and the reductions. A failure here is
		// the apparatus and not the sampling.
		let Some((gpu, mut chain, out)) = metering() else {
			return;
		};

		for level in [0.25_f32, 1.0, 4.0] {
			let levels = striped(0, level, level);
			let got = eye_on(&gpu, &mut chain, &out, &levels);

			assert!(
				(f64::from(got) - f64::from(level.log2())).abs() < 0.01,
				"a flat picture at {level} metered {got} and should be {}",
				level.log2()
			);
		}
	}

	#[test]
	fn the_bottom_rung_is_exactly_the_step_so_the_eye_reduces_it_without_guessing() {
		// the whole of why the eye needs no pass of its own: it reads the last
		// rung and writes one texel, and sixteen bilinear taps are exactly an
		// eight-by-eight block. A bottom rung of any other size makes that
		// last step sixteen samples of a block rather than all of it, and
		// nothing would say so.
		assert_eq!(bottom(), METER_STEP, "the ladder does not land on the step");
	}

	#[test]
	fn the_size_is_a_power_of_the_step() {
		// the same statement said the other way round, because this is the one
		// anybody editing `METER_SIZE` will read first.
		let mut side = METER_SIZE;

		while side > METER_STEP {
			assert_eq!(side % METER_STEP, 0, "{side} is not a whole number of steps");
			side /= METER_STEP;
		}

		assert_eq!(side, METER_STEP);
	}

	#[test]
	fn the_meter_is_three_passes() {
		// **the number `PERF-2` was about.** It was nine - eight rungs and the
		// eye - and every one of them cost five to seven microseconds of
		// barrier whatever was inside it. This is the number `--profile`
		// prints as part of `passes=`, and the one thing in that table worth
		// comparing between two runs.
		assert_eq!(rungs(), 2, "sixty-four and eight");
		assert_eq!(rungs() + 1, 3, "and the eye, which reduces as well as blending");
	}

	#[test]
	fn the_shader_declares_the_entry_points_the_pipelines_ask_for() {
		// a pipeline naming an entry point that is not there fails when the
		// device builds it, which is a rendered test away rather than here -
		// and the rename from `fragment_halve` is exactly the edit that leaves
		// one behind.
		let source = include_str!("post.wgsl");

		for entry in [
			"fragment_luminance",
			"fragment_reduce",
			"fragment_adapt",
			"fragment_threshold",
			"fragment_down",
			"fragment_up",
			"fragment_composite",
		] {
			assert!(source.contains(&format!("fn {entry}(")), "{entry} is not in post.wgsl");
		}

		assert!(!source.contains("fragment_halve"), "the halving pass outlived its pipeline");
	}

	#[test]
	fn the_first_pass_scatters_its_taps_and_the_reductions_do_not() {
		// **the whole of what `PERF-5` changed, and the only thing in the tree
		// that guards it.** The first pass samples a picture two hundred times
		// larger than itself, so where its taps land has to be irregular: a
		// regular grid five pixels apart beats with content that repeats every
		// few pixels and read a striped wall seven percent too bright,
		// measured against a meter with full coverage. The reductions are the
		// opposite case - each is exact for a step of eight - and must stay on
		// their grid.
		//
		// **It pins the mechanism, and the property is pinned next door now.**
		// This note used to say a rendered test could not be made to show the
		// difference; that was true of a test that drew a *wall*, because the
		// failure is a resonance a tenth of a pixel wide and a wall in
		// perspective has no exact period. Painting the picture instead of
		// rendering it settles it - @ref
		// [`the_meter_reads_a_picture_within_a_known_number_of_stops_of_the_truth`].
		// This one earns its lines by running where there is no adapter at all,
		// which is where the other silently skips.
		let source = include_str!("post.wgsl");
		let luminance = source
			.split("fn fragment_luminance(")
			.nth(1)
			.expect("the pass is there");
		let reduce = source
			.split("fn reduced(")
			.nth(1)
			.expect("the reduction is there");

		assert!(
			luminance.contains("scatter("),
			"the first pass takes its taps off a regular grid again"
		);
		assert!(
			!reduce
				.split(
					"
}"
				)
				.next()
				.unwrap_or(reduce)
				.contains("scatter("),
			"a reduction that is exact has nothing to scatter"
		);
	}

	#[test]
	fn the_taps_and_the_step_agree_across_the_two_languages() {
		// `TAP_ROWS` in the shader is four because the step is eight: every
		// tap is a two-by-two average, so a row of four covers eight texels.
		// Two files, one number, and nothing but this checks it.
		let source = include_str!("post.wgsl");
		let rows = METER_STEP / 2;

		assert!(
			source.contains(&format!("const TAP_ROWS: i32 = {rows};")),
			"post.wgsl takes a different number of taps than a step of {METER_STEP} needs"
		);
	}
}
