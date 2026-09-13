//! How much of the light that reaches a surface by other routes each pixel can
//! still see.
//!
//! **An integral, not a count.** The share this works out is the one a surface
//! lit evenly from every direction actually receives: the hemisphere over the
//! pixel, each direction weighted by the cosine it arrives at, with every
//! direction that meets something within [`RADIUS`] taken away. That is what a
//! ray tracer would add up, and what the fixture's second answer does add up.
//! The older kind of estimate - scatter points through a ball and count how
//! many lie behind the depth - is the volume of a ball behind a surface, which
//! no ray tracer adds up, and it needs a heuristic at the edge of its own range
//! to look right.
//!
//! **Slices and their horizons.** Around each pixel a few lines across the
//! screen are followed outwards, each of them a plane through the eye; in each
//! the highest thing within reach on either side is the horizon, and the share
//! of a cosine-weighted half-circle above two horizons has a closed form. So
//! the only thing sampled is where the horizon is, and the integral over each
//! slice is exact. **What is hidden is divided by what the same slices add up
//! to with nothing in the way**, rather than by one: a few slices of an open
//! hemisphere do not add up to exactly one off the middle of the picture, and
//! this way an open surface is one to the last bit. Spreading the slices evenly
//! around the eye instead of around the middle of the screen was built and
//! measured and is no better once the share is a ratio.
//!
//! **Half the picture on each axis, over the picture's own buffers.** The
//! estimate runs once per four pixels and every tap it takes reads the pass
//! before the scene at full size, so no depth is ever read from a smaller copy
//! and nothing has to decide which of four depths a smaller copy keeps. A texel
//! stands for the pixel at twice its place.
//!
//! **No history, so no noise left over.** Each texel turns its slices by one of
//! nine rotations across a three by three tile, and the second pass averages
//! each texel with the three by three around it that lie on its surface - which
//! on a surface is every rotation exactly once. What is left is the error of
//! twenty-seven directions and eight taps a side, and not a pattern.
//!
//! **What it does not see**: anything off the picture, and the far side of
//! anything: the depth is a surface seen from the eye, so what is behind a thin
//! thing is taken for solid, and a pole darkens the floor behind it as if it
//! were a wall. Glass, particles and the debug lines are not in the buffers it
//! reads, so they hide nothing.
//!
//! **What it multiplies is the light that arrives by other routes, and nothing
//! a light sends.** The scene reads the buffer in the fragment stage of its
//! solid and its masked half and multiplies the one term that stands in for
//! everything the renderer does not simulate; a sun or a lamp is taken away by
//! its own shadow, and a pane of glass is not in the buffer, so it reads one. A
//! frame nobody asks for the estimate in binds one texel that says all of the
//! sky, which multiplies by exactly one.

use colby_core::{
	Result,
	abi::{Camera, World},
	bytemuck::{self, Pod, Zeroable},
	err,
};
use wgpu::{
	BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
	BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
	BufferDescriptor, BufferUsages, Color, ColorTargetState, ColorWrites, CommandEncoder, Device,
	ErrorFilter, Extent3d, FragmentState, LoadOp, MultisampleState, Operations, Origin3d,
	PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState, Queue, RenderPass,
	RenderPassColorAttachment, RenderPassDescriptor, RenderPassTimestampWrites, RenderPipeline,
	RenderPipelineDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
	StoreOp, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
	TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
	TextureViewDescriptor, TextureViewDimension, VertexState,
};

use crate::{
	prepass::{Prepass, Showing},
	scene::Viewport,
	shader::Shader,
	timing::{Ends, Pass, Timings},
};

/// The console variable that says how much of what the estimate finds hidden
/// the picture takes away.
///
/// **A strength, on, and not saved.** One takes away all of it and nought none
/// of it; a frame at nought records no pass and multiplies by one texel that
/// says all of the sky, so nought is the picture exactly as it was before any
/// of this existed, which is what the variable is for beside turning the cost
/// off. On for the shadows' reason: what it corrects is the light itself, the
/// ambient term reaching into corners it cannot reach, and what a machine that
/// cannot afford the correction needs is somewhere to say so.
pub const STRENGTH: &str = "r.occlusion";

/// What [`STRENGTH`] holds until somebody sets it: all of it.
pub const DEFAULT_STRENGTH: f32 = 1.0;

/// How far away something may be and still hide the sky from a surface, in
/// world units.
///
/// **A hard edge, and that is what makes the number mean something.** Every
/// engine in the field fades its range out by a curve of its own, which is why
/// their radii run from half a unit to three and cannot be compared: this one
/// is exactly "a direction that meets something nearer than this is hidden". It
/// does not show as an edge, because a surface's share of the sky arrives at
/// the whole of it with no slope: in front of a wall it is
/// `1 - (acos k - k sqrt(1 - k^2)) / pi` at `k` of the way out, whose slope at
/// the end is nought.
pub(crate) const RADIUS: f32 = 0.5;

/// The format both buffers are written in: r the share of the sky, g how far
/// along the view it was worked out.
///
/// Thirty-two bits because the distance is compared, twice: by the average,
/// against the plane of the texel being averaged, and by the scene, against its
/// own surface. Sixteen bits of distance are a centimeter apart at ten units.
pub(crate) const FORMAT: TextureFormat = TextureFormat::Rg32Float;

/// The numbers both passes read, laid out the way `occlusion.wgsl` declares
/// them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct Tuning {
	/// World space into view space, a row an axis.
	view_x: [f32; 4],
	view_y: [f32; 4],
	view_z: [f32; 4],

	/// `[the projection's x scale, its y scale, its z_axis.z, its w_axis.z]`.
	lens: [f32; 4],

	/// `[the picture's width, its height, the reach, the strength]`.
	size: [f32; 4],
}

/// What one frame asks the estimate for.
///
/// Worked out in [`Scene::upload`](crate::Scene) beside the other effects' own,
/// for their reason: it needs the camera the frame is drawn from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Asking {
	/// World space into view space, a row an axis.
	view: [[f32; 4]; 3],

	/// `[x scale, y scale, z_axis.z, w_axis.z]` of the projection.
	lens: [f32; 4],

	/// How much of what is hidden is taken away: one is all of it.
	strength: f32,
}

/// What this frame asks for, if anything.
///
/// The picture asks at the strength [`STRENGTH`] says, and not at all at
/// nought. The view that draws the buffer asks whatever that says and at the
/// whole strength, because what it draws is how much of the sky a surface sees
/// rather than how much of it the picture takes away - and the picture is not
/// on the screen while it is.
///
/// @param world - for the aspect and the console variable
/// @param camera - the camera this frame is drawn from
/// @param showing - what the view is asked to draw, if anything
#[must_use]
pub(crate) fn asking_of(
	world: &World,
	camera: &Camera,
	showing: Option<Showing>,
) -> Option<Asking> {
	let strength = if showing == Some(Showing::Occlusion) {
		1.0
	} else {
		strength_of(world)
	};

	if strength <= 0.0 {
		return None;
	}

	let view = camera.view();
	let lens = camera.projection(world.aspect);

	Some(Asking {
		view: [0, 1, 2].map(|axis| view.row(axis).to_array()),
		lens: [lens.x_axis.x, lens.y_axis.y, lens.z_axis.z, lens.w_axis.z],
		strength,
	})
}

/// How much of what is hidden the picture asks to take away.
///
/// @param world - for the console variable
fn strength_of(world: &World) -> f32 {
	held(
		world
			.cvars
			.float(STRENGTH)
			.unwrap_or(DEFAULT_STRENGTH),
	)
}

/// A strength as somebody asked for it, held inside nought and one.
///
/// A nan is the whole strength rather than none, the lamps' rule: a variable
/// nobody meant to set should not quietly change the light.
///
/// @param asked - what the variable holds
fn held(asked: f32) -> f32 {
	if asked.is_nan() {
		return DEFAULT_STRENGTH;
	}

	asked.clamp(0.0, 1.0)
}

/// The two half-sized buffers, and the way the second pass reads the first.
struct Buffers {
	/// Their size, half the picture's on each axis rounded up.
	size: (u32, u32),

	/// What the estimate writes.
	raw: TextureView,

	/// How the average reads it.
	raw_read: BindGroup,

	/// What the average writes, and what everything after reads.
	done: TextureView,
}

/// Both passes, and what they write into.
pub(crate) struct Occlusion {
	/// The size of the picture, which the buffers are half of.
	size: (u32, u32),

	/// Both buffers, made the first frame something asks and let go the first
	/// frame nothing does, the way the pass before the scene's are.
	buffers: Option<Buffers>,

	/// Which making of [`buffers`](Self::buffers) a reader is looking at.
	///
	/// **Moved when they are made as well as when they are let go**, where the
	/// pass before the scene's moves only when its targets go: that pass's one
	/// reader makes its group only while there are targets, and the scene's
	/// frame group is made whether there are buffers or not - over
	/// [`none`](Self::none) when there are not - so a pair of buffers arriving
	/// is a change that group has to hear about too.
	epoch: u64,

	/// One texel that says a surface sees all of the sky, which is what a
	/// reader binds in a frame nobody asked for the estimate in.
	none: TextureView,

	/// How both passes read what the pass before the scene wrote, and which of
	/// that pass's buffers the group is for. @ref [`Prepass::epoch`].
	reading: Option<(u64, BindGroup)>,

	tuning: Buffer,
	numbers: BindGroup,
	prepass_layout: BindGroupLayout,
	raw_layout: BindGroupLayout,
	estimate: RenderPipeline,
	blur: RenderPipeline,
	device: Device,
}

impl Occlusion {
	/// Builds both pipelines, the block they read their numbers from and the
	/// texel a frame that asks for nothing binds.
	///
	/// No buffer yet: a frame that asks for nothing never makes one.
	///
	/// @param device - the device to build against
	/// @param queue - where the texel that says all of the sky is written
	/// @param width - the picture's width in pixels
	/// @param height - its height
	pub(crate) fn new(device: &Device, queue: &Queue, width: u32, height: u32) -> Result<Self> {
		let numbers_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("occlusion numbers"),
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
		let prepass_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("occlusion prepass"),
			entries: &[crate::depth::entry(0), crate::prepass::entry(1)],
		});
		let raw_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("occlusion raw"),
			entries: &[entry(0)],
		});
		let tuning = device.create_buffer(&BufferDescriptor {
			label: Some("occlusion tuning"),
			size: u64::try_from(size_of::<Tuning>()).map_err(|_| {
				err!(Graphics("the occlusion tuning block does not fit a buffer"))
			})?,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let numbers = device.create_bind_group(&BindGroupDescriptor {
			label: Some("occlusion numbers"),
			layout: &numbers_layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: tuning.as_entire_binding(),
			}],
		});
		// read through the shader directory like the scene's own, once: a
		// variant of the file under `COLBY_SHADERS` is how the estimate is run
		// with more slices and taps than a frame can afford, against the same
		// build
		let source = Shader::new("occlusion.wgsl", include_str!("occlusion.wgsl"));
		let scope = device.push_error_scope(ErrorFilter::Validation);
		let module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("occlusion"),
			source: ShaderSource::Wgsl(source.source().into()),
		});
		let estimate = pipeline(device, &module, "occlusion estimate", "fragment_estimate", &[
			Some(&numbers_layout),
			Some(&prepass_layout),
		]);
		let blur = pipeline(device, &module, "occlusion blur", "fragment_blur", &[
			Some(&numbers_layout),
			Some(&prepass_layout),
			Some(&raw_layout),
		]);

		if let Some(complaint) = pollster::block_on(scope.pop()) {
			return Err(err!(Graphics("the occlusion pipelines: {complaint}")));
		}

		Ok(Self {
			size: (width, height),
			buffers: None,
			epoch: 0,
			none: whole_sky(device, queue),
			reading: None,
			tuning,
			numbers,
			prepass_layout,
			raw_layout,
			estimate,
			blur,
			device: device.clone(),
		})
	}

	/// Notes a new picture size.
	///
	/// The buffers are let go rather than made again: the next frame that asks
	/// makes them at the new size.
	pub(crate) fn resize(&mut self, width: u32, height: u32) {
		if self.size == (width, height) {
			return;
		}

		self.size = (width, height);
		self.let_go();
	}

	/// Lets both buffers go, and says so to whoever holds a group over them.
	fn let_go(&mut self) {
		if self.buffers.take().is_some() {
			self.epoch = self.epoch.wrapping_add(1);
		}
	}

	/// Records both passes, or lets the buffers go.
	///
	/// @param encoder - the frame's, with the pass before the scene already in
	/// it
	/// @param queue - where the tuning block is written
	/// @param asked - what this frame wants, or nothing
	/// @param prepass - what that pass wrote
	/// @param rectangle - the part of the picture drawn into, or all of it
	/// @param timings - what the passes write their marks into
	pub(crate) fn render(
		&mut self,
		encoder: &mut CommandEncoder,
		queue: &Queue,
		asked: Option<Asking>,
		prepass: &Prepass,
		rectangle: Option<Viewport>,
		timings: &Timings,
	) {
		let (Some(asked), Some(depth), Some(surfaces)) =
			(asked, prepass.depth(), prepass.surfaces())
		else {
			// the group too: it holds the pass before the scene's buffers, which
			// that pass lets go in the same frame
			self.let_go();
			self.reading = None;

			return;
		};

		let (width, height) = self.size;
		let [view_x, view_y, view_z] = asked.view;

		queue.write_buffer(
			&self.tuning,
			0,
			bytemuck::bytes_of(&Tuning {
				view_x,
				view_y,
				view_z,
				lens: asked.lens,
				// held inside nought and one already, by the one place that makes
				// an asking
				size: [pixels(width), pixels(height), RADIUS, asked.strength],
			}),
		);

		if self.buffers.is_none() {
			self.buffers = Some(buffers(&self.device, &self.raw_layout, self.size));
			self.epoch = self.epoch.wrapping_add(1);
		}

		// kept rather than made each frame, for the reason the smear keeps its
		// depth's: this runs in every frame that asks, and a group may be kept
		// only while the views inside it are the ones that exist
		let epoch = prepass.epoch();

		if self
			.reading
			.as_ref()
			.is_none_or(|(held, _)| *held != epoch)
		{
			let group = self
				.device
				.create_bind_group(&BindGroupDescriptor {
					label: Some("occlusion prepass"),
					layout: &self.prepass_layout,
					entries: &[
						BindGroupEntry {
							binding: 0,
							resource: BindingResource::TextureView(depth),
						},
						BindGroupEntry {
							binding: 1,
							resource: BindingResource::TextureView(surfaces),
						},
					],
				});

			self.reading = Some((epoch, group));
		}

		let (Some(buffers), Some((_, reading))) = (self.buffers.as_ref(), self.reading.as_ref())
		else {
			return;
		};

		let inside = rectangle.map(|asked| asked.within(width, height));
		let scissor = match inside {
			| None => Some(Viewport::whole(buffers.size.0, buffers.size.1)),
			| Some(inside) => inside.map(|inside| halved(inside, buffers.size)),
		};

		screen_pass(
			encoder,
			"occlusion estimate",
			&buffers.raw,
			timings.writes(Pass::Occlusion, Ends::Open),
			scissor,
			|pass| {
				pass.set_pipeline(&self.estimate);
				pass.set_bind_group(0, &self.numbers, &[]);
				pass.set_bind_group(1, reading, &[]);
			},
		);
		screen_pass(
			encoder,
			"occlusion blur",
			&buffers.done,
			timings.writes(Pass::Occlusion, Ends::Close),
			scissor,
			|pass| {
				pass.set_pipeline(&self.blur);
				pass.set_bind_group(0, &self.numbers, &[]);
				pass.set_bind_group(1, reading, &[]);
				pass.set_bind_group(2, &buffers.raw_read, &[]);
			},
		);
	}

	/// What a reader binds this frame: half the picture on each axis, r the
	/// share of the sky and g the distance along the view, or nothing in a
	/// frame that did not ask.
	pub(crate) fn done(&self) -> Option<&TextureView> {
		self.buffers.as_ref().map(|buffers| &buffers.done)
	}

	/// What the scene binds for the share of the sky: [`done`](Self::done), or
	/// one texel that says all of it in a frame that did not ask.
	pub(crate) fn bound(&self) -> &TextureView { self.done().unwrap_or(&self.none) }

	/// Which making of the buffers [`bound`](Self::bound) hands back. @ref
	/// [`epoch`](Self::epoch)'s field for when it moves.
	pub(crate) const fn epoch(&self) -> u64 { self.epoch }

	/// What the last frame wrote, two floats a texel, top row first. A test's.
	///
	/// @param device - to build the staging buffer on
	/// @param queue - to submit the copy on
	#[cfg(test)]
	pub(crate) fn values(&self, device: &Device, queue: &Queue) -> Option<Vec<[f32; 2]>> {
		let texture = self.done()?.texture();
		let bytes = crate::depth::copied_out(device, queue, texture, TextureAspect::All, 8)?;

		Some(
			bytes
				.chunks_exact(8)
				.map(|texel| {
					[0, 4].map(|at| {
						texel
							.get(at..at + 4)
							.and_then(|four| four.try_into().ok())
							.map_or(0.0, f32::from_le_bytes)
					})
				})
				.collect(),
		)
	}
}

/// How a reader binds one of the buffers here: a float texture that cannot be
/// filtered, read with `textureLoad`.
///
/// @param binding - where in the reader's own group it sits
pub(crate) const fn entry(binding: u32) -> BindGroupLayoutEntry {
	BindGroupLayoutEntry {
		binding,
		visibility: ShaderStages::FRAGMENT,
		ty: BindingType::Texture {
			sample_type: TextureSampleType::Float { filterable: false },
			view_dimension: TextureViewDimension::D2,
			multisampled: false,
		},
		count: None,
	}
}

/// One texel of the buffers' format that says a surface sees all of the sky
/// and nothing was drawn there.
///
/// **One, and read with its coordinate held inside it.** A texel read past the
/// end of a texture comes back nought, and nought is the whole sky hidden, so
/// the scene holds every coordinate it reads to the size of what is bound
/// rather than trusting a one-by-one texture to answer for a whole picture.
/// Multiplying by it is multiplying by one, which leaves every bit where it
/// was.
///
/// @param device - the device to build against
/// @param queue - where the texel is written
fn whole_sky(device: &Device, queue: &Queue) -> TextureView {
	let texture = device.create_texture(&TextureDescriptor {
		label: Some("occlusion none"),
		size: Extent3d {
			width: 1,
			height: 1,
			depth_or_array_layers: 1,
		},
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
		bytemuck::cast_slice(&[1.0_f32, 0.0]),
		TexelCopyBufferLayout {
			offset: 0,
			bytes_per_row: Some(8),
			rows_per_image: Some(1),
		},
		Extent3d {
			width: 1,
			height: 1,
			depth_or_array_layers: 1,
		},
	);

	texture.create_view(&TextureViewDescriptor::default())
}

/// A count of pixels as the float the shader reads it as.
#[expect(
	clippy::as_conversions,
	clippy::cast_precision_loss,
	reason = "a picture's side, nowhere near where f32 stops holding integers"
)]
const fn pixels(count: u32) -> f32 { count as f32 }

/// A rectangle of the picture as the texels of a half-sized buffer that cover
/// it, cut to the buffer.
///
/// @param rectangle - inside the picture
/// @param size - the buffer's
fn halved(rectangle: Viewport, size: (u32, u32)) -> Viewport {
	let left = (rectangle.x / 2).min(size.0);
	let top = (rectangle.y / 2).min(size.1);
	let right = rectangle
		.x
		.saturating_add(rectangle.width)
		.div_ceil(2)
		.min(size.0);
	let bottom = rectangle
		.y
		.saturating_add(rectangle.height)
		.div_ceil(2)
		.min(size.1);

	Viewport {
		x: left,
		y: top,
		width: right.saturating_sub(left),
		height: bottom.saturating_sub(top),
	}
}

/// The two buffers at half a picture's size.
fn buffers(
	device: &Device,
	raw_layout: &BindGroupLayout,
	(width, height): (u32, u32),
) -> Buffers {
	let size = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
	let target = |label| {
		device
			.create_texture(&TextureDescriptor {
				label: Some(label),
				size: Extent3d {
					width: size.0,
					height: size.1,
					depth_or_array_layers: 1,
				},
				mip_level_count: 1,
				sample_count: 1,
				dimension: TextureDimension::D2,
				format: FORMAT,
				usage: TextureUsages::RENDER_ATTACHMENT
					| TextureUsages::TEXTURE_BINDING
					| copied(),
				view_formats: &[],
			})
			.create_view(&TextureViewDescriptor::default())
	};
	let raw = target("occlusion raw");
	let done = target("occlusion");
	let raw_read = device.create_bind_group(&BindGroupDescriptor {
		label: Some("occlusion raw"),
		layout: raw_layout,
		entries: &[BindGroupEntry {
			binding: 0,
			resource: BindingResource::TextureView(&raw),
		}],
	});

	Buffers { size, raw, raw_read, done }
}

/// The copy bit a test reads a buffer back through: a test's and only a test's.
const fn copied() -> TextureUsages {
	if cfg!(test) {
		TextureUsages::COPY_SRC
	} else {
		TextureUsages::empty()
	}
}

/// One pass over one triangle covering its target, cleared to a share of one
/// and a distance of nought first, which is what a pixel nothing was drawn in
/// reads as.
///
/// @param scissor - the texels to write, or nothing for a rectangle with
/// nothing inside the picture, which leaves the target cleared
fn screen_pass<F>(
	encoder: &mut CommandEncoder,
	label: &str,
	view: &TextureView,
	marks: Option<RenderPassTimestampWrites<'_>>,
	scissor: Option<Viewport>,
	setup: F,
) where
	F: FnOnce(&mut RenderPass<'_>),
{
	let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
		label: Some(label),
		color_attachments: &[Some(RenderPassColorAttachment {
			view,
			depth_slice: None,
			resolve_target: None,
			ops: Operations {
				load: LoadOp::Clear(Color { r: 1.0, g: 0.0, b: 0.0, a: 0.0 }),
				store: StoreOp::Store,
			},
		})],
		depth_stencil_attachment: None,
		timestamp_writes: marks,
		occlusion_query_set: None,
		multiview_mask: None,
	});

	let Some(scissor) = scissor.filter(|cut| cut.width > 0 && cut.height > 0) else {
		return;
	};

	pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
	setup(&mut pass);
	pass.draw(0..3, 0..1);
}

/// One pipeline over the full-screen triangle, writing the buffers' format with
/// nothing blended and no depth.
fn pipeline(
	device: &Device,
	module: &ShaderModule,
	label: &str,
	entry: &str,
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
				format: FORMAT,
				blend: None,
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
		abi::{
			EntityId, Light, MaterialId, MeshId, Post, Renderable, Sky, ToneMap, Transform,
			Value,
			material::{Blend, Material},
		},
		glam::{Quat, Vec2, Vec3},
	};

	use super::*;
	use crate::{Capture, Image, prepass, scene::MSAA};

	/// How big every capture here is.
	const SIZE: (u32, u32) = (320, 240);

	/// How big the buffers are, half of it rounded up.
	const HALF: (u32, u32) = (160, 120);

	/// Where the wall's face is, along z.
	const WALL: f32 = -2.0;

	/// The reach every expectation here is worked out at, written down again
	/// rather than read off [`RADIUS`]: a test that took its answer's reach
	/// from the constant under test would move with it.
	const REACH: f32 = 0.5;

	/// `PLANE` in `occlusion.wgsl`: how far off a texel's plane another may be
	/// and still be averaged with it, as a share of its distance.
	const PLANE_SHARE: f32 = 0.02;

	/// How far a texel's share may be from the crease's closed form averaged
	/// the way the second pass averages, at worst and on the mean.
	///
	/// **Measured at 0.020 and 0.0062, and set at twice each**, which is as
	/// loose as they can be and still fail on what they are for. The estimate
	/// finds a horizon a little short of the reach's end and so reads a little
	/// light, worst at the crease itself. The worst is what an estimate left
	/// unaveraged fails on and taps bunched towards the middle; the mean is
	/// what a reach a quarter longer or a tenth shorter fails on, one slice
	/// where there should be three and a hemisphere weighted evenly rather than
	/// by the cosine, because each moves every texel of the band a little
	/// rather than one texel a lot. @ref the mutations recorded for parity card
	/// C2 for the numbers each came to.
	const WORST_WITHIN: f64 = 0.04;
	const MEAN_WITHIN: f64 = 0.013;

	/// The worst the same crease may be off the middle of the picture, where
	/// its far end is seen at a slant and the horizon is found in fewer pixels.
	///
	/// **Measured at 0.045**, and set at twice that: an estimate left
	/// unaveraged is 0.23 here and a reach a quarter longer 0.087.
	const OFF_MIDDLE_WORST: f64 = 0.09;

	/// How far a crease seen through a very wide lens may be off, on the mean
	/// and at worst.
	///
	/// **The mean measured at 0.016** and set just over it, because what it is
	/// for is close: dividing what is hidden by one rather than by what the
	/// same slices add up to with nothing in the way reads 0.022 here, and
	/// nowhere else in these tests is that far apart. **The worst measured at
	/// 0.20**, where the wall is seen so nearly edge on that it is a few pixels
	/// high on the screen and a slice finds too few of them to rise to its
	/// horizon - the limit a screen has, recorded rather than hidden.
	const WIDE_MEAN_WITHIN: f64 = 0.019;
	const WIDE_WORST: f64 = 0.25;

	/// How many texels in from each edge of the buffer a crease is checked, so
	/// that no tap the estimate takes for them lands off the picture: the reach
	/// is some forty pixels across on the screen here, twenty texels.
	const MARGIN: u32 = 24;

	/// How far a byte of the picture at one may be from the picture at nought
	/// times the share, worked out in linear light and put back through the
	/// curve.
	///
	/// **Measured at 0.70 and set at twice it.** What is left is two roundings
	/// to a byte, the one the picture at nought was read back through and the
	/// one the picture at one is written with. Reading the nearest texel rather
	/// than blending four is some three bytes out across the crease, and a
	/// share multiplied in twice or by the wrong strength a great deal more.
	const PICTURE_WITHIN: f32 = 1.4;

	/// How many pixels of a box's edge in front of a dark crease may still pick
	/// the crease up at four samples a pixel, out of some eighty that do when
	/// the distance is not matched at all.
	///
	/// **Measured at 13 and set at twice it; at one sample it is nought.** At
	/// four, a fragment on an edge is shaded at the middle of a pixel the pass
	/// before the scene saw the crease at, and when none of the four texels
	/// around that pixel is the box there is nothing of the box's own to read.
	/// The one sample a pixel that pass draws with, `C1-4`, is where that comes
	/// from.
	const EDGE_AT_FOUR: usize = 26;

	/// A capture on the binary's one device, or `None` with no GPU.
	fn capture() -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, SIZE.0, SIZE.1) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// How many samples a pixel is drawn with, and what the view draws.
	fn asking(world: &mut World, samples: &str, showing: &str) {
		world.cvars.var(MSAA, Value::Float(1.0), "");
		world.cvars.set(MSAA, samples);
		world
			.cvars
			.var(prepass::VIEW, Value::Float(prepass::NO_VIEW), "");
		world.cvars.set(prepass::VIEW, showing);
	}

	/// How much of what is hidden the picture takes away.
	fn strength(world: &mut World, asked: &str) {
		world
			.cvars
			.var(STRENGTH, Value::Float(DEFAULT_STRENGTH), "");
		world.cvars.set(STRENGTH, asked);
	}

	/// The picture's own look out of the way, and the light arriving from
	/// everywhere the only light: no curve, an exposure of one, no sky, and a
	/// sun that travels straight up and so lights nothing the camera sees.
	fn flat(world: &mut World, ambient: f32) {
		world.post = Post {
			tonemap: ToneMap::None,
			auto_exposure: false,
			exposure: 1.0,
			..Post::DEFAULT
		};
		world.sky = Sky::NONE;
		world.clear = Vec3::ZERO;
		world.light = Vec3::Y;
		world.ambient = Vec3::splat(ambient);
	}

	/// The scene's own shader with one piece of it replaced, which has to be in
	/// it exactly once.
	fn variant(find: &str, replace: &str) -> String {
		let source = include_str!("shader.wgsl");

		assert_eq!(source.matches(find).count(), 1, "`{find}` is in the shader exactly once");

		source.replace(find, replace)
	}

	/// A box of the default material standing somewhere.
	fn slab(world: &mut World, position: Vec3, scale: Vec3) -> EntityId {
		tinted(world, position, scale, Vec3::ONE)
	}

	/// The same in a color of its own.
	fn tinted(world: &mut World, position: Vec3, scale: Vec3, tint: Vec3) -> EntityId {
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

	/// How many pixels of two pictures differ, and how many of those are darker
	/// in the first.
	fn darker(one: &Image, other: &Image) -> (usize, usize) {
		let pairs: Vec<(&[u8], &[u8])> = one
			.pixels
			.chunks_exact(4)
			.zip(other.pixels.chunks_exact(4))
			.filter(|(a, b)| a != b)
			.collect();
		let dimmer = pairs
			.iter()
			.filter(|(a, b)| {
				a.iter()
					.take(3)
					.map(|v| u32::from(*v))
					.sum::<u32>() < b
					.iter()
					.take(3)
					.map(|v| u32::from(*v))
					.sum::<u32>()
			})
			.count();

		(pairs.len(), dimmer)
	}

	/// A floor whose top is at nought and a wall whose face is at `wall`, seen
	/// from `eye` looking at `target`.
	fn crease_at(eye: Vec3, target: Vec3, wall: f32) -> World {
		let mut world = World::new();

		world.camera.position = eye;
		world.camera.target = target;
		slab(&mut world, Vec3::new(0.0, -0.5, 0.0), Vec3::new(40.0, 1.0, 40.0));
		slab(&mut world, Vec3::new(0.0, 5.0, wall - 0.5), Vec3::new(40.0, 10.0, 1.0));

		world
	}

	/// The crease looked at from above and in front, close enough that the
	/// reach in front of the wall is a band of texels rather than a line.
	fn crease() -> World { crease_at(Vec3::new(0.0, 2.5, 1.0), Vec3::new(0.0, 0.0, WALL), WALL) }

	/// The share of the sky a point of one of two planes meeting square sees at
	/// `k` of the reach from the other: the cosine-weighted hemisphere with
	/// every direction that meets the other plane within the reach taken away,
	/// in closed form. A half at the crease and the whole of it from the reach
	/// on.
	fn crease_share(k: f64) -> f64 {
		let k = k.clamp(0.0, 1.0);

		1.0 - (k.acos() - k * (1.0 - k * k).sqrt()) / core::f64::consts::PI
	}

	/// The ray through the middle of a pixel of the picture, in the world.
	#[expect(
		clippy::as_conversions,
		clippy::cast_precision_loss,
		reason = "pixel places inside a three-hundred-and-twenty by two-hundred-and-forty \
		          picture"
	)]
	fn ray(world: &World, pixel: (u32, u32)) -> (Vec3, Vec3) {
		let inverse = world
			.render_camera()
			.view_projection(world.aspect)
			.inverse();
		let ndc = Vec2::new(
			((pixel.0 as f32 + 0.5) / SIZE.0 as f32).mul_add(2.0, -1.0),
			((pixel.1 as f32 + 0.5) / SIZE.1 as f32).mul_add(-2.0, 1.0),
		);
		let near = inverse.project_point3(ndc.extend(0.0));

		(near, (inverse.project_point3(ndc.extend(1.0)) - near).normalize())
	}

	/// Where the ray through the middle of a pixel meets a crease with its wall
	/// at `wall`, the normal there, and how far that point is from the other
	/// plane in reaches.
	fn crease_hit(world: &World, wall: f32, pixel: (u32, u32)) -> Option<(Vec3, Vec3, f64)> {
		let (near, way) = ray(world, pixel);
		let floor = (way.y < 0.0).then(|| -near.y / way.y);
		let ahead = (way.z < 0.0).then(|| (wall - near.z) / way.z);

		match (floor, ahead) {
			| (Some(down), Some(across)) if down <= across => {
				let point = near + way * down;

				Some((point, Vec3::Y, f64::from((point.z - wall) / REACH)))
			},
			| (_, Some(across)) => {
				let point = near + way * across;

				(point.y >= 0.0).then(|| (point, Vec3::Z, f64::from(point.y / REACH)))
			},
			| _ => None,
		}
	}

	/// What the second pass hands back for a texel of the crease, worked out
	/// from the closed form rather than estimated: its average over the three
	/// by three texels around it that lie on its plane.
	fn crease_expected(world: &World, wall: f32, texel: (u32, u32)) -> Option<f64> {
		let (point, normal, _) = crease_hit(world, wall, (texel.0 * 2, texel.1 * 2))?;
		let camera = world.render_camera();
		let along_view =
			(point - camera.position).dot((camera.target - camera.position).normalize());
		let shares: Vec<f64> = [-1, 0, 1]
			.into_iter()
			.flat_map(|down| [-1, 0, 1].map(|across| (across, down)))
			.filter_map(|(across, down)| {
				let column = texel.0.checked_add_signed(across)?;
				let row = texel.1.checked_add_signed(down)?;

				(column < HALF.0 && row < HALF.1).then_some((column * 2, row * 2))
			})
			.filter_map(|pixel| crease_hit(world, wall, pixel))
			.filter(|(other, ..)| (*other - point).dot(normal).abs() <= PLANE_SHARE * along_view)
			.map(|(.., k)| crease_share(k))
			.collect();

		let count = u32::try_from(shares.len()).ok()?;

		(count > 0).then(|| shares.iter().sum::<f64>() / f64::from(count))
	}

	/// One texel of a readback.
	fn at(values: &[[f32; 2]], (column, row): (u32, u32)) -> [f32; 2] {
		values
			.get(usize::try_from(row * HALF.0 + column).unwrap_or(usize::MAX))
			.copied()
			.expect("the texel is inside the buffer")
	}

	/// How far a band of the crease is from what the closed form says, as the
	/// worst and the mean, and over how many texels.
	///
	/// The band is every texel within one and a half reaches of the crease on
	/// either plane, among the columns given and [`MARGIN`] rows in from the
	/// top and the bottom.
	fn crease_error(
		world: &World,
		wall: f32,
		values: &[[f32; 2]],
		columns: core::ops::Range<u32>,
	) -> (f64, f64, u32) {
		let offs: Vec<f64> = (MARGIN..HALF.1 - MARGIN)
			.flat_map(|row| columns.clone().map(move |column| (column, row)))
			.filter(|texel| {
				crease_hit(world, wall, (texel.0 * 2, texel.1 * 2))
					.is_some_and(|(.., k)| k <= 1.5)
			})
			.filter_map(|texel| {
				crease_expected(world, wall, texel)
					.map(|wanted| (f64::from(at(values, texel)[0]) - wanted).abs())
			})
			.collect();
		let count = u32::try_from(offs.len()).unwrap_or(0);

		(
			offs.iter().copied().fold(0.0, f64::max),
			offs.iter().sum::<f64>() / f64::from(count.max(1)),
			count,
		)
	}

	/// Draws a frame and reads the buffer back.
	fn written(capture: &mut Capture, world: &mut World) -> Vec<[f32; 2]> {
		capture.draw(world, &mut []);

		capture
			.scene_mut()
			.occlusion_values()
			.expect("asked for, so written")
	}

	/// The texel of the buffer a point of the world lands in.
	fn texel_of(world: &World, point: Vec3) -> (u32, u32) {
		let clip = world
			.render_camera()
			.view_projection(world.aspect)
			.project_point3(point);
		let across = clip.x.mul_add(0.5, 0.5) * 160.0;
		let down = clip.y.mul_add(-0.5, 0.5) * 120.0;

		assert!(
			(0.0..160.0).contains(&across) && (0.0..120.0).contains(&down),
			"{point} lands at ({across}, {down}), off the buffer"
		);

		(whole(across), whole(down))
	}

	/// A place in the buffer, asserted to be inside it, as the texel it is in.
	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "a place asserted to be inside a hundred-and-sixty by hundred-and-twenty buffer"
	)]
	const fn whole(place: f32) -> u32 { place as u32 }

	#[test]
	fn in_front_of_a_wall_the_floor_and_the_wall_see_the_share_a_crease_has_in_closed_form() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = crease();

		asking(&mut world, "1", "3");

		let once = written(&mut capture, &mut world);
		let (worst, mean, count) = crease_error(&world, WALL, &once, HALF.0 / 4..HALF.0 * 3 / 4);

		assert!(count > 2000, "the band is texels rather than a line: {count} of them");
		assert!(
			worst <= WORST_WITHIN && mean <= MEAN_WITHIN,
			"the crease is out by {worst:.4} at worst and {mean:.4} on the mean over {count} \
			 texels"
		);

		asking(&mut world, "4", "3");

		let four = written(&mut capture, &mut world);

		assert!(
			once.iter()
				.zip(&four)
				.all(|(one, other)| one.map(f32::to_bits) == other.map(f32::to_bits)),
			"and it is worked out from one sample a pixel whatever the picture is drawn with"
		);
	}

	#[test]
	fn a_crease_off_the_middle_of_the_picture_sees_what_one_in_the_middle_sees() {
		// a normal turned into view space by the wrong rows, and a slice put on
		// the screen the wrong way up, both come out nearly right straight ahead;
		// here the camera is turned a long way along the wall and down, so the
		// band runs from one corner of the picture to the other
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = crease_at(Vec3::new(-1.6, 2.2, 0.4), Vec3::new(1.2, 0.3, WALL), WALL);

		asking(&mut world, "1", "3");

		let values = written(&mut capture, &mut world);
		let (worst, mean, count) = crease_error(&world, WALL, &values, MARGIN..HALF.0 - MARGIN);

		assert!(count > 2000, "the band is texels rather than a line: {count} of them");
		assert!(
			worst <= OFF_MIDDLE_WORST && mean <= MEAN_WITHIN,
			"off the middle the crease is out by {worst:.4} at worst and {mean:.4} on the mean \
			 over {count} texels"
		);
	}

	#[test]
	fn an_open_floor_sees_the_whole_sky_to_the_last_bit_and_nothing_drawn_reads_one() {
		// a floor with nothing on it, looked at slantwise so that every texel of
		// it leans away from the eye by a different angle, and the sky above it
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();

		world.camera.position = Vec3::new(1.5, 2.0, 3.0);
		world.camera.target = Vec3::new(-2.0, 0.0, -4.0);
		slab(&mut world, Vec3::new(0.0, -0.5, 0.0), Vec3::new(400.0, 1.0, 400.0));
		asking(&mut world, "1", "3");

		let values = written(&mut capture, &mut world);
		let floor: Vec<&[f32; 2]> = values
			.iter()
			.filter(|texel| texel[1] > 0.0)
			.collect();
		let sky: Vec<&[f32; 2]> = values
			.iter()
			.filter(|texel| texel[1] <= 0.0)
			.collect();

		assert!(floor.len() > 5000 && sky.len() > 5000, "both halves are in the picture");
		assert!(
			floor
				.iter()
				.all(|texel| texel[0].to_bits() == 1.0_f32.to_bits()),
			"every texel of an open floor sees all of the sky, to the last bit: {:?}",
			floor
				.iter()
				.map(|texel| texel[0])
				.fold(1.0_f32, f32::min)
		);
		assert!(
			sky.iter()
				.all(|texel| texel[0].to_bits() == 1.0_f32.to_bits()),
			"and nothing drawn reads one"
		);
	}

	#[test]
	fn a_box_on_the_floor_darkens_the_floor_at_its_foot_and_leaves_its_top_open() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();

		world.camera.position = Vec3::new(0.0, 2.6, 3.2);
		world.camera.target = Vec3::new(0.0, 0.3, 0.0);
		slab(&mut world, Vec3::new(0.0, -0.5, 0.0), Vec3::new(40.0, 1.0, 40.0));
		slab(&mut world, Vec3::new(0.0, 0.5, 0.0), Vec3::ONE);
		asking(&mut world, "1", "3");

		let values = written(&mut capture, &mut world);
		let share_at = |point: Vec3| f64::from(at(&values, texel_of(&world, point))[0]);

		// a hand's width in front of the box's front face, on the floor: the face
		// hides most of that side of the sky within a half
		let foot = share_at(Vec3::new(0.0, 0.0, 0.58));
		let top = share_at(Vec3::new(0.0, 1.0, 0.1));
		let away = share_at(Vec3::new(1.4, 0.0, 1.2));

		assert!(foot < 0.7, "the floor at the box's foot sees {foot:.3} of the sky");
		assert!(foot > 0.5, "and more than the half the crease itself does: {foot:.3}");
		assert!((top - 1.0).abs() < 1.0e-6, "the box's top sees all of it: {top:.6}");
		assert!((away - 1.0).abs() < 1.0e-6, "and so does the floor out of reach: {away:.6}");
	}

	#[test]
	fn a_strength_nobody_could_mean_lands_somewhere_definite() {
		// no device: what a frame asks for is worked out before anything is drawn.
		// Compared as bits, and written out as literals rather than read off the
		// constants they check
		let mut world = World::new();
		let camera = world.render_camera();
		let asked = |world: &World, showing| {
			asking_of(world, &camera, showing).map(|asking| asking.strength.to_bits())
		};
		let whole = Some(1.0_f32.to_bits());

		assert_eq!(asked(&world, None), whole, "a world that never said is taken at all of it");

		for (typed, wanted) in
			[("0.25", Some(0.25_f32.to_bits())), ("0", None), ("-1", None), ("3", whole)]
		{
			strength(&mut world, typed);

			assert_eq!(asked(&world, None), wanted, "a strength of {typed}");
		}

		// the console refuses a nan typed at it, and a game setting the variable
		// does not have to
		assert_eq!(held(f32::NAN).to_bits(), 1.0_f32.to_bits(), "a nan is all of it");

		strength(&mut world, "0");

		assert_eq!(
			asked(&world, Some(Showing::Occlusion)),
			whole,
			"and the view asks at the whole of it whatever the strength says"
		);
		assert_eq!(
			asked(&world, Some(Showing::Normal)),
			None,
			"while a view of something else asks for nothing"
		);
	}

	#[test]
	fn the_passes_run_while_the_picture_or_the_view_asks_and_not_at_all_at_nought() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = crease();

		for samples in ["4", "1"] {
			asking(&mut world, samples, "0");
			strength(&mut world, "0");

			let before = capture
				.shoot(&mut world)
				.expect("the capture renders");
			let quiet = capture.scene_mut().spans().passes();
			let held_quiet = capture.scene_mut().occlusion_values().is_some();

			// the view asks whatever the strength says
			asking(&mut world, samples, "3");
			capture.draw(&mut world, &mut []);

			let viewed = capture.scene_mut().spans().passes();

			// and the picture asks at any strength above nought
			asking(&mut world, samples, "0");
			strength(&mut world, "0.25");
			capture.draw(&mut world, &mut []);

			let pictured = capture.scene_mut().spans().passes();
			let held = capture.scene_mut().occlusion_values().is_some();

			strength(&mut world, "0");

			let after = capture
				.shoot(&mut world)
				.expect("the capture renders");

			assert_eq!(
				viewed,
				quiet + 3,
				"at {samples} samples the view asks for the pass before the scene, the estimate \
				 and the average"
			);
			assert_eq!(
				pictured,
				quiet + 3,
				"and so does the picture at a quarter of the strength"
			);
			assert_eq!(capture.scene_mut().spans().passes(), quiet, "and at nought none of them");
			assert!(!held_quiet && held, "the buffers are there only while asked");
			assert!(
				capture.scene_mut().occlusion_values().is_none(),
				"and let go when nothing asks"
			);
			assert!(before.pixels == after.pixels, "and the picture at nought is the one it was");
		}
	}

	#[test]
	fn group_nought_is_made_again_only_when_the_buffers_are_made_or_let_go() {
		// kept rather than made each frame: a frame that asks what the last one
		// asked binds the group it already has, and nothing a picture shows
		// would say otherwise
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = crease();

		asking(&mut world, "1", "0");
		strength(&mut world, "1");
		capture.draw(&mut world, &mut []);

		let first = capture.scene_mut().rebinds();

		for _ in 0..4 {
			capture.draw(&mut world, &mut []);
		}

		assert!(
			first >= 1,
			"the first frame that asks makes the buffers and the group over them"
		);
		assert_eq!(
			capture.scene_mut().rebinds(),
			first,
			"four frames that ask the same make no group"
		);

		strength(&mut world, "0");
		capture.draw(&mut world, &mut []);
		capture.draw(&mut world, &mut []);

		assert_eq!(capture.scene_mut().rebinds(), first + 1, "the buffers let go is one group");

		strength(&mut world, "0.5");
		capture.draw(&mut world, &mut []);
		capture.draw(&mut world, &mut []);

		assert_eq!(capture.scene_mut().rebinds(), first + 2, "and made again is one more");
	}

	#[test]
	fn at_nought_the_picture_is_the_one_a_shader_that_multiplies_nothing_draws() {
		// the texel a frame at nought binds says all of the sky, and multiplying
		// by it has to leave every bit where it was. The second answer is the
		// shader with the multiply taken out, which draws the same picture at any
		// strength. The frame at one comes first, so the frame at nought after it
		// has let its buffers go and has to have made group nought again rather
		// than reading them stale.
		let Some(mut capture) = capture() else {
			return;
		};
		// a sun and a lamp too, so that a multiply that reached either would be
		// in the picture as well
		let mut world = lit_corner(0.6);
		let unmultiplied = variant("indirect *= lit;", "");

		for samples in ["1", "4"] {
			asking(&mut world, samples, "0");
			capture
				.scene_mut()
				.set_shader(include_str!("shader.wgsl"))
				.expect("the scene's own shader builds");
			strength(&mut world, "1");

			let occluded = capture
				.shoot(&mut world)
				.expect("the capture renders");

			strength(&mut world, "0");

			let plain = capture
				.shoot(&mut world)
				.expect("the capture renders");

			capture
				.scene_mut()
				.set_shader(&unmultiplied)
				.expect("the shader without the multiply builds");
			strength(&mut world, "1");

			let passes_asked = capture
				.shoot(&mut world)
				.expect("the capture renders");

			strength(&mut world, "0");

			let nothing = capture
				.shoot(&mut world)
				.expect("the capture renders");
			let (moved, dimmer) = darker(&occluded, &plain);

			assert!(
				plain.pixels == nothing.pixels,
				"at {samples} samples nought draws what a shader that multiplies nothing draws"
			);
			assert!(
				passes_asked.pixels == nothing.pixels,
				"and asking for the passes without the multiply changes no pixel either"
			);
			assert!(
				moved > 1000 && dimmer == moved,
				"while at one the corner is darker, and nothing is lighter: {dimmer} of {moved} \
				 pixels"
			);
		}
	}

	#[test]
	fn with_nothing_arriving_from_everywhere_the_picture_at_one_is_the_picture_at_nought() {
		// what a light sends is never multiplied: with the ambient term at nought
		// the sun on the floor and the wall and a lamp in the corner are the
		// whole picture, and it has to come out the same to the bit however much
		// of the sky the corner sees. The same world with an ambient term is what
		// says the corner is in reach of something at all.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut moved_by = |samples: &str, ambient: f32| {
			let mut world = lit_corner(ambient);

			asking(&mut world, samples, "0");
			strength(&mut world, "1");

			let occluded = capture
				.shoot(&mut world)
				.expect("the capture renders");

			strength(&mut world, "0");

			let plain = capture
				.shoot(&mut world)
				.expect("the capture renders");

			darker(&occluded, &plain).0
		};

		for samples in ["1", "4"] {
			let lights_alone = moved_by(samples, 0.0);
			let with_ambient = moved_by(samples, 0.5);

			assert_eq!(
				lights_alone, 0,
				"at {samples} samples the sun and the lamp are not multiplied: {lights_alone} \
				 pixels moved"
			);
			assert!(
				with_ambient > 1000,
				"while an ambient term in the same corner is darkened: {with_ambient} pixels"
			);
		}
	}

	/// The crease with a box on the floor near the wall, lit by a sun that
	/// reaches the floor and the wall and by a lamp in the corner, and by an
	/// ambient term of the level asked.
	fn lit_corner(ambient: f32) -> World {
		let mut world = crease();

		flat(&mut world, ambient);
		world.light = Vec3::new(0.3, -1.0, -0.6);
		slab(&mut world, Vec3::new(0.6, 0.5, WALL + 0.9), Vec3::ONE);

		let lamp = world
			.entities
			.spawn_at(Transform::at(Vec3::new(-0.8, 0.6, WALL + 0.6)));

		world
			.entities
			.set_light(lamp, Light::point(Vec3::new(1.0, 0.8, 0.6), 3.0, 4.0));

		world
	}

	#[test]
	fn half_the_strength_takes_away_half_of_what_is_hidden() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = crease();

		asking(&mut world, "1", "0");
		strength(&mut world, "1");

		let whole = written(&mut capture, &mut world);

		strength(&mut world, "0.5");

		let half = written(&mut capture, &mut world);
		let drawn = whole
			.iter()
			.zip(&half)
			.filter(|(one, _)| one[1] > 0.0);
		let mut hidden = 0;

		for (one, other) in drawn {
			hidden += u32::from(one[0] < 0.9);

			assert_eq!(one[1].to_bits(), other[1].to_bits(), "the distance is the same number");
			assert!(
				0.5_f32
					.mul_add(1.0 - one[0], other[0] - 1.0)
					.abs() <= 1.0e-6,
				"a texel that sees {} at the whole strength sees {} at half",
				one[0],
				other[0]
			);

			if one[0].to_bits() == 1.0_f32.to_bits() {
				assert_eq!(
					other[0].to_bits(),
					1.0_f32.to_bits(),
					"and all of it stays all of it"
				);
			}
		}

		assert!(hidden > 1000, "and the crease is in it: {hidden} texels under nine tenths");
	}

	/// A byte of an sRGB picture as the linear level it encodes.
	fn linear(byte: u8) -> f32 {
		let level = f32::from(byte) / 255.0;

		if level <= 0.040_45 {
			level / 12.92
		} else {
			((level + 0.055) / 1.055).powf(2.4)
		}
	}

	/// A linear level as the byte an sRGB target stores it as, before rounding.
	fn encoded(level: f32) -> f32 {
		let curved = if level <= 0.003_130_8 {
			level * 12.92
		} else {
			1.055_f32.mul_add(level.powf(1.0 / 2.4), -0.055)
		};

		curved * 255.0
	}

	/// What the scene reads for a pixel of the picture out of the half-sized
	/// buffer, worked out here rather than in the shader: the four texels
	/// around it blended by where it lies between them.
	///
	/// `None` for a pixel whose four texels do not all lie on its own surface,
	/// where the scene's match on distance takes over, and for one where
	/// nothing was drawn. **On its own surface is decided by the normal as
	/// well as the distance**: where a floor meets a wall the distance runs on
	/// without a step, and the scene still turns the floor's texel away from
	/// the wall's fragment, because the floor's distance changes faster than
	/// the wall's own slope allows.
	#[expect(
		clippy::as_conversions,
		clippy::cast_precision_loss,
		reason = "a remainder of two, which is nought or one"
	)]
	fn upsampled(
		values: &[[f32; 2]],
		surfaces: &[[f32; 4]],
		(column, row): (u32, u32),
	) -> Option<f32> {
		let (left, top) = (column / 2, row / 2);
		let (right, bottom) = ((left + 1).min(HALF.0 - 1), (top + 1).min(HALF.1 - 1));
		let places = [(left, top), (right, top), (left, bottom), (right, bottom)];
		let four = places.map(|place| at(values, place));
		let normal_at = |(across, down): (u32, u32)| {
			surfaces
				.get(usize::try_from(down * SIZE.0 + across).unwrap_or(usize::MAX))
				.map(|texel| Vec3::new(texel[0], texel[1], texel[2]))
		};
		let own = normal_at((column, row))?;
		let flat = places.iter().all(|&(across, down)| {
			normal_at((across * 2, down * 2))
				.is_some_and(|normal| normal.abs_diff_eq(own, 1.0e-3))
		});
		let near = four
			.iter()
			.map(|texel| texel[1])
			.fold(f32::MAX, f32::min);
		let far = four
			.iter()
			.map(|texel| texel[1])
			.fold(0.0, f32::max);
		// a plane's distance changes evenly across two texels; two parallel
		// surfaces one behind the other do not
		let bent = (four[0][1] + four[3][1] - four[1][1] - four[2][1]).abs();

		if !flat
			|| own == Vec3::ZERO
			|| near <= 0.0
			|| far - near > 0.02 * near
			|| bent > 1.0e-3 * near
		{
			return None;
		}

		let (across, down) = ((column % 2) as f32 * 0.5, (row % 2) as f32 * 0.5);
		let upper = (four[1][0] - four[0][0]).mul_add(across, four[0][0]);
		let lower = (four[3][0] - four[2][0]).mul_add(across, four[2][0]);

		Some((lower - upper).mul_add(down, upper))
	}

	#[test]
	fn the_picture_at_one_is_the_picture_at_nought_times_the_share_the_buffer_holds() {
		// three pictures of a room lit by nothing but the ambient term: at nought,
		// at one, and with the term itself at nought, which is black. If the share
		// multiplies exactly that term, the picture at one is the picture at
		// nought times the share, read out of the half-sized buffer the way the
		// scene reads it - which is worked out here from the buffer, for every
		// pixel whose four texels lie on one surface. The box on the floor is
		// drawn by the masked entry point, with nothing in its picture to cut
		// away, so the entry point that has to ask before its discard is in the
		// room too.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = crease();
		let cutout = world
			.materials
			.insert("test/cutout", Material { blend: Blend::Mask, ..Material::DEFAULT });
		let standing = slab(&mut world, Vec3::new(0.6, 0.5, WALL + 0.9), Vec3::ONE);

		world
			.entities
			.set_renderable(standing, Renderable::of(MeshId::CUBE, cutout, Vec3::ONE));
		asking(&mut world, "1", "0");
		strength(&mut world, "1");
		flat(&mut world, 0.0);

		let dark = capture
			.shoot(&mut world)
			.expect("the capture renders");

		flat(&mut world, 0.8);

		let occluded = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let shares = capture
			.scene_mut()
			.occlusion_values()
			.expect("asked for, so written");
		let surfaces = capture
			.scene_mut()
			.surface_values()
			.expect("asked for, so written");

		strength(&mut world, "0");

		let plain = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let mut read = 0;
		let mut hidden = 0;
		let mut worst = 0.0_f32;

		assert!(
			dark.pixels
				.chunks_exact(4)
				.all(|pixel| pixel.iter().take(3).all(|channel| *channel == 0)),
			"with nothing arriving from everywhere the room is black"
		);

		for (column, row) in
			(0..SIZE.1).flat_map(|row| (0..SIZE.0).map(move |column| (column, row)))
		{
			let Some(share) = upsampled(&shares, &surfaces, (column, row)) else {
				continue;
			};
			let before = plain.pixel(column, row);
			let after = occluded.pixel(column, row);

			read += 1;
			hidden += u32::from(share < 0.9);

			for channel in 0..3 {
				let wanted = encoded(linear(before[channel]) * share);

				worst = worst.max((f32::from(after[channel]) - wanted).abs());
			}
		}

		assert!(read > 40_000 && hidden > 1000, "the room and its corners: {read} and {hidden}");
		assert!(
			worst <= PICTURE_WITHIN,
			"the picture at one is the share of the one at nought, and it is out by {worst} of \
			 a byte over {read} pixels"
		);
	}

	#[test]
	fn a_pane_of_glass_takes_nothing_away_from_itself() {
		// nothing that blends is in the buffer, so at a pane's pixels it holds the
		// share of whatever is behind the pane, and the pane is lit with all of the
		// sky whatever that says. A pane opaque enough to hide what is behind it is
		// what makes that a comparison of its own pixels: at one and at nought they
		// are the same to the bit, while the same pane reading the buffer the way a
		// solid surface does is darker.
		let Some(mut capture) = capture() else {
			return;
		};
		let reading = variant(
			"shade(input, sampled, 1.0), sampled.a * input.tint.a",
			"shade(input, sampled, seen(input)), sampled.a * input.tint.a",
		);

		for samples in ["1", "4"] {
			let mut world = crease();

			flat(&mut world, 0.8);
			asking(&mut world, samples, "0");
			strength(&mut world, "0");
			capture
				.scene_mut()
				.set_shader(include_str!("shader.wgsl"))
				.expect("the scene's own shader builds");

			let bare = capture
				.shoot(&mut world)
				.expect("the capture renders");
			let glass = world.materials.insert("test/glass", Material {
				blend: Blend::Alpha,
				opacity: 1.0,
				..Material::colored(Vec3::new(0.2, 0.5, 0.9))
			});
			let pane = world.entities.spawn_at(Transform {
				position: Vec3::new(0.0, 0.4, WALL + 0.3),
				rotation: Quat::IDENTITY,
				scale: Vec3::new(1.6, 0.8, 0.05),
			});

			world
				.entities
				.set_renderable(pane, Renderable::of(MeshId::CUBE, glass, Vec3::ONE));

			let plain = capture
				.shoot(&mut world)
				.expect("the capture renders");

			strength(&mut world, "1");

			let occluded = capture
				.shoot(&mut world)
				.expect("the capture renders");

			capture
				.scene_mut()
				.set_shader(&reading)
				.expect("the shader whose glass reads the buffer builds");

			let wrong = capture
				.shoot(&mut world)
				.expect("the capture renders");
			// the pane's own pixels, well inside it: covered, and so is every
			// pixel around them, so that no sample of what is behind is in them
			let covered =
				|column: u32, row: u32| plain.pixel(column, row) != bare.pixel(column, row);
			let inside: Vec<(u32, u32)> = (1..SIZE.1 - 1)
				.flat_map(|row| (1..SIZE.0 - 1).map(move |column| (column, row)))
				.filter(|&(column, row)| {
					covered(column, row)
						&& covered(column - 1, row)
						&& covered(column + 1, row)
						&& covered(column, row - 1)
						&& covered(column, row + 1)
				})
				.collect();
			let same = inside
				.iter()
				.filter(|&&(column, row)| occluded.pixel(column, row) == plain.pixel(column, row))
				.count();
			let darkened = inside
				.iter()
				.filter(|&&(column, row)| wrong.pixel(column, row) != occluded.pixel(column, row))
				.count();

			assert!(inside.len() > 2000, "the pane is in the picture: {} pixels", inside.len());
			assert_eq!(same, inside.len(), "at {samples} samples the pane is not darkened");
			assert!(darkened > 500, "and glass that read the buffer would be: {darkened} pixels");
		}
	}

	#[test]
	fn the_edge_of_a_box_in_front_of_a_crease_does_not_carry_the_crease_round_it() {
		// a white box hanging in front of a black crease, further from both of its
		// planes than the reach, so every texel of the box sees all of the sky and
		// the crease behind its lower edge sees little of it. Drawn through a
		// shader that paints each surface's own color times the share it reads,
		// the box has to come out as it does through one that paints its color
		// alone: a texel of the crease beside the edge is not the box's surface
		// and is not read. The black crease paints nought either way, which is
		// what leaves the box's own samples as the whole difference at four.
		let Some(mut capture) = capture() else {
			return;
		};
		let anchor = "return vec4<f32>(shade(input, sampled, seen(input)), 1.0);";
		let painted = variant(anchor, "return vec4<f32>(input.tint.rgb * seen(input), 1.0);");
		let whole = variant(anchor, "return vec4<f32>(input.tint.rgb, 1.0);");
		let matching = "let own = abs(texels[index].g - along_view) <= within;";

		assert_eq!(painted.matches(matching).count(), 1, "the match is in the shader once");

		let unmatched = painted.replace(matching, "let own = true;");
		let mut world = World::new();

		world.camera.position = Vec3::new(0.0, 2.5, 1.0);
		world.camera.target = Vec3::new(0.0, 0.0, WALL);
		flat(&mut world, 0.8);
		tinted(&mut world, Vec3::new(0.0, -0.5, 0.0), Vec3::new(40.0, 1.0, 40.0), Vec3::ZERO);
		tinted(
			&mut world,
			Vec3::new(0.0, 5.0, WALL - 0.5),
			Vec3::new(40.0, 10.0, 1.0),
			Vec3::ZERO,
		);
		tinted(&mut world, Vec3::new(0.0, 0.9, WALL + 0.8), Vec3::splat(0.5), Vec3::ONE);
		strength(&mut world, "1");

		for samples in ["1", "4"] {
			asking(&mut world, samples, "0");

			let mut through = |source: &str| {
				capture
					.scene_mut()
					.set_shader(source)
					.expect("the variant builds");
				capture
					.shoot(&mut world)
					.expect("the capture renders")
			};
			let alone = through(&whole);
			let read = through(&painted);
			let careless = through(&unmatched);
			let lit = alone
				.pixels
				.chunks_exact(4)
				.filter(|pixel| pixel[0] > 0)
				.count();
			let (halo, _) = darker(&read, &alone);
			let (picked_up, dimmer) = darker(&careless, &alone);

			let allowed = if samples == "1" { 0 } else { EDGE_AT_FOUR };

			assert!(lit > 1000, "the box is in the picture: {lit} pixels");
			assert!(
				picked_up > 20 && dimmer == picked_up,
				"at {samples} samples an edge that read whatever texel is nearest picks the \
				 crease up: {dimmer} of {picked_up} pixels darker"
			);
			assert!(
				halo <= allowed && halo * 3 <= picked_up,
				"and one that matches the distance does not: {halo} pixels against {picked_up}"
			);
		}
	}

	#[test]
	fn a_resized_picture_or_a_moved_wall_is_worked_out_again_rather_than_read_stale() {
		// the group the estimate reads the pass before the scene through is kept
		// from frame to frame, and that pass makes its buffers again at a new
		// size; a group kept past that would read the old buffers
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = crease();

		asking(&mut world, "1", "3");
		capture.draw(&mut world, &mut []);
		capture.scene_mut().resize(200, 150);

		let mut moved =
			crease_at(Vec3::new(0.0, 2.5, 1.0), Vec3::new(0.0, 0.0, WALL), WALL + 0.3);

		asking(&mut moved, "1", "3");

		let again = written(&mut capture, &mut moved);
		let gpu = crate::gpu::shared().expect("the first capture found a device");
		let mut fresh = Capture::new(gpu, 200, 150).expect("a second capture builds");
		let alone = written(&mut fresh, &mut moved);

		assert_eq!(again.len(), 100 * 75, "the buffer is half the new size");
		assert!(
			again
				.iter()
				.zip(&alone)
				.all(|(kept, made)| kept.map(f32::to_bits) == made.map(f32::to_bits)),
			"and holds what a capture that never saw the old wall works out"
		);
	}

	#[test]
	fn the_view_draws_the_share_as_a_byte_and_white_where_nothing_is() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = crease();

		asking(&mut world, "1", "3");

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let values = capture
			.scene_mut()
			.occlusion_values()
			.expect("asked for, so written");
		let texels = (0..HALF.1).flat_map(|row| (0..HALF.0).map(move |column| (column, row)));
		let mut darker = 0;

		for (column, row) in texels {
			let share = at(&values, (column, row))[0];
			let seen = image.pixel(column * 2, row * 2);
			let wanted = (share * 255.0).round();

			darker += u32::from(share < 0.9);
			assert!(
				(f32::from(seen[0]) - wanted).abs() <= 1.0 && seen[0] == seen[2],
				"texel ({column}, {row}) holds {share} and the view draws {seen:?}"
			);
		}

		assert!(darker > 500, "and the crease is in it: {darker} texels under nine tenths");

		// and the view draws the share whatever the picture's strength says: what
		// it is for is looking at the estimate, and the picture is not on the
		// screen while it is
		strength(&mut world, "0");

		let at_nought = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert!(image.pixels == at_nought.pixels, "at nought the view is the same picture");
	}

	#[test]
	fn the_rectangle_the_picture_is_cut_to_is_the_rectangle_worked_out() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = crease();

		asking(&mut world, "1", "3");
		capture.draw_within(&mut world, Viewport { x: 60, y: 40, width: 200, height: 160 });

		let values = capture
			.scene_mut()
			.occlusion_values()
			.expect("asked for, so written");
		let texels = (0..HALF.1).flat_map(|row| (0..HALF.0).map(move |column| (column, row)));
		let (inside, outside): (Vec<_>, Vec<_>) = texels
			.partition(|(column, row)| (30..130).contains(column) && (20..100).contains(row));
		let dark = inside
			.iter()
			.filter(|texel| at(&values, **texel)[0] < 0.9)
			.count();

		assert!(dark > 300, "the crease is worked out inside the rectangle: {dark} texels");
		assert!(
			inside
				.iter()
				.all(|texel| at(&values, *texel)[1] > 0.0),
			"every texel inside it is written, to its edges"
		);
		assert!(
			outside
				.iter()
				.all(|texel| at(&values, *texel).map(f32::to_bits)
					== [1.0_f32.to_bits(), 0.0_f32.to_bits()]),
			"and nothing outside it is written"
		);
	}
	#[test]
	fn a_panel_hung_just_out_of_reach_in_front_of_a_wall_hides_nothing_from_it() {
		// what is further in front of a surface than the reach hides nothing,
		// however near it looks on the screen: a thin panel whose back is a
		// little more than the reach off a wall, and the eye a little to one
		// side of it so the wall shows all round its edges
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();
		let gap = REACH * 1.1;

		world.camera.position = Vec3::new(0.9, 2.4, 3.0);
		world.camera.target = Vec3::new(0.0, 2.0, WALL);
		slab(&mut world, Vec3::new(0.0, 5.0, WALL - 0.5), Vec3::new(40.0, 10.0, 1.0));
		slab(&mut world, Vec3::new(0.0, 2.0, WALL + gap + 0.025), Vec3::new(1.0, 1.0, 0.05));
		asking(&mut world, "1", "3");

		let values = written(&mut capture, &mut world);
		let drawn: Vec<f32> = values
			.iter()
			.filter(|texel| texel[1] > 0.0)
			.map(|texel| texel[0])
			.collect();

		assert!(drawn.len() > 10_000, "the wall and the panel fill the picture: {}", drawn.len());
		assert!(
			drawn
				.iter()
				.all(|share| share.to_bits() == 1.0_f32.to_bits()),
			"nothing within reach of either, and yet {} texels read less than all of the sky, \
			 the 			 least {}",
			drawn.iter().filter(|share| **share < 1.0).count(),
			drawn.iter().copied().fold(1.0_f32, f32::min)
		);
	}

	#[test]
	fn the_top_of_a_box_seen_against_a_crease_behind_it_is_not_averaged_with_the_crease() {
		// the average keeps to a texel's own plane: along the back edge of the
		// box's top, the texels just past it on the screen are the dark crease a
		// long way behind, and the top itself has nothing within reach
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = crease_at(Vec3::new(0.0, 5.0, 0.8), Vec3::new(0.0, 0.2, -1.6), WALL);
		let back = REACH.mul_add(1.2, WALL);

		slab(&mut world, Vec3::new(0.0, 0.5, back + 0.5), Vec3::ONE);
		asking(&mut world, "1", "3");

		let values = written(&mut capture, &mut world);
		let mut top = 0;
		let mut darker_behind = 0;

		for (column, row) in
			(0..HALF.1).flat_map(|row| (0..HALF.0).map(move |column| (column, row)))
		{
			let (near, way) = ray(&world, (column * 2, row * 2));
			let up = (way.y < 0.0).then(|| (1.0 - near.y) / way.y);
			let Some(point) = up.map(|t| near + way * t) else {
				continue;
			};

			if point.x.abs() < 0.5 && (back..back + 1.0).contains(&point.z) {
				top += 1;
				assert_eq!(
					at(&values, (column, row))[0].to_bits(),
					1.0_f32.to_bits(),
					"texel ({column}, {row}) of the top reads {}",
					at(&values, (column, row))[0]
				);

				let beyond = at(&values, (column, row.saturating_sub(2)))[0];

				darker_behind += u32::from(beyond < 0.9);
			}
		}

		assert!(top > 400, "the top is in the picture: {top} texels");
		assert!(
			darker_behind > 20,
			"and the crease right behind its edge is dark: {darker_behind}"
		);
	}

	#[test]
	fn a_crease_through_a_wide_lens_sees_what_one_through_a_narrow_lens_sees() {
		// towards the edge of a wide picture the eye's direction leans a long way
		// off the middle, and there a few slices of an open hemisphere add up to
		// well away from one - which is what dividing by what the slices add up
		// to with nothing in the way is for
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = crease_at(Vec3::new(-0.6, 1.2, -0.4), Vec3::new(1.4, 0.2, WALL), WALL);

		world.camera.fov_y = 2.2;
		asking(&mut world, "1", "3");

		let values = written(&mut capture, &mut world);
		let (worst, mean, count) = crease_error(&world, WALL, &values, MARGIN..HALF.0 - MARGIN);

		assert!(count > 1000, "the band is texels rather than a line: {count} of them");
		assert!(
			worst <= WIDE_WORST && mean <= WIDE_MEAN_WITHIN,
			"through a wide lens the crease is out by {worst:.4} at worst and {mean:.4} on the 			 mean over {count} texels"
		);
	}
}
