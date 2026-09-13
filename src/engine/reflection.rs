//! What each pixel's reflection finds on the picture.
//!
//! **What the sky cannot answer.** A surface's reflection of everything the
//! renderer does not simulate comes out of the world's environment, and the
//! environment is the sky: a polished ball standing on a floor reflects the
//! sky's ground where the floor is, and a smooth floor reflects the sky where a
//! wall stands on it. Nothing else this renderer holds knows where anything
//! else is in the direction a surface reflects - except the picture itself,
//! which knows it for everything the eye can see.
//!
//! **Three passes, before the scene's.** The first follows one reflected ray a
//! texel across the depth the pass before the scene wrote, at half the picture
//! on each axis, and lights what it meets; the second averages a rough
//! surface's texels with their neighbors on its own plane; the third brings the
//! result back to the picture's size. The scene reads the last. @ref
//! `fragment_reflections` in `shader.wgsl` for the first, and `reflection.wgsl`
//! for the other two.
//!
//! **What a ray meets is lit, not read off a picture.** Every screen-space
//! reflection in the field reads the color the picture already has where a ray
//! lands, and every one that draws forward reads it off the frame before,
//! reprojected, because the picture for this frame is drawn after the
//! reflections it needs. This one keeps no frame: the pass before the scene has
//! written what every visible surface is made of, and the first pass here
//! lights the place a ray lands with the scene's own arithmetic, towards the
//! point the ray left. So one frame is the whole of it - which is also what
//! every picture this engine is checked with is - nothing a reflection shows is
//! a frame late, and a thing seen in a mirror shows what it would show the
//! mirror rather than what it shows the eye.
//!
//! **What it does not find**: anything off the picture or hidden behind
//! something nearer, where the sky stays the answer; anything that blends,
//! particles and the debug lines, which are in no buffer; and what the thing a
//! ray meets would itself reflect, for which the sky stands in. A surface as
//! rough as `MIRROR_CUTOFF` in `shader.wgsl` or rougher follows no ray at all.

use colby_core::{
	Result,
	abi::{Camera, World},
	bytemuck::{self, Pod, Zeroable},
	err, warn,
};
use wgpu::{
	BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
	BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
	BufferDescriptor, BufferUsages, Color, ColorTargetState, ColorWrites, CommandEncoder, Device,
	ErrorFilter, Extent3d, FragmentState, LoadOp, MultisampleState, Operations,
	PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState, Queue, RenderPass,
	RenderPassColorAttachment, RenderPassDescriptor, RenderPassTimestampWrites, RenderPipeline,
	RenderPipelineDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
	StoreOp, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
	TextureViewDescriptor, VertexState,
};

use crate::{
	prepass::{Prepass, Showing},
	scene::Viewport,
	shader::Shader,
	timing::{Ends, Pass, Timings},
};

/// The roughness at and past which no reflection is followed.
///
/// `MIRROR_CUTOFF` in `shader.wgsl`, and a test says the two agree: the shader
/// holds the number, and this is where a reader of the buffer learns it.
#[cfg(test)]
pub(crate) const CUTOFF: f32 = 0.6;

/// The format all three buffers are written in: rgb light, a how much of the
/// reflection it stands for, the first already multiplied by the second.
///
/// Sixteen-bit floats, the picture's own, because what is found is light and a
/// sun reflected in a mirror does not fit between nought and one.
pub(crate) const FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// The numbers every pass reads, laid out the way `Mirror` in `shader.wgsl` and
/// `Tuning` in `reflection.wgsl` declare them.
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

	/// `[the target's width, its height, unused, the strength]`.
	size: [f32; 4],

	/// The rectangle of the target the picture is drawn into, as `[x, y,
	/// width, height]`.
	rect: [f32; 4],
}

/// What one frame asks the reflections for.
///
/// Worked out in [`Scene::upload`](crate::Scene) beside the other effects' own,
/// for their reason: it needs the camera the frame is drawn from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Asking {
	/// World space into view space, a row an axis.
	view: [[f32; 4]; 3],

	/// `[x scale, y scale, z_axis.z, w_axis.z]` of the projection.
	lens: [f32; 4],

	/// How much of what is found the picture takes: one is all of it.
	strength: f32,
}

/// What this frame asks for, if anything.
///
/// The views that draw what the reflections found ask, at the whole strength.
///
/// @param world - for the aspect
/// @param camera - the camera this frame is drawn from
/// @param showing - what the view is asked to draw, if anything
/// @param anyway - whether a test asks with nothing reading them
#[must_use]
pub(crate) fn asking_of(
	world: &World,
	camera: &Camera,
	showing: Option<Showing>,
	anyway: bool,
) -> Option<Asking> {
	if !anyway && !matches!(showing, Some(Showing::Reflections | Showing::Coverage)) {
		return None;
	}

	let view = camera.view();
	let lens = camera.projection(world.aspect);

	Some(Asking {
		view: [0, 1, 2].map(|axis| view.row(axis).to_array()),
		lens: [lens.x_axis.x, lens.y_axis.y, lens.z_axis.z, lens.w_axis.z],
		strength: 1.0,
	})
}

/// The three buffers, and the groups the passes after the first read them
/// through.
struct Buffers {
	/// The size of the first two, half the picture's on each axis rounded up.
	half: (u32, u32),

	/// What the first pass writes.
	raw: TextureView,

	/// How the average reads it.
	raw_read: BindGroup,

	/// What the average writes.
	averaged: TextureView,

	/// How the last pass reads it.
	averaged_read: BindGroup,

	/// What the last pass writes, at the picture's size, and what everything
	/// after reads.
	done: TextureView,
}

/// The three passes, and what they write into.
pub(crate) struct Reflection {
	/// The size of the picture, which the last buffer is and the first two are
	/// half of.
	size: (u32, u32),

	/// All three buffers, made the first frame something asks and let go the
	/// first frame nothing does.
	buffers: Option<Buffers>,

	/// Which making of [`buffers`](Self::buffers) a reader is looking at: moved
	/// when they are made and when they go.
	epoch: u64,

	/// How the first pass reads the tuning block and what the pass before the
	/// scene wrote, and which of that pass's buffers the group is for. @ref
	/// [`Prepass::epoch`].
	mirror: Option<(u64, BindGroup)>,

	/// How the other two read what the pass before the scene wrote, kept the
	/// same way.
	prepass_read: Option<(u64, BindGroup)>,

	/// The first pass, built from the scene's own source the first frame
	/// something asks - it lights what it finds with the scene's arithmetic, so
	/// it is the scene's shader - or nothing before then.
	trace: Option<RenderPipeline>,

	tuning: Buffer,
	numbers: BindGroup,
	mirror_layout: BindGroupLayout,
	prepass_layout: BindGroupLayout,
	source_layout: BindGroupLayout,
	average: RenderPipeline,
	upsample: RenderPipeline,
	device: Device,
}

impl Reflection {
	/// Builds the two passes that need nothing of the scene's, and the block
	/// every pass reads its numbers from.
	///
	/// No buffer and no first pass yet: a frame that asks for nothing never
	/// makes either.
	///
	/// @param device - the device to build against
	/// @param width - the picture's width in pixels
	/// @param height - its height
	pub(crate) fn new(device: &Device, width: u32, height: u32) -> Result<Self> {
		let uniform = |binding| BindGroupLayoutEntry {
			binding,
			visibility: ShaderStages::FRAGMENT,
			ty: BindingType::Buffer {
				ty: BufferBindingType::Uniform,
				has_dynamic_offset: false,
				min_binding_size: None,
			},
			count: None,
		};
		let numbers_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("reflection numbers"),
			entries: &[uniform(0)],
		});
		// the first pass's second group, at the bindings `shader.wgsl` gives
		// it: after the material's three, which the scene binds in that group
		let mirror_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("reflection mirror"),
			entries: &[
				uniform(3),
				crate::depth::entry(4),
				crate::prepass::entry(5),
				crate::prepass::entry(6),
			],
		});
		let prepass_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("reflection prepass"),
			entries: &[crate::depth::entry(0), crate::prepass::entry(1)],
		});
		let source_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("reflection source"),
			entries: &[crate::prepass::entry(0)],
		});
		let tuning = device.create_buffer(&BufferDescriptor {
			label: Some("reflection tuning"),
			size: u64::try_from(size_of::<Tuning>()).map_err(|_| {
				err!(Graphics("the reflection tuning block does not fit a buffer"))
			})?,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let numbers = device.create_bind_group(&BindGroupDescriptor {
			label: Some("reflection numbers"),
			layout: &numbers_layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: tuning.as_entire_binding(),
			}],
		});
		// read through the shader directory like the scene's own: a variant
		// under `COLBY_SHADERS` reaches this file too
		let source = Shader::new("reflection.wgsl", include_str!("reflection.wgsl"));
		let scope = device.push_error_scope(ErrorFilter::Validation);
		let module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("reflection"),
			source: ShaderSource::Wgsl(source.source().into()),
		});
		let groups = [Some(&numbers_layout), Some(&prepass_layout), Some(&source_layout)];
		let average =
			screen_pipeline(device, &module, "reflection average", "fragment_average", &groups);
		let upsample =
			screen_pipeline(device, &module, "reflection upsample", "fragment_upsample", &groups);

		if let Some(complaint) = pollster::block_on(scope.pop()) {
			return Err(err!(Graphics("the reflection pipelines: {complaint}")));
		}

		Ok(Self {
			size: (width, height),
			buffers: None,
			epoch: 0,
			mirror: None,
			prepass_read: None,
			trace: None,
			tuning,
			numbers,
			mirror_layout,
			prepass_layout,
			source_layout,
			average,
			upsample,
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

	/// Lets the buffers go, and says so to whoever holds a group over them.
	fn let_go(&mut self) {
		if self.buffers.take().is_some() {
			self.epoch = self.epoch.wrapping_add(1);
		}
	}

	/// Builds the first pass, if it has not been built.
	///
	/// @param device - the device to build against
	/// @param scene - the scene's frame group layout and its shadow atlas's,
	/// which the first pass reads as its first and third groups
	/// @param source - the WGSL the scene's own table was built from
	/// @return whether there is a first pass to record: nothing when it would
	/// not build, which is said
	pub(crate) fn ensure(
		&mut self,
		device: &Device,
		scene: [&BindGroupLayout; 2],
		source: &str,
	) -> bool {
		if self.trace.is_some() {
			return true;
		}

		match build_trace(device, scene, &self.mirror_layout, source) {
			| Ok(built) => {
				self.trace = Some(built);

				true
			},
			| Err(complaint) => {
				warn!(%complaint, "no reflection is followed this frame");

				false
			},
		}
	}

	/// The first pass built again from new source, or nothing if it has never
	/// been built at all.
	///
	/// Asked by the scene when its shader changes, beside the scene's own table
	/// and the pass before it, so that all three are replaced together or not
	/// at all: what a reflection meets and what the picture shows have to be
	/// lit by one arithmetic.
	///
	/// @param device - the device to build against
	/// @param scene - @ref [`ensure`](Self::ensure)
	/// @param source - the new WGSL
	pub(crate) fn rebuilt(
		&self,
		device: &Device,
		scene: [&BindGroupLayout; 2],
		source: &str,
	) -> Result<Option<RenderPipeline>> {
		if self.trace.is_none() {
			return Ok(None);
		}

		build_trace(device, scene, &self.mirror_layout, source).map(Some)
	}

	/// Puts a pipeline from [`rebuilt`](Self::rebuilt) in place, when it had
	/// one to give.
	pub(crate) fn replace(&mut self, trace: Option<RenderPipeline>) {
		if trace.is_some() {
			self.trace = trace;
		}
	}

	/// Records all three passes, or lets the buffers go.
	///
	/// @param encoder - the frame's, with the pass before the scene and the
	/// occlusion already in it
	/// @param queue - where the tuning block is written
	/// @param frame - what this frame asks and what the passes read
	/// @param rectangle - the part of the picture drawn into, or all of it
	pub(crate) fn render(
		&mut self,
		encoder: &mut CommandEncoder,
		queue: &Queue,
		frame: Frame<'_>,
		rectangle: Option<Viewport>,
	) {
		let (Some(asked), Some(depth), Some(surfaces), Some(material), true) = (
			frame.asked,
			frame.prepass.depth(),
			frame.prepass.surfaces(),
			frame.prepass.material(),
			self.trace.is_some(),
		) else {
			// the groups too: they hold the pass before the scene's buffers,
			// which that pass lets go in the same frame
			self.let_go();
			self.mirror = None;
			self.prepass_read = None;

			return;
		};

		let (width, height) = self.size;
		let inside = rectangle.map(|asked| asked.within(width, height));
		// a rectangle with nothing inside the picture is no picture, and the
		// block is written for the whole target so that it holds numbers
		let drawn = inside
			.flatten()
			.unwrap_or_else(|| Viewport::whole(width, height));

		self.write(queue, asked, drawn);

		if self.buffers.is_none() {
			self.buffers = Some(buffers(&self.device, &self.source_layout, self.size));
			self.epoch = self.epoch.wrapping_add(1);
		}

		self.keep(frame.prepass.epoch(), [depth, surfaces, material]);

		let (Some(trace), Some(buffers), Some((_, mirror)), Some((_, prepass_read))) = (
			self.trace.as_ref(),
			self.buffers.as_ref(),
			self.mirror.as_ref(),
			self.prepass_read.as_ref(),
		) else {
			return;
		};

		let (half, whole) = match inside {
			| None => (
				Some(Viewport::whole(buffers.half.0, buffers.half.1)),
				Some(Viewport::whole(width, height)),
			),
			| Some(inside) => (inside.map(|inside| halved(inside, buffers.half)), inside),
		};
		let marks = |ends| frame.timings.writes(Pass::Reflections, ends);

		screen_pass(
			encoder,
			("reflection trace", &buffers.raw),
			marks(Ends::Open),
			half,
			|pass| {
				pass.set_pipeline(trace);
				pass.set_bind_group(0, frame.scene, &[]);
				pass.set_bind_group(1, mirror, &[]);
				pass.set_bind_group(2, frame.shadows, &[]);
			},
		);

		for (label, pipeline, from, into, ends, scissor) in [
			(
				"reflection average",
				&self.average,
				&buffers.raw_read,
				&buffers.averaged,
				Ends::Middle,
				half,
			),
			(
				"reflection upsample",
				&self.upsample,
				&buffers.averaged_read,
				&buffers.done,
				Ends::Close,
				whole,
			),
		] {
			screen_pass(encoder, (label, into), marks(ends), scissor, |pass| {
				pass.set_pipeline(pipeline);
				pass.set_bind_group(0, &self.numbers, &[]);
				pass.set_bind_group(1, prepass_read, &[]);
				pass.set_bind_group(2, from, &[]);
			});
		}
	}

	/// Writes this frame's numbers into the block every pass reads.
	///
	/// @param queue - where it is written
	/// @param asked - what this frame asks for
	/// @param drawn - the rectangle of the target the picture is drawn into
	fn write(&self, queue: &Queue, asked: Asking, drawn: Viewport) {
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
				size: [pixels(width), pixels(height), 0.0, asked.strength],
				rect: [
					pixels(drawn.x),
					pixels(drawn.y),
					pixels(drawn.width),
					pixels(drawn.height),
				],
			}),
		);
	}

	/// Makes the groups that read what the pass before the scene wrote again,
	/// if that pass made its buffers again since they were made.
	///
	/// Kept rather than made each frame, for the occlusion's reason: a group
	/// may be kept only while the views inside it are the ones that exist.
	///
	/// @param epoch - which making of that pass's buffers these are, @ref
	/// [`Prepass::epoch`]
	/// @param [depth, surfaces, material] - the buffers
	fn keep(&mut self, epoch: u64, [depth, surfaces, material]: [&TextureView; 3]) {
		let stale = |held: Option<&(u64, BindGroup)>| held.is_none_or(|(made, _)| *made != epoch);
		let bound = |binding, texture| BindGroupEntry {
			binding,
			resource: BindingResource::TextureView(texture),
		};

		if stale(self.mirror.as_ref()) {
			let group = self
				.device
				.create_bind_group(&BindGroupDescriptor {
					label: Some("reflection mirror"),
					layout: &self.mirror_layout,
					entries: &[
						BindGroupEntry {
							binding: 3,
							resource: self.tuning.as_entire_binding(),
						},
						bound(4, depth),
						bound(5, surfaces),
						bound(6, material),
					],
				});

			self.mirror = Some((epoch, group));
		}

		if stale(self.prepass_read.as_ref()) {
			let group = self
				.device
				.create_bind_group(&BindGroupDescriptor {
					label: Some("reflection prepass"),
					layout: &self.prepass_layout,
					entries: &[bound(0, depth), bound(1, surfaces)],
				});

			self.prepass_read = Some((epoch, group));
		}
	}

	/// What a reader binds this frame: the picture's size, rgb light and a how
	/// much of the reflection it stands for, or nothing in a frame that did not
	/// ask.
	pub(crate) fn done(&self) -> Option<&TextureView> {
		self.buffers.as_ref().map(|buffers| &buffers.done)
	}

	/// Which making of the buffers [`done`](Self::done) hands back.
	#[cfg(test)]
	pub(crate) const fn epoch(&self) -> u64 { self.epoch }

	/// What the last frame wrote, four floats a pixel, top row first. A test's.
	///
	/// @param device - to build the staging buffer on
	/// @param queue - to submit the copy on
	#[cfg(test)]
	pub(crate) fn values(&self, device: &Device, queue: &Queue) -> Option<Vec<[f32; 4]>> {
		crate::prepass::halves(device, queue, self.done()?)
	}

	/// What the first pass wrote, four floats a texel of the half-sized buffer.
	/// A test's.
	///
	/// @param device - to build the staging buffer on
	/// @param queue - to submit the copy on
	#[cfg(test)]
	pub(crate) fn raw_values(&self, device: &Device, queue: &Queue) -> Option<Vec<[f32; 4]>> {
		crate::prepass::halves(device, queue, &self.buffers.as_ref()?.raw)
	}
}

/// What a frame hands the passes: what it asks, and everything they read that
/// is the scene's.
#[derive(Clone, Copy)]
pub(crate) struct Frame<'a> {
	/// What this frame wants, or nothing.
	pub(crate) asked: Option<Asking>,

	/// What the pass before the scene wrote.
	pub(crate) prepass: &'a Prepass,

	/// The scene's group nought: the frame's uniform, the environment, the
	/// table and the share of the sky, which the first pass lights with.
	pub(crate) scene: &'a BindGroup,

	/// The shadow atlas's group, which it lights with too.
	pub(crate) shadows: &'a BindGroup,

	/// What the passes write their marks into.
	pub(crate) timings: &'a Timings,
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

/// The three buffers: two at half a picture's size and one at its whole.
fn buffers(
	device: &Device,
	source_layout: &BindGroupLayout,
	(width, height): (u32, u32),
) -> Buffers {
	let half = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
	let target = |label, (across, down): (u32, u32)| {
		device
			.create_texture(&TextureDescriptor {
				label: Some(label),
				size: Extent3d {
					width: across,
					height: down,
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
	let read = |label, view: &TextureView| {
		device.create_bind_group(&BindGroupDescriptor {
			label: Some(label),
			layout: source_layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: BindingResource::TextureView(view),
			}],
		})
	};
	let raw = target("reflection raw", half);
	let averaged = target("reflection averaged", half);
	let done = target("reflection", (width.max(1), height.max(1)));

	Buffers {
		half,
		raw_read: read("reflection raw", &raw),
		averaged_read: read("reflection averaged", &averaged),
		raw,
		averaged,
		done,
	}
}

/// The copy bit a test reads a buffer back through: a test's and only a test's.
const fn copied() -> TextureUsages {
	if cfg!(test) {
		TextureUsages::COPY_SRC
	} else {
		TextureUsages::empty()
	}
}

/// One pass over one triangle covering its target, cleared to nothing found
/// first, which is what a pixel no ray was followed from reads as.
///
/// @param (label, view) - what the pass is called and what it writes
/// @param scissor - the texels to write, or nothing for a rectangle with
/// nothing inside the picture, which leaves the target cleared
fn screen_pass<F>(
	encoder: &mut CommandEncoder,
	(label, view): (&str, &TextureView),
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
				load: LoadOp::Clear(Color::TRANSPARENT),
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

/// The first pass, against the scene's own source.
///
/// **The scene's groups nought and two, and a group one of its own**: it reads
/// the frame's uniform, the environment, the table and the share of the sky,
/// and the shadow atlas, to light what it finds - and binds no material, so
/// the material's group is where its own inputs go. Group three is not
/// declared, and no entry point this pipeline runs reads the bones.
///
/// @param device - the device to build against
/// @param scene - the scene's frame group layout and its shadow atlas's
/// @param mirror_layout - the first pass's own inputs
/// @param source - the whole WGSL
fn build_trace(
	device: &Device,
	[frame, shadows]: [&BindGroupLayout; 2],
	mirror_layout: &BindGroupLayout,
	source: &str,
) -> Result<RenderPipeline> {
	let scope = device.push_error_scope(ErrorFilter::Validation);
	let module = device.create_shader_module(ShaderModuleDescriptor {
		label: Some("reflection trace"),
		source: ShaderSource::Wgsl(source.into()),
	});
	let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some("reflection trace"),
		bind_group_layouts: &[Some(frame), Some(mirror_layout), Some(shadows)],
		immediate_size: 0,
	});
	let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some("reflection trace"),
		layout: Some(&layout),
		vertex: VertexState {
			module: &module,
			// the sky's triangle, which covers the target out of nothing but
			// the vertex index; with no depth attached, where it lies in depth
			// does not matter
			entry_point: Some("vertex_sky"),
			compilation_options: PipelineCompilationOptions::default(),
			buffers: &[],
		},
		primitive: PrimitiveState::default(),
		depth_stencil: None,
		multisample: MultisampleState::default(),
		fragment: Some(FragmentState {
			module: &module,
			entry_point: Some("fragment_reflections"),
			compilation_options: PipelineCompilationOptions::default(),
			targets: &[Some(ColorTargetState {
				format: FORMAT,
				blend: None,
				write_mask: ColorWrites::ALL,
			})],
		}),
		multiview_mask: None,
		cache: None,
	});

	match pollster::block_on(scope.pop()) {
		| Some(complaint) => Err(err!(Graphics("the reflection trace: {complaint}"))),
		| None => Ok(pipeline),
	}
}

/// One pipeline over the full-screen triangle, writing the buffers' format with
/// nothing blended and no depth.
fn screen_pipeline(
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
			Decal, EntityId, Material, MaterialId, MeshId, Post, Renderable, Sky, ToneMap,
			Transform, Value, material::Blend,
		},
		glam::{Quat, Vec2, Vec3},
	};

	use super::*;
	use crate::{Capture, occlusion, prepass, scene::MSAA};

	/// How big every capture here is.
	const SIZE: (u32, u32) = (320, 240);

	/// Where the wall's face is, along z.
	const WALL: f32 = -3.0;

	/// How far a color the first pass lit may be from the one worked out here.
	///
	/// **Measured, then set at about three times the worst seen.** What is
	/// left is the sixteen-bit float the buffer holds, which the device
	/// truncates into rather than rounds, and a direction the lobe draws a
	/// fraction of a degree off the mirror one, which moves where on the wall a
	/// ray lands and so how square to it the wall is seen.
	const LIGHT_WITHIN: f32 = 3.0e-3;

	/// A whole step of a sixteen-bit float under one, which is what a number
	/// read back out of such a buffer may be off by. @ref `prepass`'s own
	/// `WITHIN` for why a whole step and not half of one.
	const STEP: f32 = 5.0e-4;

	/// A capture on the binary's one device, or `None` with no GPU.
	fn capture() -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, SIZE.0, SIZE.1) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// How many samples a pixel is drawn with and what the view draws, with no
	/// share of the sky taken away: what these tests work out is lit by the
	/// light arriving from everywhere at all of its strength.
	fn asking(world: &mut World, samples: &str, showing: &str) {
		world.cvars.var(MSAA, Value::Float(1.0), "");
		world.cvars.set(MSAA, samples);
		world
			.cvars
			.var(prepass::VIEW, Value::Float(prepass::NO_VIEW), "");
		world.cvars.set(prepass::VIEW, showing);
		world
			.cvars
			.var(occlusion::STRENGTH, Value::Float(occlusion::DEFAULT_STRENGTH), "");
		world.cvars.set(occlusion::STRENGTH, "0");
	}

	/// The picture's own look out of the way, and the light arriving from
	/// everywhere the only light: no curve, no sky, and a sun that travels
	/// straight up and so lights nothing that faces up or sideways.
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

	/// A white material, of a roughness and of how metal it is.
	fn made(world: &mut World, name: &str, roughness: f32, metallic: f32) -> MaterialId {
		world
			.materials
			.insert(name, Material { roughness, metallic, ..Material::DEFAULT })
	}

	/// A box of a material and a tint, standing somewhere.
	fn slab(
		world: &mut World,
		material: MaterialId,
		(position, scale): (Vec3, Vec3),
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

	/// The wall's tint.
	const RED: Vec3 = Vec3::new(0.8, 0.35, 0.2);

	/// How tall the room's wall is.
	const HEIGHT: f32 = 4.0;

	/// A floor whose top is at nought, of a roughness, and a wall whose face is
	/// at [`WALL`] standing on it, eight wide and [`HEIGHT`] high, of a
	/// metallic - seen from in front and a little above, looking down the
	/// room.
	fn room(floor_roughness: f32, wall_metallic: f32) -> World {
		room_of(floor_roughness, wall_metallic, HEIGHT)
	}

	/// The same room with a wall of another height.
	fn room_of(floor_roughness: f32, wall_metallic: f32, height: f32) -> World {
		let mut world = World::new();

		world.camera.position = Vec3::new(0.0, 1.2, 4.0);
		world.camera.target = Vec3::new(0.0, 0.8, WALL);
		flat(&mut world, 0.8);

		let floor = made(&mut world, "test/floor", floor_roughness, 0.0);
		let wall = made(&mut world, "test/wall", 0.8, wall_metallic);

		slab(
			&mut world,
			floor,
			(Vec3::new(0.0, -0.5, 0.0), Vec3::new(40.0, 1.0, 40.0)),
			Vec3::splat(0.3),
		);
		slab(
			&mut world,
			wall,
			(Vec3::new(0.0, height * 0.5, WALL - 0.25), Vec3::new(8.0, height, 0.5)),
			RED,
		);

		world
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

	/// Where a point of the world lands on the picture, in pixels.
	fn landing(world: &World, point: Vec3) -> Vec2 {
		let clip = world
			.render_camera()
			.view_projection(world.aspect)
			.project_point3(point);

		Vec2::new(clip.x.mul_add(0.5, 0.5) * 320.0, clip.y.mul_add(-0.5, 0.5) * 240.0)
	}

	/// Where the mirror reflection of the eye's ray through a pixel of the
	/// room's floor meets the wall's face, if the pixel sees the floor and the
	/// reflection meets the face well inside its edges and well inside the
	/// picture: the point on the floor and the point on the wall.
	fn mirrored(world: &World, pixel: (u32, u32)) -> Option<(Vec3, Vec3)> {
		let (near, way) = ray(world, pixel);

		if way.y >= 0.0 {
			return None;
		}

		let down = -near.y / way.y;
		let floor = near + way * down;
		// the wall's face stands between the eye and a floor behind it
		let across = (WALL - near.z) / way.z;
		let seen_wall = way.z < 0.0 && across < down && (near + way * across).y >= 0.0;

		if seen_wall || floor.z <= WALL + 0.05 {
			return None;
		}

		let back = Vec3::new(way.x, -way.y, way.z);

		if back.z >= 0.0 {
			return None;
		}

		let wall = floor + back * ((WALL - floor.z) / back.z);
		let landed = landing(world, wall);
		let inside = (8.0..312.0).contains(&landed.x) && (8.0..232.0).contains(&landed.y);
		let on_face = wall.x.abs() < 3.6 && (0.3..HEIGHT - 0.3).contains(&wall.y);
		// far enough along the picture for the march to take a few steps
		let traveled = (landed - landing(world, floor)).length();

		(inside && on_face && traveled > 12.0).then_some((floor, wall))
	}

	/// What fraction of the light arriving from everywhere a surface sends
	/// back, read out of the renderer's own table the way `ambient_brdf` in
	/// `shader.wgsl` reads it: the coordinate squeezed so its ends land on the
	/// middles of the first and last texels, filtered between four, and the
	/// multiple bounces put back.
	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		clippy::cast_precision_loss,
		reason = "places inside a sixty-four texel table"
	)]
	fn ambient_brdf(normal_dot_view: f32, f0: Vec3, roughness: f32) -> Vec3 {
		let side = crate::brdf::SIDE;
		let last = (side - 1) as f32;
		let table = crate::brdf::table();
		let texel = |across: u32, down: u32| {
			let at = usize::try_from((down * side + across) * 4).unwrap_or(usize::MAX);
			let half = |offset: usize| {
				let low = table.get(at + offset).copied().unwrap_or(0);
				let high = table.get(at + offset + 1).copied().unwrap_or(0);

				crate::post::half_into(u16::from_le_bytes([low, high]))
			};

			Vec2::new(half(0), half(2))
		};
		let x = normal_dot_view.clamp(0.0, 1.0) * last;
		let y = roughness.clamp(0.0, 1.0) * last;
		let (left, top) = (x.floor() as u32, y.floor() as u32);
		let (right, bottom) = ((left + 1).min(side - 1), (top + 1).min(side - 1));
		let (fx, fy) = (x - x.floor(), y - y.floor());
		let upper = texel(left, top).lerp(texel(right, top), fx);
		let lower = texel(left, bottom).lerp(texel(right, bottom), fx);
		let pair = upper.lerp(lower, fy);
		let once = f0 * pair.x + Vec3::splat(pair.y);
		let kept = (pair.x + pair.y).max(1.0e-4);

		once * (Vec3::ONE + f0 * (1.0 / kept - 1.0))
	}

	/// The light a face of the room's wall sends one way, in a world lit only
	/// by the light arriving from everywhere: `lit_at` in `shader.wgsl` with
	/// no sun reaching the face, no lamp and the flat side of its branch.
	fn wall_light(ambient: f32, towards: Vec3, metallic: f32) -> Vec3 {
		let normal_dot_view = Vec3::Z.dot(towards).max(1.0e-4);
		let f0 = Vec3::splat(0.04).lerp(RED, metallic);
		let diffuse = RED * (1.0 - metallic);
		let specular = ambient_brdf(normal_dot_view, f0, 0.8);

		(diffuse * (Vec3::ONE - specular).max(Vec3::ZERO) + specular) * ambient
	}

	/// One pixel of a readback at the picture's size.
	fn at(values: &[[f32; 4]], (column, row): (u32, u32)) -> [f32; 4] {
		values
			.get(usize::try_from(row * SIZE.0 + column).unwrap_or(usize::MAX))
			.copied()
			.expect("the pixel is inside the picture")
	}

	/// One texel of a readback at half the picture's size.
	fn texel_at(values: &[[f32; 4]], (column, row): (u32, u32)) -> [f32; 4] {
		values
			.get(usize::try_from(row * (SIZE.0 / 2) + column).unwrap_or(usize::MAX))
			.copied()
			.expect("the texel is inside the buffer")
	}

	/// Draws a frame and reads what the reflections found back.
	fn found(capture: &mut Capture, world: &mut World) -> Vec<[f32; 4]> {
		capture.draw(world, &mut []);

		capture
			.scene_mut()
			.reflection_values()
			.expect("asked for, so written")
	}

	/// Every even pixel of the floor whose mirror reflection meets the wall.
	fn mirror_pixels(world: &World) -> Vec<((u32, u32), Vec3, Vec3)> {
		(0..SIZE.1 / 2)
			.flat_map(|row| (0..SIZE.0 / 2).map(move |column| (column * 2, row * 2)))
			.filter_map(|pixel| mirrored(world, pixel).map(|(floor, wall)| (pixel, floor, wall)))
			.collect()
	}

	/// Whether two numbers are the same to within an allowance.
	fn near_to(one: f32, other: f32, within: f32) -> bool { (one - other).abs() <= within }

	/// Whether a texel holds nothing at all: no light and no share.
	fn nothing(texel: [f32; 4]) -> bool {
		texel
			.iter()
			.all(|value| value.abs() < f32::EPSILON)
	}

	#[test]
	fn a_mirror_floor_finds_the_wall_on_it_lit_as_the_wall_is_lit_from_the_floor() {
		let Some(mut capture) = capture() else {
			return;
		};

		for metallic in [0.0, 1.0] {
			let mut world = room(0.045, metallic);

			asking(&mut world, "1", "5");

			let values = found(&mut capture, &mut world);
			let pixels = mirror_pixels(&world);
			let mut worst = 0.0_f32;

			assert!(pixels.len() > 400, "the room shows a band of floor that reflects the wall");

			for (pixel, floor, wall) in &pixels {
				let [r, g, b, a] = at(&values, *pixel);
				let wanted = wall_light(0.8, (*floor - *wall).normalize(), metallic);
				let off = (Vec3::new(r, g, b) - wanted).abs().max_element();

				worst = worst.max(off);

				assert!(
					near_to(a, 1.0, STEP),
					"the floor at {pixel:?} found the wall, all of its reflection: {a}"
				);
				assert!(
					off <= LIGHT_WITHIN,
					"{metallic} metal, the floor at {pixel:?} found {r} {g} {b} where the wall \
					 sends {wanted} towards it"
				);
			}

			assert!(worst > 0.0, "a comparison that is never off is not comparing anything");
		}
	}

	#[test]
	fn a_wall_seen_in_the_floor_is_lit_towards_the_floor_and_not_towards_the_eye() {
		// the one thing reading the picture's own color cannot do: a metal wall
		// seen at a grazing angle from the floor at its foot sends back more
		// than it sends an eye looking down at it, and what is found has to be
		// the first. From high above and close to the wall, so that the floor
		// at its foot reflects steeply up it
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room(0.045, 1.0);

		world.camera.position = Vec3::new(0.0, 6.0, 1.0);
		world.camera.target = Vec3::new(0.0, 0.0, WALL + 0.6);

		asking(&mut world, "1", "5");

		let values = found(&mut capture, &mut world);
		let eye = world.render_camera().position;
		let moved = mirror_pixels(&world)
			.into_iter()
			.filter(|(pixel, floor, wall)| {
				let [r, g, b, _] = at(&values, *pixel);
				let towards_floor = wall_light(0.8, (*floor - *wall).normalize(), 1.0);
				let towards_eye = wall_light(0.8, (eye - *wall).normalize(), 1.0);
				let got = Vec3::new(r, g, b);

				(towards_eye - towards_floor).abs().max_element() > 4.0 * LIGHT_WITHIN
					&& (got - towards_floor).abs().max_element() <= LIGHT_WITHIN
			})
			.count();

		assert!(
			moved > 100,
			"only {moved} pixels tell the two directions apart and found the right one"
		);
	}

	#[test]
	fn a_surface_past_the_cutoff_follows_nothing_and_one_short_of_it_finds_a_share() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the number both shaders hold, written down again here
		assert!(
			include_str!("shader.wgsl")
				.contains(&format!("const MIRROR_CUTOFF: f32 = {CUTOFF};"))
				&& include_str!("reflection.wgsl")
					.contains(&format!("const CUTOFF: f32 = {CUTOFF};")),
			"the cutoff here and in both shaders are the same number"
		);

		// a little past it rather than at it: the roughness is read back out of
		// a sixteen-bit float the device truncates into, and the cutoff itself
		// comes back a hair under the cutoff
		let mut world = room(CUTOFF + 0.02, 0.0);

		asking(&mut world, "1", "6");

		let values = found(&mut capture, &mut world);

		assert!(
			values.iter().all(|texel| texel
				.iter()
				.all(|value| value.abs() < f32::EPSILON)),
			"past the cutoff nothing is followed and nothing is found anywhere"
		);

		let mut world = room(0.55, 0.0);

		asking(&mut world, "1", "6");
		capture.draw(&mut world, &mut []);

		let raw = capture
			.scene_mut()
			.reflection_raw_values()
			.expect("asked for, so written");
		let surfaces = capture
			.scene_mut()
			.surface_values()
			.expect("the pass before the scene ran");
		let mut shares = 0;

		for (pixel, ..) in mirror_pixels(&world) {
			let [.., a] = texel_at(&raw, (pixel.0 / 2, pixel.1 / 2));
			let roughness = at(&surfaces, pixel)[3];
			let t = ((roughness - 0.5) / 0.1).clamp(0.0, 1.0);
			let fade = (t * t).mul_add(2.0_f32.mul_add(t, -3.0), 1.0);

			// a direction the lobe draws that misses the wall finds nothing,
			// which a surface this rough has some of
			if a > 0.0 {
				shares += 1;

				assert!(
					near_to(a, fade, STEP),
					"a texel that found the wall stands for {a} of its reflection, where the \
					 fade at a roughness of {roughness} is {fade}"
				);
			}
		}

		assert!(shares > 200, "only {shares} texels found the wall to be faded");
	}

	#[test]
	fn a_reflection_that_meets_nothing_on_the_picture_finds_nothing() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room_of(0.045, 0.0, 1.5);

		asking(&mut world, "1", "5");

		let values = found(&mut capture, &mut world);
		// a floor in front of the wall whose reflection passes over its top with
		// a margin, into a sky nothing is drawn in
		let over: Vec<(u32, u32)> = (0..SIZE.1 / 2)
			.flat_map(|row| (0..SIZE.0 / 2).map(move |column| (column * 2, row * 2)))
			.filter(|pixel| {
				let (near, way) = ray(&world, *pixel);
				let floor = near + way * (-near.y / way.y);
				let back = Vec3::new(way.x, -way.y, way.z);
				let wall = floor + back * ((WALL - floor.z) / back.z);

				way.y < 0.0 && floor.z > WALL + 0.05 && back.z < 0.0 && wall.y > 1.8
			})
			.collect();

		for pixel in &over {
			assert!(
				nothing(at(&values, *pixel)),
				"the floor at {pixel:?} reflects the empty sky over the wall and found {:?}",
				at(&values, *pixel)
			);
		}

		assert!(
			over.len() > 400,
			"only {} pixels of floor reflect the sky over the wall",
			over.len()
		);
	}

	#[test]
	fn a_rough_texel_is_the_average_of_the_texels_around_it_on_its_plane() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room(0.35, 0.0);

		asking(&mut world, "1", "5");
		capture.draw(&mut world, &mut []);

		let raw = capture
			.scene_mut()
			.reflection_raw_values()
			.expect("asked for, so written");
		let values = capture
			.scene_mut()
			.reflection_values()
			.expect("asked for, so written");
		let mut checked = 0;

		for (pixel, ..) in mirror_pixels(&world) {
			let (column, row) = (pixel.0 / 2, pixel.1 / 2);

			if column == 0 || row == 0 || column >= SIZE.0 / 2 - 1 || row >= SIZE.1 / 2 - 1 {
				continue;
			}

			// a floor's three by three are all on the floor whatever the
			// reflection found in them, so all nine count, each at a whole
			let total = [-1, 0, 1]
				.into_iter()
				.flat_map(|down| [-1, 0, 1].map(|across| (across, down)))
				.map(|(across, down)| {
					texel_at(
						&raw,
						(column.saturating_add_signed(across), row.saturating_add_signed(down)),
					)
				})
				.fold([0.0_f32; 4], |sum, other| {
					core::array::from_fn(|channel| sum[channel] + other[channel] / 9.0)
				});

			let got = at(&values, pixel);

			checked += 1;

			for (channel, (value, wanted)) in got.iter().zip(total).enumerate() {
				assert!(
					near_to(*value, wanted, 2.0 * STEP),
					"channel {channel} of the texel at {pixel:?} is {value}, where the average \
					 of nine is {wanted}"
				);
			}
		}

		assert!(checked > 200, "only {checked} texels were averaged");
	}

	#[test]
	fn a_pixel_between_two_texels_of_one_surface_is_the_blend_of_the_two() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room(0.045, 0.0);

		asking(&mut world, "1", "5");
		capture.draw(&mut world, &mut []);

		let raw = capture
			.scene_mut()
			.reflection_raw_values()
			.expect("asked for, so written");
		let values = capture
			.scene_mut()
			.reflection_values()
			.expect("asked for, so written");
		let mut checked = 0;

		// the floor's pixels on both sides of the edge of what it found, where
		// the two texels are far enough apart for a blend to be told from a pick
		let floor = |pixel| {
			let (near, way) = ray(&world, pixel);

			way.y < 0.0 && (near + way * (-near.y / way.y)).z > WALL + 0.05
		};

		for pixel in (0..SIZE.1 / 2)
			.flat_map(|row| (0..SIZE.0 / 2 - 1).map(move |column| (column * 2, row * 2)))
			.filter(|pixel| floor(*pixel) && floor((pixel.0 + 2, pixel.1)))
		{
			let (column, row) = (pixel.0 / 2, pixel.1 / 2);
			let left = texel_at(&raw, (column, row));
			let right = texel_at(&raw, (column + 1, row));

			if (left[3] - right[3]).abs() < 0.5 {
				continue;
			}

			let even = at(&values, pixel);
			let odd = at(&values, (pixel.0 + 1, pixel.1));

			checked += 1;

			for channel in 0..4 {
				assert!(
					(even[channel] - left[channel]).abs() < f32::EPSILON,
					"an even pixel reads its own texel and nothing else, at {pixel:?}"
				);
				assert!(
					near_to(odd[channel], (left[channel] + right[channel]) * 0.5, STEP),
					"the pixel after {pixel:?} is halfway between its two texels"
				);
			}
		}

		assert!(
			checked > 50,
			"only {checked} pairs across the edge of what was found were checked"
		);
	}

	#[test]
	fn a_pixel_beside_another_surface_does_not_take_that_surface_s_reflection() {
		// a rough pillar in front of the wall follows nothing, and the mirror
		// floor beside it on the picture and at its foot found the wall: a pixel
		// of the pillar between its texel and the floor's reads the pillar's own
		// nothing rather than a blend with what the floor found. Four times,
		// with the pillar and the eye moved by about a pixel each time, because
		// a pixel blends the texel past it only when its own place is odd.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut beside = 0;
		let mut against = 0;

		for step in [0.0_f32, 1.0, 2.0, 3.0] {
			let mut world = room(0.045, 0.0);
			let pillar = made(&mut world, "test/pillar", 0.8, 0.0);

			world.camera.position.y = step.mul_add(0.004, world.camera.position.y);
			slab(
				&mut world,
				pillar,
				(Vec3::new(step.mul_add(0.012, 0.3), 1.0, 0.0), Vec3::new(0.3, 2.0, 0.3)),
				RED,
			);
			asking(&mut world, "1", "5");

			let values = found(&mut capture, &mut world);
			let surfaces = capture
				.scene_mut()
				.surface_values()
				.expect("the pass before the scene ran");

			for (pixel, next) in edges(&surfaces) {
				beside += 1;
				against += usize::from(!nothing(at(&values, next)));

				assert!(
					nothing(at(&values, pixel)),
					"with the pillar moved {step} times, the rough face at {pixel:?} beside the 					 floor took {:?}",
					at(&values, pixel)
				);
			}
		}

		assert!(beside > 400, "only {beside} pixels of a rough face stand beside the floor");
		assert!(against > 200, "only {against} of them stand beside floor that found something");
	}

	#[test]
	fn a_mirror_s_pixel_beside_another_mirror_blends_only_its_own_surface() {
		// a mirror box standing on the mirror floor: its face turned to the eye
		// sends its reflection back at the eye and finds nothing, and the floor
		// around its foot found the wall. A pixel of the face is a blend of the
		// face's own texels among its four, whatever the floor's hold
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room(0.045, 0.0);
		let chrome = made(&mut world, "test/chrome", 0.045, 1.0);

		slab(
			&mut world,
			chrome,
			(Vec3::new(0.2, 0.4, -0.5), Vec3::new(1.0, 0.8, 0.6)),
			Vec3::ONE,
		);
		asking(&mut world, "1", "5");

		let values = found(&mut capture, &mut world);
		let raw = capture
			.scene_mut()
			.reflection_raw_values()
			.expect("asked for, so written");
		let surfaces = capture
			.scene_mut()
			.surface_values()
			.expect("the pass before the scene ran");
		let face = |pixel| {
			let texel = at(&surfaces, pixel);

			texel[2] > 0.9 && texel[3] < 0.1
		};
		let mut tellable = 0;

		for (column, row) in (0..SIZE.1 - 2)
			.flat_map(|row| (0..SIZE.0 - 2).map(move |column| (column, row)))
			.filter(|pixel| face(*pixel))
		{
			tellable += usize::from(blends_own(&values, &raw, (column, row), face));
		}

		assert!(
			tellable > 20,
			"only {tellable} pixels of the face have a texel of the floor to take"
		);
	}

	#[test]
	fn a_rough_patch_on_a_mirror_finds_nothing_even_beside_what_the_mirror_found() {
		// a decal too rough to follow anything, thrown onto the mirror floor where
		// the floor finds the wall: its texels lie on the floor's plane and face
		// the floor's way, so only the cutoff keeps the average from taking what
		// the mirror around it found
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room(0.045, 0.0);
		let scuff = made(&mut world, "test/scuff", 0.9, 0.0);
		let decal = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, 0.0, -1.0),
			rotation: Quat::from_rotation_x(-core::f32::consts::FRAC_PI_2),
			scale: Vec3::new(1.5, 1.5, 1.0),
		});

		world
			.entities
			.set_renderable(decal, Renderable { material: scuff, ..Renderable::NOTHING });
		world.entities.set_decal(decal, Decal::BOX);
		asking(&mut world, "1", "5");

		let values = found(&mut capture, &mut world);
		let raw = capture
			.scene_mut()
			.reflection_raw_values()
			.expect("asked for, so written");
		let surfaces = capture
			.scene_mut()
			.surface_values()
			.expect("the pass before the scene ran");
		let mirror = |pixel| {
			let texel = at(&surfaces, pixel);

			texel[1] > 0.9 && texel[3] < 0.1
		};
		let mut rough = 0;
		let mut beside = 0;
		let mut tellable = 0;

		for (column, row) in
			(2..SIZE.1 - 2).flat_map(|row| (2..SIZE.0 - 2).map(move |column| (column, row)))
		{
			let texel = at(&surfaces, (column, row));

			if texel[1] < 0.9 || texel[3] < CUTOFF {
				continue;
			}

			rough += 1;

			let around =
				[(column - 2, row), (column + 2, row), (column, row - 2), (column, row + 2)];

			beside += usize::from(
				around
					.into_iter()
					.any(|other| mirror(other) && !nothing(at(&values, other))),
			);
			// and the mirror beside the patch blends its own texels only, not the
			// patch's nothing
			tellable += around
				.into_iter()
				.filter(|other| mirror(*other))
				.map(|other| usize::from(blends_own(&values, &raw, other, mirror)))
				.sum::<usize>();

			assert!(
				nothing(at(&values, (column, row))),
				"the rough patch at ({column}, {row}) took {:?}",
				at(&values, (column, row))
			);
		}

		assert!(rough > 400, "only {rough} pixels of the floor were painted rough");
		assert!(beside > 50, "only {beside} of them are beside mirror that found something");
		assert!(
			tellable > 50,
			"only {tellable} pixels of mirror beside it have a texel of the patch to take"
		);
	}

	/// Whether a pixel of the finished buffer is a blend of the texels among
	/// its four that are its own surface, asserted, and whether one of the
	/// others holds something far enough outside their range for a blend with
	/// it to be told apart.
	///
	/// @param values - the finished buffer, at the picture's size
	/// @param raw - what the first pass wrote, which is what an unaveraged
	/// mirror's texel holds
	/// @param own - whether a pixel at a texel's place is the pixel's own
	/// surface
	fn blends_own<F>(
		values: &[[f32; 4]],
		raw: &[[f32; 4]],
		(column, row): (u32, u32),
		own: F,
	) -> bool
	where
		F: Fn((u32, u32)) -> bool,
	{
		let base = (column / 2, row / 2);
		let (mine, others): (Vec<_>, Vec<_>) = [(0, 0), (1, 0), (0, 1), (1, 1)]
			.map(|(across, down)| {
				((base.0 + across).min(SIZE.0 / 2 - 1), (base.1 + down).min(SIZE.1 / 2 - 1))
			})
			.into_iter()
			.partition(|texel| own((texel.0 * 2, texel.1 * 2)));

		if mine.is_empty() {
			return false;
		}

		let range = |channel: usize| {
			mine.iter()
				.map(|texel| texel_at(raw, *texel)[channel])
				.fold((f32::MAX, f32::MIN), |(low, high), value| {
					(low.min(value), high.max(value))
				})
		};
		let got = at(values, (column, row));

		for (channel, value) in got.iter().enumerate() {
			let (low, high) = range(channel);

			assert!(
				*value >= low - STEP && *value <= high + STEP,
				"channel {channel} at ({column}, {row}) is {value}, outside its own texels' \
				 {low}..{high}"
			);
		}

		others.iter().any(|texel| {
			(0..4).any(|channel| {
				let (low, high) = range(channel);
				let theirs = texel_at(raw, *texel)[channel];

				theirs < low - 0.1 || theirs > high + 0.1
			})
		})
	}

	/// A pixel's column and row.
	type Pixel = (u32, u32);

	/// Every pixel of a rough face turned towards the eye with a pixel of the
	/// floor beside it, and that pixel of the floor.
	fn edges(surfaces: &[[f32; 4]]) -> Vec<(Pixel, Pixel)> {
		let floor = |pixel| at(surfaces, pixel)[1] > 0.9;
		let rough_face = |pixel| {
			let texel = at(surfaces, pixel);

			texel[2] > 0.9 && texel[3] >= CUTOFF
		};

		(1..SIZE.1 - 1)
			.flat_map(|row| (1..SIZE.0 - 1).map(move |column| (column, row)))
			.filter(|pixel| rough_face(*pixel))
			.filter_map(|(column, row)| {
				[(column - 1, row), (column + 1, row), (column, row - 1), (column, row + 1)]
					.into_iter()
					.find(|other| floor(*other))
					.map(|next| ((column, row), next))
			})
			.collect()
	}

	#[test]
	fn what_is_found_is_the_same_at_one_sample_and_at_four() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room(0.35, 1.0);

		asking(&mut world, "1", "5");

		let one = found(&mut capture, &mut world);

		asking(&mut world, "4", "5");

		let four = found(&mut capture, &mut world);

		assert!(one == four, "the pass before the scene is one sample a pixel at every count");
	}

	#[test]
	fn a_picture_drawn_into_a_rectangle_finds_what_the_same_picture_finds_drawn_alone() {
		// the rectangle is what the pixels are measured across: a picture drawn
		// into the middle of a window by a tool around it reflects what the
		// same picture drawn alone reflects, texel for texel
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut whole = Capture::new(gpu, 240, 180).expect("the capture builds");
		let mut framed = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = room(0.35, 0.0);

		asking(&mut world, "1", "5");
		whole.draw(&mut world, &mut []);

		let alone = whole
			.scene_mut()
			.reflection_values()
			.expect("asked for, so written");

		framed.draw_within(&mut world, Viewport { x: 40, y: 30, width: 240, height: 180 });

		let within = framed
			.scene_mut()
			.reflection_values()
			.expect("asked for, so written");
		let mut lit = 0;

		// to a step of the buffer rather than to the bit: under one graphics API
		// the two are equal to the bit, and under the other a pixel's place
		// measured across a rectangle that does not start at nought rounds a
		// channel of one pixel a step apart
		for (column, row) in (0..180).flat_map(|row| (0..240).map(move |column| (column, row))) {
			let one = alone
				.get(usize::try_from(row * 240 + column).unwrap_or(usize::MAX))
				.copied()
				.expect("the pixel is inside the picture drawn alone");
			let other = within
				.get(usize::try_from((row + 30) * 320 + column + 40).unwrap_or(usize::MAX))
				.copied()
				.expect("the pixel is inside the rectangle");

			assert!(
				one.iter()
					.zip(other)
					.all(|(value, framed)| near_to(*value, framed, STEP)),
				"the pixel ({column}, {row}) of the picture is {one:?} drawn alone and \
				 {other:?} in the rectangle"
			);
			lit += usize::from(one[3] > 0.0);
		}

		assert!(lit > 1000, "only {lit} pixels found anything to compare");
	}

	#[test]
	fn a_resized_picture_finds_what_a_picture_made_at_that_size_finds() {
		// the groups the passes read the pass before the scene through are kept
		// from frame to frame, and a picture of a new size is a pass before the
		// scene with new buffers: what is found after a resize is what a scene
		// made at the new size finds, not what the old buffers held
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut resized = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut fresh = Capture::new(gpu, 240, 180).expect("the capture builds");
		let mut world = room(0.35, 0.0);

		asking(&mut world, "1", "5");
		resized.draw(&mut world, &mut []);
		resized.scene_mut().resize(240, 180);
		resized.draw(&mut world, &mut []);
		fresh.draw(&mut world, &mut []);

		let after = resized
			.scene_mut()
			.reflection_values()
			.expect("asked for, so written");
		let made = fresh
			.scene_mut()
			.reflection_values()
			.expect("asked for, so written");

		assert_eq!(after.len(), made.len(), "the buffer is the new size");
		assert!(
			made.iter().filter(|texel| texel[3] > 0.0).count() > 1000,
			"the picture at the new size finds something to compare"
		);
		assert!(
			after.iter().zip(&made).all(|(one, other)| {
				one.iter()
					.zip(other)
					.all(|(value, wanted)| near_to(*value, *wanted, STEP))
			}),
			"what the resized scene found is what a scene made at that size found"
		);
	}

	#[test]
	fn the_passes_run_while_something_asks_and_let_their_buffers_go_after() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room(0.045, 0.0);

		asking(&mut world, "1", "0");
		capture.draw(&mut world, &mut []);

		let quiet = capture.scene_mut().spans().passes();
		let first = capture.scene_mut().reflection_epoch();

		assert!(
			capture.scene_mut().reflection_values().is_none(),
			"a frame nobody asks in makes no buffer"
		);

		asking(&mut world, "1", "5");
		capture.draw(&mut world, &mut []);

		assert!(capture.scene_mut().reflection_values().is_some(), "the view asks");
		assert_eq!(
			capture.scene_mut().spans().passes(),
			quiet + 4,
			"and it costs the pass before the scene and three passes, drawn in the picture's \
			 place"
		);

		asking(&mut world, "1", "6");
		capture.draw(&mut world, &mut []);

		let kept = capture.scene_mut().reflection_epoch();

		assert_eq!(kept, first + 1, "a second frame that asks keeps the buffers the first made");

		asking(&mut world, "1", "0");
		capture.draw(&mut world, &mut []);

		assert!(
			capture.scene_mut().reflection_values().is_none(),
			"and a frame that stops asking lets them go"
		);
		assert_eq!(capture.scene_mut().reflection_epoch(), kept + 1, "which moves the epoch");
		assert_eq!(capture.scene_mut().spans().passes(), quiet, "and costs nothing again");
	}

	#[test]
	fn asking_for_the_reflections_moves_no_pixel_of_the_picture() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room(0.045, 1.0);

		world.light = Vec3::new(-0.4, -1.0, -0.5);

		for samples in ["4", "1"] {
			asking(&mut world, samples, "0");
			capture.scene_mut().reflect_anyway(false);

			let plain = capture
				.shoot(&mut world)
				.expect("the capture renders");

			capture.scene_mut().reflect_anyway(true);

			let asked = capture
				.shoot(&mut world)
				.expect("the capture renders");

			assert!(
				capture.scene_mut().reflection_values().is_some(),
				"the reflections were worked out in the second frame"
			);
			assert!(
				plain.pixels == asked.pixels,
				"at {samples} samples a frame with the reflections in it is the same picture"
			);
		}

		capture.scene_mut().reflect_anyway(false);
	}

	#[test]
	fn a_shader_put_in_after_the_first_pass_was_built_is_the_one_it_finds_with() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room(0.045, 0.0);

		asking(&mut world, "1", "5");
		capture.draw(&mut world, &mut []);

		let source = include_str!("shader.wgsl");
		let anchor = "return vec4<f32>(light * fade, fade);";

		assert_eq!(source.matches(anchor).count(), 1, "the line this test edits has moved");

		capture
			.scene_mut()
			.set_shader(&source.replace(anchor, "return vec4<f32>(0.25, 0.5, 0.75, 1.0);"))
			.expect("the edited shader compiles");

		let values = found(&mut capture, &mut world);
		let pixels = mirror_pixels(&world);

		assert!(!pixels.is_empty(), "the room shows floor that reflects the wall");

		for (pixel, ..) in pixels {
			let [r, g, b, a] = at(&values, pixel);

			assert!(
				near_to(r, 0.25, STEP)
					&& near_to(g, 0.5, STEP)
					&& near_to(b, 0.75, STEP)
					&& near_to(a, 1.0, STEP),
				"the floor at {pixel:?} found with the shader put in: {r} {g} {b} {a}"
			);
		}
	}

	#[test]
	fn a_pane_of_glass_between_the_floor_and_the_wall_is_not_met() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room(0.045, 0.0);

		asking(&mut world, "1", "5");

		let open = found(&mut capture, &mut world);
		let glass = world.materials.insert("test/glass", Material {
			blend: Blend::Alpha,
			opacity: 0.35,
			roughness: 0.1,
			..Material::DEFAULT
		});

		slab(
			&mut world,
			glass,
			(Vec3::new(0.0, 1.0, WALL + 1.0), Vec3::new(8.0, 2.0, 0.05)),
			Vec3::new(0.6, 0.8, 0.9),
		);

		let behind = found(&mut capture, &mut world);

		assert!(open == behind, "nothing blended is in the buffers a reflection is followed in");
	}

	#[test]
	fn a_mirror_facing_the_eye_follows_what_comes_back_to_the_near_plane_and_finds_nothing() {
		// the reflection of a wall square to the eye comes straight back at it,
		// which is cut short of the near plane rather than followed through it
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();

		world.camera.position = Vec3::new(0.0, 1.5, 3.0);
		world.camera.target = Vec3::new(0.0, 1.5, 0.0);
		flat(&mut world, 0.8);

		let mirror = made(&mut world, "test/mirror", 0.045, 1.0);

		slab(
			&mut world,
			mirror,
			(Vec3::new(0.0, 1.5, -0.25), Vec3::new(10.0, 10.0, 0.5)),
			Vec3::ONE,
		);
		asking(&mut world, "1", "5");

		let values = found(&mut capture, &mut world);

		assert!(
			values.iter().all(|texel| texel
				.iter()
				.all(|value| value.abs() < f32::EPSILON)),
			"nothing is behind the eye to find, and nothing non-finite is written"
		);
	}

	#[test]
	fn the_views_draw_a_color_and_a_share_as_bytes() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room(0.045, 0.0);
		let (pixel, floor, wall) = mirror_pixels(&world)
			.into_iter()
			.nth(40)
			.expect("the room shows floor that reflects the wall");
		let wanted = wall_light(0.8, (floor - wall).normalize(), 0.0);
		let byte = |level: f32| (level * 255.0).round();

		asking(&mut world, "1", "5");

		let light = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let seen = light.pixel(pixel.0, pixel.1);

		for (channel, level) in wanted.to_array().into_iter().enumerate() {
			assert!(
				(f32::from(seen[channel]) - byte(level)).abs() <= 1.0,
				"channel {channel} of what the floor found is {}, not {}",
				seen[channel],
				byte(level)
			);
		}

		asking(&mut world, "1", "6");

		let share = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert_eq!(share.pixel(pixel.0, pixel.1), [255, 255, 255, 255], "all of it was found");
		assert_eq!(share.pixel(160, 4), [0, 0, 0, 255], "and nothing where the sky is");

		asking(&mut world, "1", "4");

		let color = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let floor_seen = color.pixel(pixel.0, pixel.1);

		assert!(
			floor_seen[..3]
				.iter()
				.all(|value| value.abs_diff(77) <= 1),
			"the floor's color, 0.3, is the byte 77, and it is {floor_seen:?}"
		);
	}
}
