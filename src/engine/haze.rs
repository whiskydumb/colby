//! The light a haze sends towards the eye, and what it takes out of the light
//! behind it.
//!
//! **What the smear around the sun cannot do.** That one is a blur away from a
//! point on the screen, and what it can show is what the picture holds past the
//! middle of the view: a lamp in a shut room gives it nothing to smear, and a
//! sun beside the view or behind it gives it no point to smear from. A haze is
//! air: the sun and every lamp the frame carries light it, whatever stands
//! between a light and the air throws a shadow through it, and a wall in front
//! of the air ends the ray that would have crossed it. @ref
//! [`shaft`](crate::shaft).
//!
//! **Three passes, after the scene's.** The first follows one ray a texel at
//! half the picture on each axis, from the eye out to what the depth says is
//! there, and adds up what the sun sends back along it through the cascades and
//! what every lamp whose reach the ray crosses sends through its own maps; the
//! second averages each texel with the ones around it that stand for the same
//! air; the third puts the result over the picture at its whole size, dimmed by
//! what the air takes and lit by what it adds. @ref `fragment_haze` and
//! `fragment_haze_apply` in `shader.wgsl` for the first and the third, and
//! `haze.wgsl` for the second.
//!
//! **The sun lights the air of every hazy world it reaches**, because a world
//! has no sun that is off: its light is a direction, and white. A room the sun
//! must not light is a room shut against it, as it is for the room's walls.
//!
//! **No history, and that was measured rather than assumed.** The engines in
//! the field that keep a volume of cells over the view turn a history on inside
//! it, because a grid of cells sixteen pixels wide shows its cells without one,
//! and every picture this engine is checked with is one frame. On a model of
//! both at seven-twenty, against a march of five hundred and twelve places a
//! pixel, a grid of 80 by 45 by 64 cells with no history left 55,724 pixels of
//! a narrow beam past two levels, and one of 320 by 180 by 128 left 16,232,
//! where these three passes left 8,211.
//!
//! **What it does not reach**: anything blended, particles and the debug lines,
//! which the depth goes through - the air behind a pane of glass is added over
//! the glass; the air past `REACH`; and the shadow of a light that has none
//! there - a lamp with no map, or the sun past the shadow distance - whose
//! light crosses the air as it crosses walls.

use colby_core::{
	Result,
	abi::{Camera, World},
	bytemuck::{self, Pod, Zeroable},
	err, warn,
};
use wgpu::{
	BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
	BindGroupLayoutEntry, BindingResource, BindingType, BlendComponent, BlendFactor,
	BlendOperation, BlendState, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, Color,
	ColorTargetState, ColorWrites, CommandEncoder, Device, ErrorFilter, Extent3d, FragmentState,
	LoadOp, MultisampleState, Operations, PipelineCompilationOptions, PipelineLayoutDescriptor,
	PrimitiveState, Queue, RenderPass, RenderPassColorAttachment, RenderPassDescriptor,
	RenderPassTimestampWrites, RenderPipeline, RenderPipelineDescriptor, ShaderModule,
	ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, TextureDescriptor,
	TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
	VertexState,
};

use crate::{
	depth::Depth,
	post::HDR_FORMAT,
	prepass::Showing,
	scene::Viewport,
	shader::Shader,
	timing::{Ends, Pass, Timings},
};

/// How far from the eye the air goes on, in world units.
///
/// **Sixty-four, where the field puts it**: one engine's volume is sixty-four
/// units deep and another's sixty. Past it nothing is taken and nothing is
/// added, and a surface further off - the far plane, where nothing was drawn -
/// is dimmed by sixty-four units of air and lit by them. A number rather than
/// the far plane, because the far plane is two hundred units away by default
/// and a haze thick enough to see a lamp's light in across a room is thick
/// enough to take all of a sky that far off.
pub(crate) const REACH: f32 = 64.0;

/// The format both half-sized buffers are written in: rgb light.
///
/// Sixteen-bit floats, the picture's own, because the air in front of a lamp
/// is as bright as the lamp makes it.
pub(crate) const FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// How many lamps one texel's ray is followed through. `AIR_LAMPS` in
/// `shader.wgsl`, and a test says the two agree.
#[cfg(test)]
pub(crate) const LAMPS: usize = 4;

/// How far a texel's distance may be from another's and still stand for the
/// same air. `AIR_PLANE` in `shader.wgsl` and `PLANE` in `haze.wgsl`, and a
/// test says all three agree.
#[cfg(test)]
pub(crate) const PLANE: f32 = 0.05;

/// The numbers every pass reads, laid out the way `Air` in `shader.wgsl` and
/// `Tuning` in `haze.wgsl` declare them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct Tuning {
	/// `[the density, how far the air goes on, the projection's z_axis.z, its
	/// w_axis.z]`.
	medium: [f32; 4],

	/// `[the target's width, its height, unused, unused]`.
	size: [f32; 4],

	/// The rectangle of the target the picture is drawn into, as `[x, y,
	/// width, height]`.
	rect: [f32; 4],
}

/// What one frame asks the haze for.
///
/// Worked out in [`Scene::upload`](crate::Scene) beside the other effects' own,
/// for their reason: it needs the camera the frame is drawn from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Asking {
	/// How much of the light crossing a unit of air it scatters.
	density: f32,

	/// The projection's `z_axis.z` and `w_axis.z`.
	lens: [f32; 2],
}

/// What this frame asks for, if anything.
///
/// A world whose air is hazy asks, and so does the view that draws what the air
/// sends - at the world's own density, which is nought and draws nothing in a
/// world whose air is clear.
///
/// @param world - for the density and the aspect
/// @param camera - the camera this frame is drawn from
/// @param showing - what the view is asked to draw, if anything
#[must_use]
pub(crate) fn asking_of(
	world: &World,
	camera: &Camera,
	showing: Option<Showing>,
) -> Option<Asking> {
	if !world.post.is_hazy() && showing != Some(Showing::Haze) {
		return None;
	}

	let lens = camera.projection(world.aspect);

	Some(Asking {
		// a nan is not hazy and reads as none; a view asking in clear air asks
		// for a density of nought
		density: world.post.haze.max(0.0),
		lens: [lens.z_axis.z, lens.w_axis.z],
	})
}

/// The two half-sized buffers, and the group the average reads the first
/// through.
struct Buffers {
	/// Their size, half the picture's on each axis rounded up.
	half: (u32, u32),

	/// What the march writes.
	raw: TextureView,

	/// How the average reads it.
	raw_read: BindGroup,

	/// What the average writes, and what the last pass and the view read.
	averaged: TextureView,
}

/// The three passes, and what they write into.
pub(crate) struct Haze {
	/// The size of the picture, which the buffers are half of.
	size: (u32, u32),

	/// Both buffers, made the first frame something asks and let go the first
	/// frame nothing does.
	buffers: Option<Buffers>,

	/// Which making of [`buffers`](Self::buffers) a reader is looking at: moved
	/// when they are made and when they go.
	epoch: u64,

	/// How the march and the last pass read the tuning block, the depth and the
	/// averaged buffer, and which making of the depth's view and of the buffers
	/// the group is over. @ref [`Depth::epoch`].
	air: Option<(u64, u64, BindGroup)>,

	/// How the average reads the depth, kept the same way.
	depth_read: Option<(u64, BindGroup)>,

	/// The march and the last pass, built from the scene's own source the first
	/// frame something asks - both read the scene's frame, so both are the
	/// scene's shader - or nothing before then.
	built: Option<(RenderPipeline, RenderPipeline)>,

	tuning: Buffer,
	numbers: BindGroup,
	air_layout: BindGroupLayout,
	depth_layout: BindGroupLayout,
	source_layout: BindGroupLayout,
	average: RenderPipeline,
	device: Device,
}

impl Haze {
	/// Builds the pass that needs nothing of the scene's, and the block every
	/// pass reads its numbers from.
	///
	/// No buffer and no march yet: a frame that asks for nothing never makes
	/// either.
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
			label: Some("haze numbers"),
			entries: &[uniform(0)],
		});
		// the march's and the last pass's second group, at the bindings
		// `shader.wgsl` gives it: past the material's three and the
		// reflections' four, which the same group holds in other pipelines
		let air_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("haze air"),
			entries: &[uniform(7), crate::depth::entry(8), crate::prepass::entry(9)],
		});
		let depth_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("haze depth"),
			entries: &[crate::depth::entry(0)],
		});
		let source_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("haze source"),
			entries: &[crate::prepass::entry(0)],
		});
		let tuning = device.create_buffer(&BufferDescriptor {
			label: Some("haze tuning"),
			size: u64::try_from(size_of::<Tuning>())
				.map_err(|_| err!(Graphics("the haze tuning block does not fit a buffer")))?,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let numbers = device.create_bind_group(&BindGroupDescriptor {
			label: Some("haze numbers"),
			layout: &numbers_layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: tuning.as_entire_binding(),
			}],
		});
		// read through the shader directory like the scene's own: a variant
		// under `COLBY_SHADERS` reaches this file too
		let source = Shader::new("haze.wgsl", include_str!("haze.wgsl"));
		let scope = device.push_error_scope(ErrorFilter::Validation);
		let module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("haze"),
			source: ShaderSource::Wgsl(source.source().into()),
		});
		let average = average_pipeline(device, &module, &[
			Some(&numbers_layout),
			Some(&depth_layout),
			Some(&source_layout),
		]);

		if let Some(complaint) = pollster::block_on(scope.pop()) {
			return Err(err!(Graphics("the haze pipelines: {complaint}")));
		}

		Ok(Self {
			size: (width, height),
			buffers: None,
			epoch: 0,
			air: None,
			depth_read: None,
			built: None,
			tuning,
			numbers,
			air_layout,
			depth_layout,
			source_layout,
			average,
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

	/// Builds the march and the last pass, if they have not been built.
	///
	/// @param device - the device to build against
	/// @param scene - the scene's frame group layout and its shadow atlas's
	/// @param source - the WGSL the scene's own table was built from
	/// @return whether there are passes to record: nothing when they would not
	/// build, which is said
	pub(crate) fn ensure(
		&mut self,
		device: &Device,
		scene: [&BindGroupLayout; 2],
		source: &str,
	) -> bool {
		if self.built.is_some() {
			return true;
		}

		match build(device, scene, &self.air_layout, source) {
			| Ok(built) => {
				self.built = Some(built);

				true
			},
			| Err(complaint) => {
				warn!(%complaint, "the air is left clear this frame");

				false
			},
		}
	}

	/// The march and the last pass built again from new source, or nothing if
	/// they have never been built at all.
	///
	/// Asked by the scene when its shader changes, beside the scene's own
	/// table, so that all of them are replaced together or not at all: the
	/// light the air sends and the light a surface gets are one lamp's.
	///
	/// @param device - the device to build against
	/// @param scene - @ref [`ensure`](Self::ensure)
	/// @param source - the new WGSL
	pub(crate) fn rebuilt(
		&self,
		device: &Device,
		scene: [&BindGroupLayout; 2],
		source: &str,
	) -> Result<Option<(RenderPipeline, RenderPipeline)>> {
		if self.built.is_none() {
			return Ok(None);
		}

		build(device, scene, &self.air_layout, source).map(Some)
	}

	/// Puts what [`rebuilt`](Self::rebuilt) handed back in place, when it had
	/// something to give.
	pub(crate) fn replace(&mut self, built: Option<(RenderPipeline, RenderPipeline)>) {
		if built.is_some() {
			self.built = built;
		}
	}

	/// Records all three passes, or lets the buffers go.
	///
	/// @param encoder - the frame's, with the scene's pass and the depth made
	/// readable already in it
	/// @param queue - where the tuning block is written
	/// @param frame - what this frame asks and what the passes read and write
	/// @param rectangle - the part of the picture drawn into, or all of it
	pub(crate) fn render(
		&mut self,
		encoder: &mut CommandEncoder,
		queue: &Queue,
		frame: Frame<'_>,
		rectangle: Option<Viewport>,
	) {
		let (Some(asked), Some(depth), true) =
			(frame.asked, frame.depth.readable(), self.built.is_some())
		else {
			// the groups too: they hold the depth's view, which goes in the same
			// frame nothing asks for it
			self.let_go();
			self.air = None;
			self.depth_read = None;

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

		self.keep(frame.depth.epoch(), depth);

		let (Some((march, apply)), Some(buffers), Some((_, _, air)), Some((_, depth_read))) = (
			self.built.as_ref(),
			self.buffers.as_ref(),
			self.air.as_ref(),
			self.depth_read.as_ref(),
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
		let marks = |ends| frame.timings.writes(Pass::Haze, ends);

		screen_pass(
			encoder,
			("haze march", &buffers.raw, true),
			marks(Ends::Open),
			half,
			|pass| {
				pass.set_pipeline(march);
				pass.set_bind_group(0, frame.scene, &[]);
				pass.set_bind_group(1, air, &[]);
				pass.set_bind_group(2, frame.shadows, &[]);
			},
		);
		screen_pass(
			encoder,
			("haze average", &buffers.averaged, true),
			marks(Ends::Middle),
			half,
			|pass| {
				pass.set_pipeline(&self.average);
				pass.set_bind_group(0, &self.numbers, &[]);
				pass.set_bind_group(1, depth_read, &[]);
				pass.set_bind_group(2, &buffers.raw_read, &[]);
			},
		);
		// and over the picture, which is loaded rather than cleared because the
		// world is already in it
		screen_pass(
			encoder,
			("haze apply", frame.picture, false),
			marks(Ends::Close),
			whole,
			|pass| {
				pass.set_pipeline(apply);
				pass.set_bind_group(0, frame.scene, &[]);
				pass.set_bind_group(1, air, &[]);
			},
		);
	}

	/// Writes this frame's numbers into the block every pass reads.
	///
	/// @param queue - where it is written
	/// @param asked - what this frame asks for
	/// @param drawn - the rectangle of the target the picture is drawn into
	fn write(&self, queue: &Queue, asked: Asking, drawn: Viewport) {
		queue.write_buffer(
			&self.tuning,
			0,
			bytemuck::bytes_of(&tuning_of(asked, self.size, drawn)),
		);
	}

	/// Makes the groups that read the depth and the averaged buffer again, if
	/// either was made again since they were made.
	///
	/// Kept rather than made each frame, for the occlusion's reason: a group
	/// may be kept only while the views inside it are the ones that exist.
	///
	/// @param epoch - which making of the depth's view this is, @ref
	/// [`Depth::epoch`]
	/// @param depth - that view
	fn keep(&mut self, epoch: u64, depth: &TextureView) {
		let Some(buffers) = self.buffers.as_ref() else {
			return;
		};

		if self
			.air
			.as_ref()
			.is_none_or(|(seen, made, _)| *seen != epoch || *made != self.epoch)
		{
			let group = self
				.device
				.create_bind_group(&BindGroupDescriptor {
					label: Some("haze air"),
					layout: &self.air_layout,
					entries: &[
						BindGroupEntry {
							binding: 7,
							resource: self.tuning.as_entire_binding(),
						},
						BindGroupEntry {
							binding: 8,
							resource: BindingResource::TextureView(depth),
						},
						BindGroupEntry {
							binding: 9,
							resource: BindingResource::TextureView(&buffers.averaged),
						},
					],
				});

			self.air = Some((epoch, self.epoch, group));
		}

		if self
			.depth_read
			.as_ref()
			.is_none_or(|(seen, _)| *seen != epoch)
		{
			let group = self
				.device
				.create_bind_group(&BindGroupDescriptor {
					label: Some("haze depth"),
					layout: &self.depth_layout,
					entries: &[BindGroupEntry {
						binding: 0,
						resource: BindingResource::TextureView(depth),
					}],
				});

			self.depth_read = Some((epoch, group));
		}
	}

	/// What the view binds this frame: half the picture's size, rgb the light
	/// the air sends along each texel's ray, averaged, or nothing in a frame
	/// that did not ask.
	pub(crate) fn found(&self) -> Option<&TextureView> {
		self.buffers
			.as_ref()
			.map(|buffers| &buffers.averaged)
	}

	/// Which making of the buffers [`found`](Self::found) hands back.
	#[cfg(test)]
	pub(crate) const fn epoch(&self) -> u64 { self.epoch }

	/// Whether the march and the last pass have been built. A test's: a pass
	/// that would not build only says so, and a frame with the air left clear
	/// renders all the same.
	#[cfg(test)]
	pub(crate) const fn built(&self) -> bool { self.built.is_some() }

	/// What the average wrote, four floats a texel of the half-sized buffer. A
	/// test's.
	///
	/// @param device - to build the staging buffer on
	/// @param queue - to submit the copy on
	#[cfg(test)]
	pub(crate) fn values(&self, device: &Device, queue: &Queue) -> Option<Vec<[f32; 4]>> {
		crate::prepass::halves(device, queue, self.found()?)
	}

	/// What the march wrote, the same way. A test's.
	///
	/// @param device - to build the staging buffer on
	/// @param queue - to submit the copy on
	#[cfg(test)]
	pub(crate) fn raw_values(&self, device: &Device, queue: &Queue) -> Option<Vec<[f32; 4]>> {
		crate::prepass::halves(device, queue, &self.buffers.as_ref()?.raw)
	}
}

/// What a frame hands the passes: what it asks, and everything they read or
/// write that is the scene's.
#[derive(Clone, Copy)]
pub(crate) struct Frame<'a> {
	/// What this frame wants, or nothing.
	pub(crate) asked: Option<Asking>,

	/// The depth the scene wrote, made readable this frame.
	pub(crate) depth: &'a Depth,

	/// The scene's group nought: the frame's uniform with its lamps, and the
	/// environment the air's light from everywhere is read out of.
	pub(crate) scene: &'a BindGroup,

	/// The shadow atlas's group, which the march reads the cascades and every
	/// lamp's map out of.
	pub(crate) shadows: &'a BindGroup,

	/// The picture, which the last pass puts the air over.
	pub(crate) picture: &'a TextureView,

	/// What the passes write their marks into.
	pub(crate) timings: &'a Timings,
}

/// The block every pass reads, for one frame.
///
/// @param asked - what the frame asks for
/// @param (width, height) - the size of the whole target
/// @param drawn - the rectangle of it the picture is drawn into
fn tuning_of(asked: Asking, (width, height): (u32, u32), drawn: Viewport) -> Tuning {
	Tuning {
		medium: [asked.density, REACH, asked.lens[0], asked.lens[1]],
		size: [pixels(width), pixels(height), 0.0, 0.0],
		rect: [pixels(drawn.x), pixels(drawn.y), pixels(drawn.width), pixels(drawn.height)],
	}
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

/// The two half-sized buffers.
fn buffers(
	device: &Device,
	source_layout: &BindGroupLayout,
	(width, height): (u32, u32),
) -> Buffers {
	let half = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
	let target = |label| {
		device
			.create_texture(&TextureDescriptor {
				label: Some(label),
				size: Extent3d {
					width: half.0,
					height: half.1,
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
	let raw = target("haze raw");
	let averaged = target("haze averaged");
	let raw_read = device.create_bind_group(&BindGroupDescriptor {
		label: Some("haze raw"),
		layout: source_layout,
		entries: &[BindGroupEntry {
			binding: 0,
			resource: BindingResource::TextureView(&raw),
		}],
	});

	Buffers { half, raw, raw_read, averaged }
}

/// The copy bit a test reads a buffer back through: a test's and only a test's.
const fn copied() -> TextureUsages {
	if cfg!(test) {
		TextureUsages::COPY_SRC
	} else {
		TextureUsages::empty()
	}
}

/// One pass over one triangle covering its target.
///
/// A buffer of the air's own is cleared to nothing first, which is what a texel
/// outside the rectangle reads as; the picture is loaded, because the world is
/// already in it.
///
/// @param (label, view, cleared) - what the pass is called, what it writes, and
/// whether it clears it first
/// @param scissor - what to write, or nothing for a rectangle with nothing
/// inside the picture
fn screen_pass<F>(
	encoder: &mut CommandEncoder,
	(label, view, cleared): (&str, &TextureView, bool),
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
				load: if cleared {
					LoadOp::Clear(Color::TRANSPARENT)
				} else {
					LoadOp::Load
				},
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

/// The march and the last pass, against the scene's own source.
///
/// **The scene's group nought and a group one of their own**, and the march the
/// shadow atlas's group two as well: both read the frame's uniform - the lamps,
/// the camera, the environment - and bind no material, so the material's group
/// is where their own inputs go. The last pass reads no map, and a layout that
/// declared group two for it would be a group bound for nothing.
///
/// @param device - the device to build against
/// @param scene - the scene's frame group layout and its shadow atlas's
/// @param air_layout - the passes' own inputs
/// @param source - the whole WGSL
fn build(
	device: &Device,
	[frame, shadows]: [&BindGroupLayout; 2],
	air_layout: &BindGroupLayout,
	source: &str,
) -> Result<(RenderPipeline, RenderPipeline)> {
	let scope = device.push_error_scope(ErrorFilter::Validation);
	let module = device.create_shader_module(ShaderModuleDescriptor {
		label: Some("haze"),
		source: ShaderSource::Wgsl(source.into()),
	});
	let march = scene_pipeline(
		device,
		&module,
		("haze march", "fragment_haze"),
		&[Some(frame), Some(air_layout), Some(shadows)],
		ColorTargetState {
			format: FORMAT,
			blend: None,
			write_mask: ColorWrites::ALL,
		},
	);
	// what the air takes out of the light behind it is the alpha, and what it
	// adds is the color: `this + picture * (1 - alpha)`, with the picture's own
	// alpha left as it was
	let apply = scene_pipeline(
		device,
		&module,
		("haze apply", "fragment_haze_apply"),
		&[Some(frame), Some(air_layout)],
		ColorTargetState {
			format: HDR_FORMAT,
			blend: Some(BlendState {
				color: BlendComponent {
					src_factor: BlendFactor::One,
					dst_factor: BlendFactor::OneMinusSrcAlpha,
					operation: BlendOperation::Add,
				},
				alpha: BlendComponent::REPLACE,
			}),
			write_mask: ColorWrites::COLOR,
		},
	);

	match pollster::block_on(scope.pop()) {
		| Some(complaint) => Err(err!(Graphics("the haze passes: {complaint}"))),
		| None => Ok((march, apply)),
	}
}

/// One pipeline of the scene's own module over the sky's triangle, which
/// covers the target out of nothing but the vertex index.
///
/// @param (label, entry) - what it is called and which fragment entry point
/// @param groups - its layout
/// @param target - what it writes and how
fn scene_pipeline(
	device: &Device,
	module: &ShaderModule,
	(label, entry): (&str, &str),
	groups: &[Option<&BindGroupLayout>],
	target: ColorTargetState,
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
			// with no depth attached, where the triangle lies in depth does not
			// matter
			entry_point: Some("vertex_sky"),
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
			targets: &[Some(target)],
		}),
		multiview_mask: None,
		cache: None,
	})
}

/// The average's pipeline, writing a buffer of the air's format with nothing
/// blended and no depth.
fn average_pipeline(
	device: &Device,
	module: &ShaderModule,
	groups: &[Option<&BindGroupLayout>],
) -> RenderPipeline {
	let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some("haze average"),
		bind_group_layouts: groups,
		immediate_size: 0,
	});

	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some("haze average"),
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
			entry_point: Some("fragment_average"),
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
	use core::f32::consts::{FRAC_PI_2, PI};

	use colby_core::{
		abi::{
			Light, LightKind, Material, MaterialId, MeshId, Post, Renderable, Sky, ToneMap,
			Transform, Value, material::Blend,
		},
		glam::{Quat, Vec2, Vec3, Vec4},
	};

	use super::*;
	use crate::{Capture, depth, occlusion, prepass, scene::MSAA, shadow};

	/// How big every capture here is.
	const SIZE: (u32, u32) = (320, 240);

	/// The size of the buffers the air is worked out in, half of that on each
	/// axis.
	const HALF: (u32, u32) = (160, 120);

	/// How much of the light crossing a unit of air the air of these tests
	/// scatters.
	const HAZE: f32 = 0.08;

	/// How many places along a stretch of a ray a lamp is sampled at, written
	/// down again: `AIR_STEPS` in `shader.wgsl`.
	const STEPS: u8 = 16;

	/// How many places along a ray the sun is sampled at, written down again:
	/// `AIR_SUN_STEPS`.
	const SUN_STEPS: u8 = 16;

	/// The asymmetry of the lobe, written down again: `AIR_ASYMMETRY`.
	const ASYMMETRY: f32 = 0.2;

	/// The nearest a ray is counted as passing a lamp: `AIR_NEAREST`.
	const NEAREST: f32 = 0.01;

	/// How alike two texels' light is to be averaged: `CLOSE` in `haze.wgsl`.
	const CLOSE: f32 = 0.25;

	/// How near a neighbor may stand to the edge of what the average takes, in
	/// its distance or in its light, and be taken either way by the device: a
	/// share of the texel's distance, and of the brighter of the two lights.
	///
	/// **Worked out, not measured.** A brightness is three products added up,
	/// which a shader compiler may fuse into multiply-adds or add in another
	/// order; each of the five operations moves it by under an ulp, so where
	/// two brightnesses stand a quarter apart the test can land about eleven
	/// ulps of the larger either side, 1.3e-6. A distance is an exact add and
	/// one divide and lands closer. Three times that.
	///
	/// **A tie is not a chance in a million here**: the march writes sixteen
	/// bits, whose steps put two neighbors at exactly three quarters of each
	/// other in every channel often enough that the room holds two such pairs
	/// in a frame, and one of them one driver averaged and another did not.
	const EDGE: f32 = 4.0e-6;

	/// The order the places of the three by three tile are visited in, @ref
	/// `tile_at` in `shader.wgsl`.
	const ORDER: [u8; 9] = [0, 5, 7, 6, 1, 3, 4, 8, 2];

	/// The eight texels around one.
	const AROUND: [(i32, i32); 8] =
		[(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

	/// How many even steps the air along a ray is summed in, where the march is
	/// held against the arithmetic it stands in for.
	const SUMMED: u16 = 60_000;

	/// How far a light read back may be from the one worked out here, as a
	/// share of the larger of the two.
	///
	/// **Measured, then set at three times the worst seen**: 1.2e-3 under one
	/// graphics API and 9.7e-4 under the other, both in the cone's air. What is
	/// left is the sixteen-bit float the buffer holds, whose step is a share of
	/// the number rather than a distance, and which the device truncates into.
	const RELATIVE: f32 = 3.5e-3;

	/// And how far past that, whatever the light: what says a texel the
	/// arithmetic leaves dark holds nothing at all.
	const ABSOLUTE: f32 = 1.0e-6;

	/// How far what sixteen places at each of nine offsets add up to may be
	/// from the air summed in even steps, as a share of it: three times the
	/// worst of eight rays, 1.0e-3.
	const WITHIN_SUMMED: f64 = 3.0e-3;

	/// The same for the sun's places: three times the worst of five rays,
	/// 7.1e-3. Looser than a lamp's because a shadow across a ray is a step,
	/// and nine offsets of sixteen places find a step only to within a
	/// hundred and forty-fourth of the ray.
	const WITHIN_SUN: f64 = 2.1e-2;

	/// How much of the light crossing a unit of air the air scatters in the
	/// test that reads what it takes out of the picture: thin enough that a
	/// wall twenty units off keeps over half of its light.
	const THIN: f32 = 0.03;

	/// The light arriving from everywhere in that test.
	const AMBIENT: f32 = 0.6;

	/// Where the air ends in that test, written down again rather than read
	/// from `REACH`, so that a change to that number is a picture that moved.
	const ENDS: f32 = 64.0;

	/// A capture on the binary's one device, or `None` with no GPU.
	fn capture() -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, SIZE.0, SIZE.1) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// How many samples a pixel is drawn with and what the view draws, with no
	/// share of the sky taken away: nothing here is about the sky.
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

	/// A world with nothing in it yet and hazy air, looking from somewhere at
	/// something, the sun's light going one way through it: the picture's
	/// own look out of the way, and the light arriving from everywhere the only
	/// light besides the sun and the lamps.
	fn sunlit(eye: Vec3, at: Vec3, light: Vec3, ambient: f32) -> World {
		let mut world = World::new();

		world.camera.position = eye;
		world.camera.target = at;
		world.post = Post {
			tonemap: ToneMap::None,
			auto_exposure: false,
			exposure: 1.0,
			haze: HAZE,
			..Post::DEFAULT
		};
		world.sky = Sky::NONE;
		world.clear = Vec3::ZERO;
		world.light = light;
		world.ambient = Vec3::splat(ambient);

		world
	}

	/// The same with the sun kept out of the air, for what the lamps and the
	/// air itself do: a world has no sun that is off.
	///
	/// **A wall at the eye's back, and the sun's light coming at the eye from
	/// behind it**, so that the wall's shadow covers all of the air in front of
	/// the eye - with the shadows drawn out past where the air ends, because
	/// past the shadow distance the sun reaches everything. The eye looks away
	/// from the wall; the wall shades what the eye sees as well, which a test
	/// here only ever holds against the same world in clear air.
	fn looking(eye: Vec3, at: Vec3, ambient: f32) -> World {
		let mut world = sunlit(eye, at, Vec3::NEG_Z, ambient);
		let shade = made(&mut world, "test/shade", 1.0);

		slab(
			&mut world,
			shade,
			(eye + Vec3::new(0.0, 0.0, 1.25), Vec3::new(400.0, 400.0, 0.5)),
			Vec3::splat(0.5),
		);
		world
			.cvars
			.var(shadow::DISTANCE, Value::Float(shadow::DEFAULT_DISTANCE), "");
		world.cvars.set(shadow::DISTANCE, "128");

		world
	}

	/// A white material of a roughness.
	fn made(world: &mut World, name: &str, roughness: f32) -> MaterialId {
		world
			.materials
			.insert(name, Material { roughness, ..Material::DEFAULT })
	}

	/// A box of a material and a tint, standing somewhere.
	fn slab(
		world: &mut World,
		material: MaterialId,
		(position, scale): (Vec3, Vec3),
		tint: Vec3,
	) {
		let id = world.entities.spawn_at(Transform {
			position,
			rotation: Quat::IDENTITY,
			scale,
		});

		world
			.entities
			.set_renderable(id, Renderable::of(MeshId::CUBE, material, tint));
	}

	/// Lamps standing in a world, handed back the way a frame carries them.
	fn shining(world: &mut World, lights: &[(Light, Transform)]) -> Vec<Packed> {
		for (light, at) in lights {
			let id = world.entities.spawn_at(*at);

			world.entities.set_light(id, *light);
		}

		packed(lights, world.camera.position)
	}

	/// Where a lamp stands, turned to point straight down.
	fn downward(position: Vec3) -> Transform {
		Transform {
			position,
			rotation: Quat::from_rotation_x(-FRAC_PI_2),
			scale: Vec3::ONE,
		}
	}

	/// A point lamp that throws no shadow.
	fn bare(color: Vec3, intensity: f32, range: f32) -> Light {
		Light {
			shadow: false,
			..Light::point(color, intensity, range)
		}
	}

	/// A floor, a pillar standing on it and a lamp beside the pillar throwing
	/// its shadow through the air: edges in the depth, and edges in the light.
	fn room() -> World {
		let mut world = looking(Vec3::new(0.0, 1.5, 6.0), Vec3::new(0.0, 0.8, -2.0), 0.0);
		let floor = made(&mut world, "test/floor", 0.8);
		let pillar = made(&mut world, "test/pillar", 0.8);

		slab(
			&mut world,
			floor,
			(Vec3::new(0.0, -0.5, 0.0), Vec3::new(40.0, 1.0, 40.0)),
			Vec3::splat(0.3),
		);
		slab(
			&mut world,
			pillar,
			(Vec3::new(0.6, 1.0, -1.0), Vec3::new(0.4, 2.0, 0.4)),
			Vec3::splat(0.5),
		);
		shining(&mut world, &[(
			Light::point(Vec3::new(1.0, 0.9, 0.8), 6.0, 5.0),
			Transform::at(Vec3::new(-0.4, 1.2, -1.5)),
		)]);

		world
	}

	/// A wall filling the whole picture, seen square from six units away with
	/// its face at a place along the view, and a lamp in front of it.
	fn square_wall(face: f32) -> World {
		let mut world = looking(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO, 0.0);
		let wall = made(&mut world, "test/wall", 0.8);

		slab(
			&mut world,
			wall,
			(Vec3::new(0.0, 0.0, face - 0.25), Vec3::new(60.0, 60.0, 0.5)),
			Vec3::splat(0.5),
		);
		shining(&mut world, &[(
			Light::point(Vec3::ONE, 6.0, 4.0),
			Transform::at(Vec3::new(0.3, 0.2, 0.0)),
		)]);

		world
	}

	/// A lamp under a ceiling far wider than its reach, seen from a height.
	fn under_a_ceiling(height: f32, shadow: bool) -> World {
		let mut world = looking(Vec3::new(0.0, height, 9.0), Vec3::new(0.0, height, 0.0), 0.0);
		let ceiling = made(&mut world, "test/ceiling", 0.8);

		slab(
			&mut world,
			ceiling,
			(Vec3::new(0.0, 1.1, 0.0), Vec3::new(40.0, 0.2, 40.0)),
			Vec3::splat(0.5),
		);
		shining(&mut world, &[(
			Light {
				shadow,
				..Light::point(Vec3::ONE, 10.0, 6.0)
			},
			Transform::at(Vec3::ZERO),
		)]);

		world
	}

	/// A wall across the left half of the picture twenty units away, a dark
	/// wall across the top of the right half a hundred units away and past
	/// where the air ends, and nothing below that.
	fn distances() -> World {
		let mut world = looking(Vec3::ZERO, Vec3::NEG_Z, AMBIENT);
		let near = made(&mut world, "test/near", 1.0);
		let far = made(&mut world, "test/far", 1.0);

		slab(
			&mut world,
			near,
			(Vec3::new(-10.0, 0.0, -20.25), Vec3::new(20.0, 40.0, 0.5)),
			Vec3::splat(0.5),
		);
		slab(
			&mut world,
			far,
			(Vec3::new(40.0, 40.0, -100.25), Vec3::new(80.0, 80.0, 0.5)),
			Vec3::splat(0.05),
		);

		world
	}

	/// What one frame's passes left behind: what the march wrote, what the
	/// average wrote, and the depth both of them read.
	struct Found {
		raw: Vec<[f32; 4]>,
		averaged: Vec<[f32; 4]>,
		depth: Vec<f32>,
	}

	/// Draws a frame and reads back what it left.
	fn drawn(capture: &mut Capture, world: &mut World) -> Found {
		capture.draw(world, &mut []);

		read(capture)
	}

	/// Reads back what the last frame left.
	fn read(capture: &mut Capture) -> Found {
		let scene = capture.scene_mut();

		Found {
			raw: scene
				.haze_raw_values()
				.expect("asked for, so written"),
			averaged: scene
				.haze_values()
				.expect("asked for, so written"),
			depth: scene.depth_values().expect("read, so readable"),
		}
	}

	/// Every texel of the half-sized buffers, row by row.
	fn every_texel() -> impl Iterator<Item = (u32, u32)> {
		(0..HALF.1).flat_map(|row| (0..HALF.0).map(move |column| (column, row)))
	}

	/// Every pixel of the picture, row by row.
	fn every_pixel() -> impl Iterator<Item = (u32, u32)> {
		(0..SIZE.1).flat_map(|row| (0..SIZE.0).map(move |column| (column, row)))
	}

	/// One texel of a buffer at half the picture's size.
	fn texel_of(values: &[[f32; 4]], texel: (u32, u32)) -> [f32; 4] {
		wide(values, HALF.0, texel)
	}

	/// The light a texel holds.
	fn light_of([r, g, b, _]: [f32; 4]) -> Vec3 { Vec3::new(r, g, b) }

	/// What the depth holds at one pixel.
	fn stored(depth: &[f32], (column, row): (u32, u32)) -> f32 {
		depth
			.get(usize::try_from(row * SIZE.0 + column).unwrap_or(usize::MAX))
			.copied()
			.expect("the pixel is inside the picture")
	}

	/// A place as the float it is.
	fn float(value: u32) -> f32 {
		f32::from(u16::try_from(value).expect("a place inside a small picture"))
	}

	/// Whether a light read back out of a sixteen-bit buffer is the one worked
	/// out here. @ref [`RELATIVE`].
	fn near_light(got: Vec3, wanted: Vec3) -> bool {
		let larger = got.max_element().max(wanted.max_element());

		(got - wanted).abs().max_element() <= RELATIVE.mul_add(larger, ABSOLUTE)
	}

	/// Holds two buffers of air against each other, texel for texel.
	///
	/// @return how many texels of the first are lit
	fn same_light(one: &[[f32; 4]], other: &[[f32; 4]], what: &str) -> usize {
		assert_eq!(one.len(), other.len(), "{what} is the same size either way");

		let mut lit = 0;

		for (index, (mine, theirs)) in one.iter().zip(other).enumerate() {
			assert!(
				near_light(light_of(*mine), light_of(*theirs)),
				"{what} at texel {index} is {mine:?} one way and {theirs:?} the other"
			);
			lit += usize::from(mine[0] > 1.0e-3);
		}

		lit
	}

	/// One ray of the air, the way `AirRay` in `shader.wgsl` holds it.
	#[derive(Clone, Copy, Debug)]
	struct Ray {
		origin: Vec3,
		way: Vec3,
		far: f32,
	}

	/// Which way the eye sees through the middle of a pixel, and how far off
	/// what the depth holds there is: over the matrix the frame was drawn with
	/// and the depth it wrote, the way `air_ray` in `shader.wgsl` asks.
	fn seen_at(world: &World, depth: &[f32], pixel: (u32, u32)) -> (Vec3, f32) {
		let camera = world.render_camera();
		let inverse = camera.view_projection(world.aspect).inverse();
		let ndc = Vec2::new(
			((float(pixel.0) + 0.5) / float(SIZE.0)).mul_add(2.0, -1.0),
			((float(pixel.1) + 0.5) / float(SIZE.1)).mul_add(-2.0, 1.0),
		);
		let point = inverse * Vec4::new(ndc.x, ndc.y, stored(depth, pixel), 1.0);
		let seen = point.truncate() / point.w - camera.position;
		let distance = seen.length();

		(seen / distance.max(1.0e-6), distance)
	}

	/// The ray of the air through the middle of a pixel, out to what the depth
	/// holds there or to where the air ends.
	fn ray_at(world: &World, depth: &[f32], pixel: (u32, u32)) -> Ray {
		let (way, distance) = seen_at(world, depth, pixel);

		Ray {
			origin: world.render_camera().position,
			way,
			far: distance.min(REACH),
		}
	}

	/// One lamp the way the march reads it, packed the way the scene packs it
	/// and worked out again here.
	#[derive(Clone, Copy, Debug)]
	struct Packed {
		position: Vec3,
		range: f32,
		color: Vec3,
		/// The cone's scale and offset: nought and one for a point.
		cone: (f32, f32),
		direction: Vec3,
	}

	impl Packed {
		/// One light in the world.
		fn of(light: Light, at: Transform) -> Self {
			let cone = if light.kind == LightKind::Spot {
				let (inner, outer) = light.cone();
				let scale = (inner.cos() - outer.cos()).max(1.0e-4).recip();

				(scale, -outer.cos() * scale)
			} else {
				(0.0, 1.0)
			};

			Self {
				position: at.position,
				range: light.range,
				color: light.color * light.intensity,
				cone,
				direction: (at.rotation * Vec3::NEG_Z).normalize_or(Vec3::NEG_Z),
			}
		}

		/// Whether it is a point, which the shader asks of the cone's scale.
		fn is_point(&self) -> bool { self.cone.0.abs() < f32::MIN_POSITIVE }

		/// How much of its light reaches a point before any shadow, and the way
		/// that light travels: the falloff and the cone squared.
		fn reaching(&self, at: Vec3) -> (f32, Vec3) {
			let leaving = at - self.position;
			let distance_square = leaving.length_squared();
			let way = leaving * distance_square.max(1.0e-8).sqrt().recip();
			let cone = self
				.direction
				.dot(way)
				.mul_add(self.cone.0, self.cone.1)
				.clamp(0.0, 1.0);

			(falloff(distance_square, self.range * self.range) * cone * cone, way)
		}
	}

	/// The lamps of a world, packed and put in the order a frame carries them:
	/// by how far the near edge of each one's reach is from the eye.
	fn packed(lights: &[(Light, Transform)], eye: Vec3) -> Vec<Packed> {
		let near = |lamp: &Packed| (lamp.position - eye).length() - lamp.range;
		let mut lamps: Vec<Packed> = lights
			.iter()
			.map(|(light, at)| Packed::of(*light, *at))
			.collect();

		lamps.sort_by(|one, other| near(one).total_cmp(&near(other)));

		lamps
	}

	/// How much of a lamp survives the distance to a point: `lamp_falloff`.
	fn falloff(distance_square: f32, range_square: f32) -> f32 {
		let factor = distance_square / range_square.max(1.0e-4);
		let smoothed = factor.mul_add(-factor, 1.0).clamp(0.0, 1.0);

		smoothed * smoothed / distance_square.max(1.0e-4)
	}

	/// How much of what a unit of air scatters goes one way per unit of solid
	/// angle: `scattered` in `shader.wgsl`.
	fn lobe(cosine: f32) -> f32 {
		let g = ASYMMETRY;
		let denominator = (2.0 * g).mul_add(-cosine, g.mul_add(g, 1.0));

		g.mul_add(-g, 1.0) / (4.0 * PI * denominator * denominator.sqrt())
	}

	/// The stretch of a ray a lamp can light: `air_span`.
	fn span_of(lamp: &Packed, ray: Ray) -> (f32, f32) {
		let offset = ray.origin - lamp.position;
		let b = ray.way.dot(offset);
		let c = lamp
			.range
			.mul_add(-lamp.range, offset.length_squared());
		let discriminant = b.mul_add(b, -c);

		if discriminant <= 0.0 {
			return (0.0, 0.0);
		}

		let root = discriminant.sqrt();
		let near = (-b - root).max(0.0);
		let far = (root - b).min(ray.far);

		if far <= near {
			(0.0, 0.0)
		} else if lamp.is_point() {
			(near, far)
		} else {
			cone_span(lamp, ray, (near, far))
		}
	}

	/// The part of a stretch of a ray inside a cone's lit half: `air_cone`.
	fn cone_span(lamp: &Packed, ray: Ray, (start, end): (f32, f32)) -> (f32, f32) {
		let axis = lamp.direction;
		let edge = -lamp.cone.1 / lamp.cone.0;
		let edge_square = edge * edge;
		let offset = ray.origin - lamp.position;
		let way_along = ray.way.dot(axis);
		let offset_along = offset.dot(axis);
		let qa = way_along.mul_add(way_along, -edge_square);
		let qb = 2.0 * way_along.mul_add(offset_along, -(ray.way.dot(offset) * edge_square));
		let qc = offset_along.mul_add(offset_along, -(offset.length_squared() * edge_square));
		let discriminant = qb.mul_add(qb, -(4.0 * qa * qc));
		let mut cuts = [start, start, end, end];

		if discriminant > 0.0 && qa.abs() > 1.0e-8 {
			let root = discriminant.sqrt();
			let one = (-qb - root) / (2.0 * qa);
			let other = (root - qb) / (2.0 * qa);

			cuts[1] = one.min(other).clamp(start, end);
			cuts[2] = one.max(other).clamp(start, end);
		}

		cuts.windows(2)
			.map(|pair| (pair[0], pair[1]))
			.filter(|(from, to)| to > from)
			.find(|(from, to)| {
				let middle = offset + ray.way * ((from + to) * 0.5);

				middle.dot(axis) > edge * middle.length()
			})
			.unwrap_or((0.0, 0.0))
	}

	/// The light one lamp sends towards the eye out of one stretch of a ray:
	/// `air_lamp`, with no shadow.
	fn scattered(lamp: &Packed, ray: Ray, (start, end): (f32, f32), offset: f32) -> Vec3 {
		let to_lamp = lamp.position - ray.origin;
		let closest = to_lamp.dot(ray.way);
		let apart = (to_lamp - ray.way * closest)
			.length()
			.max(NEAREST);
		let first = ((start - closest) / apart).atan();
		let last = ((end - closest) / apart).atan();
		let total: f32 = (0..STEPS)
			.map(|step| {
				let share = (f32::from(step) + offset) / f32::from(STEPS);
				let along = apart.mul_add((last - first).mul_add(share, first).tan(), closest);
				let (reaching, leaving) = lamp.reaching(ray.origin + ray.way * along);
				let spread = apart.mul_add(apart, (along - closest) * (along - closest));

				reaching * lobe(leaving.dot(-ray.way)) * (-HAZE * along).exp() * spread
			})
			.sum();

		lamp.color * (total * HAZE * (last - first) / (f32::from(STEPS) * apart) * PI)
	}

	/// What a texel's ray gets from the lamps a frame carries, in its order:
	/// `fragment_haze`, with no lamp in any shadow.
	fn marched(lamps: &[Packed], ray: Ray, offset: f32) -> Vec3 {
		lamps
			.iter()
			.map(|lamp| (lamp, span_of(lamp, ray)))
			.filter(|(_, span)| span.1 > span.0)
			.take(LAMPS)
			.map(|(lamp, span)| scattered(lamp, ray, span, offset))
			.sum()
	}

	/// Where in its step each of a texel's places sits.
	fn offset_of((column, row): (u32, u32)) -> f32 {
		let place = usize::try_from((row % 3) * 3 + column % 3).unwrap_or(0);

		(f32::from(ORDER[place]) + 0.5) / 9.0
	}

	/// How far along a ray each place the sun is sampled at is, the way
	/// `air_sun` spreads them.
	fn sun_places(ray: Ray, offset: f32) -> impl Iterator<Item = f32> {
		(0..SUN_STEPS)
			.map(move |step| ray.far * ((f32::from(step) + offset) / f32::from(SUN_STEPS)))
	}

	/// What the sun sends towards the eye out of the air along a ray:
	/// `air_sun`, with how much of its light reaches a place answered by
	/// `reached` - or nothing, when that cannot be said of some place along
	/// it.
	fn sun_along<F>(world: &World, ray: Ray, offset: f32, reached: F) -> Option<Vec3>
	where
		F: Fn(Vec3) -> Option<f32>,
	{
		let haze = world.post.haze;
		let total = sun_places(ray, offset)
			.map(|along| {
				reached(ray.origin + ray.way * along).map(|share| share * (-haze * along).exp())
			})
			.sum::<Option<f32>>()?;
		let travel = world.light.normalize();

		Some(Vec3::splat(
			lobe(travel.dot(-ray.way)) * haze * ray.far * total / f32::from(SUN_STEPS) * PI,
		))
	}

	/// What the air along a ray sends towards the eye from the sun, summed in
	/// even steps over the whole ray: the arithmetic `sun_along` stands in for.
	///
	/// @param reached - how much of the sun's light gets to a distance along
	/// the ray
	fn summed_sun<F>(world: &World, ray: Ray, reached: F) -> f64
	where
		F: Fn(f32) -> f32,
	{
		let haze = world.post.haze;
		let step = ray.far / f32::from(SUMMED);
		let scale = f64::from(lobe(world.light.normalize().dot(-ray.way)) * haze * PI * step);

		(0..SUMMED)
			.map(|index| {
				let along = step * (f32::from(index) + 0.5);

				f64::from(reached(along) * (-haze * along).exp()) * scale
			})
			.sum()
	}

	/// Every texel of what the march wrote held against the sun and the lamps
	/// worked out here: the lamps in no shadow, the sun in whatever `reached`
	/// says, and a texel it cannot say it of left out.
	///
	/// @return how many texels were held, and how many were left out
	fn held_with_sun<F>(world: &World, found: &Found, lamps: &[Packed], reached: F) -> (u32, u32)
	where
		F: Fn(Vec3) -> Option<f32>,
	{
		let mut counted = (0, 0);

		for texel in every_texel() {
			let ray = ray_at(world, &found.depth, (texel.0 * 2, texel.1 * 2));
			let offset = offset_of(texel);
			let Some(sun) = sun_along(world, ray, offset, &reached) else {
				counted.1 += 1;

				continue;
			};
			let wanted = marched(lamps, ray, offset) + sun;
			let got = light_of(texel_of(&found.raw, texel));

			assert!(
				near_light(got, wanted),
				"the texel at {texel:?} holds {got} where the sun and the lamps add up to \
				 {wanted}"
			);

			counted.0 += 1;
		}

		counted
	}

	/// Whether the way from a point towards the sun passes through a box
	/// grown by a margin on every side, or shrunk by one under nought.
	fn shades(at: Vec3, towards: Vec3, (low, high): (Vec3, Vec3), margin: f32) -> bool {
		let inverse = towards.recip();
		let one = (low - Vec3::splat(margin) - at) * inverse;
		let other = (high + Vec3::splat(margin) - at) * inverse;
		let (enters, leaves) = (one.min(other).max_element(), one.max(other).min_element());

		enters <= leaves && leaves > 0.0
	}

	/// How much of the sun reaches a place a box may shadow, as far as a set of
	/// cascades can say it.
	///
	/// None of it where the box shadows the place even shrunk by three texels
	/// of the place's own cascade, all of it where the box grown by as much
	/// does not, and nothing said in between: a map places a shadow's edge
	/// only to within its texels. **Past the shadow distance all of it**,
	/// unless `beyond` says to ask the box there too - which is what a sun
	/// that stops at the distance would do.
	fn under(
		world: &World,
		cascades: &shadow::Cascades,
		caster: (Vec3, Vec3),
		beyond: bool,
	) -> impl Fn(Vec3) -> Option<f32> {
		let camera = world.render_camera();
		let (eye, forward) =
			(camera.position, (camera.target - camera.position).normalize_or(Vec3::NEG_Z));
		let towards = -world.light.normalize();
		let (splits, texels) = (cascades.splits, cascades.texels);

		move |at| {
			let depth = (at - eye).dot(forward);
			let last = splits[shadow::CASCADES - 1];

			if (depth - last).abs() < 1.0e-3 {
				return None;
			}

			if depth > last && !beyond {
				return Some(1.0);
			}

			let slice = splits
				.iter()
				.position(|end| depth <= *end)
				.unwrap_or(shadow::CASCADES - 1);
			let margin = 3.0 * texels[slice];

			match (shades(at, towards, caster, -margin), shades(at, towards, caster, margin)) {
				| (true, _) => Some(0.0),
				| (false, false) => Some(1.0),
				| (false, true) => None,
			}
		}
	}

	/// What the air along a ray sends towards the eye from one lamp, summed in
	/// even steps over the whole of the lamp's reach: no stretch worked out, no
	/// cone cut out of it and no angle, only the arithmetic the march stands in
	/// for.
	fn summed(lamp: &Packed, ray: Ray) -> [f64; 3] {
		let closest = (lamp.position - ray.origin).dot(ray.way);
		let start = (closest - lamp.range).max(0.0);
		let end = (closest + lamp.range).min(ray.far);
		let step = (end - start).max(0.0) / f32::from(SUMMED);
		let mut total = [0.0_f64; 3];

		for index in 0..SUMMED {
			let along = step.mul_add(f32::from(index) + 0.5, start);
			let (reaching, leaving) = lamp.reaching(ray.origin + ray.way * along);
			let share = reaching * lobe(leaving.dot(-ray.way)) * (-HAZE * along).exp();
			let light = lamp.color * (share * HAZE * PI * step);

			for (sum, channel) in total.iter_mut().zip(light.to_array()) {
				*sum += f64::from(channel);
			}
		}

		total
	}

	/// Every texel of what the march wrote held against the march worked out
	/// here, with no lamp's light in any shadow.
	///
	/// @return how many texels were lit and how many were left dark
	fn held_against(world: &World, found: &Found, lamps: &[Packed]) -> (u32, u32) {
		let mut counted = (0, 0);

		for texel in every_texel() {
			let ray = ray_at(world, &found.depth, (texel.0 * 2, texel.1 * 2));
			let wanted = marched(lamps, ray, offset_of(texel));
			let held = texel_of(&found.raw, texel);
			let got = light_of(held);

			assert!(
				near_light(got, wanted),
				"the texel at {texel:?} holds {got} where its places add up to {wanted}"
			);
			assert!(
				(held[3] - 1.0).abs() < f32::EPSILON,
				"and one in the fourth, not {}",
				held[3]
			);

			if wanted.max_element() > 0.0 {
				counted.0 += 1;
			} else {
				counted.1 += 1;
			}
		}

		counted
	}

	/// Whether a ray crosses a lamp's reach by a clear margin, misses it by
	/// one, or passes too near its edge to say.
	fn crossing(lamp: &Packed, ray: Ray) -> Option<bool> {
		let (start, end) = span_of(lamp, ray);
		let to_lamp = lamp.position - ray.origin;
		let apart = (to_lamp - ray.way * to_lamp.dot(ray.way)).length();

		if end - start > 0.05 {
			Some(true)
		} else if apart > lamp.range + 0.05 {
			Some(false)
		} else {
			None
		}
	}

	/// How bright a light is, by the weights `haze.wgsl` gives its channels.
	fn brightness(light: Vec3) -> f32 { light.dot(Vec3::new(0.2126, 0.7152, 0.0722)) }

	/// What the average makes of one texel, worked out from what the march
	/// wrote and how far along the view each texel's surface is.
	///
	/// @return every average the device may hand back - the first with each
	/// neighbor taken or turned away as worked out here, the rest with those
	/// on an edge taken the other way, @ref [`EDGE`] - and how many neighbors
	/// were turned away for being in front of another surface and how many for
	/// being lit too differently
	fn averaged_at<F>(
		raw: &[[f32; 4]],
		(column, row): (u32, u32),
		distance: F,
	) -> (Vec<Vec3>, (u32, u32))
	where
		F: Fn((u32, u32)) -> f32,
	{
		let own = light_of(texel_of(raw, (column, row)));
		let along = distance((column, row));
		let bright = brightness(own);
		let mut neighbors = Vec::new();
		let mut edges = 0_u32;
		let mut turned = (0, 0);

		for place in AROUND
			.into_iter()
			.filter_map(|offset| beside((column, row), offset, HALF))
		{
			let light = light_of(texel_of(raw, place));
			let theirs = brightness(light);
			let apart = (distance(place) - along).abs();
			let larger = theirs.max(bright);
			let near = apart <= PLANE * along;
			let alike = (theirs - bright).abs() <= CLOSE * larger;
			let near_edge = PLANE.mul_add(-along, apart).abs() < EDGE * along;
			let alike_edge = CLOSE
				.mul_add(-larger, (theirs - bright).abs())
				.abs() < EDGE * larger;

			turned.0 += u32::from(!near);
			turned.1 += u32::from(near && !alike);

			// an edge in one test matters only where the other test may take
			// the neighbor
			let on_edge =
				(near_edge && (alike || alike_edge)) || (alike_edge && (near || near_edge));
			let bit = on_edge.then(|| {
				edges += 1;
				edges - 1
			});

			neighbors.push((light, near && alike, bit));
		}

		// each number says which neighbors on an edge are taken the other way;
		// the sum is added up in the order the shader adds it
		let averages = (0..1_u32 << edges)
			.map(|flipped| {
				let (total, weight) = neighbors
					.iter()
					.filter(|(_, taken, bit)| {
						*taken != bit.is_some_and(|bit| flipped & (1 << bit) != 0)
					})
					.fold((own, 1.0), |(total, weight), (light, ..)| {
						(total + *light, weight + 1.0)
					});

				total / weight
			})
			.collect();

		(averages, turned)
	}

	/// A linear level as the byte an sRGB target keeps of it.
	fn byte_of(level: f32) -> f32 {
		let level = level.clamp(0.0, 1.0);
		let curved = if level <= 0.003_130_8 {
			level * 12.92
		} else {
			level.powf(2.4_f32.recip()).mul_add(1.055, -0.055)
		};

		(curved * 255.0).round()
	}

	/// The linear level a byte of an sRGB target stands for.
	fn undone(byte: u8) -> f32 {
		let level = f32::from(byte) / 255.0;

		if level <= 0.040_45 {
			level / 12.92
		} else {
			((level + 0.055) / 1.055).powf(2.4)
		}
	}

	/// Whether each channel of a pixel seen through the air is what the air
	/// makes of the same pixel seen through none, within a byte.
	///
	/// @param kept - the share of the light behind that the air leaves
	fn dimmed(before: [u8; 4], after: [u8; 4], kept: f32) -> bool {
		before.iter().zip(after).take(3).all(|(was, is)| {
			let wanted = byte_of(undone(*was).mul_add(kept, AMBIENT * (1.0 - kept)));

			(f32::from(is) - wanted).abs() <= 1.0
		})
	}

	/// What the last pass puts over a pixel of a picture every texel of which
	/// stands for the same air: the four texels around it, blended by where it
	/// lies between them.
	fn blown_up(averaged: &[[f32; 4]], (column, row): (u32, u32)) -> Vec3 {
		let light = |across: u32, down: u32| {
			light_of(texel_of(
				averaged,
				((column / 2 + across).min(HALF.0 - 1), (row / 2 + down).min(HALF.1 - 1)),
			))
		};
		let (x, y) = (float(column % 2) * 0.5, float(row % 2) * 0.5);
		let upper = light(0, 0) + (light(1, 0) - light(0, 0)) * x;
		let lower = light(0, 1) + (light(1, 1) - light(0, 1)) * x;

		upper + (lower - upper) * y
	}

	#[test]
	fn a_frame_asks_while_its_air_is_hazy_or_the_view_asks_and_never_for_less_than_nothing() {
		let mut world = World::new();

		world.aspect = 4.0 / 3.0;

		let camera = world.render_camera();
		let lens = camera.projection(world.aspect);

		assert!(
			asking_of(&world, &camera, None).is_none(),
			"clear air with no view asks nothing"
		);
		assert!(
			asking_of(&world, &camera, Some(Showing::Normal)).is_none(),
			"and neither does another view"
		);

		for haze in [0.0, -0.0, -1.0, f32::NAN, f32::NEG_INFINITY] {
			world.post.haze = haze;

			assert!(asking_of(&world, &camera, None).is_none(), "air of {haze} is clear");

			let viewed = asking_of(&world, &camera, Some(Showing::Haze))
				.expect("the view asks whatever the air is");

			assert!(
				viewed.density.abs() < f32::EPSILON,
				"and in air of {haze} it asks at a density of nought, not {}",
				viewed.density
			);
		}

		world.post.haze = 0.04;

		for showing in [None, Some(Showing::Haze), Some(Showing::Normal)] {
			let asked = asking_of(&world, &camera, showing).expect("hazy air asks");

			assert!((asked.density - 0.04).abs() < f32::EPSILON, "at its own density");
			assert!(
				(asked.lens[0] - lens.z_axis.z).abs() < f32::EPSILON
					&& (asked.lens[1] - lens.w_axis.z).abs() < f32::EPSILON,
				"through the camera's own lens, not {:?}",
				asked.lens
			);
		}
	}

	#[test]
	fn the_numbers_reach_the_block_in_the_order_both_shaders_read_them() {
		let mut world = World::new();

		world.aspect = 4.0 / 3.0;
		world.post.haze = 0.03;

		let camera = world.render_camera();
		let lens = camera.projection(world.aspect);
		let asked = asking_of(&world, &camera, None).expect("hazy air asks");
		let tuning = tuning_of(asked, SIZE, Viewport { x: 40, y: 30, width: 240, height: 180 });
		let floats =
			depth::floats(bytemuck::bytes_of(&tuning)).expect("the block is whole floats");
		let wanted = [
			0.03,
			REACH,
			lens.z_axis.z,
			lens.w_axis.z,
			320.0,
			240.0,
			0.0,
			0.0,
			40.0,
			30.0,
			240.0,
			180.0,
		];

		assert_eq!(floats.len(), wanted.len(), "three vectors of four and nothing else");

		for (at, (seen, meant)) in floats.iter().zip(wanted).enumerate() {
			assert!(
				(seen - meant).abs() < 1.0e-6,
				"the number at {at} reached the block as {seen}, not {meant}"
			);
		}

		for (source, name) in [
			(include_str!("shader.wgsl"), "struct Air"),
			(include_str!("haze.wgsl"), "struct Tuning"),
		] {
			let body = source
				.split(name)
				.nth(1)
				.and_then(|rest| rest.split("};").next())
				.expect("the block is declared");
			let places: Vec<Option<usize>> =
				["medium: vec4<f32>", "size: vec4<f32>", "rect: vec4<f32>"]
					.into_iter()
					.map(|field| body.find(field))
					.collect();

			assert!(
				places.iter().all(Option::is_some) && places.is_sorted(),
				"{name} declares the three vectors in the order they are written: {places:?}"
			);
		}
	}

	#[test]
	fn the_numbers_written_down_here_are_the_ones_the_shaders_hold() {
		let scene = include_str!("shader.wgsl");
		let average = include_str!("haze.wgsl");

		for (source, line) in [
			(scene, format!("const AIR_LAMPS: u32 = {LAMPS}u;")),
			(scene, format!("const AIR_PLANE: f32 = {PLANE};")),
			(average, format!("const PLANE: f32 = {PLANE};")),
			(scene, format!("const AIR_STEPS: u32 = {STEPS}u;")),
			(scene, format!("const AIR_SUN_STEPS: u32 = {SUN_STEPS}u;")),
			(scene, format!("const AIR_ASYMMETRY: f32 = {ASYMMETRY};")),
			(scene, format!("const AIR_NEAREST: f32 = {NEAREST};")),
			(average, format!("const CLOSE: f32 = {CLOSE};")),
		] {
			assert!(source.contains(&line), "a shader does not hold `{line}`");
		}
	}

	#[test]
	fn the_lobe_sends_all_that_the_air_scatters_somewhere_and_more_of_it_onwards() {
		let steps = 20_000_u16;
		let total = (0..steps)
			.map(|index| {
				let cosine = ((f32::from(index) + 0.5) / f32::from(steps)).mul_add(2.0, -1.0);

				f64::from(lobe(cosine)) * 2.0 / f64::from(steps)
			})
			.sum::<f64>()
			* 2.0 * core::f64::consts::PI;

		assert!((total - 1.0).abs() < 1.0e-4, "over every way there is it sends {total} of it");
		assert!(
			lobe(1.0) > lobe(0.0) && lobe(0.0) > lobe(-1.0),
			"and more of it carries on than turns back"
		);
	}

	#[test]
	fn the_places_along_a_ray_add_up_to_what_the_air_along_it_sends() {
		// **what says the march is worth holding the picture against.** Sixteen
		// places spread by angle, at each of the nine offsets the tile spreads
		// them by, against the same air summed in sixty thousand even steps
		// with no stretch and no angle at all
		let point = Packed::of(
			Light::point(Vec3::new(1.0, 0.8, 0.6), 8.0, 6.0),
			Transform::at(Vec3::ZERO),
		);
		let cone = Packed::of(
			Light::spot(Vec3::new(0.9, 0.9, 1.0), 10.0, 8.0, 0.2, 0.5),
			downward(Vec3::new(0.0, 3.0, 0.0)),
		);
		let along = |origin: Vec3, far: f32| Ray { origin, way: Vec3::NEG_Z, far };
		let towards = |origin: Vec3, to: Vec3| Ray {
			origin,
			way: (to - origin).normalize(),
			far: REACH,
		};
		let cases = [
			// passing a lamp two centimeters from its middle
			(point, along(Vec3::new(0.0, 0.02, 9.0), REACH)),
			(point, along(Vec3::new(0.0, 3.0, 9.0), REACH)),
			// ending on a surface level with the lamp
			(point, along(Vec3::new(0.0, 0.5, 9.0), 9.0)),
			// starting inside its reach
			(point, towards(Vec3::new(1.0, 1.0, 2.0), Vec3::new(-3.0, -1.0, -2.0))),
			(point, towards(Vec3::new(5.0, 4.0, 9.0), Vec3::new(-2.0, -1.0, -3.0))),
			// across a cone's axis, beside it, and up towards its lamp
			(cone, along(Vec3::new(0.0, 0.0, 9.0), REACH)),
			(cone, along(Vec3::new(0.5, 1.5, 9.0), REACH)),
			(cone, towards(Vec3::new(0.3, -2.0, 9.0), Vec3::new(0.3, 1.5, 0.0))),
		];
		for (lamp, ray) in cases {
			let span = span_of(&lamp, ray);

			assert!(span.1 > span.0, "the ray {ray:?} crosses the lamp's light");

			let places = (0..9_u8)
				.map(|place| scattered(&lamp, ray, span, (f32::from(place) + 0.5) / 9.0))
				.sum::<Vec3>()
				/ 9.0;

			for (got, wanted) in places
				.to_array()
				.into_iter()
				.zip(summed(&lamp, ray))
			{
				assert!(
					(f64::from(got) - wanted).abs() < WITHIN_SUMMED * wanted,
					"along {ray:?} the places add up to {got} where the air sends {wanted}"
				);
			}
		}

		// and a ray past the edge of a cone crosses none of its light
		let past = along(Vec3::new(2.0, 0.0, 9.0), REACH);
		let (start, end) = span_of(&cone, past);

		assert!(end <= start, "a ray past the edge of the cone has no stretch in it");
		assert!(
			summed(&cone, past)
				.iter()
				.all(|channel| *channel <= 0.0),
			"and there is none"
		);
	}

	#[test]
	fn a_lamp_in_open_air_lights_each_texel_by_what_the_places_along_its_ray_add_up_to() {
		// the march held against arithmetic at every texel: from outside the
		// lamp's reach, where the corners of the picture miss it, and from
		// inside it, where every ray starts in its light
		let Some(mut capture) = capture() else {
			return;
		};

		for (eye, dark) in [(Vec3::new(0.0, 0.0, 8.0), 5000), (Vec3::new(1.0, 0.5, 3.0), 0)] {
			let mut world = looking(eye, Vec3::new(0.0, 0.0, -1.0), 0.0);
			let lamps = shining(&mut world, &[(
				bare(Vec3::new(1.0, 0.8, 0.6), 8.0, 4.0),
				Transform::at(Vec3::ZERO),
			)]);

			asking(&mut world, "1", "0");

			let found = drawn(&mut capture, &mut world);
			let counted = held_against(&world, &found, &lamps);

			assert!(counted.0 > 10_000, "seen from {eye}, only {} texels were lit", counted.0);
			assert!(counted.1 >= dark, "and {} were left dark", counted.1);
		}
	}

	#[test]
	fn a_cone_lights_the_air_inside_its_edge_and_none_outside_it() {
		// and the cone throws a shadow, in a world with nothing to throw one:
		// what its map says of air in front of nothing is that all of it is lit
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = looking(Vec3::new(0.0, 1.0, 9.0), Vec3::new(0.0, 1.0, 0.0), 0.0);
		let lamps = shining(&mut world, &[(
			Light::spot(Vec3::new(0.9, 0.9, 1.0), 12.0, 9.0, 0.1, 0.3),
			downward(Vec3::new(0.0, 4.0, 0.0)),
		)]);

		asking(&mut world, "1", "0");

		let found = drawn(&mut capture, &mut world);
		let (lit, dark) = held_against(&world, &found, &lamps);

		assert!(lit > 2000, "only {lit} texels see air inside the cone");
		assert!(dark > 12_000, "and only {dark} see none of it");
	}

	#[test]
	fn a_ceiling_between_a_lamp_and_the_air_above_it_leaves_that_air_unlit() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut raw = |mut world: World| {
			asking(&mut world, "1", "0");
			capture.draw(&mut world, &mut []);
			capture
				.scene_mut()
				.haze_raw_values()
				.expect("asked for, so written")
		};

		// from above it, where every ray the eye sends crosses only air the
		// ceiling stands between the lamp and
		let through = raw(under_a_ceiling(3.0, false));
		let above = raw(under_a_ceiling(3.0, true));
		let crossed = through
			.iter()
			.filter(|texel| texel[0] > 1.0e-3)
			.count();

		assert!(
			crossed > 6000,
			"a lamp that throws no shadow lights the air past the ceiling in only {crossed} \
			 texels"
		);
		assert!(
			above
				.iter()
				.all(|texel| light_of(*texel).max_element() < ABSOLUTE),
			"and one that throws a shadow lights none of it"
		);

		// and from under it, where the ceiling stands between the lamp and none
		// of the air. **Not to the bit**, measured: 134 texels of 14,650 lit,
		// none more than 2.8% darker, the same under both graphics APIs
		let open = raw(under_a_ceiling(-2.0, false));
		let under = raw(under_a_ceiling(-2.0, true));
		let lit = open
			.iter()
			.filter(|texel| texel[0] > 1.0e-3)
			.count();
		let mut moved = 0;

		for (index, (one, other)) in open.iter().zip(&under).enumerate() {
			let (one, other) = (light_of(*one), light_of(*other));

			assert!(
				(one - other).abs().max_element() <= 0.08 * one.max_element(),
				"the air at texel {index} under the ceiling is {other} with its shadow and \
				 {one} without"
			);

			moved += usize::from((one - other).abs().max_element() > 0.0);
		}

		assert!(lit > 10_000, "only {lit} texels see the lamp's light under the ceiling");
		assert!(moved < 400, "and {moved} of them are lit differently with its shadow");
	}

	#[test]
	fn a_wall_ends_the_ray_and_the_air_behind_it_sends_nothing() {
		// a lamp behind the wall that throws no shadow, so that its light
		// crosses the wall into the air in front of it: what a texel on the
		// wall holds is that air and not the air behind the wall as well
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = looking(Vec3::new(0.0, 1.0, 8.0), Vec3::new(0.0, 1.0, 0.0), 0.0);
		let wall = made(&mut world, "test/wall", 0.8);

		slab(
			&mut world,
			wall,
			(Vec3::new(0.0, 0.0, -0.25), Vec3::new(60.0, 60.0, 0.5)),
			Vec3::splat(0.5),
		);

		let lamps = shining(&mut world, &[(
			bare(Vec3::ONE, 8.0, 6.0),
			Transform::at(Vec3::new(0.5, 0.5, -1.0)),
		)]);

		asking(&mut world, "1", "0");

		let found = drawn(&mut capture, &mut world);
		let (lit, _) = held_against(&world, &found, &lamps);
		let cut = every_texel()
			.filter(|texel| {
				let ray = ray_at(&world, &found.depth, (texel.0 * 2, texel.1 * 2));
				let through = marched(&lamps, Ray { far: REACH, ..ray }, offset_of(*texel));
				let got = texel_of(&found.raw, *texel)[0];

				got > 1.0e-3 && through.x > 1.5 * got
			})
			.count();

		assert!(lit > 15_000, "only {lit} texels see the lamp's light in front of the wall");
		assert!(cut > 3000, "and only {cut} would have seen much more of it through the wall");
	}

	#[test]
	fn a_ray_follows_the_four_nearest_lamps_it_crosses_and_a_fifth_changes_nothing() {
		let Some(mut capture) = capture() else {
			return;
		};
		let eye = Vec3::new(0.0, 0.0, 8.0);
		let lamp =
			|x: f32, z: f32| (bare(Vec3::ONE, 3.0, 3.0), Transform::at(Vec3::new(x, 0.0, z)));
		let four = [lamp(0.0, 0.0), lamp(0.0, -2.0), lamp(0.0, -4.0), lamp(0.0, -6.0)];
		// the fifth is the furthest and is made first, so that which four are
		// followed is the frame's order and not the order they were made in
		let five = [lamp(4.0, -8.0), four[0], four[1], four[2], four[3]];
		let mut fewer = looking(eye, Vec3::ZERO, 0.0);
		let mut more = looking(eye, Vec3::ZERO, 0.0);
		let (four, five) = (shining(&mut fewer, &four), shining(&mut more, &five));

		asking(&mut fewer, "1", "0");
		asking(&mut more, "1", "0");

		let without = drawn(&mut capture, &mut fewer);
		let with = drawn(&mut capture, &mut more);

		held_against(&fewer, &without, &four);
		held_against(&more, &with, &five);

		let (mut passed, mut added) = (0, 0);

		for texel in every_texel() {
			let ray = ray_at(&more, &with.depth, (texel.0 * 2, texel.1 * 2));
			let Some(crossed) = five
				.iter()
				.map(|lamp| crossing(lamp, ray))
				.collect::<Option<Vec<bool>>>()
				.filter(|crossed| crossed[4])
			else {
				continue;
			};
			let (got, had) = (texel_of(&with.raw, texel), texel_of(&without.raw, texel));
			let fifth = scattered(&five[4], ray, span_of(&five[4], ray), offset_of(texel));

			if crossed[..4].iter().all(|it| *it) {
				passed += 1;

				assert!(
					got.iter()
						.zip(had)
						.all(|(one, other)| one.to_bits() == other.to_bits()),
					"the texel at {texel:?} crosses all five lamps and follows the first four \
					 only"
				);
			} else if fifth.x > 1.0e-2 {
				// both are held against the arithmetic above, each to its own
				// sixteen-bit step; this only says which of the two is brighter
				added += 1;

				assert!(got[0] > had[0], "the texel at {texel:?} follows the fifth as well");
			}
		}

		assert!(passed > 150, "only {passed} texels cross all five lamps");
		assert!(added > 150, "only {added} cross the fifth and fewer than four others");
	}

	#[test]
	fn a_texel_is_averaged_with_the_neighbors_in_front_of_its_own_surface_and_lit_alike() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room();

		asking(&mut world, "1", "0");

		let found = drawn(&mut capture, &mut world);
		let lens = world.render_camera().projection(world.aspect);
		let distance = |(column, row): (u32, u32)| {
			lens.w_axis.z / (stored(&found.depth, (column * 2, row * 2)) + lens.z_axis.z)
		};
		let mut turned = (0, 0);
		let mut edges = 0;

		for texel in every_texel() {
			let (averages, away) = averaged_at(&found.raw, texel, distance);
			let got = light_of(texel_of(&found.averaged, texel));

			assert!(
				averages
					.iter()
					.any(|wanted| near_light(got, *wanted)),
				"the texel at {texel:?} averaged to {got}, not any of {averages:?}"
			);

			turned.0 += away.0;
			turned.1 += away.1;
			edges += usize::from(averages.len() > 1);
		}

		assert!(turned.0 > 5000, "only {} neighbors stood in front of another surface", turned.0);
		assert!(turned.1 > 5000, "and only {} were lit too differently", turned.1);

		// four on every device measured: ten times that is an edge wider than
		// the arithmetic, and a test that holds the average to nothing much
		assert!(edges < 40, "{edges} texels had a neighbor on an edge");
	}

	#[test]
	fn the_air_dims_what_is_behind_it_by_its_distance_and_lights_it_from_everywhere() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = distances();

		world.post.haze = 0.0;
		asking(&mut world, "1", "0");

		let clear = capture
			.shoot(&mut world)
			.expect("the capture renders");

		world.post.haze = THIN;

		let hazy = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let depth = capture
			.scene_mut()
			.depth_values()
			.expect("read, so readable");
		let mut seen = [0_u32; 3];
		let mut unreached = 0;

		for pixel in every_pixel() {
			let (_, distance) = seen_at(&world, &depth, pixel);
			let kept = (-THIN * distance.min(ENDS)).exp();
			let (before, after) = (clear.pixel(pixel.0, pixel.1), hazy.pixel(pixel.0, pixel.1));

			assert!(
				dimmed(before, after, kept),
				"the pixel at {pixel:?} was {before:?} and is {after:?} through {distance} of \
				 air"
			);

			let which = usize::from(distance > 30.0) + usize::from(stored(&depth, pixel) >= 1.0);

			seen[which] += 1;
			unreached +=
				u32::from(which == 1 && !dimmed(before, after, (-THIN * distance).exp()));
		}

		assert!(
			seen.iter().all(|count| *count > 15_000),
			"a wall near, a wall far and nothing: {seen:?}"
		);
		assert!(
			unreached > 15_000,
			"and only {unreached} pixels of the far wall tell where the air ends from the wall"
		);
	}

	#[test]
	fn a_pixel_of_the_air_over_the_picture_is_its_own_texel_or_the_blend_of_those_around_it() {
		// a lamp in open air and no light from everywhere, so that the picture
		// is the air and nothing else
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = looking(Vec3::new(0.0, 0.0, 8.0), Vec3::ZERO, 0.0);

		shining(&mut world, &[(
			bare(Vec3::new(1.0, 0.8, 0.6), 1.5, 4.0),
			Transform::at(Vec3::ZERO),
		)]);
		asking(&mut world, "1", "0");

		let picture = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let averaged = capture
			.scene_mut()
			.haze_values()
			.expect("asked for, so written");
		let mut between = 0;

		for pixel in every_pixel() {
			let wanted = blown_up(&averaged, pixel).to_array().map(byte_of);
			let seen = picture.pixel(pixel.0, pixel.1);

			assert!(
				seen.iter()
					.zip(wanted)
					.all(|(byte, level)| (f32::from(*byte) - level).abs() <= 1.0),
				"the pixel at {pixel:?} is {seen:?} where the air over it is {wanted:?}"
			);

			between += u32::from((20..235).contains(&seen[0]));
		}

		assert!(between > 10_000, "only {between} pixels are neither black nor white");
	}

	#[test]
	fn a_pixel_of_a_surface_beside_bright_air_takes_none_of_that_air() {
		// a pillar near the eye and out of the lamp's reach, standing in front of
		// the lamp's glow: the air along a ray that ends on the pillar sends
		// nothing, and the air beside it on the picture sends a great deal. A
		// pixel of the pillar between its own texels and the glow's is drawn from
		// its own alone, so every pixel of the pillar is what it was in clear air,
		// dimmed by the air in front of it and lit by none. Four times, the
		// pillar moved a quarter of a pixel each time, because a pixel blends the
		// texel past it only when its own place is odd
		let Some(mut capture) = capture() else {
			return;
		};
		let (mut pillar, mut beside) = (0, 0);

		for step in [0.0_f32, 1.0, 2.0, 3.0] {
			let (seen, odd) = pillar_beside_glow(&mut capture, step.mul_add(0.0033, 0.3));

			pillar += seen;
			beside += odd;
		}

		assert!(pillar > 16_000, "only {pillar} pixels see the pillar");
		assert!(beside > 200, "and only {beside} of them stand beside the glow at an odd place");
	}

	/// The pillar test's world with the pillar at a place across, drawn in
	/// clear air and in hazy air, every pixel of the pillar held to its clear
	/// self.
	///
	/// @return how many pixels see the pillar, and how many of them are at an
	/// odd place beside the glow
	fn pillar_beside_glow(capture: &mut Capture, across: f32) -> (u32, u32) {
		let mut world = looking(Vec3::new(0.0, 0.0, 8.0), Vec3::ZERO, 0.0);
		let pillar = made(&mut world, "test/pillar", 1.0);

		slab(
			&mut world,
			pillar,
			(Vec3::new(across, 0.0, 5.0), Vec3::new(0.3, 20.0, 0.3)),
			Vec3::ZERO,
		);
		shining(&mut world, &[(bare(Vec3::ONE, 6.0, 4.0), Transform::at(Vec3::ZERO))]);
		asking(&mut world, "1", "0");
		world.post.haze = 0.0;

		let clear = capture
			.shoot(&mut world)
			.expect("the capture renders");

		world.post.haze = HAZE;

		let hazy = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let depth = capture
			.scene_mut()
			.depth_values()
			.expect("read, so readable");
		let averaged = capture
			.scene_mut()
			.haze_values()
			.expect("asked for, so written");
		let on_pillar = |pixel: (u32, u32)| seen_at(&world, &depth, pixel).1 < 4.0;
		let (mut pillar, mut beside) = (0, 0);

		for pixel in every_pixel().filter(|pixel| on_pillar(*pixel)) {
			let kept = (-HAZE * seen_at(&world, &depth, pixel).1).exp();
			let (before, after) = (clear.pixel(pixel.0, pixel.1), hazy.pixel(pixel.0, pixel.1));

			pillar += 1;

			assert!(
				before
					.iter()
					.zip(after)
					.take(3)
					.all(|(was, is)| (f32::from(is) - byte_of(undone(*was) * kept)).abs() <= 1.0),
				"the pillar at {pixel:?} was {before:?} and is {after:?}, which is light from \
				 the air beside it"
			);

			// a pixel at an odd place beside the glow, whose four texels hold the
			// glow's as well as its own
			let glow = [pixel.0.saturating_sub(1), (pixel.0 + 1).min(SIZE.0 - 1)]
				.into_iter()
				.any(|column| {
					!on_pillar((column, pixel.1))
						&& texel_of(&averaged, (column / 2, pixel.1 / 2))[0] > 0.05
				});

			beside += u32::from(pixel.0 % 2 == 1 && glow);
		}

		(pillar, beside)
	}

	#[test]
	fn the_air_is_the_same_at_one_sample_and_at_four_in_front_of_a_wall_seen_square() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = square_wall(-3.0);

		asking(&mut world, "1", "0");

		let one = drawn(&mut capture, &mut world);

		asking(&mut world, "4", "0");

		let four = drawn(&mut capture, &mut world);
		let lit = same_light(&one.raw, &four.raw, "what the march wrote");

		same_light(&one.averaged, &four.averaged, "what the average wrote");

		assert!(lit > 4000, "only {lit} texels had any light to compare");
	}

	#[test]
	fn a_picture_drawn_into_a_rectangle_lights_the_air_the_same_picture_lights_drawn_alone() {
		// texel for texel wherever the two were handed the same numbers. A
		// picture drawn into a rectangle that does not start at nought has the
		// depth of a slanted floor measured a step apart by the rasterizer at
		// thousands of pixels, and a place of the march at the edge of a shadow
		// reads a step of depth as a different tap: where the numbers differ,
		// only a few texels may come out differently
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut whole = Capture::new(gpu, 240, 180).expect("the capture builds");
		let mut framed = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = room();

		asking(&mut world, "1", "0");
		whole.draw(&mut world, &mut []);

		let alone = read(&mut whole);

		framed.draw_within(&mut world, Viewport { x: 40, y: 30, width: 240, height: 180 });

		let within = read(&mut framed);
		let pair = Framed { alone: &alone, within: &within };
		let (mut strict, mut moved, mut lit) = ((0, 0), (0, 0), 0);

		for texel in (0..90).flat_map(|row| (0..120).map(move |column| (column, row))) {
			let (same, handed) = (pair.same_depth(texel), pair.handed_the_same(texel));
			let (marched, averaged) = (pair.raw(texel), pair.averaged(texel));
			let marched_near = near_light(light_of(marched.0), light_of(marched.1));
			let averaged_near = near_light(light_of(averaged.0), light_of(averaged.1));

			assert!(
				!same || marched_near,
				"the march at {texel:?} was handed the same depth and wrote {marched:?}"
			);
			assert!(
				!handed || averaged_near,
				"the average at {texel:?} was handed the same numbers and wrote {averaged:?}"
			);

			strict.0 += u32::from(same);
			strict.1 += u32::from(handed);
			moved.0 += u32::from(!marched_near);
			moved.1 += u32::from(!averaged_near);
			lit += u32::from(marched.0[0] > 1.0e-3);
		}

		// measured: 6,546 and 5,386 handed the same, and 0 and 4 texels out of
		// the rest that came out differently under one graphics API, 0 and 0
		// under the other
		assert!(strict.0 > 5000, "only {} texels of 10,800 were handed the same depth", strict.0);
		assert!(strict.1 > 4000, "only {} were handed the same numbers to average", strict.1);
		assert!(moved.0 < 20, "the march came out differently at {} texels", moved.0);
		assert!(moved.1 < 40, "the average came out differently at {} texels", moved.1);
		assert!(lit > 1000, "only {lit} texels had any light to compare");
	}

	/// The same picture drawn alone at 240 by 180 and into a rectangle of that
	/// size at 40, 30 of a picture of [`SIZE`], read back.
	#[derive(Clone, Copy)]
	struct Framed<'a> {
		alone: &'a Found,
		within: &'a Found,
	}

	impl Framed<'_> {
		/// What the march wrote at a texel of the picture drawn alone, and at
		/// the same texel of the rectangle.
		fn raw(self, (column, row): (u32, u32)) -> ([f32; 4], [f32; 4]) {
			(
				wide(&self.alone.raw, 120, (column, row)),
				wide(&self.within.raw, HALF.0, (column + 20, row + 15)),
			)
		}

		/// And what the average wrote.
		fn averaged(self, (column, row): (u32, u32)) -> ([f32; 4], [f32; 4]) {
			(
				wide(&self.alone.averaged, 120, (column, row)),
				wide(&self.within.averaged, HALF.0, (column + 20, row + 15)),
			)
		}

		/// Whether both were handed the same depth at a texel, to the bit.
		fn same_depth(self, (column, row): (u32, u32)) -> bool {
			let at = |depth: &[f32], width: u32, (x, y): (u32, u32)| {
				depth
					.get(usize::try_from(y * width + x).unwrap_or(usize::MAX))
					.copied()
					.expect("the pixel is inside the picture")
			};
			let one = at(&self.alone.depth, 240, (column * 2, row * 2));
			let other = at(&self.within.depth, SIZE.0, (column * 2 + 40, row * 2 + 30));

			one.to_bits() == other.to_bits()
		}

		/// Whether the average at a texel was handed the same numbers in both:
		/// the same depth at it and around it, and the same light from the
		/// march, to the bit.
		fn handed_the_same(self, texel: (u32, u32)) -> bool {
			AROUND
				.into_iter()
				.chain([(0, 0)])
				.filter_map(|offset| beside(texel, offset, (120, 90)))
				.all(|place| {
					let (one, other) = self.raw(place);

					self.same_depth(place)
						&& one
							.iter()
							.zip(other)
							.all(|(mine, theirs)| mine.to_bits() == theirs.to_bits())
				})
		}
	}

	/// The texel an offset away from another, if that is inside a buffer of a
	/// size.
	fn beside(
		(column, row): (u32, u32),
		(across, down): (i32, i32),
		(width, height): (u32, u32),
	) -> Option<(u32, u32)> {
		Some((
			column
				.checked_add_signed(across)
				.filter(|x| *x < width)?,
			row.checked_add_signed(down)
				.filter(|y| *y < height)?,
		))
	}

	/// One texel of a buffer of some width.
	fn wide(values: &[[f32; 4]], width: u32, (column, row): (u32, u32)) -> [f32; 4] {
		values
			.get(usize::try_from(row * width + column).unwrap_or(usize::MAX))
			.copied()
			.expect("the texel is inside the buffer")
	}

	#[test]
	fn a_resized_picture_lights_its_air_as_a_picture_made_at_that_size_does() {
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut resized = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut fresh = Capture::new(gpu, 240, 180).expect("the capture builds");
		let mut world = room();

		asking(&mut world, "1", "0");
		resized.draw(&mut world, &mut []);
		resized.scene_mut().resize(240, 180);
		resized.draw(&mut world, &mut []);
		fresh.draw(&mut world, &mut []);

		let after = resized
			.scene_mut()
			.haze_values()
			.expect("asked for, so written");
		let made = fresh
			.scene_mut()
			.haze_values()
			.expect("asked for, so written");
		let lit = same_light(&after, &made, "the air of the resized picture");

		assert!(lit > 1000, "only {lit} texels had any light to compare");
	}

	#[test]
	fn the_passes_run_while_something_asks_and_let_their_buffers_go_after() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = looking(Vec3::new(0.0, 0.0, 8.0), Vec3::ZERO, 0.0);

		shining(&mut world, &[(Light::point(Vec3::ONE, 4.0, 4.0), Transform::at(Vec3::ZERO))]);

		for samples in ["1", "4"] {
			// at four samples the depth is made readable by a pass of its own,
			// which nothing else in this world asks for
			let resolve = u32::from(samples == "4");

			asking(&mut world, samples, "0");
			world.post.haze = 0.0;
			capture.draw(&mut world, &mut []);

			let quiet = capture.scene_mut().spans().passes();
			let first = capture.scene_mut().haze_epoch();

			assert!(
				capture.scene_mut().haze_values().is_none(),
				"a frame nobody asks in makes no buffer"
			);

			world.post.haze = HAZE;
			capture.draw(&mut world, &mut []);

			assert!(capture.scene_mut().haze_values().is_some(), "hazy air asks");
			assert_eq!(
				capture.scene_mut().spans().passes(),
				quiet + 3 + resolve,
				"and at {samples} samples it costs a march, an average and a pass over the \
				 picture"
			);

			capture.draw(&mut world, &mut []);

			let kept = capture.scene_mut().haze_epoch();

			assert_eq!(
				kept,
				first + 1,
				"a second frame that asks keeps the buffers the first made"
			);

			world.post.haze = 0.0;
			capture.draw(&mut world, &mut []);

			assert!(
				capture.scene_mut().haze_values().is_none(),
				"and a frame that stops asking lets them go"
			);
			assert_eq!(capture.scene_mut().haze_epoch(), kept + 1, "which moves the epoch");
			assert_eq!(capture.scene_mut().spans().passes(), quiet, "and costs nothing again");

			asking(&mut world, samples, "7");
			capture.draw(&mut world, &mut []);

			let viewed = capture
				.scene_mut()
				.haze_values()
				.expect("the view asks in clear air too");

			assert!(
				viewed
					.iter()
					.all(|texel| light_of(*texel).max_element().abs() < f32::EPSILON),
				"and air of no density sends nothing"
			);
		}
	}

	#[test]
	fn a_shader_put_in_after_the_passes_were_built_is_the_one_the_air_is_lit_with() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room();

		asking(&mut world, "1", "0");
		capture.draw(&mut world, &mut []);

		assert!(capture.scene_mut().haze_built(), "the first frame built the passes");

		let source = include_str!("shader.wgsl");
		let anchor = "return vec4<f32>(light, 1.0);";

		assert_eq!(source.matches(anchor).count(), 1, "the line this test edits has moved");

		capture
			.scene_mut()
			.set_shader(&source.replace(anchor, "return vec4<f32>(0.25, 0.5, 0.75, 1.0);"))
			.expect("the edited shader compiles");

		let found = drawn(&mut capture, &mut world);

		for (index, texel) in found.averaged.iter().enumerate() {
			assert!(
				near_light(light_of(*texel), Vec3::new(0.25, 0.5, 0.75)),
				"the texel at {index} was lit with the shader put in: {texel:?}"
			);
		}
	}

	#[test]
	fn the_view_draws_the_light_the_air_sends_as_bytes_and_black_in_clear_air() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = looking(Vec3::new(0.0, 0.0, 8.0), Vec3::ZERO, 0.0);

		shining(&mut world, &[(
			bare(Vec3::new(1.0, 0.8, 0.6), 6.0, 4.0),
			Transform::at(Vec3::ZERO),
		)]);

		for samples in ["4", "1"] {
			world.post.haze = HAZE;
			asking(&mut world, samples, "7");

			let view = capture
				.shoot(&mut world)
				.expect("the capture renders");
			let averaged = capture
				.scene_mut()
				.haze_values()
				.expect("asked for, so written");
			let mut between = 0;

			for pixel in every_pixel() {
				// a byte is a number: the curve is undone before it is written
				let wanted = light_of(texel_of(&averaged, (pixel.0 / 2, pixel.1 / 2)))
					.to_array()
					.map(|level| (level.clamp(0.0, 1.0) * 255.0).round());
				let seen = view.pixel(pixel.0, pixel.1);

				assert!(
					seen.iter()
						.zip(wanted)
						.all(|(byte, level)| (f32::from(*byte) - level).abs() <= 1.0),
					"at {samples} samples the pixel at {pixel:?} is {seen:?}, not {wanted:?}"
				);

				between += u32::from((20..235).contains(&seen[0]));
			}

			assert!(between > 4000, "only {between} pixels are neither black nor white");

			world.post.haze = 0.0;

			let clear = capture
				.shoot(&mut world)
				.expect("the capture renders");

			assert!(
				clear
					.pixels
					.chunks_exact(4)
					.all(|pixel| pixel == [0, 0, 0, 255]),
				"and in clear air it is black, because the air sends nothing"
			);
		}
	}

	#[test]
	fn a_frame_after_the_sample_count_changed_reads_the_depth_it_drew() {
		// **the groups the passes keep.** At four samples the depth they read is
		// a buffer of its own, made again when the count changes: a group kept
		// across that holds a view of the depth of a frame long gone, which is
		// what moving the wall in between makes visible
		let (Some(mut walked), Some(mut fresh)) = (capture(), capture()) else {
			return;
		};
		let mut far = square_wall(-3.0);
		let mut near = square_wall(-1.5);

		asking(&mut far, "4", "0");
		walked.draw(&mut far, &mut []);
		asking(&mut near, "1", "0");
		walked.draw(&mut near, &mut []);
		asking(&mut near, "4", "0");

		let after = drawn(&mut walked, &mut near);
		let straight = drawn(&mut fresh, &mut near);
		let lit =
			same_light(&after.averaged, &straight.averaged, "the air in front of the moved wall");

		assert!(lit > 4000, "only {lit} texels had any light to compare");
	}

	#[test]
	fn a_pane_of_glass_between_the_eye_and_the_air_is_not_where_a_ray_ends() {
		// out of the lamp's reach, so that nothing it could put in a map changes
		// the light either
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room();

		asking(&mut world, "1", "0");

		let open = drawn(&mut capture, &mut world);
		let glass = world.materials.insert("test/glass", Material {
			blend: Blend::Alpha,
			opacity: 0.35,
			roughness: 0.1,
			..Material::DEFAULT
		});

		slab(
			&mut world,
			glass,
			(Vec3::new(0.0, 1.0, 4.0), Vec3::new(8.0, 4.0, 0.05)),
			Vec3::new(0.6, 0.8, 0.9),
		);

		let behind = drawn(&mut capture, &mut world);

		assert!(open.depth == behind.depth, "nothing blended writes the depth");
		assert!(open.raw == behind.raw, "so nothing blended is where a ray of the air ends");
	}

	#[test]
	fn the_places_along_a_ray_add_up_to_what_the_sun_sends_through_air_a_shadow_crosses() {
		// sixteen places spread evenly, at each of the nine offsets the tile
		// spreads them by, against the same air summed in sixty thousand even
		// steps: out in the open, short of where the air ends, and with a shadow
		// across the ray near the eye, in the middle and out to the end
		let world = sunlit(Vec3::ZERO, Vec3::NEG_Z, Vec3::new(0.3, -1.0, -0.2), 0.0);
		let way = Vec3::new(0.2, 0.1, -1.0).normalize();
		let cases = [
			(REACH, (0.0, 0.0)),
			(9.0, (0.0, 0.0)),
			(REACH, (4.3, 9.7)),
			(20.0, (0.0, 3.3)),
			(REACH, (29.0, REACH)),
		];

		for (far, (from, to)) in cases {
			let ray = Ray { origin: Vec3::ZERO, way, far };
			let reached = |along: f32| if (from..to).contains(&along) { 0.0 } else { 1.0 };
			let answered = |at: Vec3| Some(reached(at.dot(way)));
			let places = (0..9_u8)
				.map(|place| sun_along(&world, ray, (f32::from(place) + 0.5) / 9.0, answered))
				.map(|light| light.expect("every place is answered").x)
				.sum::<f32>()
				/ 9.0;
			let wanted = summed_sun(&world, ray, reached);

			assert!(
				(f64::from(places) - wanted).abs() < WITHIN_SUN * wanted,
				"along {far} with a shadow from {from} to {to} the places add up to {places} \
				 where the air sends {wanted}"
			);
		}
	}

	#[test]
	fn the_sun_in_open_air_lights_each_texel_by_what_the_places_along_its_ray_add_up_to() {
		// nothing to throw a shadow, so all of the sun reaches every place: out to
		// where the air ends in the middle of the view, which is past where
		// anything is shadowed at all, and short of that towards the edges. Two
		// ways for the light to travel, so that the lobe is asked at two angles
		let Some(mut capture) = capture() else {
			return;
		};

		for light in [Vec3::new(0.3, -1.0, -0.2), Vec3::new(-0.5, 0.2, 1.0)] {
			let mut world =
				sunlit(Vec3::new(0.0, 1.0, 8.0), Vec3::new(0.0, 1.0, 0.0), light, 0.0);

			asking(&mut world, "1", "0");

			let found = drawn(&mut capture, &mut world);
			let (held, left) = held_with_sun(&world, &found, &[], |_| Some(1.0));
			let past = every_texel()
				.filter(|texel| {
					past_the_shadows(&world, &found, *texel, shadow::DEFAULT_DISTANCE)
				})
				.count();

			assert_eq!(left, 0, "with the light going {light} every place is answered");
			assert!(held > 19_000, "only {held} texels were held");
			assert!(past > 2000, "and only {past} have a place past the shadow distance");
		}
	}

	/// Whether the last of a texel's places is further along the view than a
	/// distance: past the shadow distance, where no cascade is asked.
	fn past_the_shadows(world: &World, found: &Found, texel: (u32, u32), distance: f32) -> bool {
		let camera = world.render_camera();
		let forward = (camera.target - camera.position).normalize_or(Vec3::NEG_Z);
		let ray = ray_at(world, &found.depth, (texel.0 * 2, texel.1 * 2));

		sun_places(ray, offset_of(texel)).any(|along| (ray.way * along).dot(forward) > distance)
	}

	#[test]
	fn a_ceiling_between_the_sun_and_the_air_leaves_the_air_under_it_unlit_out_to_the_shadow_distance()
	 {
		// the eye under a ceiling it cannot see the end of, with a floor below.
		// Every place the ceiling shadows gets none of the sun, every place it
		// does not gets all of it, and a place past the shadow distance gets all
		// of it under the ceiling too, as a surface there does - asked at the
		// default distance, where no place under the ceiling is past it, and at
		// four units, where most are
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = sunlit(
			Vec3::new(0.0, 1.0, 0.0),
			Vec3::new(0.0, 1.6, -10.0),
			Vec3::new(0.3, -1.0, -0.2),
			0.0,
		);
		let surface = made(&mut world, "test/surface", 0.8);
		let ceiling = (Vec3::new(-6.0, 3.0, -14.0), Vec3::new(6.0, 3.2, 4.0));

		slab(
			&mut world,
			surface,
			((ceiling.0 + ceiling.1) * 0.5, ceiling.1 - ceiling.0),
			Vec3::splat(0.5),
		);
		slab(
			&mut world,
			surface,
			(Vec3::new(0.0, -0.1, 0.0), Vec3::new(200.0, 0.2, 200.0)),
			Vec3::splat(0.5),
		);
		asking(&mut world, "1", "0");
		world
			.cvars
			.var(shadow::DISTANCE, Value::Float(shadow::DEFAULT_DISTANCE), "");

		for (distance, words) in [(shadow::DEFAULT_DISTANCE, "50"), (4.0, "4")] {
			world.cvars.set(shadow::DISTANCE, words);

			let found = drawn(&mut capture, &mut world);
			let cascades =
				shadow::fit(&world.render_camera(), world.aspect, world.light, distance);
			let (held, left) =
				held_with_sun(&world, &found, &[], under(&world, &cascades, ceiling, false));
			let (dark, reprieved) = shadowed_places(&world, &found, &cascades, ceiling);

			// measured: 18,603 held and 597 left out at the default distance,
			// 19,093 and 107 at four; 14,921 and 6,347 in the shadow at every
			// place; 12,853 with a place past four units that the ceiling shadows
			assert!(held > 17_000, "at a distance of {distance} only {held} texels were held");
			assert!(left < 1500, "and {left} were left out");

			if distance < shadow::DEFAULT_DISTANCE {
				assert!(dark > 4000, "and only {dark} have every place in the ceiling's shadow");
				assert!(
					reprieved > 10_000,
					"and only {reprieved} have a place in its shadow lit for being past the \
					 distance"
				);
			} else {
				assert!(
					dark > 10_000,
					"and only {dark} have every place in the ceiling's shadow"
				);
				assert_eq!(reprieved, 0, "and none is past the distance");
			}
		}

		// and with shadows off, the sun reaches all of the air, as it reaches every
		// surface
		world
			.cvars
			.var(shadow::ENABLED, Value::Bool(false), "off for this frame");

		let found = drawn(&mut capture, &mut world);
		let (held, left) = held_with_sun(&world, &found, &[], |_| Some(1.0));

		assert_eq!((held, left), (19_200, 0), "with shadows off every texel is the open air's");
	}

	/// How many texels have every place in a box's shadow by a clear margin,
	/// and how many have a place in it that is lit for being past the shadow
	/// distance.
	fn shadowed_places(
		world: &World,
		found: &Found,
		cascades: &shadow::Cascades,
		caster: (Vec3, Vec3),
	) -> (u32, u32) {
		let (asked, beyond) =
			(under(world, cascades, caster, false), under(world, cascades, caster, true));
		let mut counted = (0, 0);

		for texel in every_texel() {
			let ray = ray_at(world, &found.depth, (texel.0 * 2, texel.1 * 2));
			let places: Vec<Vec3> = sun_places(ray, offset_of(texel))
				.map(|along| ray.origin + ray.way * along)
				.collect();

			counted.0 += u32::from(places.iter().all(|at| asked(*at) == Some(0.0)));
			counted.1 += u32::from(
				places
					.iter()
					.any(|at| asked(*at) == Some(1.0) && beyond(*at) == Some(0.0)),
			);
		}

		counted
	}

	#[test]
	fn air_just_in_front_of_a_wall_the_sun_lights_is_not_in_the_walls_shadow() {
		// the wall faces the sun and the eye together and nothing stands between
		// the sun and the air, so all of the sun reaches every place - down to
		// the last of each ray, which for one texel in nine sits nearer the wall
		// than two texels of its cascade. What keeps it out of the wall's own
		// shadow is the bias the cascades are drawn with: nothing moves the point
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world =
			sunlit(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO, Vec3::new(0.35, -0.3, -1.0), 0.0);
		let wall = made(&mut world, "test/wall", 0.8);

		slab(
			&mut world,
			wall,
			(Vec3::new(0.0, 0.0, -0.25), Vec3::new(60.0, 60.0, 0.5)),
			Vec3::splat(0.5),
		);
		asking(&mut world, "1", "0");

		let found = drawn(&mut capture, &mut world);
		let (held, left) = held_with_sun(&world, &found, &[], |_| Some(1.0));
		let cascades = shadow::fit(
			&world.render_camera(),
			world.aspect,
			world.light,
			shadow::DEFAULT_DISTANCE,
		);
		let close = every_texel()
			.filter(|texel| {
				let ray = ray_at(&world, &found.depth, (texel.0 * 2, texel.1 * 2));
				let last = sun_places(ray, offset_of(*texel))
					.last()
					.unwrap_or(0.0);
				let at = ray.origin + ray.way * last;

				at.z < 2.0 * cascades.texels[1]
			})
			.count();

		assert_eq!((held, left), (19_200, 0), "every texel is the open air's");
		assert!(
			close > 1000,
			"but only {close} texels have a place within two texels of the wall"
		);
	}

	#[test]
	fn the_sun_lights_the_air_of_a_ray_that_follows_four_lamps_as_well() {
		// the sun is none of the four lamps a texel's ray follows: five lamps in
		// a row and the sun in open air, and a texel crossing four of them gets
		// the sun as well as the four
		let Some(mut capture) = capture() else {
			return;
		};
		let lamp =
			|x: f32, z: f32| (bare(Vec3::ONE, 3.0, 3.0), Transform::at(Vec3::new(x, 0.0, z)));
		let mut world =
			sunlit(Vec3::new(0.0, 0.0, 8.0), Vec3::ZERO, Vec3::new(0.3, -1.0, -0.2), 0.0);
		let lamps = shining(&mut world, &[
			lamp(4.0, -8.0),
			lamp(0.0, 0.0),
			lamp(0.0, -2.0),
			lamp(0.0, -4.0),
			lamp(0.0, -6.0),
		]);

		asking(&mut world, "1", "0");

		let found = drawn(&mut capture, &mut world);
		let (held, left) = held_with_sun(&world, &found, &lamps, |_| Some(1.0));
		let four = every_texel()
			.filter(|texel| {
				let ray = ray_at(&world, &found.depth, (texel.0 * 2, texel.1 * 2));

				lamps
					.iter()
					.filter(|lamp| crossing(lamp, ray) == Some(true))
					.count() >= 4
			})
			.count();

		assert_eq!((held, left), (19_200, 0), "every texel is the sun's and the lamps'");
		assert!(four > 150, "but only {four} texels cross four lamps or more");
	}
}
