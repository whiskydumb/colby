//! The pass before the scene: what every pixel's surface is, before anything
//! lights it.
//!
//! One pass over the solid half of the picture's own list - two in a frame that
//! tests for what is behind something nearer, the large things ahead of the
//! test and the small ones it kept after it, @ref [`cover`](crate::cover) -
//! into targets of its own, writing four things a pixel: how far away the
//! nearest surface is, the normal that surface is about to be lit with, how
//! rough it is there, and what it is made of. What reads them is whatever has
//! to know about a surface *before* the scene's pass lights it. An occlusion
//! term that darkens only the light a lamp did not send is exactly that, and it
//! cannot take the depth the scene writes: by the time that depth exists the
//! lighting is done. So is a reflection of one surface in another, which has to
//! light the surface it finds before the scene lights the one it is found in.
//! @ref [`depth`](crate::depth) for the depth a pass *after* the scene reads.
//!
//! **Its own depth, not the scene's.** The scene's pass clears its buffer and
//! draws exactly as it did before this existed, and this pass keeps a buffer of
//! its own. So a frame that asks for it pays for a second pass over every solid
//! thing in view and gets nothing back in the first. The other arrangement -
//! the scene testing against this depth and throwing away what lies behind it
//! before it is shaded - would buy the pass back, and it would cost a position
//! worked out identically across pipelines, a target of normals at four samples
//! a pixel and a pass to resolve it. A frame nobody asks for it in pays nothing
//! at all: no pass, no buffer, and no pipeline until the first frame that does.
//!
//! **One sample a pixel, whatever the picture is drawn with.** A reader binds
//! the same thing at every sample count - [`entry`] for the surfaces, and
//! [`depth::entry`](crate::depth::entry) for the depth written beside them,
//! which carries the flag that lets it be bound - and what it reads at an edge
//! is the middle of the pixel rather than the nearest of four samples.
//!
//! **What the buffers hold**, one texel a pixel in sixteen-bit floats. The
//! surfaces: xyz the normal in the world, signed and of unit length, with
//! nothing packed; w the roughness the lobe reads. Nought in every channel is a
//! pixel nothing was drawn in, which a surface cannot write because no surface
//! is smoother than the smoothest the lobe is drawn at. The material: rgb the
//! color the surface is lit with, its picture sampled and its decals painted,
//! and a how metal it is. What neither holds: anything that blends, so a reader
//! sees what is behind a pane of glass, as the depth after the scene does;
//! particles, the debug lines and the sky; and how a surface moved.
//!
//! **The surface the scene lights, not a cheaper cousin of it.** The fragment
//! stage asks `surface_at` in `shader.wgsl`, which is the function the scene's
//! own shading asks, so the normal map and every decal over a point have turned
//! the normal by the time it is written, and bones have moved it through the
//! same vertex entry points.

use colby_core::{
	Result,
	abi::{World, material::Blend},
	err, warn,
};
use wgpu::{
	BindGroupLayout, BindGroupLayoutEntry, BindingType, BlendState, Color, ColorTargetState,
	ColorWrites, CommandEncoder, CompareFunction, DepthBiasState, DepthStencilState, Device,
	ErrorFilter, Extent3d, Face, FragmentState, FrontFace, LoadOp, MultisampleState, Operations,
	PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, PolygonMode,
	PrimitiveState, PrimitiveTopology, RenderPass, RenderPassColorAttachment,
	RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPassTimestampWrites,
	RenderPipeline, RenderPipelineDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource,
	ShaderStages, StencilState, StoreOp, TextureDescriptor, TextureDimension, TextureFormat,
	TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
	VertexState,
};

use crate::scene::{DEPTH_FORMAT, vertex_buffers};

/// The console variable that draws what this pass wrote instead of the picture.
///
/// One draws the normal as a color, two the roughness as a grey, three how much
/// of the sky each pixel sees - the first thing worked out from what this pass
/// wrote, @ref [`occlusion`](crate::occlusion) - four the color each surface is
/// lit with, five what each pixel's reflection finds on the picture and six how
/// much of its reflection it found there, @ref
/// [`reflection`](crate::reflection), and seven the light a haze sends towards
/// the eye, @ref [`haze`](crate::haze). Anything else is the picture. A tool,
/// like the depth view: off until somebody sets it and never saved.
pub const VIEW: &str = "r.normals";

/// What [`VIEW`] holds until somebody sets it, which is the picture.
pub const NO_VIEW: f32 = 0.0;

/// The format the surfaces are written in.
///
/// Sixteen-bit floats, because a direction is signed and a float holds the sign
/// with no packing at all, and because the fourth channel beside the three of a
/// normal is free for the roughness. Just under one a step of these is two to
/// the minus eleventh, a quarter of the step ten unsigned bits spread over
/// minus one to one would leave.
pub(crate) const FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// What the view draws of the buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Showing {
	/// The normal, each axis from minus one to one laid over nought to one.
	Normal,

	/// The roughness, as a grey.
	Roughness,

	/// How much of the sky each pixel sees, as a grey: not a number this pass
	/// writes but the first one worked out from what it writes. @ref
	/// [`occlusion`](crate::occlusion).
	Occlusion,

	/// The color each surface is lit with, as it is.
	Material,

	/// What each pixel's reflection found on the picture, as light, already
	/// multiplied by how much of it was found. @ref
	/// [`reflection`](crate::reflection).
	Reflections,

	/// How much of each pixel's reflection was found on the picture, as a grey.
	Coverage,

	/// The light a haze sends towards the eye along each pixel's ray, as light:
	/// not a number this pass writes, and not one it has to run for. @ref
	/// [`haze`](crate::haze).
	Haze,
}

/// What [`VIEW`] asks this frame to draw, if anything.
///
/// A number rather than a word, because every tool of this kind is one: one to
/// seven are the answers, and anything else - a nan among them - is the
/// picture.
///
/// @param world - for the console variable
#[must_use]
pub(crate) fn showing_of(world: &World) -> Option<Showing> {
	let asked = world.cvars.float(VIEW)?;
	let answers = [
		Showing::Normal,
		Showing::Roughness,
		Showing::Occlusion,
		Showing::Material,
		Showing::Reflections,
		Showing::Coverage,
		Showing::Haze,
	];

	answers
		.into_iter()
		.zip(1_u8..)
		.find(|(_, number)| {
			let number = f32::from(*number);

			(number - 0.5..number + 0.5).contains(&asked)
		})
		.map(|(showing, _)| showing)
}

/// How a reader binds [`Prepass::surfaces`]: a float texture of one sample,
/// read with `textureLoad` or through a filtering sampler alike.
///
/// @param binding - where in the reader's own group it sits
pub(crate) const fn entry(binding: u32) -> BindGroupLayoutEntry {
	BindGroupLayoutEntry {
		binding,
		visibility: ShaderStages::FRAGMENT,
		ty: BindingType::Texture {
			sample_type: TextureSampleType::Float { filterable: true },
			view_dimension: TextureViewDimension::D2,
			multisampled: false,
		},
		count: None,
	}
}

/// The three targets the pass writes.
struct Targets {
	/// How far away the nearest solid surface is, stored the way the scene's
	/// own depth stores it.
	depth: TextureView,

	/// Its normal and its roughness.
	surfaces: TextureView,

	/// Its color and how metal it is, in [`FORMAT`] as well.
	///
	/// **Floats rather than eight bits of sRGB**, which would be half the
	/// memory: a color here is lit again by a reflection, and a second answer
	/// that works that light out has to know the color the first one used to
	/// the bit rather than to the byte an sRGB write rounds it to.
	material: TextureView,
}

/// The pass before the scene, and what it writes into.
pub(crate) struct Prepass {
	/// The size of the picture, which is the size of both targets.
	size: (u32, u32),

	/// Both targets, made the first frame something asks and let go the first
	/// frame nothing does.
	///
	/// **Let go rather than kept**, the way the resolved depth is: a view that
	/// is up for a moment has no business holding eleven megabytes at
	/// seven-twenty for the rest of the run.
	targets: Option<Targets>,

	/// Which making of [`targets`](Self::targets) a reader is looking at.
	///
	/// A reader that runs every frame keeps its bind group beside this and
	/// makes the group again when the two disagree: it moves whenever the
	/// targets are let go, which a pair of new ones is always made after, and
	/// never comes back to a value it has had. @ref
	/// [`Depth::epoch`](crate::depth::Depth::epoch), which is the same bargain
	/// for the depth after the scene.
	epoch: u64,

	/// One per (the picture has holes in it, bones move it), built the first
	/// frame something asks.
	///
	/// Lazily, and for the particles' reason: every capture a test builds
	/// would otherwise compile four more pipelines for a pass almost none of
	/// them records.
	pipelines: Option<[RenderPipeline; 4]>,
}

impl Prepass {
	/// A pass that has built nothing yet.
	///
	/// @param width - the picture's width in pixels
	/// @param height - its height
	pub(crate) const fn new(width: u32, height: u32) -> Self {
		Self {
			size: (width, height),
			targets: None,
			epoch: 0,
			pipelines: None,
		}
	}

	/// Notes a new picture size.
	///
	/// The targets are let go rather than made again: the next frame that asks
	/// makes them at the new size.
	pub(crate) fn resize(&mut self, width: u32, height: u32) {
		if self.size == (width, height) {
			return;
		}

		self.size = (width, height);
		self.release();
	}

	/// Lets both targets go, for a frame nothing reads them in.
	pub(crate) fn release(&mut self) {
		if self.targets.take().is_some() {
			self.epoch = self.epoch.wrapping_add(1);
		}
	}

	/// Builds whatever this frame's pass needs and does not have yet.
	///
	/// @param device - the device to build against
	/// @param groups - the scene's four group layouts, in group order
	/// @param source - the WGSL the scene's own table was built from
	/// @return whether there is a pass to record: nothing when the pipelines
	/// would not build, which is said
	pub(crate) fn ensure(
		&mut self,
		device: &Device,
		groups: &[&BindGroupLayout],
		source: &str,
	) -> bool {
		if self.pipelines.is_none() {
			match build(device, groups, source) {
				| Ok(built) => self.pipelines = Some(built),
				| Err(complaint) => {
					warn!(%complaint, "nothing is written before the scene this frame");

					return false;
				},
			}
		}

		if self.targets.is_none() {
			self.targets = Some(targets(device, self.size));
		}

		true
	}

	/// Which making of the targets a reader is looking at. @ref
	/// [`epoch`](Self::epoch)'s field for when it moves.
	pub(crate) const fn epoch(&self) -> u64 { self.epoch }

	/// The pipelines built again from new source, or nothing if they have
	/// never been built at all.
	///
	/// Asked by the scene when its shader changes and before either table is
	/// replaced, so that the two are replaced together or not at all: a normal
	/// written by one shader and lit by another would be two answers to one
	/// question.
	///
	/// @param device - the device to build against
	/// @param groups - the scene's four group layouts, in group order
	/// @param source - the new WGSL
	/// @return the four pipelines, nothing when there were none to replace, or
	/// the compiler's complaint
	pub(crate) fn rebuilt(
		&self,
		device: &Device,
		groups: &[&BindGroupLayout],
		source: &str,
	) -> Result<Option<[RenderPipeline; 4]>> {
		if self.pipelines.is_none() {
			return Ok(None);
		}

		build(device, groups, source).map(Some)
	}

	/// Puts pipelines from [`rebuilt`](Self::rebuilt) in place, when it had
	/// any to give.
	pub(crate) fn replace(&mut self, pipelines: Option<[RenderPipeline; 4]>) {
		if pipelines.is_some() {
			self.pipelines = pipelines;
		}
	}

	/// Begins this frame's pass, cleared: nought in every channel of the
	/// surfaces and the material, and the far plane in the depth.
	///
	/// @param encoder - the frame's
	/// @param marks - the span this pass is, if anybody is measuring
	/// @return the pass, or nothing before [`ensure`](Self::ensure) has made
	/// the targets
	pub(crate) fn begin<'pass>(
		&'pass self,
		encoder: &'pass mut CommandEncoder,
		marks: Option<RenderPassTimestampWrites<'pass>>,
	) -> Option<RenderPass<'pass>> {
		self.open(encoder, marks, true)
	}

	/// Begins the pass again over what it has written this frame: the small
	/// things, drawn after the test into the targets the large ones left.
	///
	/// @param encoder - the frame's, with the first half and the test in it
	/// @param marks - the span this pass is, if anybody is measuring
	/// @return the pass, or nothing before [`ensure`](Self::ensure) has made
	/// the targets
	pub(crate) fn resume<'pass>(
		&'pass self,
		encoder: &'pass mut CommandEncoder,
		marks: Option<RenderPassTimestampWrites<'pass>>,
	) -> Option<RenderPass<'pass>> {
		self.open(encoder, marks, false)
	}

	/// Begins a pass over the targets, cleared or as they stand.
	///
	/// @param encoder - the frame's
	/// @param marks - the span this pass is, if anybody is measuring
	/// @param cleared - whether the pass starts from nothing
	fn open<'pass>(
		&'pass self,
		encoder: &'pass mut CommandEncoder,
		marks: Option<RenderPassTimestampWrites<'pass>>,
		cleared: bool,
	) -> Option<RenderPass<'pass>> {
		let targets = self.targets.as_ref()?;

		// nought is what says nothing was drawn, and it has to be there wherever
		// nothing is
		let color = |view| {
			Some(RenderPassColorAttachment {
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
			})
		};

		Some(encoder.begin_render_pass(&RenderPassDescriptor {
			label: Some(if cleared { "prepass" } else { "prepass small" }),
			color_attachments: &[color(&targets.surfaces), color(&targets.material)],
			depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
				view: &targets.depth,
				depth_ops: Some(Operations {
					load: if cleared { LoadOp::Clear(1.0) } else { LoadOp::Load },
					store: StoreOp::Store,
				}),
				stencil_ops: None,
			}),
			timestamp_writes: marks,
			occlusion_query_set: None,
			multiview_mask: None,
		}))
	}

	/// Which pipeline writes a batch.
	///
	/// A match rather than a lookup, the shadows' way: a mode nobody has
	/// thought about is a compile error here on the day it is added.
	///
	/// @param blend - how the surface reads its picture's alpha
	/// @param skinned - whether bones move it
	/// @return the pipeline, or nothing for a surface that blends, which this
	/// pass does not write, and nothing before the pipelines are built
	pub(crate) fn pipeline(&self, blend: Blend, skinned: bool) -> Option<&RenderPipeline> {
		let masked = match blend {
			| Blend::Opaque => false,
			| Blend::Mask => true,
			// the second place that says so, after the list the scene files a
			// blended surface in - which is why a batch of one never reaches
			// this. It is here for whoever comes looking.
			| Blend::Alpha => return None,
		};

		self.pipelines
			.as_ref()?
			.get(usize::from(masked) * 2 + usize::from(skinned))
	}

	/// What a reader binds for the surfaces this frame, one sample a pixel.
	pub(crate) fn surfaces(&self) -> Option<&TextureView> {
		self.targets
			.as_ref()
			.map(|targets| &targets.surfaces)
	}

	/// The depth this pass wrote, one sample a pixel.
	///
	/// Not the depth the scene tests against, and at four samples not the
	/// nearest of four: the middle of the pixel, the same raster the surfaces
	/// beside it came out of. @ref [`occlusion`](crate::occlusion), which reads
	/// both.
	pub(crate) fn depth(&self) -> Option<&TextureView> {
		self.targets
			.as_ref()
			.map(|targets| &targets.depth)
	}

	/// What a reader binds for the material this frame, one sample a pixel:
	/// rgb the color and a how metal it is. Bound with [`entry`], like the
	/// surfaces.
	pub(crate) fn material(&self) -> Option<&TextureView> {
		self.targets
			.as_ref()
			.map(|targets| &targets.material)
	}

	/// What the material holds, four floats a pixel, top row first. A test's.
	///
	/// @param device - to build the staging buffer on
	/// @param queue - to submit the copy on
	#[cfg(test)]
	pub(crate) fn material_values(
		&self,
		device: &Device,
		queue: &wgpu::Queue,
	) -> Option<Vec<[f32; 4]>> {
		halves(device, queue, self.material()?)
	}

	/// What the surfaces hold, four floats a pixel, top row first. A test's.
	///
	/// @param device - to build the staging buffer on
	/// @param queue - to submit the copy on
	#[cfg(test)]
	pub(crate) fn surface_values(
		&self,
		device: &Device,
		queue: &wgpu::Queue,
	) -> Option<Vec<[f32; 4]>> {
		halves(device, queue, self.surfaces()?)
	}

	/// What the depth this pass wrote holds, one float a pixel. A test's.
	///
	/// @param device - to build the staging buffer on
	/// @param queue - to submit the copy on
	#[cfg(test)]
	pub(crate) fn depth_values(&self, device: &Device, queue: &wgpu::Queue) -> Option<Vec<f32>> {
		let texture = self.depth()?.texture();

		crate::depth::floats(&crate::depth::copied_out(
			device,
			queue,
			texture,
			wgpu::TextureAspect::DepthOnly,
			4,
		)?)
	}
}

/// The four pipelines, against one module, or the first complaint wgpu had.
///
/// @param device - the device to build against
/// @param groups - the scene's four group layouts, in group order
/// @param source - the whole WGSL
fn build(
	device: &Device,
	groups: &[&BindGroupLayout],
	source: &str,
) -> Result<[RenderPipeline; 4]> {
	let scope = device.push_error_scope(ErrorFilter::Validation);
	let shader = device.create_shader_module(ShaderModuleDescriptor {
		label: Some("prepass"),
		source: ShaderSource::Wgsl(source.into()),
	});
	// the scene's four groups exactly, the shadow atlas's among them though no
	// entry point here reads it: a pipeline may not skip a group, and a hole in
	// a layout moves every group after it somewhere else without a word
	let declared: Vec<Option<&BindGroupLayout>> = groups.iter().copied().map(Some).collect();
	let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some("prepass"),
		bind_group_layouts: &declared,
		immediate_size: 0,
	});
	let built = core::array::from_fn(|index| {
		pipeline(device, &shader, &layout, Wanted {
			masked: index >= 2,
			skinned: index % 2 == 1,
		})
	});

	match pollster::block_on(scope.pop()) {
		| Some(complaint) => Err(err!(Graphics("the pass before the scene: {complaint}"))),
		| None => Ok(built),
	}
}

/// Which of the four pipelines to build.
#[derive(Clone, Copy, Debug)]
struct Wanted {
	/// Whether the surface's picture has to be sampled before anything is
	/// written for it.
	masked: bool,

	/// Whether bones move the geometry.
	skinned: bool,
}

/// One of the four.
///
/// **The scene's own vertex entry points, buffers and culling**, so that a
/// normal is written for exactly the triangles the scene lights and in exactly
/// the places it lights them. What differs is the rest: one sample, no coverage
/// from alpha, a target of floats and the fragment entry points that write a
/// surface rather than light it.
fn pipeline(
	device: &Device,
	shader: &ShaderModule,
	layout: &PipelineLayout,
	wanted: Wanted,
) -> RenderPipeline {
	let Wanted { masked, skinned } = wanted;
	let buffers = vertex_buffers(skinned);

	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some(label_of(wanted)),
		layout: Some(layout),
		vertex: VertexState {
			module: shader,
			entry_point: Some(if skinned { "vertex_skinned" } else { "vertex_main" }),
			compilation_options: PipelineCompilationOptions::default(),
			buffers: &buffers,
		},
		primitive: PrimitiveState {
			topology: PrimitiveTopology::TriangleList,
			strip_index_format: None,
			front_face: FrontFace::Ccw,
			cull_mode: Some(Face::Back),
			unclipped_depth: false,
			polygon_mode: PolygonMode::Fill,
			conservative: false,
		},
		depth_stencil: Some(DepthStencilState {
			format: DEPTH_FORMAT,
			depth_write_enabled: Some(true),
			depth_compare: Some(CompareFunction::Less),
			stencil: StencilState::default(),
			bias: DepthBiasState::default(),
		}),
		multisample: MultisampleState::default(),
		fragment: Some(FragmentState {
			module: shader,
			entry_point: Some(if masked {
				"fragment_prepass_masked"
			} else {
				"fragment_prepass"
			}),
			compilation_options: PipelineCompilationOptions::default(),
			// the surfaces, then the material, in the order `Prepared` in
			// `shader.wgsl` lays its two outputs out
			targets: &[WRITTEN, WRITTEN],
		}),
		multiview_mask: None,
		cache: None,
	})
}

/// How each of the pass's two color targets is written: in [`FORMAT`], every
/// channel, replacing what was there.
const WRITTEN: Option<ColorTargetState> = Some(ColorTargetState {
	format: FORMAT,
	blend: Some(BlendState::REPLACE),
	write_mask: ColorWrites::ALL,
});

/// What one of the four is called in a graphics debugger.
const fn label_of(wanted: Wanted) -> &'static str {
	match (wanted.masked, wanted.skinned) {
		| (false, false) => "prepass",
		| (false, true) => "prepass skinned",
		| (true, false) => "prepass masked",
		| (true, true) => "prepass masked skinned",
	}
}

/// All three targets, at a size.
fn targets(device: &Device, (width, height): (u32, u32)) -> Targets {
	let target = |label, format| {
		device
			.create_texture(&TextureDescriptor {
				label: Some(label),
				size: Extent3d {
					width: width.max(1),
					height: height.max(1),
					depth_or_array_layers: 1,
				},
				mip_level_count: 1,
				sample_count: 1,
				dimension: TextureDimension::D2,
				format,
				usage: TextureUsages::RENDER_ATTACHMENT
					| TextureUsages::TEXTURE_BINDING
					| copied(),
				view_formats: &[],
			})
			.create_view(&TextureViewDescriptor::default())
	};

	Targets {
		depth: target("prepass depth", DEPTH_FORMAT),
		surfaces: target("prepass surfaces", FORMAT),
		material: target("prepass material", FORMAT),
	}
}

/// The copy bit a test reads a target back through: a test's and only a
/// test's, the way the depth buffer's is.
const fn copied() -> TextureUsages {
	if cfg!(test) {
		TextureUsages::COPY_SRC
	} else {
		TextureUsages::empty()
	}
}

/// Four half floats a texel of a buffer this pass or a reader of it wrote, read
/// back, top row first. A test's.
///
/// @param device - to build the staging buffer on
/// @param queue - to submit the copy on
/// @param view - a view of the whole of an `Rgba16Float` texture
#[cfg(test)]
pub(crate) fn halves(
	device: &Device,
	queue: &wgpu::Queue,
	view: &TextureView,
) -> Option<Vec<[f32; 4]>> {
	let bytes =
		crate::depth::copied_out(device, queue, view.texture(), wgpu::TextureAspect::All, 8)?;

	Some(
		bytes
			.chunks_exact(8)
			.map(|texel| {
				[0, 2, 4, 6].map(|at| {
					let low = texel.get(at).copied().unwrap_or(0);
					let high = texel.get(at + 1).copied().unwrap_or(0);

					crate::post::half_into(u16::from_le_bytes([low, high]))
				})
			})
			.collect(),
	)
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			Decal, EntityId, Material, MaterialId, MeshData, MeshId, MeshVertex, Pose,
			Renderable, SkinVertex, Texel, TextureData, TextureId, Transform, Value, mesh,
			skeleton::{Bone, SkeletonData},
		},
		glam::{Mat4, Quat, Vec2, Vec3},
	};

	use super::*;
	use crate::{Capture, cover, depth, occlusion, reflection, scene::MSAA};

	/// How big every capture here is.
	const SIZE: (u32, u32) = (320, 240);

	/// How far a component of a normal or a roughness read back may be from one
	/// worked out and still be the same number.
	///
	/// The worked-out one follows the shader step for step in single precision,
	/// so what is left is the sixteen-bit float the buffer holds - and **the
	/// device truncates on the way into one rather than rounding**, measured: a
	/// roughness of 0.6 comes back 3.9e-4 low, which is 1228 steps of two to
	/// the minus eleventh where 1228.8 were asked for. So a whole step is the
	/// error to allow and not half of one. **Measured, then set at four times
	/// it**: the worst seen over every texel these tests read is 4.7e-4.
	const WITHIN: f32 = 2.0e-3;

	/// How far a depth this pass wrote may be from the one the scene wrote.
	///
	/// **Measured at nought**: bit for bit at every pixel, under both graphics
	/// APIs on this machine, with nothing asking the two pipelines to compute a
	/// position identically - the same vertex entry point over the same buffers
	/// was enough. The allowance is the depth view's own, a few steps of a
	/// float that close to one, so a device that rounds one pipeline
	/// differently from the other is not called a failure here; the scene
	/// testing against this depth would be, and that is the day this number
	/// has to become nought.
	const DEPTH_WITHIN: f32 = 1.2e-6;

	/// The bytes of the normal map every material without one reads.
	///
	/// **Not straight out**: a hundred and twenty-eight of two hundred and
	/// fifty-five is a hair over a half, so every surface leans a fifth of a
	/// degree. What is worked out here leans with it, because what is written
	/// does.
	const FLAT: [u8; 3] = [128, 128, 255];

	/// A capture on the binary's one device, or `None` with no GPU.
	fn capture() -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, SIZE.0, SIZE.1) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// How many samples a pixel is drawn with, and what the view draws.
	///
	/// **And no share of the sky taken away and no reflections mixed in**,
	/// either of which would ask for this pass in every frame: what these
	/// tests are about is the pass as the view or a test asks for it, so
	/// nothing else may ask.
	fn asking(world: &mut World, samples: &str, showing: &str) {
		world.cvars.var(MSAA, Value::Float(1.0), "");
		world.cvars.set(MSAA, samples);
		world.cvars.var(VIEW, Value::Float(NO_VIEW), "");
		world.cvars.set(VIEW, showing);
		world
			.cvars
			.var(occlusion::STRENGTH, Value::Float(occlusion::DEFAULT_STRENGTH), "");
		world.cvars.set(occlusion::STRENGTH, "0");
		world
			.cvars
			.var(reflection::STRENGTH, Value::Float(reflection::DEFAULT_STRENGTH), "");
		world.cvars.set(reflection::STRENGTH, "0");
		// and nothing left out for being behind something nearer: that is one
		// pass more wherever the pass before the scene runs, and these tests
		// count passes
		world
			.cvars
			.var(cover::ENABLED, Value::Bool(true), "");
		world.cvars.set(cover::ENABLED, "false");
	}

	/// A camera eight units back from the origin and looking at it.
	fn stage() -> World {
		let mut world = World::new();

		world.camera.position = Vec3::new(0.0, 0.0, 8.0);
		world.camera.target = Vec3::ZERO;

		world
	}

	/// An entity drawing a mesh in a material, standing somewhere.
	fn place(world: &mut World, mesh: MeshId, material: MaterialId, at: Transform) -> EntityId {
		let id = world.entities.spawn_at(at);

		world
			.entities
			.set_renderable(id, Renderable::of(mesh, material, Vec3::ONE));

		id
	}

	/// A material that is the default one but for how rough it is.
	fn rough(world: &mut World, name: &str, roughness: f32) -> MaterialId {
		world
			.materials
			.insert(name, Material { roughness, ..Material::DEFAULT })
	}

	/// A four-texel-square picture of one texel repeated.
	fn picture(world: &mut World, name: &str, texel: Texel, bytes: [u8; 4]) -> TextureId {
		world.textures.insert(name, TextureData {
			width: 4,
			height: 4,
			faces: 1,
			texel,
			levels: vec![bytes.repeat(16)],
		})
	}

	/// The first vertex of a mesh facing a way, which on a flat face stands for
	/// every vertex of it: they share a normal and a tangent.
	fn facing(data: &MeshData, normal: Vec3) -> MeshVertex {
		data.vertices
			.iter()
			.find(|vertex| Vec3::from_array(vertex.normal).abs_diff_eq(normal, 1.0e-6))
			.copied()
			.expect("the mesh has a face that way")
	}

	/// The normal `shading_normal` in `shader.wgsl` lights a vertex of a flat
	/// face with, step for step: the normal carried by the matrix after one
	/// over the square of the scale, the tangent carried as it is and squared
	/// up against it, and the map's direction laid along the two.
	///
	/// @param model - what carries the vertex into the world, bones included
	/// @param scale - the entity's own scale
	/// @param vertex - the vertex
	/// @param texel - the normal map's bytes
	fn shaded(model: Mat4, scale: Vec3, vertex: MeshVertex, texel: [u8; 3]) -> Vec3 {
		let normal = model
			.transform_vector3(Vec3::from_array(vertex.normal) / (scale * scale))
			.normalize();
		let leaning = model.transform_vector3(vertex.tangent_axis());
		let along_u = (leaning - normal * normal.dot(leaning)).normalize();
		let along_v = normal.cross(along_u) * vertex.tangent[3];
		let [across, down, out] = numbers(texel);

		(along_u * across + along_v * down + normal * out).normalize()
	}

	/// A normal map's bytes as the direction the shader reads out of them.
	fn numbers(texel: [u8; 3]) -> [f32; 3] {
		texel.map(|byte| (f32::from(byte) / 255.0).mul_add(2.0, -1.0))
	}

	/// A decal's normal map laid on a surface, as `bent` in `shader.wgsl` lays
	/// it, and mixed in by how much of the decal lands.
	fn bent(normal: Vec3, right: Vec3, texel: [u8; 3], amount: f32) -> Vec3 {
		let flat = right - normal * normal.dot(right);
		let along_u = flat.normalize();
		let along_v = -normal.cross(along_u);
		let [across, down, out] = numbers(texel);
		let turned = (along_u * across + along_v * down + normal * out).normalize();

		normal.lerp(turned, amount).normalize()
	}

	/// Where a point of the world lands in the picture, as an index into a
	/// readback.
	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "a point asserted to be inside a three-hundred-and-twenty by \
		          two-hundred-and-forty picture, as the pixel it falls in"
	)]
	fn pixel_of(world: &World, point: Vec3) -> usize {
		let clip = world
			.render_camera()
			.view_projection(world.aspect)
			* point.extend(1.0);
		let across = (clip.x / clip.w).mul_add(0.5, 0.5) * 320.0;
		let down = (clip.y / clip.w).mul_add(-0.5, 0.5) * 240.0;

		assert!(
			(0.0..320.0).contains(&across) && (0.0..240.0).contains(&down),
			"{point} lands at ({across}, {down}), off the picture"
		);

		(down as usize) * 320 + across as usize
	}

	/// Points well inside one face of a unit cube: its middle and four more a
	/// fifth of the way towards its corners.
	fn inside(normal: Vec3) -> [Vec3; 5] {
		let (first, second) = if normal.x.abs() > 0.5 {
			(Vec3::Y, Vec3::Z)
		} else if normal.y.abs() > 0.5 {
			(Vec3::X, Vec3::Z)
		} else {
			(Vec3::X, Vec3::Y)
		};
		let middle = normal * 0.5;

		[(0.0, 0.0), (0.2, 0.2), (-0.2, 0.2), (0.2, -0.2), (-0.2, -0.2)]
			.map(|(one, other)| middle + first * one + second * other)
	}

	/// How far a texel read back is from a normal and a roughness, as the
	/// largest difference on any of its four numbers.
	fn off(texel: [f32; 4], normal: Vec3, roughness: f32) -> f32 {
		let [x, y, z, w] = texel;

		(Vec3::new(x, y, z) - normal)
			.abs()
			.max_element()
			.max((w - roughness).abs())
	}

	/// One texel of a readback.
	fn at(values: &[[f32; 4]], index: usize) -> [f32; 4] {
		values
			.get(index)
			.copied()
			.expect("the index is inside the picture")
	}

	/// A slanted square: a unit across and a unit up, its face turned half way
	/// between `+x` and `+z`, and its tangents worked out the way every mesh
	/// gets them.
	///
	/// **What a box cannot test.** A box's faces lie along its own axes, and an
	/// axis stretched or squashed along itself points the same way afterwards,
	/// so a normal carried by the plain model matrix and one carried the right
	/// way round come out equal on every face of every box. A face at a slant
	/// is where the two part.
	fn ramp() -> MeshData {
		let normal = Vec3::new(1.0, 0.0, 1.0).normalize();
		let across = Vec3::new(1.0, 0.0, -1.0).normalize();
		let mut data = MeshData::default();

		for (u, v) in [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)] {
			let position = across * (u - 0.5) + Vec3::Y * (0.5 - v);

			data.vertices
				.push(MeshVertex::new(position, normal, Vec2::new(u, v)));
		}

		data.indices.extend([0, 1, 2, 0, 2, 3]);
		mesh::tangents(&mut data);

		data
	}

	#[test]
	fn a_turned_box_a_stretched_slant_and_a_mapped_face_write_the_normal_each_is_lit_with() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();
		let cube = mesh::cube();
		let slant = ramp();
		let slanted = world.meshes.insert("test/ramp", slant.clone());

		// three faces of a box at a slant to everything, uniformly scaled: the
		// corner between them turned to face the camera, so each is seen at the
		// same angle and none of them edge on
		let turned = Transform {
			position: Vec3::new(-2.2, 1.0, 0.0),
			rotation: corner(),
			scale: Vec3::splat(1.6),
		};
		// a slanted face under a stretch, and a roughness smoother than anything
		// is drawn at
		let stretched = Transform {
			position: Vec3::new(2.2, 1.0, 0.0),
			rotation: Quat::from_rotation_y(-0.3),
			scale: Vec3::new(2.0, 1.5, 1.0),
		};
		// a face square to the camera, turned about its own normal so its
		// tangent is square to nothing, with a normal map that leans a long way
		let mapped = Transform {
			position: Vec3::new(0.0, -1.8, 0.0),
			rotation: Quat::from_rotation_z(0.25),
			scale: Vec3::new(3.0, 1.0, 1.0),
		};
		let leaning = [200, 128, 230];
		let bumps =
			picture(&mut world, "test/lean_normal", Texel::Rgba8Unorm, [200, 128, 230, 255]);

		let first = rough(&mut world, "test/turned", 0.6);
		let second = rough(&mut world, "test/stretched", 0.02);
		let third = world.materials.insert(
			"test/mapped",
			Material { roughness: 0.35, ..Material::DEFAULT }.bumped(bumps),
		);

		place(&mut world, MeshId::CUBE, first, turned);
		place(&mut world, slanted, second, stretched);
		place(&mut world, MeshId::CUBE, third, mapped);

		asking(&mut world, "1", "0");
		capture.scene_mut().prepare_anyway(true);
		capture
			.shoot(&mut world)
			.expect("the capture renders");

		let values = capture
			.scene_mut()
			.surface_values()
			.expect("asked for, so written");
		let eye = world.render_camera().position;
		let mut faces = 0;

		for normal in [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z] {
			let model = turned.matrix();
			let middle = model.transform_point3(normal * 0.5);
			let expected = shaded(model, turned.scale, facing(&cube, normal), FLAT);

			if expected.dot((eye - middle).normalize()) < 0.3 {
				continue;
			}

			faces += 1;

			for point in inside(normal).map(|point| model.transform_point3(point)) {
				let texel = at(&values, pixel_of(&world, point));

				assert!(
					off(texel, expected, 0.6) <= WITHIN,
					"the box's face {normal} wrote {texel:?} at {point}, where {expected} and \
					 0.6 were worked out"
				);
			}
		}

		assert_eq!(faces, 3, "the turn shows three faces of the box");

		let model = stretched.matrix();
		let expected = shaded(model, stretched.scale, facing(&slant, slant_normal()), FLAT);
		let naive = model
			.transform_vector3(slant_normal())
			.normalize();

		assert!(
			(expected - naive).length() > 0.3,
			"the stretch is one that tells the right normal from the plain matrix's: {expected} \
			 against {naive}"
		);

		for (one, other) in
			[(0.0, 0.0), (0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25)]
		{
			let point = model
				.transform_point3(Vec3::new(1.0, 0.0, -1.0).normalize() * one + Vec3::Y * other);
			let texel = at(&values, pixel_of(&world, point));

			assert!(
				off(texel, expected, 0.045) <= WITHIN,
				"the slant wrote {texel:?} at {point}, where {expected} and the smoothest a \
				 surface is drawn at were worked out"
			);
		}

		let model = mapped.matrix();
		let expected = shaded(model, mapped.scale, facing(&cube, Vec3::Z), leaning);

		for point in inside(Vec3::Z).map(|point| model.transform_point3(point)) {
			let texel = at(&values, pixel_of(&world, point));

			assert!(
				off(texel, expected, 0.35) <= WITHIN,
				"the mapped face wrote {texel:?} at {point}, where {expected} and 0.35 were \
				 worked out"
			);
		}
	}

	/// Which way the slanted square faces, in its own space.
	fn slant_normal() -> Vec3 { Vec3::new(1.0, 0.0, 1.0).normalize() }

	/// A turn that brings a cube's corner between `+x`, `+y` and `+z` round to
	/// face down `+z`, so a camera looking back along it sees three faces
	/// alike.
	fn corner() -> Quat { Quat::from_rotation_arc(Vec3::ONE.normalize(), Vec3::Z) }

	#[test]
	fn the_material_is_the_color_a_surface_is_lit_with_and_how_metal_it_is() {
		// the entity's tint times the material's color times the picture's texel
		// undone out of sRGB, and how metal it is as the material says - what
		// `shade` lights, and what a reflection that finds this surface lights
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();
		let texel = [128, 64, 255, 255];
		let picture = picture(&mut world, "test/albedo", Texel::Rgba8Srgb, texel);
		let base = Vec3::new(0.5, 1.0, 0.8);
		let tint = Vec3::new(0.9, 0.6, 0.4);
		let material = world.materials.insert("test/painted", Material {
			base_color: base,
			metallic: 0.7,
			roughness: 0.4,
			..Material::textured(picture)
		});
		let id = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: corner(),
			scale: Vec3::splat(2.0),
		});

		world
			.entities
			.set_renderable(id, Renderable::of(MeshId::CUBE, material, tint));
		asking(&mut world, "1", "0");
		capture.scene_mut().prepare_anyway(true);
		capture
			.shoot(&mut world)
			.expect("the capture renders");

		let values = capture
			.scene_mut()
			.material_values()
			.expect("asked for, so written");
		let undone = |byte: u8| {
			let level = f32::from(byte) / 255.0;

			if level <= 0.04045 {
				level / 12.92
			} else {
				((level + 0.055) / 1.055).powf(2.4)
			}
		};
		let wanted =
			tint * base * Vec3::new(undone(texel[0]), undone(texel[1]), undone(texel[2]));
		let middle = at(&values, pixel_of(&world, Vec3::ZERO));

		assert!(
			(Vec3::new(middle[0], middle[1], middle[2]) - wanted)
				.abs()
				.max_element()
				<= WITHIN,
			"the color written is {middle:?}, where {wanted} is what the surface is lit with"
		);
		assert!(
			(middle[3] - 0.7).abs() <= WITHIN,
			"and how metal it is, {}, where the material says 0.7",
			middle[3]
		);
		assert!(
			at(&values, 0)
				.iter()
				.all(|value| value.abs() < f32::EPSILON),
			"and nought in the corner, where nothing is"
		);
	}

	#[test]
	fn a_decal_turns_the_normal_and_the_roughness_inside_its_box_and_nowhere_else() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();
		let cube = mesh::cube();
		let slab = Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(6.0, 4.0, 0.2),
		};
		let under = rough(&mut world, "test/slab", 0.3);
		let texel = [60, 200, 220];
		let bumps =
			picture(&mut world, "test/splash_normal", Texel::Rgba8Unorm, [60, 200, 220, 255]);
		let splash = world.materials.insert(
			"test/splash",
			Material { roughness: 0.9, ..Material::DEFAULT }.bumped(bumps),
		);
		let thrown = Transform {
			position: Vec3::new(1.5, 0.0, 0.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(1.5, 1.5, 1.0),
		};

		place(&mut world, MeshId::CUBE, under, slab);

		let decal = world.entities.spawn_at(thrown);

		world
			.entities
			.set_renderable(decal, Renderable { material: splash, ..Renderable::NOTHING });
		world.entities.set_decal(decal, Decal::BOX);

		asking(&mut world, "1", "0");
		capture.scene_mut().prepare_anyway(true);
		capture
			.shoot(&mut world)
			.expect("the capture renders");

		let values = capture
			.scene_mut()
			.surface_values()
			.expect("asked for, so written");
		let plain = shaded(slab.matrix(), slab.scale, facing(&cube, Vec3::Z), FLAT);
		let outside = at(&values, pixel_of(&world, Vec3::new(-1.5, 0.0, 0.1)));

		assert!(
			off(outside, plain, 0.3) <= WITHIN,
			"outside the box the slab keeps its own normal and roughness: {outside:?}"
		);

		// the slab's face is a tenth of a unit into a box a unit deep, which is a
		// fifth of the way to its face, and a decal fades towards its faces by
		// the eighth power of that
		let amount = 1.0 - 0.2_f32.powi(8);
		let right = thrown
			.matrix()
			.inverse()
			.row(0)
			.truncate()
			.normalize();
		let expected = bent(plain, right, texel, amount);
		let inside = at(&values, pixel_of(&world, Vec3::new(1.5, 0.0, 0.1)));

		assert!(
			(expected - plain).length() > 0.3,
			"the decal's map is one that turns a normal a long way: {expected} against {plain}"
		);
		assert!(
			off(inside, expected, 0.6_f32.mul_add(amount, 0.3)) <= WITHIN,
			"inside the box the decal's map and roughness were written: {inside:?}, where \
			 {expected} was worked out"
		);
	}

	#[test]
	fn a_hole_in_a_masked_picture_and_a_pane_of_glass_both_show_what_is_behind_them() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();
		let cube = mesh::cube();
		let wall = Transform {
			position: Vec3::new(0.0, 0.0, -2.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(10.0, 6.0, 0.2),
		};
		let back = rough(&mut world, "test/wall", 0.8);
		// the left half of every row a hole and the right half solid, so the
		// left half of a face facing the camera is see-through
		let holes = world.textures.insert("test/holes", TextureData {
			width: 4,
			height: 4,
			faces: 1,
			texel: Texel::Rgba8Srgb,
			levels: vec![
				[255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255]
					.repeat(4),
			],
		});
		let masked = world.materials.insert("test/masked", Material {
			blend: Blend::Mask,
			roughness: 0.5,
			..Material::textured(holes)
		});
		let glass = world.materials.insert("test/glass", Material {
			blend: Blend::Alpha,
			opacity: 0.5,
			roughness: 0.2,
			..Material::DEFAULT
		});
		let panel = |x| Transform {
			position: Vec3::new(x, 0.0, 0.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(2.0, 2.0, 0.1),
		};

		place(&mut world, MeshId::CUBE, back, wall);
		place(&mut world, MeshId::CUBE, masked, panel(-1.5));
		place(&mut world, MeshId::CUBE, glass, panel(1.5));

		asking(&mut world, "1", "0");
		capture.scene_mut().prepare_anyway(true);
		capture
			.shoot(&mut world)
			.expect("the capture renders");

		let values = capture
			.scene_mut()
			.surface_values()
			.expect("asked for, so written");
		let behind = shaded(wall.matrix(), wall.scale, facing(&cube, Vec3::Z), FLAT);
		let front = shaded(panel(-1.5).matrix(), panel(-1.5).scale, facing(&cube, Vec3::Z), FLAT);

		for (point, normal, roughness, what) in [
			(Vec3::new(-2.1, 0.0, 0.05), behind, 0.8, "the hole in the masked panel"),
			(Vec3::new(-0.9, 0.0, 0.05), front, 0.5, "the solid half of the masked panel"),
			(Vec3::new(1.5, 0.0, 0.05), behind, 0.8, "the pane of glass"),
		] {
			let texel = at(&values, pixel_of(&world, point));

			assert!(
				off(texel, normal, roughness) <= WITHIN,
				"{what} wrote {texel:?}, where {normal} and {roughness} were worked out"
			);
		}
	}

	#[test]
	fn a_limb_its_bones_turned_writes_the_normal_its_pose_gave_it() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();

		// above and to the left, where the top face leans once the bone turns
		world.camera.position = Vec3::new(-3.0, 5.0, 8.0);

		let mut rigid = mesh::cube();
		rigid.skin = vec![SkinVertex::rigid(0); rigid.vertices.len()];

		let cube = rigid.clone();
		let limb = world.meshes.insert("test/limb", rigid);
		let skeleton = world.skeletons.insert("test/rig", SkeletonData {
			bones: vec![Bone {
				name: "root".to_owned(),
				..Bone::default()
			}],
		});
		let pose = world
			.poses
			.spawn(Pose::resting(skeleton, world.skeletons.bones(skeleton)));
		let material = rough(&mut world, "test/limb", 0.7);
		let id = world.entities.spawn_at(Transform::at(Vec3::ZERO));

		world
			.entities
			.set_renderable(id, Renderable::of(limb, material, Vec3::ONE).posed(pose));

		// a turn that is not a quarter: a cube turned a quarter lands on the
		// shape it started as, and only its tangents would tell
		let bend = Quat::from_rotation_z(0.5);

		world.advance();
		world
			.poses
			.get_mut(pose)
			.expect("the pose is there")
			.set(0, Transform { rotation: bend, ..Transform::IDENTITY });
		world.poses.snap_all();
		world.settle();

		asking(&mut world, "1", "0");
		capture.scene_mut().prepare_anyway(true);
		capture
			.shoot(&mut world)
			.expect("the capture renders");

		let values = capture
			.scene_mut()
			.surface_values()
			.expect("asked for, so written");
		let model = Mat4::from_quat(bend);
		let expected = shaded(model, Vec3::ONE, facing(&cube, Vec3::Y), FLAT);

		for point in inside(Vec3::Y).map(|point| model.transform_point3(point)) {
			let texel = at(&values, pixel_of(&world, point));

			assert!(
				off(texel, expected, 0.7) <= WITHIN,
				"the posed top face wrote {texel:?} at {point}, where {expected} was worked out"
			);
		}
	}

	#[test]
	fn what_is_written_is_one_sample_a_pixel_whatever_the_picture_is_drawn_with() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();
		let material = rough(&mut world, "test/turned", 0.6);

		place(&mut world, MeshId::CUBE, material, Transform {
			position: Vec3::ZERO,
			rotation: corner(),
			scale: Vec3::splat(2.0),
		});
		capture.scene_mut().prepare_anyway(true);

		let mut read = |samples: &str| {
			asking(&mut world, samples, "0");
			capture
				.shoot(&mut world)
				.expect("the capture renders");

			(
				capture
					.scene_mut()
					.surface_values()
					.expect("asked for, so written"),
				capture
					.scene_mut()
					.prepass_depth_values()
					.expect("and its depth beside it"),
			)
		};

		let one = read("1");
		let four = read("4");
		let bits = |values: &[[f32; 4]]| -> Vec<u32> {
			values
				.iter()
				.flat_map(|texel| texel.map(f32::to_bits))
				.collect()
		};

		assert!(
			bits(&one.0) == bits(&four.0),
			"the surfaces at four samples are the surfaces at one, bit for bit"
		);
		assert!(
			one.1
				.iter()
				.copied()
				.map(f32::to_bits)
				.eq(four.1.iter().copied().map(f32::to_bits)),
			"and so is the depth"
		);
	}

	#[test]
	fn the_depth_this_pass_writes_is_the_depth_the_scene_writes_at_one_sample() {
		// the same geometry through the same vertex entry points, so the same
		// depths - which is what says the two passes drew the same triangles in
		// the same places
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();
		let material = rough(&mut world, "test/turned", 0.6);

		place(&mut world, MeshId::CUBE, material, Transform {
			position: Vec3::new(1.5, 0.2, 0.0),
			rotation: corner(),
			scale: Vec3::splat(3.0),
		});
		place(&mut world, MeshId::SPHERE, material, Transform {
			position: Vec3::new(-2.5, -0.5, 1.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::splat(3.0),
		});
		// a wall behind both, spawned after them so that it is the later
		// instance of the cube's own batch and drawn after the cube: a pass
		// that wrote whatever came last instead of whatever is nearest would
		// paint the wall over them, and nothing else in these tests draws a
		// farther surface after a nearer one
		place(&mut world, MeshId::CUBE, material, Transform {
			position: Vec3::new(0.0, 0.0, -4.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(24.0, 14.0, 0.5),
		});

		asking(&mut world, "1", "0");
		capture.scene_mut().prepare_anyway(true);
		capture
			.shoot(&mut world)
			.expect("the capture renders");

		let written = capture
			.scene_mut()
			.prepass_depth_values()
			.expect("asked for, so written");
		let scene = capture
			.scene_mut()
			.depth_values()
			.expect("at one sample the scene's depth is readable");
		let near = written
			.iter()
			.filter(|value| **value < 1.0)
			.count();
		let worst = written
			.iter()
			.zip(&scene)
			.map(|(one, other)| (one - other).abs())
			.fold(0.0_f32, f32::max);

		assert!(near > 10_000, "the two shapes cover the picture: {near} pixels");
		assert!(worst <= DEPTH_WITHIN, "the two depths are out by {worst:e} at worst");
	}

	#[test]
	fn the_rectangle_the_picture_is_cut_to_is_the_rectangle_the_surfaces_are_written_in() {
		// a window with tools around its picture draws the world into the
		// middle, and a buffer written over the whole target instead would put
		// every surface somewhere the picture is not. The scene's depth is the
		// witness: both passes are cut to one rectangle, so both depths agree
		// everywhere, the pixels outside it included.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();
		let material = rough(&mut world, "test/turned", 0.6);
		let view = crate::Viewport { x: 160, y: 40, width: 150, height: 190 };

		place(&mut world, MeshId::CUBE, material, Transform {
			position: Vec3::ZERO,
			rotation: corner(),
			scale: Vec3::splat(4.0),
		});
		asking(&mut world, "1", "0");
		capture.scene_mut().prepare_anyway(true);
		capture.draw_within(&mut world, view);

		let written = capture
			.scene_mut()
			.surface_values()
			.expect("asked for, so written");
		let depth = capture
			.scene_mut()
			.prepass_depth_values()
			.expect("and its depth beside it");
		let scene = capture
			.scene_mut()
			.depth_values()
			.expect("at one sample the scene's depth is readable");
		let (mut inside, mut outside) = (0_u32, 0_u32);

		for (at, texel) in written.iter().enumerate() {
			let (column, row) =
				(u32::try_from(at % 320).unwrap_or(0), u32::try_from(at / 320).unwrap_or(0));
			let within = (view.x..view.x + view.width).contains(&column)
				&& (view.y..view.y + view.height).contains(&row);
			let drawn = texel[3] > 0.0;

			inside += u32::from(within && drawn);
			outside += u32::from(!within && drawn);
		}

		let worst = depth
			.iter()
			.zip(&scene)
			.map(|(one, other)| (one - other).abs())
			.fold(0.0_f32, f32::max);

		assert!(inside > 10_000, "the cube fills the rectangle: {inside} texels written in it");
		assert_eq!(outside, 0, "and nothing is written outside it");
		assert!(worst <= DEPTH_WITHIN, "the two depths are out by {worst:e} at worst");
	}

	#[test]
	fn the_pass_runs_only_while_something_asks_and_lets_its_buffers_go() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();
		let material = rough(&mut world, "test/turned", 0.6);

		place(&mut world, MeshId::CUBE, material, Transform::at(Vec3::ZERO));

		let mut frame = |world: &mut World, asked: bool, samples: &str| {
			asking(world, samples, "0");
			capture.scene_mut().prepare_anyway(asked);
			capture.draw(world, &mut []);

			(
				capture.scene_mut().spans().passes(),
				capture.scene_mut().surface_values().is_some(),
			)
		};

		for samples in ["4", "1"] {
			let quiet = frame(&mut world, false, samples);
			let asked = frame(&mut world, true, samples);
			let again = frame(&mut world, false, samples);

			assert_eq!(asked.0, quiet.0 + 1, "at {samples} samples the pass is one pass");
			assert_eq!(again.0, quiet.0, "and it goes when nothing asks");
			assert!(
				!quiet.1 && asked.1 && !again.1,
				"the buffers are only held while asked: {quiet:?} {asked:?} {again:?}"
			);
		}
	}

	#[test]
	fn a_shader_put_in_after_the_pass_was_built_is_the_one_the_pass_writes_with() {
		// the pass's pipelines are built from the scene's own source the first
		// frame something asks, so a shader edited after that frame has to reach
		// them as well as the picture's table - or a normal is written by one
		// shader and lit by another
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();
		let material = rough(&mut world, "test/turned", 0.6);

		place(&mut world, MeshId::CUBE, material, Transform {
			position: Vec3::ZERO,
			rotation: corner(),
			scale: Vec3::splat(3.0),
		});
		asking(&mut world, "1", "0");
		capture.scene_mut().prepare_anyway(true);
		capture.draw(&mut world, &mut []);

		let source = include_str!("shader.wgsl");
		let anchor = "vec4<f32>(surface.normal, clamp(surface.roughness, MIN_ROUGHNESS, 1.0)),";

		assert_eq!(source.matches(anchor).count(), 1, "the line this test edits has moved");

		capture
			.scene_mut()
			.set_shader(&source.replace(anchor, "vec4<f32>(0.0, 0.0, 1.0, 0.25),"))
			.expect("the edited shader compiles");
		capture.draw(&mut world, &mut []);

		let written = capture
			.scene_mut()
			.surface_values()
			.expect("asked for, so written");
		let middle = at(&written, 120 * 320 + 160);

		assert!(
			off(middle, Vec3::Z, 0.25) <= WITHIN,
			"the pass wrote with the shader put in after it was built: {middle:?}"
		);
	}

	#[test]
	fn asking_for_the_pass_moves_no_pixel_of_the_picture() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();
		let material = rough(&mut world, "test/turned", 0.6);

		world.light = Vec3::new(-0.4, -1.0, -0.5);
		place(&mut world, MeshId::CUBE, material, Transform {
			position: Vec3::ZERO,
			rotation: corner(),
			scale: Vec3::splat(2.0),
		});

		for samples in ["4", "1"] {
			asking(&mut world, samples, "0");
			capture.scene_mut().prepare_anyway(false);

			let plain = capture
				.shoot(&mut world)
				.expect("the capture renders");

			capture.scene_mut().prepare_anyway(true);

			let asked = capture
				.shoot(&mut world)
				.expect("the capture renders");

			assert!(
				plain.pixels == asked.pixels,
				"at {samples} samples a frame with the pass in it is the same picture"
			);
		}
	}

	#[test]
	fn the_view_draws_a_normal_and_a_roughness_as_bytes_and_black_where_nothing_is() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();
		let cube = mesh::cube();
		let material = rough(&mut world, "test/turned", 0.6);
		let turned = Transform {
			position: Vec3::ZERO,
			rotation: corner(),
			scale: Vec3::splat(2.0),
		};

		place(&mut world, MeshId::CUBE, material, turned);

		let model = turned.matrix();
		let normal = shaded(model, turned.scale, facing(&cube, Vec3::Z), FLAT);
		let byte = |level: f32| (level * 255.0).round();

		for samples in ["4", "1"] {
			asking(&mut world, samples, "1");

			let image = capture
				.shoot(&mut world)
				.expect("the capture renders");
			let index = pixel_of(&world, model.transform_point3(Vec3::Z * 0.5));
			let (column, row) = (
				u32::try_from(index % 320).unwrap_or(0),
				u32::try_from(index / 320).unwrap_or(0),
			);
			let seen = image.pixel(column, row);

			for (channel, axis) in [normal.x, normal.y, normal.z]
				.into_iter()
				.enumerate()
			{
				let wanted = byte(axis.mul_add(0.5, 0.5));

				assert!(
					(f32::from(seen[channel]) - wanted).abs() <= 1.0,
					"at {samples} samples channel {channel} of the face is {}, not {wanted}",
					seen[channel]
				);
			}

			assert_eq!(image.pixel(2, 2), [0, 0, 0, 255], "and the corner sees nothing at all");

			asking(&mut world, samples, "2");

			let image = capture
				.shoot(&mut world)
				.expect("the capture renders");
			let seen = image.pixel(column, row);

			assert!(
				seen[0].abs_diff(153) <= 1 && seen[0] == seen[1] && seen[1] == seen[2],
				"at {samples} samples a roughness of 0.6 is the grey 153, and it is {seen:?}"
			);
		}
	}

	#[test]
	fn a_view_that_is_not_one_to_seven_draws_the_picture_and_the_depth_view_wins() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();
		let material = rough(&mut world, "test/turned", 0.6);

		place(&mut world, MeshId::CUBE, material, Transform::at(Vec3::ZERO));
		asking(&mut world, "4", "0");

		let picture = capture
			.shoot(&mut world)
			.expect("the capture renders");

		for asked in ["8", "-1", "0.25", "7.5"] {
			asking(&mut world, "4", asked);

			let again = capture
				.shoot(&mut world)
				.expect("the capture renders");

			assert!(again.pixels == picture.pixels, "a view of {asked} draws the picture");
		}

		world
			.cvars
			.var(depth::VIEW, Value::Float(depth::NO_VIEW), "");
		world.cvars.set(depth::VIEW, "12.5");
		asking(&mut world, "4", "0");

		let depth = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let alone = capture.scene_mut().spans().passes();

		asking(&mut world, "4", "1");

		let both = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert!(both.pixels == depth.pixels, "with both asked for, the depth is what is drawn");
		assert_eq!(
			capture.scene_mut().spans().passes(),
			alone,
			"and a view nobody sees asks for no pass"
		);
	}

	#[test]
	fn a_resized_scene_writes_what_it_writes_at_the_new_size() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = stage();
		let material = rough(&mut world, "test/turned", 0.6);

		place(&mut world, MeshId::CUBE, material, Transform::at(Vec3::ZERO));
		capture.scene_mut().prepare_anyway(true);

		for samples in ["4", "1"] {
			asking(&mut world, samples, "0");
			capture.draw(&mut world, &mut []);
			capture.scene_mut().resize(160, 120);
			capture.draw(&mut world, &mut []);

			let written = capture
				.scene_mut()
				.surface_values()
				.expect("asked for, so written");

			assert_eq!(
				written.len(),
				160 * 120,
				"at {samples} samples the buffer is the new size"
			);

			capture.scene_mut().resize(SIZE.0, SIZE.1);
		}
	}
}
