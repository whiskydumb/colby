//! Everything about drawing a world that does not involve a window.
//!
//! Split out from [`Renderer`](crate::Renderer) so that the drawing does not
//! know where it is going. A window is one destination; an offscreen texture is
//! the other, and that one can be looked at by a test. @ref
//! [`capture`](crate::capture).
//!
//! Three of the world's tables become GPU resources here - meshes, textures and
//! materials - and all three are kept level with the world the same way: each
//! uploaded thing remembers the registry revision it was built from, and a
//! frame that finds the two disagreeing rebuilds it. Nothing tells the renderer
//! that an asset changed. It looks.
//!
//! Nothing here reads a transform or a camera straight out of the world,
//! either. The simulation runs at a fixed rate and this runs at the display's,
//! so every pose comes through [`World::render_transform`] and
//! [`World::render_camera`], which place it between the last two simulated
//! states. That is the whole of the renderer's part in the fixed timestep.

use core::mem::offset_of;

use colby_core::{
	Result,
	abi::{
		EntityId, Light, LightKind, MAX_ENTITIES, Material, MeshData, MeshVertex, Meshes,
		SkinVertex, Texel, TextureData, TextureId, Textures, Transform, World,
		material::{Blend, MaterialEntry, Wrap},
		registry::Entry,
	},
	bytemuck::{self, Pod, Zeroable},
	err, error,
	glam::Vec3,
	info, warn,
};
use wgpu::{
	AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
	BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
	Buffer, BufferAddress, BufferBindingType, BufferDescriptor, BufferUsages, Color,
	ColorTargetState, ColorWrites, CommandEncoder, CommandEncoderDescriptor, CompareFunction,
	DepthBiasState, DepthStencilState, Device, ErrorFilter, Extent3d, Face, FilterMode,
	FragmentState, FrontFace, IndexFormat, LoadOp, MipmapFilterMode, MultisampleState,
	Operations, Origin3d, PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode,
	PrimitiveState, PrimitiveTopology, Queue, RenderPass, RenderPassColorAttachment,
	RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPassTimestampWrites,
	RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor,
	ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState, StoreOp,
	TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
	TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
	TextureViewDescriptor, TextureViewDimension, VertexAttribute, VertexBufferLayout,
	VertexFormat, VertexState, VertexStepMode,
};

use crate::{
	gpu::Gpu,
	lines::Lines,
	post,
	shader::Shader,
	shadow::{self, CASCADES, Cascades, Maps},
	skin::Joints,
	sparks::Sparks,
	timing::{Ends, Pass, Timings, Work},
};

/// The depth format. Thirty-two bits is more than a scene this size needs and
/// is supported everywhere, which is worth more right now than the memory.
pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

/// The part of a target a scene is drawn into, in physical pixels from the
/// top left.
///
/// A window with tools around its picture draws the world into the middle
/// and leaves the edges to whatever the tools paint there; a window with no
/// tools, a picture and a test draw into the whole target. What is drawn is
/// the same either way - the projection is built for the rectangle's own
/// shape, so nothing stretches - and the target is cleared edge to edge
/// first, because a load operation is not something a scissor cuts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Viewport {
	/// The left edge.
	pub x: u32,

	/// The top edge.
	pub y: u32,

	/// How wide.
	pub width: u32,

	/// How tall.
	pub height: u32,
}

impl Viewport {
	/// The whole of a target.
	///
	/// @param width - the target's width in pixels
	/// @param height - its height
	#[must_use]
	pub const fn whole(width: u32, height: u32) -> Self { Self { x: 0, y: 0, width, height } }

	/// This rectangle cut down to what lies inside a target.
	///
	/// A rectangle that reaches past the target is a validation error rather
	/// than a picture, and a window that was resized a frame ago hands out
	/// exactly that for one frame.
	///
	/// @param width - the target's width in pixels
	/// @param height - its height
	/// @return the part inside, or `None` if none of it is
	#[must_use]
	pub fn within(self, width: u32, height: u32) -> Option<Self> {
		let x = self.x.min(width);
		let y = self.y.min(height);
		let right = self.x.saturating_add(self.width).min(width);
		let bottom = self.y.saturating_add(self.height).min(height);

		if right <= x || bottom <= y {
			return None;
		}

		Some(Self {
			x,
			y,
			width: right - x,
			height: bottom - y,
		})
	}

	/// Width over height, which is what the projection is built for.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		clippy::cast_precision_loss,
		reason = "u32 to f32 loses precision above 2^24, which is four thousand times wider \
		          than any target"
	)]
	pub fn aspect(self) -> f32 { self.width.max(1) as f32 / self.height.max(1) as f32 }
}

/// How many local lights one frame may carry.
///
/// Matched by `MAX_LAMPS` in `shader.wgsl`, which sizes the uniform this
/// fills. Thirty-two is the smallest number in the field that is a *frame*
/// budget rather than a tile's: Unreal's forward grid allows thirty-two per
/// sixty-four-pixel cell, Godot's mobile path eight per object. A whole-frame
/// array is a weaker mechanism than either, so the number it carries is the
/// generous end rather than the mean one.
pub const MAX_LAMPS: usize = 32;

/// What [`LAMPS`] holds until somebody sets it, as a console variable holds it.
///
/// Written out rather than converted from [`MAX_LAMPS`], so that the whole
/// number and the number a variable holds are the same literal and the const
/// below is what checks they agree.
pub const DEFAULT_LAMPS: f32 = 32.0;

/// The console variable that says how many of them a frame actually sends.
///
/// A ceiling on the ceiling, so that the cost of a room full of lamps can be
/// measured rather than guessed at, and so that a machine which cannot afford
/// the loop has somewhere to say so.
pub const LAMPS: &str = "r.lights";

/// The console variable that says how many samples a pixel is drawn with.
///
/// **Off or four**, and the two are the only answers: @ref
/// [`post::SAMPLES`](crate::post::SAMPLES) for why wgpu makes it so. Anything
/// above one is read as four rather than refused, so `r.msaa 8` on a machine
/// that would like eight gets the four it can have instead of an error nobody
/// can act on.
///
/// A number rather than a switch, because the day two and eight are asked for
/// behind a feature this reads the same and means more.
pub const MSAA: &str = "r.msaa";

/// How many samples a pixel is drawn with until somebody says otherwise.
///
/// **On**, which is bevy's answer (`Msaa::Sample4` is its `#[default]`,
/// `bevy_render/src/view/mod.rs:248-253`) and not Godot's or Wicked's, and it
/// is the one that follows from what this renderer is: colby is forward, and
/// forward is the one shading path Unreal *forces* MSAA for when you pick it
/// (`RendererSettings.cpp:202-206`) and the only one it allows MSAA in at all
/// (`SceneUtils.h:49`). A world of hard-edged boxes with no anti-aliasing
/// looks unfinished, and the cost is four times the bandwidth of one pass in
/// a renderer whose passes nobody has yet measured.
pub const DEFAULT_MSAA: f32 = 4.0;

/// One local light, as the shader reads it.
///
/// Three vectors, and the kind is not one of them: a cone is
/// `saturate(cos * scale + offset)` and a point is that line with a scale of
/// nought and an offset of one. @ref `Lamp` in `shader.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct Lamp {
	/// `[x, y, z, range]`.
	position_range: [f32; 4],

	/// `[r, g, b, cone scale]`, the color already multiplied by the intensity.
	color: [f32; 4],

	/// `[x, y, z, cone offset]`, the way a cone points.
	direction: [f32; 4],
}

impl Lamp {
	/// A lamp that is not there, for the tail of the array.
	const DARK: Self = Self {
		position_range: [0.0; 4],
		color: [0.0; 4],
		direction: [0.0, 0.0, -1.0, 1.0],
	};

	/// One light in the world, packed.
	///
	/// @param light - what it shines
	/// @param at - where it stands and which way it is turned, in the world
	fn of(light: Light, at: Transform) -> Self {
		let mut packed = Self {
			position_range: at.position.extend(light.range).to_array(),
			color: (light.color * light.intensity)
				.extend(0.0)
				.to_array(),
			direction: (at.rotation * Vec3::NEG_Z)
				.normalize_or(Vec3::NEG_Z)
				.extend(1.0)
				.to_array(),
		};

		if light.kind == LightKind::Spot {
			let (inner, outer) = light.cone();
			let (cos_inner, cos_outer) = (inner.cos(), outer.cos());
			// one over the width of the falloff band, and the offset that puts
			// the far edge of it at nought. Filament's two numbers, worked out
			// here so the shader does no trigonometry per fragment.
			let scale = 1.0 / (cos_inner - cos_outer).max(1.0e-4);

			packed.color[3] = scale;
			packed.direction[3] = -cos_outer * scale;
		}

		packed
	}
}

/// What the shader needs to know that is neither per-vertex nor per-instance.
///
/// @note: the `crate` attribute points the derive at colby_core's re-export.
/// Without it the generated code says `::bytemuck` and every crate deriving a
/// bytemuck trait would need its own dependency on it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct Globals {
	view_projection: [[f32; 4]; 4],
	/// Clip space back into the world, for the sky.
	inverse_view_projection: [[f32; 4]; 4],
	light: [f32; 4],
	ambient: [f32; 4],
	/// xyz is where the camera is; w is unused. Needed by anything that depends
	/// on the viewing angle, which is all of the specular term.
	eye: [f32; 4],
	/// xyz is the direction it looks in; w is unused. A fragment projects
	/// itself onto this to get its own view depth, which is what picks a
	/// cascade - the same quantity the slices were cut on.
	forward: [f32; 4],
	/// World space into each cascade's clip space, nearest slice first.
	light_view_projection: [[[f32; 4]; 4]; CASCADES],
	/// The view depth each cascade stops at.
	splits: [f32; CASCADES],
	/// How many world units one texel of each cascade covers.
	cascade_texels: [f32; CASCADES],
	/// `[one texel in map coordinates, unused, shadows on, tint by cascade]`.
	shadow: [f32; 4],

	/// `[r, g, b, how quickly a surface fades with distance]`.
	fog: [f32; 4],

	/// `[r, g, b, whether a sky is drawn]` straight up.
	sky_zenith: [f32; 4],

	/// `[r, g, b, unused]` at eye level.
	sky_horizon: [f32; 4],

	/// `[r, g, b, unused]` straight down.
	sky_ground: [f32; 4],

	/// `[how many lamps are real, unused, unused, unused]`.
	counts: [u32; 4],

	/// The local lights, nearest first; the rest is [`Lamp::DARK`].
	lamps: [Lamp; MAX_LAMPS],
}

/// One entity, flattened into what the vertex stage reads.
///
/// The material's numbers ride along per instance rather than living in a
/// uniform buffer of their own. They are four floats; a buffer per material
/// would mean a binding per material for the sake of sixteen bytes, and the
/// bind group that does exist is only there because a texture cannot travel in
/// a vertex attribute.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct Placement {
	model: [[f32; 4]; 4],
	/// The material's base color times the entity's own tint.
	tint: [f32; 4],
	/// `[metallic, roughness, uv scale x, uv scale y]`.
	surface: [f32; 4],
	/// One over the square of each axis of the entity's scale; `w` is unused.
	///
	/// This is the whole of the normal matrix, and it is three floats rather
	/// than a `mat3` because a [`Transform`](colby_core::abi::Transform) is a
	/// translation, a rotation and a scale rather than an arbitrary matrix. For
	/// `M = T * R * S` the matrix that carries normals is `(M^-1)^T = R *
	/// S^-1`, and `mat3(M) = R * S`, so `R * S^-1 = mat3(M) * S^-2`. The
	/// shader therefore multiplies the normal by this before the model matrix
	/// and gets the exact answer for three multiplies and sixteen bytes,
	/// instead of an inverse and a transpose per vertex or a second matrix per
	/// instance.
	///
	/// An axis scaled to nothing has no reciprocal, so a zero is written as a
	/// one: a flattened entity draws with the normals it had rather than with
	/// infinities.
	normal_scale: [f32; 4],

	/// `[where this instance's joint matrices start, how many, 0, 0]`.
	///
	/// Read by the skinned pipeline and by nothing else; the static one
	/// declares the attribute and never looks at it, which a pipeline allows.
	/// Zero and zero is a thing bones do not move - @ref
	/// [`NO_JOINTS`](crate::skin::NO_JOINTS).
	skin: [u32; 4],
}

/// One mesh, uploaded.
///
/// `revision` is the world registry's revision at the time these buffers were
/// filled. That one number is the whole of asset hot-reload on this side: the
/// host rewrites a registry entry when its file changes, the numbers stop
/// matching, and the next frame re-uploads.
struct GpuMesh {
	vertices: Buffer,
	indices: Buffer,
	/// The bones and weights, for a mesh that has them.
	skin: Option<Buffer>,
	index_count: u32,
	revision: u32,

	/// The middle of the mesh's bounds, in the shape it was modeled in.
	///
	/// Kept here rather than asked for per frame because
	/// [`MeshData::bounds`](colby_core::abi::MeshData::bounds) walks every
	/// vertex, and what wants it - sorting the blended half of a frame - would
	/// ask once per entity per frame. It is worked out once per upload
	/// instead, which is once per asset reload.
	///
	/// The middle rather than the origin, which is what the two engines that
	/// publish their sort key both use: a floor slab modeled from a corner
	/// would otherwise sort as though it were at that corner.
	center: Vec3,
}

/// One texture, uploaded, with its whole mip chain.
pub(crate) struct GpuTexture {
	pub(crate) view: TextureView,
	pub(crate) revision: u32,
}

/// Which texture a material's bind group is holding a view of.
///
/// A slot and a revision rather than the view itself, because that is what
/// staleness is measured in: the group holds a view of one particular texture,
/// and re-uploading an image makes a new one that this group knows nothing
/// about.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Bound {
	slot: u32,
	revision: u32,
}

/// One material's bind group.
///
/// Three things it can go stale for, all unrelated: the material itself
/// changed, or either of the two textures it names was re-uploaded.
struct GpuMaterial {
	bindings: BindGroup,
	material_revision: u32,
	albedo: Bound,
	normal: Bound,
}

/// How finely a blended surface's distance is measured before it is sorted.
///
/// A thousandth of a world unit, which is [`Fyrox`]-shaped: fine enough that
/// two panes a hand apart never tie, coarse enough that a row of identical
/// props at one distance does tie and stays one batch. The sort is by distance
/// first and by mesh and material second, so a tie is the only thing that can
/// give the blended half any instancing at all.
///
/// [`Fyrox`]: https://fyrox.rs
const DEPTH_GRAIN: f32 = 1000.0;

/// One entity, in the order it is going to be written into the frame.
///
/// Sorted by everything in it, in this order, which is what makes one pass over
/// the sorted list produce both halves of the frame with every batch
/// contiguous. @ref [`Scene::group`].
#[derive(Clone, Copy)]
struct Sorted {
	/// Nought for the solid half and one for the blended one, so everything
	/// solid is written first and nothing has to be walked twice.
	///
	/// @note: what this really buys is that each list's placements are one run
	/// of the instance buffer, because a batch is a `first` and a `count` into
	/// it. Nothing else guarantees that - and **no test reaches the case**,
	/// because in every world anybody has built the blended half is in front of
	/// the camera and so has a negative key, which puts it before the solid
	/// half's nought anyway. It bites for a blended surface at or behind the
	/// eye, where the two would interleave and a batch would draw its
	/// neighbor's instances.
	pass: u8,

	/// Minus the view depth in thousandths, so that ascending is far to near.
	///
	/// Always nought in the solid half, which is not sorted by distance at all:
	/// the depth buffer settles that, and sorting it would cost the batching
	/// for nothing.
	depth: i32,

	mesh: u32,
	material: u32,

	/// Its slot decides nothing but the order of two entities a frame could not
	/// otherwise tell apart, which is what keeps a frame the same picture
	/// twice - an unstable sort may put equal keys either way round.
	entity: EntityId,

	/// Worked out once here and used for both the pass and the batch, so the
	/// list an entity is in and the pipeline its batch is drawn with can never
	/// be two different answers.
	blend: Blend,
}

impl Sorted {
	/// What the list is ordered on, in order of priority.
	///
	/// The mode is not in it and does not have to be: `pass` is a function of
	/// it and comes first, so the two halves are already apart.
	const fn key(&self) -> (u8, i32, u32, u32, usize) {
		(self.pass, self.depth, self.mesh, self.material, self.entity.slot())
	}
}

/// A run of instances that share both a mesh and a material.
struct Batch {
	mesh: usize,
	material: usize,
	first: u32,
	count: u32,
	/// Whether bones move this mesh, which decides which pipeline draws it.
	///
	/// A property of the mesh rather than of the instance, so a batch is
	/// never half one and half the other: the geometry either carries a skin
	/// block or it does not.
	skinned: bool,

	/// How this material's alpha is read, which decides the other axis of the
	/// same table.
	///
	/// A property of the material for the reason above, and the material is
	/// half of what a batch is keyed on, so a batch is never half one mode and
	/// half another either.
	blend: Blend,
}

/// One pipeline per way of drawing the scene.
///
/// A table rather than a field each. The two axes are independent - bones do
/// not care how the alpha is read - and one of them grows, so a pair of named
/// fields would mean two more of them and a new arm at every call site the next
/// time a mode is added.
struct Pipelines {
	entries: [RenderPipeline; Blend::COUNT * 2],
	/// The one that draws what is behind the world.
	///
	/// Not in the table, because it is not a point on the table's two axes: it
	/// reads no vertex buffer, no material and no bone, and the whole of what
	/// it has in common with the six is the source file and the globals. In
	/// the same struct all the same, so that a shader edit rebuilds all seven
	/// together or none of them.
	sky: RenderPipeline,

	/// How many samples every one of them was built for.
	///
	/// Kept so the scene can tell whether the table it is holding still
	/// matches the target it is about to draw into: wgpu refuses a pass whose
	/// attachments disagree with its pipeline, and it refuses it as a
	/// validation error rather than as a wrong picture.
	samples: u32,
}

impl Pipelines {
	/// Builds every one of them, or reports the first complaint wgpu had.
	///
	/// All of them or none: half a table is a world where the crates were drawn
	/// by the new shader and the fences by the old one.
	///
	/// @param device - the device to build against
	/// @param format - the color format the fragment stage writes
	/// @param layouts - the bind group layouts, in group order
	/// @param source - the whole WGSL
	/// @return the table, or the shader compiler's complaint
	fn build(
		device: &Device,
		format: TextureFormat,
		layouts: &[&BindGroupLayout],
		source: &str,
		samples: u32,
	) -> Result<Self> {
		let at = |blend, skinned| {
			compile_pipeline(device, format, layouts, source, blend, skinned, samples)
		};

		Ok(Self {
			entries: [
				at(Blend::Opaque, false)?,
				at(Blend::Opaque, true)?,
				at(Blend::Mask, false)?,
				at(Blend::Mask, true)?,
				at(Blend::Alpha, false)?,
				at(Blend::Alpha, true)?,
			],
			sky: compile_sky(device, format, layouts, source, samples)?,
			samples,
		})
	}

	/// Which one draws a batch.
	///
	/// Indexed rather than looked up: [`Blend::row`] is a match over the whole
	/// enum and the array is exactly [`Blend::COUNT`] pairs long, so there is
	/// no pair this can miss.
	fn get(&self, blend: Blend, skinned: bool) -> &RenderPipeline {
		&self.entries[blend.row() * 2 + usize::from(skinned)]
	}
}

/// A device, a table of pipelines, the resources uploaded so far, and a
/// frame's buffers.
pub struct Scene {
	/// A share of the process's one device, and of its queue. @ref [`Gpu`],
	/// which is where both are made and the only place they are.
	device: Device,
	queue: Queue,
	pipelines: Pipelines,
	globals: Buffer,
	bindings: BindGroup,
	globals_layout: BindGroupLayout,
	material_layout: BindGroupLayout,
	/// One per [`Wrap`], in its discriminant order.
	samplers: [Sampler; 2],
	shader: Shader,

	/// The WGSL the table in hand was built from.
	///
	/// **Not `shader.source()`**, and the difference is the whole point: that
	/// is what is on disk, and this is what is on the GPU. A caller may install
	/// source of its own through [`set_shader`](Self::set_shader) - a test
	/// does, and so would a material editor - and anything that rebuilds the
	/// table for a reason of its own has to rebuild *that* rather than
	/// quietly reverting to the file.
	built: String,
	/// The target's size, which the depth buffer was built for and which a
	/// viewport is cut down to.
	size: (u32, u32),
	depth: TextureView,
	/// The depth array the light writes and the scene samples.
	shadows: Maps,
	/// This frame's light matrices, fitted in `upload` and drawn in `render`.
	cascades: Cascades,
	/// Whether the console left the shadow passes switched on this frame.
	shadowing: bool,
	/// The debug renderer, drawn into this scene's pass and its depth buffer.
	lines: Lines,
	/// The particle renderer, drawn into the same pass after everything else.
	sparks: Sparks,
	/// This frame's joint matrices, and where each pose's run is in them.
	joints: Joints,
	/// One per registry slot, in the same order, filled on demand.
	meshes: Vec<GpuMesh>,
	textures: Vec<GpuTexture>,
	materials: Vec<GpuMaterial>,
	instances: Buffer,
	/// Scratch the instance buffer is built in, kept so it allocates once.
	placements: Vec<Placement>,
	/// Which run of `placements` belongs to which mesh and material, for
	/// everything that writes depth.
	batches: Vec<Batch>,
	/// The same for what does not, drawn after it and in its own order.
	///
	/// A second list rather than a flag on the first, because the two are
	/// sorted on different keys and only one of them casts - so a single list
	/// would have to be split again at both of the places that walk it.
	blended: Vec<Batch>,
	/// Every drawn entity, sorted. Sorting twenty-byte keys and looking the
	/// entities up again beats sorting the hundred-and-twelve-byte placements.
	order: Vec<Sorted>,
	/// Every lit entity with how far its reach is from the eye, kept so it
	/// allocates once. @ref [`Scene::lamps`].
	lit: Vec<(f32, Lamp)>,
	/// The float target the world is drawn into, and everything that squeezes
	/// it back down. @ref [`post`](crate::post).
	post: post::Chain,
	/// Whether this frame draws a sky behind the world.
	///
	/// Read off the world once in [`Scene::upload`] rather than again in the
	/// pass, for the reason the shadow flag is: what the frame does and what
	/// its uniform says have to be the same answer.
	sky: bool,
	/// What each part of the frame costs, once somebody has asked.
	///
	/// Inert until [`Timings::start`], and inert forever on an adapter with no
	/// timestamps. @ref [`timing`](crate::timing).
	timings: Timings,
}

impl Scene {
	/// Builds the pipelines, the depth buffer and the bind group layouts.
	///
	/// @param gpu - the device to build against and the queue every upload
	/// goes through; a share of each is kept
	/// @param format - the color format the fragment stage writes
	/// @param width - the target's width in pixels
	/// @param height - the target's height in pixels
	pub fn new(gpu: &Gpu, format: TextureFormat, width: u32, height: u32) -> Result<Self> {
		let device = gpu.device().clone();
		let queue = gpu.queue().clone();

		let globals = device.create_buffer(&BufferDescriptor {
			label: Some("globals"),
			size: size_bytes::<Globals>(1)?,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let globals_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("globals"),
			entries: &[BindGroupLayoutEntry {
				binding: 0,
				visibility: ShaderStages::VERTEX_FRAGMENT,
				ty: BindingType::Buffer {
					ty: BufferBindingType::Uniform,
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			}],
		});

		let material_layout = material_layout(&device);

		let bindings = device.create_bind_group(&BindGroupDescriptor {
			label: Some("globals"),
			layout: &globals_layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: globals.as_entire_binding(),
			}],
		});

		// one per wrap mode rather than one per material: a sampler is a small
		// piece of fixed-function state with two settings anybody actually
		// wants, so the table is two entries long and is built once. A material
		// picks with an index. @ref [`Wrap`].
		let samplers =
			[build_sampler(&device, Wrap::Repeat), build_sampler(&device, Wrap::Clamp)];

		// before the maps and before the pipelines: the depth pass reads the
		// joints as its second group and the scene reads them as its fourth,
		// so the layout has to exist before either is built.
		let joints = Joints::new(&device)?;
		let shadows = Maps::new(&device, joints.layout(), &material_layout)?;
		let shader = Shader::new("shader.wgsl", include_str!("shader.wgsl"));
		let groups =
			[&globals_layout, &material_layout, shadows.sample_layout(), joints.layout()];
		// the six, the sky and the lines all draw into the float target rather
		// than into the window: what reaches the window is the composite, and
		// it is the only thing built for the window's own format.
		// one sample here whatever the console will say, because there is no
		// console yet: a `Scene` is built before the world it draws exists.
		// The first frame reads the variable and rebuilds if it has to, which
		// is the same path a person turning it on mid-run takes. @ref
		// `sampling`.
		let pipelines = Pipelines::build(
			&device,
			post::HDR_FORMAT,
			&groups,
			shader.source(),
			post::NO_SAMPLES,
		)?;
		let depth = depth_view(&device, post::NO_SAMPLES, width, height);
		let lines = Lines::new(&device, post::HDR_FORMAT, &globals_layout, post::NO_SAMPLES)?;
		let sparks = Sparks::new(&device, post::HDR_FORMAT, post::NO_SAMPLES);
		let post = post::Chain::new(&device, format, width, height)?;

		let instances = device.create_buffer(&BufferDescriptor {
			label: Some("placements"),
			size: size_bytes::<Placement>(MAX_ENTITIES)?,
			usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		Ok(Self {
			device,
			queue,
			pipelines,
			built: shader.source().to_owned(),
			globals,
			bindings,
			globals_layout,
			material_layout,
			samplers,
			shader,
			size: (width, height),
			depth,
			shadows,
			cascades: Cascades::NONE,
			shadowing: false,
			lines,
			sparks,
			joints,
			// nothing is uploaded until a frame says what the world holds: the
			// registries belong to the host, and a scene built before the host
			// has loaded its assets would only have to be rebuilt afterwards.
			meshes: Vec::new(),
			textures: Vec::new(),
			materials: Vec::new(),
			instances,
			placements: Vec::with_capacity(MAX_ENTITIES),
			batches: Vec::new(),
			blended: Vec::new(),
			order: Vec::with_capacity(MAX_ENTITIES),
			lit: Vec::with_capacity(MAX_LAMPS),
			sky: false,
			post,
			// the period is a property of the queue and never changes, so it
			// is read once here rather than per frame. It is nought on an
			// adapter with no timestamps, which the apparatus knows to treat
			// as "nothing was measured" rather than as "every pass is free".
			timings: Timings::new(gpu.queue().get_timestamp_period()),
		})
	}

	/// Rebuilds the depth buffer for a new target size.
	pub fn resize(&mut self, width: u32, height: u32) {
		self.size = (width, height);
		self.depth = depth_view(&self.device, self.pipelines.samples, width, height);
		self.post.resize(&self.device, width, height);
	}

	/// Rebuilds everything that has to agree about how many samples a pixel is.
	///
	/// **Three things have to agree or wgpu refuses the pass**: the color
	/// attachment, the depth attachment, and every pipeline recorded into it.
	/// So a change here is the target, the depth buffer and eight pipelines,
	/// and it is all of them or none - a table half rebuilt is a validation
	/// error on the next draw rather than a picture anybody can see is wrong.
	///
	/// Called at the top of every frame. It costs one console lookup and a
	/// comparison when the answer has not moved, which is every frame but the
	/// first and the one somebody turns it on in.
	///
	/// @param world - for the console variable
	fn sampling(&mut self, world: &World) {
		let asked = world
			.cvars
			.float(MSAA)
			.map_or_else(|| samples_of(DEFAULT_MSAA), samples_of);

		if asked == self.pipelines.samples {
			return;
		}

		let groups = [
			&self.globals_layout,
			&self.material_layout,
			self.shadows.sample_layout(),
			self.joints.layout(),
		];

		match Pipelines::build(&self.device, post::HDR_FORMAT, &groups, &self.built, asked) {
			| Ok(table) => {
				self.pipelines = table;
				// after the table and not before: if the lines refuse, the
				// scene's own pipelines are already the new count and the
				// pass would disagree with itself. This one cannot fail for a
				// reason the table did not, which is why it is a line rather
				// than a second arm.
				if let Err(complaint) = self.lines.set_samples(
					&self.device,
					post::HDR_FORMAT,
					&self.globals_layout,
					asked,
				) {
					warn!(%complaint, "the debug lines kept the pipelines they had");
				}

				// and the particles, on the same terms and for the same
				// reason: a pipeline built for another sample count is a
				// validation error rather than a picture, and a cloud that
				// kept the pair it had is the same "a picture with hard edges
				// beats no picture" the table itself takes.
				self.sparks.set_samples(post::HDR_FORMAT, asked);

				self.depth = depth_view(&self.device, asked, self.size.0, self.size.1);
				self.post.set_samples(&self.device, asked);

				info!(samples = asked, "the scene is drawn with this many samples a pixel");
			},
			// the source has not changed, so this is a device that will not
			// build a multisampled pipeline at all. Said once, and the old
			// table is kept: a picture with hard edges beats no picture.
			| Err(complaint) => {
				warn!(samples = asked, %complaint, "the pipelines could not be rebuilt");
			},
		}
	}

	/// Builds the pipelines from new shader source, keeping the ones that work
	/// if the new source does not compile.
	///
	/// They are only replaced once wgpu has confirmed every one of them is
	/// valid, so a shader with a typo in it costs a message and nothing else -
	/// the same bargain a game module that panics gets, and for the same
	/// reason: the code being edited is expected to be wrong sometimes.
	///
	/// @param source - the whole WGSL
	/// @return the compiler's complaint, if it had one
	pub fn set_shader(&mut self, source: &str) -> Result {
		let samples = self.pipelines.samples;
		let groups = [
			&self.globals_layout,
			&self.material_layout,
			self.shadows.sample_layout(),
			self.joints.layout(),
		];
		// the whole table, and none of it is assigned until all of it has
		// compiled: half a reload is a world where the crates moved and the
		// characters did not.
		self.pipelines =
			Pipelines::build(&self.device, post::HDR_FORMAT, &groups, source, samples)?;
		source.clone_into(&mut self.built);

		Ok(())
	}

	/// Uploads this frame and records it into a target.
	///
	/// @param target - what to draw into
	/// @param world - the state to draw
	/// @param view - the part of the target to draw into, or the whole of it;
	/// the projection is the caller's to match, through `world.aspect`
	pub fn render(
		&mut self,
		target: &TextureView,
		world: &World,
		view: Option<Viewport>,
		seconds: f32,
	) {
		self.timings.begin();
		self.reload_shader();
		self.sampling(world);

		self.timings.open(Work::Upload);
		self.upload(world);
		self.timings.close(Work::Upload);

		// from here to the submit, which is what the label means: how long
		// this thread takes to describe the frame, not how long the hardware
		// takes to run it. The two are separate answers and a frame can be
		// short of either. @ref [`timing`](crate::timing).
		self.timings.open(Work::Record);

		let mut encoder = self
			.device
			.create_command_encoder(&CommandEncoderDescriptor { label: Some("frame") });

		// first, and once per cascade: the scene's own pass samples what these
		// wrote, so they have to be recorded ahead of it. Skipped entirely when
		// the console has turned shadows off, which leaves the maps holding
		// whatever was in them and is safe because nothing then reads them.
		if self.shadowing {
			for slice in 0..CASCADES {
				// one span over all four rather than four spans: they are one
				// feature and one console variable, and what anybody wants to
				// know is what shadows cost.
				self.cast(&mut encoder, slice, self.timings.writes(Pass::Shadow, cascade(slice)));
			}
		}

		let scene_marks = self.timings.writes(Pass::Scene, Ends::Both);
		let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
			label: Some("scene"),
			color_attachments: &[Some(RenderPassColorAttachment {
				view: self.post.target(),
				depth_slice: None,
				// at the end of this pass rather than in one of its own, which
				// is what makes multisampling cost no pass at all: the
				// hardware averages the samples as the attachment is stored,
				// and everything after this reads the plain texture it wrote.
				resolve_target: self.post.resolve_into(),
				ops: Operations {
					load: LoadOp::Clear(clear_color(world)),
					// nothing reads the multisampled texture afterwards, so
					// the samples themselves need not survive the pass that
					// resolved them. On a tiled device that is the difference
					// between four samples living in tile memory and four
					// samples being written to main memory for nobody.
					store: if self.post.resolve_into().is_some() {
						StoreOp::Discard
					} else {
						StoreOp::Store
					},
				},
			})],
			depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
				view: &self.depth,
				// cleared to the far plane, which is one under wgpu's
				// zero-to-one depth range.
				depth_ops: Some(Operations {
					load: LoadOp::Clear(1.0),
					store: StoreOp::Store,
				}),
				stencil_ops: None,
			}),
			timestamp_writes: scene_marks,
			occlusion_query_set: None,
			multiview_mask: None,
		});

		// after the clear, which the pass has already done edge to edge, and
		// before anything is drawn. A rectangle with nothing inside the
		// target is a frame of clear color and nothing else, which is what a
		// window squeezed down to its tools has to show.
		if let Some(asked) = view {
			let Some(view) = asked.within(self.size.0, self.size.1) else {
				// a rectangle with nothing inside the target. The scene pass
				// has already cleared the float target edge to edge, and the
				// composite still has to run: what reaches the window is
				// written by it and by nothing else, so returning here would
				// leave the frame holding whatever was in it last.
				drop(pass);
				self.post.resolve(
					&mut encoder,
					&self.queue,
					world.post,
					seconds,
					target,
					&self.timings,
				);
				self.timings.resolve(&mut encoder);
				self.timings.close(Work::Record);
				self.queue.submit([encoder.finish()]);

				return;
			};

			#[expect(
				clippy::as_conversions,
				clippy::cast_precision_loss,
				reason = "pixel counts, nowhere near where f32 stops holding integers"
			)]
			pass.set_viewport(
				view.x as f32,
				view.y as f32,
				view.width as f32,
				view.height as f32,
				0.0,
				1.0,
			);
			pass.set_scissor_rect(view.x, view.y, view.width, view.height);
		}

		pass.set_bind_group(0, &self.bindings, &[]);
		pass.set_bind_group(2, self.shadows.bindings(), &[]);
		pass.set_bind_group(3, self.joints.bindings(), &[]);
		pass.set_vertex_buffer(1, self.instances.slice(..));

		self.draw(&mut pass, &self.batches);

		// after everything opaque and before everything else. Every pixel a
		// wall covered is thrown away by the depth test before it is shaded,
		// which is why this is here rather than in front of the batches; and
		// it is before the lines and the blended pass because both of those
		// composite with what is already in the target, and what is behind
		// them has to be there first.
		if self.sky {
			pass.set_pipeline(&self.pipelines.sky);
			pass.draw(0..3, 0..1);
		}

		// then, into the same pass and therefore against the same depth buffer:
		// whether a debug line is hidden by a wall is the most useful thing it
		// has to say, and an overlay is handed the color target alone.
		self.lines.draw(&mut pass, &self.bindings);

		// and last of all, over a finished picture. Anything blended composites
		// with what is already in the target, so everything that could be
		// behind it has to be there first - the debug lines included, which is
		// what makes a line seen through glass read as being behind it.
		self.draw(&mut pass, &self.blended);

		// and after even that. A particle writes no depth, so nothing it draws
		// can hold anything else out, and it composites - which means every
		// solid thing, every masked thing, every line and every pane of glass
		// has to be in the target before it.
		//
		// **The cost, said out loud: a particle behind a pane of glass is
		// drawn over the glass.** Two lists cannot interleave without being
		// one list, and no engine read here makes them one - Fyrox and Godot
		// sort their systems into the transparent queue as whole objects,
		// which has the same failure at emitter granularity rather than at
		// list granularity. It is `PT-1` on the audit list with a trigger
		// rather than a thing to fix now.
		self.sparks.draw(&mut pass, &self.bindings);

		drop(pass);

		// and last: the picture is measured, the eye moves, and the whole
		// thing is squeezed onto the screen. Everything above this line drew
		// into sixteen-bit floats, and nothing above it wrote a pixel of the
		// frame that is about to be shown.
		self.post
			.resolve(&mut encoder, &self.queue, world.post, seconds, target, &self.timings);

		// last of all, and into this frame's own encoder: the ten timestamps
		// are copied out of the query set where the passes that wrote them can
		// still be told apart. Nothing at all when nobody is measuring.
		self.timings.resolve(&mut encoder);
		self.timings.close(Work::Record);
		self.queue.submit([encoder.finish()]);
	}

	/// What this frame cost, for whoever asked to be told.
	///
	/// Blocks on the queue. @ref [`Timings::settle`] for why, and for why the
	/// mode that calls it opens no window.
	pub fn settle(&mut self) -> crate::timing::Frame {
		let device = self.device.clone();

		self.timings.settle(&device)
	}

	/// What a frame cost, for whoever asked without being able to wait.
	///
	/// **Does not block.** The window's half of [`settle`](Self::settle): it
	/// pumps the device's callbacks and hands back a frame two or three frames
	/// after the frame it is about, or nothing. @ref [`Timings::poll`].
	pub fn collect(&mut self) -> Option<crate::timing::Frame> {
		let device = self.device.clone();

		self.timings.poll(&device)
	}

	/// What this frame's wall clock said, without waiting for anything.
	///
	/// The half of a frame's cost that needs no readback: the two recording
	/// spans, ready the moment the frame has been recorded. The hardware half
	/// arrives later through [`collect`](Self::collect).
	pub fn spans(&self) -> crate::timing::Frame { self.timings.spans() }

	/// How many particles the last frame drew.
	///
	/// A count rather than a duration, and the second stable number
	/// `--profile` prints: a time is a fact about the afternoon, and how many
	/// particles a project has in the air at a given step is a fact about the
	/// project. A number that moved means somebody changed something.
	#[must_use]
	pub fn sparks(&self) -> usize { self.sparks.drawn() }

	/// Whether the particle pipelines have been built.
	///
	/// The one table here built lazily, and the one whose failure to build is
	/// a warning rather than an error - so this is what says a frame with a
	/// cloud in it drew the cloud rather than warning about it. @ref
	/// `Sparks::built`, which like this is only there for a test.
	#[cfg(test)]
	pub(crate) const fn sparks_built(&self) -> bool { self.sparks.built() }

	/// Whether anything is being measured.
	pub const fn measuring(&self) -> bool { self.timings.timing() }

	/// Starts measuring what a frame costs.
	///
	/// @return whether the hardware side came up; the wall clock works either
	/// way
	pub fn measure(&mut self) -> bool { self.timings.start(&self.device.clone()) }

	/// Stops measuring, and gives the query set and both buffers back.
	///
	/// The apparatus is documented as inert until it is started, and a panel
	/// that turns it on when it opens has to be able to turn it off when it
	/// closes - otherwise every window that ever showed the profiler goes on
	/// paying a query set and two buffers for the rest of the run.
	pub fn unmeasure(&mut self) { self.timings.stop(); }

	/// Which local lights this frame carries, nearest first.
	///
	/// The scratch list is the only thing about this that belongs to a
	/// `Scene`; the rule itself is [`chosen`], which needs no device and is
	/// therefore testable.
	///
	/// @param world - the world being drawn
	/// @param eye - where the camera is
	/// @return the array the uniform holds, and how many of it is real
	fn lamps(&mut self, world: &World, eye: Vec3) -> ([Lamp; MAX_LAMPS], u32) {
		let room = world
			.cvars
			.float(LAMPS)
			.map_or(MAX_LAMPS, lamp_room);

		chosen(world, eye, room, &mut self.lit)
	}

	/// Records one list of batches into a pass that is already set up.
	///
	/// Both halves of a frame go through this: what differs between them is the
	/// order they were sorted in and the pipelines their materials name, and
	/// neither of those is visible from here.
	///
	/// @param pass - the pass to record into, with groups nought, two and three
	/// already bound
	/// @param batches - the runs to draw, in order
	fn draw(&self, pass: &mut RenderPass<'_>, batches: &[Batch]) {
		// swapped when a batch wants another one rather than once per batch.
		// The solid half is ordered by mesh and material, so a world of crates
		// with one character in it changes pipeline twice however many crates
		// there are.
		let mut bound = None;

		for batch in batches {
			let (Some(mesh), Some(material)) =
				(self.meshes.get(batch.mesh), self.materials.get(batch.material))
			else {
				continue;
			};

			let wanted = (batch.blend, batch.skinned);
			if bound != Some(wanted) {
				pass.set_pipeline(self.pipelines.get(batch.blend, batch.skinned));
				bound = Some(wanted);
			}

			if let Some(skin) = mesh.skin.as_ref() {
				pass.set_vertex_buffer(2, skin.slice(..));
			}

			pass.set_bind_group(1, &material.bindings, &[]);
			pass.set_vertex_buffer(0, mesh.vertices.slice(..));
			pass.set_index_buffer(mesh.indices.slice(..), IndexFormat::Uint32);
			pass.draw_indexed(0..mesh.index_count, 0, batch.first..batch.first + batch.count);
		}
	}

	/// Which one draws a batch into a cascade.
	///
	/// A match rather than a lookup, so that a mode nobody has thought about
	/// yet is a compile error here on the day it is added rather than a
	/// fence-shaped hole in the light.
	///
	/// @return the pipeline, or nothing for a surface that does not cast
	fn casting(&self, blend: Blend, skinned: bool) -> Option<&RenderPipeline> {
		let masked = match blend {
			| Blend::Opaque => false,
			| Blend::Mask => true,
			// this is the second of two places that say a blended surface casts
			// nothing - the first being which list `group` filed it in, which
			// is why this pass never actually meets one. It is kept because it
			// is the line a reader comes here looking for.
			| Blend::Alpha => return None,
		};

		Some(self.shadows.casting(masked, skinned))
	}

	/// Records one cascade's depth pass.
	///
	/// The same batches the scene draws, through a pipeline with no fragment
	/// stage and no color target, so the whole pass is geometry against depth.
	///
	/// Walks the solid half of the frame only. A blended surface writes no
	/// depth and so has nothing to say about what a light can reach, which is
	/// what every engine checked does with one; here it is not a rule anywhere
	/// but a consequence of which list [`group`](Self::group) filed it in.
	///
	/// @param encoder - what to record into
	/// @param slice - which cascade, nearest first
	/// @param marks - which end of the shadow span this cascade carries, and
	/// `None` in a frame nobody is measuring
	fn cast(
		&self,
		encoder: &mut CommandEncoder,
		slice: usize,
		marks: Option<RenderPassTimestampWrites<'_>>,
	) {
		let (Some(layer), Some(slot)) = (self.shadows.layer(slice), self.shadows.slot(slice))
		else {
			return;
		};

		let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
			label: Some("shadow"),
			color_attachments: &[],
			depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
				view: layer,
				depth_ops: Some(Operations {
					load: LoadOp::Clear(1.0),
					store: StoreOp::Store,
				}),
				stencil_ops: None,
			}),
			timestamp_writes: marks,
			occlusion_query_set: None,
			multiview_mask: None,
		});

		pass.set_bind_group(0, slot, &[]);
		pass.set_bind_group(1, self.joints.bindings(), &[]);
		pass.set_vertex_buffer(1, self.instances.slice(..));

		// the same swap the scene pass makes, and it has to be made here too:
		// a character whose shadow were cast from its bind pose would stand in
		// one attitude and be shadowed in another.
		let mut bound = None;

		for batch in &self.batches {
			let (Some(mesh), Some(material)) =
				(self.meshes.get(batch.mesh), self.materials.get(batch.material))
			else {
				continue;
			};

			let Some(pipeline) = self.casting(batch.blend, batch.skinned) else {
				continue;
			};

			let wanted = (batch.blend, batch.skinned);
			if bound != Some(wanted) {
				pass.set_pipeline(pipeline);
				bound = Some(wanted);
			}

			// bound for every batch and not only the masked ones: the group is
			// declared on all four pipelines so that one of them may read it,
			// and a group a pipeline's layout declares has to be there.
			pass.set_bind_group(2, &material.bindings, &[]);

			if let Some(skin) = mesh.skin.as_ref() {
				pass.set_vertex_buffer(2, skin.slice(..));
			}

			pass.set_vertex_buffer(0, mesh.vertices.slice(..));
			pass.set_index_buffer(mesh.indices.slice(..), IndexFormat::Uint32);
			pass.draw_indexed(0..mesh.index_count, 0, batch.first..batch.first + batch.count);
		}
	}

	/// The device this scene was built on.
	#[must_use]
	pub const fn device(&self) -> &Device { &self.device }

	/// The queue every upload goes through.
	#[must_use]
	pub const fn queue(&self) -> &Queue { &self.queue }

	/// Rebuilds the pipeline if the shader file has been written.
	///
	/// Rate-limited inside [`Shader`], so this costs one `stat` every quarter
	/// second rather than one per frame.
	fn reload_shader(&mut self) {
		if !self.shader.changed() {
			return;
		}

		let source = self.shader.source().to_owned();
		match self.set_shader(&source) {
			| Ok(()) => info!(path = ?self.shader.path(), "shader reloaded"),
			| Err(error) =>
				error!(%error, "shader did not compile; keeping the pipeline that works"),
		}
	}

	/// Writes this frame's resources, globals and entity instances to the GPU.
	fn upload(&mut self, world: &World) {
		self.sync_meshes(&world.meshes);
		self.sync_textures(&world.textures);
		self.sync_materials(world);
		self.lines
			.upload(&self.device, &self.queue, world);
		// after `sync_textures`, which is what makes the picture a particle
		// names one this frame can bind: the emitter holds a registry handle
		// and the group it needs is over an uploaded view.
		self.sparks.upload(
			&self.device,
			&self.queue,
			world,
			&self.textures,
			&self.globals_layout,
		);

		// asked for once and used twice on purpose. This is where the frame
		// stops being the simulation's and becomes the picture's: the camera
		// the world holds is where the last step left it, and this one is
		// where it is *now*, part of the way to the next. Taking the matrix
		// from one and the eye position from the other would light the scene
		// from a camera that is not the one looking at it.
		let camera = world.render_camera();

		// the same camera the picture is drawn from, so the cascades cannot be
		// fitted to a pose the frame does not use. Off is off all the way to
		// the shader: nothing is drawn into the maps and nothing samples them.
		self.shadowing = world.cvars.bool(shadow::ENABLED).unwrap_or(true);
		self.cascades = if self.shadowing {
			let distance = world
				.cvars
				.float(shadow::DISTANCE)
				.unwrap_or(shadow::DEFAULT_DISTANCE);

			shadow::fit(&camera, world.aspect, world.light, distance)
		} else {
			Cascades::NONE
		};

		let mut light_view_projection = [[[0.0; 4]; 4]; CASCADES];
		for (slot, matrix) in light_view_projection
			.iter_mut()
			.zip(self.cascades.matrices)
		{
			*slot = matrix.to_cols_array_2d();
		}

		let (lamps, count) = self.lamps(world, camera.position);
		let projection = camera.view_projection(world.aspect);
		self.sky = world.sky.is_drawn();

		self.queue.write_buffer(
			&self.globals,
			0,
			bytemuck::bytes_of(&Globals {
				view_projection: projection.to_cols_array_2d(),
				inverse_view_projection: projection.inverse().to_cols_array_2d(),
				light: world.light.extend(0.0).to_array(),
				ambient: world.ambient.extend(0.0).to_array(),
				eye: camera.position.extend(1.0).to_array(),
				forward: (camera.target - camera.position)
					.normalize_or(Vec3::NEG_Z)
					.extend(0.0)
					.to_array(),
				light_view_projection,
				splits: self.cascades.splits,
				cascade_texels: self.cascades.texels,
				shadow: [
					1.0 / shadow::resolution(),
					0.0,
					if self.shadowing { 1.0 } else { 0.0 },
					if world.cvars.bool(shadow::TINT).unwrap_or(false) {
						1.0
					} else {
						0.0
					},
				],
				fog: world
					.post
					.fog
					.extend(world.post.fog_density.max(0.0))
					.to_array(),
				sky_zenith: world
					.sky
					.zenith
					.extend(if self.sky { 1.0 } else { 0.0 })
					.to_array(),
				sky_horizon: world.sky.horizon.extend(0.0).to_array(),
				sky_ground: world.sky.ground.extend(0.0).to_array(),
				counts: [count, 0, 0, 0],
				lamps,
			}),
		);

		if self.shadowing {
			self.shadows.upload(&self.queue, &self.cascades);
		}

		self.group(world);

		if self.placements.is_empty() {
			return;
		}

		self.queue
			.write_buffer(&self.instances, 0, bytemuck::cast_slice(&self.placements));
		self.joints.upload(&self.queue);
	}

	/// Brings the uploaded meshes level with the world's registry.
	///
	/// Runs every frame and normally does nothing: a slot whose revision has
	/// not moved is left alone. New slots are appended, changed ones have
	/// their buffers rebuilt, and the old buffers are freed when the `GpuMesh`
	/// they belonged to is dropped.
	fn sync_meshes(&mut self, meshes: &Meshes) {
		for (slot, mesh) in meshes.iter().enumerate() {
			if self
				.meshes
				.get(slot)
				.is_some_and(|uploaded| uploaded.revision == mesh.revision())
			{
				continue;
			}

			let uploaded = upload_mesh(&self.device, &self.queue, mesh.value(), mesh.revision());
			match self.meshes.get_mut(slot) {
				| Some(existing) => *existing = uploaded,
				| None => self.meshes.push(uploaded),
			}
		}
	}

	/// The same, for textures.
	fn sync_textures(&mut self, textures: &Textures) {
		for (slot, texture) in textures.iter().enumerate() {
			if self
				.textures
				.get(slot)
				.is_some_and(|uploaded| uploaded.revision == texture.revision())
			{
				continue;
			}

			let uploaded =
				upload_texture(&self.device, &self.queue, texture.value(), texture.revision());
			match self.textures.get_mut(slot) {
				| Some(existing) => *existing = uploaded,
				| None => self.textures.push(uploaded),
			}
		}
	}

	/// The same, for the bind group each material needs.
	///
	/// Rebuilt when the material moved *or* when either texture it names did,
	/// because a bind group holds a view of one particular texture and
	/// re-uploading an image makes a new one.
	fn sync_materials(&mut self, world: &World) {
		for (slot, entry) in world.materials.iter().enumerate() {
			let bound = |id| Bound {
				slot: TextureId::index(id),
				revision: world.textures.get(id).map_or(0, Entry::revision),
			};

			let (albedo, normal) = (bound(entry.value().albedo), bound(entry.value().normal));
			let current = self.materials.get(slot).is_some_and(|uploaded| {
				uploaded.material_revision == entry.revision()
					&& uploaded.albedo == albedo
					&& uploaded.normal == normal
			});

			if current {
				continue;
			}

			let Some(uploaded) = self.build_material(entry, albedo, normal) else {
				continue;
			};

			match self.materials.get_mut(slot) {
				| Some(existing) => *existing = uploaded,
				| None => self.materials.push(uploaded),
			}
		}
	}

	/// Builds one material's bind group.
	///
	/// A material naming a texture that has not been uploaded falls back to the
	/// texture the *handle it should have held* points at: slot zero for an
	/// albedo, which is the white texel, and the flat normal map for a normal.
	/// So a material pointing at nothing draws its own color on a surface that
	/// is as flat as its geometry, rather than failing to draw.
	fn build_material(
		&self,
		entry: &MaterialEntry,
		albedo: Bound,
		normal: Bound,
	) -> Option<GpuMaterial> {
		let uploaded = |bound: Bound, fallback: u32| {
			usize::try_from(bound.slot)
				.ok()
				.and_then(|slot| self.textures.get(slot))
				.or_else(|| {
					usize::try_from(fallback)
						.ok()
						.and_then(|slot| self.textures.get(slot))
				})
		};

		let color = uploaded(albedo, TextureId::NONE.index())?;
		let bumps = uploaded(normal, TextureId::FLAT_NORMAL.index())?;
		let sampler = self
			.samplers
			.get(usize::try_from(entry.value().wrap.code()).unwrap_or(0))
			.or_else(|| self.samplers.first())?;

		let bindings = self
			.device
			.create_bind_group(&BindGroupDescriptor {
				label: Some("material"),
				layout: &self.material_layout,
				entries: &[
					BindGroupEntry {
						binding: 0,
						resource: BindingResource::TextureView(&color.view),
					},
					BindGroupEntry {
						binding: 1,
						resource: BindingResource::Sampler(sampler),
					},
					BindGroupEntry {
						binding: 2,
						resource: BindingResource::TextureView(&bumps.view),
					},
				],
			});

		Some(GpuMaterial {
			bindings,
			material_revision: entry.revision(),
			albedo,
			normal,
		})
	}

	/// Lays every entity out in the instance buffer, grouped by mesh and
	/// material.
	///
	/// A sort rather than the counting pass this used to be. Counting works
	/// while the key is one small index; the key is a pair now, and a counter
	/// per combination would be a table of meshes times materials for the sake
	/// of the handful of pairs a scene actually uses.
	fn group(&mut self, world: &World) {
		let (meshes, materials) = (world.meshes.len(), world.materials.len());
		let camera = world.render_camera();
		let forward = (camera.target - camera.position).normalize_or(Vec3::NEG_Z);

		self.order.clear();
		for (id, _, renderable) in world.entities.iter() {
			let mesh = renderable.mesh.slot();
			if mesh == 0 || mesh >= meshes {
				continue;
			}

			let material = renderable
				.material
				.slot()
				.min(materials.saturating_sub(1));
			let (Ok(mesh), Ok(material)) = (u32::try_from(mesh), u32::try_from(material)) else {
				continue;
			};

			let blend = world
				.materials
				.get(renderable.material)
				.map_or(Material::DEFAULT.blend, |surface| surface.blend);
			let blended = blend == Blend::Alpha;

			self.order.push(Sorted {
				pass: u8::from(blended),
				// worked out only for the half that is sorted on it. The solid
				// half would pay a matrix multiply per entity per frame for a
				// number nothing then reads.
				depth: if blended {
					-grain(self.view_depth(world, id, mesh, camera.position, forward))
				} else {
					0
				},
				mesh,
				material,
				entity: id,
				blend,
			});
		}

		self.order.sort_unstable_by_key(Sorted::key);

		self.placements.clear();
		self.batches.clear();
		self.blended.clear();
		self.joints.begin(world);

		for index in 0..self.order.len() {
			self.place(world, index);
		}
	}

	/// How far along the view a mesh's middle stands.
	///
	/// The same quantity the cascades are cut on and the shader picks a slice
	/// with - a projection onto the direction the camera looks, rather than the
	/// distance to it. Two panes side by side are then at one depth, which is
	/// what somebody looking at them would say.
	///
	/// @param world - the transforms to draw with
	/// @param id - the entity being measured
	/// @param mesh - its uploaded geometry's slot
	/// @param eye - where the camera is this frame
	/// @param forward - the direction it looks, of unit length
	/// @return how far in front of the eye the mesh's middle is
	fn view_depth(
		&self,
		world: &World,
		id: EntityId,
		mesh: u32,
		eye: Vec3,
		forward: Vec3,
	) -> f32 {
		let Some(transform) = world.render_transform(id) else {
			return 0.0;
		};

		let center = usize::try_from(mesh)
			.ok()
			.and_then(|slot| self.meshes.get(slot))
			.map_or(Vec3::ZERO, |uploaded| uploaded.center);

		(transform.matrix().transform_point3(center) - eye).dot(forward)
	}

	/// Writes one entity of the sorted order into the instance buffer, opening
	/// a new batch when its pair differs from the one before it.
	fn place(&mut self, world: &World, index: usize) {
		let Some(Sorted { mesh, material, entity: id, blend, .. }) =
			self.order.get(index).copied()
		else {
			return;
		};

		// the transform to *draw* with, which is not the one the game wrote:
		// it is somewhere between that one and the one before it. @ref
		// [`World::render_transform`].
		let (Some(transform), Some(renderable)) =
			(world.render_transform(id), world.entities.renderable(id))
		else {
			return;
		};

		let surface = world
			.materials
			.get(renderable.material)
			.copied()
			.unwrap_or(Material::DEFAULT);

		let Ok(at) = u32::try_from(self.placements.len()) else {
			return;
		};

		self.placements.push(Placement {
			model: transform.matrix().to_cols_array_2d(),
			// the fourth channel is the material's opacity, which only the
			// blended pipeline's fragment stage reads. @ref
			// [`Material::opacity`].
			tint: (renderable.color * surface.base_color)
				.extend(surface.opacity)
				.to_array(),
			surface: [
				surface.metallic,
				surface.roughness,
				surface.uv_scale.x,
				surface.uv_scale.y,
			],
			normal_scale: normal_scale(transform.scale)
				.extend(0.0)
				.to_array(),
			// the first entity of the frame to name a pose is what gathers it;
			// the second finds the same run rather than a second copy of it.
			skin: self.joints.take(world, renderable.pose),
		});

		let (mesh, material) =
			(usize::try_from(mesh).unwrap_or(0), usize::try_from(material).unwrap_or(0));
		// asked of the uploaded geometry rather than of the entity: what
		// decides the pipeline is whether there are bones and weights to read,
		// and an entity naming a pose over a mesh that has none is drawn as
		// the shape it is.
		let skinned = self
			.meshes
			.get(mesh)
			.is_some_and(|uploaded| uploaded.skin.is_some());

		// the list the sort already put this entity in. Reading the mode again
		// here rather than taking the one `group` decided on would be two
		// answers to one question, and the day they differed a batch would be
		// drawn in a pass its neighbors are not in.
		let batches = if blend == Blend::Alpha {
			&mut self.blended
		} else {
			&mut self.batches
		};

		match batches.last_mut() {
			| Some(batch) if batch.mesh == mesh && batch.material == material => batch.count += 1,
			| _ => batches.push(Batch {
				mesh,
				material,
				first: at,
				count: 1,
				skinned,
				blend,
			}),
		}
	}
}

/// The layout every material's group is built against.
///
/// Lifted out of the constructor rather than written inline: a builder that
/// creates two layouts, two pipelines, a depth buffer and three tables is a
/// hundred lines of nothing, and this is the half of it with no logic at all.
///
/// @param device - the device to build against
fn material_layout(device: &Device) -> BindGroupLayout {
	device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("material"),
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
			// the normal map, sampled through the same sampler: it is the
			// same surface under the same unwrap, so a second one could
			// only ever disagree with the first.
			BindGroupLayoutEntry {
				binding: 2,
				visibility: ShaderStages::FRAGMENT,
				ty: BindingType::Texture {
					sample_type: TextureSampleType::Float { filterable: true },
					view_dimension: TextureViewDimension::D2,
					multisampled: false,
				},
				count: None,
			},
		],
	})
}

/// One sampler, for one wrap mode.
///
/// Anisotropy is sixteen, which is the highest every desktop backend supports
/// and is what makes a tiled floor at a grazing angle look like a floor rather
/// than like a smear. It needs all three filters to be linear, which they are.
fn build_sampler(device: &Device, wrap: Wrap) -> Sampler {
	let mode = match wrap {
		| Wrap::Repeat => AddressMode::Repeat,
		| Wrap::Clamp => AddressMode::ClampToEdge,
	};

	device.create_sampler(&SamplerDescriptor {
		label: Some("material"),
		address_mode_u: mode,
		address_mode_v: mode,
		address_mode_w: mode,
		mag_filter: FilterMode::Linear,
		min_filter: FilterMode::Linear,
		mipmap_filter: MipmapFilterMode::Linear,
		anisotropy_clamp: 16,
		..SamplerDescriptor::default()
	})
}

/// What a normal has to be multiplied by before the model matrix.
///
/// @ref [`Placement::normal_scale`] for why this is the whole normal matrix.
/// An axis of zero would divide by nothing, so it is left at one - a normal
/// that is merely wrong is a shading bug, and an infinite one is a triangle
/// that disappears.
///
/// @param scale - the entity's scale along each axis
/// @return one over the square of each, or one where that has no answer
fn normal_scale(scale: Vec3) -> Vec3 {
	let squared = scale * scale;

	Vec3::select(
		squared
			.abs()
			.cmpgt(Vec3::splat(f32::MIN_POSITIVE)),
		squared.recip(),
		Vec3::ONE,
	)
}

/// Uploads one mesh's geometry.
fn upload_mesh(device: &Device, queue: &Queue, data: &MeshData, revision: u32) -> GpuMesh {
	GpuMesh {
		vertices: create_buffer(
			device,
			queue,
			"mesh vertices",
			bytemuck::cast_slice(&data.vertices),
			BufferUsages::VERTEX,
		),
		indices: create_buffer(
			device,
			queue,
			"mesh indices",
			bytemuck::cast_slice(&data.indices),
			BufferUsages::INDEX,
		),
		// nothing at all rather than an empty buffer for a mesh nothing bends,
		// which is almost all of them: the buffer is what decides which
		// pipeline draws the mesh, so its absence has to be the same claim as
		// the absence of the block it came from.
		skin: data.is_skinned().then(|| {
			create_buffer(
				device,
				queue,
				"mesh skin",
				bytemuck::cast_slice(&data.skin),
				BufferUsages::VERTEX,
			)
		}),
		index_count: u32::try_from(data.indices.len()).unwrap_or(0),
		revision,
		center: {
			let (low, high) = data.bounds();

			(low + high) * 0.5
		},
	}
}

/// Uploads one texture and every level of its mip chain.
///
/// A level whose byte count does not match its size is skipped rather than
/// written, leaving it as whatever the texture was created holding. The
/// registry checks that before anything gets this far, @ref
/// [`TextureData::is_consistent`]; this is the second line of the same defense,
/// because the alternative is a validation error inside a driver.
fn upload_texture(
	device: &Device,
	queue: &Queue,
	data: &TextureData,
	revision: u32,
) -> GpuTexture {
	let levels = u32::try_from(data.levels.len())
		.unwrap_or(1)
		.max(1);
	let texture = device.create_texture(&TextureDescriptor {
		label: Some("material texture"),
		size: Extent3d {
			width: data.width.max(1),
			height: data.height.max(1),
			depth_or_array_layers: 1,
		},
		mip_level_count: levels,
		sample_count: 1,
		dimension: TextureDimension::D2,
		format: texel_format(data.texel),
		usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
		view_formats: &[],
	});

	for level in 0..levels {
		let Some(bytes) = usize::try_from(level)
			.ok()
			.and_then(|index| data.levels.get(index))
		else {
			continue;
		};

		let (width, height) = data.level_size(level);
		if bytes.len() != data.level_bytes(level) {
			continue;
		}

		queue.write_texture(
			TexelCopyTextureInfo {
				texture: &texture,
				mip_level: level,
				origin: Origin3d::ZERO,
				aspect: TextureAspect::All,
			},
			bytes,
			TexelCopyBufferLayout {
				offset: 0,
				bytes_per_row: Some(width * u32::try_from(data.texel.bytes()).unwrap_or(4)),
				rows_per_image: Some(height),
			},
			Extent3d { width, height, depth_or_array_layers: 1 },
		);
	}

	GpuTexture {
		view: texture.create_view(&TextureViewDescriptor::default()),
		revision,
	}
}

/// The wgpu format one of the ABI's texel layouts stands for.
const fn texel_format(texel: Texel) -> TextureFormat {
	match texel {
		| Texel::Rgba8Srgb => TextureFormat::Rgba8UnormSrgb,
		| Texel::Rgba8Unorm => TextureFormat::Rgba8Unorm,
	}
}

/// Creates a buffer holding exactly these bytes.
///
/// @note: written through the queue rather than mapped at creation. The mapped
/// path in wgpu 30 hands back a write-only view whose length has to match the
/// buffer exactly, and the empty meshes here do not have a length to match.
/// A buffer of size zero is not allowed either, so empty ones get four bytes
/// nobody reads - cheaper than a branch at every use.
fn create_buffer(
	device: &Device,
	queue: &Queue,
	label: &str,
	bytes: &[u8],
	usage: BufferUsages,
) -> Buffer {
	let size = BufferAddress::try_from(bytes.len())
		.unwrap_or(0)
		.max(4);
	let buffer = device.create_buffer(&BufferDescriptor {
		label: Some(label),
		size,
		usage: usage | BufferUsages::COPY_DST,
		mapped_at_creation: false,
	});

	if !bytes.is_empty() {
		queue.write_buffer(&buffer, 0, bytes);
	}

	buffer
}

/// Which end of the shadow span one cascade carries.
///
/// The first cascade opens it, the last closes it and the ones between are
/// only counted, so four passes are timed as the one feature they are. A build
/// with a single cascade is both ends of its own span, which is not a case
/// that exists today and is one `CASCADES` could be edited into.
///
/// @param slice - which cascade, nearest first
/// @return what this cascade's pass writes
const fn cascade(slice: usize) -> Ends {
	match (slice, CASCADES) {
		| (0, 1) => Ends::Both,
		| (0, _) => Ends::Open,
		| (at, all) if at + 1 == all => Ends::Close,
		| _ => Ends::Middle,
	}
}

/// Creates a depth buffer of a given size.
fn depth_view(device: &Device, samples: u32, width: u32, height: u32) -> TextureView {
	let texture = device.create_texture(&TextureDescriptor {
		label: Some("depth"),
		size: Extent3d {
			width: width.max(1),
			height: height.max(1),
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		// and never resolved: a depth buffer is read by the pass that writes
		// it and by nothing after it, which is why `Depth32Float` needing no
		// `MULTISAMPLE_RESOLVE` costs nothing here.
		sample_count: samples,
		dimension: TextureDimension::D2,
		format: DEPTH_FORMAT,
		usage: TextureUsages::RENDER_ATTACHMENT,
		view_formats: &[],
	});

	texture.create_view(&TextureViewDescriptor::default())
}

/// How many samples a console variable is asking for.
///
/// Anything above one is four, because four is the only count above one that
/// needs no feature. A number below one, a nought, or something that is not a
/// number at all is one sample - which is the answer that always works.
///
/// @param asked - what the variable holds
#[must_use]
fn samples_of(asked: f32) -> u32 {
	if asked.is_finite() && asked > 1.0 {
		post::SAMPLES
	} else {
		post::NO_SAMPLES
	}
}

/// The size in bytes of `count` values of `T`, as a buffer size.
fn size_bytes<T>(count: usize) -> Result<BufferAddress> {
	let bytes = size_of::<T>()
		.checked_mul(count)
		.ok_or_else(|| err!(Graphics("buffer size overflows")))?;

	BufferAddress::try_from(bytes)
		.map_err(|error| err!(Graphics("buffer size does not fit: {error}")))
}

/// The clear color the game asked for.
fn clear_color(world: &World) -> Color {
	let clear = world.clear.clamp(Vec3::ZERO, Vec3::ONE);

	Color {
		r: f64::from(clear.x),
		g: f64::from(clear.y),
		b: f64::from(clear.z),
		a: 1.0,
	}
}

/// A distance as a whole number of thousandths, for a sort key.
///
/// Saturating rather than wrapping: something two thousand units away is
/// further than anything a sort has to tell apart, and a wrap would put it in
/// front of everything.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "held inside the range of the target on the line above the cast, which is the \
	          check 	          the cast itself would not make"
)]
fn grain(depth: f32) -> i32 {
	let scaled = depth * DEPTH_GRAIN;
	if scaled.is_nan() {
		return 0;
	}

	scaled.clamp(f32::from(i16::MIN) * DEPTH_GRAIN, f32::from(i16::MAX) * DEPTH_GRAIN) as i32
}

/// Which lamps a frame carries, and in what order.
///
/// **The nearest by the edge of their reach, not by their middle.** A lamp
/// whose sphere the camera is standing inside scores negative and comes first
/// however far its origin is, which is the answer wanted: what decides whether
/// a light matters to a picture is whether the picture is in it. Sorting by
/// the origin would drop the huge lamp the room is lit by in favor of a small
/// one behind the eye.
///
/// There is no frustum test. A sphere behind the camera lights nothing, but
/// working that out costs six plane tests per lamp per frame to save a slot in
/// an array that is rarely full, and a lamp just off the edge of the screen
/// still lights what is on it through a surface facing away from the eye. When
/// a frame is measured to be spending real time in this loop the answer is a
/// light grid rather than a better sort.
///
/// @param world - the world being drawn
/// @param eye - where the camera is
/// @param room - how many the frame may carry
/// @param scratch - the caller's list, so this allocates nothing per frame
/// @return the array the uniform holds, and how many of it is real
fn chosen(
	world: &World,
	eye: Vec3,
	room: usize,
	scratch: &mut Vec<(f32, Lamp)>,
) -> ([Lamp; MAX_LAMPS], u32) {
	scratch.clear();

	for (id, ..) in world.entities.iter() {
		let Some(light) = world
			.entities
			.light(id)
			.copied()
			.filter(|it| it.is_lit())
		else {
			continue;
		};

		let Some(at) = world.render_transform(id) else {
			continue;
		};

		scratch.push(((at.position - eye).length() - light.range, Lamp::of(light, at)));
	}

	// `total_cmp` rather than a partial compare: a lamp at a nan distance is a
	// world that has blown up, and it should sort somewhere definite rather
	// than making the order depend on which pairs were compared.
	scratch.sort_by(|(near, _), (other, _)| near.total_cmp(other));

	let mut lamps = [Lamp::DARK; MAX_LAMPS];
	let mut count = 0;

	for (slot, (_, lamp)) in lamps
		.iter_mut()
		.zip(scratch.iter().take(room.min(MAX_LAMPS)))
	{
		*slot = *lamp;
		count += 1;
	}

	(lamps, u32::try_from(count).unwrap_or(0))
}

/// How many lamps a console variable is asking for.
///
/// A variable holds a number rather than a count, so this is where the two
/// meet. Anything below nought is none and anything above the ceiling is the
/// ceiling; a nan is the ceiling too, because a variable nobody meant to set
/// should not put the lights out.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "clamped to the array's own length first, so the cast cannot lose anything or go 	          negative"
)]
fn lamp_room(asked: f32) -> usize {
	if asked.is_nan() {
		return MAX_LAMPS;
	}

	#[expect(
		clippy::cast_precision_loss,
		reason = "thirty-two is exact in an f32"
	)]
	let ceiling = MAX_LAMPS as f32;

	asked.clamp(0.0, ceiling) as usize
}

/// Builds one render pipeline and reports whether wgpu accepted it.
///
/// wgpu's default answer to a bad shader is to log the error and hand back a
/// handle that fails at draw time - which, for source someone is editing while
/// the engine runs, means the picture silently stops. An error scope turns that
/// into a value the caller can act on.
///
/// @note: the popped future resolves immediately on a native backend; nothing
/// was submitted and nothing has to be polled, so blocking on it here does not
/// wait for the GPU.
///
/// @param device - the device to build against
/// @param format - the color format the fragment stage writes
/// @param layouts - the bind group layouts, in group order
/// @param source - the whole WGSL
/// @param blend - how the fragment stage reads the albedo's alpha
/// @param skinned - whether to build the variant that reads bones
/// @param samples - how many samples the target it draws into has
fn compile_pipeline(
	device: &Device,
	format: TextureFormat,
	layouts: &[&BindGroupLayout],
	source: &str,
	blend: Blend,
	skinned: bool,
	samples: u32,
) -> Result<RenderPipeline> {
	let scope = device.push_error_scope(ErrorFilter::Validation);
	let pipeline = build_pipeline(device, format, layouts, source, blend, skinned, samples);

	match pollster::block_on(scope.pop()) {
		| Some(complaint) => Err(err!(Graphics("{complaint}"))),
		| None => Ok(pipeline),
	}
}

/// Builds the sky's pipeline, and reports whether wgpu liked it.
///
/// The same arguments and the same error scope as [`compile_pipeline`]; what
/// differs is the whole of what a sky is. @ref [`build_sky`].
fn compile_sky(
	device: &Device,
	format: TextureFormat,
	layouts: &[&BindGroupLayout],
	source: &str,
	samples: u32,
) -> Result<RenderPipeline> {
	let scope = device.push_error_scope(ErrorFilter::Validation);
	let pipeline = build_sky(device, format, layouts, source, samples);

	match pollster::block_on(scope.pop()) {
		| Some(complaint) => Err(err!(Graphics("{complaint}"))),
		| None => Ok(pipeline),
	}
}

/// The pipeline that draws what is behind the world.
///
/// Three things about it are the whole design and each is deliberate.
///
/// **No vertex buffers.** The triangle is arithmetic on the vertex index, so
/// the draw is three vertices and nothing bound. @ref `vertex_sky`.
///
/// **The depth test on and the depth write off, comparing less-or-equal.** The
/// sky is emitted at the far plane, which is what the depth buffer was cleared
/// to, so it passes exactly where nothing was drawn and is discarded - before
/// the fragment stage - everywhere a wall already wrote a nearer depth. Writing
/// depth would achieve nothing and would stop the blended pass behind it.
///
/// **The globals and nothing else in its layout.** A pipeline layout is what a
/// pipeline *requires*, not what it may ignore, so declaring the scene's four
/// groups would mean the sky could not be drawn until a material had been
/// bound - and group one is bound per batch, so an empty world would refuse the
/// draw. That is a wgpu validation error rather than a black screen, and the
/// rendered test found it.
fn build_sky(
	device: &Device,
	format: TextureFormat,
	layouts: &[&BindGroupLayout],
	source: &str,
	samples: u32,
) -> RenderPipeline {
	let shader = device.create_shader_module(ShaderModuleDescriptor {
		label: Some("sky"),
		source: ShaderSource::Wgsl(source.into()),
	});

	let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some("sky"),
		bind_group_layouts: &[layouts.first().copied()],
		immediate_size: 0,
	});

	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some("sky"),
		layout: Some(&layout),
		vertex: VertexState {
			module: &shader,
			entry_point: Some("vertex_sky"),
			compilation_options: PipelineCompilationOptions::default(),
			buffers: &[],
		},
		primitive: PrimitiveState {
			topology: PrimitiveTopology::TriangleList,
			strip_index_format: None,
			front_face: FrontFace::Ccw,
			// nothing: the triangle's winding depends on which corners the
			// index arithmetic lands on, and a sky culled by accident is a
			// black screen with no error anywhere.
			cull_mode: None,
			unclipped_depth: false,
			polygon_mode: PolygonMode::Fill,
			conservative: false,
		},
		depth_stencil: Some(DepthStencilState {
			format: DEPTH_FORMAT,
			depth_write_enabled: Some(false),
			depth_compare: Some(CompareFunction::LessEqual),
			stencil: StencilState::default(),
			bias: DepthBiasState::default(),
		}),
		multisample: MultisampleState {
			count: samples,
			..MultisampleState::default()
		},
		fragment: Some(FragmentState {
			module: &shader,
			entry_point: Some("fragment_sky"),
			compilation_options: PipelineCompilationOptions::default(),
			targets: &[Some(ColorTargetState {
				format,
				blend: Some(BlendState::REPLACE),
				write_mask: ColorWrites::ALL,
			})],
		}),
		multiview_mask: None,
		cache: None,
	})
}

/// What one of the table's pipelines is called in a graphics debugger.
///
/// Written out rather than formatted, because a label is borrowed for the
/// length of the call and building one would mean a `String` per pipeline for
/// the sake of a name nothing reads at run time.
const fn label_of(blend: Blend, skinned: bool) -> &'static str {
	match (blend, skinned) {
		| (Blend::Opaque, false) => "scene",
		| (Blend::Opaque, true) => "scene skinned",
		| (Blend::Mask, false) => "scene masked",
		| (Blend::Mask, true) => "scene masked skinned",
		| (Blend::Alpha, false) => "scene blended",
		| (Blend::Alpha, true) => "scene blended skinned",
	}
}

/// Builds one of them, without checking whether wgpu liked it.
///
/// @param device - the device to build against
/// @param format - the color format the fragment stage writes
/// @param layouts - the bind group layouts, in group order
/// @param source - the whole WGSL
/// @param blend - how the fragment stage reads the albedo's alpha
/// @param skinned - whether to bind a third vertex buffer and read bones
fn build_pipeline(
	device: &Device,
	format: TextureFormat,
	layouts: &[&BindGroupLayout],
	source: &str,
	blend: Blend,
	skinned: bool,
	samples: u32,
) -> RenderPipeline {
	let shader = device.create_shader_module(ShaderModuleDescriptor {
		label: Some("scene"),
		source: ShaderSource::Wgsl(source.into()),
	});

	let groups: Vec<Option<&BindGroupLayout>> = layouts.iter().copied().map(Some).collect();
	let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some("scene"),
		bind_group_layouts: &groups,
		immediate_size: 0,
	});

	let (vertex_stride, instance_stride) = strides();

	let vertices = VertexBufferLayout {
		array_stride: vertex_stride,
		step_mode: VertexStepMode::Vertex,
		attributes: &VERTEX_ATTRIBUTES,
	};

	let instances = VertexBufferLayout {
		array_stride: instance_stride,
		step_mode: VertexStepMode::Instance,
		attributes: &INSTANCE_ATTRIBUTES,
	};
	let skin = VertexBufferLayout {
		array_stride: skin_stride(),
		step_mode: VertexStepMode::Vertex,
		attributes: &SKIN_ATTRIBUTES,
	};
	// the third buffer only where it is read. Declaring it on both would mean
	// binding one for every crate in the world, and there is nothing to bind.
	let buffers: &[Option<VertexBufferLayout<'_>>] = if skinned {
		&[Some(vertices), Some(instances), Some(skin)]
	} else {
		&[Some(vertices), Some(instances)]
	};

	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some(label_of(blend, skinned)),
		layout: Some(&layout),
		vertex: VertexState {
			module: &shader,
			entry_point: Some(if skinned { "vertex_skinned" } else { "vertex_main" }),
			compilation_options: PipelineCompilationOptions::default(),
			buffers,
		},
		primitive: PrimitiveState {
			topology: PrimitiveTopology::TriangleList,
			strip_index_format: None,
			front_face: FrontFace::Ccw,
			// @note: on, and checked. `capture::tests` renders a cube and reads
			// the pixels back; if the winding convention were the other way
			// round the near faces would be discarded and the test would see
			// the clear color where it expects a lit face.
			cull_mode: Some(Face::Back),
			unclipped_depth: false,
			polygon_mode: PolygonMode::Fill,
			conservative: false,
		},
		depth_stencil: Some(DepthStencilState {
			format: DEPTH_FORMAT,
			// a blended surface reads the depth buffer and does not write it,
			// which is the whole of what "sorted back to front" is for: two
			// panes one behind the other both have to survive, and a written
			// depth would let whichever was drawn first hold the other out.
			depth_write_enabled: Some(blend != Blend::Alpha),
			depth_compare: Some(CompareFunction::Less),
			stencil: StencilState::default(),
			bias: DepthBiasState::default(),
		}),
		multisample: MultisampleState {
			count: samples,
			// **and this is the whole of what anti-aliases a cutout.** A
			// masked fragment is kept or thrown away whole, so multisampling
			// alone leaves a leaf's edge exactly as hard as it was; alpha to
			// coverage turns the alpha into how many of the pixel's samples
			// survive, which is the thing every renderer with a mask mode
			// reaches for the moment it has samples to spend. Off where there
			// is one sample, because with one sample it is the same hard
			// threshold with a slower path to it. @ref
			// `colby_core::abi::Blend::Mask`, whose own note says this is what
			// its threshold falls back from.
			alpha_to_coverage_enabled: samples > 1 && blend == Blend::Mask,
			..MultisampleState::default()
		},
		fragment: Some(FragmentState {
			module: &shader,
			entry_point: Some(match blend {
				| Blend::Opaque => "fragment_main",
				| Blend::Mask => "fragment_masked",
				| Blend::Alpha => "fragment_blended",
			}),
			compilation_options: PipelineCompilationOptions::default(),
			targets: &[Some(ColorTargetState {
				format,
				// straight rather than premultiplied alpha, which is what the
				// interface already blends with and what a person authoring a
				// picture in any editor produces. The target is an sRGB format,
				// so the hardware does the blend in linear light.
				blend: Some(if blend == Blend::Alpha {
					BlendState::ALPHA_BLENDING
				} else {
					BlendState::REPLACE
				}),
				write_mask: ColorWrites::ALL,
			})],
		}),
		multiview_mask: None,
		cache: None,
	})
}

/// What one [`MeshVertex`] hands the vertex stage.
///
/// @note: offsets written out rather than taken from a macro, so that a change
/// to `MeshVertex` shows up here as a mismatch to fix instead of as garbled
/// geometry. @ref [`strides`].
pub(crate) const VERTEX_ATTRIBUTES: [VertexAttribute; 4] = [
	VertexAttribute {
		format: VertexFormat::Float32x3,
		offset: 0,
		shader_location: 0,
	},
	VertexAttribute {
		format: VertexFormat::Float32x3,
		offset: 12,
		shader_location: 1,
	},
	VertexAttribute {
		format: VertexFormat::Float32x2,
		offset: 24,
		shader_location: 2,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 32,
		shader_location: 3,
	},
];

/// What one [`Placement`] hands it, once per instance.
///
/// The model matrix takes four of these because wgsl has no matrix vertex
/// attribute; the shader puts it back together.
///
/// @note: locations continue where [`VERTEX_ATTRIBUTES`] stopped. A shader
/// location is a property of the pipeline rather than of one buffer, so the two
/// tables share a numbering and growing the vertex pushes the instance along.
pub(crate) const INSTANCE_ATTRIBUTES: [VertexAttribute; 8] = [
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 0,
		shader_location: 4,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 16,
		shader_location: 5,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 32,
		shader_location: 6,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 48,
		shader_location: 7,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 64,
		shader_location: 8,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 80,
		shader_location: 9,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 96,
		shader_location: 10,
	},
	VertexAttribute {
		format: VertexFormat::Uint32x4,
		offset: 112,
		shader_location: 11,
	},
];

/// What one [`SkinVertex`] hands it, for a mesh bones move.
///
/// A third buffer rather than four more fields on the vertex: almost no mesh
/// is skinned, and a world of crates would pay twelve bytes a vertex for
/// something none of them reads. @ref
/// [`SkinVertex`](colby_core::abi::SkinVertex).
pub(crate) const SKIN_ATTRIBUTES: [VertexAttribute; 2] = [
	VertexAttribute {
		format: VertexFormat::Uint16x4,
		offset: 0,
		shader_location: 12,
	},
	// normalized on the way in, so the shader reads four fractions rather
	// than four numbers out of 255.
	VertexAttribute {
		format: VertexFormat::Unorm8x4,
		offset: 8,
		shader_location: 13,
	},
];

/// The stride of the skin buffer, asserted against the attributes above.
pub(crate) const fn skin_stride() -> BufferAddress {
	const {
		assert!(
			size_of::<SkinVertex>() == 12,
			"SkinVertex is no longer four shorts and four bytes"
		);
		assert!(align_of::<SkinVertex>() == 2, "SkinVertex gained padding");
	}

	12
}

/// The vertex and instance strides, asserted to match the attributes above.
pub(crate) const fn strides() -> (BufferAddress, BufferAddress) {
	const {
		assert!(
			size_of::<MeshVertex>() == 48,
			"MeshVertex is no longer two vec3s, a vec2 and a vec4"
		);
		assert!(align_of::<MeshVertex>() == 4, "MeshVertex gained padding");
		assert!(
			size_of::<Placement>() == 128,
			"Placement is no longer a mat4, three vec4s and four words"
		);
		assert!(align_of::<Placement>() == 4, "Placement gained padding");
		assert!(size_of::<Lamp>() == 48, "a Lamp is no longer three vec4s");
		// a uniform array's stride is its element rounded up to sixteen, so an
		// element that is already a multiple of it is laid out here exactly as
		// the shader reads it - which is the whole reason a lamp is three
		// vectors rather than a struct of named floats.
		assert!(size_of::<Lamp>().is_multiple_of(16), "and a uniform array's stride is not it");
		assert!(
			size_of::<Globals>() == 576 + size_of::<Lamp>() * MAX_LAMPS,
			"the two camera matrices, the light, the cascades, the fog, the sky, the counts and 			 the lamps"
		);
		assert!(size_of::<Globals>().is_multiple_of(16), "and a uniform struct has to be");
		// lines.wgsl declares only the first field of this struct and reads
		// only that, which a uniform binding allows: what it needs is for the
		// field to stay first, and this is where that is checked.
		assert!(
			offset_of!(Globals, view_projection) == 0,
			"lines.wgsl reads the camera out of the head of this struct"
		);
		assert!(CASCADES == 4, "the shader indexes four cascades by name");
	}

	(48, 128)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A short count as a distance, without an `as`.
	fn far(step: usize) -> f32 { f32::from(u8::try_from(step).expect("a short count")) }

	/// A world with a lamp of that range standing that far along x.
	fn lit_world(lamps: &[(f32, f32)]) -> World {
		let mut world = World::new();

		for &(along, range) in lamps {
			let id = world
				.entities
				.spawn_at(Transform::at(Vec3::X * along));

			world
				.entities
				.set_light(id, Light::point(Vec3::ONE, 1.0, range));
		}

		world
	}

	#[test]
	fn a_lamp_the_eye_is_standing_inside_comes_before_a_nearer_one_it_is_not() {
		// the small one is four units away and the big one is ten, but the big
		// one's sphere reaches the camera and the small one's does not. The
		// picture is inside the big one, so it is the one that matters.
		let world = lit_world(&[(4.0, 1.0), (10.0, 20.0)]);
		let mut scratch = Vec::new();
		let (lamps, count) = chosen(&world, Vec3::ZERO, MAX_LAMPS, &mut scratch);

		assert_eq!(count, 2, "both are carried");
		assert!(
			(lamps[0].position_range[3] - 20.0).abs() < 1.0e-6,
			"and the one the eye is inside is first: {:?}",
			lamps[0].position_range
		);
	}

	#[test]
	fn a_frame_carries_the_nearest_of_more_lamps_than_it_has_room_for() {
		let standing: Vec<(f32, f32)> = (0..MAX_LAMPS + 8)
			.map(|step| (2.0 + far(step), 1.0))
			.collect();
		let world = lit_world(&standing);
		let mut scratch = Vec::new();
		let (lamps, count) = chosen(&world, Vec3::ZERO, MAX_LAMPS, &mut scratch);

		assert_eq!(usize::try_from(count), Ok(MAX_LAMPS), "the array fills and no further");
		assert!(
			(lamps[0].position_range[0] - 2.0).abs() < 1.0e-6,
			"the nearest is first: {:?}",
			lamps[0].position_range
		);
		assert!(
			(lamps[MAX_LAMPS - 1].position_range[0] - (1.0 + far(MAX_LAMPS))).abs() < 1.0e-6,
			"and the last one carried is the last one that fits: {:?}",
			lamps[MAX_LAMPS - 1].position_range
		);
	}

	#[test]
	fn the_variable_takes_lamps_away_from_a_frame_nearest_last() {
		let world = lit_world(&[(2.0, 1.0), (4.0, 1.0), (6.0, 1.0)]);
		let mut scratch = Vec::new();

		for room in 0..=3 {
			let (lamps, count) = chosen(&world, Vec3::ZERO, room, &mut scratch);

			assert_eq!(usize::try_from(count), Ok(room), "asking for {room} carries {room}");

			for (slot, lamp) in lamps.iter().enumerate().take(room) {
				assert!(
					(lamp.position_range[0] - 2.0_f32.mul_add(far(slot), 2.0)).abs() < 1.0e-6,
					"and what it drops is the far end: {:?}",
					lamp.position_range
				);
			}
		}
	}

	#[test]
	fn an_entity_that_is_not_a_lamp_is_not_sent_to_the_shader() {
		let mut world = lit_world(&[(3.0, 5.0)]);
		// three that are not: no kind, no reach, and turned all the way down
		world.entities.spawn_at(Transform::at(Vec3::Y));
		let dark = world.entities.spawn_at(Transform::at(Vec3::Z));
		world
			.entities
			.set_light(dark, Light::point(Vec3::ONE, 1.0, 0.0));
		let off = world.entities.spawn_at(Transform::at(-Vec3::Z));
		world
			.entities
			.set_light(off, Light::point(Vec3::ONE, 0.0, 5.0));

		let mut scratch = Vec::new();
		let (_, count) = chosen(&world, Vec3::ZERO, MAX_LAMPS, &mut scratch);

		assert_eq!(count, 1, "one lamp among four entities");
	}

	#[test]
	fn the_lamp_ceiling_and_what_the_variable_starts_at_are_the_same_number() {
		assert_eq!(
			lamp_room(DEFAULT_LAMPS),
			MAX_LAMPS,
			"a fresh config asks for exactly what the array holds"
		);
	}

	#[test]
	fn asking_for_a_number_of_lamps_nobody_could_mean_lands_somewhere_definite() {
		assert_eq!(lamp_room(-4.0), 0, "below nought is none");
		assert_eq!(lamp_room(0.0), 0, "and so is nought");
		assert_eq!(lamp_room(3.9), 3, "a fraction is the whole number under it");
		assert_eq!(lamp_room(1.0e9), MAX_LAMPS, "past the ceiling is the ceiling");
		assert_eq!(lamp_room(f32::INFINITY), MAX_LAMPS, "and so is an infinity");
		assert_eq!(
			lamp_room(f32::NAN),
			MAX_LAMPS,
			"a nan is the ceiling rather than none: a variable nobody meant to set should not 			 put the lights out"
		);
	}

	#[test]
	fn a_point_light_packs_a_cone_that_answers_one_at_every_angle() {
		let packed = Lamp::of(Light::point(Vec3::ONE, 2.0, 5.0), Transform::IDENTITY);

		let near = |written: [f32; 4], wanted: [f32; 4]| {
			written
				.iter()
				.zip(wanted)
				.all(|(held, want)| (held - want).abs() < 1.0e-6)
		};

		assert!(
			near(packed.position_range, [0.0, 0.0, 0.0, 5.0]),
			"where it is and how far: {:?}",
			packed.position_range
		);
		assert!(
			near(packed.color, [2.0, 2.0, 2.0, 0.0]),
			"the color times the intensity: {:?}",
			packed.color
		);
		assert!(
			(packed.direction[3] - 1.0).abs() < 1.0e-6,
			"and the offset is one, which is the whole of how a point avoids a branch"
		);

		// the shader's line, with the scale of nought and the offset of one
		for along in [-1.0, 0.0, 0.5, 1.0_f32] {
			let cone = along.mul_add(packed.color[3], packed.direction[3]);

			assert!((cone - 1.0).abs() < 1.0e-6, "a point is lit at {along} along its axis");
		}
	}

	#[test]
	fn a_cone_is_full_inside_its_middle_and_nothing_past_its_edge() {
		let (inner, outer) = (0.3_f32, 0.6_f32);
		let packed =
			Lamp::of(Light::spot(Vec3::ONE, 1.0, 5.0, inner, outer), Transform::IDENTITY);
		let cone = |angle: f32| {
			angle
				.cos()
				.mul_add(packed.color[3], packed.direction[3])
				.clamp(0.0, 1.0)
		};

		assert!((cone(0.0) - 1.0).abs() < 1.0e-6, "straight down the axis is the whole light");
		assert!((cone(inner) - 1.0).abs() < 1.0e-6, "and so is the edge of the bright middle");
		assert!(cone(outer) < 1.0e-6, "the edge of the cone is nothing");
		assert!(cone(outer + 0.1) < 1.0e-6, "and so is past it");

		let between = cone((inner + outer) * 0.5);

		assert!(between > 0.0 && between < 1.0, "and in between it falls off: {between}");
		assert!(
			(packed.direction[2] + 1.0).abs() < 1.0e-6
				&& packed.direction[0].abs() < 1.0e-6
				&& packed.direction[1].abs() < 1.0e-6,
			"a cone points down the entity's own -z: {:?}",
			packed.direction
		);
	}

	#[test]
	fn a_distance_is_measured_in_thousandths_and_never_wraps() {
		assert_eq!(grain(1.0), 1000, "one unit is a thousand of them");
		assert_eq!(grain(-1.0), -1000, "and it keeps its sign");
		assert_eq!(
			grain(4.0),
			grain(4.0004),
			"two surfaces less than a thousandth apart tie, which is what keeps a row of 			 \
			 identical props one batch"
		);
		assert_ne!(grain(4.0), grain(4.002), "and two further apart than that do not");

		// the half that matters: a wrap would put the furthest thing in the
		// world in front of everything, which is the one answer worse than
		// being unsorted. Past the range it saturates, so two absurd distances
		// stop being told apart - which is the trade, and it happens tens of
		// thousands of units out.
		assert!(grain(f32::MAX) > 0, "something absurdly far is still in front, not behind");
		assert!(grain(-f32::MAX) < 0, "and something absurdly behind is still behind");
		assert_eq!(
			grain(f32::MAX),
			grain(f32::MAX / 2.0),
			"two of them past the range are one number rather than a wrapped one"
		);
		assert_eq!(grain(f32::NAN), 0, "and a number that is not one sorts as no distance");
	}

	#[test]
	fn the_buffers_are_sized_for_what_goes_in_them() {
		assert_eq!(
			size_bytes::<MeshVertex>(24).expect("the size fits"),
			24 * 48,
			"a cube's worth of position, normal, texture coordinate and tangent"
		);
		assert_eq!(
			size_bytes::<Placement>(MAX_ENTITIES).expect("the size fits"),
			128 * BufferAddress::try_from(MAX_ENTITIES).expect("the count fits"),
			"one placement per entity the world can hold"
		);
		assert_eq!(
			skin_stride(),
			BufferAddress::try_from(size_of::<SkinVertex>()).expect("a vertex is small"),
			"and the third buffer is read at the width the block was written in"
		);
	}

	#[test]
	fn clear_color_is_held_inside_the_range_wgpu_accepts() {
		let mut world = World::new();
		world.clear = Vec3::new(-1.0, 0.5, 4.0);

		let color = clear_color(&world);

		assert!(color.r.abs() < f64::EPSILON, "below zero clamps up");
		assert!((color.g - 0.5).abs() < f64::EPSILON, "in range passes through");
		assert!((color.b - 1.0).abs() < f64::EPSILON, "above one clamps down");
	}

	#[test]
	fn every_texel_layout_maps_to_a_format_the_gpu_knows() {
		assert_eq!(
			texel_format(Texel::Rgba8Srgb),
			TextureFormat::Rgba8UnormSrgb,
			"sRGB in the file, sRGB on the GPU, linear by the time the shader sees it"
		);
	}
}
