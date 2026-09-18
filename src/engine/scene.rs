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

use core::{mem::offset_of, ops::Range};
use std::sync::Arc;

use colby_core::{
	Result,
	abi::{
		Camera, DRAWING, EntityId, Light, LightKind, MAX_ENTITIES, Material, MeshData,
		MeshVertex, Meshes, PaintVertex, Renderable, SkinVertex, Texel, TextureData, TextureId,
		Textures, Transform, World,
		material::{Blend, MaterialEntry, Wrap},
		registry::Entry,
	},
	bytemuck::{self, Pod, Zeroable},
	err, error,
	glam::{Mat4, Vec3},
	info, warn,
};
use wgpu::{
	AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
	BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
	Buffer, BufferAddress, BufferBindingType, BufferDescriptor, BufferUsages, Color,
	ColorTargetState, ColorWrites, CommandEncoder, CommandEncoderDescriptor, CompareFunction,
	DepthBiasState, DepthStencilState, Device, DownlevelFlags, ErrorFilter, Extent3d, Face,
	FilterMode, FragmentState, FrontFace, IndexFormat, LoadOp, MipmapFilterMode,
	MultisampleState, Operations, Origin3d, PipelineCompilationOptions, PipelineLayoutDescriptor,
	PolygonMode, PrimitiveState, PrimitiveTopology, Queue, RenderPass, RenderPassColorAttachment,
	RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPassTimestampWrites,
	RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor,
	ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState, StoreOp,
	TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
	TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
	TextureViewDescriptor, TextureViewDimension, VertexAttribute, VertexBufferLayout,
	VertexFormat, VertexState, VertexStepMode,
};

use crate::{
	brdf::{self, Split},
	cover::{self, COMMAND_SIZE, Cover, Reach},
	cull::{self, Bounds, Drawn, Frustum, Placed},
	decal::{self, Atlas, Chosen, DECALS, Key, MAX_DECALS, Paint},
	depth::{self, Depth},
	detail,
	env::{self, Environment},
	focus::{self, Focus},
	gpu::Gpu,
	haze::{self, Haze},
	lines::Lines,
	occlusion::{self, Occlusion},
	post,
	prepass::{self, Prepass},
	reflection::{self, Reflection},
	shader::Shader,
	shadow::{self, CASCADES, Cascades, LOCAL_TILES, Maps, Slots, Tile},
	shaft::{self, Asking, Shaft},
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

/// One lamp a frame carries, with what deciding its shadow needs.
///
/// The packed [`Lamp`] is what the shader reads and holds no kind, no cone
/// angle and no rotation - a cone's falloff is two numbers by then. Working
/// out where a shadow map looks from needs all three back, so they ride along
/// here rather than being unpicked from the packing.
struct Shining {
	/// How far the near edge of its reach is from the eye, which is the order
	/// [`chosen`] sorts on.
	near: f32,

	/// What the shader reads.
	lamp: Lamp,

	/// What it shines, as the world holds it.
	light: Light,

	/// Where it stands and which way it is turned.
	at: Transform,
}

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

	/// `[the first atlas tile, how many, one texel per unit of distance, 0]`.
	///
	/// A count of nought is a lamp that throws no shadow: it was told not to,
	/// or the atlas was full when its turn came. @ref
	/// [`shadow::Slots`](crate::shadow::Slots).
	shadow: [f32; 4],
}

impl Lamp {
	/// A lamp that is not there, for the tail of the array.
	const DARK: Self = Self {
		position_range: [0.0; 4],
		color: [0.0; 4],
		direction: [0.0, 0.0, -1.0, 1.0],
		shadow: [0.0; 4],
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
			shadow: [0.0; 4],
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

	/// Where each cascade's map sits in the atlas, nearest slice first.
	///
	/// A whole layer each, which is why the atlas cost the cascades nothing.
	/// @ref [`Tile::layer`].
	cascade_tiles: [Tile; CASCADES],

	/// Where each local map sits in the atlas, in the order they were handed
	/// out: a cone takes one and a point takes six in a row.
	lamp_tiles: [Tile; LOCAL_TILES],

	/// World space into each local map's clip space, one per tile above.
	lamp_views: [[[f32; 4]; 4]; LOCAL_TILES],

	/// `[r, g, b, how quickly a surface fades with distance]`.
	fog: [f32; 4],

	/// `[r, g, b, whether a sky is drawn]` straight up.
	sky_zenith: [f32; 4],

	/// `[r, g, b, unused]` at eye level.
	sky_horizon: [f32; 4],

	/// `[r, g, b, unused]` straight down.
	sky_ground: [f32; 4],

	/// `[how many lamps are real, how many decals are, unused, unused]`.
	counts: [u32; 4],

	/// The local lights, nearest first; the rest is [`Lamp::DARK`].
	lamps: [Lamp; MAX_LAMPS],

	/// The decals, in the order they are painted; the rest is
	/// [`Paint::NOTHING`]. @ref [`decal`].
	decals: [Paint; MAX_DECALS],
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

	/// `[where this instance's joint matrices start, how many, flags, 0]`.
	///
	/// The flags are the entity's own, and the one bit there is says decals
	/// leave it alone - @ref
	/// [`Entities::takes_decals`](colby_core::abi::Entities::takes_decals).
	/// Here rather than in an attribute of its own because this word was spare.
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
	/// The mesh's own indices, and every coarser level's after them.
	indices: Buffer,
	/// The bones and weights, for a mesh that has them.
	skin: Option<Buffer>,
	/// What each vertex was painted with, for a mesh somebody painted.
	///
	/// A mesh nobody painted draws with the scene's [`Plain`] buffer in its
	/// place, because every pipeline reads the paint and the buffer it reads
	/// has to be there. @ref [`PaintVertex`].
	paint: Option<Buffer>,
	/// How many vertices it has, which is how long a plain buffer standing in
	/// for its paint has to be.
	vertex_count: usize,
	/// How many of [`indices`](Self::indices) are the mesh itself.
	index_count: u32,
	revision: u32,

	/// Every coarser level, finest first: where its indices start in the one
	/// buffer, how many there are, and how far it stands from the mesh.
	///
	/// Empty for a mesh drawn only whole. The level a batch draws is found here
	/// by its number, nought being the mesh itself. @ref [`detail`].
	levels: Vec<(Range<u32>, f32)>,

	/// How far each of [`levels`](Self::levels) stands from the mesh, apart,
	/// for the walk that picks one. @ref [`detail::Eye::level`].
	errors: Vec<f32>,

	/// The box around the mesh, in the shape it was modeled in.
	///
	/// Kept here rather than asked for per frame because
	/// [`MeshData::bounds`](colby_core::abi::MeshData::bounds) walks every
	/// vertex, and what wants it - leaving out what a pass cannot see, and
	/// sorting the blended half of a frame - asks once per entity per frame. It
	/// is worked out once per upload instead, which is once per asset reload.
	///
	/// Its middle is what the blended half sorts on, rather than the origin,
	/// which is what the two engines that publish their sort key both use: a
	/// floor slab modeled from a corner would otherwise sort as though it were
	/// at that corner.
	bounds: Bounds,

	/// Each bone's box, for a mesh bones move; empty for one they do not.
	///
	/// Worked out at upload beside [`bounds`](Self::bounds), for the same
	/// reason and from the same walk over the vertices. @ref [`cull::bones`].
	bones: Vec<Option<Bounds>>,
}

impl GpuMesh {
	/// The run of indices one level draws: the mesh itself for nought, and the
	/// mesh itself for a level it does not have.
	///
	/// @param level - nought for the mesh, one for its finest coarser level
	fn run(&self, level: u8) -> Range<u32> {
		usize::from(level)
			.checked_sub(1)
			.and_then(|at| self.levels.get(at))
			.map_or(0..self.index_count, |(run, _)| run.clone())
	}
}

/// What every mesh nobody painted draws its paint from: one buffer of plain
/// entries, white and at the corner of the picture, as long as the longest such
/// mesh.
///
/// **One buffer rather than one a mesh, and it is read at every vertex.** A
/// pipeline fixes the stride of each buffer it reads when it is built, so a
/// mesh without paint cannot bind a buffer of one entry and have every vertex
/// read it - that is a stride of nought, which would mean a second table of
/// pipelines for the meshes that have none. It binds this instead, which holds
/// the same entry at every index; the file and the mesh's own upload carry no
/// paint at all, and a world of crates pays for one buffer rather than one
/// each.
struct Plain {
	buffer: Buffer,
	/// How many entries it holds, which is how many vertices a mesh reading it
	/// may have.
	vertices: usize,
}

impl Plain {
	/// How many entries a scene starts with, before any mesh has asked for
	/// more.
	const FIRST: usize = 1024;

	/// A buffer of plain entries for meshes of up to so many vertices.
	///
	/// @param device - the device to build against
	/// @param queue - the queue to fill it through
	/// @param vertices - how many entries it has to hold at least
	fn new(device: &Device, queue: &Queue, vertices: usize) -> Self {
		let vertices = vertices.max(Self::FIRST).next_power_of_two();

		Self {
			buffer: create_buffer(
				device,
				queue,
				"plain paint",
				bytemuck::cast_slice(&vec![PaintVertex::PLAIN; vertices]),
				BufferUsages::VERTEX,
			),
			vertices,
		}
	}
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

/// How many lists one entity can be in: the picture's, one per cascade, and one
/// per tile of the atlas a lamp may take.
///
/// Each list is a run of the instance buffer of its own, because a batch is a
/// first and a count into it; so the buffer holds this many placements for
/// every entity the world can hold. At a hundred and twenty-eight bytes a
/// placement that is two and three quarter megabytes, beside twenty of atlas -
/// and the great majority of the runs are empty in any real world, because a
/// lamp's map only sees what is inside its reach.
const LISTS: usize = 1 + MAPS;

/// How many shadow maps one frame can draw: the cascades and the atlas's tiles.
///
/// A list index past the picture's is one of these, in the same order the
/// atlas's slots are: the cascades first, the lamps' tiles after them.
const MAPS: usize = CASCADES + LOCAL_TILES;

/// What one frame can see, worked out once in [`Scene::upload`] and asked of
/// everything that might be drawn.
struct Sight {
	/// Where the camera is.
	eye: Vec3,

	/// The way it looks, of unit length: what the blended half sorts along.
	forward: Vec3,

	/// What the picture is drawn through.
	view: Frustum,

	/// Each cascade's box, nearest first, or nothing in a frame with the
	/// shadows off.
	cascades: Option<[Frustum; CASCADES]>,

	/// What each local map this frame draws can see.
	///
	/// A face of a point light is a right-angled perspective and a cone is one
	/// as wide as it opens, and both are volumes six planes hold - so the same
	/// test that leaves geometry out of a cascade leaves it out of these,
	/// unchanged, and no second mechanism had to be written. @ref
	/// [`Frustum::of`]. Only the first [`maps`](Self::maps) are real.
	faces: [Frustum; LOCAL_TILES],

	/// How many of [`faces`](Self::faces) the frame handed out.
	maps: usize,

	/// Whether to ask at all. @ref [`cull::ENABLED`].
	culling: bool,

	/// What every thing's level is asked against. @ref [`detail`].
	detail: detail::Eye,

	/// How many pixels across a solid thing has to stand to be drawn into the
	/// pass before the scene ahead of the test, or nothing in a frame that
	/// draws everything ahead of it: one that runs no test, or one whose
	/// variable says so. @ref [`cover::SIZE`].
	small_under: Option<f32>,
}

impl Sight {
	/// Whether one thing stands too small to be drawn into the pass before the
	/// scene ahead of the test this frame: never in a frame that runs no test.
	/// @ref [`cover::SIZE`].
	///
	/// @param placed - the thing's box, in the world
	fn small(&self, placed: &Placed) -> bool {
		self.small_under
			.is_some_and(|size| self.detail.under(placed, size))
	}
}

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

	/// Which of its mesh's levels it is drawn at, nought being the mesh itself.
	///
	/// In the key between the mesh and the material, so the things drawn at one
	/// level of one mesh are one batch: a level is a run of the mesh's indices,
	/// and a batch draws one run. @ref [`detail`].
	level: u8,

	material: u32,

	/// Whether it stands too small to be drawn into the pass before the scene
	/// ahead of the test, and so is drawn into it after the test and only if
	/// the test kept it. @ref [`cover::SIZE`].
	///
	/// In the key between the material and the slot, so the things of one mesh
	/// at one level in one material are two batches at most, its large things
	/// and then its small ones, and the scene's own order moves only between
	/// those two: a large thing is drawn ahead of a small one it followed by
	/// slot, which only two surfaces at exactly one depth could tell apart.
	/// False for everything blended, for everything in a frame that runs no
	/// test or holds nothing solid that is large, and for what a map draws,
	/// which draws both alike.
	small: bool,

	/// Its slot decides nothing but the order of two entities a frame could not
	/// otherwise tell apart, which is what keeps a frame the same picture
	/// twice - an unstable sort may put equal keys either way round.
	entity: EntityId,

	/// Worked out once here and used for both the pass and the batch, so the
	/// list an entity is in and the pipeline its batch is drawn with can never
	/// be two different answers.
	blend: Blend,

	/// Where its placement is in [`Scene::staged`], which is worked out once
	/// however many of the frame's lists the entity is in.
	at: u32,

	/// Which maps can see it, one bit each: the cascades lowest, then the
	/// lamps' tiles in the order they were handed out.
	///
	/// Nought for everything blended, which casts nothing, and for everything
	/// in a frame that is not culling, whose cascades draw the picture's solid
	/// half instead of lists of their own.
	casts: u32,
}

impl Sorted {
	/// What the list is ordered on, in order of priority.
	///
	/// The mode is not in it and does not have to be: `pass` is a function of
	/// it and comes first, so the two halves are already apart.
	const fn key(&self) -> (u8, i32, u32, u8, u32, bool, usize) {
		(
			self.pass,
			self.depth,
			self.mesh,
			self.level,
			self.material,
			self.small,
			self.entity.slot(),
		)
	}
}

/// What the test for what is behind something nearer works from, laid out while
/// the frame is grouped. @ref [`cover`].
struct Covering {
	/// The test itself.
	cover: Cover,

	/// Whether this frame asks for it: the variable, the frustum test, and a
	/// pass before the scene to read.
	asked: bool,

	/// Every staged entity's box, in [`Scene::staged`]'s order, while asked.
	reached: Vec<Placed>,

	/// One record a thing of the picture's lists, in placement order.
	reaches: Vec<Reach>,

	/// Five words a batch of those lists, the solid batches first.
	commands: Vec<u32>,

	/// The matrix this frame's picture is drawn through.
	projection: Mat4,
}

/// How a list of batches is drawn: an instance range a batch, or through what
/// the test kept. @ref [`Scene::draw_through`].
#[derive(Clone, Copy)]
enum Through<'a> {
	/// Each batch's own run of the placements.
	Instances,

	/// Each batch's run of what the test kept, as many as its command says.
	Kept {
		/// The kept placements.
		placements: &'a Buffer,

		/// One command a batch.
		commands: &'a Buffer,

		/// How many bytes a placement is.
		stride: BufferAddress,

		/// Which command the list's first batch has: the blended list's come
		/// after the solid list's.
		offset: usize,
	},
}

/// Counts one batch drawn through its own placements or through what the test
/// kept, for a test. @ref [`Scene::draws`].
#[cfg(test)]
const fn tally(drew: &mut (usize, usize), through: Through<'_>) {
	match through {
		| Through::Instances => drew.0 += 1,
		| Through::Kept { .. } => drew.1 += 1,
	}
}

/// A run of instances that share a mesh, a level of it and a material.
struct Batch {
	mesh: usize,
	/// Which run of the mesh's indices it draws, nought being the mesh itself.
	level: u8,
	material: usize,
	first: u32,
	count: u32,
	/// Whether its things are drawn into the pass before the scene after the
	/// test rather than ahead of it: true of every thing in the run, because it
	/// is in the key the runs are cut on. @ref [`Sorted::small`].
	small: bool,
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
	/// Every picture the world's decals throw. @ref [`decal`].
	atlas: Atlas,
	/// What reads the atlas. @ref [`decal_sampler`].
	decal_sampler: Sampler,
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
	/// The depth buffer the scene's pass tests against, and the one a pass
	/// after it reads. @ref [`depth`].
	depth: Depth,
	/// The depth drawn instead of the picture this frame, if somebody asked:
	/// how far away white is, and the projection's two numbers a stored depth
	/// is turned back into a distance with. Read in [`Scene::upload`], for the
	/// reason [`sky`](Self::sky) is. @ref [`depth::VIEW`].
	seeing: Option<(f32, [f32; 2])>,
	/// The three passes that smear the sky around the sun. @ref [`shaft`].
	shaft: Shaft,
	/// What those passes are asked for this frame, or nothing. Read in
	/// [`Scene::upload`], for the reason [`seeing`](Self::seeing) is.
	shafting: Option<Asking>,
	/// The three passes that blur what the lens is not focused on. @ref
	/// [`focus`].
	focus: Focus,
	/// What those passes are asked for this frame, or nothing. Read in
	/// [`Scene::upload`], for the reason [`seeing`](Self::seeing) is.
	focusing: Option<focus::Asking>,
	/// The pass before the scene, which writes every pixel's depth, normal and
	/// roughness for whatever reads them before anything is lit. @ref
	/// [`prepass`].
	prepass: Prepass,
	/// What that pass wrote, drawn instead of the picture this frame, if
	/// somebody asked. Read in [`Scene::upload`], for the reason
	/// [`seeing`](Self::seeing) is. @ref [`prepass::VIEW`].
	showing: Option<prepass::Showing>,
	/// How much of the sky each pixel sees, worked out from what that pass
	/// wrote. @ref [`occlusion`].
	occlusion: Occlusion,
	/// What that is asked for this frame, or nothing. Read in
	/// [`Scene::upload`], for the reason [`seeing`](Self::seeing) is.
	occluding: Option<occlusion::Asking>,
	/// Which making of that share group nought was made over. @ref
	/// [`Occlusion::epoch`].
	occluded: u64,
	/// What each pixel's reflection finds on the picture, worked out from what
	/// the pass before the scene wrote. @ref [`reflection`].
	reflection: Reflection,
	/// What that is asked for this frame, or nothing. Read in
	/// [`Scene::upload`], for the reason [`seeing`](Self::seeing) is.
	reflecting: Option<reflection::Asking>,
	/// Which making of what that found group nought was made over. @ref
	/// [`Reflection::epoch`].
	reflected: u64,
	/// The light a haze sends towards the eye, worked out after the scene from
	/// the depth it wrote. @ref [`haze`].
	haze: Haze,
	/// What that is asked for this frame, or nothing. Read in
	/// [`Scene::upload`], for the reason [`seeing`](Self::seeing) is.
	hazing: Option<haze::Asking>,
	/// How many times group nought has been made again, which is how a test
	/// shows a kept group is kept. @ref [`rebinds`](Self::rebinds).
	#[cfg(test)]
	rebound: u32,
	/// Whether a test asked for the pass before the scene with nothing to read
	/// it, which is how a test shows that the pass moves no pixel of the
	/// picture. @ref [`prepare_anyway`](Self::prepare_anyway).
	#[cfg(test)]
	anyway: bool,
	/// What every list drawn this frame was drawn through, which is how a test
	/// shows what each half of the pass before the scene drew. @ref
	/// [`draws`](Self::draws).
	#[cfg(test)]
	drew: core::cell::RefCell<Vec<(usize, usize)>>,
	/// The depth array the light writes and the scene samples.
	shadows: Maps,
	/// The cube a surface's reflections are read out of. @ref [`env`].
	environment: Environment,
	/// How much of whatever it reflects a surface sends back. @ref [`brdf`].
	split: Arc<Split>,
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
	/// What a mesh nobody painted reads its paint from, made at the first
	/// upload of one and grown whenever a longer one arrives. @ref [`Plain`].
	plain: Option<Plain>,
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
	/// Every entity the picture draws, sorted. Sorting thirty-byte keys and
	/// looking the placements up again beats sorting the hundred-and-twenty-
	/// eight-byte placements.
	order: Vec<Sorted>,
	/// Every entity some pass draws this frame, flattened once.
	///
	/// An entity the picture and two cascades all draw is written into the
	/// instance buffer three times, once into each list's run, and worked out
	/// here once.
	staged: Vec<Placement>,
	/// The solid entities some cascade can see, sorted the way the picture's
	/// solid half is, each carrying which cascades.
	casters: Vec<Sorted>,
	/// Each shadow map's own batches, in a frame that is culling: the cascades
	/// first, the lamps' tiles after them. @ref [`culling`](Self::culling).
	casting: [Vec<Batch>; MAPS],

	/// Where each local map this frame draws sits in the atlas.
	lamp_tiles: [Tile; LOCAL_TILES],

	/// World space into each of their clip spaces.
	lamp_views: [[[f32; 4]; 4]; LOCAL_TILES],

	/// How many tiles the frame handed out, which is how many of the two
	/// arrays above are real and how many passes the local layer records.
	lamp_maps: usize,
	/// Whether this frame's cascades each draw a list of their own.
	///
	/// `false` is every frame before culling existed, kept exactly: all four
	/// draw the picture's solid half, which with nothing left out of it is
	/// every solid entity there is. @ref [`cull::ENABLED`].
	culling: bool,
	/// How much of the world the last frame drew.
	drawn: Drawn,
	/// Every lit entity with how far its reach is from the eye, kept so it
	/// allocates once. @ref [`Scene::lamps`].
	lit: Vec<Shining>,
	/// The decals this frame carries, kept so it allocates once. @ref
	/// [`Scene::decals`].
	painted: Vec<Chosen>,
	/// Every picture the world's decals throw, as the atlas knows them.
	pictures: Vec<Key>,
	/// What is left out of the scene's lists for being behind something nearer.
	/// @ref [`cover`].
	covering: Covering,
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

		let globals = globals_uniform(&device)?;
		let globals_layout = frame_layout(&device);
		let material_layout = material_layout(&device);
		// empty until a decal exists, and whole all the same: the group it is
		// bound in has to be, for every pipeline that reads group nought.
		let atlas = Atlas::new(&device);
		let decal_sampler = decal_sampler(&device);
		// before group nought, because the cube is one of its entries: there is
		// no fifth group to put it in. @ref [`env`].
		let environment = Environment::new(&device, &queue)?;
		// and beside it, taken from the device rather than built here: the
		// table is the same one for every scene this device draws. @ref
		// [`Gpu::split`].
		let split = Arc::clone(gpu.split());
		// and before group nought too, for the cube's reason: what the share of
		// the sky and what the reflections found are read out of are its ninth
		// and tenth entries, and a scene that has asked for nothing yet binds the
		// texel that says all of the one and the texel that says none of the other
		let (occlusion, reflection, haze) = readers(&device, &queue, (width, height))?;
		let bindings = frame_bindings(&device, &globals_layout, &globals, &atlas, &Held {
			decals: &decal_sampler,
			environment: &environment,
			split: &split,
			occlusion: occlusion.bound(),
			reflection: reflection.bound(),
		});

		let samplers = wraps(&device);

		// before the maps and before the pipelines: the depth pass reads the
		// joints as its second group and the scene reads them as its fourth,
		// so the layout has to exist before either is built.
		let joints = Joints::new(&device)?;
		let shadows = Maps::new(&device, joints.layout(), &material_layout)?;
		let shader = Shader::new("shader.wgsl", include_str!("shader.wgsl"));
		let (pipelines, depth, lines, sparks) = drawing(
			&device,
			[&globals_layout, &material_layout, shadows.sample_layout(), joints.layout()],
			shader.source(),
			(width, height),
		)?;
		let post = post::Chain::new(&device, format, width, height)?;
		let shaft = Shaft::new(&device, width, height)?;
		let focus = Focus::new(&device, width, height)?;

		let (instances, covering) = (placements(&device)?, covering(gpu, (width, height))?);

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
			atlas,
			decal_sampler,
			shader,
			size: (width, height),
			depth,
			seeing: None,
			shaft,
			shafting: None,
			focus,
			focusing: None,
			prepass: Prepass::new(width, height),
			showing: None,
			occluded: occlusion.epoch(),
			occlusion,
			occluding: None,
			reflected: reflection.epoch(),
			reflection,
			reflecting: None,
			haze,
			hazing: None,
			#[cfg(test)]
			rebound: 0,
			#[cfg(test)]
			anyway: false,
			#[cfg(test)]
			drew: core::cell::RefCell::new(Vec::new()),
			shadows,
			environment,
			split,
			cascades: Cascades::NONE,
			shadowing: false,
			lines,
			sparks,
			joints,
			// nothing is uploaded until a frame says what the world holds: the
			// registries belong to the host, and a scene built before the host
			// has loaded its assets would only have to be rebuilt afterwards.
			meshes: Vec::new(),
			plain: None,
			textures: Vec::new(),
			materials: Vec::new(),
			instances,
			placements: Vec::with_capacity(MAX_ENTITIES * LISTS),
			batches: Vec::new(),
			blended: Vec::new(),
			order: Vec::with_capacity(MAX_ENTITIES),
			staged: Vec::with_capacity(MAX_ENTITIES),
			casters: Vec::with_capacity(MAX_ENTITIES),
			casting: core::array::from_fn(|_| Vec::new()),
			lamp_tiles: [Tile::local(0); LOCAL_TILES],
			lamp_views: [[[0.0; 4]; 4]; LOCAL_TILES],
			lamp_maps: 0,
			culling: false,
			drawn: Drawn::default(),
			lit: Vec::with_capacity(MAX_LAMPS),
			painted: Vec::with_capacity(MAX_DECALS),
			pictures: Vec::new(),
			covering,
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
		self.depth.resize(&self.device, width, height);
		self.post.resize(&self.device, width, height);
		self.shaft.resize(width, height);
		self.focus.resize(width, height);
		self.prepass.resize(width, height);
		self.occlusion.resize(width, height);
		self.reflection.resize(width, height);
		self.haze.resize(width, height);
		self.covering.cover.resize(width, height);
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

				self.depth.set_samples(&self.device, asked);
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
		let table = Pipelines::build(&self.device, post::HDR_FORMAT, &groups, source, samples)?;
		// and the pass before the scene with it, if that was ever built, for the
		// same reason: a normal written by one shader and lit by another is two
		// answers to one question.
		let prepass = self
			.prepass
			.rebuilt(&self.device, &groups, source)?;
		// and the reflections' first pass, which lights what it finds with this
		// same source: what a mirror shows and what the eye sees are lit by one
		// arithmetic or by none
		let reflection = self.reflection.rebuilt(
			&self.device,
			[&self.globals_layout, self.shadows.sample_layout()],
			source,
		)?;
		// and the air's two passes, which light it with the same lamps: the light
		// a lamp throws on a wall and the light it throws into the air in front of
		// the wall are one lamp's or none
		let haze = self.haze.rebuilt(
			&self.device,
			[&self.globals_layout, self.shadows.sample_layout()],
			source,
		)?;

		self.pipelines = table;
		self.prepass.replace(prepass);
		self.reflection.replace(reflection);
		self.haze.replace(haze);
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

		#[cfg(test)]
		self.drew.get_mut().clear();

		self.timings.open(Work::Upload);
		self.upload(world, view);
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

			// and one more over the atlas's last layer, for every lamp that
			// asked for a tile. A span of its own rather than four more of the
			// one above, because what it costs is a different question: the
			// cascades are the sun and are always four, and this is however
			// many lamps the frame is standing near.
			if self.lamp_maps > 0 {
				self.cast_lamps(&mut encoder, self.timings.writes(Pass::Lamps, Ends::Both));
			}
		}

		// after the shadows and before the scene's own pass, because what reads
		// a surface before it is lit has to find it written already: the large
		// things of it here, and the small ones below. @ref [`prepass`].
		let prepared = self.prepare(&mut encoder, view);

		// straight after it, because what it reads is the depth that pass has just
		// written, and before everything that reads that pass too: none of them
		// draws the lists this leaves things out of
		self.cover(&mut encoder, view, prepared);

		// and straight after that, the small things through what it kept: the
		// last of what the pass before the scene writes, before anything reads it
		self.prepare_small(&mut encoder, view, prepared);

		// and straight after it, for the same reason and for the one after it:
		// the scene is about to light what this works out
		self.occlusion.render(
			&mut encoder,
			&self.queue,
			self.occluding,
			&self.prepass,
			view,
			&self.timings,
		);

		// and group nought made again over what that left, in a frame that made
		// its buffers or let them go. After the pass before the scene has bound
		// the old group rather than before, which that pass can afford: none of
		// its entry points reads the share.
		if self.occlusion.epoch() != self.occluded {
			self.rebind();
		}

		// and after that, because what a reflection meets is lit with the share
		// of the sky as it now stands, and before the scene, which reads what
		// the reflections found
		self.reflect(&mut encoder, view);

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
				view: self.depth.attachment(),
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
				self.finish(encoder, world, (seconds, view), target);

				return;
			};

			cut(&mut pass, view);
		}

		pass.set_bind_group(0, &self.bindings, &[]);
		pass.set_bind_group(2, self.shadows.bindings(), &[]);
		pass.set_bind_group(3, self.joints.bindings(), &[]);
		pass.set_vertex_buffer(1, self.instances.slice(..));

		self.draw(&mut pass, &self.batches, 0);

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
		self.draw(&mut pass, &self.blended, self.batches.len());

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
		self.finish(encoder, world, (seconds, view), target);
	}

	/// What is wholly behind what the pass before the scene drew, found and
	/// left out of the lists the scene's pass is about to draw. @ref
	/// [`cover`].
	///
	/// @param encoder - the frame's, with the pass before the scene in it
	/// @param view - the part of the target the picture is drawn into, or all
	/// @param prepared - whether that pass drew this frame: a depth left from a
	/// frame before is not this frame's, and nothing is tested against it
	///
	/// @note: no frame today asks for the test with that pass unrecorded and a
	/// depth still held - the pass records whenever its targets exist and the
	/// rectangle is inside the picture, and the test asks both - so a mutation
	/// pass that dropped `prepared` passed everything. It stays because a depth
	/// being held is not proof of whose frame it is.
	fn cover(&mut self, encoder: &mut CommandEncoder, view: Option<Viewport>, prepared: bool) {
		let instances = u32::try_from(self.covering.reaches.len()).unwrap_or(0);

		self.covering.cover.render(
			encoder,
			&self.queue,
			cover::Frame {
				asked: self.covering.asked && prepared,
				prepass: &self.prepass,
				placements: &self.instances,
				view_projection: self.covering.projection,
				instances,
				timings: &self.timings,
			},
			view,
		);
	}

	/// What each pixel's reflection finds on the picture, recorded between the
	/// share of the sky and the scene's own pass, and group nought made again
	/// over it in a frame that made its buffers or let them go.
	///
	/// **The group after the passes rather than before them**, where the
	/// share's is made before the passes that follow it: the buffers are let
	/// go inside [`Reflection::render`], and the first of its passes, which
	/// binds the old group, lights what it finds with nothing found and never
	/// reads that entry.
	///
	/// @param encoder - the frame's, with the share of the sky already in it
	/// @param view - the part of the target the picture is drawn into, or all
	fn reflect(&mut self, encoder: &mut CommandEncoder, view: Option<Viewport>) {
		if self.reflecting.is_some() {
			self.reflection.ensure(
				&self.device,
				[&self.globals_layout, self.shadows.sample_layout()],
				&self.built,
			);
		}

		self.reflection.render(
			encoder,
			&self.queue,
			reflection::Frame {
				asked: self.reflecting,
				prepass: &self.prepass,
				scene: &self.bindings,
				shadows: self.shadows.bindings(),
				timings: &self.timings,
			},
			view,
		);

		if self.reflection.epoch() != self.reflected {
			self.rebind();
		}
	}

	/// Everything after the scene's own pass: the depth made readable if
	/// anything reads it, the post-processing, and the timestamps copied out.
	///
	/// A method of its own because a frame whose rectangle held nothing needs
	/// all of it as well: what reaches the window is written by the last of
	/// these passes and by nothing else.
	///
	/// @param encoder - the frame's, with the scene's pass in it
	/// @param world - for what the post-processing is asked to do
	/// @param (seconds, view) - how long the frame was, for the eye, and the
	/// part of the target the picture is drawn into, or the whole of it
	/// @param target - what the last pass writes
	fn finish(
		&mut self,
		mut encoder: CommandEncoder,
		world: &World,
		(seconds, view): (f32, Option<Viewport>),
		target: &TextureView,
	) {
		// between the scene that wrote the depth and everything after it that
		// reads it, and only in a frame something does. **Anything**: the view
		// that draws the depth was the first reader, the smear around the sun
		// the second, the lens the third and the haze the fourth, so the
		// question is whether one of them asked rather than whether that one
		// did.
		self.depth.make_readable(
			&self.device,
			&mut encoder,
			self.seeing.is_some()
				|| self.shafting.is_some()
				|| self.focusing.is_some()
				|| self.hazing.is_some(),
			&self.timings,
		);

		// first of everything after the scene, because the air is between the
		// eye and everything the picture holds: the smear around the sun, the
		// lens, the eye and the bloom all read a picture the air is already in
		if self.hazing.is_some() {
			self.haze.ensure(
				&self.device,
				[&self.globals_layout, self.shadows.sample_layout()],
				&self.built,
			);
		}

		self.haze.render(
			&mut encoder,
			&self.queue,
			haze::Frame {
				asked: self.hazing,
				depth: &self.depth,
				scene: &self.bindings,
				shadows: self.shadows.bindings(),
				picture: self.post.picture(),
				timings: &self.timings,
			},
			view,
		);

		// before the eye is measured and before the bloom is gathered, so
		// that light the air caught is part of the picture both of them read
		self.shaft.render(
			&mut encoder,
			&self.queue,
			self.shafting,
			self.post.picture(),
			&self.depth,
			&self.timings,
		);

		// and after it, because light the air caught is in the picture and a
		// lens is out of focus about the whole picture. Three of the four
		// engines with this effect put it before the bloom and the meter, as
		// this does, and all four put it before the curve: a blur is an
		// average of light, and an average taken after a curve is an average
		// of the wrong numbers.
		self.focus.render(
			&mut encoder,
			&self.queue,
			self.focusing,
			self.post.picture(),
			&self.depth,
			&self.timings,
		);

		let depth = self.seeing.and_then(|(white, lens)| {
			self.depth
				.readable()
				.map(|view| post::Seen { view, white, lens })
		});
		let surfaces = self.showing.and_then(|showing| {
			let view = match showing {
				| prepass::Showing::Occlusion => self.occlusion.done(),
				| prepass::Showing::Normal | prepass::Showing::Roughness =>
					self.prepass.surfaces(),
				| prepass::Showing::Material => self.prepass.material(),
				| prepass::Showing::Reflections | prepass::Showing::Coverage =>
					self.reflection.done(),
				| prepass::Showing::Haze => self.haze.found(),
			};

			view.map(|view| post::Prepared { view, showing })
		});

		self.post.resolve(
			&mut encoder,
			&self.queue,
			world.post,
			seconds,
			post::Last { into: target, depth, surfaces },
			&self.timings,
		);

		// last of all, and into this frame's own encoder: the timestamps are
		// copied out of the query set where the passes that wrote them can
		// still be told apart. Nothing at all when nobody is measuring.
		self.timings.resolve(&mut encoder);
		self.timings.close(Work::Record);
		self.queue.submit([encoder.finish()]);
	}

	/// What a pass after the scene reads, one float a pixel, for a test.
	/// @ref [`Depth::values`].
	#[cfg(test)]
	pub(crate) fn depth_values(&self) -> Option<Vec<f32>> {
		self.depth.values(&self.device, &self.queue)
	}

	/// What the pass before the scene wrote for each pixel's surface, for a
	/// test. @ref [`Prepass::surface_values`].
	#[cfg(test)]
	pub(crate) fn surface_values(&self) -> Option<Vec<[f32; 4]>> {
		self.prepass
			.surface_values(&self.device, &self.queue)
	}

	/// What the pass before the scene wrote for each pixel's depth, for a test.
	#[cfg(test)]
	pub(crate) fn prepass_depth_values(&self) -> Option<Vec<f32>> {
		self.prepass
			.depth_values(&self.device, &self.queue)
	}

	/// How much of the sky each texel of the half-sized buffer sees and how far
	/// along the view it is, for a test. @ref [`Occlusion::values`].
	#[cfg(test)]
	pub(crate) fn occlusion_values(&self) -> Option<Vec<[f32; 2]>> {
		self.occlusion.values(&self.device, &self.queue)
	}

	/// What the pass before the scene wrote for each pixel's material, for a
	/// test. @ref [`Prepass::material_values`].
	#[cfg(test)]
	pub(crate) fn material_values(&self) -> Option<Vec<[f32; 4]>> {
		self.prepass
			.material_values(&self.device, &self.queue)
	}

	/// What the reflections found for each pixel, for a test. @ref
	/// [`Reflection::values`].
	#[cfg(test)]
	pub(crate) fn reflection_values(&self) -> Option<Vec<[f32; 4]>> {
		self.reflection.values(&self.device, &self.queue)
	}

	/// What their first pass found for each texel of the half-sized buffer,
	/// for a test. @ref [`Reflection::raw_values`].
	#[cfg(test)]
	pub(crate) fn reflection_raw_values(&self) -> Option<Vec<[f32; 4]>> {
		self.reflection
			.raw_values(&self.device, &self.queue)
	}

	/// Which making of the reflections' buffers the scene holds, for a test.
	#[cfg(test)]
	pub(crate) const fn reflection_epoch(&self) -> u64 { self.reflection.epoch() }

	/// What the air along each texel's ray sends towards the eye, averaged, at
	/// half the picture's size, for a test. @ref [`Haze::values`].
	#[cfg(test)]
	pub(crate) fn haze_values(&self) -> Option<Vec<[f32; 4]>> {
		self.haze.values(&self.device, &self.queue)
	}

	/// The same before the average, for a test. @ref [`Haze::raw_values`].
	#[cfg(test)]
	pub(crate) fn haze_raw_values(&self) -> Option<Vec<[f32; 4]>> {
		self.haze.raw_values(&self.device, &self.queue)
	}

	/// Which making of the haze's buffers the scene holds, for a test.
	#[cfg(test)]
	pub(crate) const fn haze_epoch(&self) -> u64 { self.haze.epoch() }

	/// Whether the haze's passes were built, for a test. @ref [`Haze::built`].
	#[cfg(test)]
	pub(crate) const fn haze_built(&self) -> bool { self.haze.built() }

	/// What the test for what is behind something nearer is working with, for a
	/// test: its pipelines, whether it made a pyramid, this frame's records and
	/// the matrix the picture was drawn through.
	#[cfg(test)]
	pub(crate) const fn cover_state(&self) -> &Cover { &self.covering.cover }

	/// This frame's records of the picture's lists, in placement order, for a
	/// test.
	#[cfg(test)]
	pub(crate) fn cover_reaches(&self) -> &[Reach] { &self.covering.reaches }

	/// The matrix this frame's picture was drawn through, for a test.
	#[cfg(test)]
	pub(crate) const fn cover_projection(&self) -> Mat4 { self.covering.projection }

	/// One level of the pyramid, for a test. @ref [`Cover::level_values`].
	#[cfg(test)]
	pub(crate) fn cover_level_values(&self, level: u32) -> Option<(u32, Vec<f32>)> {
		self.covering
			.cover
			.level_values(&self.device, &self.queue, level)
	}

	/// The commands this frame's lists were drawn through, five words a batch,
	/// for a test.
	#[cfg(test)]
	pub(crate) fn cover_command_values(&self) -> Option<Vec<u32>> {
		let batches = u32::try_from(self.batches.len() + self.blended.len()).ok()?;

		self.covering
			.cover
			.command_values(&self.device, &self.queue, batches)
	}

	/// Whether the test kept each thing of this frame's lists, for a test.
	#[cfg(test)]
	pub(crate) fn cover_kept_values(&self) -> Option<Vec<u32>> {
		let count = u32::try_from(self.covering.reaches.len()).ok()?;

		self.covering
			.cover
			.kept_values(&self.device, &self.queue, count)
	}

	/// Asks for the pass before the scene whether or not anything reads it,
	/// for a test.
	///
	/// **The one way to have the pass and the picture in the same frame**
	/// before anything that reads the buffer draws the picture as well: the
	/// view that does read it draws the buffer in the picture's place.
	#[cfg(test)]
	pub(crate) const fn prepare_anyway(&mut self, asked: bool) { self.anyway = asked; }

	/// What every list drawn this frame was drawn through, in the order they
	/// were drawn, for a test: for each, how many batches through their own
	/// placements and how many through what the test kept. The large things of
	/// the pass before the scene come first, its small things next in a frame
	/// that has any, and the scene's solid and blended lists last.
	#[cfg(test)]
	pub(crate) fn draws(&self) -> Vec<(usize, usize)> { self.drew.borrow().clone() }

	/// How many batches each shadow map's list was cut into this frame, the
	/// cascades first, for a test.
	#[cfg(test)]
	pub(crate) fn map_batches(&self) -> Vec<usize> { self.casting.iter().map(Vec::len).collect() }

	/// Whether the reflections' first pass was built, for a test. @ref
	/// [`Reflection::built`].
	#[cfg(test)]
	pub(crate) const fn reflection_built(&self) -> bool { self.reflection.built() }

	/// How many times group nought has been made again since the scene was
	/// built, for a test.
	#[cfg(test)]
	pub(crate) const fn rebinds(&self) -> u32 { self.rebound }

	/// What this frame cost, for whoever asked to be told.
	///
	/// Blocks on the queue. @ref [`Timings::settle`] for why, and for why the
	/// mode that calls it opens no window.
	pub fn settle(&mut self) -> crate::timing::Frame {
		let device = self.device.clone();
		let frame = self.timings.settle(&device);

		// after the timestamps, which have already waited for the frame: what
		// the test left out is read the same way, and belongs to the same frame
		self.covering.cover.settle(&device);

		frame
	}

	/// What a frame cost, for whoever asked without being able to wait.
	///
	/// **Does not block.** The window's half of [`settle`](Self::settle): it
	/// pumps the device's callbacks and hands back a frame two or three frames
	/// after the frame it is about, or nothing. @ref [`Timings::poll`].
	pub fn collect(&mut self) -> Option<crate::timing::Frame> {
		let device = self.device.clone();

		self.covering.cover.poll(&device);
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

	/// How much of the world the last frame drew, and how much it had.
	///
	/// Counts for a report, in the same spirit as [`sparks`](Self::sparks): a
	/// number about the project at a given step, which a moved camera or a
	/// switched-off test changes and nothing else does. @ref [`Drawn`].
	///
	/// What the test for what is behind something nearer left out is counted
	/// where it was worked out, and is here once a measuring frame has read it
	/// back: nought in a frame nobody measures. @ref [`cover`].
	#[must_use]
	pub fn drawn(&self) -> Drawn {
		let counted = self.covering.cover.counted();
		let count = |number: u32| usize::try_from(number).unwrap_or(usize::MAX);

		Drawn {
			covered: count(counted.instances),
			covered_triangles: count(counted.triangles),
			..self.drawn
		}
	}

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
	/// The tiles of the atlas are handed out here too, because which lamps
	/// throw a shadow is decided by the order this picked them in. @ref
	/// [`allocate`].
	///
	/// @param world - the world being drawn
	/// @param sight - where the camera is, and what it can see
	/// @return the array the uniform holds, and how many of it is real
	fn lamps(&mut self, world: &World, sight: &Sight) -> ([Lamp; MAX_LAMPS], u32) {
		let room = world
			.cvars
			.float(LAMPS)
			.map_or(MAX_LAMPS, lamp_room);

		let (mut lamps, count) =
			chosen(world, sight.eye, sight.culling.then_some(&sight.view), room, &mut self.lit);

		// the cascades are off means every shadow is off: `r.shadows` is the
		// feature's switch and a lamp's map is part of the feature.
		let tiles = if self.shadowing {
			world
				.cvars
				.float(shadow::LOCAL_LAMPS)
				.map_or(LOCAL_TILES, lamp_room)
				.min(LOCAL_TILES)
		} else {
			0
		};

		self.lamp_maps =
			allocate(&self.lit, &mut lamps, tiles, &mut self.lamp_tiles, &mut self.lamp_views);

		(lamps, count)
	}

	/// Which decals this frame carries, packed for the shader, with the atlas
	/// brought level with every picture the world's decals throw first.
	///
	/// A decal whose picture the atlas should hold and does not is left out
	/// rather than drawn as its tint alone: the atlas gives up only when the
	/// pictures do not fit it at all, and it says so when it does.
	///
	/// @param world - the world being drawn
	/// @param sight - where the camera is, and what it can see
	/// @return the array the uniform holds, and how many of it is real
	fn decals(&mut self, world: &World, sight: &Sight) -> ([Paint; MAX_DECALS], u32) {
		decal::pictures(world, &mut self.pictures);

		if self
			.atlas
			.ensure(&self.device, &self.queue, &world.textures, &self.pictures)
		{
			self.rebind();
		}

		let room = world
			.cvars
			.float(DECALS)
			.map_or(MAX_DECALS, decal::room);

		decal::chosen(
			world,
			sight.eye,
			sight.culling.then_some(&sight.view),
			room,
			&mut self.painted,
		);

		let mut paints = [Paint::NOTHING; MAX_DECALS];
		let mut count = 0;

		for chosen in &self.painted {
			let [color, normal] = chosen.pictures();
			let (Some(color), Some(normal)) =
				(self.placed(&world.textures, color), self.placed(&world.textures, normal))
			else {
				continue;
			};

			if let (Some(paint), Some(slot)) =
				(Paint::of(chosen, color, normal), paints.get_mut(count))
			{
				*slot = paint;
				count += 1;
			}
		}

		(paints, u32::try_from(count).unwrap_or(0))
	}

	/// Where one picture a decal throws is in the atlas.
	///
	/// @param textures - the world's registry
	/// @param picture - the picture, or nothing for a decal throwing none
	/// @return where it is, [`NO_PICTURE`](decal::NO_PICTURE) for no picture
	/// at all, and nothing for a picture the atlas does not hold
	fn placed(&self, textures: &Textures, picture: Option<TextureId>) -> Option<[f32; 4]> {
		picture.map_or(Some(decal::NO_PICTURE), |id| {
			decal::key(textures, id).and_then(|key| self.atlas.rect(key))
		})
	}

	/// Records one list of batches into a pass that is already set up.
	///
	/// Both halves of a frame go through this: what differs between them is the
	/// order they were sorted in and the pipelines their materials name, and
	/// neither of those is visible from here.
	///
	/// **Through what the test kept, in a frame it ran**: the same batches in
	/// the same order, each drawing its run of the kept placements as one
	/// command says. @ref [`cover`].
	///
	/// @param pass - the pass to record into, with groups nought, two and three
	/// already bound
	/// @param batches - the runs to draw, in order
	/// @param offset - which command the list's first batch has
	fn draw(&self, pass: &mut RenderPass<'_>, batches: &[Batch], offset: usize) {
		let through = self.covering.cover.drawn_through().map_or(
			Through::Instances,
			|(placements, commands)| Through::Kept {
				placements,
				commands,
				stride: self.covering.cover.placement(),
				offset,
			},
		);

		self.draw_through(pass, (batches, through), |batch| {
			Some(self.pipelines.get(batch.blend, batch.skinned))
		});
	}

	/// The same, through whichever pipeline `pick` names for each batch.
	///
	/// The scene's table picks for the picture and the pass before the scene
	/// picks its own; everything else about drawing a batch - its geometry, its
	/// skin, its material's group and its run of instances - is the same in
	/// both, which is what keeps the two passes drawing the same triangles.
	///
	/// @param pass - the pass to record into, with groups nought, two and three
	/// already bound
	/// @param (batches, through) - the runs to draw, in order, and whether each
	/// draws its own instances or what the test kept of them
	/// @param pick - the pipeline for a batch, from how it blends and whether
	/// bones move its mesh, or nothing for a batch this pass does not draw
	fn draw_through<'a, F>(
		&'a self,
		pass: &mut RenderPass<'_>,
		(batches, through): (&[Batch], Through<'_>),
		pick: F,
	) where
		F: Fn(&Batch) -> Option<&'a RenderPipeline>,
	{
		// swapped when a batch wants another one rather than once per batch.
		// The solid half is ordered by mesh and material, so a world of crates
		// with one character in it changes pipeline twice however many crates
		// there are.
		let mut bound = None;
		#[cfg(test)]
		let mut drew = (0, 0);

		for (index, batch) in batches.iter().enumerate() {
			let (Some(mesh), Some(material), Some(pipeline)) =
				(self.meshes.get(batch.mesh), self.materials.get(batch.material), pick(batch))
			else {
				continue;
			};
			let Some(paint) = self.paint_of(mesh) else {
				continue;
			};

			let wanted = (batch.blend, batch.skinned);
			if bound != Some(wanted) {
				pass.set_pipeline(pipeline);
				bound = Some(wanted);
			}

			if let Some(skin) = mesh.skin.as_ref() {
				pass.set_vertex_buffer(SKIN_SLOT, skin.slice(..));
			}

			pass.set_bind_group(1, &material.bindings, &[]);
			pass.set_vertex_buffer(0, mesh.vertices.slice(..));
			pass.set_vertex_buffer(PAINT_SLOT, paint.slice(..));
			pass.set_index_buffer(mesh.indices.slice(..), IndexFormat::Uint32);

			match through {
				| Through::Instances => pass.draw_indexed(
					mesh.run(batch.level),
					0,
					batch.first..batch.first + batch.count,
				),
				| Through::Kept { placements, commands, stride, offset } => {
					// the batch's run of what was kept begins where its run of
					// everything did, and the command's first instance is nought
					pass.set_vertex_buffer(
						1,
						placements.slice(u64::from(batch.first) * stride..),
					);
					pass.draw_indexed_indirect(
						commands,
						u64::try_from(offset + index).unwrap_or(u64::MAX / COMMAND_SIZE)
							* COMMAND_SIZE,
					);
				},
			}

			#[cfg(test)]
			tally(&mut drew, through);
		}

		#[cfg(test)]
		self.drew.borrow_mut().push(drew);
	}

	/// Records the pass before the scene, or lets what it writes into go.
	///
	/// The solid half of the picture's own list, through the rectangle the
	/// picture is drawn into: what the view leaves out and what is hidden are
	/// left out of this as well, so a surface is written for exactly the pixels
	/// the scene's pass is about to light. @ref [`prepass`].
	///
	/// **Its large things alone**, in a frame that runs the test for what is
	/// behind something nearer: they are the depth the test reads, and what is
	/// small is drawn after it, @ref [`prepare_small`](Self::prepare_small). A
	/// frame that runs no test has nothing small in it.
	///
	/// @param encoder - the frame's, with the shadows already in it
	/// @param view - the part of the target the picture is drawn into, or the
	/// whole of it
	/// @return whether the pass was recorded this frame
	fn prepare(&mut self, encoder: &mut CommandEncoder, view: Option<Viewport>) -> bool {
		if !self.preparing() {
			self.prepass.release();

			return false;
		}

		let groups = [
			&self.globals_layout,
			&self.material_layout,
			self.shadows.sample_layout(),
			self.joints.layout(),
		];

		if !self
			.prepass
			.ensure(&self.device, &groups, &self.built)
		{
			return false;
		}

		let Some(mut pass) = self
			.prepass
			.begin(encoder, self.timings.writes(Pass::Prepass, Ends::Both))
		else {
			return false;
		};

		// the rectangle the picture is cut to, and one with nothing inside the
		// target leaves the pass cleared and nothing else, as the picture is
		if let Some(asked) = view {
			let Some(inside) = asked.within(self.size.0, self.size.1) else {
				return false;
			};

			cut(&mut pass, inside);
		}

		pass.set_bind_group(0, &self.bindings, &[]);
		pass.set_bind_group(2, self.shadows.bindings(), &[]);
		pass.set_bind_group(3, self.joints.bindings(), &[]);
		pass.set_vertex_buffer(1, self.instances.slice(..));

		self.draw_through(&mut pass, (&self.batches, Through::Instances), |batch| {
			self.prepass
				.pipeline(batch.blend, batch.skinned)
				.filter(|_| !batch.small)
		});

		true
	}

	/// Records the small things of the pass before the scene: what the test
	/// kept of every solid thing too small to be drawn ahead of it. @ref
	/// [`cover::SIZE`].
	///
	/// A pass of its own over the targets the large things left, between the
	/// test and everything that reads what the pass before the scene wrote, and
	/// only in a frame whose list holds something small - which only a frame
	/// that asks for the test can. Through what the test kept when it ran, and
	/// every small thing when it did not: a device that cannot run it, or a
	/// test that would not build, still has every surface written.
	///
	/// @param encoder - the frame's, with the test in it
	/// @param view - the part of the target the picture is drawn into, or the
	/// whole of it
	/// @param prepared - whether the large things' pass was recorded this frame
	///
	/// @note: a frame whose first half was not recorded either holds no targets
	/// to begin again over or has a rectangle with nothing inside the target,
	/// which returns below before anything is drawn - so a mutation pass that
	/// took `prepared` out passed everything. It stays because targets being
	/// held is not proof of whose frame they are.
	fn prepare_small(
		&self,
		encoder: &mut CommandEncoder,
		view: Option<Viewport>,
		prepared: bool,
	) {
		if !prepared || !self.batches.iter().any(|batch| batch.small) {
			return;
		}

		let through = self.covering.cover.drawn_through().map_or(
			Through::Instances,
			|(placements, commands)| Through::Kept {
				placements,
				commands,
				stride: self.covering.cover.placement(),
				offset: 0,
			},
		);
		let Some(mut pass) = self
			.prepass
			.resume(encoder, self.timings.writes(Pass::Small, Ends::Both))
		else {
			return;
		};

		// the same rectangle as the first half: a new pass is cut to the whole
		// target until it is told otherwise
		if let Some(asked) = view {
			let Some(inside) = asked.within(self.size.0, self.size.1) else {
				return;
			};

			cut(&mut pass, inside);
		}

		pass.set_bind_group(0, &self.bindings, &[]);
		pass.set_bind_group(2, self.shadows.bindings(), &[]);
		pass.set_bind_group(3, self.joints.bindings(), &[]);
		pass.set_vertex_buffer(1, self.instances.slice(..));

		self.draw_through(&mut pass, (&self.batches, through), |batch| {
			self.prepass
				.pipeline(batch.blend, batch.skinned)
				.filter(|_| batch.small)
		});
	}

	/// Whether anything reads what the pass before the scene writes, this
	/// frame.
	///
	/// Its readers are the views that draw it or what is worked out from it,
	/// the occlusion and the reflections - and not the view of the air, which
	/// is worked out from the depth after the scene. A test can ask for it with
	/// nothing reading it at all, @ref
	/// [`prepare_anyway`](Self::prepare_anyway).
	fn preparing(&self) -> bool {
		self.showing
			.is_some_and(|showing| showing != prepass::Showing::Haze)
			|| self.occluding.is_some()
			|| self.reflecting.is_some()
			|| self.asked_anyway()
	}

	/// Whether a test asked for the pass with nothing to read it.
	#[cfg(test)]
	const fn asked_anyway(&self) -> bool { self.anyway }

	/// Nothing outside a test asks for a pass nobody reads.
	#[cfg(not(test))]
	#[expect(
		clippy::unused_self,
		reason = "the test build's twin reads a field this build does not have"
	)]
	const fn asked_anyway(&self) -> bool { false }

	/// Which one draws a batch into a shadow map.
	///
	/// A match rather than a lookup, so that a mode nobody has thought about
	/// yet is a compile error here on the day it is added rather than a
	/// fence-shaped hole in the light.
	///
	/// @param local - whether the map is a lamp's rather than a cascade's
	/// @param blend - how the surface reads its picture's alpha
	/// @param skinned - whether bones move it
	/// @return the pipeline, or nothing for a surface that does not cast
	fn casting(&self, local: bool, blend: Blend, skinned: bool) -> Option<&RenderPipeline> {
		let masked = match blend {
			| Blend::Opaque => false,
			| Blend::Mask => true,
			// this is the second of two places that say a blended surface casts
			// nothing - the first being which list `group` filed it in, which
			// is why this pass never actually meets one. It is kept because it
			// is the line a reader comes here looking for.
			| Blend::Alpha => return None,
		};

		Some(self.shadows.casting(local, masked, skinned))
	}

	/// Records one cascade's depth pass.
	///
	/// Everything solid this cascade's box can see, through a pipeline with no
	/// fragment stage and no color target, so the whole pass is geometry
	/// against depth. @ref [`casts_of`](Self::casts_of) for which batches.
	///
	/// Solid only. A blended surface writes no depth and so has nothing to say
	/// about what a light can reach, which is what every engine checked does
	/// with one; here it is not a rule anywhere but a consequence of which
	/// lists [`consider`](Self::consider) filed it in.
	///
	/// The pass is begun and its layer cleared whatever the list holds, empty
	/// or not: a layer left alone holds last frame's depths, and a cascade
	/// whose box sees nothing solid has to read as lit rather than as whatever
	/// stood there a frame ago.
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
		let Some(layer) = self.shadows.layer(slice) else {
			return;
		};

		let mut pass = shadow_pass(encoder, layer, marks);
		pass.set_bind_group(1, self.joints.bindings(), &[]);
		pass.set_vertex_buffer(1, self.instances.slice(..));

		self.draw_map(&mut pass, slice, false);
	}

	/// Records every local light's map, in one pass over the atlas's last
	/// layer.
	///
	/// **One pass, not one per tile.** The layer is cleared once, and each map
	/// is then drawn through a viewport and a scissor cut to its own
	/// rectangle, which is what an atlas is for and what keeps sixteen shadow
	/// maps costing the frame one pass rather than sixteen. A clear cannot be
	/// cut to a rectangle, so the order has to be that way round.
	///
	/// The pass is begun and the layer cleared whether or not any lamp asked
	/// for a tile, for the reason a cascade's is: a layer left alone holds last
	/// frame's depths.
	///
	/// @param encoder - what to record into
	/// @param marks - the span, and `None` in a frame nobody is measuring
	fn cast_lamps(
		&self,
		encoder: &mut CommandEncoder,
		marks: Option<RenderPassTimestampWrites<'_>>,
	) {
		let Some(layer) = self.shadows.layer(CASCADES) else {
			return;
		};

		let mut pass = shadow_pass(encoder, layer, marks);
		pass.set_bind_group(1, self.joints.bindings(), &[]);
		pass.set_vertex_buffer(1, self.instances.slice(..));

		for tile in 0..self.lamp_maps.min(LOCAL_TILES) {
			let [x, y, side] = Tile::local(tile).viewport();
			let (at_x, at_y, extent) = (texels(x), texels(y), texels(side));

			pass.set_viewport(at_x, at_y, extent, extent, 0.0, 1.0);
			pass.set_scissor_rect(x, y, side, side);

			self.draw_map(&mut pass, CASCADES.saturating_add(tile), true);
		}
	}

	/// Draws one shadow map's list into a pass already begun.
	///
	/// @param pass - the pass, with its joints and instances already bound
	/// @param map - which of the frame's maps: a cascade, then a lamp's tile
	/// @param local - whether it is a lamp's, which picks the unbiased
	/// pipelines
	fn draw_map(&self, pass: &mut RenderPass<'_>, map: usize, local: bool) {
		let Some(slot) = self.shadows.slot(map) else {
			return;
		};

		pass.set_bind_group(0, slot, &[]);

		// the same swap the scene pass makes, and it has to be made here too:
		// a character whose shadow were cast from its bind pose would stand in
		// one attitude and be shadowed in another.
		let mut bound = None;

		for batch in self.casts_of(map) {
			let (Some(mesh), Some(material)) =
				(self.meshes.get(batch.mesh), self.materials.get(batch.material))
			else {
				continue;
			};
			let Some(paint) = self.paint_of(mesh) else {
				continue;
			};

			let Some(pipeline) = self.casting(local, batch.blend, batch.skinned) else {
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
				pass.set_vertex_buffer(SKIN_SLOT, skin.slice(..));
			}

			pass.set_vertex_buffer(0, mesh.vertices.slice(..));
			pass.set_vertex_buffer(PAINT_SLOT, paint.slice(..));
			pass.set_index_buffer(mesh.indices.slice(..), IndexFormat::Uint32);
			pass.draw_indexed(mesh.run(batch.level), 0, batch.first..batch.first + batch.count);
		}
	}

	/// The batches one cascade draws.
	///
	/// Its own list in a frame that is culling. In one that is not, the
	/// picture's solid half - which, with nothing left out of the picture, is
	/// every solid entity in the world, exactly what every frame drew into
	/// every cascade before there was a test.
	///
	/// @param map - which of the frame's maps: a cascade, then a lamp's tile
	fn casts_of(&self, map: usize) -> &[Batch] {
		if self.culling {
			self.casting.get(map).map_or(&[], Vec::as_slice)
		} else {
			&self.batches
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
	/// Puts the world's environment on the device, and rebuilds group nought
	/// when it moved.
	///
	/// A texture rather than a matrix, so it is done here rather than in the
	/// uniform the rest of the frame is written into. A frame whose sky did not
	/// change does nothing at all.
	fn sync_environment(&mut self, world: &World) {
		let wanted = world.cvars.bool(env::ENABLED).unwrap_or(true);

		if self
			.environment
			.update(&self.device, &self.queue, world, wanted)
		{
			self.rebind();
		}
	}

	/// Everything the frame reads that lives in a registry rather than in the
	/// uniform.
	///
	/// Geometry, pictures, materials, the debug pen's lines, the particles'
	/// groups and the environment: all of them are "what is on the device
	/// matching what the world holds", and none of them depends on where the
	/// camera is this frame.
	fn sync_tables(&mut self, world: &World) {
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
		self.sync_environment(world);
	}

	/// How many roughness levels the bound environment has, nought for none.
	///
	/// **The word the shader branches on.** Nought says there is no environment
	/// and the shading takes the line it took before there were any, which is
	/// what makes a world with no sky draw what it drew. It rides on the sky's
	/// own vector rather than in a word of its own, because a number about the
	/// sky belongs beside the sky's colors.
	fn roughness_levels(&self) -> f32 {
		f32::from(u16::try_from(self.environment.levels()).unwrap_or(0))
	}

	/// Group nought, rebuilt around whatever the atlas, the environment, the
	/// share of the sky and what the reflections found are now.
	///
	/// Four things make it stale. Two are rare, the decals' atlas growing and
	/// the world naming a different environment; the other two are the share's
	/// buffers and the reflections' being made or let go, which is a frame that
	/// starts or stops asking for one of them and a picture that changes size.
	/// @ref [`frame_bindings`].
	fn rebind(&mut self) {
		self.bindings = frame_bindings(
			&self.device,
			&self.globals_layout,
			&self.globals,
			&self.atlas,
			&Held {
				decals: &self.decal_sampler,
				environment: &self.environment,
				split: &self.split,
				occlusion: self.occlusion.bound(),
				reflection: self.reflection.bound(),
			},
		);
		self.occluded = self.occlusion.epoch();
		self.reflected = self.reflection.epoch();

		#[cfg(test)]
		{
			self.rebound += 1;
		}
	}

	fn upload(&mut self, world: &World, view: Option<Viewport>) {
		self.sync_tables(world);

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
		self.cascades = self.fitted(world, &camera);

		let light_view_projection = self
			.cascades
			.matrices
			.map(|it| it.to_cols_array_2d());
		let projection = camera.view_projection(world.aspect);
		// what everything below is asked against: the view the picture is
		// drawn through and each cascade's box, out of the very matrices the
		// passes use, so that what is left out and what the hardware would have
		// clipped whole are one thing. @ref [`cull`].
		let sight = Sight {
			eye: camera.position,
			forward: (camera.target - camera.position).normalize_or(Vec3::NEG_Z),
			view: Frustum::of(projection),
			cascades: self
				.shadowing
				.then(|| self.cascades.matrices.map(Frustum::of)),
			// filled in below, once the lamps have been picked and their tiles
			// handed out: a face's volume is the very matrix its pass draws
			// through, which does not exist until then.
			faces: [Frustum::of(Mat4::IDENTITY); LOCAL_TILES],
			maps: 0,
			culling: world.cvars.bool(cull::ENABLED).unwrap_or(true),
			detail: self.eye_of(world, &camera, view),
			// filled in below, once whether the test runs this frame is known
			small_under: None,
		};
		let (lamps, count) = self.lamps(world, &sight);
		let sight = Sight {
			faces: self.lamp_volumes(),
			maps: self.lamp_maps,
			..sight
		};
		let (decals, painted) = self.decals(world, &sight);
		self.sky = world.sky.is_drawn();
		self.seeing = seeing_of(world, &camera);
		// the depth view wins when both are asked for, and a view nobody sees
		// asks for no pass
		self.showing = prepass::showing_of(world).filter(|_| self.seeing.is_none());
		self.occluding = occlusion::asking_of(world, &camera, self.showing);
		self.reflecting = reflection::asking_of(world, &camera, self.showing);
		self.hazing = haze::asking_of(world, &camera, self.showing);
		self.shafting = shaft::asking_of(world, &camera);
		self.focusing = focus::asking_of(world, &camera);
		// after every reader of the pass before the scene has said whether it
		// reads: the test reads that pass's depth and does not ask for it
		let sight = self.ask_cover(world, &sight);

		self.queue.write_buffer(
			&self.globals,
			0,
			bytemuck::bytes_of(&Globals {
				view_projection: projection.to_cols_array_2d(),
				inverse_view_projection: projection.inverse().to_cols_array_2d(),
				light: world.light.extend(0.0).to_array(),
				ambient: world.ambient.extend(0.0).to_array(),
				eye: camera.position.extend(1.0).to_array(),
				forward: sight.forward.extend(0.0).to_array(),
				light_view_projection,
				splits: self.cascades.splits,
				cascade_texels: self.cascades.texels,
				cascade_tiles: shadow::cascade_tiles(),
				lamp_tiles: self.lamp_tiles,
				lamp_views: self.lamp_views,
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
				sky_horizon: world
					.sky
					.horizon
					.extend(self.roughness_levels())
					.to_array(),
				sky_ground: world.sky.ground.extend(0.0).to_array(),
				counts: [count, painted, 0, 0],
				lamps,
				decals,
			}),
		);

		if self.shadowing {
			self.shadows.upload(&self.queue, &self.cascades);
			self.upload_lamp_views();
		}

		self.group(world, &sight);
		// after the grouping, which starts every count over
		self.drawn.lamps = usize::try_from(count).unwrap_or(0);
		self.drawn.decals = usize::try_from(painted).unwrap_or(0);

		self.hand_over_to_cover(projection);

		if self.placements.is_empty() {
			return;
		}

		self.queue
			.write_buffer(&self.instances, 0, bytemuck::cast_slice(&self.placements));
		self.joints.upload(&self.queue);
	}

	/// Whether this frame runs the test for what is behind something nearer,
	/// and how small a thing it draws into the pass before the scene after it.
	///
	/// @param world - the world being drawn, for the console
	/// @param sight - what this frame can see, with the frustum test's switch
	/// @return the same sight, saying what is small in a frame that runs the
	/// test and that nothing is in one that does not
	fn ask_cover(&mut self, world: &World, sight: &Sight) -> Sight {
		self.covering.asked = sight.culling && cover::asking_of(world) && self.preparing();

		Sight {
			small_under: cover::sizing_of(world).filter(|_| self.covering.asked),
			..*sight
		}
	}

	/// What this frame asks every thing's level against. @ref [`detail`].
	///
	/// @param world - the world being drawn, for its aspect and the console
	/// @param camera - the camera the picture is drawn from
	/// @param view - the rectangle the picture is drawn into, when it is drawn
	/// into one: a level is a pixel's worth of the picture that is seen, and
	/// the rest of the target is not it
	fn eye_of(&self, world: &World, camera: &Camera, view: Option<Viewport>) -> detail::Eye {
		detail::Eye::new(
			camera.position,
			camera.projection(world.aspect),
			view.and_then(|asked| asked.within(self.size.0, self.size.1))
				.map_or(self.size.1, |inside| inside.height),
			world
				.cvars
				.float(detail::THRESHOLD)
				.unwrap_or(detail::DEFAULT_THRESHOLD),
		)
	}

	/// This frame's cascades, or none at all with the shadows switched off.
	///
	/// @param world - the world being drawn, for the console
	/// @param camera - the camera the *picture* is drawn from, which is the
	/// one the slices have to be cut along
	fn fitted(&self, world: &World, camera: &Camera) -> Cascades {
		if !self.shadowing {
			return Cascades::NONE;
		}

		let distance = world
			.cvars
			.float(shadow::DISTANCE)
			.unwrap_or(shadow::DEFAULT_DISTANCE);

		shadow::fit(camera, world.aspect, world.light, distance)
	}

	/// What each local map this frame draws can see.
	///
	/// One volume per tile, out of the very matrix that tile's pass draws
	/// through - the same rule the cascades' boxes follow, so that what a map
	/// leaves out and what the hardware would have clipped whole are one
	/// thing. Only the first [`lamp_maps`](Self::lamp_maps) are real.
	fn lamp_volumes(&self) -> [Frustum; LOCAL_TILES] {
		core::array::from_fn(|tile| {
			let view = self
				.lamp_views
				.get(tile)
				.copied()
				.unwrap_or_else(|| Mat4::IDENTITY.to_cols_array_2d());

			Frustum::of(Mat4::from_cols_array_2d(&view))
		})
	}

	/// Writes each local map's matrix into the slot its pass reads.
	///
	/// Beside [`Maps::upload`], which does the same for the cascades; apart
	/// from it because a frame has four of those and however many of these the
	/// atlas had room for.
	fn upload_lamp_views(&self) {
		for tile in 0..self.lamp_maps.min(LOCAL_TILES) {
			let Some(view) = self.lamp_views.get(tile) else {
				continue;
			};

			self.shadows
				.upload_local(&self.queue, tile, Mat4::from_cols_array_2d(view));
		}
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

			if uploaded.paint.is_none() {
				self.fit_plain(uploaded.vertex_count);
			}

			match self.meshes.get_mut(slot) {
				| Some(existing) => *existing = uploaded,
				| None => self.meshes.push(uploaded),
			}
		}
	}

	/// Makes sure the plain buffer holds at least so many entries: made the
	/// first time a mesh nobody painted is uploaded, and grown, never shrunk,
	/// when a longer one arrives. Always before anything draws, because the
	/// meshes are uploaded at the top of a frame.
	///
	/// @param vertices - how many vertices the mesh reading it has
	fn fit_plain(&mut self, vertices: usize) {
		if self
			.plain
			.as_ref()
			.is_some_and(|plain| plain.vertices >= vertices)
		{
			return;
		}

		self.plain = Some(Plain::new(&self.device, &self.queue, vertices));
	}

	/// The buffer a mesh's paint is read from: its own, or the plain one.
	///
	/// Nothing only for a mesh drawn before any mesh nobody painted was ever
	/// uploaded, which the order of a frame rules out; a draw that met it would
	/// be skipped rather than bound to nothing.
	///
	/// @param mesh - the mesh being drawn
	fn paint_of<'a>(&'a self, mesh: &'a GpuMesh) -> Option<&'a Buffer> {
		mesh.paint
			.as_ref()
			.or_else(|| self.plain.as_ref().map(|plain| &plain.buffer))
	}

	/// How many vertices a mesh nobody painted may have before the plain
	/// buffer has to grow, which is how a test shows that it did.
	#[cfg(test)]
	pub(crate) fn plain_vertices(&self) -> usize {
		self.plain
			.as_ref()
			.map_or(0, |plain| plain.vertices)
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

			// an environment is in the same registry as every picture, because
			// it is compiled by the same compiler into the same format - but
			// nothing here can sample one. A material's group binds a flat
			// picture and there is no arithmetic in the shader that would know
			// which of six faces to read. So the slot gets the white texel and
			// the cube is put on the device by @ref [`env`] instead, once, for
			// the whole frame rather than per material.
			let flat = &TextureData::white();
			let value = if texture.value().is_cube() {
				flat
			} else {
				texture.value()
			};
			let uploaded = upload_texture(&self.device, &self.queue, value, texture.revision());
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

	/// Lays everything some pass can see out in the instance buffer, grouped by
	/// mesh and material, one run per list.
	///
	/// A sort rather than the counting pass this used to be. Counting works
	/// while the key is one small index; the key is a pair now, and a counter
	/// per combination would be a table of meshes times materials for the sake
	/// of the handful of pairs a scene actually uses.
	///
	/// **Five lists, and the picture's is the one it always was.** It is sorted
	/// and batched exactly as before, less what the view cannot see, so the
	/// order anything is drawn in does not change and neither does the
	/// picture. Each cascade gets a list of its own, out of one sort of
	/// everything solid some cascade can see: the four boxes are four
	/// different volumes, and a thing is usually inside one or two of them.
	/// @ref [`cull`].
	///
	/// @param world - the world being drawn
	/// @param sight - what this frame can see, and whether to ask
	fn group(&mut self, world: &World, sight: &Sight) {
		self.order.clear();
		self.casters.clear();
		self.staged.clear();
		self.joints.begin(world);
		self.culling = sight.culling;
		self.drawn = Drawn::default();
		self.covering.reached.clear();
		self.covering.reaches.clear();
		self.covering.commands.clear();

		// one lookup of the record for the whole walk, rather than one a thing:
		// the question is asked of every thing in view in a frame the test runs
		let drawing = world.entities.column(&DRAWING);

		for (id, _, renderable) in world.entities.iter() {
			let covers = drawing
				.and_then(|column| column.get(id.slot()))
				.is_some_and(|drawing| drawing.covers());

			self.consider(world, sight, id, renderable, covers);
		}

		// a frame whose solid things are all small has nothing ahead of the test
		// to hide anything: a pyramid of nothing leaves nothing out of either
		// pass, so all of them are drawn ahead of it, which is the pass unsplit
		if !self
			.order
			.iter()
			.any(|entry| entry.blend != Blend::Alpha && !entry.small)
		{
			for entry in &mut self.order {
				entry.small = false;
			}
		}

		self.order.sort_unstable_by_key(Sorted::key);
		self.casters.sort_unstable_by_key(Sorted::key);

		self.placements.clear();
		self.batches.clear();
		self.blended.clear();

		for index in 0..self.order.len() {
			if let Some(entry) = self.order.get(index).copied() {
				self.place(entry, None);
			}
		}

		for map in 0..MAPS {
			self.place_casters(map);
		}

		self.drawn.seen = self.order.len();
		self.drawn.small = self
			.order
			.iter()
			.filter(|entry| entry.small)
			.count();
		self.drawn.lowered = self
			.order
			.iter()
			.filter(|entry| entry.level > 0)
			.count();
		self.drawn.triangles = self
			.batches
			.iter()
			.chain(&self.blended)
			.map(|batch| {
				let run = self
					.meshes
					.get(batch.mesh)
					.map_or(0, |uploaded| uploaded.run(batch.level).len());

				run / 3 * usize::try_from(batch.count).unwrap_or(0)
			})
			.sum();
		self.drawn.cast = if sight.cascades.is_none() {
			0
		} else if sight.culling {
			self.casting
				.iter()
				.take(CASCADES)
				.map(|list| instances(list))
				.sum()
		} else {
			CASCADES * instances(&self.batches)
		};
		self.drawn.lamp_casts = if sight.culling {
			self.casting
				.iter()
				.skip(CASCADES)
				.take(self.lamp_maps)
				.map(|list| instances(list))
				.sum()
		} else {
			self.lamp_maps * instances(&self.batches)
		};
	}

	/// Decides which of the frame's lists one entity is in: the picture's,
	/// some cascades', both, or none at all.
	///
	/// @param world - the world being drawn
	/// @param sight - what this frame can see
	/// @param id - the entity
	/// @param renderable - what it draws
	/// @param covers - whether its record says it is drawn ahead of the test
	/// for what is hidden whatever its size. @ref `colby_core::abi::Drawing`
	fn consider(
		&mut self,
		world: &World,
		sight: &Sight,
		id: EntityId,
		renderable: &Renderable,
		covers: bool,
	) {
		let mesh = renderable.mesh.slot();
		if mesh == 0 || mesh >= world.meshes.len() {
			return;
		}

		let material = renderable
			.material
			.slot()
			.min(world.materials.len().saturating_sub(1));
		let (Ok(mesh), Ok(material)) = (u32::try_from(mesh), u32::try_from(material)) else {
			return;
		};

		// before anything else is asked of it, so a hidden entity costs the
		// walk up its chain and nothing more: no transform, no bounds and no
		// pose gathered. And before the count, so that `meshes` is what could
		// be drawn and `hidden` is what the flag took out, the shadows along
		// with the picture. @ref `Entities::shown`.
		if !world.entities.shown(id) {
			self.drawn.hidden += 1;

			return;
		}

		// the transform to *draw* with, which is not the one the game wrote:
		// it is somewhere between that one and the one before it. @ref
		// [`World::render_transform`].
		let Some(transform) = world.render_transform(id) else {
			return;
		};

		let blend = world
			.materials
			.get(renderable.material)
			.map_or(Material::DEFAULT.blend, |surface| surface.blend);

		self.drawn.meshes += 1;

		let placed = self.reach(world, renderable, mesh, transform);
		let seen = !sight.culling || sight.view.holds(&placed);
		// a blended surface writes no depth and so casts nothing, and a frame
		// that is not culling has no shadow lists to put anything into. The
		// lamps' maps are the same rule with the same test over a different
		// set of volumes, and their bits sit past the cascades'.
		let casting = sight.culling && blend != Blend::Alpha;
		let casts = sight
			.cascades
			.as_ref()
			.filter(|_| casting)
			.map_or(0, |cascades| casts_into(cascades, &placed))
			| if casting { throws_into(sight, &placed) } else { 0 };

		if !seen && casts == 0 {
			return;
		}

		let Some(at) = self.stage(world, id, renderable, transform) else {
			return;
		};

		// beside the placement it was staged with, at the same index: the test
		// asks the very box the frustum was asked
		if self.covering.asked {
			self.covering.reached.push(placed);
		}

		let blended = blend == Blend::Alpha;
		// once, here, for every list the thing goes into: the picture, the pass
		// before it, the test for what is behind something nearer and every map
		// draw the same surface. @ref [`detail`].
		let level = self.level_of(sight, &placed, mesh, transform);
		// asked only of a solid thing in view in a frame that runs the test: the
		// only list it changes is the picture's solid one
		//
		// @note: a thing out of view goes only into the maps' lists, whose
		// entries carry no size, so a mutation pass that asked it of those too
		// passed everything. It is here so that they take no distance.
		//
		// and never of a thing that covers: its record says it is one of many
		// small pieces of something that hides a great deal, which only the
		// pass ahead of the test puts into the depth the test reads
		let small = seen && !blended && !covers && sight.small(&placed);
		let entry = Sorted {
			pass: u8::from(blended),
			// worked out only for the half that is sorted on it, as how far
			// along the view the middle of the entity's box stands: the same
			// quantity the cascades are cut on and the shader picks a slice
			// with, a projection onto the direction the camera looks rather
			// than the distance to it, so two panes side by side are at one
			// depth. The box is its bones' box for a mesh bones move, which
			// sorts a character that fell over where it lies.
			depth: if blended {
				-grain((placed.center - sight.eye).dot(sight.forward))
			} else {
				0
			},
			mesh,
			level,
			material,
			small,
			entity: id,
			blend,
			at,
			casts,
		};

		if seen {
			self.order.push(entry);
		}

		// a map draws large and small alike, so its batches are not cut on it
		if casts != 0 {
			self.casters
				.push(Sorted { small: false, ..entry });
		}
	}

	/// Which of a mesh's levels one thing is drawn at this frame.
	///
	/// @param sight - what this frame can see, the eye among it
	/// @param placed - the thing's box, in the world
	/// @param mesh - its uploaded geometry's slot
	/// @param transform - where it is drawn, for its largest size
	/// @return nought for the mesh itself, one for its finest coarser level
	fn level_of(&self, sight: &Sight, placed: &Placed, mesh: u32, transform: Transform) -> u8 {
		let Some(uploaded) = usize::try_from(mesh)
			.ok()
			.and_then(|slot| self.meshes.get(slot))
		else {
			return 0;
		};
		let size = transform.scale.abs().max_element();

		u8::try_from(sight.detail.level(placed, size, &uploaded.errors)).unwrap_or(u8::MAX)
	}

	/// The box an entity's geometry fills in the world this frame.
	///
	/// The mesh's own box carried by the entity's matrix, or, for a mesh bones
	/// move, the box its bones have put it in, carried the same way. Asking
	/// for a pose's matrices here gathers them a little before drawing would
	/// have, and drawing then finds the same run. @ref [`cull::posed`].
	///
	/// @param world - the world being drawn, for the pose
	/// @param renderable - what the entity draws
	/// @param mesh - its uploaded geometry's slot
	/// @param transform - where it is drawn this frame
	/// @return the box, in the world
	fn reach(
		&mut self,
		world: &World,
		renderable: &Renderable,
		mesh: u32,
		transform: Transform,
	) -> Placed {
		let matrix = transform.matrix();
		let Some(uploaded) = usize::try_from(mesh)
			.ok()
			.and_then(|slot| self.meshes.get(slot))
		else {
			// a slot the upload has not reached draws nothing either, and a
			// point where the entity stands is as good an answer as any
			return Bounds::default().carried(matrix);
		};

		if uploaded.bones.is_empty() {
			return uploaded.bounds.carried(matrix);
		}

		let run = self.joints.take(world, renderable.pose);

		cull::posed(&uploaded.bones, self.joints.run(run))
			.unwrap_or(uploaded.bounds)
			.carried(matrix)
	}

	/// Flattens one entity into what the vertex stage reads.
	///
	/// Once per frame however many lists the entity is in; each list copies
	/// the result. @ref [`staged`](Self::staged).
	///
	/// @param world - the world being drawn, for the material and the pose
	/// @param id - the entity, for its own flags
	/// @param renderable - what the entity draws
	/// @param transform - where it is drawn this frame
	/// @return where in [`staged`](Self::staged) it went
	fn stage(
		&mut self,
		world: &World,
		id: EntityId,
		renderable: &Renderable,
		transform: Transform,
	) -> Option<u32> {
		let surface = world
			.materials
			.get(renderable.material)
			.copied()
			.unwrap_or(Material::DEFAULT);

		let at = u32::try_from(self.staged.len()).ok()?;
		// the first entity of the frame to name a pose is what gathers it;
		// the second finds the same run rather than a second copy of it. The
		// third word is the entity's own flags, which the joints leave alone.
		let mut skin = self.joints.take(world, renderable.pose);
		skin[2] = u32::from(!world.entities.takes_decals(id));

		self.staged.push(Placement {
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
			skin,
		});

		Some(at)
	}

	/// Writes one cascade's list: everything solid its box can see, in the
	/// order the casters were sorted in.
	///
	/// @param slice - which cascade, nearest first
	fn place_casters(&mut self, map: usize) {
		if let Some(list) = self.casting.get_mut(map) {
			list.clear();
		}

		for index in 0..self.casters.len() {
			let Some(entry) = self
				.casters
				.get(index)
				.copied()
				.filter(|entry| (entry.casts & (1 << map)) != 0)
			else {
				continue;
			};

			self.place(entry, Some(map));
		}
	}

	/// Writes one sorted entry into the instance buffer, opening a new batch
	/// when its pair differs from the one before it.
	///
	/// @param entry - what to write
	/// @param into - which list: `None` for the picture's, a cascade's index
	/// for one of theirs
	fn place(&mut self, entry: Sorted, into: Option<usize>) {
		let Some(placement) = usize::try_from(entry.at)
			.ok()
			.and_then(|at| self.staged.get(at))
			.copied()
		else {
			return;
		};

		let (mesh, material) = (
			usize::try_from(entry.mesh).unwrap_or(0),
			usize::try_from(entry.material).unwrap_or(0),
		);
		// asked of the uploaded geometry rather than of the entity: what
		// decides the pipeline is whether there are bones and weights to read,
		// and an entity naming a pose over a mesh that has none is drawn as
		// the shape it is.
		let skinned = self
			.meshes
			.get(mesh)
			.is_some_and(|uploaded| uploaded.skin.is_some());

		let Ok(first) = u32::try_from(self.placements.len()) else {
			return;
		};

		// the list the sort already put this entity in. Reading the mode again
		// here rather than taking the one `consider` decided on would be two
		// answers to one question, and the day they differed a batch would be
		// drawn in a pass its neighbors are not in.
		let batches = match into {
			| Some(slice) => match self.casting.get_mut(slice) {
				| Some(list) => list,
				| None => return,
			},
			| None if entry.blend == Blend::Alpha => &mut self.blended,
			| None => &mut self.batches,
		};

		self.placements.push(placement);

		match batches.last_mut() {
			| Some(batch)
				if batch.mesh == mesh
					&& batch.level == entry.level
					&& batch.material == material
					&& batch.small == entry.small =>
				batch.count += 1,
			| _ => batches.push(Batch {
				mesh,
				level: entry.level,
				material,
				first,
				count: 1,
				small: entry.small,
				skinned,
				blend: entry.blend,
			}),
		}

		if into.is_none() && self.covering.asked {
			self.lay_down_reach(entry, mesh);
		}
	}

	/// The test's record of one placement of the picture's lists, laid down in
	/// the same order as the placement.
	///
	/// Its batch is the last one of its list, which is the one [`place`]
	/// (Self::place) has just put it in; the blended batches' commands come
	/// after the solid ones', because the solid list is laid out first.
	///
	/// @param entry - what was placed
	/// @param mesh - its uploaded geometry's slot
	fn lay_down_reach(&mut self, entry: Sorted, mesh: usize) {
		let (list, before) = if entry.blend == Blend::Alpha {
			(&self.blended, self.batches.len())
		} else {
			(&self.batches, 0)
		};
		let (Some(batch), Some(placed)) = (
			list.last(),
			usize::try_from(entry.at)
				.ok()
				.and_then(|at| self.covering.reached.get(at)),
		) else {
			return;
		};
		let number = u32::try_from(before + list.len() - 1).unwrap_or(u32::MAX);
		let triangles = self
			.meshes
			.get(mesh)
			.map_or(0, |uploaded| uploaded.run(batch.level).len() / 3);
		let triangles = u32::try_from(triangles).unwrap_or(u32::MAX);

		self.covering
			.reaches
			.push(Reach::of(placed, number, batch.first, triangles));
	}

	/// What the grouping laid out for the test, handed to it: this frame's
	/// matrix, and in a frame that asks, one command a batch and every thing's
	/// record.
	///
	/// @param projection - the matrix this frame's picture is drawn through
	fn hand_over_to_cover(&mut self, projection: Mat4) {
		self.covering.projection = projection;

		if !self.covering.asked {
			return;
		}

		self.lay_out_commands();
		self.covering
			.cover
			.upload(&self.queue, &self.covering.reaches, &self.covering.commands);
	}

	/// One command a batch of the picture's lists, the solid list first: the
	/// run of its mesh's indices its level draws - how many, and where the run
	/// starts - every instance count at nought until the test counts what it
	/// kept. @ref [`cover`].
	fn lay_out_commands(&mut self) {
		for batch in self.batches.iter().chain(&self.blended) {
			let run = self
				.meshes
				.get(batch.mesh)
				.map_or(0..0, |uploaded| uploaded.run(batch.level));

			self.covering
				.commands
				.extend_from_slice(&[run.end - run.start, 0, run.start, 0, 0]);
		}
	}
}

/// The two things that read what the pass before the scene writes, and the
/// haze, which reads the depth after it: none of them has made a buffer yet.
///
/// @param device - the device to build against
/// @param queue - where the occlusion's and the reflections' one texels are
/// written
/// @param (width, height) - the picture's size
fn readers(
	device: &Device,
	queue: &Queue,
	(width, height): (u32, u32),
) -> Result<(Occlusion, Reflection, Haze)> {
	Ok((
		Occlusion::new(device, queue, width, height)?,
		Reflection::new(device, queue, width, height)?,
		Haze::new(device, width, height)?,
	))
}

/// Everything the scene's own pass draws with that has to agree about how
/// many samples a pixel is: the table of pipelines, the depth they test
/// against, the lines and the particles. @ref `Scene::sampling`.
///
/// All of them draw into the float target rather than into the window: what
/// reaches the window is the composite, and it is the only thing built for the
/// window's own format. And at one sample whatever the console will say,
/// because there is no console yet: a `Scene` is built before the world it
/// draws exists. The first frame reads the variable and rebuilds if it has to,
/// which is the same path a person turning it on mid-run takes.
///
/// @param device - the device to build against
/// @param groups - the scene's four bind group layouts, in group order
/// @param source - the WGSL the table is built from
/// @param (width, height) - the picture's size
fn drawing(
	device: &Device,
	groups: [&BindGroupLayout; 4],
	source: &str,
	(width, height): (u32, u32),
) -> Result<(Pipelines, Depth, Lines, Sparks)> {
	Ok((
		Pipelines::build(device, post::HDR_FORMAT, &groups, source, post::NO_SAMPLES)?,
		Depth::new(device, post::NO_SAMPLES, width, height)?,
		Lines::new(device, post::HDR_FORMAT, groups[0], post::NO_SAMPLES)?,
		Sparks::new(device, post::HDR_FORMAT, post::NO_SAMPLES),
	))
}

/// The layout of group nought: the frame's uniform; the decals' atlas read two
/// ways with the sampler that reads it; the environment and the split-sum
/// table, each with its own; how much of the sky each pixel sees; and what each
/// pixel's reflection found.
///
/// **One group for all of it**, rather than a fifth for the atlas: a device
/// need only allow four, and all four are spoken for. The atlas belongs with
/// the uniform anyway, because both are the frame's and neither is a
/// material's. Every pipeline built against this - the sky, the debug lines,
/// the particles, the pass before the scene - declares the entries it does not
/// read, which a layout allows.
///
/// @param device - the device to build against
fn frame_layout(device: &Device) -> BindGroupLayout {
	let environment = env::layout_entries(ENVIRONMENT_TEXTURE, ENVIRONMENT_SAMPLER);
	let split = brdf::layout_entries(SPLIT_TEXTURE, SPLIT_SAMPLER);
	let picture = |binding| BindGroupLayoutEntry {
		binding,
		visibility: ShaderStages::FRAGMENT,
		ty: BindingType::Texture {
			sample_type: TextureSampleType::Float { filterable: true },
			view_dimension: TextureViewDimension::D2,
			multisampled: false,
		},
		count: None,
	};

	device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("globals"),
		entries: &[
			BindGroupLayoutEntry {
				binding: 0,
				visibility: ShaderStages::VERTEX_FRAGMENT,
				ty: BindingType::Buffer {
					ty: BufferBindingType::Uniform,
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			},
			picture(1),
			picture(2),
			BindGroupLayoutEntry {
				binding: 3,
				visibility: ShaderStages::FRAGMENT,
				ty: BindingType::Sampler(SamplerBindingType::Filtering),
				count: None,
			},
			environment[0],
			environment[1],
			split[0],
			split[1],
			occlusion::entry(OCCLUSION_TEXTURE),
			prepass::entry(REFLECTION_TEXTURE),
		],
	})
}

/// Which binding of group nought the environment's cube takes.
///
/// **In the frame's own group rather than in one of its own.** There is no
/// fifth group - the scene binds four and four is the floor a device has to
/// offer - and of the four this is the one every pass binds and no draw
/// changes. It also has to be this one rather than the shadow atlas's: the sky
/// is drawn by a pipeline whose layout declares group nought and nothing else,
/// and a pipeline layout that skipped a group would make the shader's own group
/// numbers resolve somewhere else. @ref [`env`](crate::env).
const ENVIRONMENT_TEXTURE: u32 = 4;

/// Which binding its sampler takes.
const ENVIRONMENT_SAMPLER: u32 = 5;

/// Which binding the split-sum table takes.
///
/// Beside the environment and for the same reason - the frame reads it and no
/// draw changes it - but it is here whether there is an environment or not: the
/// ambient reflection of a flat color goes through the same table the
/// reflection of a sky does. @ref [`brdf`](crate::brdf).
const SPLIT_TEXTURE: u32 = 6;

/// Which binding its sampler takes.
const SPLIT_SAMPLER: u32 = 7;

/// Which binding how much of the sky each pixel sees takes.
///
/// Beside the table for the table's reason, and bound in every frame for the
/// same one: a frame nobody asked for the estimate in binds one texel that says
/// all of it, so the fragment stage multiplies by one rather than branching.
/// Read with `textureLoad`, so it needs no sampler. @ref
/// [`occlusion`](crate::occlusion).
const OCCLUSION_TEXTURE: u32 = 8;

/// Which binding what each pixel's reflection found takes.
///
/// Beside the share for the share's reason, and bound in every frame for the
/// same one: a frame nobody asked for the reflections in binds one texel that
/// says nothing was found, so the fragment stage adds a nought rather than
/// branching. Read with `textureLoad`, so it needs no sampler. @ref
/// [`reflection`](crate::reflection).
const REFLECTION_TEXTURE: u32 = 9;

/// Group nought, over this frame's uniform and the atlas as it stands.
///
/// Rebuilt whenever the atlas is, because a group holds a view of one texture
/// and a rebuilt atlas is a new one.
///
/// @param device - the device to build against
/// @param layout - [`frame_layout`]
/// @param globals - the uniform
/// @param atlas - the decals' pictures
/// @param sampler - what reads them
fn frame_bindings(
	device: &Device,
	layout: &BindGroupLayout,
	globals: &Buffer,
	atlas: &Atlas,
	held: &Held<'_>,
) -> BindGroup {
	device.create_bind_group(&BindGroupDescriptor {
		label: Some("globals"),
		layout,
		entries: &[
			BindGroupEntry {
				binding: 0,
				resource: globals.as_entire_binding(),
			},
			BindGroupEntry {
				binding: 1,
				resource: BindingResource::TextureView(atlas.colors()),
			},
			BindGroupEntry {
				binding: 2,
				resource: BindingResource::TextureView(atlas.numbers()),
			},
			BindGroupEntry {
				binding: 3,
				resource: BindingResource::Sampler(held.decals),
			},
			BindGroupEntry {
				binding: ENVIRONMENT_TEXTURE,
				resource: BindingResource::TextureView(held.environment.view()),
			},
			BindGroupEntry {
				binding: ENVIRONMENT_SAMPLER,
				resource: BindingResource::Sampler(held.environment.sampler()),
			},
			BindGroupEntry {
				binding: SPLIT_TEXTURE,
				resource: BindingResource::TextureView(held.split.view()),
			},
			BindGroupEntry {
				binding: SPLIT_SAMPLER,
				resource: BindingResource::Sampler(held.split.sampler()),
			},
			BindGroupEntry {
				binding: OCCLUSION_TEXTURE,
				resource: BindingResource::TextureView(held.occlusion),
			},
			BindGroupEntry {
				binding: REFLECTION_TEXTURE,
				resource: BindingResource::TextureView(held.reflection),
			},
		],
	})
}

/// What group nought holds besides the uniform and the decals' atlas.
///
/// A struct rather than two more arguments, because the builder is already at
/// the count the lints allow and the two are read together.
struct Held<'a> {
	/// What the decals' pictures are read through.
	decals: &'a Sampler,

	/// The cube every reflection and a cubemap sky are read out of.
	environment: &'a Environment,

	/// The table that says how much of either one a surface sends back.
	split: &'a Split,

	/// How much of the sky each pixel sees, or one texel that says all of it.
	/// @ref [`Occlusion::bound`].
	occlusion: &'a TextureView,

	/// What each pixel's reflection found, or one texel that says nothing was.
	/// @ref [`Reflection::bound`].
	reflection: &'a TextureView,
}

/// The sampler every decal's picture is read through.
///
/// Clamped, and the one sampler here without anisotropy, on purpose: in an
/// atlas a sample stretched along a grazing angle reaches further than the gap
/// between two pictures, and a smear of the neighbor is worse than a little
/// blur on a decal seen edge on.
///
/// @param device - the device to build against
fn decal_sampler(device: &Device) -> Sampler {
	device.create_sampler(&SamplerDescriptor {
		label: Some("decal"),
		address_mode_u: AddressMode::ClampToEdge,
		address_mode_v: AddressMode::ClampToEdge,
		address_mode_w: AddressMode::ClampToEdge,
		mag_filter: FilterMode::Linear,
		min_filter: FilterMode::Linear,
		mipmap_filter: MipmapFilterMode::Linear,
		..SamplerDescriptor::default()
	})
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

/// The buffer every frame's [`Globals`] is written into, the uniform at group
/// nought's first entry.
///
/// @param device - the device to build against
fn globals_uniform(device: &Device) -> Result<Buffer> {
	Ok(device.create_buffer(&BufferDescriptor {
		label: Some("globals"),
		size: size_bytes::<Globals>(1)?,
		usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
		mapped_at_creation: false,
	}))
}

/// Every sampler a material can pick, in [`Wrap`]'s discriminant order.
///
/// One per wrap mode rather than one per material: a sampler is a small piece
/// of fixed-function state with two settings anybody actually wants, so the
/// table is two entries long and is built once. A material picks with an index.
///
/// @param device - the device to build against
fn wraps(device: &Device) -> [Sampler; 2] {
	[build_sampler(device, Wrap::Repeat), build_sampler(device, Wrap::Clamp)]
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
///
/// The mesh's own indices first and every coarser level's after them, in one
/// buffer, so a level is a run of it and the draw that picks one binds nothing
/// new. @ref [`GpuMesh::run`].
fn upload_mesh(device: &Device, queue: &Queue, data: &MeshData, revision: u32) -> GpuMesh {
	let mut indices = data.indices.clone();
	let mut levels = Vec::with_capacity(data.levels.len());

	for level in &data.levels {
		let first = u32::try_from(indices.len()).unwrap_or(u32::MAX);

		indices.extend_from_slice(&level.indices);
		levels.push((first..u32::try_from(indices.len()).unwrap_or(u32::MAX), level.error));
	}

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
			bytemuck::cast_slice(&indices),
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
		// and nothing for a mesh nobody painted, for the skin's reason: the
		// plain buffer stands in for it, and one buffer of white a mesh would be
		// the very cost the block was kept out of the vertex to avoid
		paint: data.is_painted().then(|| {
			create_buffer(
				device,
				queue,
				"mesh paint",
				bytemuck::cast_slice(&data.paint),
				BufferUsages::VERTEX,
			)
		}),
		vertex_count: data.vertices.len(),
		index_count: u32::try_from(data.indices.len()).unwrap_or(0),
		revision,
		errors: levels.iter().map(|(_, error)| *error).collect(),
		levels,
		bounds: Bounds::of(data),
		bones: cull::bones(data),
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
		| Texel::Rgba16Float => TextureFormat::Rgba16Float,
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
/// Begins one shadow pass over one layer of the atlas, cleared.
///
/// @param encoder - what to record into
/// @param layer - the layer to draw into
/// @param marks - the span this pass carries an end of, if any
fn shadow_pass<'pass>(
	encoder: &'pass mut CommandEncoder,
	layer: &'pass TextureView,
	marks: Option<RenderPassTimestampWrites<'pass>>,
) -> RenderPass<'pass> {
	encoder.begin_render_pass(&RenderPassDescriptor {
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
	})
}

/// A count of texels as a float, for a viewport.
fn texels(count: u32) -> f32 { f32::from(u16::try_from(count).unwrap_or(0)) }

const fn cascade(slice: usize) -> Ends {
	match (slice, CASCADES) {
		| (0, 1) => Ends::Both,
		| (0, _) => Ends::Open,
		| (at, all) if at + 1 == all => Ends::Close,
		| _ => Ends::Middle,
	}
}

/// What the depth view asks of this frame, if anything.
///
/// How far away white is, and the two numbers of the frame's own projection a
/// stored depth is turned back into a distance with. A distance that is not
/// one, nought or less or not a number at all, is the picture.
///
/// @param world - for the console variable and the aspect
/// @param camera - the camera this frame is drawn from
fn seeing_of(world: &World, camera: &Camera) -> Option<(f32, [f32; 2])> {
	let lens = camera.projection(world.aspect);

	world
		.cvars
		.float(depth::VIEW)
		.filter(|white| white.is_finite() && *white > 0.0)
		.map(|white| (white, [lens.z_axis.z, lens.w_axis.z]))
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

/// The buffer every placement a frame draws is written into: room for every
/// entity in every list a frame draws it in.
///
/// @param device - the device to build against
fn placements(device: &Device) -> Result<Buffer> {
	Ok(device.create_buffer(&BufferDescriptor {
		label: Some("placements"),
		size: size_bytes::<Placement>(MAX_ENTITIES * LISTS)?,
		// read as storage too, by the copy of what the test for what is behind
		// something nearer kept. @ref [`cover`].
		usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
		mapped_at_creation: false,
	}))
}

/// What the test for what is behind something nearer starts from: nothing
/// built, and the lists empty.
///
/// @param gpu - the device, and whether its adapter can run a compute pass and
/// draw from an indirect command at all
/// @param size - the picture's size
fn covering(gpu: &Gpu, size: (u32, u32)) -> Result<Covering> {
	let able = gpu
		.adapter()
		.get_downlevel_capabilities()
		.flags
		.contains(DownlevelFlags::COMPUTE_SHADERS | DownlevelFlags::INDIRECT_EXECUTION);

	Ok(Covering {
		cover: Cover::new(gpu.device(), able, size_bytes::<Placement>(1)?, size)?,
		asked: false,
		reached: Vec::with_capacity(MAX_ENTITIES),
		reaches: Vec::with_capacity(MAX_ENTITIES),
		commands: Vec::new(),
		projection: Mat4::IDENTITY,
	})
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

/// Which cascades can see a box, as one bit each, the nearest lowest.
///
/// @param cascades - each cascade's box, nearest first
/// @param placed - the entity's box, in the world
fn casts_into(cascades: &[Frustum; CASCADES], placed: &Placed) -> u32 {
	volumes(cascades.iter(), placed, 0)
}

/// Which of a frame's local maps can see a box, as bits past the cascades'.
///
/// @param sight - what this frame can see, for its faces and how many are real
/// @param placed - the box, in the world
fn throws_into(sight: &Sight, placed: &Placed) -> u32 {
	volumes(sight.faces.iter().take(sight.maps), placed, CASCADES)
}

/// Which of a run of volumes hold a box, as bits from a given place.
fn volumes<'a>(over: impl Iterator<Item = &'a Frustum>, placed: &Placed, from: usize) -> u32 {
	over.enumerate().fold(0, |mask, (index, volume)| {
		if volume.holds(placed) {
			mask | (1 << (index + from))
		} else {
			mask
		}
	})
}

/// How many instances a list of batches draws.
fn instances(batches: &[Batch]) -> usize {
	batches
		.iter()
		.map(|batch| usize::try_from(batch.count).unwrap_or(0))
		.sum()
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
/// **And only among the lamps whose reach touches the view.** A lamp whose
/// whole sphere is outside the frustum cannot light a pixel of the picture,
/// since nothing further from it than its range is lit by it at all, so it is
/// not carried: a room lit from behind the camera no longer spends the budget
/// on lights nobody sees. The test is the sphere and not the lamp's position,
/// which is what keeps a lamp just off the edge of the screen: its reach
/// crosses the edge, and what it lights on this side of it is lit. When a
/// frame is measured to be spending real time in this loop the answer is
/// still a light grid rather than a better sort.
///
/// @param world - the world being drawn
/// @param eye - where the camera is
/// @param view - what the picture can see, or `None` to carry lamps wherever
/// they are
/// @param room - how many the frame may carry
/// @param scratch - the caller's list, so this allocates nothing per frame
/// @return the array the uniform holds, and how many of it is real
fn chosen(
	world: &World,
	eye: Vec3,
	view: Option<&Frustum>,
	room: usize,
	scratch: &mut Vec<Shining>,
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

		// a hidden lamp is dark, and so is one under something hidden: a lamp
		// is part of the picture, and the picture is what asks
		if !world.entities.shown(id) {
			continue;
		}

		let Some(at) = world.render_transform(id) else {
			continue;
		};

		if view.is_some_and(|view| !view.holds_ball(at.position, light.range)) {
			continue;
		}

		scratch.push(Shining {
			near: (at.position - eye).length() - light.range,
			lamp: Lamp::of(light, at),
			light,
			at,
		});
	}

	// `total_cmp` rather than a partial compare: a lamp at a nan distance is a
	// world that has blown up, and it should sort somewhere definite rather
	// than making the order depend on which pairs were compared.
	scratch.sort_by(|one, other| one.near.total_cmp(&other.near));
	scratch.truncate(room.min(MAX_LAMPS));

	let mut lamps = [Lamp::DARK; MAX_LAMPS];
	let mut count = 0;

	for (slot, shining) in lamps.iter_mut().zip(scratch.iter()) {
		*slot = shining.lamp;
		count += 1;
	}

	(lamps, u32::try_from(count).unwrap_or(0))
}

/// Hands the frame's atlas tiles to the lamps that asked for one, nearest
/// first, and works out where each of their maps looks from.
///
/// **The order is the one [`chosen`] already sorted**, which is the same
/// quantity the field measures for this: one engine scales a map by the reach
/// over the distance to the eye, another by how much of the screen the light
/// covers, and the nearest edge of a lamp's reach is that answer already
/// computed. Nothing new is measured here.
///
/// **A lamp that does not fit is skipped rather than stopping the walk.** Six
/// tiles is a point and one is a cone, so a cone behind a point that filled
/// the atlas still gets its map - which is strictly better than the far ones
/// simply going dark in order.
///
/// @param shining - the lamps the frame carries, nearest first
/// @param lamps - their packed forms, written with where their maps landed
/// @param room - how many tiles the console left the frame
/// @param tiles - written with where each map sits in the atlas
/// @param views - written with world space into each map's clip space
/// @return how many tiles were handed out
fn allocate(
	shining: &[Shining],
	lamps: &mut [Lamp; MAX_LAMPS],
	room: usize,
	tiles: &mut [Tile; LOCAL_TILES],
	views: &mut [[[f32; 4]; 4]; LOCAL_TILES],
) -> usize {
	// every tile is put back to something finite first: a view left over from
	// last frame would be read as a volume by the culler, and a zero matrix
	// would be read as nothing at all.
	*tiles = [Tile::local(0); LOCAL_TILES];
	*views = [Mat4::IDENTITY.to_cols_array_2d(); LOCAL_TILES];

	let mut slots = Slots::with_room(room);

	for (index, lit) in shining.iter().enumerate() {
		let wanted = match lit.light.kind {
			| LightKind::Point => 6,
			| LightKind::Spot => 1,
			| LightKind::None => continue,
		};

		if !lit.light.shadow {
			continue;
		}

		let (Some(first), Some(lamp)) = (slots.take(wanted), lamps.get_mut(index)) else {
			continue;
		};

		let spread = if wanted == 6 {
			shadow::point_spread()
		} else {
			shadow::cone_spread(lit.light.cone().1)
		};

		lamp.shadow = [whole(first), whole(wanted), spread, 0.0];

		for face in 0..wanted {
			let matrix = if wanted == 6 {
				shadow::faces(lit.at.position, lit.light.range)[face]
			} else {
				shadow::cone(
					lit.at.position,
					lit.at.rotation * Vec3::NEG_Z,
					lit.light.range,
					lit.light.cone().1,
				)
			};

			let at = first + face;
			if let (Some(tile), Some(view)) = (tiles.get_mut(at), views.get_mut(at)) {
				*tile = Tile::local(at);
				*view = matrix.to_cols_array_2d();
			}
		}
	}

	slots.used()
}

/// A small whole number as a float, the way the tiles are written.
fn whole(value: usize) -> f32 { f32::from(u8::try_from(value).unwrap_or(0)) }

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
///
/// **It is also why the environment is in group nought.** A cubemap sky is
/// drawn out of the environment, so this pipeline has to be able to read it;
/// putting it in a later group would mean either declaring that group here -
/// which needs the ones before it, which is the paragraph above - or leaving a
/// hole, which makes the shader's own group numbers resolve to a different
/// group entirely. That one is not a validation error at all: the sky comes
/// back one flat color and nothing anywhere says why.
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

	let buffers = vertex_buffers(skinned);

	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some(label_of(blend, skinned)),
		layout: Some(&layout),
		vertex: VertexState {
			module: &shader,
			entry_point: Some(if skinned { "vertex_skinned" } else { "vertex_main" }),
			compilation_options: PipelineCompilationOptions::default(),
			buffers: &buffers,
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

/// Where the paint is bound, in every pipeline that draws a batch.
pub(crate) const PAINT_SLOT: u32 = 2;

/// Where the skin is bound, in the pipelines that read bones.
pub(crate) const SKIN_SLOT: u32 = 3;

/// The vertex buffers a pipeline that draws a batch reads: the geometry, the
/// placements and the paint always, and the skin for a mesh bones move.
///
/// The skin only where it is read. Declaring it on the static pipelines would
/// mean binding one for every crate in the world, and there is nothing to bind.
/// **The paint is declared everywhere**, because a mesh nobody painted still
/// has something to bind - the scene's [`Plain`] buffer - and a shader that
/// reads none of its attributes, as a shadow's plain entry point does, fetches
/// none of them. Shared by the scene's table, the shadows' and the pass before
/// the scene, which all draw the same batches out of the same buffers and have
/// to agree about how.
///
/// @param skinned - whether the pipeline reads bones
pub(crate) fn vertex_buffers(skinned: bool) -> Vec<Option<VertexBufferLayout<'static>>> {
	let (vertex_stride, instance_stride) = strides();
	let mut buffers = vec![
		Some(VertexBufferLayout {
			array_stride: vertex_stride,
			step_mode: VertexStepMode::Vertex,
			attributes: &VERTEX_ATTRIBUTES,
		}),
		Some(VertexBufferLayout {
			array_stride: instance_stride,
			step_mode: VertexStepMode::Instance,
			attributes: &INSTANCE_ATTRIBUTES,
		}),
		Some(VertexBufferLayout {
			array_stride: paint_stride(),
			step_mode: VertexStepMode::Vertex,
			attributes: &PAINT_ATTRIBUTES,
		}),
	];

	if skinned {
		buffers.push(Some(VertexBufferLayout {
			array_stride: skin_stride(),
			step_mode: VertexStepMode::Vertex,
			attributes: &SKIN_ATTRIBUTES,
		}));
	}

	buffers
}

/// Holds a pass to a rectangle of its target, by its viewport and its scissor
/// alike.
///
/// @param pass - the pass, already begun
/// @param view - the rectangle, already cut down to what lies inside the target
fn cut(pass: &mut RenderPass<'_>, view: Viewport) {
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

/// What one [`PaintVertex`] hands the vertex stage: the color it was painted,
/// normalized on the way in so the shader reads four fractions, and where it
/// samples the second set.
///
/// @note: locations after the skin's rather than between the vertex's and the
/// instance's, so that nothing that was numbered before it moved.
pub(crate) const PAINT_ATTRIBUTES: [VertexAttribute; 2] = [
	VertexAttribute {
		format: VertexFormat::Unorm16x4,
		offset: 0,
		shader_location: 14,
	},
	VertexAttribute {
		format: VertexFormat::Float32x2,
		offset: 8,
		shader_location: 15,
	},
];

/// The stride of the paint buffer, asserted against the attributes above.
pub(crate) const fn paint_stride() -> BufferAddress {
	const {
		assert!(
			size_of::<PaintVertex>() == 16,
			"PaintVertex is no longer four shorts and two floats"
		);
		assert!(align_of::<PaintVertex>() == 4, "PaintVertex gained padding");
	}

	16
}

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
		assert!(size_of::<Tile>() == 32, "a Tile is no longer two vec4s");
		assert!(size_of::<Lamp>() == 64, "a Lamp is no longer four vec4s");
		// a uniform array's stride is its element rounded up to sixteen, so an
		// element that is already a multiple of it is laid out here exactly as
		// the shader reads it - which is the whole reason a lamp is four
		// vectors rather than a struct of named floats.
		assert!(size_of::<Lamp>().is_multiple_of(16), "and a uniform array's stride is not it");
		assert!(
			size_of::<Globals>()
				== 576
					+ size_of::<Tile>() * CASCADES
					+ (size_of::<Tile>() + 64) * LOCAL_TILES
					+ size_of::<Lamp>() * MAX_LAMPS
					+ size_of::<Paint>() * MAX_DECALS,
			"the camera, the cascades and their tiles, the lamps' tiles and views, the sky and 			 the lamps"
		);
		assert!(size_of::<Globals>().is_multiple_of(16), "and a uniform struct has to be");
		assert!(size_of::<Paint>() == 112, "a decal is no longer seven vec4s");
		assert!(
			size_of::<Paint>().is_multiple_of(16),
			"and its array's stride in a uniform would not be it"
		);
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
	fn the_slots_the_draws_bind_are_the_places_of_the_buffers_every_pipeline_declares() {
		let paint = usize::try_from(PAINT_SLOT).expect("a small number");
		let skin = usize::try_from(SKIN_SLOT).expect("a small number");
		let still = vertex_buffers(false);
		let bent = vertex_buffers(true);

		assert_eq!(still.len(), paint + 1, "a static pipeline ends with the paint");
		assert_eq!(bent.len(), skin + 1, "and a skinned one with the skin after it");

		for (name, buffers) in [("static", &still), ("skinned", &bent)] {
			assert!(
				buffers[paint]
					.as_ref()
					.is_some_and(|layout| layout.attributes == PAINT_ATTRIBUTES),
				"the {name} pipeline reads the paint where the draws bind it"
			);
		}

		assert!(
			bent[skin]
				.as_ref()
				.is_some_and(|layout| layout.attributes == SKIN_ATTRIBUTES),
			"and the skinned one the skin where the draws bind that"
		);
	}

	#[test]
	fn a_lamp_the_eye_is_standing_inside_comes_before_a_nearer_one_it_is_not() {
		// the small one is four units away and the big one is ten, but the big
		// one's sphere reaches the camera and the small one's does not. The
		// picture is inside the big one, so it is the one that matters.
		let world = lit_world(&[(4.0, 1.0), (10.0, 20.0)]);
		let mut scratch = Vec::new();
		let (lamps, count) = chosen(&world, Vec3::ZERO, None, MAX_LAMPS, &mut scratch);

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
		let (lamps, count) = chosen(&world, Vec3::ZERO, None, MAX_LAMPS, &mut scratch);

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
			let (lamps, count) = chosen(&world, Vec3::ZERO, None, room, &mut scratch);

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
		let (_, count) = chosen(&world, Vec3::ZERO, None, MAX_LAMPS, &mut scratch);

		assert_eq!(count, 1, "one lamp among four entities");
	}

	#[test]
	fn a_hidden_lamp_is_not_carried_and_nor_is_one_under_something_hidden() {
		// three lamps: one hidden by its own word, one hung off something
		// hidden, and the one in the middle that nothing hides
		let mut world = lit_world(&[(2.0, 5.0), (4.0, 5.0), (6.0, 5.0)]);
		let lamps: Vec<EntityId> = world.entities.iter().map(|(id, ..)| id).collect();
		let group = world.entities.spawn();
		assert!(world.entities.set_hidden(lamps[0], true));
		assert!(world.entities.set_parent(lamps[2], group));
		assert!(world.entities.set_hidden(group, true));

		let mut scratch = Vec::new();
		let (carried, count) = chosen(&world, Vec3::ZERO, None, MAX_LAMPS, &mut scratch);

		assert_eq!(count, 1, "one lamp of the three is lit");
		assert!(
			(carried[0].position_range[0] - 4.0).abs() < 1.0e-6,
			"and it is the one nothing hides: {:?}",
			carried[0].position_range
		);
	}

	/// A world with a lamp of each range standing at each point.
	fn lamps_at(lamps: &[(Vec3, f32)]) -> World {
		let mut world = World::new();

		for &(at, range) in lamps {
			let id = world.entities.spawn_at(Transform::at(at));

			world
				.entities
				.set_light(id, Light::point(Vec3::ONE, 1.0, range));
		}

		world
	}

	/// What a camera at the origin looking down `-z` can see, square.
	fn ahead() -> Frustum {
		let camera = Camera {
			position: Vec3::ZERO,
			target: Vec3::NEG_Z,
			..Camera::DEFAULT
		};

		Frustum::of(camera.view_projection(1.0))
	}

	#[test]
	fn a_lamp_whose_reach_misses_the_view_gives_its_slot_to_one_that_reaches_it() {
		// one three units behind the eye with a reach of one, which stops two
		// short of the near plane, and one ten ahead: with room for one, the
		// nearer wins while nothing asks what the picture can see, and the one
		// the picture can see wins once something does
		let world =
			lamps_at(&[(Vec3::new(0.0, 0.0, 3.0), 1.0), (Vec3::new(0.0, 0.0, -10.0), 1.0)]);
		let mut scratch = Vec::new();

		let (blind, _) = chosen(&world, Vec3::ZERO, None, 1, &mut scratch);
		let (seeing, count) = chosen(&world, Vec3::ZERO, Some(&ahead()), 1, &mut scratch);

		assert!(
			(blind[0].position_range[2] - 3.0).abs() < 1.0e-6,
			"unasked, the nearer is carried: {:?}",
			blind[0].position_range
		);
		assert_eq!(count, 1, "one is carried either way");
		assert!(
			(seeing[0].position_range[2] + 10.0).abs() < 1.0e-6,
			"asked, the one ahead is: {:?}",
			seeing[0].position_range
		);
	}

	#[test]
	fn a_lamp_off_the_edge_is_carried_while_its_reach_crosses_the_edge() {
		// ten units ahead, the right-hand edge of a square view a radian across
		// is 5.46 out. A lamp at eight lights across it with a reach of four and
		// does not with a reach of one. **The case testing the lamp's position
		// would get wrong**: its middle is outside the view either way.
		let mut scratch = Vec::new();
		let at = Vec3::new(8.0, 0.0, -10.0);
		let view = ahead();

		let (_, reaching) =
			chosen(&lamps_at(&[(at, 4.0)]), Vec3::ZERO, Some(&view), MAX_LAMPS, &mut scratch);
		let (_, short) =
			chosen(&lamps_at(&[(at, 1.0)]), Vec3::ZERO, Some(&view), MAX_LAMPS, &mut scratch);

		assert_eq!(reaching, 1, "a reach across the edge is carried");
		assert_eq!(short, 0, "and one that stops short of it is not");
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
			size_bytes::<Placement>(MAX_ENTITIES * LISTS).expect("the size fits"),
			128 * 21 * BufferAddress::try_from(MAX_ENTITIES).expect("the count fits"),
			"and twenty-one of them for every one: the picture's list, each cascade's, and one 			 per tile of the shadow atlas"
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
