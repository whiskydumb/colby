//! Where the light looks from, and the maps it writes.
//!
//! Two halves. The first is arithmetic over a [`Camera`] and a direction with
//! no device in it, which is deliberate: everything that goes wrong with
//! cascaded shadows goes wrong there rather than in the pipeline, and a wrong
//! matrix is a picture nobody can read. Kept apart, it is a unit test instead.
//! The second is [`Maps`], which owns the depth array, the depth-only pipeline
//! and the bind groups, the way [`Lines`](crate::lines) owns the debug
//! renderer's.
//!
//! **Four slices, cut logarithmically and linearly at once.** A purely
//! logarithmic cut puts three of the four cascades inside the first meter,
//! because the camera's near plane is a tenth of a unit; a purely linear one
//! spends most of its resolution on the distance nobody looks at. The blend
//! between them is the usual answer and [`LAMBDA`] is how much of each.
//!
//! **Each slice is enclosed by a sphere rather than by a box in light space**,
//! and the reason is that a sphere does not care which way the camera is
//! pointed. A box fitted to the slice's eight corners changes size as the
//! camera turns, which makes every shadow edge in the world crawl while it
//! turns. The sphere depends only on the near and far distances, the field of
//! view and the aspect - none of which move when the camera does.
//!
//! **The maps all live in one atlas**, a depth array of [`LAYERS`] layers a
//! thousand and twenty-four texels square. A cascade takes a whole layer and a
//! local light's map takes a [`LOCAL`]-texel [`Tile`] of the layer past them,
//! sixteen to the layer. One texture is one binding, one sampler and one
//! budget; and because a whole-layer tile's arithmetic is the identity, the
//! cascades moved into it without a pixel of any picture moving.
//!
//! **And the whole grid is then snapped to whole texels.** Even with a sphere,
//! the light's box slides continuously as the camera walks, so a shadow edge
//! shimmers between one texel and the next. Rounding the position of a fixed
//! world point onto the texel lattice pins the lattice to the world, and the
//! shimmer stops. It is ten lines and it is the difference between shadows that
//! look finished and shadows that look broken.

use colby_core::{
	Result,
	abi::Camera,
	bytemuck::{self, Pod, Zeroable},
	err,
	glam::{
		Mat4, Vec3,
		camera::rh::{proj::directx::orthographic, view::look_at_mat4},
	},
};
use wgpu::{
	AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
	BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
	BufferBinding, BufferBindingType, BufferDescriptor, BufferUsages, CompareFunction,
	DepthBiasState, DepthStencilState, Device, ErrorFilter, Extent3d, Face, FilterMode,
	FragmentState, FrontFace, MultisampleState, PipelineCompilationOptions,
	PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, Queue,
	RenderPipeline, RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor,
	ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState, Texture, TextureAspect,
	TextureDescriptor, TextureDimension, TextureSampleType, TextureUsages, TextureView,
	TextureViewDescriptor, TextureViewDimension, VertexBufferLayout, VertexState, VertexStepMode,
};

use crate::scene::{
	DEPTH_FORMAT, INSTANCE_ATTRIBUTES, SKIN_ATTRIBUTES, VERTEX_ATTRIBUTES, skin_stride, strides,
};

/// How many slices the shadow distance is cut into.
///
/// Four is what the resolution is worth: at [`RESOLUTION`] each map costs four
/// megabytes, so the set is sixteen and the whole atlas twenty, which a machine
/// running twenty pixel tests side by side can afford and sixty-four is not.
pub const CASCADES: usize = 4;

/// How many texels one layer of the atlas is on a side.
///
/// A cascade takes a whole layer, so this is also a cascade's own resolution
/// and the name it kept from before there was an atlas.
pub const RESOLUTION: u32 = 1024;

/// How many texels one local light's map is on a side.
///
/// A quarter of a layer's side, so sixteen of them tile one layer. The number
/// is the field's for exactly this map: the engine nearest colby in shape
/// keeps a 2D map at a thousand and twenty-four and a cube face at two hundred
/// and fifty-six, and the one with the largest renderer halves its object
/// resolution for a cube before quantizing it further, saying out loud that a
/// cube costs a lot of memory. Six faces here are a megabyte and a half, a
/// tenth of what the sun already spends.
pub const LOCAL: u32 = 256;

/// How many local tiles fit across one layer.
const LOCAL_ROW: usize = 4;

/// How many local maps the atlas holds at once.
///
/// One layer of them, [`LOCAL_ROW`] squared. A cone takes one and a point
/// takes six, so the frame can carry two points and four cones, or sixteen
/// cones, and what does not fit throws no shadow. @ref [`Slots`].
pub const LOCAL_TILES: usize = 16;

/// How many layers the atlas has: one per cascade, and one of local tiles.
pub const LAYERS: usize = CASCADES + 1;

/// [`LOCAL`] as a float, for the tile arithmetic below.
const LOCAL_F: f32 = 256.0;

const _: () = {
	assert!(RESOLUTION == LOCAL * 4, "a layer no longer holds four local tiles across");
	assert!(
		LOCAL_TILES == LOCAL_ROW * LOCAL_ROW,
		"and LOCAL_TILES is no longer that squared"
	);
	assert!(LOCAL == 256, "LOCAL and LOCAL_F disagree");
};

/// How far from the camera anything is shadowed at all, in world units.
///
/// Not the camera's far plane, which is two hundred: a shadow map stretched
/// over that would be four texels to the unit even in the last cascade. Fifty
/// is more than a scene at this scale ever shows.
pub const DEFAULT_DISTANCE: f32 = 50.0;

/// The narrowest and widest shadow distance the console will accept.
pub const DISTANCE_RANGE: (f32, f32) = (1.0, 500.0);

/// The console variable that turns every cascade off.
///
/// On by default, unlike the physics drawings: this is a feature rather than a
/// tool, so what it is for is being on. `--shot` has no console and therefore
/// takes the default, which is what puts shadows in a screenshot.
pub const ENABLED: &str = "r.shadows";

/// The console variable that says how far out anything is shadowed.
pub const DISTANCE: &str = "r.shadow_distance";

/// The console variable that colors every pixel by the cascade it read.
///
/// The one tool the cascades need: which slice a surface fell into decides its
/// resolution, and there is no other way to see where the cuts landed.
pub const TINT: &str = "r.shadow_cascades";

/// How much of the split is logarithmic rather than linear.
///
/// Zero is even slices and one is even *ratios*. Seven tenths is the usual
/// answer and it puts the first cut about four units out, which is roughly
/// where a player's own feet stop being the interesting thing.
const LAMBDA: f32 = 0.7;

/// How far behind a slice the light still looks for casters, in world units.
///
/// A wall outside the view that shadows something inside it is the whole reason
/// this is not simply the slice's own depth range. Fifty is generous for a
/// scene this size and costs nothing but depth precision, of which a
/// thirty-two-bit float has plenty.
const CASTER_REACH: f32 = 50.0;

/// [`RESOLUTION`] as a float, so that nothing here has to cast.
const RESOLUTION_F: f32 = 1024.0;

/// [`CASCADES`] as a float, for the same reason.
const CASCADES_F: f32 = 4.0;

const _: () = {
	assert!(RESOLUTION == 1024, "RESOLUTION and RESOLUTION_F disagree");
	assert!(CASCADES == 4, "CASCADES and CASCADES_F disagree");
};

/// [`RESOLUTION`] as a float, for anything that has to divide by it.
///
/// @note: how far a sample is pushed along its own normal before it is looked
/// up - the other number that decides whether a lit surface stripes itself - is
/// **not** here. It lives in `shader.wgsl`, where it is used, because nothing
/// on this side can read a constant out of a shader and two copies of a tuning
/// value are worse than one in the wrong crate.
#[must_use]
pub const fn resolution() -> f32 { RESOLUTION_F }

/// One fitted set of cascades, ready to be handed to the GPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cascades {
	/// World space into each cascade's clip space, nearest slice first.
	pub matrices: [Mat4; CASCADES],

	/// The view depth each slice stops at, in world units.
	///
	/// A fragment picks its cascade by comparing against these in order, which
	/// is the same quantity the slices were cut on - so a point inside slice
	/// `i` is guaranteed to be inside the sphere fitted around slice `i`.
	pub splits: [f32; CASCADES],

	/// How many world units one texel of each cascade covers.
	///
	/// Read by the shader, which pushes a sample along its own normal by about
	/// this much before looking it up. That is what stops a lit surface
	/// shadowing itself where it is nearly edge on to the light.
	pub texels: [f32; CASCADES],
}

impl Cascades {
	/// Cascades that shadow nothing, for a world with no light to speak of.
	pub const NONE: Self = Self {
		matrices: [Mat4::ZERO; CASCADES],
		splits: [0.0; CASCADES],
		texels: [0.0; CASCADES],
	};
}

/// Where one shadow map sits in the atlas.
///
/// Two vectors, the way a lamp is three: a fragment reading a map has to know
/// where its rectangle starts, how much of a layer it covers, which layer it
/// is on, and how far a tap may wander before it leaves the rectangle and
/// begins reading somebody else's map.
///
/// **The last of those is the one thing an atlas has to say that a stack of
/// separate maps does not**, and it is why the bounds are carried rather than
/// worked out where they are read: a tile that is a whole layer is bounded by
/// the layer itself, which is exactly what the sampler's clamp already does,
/// and a tile inside a layer has to stop half a texel short of its own edge
/// instead. The two rules are different, so the answer is settled here.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Tile {
	/// The rectangle a tap is held inside, as `[min u, min v, max u, max v]`.
	pub bounds: [f32; 4],

	/// `[origin u, origin v, how much of a layer's side it covers, its layer]`.
	pub place: [f32; 4],
}

impl Tile {
	/// The whole of one layer, which is what a cascade takes.
	///
	/// **Its arithmetic is the identity, and that is not a coincidence.** The
	/// scale is one, the origin is nought and the bounds are the layer's own,
	/// so a lookup that used to be `at + offset` still is - exactly, in
	/// floating point, whether or not the compiler folds the multiply and the
	/// add together - and clamping to nought and one before a sampler that
	/// clamps to the edge changes nothing. That is what let the cascades move
	/// into an atlas without a pixel of any picture moving with them.
	///
	/// @param layer - which layer, nearest cascade first
	#[must_use]
	pub fn layer(layer: usize) -> Self {
		Self {
			bounds: [0.0, 0.0, 1.0, 1.0],
			place: [0.0, 0.0, 1.0, whole(layer)],
		}
	}

	/// One local map's tile, by its place in the local layer.
	///
	/// Sixteen of them, four across, on the layer past the last cascade.
	///
	/// @param slot - which tile, along the top row first
	#[must_use]
	pub fn local(slot: usize) -> Self {
		let across = whole(slot % LOCAL_ROW);
		let down = whole(slot / LOCAL_ROW % LOCAL_ROW);
		let side = LOCAL_F / RESOLUTION_F;
		let (origin_u, origin_v) = (across * side, down * side);

		// half a texel in from every edge. A tap that wandered past it would
		// read the neighboring map rather than the edge of its own, which is
		// the one way an atlas can be wrong that a stack of maps cannot.
		let inset = 0.5 / RESOLUTION_F;

		Self {
			bounds: [
				origin_u + inset,
				origin_v + inset,
				origin_u + side - inset,
				origin_v + side - inset,
			],
			place: [origin_u, origin_v, side, whole(CASCADES)],
		}
	}

	/// Which layer of the atlas it is on.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "written by `whole` out of a layer index, so it is a small whole number"
	)]
	pub fn layer_index(self) -> usize { self.place[3].max(0.0) as usize }

	/// The rectangle it covers in its layer, in texels: `[x, y, side]`.
	///
	/// What a pass sets its viewport and its scissor to before drawing into
	/// it. A cascade's is the whole layer, which is what a pass does anyway.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "a fraction of a layer times its side, so a texel count under RESOLUTION"
	)]
	pub fn viewport(self) -> [u32; 3] {
		let side = (self.place[2] * RESOLUTION_F).round().max(1.0);

		[
			(self.place[0] * RESOLUTION_F).round() as u32,
			(self.place[1] * RESOLUTION_F).round() as u32,
			side as u32,
		]
	}
}

/// Where each cascade's map sits in the atlas, nearest slice first.
///
/// Fixed for the life of the process - the cascades take the first [`CASCADES`]
/// layers whole - so this is a constant written as a function rather than
/// anything a frame decides.
#[must_use]
pub fn cascade_tiles() -> [Tile; CASCADES] { std::array::from_fn(Tile::layer) }

/// A small whole number as a float, without a cast anybody has to justify.
fn whole(value: usize) -> f32 { f32::from(u8::try_from(value).unwrap_or(0)) }

/// How one frame's local tiles are handed out.
///
/// A bump index and a ceiling, and that is the whole allocator - because
/// nothing here is kept between frames. Every shadow map colby draws is drawn
/// again every frame, so a lamp that lands in a different tile than it had
/// last time reads exactly the same depths out of it; the thing an atlas that
/// *caches* has to do - hold a tile to its light, and evict by least recently
/// used when it cannot - is work colby does not have to do. One engine in the
/// field caches and keeps a five-hundred-millisecond tolerance to stop tiles
/// thrashing; the one that redraws every frame repacks from nothing every
/// frame, which is this.
///
/// **A lamp that does not fit is skipped rather than stopping the walk.** The
/// lamps arrive nearest first, so the rule is mostly "the far ones go dark" -
/// but a cone wanting one tile behind a point that wanted six and did not get
/// them still gets its tile, and that is strictly better than stopping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slots {
	/// How many tiles have been handed out.
	used: usize,

	/// How many there are to hand out at all.
	room: usize,
}

impl Slots {
	/// A frame's worth, holding at most so many tiles.
	///
	/// @param room - the ceiling, held inside what the atlas has
	#[must_use]
	pub const fn with_room(room: usize) -> Self {
		Self {
			used: 0,
			room: if room > LOCAL_TILES { LOCAL_TILES } else { room },
		}
	}

	/// Takes a run of tiles, if there is one.
	///
	/// @param wanted - how many in a row: one for a cone, six for a point
	/// @return the first one's place in the local layer, or nothing
	pub fn take(&mut self, wanted: usize) -> Option<usize> {
		let after = self.used.checked_add(wanted)?;
		if wanted == 0 || after > self.room {
			return None;
		}

		let first = self.used;
		self.used = after;

		Some(first)
	}

	/// How many tiles the frame has handed out.
	#[must_use]
	pub const fn used(self) -> usize { self.used }
}

/// Fits one set of cascades to a camera and a light.
///
/// @param camera - where the view is, and how wide
/// @param aspect - the target's width divided by its height
/// @param light - the direction the light travels, of any length
/// @param distance - how far out to shadow at all
/// @return one matrix, split and texel size per slice
#[must_use]
pub fn fit(camera: &Camera, aspect: f32, light: Vec3, distance: f32) -> Cascades {
	let near = camera.near.max(0.001);
	let far = distance
		.clamp(DISTANCE_RANGE.0, DISTANCE_RANGE.1)
		.clamp(near + 0.01, camera.far.max(near + 0.02));
	let direction = light.normalize_or(Vec3::NEG_Y);

	let mut cascades = Cascades::NONE;
	let mut start = near;

	for slice in 0..CASCADES {
		let end = split_at(near, far, slice + 1);
		let (center, radius) = enclose(camera, aspect, start, end);
		let texel = 2.0 * radius / RESOLUTION_F;
		let extent = radius + texel;

		cascades.matrices[slice] = look(center, extent, direction);
		cascades.splits[slice] = end;
		cascades.texels[slice] = 2.0 * extent / RESOLUTION_F;
		start = end;
	}

	cascades
}

/// Where one cut falls, blending an even split with an even ratio.
///
/// @param near - the camera's near plane
/// @param far - the shadow distance
/// @param index - which cut, `1` being the first and [`CASCADES`] the last
/// @return the view depth that cut sits at
fn split_at(near: f32, far: f32, index: usize) -> f32 {
	let step = f32::from(u8::try_from(index).unwrap_or(0));
	let fraction = step / CASCADES_F;

	let logarithmic = near * (far / near).powf(fraction);
	let uniform = (far - near).mul_add(fraction, near);

	(1.0 - LAMBDA).mul_add(uniform, LAMBDA * logarithmic)
}

/// The smallest sphere holding one slice of the view.
///
/// Both rings of corners lie on a circle around the line of sight, so the
/// distance from any point on that line to a whole ring is one number, and the
/// sphere is found by placing the center where the two rings are equally far.
/// When that point would sit past the far plane the far ring is what bounds the
/// slice on its own, and the center is clamped there; taking the larger of the
/// two distances afterwards is what makes the result hold either way.
///
/// @param camera - where the view is
/// @param aspect - the target's width divided by its height
/// @param near - where this slice starts, as a view depth
/// @param far - where it stops
/// @return the sphere's center in world space, and its radius
fn enclose(camera: &Camera, aspect: f32, near: f32, far: f32) -> (Vec3, f32) {
	let forward = (camera.target - camera.position).normalize_or(Vec3::NEG_Z);

	// how far a corner sits from the line of sight, per unit of depth: the
	// half-height times the diagonal of a one-by-aspect rectangle.
	let spread = (camera.fov_y.clamp(0.1, 3.0) * 0.5).tan()
		* aspect
			.max(0.001)
			.mul_add(aspect.max(0.001), 1.0)
			.sqrt();

	let squared = spread * spread;
	let depth = ((far + near) * (1.0 + squared) * 0.5).min(far);
	let radius = (far - depth)
		.hypot(far * spread)
		.max((depth - near).hypot(near * spread));

	(forward.mul_add(Vec3::splat(depth), camera.position), radius.max(0.001))
}

/// The matrix taking the world into one cascade's clip space.
///
/// @param center - the middle of the sphere this cascade covers
/// @param extent - its radius, plus a texel of slack so snapping cannot push a
/// corner out
/// @param direction - the unit direction the light travels
fn look(center: Vec3, extent: f32, direction: Vec3) -> Mat4 {
	// far enough back that something standing between the light and the slice
	// is still in front of the near plane.
	let back = extent + CASTER_REACH;

	// any axis the light is not already pointing along. A light straight down
	// is the common case and is exactly the one that breaks the usual choice.
	let up = if direction.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };

	let view = look_at_mat4(center - direction * back, center, up);
	let projection = orthographic(-extent, extent, -extent, extent, 0.0, back + extent);

	snap(projection * view)
}

/// Pins a cascade's texel grid to the world rather than to the camera.
///
/// The matrix is a fixed rotation and a fixed scale with a translation that
/// slides as the camera walks, so rounding where one fixed world point lands on
/// the texel lattice removes the part of that translation which is smaller than
/// a texel. Every shadow edge then stays on the same texels while the camera
/// moves, instead of crawling between them.
///
/// The point is the world origin, which is arbitrary and does not matter: any
/// point fixed in the world pins the same lattice.
///
/// @param matrix - world space into clip space, before snapping
/// @return the same matrix, translated by less than one texel
fn snap(matrix: Mat4) -> Mat4 {
	let half = RESOLUTION_F * 0.5;
	let landed = matrix.project_point3(Vec3::ZERO) * half;
	if !landed.is_finite() {
		return matrix;
	}

	let offset = (landed.round() - landed) / half;

	let mut snapped = matrix;
	snapped.w_axis.x += offset.x;
	snapped.w_axis.y += offset.y;

	snapped
}

/// How far the uniform slots of one buffer have to be apart.
///
/// wgpu's floor for a uniform binding's offset, and every backend's. One matrix
/// is sixty-four bytes and the rest of each slot is nothing anybody reads.
const SLOT: u64 = 256;

/// The atlas, the depth-only pipeline, and the groups both ends bind.
pub(crate) struct Maps {
	/// One view per layer of the atlas, drawn into.
	layers: Vec<TextureView>,

	/// The matrix each pass reads, one [`SLOT`] per cascade.
	uniforms: Buffer,

	/// One group per cascade, over that cascade's own slot.
	slots: Vec<BindGroup>,

	/// What the scene binds to read every map at once.
	sampled: BindGroup,

	/// Kept so the scene's pipeline can be built again when its shader is.
	sample_layout: BindGroupLayout,

	/// One per (whether the picture has to be sampled, whether bones move it).
	///
	/// Two axes and four entries, the same shape the scene's table has - but
	/// keyed on a *bool* rather than on `Blend`, because what a cascade wants
	/// to know is only whether the surface can have holes in it. A mode that
	/// does not cast at all has no row here rather than an unused one.
	pipelines: [RenderPipeline; 4],
}

impl Maps {
	/// Builds the atlas, all four pipelines and every group.
	///
	/// @param device - the device to build against
	/// @param joints - the layout of the frame's joint matrices, which the
	/// skinned pipelines read as their second group
	/// @param material - the scene's own material layout, declared on every
	/// pipeline here so that one bind group serves both passes
	/// @return the maps, or the compiler's complaint about the depth shader
	pub(crate) fn new(
		device: &Device,
		joints: &BindGroupLayout,
		material: &BindGroupLayout,
	) -> Result<Self> {
		let cascade_layout = cascade_layout(device);
		let sample_layout = sample_layout(device);
		let texture = device.create_texture(&TextureDescriptor {
			label: Some("shadow atlas"),
			size: Extent3d {
				width: RESOLUTION,
				height: RESOLUTION,
				depth_or_array_layers: u32::try_from(LAYERS).unwrap_or(1),
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: DEPTH_FORMAT,
			usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
			view_formats: &[],
		});

		let layers = (0..LAYERS)
			.map(|layer| {
				texture.create_view(&TextureViewDescriptor {
					label: Some("shadow layer"),
					dimension: Some(TextureViewDimension::D2),
					base_array_layer: u32::try_from(layer).unwrap_or(0),
					array_layer_count: Some(1),
					aspect: TextureAspect::DepthOnly,
					..TextureViewDescriptor::default()
				})
			})
			.collect();

		let sampled = sample_group(device, &sample_layout, &texture);

		let uniforms = device.create_buffer(&BufferDescriptor {
			label: Some("cascades"),
			size: SLOT * u64::try_from(CASCADES).unwrap_or(1),
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let slots = (0..CASCADES)
			.map(|slice| {
				device.create_bind_group(&BindGroupDescriptor {
					label: Some("cascade"),
					layout: &cascade_layout,
					entries: &[BindGroupEntry {
						binding: 0,
						resource: BindingResource::Buffer(BufferBinding {
							buffer: &uniforms,
							offset: SLOT * u64::try_from(slice).unwrap_or(0),
							size: None,
						}),
					}],
				})
			})
			.collect();

		let scope = device.push_error_scope(ErrorFilter::Validation);
		let groups = Groups {
			cascade: &cascade_layout,
			joints,
			material,
		};
		let pipelines = [
			build_pipeline(device, &groups, false, false),
			build_pipeline(device, &groups, false, true),
			build_pipeline(device, &groups, true, false),
			build_pipeline(device, &groups, true, true),
		];

		if let Some(complaint) = pollster::block_on(scope.pop()) {
			return Err(err!(Graphics("the shadow pipeline: {complaint}")));
		}

		Ok(Self {
			layers,
			uniforms,
			slots,
			sampled,
			sample_layout,
			pipelines,
		})
	}

	/// The layout of the group the scene samples through.
	pub(crate) const fn sample_layout(&self) -> &BindGroupLayout { &self.sample_layout }

	/// The group the scene binds to read every map.
	pub(crate) const fn bindings(&self) -> &BindGroup { &self.sampled }

	/// The pipeline a cascade's pass runs for one batch.
	///
	/// @param masked - whether the surface's picture has to be sampled before
	/// its depth is allowed to be written
	/// @param skinned - whether bones move the geometry
	pub(crate) fn casting(&self, masked: bool, skinned: bool) -> &RenderPipeline {
		&self.pipelines[usize::from(masked) * 2 + usize::from(skinned)]
	}

	/// One layer of the atlas, to draw into.
	pub(crate) fn layer(&self, layer: usize) -> Option<&TextureView> { self.layers.get(layer) }

	/// One cascade's group, holding its matrix.
	pub(crate) fn slot(&self, slice: usize) -> Option<&BindGroup> { self.slots.get(slice) }

	/// Writes this frame's matrices, one per slot.
	pub(crate) fn upload(&self, queue: &Queue, cascades: &Cascades) {
		for (slice, matrix) in cascades.matrices.iter().enumerate() {
			let at = SLOT * u64::try_from(slice).unwrap_or(0);
			queue.write_buffer(&self.uniforms, at, bytemuck::bytes_of(&matrix.to_cols_array()));
		}
	}
}

/// The group one depth pass reads its matrix through.
fn cascade_layout(device: &Device) -> BindGroupLayout {
	device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("cascade"),
		entries: &[BindGroupLayoutEntry {
			binding: 0,
			visibility: ShaderStages::VERTEX,
			ty: BindingType::Buffer {
				ty: BufferBindingType::Uniform,
				has_dynamic_offset: false,
				min_binding_size: None,
			},
			count: None,
		}],
	})
}

/// The group the scene samples every cascade through.
///
/// Lifted out of the constructor rather than written inline, which is the shape
/// this lint wants in renderer code: it is a view, a sampler and a group with
/// no logic in it at all.
///
/// @param device - the device to build against
/// @param layout - the layout the group is built against
/// @param maps - the atlas every map is a rectangle of
fn sample_group(device: &Device, layout: &BindGroupLayout, maps: &Texture) -> BindGroup {
	let map = maps.create_view(&TextureViewDescriptor {
		label: Some("shadow atlas"),
		dimension: Some(TextureViewDimension::D2Array),
		aspect: TextureAspect::DepthOnly,
		..TextureViewDescriptor::default()
	});

	// clamped, because a sample that fell off the edge of a cascade should read
	// that edge rather than wrap around to the far side of the world.
	let sampler = device.create_sampler(&SamplerDescriptor {
		label: Some("shadow"),
		address_mode_u: AddressMode::ClampToEdge,
		address_mode_v: AddressMode::ClampToEdge,
		address_mode_w: AddressMode::ClampToEdge,
		mag_filter: FilterMode::Linear,
		min_filter: FilterMode::Linear,
		compare: Some(CompareFunction::LessEqual),
		..SamplerDescriptor::default()
	});

	device.create_bind_group(&BindGroupDescriptor {
		label: Some("shadow maps"),
		layout,
		entries: &[
			BindGroupEntry {
				binding: 0,
				resource: BindingResource::TextureView(&map),
			},
			BindGroupEntry {
				binding: 1,
				resource: BindingResource::Sampler(&sampler),
			},
		],
	})
}

/// The group the scene reads every map through.
fn sample_layout(device: &Device) -> BindGroupLayout {
	device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("shadow maps"),
		entries: &[
			BindGroupLayoutEntry {
				binding: 0,
				visibility: ShaderStages::FRAGMENT,
				// depth rather than float, and compared rather than filtered:
				// the hardware does the "is this behind what the light saw"
				// test inside the sampler, and blends the *answers* rather than
				// the depths, which is what makes one tap already soft and an
				// average of depths meaningless.
				ty: BindingType::Texture {
					sample_type: TextureSampleType::Depth,
					view_dimension: TextureViewDimension::D2Array,
					multisampled: false,
				},
				count: None,
			},
			BindGroupLayoutEntry {
				binding: 1,
				visibility: ShaderStages::FRAGMENT,
				ty: BindingType::Sampler(SamplerBindingType::Comparison),
				count: None,
			},
		],
	})
}

/// The three group layouts every pipeline here is built against.
///
/// A struct rather than three arguments, because the pair of flags below is
/// what a caller varies and the layouts are the same every time. It is also
/// what keeps the builder under the argument count the lints allow.
struct Groups<'a> {
	/// The group holding one cascade's matrix.
	cascade: &'a BindGroupLayout,

	/// The group holding the frame's joint matrices.
	joints: &'a BindGroupLayout,

	/// The scene's own material group, which only the masked pipelines read.
	material: &'a BindGroupLayout,
}

/// Builds one depth pipeline.
///
/// Over the same two vertex buffers the scene draws from: the shader reads the
/// position, the model matrix and - where the surface can have holes in it -
/// the texture coordinate, and lets the pipeline supply the rest.
///
/// @param device - the device to build against
/// @param groups - the bind group layouts, in group order
/// @param masked - whether to build the variant that samples the picture and
/// discards, rather than the one with no fragment stage at all
/// @param skinned - whether to build the variant that reads bones
fn build_pipeline(
	device: &Device,
	groups: &Groups<'_>,
	masked: bool,
	skinned: bool,
) -> RenderPipeline {
	let shader = device.create_shader_module(ShaderModuleDescriptor {
		label: Some("shadow"),
		source: ShaderSource::Wgsl(include_str!("shadow.wgsl").into()),
	});

	// the joints and the material are declared on all four, so that one bind
	// group of each serves every pass and nothing has to be unbound between two
	// batches. A pipeline is allowed to declare a group its shader never reads.
	let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some("shadow"),
		bind_group_layouts: &[Some(groups.cascade), Some(groups.joints), Some(groups.material)],
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
	let buffers: &[Option<VertexBufferLayout<'_>>] = if skinned {
		&[Some(vertices), Some(instances), Some(skin)]
	} else {
		&[Some(vertices), Some(instances)]
	};

	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some(label_of(masked, skinned)),
		layout: Some(&pipeline_layout),
		vertex: VertexState {
			module: &shader,
			entry_point: Some(match (masked, skinned) {
				| (false, false) => "vertex_main",
				| (false, true) => "vertex_skinned",
				| (true, false) => "vertex_masked",
				| (true, true) => "vertex_masked_skinned",
			}),
			compilation_options: PipelineCompilationOptions::default(),
			buffers,
		},
		primitive: PrimitiveState {
			topology: PrimitiveTopology::TriangleList,
			strip_index_format: None,
			front_face: FrontFace::Ccw,
			// @note: the same culling the scene uses, and deliberately not the
			// front-face culling the usual trick calls for. That trick works on
			// closed meshes and fails on the one thing every scene has: a floor
			// is a single quad, and a quad whose front is culled casts no
			// shadow at all.
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
			// a slope-scaled bias, which is what stops a surface nearly edge on
			// to the light from striping itself. It is not enough on its own
			// against a thirty-two-bit float depth, where the constant term is
			// scaled by a value nobody can predict; the normal offset in the
			// scene's shader is the half that does the work.
			bias: DepthBiasState {
				constant: 4,
				slope_scale: 2.5,
				clamp: 0.0,
			},
		}),
		multisample: MultisampleState::default(),
		// no color target either way. Where there is a stage at all, the only
		// thing it can do is throw the fragment away, which is the whole point
		// of it.
		fragment: masked.then(|| FragmentState {
			module: &shader,
			entry_point: Some("fragment_masked"),
			compilation_options: PipelineCompilationOptions::default(),
			targets: &[],
		}),
		multiview_mask: None,
		cache: None,
	})
}

/// What one of the four is called in a graphics debugger.
const fn label_of(masked: bool, skinned: bool) -> &'static str {
	match (masked, skinned) {
		| (false, false) => "shadow",
		| (false, true) => "shadow skinned",
		| (true, false) => "shadow masked",
		| (true, true) => "shadow masked skinned",
	}
}

#[cfg(test)]
mod tests {
	use colby_core::glam::Vec4Swizzles;

	use super::*;

	/// A camera a test can reason about: at the origin, looking down `-z`.
	fn camera() -> Camera {
		Camera {
			position: Vec3::ZERO,
			target: Vec3::NEG_Z,
			..Camera::DEFAULT
		}
	}

	/// The eight corners of one slice of a camera's view.
	fn corners(camera: &Camera, aspect: f32, near: f32, far: f32) -> Vec<Vec3> {
		let forward = (camera.target - camera.position).normalize();
		let right = forward.cross(camera.up).normalize();
		let up = right.cross(forward);

		let mut out = Vec::with_capacity(8);
		for depth in [near, far] {
			let half_height = (camera.fov_y * 0.5).tan() * depth;
			let half_width = half_height * aspect;

			for (across, along) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
				out.push(
					camera.position
						+ forward * depth + right * (half_width * across)
						+ up * (half_height * along),
				);
			}
		}

		out
	}

	#[test]
	fn a_cascade_takes_a_whole_layer_and_its_arithmetic_is_the_identity() {
		for slice in 0..CASCADES {
			let tile = Tile::layer(slice);

			// compared by their bits, because "exactly" is the claim: an
			// origin of nearly nought and a scale of nearly one would move
			// every cascade sample by a fraction of a texel.
			let bits = |values: [f32; 4]| values.map(f32::to_bits);

			assert_eq!(
				bits(tile.bounds),
				bits([0.0, 0.0, 1.0, 1.0]),
				"slice {slice} is bounded by the layer"
			);
			assert_eq!(
				bits(tile.place),
				bits([0.0, 0.0, 1.0, whole(slice)]),
				"and starts at nought, at scale one, on its own layer"
			);
			assert_eq!(tile.layer_index(), slice, "which is the one it says");
			assert_eq!(tile.viewport(), [0, 0, RESOLUTION], "and its pass covers all of it");
		}

		// what the identity means where it is read: `origin + at * scale` has
		// to hand back `at` bit for bit, for every value a projection can
		// produce, or the atlas moved a picture.
		let tile = Tile::layer(0);
		for step in 0..=1024_u32 {
			let at = f32::from(u16::try_from(step).expect("a thousand")) / RESOLUTION_F;
			let landed = tile.place[2].mul_add(at, tile.place[0]);

			assert!(landed.to_bits() == at.to_bits(), "{at} came back as {landed}");
		}
	}

	#[test]
	fn the_local_tiles_cover_their_layer_without_touching_each_other() {
		let side = LOCAL_F / RESOLUTION_F;
		let mut seen = Vec::with_capacity(LOCAL_TILES);

		for slot in 0..LOCAL_TILES {
			let tile = Tile::local(slot);

			assert_eq!(tile.layer_index(), CASCADES, "tile {slot} is past the last cascade");
			assert!(
				(tile.place[2] - side).abs() < 1.0e-7,
				"tile {slot} covers {} of a layer rather than {side}",
				tile.place[2]
			);

			let [x, y, extent] = tile.viewport();
			assert_eq!(extent, LOCAL, "tile {slot} is not a local map wide");
			assert!(
				x + extent <= RESOLUTION && y + extent <= RESOLUTION,
				"tile {slot} hangs off"
			);

			// the bounds sit strictly inside the rectangle, by half a texel,
			// which is what stops a tap reading the map next door.
			let inset = 0.5 / RESOLUTION_F;
			assert!(
				(tile.bounds[0] - (tile.place[0] + inset)).abs() < 1.0e-7
					&& (tile.bounds[2] - (tile.place[0] + side - inset)).abs() < 1.0e-7,
				"tile {slot} is bounded at {:?} rather than half a texel inside itself",
				tile.bounds
			);

			for (other, seen_x, seen_y) in &seen {
				assert!(
					x + extent <= *seen_x
						|| *seen_x + extent <= x
						|| y + extent <= *seen_y
						|| *seen_y + extent <= y,
					"tiles {slot} and {other} overlap"
				);
			}

			seen.push((slot, x, y));
		}
	}

	#[test]
	fn the_atlas_hands_out_runs_until_it_runs_out_and_then_keeps_going() {
		let mut slots = Slots::with_room(LOCAL_TILES);

		assert_eq!(slots.take(6), Some(0), "the first point takes the first six");
		assert_eq!(slots.take(6), Some(6), "the second takes the next six");
		assert_eq!(slots.take(6), None, "the third does not fit");
		assert_eq!(slots.take(1), Some(12), "but a cone behind it still gets a tile");
		assert_eq!(slots.used(), 13, "and the count is what was handed out");

		assert_eq!(slots.take(0), None, "nobody may ask for nothing");
		assert_eq!(slots.used(), 13, "and asking for nothing takes nothing");

		let mut none = Slots::with_room(0);
		assert_eq!(none.take(1), None, "a frame told to hand out nothing hands out nothing");

		let mut over = Slots::with_room(LOCAL_TILES + 100);
		assert_eq!(over.take(LOCAL_TILES), Some(0), "a ceiling past the atlas is the atlas");
		assert_eq!(over.take(1), None, "and it stops there");
	}

	#[test]
	fn the_cuts_march_outwards_and_end_where_they_were_told_to() {
		let cascades = fit(&camera(), 16.0 / 9.0, Vec3::NEG_Y, 60.0);

		let mut previous = camera().near;
		for (slice, split) in cascades.splits.iter().enumerate() {
			assert!(*split > previous, "cut {slice} at {split} is not past {previous}");
			previous = *split;
		}

		assert!(
			(cascades.splits[CASCADES - 1] - 60.0).abs() < 1.0e-3,
			"the last one is the shadow distance, got {}",
			cascades.splits[CASCADES - 1]
		);
	}

	#[test]
	fn every_corner_of_a_slice_lands_inside_the_cascade_that_covers_it() {
		let (camera, aspect) = (camera(), 16.0 / 9.0);
		let cascades = fit(&camera, aspect, Vec3::new(-0.4, -1.0, -0.3), DEFAULT_DISTANCE);

		let mut near = camera.near;
		for slice in 0..CASCADES {
			let far = cascades.splits[slice];

			for corner in corners(&camera, aspect, near, far) {
				let landed = cascades.matrices[slice] * corner.extend(1.0);

				assert!(
					landed.xy().abs().max_element() <= 1.0,
					"corner {corner} of slice {slice} falls outside the map at {landed}"
				);
				assert!(
					(0.0..=1.0).contains(&landed.z),
					"corner {corner} of slice {slice} is outside the depth range at {}",
					landed.z
				);
			}

			near = far;
		}
	}

	#[test]
	fn a_caster_standing_between_the_light_and_the_slice_is_still_drawn() {
		let (camera, aspect) = (camera(), 16.0 / 9.0);
		let light = Vec3::NEG_Y;
		let cascades = fit(&camera, aspect, light, DEFAULT_DISTANCE);

		// straight up from the middle of the nearest slice, which is where a
		// roof would be. Nothing about the slice itself reaches this high.
		let overhead = camera.position - camera.up * 0.0 - light * 40.0 + camera.target * 2.0;
		let landed = cascades.matrices[0] * overhead.extend(1.0);

		assert!(
			landed.xy().abs().max_element() <= 1.0 && (0.0..=1.0).contains(&landed.z),
			"a caster forty units above the slice is outside the light's box at {landed}"
		);
	}

	#[test]
	fn the_grid_stays_on_the_world_while_the_camera_walks() {
		let aspect = 16.0 / 9.0;
		let light = Vec3::new(-0.3, -1.0, -0.2);

		let mut moved = camera();
		let still = fit(&camera(), aspect, light, DEFAULT_DISTANCE);

		// a nudge smaller than one texel of the nearest cascade. A grid tied to
		// the camera slides by exactly this much and a shadow edge crawls with
		// it; one tied to the world does not move at all.
		let fraction = 0.4;
		let nudge = still.texels[0] * fraction;
		moved.position.x += nudge;
		moved.target.x += nudge;

		let walked = fit(&moved, aspect, light, DEFAULT_DISTANCE);
		let landed = |matrix: Mat4| matrix.project_point3(Vec3::new(1.0, 0.0, -3.0));
		let slid = landed(walked.matrices[0]).distance(landed(still.matrices[0]));

		// the same nudge measured in clip space, which is what an unsnapped
		// grid would give back: a texel is two over the resolution there, so
		// this is that fraction of one and nothing else enters into it.
		let unsnapped = fraction * 2.0 / RESOLUTION_F;

		assert!(
			slid < unsnapped / 20.0,
			"a fixed point moved {slid} in clip space, against the {unsnapped} a grid tied to 			 the camera would have moved"
		);
	}

	#[test]
	fn a_light_pointing_straight_down_still_has_a_matrix() {
		for light in [Vec3::NEG_Y, Vec3::Y, Vec3::ZERO, Vec3::new(0.0, -1.0, 0.001)] {
			let cascades = fit(&camera(), 16.0 / 9.0, light, DEFAULT_DISTANCE);

			for (slice, matrix) in cascades.matrices.iter().enumerate() {
				assert!(
					matrix
						.to_cols_array()
						.iter()
						.all(|value| value.is_finite()),
					"slice {slice} under a light of {light} came out as {matrix}"
				);
			}
		}
	}

	#[test]
	fn a_camera_that_has_not_been_set_up_yet_does_not_produce_nonsense() {
		let broken = Camera {
			position: Vec3::ZERO,
			target: Vec3::ZERO,
			fov_y: 0.0,
			near: 0.0,
			far: 0.0,
			..Camera::DEFAULT
		};

		let cascades = fit(&broken, 0.0, Vec3::ZERO, 0.0);

		for (slice, matrix) in cascades.matrices.iter().enumerate() {
			assert!(
				matrix
					.to_cols_array()
					.iter()
					.all(|value| value.is_finite()),
				"slice {slice} of a camera looking at itself came out as {matrix}"
			);
		}
		for texel in cascades.texels {
			assert!(texel > 0.0 && texel.is_finite(), "and a texel has a size, got {texel}");
		}
	}

	#[test]
	fn a_nearer_cascade_has_smaller_texels_than_a_further_one() {
		let cascades = fit(&camera(), 16.0 / 9.0, Vec3::NEG_Y, DEFAULT_DISTANCE);

		for slice in 1..CASCADES {
			assert!(
				cascades.texels[slice] > cascades.texels[slice - 1],
				"cascade {slice} covers {} to a texel and the one before it {}",
				cascades.texels[slice],
				cascades.texels[slice - 1]
			);
		}
		assert!(
			cascades.texels[0] < 0.05,
			"and the nearest is fine enough to be worth having, got {}",
			cascades.texels[0]
		);
	}
}
