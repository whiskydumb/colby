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
		camera::rh::{
			proj::directx::{orthographic, perspective},
			view::look_at_mat4,
		},
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
	TextureViewDescriptor, TextureViewDimension, VertexState,
};

use crate::scene::{DEPTH_FORMAT, vertex_buffers};

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

/// The console variable that says how many atlas tiles the lamps may have.
///
/// A ceiling on the ceiling, the way [`LAMPS`](crate::scene::LAMPS) is one on
/// how many lamps a frame carries at all. Nought turns every local shadow off
/// without turning a lamp off, which is what a picture measured against the
/// build before this one is shot with.
pub const LOCAL_LAMPS: &str = "r.shadow_lamps";

/// What [`LOCAL_LAMPS`] holds until somebody sets it: the whole atlas.
pub const DEFAULT_LOCAL_LAMPS: f32 = 16.0;

const _: () = {
	assert!(LOCAL_TILES == 16, "LOCAL_TILES and DEFAULT_LOCAL_LAMPS disagree");
};

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

/// The widest a cone's shadow may be drawn, in radians.
///
/// A cone may open to nearly a hemisphere - `Light::MAX_CONE` is 88.9 degrees
/// of half-angle - and
/// a perspective projection at a hundred and seventy-eight degrees across
/// spends nearly all of its texels on the four corners of a square that the
/// cone does not reach. A hundred and fifty is where that stops being worth
/// drawing; past it the shadow is drawn at this width and the edge of a wider
/// cone falls outside its own map, which reads as lit. A limit, and written
/// down as one.
const CONE_CEILING: f32 = 2.617_993_9;

/// How near a local light's map starts, as a fraction of how far it reaches.
///
/// A hundredth, held at a centimeter, which puts a lamp of the usual few units
/// of reach at about a twentieth. Every texel of depth precision behind the
/// near plane is spent on nothing, and a light with something touching it is
/// the case that decides the number.
const LOCAL_NEAR: f32 = 0.01;

/// The six views a point light's shadow is drawn from.
///
/// **Six faces rather than two.** Four of the five engines read draw a point
/// light's shadow as six square views - three as a cube texture and one as six
/// rectangles of an atlas, which is this. The fifth warps the sphere onto two
/// paraboloids, and the reason not to follow it is that a paraboloid is not a
/// projection: it is right at every vertex and wrong everywhere between them,
/// so a wall crossing one bulges unless it is cut into pieces small enough to
/// hide it. The engine that does it compensates in its vertex stage; six flat
/// views need no compensating at all.
///
/// The axis order is `+x -x +y -y +z -z`, which the shader picks a face out of
/// by the largest component of the direction. The up vectors are colby's own:
/// nothing outside this file and its reader sees them, so the only rule they
/// have to keep is not lying along their own axis.
///
/// @param position - where the lamp stands
/// @param range - how far it reaches, which is the far plane
/// @return one matrix per face, world space into that face's clip space
#[must_use]
pub fn faces(position: Vec3, range: f32) -> [Mat4; 6] {
	let axes = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];
	let ups = [Vec3::Y, Vec3::Y, Vec3::Z, Vec3::NEG_Z, Vec3::Y, Vec3::Y];
	let projection = local_projection(std::f32::consts::FRAC_PI_2, range);

	std::array::from_fn(|face| {
		let (axis, up) = (axes[face], ups[face]);

		projection * look_at_mat4(position, position + axis, up)
	})
}

/// The one view a cone's shadow is drawn from.
///
/// A single perspective as wide as the cone is, so the cone is the circle
/// inscribed in the square the map covers: everything the cone lights at all is
/// inside it, and the corners it wastes are the corners the falloff has already
/// taken to nothing.
///
/// @param position - where the lamp stands
/// @param direction - the way it points, of any length
/// @param range - how far it reaches
/// @param outer - the half-angle of its edge, in radians
/// @return world space into the cone's clip space
#[must_use]
pub fn cone(position: Vec3, direction: Vec3, range: f32, outer: f32) -> Mat4 {
	let along = direction.normalize_or(Vec3::NEG_Z);
	let up = if along.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };

	local_projection(cone_fov(outer), range) * look_at_mat4(position, position + along, up)
}

/// How wide a cone's map is drawn, held inside what a projection can do.
fn cone_fov(outer: f32) -> f32 { (outer * 2.0).clamp(0.01, CONE_CEILING) }

/// How much of the world one texel of a local map covers, per unit of distance.
///
/// What the shader multiplies by the distance to the lamp to get the size of
/// the texel a point landed in, which is what its normal offset is measured in
/// which is the same quantity [`Cascades::texels`] holds for a cascade, except
/// that a perspective map's texel grows with distance and so cannot be one
/// number.
///
/// @param fov - how wide the view is, in radians: a right angle for a face of
/// a point light, the cone's own width for a cone
#[must_use]
pub fn spread(fov: f32) -> f32 { 2.0 * (fov * 0.5).tan() / LOCAL_F }

/// [`spread`] for one face of a point light.
#[must_use]
pub fn point_spread() -> f32 { spread(std::f32::consts::FRAC_PI_2) }

/// [`spread`] for a cone of a given edge.
#[must_use]
pub fn cone_spread(outer: f32) -> f32 { spread(cone_fov(outer)) }

/// The projection every local map is drawn through.
fn local_projection(fov: f32, range: f32) -> Mat4 {
	let far = range.max(0.02);
	let near = (far * LOCAL_NEAR).clamp(0.01, far * 0.5);

	perspective(fov, 1.0, near, far)
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

/// How many maps the atlas can be drawn into, and so how many slots the
/// uniform holds: one per cascade and one per local tile.
const SLOTS: usize = CASCADES + LOCAL_TILES;

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

	/// One per (a local map or a cascade, the picture is sampled, bones move
	/// it).
	///
	/// Three axes and eight entries. The last two are the same shape the
	/// scene's table has - keyed on a *bool* rather than on `Blend`, because
	/// what a shadow pass wants to know is only whether the surface can have
	/// holes in it, and a mode that does not cast has no row here rather than
	/// an unused one.
	///
	/// **The first axis exists for one reason: the depth bias.** A cascade is
	/// drawn with a slope-scaled bias in the pipeline; a local map is drawn
	/// with none at all, and leans entirely on the normal offset the scene's
	/// shader applies. That is a deliberate choice and its reason is not
	/// graphical - a slope-scaled bias on a floating-point depth is scaled by
	/// a value the specification leaves to the implementation, so a second
	/// answer worked out away from the device cannot predict it, and a shadow
	/// nobody can predict is a shadow nobody can check.
	pipelines: [RenderPipeline; 8],
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
			label: Some("shadow views"),
			size: SLOT * u64::try_from(SLOTS).unwrap_or(1),
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let slots = (0..SLOTS)
			.map(|slot| {
				device.create_bind_group(&BindGroupDescriptor {
					label: Some("shadow view"),
					layout: &cascade_layout,
					entries: &[BindGroupEntry {
						binding: 0,
						resource: BindingResource::Buffer(BufferBinding {
							buffer: &uniforms,
							offset: SLOT * u64::try_from(slot).unwrap_or(0),
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
		let pipelines = std::array::from_fn(|index| {
			build_pipeline(device, &groups, Wanted {
				local: index >= 4,
				masked: index % 4 >= 2,
				skinned: index % 2 == 1,
			})
		});

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

	/// The pipeline a shadow pass runs for one batch.
	///
	/// @param local - whether this is a lamp's map rather than a cascade
	/// @param masked - whether the surface's picture has to be sampled before
	/// its depth is allowed to be written
	/// @param skinned - whether bones move the geometry
	pub(crate) fn casting(&self, local: bool, masked: bool, skinned: bool) -> &RenderPipeline {
		&self.pipelines[usize::from(local) * 4 + usize::from(masked) * 2 + usize::from(skinned)]
	}

	/// One layer of the atlas, to draw into.
	pub(crate) fn layer(&self, layer: usize) -> Option<&TextureView> { self.layers.get(layer) }

	/// One map's group, holding the matrix its pass draws through.
	///
	/// The cascades take the first [`CASCADES`] and the local tiles the rest,
	/// in the order [`Slots`] handed them out.
	pub(crate) fn slot(&self, slot: usize) -> Option<&BindGroup> { self.slots.get(slot) }

	/// Writes this frame's cascade matrices, one per slot.
	pub(crate) fn upload(&self, queue: &Queue, cascades: &Cascades) {
		for (slice, matrix) in cascades.matrices.iter().enumerate() {
			self.write(queue, slice, *matrix);
		}
	}

	/// Writes one local map's matrix, by its tile.
	///
	/// @param queue - what to write through
	/// @param tile - which local tile, as [`Slots`] handed it out
	/// @param matrix - world space into that map's clip space
	pub(crate) fn upload_local(&self, queue: &Queue, tile: usize, matrix: Mat4) {
		self.write(queue, CASCADES.saturating_add(tile), matrix);
	}

	/// Writes one matrix into one slot of the uniform.
	fn write(&self, queue: &Queue, slot: usize, matrix: Mat4) {
		if slot >= SLOTS {
			return;
		}

		let at = SLOT * u64::try_from(slot).unwrap_or(0);
		queue.write_buffer(&self.uniforms, at, bytemuck::bytes_of(&matrix.to_cols_array()));
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

/// Which of the eight depth pipelines to build.
///
/// A struct rather than three flags, for the reason [`Groups`] is one: it is
/// what the caller varies, and three bools in a row at a call site say nothing
/// about which is which.
#[derive(Clone, Copy, Debug)]
struct Wanted {
	/// Whether it draws a lamp's map, which is the variant with no bias.
	local: bool,

	/// Whether the surface's picture has to be sampled before its depth is
	/// allowed to be written.
	masked: bool,

	/// Whether bones move the geometry.
	skinned: bool,
}

/// Builds one depth pipeline.
///
/// Over the same two vertex buffers the scene draws from: the shader reads the
/// position, the model matrix and - where the surface can have holes in it -
/// the texture coordinate, and lets the pipeline supply the rest.
///
/// @param device - the device to build against
/// @param groups - the bind group layouts, in group order
/// @param wanted - which of the eight
fn build_pipeline(device: &Device, groups: &Groups<'_>, wanted: Wanted) -> RenderPipeline {
	let Wanted { local, masked, skinned } = wanted;

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

	let buffers = vertex_buffers(skinned);

	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some(label_of(wanted)),
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
			buffers: &buffers,
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
			//
			// **And a local map takes none of it**, for that same sentence
			// read the other way: what nobody can predict, no second answer
			// can check. A lamp's map leans on the normal offset alone, which
			// is arithmetic anybody can repeat.
			bias: if local {
				DepthBiasState::default()
			} else {
				DepthBiasState {
					constant: 4,
					slope_scale: 2.5,
					clamp: 0.0,
				}
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

/// What one of the eight is called in a graphics debugger.
const fn label_of(wanted: Wanted) -> &'static str {
	match (wanted.local, wanted.masked, wanted.skinned) {
		| (false, false, false) => "shadow",
		| (false, false, true) => "shadow skinned",
		| (false, true, false) => "shadow masked",
		| (false, true, true) => "shadow masked skinned",
		| (true, false, false) => "lamp shadow",
		| (true, false, true) => "lamp shadow skinned",
		| (true, true, false) => "lamp shadow masked",
		| (true, true, true) => "lamp shadow masked skinned",
	}
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			Light, MeshId, Post, Renderable, Sky, ToneMap, Transform, Value, World,
			material::{Blend, Material},
		},
		glam::{Quat, Vec4Swizzles},
	};

	use super::*;
	use crate::{
		Capture, Image,
		capture::rgb,
		scene::{LAMPS, MSAA},
	};

	/// How big every picture here is.
	const PICTURE: (u32, u32) = (240, 180);

	/// A capture on the binary's one device, or `None` with no GPU.
	fn capture() -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, PICTURE.0, PICTURE.1) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// A shut room with one lamp in it, looked at from inside.
	///
	/// The sun travels straight *up*, so every surface the camera can see has
	/// the sun's whole term multiplied by nought and the lamp is the only
	/// thing lighting the picture. No curve and no eye, for the reason every
	/// pixel test here has: what is measured is a difference between two
	/// pictures, and an eye that adapted to one of them would move all of it.
	fn room() -> World {
		let mut world = World::new();

		world.post = Post {
			tonemap: ToneMap::None,
			auto_exposure: false,
			exposure: 1.0,
			..Post::DEFAULT
		};
		world.ambient = Vec3::splat(0.02);
		world.sky = Sky::NONE;
		world.clear = Vec3::ZERO;
		world.aspect = f32::from(u16::try_from(PICTURE.0).unwrap_or(u16::MAX))
			/ f32::from(u16::try_from(PICTURE.1).unwrap_or(u16::MAX));
		// high enough that the near floor - which is where a shadow thrown
		// towards the camera lands - fills the bottom of the frame
		world.camera.position = Vec3::new(0.0, 4.0, 6.0);
		world.camera.target = Vec3::new(0.0, 0.0, -2.0);
		// traveling up, so nothing the camera sees is lit by it
		world.light = Vec3::Y;
		world.cvars.var(MSAA, Value::Float(1.0), "");
		world.cvars.var(LAMPS, Value::Float(32.0), "");
		world
			.cvars
			.var(LOCAL_LAMPS, Value::Float(DEFAULT_LOCAL_LAMPS), "");

		// the floor
		block(&mut world, Vec3::new(0.0, -0.5, 0.0), Vec3::new(24.0, 1.0, 24.0));
		// the back of the room, which the lamp lights and the pillar shades
		block(&mut world, Vec3::new(0.0, 3.0, -7.0), Vec3::new(24.0, 6.0, 1.0));

		world
	}

	/// A grey box in a world.
	fn block(world: &mut World, position: Vec3, scale: Vec3) -> colby_core::abi::EntityId {
		let id = world.entities.spawn_at(Transform {
			position,
			rotation: Quat::IDENTITY,
			scale,
		});

		world
			.entities
			.set_renderable(id, Renderable::new(MeshId::CUBE, rgb(0.6, 0.6, 0.6)));

		id
	}

	/// A lamp standing somewhere, throwing a shadow or not.
	fn lamp(world: &mut World, position: Vec3, light: Light) -> colby_core::abi::EntityId {
		let id = world.entities.spawn_at(Transform {
			position,
			rotation: Quat::IDENTITY,
			scale: Vec3::ONE,
		});

		world.entities.set_light(id, light);

		id
	}

	/// A point lamp of the brightness these tests use, throwing a shadow.
	fn bright() -> Light { Light::point(Vec3::ONE, 12.0, 14.0) }

	/// The same lamp told not to throw one.
	fn quiet() -> Light { Light { shadow: false, ..bright() } }

	/// How bright one pixel of a picture is, in the red channel.
	fn level(picture: &Image, column: u32, row: u32) -> i32 {
		i32::from(picture.pixel(column, row)[0])
	}

	/// The picture a world draws.
	fn shot(capture: &mut Capture, world: &mut World) -> Image {
		capture.shoot(world).expect("the capture renders")
	}

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
	fn the_six_faces_of_a_point_cover_every_direction_once() {
		let (at, reach) = (Vec3::new(1.0, 2.0, -3.0), 10.0);
		let views = faces(at, reach);
		let mut used = [0_u32; 6];

		// a spiral rather than a grid, so no sample lands on an axis where two
		// faces are equally good and the pick is a tie
		for step in 0..600 {
			let turn = f32::from(u16::try_from(step).expect("six hundred"));
			let around = turn * 2.399_963_2;
			let up = 1.0 - (turn + 0.5) / 300.0;
			let radius = (1.0 - up * up).max(0.0).sqrt();
			let way = Vec3::new(radius * around.cos(), up, radius * around.sin());
			let point = at + way * (reach * 0.5);

			// the shader's rule, written out again
			let size = way.abs();
			let face = if size.x >= size.y && size.x >= size.z {
				usize::from(way.x <= 0.0)
			} else if size.y >= size.z {
				2 + usize::from(way.y <= 0.0)
			} else {
				4 + usize::from(way.z <= 0.0)
			};

			let clip = views[face] * point.extend(1.0);
			let ndc = clip.xyz() / clip.w;

			assert!(clip.w > 0.0, "{way} is behind the face the rule picked");
			assert!(
				ndc.truncate().abs().max_element() <= 1.0 + 1.0e-4,
				"{way} falls outside face {face} at {ndc}"
			);
			assert!((0.0..=1.0).contains(&ndc.z), "and outside its depth range at {}", ndc.z);
			used[face] += 1;
		}

		assert!(used.iter().all(|count| *count > 50), "some face was never used: {used:?}");
	}

	#[test]
	fn a_cone_holds_what_it_lights_and_is_held_back_when_it_opens_too_wide() {
		let (at, reach, outer) = (Vec3::new(0.0, 4.0, 0.0), 12.0, 0.6_f32);
		let view = cone(at, Vec3::NEG_Y, reach, outer);

		// the edge of the cone lands on the edge of the map, which is what
		// makes the map exactly as wide as the light and no wider
		let edge = at + Vec3::new(outer.tan() * 5.0, -5.0, 0.0);
		let clip = view * edge.extend(1.0);
		let ndc = clip.xyz() / clip.w;

		assert!((ndc.x.abs() - 1.0).abs() < 1.0e-3, "the cone's edge is at {}", ndc.x);

		let inside = at + Vec3::new(outer.tan() * 2.5, -5.0, 0.0);
		let clip = view * inside.extend(1.0);

		assert!(
			(clip.xyz() / clip.w)
				.truncate()
				.abs()
				.max_element()
				< 1.0,
			"and everything the cone lights is inside it"
		);

		// a cone may open to nearly a hemisphere and a projection may not
		assert!(
			(cone_fov(1.55) - CONE_CEILING).abs() < 1.0e-6,
			"a cone past the ceiling is drawn at the ceiling"
		);
		assert!((cone_fov(0.4) - 0.8).abs() < 1.0e-6, "and a narrow one at twice its own edge");
	}

	#[test]
	fn a_local_map_starts_a_hundredth_of_the_way_in_and_ends_at_the_reach() {
		for reach in [0.5_f32, 4.0, 16.0, 200.0] {
			let view = cone(Vec3::ZERO, Vec3::NEG_Z, reach, 0.5);
			let near = (reach * LOCAL_NEAR).clamp(0.01, reach * 0.5);

			let at_near = view * Vec3::new(0.0, 0.0, -near).extend(1.0);
			let at_far = view * Vec3::new(0.0, 0.0, -reach).extend(1.0);

			assert!(
				(at_near.z / at_near.w).abs() < 1.0e-4,
				"a reach of {reach} does not start at nought"
			);
			assert!((at_far.z / at_far.w - 1.0).abs() < 1.0e-4, "and does not end at one");
		}
	}

	#[test]
	fn the_offset_a_sample_is_pushed_by_is_a_texel_of_its_own_map() {
		// two units of world for every unit of distance across a right angle,
		// divided by the map's side - so a point ten away sits in a texel this
		// wide, and the shader pushes by two to four of them.
		let spread = point_spread();

		assert!(spread.mul_add(10.0, -(20.0 / LOCAL_F)).abs() < 1.0e-7, "a face's texel is not");
		assert!(cone_spread(0.5) < spread, "a cone narrower than a right angle has a finer one");
		assert!(cone_spread(1.2) > spread, "and a wider one a coarser");
	}

	#[test]
	fn the_atlas_runs_out_and_a_cone_behind_a_point_still_gets_a_tile() {
		// six tiles a point and one a cone, sixteen in all: two points fill
		// twelve, a third does not fit, and the rule is that the walk carries
		// on rather than stopping - so what is behind it is still drawn.
		let mut slots = Slots::with_room(LOCAL_TILES);
		let given: Vec<Option<usize>> = [6, 6, 6, 1, 1, 1, 1, 1]
			.into_iter()
			.map(|wanted| slots.take(wanted))
			.collect();

		assert_eq!(
			given,
			vec![Some(0), Some(6), None, Some(12), Some(13), Some(14), Some(15), None],
			"two points, then whatever fits behind them"
		);
	}

	#[test]
	fn a_lamp_behind_a_wall_does_not_light_what_the_wall_hides() {
		// the whole card in one measurement: two places on the floor the same
		// distance from the lamp, one of them with a pillar in the way. Told
		// not to cast, they are lit alike; told to, they are not.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room();

		// the lamp behind the pillar, so what it hides is the near floor and
		// the near floor is the bottom of the picture
		block(&mut world, Vec3::new(0.0, 1.0, -1.0), Vec3::new(1.2, 2.0, 0.6));
		let id = lamp(&mut world, Vec3::new(0.0, 2.4, -4.0), quiet());

		let (shaded, beside) = (PICTURE.0 / 2, PICTURE.0 / 8);
		let row = PICTURE.1 * 3 / 4;

		let plain = shot(&mut capture, &mut world);
		let (was_shaded, was_beside) = (level(&plain, shaded, row), level(&plain, beside, row));

		world.entities.set_light(id, bright());

		let cast = shot(&mut capture, &mut world);
		let (now_shaded, now_beside) = (level(&cast, shaded, row), level(&cast, beside, row));

		assert!(
			now_shaded < was_shaded - 20,
			"what the pillar hides goes dark: {was_shaded} to {now_shaded}"
		);
		assert!(
			(now_beside - was_beside).abs() <= 1,
			"and what it does not hide is left alone: {was_beside} to {now_beside}"
		);
	}

	#[test]
	fn a_lamp_told_not_to_cast_draws_what_the_console_drew_with_no_atlas() {
		// two ways of saying no, and they have to be the same picture: the
		// field on the light, and the ceiling on how many tiles a frame may
		// hand out. The second is what a screenshot from the build before this
		// card is compared against.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room();

		block(&mut world, Vec3::new(0.0, 1.0, -1.0), Vec3::new(1.2, 2.0, 0.6));
		let id = lamp(&mut world, Vec3::new(0.0, 2.4, -4.0), quiet());

		let by_field = shot(&mut capture, &mut world);

		world.entities.set_light(id, bright());
		world.cvars.set(LOCAL_LAMPS, "0");

		let by_console = shot(&mut capture, &mut world);

		assert!(
			by_field.pixels == by_console.pixels,
			"a light that will not cast and an atlas with no room are one picture"
		);

		world.cvars.set(LOCAL_LAMPS, "16");
		let casting = shot(&mut capture, &mut world);

		assert!(by_field.pixels != casting.pixels, "and giving it room is a different one");
	}

	#[test]
	fn a_cone_casts_from_the_one_face_it_has() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room();

		block(&mut world, Vec3::new(0.0, 1.0, -1.0), Vec3::new(1.2, 2.0, 0.6));

		// where the point lamp stood, pointing back at the camera and down, so
		// the pillar stands between it and the near floor
		let at = Vec3::new(0.0, 2.8, -4.0);
		let id = world.entities.spawn_at(Transform {
			position: at,
			rotation: Quat::from_rotation_arc(Vec3::NEG_Z, Vec3::new(0.0, -0.3, 1.0).normalize()),
			scale: Vec3::ONE,
		});
		let shape = Light::spot(Vec3::ONE, 60.0, 16.0, 0.15, 0.7);
		world
			.entities
			.set_light(id, Light { shadow: false, ..shape });

		let plain = shot(&mut capture, &mut world);

		world.entities.set_light(id, shape);

		let cast = shot(&mut capture, &mut world);

		assert!(plain.pixels != cast.pixels, "a cone that casts draws a different picture");

		let (middle, row) = (PICTURE.0 / 2, PICTURE.1 * 3 / 4);

		assert!(
			level(&cast, middle, row) < level(&plain, middle, row) - 10,
			"and what the pillar hides from it is darker"
		);
	}

	#[test]
	fn a_hidden_lamp_and_a_blended_caster_are_both_left_out() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room();

		let pillar = block(&mut world, Vec3::new(0.0, 1.0, -1.0), Vec3::new(1.2, 2.0, 0.6));
		let id = lamp(&mut world, Vec3::new(0.0, 2.4, -4.0), bright());

		let casting = shot(&mut capture, &mut world);

		world.entities.set_hidden(id, true);

		assert!(
			shot(&mut capture, &mut world).pixels != casting.pixels,
			"a hidden lamp lights nothing at all"
		);

		world.entities.set_hidden(id, false);
		// a surface what is behind shows through writes no depth, so it says
		// nothing about what a light can reach - the same rule the cascades
		// have, and it is which list `consider` filed it in rather than a test
		// of its own
		let glass = world.materials.insert("test/glass", Material {
			blend: Blend::Alpha,
			opacity: 0.5,
			..Material::DEFAULT
		});
		world
			.entities
			.set_renderable(pillar, Renderable::of(MeshId::CUBE, glass, rgb(0.6, 0.6, 0.6)));

		// **one pixel rather than the whole picture**, because a glass pillar
		// changes every pixel it covers whether or not it casts. The place
		// read is the floor the pillar hid a moment ago: with the pillar solid
		// it is dark, and with it glass it has to read the same as it does
		// with the atlas taken away altogether.
		let (probe, row) = (PICTURE.0 / 2, PICTURE.1 * 3 / 4);
		let through = shot(&mut capture, &mut world);

		world.cvars.set(LOCAL_LAMPS, "0");
		let nothing = shot(&mut capture, &mut world);

		assert!(
			level(&casting, probe, row) < level(&through, probe, row) - 20,
			"a solid pillar throws a shadow there"
		);
		assert!(
			(level(&through, probe, row) - level(&nothing, probe, row)).abs() <= 1,
			"and a glass one throws none: {} against {}",
			level(&through, probe, row),
			level(&nothing, probe, row)
		);
	}

	#[test]
	fn a_lamp_shadows_sideways_as_well_as_ahead() {
		// **the test that says the shader picks the right face of the six.**
		// Everything else here is shadowed through the face pointing away from
		// the camera, so a shader that read the same face whichever way a
		// point lay would pass all of it. These two places lie to either side
		// of the lamp, the same distance out, one of them behind a pillar -
		// and the face each reads is the one pointing its own way.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room();

		let at = Vec3::new(0.0, 2.4, -4.0);
		block(&mut world, at + Vec3::new(2.0, -1.4, 0.0), Vec3::new(1.2, 2.0, 0.6));
		let id = lamp(&mut world, at, quiet());

		// where the floor four units back and three and a half out lands, out
		// of the same projection the camera has. The right-hand one is behind
		// the pillar and the left-hand one is not.
		let (blocked, clear, row) = (173, 66, 76);

		let plain = shot(&mut capture, &mut world);
		let (was_blocked, was_clear) = (level(&plain, blocked, row), level(&plain, clear, row));

		assert!(
			(was_blocked - was_clear).abs() < 6,
			"the two sides are lit alike before anything casts: {was_blocked} and {was_clear}"
		);

		world.entities.set_light(id, bright());

		let cast = shot(&mut capture, &mut world);
		let (now_blocked, now_clear) = (level(&cast, blocked, row), level(&cast, clear, row));

		assert!(
			now_blocked < was_blocked - 20,
			"the side with the pillar goes dark: {was_blocked} to {now_blocked}"
		);
		assert!(
			(now_clear - was_clear).abs() <= 1,
			"and the side without it does not: {was_clear} to {now_clear}"
		);
	}

	#[test]
	fn the_lamps_cost_one_pass_however_many_of_them_there_are() {
		// the whole point of an atlas: sixteen maps in one layer, and a layer
		// is a pass. A map each would have been sixteen.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = room();

		block(&mut world, Vec3::new(0.0, 1.0, -1.0), Vec3::new(1.2, 2.0, 0.6));

		let mut frame = |world: &mut World| {
			capture.draw(world, &mut []);

			capture.scene_mut().spans().passes()
		};

		let none = frame(&mut world);
		let first = lamp(&mut world, Vec3::new(0.0, 2.4, -4.0), bright());
		let casting = frame(&mut world);

		assert_eq!(casting, none + 1, "every lamp's map is one pass over one layer");

		let second = lamp(&mut world, Vec3::new(-2.0, 2.4, -3.0), bright());

		assert_eq!(frame(&mut world), casting, "and a second lamp is still that one pass");

		world.entities.set_light(first, quiet());
		world.entities.set_light(second, quiet());

		assert_eq!(frame(&mut world), none, "and no lamp asking is no pass at all");
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
