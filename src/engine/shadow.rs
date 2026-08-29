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
//! **And the whole grid is then snapped to whole texels.** Even with a sphere,
//! the light's box slides continuously as the camera walks, so a shadow edge
//! shimmers between one texel and the next. Rounding the position of a fixed
//! world point onto the texel lattice pins the lattice to the world, and the
//! shimmer stops. It is ten lines and it is the difference between shadows that
//! look finished and shadows that look broken.

use colby_core::{
	Result,
	abi::Camera,
	bytemuck, err,
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
	FrontFace, MultisampleState, PipelineCompilationOptions, PipelineLayoutDescriptor,
	PolygonMode, PrimitiveState, PrimitiveTopology, Queue, RenderPipeline,
	RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor,
	ShaderSource, ShaderStages, StencilState, TextureAspect, TextureDescriptor, TextureDimension,
	TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
	VertexBufferLayout, VertexState, VertexStepMode,
};

use crate::scene::{DEPTH_FORMAT, INSTANCE_ATTRIBUTES, VERTEX_ATTRIBUTES, strides};

/// How many slices the shadow distance is cut into.
///
/// Four is what the resolution is worth: at [`RESOLUTION`] each map costs four
/// megabytes, so the set is sixteen, which a machine running twenty pixel tests
/// side by side can afford and sixty-four is not.
pub const CASCADES: usize = 4;

/// How many texels one cascade's map is on a side.
pub const RESOLUTION: u32 = 1024;

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

/// The depth array, the depth-only pipeline, and the groups both ends bind.
pub(crate) struct Maps {
	/// One view per cascade, each one layer of the array, drawn into.
	layers: Vec<TextureView>,

	/// The matrix each pass reads, one [`SLOT`] per cascade.
	uniforms: Buffer,

	/// One group per cascade, over that cascade's own slot.
	slots: Vec<BindGroup>,

	/// What the scene binds to read every map at once.
	sampled: BindGroup,

	/// Kept so the scene's pipeline can be built again when its shader is.
	sample_layout: BindGroupLayout,

	pipeline: RenderPipeline,
}

impl Maps {
	/// Builds the array, the pipeline and every group.
	///
	/// @param device - the device to build against
	/// @return the maps, or the compiler's complaint about the depth shader
	pub(crate) fn new(device: &Device) -> Result<Self> {
		let cascade_layout = cascade_layout(device);
		let sample_layout = sample_layout(device);
		let texture = device.create_texture(&TextureDescriptor {
			label: Some("shadow maps"),
			size: Extent3d {
				width: RESOLUTION,
				height: RESOLUTION,
				depth_or_array_layers: u32::try_from(CASCADES).unwrap_or(1),
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: DEPTH_FORMAT,
			usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
			view_formats: &[],
		});

		let layers = (0..CASCADES)
			.map(|slice| {
				texture.create_view(&TextureViewDescriptor {
					label: Some("shadow cascade"),
					dimension: Some(TextureViewDimension::D2),
					base_array_layer: u32::try_from(slice).unwrap_or(0),
					array_layer_count: Some(1),
					aspect: TextureAspect::DepthOnly,
					..TextureViewDescriptor::default()
				})
			})
			.collect();

		let map = texture.create_view(&TextureViewDescriptor {
			label: Some("shadow maps"),
			dimension: Some(TextureViewDimension::D2Array),
			aspect: TextureAspect::DepthOnly,
			..TextureViewDescriptor::default()
		});

		// clamped, because a sample that fell off the edge of a cascade should
		// read that edge rather than wrap around to the far side of the world.
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

		let sampled = device.create_bind_group(&BindGroupDescriptor {
			label: Some("shadow maps"),
			layout: &sample_layout,
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
		});

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
		let pipeline = build_pipeline(device, &cascade_layout);

		if let Some(complaint) = pollster::block_on(scope.pop()) {
			return Err(err!(Graphics("the shadow pipeline: {complaint}")));
		}

		Ok(Self {
			layers,
			uniforms,
			slots,
			sampled,
			sample_layout,
			pipeline,
		})
	}

	/// The layout of the group the scene samples through.
	pub(crate) const fn sample_layout(&self) -> &BindGroupLayout { &self.sample_layout }

	/// The group the scene binds to read every map.
	pub(crate) const fn bindings(&self) -> &BindGroup { &self.sampled }

	/// The depth-only pipeline every cascade's pass runs.
	pub(crate) const fn pipeline(&self) -> &RenderPipeline { &self.pipeline }

	/// One cascade's layer, to draw into.
	pub(crate) fn layer(&self, slice: usize) -> Option<&TextureView> { self.layers.get(slice) }

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

/// Builds the depth-only pipeline.
///
/// No fragment stage and no color target, over the same two vertex buffers the
/// scene draws from: the shader reads the position and the model matrix and
/// lets the pipeline supply the rest.
///
/// @param device - the device to build against
/// @param layout - the group holding one cascade's matrix
fn build_pipeline(device: &Device, layout: &BindGroupLayout) -> RenderPipeline {
	let shader = device.create_shader_module(ShaderModuleDescriptor {
		label: Some("shadow"),
		source: ShaderSource::Wgsl(include_str!("shadow.wgsl").into()),
	});

	let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some("shadow"),
		bind_group_layouts: &[Some(layout)],
		immediate_size: 0,
	});

	let (vertex_stride, instance_stride) = strides();

	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some("shadow"),
		layout: Some(&pipeline_layout),
		vertex: VertexState {
			module: &shader,
			entry_point: Some("vertex_main"),
			compilation_options: PipelineCompilationOptions::default(),
			buffers: &[
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
			],
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
		fragment: None,
		multiview_mask: None,
		cache: None,
	})
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
