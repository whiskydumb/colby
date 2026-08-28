//! The GPU half of the debug renderer: one buffer of segments, two pipelines.
//!
//! Part of [`Scene`](crate::Scene) rather than an
//! [`Overlay`](crate::Overlay), and the reason is the depth buffer. An overlay
//! is handed a frame that has already been drawn into and nothing else, which
//! is right for an interface and wrong for this: a debug line is world-space
//! geometry seen through the scene's camera, and whether it is hidden by a wall
//! is the most useful thing it can tell you. So it draws inside the scene's own
//! pass, with the scene's depth attachment and the scene's globals - which also
//! means the camera cannot disagree between the two.
//!
//! **Two pipelines, not one.** The same descriptor built twice, once testing
//! depth and once ignoring it. The second one exists because the interesting
//! debug geometry starts *inside* something: a contact normal begins on the
//! surface that produced it, and a shape outline is flush with the shape. Both
//! would be half invisible with only the first.
//!
//! Widths are one pixel and cannot be otherwise: wgpu has no line width, and a
//! thick line is a quad built in a vertex shader, which is a different piece of
//! work than this one.

use colby_core::{
	Result,
	abi::{World, debug::Line},
	bytemuck::{self, Pod, Zeroable},
	err,
};
use wgpu::{
	BindGroup, BindGroupLayout, BlendState, Buffer, BufferAddress, BufferDescriptor,
	BufferUsages, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
	DepthStencilState, Device, ErrorFilter, Face, FragmentState, FrontFace, MultisampleState,
	PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode, PrimitiveState,
	PrimitiveTopology, Queue, RenderPass, RenderPipeline, RenderPipelineDescriptor,
	ShaderModuleDescriptor, ShaderSource, StencilState, TextureFormat, VertexAttribute,
	VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
};

use crate::scene::DEPTH_FORMAT;

/// How many vertices a fresh buffer has room for.
///
/// Two thousand segments, which is a scene's worth of collision outlines. It
/// grows from here and never shrinks, like the interface's.
const INITIAL_VERTICES: u64 = 4096;

/// One end of one segment.
///
/// Twenty-four bytes, and deliberately nothing else in it: a debug line is
/// unlit, so there is no normal, and it samples nothing, so there is no
/// texture coordinate.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct LineVertex {
	position: [f32; 3],
	color: [f32; 3],
}

/// Every debug segment this frame, and what it takes to draw them.
pub(crate) struct Lines {
	/// Draws the segments that the world may hide.
	behind: RenderPipeline,

	/// Draws the segments that ignore it.
	over: RenderPipeline,

	vertices: Buffer,
	capacity: u64,

	/// The vertex data this frame, depth-tested segments first.
	///
	/// A field rather than a local so that a frame with debug drawing on does
	/// not allocate.
	scratch: Vec<LineVertex>,

	/// How many of `scratch` belong to the first run.
	tested: u32,
}

impl Lines {
	/// Builds both pipelines and the first buffer.
	///
	/// @param device - the device to build against
	/// @param format - the color format the fragment stage writes
	/// @param globals - the scene's group 0 layout, shared rather than copied
	pub(crate) fn new(
		device: &Device,
		format: TextureFormat,
		globals: &BindGroupLayout,
	) -> Result<Self> {
		let scope = device.push_error_scope(ErrorFilter::Validation);
		let behind = build_pipeline(device, format, globals, true);
		let over = build_pipeline(device, format, globals, false);

		if let Some(complaint) = pollster::block_on(scope.pop()) {
			return Err(err!(Graphics("the debug line pipeline: {complaint}")));
		}

		Ok(Self {
			behind,
			over,
			vertices: buffer(device, INITIAL_VERTICES),
			capacity: INITIAL_VERTICES,
			scratch: Vec::new(),
			tested: 0,
		})
	}

	/// Lays this frame's segments out and writes them to the GPU.
	///
	/// Sorted into the two runs here rather than at draw time, so that each
	/// pipeline is one contiguous `draw` however the calls were interleaved.
	///
	/// @param device - the device the buffer belongs to
	/// @param queue - where to write
	/// @param world - whose `debug` table is read
	pub(crate) fn upload(&mut self, device: &Device, queue: &Queue, world: &World) {
		self.scratch.clear();
		self.tested = 0;

		for line in world
			.debug
			.lines()
			.iter()
			.filter(|line| !line.on_top)
		{
			push(&mut self.scratch, line);
		}

		self.tested = u32::try_from(self.scratch.len()).unwrap_or(0);

		for line in world
			.debug
			.lines()
			.iter()
			.filter(|line| line.on_top)
		{
			push(&mut self.scratch, line);
		}

		if self.scratch.is_empty() {
			return;
		}

		let wanted = u64::try_from(self.scratch.len()).unwrap_or(0);
		if wanted > self.capacity {
			self.capacity = wanted.next_power_of_two();
			self.vertices = buffer(device, self.capacity);
		}

		queue.write_buffer(&self.vertices, 0, bytemuck::cast_slice(&self.scratch));
	}

	/// Records both runs into a pass the scene has already drawn into.
	///
	/// Group 0 is set again rather than inherited: what the scene left bound is
	/// the same buffer, but a draw that depends on the caller's last statement
	/// is a draw that breaks when the caller is reordered.
	///
	/// @param pass - the scene's pass, with its depth attachment
	/// @param globals - the camera and the light
	pub(crate) fn draw(&self, pass: &mut RenderPass<'_>, globals: &BindGroup) {
		let total = u32::try_from(self.scratch.len()).unwrap_or(0);
		if total == 0 {
			return;
		}

		pass.set_bind_group(0, globals, &[]);
		pass.set_vertex_buffer(0, self.vertices.slice(..));

		if self.tested > 0 {
			pass.set_pipeline(&self.behind);
			pass.draw(0..self.tested, 0..1);
		}

		if total > self.tested {
			pass.set_pipeline(&self.over);
			pass.draw(self.tested..total, 0..1);
		}
	}
}

/// Appends one segment's two ends.
fn push(into: &mut Vec<LineVertex>, line: &Line) {
	let color = line.color.to_array();

	into.push(LineVertex { position: line.from.to_array(), color });
	into.push(LineVertex { position: line.to.to_array(), color });
}

/// A vertex buffer with room for this many ends.
fn buffer(device: &Device, vertices: u64) -> Buffer {
	device.create_buffer(&BufferDescriptor {
		label: Some("debug lines"),
		size: vertices.max(1) * stride(),
		usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
		mapped_at_creation: false,
	})
}

/// Builds one of the two pipelines.
///
/// @param depth - whether the world is allowed to hide what it draws. The pass
/// that ignores depth also does not *write* it, so a line drawn over everything
/// does not go on to hide the one behind it.
fn build_pipeline(
	device: &Device,
	format: TextureFormat,
	globals: &BindGroupLayout,
	depth: bool,
) -> RenderPipeline {
	let shader = device.create_shader_module(ShaderModuleDescriptor {
		label: Some("debug lines"),
		source: ShaderSource::Wgsl(include_str!("lines.wgsl").into()),
	});

	let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some("debug lines"),
		bind_group_layouts: &[Some(globals)],
		immediate_size: 0,
	});

	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some("debug lines"),
		layout: Some(&layout),
		vertex: VertexState {
			module: &shader,
			entry_point: Some("vertex_main"),
			compilation_options: PipelineCompilationOptions::default(),
			buffers: &[Some(VertexBufferLayout {
				array_stride: stride(),
				step_mode: VertexStepMode::Vertex,
				attributes: &VERTEX_ATTRIBUTES,
			})],
		},
		primitive: PrimitiveState {
			topology: PrimitiveTopology::LineList,
			strip_index_format: None,
			front_face: FrontFace::Ccw,
			// a segment has no facing, so there is no back of one to cull.
			cull_mode: None::<Face>,
			unclipped_depth: false,
			polygon_mode: PolygonMode::Fill,
			conservative: false,
		},
		depth_stencil: Some(DepthStencilState {
			format: DEPTH_FORMAT,
			depth_write_enabled: Some(depth),
			depth_compare: Some(if depth {
				CompareFunction::Less
			} else {
				CompareFunction::Always
			}),
			stencil: StencilState::default(),
			bias: DepthBiasState::default(),
		}),
		multisample: MultisampleState::default(),
		fragment: Some(FragmentState {
			module: &shader,
			entry_point: Some("fragment_main"),
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

/// What one [`LineVertex`] hands the vertex stage.
const VERTEX_ATTRIBUTES: [VertexAttribute; 2] = [
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
];

/// The vertex stride, asserted to match the attributes above.
const fn stride() -> BufferAddress {
	const {
		assert!(size_of::<LineVertex>() == 24, "LineVertex is no longer two vec3s");
		assert!(align_of::<LineVertex>() == 4, "LineVertex gained padding");
	}

	24
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::debug::{RED, WHITE},
		glam::Vec3,
	};

	use super::*;

	/// Lays a world's debug lines out the way [`Lines::upload`] does.
	fn laid_out(world: &World) -> (Vec<LineVertex>, u32) {
		let mut scratch = Vec::new();

		for line in world
			.debug
			.lines()
			.iter()
			.filter(|line| !line.on_top)
		{
			push(&mut scratch, line);
		}

		let tested = u32::try_from(scratch.len()).unwrap_or(0);

		for line in world
			.debug
			.lines()
			.iter()
			.filter(|line| line.on_top)
		{
			push(&mut scratch, line);
		}

		(scratch, tested)
	}

	#[test]
	fn each_segment_becomes_exactly_two_ends() {
		let mut world = World::new();
		world.debug.line(Vec3::ZERO, Vec3::X, WHITE);
		world.debug.line(Vec3::Y, Vec3::Z, RED);

		let (vertices, _) = laid_out(&world);

		assert_eq!(vertices.len(), 4, "two segments, two ends each");
		assert!(
			(vertices[0].position[0] - 0.0).abs() < f32::EPSILON
				&& (vertices[1].position[0] - 1.0).abs() < f32::EPSILON,
			"and the ends are the ends that were asked for, in order"
		);
	}

	#[test]
	fn the_depth_tested_run_comes_first_and_the_split_says_where_it_ends() {
		let mut world = World::new();

		// interleaved on purpose: the order they were submitted in must not
		// decide how many draw calls this costs.
		world
			.debug
			.on_top()
			.line(Vec3::ZERO, Vec3::X, RED);
		world.debug.line(Vec3::ZERO, Vec3::Y, WHITE);
		world
			.debug
			.on_top()
			.line(Vec3::ZERO, Vec3::Z, RED);

		let (vertices, tested) = laid_out(&world);

		assert_eq!(vertices.len(), 6, "three segments");
		assert_eq!(tested, 2, "one of which is hidden by the world");
		assert!(
			(vertices[1].position[1] - 1.0).abs() < f32::EPSILON,
			"and it is the one that was submitted second, moved to the front of the buffer"
		);
	}

	#[test]
	fn a_world_with_nothing_to_debug_draws_nothing() {
		let world = World::new();
		let (vertices, tested) = laid_out(&world);

		assert!(vertices.is_empty(), "no segments");
		assert_eq!(tested, 0, "and no run to draw them in");
	}
}
