//! The GPU half of the interface: one pipeline, one shader, one draw call per
//! bound texture.
//!
//! Written straight against wgpu, like the scene it draws over. What it does
//! *not* do is share the scene's textures: an image in a document is uploaded
//! again here rather than reaching into `Scene`'s table, which costs a second
//! copy of whatever the interface shows and buys the renderer keeping no
//! opinion about the interface at all. There are a handful of them, and the day
//! that is wrong the answer is a shared texture cache rather than a seam
//! between these two.
//!
//! Font atlases are a single channel of distance field, and images are the same
//! sRGB textures the scene samples. Both go through one bind group layout: a
//! layout cares that a texture is float and filterable, not what its format is.

use std::path::Path;

use colby_core::{
	Result,
	abi::{FontId, TextureId, World, texture::Texel},
	bytemuck, debug, err,
};
use colby_engine::Shader;
use wgpu::{
	AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
	BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
	Buffer, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
	CommandEncoderDescriptor, Device, ErrorFilter, Extent3d, FilterMode, FragmentState,
	FrontFace, IndexFormat, LoadOp, MultisampleState, Operations, PipelineCompilationOptions,
	PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, Queue,
	RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
	SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
	StoreOp, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
	TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
	TextureViewDescriptor, TextureViewDimension, VertexAttribute, VertexBufferLayout,
	VertexFormat, VertexState, VertexStepMode,
};

use crate::draw::{Binding, DrawList, Vertex};

/// The shader file, watched for edits like the scene's.
const SHADER: &str = "ui.wgsl";

/// How many vertices a fresh buffer has room for.
const INITIAL_VERTICES: u64 = 4096;

/// The screen uniform: the layout area and its padding to sixteen bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct Screen {
	viewport: [f32; 2],
	padding: [f32; 2],
}

/// One texture the interface has put on the GPU, and what it was made from.
struct Uploaded {
	binding: Binding,
	revision: u32,
	group: BindGroup,
}

/// Everything the interface keeps on the GPU.
pub struct Painter {
	pipeline: RenderPipeline,
	shader: Shader,
	format: TextureFormat,
	screen_layout: BindGroupLayout,
	texture_layout: BindGroupLayout,
	screen: Buffer,
	screen_group: BindGroup,
	sampler: wgpu::Sampler,
	vertices: Buffer,
	indices: Buffer,
	vertex_capacity: u64,
	index_capacity: u64,
	uploaded: Vec<Uploaded>,
	blank: BindGroup,
}

impl Painter {
	/// Builds the pipeline and the buffers.
	///
	/// @param device - the device the frames belong to
	/// @param format - the color format the target was configured with
	pub fn new(device: &Device, format: TextureFormat) -> Result<Self> {
		let shader = Shader::at(Path::new(env!("CARGO_MANIFEST_DIR")), SHADER, built_in());

		let screen_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("ui screen"),
			entries: &[BindGroupLayoutEntry {
				binding: 0,
				visibility: ShaderStages::VERTEX_FRAGMENT,
				ty: BindingType::Buffer {
					ty: wgpu::BufferBindingType::Uniform,
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			}],
		});

		let texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("ui texture"),
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
			],
		});

		let screen = device.create_buffer(&BufferDescriptor {
			label: Some("ui screen"),
			size: u64::try_from(size_of::<Screen>()).unwrap_or(16),
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let screen_group = device.create_bind_group(&BindGroupDescriptor {
			label: Some("ui screen"),
			layout: &screen_layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: screen.as_entire_binding(),
			}],
		});

		// clamped rather than repeated, unlike the scene's: a glyph sampled
		// past the edge of its cell must not wrap round to the other side of
		// the atlas, which is a stray line of some other letter along an edge.
		let sampler = device.create_sampler(&SamplerDescriptor {
			label: Some("ui"),
			address_mode_u: AddressMode::ClampToEdge,
			address_mode_v: AddressMode::ClampToEdge,
			address_mode_w: AddressMode::ClampToEdge,
			mag_filter: FilterMode::Linear,
			min_filter: FilterMode::Linear,
			..SamplerDescriptor::default()
		});

		let pipeline =
			build_pipeline(device, format, &[&screen_layout, &texture_layout], shader.source())?;

		let vertices = device.create_buffer(&BufferDescriptor {
			label: Some("ui vertices"),
			size: INITIAL_VERTICES * vertex_stride(),
			usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let indices = device.create_buffer(&BufferDescriptor {
			label: Some("ui indices"),
			size: INITIAL_VERTICES * 6 * 4,
			usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let blank = white(device, &texture_layout, &sampler);

		Ok(Self {
			pipeline,
			shader,
			format,
			screen_layout,
			texture_layout,
			screen,
			screen_group,
			sampler,
			vertices,
			indices,
			vertex_capacity: INITIAL_VERTICES,
			index_capacity: INITIAL_VERTICES * 6,
			uploaded: Vec::new(),
			blank,
		})
	}

	/// Rebuilds the pipeline when somebody edits the shader.
	///
	/// A shader that will not compile costs a log line rather than the process,
	/// exactly as the scene's does: the old pipeline stays and the next save
	/// gets another go.
	pub fn reload_shader(&mut self, device: &Device) {
		if !self.shader.changed() {
			return;
		}

		match build_pipeline(
			device,
			self.format,
			&[&self.screen_layout, &self.texture_layout],
			self.shader.source(),
		) {
			| Ok(pipeline) => {
				self.pipeline = pipeline;
				debug!(path = ?self.shader.path(), "interface shader reloaded");
			},
			| Err(error) => {
				colby_core::warn!(%error, "the interface shader did not compile; keeping the \
				 one that did");
			},
		}
	}

	/// Puts everything this frame's list samples on the GPU.
	///
	/// @param device - the device to make textures on
	/// @param queue - where to write their bytes
	/// @param world - where the fonts and textures live
	/// @param list - what is about to be drawn
	pub fn upload(&mut self, device: &Device, queue: &Queue, world: &World, list: &DrawList) {
		for batch in &list.batches {
			match batch.binding {
				| Binding::Blank => {},
				| Binding::Font(id) => self.upload_font(device, queue, world, id),
				| Binding::Image(id) => self.upload_image(device, queue, world, id),
			}
		}
	}

	/// Writes the list into the vertex and index buffers, growing them if it
	/// does not fit.
	pub fn write(&mut self, device: &Device, queue: &Queue, list: &DrawList, viewport: [f32; 2]) {
		queue.write_buffer(
			&self.screen,
			0,
			bytemuck::bytes_of(&Screen { viewport, padding: [0.0; 2] }),
		);

		let wanted = u64::try_from(list.vertices.len()).unwrap_or(0);
		if wanted > self.vertex_capacity {
			self.vertex_capacity = wanted.next_power_of_two();
			self.vertices = device.create_buffer(&BufferDescriptor {
				label: Some("ui vertices"),
				size: self.vertex_capacity * vertex_stride(),
				usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
				mapped_at_creation: false,
			});
		}

		let wanted = u64::try_from(list.indices.len()).unwrap_or(0);
		if wanted > self.index_capacity {
			self.index_capacity = wanted.next_power_of_two();
			self.indices = device.create_buffer(&BufferDescriptor {
				label: Some("ui indices"),
				size: self.index_capacity * 4,
				usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
				mapped_at_creation: false,
			});
		}

		if !list.vertices.is_empty() {
			queue.write_buffer(&self.vertices, 0, bytemuck::cast_slice(&list.vertices));
		}

		if !list.indices.is_empty() {
			queue.write_buffer(&self.indices, 0, bytemuck::cast_slice(&list.indices));
		}
	}

	/// Records and submits the interface over a frame that already has a scene
	/// in it.
	pub fn render(&self, device: &Device, queue: &Queue, target: &TextureView, list: &DrawList) {
		if list.is_empty() {
			return;
		}

		let mut encoder =
			device.create_command_encoder(&CommandEncoderDescriptor { label: Some("ui") });

		{
			let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
				label: Some("ui"),
				color_attachments: &[Some(RenderPassColorAttachment {
					view: target,
					depth_slice: None,
					resolve_target: None,
					ops: Operations {
						// `Load`, not `Clear`: the scene is already in here.
						load: LoadOp::Load,
						store: StoreOp::Store,
					},
				})],
				depth_stencil_attachment: None,
				timestamp_writes: None,
				occlusion_query_set: None,
				multiview_mask: None,
			});

			pass.set_pipeline(&self.pipeline);
			pass.set_bind_group(0, &self.screen_group, &[]);
			pass.set_vertex_buffer(0, self.vertices.slice(..));
			pass.set_index_buffer(self.indices.slice(..), IndexFormat::Uint32);

			for batch in list
				.batches
				.iter()
				.filter(|batch| batch.count > 0)
			{
				pass.set_bind_group(1, self.group_for(batch.binding), &[]);
				pass.draw_indexed(batch.first..batch.first + batch.count, 0, 0..1);
			}
		}

		queue.submit([encoder.finish()]);
	}

	/// The bind group a batch samples from, falling back to the blank texture.
	fn group_for(&self, binding: Binding) -> &BindGroup {
		self.uploaded
			.iter()
			.find(|uploaded| uploaded.binding == binding)
			.map_or(&self.blank, |uploaded| &uploaded.group)
	}

	/// Makes sure a font's atlas is on the GPU and up to date.
	fn upload_font(&mut self, device: &Device, queue: &Queue, world: &World, id: FontId) {
		let Some(entry) = world.fonts.get(id) else {
			return;
		};

		let binding = Binding::Font(id);
		if self.is_current(binding, entry.revision()) {
			return;
		}

		let font = entry.value();
		if font.atlas_width == 0 || font.atlas_height == 0 {
			return;
		}

		let group = self.make_texture(
			device,
			queue,
			"ui font",
			TextureFormat::R8Unorm,
			font.atlas_width,
			font.atlas_height,
			1,
			&font.atlas,
		);

		self.remember(binding, entry.revision(), group);
	}

	/// The same for a texture a document draws.
	fn upload_image(&mut self, device: &Device, queue: &Queue, world: &World, id: TextureId) {
		let Some(entry) = world.textures.get(id) else {
			return;
		};

		let binding = Binding::Image(id);
		if self.is_current(binding, entry.revision()) {
			return;
		}

		let data = entry.value();
		let Some(level) = data.levels.first() else {
			return;
		};

		let bytes = u32::try_from(data.texel.bytes()).unwrap_or(4);
		let format = match data.texel {
			| Texel::Rgba8Srgb => TextureFormat::Rgba8UnormSrgb,
		};

		let group = self.make_texture(
			device,
			queue,
			"ui image",
			format,
			data.width,
			data.height,
			bytes,
			level,
		);

		self.remember(binding, entry.revision(), group);
	}

	/// Whether what is on the GPU already matches what is in the table.
	fn is_current(&self, binding: Binding, revision: u32) -> bool {
		self.uploaded
			.iter()
			.any(|uploaded| uploaded.binding == binding && uploaded.revision == revision)
	}

	/// Records an upload, replacing any older one for the same binding.
	fn remember(&mut self, binding: Binding, revision: u32, group: BindGroup) {
		self.uploaded
			.retain(|uploaded| uploaded.binding != binding);
		self.uploaded
			.push(Uploaded { binding, revision, group });
	}

	/// Makes one texture and the bind group that reads it.
	#[expect(
		clippy::too_many_arguments,
		reason = "a texture is described by exactly these, and a struct holding them would be \
		          the same arguments behind one name"
	)]
	fn make_texture(
		&self,
		device: &Device,
		queue: &Queue,
		label: &str,
		format: TextureFormat,
		width: u32,
		height: u32,
		bytes_per_texel: u32,
		bytes: &[u8],
	) -> BindGroup {
		let texture = device.create_texture(&TextureDescriptor {
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
			usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
			view_formats: &[],
		});

		queue.write_texture(
			TexelCopyTextureInfo {
				texture: &texture,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: TextureAspect::All,
			},
			bytes,
			TexelCopyBufferLayout {
				offset: 0,
				bytes_per_row: Some(width.max(1) * bytes_per_texel),
				rows_per_image: Some(height.max(1)),
			},
			Extent3d {
				width: width.max(1),
				height: height.max(1),
				depth_or_array_layers: 1,
			},
		);

		device.create_bind_group(&BindGroupDescriptor {
			label: Some(label),
			layout: &self.texture_layout,
			entries: &[
				BindGroupEntry {
					binding: 0,
					resource: BindingResource::TextureView(
						&texture.create_view(&TextureViewDescriptor::default()),
					),
				},
				BindGroupEntry {
					binding: 1,
					resource: BindingResource::Sampler(&self.sampler),
				},
			],
		})
	}
}

/// The one white texel a batch that samples nothing is bound to.
///
/// A rectangle ignores what it sampled, but a pipeline still has to have
/// something bound at every slot its layout declares.
fn white(device: &Device, layout: &BindGroupLayout, sampler: &wgpu::Sampler) -> BindGroup {
	let texture = device.create_texture(&TextureDescriptor {
		label: Some("ui blank"),
		size: Extent3d {
			width: 1,
			height: 1,
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		sample_count: 1,
		dimension: TextureDimension::D2,
		format: TextureFormat::Rgba8UnormSrgb,
		usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
		view_formats: &[],
	});

	device.create_bind_group(&BindGroupDescriptor {
		label: Some("ui blank"),
		layout,
		entries: &[
			BindGroupEntry {
				binding: 0,
				resource: BindingResource::TextureView(
					&texture.create_view(&TextureViewDescriptor::default()),
				),
			},
			BindGroupEntry {
				binding: 1,
				resource: BindingResource::Sampler(sampler),
			},
		],
	})
}

/// Builds the pipeline, reporting a shader that will not compile.
fn build_pipeline(
	device: &Device,
	format: TextureFormat,
	layouts: &[&BindGroupLayout],
	source: &str,
) -> Result<RenderPipeline> {
	let scope = device.push_error_scope(ErrorFilter::Validation);

	let shader = device.create_shader_module(ShaderModuleDescriptor {
		label: Some("ui"),
		source: ShaderSource::Wgsl(source.into()),
	});

	let groups: Vec<Option<&BindGroupLayout>> = layouts.iter().copied().map(Some).collect();
	let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some("ui"),
		bind_group_layouts: &groups,
		immediate_size: 0,
	});

	let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some("ui"),
		layout: Some(&layout),
		vertex: VertexState {
			module: &shader,
			entry_point: Some("vs_main"),
			compilation_options: PipelineCompilationOptions::default(),
			buffers: &[Some(VertexBufferLayout {
				array_stride: vertex_stride(),
				step_mode: VertexStepMode::Vertex,
				attributes: &VERTEX_ATTRIBUTES,
			})],
		},
		primitive: PrimitiveState {
			topology: PrimitiveTopology::TriangleList,
			strip_index_format: None,
			front_face: FrontFace::Ccw,
			// off: an interface is flat quads, and a document that lays one out
			// with a negative size should draw nothing rather than draw
			// something only from behind.
			cull_mode: None,
			unclipped_depth: false,
			polygon_mode: PolygonMode::Fill,
			conservative: false,
		},
		depth_stencil: None,
		multisample: MultisampleState::default(),
		fragment: Some(FragmentState {
			module: &shader,
			entry_point: Some("fs_main"),
			compilation_options: PipelineCompilationOptions::default(),
			targets: &[Some(ColorTargetState {
				format,
				// straight alpha over whatever is already in the frame, which
				// is the whole point of drawing after the scene.
				blend: Some(BlendState::ALPHA_BLENDING),
				write_mask: ColorWrites::ALL,
			})],
		}),
		multiview_mask: None,
		cache: None,
	});

	if let Some(complaint) = pollster::block_on(scope.pop()) {
		return Err(err!(Graphics("the interface shader did not compile: {complaint}")));
	}

	Ok(pipeline)
}

/// The shader compiled into the binary.
const fn built_in() -> &'static str { include_str!("ui.wgsl") }

/// How many bytes one [`Vertex`] takes.
fn vertex_stride() -> u64 { u64::try_from(size_of::<Vertex>()).unwrap_or(56) }

/// What one [`Vertex`] hands the vertex stage.
///
/// @note: offsets written out rather than taken from a macro, so that a change
/// to `Vertex` shows up here as a mismatch to fix instead of as a picture made
/// of garbage.
const VERTEX_ATTRIBUTES: [VertexAttribute; 7] = [
	VertexAttribute {
		format: VertexFormat::Float32x2,
		offset: 0,
		shader_location: 0,
	},
	VertexAttribute {
		format: VertexFormat::Float32x2,
		offset: 8,
		shader_location: 1,
	},
	VertexAttribute {
		format: VertexFormat::Float32x2,
		offset: 16,
		shader_location: 2,
	},
	VertexAttribute {
		format: VertexFormat::Float32x2,
		offset: 24,
		shader_location: 3,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 32,
		shader_location: 4,
	},
	VertexAttribute {
		format: VertexFormat::Float32,
		offset: 48,
		shader_location: 5,
	},
	VertexAttribute {
		format: VertexFormat::Float32,
		offset: 52,
		shader_location: 6,
	},
];

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_vertex_layout_matches_the_struct_it_describes() {
		assert_eq!(vertex_stride(), 56, "two, two, two, two, four floats and two more");

		let last = VERTEX_ATTRIBUTES
			.last()
			.expect("there are attributes");

		assert_eq!(
			last.offset + 4,
			vertex_stride(),
			"the last attribute has to end where the vertex does, or the pipeline reads the \
			 next one's first field as this one's last"
		);
	}

	#[test]
	fn the_screen_uniform_is_a_whole_number_of_sixteen_byte_rows() {
		assert!(
			size_of::<Screen>().is_multiple_of(16),
			"a uniform buffer that is not is a validation error rather than a wrong picture"
		);
	}
}
