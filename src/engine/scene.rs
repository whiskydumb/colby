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

use colby_core::{
	Result,
	abi::{
		EntityId, MAX_ENTITIES, Material, MeshData, MeshVertex, Meshes, Texel, TextureData,
		Textures, World, material::MaterialEntry, registry::Entry,
	},
	bytemuck::{self, Pod, Zeroable},
	err, error,
	glam::Vec3,
	info,
};
use wgpu::{
	AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
	BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
	Buffer, BufferAddress, BufferBindingType, BufferDescriptor, BufferUsages, Color,
	ColorTargetState, ColorWrites, CommandEncoderDescriptor, CompareFunction, DepthBiasState,
	DepthStencilState, Device, ErrorFilter, Extent3d, Face, FilterMode, FragmentState, FrontFace,
	IndexFormat, LoadOp, MipmapFilterMode, MultisampleState, Operations, Origin3d,
	PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode, PrimitiveState,
	PrimitiveTopology, Queue, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
	RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerBindingType,
	SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState, StoreOp,
	TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
	TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
	TextureViewDescriptor, TextureViewDimension, VertexAttribute, VertexBufferLayout,
	VertexFormat, VertexState, VertexStepMode,
};

use crate::shader::Shader;

/// The depth format. Thirty-two bits is more than a scene this size needs and
/// is supported everywhere, which is worth more right now than the memory.
pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

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
	light: [f32; 4],
	ambient: [f32; 4],
	/// xyz is where the camera is; w is unused. Needed by anything that depends
	/// on the viewing angle, which is all of the specular term.
	eye: [f32; 4],
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
	index_count: u32,
	revision: u32,
}

/// One texture, uploaded, with its whole mip chain.
struct GpuTexture {
	view: TextureView,
	revision: u32,
}

/// One material's bind group.
///
/// Two revisions, because it can go stale for two unrelated reasons: the
/// material itself changed, or the texture it names was re-uploaded and the
/// view this group holds now points at a texture nobody else is using.
struct GpuMaterial {
	bindings: BindGroup,
	material_revision: u32,
	texture_slot: u32,
	texture_revision: u32,
}

/// A run of instances that share both a mesh and a material.
struct Batch {
	mesh: usize,
	material: usize,
	first: u32,
	count: u32,
}

/// A device, a pipeline, the resources uploaded so far, and a frame's buffers.
pub struct Scene {
	device: Device,
	queue: Queue,
	pipeline: RenderPipeline,
	globals: Buffer,
	bindings: BindGroup,
	/// Kept so the pipeline and the per-material groups can be built again.
	format: TextureFormat,
	globals_layout: BindGroupLayout,
	material_layout: BindGroupLayout,
	sampler: Sampler,
	shader: Shader,
	depth: TextureView,
	/// One per registry slot, in the same order, filled on demand.
	meshes: Vec<GpuMesh>,
	textures: Vec<GpuTexture>,
	materials: Vec<GpuMaterial>,
	instances: Buffer,
	/// Scratch the instance buffer is built in, kept so it allocates once.
	placements: Vec<Placement>,
	/// Which run of `placements` belongs to which mesh and material.
	batches: Vec<Batch>,
	/// `(mesh slot, material slot, entity)`, sorted to find the runs. Sorting
	/// sixteen-byte keys and looking the entities up again beats sorting the
	/// ninety-six-byte placements.
	order: Vec<(u32, u32, EntityId)>,
}

impl Scene {
	/// Builds the pipeline, the depth buffer and the bind group layouts.
	///
	/// @param device - the device to build against
	/// @param queue - the queue every upload goes through
	/// @param format - the color format the fragment stage writes
	/// @param width - the target's width in pixels
	/// @param height - the target's height in pixels
	pub fn new(
		device: Device,
		queue: Queue,
		format: TextureFormat,
		width: u32,
		height: u32,
	) -> Result<Self> {
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

		let material_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
			],
		});

		let bindings = device.create_bind_group(&BindGroupDescriptor {
			label: Some("globals"),
			layout: &globals_layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: globals.as_entire_binding(),
			}],
		});

		// one sampler for everything. Per-material sampling belongs in the
		// material the day one wants to clamp rather than repeat; until then a
		// second one would only be a second thing to keep in step.
		let sampler = device.create_sampler(&SamplerDescriptor {
			label: Some("material"),
			address_mode_u: AddressMode::Repeat,
			address_mode_v: AddressMode::Repeat,
			address_mode_w: AddressMode::Repeat,
			mag_filter: FilterMode::Linear,
			min_filter: FilterMode::Linear,
			mipmap_filter: MipmapFilterMode::Linear,
			..SamplerDescriptor::default()
		});

		let shader = Shader::new("shader.wgsl", include_str!("shader.wgsl"));
		let pipeline = compile_pipeline(
			&device,
			format,
			&[&globals_layout, &material_layout],
			shader.source(),
		)?;
		let depth = depth_view(&device, width, height);

		let instances = device.create_buffer(&BufferDescriptor {
			label: Some("placements"),
			size: size_bytes::<Placement>(MAX_ENTITIES)?,
			usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		Ok(Self {
			device,
			queue,
			pipeline,
			globals,
			bindings,
			format,
			globals_layout,
			material_layout,
			sampler,
			shader,
			depth,
			// nothing is uploaded until a frame says what the world holds: the
			// registries belong to the host, and a scene built before the host
			// has loaded its assets would only have to be rebuilt afterwards.
			meshes: Vec::new(),
			textures: Vec::new(),
			materials: Vec::new(),
			instances,
			placements: Vec::with_capacity(MAX_ENTITIES),
			batches: Vec::new(),
			order: Vec::with_capacity(MAX_ENTITIES),
		})
	}

	/// Rebuilds the depth buffer for a new target size.
	pub fn resize(&mut self, width: u32, height: u32) {
		self.depth = depth_view(&self.device, width, height);
	}

	/// Builds the pipeline from new shader source, keeping the old one if the
	/// new source does not compile.
	///
	/// The pipeline is only replaced once wgpu has confirmed it is valid, so a
	/// shader with a typo in it costs a message and nothing else - the same
	/// bargain a game module that panics gets, and for the same reason: the
	/// code being edited is expected to be wrong sometimes.
	///
	/// @param source - the whole WGSL
	/// @return the compiler's complaint, if it had one
	pub fn set_shader(&mut self, source: &str) -> Result {
		self.pipeline = compile_pipeline(
			&self.device,
			self.format,
			&[&self.globals_layout, &self.material_layout],
			source,
		)?;

		Ok(())
	}

	/// Uploads this frame and records it into a target.
	///
	/// @param target - what to draw into
	/// @param world - the state to draw
	pub fn render(&mut self, target: &TextureView, world: &World) {
		self.reload_shader();
		self.upload(world);

		let mut encoder = self
			.device
			.create_command_encoder(&CommandEncoderDescriptor { label: Some("frame") });

		let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
			label: Some("scene"),
			color_attachments: &[Some(RenderPassColorAttachment {
				view: target,
				depth_slice: None,
				resolve_target: None,
				ops: Operations {
					load: LoadOp::Clear(clear_color(world)),
					store: StoreOp::Store,
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
			timestamp_writes: None,
			occlusion_query_set: None,
			multiview_mask: None,
		});

		pass.set_pipeline(&self.pipeline);
		pass.set_bind_group(0, &self.bindings, &[]);
		pass.set_vertex_buffer(1, self.instances.slice(..));

		for batch in &self.batches {
			let (Some(mesh), Some(material)) =
				(self.meshes.get(batch.mesh), self.materials.get(batch.material))
			else {
				continue;
			};

			pass.set_bind_group(1, &material.bindings, &[]);
			pass.set_vertex_buffer(0, mesh.vertices.slice(..));
			pass.set_index_buffer(mesh.indices.slice(..), IndexFormat::Uint32);
			pass.draw_indexed(0..mesh.index_count, 0, batch.first..batch.first + batch.count);
		}

		drop(pass);
		self.queue.submit([encoder.finish()]);
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

		// asked for once and used twice on purpose. This is where the frame
		// stops being the simulation's and becomes the picture's: the camera
		// the world holds is where the last step left it, and this one is
		// where it is *now*, part of the way to the next. Taking the matrix
		// from one and the eye position from the other would light the scene
		// from a camera that is not the one looking at it.
		let camera = world.render_camera();

		self.queue.write_buffer(
			&self.globals,
			0,
			bytemuck::bytes_of(&Globals {
				view_projection: camera
					.view_projection(world.aspect)
					.to_cols_array_2d(),
				light: world.light.extend(0.0).to_array(),
				ambient: world.ambient.extend(0.0).to_array(),
				eye: camera.position.extend(1.0).to_array(),
			}),
		);

		self.group(world);

		if self.placements.is_empty() {
			return;
		}

		self.queue
			.write_buffer(&self.instances, 0, bytemuck::cast_slice(&self.placements));
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
	/// Rebuilt when the material moved *or* when the texture it names did,
	/// because a bind group holds a view of one particular texture and
	/// re-uploading an image makes a new one.
	fn sync_materials(&mut self, world: &World) {
		for (slot, entry) in world.materials.iter().enumerate() {
			let texture_slot = entry.value().albedo.index();
			let texture_revision = world
				.textures
				.get(entry.value().albedo)
				.map_or(0, Entry::revision);

			let current = self.materials.get(slot).is_some_and(|uploaded| {
				uploaded.material_revision == entry.revision()
					&& uploaded.texture_slot == texture_slot
					&& uploaded.texture_revision == texture_revision
			});

			if current {
				continue;
			}

			let Some(uploaded) = self.build_material(entry, texture_slot, texture_revision)
			else {
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
	/// A material naming a texture that has not been uploaded falls back to
	/// slot zero, which is the white texel - so a material pointing at nothing
	/// draws its own color rather than failing to draw.
	fn build_material(
		&self,
		entry: &MaterialEntry,
		texture_slot: u32,
		texture_revision: u32,
	) -> Option<GpuMaterial> {
		let view = usize::try_from(texture_slot)
			.ok()
			.and_then(|slot| self.textures.get(slot))
			.or_else(|| self.textures.first())?;

		let bindings = self
			.device
			.create_bind_group(&BindGroupDescriptor {
				label: Some("material"),
				layout: &self.material_layout,
				entries: &[
					BindGroupEntry {
						binding: 0,
						resource: BindingResource::TextureView(&view.view),
					},
					BindGroupEntry {
						binding: 1,
						resource: BindingResource::Sampler(&self.sampler),
					},
				],
			});

		Some(GpuMaterial {
			bindings,
			material_revision: entry.revision(),
			texture_slot,
			texture_revision,
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

			self.order.push((mesh, material, id));
		}

		self.order
			.sort_unstable_by_key(|(mesh, material, _)| (*mesh, *material));

		self.placements.clear();
		self.batches.clear();
		for index in 0..self.order.len() {
			self.place(world, index);
		}
	}

	/// Writes one entity of the sorted order into the instance buffer, opening
	/// a new batch when its pair differs from the one before it.
	fn place(&mut self, world: &World, index: usize) {
		let Some((mesh, material, id)) = self.order.get(index).copied() else {
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
			tint: (renderable.color * surface.base_color)
				.extend(1.0)
				.to_array(),
			surface: [
				surface.metallic,
				surface.roughness,
				surface.uv_scale.x,
				surface.uv_scale.y,
			],
		});

		let (mesh, material) =
			(usize::try_from(mesh).unwrap_or(0), usize::try_from(material).unwrap_or(0));

		match self.batches.last_mut() {
			| Some(batch) if batch.mesh == mesh && batch.material == material => batch.count += 1,
			| _ => self
				.batches
				.push(Batch { mesh, material, first: at, count: 1 }),
		}
	}
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
		index_count: u32::try_from(data.indices.len()).unwrap_or(0),
		revision,
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

/// Creates a depth buffer of a given size.
fn depth_view(device: &Device, width: u32, height: u32) -> TextureView {
	let texture = device.create_texture(&TextureDescriptor {
		label: Some("depth"),
		size: Extent3d {
			width: width.max(1),
			height: height.max(1),
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		sample_count: 1,
		dimension: TextureDimension::D2,
		format: DEPTH_FORMAT,
		usage: TextureUsages::RENDER_ATTACHMENT,
		view_formats: &[],
	});

	texture.create_view(&TextureViewDescriptor::default())
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

/// Builds the render pipeline and reports whether wgpu accepted it.
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
fn compile_pipeline(
	device: &Device,
	format: TextureFormat,
	layouts: &[&BindGroupLayout],
	source: &str,
) -> Result<RenderPipeline> {
	let scope = device.push_error_scope(ErrorFilter::Validation);
	let pipeline = build_pipeline(device, format, layouts, source);

	match pollster::block_on(scope.pop()) {
		| Some(complaint) => Err(err!(Graphics("{complaint}"))),
		| None => Ok(pipeline),
	}
}

/// Builds the single render pipeline.
///
/// @param device - the device to build against
/// @param format - the color format the fragment stage writes
/// @param layouts - the bind group layouts, in group order
/// @param source - the whole WGSL
fn build_pipeline(
	device: &Device,
	format: TextureFormat,
	layouts: &[&BindGroupLayout],
	source: &str,
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

	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some("scene"),
		layout: Some(&layout),
		vertex: VertexState {
			module: &shader,
			entry_point: Some("vertex_main"),
			compilation_options: PipelineCompilationOptions::default(),
			buffers: &[Some(vertices), Some(instances)],
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
			depth_write_enabled: Some(true),
			depth_compare: Some(CompareFunction::Less),
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

/// What one [`MeshVertex`] hands the vertex stage.
///
/// @note: offsets written out rather than taken from a macro, so that a change
/// to `MeshVertex` shows up here as a mismatch to fix instead of as garbled
/// geometry. @ref [`strides`].
const VERTEX_ATTRIBUTES: [VertexAttribute; 3] = [
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
];

/// What one [`Placement`] hands it, once per instance.
///
/// The model matrix takes four of these because wgsl has no matrix vertex
/// attribute; the shader puts it back together.
const INSTANCE_ATTRIBUTES: [VertexAttribute; 6] = [
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 0,
		shader_location: 3,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 16,
		shader_location: 4,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 32,
		shader_location: 5,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 48,
		shader_location: 6,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 64,
		shader_location: 7,
	},
	VertexAttribute {
		format: VertexFormat::Float32x4,
		offset: 80,
		shader_location: 8,
	},
];

/// The vertex and instance strides, asserted to match the attributes above.
const fn strides() -> (BufferAddress, BufferAddress) {
	const {
		assert!(size_of::<MeshVertex>() == 32, "MeshVertex is no longer two vec3s and a vec2");
		assert!(align_of::<MeshVertex>() == 4, "MeshVertex gained padding");
		assert!(size_of::<Placement>() == 96, "Placement is no longer a mat4 and two vec4s");
		assert!(align_of::<Placement>() == 4, "Placement gained padding");
		assert!(size_of::<Globals>() == 112, "a uniform struct has to be a multiple of 16");
	}

	(32, 96)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_buffers_are_sized_for_what_goes_in_them() {
		assert_eq!(
			size_bytes::<MeshVertex>(24).expect("the size fits"),
			24 * 32,
			"a cube's worth of position, normal and texture coordinate"
		);
		assert_eq!(
			size_bytes::<Placement>(MAX_ENTITIES).expect("the size fits"),
			96 * BufferAddress::try_from(MAX_ENTITIES).expect("the count fits"),
			"one placement per entity the world can hold"
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
