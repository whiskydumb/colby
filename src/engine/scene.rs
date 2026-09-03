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
		EntityId, MAX_ENTITIES, Material, MeshData, MeshVertex, Meshes, SkinVertex, Texel,
		TextureData, TextureId, Textures, World,
		material::{Blend, MaterialEntry, Wrap},
		registry::Entry,
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
	ColorTargetState, ColorWrites, CommandEncoder, CommandEncoderDescriptor, CompareFunction,
	DepthBiasState, DepthStencilState, Device, ErrorFilter, Extent3d, Face, FilterMode,
	FragmentState, FrontFace, IndexFormat, LoadOp, MipmapFilterMode, MultisampleState,
	Operations, Origin3d, PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode,
	PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment,
	RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline,
	RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor,
	ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState, StoreOp,
	TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
	TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
	TextureViewDescriptor, TextureViewDimension, VertexAttribute, VertexBufferLayout,
	VertexFormat, VertexState, VertexStepMode,
};

use crate::{
	lines::Lines,
	shader::Shader,
	shadow::{self, CASCADES, Cascades, Maps},
	skin::Joints,
};

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
}

/// One texture, uploaded, with its whole mip chain.
struct GpuTexture {
	view: TextureView,
	revision: u32,
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
	) -> Result<Self> {
		Ok(Self {
			entries: [
				compile_pipeline(device, format, layouts, source, Blend::Opaque, false)?,
				compile_pipeline(device, format, layouts, source, Blend::Opaque, true)?,
				compile_pipeline(device, format, layouts, source, Blend::Mask, false)?,
				compile_pipeline(device, format, layouts, source, Blend::Mask, true)?,
			],
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
	device: Device,
	queue: Queue,
	pipelines: Pipelines,
	globals: Buffer,
	bindings: BindGroup,
	/// Kept so the pipelines and the per-material groups can be built again.
	format: TextureFormat,
	globals_layout: BindGroupLayout,
	material_layout: BindGroupLayout,
	/// One per [`Wrap`], in its discriminant order.
	samplers: [Sampler; 2],
	shader: Shader,
	depth: TextureView,
	/// The depth array the light writes and the scene samples.
	shadows: Maps,
	/// This frame's light matrices, fitted in `upload` and drawn in `render`.
	cascades: Cascades,
	/// Whether the console left the shadow passes switched on this frame.
	shadowing: bool,
	/// The debug renderer, drawn into this scene's pass and its depth buffer.
	lines: Lines,
	/// This frame's joint matrices, and where each pose's run is in them.
	joints: Joints,
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
	/// hundred-and-twelve-byte placements.
	order: Vec<(u32, u32, EntityId)>,
}

impl Scene {
	/// Builds the pipelines, the depth buffer and the bind group layouts.
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
		let pipelines = Pipelines::build(&device, format, &groups, shader.source())?;
		let depth = depth_view(&device, width, height);
		let lines = Lines::new(&device, format, &globals_layout)?;

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
			globals,
			bindings,
			format,
			globals_layout,
			material_layout,
			samplers,
			shader,
			depth,
			shadows,
			cascades: Cascades::NONE,
			shadowing: false,
			lines,
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
			order: Vec::with_capacity(MAX_ENTITIES),
		})
	}

	/// Rebuilds the depth buffer for a new target size.
	pub fn resize(&mut self, width: u32, height: u32) {
		self.depth = depth_view(&self.device, width, height);
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
		let groups = [
			&self.globals_layout,
			&self.material_layout,
			self.shadows.sample_layout(),
			self.joints.layout(),
		];
		// the whole table, and none of it is assigned until all of it has
		// compiled: half a reload is a world where the crates moved and the
		// characters did not.
		self.pipelines = Pipelines::build(&self.device, self.format, &groups, source)?;

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

		// first, and once per cascade: the scene's own pass samples what these
		// wrote, so they have to be recorded ahead of it. Skipped entirely when
		// the console has turned shadows off, which leaves the maps holding
		// whatever was in them and is safe because nothing then reads them.
		if self.shadowing {
			for slice in 0..CASCADES {
				self.cast(&mut encoder, slice);
			}
		}

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

		pass.set_bind_group(0, &self.bindings, &[]);
		pass.set_bind_group(2, self.shadows.bindings(), &[]);
		pass.set_bind_group(3, self.joints.bindings(), &[]);
		pass.set_vertex_buffer(1, self.instances.slice(..));

		// swapped when a batch wants another one rather than once per batch.
		// The order is by mesh and material, so a world of crates with one
		// character in it changes pipeline twice however many crates there are.
		let mut bound = None;

		for batch in &self.batches {
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

		// last, into the same pass and therefore against the same depth buffer:
		// whether a debug line is hidden by a wall is the most useful thing it
		// has to say, and an overlay is handed the color target alone.
		self.lines.draw(&mut pass, &self.bindings);

		drop(pass);
		self.queue.submit([encoder.finish()]);
	}

	/// Which one draws a batch into a cascade.
	///
	/// A match rather than a lookup, so that a mode which should not cast at
	/// all - and blending is exactly that - is a compile error here on the day
	/// it is added rather than a fence-shaped hole in the light.
	fn casting(&self, blend: Blend, skinned: bool) -> &RenderPipeline {
		let masked = match blend {
			| Blend::Opaque => false,
			| Blend::Mask => true,
		};

		self.shadows.casting(masked, skinned)
	}

	/// Records one cascade's depth pass.
	///
	/// The same batches the scene draws, through a pipeline with no fragment
	/// stage and no color target, so the whole pass is geometry against depth.
	///
	/// The material is not bound at all, which used to be free and is now a
	/// known gap: a [`Blend::Mask`] surface casts the shadow of the shape it
	/// was cut out of rather than the shape that is left, because nothing here
	/// samples the picture the holes are in. Closing it is a third depth-only
	/// pipeline with a fragment stage that discards.
	///
	/// @param encoder - what to record into
	/// @param slice - which cascade, nearest first
	fn cast(&self, encoder: &mut CommandEncoder, slice: usize) {
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
			timestamp_writes: None,
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

			let wanted = (batch.blend, batch.skinned);
			if bound != Some(wanted) {
				pass.set_pipeline(self.casting(batch.blend, batch.skinned));
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
		self.joints.begin(world);

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

		match self.batches.last_mut() {
			| Some(batch) if batch.mesh == mesh && batch.material == material => batch.count += 1,
			| _ => self.batches.push(Batch {
				mesh,
				material,
				first: at,
				count: 1,
				skinned,
				// taken from the same value the numbers above came from rather
				// than looked up again, so that however a handle resolved, the
				// mode a batch is drawn under is the mode of the material its
				// instances were written from.
				blend: surface.blend,
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
fn compile_pipeline(
	device: &Device,
	format: TextureFormat,
	layouts: &[&BindGroupLayout],
	source: &str,
	blend: Blend,
	skinned: bool,
) -> Result<RenderPipeline> {
	let scope = device.push_error_scope(ErrorFilter::Validation);
	let pipeline = build_pipeline(device, format, layouts, source, blend, skinned);

	match pollster::block_on(scope.pop()) {
		| Some(complaint) => Err(err!(Graphics("{complaint}"))),
		| None => Ok(pipeline),
	}
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
			depth_write_enabled: Some(true),
			depth_compare: Some(CompareFunction::Less),
			stencil: StencilState::default(),
			bias: DepthBiasState::default(),
		}),
		multisample: MultisampleState::default(),
		fragment: Some(FragmentState {
			module: &shader,
			entry_point: Some(match blend {
				| Blend::Opaque => "fragment_main",
				| Blend::Mask => "fragment_masked",
			}),
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
		assert!(size_of::<Globals>() == 432, "a uniform struct has to be a multiple of 16");
		assert!(size_of::<Globals>().is_multiple_of(16), "and this one is not");
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
