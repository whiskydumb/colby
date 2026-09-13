//! The depth buffer, and how a pass after the scene reads it.
//!
//! The scene's pass writes depth and tests against it. What reads it afterwards
//! is a pass after the scene: a light shaft, a blur by distance, fog that fills
//! a volume, and first of all the view that draws the depth instead of the
//! picture. Every one of them wants one number a pixel.
//!
//! **At one sample a pixel the buffer is that number**, and nothing runs: it is
//! bound as a texture like any other. **At four it is not.** A reader could
//! bind the multisampled buffer and pick a sample, which costs nothing and
//! makes every reader two readers, because the two sample counts are two
//! binding types and one shader cannot be written over both. Or one pass can
//! turn the four samples into one, which costs a pass and a buffer and leaves
//! every reader the same binding at every sample count. This is the second, and
//! it runs only in a frame something reads the depth in: a frame nothing asks
//! for has no pass and no buffer. @ref `depth.wgsl` for why the pass keeps the
//! nearest sample rather than the mean or the first.
//!
//! **What the depth holds is what wrote it**: the solid and the masked halves,
//! and the debug lines that test against it. Glass, particles and the sky write
//! none, so a reader sees the depth behind a pane of glass. The buffer is whole
//! the moment the scene's one pass ends, which is where it is made readable. A
//! reader that wants the depth during that pass, to fade a particle into the
//! floor, is a different job, because the pass would have to be cut in two
//! around it.

use colby_core::{Result, err};
use wgpu::{
	BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
	BindGroupLayoutEntry, BindingResource, BindingType, CommandEncoder, CompareFunction,
	DepthBiasState, DepthStencilState, Device, ErrorFilter, Extent3d, FragmentState, LoadOp,
	MultisampleState, Operations, PipelineCompilationOptions, PipelineLayoutDescriptor,
	PrimitiveState, RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline,
	RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState,
	StoreOp, TextureDescriptor, TextureDimension, TextureSampleType, TextureUsages, TextureView,
	TextureViewDescriptor, TextureViewDimension, VertexState,
};

use crate::{
	scene::DEPTH_FORMAT,
	timing::{Ends, Pass, Timings},
};

/// The console variable that draws the depth instead of the picture.
///
/// A distance in world units: black at the eye, white this far along the view,
/// straight between. Nought is the picture. A tool, like the cascade tint, so
/// off until somebody sets it and never saved.
pub const VIEW: &str = "r.depth";

/// What [`VIEW`] holds until somebody sets it, which is the picture.
pub const NO_VIEW: f32 = 0.0;

/// The buffer the scene's pass tests against, and the one a reader reads.
pub(crate) struct Depth {
	/// How many samples a pixel of the buffer has.
	samples: u32,

	/// The size the buffer was built for.
	size: (u32, u32),

	/// What the scene's pass writes, at the scene's sample count.
	buffer: TextureView,

	/// The buffer as the resolve reads it, or `None` at one sample, where
	/// there is nothing to resolve.
	multi: Option<BindGroup>,

	/// The nearest of the buffer's samples, one a pixel.
	///
	/// Only at four samples and only while something asks: a frame that reads
	/// no depth drops it, so its memory is spent exactly as long as a reader is
	/// on, which is four bytes a pixel and three and a half megabytes at
	/// seven-twenty.
	single: Option<TextureView>,

	/// What [`multi`](Self::multi) is laid out as.
	layout: BindGroupLayout,

	/// Which rebuild of [`readable`](Self::readable)'s view this is.
	///
	/// A pass after the scene that reads the depth every frame wants its bind
	/// group kept rather than made, and a bind group may only be kept as long
	/// as the view in it is the one that exists. This is what a reader keeps
	/// beside its group and compares: the number moves whenever the buffer is
	/// rebuilt for a size or a sample count, and whenever the resolved one is
	/// made or let go. It never comes back to a value it has had.
	epoch: u64,

	/// The pass that fills [`single`](Self::single).
	resolve: RenderPipeline,
}

impl Depth {
	/// Builds the buffer and the pipeline that resolves it.
	///
	/// @param device - the device to build against
	/// @param samples - how many samples a pixel the scene is drawn with
	/// @param width - the target's width in pixels
	/// @param height - its height
	pub(crate) fn new(device: &Device, samples: u32, width: u32, height: u32) -> Result<Self> {
		let layout = multi_layout(device);
		let scope = device.push_error_scope(ErrorFilter::Validation);
		let resolve = resolve_pipeline(device, &layout);

		if let Some(complaint) = pollster::block_on(scope.pop()) {
			return Err(err!(Graphics("the depth resolve: {complaint}")));
		}

		let buffer = buffer(device, samples, width, height);
		let multi = read_multi(device, &layout, samples, &buffer);

		Ok(Self {
			samples,
			size: (width, height),
			buffer,
			multi,
			single: None,
			layout,
			epoch: 0,
			resolve,
		})
	}

	/// Rebuilds the buffer for a new target size.
	pub(crate) fn resize(&mut self, device: &Device, width: u32, height: u32) {
		self.rebuild(device, self.samples, (width, height));
	}

	/// Rebuilds the buffer for a new sample count.
	///
	/// The scene's pipelines are the caller's, as they are for the color
	/// target: all three have to agree or the pass is refused.
	pub(crate) fn set_samples(&mut self, device: &Device, samples: u32) {
		self.rebuild(device, samples, self.size);
	}

	/// Everything that has to agree with the sample count and the size.
	///
	/// The resolved buffer is dropped rather than rebuilt: the next frame that
	/// asks makes one at the new size.
	fn rebuild(&mut self, device: &Device, samples: u32, size: (u32, u32)) {
		self.samples = samples;
		self.size = size;
		self.buffer = buffer(device, samples, size.0, size.1);
		self.multi = read_multi(device, &self.layout, samples, &self.buffer);
		self.single = None;
		self.epoch = self.epoch.wrapping_add(1);
	}

	/// What the scene's pass writes and tests against.
	pub(crate) const fn attachment(&self) -> &TextureView { &self.buffer }

	/// Makes the depth readable for this frame, or lets it go.
	///
	/// At four samples and asked, one pass writes the nearest of each pixel's
	/// samples into a buffer of one, which is built the first frame that asks.
	/// At one sample there is nothing to do. Not asked, the resolved buffer is
	/// dropped.
	///
	/// @param device - to build the resolved buffer on, the first time
	/// @param encoder - the frame's, with the scene's pass already in it
	/// @param asked - whether anything reads the depth this frame
	/// @param timings - what the pass writes its marks into
	pub(crate) fn make_readable(
		&mut self,
		device: &Device,
		encoder: &mut CommandEncoder,
		asked: bool,
		timings: &Timings,
	) {
		if !asked {
			if self.single.take().is_some() {
				self.epoch = self.epoch.wrapping_add(1);
			}

			return;
		}

		if self.multi.is_some() && self.single.is_none() {
			self.single = Some(one_sample(device, self.size));
			self.epoch = self.epoch.wrapping_add(1);
		}

		let (Some(multi), Some(single)) = (self.multi.as_ref(), self.single.as_ref()) else {
			return;
		};

		let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
			label: Some("depth resolve"),
			color_attachments: &[],
			depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
				view: single,
				// every pixel is written, so what was there does not matter,
				// and a clear reads nothing where a load would read it all
				depth_ops: Some(Operations {
					load: LoadOp::Clear(1.0),
					store: StoreOp::Store,
				}),
				stencil_ops: None,
			}),
			timestamp_writes: timings.writes(Pass::Depth, Ends::Both),
			occlusion_query_set: None,
			multiview_mask: None,
		});

		pass.set_pipeline(&self.resolve);
		pass.set_bind_group(0, multi, &[]);
		pass.draw(0..3, 0..1);
	}

	/// Which rebuild of [`readable`](Self::readable)'s view this is.
	///
	/// A reader that runs every frame keeps this beside the bind group it made
	/// and rebuilds the group when the two disagree, which is the whole of how
	/// a group is kept safely: the view inside one has to be the view that
	/// exists. The field it returns says when it moves.
	pub(crate) const fn epoch(&self) -> u64 { self.epoch }

	/// What a reader binds this frame: one sample a pixel, whatever the scene
	/// drew with.
	///
	/// The buffer itself at one sample. At four, the resolved one, which only
	/// exists once [`make_readable`](Self::make_readable) was asked this frame.
	pub(crate) fn readable(&self) -> Option<&TextureView> {
		if self.multi.is_some() {
			self.single.as_ref()
		} else {
			Some(&self.buffer)
		}
	}

	/// What a reader would read, one float a pixel, top row first.
	///
	/// A test's, and only a test's. It copies the texture out, which a
	/// multisampled one cannot be, so it answers for whatever
	/// [`readable`](Self::readable) answers for and for nothing else.
	///
	/// @param device - to build the staging buffer on
	/// @param queue - to submit the copy on
	#[cfg(test)]
	pub(crate) fn values(&self, device: &Device, queue: &wgpu::Queue) -> Option<Vec<f32>> {
		let texture = self.readable()?.texture();

		floats(&copied_out(device, queue, texture, wgpu::TextureAspect::DepthOnly, 4)?)
	}
}

/// A texture's texels copied out to the processor, top row first, with the
/// padding every row of a copy is rounded up to taken off again.
///
/// A test's, and only a test's: how a buffer nothing ever puts on a screen is
/// read back and held against arithmetic. The texture has to carry the copy
/// bit, which every buffer here carries in a test build and in no other.
///
/// @param device - to build the staging buffer on
/// @param queue - to submit the copy on
/// @param texture - what to copy, one sample a pixel
/// @param aspect - the depth of a depth texture, or all of a color one
/// @param stride - how many bytes one texel of it is
#[cfg(test)]
pub(crate) fn copied_out(
	device: &Device,
	queue: &wgpu::Queue,
	texture: &wgpu::Texture,
	aspect: wgpu::TextureAspect,
	stride: u32,
) -> Option<Vec<u8>> {
	let (width, height) = (texture.width(), texture.height());
	let row = width * stride;
	let padded =
		row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
	let staging = device.create_buffer(&wgpu::BufferDescriptor {
		label: Some("readback"),
		size: u64::from(padded) * u64::from(height),
		usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
		mapped_at_creation: false,
	});
	let mut encoder = device
		.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });

	encoder.copy_texture_to_buffer(
		wgpu::TexelCopyTextureInfo {
			texture,
			mip_level: 0,
			origin: wgpu::Origin3d::ZERO,
			aspect,
		},
		wgpu::TexelCopyBufferInfo {
			buffer: &staging,
			layout: wgpu::TexelCopyBufferLayout {
				offset: 0,
				bytes_per_row: Some(padded),
				rows_per_image: Some(height),
			},
		},
		Extent3d { width, height, depth_or_array_layers: 1 },
	);
	queue.submit([encoder.finish()]);

	let slice = staging.slice(..);
	slice.map_async(wgpu::MapMode::Read, |_| {});
	device
		.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
		.ok()?;

	let mapped = slice.get_mapped_range().ok()?;
	let wide = usize::try_from(row).ok()?;
	let mut bytes = Vec::with_capacity(wide * usize::try_from(height).ok()?);

	for line in mapped.chunks(usize::try_from(padded).ok()?) {
		bytes.extend_from_slice(line.get(..wide)?);
	}

	drop(mapped);
	staging.unmap();

	Some(bytes)
}

/// Four bytes at a time as the little-endian floats they are. A tail shorter
/// than four is not a float and is left off.
#[cfg(test)]
pub(crate) fn floats(bytes: &[u8]) -> Option<Vec<f32>> {
	bytes
		.chunks_exact(4)
		.map(|four| four.try_into().ok().map(f32::from_le_bytes))
		.collect()
}

/// How a reader binds [`Depth::readable`]: a depth texture of one sample, read
/// with `textureLoad`, whatever the scene's sample count is.
///
/// @param binding - where in the reader's own group it sits
pub(crate) const fn entry(binding: u32) -> BindGroupLayoutEntry {
	BindGroupLayoutEntry {
		binding,
		visibility: ShaderStages::FRAGMENT,
		ty: BindingType::Texture {
			sample_type: TextureSampleType::Depth,
			view_dimension: TextureViewDimension::D2,
			multisampled: false,
		},
		count: None,
	}
}

/// A depth buffer of a size and a sample count.
///
/// Bound as well as written: at one sample a reader reads this, and at four
/// the resolve does.
fn buffer(device: &Device, samples: u32, width: u32, height: u32) -> TextureView {
	device
		.create_texture(&TextureDescriptor {
			label: Some("depth"),
			size: Extent3d {
				width: width.max(1),
				height: height.max(1),
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: samples,
			dimension: TextureDimension::D2,
			format: DEPTH_FORMAT,
			usage: TextureUsages::RENDER_ATTACHMENT
				| TextureUsages::TEXTURE_BINDING
				| copied(samples),
			view_formats: &[],
		})
		.create_view(&TextureViewDescriptor::default())
}

/// The nearest of a buffer's samples: one a pixel, at the buffer's size.
fn one_sample(device: &Device, (width, height): (u32, u32)) -> TextureView {
	device
		.create_texture(&TextureDescriptor {
			label: Some("depth, one sample"),
			size: Extent3d {
				width: width.max(1),
				height: height.max(1),
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: DEPTH_FORMAT,
			usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING | copied(1),
			view_formats: &[],
		})
		.create_view(&TextureViewDescriptor::default())
}

/// The copy bit a test reads a buffer back through, where one can be.
///
/// **A test's and only a test's**, the way the float target's is: a shipping
/// buffer carries the two bits a frame needs. And never at four samples, since
/// a multisampled texture cannot be copied out at all.
const fn copied(samples: u32) -> TextureUsages {
	if cfg!(test) && samples == 1 {
		TextureUsages::COPY_SRC
	} else {
		TextureUsages::empty()
	}
}

/// The buffer as the resolve reads it, or nothing at one sample.
fn read_multi(
	device: &Device,
	layout: &BindGroupLayout,
	samples: u32,
	buffer: &TextureView,
) -> Option<BindGroup> {
	(samples > 1).then(|| {
		device.create_bind_group(&BindGroupDescriptor {
			label: Some("depth, multisampled"),
			layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: BindingResource::TextureView(buffer),
			}],
		})
	})
}

/// How the resolve is handed the multisampled buffer: one texture read a
/// sample at a time with `textureLoad`, so no sampler beside it.
fn multi_layout(device: &Device) -> BindGroupLayout {
	device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("depth, multisampled"),
		entries: &[BindGroupLayoutEntry {
			binding: 0,
			visibility: ShaderStages::FRAGMENT,
			ty: BindingType::Texture {
				sample_type: TextureSampleType::Depth,
				view_dimension: TextureViewDimension::D2,
				multisampled: true,
			},
			count: None,
		}],
	})
}

/// The pass that keeps the nearest sample: a triangle over the target, no
/// color at all, and a depth written whatever was there before.
fn resolve_pipeline(device: &Device, layout: &BindGroupLayout) -> RenderPipeline {
	let module = device.create_shader_module(ShaderModuleDescriptor {
		label: Some("depth"),
		source: ShaderSource::Wgsl(include_str!("depth.wgsl").into()),
	});
	let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some("depth resolve"),
		bind_group_layouts: &[Some(layout)],
		immediate_size: 0,
	});

	device.create_render_pipeline(&RenderPipelineDescriptor {
		label: Some("depth resolve"),
		layout: Some(&pipeline_layout),
		vertex: VertexState {
			module: &module,
			entry_point: Some("vertex_screen"),
			compilation_options: PipelineCompilationOptions::default(),
			buffers: &[],
		},
		primitive: PrimitiveState::default(),
		depth_stencil: Some(DepthStencilState {
			format: DEPTH_FORMAT,
			depth_write_enabled: Some(true),
			depth_compare: Some(CompareFunction::Always),
			stencil: StencilState::default(),
			bias: DepthBiasState::default(),
		}),
		multisample: MultisampleState::default(),
		fragment: Some(FragmentState {
			module: &module,
			entry_point: Some("fragment_nearest"),
			compilation_options: PipelineCompilationOptions::default(),
			targets: &[],
		}),
		multiview_mask: None,
		cache: None,
	})
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{MeshId, Renderable, Transform, Value, World},
		glam::{Quat, Vec3},
	};

	use super::*;
	use crate::{Capture, capture::rgb, scene::MSAA};

	/// How big every capture here is.
	const SIZE: (u32, u32) = (320, 240);

	/// How far apart a depth read back and one worked out may be and still be
	/// the same depth.
	///
	/// The worked-out one comes from the frame's own camera and projection,
	/// so what is left is rounding. **Measured, then set at four times it**:
	/// the worst seen on this RX is 3.0e-7 on the floor at one sample, five
	/// steps of a float that close to one, and 6e-8 on the wall at either
	/// count. The nearest sample reads at least 8.1e-5 under the center of any
	/// steep pixel of floor, sixty-seven times this, so the test that tells it
	/// from the mean is nowhere near the edge of it.
	const WITHIN: f64 = 1.2e-6;

	/// How far the wall's face is from the eye.
	const WALL: f32 = 5.0;

	/// Where the wall's two edges fall, in pixels from the left: three
	/// quarters of the way across one column and a quarter of the way across
	/// another.
	///
	/// **Where the four samples of the standard pattern sit decides both.** At
	/// an eighth, three eighths, five eighths and seven eighths of a pixel,
	/// the left edge leaves one sample of its column on the wall and the right
	/// edge leaves one of its own, and neither leaves the center there. So the
	/// nearest sample says wall in both columns, the center says nothing, the
	/// mean says neither, and the first sample, three eighths across, says
	/// nothing in both.
	const EDGES: (f32, f32) = (100.75, 219.25);

	/// The columns a wall of [`EDGES`] covers when a pixel is its nearest
	/// sample, and when it is its center.
	const NEAREST: (u32, u32) = (100, 219);
	const CENTERS: (u32, u32) = (101, 218);

	/// The floor's top, as the world below builds it: sixty wide and deep, its
	/// near edge ten units behind the eye.
	const FLOOR: (f64, f64, f64) = (30.0, -50.0, 10.0);

	/// A capture on the binary's one device, or `None` with no GPU.
	fn capture() -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, SIZE.0, SIZE.1) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// How many samples a pixel is drawn with and how far away the view's
	/// white is, the second nought for the picture.
	fn asking(world: &mut World, samples: &str, white: &str) {
		world.cvars.var(MSAA, Value::Float(1.0), "");
		world.cvars.set(MSAA, samples);
		world.cvars.var(VIEW, Value::Float(NO_VIEW), "");
		world.cvars.set(VIEW, white);
	}

	/// Which pixel an index into a readback is: its column and its row.
	fn pixel(at: usize) -> (u32, u32) {
		let width = usize::try_from(SIZE.0).unwrap_or(1);

		(
			u32::try_from(at % width).unwrap_or(u32::MAX),
			u32::try_from(at / width).unwrap_or(u32::MAX),
		)
	}

	/// A camera two units over a floor and looking down the length of it: a
	/// surface whose depth changes across every pixel, fastest at the far edge.
	fn floor() -> World {
		let mut world = World::new();

		world.camera.position = Vec3::new(0.0, 2.0, 0.0);
		world.camera.target = Vec3::new(0.0, 0.0, -6.0);

		let id = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, -0.5, -20.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(60.0, 1.0, 60.0),
		});

		world
			.entities
			.set_renderable(id, Renderable::new(MeshId::CUBE, rgb(0.5, 0.5, 0.5)));

		world
	}

	/// Where the ray through a point of the target meets the floor's top, as
	/// the depth stored there, or `None` where it misses.
	///
	/// From the frame's own camera and without inverting anything: its three
	/// axes built the way its view matrix builds them, the ray through the
	/// point by the projection's own two scales, and the depth that projection
	/// stores at the distance along the view where the ray meets the plane.
	///
	/// @param x - pixels from the left edge of the target
	/// @param y - pixels from the top
	fn floor_at(world: &World, x: f32, y: f32) -> Option<f64> {
		let camera = world.render_camera();
		let lens = camera.projection(world.aspect);
		let forward = (camera.target - camera.position).normalize();
		let side = forward.cross(camera.up).normalize();
		let up = side.cross(forward);
		let across = (x / 320.0).mul_add(2.0, -1.0) / lens.x_axis.x;
		let rise = (y / 240.0).mul_add(-2.0, 1.0) / lens.y_axis.y;
		// a unit along the view for every unit of this, so how far along the
		// ray the plane is, is how far along the view it is
		let ray = forward + side * across + up * rise;
		let along = -camera.position.y / ray.y;

		if ray.y >= 0.0 || !(camera.near..=camera.far).contains(&along) {
			return None;
		}

		let hit = camera.position + ray * along;
		let (half, far, close) = FLOOR;

		(f64::from(hit.x).abs() <= half && (far..=close).contains(&f64::from(hit.z)))
			.then(|| f64::from(lens.w_axis.z / along - lens.z_axis.z))
	}

	/// What the floor is at one pixel.
	enum Under {
		/// Inside the floor: its depth at the pixel's center, and at the
		/// pixel's nearest corner.
		Floor {
			center: f64,
			corner: f64,
		},

		/// Wholly past it, where nothing is drawn.
		Past,

		/// An edge of the floor runs through the pixel.
		Edge,
	}

	/// The floor's depth at a pixel's center and at its four corners.
	fn footprint(world: &World, (column, row): (u32, u32)) -> Under {
		let (left, top) = (
			f32::from(u16::try_from(column).unwrap_or(u16::MAX)),
			f32::from(u16::try_from(row).unwrap_or(u16::MAX)),
		);
		let center = floor_at(world, left + 0.5, top + 0.5);
		let corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
			.map(|(right, down)| floor_at(world, left + right, top + down));
		let nearest = corners
			.iter()
			.try_fold(f64::INFINITY, |least, corner| corner.map(|depth| least.min(depth)));

		match (center, nearest) {
			| (Some(center), Some(corner)) => Under::Floor { center, corner },
			| (None, _) if corners.iter().all(Option::is_none) => Under::Past,
			| _ => Under::Edge,
		}
	}

	/// A wall square to the view and [`WALL`] away, running off the top and
	/// the bottom of the picture, with its edges at [`EDGES`] and nothing
	/// behind it.
	fn wall() -> World {
		let mut world = World::new();

		world.camera.position = Vec3::ZERO;
		world.camera.target = Vec3::NEG_Z;

		// the aspect every capture here has, so that a column is where the
		// wall was built to put it; the view is the identity, looking down -z
		let across = world.camera.projection(320.0 / 240.0).x_axis.x;
		let x_of = |column: f32| (column / 320.0).mul_add(2.0, -1.0) * WALL / across;
		let (left, right) = (x_of(EDGES.0), x_of(EDGES.1));
		let id = world.entities.spawn_at(Transform {
			position: Vec3::new((left + right) * 0.5, 0.0, -WALL - 0.5),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(right - left, 20.0, 1.0),
		});

		world
			.entities
			.set_renderable(id, Renderable::new(MeshId::CUBE, rgb(0.5, 0.5, 0.5)));

		world
	}

	/// The depth the wall's face is stored at, everywhere on it: what the
	/// frame's projection stores for a point [`WALL`] along the view.
	fn wall_depth(world: &World) -> f64 {
		let lens = world.render_camera().projection(world.aspect);

		f64::from(lens.w_axis.z / WALL - lens.z_axis.z)
	}

	/// Whether a depth is the clear's one exactly: nothing was drawn there.
	fn cleared(value: f32) -> bool { value.to_bits() == 1.0_f32.to_bits() }

	#[test]
	fn at_one_sample_what_is_read_is_the_buffer_at_every_pixel_center() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = floor();

		asking(&mut world, "1", "20");
		capture
			.shoot(&mut world)
			.expect("the capture renders");

		let values = capture
			.scene_mut()
			.depth_values()
			.expect("at one sample the buffer is readable in every frame");
		let (mut worst, mut on, mut past) = (0.0_f64, 0_u32, 0_u32);

		for (at, value) in values.iter().enumerate() {
			match footprint(&world, pixel(at)) {
				| Under::Floor { center, .. } => {
					worst = worst.max((f64::from(*value) - center).abs());
					on += 1;
				},
				| Under::Past => {
					assert!(
						cleared(*value),
						"{:?} sees past the floor and read {value}",
						pixel(at)
					);
					past += 1;
				},
				| Under::Edge => {},
			}
		}

		assert!(worst <= WITHIN, "the floor's depth is out by {worst:e} at worst");
		assert!(on > 20_000 && past > 10_000, "the fixture sees both: {on} on, {past} past");
	}

	#[test]
	fn at_four_samples_a_surface_reads_nearer_than_its_center_and_no_nearer_than_its_corners() {
		// what tells the nearest sample from the other two answers inside a
		// surface: the mean of four samples in the standard pattern is the
		// center exactly, and the first of them sits in the upper, farther
		// part of a pixel of floor, so it reads beyond the center
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = floor();

		asking(&mut world, "4", "20");
		capture
			.shoot(&mut world)
			.expect("the capture renders");

		let values = capture
			.scene_mut()
			.depth_values()
			.expect("asked for at four samples, so resolved");
		let (mut steep, mut nearer) = (0_u32, 0_u32);

		for (at, value) in values.iter().enumerate() {
			let read = f64::from(*value);

			match footprint(&world, pixel(at)) {
				| Under::Floor { center, corner } => {
					assert!(
						read >= corner - WITHIN && read <= center + WITHIN,
						"{:?} read {read}, outside {corner} ..= {center}",
						pixel(at)
					);

					let slope = center - corner > 4.0 * WITHIN;

					steep += u32::from(slope);
					nearer += u32::from(slope && read < center - WITHIN);
				},
				| Under::Past => {
					assert!(
						cleared(*value),
						"{:?} sees past the floor and read {read}",
						pixel(at)
					);
				},
				| Under::Edge => {},
			}
		}

		assert!(steep > 1_000, "the fixture has a slope worth the name: {steep} pixels");
		assert_eq!(nearer, steep, "every steep pixel reads nearer than its center");
	}

	#[test]
	fn at_four_samples_an_edge_reads_its_nearest_sample_and_at_one_its_center() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = wall();

		for (samples, (first, last)) in [("4", NEAREST), ("1", CENTERS)] {
			asking(&mut world, samples, "12.5");
			capture
				.shoot(&mut world)
				.expect("the capture renders");

			let expected = wall_depth(&world);
			let values = capture
				.scene_mut()
				.depth_values()
				.expect("asked for, so readable");

			for (at, value) in values.iter().enumerate() {
				let (column, row) = pixel(at);
				let on = (first..=last).contains(&column);
				let right = (on && (f64::from(*value) - expected).abs() <= WITHIN)
					|| (!on && cleared(*value));

				assert!(
					right,
					"at {samples} samples ({column}, {row}) read {value}, where the wall is \
					 {expected} and the pixel being on it is {on}"
				);
			}
		}
	}

	#[test]
	fn the_view_draws_a_distance_as_a_byte_at_either_sample_count() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = wall();

		// five units of twelve and a half is two fifths, and two fifths of 255
		// is 102; past the wall is the far plane, which is past white
		for (samples, edges) in [("4", 102), ("1", 255)] {
			asking(&mut world, samples, "12.5");

			let image = capture
				.shoot(&mut world)
				.expect("the capture renders");

			for (column, expected) in
				[(160, 102), (NEAREST.0, edges), (NEAREST.1, edges), (20, 255), (300, 255)]
			{
				let seen = image.pixel(column, SIZE.1 / 2);

				assert!(
					seen[0].abs_diff(expected) <= 1 && seen[0] == seen[1] && seen[1] == seen[2],
					"at {samples} samples column {column} is {seen:?}, not a grey of {expected}"
				);
			}
		}
	}

	#[test]
	fn the_depth_costs_a_pass_at_four_samples_and_only_while_something_reads_it() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = wall();
		let mut frame = |world: &mut World, samples: &str, white: &str| {
			asking(world, samples, white);
			capture.draw(world, &mut []);

			let passes = capture.scene_mut().spans().passes();

			(passes, capture.scene_mut().depth_values().is_some())
		};

		let quiet = frame(&mut world, "4", "0");
		let asked = frame(&mut world, "4", "12.5");
		let again = frame(&mut world, "4", "0");
		let single = frame(&mut world, "1", "0");
		let viewed = frame(&mut world, "1", "12.5");

		assert_eq!(asked.0, quiet.0 + 1, "the resolve is one pass, and the view replaces one");
		assert_eq!(again.0, quiet.0, "and it goes when nothing asks");
		assert!(
			!quiet.1 && asked.1 && !again.1,
			"the resolved buffer is only held while asked: {quiet:?} {asked:?} {again:?}"
		);
		assert_eq!(viewed.0, single.0, "at one sample nothing is added");
		assert!(single.1 && viewed.1, "and the buffer is readable either way");
	}

	#[test]
	fn a_resized_scene_reads_its_depth_at_the_new_size_at_either_count() {
		// the buffer and the resolved one are both rebuilt when the target
		// changes size, and a resolved buffer of the old size kept past a
		// rebuild would still be resolved into and read without complaint
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = wall();

		for samples in ["4", "1"] {
			asking(&mut world, samples, "12.5");
			capture.draw(&mut world, &mut []);
			capture.scene_mut().resize(160, 120);
			capture.draw(&mut world, &mut []);

			let values = capture
				.scene_mut()
				.depth_values()
				.expect("asked for, so readable");

			assert_eq!(values.len(), 160 * 120, "at {samples} samples the depth is the new size");

			capture.scene_mut().resize(SIZE.0, SIZE.1);
		}
	}

	#[test]
	fn a_distance_that_is_not_one_draws_the_picture() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = wall();

		asking(&mut world, "4", "0");

		let picture = capture
			.shoot(&mut world)
			.expect("the capture renders");

		for asked in ["-3", "0.0"] {
			asking(&mut world, "4", asked);

			let again = capture
				.shoot(&mut world)
				.expect("the capture renders");

			assert!(
				again.pixels == picture.pixels,
				"a view distance of {asked} draws the picture"
			);
		}
	}
}
