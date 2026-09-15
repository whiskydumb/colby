//! What is wholly behind what the pass before the scene drew, left out of the
//! scene's own pass.
//!
//! **What a hidden thing costs a frame.** Everything the view holds is drawn by
//! the pass before the scene and again by the scene, and a thing behind a wall
//! is geometry both passes carry and fragments the scene may shade before the
//! wall is drawn over them. Measured before any of this was written, on a
//! street of houses whose walls hide nine tenths of what the view holds: taking
//! the hidden things out of the scene's pass alone took seventy microseconds
//! out of two hundred and fifty, and collapsing them in the vertex stage
//! instead took only thirty-three, so what has to go is the draw and not only
//! the pixels.
//!
//! **The depth this frame drew, not the last.** One compute pass between the
//! pass before the scene and the scene's own folds the depth that pass has just
//! written into a pyramid of farthest depths, asks of every box in the
//! picture's lists whether it lies behind that, and copies what is kept into a
//! second run of placements that the scene draws through one indirect command
//! a batch. Everything the view holds was drawn into that depth a moment ago,
//! so a thing that has come out from behind a wall this frame is in it already:
//! nothing is a frame late, nothing needs a history, and a screenshot - one
//! frame - is left out of exactly as a window's frame is. The engines that ask
//! the hardware which boxes passed read the answer a frame or two later, and
//! whatever appears in between is missing for that long, from a mirror too.
//!
//! **The pass before the scene in two halves.** Everything large in view is
//! drawn into it ahead of the test, and that is the depth all of this reads;
//! everything too small to hide much is drawn into it after the test, through
//! what the test kept, so what is small and behind a wall never reaches either
//! pass. On that street it took the pass from 112 microseconds to 54, and a
//! frame with nothing small in it records the second half not at all. @ref
//! [`SIZE`].
//!
//! **Split on the size, not on the last frame.** To split it on the last frame,
//! drawing ahead of the test what that frame kept or testing everything first
//! against that frame's depth as the one engine read that splits this pass
//! does, takes a history and a second pyramid a frame. Measured in the first
//! shape, with the pass that picks the first half out, it cost more on every
//! world but the street than it saved there; and a screenshot has no last
//! frame to split on.
//!
//! **What a small thing hides is not in the depth.** A thing hidden only by
//! things smaller than [`SIZE`] is drawn, into both passes: a thing under
//! thirty-two pixels across can hide only what is smaller still, and every
//! world measured left out as many things of the scene's pass with the split
//! as without it.
//!
//! **What it leaves in.** Nothing is left out of a shadow map: a thing the eye
//! cannot see can still throw a shadow the eye can, and taking the hidden
//! things out of the shadows moved a hundred and forty pixels of a field of
//! cubes by up to seventy-three levels. Lamps, decals, particles and the debug
//! lines are not asked.
//!
//! **Four samples a pixel see a little more than the one this reads.** The pass
//! before the scene takes the middle of each pixel and the scene may take four
//! samples around it, so the rectangle a box can touch is grown by a pixel on
//! every side, which answers a surface leaning away. What it does not answer is
//! a gap narrower than a pixel between two things, through which a sample off
//! the middle could see what every middle around it cannot.
//!
//! **Counted on the device.** What was left out is known where it was worked
//! out, so the counts come back the way the timestamps do: waited for by a
//! measuring run, picked up two or three frames late by a window, and only
//! while somebody is measuring.

use std::sync::{
	Arc,
	atomic::{AtomicBool, Ordering},
};

use colby_core::{
	Result,
	abi::{MAX_ENTITIES, World},
	bytemuck::{self, Pod, Zeroable},
	err,
	glam::Mat4,
	warn,
};
use wgpu::{
	BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
	BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferAddress, BufferBindingType,
	BufferDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor, ComputePipeline,
	ComputePipelineDescriptor, Device, ErrorFilter, Extent3d, MapMode,
	PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, Queue, ShaderModule,
	ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, TextureDescriptor,
	TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
	TextureViewDescriptor, TextureViewDimension,
};

use crate::{
	cull::Placed,
	prepass::Prepass,
	scene::Viewport,
	shader::Shader,
	timing::{Ends, Pass, Timings},
};

/// The console variable that turns the test off.
///
/// **On, and not saved**, `r.cull`'s terms: leaving out what is behind
/// something nearer is the same picture for less, and what the switch is for is
/// the other way round - drawing what is hidden too, which is how what the test
/// saves is measured and how a picture is shown not to depend on it. `r.cull`
/// off leaves out nothing at all, this included.
pub const ENABLED: &str = "r.cover";

/// The console variable that says how many pixels across a solid thing has to
/// stand to be drawn into the pass before the scene ahead of the test.
///
/// **Thirty-two, and not saved.** A smaller thing is drawn into that pass
/// after the test and only if the test kept it, so what is small and behind a
/// wall reaches neither pass; a larger one is drawn ahead of it, into the depth
/// the test reads. Nought draws everything ahead of the test, which is the pass
/// as it was before it was split and how what the split saves is measured. The
/// measure is a level's, @ref [`Eye::under`](crate::detail::Eye::under).
pub const SIZE: &str = "r.cover_size";

/// How many pixels across a thing has to stand to be drawn ahead of the test
/// when nothing says otherwise.
pub const DEFAULT_SIZE: f32 = 32.0;

/// The format the pyramid is written in: one depth a texel.
const FORMAT: TextureFormat = TextureFormat::R32Float;

/// How many bytes one indirect command is: five words.
pub(crate) const COMMAND_SIZE: BufferAddress = 20;

/// How many invocations one dispatch of the test and of the copy groups, the
/// `@workgroup_size` both entry points declare.
const ROW: u32 = 64;

/// How many texels a side one dispatch of the pyramid groups.
const TILE: u32 = 8;

/// How many things one chunk of the count is, `CHUNK` in `cover.wgsl`.
const CHUNK: u32 = 16;

/// One thing in the picture's lists, laid out the way `Reach` in `cover.wgsl`
/// declares it: its box in the world and where it is drawn.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub(crate) struct Reach {
	center: [f32; 3],
	batch: u32,
	x: [f32; 3],
	first: u32,
	y: [f32; 3],
	triangles: u32,
	z: [f32; 3],
	spare: u32,
}

impl Reach {
	/// One thing's record.
	///
	/// @param placed - its box, as the frustum test was asked it
	/// @param batch - which of the frame's batches draws it, the solid ones
	/// first and the blended ones after
	/// @param first - where that batch's run of placements begins
	/// @param triangles - how many triangles its mesh has
	pub(crate) fn of(placed: &Placed, batch: u32, first: u32, triangles: u32) -> Self {
		let [x, y, z] = placed.edges.map(|edge| edge.to_array());

		Self {
			center: placed.center.to_array(),
			batch,
			x,
			first,
			y,
			triangles,
			z,
			spare: 0,
		}
	}
}

/// The numbers every entry point reads, laid out as `Tuning` in `cover.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct Tuning {
	view_projection: [[f32; 4]; 4],

	/// `[x, y, width, height]` of the rectangle drawn into.
	rect: [f32; 4],

	/// `[things in the lists, levels over the rectangle, 0, 0]`.
	counts: [u32; 4],
}

/// Which level one pass of the pyramid writes, as `Level` in `cover.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct Level {
	index: u32,
	spare: [u32; 3],
}

/// How much one frame left out of the scene's lists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Covered {
	/// Things.
	pub(crate) instances: u32,

	/// The triangles of their meshes, added together.
	pub(crate) triangles: u32,
}

/// Whether this frame asks for the test.
///
/// @param world - for the console variable
#[must_use]
pub(crate) fn asking_of(world: &World) -> bool { world.cvars.bool(ENABLED).unwrap_or(true) }

/// How many pixels across a thing has to stand to be drawn ahead of the test,
/// or nothing when the variable draws everything ahead of it.
///
/// @param world - for the console variable
///
/// @note: nought, less and not a number are all nothing here, and every one
/// of them would call nothing small anyway - @ref
/// [`Eye::under`](crate::detail::Eye::under) is a product against a product -
/// so a mutation pass that let them through passed everything. It is here so
/// that a frame that splits nothing takes no distance to every thing in view.
#[must_use]
pub(crate) fn sizing_of(world: &World) -> Option<f32> {
	Some(world.cvars.float(SIZE).unwrap_or(DEFAULT_SIZE)).filter(|size| *size > 0.0)
}

/// The buffers the test reads and writes, the same size for every picture.
struct Lists {
	/// One [`Reach`] a thing, written by the scene before the frame.
	reaches: Buffer,

	/// One word a thing: whether the test kept it.
	kept: Buffer,

	/// What is kept, batch by batch, where the scene's pass reads its
	/// placements from.
	compacted: Buffer,

	/// One draw command a batch, written by the scene with every count at
	/// nought and counted by the copy.
	commands: Buffer,

	/// What the test left out, two words.
	tally: Buffer,

	/// How many kept things every chunk of the lists holds, and how many come
	/// before it. @ref `CHUNK` in `cover.wgsl`.
	chunks: Buffer,

	/// The mappable copy of those two words.
	read: Buffer,
}

/// The six pipelines, and what they are laid out as.
struct Pipelines {
	first: ComputePipeline,
	next: ComputePipeline,
	test: ComputePipeline,
	count: ComputePipeline,
	gather: ComputePipeline,
	compact: ComputePipeline,
	first_layout: BindGroupLayout,
	next_layout: BindGroupLayout,
	lists_layout: BindGroupLayout,
}

/// The pyramid of farthest depths, and every group that reads or writes it.
struct Pyramid {
	/// Each level on its own, as the pass that writes it writes it and the pass
	/// above it reads it.
	levels_one_by_one: Vec<TextureView>,

	/// How each level after the first reads the one below and writes itself.
	next: Vec<BindGroup>,

	/// How the test and the copy read everything.
	lists: BindGroup,

	/// How many levels the texture has.
	levels: u32,

	/// The whole texture, for a test to read a level of.
	#[cfg(test)]
	texture: wgpu::Texture,
}

/// The test, and everything it keeps from frame to frame.
pub(crate) struct Cover {
	/// The size of the picture, which the pyramid is made for.
	size: (u32, u32),

	/// Whether the device can run a compute pass and draw from an indirect
	/// command at all. A device that cannot leaves nothing out.
	able: bool,

	/// How many bytes one of the scene's placements is.
	placement: BufferAddress,

	/// Built the first frame something asks, and nothing when they would not
	/// build, which is said once.
	pipelines: Option<Pipelines>,

	/// Whether building them was refused, so it is not asked again every frame.
	refused: bool,

	/// Made with the pipelines.
	lists: Option<Lists>,

	/// Made the first frame something asks at a size, and let go the first
	/// frame nothing does.
	pyramid: Option<Pyramid>,

	/// How the first level reads the pass before the scene's depth, and which
	/// making of that pass's buffers it was made over. @ref
	/// [`Prepass::epoch`].
	first: Option<(u64, BindGroup)>,

	/// The numbers every entry point reads.
	tuning: Buffer,

	/// Whether this frame's counts were copied out and not yet read.
	copied: bool,

	/// Whether a map of the counts has been asked for and not yet read.
	mapping: bool,

	/// Set by the map's callback.
	ready: Arc<AtomicBool>,

	/// What the last frame whose counts came back left out.
	counted: Covered,

	/// Whether the scene's pass may draw through what this wrote.
	covered: bool,

	device: Device,
}

impl Cover {
	/// A test that has built nothing yet.
	///
	/// @param device - the device to build against
	/// @param able - whether it can run a compute pass and draw indirectly
	/// @param placement - how many bytes one of the scene's placements is
	/// @param (width, height) - the picture's size
	pub(crate) fn new(
		device: &Device,
		able: bool,
		placement: BufferAddress,
		(width, height): (u32, u32),
	) -> Result<Self> {
		let tuning = device.create_buffer(&BufferDescriptor {
			label: Some("cover tuning"),
			size: bytes_of_count::<Tuning>(1)?,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		Ok(Self {
			size: (width, height),
			able,
			placement,
			pipelines: None,
			refused: false,
			lists: None,
			pyramid: None,
			first: None,
			tuning,
			copied: false,
			mapping: false,
			ready: Arc::new(AtomicBool::new(false)),
			counted: Covered::default(),
			covered: false,
			device: device.clone(),
		})
	}

	/// Notes a new picture size; the pyramid is made again by the next frame
	/// that asks.
	pub(crate) fn resize(&mut self, width: u32, height: u32) {
		if self.size == (width, height) {
			return;
		}

		self.size = (width, height);
		self.let_go();
	}

	/// Lets the pyramid and the group over the pass before the scene go.
	fn let_go(&mut self) {
		self.pyramid = None;
		self.first = None;
	}

	/// Writes what the scene laid out for this frame: every thing's box, and
	/// one command a batch.
	///
	/// @param queue - where they are written
	/// @param reaches - one a thing of the picture's lists, in placement order
	/// @param commands - five words a batch, the solid batches first
	pub(crate) fn upload(&mut self, queue: &Queue, reaches: &[Reach], commands: &[u32]) {
		if !self.able || reaches.is_empty() || !self.ensure() {
			return;
		}

		let Some(lists) = self.lists.as_ref() else {
			return;
		};

		queue.write_buffer(&lists.reaches, 0, bytemuck::cast_slice(reaches));
		queue.write_buffer(&lists.commands, 0, bytemuck::cast_slice(commands));
		queue.write_buffer(&lists.tally, 0, &[0_u8; 8]);
	}

	/// Builds the pipelines and the buffers, if they have not been.
	///
	/// @return whether there is a test to run
	fn ensure(&mut self) -> bool {
		if self.pipelines.is_some() && self.lists.is_some() {
			return true;
		}

		if self.refused {
			return false;
		}

		match build(&self.device)
			.and_then(|built| lists(&self.device, self.placement).map(|lists| (built, lists)))
		{
			| Ok((built, lists)) => {
				self.pipelines = Some(built);
				self.lists = Some(lists);

				true
			},
			| Err(complaint) => {
				warn!(%complaint, "nothing is left out of the scene for being behind something");
				self.refused = true;

				false
			},
		}
	}

	/// Records the pass, or lets the pyramid go.
	///
	/// @param encoder - the frame's, with the pass before the scene in it
	/// @param queue - where the numbers are written
	/// @param frame - what this frame asks and what the pass reads
	/// @param rectangle - the part of the picture drawn into, or all of it
	/// @return whether the scene's pass draws its lists through what this wrote
	pub(crate) fn render(
		&mut self,
		encoder: &mut CommandEncoder,
		queue: &Queue,
		frame: Frame<'_>,
		rectangle: Option<Viewport>,
	) -> bool {
		self.covered = self.record(encoder, queue, frame, rectangle);

		if !self.covered {
			self.let_go();
			self.counted = Covered::default();
		}

		self.covered
	}

	/// The whole of [`render`](Self::render) but for what a frame that records
	/// nothing lets go.
	fn record(
		&mut self,
		encoder: &mut CommandEncoder,
		queue: &Queue,
		frame: Frame<'_>,
		rectangle: Option<Viewport>,
	) -> bool {
		let Some(depth) = frame.prepass.depth() else {
			return false;
		};

		if !frame.asked || !self.able || frame.instances == 0 || !self.ensure() {
			return false;
		}

		let (width, height) = self.size;
		let drawn = match rectangle {
			| None => Viewport::whole(width, height),
			| Some(asked) => match asked.within(width, height) {
				| Some(inside) => inside,
				| None => return false,
			},
		};
		let levels = levels_over(drawn);

		self.keep(frame.placements, frame.prepass.epoch(), depth);

		let (Some(pipelines), Some(lists), Some(pyramid), Some((_, first))) = (
			self.pipelines.as_ref(),
			self.lists.as_ref(),
			self.pyramid.as_ref(),
			self.first.as_ref(),
		) else {
			return false;
		};

		if levels > pyramid.levels {
			return false;
		}

		queue.write_buffer(
			&self.tuning,
			0,
			bytemuck::bytes_of(&Tuning {
				view_projection: frame.view_projection.to_cols_array_2d(),
				rect: [drawn.x, drawn.y, drawn.width, drawn.height].map(pixels),
				counts: [frame.instances, levels, 0, 0],
			}),
		);

		dispatch(
			encoder,
			(pipelines, pyramid, first),
			(drawn, levels, frame.instances),
			frame.timings,
		);

		// into the mappable copy only while nobody is mapping it, which wgpu
		// would refuse; the frame's counts are then simply not collected
		self.copied = !self.mapping;

		if self.copied {
			encoder.copy_buffer_to_buffer(&lists.tally, 0, &lists.read, 0, 8);
		}

		true
	}

	/// Makes the pyramid and the groups over it again, if the picture changed
	/// size or the pass before the scene made its buffers again since.
	///
	/// @param placements - the scene's placements, which the copy reads
	/// @param epoch - which making of that pass's buffers the depth is
	/// @param depth - the depth it wrote
	fn keep(&mut self, placements: &Buffer, epoch: u64, depth: &TextureView) {
		let (Some(pipelines), Some(lists)) = (self.pipelines.as_ref(), self.lists.as_ref())
		else {
			return;
		};

		if self.pyramid.is_none() {
			self.pyramid = Some(pyramid(&self.device, self.size, Groups {
				pipelines,
				lists,
				placements,
				tuning: &self.tuning,
			}));
			self.first = None;
		}

		let Some(pyramid) = self.pyramid.as_ref() else {
			return;
		};

		// @note: every path that makes that pass's buffers again today lets this
		// group go first - a resize tells both, and a frame without that pass
		// runs no test - so a mutation pass that never made it again passed
		// everything. It stays as the bargain every reader of that pass keeps.
		if self
			.first
			.as_ref()
			.is_some_and(|(made, _)| *made == epoch)
		{
			return;
		}

		let Some(written) = pyramid.levels_one_by_one.first() else {
			return;
		};

		let group = self
			.device
			.create_bind_group(&BindGroupDescriptor {
				label: Some("cover first level"),
				layout: &pipelines.first_layout,
				entries: &[
					BindGroupEntry {
						binding: 0,
						resource: self.tuning.as_entire_binding(),
					},
					BindGroupEntry {
						binding: 2,
						resource: BindingResource::TextureView(depth),
					},
					BindGroupEntry {
						binding: 4,
						resource: BindingResource::TextureView(written),
					},
				],
			});

		self.first = Some((epoch, group));
	}

	/// The run of kept placements, and the commands that say how much of each
	/// batch is in it, while the scene may draw through them.
	pub(crate) fn drawn_through(&self) -> Option<(&Buffer, &Buffer)> {
		self.lists
			.as_ref()
			.filter(|_| self.covered)
			.map(|lists| (&lists.compacted, &lists.commands))
	}

	/// How many bytes one placement is, where a batch's run begins.
	pub(crate) const fn placement(&self) -> BufferAddress { self.placement }

	/// What the last frame whose counts came back left out. Nothing for a
	/// frame that ran no test.
	pub(crate) const fn counted(&self) -> Covered { self.counted }

	/// Waits for this frame's counts and reads them.
	///
	/// **This blocks**, for [`Timings::settle`]'s reason and in the same place:
	/// after the frame has been submitted, in a run that measures.
	///
	/// @param device - the device to wait on
	pub(crate) fn settle(&mut self, device: &Device) {
		if !self.copied {
			return;
		}

		self.copied = false;

		let Some(lists) = self.lists.as_ref() else {
			return;
		};
		let slice = lists.read.slice(..);

		slice.map_async(MapMode::Read, |_| {});

		if device
			.poll(PollType::Wait { submission_index: None, timeout: None })
			.is_ok() && let Ok(view) = slice.get_mapped_range()
		{
			self.counted = unpack(&view);
		}

		lists.read.unmap();
	}

	/// Asks for this frame's counts without waiting, and reads a frame's when a
	/// map has landed: [`Timings::poll`]'s half, for a window.
	///
	/// @param device - the device whose callbacks are pumped
	pub(crate) fn poll(&mut self, device: &Device) {
		let Some(lists) = self.lists.as_ref() else {
			return;
		};

		drop(device.poll(PollType::Poll));

		if !self.mapping {
			self.request();

			return;
		}

		if !self.ready.load(Ordering::Acquire) {
			return;
		}

		if let Ok(view) = lists.read.slice(..).get_mapped_range() {
			self.counted = unpack(&view);
		}

		lists.read.unmap();
		self.mapping = false;
		self.ready.store(false, Ordering::Release);
	}

	/// Asks for a map of the counts a frame copied out, for a later
	/// [`poll`](Self::poll) to read.
	fn request(&mut self) {
		let (true, Some(lists)) = (self.copied, self.lists.as_ref()) else {
			return;
		};
		let ready = Arc::clone(&self.ready);

		self.ready.store(false, Ordering::Release);
		self.mapping = true;
		self.copied = false;
		lists
			.read
			.slice(..)
			.map_async(MapMode::Read, move |outcome| {
				drop(outcome);
				ready.store(true, Ordering::Release);
			});
	}

	/// Whether the pipelines were built. A test's.
	#[cfg(test)]
	pub(crate) const fn built(&self) -> bool { self.pipelines.is_some() }

	/// Whether a pyramid is being held. A test's.
	#[cfg(test)]
	pub(crate) const fn holding(&self) -> bool { self.pyramid.is_some() }

	/// One level of the pyramid, a float a texel over the whole texture's
	/// level, top row first, and how many texels across it is. A test's.
	///
	/// @param device - to build the staging buffer on
	/// @param queue - to submit the copy on
	/// @param level - which level
	#[cfg(test)]
	pub(crate) fn level_values(
		&self,
		device: &Device,
		queue: &Queue,
		level: u32,
	) -> Option<(u32, Vec<f32>)> {
		let texture = &self.pyramid.as_ref()?.texture;
		let across = (texture.width() >> level).max(1);
		let down = (texture.height() >> level).max(1);
		let row = across * 4;
		let padded =
			row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
		let staging = device.create_buffer(&BufferDescriptor {
			label: Some("cover level readback"),
			size: u64::from(padded) * u64::from(down),
			usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
			mapped_at_creation: false,
		});
		let mut encoder =
			device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

		encoder.copy_texture_to_buffer(
			wgpu::TexelCopyTextureInfo {
				texture,
				mip_level: level,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			wgpu::TexelCopyBufferInfo {
				buffer: &staging,
				layout: wgpu::TexelCopyBufferLayout {
					offset: 0,
					bytes_per_row: Some(padded),
					rows_per_image: Some(down),
				},
			},
			Extent3d {
				width: across,
				height: down,
				depth_or_array_layers: 1,
			},
		);
		queue.submit([encoder.finish()]);

		let bytes = mapped(device, &staging)?;
		let wide = usize::try_from(row).ok()?;
		let mut values = Vec::new();

		for line in bytes.chunks(usize::try_from(padded).ok()?) {
			values.extend(
				line.get(..wide)?
					.chunks_exact(4)
					.map(|four| f32::from_le_bytes(four.try_into().unwrap_or([0; 4]))),
			);
		}

		Some((across, values))
	}

	/// Whether the test kept each thing of the last frame's lists. A test's.
	///
	/// @param device - to build the staging buffer on
	/// @param queue - to submit the copy on
	/// @param count - how many things the lists held
	#[cfg(test)]
	pub(crate) fn kept_values(
		&self,
		device: &Device,
		queue: &Queue,
		count: u32,
	) -> Option<Vec<u32>> {
		words(device, queue, &self.lists.as_ref()?.kept, count)
	}

	/// The commands the last frame drew through, five words a batch. A test's.
	///
	/// @param device - to build the staging buffer on
	/// @param queue - to submit the copy on
	/// @param batches - how many batches the lists held
	#[cfg(test)]
	pub(crate) fn command_values(
		&self,
		device: &Device,
		queue: &Queue,
		batches: u32,
	) -> Option<Vec<u32>> {
		words(device, queue, &self.lists.as_ref()?.commands, batches * 5)
	}
}

/// What a frame hands the test.
#[derive(Clone, Copy)]
pub(crate) struct Frame<'a> {
	/// Whether the frame asks: the variable, the frustum test, and the pass
	/// before the scene having drawn this frame.
	pub(crate) asked: bool,

	/// What the pass before the scene wrote.
	pub(crate) prepass: &'a Prepass,

	/// The scene's placements, the first of which are the picture's lists.
	pub(crate) placements: &'a Buffer,

	/// The matrix the picture is drawn through.
	pub(crate) view_projection: Mat4,

	/// How many things the picture's lists hold.
	pub(crate) instances: u32,

	/// What the pass writes its marks into.
	pub(crate) timings: &'a Timings,
}

/// Records the one compute pass: the pyramid level by level, the test, the two
/// counts and the copy.
///
/// @param encoder - the frame's
/// @param (pipelines, pyramid, first) - what runs, and the groups it reads
/// @param (drawn, levels, instances) - the rectangle, how many levels it folds
/// into, and how many things the lists hold
/// @param timings - what the pass writes its marks into
fn dispatch(
	encoder: &mut CommandEncoder,
	(pipelines, pyramid, first): (&Pipelines, &Pyramid, &BindGroup),
	(drawn, levels, instances): (Viewport, u32, u32),
	timings: &Timings,
) {
	let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
		label: Some("cover"),
		timestamp_writes: timings.compute_writes(Pass::Cover, Ends::Both),
	});
	let (across, down) = level_size(0, drawn);

	pass.set_pipeline(&pipelines.first);
	pass.set_bind_group(0, first, &[]);
	pass.dispatch_workgroups(across.div_ceil(TILE), down.div_ceil(TILE), 1);
	pass.set_pipeline(&pipelines.next);

	for (index, group) in (1..levels).zip(&pyramid.next) {
		let (across, down) = level_size(index, drawn);

		pass.set_bind_group(0, group, &[]);
		pass.dispatch_workgroups(across.div_ceil(TILE), down.div_ceil(TILE), 1);
	}

	pass.set_pipeline(&pipelines.test);
	pass.set_bind_group(0, &pyramid.lists, &[]);
	pass.dispatch_workgroups(instances.div_ceil(ROW), 1, 1);
	pass.set_pipeline(&pipelines.count);
	pass.dispatch_workgroups(instances.div_ceil(CHUNK).div_ceil(ROW), 1, 1);
	pass.set_pipeline(&pipelines.gather);
	pass.dispatch_workgroups(instances.div_ceil(CHUNK).div_ceil(ROW), 1, 1);
	pass.set_pipeline(&pipelines.compact);
	pass.dispatch_workgroups(instances.div_ceil(ROW), 1, 1);
}

/// What a group over the pyramid is made of besides the pyramid.
#[derive(Clone, Copy)]
struct Groups<'a> {
	pipelines: &'a Pipelines,
	lists: &'a Lists,
	placements: &'a Buffer,
	tuning: &'a Buffer,
}

/// How many texels a level has over a rectangle, each level half the one below
/// it rounded up: `level_size` in `cover.wgsl`.
///
/// @param index - which level, nought the first
/// @param drawn - the rectangle
fn level_size(index: u32, drawn: Viewport) -> (u32, u32) {
	let shift = index.saturating_add(1).min(31);
	let halve = |length: u32| {
		(u64::from(length) + (1_u64 << shift) - 1)
			.checked_shr(shift)
			.and_then(|halved| u32::try_from(halved).ok())
			.unwrap_or(1)
			.max(1)
	};

	(halve(drawn.width), halve(drawn.height))
}

/// How many levels a rectangle folds into before one texel is left.
///
/// @param drawn - the rectangle
fn levels_over(drawn: Viewport) -> u32 {
	let mut levels = 1;

	while level_size(levels - 1, drawn) != (1, 1) && levels < 32 {
		levels += 1;
	}

	levels
}

/// A count of pixels as the float the shader reads it as.
#[expect(
	clippy::as_conversions,
	clippy::cast_precision_loss,
	reason = "a picture's side, nowhere near where f32 stops holding integers"
)]
const fn pixels(count: u32) -> f32 { count as f32 }

/// The two words of a tally, as they were read back.
fn unpack(view: &[u8]) -> Covered {
	let word = |at: usize| {
		view.get(at..at + 4)
			.and_then(|four| four.try_into().ok())
			.map_or(0, u32::from_le_bytes)
	};

	Covered { instances: word(0), triangles: word(4) }
}

/// How many bytes `count` values of `T` are, as a buffer size.
fn bytes_of_count<T>(count: usize) -> Result<BufferAddress> {
	size_of::<T>()
		.checked_mul(count)
		.and_then(|bytes| BufferAddress::try_from(bytes).ok())
		.ok_or_else(|| err!(Graphics("a cover buffer's size does not fit")))
}

/// The copy bit a test reads a buffer or the pyramid back through: a test's
/// and only a test's.
const fn copied() -> BufferUsages {
	if cfg!(test) {
		BufferUsages::COPY_SRC
	} else {
		BufferUsages::empty()
	}
}

/// The buffers, sized for the most things a world holds.
///
/// @param device - the device to build on
/// @param placement - how many bytes one of the scene's placements is
fn lists(device: &Device, placement: BufferAddress) -> Result<Lists> {
	let things = BufferAddress::try_from(MAX_ENTITIES)
		.map_err(|_| err!(Graphics("the most things a world holds does not fit a size")))?;
	let buffer = |label, size, usage| {
		device.create_buffer(&BufferDescriptor {
			label: Some(label),
			size,
			usage: usage | copied(),
			mapped_at_creation: false,
		})
	};

	Ok(Lists {
		reaches: buffer(
			"cover reaches",
			bytes_of_count::<Reach>(MAX_ENTITIES)?,
			BufferUsages::STORAGE | BufferUsages::COPY_DST,
		),
		kept: buffer("cover kept", things * 4, BufferUsages::STORAGE),
		compacted: buffer(
			"cover placements",
			things * placement,
			BufferUsages::STORAGE | BufferUsages::VERTEX,
		),
		// a batch is at least one thing, so there are never more of them
		commands: buffer(
			"cover commands",
			things * COMMAND_SIZE,
			BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
		),
		tally: buffer(
			"cover tally",
			8,
			BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
		),
		// what each chunk holds, and what comes before it
		chunks: buffer(
			"cover chunks",
			things.div_ceil(BufferAddress::from(CHUNK)) * 8,
			BufferUsages::STORAGE,
		),
		read: device.create_buffer(&BufferDescriptor {
			label: Some("cover tally read"),
			size: 8,
			usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		}),
	})
}

/// The pyramid for a picture's size, and the groups over it.
///
/// **A power of two each way**, no smaller than half the picture: every level
/// of a texture is half the one below it rounded down, and a rectangle's levels
/// are half rounded up, so a texture that halves exactly holds every level of
/// every rectangle inside the picture.
///
/// @param device - the device to build on
/// @param (width, height) - the picture's size
/// @param groups - what the groups over it are made of besides it
fn pyramid(device: &Device, (width, height): (u32, u32), groups: Groups<'_>) -> Pyramid {
	let across = width.div_ceil(2).max(1).next_power_of_two();
	let down = height.div_ceil(2).max(1).next_power_of_two();
	let levels = across.max(down).trailing_zeros() + 1;
	let texture = device.create_texture(&TextureDescriptor {
		label: Some("cover pyramid"),
		size: Extent3d {
			width: across,
			height: down,
			depth_or_array_layers: 1,
		},
		mip_level_count: levels,
		sample_count: 1,
		dimension: TextureDimension::D2,
		format: FORMAT,
		usage: TextureUsages::STORAGE_BINDING
			| TextureUsages::TEXTURE_BINDING
			| if cfg!(test) {
				TextureUsages::COPY_SRC
			} else {
				TextureUsages::empty()
			},
		view_formats: &[],
	});
	let level = |index| {
		texture.create_view(&TextureViewDescriptor {
			label: Some("cover level"),
			base_mip_level: index,
			mip_level_count: Some(1),
			..TextureViewDescriptor::default()
		})
	};
	let one_by_one: Vec<TextureView> = (0..levels).map(level).collect();
	let next = (1..levels)
		.filter_map(|index| {
			let number = device.create_buffer(&BufferDescriptor {
				label: Some("cover level number"),
				size: 16,
				usage: BufferUsages::UNIFORM,
				mapped_at_creation: true,
			});

			number
				.slice(..)
				.get_mapped_range_mut()
				.ok()?
				.copy_from_slice(bytemuck::bytes_of(&Level { index, spare: [0; 3] }));
			number.unmap();

			let below = one_by_one.get(usize::try_from(index - 1).ok()?)?;
			let into = one_by_one.get(usize::try_from(index).ok()?)?;

			Some(device.create_bind_group(&BindGroupDescriptor {
				label: Some("cover next level"),
				layout: &groups.pipelines.next_layout,
				entries: &[
					BindGroupEntry {
						binding: 0,
						resource: groups.tuning.as_entire_binding(),
					},
					BindGroupEntry {
						binding: 1,
						resource: number.as_entire_binding(),
					},
					BindGroupEntry {
						binding: 3,
						resource: BindingResource::TextureView(below),
					},
					BindGroupEntry {
						binding: 4,
						resource: BindingResource::TextureView(into),
					},
				],
			}))
		})
		.collect();
	let all = texture.create_view(&TextureViewDescriptor::default());
	let lists = device.create_bind_group(&BindGroupDescriptor {
		label: Some("cover lists"),
		layout: &groups.pipelines.lists_layout,
		entries: &[
			whole(0, groups.tuning),
			BindGroupEntry {
				binding: 5,
				resource: BindingResource::TextureView(&all),
			},
			whole(6, &groups.lists.reaches),
			whole(7, &groups.lists.kept),
			whole(8, groups.placements),
			whole(9, &groups.lists.compacted),
			whole(10, &groups.lists.commands),
			whole(11, &groups.lists.tally),
			whole(12, &groups.lists.chunks),
		],
	});

	Pyramid {
		levels_one_by_one: one_by_one,
		next,
		lists,
		levels,
		#[cfg(test)]
		texture,
	}
}

/// A whole buffer at a binding.
fn whole(binding: u32, buffer: &Buffer) -> BindGroupEntry<'_> {
	BindGroupEntry {
		binding,
		resource: buffer.as_entire_binding(),
	}
}

/// The six pipelines against `cover.wgsl`, read through the shader directory so
/// a variant under `COLBY_SHADERS` reaches it, or the first complaint wgpu had.
///
/// @param device - the device to build against
fn build(device: &Device) -> Result<Pipelines> {
	let source = Shader::new("cover.wgsl", include_str!("cover.wgsl"));
	let scope = device.push_error_scope(ErrorFilter::Validation);
	let module = device.create_shader_module(ShaderModuleDescriptor {
		label: Some("cover"),
		source: ShaderSource::Wgsl(source.source().into()),
	});
	let layout = |label, entries: &[BindGroupLayoutEntry]| {
		device
			.create_bind_group_layout(&BindGroupLayoutDescriptor { label: Some(label), entries })
	};
	let first_layout =
		layout("cover first level", &[uniform(0), depth_entry(2), written_entry(4)]);
	let next_layout =
		layout("cover next level", &[uniform(0), uniform(1), level_entry(3), written_entry(4)]);
	let lists_layout = layout("cover lists", &[
		uniform(0),
		level_entry(5),
		storage(6, true),
		storage(7, false),
		storage(8, true),
		storage(9, false),
		storage(10, false),
		storage(11, false),
		storage(12, false),
	]);
	let pipelines = Pipelines {
		first: compute(device, &module, "pyramid_first", &first_layout),
		next: compute(device, &module, "pyramid_next", &next_layout),
		test: compute(device, &module, "test", &lists_layout),
		count: compute(device, &module, "count", &lists_layout),
		gather: compute(device, &module, "gather", &lists_layout),
		compact: compute(device, &module, "compact", &lists_layout),
		first_layout,
		next_layout,
		lists_layout,
	};

	match pollster::block_on(scope.pop()) {
		| Some(complaint) => Err(err!(Graphics("the cover pipelines: {complaint}"))),
		| None => Ok(pipelines),
	}
}

/// One compute pipeline over one group.
fn compute(
	device: &Device,
	module: &ShaderModule,
	entry: &str,
	group: &BindGroupLayout,
) -> ComputePipeline {
	let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some(entry),
		bind_group_layouts: &[Some(group)],
		immediate_size: 0,
	});

	device.create_compute_pipeline(&ComputePipelineDescriptor {
		label: Some(entry),
		layout: Some(&layout),
		module,
		entry_point: Some(entry),
		compilation_options: PipelineCompilationOptions::default(),
		cache: None,
	})
}

/// A uniform block, read by the compute stage.
const fn uniform(binding: u32) -> BindGroupLayoutEntry {
	BindGroupLayoutEntry {
		binding,
		visibility: ShaderStages::COMPUTE,
		ty: BindingType::Buffer {
			ty: BufferBindingType::Uniform,
			has_dynamic_offset: false,
			min_binding_size: None,
		},
		count: None,
	}
}

/// A storage buffer, read or read and written by the compute stage.
const fn storage(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
	BindGroupLayoutEntry {
		binding,
		visibility: ShaderStages::COMPUTE,
		ty: BindingType::Buffer {
			ty: BufferBindingType::Storage { read_only },
			has_dynamic_offset: false,
			min_binding_size: None,
		},
		count: None,
	}
}

/// The pass before the scene's depth, read with `textureLoad`.
const fn depth_entry(binding: u32) -> BindGroupLayoutEntry {
	BindGroupLayoutEntry {
		binding,
		visibility: ShaderStages::COMPUTE,
		ty: BindingType::Texture {
			sample_type: TextureSampleType::Depth,
			view_dimension: TextureViewDimension::D2,
			multisampled: false,
		},
		count: None,
	}
}

/// A level or all the levels of the pyramid, read with `textureLoad`.
const fn level_entry(binding: u32) -> BindGroupLayoutEntry {
	BindGroupLayoutEntry {
		binding,
		visibility: ShaderStages::COMPUTE,
		ty: BindingType::Texture {
			sample_type: TextureSampleType::Float { filterable: false },
			view_dimension: TextureViewDimension::D2,
			multisampled: false,
		},
		count: None,
	}
}

/// A level of the pyramid, written.
const fn written_entry(binding: u32) -> BindGroupLayoutEntry {
	BindGroupLayoutEntry {
		binding,
		visibility: ShaderStages::COMPUTE,
		ty: BindingType::StorageTexture {
			access: StorageTextureAccess::WriteOnly,
			format: FORMAT,
			view_dimension: TextureViewDimension::D2,
		},
		count: None,
	}
}

/// A buffer's first `count` words, read back. A test's.
#[cfg(test)]
fn words(device: &Device, queue: &Queue, buffer: &Buffer, count: u32) -> Option<Vec<u32>> {
	let size = u64::from(count) * 4;
	let staging = device.create_buffer(&BufferDescriptor {
		label: Some("cover words readback"),
		size: size.max(4),
		usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
		mapped_at_creation: false,
	});
	let mut encoder =
		device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

	encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size.max(4));
	queue.submit([encoder.finish()]);

	let bytes = mapped(device, &staging)?;

	Some(
		bytes
			.chunks_exact(4)
			.take(usize::try_from(count).ok()?)
			.map(|four| u32::from_le_bytes(four.try_into().unwrap_or([0; 4])))
			.collect(),
	)
}

/// A staging buffer's bytes, once the queue has written them. A test's.
#[cfg(test)]
fn mapped(device: &Device, staging: &Buffer) -> Option<Vec<u8>> {
	let slice = staging.slice(..);

	slice.map_async(MapMode::Read, |_| {});
	device
		.poll(PollType::Wait { submission_index: None, timeout: None })
		.ok()?;

	let bytes = slice.get_mapped_range().ok()?.to_vec();

	staging.unmap();

	Some(bytes)
}

#[cfg(test)]
mod tests {
	use std::f32::consts::TAU;

	use colby_core::{
		abi::{
			EntityId, Material, MaterialId, MeshId, Renderable, Transform, Value, material::Blend,
		},
		glam::{EulerRot, Quat, Vec3, Vec4},
	};

	use super::*;
	use crate::{Capture, Image, cull, occlusion, reflection, scene::MSAA};

	/// How big every capture here is.
	const SIZE: (u32, u32) = (320, 240);

	/// The wall's tint, and the tints of what stands behind and beside it.
	const GREY: Vec3 = Vec3::new(0.6, 0.58, 0.55);
	const GREEN: Vec3 = Vec3::new(0.25, 0.7, 0.3);
	const BLUE: Vec3 = Vec3::new(0.2, 0.35, 0.85);

	/// A capture on the binary's one device, or `None` with no GPU.
	fn capture() -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, SIZE.0, SIZE.1) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// How many samples a pixel is drawn with.
	fn sampled(world: &mut World, samples: &str) {
		world.cvars.var(MSAA, Value::Float(1.0), "");
		world.cvars.set(MSAA, samples);
	}

	/// Whether the test runs.
	fn covering(world: &mut World, asked: bool) {
		world.cvars.var(ENABLED, Value::Bool(true), "");
		world
			.cvars
			.set(ENABLED, if asked { "true" } else { "false" });
	}

	/// A box of a tint and a material, standing somewhere.
	fn slab(
		world: &mut World,
		material: MaterialId,
		(position, scale): (Vec3, Vec3),
		tint: Vec3,
	) -> EntityId {
		stood(
			world,
			material,
			Transform {
				position,
				rotation: Quat::IDENTITY,
				scale,
			},
			tint,
		)
	}

	/// A box of a tint and a material, standing and turned as a transform says.
	fn stood(
		world: &mut World,
		material: MaterialId,
		transform: Transform,
		tint: Vec3,
	) -> EntityId {
		let id = world.entities.spawn_at(transform);

		world
			.entities
			.set_renderable(id, Renderable::of(MeshId::CUBE, material, tint));

		id
	}

	/// A floor, and a wall eight wide and four high across the view with its
	/// front face at a quarter in front of nought, seen from six units in front
	/// of it and a little above. A sun and the renderer's own look: nothing
	/// here is worked out from a picture, only held against another.
	///
	/// @param floor_roughness - the floor's, so that a mirror can show what the
	/// wall hides when it comes out
	fn street(floor_roughness: f32) -> World {
		let mut world = World::new();

		world.camera.position = Vec3::new(0.0, 1.5, 6.0);
		world.camera.target = Vec3::new(0.0, 1.0, 0.0);
		world.light = Vec3::new(-0.4, -1.0, -0.3).normalize();

		let floor = world.materials.insert("test/floor", Material {
			roughness: floor_roughness,
			..Material::DEFAULT
		});

		slab(
			&mut world,
			floor,
			(Vec3::new(0.0, -0.5, 0.0), Vec3::new(40.0, 1.0, 40.0)),
			Vec3::splat(0.35),
		);
		slab(
			&mut world,
			MaterialId::NONE,
			(Vec3::new(0.0, 2.0, 0.0), Vec3::new(8.0, 4.0, 0.5)),
			GREY,
		);

		world
	}

	/// A unit cube of a tint on the default material.
	fn cube(world: &mut World, at: Vec3, tint: Vec3) -> EntityId {
		slab(world, MaterialId::NONE, (at, Vec3::ONE), tint)
	}

	/// Moves a thing and says it jumped there, so no frame draws it between.
	fn moved(world: &mut World, id: EntityId, at: Vec3) {
		if let Some(transform) = world.entities.transform_mut(id) {
			transform.position = at;
		}

		world.entities.snap(id);
		world.settle();
	}

	/// How many pixels across a thing has to stand to be drawn into the pass
	/// before the scene ahead of the test.
	fn sized(world: &mut World, size: &str) {
		world
			.cvars
			.var(super::SIZE, Value::Float(DEFAULT_SIZE), "");
		world.cvars.set(super::SIZE, size);
	}

	/// The depth, the surfaces and the material of a pass before the scene, as
	/// bits.
	type Written = (Vec<u32>, Vec<[u32; 4]>, Vec<[u32; 4]>);

	/// Everything the pass before the scene wrote, to the bit.
	fn written(capture: &mut Capture) -> Written {
		let scene = capture.scene_mut();
		let texels = |values: Vec<[f32; 4]>| -> Vec<[u32; 4]> {
			values
				.into_iter()
				.map(|texel| texel.map(f32::to_bits))
				.collect()
		};

		(
			scene
				.prepass_depth_values()
				.expect("the pass before the scene wrote its depth")
				.into_iter()
				.map(f32::to_bits)
				.collect(),
			texels(scene.surface_values().expect("and its surfaces")),
			texels(scene.material_values().expect("and its material")),
		)
	}

	/// A frame drawn, and what the test left out of it, read back.
	fn shot(capture: &mut Capture, world: &mut World) -> (Image, cull::Drawn) {
		let image = capture.shoot(world).expect("the capture renders");

		capture.scene_mut().settle();

		(image, capture.scene_mut().drawn())
	}

	/// How many pixels of two pictures differ at all.
	fn apart(one: &Image, other: &Image) -> usize {
		one.pixels
			.chunks_exact(4)
			.zip(other.pixels.chunks_exact(4))
			.filter(|(a, b)| a != b)
			.count()
	}

	#[test]
	fn a_box_wholly_behind_a_wall_is_left_out_and_the_picture_is_the_one_with_it_drawn() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = street(0.8);

		cube(&mut world, Vec3::new(0.0, 0.5, -3.0), GREEN);
		cube(&mut world, Vec3::new(3.0, 0.5, 2.0), BLUE);

		for samples in ["1", "4"] {
			sampled(&mut world, samples);
			covering(&mut world, true);

			let (left_out, drawn) = shot(&mut capture, &mut world);

			assert_eq!(drawn.seen, 4, "the view holds the floor, the wall and both boxes");

			let commands = capture
				.scene_mut()
				.cover_command_values()
				.expect("the test ran");
			let counted: Vec<u32> = commands
				.chunks_exact(5)
				.map(|command| command[1])
				.collect();

			assert_eq!(
				counted,
				[2, 1],
				"the default material's batch, sorted first, draws the wall and the box in \
				 front of it, and the floor's batch the floor"
			);
			assert_eq!(
				(drawn.covered, drawn.covered_triangles),
				(1, 12),
				"at {samples} samples the box behind the wall is left out, and its twelve \
				 triangles with it"
			);

			covering(&mut world, false);

			let (whole, drawn) = shot(&mut capture, &mut world);

			assert_eq!(drawn.covered, 0, "with the test off nothing is left out");
			assert_eq!(apart(&left_out, &whole), 0, "and the picture is the same to the bit");
		}
	}

	#[test]
	fn a_box_that_shows_past_the_wall_or_through_a_gap_in_it_is_drawn() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();

		world.camera.position = Vec3::new(0.0, 1.5, 6.0);
		world.camera.target = Vec3::new(0.0, 1.0, 0.0);

		// two halves of a wall with a gap a twentieth of a unit wide between
		// them, a pixel and three quarters at this distance, and a box behind
		// the gap and one behind the wall's end reaching past it
		slab(
			&mut world,
			MaterialId::NONE,
			(Vec3::new(-2.0125, 2.0, 0.0), Vec3::new(3.975, 4.0, 0.5)),
			GREY,
		);
		slab(
			&mut world,
			MaterialId::NONE,
			(Vec3::new(2.0125, 2.0, 0.0), Vec3::new(3.975, 4.0, 0.5)),
			GREY,
		);
		cube(&mut world, Vec3::new(0.0, 1.0, -2.0), GREEN);
		cube(&mut world, Vec3::new(4.2, 1.0, -1.0), BLUE);

		for samples in ["1", "4"] {
			sampled(&mut world, samples);
			covering(&mut world, true);

			let (left_out, drawn) = shot(&mut capture, &mut world);

			assert_eq!(drawn.seen, 4, "both halves and both boxes are in the view");
			assert_eq!(drawn.covered, 0, "at {samples} samples neither box is wholly hidden");

			covering(&mut world, false);

			let (whole, _) = shot(&mut capture, &mut world);

			assert_eq!(apart(&left_out, &whole), 0, "and the picture is the same to the bit");
		}
	}

	#[test]
	fn a_box_coming_out_from_behind_a_wall_is_in_the_first_frame_it_shows_and_in_the_mirror() {
		// one scene drawing every frame and keeping its pyramid, its groups and
		// its lists from one to the next, against a second scene drawing each
		// frame with nothing left out; a mirror floor, so that what comes out is
		// found in the floor in front of the wall in the same frame too
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut kept = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut whole = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = street(0.045);
		let moving = cube(&mut world, Vec3::new(0.0, 0.5, -1.5), GREEN);
		let (mut hidden, mut shown) = (0, 0);

		sampled(&mut world, "4");

		for step in 0..24_u8 {
			moved(&mut world, moving, Vec3::new(f32::from(step) * 0.25, 0.5, -1.5));
			covering(&mut world, true);

			let (left_out, drawn) = shot(&mut kept, &mut world);

			covering(&mut world, false);

			let (drawn_whole, _) = shot(&mut whole, &mut world);

			assert_eq!(
				apart(&left_out, &drawn_whole),
				0,
				"at step {step} the scene that has been leaving the box out draws the picture \
				 the scene drawing everything draws"
			);

			if drawn.covered == 1 {
				hidden += 1;
			} else {
				shown += 1;
			}
		}

		assert!(
			hidden >= 4 && shown >= 4,
			"the box was behind the wall for {hidden} frames and out of it for {shown}"
		);
	}

	#[test]
	fn a_camera_that_steps_or_jumps_past_a_wall_s_end_is_drawn_what_it_sees_in_that_frame() {
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut kept = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut whole = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = street(0.8);

		cube(&mut world, Vec3::new(2.5, 0.5, -3.0), GREEN);
		cube(&mut world, Vec3::new(-2.5, 0.5, -3.0), BLUE);
		sampled(&mut world, "1");

		// half a unit a frame to the right, and then a jump back to the far left
		let places = (0..16_u8)
			.map(|step| f32::from(step) * 0.5)
			.chain([-7.5, 0.0]);
		let mut covered = Vec::new();

		for across in places {
			world.camera.position = Vec3::new(across, 1.5, 6.0);
			world.camera.target = Vec3::new(across * 0.5, 1.0, 0.0);
			world.snap_camera();
			world.settle();
			covering(&mut world, true);

			let (left_out, drawn) = shot(&mut kept, &mut world);

			covering(&mut world, false);

			let (drawn_whole, _) = shot(&mut whole, &mut world);

			assert_eq!(
				apart(&left_out, &drawn_whole),
				0,
				"with the eye at {across} the picture is the one with everything drawn"
			);
			covered.push(drawn.covered);
		}

		assert!(
			covered.contains(&2) && covered.iter().any(|count| *count < 2),
			"both boxes were behind the wall in some frames and not in others: {covered:?}"
		);
	}

	#[test]
	fn a_picture_drawn_into_a_rectangle_leaves_out_what_the_same_picture_drawn_alone_does() {
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut alone = Capture::new(gpu, 211, 137).expect("the capture builds");
		let mut framed = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = street(0.8);
		let view = Viewport { x: 71, y: 83, width: 211, height: 137 };

		// a row of boxes behind the wall, and one beside its end
		for along in -3..=3_i8 {
			cube(&mut world, Vec3::new(f32::from(along), 0.5, -2.5), GREEN);
		}
		cube(&mut world, Vec3::new(5.0, 0.5, -2.5), BLUE);

		for samples in ["1", "4"] {
			sampled(&mut world, samples);
			covering(&mut world, true);
			alone.draw(&mut world, &mut []);
			alone.scene_mut().settle();

			let drawn_alone = alone.scene_mut().drawn();
			let left_out = framed
				.shoot_within(&mut world, view)
				.expect("the capture renders");

			framed.scene_mut().settle();

			let drawn_framed = framed.scene_mut().drawn();

			covering(&mut world, false);

			let everything = framed
				.shoot_within(&mut world, view)
				.expect("the capture renders");

			assert!(drawn_alone.covered >= 5, "the wall hides most of the row: {drawn_alone:?}");
			assert_eq!(
				drawn_framed.covered, drawn_alone.covered,
				"at {samples} samples the rectangle leaves out what the picture alone does"
			);
			// a unit box eight away is twenty-seven pixels across in a picture 137
			// tall and forty-seven in one 240 tall, so the row is small only when
			// measured against the rectangle
			assert!(drawn_alone.small >= 5, "the row is small in a picture this size");
			assert_eq!(
				drawn_framed.small, drawn_alone.small,
				"and in the rectangle, which is measured against its own height"
			);
			assert_eq!(apart(&left_out, &everything), 0, "and draws the same picture as without");
		}
	}

	/// The pyramid a rectangle of a depth folds into, worked out here: each
	/// level the farthest of two by two of the one below, a last row or column
	/// that has no neighbor taking its own.
	fn folded(depth: &[f32], width: u32, view: Viewport) -> Vec<(u32, u32, Vec<f32>)> {
		let at = |x: u32, y: u32| {
			depth
				.get(usize::try_from(y * width + x).unwrap_or(usize::MAX))
				.copied()
				.unwrap_or(f32::NAN)
		};
		let mut levels = Vec::new();
		let (mut across, mut down) = (view.width, view.height);
		let mut below: Vec<f32> = (0..view.height)
			.flat_map(|row| (0..view.width).map(move |column| (column, row)))
			.map(|(column, row)| at(view.x + column, view.y + row))
			.collect();

		loop {
			let (next_across, next_down) = (across.div_ceil(2), down.div_ceil(2));
			let texel = |(x, y): (u32, u32)| {
				below
					.get(
						usize::try_from(y.min(down - 1) * across + x.min(across - 1))
							.unwrap_or(usize::MAX),
					)
					.copied()
					.unwrap_or(f32::NAN)
			};
			let level: Vec<f32> = (0..next_down)
				.flat_map(|row| (0..next_across).map(move |column| (column, row)))
				.map(|(column, row)| {
					[(0, 0), (1, 0), (0, 1), (1, 1)]
						.map(|(dx, dy)| texel((column * 2 + dx, row * 2 + dy)))
						.into_iter()
						.fold(0.0_f32, f32::max)
				})
				.collect();

			levels.push((next_across, next_down, level.clone()));

			if (next_across, next_down) == (1, 1) {
				return levels;
			}

			(across, down, below) = (next_across, next_down, level);
		}
	}

	#[test]
	fn the_pyramid_is_the_farthest_of_what_the_pass_before_the_scene_drew_level_by_level() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = street(0.8);
		// odd on both sides and away from the corner, so that the last row and
		// column and the rectangle's origin are all asked
		let view = Viewport { x: 17, y: 9, width: 211, height: 137 };

		for along in -4..=4_i8 {
			cube(
				&mut world,
				Vec3::new(f32::from(along) * 0.9, 0.5, -2.0 - f32::from(along).abs()),
				GREEN,
			);
		}

		sampled(&mut world, "1");
		covering(&mut world, true);
		// everything ahead of the test, so that the depth read back after the
		// frame is the depth the pyramid was folded from: the farther boxes are
		// small in a rectangle this size, and would be drawn after it
		sized(&mut world, "0");
		capture.draw_within(&mut world, view);

		let depth = capture
			.scene_mut()
			.prepass_depth_values()
			.expect("the pass before the scene ran");
		let expected = folded(&depth, SIZE.0, view);

		assert_eq!(expected.len(), 8, "a rectangle of 211 by 137 folds into eight levels");

		for (level, (across, down, wanted)) in (0_u32..).zip(&expected) {
			let (stride, values) = capture
				.scene_mut()
				.cover_level_values(level)
				.expect("the pyramid is held");
			let texels = (0..*down).flat_map(|row| (0..*across).map(move |column| (column, row)));

			for (column, row) in texels {
				let made = values
					.get(usize::try_from(row * stride + column).unwrap_or(usize::MAX))
					.copied();
				let worked = wanted
					.get(usize::try_from(row * across + column).unwrap_or(usize::MAX))
					.copied();

				assert!(
					made.zip(worked)
						.is_some_and(|(made, worked)| made.to_bits() == worked.to_bits()),
					"level {level} texel ({column}, {row}) is {made:?} and {worked:?} worked \
					 out here"
				);
			}
		}
	}

	/// Whether a box is behind a pyramid, worked out here the way `behind` in
	/// `cover.wgsl` works it out, or nothing where rounding could answer either
	/// way: a corner landing within a thousandth of a pixel's edge, or a
	/// nearest corner within two millionths of the farthest depth and the
	/// slack.
	fn behind_here(
		reach: &Reach,
		projection: Mat4,
		view: Viewport,
		levels: &[(u32, Vec<f32>)],
	) -> Option<bool> {
		let rect = [view.x, view.y, view.width, view.height].map(pixels);
		let (mut low, mut high, mut nearest) = ([f32::MAX; 2], [f32::MIN; 2], 1.0_f32);

		for corner in 0..8_u8 {
			let way = |bit: u8| if corner & bit == 0 { -1.0 } else { 1.0 };
			let at = Vec3::from_array(reach.center)
				+ Vec3::from_array(reach.x) * way(1)
				+ Vec3::from_array(reach.y) * way(2)
				+ Vec3::from_array(reach.z) * way(4);
			let clip = projection * Vec4::new(at.x, at.y, at.z, 1.0);

			if clip.w <= 0.0 || clip.z < 0.0 {
				return Some(false);
			}

			let ndc = clip.truncate() / clip.w;
			let pixel = [
				ndc.x.mul_add(0.5, 0.5).mul_add(rect[2], rect[0]),
				(-ndc.y)
					.mul_add(0.5, 0.5)
					.mul_add(rect[3], rect[1]),
			];

			if pixel
				.iter()
				.any(|place| (place - place.round()).abs() < 1.0e-3)
			{
				return None;
			}

			low = [low[0].min(pixel[0]), low[1].min(pixel[1])];
			high = [high[0].max(pixel[0]), high[1].max(pixel[1])];
			nearest = nearest.min(ndc.z);
		}

		let span = [rect[2], rect[3]];
		let lowest = [low[0] - rect[0], low[1] - rect[1]];
		let highest = [high[0] - rect[0], high[1] - rect[1]];

		if highest[0] < 0.0 || highest[1] < 0.0 || lowest[0] >= span[0] || lowest[1] >= span[1] {
			return Some(false);
		}

		let clamped = |value: f32, axis: usize| value.clamp(0.0, span[axis] - 1.0);
		let first: [u32; 2] =
			[0, 1].map(|axis| whole_pixels(clamped(lowest[axis].floor() - 1.0, axis)));
		let last: [u32; 2] =
			[0, 1].map(|axis| whole_pixels(clamped(highest[axis].floor() + 1.0, axis)));
		let side = (last[0] - first[0]).max(last[1] - first[1]) + 1;
		let fits = 32 - (side - 1).leading_zeros();
		let index = (fits.max(3) - 3).min(u32::try_from(levels.len()).unwrap_or(1) - 1);
		let (across, down) = level_size(index, view);
		let (stride, texels) = levels.get(usize::try_from(index).ok()?)?;
		let mut farthest = 0.0_f32;

		for row in (first[1] >> (index + 1))..=(last[1] >> (index + 1)).min(down - 1) {
			for column in (first[0] >> (index + 1))..=(last[0] >> (index + 1)).min(across - 1) {
				farthest =
					farthest.max(*texels.get(usize::try_from(row * stride + column).ok()?)?);
			}
		}

		if (nearest - (farthest + 1.0e-6)).abs() < 2.0e-6 {
			return None;
		}

		Some(nearest > farthest + 1.0e-6)
	}

	/// A place in pixels held in a rectangle, as a whole count.
	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "held inside a picture a few hundred pixels across and never below nought"
	)]
	fn whole_pixels(value: f32) -> u32 { value as u32 }

	/// A run of numbers between nought and one, the same run for the same seed.
	fn xorshift(mut seed: u32) -> impl FnMut() -> f32 {
		move || {
			seed ^= seed << 13;
			seed ^= seed >> 17;
			seed ^= seed << 5;

			f32::from(u16::try_from(seed >> 16).unwrap_or(0)) / 65_535.0
		}
	}

	/// What a frame drawn into a rectangle with the test on left out, held
	/// thing by thing against [`behind_here`] over the pyramid the device made,
	/// and its picture against the same frame drawn with the test off.
	///
	/// @param samples - how many samples a pixel both frames take
	/// @return how many answers agreed, how many were set aside, and how many
	/// of those that agreed were left out
	fn held_against_here(
		capture: &mut Capture,
		world: &mut World,
		view: Viewport,
		samples: &str,
	) -> (u32, usize, u32) {
		sampled(world, samples);
		covering(world, true);

		let left_out = capture
			.shoot_within(world, view)
			.expect("the capture renders");
		let scene = capture.scene_mut();
		let kept = scene.cover_kept_values().expect("the test ran");
		let levels: Vec<(u32, Vec<f32>)> = (0..levels_over(view))
			.map(|level| {
				scene
					.cover_level_values(level)
					.expect("the pyramid is held")
			})
			.collect();
		let projection = scene.cover_projection();
		let answers: Vec<(usize, Option<bool>)> = scene
			.cover_reaches()
			.iter()
			.map(|reach| behind_here(reach, projection, view, &levels))
			.enumerate()
			.collect();
		let aside = answers
			.iter()
			.filter(|(_, here)| here.is_none())
			.count();
		let (mut agreed, mut hidden) = (0, 0);

		for (index, here) in answers
			.into_iter()
			.filter_map(|(index, here)| here.map(|here| (index, here)))
		{
			let there = kept.get(index).copied() == Some(0);

			assert_eq!(here, there, "at {samples} samples thing {index} of the lists");
			agreed += 1;
			hidden += u32::from(there);
		}

		covering(world, false);

		let whole = capture
			.shoot_within(world, view)
			.expect("the capture renders");

		assert_eq!(
			apart(&left_out, &whole),
			0,
			"at {samples} samples the picture is the one with everything drawn"
		);

		(agreed, aside, hidden)
	}

	#[test]
	fn every_box_the_test_keeps_or_leaves_out_is_what_the_same_arithmetic_says_here() {
		// the arithmetic done twice over the very pyramid the device made: a
		// crowd of boxes of every size turned every way, in front of the wall,
		// behind it, across its ends, off the picture and through the near
		// plane, more than one chunk of the count's worth
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = street(0.8);
		let view = Viewport { x: 23, y: 11, width: 283, height: 211 };
		let mut next = xorshift(0x2545_F491);

		for _ in 0..120 {
			let position = Vec3::new(
				next().mul_add(16.0, -8.0),
				next().mul_add(3.0, -0.2),
				next().mul_add(16.0, -12.0),
			);
			let scale = Vec3::new(
				next().mul_add(1.5, 0.05),
				next().mul_add(1.5, 0.05),
				next().mul_add(1.5, 0.05),
			);
			let rotation =
				Quat::from_euler(EulerRot::YXZ, next() * TAU, next() * TAU, next() * TAU);

			stood(&mut world, MaterialId::NONE, Transform { position, rotation, scale }, GREEN);
		}

		// long boxes pushed through the wall at its middle height and turned a
		// sixth of a turn apart, so that the corner the shader reaches last is
		// the nearest for some and behind the wall for others
		for turn in 0..6_u8 {
			let turned = f32::from(turn);

			stood(
				&mut world,
				MaterialId::NONE,
				Transform {
					position: Vec3::new(turned.mul_add(1.2, -3.0), 2.0, 0.0),
					rotation: Quat::from_euler(EulerRot::YXZ, turned * TAU / 6.0, 0.4, 0.0),
					scale: Vec3::new(0.5, 0.5, 2.0),
				},
				BLUE,
			);
		}

		for samples in ["1", "4"] {
			let (agreed, aside, left_out) =
				held_against_here(&mut capture, &mut world, view, samples);

			assert!(agreed > 60, "{agreed} answers compared and {aside} set aside");
			assert!(left_out > 10, "and the wall hid {left_out} of them");
		}
	}

	#[test]
	fn a_box_a_part_of_a_pixel_inside_a_wall_s_edge_is_what_the_same_arithmetic_says_here() {
		// boxes behind all four edges of a wall with only the sky past them,
		// each placed so that its side nearest the edge lands a random part of
		// a pixel to a few pixels inside the line the eye sees the edge along,
		// or a little past it, and small enough to be read at the finest level
		// or the next: where growing the rectangle by a pixel and the level it
		// is read at decide. The eye is off the wall's middle so that no edge
		// lies along a texel's boundary.
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();
		let view = Viewport {
			x: 0,
			y: 0,
			width: SIZE.0,
			height: SIZE.1,
		};
		let eye = Vec3::new(0.37, 2.23, 14.0);
		let across = 2.0 * (world.camera.fov_y * 0.5).tan() / pixels(SIZE.1);
		let mut next = xorshift(0x9E37_79B9);

		world.camera.position = eye;
		world.camera.target = Vec3::new(eye.x, eye.y, 0.0);
		slab(
			&mut world,
			MaterialId::NONE,
			(Vec3::new(0.0, 2.0, 0.0), Vec3::new(8.0, 4.0, 0.5)),
			GREY,
		);

		for side in 0..240_u16 {
			let along = next().mul_add(2.0, -1.0);
			// a point on the edge's front, and which way is into the wall
			let (edge, inward) = match side % 4 {
				| 0 => (Vec3::new(along * 3.4, 4.0, 0.25), Vec3::NEG_Y),
				| 1 => (Vec3::new(along * 3.4, 0.0, 0.25), Vec3::Y),
				| 2 => (Vec3::new(4.0, along.mul_add(1.4, 2.0), 0.25), Vec3::NEG_X),
				| _ => (Vec3::new(-4.0, along.mul_add(1.4, 2.0), 0.25), Vec3::X),
			};
			let distance = next().mul_add(5.0, 14.5);
			let size = Vec3::new(
				next().mul_add(0.85, 0.05),
				next().mul_add(0.85, 0.05),
				next().mul_add(0.85, 0.05),
			);
			let inside = next().mul_add(6.0, -1.5) * across * distance;
			// where the eye's line past the edge is at that distance, moved inward
			let front = eye + (edge - eye) * (distance / (eye.z - edge.z)) + inward * inside;
			let position =
				front + inward * (inward.abs().dot(size) * 0.5) - Vec3::Z * (size.z * 0.5);

			slab(&mut world, MaterialId::NONE, (position, size), GREEN);
		}

		for samples in ["1", "4"] {
			let (agreed, aside, left_out) =
				held_against_here(&mut capture, &mut world, view, samples);

			assert!(agreed > 200, "{agreed} answers compared and {aside} set aside");
			assert!(
				left_out > 60 && agreed - left_out > 60,
				"and of them {left_out} were left out, which has to be neither few nor most"
			);
		}
	}

	#[test]
	fn a_box_a_centimeter_behind_a_wall_fifty_units_away_is_drawn_and_one_six_behind_is_not() {
		// the slack's own number: a millionth of depth fifty units from an eye
		// whose near plane is a tenth away is two and a half centimeters, so of
		// two boxes behind a wall square to the eye and wider than the view the
		// one a centimeter behind the wall's face is kept and the one six
		// behind it is left out
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();

		world.camera.position = Vec3::new(0.0, 0.0, 50.0);
		world.camera.target = Vec3::ZERO;
		world.camera.near = 0.1;
		world.camera.far = 200.0;

		// five millimeters thick, its face at nought
		slab(
			&mut world,
			MaterialId::NONE,
			(Vec3::new(0.0, 0.0, -0.0025), Vec3::new(120.0, 90.0, 0.005)),
			GREY,
		);
		slab(
			&mut world,
			MaterialId::NONE,
			(Vec3::new(-6.0, 0.0, -1.01), Vec3::splat(2.0)),
			GREEN,
		);
		slab(
			&mut world,
			MaterialId::NONE,
			(Vec3::new(6.0, 0.0, -1.06), Vec3::splat(2.0)),
			BLUE,
		);
		sampled(&mut world, "1");
		covering(&mut world, true);

		let (_, drawn) = shot(&mut capture, &mut world);

		assert_eq!(drawn.seen, 3, "the wall and both boxes are in the view");
		assert_eq!(
			(drawn.covered, drawn.covered_triangles),
			(1, 12),
			"the box six centimeters behind is left out and the one a centimeter behind is kept"
		);
	}

	#[test]
	fn a_pane_of_glass_behind_a_wall_is_left_out_and_two_in_front_of_it_are_not() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = street(0.8);
		let glass = world.materials.insert("test/glass", Material {
			blend: Blend::Alpha,
			opacity: 0.4,
			..Material::DEFAULT
		});

		slab(&mut world, glass, (Vec3::new(0.0, 1.0, -2.0), Vec3::new(2.0, 1.5, 0.1)), BLUE);
		sampled(&mut world, "4");
		covering(&mut world, true);

		let (left_out, drawn) = shot(&mut capture, &mut world);

		assert_eq!(drawn.covered, 1, "the pane behind the wall is left out");

		covering(&mut world, false);

		let (whole, _) = shot(&mut capture, &mut world);

		assert_eq!(apart(&left_out, &whole), 0, "and the picture is the same to the bit");

		// two in front, so that the blended batch keeps a different count from
		// the solid batch the wall is in and a command read from the wrong list
		// draws the wrong number of panes
		slab(&mut world, glass, (Vec3::new(-1.2, 1.0, 2.0), Vec3::new(2.0, 1.5, 0.1)), BLUE);
		slab(&mut world, glass, (Vec3::new(1.2, 1.0, 2.5), Vec3::new(2.0, 1.5, 0.1)), GREEN);
		covering(&mut world, true);

		let (in_front, drawn) = shot(&mut capture, &mut world);

		assert_eq!(
			drawn.covered, 1,
			"the panes in front of the wall are drawn, the one behind is not"
		);

		covering(&mut world, false);

		let (whole, _) = shot(&mut capture, &mut world);

		assert_eq!(apart(&in_front, &whole), 0, "and the picture is still the same to the bit");
	}

	#[test]
	fn a_small_box_behind_a_wall_is_left_out_of_the_pass_before_the_scene_too_and_no_bit_it_wrote_moves()
	 {
		// two boxes too small to be drawn ahead of the test, one behind the wall
		// and one in front of it, beside a box that is not small: the pass before
		// the scene draws the large things, the test runs, and the pass draws the
		// small box it kept - and every buffer it wrote is the one it writes
		// drawing everything ahead of the test, and the one it writes with no
		// test at all
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = street(0.8);
		let glass = world.materials.insert("test/glass", Material {
			blend: Blend::Alpha,
			opacity: 0.4,
			..Material::DEFAULT
		});
		let small = Vec3::splat(0.2);

		// the large box between the two small ones in slot order, so that only a
		// key that puts the small ones after the large ones makes them one batch
		slab(&mut world, MaterialId::NONE, (Vec3::new(0.5, 0.5, -3.0), small), GREEN);
		cube(&mut world, Vec3::new(3.0, 0.5, 2.0), BLUE);
		slab(&mut world, MaterialId::NONE, (Vec3::new(1.2, 0.3, 3.0), small), GREEN);
		// and a pane as small, which that pass never draws at all
		slab(&mut world, glass, (Vec3::new(-1.5, 0.5, 2.5), Vec3::new(0.2, 0.2, 0.05)), BLUE);

		for samples in ["1", "4"] {
			sampled(&mut world, samples);
			covering(&mut world, true);
			sized(&mut world, "32");

			let (halves, drawn) = shot(&mut capture, &mut world);
			let passes = capture.scene_mut().spans().passes();
			let maps = capture.scene_mut().map_batches();
			let bits = written(&mut capture);

			assert_eq!(
				(drawn.small, drawn.covered),
				(2, 1),
				"at {samples} samples both small boxes are drawn after the test, and the one \
				 behind the wall is left out"
			);
			assert_eq!(
				capture.scene_mut().draws(),
				[(2, 0), (0, 1), (0, 3), (0, 1)],
				"the wall's and the floor's batches ahead of the test, the small boxes' batch \
				 through what it kept, and the scene's two lists through it as well"
			);

			sized(&mut world, "0");

			let (ahead, drawn) = shot(&mut capture, &mut world);

			assert_eq!((drawn.small, drawn.covered), (0, 1), "nought draws everything ahead");
			assert_eq!(
				capture.scene_mut().draws(),
				[(2, 0), (0, 2), (0, 1)],
				"in one pass of two batches, the box and the wall in one of them"
			);
			assert_eq!(
				capture.scene_mut().spans().passes(),
				passes - 1,
				"which is one pass fewer"
			);
			assert_eq!(
				capture.scene_mut().map_batches(),
				maps,
				"and every shadow map was cut into the same batches, large and small alike"
			);
			assert!(
				written(&mut capture) == bits,
				"at {samples} samples the pass before the scene wrote the same bits in two \
				 halves as in one"
			);
			assert_eq!(apart(&halves, &ahead), 0, "and the picture is the same to the bit");

			covering(&mut world, false);
			sized(&mut world, "32");

			let (whole, drawn) = shot(&mut capture, &mut world);

			assert_eq!(
				(drawn.small, drawn.covered),
				(0, 0),
				"with the test off nothing is small and nothing is left out"
			);
			assert_eq!(
				capture.scene_mut().draws(),
				[(2, 0), (2, 0), (1, 0)],
				"and every list is drawn through its own placements"
			);
			assert!(written(&mut capture) == bits, "that pass writes the same bits again");
			assert_eq!(apart(&halves, &whole), 0, "and the picture is the same again");
		}
	}

	#[test]
	fn a_small_box_out_from_behind_a_wall_and_up_to_the_eye_is_in_every_frame_it_shows_and_in_the_mirror()
	 {
		// the series above for a box small enough to be drawn after the test:
		// out past the end of a wall and then towards the eye until it is small
		// no more, over a mirror floor, so that the pass before the scene has to
		// hold the box in the very frame it shows. One scene leaving things out
		// every frame against one drawing everything, frame by frame
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut kept = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut whole = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = World::new();

		world.camera.position = Vec3::new(0.0, 1.5, 6.0);
		world.camera.target = Vec3::new(0.0, 1.0, 0.0);
		world.light = Vec3::new(-0.4, -1.0, -0.3).normalize();

		let floor = world
			.materials
			.insert("test/floor", Material { roughness: 0.045, ..Material::DEFAULT });

		slab(
			&mut world,
			floor,
			(Vec3::new(0.0, -0.5, 0.0), Vec3::new(40.0, 1.0, 40.0)),
			Vec3::splat(0.35),
		);
		// four wide and ending at one, so that the box shows past its end well
		// inside the view
		slab(
			&mut world,
			MaterialId::NONE,
			(Vec3::new(-1.0, 2.0, 0.0), Vec3::new(4.0, 4.0, 0.5)),
			GREY,
		);

		let moving = slab(
			&mut world,
			MaterialId::NONE,
			(Vec3::new(-1.0, 0.175, -1.5), Vec3::splat(0.35)),
			GREEN,
		);
		// behind the wall and out past its end, and then towards the eye until
		// the box is nearer than it has to be to stand thirty-two pixels across
		let places: Vec<Vec3> = (0..16_u8)
			.map(|step| Vec3::new(f32::from(step).mul_add(0.3, -1.0), 0.175, -1.5))
			.chain((1..=16_u8).map(|step| {
				Vec3::new(3.5, 0.175, -1.5)
					.lerp(Vec3::new(1.3, 0.175, 3.3), f32::from(step) / 16.0)
			}))
			.collect();
		let (mut hidden, mut small, mut large) = (0, 0, 0);

		sampled(&mut world, "4");
		sized(&mut world, "32");

		for (step, at) in places.into_iter().enumerate() {
			moved(&mut world, moving, at);
			covering(&mut world, true);

			let (left_out, drawn) = shot(&mut kept, &mut world);

			covering(&mut world, false);

			let (drawn_whole, _) = shot(&mut whole, &mut world);

			assert_eq!(
				apart(&left_out, &drawn_whole),
				0,
				"at step {step} the scene that has been leaving things out draws the picture \
				 the scene drawing everything draws"
			);

			match (drawn.covered, drawn.small) {
				| (1, 1) => hidden += 1,
				| (0, 1) => small += 1,
				| (0, 0) => large += 1,
				| counts =>
					panic!("at step {step} the box alone can be small or left out: {counts:?}"),
			}
		}

		assert!(
			hidden >= 3 && small >= 3 && large >= 3,
			"the box was behind the wall for {hidden} frames, out and small for {small} and \
			 large for {large}"
		);
	}

	#[test]
	fn what_only_a_small_thing_hides_is_drawn_and_the_picture_is_the_one_that_left_it_out() {
		// the price of the split: a box small enough to be drawn after the test
		// hides a smaller one behind it, and the test reads only what the large
		// things drew, so the one behind is drawn - where drawing everything
		// ahead of the test leaves it out. Either way the same picture
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();

		world.camera.position = Vec3::new(0.0, 1.5, 6.0);
		world.camera.target = Vec3::new(0.0, 1.0, 0.0);

		let floor = world
			.materials
			.insert("test/floor", Material::DEFAULT);

		slab(
			&mut world,
			floor,
			(Vec3::new(0.0, -0.5, 0.0), Vec3::new(40.0, 1.0, 40.0)),
			Vec3::splat(0.35),
		);
		// a unit box fourteen away is twenty-eight pixels across in this picture,
		// and a quarter of one behind it on the line from the eye is six
		cube(&mut world, Vec3::new(0.0, 0.5, -8.0), GREY);
		slab(
			&mut world,
			MaterialId::NONE,
			(Vec3::new(0.0, 0.321, -10.5), Vec3::splat(0.25)),
			GREEN,
		);

		for samples in ["1", "4"] {
			sampled(&mut world, samples);
			covering(&mut world, true);
			sized(&mut world, "32");

			let (split, drawn) = shot(&mut capture, &mut world);

			assert_eq!(
				(drawn.small, drawn.covered),
				(2, 0),
				"at {samples} samples both boxes are small, and nothing large hides the one \
				 behind"
			);
			assert_eq!(
				capture.scene_mut().draws(),
				[(1, 0), (0, 1), (0, 2), (0, 0)],
				"the floor ahead of the test and both boxes after it"
			);

			sized(&mut world, "0");

			let (ahead, drawn) = shot(&mut capture, &mut world);

			assert_eq!(
				(drawn.small, drawn.covered),
				(0, 1),
				"drawn ahead of the test, the nearer box hides the one behind"
			);
			assert_eq!(apart(&split, &ahead), 0, "and the pictures are the same to the bit");
		}
	}

	#[test]
	fn a_world_of_nothing_but_small_things_draws_them_all_ahead_of_the_test() {
		// the same two boxes with no floor under them: nothing solid is large,
		// and a pyramid of nothing would leave nothing out of either pass - so
		// both are drawn ahead of the test, and the nearer hides the one behind
		// as it does with no split at all. A pane of glass beside them is large
		// and changes none of it, because it is never drawn into that pass
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = World::new();
		let glass = world.materials.insert("test/glass", Material {
			blend: Blend::Alpha,
			opacity: 0.4,
			..Material::DEFAULT
		});

		world.camera.position = Vec3::new(0.0, 1.5, 6.0);
		world.camera.target = Vec3::new(0.0, 1.0, 0.0);
		cube(&mut world, Vec3::new(0.0, 0.5, -8.0), GREY);
		slab(
			&mut world,
			MaterialId::NONE,
			(Vec3::new(0.0, 0.321, -10.5), Vec3::splat(0.25)),
			GREEN,
		);
		slab(&mut world, glass, (Vec3::new(-3.0, 1.0, 0.0), Vec3::new(2.0, 1.5, 0.1)), BLUE);
		sampled(&mut world, "1");
		covering(&mut world, true);
		sized(&mut world, "32");

		let (left_out, drawn) = shot(&mut capture, &mut world);
		let passes = capture.scene_mut().spans().passes();

		assert_eq!((drawn.small, drawn.covered), (0, 1), "nothing is small, and one is behind");
		assert_eq!(
			capture.scene_mut().draws(),
			[(1, 0), (0, 1), (0, 1)],
			"one pass before the scene, then the scene's two lists"
		);

		sized(&mut world, "0");

		let (ahead, drawn) = shot(&mut capture, &mut world);

		assert_eq!(drawn.covered, 1, "and drawing everything ahead leaves out the same");
		assert_eq!(capture.scene_mut().spans().passes(), passes, "in as many passes");
		assert_eq!(apart(&left_out, &ahead), 0, "and draws the same picture");
	}

	#[test]
	fn two_boxes_of_one_batch_at_one_depth_are_drawn_in_the_order_they_were_with_one_left_out_between()
	 {
		// the one thing the copy's order decides: two things at exactly the same
		// depth, where the first drawn keeps the pixel. The one sorted between
		// them is hidden, so the second is copied one place earlier than it was
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = street(0.8);

		cube(&mut world, Vec3::new(-1.0, 0.5, 2.0), GREEN);
		cube(&mut world, Vec3::new(0.0, 0.5, -3.0), GREY);
		cube(&mut world, Vec3::new(-1.0, 0.5, 2.0), BLUE);

		for samples in ["1", "4"] {
			sampled(&mut world, samples);
			covering(&mut world, true);

			let (left_out, drawn) = shot(&mut capture, &mut world);

			assert_eq!(drawn.covered, 1, "the middle one is behind the wall");

			covering(&mut world, false);

			let (whole, _) = shot(&mut capture, &mut world);

			assert_eq!(apart(&left_out, &whole), 0, "at {samples} samples the first still wins");
		}
	}

	#[test]
	fn nothing_is_left_out_while_nothing_reads_the_pass_before_the_scene_or_the_frustum_test_is_off()
	 {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = street(0.8);

		cube(&mut world, Vec3::new(0.0, 0.5, -3.0), GREEN);
		sampled(&mut world, "1");
		covering(&mut world, true);

		let (_, drawn) = shot(&mut capture, &mut world);
		let passes = capture.scene_mut().spans().passes();

		assert_eq!(drawn.covered, 1, "with the defaults the box is left out");
		assert!(capture.scene_mut().cover_state().holding(), "and the pyramid is held");

		// no share of the sky and no reflections: no pass before the scene to read
		world
			.cvars
			.var(occlusion::STRENGTH, Value::Float(occlusion::DEFAULT_STRENGTH), "");
		world.cvars.set(occlusion::STRENGTH, "0");
		world
			.cvars
			.var(reflection::STRENGTH, Value::Float(reflection::DEFAULT_STRENGTH), "");
		world.cvars.set(reflection::STRENGTH, "0");

		let (_, drawn) = shot(&mut capture, &mut world);

		assert_eq!(drawn.covered, 0, "nothing is left out with no depth to test against");
		assert!(!capture.scene_mut().cover_state().holding(), "and the pyramid is let go");
		assert!(
			capture.scene_mut().cover_reaches().is_empty(),
			"and no record of the lists is laid down for a test that will not run"
		);
		assert_eq!(
			capture.scene_mut().spans().passes(),
			passes - 7,
			"which is the pass before the scene, the share's two, the reflections' three and \
			 this one fewer, and no more"
		);

		// the share back on, and the frustum test off
		world.cvars.set(occlusion::STRENGTH, "1");
		world
			.cvars
			.var(cull::ENABLED, Value::Bool(true), "");
		world.cvars.set(cull::ENABLED, "false");

		let (_, drawn) = shot(&mut capture, &mut world);

		assert_eq!(drawn.covered, 0, "and nothing is left out with the frustum test off");
		assert!(capture.scene_mut().cover_reaches().is_empty(), "nor laid down for it");
	}

	#[test]
	fn a_frame_that_runs_no_test_after_one_that_did_draws_its_own_lists() {
		// the kept run and its commands stay what the last test wrote until the
		// next one runs, so a frame without it has to draw its own placements:
		// here the box the first frame left out has come round in front of the
		// wall by the second
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut switched = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut whole = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = street(0.8);
		let moving = cube(&mut world, Vec3::new(0.0, 0.5, -3.0), GREEN);

		sampled(&mut world, "1");
		covering(&mut world, true);

		let (_, drawn) = shot(&mut switched, &mut world);

		assert_eq!(drawn.covered, 1, "the box behind the wall is left out");

		// the other scene draws the first frame too, so that both have the same
		// history behind the second
		covering(&mut world, false);
		shot(&mut whole, &mut world);
		moved(&mut world, moving, Vec3::new(2.5, 0.5, 2.0));

		let (after, drawn) = shot(&mut switched, &mut world);
		let (fresh, _) = shot(&mut whole, &mut world);

		assert_eq!(drawn.covered, 0, "the second frame leaves nothing out");
		assert_eq!(
			apart(&after, &fresh),
			0,
			"and draws the box where it stands now, as a scene that never ran the test does"
		);
	}

	#[test]
	fn a_window_that_polls_is_told_what_was_left_out_a_few_frames_later() {
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = street(0.8);

		cube(&mut world, Vec3::new(0.0, 0.5, -3.0), GREEN);
		cube(&mut world, Vec3::new(1.5, 0.5, -3.0), GREEN);

		let mut told = None;

		for _ in 0..20 {
			capture.draw(&mut world, &mut []);
			let _landed = capture.scene_mut().collect();

			if capture.scene_mut().drawn().covered > 0 {
				told = Some(capture.scene_mut().drawn());

				break;
			}
		}

		let told = told.expect("the counts landed inside twenty frames without anything waiting");

		assert_eq!(told.covered, 2, "both boxes behind the wall: {told:?}");
	}

	#[test]
	fn a_picture_made_again_at_another_size_leaves_out_what_one_made_at_that_size_does() {
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		// made small and grown, so that a pyramid kept from the smaller picture
		// would be a level short, and then made small again; the target stays
		// the size it was made at, so every frame is drawn at one aspect
		let mut resized = Capture::new(gpu, 240, 180).expect("the capture builds");
		let mut fresh = Capture::new(gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = street(0.8);
		let covered = |capture: &mut Capture, world: &mut World| {
			capture.draw(world, &mut []);
			capture.scene_mut().settle();
			capture.scene_mut().drawn().covered
		};

		for along in -2..=2_i8 {
			cube(&mut world, Vec3::new(f32::from(along) * 1.2, 0.5, -2.5), GREEN);
		}

		sampled(&mut world, "1");

		let small = covered(&mut resized, &mut world);

		resized.scene_mut().resize(SIZE.0, SIZE.1);

		let grown = covered(&mut resized, &mut world);

		resized.scene_mut().resize(240, 180);

		let shrunk = covered(&mut resized, &mut world);

		assert_eq!(covered(&mut fresh, &mut world), 5, "the wall hides all five");
		assert_eq!(
			(small, grown, shrunk),
			(5, 5, 5),
			"and the pyramid made again at each new size answers as one made at it"
		);
	}

	#[test]
	fn the_count_s_chunks_cover_the_most_things_a_world_holds() {
		let source = include_str!("cover.wgsl");
		let chunk = format!("const CHUNK: u32 = {CHUNK}u;");
		let chunks = format!("const CHUNKS: u32 = {}u;", MAX_ENTITIES.div_ceil(16));

		assert_eq!(source.matches(&chunk).count(), 1, "the shader's chunk is this module's");
		assert_eq!(source.matches(&chunks).count(), 1, "and it has as many as the lists need");
	}
}
