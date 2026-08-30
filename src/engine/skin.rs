//! The joint matrices of every posed character in a frame, in one buffer.
//!
//! **One storage buffer for the whole frame, not one binding per character.**
//! The scene batches by mesh and material and draws a run of instances per
//! batch; a uniform block per character would mean a bind group per character,
//! which is a draw call per character, which is the batching thrown away. So
//! every pose's matrices go into one buffer back to back and each instance
//! carries the offset of its own run - @ref
//! [`Placement`](crate::scene::Placement). That is what the field settled on
//! once storage buffers were available everywhere it cared about.
//!
//! **It cannot overflow.** There are at most
//! [`MAX_POSES`](colby_core::abi::pose::MAX_POSES) poses and a pose writes one
//! matrix per bone of its skeleton, which the loader caps at
//! [`MAX_BONES`](colby_core::abi::skeleton::MAX_BONES). The buffer is sized for
//! the product, so no frame can ask for more than it holds and there is no
//! partly drawn character to decide what to do about.
//!
//! **A matrix is written once per frame even if six entities share the pose.**
//! Two entities of one character - a model of two materials is exactly that -
//! look their run up by the pose's slot and find the same offset.

use colby_core::{
	Result,
	abi::{PoseId, World, pose::MAX_POSES, skeleton::MAX_BONES},
	bytemuck, err,
	glam::Mat4,
};
use wgpu::{
	BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
	BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
	Device, Queue, ShaderStages,
};

/// What an instance holds when nothing moves it.
///
/// Zero is a perfectly good offset, so "no bones" cannot be one: the count
/// beside it is what says so, and this is only ever read as a pair.
pub(crate) const NO_JOINTS: [u32; 4] = [0, 0, 0, 0];

/// How many matrices the buffer holds.
///
/// Every pose the world can have, at the widest skeleton it can wear. Four
/// megabytes, allocated once, against a shadow map array that is four times
/// that - and in exchange the gather can never run out and there is no
/// half-drawn character to have a policy for.
const MATRICES: usize = MAX_POSES * MAX_BONES;

/// The frame's joint matrices, and where each pose's run starts in them.
pub(crate) struct Joints {
	buffer: Buffer,
	layout: BindGroupLayout,
	bindings: BindGroup,
	/// What [`World::render_skinning`] appends into, kept so it allocates once.
	gathered: Vec<Mat4>,
	/// The same as the bytes the GPU reads. Two buffers because glam's matrix
	/// is not plain data here - the library is built without bytemuck, which
	/// is what keeps a physics type from accidentally becoming castable.
	written: Vec<[[f32; 4]; 4]>,
	/// Where each pose slot's run starts and how long it is, or
	/// [`NO_JOINTS`] for a slot whose pose nothing in this frame draws.
	///
	/// The length is stored rather than worked out from where the next run
	/// begins: a pose asked for twice is asked for once before other poses
	/// have been gathered and once after, and the distance to the end of the
	/// buffer is only its length the first time.
	runs: Vec<[u32; 4]>,
}

impl Joints {
	/// Builds the buffer, its layout and the group that binds it.
	///
	/// @param device - the device to build against
	/// @return the table, or why the buffer could not be described
	pub(crate) fn new(device: &Device) -> Result<Self> {
		let size = u64::try_from(MATRICES * size_of::<[[f32; 4]; 4]>())
			.map_err(|_| err!(Graphics("the joint buffer is too large to describe")))?;
		let buffer = device.create_buffer(&BufferDescriptor {
			label: Some("joints"),
			size,
			usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("joints"),
			entries: &[BindGroupLayoutEntry {
				binding: 0,
				// read in the vertex stage and nowhere else: a skinned vertex
				// is moved before anything shades it, and the fragment stage
				// is handed the result rather than the bones.
				visibility: ShaderStages::VERTEX,
				ty: BindingType::Buffer {
					ty: BufferBindingType::Storage { read_only: true },
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			}],
		});
		let bindings = device.create_bind_group(&BindGroupDescriptor {
			label: Some("joints"),
			layout: &layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: buffer.as_entire_binding(),
			}],
		});

		Ok(Self {
			buffer,
			layout,
			bindings,
			gathered: Vec::new(),
			written: Vec::new(),
			runs: Vec::new(),
		})
	}

	/// The layout a pipeline declares to read these.
	pub(crate) const fn layout(&self) -> &BindGroupLayout { &self.layout }

	/// The group to bind before drawing anything skinned.
	pub(crate) const fn bindings(&self) -> &BindGroup { &self.bindings }

	/// Forgets last frame's matrices and sizes the table for this world.
	///
	/// @param world - the world about to be drawn
	pub(crate) fn begin(&mut self, world: &World) {
		self.gathered.clear();
		self.runs.clear();
		self.runs.resize(world.poses.slots(), NO_JOINTS);
	}

	/// The offset one pose's run starts at, gathering it if this is the first
	/// entity of the frame to ask.
	///
	/// @param world - the world being drawn, for the pose and its skeleton
	/// @param id - the pose an entity named
	/// @return `[offset, bones, 0, 0]`, or [`NO_JOINTS`] when nothing moves it
	pub(crate) fn take(&mut self, world: &World, id: PoseId) -> [u32; 4] {
		let Some(run) = self
			.runs
			.get(id.slot())
			.copied()
			.filter(|_| world.poses.alive(id))
		else {
			return NO_JOINTS;
		};

		// somebody already asked this frame, and the answer does not change:
		// one character drawn as two entities is one run of matrices.
		if run != NO_JOINTS {
			return run;
		}

		let Ok(at) = u32::try_from(self.gathered.len()) else {
			return NO_JOINTS;
		};

		let bones = world.render_skinning(id, &mut self.gathered);
		let Ok(bones) = u32::try_from(bones) else {
			return NO_JOINTS;
		};

		// cannot happen: the buffer is sized for every pose the world can hold
		// at the widest skeleton the loader lets through. Cutting it back
		// rather than trusting that is four lines, and what the alternative
		// costs if the arithmetic above ever changes is a write past the end
		// of a buffer the GPU is reading.
		if self.gathered.len() > MATRICES {
			self.gathered.truncate(MATRICES);

			return NO_JOINTS;
		}

		if bones == 0 {
			return NO_JOINTS;
		}

		let run = [at, bones, 0, 0];

		if let Some(entry) = self.runs.get_mut(id.slot()) {
			*entry = run;
		}

		run
	}

	/// Writes what was gathered.
	///
	/// @param queue - the queue to write through
	pub(crate) fn upload(&mut self, queue: &Queue) {
		self.written.clear();
		self.written
			.extend(self.gathered.iter().map(Mat4::to_cols_array_2d));

		if self.written.is_empty() {
			return;
		}

		queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.written));
	}

	/// How many matrices this frame has gathered so far.
	///
	/// For the one claim about this table that a picture cannot show: a pose
	/// six entities share is gathered once. That is not a saving, it is what
	/// makes the buffer's size a bound - gathering per entity rather than per
	/// pose would ask for
	/// [`MAX_ENTITIES`](colby_core::abi::MAX_ENTITIES) runs, which is four
	/// times what is allocated.
	#[cfg(test)]
	pub(crate) fn len(&self) -> usize { self.gathered.len() }
}
