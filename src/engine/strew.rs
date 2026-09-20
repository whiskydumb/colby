//! What a strewing laid, drawn: one buffer of placements a strewing, and its
//! patches asked of every volume a frame has.
//!
//! **The copies are not entities and are not in the frame's one instance
//! buffer.** That buffer holds room for every entity in every list a frame
//! draws it in - a thousand entities and twenty-one lists - and one strewing
//! lays up to a quarter of a million copies. So each strewing gets a buffer of
//! its own, built from what [`strew`](colby_core::abi::strew) laid and bound by
//! the draws that read it.
//!
//! **Built when something it was built from changes, and not once a frame.**
//! The layout has a revision; the ground it stands on has a drawn transform;
//! the entity has a mesh, a material and a tint. Those are the whole of what a
//! placement holds, so comparing them bit for bit is the whole of whether the
//! buffer is still true. A world whose ground stands still builds each buffer
//! once, because a transform interpolated between two equal steps is the same
//! transform bit for bit. @ref [`Built`].
//!
//! **A copy is placed the way a child is placed.** The ground's drawn transform
//! carried onto the copy's own, through the same composition every chain of
//! entities goes through rather than a product of matrices, so a copy and an
//! entity holding the same mesh at the same place are the same sixteen floats.
//!
//! **Asked per patch, not per copy.** A patch is a square of the ground eight
//! units across with a box around every copy in it, so a frame asks a few
//! hundred questions where it would otherwise ask a hundred thousand: whether
//! the view holds it, which cascades and which lamps' tiles it throws into,
//! which level of the mesh it is drawn at, whether it reaches into the probes'
//! grid, and how many of its copies the reach still draws. Consecutive patches
//! that agree on all of it and are drawn whole become one draw.
//!
//! **The reach thins rather than cuts.** Inside a patch the copies are in an
//! order drawn at random, so the first part of a run is an even thinning of the
//! whole of it: a patch in the fading band draws a share of its run and the
//! field grows sparse with distance instead of ending at a line. Nothing in the
//! field read for this does that.
//!
//! **Two kinds of strewing are refused with a word**: one whose material is
//! blended, because the blended half of a frame is sorted by depth one entity
//! at a time and a hundred thousand copies would be a sort of its own every
//! frame; and one whose mesh has bones, because every copy would be bent by the
//! one pose the entity names. A masked surface, which is what cut-out foliage
//! is, is solid as far as every list here is concerned and is drawn.

use std::time::Instant;

use colby_core::{
	abi::{
		EntityId, Material, Renderable, Transform, World,
		material::Blend,
		strew::{Layout, Patch},
	},
	bytemuck::{self, Pod, Zeroable},
	glam::{Mat4, Vec3},
	trace, warn,
};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

use crate::{
	cull::{Bounds, Placed},
	detail,
	scene::{MAPS, Placement},
};

/// Whether a world's strewings are drawn at all.
///
/// On, and not saved, for the frustum test's reason: what the switch is for is
/// measuring what the copies cost and taking the picture a build from before
/// them would have taken. Off, every copy is still laid - a rule is the
/// simulation's business - and none of them is drawn.
pub const ENABLED: &str = "r.strew";

/// Its material is masked rather than solid, which picks a pipeline.
const MASKED: u32 = 1;

/// Its material is unlit, which is the one thing that keeps a copy from
/// reading the probes.
const UNLIT: u32 = 2;

/// Decals leave its copies alone.
const UNPAINTED: u32 = 4;

/// Why a strewing draws nothing, when a rule alone would have drawn it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Refused {
	/// Its material is blended.
	Blended,

	/// Bones move its mesh.
	Boned,
}

impl Refused {
	/// What to say about it, once.
	const fn why(self) -> &'static str {
		match self {
			| Self::Blended =>
				"a strewing whose material is blended draws nothing: the blended half of a frame \
				 is sorted one entity at a time",
			| Self::Boned =>
				"a strewing whose mesh has bones draws nothing: every copy would be bent by the \
				 one pose the entity names",
		}
	}
}

/// What one strewing's buffer was built from: when any of it changes, the
/// buffer is built again.
///
/// Compared bit for bit, [`Strewing`](colby_core::abi::Strewing)'s arrangement
/// and for its reason - a number that is not one is still the same as itself,
/// and a world holding one is not built again every frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
struct Built {
	/// Which laying this is, counted across the whole table. @ref
	/// [`Strewn`](colby_core::abi::strew::Strewn).
	revision: u32,

	/// And sixty-four bits of every copy in it, low half first.
	///
	/// Not what the revision already says: a world put back by a load, a tab or
	/// a step back brings a table whose count starts again, so two layings that
	/// are not the same copies can wear one number. The digest is what the
	/// copies themselves are. @ref [`Laid`](colby_core::abi::strew::Laid).
	digest: [u32; 2],

	/// What is strewn and what it is made of, as the renderer's own slots.
	mesh: u32,
	material: u32,

	/// [`MASKED`], [`UNLIT`] and [`UNPAINTED`].
	flags: u32,

	/// The ground's drawn transform, which carries every copy.
	ground: [f32; 16],

	/// The entity's tint times the material's color, and the material's
	/// opacity: what a placement holds.
	tint: [f32; 4],

	/// `[metallic, roughness, uv scale x, uv scale y]`, the rest of it.
	surface: [f32; 4],
}

impl Built {
	/// Whether two would build the same buffer, compared bit for bit.
	fn same(&self, other: &Self) -> bool { bytemuck::bytes_of(self) == bytemuck::bytes_of(other) }

	/// Whether its material is masked rather than solid.
	const fn masked(&self) -> bool { self.flags & MASKED != 0 }

	/// Whether its material is lit at all.
	const fn lit(&self) -> bool { self.flags & UNLIT == 0 }
}

/// One strewing as the renderer holds it.
struct Held {
	/// Who laid it: a slot that changed hands is not the entity that had it.
	entity: EntityId,

	/// What the buffer was built from.
	built: Built,

	/// The copies, as the vertex stage reads them.
	buffer: Buffer,

	/// How many copies it holds, and how many it has room for.
	count: usize,
	room: usize,

	/// Why it draws nothing, when that is the answer: kept so the word is said
	/// once rather than every frame.
	refused: Option<Refused>,
}

/// Every strewing's copies as the renderer holds them, by the strewing
/// entity's slot.
pub(crate) struct Strew {
	/// One slot of the entity table each.
	held: Vec<Option<Held>>,

	/// Scratch the placements are built in, kept so it allocates once.
	building: Vec<Placement>,

	/// The runs of copies the picture draws, patch by patch and joined where
	/// two patches are one draw.
	picture: Vec<Run>,

	/// And the runs each shadow map draws, in the order the atlas's slots are:
	/// the cascades first, the lamps' tiles after them.
	///
	/// A list a map whether or not the frame is culling, where an entity's
	/// cascade falls back to the picture's solid half. That fallback is there
	/// to keep a picture taken before there was a frustum test; nothing strewn
	/// has such a picture, and a strewing that throws no shadow has to throw
	/// none however the console is set.
	casts: [Vec<Run>; MAPS],
}

/// One run of copies to draw: a patch, or several patches next to one another
/// that agree on everything and are drawn whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Run {
	/// Which strewing's buffer it draws out of.
	pub(crate) slot: usize,

	/// Its uploaded geometry.
	pub(crate) mesh: usize,

	/// Which run of that mesh's indices, nought being the mesh itself.
	pub(crate) level: u8,

	/// What it is made of.
	pub(crate) material: usize,

	/// Whether its material is masked, which picks the pipeline.
	pub(crate) masked: bool,

	/// Whether its copies read the probes rather than the sky. Never the
	/// lightmap, which keeps no place for a copy.
	pub(crate) probed: bool,

	/// Where its copies start in the strewing's buffer, and how many.
	pub(crate) first: u32,
	pub(crate) count: u32,
}

impl Strew {
	/// Nothing held.
	pub(crate) fn new() -> Self {
		Self {
			held: Vec::new(),
			building: Vec::new(),
			picture: Vec::new(),
			casts: core::array::from_fn(|_| Vec::new()),
		}
	}

	/// The frame's lists, taken out to be laid out again.
	///
	/// Handed over rather than filled in place, because what fills them reads
	/// the buffers beside them. They come back through [`keep`](Self::keep)
	/// with whatever room they had grown to.
	pub(crate) fn lists(&mut self) -> (Vec<Run>, [Vec<Run>; MAPS]) {
		let mut picture = core::mem::take(&mut self.picture);
		let mut casts = core::mem::replace(&mut self.casts, core::array::from_fn(|_| Vec::new()));

		picture.clear();

		for list in &mut casts {
			list.clear();
		}

		(picture, casts)
	}

	/// The frame's lists, laid out and put back.
	pub(crate) fn keep(&mut self, picture: Vec<Run>, casts: [Vec<Run>; MAPS]) {
		self.picture = picture;
		self.casts = casts;
	}

	/// The runs the picture draws.
	pub(crate) fn drawn(&self) -> &[Run] { &self.picture }

	/// The runs one shadow map draws.
	///
	/// @param map - which of the frame's maps: a cascade, then a lamp's tile
	pub(crate) fn cast(&self, map: usize) -> &[Run] {
		self.casts.get(map).map_or(&[], Vec::as_slice)
	}

	/// Brings every buffer in line with what the world laid, and lets go of
	/// what no strewing lays any more.
	///
	/// @param world - the world being drawn, for the layouts and the ground
	/// @param (device, queue) - what the buffers are built on and written
	/// through
	/// @param boned - whether the uploaded geometry in a mesh slot has bones,
	/// which the world does not know and the renderer does
	pub(crate) fn sync(
		&mut self,
		world: &World,
		(device, queue): (&Device, &Queue),
		boned: &dyn Fn(usize) -> bool,
	) {
		for (slot, held) in self.held.iter_mut().enumerate() {
			let gone = match (held.as_ref(), world.strewn.get(slot)) {
				| (Some(had), Some(layout)) => had.entity != layout.entity,
				| (Some(_), None) => true,
				| (None, _) => false,
			};

			if gone {
				*held = None;
			}
		}

		for (slot, layout) in world.strewn.iter() {
			let Some((renderable, surface)) = drawn_as(world, layout) else {
				continue;
			};
			let Some(ground) = ground_of(world, layout) else {
				continue;
			};
			// the world's own slots: the renderer's tables are filled from the
			// same registries in the same order, and a material past the end of
			// its table is the table's last, exactly as an entity's is
			let mesh = renderable.mesh.slot();
			let built = Built {
				revision: layout.revision,
				digest: [
					u32::try_from(layout.laid.digest & u64::from(u32::MAX)).unwrap_or(0),
					u32::try_from(layout.laid.digest >> 32).unwrap_or(0),
				],
				mesh: slot_of(mesh),
				material: slot_of(
					renderable
						.material
						.slot()
						.min(world.materials.len().saturating_sub(1)),
				),
				flags: (u32::from(surface.blend == Blend::Mask) * MASKED)
					| (u32::from(surface.unlit) * UNLIT)
					| (u32::from(!world.entities.takes_decals(layout.entity)) * UNPAINTED),
				ground: ground.matrix().to_cols_array(),
				tint: (renderable.color * surface.base_color)
					.extend(surface.opacity)
					.to_array(),
				surface: [
					surface.metallic,
					surface.roughness,
					surface.uv_scale.x,
					surface.uv_scale.y,
				],
			};
			let refused = refusal(surface.blend, boned(mesh));

			self.put(slot, layout, (device, queue), (built, ground, refused));
		}
	}

	/// Builds or keeps one strewing's buffer, and says why it draws nothing
	/// when that is the answer.
	fn put(
		&mut self,
		slot: usize,
		layout: &Layout,
		(device, queue): (&Device, &Queue),
		(built, ground, refused): (Built, Transform, Option<Refused>),
	) {
		if slot >= self.held.len() {
			self.held
				.resize_with(slot.saturating_add(1), || None);
		}

		let had = self.held.get(slot).and_then(Option::as_ref);

		// said when the answer changes, and by a slot that has never been built
		// when there is one to give: a frame does not repeat itself
		if let Some(why) = refused.filter(|_| had.is_none_or(|held| held.refused != refused)) {
			warn!(entity = slot, "{}", why.why());
		}

		if had.is_some_and(|held| {
			held.entity == layout.entity && held.built.same(&built) && held.refused == refused
		}) {
			return;
		}

		let began = Instant::now();
		// a refused strewing keeps its slot and its buffer and holds no copies,
		// so the day its material stops being glass it is built like any other
		let copies = if refused.is_some() { 0 } else { layout.laid.pieces.len() };
		let (buffer, room) = match had.filter(|held| held.room >= copies) {
			| Some(held) => (held.buffer.clone(), held.room),
			| None => (placements(device, copies), copies.max(1)),
		};

		self.building.clear();
		self.building.reserve(copies);

		if refused.is_none() {
			let held_at = layout.key.local.scale;

			for piece in &layout.laid.pieces {
				self.building.push(Placement::strewn(
					&ground.then(piece.transform(held_at)),
					shaded(built.tint, piece.shade),
					built.surface,
					built.flags & UNPAINTED != 0,
				));
			}

			if !self.building.is_empty() {
				queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&self.building));
			}
		}

		trace!(
			entity = slot,
			copies,
			refused = refused.is_some(),
			took_us = began.elapsed().as_micros(),
			"copies built"
		);

		if let Some(held) = self.held.get_mut(slot) {
			*held = Some(Held {
				entity: layout.entity,
				built,
				buffer,
				count: copies,
				room,
				refused,
			});
		}
	}

	/// The buffer one strewing's copies are in, for a draw that reads them.
	///
	/// @param slot - the strewing entity's slot
	pub(crate) fn buffer(&self, slot: usize) -> Option<&Buffer> {
		let held = self.held.get(slot)?.as_ref()?;

		(held.count > 0).then_some(&held.buffer)
	}

	/// How the copies one strewing laid are drawn, or nothing for a strewing
	/// that draws none - one refused, one whose rule laid nothing, one whose
	/// buffer has not been built.
	///
	/// @param slot - the strewing entity's slot
	pub(crate) fn drawing(&self, slot: usize) -> Option<Drawing> {
		let held = self.held.get(slot)?.as_ref()?;

		(held.count > 0).then(|| Drawing {
			mesh: at(held.built.mesh),
			material: at(held.built.material),
			masked: held.built.masked(),
			lit: held.built.lit(),
			ground: Mat4::from_cols_array(&held.built.ground),
		})
	}
}

/// How one strewing's copies are drawn, as the buffer holding them was built.
///
/// The ground's matrix among it, and out of the key rather than asked for
/// again: the boxes a frame tests are then the very matrix the copies in the
/// buffer were placed by, so the two can never disagree about where the field
/// is.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Drawing {
	/// Its uploaded geometry, and what it is made of.
	pub(crate) mesh: usize,
	pub(crate) material: usize,

	/// Whether its material is masked, and whether it is lit at all.
	pub(crate) masked: bool,
	pub(crate) lit: bool,

	/// The ground's drawn transform, which carries every patch's box.
	pub(crate) ground: Mat4,
}

/// A tint with a copy's own shade worked into its color and its opacity left
/// alone.
fn shaded(tint: [f32; 4], shade: f32) -> [f32; 4] {
	[tint[0] * shade, tint[1] * shade, tint[2] * shade, tint[3]]
}

/// What a slot holds as a number the rest of the renderer indexes with.
fn at(slot: u32) -> usize { usize::try_from(slot).unwrap_or(0) }

/// A slot as the key holds it.
fn slot_of(slot: usize) -> u32 { u32::try_from(slot).unwrap_or(u32::MAX) }

/// What one strewing draws and what it is made of, or nothing for a stale
/// handle.
fn drawn_as<'a>(world: &'a World, layout: &Layout) -> Option<(&'a Renderable, Material)> {
	let renderable = world.entities.renderable(layout.entity)?;
	let surface = world
		.materials
		.get(renderable.material)
		.copied()
		.unwrap_or(Material::DEFAULT);

	Some((renderable, surface))
}

/// Where the ground one strewing stands on is drawn this frame.
///
/// The copies are laid in the ground's own space, so what carries them is the
/// drawn transform of the entity the strewing hangs off; a strewing hanging off
/// nothing lays nothing.
///
/// @note: the hidden question here is **not** what keeps a hidden strewing from
/// being drawn - that is asked where the frame's lists are laid out, because
/// one hidden after its buffer was built still holds its copies. This is so a
/// hidden strewing costs no buffer and no upload, and a mutation pass that took
/// it out passed everything. It stays for what it saves rather than for what it
/// decides.
fn ground_of(world: &World, layout: &Layout) -> Option<Transform> {
	if !world.entities.shown(layout.entity) {
		return None;
	}

	world.render_transform(world.entities.parent(layout.entity))
}

/// Why a strewing draws nothing, or nothing at all.
///
/// @param blend - how its material reads its picture's alpha
/// @param boned - whether bones move its mesh
const fn refusal(blend: Blend, boned: bool) -> Option<Refused> {
	if boned {
		Some(Refused::Boned)
	} else if matches!(blend, Blend::Alpha) {
		Some(Refused::Blended)
	} else {
		None
	}
}

/// The buffer one strewing's copies are written into.
///
/// At least one copy's worth, because a buffer of nothing is one wgpu refuses
/// to make and a strewing that laid nothing still holds a slot.
fn placements(device: &Device, copies: usize) -> Buffer {
	let size = copies
		.max(1)
		.saturating_mul(size_of::<Placement>());

	device.create_buffer(&BufferDescriptor {
		label: Some("strewn placements"),
		size: u64::try_from(size).unwrap_or(u64::MAX),
		usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
		mapped_at_creation: false,
	})
}

/// The box one patch's copies fill, in the world.
///
/// The patch's box is upright in the ground's own space, and the ground's
/// matrix turns and stretches it exactly the way a mesh's box is turned and
/// stretched for the frustum test. @ref [`Bounds::carried`].
///
/// @param patch - the run and its box
/// @param ground - the ground's drawn transform, as a matrix
pub(crate) fn boxed(patch: &Patch, ground: Mat4) -> Placed {
	Bounds::between(Vec3::from_array(patch.low), Vec3::from_array(patch.high)).carried(ground)
}

/// The most a ground stretches anything standing on it, along any one axis.
///
/// The lengths of the matrix's three columns, which for a transform are its
/// scale. A patch says how big the largest copy in it is held by the strewing
/// entity, and this is what the ground does to that: the two multiplied are
/// what a level of the mesh is chosen by. @ref [`detail::Eye::level`].
///
/// **An upper bound and not the answer**, for a ground stretched unevenly: a
/// copy leaning across two of its axes is smaller than the largest of them
/// says, and a bound that errs upwards draws it at a finer level rather than a
/// coarser one. @ref `STREW-4`.
///
/// @param ground - the ground's drawn transform, as a matrix
pub(crate) fn stretch(ground: Mat4) -> f32 {
	[ground.x_axis, ground.y_axis, ground.z_axis]
		.into_iter()
		.map(|axis| axis.truncate().length())
		.fold(0.0_f32, f32::max)
}

/// How many of a patch's copies the reach still draws.
///
/// **A share of the run rather than a cut**: the copies inside a patch are in
/// an order drawn at random, so the first part of a run is an even thinning of
/// the whole of it, and a patch in the fading band draws a field that is
/// sparser rather than one that stops at a line.
///
/// **One distance for the whole patch, from the nearest point of its box**, the
/// measure a level is chosen by and the measure a thing is called small by. A
/// patch is let go only when the whole of it is past the reach, so the field is
/// never thinner than the reach says anywhere inside it.
///
/// @param patch - the run
/// @param placed - its box, in the world
/// @param eye - where the camera is
/// @param (reach, fade) - how far a copy is drawn, nought for however far, and
/// what share of that, at the far end, the copies thin out over
pub(crate) fn thinned(
	patch: &Patch,
	placed: &Placed,
	eye: Vec3,
	(reach, fade): (f32, f32),
) -> u32 {
	if reach.is_nan() || reach <= 0.0 {
		return patch.count;
	}

	let away = detail::distance(placed, eye);
	let near = reach * (1.0 - fade.clamp(0.0, 1.0));

	if away <= near {
		return patch.count;
	}

	if away >= reach || near >= reach {
		return 0;
	}

	share(patch.count, (reach - away) / (reach - near))
}

/// Puts one patch's run at the end of a list, joining it to the one before it
/// when the two are one draw.
///
/// **Contiguity is the whole test**, beside the four things that pick a
/// pipeline and a mesh: a patch's copies begin where the patch before it ended,
/// so two runs meet only when the first was drawn whole. A thinned run stops
/// short of the next patch's first copy and can never be joined to it, which is
/// what keeps the copies the reach took away from being drawn by their
/// neighbor's draw.
///
/// @param list - the list to add to
/// @param run - the patch's run
pub(crate) fn pushed(list: &mut Vec<Run>, run: Run) {
	match list.last_mut() {
		| Some(last)
			if last.slot == run.slot
				&& last.mesh == run.mesh
				&& last.level == run.level
				&& last.material == run.material
				&& last.masked == run.masked
				&& last.probed == run.probed
				&& last.first.saturating_add(last.count) == run.first =>
			last.count = last.count.saturating_add(run.count),
		| _ => list.push(run),
	}
}

/// A share of a run, rounded down and never past the run.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "a float has no fallible form of this, and the product is held inside the count"
)]
fn share(count: u32, share: f32) -> u32 {
	let kept = f64::from(count) * f64::from(share.clamp(0.0, 1.0));

	(kept as u32).min(count)
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			MaterialId, MeshData, MeshId, Post, STREWING, Sky, Strewing, Terrain, Texel,
			TextureData, ToneMap,
			cvar::Value,
			probes::{Grid, Probes as Named},
			registry::Entry,
			strew::{Key, Piece, lay_out},
		},
		glam::{Quat, Vec4},
		utils::half::half,
	};

	use super::*;
	use crate::{Capture, Image, cull::Drawn, scene::MSAA};

	/// How big every capture here is.
	const SIZE: (u32, u32) = (320, 240);

	/// What the floor is tinted.
	const FLOOR: Vec3 = Vec3::new(0.30, 0.38, 0.24);

	/// And what is strewn over it.
	const GRASS: Vec3 = Vec3::new(0.45, 0.70, 0.25);

	/// A capture on the binary's one device, or `None` with no GPU.
	fn capture() -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, SIZE.0, SIZE.1) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// A console variable declared with its default and set.
	fn asked(world: &mut World, name: &str, default: Value, value: &str) {
		world.cvars.var(name, default, "");
		world.cvars.set(name, value);
	}

	/// A world with the picture's own look out of the way: no curve, an
	/// exposure of one, no sky, and a sun at a slant over a flat ambient.
	fn plainly() -> World {
		let mut world = World::new();

		world.post = Post {
			tonemap: ToneMap::None,
			auto_exposure: false,
			exposure: 1.0,
			..Post::DEFAULT
		};
		world.sky = Sky::NONE;
		world.clear = Vec3::ZERO;
		world.ambient = Vec3::splat(0.22);
		world.light = Vec3::new(0.3, -1.0, -0.4);
		// looking along the ground rather than down at it, so that part of the
		// field is in view and part of it is not: a patch is eight units across
		// and the ground below is twenty-four, which is nine of them
		world.camera.position = Vec3::new(0.0, 2.0, 9.0);
		world.camera.target = Vec3::new(0.0, 0.0, -2.0);

		world
	}

	/// A world with a floor and a strewing hung off it, laid the way the
	/// runtime lays one.
	///
	/// There is no runtime here, so the test does what
	/// `colby_runtime::strew::sync` does: it reads the ground's mesh, calls the
	/// laying and puts what comes back in the world's own table.
	///
	/// @param rule - the strewing's record
	/// @param held - how the strewing entity holds one copy
	/// @return the world and the strewing entity
	fn meadow(rule: Strewing, held: Vec3) -> (World, EntityId) {
		let mut world = plainly();
		// ground twenty-four units across in its own space, which is nine
		// patches: a cube would be one unit and so one patch, and a test over
		// one patch cannot see a patch left out
		let ground = world.meshes.insert(
			"ground",
			Terrain {
				size: 24.0,
				height: 0.0,
				side: 25,
				..Terrain::of(3)
			}
			.build(),
		);
		// turned and moved, and not the identity: a ground at the origin makes
		// carrying the copies by it and not carrying them the same picture,
		// which is a whole line of the renderer no test could see
		let floor = world.entities.spawn_at(Transform {
			position: Vec3::new(1.0, -0.5, 2.0),
			rotation: Quat::from_rotation_y(0.35),
			scale: Vec3::splat(1.2),
		});
		let strewn = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: held,
		});

		if let Some(renderable) = world.entities.renderable_mut(floor) {
			*renderable = Renderable::of(ground, MaterialId::DEFAULT, FLOOR);
		}

		if let Some(renderable) = world.entities.renderable_mut(strewn) {
			*renderable = Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, GRASS);
		}

		world.entities.set_parent(strewn, floor);

		if let Some(held) = world.entities.record_mut(&STREWING, strewn) {
			*held = rule;
		}

		lay(&mut world, strewn);

		(world, strewn)
	}

	/// Lays one strewing's copies into the world's table.
	fn lay(world: &mut World, strewn: EntityId) {
		let (Some(rule), Some(local), Some(renderable)) = (
			world.entities.record(&STREWING, strewn).copied(),
			world.entities.transform(strewn).copied(),
			world.entities.renderable(strewn).copied(),
		) else {
			return;
		};
		let ground = world
			.entities
			.renderable(world.entities.parent(strewn))
			.map_or(MeshId::NONE, |parent| parent.mesh);
		let nothing = MeshData::default();
		let (floor, mesh) = (
			world
				.meshes
				.get(ground)
				.map_or(&nothing, Entry::value),
			world
				.meshes
				.get(renderable.mesh)
				.map_or(&nothing, Entry::value),
		);
		let laid = lay_out(floor, &rule, &local, mesh.bounds(), None);

		world.strewn.put(strewn.slot(), Layout {
			entity: strewn,
			key: Key {
				mask: 0,
				rule,
				ground: (ground, 0),
				mesh: (renderable.mesh, 0),
				local,
			},
			laid,
			solid: MeshId::NONE,
			revision: 0,
		});
	}

	/// Every copy one strewing laid.
	fn pieces(world: &World, strewn: EntityId) -> Vec<Piece> {
		world
			.strewn
			.get(strewn.slot())
			.map(|layout| layout.laid.pieces.clone())
			.unwrap_or_default()
	}

	/// The same world with every copy standing as an entity of its own, hung
	/// off the same floor and held the same way, and the strewing gone.
	///
	/// **The one comparison this module is for**: a copy the renderer draws out
	/// of a strewing's buffer and an entity the renderer draws out of the
	/// frame's have to be the same pixels, or a field is a second way of
	/// drawing a mesh.
	fn twinned(rule: Strewing, held: Vec3) -> World {
		let (mut world, strewn) = meadow(rule, held);
		let copies = pieces(&world, strewn);
		let floor = world.entities.parent(strewn);

		world.strewn.take(strewn.slot());
		world.entities.despawn(strewn);

		for piece in copies {
			let id = world.entities.spawn_at(piece.transform(held));

			world.entities.set_parent(id, floor);

			if let Some(renderable) = world.entities.renderable_mut(id) {
				*renderable =
					Renderable::of(MeshId::CUBE, MaterialId::DEFAULT, GRASS * piece.shade);
			}
		}

		world
	}

	/// The same world with no strewing in it at all: the floor, and nothing
	/// standing on it.
	///
	/// Built out of [`meadow`] and then emptied rather than written again, so
	/// that the floor a control is compared against is the very floor the field
	/// stood on.
	fn bare() -> World {
		let (mut world, strewn) = meadow(field(), Vec3::ONE);

		world.strewn.take(strewn.slot());
		world.entities.despawn(strewn);

		world
	}

	/// A frame drawn, and its counts.
	fn shot(capture: &mut Capture, world: &mut World) -> (Image, Drawn) {
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

	/// A grid of probes over the floor, each probe the same light on every
	/// face, named as the world's.
	///
	/// What it is for is the one branch a copy has that a picture can see: a
	/// thing whose box reaches into the grid is drawn by a pipeline that reads
	/// the probes rather than the sky, and a copy has to pick the same one an
	/// entity standing there picks.
	fn probed(world: &mut World) {
		let grid = Grid {
			from: Vec3::new(-8.0, -1.0, -8.0),
			step: 4.0,
			counts: [5, 2, 5],
		};
		let [width, height] = grid.picture().expect("a small grid");
		let mut texels = vec![[0.0_f32; 3]; usize::try_from(width * height).expect("small")];

		for index in 0..grid.len() {
			let place = grid.place(index);

			for face in 0..6 {
				let [column, row] = grid.texel(place, face);

				texels[usize::try_from(row * width + column).expect("small")] = [0.1, 0.3, 0.5];
			}
		}

		let picture = world
			.textures
			.insert("lightmaps/probes/test", TextureData {
				width,
				height,
				faces: 1,
				texel: Texel::Rgba16Float,
				levels: vec![
					texels
						.iter()
						.flat_map(|[red, green, blue]| [*red, *green, *blue, 1.0])
						.flat_map(|channel| half(channel).to_le_bytes())
						.collect(),
				],
			});

		world.probes = Named { picture, grid };
	}

	/// A rule that lays a handful of copies over a floor, leaning and shaded.
	fn field() -> Strewing {
		Strewing {
			strews: 1,
			seed: 19,
			density: 0.05,
			size: [0.6, 1.4],
			tilt: 12.0,
			shade: 0.4,
			..Strewing::NONE
		}
	}

	/// A patch of a given count, standing in a box.
	fn patch(count: u32, low: [f32; 3], high: [f32; 3]) -> Patch {
		Patch { first: 0, count, low, high, largest: 1.0 }
	}

	#[test]
	fn a_rule_with_no_reach_draws_every_copy_however_far_away_it_is() {
		let one = patch(1000, [-1.0, 0.0, -1.0], [1.0, 1.0, 1.0]);
		let placed = boxed(&one, Mat4::IDENTITY);
		let far = Vec3::new(0.0, 0.0, 9999.0);

		assert_eq!(thinned(&one, &placed, far, (0.0, 0.25)), 1000);
		assert_eq!(
			thinned(&one, &placed, far, (-1.0, 0.25)),
			1000,
			"and nor does a reach nobody could mean take anything away"
		);
	}

	#[test]
	fn the_reach_thins_a_patch_over_the_far_share_of_it_and_then_lets_it_go() {
		let one = patch(1000, [-0.5, 0.0, -0.5], [0.5, 1.0, 0.5]);
		let placed = boxed(&one, Mat4::IDENTITY);
		let at =
			|away: f32| thinned(&one, &placed, Vec3::new(0.0, 0.5, away + 0.5), (40.0, 0.25));

		assert_eq!(at(0.0), 1000, "standing on it");
		assert_eq!(at(29.9), 1000, "inside the part that is not thinned");
		assert_eq!(at(30.0), 1000, "and at the edge of it");
		assert_eq!(at(35.0), 500, "half way through the band is half the run");
		assert_eq!(at(39.0), 100, "and a tenth of the way from its end is a tenth");
		assert_eq!(at(40.0), 0, "past the reach is nothing");
		assert_eq!(at(400.0), 0, "and it stays nothing");
	}

	#[test]
	fn a_reach_with_no_fade_is_a_cut_rather_than_a_band() {
		let one = patch(64, [-0.5, 0.0, -0.5], [0.5, 1.0, 0.5]);
		let placed = boxed(&one, Mat4::IDENTITY);
		let at = |away: f32| thinned(&one, &placed, Vec3::new(0.0, 0.5, away + 0.5), (10.0, 0.0));

		assert_eq!(at(9.9), 64, "every copy up to the line");
		assert_eq!(at(10.1), 0, "and none past it");
	}

	#[test]
	fn a_patch_is_measured_from_the_nearest_point_of_its_box_and_not_its_middle() {
		// twenty units long and a hair wide, the eye at one end of it: its
		// middle is ten units away and its nearest point is none
		let one = patch(100, [-0.5, 0.0, 0.0], [0.5, 1.0, 20.0]);
		let placed = boxed(&one, Mat4::IDENTITY);

		// its middle is ten away and a cut at eight would take the whole patch
		assert_eq!(
			thinned(&one, &placed, Vec3::new(0.0, 0.5, -0.1), (8.0, 0.0)),
			100,
			"the near end is under the eye, so the whole run is drawn"
		);
	}

	#[test]
	fn the_box_of_a_patch_is_carried_by_the_ground_it_stands_on() {
		let one = patch(1, [-1.0, 0.0, -2.0], [1.0, 4.0, 2.0]);
		let ground = Transform {
			position: Vec3::new(10.0, 0.0, 0.0),
			rotation: Quat::from_rotation_y(core::f32::consts::FRAC_PI_2),
			scale: Vec3::splat(2.0),
		};
		let placed = boxed(&one, ground.matrix());

		assert!(
			placed
				.center
				.abs_diff_eq(Vec3::new(10.0, 4.0, 0.0), 1.0e-4),
			"the middle of the box, carried: {}",
			placed.center
		);
		// the turn moves which way each half-edge points and the scale doubles
		// its length, which is what a box carried by a matrix is
		assert!(
			(placed.edges[0].length() - 2.0).abs() < 1.0e-3,
			"the first half-edge is one unit of x scaled by two: {}",
			placed.edges[0].length()
		);
		assert!(
			(placed.edges[2].length() - 4.0).abs() < 1.0e-3,
			"and the third is two units of z scaled by two: {}",
			placed.edges[2].length()
		);
	}

	#[test]
	fn a_blended_strewing_and_a_boned_one_are_refused_and_nothing_else_is() {
		assert_eq!(refusal(Blend::Alpha, false), Some(Refused::Blended));
		assert_eq!(refusal(Blend::Opaque, true), Some(Refused::Boned));
		assert_eq!(
			refusal(Blend::Alpha, true),
			Some(Refused::Boned),
			"bones first, because a mesh with them is refused whatever it is made of"
		);
		assert_eq!(refusal(Blend::Opaque, false), None);
		assert_eq!(refusal(Blend::Mask, false), None, "a cut-out leaf is drawn");
	}

	/// A run of a given place and length, everything else alike.
	fn run(first: u32, count: u32) -> Run {
		Run {
			slot: 1,
			mesh: 2,
			level: 0,
			material: 3,
			masked: false,
			probed: false,
			first,
			count,
		}
	}

	#[test]
	fn patches_next_to_one_another_and_drawn_whole_become_one_draw() {
		let mut list = Vec::new();

		pushed(&mut list, run(0, 100));
		pushed(&mut list, run(100, 50));
		pushed(&mut list, run(150, 25));

		assert_eq!(list.len(), 1, "three patches in a row are one draw");
		assert_eq!(list.first().map(|one| one.count), Some(175));
	}

	#[test]
	fn a_thinned_patch_stops_short_and_is_a_draw_of_its_own() {
		let mut list = Vec::new();

		// the first patch holds a hundred and draws forty of them
		pushed(&mut list, run(0, 40));
		pushed(&mut list, run(100, 50));

		assert_eq!(list.len(), 2, "the copies the reach took away are not drawn by the next");
		assert_eq!(list.first().map(|one| one.count), Some(40));
	}

	#[test]
	fn two_patches_that_disagree_about_anything_are_two_draws() {
		let apart = [
			Run { level: 1, ..run(100, 50) },
			Run { probed: true, ..run(100, 50) },
			Run { masked: true, ..run(100, 50) },
			Run { material: 9, ..run(100, 50) },
			Run { mesh: 9, ..run(100, 50) },
			Run { slot: 9, ..run(100, 50) },
		];

		for one in apart {
			let mut list = vec![run(0, 100)];

			pushed(&mut list, one);

			assert_eq!(list.len(), 2, "{one:?} is not the run before it");
		}
	}

	#[test]
	fn a_share_of_a_run_never_leaves_it() {
		assert_eq!(share(100, 0.5), 50);
		assert_eq!(share(100, 1.5), 100, "a share past one is the whole run");
		assert_eq!(share(100, -1.0), 0);
		assert_eq!(share(100, f32::NAN), 0, "and one that is not a number is none of it");
		assert_eq!(share(u32::MAX, 1.0), u32::MAX);
	}

	#[test]
	fn a_shade_darkens_the_color_of_a_copy_and_leaves_its_opacity_alone() {
		let half = Vec4::from_array(shaded([0.8, 0.4, 0.2, 0.5], 0.5));
		let whole = Vec4::from_array(shaded([0.8, 0.4, 0.2, 0.5], 1.0));

		assert!(half.abs_diff_eq(Vec4::new(0.4, 0.2, 0.1, 0.5), 1.0e-6), "{half}");
		assert!(whole.abs_diff_eq(Vec4::new(0.8, 0.4, 0.2, 0.5), 1.0e-6), "{whole}");
	}

	#[test]
	fn two_keys_are_the_same_when_every_bit_of_them_is() {
		let one = Built {
			revision: 3,
			digest: [7, 11],
			mesh: 1,
			material: 2,
			flags: MASKED,
			ground: Mat4::IDENTITY.to_cols_array(),
			tint: [1.0, 1.0, 1.0, 1.0],
			surface: [0.0, 0.5, 1.0, 1.0],
		};

		assert!(one.same(&one));
		assert!(!one.same(&Built { revision: 4, ..one }), "a laying of its own");
		assert!(
			!one.same(&Built { digest: [7, 12], ..one }),
			"and copies that are not the same copies, whatever the count says"
		);
		assert!(
			!one.same(&Built {
				ground: Mat4::from_translation(Vec3::X).to_cols_array(),
				..one
			}),
			"a ground that moved"
		);
		assert!(!one.same(&Built { tint: [1.0, 1.0, 1.0, 0.5], ..one }), "a tint that changed");
		assert!(one.masked(), "and the flags read back");
		assert!(one.lit());
		assert!(!Built { flags: UNLIT, ..one }.lit());
	}

	#[test]
	fn a_copy_a_strewing_drew_is_the_pixels_an_entity_standing_there_would_be() {
		let (Some(mut capture), Some(mut other)) = (capture(), capture()) else {
			return;
		};
		let held = Vec3::new(0.12, 0.6, 0.12);

		for samples in ["1", "4"] {
			let (mut strewn, laid) = meadow(field(), held);
			let mut twins = twinned(field(), held);

			assert!(pieces(&strewn, laid).len() > 8, "the rule laid a field to look at");
			asked(&mut strewn, MSAA, Value::Float(1.0), samples);
			asked(&mut twins, MSAA, Value::Float(1.0), samples);

			let (drawn, counts) = shot(&mut capture, &mut strewn);
			let (stood, twin_counts) = shot(&mut other, &mut twins);

			assert_eq!(counts.strewn, pieces(&strewn, laid).len(), "every copy is counted");
			assert_eq!(twin_counts.strewn, 0, "and the twin world strews nothing");
			assert_eq!(
				apart(&drawn, &stood),
				0,
				"at {samples} samples the field and the entities standing where it stands are 				 one picture, shadows and all"
			);
		}
	}

	#[test]
	fn the_switch_takes_every_copy_out_and_leaves_the_ground_it_stood_on() {
		let (Some(mut capture), Some(mut other)) = (capture(), capture()) else {
			return;
		};
		let held = Vec3::new(0.12, 0.6, 0.12);
		let (mut strewn, laid) = meadow(field(), held);
		let mut bare = bare();

		asked(&mut strewn, ENABLED, Value::Bool(true), "false");

		let (off, counts) = shot(&mut capture, &mut strewn);
		let (nothing, bare_counts) = shot(&mut other, &mut bare);

		assert_eq!(
			counts.strewn,
			pieces(&strewn, laid).len(),
			"the copies are laid all the same"
		);
		assert_eq!((counts.strewn_drawn, counts.strewn_cast), (0, 0), "and none of them drawn");
		assert_eq!(bare_counts.strewn, 0, "and a world with no strewing in it lays none");
		assert_eq!(
			apart(&off, &nothing),
			0,
			"the switch off is the picture of a world with no field in it"
		);

		asked(&mut strewn, ENABLED, Value::Bool(true), "true");

		let (on, _) = shot(&mut capture, &mut strewn);

		assert!(apart(&on, &off) > 100, "and the switch on is another picture");
	}

	#[test]
	fn the_copies_a_frame_drew_and_cast_are_counted_and_the_reach_takes_them_away() {
		let Some(mut capture) = capture() else {
			return;
		};
		let held = Vec3::new(0.12, 0.6, 0.12);
		let (mut world, laid) = meadow(field(), held);
		let all = pieces(&world, laid).len();
		let (_, counts) = shot(&mut capture, &mut world);

		assert_eq!(counts.strewn, all);
		assert!(
			counts.strewn_drawn > 0 && counts.strewn_drawn < all,
			"the view holds some of the nine patches and not the rest: {} of {all}",
			counts.strewn_drawn
		);
		assert!(counts.strewn_cast > 0, "and the cascades draw them");

		// a reach shorter than the floor is wide leaves the far copies out
		let (mut near, _) = meadow(Strewing { reach: 3.0, ..field() }, held);
		let (_, close) = shot(&mut capture, &mut near);

		assert!(
			close.strewn_drawn < counts.strewn_drawn,
			"a reach of three draws fewer than however far: {} against {}",
			close.strewn_drawn,
			counts.strewn_drawn
		);

		// and one shorter than anything draws none at all
		let (mut none, _) = meadow(Strewing { reach: 0.01, ..field() }, held);
		let (_, cut) = shot(&mut capture, &mut none);

		assert_eq!(cut.strewn_drawn, 0, "a reach of nothing draws nothing");
		assert_eq!(cut.strewn_cast, 0, "and casts nothing either");
	}

	#[test]
	fn a_strewing_that_throws_no_shadow_is_drawn_and_casts_nothing() {
		let Some(mut capture) = capture() else {
			return;
		};
		let held = Vec3::new(0.12, 0.6, 0.12);
		let (mut world, _) = meadow(Strewing { shadows: 0, ..field() }, held);
		let (_, counts) = shot(&mut capture, &mut world);

		assert!(counts.strewn_drawn > 0, "the picture still holds it");
		assert_eq!(counts.strewn_cast, 0, "and no map draws a copy of it");
	}

	#[test]
	fn a_strewing_whose_material_is_glass_draws_nothing_and_says_so() {
		let (Some(mut capture), Some(mut other)) = (capture(), capture()) else {
			return;
		};
		let held = Vec3::new(0.12, 0.6, 0.12);
		let (mut world, strewn) = meadow(field(), held);
		let mut bare = bare();
		let glass = world.materials.insert("glass", Material {
			blend: Blend::Alpha,
			opacity: 0.5,
			..Material::DEFAULT
		});

		if let Some(renderable) = world.entities.renderable_mut(strewn) {
			*renderable = Renderable::of(MeshId::CUBE, glass, GRASS);
		}

		let (refused, counts) = shot(&mut capture, &mut world);
		let (nothing, _) = shot(&mut other, &mut bare);

		assert!(counts.strewn > 0, "the rule laid its copies");
		assert_eq!(counts.strewn_drawn, 0, "and none of them is drawn");
		assert_eq!(apart(&refused, &nothing), 0, "which is the picture of no field at all");
	}

	#[test]
	fn a_strewing_hidden_after_it_was_drawn_stops_being_drawn() {
		let (Some(mut capture), Some(mut other)) = (capture(), capture()) else {
			return;
		};
		let held = Vec3::new(0.12, 0.6, 0.12);
		let (mut world, strewn) = meadow(field(), held);
		let mut bare = bare();
		// drawn first, so the copies are built and held: a strewing hidden
		// before anything was built is the easy half of this
		let (shown, first) = shot(&mut capture, &mut world);

		assert!(first.strewn_drawn > 0, "the field is there to begin with");

		world.entities.set_hidden(strewn, true);

		let (hidden, counts) = shot(&mut capture, &mut world);
		let (nothing, _) = shot(&mut other, &mut bare);

		assert_eq!(counts.strewn_drawn, 0, "a hidden strewing draws no copy");
		assert_eq!(apart(&hidden, &nothing), 0, "and the picture is the bare floor");
		assert!(apart(&shown, &hidden) > 100, "which is not the picture it drew before");
	}

	#[test]
	fn a_copy_that_reads_the_probes_is_the_pixels_of_an_entity_that_reads_them() {
		let (Some(mut capture), Some(mut other)) = (capture(), capture()) else {
			return;
		};
		let held = Vec3::new(0.12, 0.6, 0.12);
		let (mut strewn, _) = meadow(field(), held);
		let mut twins = twinned(field(), held);

		probed(&mut strewn);
		probed(&mut twins);

		let (drawn, _) = shot(&mut capture, &mut strewn);
		let (stood, _) = shot(&mut other, &mut twins);

		assert_eq!(
			apart(&drawn, &stood),
			0,
			"a field over probes is the entities standing in it, pipeline and all"
		);

		// and it is another picture than the same field with the probes off,
		// which is what says the probed pipeline is the one that drew it
		asked(&mut strewn, crate::probes::ENABLED, Value::Bool(true), "false");

		let (unprobed, _) = shot(&mut capture, &mut strewn);

		assert!(apart(&drawn, &unprobed) > 100, "and the probes are what lit it");
	}

	#[test]
	fn a_world_put_back_with_another_field_in_it_builds_the_copies_again() {
		let Some(mut capture) = capture() else {
			return;
		};
		let held = Vec3::new(0.12, 0.6, 0.12);
		// two worlds, each laid once, so each layout wears the same count: what
		// tells them apart is the copies, which is why the digest is in the key
		let (mut one, first) = meadow(field(), held);
		let (mut two, second) = meadow(Strewing { seed: 71, ..field() }, held);

		assert_eq!(
			world_revision(&one, first),
			world_revision(&two, second),
			"a table that starts again gives the second world the first's count"
		);

		let (drawn, _) = shot(&mut capture, &mut one);
		let (again, _) = shot(&mut capture, &mut two);

		assert!(
			apart(&drawn, &again) > 100,
			"the same renderer handed another field draws that field, not the one it held"
		);
	}

	/// Which laying a strewing's layout is, as its own table counted it.
	fn world_revision(world: &World, strewn: EntityId) -> u32 {
		world
			.strewn
			.get(strewn.slot())
			.map_or(0, |layout| layout.revision)
	}

	#[test]
	fn what_a_level_is_chosen_by_is_never_smaller_than_the_copy_it_is_about() {
		let ground = Transform {
			position: Vec3::new(3.0, 0.0, -1.0),
			rotation: Quat::from_rotation_y(0.7),
			scale: Vec3::new(2.0, 1.0, 3.0),
		};
		let held = Vec3::new(0.1, 0.5, 0.1);
		let piece = Piece {
			at: [0.0, 0.0, 0.0],
			turn: Quat::from_rotation_x(0.4).to_array(),
			size: 1.4,
			shade: 1.0,
		};
		// what a patch carries, times what the ground does to it
		let asked = piece.size.abs() * held.abs().max_element() * stretch(ground.matrix());
		// and what the copy really is, the way an entity standing there is
		// measured: the scale of the transform the two compose to
		let truly = ground
			.then(piece.transform(held))
			.scale
			.abs()
			.max_element();

		assert!(
			asked >= truly,
			"the bound errs upwards, which draws a finer level and never a coarser: {asked} 			 \
			 against {truly}"
		);
		assert!(asked < truly * 4.0, "and not by more than the ground's own spread");
	}

	#[test]
	fn a_reach_that_reaches_every_patch_still_draws_fewer_copies_than_it_laid() {
		let Some(mut capture) = capture() else {
			return;
		};
		let held = Vec3::new(0.12, 0.6, 0.12);
		// looked at from above, so that every patch is in view and nothing is
		// dropped by the frustum: what moves the count is the thinning alone
		let over = |rule| {
			let (mut world, laid) = meadow(rule, held);

			world.camera.position = Vec3::new(1.0, 30.0, 2.0);
			world.camera.target = Vec3::new(1.0, 0.0, 2.0);

			(world, laid)
		};
		// thick enough that a patch holds a hundred copies and more: thinning a
		// patch of three rounds two of them away and the third to nothing,
		// which is a patch dropped whole wearing a thinning's clothes
		let thick = Strewing { density: 2.0, ..field() };
		let (mut whole, laid) = over(thick);
		let (_, all) = shot(&mut capture, &mut whole);

		assert_eq!(all.strewn_drawn, all.strewn, "from above the view holds the whole field");
		assert!(pieces(&whole, laid).len() > 500, "and every patch holds a great many");

		// a reach half as far again as the eye stands, fading over all of it:
		// every patch is inside it and every patch is thinned
		let (mut thinned, _) = over(Strewing { reach: 45.0, fade: 1.0, ..thick });
		let (_, some) = shot(&mut capture, &mut thinned);

		assert_eq!(some.strewn, all.strewn, "the same rule lays the same copies");
		assert!(
			some.strewn_drawn > 0 && some.strewn_drawn < all.strewn_drawn,
			"a share of every patch is drawn, not all of it and not none: {} of {}",
			some.strewn_drawn,
			all.strewn_drawn
		);
	}
}
