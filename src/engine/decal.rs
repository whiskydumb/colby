//! Pictures thrown onto the world: which decals a frame carries, and the one
//! texture every picture they throw is packed into.
//!
//! **A decal is painted by the surface it lands on, not drawn over it.** Every
//! fragment of every mesh walks the frame's list of decals, asks whether it
//! stands inside each one's box, and where it does mixes the picture into its
//! own color, normal, roughness and how metal it is before it is lit. So a
//! decal is lit exactly as the wall under it is, a pane of glass takes one on
//! the same terms as a wall, and nothing needs a depth buffer to find the wall.
//! The list has the lamps' shape: flat, budgeted, the nearest few whose box the
//! view can see, and a console variable over the number.
//!
//! **Every picture is in one atlas.** A fragment may stand in any of the decals
//! at once, so every picture has to be bound at once, and there are three ways
//! to do that: a texture array, which makes every picture one size; a binding
//! array, which is a device feature this renderer asks for nowhere else and a
//! second path for every device without it; and an atlas, one texture and a
//! rectangle a picture. This is the third, the one that needs nothing a device
//! might lack. Each picture's own mip levels are copied in, on a grid coarse
//! enough that the first [`LEVELS`] of them land on whole texels, and the atlas
//! is rebuilt only when the set of pictures changes, never because the camera
//! moved.

use colby_core::{
	abi::{
		Decal, EntityId, Material, TextureData, TextureId, Textures, Transform, World,
		registry::Entry,
	},
	bytemuck::{Pod, Zeroable},
	glam::Vec3,
	warn,
};
use wgpu::{
	Device, Extent3d, Origin3d, Queue, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture,
	TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
	TextureView, TextureViewDescriptor,
};

use crate::cull::Frustum;

/// How many decals one frame may carry.
///
/// Matched by `MAX_DECALS` in `shader.wgsl`, which sizes the uniform this
/// fills. **Thirty-two because it was measured**, on the list's own shape
/// before any of this was written: every fragment walking a flat list of boxes
/// cost nothing measurable at eight, twenty to thirty microseconds of a 1280 by
/// 720 frame at sixteen and at thirty-two, and a hundred at sixty-four, with
/// every decal missing every pixel. A cliff, and this is the near side of it.
pub const MAX_DECALS: usize = 32;

/// What [`DECALS`] holds until somebody sets it, as a console variable holds
/// it.
///
/// Written out rather than converted from [`MAX_DECALS`], the lamps'
/// arrangement, so that a test can say the two are one number.
pub const DEFAULT_DECALS: f32 = 32.0;

/// The console variable that says how many decals a frame carries.
///
/// A ceiling on the ceiling, as `r.lights` is: a machine that cannot afford
/// the loop has somewhere to say so, and nought is the picture exactly as it
/// was before there were decals, which is what makes it the switch a
/// comparison is made with.
pub const DECALS: &str = "r.decals";

/// How many mip levels the atlas keeps.
///
/// Five, which is what a grid of [`CELL`] texels holds whole: a picture set
/// half a cell in from its cell's corner lands on a whole texel in every one of
/// them. A decal far enough away to want a sixth reads the fifth, which is a
/// little sharp rather than wrong.
const LEVELS: u32 = 5;

/// The grid pictures are packed on, in texels of the atlas's largest level.
///
/// Twice the coarsest level's texel. Each picture sits half a cell in from its
/// cell's corner, which is a whole texel five levels down, and that half cell
/// is also the gap between one picture and the next: what a sample at a
/// picture's edge bleeds into rather than into its neighbor.
const CELL: u32 = 1 << LEVELS;

/// The widest and the tallest the atlas grows.
const MAX_SIDE: u32 = 8192;

/// The longest side a picture is packed at before anything has had to give
/// way.
const MAX_PICTURE: u32 = 1024;

/// The smallest that may shrink to before the atlas gives up.
const MIN_PICTURE: u32 = 64;

/// A box flattened along an axis past this has no inside to paint.
const FLAT: f32 = 1.0e-6;

/// A picture a decal throws, as the atlas knows it: a texture's registry slot
/// and the revision it was at, so that a picture re-imported under a running
/// process is a new key and the atlas is rebuilt with it.
pub(crate) type Key = (u32, u32);

/// One decal a frame carries, everything about it read off the world once.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Chosen {
	/// Which entity, for the tie between two decals of one order.
	entity: EntityId,

	/// What it paints.
	decal: Decal,

	/// Its box: where it is drawn this frame.
	at: Transform,

	/// What it throws, which is the entity's own material.
	material: Material,

	/// The entity's own tint.
	tint: Vec3,

	/// How far the edge of its box is from the eye.
	distance: f32,
}

impl Chosen {
	/// The two pictures it throws. @ref [`thrown`].
	#[must_use]
	pub(crate) fn pictures(&self) -> [Option<TextureId>; 2] { thrown(&self.material) }
}

/// Which decals a frame carries, and in the order they are painted.
///
/// **Chosen by distance and painted by order**, two sorts for two questions.
/// Which decals the budget is spent on is the lamps' answer, the nearest by
/// the edge of their box among those whose box the view can see; which of two
/// that overlap is on top is [`Decal::order`] and then the entity's slot,
/// neither of which moves when the camera does. One sort doing both would turn
/// a pile of decals over as somebody walked past it.
///
/// A hidden decal paints nothing, and neither does one hanging under
/// something hidden: a decal is part of the picture, and the picture is what
/// asks. @ref `Entities::shown`.
///
/// @param world - the world being drawn
/// @param eye - where the camera is
/// @param view - what the picture can see, or `None` to carry decals wherever
/// they are
/// @param room - how many the frame may carry
/// @param carried - the caller's list, left holding the answer in painting
/// order, so that nothing here allocates once it has grown
pub(crate) fn chosen(
	world: &World,
	eye: Vec3,
	view: Option<&Frustum>,
	room: usize,
	carried: &mut Vec<Chosen>,
) {
	carried.clear();

	for (id, _, renderable) in world.entities.iter() {
		let Some(decal) = world
			.entities
			.decal(id)
			.copied()
			.filter(|it| it.paints())
		else {
			continue;
		};

		if !world.entities.shown(id) {
			continue;
		}

		let Some(at) = world.render_transform(id) else {
			continue;
		};

		// the box's own ball, and nothing at all for a box with no inside
		let Some(reach) = reach(at) else {
			continue;
		};

		if view.is_some_and(|view| !view.holds_ball(at.position, reach)) {
			continue;
		}

		carried.push(Chosen {
			entity: id,
			decal,
			at,
			material: world
				.materials
				.get(renderable.material)
				.copied()
				.unwrap_or(Material::DEFAULT),
			tint: renderable.color,
			distance: (at.position - eye).length() - reach,
		});
	}

	// `total_cmp` for the lamps' reason: a decal at a distance that is not a
	// number is a world that has blown up, and it should sort somewhere
	// definite rather than making the order depend on which pairs were asked.
	carried.sort_by(|near, far| near.distance.total_cmp(&far.distance));
	carried.truncate(room.min(MAX_DECALS));
	carried.sort_by_key(|it| (it.decal.order, it.entity.slot()));
}

/// Every picture any decal in the world throws, as keys, sorted and each once.
///
/// **Every decal, and not only the ones a frame carries.** The atlas is
/// rebuilt when this set changes, and a set taken from what one frame carries
/// would change every time the camera turned a decal out of the budget and
/// back into it. A hidden decal's pictures are in it too, so showing one again
/// is not a rebuild either.
///
/// @param world - the world being drawn
/// @param out - the caller's list, left holding the answer
pub(crate) fn pictures(world: &World, out: &mut Vec<Key>) {
	out.clear();

	for (id, _, renderable) in world.entities.iter() {
		if !world
			.entities
			.decal(id)
			.is_some_and(|it| it.paints())
		{
			continue;
		}

		let material = world
			.materials
			.get(renderable.material)
			.copied()
			.unwrap_or(Material::DEFAULT);

		out.extend(
			thrown(&material)
				.into_iter()
				.flatten()
				.filter_map(|picture| key(&world.textures, picture)),
		);
	}

	out.sort_unstable();
	out.dedup();
}

/// Which picture a handle is, as the atlas knows it.
///
/// @param textures - the world's registry
/// @param picture - the handle
/// @return its key, or nothing for a handle the registry does not have
#[must_use]
pub(crate) fn key(textures: &Textures, picture: TextureId) -> Option<Key> {
	textures
		.get(picture)
		.map(|entry| (picture.index(), entry.revision()))
}

/// The pictures a material throws as a decal: its color, unless it has none
/// and throws its tint alone, and its normal map, unless it is the flat one
/// that would leave a surface as it was.
fn thrown(material: &Material) -> [Option<TextureId>; 2] {
	[
		Some(material.albedo).filter(|it| *it != TextureId::NONE),
		Some(material.normal)
			.filter(|it| *it != TextureId::FLAT_NORMAL && *it != TextureId::NONE),
	]
}

/// How far a decal's box reaches from its middle: half its diagonal, or
/// nothing for a box flattened along some axis, which has no inside to paint
/// and whose inverse is not a number.
fn reach(at: Transform) -> Option<f32> {
	let size = at.scale.abs();

	(size.min_element() > FLAT).then(|| size.length() * 0.5)
}

/// How many decals a console variable is asking for.
///
/// The lamps' rule: below nought is none, above the ceiling is the ceiling, and
/// a number that is not one is the ceiling too, because a variable nobody
/// meant to set should not wipe the decals off the world.
#[must_use]
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "clamped to the array's own length first, so the cast cannot lose anything or go \
	          negative"
)]
pub(crate) fn room(asked: f32) -> usize {
	if asked.is_nan() {
		return MAX_DECALS;
	}

	#[expect(
		clippy::cast_precision_loss,
		reason = "thirty-two is exact in an f32"
	)]
	let ceiling = MAX_DECALS as f32;

	asked.clamp(0.0, ceiling) as usize
}

/// Where a decal that throws no picture has one: nowhere, all nought.
///
/// The shader reads a rectangle whose corner is at nought as no picture,
/// which no real one can be: every picture sits at least half a cell in from
/// the atlas's corner. @ref [`inset`].
pub(crate) const NO_PICTURE: [f32; 4] = [0.0; 4];

/// One decal, as the shader reads it.
///
/// Seven vectors, so that an array of them is laid out in a uniform exactly as
/// it is here: a uniform array's stride is its element rounded up to sixteen,
/// and this is already a multiple of it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub(crate) struct Paint {
	/// The world into the box's own space, one row an axis: a point's place
	/// along an axis is the dot of the row's xyz with it plus the row's w, and
	/// the box is where all three places are within a half of nought.
	rows: [[f32; 4]; 3],

	/// Where the color picture is in the atlas, `[u, v, width, height]`, or
	/// all nought for a decal that throws its tint alone.
	color: [f32; 4],

	/// Where the normal map is, the same way, or all nought for none.
	normal: [f32; 4],

	/// `[r, g, b, opacity]`: the material's color times the entity's, and the
	/// material's opacity, which a decal reads whatever the material's blend
	/// says because a decal is always blended.
	tint: [f32; 4],

	/// `[metallic, roughness, fade, unused]`.
	surface: [f32; 4],
}

impl Paint {
	/// A decal that is not there, for the tail of the array.
	///
	/// Its box is somewhere no point is, rather than the unit box at the
	/// origin that all noughts would make: the shader reads only as far as the
	/// count, and this is what it would find if it ever read further.
	pub(crate) const NOTHING: Self = Self {
		rows: [[0.0, 0.0, 0.0, 2.0]; 3],
		color: [0.0; 4],
		normal: [0.0; 4],
		tint: [0.0; 4],
		surface: [0.0; 4],
	};

	/// One chosen decal, packed.
	///
	/// @param chosen - the decal
	/// @param color - where its color picture is in the atlas, or
	/// [`NO_PICTURE`]
	/// @param normal - where its normal map is, or [`NO_PICTURE`]
	/// @return the record, or nothing for a box whose inverse is not a number
	#[must_use]
	pub(crate) fn of(chosen: &Chosen, color: [f32; 4], normal: [f32; 4]) -> Option<Self> {
		let inverse = chosen.at.matrix().inverse();
		let rows = [inverse.row(0), inverse.row(1), inverse.row(2)].map(|row| row.to_array());

		if rows.iter().flatten().any(|it| !it.is_finite()) {
			return None;
		}

		let surface = chosen.material;

		Some(Self {
			rows,
			color,
			normal,
			tint: (surface.base_color * chosen.tint)
				.extend(surface.opacity)
				.to_array(),
			surface: [surface.metallic, surface.roughness, chosen.decal.fading(), 0.0],
		})
	}
}

/// Where the pictures are laid out, before any of them is written.
#[derive(Debug, PartialEq, Eq)]
struct Packed {
	/// How big the atlas is, in texels of its largest level.
	size: (u32, u32),

	/// Where each picture lands, in the order they were asked for: `[x, y,
	/// width, height]` in texels of the atlas's largest level.
	spots: Vec<[u32; 4]>,
}

/// Every picture the world's decals throw, in one texture.
pub(crate) struct Atlas {
	/// The whole of it, read as a color: what a color picture is sampled
	/// through.
	colors: TextureView,

	/// The same texels read as numbers: what a normal map is sampled through.
	numbers: TextureView,

	/// How big it is, in texels of its largest level.
	size: (u32, u32),

	/// Which pictures were asked for last, sorted and each once.
	asked: Vec<Key>,

	/// Where each of them landed, in the same order; empty when they did not
	/// fit.
	spots: Vec<[u32; 4]>,
}

impl Atlas {
	/// An atlas holding nothing: one cell, clear in every level, so that the
	/// group it is bound in is whole before a decal exists.
	///
	/// @param device - the device to build against
	#[must_use]
	pub(crate) fn new(device: &Device) -> Self {
		let (colors, numbers) = views(&texture(device, (CELL, CELL)));

		Self {
			colors,
			numbers,
			size: (CELL, CELL),
			asked: Vec::new(),
			spots: Vec::new(),
		}
	}

	/// The atlas read as a color.
	#[must_use]
	pub(crate) const fn colors(&self) -> &TextureView { &self.colors }

	/// The atlas read as numbers.
	#[must_use]
	pub(crate) const fn numbers(&self) -> &TextureView { &self.numbers }

	/// Brings it level with the pictures the world's decals throw.
	///
	/// Nothing at all when the set is the one it already holds, which is every
	/// frame but the ones a decal was added in or given a new picture in.
	/// Otherwise the whole of it is laid out and written again: a picture that
	/// grew moves everything packed after it, and a set of pictures is small.
	///
	/// @param device - the device to build against
	/// @param queue - where the pictures are written through
	/// @param textures - the world's registry, which holds every picture's
	/// levels
	/// @param wanted - the keys, sorted and each once; @ref [`pictures`]
	/// @return whether it was rebuilt, which is when a group holding its two
	/// views has to be rebuilt as well
	#[must_use]
	pub(crate) fn ensure(
		&mut self,
		device: &Device,
		queue: &Queue,
		textures: &Textures,
		wanted: &[Key],
	) -> bool {
		if wanted == self.asked.as_slice() {
			return false;
		}

		let pictures: Vec<Option<&TextureData>> = wanted
			.iter()
			.map(|(slot, _)| {
				textures
					.get(TextureId::new(*slot))
					.map(Entry::value)
			})
			.collect();

		let Some((firsts, packed)) = laid_out(&pictures) else {
			warn!(
				pictures = wanted.len(),
				"the decals' pictures do not fit in one atlas, so none of them is painted"
			);
			*self = Self {
				asked: wanted.to_vec(),
				..Self::new(device)
			};

			return true;
		};

		let texture = texture(device, packed.size);
		for ((picture, first), spot) in pictures.iter().zip(&firsts).zip(&packed.spots) {
			if let Some(data) = picture {
				write_picture(queue, &texture, *spot, *first, data);
			}
		}

		let (colors, numbers) = views(&texture);
		*self = Self {
			colors,
			numbers,
			size: packed.size,
			asked: wanted.to_vec(),
			spots: packed.spots,
		};

		true
	}

	/// Where a picture is, as a fraction of the atlas. @ref [`inset`].
	///
	/// @param key - the picture
	/// @return `[u, v, width, height]`, or nothing for a picture it does not
	/// hold
	#[must_use]
	pub(crate) fn rect(&self, key: Key) -> Option<[f32; 4]> {
		let at = self.asked.binary_search(&key).ok()?;

		self.spots
			.get(at)
			.map(|spot| inset(*spot, self.size))
	}
}

/// A spot as a fraction of the atlas, inset by half a texel on every side.
///
/// The inset is what keeps a sample at the picture's own edge reading the
/// picture's edge rather than half of the gap beside it; a picture of one
/// texel is then a rectangle of no size, which reads that texel wherever it is
/// asked.
///
/// @param spot - `[x, y, width, height]` in texels
/// @param size - how big the atlas is
/// @return `[u, v, width, height]`
fn inset(spot: [u32; 4], size: (u32, u32)) -> [f32; 4] {
	let [x, y, width, height] = spot;
	let (across, down) = (whole(size.0), whole(size.1));

	[
		(whole(x) + 0.5) / across,
		(whole(y) + 0.5) / down,
		(whole(width) - 1.0).max(0.0) / across,
		(whole(height) - 1.0).max(0.0) / down,
	]
}

/// The atlas's texture, empty.
///
/// Stored as numbers and read two ways: through a view that decodes sRGB, for
/// the color pictures, whose bytes are sRGB, and through a plain one for the
/// normal maps, whose bytes are directions. One texture and two views rather
/// than two atlases, because the bytes of both kinds are copied in as they are
/// and only the reading differs.
///
/// @param device - the device to build against
/// @param size - how big, in texels of the largest level
fn texture(device: &Device, size: (u32, u32)) -> Texture {
	device.create_texture(&TextureDescriptor {
		label: Some("decal atlas"),
		size: Extent3d {
			width: size.0,
			height: size.1,
			depth_or_array_layers: 1,
		},
		mip_level_count: LEVELS,
		sample_count: 1,
		dimension: TextureDimension::D2,
		format: TextureFormat::Rgba8Unorm,
		usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
		view_formats: &[TextureFormat::Rgba8UnormSrgb],
	})
}

/// The two ways the atlas is read: as colors, then as numbers.
fn views(texture: &Texture) -> (TextureView, TextureView) {
	(
		texture.create_view(&TextureViewDescriptor {
			label: Some("decal colors"),
			format: Some(TextureFormat::Rgba8UnormSrgb),
			..TextureViewDescriptor::default()
		}),
		texture.create_view(&TextureViewDescriptor {
			label: Some("decal numbers"),
			..TextureViewDescriptor::default()
		}),
	)
}

/// Where every picture goes, shrinking the largest until they all fit.
///
/// @param pictures - each picture, or nothing for a key the registry did not
/// answer, which is packed as one texel and never written
/// @return each picture's first level and where it landed, or nothing when
/// even at [`MIN_PICTURE`] they do not fit in [`MAX_SIDE`]
fn laid_out(pictures: &[Option<&TextureData>]) -> Option<(Vec<u32>, Packed)> {
	let mut cap = MAX_PICTURE;

	loop {
		let shrunk: Vec<(u32, (u32, u32))> = pictures
			.iter()
			.map(|picture| picture.map_or((0, (1, 1)), |data| shrunk(data, cap)))
			.collect();
		let sizes: Vec<(u32, u32)> = shrunk.iter().map(|(_, size)| *size).collect();

		if let Some(packed) = pack(&sizes) {
			return Some((
				shrunk
					.into_iter()
					.map(|(first, _)| first)
					.collect(),
				packed,
			));
		}

		if cap <= MIN_PICTURE {
			return None;
		}

		cap /= 2;
	}
}

/// Which of a picture's levels is packed as its largest, and how big that is.
///
/// The first whose longer side is no more than the cap, so a picture two
/// thousand texels across is packed from its third level at a cap of five
/// hundred. A picture with no level that small is packed from its smallest.
///
/// @param data - the picture
/// @param cap - the longest side any picture may be packed at
fn shrunk(data: &TextureData, cap: u32) -> (u32, (u32, u32)) {
	let last = u32::try_from(data.levels.len())
		.unwrap_or(1)
		.saturating_sub(1);
	let mut first = 0;

	while first < last && (data.width.max(data.height) >> first) > cap {
		first += 1;
	}

	(first, data.level_size(first))
}

/// Lays pictures of these sizes out in one atlas: a skyline, on a grid of
/// [`CELL`] texels.
///
/// Widest first, then tallest, each put where the skyline under it is lowest.
/// The atlas starts at eight cells wide, or the widest picture if that is
/// wider, and doubles until the pictures fit no more than twice as tall as it
/// is wide; its height is then rounded up to a power of two. Every picture
/// sits half a cell in from its cell's corner.
///
/// @param sizes - each picture's width and height, in texels
/// @return the atlas's size and where each picture landed, in the order
/// given, or nothing when they do not fit in [`MAX_SIDE`]
fn pack(sizes: &[(u32, u32)]) -> Option<Packed> {
	let cells: Vec<(u32, u32)> = sizes
		.iter()
		.map(|&(width, height)| (units(width), units(height)))
		.collect();
	let mut order: Vec<usize> = (0..cells.len()).collect();
	order.sort_by_key(|at| core::cmp::Reverse(cells[*at]));

	let widest = cells
		.iter()
		.map(|&(width, _)| width)
		.max()
		.unwrap_or(1);
	let mut across = widest.next_power_of_two().max(8);

	while across.saturating_mul(CELL) <= MAX_SIDE {
		if let Some(packed) = packed_across(sizes, &cells, &order, across) {
			return Some(packed);
		}

		across *= 2;
	}

	None
}

/// One attempt at [`pack`], at one width.
///
/// @param sizes - each picture's size in texels
/// @param cells - each picture's size in cells
/// @param order - which picture goes down first, and after it which
/// @param across - how many cells wide
/// @return the layout, or nothing when it came out more than twice as tall as
/// it is wide or taller than the atlas may be
fn packed_across(
	sizes: &[(u32, u32)],
	cells: &[(u32, u32)],
	order: &[usize],
	across: u32,
) -> Option<Packed> {
	let mut skyline = vec![0; usize::try_from(across).ok()?];
	let mut spots = vec![[0; 4]; cells.len()];
	let mut tallest = 0;

	for &at in order {
		let (width, height) = cells[at];
		let (x, y) = settle(&mut skyline, width, height)?;
		let (texels_across, texels_down) = sizes[at];

		spots[at] = [x * CELL + CELL / 2, y * CELL + CELL / 2, texels_across, texels_down];
		tallest = tallest.max(y + height);
	}

	let down = tallest
		.saturating_mul(CELL)
		.next_power_of_two()
		.max(CELL);

	if tallest > across.saturating_mul(2) || down > MAX_SIDE {
		return None;
	}

	Some(Packed { size: (across * CELL, down), spots })
}

/// Puts one rectangle of cells where the skyline under it is lowest, and
/// raises the skyline over it.
///
/// @param skyline - how high the packing stands over each column of cells
/// @param width - how many cells wide the rectangle is
/// @param height - how many tall
/// @return where it went, in cells, or nothing for one wider than the skyline
fn settle(skyline: &mut [u32], width: u32, height: u32) -> Option<(u32, u32)> {
	let span = usize::try_from(width).ok()?;

	if span == 0 || span > skyline.len() {
		return None;
	}

	let (mut best, mut lowest) = (0, u32::MAX);
	for start in 0..=skyline.len() - span {
		let under = skyline[start..start + span]
			.iter()
			.copied()
			.max()
			.unwrap_or(0);

		if under < lowest {
			(best, lowest) = (start, under);
		}
	}

	for column in &mut skyline[best..best + span] {
		*column = lowest + height;
	}

	Some((u32::try_from(best).ok()?, lowest))
}

/// How many cells a picture of this many texels takes: itself and half a cell
/// either side of it, rounded up.
const fn units(texels: u32) -> u32 { texels.saturating_add(CELL).div_ceil(CELL) }

/// Copies one picture into the atlas, a level of it into each level there.
///
/// Level `first` of the picture into the atlas's largest, the next into the
/// next, and so on. A picture with fewer levels than the atlas repeats its
/// smallest where it runs out, which for a full chain is its last single texel
/// and is the answer a sample from that far away wants. A level whose bytes do
/// not add up, or that would not fit the room the picture has at that level,
/// is left out rather than written, because either is a validation error
/// inside a driver.
///
/// @param queue - where the writes go
/// @param texture - the atlas
/// @param spot - where the picture lands, `[x, y, width, height]`
/// @param first - which of the picture's levels is its largest here
/// @param data - the picture
fn write_picture(
	queue: &Queue,
	texture: &Texture,
	spot: [u32; 4],
	first: u32,
	data: &TextureData,
) {
	let last = u32::try_from(data.levels.len())
		.unwrap_or(1)
		.saturating_sub(1);
	let texel = u32::try_from(data.texel.bytes()).unwrap_or(4);

	for level in 0..LEVELS {
		let source = first.saturating_add(level).min(last);
		let (width, height) = data.level_size(source);
		let room = ((spot[2] >> level).max(1), (spot[3] >> level).max(1));

		let Some(bytes) = usize::try_from(source)
			.ok()
			.and_then(|at| data.levels.get(at))
		else {
			continue;
		};

		if bytes.len() != data.level_bytes(source) || width > room.0 || height > room.1 {
			continue;
		}

		queue.write_texture(
			TexelCopyTextureInfo {
				texture,
				mip_level: level,
				origin: Origin3d {
					x: spot[0] >> level,
					y: spot[1] >> level,
					z: 0,
				},
				aspect: TextureAspect::All,
			},
			bytes,
			TexelCopyBufferLayout {
				offset: 0,
				bytes_per_row: Some(width * texel),
				rows_per_image: Some(height),
			},
			Extent3d { width, height, depth_or_array_layers: 1 },
		);
	}
}

/// A texel count as a float, without an `as`: nothing here is wider than
/// [`MAX_SIDE`].
fn whole(texels: u32) -> f32 { f32::from(u16::try_from(texels).unwrap_or(u16::MAX)) }

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{Renderable, Texel},
		glam::{Quat, Vec4},
	};

	use super::*;

	/// A world with a unit decal standing at each point, each of the order
	/// beside it.
	fn decals_at(standing: &[(Vec3, i32)]) -> World {
		let mut world = World::new();

		for &(position, order) in standing {
			let id = world.entities.spawn_at(Transform::at(position));
			assert!(
				world
					.entities
					.set_decal(id, Decal { order, ..Decal::BOX })
			);
		}

		world
	}

	/// Which slots a frame carried, in the order it paints them.
	fn slots(carried: &[Chosen]) -> Vec<usize> {
		carried
			.iter()
			.map(|it| it.entity.slot())
			.collect()
	}

	/// A decal standing somewhere, as a frame would have chosen it.
	fn standing(at: Transform, material: Material, tint: Vec3, decal: Decal) -> Chosen {
		Chosen {
			entity: EntityId::NONE,
			decal,
			at,
			material,
			tint,
			distance: 0.0,
		}
	}

	/// Where a point lands in a packed decal's own space.
	fn inside(paint: &Paint, point: Vec3) -> Vec3 {
		let point = point.extend(1.0);
		let along = |axis: usize| Vec4::from_array(paint.rows[axis]).dot(point);

		Vec3::new(along(0), along(1), along(2))
	}

	/// Whether two spots, each grown by half a cell on every side, overlap.
	fn crowd(one: [u32; 4], other: [u32; 4]) -> bool {
		let half = CELL / 2;
		let span =
			|spot: [u32; 4], axis: usize| (spot[axis] - half, spot[axis] + spot[axis + 2] + half);
		let ((left, right), (top, bottom)) = (span(one, 0), span(one, 1));
		let ((other_left, other_right), (other_top, other_bottom)) =
			(span(other, 0), span(other, 1));

		left < other_right && other_left < right && top < other_bottom && other_top < bottom
	}

	/// A picture of this size with every level of its chain, all nought.
	fn chain(width: u32, height: u32) -> TextureData {
		let mut data = TextureData {
			width,
			height,
			faces: 1,
			texel: Texel::Rgba8Srgb,
			levels: Vec::new(),
		};

		for level in 0..TextureData::full_chain(width, height) {
			let bytes = data.level_bytes(level);
			data.levels.push(vec![0; bytes]);
		}

		data
	}

	/// Whether two records are the same numbers to within a hair.
	fn close(one: [f32; 4], other: [f32; 4]) -> bool {
		one.iter()
			.zip(other)
			.all(|(a, b)| (a - b).abs() < 1.0e-6)
	}

	#[test]
	fn a_cell_holds_its_picture_and_half_a_cell_either_side_of_it() {
		for texels in [1, 15, 16, 31, 32, 33, 50, 64, 100, 1024] {
			let room = units(texels) * CELL;

			assert!(
				room >= texels + CELL,
				"{texels} texels got {room}, short of a half cell a side"
			);
			assert!(
				room < texels + 2 * CELL,
				"and no more than a cell over: {texels} got {room}"
			);
		}
	}

	#[test]
	fn every_picture_lands_inside_on_whole_texels_and_clear_of_every_other() {
		let sizes = [(512, 512), (256, 128), (50, 70), (1, 1), (1024, 256), (64, 64), (33, 900)];
		let packed = pack(&sizes).expect("seven pictures fit");
		let (width, height) = packed.size;

		assert!(width.is_power_of_two() && height.is_power_of_two(), "{width} by {height}");
		assert!(height <= width * 2, "and no more than twice as tall as it is wide");

		for (at, spot) in packed.spots.iter().enumerate() {
			assert_eq!((spot[2], spot[3]), sizes[at], "a picture keeps its size: {spot:?}");
			assert!(
				spot[0].is_multiple_of(CELL / 2) && spot[1].is_multiple_of(CELL / 2),
				"on a whole texel {LEVELS} levels down: {spot:?}"
			);
			assert!(
				spot[0] + spot[2] + CELL / 2 <= width && spot[1] + spot[3] + CELL / 2 <= height,
				"with half a cell clear inside the atlas's edge: {spot:?}"
			);

			for other in packed.spots.iter().skip(at + 1) {
				assert!(!crowd(*spot, *other), "{spot:?} and {other:?} are closer than a cell");
			}
		}
	}

	#[test]
	fn an_atlas_grows_rather_than_crowding_its_pictures() {
		let sizes = [(256, 256); 40];
		let packed = pack(&sizes).expect("forty pictures fit");

		assert!(packed.size.0 >= 1024, "forty of them do not fit eight cells wide");

		for (at, spot) in packed.spots.iter().enumerate() {
			for other in packed.spots.iter().skip(at + 1) {
				assert!(!crowd(*spot, *other), "{spot:?} and {other:?} are closer than a cell");
			}
		}
	}

	#[test]
	fn what_no_atlas_can_hold_is_refused_rather_than_packed_past_its_edge() {
		assert!(pack(&[(9000, 16)]).is_none(), "a picture wider than the widest atlas");
		assert!(pack(&[(4096, 4096); 8]).is_none(), "and eight that would be twice its area");
		assert_eq!(
			pack(&[]).map(|it| it.size),
			Some((256, CELL)),
			"and nothing is an empty strip"
		);
	}

	#[test]
	fn a_picture_past_the_cap_is_packed_from_a_smaller_level_of_itself() {
		assert_eq!(
			shrunk(&chain(2048, 512), 512),
			(2, (512, 128)),
			"the first level whose longer side is the cap"
		);
		assert_eq!(shrunk(&chain(64, 64), 512), (0, (64, 64)), "a picture under it is whole");

		let alone = TextureData {
			width: 2048,
			height: 512,
			faces: 1,
			texel: Texel::Rgba8Srgb,
			levels: vec![Vec::new()],
		};

		assert_eq!(
			shrunk(&alone, 512).0,
			0,
			"a picture with no smaller level is packed from the one it has"
		);
	}

	#[test]
	fn pictures_that_do_not_fit_whole_are_shrunk_until_they_do() {
		// the levels are empty because nothing here reads them: which level is
		// packed first is a matter of how many there are and how big the first
		// one is
		let big = TextureData {
			width: 1024,
			height: 1024,
			faces: 1,
			texel: Texel::Rgba8Srgb,
			levels: vec![Vec::new(); 11],
		};
		let pictures = vec![Some(&big); 96];
		let (firsts, packed) = laid_out(&pictures).expect("they fit, smaller");

		assert!(firsts.iter().all(|first| *first >= 1), "none of them whole: {firsts:?}");
		assert!(packed.size.0 <= MAX_SIDE && packed.size.1 <= MAX_SIDE, "{:?}", packed.size);
	}

	#[test]
	fn a_spot_is_a_fraction_of_the_atlas_inset_by_half_a_texel() {
		assert!(close(inset([16, 48, 64, 32], (256, 128)), [
			16.5 / 256.0,
			48.5 / 128.0,
			63.0 / 256.0,
			31.0 / 128.0
		]));
		assert!(
			close(inset([16, 16, 1, 1], (64, 64)), [16.5 / 64.0, 16.5 / 64.0, 0.0, 0.0]),
			"a picture of one texel reads that texel wherever it is asked"
		);
	}

	#[test]
	fn a_frame_carries_the_nearest_it_has_room_for_and_paints_them_by_order() {
		// four in a row: the two nearest are carried, and of those the farther
		// one is painted first because its order is the lower
		let world = decals_at(&[
			(Vec3::new(0.0, 0.0, -2.0), 5),
			(Vec3::new(0.0, 0.0, -4.0), 1),
			(Vec3::new(0.0, 0.0, -6.0), 0),
			(Vec3::new(0.0, 0.0, -8.0), -3),
		]);
		let mut carried = Vec::new();
		chosen(&world, Vec3::ZERO, None, 2, &mut carried);

		let orders: Vec<i32> = carried.iter().map(|it| it.decal.order).collect();

		assert_eq!(orders, vec![1, 5], "the nearest two, the lower order first");
	}

	#[test]
	fn two_of_one_order_are_painted_in_the_order_of_their_slots_wherever_the_eye_is() {
		let world = decals_at(&[(Vec3::new(0.0, 0.0, -2.0), 0), (Vec3::new(0.0, 0.0, -4.0), 0)]);
		let mut carried = Vec::new();

		for eye in [Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0)] {
			chosen(&world, eye, None, MAX_DECALS, &mut carried);

			assert_eq!(slots(&carried), vec![0, 1], "seen from {eye}, the first is under");
		}
	}

	#[test]
	fn a_hidden_decal_a_flattened_box_and_an_entity_that_paints_nothing_are_not_carried() {
		let mut world =
			decals_at(&[(Vec3::NEG_Z, 0), (Vec3::NEG_Z * 3.0, 0), (Vec3::NEG_Z * 4.0, 0)]);
		let ids: Vec<EntityId> = world.entities.iter().map(|(id, ..)| id).collect();
		let flat = world.entities.spawn_at(Transform {
			scale: Vec3::new(1.0, 1.0, 0.0),
			..Transform::at(Vec3::NEG_Z * 2.0)
		});
		assert!(world.entities.set_decal(flat, Decal::BOX));
		assert!(world.entities.set_hidden(ids[0], true));
		assert!(world.entities.set_decal(ids[1], Decal::NONE));

		let mut carried = Vec::new();
		chosen(&world, Vec3::ZERO, None, MAX_DECALS, &mut carried);

		assert_eq!(slots(&carried), vec![ids[2].slot()], "only the one nothing is wrong with");
	}

	#[test]
	fn a_decal_whose_box_the_view_cannot_see_is_not_carried() {
		let mut world =
			decals_at(&[(Vec3::new(0.0, 0.0, -5.0), 0), (Vec3::new(0.0, 0.0, 5.0), 0)]);
		world.camera.position = Vec3::ZERO;
		world.camera.target = Vec3::NEG_Z;

		let view = Frustum::of(world.camera.view_projection(1.0));
		let mut carried = Vec::new();
		chosen(&world, Vec3::ZERO, Some(&view), MAX_DECALS, &mut carried);

		assert_eq!(slots(&carried), vec![0], "the one ahead and not the one behind the eye");
	}

	#[test]
	fn how_near_a_decal_is_is_how_near_its_box_reaches_and_not_where_its_middle_is() {
		// a small decal three off and a big one twelve off whose box reaches
		// past the small one towards the eye: a frame with room for one
		// carries the big one, the lamps' rule of a reach counting towards how
		// near a thing is
		let mut world = decals_at(&[(Vec3::new(0.0, 0.0, -3.0), 0)]);
		let big = world.entities.spawn_at(Transform {
			scale: Vec3::splat(20.0),
			..Transform::at(Vec3::new(0.0, 0.0, -12.0))
		});
		assert!(world.entities.set_decal(big, Decal::BOX));

		let mut carried = Vec::new();
		chosen(&world, Vec3::ZERO, None, 1, &mut carried);

		assert_eq!(slots(&carried), vec![big.slot()], "the box that reaches nearer is nearer");
	}

	#[test]
	fn a_decal_whose_corner_is_in_view_is_carried_though_its_middle_is_behind_the_eye() {
		// a cube turned so that a corner points down the view, its middle
		// behind the eye by less than half its diagonal: the corner is in
		// front of the eye and paints what is there, and only a ball as big as
		// the box itself says so
		let mut world = World::new();
		world.camera.position = Vec3::ZERO;
		world.camera.target = Vec3::NEG_Z;

		let at = Transform {
			position: Vec3::new(0.0, 0.0, 2.8),
			rotation: Quat::from_rotation_arc(Vec3::ONE.normalize(), Vec3::NEG_Z),
			scale: Vec3::splat(4.0),
		};
		let cube = world.entities.spawn_at(at);
		assert!(world.entities.set_decal(cube, Decal::BOX));

		let view = Frustum::of(world.camera.view_projection(1.0));
		let corner = at.matrix().transform_point3(Vec3::splat(0.5));

		assert!(view.holds_ball(corner, 0.0), "the corner is in the view: {corner}");
		assert!(!view.holds_ball(at.position, 0.0), "and the middle is not");

		let mut carried = Vec::new();
		chosen(&world, Vec3::ZERO, Some(&view), MAX_DECALS, &mut carried);

		assert_eq!(slots(&carried), vec![cube.slot()], "so the decal is carried");
	}

	#[test]
	fn every_picture_a_decal_throws_is_asked_for_once_and_nothing_a_mesh_throws_is() {
		let mut world = World::new();
		let picture = world
			.textures
			.insert("test/splash", chain(64, 64));
		let bumps = world
			.textures
			.insert("test/splash_normal", chain(32, 32));
		let painted = world
			.materials
			.insert("test/splash", Material::textured(picture).bumped(bumps));
		let plain = world
			.materials
			.insert("test/plain", Material::textured(bumps));

		for (material, decal) in
			[(painted, Decal::BOX), (painted, Decal::BOX), (plain, Decal::NONE)]
		{
			let id = world.entities.spawn();
			assert!(
				world
					.entities
					.set_renderable(id, Renderable { material, ..Renderable::NOTHING })
			);
			assert!(world.entities.set_decal(id, decal));
		}

		let mut keys = Vec::new();
		pictures(&world, &mut keys);

		let mut expected = vec![
			key(&world.textures, picture).expect("registered"),
			key(&world.textures, bumps).expect("registered"),
		];
		expected.sort_unstable();

		assert_eq!(keys, expected, "the two pictures, once each, and nothing of the mesh's");
	}

	#[test]
	fn a_decal_with_no_picture_throws_its_tint_and_a_flat_map_throws_nothing() {
		assert_eq!(thrown(&Material::DEFAULT), [None, None], "a plain material throws its tint");
		assert_eq!(
			thrown(&Material {
				normal: TextureId::NONE,
				..Material::DEFAULT
			}),
			[None, None],
			"and a map of the white texel is not a direction"
		);
	}

	#[test]
	fn the_rows_carry_a_point_into_the_box_whichever_way_it_is_turned() {
		// turned to look down, which is how a decal on a floor stands
		let at = Transform {
			position: Vec3::new(3.0, 1.0, -2.0),
			rotation: Quat::from_rotation_x(-core::f32::consts::FRAC_PI_2),
			scale: Vec3::new(2.0, 4.0, 0.5),
		};
		let paint = Paint::of(
			&standing(at, Material::DEFAULT, Vec3::ONE, Decal::BOX),
			NO_PICTURE,
			NO_PICTURE,
		)
		.expect("a box with an inside");

		assert!(inside(&paint, at.position).abs().max_element() < 1.0e-5, "the middle is nought");

		let corner = at.matrix().transform_point3(Vec3::splat(0.5));
		assert!(
			(inside(&paint, corner) - Vec3::splat(0.5))
				.abs()
				.max_element()
				< 1.0e-5,
			"and a corner is a half along every axis"
		);
		assert!(
			inside(&paint, at.position + Vec3::NEG_Y * 0.2).z < 0.0,
			"and below a box that looks down is further along the way it throws"
		);
	}

	#[test]
	fn a_decal_is_packed_with_its_material_its_tint_and_its_fade() {
		let material = Material {
			base_color: Vec3::new(0.5, 1.0, 1.0),
			metallic: 0.25,
			roughness: 0.75,
			opacity: 0.5,
			..Material::DEFAULT
		};
		let decal = Decal { fade: 3.0, ..Decal::BOX };
		let paint = Paint::of(
			&standing(Transform::IDENTITY, material, Vec3::new(1.0, 0.5, 1.0), decal),
			[0.1, 0.2, 0.3, 0.4],
			NO_PICTURE,
		)
		.expect("a unit box");

		assert!(close(paint.tint, [0.5, 0.5, 1.0, 0.5]), "{:?}", paint.tint);
		assert!(
			close(paint.surface, [0.25, 0.75, colby_core::abi::MAX_FADE, 0.0]),
			"{:?}",
			paint.surface
		);
		assert!(close(paint.color, [0.1, 0.2, 0.3, 0.4]), "the picture where it was put");
		assert!(close(paint.normal, [0.0; 4]), "and no map is all nought");
	}

	#[test]
	fn a_flattened_box_is_not_packed_at_all() {
		let flat = Transform {
			scale: Vec3::new(1.0, 0.0, 1.0),
			..Transform::IDENTITY
		};

		assert!(
			Paint::of(
				&standing(flat, Material::DEFAULT, Vec3::ONE, Decal::BOX),
				NO_PICTURE,
				NO_PICTURE,
			)
			.is_none(),
			"a box with no inside has an inverse that is not a number"
		);
	}

	#[test]
	fn the_decal_ceiling_and_what_the_variable_starts_at_are_the_same_number() {
		assert_eq!(room(DEFAULT_DECALS), MAX_DECALS);
	}

	#[test]
	fn asking_for_a_number_of_decals_nobody_could_mean_lands_somewhere_definite() {
		assert_eq!(room(-4.0), 0, "below nought is none");
		assert_eq!(room(7.9), 7, "a fraction is the whole number under it");
		assert_eq!(room(1.0e9), MAX_DECALS, "past the ceiling is the ceiling");
		assert_eq!(room(f32::NAN), MAX_DECALS, "and a number that is not one is the ceiling too");
	}
}
