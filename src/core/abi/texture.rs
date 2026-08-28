//! Textures: the texel layout, the pixels, and the host's registry of them.
//!
//! Here rather than in the engine for the same reason meshes are: [`TextureId`]
//! crosses the boundary, and a handle is only meaningful next to the table it
//! indexes. @ref [`registry`](super::registry) for the shape all three tables
//! share.
//!
//! Pixels arrive already in the layout the GPU takes. Nothing here decodes
//! anything - that happens in the asset compiler, offline, and what reaches
//! this module is the result.

use super::registry::{Entry, Registry};
use crate::registry_handle;

/// The name the always-present white texture is registered under.
pub const WHITE_NAME: &str = "white";

/// The name the always-present flat normal map is registered under.
pub const FLAT_NORMAL_NAME: &str = "flat_normal";

/// How a texel is laid out.
///
/// Two variants, and the difference between them is not the bytes but what the
/// bytes mean. It is an enum rather than an assumption because the asset format
/// stores it, and the day block compression arrives the reader has to be able
/// to say "this build does not know that layout" instead of reading four bytes
/// per texel from a file that has one.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Texel {
	/// Eight bits per channel, four channels, values in sRGB.
	///
	/// The GPU converts to linear on sample, which is what makes the lighting
	/// arithmetic correct without the shader doing anything about it.
	#[default]
	Rgba8Srgb = 0,

	/// The same bytes, sampled as they are.
	///
	/// For an image whose channels are not a color: a normal map, a roughness
	/// mask, anything the shader reads as a number. Putting one of those
	/// through the sRGB curve bends every value it holds, which on a normal map
	/// is a surface tilted the wrong way everywhere it is not flat.
	///
	/// It also changes how the mip chain is built - @ref
	/// `colby_asset::texture::build_chain`. Averaging colors means averaging
	/// the light they stand for; averaging numbers means averaging the numbers.
	Rgba8Unorm = 1,
}

impl Texel {
	/// How many bytes one texel takes.
	#[must_use]
	pub const fn bytes(self) -> usize {
		match self {
			| Self::Rgba8Srgb | Self::Rgba8Unorm => 4,
		}
	}

	/// Whether sampling this layout runs the values through a transfer
	/// function.
	#[must_use]
	pub const fn is_color(self) -> bool { matches!(self, Self::Rgba8Srgb) }

	/// The number the asset format stores.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "a fieldless repr(u32) enum casts to its own discriminant, which is exactly \
		          the number being asked for"
	)]
	pub const fn code(self) -> u32 { self as u32 }

	/// The layout a stored number stands for.
	///
	/// @param code - what the file said
	/// @return the layout, or `None` when this build does not know it
	#[must_use]
	pub const fn from_code(code: u32) -> Option<Self> {
		match code {
			| 0 => Some(Self::Rgba8Srgb),
			| 1 => Some(Self::Rgba8Unorm),
			| _ => None,
		}
	}
}

/// An image and its mip chain, as plain data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureData {
	/// Width of the largest level, in texels.
	pub width: u32,

	/// Height of the largest level, in texels.
	pub height: u32,

	/// How its texels are laid out.
	pub texel: Texel,

	/// One entry per mip level, largest first, always at least one.
	///
	/// A chain rather than one image because a floor seen at a grazing angle
	/// without mips does not look like a floor, it looks like static.
	pub levels: Vec<Vec<u8>>,
}

impl TextureData {
	/// The one texel every material gets when it names no image.
	///
	/// White, so that multiplying by it changes nothing - which is what lets a
	/// flat-colored material and a textured one go through the same shader and
	/// the same bind group layout, with no branch and no second pipeline.
	#[must_use]
	pub fn white() -> Self {
		Self {
			width: 1,
			height: 1,
			texel: Texel::Rgba8Srgb,
			levels: vec![vec![0xFF; 4]],
		}
	}

	/// The one texel every material gets when it names no normal map.
	///
	/// A direction rather than a color, and the direction is straight out of
	/// the surface: `(0.5, 0.5, 1.0)` is what `(0, 0, 1)` becomes once it is
	/// folded into `0 ..= 1`. Its whole job is the white texel's - to make an
	/// unmapped material a mapped one whose map says nothing, so that there is
	/// no branch in the shader and no second pipeline.
	///
	/// A hundred and twenty-eight rather than a hundred and twenty-seven and a
	/// half, because a byte cannot be a half. The shader normalizes what it
	/// reads, so the four-thousandth of a unit that costs is gone by the time
	/// anything uses it.
	#[must_use]
	pub fn flat_normal() -> Self {
		Self {
			width: 1,
			height: 1,
			texel: Texel::Rgba8Unorm,
			levels: vec![vec![128, 128, 255, 255]],
		}
	}

	/// How many mip levels a full chain for this size would have.
	///
	/// @param width - the largest level's width
	/// @param height - its height
	/// @return at least one, down to the level that is a single texel
	#[must_use]
	pub const fn full_chain(width: u32, height: u32) -> u32 {
		let mut longest = if width > height { width } else { height };
		let mut levels = 1;

		while longest > 1 {
			longest /= 2;
			levels += 1;
		}

		levels
	}

	/// The size of one mip level, never smaller than one texel either way.
	///
	/// @param level - which level, zero being the largest
	#[must_use]
	pub const fn level_size(&self, level: u32) -> (u32, u32) {
		let shift = if level > 31 { 31 } else { level };
		let width = self.width >> shift;
		let height = self.height >> shift;

		(if width > 0 { width } else { 1 }, if height > 0 { height } else { 1 })
	}

	/// How many bytes one mip level should hold.
	#[must_use]
	pub fn level_bytes(&self, level: u32) -> usize {
		let (width, height) = self.level_size(level);
		let area = usize::try_from(width)
			.unwrap_or(0)
			.saturating_mul(usize::try_from(height).unwrap_or(0));

		area.saturating_mul(self.texel.bytes())
	}

	/// How many bytes the whole chain holds.
	#[must_use]
	pub fn bytes(&self) -> usize { self.levels.iter().map(Vec::len).sum() }

	/// Whether every level is present and the size it should be.
	///
	/// Checked wherever pixels enter the process, because the GPU would not: it
	/// would read past the end of a short level and show whatever is there.
	#[must_use]
	pub fn is_consistent(&self) -> bool {
		if self.width == 0 || self.height == 0 || self.levels.is_empty() {
			return false;
		}

		let Ok(count) = u32::try_from(self.levels.len()) else {
			return false;
		};

		if count > Self::full_chain(self.width, self.height) {
			return false;
		}

		(0..count).all(|level| {
			usize::try_from(level)
				.ok()
				.and_then(|index| self.levels.get(index))
				.is_some_and(|bytes| bytes.len() == self.level_bytes(level))
		})
	}
}

impl Default for TextureData {
	fn default() -> Self { Self::white() }
}

registry_handle! {
	/// A handle to a texture in the world's [`Textures`] registry.
	///
	/// Never removed, so it stays valid for the life of the process - which is
	/// what lets reloading an image leave every material pointing at it alone.
	TextureId
}

impl TextureId {
	/// The flat normal map, which says every surface is as it looks.
	///
	/// Always registered, in the same slot in every world, for the same reason
	/// the built-in meshes are: a material's default has to name something and
	/// a handle is only meaningful next to its table.
	pub const FLAT_NORMAL: Self = Self::new(2);
}

/// One entry of the texture registry.
pub type Texture = Entry<TextureData>;

/// Every texture the renderer can sample, addressed by [`TextureId`].
///
/// Slot zero is [`TextureId::NONE`] and is one white texel, so a material that
/// names no image samples something harmless rather than nothing at all.
#[derive(Clone, Debug)]
pub struct Textures {
	entries: Registry<TextureData>,
}

impl Textures {
	/// A registry holding the null texture, the white one - which are the same
	/// texel under two names - and the flat normal map.
	#[must_use]
	pub fn new() -> Self {
		let mut textures = Self {
			entries: Registry::new(TextureData::white()),
		};
		textures.insert(WHITE_NAME, TextureData::white());
		textures.insert(FLAT_NORMAL_NAME, TextureData::flat_normal());

		textures
	}

	/// Looks a texture up by name.
	///
	/// @return its handle, or [`TextureId::NONE`] if nothing answers to it
	#[must_use]
	pub fn find(&self, name: &str) -> TextureId { TextureId::new(self.entries.find(name)) }

	/// Registers pixels under a name, replacing whatever was there.
	///
	/// @return the handle, the same one as last time if the name is known
	pub fn insert(&mut self, name: &str, data: TextureData) -> TextureId {
		TextureId::new(self.entries.insert(name, data))
	}

	/// One texture, by handle.
	#[must_use]
	pub fn get(&self, id: TextureId) -> Option<&Texture> { self.entries.entry(id.index()) }

	/// How many textures there are, counting the null one.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Always `false`: slot zero always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// Every texture, in slot order.
	pub fn iter(&self) -> impl Iterator<Item = &Texture> { self.entries.iter() }
}

impl Default for Textures {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A texture of a size, with a full chain of the right lengths.
	fn chained(width: u32, height: u32) -> TextureData {
		let mut data = TextureData {
			width,
			height,
			texel: Texel::Rgba8Srgb,
			levels: Vec::new(),
		};

		for level in 0..TextureData::full_chain(width, height) {
			data.levels.push(vec![0; data.level_bytes(level)]);
		}

		data
	}

	#[test]
	fn a_texel_knows_its_size_and_its_stored_number() {
		assert_eq!(Texel::Rgba8Srgb.bytes(), 4, "four channels of eight bits");
		assert_eq!(Texel::Rgba8Srgb.code(), 0, "and the number the format writes");
		assert_eq!(Texel::from_code(0), Some(Texel::Rgba8Srgb), "which reads back");
		assert_eq!(Texel::from_code(7), None, "and an unknown one is refused, not guessed");
	}

	#[test]
	fn the_white_texture_is_one_opaque_texel() {
		let white = TextureData::white();

		assert_eq!((white.width, white.height), (1, 1), "one texel");
		assert_eq!(white.levels, vec![vec![0xFF, 0xFF, 0xFF, 0xFF]], "opaque white");
		assert!(white.is_consistent(), "and it is its own whole mip chain");
	}

	#[test]
	fn a_chain_runs_down_to_one_texel() {
		assert_eq!(TextureData::full_chain(1, 1), 1, "a single texel is its own chain");
		assert_eq!(TextureData::full_chain(8, 8), 4, "8, 4, 2, 1");
		assert_eq!(TextureData::full_chain(8, 2), 4, "the longer side decides");
		assert_eq!(TextureData::full_chain(256, 256), 9, "and it is a log, not a loop");
	}

	#[test]
	fn a_level_never_shrinks_below_one_texel() {
		let data = chained(8, 2);

		assert_eq!(data.level_size(0), (8, 2), "the largest level is the size given");
		assert_eq!(data.level_size(1), (4, 1), "halved, and clamped on the short side");
		assert_eq!(data.level_size(2), (2, 1), "which stays at one");
		assert_eq!(data.level_size(3), (1, 1), "down to a single texel");
	}

	#[test]
	fn a_full_chain_is_consistent_and_a_short_level_is_not() {
		let mut data = chained(8, 8);

		assert!(data.is_consistent(), "every level is the length it should be");
		assert_eq!(data.bytes(), 4 * (64 + 16 + 4 + 1), "and the total is the sum of them");

		data.levels[2].pop();

		assert!(!data.is_consistent(), "one byte short is caught");
	}

	#[test]
	fn a_texture_with_no_levels_or_no_size_is_not_consistent() {
		let mut data = chained(4, 4);
		data.levels.clear();

		assert!(!data.is_consistent(), "there has to be at least one level");

		let empty = TextureData {
			width: 0,
			height: 4,
			texel: Texel::Rgba8Srgb,
			levels: vec![Vec::new()],
		};

		assert!(!empty.is_consistent(), "and a zero dimension is not a texture");
	}

	#[test]
	fn a_new_registry_answers_to_the_built_in_names_and_to_nothing_else() {
		let textures = Textures::new();

		assert_eq!(textures.len(), 3, "the null texture, the white one and the flat normal");
		assert!(textures.find(WHITE_NAME).is_some(), "white is always there");
		assert_eq!(
			textures.find(FLAT_NORMAL_NAME),
			TextureId::FLAT_NORMAL,
			"and the flat normal is at the slot the constant names"
		);
		assert_eq!(textures.find("stone"), TextureId::NONE, "and nothing else is");
		assert!(!TextureId::NONE.is_some(), "the null handle knows it is null");
	}

	#[test]
	fn the_flat_normal_map_points_straight_out_of_the_surface() {
		let textures = Textures::new();
		let flat = textures
			.get(TextureId::FLAT_NORMAL)
			.expect("the slot is seeded");
		let texel = flat
			.value()
			.levels
			.first()
			.expect("it has its one level");

		assert_eq!(
			flat.value().texel,
			Texel::Rgba8Unorm,
			"a direction is not a color, so nothing bends it on the way in"
		);

		let direction: Vec<f32> = texel
			.iter()
			.take(3)
			.map(|channel| (f32::from(*channel) / 255.0).mul_add(2.0, -1.0))
			.collect();

		assert!(direction[0].abs() < 0.01, "no lean across, got {}", direction[0]);
		assert!(direction[1].abs() < 0.01, "and none along, got {}", direction[1]);
		assert!(direction[2] > 0.99, "and all of it outwards, got {}", direction[2]);
	}

	#[test]
	fn the_null_texture_samples_as_white() {
		let textures = Textures::new();
		let none = textures
			.get(TextureId::NONE)
			.expect("slot zero always exists");

		assert_eq!(
			none.value().levels,
			TextureData::white().levels,
			"a material naming no image multiplies by one"
		);
	}

	#[test]
	fn replacing_a_texture_keeps_its_handle_and_bumps_its_revision() {
		let mut textures = Textures::new();
		let first = textures.insert("stone", chained(8, 8));
		let second = textures.insert("stone", chained(4, 4));

		assert_eq!(first, second, "the handle survives, so materials need not be told");
		assert_eq!(
			textures.get(first).map(Texture::revision),
			Some(1),
			"and the revision is how the renderer finds out"
		);
		assert_eq!(
			textures
				.get(first)
				.map(|entry| entry.value().width),
			Some(4),
			"and the pixels really are the new ones"
		);
	}
}
