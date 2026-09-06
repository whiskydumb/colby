//! What is behind everything, when there is anything.
//!
//! Three colors and a word: the top of the sky, the band at eye level and what
//! is under it. That is Wicked's `horizon` and `zenith` sitting on the same
//! record as its sun and its gravity, and Godot's `ProceduralSkyMaterial`,
//! which is what a new world there gets by default. A cubemap is the other
//! shape the field ships and it needs a texture that has six faces, which this
//! engine's `.ctex` has no spelling for yet.
//!
//! **This is the sky that is drawn, not the sky that lights.** Unreal splits
//! them into two components for a reason worth copying: `USkyAtmosphere` is
//! the picture and `USkyLightComponent` is the light, and the second is a
//! *convolution* of an environment rather than a color anybody types. colby's
//! stand-in for the second is [`Stage::ambient`](super::scene::Stage::ambient)
//! and it stays exactly where it is - turning the sky on changes what is
//! behind the world and nothing about what is lit by it. The day there is an
//! environment probe, that is what feeds the ambient.
//!
//! **It costs a fragment nothing where geometry stands.** The sky is drawn
//! after everything opaque, at the far plane, with the depth test on and the
//! depth write off - so every pixel a wall covered is thrown away before it is
//! shaded. @ref `colby_engine`.

use super::field::{Field, field, word};
use crate::glam::Vec3;

/// What is drawn behind the world.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SkyKind {
	/// Nothing: whatever the clear color is, and that is the whole of it.
	#[default]
	None,

	/// A gradient from the ground, through the horizon, to the zenith.
	Gradient,
}

impl SkyKind {
	/// The word each kind is written as, in declaration order.
	///
	/// A file's vocabulary and an inspector's drop-down. A cubemap is the
	/// third word when there is a texture with six faces to name.
	pub const WORDS: &[&str] = &["none", "gradient"];

	/// The kind at a place in [`WORDS`](Self::WORDS), if there is one.
	///
	/// @param index - the place
	#[must_use]
	pub const fn at(index: u32) -> Option<Self> {
		match index {
			| 0 => Some(Self::None),
			| 1 => Some(Self::Gradient),
			| _ => None,
		}
	}

	/// Where this kind is in [`WORDS`](Self::WORDS).
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "the discriminant is the place in the list, by declaration order"
	)]
	pub const fn index(self) -> u32 { self as u32 }

	/// The word this kind is written as.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "u32 to usize is lossless on every target this builds for, and try_from is not \
		          available in a const fn"
	)]
	pub const fn word(self) -> &'static str { Self::WORDS[self.index() as usize] }

	/// Whether a sky of this kind is drawn at all.
	#[must_use]
	pub const fn is_drawn(self) -> bool { !matches!(self, Self::None) }
}

/// The sky: what kind, and the three colors it is made of.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sky {
	/// What is drawn, or [`SkyKind::None`] for the clear color alone.
	pub kind: SkyKind,

	/// Straight up, linear RGB.
	pub zenith: Vec3,

	/// The band at eye level.
	pub horizon: Vec3,

	/// Straight down.
	///
	/// A color of its own rather than a mirror of the zenith, because the half
	/// under the horizon is the ground rather than more sky. Unreal keeps the
	/// same distinction on the light half of it, as `LowerHemisphereColor`.
	pub ground: Vec3,
}

impl Sky {
	/// Its fields, for an inspector, a reader and a writer. @ref
	/// [`field`](super::field).
	pub const FIELDS: &[Field<Self>] = &[
		word!(
			"kind",
			kind,
			SkyKind::WORDS,
			SkyKind::at,
			SkyKind::index,
			"what is drawn behind the world, or none"
		),
		field!(Color, "zenith", zenith, "the color straight up"),
		field!(Color, "horizon", horizon, "the color at eye level"),
		field!(Color, "ground", ground, "the color straight down"),
	];
	/// No sky at all: the clear color, and nothing over it.
	///
	/// The three colors are still a sky somebody would recognize, so that
	/// turning the word from `none` to `gradient` shows one rather than a
	/// black dome nobody can tell from the default clear.
	pub const NONE: Self = Self {
		kind: SkyKind::None,
		zenith: Vec3::new(0.12, 0.24, 0.52),
		horizon: Vec3::new(0.55, 0.66, 0.82),
		ground: Vec3::new(0.10, 0.09, 0.08),
	};

	/// A gradient of three colors.
	///
	/// @param zenith - straight up
	/// @param horizon - at eye level
	/// @param ground - straight down
	#[must_use]
	pub const fn gradient(zenith: Vec3, horizon: Vec3, ground: Vec3) -> Self {
		Self {
			kind: SkyKind::Gradient,
			zenith,
			horizon,
			ground,
		}
	}

	/// The three colors a world starts with, drawn.
	#[must_use]
	pub const fn day() -> Self { Self { kind: SkyKind::Gradient, ..Self::NONE } }

	/// Whether anything is drawn behind the world.
	#[must_use]
	pub const fn is_drawn(self) -> bool { self.kind.is_drawn() }
}

impl Default for Sky {
	fn default() -> Self { Self::NONE }
}

#[cfg(test)]
mod tests {
	use super::{
		super::field::{Kind, Value},
		*,
	};

	#[test]
	fn a_kind_is_its_place_in_the_words() {
		for (index, word) in SkyKind::WORDS.iter().enumerate() {
			let index = u32::try_from(index).expect("two of them");
			let kind = SkyKind::at(index).expect("every place has a kind");

			assert_eq!(kind.index(), index, "{word} is at {index}");
			assert_eq!(kind.word(), *word, "and is written as itself");
		}

		assert!(
			SkyKind::at(u32::try_from(SkyKind::WORDS.len()).expect("two")).is_none(),
			"one past the end is no kind"
		);
	}

	#[test]
	fn nothing_is_what_a_world_starts_with_and_the_colors_survive_it() {
		assert!(!Sky::NONE.is_drawn(), "a fresh world has the clear color behind it");
		assert_eq!(Sky::default(), Sky::NONE, "and that is the default");
		assert!(Sky::day().is_drawn(), "the same three colors, turned on");
		assert_eq!(
			(Sky::day().zenith, Sky::day().horizon, Sky::day().ground),
			(Sky::NONE.zenith, Sky::NONE.horizon, Sky::NONE.ground),
			"which is what makes turning the word on show a sky rather than a black dome"
		);
	}

	#[test]
	fn every_field_reads_back_what_it_was_written() {
		let mut sky = Sky::NONE;

		for entry in Sky::FIELDS {
			let written = match entry.kind {
				| Kind::Word(words) =>
					Value::Word(u32::try_from(words.len()).expect("a short list") - 1),
				| Kind::Color => Value::Color(Vec3::new(0.25, 0.5, 0.75)),
				| kind => panic!("a sky has no field of {kind:?}"),
			};

			assert!(entry.set(&mut sky, written.clone()), "{} takes its own kind", entry.name);
			assert_eq!(entry.get(&sky), written, "{} hands back what it took", entry.name);
		}

		assert_eq!(sky.kind, SkyKind::Gradient, "the last word is the last kind");
	}
}
