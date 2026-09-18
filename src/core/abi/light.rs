//! What an entity shines, when it shines anything.
//!
//! One record with a kind rather than a type per kind. That is Wicked's shape,
//! one `LightComponent` with a `LightType`, and it is the one that survives
//! a new kind arriving: a word added to [`LightKind::WORDS`] costs a file
//! nothing, where a third struct would cost every reader, every writer and
//! every panel a third branch.
//!
//! **A light has no position and no direction of its own.** Both are the
//! entity's: where it stands is [`Entities::placed`](super::Entities::placed),
//! and a cone points down the entity's own -z, which is the direction the
//! camera and the listener already agree on. So a light hangs off a parent,
//! moves with a gizmo, is written down by a scene and comes back from an undo
//! without any of those knowing what a light is - the argument for putting it
//! on an entity at all.
//!
//! **The sun is not here.** It is [`Stage::light`](super::scene::Stage::light),
//! one direction for the whole world, and it is what the shadow cascades are
//! cut for. Unreal splits its classes on the same line: `AttenuationRadius`
//! lives on `ULocalLightComponent`, and `UDirectionalLightComponent` does not
//! derive from it. A world has one sun and any number of lamps.
//!
//! **The intensity is a multiplier, not a photometric unit.** Candelas and
//! lumens only mean anything under an exposure; Unreal's own component
//! constructor starts at `ELightUnits::Unitless` for the same kind of reason.
//! Where a file does say candela, one is 683 pi of them: the number a lamp read
//! from the exchange format is divided by on the way in, measured against a
//! path tracer's picture of the same lamp. @ref [`colby_engine`] for what the
//! shader does with it, and `colby_asset`'s glTF reader for the division.

use super::field::{Field, field, word};
use crate::glam::Vec3;

/// What shape a light throws.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LightKind {
	/// Nothing: the entity carries no light at all.
	///
	/// A variant rather than an `Option`, because every slot of the table
	/// holds one of these and a record that is written down needs a spelling
	/// for "no light" as much as for the other two.
	#[default]
	None,

	/// A point, throwing light in every direction.
	Point,

	/// A cone, throwing light down the entity's own -z.
	Spot,
}

impl LightKind {
	/// The word each kind is written as, in declaration order.
	///
	/// A file's vocabulary and an inspector's drop-down, @ref
	/// [`field::Kind::Word`](super::field::Kind::Word). A fourth word may be
	/// appended without moving any format: what is stored is the place in
	/// this list.
	pub const WORDS: &[&str] = &["none", "point", "spot"];

	/// The kind at a place in [`WORDS`](Self::WORDS), if there is one.
	///
	/// @param index - the place
	#[must_use]
	pub const fn at(index: u32) -> Option<Self> {
		match index {
			| 0 => Some(Self::None),
			| 1 => Some(Self::Point),
			| 2 => Some(Self::Spot),
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

	/// Whether a light of this kind throws anything.
	#[must_use]
	pub const fn is_lit(self) -> bool { !matches!(self, Self::None) }
}

/// The widest a cone may open, in radians.
///
/// Just under a hemisphere, and the number is Unreal's: `GetClampedConeAngles`
/// caps both angles at 88.9 degrees. The reason is the cone's own arithmetic -
/// the falloff is built from one over the difference of two cosines, and a
/// cone at ninety degrees has a cosine of zero at its edge and a denominator
/// that stops meaning anything just past it.
pub const MAX_CONE: f32 = 1.551_425_2;

/// The narrowest a cone's edge may be past its bright middle, in radians.
///
/// A cone whose two angles are equal divides by zero. Unreal pushes the outer
/// angle to the inner plus a thousandth of a radian; this is that number, and
/// it is applied by [`Light::cone`] rather than by whoever writes the fields.
pub const MIN_SPREAD: f32 = 0.001;

/// One entity's light: what shape, what color, how bright, how far.
///
/// Plain data behind [`World`](super::World) like everything else an entity
/// carries. Not `#[repr(C)]` and not `Pod`: it never crosses as raw bytes, and
/// the file that writes it has a record of its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
	/// What shape it throws, or [`LightKind::None`] for an entity that is not
	/// a light.
	pub kind: LightKind,

	/// Its color, linear RGB.
	pub color: Vec3,

	/// How bright, as a plain multiplier on the color.
	pub intensity: f32,

	/// How far its contribution reaches, in world units.
	///
	/// Not physically meaningful, because an inverse square never quite
	/// reaches zero, and it is what makes a light affordable: a fragment past
	/// this reads nothing, and a light whose sphere the camera cannot see is
	/// not sent to the shader at all. Unreal says the same thing about its
	/// `AttenuationRadius`, in the same words and for the same reason.
	pub range: f32,

	/// The half-angle of a cone's bright middle, in radians.
	///
	/// Everything inside this gets the whole of the light. Ignored by a point.
	pub inner: f32,

	/// The half-angle of a cone's edge, in radians.
	///
	/// Between [`inner`](Self::inner) and this the light falls off to nothing.
	/// Ignored by a point.
	pub outer: f32,

	/// Whether what it lights throws a shadow.
	///
	/// **On, and the field it was decided against is split exactly three to
	/// three.** Three engines have it on and three off, and the reason the
	/// three have it off is the same in all three: the number of lights is
	/// unbounded, so a scene with fifty lamps would want fifty shadow maps and
	/// the cost has to be the user's to accept. colby's is bounded by
	/// construction - sixteen tiles of an atlas, and lamps past them throw
	/// nothing - so the case they are guarding against cannot arise here.
	///
	/// What settles it beyond that is the engine's own consistency: the sun
	/// casts by default, and a lamp that does not while the sun does is a lamp
	/// whose light goes through walls in a picture nobody asked a question
	/// about. Every other switch of this family is a *look* a world opts into
	/// with a number; this one is whether the picture is right.
	pub shadow: bool,
}

impl Light {
	/// Its fields, for an inspector, a reader and a writer. @ref
	/// [`field`](super::field).
	///
	/// The two angles are in radians here, as they are everywhere else in the
	/// world; a panel that wants degrees converts on the way to the eye.
	pub const FIELDS: &[Field<Self>] = &[
		word!(
			"kind",
			kind,
			LightKind::WORDS,
			LightKind::at,
			LightKind::index,
			"what shape of light it throws, or none"
		),
		field!(Color, "color", color, "the color it throws"),
		field!(Float, "intensity", intensity, "how bright, as a multiplier on the color"),
		field!(Float, "range", range, "how far its contribution reaches"),
		field!(Float, "inner", inner, "the half-angle of a cone's bright middle, in radians"),
		field!(Float, "outer", outer, "the half-angle of a cone's edge, in radians"),
		field!(Bool, "shadow", shadow, "whether what it lights throws a shadow"),
	];
	/// No light at all.
	///
	/// The numbers are still the ones a light would start with, so that
	/// turning `kind` from `none` to `point` in an inspector lights something
	/// rather than handing back a black lamp of zero reach.
	pub const NONE: Self = Self {
		kind: LightKind::None,
		color: Vec3::ONE,
		intensity: 1.0,
		range: 10.0,
		// zero and a quarter turn: a cone with no bright middle, falling off
		// from its axis to its edge. Unreal starts a spot at exactly this -
		// `InnerConeAngle = 0`, `OuterConeAngle = 44` degrees - and the reason
		// is that a cone with a hard middle reads as a spotlight with a disc
		// stamped in it.
		inner: 0.0,
		outer: std::f32::consts::FRAC_PI_4,
		shadow: true,
	};

	/// A point of a color, reaching so far.
	///
	/// It throws a shadow, which is what [`NONE`](Self::NONE) says.
	///
	/// @param color - linear RGB
	/// @param intensity - the multiplier on it
	/// @param range - how far it reaches
	#[must_use]
	pub const fn point(color: Vec3, intensity: f32, range: f32) -> Self {
		Self {
			kind: LightKind::Point,
			color,
			intensity,
			range,
			..Self::NONE
		}
	}

	/// A cone of a color, reaching so far and opening so wide.
	///
	/// It throws a shadow, which is what [`NONE`](Self::NONE) says.
	///
	/// @param color - linear RGB
	/// @param intensity - the multiplier on it
	/// @param range - how far it reaches
	/// @param inner - the half-angle of the bright middle, in radians
	/// @param outer - the half-angle of the edge, in radians
	#[must_use]
	pub const fn spot(color: Vec3, intensity: f32, range: f32, inner: f32, outer: f32) -> Self {
		Self {
			kind: LightKind::Spot,
			color,
			intensity,
			range,
			inner,
			outer,
			shadow: Self::NONE.shadow,
		}
	}

	/// Whether this throws anything at all.
	///
	/// A kind of `none`, a range of nothing or a black lamp all answer `false`,
	/// and each of the three is a light a frame may leave out entirely.
	#[must_use]
	pub fn is_lit(self) -> bool {
		self.kind.is_lit()
			&& self.range > 0.0
			&& self.intensity > 0.0
			&& self.color.max_element() > 0.0
	}

	/// The two cone angles, put in order and held inside what a cone may be.
	///
	/// Asked by whoever is about to build a falloff out of them rather than
	/// enforced when they are written, so that a person dragging one angle
	/// past the other in an inspector sees the cone close rather than the
	/// number snap out from under the pointer.
	///
	/// @return `(inner, outer)`, with `outer` at least [`MIN_SPREAD`] past
	/// `inner` and neither past [`MAX_CONE`]
	#[must_use]
	pub fn cone(self) -> (f32, f32) {
		let inner = self.inner.clamp(0.0, MAX_CONE);
		let outer = self
			.outer
			.clamp(inner + MIN_SPREAD, MAX_CONE + MIN_SPREAD);

		(inner, outer)
	}
}

impl Default for Light {
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
		for (index, word) in LightKind::WORDS.iter().enumerate() {
			let index = u32::try_from(index).expect("three of them");
			let kind = LightKind::at(index).expect("every place has a kind");

			assert_eq!(kind.index(), index, "{word} is at {index}");
			assert_eq!(kind.word(), *word, "and is written as itself");
		}

		assert!(
			LightKind::at(u32::try_from(LightKind::WORDS.len()).expect("three")).is_none(),
			"one past the end is no kind"
		);
	}

	#[test]
	fn nothing_is_the_kind_a_light_starts_as() {
		assert_eq!(LightKind::default(), LightKind::None, "an entity is not a lamp");
		assert!(!Light::NONE.is_lit(), "and neither is the record it carries");
		assert!(!LightKind::None.is_lit(), "nor the kind by itself");
		assert!(LightKind::Point.is_lit() && LightKind::Spot.is_lit(), "the other two do");
	}

	#[test]
	fn a_light_with_nothing_to_give_is_not_lit() {
		let lamp = Light::point(Vec3::ONE, 1.0, 10.0);

		assert!(lamp.is_lit(), "a plain lamp throws something");
		assert!(!Light { range: 0.0, ..lamp }.is_lit(), "one with no reach does not");
		assert!(!Light { intensity: 0.0, ..lamp }.is_lit(), "nor one turned all the way down");
		assert!(!Light { color: Vec3::ZERO, ..lamp }.is_lit(), "nor a black one");
		assert!(
			Light { color: Vec3::new(0.0, 0.0, 0.3), ..lamp }.is_lit(),
			"but one channel is enough"
		);
	}

	#[test]
	fn a_cone_is_put_in_order_when_it_is_asked_for() {
		let (inner, outer) = Light::spot(Vec3::ONE, 1.0, 5.0, 0.9, 0.2).cone();

		assert!((inner - 0.9).abs() < 1.0e-6, "the middle is left where it was written");
		assert!(
			(outer - (0.9 + MIN_SPREAD)).abs() < 1.0e-6,
			"and the edge is pushed just past it rather than left inside"
		);

		let (inner, outer) = Light::spot(Vec3::ONE, 1.0, 5.0, 3.0, 3.0).cone();

		assert!((inner - MAX_CONE).abs() < 1.0e-6, "a middle past a hemisphere is held back");
		assert!((outer - (MAX_CONE + MIN_SPREAD)).abs() < 1.0e-6, "and so is the edge");
	}

	#[test]
	fn every_field_reads_back_what_it_was_written() {
		let mut lamp = Light::NONE;

		for entry in Light::FIELDS {
			let written = match entry.kind {
				| Kind::Word(words) =>
					Value::Word(u32::try_from(words.len()).expect("a short list") - 1),
				| Kind::Color => Value::Color(Vec3::new(0.25, 0.5, 0.75)),
				| Kind::Float => Value::Float(1.5),
				| Kind::Bool => Value::Bool(false),
				| kind => panic!("a light has no field of {kind:?}"),
			};

			assert!(entry.set(&mut lamp, written.clone()), "{} takes its own kind", entry.name);
			assert_eq!(entry.get(&lamp), written, "{} hands back what it took", entry.name);
		}

		assert_eq!(lamp.kind, LightKind::Spot, "the last word is the last kind");
	}
}
