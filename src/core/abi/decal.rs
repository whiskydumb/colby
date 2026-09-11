//! What an entity paints onto whatever is around it, when it paints anything.
//!
//! A decal is a box, and every surface inside the box has a picture laid over
//! it. The picture is thrown down the box's own -z, the way a cone light
//! points, and it lies in the box's xy plane: its right edge along +x and its
//! top along +y. A bullet hole on a wall is a thin box facing the wall; a
//! puddle is a flat one turned to look down at the floor.
//!
//! **A decal has no size and no picture of its own.** The box is the entity's
//! own transform, where it stands, how it is turned and how big it is, so one
//! unit of scale is one unit of box: the same unit cube a mesh called `cube`
//! fills. The picture, its color, its normal map and how rough and how metal
//! it makes a surface are the entity's own material, the record a mesh is
//! drawn with. So a decal hangs off a parent, moves and grows under the gizmo,
//! is written down by a scene and comes back from an undo without any of those
//! knowing what a decal is, which is the argument a light makes for living on
//! an entity, made a second time.
//!
//! **What is left is three numbers**: which kind of decal it is, how much it
//! fades on a surface turned away from it, and which of two decals that
//! overlap is painted on top. @ref `colby_engine::decal` for what a frame does
//! with them.

use super::field::{Field, Kind, Value, field, word};

/// What shape a decal paints into.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DecalKind {
	/// Nothing: the entity paints nothing at all.
	///
	/// A variant rather than an `Option`, for the light's reason: every slot
	/// of the table holds one of these, and a record that is written down
	/// needs a spelling for "no decal" as much as for the other one.
	#[default]
	None,

	/// A box, painting what is inside it with a picture thrown down its -z.
	Box,
}

impl DecalKind {
	/// The word each kind is written as, in declaration order.
	///
	/// A file's vocabulary and an inspector's drop-down, @ref
	/// [`field::Kind::Word`](super::field::Kind::Word). A third word may be
	/// appended without moving any format: what is stored is the place in this
	/// list.
	pub const WORDS: &[&str] = &["none", "box"];

	/// The kind at a place in [`WORDS`](Self::WORDS), if there is one.
	///
	/// @param index - the place
	#[must_use]
	pub const fn at(index: u32) -> Option<Self> {
		match index {
			| 0 => Some(Self::None),
			| 1 => Some(Self::Box),
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

	/// Whether a decal of this kind paints anything.
	#[must_use]
	pub const fn paints(self) -> bool { !matches!(self, Self::None) }
}

/// The most [`Decal::fade`] is read as.
///
/// Just short of one, because the fade is a smooth step from this number up
/// to one and a step whose two ends meet divides by nothing. A decal asked to
/// fade by exactly one would fade everything it paints, a face turned squarely
/// towards it included, which is not a decal anybody asked for either.
pub const MAX_FADE: f32 = 0.999;

/// How much a decal fades on a turned surface until somebody says otherwise.
///
/// A half: a face turned squarely towards the decal is painted whole, one
/// turned edge on is not painted at all, and one turned sixty degrees away is
/// painted at half strength. Nought would paint the sides of everything the
/// box reaches with the picture smeared along them, which is the one thing
/// every projected decal is known for, and the far side of a thin wall with
/// the picture backwards.
pub const DEFAULT_FADE: f32 = 0.5;

/// One entity's decal: what kind, how it fades, and where it goes in a pile.
///
/// Plain data behind [`World`](super::World) like everything else an entity
/// carries. Not `#[repr(C)]` and not `Pod`: it never crosses as raw bytes, and
/// the file that writes it has a record of its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decal {
	/// What shape it paints into, or [`DecalKind::None`] for an entity that
	/// paints nothing.
	pub kind: DecalKind,

	/// How much it fades on a surface turned away from it, from nought to
	/// [`MAX_FADE`].
	///
	/// Nought paints every surface inside the box whichever way it faces.
	/// Above that, a surface fades by a smooth step over how squarely it faces
	/// back up the way the picture is thrown, and this number is where the step
	/// starts. A face turned away from the decal is never painted by one that
	/// fades at all.
	pub fade: f32,

	/// Which of two decals painting one surface is on top: the higher.
	///
	/// Two decals of one order go in the order of their entities' slots, which
	/// is stable while nothing is spawned or despawned. Not by distance, which
	/// is the other answer and the worse one: a pile ordered by how far each
	/// decal is from the eye turns itself over as somebody walks past it.
	pub order: i32,
}

impl Decal {
	/// A box, with every other number at what a decal starts with.
	pub const BOX: Self = Self { kind: DecalKind::Box, ..Self::NONE };
	/// Its fields, for an inspector, a reader and a writer. @ref
	/// [`field`](super::field).
	pub const FIELDS: &[Field<Self>] = &[
		word!(
			"kind",
			kind,
			DecalKind::WORDS,
			DecalKind::at,
			DecalKind::index,
			"what shape it paints into, or none"
		),
		field!(
			Float,
			"fade",
			fade,
			"how much it fades on a surface turned away from it; nought paints every face"
		),
		// by hand rather than through the macro, for the reason an emitter's cap
		// is: a `Value::Int` is sixty-four bits wide and this is thirty-two, so
		// the way back has to be written where it can refuse a number that will
		// not fit.
		Field {
			name: "order",
			help: "which of two overlapping decals is on top: the higher",
			kind: Kind::Int,
			get: |decal| Value::Int(i64::from(decal.order)),
			set: |decal, value| match value {
				| Value::Int(held) => match i32::try_from(held) {
					| Ok(order) => {
						decal.order = order;

						true
					},
					| Err(_) => false,
				},
				| _ => false,
			},
		},
	];
	/// No decal at all.
	///
	/// The numbers are still the ones a decal would start with, so that turning
	/// `kind` from `none` to `box` in an inspector paints something rather than
	/// handing back a decal that fades out on everything it touches.
	pub const NONE: Self = Self {
		kind: DecalKind::None,
		fade: DEFAULT_FADE,
		order: 0,
	};

	/// Whether this paints anything at all.
	#[must_use]
	pub const fn paints(self) -> bool { self.kind.paints() }

	/// The fade, held inside what the smooth step can take.
	///
	/// Asked by whoever is about to build the step out of it rather than
	/// enforced when it is written, the rule a light's cone keeps, so that a
	/// person dragging the number past its end in an inspector sees the decal
	/// stop changing rather than the number snap back under the pointer. A
	/// number that is not one reads as no fade at all.
	#[must_use]
	pub fn fading(self) -> f32 {
		if self.fade.is_nan() {
			return 0.0;
		}

		self.fade.clamp(0.0, MAX_FADE)
	}
}

impl Default for Decal {
	fn default() -> Self { Self::NONE }
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_kind_is_its_place_in_the_words() {
		for (index, word) in DecalKind::WORDS.iter().enumerate() {
			let index = u32::try_from(index).expect("two of them");
			let kind = DecalKind::at(index).expect("every place has a kind");

			assert_eq!(kind.index(), index, "{word} is at {index}");
			assert_eq!(kind.word(), *word, "and is written as itself");
		}

		assert!(
			DecalKind::at(u32::try_from(DecalKind::WORDS.len()).expect("two")).is_none(),
			"one past the end is no kind"
		);
	}

	#[test]
	fn nothing_is_the_kind_a_decal_starts_as_and_a_box_paints() {
		assert_eq!(DecalKind::default(), DecalKind::None, "an entity is not a decal");
		assert_eq!(Decal::default(), Decal::NONE, "and neither is the record it carries");
		assert!(!Decal::NONE.paints(), "which paints nothing");
		assert!(Decal::BOX.paints(), "where a box does");
		assert_eq!(
			(Decal::BOX.fade, Decal::BOX.order),
			(Decal::NONE.fade, Decal::NONE.order),
			"and differs from the nothing in its kind alone"
		);
	}

	#[test]
	fn the_fade_is_held_inside_what_a_smooth_step_can_take() {
		let fading = |fade: f32| Decal { fade, ..Decal::BOX }.fading();

		assert!((fading(0.25) - 0.25).abs() < 1.0e-6, "a fade inside the range is left alone");
		assert!(fading(-3.0).abs() < 1.0e-6, "below nought is no fade");
		assert!((fading(1.0) - MAX_FADE).abs() < 1.0e-6, "one is held just short of one");
		assert!((fading(7.0) - MAX_FADE).abs() < 1.0e-6, "and so is anything past it");
		assert!(fading(f32::NAN).abs() < 1.0e-6, "and a number that is not one fades nothing");
	}

	#[test]
	fn every_field_reads_back_what_it_was_written() {
		let mut decal = Decal::NONE;

		for entry in Decal::FIELDS {
			let written = match entry.kind {
				| Kind::Word(words) =>
					Value::Word(u32::try_from(words.len()).expect("a short list") - 1),
				| Kind::Float => Value::Float(0.25),
				| Kind::Int => Value::Int(-3),
				| kind => panic!("a decal has no field of {kind:?}"),
			};

			assert!(entry.set(&mut decal, written.clone()), "{} takes its own kind", entry.name);
			assert_eq!(entry.get(&decal), written, "{} hands back what it took", entry.name);
		}

		assert_eq!(decal.kind, DecalKind::Box, "the last word is the last kind");
		assert_eq!(decal.order, -3, "an order may be below nought");
	}

	#[test]
	fn an_order_that_will_not_fit_is_refused_and_writes_nothing() {
		let order = Decal::FIELDS
			.iter()
			.find(|entry| entry.name == "order")
			.expect("a decal has an order");
		let mut decal = Decal { order: 7, ..Decal::BOX };

		assert!(!order.set(&mut decal, Value::Int(i64::MAX)), "past thirty-two bits is refused");
		assert!(!order.set(&mut decal, Value::Float(1.0)), "and so is a number of another kind");
		assert_eq!(decal.order, 7, "and neither wrote anything");
	}
}
