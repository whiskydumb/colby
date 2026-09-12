//! What happens to a picture between the last surface drawn and the screen.
//!
//! **One flat record, and no override bits.** Unreal carries 206 of them in
//! `FPostProcessSettings`, and they exist for exactly one thing: blending
//! volumes, where `LERP_PP(NAME)` is `if (Src.bOverride_NAME) Dest.NAME =
//! Lerp(Dest.NAME, Src.NAME, Weight)`. A world with no volumes in it has
//! nothing to blend, so a bit per field would be a bit nobody reads. Godot has
//! none at all - its `Environment` is swapped whole. The bits arrive with
//! volumes and not before.
//!
//! **The tonemap is what makes a light brighter than white mean anything.**
//! Before it, the target was eight bits per channel and everything past one
//! was the same white; the frame is now drawn into sixteen-bit floats and
//! squeezed down at the end, so a lamp of intensity forty is a highlight
//! rather than a hole. That is also the thing photometric units were waiting
//! for. @ref [`light`](super::light).
//!
//! **The eye adapts, and it opens already adapted.** When there is no history,
//! which is the first frame a surface ever draws, the adapted luminance is set
//! to the measured one rather than moved towards it. Without that a picture
//! taken by a process that renders exactly one frame, which is what `--shot`
//! is, would be a picture of a half-open eye, and it would disagree with the
//! same scene in a window that had been up for a second.

use super::field::{Field, field, word};
use crate::glam::Vec3;

/// How the numbers a renderer works in are squeezed into the ones a screen
/// shows.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ToneMap {
	/// Nothing but a clamp.
	///
	/// What the renderer did before there was a curve at all, and what Godot
	/// calls `LINEAR`. Exposure still applies.
	None,

	/// `c / (1 + c)`, with a white point that says which value reaches one.
	///
	/// The cheapest curve that is a curve. It desaturates nothing and shifts
	/// no hues, which reads as flat beside the one below.
	Reinhard,

	/// The fitted ACES curve.
	///
	/// Narkowicz's six-line approximation of the filmic response the film
	/// industry's transform has: contrast in the middle, a long shoulder, and
	/// brights that desaturate towards white the way an exposed negative does.
	/// Not neutral - it has a look, and the look is the point.
	#[default]
	Aces,
}

impl ToneMap {
	/// The word each curve is written as, in declaration order.
	///
	/// The two the field ships that are not here both want a three-dimensional
	/// lookup texture - AgX and Tony McMapface - which is an asset and a
	/// texture kind this engine has no spelling for. They are the fourth and
	/// fifth words when it does.
	pub const WORDS: &[&str] = &["none", "reinhard", "aces"];

	/// The curve at a place in [`WORDS`](Self::WORDS), if there is one.
	///
	/// @param index - the place
	#[must_use]
	pub const fn at(index: u32) -> Option<Self> {
		match index {
			| 0 => Some(Self::None),
			| 1 => Some(Self::Reinhard),
			| 2 => Some(Self::Aces),
			| _ => None,
		}
	}

	/// Where this curve is in [`WORDS`](Self::WORDS).
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "the discriminant is the place in the list, by declaration order"
	)]
	pub const fn index(self) -> u32 { self as u32 }

	/// The word this curve is written as.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "u32 to usize is lossless on every target this builds for, and try_from is not \
		          available in a const fn"
	)]
	pub const fn word(self) -> &'static str { Self::WORDS[self.index() as usize] }
}

/// The brightness the eye aims to see a picture's average at.
///
/// The middle grey a photographer meters for, and it is a constant rather than
/// a field because it is the definition of "correctly exposed" rather than a
/// preference: what a person wants to move is
/// [`Post::exposure_bias`](Post::exposure_bias), in stops, on top of it.
pub const MIDDLE_GREY: f32 = 0.18;

/// The dimmest average a measurement is believed at.
///
/// A frame of pure black would ask for an infinite exposure, and the clamp on
/// the result would catch it - but the division would go through an infinity
/// first, and an infinity that reaches a uniform is a picture of nothing.
pub const DARKEST: f32 = 1.0e-4;

/// Everything that happens to a picture after the world is drawn into it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Post {
	/// Which curve squeezes the picture onto the screen.
	pub tonemap: ToneMap,

	/// The value [`Reinhard`](ToneMap::Reinhard) maps to one.
	///
	/// Anything at or above this is white. Godot calls it `tonemap_white` and
	/// starts it at one; four is a more useful place to start, because a scene
	/// whose brightest highlight is exactly the white point has no highlight.
	/// Read by nothing else.
	pub white: f32,

	/// Whether the exposure is measured from the picture rather than set.
	pub auto_exposure: bool,

	/// The multiplier the picture is scaled by before the curve, when the
	/// exposure is not measured.
	pub exposure: f32,

	/// Stops added to a measured exposure.
	///
	/// What a person turns when a scene is correctly exposed and still not
	/// what they meant. Unreal calls the same field `AutoExposureBias`.
	/// Ignored when [`auto_exposure`](Self::auto_exposure) is off.
	pub exposure_bias: f32,

	/// The smallest a measured exposure may be.
	pub exposure_min: f32,

	/// The largest it may be.
	///
	/// The pair is what stops a dark room being lifted to daylight and a
	/// bright one being crushed to nothing; both s&box and Unreal carry the
	/// same two numbers.
	pub exposure_max: f32,

	/// How fast the eye adapts, in units a second.
	///
	/// The blend towards a new measurement is `1 - exp(-dt * rate)`, so this
	/// is a rate rather than a fraction and a frame twice as long adapts by
	/// the right amount rather than by twice as much. Zero holds the eye
	/// wherever it is.
	pub exposure_rate: f32,

	/// How much of the bright pass is added back over the picture.
	///
	/// Zero is no bloom at all, and no work either: the chain is skipped
	/// rather than run and multiplied by nothing.
	pub bloom: f32,

	/// How bright a pixel has to be to bloom.
	pub bloom_threshold: f32,

	/// The color a distant surface fades towards, linear RGB.
	pub fog: Vec3,

	/// How quickly it fades, per unit of distance.
	///
	/// Zero is no fog. The falloff is `exp(-(d * density)^2)`, which leaves
	/// what is near untouched and closes over the far distance rather than
	/// greying everything evenly.
	pub fog_density: f32,

	/// How much light the air catches around the sun.
	///
	/// Zero is none, and no work either: the passes are skipped rather than
	/// run and multiplied by nothing, the way [`bloom`](Self::bloom) is. The
	/// light it smears is whatever the picture holds past the middle of the
	/// view - the sky, in a world that has one - so it needs no color of its
	/// own.
	///
	/// **One number, and the rest are constants.** The shape of the smear has
	/// four more: how far along the way to the sun it reaches, how much each
	/// tap adds, how fast a tap fades with distance, and how wide of the sun
	/// anything happens at all. Every one of them is a look rather than a
	/// choice a world makes, and the field ships them fixed; the one engine
	/// here that has this effect and a knob for it exposes exactly this
	/// number.
	///
	/// Off in a fresh world, which is where both engines that have it start.
	/// @ref [`colby_engine`] for the passes.
	pub shafts: f32,
}

impl Post {
	/// What a world starts with: a filmic curve, an eye that adapts, and
	/// neither bloom nor fog.
	///
	/// **The curve is on and the other two are off**, which is not an
	/// inconsistency. A tonemap is a property of the renderer - a frame drawn
	/// into sixteen-bit floats and handed to a screen unsqueezed is strictly
	/// worse than the eight-bit frame that came before it - while bloom and
	/// fog are things a particular world wants.
	pub const DEFAULT: Self = Self {
		tonemap: ToneMap::Aces,
		white: 4.0,
		auto_exposure: true,
		exposure: 1.0,
		exposure_bias: 0.0,
		exposure_min: 0.05,
		exposure_max: 8.0,
		exposure_rate: 2.0,
		bloom: 0.0,
		bloom_threshold: 1.0,
		fog: Vec3::new(0.55, 0.60, 0.68),
		fog_density: 0.0,
		shafts: 0.0,
	};
	/// Its fields, for an inspector, a reader and a writer. @ref
	/// [`field`](super::field).
	pub const FIELDS: &[Field<Self>] = &[
		word!(
			"tonemap",
			tonemap,
			ToneMap::WORDS,
			ToneMap::at,
			ToneMap::index,
			"which curve squeezes the picture onto the screen"
		),
		field!(Float, "white", white, "the value reinhard maps to white"),
		field!(Bool, "auto_exposure", auto_exposure, "measure the exposure from the picture"),
		field!(Float, "exposure", exposure, "the multiplier when the exposure is not measured"),
		field!(Float, "exposure_bias", exposure_bias, "stops added to a measured exposure"),
		field!(Float, "exposure_min", exposure_min, "the smallest a measured exposure may be"),
		field!(Float, "exposure_max", exposure_max, "the largest it may be"),
		field!(Float, "exposure_rate", exposure_rate, "how fast the eye adapts, per second"),
		field!(Float, "bloom", bloom, "how much of the bright pass is added back"),
		field!(
			Float,
			"bloom_threshold",
			bloom_threshold,
			"how bright a pixel has to be to bloom"
		),
		field!(Color, "fog", fog, "the color a distant surface fades towards"),
		field!(Float, "fog_density", fog_density, "how quickly it fades, per unit of distance"),
		field!(Float, "shafts", shafts, "how much light the air catches around the sun"),
	];

	/// The exposure a measured average asks for, before the clamp.
	///
	/// @param average - the picture's average luminance
	/// @return the multiplier to scale the picture by
	#[must_use]
	pub fn metered(self, average: f32) -> f32 {
		let asked = MIDDLE_GREY / average.max(DARKEST) * self.exposure_bias.exp2();

		asked.clamp(self.exposure_min.min(self.exposure_max), self.exposure_max)
	}

	/// How far towards a new measurement one frame moves the eye.
	///
	/// `1 - exp(-dt * rate)`, which is the shape that makes a frame twice as
	/// long adapt by the right amount rather than by twice as much. A rate of
	/// nothing holds the eye still; a frame of no length moves it nowhere.
	///
	/// @param seconds - how long the frame was
	/// @return a fraction in `0.0 ..= 1.0`
	#[must_use]
	pub fn adapt(self, seconds: f32) -> f32 {
		if self.exposure_rate <= 0.0 || seconds <= 0.0 {
			return 0.0;
		}

		1.0 - (-seconds * self.exposure_rate).exp()
	}

	/// Whether anything is added back over the picture.
	#[must_use]
	pub fn is_blooming(self) -> bool { self.bloom > 0.0 }

	/// Whether distance takes anything away from a surface.
	#[must_use]
	pub fn is_foggy(self) -> bool { self.fog_density > 0.0 }

	/// Whether the air catches anything around the sun.
	#[must_use]
	pub fn is_shafting(self) -> bool { self.shafts > 0.0 }
}

impl Default for Post {
	fn default() -> Self { Self::DEFAULT }
}

#[cfg(test)]
mod tests {
	use super::{
		super::field::{Kind, Value},
		*,
	};

	#[test]
	fn a_curve_is_its_place_in_the_words() {
		for (index, word) in ToneMap::WORDS.iter().enumerate() {
			let index = u32::try_from(index).expect("three of them");
			let curve = ToneMap::at(index).expect("every place has a curve");

			assert_eq!(curve.index(), index, "{word} is at {index}");
			assert_eq!(curve.word(), *word, "and is written as itself");
		}

		assert!(
			ToneMap::at(u32::try_from(ToneMap::WORDS.len()).expect("three")).is_none(),
			"one past the end is no curve"
		);
		assert_eq!(ToneMap::default(), ToneMap::Aces, "and a world starts filmic");
	}

	#[test]
	fn a_world_starts_with_a_curve_and_without_bloom_fog_or_shafts() {
		assert!(Post::DEFAULT.auto_exposure, "the eye adapts");
		assert!(!Post::DEFAULT.is_blooming(), "nothing is added back");
		assert!(!Post::DEFAULT.is_foggy(), "distance takes nothing away");
		assert!(!Post::DEFAULT.is_shafting(), "and the air catches nothing");
	}

	#[test]
	fn the_air_catches_light_only_at_a_strength_above_nothing() {
		// what decides whether three passes are recorded at all, so a number
		// somebody typed backwards has to read as off rather than as a smear
		// multiplied by a negative
		for asked in [0.0, -0.0, -1.0, f32::NEG_INFINITY] {
			assert!(
				!Post { shafts: asked, ..Post::DEFAULT }.is_shafting(),
				"a strength of {asked} is no shafts"
			);
		}

		assert!(
			Post { shafts: 1.0e-6, ..Post::DEFAULT }.is_shafting(),
			"and anything above nothing is some"
		);
	}

	#[test]
	fn a_meter_asks_for_the_exposure_that_puts_the_average_on_middle_grey() {
		let post = Post {
			exposure_min: 0.0,
			exposure_max: 1000.0,
			..Post::DEFAULT
		};

		for average in [0.02, 0.18, 0.5, 4.0_f32] {
			let asked = post.metered(average);

			assert!(
				asked.mul_add(average, -MIDDLE_GREY).abs() < 1.0e-4,
				"an average of {average} scaled by {asked} lands on middle grey"
			);
		}
	}

	#[test]
	fn a_meter_is_held_between_the_two_numbers_and_moved_by_the_bias() {
		let post = Post::DEFAULT;

		assert!(
			(post.metered(0.0) - post.exposure_max).abs() < 1.0e-6,
			"a black picture asks for everything and gets the ceiling rather than an infinity"
		);
		assert!(
			(post.metered(1.0e9) - post.exposure_min).abs() < 1.0e-6,
			"and a blinding one gets the floor"
		);

		let lifted = Post { exposure_bias: 1.0, ..post };

		assert!(
			2.0_f32
				.mul_add(-post.metered(0.18), lifted.metered(0.18))
				.abs() < 1.0e-5,
			"a stop of bias is a doubling"
		);

		let crossed = Post {
			exposure_min: 9.0,
			exposure_max: 1.0,
			..post
		};

		assert!(
			(crossed.metered(0.18) - 1.0).abs() < 1.0e-6,
			"two numbers written the wrong way round clamp to the ceiling rather than panicking"
		);
	}

	#[test]
	fn an_eye_adapts_by_a_rate_rather_than_by_a_fraction_of_a_frame() {
		let post = Post::DEFAULT;
		let once = post.adapt(0.1);
		let twice = post.adapt(0.2);

		assert!(once > 0.0 && once < 1.0, "a tenth of a second moves it partway: {once}");
		assert!(twice > once, "and twice as long moves it further");
		assert!(
			twice < once * 2.0,
			"but by less than twice as much, which is what makes it a rate: {twice} vs {once}"
		);
		assert!((post.adapt(100.0) - 1.0).abs() < 1.0e-6, "a frame that took forever arrives");

		assert!((post.adapt(0.0) - 0.0).abs() < 1.0e-9, "a frame of no length moves nothing");
		assert!(
			(Post { exposure_rate: 0.0, ..post }.adapt(1.0) - 0.0).abs() < 1.0e-9,
			"and a rate of nothing holds the eye still"
		);
	}

	#[test]
	fn every_field_reads_back_what_it_was_written() {
		let mut post = Post::DEFAULT;

		for entry in Post::FIELDS {
			let written = match entry.kind {
				| Kind::Word(words) =>
					Value::Word(u32::try_from(words.len()).expect("a short list") - 1),
				| Kind::Color => Value::Color(Vec3::new(0.25, 0.5, 0.75)),
				| Kind::Float => Value::Float(1.5),
				| Kind::Bool => Value::Bool(false),
				| kind => panic!("a post record has no field of {kind:?}"),
			};

			assert!(entry.set(&mut post, written.clone()), "{} takes its own kind", entry.name);
			assert_eq!(entry.get(&post), written, "{} hands back what it took", entry.name);
		}
	}
}
