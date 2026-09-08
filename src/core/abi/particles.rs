//! What an entity throws off, and the live cloud it has thrown.
//!
//! Two things, and they are deliberately not one. [`Emitter`] is a
//! *description*: one record with a kind, exactly as [`Light`](super::light)
//! and [`Water`](super::water) are, sitting in a per-slot array on
//! [`Entities`](super::Entities) so that a plume hangs off a parent, moves
//! with a gizmo, is written down by a scene and comes back from an undo
//! without any of those knowing what a particle is. [`Sparks`] is the *cloud*:
//! one flat pool for the whole world, written by the step and read by the
//! renderer, and it is not a description of anything.
//!
//! **A particle is not an entity, and never can be here.**
//! [`MAX_ENTITIES`](super::MAX_ENTITIES) is a thousand and twenty-four and the
//! renderer spends a hundred and twenty-eight bytes of instance on each of
//! them; one plume of smoke would be the whole world's budget. So none of the
//! entity table, the sort key or the batching in `colby_engine::Scene` is
//! involved at all - the pool is its own buffer and its own pipeline, which is
//! the arrangement the debug renderer already has.
//!
//! **The cloud is not written down.** A saved scene carries the emitters and
//! not what they had thrown when somebody pressed the button, which is what
//! four of the five engines read for this do; a fire re-lights itself in a
//! second and a file that remembered its smoke would be describing one moment
//! of a thing whose whole point is that it moves.
//!
//! **The simulation is not here.** It is `colby_runtime::sparks`, run inside
//! the fixed step beside the solver, and the boundary carries the description
//! and not the behavior - the line [`Water`](super::water) draws in the same
//! words. What is here is the two records and the arithmetic that reads them,
//! which is what a test can hold still.

use super::{
	EntityId, TextureId,
	field::{Field, Kind, Value, field, word},
};
use crate::glam::Vec3;

/// How many particles the whole world may have alive at once.
///
/// Bounded rather than fixed, and bounded for
/// [`MAX_ENTITIES`](super::MAX_ENTITIES)'s reason: an emitter's rate is a
/// number somebody typed, and a world that ran out of memory because of a
/// typed number would be a worse failure than one that stops throwing. Four
/// thousand and ninety-six is four times the entity table and about a
/// hundred and thirty kilobytes of pool.
///
/// The field's own numbers are the same size: Wicked starts an emitter at a
/// thousand (`wiEmittedParticle.h`, `MAX_PARTICLES = 1000`) and s&box at a
/// thousand (`ParticleEffect.cs`, `MaxParticles = 1000`), both **per
/// emitter**; this is the ceiling over all of them together.
pub const MAX_SPARKS: usize = 4096;

/// A count as a `u32`, saturating rather than wrapping.
///
/// One line, and it is here so that [`MAX_SPARKS`] can be compared against a
/// field without an `as` anywhere: this workspace does not allow one, and a
/// ceiling that silently became nought on a machine with a narrower `usize`
/// would be the worst possible way to find that out.
///
/// @param count - how many
/// @return the same, or `u32::MAX` if it does not fit
fn whole(count: usize) -> u32 {
	match u32::try_from(count) {
		| Ok(whole) => whole,
		| Err(_) => u32::MAX,
	}
}

/// What shape an emitter throws particles into.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum EmitterKind {
	/// Nothing: the entity throws no particles at all.
	///
	/// A variant rather than an `Option`, for [`LightKind::None`]'s reason:
	/// every slot of the table holds one of these, and a record that is
	/// written down needs a spelling for "no emitter" as much as for the
	/// other two.
	///
	/// [`LightKind::None`]: super::LightKind::None
	#[default]
	None,

	/// A point, throwing in every direction.
	Point,

	/// A cone, throwing down the entity's own -z.
	///
	/// The same forward a spot light, a camera and a listener already agree
	/// on, so an emitter aimed with the gizmo aims the way the arrow points.
	Cone,
}

impl EmitterKind {
	/// The word each kind is written as, in declaration order.
	///
	/// A file's vocabulary and an inspector's drop-down. A fourth word - `box`
	/// and `sphere` are the two the field has that this does not - may be
	/// appended without moving any format: what is stored is the place in this
	/// list. @ref [`field::Kind::Word`](super::field::Kind::Word).
	pub const WORDS: &[&str] = &["none", "point", "cone"];

	/// The kind at a place in [`WORDS`](Self::WORDS), if there is one.
	///
	/// @param index - the place
	#[must_use]
	pub const fn at(index: u32) -> Option<Self> {
		match index {
			| 0 => Some(Self::None),
			| 1 => Some(Self::Point),
			| 2 => Some(Self::Cone),
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

	/// Whether this kind throws anything at all.
	#[must_use]
	pub const fn throws(self) -> bool { !matches!(self, Self::None) }
}

/// How a particle's color reaches the picture.
///
/// **Two, and this is not [`Blend`](super::material::Blend).** A surface's
/// modes answer "how is this material's alpha read", and the answer for every
/// one of them is a pipeline in the scene's own six-row table. A particle is
/// unlit geometry in a buffer of its own with a pipeline of its own, and the
/// question it asks is a different one: whether the cloud lightens what is
/// behind it or covers it.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SparkBlend {
	/// The particle's color is added to what is behind it.
	///
	/// **The default, and it is the answer to sorting rather than a taste.**
	/// Addition is commutative, so a cloud drawn in any order is the same
	/// cloud - which is why fire, sparks and muzzle flashes are additive in
	/// every engine that has them, and why not sorting particles costs
	/// nothing at all for the effects people actually build. @ref [`Sparks`]
	/// for the sorting this replaces.
	#[default]
	Additive,

	/// The particle's alpha says how much of what is behind it still shows.
	///
	/// What smoke and dust want. Order matters here and this build does not
	/// sort within a cloud, so a dense alpha plume seen edge-on can show its
	/// own particles in the wrong order. That is the same limitation Niagara,
	/// Godot, Wicked and s&box all ship *by default*, and the same one their
	/// users answer with an author's switch rather than with a better sort.
	Alpha,
}

impl SparkBlend {
	/// The word each one is written as, in declaration order.
	pub const WORDS: &[&str] = &["additive", "alpha"];

	/// The one at a place in [`WORDS`](Self::WORDS), if there is one.
	///
	/// @param index - the place
	#[must_use]
	pub const fn at(index: u32) -> Option<Self> {
		match index {
			| 0 => Some(Self::Additive),
			| 1 => Some(Self::Alpha),
			| _ => None,
		}
	}

	/// Where this one is in [`WORDS`](Self::WORDS).
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "the discriminant is the place in the list, by declaration order"
	)]
	pub const fn index(self) -> u32 { self as u32 }

	/// The word this one is written as.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "u32 to usize is lossless on every target this builds for, and try_from is not \
		          available in a const fn"
	)]
	pub const fn word(self) -> &'static str { Self::WORDS[self.index() as usize] }

	/// The row of the renderer's pipeline table this one is built into.
	///
	/// A match rather than a cast, so a third mode is a compile error here
	/// rather than a row nobody built - the same guard
	/// [`Blend::row`](super::material::Blend::row) has.
	#[must_use]
	pub const fn row(self) -> usize {
		match self {
			| Self::Additive => 0,
			| Self::Alpha => 1,
		}
	}
}

/// One entity's emitter: what it throws, how fast, and what becomes of it.
///
/// Plain data behind [`World`](super::World) like everything else an entity
/// carries. Not `#[repr(C)]` and not `Pod`: it never crosses as raw bytes, and
/// the file that writes it has a record of its own.
///
/// **An emitter has no position and no direction of its own**, exactly as a
/// light has none: where it stands is
/// [`Entities::placed`](super::Entities::placed), and a cone throws down the
/// entity's own -z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emitter {
	/// What shape it throws into, or [`EmitterKind::None`] for an entity that
	/// throws nothing.
	pub kind: EmitterKind,

	/// How the cloud reaches the picture.
	pub blend: SparkBlend,

	/// The picture one particle is drawn with, or
	/// [`TextureId::NONE`](super::TextureId::NONE) for a plain square.
	///
	/// A texture rather than a material, and that is a decision: a
	/// [`Material`](super::Material) carries metallic, roughness, a normal map
	/// and a blend mode, and an unlit billboard reads not one of them. Fyrox's
	/// own particle shader samples a single channel of its picture; this reads
	/// all four and multiplies them in.
	pub texture: TextureId,

	/// How many particles a second this throws while it is running.
	pub rate: f32,

	/// The most this one emitter may have alive at once.
	///
	/// Its own ceiling under [`MAX_SPARKS`], so that one runaway emitter
	/// starves itself rather than every other emitter in the world.
	pub cap: u32,

	/// How long a particle lives, in seconds.
	pub life: f32,

	/// How much of [`life`](Self::life) is thrown away at random, from nought
	/// to one.
	///
	/// At nought every particle lives exactly as long, which reads as a
	/// pulsing ring rather than a plume; at one a particle lives anywhere from
	/// nothing to its whole life.
	pub life_spread: f32,

	/// How fast a particle leaves, in units a second.
	pub speed: f32,

	/// How much of [`speed`](Self::speed) is thrown away at random, from
	/// nought to one.
	pub speed_spread: f32,

	/// The half-angle of a cone's mouth, in radians. Ignored by a point.
	pub spread: f32,

	/// How wide a particle is when it is thrown, in world units.
	pub size: f32,

	/// How wide it is when it dies.
	///
	/// Interpolated across the particle's own life, which is where smoke gets
	/// its billow and a spark its taper.
	pub size_end: f32,

	/// The color it is thrown with, linear RGB.
	pub color: Vec3,

	/// The color it dies with, linear RGB.
	pub color_end: Vec3,

	/// How opaque a particle is at its brightest, from nought to one.
	///
	/// The curve itself is not a field: a particle fades in over the first
	/// tenth of its life and out over the last third, which is the shape every
	/// engine's default opacity curve has and is not worth four numbers until
	/// somebody has a curve widget to draw it with.
	pub opacity: f32,

	/// How much of [`World::gravity`](super::World::gravity) a particle feels.
	///
	/// A multiplier rather than an acceleration of its own, so that a world
	/// which points gravity sideways takes its smoke with it. Nought is a
	/// spark in a vacuum; one is a stone.
	pub gravity: f32,

	/// How much of its speed a particle loses a second, as a share.
	///
	/// Nought is a stone and one takes almost everything off in a second,
	/// which is what makes smoke slow down as it rises.
	pub drag: f32,
}

impl Emitter {
	/// Its fields, for an inspector, a reader and a writer. @ref
	/// [`field`](super::field).
	pub const FIELDS: &[Field<Self>] = &[
		word!(
			"kind",
			kind,
			EmitterKind::WORDS,
			EmitterKind::at,
			EmitterKind::index,
			"what shape it throws into, or none"
		),
		word!(
			"blend",
			blend,
			SparkBlend::WORDS,
			SparkBlend::at,
			SparkBlend::index,
			"whether the cloud lightens what is behind it or covers it"
		),
		field!(Texture, "texture", texture, "the picture one particle is drawn with, or none"),
		field!(Float, "rate", rate, "how many particles a second it throws"),
		// by hand rather than through the macro, for the reason
		// `Body::FIELDS`'s two bit masks are: a `Value::Int` is signed and
		// sixty-four bits wide, and this is neither, so the two conversions
		// have to be written where they can refuse a number that will not fit.
		Field {
			name: "cap",
			help: "the most it may have alive at once",
			kind: Kind::Int,
			get: |emitter| Value::Int(i64::from(emitter.cap)),
			set: |emitter, value| match value {
				| Value::Int(held) => match u32::try_from(held) {
					| Ok(cap) => {
						emitter.cap = cap.min(whole(MAX_SPARKS));

						true
					},
					// a negative cap is a refusal rather than a nought: "throw
					// nothing" is spelled with a kind, and somebody who typed
					// minus one meant something this cannot do.
					| Err(_) => false,
				},
				| _ => false,
			},
		},
		field!(Float, "life", life, "how long a particle lives, in seconds"),
		field!(Float, "life_spread", life_spread, "how much of that is thrown away at random"),
		field!(Float, "speed", speed, "how fast a particle leaves, in units a second"),
		field!(Float, "speed_spread", speed_spread, "how much of that is thrown away at random"),
		field!(Float, "spread", spread, "the half-angle of a cone's mouth, in radians"),
		field!(Float, "size", size, "how wide a particle is when it is thrown"),
		field!(Float, "size_end", size_end, "how wide it is when it dies"),
		field!(Color, "color", color, "the color it is thrown with"),
		field!(Color, "color_end", color_end, "the color it dies with"),
		field!(Float, "opacity", opacity, "how opaque a particle is at its brightest"),
		field!(Float, "gravity", gravity, "how much of the world's gravity a particle feels"),
		field!(Float, "drag", drag, "how much of its speed it loses a second, as a share"),
	];
	/// No emitter at all.
	///
	/// The numbers are still the ones an emitter would start with, so that
	/// turning `kind` from `none` to `point` in an inspector throws something
	/// rather than handing back a dead one. Same rule as
	/// [`Light::NONE`](super::Light::NONE), and it is the rule that makes an
	/// inspector's drop-down usable at all.
	pub const NONE: Self = Self {
		kind: EmitterKind::None,
		blend: SparkBlend::Additive,
		texture: TextureId::NONE,
		rate: 32.0,
		cap: 256,
		life: 1.5,
		life_spread: 0.3,
		speed: 2.0,
		speed_spread: 0.4,
		// a quarter turn, which is the mouth Unreal starts a spot light's cone
		// at and reads as a plume rather than as a line or a ball
		spread: std::f32::consts::FRAC_PI_4,
		size: 0.2,
		size_end: 0.6,
		color: Vec3::ONE,
		color_end: Vec3::ONE,
		opacity: 1.0,
		gravity: 0.0,
		drag: 0.5,
	};

	/// A point throwing in every direction.
	///
	/// @param rate - particles a second
	/// @param life - how long each lives, in seconds
	#[must_use]
	pub const fn point(rate: f32, life: f32) -> Self {
		Self {
			kind: EmitterKind::Point,
			rate,
			life,
			..Self::NONE
		}
	}

	/// A cone down the entity's own -z.
	///
	/// @param rate - particles a second
	/// @param life - how long each lives, in seconds
	/// @param spread - the half-angle of its mouth, in radians
	#[must_use]
	pub const fn cone(rate: f32, life: f32, spread: f32) -> Self {
		Self {
			kind: EmitterKind::Cone,
			rate,
			life,
			spread,
			..Self::NONE
		}
	}

	/// Whether this throws anything at all.
	///
	/// The one question the step and the renderer both ask, and it is three
	/// facts rather than the kind alone: an emitter of no kind, one throwing
	/// nothing a second and one whose particles die the moment they are born
	/// are the same absence. Nothing downstream should have to know that.
	#[must_use]
	pub fn throws(&self) -> bool {
		self.kind.throws() && self.rate > 0.0 && self.life > 0.0 && self.cap > 0
	}

	/// How many particles this would like to have thrown by now.
	///
	/// Kept as a float and carried across steps by [`Sparks::owed`], because a
	/// rate under one a step is the common case and rounding it down every
	/// step would throw nothing at all forever.
	///
	/// @param dt - how long the step is, in seconds
	#[must_use]
	pub fn owed(&self, dt: f32) -> f32 { self.rate * dt }
}

impl Default for Emitter {
	fn default() -> Self { Self::NONE }
}

/// One live particle.
///
/// Thirty-six bytes, and everything in it is something the step writes or the
/// renderer reads. What is *not* in it is anything the emitter already says:
/// the size, the two colors and the opacity are read back through
/// [`owner`](Self::owner) when the frame is built, so that turning a knob in
/// the inspector changes the cloud that is already in the air.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spark {
	/// Where it is, in world space.
	///
	/// World rather than the emitter's space, which is the choice with a
	/// consequence: a plume thrown by a moving entity is left behind it rather
	/// than dragged along with it, and that is what smoke does. Godot and
	/// Fyrox both offer the other one as a switch; it is a word on the emitter
	/// the day somebody wants a shield rather than a fire.
	pub position: Vec3,

	/// Which way and how fast it is going, in units a second.
	pub velocity: Vec3,

	/// How long it has been alive, in seconds.
	pub age: f32,

	/// How long it gets, in seconds. Never nought - a particle with no life
	/// is not spawned.
	pub life: f32,

	/// The entity whose emitter threw it.
	///
	/// A handle rather than a slot, so that an emitter which died and whose
	/// slot something else took leaves particles belonging to nobody rather
	/// than particles that suddenly belong to a crate. They are swept on the
	/// step that finds them.
	pub owner: EntityId,
}

impl Spark {
	/// How far through its life it is, from nought to one.
	#[must_use]
	pub fn through(&self) -> f32 {
		if self.life > 0.0 {
			(self.age / self.life).clamp(0.0, 1.0)
		} else {
			1.0
		}
	}

	/// Whether it is still alive.
	#[must_use]
	pub fn alive(&self) -> bool { self.age < self.life }

	/// How opaque it is right now, before the emitter's own opacity.
	///
	/// **In rather than out**, and the two shoulders are not the same width:
	/// a tenth of its life to appear and a third of it to go. A particle that
	/// appeared at full brightness pops, and one that faded symmetrically
	/// spends half its life invisible. Every engine read here ships a default
	/// opacity curve of roughly this shape; Wicked writes it out as
	/// `opacityCurveControlPeakStart = 0.1` and a peak end just past the
	/// middle.
	#[must_use]
	pub fn fade(&self) -> f32 {
		const IN: f32 = 0.1;
		const OUT: f32 = 0.667;

		let through = self.through();

		if through < IN {
			through / IN
		} else if through > OUT {
			(1.0 - through) / (1.0 - OUT)
		} else {
			1.0
		}
	}
}

/// Every particle alive in the world, and what each emitter is owed.
///
/// One flat pool rather than a buffer per emitter, and the reason is the
/// picture: what the renderer wants is one instance buffer sorted by what
/// decides a pipeline and a bind group, and a pool per emitter would have to
/// be gathered into exactly that every frame anyway. The pool is compacted by
/// the step, so the live ones are always a prefix of it.
///
/// **Host-written and not saved.** A game may read it; nothing in the ABI lets
/// a game push a particle in, because the arithmetic that would have to agree
/// with it - the spawn, the fade, the compaction - is the host's. @ref
/// `colby_runtime::sparks`.
pub struct Sparks {
	/// The live ones, in no order anybody may rely on.
	live: Vec<Spark>,

	/// What each emitting entity is owed, as a fraction of a particle.
	///
	/// A parallel array over the entity table's slots, like the emitters
	/// themselves, and the whole of why a rate under one a step works: the
	/// remainder is kept rather than rounded away. Reset with the slot.
	owed: Vec<f32>,

	/// How many particles the pool refused to spawn since the world started.
	///
	/// A counter rather than a warning, because the honest answer to "this
	/// emitter is at its cap" is a number on `--profile` rather than a line
	/// in a log every step.
	pub refused: u64,
}

impl Sparks {
	/// An empty pool.
	#[must_use]
	pub const fn new() -> Self {
		Self {
			live: Vec::new(),
			owed: Vec::new(),
			refused: 0,
		}
	}

	/// How many particles are alive.
	#[must_use]
	pub fn len(&self) -> usize { self.live.len() }

	/// Whether none are.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.live.is_empty() }

	/// Every live particle, in the pool's own order.
	pub fn iter(&self) -> impl Iterator<Item = &Spark> { self.live.iter() }

	/// Every live particle, to be moved.
	pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Spark> { self.live.iter_mut() }

	/// How much of a particle an entity's slot has been owed.
	///
	/// @param slot - the entity's place in the table
	#[must_use]
	pub fn owed(&self, slot: usize) -> f32 { self.owed.get(slot).copied().unwrap_or(0.0) }

	/// Records what an entity's slot is owed after this step's spawning.
	///
	/// @param slot - the entity's place in the table
	/// @param owed - what is left over, which should be under one
	pub fn set_owed(&mut self, slot: usize, owed: f32) {
		if slot >= self.owed.len() {
			self.owed.resize(slot.saturating_add(1), 0.0);
		}

		if let Some(held) = self.owed.get_mut(slot) {
			*held = owed;
		}
	}

	/// Puts a particle in the pool, if there is room in the world for it.
	///
	/// @param spark - the particle
	/// @return whether it went in
	pub fn push(&mut self, spark: Spark) -> bool {
		if self.live.len() >= MAX_SPARKS {
			self.refused = self.refused.saturating_add(1);

			return false;
		}

		self.live.push(spark);

		true
	}

	/// How many of the pool belong to one entity.
	///
	/// @param owner - the entity to count for
	#[must_use]
	pub fn count(&self, owner: EntityId) -> usize {
		self.live
			.iter()
			.filter(|spark| spark.owner == owner)
			.count()
	}

	/// Drops every particle that is dead or whose emitter is gone.
	///
	/// @param alive - answers whether an owner is still emitting
	pub fn sweep<F: Fn(EntityId) -> bool>(&mut self, alive: F) {
		self.live
			.retain(|spark| spark.alive() && alive(spark.owner));
	}

	/// Drops every particle an entity threw.
	///
	/// @param owner - whose to drop
	pub fn forget(&mut self, owner: EntityId) { self.live.retain(|spark| spark.owner != owner); }

	/// Empties the pool.
	///
	/// What a scene load and a restore do: the cloud belongs to the world that
	/// was here, and the world that arrives has not thrown anything yet.
	pub fn clear(&mut self) {
		self.live.clear();
		self.owed.clear();
		self.refused = 0;
	}
}

impl Default for Sparks {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_word_list_and_its_variants_are_the_same_length() {
		assert_eq!(EmitterKind::WORDS.len(), 3, "three kinds");
		assert!(EmitterKind::at(3).is_none(), "and nothing past them");
		assert_eq!(SparkBlend::WORDS.len(), 2, "two ways to reach the picture");
		assert!(SparkBlend::at(2).is_none(), "and nothing past them");

		for (index, word) in EmitterKind::WORDS.iter().enumerate() {
			let index = u32::try_from(index).expect("three fits");
			let kind = EmitterKind::at(index).expect("every place has a kind");

			assert_eq!(kind.word(), *word, "the word round-trips through the place");
			assert_eq!(kind.index(), index, "and the place through the kind");
		}

		for (index, word) in SparkBlend::WORDS.iter().enumerate() {
			let index = u32::try_from(index).expect("two fits");
			let blend = SparkBlend::at(index).expect("every place has one");

			assert_eq!(blend.word(), *word, "the word round-trips through the place");
			assert_eq!(blend.index(), index, "and the place through it");
			assert_eq!(
				u32::try_from(blend.row()).expect("two rows fit"),
				index,
				"and the pipeline row is the place"
			);
		}
	}

	#[test]
	fn an_emitter_that_is_none_throws_nothing_but_is_ready_to() {
		let none = Emitter::NONE;

		assert!(!none.throws(), "an emitter of no kind throws nothing");
		assert!(none.rate > 0.0, "but its rate is one somebody could use");
		assert!(none.life > 0.0, "and so is its life");
		assert!(none.cap > 0, "and so is its cap");

		let lit = Emitter { kind: EmitterKind::Point, ..none };

		assert!(lit.throws(), "so turning the word on throws something at once");
	}

	#[test]
	fn an_emitter_with_a_dead_number_in_it_throws_nothing() {
		for dead in [
			Emitter { rate: 0.0, ..Emitter::point(8.0, 1.0) },
			Emitter { life: 0.0, ..Emitter::point(8.0, 1.0) },
			Emitter { cap: 0, ..Emitter::point(8.0, 1.0) },
		] {
			assert!(
				!dead.throws(),
				"a rate, a life or a cap of nothing is the same absence as no kind"
			);
		}
	}

	#[test]
	fn a_particle_fades_in_faster_than_it_fades_out() {
		let spark = |age: f32| Spark {
			position: Vec3::ZERO,
			velocity: Vec3::ZERO,
			age,
			life: 1.0,
			owner: EntityId::NONE,
		};

		assert!(spark(0.0).fade().abs() < 1e-6, "it appears from nothing");
		assert!((spark(0.1).fade() - 1.0).abs() < 1e-6, "and is whole a tenth of the way in");
		assert!((spark(0.5).fade() - 1.0).abs() < 1e-6, "and stays whole through the middle");
		assert!(spark(0.9).fade() < 1.0, "and is going by nine tenths");
		assert!(spark(1.0).fade().abs() < 1e-6, "and is gone at the end");
		assert!(
			spark(0.05).fade() > spark(0.95).fade(),
			"the shoulders are not the same width, which is what stops a particle popping"
		);
	}

	#[test]
	fn a_particle_with_no_life_is_wholly_through_rather_than_dividing_by_nought() {
		let dead = Spark {
			position: Vec3::ZERO,
			velocity: Vec3::ZERO,
			age: 0.0,
			life: 0.0,
			owner: EntityId::NONE,
		};

		assert!((dead.through() - 1.0).abs() < 1e-6, "wholly through rather than a nan");
		assert!(!dead.alive(), "and not alive");
	}

	#[test]
	fn the_pool_stops_at_the_world_ceiling_and_counts_what_it_refused() {
		let mut pool = Sparks::new();
		let spark = Spark {
			position: Vec3::ZERO,
			velocity: Vec3::ZERO,
			age: 0.0,
			life: 1.0,
			owner: EntityId::NONE,
		};

		for _ in 0..MAX_SPARKS {
			assert!(pool.push(spark), "there is room until there is not");
		}

		assert_eq!(pool.len(), MAX_SPARKS, "the pool is full");
		assert!(!pool.push(spark), "and the next one does not go in");
		assert_eq!(pool.refused, 1, "and is counted rather than logged");
	}

	#[test]
	fn a_sweep_drops_the_dead_and_the_orphaned_and_keeps_the_rest() {
		let mut pool = Sparks::new();
		let mine = EntityId::at(1, 1);
		let theirs = EntityId::at(2, 1);
		let spark = |owner, age| Spark {
			position: Vec3::ZERO,
			velocity: Vec3::ZERO,
			age,
			life: 1.0,
			owner,
		};

		assert!(pool.push(spark(mine, 0.0)), "a live one of mine");
		assert!(pool.push(spark(mine, 2.0)), "a dead one of mine");
		assert!(pool.push(spark(theirs, 0.0)), "and a live one of an emitter that is going");

		pool.sweep(|owner| owner == mine);

		assert_eq!(pool.len(), 1, "the dead one and the orphan both go");
		assert_eq!(pool.count(mine), 1, "and the live one of mine stays");
		assert_eq!(pool.count(theirs), 0, "and nothing of theirs is left");
	}

	#[test]
	fn what_a_slot_is_owed_survives_a_step_and_a_clear_takes_it_away() {
		let mut pool = Sparks::new();

		assert!(pool.owed(7).abs() < 1e-6, "a slot nobody has written is owed nothing");

		pool.set_owed(7, 0.75);

		assert!((pool.owed(7) - 0.75).abs() < 1e-6, "and what is left over is kept");
		assert!(pool.owed(6).abs() < 1e-6, "without disturbing the slot before it");

		pool.clear();

		assert!(pool.owed(7).abs() < 1e-6, "and a load takes the whole thing away");
	}

	#[test]
	fn a_rate_under_one_a_step_still_throws_something() {
		// half a particle a second at a sixtieth of a second a step: a hundred
		// and twenty steps for one particle, which rounding down would never
		// reach.
		//
		// @note: five seconds rather than four, and the odd number is the
		// point. Two hundred and forty steps of `0.5 / 60.0` add up to a
		// whisker *under* two, so the honest count there is one - a
		// remainder carried in a float is a remainder carried in a float, and
		// a test written on a boundary would be testing the rounding rather
		// than the mechanism.
		let slow = Emitter { rate: 0.5, ..Emitter::point(0.5, 4.0) };
		let dt = 1.0 / 60.0;
		let mut owed = 0.0;
		let mut thrown = 0_u32;

		for _ in 0..300 {
			owed += slow.owed(dt);

			// the same `floor` the step itself takes, rather than a loop that
			// subtracts one at a time: one statement about a float instead of
			// a condition that depends on how often it has been rounded. At
			// half a particle a second no step can owe more than one, so the
			// count is a question rather than a conversion.
			let whole = owed.floor().max(0.0);

			owed -= whole;

			if whole > 0.5 {
				thrown += 1;
			}
		}

		assert_eq!(thrown, 2, "five seconds at half a second is two particles");
		assert!(owed > 0.0, "and the third is half owed rather than thrown away");
	}

	#[test]
	fn forgetting_an_entity_leaves_every_other_cloud_alone() {
		let mut pool = Sparks::new();
		let mine = EntityId::at(1, 1);
		let theirs = EntityId::at(2, 1);
		let spark = |owner| Spark {
			position: Vec3::ZERO,
			velocity: Vec3::ZERO,
			age: 0.0,
			life: 1.0,
			owner,
		};

		assert!(pool.push(spark(mine)), "one of mine");
		assert!(pool.push(spark(theirs)), "and one of theirs");

		pool.forget(mine);

		assert_eq!(pool.count(mine), 0, "mine is gone");
		assert_eq!(pool.count(theirs), 1, "and theirs is not");
	}
}
