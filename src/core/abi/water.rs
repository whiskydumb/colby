//! What a body is filled with, when it is filled with anything.
//!
//! One record with a kind, exactly as [`Light`](super::light::Light) is one
//! record with a kind, and for the same reason: a word appended to
//! [`WaterKind::WORDS`] costs no format anything, where a second struct would
//! cost every reader, every writer and every panel a second branch. The word
//! that is coming is `plane` - an unbounded surface at a height, which a lake
//! wants and a box cannot describe.
//!
//! **Water is a body, not a record on the world and not a shape of its own.**
//! A body already has a box, a transform, a place in a file, a row in the
//! hierarchy, a gizmo and an undo, and none of those had to learn what water
//! is. The engine this follows is Unreal, whose `UBuoyancyComponent` finds its
//! water through ordinary overlap events
//! (`BuoyancyComponent.cpp:262`, `SetGenerateOverlapEvents(true)`); Jolt keeps
//! an `AABox` and a plane and queries the broadphase by hand
//! (`Samples/Tests/Water/WaterShapeTest.cpp:98-131`), but that is a sample
//! rather than a feature, and colby has the machinery Jolt chose not to use.
//!
//! **A body that is water does not push.** [`Body::solid`](super::Body::solid)
//! reads this as well as the sensor flag, so a pool is felt by the narrow
//! phase, reported through [`Bodies::overlaps`](super::Bodies::overlaps), and
//! walked through. Two ticks would be one too many: a pool that shoved things
//! out of itself is not a thing anybody wants and would be the first bug
//! reported.
//!
//! **What is *not* here is what a body does about it.** The buoyancy is the
//! solver's, it is worked out from the shape's own submerged volume, and it is
//! applied through [`Bodies::apply_force_at`](super::Bodies::apply_force_at) -
//! which is what makes the righting torque a consequence of where the push
//! lands rather than a term somebody wrote. @ref `colby_physics`.

use super::field::{Field, field, word};
use crate::glam::Vec3;

/// How dense a fluid is unless it says otherwise, in mass per cubic unit.
///
/// **Two, and the number is not a claim about water.** colby's masses are in
/// whatever unit a scene is consistent about, and the body this engine hands
/// out by default - [`Shape::UNIT`](super::Shape::UNIT), a cube one across, at
/// [`Body::MASS`](super::Body::MASS) - has a volume of one and a mass of one,
/// so its density is one. A fluid at twice that floats it with exactly half of
/// it above the surface, which is what a person dropping the default crate
/// into the default pool should see. A fluid at one would leave it neutrally
/// buoyant and hanging wherever it was let go, which reads as a bug.
pub const DENSITY: f32 = 2.0;

/// How hard a fluid resists being moved through, unless it says otherwise.
///
/// Jolt's own sample number (`WaterShapeTest.cpp:114` passes `0.3f`), against
/// the same quadratic law. @ref [`Water::linear_drag`].
pub const LINEAR_DRAG: f32 = 0.3;

/// How hard a fluid stops a bob, unless it says otherwise.
///
/// **Eight and a half, far larger than the drags below, and it is the number
/// that makes water settle rather than ring.** It is just under the critical
/// damping of the body this engine hands out: a unit cube of mass one in a
/// fluid of [`DENSITY`] is a spring of stiffness `density * area * gravity`,
/// so its critical coefficient is about `8.9`, being
/// `2 * sqrt(mass * density * area * gravity)`; this term contributes
/// `damp * density * area * wet`, which at the half-submerged waterline is
/// `damp`. Just under it, so a crate dropped in overshoots once and stops.
///
/// The number matters because the quadratic drag cannot do this job. A drag
/// in the square of the speed takes almost nothing off a *small* oscillation:
/// a crate a tenth of a unit off its waterline loses about a hundredth of
/// its energy a cycle at Jolt's coefficient, so a crate left to it alone
/// bobs for minutes of simulated time. That is measured rather than
/// estimated: it is what the first version of this did. Unreal has the same
/// term for the same reason and calls it `BuoyancyDamp`
/// (`BuoyancyComponent.cpp:357`). @ref [`Water::damp`].
pub const DAMP: f32 = 8.5;

/// How hard a fluid resists being turned in, unless it says otherwise.
///
/// Jolt's sample number again (`0.05f`), and much smaller than the linear one
/// because it multiplies a squared width. @ref [`Water::angular_drag`].
pub const ANGULAR_DRAG: f32 = 0.05;

/// What kind of fluid a body holds.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WaterKind {
	/// Nothing: the body is not water at all.
	///
	/// A variant rather than an `Option`, because every body holds one of
	/// these and a record that is written down needs a spelling for "no
	/// water" as much as for the other one.
	#[default]
	None,

	/// The whole of the body's shape, filled to the top.
	///
	/// The surface is the highest point of the shape's world-space bounds and
	/// it is level, which is what makes it a plane the arithmetic can clip
	/// against. A tipped-up pool is a pool with a level surface and sloping
	/// walls, not a slope of water.
	Volume,
}

impl WaterKind {
	/// The word each kind is written as, in declaration order.
	///
	/// A file's vocabulary and an inspector's drop-down, @ref
	/// [`field::Kind::Word`](super::field::Kind::Word). A third word may be
	/// appended without moving any format: what is stored is the place in
	/// this list.
	pub const WORDS: &[&str] = &["none", "volume"];

	/// The kind at a place in [`WORDS`](Self::WORDS), if there is one.
	///
	/// @param index - the place
	#[must_use]
	pub const fn at(index: u32) -> Option<Self> {
		match index {
			| 0 => Some(Self::None),
			| 1 => Some(Self::Volume),
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

	/// Whether a body of this kind holds any fluid at all.
	#[must_use]
	pub const fn is_wet(self) -> bool { !matches!(self, Self::None) }
}

/// One body's fluid: what kind, how dense, how thick, and which way it runs.
///
/// Plain data behind [`World`](super::World) like everything else a body
/// carries. Not `#[repr(C)]` and not `Pod`: it never crosses as raw bytes, and
/// the file that writes it has a record of its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Water {
	/// What fills it, or [`WaterKind::None`] for a body that is not water.
	pub kind: WaterKind,

	/// How heavy the fluid is, in mass per cubic unit.
	///
	/// **This is the whole of what decides where something floats**, because
	/// the push up is this times the volume under the surface times gravity,
	/// and the pull down is the body's own mass times gravity. A body settles
	/// where those meet, so one twice as heavy sits twice as deep until it
	/// runs out of shape and sinks.
	///
	/// A real number rather than Jolt's unitless multiplier, which is the one
	/// place this deliberately parts from it: Jolt's `inBuoyancy` is a
	/// property of the *pair* - it derives `fluid_density = buoyancy / (volume
	/// * inverse_mass)` at `Body.cpp:218`, so the same water floats two
	/// bodies by two different rules. Here the fluid has a density and the
	/// body has a mass, and neither needs to know the other. @ref [`DENSITY`].
	pub density: f32,

	/// How hard it stops something bobbing, against its vertical speed.
	///
	/// First order and vertical only, which is Unreal's `BuoyancyDamp`
	/// (`BuoyancyComponent.cpp:357-360`, where it is subtracted from the
	/// buoyant force rather than added beside it). It is a separate number
	/// from the drag below and not a duplicate of it: the drag is what the
	/// fluid does to something crossing it, and this is what the fluid does
	/// to something floating in it, and only the second one decides whether a
	/// crate ever comes to rest. @ref [`DAMP`] for the arithmetic behind the
	/// default and for why the drag cannot stand in for this.
	///
	/// **Symmetric, where Unreal's is one-sided**: `FMath::Max(.., 0.f)`
	/// there means the term only ever reduces the push, so a hull is damped
	/// on the way up and not on the way down. A crate that fell into water
	/// fast and rose out of it slowly would look wrong, and a real fluid
	/// resists both.
	pub damp: f32,

	/// How hard it is to be dragged through, against the square of the speed.
	///
	/// Quadratic rather than linear, which is Jolt's deliberate deviation from
	/// the article it follows (`Body.cpp:232-236`) and is what real drag does
	/// above a crawl. Multiplied by how much of the body is under the surface
	/// and by the area it presents to the flow, so a plank edge-on slows less
	/// than the same plank face-on.
	///
	/// **On by default**, which is where this parts from Unreal:
	/// `bApplyDragForcesInWater` is `false` there (`BuoyancyTypes.h:220`)
	/// because that plugin is for boats that are meant to feel fast. A crate
	/// dropped in a pool is meant to stop, and water with no drag is a
	/// trampoline.
	pub linear_drag: f32,

	/// How hard it is to be turned in, against the spin.
	///
	/// Linear in the angular velocity rather than quadratic, which is what
	/// both references do (`Body.cpp:272`, `BuoyancyComponent.cpp:784`).
	pub angular_drag: f32,

	/// Which way the fluid itself is moving, in units a second, world space.
	///
	/// Zero is still water. The drag is computed against the body's speed
	/// *relative to this*, so a current carries what is floating in it without
	/// any second mechanism - which is Jolt's `inFluidVelocity`
	/// (`Body.cpp:196`) and is the whole of what a river needs from the
	/// physics half.
	pub flow: Vec3,
}

impl Water {
	/// Its fields, for an inspector, a reader and a writer. @ref
	/// [`field`](super::field).
	pub const FIELDS: &[Field<Self>] = &[
		word!(
			"kind",
			kind,
			WaterKind::WORDS,
			WaterKind::at,
			WaterKind::index,
			"what fills the body, or none"
		),
		field!(Float, "density", density, "how heavy the fluid is, in mass per cubic unit"),
		field!(Float, "damp", damp, "how hard it stops something bobbing"),
		field!(Float, "linear_drag", linear_drag, "how hard it resists being moved through"),
		field!(Float, "angular_drag", angular_drag, "how hard it resists being turned in"),
		field!(Vec3, "flow", flow, "which way the fluid runs, in units a second"),
	];
	/// No fluid at all: an ordinary body, and nothing floats in it.
	///
	/// The three numbers are still a fluid somebody would recognize, so that
	/// turning `kind` from `none` to `volume` in an inspector floats something
	/// rather than swallowing it. Same rule
	/// [`Sky::NONE`](super::sky::Sky::NONE) keeps for its colors.
	pub const NONE: Self = Self {
		kind: WaterKind::None,
		density: DENSITY,
		damp: DAMP,
		linear_drag: LINEAR_DRAG,
		angular_drag: ANGULAR_DRAG,
		flow: Vec3::ZERO,
	};

	/// Whether this body holds any fluid at all.
	#[must_use]
	pub const fn is_wet(self) -> bool { self.kind.is_wet() }

	/// The three defaults, filled.
	#[must_use]
	pub const fn pool() -> Self { Self { kind: WaterKind::Volume, ..Self::NONE } }

	/// A fluid of a density, otherwise the defaults.
	///
	/// @param density - mass per cubic unit
	#[must_use]
	pub const fn of(density: f32) -> Self { Self { density, ..Self::pool() } }

	/// How fast the body at a point is moving through this fluid.
	///
	/// The one place [`flow`](Self::flow) is read, kept here so that a caller
	/// cannot get the subtraction the wrong way round: the answer is what the
	/// *fluid* sees the body doing, which is what a drag opposes.
	///
	/// @param velocity - how fast the point is moving, world space
	/// @return the body's velocity relative to the fluid
	#[must_use]
	pub fn against(&self, velocity: Vec3) -> Vec3 { velocity - self.flow }
}

impl Default for Water {
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
		for (index, word) in WaterKind::WORDS.iter().enumerate() {
			let index = u32::try_from(index).expect("two of them");
			let kind = WaterKind::at(index).expect("every place has a kind");

			assert_eq!(kind.index(), index, "{word} is at {index}");
			assert_eq!(kind.word(), *word, "and is written as itself");
		}

		assert!(
			WaterKind::at(u32::try_from(WaterKind::WORDS.len()).expect("two")).is_none(),
			"one past the end is no kind"
		);
	}

	#[test]
	fn nothing_is_what_a_body_starts_with_and_the_numbers_survive_it() {
		assert!(!Water::NONE.is_wet(), "an ordinary body holds nothing");
		assert_eq!(Water::default(), Water::NONE, "and that is the default");
		assert!(Water::pool().is_wet(), "the same numbers, turned on");
		assert_eq!(
			(Water::pool().density, Water::pool().damp, Water::pool().linear_drag),
			(Water::NONE.density, Water::NONE.damp, Water::NONE.linear_drag),
			"which is what makes turning the word on float something"
		);
	}

	#[test]
	fn the_default_density_floats_the_default_body_half_out() {
		// the body this engine hands out: a cube one across, mass one. Half of
		// it under a fluid of `DENSITY` displaces a mass equal to its own, so
		// that is where it settles - and the doc comment on `DENSITY` says so.
		let (volume, mass) = (1.0_f32, 1.0_f32);
		let submerged = mass / DENSITY;

		assert!(
			(submerged / volume - 0.5).abs() < f32::EPSILON,
			"half of the default crate stands out of the default pool"
		);
	}

	#[test]
	fn the_default_damping_is_just_under_what_stops_the_default_crate_dead() {
		// the derivation the constant's own doc gives, checked rather than
		// asserted in prose: a unit cube of mass one floating at half its
		// depth is a spring of this stiffness, and this is its critical
		// coefficient.
		let (mass, area, gravity) = (1.0_f32, 1.0_f32, 9.81_f32);
		let critical = 2.0 * (mass * DENSITY * area * gravity).sqrt();
		// what the term contributes at the waterline, where half is under
		let ours = DAMP * DENSITY * area * 0.5;

		assert!(ours < critical, "under it, so a crate dropped in overshoots once");
		assert!(
			ours > critical * 0.9,
			"but only just, so it does not ring: {ours} against {critical}"
		);
	}

	#[test]
	fn a_flow_is_subtracted_the_way_a_drag_wants_it() {
		let running = Water {
			flow: Vec3::new(3.0, 0.0, 0.0),
			..Water::pool()
		};

		assert_eq!(
			running.against(Vec3::new(3.0, 0.0, 0.0)),
			Vec3::ZERO,
			"something carried along by the current is not moving through it at all"
		);
		assert_eq!(
			running.against(Vec3::ZERO),
			Vec3::new(-3.0, 0.0, 0.0),
			"and something held still is being dragged upstream through it"
		);
	}

	#[test]
	fn every_field_reads_back_what_it_was_written() {
		let mut water = Water::NONE;

		for entry in Water::FIELDS {
			let written = match entry.kind {
				| Kind::Word(words) =>
					Value::Word(u32::try_from(words.len()).expect("a short list") - 1),
				| Kind::Float => Value::Float(0.75),
				| Kind::Vec3 => Value::Vec3(Vec3::new(0.25, 0.5, 0.75)),
				| kind => panic!("water has no field of {kind:?}"),
			};

			assert!(entry.set(&mut water, written.clone()), "{} takes its own kind", entry.name);
			assert_eq!(entry.get(&water), written, "{} hands back what it took", entry.name);
		}

		assert_eq!(water.kind, WaterKind::Volume, "the last word is the last kind");
	}
}
