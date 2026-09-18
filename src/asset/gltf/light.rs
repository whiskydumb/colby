//! Lamps: the `KHR_lights_punctual` extension, turned into colby's lights.
//!
//! The extension declares its lights once, in a list at the top of the
//! document, and a node names one by its place in that list - the way a node
//! names a mesh. So this reads the list, and the walk in `geometry` stands a
//! lamp wherever a node names one.
//!
//! **Three decisions, each a place the extension and colby disagree:**
//!
//! - **The units.** The file says candela, and colby's intensity is a
//!   multiplier whose one is the brightness a path tracer's render of the same
//!   lamp gives a white surface: 683 lumens a watt, which is how an exporter
//!   turns a lamp's watts into candela, times the pi its shading divides a
//!   rough surface by. So a lamp's candela are divided by [`PER_UNIT`] and
//!   nothing else is done to them.
//! - **The reach.** A lamp the file gave no range reaches for ever by the
//!   extension's own words, and every light in colby has an edge past which it
//!   is not sent to the shader at all. It gets the distance at which what it
//!   sends a surface facing it has fallen to a 256th of one, the light of a sun
//!   of one on a white floor: `sqrt(intensity x 256)` by the inverse square.
//! - **The sun.** A directional light is left out, with a word: colby's sun
//!   belongs to the world rather than to anything standing in it, and a model
//!   bringing a second one would have nowhere to put it.
//!
//! The color is linear in the file and in colby, and the two cone angles are
//! the same half-angles in both, so those cross as they are; a lamp throws a
//! shadow, which is what a lamp placed by hand does.

use colby_core::abi::{Light, LightKind};

use super::{Gltf, material::color};
use crate::json::Value;

/// The extension, by name.
pub(super) const PUNCTUAL: &str = "KHR_lights_punctual";

/// How many candela one unit of a lamp's intensity is.
///
/// Measured rather than chosen: the same lamp rendered by a path tracer and by
/// colby lights a white surface to the same number when the file's candela are
/// divided by this. @ref `colby-gltf-large-references` in memory.
const PER_UNIT: f32 = 683.0 * std::f32::consts::PI;

/// How far below a lamp's own brightness its reach ends, when the file gives it
/// none: one part in this many.
const FAINTEST: f32 = 256.0;

/// Every lamp a file declares, in the file's own order.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Lamps {
	/// One per entry of the extension's list, so a node's `light` indexes it
	/// directly. A light colby does not take - a sun, a kind it has no word
	/// for - is here as nothing, and has been warned about.
	pub lights: Vec<Option<Light>>,

	/// What could not be used. Not a failure.
	pub warnings: Vec<String>,
}

/// Reads every lamp a file declares.
///
/// @param file - the document
/// @return one light per entry of the list, and what could not be used
#[must_use]
pub(super) fn read(file: &Gltf) -> Lamps {
	let listed = file
		.document()
		.get("extensions")
		.and_then(|all| all.get(PUNCTUAL))
		.and_then(|punctual| punctual.get("lights"))
		.map_or(&[][..], Value::as_array);
	let mut out = Lamps::default();

	for (index, entry) in listed.iter().enumerate() {
		let light = lamp(index, entry, &mut out.warnings);

		out.lights.push(light);
	}

	out
}

/// Which of the file's lights a node stands, if it names one.
///
/// @param node - the node
/// @return the index it wrote, which may be past the end of the list
#[must_use]
pub(super) fn named_by(node: &Value) -> Option<usize> {
	node.get("extensions")
		.and_then(|all| all.get(PUNCTUAL))
		.and_then(|punctual| punctual.get("light"))
		.and_then(Value::as_usize)
}

/// One light of the list, as colby's, or nothing with a word about why.
fn lamp(index: usize, entry: &Value, warnings: &mut Vec<String>) -> Option<Light> {
	let called = entry
		.get("name")
		.and_then(Value::as_str)
		.map_or_else(String::new, |name| format!(" ({name})"));
	let kind = match entry.get("type").and_then(Value::as_str) {
		| Some("point") => LightKind::Point,
		| Some("spot") => LightKind::Spot,
		| Some("directional") => {
			warnings.push(format!(
				"light {index}{called} is directional, and is left out: colby's sun belongs to \
				 the world rather than to a model"
			));

			return None;
		},
		| written => {
			warnings.push(format!(
				"light {index}{called} is of type {}, which colby does not have, and is left out",
				written.unwrap_or("nothing")
			));

			return None;
		},
	};
	let intensity = entry
		.get("intensity")
		.and_then(Value::as_f32)
		.unwrap_or(1.0)
		/ PER_UNIT;
	let written = entry.get("range").and_then(Value::as_f32);
	let range = match written {
		| Some(range) if range > 0.0 => range,
		| Some(range) => {
			warnings.push(format!(
				"light {index}{called} reaches {range}, which is not a distance, and is given \
				 the reach its brightness implies"
			));

			reach(intensity)
		},
		| None => reach(intensity),
	};
	let cone = entry.get("spot");

	Some(Light {
		kind,
		color: color(entry.get("color")).unwrap_or(Light::NONE.color),
		intensity,
		range,
		inner: angle(cone, "innerConeAngle").unwrap_or(0.0),
		outer: angle(cone, "outerConeAngle").unwrap_or(std::f32::consts::FRAC_PI_4),
		..Light::NONE
	})
}

/// How far a lamp of this brightness reaches when the file does not say.
///
/// Nowhere for a brightness below nothing, which the extension does not allow
/// and which would otherwise be a reach that is not a number.
///
/// @param intensity - colby's, after the units
fn reach(intensity: f32) -> f32 { (intensity * FAINTEST).max(0.0).sqrt() }

/// One of a cone's two angles, when the file wrote it.
fn angle(cone: Option<&Value>, name: &str) -> Option<f32> {
	cone.and_then(|written| written.get(name))
		.and_then(Value::as_f32)
}

#[cfg(test)]
mod tests {
	use std::path::Path;

	use colby_core::glam::Vec3;

	use super::*;

	/// A document declaring these lights and nothing else.
	fn declared(lights: &str) -> Lamps {
		let text = format!(
			"{{ \"asset\": {{ \"version\": \"2.0\" }}, \"extensionsUsed\": [ \
			 \"KHR_lights_punctual\" ], \"extensions\": {{ \"KHR_lights_punctual\": {{ \
			 \"lights\": [ {lights} ] }} }} }}"
		);
		let file = Gltf::read(text.as_bytes(), Path::new("lamps.gltf"), Path::new(""))
			.expect("the document reads");

		read(&file)
	}

	#[test]
	fn a_lamps_candela_are_divided_by_the_measured_unit_and_nothing_else() {
		// a thousand-watt bulb as an exporter writes it: a thousand over four pi,
		// times 683
		let candela = 1000.0 / (4.0 * std::f64::consts::PI) * 683.0;
		let lamps = declared(&format!("{{ \"type\": \"point\", \"intensity\": {candela} }}"));
		let lamp = lamps.lights[0].expect("a point is taken");
		// the division as the importer does it, to the bit: the file's number
		// narrowed the way the reader narrows one, over the unit in floats
		let narrowed = Value::Number(candela).as_f32().expect("a number");

		assert_eq!(lamp.intensity.to_bits(), (narrowed / PER_UNIT).to_bits(), "{lamp:?}");
		assert!(
			(lamp.intensity - 25.330_296).abs() < 1e-4,
			"a thousand watts is a thousand over four pi squared: {}",
			lamp.intensity
		);
		assert_eq!(lamp.kind, LightKind::Point, "and a point it stays");
		assert!(lamps.warnings.is_empty(), "with nothing to say: {:?}", lamps.warnings);
	}

	#[test]
	fn a_lamp_the_file_gave_no_reach_reaches_where_it_falls_to_a_256th() {
		let lamps = declared(
			"{ \"type\": \"point\", \"intensity\": 5000 }, { \"type\": \"point\", \
			 \"intensity\": 5000, \"range\": 3.5 }",
		);
		let [Some(open), Some(given)] = lamps.lights[..] else {
			panic!("both are taken: {lamps:?}");
		};

		assert_eq!(open.range.to_bits(), (open.intensity * 256.0).sqrt().to_bits());
		assert!(
			(open.range * open.range / 256.0 - open.intensity).abs() < 1e-3,
			"which is where an inverse square of it is a 256th: {open:?}"
		);
		assert_eq!(given.range.to_bits(), 3.5_f32.to_bits(), "a range the file gave is kept");
		assert_eq!(open.intensity.to_bits(), given.intensity.to_bits(), "and nothing else moves");
	}

	#[test]
	fn a_range_that_is_not_a_distance_is_said_and_replaced() {
		let lamps = declared("{ \"type\": \"point\", \"intensity\": 5000, \"range\": 0 }");
		let lamp = lamps.lights[0].expect("the lamp is still taken");

		assert_eq!(lamp.range.to_bits(), (lamp.intensity * 256.0).sqrt().to_bits());
		assert_eq!(lamps.warnings.len(), 1, "and it is said once: {:?}", lamps.warnings);
	}

	#[test]
	fn a_cone_and_a_color_cross_as_they_are() {
		let lamps = declared(
			"{ \"type\": \"spot\", \"color\": [ 0.6, 0.8, 1.0 ], \"spot\": { \
			 \"innerConeAngle\": 0.25, \"outerConeAngle\": 0.6 } }",
		);
		let lamp = lamps.lights[0].expect("a spot is taken");

		assert_eq!(lamp.kind, LightKind::Spot, "a spot");
		assert_eq!(lamp.color, Vec3::new(0.6, 0.8, 1.0), "linear in both");
		assert_eq!(
			[lamp.inner, lamp.outer].map(f32::to_bits),
			[0.25_f32, 0.6].map(f32::to_bits),
			"the same half-angles in both"
		);
		assert!(lamp.shadow, "and it throws a shadow, as a lamp placed by hand does");
	}

	#[test]
	fn what_the_file_leaves_out_is_what_the_extension_says_it_means() {
		let lamps = declared("{ \"type\": \"spot\" }");
		let lamp = lamps.lights[0].expect("a bare spot is taken");

		assert_eq!(lamp.color, Vec3::ONE, "white");
		assert_eq!(lamp.intensity.to_bits(), (1.0 / PER_UNIT).to_bits(), "one candela");
		assert_eq!(lamp.inner.to_bits(), 0.0_f32.to_bits(), "a cone with no bright middle");
		assert_eq!(
			lamp.outer.to_bits(),
			std::f32::consts::FRAC_PI_4.to_bits(),
			"a quarter turn across"
		);
	}

	#[test]
	fn a_sun_and_a_kind_colby_does_not_have_are_left_out_and_said() {
		let lamps = declared(
			"{ \"type\": \"directional\", \"name\": \"Sun\", \"intensity\": 3 }, { \"type\": \
			 \"area\" }, { \"type\": \"point\" }",
		);

		assert_eq!(
			lamps.lights.len(),
			3,
			"every entry keeps its place, for the nodes naming one"
		);
		assert!(lamps.lights[0].is_none() && lamps.lights[1].is_none(), "{:?}", lamps.lights);
		assert!(lamps.lights[2].is_some(), "and the point after them is still read");
		assert_eq!(lamps.warnings.len(), 2, "one word each: {:?}", lamps.warnings);
		assert!(
			lamps.warnings[0].contains("light 0 (Sun)")
				&& lamps.warnings[0].contains("directional"),
			"the sun named: {:?}",
			lamps.warnings
		);
		assert!(lamps.warnings[1].contains("area"), "and the kind named: {:?}", lamps.warnings);
	}

	#[test]
	fn a_brightness_below_nothing_reaches_nowhere_rather_than_nowhere_a_number_can_say() {
		let lamps = declared("{ \"type\": \"point\", \"intensity\": -5 }");
		let lamp = lamps.lights[0].expect("taken, and shining nothing");

		assert_eq!(lamp.range.to_bits(), 0.0_f32.to_bits(), "no reach, and not a NaN");
		assert!(!lamp.is_lit(), "so it is a light a frame leaves out");
	}

	#[test]
	fn a_file_with_no_lamps_declares_none() {
		let text = "{ \"asset\": { \"version\": \"2.0\" } }";
		let file = Gltf::read(text.as_bytes(), Path::new("bare.gltf"), Path::new(""))
			.expect("the document reads");

		assert_eq!(read(&file), Lamps::default());
	}
}
