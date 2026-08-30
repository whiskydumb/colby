//! Animations: the file's channels and samplers, turned into clips.
//!
//! The exchange format keeps an animation as two lists that point at each
//! other. A *sampler* is a pair of accessors - when the keys are, and what
//! they hold - plus a rule for moving between them. A *channel* says which
//! node's translation, rotation or scale one sampler drives. A clip here is
//! neither: it is a flat list of tracks, each naming a bone with text.
//!
//! Three things happen on the way, and each is a place the format and colby
//! disagree:
//!
//! - **a channel names a node and a track names a bone.** The node's name is
//!   not used: the skins were read first and they already decided what each
//!   joint is called, numbering a collision if there was one, so the name is
//!   taken from there. A channel pointing at a node no skin claims is an
//!   animated object rather than an animated bone, and colby has nowhere to put
//!   one.
//! - **CUBICSPLINE is refused by name**, like a sparse accessor. The middle of
//!   its three values per key is the key itself and reading only that would
//!   play the animation without the smoothing it was authored with - a change
//!   in what the file means, made silently.
//! - **keys that do not advance are dropped.** Sampling searches the times, so
//!   they have to ascend; a file with two keys at one moment gets the later of
//!   them and a count in a warning.
//!
//! Everything here reports rather than refuses. A clip that cannot be read is
//! left out and says so once; the rest of the file still imports.

use colby_core::{
	abi::anim::{Channel, ClipData, Interpolation, MAX_KEYS, MAX_TRACKS, Track},
	glam::Quat,
};

use super::{Gltf, Skin, tidy, unique};
use crate::json::Value;

/// What one glTF animation becomes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Clip {
	/// What it registers under, inside the model's own name.
	pub name: String,

	/// Its tracks. Empty if it could not be read.
	pub data: ClipData,
}

/// Every animation of one file, in the file's own order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Clips {
	/// One per entry of the document's `animations`. One that could not be
	/// read is here with no tracks in it.
	pub clips: Vec<Clip>,

	/// What could not be used. Not a failure.
	pub warnings: Vec<String>,
}

/// Reads every animation a file declares.
///
/// @param file - the document with its buffers, from [`Gltf::open`]
/// @param skins - the skins already read, which is what names the bones
/// @return one clip per animation, and what could not be read
#[must_use]
pub(super) fn read(file: &Gltf, skins: &[Skin]) -> Clips {
	let declared = file.table("animations").len();

	if declared == 0 {
		return Clips::default();
	}

	let bones = bone_names(file, skins);
	let mut taken = Vec::with_capacity(declared);
	let mut out = Clips::default();

	for index in 0..declared {
		let clip = one(file, index, &bones, &mut taken, &mut out.warnings);

		out.clips.push(clip);
	}

	out
}

/// What each node is called as a bone, over the whole document.
///
/// Taken from the skins rather than from the nodes, because a skin is what
/// turned a node into a bone and what settled its name. A node two skins both
/// claim keeps the first one's name, which is the same rule a mesh stood by
/// two skins follows.
fn bone_names(file: &Gltf, skins: &[Skin]) -> Vec<Option<String>> {
	let mut out = vec![None; file.table("nodes").len()];

	for (index, rig) in skins.iter().enumerate() {
		let joints = file
			.table("skins")
			.get(index)
			.and_then(|skin| skin.get("joints"))
			.map_or(&[][..], Value::as_array);

		for (slot, node) in joints
			.iter()
			.filter_map(Value::as_usize)
			.enumerate()
		{
			let Some(cell) = out.get_mut(node) else {
				continue;
			};

			if cell.is_some() {
				continue;
			}

			*cell = rig
				.slots
				.get(slot)
				.and_then(|bone| rig.data.bones.get(usize::from(*bone)))
				.map(|bone| bone.name.clone());
		}
	}

	out
}

/// Reads one animation, or says why it has no tracks.
fn one(
	file: &Gltf,
	index: usize,
	bones: &[Option<String>],
	taken: &mut Vec<String>,
	warnings: &mut Vec<String>,
) -> Clip {
	let entry = file.table("animations").get(index).cloned();
	let written = entry
		.as_ref()
		.and_then(|animation| animation.get("name"))
		.and_then(Value::as_str)
		.unwrap_or("");
	let mut base = tidy(written);

	if base.is_empty() {
		base = format!("clip{index}");
	}

	let name = unique(taken, &base);
	let channels = entry
		.as_ref()
		.and_then(|animation| animation.get("channels"))
		.map_or(&[][..], Value::as_array)
		.to_vec();
	let samplers = entry
		.as_ref()
		.and_then(|animation| animation.get("samplers"))
		.map_or(&[][..], Value::as_array)
		.to_vec();

	if channels.is_empty() {
		warnings.push(format!("animation {index} has no channels, and moves nothing"));

		return Clip { name, ..Clip::default() };
	}

	let mut tracks = Vec::with_capacity(channels.len());
	let mut objects = 0_usize;

	for (slot, channel) in channels.iter().enumerate() {
		match track(file, index, slot, channel, &samplers, bones, warnings) {
			| Read::Track(one) => tracks.push(one),
			| Read::Object => objects += 1,
			| Read::Nothing => (),
		}
	}

	if objects > 0 {
		warnings.push(format!(
			"animation {index} moves {objects} things that are not bones, and colby animates \
			 nothing else yet"
		));
	}

	let data = ClipData { tracks };

	if data.is_empty() {
		warnings.push(format!("animation {index} moves no bone colby knows, and is left out"));

		return Clip { name, ..Clip::default() };
	}

	if data.len() > MAX_TRACKS {
		warnings.push(format!(
			"animation {index} has {} tracks, past the {MAX_TRACKS} a clip may hold, and is \
			 left out",
			data.len()
		));

		return Clip { name, ..Clip::default() };
	}

	if data.keys() > MAX_KEYS {
		warnings.push(format!(
			"animation {index} has {} keys, past the {MAX_KEYS} a clip may hold, and is left out",
			data.keys()
		));

		return Clip { name, ..Clip::default() };
	}

	Clip { name, data }
}

/// What one channel turned out to be.
enum Read {
	/// A bone moving.
	Track(Track),

	/// Something that is not a bone moving.
	Object,

	/// Nothing usable, and a warning has already been written about it.
	Nothing,
}

/// Reads one channel into a track.
fn track(
	file: &Gltf,
	animation: usize,
	slot: usize,
	channel: &Value,
	samplers: &[Value],
	bones: &[Option<String>],
	warnings: &mut Vec<String>,
) -> Read {
	let target = channel.get("target");
	let path = target
		.and_then(|target| target.get("path"))
		.and_then(Value::as_str)
		.unwrap_or_default();
	let Some(part) = part_of(path) else {
		if path == "weights" {
			warnings.push(format!(
				"animation {animation} channel {slot} morphs a mesh into another shape, which \
				 colby does not draw"
			));
		} else {
			warnings.push(format!(
				"animation {animation} channel {slot} drives \"{path}\", which colby does not \
				 animate"
			));
		}

		return Read::Nothing;
	};
	let node = target
		.and_then(|target| target.get("node"))
		.and_then(Value::as_usize);
	let Some(bone) = node
		.and_then(|node| bones.get(node))
		.and_then(Option::as_ref)
	else {
		return Read::Object;
	};
	let Some(sampler) = channel
		.get("sampler")
		.and_then(Value::as_usize)
		.and_then(|index| samplers.get(index))
	else {
		warnings.push(format!(
			"animation {animation} channel {slot} names a sampler that is not there"
		));

		return Read::Nothing;
	};
	let rule = sampler
		.get("interpolation")
		.and_then(Value::as_str)
		.unwrap_or("LINEAR");
	let Some(interpolation) = rule_of(rule) else {
		warnings.push(format!(
			"animation {animation} channel {slot} is interpolated by {rule}, which colby does \
			 not read; re-export it as LINEAR or STEP"
		));

		return Read::Nothing;
	};

	let (Some(input), Some(output)) = (
		sampler.get("input").and_then(Value::as_usize),
		sampler.get("output").and_then(Value::as_usize),
	) else {
		warnings
			.push(format!("animation {animation} channel {slot} has a sampler naming no keys"));

		return Read::Nothing;
	};

	keys(file, (animation, slot), (input, output), (part, interpolation), warnings).map_or(
		Read::Nothing,
		|(times, values)| {
			Read::Track(Track {
				bone: bone.clone(),
				channel: part,
				interpolation,
				times,
				values,
			})
		},
	)
}

/// Reads one sampler's two accessors into keys colby can search.
fn keys(
	file: &Gltf,
	at: (usize, usize),
	accessors: (usize, usize),
	how: (Channel, Interpolation),
	warnings: &mut Vec<String>,
) -> Option<(Vec<f32>, Vec<f32>)> {
	let (animation, slot) = at;
	let (channel, _) = how;
	let times = match file.floats(accessors.0) {
		| Ok(read) => read,
		| Err(error) => {
			warnings.push(format!(
				"animation {animation} channel {slot} has key times that could not be read \
				 ({error})"
			));

			return None;
		},
	};
	let values = match file.floats(accessors.1) {
		| Ok(read) => read,
		| Err(error) => {
			warnings.push(format!(
				"animation {animation} channel {slot} has keys that could not be read ({error})"
			));

			return None;
		},
	};

	if times.lanes() != 1 {
		warnings.push(format!(
			"animation {animation} channel {slot} is keyed at times of {} numbers each, and a \
			 moment is one",
			times.lanes()
		));

		return None;
	}

	if values.lanes() != channel.lanes() {
		warnings.push(format!(
			"animation {animation} channel {slot} drives {channel:?} with values of {} numbers \
			 each, and it takes {}",
			values.lanes(),
			channel.lanes()
		));

		return None;
	}

	if values.rows() != times.rows() {
		warnings.push(format!(
			"animation {animation} channel {slot} has {} keys at {} moments",
			values.rows(),
			times.rows()
		));

		return None;
	}

	if times.rows() == 0 {
		warnings.push(format!("animation {animation} channel {slot} has no keys at all"));

		return None;
	}

	Some(ascending(file_keys(&times, &values, channel), at, warnings))
}

/// The keys as the file holds them, with rotations made unit length.
///
/// The specification says a rotation is a unit quaternion, and a file storing
/// one quantized into integers only approximately is. What sampling does with
/// a quaternion that is nearly but not quite unit is turn the bone slightly
/// wrong and scale whatever hangs off it, so it is made unit here rather than
/// per frame.
fn file_keys(
	times: &super::Floats,
	values: &super::Floats,
	channel: Channel,
) -> (Vec<f32>, Vec<f32>) {
	let mut out = values.values().to_vec();

	if channel == Channel::Rotation {
		for key in out.chunks_exact_mut(4) {
			let turn = Quat::from_xyzw(key[0], key[1], key[2], key[3]);
			let unit = if turn.length_squared() > f32::EPSILON {
				turn.normalize()
			} else {
				Quat::IDENTITY
			};

			key.copy_from_slice(&unit.to_array());
		}
	}

	(times.values().to_vec(), out)
}

/// The same keys with every one that does not advance the clock dropped.
///
/// Sampling searches the times, so they have to ascend and no two may share a
/// moment. Whichever key was written later at a repeated moment is the one
/// kept, because that is what a file overwriting a key means.
fn ascending(
	keys: (Vec<f32>, Vec<f32>),
	at: (usize, usize),
	warnings: &mut Vec<String>,
) -> (Vec<f32>, Vec<f32>) {
	let (times, values) = keys;

	if times.windows(2).all(|pair| pair[0] < pair[1]) && times.iter().all(|time| time.is_finite())
	{
		return (times, values);
	}

	let lanes = values.len() / times.len().max(1);
	let mut kept: Vec<usize> = Vec::with_capacity(times.len());

	// backwards, so that of two keys sharing a moment the later one is the one
	// that survives - which is what a file overwriting a key means.
	for (index, time) in times.iter().enumerate().rev() {
		let later = kept.last().and_then(|index| times.get(*index));

		if time.is_finite() && later.is_none_or(|later| *time < *later) {
			kept.push(index);
		}
	}

	kept.reverse();

	let (animation, slot) = at;

	warnings.push(format!(
		"animation {animation} channel {slot} has {} keys that do not advance its clock, and \
		 they are dropped",
		times.len() - kept.len()
	));

	(
		kept.iter()
			.filter_map(|index| times.get(*index).copied())
			.collect(),
		kept.iter()
			.flat_map(|index| {
				values
					.get(index * lanes..(index + 1) * lanes)
					.unwrap_or_default()
			})
			.copied()
			.collect(),
	)
}

/// Which part of a transform a target path drives.
fn part_of(path: &str) -> Option<Channel> {
	match path {
		| "translation" => Some(Channel::Position),
		| "rotation" => Some(Channel::Rotation),
		| "scale" => Some(Channel::Scale),
		| _ => None,
	}
}

/// Which rule a sampler's interpolation names.
fn rule_of(rule: &str) -> Option<Interpolation> {
	match rule {
		| "LINEAR" => Some(Interpolation::Linear),
		| "STEP" => Some(Interpolation::Step),
		| _ => None,
	}
}

#[cfg(test)]
mod tests {
	use std::path::Path;

	use super::{super::skin, *};

	/// Ninety-six bytes: three key times, then three rotations, then three
	/// positions. Written by an outside tool and pasted in, like every other
	/// binary fixture here.
	const KEYS: &str = "AAAAAAAAgD8AAABAAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAA8wQ1P/\
	                    MENT8AAAAAAAAAAAAAAAAAAIA/\
	                    AAAAAAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAEAAAAAA";

	/// Sixty bytes: three times, then three turns none of which is unit length,
	/// one of them nothing at all.
	const CROOKED: &str =
		"AAAAAAAAgD8AAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQEAAAEBA";

	/// Two views and accessors over the [`CROOKED`] buffer.
	const TURNS: &str = "\"bufferViews\": [ { \"buffer\": 0, \"byteOffset\": 0, \"byteLength\": \
	                     12 }, { \"buffer\": 0, \"byteOffset\": 12, \"byteLength\": 48 } ], \
	                     \"accessors\": [ { \"bufferView\": 0, \"componentType\": 5126, \
	                     \"count\": 3, \"type\": \"SCALAR\" }, { \"bufferView\": 1, \
	                     \"componentType\": 5126, \"count\": 3, \"type\": \"VEC4\" } ]";

	/// Forty-eight bytes: three times of which the last two are one moment,
	/// then three positions a unit apart.
	const REPEATED: &str = "AAAAAAAAgD8AAIA/AACAPwAAAAAAAAAAAAAAQAAAAAAAAAAAAABAQAAAAAAAAAAA";

	/// The three views and accessors the [`KEYS`] buffer holds.
	const SPREAD: &str =
		"\"bufferViews\": [ { \"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 12 }, { \
		 \"buffer\": 0, \"byteOffset\": 12, \"byteLength\": 48 }, { \"buffer\": 0, \
		 \"byteOffset\": 60, \"byteLength\": 36 } ], \"accessors\": [ { \"bufferView\": 0, \
		 \"componentType\": 5126, \"count\": 3, \"type\": \"SCALAR\" }, { \"bufferView\": 1, \
		 \"componentType\": 5126, \"count\": 3, \"type\": \"VEC4\" }, { \"bufferView\": 2, \
		 \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\" } ]";

	/// Two views and accessors over the [`REPEATED`] buffer.
	const DOUBLED: &str =
		"\"bufferViews\": [ { \"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 12 }, { \
		 \"buffer\": 0, \"byteOffset\": 12, \"byteLength\": 36 } ], \"accessors\": [ { \
		 \"bufferView\": 0, \"componentType\": 5126, \"count\": 3, \"type\": \"SCALAR\" }, { \
		 \"bufferView\": 1, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\" } ]";

	/// The usual pair of channels: the spine turns and the hips travel.
	const BOTH: &str = "{ \"name\": \"Walk Cycle\", \"samplers\": [ { \"input\": 0, \"output\": \
	                    1, \"interpolation\": \"LINEAR\" }, { \"input\": 0, \"output\": 2, \
	                    \"interpolation\": \"STEP\" } ], \"channels\": [ { \"sampler\": 0, \
	                    \"target\": { \"node\": 1, \"path\": \"rotation\" } }, { \"sampler\": \
	                    1, \"target\": { \"node\": 0, \"path\": \"translation\" } } ] }";

	/// A document with a two-bone rig, a spare node that is no bone, and
	/// whatever animation the caller writes.
	fn around(buffer: &str, length: usize, views: &str, animation: &str) -> String {
		format!(
			"{{ \"asset\": {{ \"version\": \"2.0\" }}, \"buffers\": [ {{ \"byteLength\": \
			 {length}, \"uri\": \"data:application/octet-stream;base64,{buffer}\" }} ], \
			 {views}, \"nodes\": [ {{ \"name\": \"hips\", \"children\": [1] }}, {{ \"name\": \
			 \"spine\" }}, {{ \"name\": \"lamp\" }} ], \"skins\": [ {{ \"name\": \"rig\", \
			 \"joints\": [1, 0] }} ], \"animations\": [ {animation} ] }}"
		)
	}

	/// One channel, with whatever the caller wants said about it.
	fn one_channel(sampler: &str, target: &str) -> String {
		format!(
			"{{ \"name\": \"take\", \"samplers\": [ {sampler} ], \"channels\": [ {{ \
			 \"sampler\": 0, \"target\": {target} }} ] }}"
		)
	}

	/// Reads a document written as text.
	fn read_text(text: &str) -> Clips {
		let file = Gltf::read(text.as_bytes(), Path::new("clip.gltf"), Path::new(""))
			.expect("the document reads");
		let skins = skin::read(&file);

		read(&file, &skins.skins)
	}

	#[test]
	fn an_animation_becomes_a_clip_of_tracks_that_name_bones() {
		let clips = read_text(&around(KEYS, 96, SPREAD, BOTH));
		let clip = &clips.clips[0];

		assert_eq!(clips.warnings, Vec::<String>::new(), "nothing is wrong with it");
		assert_eq!(clip.name, "walk_cycle", "tidied into something a file can be called");
		assert_eq!(clip.data.len(), 2, "two channels, two tracks");

		let turn = &clip.data.tracks[0];

		assert_eq!(turn.bone, "spine", "the bone the skin named, not the node index");
		assert_eq!(turn.channel, Channel::Rotation);
		assert_eq!(turn.interpolation, Interpolation::Linear);
		assert_eq!(turn.keys(), 3, "three keys");
		assert_eq!(turn.times, vec![0.0, 1.0, 2.0], "at the moments the file gave");

		let travel = &clip.data.tracks[1];

		assert_eq!(travel.bone, "hips");
		assert_eq!(travel.channel, Channel::Position);
		assert_eq!(travel.interpolation, Interpolation::Step, "the sampler said so");
		assert_eq!(travel.values, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0]);
		assert!(
			(clip.data.duration() - 2.0).abs() < 1.0e-6,
			"and it runs as long as its last key"
		);
	}

	#[test]
	fn a_track_is_named_by_the_skin_rather_than_by_the_node_it_points_at() {
		// the skin lists the spine first, so the sort renumbers both bones and
		// a track carrying an index would name the wrong one.
		let clips = read_text(&around(KEYS, 96, SPREAD, BOTH));

		assert_eq!(
			clips.clips[0]
				.data
				.tracks
				.iter()
				.map(|track| track.bone.as_str())
				.collect::<Vec<_>>(),
			vec!["spine", "hips"],
			"node one is the spine whatever slot the skin gave it"
		);
	}

	#[test]
	fn a_channel_driving_something_that_is_no_bone_is_counted_rather_than_read() {
		let animation = one_channel(
			"{ \"input\": 0, \"output\": 2, \"interpolation\": \"LINEAR\" }",
			"{ \"node\": 2, \"path\": \"translation\" }",
		);
		let clips = read_text(&around(KEYS, 96, SPREAD, &animation));

		assert!(
			clips
				.warnings
				.iter()
				.any(|said| said.contains("1 things that are not bones")),
			"the lamp is counted: {:?}",
			clips.warnings
		);
		assert!(clips.clips[0].data.is_empty(), "and the clip moves nothing");
	}

	#[test]
	fn a_cubic_spline_is_refused_by_name_rather_than_read_as_a_straight_line() {
		let animation = one_channel(
			"{ \"input\": 0, \"output\": 1, \"interpolation\": \"CUBICSPLINE\" }",
			"{ \"node\": 1, \"path\": \"rotation\" }",
		);
		let clips = read_text(&around(KEYS, 96, SPREAD, &animation));

		assert!(
			clips
				.warnings
				.iter()
				.any(|said| said.contains("CUBICSPLINE")),
			"said which rule it could not read: {:?}",
			clips.warnings
		);
		assert!(clips.clips[0].data.is_empty(), "and read none of it");
	}

	#[test]
	fn morphing_a_mesh_into_another_shape_is_named_and_left() {
		let animation = one_channel(
			"{ \"input\": 0, \"output\": 0, \"interpolation\": \"LINEAR\" }",
			"{ \"node\": 1, \"path\": \"weights\" }",
		);
		let clips = read_text(&around(KEYS, 96, SPREAD, &animation));

		assert!(
			clips
				.warnings
				.iter()
				.any(|said| said.contains("morphs a mesh")),
			"morph weights are worth a word: {:?}",
			clips.warnings
		);
	}

	#[test]
	fn keys_that_do_not_advance_the_clock_are_dropped_and_the_later_one_wins() {
		let animation = one_channel(
			"{ \"input\": 0, \"output\": 1, \"interpolation\": \"LINEAR\" }",
			"{ \"node\": 1, \"path\": \"translation\" }",
		);
		let clips = read_text(&around(REPEATED, 48, DOUBLED, &animation));
		let track = &clips.clips[0].data.tracks[0];

		assert!(
			clips
				.warnings
				.iter()
				.any(|said| said.contains("1 keys that do not advance")),
			"said how many it dropped: {:?}",
			clips.warnings
		);
		assert_eq!(track.times, vec![0.0, 1.0], "two moments out of three keys");
		assert_eq!(
			track.values,
			vec![1.0, 0.0, 0.0, 3.0, 0.0, 0.0],
			"and the second of the two keys at one moment is the one kept"
		);
		assert!(track.is_ordered() && track.is_whole(), "which leaves it sound");
	}

	#[test]
	fn a_rotation_that_is_not_unit_length_is_made_one() {
		// the fixture's turns are two, nothing at all, and one of length
		// four and a quarter, so a reader that passed them through would be
		// visible here. The specification says a rotation is a unit
		// quaternion; a file quantizing one into integers only nearly is.
		let animation = one_channel(
			"{ \"input\": 0, \"output\": 1 }",
			"{ \"node\": 1, \"path\": \"rotation\" }",
		);
		let clips = read_text(&around(CROOKED, 60, TURNS, &animation));

		for key in clips.clips[0].data.tracks[0]
			.values
			.chunks_exact(4)
		{
			let length: f32 = key.iter().map(|value| value * value).sum();

			assert!((length - 1.0).abs() < 1.0e-6, "{key:?} is a turn and nothing else");
		}

		assert_eq!(
			clips.clips[0].data.tracks[0].values[4..8],
			[0.0, 0.0, 0.0, 1.0],
			"and a turn of no length at all is no turn rather than a hole in the pose"
		);
	}

	#[test]
	fn an_animation_with_no_channels_moves_nothing_and_says_so() {
		let clips = read_text(&around(
			KEYS,
			96,
			SPREAD,
			"{ \"name\": \"empty\", \"samplers\": [], \"channels\": [] }",
		));

		assert!(
			clips
				.warnings
				.iter()
				.any(|said| said.contains("no channels")),
			"{:?}",
			clips.warnings
		);
		assert!(clips.clips[0].data.is_empty());
	}

	#[test]
	fn a_sampler_whose_values_are_the_wrong_width_is_left_out() {
		// three numbers a key read as a rotation, which takes four.
		let animation = one_channel(
			"{ \"input\": 0, \"output\": 2, \"interpolation\": \"LINEAR\" }",
			"{ \"node\": 1, \"path\": \"rotation\" }",
		);
		let clips = read_text(&around(KEYS, 96, SPREAD, &animation));

		assert!(
			clips
				.warnings
				.iter()
				.any(|said| said.contains("it takes 4")),
			"said what it wanted: {:?}",
			clips.warnings
		);
		assert!(clips.clips[0].data.is_empty());
	}

	#[test]
	fn an_animation_with_no_name_is_numbered_and_two_alike_are_told_apart() {
		let both = format!("{BOTH}, {BOTH}");
		let clips = read_text(&around(KEYS, 96, SPREAD, &both));

		assert_eq!(clips.clips.len(), 2, "both are read");
		assert_eq!(clips.clips[0].name, "walk_cycle");
		assert_eq!(clips.clips[1].name, "walk_cycle_1", "and the second is told apart");

		let unnamed = read_text(&around(
			KEYS,
			96,
			SPREAD,
			"{ \"samplers\": [ { \"input\": 0, \"output\": 1 } ], \"channels\": [ { \
			 \"sampler\": 0, \"target\": { \"node\": 1, \"path\": \"rotation\" } } ] }",
		));

		assert_eq!(unnamed.clips[0].name, "clip0", "named after where it stands");
		assert_eq!(
			unnamed.clips[0].data.tracks[0].interpolation,
			Interpolation::Linear,
			"and a sampler that says nothing is a straight line, as the format has it"
		);
	}

	#[test]
	fn a_file_with_no_animations_at_all_reads_as_none() {
		let file = Gltf::read(
			b"{ \"asset\": { \"version\": \"2.0\" } }",
			Path::new("clip.gltf"),
			Path::new(""),
		)
		.expect("the document reads");

		assert_eq!(read(&file, &[]), Clips::default(), "nothing, and no complaints");
	}
}
