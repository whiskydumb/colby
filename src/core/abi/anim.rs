//! Animation clips: the keys that move a skeleton's bones over time.
//!
//! A [`SkeletonData`](super::skeleton::SkeletonData) says what bones there are
//! and a [`Pose`](super::pose::Pose) says where they are now. A clip is the
//! third thing: a recording of where they were, key by key, that something
//! plays into a pose. It holds no skeleton, no pose and no time of its own.
//!
//! **A track names its bone with text.** Not an index, and the reason is a
//! measurement rather than a preference: the same biped exported twice, once
//! with its socket bones and once without, has `bip001-neck` at index seven in
//! one file and index five in the other. Every index below the neck moves.
//! Text is the only thing the two have in common, so a clip authored against
//! one rig plays on the other only if the tracks say what they mean. Turning
//! the text into an index happens once, against the skeleton a clip is
//! actually being played on.
//!
//! **A track is one channel of one bone, not a whole transform.** That is what
//! the exchange format stores, so reading one is a copy rather than a
//! resample, and it is what every engine checked does. A bone whose rotation
//! is animated and whose position is not costs one track, and its position
//! stays wherever the skeleton put it.
//!
//! **The times are the file's own, and they are strictly ascending.** Nothing
//! is resampled onto a fixed rate on the way in: a ten-second idle with four
//! keys stays four keys, `Step` stays exact, and finding the pair of keys a
//! moment sits between is a search over a handful of numbers. The ascending
//! rule is the invariant the search stands on, the way parents-before-children
//! is the one a pose stands on, and it is checked where a clip is read.
//!
//! **A clip does not say how long it is.** [`ClipData::duration`] works it out
//! from the tracks, because a stored length is a second copy of something the
//! keys already say and the two would eventually disagree. It is a walk over
//! the tracks rather than over the keys, and it is not on any path that runs
//! per bone.

/// The most tracks one clip may have.
///
/// Three channels for each of [`MAX_BONES`](super::skeleton::MAX_BONES) bones
/// is seven hundred and sixty-eight, so this is that with room over. Like
/// every other bound here it is a limit on a file rather than a budget.
pub const MAX_TRACKS: usize = 1024;

/// The most keys one clip may hold, counted over all of its tracks.
///
/// A minute of a two-hundred-bone rig at thirty keys a second is about a
/// million and a half, so a clip this big is a file that has gone wrong rather
/// than a clip somebody made.
pub const MAX_KEYS: usize = 1 << 20;

/// Which part of a bone's transform a track writes.
///
/// The three the exchange format has, minus the fourth: morph weights change a
/// mesh's shape rather than where its bones are, and nothing in this engine
/// reads them.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Channel {
	/// Where the bone sits relative to its parent. Three numbers a key.
	#[default]
	Position = 0,

	/// How it is turned there, xyzw. Four numbers a key, and the only channel
	/// interpolated on a sphere rather than a line.
	Rotation = 1,

	/// How big it is, along each axis. Three numbers a key.
	Scale = 2,
}

impl Channel {
	/// How many numbers one key of this channel is.
	#[must_use]
	pub const fn lanes(self) -> usize {
		match self {
			| Self::Position | Self::Scale => 3,
			| Self::Rotation => 4,
		}
	}

	/// What a file stores for this channel.
	#[must_use]
	pub const fn code(self) -> u8 {
		match self {
			| Self::Position => 0,
			| Self::Rotation => 1,
			| Self::Scale => 2,
		}
	}

	/// The channel a file's number stands for.
	///
	/// @param code - what the record held
	/// @return the channel, or `None` when this build does not know it
	#[must_use]
	pub const fn from_code(code: u8) -> Option<Self> {
		match code {
			| 0 => Some(Self::Position),
			| 1 => Some(Self::Rotation),
			| 2 => Some(Self::Scale),
			| _ => None,
		}
	}
}

/// How a track's value moves between two keys.
///
/// The exchange format has a third, a cubic spline with a tangent either side
/// of every key, and colby refuses it by name where it is read rather than
/// quietly reading its middle value as though the curve were straight. A file
/// that was authored with smoothing and plays without it is a change nobody
/// asked for and nobody is told about.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Interpolation {
	/// Straight from one key to the next, on a sphere for a rotation.
	#[default]
	Linear = 0,

	/// The earlier key holds until the later one is reached.
	Step = 1,
}

impl Interpolation {
	/// What a file stores for this rule.
	#[must_use]
	pub const fn code(self) -> u8 {
		match self {
			| Self::Linear => 0,
			| Self::Step => 1,
		}
	}

	/// The rule a file's number stands for.
	///
	/// @param code - what the record held
	/// @return the rule, or `None` when this build does not know it
	#[must_use]
	pub const fn from_code(code: u8) -> Option<Self> {
		match code {
			| 0 => Some(Self::Linear),
			| 1 => Some(Self::Step),
			| _ => None,
		}
	}
}

/// One channel of one bone over time.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Track {
	/// The bone this moves, as the skeleton calls it.
	pub bone: String,

	/// Which part of that bone's transform it writes.
	pub channel: Channel,

	/// How it moves between two of its keys.
	pub interpolation: Interpolation,

	/// When each key is, in seconds, strictly ascending.
	pub times: Vec<f32>,

	/// The keys themselves, [`Channel::lanes`] numbers each, in the same
	/// order as [`times`](Self::times).
	pub values: Vec<f32>,
}

impl Track {
	/// How many keys it has.
	#[must_use]
	pub fn keys(&self) -> usize { self.times.len() }

	/// How many numbers one of its keys is.
	#[must_use]
	pub const fn lanes(&self) -> usize { self.channel.lanes() }

	/// When its last key is, or zero when it has none.
	///
	/// What [`ClipData::duration`] is the largest of.
	#[must_use]
	pub fn end(&self) -> f32 { self.times.last().copied().unwrap_or(0.0) }

	/// Whether its times ascend and are numbers at all.
	///
	/// The invariant everything that samples one stands on: finding the pair
	/// of keys a moment sits between is only a search if the times are in
	/// order, and dividing by the gap between two of them is only safe if no
	/// gap is zero. A file that does not pass this is refused rather than read
	/// and sampled into something nobody can explain.
	#[must_use]
	pub fn is_ordered(&self) -> bool {
		self.times.iter().all(|time| time.is_finite())
			&& self
				.times
				.windows(2)
				.all(|pair| pair[0] < pair[1])
	}

	/// Whether it holds exactly the numbers its keys and channel imply.
	///
	/// A track with no keys is not whole either: it says a bone is animated
	/// and then declines to say how.
	#[must_use]
	pub fn is_whole(&self) -> bool {
		!self.times.is_empty() && self.values.len() == self.keys().saturating_mul(self.lanes())
	}
}

/// One animation, as the world holds it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClipData {
	/// Every track, in no particular order.
	///
	/// Two tracks may name the same bone: one turns it and another moves it.
	/// Two writing the same channel of the same bone is a file contradicting
	/// itself, and the later one wins for the same reason the later of two
	/// registrations under one name does.
	pub tracks: Vec<Track>,
}

impl ClipData {
	/// How long it runs, in seconds.
	///
	/// The last key of whichever track ends latest, and never less than zero.
	/// Worked out rather than stored: a length beside the keys is a second
	/// copy of what the keys already say, and the two disagree the first time
	/// somebody edits one of them.
	#[must_use]
	pub fn duration(&self) -> f32 {
		self.tracks
			.iter()
			.map(Track::end)
			.fold(0.0_f32, f32::max)
	}

	/// How many tracks it has.
	#[must_use]
	pub fn len(&self) -> usize { self.tracks.len() }

	/// Whether it has none, which is what the null clip is.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.tracks.is_empty() }

	/// How many keys it holds over all of its tracks.
	#[must_use]
	pub fn keys(&self) -> usize { self.tracks.iter().map(Track::keys).sum() }

	/// Whether every track's times ascend. @ref [`Track::is_ordered`].
	#[must_use]
	pub fn is_ordered(&self) -> bool { self.tracks.iter().all(Track::is_ordered) }

	/// Whether every track holds the numbers it implies. @ref
	/// [`Track::is_whole`].
	#[must_use]
	pub fn is_whole(&self) -> bool { self.tracks.iter().all(Track::is_whole) }
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A track of three keys on one channel.
	fn track(channel: Channel, times: &[f32]) -> Track {
		let lanes = channel.lanes();

		Track {
			bone: "spine".to_owned(),
			channel,
			interpolation: Interpolation::Linear,
			times: times.to_vec(),
			values: vec![0.0; times.len() * lanes],
		}
	}

	#[test]
	fn a_rotation_is_four_numbers_and_the_others_are_three() {
		assert_eq!(Channel::Rotation.lanes(), 4, "a quaternion");
		assert_eq!(Channel::Position.lanes(), 3, "a point");
		assert_eq!(Channel::Scale.lanes(), 3, "a size along each axis");
	}

	#[test]
	fn every_channel_and_rule_survives_the_trip_through_a_file() {
		for channel in [Channel::Position, Channel::Rotation, Channel::Scale] {
			assert_eq!(
				Channel::from_code(channel.code()),
				Some(channel),
				"{channel:?} is written and read back as itself"
			);
		}

		for rule in [Interpolation::Linear, Interpolation::Step] {
			assert_eq!(
				Interpolation::from_code(rule.code()),
				Some(rule),
				"{rule:?} is written and read back as itself"
			);
		}

		assert_eq!(Channel::from_code(3), None, "and a fourth channel is not known");
		assert_eq!(Interpolation::from_code(2), None, "nor a third rule");
	}

	#[test]
	fn times_that_do_not_ascend_are_refused_and_ones_that_do_are_not() {
		assert!(track(Channel::Position, &[0.0, 0.5, 1.0]).is_ordered(), "ascending");
		assert!(!track(Channel::Position, &[0.0, 1.0, 0.5]).is_ordered(), "out of order");
		assert!(!track(Channel::Position, &[0.0, 0.5, 0.5]).is_ordered(), "a zero-long gap");
		assert!(!track(Channel::Position, &[0.0, f32::NAN]).is_ordered(), "not a number");
		assert!(
			!track(Channel::Position, &[f32::NAN]).is_ordered(),
			"and a single key at no moment at all, which comparing pairs cannot catch because a \
			 lone key is not a pair"
		);
		assert!(track(Channel::Position, &[]).is_ordered(), "and nothing is in order");
	}

	#[test]
	fn a_track_holds_as_many_numbers_as_its_keys_and_channel_say() {
		assert!(track(Channel::Rotation, &[0.0, 1.0]).is_whole(), "two keys of four");
		assert!(!track(Channel::Position, &[]).is_whole(), "and no keys at all is not whole");

		let mut short = track(Channel::Rotation, &[0.0, 1.0]);
		short.values.pop();

		assert!(!short.is_whole(), "seven numbers for two rotations is not two rotations");

		let mut widened = track(Channel::Position, &[0.0, 1.0]);
		widened.channel = Channel::Rotation;

		assert!(
			!widened.is_whole(),
			"and six numbers read as rotations is not two of them either"
		);
	}

	#[test]
	fn a_clip_is_as_long_as_its_latest_last_key() {
		let clip = ClipData {
			tracks: vec![
				track(Channel::Position, &[0.0, 0.25]),
				track(Channel::Rotation, &[0.0, 0.5, 1.75]),
				track(Channel::Scale, &[0.0, 1.0]),
			],
		};

		assert!(
			(clip.duration() - 1.75).abs() < f32::EPSILON,
			"the rotation runs longest, so the clip is as long as it is"
		);
		assert_eq!(clip.len(), 3, "three tracks");
		assert_eq!(clip.keys(), 7, "and seven keys between them");
		assert!(clip.is_ordered() && clip.is_whole(), "all of it sound");
	}

	#[test]
	fn a_clip_with_no_tracks_is_empty_and_lasts_no_time() {
		let clip = ClipData::default();

		assert!(clip.is_empty(), "nothing in it");
		assert!(
			(clip.duration() - 0.0).abs() < f32::EPSILON,
			"and a clip of nothing does not run backwards"
		);
	}
}
