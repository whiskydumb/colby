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

use super::{
	entity::Transform,
	registry::{Entry, Registry},
	skeleton::{SkeletonData, SkeletonId, Skeletons},
};
use crate::{
	glam::{Quat, Vec3},
	registry_handle,
};

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

/// What a track's bone resolved to when the skeleton has no such bone.
///
/// A sentinel rather than an `Option` because a binding is one number per
/// track and there are a great many of them, and because a track pointing at
/// nothing is the ordinary case rather than an error: a walk cycle authored on
/// a rig with fingers, played on one without, has ten tracks that land here and
/// is otherwise perfectly good.
pub const NO_BONE: u16 = u16::MAX;

registry_handle! {
	/// A handle to a clip in the world's [`Clips`] registry.
	///
	/// No generation, like every other asset handle: entries are never
	/// removed, so recompiling a clip rewrites the entry the id already points
	/// at and a game holding one does not re-resolve.
	ClipId
}

/// Reads one key of a track as three numbers.
fn triple(key: &[f32]) -> Vec3 {
	Vec3::new(
		key.first().copied().unwrap_or(0.0),
		key.get(1).copied().unwrap_or(0.0),
		key.get(2).copied().unwrap_or(0.0),
	)
}

/// Reads one key of a track as a turn.
fn turn(key: &[f32]) -> Quat {
	Quat::from_xyzw(
		key.first().copied().unwrap_or(0.0),
		key.get(1).copied().unwrap_or(0.0),
		key.get(2).copied().unwrap_or(0.0),
		key.get(3).copied().unwrap_or(1.0),
	)
}

impl Track {
	/// One key, as the numbers the channel takes.
	#[must_use]
	pub fn key(&self, index: usize) -> &[f32] {
		let lanes = self.lanes();
		let at = index.saturating_mul(lanes);

		self.values
			.get(at..at.saturating_add(lanes))
			.unwrap_or_default()
	}

	/// Which pair of keys a moment sits between, and how far along it is.
	///
	/// Outside the keys the nearer end holds rather than the value carrying
	/// on, which is what the exchange format says: before and after the range
	/// the output is clamped to the nearest end of it. That is also what makes
	/// a clip whose tracks are different lengths behave: the short ones stop
	/// and hold instead of running off.
	fn span(&self, time: f32) -> (usize, usize, f32) {
		// the predicate is `at or before`, so the search lands past every key
		// sharing a moment and the pair it picks is never two of those.
		let after = self.times.partition_point(|key| *key <= time);
		let Some(before) = after.checked_sub(1) else {
			// before the first key, which holds.
			return (0, 0, 0.0);
		};
		let (Some(earlier), Some(later)) = (self.times.get(before), self.times.get(after)) else {
			// past the last key, which holds.
			return (before, before, 0.0);
		};
		let gap = later - earlier;

		// @note: while the predicate above is `at or before`, the search lands
		// past every key sharing a moment, so the two it picks never share one
		// and this gap is never zero - which makes the guard unreachable and
		// untested rather than decoration. It is what a division by nothing
		// would otherwise cost, and the predicate is one character away from
		// making it reachable.
		let along = if gap > 0.0 { (time - earlier) / gap } else { 0.0 };

		(before, after, along)
	}

	/// Writes what this track says at a moment into a transform.
	///
	/// Only the channel it drives: a track that turns a bone leaves where the
	/// bone is alone, so what a clip does not say stays whatever the caller
	/// put there. @ref [`ClipData::sample`], which is what puts the rest there.
	///
	/// @param time - a moment inside the clip, in seconds
	/// @param into - the bone's local transform, written in place
	pub fn apply(&self, time: f32, into: &mut Transform) {
		let (before, after, along) = self.span(time);
		let earlier = self.key(before);

		if earlier.is_empty() {
			return;
		}

		let later = self.key(after);
		let blend = match self.interpolation {
			| Interpolation::Linear => along,
			// the earlier key holds until the later one is reached, which is
			// the whole of what the rule means.
			| Interpolation::Step => 0.0,
		};

		match self.channel {
			| Channel::Position => into.position = triple(earlier).lerp(triple(later), blend),
			| Channel::Scale => into.scale = triple(earlier).lerp(triple(later), blend),
			// on a sphere rather than a line, which the format asks for by
			// name. The shortest way round is glam's: it flips the far end
			// when the two point into opposite halves, so nothing here has to.
			| Channel::Rotation => into.rotation = turn(earlier).slerp(turn(later), blend),
		}
	}
}

impl ClipData {
	/// The moment inside the clip that a time on somebody's own clock lands on.
	///
	/// Looping is a property of the playing rather than of the file, which is
	/// what three of the four engines checked do and what the exchange format
	/// implies by saying nothing: a walk is a walk whether something plays it
	/// once or forever.
	///
	/// @param time - seconds on whatever clock the caller keeps
	/// @param looping - whether it starts again rather than holding its end
	/// @return a moment inside `0 ..= duration`
	#[must_use]
	pub fn moment(&self, time: f32, looping: bool) -> f32 {
		let length = self.duration();

		if length <= 0.0 {
			return 0.0;
		}

		if looping {
			time.rem_euclid(length)
		} else {
			time.clamp(0.0, length)
		}
	}

	/// Writes the clip at a moment over a skeleton's local transforms.
	///
	/// **What comes in has to be the pose the clip is played over**, normally
	/// the skeleton at rest: a clip names only the bones it moves, so
	/// everything it says nothing about is left exactly as it was found. That
	/// is what makes a sampled pose complete, and a complete pose is what
	/// makes blending two of them mean anything.
	///
	/// @param time - seconds on the caller's clock, wrapped by
	/// [`moment`](Self::moment)
	/// @param looping - whether the clip starts again rather than holding
	/// @param bones - which bone each track moves, from [`Clips::bones`];
	/// [`NO_BONE`] for a track this skeleton has no bone for
	/// @param into - one local transform per bone, written in place
	pub fn sample(&self, time: f32, looping: bool, bones: &[u16], into: &mut [Transform]) {
		let moment = self.moment(time, looping);

		for (track, bone) in self.tracks.iter().zip(bones) {
			if *bone == NO_BONE {
				continue;
			}

			if let Some(local) = into.get_mut(usize::from(*bone)) {
				track.apply(moment, local);
			}
		}
	}
}

/// One entry of the clip registry.
pub type Clip = Entry<ClipData>;

/// One clip's tracks, resolved against one skeleton's bones.
///
/// The thing that makes a track naming a bone with text cost nothing per step.
/// It is kept beside the registry rather than handed to the game because the
/// game's own memory is plain bytes and cannot hold a list, and because the
/// two revisions it is checked against are only visible here.
#[derive(Clone, Debug)]
struct Binding {
	clip: ClipId,
	skeleton: SkeletonId,
	clip_revision: u32,
	skeleton_revision: u32,
	bones: Vec<u16>,
}

/// Which bone of a skeleton each of a clip's tracks moves.
fn resolve(clip: &ClipData, skeleton: Option<&SkeletonData>) -> Vec<u16> {
	clip.tracks
		.iter()
		.map(|track| {
			skeleton
				.and_then(|bones| bones.find(&track.bone))
				.unwrap_or(NO_BONE)
		})
		.collect()
}

/// Every clip the host has loaded, addressed by [`ClipId`], and what each of
/// them means on each skeleton.
///
/// Slot zero is [`ClipId::NONE`] and has no tracks, so a game asking for a clip
/// that is not there animates nothing rather than failing.
#[derive(Clone, Debug)]
pub struct Clips {
	entries: Registry<ClipData>,
	bindings: Vec<Binding>,
}

impl Clips {
	/// A registry holding the null clip and nothing else.
	#[must_use]
	pub fn new() -> Self {
		Self {
			entries: Registry::new(ClipData::default()),
			bindings: Vec::new(),
		}
	}

	/// Looks a clip up by name.
	///
	/// @param name - the name it was registered under, e.g. `models/hero/walk`
	/// @return its handle, or [`ClipId::NONE`] if nothing answers to it
	#[must_use]
	pub fn find(&self, name: &str) -> ClipId { ClipId::new(self.entries.find(name)) }

	/// Registers a clip under a name, replacing whatever was there.
	///
	/// @param name - what the game will ask for
	/// @param data - its tracks
	/// @return the handle, the same one as last time if the name is known
	pub fn insert(&mut self, name: &str, data: ClipData) -> ClipId {
		ClipId::new(self.entries.insert(name, data))
	}

	/// One clip, by handle.
	#[must_use]
	pub fn get(&self, id: ClipId) -> Option<&Clip> { self.entries.entry(id.index()) }

	/// The tracks of one clip, by handle.
	///
	/// A handle to nothing gives no tracks, which is what makes playing
	/// something that failed to load a loop over an empty list.
	#[must_use]
	pub fn data(&self, id: ClipId) -> &ClipData {
		static NOTHING: ClipData = ClipData { tracks: Vec::new() };

		self.get(id).map_or(&NOTHING, Entry::value)
	}

	/// How many clips there are, counting the null one.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Always `false`: slot zero always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// Every clip, in slot order, starting with the null one.
	pub fn iter(&self) -> impl Iterator<Item = &Clip> { self.entries.iter() }

	/// Works out which bone of a skeleton each of a clip's tracks moves, once.
	///
	/// Idempotent, like every other registration here: asking twice for a pair
	/// already worked out costs two comparisons. It has to be a call rather
	/// than something done when a clip is loaded, because a clip names no
	/// skeleton - the same walk plays on every rig whose bones answer to the
	/// same names, and which rig that is only the caller knows.
	///
	/// @param clip - what to play
	/// @param skeleton - what to play it on
	/// @param skeletons - the registry the second of those lives in
	pub fn bind(&mut self, clip: ClipId, skeleton: SkeletonId, skeletons: &Skeletons) {
		if let Some(index) = self.slot_of(clip, skeleton) {
			self.refresh(index, skeletons);

			return;
		}

		let bones = resolve(self.data(clip), skeletons.get(skeleton).map(Entry::value));

		self.bindings.push(Binding {
			clip,
			skeleton,
			clip_revision: self.revision(clip),
			skeleton_revision: revision_of(skeletons, skeleton),
			bones,
		});
	}

	/// Which bone of a skeleton each of a clip's tracks moves.
	///
	/// @param clip - what is being played
	/// @param skeleton - what it is being played on
	/// @return one bone index per track, or nothing at all if the pair was
	/// never bound - which samples nothing rather than moving the wrong bones
	#[must_use]
	pub fn bones(&self, clip: ClipId, skeleton: SkeletonId) -> &[u16] {
		self.slot_of(clip, skeleton)
			.map_or(&[], |index| self.bindings[index].bones.as_slice())
	}

	/// Works every binding out again whose clip or skeleton has been rewritten.
	///
	/// Called by the host after a pass over the compiled tree, which is the
	/// one moment either of them can change. Without it, editing a rig so that
	/// it gains a bone leaves every clip playing on it one bone out - and
	/// silently, because the indices all still resolve to something.
	///
	/// @param skeletons - the registry to work them out against
	pub fn relink(&mut self, skeletons: &Skeletons) {
		for index in 0..self.bindings.len() {
			self.refresh(index, skeletons);
		}
	}

	/// How many clip-and-skeleton pairs have been worked out.
	///
	/// Not bounded, unlike the tables a game spawns into: this grows with the
	/// clips and the skeletons that are actually played together, and both of
	/// those are the compiled asset tree rather than anything gameplay makes.
	#[must_use]
	pub fn bindings(&self) -> usize { self.bindings.len() }

	/// Where a pair's binding is kept, if it has one.
	fn slot_of(&self, clip: ClipId, skeleton: SkeletonId) -> Option<usize> {
		self.bindings
			.iter()
			.position(|binding| binding.clip == clip && binding.skeleton == skeleton)
	}

	/// The revision of one clip, or zero when there is no such clip.
	fn revision(&self, clip: ClipId) -> u32 { self.get(clip).map_or(0, Entry::revision) }

	/// Works one binding out again, if either side has moved since it was made.
	fn refresh(&mut self, index: usize, skeletons: &Skeletons) {
		let Some(binding) = self.bindings.get(index) else {
			return;
		};
		let (clip, skeleton) = (binding.clip, binding.skeleton);
		let clip_revision = self.revision(clip);
		let skeleton_revision = revision_of(skeletons, skeleton);

		if binding.clip_revision == clip_revision
			&& binding.skeleton_revision == skeleton_revision
		{
			return;
		}

		let bones = resolve(self.data(clip), skeletons.get(skeleton).map(Entry::value));

		if let Some(binding) = self.bindings.get_mut(index) {
			binding.bones = bones;
			binding.clip_revision = clip_revision;
			binding.skeleton_revision = skeleton_revision;
		}
	}
}

impl Default for Clips {
	fn default() -> Self { Self::new() }
}

/// The revision of one skeleton, or zero when there is no such skeleton.
fn revision_of(skeletons: &Skeletons, skeleton: SkeletonId) -> u32 {
	skeletons.get(skeleton).map_or(0, Entry::revision)
}

#[cfg(test)]
mod tests {
	use super::{
		super::{
			World,
			pose::{Pose, PoseId},
			skeleton::{Bone, NO_PARENT},
		},
		*,
	};

	/// A bone hanging off another, a stride along `x` from it.
	fn bone(name: &str, parent: u16, along: f32) -> Bone {
		Bone {
			name: name.to_owned(),
			parent,
			rest: Transform::at(Vec3::new(along, 0.0, 0.0)),
			..Bone::default()
		}
	}

	/// Three bones in a row.
	fn arm() -> SkeletonData {
		SkeletonData {
			bones: vec![
				bone("shoulder", NO_PARENT, 0.0),
				bone("elbow", 0, 1.0),
				bone("wrist", 1, 2.0),
			],
		}
	}

	/// The same three with two more above them, so every index is different.
	///
	/// What a clip authored against one rig and played on another actually
	/// looks like: the names are shared and nothing else is.
	fn taller_arm() -> SkeletonData {
		SkeletonData {
			bones: vec![
				bone("root", NO_PARENT, 0.0),
				bone("spine", 0, 0.5),
				bone("shoulder", 1, 0.0),
				bone("elbow", 2, 1.0),
				bone("wrist", 3, 2.0),
			],
		}
	}

	/// A quarter turn about `z`, as four numbers.
	fn quarter() -> [f32; 4] {
		let half = std::f32::consts::FRAC_PI_4;

		[0.0, 0.0, half.sin(), half.cos()]
	}

	/// A clip that walks the elbow along `y` and turns the wrist.
	fn take() -> ClipData {
		let turned = quarter();

		ClipData {
			tracks: vec![
				Track {
					bone: "elbow".to_owned(),
					channel: Channel::Position,
					interpolation: Interpolation::Linear,
					times: vec![0.0, 1.0, 2.0],
					values: vec![0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 8.0, 0.0],
				},
				Track {
					bone: "wrist".to_owned(),
					channel: Channel::Rotation,
					interpolation: Interpolation::Linear,
					times: vec![0.0, 2.0],
					values: vec![0.0, 0.0, 0.0, 1.0, turned[0], turned[1], turned[2], turned[3]],
				},
			],
		}
	}

	/// Three transforms, all at the origin.
	fn blank() -> Vec<Transform> { vec![Transform::IDENTITY; 3] }

	/// A world holding one skeleton, one clip and one pose of the first.
	fn posed(skeleton: SkeletonData, clip: ClipData) -> (World, PoseId, ClipId) {
		let mut world = World::new();
		let rig = world.skeletons.insert("rig", skeleton);
		let played = world.clips.insert("take", clip);
		let pose = world
			.poses
			.spawn(Pose::resting(rig, world.skeletons.bones(rig)));

		(world, pose, played)
	}

	#[test]
	fn a_key_is_read_exactly_at_the_moment_it_sits_at() {
		let clip = take();
		let mut into = blank();

		clip.sample(1.0, false, &[1, 2], &mut into);

		assert!(
			into[1]
				.position
				.abs_diff_eq(Vec3::new(0.0, 4.0, 0.0), 1.0e-6),
			"the middle key, not a blend of the two either side of it"
		);
	}

	#[test]
	fn between_two_keys_a_position_travels_and_a_turn_goes_round() {
		let clip = take();
		let mut into = blank();

		clip.sample(0.5, false, &[1, 2], &mut into);

		assert!(
			into[1]
				.position
				.abs_diff_eq(Vec3::new(0.0, 2.0, 0.0), 1.0e-6),
			"halfway between nothing and four"
		);

		let want = Quat::from_rotation_z(std::f32::consts::FRAC_PI_8);

		assert!(
			into[2].rotation.abs_diff_eq(want, 1.0e-6),
			"a quarter of the way through a clip that turns a right angle is a quarter of that \
			 angle, and it is the angle that is divided rather than the four numbers - dividing \
			 those lands about a degree away"
		);
	}

	#[test]
	fn a_step_track_holds_its_earlier_key_until_the_later_one_is_reached() {
		let mut clip = take();
		clip.tracks[0].interpolation = Interpolation::Step;

		let mut into = blank();

		clip.sample(0.99, false, &[1, 2], &mut into);

		assert!(
			into[1].position.abs_diff_eq(Vec3::ZERO, 1.0e-6),
			"still at the first key with a hundredth of a second to go"
		);

		clip.sample(1.0, false, &[1, 2], &mut into);

		assert!(
			into[1]
				.position
				.abs_diff_eq(Vec3::new(0.0, 4.0, 0.0), 1.0e-6),
			"and at the second one the moment it arrives"
		);
	}

	#[test]
	fn outside_its_keys_the_nearer_end_holds_rather_than_carrying_on() {
		let clip = take();
		let mut early = blank();
		let mut late = blank();

		clip.sample(-5.0, false, &[1, 2], &mut early);
		clip.sample(50.0, false, &[1, 2], &mut late);

		assert!(
			early[1].position.abs_diff_eq(Vec3::ZERO, 1.0e-6),
			"before the first key the first key holds, which is what the format says"
		);
		assert!(
			late[1]
				.position
				.abs_diff_eq(Vec3::new(0.0, 8.0, 0.0), 1.0e-6),
			"and after the last one the last one does, rather than running off"
		);
	}

	#[test]
	fn a_looping_clip_starts_again_and_one_that_does_not_holds_its_end() {
		let clip = take();

		assert!(
			(clip.moment(2.5, true) - 0.5).abs() < 1.0e-6,
			"half a second into the second go"
		);
		assert!((clip.moment(2.5, false) - 2.0).abs() < 1.0e-6, "or the end, held");
		assert!(
			(clip.moment(-0.5, true) - 1.5).abs() < 1.0e-6,
			"and running it backwards past the start comes round the other side"
		);
		assert!((clip.moment(-0.5, false) - 0.0).abs() < 1.0e-6, "or holds the start");
	}

	#[test]
	fn a_clip_of_no_length_is_read_at_its_only_moment() {
		let clip = ClipData {
			tracks: vec![Track {
				bone: "elbow".to_owned(),
				channel: Channel::Position,
				interpolation: Interpolation::Linear,
				times: vec![0.0],
				values: vec![0.0, 3.0, 0.0],
			}],
		};
		let mut into = blank();

		assert!((clip.moment(9.0, true) - 0.0).abs() < 1.0e-6, "and no division by its length");

		clip.sample(9.0, true, &[1], &mut into);

		assert!(
			into[1]
				.position
				.abs_diff_eq(Vec3::new(0.0, 3.0, 0.0), 1.0e-6)
		);
	}

	#[test]
	fn a_bone_no_track_names_is_left_exactly_as_it_was_found() {
		let clip = take();
		let mut into = blank();
		into[0].position = Vec3::new(7.0, 7.0, 7.0);

		clip.sample(1.0, false, &[1, 2], &mut into);

		assert!(
			into[0]
				.position
				.abs_diff_eq(Vec3::new(7.0, 7.0, 7.0), 1.0e-6),
			"the shoulder is not in the clip, so whatever was there stays - which is what makes \
			 a sampled pose complete rather than half empty"
		);
	}

	#[test]
	fn a_track_the_skeleton_has_no_bone_for_moves_nothing() {
		let clip = take();
		let mut into = blank();

		clip.sample(1.0, false, &[NO_BONE, 2], &mut into);

		assert!(into[1].position.abs_diff_eq(Vec3::ZERO, 1.0e-6), "nothing was written for it");
		assert!(
			!into[2]
				.rotation
				.abs_diff_eq(Quat::IDENTITY, 1.0e-6),
			"and the track beside it still played"
		);
	}

	#[test]
	fn a_channel_writes_its_own_part_of_a_transform_and_no_other() {
		let clip = take();
		let mut into = blank();
		into[1].scale = Vec3::splat(3.0);
		into[2].position = Vec3::new(0.0, 0.0, 5.0);

		clip.sample(2.0, false, &[1, 2], &mut into);

		assert!(
			into[1]
				.scale
				.abs_diff_eq(Vec3::splat(3.0), 1.0e-6),
			"a position track does not touch a scale"
		);
		assert!(
			into[2]
				.position
				.abs_diff_eq(Vec3::new(0.0, 0.0, 5.0), 1.0e-6),
			"nor a rotation track a position"
		);
	}

	#[test]
	fn a_clip_whose_times_run_backwards_still_gives_somewhere_to_stand() {
		// a file like this is refused at both ends, but a clip is public plain
		// data and can be built by hand. What a bad one gives is a pose rather
		// than a number that is not one.
		let clip = ClipData {
			tracks: vec![Track {
				bone: "elbow".to_owned(),
				channel: Channel::Position,
				interpolation: Interpolation::Linear,
				times: vec![2.0, 1.0],
				values: vec![0.0, 3.0, 0.0, 0.0, 9.0, 0.0],
			}],
		};
		let mut into = blank();

		for moment in [0.0, 1.5, 3.0] {
			clip.sample(moment, false, &[1], &mut into);

			assert!(
				into[1].position.is_finite(),
				"a clip whose times go backwards still gives somewhere to stand at {moment}"
			);
		}
	}

	#[test]
	fn asking_again_for_a_binding_whose_rig_has_moved_works_it_out_again() {
		// the host relinks after every pass over the tree, but playing asks
		// for a binding every step and a reload can land between two of them.
		let mut clips = Clips::new();
		let mut skeletons = Skeletons::new();
		let rig = skeletons.insert("rig", arm());
		let clip = clips.insert("take", take());

		clips.bind(clip, rig, &skeletons);
		skeletons.insert("rig", taller_arm());
		clips.bind(clip, rig, &skeletons);

		assert_eq!(
			clips.bones(clip, rig),
			&[3, 4],
			"binding again is what noticed, with nobody having called relink"
		);
		assert_eq!(clips.bindings(), 1, "and it is still the one binding");
	}

	#[test]
	fn binding_finds_each_track_the_bone_of_that_name() {
		let mut clips = Clips::new();
		let mut skeletons = Skeletons::new();
		let rig = skeletons.insert("rig", arm());
		let clip = clips.insert("take", take());

		clips.bind(clip, rig, &skeletons);

		assert_eq!(clips.bones(clip, rig), &[1, 2], "the elbow and the wrist, by name");
	}

	#[test]
	fn a_clip_plays_on_a_rig_whose_bones_are_numbered_differently() {
		// the reason a track carries text at all: the same three bones sit at
		// three different indices in the second rig, and every one of them
		// still answers.
		let mut clips = Clips::new();
		let mut skeletons = Skeletons::new();
		let short = skeletons.insert("short", arm());
		let tall = skeletons.insert("tall", taller_arm());
		let clip = clips.insert("take", take());

		clips.bind(clip, short, &skeletons);
		clips.bind(clip, tall, &skeletons);

		assert_eq!(clips.bones(clip, short), &[1, 2]);
		assert_eq!(clips.bones(clip, tall), &[3, 4], "two bones further down the same names");
		assert_eq!(clips.bindings(), 2, "one for each pair, and no more");
	}

	#[test]
	fn a_track_naming_a_bone_the_rig_has_not_binds_to_nothing() {
		let mut clips = Clips::new();
		let mut skeletons = Skeletons::new();
		let rig = skeletons.insert("rig", SkeletonData {
			bones: vec![bone("shoulder", NO_PARENT, 0.0), bone("elbow", 0, 1.0)],
		});
		let clip = clips.insert("take", take());

		clips.bind(clip, rig, &skeletons);

		assert_eq!(
			clips.bones(clip, rig),
			&[1, NO_BONE],
			"the rig has no wrist, and that track lands nowhere rather than on the elbow"
		);
	}

	#[test]
	fn binding_a_pair_twice_binds_it_once() {
		let mut clips = Clips::new();
		let mut skeletons = Skeletons::new();
		let rig = skeletons.insert("rig", arm());
		let clip = clips.insert("take", take());

		clips.bind(clip, rig, &skeletons);
		clips.bind(clip, rig, &skeletons);
		clips.bind(clip, rig, &skeletons);

		assert_eq!(clips.bindings(), 1, "registering is idempotent here as everywhere else");
	}

	#[test]
	fn a_pair_that_was_never_bound_moves_nothing_rather_than_the_wrong_bones() {
		let clips = Clips::new();

		assert!(
			clips
				.bones(ClipId::new(1), SkeletonId::new(1))
				.is_empty(),
			"no binding is no bones, and sampling with none writes nothing"
		);
	}

	#[test]
	fn a_rig_reloaded_with_its_bones_renumbered_takes_its_bindings_with_it() {
		let mut clips = Clips::new();
		let mut skeletons = Skeletons::new();
		let rig = skeletons.insert("rig", arm());
		let clip = clips.insert("take", take());

		clips.bind(clip, rig, &skeletons);

		assert_eq!(clips.bones(clip, rig), &[1, 2]);

		// the same name, a wider rig: this is what recompiling a model that
		// gained a bone does, and the handle does not move.
		skeletons.insert("rig", taller_arm());
		clips.relink(&skeletons);

		assert_eq!(
			clips.bones(clip, rig),
			&[3, 4],
			"the binding followed the names rather than staying on two indices that now mean \
			 something else"
		);
	}

	#[test]
	fn a_clip_reloaded_with_another_track_takes_its_binding_with_it() {
		let mut clips = Clips::new();
		let mut skeletons = Skeletons::new();
		let rig = skeletons.insert("rig", arm());
		let clip = clips.insert("take", take());

		clips.bind(clip, rig, &skeletons);

		let mut wider = take();
		wider.tracks.push(Track {
			bone: "shoulder".to_owned(),
			channel: Channel::Scale,
			interpolation: Interpolation::Linear,
			times: vec![0.0],
			values: vec![2.0, 2.0, 2.0],
		});
		clips.insert("take", wider);
		clips.relink(&skeletons);

		assert_eq!(clips.bones(clip, rig), &[1, 2, 0], "the new track is bound too");
	}

	#[test]
	fn playing_a_clip_writes_what_it_names_and_rests_everything_else() {
		let (mut world, pose, clip) = posed(arm(), take());

		assert!(world.play(pose, clip, 2.0, false), "the pose is there");

		let locals = &world.poses.get(pose).expect("still there").locals;

		assert!(
			locals[0].position.abs_diff_eq(Vec3::ZERO, 1.0e-6),
			"the shoulder rests where the skeleton puts it"
		);
		assert!(
			locals[1]
				.position
				.abs_diff_eq(Vec3::new(0.0, 8.0, 0.0), 1.0e-6),
			"the elbow is where the clip's last key says, not where its rest is"
		);
		assert!(
			locals[2]
				.rotation
				.abs_diff_eq(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2), 1.0e-6),
			"and the wrist has turned the right angle"
		);
	}

	#[test]
	fn playing_puts_a_bone_the_clip_stopped_naming_back_at_its_rest() {
		// the reason resting comes first: without it a bone would keep last
		// step's attitude forever, which is what a pose that is only ever
		// written over looks like.
		let (mut world, pose, clip) = posed(arm(), take());

		assert!(world.play(pose, clip, 2.0, false));

		world.clips.insert("take", ClipData::default());
		world.clips.relink(&world.skeletons);

		assert!(world.play(pose, clip, 2.0, false));

		let locals = &world.poses.get(pose).expect("still there").locals;

		assert!(
			locals[1]
				.position
				.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1.0e-6),
			"back at the rest the skeleton gave it, not left at where the clip had put it"
		);
	}

	#[test]
	fn playing_into_a_handle_nobody_answers_to_writes_nothing() {
		let (mut world, pose, clip) = posed(arm(), take());

		world.poses.despawn(pose);

		assert!(!world.play(pose, clip, 1.0, false), "a stale pose is refused rather than made");
	}

	#[test]
	fn playing_a_clip_that_is_not_there_rests_the_pose_rather_than_failing() {
		let (mut world, pose, _) = posed(arm(), take());

		assert!(world.play(pose, ClipId::NONE, 1.0, false), "the pose was written");

		let locals = &world.poses.get(pose).expect("still there").locals;

		assert_eq!(locals.len(), 3, "all three bones");
		assert!(
			locals[1]
				.position
				.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1.0e-6),
			"and every one of them at its rest"
		);
	}

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
