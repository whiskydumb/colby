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
//!
//! **A tree can carry its character over the ground.** Root motion is the
//! travel of one bone, the tree's [`Tree::motion`], taken out of the pose and
//! handed back as a [`Travel`] for the game to move its character by. Nothing
//! here writes anybody's transform: a character is moved by whatever moves it,
//! which in this engine is a game calling a controller that knows about walls.
//! [`travel`] works the answer out from the clip clock alone and keeps nothing
//! between two calls, so a step replayed for a prediction, a world put back
//! from a save and a clock run at another rate all get the same one; and
//! [`evaluate`] pins the same bone where its rest stands, so that the pose and
//! the travel together are the clip exactly as it was authored.

use super::{
	entity::Transform,
	registry::{Entry, Registry},
	skeleton::{Bone, NO_PARENT, SkeletonData, SkeletonId, Skeletons},
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
///
/// Also what [`Tree::motion`] holds when nothing is to travel, which is what a
/// tree holds unless it is told otherwise.
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

	/// How far this clip carries a character between two moments.
	///
	/// Read off one bone, the one whose travel is the character's, and answered
	/// as what that bone did over the ground with its height left out: the
	/// ground it crossed and the turn about straight up it made, measured in
	/// the frame the character stood in at `from`. That frame is what brings a
	/// clip that turns as it walks out on a curve rather than a zigzag, once a
	/// game turns its character by [`Travel::turned`] as it goes.
	///
	/// **A pure function of the two moments, and that is the whole design.**
	/// Nothing is kept between two calls, so the travel over one step is the
	/// same number whoever asks and however often: a prediction replaying its
	/// commands, a world put back from a save, a host and its client. The seam
	/// of a looping clip is worked out rather than remembered: a clock that
	/// ran past the end has done a whole lap more, and a lap is the travel
	/// from the clip's first moment to its last, so one step may cross any
	/// number of seams, either way, and still land where an unbroken walk
	/// would have. A clip that does not loop holds each of its ends, so a
	/// character whose clip has finished stops.
	///
	/// @param bone - the name of the bone that carries the character
	/// @param rest - that bone's rest, which is what its travel is measured
	/// from @param from - where the caller's clock stood at the start, in
	/// seconds @param to - where it stands now
	/// @param looping - whether the clip starts again rather than holding its
	/// end @return how far it carried the character; no travel at all for a
	/// clip of no length, a bone with no name or a moment that is not a number
	#[must_use]
	pub fn travel(
		&self,
		bone: &str,
		rest: Transform,
		from: f32,
		to: f32,
		looping: bool,
	) -> Travel {
		let length = self.duration();

		if bone.is_empty() || length <= 0.0 || !from.is_finite() || !to.is_finite() {
			return Travel::NONE;
		}

		// a clip that does not loop wants no clamp here: sampling already holds
		// each of its ends past it, which is the rule its pose is played by.
		if !looping {
			return self
				.footprint(bone, rest, from)
				.inverse()
				.then(self.footprint(bone, rest, to));
		}

		let was = self.footprint(bone, rest, from.rem_euclid(length));
		let now = self.footprint(bone, rest, to.rem_euclid(length));
		let crossed = to.div_euclid(length) - from.div_euclid(length);

		// the common case, a step inside one lap, is one difference and nothing
		// composed around it, so it is exactly what a clip that does not loop
		// would say about the same two moments.
		//
		// @note: no test can tell this from the composition below it, which
		// comes back to the bit through the start and its inverse on every
		// fixture tried. It is kept because it is the common case, and it saves
		// the two footprints the start and the lap would cost every step.
		if crossed == 0.0 {
			return was.inverse().then(now);
		}

		let start = self.footprint(bone, rest, 0.0);
		let lap = start
			.inverse()
			.then(self.footprint(bone, rest, length));

		was.inverse()
			.then(start)
			.then(laps(lap, crossed))
			.then(start.inverse())
			.then(now)
	}

	/// Where one bone stands over the ground at a moment, against its rest.
	///
	/// The motion across the ground that carries the bone's rest to where the
	/// clip has the bone: the turn about straight up, counted the whole way
	/// round rather than folded into one turn, and the ground crossed. It is
	/// the rest *turned by that much* that is carried onto the bone, which is
	/// what makes a turn in place a turn about the bone rather than about the
	/// middle of the character.
	fn footprint(&self, bone: &str, rest: Transform, moment: f32) -> Travel {
		let mut local = rest;

		// every track naming the bone, in order, which is exactly what sampling
		// does to it: a later track writing the same channel wins here as well.
		for track in self
			.tracks
			.iter()
			.filter(|track| track.bone == bone)
		{
			track.apply(moment, &mut local);
		}

		let turned = self
			.tracks
			.iter()
			.rfind(|track| track.bone == bone && track.channel == Channel::Rotation)
			.map_or(0.0, |track| heading_along(track, rest.rotation, moment));

		Travel {
			moved: flat(local.position) - Quat::from_rotation_y(turned) * flat(rest.position),
			turned,
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

/// The most nodes one blend tree may have.
///
/// A tree is built by the game every step out of a handful of nodes - a couple
/// of clips, a blend between them, a layer over the top. This is a bound on a
/// runaway rather than a budget, and it is what keeps the scratch a blend
/// works in from being asked to hold an arbitrary number of poses.
pub const MAX_NODES: usize = 64;

/// One step of a blend.
///
/// Plain data, and small enough to copy: a game builds a `Vec` of these every
/// step out of numbers it worked out that step, rather than editing a graph
/// somebody authored. That is the one place this engine departs from what the
/// field does, and it is a departure in the tooling rather than in the shape -
/// every engine checked carries its graph as data too, and two of the five
/// build one from code as readily as from an editor. Revisit when there is a
/// graph view to author one in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Node {
	/// One clip, at a moment on the game's own clock.
	///
	/// The leaf, and the only node that reads nothing: what comes out is the
	/// skeleton at rest with the clip written over it, which is what makes
	/// every pose in the tree a whole one.
	Clip {
		/// What to play.
		clip: ClipId,

		/// Seconds on whatever clock the game keeps for it.
		time: f32,

		/// Where that clock stood one step ago.
		///
		/// Read by nothing but [`travel`], which is how far the clip carried
		/// its character between the two, so a tree with no motion bone may
		/// put anything here. A game advancing its clock by `dt` writes what it
		/// held before adding it.
		from: f32,

		/// Whether it starts again rather than holding its last key.
		looping: bool,
	},

	/// Two poses mixed evenly over every bone.
	///
	/// The two-input blend every engine has under some name. A weight of zero
	/// is the first input untouched and a weight of one is the second; a blend
	/// space over several clips is a chain of these, which is what a blend
	/// space *is* once the neighbors have been picked.
	Blend {
		/// Where the pose at weight zero comes from.
		first: u16,

		/// Where the pose at weight one comes from.
		second: u16,

		/// How far between them, `0.0 ..= 1.0`.
		weight: f32,
	},

	/// Two poses mixed over one branch of the skeleton and not the rest.
	///
	/// An upper body doing one thing while the legs do another: the bone named
	/// by `branch` and everything hanging off it are blended towards the second
	/// input, and every other bone is the first input untouched. One bone
	/// rather than a set of them because a branch is what a body part *is*, and
	/// because two of these chained cover the case a single set would.
	Mask {
		/// Where the pose outside the branch comes from, whole.
		first: u16,

		/// Where the pose inside the branch is blended towards.
		second: u16,

		/// The bone the branch starts at, and which is itself inside it. A
		/// bone this skeleton does not have leaves the whole pose as `first`.
		branch: u16,

		/// How far towards the second input inside the branch, `0.0 ..= 1.0`.
		weight: f32,
	},
}

/// A blend, as a list of steps with the answer at one of them.
///
/// **A child is always written before the node that reads it**, which is the
/// same rule a skeleton's bones follow and buys the same three things: the
/// whole tree evaluates in one forward pass with no recursion, a cycle cannot
/// be built, and an index that passes the check is an index that exists. @ref
/// [`Self::is_ordered`].
#[derive(Clone, Debug, PartialEq)]
pub struct Tree {
	/// Every step, children before the nodes that read them.
	pub nodes: Vec<Node>,

	/// Which of them is the answer.
	pub root: u16,

	/// The bone that carries the character, or [`NO_BONE`] for none.
	///
	/// Named on the tree rather than on a clip, because a clip names no
	/// skeleton and whether a walk moves its character is a question about how
	/// it is played: the same walk plays in place in a preview. One bone for
	/// the whole tree, since a character goes one way at a time, and a clip in
	/// the mix that stays where it is carries nothing on its own.
	///
	/// Set, it is taken out of the pose [`evaluate`] works out: held where its
	/// rest stands over the ground and facing the way its rest faces, with its
	/// height and every turn that is not about straight up left in the pose.
	/// What it would have done is what [`travel`] hands back. An index like a
	/// mask's branch, found once with
	/// [`SkeletonData::find`](super::skeleton::SkeletonData::find).
	pub motion: u16,
}

impl Default for Tree {
	/// Written out rather than derived, because the derived one would name
	/// bone nought as the bone that travels.
	fn default() -> Self { Self::new() }
}

impl Tree {
	/// A tree with nothing in it, and nothing to travel.
	#[must_use]
	pub const fn new() -> Self {
		Self {
			nodes: Vec::new(),
			root: 0,
			motion: NO_BONE,
		}
	}

	/// Adds a step, and makes it the answer.
	///
	/// Building bottom up is the only order the ordering rule allows, so the
	/// node added last is the root of every tree that is built at all. Setting
	/// it here means a game never has to say so, and one that wants something
	/// else writes [`root`](Self::root) itself afterwards.
	///
	/// @param node - the step to add
	/// @return where it landed, which is what a later node names it by
	pub fn push(&mut self, node: Node) -> u16 {
		let at = u16::try_from(self.nodes.len()).unwrap_or(u16::MAX);

		self.nodes.push(node);
		self.root = at;

		at
	}

	/// How many steps it has.
	#[must_use]
	pub fn len(&self) -> usize { self.nodes.len() }

	/// Whether it has none, which cannot be evaluated.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

	/// Whether every node's inputs come before it, and the root is one of them.
	///
	/// The invariant evaluation stands on. It also proves every input is a real
	/// index, which is the same statement: an input below a node's own index is
	/// below the length as well.
	#[must_use]
	pub fn is_ordered(&self) -> bool {
		// a tree with no nodes is refused by this too, because every index is
		// past the end of nothing.
		if usize::from(self.root) >= self.nodes.len() {
			return false;
		}

		self.nodes
			.iter()
			.enumerate()
			.all(|(at, node)| match *node {
				| Node::Clip { .. } => true,
				| Node::Blend { first, second, .. } | Node::Mask { first, second, .. } =>
					usize::from(first) < at && usize::from(second) < at,
			})
	}
}

/// Whether one bone is the branch's own bone or hangs off it.
///
/// A walk up the parents rather than a pass down the children, because a bone
/// knows its parent and nothing knows its children. The walk is bounded by the
/// number of bones, so a skeleton whose parents contradict themselves stops
/// rather than climbing forever - though one that does is refused where it is
/// read. The cost is the depth of the skeleton per bone, which for a biped is
/// about eight.
///
/// @note: the check for [`NO_PARENT`] is deliberately redundant with the bounds
/// check below it, because the sentinel is past any index a skeleton can hold
/// and asking for it would miss anyway. It is kept because it is the only line
/// that says a walk stops at a root, and because that being safe otherwise
/// depends on the sentinel's value rather than on anything written down.
fn inside(bones: &[Bone], bone: usize, branch: u16) -> bool {
	let mut at = bone;

	for _ in 0..bones.len() {
		if u16::try_from(at).is_ok_and(|index| index == branch) {
			return true;
		}

		let Some(parent) = bones.get(at).map(|bone| bone.parent) else {
			return false;
		};

		if parent == NO_PARENT {
			return false;
		}

		at = usize::from(parent);
	}

	false
}

/// One bone of a masked blend.
///
/// Its own function because the alternative is a conditional three levels deep
/// inside a loop inside a match arm, which reads as nesting rather than as the
/// one sentence it is: inside the branch a bone moves towards the layer, and
/// outside it the bone is left alone.
fn masked(outside: Transform, within: Transform, weight: f32, within_branch: bool) -> Transform {
	if within_branch {
		outside.lerp(within, weight)
	} else {
		outside
	}
}

/// One already-written pose out of the scratch.
fn run(done: &[Transform], index: u16, width: usize) -> &[Transform] {
	let at = usize::from(index).saturating_mul(width);

	done.get(at..at.saturating_add(width))
		.unwrap_or_default()
}

/// Works a blend tree out into one pose.
///
/// A single forward pass: each node writes its own run of the scratch out of
/// runs that are already written, because a child always comes first. Nothing
/// recurses, nothing is visited twice, and a tree that could loop cannot be
/// built - @ref [`Tree::is_ordered`], which is checked before anything is
/// written.
///
/// The scratch is one pose per node and belongs to the caller, so a game
/// animating ten characters allocates nothing after the first of them.
///
/// @param tree - what to work out
/// @param clips - the registry the leaves name, with its bindings
/// @param skeleton - which skeleton the leaves are bound against
/// @param bones - that skeleton's bones, parents first
/// @param scratch - a buffer the caller keeps between calls, resized as needed
/// @param out - the root's pose, one local transform per bone
/// @return `false` if the tree could not be worked out, in which case `out` is
/// the skeleton at rest rather than anything half written
pub fn evaluate(
	tree: &Tree,
	clips: &Clips,
	skeleton: SkeletonId,
	bones: &[Bone],
	scratch: &mut Vec<Transform>,
	out: &mut Vec<Transform>,
) -> bool {
	out.clear();
	out.extend(bones.iter().map(|bone| bone.rest));

	if tree.len() > MAX_NODES || !tree.is_ordered() || bones.is_empty() {
		return false;
	}

	let width = bones.len();

	// @note: the clear cannot be observed, because every node writes the whole
	// of its own run before anything reads it. It is kept so that what a node
	// would inherit if one ever did not is a transform that means nothing
	// rather than a pose left over from another tree.
	scratch.clear();
	scratch.resize(tree.len().saturating_mul(width), Transform::IDENTITY);

	for (at, node) in tree.nodes.iter().enumerate() {
		let (done, rest) = scratch.split_at_mut(at.saturating_mul(width));
		let Some(here) = rest.get_mut(..width) else {
			return false;
		};

		match *node {
			| Node::Clip { clip, time, looping, .. } => {
				for (slot, bone) in here.iter_mut().zip(bones) {
					*slot = bone.rest;
				}

				clips
					.data(clip)
					.sample(time, looping, clips.bones(clip, skeleton), here);
			},
			| Node::Blend { first, second, weight } => {
				let (one, two) = (run(done, first, width), run(done, second, width));

				for ((slot, earlier), later) in here.iter_mut().zip(one).zip(two) {
					*slot = earlier.lerp(*later, weight);
				}
			},
			| Node::Mask { first, second, branch, weight } => {
				let (one, two) = (run(done, first, width), run(done, second, width));

				for (index, ((slot, outside), within)) in
					here.iter_mut().zip(one).zip(two).enumerate()
				{
					*slot = masked(*outside, *within, weight, inside(bones, index, branch));
				}
			},
		}
	}

	let answer = usize::from(tree.root).saturating_mul(width);
	let Some(pose) = scratch.get(answer..answer.saturating_add(width)) else {
		return false;
	};

	out.clear();
	out.extend_from_slice(pose);

	// the bone that carries the character stays where its rest stands, and what
	// it would have done is `travel`'s to hand back. A bone this skeleton has
	// not got pins nothing.
	if let (Some(bone), Some(local)) =
		(bones.get(usize::from(tree.motion)), out.get_mut(usize::from(tree.motion)))
	{
		*local = pinned(*local, bone.rest);
	}

	true
}

/// The most laps of a looping clip one travel counts.
///
/// A step at sixty a second crosses the seam of a one-second walk at most once,
/// and a clock that jumped further than this in one step was set rather than
/// run: a game restarting its clock, not a character that walked sixty-four
/// laps between two frames. Past it the travel stops counting, which is a bound
/// on a loop rather than a promise about where such a character ends up.
pub const MAX_LAPS: usize = 64;

/// How far a character is carried over the ground, and how far it is turned.
///
/// What [`travel`] hands back, and the shape of a motion that stays on the
/// ground: a distance across it and a turn about straight up, with the height
/// left to whatever holds a character up. Measured in the character's own
/// frame as it stood at the start, so the same travel is the same walk
/// whichever way the character happens to face.
///
/// ```text
///   let travel = world.travel(pose, &tree);
///   world.animate(pose, &tree);
///   let there = travel.applied(standing);
///   let velocity = (there.position - standing.position) / world.dt;
///   // move_and_slide with that velocity, then face there.rotation
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Travel {
	/// The ground crossed, in the character's own frame at the start. Its `y`
	/// is always nought.
	pub moved: Vec3,

	/// The turn about straight up, in radians, counterclockwise seen from
	/// above.
	pub turned: f32,
}

impl Travel {
	/// Going nowhere and turning not at all.
	pub const NONE: Self = Self { moved: Vec3::ZERO, turned: 0.0 };

	/// This travel and then another, the second measured from where the first
	/// ended.
	///
	/// @param next - the travel that follows, in the frame this one ends in
	#[must_use]
	pub fn then(self, next: Self) -> Self {
		Self {
			moved: self.moved + Quat::from_rotation_y(self.turned) * next.moved,
			turned: self.turned + next.turned,
		}
	}

	/// The travel that undoes this one.
	#[must_use]
	pub fn inverse(self) -> Self {
		Self {
			moved: -(Quat::from_rotation_y(-self.turned) * self.moved),
			turned: -self.turned,
		}
	}

	/// This travel part of the way towards another, which is what a blend of
	/// two clips travels.
	///
	/// Both ends exact, as [`Transform::lerp`] has them and for its reason: a
	/// weight of one is the second input, not something an ulp away from it.
	///
	/// @param other - the travel at the far end
	/// @param t - zero for this one, one for the other
	#[must_use]
	pub fn lerp(self, other: Self, t: f32) -> Self {
		if t <= 0.0 || self == other {
			return self;
		}

		if t >= 1.0 {
			return other;
		}

		Self {
			moved: self.moved.lerp(other.moved, t),
			turned: (other.turned - self.turned).mul_add(t, self.turned),
		}
	}

	/// Where something standing at `at` is carried to.
	///
	/// The travel laid inside the transform, so it goes the way `at` faces and
	/// as far as `at` is scaled: a character twice the size walks twice as far
	/// on the same clip, which is what legs twice as long are for.
	///
	/// @param at - where it stands now
	#[must_use]
	pub fn applied(self, at: Transform) -> Transform {
		at.then(Transform {
			position: self.moved,
			rotation: Quat::from_rotation_y(self.turned),
			scale: Vec3::ONE,
		})
	}
}

/// How far a tree carries its character, by the travel of its motion bone.
///
/// The tree worked out the way [`evaluate`] works it out, one node at a time
/// and children first, except that a node holds a travel rather than a pose: a
/// clip's own ([`ClipData::travel`]), a blend's two inputs mixed by its weight,
/// and a mask's layer only when the motion bone is inside its branch. The
/// travel goes wherever the motion bone's own transform goes, which is what
/// keeps it the travel of the pose that bone ends up in.
///
/// Nothing is written and nothing is kept, and no binding is needed: the
/// motion bone's tracks are found by its name.
///
/// @param tree - the tree, with its motion bone
/// @param clips - the registry its leaves name
/// @param bones - the skeleton it is worked out over, parents first
/// @return how far it carried the character; no travel at all for a tree with
/// no motion bone, one that cannot be worked out, or a bone whose name an
/// earlier bone already answers to
#[must_use]
pub fn travel(tree: &Tree, clips: &Clips, bones: &[Bone]) -> Travel {
	let motion = usize::from(tree.motion);
	let Some(bone) = bones.get(motion) else {
		return Travel::NONE;
	};

	// a track names its bone the way a binding resolves it, by the first bone
	// answering to the name, so a later bone sharing it is moved by no track at
	// all. One with no name is moved by none either, which the clip's own
	// travel answers for.
	let named = bones
		.iter()
		.position(|each| each.name == bone.name);

	if named != Some(motion) || tree.len() > MAX_NODES || !tree.is_ordered() {
		return Travel::NONE;
	}

	let mut done = [Travel::NONE; MAX_NODES];

	for (at, node) in tree.nodes.iter().enumerate() {
		let here = match *node {
			| Node::Clip { clip, time, from, looping } => clips
				.data(clip)
				.travel(&bone.name, bone.rest, from, time, looping),
			| Node::Blend { first, second, weight } =>
				carried(&done, first).lerp(carried(&done, second), weight),
			| Node::Mask { first, second, branch, weight } => layered(
				carried(&done, first),
				carried(&done, second),
				weight,
				inside(bones, motion, branch),
			),
		};

		if let Some(slot) = done.get_mut(at) {
			*slot = here;
		}
	}

	carried(&done, tree.root)
}

/// One already-worked travel out of the list, or none for an index past it.
fn carried(done: &[Travel], index: u16) -> Travel {
	done.get(usize::from(index))
		.copied()
		.unwrap_or(Travel::NONE)
}

/// The travel of a masked blend, which is the layer's only inside its branch.
///
/// [`masked`] for a travel: the motion bone is either inside the branch and
/// moves towards the layer, or outside it and keeps the first input, and its
/// travel does exactly what it does.
fn layered(outside: Travel, within: Travel, weight: f32, within_branch: bool) -> Travel {
	if within_branch {
		outside.lerp(within, weight)
	} else {
		outside
	}
}

/// The turn about straight up that a rotation holds, in radians.
///
/// The rotation taken apart as a turn about `y` after one about a level axis,
/// and the first of those read as an angle: the part of a quaternion about `y`
/// is its `y` and its `w` alone.
///
/// @note: what comes back is only ever used modulo a whole turn, as a
/// difference [`wrapped`] into half a turn either side or as a rotation built
/// from it. So neither sign of the quaternion is picked, and a half turn about
/// a level axis, which faces no way at all, reads as some whole number of
/// turns: both are the same answer to everything that asks.
fn heading(turn: Quat) -> f32 { 2.0 * turn.y.atan2(turn.w) }

/// An angle brought inside half a turn either side of nought.
fn wrapped(angle: f32) -> f32 {
	(angle + core::f32::consts::PI).rem_euclid(core::f32::consts::TAU) - core::f32::consts::PI
}

/// A point on the ground under another, its height taken away.
const fn flat(point: Vec3) -> Vec3 { Vec3::new(point.x, 0.0, point.z) }

/// How far a rotation track has turned its bone about straight up by a moment,
/// counted the whole way round.
///
/// The turn is read at every key up to the moment and the steps between them
/// added, each brought inside half a turn: two neighboring keys of a rotation
/// are played the short way round, so no step between them turns half a
/// circle. A clip that turns a whole circle therefore says a whole circle
/// rather than nothing, which is what makes one lap of it turn its character
/// all the way round.
///
/// @param track - a rotation track
/// @param rest - the bone's rest rotation, which is the heading of nought
/// @param moment - where in the clip
fn heading_along(track: &Track, rest: Quat, moment: f32) -> f32 {
	if track.times.is_empty() {
		return 0.0;
	}

	let against = rest.inverse();
	let mut last = heading(turn(track.key(0)) * against);
	let mut total = last;

	for (index, time) in track.times.iter().enumerate().skip(1) {
		if *time > moment {
			break;
		}

		let now = heading(turn(track.key(index)) * against);

		total += wrapped(now - last);
		last = now;
	}

	let mut at = Transform::IDENTITY;

	track.apply(moment, &mut at);

	total + wrapped(heading(at.rotation * against) - last)
}

/// One lap's travel taken a number of times, backwards for a negative number.
///
/// A loop rather than a formula, and bounded by [`MAX_LAPS`]: an honest step
/// crosses one seam or none.
///
/// @param lap - one whole lap's travel
/// @param count - how many, a whole number however it is stored
fn laps(lap: Travel, count: f32) -> Travel {
	let step = if count < 0.0 { lap.inverse() } else { lap };
	let mut left = count.abs();
	let mut out = Travel::NONE;

	for _ in 0..MAX_LAPS {
		if left < 0.5 {
			break;
		}

		out = out.then(step);
		left -= 1.0;
	}

	out
}

/// A motion bone with its travel over the ground taken out.
///
/// Where its rest stands over the ground and facing the way its rest faces,
/// with its height, its scale and every turn that is not about straight up
/// left as the pose had them: a hip that bobs and rocks as it walks still bobs
/// and rocks. What was taken out is [`travel`]'s to hand back.
///
/// @param local - the bone as the tree left it, relative to its parent
/// @param rest - the bone's rest
fn pinned(local: Transform, rest: Transform) -> Transform {
	let facing = Quat::from_rotation_y(heading(local.rotation * rest.rotation.inverse()));

	Transform {
		position: Vec3::new(rest.position.x, local.position.y, rest.position.z),
		rotation: facing.inverse() * local.rotation,
		scale: local.scale,
	}
}

// the two an identity needs, for the table above. @ref `registry_identity!`
crate::registry_identity!(Clips, ClipId, entries);

#[cfg(test)]
mod tests {
	use super::{
		super::{
			World,
			character::{self, Motion},
			net::Command,
			pose::{Pose, PoseId},
		},
		*,
	};
	use crate::glam::Mat4;

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

	/// A clip that puts the elbow four along `y` and leaves it there.
	fn held(along: f32) -> ClipData {
		ClipData {
			tracks: vec![Track {
				bone: "elbow".to_owned(),
				channel: Channel::Position,
				interpolation: Interpolation::Linear,
				times: vec![0.0],
				values: vec![0.0, along, 0.0],
			}],
		}
	}

	/// A clip that turns the wrist a right angle and leaves it there.
	fn twisted() -> ClipData {
		let turned = quarter();

		ClipData {
			tracks: vec![Track {
				bone: "wrist".to_owned(),
				channel: Channel::Rotation,
				interpolation: Interpolation::Linear,
				times: vec![0.0],
				values: turned.to_vec(),
			}],
		}
	}

	/// A registry of two clips over one skeleton, both already bound.
	fn stage(one: ClipData, two: ClipData) -> (Clips, Skeletons, SkeletonId, ClipId, ClipId) {
		let mut clips = Clips::new();
		let mut skeletons = Skeletons::new();
		let rig = skeletons.insert("rig", arm());
		let first = clips.insert("one", one);
		let second = clips.insert("two", two);

		clips.bind(first, rig, &skeletons);
		clips.bind(second, rig, &skeletons);

		(clips, skeletons, rig, first, second)
	}

	/// Works a tree out over the three-bone arm.
	fn worked(
		tree: &Tree,
		clips: &Clips,
		skeletons: &Skeletons,
		rig: SkeletonId,
	) -> Vec<Transform> {
		let mut scratch = Vec::new();
		let mut out = Vec::new();

		evaluate(tree, clips, rig, skeletons.bones(rig), &mut scratch, &mut out);

		out
	}

	#[test]
	fn a_tree_of_one_clip_is_the_clip() {
		let (clips, skeletons, rig, first, _) = stage(held(4.0), held(8.0));
		let mut tree = Tree::new();

		tree.push(Node::Clip {
			clip: first,
			time: 0.0,
			from: 0.0,
			looping: false,
		});

		let out = worked(&tree, &clips, &skeletons, rig);

		assert_eq!(out.len(), 3, "one transform per bone");
		assert!(
			out[1]
				.position
				.abs_diff_eq(Vec3::new(0.0, 4.0, 0.0), 1.0e-6),
			"the elbow where the clip put it"
		);
		assert!(
			out[0].position.abs_diff_eq(Vec3::ZERO, 1.0e-6),
			"and the shoulder at the rest the leaf started from"
		);
	}

	#[test]
	fn a_blend_is_the_first_at_nothing_the_second_at_one_and_between_them_between() {
		let (clips, skeletons, rig, first, second) = stage(held(4.0), held(8.0));

		for (weight, want) in [(0.0, 4.0), (0.25, 5.0), (0.5, 6.0), (1.0, 8.0)] {
			let mut tree = Tree::new();
			let one = tree.push(Node::Clip {
				clip: first,
				time: 0.0,
				from: 0.0,
				looping: false,
			});
			let two = tree.push(Node::Clip {
				clip: second,
				time: 0.0,
				from: 0.0,
				looping: false,
			});

			tree.push(Node::Blend { first: one, second: two, weight });

			let out = worked(&tree, &clips, &skeletons, rig);

			assert!(
				out[1]
					.position
					.abs_diff_eq(Vec3::new(0.0, want, 0.0), 1.0e-6),
				"at {weight} the elbow should be {want} along, and it is {:?}",
				out[1].position
			);
		}
	}

	#[test]
	fn a_blend_of_a_clip_with_itself_is_the_clip_at_every_weight() {
		let (clips, skeletons, rig, first, _) = stage(held(4.0), held(8.0));
		let mut tree = Tree::new();
		let one = tree.push(Node::Clip {
			clip: first,
			time: 0.0,
			from: 0.0,
			looping: false,
		});

		tree.push(Node::Blend { first: one, second: one, weight: 0.5 });

		let out = worked(&tree, &clips, &skeletons, rig);

		assert!(
			out[1]
				.position
				.abs_diff_eq(Vec3::new(0.0, 4.0, 0.0), 1.0e-6),
			"both sides are the same run of the scratch, read twice rather than taken once"
		);
	}

	#[test]
	fn a_bone_only_one_side_moves_blends_against_that_side_s_rest() {
		// what makes a leaf start from the rest worth the cost: the second
		// clip says nothing about the elbow, so the blend is between where the
		// first clip put it and where the skeleton has it, rather than between
		// it and wherever it happened to be last step.
		let (clips, skeletons, rig, first, second) = stage(held(4.0), twisted());
		let mut tree = Tree::new();
		let one = tree.push(Node::Clip {
			clip: first,
			time: 0.0,
			from: 0.0,
			looping: false,
		});
		let two = tree.push(Node::Clip {
			clip: second,
			time: 0.0,
			from: 0.0,
			looping: false,
		});

		tree.push(Node::Blend { first: one, second: two, weight: 0.5 });

		let out = worked(&tree, &clips, &skeletons, rig);

		assert!(
			out[1]
				.position
				.abs_diff_eq(Vec3::new(0.5, 2.0, 0.0), 1.0e-6),
			"halfway between four along y and the elbow's own rest one along x, which is {:?}",
			out[1].position
		);
		assert!(
			out[2]
				.rotation
				.abs_diff_eq(Quat::from_rotation_z(std::f32::consts::FRAC_PI_4), 1.0e-6),
			"and the wrist half of the way to the right angle the other clip turns it"
		);
	}

	#[test]
	fn a_mask_moves_the_branch_and_leaves_the_rest_of_the_skeleton_alone() {
		let (clips, skeletons, rig, first, second) = stage(held(4.0), twisted());
		let mut tree = Tree::new();
		let one = tree.push(Node::Clip {
			clip: first,
			time: 0.0,
			from: 0.0,
			looping: false,
		});
		let two = tree.push(Node::Clip {
			clip: second,
			time: 0.0,
			from: 0.0,
			looping: false,
		});

		// the branch starts at the elbow, so the elbow and the wrist are in it
		// and the shoulder is not.
		tree.push(Node::Mask {
			first: one,
			second: two,
			branch: 1,
			weight: 1.0,
		});

		let out = worked(&tree, &clips, &skeletons, rig);

		assert!(
			out[1]
				.position
				.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1.0e-6),
			"the elbow is inside the branch, so it comes from the layer - which says nothing \
			 about it and therefore rests it"
		);
		assert!(
			out[2]
				.rotation
				.abs_diff_eq(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2), 1.0e-6),
			"the wrist hangs off the elbow, so it is inside too and the layer turns it"
		);
		assert!(
			out[0].position.abs_diff_eq(Vec3::ZERO, 1.0e-6),
			"and the shoulder is outside and keeps the first input"
		);
	}

	#[test]
	fn a_mask_at_half_weight_takes_the_branch_half_the_way() {
		let (clips, skeletons, rig, first, second) = stage(held(4.0), twisted());
		let mut tree = Tree::new();
		let one = tree.push(Node::Clip {
			clip: first,
			time: 0.0,
			from: 0.0,
			looping: false,
		});
		let two = tree.push(Node::Clip {
			clip: second,
			time: 0.0,
			from: 0.0,
			looping: false,
		});

		tree.push(Node::Mask {
			first: one,
			second: two,
			branch: 2,
			weight: 0.5,
		});

		let out = worked(&tree, &clips, &skeletons, rig);

		assert!(
			out[1]
				.position
				.abs_diff_eq(Vec3::new(0.0, 4.0, 0.0), 1.0e-6),
			"the elbow is above the branch, so it is the first input whole"
		);
		assert!(
			out[2]
				.rotation
				.abs_diff_eq(Quat::from_rotation_z(std::f32::consts::FRAC_PI_4), 1.0e-6),
			"and the wrist, which is the branch, is half of the way over"
		);
	}

	#[test]
	fn a_mask_over_a_bone_the_rig_has_not_leaves_every_bone_as_the_first_input() {
		let (clips, skeletons, rig, first, second) = stage(held(4.0), twisted());
		let mut tree = Tree::new();
		let one = tree.push(Node::Clip {
			clip: first,
			time: 0.0,
			from: 0.0,
			looping: false,
		});
		let two = tree.push(Node::Clip {
			clip: second,
			time: 0.0,
			from: 0.0,
			looping: false,
		});

		tree.push(Node::Mask {
			first: one,
			second: two,
			branch: 40,
			weight: 1.0,
		});

		let out = worked(&tree, &clips, &skeletons, rig);

		assert!(
			out[1]
				.position
				.abs_diff_eq(Vec3::new(0.0, 4.0, 0.0), 1.0e-6),
			"nothing is inside a branch that is not there"
		);
		assert!(
			out[2]
				.rotation
				.abs_diff_eq(Quat::IDENTITY, 1.0e-6),
			"including the wrist"
		);
	}

	#[test]
	fn a_tree_whose_inputs_do_not_come_first_is_refused_rather_than_read() {
		let (clips, skeletons, rig, first, _) = stage(held(4.0), held(8.0));
		let backwards = Tree {
			nodes: vec![Node::Blend { first: 1, second: 1, weight: 0.5 }, Node::Clip {
				clip: first,
				time: 0.0,
				from: 0.0,
				looping: false,
			}],
			root: 0,
			motion: NO_BONE,
		};

		assert!(!backwards.is_ordered(), "the blend reads a node written after it");

		let mut scratch = Vec::new();
		let mut out = Vec::new();

		assert!(
			!evaluate(&backwards, &clips, rig, skeletons.bones(rig), &mut scratch, &mut out),
			"and evaluating it says so"
		);
		assert_eq!(out.len(), 3, "with the pose left at rest rather than half written");
		assert!(out[1].position.abs_diff_eq(Vec3::X, 1.0e-6), "which is where the rest is");
	}

	#[test]
	fn a_node_that_reads_itself_is_refused_on_either_side() {
		// the cycle the ordering rule exists to make unbuildable, and it is
		// worth having both sides of it: a node reading itself through its
		// second input is a loop exactly as much as one reading itself through
		// its first.
		let (clips, skeletons, rig, first, _) = stage(held(4.0), held(8.0));
		let leaf = Node::Clip {
			clip: first,
			time: 0.0,
			from: 0.0,
			looping: false,
		};

		for (one, two) in [(1_u16, 0_u16), (0, 1)] {
			let looping = Tree {
				nodes: vec![leaf, Node::Blend { first: one, second: two, weight: 0.5 }],
				root: 1,
				motion: NO_BONE,
			};

			assert!(
				!looping.is_ordered(),
				"a blend at one reading node one is reading itself, whichever input it is"
			);

			let mut scratch = Vec::new();
			let mut out = Vec::new();

			assert!(
				!evaluate(&looping, &clips, rig, skeletons.bones(rig), &mut scratch, &mut out),
				"and working it out says so rather than reading a pose nobody wrote"
			);
		}
	}

	#[test]
	fn the_answer_is_the_node_the_root_names_rather_than_the_last_one() {
		// pushing makes the last node the answer, which is the convention and
		// not the rule. A game that writes the field gets what it wrote.
		let (clips, skeletons, rig, first, second) = stage(held(4.0), held(8.0));
		let mut tree = Tree::new();
		let one = tree.push(Node::Clip {
			clip: first,
			time: 0.0,
			from: 0.0,
			looping: false,
		});
		let two = tree.push(Node::Clip {
			clip: second,
			time: 0.0,
			from: 0.0,
			looping: false,
		});

		tree.push(Node::Blend { first: one, second: two, weight: 1.0 });
		tree.root = one;

		let out = worked(&tree, &clips, &skeletons, rig);

		assert!(
			out[1]
				.position
				.abs_diff_eq(Vec3::new(0.0, 4.0, 0.0), 1.0e-6),
			"the first clip, which is what the root names, rather than the blend at the end"
		);
	}

	#[test]
	fn an_empty_tree_and_a_root_that_is_not_a_node_are_both_refused() {
		let (clips, skeletons, rig, first, _) = stage(held(4.0), held(8.0));

		assert!(!Tree::new().is_ordered(), "nothing to work out");

		let stray = Tree {
			nodes: vec![Node::Clip {
				clip: first,
				time: 0.0,
				from: 0.0,
				looping: false,
			}],
			root: 7,
			motion: NO_BONE,
		};

		assert!(!stray.is_ordered(), "and an answer at a node that is not there");

		let mut scratch = Vec::new();
		let mut out = Vec::new();

		assert!(!evaluate(&stray, &clips, rig, skeletons.bones(rig), &mut scratch, &mut out));
	}

	#[test]
	fn more_nodes_than_a_tree_may_hold_are_refused() {
		let (clips, skeletons, rig, first, _) = stage(held(4.0), held(8.0));
		let leaf = Node::Clip {
			clip: first,
			time: 0.0,
			from: 0.0,
			looping: false,
		};
		let mut tree = Tree::new();

		for _ in 0..=MAX_NODES {
			tree.push(leaf);
		}

		assert!(tree.is_ordered(), "every one of them reads nothing, so the order is fine");

		let mut scratch = Vec::new();
		let mut out = Vec::new();

		assert!(
			!evaluate(&tree, &clips, rig, skeletons.bones(rig), &mut scratch, &mut out),
			"and it is the count that refuses it"
		);
	}

	#[test]
	fn pushing_a_node_makes_it_the_answer() {
		let leaf = Node::Clip {
			clip: ClipId::NONE,
			time: 0.0,
			from: 0.0,
			looping: false,
		};
		let mut tree = Tree::new();

		assert_eq!(tree.push(leaf), 0, "the first lands at nothing");
		assert_eq!(tree.root, 0, "and is the answer");
		assert_eq!(tree.push(leaf), 1);
		assert_eq!(tree.root, 1, "and so is whatever was added last");
		assert_eq!(tree.len(), 2);
		assert!(!tree.is_empty());
	}

	#[test]
	fn a_scratch_is_reused_between_two_trees_of_different_sizes() {
		// the buffer belongs to the caller so that a crowd allocates once, and
		// what it held last time has to be nothing to a later call.
		let (clips, skeletons, rig, first, second) = stage(held(4.0), held(8.0));
		let mut scratch = Vec::new();
		let mut out = Vec::new();
		let mut big = Tree::new();
		let one = big.push(Node::Clip {
			clip: first,
			time: 0.0,
			from: 0.0,
			looping: false,
		});
		let two = big.push(Node::Clip {
			clip: second,
			time: 0.0,
			from: 0.0,
			looping: false,
		});

		big.push(Node::Blend { first: one, second: two, weight: 1.0 });

		assert!(evaluate(&big, &clips, rig, skeletons.bones(rig), &mut scratch, &mut out));

		let mut small = Tree::new();
		small.push(Node::Clip {
			clip: first,
			time: 0.0,
			from: 0.0,
			looping: false,
		});

		assert!(evaluate(&small, &clips, rig, skeletons.bones(rig), &mut scratch, &mut out));
		assert!(
			out[1]
				.position
				.abs_diff_eq(Vec3::new(0.0, 4.0, 0.0), 1.0e-6),
			"the second tree's answer, not a run left over from the first"
		);
	}

	#[test]
	fn animating_a_pose_binds_every_clip_the_tree_names_and_writes_the_answer() {
		let mut world = World::new();
		let rig = world.skeletons.insert("rig", arm());
		let first = world.clips.insert("one", held(4.0));
		let second = world.clips.insert("two", held(8.0));
		let pose = world
			.poses
			.spawn(Pose::resting(rig, world.skeletons.bones(rig)));
		let mut tree = Tree::new();
		let one = tree.push(Node::Clip {
			clip: first,
			time: 0.0,
			from: 0.0,
			looping: false,
		});
		let two = tree.push(Node::Clip {
			clip: second,
			time: 0.0,
			from: 0.0,
			looping: false,
		});

		tree.push(Node::Blend { first: one, second: two, weight: 0.5 });

		assert!(world.animate(pose, &tree), "the pose is there and the tree works out");
		assert_eq!(world.clips.bindings(), 2, "both leaves were bound, and nobody asked");

		let locals = &world.poses.get(pose).expect("still there").locals;

		assert!(
			locals[1]
				.position
				.abs_diff_eq(Vec3::new(0.0, 6.0, 0.0), 1.0e-6),
			"halfway between the two clips"
		);
	}

	#[test]
	fn animating_a_stale_pose_handle_writes_nothing() {
		let mut world = World::new();
		let rig = world.skeletons.insert("rig", arm());
		let pose = world
			.poses
			.spawn(Pose::resting(rig, world.skeletons.bones(rig)));
		let mut tree = Tree::new();

		tree.push(Node::Clip {
			clip: ClipId::NONE,
			time: 0.0,
			from: 0.0,
			looping: false,
		});
		world.poses.despawn(pose);

		assert!(!world.animate(pose, &tree), "a stale pose is refused rather than made");
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

	/// The walker: a root off the middle of the character, turned and tilted at
	/// rest, with a foot hanging off it and a spine beside the foot.
	///
	/// Off the middle, so that a turn about the root and a turn about the
	/// character's own origin are different answers; turned and tilted, so that
	/// a heading read against the rest and one read against nothing are
	/// different too.
	fn walker() -> SkeletonData {
		SkeletonData {
			bones: vec![
				Bone {
					name: "root".to_owned(),
					rest: Transform {
						position: Vec3::new(0.25, 0.9, -0.5),
						rotation: Quat::from_rotation_y(0.5) * Quat::from_rotation_x(0.3),
						scale: Vec3::ONE,
					},
					..Bone::default()
				},
				bone("foot", 0, 0.1),
				bone("spine", 0, -0.2),
			],
		}
	}

	/// The walker's root at rest.
	fn rooted() -> Transform { walker().bones[0].rest }

	/// The root's position at five moments along a path that is no straight
	/// line: along `x` three times the square of the time, the height bobbing
	/// and `z` wandering.
	fn stride() -> Track {
		Track {
			bone: "root".to_owned(),
			channel: Channel::Position,
			interpolation: Interpolation::Linear,
			times: vec![0.0, 0.25, 0.5, 0.75, 1.0],
			values: vec![
				0.0, 0.9, -0.5, 0.1875, 0.95, -0.42, 0.75, 0.88, -0.41, 1.6875, 0.93, -0.45, 3.0,
				0.9, -0.4,
			],
		}
	}

	/// A clip of nothing but the stride.
	fn striding() -> ClipData { ClipData { tracks: vec![stride()] } }

	/// The root turned about straight up by so many degrees at each of so many
	/// moments, over its tilted rest.
	fn turning(times: &[f32], degrees: &[f32]) -> Track {
		let rest = rooted().rotation;

		Track {
			bone: "root".to_owned(),
			channel: Channel::Rotation,
			interpolation: Interpolation::Linear,
			times: times.to_vec(),
			values: degrees
				.iter()
				.flat_map(|angle| (Quat::from_rotation_y(angle.to_radians()) * rest).to_array())
				.collect(),
		}
	}

	/// The root sliding along `x` by `far` over one second, and nothing else.
	fn sliding(far: f32) -> ClipData {
		ClipData {
			tracks: vec![Track {
				bone: "root".to_owned(),
				channel: Channel::Position,
				interpolation: Interpolation::Linear,
				times: vec![0.0, 1.0],
				values: vec![0.0, 0.9, -0.5, far, 0.9, -0.5],
			}],
		}
	}

	/// A root that walks a curve as it turns and rocks, keyed at nine moments
	/// over the walker's rest, starting turned.
	fn curving() -> ClipData {
		let rest = rooted();
		let times: Vec<f32> = (0..9_u8)
			.map(|key| f32::from(key) / 8.0)
			.collect();
		let mut places = Vec::with_capacity(times.len() * 3);
		let mut turns = Vec::with_capacity(times.len() * 4);

		for time in &times {
			let facing = 0.6_f32.mul_add(time * time, 1.2_f32.mul_add(*time, 0.4));
			let rock = Quat::from_rotation_x(0.1 * (5.0 * time).sin());

			places.extend_from_slice(&[
				1.5_f32.mul_add(facing.sin(), rest.position.x),
				0.05_f32.mul_add((6.0 * time).sin(), rest.position.y),
				1.5_f32.mul_add(1.0 - facing.cos(), rest.position.z),
			]);
			turns.extend_from_slice(
				&(Quat::from_rotation_y(facing) * rock * rest.rotation).to_array(),
			);
		}

		ClipData {
			tracks: vec![
				Track {
					bone: "root".to_owned(),
					channel: Channel::Position,
					interpolation: Interpolation::Linear,
					times: times.clone(),
					values: places,
				},
				Track {
					bone: "root".to_owned(),
					channel: Channel::Rotation,
					interpolation: Interpolation::Linear,
					times,
					values: turns,
				},
			],
		}
	}

	/// A registry of the walker and the clips handed in, every one of them
	/// bound, the first called `clip0` and so on.
	fn walking(taken: Vec<ClipData>) -> (Clips, Skeletons, SkeletonId) {
		let mut clips = Clips::new();
		let mut skeletons = Skeletons::new();
		let rig = skeletons.insert("walker", walker());

		for (index, data) in taken.into_iter().enumerate() {
			let clip = clips.insert(&format!("clip{index}"), data);

			clips.bind(clip, rig, &skeletons);
		}

		(clips, skeletons, rig)
	}

	/// A tree of one clip, carried by the walker's root.
	fn carried_by_root(clip: ClipId, from: f32, time: f32, looping: bool) -> Tree {
		let mut tree = Tree::new();

		tree.push(Node::Clip { clip, time, from, looping });
		tree.motion = 0;

		tree
	}

	/// Where the walker's foot stands in its model, for the pose a tree works
	/// out.
	fn foot_at(tree: &Tree, clips: &Clips, rig: SkeletonId, bones: &[Bone]) -> Mat4 {
		let (mut scratch, mut out) = (Vec::new(), Vec::new());

		assert!(evaluate(tree, clips, rig, bones, &mut scratch, &mut out), "the tree works out");

		out[0].matrix() * out[1].matrix()
	}

	/// A clip's travel taken one step at a time and added up.
	fn stepwise(clip: &ClipData, rate: u16, steps: u16) -> Travel {
		let mut went = Travel::NONE;

		for step in 0..steps {
			let from = f32::from(step) / f32::from(rate);
			let to = f32::from(step + 1) / f32::from(rate);

			went = went.then(clip.travel("root", rooted(), from, to, true));
		}

		went
	}

	#[test]
	fn a_tree_names_no_bone_to_travel_unless_it_is_told_one() {
		assert_eq!(Tree::new().motion, NO_BONE, "a new tree carries nothing");
		assert_eq!(
			Tree::default().motion,
			NO_BONE,
			"and neither does a default one, which a derived default would have given bone \
			 nought"
		);
	}

	#[test]
	fn a_clip_carries_its_character_as_far_as_its_root_crosses_the_ground() {
		let went = striding().travel("root", rooted(), 0.25, 0.75, false);

		assert!(
			went.moved
				.abs_diff_eq(Vec3::new(1.5, 0.0, -0.03), 1.0e-5),
			"from the second key to the fourth, with the height left out: {}",
			went.moved
		);
		assert!(went.turned.abs() < 1.0e-6, "and a root that never turns turns nothing");
	}

	#[test]
	fn a_step_across_the_seam_is_the_end_of_one_lap_and_the_start_of_the_next() {
		let clip = striding();
		let went = clip.travel("root", rooted(), 0.75, 1.25, true);

		// the last quarter of one lap and the first quarter of the next: one and
		// five sixteenths, and three sixteenths, along x, and along z what each
		// of the two quarters wandered.
		assert!(
			went.moved
				.abs_diff_eq(Vec3::new(1.5, 0.0, 0.13), 1.0e-5),
			"{}",
			went.moved
		);

		let back = clip.travel("root", rooted(), 1.25, 0.75, true);

		assert!(
			back.moved
				.abs_diff_eq(Vec3::new(-1.5, 0.0, -0.13), 1.0e-5),
			"and backwards across it is the same way back: {}",
			back.moved
		);
	}

	#[test]
	fn a_step_over_several_seams_counts_every_lap_it_crossed() {
		let went = striding().travel("root", rooted(), 0.25, 3.25, true);

		assert!(
			went.moved
				.abs_diff_eq(Vec3::new(9.0, 0.0, 0.3), 1.0e-4),
			"three whole laps of three along x and a tenth along z: {}",
			went.moved
		);
	}

	#[test]
	fn a_clip_that_does_not_loop_stops_carrying_its_character_at_its_end() {
		let clip = striding();

		assert!(
			clip.travel("root", rooted(), 0.75, 1.5, false)
				.moved
				.abs_diff_eq(Vec3::new(1.3125, 0.0, 0.05), 1.0e-5),
			"the last quarter, and nothing past the end"
		);
		assert!(
			clip.travel("root", rooted(), 1.2, 2.0, false)
				.moved
				.abs_diff_eq(Vec3::ZERO, 1.0e-6),
			"and nothing at all once it has finished"
		);
	}

	#[test]
	fn inside_one_lap_a_looping_clip_says_exactly_what_one_that_does_not_would() {
		let clip = ClipData {
			tracks: vec![stride(), turning(&[0.0, 0.5, 1.0], &[20.0, 25.0, 40.0])],
		};
		let looping = clip.travel("root", rooted(), 0.3, 0.9, true);
		let once = clip.travel("root", rooted(), 0.3, 0.9, false);

		assert!(
			looping.moved.abs_diff_eq(once.moved, 0.0)
				&& (looping.turned - once.turned).abs() <= 0.0,
			"the same arithmetic, not merely close: {looping:?} against {once:?}"
		);
	}

	#[test]
	fn the_steps_of_a_walk_add_up_to_the_walk_at_every_rate() {
		// what taking the seam in closed form buys: two and a half laps are as
		// far at thirty steps a second as at a hundred and forty-four.
		let whole = striding().travel("root", rooted(), 0.0, 2.5, true);

		assert!(
			whole
				.moved
				.abs_diff_eq(Vec3::new(6.75, 0.0, 0.29), 1.0e-4),
			"two laps and half of a third, in one go: {}",
			whole.moved
		);

		for rate in [30_u16, 60, 144] {
			let went = stepwise(&striding(), rate, rate * 5 / 2);

			assert!(
				went.moved.abs_diff_eq(whole.moved, 1.0e-4),
				"at {rate} a second: {}",
				went.moved
			);
		}
	}

	#[test]
	fn a_turn_in_place_turns_the_character_about_its_root_and_not_its_middle() {
		let clip = ClipData {
			tracks: vec![turning(&[0.0, 0.5, 1.0], &[0.0, 30.0, 90.0])],
		};
		let went = clip.travel("root", rooted(), 0.0, 1.0, false);

		assert!(
			(went.turned - core::f32::consts::FRAC_PI_2).abs() < 1.0e-5,
			"a quarter turn, read against the tilted rest: {}",
			went.turned
		);
		// the root stands a quarter of a unit across and half a unit back from
		// the character's own middle, so turning about it moves the middle.
		assert!(
			went.moved
				.abs_diff_eq(Vec3::new(0.75, 0.0, -0.25), 1.0e-5),
			"{}",
			went.moved
		);

		let there = went.applied(Transform::IDENTITY);
		let ground = there.rotation * flat(rooted().position) + there.position;

		assert!(
			ground.abs_diff_eq(flat(rooted().position), 1.0e-5),
			"and where the root stands on the ground has not moved: {ground}"
		);
	}

	#[test]
	fn a_clip_that_turns_a_whole_circle_turns_its_character_a_whole_circle() {
		// a quarter turn a key. The first key and the last face the same way, so
		// a travel read off those two alone would turn nothing at all.
		let clip = ClipData {
			tracks: vec![turning(&[0.0, 0.25, 0.5, 0.75, 1.0], &[
				0.0, 90.0, 180.0, 270.0, 360.0,
			])],
		};
		let once = clip.travel("root", rooted(), 0.0, 1.0, false);
		let twice = clip.travel("root", rooted(), 0.0, 2.0, true);

		assert!(
			(once.turned - core::f32::consts::TAU).abs() < 1.0e-4,
			"once round: {}",
			once.turned
		);
		assert!(
			2.0_f32
				.mul_add(-core::f32::consts::TAU, twice.turned)
				.abs() < 1.0e-4,
			"and twice round for two laps: {}",
			twice.turned
		);
	}

	#[test]
	fn the_root_that_carries_the_character_is_pinned_over_its_rest_keeping_its_height_and_tilt() {
		let tilt = Quat::from_rotation_z(0.2);
		let rest = rooted();
		let turn = Track {
			bone: "root".to_owned(),
			channel: Channel::Rotation,
			interpolation: Interpolation::Linear,
			times: vec![0.0],
			values: (Quat::from_rotation_y(0.7) * tilt * rest.rotation)
				.to_array()
				.to_vec(),
		};
		let (clips, skeletons, rig) = walking(vec![ClipData { tracks: vec![stride(), turn] }]);
		let tree = carried_by_root(clips.find("clip0"), 0.0, 0.5, false);
		let (mut scratch, mut out) = (Vec::new(), Vec::new());

		assert!(evaluate(&tree, &clips, rig, skeletons.bones(rig), &mut scratch, &mut out));
		assert!(
			out[0]
				.position
				.abs_diff_eq(Vec3::new(0.25, 0.88, -0.5), 1.0e-6),
			"over its rest on the ground, at the height the clip gives it: {}",
			out[0].position
		);

		let want = tilt * rest.rotation;

		for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
			assert!(
				(out[0].rotation * axis).abs_diff_eq(want * axis, 1.0e-5),
				"the tilt stays and the turn goes, seen along {axis}"
			);
		}

		assert!(
			out[1]
				.position
				.abs_diff_eq(Vec3::new(0.1, 0.0, 0.0), 1.0e-6),
			"and nothing below it is touched"
		);
	}

	#[test]
	fn a_tree_with_no_bone_to_travel_pins_nothing_and_carries_nothing() {
		let (clips, skeletons, rig) = walking(vec![striding()]);
		let bones = skeletons.bones(rig);

		for motion in [NO_BONE, 40] {
			let mut tree = carried_by_root(clips.find("clip0"), 0.0, 0.5, false);
			let (mut scratch, mut out) = (Vec::new(), Vec::new());

			tree.motion = motion;

			assert!(evaluate(&tree, &clips, rig, bones, &mut scratch, &mut out));
			assert!(
				out[0]
					.position
					.abs_diff_eq(Vec3::new(0.75, 0.88, -0.41), 1.0e-6),
				"motion {motion}: the root where the clip has it"
			);
			assert_eq!(travel(&tree, &clips, bones), Travel::NONE, "motion {motion}");
		}
	}

	#[test]
	fn a_bone_that_answers_to_a_name_an_earlier_bone_holds_carries_nothing() {
		// a track names its bone by the first bone of that name, so a spine
		// renamed to the root's name is moved by none of the root's tracks.
		let mut skeleton = walker();
		let mut clips = Clips::new();
		let clip = clips.insert("stride", striding());
		let mut tree = carried_by_root(clip, 0.25, 0.75, false);

		skeleton.bones[2].name = "root".to_owned();
		tree.motion = 2;

		assert_eq!(travel(&tree, &clips, &skeleton.bones), Travel::NONE, "the spine is no root");

		tree.motion = 0;

		assert_ne!(
			travel(&tree, &clips, &skeleton.bones),
			Travel::NONE,
			"the first of the two is"
		);

		// and with the first bone's name taken away, the tracks name the second.
		skeleton.bones[0].name = String::new();

		assert_eq!(
			travel(&tree, &clips, &skeleton.bones),
			Travel::NONE,
			"a bone with no name carries nothing"
		);

		tree.motion = 2;

		assert_ne!(
			travel(&tree, &clips, &skeleton.bones),
			Travel::NONE,
			"and the spine answers now"
		);
	}

	#[test]
	fn a_blend_carries_its_character_between_its_two_clips_by_its_weight() {
		let (clips, skeletons, rig) = walking(vec![striding(), sliding(6.0)]);
		let mut tree = Tree::new();
		let one = tree.push(Node::Clip {
			clip: clips.find("clip0"),
			time: 0.75,
			from: 0.25,
			looping: false,
		});
		let two = tree.push(Node::Clip {
			clip: clips.find("clip1"),
			time: 0.75,
			from: 0.25,
			looping: false,
		});

		tree.push(Node::Blend { first: one, second: two, weight: 0.25 });
		tree.motion = 0;

		let went = travel(&tree, &clips, skeletons.bones(rig));

		assert!(
			(went.moved.x - 1.875).abs() < 1.0e-5,
			"a quarter of the way from the stride's one and a half to the slide's three: {}",
			went.moved
		);
	}

	#[test]
	fn a_mask_carries_its_character_by_its_layer_only_when_the_root_is_inside_its_branch() {
		let (clips, skeletons, rig) = walking(vec![striding(), sliding(6.0)]);

		for (branch, want) in [(0_u16, 3.0), (2, 1.5)] {
			let mut tree = Tree::new();
			let one = tree.push(Node::Clip {
				clip: clips.find("clip0"),
				time: 0.75,
				from: 0.25,
				looping: false,
			});
			let two = tree.push(Node::Clip {
				clip: clips.find("clip1"),
				time: 0.75,
				from: 0.25,
				looping: false,
			});

			tree.push(Node::Mask {
				first: one,
				second: two,
				branch,
				weight: 1.0,
			});
			tree.motion = 0;

			let went = travel(&tree, &clips, skeletons.bones(rig));

			assert!(
				(went.moved.x - want).abs() < 1.0e-5,
				"a branch at bone {branch}: {}",
				went.moved
			);
		}
	}

	#[test]
	fn of_two_tracks_turning_the_root_the_later_one_turns_the_character() {
		// the rule sampling has, and the travel has to keep it or the pose and
		// the character would disagree about which way the walker faces.
		let clip = ClipData {
			tracks: vec![turning(&[0.0, 1.0], &[0.0, 30.0]), turning(&[0.0, 1.0], &[0.0, 60.0])],
		};
		let went = clip.travel("root", rooted(), 0.0, 1.0, false);

		assert!(
			(went.turned - 60.0_f32.to_radians()).abs() < 1.0e-5,
			"the second track's sixty degrees: {}",
			went.turned
		);
	}

	#[test]
	fn a_tree_that_cannot_be_worked_out_carries_nothing_as_it_poses_nothing() {
		// working such a tree out leaves the pose at rest, so a travel out of it
		// would carry a character the pose says is standing still.
		let (clips, skeletons, rig) = walking(vec![striding()]);
		let bones = skeletons.bones(rig);
		let leaf = Node::Clip {
			clip: clips.find("clip0"),
			time: 0.75,
			from: 0.25,
			looping: false,
		};
		let mut crowded = Tree::new();

		for _ in 0..=MAX_NODES {
			crowded.push(leaf);
		}

		crowded.root = 0;
		crowded.motion = 0;

		assert_eq!(travel(&crowded, &clips, bones), Travel::NONE, "one node too many");

		let backwards = Tree {
			nodes: vec![Node::Blend { first: 1, second: 1, weight: 0.5 }, leaf],
			root: 1,
			motion: 0,
		};

		assert_eq!(
			travel(&backwards, &clips, bones),
			Travel::NONE,
			"a node read before it is written"
		);
	}

	#[test]
	fn a_clock_that_is_no_number_carries_nothing() {
		let clip = striding();

		for (from, to) in [(f32::NAN, 0.5), (0.25, f32::INFINITY)] {
			assert_eq!(
				clip.travel("root", rooted(), from, to, true),
				Travel::NONE,
				"{from} to {to}"
			);
		}

		// a bone with no name is moved by no track, not even one with no name
		// either, because a binding finds no bone for an empty name.
		let mut nameless = striding();

		nameless.tracks[0].bone = String::new();

		assert_eq!(
			nameless.travel("", rooted(), 0.25, 0.75, true),
			Travel::NONE,
			"nor a bone with no name"
		);
		assert_eq!(
			ClipData::default().travel("root", rooted(), 0.25, 0.75, true),
			Travel::NONE,
			"nor a clip of no length"
		);
	}

	#[test]
	fn a_clock_set_a_thousand_laps_on_counts_no_more_laps_than_the_bound() {
		let went = striding().travel("root", rooted(), 0.25, 1000.25, true);
		let bound = f32::from(u8::try_from(MAX_LAPS).expect("a small number"));

		assert!(
			went.moved
				.abs_diff_eq(Vec3::new(3.0 * bound, 0.0, 0.1 * bound), 1.0e-2),
			"{MAX_LAPS} laps' worth rather than a thousand, and an answer rather than a hang: {}",
			went.moved
		);
	}

	#[test]
	fn travels_follow_one_another_undo_and_carry_what_they_are_laid_inside() {
		let one = Travel {
			moved: Vec3::new(1.0, 0.0, 2.0),
			turned: 0.5,
		};
		let two = Travel {
			moved: Vec3::new(-0.5, 0.0, 0.25),
			turned: -1.25,
		};
		let both = one.then(two);
		let want =
			Vec3::new(1.0, 0.0, 2.0) + Quat::from_rotation_y(0.5) * Vec3::new(-0.5, 0.0, 0.25);

		assert!(
			both.moved.abs_diff_eq(want, 1.0e-6),
			"the second turned by the first: {}",
			both.moved
		);
		assert!((both.turned + 0.75).abs() < 1.0e-6, "and the turns added: {}", both.turned);

		let undone = both.then(both.inverse());

		assert!(
			undone.moved.abs_diff_eq(Vec3::ZERO, 1.0e-6) && undone.turned.abs() < 1.0e-6,
			"undone: {undone:?}"
		);

		let at = Transform {
			position: Vec3::new(3.0, 1.0, -2.0),
			rotation: Quat::from_rotation_y(1.0),
			scale: Vec3::splat(2.0),
		};
		let there = one.applied(at);

		assert!(
			there
				.position
				.abs_diff_eq(at.position + at.rotation * Vec3::new(2.0, 0.0, 4.0), 1.0e-5),
			"the way it faces and as far as it is scaled: {}",
			there.position
		);

		for axis in [Vec3::X, Vec3::Z] {
			assert!(
				(there.rotation * axis).abs_diff_eq(Quat::from_rotation_y(1.5) * axis, 1.0e-5),
				"turned on by half a radian"
			);
		}

		assert_eq!(one.lerp(two, 0.0), one, "a blend at nought is the first, exactly");
		assert_eq!(one.lerp(two, 1.0), two, "and at one the second");

		// turns whose difference rounds, so a blend that worked the far end out
		// rather than taking it would land an ulp away from it.
		let (near, far) =
			(Travel { moved: Vec3::X, turned: 0.2 }, Travel { moved: Vec3::Z, turned: -0.4 });

		assert_eq!(near.lerp(far, 1.0), far, "the far end, not something next to it");
		assert!((one.lerp(two, 0.5).turned + 0.375).abs() < 1.0e-6, "and halfway is halfway");
	}

	#[test]
	fn a_character_carried_by_its_root_moves_exactly_as_the_clip_does_in_place() {
		// the equality the whole of root motion rests on. The pose pinned and the
		// character carried by its travel, against the same clip played in place
		// on a character that never moves: at every step the one is the other
		// moved by the same rigid motion, the one between them at the start. So
		// nothing slides, nothing drifts and nothing turns wrong.
		let (clips, skeletons, rig) = walking(vec![curving()]);
		let bones = skeletons.bones(rig);
		let clip = clips.find("clip0");
		let mut standing = Transform::IDENTITY;
		let mut first = None;

		for step in 0..=60_u8 {
			let from = f32::from(step.saturating_sub(1)) / 60.0;
			let time = f32::from(step) / 60.0;
			let pinning = carried_by_root(clip, from, time, false);
			let mut in_place = pinning.clone();

			in_place.motion = NO_BONE;
			standing = travel(&pinning, &clips, bones).applied(standing);

			let gap = standing.matrix()
				* foot_at(&pinning, &clips, rig, bones)
				* foot_at(&in_place, &clips, rig, bones).inverse();
			let start = *first.get_or_insert(gap);

			assert!(gap.abs_diff_eq(start, 1.0e-4), "step {step}: {gap:?} against {start:?}");
		}

		assert!(
			standing.position.length() > 1.0,
			"and it did go somewhere: {}",
			standing.position
		);
	}

	#[test]
	fn a_world_says_how_far_a_tree_carries_a_pose_without_binding_anything() {
		let mut world = World::new();
		let rig = world.skeletons.insert("walker", walker());
		let clip = world.clips.insert("stride", striding());
		let pose = world
			.poses
			.spawn(Pose::resting(rig, world.skeletons.bones(rig)));
		let tree = carried_by_root(clip, 0.25, 0.75, false);
		let went = world.travel(pose, &tree);

		assert!(
			went.moved
				.abs_diff_eq(Vec3::new(1.5, 0.0, -0.03), 1.0e-5),
			"{}",
			went.moved
		);
		assert_eq!(
			world.clips.bindings(),
			0,
			"asked through a shared borrow, so nothing was bound and nothing had to be"
		);
		assert!(world.animate(pose, &tree), "the pose is there");

		let root = world.poses.get(pose).expect("still there").locals[0];

		assert!(
			root.position
				.abs_diff_eq(Vec3::new(0.25, 0.93, -0.5), 1.0e-6),
			"and working the tree out into the pose pins the root over its rest: {}",
			root.position
		);

		world.poses.despawn(pose);

		assert_eq!(
			world.travel(pose, &tree),
			Travel::NONE,
			"a pose that is gone is carried nowhere"
		);
	}

	#[test]
	fn a_prediction_replaying_its_commands_twice_is_carried_the_same_way_twice() {
		// what a client does after a correction: run the commands the host has
		// not confirmed again, from the same state. The travel is asked from
		// inside the replay, which holds the world, and it keeps nothing between
		// two calls, so the second run cannot tell it is the second. The run
		// crosses the clip's seam on the way.
		let mut world = World::new();
		let rig = world.skeletons.insert("walker", walker());
		let clip = world.clips.insert("curve", curving());
		let pose = world
			.poses
			.spawn(Pose::resting(rig, world.skeletons.bones(rig)));
		let commands: Vec<Command> = (0..48_u32)
			.map(|number| Command {
				step: 100 + u64::from(number) * 2,
				number,
				buttons: 0,
				yaw: 0.0,
				pitch: 0.0,
			})
			.collect();
		let start =
			Motion::new(Vec3::new(4.0, 0.0, -3.0), Vec3::ZERO, Vec3::splat(0.3), 1.0 / 60.0);
		let facing = Quat::from_rotation_y(0.4);
		let run = || {
			let (mut clock, mut toward) = (0.1_f32, facing);
			let moved =
				character::replay(&world, &start, 98, &commands, |_command, _before, motion| {
					let went = world
						.travel(pose, &carried_by_root(clip, clock, clock + motion.dt, true));
					let there =
						went.applied(Transform { rotation: toward, ..Transform::IDENTITY });

					motion.velocity = there.position / motion.dt;
					toward = there.rotation;
					clock += motion.dt;
				});

			(moved.position, toward, clock)
		};
		let (once, twice) = (run(), run());

		assert!(once.0.abs_diff_eq(twice.0, 0.0), "to the bit: {} against {}", once.0, twice.0);
		assert!(once.1.abs_diff_eq(twice.1, 0.0), "and facing the same way to the bit");
		assert!(once.2 > 1.0, "the run crossed the seam, at {}", once.2);

		let whole = world.travel(pose, &carried_by_root(clip, 0.1, once.2, true));
		let there = whole.applied(Transform {
			position: start.position,
			rotation: facing,
			scale: Vec3::ONE,
		});

		assert!(
			once.0.abs_diff_eq(there.position, 1.0e-3),
			"and it went where one travel over the whole run says: {} against {}",
			once.0,
			there.position
		);
	}
}
