//! Sounds: the samples a noise is made of, and the host's registry of them.
//!
//! Here rather than in the mixer for the same reason meshes and textures are in
//! `colby_core` rather than in the renderer: [`SoundId`] crosses the boundary,
//! and a handle means nothing away from the table it indexes. @ref
//! [`registry`](super::registry) for the shape all five tables share.
//!
//! **Nothing in this module reads a `.wav`.** A sound arrives already decoded:
//! the asset compiler turns whatever a recorder wrote into interleaved
//! sixteen-bit samples, and what reaches here is the result. That is the same
//! bargain textures and fonts made - the decoder lands offline, and neither the
//! engine nor the runner links an audio library to open a file.
//!
//! **Sixteen bits, at the file's own rate.** Sixteen because that is what every
//! engine stores and because a ten-second stereo sound is 1.7 megabytes rather
//! than 3.5; the mixer widens to `f32` on the way past, which is one multiply
//! per sample. The file's own rate rather than one fixed rate because the
//! device's rate is not knowable when the file is compiled - and a mixer that
//! steps a fractional index has to exist anyway, since that is also what
//! playing something at another pitch is.
//!
//! **Everything is held whole in memory.** There is no streaming and no
//! compression, which is why [`MAX_SAMPLES`] is what it is: about three minutes
//! of stereo at forty-eight kilohertz. The thing that forces a codec is music,
//! and there is no music yet.
//!
//! The other half of this module is [`Voices`], which is a *playing* sound
//! rather than a recorded one: bounded, generational, and shaped exactly like
//! the body table. The one rule worth reading before anything else here is
//! whose clock it is on. **A voice is advanced by the simulation step**, by
//! `dt` a step, so a one-second sound ends after exactly sixty steps on every
//! machine and a screenshot taken at step ninety is the same picture of the
//! same world. A mixer running on a device's own thread is downstream of that
//! and is allowed to drift from it by a block; it is not allowed to decide
//! when a voice ends, because then gameplay would depend on how long a driver
//! took.
//!
//! **Nothing here is written into a `.cscene`.** A voice is a moment rather
//! than a thing, and the one case that would want saving - ambience that
//! loops - is better started again by whatever started it the first time. What
//! a restore *does* owe is stopping everything, because the handles in the
//! arena it just put back address a table it did not.

use super::{
	camera::Camera,
	registry::{Entry, Registry},
};
use crate::{glam::Vec3, registry_handle};

/// The most samples one sound may hold, counting every channel.
///
/// About three minutes of stereo at forty-eight kilohertz, or thirty-two
/// megabytes. This is the number that says the engine does not stream: a sound
/// is decompressed once and then held, so the limit is a limit on memory rather
/// than on length as such.
pub const MAX_SAMPLES: usize = 1 << 24;

/// The most channels one sound may have.
///
/// Mono or stereo. A positioned sound is mixed down to one channel anyway, and
/// nothing here knows what to do with the fifth speaker of a file that has one.
pub const MAX_CHANNELS: u16 = 2;

/// The slowest sample rate that is taken seriously.
pub const MIN_RATE: u32 = 1000;

/// The fastest.
///
/// The same ceiling the one engine that resamples on import uses.
pub const MAX_RATE: u32 = 192_000;

/// The rate an empty sound reports, so that nothing divides by nothing.
pub const DEFAULT_RATE: u32 = 48_000;

registry_handle! {
	/// Which sound in [`Sounds`].
	///
	/// Not generational, like every other resource handle here: a sound
	/// resolved by name in `init` stays valid for the life of the process, and
	/// recompiling the source rewrites the entry the handle already points at.
	/// @ref [`registry`](super::registry).
	SoundId
}

/// One entry of the sound registry.
pub type Sound = Entry<SoundData>;

/// A decoded sound: interleaved samples and what they mean.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoundData {
	/// Every sample, interleaved, [`channels`](Self::channels) to a frame.
	///
	/// A stereo sound runs left, right, left, right. The length is always a
	/// whole number of frames, which [`check`](Self::check) is what enforces.
	pub samples: Vec<i16>,

	/// Frames a second, as the file was recorded.
	pub rate: u32,

	/// How many samples make one frame: one for mono, two for stereo.
	pub channels: u16,
}

impl SoundData {
	/// A sound with nothing in it.
	///
	/// What slot zero holds, and what a name nobody compiled resolves to. It
	/// plays as silence and ends immediately rather than being a handle that
	/// resolves to nothing, which is the same discipline `MeshId::NONE` and
	/// the empty font follow.
	#[must_use]
	pub const fn silence() -> Self {
		Self {
			samples: Vec::new(),
			rate: DEFAULT_RATE,
			channels: 1,
		}
	}

	/// How many frames there are, whatever the channel count is.
	#[must_use]
	pub fn frames(&self) -> usize {
		if self.channels == 0 {
			return 0;
		}

		self.samples.len() / usize::from(self.channels)
	}

	/// How long it lasts, in seconds.
	///
	/// The number a voice's playhead is measured against, so it is worked out
	/// the same way everywhere rather than at each call site.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		clippy::cast_precision_loss,
		reason = "a frame count past what an f32 counts exactly is past MAX_SAMPLES, which is \
		          refused, and a rate is at most MAX_RATE"
	)]
	pub fn seconds(&self) -> f32 {
		if self.rate == 0 {
			return 0.0;
		}

		self.frames() as f32 / self.rate as f32
	}

	/// Whether there is anything to play.
	#[must_use]
	pub const fn is_empty(&self) -> bool { self.samples.is_empty() }

	/// Everything that has to hold before this is a sound rather than numbers.
	///
	/// One definition, called by whatever builds a [`SoundData`] - the
	/// importer on the way in and the format writer on the way out - so that
	/// the two cannot come to different conclusions about the same file.
	///
	/// @return nothing, or why these are not samples anybody can play
	///
	/// # Errors
	///
	/// If the channel count, the rate or the length is one nothing downstream
	/// can work with.
	pub fn check(&self) -> Result<(), String> {
		if self.channels == 0 || self.channels > MAX_CHANNELS {
			return Err(format!(
				"has {} channels, and a sound here is mono or stereo",
				self.channels
			));
		}

		if self.rate < MIN_RATE || self.rate > MAX_RATE {
			return Err(format!(
				"is recorded at {} frames a second, outside the {MIN_RATE} to {MAX_RATE} a \
				 sound may be",
				self.rate
			));
		}

		if self.samples.len() > MAX_SAMPLES {
			return Err(format!(
				"holds {} samples, past the {MAX_SAMPLES} one sound may have; nothing here \
				 streams, so a sound is held whole",
				self.samples.len()
			));
		}

		if !self
			.samples
			.len()
			.is_multiple_of(usize::from(self.channels))
		{
			return Err(format!(
				"holds {} samples, which is not a whole number of {}-channel frames",
				self.samples.len(),
				self.channels
			));
		}

		Ok(())
	}
}

impl Default for SoundData {
	fn default() -> Self { Self::silence() }
}

/// Every sound the mixer can play, addressed by [`SoundId`].
///
/// Slot zero is [`SoundId::NONE`] and is [`SoundData::silence`], so a game
/// naming a sound nobody compiled plays nothing rather than reaching for an
/// entry that is not there.
#[derive(Clone, Debug)]
pub struct Sounds {
	entries: Registry<SoundData>,
}

impl Sounds {
	/// A registry holding nothing but silence.
	#[must_use]
	pub fn new() -> Self {
		Self {
			entries: Registry::new(SoundData::silence()),
		}
	}

	/// Looks a sound up by name.
	///
	/// @param name - what the compiler registered it as, e.g. `sounds/thud`
	/// @return its handle, or [`SoundId::NONE`] if nothing answers to it
	#[must_use]
	pub fn find(&self, name: &str) -> SoundId { SoundId::new(self.entries.find(name)) }

	/// Registers a decoded sound under a name, replacing whatever was there.
	///
	/// @return the handle, the same one as last time if the name is known
	pub fn insert(&mut self, name: &str, data: SoundData) -> SoundId {
		SoundId::new(self.entries.insert(name, data))
	}

	/// One sound, by handle.
	#[must_use]
	pub fn get(&self, id: SoundId) -> Option<&Sound> { self.entries.entry(id.index()) }

	/// One sound's samples, by handle, falling back to silence.
	///
	/// What the mixer calls: it has nothing useful to do about a handle from
	/// another table, and would rather play nothing than branch.
	///
	/// @note: no second look at slot zero, unlike the font table's version of
	/// this. Slot zero is unreachable - `insert` never returns it, because the
	/// empty name is not a name - so it always holds what it was built with,
	/// and a fall back to it and a fall back to [`SILENCE`] are the same
	/// answer. The line was written and taken out again because no mutation of
	/// it could fail a test.
	#[must_use]
	pub fn data(&self, id: SoundId) -> &SoundData {
		self.entries
			.entry(id.index())
			.map_or(&SILENCE, Entry::value)
	}

	/// How many sounds there are, counting the null one.
	#[must_use]
	pub fn len(&self) -> usize { self.entries.len() }

	/// Always `false`: slot zero always exists.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.entries.is_empty() }

	/// Every sound, in slot order, starting with the null one.
	pub fn iter(&self) -> impl Iterator<Item = &Sound> { self.entries.iter() }
}

impl Default for Sounds {
	fn default() -> Self { Self::new() }
}

/// The most sounds that may be playing at once.
///
/// Bounded like every other table here, and refusing rather than stealing: a
/// full table hands back [`VoiceId::NONE`] and counts the refusal, which is
/// what `Bodies::spawn` does. The engine one field over cuts off its oldest
/// sound instead; that is a policy a game can implement over this and cannot
/// undo if the table implements it first.
pub const MAX_VOICES: usize = 64;

/// The slowest anything may be played back, as a multiple of its own rate.
///
/// Not zero: a voice that does not advance never reaches its end, so it holds
/// a slot until somebody remembers it. A game that wants a sound to stop
/// stops it.
pub const MIN_PITCH: f32 = 0.05;

/// The fastest.
pub const MAX_PITCH: f32 = 8.0;

/// The loudest a single voice may ask to be.
///
/// Above one on purpose: a recording made quiet is made loud here rather than
/// re-exported. What stops the sum of sixty-four of these from tearing the
/// speakers is the mixer, which is the only place that can see the sum.
pub const MAX_VOLUME: f32 = 4.0;

/// How far from a positioned voice it is still at full volume, by default.
pub const DEFAULT_REFERENCE: f32 = 1.0;

/// How far away it stops getting any quieter, by default.
pub const DEFAULT_MAXIMUM: f32 = 100.0;

/// Which set of sounds a voice belongs to, and therefore which volume it is
/// scaled by.
///
/// Three, which is the shape the one engine that writes its own thin layer
/// over the system mixer has: sound effects, music, and a spare. A graph of
/// buses with effects on them is the other end of this and is not what a
/// sandbox needs to make a noise when a crate lands.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Category {
	/// Something in the world. Almost everything.
	#[default]
	Effect,

	/// Something that is not in the world and should not duck with it.
	Music,

	/// A click, a beep, a menu. Never positioned.
	Interface,
}

/// How loud each [`Category`] is, and everything together.
///
/// Game-written, like `light` and `ambient` and unlike anything the mixer
/// derives, and written by the host's console variables too - the same
/// arrangement `gravity` has, and for the same reason: whoever said something
/// last wins, and standing still says nothing.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mix {
	/// Scales everything, whatever else it is scaled by.
	pub master: f32,

	/// Scales [`Category::Effect`].
	pub effects: f32,

	/// Scales [`Category::Music`].
	pub music: f32,

	/// Scales [`Category::Interface`].
	pub interface: f32,
}

impl Mix {
	/// Everything at full volume.
	pub const FULL: Self = Self {
		master: 1.0,
		effects: 1.0,
		music: 1.0,
		interface: 1.0,
	};

	/// What one category is scaled by, master included.
	///
	/// @param category - which set of sounds
	/// @return the multiplier, never negative
	#[must_use]
	pub fn of(&self, category: Category) -> f32 {
		let own = match category {
			| Category::Effect => self.effects,
			| Category::Music => self.music,
			| Category::Interface => self.interface,
		};

		(self.master * own).max(0.0)
	}
}

impl Default for Mix {
	fn default() -> Self { Self::FULL }
}

/// Where the world is being heard from.
///
/// Game-written, like the camera, and deliberately *not* derived from it. A
/// game whose camera is a spectator's or an orbit control's would otherwise
/// have the world heard from wherever somebody happened to be looking, and a
/// default that quietly means something else is worse than one line in
/// `update`. @ref [`Listener::at_camera`] for that line.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Listener {
	/// Where the ears are, in world space.
	pub at: Vec3,

	/// Which way they face. Need not be normalized.
	pub forward: Vec3,

	/// Which way is up, so that left and right can be told apart.
	pub up: Vec3,
}

impl Listener {
	/// At the origin, facing down negative z with y up.
	pub const DEFAULT: Self = Self {
		at: Vec3::ZERO,
		forward: Vec3::new(0.0, 0.0, -1.0),
		up: Vec3::Y,
	};

	/// The listener a camera implies.
	///
	/// @param camera - where the renderer is looking from
	/// @return a listener at the camera, facing what it faces
	#[must_use]
	pub fn at_camera(camera: &Camera) -> Self {
		Self {
			at: camera.position,
			forward: camera.target - camera.position,
			up: camera.up,
		}
	}

	/// The listener's own axes, right-handed and orthonormal.
	///
	/// @return `(right, up, forward)`, falling back to
	/// [`DEFAULT`](Self::DEFAULT)'s where the two given vectors say nothing -
	/// a zero-length forward, or an up parallel to it. A panner that divided
	/// by that would send everything to one ear.
	#[must_use]
	pub fn frame(&self) -> (Vec3, Vec3, Vec3) {
		let forward = self
			.forward
			.try_normalize()
			.unwrap_or(Vec3::NEG_Z);
		let right = forward
			.cross(self.up)
			.try_normalize()
			.unwrap_or_else(|| {
				forward
					.cross(Vec3::Y)
					.try_normalize()
					.unwrap_or(Vec3::X)
			});

		(right, right.cross(forward), forward)
	}
}

impl Default for Listener {
	fn default() -> Self { Self::DEFAULT }
}

/// A handle to a playing sound.
///
/// Generational, like [`BodyId`](super::physics::BodyId) and unlike a resource
/// handle: a voice ends and its slot is reused, so a handle kept across that
/// has to fail rather than reach whoever moved in. A game that starts a
/// footstep and forgets the handle is the common case and is what the
/// generation makes safe.
///
/// `Pod` for the same reason a body handle is: a game keeps its handles in the
/// arena, and a zeroed arena has to read back as [`VoiceId::NONE`].
#[repr(C)]
#[derive(
	Clone,
	Copy,
	Debug,
	Default,
	PartialEq,
	Eq,
	Hash,
	crate::bytemuck::Pod,
	crate::bytemuck::Zeroable,
)]
pub struct VoiceId {
	index: u32,
	generation: u32,
}

impl VoiceId {
	/// A handle that refers to nothing, and always will.
	pub const NONE: Self = Self { index: 0, generation: 0 };

	/// Whether this handle could refer to anything at all.
	///
	/// A `true` here does not mean the voice is still playing - only
	/// [`Voices::alive`] answers that.
	#[must_use]
	pub const fn is_some(self) -> bool { self.generation != 0 }

	/// The slot this addresses, whatever is in it now.
	///
	/// The mixer's, for keying its own per-slot playhead. Paired with
	/// [`generation`](Self::generation), which is how it notices that the slot
	/// is somebody else's sound now and starts it from the beginning.
	#[must_use]
	pub const fn slot(self) -> u32 { self.index }

	/// Which occupant of that slot this handle is for.
	#[must_use]
	pub const fn generation(self) -> u32 { self.generation }

	/// The handle for one slot, whatever is in it.
	///
	/// **Public, and it is not a way to be handed something**, for the reason
	/// [`EntityId::at`](super::EntityId::at) is: this type is `Pod`, so any
	/// crate could already turn eight bytes into one, and what this adds is a
	/// spelling for it - which is what anything carrying a handle outside Rust
	/// needs. A voice resolves only where the table agrees one is playing.
	/// @ref [`Voices::alive`].
	///
	/// @param index - the slot
	/// @param generation - which occupant of it
	#[must_use]
	pub const fn at(index: u32, generation: u32) -> Self { Self { index, generation } }
}

/// One sound being played.
///
/// `#[repr(C)] + Copy` and deliberately **not** `Pod`: glam is built without
/// bytemuck, so nothing holding a `Vec3` can be, and only the arena needs
/// `Pod` anyway. Same as `TraceInfo`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Voice {
	/// What is playing. [`SoundId::NONE`] is silence and ends at once.
	pub sound: SoundId,

	/// Where it is, in world space. Meaningless unless
	/// [`positioned`](Self::positioned).
	pub at: Vec3,

	/// How loud, before the category and the distance. Clamped to
	/// `0 ..= MAX_VOLUME` where it is read. @ref [`Voice::gain`].
	pub volume: f32,

	/// How fast, as a multiple of the recording's own rate. Clamped to
	/// `MIN_PITCH ..= MAX_PITCH` where it is read. @ref [`Voice::speed`].
	pub pitch: f32,

	/// How far into the recording it is, in seconds of the recording.
	///
	/// Written by the host every step, and readable by a game that wants to
	/// know how far along something is. Writing it is a seek.
	pub head: f32,

	/// How close it has to be to play at full volume.
	pub reference: f32,

	/// How far away it stops getting any quieter.
	///
	/// Not a cutoff: past this the volume holds at `reference / maximum`
	/// rather than falling to nothing, which is what the one distance model
	/// with a written specification does. A game that wants silence past a
	/// distance stops the voice.
	pub maximum: f32,

	/// Which volume scales it.
	pub category: Category,

	/// Whether it starts over when it reaches the end.
	pub looping: bool,

	/// Whether it is somewhere in the world rather than simply audible.
	pub positioned: bool,
}

impl Voice {
	/// A sound that is simply audible, with no place in the world.
	///
	/// Music, a menu click, a line of speech nobody is standing next to.
	///
	/// @param sound - what to play
	#[must_use]
	pub const fn flat(sound: SoundId) -> Self {
		Self {
			sound,
			at: Vec3::ZERO,
			volume: 1.0,
			pitch: 1.0,
			head: 0.0,
			reference: DEFAULT_REFERENCE,
			maximum: DEFAULT_MAXIMUM,
			category: Category::Effect,
			looping: false,
			positioned: false,
		}
	}

	/// A sound coming from somewhere.
	///
	/// @param sound - what to play
	/// @param at - where it is, in world space
	#[must_use]
	pub const fn at(sound: SoundId, at: Vec3) -> Self {
		Self {
			at,
			positioned: true,
			..Self::flat(sound)
		}
	}

	/// The same voice, playing over and over.
	#[must_use]
	pub const fn looping(self) -> Self { Self { looping: true, ..self } }

	/// The same voice, at another volume.
	#[must_use]
	pub const fn volume(self, volume: f32) -> Self { Self { volume, ..self } }

	/// The same voice, at another speed.
	#[must_use]
	pub const fn pitch(self, pitch: f32) -> Self { Self { pitch, ..self } }

	/// The same voice, in another category.
	#[must_use]
	pub const fn category(self, category: Category) -> Self { Self { category, ..self } }

	/// The same voice, carrying further or less far.
	///
	/// @param reference - how close it is at full volume
	/// @param maximum - how far away it stops getting quieter
	#[must_use]
	pub const fn range(self, reference: f32, maximum: f32) -> Self {
		Self { reference, maximum, ..self }
	}

	/// How loud it actually is before anything else scales it.
	#[must_use]
	pub fn gain(&self) -> f32 { self.volume.clamp(0.0, MAX_VOLUME) }

	/// How fast the playhead actually moves.
	#[must_use]
	pub fn speed(&self) -> f32 {
		if self.pitch.is_nan() {
			return 1.0;
		}

		self.pitch.clamp(MIN_PITCH, MAX_PITCH)
	}

	/// How much of the distance is left before it is heard at all.
	///
	/// The inverse distance law, clamped at both ends:
	/// `reference / clamp(distance, reference, maximum)`. That is the
	/// specified model with a rolloff factor of one, which is what every
	/// engine checked either uses or approximates - and with the factor fixed
	/// there is one knob fewer to explain, because moving `reference` does the
	/// same job.
	///
	/// @param from - where the listener is
	/// @return one at the reference distance and closer, falling away by the
	/// inverse of the distance after it, and holding at `reference / maximum`
	/// past that. Always one for a voice with no place in the world.
	#[must_use]
	pub fn attenuation(&self, from: Vec3) -> f32 {
		if !self.positioned {
			return 1.0;
		}

		let reference = self.reference.max(f32::MIN_POSITIVE);
		let maximum = self.maximum.max(reference);

		reference / self.at.distance(from).clamp(reference, maximum)
	}
}

impl Default for Voice {
	fn default() -> Self { Self::flat(SoundId::NONE) }
}

/// Every sound being played, reached by handle.
///
/// Bounded and generational, exactly like the body table and for the same
/// reasons. **The playhead in here is on the simulation's clock, not the
/// device's**: [`advance`](Self::advance) moves it by `dt` a step, so a
/// one-second sound ends after exactly sixty steps on every machine and a game
/// asking whether something has finished gets an answer a screenshot can
/// reproduce. What a mixer does with the samples is downstream of this and may
/// drift from it by a block; nothing above cares, and nothing below is allowed
/// to decide when a voice ends.
#[derive(Clone, Debug)]
pub struct Voices {
	voices: Vec<Voice>,
	generations: Vec<u32>,
	alive: Vec<bool>,
	free: Vec<u32>,
	live: usize,
	refused: u32,
}

impl Voices {
	/// A table with nothing playing.
	#[must_use]
	pub const fn new() -> Self {
		Self {
			voices: Vec::new(),
			generations: Vec::new(),
			alive: Vec::new(),
			free: Vec::new(),
			live: 0,
			refused: 0,
		}
	}

	/// Starts a sound.
	///
	/// @param voice - what to play, and how
	/// @return its handle, or [`VoiceId::NONE`] if the table is full, in which
	/// case the refusal is counted. A game that does not care about stopping
	/// what it started may drop the handle.
	pub fn play(&mut self, voice: Voice) -> VoiceId {
		let Some(slot) = self.take_slot() else {
			self.refused = self.refused.saturating_add(1);

			return VoiceId::NONE;
		};

		let Ok(index) = u32::try_from(slot) else {
			return VoiceId::NONE;
		};

		self.generations[slot] = self.generations[slot].saturating_add(1);
		self.alive[slot] = true;
		self.voices[slot] = voice;
		self.live += 1;

		VoiceId {
			index,
			generation: self.generations[slot],
		}
	}

	/// Stops one.
	///
	/// @param id - the handle to stop
	/// @return `true` if it was playing, `false` if the handle was stale
	pub fn stop(&mut self, id: VoiceId) -> bool {
		let Some(slot) = self.slot(id) else {
			return false;
		};

		self.release(slot);

		true
	}

	/// Stops everything.
	///
	/// What a game calls from `init` when the arena came back fresh: the
	/// handles it was holding are gone with the arena, so anything still
	/// looping would play until the process ended with nothing able to name
	/// it. The host calls it too, when a saved world is put back.
	pub fn stop_all(&mut self) {
		for slot in 0..self.alive.len() {
			if self.alive[slot] {
				self.release(slot);
			}
		}
	}

	/// Moves every playhead on by one step, and frees whatever has finished.
	///
	/// Called at the *top* of the step, beside the debug table's sweep and for
	/// a related reason: a voice started during a step should have been
	/// audible for none of it by the time the step ends, and one started
	/// during the previous step should have been audible for exactly one.
	///
	/// @note: the playhead is a running sum of `dt`, so it drifts from
	/// `steps * dt` by whatever f32 addition costs - a one-second sound ends
	/// on the sixty-first step rather than the sixtieth, because sixty
	/// sixtieths add up to 0.9999997. That is deterministic, which is the
	/// property this is here for; it is not exact, and nothing should be
	/// written that needs it to be.
	///
	/// @param sounds - the registry, for how long each recording is
	/// @param dt - how long a step is, in seconds
	pub fn advance(&mut self, sounds: &Sounds, dt: f32) {
		// reset here rather than at the bottom, so that what `dropped` reports
		// is what this step refused. The same arrangement `Debug::dropped` has.
		self.refused = 0;

		for slot in 0..self.alive.len() {
			if !self.alive[slot] {
				continue;
			}

			let voice = self.voices[slot];
			let length = sounds.data(voice.sound).seconds();

			// a recording of no length is over before it starts, whatever it
			// was asked to do. Without this a looping voice on a sound nobody
			// compiled holds a slot for the life of the process.
			if length <= 0.0 {
				self.release(slot);

				continue;
			}

			let head = dt.mul_add(voice.speed(), voice.head);
			if head < length {
				self.voices[slot].head = head;

				continue;
			}

			if voice.looping {
				self.voices[slot].head = head.rem_euclid(length);

				continue;
			}

			self.release(slot);
		}
	}

	/// Whether a handle refers to something still playing.
	#[must_use]
	pub fn alive(&self, id: VoiceId) -> bool { self.slot(id).is_some() }

	/// One voice.
	#[must_use]
	pub fn get(&self, id: VoiceId) -> Option<&Voice> {
		self.slot(id).map(|slot| &self.voices[slot])
	}

	/// One voice, to change.
	///
	/// How a looping sound is moved: a game writes `at` every step for as long
	/// as the thing making the noise is moving.
	pub fn get_mut(&mut self, id: VoiceId) -> Option<&mut Voice> {
		self.slot(id).map(|slot| &mut self.voices[slot])
	}

	/// Every voice still playing, with its handle.
	pub fn iter(&self) -> impl Iterator<Item = (VoiceId, &Voice)> {
		self.voices
			.iter()
			.enumerate()
			.filter(|&(slot, _)| self.alive[slot])
			.filter_map(|(slot, voice)| {
				let index = u32::try_from(slot).ok()?;

				Some((
					VoiceId {
						index,
						generation: self.generations[slot],
					},
					voice,
				))
			})
	}

	/// How many are playing.
	#[must_use]
	pub const fn len(&self) -> usize { self.live }

	/// Whether nothing is.
	#[must_use]
	pub const fn is_empty(&self) -> bool { self.live == 0 }

	/// How many slots the table has ever handed out.
	///
	/// The mixer's, for sizing its own per-slot state.
	#[must_use]
	pub fn slots(&self) -> usize { self.alive.len() }

	/// How many sounds were refused during the last step, for want of a slot.
	///
	/// Reset by [`advance`](Self::advance), so it is about one step rather
	/// than about the run. A number that is never zero is a game that wants
	/// fewer sounds or a bigger table.
	#[must_use]
	pub const fn dropped(&self) -> u32 { self.refused }

	/// Empties a slot.
	fn release(&mut self, slot: usize) {
		self.alive[slot] = false;
		self.voices[slot] = Voice::default();
		if let Ok(index) = u32::try_from(slot) {
			self.free.push(index);
		}

		self.live = self.live.saturating_sub(1);
	}

	/// The slot a handle addresses, if it is still the voice it was.
	fn slot(&self, id: VoiceId) -> Option<usize> {
		if !id.is_some() {
			return None;
		}

		let slot = usize::try_from(id.index).ok()?;

		(self.alive.get(slot) == Some(&true)
			&& self.generations.get(slot) == Some(&id.generation))
		.then_some(slot)
	}

	/// A free slot, reused or newly grown.
	fn take_slot(&mut self) -> Option<usize> {
		if let Some(index) = self.free.pop() {
			return usize::try_from(index).ok();
		}

		if self.voices.len() >= MAX_VOICES {
			return None;
		}

		self.voices.push(Voice::default());
		self.generations.push(0);
		self.alive.push(false);

		Some(self.voices.len() - 1)
	}
}

impl Default for Voices {
	fn default() -> Self { Self::new() }
}

/// What [`Sounds::data`] hands back when even slot zero is missing.
///
/// It cannot be, but the alternative is an `unwrap` in the one call the mixer
/// makes per voice per block.
static SILENCE: SoundData = SoundData {
	samples: Vec::new(),
	rate: DEFAULT_RATE,
	channels: 1,
};

#[cfg(test)]
mod tests {
	use super::*;

	/// A second of stereo at a rate that divides evenly.
	fn beep() -> SoundData {
		SoundData {
			samples: vec![0; 2000],
			rate: 1000,
			channels: 2,
		}
	}

	#[test]
	fn a_sound_reports_its_length_from_its_samples_and_its_rate() {
		let sound = beep();

		assert_eq!(sound.frames(), 1000, "two samples to a frame");
		assert!(
			(sound.seconds() - 1.0).abs() < 1e-6,
			"a thousand frames at a thousand a second is a second"
		);
		assert!(!sound.is_empty(), "and there is something in it");
	}

	#[test]
	fn silence_is_empty_without_being_a_sound_of_no_rate() {
		let quiet = SoundData::silence();

		assert!(quiet.is_empty(), "there is nothing to play");
		assert_eq!(quiet.frames(), 0, "not one frame");
		assert!(
			(quiet.seconds() - 0.0).abs() < f32::EPSILON,
			"and it lasts no time at all rather than dividing by nothing"
		);
		assert_eq!(quiet.rate, DEFAULT_RATE, "the rate is still a number the mixer can use");
	}

	#[test]
	fn a_well_formed_sound_passes_the_check_and_the_broken_ones_say_why() {
		beep()
			.check()
			.expect("a second of stereo is a sound");

		let cases = [
			(SoundData { channels: 0, ..beep() }, "mono or stereo"),
			(SoundData { channels: 6, ..beep() }, "mono or stereo"),
			(SoundData { rate: 0, ..beep() }, "frames a second"),
			(SoundData { rate: MAX_RATE + 1, ..beep() }, "frames a second"),
			(SoundData { samples: vec![0; 999], ..beep() }, "whole number"),
			(
				SoundData {
					samples: vec![0; MAX_SAMPLES + 2],
					..beep()
				},
				"streams",
			),
		];

		for (sound, expected) in cases {
			let message = sound
				.check()
				.expect_err("this one is not a sound anybody can play");

			assert!(
				message.contains(expected),
				"the message names what is wrong: wanted {expected:?}, got {message:?}"
			);
		}
	}

	#[test]
	fn a_sound_too_long_to_hold_is_measured_before_it_is_refused() {
		// the length check has to come before the frame check, or a file that
		// is both too long and ragged reports the cheaper complaint and
		// somebody fixes the wrong thing.
		let ragged = SoundData {
			samples: vec![0; MAX_SAMPLES + 1],
			..beep()
		};

		assert!(
			ragged
				.check()
				.expect_err("it is too long")
				.contains("streams"),
			"length is the complaint, not the odd sample"
		);
	}

	#[test]
	fn slot_zero_is_silence_and_answers_to_nothing() {
		let sounds = Sounds::new();

		assert_eq!(sounds.len(), 1, "a new table holds only its null entry");
		assert!(!sounds.is_empty(), "which is why it is never empty");
		assert_eq!(sounds.find("sounds/thud"), SoundId::NONE, "nothing answers to that yet");
		assert!(sounds.data(SoundId::NONE).is_empty(), "and the null sound plays nothing");
	}

	#[test]
	fn a_name_takes_a_slot_and_keeps_it_across_a_recompile() {
		let mut sounds = Sounds::new();
		let first = sounds.insert("sounds/thud", beep());
		let again = sounds.insert("sounds/thud", SoundData { rate: 2000, ..beep() });

		assert_eq!(first, again, "the handle a game kept still points at the same slot");
		assert_eq!(sounds.data(first).rate, 2000, "and now at the new samples");
		assert_eq!(
			sounds.get(first).map(Entry::revision),
			Some(1),
			"which is what tells the mixer to start the voice over"
		);
	}

	/// A registry holding one sound of exactly one second.
	fn one_second() -> (Sounds, SoundId) {
		let mut sounds = Sounds::new();
		let id = sounds.insert("sounds/second", SoundData {
			samples: vec![0; 1000],
			rate: 1000,
			channels: 1,
		});

		(sounds, id)
	}

	/// Runs the top of a step, the number of times asked.
	fn steps(voices: &mut Voices, sounds: &Sounds, count: u32) {
		for _ in 0..count {
			voices.advance(sounds, crate::time::STEP_SECONDS);
		}
	}

	#[test]
	fn a_category_is_scaled_by_its_own_volume_and_by_the_master() {
		let mix = Mix {
			master: 0.5,
			effects: 0.5,
			music: 1.0,
			interface: 0.0,
		};

		assert!((mix.of(Category::Effect) - 0.25).abs() < 1e-6, "both, multiplied");
		assert!((mix.of(Category::Music) - 0.5).abs() < 1e-6, "the master alone");
		assert!((mix.of(Category::Interface) - 0.0).abs() < f32::EPSILON, "off is off");
		assert!(
			(Mix::FULL.of(Category::Effect) - 1.0).abs() < f32::EPSILON,
			"and everything at once is everything"
		);
	}

	#[test]
	fn a_negative_volume_is_silence_rather_than_a_sound_turned_inside_out() {
		let mix = Mix { master: -2.0, ..Mix::FULL };

		assert!(
			mix.of(Category::Effect) >= 0.0,
			"a negative multiplier would invert the waveform, which is not what anybody typing \
			 \\
			 a negative number meant"
		);
	}

	#[test]
	fn a_listener_takes_a_cameras_place_and_its_aim() {
		let camera = Camera {
			position: Vec3::new(1.0, 2.0, 3.0),
			target: Vec3::new(1.0, 2.0, 8.0),
			..Camera::DEFAULT
		};
		let listener = Listener::at_camera(&camera);

		assert_eq!(listener.at, camera.position, "the ears are where the eye is");
		assert!(
			listener
				.forward
				.abs_diff_eq(Vec3::new(0.0, 0.0, 5.0), 1e-6),
			"and face what it looks at, unnormalized"
		);

		let (right, up, forward) = listener.frame();

		assert!(forward.abs_diff_eq(Vec3::Z, 1e-6), "which normalizes to plain z");
		assert!(right.abs_diff_eq(Vec3::NEG_X, 1e-6), "so right is the other way");
		assert!(up.abs_diff_eq(Vec3::Y, 1e-6), "and up is still up");
	}

	#[test]
	fn a_listener_facing_nowhere_still_has_three_axes() {
		// a panner dividing by any of these would send everything to one ear,
		// so each degenerate case has to come back with a frame rather than
		// with a zero.
		let cases = [
			Listener { forward: Vec3::ZERO, ..Listener::DEFAULT },
			Listener { up: Vec3::ZERO, ..Listener::DEFAULT },
			Listener {
				forward: Vec3::Y,
				up: Vec3::Y,
				..Listener::DEFAULT
			},
		];

		for listener in cases {
			let (right, up, forward) = listener.frame();

			for (name, axis) in [("right", right), ("up", up), ("forward", forward)] {
				assert!(
					(axis.length() - 1.0).abs() < 1e-5,
					"{name} came back {axis:?}, which is not a direction"
				);
			}

			assert!(right.dot(forward).abs() < 1e-5, "and the frame is square");
			assert!(up.dot(forward).abs() < 1e-5, "in both the other pairs");
			assert!(right.dot(up).abs() < 1e-5);
		}
	}

	#[test]
	fn a_sound_gets_quieter_by_the_inverse_of_the_distance() {
		let voice = Voice::at(SoundId::NONE, Vec3::ZERO).range(2.0, 20.0);

		assert!(
			(voice.attenuation(Vec3::ZERO) - 1.0).abs() < 1e-6,
			"standing on it is full volume"
		);
		assert!(
			(voice.attenuation(Vec3::X) - 1.0).abs() < 1e-6,
			"and so is anywhere inside the reference distance"
		);
		assert!(
			(voice.attenuation(Vec3::X * 4.0) - 0.5).abs() < 1e-6,
			"twice the reference distance is half as loud"
		);
		assert!(
			(voice.attenuation(Vec3::X * 8.0) - 0.25).abs() < 1e-6,
			"and four times is a quarter"
		);
	}

	#[test]
	fn past_the_maximum_a_sound_holds_rather_than_stopping() {
		let voice = Voice::at(SoundId::NONE, Vec3::ZERO).range(2.0, 20.0);
		let edge = voice.attenuation(Vec3::X * 20.0);

		assert!((edge - 0.1).abs() < 1e-6, "at the maximum it is reference over maximum");
		assert!(
			(voice.attenuation(Vec3::X * 1000.0) - edge).abs() < 1e-6,
			"and past it, the same; a cutoff would pop as somebody walked over the line"
		);
	}

	#[test]
	fn a_voice_with_no_place_in_the_world_is_never_attenuated() {
		let voice = Voice::flat(SoundId::NONE);

		assert!(
			(voice.attenuation(Vec3::X * 1000.0) - 1.0).abs() < f32::EPSILON,
			"music does not get quieter when somebody walks away"
		);
	}

	#[test]
	fn a_reference_distance_of_nothing_does_not_divide_by_nothing() {
		let voice = Voice::at(SoundId::NONE, Vec3::ZERO).range(0.0, 0.0);
		let gain = voice.attenuation(Vec3::X * 5.0);

		assert!(gain.is_finite(), "it came back {gain}, which is not a volume");
		assert!((0.0..=1.0).contains(&gain), "and it is one: {gain}");
	}

	#[test]
	fn the_volume_and_the_pitch_are_clamped_where_they_are_read() {
		let loud = Voice::flat(SoundId::NONE).volume(1000.0);
		let quiet = Voice::flat(SoundId::NONE).volume(-1.0);

		assert!((loud.gain() - MAX_VOLUME).abs() < f32::EPSILON, "as loud as one voice may ask");
		assert!((quiet.gain() - 0.0).abs() < f32::EPSILON, "and never below silence");

		let fast = Voice::flat(SoundId::NONE).pitch(100.0);
		let stopped = Voice::flat(SoundId::NONE).pitch(0.0);

		assert!((fast.speed() - MAX_PITCH).abs() < f32::EPSILON);
		assert!(
			(stopped.speed() - MIN_PITCH).abs() < f32::EPSILON,
			"a playhead that does not move never reaches an end, and the slot is never freed"
		);
		assert!(
			(Voice::flat(SoundId::NONE).pitch(f32::NAN).speed() - 1.0).abs() < f32::EPSILON,
			"and a pitch that is not a number plays at the recording's own rate"
		);
	}

	#[test]
	fn a_voice_is_played_and_stopped_by_handle() {
		let mut voices = Voices::new();
		let id = voices.play(Voice::flat(SoundId::new(1)));

		assert!(id.is_some(), "something is playing");
		assert!(voices.alive(id), "and the handle finds it");
		assert_eq!(voices.len(), 1);
		assert!(!voices.is_empty());

		assert!(voices.stop(id), "it was there to stop");
		assert!(!voices.alive(id), "and now is not");
		assert!(!voices.stop(id), "stopping it twice says so");
		assert!(voices.is_empty());
	}

	#[test]
	fn a_stale_voice_handle_does_not_pick_up_its_successor() {
		// the whole reason the handle is generational: a game that starts a
		// footstep and keeps the handle must not, four seconds later, turn
		// down somebody else's music.
		let mut voices = Voices::new();
		let first = voices.play(Voice::flat(SoundId::new(1)));

		assert!(voices.stop(first));

		let second = voices.play(Voice::flat(SoundId::new(2)));

		assert_eq!(first.slot(), second.slot(), "the slot really was reused");
		assert_ne!(first, second, "and the handles differ all the same");
		assert!(!voices.alive(first), "so the old one finds nothing");
		assert_eq!(voices.get(second).map(|voice| voice.sound), Some(SoundId::new(2)));
	}

	#[test]
	fn a_full_table_refuses_and_counts_the_refusal() {
		let mut voices = Voices::new();

		for _ in 0..MAX_VOICES {
			assert!(voices.play(Voice::default()).is_some(), "there was room");
		}

		assert_eq!(voices.len(), MAX_VOICES, "and now there is not");
		assert_eq!(voices.play(Voice::default()), VoiceId::NONE, "so it is refused");
		assert_eq!(voices.dropped(), 1, "and counted rather than lost quietly");
		assert_eq!(voices.play(Voice::default()), VoiceId::NONE);
		assert_eq!(voices.dropped(), 2, "each time");
	}

	#[test]
	fn the_refusal_count_is_about_one_step_rather_than_the_run() {
		let (sounds, _) = one_second();
		let mut voices = Voices::new();

		for _ in 0..=MAX_VOICES {
			voices.play(Voice::default());
		}

		assert_eq!(voices.dropped(), 1, "one was refused this step");
		voices.advance(&sounds, crate::time::STEP_SECONDS);
		assert_eq!(voices.dropped(), 0, "and the next step starts over");
	}

	#[test]
	fn a_playhead_moves_by_one_step_a_step() {
		let (sounds, sound) = one_second();
		let mut voices = Voices::new();
		let id = voices.play(Voice::flat(sound));

		steps(&mut voices, &sounds, 6);

		let head = voices.get(id).expect("still playing").head;

		assert!(
			6.0_f32
				.mul_add(-crate::time::STEP_SECONDS, head)
				.abs() < 1e-6,
			"six steps in, it is six steps along: {head}"
		);
	}

	#[test]
	fn a_one_second_sound_ends_after_the_same_number_of_steps_every_time() {
		// the property the whole arrangement exists for. If this ever depends
		// on how long a device took to ask for samples, a screenshot stops
		// being reproducible.
		//
		// Sixty-one rather than sixty, and the number is worth writing down
		// rather than rounding away: `STEP_SECONDS` is the nearest f32 to a
		// sixtieth, summed sixty times it comes to 0.9999997, and a second is
		// therefore over on the step after the one arithmetic would suggest.
		// It is the same 0.9999997 everywhere, which is the property that
		// actually matters.
		let (sounds, sound) = one_second();
		let mut voices = Voices::new();
		let id = voices.play(Voice::flat(sound));

		steps(&mut voices, &sounds, 60);
		assert!(voices.alive(id), "sixty steps come to three ten-millionths short of a second");

		steps(&mut voices, &sounds, 1);
		assert!(!voices.alive(id), "and the sixty-first is over it");
		assert!(voices.is_empty(), "with the slot handed back");
	}

	#[test]
	fn a_faster_voice_gets_there_sooner() {
		let (sounds, sound) = one_second();
		let mut voices = Voices::new();
		let id = voices.play(Voice::flat(sound).pitch(2.0));

		steps(&mut voices, &sounds, 29);
		assert!(voices.alive(id), "half a second is twenty-nine steps and a bit");

		steps(&mut voices, &sounds, 1);
		assert!(!voices.alive(id), "and thirty is over it");
	}

	#[test]
	fn the_step_count_a_sound_lasts_is_the_same_on_every_run() {
		// the same claim from the other side, and the one that would catch a
		// playhead that had started depending on anything outside this table.
		let (sounds, sound) = one_second();
		let counted = |pitch: f32| {
			let mut voices = Voices::new();
			let id = voices.play(Voice::flat(sound).pitch(pitch));
			let mut count = 0_u32;

			while voices.alive(id) && count < 1000 {
				voices.advance(&sounds, crate::time::STEP_SECONDS);
				count += 1;
			}

			count
		};

		assert_eq!(counted(1.0), 61, "a second, in steps");
		assert_eq!(counted(1.0), counted(1.0), "and the same number every time it is asked");
		assert_eq!(counted(0.5), 121, "half speed is twice as long");
		assert_eq!(counted(2.0), 30, "and double speed is half");
	}

	#[test]
	fn a_looping_voice_wraps_instead_of_ending() {
		let (sounds, sound) = one_second();
		let mut voices = Voices::new();
		let id = voices.play(Voice::flat(sound).looping());

		steps(&mut voices, &sounds, 61);

		let head = voices.get(id).expect("still going round").head;

		assert!(head < 1.0, "it came back to the start rather than running past the end: {head}");
		assert!(head > 0.0, "and it is not sitting exactly on it either");

		steps(&mut voices, &sounds, 600);
		assert!(voices.alive(id), "ten seconds later it is still playing");
	}

	#[test]
	fn a_voice_on_a_sound_of_no_length_is_freed_at_once() {
		// including a looping one, which is what keeps a name nobody compiled
		// from holding a slot for the life of the process.
		let (sounds, _) = one_second();

		for voice in [Voice::flat(SoundId::NONE), Voice::flat(SoundId::NONE).looping()] {
			let mut voices = Voices::new();
			let id = voices.play(voice);

			voices.advance(&sounds, crate::time::STEP_SECONDS);

			assert!(!voices.alive(id), "there was nothing to play");
			assert!(voices.is_empty(), "and the slot came back");
		}
	}

	#[test]
	fn stopping_everything_leaves_a_table_that_still_works() {
		let (sounds, sound) = one_second();
		let mut voices = Voices::new();
		let mut handles = Vec::new();
		for _ in 0..8 {
			handles.push(voices.play(Voice::flat(sound)));
		}

		voices.stop_all();

		assert!(voices.is_empty(), "nothing is playing");
		for id in handles {
			assert!(!voices.alive(id), "and no handle finds anything");
		}

		let again = voices.play(Voice::flat(sound));
		steps(&mut voices, &sounds, 1);

		assert!(voices.alive(again), "and something started afterwards plays");
	}

	#[test]
	fn a_voice_is_moved_by_writing_to_it() {
		let (sounds, sound) = one_second();
		let mut voices = Voices::new();
		let id = voices.play(Voice::at(sound, Vec3::ZERO));

		voices.get_mut(id).expect("it is playing").at = Vec3::X * 5.0;

		steps(&mut voices, &sounds, 1);

		assert_eq!(
			voices.get(id).map(|voice| voice.at),
			Some(Vec3::X * 5.0),
			"a step does not undo what the game wrote"
		);
	}

	#[test]
	fn every_playing_voice_comes_back_from_the_walk_and_no_stopped_one_does() {
		let (_, sound) = one_second();
		let mut voices = Voices::new();
		let first = voices.play(Voice::flat(sound));
		let second = voices.play(Voice::flat(sound).looping());
		let third = voices.play(Voice::flat(sound));

		voices.stop(second);

		let seen: Vec<VoiceId> = voices.iter().map(|(id, _)| id).collect();

		assert_eq!(seen, vec![first, third], "in slot order, without the gap");
		assert!(voices.slots() >= 3, "and the slot the stopped one had is still there");
	}

	#[test]
	fn a_handle_from_another_table_falls_back_to_silence_rather_than_nothing() {
		let sounds = Sounds::new();

		assert!(sounds.get(SoundId::new(9)).is_none(), "there is no slot nine");
		assert!(sounds.data(SoundId::new(9)).is_empty(), "and asking for it plays nothing");
	}
}
