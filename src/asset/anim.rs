//! colby's runtime animation format: `.canim`.
//!
//! One file is one clip, which is what four of the five engines checked do:
//! an animation is a thing somebody plays by name, and a library holding six
//! of them is a name that has to be taken apart before anything can be played.
//! A clip is written beside the meshes of the model it came out of, under that
//! model's own directory, so `assets/models/hero.glb` holding a walk becomes
//! `target/assets/models/hero/walk.canim` and registers as `models/hero/walk`.
//!
//! ```text
//!   0  ClipHeader                       64 bytes
//!  64  [Curve; track_count]             20 bytes each
//!   .  [f32; time_count]                every track's times, end to end
//!   .  [f32; value_count]               every track's values, end to end
//!   .  the string blob, NUL-separated UTF-8
//! ```
//!
//! **The times and the values are two pools rather than two arrays per
//! track.** A record has to be a fixed width to be cast in place, and a track
//! holds a number of keys that only it knows, so the same trick names already
//! use is the one that works here: the record says where its run starts and
//! how long it is. Two tracks whose times are identical share one run, which
//! is not an optimization for its own sake - an exporter writes one key list
//! per channel and they are usually the same list, so this is about a quarter
//! of the file.
//!
//! **A track's times ascend, and the reader refuses a file where they do
//! not.** That is the invariant sampling stands on, exactly as
//! parents-before-children is the one a pose stands on, and it is checked here
//! for the same reason: this is plain data read out of a file, and everything
//! downstream searches it rather than validating it. @ref
//! [`Track::is_ordered`].
//!
//! **A clip does not store how long it is.** @ref [`ClipData::duration`],
//! which works it out of the keys, so there is nothing in the header that can
//! come to disagree with them.

use std::path::Path;

use colby_core::{
	Result,
	abi::anim::{Channel, ClipData, Interpolation, MAX_KEYS, MAX_TRACKS, Track},
	bytemuck::{self, Pod, Zeroable},
	err,
};

use crate::bytes::{AlignedBytes, fits, span};

/// The eight bytes every `.canim` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYANM";

/// The revision of everything in this module.
///
/// Bump it whenever the header or the record changes shape. A file carrying a
/// different number is refused with a message rather than read as if it
/// agreed.
pub const FORMAT_VERSION: u32 = 1;

/// The extension a compiled clip is written with.
pub const EXTENSION: &str = "canim";

/// How big [`ClipHeader`] is, and where the record block starts.
pub const HEADER_BYTES: usize = 64;

/// The largest string blob the reader will accept, in bytes.
///
/// A clip's names are one short word per track. This is how wrong a file has
/// to be before the reader stops rather than allocating what it was told to.
pub const MAX_NAMES: usize = 1 << 20;

/// The fixed head of a `.canim`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct ClipHeader {
	/// [`MAGIC`]. Anything else is not one of these files.
	pub magic: [u8; 8],

	/// [`FORMAT_VERSION`] at the time the file was written.
	pub version: u32,

	/// Reserved for optional blocks. Every bit is zero in version one, and a
	/// reader refuses a bit it does not know rather than ignoring it.
	pub flags: u32,

	/// Bytes per track record. Must be `size_of::<Curve>()`.
	pub track_stride: u32,

	/// How many tracks there are.
	pub track_count: u32,

	/// Where the record block starts, in bytes from the start of the file.
	pub track_offset: u32,

	/// Where the pool of key times starts.
	pub time_offset: u32,

	/// How many times are in it, counted as numbers rather than bytes.
	pub time_count: u32,

	/// Where the pool of key values starts.
	pub value_offset: u32,

	/// How many numbers are in it.
	pub value_count: u32,

	/// Where the string blob starts.
	pub names_offset: u32,

	/// How long the string blob is.
	pub names_length: u32,

	/// Spare, so the header is sixty-four bytes and the block after it
	/// inherits the buffer's alignment.
	pub reserved: [u32; 3],
}

const _: () = assert!(
	size_of::<ClipHeader>() == HEADER_BYTES,
	"the header has to stay sixty-four bytes for the block after it to be readable"
);

/// One track, as the file holds it.
///
/// [`Track`] is the same thing with its keys owned rather than pointed at;
/// this is the shape that can be cast straight out of the buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Curve {
	/// Offset into the blob of the bone this moves.
	pub name: u32,

	/// [`Channel::code`] of what it writes.
	pub channel: u8,

	/// [`Interpolation::code`] of how it moves between keys.
	pub interpolation: u8,

	/// Nothing yet, and a reader refuses a file that puts something here.
	pub reserved: u16,

	/// How many keys it has.
	pub keys: u32,

	/// Where its times start in the time pool, counted in numbers.
	pub time_at: u32,

	/// Where its values start in the value pool, counted in numbers.
	pub value_at: u32,
}

const _: () = assert!(
	size_of::<Curve>() == 20,
	"a track record is twenty bytes, and a file written by another build of it is refused"
);

/// A `.canim` held in memory, checked, and ready to be read in place.
#[derive(Clone, Debug)]
pub struct ClipFile {
	bytes: AlignedBytes,
	header: ClipHeader,
}

impl ClipFile {
	/// Reads and checks a compiled clip.
	///
	/// @param path - the `.canim` to read
	/// @return the file, or why it could not be used
	pub fn open(path: &Path) -> Result<Self> {
		let bytes = AlignedBytes::read(path)?;
		let header = check(bytes.as_slice())
			.map_err(|reason| err!(Asset("{}: {reason}", path.display())))?;

		Ok(Self { bytes, header })
	}

	/// Checks bytes that are already in memory.
	///
	/// @param bytes - the whole file
	/// @return the file, or why it could not be used
	pub fn from_bytes(bytes: AlignedBytes) -> Result<Self> {
		let header = check(bytes.as_slice()).map_err(|reason| err!(Asset("{reason}")))?;

		Ok(Self { bytes, header })
	}

	/// The header, as it was read.
	#[must_use]
	pub const fn header(&self) -> &ClipHeader { &self.header }

	/// The record block, borrowed out of the buffer.
	#[must_use]
	pub fn curves(&self) -> &[Curve] {
		self.block::<Curve>(self.header.track_offset, self.header.track_count)
	}

	/// Every track's times, end to end.
	#[must_use]
	pub fn times(&self) -> &[f32] {
		self.block::<f32>(self.header.time_offset, self.header.time_count)
	}

	/// Every track's values, end to end.
	#[must_use]
	pub fn values(&self) -> &[f32] {
		self.block::<f32>(self.header.value_offset, self.header.value_count)
	}

	/// One name out of the blob.
	///
	/// @param offset - what a record stored
	/// @return the text up to its terminator, or nothing when the offset is
	/// not one this file wrote
	#[must_use]
	pub fn name(&self, offset: u32) -> &str {
		let (Ok(start), Ok(base), Ok(length)) = (
			usize::try_from(offset),
			usize::try_from(self.header.names_offset),
			usize::try_from(self.header.names_length),
		) else {
			return "";
		};

		let blob = self
			.bytes
			.as_slice()
			.get(base..base.saturating_add(length))
			.unwrap_or_default();
		let rest = blob.get(start..).unwrap_or_default();
		let end = rest
			.iter()
			.position(|byte| *byte == 0)
			.unwrap_or(rest.len());

		std::str::from_utf8(rest.get(..end).unwrap_or_default()).unwrap_or("")
	}

	/// Copies the whole file into owned data.
	///
	/// The one copy in the path, and it is here for the same reason a mesh's
	/// is: what the host holds can also be built rather than read.
	#[must_use]
	pub fn to_clip_data(&self) -> ClipData {
		let times = self.times();
		let values = self.values();

		ClipData {
			tracks: self
				.curves()
				.iter()
				.map(|curve| {
					let channel = Channel::from_code(curve.channel).unwrap_or_default();
					let keys = usize::try_from(curve.keys).unwrap_or(0);
					let time_at = usize::try_from(curve.time_at).unwrap_or(0);
					let value_at = usize::try_from(curve.value_at).unwrap_or(0);
					let width = keys.saturating_mul(channel.lanes());

					Track {
						bone: self.name(curve.name).to_owned(),
						channel,
						interpolation: Interpolation::from_code(curve.interpolation)
							.unwrap_or_default(),
						times: times
							.get(time_at..time_at.saturating_add(keys))
							.unwrap_or_default()
							.to_vec(),
						values: values
							.get(value_at..value_at.saturating_add(width))
							.unwrap_or_default()
							.to_vec(),
					}
				})
				.collect(),
		}
	}

	/// One block of the file, cast in place.
	fn block<T: Pod>(&self, offset: u32, count: u32) -> &[T] {
		let Some(range) = span::<T>(offset, count) else {
			return &[];
		};

		self.bytes
			.as_slice()
			.get(range)
			.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
			.unwrap_or(&[])
	}
}

/// Writes a clip out as a `.canim`.
///
/// @param data - the tracks to write, each with its times in order
/// @return the whole file, ready to put on disk
pub fn encode(data: &ClipData) -> Result<Vec<u8>> {
	if data.len() > MAX_TRACKS {
		return Err(err!(Asset(
			"the clip has {} tracks, and {MAX_TRACKS} is the most one may have",
			data.len()
		)));
	}

	if data.keys() > MAX_KEYS {
		return Err(err!(Asset(
			"the clip has {} keys, and {MAX_KEYS} is the most one may have",
			data.keys()
		)));
	}

	if !data.is_whole() {
		return Err(err!(Asset(
			"the clip has a track holding a different number of values than its keys and \
			 channel call for"
		)));
	}

	if !data.is_ordered() {
		return Err(err!(Asset(
			"the clip has a track whose key times do not ascend, and everything that samples \
			 one searches them"
		)));
	}

	let mut names = Names::default();
	let mut pools = Pools::default();
	let curves: Vec<Curve> = data
		.tracks
		.iter()
		.map(|track| {
			Ok(Curve {
				name: names.put(&track.bone),
				channel: track.channel.code(),
				interpolation: track.interpolation.code(),
				reserved: 0,
				keys: count(track.keys())?,
				time_at: pools.put_times(&track.times)?,
				value_at: pools.put_values(&track.values)?,
			})
		})
		.collect::<Result<Vec<Curve>>>()?;

	let track_offset = HEADER_BYTES;
	let time_offset = track_offset + size_of_val(curves.as_slice());
	let value_offset = time_offset + size_of_val(pools.times.as_slice());
	let names_offset = value_offset + size_of_val(pools.values.as_slice());
	let header = ClipHeader {
		magic: MAGIC,
		version: FORMAT_VERSION,
		flags: 0,
		track_stride: width::<Curve>()?,
		track_count: count(curves.len())?,
		track_offset: count(track_offset)?,
		time_offset: count(time_offset)?,
		time_count: count(pools.times.len())?,
		value_offset: count(value_offset)?,
		value_count: count(pools.values.len())?,
		names_offset: count(names_offset)?,
		names_length: count(names.blob.len())?,
		reserved: [0; 3],
	};

	let mut out = Vec::with_capacity(names_offset + names.blob.len());
	out.extend_from_slice(bytemuck::bytes_of(&header));
	out.extend_from_slice(bytemuck::cast_slice(&curves));
	out.extend_from_slice(bytemuck::cast_slice(&pools.times));
	out.extend_from_slice(bytemuck::cast_slice(&pools.values));
	out.extend_from_slice(&names.blob);

	Ok(out)
}

/// The version a `.canim` claims, without reading the rest of it.
///
/// What the compiler asks so that a file written by another build of the
/// engine is stale however new it is.
///
/// @param path - the file to look at
/// @return its version, or nothing when it is not one of these at all
#[must_use]
pub fn version_of(path: &Path) -> Option<u32> {
	let mut head = [0_u8; 12];
	let mut file = std::fs::File::open(path).ok()?;
	std::io::Read::read_exact(&mut file, &mut head).ok()?;

	if head.get(..MAGIC.len()) != Some(&MAGIC[..]) {
		return None;
	}

	let version: [u8; 4] = head.get(8..12)?.try_into().ok()?;

	Some(u32::from_le_bytes(version))
}

/// Everything that has to hold before a [`ClipFile`] exists.
fn check(bytes: &[u8]) -> std::result::Result<ClipHeader, String> {
	let head = bytes.get(..HEADER_BYTES).ok_or_else(|| {
		format!(
			"only {} bytes long, too short to hold a {HEADER_BYTES}-byte header",
			bytes.len()
		)
	})?;
	let header: &ClipHeader = bytemuck::try_from_bytes(head)
		.map_err(|error| format!("the header could not be read: {error}"))?;

	if header.magic != MAGIC {
		return Err(format!(
			"not a colby animation: expected {:?} at the start, found {:?}",
			String::from_utf8_lossy(&MAGIC),
			String::from_utf8_lossy(&header.magic)
		));
	}

	if header.version != FORMAT_VERSION {
		return Err(format!(
			"written by asset format version {}, and this build reads version {FORMAT_VERSION}; \
			 run `just assets --force` to recompile it",
			header.version
		));
	}

	if header.flags != 0 {
		return Err(format!(
			"sets flag bits {:#010X} that this build does not know about",
			header.flags
		));
	}

	check_blocks(bytes, header)?;
	check_curves(bytes, header)?;

	Ok(*header)
}

/// Checks that every block is inside the file and sits where it can be read.
fn check_blocks(bytes: &[u8], header: &ClipHeader) -> std::result::Result<(), String> {
	if header.track_stride != width::<Curve>().unwrap_or(0) {
		return Err(format!(
			"has {}-byte tracks, and this build reads {}-byte ones",
			header.track_stride,
			size_of::<Curve>()
		));
	}

	if header.track_count > u32::try_from(MAX_TRACKS).unwrap_or(u32::MAX) {
		return Err(format!(
			"has {} tracks, and {MAX_TRACKS} is the most one may have",
			header.track_count
		));
	}

	if header.time_count > u32::try_from(MAX_KEYS).unwrap_or(u32::MAX) {
		return Err(format!(
			"has {} keys, and {MAX_KEYS} is the most one may have",
			header.time_count
		));
	}

	if header.names_length > u32::try_from(MAX_NAMES).unwrap_or(u32::MAX) {
		return Err(format!(
			"declares a {}-byte name blob, past the {MAX_NAMES} a clip may have",
			header.names_length
		));
	}

	fits::<Curve>(bytes, HEADER_BYTES, (header.track_offset, header.track_count), "tracks")?;
	fits::<f32>(bytes, HEADER_BYTES, (header.time_offset, header.time_count), "key times")?;
	fits::<f32>(bytes, HEADER_BYTES, (header.value_offset, header.value_count), "key values")?;

	let names_end = usize::try_from(header.names_offset)
		.unwrap_or(usize::MAX)
		.saturating_add(usize::try_from(header.names_length).unwrap_or(usize::MAX));

	if names_end > bytes.len() {
		return Err(format!(
			"declares a name blob ending at {names_end} past the file's {}",
			bytes.len()
		));
	}

	Ok(())
}

/// Checks every track's run against the pools, and its times against the
/// invariant sampling stands on.
fn check_curves(bytes: &[u8], header: &ClipHeader) -> std::result::Result<(), String> {
	let Some(records) = span::<Curve>(header.track_offset, header.track_count) else {
		return Err("declares a track block that overflows its offsets".to_owned());
	};
	let Some(pool) = span::<f32>(header.time_offset, header.time_count) else {
		return Err("declares a time pool that overflows its offsets".to_owned());
	};

	let curves: &[Curve] = bytes
		.get(records)
		.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
		.ok_or_else(|| "has a track block that cannot be read in place".to_owned())?;
	let times: &[f32] = bytes
		.get(pool)
		.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
		.ok_or_else(|| "has a time pool that cannot be read in place".to_owned())?;

	for (index, curve) in curves.iter().enumerate() {
		check_curve(index, curve, header, times)?;
	}

	Ok(())
}

/// The same for one track.
fn check_curve(
	index: usize,
	curve: &Curve,
	header: &ClipHeader,
	times: &[f32],
) -> std::result::Result<(), String> {
	if curve.reserved != 0 {
		return Err(format!(
			"puts {:#06X} in a word of track {index} that has no meaning yet",
			curve.reserved
		));
	}

	let Some(channel) = Channel::from_code(curve.channel) else {
		return Err(format!(
			"has track {index} writing channel {}, which this build does not know",
			curve.channel
		));
	};

	if Interpolation::from_code(curve.interpolation).is_none() {
		return Err(format!(
			"has track {index} interpolated by rule {}, which this build does not know",
			curve.interpolation
		));
	}

	if curve.keys == 0 {
		return Err(format!("has track {index} saying a bone is animated and holding no keys"));
	}

	let keys = usize::try_from(curve.keys).unwrap_or(usize::MAX);
	let start = usize::try_from(curve.time_at).unwrap_or(usize::MAX);
	let end = start.saturating_add(keys);

	if end > times.len() {
		return Err(format!(
			"has track {index} taking times {start} to {end} out of a pool of {}",
			times.len()
		));
	}

	let values = usize::try_from(curve.value_at)
		.unwrap_or(usize::MAX)
		.saturating_add(keys.saturating_mul(channel.lanes()));

	if values > usize::try_from(header.value_count).unwrap_or(0) {
		return Err(format!(
			"has track {index} taking values up to {values} out of a pool of {}",
			header.value_count
		));
	}

	let run = times.get(start..end).unwrap_or_default();

	if !run.iter().all(|time| time.is_finite()) {
		return Err(format!("has track {index} keyed at a time that is not a number"));
	}

	if !run.windows(2).all(|pair| pair[0] < pair[1]) {
		return Err(format!(
			"has track {index} whose key times do not ascend, and sampling one searches them"
		));
	}

	Ok(())
}

/// The size of `T` as a header stores it.
fn width<T>() -> Result<u32> {
	u32::try_from(size_of::<T>())
		.map_err(|_| err!(Asset("a record of {} bytes cannot be described", size_of::<T>())))
}

/// A count as a header stores it.
fn count(value: usize) -> Result<u32> {
	u32::try_from(value).map_err(|_| err!(Asset("{value} is more than a u32 can address")))
}

/// The two pools being built, and where each run of times already in one
/// starts.
///
/// Times are shared and values are not. An exporter writes one key list per
/// channel and they are almost always the same list, so a rig of a hundred
/// tracks writes one run instead of a hundred; two tracks with the same values
/// would be a coincidence rather than the usual case, and looking for one
/// would cost more than it saves.
#[derive(Default)]
struct Pools {
	times: Vec<f32>,
	values: Vec<f32>,
	runs: Vec<(u32, u32)>,
}

impl Pools {
	/// Puts a run of times in, or finds the one already there.
	fn put_times(&mut self, times: &[f32]) -> Result<u32> {
		let wanted = count(times.len())?;

		for (at, length) in &self.runs {
			if *length != wanted {
				continue;
			}

			let start = usize::try_from(*at).unwrap_or(0);
			let already = self
				.times
				.get(start..start.saturating_add(times.len()))
				.unwrap_or_default();

			// by bit pattern rather than by value, because this is a question
			// about whether two runs are the same bytes.
			if already
				.iter()
				.zip(times)
				.all(|(one, other)| one.to_bits() == other.to_bits())
			{
				return Ok(*at);
			}
		}

		let at = count(self.times.len())?;

		self.times.extend_from_slice(times);
		self.runs.push((at, wanted));

		Ok(at)
	}

	/// Puts a run of values in.
	fn put_values(&mut self, values: &[f32]) -> Result<u32> {
		let at = count(self.values.len())?;

		self.values.extend_from_slice(values);

		Ok(at)
	}
}

/// The blob being built, and where each name already in it starts.
#[derive(Default)]
struct Names {
	blob: Vec<u8>,
	written: Vec<(String, u32)>,
}

impl Names {
	/// Puts a name in, or finds the one already there.
	///
	/// The empty name is offset zero and is written once, at the head, because
	/// that is what a record naming nothing stores and a reader has to find a
	/// terminator there.
	fn put(&mut self, name: &str) -> u32 {
		if self.blob.is_empty() {
			self.blob.push(0);
		}

		if name.is_empty() {
			return 0;
		}

		if let Some((_, at)) = self
			.written
			.iter()
			.find(|(already, _)| already == name)
		{
			return *at;
		}

		let at = u32::try_from(self.blob.len()).unwrap_or(0);

		self.blob.extend_from_slice(name.as_bytes());
		self.blob.push(0);
		self.written.push((name.to_owned(), at));

		at
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Where each field of the header sits, for a test that has to break one.
	const VERSION_AT: usize = 8;
	const FLAGS_AT: usize = 12;
	const TIME_COUNT_AT: usize = 32;

	/// Where each field of the first record sits, for the same reason.
	const CHANNEL_AT: usize = HEADER_BYTES + 4;
	const RULE_AT: usize = HEADER_BYTES + 5;
	const RESERVED_AT: usize = HEADER_BYTES + 6;
	const KEYS_AT: usize = HEADER_BYTES + 8;
	const TIME_AT: usize = HEADER_BYTES + 12;

	/// One track of a named bone, keyed at the moments given.
	fn track(bone: &str, channel: Channel, times: &[f32]) -> Track {
		let lanes = channel.lanes();

		Track {
			bone: bone.to_owned(),
			channel,
			interpolation: Interpolation::Linear,
			times: times.to_vec(),
			values: (0..times.len() * lanes)
				.map(|index| f32::from(u8::try_from(index).unwrap_or(0)) * 0.25)
				.collect(),
		}
	}

	/// Two bones moving, three of the four tracks keyed at the same moments.
	fn clip() -> ClipData {
		ClipData {
			tracks: vec![
				track("hips", Channel::Position, &[0.0, 0.5, 1.0]),
				track("hips", Channel::Rotation, &[0.0, 0.5, 1.0]),
				track("spine", Channel::Rotation, &[0.0, 0.5, 1.0]),
				track("spine", Channel::Scale, &[0.0, 1.0]),
			],
		}
	}

	/// A clip written out and read back.
	fn round_trip(data: &ClipData) -> ClipFile {
		let bytes = encode(data).expect("the clip is written");

		ClipFile::from_bytes(AlignedBytes::from_slice(&bytes)).expect("and read back")
	}

	/// Why a file was refused.
	fn refusal(bytes: &[u8]) -> String {
		ClipFile::from_bytes(AlignedBytes::from_slice(bytes))
			.expect_err("the file should be refused")
			.to_string()
	}

	/// A good file with one byte changed.
	fn broken(at: usize, to: u8) -> Vec<u8> {
		let mut bytes = encode(&clip()).expect("the clip is written");

		bytes[at] = to;

		bytes
	}

	#[test]
	fn a_clip_survives_the_trip_through_a_file() {
		let original = clip();
		let file = round_trip(&original);

		assert_eq!(file.header().version, FORMAT_VERSION);
		assert_eq!(file.header().track_count, 4, "four tracks");
		assert_eq!(file.to_clip_data(), original, "and every one of them read back as itself");
	}

	#[test]
	fn tracks_keyed_at_the_same_moments_share_one_run_of_times() {
		let file = round_trip(&clip());

		assert_eq!(
			file.header().time_count,
			5,
			"three moments written once for the three tracks that share them, and two for the \
			 one that does not"
		);

		let curves = file.curves();

		assert_eq!(curves[0].time_at, curves[1].time_at, "the first two point at one run");
		assert_eq!(curves[1].time_at, curves[2].time_at, "and so does the third");
		assert_ne!(curves[3].time_at, curves[0].time_at, "the odd one out has its own");
		assert_ne!(curves[0].value_at, curves[1].value_at, "and no two share values");
	}

	#[test]
	fn a_name_is_written_once_however_many_tracks_move_the_bone() {
		let file = round_trip(&clip());
		let curves = file.curves();

		assert_eq!(curves[0].name, curves[1].name, "both of the hips' tracks name one string");
		assert_ne!(curves[0].name, curves[2].name, "and the spine is a different one");
		assert_eq!(file.name(curves[2].name), "spine", "which reads back as itself");
	}

	#[test]
	fn a_clip_with_no_tracks_is_a_file_all_the_same() {
		let file = round_trip(&ClipData::default());

		assert_eq!(file.header().track_count, 0);
		assert_eq!(file.to_clip_data(), ClipData::default(), "and reads back as nothing");
	}

	#[test]
	fn the_version_is_readable_without_reading_the_rest() {
		let dir = std::env::temp_dir().join("colby-anim-tests");
		let path = dir.join("walk.canim");

		std::fs::create_dir_all(&dir).expect("the directory is made");
		std::fs::write(&path, encode(&clip()).expect("the clip is written"))
			.expect("and written to disk");

		assert_eq!(version_of(&path), Some(FORMAT_VERSION), "its own version");

		std::fs::write(&path, b"not a clip at all").expect("something else is written");

		assert_eq!(version_of(&path), None, "and something that is not one has none");
	}

	#[test]
	fn a_track_whose_times_do_not_ascend_is_refused_on_the_way_out() {
		let mut data = clip();
		data.tracks[0].times = vec![0.0, 1.0, 0.5];

		let reason = encode(&data)
			.expect_err("it should be refused")
			.to_string();

		assert!(reason.contains("do not ascend"), "{reason}");
	}

	#[test]
	fn a_track_holding_the_wrong_number_of_values_is_refused_on_the_way_out() {
		let mut data = clip();
		data.tracks[0].values.pop();

		let reason = encode(&data)
			.expect_err("it should be refused")
			.to_string();

		assert!(reason.contains("different number of values"), "{reason}");
	}

	#[test]
	fn more_tracks_than_a_clip_may_hold_are_refused_on_the_way_out() {
		let data = ClipData {
			tracks: vec![track("hips", Channel::Position, &[0.0]); MAX_TRACKS + 1],
		};
		let reason = encode(&data)
			.expect_err("it should be refused")
			.to_string();

		assert!(reason.contains("is the most one may have"), "{reason}");
	}

	#[test]
	fn a_file_that_is_not_one_is_refused_by_name() {
		assert!(refusal(&broken(0, b'X')).contains("not a colby animation"));
		assert!(refusal(&[0; 8]).contains("too short to hold"));
	}

	#[test]
	fn a_file_from_another_build_is_refused_rather_than_read() {
		let reason = refusal(&broken(VERSION_AT, 99));

		assert!(reason.contains("just assets --force"), "and says what to do about it: {reason}");
	}

	#[test]
	fn a_flag_this_build_does_not_know_is_refused() {
		let reason = refusal(&broken(FLAGS_AT, 1));

		assert!(reason.contains("flag bits"), "{reason}");
	}

	#[test]
	fn a_word_with_no_meaning_yet_has_to_be_empty() {
		let reason = refusal(&broken(RESERVED_AT, 7));

		assert!(reason.contains("has no meaning yet"), "{reason}");
	}

	#[test]
	fn a_channel_or_a_rule_this_build_does_not_know_is_refused() {
		assert!(refusal(&broken(CHANNEL_AT, 9)).contains("channel 9"));
		assert!(refusal(&broken(RULE_AT, 9)).contains("rule 9"));
	}

	#[test]
	fn a_track_with_no_keys_is_refused_rather_than_read_as_a_bone_standing_still() {
		let reason = refusal(&broken(KEYS_AT, 0));

		assert!(reason.contains("holding no keys"), "{reason}");
	}

	#[test]
	fn a_track_reaching_past_the_pool_it_takes_its_times_from_is_refused() {
		let reason = refusal(&broken(TIME_AT, 200));

		assert!(reason.contains("out of a pool of"), "{reason}");
	}

	#[test]
	fn a_file_whose_times_do_not_ascend_is_refused_where_it_is_read() {
		// the invariant sampling stands on, and the encoder is not the only
		// thing that could ever write one of these files.
		let mut bytes = encode(&clip()).expect("the clip is written");
		let times = usize::try_from(
			bytemuck::from_bytes::<ClipHeader>(&bytes[..HEADER_BYTES]).time_offset,
		)
		.expect("the offset fits");

		bytes[times + 4..times + 8].copy_from_slice(&(-1.0_f32).to_le_bytes());

		assert!(refusal(&bytes).contains("do not ascend"), "the second key goes backwards");

		let mut bytes = encode(&clip()).expect("the clip is written");
		bytes[times + 4..times + 8].copy_from_slice(&f32::NAN.to_le_bytes());

		assert!(refusal(&bytes).contains("not a number"), "and a key at no moment at all");
	}

	#[test]
	fn a_pool_bigger_than_a_clip_may_have_is_refused_before_anything_is_allocated() {
		let mut bytes = encode(&clip()).expect("the clip is written");
		let over = u32::try_from(MAX_KEYS + 1).expect("the bound fits");

		bytes[TIME_COUNT_AT..TIME_COUNT_AT + 4].copy_from_slice(&over.to_le_bytes());

		assert!(refusal(&bytes).contains("is the most one may have"));
	}
}
