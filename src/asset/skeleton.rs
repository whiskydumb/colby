//! colby's runtime skeleton format: `.cskel`.
//!
//! A skeleton is what a skinned mesh is moved by, and it is its own file for
//! the reason every large engine keeps it as its own asset: several meshes
//! share one, and so does every animation. A mesh's skin block names bones by
//! index and an animation names them by text; both of those are answered here
//! and nowhere else.
//!
//! ```text
//!   0  SkeletonHeader                   64 bytes
//!  64  [Limb; bone_count]              112 bytes each
//!   .  the string blob, NUL-separated UTF-8
//! ```
//!
//! The record block is `#[repr(C)]` and cast in place out of an
//! [`AlignedBytes`](crate::AlignedBytes), the same trick `.cmesh` and
//! `.cmodel` use, and names are offsets into one blob at the end for the same
//! reason: they vary in length and a record may not.
//!
//! **A parent is written before its child, and the reader refuses a file where
//! one is not.** That is the invariant a pose stands on - resolving one into
//! world matrices is a single forward walk, with no recursion and no scratch,
//! only because a bone's parent is already done by the time the bone is
//! reached. The exchange format promises nothing of the sort, so the importer
//! sorts and this is where the sorting is checked. @ref
//! [`SkeletonData::is_ordered`].

use std::path::Path;

use colby_core::{
	Result,
	abi::{
		Transform,
		skeleton::{Bone, MAX_BONES, NO_PARENT, SkeletonData},
	},
	bytemuck::{self, Pod, Zeroable},
	err,
	glam::{Mat4, Quat, Vec3},
};

use crate::bytes::{AlignedBytes, span};

/// The eight bytes every `.cskel` starts with.
pub const MAGIC: [u8; 8] = *b"COLBYSKL";

/// The revision of everything in this module.
///
/// Bump it whenever the header or the record changes shape. A file carrying a
/// different number is refused with a message rather than read as if it
/// agreed.
pub const FORMAT_VERSION: u32 = 1;

/// The extension a compiled skeleton is written with.
pub const EXTENSION: &str = "cskel";

/// How big [`SkeletonHeader`] is, and where the record block starts.
pub const HEADER_BYTES: usize = 64;

/// The largest string blob the reader will accept, in bytes.
///
/// A skeleton's names are a couple of hundred short words. This is how wrong a
/// file has to be before the reader stops rather than allocating what it was
/// told to.
pub const MAX_NAMES: usize = 1 << 20;

/// The fixed head of a `.cskel`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct SkeletonHeader {
	/// [`MAGIC`]. Anything else is not one of these files.
	pub magic: [u8; 8],

	/// [`FORMAT_VERSION`] at the time the file was written.
	pub version: u32,

	/// Reserved for optional blocks. Every bit is zero in version one, and a
	/// reader refuses a bit it does not know rather than ignoring it.
	pub flags: u32,

	/// Bytes per bone record. Must be `size_of::<Limb>()`.
	pub bone_stride: u32,

	/// How many bones there are.
	pub bone_count: u32,

	/// Where the record block starts, in bytes from the start of the file.
	pub bone_offset: u32,

	/// Where the string blob starts.
	pub names_offset: u32,

	/// How long the string blob is.
	pub names_length: u32,

	/// Spare, so the header is sixty-four bytes and the block after it
	/// inherits the buffer's alignment.
	pub reserved: [u32; 7],
}

const _: () = assert!(
	size_of::<SkeletonHeader>() == HEADER_BYTES,
	"the header has to stay sixty-four bytes for the block after it to be readable"
);

/// One bone, as the file holds it.
///
/// [`Bone`] is the same thing with its name owned rather than pointed at; this
/// is the shape that can be cast straight out of the buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[bytemuck(crate = "::colby_core::bytemuck")]
pub struct Limb {
	/// Offset into the blob of what this bone is called.
	pub name: u32,

	/// Which bone it hangs off, or [`NO_PARENT`].
	///
	/// Always less than this record's own index, which the reader checks.
	pub parent: u16,

	/// Nothing yet, and a reader refuses a file that puts something here.
	pub reserved: u16,

	/// The mesh's own space into this bone's, as the mesh was authored.
	pub inverse_bind: [[f32; 4]; 4],

	/// Where the bone sits relative to its parent with nothing animating it.
	pub position: [f32; 3],

	/// How it is turned there, xyzw.
	pub rotation: [f32; 4],

	/// How big it is there, along each axis.
	pub scale: [f32; 3],
}

/// A `.cskel` held in memory, checked, and ready to be read in place.
#[derive(Clone, Debug)]
pub struct SkeletonFile {
	bytes: AlignedBytes,
	header: SkeletonHeader,
}

impl SkeletonFile {
	/// Reads and checks a compiled skeleton.
	///
	/// @param path - the `.cskel` to read
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
	pub const fn header(&self) -> &SkeletonHeader { &self.header }

	/// The record block, borrowed out of the buffer.
	#[must_use]
	pub fn limbs(&self) -> &[Limb] {
		let Some(range) = span::<Limb>(self.header.bone_offset, self.header.bone_count) else {
			return &[];
		};

		self.bytes
			.as_slice()
			.get(range)
			.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
			.unwrap_or(&[])
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
	pub fn to_skeleton_data(&self) -> SkeletonData {
		SkeletonData {
			bones: self
				.limbs()
				.iter()
				.map(|limb| Bone {
					name: self.name(limb.name).to_owned(),
					parent: limb.parent,
					inverse_bind: Mat4::from_cols_array_2d(&limb.inverse_bind),
					rest: Transform {
						position: Vec3::from_array(limb.position),
						rotation: Quat::from_array(limb.rotation),
						scale: Vec3::from_array(limb.scale),
					},
				})
				.collect(),
		}
	}
}

/// Writes a skeleton out as a `.cskel`.
///
/// @param data - the bones to write, parents before children
/// @return the whole file, ready to put on disk
pub fn encode(data: &SkeletonData) -> Result<Vec<u8>> {
	if data.len() > MAX_BONES {
		return Err(err!(Asset(
			"the skeleton has {} bones, and {MAX_BONES} is the most one may have",
			data.len()
		)));
	}

	if !data.is_ordered() {
		return Err(err!(Asset(
			"the skeleton has a bone written before the one it hangs off, and everything that \
			 reads one walks it forwards"
		)));
	}

	let mut names = Names::default();
	let limbs: Vec<Limb> = data
		.bones
		.iter()
		.map(|bone| Limb {
			name: names.put(&bone.name),
			parent: bone.parent,
			reserved: 0,
			inverse_bind: bone.inverse_bind.to_cols_array_2d(),
			position: bone.rest.position.to_array(),
			rotation: bone.rest.rotation.to_array(),
			scale: bone.rest.scale.to_array(),
		})
		.collect();

	let bone_offset = HEADER_BYTES;
	let names_offset = bone_offset + size_of_val(limbs.as_slice());
	let header = SkeletonHeader {
		magic: MAGIC,
		version: FORMAT_VERSION,
		flags: 0,
		bone_stride: width::<Limb>()?,
		bone_count: count(limbs.len())?,
		bone_offset: count(bone_offset)?,
		names_offset: count(names_offset)?,
		names_length: count(names.blob.len())?,
		reserved: [0; 7],
	};

	let mut out = Vec::with_capacity(names_offset + names.blob.len());
	out.extend_from_slice(bytemuck::bytes_of(&header));
	out.extend_from_slice(bytemuck::cast_slice(&limbs));
	out.extend_from_slice(&names.blob);

	Ok(out)
}

/// The version a `.cskel` claims, without reading the rest of it.
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

/// Everything that has to hold before a [`SkeletonFile`] exists.
fn check(bytes: &[u8]) -> std::result::Result<SkeletonHeader, String> {
	let head = bytes.get(..HEADER_BYTES).ok_or_else(|| {
		format!(
			"only {} bytes long, too short to hold a {HEADER_BYTES}-byte header",
			bytes.len()
		)
	})?;
	let header: &SkeletonHeader = bytemuck::try_from_bytes(head)
		.map_err(|error| format!("the header could not be read: {error}"))?;

	if header.magic != MAGIC {
		return Err(format!(
			"not a colby skeleton: expected {:?} at the start, found {:?}",
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

	check_blocks(header, bytes.len())?;
	check_bones(bytes, header)?;

	Ok(*header)
}

/// Checks that both blocks are inside the file and sit where they can be read.
fn check_blocks(header: &SkeletonHeader, len: usize) -> std::result::Result<(), String> {
	if header.bone_stride != width::<Limb>().unwrap_or(0) {
		return Err(format!(
			"has {}-byte bones, and this build reads {}-byte ones",
			header.bone_stride,
			size_of::<Limb>()
		));
	}

	if header.bone_count > u32::try_from(MAX_BONES).unwrap_or(u32::MAX) {
		return Err(format!(
			"has {} bones, and {MAX_BONES} is the most one may have",
			header.bone_count
		));
	}

	if header.names_length > u32::try_from(MAX_NAMES).unwrap_or(u32::MAX) {
		return Err(format!(
			"declares a {}-byte name blob, past the {MAX_NAMES} a skeleton may have",
			header.names_length
		));
	}

	let range = span::<Limb>(header.bone_offset, header.bone_count)
		.ok_or_else(|| "declares a bone block that overflows its offsets".to_owned())?;

	if range.end > len {
		return Err(format!(
			"declares a bone block ending at {} in a file {len} bytes long",
			range.end
		));
	}

	if usize::try_from(header.bone_offset).unwrap_or(usize::MAX) % 4 != 0 {
		return Err(format!(
			"puts its bone block at {}, which is not four-aligned",
			header.bone_offset
		));
	}

	let names_end = usize::try_from(header.names_offset)
		.unwrap_or(usize::MAX)
		.saturating_add(usize::try_from(header.names_length).unwrap_or(usize::MAX));

	if names_end > len {
		return Err(format!("declares a name blob ending at {names_end} past the file's {len}"));
	}

	Ok(())
}

/// Checks the one thing everything that reads a skeleton depends on.
fn check_bones(bytes: &[u8], header: &SkeletonHeader) -> std::result::Result<(), String> {
	let Some(range) = span::<Limb>(header.bone_offset, header.bone_count) else {
		return Err("declares a bone block that overflows its offsets".to_owned());
	};

	let limbs: &[Limb] = bytes
		.get(range)
		.and_then(|slice| bytemuck::try_cast_slice(slice).ok())
		.ok_or_else(|| "has a bone block that cannot be read in place".to_owned())?;

	for (index, limb) in limbs.iter().enumerate() {
		if limb.reserved != 0 {
			return Err(format!(
				"puts {:#06X} in a word of bone {index} that has no meaning yet",
				limb.reserved
			));
		}

		if limb.parent == NO_PARENT {
			continue;
		}

		if usize::from(limb.parent) >= index {
			return Err(format!(
				"has bone {index} hanging off bone {}, which is not written before it",
				limb.parent
			));
		}
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

	/// Three bones in a row, with the inverse binds worked out from the rests.
	fn arm() -> SkeletonData {
		let mut data = SkeletonData {
			bones: vec![
				Bone {
					name: "shoulder".to_owned(),
					..Bone::default()
				},
				Bone {
					name: "elbow".to_owned(),
					parent: 0,
					rest: Transform::at(Vec3::X),
					..Bone::default()
				},
				Bone {
					name: "wrist".to_owned(),
					parent: 1,
					rest: Transform::at(Vec3::X * 2.0),
					..Bone::default()
				},
			],
		};
		let mut model: Vec<Mat4> = Vec::with_capacity(data.len());

		for bone in &data.bones {
			let local = bone.rest.matrix();
			let world = if bone.parent == NO_PARENT {
				local
			} else {
				model[usize::from(bone.parent)] * local
			};

			model.push(world);
		}

		for (bone, world) in data.bones.iter_mut().zip(&model) {
			bone.inverse_bind = world.inverse();
		}

		data
	}

	/// The arm, encoded.
	fn encoded() -> Vec<u8> { encode(&arm()).expect("an arm encodes") }

	/// The encoded arm, read back.
	fn opened() -> SkeletonFile {
		SkeletonFile::from_bytes(AlignedBytes::from_slice(&encoded())).expect("and reads back")
	}

	/// What reading these bytes went wrong with.
	fn refused(bytes: &[u8]) -> String {
		SkeletonFile::from_bytes(AlignedBytes::from_slice(bytes))
			.expect_err("the file is not valid")
			.to_string()
	}

	#[test]
	fn the_header_and_the_record_are_the_sizes_the_layout_depends_on() {
		assert_eq!(size_of::<SkeletonHeader>(), HEADER_BYTES, "sixty-four bytes, exactly");
		assert_eq!(size_of::<Limb>(), 112, "and a bone is a matrix, a transform and two words");
		assert_eq!(align_of::<Limb>(), 4, "with no padding in it");
	}

	#[test]
	fn a_skeleton_survives_the_trip_to_bytes_and_back() {
		for original in [arm(), SkeletonData::default()] {
			let bytes = encode(&original).expect("it encodes");
			let file = SkeletonFile::from_bytes(AlignedBytes::from_slice(&bytes))
				.expect("and reads back");

			assert_eq!(file.to_skeleton_data(), original, "every bone, name and matrix");
		}
	}

	#[test]
	fn the_bones_are_borrowed_out_of_the_buffer_rather_than_copied() {
		let file = opened();
		let base = file.bytes.as_slice().as_ptr().addr();

		assert_eq!(
			file.limbs().as_ptr().addr(),
			base + HEADER_BYTES,
			"the bones are the file's own bytes"
		);
		assert_eq!(file.limbs().len(), 3, "all three of them");
		assert_eq!(file.name(file.limbs()[2].name), "wrist", "with their names beside them");
	}

	#[test]
	fn a_name_two_bones_share_is_written_once() {
		let mut twins = arm();
		twins.bones[2].name = "elbow".to_owned();

		let file = SkeletonFile::from_bytes(AlignedBytes::from_slice(
			&encode(&twins).expect("it encodes"),
		))
		.expect("and reads back");
		let limbs = file.limbs();

		assert_eq!(limbs[1].name, limbs[2].name, "one offset, one copy of the text");
		assert_eq!(file.name(limbs[2].name), "elbow");
	}

	#[test]
	fn a_bone_written_before_the_one_it_hangs_off_is_not_written_at_all() {
		let mut backwards = arm();
		backwards.bones.swap(0, 1);

		let message = encode(&backwards)
			.expect_err("it is not writable")
			.to_string();

		assert!(
			message.contains("before the one it hangs off"),
			"the invariant is named rather than the field: {message}"
		);
	}

	#[test]
	fn a_skeleton_with_more_bones_than_any_may_have_is_not_written() {
		let crowded = SkeletonData {
			bones: vec![Bone::default(); MAX_BONES + 1],
		};

		assert!(
			encode(&crowded)
				.expect_err("it is not writable")
				.to_string()
				.contains(&MAX_BONES.to_string()),
			"and the message says what the limit is"
		);
	}

	#[test]
	fn a_file_whose_bones_are_out_of_order_is_refused_on_the_way_in() {
		let mut bytes = encoded();
		// bone one's parent sits four bytes into its record, which is the
		// second one after a sixty-four byte header.
		let parent_of_second = HEADER_BYTES + size_of::<Limb>() + 4;
		bytes[parent_of_second..parent_of_second + 2].copy_from_slice(&2_u16.to_le_bytes());

		let message = refused(&bytes);

		assert!(
			message.contains("not written before it"),
			"a forward walk would read a parent nobody had placed: {message}"
		);
	}

	#[test]
	fn a_file_from_another_version_says_so_instead_of_panicking() {
		let mut bytes = encoded();
		bytes[8..12].copy_from_slice(&99_u32.to_le_bytes());

		let message = refused(&bytes);

		assert!(message.contains("version 99"), "it names the version found: {message}");
		assert!(message.contains("just assets"), "and what to do about it: {message}");
	}

	#[test]
	fn something_that_is_not_a_skeleton_at_all_is_refused() {
		let mut bytes = encoded();
		bytes[0] = b'X';

		assert!(refused(&bytes).contains("not a colby skeleton"), "the magic is checked first");
	}

	#[test]
	fn a_flag_this_build_does_not_know_is_refused() {
		let mut bytes = encoded();
		bytes[12..16].copy_from_slice(&1_u32.to_le_bytes());

		assert!(refused(&bytes).contains("flag bits"), "an unknown flag means an unknown block");
	}

	#[test]
	fn a_word_a_bone_has_no_meaning_for_yet_is_refused() {
		let mut bytes = encoded();
		// the spare half-word sits six bytes into the first record.
		bytes[HEADER_BYTES + 6..HEADER_BYTES + 8].copy_from_slice(&1_u16.to_le_bytes());

		assert!(
			refused(&bytes).contains("no meaning yet"),
			"the same rule the flags word follows"
		);
	}

	#[test]
	fn a_block_that_runs_past_the_end_is_refused() {
		let mut bytes = encoded();
		// where the bones start, which is the twenty-fifth byte.
		bytes[24..28].copy_from_slice(&9000_u32.to_le_bytes());

		assert!(refused(&bytes).contains("bytes long"), "counted against the file's real length");
	}

	#[test]
	fn a_file_claiming_more_bones_than_any_skeleton_may_have_is_refused() {
		let mut bytes = encoded();
		// how many bones there are, which is the twenty-first.
		bytes[20..24].copy_from_slice(&9000_u32.to_le_bytes());

		let message = refused(&bytes);

		assert!(message.contains("9000 bones"), "it says what was claimed: {message}");
		assert!(
			message.contains(&MAX_BONES.to_string()),
			"and refuses it before allocating for it: {message}"
		);
	}

	#[test]
	fn a_truncated_file_is_refused_rather_than_read_short() {
		assert!(refused(&encoded()[..20]).contains("too short"));
	}

	#[test]
	fn the_version_of_a_file_can_be_read_without_reading_the_file() {
		let path = std::env::temp_dir().join("colby-skeleton-version.cskel");

		std::fs::write(&path, encoded()).expect("the fixture is written");

		assert_eq!(version_of(&path), Some(FORMAT_VERSION));
		assert_eq!(version_of(Path::new("nothing.cskel")), None, "and a missing file is none");

		drop(std::fs::remove_file(&path));
	}
}
