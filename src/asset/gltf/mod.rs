//! Reading a glTF file: the container, its buffers, and its accessors.
//!
//! This is the half of glTF import that is about the *file*. What a mesh or a
//! node means is somebody else's problem; what is here is the machinery that
//! turns "accessor seven" into thirty-two triples of floats, and it is the
//! whole of what stands between a `.glb` on disk and plain numbers.
//!
//! **Two containers, one document.** A `.gltf` is a JSON text with its buffers
//! beside it, and a `.glb` is that same JSON in a chunk of a binary file with
//! the buffer in the next chunk. Both arrive here and leave as a [`Gltf`],
//! which is the parsed document plus every buffer already in memory. A buffer
//! may be a file next to the document, a `data:` URI inside it, or the binary
//! chunk of a `.glb`, and nothing above this module can tell which it was.
//!
//! **An accessor is the only way to read a number.** glTF never points at bytes
//! directly: an accessor names a slice of a buffer view, a component type and a
//! count, and everything from a position to a texture coordinate to an index
//! comes through one. [`Gltf::floats`] widens any of the value types to `f32`
//! and applies the normalization rule; [`Gltf::integers`] reads the unsigned
//! ones as they are. Between them that is every accessor an importer of static
//! geometry asks for.
//!
//! **What is refused, and it is refused by name.** A file that declares an
//! extension as *required* is refused, so a mesh compressed with one of the
//! geometry extensions says so instead of arriving empty. A sparse accessor is
//! refused: no exporter this engine is aimed at writes one, and guessing at it
//! is worse than saying no. So is a version that is not 2.
//!
//! ```text
//!   model.glb                              model.gltf + model.bin
//!     header, JSON chunk, BIN chunk          the JSON, and a file beside it
//!            |                                        |
//!            +----------------> Gltf <----------------+
//!                                |
//!                    floats(2) -> [f32; 2] x 32     (a texture coordinate)
//!                    integers(3) -> [u32] x 42      (an index buffer)
//! ```

use std::path::{self, Path, PathBuf};

use colby_core::{Result, err};

use crate::json::{self, Value};

mod geometry;
mod material;
mod skin;

pub use self::{
	geometry::{Model, Piece, Placement, import},
	material::{Extracted, Picture, Surface},
	skin::{Skin, Skins},
};

/// The extension of a glTF written as JSON with its buffers beside it.
pub const EXTENSION: &str = "gltf";

/// The extension of a glTF written as one binary file.
pub const BINARY_EXTENSION: &str = "glb";

/// The four bytes a `.glb` starts with.
const BINARY_MAGIC: [u8; 4] = *b"glTF";

/// The chunk type that holds the document.
const JSON_CHUNK: u32 = 0x4E4F_534A;

/// The chunk type that holds the buffer.
const BINARY_CHUNK: u32 = 0x004E_4942;

/// What every `data:` URI starts with.
const DATA_PREFIX: &str = "data:";

/// A glTF file, checked, with every buffer it names already in memory.
///
/// Buffers are read once, here, rather than on the first accessor that wants
/// one: a document that names a file which is not there is a failure of the
/// whole import, and finding that out halfway through building a mesh would
/// mean deciding what to do with the half that worked.
#[derive(Clone, Debug)]
pub struct Gltf {
	document: Value,
	buffers: Vec<Vec<u8>>,
	source: PathBuf,
	root: PathBuf,
}

impl Gltf {
	/// Reads a `.gltf` or a `.glb` from disk.
	///
	/// @param source - the file, inside the source tree
	/// @param root - the source tree, which a buffer may not leave
	/// @return the document with its buffers, or why it could not be used
	pub fn open(source: &Path, root: &Path) -> Result<Self> {
		let bytes = std::fs::read(source)
			.map_err(|error| err!(Asset("reading {}: {error}", source.display())))?;

		Self::read(&bytes, source, root)
			.map_err(|error| err!(Asset("{}: {error}", source.display())))
	}

	/// The same, from bytes that are already in hand.
	///
	/// @param bytes - the whole file, either container
	/// @param source - where it came from, for resolving what it names
	/// @param root - the source tree, which a buffer may not leave
	/// @return the document with its buffers, or why it could not be used
	pub fn read(bytes: &[u8], source: &Path, root: &Path) -> Result<Self> {
		let (text, binary) = if bytes.starts_with(&BINARY_MAGIC) {
			let (json, chunk) = split_binary(bytes)?;
			let text = std::str::from_utf8(json)
				.map_err(|error| err!(Asset("the document chunk is not text: {error}")))?;

			(text.to_owned(), chunk.map(<[u8]>::to_vec))
		} else {
			let text = std::str::from_utf8(bytes)
				.map_err(|error| err!(Asset("this is not a text document: {error}")))?;

			(text.to_owned(), None)
		};

		let document = json::parse(&text)?;

		check_version(&document)?;
		check_extensions(&document)?;

		let buffers = load_buffers(&document, binary, source, root)?;

		Ok(Self {
			document,
			buffers,
			source: source.to_path_buf(),
			root: root.to_path_buf(),
		})
	}

	/// The document, for walking what this module does not interpret.
	#[must_use]
	pub const fn document(&self) -> &Value { &self.document }

	/// One of the document's top-level arrays.
	///
	/// @param name - `meshes`, `nodes`, `accessors` and the rest
	/// @return its entries, or nothing when the document has none
	#[must_use]
	pub fn table(&self, name: &str) -> &[Value] {
		self.document
			.get(name)
			.map_or(&[], Value::as_array)
	}

	/// A file this document names, resolved beside it and checked.
	///
	/// The same rule the buffers follow, exposed because a material names
	/// pictures the same way and the check that they stay inside the asset
	/// tree should have one implementation.
	///
	/// @param uri - the address as the document wrote it, escapes and all
	/// @return where it is, or nothing when it is not somewhere allowed
	#[must_use]
	pub fn beside(&self, uri: &str) -> Option<PathBuf> {
		let named = unescape(uri)?;
		let path = lexical(&self.source.parent()?.join(named));

		path.starts_with(&self.root).then_some(path)
	}

	/// The source tree this file was read from inside.
	#[must_use]
	pub fn root(&self) -> &Path { &self.root }

	/// One buffer view's bytes, for something stored whole rather than as
	/// numbers.
	///
	/// An accessor is how a document points at *values*; this is how it
	/// points at a picture, which is a file that happens to live inside
	/// another one.
	///
	/// @param view - its index in the document's `bufferViews`
	/// @return the bytes, or why they could not be reached
	pub fn view(&self, view: usize) -> Result<&[u8]> {
		let entry = self
			.table("bufferViews")
			.get(view)
			.ok_or_else(|| err!(Asset("there is no buffer view {view}")))?;
		let index = entry
			.get("buffer")
			.and_then(Value::as_usize)
			.ok_or_else(|| {
				err!(Asset("buffer view {view} does not say which buffer it is in"))
			})?;
		let bytes = self.buffers.get(index).ok_or_else(|| {
			err!(Asset("buffer view {view} names buffer {index}, which is not there"))
		})?;
		let start = number(entry, "byteOffset");
		let length = entry
			.get("byteLength")
			.and_then(Value::as_usize)
			.ok_or_else(|| err!(Asset("buffer view {view} does not say how long it is")))?;

		bytes
			.get(start..start.saturating_add(length))
			.ok_or_else(|| {
				err!(Asset("buffer view {view} reaches past the end of buffer {index}"))
			})
	}

	/// One accessor's values, widened to floats.
	///
	/// Integer components are converted, and a `normalized` accessor is scaled
	/// into `0..1` or `-1..1` the way the specification says. An accessor that
	/// names no buffer view reads as zeros, which is what it means.
	///
	/// @param accessor - its index in the document's `accessors`
	/// @return the values, row by row, or why they could not be read
	pub fn floats(&self, accessor: usize) -> Result<Floats> {
		let plan = self.plan(accessor)?;

		if plan.component == Component::U32 {
			return Err(err!(Asset(
				"accessor {accessor} holds unsigned integers, which is not a kind of value a \
				 vertex has"
			)));
		}

		let mut values = Vec::with_capacity(plan.count.saturating_mul(plan.lanes));

		self.walk(accessor, &plan, |bytes, at| {
			values.push(plan.component.float(bytes, at, plan.normalized));
		})?;

		Ok(Floats { values, lanes: plan.lanes })
	}

	/// One accessor's values, as the whole numbers they are.
	///
	/// For index buffers, which the specification says are unsigned and scalar.
	/// A signed or floating component here is a file saying something it does
	/// not mean, so it is refused rather than rounded.
	///
	/// @param accessor - its index in the document's `accessors`
	/// @return one number per element, or why they could not be read
	pub fn integers(&self, accessor: usize) -> Result<Vec<u32>> {
		let plan = self.plan(accessor)?;

		if plan.lanes != 1 {
			return Err(err!(Asset("accessor {accessor} is not a list of single numbers")));
		}

		if !plan.component.is_unsigned() {
			return Err(err!(Asset(
				"accessor {accessor} holds numbers that may be negative or fractional, which an \
				 index may not be"
			)));
		}

		let mut values = Vec::with_capacity(plan.count);

		self.walk(accessor, &plan, |bytes, at| {
			values.push(plan.component.whole(bytes, at));
		})?;

		Ok(values)
	}

	/// One accessor's values as the whole numbers they are, however wide.
	///
	/// For the bone indices of a skin, which the specification stores as
	/// unsigned bytes or unsigned shorts in fours. [`Self::integers`] is the
	/// scalar case and refuses anything wider, because an index buffer is
	/// scalar by definition; this is the other shape, and it refuses signed
	/// and fractional components for the same reason that one does.
	///
	/// A `normalized` accessor is a file saying its integers stand for
	/// fractions, which an index never does, so it is refused rather than
	/// scaled.
	///
	/// @param accessor - its index in the document's `accessors`
	/// @return the values, row by row, or why they could not be read
	pub fn wholes(&self, accessor: usize) -> Result<Wholes> {
		let plan = self.plan(accessor)?;

		if !plan.component.is_unsigned() {
			return Err(err!(Asset(
				"accessor {accessor} holds numbers that may be negative or fractional, which an \
				 index into anything may not be"
			)));
		}

		if plan.normalized {
			return Err(err!(Asset(
				"accessor {accessor} says its whole numbers stand for fractions, which an index \
				 does not"
			)));
		}

		let mut values = Vec::with_capacity(plan.count.saturating_mul(plan.lanes));

		self.walk(accessor, &plan, |bytes, at| {
			values.push(plan.component.whole(bytes, at));
		})?;

		Ok(Wholes { values, lanes: plan.lanes })
	}

	/// Works out where an accessor's values are and how to read them.
	fn plan(&self, accessor: usize) -> Result<Plan> {
		let entry = self
			.table("accessors")
			.get(accessor)
			.ok_or_else(|| err!(Asset("there is no accessor {accessor}")))?;

		if entry.get("sparse").is_some() {
			return Err(err!(Asset(
				"accessor {accessor} is sparse, which colby does not read; re-export without it"
			)));
		}

		let code = entry
			.get("componentType")
			.and_then(Value::as_u32)
			.ok_or_else(|| {
				err!(Asset("accessor {accessor} does not say what kind of numbers it holds"))
			})?;
		let component = Component::from_code(code).ok_or_else(|| {
			err!(Asset("accessor {accessor} holds numbers of an unknown kind {code}"))
		})?;
		let shape = entry
			.get("type")
			.and_then(Value::as_str)
			.ok_or_else(|| {
				err!(Asset("accessor {accessor} does not say what shape its values are"))
			})?;
		let lanes = lanes_of(shape)
			.ok_or_else(|| err!(Asset("accessor {accessor} is of an unknown shape {shape}")))?;
		let count = entry
			.get("count")
			.and_then(Value::as_usize)
			.ok_or_else(|| {
				err!(Asset("accessor {accessor} does not say how many values it holds"))
			})?;

		Ok(Plan {
			component,
			lanes,
			count,
			normalized: entry
				.get("normalized")
				.and_then(Value::as_bool)
				.unwrap_or(false),
			view: entry.get("bufferView").and_then(Value::as_usize),
			offset: number(entry, "byteOffset"),
		})
	}

	/// Hands every value of an accessor to a reader, in order.
	///
	/// The one place that knows a buffer view may be interleaved: an element
	/// starts every `stride` bytes and its lanes are packed inside it. An
	/// accessor with no view has no bytes at all, and each of its values is a
	/// zero, which is what the specification says one means.
	fn walk<F>(&self, accessor: usize, plan: &Plan, mut take: F) -> Result<()>
	where
		F: FnMut(&[u8], usize),
	{
		let Some(view) = plan.view else {
			for _ in 0..plan.count.saturating_mul(plan.lanes) {
				take(&[], 0);
			}

			return Ok(());
		};

		let (bytes, start, stride) = self.window(accessor, plan, view)?;
		let width = plan.component.bytes();

		for element in 0..plan.count {
			let at = start + element * stride;

			for lane in 0..plan.lanes {
				take(bytes, at + lane * width);
			}
		}

		Ok(())
	}

	/// The bytes an accessor reads from, where it starts inside them, and how
	/// far apart its elements are.
	fn window(&self, accessor: usize, plan: &Plan, view: usize) -> Result<(&[u8], usize, usize)> {
		let entry = self
			.table("bufferViews")
			.get(view)
			.ok_or_else(|| {
				err!(Asset("accessor {accessor} names buffer view {view}, which is not there"))
			})?;
		let index = entry
			.get("buffer")
			.and_then(Value::as_usize)
			.ok_or_else(|| {
				err!(Asset("buffer view {view} does not say which buffer it is in"))
			})?;
		let bytes = self.buffers.get(index).ok_or_else(|| {
			err!(Asset("buffer view {view} names buffer {index}, which is not there"))
		})?;

		let packed = plan.lanes * plan.component.bytes();
		let stride = entry
			.get("byteStride")
			.and_then(Value::as_usize)
			.unwrap_or(packed);

		if stride < packed {
			return Err(err!(Asset(
				"buffer view {view} packs its elements tighter than they are"
			)));
		}

		let start = number(entry, "byteOffset") + plan.offset;
		let span = plan.count.saturating_sub(1) * stride + packed;

		if plan.count > 0 && start.saturating_add(span) > bytes.len() {
			return Err(err!(Asset(
				"accessor {accessor} reads past the end of buffer {index}, which is {} bytes",
				bytes.len()
			)));
		}

		Ok((bytes, start, stride))
	}
}

/// The values one accessor holds, widened to floats.
///
/// Flat rather than a list of arrays because the width is not known until the
/// file says so, and every caller either walks rows or asks for one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Floats {
	values: Vec<f32>,
	lanes: usize,
}

impl Floats {
	/// How many numbers make one value: one for a scalar, three for a position.
	#[must_use]
	pub const fn lanes(&self) -> usize { self.lanes }

	/// How many values there are.
	#[must_use]
	pub const fn rows(&self) -> usize {
		match self.values.len().checked_div(self.lanes) {
			| Some(rows) => rows,
			| None => 0,
		}
	}

	/// One value, or nothing where there is none.
	#[must_use]
	pub fn row(&self, index: usize) -> &[f32] {
		let at = index * self.lanes;

		self.values
			.get(at..at + self.lanes)
			.unwrap_or(&[])
	}

	/// Every number, row after row.
	#[must_use]
	pub fn values(&self) -> &[f32] { &self.values }
}

/// The values one accessor holds, as whole numbers.
///
/// The same shape [`Floats`] has and for the same reason; the two are apart
/// because widening an index to a float and back is a loss nobody asked for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Wholes {
	values: Vec<u32>,
	lanes: usize,
}

impl Wholes {
	/// How many numbers make one value: four, for a set of bone indices.
	#[must_use]
	pub const fn lanes(&self) -> usize { self.lanes }

	/// How many values there are.
	#[must_use]
	pub const fn rows(&self) -> usize {
		match self.values.len().checked_div(self.lanes) {
			| Some(rows) => rows,
			| None => 0,
		}
	}

	/// One value, or nothing where there is none.
	#[must_use]
	pub fn row(&self, index: usize) -> &[u32] {
		let at = index * self.lanes;

		self.values
			.get(at..at + self.lanes)
			.unwrap_or(&[])
	}
}

/// What reading one accessor takes.
struct Plan {
	component: Component,
	lanes: usize,
	count: usize,
	normalized: bool,
	view: Option<usize>,
	offset: usize,
}

/// The kinds of number a glTF accessor may hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Component {
	/// A signed byte.
	I8,

	/// An unsigned byte.
	U8,

	/// A signed short.
	I16,

	/// An unsigned short.
	U16,

	/// An unsigned int, which only an index buffer may be.
	U32,

	/// A single-precision float.
	F32,
}

impl Component {
	/// The kind a stored number stands for.
	const fn from_code(code: u32) -> Option<Self> {
		match code {
			| 5120 => Some(Self::I8),
			| 5121 => Some(Self::U8),
			| 5122 => Some(Self::I16),
			| 5123 => Some(Self::U16),
			| 5125 => Some(Self::U32),
			| 5126 => Some(Self::F32),
			| _ => None,
		}
	}

	/// How wide one of them is.
	const fn bytes(self) -> usize {
		match self {
			| Self::I8 | Self::U8 => 1,
			| Self::I16 | Self::U16 => 2,
			| Self::U32 | Self::F32 => 4,
		}
	}

	/// Whether a value of this kind is a whole number that cannot be negative.
	const fn is_unsigned(self) -> bool { matches!(self, Self::U8 | Self::U16 | Self::U32) }

	/// One value, widened.
	///
	/// A missing byte reads as zero rather than as a failure: the bounds were
	/// checked before the walk started, so the only way to get here is an
	/// accessor with no buffer view, whose values are zeros by definition.
	fn float(self, bytes: &[u8], at: usize, normalized: bool) -> f32 {
		match self {
			| Self::I8 => scale(f32::from(read_i8(bytes, at)), normalized, 127.0),
			| Self::U8 => scale(f32::from(read_u8(bytes, at)), normalized, 255.0),
			| Self::I16 => scale(f32::from(read_i16(bytes, at)), normalized, 32767.0),
			| Self::U16 => scale(f32::from(read_u16(bytes, at)), normalized, 65535.0),
			| Self::F32 => read_f32(bytes, at),
			| Self::U32 => 0.0,
		}
	}

	/// One value, as the whole number it is.
	fn whole(self, bytes: &[u8], at: usize) -> u32 {
		match self {
			| Self::U8 => u32::from(read_u8(bytes, at)),
			| Self::U16 => u32::from(read_u16(bytes, at)),
			| Self::U32 => read_u32(bytes, at),
			| Self::I8 | Self::I16 | Self::F32 => 0,
		}
	}
}

/// How many numbers a named shape holds.
const fn lanes_of(shape: &str) -> Option<usize> {
	match shape.as_bytes() {
		| b"SCALAR" => Some(1),
		| b"VEC2" => Some(2),
		| b"VEC3" => Some(3),
		| b"VEC4" | b"MAT2" => Some(4),
		| b"MAT3" => Some(9),
		| b"MAT4" => Some(16),
		| _ => None,
	}
}

/// A stored integer as the fraction it stands for, when it stands for one.
fn scale(value: f32, normalized: bool, most: f32) -> f32 {
	if normalized { (value / most).max(-1.0) } else { value }
}

/// A byte offset a document wrote, or zero, which is its default everywhere.
fn number(entry: &Value, name: &str) -> usize {
	entry
		.get(name)
		.and_then(Value::as_usize)
		.unwrap_or(0)
}

/// Splits a `.glb` into its document and its buffer.
///
/// @param bytes - the whole file
/// @return the JSON chunk and the binary chunk, if there is one
/// Every file a document names beside itself: its buffers and its pictures.
///
/// What the compiler asks so that editing a `.bin` or a `.png` a model links
/// to rebuilds the model, the same way editing a stylesheet rebuilds every
/// document that links it. Only the document is read, and for a `.glb` only
/// the chunk that holds it, because this runs on every pass over an asset
/// tree and a model may be very large.
///
/// Anything unreadable answers with nothing, which makes the output stale by
/// the check above rather than by an error nobody can act on.
///
/// @param source - the `.gltf` or `.glb`
/// @param root - the source tree, which a reference may not leave
/// @return the files it names, in the order it named them
#[must_use]
pub fn linked(source: &Path, root: &Path) -> Vec<PathBuf> {
	let Some(text) = document_of(source) else {
		return Vec::new();
	};
	let Ok(document) = json::parse(&text) else {
		return Vec::new();
	};
	let Some(directory) = source.parent() else {
		return Vec::new();
	};

	["buffers", "images"]
		.iter()
		.flat_map(|table| {
			document
				.get(table)
				.map_or(&[][..], Value::as_array)
		})
		.filter_map(|entry| entry.get("uri").and_then(Value::as_str))
		.filter(|uri| !uri.starts_with(DATA_PREFIX))
		.filter_map(|uri| {
			let path = lexical(&directory.join(unescape(uri)?));

			path.starts_with(root).then_some(path)
		})
		.collect()
}

/// The JSON of a glTF, without reading any more of the file than that.
fn document_of(source: &Path) -> Option<String> {
	use std::io::Read as _;

	let mut file = std::fs::File::open(source).ok()?;
	let mut head = [0_u8; 20];

	if file.read_exact(&mut head).is_err() || head.get(..4) != Some(&BINARY_MAGIC[..]) {
		return std::fs::read_to_string(source).ok();
	}

	if read_u32(&head, 16) != JSON_CHUNK {
		return None;
	}

	let length = usize::try_from(read_u32(&head, 12)).ok()?;
	let mut body = vec![0_u8; length];

	file.read_exact(&mut body).ok()?;

	String::from_utf8(body).ok()
}

/// The two chunks a `.glb` is made of: its document, and its buffer when it
/// carries one.
type Chunks<'a> = (&'a [u8], Option<&'a [u8]>);

fn split_binary(bytes: &[u8]) -> Result<Chunks<'_>> {
	let version = read_u32(bytes, 4);
	let declared = read_u32(bytes, 8);

	if version != 2 {
		return Err(err!(Asset(
			"this is a version {version} binary file, and colby reads version 2"
		)));
	}

	let length = usize::try_from(declared).unwrap_or(usize::MAX);

	if length > bytes.len() {
		return Err(err!(Asset("the file says it is {length} bytes and it is {}", bytes.len())));
	}

	let mut at = 12;
	let mut document = None;
	let mut buffer = None;

	while let Some(chunk) = next_chunk(bytes, &mut at, length) {
		match chunk.0 {
			| JSON_CHUNK if document.is_none() => document = Some(chunk.1),
			| BINARY_CHUNK if buffer.is_none() => buffer = Some(chunk.1),
			| _ => {},
		}
	}

	let document = document.ok_or_else(|| err!(Asset("the file holds no document chunk")))?;

	Ok((document, buffer))
}

/// One chunk of a `.glb`, and where the next one starts.
///
/// A chunk's declared length already counts the padding that puts the next one
/// on a four-byte boundary, so there is nothing here to round up. Rounding it
/// anyway was tried and is strictly worse: it agrees on every legal file and
/// steps over the next header on a file somebody wrote by hand.
fn next_chunk<'a>(bytes: &'a [u8], at: &mut usize, end: usize) -> Option<(u32, &'a [u8])> {
	let header = at.checked_add(8)?;

	if header > end {
		return None;
	}

	let length = usize::try_from(read_u32(bytes, *at)).ok()?;
	let kind = read_u32(bytes, *at + 4);
	let body = bytes.get(header..header.checked_add(length)?)?;

	*at = header + length;

	Some((kind, body))
}

/// Refuses a document that is not glTF 2.
fn check_version(document: &Value) -> Result<()> {
	let version = document
		.get("asset")
		.and_then(|asset| asset.get("version"))
		.and_then(Value::as_str)
		.ok_or_else(|| err!(Asset("the document does not say which version of glTF it is")))?;

	if !version.starts_with("2.") {
		return Err(err!(Asset("this is glTF {version}, and colby reads version 2")));
	}

	Ok(())
}

/// Refuses a document that needs something this reader does not have.
///
/// The list of what is implemented is empty on purpose. An extension a file
/// merely *uses* is ignored, because that is what the specification says it is
/// for; one it *requires* changes what the rest of the file means, and the two
/// that matter in practice both compress geometry into something that would
/// otherwise be read as nonsense.
fn check_extensions(document: &Value) -> Result<()> {
	let required = document
		.get("extensionsRequired")
		.map_or(&[][..], Value::as_array);

	if let Some(named) = required.first().and_then(Value::as_str) {
		return Err(err!(Asset(
			"this file requires the {named} extension, which colby does not read; re-export \
			 without it"
		)));
	}

	Ok(())
}

/// Reads every buffer a document names.
fn load_buffers(
	document: &Value,
	binary: Option<Vec<u8>>,
	source: &Path,
	root: &Path,
) -> Result<Vec<Vec<u8>>> {
	let mut binary = binary;
	let mut buffers = Vec::new();

	for (index, entry) in document
		.get("buffers")
		.map_or(&[][..], Value::as_array)
		.iter()
		.enumerate()
	{
		let declared = entry
			.get("byteLength")
			.and_then(Value::as_usize)
			.ok_or_else(|| err!(Asset("buffer {index} does not say how long it is")))?;

		let mut bytes = match entry.get("uri").and_then(Value::as_str) {
			| None => binary.take().ok_or_else(|| {
				err!(Asset("buffer {index} names no file and there is none inside"))
			})?,
			| Some(uri) if uri.starts_with(DATA_PREFIX) => inline(uri).ok_or_else(|| {
				err!(Asset("buffer {index} holds a data address colby cannot read"))
			})?,
			| Some(uri) => read_beside(uri, source, root)
				.map_err(|error| err!(Asset("buffer {index}: {error}")))?,
		};

		if bytes.len() < declared {
			return Err(err!(Asset(
				"buffer {index} says it is {declared} bytes and it is {}",
				bytes.len()
			)));
		}

		bytes.truncate(declared);
		buffers.push(bytes);
	}

	Ok(buffers)
}

/// Reads a file a document names, from beside the document.
///
/// The same rule a stylesheet follows: a reference that climbs out of the
/// source tree is refused rather than followed, because a file from somewhere
/// else is one the compiler's staleness check cannot see.
fn read_beside(uri: &str, source: &Path, root: &Path) -> std::result::Result<Vec<u8>, String> {
	let named = unescape(uri).ok_or_else(|| format!("{uri} is not an address colby can read"))?;
	let directory = source
		.parent()
		.ok_or_else(|| format!("{uri} has nothing to be beside"))?;
	let path = lexical(&directory.join(named));

	if !path.starts_with(root) {
		return Err(format!("{uri} is outside the asset tree"));
	}

	std::fs::read(&path).map_err(|error| format!("reading {}: {error}", path.display()))
}

/// Resolves `.` and `..` without touching the filesystem.
///
/// The same reasoning a document's links follow: the check above compares
/// components, so a path that still holds a `..` passes it while pointing
/// somewhere else entirely.
fn lexical(path: &Path) -> PathBuf {
	let mut out = PathBuf::new();

	for part in path.components() {
		match part {
			| path::Component::CurDir => {},
			| path::Component::ParentDir => {
				out.pop();
			},
			| other => out.push(other),
		}
	}

	out
}

/// The bytes a `data:` address carries, when it carries them as base64.
fn inline(uri: &str) -> Option<Vec<u8>> {
	let (head, payload) = uri.split_once(',')?;

	if !head.ends_with(";base64") {
		return None;
	}

	decode(payload)
}

/// Undoes the percent escapes in an address.
fn unescape(uri: &str) -> Option<String> {
	let mut out = Vec::with_capacity(uri.len());
	let bytes = uri.as_bytes();
	let mut at = 0;

	while let Some(&byte) = bytes.get(at) {
		if byte == b'%' {
			let high = char::from(*bytes.get(at + 1)?).to_digit(16)?;
			let low = char::from(*bytes.get(at + 2)?).to_digit(16)?;

			out.push(u8::try_from(high * 16 + low).ok()?);
			at += 3;
		} else {
			out.push(byte);
			at += 1;
		}
	}

	String::from_utf8(out).ok()
}

/// Reads base64, strictly.
///
/// Its own thirty lines rather than a dependency, for the same reason the rest
/// of this crate's readers are: it is a table and a shift, and the crate that
/// does it would be linked by the runner.
fn decode(text: &str) -> Option<Vec<u8>> {
	let mut out = Vec::with_capacity(text.len() / 4 * 3);
	let mut held: u32 = 0;
	let mut bits = 0;

	for byte in text.bytes() {
		if byte == b'=' {
			break;
		}

		let value = sextet(byte)?;
		held = (held << 6) | value;
		bits += 6;

		if bits >= 8 {
			bits -= 8;
			out.push(u8::try_from((held >> bits) & 0xFF).ok()?);
		}
	}

	Some(out)
}

/// One base64 letter as the six bits it stands for.
fn sextet(byte: u8) -> Option<u32> {
	match byte {
		| b'A'..=b'Z' => Some(u32::from(byte - b'A')),
		| b'a'..=b'z' => Some(u32::from(byte - b'a') + 26),
		| b'0'..=b'9' => Some(u32::from(byte - b'0') + 52),
		| b'+' => Some(62),
		| b'/' => Some(63),
		| _ => None,
	}
}

/// A name from a file, as something that can be part of an asset's name.
///
/// Lowercase, and everything outside letters, digits, a dash and an underscore
/// becomes an underscore. Leading and trailing separators go, so a name that
/// was only punctuation comes back empty and the caller numbers it instead.
///
/// **A dot is not kept, and that is not tidiness.** These names become the
/// stems of files, and an asset name is worked out by taking the extension
/// off one - so a mesh called `panel.0` is a file called `panel.0.cmesh`, and
/// every path helper that asks what its extension is answers `0`. Keeping no
/// dots at all means the last one in a path is always the extension.
fn tidy(name: &str) -> String {
	let mut out = String::with_capacity(name.len());

	for letter in name.chars() {
		out.push(match letter {
			| 'a'..='z' | '0'..='9' | '-' | '_' | '.' => letter,
			| 'A'..='Z' => letter.to_ascii_lowercase(),
			| _ => '_',
		});
	}

	out.trim_matches(['_', '-'].as_slice()).to_owned()
}

/// A name nothing else in the list has, remembered.
fn unique(taken: &mut Vec<String>, wanted: &str) -> String {
	let mut name = wanted.to_owned();
	let mut attempt = 1;

	while taken.contains(&name) {
		name = format!("{wanted}_{attempt}");
		attempt += 1;
	}

	taken.push(name.clone());

	name
}

/// One byte, or zero where there is none.
fn read_u8(bytes: &[u8], at: usize) -> u8 { bytes.get(at).copied().unwrap_or(0) }

/// The same, signed.
fn read_i8(bytes: &[u8], at: usize) -> i8 { read_u8(bytes, at).cast_signed() }

/// Two bytes, little-endian, or zero where there are none.
fn read_u16(bytes: &[u8], at: usize) -> u16 {
	bytes
		.get(at..at + 2)
		.and_then(|piece| piece.try_into().ok())
		.map_or(0, u16::from_le_bytes)
}

/// The same, signed.
fn read_i16(bytes: &[u8], at: usize) -> i16 { read_u16(bytes, at).cast_signed() }

/// Four bytes, little-endian, or zero where there are none.
fn read_u32(bytes: &[u8], at: usize) -> u32 {
	bytes
		.get(at..at + 4)
		.and_then(|piece| piece.try_into().ok())
		.map_or(0, u32::from_le_bytes)
}

/// The same, as a float.
fn read_f32(bytes: &[u8], at: usize) -> f32 {
	bytes
		.get(at..at + 4)
		.and_then(|piece| piece.try_into().ok())
		.map_or(0.0, f32::from_le_bytes)
}

#[cfg(test)]
mod tests {
	use std::fs;

	use super::*;

	/// The scene an exporter wrote, as one binary file.
	///
	/// Made with **Blender 5.2.1**, headless, from an empty scene, and it is
	/// the output of a real exporter on purpose: a reader tested only against
	/// text this project wrote would prove that the two agree and nothing
	/// else. Four objects, each there for one reason:
	///
	/// - `column`, a cube scaled unevenly at the origin. The plain case.
	/// - `arm`, a cone parented to the column, so where it stands in the world
	///   is a product of two transforms rather than one.
	/// - `arm_mirror`, a second object sharing `arm`'s mesh with one axis
	///   negated. Its world transform has a negative determinant, which by the
	///   specification reverses the winding of every triangle it draws.
	/// - `panel`, one object wearing two materials, which the exporter splits
	///   into two primitives.
	///
	/// UVs on all of them, and the column's material wears both kinds of
	/// picture, a color and a normal map, so a real exporter's way of writing
	/// images is read rather than guessed at. Rebuilding it is a
	/// half-hour with those five sentences and Blender; nothing asserts on an
	/// exact byte, and whichever version wrote it is in `asset.generator`.
	const BINARY: &[u8] = include_bytes!("fixtures/model.glb");

	/// The same scene exported a second time, as a document with its buffer in
	/// a file beside it.
	///
	/// Two containers rather than one because they are two different paths
	/// through the reader, and only one of them can be tested at a time.
	const TEXT: &[u8] = include_bytes!("fixtures/model.gltf");

	/// The two pictures the separate export leaves beside the document.
	///
	/// Named so that the loose-file rule and the material agree about which
	/// one holds directions, which is what a warning exists to notice when
	/// they do not.
	const COLOR: &[u8] = include_bytes!("fixtures/tiles.png");

	/// The other one.
	const BUMP: &[u8] = include_bytes!("fixtures/tiles_normal.png");

	/// That buffer.
	const BESIDE: &[u8] = include_bytes!("fixtures/model.bin");

	/// Six floats, one after another, written by a tool that is not this one.
	const PLAIN: &str = "AACAPwAAAEAAAEBAAACAQAAAoEAAAMBA";

	/// The same six with four bytes of padding after each triple.
	const INTERLEAVED: &str = "AACAPwAAAEAAAEBAAAAAAAAAgEAAAKBAAADAQAAAAAA=";

	/// Four bytes: 0, 255, 128, 64.
	const BYTES: &str = "AP+AQA==";

	/// The scene, read out of the one file that holds all of it.
	fn binary() -> Gltf {
		Gltf::read(BINARY, Path::new("model.glb"), Path::new(""))
			.expect("the exported binary reads")
	}

	/// A document built around one buffer written as a data address.
	fn around(buffer: &str, length: usize, body: &str) -> String {
		format!(
			"{{ \"asset\": {{ \"version\": \"2.0\" }}, \"buffers\": [ {{ \"byteLength\": \
			 {length}, \"uri\": \"data:application/octet-stream;base64,{buffer}\" }} ], {body} \
			 }}"
		)
	}

	/// The same, read.
	fn read(text: &str) -> Result<Gltf> {
		Gltf::read(text.as_bytes(), Path::new("model.gltf"), Path::new(""))
	}

	/// Why a document was refused.
	fn refusal(text: &str) -> String {
		read(text)
			.expect_err("the document should be refused")
			.to_string()
	}

	/// A directory nobody else is using, removed and recreated.
	fn workspace(name: &str) -> PathBuf {
		let dir = std::env::temp_dir()
			.join("colby-gltf-tests")
			.join(name);

		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(dir.join("models")).expect("the fixture is made");

		dir
	}

	#[test]
	fn the_exported_scene_arrives_with_everything_in_it() {
		let file = binary();

		assert_eq!(file.table("nodes").len(), 4, "a column, two arms and a panel");
		assert_eq!(file.table("meshes").len(), 3, "the two arms share one mesh");
		assert_eq!(file.table("buffers").len(), 1, "and the buffer is inside the file");

		let names: Vec<&str> = file
			.table("meshes")
			.iter()
			.filter_map(|mesh| mesh.get("name").and_then(Value::as_str))
			.collect();

		assert_eq!(names, vec!["arm", "column", "panel"]);
	}

	#[test]
	fn an_object_with_two_materials_arrives_as_two_primitives() {
		let file = binary();
		let counts: Vec<usize> = file
			.table("meshes")
			.iter()
			.map(|mesh| {
				mesh.get("primitives")
					.map_or(&[][..], Value::as_array)
					.len()
			})
			.collect();

		assert_eq!(counts, vec![1, 1, 2], "the panel is the one wearing two");
	}

	#[test]
	fn a_position_accessor_reads_back_the_bounds_it_declares() {
		// the strongest check there is on the offset and stride arithmetic, and
		// it costs nothing: an exporter writes the true extent of every
		// position accessor into the document, so the numbers read out of the
		// buffer have to arrive at the same two corners.
		let file = binary();
		let mut checked = 0;

		for (index, accessor) in file.table("accessors").iter().enumerate() {
			let (Some(low), Some(high)) = (accessor.get("min"), accessor.get("max")) else {
				continue;
			};

			let values = file.floats(index).expect("the accessor reads");

			for lane in 0..values.lanes() {
				let read: Vec<f32> = (0..values.rows())
					.map(|row| values.row(row)[lane])
					.collect();
				let least = read.iter().copied().fold(f32::INFINITY, f32::min);
				let most = read
					.iter()
					.copied()
					.fold(f32::NEG_INFINITY, f32::max);

				assert!(
					(least - low.as_array()[lane].as_f32().unwrap_or_default()).abs() < 1e-5,
					"accessor {index} lane {lane} starts at {least}"
				);
				assert!(
					(most - high.as_array()[lane].as_f32().unwrap_or_default()).abs() < 1e-5,
					"accessor {index} lane {lane} ends at {most}"
				);
			}

			checked += 1;
		}

		assert!(checked >= 4, "every position accessor was measured, and there were {checked}");
	}

	/// Every index of one primitive, against the vertices it may name.
	fn indices_fit(file: &Gltf, primitive: &Value) -> bool {
		let Some(indices) = primitive.get("indices").and_then(Value::as_usize) else {
			return false;
		};
		let position = primitive
			.get("attributes")
			.and_then(|attributes| attributes.get("POSITION"))
			.and_then(Value::as_usize)
			.expect("a primitive has positions");
		let vertices = file
			.floats(position)
			.expect("the positions read")
			.rows();

		for index in file.integers(indices).expect("the indices read") {
			let index = usize::try_from(index).expect("an index fits");

			assert!(index < vertices, "index {index} against {vertices} vertices");
		}

		true
	}

	#[test]
	fn every_index_points_at_a_vertex_that_exists() {
		let file = binary();
		let checked = file
			.table("meshes")
			.iter()
			.flat_map(|mesh| {
				mesh.get("primitives")
					.map_or(&[][..], Value::as_array)
			})
			.filter(|primitive| indices_fit(&file, primitive))
			.count();

		assert_eq!(checked, 4, "four primitives across three meshes");
	}

	#[test]
	fn the_two_containers_hold_the_same_numbers() {
		// the same scene written both ways, so this is the one check that the
		// binary chunk and a file beside the document arrive at one answer.
		let dir = workspace("beside");
		let source = dir.join("models").join("model.gltf");

		fs::write(&source, TEXT).expect("the document is written");
		fs::write(dir.join("models").join("model.bin"), BESIDE).expect("the buffer is written");
		fs::write(dir.join("models").join("tiles.png"), COLOR).expect("the color is written");
		fs::write(dir.join("models").join("tiles_normal.png"), BUMP)
			.expect("the normal map is written");

		let text = Gltf::open(&source, &dir).expect("the document reads");
		let file = binary();

		for index in 0..file.table("accessors").len() {
			assert_eq!(
				file.floats(index).ok(),
				text.floats(index).ok(),
				"accessor {index} disagrees between the two containers"
			);
			assert_eq!(
				file.integers(index).ok(),
				text.integers(index).ok(),
				"accessor {index} disagrees as whole numbers"
			);
		}

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn a_buffer_written_into_the_document_is_read() {
		let text = around(
			PLAIN,
			24,
			"\"bufferViews\": [ { \"buffer\": 0, \"byteLength\": 24 } ], \"accessors\": [ { \
			 \"bufferView\": 0, \"componentType\": 5126, \"count\": 2, \"type\": \"VEC3\" } ]",
		);
		let values = read(&text)
			.expect("the document reads")
			.floats(0)
			.expect("the accessor reads");

		assert_eq!(values.lanes(), 3);
		assert_eq!(values.rows(), 2);
		assert_eq!(values.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
	}

	#[test]
	fn elements_spaced_further_apart_than_they_are_wide_are_stepped_over() {
		let text = around(
			INTERLEAVED,
			32,
			"\"bufferViews\": [ { \"buffer\": 0, \"byteLength\": 32, \"byteStride\": 16 } ], \
			 \"accessors\": [ { \"bufferView\": 0, \"componentType\": 5126, \"count\": 2, \
			 \"type\": \"VEC3\" } ]",
		);
		let values = read(&text)
			.expect("the document reads")
			.floats(0)
			.expect("the accessor reads");

		assert_eq!(values.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "the padding is skipped");
	}

	#[test]
	fn a_normalized_accessor_is_scaled_into_a_fraction() {
		let text = around(
			BYTES,
			4,
			"\"bufferViews\": [ { \"buffer\": 0, \"byteLength\": 4 } ], \"accessors\": [ { \
			 \"bufferView\": 0, \"componentType\": 5121, \"normalized\": true, \"count\": 2, \
			 \"type\": \"VEC2\" } ]",
		);
		let values = read(&text)
			.expect("the document reads")
			.floats(0)
			.expect("the accessor reads");

		assert_eq!(values.row(0), &[0.0, 1.0], "nought and the whole of it");
		assert!((values.row(1)[0] - 128.0 / 255.0).abs() < 1e-6, "and a fraction between");
	}

	#[test]
	fn the_same_bytes_unnormalized_are_the_numbers_themselves() {
		let text = around(
			BYTES,
			4,
			"\"bufferViews\": [ { \"buffer\": 0, \"byteLength\": 4 } ], \"accessors\": [ { \
			 \"bufferView\": 0, \"componentType\": 5121, \"count\": 2, \"type\": \"VEC2\" } ]",
		);
		let values = read(&text)
			.expect("the document reads")
			.floats(0)
			.expect("the accessor reads");

		assert_eq!(values.values(), &[0.0, 255.0, 128.0, 64.0]);
	}

	#[test]
	fn an_accessor_with_no_bytes_behind_it_reads_as_zeros() {
		let text = around(
			PLAIN,
			24,
			"\"accessors\": [ { \"componentType\": 5126, \"count\": 2, \"type\": \"VEC3\" } ]",
		);
		let values = read(&text)
			.expect("the document reads")
			.floats(0)
			.expect("the accessor reads");

		assert_eq!(values.values(), &[0.0; 6]);
	}

	#[test]
	fn an_index_reads_as_a_whole_number_and_nothing_else_may_pretend_to_be_one() {
		let file = binary();

		assert_eq!(
			file.integers(3)
				.expect("the arm's indices read")
				.len(),
			42
		);
		assert!(
			file.integers(0)
				.is_err_and(|error| error.to_string().contains("single numbers")),
			"a position is three numbers at a time, so it is not an index"
		);

		// one number at a time and still not an index, which is the other half
		// of the rule and the half a shape check would miss.
		let text = around(
			PLAIN,
			24,
			"\"bufferViews\": [ { \"buffer\": 0, \"byteLength\": 24 } ], \"accessors\": [ { \
			 \"bufferView\": 0, \"componentType\": 5126, \"count\": 6, \"type\": \"SCALAR\" } ]",
		);

		assert!(
			read(&text)
				.expect("the document reads")
				.integers(0)
				.is_err_and(|error| error.to_string().contains("may not be")),
			"a float is not an index however few of them there are"
		);
	}

	#[test]
	fn a_document_that_needs_something_colby_does_not_read_says_which() {
		let text = "{ \"asset\": { \"version\": \"2.0\" }, \"extensionsRequired\": [ \
		            \"KHR_draco_mesh_compression\" ] }";
		let message = refusal(text);

		assert!(message.contains("KHR_draco_mesh_compression"), "got {message}");
		assert!(message.contains("re-export"), "and says what to do about it: {message}");
	}

	#[test]
	fn a_version_that_is_not_two_is_refused_and_a_later_minor_one_is_not() {
		assert!(refusal("{ \"asset\": { \"version\": \"1.0\" } }").contains("glTF 1.0"));
		assert!(refusal("{ \"asset\": {} }").contains("does not say which version"));
		assert!(
			read("{ \"asset\": { \"version\": \"2.7\" } }").is_ok(),
			"a minor version is forward compatible by the specification"
		);
	}

	#[test]
	fn a_sparse_accessor_is_refused_rather_than_read_as_dense() {
		let text = around(
			PLAIN,
			24,
			"\"accessors\": [ { \"componentType\": 5126, \"count\": 2, \"type\": \"VEC3\", \
			 \"sparse\": { \"count\": 1 } } ]",
		);
		let message = read(&text)
			.expect("the document reads")
			.floats(0)
			.expect_err("the accessor is refused")
			.to_string();

		assert!(message.contains("sparse"), "got {message}");
	}

	#[test]
	fn an_accessor_that_reads_past_its_buffer_is_refused() {
		let text = around(
			PLAIN,
			24,
			"\"bufferViews\": [ { \"buffer\": 0, \"byteLength\": 24 } ], \"accessors\": [ { \
			 \"bufferView\": 0, \"componentType\": 5126, \"count\": 9, \"type\": \"VEC3\" } ]",
		);
		let message = read(&text)
			.expect("the document reads")
			.floats(0)
			.expect_err("the accessor is refused")
			.to_string();

		assert!(message.contains("past the end"), "got {message}");
	}

	#[test]
	fn elements_packed_tighter_than_they_are_wide_are_refused() {
		// nothing legal says this, and read as written it would hand back
		// values that overlap each other rather than a mesh.
		let text = around(
			PLAIN,
			24,
			"\"bufferViews\": [ { \"buffer\": 0, \"byteLength\": 24, \"byteStride\": 4 } ], \
			 \"accessors\": [ { \"bufferView\": 0, \"componentType\": 5126, \"count\": 2, \
			 \"type\": \"VEC3\" } ]",
		);
		let message = read(&text)
			.expect("the document reads")
			.floats(0)
			.expect_err("the accessor is refused")
			.to_string();

		assert!(message.contains("tighter"), "got {message}");
	}

	#[test]
	fn a_buffer_shorter_than_it_claims_is_refused() {
		let text =
			around(PLAIN, 64, "\"bufferViews\": [ { \"buffer\": 0, \"byteLength\": 24 } ]");
		let message = refusal(&text);

		assert!(message.contains("64 bytes and it is 24"), "got {message}");
	}

	#[test]
	fn a_buffer_that_climbs_out_of_the_asset_tree_is_refused() {
		let dir = workspace("escape");
		let source = dir.join("models").join("model.gltf");
		let text = "{ \"asset\": { \"version\": \"2.0\" }, \"buffers\": [ { \"byteLength\": 4, \
		            \"uri\": \"../../secrets.bin\" } ] }";

		fs::write(&source, text).expect("the document is written");

		let message = Gltf::open(&source, &dir)
			.expect_err("the buffer is refused")
			.to_string();

		assert!(message.contains("outside the asset tree"), "got {message}");

		drop(fs::remove_dir_all(&dir));
	}

	#[test]
	fn a_binary_file_that_is_not_one_is_refused_by_name() {
		let mut wrong = BINARY.to_vec();
		wrong[4] = 1;

		assert!(
			Gltf::read(&wrong, Path::new("model.glb"), Path::new(""))
				.expect_err("a version one binary is refused")
				.to_string()
				.contains("version 1 binary"),
		);

		let short = &BINARY[..BINARY.len() - 1];

		assert!(
			Gltf::read(short, Path::new("model.glb"), Path::new(""))
				.expect_err("a truncated file is refused")
				.to_string()
				.contains("bytes and it is"),
		);
	}

	#[test]
	fn a_percent_escape_in_an_address_is_undone() {
		let dir = workspace("escaped");
		let source = dir.join("models").join("model.gltf");
		let text = "{ \"asset\": { \"version\": \"2.0\" }, \"buffers\": [ { \"byteLength\": 4, \
		            \"uri\": \"a%20buffer.bin\" } ] }";

		fs::write(&source, text).expect("the document is written");
		fs::write(dir.join("models").join("a buffer.bin"), [1, 2, 3, 4])
			.expect("the buffer is written");

		let file = Gltf::open(&source, &dir).expect("the document reads");

		assert_eq!(file.table("buffers").len(), 1, "the address named a real file");

		drop(fs::remove_dir_all(&dir));
	}
}
