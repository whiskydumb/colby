//! What an entity carries that the engine does not have to know the type of.
//!
//! **A record is a set of fields every entity carries**, declared once by
//! whoever owns them - the engine for what the renderer reads, a game for
//! everything else - and then reached by everything that walks a world: an
//! inspector draws a row per field, a scene source says `"records": { "door":
//! { "speed": 2 } }` on an entity, a save and a piece of the world on the wire
//! carry the values, and the code that declared the record reads the struct it
//! declared it over, by handle, every step.
//!
//! **A row is a name, a kind and an offset, and nothing the host has to
//! call.** The bytes live here, in the host, because the module holds no
//! state; what a game hands over is where in its struct each field sits. A
//! table of function pointers would do the same job and carry the module's
//! lifetime with it: a pointer into an image a reload frees, and nothing left
//! to read the old values with once it has. An offset is data, and survives
//! the image it came from.
//!
//! **Everything a declaration names is copied**, the rule a console variable
//! follows and for its reason: a `&'static str` from a game module points into
//! an image the host is about to unload. Nothing here keeps a [`Row`] or a
//! [`Record`]; a declaration is read once and forgotten.
//!
//! **Every entity carries every declared record**, at the record's own default
//! until something writes it - the arrangement a light has, where an entity
//! that is not a lamp holds a light that is off. There is no attaching and no
//! detaching: a game that wants to know whether a thing is a door gives the
//! record a field that says so.
//!
//! **What is written down is written by name.** A value leaves a world as the
//! record's name, the field's name and a [`Spelled`] that carries no kind -
//! true or false, a number, two to four numbers, a word - and lands again
//! wherever a declared record has a field of that name the spelling fits. So a
//! world saved by one build of a game loads into the next: a field renamed or
//! retyped is dropped with a warning, and a record nobody has declared yet is
//! kept by name until somebody does.
//!
//! **That last case is not rare, it is how a world starts.** The startup scene
//! is put in place before the game module is loaded, so every value a scene
//! gives a game's record waits, by name, for the `init` that declares it - and
//! a module that fails to load at all loses nothing a save would write back.
//!
//! **A reload is a save and a load.** A module's records are marked when it
//! goes; the new build declares them again, and a record declared with other
//! fields takes what the old one held by name, onto the new defaults, exactly
//! as a file would put it back. A record the new build no longer declares
//! waits by name like any other.

use super::{
	cvar::Owner,
	field::{Kind, Value},
	names::MAX_NAME,
};
use crate::{
	Err, Result,
	bytemuck::{self, Pod},
	glam::{Quat, Vec2, Vec3},
	info, warn,
};

/// How many records one world may declare.
///
/// A handful is what a game has - a door, a spawner, a pickup - and every
/// entity carries every one of them, so the ceiling is also what keeps a
/// thousand entities from carrying a thousand kinds of nothing.
pub const MAX_RECORDS: usize = 16;

/// How many four-byte words one record may be.
pub const MAX_WORDS: usize = 32;

/// How many fields one record may have.
pub const MAX_ROWS: usize = 32;

/// How many bytes one word is: every field of a record is a whole number of
/// them, so a record is words and nothing but words, and has no padding for a
/// value to hide in.
const WORD: usize = 4;

/// How the editor draws a field of a record, when it draws one at all.
///
/// **A game's own thing with nothing to look at is the one place a helper
/// cannot be code.** The engine's kinds - a lamp, a thrower, a decal - are few
/// and the editor draws each by hand; a game's record is a shape nobody here
/// has ever seen, its code does not run while a world is being edited, and a
/// pointer into a module would die at the next reload. What crosses instead is
/// this word beside the field, the way the name, the help line and the kind
/// already do. One other engine in the field offers the same thing and offers
/// exactly the second of these: a place you drag.
///
/// Both are read in the entity's **own** terms - a radius in world units out
/// from where it stands, a place inside it - so a door carried across a room
/// takes its numbers with it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Draw {
	/// Nothing: the field is a number in a panel and that is all.
	#[default]
	None,

	/// How far something reaches, in world units: three circles around the
	/// thing, and a handle on the rim. Only a [`Float`](super::field::Kind).
	Radius,

	/// A place in the thing's own space: a line out to it, and a handle there.
	/// Only a [`Vec3`](super::field::Kind).
	Point,
}

impl Draw {
	/// The word it is named by in a message.
	#[must_use]
	pub const fn word(self) -> &'static str {
		match self {
			| Self::None => "nothing",
			| Self::Radius => "a radius",
			| Self::Point => "a place",
		}
	}

	/// Whether a field of this kind can be drawn this way.
	///
	/// @param kind - what the field holds
	#[must_use]
	pub const fn fits(self, kind: Kind) -> bool {
		match self {
			| Self::None => true,
			| Self::Radius => matches!(kind, Kind::Float),
			| Self::Point => matches!(kind, Kind::Vec3),
		}
	}
}

/// One field of a record: what it is called, what it holds, where it sits, and
/// how the editor draws it.
///
/// Written with [`row!`](crate::row) out of the struct's own field, so that
/// the name, the offset and the type cannot disagree with the struct.
#[derive(Clone, Copy, Debug)]
pub struct Row {
	/// The struct's own field name: a file's key and an inspector's label.
	pub name: &'static str,

	/// One line saying what it is, for a tooltip.
	pub help: &'static str,

	/// What it holds. A record holds flags, numbers and words; text and
	/// handles are refused when it is declared, because a file can spell
	/// neither without a world to resolve it in.
	pub kind: Kind,

	/// How far into the record it starts, in bytes.
	pub offset: usize,

	/// How the editor draws it, or [`Draw::None`] for a field it does not.
	pub draw: Draw,
}

impl Row {
	/// The same row, drawn.
	///
	/// Written after [`row!`](crate::row) rather than inside it, so that a game
	/// that draws nothing writes what it always wrote:
	///
	/// ```ignore
	/// row!(Float, Door, reach, "how far it notices somebody").drawn(Draw::Radius)
	/// ```
	///
	/// @param draw - how to draw it; refused when the field does not hold what
	/// that way of drawing needs
	#[must_use]
	pub const fn drawn(self, draw: Draw) -> Self {
		Self {
			name: self.name,
			help: self.help,
			kind: self.kind,
			offset: self.offset,
			draw,
		}
	}
}

/// A record every entity carries, as the code declaring it spells it.
///
/// ```ignore
/// #[repr(C)]
/// #[derive(Clone, Copy, colby_core::bytemuck::Pod, colby_core::bytemuck::Zeroable)]
/// #[bytemuck(crate = "::colby_core::bytemuck")]
/// struct Door { open: u32, speed: f32 }
///
/// const DOOR: Record<Door> = Record {
///     name: "door",
///     help: "a thing that swings",
///     rows: &[
///         row!(Bool, Door, open, "whether it stands open"),
///         row!(Float, Door, speed, "how far it swings in a second, in turns"),
///     ],
///     default: Door { open: 0, speed: 0.25 },
/// };
///
/// world.entities.declare(&DOOR)?;                          // in init
/// if let Some(door) = world.entities.record_mut(&DOOR, id) { door.open = 1; }
/// ```
///
/// The struct is `#[repr(C)]` and `Pod` because the host reads it as bytes; a
/// flag is a `u32` rather than a `bool` for that reason, and anything but
/// nought reads as true.
#[derive(Clone, Copy, Debug)]
pub struct Record<T> {
	/// What the record is called: lowercase letters, digits and underscores.
	pub name: &'static str,

	/// One line saying what it is.
	pub help: &'static str,

	/// Its fields, in the order an inspector and a file list them.
	pub rows: &'static [Row],

	/// What every entity's copy starts as.
	pub default: T,
}

impl<T> Record<T> {
	/// Its name and its fields, without the type it is declared over.
	#[must_use]
	pub const fn shape(&self) -> Shape { Shape { name: self.name, rows: self.rows } }
}

/// A record's name and its fields: what a reader that never holds one needs.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
	/// What the record is called.
	pub name: &'static str,

	/// Its fields.
	pub rows: &'static [Row],
}

/// The records the engine declares itself, which every world carries.
///
/// A scene source that writes one of these is checked against it when it is
/// compiled, the way every other table the engine owns is. A record a game
/// declares cannot be: the compiler never loads a game.
pub const ENGINE: &[Shape] = &[
	super::entity::DRAWING.shape(),
	super::entity::EDITING.shape(),
	super::entity::BAKING.shape(),
];

/// Checks that a field is stored as the type its kind is held in.
///
/// Called by [`row!`](crate::row) and by nothing else: the reach is never
/// called, it only has to type-check.
#[doc(hidden)]
pub const fn stored_as<R, T>(_reach: fn(&R) -> &T) {}

/// One [`Row`] of a record, out of the struct's own field.
///
/// The name is the field's, the offset is where the struct puts it, and the
/// field has to be the type its kind is held as or the row does not compile: a
/// flag and a word are a `u32`, a whole number an `i32`, a number an `f32`, and
/// two, three or four numbers an array of `f32`.
///
/// ```ignore
/// row!(Float, Door, speed, "how far it swings in a second, in turns")
/// row!(Word(&["swing", "slide"]), Door, style, "how it opens")
/// ```
#[macro_export]
macro_rules! row {
	(Bool, $record:ty, $field:ident, $help:literal) => {
$crate::row!(@stored Bool, u32, $record, $field, $help)
	};
	(Int, $record:ty, $field:ident, $help:literal) => {
$crate::row!(@stored Int, i32, $record, $field, $help)
	};
	(Float, $record:ty, $field:ident, $help:literal) => {
$crate::row!(@stored Float, f32, $record, $field, $help)
	};
	(Vec2, $record:ty, $field:ident, $help:literal) => {
$crate::row!(@stored Vec2, [f32; 2], $record, $field, $help)
	};
	(Vec3, $record:ty, $field:ident, $help:literal) => {
$crate::row!(@stored Vec3, [f32; 3], $record, $field, $help)
	};
	(Color, $record:ty, $field:ident, $help:literal) => {
$crate::row!(@stored Color, [f32; 3], $record, $field, $help)
	};
	(Quat, $record:ty, $field:ident, $help:literal) => {
$crate::row!(@stored Quat, [f32; 4], $record, $field, $help)
	};
	(Word($words:expr), $record:ty, $field:ident, $help:literal) => {
		$crate::abi::record::Row {
			name: ::core::stringify!($field),
			help: $help,
			kind: $crate::abi::field::Kind::Word($words),
			offset: {
				$crate::abi::record::stored_as::<$record, u32>(|record| &record.$field);
				::core::mem::offset_of!($record, $field)
			},
			draw: $crate::abi::record::Draw::None,
		}
	};
	(@stored $kind:ident, $stored:ty, $record:ty, $field:ident, $help:literal) => {
		$crate::abi::record::Row {
			name: ::core::stringify!($field),
			help: $help,
			kind: $crate::abi::field::Kind::$kind,
			draw: $crate::abi::record::Draw::None,
			offset: {
				$crate::abi::record::stored_as::<$record, $stored>(|record| &record.$field);
				::core::mem::offset_of!($record, $field)
			},
		}
	};
}

/// A value written down without a kind, the way a file spells one.
///
/// What a record's field holds is decided by the record, and the record may
/// not be there when the value is read - it may not have been declared yet, or
/// declared by a build that has since changed it. So a value is carried as the
/// spelling a person would write, and becomes a kind of value only when it
/// meets a field.
#[derive(Clone, Debug, PartialEq)]
pub enum Spelled {
	/// `true` or `false`: a flag.
	Truth(bool),

	/// One number: a whole number or a number. A double, so that every whole
	/// number a field holds survives the trip.
	Number(f64),

	/// Two to four numbers: a pair, a vector, a color or a rotation.
	Numbers(Vec<f32>),

	/// One of a field's words, by its spelling rather than its place, so a
	/// list that grew or was reordered still finds it.
	Word(String),
}

/// One field of one record on one entity, written down by name.
#[derive(Clone, Debug, PartialEq)]
pub struct Noted {
	/// Which record.
	pub record: String,

	/// Which of its fields.
	pub field: String,

	/// What it holds.
	pub value: Spelled,
}

/// A value written down that no declared record could take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused {
	/// The record it named.
	pub record: String,

	/// The field it named.
	pub field: String,

	/// Why it was not taken.
	pub why: Why,
}

impl Refused {
	/// A refusal of one written value.
	fn of(noted: &Noted, why: Why) -> Self {
		Self {
			record: noted.record.clone(),
			field: noted.field.clone(),
			why,
		}
	}
}

/// Why a written value was not taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
	/// The record has no field of that name: renamed, or never there.
	Missing,

	/// It has one, and the value is not something it holds: retyped, or a word
	/// that is not one of its words.
	Unfit,
}

/// What a declaration came to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Declared {
	/// How many values already in the world it took: ones waiting for it by
	/// name, or what a previous build of it held, carried over.
	pub kept: usize,

	/// What it could not take.
	pub refused: Vec<Refused>,
}

/// What a declared field holds, as the host keeps it: a [`Kind`], with a
/// word's list copied out of the image that declared it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColumnKind {
	/// A flag, one word.
	Bool,

	/// A whole number, one word.
	Int,

	/// A number, one word.
	Float,

	/// Two numbers.
	Vec2,

	/// Three numbers.
	Vec3,

	/// A color, three numbers.
	Color,

	/// A rotation, four numbers xyzw.
	Quat,

	/// One of a few words, held as its place in the list.
	Word(Vec<String>),
}

impl ColumnKind {
	/// The kind a row declares, or nothing for one a record cannot hold.
	fn of(kind: Kind) -> Option<Self> {
		Some(match kind {
			| Kind::Bool => Self::Bool,
			| Kind::Int => Self::Int,
			| Kind::Float => Self::Float,
			| Kind::Vec2 => Self::Vec2,
			| Kind::Vec3 => Self::Vec3,
			| Kind::Color => Self::Color,
			| Kind::Quat => Self::Quat,
			| Kind::Word(words) => Self::Word(
				words
					.iter()
					.map(|word| (*word).to_owned())
					.collect(),
			),
			| Kind::Text
			| Kind::Entity
			| Kind::Body
			| Kind::Joint
			| Kind::Pose
			| Kind::Mesh
			| Kind::Material
			| Kind::Texture => return None,
		})
	}

	/// How many words a value of this kind takes.
	#[must_use]
	pub const fn width(&self) -> usize {
		match self {
			| Self::Bool | Self::Int | Self::Float | Self::Word(_) => 1,
			| Self::Vec2 => 2,
			| Self::Vec3 | Self::Color => 3,
			| Self::Quat => 4,
		}
	}

	/// The words a [`Word`](Self::Word) may be, or none for any other kind.
	#[must_use]
	pub fn words(&self) -> &[String] {
		match self {
			| Self::Word(words) => words,
			| _ => &[],
		}
	}
}

/// Whether a spelled value is one a field of this kind could hold.
///
/// For a reader that has the row and not the world: a scene source naming one
/// of the engine's own records is checked with this when it is compiled.
///
/// @param kind - what the field holds
/// @param value - what was written
#[must_use]
pub fn fits(kind: Kind, value: &Spelled) -> bool {
	ColumnKind::of(kind).is_some_and(|kind| fit(&kind, value).is_some())
}

/// One field of a declared record, as the host keeps it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Column {
	name: String,
	help: String,
	kind: ColumnKind,
	/// Which word of the record it starts at.
	at: usize,
	draw: Draw,
}

impl Column {
	/// What it is called.
	#[must_use]
	pub fn name(&self) -> &str { &self.name }

	/// One line saying what it is.
	#[must_use]
	pub fn help(&self) -> &str { &self.help }

	/// What it holds.
	#[must_use]
	pub const fn kind(&self) -> &ColumnKind { &self.kind }

	/// How the editor draws it, or [`Draw::None`] for a field it does not.
	#[must_use]
	pub const fn draw(&self) -> Draw { self.draw }

	/// The words its value takes up in one record.
	const fn span(&self) -> core::ops::Range<usize> { self.at..self.at + self.kind.width() }
}

/// One declared record: its fields, its default, and every slot's copy.
#[derive(Clone, Debug)]
pub struct Table {
	name: String,
	help: String,
	columns: Vec<Column>,
	/// How many words one record is.
	words: usize,
	/// What a record starts as, word for word.
	default: Vec<u32>,
	owner: Owner,
	/// Whether the module that declared it has gone without declaring it
	/// again. @ref [`Records::sweep`].
	stale: bool,
	/// Every slot's record, back to back.
	values: Vec<u32>,
}

impl Table {
	/// What it is called.
	#[must_use]
	pub fn name(&self) -> &str { &self.name }

	/// One line saying what it is.
	#[must_use]
	pub fn help(&self) -> &str { &self.help }

	/// Its fields, in the order they were declared.
	#[must_use]
	pub fn columns(&self) -> &[Column] { &self.columns }

	/// Who declared it.
	#[must_use]
	pub const fn owner(&self) -> Owner { self.owner }

	/// One slot's record.
	fn slot(&self, slot: usize) -> Option<&[u32]> {
		self.values
			.get(slot * self.words..(slot + 1) * self.words)
	}

	/// One slot's record, to change.
	fn slot_mut(&mut self, slot: usize) -> Option<&mut [u32]> {
		self.values
			.get_mut(slot * self.words..(slot + 1) * self.words)
	}

	/// Whether another declaration describes the same record: the same fields
	/// in the same places, and the same default. A help line that changed is
	/// not a different record.
	fn same_shape(&self, other: &Self) -> bool {
		self.words == other.words
			&& self.default == other.default
			&& self.columns.len() == other.columns.len()
			&& self
				.columns
				.iter()
				.zip(&other.columns)
				.all(|(one, two)| {
					one.name == two.name && one.kind == two.kind && one.at == two.at
				})
	}

	/// What one slot's record holds that its default does not, written down.
	///
	/// A word past the end of its list has no spelling and is left out, which
	/// reads back as the default: the only answer a list can give a number it
	/// does not have.
	fn noted(&self, slot: usize) -> Vec<Noted> {
		let Some(held) = self.slot(slot) else {
			return Vec::new();
		};

		self.columns
			.iter()
			.filter_map(|column| {
				let words = held.get(column.span())?;

				if Some(words) == self.default.get(column.span()) {
					return None;
				}

				Some(Noted {
					record: self.name.clone(),
					field: column.name.clone(),
					value: spell(&column.kind, words)?,
				})
			})
			.collect()
	}

	/// Puts one written value into one slot's record, if a field takes it.
	fn take(&mut self, slot: usize, noted: &Noted) -> core::result::Result<(), Why> {
		let column = self
			.columns
			.iter()
			.find(|column| column.name == noted.field)
			.ok_or(Why::Missing)?;
		let span = column.span();
		let words = fit(&column.kind, &noted.value).ok_or(Why::Unfit)?;

		// a slot the table does not have is a caller that resolved nothing, and
		// there is nothing for the value to be dropped from
		if let Some(held) = self
			.slot_mut(slot)
			.and_then(|record| record.get_mut(span.clone()))
		{
			held.copy_from_slice(words.get(..span.len()).unwrap_or_default());
		}

		Ok(())
	}
}

/// Every declared record, and every slot's copy of each.
///
/// Held by [`Entities`](super::Entities), which hands a slot out, sizes the
/// table for a restore and grows it, and resolves a handle before anything
/// here is asked about a slot - the arrangement the names beside it have.
#[derive(Clone, Debug, Default)]
pub struct Records {
	tables: Vec<Table>,

	/// What each slot was told about records nobody has declared, by name.
	waiting: Vec<Vec<Noted>>,

	/// Who a declaration from here on belongs to. @ref [`Records::attribute`].
	owner: Owner,
}

impl Records {
	/// No records, for no slots.
	#[must_use]
	pub const fn new() -> Self {
		Self {
			tables: Vec::new(),
			waiting: Vec::new(),
			owner: Owner::Engine,
		}
	}

	/// Declares a record, or declares it again.
	///
	/// Idempotent, because a game declares from `init` and `init` runs again
	/// on every reload. What happens the second time:
	///
	/// - **the same fields** change nothing, and every value is left where it
	///   is;
	/// - **other fields, across a reload** take what the old record held by
	///   name, onto the new defaults, and say how much was carried and what was
	///   not;
	/// - **other fields otherwise** are refused: two records of one name in one
	///   build are two things that would overwrite each other.
	///
	/// A new record takes whatever waited for it by name.
	///
	/// @param record - what to declare
	/// @return what it took and what it refused
	///
	/// # Errors
	///
	/// When the record cannot be held - a name that is not one, a field that
	/// is text or a handle, a field past the end of the struct or over another,
	/// too many of anything - or when another record already has its name.
	/// Nothing is changed then, and the reason is logged as well as returned.
	pub fn declare<T: Pod>(&mut self, record: &Record<T>) -> Result<Declared> {
		let mut table = table_of(record, self.owner)?;
		table.values = table.default.repeat(self.waiting.len());

		if let Some(index) = self
			.tables
			.iter()
			.position(|held| held.name == table.name)
		{
			return self.again(index, table);
		}

		if self.tables.len() >= MAX_RECORDS {
			return Err!(Module(error!(
				"the record {} is one more than the {MAX_RECORDS} a world holds",
				table.name
			)));
		}

		self.tables.push(table);

		let mut declared = Declared::default();
		self.claim(self.tables.len() - 1, &mut declared);
		report(&declared.refused, "declaring a record");

		Ok(declared)
	}

	/// Marks everything declared from here on as one owner's.
	///
	/// The host's to call: it sets the game module's around its `init` and
	/// leaves it, the way the console table is attributed. @ref
	/// [`forget_module`](Self::forget_module).
	///
	/// @param owner - who declares
	pub const fn attribute(&mut self, owner: Owner) { self.owner = owner; }

	/// Marks the game module's records, because the module is going.
	///
	/// Nothing is dropped: the values are plain data, and a reload should not
	/// throw away a number somebody typed into an inspector. What the mark is
	/// for is the record the next build does not declare again, which
	/// [`sweep`](Self::sweep) turns into values waiting by name.
	pub fn forget_module(&mut self) {
		for table in &mut self.tables {
			if table.owner == Owner::Module {
				table.stale = true;
			}
		}
	}

	/// Takes away what the reloaded module did not declare again.
	///
	/// The host's to call after the module's `init`. What such a record held is
	/// kept by name, waiting, so a build that brings the record back finds it
	/// and a save written meanwhile still carries it.
	pub fn sweep(&mut self) {
		let (gone, kept): (Vec<Table>, Vec<Table>) = core::mem::take(&mut self.tables)
			.into_iter()
			.partition(|table| table.stale);

		self.tables = kept;

		for table in gone {
			let mut waiting = 0;

			for (slot, notes) in self.waiting.iter_mut().enumerate() {
				let noted = table.noted(slot);

				waiting += noted.len();
				notes.extend(noted);
			}

			warn!(
				record = table.name,
				waiting, "a record nobody declares any more; what it held waits for it by name"
			);
		}
	}

	/// Every declared record, in the order it was declared.
	#[must_use]
	pub fn tables(&self) -> &[Table] { &self.tables }

	/// Adds a slot, holding every record's default and nothing waiting.
	pub(crate) fn push(&mut self) {
		self.waiting.push(Vec::new());

		for table in &mut self.tables {
			table.values.extend_from_slice(&table.default);
		}
	}

	/// Sizes every record for a restore, each slot at every default.
	///
	/// @param slots - how many slots the table ends up with
	pub(crate) fn reset(&mut self, slots: usize) {
		self.waiting.clear();
		self.waiting.resize_with(slots, Vec::new);

		for table in &mut self.tables {
			table.values = table.default.repeat(slots);
		}
	}

	/// Puts one slot back to every default, with nothing waiting.
	///
	/// Where a slot is handed out rather than where one is given back, the
	/// rule a name follows: nothing reads a dead slot's record.
	///
	/// @param slot - the array index, not a handle
	pub(crate) fn clear(&mut self, slot: usize) {
		if let Some(waiting) = self.waiting.get_mut(slot) {
			waiting.clear();
		}

		for table in &mut self.tables {
			let words = table.words;

			if let Some(held) = table
				.values
				.get_mut(slot * words..(slot + 1) * words)
			{
				held.copy_from_slice(&table.default);
			}
		}
	}

	/// One slot's record, as the struct it was declared over.
	///
	/// @return nothing for a record nobody declared, or a struct of another
	/// size than the declared one
	pub(crate) fn view<T: Pod>(&self, record: &Record<T>, slot: usize) -> Option<&T> {
		let table = self.table_for(record)?;

		bytemuck::try_from_bytes(bytemuck::cast_slice(table.slot(slot)?)).ok()
	}

	/// One slot's record, to change.
	pub(crate) fn view_mut<T: Pod>(&mut self, record: &Record<T>, slot: usize) -> Option<&mut T> {
		let index = self.index_for(record)?;
		let words = self.tables.get_mut(index)?.slot_mut(slot)?;

		bytemuck::try_from_bytes_mut(bytemuck::cast_slice_mut(words)).ok()
	}

	/// Every slot's record, in slot order, as the struct it was declared over.
	pub(crate) fn column<T: Pod>(&self, record: &Record<T>) -> Option<&[T]> {
		bytemuck::try_cast_slice(&self.table_for(record)?.values).ok()
	}

	/// One field of one slot's record, whatever it holds.
	///
	/// @param table - the record's place in [`tables`](Self::tables)
	/// @param column - the field's place in its columns
	/// @param slot - the array index, not a handle
	pub(crate) fn field(&self, table: usize, column: usize, slot: usize) -> Option<Value> {
		let held = self.tables.get(table)?;
		let column = held.columns.get(column)?;

		value_of(&column.kind, held.slot(slot)?.get(column.span())?)
	}

	/// Writes one field of one slot's record.
	///
	/// @return whether it was written: refused, with nothing written, when the
	/// value is not of the field's kind, a whole number past what the field
	/// holds, or a word past its list
	pub(crate) fn set_field(
		&mut self,
		table: usize,
		column: usize,
		slot: usize,
		value: &Value,
	) -> bool {
		let Some(held) = self.tables.get_mut(table) else {
			return false;
		};
		let Some((span, words)) = held
			.columns
			.get(column)
			.and_then(|column| Some((column.span(), words_of(&column.kind, value)?)))
		else {
			return false;
		};
		let Some(record) = held
			.slot_mut(slot)
			.and_then(|record| record.get_mut(span.clone()))
		else {
			return false;
		};

		record.copy_from_slice(words.get(..span.len()).unwrap_or_default());

		true
	}

	/// What one slot holds that is worth writing down, by name.
	///
	/// Every declared record's fields that differ from its default, in the
	/// order the records and their fields were declared, and after them
	/// whatever waits for a record nobody has declared - so a world written
	/// down while its game is not loaded loses nothing.
	pub(crate) fn noted(&self, slot: usize) -> Vec<Noted> {
		let mut noted: Vec<Noted> = self
			.tables
			.iter()
			.flat_map(|table| table.noted(slot))
			.collect();

		if let Some(waiting) = self.waiting.get(slot) {
			noted.extend(waiting.iter().cloned());
		}

		noted
	}

	/// What waits in one slot for a record nobody has declared.
	pub(crate) fn waiting(&self, slot: usize) -> &[Noted] {
		self.waiting.get(slot).map_or(&[], Vec::as_slice)
	}

	/// Puts written values into one slot.
	///
	/// A value for a declared record lands in the field of its name when the
	/// spelling fits and is refused otherwise; a value for a record nobody has
	/// declared waits, replacing whatever waited under the same two names.
	///
	/// @param slot - the array index, not a handle
	/// @param noted - what to put
	/// @return what was refused, for the caller to report once for a whole load
	pub(crate) fn note(&mut self, slot: usize, noted: &[Noted]) -> Vec<Refused> {
		noted
			.iter()
			.filter_map(|one| self.put(slot, one))
			.collect()
	}

	/// One written value into one slot: taken, refused, or left waiting.
	///
	/// @return the refusal, if the record is declared and did not take it
	fn put(&mut self, slot: usize, one: &Noted) -> Option<Refused> {
		let Some(table) = self
			.tables
			.iter_mut()
			.find(|table| table.name == one.record)
		else {
			if let Some(waiting) = self.waiting.get_mut(slot) {
				waiting.retain(|held| !(held.record == one.record && held.field == one.field));
				waiting.push(one.clone());
			}

			return None;
		};

		table
			.take(slot, one)
			.err()
			.map(|why| Refused::of(one, why))
	}

	/// A record declared under a name something already has.
	fn again(&mut self, index: usize, table: Table) -> Result<Declared> {
		let slots = self.waiting.len();
		let Some(held) = self.tables.get_mut(index) else {
			return Ok(Declared::default());
		};

		if held.owner != table.owner && !held.stale {
			return Err!(Module(error!(
				"a record called {} is declared already, by {}",
				table.name,
				owner_word(held.owner)
			)));
		}

		if held.same_shape(&table) {
			held.stale = false;
			held.owner = table.owner;
			held.help = table.help;

			// the words a build says *about* its fields rather than the fields
			// themselves: a help line rewritten or a field newly drawn is not a
			// change any value has to be carried through
			for (column, now) in held.columns.iter_mut().zip(&table.columns) {
				column.help.clone_from(&now.help);
				column.draw = now.draw;
			}

			return Ok(Declared::default());
		}

		if !held.stale {
			return Err!(Module(error!(
				"a record called {} is declared already with other fields; a record's fields \
				 change across a reload and not within one build",
				table.name
			)));
		}

		// carried over by name onto the new defaults, which is exactly what
		// writing the world down and reading it back would do: a value the old
		// build held at its default was never a value anybody set, and takes the
		// new default like everything else
		let old = core::mem::replace(held, table);
		let mut declared = Declared::default();

		for slot in 0..slots {
			let Some(new) = self.tables.get_mut(index) else {
				break;
			};

			for one in old.noted(slot) {
				match new.take(slot, &one) {
					| Ok(()) => declared.kept += 1,
					| Err(why) => declared.refused.push(Refused::of(&one, why)),
				}
			}
		}

		info!(
			record = old.name,
			kept = declared.kept,
			refused = declared.refused.len(),
			"a record changed its fields; what it held is carried over by name"
		);
		report(&declared.refused, "reloading a record");

		Ok(declared)
	}

	/// Takes what waits for one record, now that it is declared.
	fn claim(&mut self, index: usize, declared: &mut Declared) {
		let Some(table) = self.tables.get_mut(index) else {
			return;
		};

		for (slot, waiting) in self.waiting.iter_mut().enumerate() {
			let (mine, theirs): (Vec<Noted>, Vec<Noted>) = core::mem::take(waiting)
				.into_iter()
				.partition(|one| one.record == table.name);

			*waiting = theirs;

			for one in mine {
				match table.take(slot, &one) {
					| Ok(()) => declared.kept += 1,
					| Err(why) => declared.refused.push(Refused::of(&one, why)),
				}
			}
		}
	}

	/// The declared record a typed caller means, if its struct is the size the
	/// declaration was.
	fn table_for<T>(&self, record: &Record<T>) -> Option<&Table> {
		self.tables.get(self.index_for(record)?)
	}

	/// Where the declared record a typed caller means is.
	fn index_for<T>(&self, record: &Record<T>) -> Option<usize> {
		self.tables
			.iter()
			.position(|table| table.name == record.name && table.words * WORD == size_of::<T>())
	}
}

/// Says once what a load or a declaration could not take: one line a field,
/// with how many values it was.
///
/// @param refused - what was refused, in any order and with repeats
/// @param during - what was being done, for the message
pub fn report(refused: &[Refused], during: &str) {
	let mut seen: Vec<(&Refused, usize)> = Vec::new();

	for one in refused {
		match seen.iter_mut().find(|(held, _)| *held == one) {
			| Some((_, count)) => *count += 1,
			| None => seen.push((one, 1)),
		}
	}

	for (one, count) in seen {
		let why = match one.why {
			| Why::Missing => "the record has no field of that name",
			| Why::Unfit => "the field does not hold a value like it",
		};

		warn!(record = one.record, field = one.field, count, "{during} dropped a value: {why}");
	}
}

/// What to call an owner in a message.
const fn owner_word(owner: Owner) -> &'static str {
	match owner {
		| Owner::Engine => "the engine",
		| Owner::Module => "the game",
		| Owner::Script => "a program",
	}
}

/// Whether a record or a field may be called this: what a file's key and a
/// console line can both hold without quoting.
///
/// @param name - the name in question
#[must_use]
pub fn is_name(name: &str) -> bool {
	!name.is_empty()
		&& name.len() <= MAX_NAME
		&& name
			.bytes()
			.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

/// A declaration, checked and copied into what the host keeps.
fn table_of<T: Pod>(record: &Record<T>, owner: Owner) -> Result<Table> {
	let name = record.name;
	let size = size_of::<T>();

	if !is_name(name) {
		return Err!(Module(error!(
			"a record is called {name:?}; a record's name is lowercase letters, digits and \
			 underscores"
		)));
	}

	if size == 0
		|| !size.is_multiple_of(WORD)
		|| size > MAX_WORDS * WORD
		|| align_of::<T>() > WORD
	{
		return Err!(Module(error!(
			"the record {name} is {size} bytes; a record is one to {MAX_WORDS} four-byte words"
		)));
	}

	if record.rows.is_empty() || record.rows.len() > MAX_ROWS {
		return Err!(Module(error!(
			"the record {name} has {} fields; a record has one to {MAX_ROWS}",
			record.rows.len()
		)));
	}

	let mut taken = vec![false; size / WORD];
	let mut columns: Vec<Column> = Vec::with_capacity(record.rows.len());

	for row in record.rows {
		let column = column_of(name, row, &mut taken, &columns)?;

		columns.push(column);
	}

	let default: Vec<u32> = bytemuck::bytes_of(&record.default)
		.chunks_exact(WORD)
		.map(|chunk| u32::from_ne_bytes(<[u8; WORD]>::try_from(chunk).unwrap_or_default()))
		.collect();

	for column in &columns {
		let words = column.kind.words();
		let index = default.get(column.at).copied().unwrap_or(0);

		if !words.is_empty() && !usize::try_from(index).is_ok_and(|index| index < words.len()) {
			return Err!(Module(error!(
				"the record {name} starts {} at word {index}, which its list does not have",
				column.name
			)));
		}
	}

	Ok(Table {
		name: name.to_owned(),
		help: record.help.to_owned(),
		columns,
		words: size / WORD,
		default,
		owner,
		stale: false,
		values: Vec::new(),
	})
}

/// One row, checked against the record and the rows before it, and copied.
///
/// @param record - the record's name, for a message
/// @param row - the row
/// @param taken - which of the record's words a row before this one holds
/// @param before - the rows before this one
fn column_of(record: &str, row: &Row, taken: &mut [bool], before: &[Column]) -> Result<Column> {
	let field = row.name;

	if !is_name(field) || before.iter().any(|column| column.name == field) {
		return Err!(Module(error!(
			"the record {record} has a field called {field:?}, which is not a name or is there \
			 twice"
		)));
	}

	let Some(kind) = ColumnKind::of(row.kind) else {
		return Err!(Module(error!(
			"{record}.{field} holds {}; a record holds flags, numbers and words",
			row.kind.name()
		)));
	};

	let words = kind.words();
	let repeated = words
		.iter()
		.enumerate()
		.any(|(index, word)| word.is_empty() || words[..index].contains(word));

	if matches!(kind, ColumnKind::Word(_)) && (words.is_empty() || repeated) {
		return Err!(Module(error!(
			"{record}.{field} is one of a list of words that is empty, or holds one twice"
		)));
	}

	let at = row.offset / WORD;
	let held = taken
		.get_mut(at..at + kind.width())
		.filter(|held| row.offset.is_multiple_of(WORD) && !held.iter().any(|word| *word));

	let Some(held) = held else {
		return Err!(Module(error!(
			"{record}.{field} is at byte {}, which is not a whole word of the record or is \
			 another field's",
			row.offset
		)));
	};

	if !row.draw.fits(row.kind) {
		return Err!(Module(error!(
			"{record}.{field} is drawn as {}, which is not what {} holds",
			row.draw.word(),
			row.kind.name()
		)));
	}

	held.fill(true);

	Ok(Column {
		name: field.to_owned(),
		help: row.help.to_owned(),
		kind,
		at,
		draw: row.draw,
	})
}

/// A field's words, spelled.
///
/// @return nothing for a word past the end of its list, which has no spelling
fn spell(kind: &ColumnKind, words: &[u32]) -> Option<Spelled> {
	let first = *words.first()?;

	Some(match kind {
		| ColumnKind::Bool => Spelled::Truth(first != 0),
		| ColumnKind::Int => Spelled::Number(f64::from(first.cast_signed())),
		| ColumnKind::Float => Spelled::Number(f64::from(f32::from_bits(first))),
		| ColumnKind::Vec2 | ColumnKind::Vec3 | ColumnKind::Color | ColumnKind::Quat =>
			Spelled::Numbers(
				words
					.iter()
					.take(kind.width())
					.map(|word| f32::from_bits(*word))
					.collect(),
			),
		| ColumnKind::Word(list) =>
			Spelled::Word(list.get(usize::try_from(first).ok()?)?.clone()),
	})
}

/// A spelled value as a field's words, if the field holds a value like it.
///
/// A whole number has to be one, and one the field's `i32` holds; a number has
/// to be finite and inside what an `f32` holds; two to four numbers have to be
/// as many as the field takes and finite, and a rotation a unit one; a word has
/// to be one of the field's words.
fn fit(kind: &ColumnKind, value: &Spelled) -> Option<[u32; 4]> {
	let mut words = [0; 4];

	match (kind, value) {
		| (ColumnKind::Bool, Spelled::Truth(truth)) => words[0] = u32::from(*truth),
		| (ColumnKind::Int, Spelled::Number(number)) =>
			words[0] = whole(*number)?.cast_unsigned(),
		| (ColumnKind::Float, Spelled::Number(number)) => words[0] = narrow(*number)?.to_bits(),
		| (
			ColumnKind::Vec2 | ColumnKind::Vec3 | ColumnKind::Color | ColumnKind::Quat,
			Spelled::Numbers(numbers),
		) => {
			if numbers.len() != kind.width() || !numbers.iter().all(|it| it.is_finite()) {
				return None;
			}

			if *kind == ColumnKind::Quat && !Quat::from_slice(numbers).is_normalized() {
				return None;
			}

			put(&mut words, numbers);
		},
		| (ColumnKind::Word(list), Spelled::Word(word)) =>
			words[0] = u32::try_from(list.iter().position(|it| it == word)?).ok()?,
		| _ => return None,
	}

	Some(words)
}

/// A field's words as a value an inspector edits.
fn value_of(kind: &ColumnKind, words: &[u32]) -> Option<Value> {
	let number = |at: usize| words.get(at).map(|word| f32::from_bits(*word));

	Some(match kind {
		| ColumnKind::Bool => Value::Bool(*words.first()? != 0),
		| ColumnKind::Int => Value::Int(i64::from(words.first()?.cast_signed())),
		| ColumnKind::Float => Value::Float(number(0)?),
		| ColumnKind::Vec2 => Value::Vec2(Vec2::new(number(0)?, number(1)?)),
		| ColumnKind::Vec3 => Value::Vec3(Vec3::new(number(0)?, number(1)?, number(2)?)),
		| ColumnKind::Color => Value::Color(Vec3::new(number(0)?, number(1)?, number(2)?)),
		| ColumnKind::Quat =>
			Value::Quat(Quat::from_xyzw(number(0)?, number(1)?, number(2)?, number(3)?)),
		| ColumnKind::Word(_) => Value::Word(*words.first()?),
	})
}

/// A value an inspector edited, as a field's words.
fn words_of(kind: &ColumnKind, value: &Value) -> Option<[u32; 4]> {
	let mut words = [0; 4];

	match (kind, value) {
		| (ColumnKind::Bool, Value::Bool(truth)) => words[0] = u32::from(*truth),
		| (ColumnKind::Int, Value::Int(whole)) =>
			words[0] = i32::try_from(*whole).ok()?.cast_unsigned(),
		| (ColumnKind::Float, Value::Float(number)) => words[0] = number.to_bits(),
		| (ColumnKind::Vec2, Value::Vec2(pair)) => put(&mut words, &pair.to_array()),
		| (ColumnKind::Vec3, Value::Vec3(triple)) | (ColumnKind::Color, Value::Color(triple)) =>
			put(&mut words, &triple.to_array()),
		| (ColumnKind::Quat, Value::Quat(turn)) => put(&mut words, &turn.to_array()),
		| (ColumnKind::Word(list), Value::Word(index))
			if usize::try_from(*index).is_ok_and(|index| index < list.len()) =>
			words[0] = *index,
		| _ => return None,
	}

	Some(words)
}

/// Numbers into words, bit for bit.
fn put(words: &mut [u32; 4], numbers: &[f32]) {
	for (word, number) in words.iter_mut().zip(numbers) {
		*word = number.to_bits();
	}
}

/// A number as a whole one an `i32` holds, or nothing.
fn whole(number: f64) -> Option<i32> {
	let inside = (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&number);

	(inside && number.fract() == 0.0).then(|| truncated(number))
}

/// A double already checked to be a whole number an `i32` holds.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "the only conversion from a float to a whole number in the language, and the \
	          caller has ruled out every case it would round or saturate"
)]
const fn truncated(number: f64) -> i32 { number as i32 }

/// A number as an `f32`, or nothing for one an `f32` cannot hold.
fn narrow(number: f64) -> Option<f32> {
	(number.is_finite() && number.abs() <= f64::from(f32::MAX)).then(|| narrowed(number))
}

/// A double already checked to be finite and inside what an `f32` holds.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	reason = "the only narrowing from a double in the language, and every number in a record is \
	          stored at this width"
)]
const fn narrowed(number: f64) -> f32 { number as f32 }

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{bytemuck::Zeroable, random::Random};

	/// A record with a field of every kind a record holds, none of them at a
	/// word the one before it ends on, so a wrong offset lands in another
	/// field.
	#[repr(C)]
	#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
	struct Door {
		open: u32,
		speed: f32,
		hinge: [f32; 3],
		style: u32,
		turns: i32,
		tint: [f32; 3],
		lean: [f32; 4],
		mark: [f32; 2],
	}

	/// The words a door's `style` may be.
	const STYLES: &[&str] = &["swing", "slide", "fold"];

	/// The same door as a build that draws two of its fields spells it.
	const DRAWN_DOOR: Record<Door> = Record {
		name: "door",
		help: "a thing that swings",
		rows: &[
			crate::row!(Bool, Door, open, "whether it stands open"),
			crate::row!(Float, Door, speed, "how far it swings in a second").drawn(Draw::Radius),
			crate::row!(Vec3, Door, hinge, "what it swings about").drawn(Draw::Point),
			crate::row!(Word(STYLES), Door, style, "how it opens"),
			crate::row!(Int, Door, turns, "how many times it has"),
			crate::row!(Color, Door, tint, "its paint"),
			crate::row!(Quat, Door, lean, "how it hangs"),
			crate::row!(Vec2, Door, mark, "where its handle is"),
		],
		default: DOOR.default,
	};

	/// A door drawn in a way its field cannot be: a flag is not a radius.
	const WRONGLY_DRAWN: Record<Door> = Record {
		name: "door",
		help: "a thing that swings",
		rows: &[
			crate::row!(Bool, Door, open, "whether it stands open").drawn(Draw::Radius),
			crate::row!(Float, Door, speed, "how far it swings in a second"),
			crate::row!(Vec3, Door, hinge, "what it swings about"),
			crate::row!(Word(STYLES), Door, style, "how it opens"),
			crate::row!(Int, Door, turns, "how many times it has"),
			crate::row!(Color, Door, tint, "its paint"),
			crate::row!(Quat, Door, lean, "how it hangs"),
			crate::row!(Vec2, Door, mark, "where its handle is"),
		],
		default: DOOR.default,
	};

	/// A door as the game spells it.
	const DOOR: Record<Door> = Record {
		name: "door",
		help: "a thing that swings",
		rows: &[
			crate::row!(Bool, Door, open, "whether it stands open"),
			crate::row!(Float, Door, speed, "how far it swings in a second"),
			crate::row!(Vec3, Door, hinge, "what it swings about"),
			crate::row!(Word(STYLES), Door, style, "how it opens"),
			crate::row!(Int, Door, turns, "how many times it has"),
			crate::row!(Color, Door, tint, "its paint"),
			crate::row!(Quat, Door, lean, "how it hangs"),
			crate::row!(Vec2, Door, mark, "where its handle is"),
		],
		default: Door {
			open: 0,
			speed: 0.25,
			hinge: [0.0, 1.0, 0.0],
			style: 0,
			turns: 0,
			tint: [1.0, 1.0, 1.0],
			lean: [0.0, 0.0, 0.0, 1.0],
			mark: [0.5, 0.5],
		},
	};

	/// The same record as a later build of the game spells it: `speed` is
	/// `pace` now, `turns` became a number, `open` moved, the words were
	/// reordered, and a field is new.
	#[repr(C)]
	#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
	struct Later {
		creak: f32,
		open: u32,
		pace: f32,
		hinge: [f32; 3],
		style: u32,
		turns: f32,
		tint: [f32; 3],
	}

	/// A later build's door.
	const LATER: Record<Later> = Record {
		name: "door",
		help: "a thing that swings, in a later build",
		rows: &[
			crate::row!(Float, Later, creak, "how loud it is"),
			crate::row!(Bool, Later, open, "whether it stands open"),
			crate::row!(Float, Later, pace, "how far it swings in a second"),
			crate::row!(Vec3, Later, hinge, "what it swings about"),
			crate::row!(Word(&["slide", "swing"]), Later, style, "how it opens"),
			crate::row!(Float, Later, turns, "how far it has turned"),
			crate::row!(Color, Later, tint, "its paint"),
		],
		default: Later {
			creak: 0.0,
			open: 0,
			pace: 1.0,
			hinge: [0.0, 1.0, 0.0],
			style: 1,
			turns: 0.0,
			tint: [0.5, 0.5, 0.5],
		},
	};

	/// The door with `open` and `style` in each other's places and nothing else
	/// changed, not even a default word: the one change a comparison of names,
	/// kinds and defaults cannot see.
	#[repr(C)]
	#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
	struct Moved {
		style: u32,
		speed: f32,
		hinge: [f32; 3],
		open: u32,
		turns: i32,
		tint: [f32; 3],
		lean: [f32; 4],
		mark: [f32; 2],
	}

	/// The moved door, its rows in the door's order.
	const MOVED: Record<Moved> = Record {
		name: "door",
		help: "a thing that swings",
		rows: &[
			crate::row!(Bool, Moved, open, "whether it stands open"),
			crate::row!(Float, Moved, speed, "how far it swings in a second"),
			crate::row!(Vec3, Moved, hinge, "what it swings about"),
			crate::row!(Word(STYLES), Moved, style, "how it opens"),
			crate::row!(Int, Moved, turns, "how many times it has"),
			crate::row!(Color, Moved, tint, "its paint"),
			crate::row!(Quat, Moved, lean, "how it hangs"),
			crate::row!(Vec2, Moved, mark, "where its handle is"),
		],
		default: Moved {
			style: 0,
			speed: 0.25,
			hinge: [0.0, 1.0, 0.0],
			open: 0,
			turns: 0,
			tint: [1.0, 1.0, 1.0],
			lean: [0.0, 0.0, 0.0, 1.0],
			mark: [0.5, 0.5],
		},
	};

	/// A second record, so a declaration and a claim can be seen to leave
	/// another record's values alone.
	#[repr(C)]
	#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
	struct Lamp {
		glow: f32,
	}

	/// A lamp as the game spells it.
	const LAMP: Record<Lamp> = Record {
		name: "lamp",
		help: "a thing that glows",
		rows: &[crate::row!(Float, Lamp, glow, "how brightly")],
		default: Lamp { glow: 1.0 },
	};

	/// Two words, for records that are wrong in one way each.
	#[repr(C)]
	#[derive(Clone, Copy, Pod, Zeroable)]
	struct Pair {
		one: u32,
		two: u32,
	}

	/// Nine bytes, which are not words.
	#[repr(C)]
	#[derive(Clone, Copy, Pod, Zeroable)]
	struct Odd {
		bytes: [u8; 8],
		tail: u8,
	}

	/// More words than a record may be.
	#[repr(C)]
	#[derive(Clone, Copy, Pod, Zeroable)]
	struct Wide {
		words: [u32; 64],
	}

	/// A field of a pair, by hand, so that it can be wrong.
	const fn loose(name: &'static str, kind: Kind, offset: usize) -> Row {
		Row {
			name,
			help: "a field",
			kind,
			offset,
			draw: Draw::None,
		}
	}

	/// A pair record of these fields.
	const fn pair(name: &'static str, rows: &'static [Row]) -> Record<Pair> {
		Record {
			name,
			help: "a record",
			rows,
			default: Pair { one: 0, two: 0 },
		}
	}

	/// A table of this many slots with the door declared.
	fn doors(slots: usize) -> Records {
		let mut records = Records::new();

		for _ in 0..slots {
			records.push();
		}

		records
			.declare(&DOOR)
			.expect("the door is a record a world holds");

		records
	}

	/// One written value.
	fn noted(record: &str, field: &str, value: Spelled) -> Noted {
		Noted {
			record: record.to_owned(),
			field: field.to_owned(),
			value,
		}
	}

	/// A door's speed, as bits, so a number is compared exactly.
	fn speed_of(records: &Records, slot: usize) -> Option<u32> {
		records
			.view(&DOOR, slot)
			.map(|door| door.speed.to_bits())
	}

	#[test]
	fn a_declared_record_is_carried_by_every_slot_at_its_default() {
		let mut records = Records::new();
		records.push();
		records.push();

		assert!(records.view(&DOOR, 0).is_none(), "nothing declared, nothing carried");

		let declared = records
			.declare(&DOOR)
			.expect("the door is a record a world holds");

		assert_eq!(declared, Declared::default(), "nothing waited for it");
		assert_eq!(records.view(&DOOR, 0), Some(&DOOR.default), "the slot there before it");

		records.push();

		assert_eq!(records.view(&DOOR, 2), Some(&DOOR.default), "and a slot added after it");
		assert!(records.view(&DOOR, 3).is_none(), "and no slot the table does not have");
		assert_eq!(records.column(&DOOR).map(<[Door]>::len), Some(3), "the column is every slot");
	}

	#[test]
	fn a_record_is_read_and_written_as_the_struct_it_was_declared_over() {
		let mut records = doors(3);

		if let Some(door) = records.view_mut(&DOOR, 1) {
			door.speed = 3.0;
			door.style = 2;
			door.lean = [0.0, 1.0, 0.0, 0.0];
		}

		let door = records
			.view(&DOOR, 1)
			.copied()
			.expect("slot one is a door");

		assert_eq!(door.speed.to_bits(), 3.0_f32.to_bits(), "the number written");
		assert_eq!((door.style, door.open), (2, 0), "the word, and its neighbor untouched");
		assert_eq!(records.view(&DOOR, 0), Some(&DOOR.default), "and the slot before");
		assert_eq!(records.view(&DOOR, 2), Some(&DOOR.default), "and the slot after");
		assert_eq!(
			records
				.column(&DOOR)
				.and_then(|column| column.get(1))
				.copied(),
			Some(door),
			"the column reads the same bytes"
		);

		// a struct of another size is not the record, whatever it is called
		let lamp_named_door = Record { name: "door", ..LAMP };

		assert!(records.view(&lamp_named_door, 1).is_none(), "four bytes are not a door");
		assert!(records.column(&lamp_named_door).is_none(), "nor is a column of them");
	}

	#[test]
	fn a_field_is_read_and_written_by_its_place_whatever_its_kind() {
		let mut records = doors(2);
		let values = [
			Value::Bool(true),
			Value::Float(2.5),
			Value::Vec3(Vec3::new(1.0, 2.0, 3.0)),
			Value::Word(2),
			Value::Int(-7),
			Value::Color(Vec3::new(0.2, 0.4, 0.6)),
			Value::Quat(Quat::from_rotation_y(0.5)),
			Value::Vec2(Vec2::new(0.25, 0.75)),
		];

		for (column, value) in values.iter().enumerate() {
			assert_ne!(
				records.field(0, column, 1).as_ref(),
				Some(value),
				"column {column} does not start as the value written"
			);
			assert!(records.set_field(0, column, 1, value), "column {column} takes its own kind");
			assert_eq!(records.field(0, column, 1).as_ref(), Some(value), "and reads it back");
			assert!(
				!records.set_field(0, column, 1, &Value::Text("wrong".to_owned())),
				"and refuses another"
			);
		}

		let door = records
			.view(&DOOR, 1)
			.copied()
			.expect("slot one is a door");

		assert_eq!(door.turns, -7, "the erased write is the typed field");
		assert_eq!(records.view(&DOOR, 0), Some(&DOOR.default), "and no other slot moved");

		assert!(!records.set_field(0, 3, 1, &Value::Word(3)), "a word past its list");
		assert!(!records.set_field(0, 4, 1, &Value::Int(1 << 40)), "a whole number past an i32");
		assert!(
			!records.set_field(0, 0, 9, &Value::Bool(true)),
			"a slot the table does not have"
		);
		assert!(!records.set_field(0, 8, 1, &Value::Bool(true)), "a column it does not have");
		assert!(!records.set_field(1, 0, 1, &Value::Bool(true)), "a record nobody declared");
		assert_eq!(
			records
				.view(&DOOR, 1)
				.map(|door| (door.style, door.turns)),
			Some((2, -7)),
			"and nothing refused was written"
		);
	}

	#[test]
	fn a_record_that_cannot_be_held_is_refused_naming_what() {
		const RENAMED: &[Row] = &[Row {
			name: "a b",
			..crate::row!(Bool, Pair, one, "x")
		}];
		const TWICE: &[Row] =
			&[crate::row!(Bool, Pair, one, "x"), crate::row!(Bool, Pair, one, "y")];
		const TEXT: &[Row] = &[loose("label", Kind::Text, 0)];
		const HANDLE: &[Row] = &[loose("target", Kind::Entity, 0)];
		const PAST: &[Row] = &[loose("hinge", Kind::Vec3, 0)];
		const BETWEEN: &[Row] = &[loose("one", Kind::Bool, 2)];
		const OVER: &[Row] = &[crate::row!(Bool, Pair, one, "x"), loose("two", Kind::Int, 0)];
		const NO_WORDS: &[Row] = &[loose("style", Kind::Word(&[]), 0)];
		const SAME_WORDS: &[Row] = &[loose("style", Kind::Word(&["a", "a"]), 0)];
		const TWO_WORDS: &[Row] = &[loose("style", Kind::Word(&["a", "b"]), 0)];

		let started_past = Record {
			default: Pair { one: 5, two: 0 },
			..pair("door", TWO_WORDS)
		};
		let cases: [(Record<Pair>, &str); 12] = [
			(pair("Door", TWICE), "a record's name is lowercase"),
			(pair("door", &[]), "one to 32"),
			(pair("door", RENAMED), "not a name"),
			(pair("door", TWICE), "there twice"),
			(pair("door", TEXT), "holds text"),
			(pair("door", HANDLE), "holds an entity"),
			(pair("door", PAST), "not a whole word"),
			(pair("door", BETWEEN), "not a whole word"),
			(pair("door", OVER), "another field's"),
			(pair("door", NO_WORDS), "is empty"),
			(pair("door", SAME_WORDS), "holds one twice"),
			(started_past, "its list does not have"),
		];

		for (record, said) in cases {
			let mut records = Records::new();
			let refused = records
				.declare(&record)
				.expect_err("the record cannot be held");

			assert!(refused.to_string().contains(said), "expected {said:?}, got {refused}");
			assert!(records.tables().is_empty(), "and nothing was declared: {refused}");
		}

		let odd = Record {
			name: "odd",
			help: "a record that is not words",
			rows: &[],
			default: Odd { bytes: [0; 8], tail: 0 },
		};
		let refused = Records::new()
			.declare(&odd)
			.expect_err("nine bytes are not words");

		assert!(refused.to_string().contains("four-byte words"), "got {refused}");
	}

	#[test]
	fn the_ceilings_are_where_they_say() {
		let mut records = Records::new();
		let names: Vec<&'static str> = (0..=MAX_RECORDS)
			.map(|index| &*format!("record_{index}").leak())
			.collect();

		for name in names.iter().copied().take(MAX_RECORDS) {
			records
				.declare(&Record { name, ..LAMP })
				.expect("a world holds this many");
		}

		let refused = records
			.declare(&Record { name: names[MAX_RECORDS], ..LAMP })
			.expect_err("one more is refused");

		assert!(refused.to_string().contains("one more than"), "got {refused}");
		assert_eq!(records.tables().len(), MAX_RECORDS, "and it was not declared");

		let wide = Record {
			name: "wide",
			help: "wider than a record may be",
			rows: &[Row {
				name: "words",
				help: "x",
				kind: Kind::Bool,
				offset: 0,
				draw: Draw::None,
			}],
			default: Wide { words: [0; 64] },
		};

		assert!(Records::new().declare(&wide).is_err(), "a record past the words it may be");
	}

	#[test]
	fn declaring_the_same_record_again_changes_nothing_and_other_fields_are_refused_within_a_build()
	 {
		let mut records = doors(2);

		if let Some(door) = records.view_mut(&DOOR, 1) {
			door.speed = 9.0;
		}

		let again = Record { help: "a door, said again", ..DOOR };

		assert_eq!(records.declare(&again).ok(), Some(Declared::default()), "the same fields");
		assert_eq!(speed_of(&records, 1), Some(9.0_f32.to_bits()), "and the value set is kept");
		assert_eq!(
			records.tables()[0].help(),
			"a door, said again",
			"the help line is the newest"
		);

		let refused = records
			.declare(&LATER)
			.expect_err("other fields, and no reload");

		assert!(refused.to_string().contains("other fields"), "got {refused}");
		assert!(records.view(&DOOR, 1).is_some(), "and the record is still the first door");

		let mut engine = Records::new();
		engine
			.declare(&LAMP)
			.expect("the engine declares a lamp");
		engine.attribute(Owner::Module);

		let refused = engine
			.declare(&LAMP)
			.expect_err("the game may not declare the engine's");

		assert!(refused.to_string().contains("by the engine"), "got {refused}");
	}

	#[test]
	fn only_what_differs_from_the_default_is_written_down_in_the_order_it_was_declared() {
		let mut records = doors(2);
		records
			.declare(&LAMP)
			.expect("a lamp is a record a world holds");

		assert!(records.noted(1).is_empty(), "a slot at every default writes nothing");

		if let Some(lamp) = records.view_mut(&LAMP, 1) {
			lamp.glow = 0.5;
		}

		if let Some(door) = records.view_mut(&DOOR, 1) {
			door.mark = [0.5, 0.25];
			door.open = 7;
			door.style = 9;
		}

		assert_eq!(
			records.noted(1),
			vec![
				noted("door", "open", Spelled::Truth(true)),
				noted("door", "mark", Spelled::Numbers(vec![0.5, 0.25])),
				noted("lamp", "glow", Spelled::Number(0.5)),
			],
			"the door's fields in its order, then the lamp's; a word past its list has no \
			 spelling, and a flag of seven is a flag"
		);
		assert!(records.noted(0).is_empty(), "and the other slot is still at its defaults");
	}

	#[test]
	fn a_written_value_lands_where_a_field_of_its_name_fits_and_is_refused_otherwise() {
		let mut records = doors(1);
		let written = [
			noted("door", "open", Spelled::Truth(true)),
			noted("door", "speed", Spelled::Number(4.0)),
			noted("door", "hinge", Spelled::Numbers(vec![1.0, 0.0, 0.0])),
			noted("door", "style", Spelled::Word("fold".to_owned())),
			noted("door", "turns", Spelled::Number(-3.0)),
			noted("door", "lean", Spelled::Numbers(vec![0.0, 0.0, 1.0, 0.0])),
			noted("door", "pace", Spelled::Number(1.0)),
			noted("door", "open", Spelled::Number(1.0)),
			noted("door", "turns", Spelled::Number(0.5)),
			noted("door", "turns", Spelled::Number(3e10)),
			noted("door", "speed", Spelled::Number(1e39)),
			noted("door", "hinge", Spelled::Numbers(vec![1.0, 0.0])),
			noted("door", "hinge", Spelled::Numbers(vec![f32::NAN, 0.0, 0.0])),
			noted("door", "lean", Spelled::Numbers(vec![0.0, 0.0, 2.0, 0.0])),
			noted("door", "style", Spelled::Word("roll".to_owned())),
			noted("door", "style", Spelled::Truth(false)),
		];

		let refused = records.note(0, &written);
		let door = records
			.view(&DOOR, 0)
			.copied()
			.expect("slot nought is a door");

		assert_eq!(
			(door.open, door.speed.to_bits(), door.style, door.turns),
			(1, 4.0_f32.to_bits(), 2, -3),
			"every value that fits landed"
		);
		assert_eq!(
			(door.hinge.map(f32::to_bits), door.lean.map(f32::to_bits)),
			(
				[1.0_f32, 0.0, 0.0].map(f32::to_bits),
				[0.0_f32, 0.0, 1.0, 0.0].map(f32::to_bits)
			),
			"and nothing that did not moved it"
		);

		let why: Vec<(&str, Why)> = refused
			.iter()
			.map(|one| (one.field.as_str(), one.why))
			.collect();

		assert_eq!(
			why,
			vec![
				("pace", Why::Missing),
				("open", Why::Unfit),
				("turns", Why::Unfit),
				("turns", Why::Unfit),
				("speed", Why::Unfit),
				("hinge", Why::Unfit),
				("hinge", Why::Unfit),
				("lean", Why::Unfit),
				("style", Why::Unfit),
				("style", Why::Unfit),
			],
			"and each one that did not is named with why"
		);
		assert!(records.waiting(0).is_empty(), "nothing waits: the record is declared");
	}

	#[test]
	fn a_value_for_a_record_nobody_declared_waits_and_is_claimed_when_somebody_does() {
		let mut records = Records::new();
		records.push();
		records.push();
		records
			.declare(&LAMP)
			.expect("a lamp is a record a world holds");

		let refused = records.note(1, &[
			noted("door", "speed", Spelled::Number(2.0)),
			noted("lamp", "glow", Spelled::Number(3.0)),
			noted("door", "speed", Spelled::Number(5.0)),
			noted("door", "roll", Spelled::Truth(true)),
			noted("door", "open", Spelled::Number(1.0)),
		]);

		assert!(refused.is_empty(), "nothing is refused for a record nobody declared");
		assert_eq!(
			records.waiting(1),
			&[
				noted("door", "speed", Spelled::Number(5.0)),
				noted("door", "roll", Spelled::Truth(true)),
				noted("door", "open", Spelled::Number(1.0)),
			],
			"the door's wait, the later of two for one field; the lamp's landed"
		);
		assert_eq!(
			records.noted(1),
			vec![
				noted("lamp", "glow", Spelled::Number(3.0)),
				noted("door", "speed", Spelled::Number(5.0)),
				noted("door", "roll", Spelled::Truth(true)),
				noted("door", "open", Spelled::Number(1.0)),
			],
			"and a world written down meanwhile writes what waits"
		);

		let declared = records
			.declare(&DOOR)
			.expect("the door is a record a world holds");

		assert_eq!(declared.kept, 1, "the speed was taken");
		assert_eq!(
			declared.refused,
			vec![
				Refused {
					record: "door".to_owned(),
					field: "roll".to_owned(),
					why: Why::Missing
				},
				Refused {
					record: "door".to_owned(),
					field: "open".to_owned(),
					why: Why::Unfit
				},
			],
			"and the two that fit nothing are said"
		);
		assert_eq!(
			speed_of(&records, 1),
			Some(5.0_f32.to_bits()),
			"the waiting value is the door's"
		);
		assert!(records.waiting(1).is_empty(), "and nothing is left waiting");
		assert_eq!(
			records
				.view(&LAMP, 1)
				.map(|lamp| lamp.glow.to_bits()),
			Some(3.0_f32.to_bits()),
			"and the lamp did not move"
		);
	}

	#[test]
	fn a_reloaded_record_with_other_fields_keeps_what_it_held_by_name() {
		let mut records = Records::new();
		for _ in 0..3 {
			records.push();
		}
		records.attribute(Owner::Module);
		records
			.declare(&DOOR)
			.expect("the game declares its door");

		if let Some(door) = records.view_mut(&DOOR, 1) {
			door.open = 1;
			door.speed = 7.0;
			door.style = 1;
			door.turns = 4;
			door.tint = [0.1, 0.2, 0.3];
		}

		if let Some(door) = records.view_mut(&DOOR, 2) {
			door.hinge = [1.0, 0.0, 0.0];
		}

		records.forget_module();

		let declared = records
			.declare(&LATER)
			.expect("a reload may change a record's fields");
		let first = records
			.view(&LATER, 1)
			.copied()
			.expect("slot one is still a door");
		let second = records
			.view(&LATER, 2)
			.copied()
			.expect("and slot two");

		assert_eq!(first.open, 1, "a flag that moved in the struct followed its name");
		assert_eq!(first.style, 0, "a word found by its spelling, at another place in the list");
		assert_eq!(
			first.tint.map(f32::to_bits),
			[0.1_f32, 0.2, 0.3].map(f32::to_bits),
			"a color"
		);
		assert_eq!(first.pace.to_bits(), 1.0_f32.to_bits(), "a renamed field takes the default");
		assert_eq!(first.turns.to_bits(), 4.0_f32.to_bits(), "a whole number is a number too");
		assert_eq!(
			second.hinge.map(f32::to_bits),
			[1.0_f32, 0.0, 0.0].map(f32::to_bits),
			"slot two"
		);
		assert_eq!(
			second.tint.map(f32::to_bits),
			[0.5_f32, 0.5, 0.5].map(f32::to_bits),
			"and a default that changed is taken where nobody set one"
		);
		assert_eq!(declared.kept, 5, "every value carried over is counted");
		assert_eq!(
			declared
				.refused
				.iter()
				.map(|one| (one.field.as_str(), one.why))
				.collect::<Vec<_>>(),
			vec![("speed", Why::Missing)],
			"and the one that went nowhere is said"
		);

		records.sweep();

		assert_eq!(records.tables().len(), 1, "a record declared again is not swept");
		assert!(records.waiting(1).is_empty(), "and nothing of it waits");
	}

	#[test]
	fn a_field_moved_or_a_default_changed_by_a_reload_is_carried_by_name_too() {
		// the two changes that leave every name and kind where it was: to the
		// values the old build laid down both are other fields, and neither was
		// caught until a mutation pass took each out of the comparison
		let mut records = Records::new();
		records.push();
		records.push();
		records.attribute(Owner::Module);
		records
			.declare(&DOOR)
			.expect("the game declares its door");

		if let Some(door) = records.view_mut(&DOOR, 1) {
			door.open = 1;
			door.style = 2;
			door.speed = 7.0;
		}

		records.forget_module();

		let declared = records
			.declare(&MOVED)
			.expect("a reload may move a field");
		let moved = records
			.view(&MOVED, 1)
			.copied()
			.expect("slot one is still a door");

		assert_eq!((moved.open, moved.style), (1, 2), "each followed its name to its new place");
		assert_eq!(declared.kept, 3, "and was counted");

		records.forget_module();

		let slower = Record {
			default: Moved { speed: 0.5, ..MOVED.default },
			..MOVED
		};
		let declared = records
			.declare(&slower)
			.expect("a reload may change only a default");
		let speeds: Vec<Option<u32>> = (0..2)
			.map(|slot| {
				records
					.view(&slower, slot)
					.map(|door| door.speed.to_bits())
			})
			.collect();

		assert_eq!(
			speeds,
			vec![Some(0.5_f32.to_bits()), Some(7.0_f32.to_bits())],
			"a door that never set its speed takes the new default, and one that did keeps it"
		);
		assert_eq!(declared.kept, 3, "what was set is carried again");

		records.push();

		assert_eq!(
			records
				.view(&slower, 2)
				.map(|door| door.speed.to_bits()),
			Some(0.5_f32.to_bits()),
			"and a slot handed out after starts at the new default"
		);
	}

	#[test]
	fn a_record_the_reloaded_module_did_not_declare_again_waits_by_name() {
		let mut records = Records::new();
		records.push();
		records.push();
		records.declare(&LAMP).expect("the engine's lamp");
		records.attribute(Owner::Module);
		records.declare(&DOOR).expect("the game's door");

		if let Some(door) = records.view_mut(&DOOR, 0) {
			door.speed = 2.0;
		}

		records.forget_module();
		records.sweep();

		assert_eq!(records.tables().len(), 1, "the lamp is the engine's and stays");
		assert!(records.view(&DOOR, 0).is_none(), "the door is gone");
		assert_eq!(
			records.waiting(0),
			&[noted("door", "speed", Spelled::Number(2.0))],
			"and what it held waits by name"
		);
		assert!(records.waiting(1).is_empty(), "a door at its defaults leaves nothing to wait");

		let declared = records
			.declare(&DOOR)
			.expect("the next build brings it back");

		assert_eq!(declared.kept, 1, "and takes it");
		assert_eq!(speed_of(&records, 0), Some(2.0_f32.to_bits()), "as it was");
	}

	#[test]
	fn a_slot_handed_out_again_holds_every_default_and_nothing_waiting() {
		let mut records = doors(2);
		let refused = records.note(1, &[
			noted("door", "speed", Spelled::Number(2.0)),
			noted("gate", "open", Spelled::Truth(true)),
		]);

		assert!(refused.is_empty(), "both were taken, one of them to wait");

		records.clear(1);

		assert_eq!(records.view(&DOOR, 1), Some(&DOOR.default), "the record is at its default");
		assert!(records.waiting(1).is_empty(), "and nothing waits");

		records.note(0, &[noted("gate", "open", Spelled::Truth(true))]);
		records.reset(3);

		assert_eq!(
			records.column(&DOOR).map(<[Door]>::len),
			Some(3),
			"a reset sizes every record"
		);
		assert!(records.waiting(0).is_empty(), "and empties what waited");
		assert!(
			(0..3).all(|slot| records.view(&DOOR, slot) == Some(&DOOR.default)),
			"with every slot at the default"
		);
	}

	#[test]
	fn every_spelling_reads_back_as_the_value_it_was_spelled_from() {
		let mut random = Random::new(0xE1);
		let mut records = doors(1);
		let mut tried = 0;
		let number =
			|random: &mut Random| f32::from_bits(u32::try_from(random.draw() >> 32).unwrap_or(0));

		while tried < 5_000 {
			let mut door = DOOR.default;

			door.open = u32::from(random.below(2) == 1);
			door.speed = number(&mut random);
			door.hinge = [number(&mut random), number(&mut random), number(&mut random)];
			door.style = u32::try_from(random.below(3)).unwrap_or(0);
			door.turns = u32::try_from(random.draw() >> 32)
				.unwrap_or(0)
				.cast_signed();
			door.tint = [number(&mut random), 0.5, -0.0];
			door.mark = [number(&mut random), number(&mut random)];

			let finite = door.speed.is_finite()
				&& door
					.hinge
					.iter()
					.chain(&door.tint)
					.chain(&door.mark)
					.all(|it| it.is_finite());

			if !finite {
				continue;
			}

			tried += 1;

			if let Some(held) = records.view_mut(&DOOR, 0) {
				*held = door;
			}

			let noted = records.noted(0);

			records.clear(0);

			assert!(records.note(0, &noted).is_empty(), "every spelling fits: {noted:?}");

			let back = records
				.view(&DOOR, 0)
				.copied()
				.expect("slot nought is a door");

			assert_eq!(
				bytemuck::bytes_of(&back),
				bytemuck::bytes_of(&door),
				"{door:?} came back"
			);
		}
	}

	#[test]
	fn a_spelling_is_checked_against_a_row_without_a_world() {
		assert!(fits(Kind::Bool, &Spelled::Truth(true)), "a flag is true or false");
		assert!(!fits(Kind::Bool, &Spelled::Number(1.0)), "and not a number");
		assert!(fits(Kind::Int, &Spelled::Number(-2.0)), "a whole number is a number");
		assert!(!fits(Kind::Int, &Spelled::Number(2.5)), "that is whole");
		assert!(
			fits(Kind::Word(STYLES), &Spelled::Word("slide".to_owned())),
			"a word of its list"
		);
		assert!(!fits(Kind::Word(STYLES), &Spelled::Word("roll".to_owned())), "and only those");
		assert!(!fits(Kind::Text, &Spelled::Word("hello".to_owned())), "and text is nothing");
	}

	#[test]
	fn a_record_s_default_is_its_struct_s_bytes() {
		let records = doors(1);
		let table = &records.tables()[0];

		assert_eq!(
			bytemuck::cast_slice::<u32, u8>(&table.default),
			bytemuck::bytes_of(&DOOR.default),
			"the default word for word"
		);
		assert_eq!(table.columns()[2].span(), 2..5, "and the hinge's three words after two");
		assert_eq!(Door::zeroed().open, 0, "and a door of nothing is a closed one");
	}

	#[test]
	fn the_engine_s_own_records_are_ones_a_world_holds() {
		let mut records = Records::new();

		records
			.declare(&super::super::entity::DRAWING)
			.expect("drawing is a record a world holds");
		records
			.declare(&super::super::entity::EDITING)
			.expect("editing is a record a world holds");
		records
			.declare(&super::super::entity::BAKING)
			.expect("baking is a record a world holds");

		assert_eq!(ENGINE.len(), records.tables().len(), "every engine record, and only them");

		for (shape, table) in ENGINE.iter().zip(records.tables()) {
			assert_eq!(shape.name, table.name(), "by name, in the order a world declares them");
		}

		let world = super::super::World::new();
		let declared: Vec<&str> = world
			.entities
			.records()
			.tables()
			.iter()
			.map(Table::name)
			.collect();
		let listed: Vec<&str> = ENGINE.iter().map(|shape| shape.name).collect();

		assert_eq!(declared, listed, "and a world declares exactly the records the list names");
	}

	#[test]
	fn a_field_drawn_in_a_way_it_cannot_be_is_refused_naming_it() {
		let mut records = Records::new();
		records.push();

		let why = records
			.declare(&WRONGLY_DRAWN)
			.expect_err("a flag is not a radius");
		let said = format!("{why}");

		assert!(said.contains("door.open"), "it names the field: {said}");
		assert!(said.contains("radius"), "and what it was asked to be: {said}");
		assert!(records.tables().is_empty(), "and nothing was declared");
	}

	#[test]
	fn a_field_says_how_it_is_drawn_and_nothing_else_does() {
		let mut records = Records::new();
		records.push();
		records
			.declare(&DRAWN_DOOR)
			.expect("a door that draws two of its fields");

		let table = &records.tables()[0];
		let drawn: Vec<Draw> = table.columns().iter().map(Column::draw).collect();

		assert_eq!(drawn[0], Draw::None, "a flag is a checkbox and nothing more");
		assert_eq!(drawn[1], Draw::Radius);
		assert_eq!(drawn[2], Draw::Point);
	}

	#[test]
	fn a_build_that_draws_a_field_it_did_not_draw_before_moves_no_value() {
		let mut records = Records::new();
		records.push();
		records.declare(&DOOR).expect("a door");

		if let Some(door) = records.view_mut(&DOOR, 0) {
			door.speed = 4.0;
		}

		// the same fields at the same places: what changed is a word about how
		// a panel and a picture show one of them
		let again = records
			.declare(&DRAWN_DOOR)
			.expect("the same door, drawn");

		assert_eq!(again.kept, 0, "nothing was carried, because nothing moved");
		assert_eq!(records.tables().len(), 1);
		assert_eq!(records.tables()[0].columns()[1].draw(), Draw::Radius, "and it is drawn now");
		assert!(
			records
				.view(&DOOR, 0)
				.is_some_and(|door| (door.speed - 4.0).abs() < 1.0e-6),
			"with the number somebody typed left alone"
		);
	}
}
