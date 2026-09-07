//! A record's fields, spelled out: what each is called, what kind of value it
//! holds, and how to read and write it without knowing the record.
//!
//! **One table per record, and every consumer walks it.** An inspector draws
//! a widget per row, a scene source reads a key per row and writes one back,
//! and a settings panel will list a table of console variables the same way.
//! Before this the inspector was hand-written per field, the reader kept an
//! allow-list per record and the writer a `put_*` call per field, and one new
//! field on a body was measured at eight files and twenty-seven sites - none
//! of which was the inspector, which is why the inspector never showed it.
//!
//! **A [`Value`] is flat and its own.** The console's `Value` has four kinds
//! because a console variable is a number or a word; a record's field is also
//! a vector, a rotation, a color, one of a few words, or a handle into a
//! table. Growing the console's type would have given `sim.rate` a rotation
//! it can never hold, so this is a second type with the four console kinds
//! inside it and a conversion each way.
//!
//! **A handle is a reference, and a reference is described but not spelled.**
//! A body's `entity` and a joint's two bodies are in the tables, because an
//! inspector wants to say what they point at; what nothing here can do is
//! write one down as text, since a handle is an index into a world and a file
//! names things instead. So [`Kind::is_reference`] is the question a reader
//! or a writer asks first, and the references are handled by hand beside the
//! table rather than through it.
//!
//! **Names are static and copied nowhere**, which is fine for the tables
//! this crate declares: they are in an image that is never unloaded. A table
//! declared by a game module would hold names inside the module's image, and
//! anything that kept a [`Field`] across a reload would then read freed
//! memory. Nothing keeps one; a consumer walks a table and forgets it.

use super::{
	cvar, entity::EntityId, joint::JointId, material::MaterialId, mesh::MeshId, physics::BodyId,
	pose::PoseId, texture::TextureId,
};
use crate::glam::{Quat, Vec2, Vec3};

/// What kind of value a field holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
	/// `true` or `false`.
	Bool,

	/// A whole number.
	Int,

	/// A number.
	Float,

	/// Text.
	Text,

	/// Two numbers: how far a texture repeats, a point on a picture.
	Vec2,

	/// Three numbers: a position, a direction, a size.
	Vec3,

	/// A rotation, as a unit quaternion.
	Quat,

	/// A color, linear RGB with each channel in `0.0 ..= 1.0`.
	///
	/// Three numbers like a [`Vec3`](Self::Vec3), and a kind of its own so
	/// that an inspector can offer a color picker and a file can be read for
	/// one; the value is a `Vec3` all the same.
	Color,

	/// One of a fixed list of words, held as an index into it.
	///
	/// What an enumeration is once it has to be shown, read and written: the
	/// spellings are declared once, here, and the same list is a file's
	/// vocabulary and an inspector's drop-down. **Declaration order is the
	/// index**, so the list is in the order the enumeration declares its
	/// variants.
	Word(&'static [&'static str]),

	/// A handle to an entity.
	Entity,

	/// A handle to a body.
	Body,

	/// A handle to a joint.
	Joint,

	/// A handle to a pose.
	Pose,

	/// A handle to a mesh.
	Mesh,

	/// A handle to a material.
	Material,

	/// A handle to a texture.
	Texture,
}

impl Kind {
	/// Whether this is a handle into one of the world's tables.
	///
	/// The question a reader or a writer asks before anything else: a
	/// reference is described here so that an inspector can show it, and it
	/// has no spelling of its own, so a file names the thing it points at by
	/// hand instead. @ref the module comment.
	#[must_use]
	pub const fn is_reference(self) -> bool {
		matches!(
			self,
			Self::Entity
				| Self::Body | Self::Joint
				| Self::Pose | Self::Mesh
				| Self::Material
				| Self::Texture
		)
	}

	/// The words a [`Word`](Self::Word) may be, or none for any other kind.
	#[must_use]
	pub const fn words(self) -> &'static [&'static str] {
		match self {
			| Self::Word(words) => words,
			| _ => &[],
		}
	}

	/// Which of a [`Word`](Self::Word)'s words this is, if it is one of them.
	///
	/// @param word - the spelling
	/// @return its index, which is what a [`Value::Word`] holds
	#[must_use]
	pub fn word(self, word: &str) -> Option<u32> {
		self.words()
			.iter()
			.position(|it| *it == word)
			.and_then(|index| u32::try_from(index).ok())
	}

	/// What to call a value of this kind in a message.
	#[must_use]
	pub const fn name(self) -> &'static str {
		match self {
			| Self::Bool => "true or false",
			| Self::Int => "a whole number",
			| Self::Float => "a number",
			| Self::Text => "text",
			| Self::Vec2 => "two numbers",
			| Self::Vec3 => "three numbers",
			| Self::Quat => "a rotation, four numbers xyzw",
			| Self::Color => "a color, three numbers",
			| Self::Word(_) => "one of a few words",
			| Self::Entity => "an entity",
			| Self::Body => "a body",
			| Self::Joint => "a joint",
			| Self::Pose => "a pose",
			| Self::Mesh => "a mesh",
			| Self::Material => "a material",
			| Self::Texture => "a texture",
		}
	}
}

/// What a field holds, read out or about to be written in.
///
/// One variant per [`Kind`], and a value is the kind of its variant: a field
/// of one kind handed a value of another refuses it and writes nothing, the
/// way a console variable does.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
	/// `true` or `false`.
	Bool(bool),

	/// A whole number.
	Int(i64),

	/// A number.
	Float(f32),

	/// Text.
	Text(String),

	/// Two numbers.
	Vec2(Vec2),

	/// Three numbers.
	Vec3(Vec3),

	/// A rotation.
	Quat(Quat),

	/// A color, linear RGB.
	Color(Vec3),

	/// Which of a [`Kind::Word`]'s words, by index.
	Word(u32),

	/// A handle to an entity.
	Entity(EntityId),

	/// A handle to a body.
	Body(BodyId),

	/// A handle to a joint.
	Joint(JointId),

	/// A handle to a pose.
	Pose(PoseId),

	/// A handle to a mesh.
	Mesh(MeshId),

	/// A handle to a material.
	Material(MaterialId),

	/// A handle to a texture.
	Texture(TextureId),
}

impl Value {
	/// Whether this is a value of that kind.
	///
	/// A [`Word`](Self::Word) is of every word kind: which list it indexes is
	/// the field's business, and a field checks the index against its own
	/// list when it is written.
	///
	/// @param kind - what to compare against
	#[must_use]
	pub const fn is(&self, kind: Kind) -> bool {
		matches!(
			(self, kind),
			(Self::Bool(_), Kind::Bool)
				| (Self::Int(_), Kind::Int)
				| (Self::Float(_), Kind::Float)
				| (Self::Text(_), Kind::Text)
				| (Self::Vec2(_), Kind::Vec2)
				| (Self::Vec3(_), Kind::Vec3)
				| (Self::Quat(_), Kind::Quat)
				| (Self::Color(_), Kind::Color)
				| (Self::Word(_), Kind::Word(_))
				| (Self::Entity(_), Kind::Entity)
				| (Self::Body(_), Kind::Body)
				| (Self::Joint(_), Kind::Joint)
				| (Self::Pose(_), Kind::Pose)
				| (Self::Mesh(_), Kind::Mesh)
				| (Self::Material(_), Kind::Material)
				| (Self::Texture(_), Kind::Texture)
		)
	}

	/// Whether every number in it is one that can be written down.
	///
	/// A file has no spelling for an infinity or a nan, and a world that has
	/// blown up is full of both; a writer asks this before it writes.
	#[must_use]
	pub fn is_finite(&self) -> bool {
		match self {
			| Self::Float(held) => held.is_finite(),
			| Self::Vec3(held) | Self::Color(held) => held.is_finite(),
			| Self::Quat(held) => held.is_finite(),
			| _ => true,
		}
	}
}

/// A console value is a field value; the four kinds are the same four.
impl From<cvar::Value> for Value {
	fn from(value: cvar::Value) -> Self {
		match value {
			| cvar::Value::Bool(held) => Self::Bool(held),
			| cvar::Value::Int(held) => Self::Int(held),
			| cvar::Value::Float(held) => Self::Float(held),
			| cvar::Value::Text(held) => Self::Text(held),
		}
	}
}

/// A field value is a console value only when it is one of the console's four
/// kinds; the rest come back as the error, unchanged, so that nothing is lost
/// by asking.
impl TryFrom<Value> for cvar::Value {
	type Error = Value;

	fn try_from(value: Value) -> Result<Self, Value> {
		match value {
			| Value::Bool(held) => Ok(Self::Bool(held)),
			| Value::Int(held) => Ok(Self::Int(held)),
			| Value::Float(held) => Ok(Self::Float(held)),
			| Value::Text(held) => Ok(Self::Text(held)),
			| other => Err(other),
		}
	}
}

/// One field of a record: its name, its kind, and the two functions that
/// reach it.
///
/// Plain function pointers rather than a trait, because a table of these is
/// a `const` the record declares beside itself, and a `const` can hold a
/// pointer and cannot hold a boxed closure. Nothing is captured: the field
/// knows where it is in the record and nothing else.
pub struct Field<T> {
	/// What the field is called: the record's own field name, dotted for one
	/// inside another (`shape.radius`).
	///
	/// A file's key and an inspector's label, so it is the name a person
	/// would look for in the struct.
	pub name: &'static str,

	/// One line saying what it is, for a tooltip or a listing.
	pub help: &'static str,

	/// What kind of value it holds.
	pub kind: Kind,

	/// Reads it.
	pub get: fn(&T) -> Value,

	/// Writes it.
	///
	/// Refuses, writing nothing, when the value is not of the field's kind or
	/// is a [`Value::Word`] past the end of the field's list.
	pub set: fn(&mut T, Value) -> bool,
}

impl<T> Field<T> {
	/// Reads the field out of a record.
	///
	/// @param record - what to read from
	#[must_use]
	pub fn get(&self, record: &T) -> Value { (self.get)(record) }

	/// Writes the field into a record.
	///
	/// @param record - what to write into
	/// @param value - what to write; refused, with nothing written, when it is
	/// not of the field's kind
	/// @return whether it was written
	pub fn set(&self, record: &mut T, value: Value) -> bool { (self.set)(record, value) }
}

/// A field is a name and two pointers, whatever the record is.
impl<T> Clone for Field<T> {
	fn clone(&self) -> Self { *self }
}

impl<T> Copy for Field<T> {}

impl<T> core::fmt::Debug for Field<T> {
	fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		formatter
			.debug_struct("Field")
			.field("name", &self.name)
			.field("kind", &self.kind)
			.finish_non_exhaustive()
	}
}

/// One row of a table, for a field whose type is the value's own.
///
/// The kind is the [`Value`] variant, the path is the field's place in the
/// record, and the two functions are written out from those. A field that
/// is not one of its value's own type - an enumeration, a `u32` read as a
/// whole number - is written by hand beside the rows this makes.
macro_rules! field {
	($kind:ident, $name:literal, $($path:ident).+, $help:literal) => {
		$crate::abi::field::Field {
			name: $name,
			help: $help,
			kind: $crate::abi::field::Kind::$kind,
			get: |record| $crate::abi::field::Value::$kind(record.$($path).+),
			set: |record, value| match value {
				| $crate::abi::field::Value::$kind(held) => {
					record.$($path).+ = held;

					true
				},
				| _ => false,
			},
		}
	};
}

pub(crate) use field;

/// One row for a field that is one of a few words.
///
/// The enumeration says what its words are and how to get from an index to a
/// variant and back; this writes the two functions around that.
///
/// @param $words - the enumeration's list of spellings
/// @param $at - the enumeration's `fn(u32) -> Option<Self>`
/// @param $index - the enumeration's `fn(self) -> u32`
macro_rules! word {
	($name:literal, $($path:ident).+, $words:expr, $at:expr, $index:expr, $help:literal) => {
		$crate::abi::field::Field {
			name: $name,
			help: $help,
			kind: $crate::abi::field::Kind::Word($words),
			get: |record| $crate::abi::field::Value::Word($index(record.$($path).+)),
			set: |record, value| match value {
				| $crate::abi::field::Value::Word(index) => match $at(index) {
					| Some(held) => {
						record.$($path).+ = held;

						true
					},
					| None => false,
				},
				| _ => false,
			},
		}
	};
}

pub(crate) use word;

#[cfg(test)]
mod tests {
	use super::*;
	use crate::abi::{
		Body, BodyKind, Camera, Joint, JointKind, Layers, Renderable, Shape, ShapeKind,
		Transform, scene::Stage,
	};

	/// A value of each kind, none of them what a fresh record holds.
	fn sample(kind: Kind) -> Value {
		match kind {
			| Kind::Bool => Value::Bool(true),
			// small and positive, so that a whole number read as a bit mask takes
			// it too
			| Kind::Int => Value::Int(6),
			| Kind::Float => Value::Float(2.5),
			| Kind::Text => Value::Text("hello".to_owned()),
			| Kind::Vec2 => Value::Vec2(Vec2::new(1.5, 2.5)),
			| Kind::Vec3 => Value::Vec3(Vec3::new(1.0, 2.0, 3.0)),
			| Kind::Quat => Value::Quat(Quat::from_rotation_y(0.5)),
			| Kind::Color => Value::Color(Vec3::new(0.2, 0.4, 0.6)),
			| Kind::Word(words) => Value::Word(u32::try_from(words.len()).unwrap_or(1) - 1),
			| Kind::Entity => Value::Entity(EntityId::at(3, 2)),
			| Kind::Body => Value::Body(BodyId::at(3, 2)),
			| Kind::Joint => Value::Joint(JointId::NONE),
			| Kind::Pose => Value::Pose(PoseId::NONE),
			| Kind::Mesh => Value::Mesh(MeshId::new(4)),
			| Kind::Material => Value::Material(MaterialId::new(2)),
			| Kind::Texture => Value::Texture(TextureId::new(5)),
		}
	}

	/// A value of some other kind than the one asked for.
	fn other(kind: Kind) -> Value {
		if kind == Kind::Text {
			Value::Bool(true)
		} else {
			Value::Text("wrong".to_owned())
		}
	}

	/// What every table has to be, whatever it describes.
	fn table_holds<T: Clone + PartialEq + core::fmt::Debug>(fresh: &T, fields: &[Field<T>]) {
		assert!(!fields.is_empty(), "a table with nothing in it describes nothing");

		for (index, field) in fields.iter().enumerate() {
			assert!(!field.name.is_empty(), "row {index} has no name");
			assert!(!field.help.is_empty(), "{} has no help", field.name);
			assert!(
				!fields[..index]
					.iter()
					.any(|it| it.name == field.name),
				"{} is in the table twice",
				field.name
			);
			assert!(
				field.get(fresh).is(field.kind),
				"{} reads as something other than its kind: {:?}",
				field.name,
				field.get(fresh)
			);

			// a value of the field's kind lands, and reads back as itself.
			let mut record = fresh.clone();
			let value = sample(field.kind);

			assert!(field.set(&mut record, value.clone()), "{} refused its own kind", field.name);
			assert_eq!(
				field.get(&record),
				value,
				"{} did not read back what was written",
				field.name
			);

			// and one of another kind is refused, with nothing written.
			let mut untouched = fresh.clone();

			assert!(
				!field.set(&mut untouched, other(field.kind)),
				"{} took a value of the wrong kind",
				field.name
			);
			assert_eq!(untouched, *fresh, "{} wrote something on being refused", field.name);
		}
	}

	#[test]
	fn every_table_reads_back_what_it_writes_and_refuses_the_wrong_kind() {
		table_holds(&Transform::IDENTITY, Transform::FIELDS);
		table_holds(&Renderable::NOTHING, Renderable::FIELDS);
		table_holds(&Body::default(), Body::FIELDS);
		table_holds(&Joint::default(), Joint::FIELDS);
		table_holds(&Camera::DEFAULT, Camera::FIELDS);
		table_holds(&Stage::DEFAULT, Stage::FIELDS);
	}

	#[test]
	fn a_word_past_the_end_of_its_list_is_refused() {
		let mut body = Body::default();
		let kind = Body::FIELDS
			.iter()
			.find(|it| it.name == "kind")
			.expect("a body has a kind");
		let words = u32::try_from(kind.kind.words().len()).expect("a short list");

		assert!(!kind.set(&mut body, Value::Word(words)), "one past the end is not a word");
		assert_eq!(body.kind, BodyKind::Static, "and nothing was written");
		assert!(kind.set(&mut body, Value::Word(words - 1)), "the last one is");
		assert_eq!(body.kind, BodyKind::Dynamic, "and it is the last variant declared");
	}

	#[test]
	fn a_words_list_is_the_enumeration_in_declaration_order() {
		// each list is read by a file and shown by an inspector, so the
		// spelling and the order are both part of the format.
		assert_eq!(BodyKind::WORDS, &["static", "kinematic", "dynamic"]);
		assert_eq!(ShapeKind::WORDS, &["box", "sphere", "mesh"]);
		assert_eq!(JointKind::WORDS, &["rope", "weld", "axis", "ball"]);

		for (index, word) in JointKind::WORDS.iter().enumerate() {
			let index = u32::try_from(index).expect("a short list");
			let kind = JointKind::at(index).expect("every word is a variant");

			assert_eq!(kind.index(), index, "{word} goes there and back");
			assert_eq!(
				Kind::Word(JointKind::WORDS).word(word),
				Some(index),
				"and is found by name"
			);
		}

		assert_eq!(JointKind::at(4), None, "and there is no fifth joint");
		assert_eq!(Kind::Word(JointKind::WORDS).word("spring"), None, "nor a spring");
		assert_eq!(Kind::Float.word("rope"), None, "and a number has no words at all");
	}

	#[test]
	fn a_reference_is_a_handle_and_nothing_else_is() {
		for kind in
			[Kind::Entity, Kind::Body, Kind::Joint, Kind::Pose, Kind::Mesh, Kind::Material]
		{
			assert!(kind.is_reference(), "{kind:?} points into a table");
		}

		for kind in [
			Kind::Bool,
			Kind::Int,
			Kind::Float,
			Kind::Text,
			Kind::Vec3,
			Kind::Quat,
			Kind::Color,
			Kind::Word(BodyKind::WORDS),
		] {
			assert!(!kind.is_reference(), "{kind:?} is plain data a file can spell");
		}
	}

	#[test]
	fn a_value_is_of_its_variants_kind_and_a_word_is_of_every_word_kind() {
		assert!(Value::Float(1.0).is(Kind::Float));
		assert!(!Value::Float(1.0).is(Kind::Int), "a number is not a whole number");
		assert!(!Value::Vec3(Vec3::ONE).is(Kind::Color), "and three numbers are not a color");
		assert!(Value::Word(0).is(Kind::Word(BodyKind::WORDS)));
		assert!(Value::Word(0).is(Kind::Word(JointKind::WORDS)), "whichever list");
	}

	#[test]
	fn a_number_that_cannot_be_written_says_so() {
		assert!(Value::Float(1.0).is_finite());
		assert!(!Value::Float(f32::NAN).is_finite());
		assert!(!Value::Vec3(Vec3::new(0.0, f32::INFINITY, 0.0)).is_finite());
		assert!(!Value::Color(Vec3::splat(f32::NAN)).is_finite());
		assert!(!Value::Quat(Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0)).is_finite());
		assert!(Value::Word(3).is_finite(), "and a word has no number in it to be wrong");
	}

	#[test]
	fn the_consoles_four_kinds_cross_both_ways_and_the_rest_come_back() {
		assert_eq!(Value::from(cvar::Value::Float(2.5)), Value::Float(2.5));
		assert_eq!(Value::from(cvar::Value::Bool(true)), Value::Bool(true));
		assert_eq!(Value::from(cvar::Value::Int(-3)), Value::Int(-3));
		assert_eq!(Value::from(cvar::Value::Text("a".to_owned())), Value::Text("a".to_owned()));

		assert_eq!(cvar::Value::try_from(Value::Float(2.5)), Ok(cvar::Value::Float(2.5)));
		assert_eq!(
			cvar::Value::try_from(Value::Vec3(Vec3::ONE)),
			Err(Value::Vec3(Vec3::ONE)),
			"a console variable cannot hold three numbers, and the three come back intact"
		);
		assert_eq!(
			cvar::Value::try_from(Value::Entity(EntityId::NONE)),
			Err(Value::Entity(EntityId::NONE)),
			"nor a handle"
		);
	}

	#[test]
	fn a_field_is_read_through_the_table_as_it_is_in_the_record() {
		let body = Body {
			mass: 4.0,
			shape: Shape { radius: 2.0, ..Shape::UNIT },
			layers: Layers::new(1, 5),
			..Body::default()
		};

		let read = |name: &str| {
			Body::FIELDS
				.iter()
				.find(|it| it.name == name)
				.map(|it| it.get(&body))
				.unwrap_or_else(|| panic!("a body has {name}"))
		};

		assert_eq!(read("mass"), Value::Float(4.0));
		assert_eq!(read("shape.radius"), Value::Float(2.0), "a field inside another, dotted");
		assert_eq!(read("layers.mask"), Value::Int(5), "a bit mask reads as a whole number");
		assert_eq!(read("kind"), Value::Word(0), "and an enumeration as its index");
	}

	#[test]
	fn a_whole_number_a_mask_cannot_hold_is_refused() {
		let mut body = Body::default();
		let mask = Body::FIELDS
			.iter()
			.find(|it| it.name == "layers.mask")
			.expect("a body has a mask");

		assert!(!mask.set(&mut body, Value::Int(-1)), "a mask has no sign");
		assert!(!mask.set(&mut body, Value::Int(1 << 40)), "nor forty bits");
		assert_eq!(body.layers.mask, u32::MAX, "and neither was written");
		assert!(mask.set(&mut body, Value::Int(6)));
		assert_eq!(body.layers.mask, 6);
	}

	#[test]
	fn a_field_prints_as_its_name_and_kind() {
		let printed = format!("{:?}", Transform::FIELDS[0]);

		assert!(printed.contains("position"), "got {printed}");
		assert!(printed.contains("Vec3"), "got {printed}");
	}
}
