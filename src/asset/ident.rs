//! What an asset *is*, as against what it is called.
//!
//! An asset's name is its path under the source tree with the extension
//! dropped - @ref [`compile::asset_name`](crate::compile::asset_name) - and a
//! name is an address, not an identity. Rename `meshes/crystal.obj` and every
//! scene that named it points at nothing: a mesh handle resolves to slot zero,
//! which means "nothing", and the entity draws no mesh without a word said. An
//! identity is the second half of the answer: a number a file keeps for as long
//! as it exists, wherever it is moved to and whatever it is called.
//!
//! ```text
//!   assets/meshes/crystal.obj        the source
//!   assets/meshes/crystal.obj.id     id://abcdefghijklm
//!   .colby/assets/.ids               id://abcdefghijklm   meshes/crystal
//! ```
//!
//! **The sidecar is the truth and the table is derived.** A `.id` beside a
//! source is a file in the source tree that is an input and not a source - the
//! third of that shape, after a stylesheet and an import sidecar - and it is
//! the only copy that survives `just clean`, so it is what belongs in version
//! control. The table under the output tree is rebuilt from the sidecars on
//! every pass, sits beside the stamps for the same reasons they do, and is what
//! a lookup actually reads.
//!
//! **An id is born deterministically**, out of the project's own id and the
//! name the asset first had, so that two machines compiling one fresh tree
//! agree about every id in it. That is not a nicety here: a fixture built on
//! one machine and the same fixture built in CI are compared byte for byte, and
//! an identity drawn at random would have to be excepted from every one of
//! those comparisons. A number that is already taken is redrawn - @ref
//! [`born`] - and the redraw is deterministic too, because the walk that visits
//! the sources is sorted.
//!
//! **Determinism only reaches as far as the birth.** Afterwards the sidecar
//! holds the number and nothing recomputes it; that is the whole point, since a
//! renamed file must keep the id its old name gave it. Two files that swap
//! names keep their own ids, and only a tree with no sidecars at all would hand
//! them each other's.
//!
//! **What refers by id.** Sources do: a `.scene` and a `.material` take
//! `id://...` anywhere they take a name, and may carry a block naming the ids
//! behind the names in their body - @ref [`block`]. Compiled files do not: the
//! compiler resolves every id to the name it stands for and writes the name, so
//! `.cscene`, `.cmat` and `.cmodel` are exactly what they were, and so are the
//! console, the scripting API and the wire.
//!
//! **The registries are the one thing past the compiler that carries one**, and
//! it is not addressing - nothing looks an asset up by id to draw it. It is so
//! that a rename can be *followed*: the asset loop hands each table the
//! identity of what it just loaded, a table that already holds that identity
//! under the old name moves the entry rather than appending a second one, and
//! every handle the live world is holding goes on resolving. @ref
//! [`Registry::adopt`](colby_core::abi::registry::Registry::adopt).
//!
//! @ref [`stamp`](crate::stamp) for how a rename reaches the outputs that have
//! to be built again.

use std::{
	collections::BTreeMap,
	fs,
	path::{Path, PathBuf},
};

/// Where a source keeps its identity, and what the whole tree's table says.
///
/// The spelling itself - the scheme, the thirteen digits, [`Id`] - lives in
/// [`colby_core::abi::ident`] and is re-exported here, because a registry entry
/// carries one and the engine has no compiler in it. What is below is what only
/// a compiler has any use for: where an identity comes from, where it is kept
/// beside a source, and how a tree of them is read back.
pub use colby_core::abi::ident::{DIGITS, Id, SCHEME};
use colby_core::{Result, err};

use crate::{json::Value, project};

/// The extension a sidecar is written with, after the source's whole name.
///
/// `crystal.obj.id`, the spelling an import sidecar already uses and for its
/// reason: `crystal.obj` and `crystal.gltf` may stand in one directory, and a
/// `crystal.id` between them would belong to neither.
pub const EXTENSION: &str = "id";

/// What the table is called, inside the output tree.
///
/// A leading dot and none of the extensions the compiler writes, so that the
/// walk collecting outputs and the sweep deleting orphans both look straight
/// past it - the same rule, and the same reason, as the stamps beside it.
pub const FILE: &str = ".ids";

/// The first line of the table, and what a reader checks before the rest.
pub const MARK: &str = "colby ids 1";

/// The key a source writes its block of ids under.
pub const BLOCK: &str = "assets";

/// The id a name is born with, on the given attempt.
///
/// Out of the project's id and the asset's name, and nothing else: not the
/// file's contents, which would tie every id to bytes that generators of
/// fixtures have no reason to keep identical, and not the clock or a random
/// source, which would make two machines disagree about a tree neither has
/// touched.
///
/// **The project's id is in it** so that two projects holding a
/// `meshes/crystal` do not hold one id between them. Without it the collision
/// would not be unlikely, it would be certain, and it would be discovered the
/// day somebody copied one tree into another.
///
/// **The name is folded to lowercase** first. A file system that does not tell
/// `Crystal.obj` from `crystal.obj` would otherwise hand one file two ids
/// depending on how a directory listing happened to spell it.
///
/// An attempt past the first is how a collision is settled: the caller counts
/// up until the number it gets is free. Both the draw and the order the sources
/// are visited in are fixed, so the settlement is as reproducible as the first
/// draw. The number nought is treated as taken by every caller, since it means
/// "no id"; it comes up about as often as any other.
///
/// @param salt - the project's id, or nothing for a tree that has no project
/// @param name - the asset name, `meshes/crystal`
/// @param attempt - nought first, then up while the draw is taken
#[must_use]
pub fn born(salt: &str, name: &str, attempt: u32) -> Id {
	let lowered = name.to_lowercase();
	// a separator no asset name may hold, so that a project `a` with an asset
	// `b/c` cannot be spelled the same way as a project `a/b` with an asset `c`
	let seeded = fnv(salt.bytes().chain([0]).chain(lowered.bytes()));

	Id::from_bits(mix(seeded.wrapping_add(u64::from(attempt))))
}

/// Sixty-four bits of FNV-1a over a run of bytes.
fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
	bytes.fold(0xCBF2_9CE4_8422_2325, |held, byte| {
		(held ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
	})
}

/// Spreads a seed's bits over the whole word.
///
/// FNV moves the low bits well and the high ones poorly, and an id is read
/// from the top down, so the digits a person sees first would otherwise hardly
/// differ between two neighboring names.
fn mix(seed: u64) -> u64 {
	let mut mixed = seed;

	mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
	mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

	mixed ^ (mixed >> 31)
}

/// Every id in a tree, both ways round.
///
/// Built from the sidecars on every pass and written out as [`FILE`]; read back
/// by whoever has to turn an id into a name. Losing it costs one pass of
/// rebuilding and nothing else, which is what makes the output tree - already
/// derived, already taken by `just clean` - the right place for it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ids {
	/// What each id is called.
	names: BTreeMap<Id, String>,

	/// What each name is, keyed the other way so that the pass which restores
	/// a lost sidecar does not have to walk the whole table for one name.
	ids: BTreeMap<String, Id>,
}

impl Ids {
	/// An empty table.
	#[must_use]
	pub fn new() -> Self { Self::default() }

	/// How many assets it holds.
	#[must_use]
	pub fn len(&self) -> usize { self.names.len() }

	/// Whether it holds none.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.names.is_empty() }

	/// What an id is called, if anything here is.
	///
	/// @param id - the identity
	#[must_use]
	pub fn name(&self, id: Id) -> Option<&str> { self.names.get(&id).map(String::as_str) }

	/// What a name's identity is, if it has one here.
	///
	/// @param name - the asset name
	#[must_use]
	pub fn id(&self, name: &str) -> Option<Id> { self.ids.get(name).copied() }

	/// Whether anything here already carries this id.
	///
	/// @param id - the identity
	#[must_use]
	pub fn holds(&self, id: Id) -> bool { self.names.contains_key(&id) }

	/// Files an id against a name.
	///
	/// **An id already here is refused rather than moved**, and what comes back
	/// is who holds it. Two sources carrying one id is the shape a copied file
	/// takes when its sidecar is copied with it, and the honest answer is to
	/// say so: quietly letting the second win would leave one of the two
	/// loading as the other, which is the fault this whole module exists to
	/// remove.
	///
	/// A name that was here under a different id moves, since that is what
	/// editing a sidecar by hand means.
	///
	/// @param id - the identity, which may not be [`Id::NONE`]
	/// @param name - the asset name
	/// @return nothing when it was filed, or what already holds that id
	pub fn put(&mut self, id: Id, name: &str) -> Option<&str> {
		if id.is_none() {
			return None;
		}

		if self.names.contains_key(&id) {
			return self.names.get(&id).map(String::as_str);
		}

		if let Some(held) = self.ids.insert(name.to_owned(), id) {
			self.names.remove(&held);
		}

		self.names.insert(id, name.to_owned());

		None
	}

	/// The table as it is written down: a mark, then a line per asset.
	///
	/// Text, so that a person looking at a tree that will not settle can read
	/// the answer, and sorted by name so that two passes over one tree write
	/// the same bytes.
	#[must_use]
	pub fn encode(&self) -> String {
		let mut text = String::from(MARK);

		text.push('\n');

		for (name, id) in &self.ids {
			text.push_str(&id.to_string());
			text.push('\t');
			text.push_str(name);
			text.push('\n');
		}

		text
	}

	/// Reads a table back.
	///
	/// @param text - the whole file
	/// @return the table, or nothing when the mark is not the one this build
	/// writes or a line is not a pair
	#[must_use]
	pub fn decode(text: &str) -> Option<Self> {
		let mut lines = text.lines();

		if lines.next()? != MARK {
			return None;
		}

		let mut held = Self::new();

		for line in lines.filter(|line| !line.is_empty()) {
			let (id, name) = line.split_once('\t')?;

			held.put(Id::parse(id)?, name);
		}

		Some(held)
	}

	/// The table an output tree holds, or an empty one.
	///
	/// Never an error: every way this can fail - no file, a file from another
	/// build, a half-written line - means the same thing, which is that the
	/// ids have to be worked out again, and a pass that does that is a pass
	/// that costs one rebuild.
	///
	/// @param out - the output tree
	#[must_use]
	pub fn read(out: &Path) -> Self {
		fs::read_to_string(out.join(FILE))
			.ok()
			.and_then(|text| Self::decode(&text))
			.unwrap_or_default()
	}

	/// Writes the table into an output tree.
	///
	/// @param out - the output tree, which is created if it is not there
	///
	/// # Errors
	///
	/// If the directory cannot be made or the file cannot be written.
	pub fn write(&self, out: &Path) -> Result<()> {
		fs::create_dir_all(out)?;
		fs::write(out.join(FILE), self.encode())?;

		Ok(())
	}
}

/// Where the id sidecar of a source would be, whether or not there is one.
///
/// @param source - the file it would accompany
#[must_use]
pub fn beside(source: &Path) -> PathBuf {
	let name = source
		.file_name()
		.map(|name| name.to_string_lossy().into_owned())
		.unwrap_or_default();

	source.with_file_name(format!("{name}.{EXTENSION}"))
}

/// The id a source carries, if it carries one.
///
/// Nothing rather than [`Id::NONE`] when there is no file, so that a caller can
/// tell "has never been given one" from "has one and it is unreadable" - the
/// first is the ordinary case for a file somebody has just added and the second
/// is a file that must not be given a new id behind its owner's back.
///
/// @param source - the file in the source tree
/// @return its id, or nothing when it has no sidecar
///
/// # Errors
///
/// If the sidecar is there and does not hold exactly one id.
pub fn read_beside(source: &Path) -> Result<Option<Id>> {
	let path = beside(source);

	if !path.is_file() {
		return Ok(None);
	}

	let text = fs::read_to_string(&path)?;
	let read = text
		.lines()
		.find(|line| !line.trim().is_empty())
		.unwrap_or_default();

	Id::parse(read.trim()).map(Some).ok_or_else(|| {
		err!(Asset(
			"{}: {read:?} is not an identity this build reads; an id is {SCHEME} and thirteen \
			 letters, and deleting this file gives the asset a new one",
			path.display()
		))
	})
}

/// Writes a source's id beside it.
///
/// One line and nothing else: it is read by a person about as often as it is
/// read by the compiler, and there is exactly one thing to say.
///
/// @param source - the file in the source tree
/// @param id - what it is
///
/// # Errors
///
/// If the file cannot be written.
pub fn write_beside(source: &Path, id: Id) -> Result<()> {
	fs::write(beside(source), format!("{id}\n"))?;

	Ok(())
}

/// What a source tree's ids are salted with: the project's own id.
///
/// Read quietly and by hand rather than through [`Project::open`], which warns
/// about an engine version and a directory name - this is asked four times a
/// second by the runner's compile loop, and a warning said that often is a
/// warning nobody reads. A tree with no project above it salts with nothing,
/// which is what a workspace built by a test is.
///
/// @param root - the source tree, `<project>/assets`
///
/// [`Project::open`]: crate::Project::open
#[must_use]
pub fn salt(root: &Path) -> String {
	root.parent()
		.and_then(|dir| {
			let text = fs::read_to_string(dir.join(project::FILE)).ok()?;

			project::Project::parse(dir, &text)
				.ok()
				.map(|project| project.id().to_owned())
		})
		.unwrap_or_default()
}

/// The block a source may carry, naming the ids behind the names in its body.
///
/// ```json
/// { "assets": { "meshes/crystal": "id://abcdefghijklm" } }
/// ```
///
/// **Why a block rather than an id in every field.** A source is something a
/// person reads and writes, and a body whose every reference is thirteen random
/// letters is one nobody can read. With the block the body keeps naming things
/// by name, the ids sit in one place at the top, and - this is the part that
/// matters - **a rename changes nothing in the file at all**: the id still
/// resolves, and the name beside it is refreshed the next time the editor
/// writes the file out.
///
/// A name in the block that the body never uses is not an error. The block is
/// what the file knows, and a reference somebody deleted leaves a line behind
/// until the file is written again.
///
/// @param root - the whole parsed source
/// @return every pair, in the order they were written
///
/// # Errors
///
/// If the block is not an object, or a value in it is not an id.
pub fn block(root: &Value) -> Result<Vec<(String, Id)>> {
	let Some(held) = root.get(BLOCK) else {
		return Ok(Vec::new());
	};

	if !matches!(held, Value::Object(_)) {
		return Err(err!(Asset("{BLOCK} is a table of asset names to the ids behind them")));
	}

	held.as_object()
		.iter()
		.map(|(name, value)| {
			let Some(written) = value.as_str() else {
				return Err(err!(Asset("{BLOCK}: {name} is an id, written as text")));
			};

			let id = Id::parse(written).ok_or_else(|| {
				err!(Asset(
					"{BLOCK}: {name} is {written:?}, which is not an identity this build reads"
				))
			})?;

			Ok((name.clone(), id))
		})
		.collect()
}

/// Every identity a parsed source refers to, however it spelled it.
///
/// A walk over the whole document rather than a list of the fields that may
/// hold a reference, and deliberately: what a scene depends on beyond its own
/// text is a question the staleness sweep asks about every kind of source, and
/// a list per kind would have to be kept in step with every field either format
/// grows. An id is a spelling nothing else in either file can take, so looking
/// for the spelling is both simpler and harder to get wrong. Both a written-out
/// reference and a line of the block are found by the same pass, since the
/// block's values are text in the same document.
///
/// @param root - the whole parsed source
/// @return every id in it, in the order they were written, with none repeated
#[must_use]
pub fn referenced(root: &Value) -> Vec<Id> {
	let mut found = Vec::new();

	gather(root, &mut found);

	found
}

/// Adds every id under a value to a list, keeping none twice.
fn gather(value: &Value, found: &mut Vec<Id>) {
	match value {
		| Value::String(written) =>
			if let Some(id) = Id::parse(written)
				&& !found.contains(&id)
			{
				found.push(id);
			},
		| Value::Array(held) =>
			for value in held {
				gather(value, found);
			},
		| Value::Object(held) =>
			for (_, value) in held {
				gather(value, found);
			},
		| Value::Null | Value::Bool(_) | Value::Number(_) => (),
	}
}

/// What the names in one source file really point at.
///
/// Holds the file's own block and the tree's table, and answers the one
/// question a reader has about every reference in the file: which asset is
/// this. @ref [`Resolve::name`].
pub struct Resolve<'a> {
	/// Every id in the tree.
	ids: &'a Ids,

	/// What the file itself said about the names in its body.
	block: Vec<(String, Id)>,
}

impl<'a> Resolve<'a> {
	/// A reader for a file that carries no block.
	///
	/// @param ids - the tree's table
	#[must_use]
	pub const fn over(ids: &'a Ids) -> Self { Self { ids, block: Vec::new() } }

	/// A reader for a file and the block it carries.
	///
	/// @param ids - the tree's table
	/// @param block - what [`block`] read out of the file
	#[must_use]
	pub const fn with(ids: &'a Ids, block: Vec<(String, Id)>) -> Self { Self { ids, block } }

	/// What a written reference names.
	///
	/// Three answers in one rule, in this order: an id written out is that
	/// asset; a name the block speaks for is whatever that id is now called;
	/// anything else is itself. So a file may mix the two spellings freely, a
	/// file with no block behaves exactly as it did before ids existed, and a
	/// name in the block is a comment - correct when it was written and
	/// harmless when it is not.
	///
	/// **An id nothing answers to is an error**, where a *name* nothing answers
	/// to is not. The difference is deliberate: a name may be one an asset
	/// compiled later in the same pass will take, and the world says so at
	/// load; an id can only have come from a file that once existed, so a
	/// dangling one means something was deleted and nobody was told.
	///
	/// @param written - what the file said
	///
	/// # Errors
	///
	/// If the reference is spelled as an id and is not one, or is one that
	/// nothing in the tree carries.
	pub fn name(&self, written: &str) -> Result<String> {
		if written.is_empty() {
			return Ok(String::new());
		}

		if Id::spelled(written) {
			let id = Id::parse(written)
				.ok_or_else(|| err!(Asset("{written:?} is not an identity this build reads")))?;

			return self.named(id);
		}

		match self
			.block
			.iter()
			.find(|(name, _)| name == written)
		{
			| Some((_, id)) => self.named(*id),
			| None => Ok(written.to_owned()),
		}
	}

	/// What the tree calls an id, or why it cannot say.
	fn named(&self, id: Id) -> Result<String> {
		self.ids
			.name(id)
			.map(str::to_owned)
			.ok_or_else(|| {
				err!(Asset(
					"nothing in this project carries {id}; the asset it named has been deleted, \
					 or its {EXTENSION} file has"
				))
			})
	}
}

#[cfg(test)]
mod tests {
	use std::env;

	use super::*;

	/// A directory nobody else is using, removed and recreated.
	fn workspace(name: &str) -> PathBuf {
		let dir = env::temp_dir()
			.join("colby-ident-tests")
			.join(name);

		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(&dir).expect("the fixture is made");

		dir
	}

	#[test]
	fn a_born_id_reads_back_the_way_it_was_written() {
		for name in ["meshes/crystal", "a", "scenes/yard/floor", ""] {
			let id = born("yard", name, 0);
			let written = id.to_string();

			assert_eq!(written.len(), SCHEME.len() + DIGITS, "{written} is the width it is");
			assert_eq!(Id::parse(&written), Some(id), "{written} reads back");
		}
	}

	#[test]
	fn one_name_is_born_the_same_id_every_time() {
		let once = born("yard", "meshes/crystal", 0);
		let again = born("yard", "meshes/crystal", 0);

		assert_eq!(once, again, "the draw is a function of what went into it");
		assert!(!once.is_none(), "and it is an id");
	}

	#[test]
	fn two_names_and_two_projects_are_born_apart() {
		let crystal = born("yard", "meshes/crystal", 0);

		assert_ne!(crystal, born("yard", "meshes/gem", 0), "two names differ");
		assert_ne!(crystal, born("shed", "meshes/crystal", 0), "two projects differ");
		assert_ne!(crystal, born("", "meshes/crystal", 0), "and so does no project");
	}

	#[test]
	fn a_project_and_a_name_cannot_be_spelled_into_each_other() {
		assert_ne!(
			born("yard", "meshes/crystal", 0),
			born("yardmeshes", "/crystal", 0),
			"the separator between them is a character neither may hold"
		);
	}

	#[test]
	fn a_second_attempt_draws_a_different_number() {
		let first = born("yard", "meshes/crystal", 0);
		let second = born("yard", "meshes/crystal", 1);

		assert_ne!(first, second, "a collision has somewhere to go");
		assert_eq!(second, born("yard", "meshes/crystal", 1), "and settles the same way twice");
	}

	#[test]
	fn how_a_name_is_capitalized_does_not_change_its_id() {
		assert_eq!(
			born("yard", "meshes/Crystal", 0),
			born("yard", "meshes/crystal", 0),
			"a file system that does not tell the two apart gives one file one id"
		);
	}

	#[test]
	fn a_table_reads_back_what_it_wrote() {
		let mut ids = Ids::new();

		ids.put(born("yard", "meshes/crystal", 0), "meshes/crystal");
		ids.put(born("yard", "scenes/room", 0), "scenes/room");

		let text = ids.encode();

		assert_eq!(Ids::decode(&text), Some(ids.clone()), "it reads back");
		assert_eq!(ids.encode(), text, "and writes the same bytes twice");
		assert!(text.starts_with(MARK), "with the mark at the top");
	}

	#[test]
	fn a_table_this_build_did_not_write_is_not_read() {
		let id = born("yard", "meshes/crystal", 0);
		let spaced = format!("{MARK}\n{id} meshes/crystal\n");
		let nonsense = format!("{MARK}\nnot-an-id\tmeshes/crystal\n");

		assert_eq!(Ids::decode(""), None, "an empty file is not a table");
		assert_eq!(Ids::decode("colby ids 2\n"), None, "nor one from another build");
		assert_eq!(Ids::decode(&spaced), None, "nor a line with no tab");
		assert_eq!(Ids::decode(&nonsense), None, "nor one whose id is not one");
	}

	#[test]
	fn an_id_something_already_carries_is_refused_and_says_who_has_it() {
		let id = born("yard", "meshes/crystal", 0);
		let mut ids = Ids::new();

		assert_eq!(ids.put(id, "meshes/crystal"), None, "the first is filed");
		assert_eq!(ids.put(id, "meshes/gem"), Some("meshes/crystal"), "the second is not");
		assert_eq!(ids.name(id), Some("meshes/crystal"), "and the first still holds it");
		assert_eq!(ids.id("meshes/gem"), None, "the second holds nothing");
	}

	#[test]
	fn a_name_moved_to_another_id_leaves_the_first_behind() {
		let first = born("yard", "meshes/crystal", 0);
		let second = born("yard", "meshes/crystal", 1);
		let mut ids = Ids::new();

		ids.put(first, "meshes/crystal");
		ids.put(second, "meshes/crystal");

		assert_eq!(ids.id("meshes/crystal"), Some(second), "the sidecar somebody edited wins");
		assert!(!ids.holds(first), "and what it held is free again");
		assert_eq!(ids.len(), 1, "one asset, one line");
	}

	#[test]
	fn a_sidecar_is_named_after_the_whole_of_the_source_beside_it() {
		let dir = workspace("beside");
		let source = dir.join("crystal.obj");

		assert_eq!(beside(&source), dir.join("crystal.obj.id"), "its whole name and ours");
		assert_ne!(
			beside(&dir.join("crystal.gltf")),
			beside(&source),
			"two sources, two sidecars"
		);
	}

	#[test]
	fn a_sidecar_reads_back_what_was_written_beside_it() {
		let dir = workspace("sidecar");
		let source = dir.join("crystal.obj");
		let id = born("yard", "meshes/crystal", 0);

		fs::write(&source, "v 0 0 0\n").expect("the source is written");

		assert_eq!(
			read_beside(&source).expect("no file is not a failure"),
			None,
			"none to start"
		);

		write_beside(&source, id).expect("the sidecar is written");

		assert_eq!(read_beside(&source).expect("and is read"), Some(id), "what was written");
		assert_eq!(
			fs::read_to_string(beside(&source)).expect("it is text"),
			format!("{id}\n"),
			"one line and nothing else"
		);
	}

	#[test]
	fn a_sidecar_that_is_not_an_id_is_refused_rather_than_replaced() {
		let dir = workspace("sidecar-bad");
		let source = dir.join("crystal.obj");

		fs::write(&source, "v 0 0 0\n").expect("the source is written");
		fs::write(beside(&source), "meshes/crystal\n").expect("the sidecar is written");

		let error = read_beside(&source).expect_err("a name is not an identity");

		assert!(
			error.to_string().contains("meshes/crystal"),
			"and the message says what it found: {error}"
		);
	}

	#[test]
	fn a_block_is_read_and_an_id_in_it_is_checked() {
		let id = born("yard", "meshes/crystal", 0);
		let text = format!("{{ \"{BLOCK}\": {{ \"meshes/crystal\": \"{id}\" }} }}");
		let root = crate::json::parse(&text).expect("it is JSON");

		assert_eq!(block(&root).expect("and a block"), vec![("meshes/crystal".to_owned(), id)]);

		let empty = crate::json::parse("{}").expect("it is JSON");

		assert!(
			block(&empty)
				.expect("no block is not a failure")
				.is_empty(),
			"and holds nothing"
		);

		for bad in [
			"{ \"assets\": [] }",
			"{ \"assets\": { \"a\": 1 } }",
			"{ \"assets\": { \"a\": \"b\" } }",
		] {
			let root = crate::json::parse(bad).expect("it is JSON");

			assert!(block(&root).is_err(), "{bad} is refused");
		}
	}

	#[test]
	fn a_reference_resolves_by_id_by_block_and_by_name() {
		let id = born("yard", "meshes/crystal", 0);
		let mut ids = Ids::new();

		ids.put(id, "meshes/gem");

		let by = Resolve::with(&ids, vec![("meshes/crystal".to_owned(), id)]);

		assert_eq!(by.name(&id.to_string()).expect("an id resolves"), "meshes/gem");
		assert_eq!(
			by.name("meshes/crystal")
				.expect("a block name resolves"),
			"meshes/gem"
		);
		assert_eq!(
			by.name("meshes/wall")
				.expect("a plain name is itself"),
			"meshes/wall"
		);
		assert_eq!(by.name("").expect("and nothing is nothing"), "");
	}

	#[test]
	fn an_id_nothing_carries_is_an_error_where_a_name_is_not() {
		let ids = Ids::new();
		let by = Resolve::over(&ids);
		let id = born("yard", "meshes/crystal", 0);

		let error = by
			.name(&id.to_string())
			.expect_err("a dangling id is refused");

		assert!(error.to_string().contains(&id.to_string()), "and says which: {error}");
		assert!(by.name("id://nonsense").is_err(), "and so is a misspelled one");
		assert_eq!(
			by.name("meshes/nowhere")
				.expect("a name is not checked here"),
			"meshes/nowhere",
			"because one may belong to an asset compiled later in the same pass"
		);
	}

	#[test]
	fn every_id_a_source_refers_to_is_found_wherever_it_stands() {
		let first = born("yard", "meshes/crystal", 0);
		let second = born("yard", "meshes/gem", 0);
		let text = format!(
			"{{ \"{BLOCK}\": {{ \"meshes/crystal\": \"{first}\" }},
			    \"entities\": [ {{ \"mesh\": \"{second}\" }}, {{ \"mesh\": \"{second}\" }},
			                   {{ \"mesh\": \"meshes/wall\" }} ] }}"
		);
		let root = crate::json::parse(&text).expect("it is JSON");

		assert_eq!(referenced(&root), vec![first, second], "both spellings, neither twice");
	}

	#[test]
	fn a_tree_with_no_project_over_it_salts_with_nothing() {
		let dir = workspace("salt");
		let assets = dir.join("assets");

		fs::create_dir_all(&assets).expect("the tree is made");

		assert_eq!(salt(&assets), "", "no project file is no salt");

		fs::write(
			dir.join(project::FILE),
			"{ \"schema\": 1, \"engine\": \"0.1.0\", \"id\": \"yard\", \"name\": \"The Yard\" }",
		)
		.expect("the project is written");

		assert_eq!(salt(&assets), "yard", "and one over it is its id");
	}
}
