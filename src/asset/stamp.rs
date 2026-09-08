//! What each output was built out of, and how those files looked when it was.
//!
//! **Nothing here compares two files' times.** The compiler used to ask
//! whether an output was newer than the source it came from, and a filesystem
//! cannot always answer that: an inode's timestamp is taken from a clock the
//! kernel moves on a timer tick, so a source written and an output compiled
//! inside one tick carry the same number down to the nanosecond. "No newer
//! than its source" is then said about an output that is perfectly current,
//! and the whole tree is built again on every pass - four times a second, for
//! as long as the runner is up. Measured on an ext4 whose inodes are too small
//! to hold a nanosecond field at all: source and output equal, every asset,
//! every pass.
//!
//! So a pass writes down the time and the length of every file it read, filed
//! under the name of what it wrote, and the next pass asks whether those are
//! still what is on disk. Equal is untouched. Anything else - newer, older,
//! longer, shorter, gone, or a file that was not in the list before - is a
//! change.
//!
//! **The length is there because a time cannot answer a file written twice
//! inside one tick.** The clock does not move between the two writes, so the
//! time says nothing about the second one; the length usually does, and it
//! costs nothing to ask - the same `stat` that answers one answers both. What
//! is left after that is an edit of exactly the same length inside one tick,
//! which is the one thing no timestamp scheme catches and only reading the
//! file itself would.
//!
//! **Older is the half that comparing two times can never see.** A source
//! restored from a backup or moved back into place carries the time it had
//! rather than the time it arrived, which is older than the output standing
//! beside it, and every rule of the shape "is the source newer" leaves that
//! output alone for good.
//!
//! One file for the whole output tree, [`FILE`], rewritten only by a pass that
//! moved something. Losing it costs one rebuild of the tree and nothing else,
//! which is what makes the output tree - already derived, already deleted by
//! `just clean` - the right place to keep it.
//!
//! # The file
//!
//! Text, so that a person looking into a tree that will not settle can read
//! the answer. `colby stamps 1` on the first line; then a line per output,
//! naming its path under the output tree; then a line per input it was built
//! from, beginning with a tab, holding the time, the length and the path with
//! a tab between each. A time is nanoseconds since the epoch, negative before
//! it, and both numbers are `-` for a file that was not there to read.
//!
//! @note **a time is read before the compiler reads a byte of the file.** A
//! source edited while it is being compiled then carries a time that is not
//! the one on file, and the next pass builds it again. Taking the time
//! afterwards would file the new time against the old contents and lose that
//! edit for good.

use std::{
	collections::{BTreeMap, BTreeSet},
	fs,
	path::Path,
	time::UNIX_EPOCH,
};

use colby_core::Result;

/// What the file is called, inside the output tree.
///
/// A leading dot and none of the extensions the compiler writes, so that the
/// walk collecting outputs and the sweep deleting orphans both look straight
/// past it - @ref `crate::compile::outputs`.
pub const FILE: &str = ".stamps";

/// The first line, and what a reader checks before trusting the rest.
///
/// It carries a number so that a change to the shape of the file is one
/// rebuild of the tree rather than a misreading of it.
const HEADER: &str = "colby stamps 1";

/// What a file that was not there to read is written as.
const ABSENT: &str = "-";

/// One file a compile read, and how it looked when it was read.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Input {
	/// Where it is, under the tree it was found in, with forward slashes.
	pub path: String,

	/// When it was last written - nanoseconds since the epoch, negative before
	/// it - and how many bytes it held. Nothing at all when there was no file
	/// to read: two files that are not there compare equal, which is what
	/// makes a document linking a stylesheet nobody wrote cost one rebuild
	/// rather than one on every pass.
	pub found: Option<(i128, u64)>,
}

impl Input {
	/// Reads one file the compiler is about to open.
	///
	/// @param path - the file, as the compiler will open it
	/// @param root - the tree it lives under, which its name is relative to
	/// @return the name and what to remember it by
	#[must_use]
	pub fn of(path: &Path, root: &Path) -> Self {
		Self {
			path: key(root, path),
			found: found(path),
		}
	}
}

/// Every output's inputs, as the pass that last built each one saw them.
#[derive(Debug, Default)]
pub struct Stamps {
	/// keyed by the output's path under the output tree
	built: BTreeMap<String, Vec<Input>>,

	/// whether anything has been written down or dropped since this was read
	moved: bool,
}

impl Stamps {
	/// Forgets every output but these.
	///
	/// What a run wanted is what the tree is supposed to hold; anything else
	/// was deleted, renamed or failed to compile, and a note about it would
	/// otherwise sit in the file forever.
	///
	/// @param wanted - the names to keep
	pub fn keep(&mut self, wanted: &BTreeSet<String>) {
		let before = self.built.len();

		self.built
			.retain(|output, _| wanted.contains(output));

		self.moved |= self.built.len() != before;
	}

	/// Whether an output is filed under exactly these inputs, as they are now.
	///
	/// @param output - the output's name, from @ref [`key`]
	/// @param read - what compiling it would read now
	/// @return whether it can be left alone
	#[must_use]
	pub fn matches(&self, output: &str, read: &[Input]) -> bool {
		self.built
			.get(output)
			.is_some_and(|had| had == read)
	}

	/// Reads what the last pass left in a tree.
	///
	/// A file that is not there, cannot be read, or was written to another
	/// shape is nothing remembered at all, which costs one rebuild of the
	/// tree. There is no other failure here: none of this is worth stopping a
	/// compile over.
	///
	/// @param out - the output tree
	/// @return what the last pass built, or nothing remembered
	#[must_use]
	pub fn read(out: &Path) -> Self {
		let Ok(text) = fs::read_to_string(out.join(FILE)) else {
			return Self::default();
		};

		let mut lines = text.lines();
		if lines.next() != Some(HEADER) {
			return Self::default();
		}

		let mut built: BTreeMap<String, Vec<Input>> = BTreeMap::new();
		let mut output = String::new();

		for line in lines {
			let Some(input) = line.strip_prefix('\t') else {
				line.clone_into(&mut output);
				built.entry(output.clone()).or_default();

				continue;
			};

			// an input line before any output names one, or one that cannot be
			// read, is a file somebody else wrote into the tree: leave it out
			// and let whatever it belonged to be built again.
			if let (Some(read), Some(taken)) = (built.get_mut(&output), parse(input)) {
				read.push(taken);
			}
		}

		Self { built, moved: false }
	}

	/// Writes down what one output was built out of.
	///
	/// @param output - the output's name, from @ref [`key`]
	/// @param read - what the compiler read to build it, times and lengths
	pub fn set(&mut self, output: String, read: Vec<Input>) {
		self.built.insert(output, read);
		self.moved = true;
	}

	/// Writes the file, if a pass moved anything.
	///
	/// A pass that compiled nothing and deleted nothing writes nothing: the
	/// runner calls the compiler four times a second, and a tree that has
	/// settled should cost a stat per file and no disk at all.
	///
	/// @param out - the output tree, which is made if it is not there
	/// @return whether it could be written
	pub fn write(&self, out: &Path) -> Result {
		if !self.moved {
			return Ok(());
		}

		let mut text = String::from(HEADER);
		text.push('\n');

		for (output, read) in &self.built {
			// a name with a line break in it, or one that would read back as
			// an input line, cannot be written down here at all. Leaving it
			// out costs that one output a rebuild on every pass, which is the
			// price of a file name nothing else in the tree would survive
			// either.
			if !writable(output) || !read.iter().all(|input| writable(&input.path)) {
				continue;
			}

			text.push_str(output);
			text.push('\n');

			for input in read {
				let (time, size) = input.found.map_or_else(
					|| (ABSENT.to_owned(), ABSENT.to_owned()),
					|(time, size)| (time.to_string(), size.to_string()),
				);

				text.push('\t');
				text.push_str(&time);
				text.push('\t');
				text.push_str(&size);
				text.push('\t');
				text.push_str(&input.path);
				text.push('\n');
			}
		}

		fs::create_dir_all(out)?;
		fs::write(out.join(FILE), text)?;

		Ok(())
	}
}

/// The name a file is filed under: its path inside a tree, with forward
/// slashes.
///
/// The shape of an asset's name - @ref `crate::compile::asset_name` - but with
/// the extension still on it, because two sources of one name compile to two
/// outputs and those two have to be told apart.
///
/// @note a file that is not inside the tree keeps its whole path, which is a
/// name like any other. Nothing the compiler reads is outside its tree - @ref
/// `crate::compile::within` - so that is an answer to a case that cannot
/// happen rather than one that is allowed.
///
/// @param root - the tree the file is under
/// @param path - the file
/// @return the name to file it under
#[must_use]
pub fn key(root: &Path, path: &Path) -> String {
	let relative = path.strip_prefix(root).unwrap_or(path);
	let name: Vec<String> = relative
		.components()
		.map(|part| part.as_os_str().to_string_lossy().into_owned())
		.collect();

	name.join("/")
}

/// Reads one input line: the time, the length, and the path.
///
/// The path is everything after the second tab, so a file with a tab in its
/// name reads back as itself.
///
/// @param line - the line without its leading tab
/// @return what it says, or nothing when it says nothing readable
fn parse(line: &str) -> Option<Input> {
	let (time, rest) = line.split_once('\t')?;
	let (size, path) = rest.split_once('\t')?;

	if path.is_empty() {
		return None;
	}

	let found = match (time, size) {
		| (ABSENT, ABSENT) => None,
		| (time, size) => Some((time.parse().ok()?, size.parse().ok()?)),
	};

	Some(Input { path: path.to_owned(), found })
}

/// Whether a name can be written into the file and read back as itself.
fn writable(name: &str) -> bool { !name.starts_with('\t') && !name.contains(['\n', '\r']) }

/// How a file looks right now: when it was written, and how long it is.
///
/// One `stat` answers both, which is the whole argument for keeping the
/// length - @ref the note at the top of this file.
///
/// @param path - the file to read
/// @return nanoseconds since the epoch, negative before it, and the byte
/// count; or nothing when there is no file to read
fn found(path: &Path) -> Option<(i128, u64)> {
	let meta = fs::metadata(path).ok()?;
	let nanos = match meta.modified().ok()?.duration_since(UNIX_EPOCH) {
		| Ok(since) => i128::try_from(since.as_nanos()).ok(),
		| Err(before) => i128::try_from(before.duration().as_nanos())
			.ok()
			.map(|nanos| -nanos),
	}?;

	Some((nanos, meta.len()))
}

#[cfg(test)]
mod tests {
	use std::{
		env,
		fs::File,
		path::PathBuf,
		time::{Duration, SystemTime},
	};

	use super::*;

	/// A scratch output tree nobody else is using.
	fn tree(name: &str) -> PathBuf {
		let dir = env::temp_dir()
			.join("colby-stamp-tests")
			.join(name);

		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(&dir).expect("the directory is made");

		dir
	}

	/// One input, at a time and a length of its own.
	fn input(path: &str, time: i128, size: u64) -> Input {
		Input {
			path: path.to_owned(),
			found: Some((time, size)),
		}
	}

	/// One input that was not there to read.
	fn missing(path: &str) -> Input { Input { path: path.to_owned(), found: None } }

	#[test]
	fn nothing_is_remembered_about_a_tree_nobody_has_compiled() {
		let dir = tree("empty");
		let stamps = Stamps::read(&dir);

		assert!(
			!stamps.matches("meshes/box.cmesh", &[input("meshes/box.obj", 1, 8)]),
			"an output nobody wrote down is one nobody can leave alone"
		);
	}

	#[test]
	fn what_one_pass_wrote_down_the_next_pass_reads_back() {
		let dir = tree("round-trip");
		let mut stamps = Stamps::default();
		let document =
			vec![input("ui/hud.html", 1_788_891_849_000_000_000, 8), missing("ui/missing.css")];

		stamps.set("meshes/box.cmesh".to_owned(), vec![input("meshes/box.obj", -12, 8)]);
		stamps.set("ui/hud.cdoc".to_owned(), document.clone());
		stamps.write(&dir).expect("the file is written");

		let read = Stamps::read(&dir);

		assert!(
			read.matches("meshes/box.cmesh", &[input("meshes/box.obj", -12, 8)]),
			"a time before the epoch reads back as itself"
		);
		assert!(read.matches("ui/hud.cdoc", &document), "and so does a file that was not there");
		assert!(
			!read.matches("ui/hud.cdoc", &document[..1]),
			"while an input that has gone from the list is a change"
		);
	}

	#[test]
	fn a_time_that_moved_either_way_is_a_change() {
		let dir = tree("moved");
		let mut stamps = Stamps::default();

		stamps.set("meshes/box.cmesh".to_owned(), vec![input("meshes/box.obj", 500, 8)]);
		stamps.write(&dir).expect("the file is written");

		let read = Stamps::read(&dir);

		assert!(
			read.matches("meshes/box.cmesh", &[input("meshes/box.obj", 500, 8)]),
			"the same time"
		);
		assert!(
			!read.matches("meshes/box.cmesh", &[input("meshes/box.obj", 501, 8)]),
			"a newer one is a change"
		);
		assert!(
			!read.matches("meshes/box.cmesh", &[input("meshes/box.obj", 499, 8)]),
			"an older one is the change comparing two times could never see"
		);
		assert!(
			!read.matches("meshes/box.cmesh", &[input("meshes/box.obj", 500, 9)]),
			"and a file of exactly the same age that got longer is the change no clock can see"
		);
	}

	#[test]
	fn a_file_of_another_shape_is_read_as_nothing_remembered() {
		let dir = tree("another-shape");

		fs::write(dir.join(FILE), "colby stamps 0\nmeshes/box.cmesh\n\t1\tmeshes/box.obj\n")
			.expect("the file is written");

		assert!(
			!Stamps::read(&dir).matches("meshes/box.cmesh", &[input("meshes/box.obj", 1, 8)]),
			"a file this build does not know how to read is a tree it builds again"
		);
	}

	#[test]
	fn a_line_that_cannot_be_read_takes_its_own_output_and_no_other() {
		let dir = tree("torn");

		fs::write(
			dir.join(FILE),
			"colby stamps 1\nmeshes/box.cmesh\n\tnot a \
			 number\t8\tmeshes/box.obj\nui/hud.clua\n\t7\t8\tui/hud.lua\n",
		)
		.expect("the file is written");

		let read = Stamps::read(&dir);

		assert!(
			!read.matches("meshes/box.cmesh", &[input("meshes/box.obj", 1, 8)]),
			"the torn one is built again"
		);
		assert!(
			read.matches("ui/hud.clua", &[input("ui/hud.lua", 7, 8)]),
			"and the next one is not"
		);
	}

	#[test]
	fn a_pass_that_moved_nothing_writes_no_file() {
		let dir = tree("quiet");

		Stamps::read(&dir)
			.write(&dir)
			.expect("a pass with nothing to say still succeeds");

		assert!(
			!dir.join(FILE).exists(),
			"a settled tree costs the disk nothing, four times a second"
		);
	}

	#[test]
	fn an_output_nobody_wants_any_more_is_forgotten() {
		let dir = tree("forgotten");
		let mut stamps = Stamps::default();

		stamps.set("meshes/box.cmesh".to_owned(), vec![input("meshes/box.obj", 1, 8)]);
		stamps.set("meshes/gone.cmesh".to_owned(), vec![input("meshes/gone.obj", 2, 8)]);
		stamps.keep(&BTreeSet::from(["meshes/box.cmesh".to_owned()]));
		stamps.write(&dir).expect("the file is written");

		let read = Stamps::read(&dir);

		assert!(
			read.matches("meshes/box.cmesh", &[input("meshes/box.obj", 1, 8)]),
			"the one kept"
		);
		assert!(
			!read.matches("meshes/gone.cmesh", &[input("meshes/gone.obj", 2, 8)]),
			"and the one whose source is gone does not sit in the file forever"
		);
	}

	#[test]
	fn a_name_that_would_not_read_back_is_left_out_rather_than_written_wrong() {
		let dir = tree("unwritable");
		let mut stamps = Stamps::default();

		stamps.set("meshes/one\nline.cmesh".to_owned(), vec![input(
			"meshes/one\nline.obj",
			1,
			8,
		)]);
		stamps.set("meshes/box.cmesh".to_owned(), vec![input("meshes/box.obj", 2, 8)]);
		stamps.write(&dir).expect("the file is written");

		let read = Stamps::read(&dir);

		assert!(
			!read.matches("meshes/one\nline.cmesh", &[input("meshes/one\nline.obj", 1, 8)]),
			"a name with a line break in it costs its own output a rebuild"
		);
		assert!(
			read.matches("meshes/box.cmesh", &[input("meshes/box.obj", 2, 8)]),
			"and costs the file beside it nothing"
		);
	}

	#[test]
	fn a_file_is_read_under_its_name_in_the_tree_and_at_its_own_time() {
		let dir = tree("read");
		let path = dir.join("meshes").join("box.obj");

		fs::create_dir_all(dir.join("meshes")).expect("the directory is made");
		fs::write(&path, "v 0 0 0\n").expect("the file is written");

		let first = Input::of(&path, &dir);

		assert_eq!(first.path, "meshes/box.obj", "named by where it is, with forward slashes");
		assert!(first.found.is_some(), "and it has a time and a length");

		File::options()
			.write(true)
			.open(&path)
			.expect("the file opens")
			.set_modified(SystemTime::now() - Duration::from_mins(1))
			.expect("and takes a time");

		assert_ne!(
			Input::of(&path, &dir),
			first,
			"a file put back an hour older is not the same"
		);
	}
}
