//! What was done to the world, and how to undo it.
//!
//! **A record is the world before and the world after**, written down with
//! [`scene::capture`], and undoing one is putting the world before back with
//! `scene::restore` - which rebuilds every table slot for slot, so a handle a
//! panel is holding points at the same thing afterwards and the selection
//! survives an undo without anybody doing anything about it. Snapshots of the
//! whole world rather than commands that know how to revert themselves,
//! because a restore is exact by construction, a description of a world this
//! size is tens of kilobytes, and what the command shape buys - a stack that
//! reads as a list of verbs - a label on each record buys as well. The piece
//! that cuts a contraption out would not do: it is cut by bodies, and an
//! entity with no body under it, or the stage, is not in one.
//!
//! **A record is a gesture, not a frame.** A drag writes the world every
//! frame it lasts and is one thing to undo; so is a number typed into a
//! field. The rule is the one an inspector's merge mode spells elsewhere, in
//! one sentence: a record opens at the first write and closes on the first
//! frame in which nothing writes, and everything written in between is one
//! record. Nothing here counts frames - the editor calls [`History::settle`]
//! once a frame, and that is the clock.
//!
//! **Putting a world back is the runner's to do**, not this crate's: a
//! restore leaves the solver holding caches of a world that is gone, and the
//! solver is out of reach from here. So an undo hands back the description to
//! put back, the editor carries it out of the frame - @ref
//! [`Frame`](crate::Frame) - and the frame loop does what a stopped play
//! does, restore and forget, in the one place that already does it.
//!
//! **While the world is played, nothing is recorded**: the game writes the
//! world every step, and a record of a world it has since rewritten is a
//! record of nothing. The records that exist are dropped when play starts,
//! because the world that comes back when play stops is the one play started
//! from and the records were made against the one before that - which may be
//! the same world, and may not, and a stack that is sometimes wrong is worse
//! than an empty one.

use colby_core::abi::{
	World,
	scene::{self, SceneData},
};

/// How many records are kept before the oldest is dropped.
///
/// A record is two descriptions of a world, tens of kilobytes each for a
/// world of a few hundred things; sixty-four of them is a few megabytes at
/// most, and more steps back than anybody counts.
pub(crate) const LIMIT: usize = 64;

/// One thing that was done: the world on either side of it.
#[derive(Clone, Debug)]
struct Record {
	/// What was done, in a word or two, for a button.
	label: String,

	/// The world before it.
	before: Box<SceneData>,

	/// The world after it.
	after: Box<SceneData>,
}

/// A record that has begun and not ended: the gesture in progress.
#[derive(Clone, Debug)]
struct Open {
	label: String,
	before: Box<SceneData>,
}

/// Everything that was done, in order, and how far back it has been undone.
#[derive(Clone, Debug)]
pub(crate) struct History {
	records: Vec<Record>,

	/// How many of the records stand: everything before this is undoable and
	/// everything from it on is redoable.
	done: usize,

	/// The gesture in progress, if one is.
	open: Option<Open>,

	/// Whether anything wrote this frame, which is what keeps an open record
	/// open across the frames of a drag.
	written: bool,

	/// How many records are kept.
	limit: usize,
}

impl Default for History {
	fn default() -> Self { Self::new(LIMIT) }
}

impl History {
	/// A history with nothing in it.
	///
	/// @param limit - how many records to keep; the oldest goes past it
	pub(crate) const fn new(limit: usize) -> Self {
		Self {
			records: Vec::new(),
			done: 0,
			open: None,
			written: false,
			limit,
		}
	}

	/// Says that something is about to be written, so that the world as it
	/// stands is what an undo goes back to.
	///
	/// Called by every writer, before it writes, every frame it writes: the
	/// first call opens a record and every call keeps it open, @ref the
	/// module comment. Nothing is written down while the world is being
	/// played rather than edited.
	///
	/// @param label - what is being done, for the first call of a gesture;
	/// later calls in the same gesture do not rename it
	/// @param world - the world as it stands, before the write
	pub(crate) fn begin(&mut self, label: &str, world: &World) {
		if !world.editing {
			return;
		}

		self.written = true;

		if self.open.is_none() {
			self.open = Some(Open {
				label: label.to_owned(),
				before: Box::new(scene::capture(world)),
			});
		}
	}

	/// Ends the frame: a gesture nobody wrote to this frame is over.
	///
	/// Called once a frame by the editor, after every panel has drawn, which
	/// is what makes a record a gesture rather than a frame.
	///
	/// @param world - the world as it stands, which is what the record ends
	/// with
	/// @return whether a record was written down
	pub(crate) fn settle(&mut self, world: &World) -> bool {
		let written = std::mem::take(&mut self.written);

		if written {
			return false;
		}

		self.close(world)
	}

	/// Ends the gesture in progress, if there is one.
	///
	/// @return whether it came to anything: a gesture that left the world
	/// exactly as it found it is not worth a step back
	fn close(&mut self, world: &World) -> bool {
		let Some(open) = self.open.take() else {
			return false;
		};

		let after = scene::capture(world);
		if after == *open.before {
			return false;
		}

		// whatever had been undone is gone: a new thing was done instead of
		// it, and there is no second timeline to keep.
		self.records.truncate(self.done);
		self.records.push(Record {
			label: open.label,
			before: open.before,
			after: Box::new(after),
		});

		if self.records.len() > self.limit {
			self.records.remove(0);
		}

		self.done = self.records.len();

		true
	}

	/// Takes one step back.
	///
	/// A gesture in progress is ended first, so that what comes back is the
	/// world before it rather than the world before the one before.
	///
	/// @param world - the world as it stands
	/// @return the world to put back, or nothing if there is nothing to undo
	pub(crate) fn undo(&mut self, world: &World) -> Option<Box<SceneData>> {
		self.close(world);
		self.written = false;

		if self.done == 0 {
			return None;
		}

		self.done -= 1;

		self.records
			.get(self.done)
			.map(|record| record.before.clone())
	}

	/// Takes one step forward again.
	///
	/// @return the world to put back, or nothing if there is nothing to redo
	pub(crate) fn redo(&mut self) -> Option<Box<SceneData>> {
		let record = self.records.get(self.done)?;
		self.done += 1;

		Some(record.after.clone())
	}

	/// What an undo would undo, for a button.
	pub(crate) fn undoable(&self) -> Option<&str> {
		self.done
			.checked_sub(1)
			.and_then(|index| self.records.get(index))
			.map(|record| record.label.as_str())
	}

	/// What a redo would do again, for a button.
	pub(crate) fn redoable(&self) -> Option<&str> {
		self.records
			.get(self.done)
			.map(|record| record.label.as_str())
	}

	/// Drops everything: the records and the gesture in progress.
	pub(crate) fn clear(&mut self) {
		self.records.clear();
		self.done = 0;
		self.open = None;
		self.written = false;
	}

	/// How many records there are, undone ones included.
	#[cfg(test)]
	pub(crate) fn len(&self) -> usize { self.records.len() }
}

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{EntityId, Transform},
		glam::Vec3,
	};

	use super::*;

	/// A world being edited, with one named thing in it.
	fn edited() -> (World, EntityId) {
		let mut world = World::new();
		world.editing = true;
		let crate_ = world.entities.spawn_at(Transform::at(Vec3::Y));
		world.entities.set_name(crate_, "crate");

		(world, crate_)
	}

	/// Where the thing is.
	fn at(world: &World, id: EntityId) -> Vec3 {
		world
			.entities
			.transform(id)
			.map(|it| it.position)
			.unwrap_or(Vec3::NAN)
	}

	/// Moves the thing, the way a writer does: a word first, then the write.
	fn shove(history: &mut History, world: &mut World, id: EntityId, to: Vec3) {
		history.begin("move", world);
		if let Some(transform) = world.entities.transform_mut(id) {
			transform.position = to;
		}
	}

	#[test]
	fn a_gesture_is_one_record_however_many_frames_it_took() {
		let (mut world, id) = edited();
		let mut history = History::default();

		// three frames of a drag, each writing once
		for step in [1.0, 2.0, 3.0] {
			shove(&mut history, &mut world, id, Vec3::X * step);
			assert!(!history.settle(&world), "a frame that wrote keeps the gesture open");
		}

		assert!(history.settle(&world), "and the first quiet frame closes it");
		assert_eq!(history.len(), 1, "as one record");
		assert_eq!(history.undoable(), Some("move"));
		assert_eq!(
			history
				.undo(&world)
				.map(|before| before.things[0].transform.position),
			Some(Vec3::Y),
			"which goes back to before the drag began"
		);
	}

	#[test]
	fn the_first_word_of_a_gesture_names_it() {
		let (mut world, id) = edited();
		let mut history = History::default();

		history.begin("move", &world);
		shove(&mut history, &mut world, id, Vec3::X);
		history.begin("rename", &world);
		history.settle(&world);
		history.settle(&world);

		assert_eq!(history.undoable(), Some("move"), "the word that opened it");
	}

	#[test]
	fn a_gesture_that_changed_nothing_is_not_a_record() {
		let (world, _) = edited();
		let mut history = History::default();

		history.begin("nothing", &world);
		history.settle(&world);
		assert!(!history.settle(&world), "the world is as it was");
		assert_eq!(history.len(), 0);
		assert_eq!(history.undoable(), None);
	}

	#[test]
	fn undo_goes_back_and_redo_goes_forward_and_a_new_record_drops_the_redo() {
		let (mut world, id) = edited();
		let mut history = History::default();

		shove(&mut history, &mut world, id, Vec3::X);
		history.settle(&world);
		history.settle(&world);
		shove(&mut history, &mut world, id, Vec3::Z);
		history.settle(&world);
		history.settle(&world);
		assert_eq!(history.len(), 2);

		let back = history.undo(&world).expect("a step back");
		assert_eq!(back.things[0].transform.position, Vec3::X, "the world after the first move");
		assert_eq!(history.undoable(), Some("move"), "one more to undo");
		assert_eq!(history.redoable(), Some("move"), "and one to redo");

		let forward = history.redo().expect("a step forward");
		assert_eq!(forward.things[0].transform.position, Vec3::Z);
		assert!(history.redo().is_none(), "and nothing past the end");

		history.undo(&world);
		history.undo(&world);
		assert!(history.undo(&world).is_none(), "nor before the beginning");

		// something new after two undos, and the two redos are gone
		shove(&mut history, &mut world, id, Vec3::NEG_X);
		history.settle(&world);
		history.settle(&world);
		assert_eq!(history.len(), 1, "the undone records were dropped");
		assert!(history.redoable().is_none());
	}

	#[test]
	fn the_oldest_record_goes_past_the_limit() {
		let (mut world, id) = edited();
		let mut history = History::new(3);

		for step in [1.0, 2.0, 3.0, 4.0, 5.0] {
			shove(&mut history, &mut world, id, Vec3::X * step);
			history.settle(&world);
			history.settle(&world);
		}

		assert_eq!(history.len(), 3, "three kept");
		let mut oldest = None;
		while let Some(before) = history.undo(&world) {
			oldest = Some(before.things[0].transform.position);
		}
		assert_eq!(oldest, Some(Vec3::X * 2.0), "and the one before the third is the floor");
	}

	#[test]
	fn a_world_being_played_records_nothing() {
		let (mut world, id) = edited();
		world.editing = false;
		let mut history = History::default();

		shove(&mut history, &mut world, id, Vec3::X);
		history.settle(&world);
		history.settle(&world);

		assert_eq!(history.len(), 0, "the game writes the world; nothing here does");
	}

	#[test]
	fn an_undo_mid_gesture_ends_the_gesture_first() {
		let (mut world, id) = edited();
		let mut history = History::default();

		shove(&mut history, &mut world, id, Vec3::X);
		let back = history
			.undo(&world)
			.expect("the gesture so far is a record");
		assert_eq!(back.things[0].transform.position, Vec3::Y, "back to before it");
		assert_eq!(history.len(), 1);
	}

	#[test]
	fn clearing_drops_the_records_and_the_gesture() {
		let (mut world, id) = edited();
		let mut history = History::default();

		shove(&mut history, &mut world, id, Vec3::X);
		history.settle(&world);
		history.settle(&world);
		shove(&mut history, &mut world, id, Vec3::Z);
		history.clear();
		history.settle(&world);
		history.settle(&world);

		assert_eq!(history.len(), 0);
		assert!(history.undo(&world).is_none());
	}

	#[test]
	fn what_an_undo_hands_back_puts_the_world_back_when_restored() {
		// the whole loop, with the runner's half done here: the description an
		// undo hands back goes through the same restore a stopped play does.
		let (mut world, id) = edited();
		let mut history = History::default();

		shove(&mut history, &mut world, id, Vec3::new(3.0, 4.0, 5.0));
		history.settle(&world);
		history.settle(&world);

		let before = history.undo(&world).expect("a step back");
		scene::restore(&mut world, &before).expect("the world it came from takes it");
		assert_eq!(at(&world, id), Vec3::Y, "the move is undone");
		assert!(
			world.entities.alive(id),
			"and the handle still resolves: same slot, same generation"
		);
		assert_eq!(world.entities.name(id), "crate");

		let after = history.redo().expect("a step forward");
		scene::restore(&mut world, &after).expect("and so does this one");
		assert_eq!(at(&world, id), Vec3::new(3.0, 4.0, 5.0), "the move is done again");
	}
}
