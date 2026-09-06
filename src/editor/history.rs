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
//! **While the world is played, nothing is recorded** by a panel: the game
//! writes the world every step, and a record of a world it has since
//! rewritten is a record of nothing. [`History::begin`] refuses while a world
//! is played, which is the whole of that rule.
//!
//! **The play itself is one record**, and the one thing here that is not a
//! gesture. [`History::hold`] opens it as play starts and
//! [`History::release`] closes it as play stops, whatever the thousands of
//! frames in between do; a stop that puts the world back closes it with the
//! world it opened with, so nothing is written down, and a stop that keeps
//! what the game did leaves exactly one step to go back over. The records
//! made before the play are kept either way: the first case leaves the world
//! they were made against, and the second leaves a record of the difference
//! standing in front of them.

use colby_core::{
	abi::{
		World,
		scene::{self, SceneData},
	},
	trace,
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

	/// Whether the open record is a play rather than a gesture, and is
	/// therefore kept open until whoever started it says otherwise.
	holding: bool,

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
			holding: false,
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

	/// Opens a record that outlasts the frame, and the frames after it.
	///
	/// A gesture is over when nobody writes to it, @ref
	/// [`settle`](Self::settle), which is the right rule for a drag and the
	/// wrong one for a play: a game running is thousands of frames in which
	/// the editor writes nothing. This opens a record and holds it open until
	/// [`release`](Self::release), whatever happens in between.
	///
	/// The world is not asked whether it is being edited. What this is for is
	/// the moment before it stops being edited, and a caller who has reached
	/// for it means it.
	///
	/// @param label - what the stretch is, for the button that undoes it
	/// @param world - the world as it stands, which is what an undo goes
	/// back to
	pub(crate) fn hold(&mut self, label: &str, world: &World) {
		// a held record already open is not replaced: two starts without a
		// stop between them would otherwise throw away the older world, which
		// is the one worth going back to
		if self.holding {
			return;
		}

		self.close(world);
		self.holding = true;
		self.open = Some(Open {
			label: label.to_owned(),
			before: Box::new(scene::capture(world)),
		});
	}

	/// Ends a held record.
	///
	/// @param world - the world as it stands, which is what the record ends
	/// with
	/// @return whether it came to anything: a play that left the world as it
	/// found it - which is what a stop that puts the world back does - is not
	/// a step to go back over
	pub(crate) fn release(&mut self, world: &World) -> bool {
		if !self.holding {
			return false;
		}

		self.holding = false;
		self.written = false;

		self.close(world)
	}

	/// Whether a stretch is being held open.
	pub(crate) const fn holding(&self) -> bool { self.holding }

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

		// a play is not a gesture and does not end because a frame went by
		// without a panel writing to the world
		if written || self.holding {
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

		// the clock aside, @ref `SceneData::same_world`: `time` and `steps`
		// advance in every step and a step runs while a world is edited as
		// well as while it is played, so a plain comparison calls every
		// gesture a change and every play a step to go back over - which is
		// exactly what it did until a window was driven and said so.
		if open.before.same_world(&after) {
			return false;
		}

		// which part moved, for the next time that question comes up
		trace!(
			label = open.label,
			stage = after.stage == open.before.stage,
			things = after.things == open.before.things,
			solids = after.solids == open.before.solids,
			links = after.links == open.before.links,
			posed = after.posed == open.before.posed,
			arena = after.arena == open.before.arena,
			"a record is being written down"
		);

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
	fn a_held_record_outlasts_every_frame_that_writes_nothing_to_it() {
		let (mut world, id) = edited();
		let mut history = History::default();

		history.hold("play", &world);
		// a thousand frames of a game running, in which no panel writes
		for _ in 0..5 {
			assert!(!history.settle(&world), "a settle does not end a play");
			assert!(history.holding());
		}

		// the game moved it, and the stop kept where it got to
		if let Some(transform) = world.entities.transform_mut(id) {
			transform.position = Vec3::X * 7.0;
		}

		assert!(history.release(&world), "the play is a step to go back over");
		assert!(!history.holding());
		assert_eq!(history.undoable(), Some("play"));
		assert_eq!(
			history
				.undo(&world)
				.expect("the world before it")
				.things[0]
				.transform
				.position,
			Vec3::Y,
			"back to where the play started"
		);
	}

	#[test]
	fn a_play_that_ends_where_it_began_is_not_a_step() {
		let (mut world, _) = edited();
		let mut history = History::default();

		history.hold("play", &world);
		// the clock runs while a world is edited as well as while it is
		// played, so a play that changed nothing still ends at a later
		// second than it began. A window said so before this test did.
		world.time += 4.5;
		world.steps += 270;

		assert!(!history.release(&world), "a stop that put the world back changed nothing");
		assert_eq!(history.len(), 0);
		assert!(!history.release(&world), "and releasing again is nothing at all");
	}

	#[test]
	fn a_gesture_that_left_the_world_as_it_found_it_is_not_a_step_either() {
		let (mut world, id) = edited();
		let mut history = History::default();

		// a drag that went nowhere: the writer wrote, and wrote the same
		// value, over frames in which the clock kept going
		for _ in 0..3 {
			shove(&mut history, &mut world, id, Vec3::Y);
			world.time += 0.016;
			world.steps += 1;
		}
		history.settle(&world);
		history.settle(&world);

		assert_eq!(history.len(), 0, "nothing moved, so there is nothing to go back over");
	}

	#[test]
	fn a_play_does_not_throw_away_what_was_done_before_it() {
		let (mut world, id) = edited();
		let mut history = History::default();
		shove(&mut history, &mut world, id, Vec3::X);
		history.settle(&world);
		history.settle(&world);
		assert_eq!(history.len(), 1);

		history.hold("play", &world);
		if let Some(transform) = world.entities.transform_mut(id) {
			transform.position = Vec3::Z;
		}
		assert!(history.release(&world));

		assert_eq!(history.len(), 2, "the play stands in front of the move");
		assert_eq!(history.undoable(), Some("play"));
	}

	#[test]
	fn a_second_hold_without_a_release_keeps_the_older_world() {
		let (mut world, id) = edited();
		let mut history = History::default();

		history.hold("play", &world);
		if let Some(transform) = world.entities.transform_mut(id) {
			transform.position = Vec3::X;
		}
		history.hold("play", &world);
		if let Some(transform) = world.entities.transform_mut(id) {
			transform.position = Vec3::Z;
		}
		assert!(history.release(&world));

		assert_eq!(
			history.undo(&world).expect("a world").things[0]
				.transform
				.position,
			Vec3::Y,
			"the world the first hold saw, not the second"
		);
	}

	#[test]
	fn a_gesture_open_when_a_play_starts_is_closed_first() {
		let (mut world, id) = edited();
		let mut history = History::default();
		shove(&mut history, &mut world, id, Vec3::X);

		history.hold("play", &world);

		assert_eq!(history.len(), 1, "the move became a record of its own");
		assert_eq!(history.undoable(), Some("move"));
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
