//! What one item in the reliable ring is, and how something far too long for
//! one item gets across anyway.
//!
//! ```text
//!    0  kind    u8    a console line, part of a world, or joints that are gone
//!    1  index   u16   which piece of it this is, counting from nil
//!    3  pieces  u16   how many pieces it was cut into, one meaning whole
//!    5  the piece, up to [`MAX_PIECE`] bytes of it
//! ```
//!
//! **A console line is a parcel of one piece**, which is the whole of what it
//! used to be: the ring carried a line and nothing else, and everything that
//! had to cross reliably had to be sayable as one. A world cannot be. What
//! appears during play is a description of some bodies, at the numbers the host
//! keeps them at, and one small prop of it is already half a kilobyte - so the
//! thing that carries it is the ring, cut into pieces the way
//! [`fragment`](crate::fragment) cuts a message into datagrams, and for the
//! same reason.
//!
//! **The cutting is easier here than it is a layer down, because the stream
//! below is in order and loses nothing.** A datagram may arrive before the one
//! in front of it and a reliable item may not, so there is no map of which
//! pieces have turned up and no room taken for the ones that have not: a piece
//! is appended to what came before it or the peer is not making sense. What is
//! kept from the layer below is the two rules that stop a peer describing one
//! parcel two ways - **the piece count is on the wire**, and **every piece but
//! the last is exactly full**.
//!
//! **A whole parcel may sit between the pieces of a long one.** A line somebody
//! types while a world is crossing does not have to wait for it, and does not
//! disturb it: an item that says it is one piece of one is handed over where it
//! stands. Only a parcel of several touches what is being put back together,
//! and a second one starting before the first has finished is a peer that is
//! broken.
//!
//! **A parcel is at most a ringful.** [`MAX_PIECES`] is the ring's own depth,
//! so the longest parcel there is fills the ring exactly and nothing else can
//! be queued until the far end has taken some of it. That is a real cost and it
//! is the honest ceiling: a parcel longer than what can be waiting at once
//! could never be put in the ring in the first place.

use colby_core::{Result, err};

use crate::{
	packet::{u16_at, u32_at},
	reliable::{MAX_ITEM, MAX_ITEMS, Reliable},
};

/// How many bytes come before a piece.
pub const HEAD: usize = 5;

/// The most of a parcel one item can carry.
pub const MAX_PIECE: usize = MAX_ITEM - HEAD;

/// The most pieces one parcel may be cut into.
///
/// The ring's own depth, because a parcel is only worth cutting if every piece
/// of it can be waiting at once: a longer one could not be queued whole, and
/// half a parcel in the ring is worse than none.
pub const MAX_PIECES: u16 = 64;

/// The longest parcel there is, in bytes.
#[expect(
	clippy::as_conversions,
	reason = "a const, where try_from is not available"
)]
pub const MAX_PARCEL: usize = MAX_PIECES as usize * MAX_PIECE;

#[expect(
	clippy::as_conversions,
	reason = "a const, where try_from is not available"
)]
const _: () = assert!(
	MAX_PIECES as u32 == MAX_ITEMS,
	"a parcel is at most a ringful, and the two numbers are written apart"
);

const KIND_AT: usize = 0;
const INDEX_AT: usize = 1;
const PIECES_AT: usize = 3;

/// What a parcel is, as the byte at the head of every piece of it says.
///
/// A number rather than a length-prefixed name, and it is on *every* piece
/// rather than only the first: a piece that disagrees with the parcel it claims
/// to belong to is then a thing this side can notice, and it costs one byte in
/// a thousand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
	/// A console line, the way every reliable item once was.
	Line,

	/// A description of some part of a world, as a compiled scene has it.
	///
	/// Nothing in this crate reads one. What it is for is the message that
	/// says a thing has appeared: the far end has slots this end has bodies
	/// in, and the only thing that can say what a body *is* is the whole
	/// description of it.
	Scene,

	/// Joints that are no longer there.
	///
	/// **The one removal this wire has to carry.** A body that has gone is
	/// already said by a snapshot - a slot that emptied is written as such -
	/// and a joint is in no snapshot at all, so nothing else could ever say
	/// one has been undone. A physgun makes and unmakes a joint every time
	/// somebody picks a prop up, so a wire that carried the making and not the
	/// unmaking would weld a player to everything it ever touched.
	Untied,
}

/// One joint that is no longer there.
///
/// **Both halves, and not the slot alone.** A slot is reused, and a removal
/// that named only the slot would take away whatever is in it now if the two
/// ends were even a moment out of step.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Untied {
	/// Which slot it was in.
	pub slot: u16,

	/// Which occupant of that slot it was.
	pub generation: u32,
}

impl Kind {
	/// The byte that stands for it.
	#[must_use]
	pub const fn byte(self) -> u8 {
		match self {
			| Self::Line => 1,
			| Self::Scene => 2,
			| Self::Untied => 3,
		}
	}

	/// Which kind a byte stands for, if it stands for one.
	///
	/// @param byte - what was at the head of a piece
	#[must_use]
	pub const fn of(byte: u8) -> Option<Self> {
		match byte {
			| 1 => Some(Self::Line),
			| 2 => Some(Self::Scene),
			| 3 => Some(Self::Untied),
			| _ => None,
		}
	}
}

/// One whole parcel, once every piece of it has arrived.
///
/// The kinds are apart rather than a kind beside a bag of bytes, because what
/// makes a line a line is a rule about its contents and this is where that rule
/// is kept. A caller holding one of these has been told the text is text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Parcel {
	/// A console line, checked to be one.
	Line(String),

	/// A description of some part of a world, for whoever knows how to read
	/// one.
	Scene(Vec<u8>),

	/// The joints that have gone, checked to be a list of them.
	Untied(Vec<Untied>),
}

impl Parcel {
	/// Which kind it is.
	#[must_use]
	pub const fn kind(&self) -> Kind {
		match *self {
			| Self::Line(_) => Kind::Line,
			| Self::Scene(_) => Kind::Scene,
			| Self::Untied(_) => Kind::Untied,
		}
	}
}

/// How many pieces a parcel of this length takes.
///
/// @param length - the whole parcel, in bytes
/// @return the count, or an error when there is nothing to send or far too
/// much of it
pub fn pieces(length: usize) -> Result<u16> {
	if length == 0 {
		return Err(err!(Network("an empty parcel says nothing")));
	}

	if length > MAX_PARCEL {
		return Err(err!(Network(
			"a parcel of {length} bytes is past the {MAX_PARCEL} one may be"
		)));
	}

	u16::try_from(length.div_ceil(MAX_PIECE))
		.map_err(|_| err!(Network("a parcel of {length} bytes is more pieces than there are")))
}

/// Cuts a parcel into pieces and puts every one of them in the ring, or none.
///
/// **All or nothing, and that is the whole reason this is not a loop at the
/// call site.** The pieces of a parcel only mean anything together: a ring that
/// took the front half and refused the rest would leave the far end holding
/// something it can never finish, and the next parcel behind it would be
/// refused for interrupting one.
///
/// @param ring - the reliable stream to put it in
/// @param kind - what the bytes are
/// @param bytes - the whole parcel
/// @return an error when the parcel is not one, or when the ring has no room
/// for the whole of it
pub fn post(ring: &mut Reliable, kind: Kind, bytes: &[u8]) -> Result {
	let count = pieces(bytes.len())?;

	// the rule about what a kind's bytes are, checked before anything is
	// queued rather than after half of it has been. @ref `text` and `untied`,
	// which are the same rules read from the wire.
	match kind {
		| Kind::Line => {
			text(bytes)?;
		},
		| Kind::Untied => {
			untied(bytes)?;
		},
		| Kind::Scene => {},
	}

	let room = ring.room();

	if u32::from(count) > room {
		return Err(err!(Network(
			"a parcel of {count} pieces needs more room than the {room} the ring has left"
		)));
	}

	if u32::MAX - ring.sent() < u32::from(count) {
		return Err(err!(Network("the reliable numbering has run out")));
	}

	let mut item = [0_u8; MAX_ITEM];

	item[KIND_AT] = kind.byte();
	item[PIECES_AT..HEAD].copy_from_slice(&count.to_le_bytes());

	for index in 0..count {
		let at = usize::from(index) * MAX_PIECE;
		let piece = &bytes[at..bytes.len().min(at + MAX_PIECE)];

		item[INDEX_AT..PIECES_AT].copy_from_slice(&index.to_le_bytes());
		item[HEAD..HEAD + piece.len()].copy_from_slice(piece);
		// @note: nothing above lets this fail - the room and the numbering are
		// both checked, a piece is never empty and never past the ceiling - and
		// it is written as a question rather than as an assumption because the
		// far end refuses half a parcel rather than misreading one, so being
		// wrong here costs a conversation and not a world.
		ring.queue(&item[..HEAD + piece.len()])?;
	}

	Ok(())
}

/// The pieces of one parcel, being put back together.
///
/// One of these per conversation, beside the ring it reads from. It holds
/// whatever the parcel being built has so far and nothing else: the room it
/// takes is in proportion to what has actually arrived rather than to a count
/// somebody sent, which is the promise every other reader on this wire makes.
#[derive(Clone, Debug, Default)]
pub struct Pieces {
	kind: u8,
	pieces: u16,
	next: u16,
	bytes: Vec<u8>,
}

impl Pieces {
	/// Nothing being put together.
	#[must_use]
	pub fn new() -> Self { Self::default() }

	/// Throws away whatever was half built.
	///
	/// For a conversation that has ended: the pieces of a parcel belong to the
	/// stream that was carrying them, and the numbering of that stream is about
	/// to start again. @ref [`Reliable::forget`].
	pub fn forget(&mut self) {
		self.kind = 0;
		self.pieces = 0;
		self.next = 0;
		self.bytes.clear();
	}

	/// Whether a parcel is part way here.
	#[must_use]
	pub const fn building(&self) -> bool { self.pieces != 0 }

	/// Takes one item out of the ring.
	///
	/// @param item - the bytes of one reliable item, head and all
	/// @return the parcel, once the last piece of it has arrived
	pub fn take(&mut self, item: &[u8]) -> Result<Option<Parcel>> {
		let length = item.len();

		if length < HEAD {
			return Err(err!(Network("a reliable item of {length} bytes has no head in it")));
		}

		let kind = item[KIND_AT];
		let index = u16_at(item, INDEX_AT);
		let count = u16_at(item, PIECES_AT);

		if Kind::of(kind).is_none() {
			return Err(err!(Network("a parcel of a kind number {kind} that means nothing")));
		}

		// @note: the first of the three cannot be observed on its own, and it
		// is kept because it is the one that says what the rule is. A piece
		// index is never below nil, so a count of nil already fails the third,
		// and no mutation of this clause alone can break a test. The datagram
		// head says the same thing about the same three.
		if count == 0 || count > MAX_PIECES || index >= count {
			return Err(err!(Network(
				"a parcel that says it is piece {index} of {count}, which cannot be"
			)));
		}

		let piece = &item[HEAD..];
		let last = index + 1 == count;

		if piece.is_empty() {
			return Err(err!(Network("a piece of a parcel with nothing in it")));
		}

		// the rule the datagram head keeps for the same reason: without it a
		// peer could describe one parcel at two different lengths, and the two
		// ends would disagree about where a piece belongs.
		if !last && piece.len() != MAX_PIECE {
			return Err(err!(Network(
				"a piece that is not the last one is {} bytes rather than {MAX_PIECE}",
				piece.len()
			)));
		}

		// a whole parcel is handed over where it stands and touches nothing.
		// That is what lets a console line cross while a world is still
		// arriving, which is the ordinary case rather than a clever one.
		if count == 1 {
			return whole(kind, piece).map(Some);
		}

		if index == 0 {
			if self.building() {
				return Err(err!(Network(
					"a peer began a parcel with {} pieces of another one still owed",
					self.pieces - self.next
				)));
			}

			self.kind = kind;
			self.pieces = count;
			self.next = 0;
			self.bytes.clear();
		} else if self.kind != kind || self.pieces != count || self.next != index {
			return Err(err!(Network(
				"a parcel's piece {index} of {count} does not follow what came before it"
			)));
		}

		self.bytes.extend_from_slice(piece);
		self.next += 1;

		if !last {
			return Ok(None);
		}

		self.pieces = 0;
		// **left where it is rather than taken out**, so that emptying it is
		// the business of the piece that begins the next parcel and of nothing
		// else. Taking it would put the buffer back empty and make the line up
		// there that says so unobservable - and it would give up the room it
		// has already grown to, which is the same room the next world of the
		// same size wants.
		whole(kind, &self.bytes).map(Some)
	}
}

/// A parcel, out of the kind byte and the bytes that were under it.
fn whole(kind: u8, bytes: &[u8]) -> Result<Parcel> {
	match Kind::of(kind) {
		| Some(Kind::Line) => Ok(Parcel::Line(text(bytes)?.to_owned())),
		| Some(Kind::Scene) => Ok(Parcel::Scene(bytes.to_vec())),
		| Some(Kind::Untied) => Ok(Parcel::Untied(untied(bytes)?)),
		| None => Err(err!(Network("a parcel of a kind number {kind} that means nothing"))),
	}
}

/// How many joints one parcel may say have gone.
///
/// A hundred and ninety-four bytes at that count, which is one item, and more
/// joints than anything has ever undone in a twentieth of a second. What is
/// past it waits for the next one.
pub const MAX_UNTIED: usize = 32;

/// Bytes before the first joint in a parcel of them.
const UNTIED_HEAD: usize = 2;

/// Bytes per joint.
const UNTIED_RECORD: usize = 6;

/// Writes the joints that have gone as the bytes of a parcel.
///
/// ```text
///    0  count  u16   how many follow
///    2  per joint: u16 slot, u32 generation
/// ```
///
/// @param gone - at most [`MAX_UNTIED`] of them
/// @param out - where the bytes go, appended
/// @return an error when there are none or far too many, because a parcel of
/// nothing is not one and a count nobody could have meant is a mistake worth
/// hearing about at the end that made it
pub fn parting(gone: &[Untied], out: &mut Vec<u8>) -> Result {
	let count = gone.len();

	if count == 0 || count > MAX_UNTIED {
		return Err(err!(Network(
			"a parting of {count} joints, where one to {MAX_UNTIED} is what there may be"
		)));
	}

	out.extend_from_slice(&u16::try_from(count).unwrap_or(0).to_le_bytes());

	for joint in gone {
		out.extend_from_slice(&joint.slot.to_le_bytes());
		out.extend_from_slice(&joint.generation.to_le_bytes());
	}

	Ok(())
}

/// The joints a parcel says have gone.
///
/// The same rule [`parting`] writes, read from the wire, where the bytes were
/// chosen by somebody else.
///
/// @param bytes - the whole parcel
fn untied(bytes: &[u8]) -> Result<Vec<Untied>> {
	let length = bytes.len();

	if length < UNTIED_HEAD {
		return Err(err!(Network("a parting of {length} bytes has no head in it")));
	}

	let count = usize::from(u16_at(bytes, 0));

	if count == 0 || count > MAX_UNTIED {
		return Err(err!(Network(
			"a parting of {count} joints, where one to {MAX_UNTIED} is what there may be"
		)));
	}

	// **exactly**, rather than at least: a parting with something after it is a
	// parcel this end does not understand, and reading the front of one is
	// worse than refusing the whole.
	if length != UNTIED_HEAD + count * UNTIED_RECORD {
		return Err(err!(Network(
			"a parting of {count} joints is {length} bytes, which is not what that many are"
		)));
	}

	(0..count)
		.map(|index| {
			let at = UNTIED_HEAD + index * UNTIED_RECORD;
			let joint = Untied {
				slot: u16_at(bytes, at),
				generation: u32_at(bytes, at + 2),
			};

			// a slot nobody has ever been in. Nothing this end holds could
			// match it, so a record saying so is a peer that is confused about
			// its own table rather than one telling this end anything.
			if joint.generation == 0 {
				return Err(err!(Network("a joint said to be gone that was never in a slot")));
			}

			Ok(joint)
		})
		.collect()
}

/// The one rule a console line has, read in both directions.
///
/// A line is text and it is one line. Nothing downstream of here treats it as
/// a format string, so there is nothing else to strip - what a line break in
/// the middle would do is make one line into two on the far side, and what a
/// byte that is not text would do is make it unprintable.
///
/// @param bytes - what the parcel holds
fn text(bytes: &[u8]) -> Result<&str> {
	let line = std::str::from_utf8(bytes)
		.map_err(|reason| err!(Network("a console line that is not text: {reason}")))?;

	if line
		.bytes()
		.any(|byte| byte < 0x20 || byte == 0x7F)
	{
		return Err(err!(Network("a console line is one line and this one is not")));
	}

	Ok(line)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A parcel whose bytes are a non-linear function of where they are.
	///
	/// Squares rather than a ramp or a fill, so that two pieces put back in the
	/// wrong order, or one piece written twice, is a different answer rather
	/// than the same one.
	fn awkward(length: usize) -> Vec<u8> {
		(0..length)
			.map(|at| u8::try_from(at * at % 251).expect("a remainder below a byte"))
			.collect()
	}

	/// Everything one side queued, as the other side would read it.
	fn items(ring: &Reliable) -> Vec<Vec<u8>> {
		let (mut wire, mut out) = (Vec::new(), Vec::new());

		ring.write(&mut wire);
		Reliable::new()
			.read(&wire, &mut out)
			.expect("a block this crate wrote");
		out
	}

	/// One parcel through a ring and out the far side.
	fn across(kind: Kind, bytes: &[u8]) -> Result<Option<Parcel>> {
		let mut ring = Reliable::new();

		post(&mut ring, kind, bytes)?;

		let mut pieces = Pieces::new();
		let mut last = None;

		for item in items(&ring) {
			last = pieces.take(&item)?;
		}

		Ok(last)
	}

	/// One piece, built by hand, so that a bad one can be built.
	fn piece(kind: u8, index: u16, count: u16, bytes: &[u8]) -> Vec<u8> {
		let mut out = vec![kind];

		out.extend_from_slice(&index.to_le_bytes());
		out.extend_from_slice(&count.to_le_bytes());
		out.extend_from_slice(bytes);
		out
	}

	#[test]
	fn a_console_line_is_a_parcel_of_one_piece() {
		let mut ring = Reliable::new();

		post(&mut ring, Kind::Line, b"game.spawn crate").expect("a line");
		assert_eq!(ring.pending(), 1, "one item and no more");

		let mut pieces = Pieces::new();
		let got = pieces.take(&items(&ring)[0]).expect("a parcel");

		assert_eq!(got, Some(Parcel::Line("game.spawn crate".to_owned())));
		assert!(!pieces.building(), "and nothing is being put together");
	}

	#[test]
	fn something_too_long_for_one_item_crosses_in_pieces_and_comes_back_whole() {
		let bytes = awkward(MAX_PIECE * 3 + 17);
		let mut ring = Reliable::new();

		post(&mut ring, Kind::Scene, &bytes).expect("a world");
		assert_eq!(ring.pending(), 4, "three full pieces and a tail");
		assert_eq!(across(Kind::Scene, &bytes).expect("it crosses"), Some(Parcel::Scene(bytes)));
	}

	#[test]
	fn every_piece_but_the_last_is_exactly_full() {
		let bytes = awkward(MAX_PIECE * 2 + 5);
		let mut ring = Reliable::new();

		post(&mut ring, Kind::Scene, &bytes).expect("a world");

		let sent = items(&ring);
		let lengths: Vec<usize> = sent
			.iter()
			.map(|item| item.len() - HEAD)
			.collect();

		assert_eq!(lengths, [MAX_PIECE, MAX_PIECE, 5]);

		for (index, item) in sent.iter().enumerate() {
			assert_eq!(item[KIND_AT], Kind::Scene.byte(), "every piece says what it is");
			assert_eq!(usize::from(u16_at(item, INDEX_AT)), index);
			assert_eq!(u16_at(item, PIECES_AT), 3);
		}
	}

	#[test]
	fn a_parcel_is_cut_only_when_it_has_to_be() {
		let mut ring = Reliable::new();

		assert_eq!(pieces(1).expect("one byte"), 1);
		assert_eq!(pieces(MAX_PIECE).expect("exactly one piece"), 1);
		assert_eq!(pieces(MAX_PIECE + 1).expect("one byte over"), 2);
		assert_eq!(pieces(MAX_PARCEL).expect("the longest there is"), MAX_PIECES);

		post(&mut ring, Kind::Scene, &awkward(MAX_PIECE)).expect("a world");
		assert_eq!(ring.pending(), 1, "and exactly a piece full is one item");
	}

	#[test]
	fn a_parcel_of_nothing_or_of_far_too_much_is_refused() {
		let mut ring = Reliable::new();

		pieces(0).expect_err("nothing to say");
		pieces(MAX_PARCEL + 1).expect_err("one byte past a ringful");
		post(&mut ring, Kind::Scene, &[]).expect_err("nothing to say");
		post(&mut ring, Kind::Scene, &vec![7; MAX_PARCEL + 1]).expect_err("past a ringful");
		assert_eq!(ring.pending(), 0, "and neither of them queued a thing");
	}

	#[test]
	fn the_longest_parcel_there_is_fills_the_ring_exactly() {
		let bytes = awkward(MAX_PARCEL);
		let mut ring = Reliable::new();

		post(&mut ring, Kind::Scene, &bytes).expect("a ringful");
		assert_eq!(ring.pending(), u32::from(MAX_PIECES));
		assert_eq!(ring.room(), 0);
		post(&mut ring, Kind::Line, b"game.spawn crate").expect_err("and nothing else fits");
	}

	#[test]
	fn a_ring_without_room_for_the_whole_parcel_takes_none_of_it() {
		// the whole reason posting is not a loop at the call site. Half a
		// parcel in the stream is worse than none: the far end can never
		// finish it and refuses whatever comes next for interrupting it.
		let mut ring = Reliable::new();

		for index in 0..60 {
			ring.queue(format!("item {index}").as_bytes())
				.expect("an item");
		}

		assert_eq!(ring.room(), 4);
		post(&mut ring, Kind::Scene, &awkward(MAX_PIECE * 5))
			.expect_err("five pieces into four slots");
		assert_eq!(ring.pending(), 60, "and not one of the five was queued");
		post(&mut ring, Kind::Scene, &awkward(MAX_PIECE * 4)).expect("four fit exactly");
		assert_eq!(ring.pending(), 64);
	}

	#[test]
	fn a_line_that_is_not_one_is_refused_on_the_way_out() {
		let mut ring = Reliable::new();

		post(&mut ring, Kind::Line, b"game.spawn box\ngame.freeze box")
			.expect_err("two lines pretending to be one");
		post(&mut ring, Kind::Line, b"game.spawn\tbox").expect_err("nor one with a tab in it");
		post(&mut ring, Kind::Line, b"game.spawn\x7Fbox")
			.expect_err("nor one with a delete in it");
		post(&mut ring, Kind::Line, &[0xFF, 0xFE]).expect_err("nor bytes that are not text");
		assert_eq!(ring.pending(), 0, "and none of them was queued");
		post(&mut ring, Kind::Line, b"game.spawn box").expect("an ordinary one");

		// and the same bytes are fine as a world, because the rule is the
		// line's rather than the ring's.
		post(&mut ring, Kind::Scene, &[0xFF, 0xFE, b'\n']).expect("a world is bytes");
	}

	#[test]
	fn a_line_that_is_not_one_is_refused_on_the_way_in_as_well() {
		let mut pieces = Pieces::new();

		pieces
			.take(&piece(Kind::Line.byte(), 0, 1, b"one\ntwo"))
			.expect_err("two lines");
		pieces
			.take(&piece(Kind::Line.byte(), 0, 1, &[0xFF]))
			.expect_err("not text at all");
		assert_eq!(
			pieces
				.take(&piece(Kind::Line.byte(), 0, 1, b"quit"))
				.expect("a line"),
			Some(Parcel::Line("quit".to_owned()))
		);
	}

	#[test]
	fn a_long_line_is_checked_once_it_is_whole_rather_than_piece_by_piece() {
		// the pieces are cut at a fixed stride and a line break may land
		// anywhere, so the rule is about the parcel and not about the piece.
		let mut ring = Reliable::new();
		let mut long = "x".repeat(MAX_PIECE * 2);

		post(&mut ring, Kind::Line, long.as_bytes()).expect("a very long line");
		assert_eq!(ring.pending(), 2);

		let mut pieces = Pieces::new();
		let mut got = None;

		for item in items(&ring) {
			got = pieces.take(&item).expect("a piece");
		}

		assert_eq!(got, Some(Parcel::Line(long.clone())));

		long.insert(MAX_PIECE, '\n');
		post(&mut ring, Kind::Line, long.as_bytes())
			.expect_err("and a break anywhere in it is two lines");
	}

	#[test]
	fn a_kind_nobody_knows_is_refused() {
		let mut pieces = Pieces::new();

		for kind in [0_u8, 3, 255] {
			pieces
				.take(&piece(kind, 0, 1, b"whatever"))
				.expect_err("no such kind");
		}

		assert_eq!(Kind::of(1), Some(Kind::Line));
		assert_eq!(Kind::of(2), Some(Kind::Scene));
		assert_eq!(Kind::Line.byte(), 1);
		assert_eq!(Kind::Scene.byte(), 2);
	}

	#[test]
	fn a_piece_count_that_cannot_be_right_is_refused() {
		let mut pieces = Pieces::new();
		let scene = Kind::Scene.byte();

		pieces
			.take(&piece(scene, 0, 0, b"x"))
			.expect_err("no parcel has no pieces");
		pieces
			.take(&piece(scene, 0, MAX_PIECES + 1, &vec![0; MAX_PIECE]))
			.expect_err("more pieces than a ring holds");
		pieces
			.take(&piece(scene, 3, 3, b"x"))
			.expect_err("three of three is the fourth");
		pieces
			.take(&piece(scene, 0, 1, b""))
			.expect_err("a piece of nothing");
		// the one case where an index past the count is the *only* thing wrong
		// with a piece: a full one, of a parcel that is whole, so neither the
		// middle-piece rule nor the turn below it has anything to say. Without
		// this rule it would be handed over as a whole parcel.
		pieces
			.take(&piece(scene, 1, 1, &vec![3; MAX_PIECE]))
			.expect_err("the second piece of a parcel that has one");
		assert!(!pieces.building(), "and none of them started anything");
	}

	#[test]
	fn a_middle_piece_that_is_not_full_is_refused() {
		let mut pieces = Pieces::new();
		let scene = Kind::Scene.byte();

		pieces
			.take(&piece(scene, 0, 2, &vec![1; MAX_PIECE - 1]))
			.expect_err("a short middle piece would make the parcel two lengths at once");
		pieces
			.take(&piece(scene, 0, 2, &vec![1; MAX_PIECE]))
			.expect("a full one is fine");
		assert!(pieces.building());
	}

	#[test]
	fn a_piece_out_of_its_turn_is_refused() {
		let mut pieces = Pieces::new();
		let scene = Kind::Scene.byte();
		let full = vec![1_u8; MAX_PIECE];

		pieces
			.take(&piece(scene, 0, 3, &full))
			.expect("the first piece");
		pieces
			.take(&piece(scene, 2, 3, b"tail"))
			.expect_err("the third before the second");
		pieces
			.take(&piece(scene, 1, 4, &full))
			.expect_err("nor one from a differently cut parcel");
		pieces
			.take(&piece(Kind::Line.byte(), 1, 3, &full))
			.expect_err("nor one of another kind");
		pieces
			.take(&piece(scene, 1, 3, &full))
			.expect("and the second one goes on");
	}

	#[test]
	fn a_second_parcel_beginning_before_the_first_has_finished_is_refused() {
		let mut pieces = Pieces::new();
		let scene = Kind::Scene.byte();
		let full = vec![1_u8; MAX_PIECE];

		pieces
			.take(&piece(scene, 0, 2, &full))
			.expect("the first piece of one");

		let error = pieces
			.take(&piece(scene, 0, 2, &full))
			.expect_err("and the first piece of another");

		assert!(error.to_string().contains('1'), "with what is still owed in it: {error}");
	}

	#[test]
	fn a_whole_parcel_crosses_between_the_pieces_of_a_long_one() {
		// a line somebody types while a world is arriving does not wait for it
		// and does not disturb it, which is what keeps the console usable while
		// a big thing is on the wire.
		let mut pieces = Pieces::new();
		let scene = Kind::Scene.byte();
		let bytes = awkward(MAX_PIECE * 2 + 9);

		assert_eq!(
			pieces
				.take(&piece(scene, 0, 3, &bytes[..MAX_PIECE]))
				.expect("a piece"),
			None
		);
		assert_eq!(
			pieces
				.take(&piece(Kind::Line.byte(), 0, 1, b"net.status"))
				.expect("a line"),
			Some(Parcel::Line("net.status".to_owned())),
			"handed over where it stands"
		);
		assert!(pieces.building(), "and the world is still being put together");
		pieces
			.take(&piece(scene, 1, 3, &bytes[MAX_PIECE..MAX_PIECE * 2]))
			.expect("a piece");

		assert_eq!(
			pieces
				.take(&piece(scene, 2, 3, &bytes[MAX_PIECE * 2..]))
				.expect("the last piece"),
			Some(Parcel::Scene(bytes))
		);
	}

	#[test]
	fn two_long_parcels_in_a_row_are_two_parcels() {
		// the one thing a single crossing cannot show: what is left over from
		// the first has to be gone before the second begins, and the state that
		// says a parcel is being built has to be put down when it finishes.
		let mut pieces = Pieces::new();
		let one = awkward(MAX_PIECE + 3);
		let two: Vec<u8> = awkward(MAX_PIECE + 8)
			.into_iter()
			.map(|byte| byte ^ 0x5A)
			.collect();
		let mut got = Vec::new();

		for whole in [&one, &two] {
			for (at, part) in whole.chunks(MAX_PIECE).enumerate() {
				let index = u16::try_from(at).expect("two of them");

				let item = piece(Kind::Scene.byte(), index, 2, part);

				got.extend(pieces.take(&item).expect("a piece"));
			}
		}

		assert_eq!(got, [Parcel::Scene(one), Parcel::Scene(two)]);
		assert!(!pieces.building(), "and nothing is left half built");
	}

	#[test]
	fn what_was_half_built_is_thrown_away_with_the_conversation() {
		let mut pieces = Pieces::new();
		let scene = Kind::Scene.byte();
		let full = vec![1_u8; MAX_PIECE];

		pieces
			.take(&piece(scene, 0, 2, &full))
			.expect("the first piece");
		assert!(pieces.building());
		pieces.forget();
		assert!(!pieces.building(), "and the parcel it belonged to is gone");
		pieces
			.take(&piece(scene, 0, 2, &full))
			.expect("so a new one may begin");
	}

	#[test]
	fn an_item_with_no_head_in_it_is_refused() {
		let mut pieces = Pieces::new();

		for length in 0..HEAD {
			pieces
				.take(&vec![Kind::Line.byte(); length])
				.expect_err("shorter than a head");
		}
	}

	#[test]
	fn a_piece_is_laid_down_at_the_offsets_the_module_publishes() {
		// the offsets are the contract rather than a detail, because the thing
		// on the far end may not have been built here. Every other test in this
		// file writes and reads through the same constants and would not notice
		// two of them swapping.
		let mut ring = Reliable::new();

		post(&mut ring, Kind::Scene, &awkward(MAX_PIECE + 2)).expect("two pieces");

		let sent = items(&ring);

		assert_eq!(sent[1][..HEAD], [2, 1, 0, 2, 0], "the kind, then the index and the count");
		assert_eq!(sent[1][HEAD..], awkward(MAX_PIECE + 2)[MAX_PIECE..]);
		assert_eq!(HEAD, 5, "a byte and two shorts");
		assert_eq!(MAX_PIECE, 1019, "which is what a piece is short of an item by");
		assert_eq!(MAX_PARCEL, 65_216, "and a ringful of those is the longest parcel");
	}

	/// A handful of joints that have gone, with two different generations in
	/// them so a fixture of first occupants cannot answer for both fields.
	fn parted() -> Vec<Untied> {
		vec![Untied { slot: 0, generation: 1 }, Untied { slot: 7, generation: 4 }, Untied {
			slot: 300,
			generation: 2,
		}]
	}

	#[test]
	fn joints_that_have_gone_cross_and_come_back_as_they_went() {
		let mut ring = Reliable::new();
		let gone = parted();
		let mut bytes = Vec::new();

		parting(&gone, &mut bytes).expect("a parting");
		post(&mut ring, Kind::Untied, &bytes).expect("and it goes in the ring");
		assert_eq!(ring.pending(), 1, "which is one item");
		assert_eq!(across(Kind::Untied, &bytes).expect("it crosses"), Some(Parcel::Untied(gone)));
	}

	#[test]
	fn a_parting_is_laid_down_at_the_offsets_the_module_publishes() {
		// the offsets are the contract, because the end that reads them may
		// not have been built here. Every other test writes and reads through
		// the same two functions and would not notice the two fields swapping.
		let mut bytes = Vec::new();

		parting(&[Untied { slot: 0x0201, generation: 0x0605_0403 }], &mut bytes)
			.expect("one joint");
		assert_eq!(bytes, [1, 0, 1, 2, 3, 4, 5, 6], "a count, then a slot and a generation");
		assert_eq!(UNTIED_HEAD, 2);
		assert_eq!(UNTIED_RECORD, 6);
		assert_eq!(MAX_UNTIED, 32, "and a full parting is one hundred and ninety-four bytes");
	}

	#[test]
	fn a_parting_of_nothing_or_of_far_too_many_is_refused_both_ways() {
		let mut ring = Reliable::new();
		let mut bytes = Vec::new();

		parting(&[], &mut bytes).expect_err("a parting that says nothing is not one");
		assert!(bytes.is_empty(), "and it wrote nothing before saying so");

		let many = vec![Untied { slot: 1, generation: 1 }; MAX_UNTIED + 1];

		parting(&many, &mut bytes).expect_err("one past the ceiling");
		post(&mut ring, Kind::Untied, &[0, 0]).expect_err("nor is a count of nil sent");

		let mut full = Vec::new();

		parting(&many[..MAX_UNTIED], &mut full).expect("exactly the ceiling is fine");
		post(&mut ring, Kind::Untied, &full).expect("and it goes");

		let mut over = full.clone();

		over[..2].copy_from_slice(
			&u16::try_from(MAX_UNTIED + 1)
				.expect("small")
				.to_le_bytes(),
		);
		Pieces::new()
			.take(&piece(Kind::Untied.byte(), 0, 1, &over))
			.expect_err("a count past the ceiling on the way in as well");
	}

	#[test]
	fn a_parting_that_is_not_the_length_it_says_is_refused() {
		let mut pieces = Pieces::new();
		let untied = Kind::Untied.byte();
		let mut bytes = Vec::new();

		parting(&parted(), &mut bytes).expect("three of them");

		for length in 0..bytes.len() {
			assert!(
				pieces
					.take(&piece(untied, 0, 1, &bytes[..length]))
					.is_err(),
				"a parting cut to {length} bytes was read anyway"
			);
		}

		let mut longer = bytes.clone();

		longer.push(0);
		pieces
			.take(&piece(untied, 0, 1, &longer))
			.expect_err("and one with something after it, which this end cannot read");
		assert!(
			pieces
				.take(&piece(untied, 0, 1, &bytes))
				.expect("the whole of it")
				.is_some()
		);
	}

	#[test]
	fn a_joint_that_was_never_in_a_slot_is_refused() {
		// nothing this end holds could match a generation of nil, so a record
		// saying so is a peer confused about its own table.
		let mut ring = Reliable::new();
		let mut bytes = Vec::new();

		parting(&[Untied { slot: 4, generation: 0 }], &mut bytes).expect("it writes");
		post(&mut ring, Kind::Untied, &bytes).expect_err("and is refused on the way out");
		Pieces::new()
			.take(&piece(Kind::Untied.byte(), 0, 1, &bytes))
			.expect_err("and on the way in");
	}

	#[test]
	fn the_kind_a_parcel_answers_to_is_the_one_it_arrived_as() {
		assert_eq!(Parcel::Line(String::new()).kind(), Kind::Line);
		assert_eq!(Parcel::Scene(Vec::new()).kind(), Kind::Scene);
	}
}
