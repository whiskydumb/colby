//! One end of a conversation: what has gone out, what has come in, and how far
//! away the other end is.
//!
//! ```text
//!   send(payload, now, |datagram| ..)   one message, one sequence number,
//!                                       as many datagrams as it takes
//!   receive(datagram, now) -> Delivery  a whole message, a piece of one,
//!                                       or why it was worth nothing
//! ```
//!
//! **Nothing here knows what a socket is.** Datagrams leave through a closure
//! and arrive as a slice, and both sides of that belong to somebody else. Time
//! arrives the same way: a [`Duration`] the caller measured, never a clock read
//! in here, which is what makes the round-trip estimate and the loss count
//! exactly reproducible in a test.
//!
//! **Acknowledgement is about messages, and about *delivery* rather than
//! arrival.** A message whose pieces did not all turn up is never acknowledged,
//! because a message nobody could read did not arrive as far as anything above
//! this cares. A message arriving *behind* one already handed over is thrown
//! away and is likewise never acknowledged. So the far end's picture of what
//! got through is the picture that matters, and a reordered datagram counts
//! against the link. That is a deliberate slight pessimism and it is what the
//! engine this wire is modeled on does.
//!
//! **The acknowledgement is read before the payload is judged.** The two halves
//! of a head are about opposite directions of travel, and one of them being
//! worthless says nothing about the other. What makes that matter rather than
//! being tidiness: a large message from the far end arrives as several
//! datagrams over several frames, and every one of them is carrying news about
//! this side's own traffic while the message itself is still incomplete.
//!
//! **Sequence nil is never sent.** It is the reading of "nothing has arrived
//! yet" in the acknowledgement field, which saves carrying a flag beside it and
//! is the trick the same engine uses. The number is skipped on the wrap, so
//! there are 65535 of them rather than 65536, and a datagram carrying nil is
//! refused when it is read rather than believed.

use std::time::Duration;

use colby_core::{Result, err};

use crate::{
	fragment::{self, Assembly},
	packet::{
		ACK_BITS, HEADER_BYTES, Header, MAX_DATAGRAM, MAX_MESSAGE, Reason, after, distance,
	},
};

/// How many messages back the channel remembers when each one went out.
///
/// Twice what the acknowledgement field can reach, so a message is only counted
/// lost once it is well past being acknowledgeable rather than the moment it
/// slips out of the field. What falls out of here unacknowledged is what the
/// far end never got.
pub const HISTORY: usize = 64;

/// The weight a fresh round-trip sample is given against the estimate so far.
///
/// One eighth, which is what the transport every game protocol is built beside
/// uses and what the networking libraries in this field copied from it. Worked
/// in whole nanoseconds rather than as a float, so the estimate is bit for bit
/// the same on every machine and a recorded run of two endpoints can be hashed.
const GAIN: u128 = 8;

/// What one datagram turned into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery<'a> {
	/// A whole message, ready to be read.
	Message(&'a [u8]),

	/// A piece of one that is not finished yet.
	Fragment,

	/// Nothing, and this is why. @ref [`Reason`].
	Ignored(Reason),
}

/// @note: every count below is a `saturating_add` rather than a `+=`, and it is
/// not tidiness. A shipping build here keeps overflow checks on, and the number
/// of datagrams thrown away is driven entirely by whatever is on the far end of
/// the socket - so a plain increment is a peer that can stop the process by
/// sending four billion pieces of garbage. No test can reach it, which is the
/// reason it is written down here instead.
///
/// One outgoing message and when it left.
#[derive(Clone, Copy, Debug)]
struct Sent {
	sequence: u16,
	at: Duration,
	acked: bool,
}

/// One end of a conversation with one other endpoint.
///
/// Holds room for the largest message there is and for [`HISTORY`] outgoing
/// ones, both taken once when it is made. Nothing it does afterwards allocates.
#[derive(Clone, Debug)]
pub struct Channel {
	outgoing: u16,
	incoming: u16,
	received: u32,
	history: Vec<Option<Sent>>,
	assembly: Assembly,
	rtt: Duration,
	samples: u32,
	sent: u32,
	acknowledged: u32,
	lost: u32,
	delivered: u32,
	ignored: u32,

	/// Which conversation this end is having.
	///
	/// Never changes while the process runs: it names the process rather than
	/// the conversation, which is what lets a far end that restarted be told
	/// apart from a far end that is merely quiet.
	session: u32,
}

impl Channel {
	/// A channel that has said nothing and heard nothing.
	///
	/// @param session - which conversation this end is having, written into
	/// every datagram it sends. @ref [`Header::session`](crate::Header)
	#[must_use]
	pub fn new(session: u32) -> Self {
		Self {
			session,
			outgoing: 1,
			incoming: 0,
			received: 0,
			history: vec![None; HISTORY],
			assembly: Assembly::new(),
			rtt: Duration::ZERO,
			samples: 0,
			sent: 0,
			acknowledged: 0,
			lost: 0,
			delivered: 0,
			ignored: 0,
		}
	}

	/// Forgets everything heard, keeping everything said.
	///
	/// For the moment the far end turns out to be a process that started
	/// again: what it has told this end is worth nothing, and the numbers it
	/// is about to use start over.
	///
	/// **The outgoing sequence carries on**, and getting that wrong is what
	/// makes a restart a deadlock rather than a hiccup. The far end has heard
	/// this end counting - if it restarted it has heard nothing and takes
	/// whatever comes, and if it did not restart it is holding the number this
	/// end last used. Starting over would be refused for as many messages as
	/// the conversation before it had, which is the exact fault this whole
	/// mechanism exists to remove. It is the same rule
	/// [`Ring::forget`](crate::Ring::forget) follows and for the same reason.
	pub fn forget(&mut self) {
		self.incoming = 0;
		self.received = 0;
		self.assembly = Assembly::new();
		self.rtt = Duration::ZERO;
		self.samples = 0;

		for slot in &mut self.history {
			*slot = None;
		}
	}

	/// Sends one message, as however many datagrams it takes.
	///
	/// The closure is handed each datagram in turn and is expected to put it on
	/// the wire. It cannot fail, which is deliberate: a datagram a socket
	/// refused is a datagram that was lost, and everything here already deals
	/// with that.
	///
	/// @param payload - the message, up to [`MAX_MESSAGE`] bytes
	/// @param now - how long this endpoint has been running
	/// @param into - where each datagram goes
	/// @return an error only when the message is too big to be cut up
	pub fn send<F>(&mut self, payload: &[u8], now: Duration, mut into: F) -> Result
	where
		F: FnMut(&[u8]),
	{
		let length = payload.len();

		if length > MAX_MESSAGE {
			return Err(err!(Network(
				"a message of {length} bytes is past the {MAX_MESSAGE} that fit in one set of \
				 datagrams"
			)));
		}

		let sequence = self.outgoing;
		let pieces = fragment::count(length);
		let fragments = u16::try_from(pieces).expect("a piece count of at most sixty-four fits");

		self.outgoing = advance(self.outgoing);
		self.remember(sequence, now);
		self.sent = self.sent.saturating_add(1);

		let mut datagram = [0_u8; MAX_DATAGRAM];
		let mut head = [0_u8; HEADER_BYTES];

		for index in 0..pieces {
			let piece = fragment::piece(payload, index);
			let header = Header {
				session: self.session,
				sequence,
				ack: self.incoming,
				ack_bits: self.received,
				fragment: u16::try_from(index).expect("a piece index below sixty-four fits"),
				fragments,
			};

			header.write(&mut head);
			datagram[..HEADER_BYTES].copy_from_slice(&head);
			datagram[HEADER_BYTES..HEADER_BYTES + piece.len()].copy_from_slice(piece);
			into(&datagram[..HEADER_BYTES + piece.len()]);
		}

		Ok(())
	}

	/// Takes one datagram and says what it was.
	///
	/// @param datagram - the whole of what arrived
	/// @param now - how long this endpoint has been running
	pub fn receive(&mut self, datagram: &[u8], now: Duration) -> Delivery<'_> {
		match self.accept(datagram, now) {
			| Err(reason) => {
				self.ignored = self.ignored.saturating_add(1);
				Delivery::Ignored(reason)
			},
			| Ok(false) => Delivery::Fragment,
			| Ok(true) => Delivery::Message(self.assembly.message()),
		}
	}

	/// The number the next message will go out with.
	#[must_use]
	pub const fn sequence(&self) -> u16 { self.outgoing }

	/// The newest message taken whole from the far end, nil if none has been.
	#[must_use]
	pub const fn incoming(&self) -> u16 { self.incoming }

	/// How long a message and its acknowledgement take, as far as anyone knows.
	///
	/// Zero until the first one comes back. @ref [`samples`](Self::samples).
	#[must_use]
	pub const fn rtt(&self) -> Duration { self.rtt }

	/// How many round trips the estimate is built on.
	#[must_use]
	pub const fn samples(&self) -> u32 { self.samples }

	/// How many messages have been sent.
	#[must_use]
	pub const fn sent(&self) -> u32 { self.sent }

	/// How many of them the far end has said it took whole.
	#[must_use]
	pub const fn acknowledged(&self) -> u32 { self.acknowledged }

	/// How many of them fell out of the history unacknowledged.
	///
	/// A message still inside the history is neither: its fate is not known
	/// yet, so a share of messages lost is [`lost`](Self::lost) against
	/// `lost + acknowledged` rather than against
	/// [`sent`](Self::sent).
	#[must_use]
	pub const fn lost(&self) -> u32 { self.lost }

	/// How many whole messages have been handed over.
	#[must_use]
	pub const fn delivered(&self) -> u32 { self.delivered }

	/// How many arriving datagrams were worth nothing.
	#[must_use]
	pub const fn ignored(&self) -> u32 { self.ignored }

	/// Everything about a datagram except the counting and the borrow.
	///
	/// @return whether a whole message is now sitting in the assembly
	fn accept(&mut self, datagram: &[u8], now: Duration) -> Result<bool, Reason> {
		let header = Header::read(datagram)?;

		self.confirm(header.ack, now);

		let mut bits = header.ack_bits;
		let mut back = 1_u16;

		while bits != 0 {
			if bits & 1 != 0 {
				self.confirm(header.ack.wrapping_sub(back), now);
			}

			bits >>= 1;
			back += 1;
		}

		if !after(header.sequence, self.incoming) {
			return Err(Reason::Stale);
		}

		if !self.assembly.building() {
			self.assembly
				.start(header.sequence, header.fragments);
		} else if header.sequence != self.assembly.sequence() {
			if !after(header.sequence, self.assembly.sequence()) {
				return Err(Reason::Stale);
			}

			self.assembly
				.start(header.sequence, header.fragments);
		} else if header.fragments != self.assembly.fragments() {
			return Err(Reason::Divided);
		}

		if !self
			.assembly
			.put(header.fragment, Header::piece(datagram))
		{
			return Ok(false);
		}

		self.assembly.finish();
		self.note(header.sequence);
		self.delivered = self.delivered.saturating_add(1);
		Ok(true)
	}

	/// Writes an outgoing message into the history, retiring what it displaces.
	fn remember(&mut self, sequence: u16, now: Duration) {
		let slot = usize::from(sequence) % HISTORY;

		if self.history[slot].is_some_and(|previous| !previous.acked) {
			self.lost = self.lost.saturating_add(1);
		}

		self.history[slot] = Some(Sent { sequence, at: now, acked: false });
	}

	/// Marks one outgoing message as having got there, and takes a round-trip
	/// sample from it.
	///
	/// Only the first acknowledgement of a message counts: the field carries
	/// the same one over and over, and a sample per repeat would drag the
	/// estimate towards however long the far end has been repeating itself.
	fn confirm(&mut self, sequence: u16, now: Duration) {
		// @note: nothing can observe this on its own, and it is kept because it
		// is the one line that says nil is not a message. No entry in the
		// history ever holds nil - the sequence starts at one and steps over it
		// on the wrap - so the slot's own number below refuses it as well. That
		// check is load-bearing in its own right and is tested; this one is
		// documentation.
		if sequence == 0 {
			return;
		}

		let slot = usize::from(sequence) % HISTORY;
		let Some(entry) = self.history[slot].as_mut() else {
			return;
		};

		if entry.sequence != sequence || entry.acked {
			return;
		}

		entry.acked = true;

		let sample = now.saturating_sub(entry.at);

		self.rtt = if self.samples == 0 {
			sample
		} else {
			blend(self.rtt, sample)
		};
		self.samples = self.samples.saturating_add(1);
		self.acknowledged = self.acknowledged.saturating_add(1);
	}

	/// Records that a message arrived whole, and shifts the acknowledgement
	/// field up to it.
	///
	/// @note: the bit for the message this one displaces is only set when there
	/// *was* one. Nil is the reading of "nothing yet" rather than a message, so
	/// the first thing ever delivered leaves the field empty rather than
	/// claiming a message nobody sent.
	fn note(&mut self, sequence: u16) {
		let shift = distance(sequence, self.incoming);
		let displaced = self.incoming != 0;

		self.received = if shift >= ACK_BITS {
			0
		} else {
			self.received << u32::from(shift)
		};

		if displaced && shift <= ACK_BITS {
			self.received |= 1_u32 << u32::from(shift - 1);
		}

		self.incoming = sequence;
	}
}

/// The next sequence number, stepping over nil.
const fn advance(sequence: u16) -> u16 {
	match sequence.wrapping_add(1) {
		| 0 => 1,
		| next => next,
	}
}

/// Folds one round-trip sample into the estimate.
fn blend(previous: Duration, sample: Duration) -> Duration {
	let mixed = (previous.as_nanos() * (GAIN - 1) + sample.as_nanos()) / GAIN;

	Duration::from_nanos(u64::try_from(mixed).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
	/// The two sessions the ends of these conversations are having.
	///
	/// Different from each other on purpose: a fixture where both ends used
	/// one number could not tell a channel writing its own session from one
	/// writing whatever it last heard.
	const ONE: u32 = 0x1111_1111;
	const TWO: u32 = 0x2222_2222;

	use super::*;
	use crate::packet::{MAX_PAYLOAD, PROTOCOL_VERSION};

	fn at(millis: u64) -> Duration { Duration::from_millis(millis) }

	/// Every datagram one message becomes.
	fn wire(channel: &mut Channel, payload: &[u8], now: Duration) -> Vec<Vec<u8>> {
		let mut out = Vec::new();

		channel
			.send(payload, now, |datagram| out.push(datagram.to_vec()))
			.expect("the message fits");
		out
	}

	/// Sends one message that nothing ever receives.
	fn lose(channel: &mut Channel, payload: &[u8], now: Duration) {
		channel
			.send(payload, now, |_| ())
			.expect("the message fits");
	}

	/// Sends one message and hands every datagram of it to the far end.
	///
	/// @return the whole messages the far end made of them
	fn carry(
		from: &mut Channel,
		to: &mut Channel,
		payload: &[u8],
		now: Duration,
	) -> Vec<Vec<u8>> {
		let mut got = Vec::new();

		for datagram in wire(from, payload, now) {
			if let Delivery::Message(message) = to.receive(&datagram, now) {
				got.push(message.to_vec());
			}
		}

		got
	}

	/// The same, for when what came back does not matter.
	fn post(from: &mut Channel, to: &mut Channel, payload: &[u8], now: Duration) {
		let got = carry(from, to, payload, now);

		assert_eq!(got.len(), 1, "the message got through");
	}

	/// A message of `length` bytes, no two neighbors alike.
	fn message(length: usize) -> Vec<u8> {
		(0..length)
			.map(|at| u8::try_from(at % 251).unwrap_or(0))
			.collect()
	}

	/// The head of a datagram.
	fn head(datagram: &[u8]) -> Header { Header::read(datagram).expect("it is a datagram") }

	#[test]
	fn a_message_that_fits_crosses_in_one_datagram() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));
		let sent = wire(&mut host, b"hello", at(0));

		assert_eq!(sent.len(), 1, "five bytes is one datagram");
		assert_eq!(sent[0].len(), HEADER_BYTES + 5, "a head and the message");
		assert_eq!(client.receive(&sent[0], at(1)), Delivery::Message(b"hello"));
		assert_eq!(client.delivered(), 1);
	}

	#[test]
	fn a_message_too_big_for_one_datagram_crosses_in_several_and_comes_back_whole() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));
		let payload = message(MAX_PAYLOAD * 2 + 3);
		let sent = wire(&mut host, &payload, at(0));

		assert_eq!(sent.len(), 3, "two full pieces and a tail");

		let mut delivered = 0;

		for (index, datagram) in sent.iter().enumerate() {
			match client.receive(datagram, at(1)) {
				| Delivery::Message(message) => {
					assert_eq!(message, payload.as_slice(), "every byte back where it was");
					delivered += 1;
				},
				| Delivery::Fragment => assert!(index < 2, "only the last one finishes it"),
				| other => panic!("{other:?}"),
			}
		}

		assert_eq!(delivered, 1, "three datagrams are one message");
	}

	#[test]
	fn the_largest_message_there_is_crosses() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));
		let payload = message(MAX_MESSAGE);
		let got = carry(&mut host, &mut client, &payload, at(0));

		assert_eq!(got.len(), 1);
		assert_eq!(got[0].len(), MAX_MESSAGE);
		assert_eq!(got[0], payload, "sixty-four datagrams and not a byte out of place");
	}

	#[test]
	fn a_message_past_the_ceiling_is_refused_rather_than_cut_up() {
		let mut host = Channel::new(ONE);
		let payload = vec![0_u8; MAX_MESSAGE + 1];
		let error = host
			.send(&payload, at(0), |_| panic!("nothing should go out"))
			.expect_err("one byte past the ceiling");

		assert!(error.to_string().starts_with("network: "), "{error}");
		assert_eq!(host.sequence(), 1, "and the sequence did not move");
		assert_eq!(host.sent(), 0);
	}

	#[test]
	fn the_sequence_moves_once_a_message_rather_than_once_a_datagram() {
		let mut host = Channel::new(ONE);

		assert_eq!(host.sequence(), 1, "nil is reserved, so the first message is one");
		lose(&mut host, &message(MAX_PAYLOAD * 3), at(0));
		assert_eq!(host.sequence(), 2, "three datagrams, one number");
		lose(&mut host, b"x", at(0));
		assert_eq!(host.sequence(), 3);
	}

	#[test]
	fn every_datagram_of_one_message_carries_the_same_number() {
		let mut host = Channel::new(ONE);
		let sent = wire(&mut host, &message(MAX_PAYLOAD * 2), at(0));
		let (first, second) = (head(&sent[0]), head(&sent[1]));

		assert_eq!(sent.len(), 2);
		assert_eq!(first.sequence, second.sequence, "one message, one sequence");
		assert_eq!(first.fragment, 0);
		assert_eq!(second.fragment, 1);
		assert_eq!(first.fragments, 2, "and both say how many there are");
		assert_eq!(second.fragments, 2);
	}

	#[test]
	fn the_same_message_twice_is_taken_once() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));
		let sent = wire(&mut host, b"once", at(0));

		assert!(matches!(client.receive(&sent[0], at(1)), Delivery::Message(_)));
		assert_eq!(
			client.receive(&sent[0], at(2)),
			Delivery::Ignored(Reason::Stale),
			"a repeat is a repeat"
		);
		assert_eq!(client.delivered(), 1);
		assert_eq!(client.ignored(), 1);
	}

	#[test]
	fn a_message_behind_one_already_handed_over_is_thrown_away() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));
		let first = wire(&mut host, b"one", at(0));
		let second = wire(&mut host, b"two", at(0));

		assert!(matches!(client.receive(&second[0], at(1)), Delivery::Message(_)));
		assert_eq!(
			client.receive(&first[0], at(1)),
			Delivery::Ignored(Reason::Stale),
			"the older one has been overtaken and is worth nothing"
		);
	}

	#[test]
	fn a_piece_of_a_newer_message_throws_away_the_one_being_built() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));
		let first = wire(&mut host, &message(MAX_PAYLOAD * 2), at(0));
		let second = wire(&mut host, b"newer", at(0));

		assert_eq!(client.receive(&first[0], at(1)), Delivery::Fragment);
		assert_eq!(
			client.receive(&second[0], at(1)),
			Delivery::Message(b"newer"),
			"a whole newer message beats half an older one"
		);
		assert_eq!(
			client.receive(&first[1], at(1)),
			Delivery::Ignored(Reason::Stale),
			"and the rest of the older one is then worth nothing"
		);
	}

	#[test]
	fn a_message_missing_a_piece_is_never_acknowledged() {
		// the property the whole acknowledgement scheme rests on: what is
		// acknowledged is what somebody could actually read. Two are taken
		// whole first, so that the answer is a field which did not move rather
		// than a field that was empty either way.
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		post(&mut host, &mut client, b"a", at(0));
		post(&mut host, &mut client, b"b", at(0));

		let sent = wire(&mut host, &message(MAX_PAYLOAD * 2), at(0));

		assert_eq!(client.receive(&sent[0], at(1)), Delivery::Fragment);
		assert_eq!(client.incoming(), 2, "the third has not been taken whole");

		let back = wire(&mut client, b"", at(2));

		assert_eq!(head(&back[0]).ack, 2, "so there is nothing new to acknowledge");
		assert_eq!(head(&back[0]).ack_bits, 0b1, "and the field is where it was");
	}

	#[test]
	fn the_pieces_of_one_message_may_reach_a_channel_in_any_order() {
		// the reassembly is checked directly elsewhere; this is the path that
		// reads the piece index off the wire and hands it over.
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));
		let payload = message(MAX_PAYLOAD * 2 + 9);
		let sent = wire(&mut host, &payload, at(0));

		assert_eq!(sent.len(), 3);
		assert_eq!(client.receive(&sent[2], at(1)), Delivery::Fragment, "the tail first");
		assert_eq!(client.receive(&sent[0], at(1)), Delivery::Fragment);

		match client.receive(&sent[1], at(1)) {
			| Delivery::Message(got) => assert_eq!(got, payload.as_slice(), "and it is whole"),
			| other => panic!("{other:?}"),
		}
	}

	#[test]
	fn a_piece_of_an_older_message_arriving_after_a_newer_one_started_is_thrown_away() {
		// reordering across two messages, neither of them complete - so it is
		// the message being built rather than the last one handed over that has
		// to decide. Nothing is delivered until the newer one finishes.
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));
		let one = vec![1_u8; MAX_PAYLOAD * 2];
		let two = vec![2_u8; MAX_PAYLOAD * 2];
		let first = wire(&mut host, &one, at(0));
		let second = wire(&mut host, &two, at(0));

		assert_eq!(client.receive(&first[0], at(1)), Delivery::Fragment);
		assert_eq!(client.receive(&second[0], at(1)), Delivery::Fragment, "the newer one wins");
		assert_eq!(
			client.receive(&first[1], at(1)),
			Delivery::Ignored(Reason::Stale),
			"and the rest of the older one is not worth starting over for"
		);
		assert_eq!(client.incoming(), 0, "nothing has been taken whole yet");

		match client.receive(&second[1], at(1)) {
			| Delivery::Message(got) => assert_eq!(got, two.as_slice(), "the newer one, whole"),
			| other => panic!("{other:?}"),
		}
	}

	#[test]
	fn a_message_is_acknowledged_by_a_bit_when_the_answer_naming_it_was_lost() {
		// the whole reason four bytes of acknowledgement field are on the wire.
		// The far end names its newest and repeats the thirty-two before it, so
		// one answer getting through covers the answers that did not.
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		for step in 0..3_u64 {
			post(&mut host, &mut client, b"x", at(step));
		}

		lose(&mut client, b"", at(10));
		lose(&mut client, b"", at(11));

		let last = wire(&mut client, b"", at(12));

		assert_eq!(head(&last[0]).ack, 3, "the newest it took");
		assert_eq!(head(&last[0]).ack_bits, 0b11, "and the two before it, again");
		assert!(matches!(host.receive(&last[0], at(12)), Delivery::Message(_)));
		assert_eq!(host.acknowledged(), 3, "all three of them, off one answer");
		assert_eq!(host.samples(), 3);
		assert_eq!(host.lost(), 0);
	}

	#[test]
	fn forgetting_drops_what_was_heard_and_keeps_what_was_said() {
		// **the asymmetry is the whole of it.** A far end that started again
		// has heard nothing and takes whatever number arrives, so counting on
		// costs nothing; a far end that did *not* start again is holding the
		// number this end last used and would refuse everything below it. Only
		// one of those two is safe, and it is the same rule the snapshot ring
		// on the sending side follows.
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		for _ in 0..5 {
			post(&mut host, &mut client, b"x", at(0));
			post(&mut client, &mut host, b"y", at(1));
		}

		let said = host.sequence();

		assert!(said > 5, "five messages each way, so the numbering has moved");
		assert_eq!(host.incoming(), 5, "and five have arrived");

		host.forget();

		assert_eq!(host.sequence(), said, "what this end says carries on");
		assert_eq!(host.incoming(), 0, "and what it heard is gone");
	}

	#[test]
	fn a_forgotten_channel_takes_a_far_end_counting_from_one_again() {
		// the fault this exists for, at the size of one channel: a far end
		// that restarted sends message one, and a channel holding what the
		// conversation before it reached would refuse that and everything
		// after it until it had counted back up.
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		for _ in 0..40 {
			post(&mut client, &mut host, b"y", at(0));
		}

		assert_eq!(host.incoming(), 40, "forty messages in");

		// a fresh far end, counting from one, which is what a restart is.
		let mut again = Channel::new(0x3333_3333);
		let first = wire(&mut again, b"hello", at(1));

		assert_eq!(
			host.receive(&first[0], at(1)),
			Delivery::Ignored(Reason::Stale),
			"which a channel that remembers the last conversation refuses"
		);

		host.forget();

		let second = wire(&mut again, b"hello", at(2));

		assert!(
			matches!(host.receive(&second[0], at(2)), Delivery::Message(_)),
			"and one that has forgotten takes"
		);
	}

	#[test]
	fn a_channel_writes_its_own_session_into_every_datagram_it_sends() {
		let mut host = Channel::new(ONE);
		let sent = wire(&mut host, &message(MAX_PAYLOAD * 3), at(0));

		assert_eq!(sent.len(), 3, "three pieces of one message");

		for piece in &sent {
			assert_eq!(head(piece).session, ONE, "every piece of it, not only the first");
		}
	}

	#[test]
	fn a_piece_claiming_a_different_division_is_refused() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));
		let sent = wire(&mut host, &message(MAX_PAYLOAD * 2), at(0));
		let mut forged = sent[1].clone();

		assert_eq!(client.receive(&sent[0], at(1)), Delivery::Fragment);

		// say the same message is three pieces rather than two. The piece count
		// is the last field of the head, named that way rather than by a
		// number: the head has grown once already and a literal offset here
		// would have gone on forging a different field in silence.
		forged[HEADER_BYTES - 2..HEADER_BYTES].copy_from_slice(&3_u16.to_le_bytes());
		assert_eq!(client.receive(&forged, at(1)), Delivery::Ignored(Reason::Divided));
	}

	#[test]
	fn what_arrived_is_written_into_the_acknowledgement_field() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		for _ in 0..3 {
			post(&mut host, &mut client, b"x", at(0));
		}

		let back = wire(&mut client, b"", at(2));

		assert_eq!(head(&back[0]).ack, 3, "the newest one taken whole");
		assert_eq!(head(&back[0]).ack_bits, 0b11, "and the two before it");
	}

	#[test]
	fn the_first_message_ever_taken_claims_nothing_before_itself() {
		// nil in the field means "nothing yet", so the message it displaces has
		// to be a real one before a bit is set for it.
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		post(&mut host, &mut client, b"x", at(0));

		let back = wire(&mut client, b"", at(1));

		assert_eq!(head(&back[0]).ack, 1);
		assert_eq!(head(&back[0]).ack_bits, 0, "there was no message nil");
	}

	#[test]
	fn a_message_that_never_arrived_shows_as_a_hole_in_the_field() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		post(&mut host, &mut client, b"a", at(0));
		lose(&mut host, b"b", at(0));
		post(&mut host, &mut client, b"c", at(0));

		let back = wire(&mut client, b"", at(2));

		assert_eq!(head(&back[0]).ack, 3);
		assert_eq!(head(&back[0]).ack_bits, 0b10, "the second is a hole and the first is not");
	}

	#[test]
	fn a_jump_of_exactly_the_field_keeps_the_oldest_bit() {
		// three of them first, so that the field has something in it which a
		// jump of exactly its width has to push out. Starting from an empty
		// field cannot tell a field that was cleared from one that was shifted
		// by nothing.
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		for _ in 0..3 {
			post(&mut host, &mut client, b"a", at(0));
		}

		for _ in 0..31 {
			lose(&mut host, b"gone", at(0));
		}

		let far = wire(&mut host, b"z", at(0));

		assert_eq!(head(&far[0]).sequence, 35);
		assert!(matches!(client.receive(&far[0], at(1)), Delivery::Message(_)));

		let back = wire(&mut client, b"", at(2));

		assert_eq!(head(&back[0]).ack, 35);
		assert_eq!(
			head(&back[0]).ack_bits,
			1 << 31,
			"message three just reaches and the two before it are gone"
		);
	}

	#[test]
	fn a_jump_past_the_field_clears_it() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		for _ in 0..3 {
			post(&mut host, &mut client, b"a", at(0));
		}

		for _ in 0..32 {
			lose(&mut host, b"gone", at(0));
		}

		let far = wire(&mut host, b"z", at(0));

		assert_eq!(head(&far[0]).sequence, 36);
		assert!(matches!(client.receive(&far[0], at(1)), Delivery::Message(_)));

		let back = wire(&mut client, b"", at(2));

		assert_eq!(head(&back[0]).ack_bits, 0, "thirty-three behind is out of reach");
	}

	#[test]
	fn the_first_round_trip_is_the_estimate_and_the_next_ones_move_it_by_an_eighth() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		assert_eq!(host.rtt(), Duration::ZERO, "nothing is known yet");
		assert_eq!(host.samples(), 0);

		post(&mut host, &mut client, b"a", at(0));
		post(&mut client, &mut host, b"", at(50));

		assert_eq!(host.rtt(), at(50), "the first sample is taken as it stands");
		assert_eq!(host.samples(), 1);

		post(&mut host, &mut client, b"b", at(100));
		post(&mut client, &mut host, b"", at(200));

		// seven eighths of fifty plus one eighth of a hundred.
		assert_eq!(host.rtt(), Duration::from_micros(56_250), "one eighth of the way over");
		assert_eq!(host.samples(), 2);
	}

	#[test]
	fn the_same_message_acknowledged_twice_is_one_sample() {
		// the field carries the same news in every datagram, so a sample per
		// repeat would drag the estimate towards however long the far end has
		// been saying the same thing.
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		post(&mut host, &mut client, b"a", at(0));
		post(&mut client, &mut host, b"", at(20));

		assert_eq!(host.rtt(), at(20));
		assert_eq!(host.samples(), 1);

		post(&mut client, &mut host, b"", at(500));

		assert_eq!(host.samples(), 1, "the same acknowledgement says nothing new");
		assert_eq!(host.rtt(), at(20), "so the estimate does not move");
	}

	#[test]
	fn an_acknowledgement_is_read_out_of_a_piece_of_a_message_that_never_finishes() {
		// the two halves of a head are about opposite directions of travel. A
		// large message from the far end arrives over several datagrams, and
		// waiting for the whole of it before believing the acknowledgement it
		// carries would hold the estimate back by exactly that long.
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		post(&mut host, &mut client, b"a", at(0));

		let big = wire(&mut client, &message(MAX_PAYLOAD * 2), at(40));

		assert_eq!(host.receive(&big[0], at(40)), Delivery::Fragment, "one of two");
		assert_eq!(host.samples(), 1, "and the news in its head still counted");
		assert_eq!(host.rtt(), at(40));
		assert_eq!(host.acknowledged(), 1);
		assert_eq!(host.delivered(), 0, "while no message has been handed over");
	}

	#[test]
	fn an_acknowledgement_of_a_message_long_gone_does_not_credit_the_one_in_its_slot() {
		// the history is a ring, so sixty-four messages later some other
		// message is sitting where this one was. Without the slot carrying its
		// own number, a very late acknowledgement would credit whatever it
		// found there and take a round-trip sample off the wrong send.
		let mut host = Channel::new(ONE);

		lose(&mut host, b"one", at(0));

		for step in 0..HISTORY {
			let step = u64::try_from(step).unwrap_or(0);

			lose(&mut host, b"x", at(step + 100));
		}

		assert_eq!(host.sequence(), 66, "so sequence sixty-five is in slot one now");

		let mut late = [0_u8; HEADER_BYTES];

		Header {
			// the far end's, because this is a datagram the far end sent.
			session: TWO,
			sequence: 1,
			ack: 1,
			ack_bits: 0,
			fragment: 0,
			fragments: 1,
		}
		.write(&mut late);
		assert!(matches!(host.receive(&late, at(1000)), Delivery::Message(_)));
		assert_eq!(host.acknowledged(), 0, "message one is gone and sixty-five is not it");
		assert_eq!(host.samples(), 0);
		assert_eq!(host.rtt(), Duration::ZERO);
	}

	#[test]
	fn an_acknowledgement_is_read_even_out_of_a_datagram_that_is_thrown_away() {
		// what pins the order of the two halves of a head. A peer that is
		// working will not produce this exact datagram - its sequence and its
		// acknowledgement grow together - but a piece of an unfinished message
		// is the same case and does arrive constantly, and the datagram below
		// is the smallest way to say the rule.
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		post(&mut host, &mut client, b"a", at(0));
		post(&mut client, &mut host, b"", at(10));
		assert_eq!(host.acknowledged(), 1);

		lose(&mut host, b"b", at(20));

		let mut stale = [0_u8; HEADER_BYTES];

		Header {
			session: TWO,
			sequence: 1,
			ack: 2,
			ack_bits: 0,
			fragment: 0,
			fragments: 1,
		}
		.write(&mut stale);
		assert_eq!(
			host.receive(&stale, at(30)),
			Delivery::Ignored(Reason::Stale),
			"its own sequence is one the host has passed"
		);
		assert_eq!(host.acknowledged(), 2, "and the news it carried is good all the same");
		assert_eq!(host.rtt(), at(10), "with a sample off the message it acknowledged");
	}

	#[test]
	fn a_message_the_far_end_never_took_is_counted_lost_when_it_falls_out_of_the_history() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		lose(&mut host, b"gone", at(0));
		assert_eq!(host.lost(), 0, "it might still be in flight");

		for step in 0..HISTORY {
			let step = u64::try_from(step).unwrap_or(0);

			post(&mut host, &mut client, b"x", at(step + 1));
		}

		assert_eq!(host.lost(), 1, "sixty-four messages later it is not coming back");
	}

	#[test]
	fn what_did_arrive_is_never_counted_lost() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		for step in 0..HISTORY * 2 {
			let step = u64::try_from(step).unwrap_or(0);

			post(&mut host, &mut client, b"x", at(step));
			post(&mut client, &mut host, b"", at(step));
		}

		assert_eq!(host.lost(), 0, "every one of them came back");
		assert_eq!(host.acknowledged(), host.sent(), "and all of it was acknowledged");
	}

	#[test]
	fn half_the_messages_lost_settles_at_half() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		for step in 0..HISTORY * 2 {
			let step = u64::try_from(step).unwrap_or(0);

			// every other message never leaves the room.
			if step.is_multiple_of(2) {
				lose(&mut host, b"gone", at(step));
			} else {
				post(&mut host, &mut client, b"x", at(step));
				post(&mut client, &mut host, b"", at(step));
			}
		}

		assert_eq!(host.sent(), 128);
		assert_eq!(host.acknowledged(), 64, "the ones that went through");
		assert_eq!(host.lost(), 32, "and the ones far enough back to have been given up on");
	}

	#[test]
	fn something_that_is_not_ours_is_counted_and_named() {
		let mut client = Channel::new(ONE);
		let mut stray = vec![0_u8; HEADER_BYTES + 4];

		stray[2..4].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
		assert_eq!(client.receive(&stray, at(0)), Delivery::Ignored(Reason::Foreign));
		assert_eq!(client.receive(&[1_u8, 2, 3], at(0)), Delivery::Ignored(Reason::Short));
		assert_eq!(client.ignored(), 2, "counted rather than logged, one per stray");
		assert_eq!(client.delivered(), 0);
	}

	#[test]
	fn two_endpoints_talking_to_each_other_agree_on_what_got_through() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		for step in 0..200_u64 {
			post(&mut host, &mut client, &message(40), at(step * 10));
			post(&mut client, &mut host, b"reply", at(step * 10 + 5));
		}

		assert_eq!(host.sent(), 200);
		assert_eq!(client.delivered(), 200, "everything the host said arrived");
		assert_eq!(host.delivered(), 200);
		assert_eq!(host.acknowledged(), 200, "and the host was told so about all of it");
		assert_eq!(host.lost(), 0);
		assert_eq!(host.ignored(), 0);
		assert_eq!(host.rtt(), at(5), "with every round trip the same length");
	}

	#[test]
	fn the_sequence_steps_over_nil_when_it_wraps() {
		// nil is the reading of "nothing has arrived yet" in the
		// acknowledgement field, so it is never a real message.
		let mut host = Channel::new(ONE);

		assert_eq!(advance(65_534), 65_535);
		assert_eq!(advance(65_535), 1, "and the wrap steps over it");

		host.outgoing = 65_535;
		lose(&mut host, b"x", at(0));
		assert_eq!(host.sequence(), 1);
	}

	#[test]
	fn a_message_sent_across_the_wrap_still_reads_as_newer() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));

		host.outgoing = 65_535;
		client.incoming = 65_534;

		let last = wire(&mut host, b"last", at(0));
		let first = wire(&mut host, b"first", at(0));

		assert_eq!(client.receive(&last[0], at(1)), Delivery::Message(b"last"));
		assert_eq!(client.receive(&first[0], at(1)), Delivery::Message(b"first"));
		assert_eq!(client.incoming(), 1, "over the top and on");
	}

	#[test]
	fn the_estimate_is_whole_nanoseconds_and_not_a_float() {
		// what makes a recorded run of two endpoints hashable.
		assert_eq!(blend(at(80), at(80)), at(80), "a sample that agrees moves nothing");
		assert_eq!(
			blend(Duration::from_nanos(8), Duration::ZERO),
			Duration::from_nanos(7),
			"and the arithmetic is exact at one nanosecond"
		);
	}

	#[test]
	fn a_clock_that_went_backwards_is_a_sample_of_nothing_rather_than_a_panic() {
		let (mut host, mut client) = (Channel::new(ONE), Channel::new(TWO));
		let sent = wire(&mut host, b"a", at(100));

		assert!(matches!(client.receive(&sent[0], at(100)), Delivery::Message(_)));

		let back = wire(&mut client, b"", at(100));

		assert!(matches!(host.receive(&back[0], at(0)), Delivery::Message(_)));
		assert_eq!(host.rtt(), Duration::ZERO, "not a subtraction that goes under");
		assert_eq!(host.samples(), 1);
	}

	#[test]
	fn the_history_reaches_back_twice_as_far_as_the_acknowledgement_field() {
		// which is what makes falling out of it mean "the far end never got
		// this" rather than "this can no longer be named".
		assert_eq!(HISTORY, 64);
		assert!(u16::try_from(HISTORY).is_ok_and(|history| history > ACK_BITS));
		assert_eq!(Channel::new(ONE).history.len(), HISTORY);
	}
}
