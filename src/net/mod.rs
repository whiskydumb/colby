//! The wire two colbys talk over: sequenced datagrams in, whole messages out.
//!
//! ```text
//!   Channel::send(message, now, |datagram| ..)  -> one or more datagrams
//!   Channel::receive(datagram, now)             -> Delivery
//!
//!   Reliable::write(&mut message)               commands, in front of the rest
//!   Reliable::read(&message, &mut commands)     the ones not run here before
//! ```
//!
//! **There is no socket in this crate**, the same way there is no output device
//! in the arithmetic that fills a sound buffer. A datagram arrives as a slice
//! of bytes and leaves through a closure; the handle they traveled over, the
//! address at the other end and the thread that read them belong to whoever
//! owns the loop. Two consequences, and they are the reason for the split:
//! every line in here is checked by running a test rather than by running a
//! program, and two endpoints can be stood up in one process with no operating
//! system between them - which is the harness the rest of the step is built on.
//!
//! **Time is a number the caller hands in.** Nothing here reads a clock, so a
//! run of two endpoints over a hundred simulated seconds produces the same
//! round-trip estimate and the same counts every time it is run, on any
//! machine.
//!
//! What is here, and what each part is for:
//!
//! - [`packet`] - the sixteen-byte head, the ceilings, and the wrap-aware
//!   reading of which sequence number is newer;
//! - [`fragment`] - cutting a message into datagrams and putting one back
//!   together, out of order if that is how it turns up;
//! - [`channel`] - one end of a conversation: sequence numbers, what the far
//!   end has acknowledged, the round-trip estimate and what was lost;
//! - [`reliable`] - the ring of numbered items that is the only thing on this
//!   wire nothing is allowed to lose;
//! - [`parcel`] - what one of those items is: a console line, or a piece of
//!   something far too long for one;
//! - [`command`] - what a client is asking for, and which of its asks the host
//!   has run;
//! - [`snapshot`] - what moved, as a difference from what the far end already
//!   has;
//! - [`ring`] - what a host remembers having told one peer, so the next telling
//!   can be that difference;
//! - [`random`] - a seeded shift register, so that a run can be repeated;
//! - [`link`] - a wire made as bad as somebody wants it, which is what there is
//!   to point a change at when the change is about loss.
//!
//! **Nothing is encrypted and nobody is authenticated.** Said out loud rather
//! than left to be discovered: anything that can reach the socket can say
//! anything a peer can say. What this crate does undertake is that no datagram,
//! however malformed or dishonest, can make it panic or reserve memory in
//! proportion to what the datagram claims. What it does **not** undertake is
//! that a peer cannot be a nuisance: one able to put a datagram on the wire can
//! drag the sequence number a long way forward, and everything honest behind it
//! then reads as older than what has already arrived. That is what having no
//! handshake means, and the answer to it is the same one encryption is - a
//! later step, said here so that nobody discovers it as a bug.

pub mod channel;
pub mod command;
pub mod fragment;
pub mod heard;
pub mod link;
pub mod packet;
pub mod parcel;
pub mod random;
pub mod reliable;
pub mod ring;
pub mod snapshot;

pub use self::{
	channel::{Channel, Delivery, HISTORY},
	command::{Block, MAX_ASKED, MAX_BLOCK, Told},
	heard::Heard,
	link::{BURST, Conditions, DUPLICATE, JITTER, LAG, LOSS, Link},
	packet::{
		ACK_BITS, HEADER_BYTES, Header, MAGIC, MAX_DATAGRAM, MAX_FRAGMENTS, MAX_MESSAGE,
		MAX_PAYLOAD, PROTOCOL_VERSION, Reason, after, distance,
	},
	parcel::{Kind, MAX_PARCEL, MAX_PIECE, MAX_PIECES, MAX_UNTIED, Parcel, Pieces, Untied},
	random::Random,
	reliable::{MAX_ITEM, MAX_ITEMS, Reliable},
	ring::{DEPTH, Ring},
	snapshot::{
		Change, FIELDS, Fault, LEAST_BASELINE, LEAST_ROOM, MAX_BASELINE, MAX_SLOTS, MAX_SNAPSHOT,
		NOTHING, SNAPSHOTS, Slot, Snapshot, Solid, WORDS, Written, every,
	},
};

#[cfg(test)]
mod tests {
	use std::time::Duration;

	use super::*;

	/// A full ring of the longest items there are, through a channel and out
	/// the far side.
	///
	/// The only test with both halves of the crate in it, and the one property
	/// the two of them share: a peer whose items have piled up as far as they
	/// can must still be able to send. @ref the assertion the build makes about
	/// the same thing in [`reliable`].
	#[test]
	fn a_full_ring_of_the_longest_items_crosses_in_one_message() {
		let (mut sending, mut receiving) = (Reliable::new(), Reliable::new());
		let (mut host, mut client) = (Channel::new(0x1111_1111), Channel::new(0x2222_2222));
		let longest = vec![0xA5_u8; MAX_ITEM];

		for _ in 0..MAX_ITEMS {
			sending
				.queue(&longest)
				.expect("the ring takes sixty-four");
		}

		let mut payload = Vec::new();

		sending.write(&mut payload);
		assert!(payload.len() > MAX_PAYLOAD * 8, "and it is a great many datagrams");

		let mut wire = Vec::new();

		host.send(&payload, Duration::ZERO, |datagram| wire.push(datagram.to_vec()))
			.expect("a full ring is inside the largest message there is");
		assert!(wire.len() > 8);

		let mut arrived = Vec::new();

		for datagram in &wire {
			if let Delivery::Message(message) = client.receive(datagram, Duration::ZERO) {
				arrived = message.to_vec();
			}
		}

		assert_eq!(arrived, payload, "every byte of the block, out the far side");

		let mut items = Vec::new();
		let read = receiving
			.read(&arrived, &mut items)
			.expect("and it reads as a block");

		assert_eq!(read, arrived.len(), "with nothing left over");
		assert_eq!(items.len(), 64);
		assert!(items.iter().all(|item| *item == longest));
	}

	/// The longest parcel there is, through the same wire, put back together.
	///
	/// The other half of the same property, one layer up: a world that fills
	/// the ring exactly still crosses in one message and comes back byte for
	/// byte.
	#[test]
	fn the_longest_parcel_there_is_crosses_in_one_message() {
		let (mut sending, mut receiving) = (Reliable::new(), Reliable::new());
		let (mut host, mut client) = (Channel::new(0x3333_3333), Channel::new(0x4444_4444));
		let scene: Vec<u8> = (0..MAX_PARCEL)
			.map(|at| u8::try_from(at * at % 251).expect("a remainder below a byte"))
			.collect();

		parcel::post(&mut sending, Kind::Scene, &scene).expect("a ringful");

		let mut payload = Vec::new();

		sending.write(&mut payload);

		let mut wire = Vec::new();

		host.send(&payload, Duration::ZERO, |datagram| wire.push(datagram.to_vec()))
			.expect("the longest parcel is inside the largest message there is");

		let mut arrived = Vec::new();

		for datagram in &wire {
			if let Delivery::Message(message) = client.receive(datagram, Duration::ZERO) {
				arrived = message.to_vec();
			}
		}

		let mut items = Vec::new();

		receiving
			.read(&arrived, &mut items)
			.expect("it reads as a block");
		assert_eq!(items.len(), usize::from(MAX_PIECES));

		let mut pieces = Pieces::new();
		let mut whole = None;

		for item in &items {
			whole = pieces.take(item).expect("a piece");
		}

		assert_eq!(whole, Some(Parcel::Scene(scene)), "every byte of it, in order");
	}
}
