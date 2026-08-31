//! The ring of numbered strings that is the only thing on this wire nothing is
//! allowed to lose.
//!
//! ```text
//!    0  received  u32   the far end's highest command this side has run
//!    4  first     u32   the number of the first command below
//!    8  count     u16   how many follow
//!   10  per command: u16 length, that many bytes of text
//! ```
//!
//! **There is no retransmission timer, no window and no state machine.** Every
//! command that has not been acknowledged is written into *every* outgoing
//! message until the far end says it has run it. A command is therefore lost
//! only when the connection is, and the whole mechanism is a ring, two counters
//! and the loop above. That is what the engine this is modeled on does with the
//! same numbers, sixty-four commands of a thousand characters, and it is why a
//! game can treat a remote call as a thing that simply happens.
//!
//! **What a command is, is a console line.** Not a generated stub and not a
//! function pointer: the text goes to the same parser a person typing at the
//! terminal goes through. A string survives a module being swapped out from
//! under it where a pointer does not, and there is already a table of named
//! commands with help text on both sides of that boundary - so this layer costs
//! a ring and thirty lines rather than a code generator.
//!
//! **Nothing here decides who may run what.** A command arriving from a peer is
//! handed back to the caller as text; whether a peer is allowed to run it is a
//! question for whoever owns the world, and it is not answered in this crate.
//!
//! **The numbers do not wrap.** They are counted in `u32`, and
//! [`queue`](Reliable::queue) refuses rather than rolling over - which at sixty
//! commands a second is two years of somebody typing without a pause.

use colby_core::{Result, err};

use crate::packet::{MAX_MESSAGE, u16_at, u32_at};

/// How many commands may be waiting for acknowledgement at once.
///
/// A sixty-fifth is refused rather than overwriting the oldest, because a
/// reliable command that was quietly dropped is worse than a connection that
/// said it had a problem.
pub const MAX_COMMANDS: u32 = 64;

/// The longest one command may be, in bytes.
pub const MAX_COMMAND: usize = 1024;

/// The same count as a length, which is what the ring is built with.
const SLOTS: usize = 64;

/// How many bytes come before the first command.
const HEAD: usize = 10;

const _: () = assert!(
	SLOTS == 64 && MAX_COMMANDS == 64,
	"the ring is indexed by the command number modulo its own length"
);

// the one invariant the two halves of this crate share: a block carrying a full
// ring of the longest commands there are has to fit in one message, or a peer
// that lets its commands pile up stops being able to send at all. Nothing
// downstream checks it, so it is checked here at the build.
const _: () = assert!(
	HEAD + SLOTS * (2 + MAX_COMMAND) <= MAX_MESSAGE,
	"a full ring of the longest commands has to fit in one message"
);

/// Both directions of the reliable command stream with one peer.
///
/// One structure rather than two because it is one block on the wire: what this
/// side is still resending and what it has run from the far end are written
/// side by side and read back together.
#[derive(Clone, Debug)]
pub struct Reliable {
	queued: Vec<String>,
	sent: u32,
	acknowledged: u32,
	received: u32,
}

impl Default for Reliable {
	fn default() -> Self { Self::new() }
}

impl Reliable {
	/// A ring with nothing in it.
	#[must_use]
	pub fn new() -> Self {
		Self {
			queued: vec![String::new(); SLOTS],
			sent: 0,
			acknowledged: 0,
			received: 0,
		}
	}

	/// Puts one command in the ring, to be resent until it is acknowledged.
	///
	/// @param text - the console line, one line and at most [`MAX_COMMAND`]
	/// bytes of it
	/// @return an error when the text is not a command or when the far end has
	/// let too many pile up
	pub fn queue(&mut self, text: &str) -> Result {
		if text.is_empty() {
			return Err(err!(Network("an empty reliable command says nothing")));
		}

		let length = text.len();

		if length > MAX_COMMAND {
			return Err(err!(Network(
				"a reliable command of {length} bytes is past the {MAX_COMMAND} one may be"
			)));
		}

		// @note: a shape rule rather than a guard against anything. A command
		// is one console line, and a line break in the middle of one would make
		// it two on the far side. Nothing downstream of here treats the text as
		// a format string, so there is nothing else to strip.
		if text
			.bytes()
			.any(|byte| byte < 0x20 || byte == 0x7F)
		{
			return Err(err!(Network("a reliable command is one line and this one is not")));
		}

		if self.pending() >= MAX_COMMANDS {
			return Err(err!(Network(
				"{MAX_COMMANDS} reliable commands are waiting and the far end has run none of \
				 them"
			)));
		}

		if self.sent == u32::MAX {
			return Err(err!(Network("the reliable command numbering has run out")));
		}

		self.sent += 1;

		let slot = slot(self.sent);

		text.clone_into(&mut self.queued[slot]);
		Ok(())
	}

	/// How many commands are waiting to be acknowledged.
	#[must_use]
	pub const fn pending(&self) -> u32 { self.sent - self.acknowledged }

	/// How many commands have been put in the ring, ever.
	#[must_use]
	pub const fn sent(&self) -> u32 { self.sent }

	/// The highest of them the far end has said it ran.
	#[must_use]
	pub const fn acknowledged(&self) -> u32 { self.acknowledged }

	/// The highest command from the far end that has been handed over.
	#[must_use]
	pub const fn received(&self) -> u32 { self.received }

	/// Writes the block onto the end of an outgoing message.
	///
	/// Appends rather than replaces, so a caller can put this in front of
	/// whatever else the message carries and [`read`](Self::read) will say
	/// where the rest begins.
	///
	/// @param out - the message being built
	pub fn write(&self, out: &mut Vec<u8>) {
		let pending = self.pending();
		let first = self.acknowledged.saturating_add(1);
		let count = u16::try_from(pending).expect("at most sixty-four are ever waiting");

		out.extend_from_slice(&self.received.to_le_bytes());
		out.extend_from_slice(&first.to_le_bytes());
		out.extend_from_slice(&count.to_le_bytes());

		let mut sequence = self.acknowledged;

		for _ in 0..pending {
			sequence += 1;

			let text = &self.queued[slot(sequence)];
			let length =
				u16::try_from(text.len()).expect("a command is at most a thousand bytes");

			out.extend_from_slice(&length.to_le_bytes());
			out.extend_from_slice(text.as_bytes());
		}
	}

	/// Reads a block off the front of an arriving message.
	///
	/// Retires whatever the far end has acknowledged, and hands over the
	/// commands that have not been run here before. Repeats are dropped: the
	/// far end resends everything unacknowledged in every message, so almost
	/// every command arrives several times and must run once.
	///
	/// @note: a block that is refused part way through may already have put
	/// the commands before the fault into `out` and counted them as run. That
	/// is not tidy and it does not matter: a block that does not parse is a
	/// peer that is broken or lying, and the only thing to do with one is stop
	/// talking to it.
	///
	/// @param bytes - the message, from the start of the block
	/// @param out - filled with the commands to run, in order, cleared first
	/// @return how many bytes the block took, so the rest of the message can be
	/// read from there
	pub fn read(&mut self, bytes: &[u8], out: &mut Vec<String>) -> Result<usize> {
		out.clear();

		let length = bytes.len();

		if length < HEAD {
			return Err(err!(Network("a reliable block of {length} bytes has no head in it")));
		}

		let first = u32_at(bytes, 4);
		let count = u32::from(u16_at(bytes, 8));

		if count > MAX_COMMANDS {
			return Err(err!(Network(
				"a block of {count} reliable commands is past the {MAX_COMMANDS} that can be \
				 waiting"
			)));
		}

		// commands are numbered from one, so a block numbered from nil shifts
		// every one of them down a place: its first command would be compared
		// against nil, dropped, and then acknowledged as though it had run.
		if first == 0 {
			return Err(err!(Network("a block of reliable commands numbered from nil")));
		}

		// @note: the far end only stops resending a command once this side has
		// said it ran it, so a block that starts past that is a far end which
		// has given up on commands that never arrived. There is nothing to be
		// done about it from here, and carrying on would run the commands after
		// the hole as if nothing were missing.
		let expected = self.received.saturating_add(1);

		if first > expected {
			return Err(err!(Network(
				"the far end has moved on to reliable command {first} with {expected} never \
				 delivered"
			)));
		}

		// read last, so that a block that is about to be refused has not
		// already emptied the ring it was meant to be acknowledging.
		self.retire(u32_at(bytes, 0));

		let mut at = HEAD;

		for index in 0..count {
			let (text, next) = take(bytes, at)?;

			at = next;

			let sequence = first.saturating_add(index);

			if sequence > self.received {
				self.received = sequence;
				out.push(text.to_owned());
			}
		}

		Ok(at)
	}

	/// Takes the far end's word for what it has run, as far as it can be taken.
	///
	/// @note: clamped rather than trusted, in both directions. A number below
	/// what is already known would un-retire commands and resend them; a number
	/// above what was ever sent would retire commands that have not been
	/// written yet and lose them. Neither can come from a peer that is working,
	/// so neither is worth an error - but both are worth being unable to
	/// happen.
	fn retire(&mut self, theirs: u32) {
		if theirs > self.acknowledged {
			self.acknowledged = theirs.min(self.sent);
		}
	}
}

/// Which slot of the ring a command number lives in.
fn slot(sequence: u32) -> usize {
	usize::try_from(sequence % MAX_COMMANDS).expect("a remainder below sixty-four fits a usize")
}

/// Reads one length-prefixed command out of a block.
///
/// @return the text and where the next one starts
fn take(bytes: &[u8], at: usize) -> Result<(&str, usize)> {
	if at + 2 > bytes.len() {
		return Err(err!(Network("a reliable block ends in the middle of a command's length")));
	}

	let length = usize::from(u16_at(bytes, at));
	let start = at + 2;

	if length == 0 || length > MAX_COMMAND {
		return Err(err!(Network("a reliable command of {length} bytes cannot be one")));
	}

	if start + length > bytes.len() {
		return Err(err!(Network(
			"a reliable command says it is {length} bytes and the block is shorter than that"
		)));
	}

	let end = start + length;
	let text = std::str::from_utf8(&bytes[start..end])
		.map_err(|reason| err!(Network("a reliable command is not text: {reason}")))?;

	Ok((text, end))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// One exchange in one direction: what `from` writes, `to` reads.
	fn carry(from: &Reliable, to: &mut Reliable) -> Vec<String> {
		let mut wire = Vec::new();
		let mut out = Vec::new();

		from.write(&mut wire);

		let read = to
			.read(&wire, &mut out)
			.expect("the block is well formed");

		assert_eq!(read, wire.len(), "and the whole of it was read");
		out
	}

	/// A block written by hand, so that a bad one can be built.
	fn block(received: u32, first: u32, commands: &[&str]) -> Vec<u8> {
		let mut out = Vec::new();
		let count = u16::try_from(commands.len()).expect("a handful");

		out.extend_from_slice(&received.to_le_bytes());
		out.extend_from_slice(&first.to_le_bytes());
		out.extend_from_slice(&count.to_le_bytes());

		for text in commands {
			let length = u16::try_from(text.len()).expect("a short one");

			out.extend_from_slice(&length.to_le_bytes());
			out.extend_from_slice(text.as_bytes());
		}

		out
	}

	#[test]
	fn a_queued_command_crosses_and_is_handed_over_once() {
		let (mut host, mut client) = (Reliable::new(), Reliable::new());

		host.queue("game.freeze crate")
			.expect("a command");
		assert_eq!(carry(&host, &mut client), ["game.freeze crate"]);
		assert_eq!(client.received(), 1);
	}

	#[test]
	fn an_unacknowledged_command_is_written_into_every_block_until_it_is_run() {
		let (mut host, mut client) = (Reliable::new(), Reliable::new());

		host.queue("one").expect("a command");
		assert_eq!(host.pending(), 1);
		assert_eq!(carry(&host, &mut client), ["one"], "the first time it is new");
		assert_eq!(carry(&host, &mut client), Vec::<String>::new(), "and after that a repeat");
		assert_eq!(host.pending(), 1, "with nothing having come back yet");

		// the client's own block carries what it has run.
		assert!(carry(&client, &mut host).is_empty());
		assert_eq!(host.pending(), 0, "and that is what retires it");
		assert_eq!(host.acknowledged(), 1);
	}

	#[test]
	fn the_block_of_a_ring_with_nothing_in_it_is_just_its_head() {
		let host = Reliable::new();
		let mut out = Vec::new();

		host.write(&mut out);
		assert_eq!(out.len(), HEAD, "ten bytes and no commands");
		assert_eq!(u32_at(&out, 4), 1, "the next command will be the first");
		assert_eq!(u16_at(&out, 8), 0);
	}

	#[test]
	fn several_commands_arrive_in_the_order_they_were_queued() {
		let (mut host, mut client) = (Reliable::new(), Reliable::new());

		for text in ["one", "two", "three"] {
			host.queue(text).expect("a command");
		}

		assert_eq!(carry(&host, &mut client), ["one", "two", "three"]);
		assert_eq!(client.received(), 3);
	}

	#[test]
	fn a_command_queued_after_an_acknowledgement_still_arrives() {
		let (mut host, mut client) = (Reliable::new(), Reliable::new());

		host.queue("one").expect("a command");
		assert_eq!(carry(&host, &mut client), ["one"]);
		assert!(carry(&client, &mut host).is_empty());
		assert_eq!(host.pending(), 0);

		host.queue("two").expect("a command");
		assert_eq!(carry(&host, &mut client), ["two"], "and only the new one");
	}

	#[test]
	fn the_whole_ring_crosses_at_once() {
		let (mut host, mut client) = (Reliable::new(), Reliable::new());

		for index in 0..MAX_COMMANDS {
			host.queue(&format!("command {index}"))
				.expect("a command");
		}

		assert_eq!(host.pending(), MAX_COMMANDS);

		let got = carry(&host, &mut client);

		assert_eq!(got.len(), 64);
		assert_eq!(got[0], "command 0");
		assert_eq!(got[63], "command 63");
		assert_eq!(client.received(), 64);
	}

	#[test]
	fn a_sixty_fifth_unacknowledged_command_is_refused_rather_than_overwriting_one() {
		let mut host = Reliable::new();

		for index in 0..MAX_COMMANDS {
			host.queue(&format!("command {index}"))
				.expect("a command");
		}

		let error = host
			.queue("one too many")
			.expect_err("the ring is full");

		assert!(error.to_string().starts_with("network: "), "{error}");
		assert_eq!(host.sent(), MAX_COMMANDS, "and the ring is as it was");
	}

	#[test]
	fn the_ring_takes_more_once_the_far_end_has_run_some() {
		let (mut host, mut client) = (Reliable::new(), Reliable::new());

		for index in 0..MAX_COMMANDS {
			host.queue(&format!("command {index}"))
				.expect("a command");
		}

		assert_eq!(carry(&host, &mut client).len(), 64);
		assert!(carry(&client, &mut host).is_empty());
		assert_eq!(host.pending(), 0, "all sixty-four have been run");
		host.queue("the sixty-fifth")
			.expect("there is room now");
		assert_eq!(carry(&host, &mut client), ["the sixty-fifth"]);
	}

	#[test]
	fn a_command_that_is_not_one_is_refused() {
		let mut host = Reliable::new();

		host.queue("")
			.expect_err("an empty line is not a command");
		host.queue(&"x".repeat(MAX_COMMAND + 1))
			.expect_err("nor one past the ceiling");
		host.queue("game.spawn box\ngame.freeze box")
			.expect_err("nor two lines pretending to be one");
		host.queue("game.spawn\tbox")
			.expect_err("nor one with a tab in it");
		assert_eq!(host.sent(), 0, "and none of them took a number");

		host.queue(&"x".repeat(MAX_COMMAND))
			.expect("exactly the ceiling is fine");
		host.queue("game.spawn box")
			.expect("and so is an ordinary one");
	}

	#[test]
	fn a_block_with_no_head_in_it_is_refused() {
		let mut client = Reliable::new();
		let mut out = Vec::new();

		for length in 0..HEAD {
			let bytes = vec![0_u8; length];

			client
				.read(&bytes, &mut out)
				.expect_err("shorter than a head");
		}
	}

	#[test]
	fn a_block_claiming_more_commands_than_can_be_waiting_is_refused() {
		// with the commands actually in it, so that it is the count being
		// refused rather than the block running out of bytes. A peer that
		// supplies them would otherwise hand over more commands at once than
		// its own ring can hold.
		let mut client = Reliable::new();
		let mut out = Vec::new();
		let many = vec!["x"; 65];

		client
			.read(&block(0, 1, &many), &mut out)
			.expect_err("sixty-five is one too many");

		let full = vec!["x"; 64];

		client
			.read(&block(0, 1, &full), &mut out)
			.expect("and sixty-four is a full ring");
		assert_eq!(out.len(), 64, "every one of them handed over");
	}

	#[test]
	fn a_block_that_stops_in_the_middle_of_a_command_is_refused() {
		let mut client = Reliable::new();
		let mut out = Vec::new();
		let whole = block(0, 1, &["a long enough command"]);

		for length in HEAD..whole.len() {
			let mut cut = whole.clone();

			cut.truncate(length);
			assert!(
				client.read(&cut, &mut out).is_err(),
				"a block cut to {length} bytes was read anyway"
			);
		}

		assert_eq!(
			client
				.read(&whole, &mut out)
				.expect("the whole of it"),
			whole.len(),
			"and the whole one reads"
		);
	}

	#[test]
	fn a_command_of_no_bytes_or_too_many_is_refused_on_the_way_in_as_well() {
		let mut client = Reliable::new();
		let mut out = Vec::new();
		let mut empty = block(0, 1, &["x"]);

		empty[HEAD..HEAD + 2].copy_from_slice(&0_u16.to_le_bytes());
		client
			.read(&empty, &mut out)
			.expect_err("a command of nothing");

		let mut huge = block(0, 1, &["x"]);

		huge[HEAD..HEAD + 2].copy_from_slice(&2000_u16.to_le_bytes());
		client
			.read(&huge, &mut out)
			.expect_err("a command of two thousand bytes");
	}

	#[test]
	fn a_block_numbered_from_nil_is_refused() {
		let mut client = Reliable::new();
		let mut out = Vec::new();

		client
			.read(&block(0, 0, &["one", "two"]), &mut out)
			.expect_err("commands are numbered from one");
		assert_eq!(client.received(), 0, "and nothing was taken out of it");
	}

	#[test]
	fn a_block_refused_after_its_head_retires_nothing() {
		// the far end's acknowledgement is only worth acting on if the rest of
		// the block turned out to be a block.
		let mut host = Reliable::new();
		let mut out = Vec::new();

		for text in ["one", "two", "three"] {
			host.queue(text).expect("a command");
		}

		let mut bad = block(9999, 1, &["x"]);

		bad[8..10].copy_from_slice(&65_u16.to_le_bytes());
		host.read(&bad, &mut out)
			.expect_err("sixty-five is one too many");
		assert_eq!(host.acknowledged(), 0, "and it retired none of the three");
		assert_eq!(host.pending(), 3);
	}

	#[test]
	fn a_command_past_the_ceiling_is_refused_with_its_bytes_present() {
		// with the bytes there, so that it is the length being refused rather
		// than the block running out - the same care the count test above it
		// takes.
		let mut client = Reliable::new();
		let mut out = Vec::new();
		let over = "x".repeat(MAX_COMMAND + 1);

		client
			.read(&block(0, 1, &[over.as_str()]), &mut out)
			.expect_err("one byte past the ceiling, bytes and all");

		let longest = "x".repeat(MAX_COMMAND);

		client
			.read(&block(0, 1, &[longest.as_str()]), &mut out)
			.expect("and exactly the ceiling goes through");
		assert_eq!(out, [longest]);
	}

	#[test]
	fn a_command_that_is_not_text_is_refused() {
		let mut client = Reliable::new();
		let mut out = Vec::new();
		let mut bytes = block(0, 1, &["ab"]);

		bytes[HEAD + 2] = 0xFF;
		client
			.read(&bytes, &mut out)
			.expect_err("not every byte string is a string");
	}

	#[test]
	fn the_same_command_arriving_twice_is_run_once() {
		let mut client = Reliable::new();
		let mut out = Vec::new();
		let bytes = block(0, 1, &["one", "two"]);

		client.read(&bytes, &mut out).expect("a block");
		assert_eq!(out, ["one", "two"]);
		client
			.read(&bytes, &mut out)
			.expect("the same block again");
		assert!(out.is_empty(), "both of them have already been run");
		assert_eq!(client.received(), 2);
	}

	#[test]
	fn a_block_that_repeats_one_command_and_adds_another_hands_over_only_the_new_one() {
		let mut client = Reliable::new();
		let mut out = Vec::new();

		client
			.read(&block(0, 1, &["one"]), &mut out)
			.expect("a block");
		assert_eq!(out, ["one"]);
		client
			.read(&block(0, 1, &["one", "two"]), &mut out)
			.expect("a block");
		assert_eq!(out, ["two"], "the first has already been run");
	}

	#[test]
	fn a_far_end_that_has_moved_past_a_command_that_never_arrived_is_refused() {
		// there is no way to ask for it again: the far end only stops resending
		// once this side says it ran it, so a hole here is a hole forever.
		let mut client = Reliable::new();
		let mut out = Vec::new();

		client
			.read(&block(0, 1, &["one"]), &mut out)
			.expect("a block");

		let error = client
			.read(&block(0, 3, &["three"]), &mut out)
			.expect_err("command two never came");

		assert!(error.to_string().contains('2'), "{error}");
	}

	#[test]
	fn an_acknowledgement_past_what_was_ever_sent_is_clamped_rather_than_believed() {
		// without this, a far end saying a large number retires commands that
		// have not been written yet, and they are then never sent at all.
		let mut host = Reliable::new();
		let mut out = Vec::new();

		host.queue("one").expect("a command");
		host.queue("two").expect("a command");
		host.read(&block(9999, 1, &[]), &mut out)
			.expect("a well formed block saying something silly");
		assert_eq!(host.acknowledged(), 2, "only what was actually sent");
		assert_eq!(host.pending(), 0);

		host.queue("three").expect("a command");

		let mut wire = Vec::new();

		host.write(&mut wire);
		assert_eq!(u32_at(&wire, 4), 3, "and the next one still goes out");
		assert_eq!(u16_at(&wire, 8), 1);
	}

	#[test]
	fn an_acknowledgement_that_goes_backwards_is_ignored() {
		let mut host = Reliable::new();
		let mut out = Vec::new();

		host.queue("one").expect("a command");
		host.read(&block(1, 1, &[]), &mut out)
			.expect("a block");
		assert_eq!(host.acknowledged(), 1);
		host.read(&block(0, 1, &[]), &mut out)
			.expect("an older block arriving late");
		assert_eq!(host.acknowledged(), 1, "what has been run has been run");
	}

	#[test]
	fn the_block_says_where_the_rest_of_the_message_begins() {
		let (mut host, mut client) = (Reliable::new(), Reliable::new());
		let mut message = Vec::new();
		let mut out = Vec::new();

		host.queue("game.spawn box").expect("a command");
		host.write(&mut message);
		message.extend_from_slice(b"and then the rest of it");

		let read = client.read(&message, &mut out).expect("a block");

		assert_eq!(out, ["game.spawn box"]);
		assert_eq!(&message[read..], b"and then the rest of it");
	}

	#[test]
	fn both_directions_ride_in_one_block() {
		let (mut host, mut client) = (Reliable::new(), Reliable::new());

		host.queue("from the host").expect("a command");
		client
			.queue("from the client")
			.expect("a command");

		assert_eq!(carry(&host, &mut client), ["from the host"]);
		assert_eq!(carry(&client, &mut host), ["from the client"]);

		// the client's block carried its own command and the acknowledgement of
		// the host's at once.
		assert_eq!(host.acknowledged(), 1);
		assert_eq!(client.received(), 1);
		assert_eq!(host.received(), 1);
		assert_eq!(client.acknowledged(), 0, "the host has not answered yet");

		assert!(carry(&host, &mut client).is_empty());
		assert_eq!(client.acknowledged(), 1);
	}

	#[test]
	fn the_ring_is_indexed_by_the_command_number() {
		assert_eq!(slot(1), 1);
		assert_eq!(slot(64), 0);
		assert_eq!(slot(65), 1, "sixty-four apart is the same slot");

		// which is what makes the ceiling exact: the sixty-four that may be
		// waiting at once are sixty-four different slots.
		let mut slots: Vec<usize> = (1..=MAX_COMMANDS).map(slot).collect();

		slots.sort_unstable();
		slots.dedup();
		assert_eq!(slots.len(), SLOTS, "and no two of them collide");
	}
}
