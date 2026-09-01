//! The socket, and one end of a conversation held over it.
//!
//! ```text
//!   net.receive(now)   everything the wire has, into the channels
//!   ... a step ...
//!   net.send(now, world)   one message to every peer, describing a world
//!                          on the steps a snapshot goes out and not otherwise
//! ```
//!
//! **The socket is read by the frame and written by the step.** Datagrams
//! arrive whenever they arrive, so they are drained once a frame into the
//! channel in front of each peer - the same arrangement a console line and a
//! scene request already have. What goes *out* goes once a step, because a
//! message has to mean "this is where things stand at this moment" and a frame
//! rate is not a moment. Neither half is inside the step body, so `--shot` and
//! `--record` are untouched by all of it, which is the property nothing in this
//! subsystem may disturb.
//!
//! **There is a lying link in front of every peer, always.** With nothing set
//! it hands datagrams straight back, so an ordinary run pays a queue of depth
//! one and one copy; with `net.loss` set it is the only way to find out what
//! everything else here does when the wire is bad. Each peer's link is seeded
//! from the endpoint's own seed and its place in the table, so two peers do not
//! lose the same datagrams and the whole run is still the same run twice.
//!
//! **A message is two blocks and always both.** The reliable ring goes first
//! because it says where what follows it begins, and a snapshot block follows
//! it every single time - even from an endpoint with nothing to describe, and
//! even on a step between snapshots. An empty one is fourteen bytes of head,
//! and the only field in it that matters is what *this* end holds: without it
//! the far end has nothing to write a difference against and every snapshot it
//! sends is the whole world. @ref [`colby_net::Snapshot`].
//!
//! **A world goes out twenty times a second and is stepped sixty.** The
//! cadence is [`colby_net::EVERY`] and the decision is the caller's: `send`
//! describes the world it is handed and says only what it holds when it is
//! handed none. That keeps the rule about *when* out of the endpoint, which has
//! no clock of its own and no idea what a step is.
//!
//! **What arrives is reported, not run.** The reliable ring carries console
//! lines and the console is the whole remote call surface - but nothing here is
//! authenticated, so anything able to reach this socket could otherwise type
//! `quit` at it, or `exec`, or load a scene over the top of what somebody was
//! doing. Deciding who may run what belongs with the world and is a later
//! commit. Until then a command that crosses is logged and handed to whoever
//! asked for it, and goes no further.

use std::{
	cell::RefCell,
	collections::VecDeque,
	io::ErrorKind,
	net::{SocketAddr, ToSocketAddrs, UdpSocket},
	rc::Rc,
	sync::Mutex,
	time::Duration,
};

use colby_core::{
	Result,
	abi::{Bodies, Cvars, Value},
	debug, err, info, warn,
};
use colby_net::{
	Channel, Conditions, Delivery, Link, MAX_DATAGRAM, NOTHING, Reliable, Ring, Slot, Snapshot,
	Solid,
};

/// The port a host listens on when nobody says otherwise.
pub(crate) const DEFAULT_PORT: u16 = 27015;

/// How many datagrams one drain takes off the wire before it gives the frame
/// back.
///
/// A ceiling rather than "everything there is", because everything there is, is
/// a number the far end chooses. Whatever is past this waits for the next
/// frame, which is what a queue is for.
const MAX_DRAIN: usize = 256;

/// How many peers one endpoint talks to at once.
///
/// A ceiling for the same reason: a host is otherwise a list that grows for as
/// long as strangers send it datagrams.
const MAX_PEERS: usize = 8;

/// How long a peer may say nothing before it is forgotten.
const QUIET: Duration = Duration::from_secs(10);

/// The variable that seeds every lying link in this endpoint.
pub(crate) const SEED: &str = "net.seed";

/// The seed a link starts from when nobody says.
pub(crate) const DEFAULT_SEED: u64 = 1;

/// The longest hold anybody may ask a link for, in milliseconds.
///
/// A minute is already an absurd wire. What this is really defending is the
/// constructor that turns it into a span, which refuses anything it cannot
/// hold by ending the process.
const MAX_MILLIS: f32 = 60_000.0;

/// The flag that asks a window to talk to a host.
const CONNECT: &str = "--connect";

/// Reads the command line for a host to talk to.
///
/// Accepts `--connect address` and `--connect=address`, where an address is
/// anything the standard library reads as one - `127.0.0.1:27015`, a name and a
/// port, or a bracketed address of the longer kind.
///
/// @return where the host is, if one was named
#[must_use]
pub(crate) fn wanted() -> Option<SocketAddr> {
	let arguments: Vec<String> = std::env::args().skip(1).collect();

	asked_for(&arguments)
}

/// The same, over arguments already collected.
fn asked_for(arguments: &[String]) -> Option<SocketAddr> {
	for (index, argument) in arguments.iter().enumerate() {
		let text = if let Some(rest) = argument.strip_prefix(&format!("{CONNECT}=")) {
			Some(rest.to_owned())
		} else if argument == CONNECT {
			arguments.get(index + 1).cloned()
		} else {
			continue;
		};

		let Some(text) = text else {
			warn!("{CONNECT} needs an address to connect to");

			return None;
		};

		return match text.to_socket_addrs() {
			| Ok(mut found) => found.next().or_else(|| {
				warn!(%text, "that address is nowhere");

				None
			}),
			| Err(error) => {
				warn!(%text, %error, "that is not an address");

				None
			},
		};
	}

	None
}

/// What a console line asked the wire to do, waiting for a frame to do it.
///
/// The same arrangement scene commands have, and for the same two reasons: a
/// command runs on a thread reading the terminal, and the endpoint it wants is
/// not reachable from a `World` - it is the runner's, like the renderer and the
/// output device. @ref `crate::saves`, which is where this shape came from.
static ASKED: Mutex<Vec<Request>> = Mutex::new(Vec::new());

/// One thing a console line asked of the wire.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Request {
	/// Queue a console line for every peer.
	Say(String),

	/// Report what the endpoint is carrying.
	Status,
}

/// Leaves a request for the next frame.
pub(crate) fn ask(request: Request) {
	if let Ok(mut waiting) = ASKED.lock() {
		waiting.push(request);
	}
}

/// Does whatever was asked, and empties the queue either way.
///
/// Called once a frame, before anything is sent. A request made in a build with
/// no endpoint is answered with a line saying so rather than kept: a command
/// typed at a process that is not on a wire has been answered.
///
/// @param net - the endpoint, if this process has one
pub(crate) fn serve(net: Option<&mut Net>) {
	let asked: Vec<Request> = match ASKED.lock() {
		| Ok(mut waiting) => std::mem::take(&mut waiting),
		| Err(_) => return,
	};

	if asked.is_empty() {
		return;
	}

	let Some(net) = net else {
		info!("this process is not on a wire");

		return;
	};

	for request in asked {
		match request {
			| Request::Say(text) =>
				if let Err(error) = net.say(&text) {
					warn!(%text, %error, "the ring would not take it");
				} else {
					info!(%text, peers = net.peers(), "queued for every peer");
				},
			| Request::Status => report(net),
		}
	}
}

/// Writes a line per peer, and one for the endpoint.
fn report(net: &Net) {
	info!(
		address = %net.address(),
		peers = net.peers(),
		sent = net.sent(),
		delivered = net.delivered(),
		ignored = net.ignored(),
		strangers = net.strangers(),
		forgotten = net.forgotten(),
		refused = net.refused(),
		crowded = net.crowded(),
		"the wire"
	);

	for (index, address) in net.addresses().into_iter().enumerate() {
		let (sent, acknowledged, lost) = net.tally(index);

		info!(
			%address,
			sent,
			acknowledged,
			lost,
			rtt_us = net.rtt(index).as_micros(),
			holding = net.holding(index),
			theirs = net.theirs(index),
			baselines = net.baselines(index),
			"a peer"
		);
	}
}

/// Something that carries datagrams.
///
/// A trait so that a test can put a pair of queues where the socket goes, which
/// is the difference between everything below being checked by running the
/// engine and being checked by running a test. @ref [`Loopback`].
pub(crate) trait Post {
	/// Puts one datagram on the wire.
	///
	/// Cannot fail, deliberately: a datagram a socket refused is a datagram
	/// that was lost, and everything above this already deals with that.
	fn send(&mut self, to: SocketAddr, datagram: &[u8]);

	/// Takes the next datagram that has arrived, if one has.
	///
	/// @param into - where to put it; at least [`MAX_DATAGRAM`] bytes
	/// @return who sent it and how long it is
	fn receive(&mut self, into: &mut [u8]) -> Option<(SocketAddr, usize)>;

	/// Where this endpoint can be reached.
	fn address(&self) -> SocketAddr;

	/// How many datagrams did not go out.
	fn refused(&self) -> u32;
}

/// A real socket, bound and non-blocking.
pub(crate) struct Socket {
	socket: UdpSocket,
	address: SocketAddr,
	refused: u32,
}

impl Socket {
	/// Binds one, on every address this machine has.
	///
	/// @param port - what to listen on; nil takes whatever is free, which is
	/// what a client wants
	pub(crate) fn bind(port: u16) -> Result<Self> {
		let socket = UdpSocket::bind(("0.0.0.0", port))
			.map_err(|error| err!(Network("binding port {port}: {error}")))?;

		socket
			.set_nonblocking(true)
			.map_err(|error| err!(Network("making the socket non-blocking: {error}")))?;

		let address = socket
			.local_addr()
			.map_err(|error| err!(Network("asking the socket its own address: {error}")))?;

		Ok(Self { socket, address, refused: 0 })
	}
}

impl Post for Socket {
	fn send(&mut self, to: SocketAddr, datagram: &[u8]) {
		if let Err(error) = self.socket.send_to(datagram, to) {
			self.refused = self.refused.saturating_add(1);
			debug!(%to, %error, "a datagram did not go out");
		}
	}

	fn receive(&mut self, into: &mut [u8]) -> Option<(SocketAddr, usize)> {
		match self.socket.recv_from(into) {
			| Ok((length, from)) => Some((from, length)),
			| Err(error) if error.kind() == ErrorKind::WouldBlock => None,
			// @note: this arm is not decoration on Windows. A datagram nobody
			// was listening for comes back as an unreachable message, and the
			// system reports it on the *next* read of this socket rather than
			// on the send that caused it - so a host talking to a client that
			// has gone away gets a reset here, and treating it as fatal would
			// be an engine that stops when somebody closes a window. It costs
			// one drain of the frame it lands on.
			| Err(error) => {
				debug!(%error, "a read off the wire failed");

				None
			},
		}
	}

	fn address(&self) -> SocketAddr { self.address }

	fn refused(&self) -> u32 { self.refused }
}

/// The inboxes a set of in-process endpoints share.
///
/// Nothing here delays, loses or reorders anything - the lying link in front of
/// each peer already does all of that, and doing it twice would mean reading
/// every number as half of itself. This is only the part that would otherwise
/// be an operating system.
#[derive(Debug, Default)]
pub(crate) struct Wire {
	boxes: Vec<(SocketAddr, Inbox)>,
	nowhere: u32,
}

/// What is waiting at one address, oldest first.
type Inbox = VecDeque<(SocketAddr, Vec<u8>)>;

impl Wire {
	/// Gives an address an inbox, if it has not got one.
	fn open(&mut self, address: SocketAddr) {
		if !self.boxes.iter().any(|(at, _)| *at == address) {
			self.boxes.push((address, VecDeque::new()));
		}
	}

	/// Puts a datagram in an inbox, or throws it away if nobody is there.
	fn put(&mut self, from: SocketAddr, to: SocketAddr, datagram: &[u8]) {
		let Some((_, inbox)) = self.boxes.iter_mut().find(|(at, _)| *at == to) else {
			self.nowhere = self.nowhere.saturating_add(1);

			return;
		};

		inbox.push_back((from, datagram.to_vec()));
	}

	/// Takes the next datagram waiting for an address.
	fn take(&mut self, at: SocketAddr) -> Option<(SocketAddr, Vec<u8>)> {
		self.boxes
			.iter_mut()
			.find(|(address, _)| *address == at)
			.and_then(|(_, inbox)| inbox.pop_front())
	}

	/// How many datagrams were addressed to nobody at all.
	#[must_use]
	pub(crate) const fn nowhere(&self) -> u32 { self.nowhere }
}

/// One end of an in-process wire.
///
/// What a test puts where the socket goes, and what the two-endpoint run is
/// built on. A datagram is a `Vec` that moves from one side's outbox to the
/// other's inbox with nothing in between, which is the shape the one genuinely
/// checkable library in this field uses - and the reason it has twenty-one test
/// files over its own wire and everybody else has none.
pub(crate) struct Loopback {
	address: SocketAddr,
	wire: Rc<RefCell<Wire>>,
	refused: u32,
}

impl Loopback {
	/// One end, on a wire somebody else may also be on.
	pub(crate) fn at(address: SocketAddr, wire: &Rc<RefCell<Wire>>) -> Self {
		wire.borrow_mut().open(address);

		Self {
			address,
			wire: Rc::clone(wire),
			refused: 0,
		}
	}
}

impl Post for Loopback {
	fn send(&mut self, to: SocketAddr, datagram: &[u8]) {
		self.wire
			.borrow_mut()
			.put(self.address, to, datagram);
	}

	fn receive(&mut self, into: &mut [u8]) -> Option<(SocketAddr, usize)> {
		let (from, datagram) = self.wire.borrow_mut().take(self.address)?;
		let length = datagram.len().min(into.len());

		into[..length].copy_from_slice(&datagram[..length]);
		Some((from, length))
	}

	fn address(&self) -> SocketAddr { self.address }

	fn refused(&self) -> u32 { self.refused }
}

/// What one channel has made of a conversation: sent, acknowledged, lost.
pub(crate) type Tally = (u32, u32, u32);

/// One endpoint's side of a conversation with one other endpoint.
struct Peer {
	address: SocketAddr,
	channel: Channel,
	reliable: Reliable,
	link: Link,
	heard: Duration,

	/// What was told to this peer, to write the next difference against.
	told: Ring,

	/// What was taken *from* it, which is a thing of its own so that reading a
	/// block can borrow it while the block itself is still borrowed from the
	/// channel beside it.
	taken: Taken,
}

/// What one peer has said about the world, and what this end made of it.
struct Taken {
	/// The worlds taken from that peer, to read its next difference against.
	///
	/// The same type the sending half uses, and it has to be: a block is
	/// written against a *numbered* snapshot, and the number it names is a
	/// round trip old whenever snapshots go out faster than one per round
	/// trip - which at twenty a second they do. Most of the time the newest
	/// world would answer the same way, because most fields that changed
	/// since then changed once. The case it would not is a field that changed
	/// and changed *back*, which the writer sees as unchanged and does not
	/// send; a reader holding the value from halfway would keep it forever.
	ring: Ring,

	/// The newest snapshot taken, which is what goes out in the head of
	/// everything sent to this peer.
	holding: u32,

	/// The newest snapshot *it* says it has, which is what the next difference
	/// to it is written against.
	theirs: u32,

	/// Whether this peer has already been told about a world that would not
	/// fit, so that the saying of it happens once rather than twenty times a
	/// second for the rest of the conversation.
	stranded: bool,

	/// How many of the snapshots taken were whole worlds rather than
	/// differences.
	///
	/// The number that says whether any of this is working. A conversation
	/// where the far end never learns what this one holds still *converges* -
	/// a baseline is correct, it is only expensive - so nothing about the
	/// world arriving would look wrong. This is what would look wrong.
	baselines: u32,
}

/// A command that crossed, and who said it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Said {
	/// Which peer said it.
	pub(crate) from: SocketAddr,

	/// The console line, exactly as it was queued.
	pub(crate) text: String,
}

/// One endpoint: a wire, and a conversation with everybody on the far end.
pub(crate) struct Net {
	post: Box<dyn Post>,
	peers: Vec<Peer>,
	/// Whether a datagram from a stranger becomes a peer or is thrown away.
	hosting: bool,
	conditions: Conditions,
	seed: u64,
	scratch: Box<[u8; MAX_DATAGRAM]>,
	payload: Vec<u8>,
	commands: Vec<String>,
	said: Vec<Said>,
	/// Where an arriving snapshot is put together before it is committed.
	///
	/// One buffer for the endpoint rather than one per peer: a message is read
	/// to the end before the next one starts, so nothing here outlives the
	/// block it belongs to.
	applying: Vec<Slot>,
	sent: u32,
	delivered: u32,
	ignored: u32,
	strangers: u32,
	forgotten: u32,
	/// How many times a world was too big to describe, counted once per peer
	/// per attempt rather than once per world.
	crowded: u32,
}

impl Net {
	/// An endpoint over whatever carries its datagrams.
	///
	/// @param post - the wire
	/// @param hosting - whether a datagram from an address this endpoint has
	/// not heard from becomes a peer
	/// @param seed - what the lying links start their chance from
	pub(crate) fn over(post: Box<dyn Post>, hosting: bool, seed: u64) -> Self {
		Self {
			post,
			peers: Vec::new(),
			hosting,
			conditions: Conditions::PERFECT,
			seed,
			scratch: Box::new([0; MAX_DATAGRAM]),
			payload: Vec::new(),
			commands: Vec::new(),
			said: Vec::new(),
			applying: Vec::new(),
			sent: 0,
			delivered: 0,
			ignored: 0,
			strangers: 0,
			forgotten: 0,
			crowded: 0,
		}
	}

	/// An endpoint listening for whoever turns up.
	pub(crate) fn host(port: u16, seed: u64) -> Result<Self> {
		let socket = Socket::bind(port)?;
		let address = socket.address();

		info!(%address, "listening");
		Ok(Self::over(Box::new(socket), true, seed))
	}

	/// An endpoint talking to exactly one other, which it already knows.
	pub(crate) fn connect(address: SocketAddr, seed: u64) -> Result<Self> {
		let socket = Socket::bind(0)?;
		let ours = socket.address();
		let mut net = Self::over(Box::new(socket), false, seed);

		net.add(address, Duration::ZERO);
		info!(%ours, %address, "talking to a host");
		Ok(net)
	}

	/// Starts a conversation with an address, without binding anything.
	///
	/// What [`connect`](Self::connect) does once it has a socket, for the
	/// endpoints that already have a wire - a test, and the two-endpoint run.
	///
	/// @param address - where the far end is
	pub(crate) fn introduce(&mut self, address: SocketAddr) { self.add(address, Duration::ZERO); }

	/// Puts one message on the wire as it stands, with no ring in front of it.
	///
	/// For a test that has to say something a working peer never would.
	///
	/// @param to - which peer
	/// @param payload - the message, whatever it is
	/// @param now - how long this endpoint has been running
	#[cfg(test)]
	fn hand(&mut self, to: SocketAddr, payload: &[u8], now: Duration) {
		let Some(peer) = self
			.peers
			.iter_mut()
			.find(|peer| peer.address == to)
		else {
			return;
		};

		let post = &mut self.post;
		let sent = peer
			.channel
			.send(payload, now, |datagram| post.send(to, datagram));

		assert!(sent.is_ok(), "a test handed over a message that will not fit");
	}

	/// Says how bad every link in front of this endpoint is from now on.
	pub(crate) fn set(&mut self, conditions: Conditions) {
		self.conditions = conditions;

		for peer in &mut self.peers {
			peer.link.set(conditions);
		}
	}

	/// Where this endpoint can be reached.
	pub(crate) fn address(&self) -> SocketAddr { self.post.address() }

	/// How many endpoints this one is talking to.
	pub(crate) fn peers(&self) -> usize { self.peers.len() }

	/// Where each of them is.
	pub(crate) fn addresses(&self) -> Vec<SocketAddr> {
		self.peers
			.iter()
			.map(|peer| peer.address)
			.collect()
	}

	/// What crossed on the last [`receive`](Self::receive).
	pub(crate) fn said(&self) -> &[Said] { &self.said }

	/// How many messages have gone out, counting one per peer.
	pub(crate) const fn sent(&self) -> u32 { self.sent }

	/// How many whole messages have arrived.
	pub(crate) const fn delivered(&self) -> u32 { self.delivered }

	/// How many arriving datagrams were worth nothing.
	pub(crate) const fn ignored(&self) -> u32 { self.ignored }

	/// How many datagrams came from an address that is nobody's.
	pub(crate) const fn strangers(&self) -> u32 { self.strangers }

	/// How many peers went quiet and were forgotten.
	pub(crate) const fn forgotten(&self) -> u32 { self.forgotten }

	/// How many worlds were too big to describe in one snapshot.
	///
	/// @ref `colby_net::MAX_SNAPSHOT`, whose doc says what happens then: a
	/// world with more moving parts than a message can hold is not sent in
	/// halves, it is not sent. This is the count of the times that happened,
	/// and it is the number that says interest management is owed.
	pub(crate) const fn crowded(&self) -> u32 { self.crowded }

	/// What one peer has told this endpoint the world looks like.
	///
	/// Answered out of the ring rather than kept beside it, which is why this
	/// takes the endpoint by exclusive reference for what is plainly a read.
	///
	/// @param index - which peer
	/// @return the world as of [`holding`](Self::holding), by slot
	pub(crate) fn world(&mut self, index: usize) -> &[Slot] {
		let Some(holding) = self
			.peers
			.get(index)
			.map(|peer| peer.taken.holding)
		else {
			return &[];
		};

		self.peers[index].taken.ring.base(holding).0
	}

	/// The newest snapshot one peer says *it* holds.
	///
	/// Named for the peer rather than for this end, because `Peer::told` right
	/// beside it is the opposite direction - what was told *to* that peer -
	/// and the two appear in one log line. What the next difference to it will
	/// be written against.
	pub(crate) fn theirs(&self, index: usize) -> u32 {
		self.peers
			.get(index)
			.map_or(NOTHING, |peer| peer.taken.theirs)
	}

	/// How many whole worlds one peer has had to send, rather than differences.
	pub(crate) fn baselines(&self, index: usize) -> u32 {
		self.peers
			.get(index)
			.map_or(0, |peer| peer.taken.baselines)
	}

	/// The newest snapshot taken from one peer.
	pub(crate) fn holding(&self, index: usize) -> u32 {
		self.peers
			.get(index)
			.map_or(NOTHING, |peer| peer.taken.holding)
	}

	/// How many datagrams the wire itself refused.
	pub(crate) fn refused(&self) -> u32 { self.post.refused() }

	/// The round trip to one peer, as far as anybody knows.
	pub(crate) fn rtt(&self, index: usize) -> Duration {
		self.peers
			.get(index)
			.map_or(Duration::ZERO, |peer| peer.channel.rtt())
	}

	/// What one peer's channel has made of the conversation so far.
	///
	/// @return `(sent, acknowledged, lost)`
	pub(crate) fn tally(&self, index: usize) -> Tally {
		self.peers.get(index).map_or((0, 0, 0), |peer| {
			(peer.channel.sent(), peer.channel.acknowledged(), peer.channel.lost())
		})
	}

	/// Queues a console line for every peer, resent until each of them has it.
	///
	/// @param text - one console line
	pub(crate) fn say(&mut self, text: &str) -> Result {
		for peer in &mut self.peers {
			peer.reliable.queue(text)?;
		}

		Ok(())
	}

	/// Takes everything off the wire and puts it through the channels.
	///
	/// @param now - how long this endpoint has been running
	pub(crate) fn receive(&mut self, now: Duration) {
		self.said.clear();
		self.drain(now);
		self.deliver(now);
		self.forget(now);
	}

	/// Puts one message to every peer on the wire.
	///
	/// @param now - how long this endpoint has been running
	/// @param world - what to describe, or nothing on a step between snapshots
	pub(crate) fn send(&mut self, now: Duration, world: Option<&[Slot]>) {
		for peer in &mut self.peers {
			self.payload.clear();
			// the reliable block first, because it says where what follows it
			// begins; then a snapshot block, always, even when it describes
			// nothing at all.
			peer.reliable.write(&mut self.payload);

			if !describe(peer, world, &mut self.payload) {
				self.crowded = self.crowded.saturating_add(1);
			}

			let address = peer.address;
			let post = &mut self.post;
			let outcome = peer
				.channel
				.send(&self.payload, now, |datagram| post.send(address, datagram));

			if let Err(error) = outcome {
				debug!(%address, %error, "a message could not be cut into datagrams");

				continue;
			}

			self.sent = self.sent.saturating_add(1);
		}
	}

	/// Off the wire and into the link in front of whoever sent it.
	///
	/// @note: the peer table sits *in front of* the lying link, so a datagram
	/// the link is about to eat has still been heard from - it makes a stranger
	/// into a peer and it keeps an existing one from going quiet. That is not
	/// what a genuinely lossy wire would do, and it is the price of a link per
	/// peer: to put a datagram in front of the right link you have to know
	/// whose it is first. It only shows with the link turned on, which is a
	/// tool rather than a wire, and the alternative is one link for the whole
	/// endpoint carrying addresses through it.
	fn drain(&mut self, now: Duration) {
		for _ in 0..MAX_DRAIN {
			let Some((from, length)) = self.post.receive(self.scratch.as_mut_slice()) else {
				break;
			};

			let Some(index) = self.find(from, now) else {
				self.strangers = self.strangers.saturating_add(1);

				continue;
			};

			self.peers[index].heard = now;
			self.peers[index]
				.link
				.receive(&self.scratch[..length], now);
		}
	}

	/// Out of each link, through each channel, and into what crossed.
	fn deliver(&mut self, now: Duration) {
		let mut broken = Vec::new();

		for index in 0..self.peers.len() {
			if !self.hand_over(index, now) {
				broken.push(index);
			}
		}

		// @note: a peer whose block does not parse is dropped rather than
		// argued with, which is what the ring says is the only thing to do with
		// one. It is not tidiness: a block refused part way through has already
		// counted the commands before the fault as run, so the numbering on
		// this side has moved and everything that peer says afterwards is
		// either a repeat or a hole. Talking to it again would silently swallow
		// its next real command.
		for index in broken.into_iter().rev() {
			let address = self.peers.remove(index).address;

			self.forgotten = self.forgotten.saturating_add(1);
			warn!(%address, "a peer that is not talking sense");
		}
	}

	/// Everything one peer's link has ready, through that peer's channel.
	///
	/// Its own function rather than the body of the loop above, because a match
	/// inside a loop inside a loop inside a function is one level deeper than
	/// the house rules allow - and they are right about this one too.
	///
	/// @return whether the peer is still talking sense
	fn hand_over(&mut self, index: usize, now: Duration) -> bool {
		while let Some(sense) = self.next_message(index, now) {
			if !sense {
				return false;
			}
		}

		true
	}

	/// One datagram out of a peer's link and through its channel.
	///
	/// @return nothing when the link has nothing ready, and otherwise whether
	/// what came out of it was something a working peer would have said
	fn next_message(&mut self, index: usize, now: Duration) -> Option<bool> {
		let peer = &mut self.peers[index];
		let datagram = peer.link.poll(now)?;

		match peer.channel.receive(datagram, now) {
			| Delivery::Message(message) => {
				let read = peer.reliable.read(message, &mut self.commands);
				let from = peer.address;

				self.delivered = self.delivered.saturating_add(1);

				let used = read.as_ref().ok().copied().unwrap_or(0);

				if !take(from, read, &mut self.commands, &mut self.said) {
					return Some(false);
				}

				// the ring said how far it read, so the rest of the message is
				// the snapshot block. `taken` and `channel` are different
				// fields, which is what lets this borrow one while the block
				// is still borrowed from the other.
				let rest = message.get(used..).unwrap_or(&[]);

				Some(absorb(&mut peer.taken, rest, &mut self.applying))
			},
			| Delivery::Fragment => Some(true),
			| Delivery::Ignored(_) => {
				self.ignored = self.ignored.saturating_add(1);

				Some(true)
			},
		}
	}

	/// Forgets whoever has gone quiet.
	fn forget(&mut self, now: Duration) {
		let before = self.peers.len();

		self.peers
			.retain(|peer| now.saturating_sub(peer.heard) < QUIET);

		let gone = u32::try_from(before.saturating_sub(self.peers.len())).unwrap_or(0);

		if gone > 0 {
			self.forgotten = self.forgotten.saturating_add(gone);
			info!(gone, "a peer went quiet");
		}
	}

	/// Which peer an address is, adding one if this endpoint is hosting.
	fn find(&mut self, address: SocketAddr, now: Duration) -> Option<usize> {
		if let Some(index) = self
			.peers
			.iter()
			.position(|peer| peer.address == address)
		{
			return Some(index);
		}

		if !self.hosting || self.peers.len() >= MAX_PEERS {
			return None;
		}

		self.add(address, now);
		Some(self.peers.len().saturating_sub(1))
	}

	/// Starts a conversation with an address.
	///
	/// @note: a *second* conversation with an address that already had one is
	/// this same function, and everything it makes starts over - the channel's
	/// sequence at one, both rings numbering from one. The far end does not
	/// know, so it is still counting from where the last conversation reached
	/// and still holds its ring. Nothing here can tell the two apart, because
	/// nothing on the wire says which conversation a datagram belongs to.
	/// That is the connection handshake's job and it is not written yet;
	/// @ref the module's note on nothing being authenticated. Until it is, a
	/// peer that is dropped and comes back is a peer that will not converge.
	fn add(&mut self, address: SocketAddr, now: Duration) {
		// a seed of its own, so that two peers do not lose the same datagrams
		// while the run as a whole is still the same run twice.
		let seed = self
			.seed
			.wrapping_mul(0x9E37_79B9_7F4A_7C15)
			.wrapping_add(u64::try_from(self.peers.len()).unwrap_or(0));
		let mut link = Link::new(seed);

		link.set(self.conditions);
		self.peers.push(Peer {
			address,
			channel: Channel::new(),
			reliable: Reliable::new(),
			link,
			heard: now,
			told: Ring::new(),
			taken: Taken {
				ring: Ring::new(),
				holding: NOTHING,
				theirs: NOTHING,
				baselines: 0,
				stranded: false,
			},
		});
		info!(%address, "a peer");
	}
}

/// The world as a table of records, for a snapshot to describe.
///
/// A slot for every place the body table has, occupied where a body is, so
/// that a slot number means the same thing on both machines without either of
/// them saying so. That is the whole of what ties a snapshot to a world.
///
/// @param bodies - the body table as the solver left it
/// @param into - where the records go, emptied first
pub(crate) fn records(bodies: &Bodies, into: &mut Vec<Slot>) {
	into.clear();
	into.resize(bodies.slots(), None);

	for (id, body) in bodies.iter() {
		let Some(slot) = into.get_mut(id.slot()) else {
			continue;
		};

		*slot = Some((id.generation(), Solid::of(body)));
	}
}

/// Writes the snapshot block that follows the reliable one.
///
/// There is always a block. With a world it is a difference from whatever the
/// peer last said it had; without one it is a head saying what this end holds
/// and nothing else, which is the field the far end needs to stop sending
/// baselines forever.
///
/// @param peer - whose conversation this is
/// @param world - what to describe, or nothing
/// @param out - the message so far, appended
/// @return whether the world fitted, which is the one way this can be asked
/// for something it cannot do
fn describe(peer: &mut Peer, world: Option<&[Slot]>, out: &mut Vec<u8>) -> bool {
	let Some(world) = world else {
		nothing(peer.taken.holding, out);

		return true;
	};

	let number = peer.told.next();
	let holding = peer.taken.holding;
	let (base, against) = peer.told.base(peer.taken.theirs);

	if Snapshot::write(number, against, holding, base, world, out).is_err() {
		// the world does not fit in one message and there is no continuation:
		// what goes out instead is the head, so the far end still hears what
		// this end holds and is not dropped for saying nothing.
		//
		// This is not a hiccup. Nothing was kept, so the number does not move
		// and the peer never learns of a base to answer with - which means the
		// next attempt is the same baseline and fails the same way, forever.
		// A counter that names nobody is not enough to find that with.
		if !peer.taken.stranded {
			peer.taken.stranded = true;
			warn!(
				address = %peer.address,
				bodies = world.len(),
				"a world too big to describe in one snapshot, and there is no continuation - \
				 this peer will be told nothing further until it shrinks"
			);
		}

		nothing(holding, out);

		return false;
	}

	peer.taken.stranded = false;

	peer.told.keep(number, world);
	true
}

/// A block that describes nothing and says only what this end holds.
fn nothing(holding: u32, out: &mut Vec<u8>) {
	let written = Snapshot::write(NOTHING, NOTHING, holding, &[], &[], out);

	debug_assert!(written.is_ok(), "a block of nothing but a head always fits");
}

/// Reads the snapshot block at the end of a message into what a peer holds.
///
/// @param taken - what this end has made of that peer's side of it
/// @param bytes - the block, and nothing is expected after it
/// @param applying - the endpoint's scratch, where the world is put together
/// @return whether it was something a working peer would have sent
fn absorb(taken: &mut Taken, bytes: &[u8], applying: &mut Vec<Slot>) -> bool {
	let Ok((number, against, holding)) = Snapshot::peek(bytes) else {
		return false;
	};

	// what the far end holds is news whether or not it described anything,
	// and it is the only news in a block sent between snapshots.
	taken.theirs = holding;

	// @note: dropping this changes nothing that can be observed, because the
	// walk below would find no records, `keep` would refuse nought and what is
	// held would come back unchanged. It is here so that the message between
	// snapshots - which is two messages in three - does not touch the ring at
	// all, and a mutation pass will report its removal surviving forever.
	if number == NOTHING {
		return true;
	}

	// against the world it names rather than the newest one held. `base`
	// answers with nothing at all for a number that has fallen out of the
	// ring, which is what makes a baseline land on an empty world.
	let (base, _) = taken.ring.base(against);

	applying.clear();
	applying.extend_from_slice(base);

	let Ok((snapshot, _)) = Snapshot::read(applying, bytes) else {
		return false;
	};

	for change in &snapshot.changes {
		let slot = usize::from(change.slot);

		if applying.len() <= slot {
			applying.resize(slot + 1, None);
		}

		applying[slot] = change
			.solid
			.map(|solid| (change.generation, solid));
	}

	// whether the ring *took* it, which is not the same question as whether
	// the ring has it: a peer can name a snapshot old enough to have been
	// refused and still be sitting in its place, and taking that in would walk
	// what this end holds backwards. The scratch above is scratch either way.
	if !taken.ring.keep(number, applying) {
		return true;
	}

	taken.holding = number;

	if against == NOTHING {
		taken.baselines = taken.baselines.saturating_add(1);
	}

	true
}

/// Takes what a block turned into, or says why it was nothing.
///
/// Its own function because the branch it holds would otherwise sit four deep
/// inside two loops and a match, which is the shape the house rules refuse -
/// and they are right to: what this does is one sentence and it reads as one
/// here.
///
/// @param from - who said it
/// @param read - what the ring made of the block
/// @param commands - what came out of it, emptied here
/// @param said - where it goes
/// @return whether the block was one at all
fn take(
	from: SocketAddr,
	read: Result<usize>,
	commands: &mut Vec<String>,
	said: &mut Vec<Said>,
) -> bool {
	if let Err(error) = read {
		debug!(%from, %error, "a peer said something that is not a block");
		// whatever was parsed out of it before the fault goes with it: it was
		// already counted as run on this side, and handing it over as well
		// would be running half of a message nobody can trust.
		//
		// @note: nothing can observe this, because the ring empties what it is
		// given before it fills it and the peer this came from is dropped a
		// moment later either way. It is kept because it is the one line that
		// says the leftovers are not somebody else's.
		commands.clear();

		return false;
	}

	for text in commands.drain(..) {
		info!(%from, %text, "a command crossed");
		said.push(Said { from, text });
	}

	true
}

/// How bad the wire is, as the console table has it.
///
/// A function of the table and nothing else, which is what makes it the piece
/// of this wiring a test can reach - the same split the volumes have and for
/// the same reason. A variable that is missing or is not a number reads as a
/// perfect wire rather than a broken one: a typo should not quietly start
/// losing datagrams.
///
/// @param cvars - the table the six were registered into
pub(crate) fn conditions(cvars: &Cvars) -> Conditions {
	let share = |name: &str| cvars.float(name).unwrap_or(0.0);
	let span = |name: &str| {
		let millis = cvars.float(name).unwrap_or(0.0);

		// through a double rather than a single, so that the fifty milliseconds
		// somebody typed comes out as fifty rather than fifty and a bit - a
		// f32 second cannot hold a round number of milliseconds and the far end
		// of this is a hold measured in nanoseconds.
		//
		// Clamped at both ends because the constructor below *panics* on a
		// value that is negative, is not a number, or is too large to hold, and
		// all three of those are things a console can be handed.
		let held = if millis.is_nan() {
			0.0
		} else {
			millis.clamp(0.0, MAX_MILLIS)
		};

		Duration::from_secs_f64(f64::from(held) / 1000.0)
	};

	Conditions {
		lag: span(colby_net::LAG),
		jitter: span(colby_net::JITTER),
		loss: share(colby_net::LOSS),
		burst: cvars.float(colby_net::BURST).unwrap_or(1.0),
		duplicate: share(colby_net::DUPLICATE),
	}
}

/// What the lying links start their chance from, as the table has it.
///
/// Nil, or anything that is not a positive number, is [`DEFAULT_SEED`] - so a
/// run nobody configured is still the same run twice.
pub(crate) fn seed(cvars: &Cvars) -> u64 {
	let asked = cvars.float(SEED).unwrap_or(0.0);

	if asked.is_nan() || asked <= 0.0 {
		return DEFAULT_SEED;
	}

	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "a positive float into an integer, where there is no fallible form and the \
		          answer is a seed rather than a measurement"
	)]
	let seed = asked as u64;

	seed
}

/// Registers everything the wire answers for.
///
/// None of it is archived. A link somebody made lossy to look at something is a
/// tool, and coming back to a session that is still dropping a tenth of
/// everything is a surprise rather than a setting - the same argument the debug
/// renderer's variables are not archived under.
///
/// @param cvars - the table to register into
pub(crate) fn install(cvars: &mut Cvars) {
	cvars.var(
		colby_net::LAG,
		Value::Float(0.0),
		"hold every arriving datagram this many milliseconds, one way",
	);
	cvars.var(
		colby_net::JITTER,
		Value::Float(0.0),
		"vary that hold by up to this many milliseconds either way",
	);
	cvars.var(
		colby_net::LOSS,
		Value::Float(0.0),
		"throw away this share of arriving datagrams, nil through one",
	);
	cvars.var(
		colby_net::BURST,
		Value::Float(1.0),
		"how many times longer a run of losses is than chance alone makes it",
	);
	cvars.var(
		colby_net::DUPLICATE,
		Value::Float(0.0),
		"deliver this share of arriving datagrams twice",
	);
	cvars.var(
		SEED,
		Value::Float(0.0),
		"what the lying links start their chance from; nil is the usual one",
	);
}

#[cfg(test)]
mod tests {
	use std::net::{IpAddr, Ipv4Addr};

	use colby_net::MAX_COMMAND;

	use super::*;

	/// An address nobody has to be able to reach.
	fn somewhere(port: u16) -> SocketAddr {
		SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
	}

	/// A host and a client on one wire, the client already pointed at the host.
	fn two() -> (Net, Net, Rc<RefCell<Wire>>) {
		let wire = Rc::new(RefCell::new(Wire::default()));
		let (one, two) = (somewhere(1), somewhere(2));
		let host = Net::over(Box::new(Loopback::at(one, &wire)), true, 1);
		let mut client = Net::over(Box::new(Loopback::at(two, &wire)), false, 2);

		client.introduce(one);
		(host, client, wire)
	}

	/// One round: both send, both take off the wire.
	fn round(host: &mut Net, client: &mut Net, step: u32) {
		let now = colby_core::time::STEP * step;

		host.send(now, None);
		client.send(now, None);
		host.receive(now);
		client.receive(now);
	}

	/// A table with the six the wire answers for in it.
	fn table() -> Cvars {
		let mut cvars = Cvars::new();

		install(&mut cvars);
		cvars
	}

	/// A world of two bodies, the first of them somewhere given.
	fn world(along: f32) -> Vec<Slot> {
		let solid = Solid {
			position: [along, 1.0, 2.0],
			scale: [1.0, 1.0, 1.0],
			kind: 2,
			..Solid::default()
		};

		vec![Some((1, solid)), None, Some((3, Solid::default()))]
	}

	#[test]
	fn a_snapshot_is_read_against_the_world_it_names_and_not_the_newest_one() {
		// the property the whole receiving-side ring exists for, and the one
		// nothing but the recorded digest could see before this.
		//
		// A host writes against what a peer *said* it had, which is a round
		// trip old. If a field changed and changed back in between, the writer
		// sees it as unchanged and does not send it - so a reader decoding
		// against the newest world it holds would keep the value from halfway
		// and never be told otherwise. Every other case is forgiving; this one
		// is not.
		let mut taken = Taken {
			ring: Ring::new(),
			holding: NOTHING,
			theirs: NOTHING,
			baselines: 0,
			stranded: false,
		};
		let mut applying = Vec::new();
		let one = world(1.0);
		let two = world(2.0);
		let mut out = Vec::new();

		// snapshot one, a baseline.
		Snapshot::write(1, NOTHING, NOTHING, &[], &one, &mut out).expect("it fits");
		assert!(absorb(&mut taken, &out, &mut applying));

		// snapshot two, moved away, taken against one.
		out.clear();
		Snapshot::write(2, 1, NOTHING, &one, &two, &mut out).expect("it fits");
		assert!(absorb(&mut taken, &out, &mut applying));
		assert_eq!(taken.holding, 2);

		// snapshot three, back where it started - and written against ONE,
		// because that is the last the sender heard of. Nothing about the
		// body is in this block at all.
		out.clear();

		let spoken = Snapshot::write(3, 1, NOTHING, &one, &one, &mut out).expect("it fits");

		assert_eq!(spoken, 0, "the sender sees nothing to say, which is the whole trap");
		assert!(absorb(&mut taken, &out, &mut applying));
		assert_eq!(taken.holding, 3);
		assert_eq!(
			taken.ring.base(3).0,
			one.as_slice(),
			"so the reader has to have gone back to the world the block named"
		);
	}

	#[test]
	fn a_snapshot_older_than_what_is_held_does_not_walk_it_backwards() {
		// a peer may put any number in a block it likes, and the channel only
		// refuses a *message* that arrives behind a newer one. A number the
		// ring has already passed is not news, and taking it in would make
		// this end advertise a world two snapshots stale.
		let mut taken = Taken {
			ring: Ring::new(),
			holding: NOTHING,
			theirs: NOTHING,
			baselines: 0,
			stranded: false,
		};
		let mut applying = Vec::new();
		let mut out = Vec::new();

		for number in 1..=3 {
			out.clear();
			Snapshot::write(number, NOTHING, NOTHING, &[], &world(1.0), &mut out).expect("fits");
			assert!(absorb(&mut taken, &out, &mut applying));
		}

		assert_eq!(taken.holding, 3);
		assert_eq!(taken.baselines, 3);

		// and now snapshot two again, which the ring still has a place for.
		out.clear();
		Snapshot::write(2, NOTHING, NOTHING, &[], &world(9.0), &mut out).expect("it fits");

		assert!(absorb(&mut taken, &out, &mut applying), "it is not a fault, only old news");
		assert_eq!(taken.holding, 3, "and what this end holds did not move");
		assert_eq!(taken.baselines, 3, "nor did the count of worlds it was sent whole");
	}

	#[test]
	fn the_world_is_taken_down_with_its_holes_and_its_generations() {
		use colby_core::{
			abi::{Bodies, Body, Transform},
			glam::Vec3,
		};

		let mut bodies = Bodies::new();
		let first = bodies.spawn(Body::default());
		let one = bodies.spawn(Body::default());
		let two = bodies.spawn(Body::default());

		// a slot emptied and filled again, so that the table holds a body whose
		// generation is not one. Every body in a fresh table is generation one,
		// and a fixture made only of those cannot tell a generation from a
		// constant.
		bodies.despawn(one);
		bodies.despawn(two);

		let again = bodies.spawn(Body {
			transform: Transform {
				position: Vec3::new(5.0, 0.0, 0.0),
				..Transform::default()
			},
			..Body::default()
		});
		let hole = if again.slot() == one.slot() { two } else { one };

		assert!(again.generation() > 1, "the reused slot is on its second occupant");

		let mut world = Vec::new();

		records(&bodies, &mut world);

		assert_eq!(world.len(), bodies.slots(), "a slot for every place the table has");
		assert!(world[first.slot()].is_some());
		assert!(world[hole.slot()].is_none(), "and a hole where a body was taken away");
		assert_eq!(
			world[again.slot()].map(|(generation, _)| generation),
			Some(again.generation()),
			"each carrying the generation that says which occupant it is"
		);
		assert_eq!(world[again.slot()].map(|(_, solid)| solid.position), Some([5.0, 0.0, 0.0]));
	}

	#[test]
	fn a_message_with_no_snapshot_block_after_the_ring_is_refused() {
		// the head is read before anything is chosen to read against, so a
		// peer that stops after its reliable block hands the reader four bytes
		// that are not there. It is refused, and the peer goes with it.
		let wire = Rc::new(RefCell::new(Wire::default()));
		let (one, two) = (somewhere(1), somewhere(2));
		let mut host = Net::over(Box::new(Loopback::at(one, &wire)), true, 1);
		let mut liar = Net::over(Box::new(Loopback::at(two, &wire)), false, 2);

		liar.introduce(one);

		// a reliable block that parses, owing nothing, and then the end of the
		// message.
		let mut bare = Vec::new();

		bare.extend_from_slice(&0_u32.to_le_bytes());
		bare.extend_from_slice(&1_u32.to_le_bytes());
		bare.extend_from_slice(&0_u16.to_le_bytes());

		liar.hand(one, &bare, Duration::ZERO);
		host.receive(Duration::ZERO);

		assert_eq!(host.peers(), 0, "a peer that sends half a message is not talked to again");
		assert_eq!(host.forgotten(), 1);
	}

	#[test]
	fn a_snapshot_block_that_is_nonsense_takes_the_peer_with_it() {
		// the same, for a block with a head that reads and a body that does
		// not: a count of one and no record after it.
		let wire = Rc::new(RefCell::new(Wire::default()));
		let (one, two) = (somewhere(1), somewhere(2));
		let mut host = Net::over(Box::new(Loopback::at(one, &wire)), true, 1);
		let mut liar = Net::over(Box::new(Loopback::at(two, &wire)), false, 2);

		liar.introduce(one);

		let mut broken = Vec::new();

		broken.extend_from_slice(&0_u32.to_le_bytes());
		broken.extend_from_slice(&1_u32.to_le_bytes());
		broken.extend_from_slice(&0_u16.to_le_bytes());
		// the snapshot block: number one, against nothing, holding nothing,
		// and one record that is not there.
		broken.extend_from_slice(&1_u32.to_le_bytes());
		broken.extend_from_slice(&NOTHING.to_le_bytes());
		broken.extend_from_slice(&NOTHING.to_le_bytes());
		broken.extend_from_slice(&1_u16.to_le_bytes());

		liar.hand(one, &broken, Duration::ZERO);
		host.receive(Duration::ZERO);

		assert_eq!(host.peers(), 0);
		assert_eq!(host.forgotten(), 1);
	}

	#[test]
	fn each_end_says_what_it_holds_and_neither_says_the_others() {
		// both ends describing, which is the only arrangement where the two
		// fields can be told apart: a block carries what its *sender* has
		// taken, and a reader that echoed what it had been told would look
		// exactly the same on a wire where only one end ever describes.
		let (mut host, mut client, _wire) = two();

		round(&mut host, &mut client, 1);

		for step in 2..6 {
			let now = colby_core::time::STEP * step;

			host.send(now, Some(&world(1.0)));
			client.send(now, Some(&world(2.0)));
			host.receive(now);
			client.receive(now);
		}

		// each block goes out before that step's arrivals are taken in, so what
		// a block says its sender holds is exactly one snapshot behind what
		// the sender holds by the end of the step. That is the lower bound the
		// ring's doc is about, and here it is as a number.
		assert!(host.holding(0) > 0, "the host took the client's worlds");
		assert_eq!(host.theirs(0) + 1, client.holding(0), "a round behind, and no further");
		assert_eq!(client.theirs(0) + 1, host.holding(0), "each way round");
		assert_ne!(host.theirs(0), host.holding(0), "and the two fields are not one field");
	}

	#[test]
	fn a_world_the_host_describes_is_the_world_the_client_ends_up_with() {
		let (mut host, mut client, _wire) = two();

		round(&mut host, &mut client, 1);

		for step in 2..8 {
			let now = colby_core::time::STEP * step;

			host.send(now, Some(&world(f32::from(step_of(step)))));
			client.send(now, None);
			host.receive(now);
			client.receive(now);
		}

		assert_eq!(client.holding(0), 6, "six snapshots, and it holds the last of them");
		assert_eq!(client.world(0), world(7.0).as_slice(), "which is the world it was sent");
	}

	#[test]
	fn a_message_between_snapshots_still_says_what_this_end_holds() {
		// the field that flows the other way. Without it a host never learns
		// that a client took anything, so every snapshot it writes is a
		// baseline - which works, and costs the whole world every time.
		let (mut host, mut client, _wire) = two();

		round(&mut host, &mut client, 1);

		let now = colby_core::time::STEP * 2;

		host.send(now, Some(&world(1.0)));
		client.send(now, None);
		host.receive(now);
		client.receive(now);

		assert_eq!(client.holding(0), 1, "the client took the first snapshot");

		// and now a step with no world in it at all, which is where the news
		// has to travel.
		let later = colby_core::time::STEP * 3;

		host.send(later, None);
		client.send(later, None);
		host.receive(later);
		client.receive(later);

		assert_eq!(host.theirs(0), 1, "the host has been told which one the client has");
		assert_eq!(host.holding(0), NOTHING, "while the client has described nothing to it");
	}

	/// Which snapshot number a step of the loop above produces.
	fn step_of(step: u32) -> u16 { u16::try_from(step).unwrap_or(0) }

	#[test]
	fn a_client_finds_a_host_and_a_host_finds_out_who_turned_up() {
		let (mut host, mut client, _wire) = two();

		assert_eq!(host.peers(), 0, "a host has heard from nobody");
		assert_eq!(client.peers(), 1, "and a client knows who it came for");

		round(&mut host, &mut client, 1);

		assert_eq!(host.peers(), 1, "and then it has");
		assert_eq!(host.addresses(), vec![somewhere(2)]);
	}

	#[test]
	fn a_command_crosses_both_ways_and_is_handed_over_once() {
		let (mut host, mut client, _wire) = two();

		round(&mut host, &mut client, 1);
		host.say("host.hello").expect("the ring takes it");
		client
			.say("client.hello")
			.expect("the ring takes it");
		round(&mut host, &mut client, 2);

		assert_eq!(client.said().len(), 1);
		assert_eq!(client.said()[0].text, "host.hello");
		assert_eq!(client.said()[0].from, somewhere(1));
		assert_eq!(host.said().len(), 1);
		assert_eq!(host.said()[0].text, "client.hello");

		round(&mut host, &mut client, 3);
		assert!(host.said().is_empty(), "and not again on the next round");
		assert!(client.said().is_empty());
	}

	#[test]
	fn a_ring_that_has_filled_refuses_rather_than_forgetting_a_command() {
		// nothing the host says can be acknowledged, so the ring fills. The
		// sixty-fifth has to be an error rather than the first one quietly
		// going missing.
		let (mut host, mut client, _wire) = two();

		round(&mut host, &mut client, 1);
		host.set(Conditions { loss: 1.0, ..Conditions::PERFECT });
		client.set(Conditions { loss: 1.0, ..Conditions::PERFECT });

		for index in 0..colby_net::MAX_COMMANDS {
			host.say(&format!("tick {index}"))
				.expect("the ring takes sixty-four");
			round(&mut host, &mut client, 2 + index);
		}

		host.say("one too many")
			.expect_err("and refuses the next");
	}

	#[test]
	fn a_command_that_will_not_fit_the_ring_is_refused_rather_than_dropped() {
		let (mut host, mut client, _wire) = two();

		round(&mut host, &mut client, 1);
		host.say(&"x".repeat(MAX_COMMAND + 1))
			.expect_err("one byte past what a command may be");
	}

	#[test]
	fn a_datagram_the_channel_refuses_is_counted() {
		// a wire that says a thing twice is the cheapest way to make a channel
		// refuse one: the second copy is a message it has already handed over.
		let (mut host, mut client, _wire) = two();
		let twice = Conditions { duplicate: 1.0, ..Conditions::PERFECT };

		client.set(twice);
		round(&mut host, &mut client, 1);
		round(&mut host, &mut client, 2);

		assert!(client.ignored() > 0, "the repeat was refused and counted");
		assert!(client.delivered() > 0, "and the first of each pair was not");
	}

	#[test]
	fn a_block_that_is_not_one_leaves_nothing_behind_for_the_next_message() {
		// a peer that is broken or lying sends something that is not a block.
		// What must not happen is that whatever was parsed out of it before the
		// fault is handed over as though the next message had said it.
		let wire = Rc::new(RefCell::new(Wire::default()));
		let (one, two) = (somewhere(1), somewhere(2));
		let mut host = Net::over(Box::new(Loopback::at(one, &wire)), true, 1);
		let mut liar = Net::over(Box::new(Loopback::at(two, &wire)), false, 2);

		liar.introduce(one);

		// a block claiming two commands and carrying one and a half.
		let mut broken = Vec::new();

		broken.extend_from_slice(&0_u32.to_le_bytes());
		broken.extend_from_slice(&1_u32.to_le_bytes());
		broken.extend_from_slice(&2_u16.to_le_bytes());
		broken.extend_from_slice(&4_u16.to_le_bytes());
		broken.extend_from_slice(b"good");
		broken.extend_from_slice(&9_u16.to_le_bytes());
		broken.extend_from_slice(b"cut");

		liar.hand(one, &broken, Duration::ZERO);
		host.receive(Duration::ZERO);

		assert!(host.said().is_empty(), "a block that does not parse says nothing");
		assert_eq!(host.peers(), 0, "and the peer that sent it is not talked to again");
		assert_eq!(host.forgotten(), 1);
	}

	#[test]
	fn a_datagram_longer_than_the_buffer_it_is_read_into_is_cut_down() {
		// nothing in this file passes a short buffer, but the wire is the piece
		// a test substitutes and a caller that got this wrong would be a panic
		// rather than a wrong answer.
		let wire = Rc::new(RefCell::new(Wire::default()));
		let (one, two) = (somewhere(1), somewhere(2));
		let mut sender = Loopback::at(one, &wire);
		let mut reader = Loopback::at(two, &wire);

		sender.send(two, &[7; 40]);

		let mut small = [0_u8; 8];
		let got = reader.receive(&mut small);

		assert_eq!(got, Some((one, 8)), "as much of it as there was room for");
		assert_eq!(small, [7; 8]);
	}

	#[test]
	fn a_client_does_not_take_a_stranger_for_a_peer() {
		let wire = Rc::new(RefCell::new(Wire::default()));
		let (one, two, three) = (somewhere(1), somewhere(2), somewhere(3));
		let mut client = Net::over(Box::new(Loopback::at(two, &wire)), false, 1);
		let mut stranger = Net::over(Box::new(Loopback::at(three, &wire)), false, 2);

		client.introduce(one);
		stranger.introduce(two);
		stranger.send(Duration::ZERO, None);
		client.receive(Duration::ZERO);

		assert_eq!(client.peers(), 1, "still only the host it came for");
		assert_eq!(client.strangers(), 1, "and it counted the one that turned up");
	}

	#[test]
	fn a_host_takes_a_stranger_for_a_peer_and_stops_at_the_ceiling() {
		let wire = Rc::new(RefCell::new(Wire::default()));
		let host_at = somewhere(1);
		let mut host = Net::over(Box::new(Loopback::at(host_at, &wire)), true, 1);
		let mut clients: Vec<Net> = (0..MAX_PEERS + 2)
			.map(|index| {
				let at = somewhere(u16::try_from(100 + index).unwrap_or(100));
				let mut client = Net::over(Box::new(Loopback::at(at, &wire)), false, 1);

				client.introduce(host_at);
				client
			})
			.collect();

		for client in &mut clients {
			client.send(Duration::ZERO, None);
		}

		host.receive(Duration::ZERO);

		assert_eq!(host.peers(), MAX_PEERS, "no more than the ceiling");
		assert_eq!(host.strangers(), 2, "and the two past it were counted");
	}

	#[test]
	fn a_peer_that_goes_quiet_is_forgotten() {
		let (mut host, mut client, _wire) = two();

		round(&mut host, &mut client, 1);
		assert_eq!(host.peers(), 1);

		// it was last heard from at the first step rather than at nil, so the
		// ceiling falls a step later than it would from nothing.
		let heard = colby_core::time::STEP;

		host.receive(
			heard
				.saturating_add(QUIET)
				.saturating_sub(Duration::from_millis(1)),
		);
		assert_eq!(host.peers(), 1, "a moment before the ceiling it is still there");

		host.receive(heard.saturating_add(QUIET));
		assert_eq!(host.peers(), 0, "and on it, it is not");
		assert_eq!(host.forgotten(), 1);
	}

	#[test]
	fn a_datagram_addressed_to_nobody_is_thrown_away_and_counted() {
		let wire = Rc::new(RefCell::new(Wire::default()));
		let mut alone = Net::over(Box::new(Loopback::at(somewhere(1), &wire)), false, 1);

		alone.introduce(somewhere(9));
		alone.send(Duration::ZERO, None);

		assert_eq!(wire.borrow().nowhere(), 1, "there is no inbox at that address");
	}

	#[test]
	fn a_lossy_wire_still_gets_every_command_there() {
		// the property the ring exists for, driven through the whole of this
		// file rather than through the crate under it: the endpoint, the
		// channel, the link and the ring.
		let (mut host, mut client, _wire) = two();
		let bad = Conditions {
			loss: 0.4,
			burst: 4.0,
			..Conditions::PERFECT
		};

		host.set(bad);
		client.set(bad);
		round(&mut host, &mut client, 1);
		host.set(bad);

		let mut heard = Vec::new();

		for step in 2..=400_u32 {
			if step.is_multiple_of(50) {
				host.say(&format!("host.tick {step}"))
					.expect("the ring takes it");
			}

			round(&mut host, &mut client, step);
			heard.extend(client.said().iter().map(|said| said.text.clone()));
		}

		assert_eq!(heard.len(), 7, "seven were said and seven arrived: {heard:?}");
		assert_eq!(heard[0], "host.tick 50");
		assert_eq!(heard[6], "host.tick 350");

		// and the wire really did lose some. Not `ignored`, which counts what a
		// channel refused: a datagram the link threw away never reaches one.
		// What loss looks like from here is a message the far end never
		// acknowledged.
		let (sent, acknowledged, lost) = host.tally(0);

		assert!(lost > 0, "the wire lost messages: {sent} sent, {acknowledged} acknowledged");
		assert!(acknowledged > 0, "and got most of them through");
	}

	#[test]
	fn the_wire_reads_its_numbers_off_the_console_table() {
		let mut cvars = table();

		assert_eq!(conditions(&cvars), Conditions::PERFECT, "nothing set is a perfect wire");

		assert!(cvars.set(colby_net::LAG, "50"), "the variable is there");
		assert!(cvars.set(colby_net::JITTER, "10"), "the variable is there");
		assert!(cvars.set(colby_net::LOSS, "0.25"), "the variable is there");
		assert!(cvars.set(colby_net::BURST, "3"), "the variable is there");
		assert!(cvars.set(colby_net::DUPLICATE, "0.05"), "the variable is there");

		let asked = conditions(&cvars);

		assert_eq!(asked.lag, Duration::from_millis(50), "milliseconds, as a person writes them");
		assert_eq!(asked.jitter, Duration::from_millis(10));
		assert!((asked.loss - 0.25).abs() < f32::EPSILON);
		assert!((asked.burst - 3.0).abs() < f32::EPSILON);
		assert!((asked.duplicate - 0.05).abs() < f32::EPSILON);
	}

	#[test]
	fn a_variable_that_is_missing_reads_as_a_perfect_wire_rather_than_a_broken_one() {
		// a typo should not quietly start losing datagrams, which is the same
		// rule the volumes follow: a missing one is full volume, not silence.
		let empty = Cvars::new();

		assert_eq!(conditions(&empty), Conditions::PERFECT);
		assert_eq!(seed(&empty), DEFAULT_SEED);
	}

	#[test]
	fn a_lag_too_large_to_be_one_is_cut_down_rather_than_ending_the_process() {
		// the constructor a span is built with *panics* on a value it cannot
		// hold, and what feeds it is a console. Nothing else in this file asks
		// it for an absurd number, so nothing else would notice.
		let mut cvars = table();

		assert!(cvars.set(colby_net::LAG, "1e30"), "the variable is there");
		assert_eq!(
			conditions(&cvars).lag,
			Duration::from_secs_f64(f64::from(MAX_MILLIS) / 1000.0),
			"held at the ceiling"
		);

		assert!(cvars.set(colby_net::JITTER, "1e30"), "the variable is there");
		assert!(conditions(&cvars).jitter < Duration::from_secs(61), "and so is the shake");
	}

	#[test]
	fn a_lag_that_is_negative_is_no_lag_rather_than_an_enormous_one() {
		let mut cvars = table();

		assert!(cvars.set(colby_net::LAG, "-50"), "the variable is there");
		assert_eq!(conditions(&cvars).lag, Duration::ZERO);
	}

	#[test]
	fn the_seed_is_the_usual_one_until_somebody_asks_for_another() {
		let mut cvars = table();

		assert_eq!(seed(&cvars), DEFAULT_SEED, "nil is the usual one");

		assert!(cvars.set(SEED, "-7"), "the variable is there");
		assert_eq!(seed(&cvars), DEFAULT_SEED, "and so is anything below it");

		assert!(cvars.set(SEED, "12345"), "the variable is there");
		assert_eq!(seed(&cvars), 12_345);
	}

	#[test]
	fn a_host_named_on_the_command_line_is_read_and_a_word_that_is_not_one_is_not() {
		assert_eq!(
			asked_for(&["--connect".to_owned(), "127.0.0.1:27015".to_owned()]),
			Some(somewhere(27_015))
		);
		assert_eq!(asked_for(&["--connect=127.0.0.1:1".to_owned()]), Some(somewhere(1)));
		assert_eq!(asked_for(&["--connect".to_owned()]), None, "with nothing after it");
		assert_eq!(
			asked_for(&["--connect".to_owned(), "not an address".to_owned()]),
			None,
			"and with something that is not one"
		);
		assert_eq!(asked_for(&["--host".to_owned()]), None, "and not somebody else's flag");
	}

	#[test]
	fn a_request_made_with_no_wire_is_answered_rather_than_kept() {
		// a command typed at a process that is not on a wire has been answered,
		// and keeping it would mean it fired the moment one appeared.
		ask(Request::Status);
		serve(None);

		let waiting = ASKED.lock().map(|held| held.len()).unwrap_or(1);

		assert_eq!(waiting, 0, "the queue is empty either way");
	}
}
