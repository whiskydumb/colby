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
//! cadence is [`colby_net::every`] and the decision is the caller's: `send`
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
	abi::{
		Bodies, Body, BodyId, BodyKind, Command, Cvars, EntityId, PeerId, Role, Value, World,
		console, cvar::Owner, net::BACKUP,
	},
	debug, err,
	glam::Vec3,
	info, warn,
};
use colby_net::{
	Block, Channel, Conditions, Delivery, Heard, Link, MAX_DATAGRAM, NOTHING, Reliable, Ring,
	Slot, Snapshot, Solid,
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
///
/// **Clients**, where the world's own ceiling counts the host as well, and the
/// two have to agree or an endpoint talks to somebody the world has no block
/// for. They used to agree by arithmetic and nothing else, and
/// [`abi::net::MAX_PEERS`](colby_core::abi::net::MAX_PEERS) says in its own
/// doc that the commit which gives a connecting client a slot is where that
/// has to stop being true. This is that commit.
const MAX_PEERS: usize = 8;

// and the world's is this plus the host's own. A check rather than a
// derivation, so that either number can be read where it is written and the
// build refuses a pair that has drifted.
const _: () = assert!(
	MAX_PEERS + 1 == colby_core::abi::net::MAX_PEERS,
	"an endpoint would talk to somebody the world keeps no block for"
);

/// How long a peer may say nothing before it is forgotten.
const QUIET: Duration = Duration::from_secs(10);

/// How far behind the moment a client draws the world it was told about.
///
/// A tenth of a second, which is what the system this is modeled on defaults
/// to and is two snapshots at the cadence with a little over.
const INTERP: &str = "net.interp";

/// The default, in seconds.
const INTERP_DEFAULT: f32 = 0.1;

/// The furthest behind anybody may ask to be, in seconds.
///
/// A second is already a world nobody could play in. What this defends is the
/// arithmetic rather than the taste: a delay longer than the ring holds asks
/// for a moment older than anything still here, which is answered with the
/// oldest world there is and would look like the world had stopped.
const MAX_INTERP: f32 = 1.0;

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
fn asked_for(arguments: &[String]) -> Option<SocketAddr> { after(arguments, CONNECT) }

/// The address after a flag, whichever flag is doing the asking.
///
/// **Written once and called twice.** Two modes want a host's address off the
/// command line - a window and the client with nothing on screen - and they
/// want it in exactly the same shapes, so a second copy of this would be a
/// second set of rules about what an address looks like, differing the first
/// time somebody fixed one of them. @ref `crate::host`, which asks with its
/// own flag.
///
/// @param arguments - the command line, without the executable
/// @param flag - which flag to look for
/// @return the first address named after it
pub(crate) fn after(arguments: &[String], flag: &str) -> Option<SocketAddr> {
	for (index, argument) in arguments.iter().enumerate() {
		let text = if let Some(rest) = argument.strip_prefix(&format!("{flag}=")) {
			Some(rest.to_owned())
		} else if argument == flag {
			arguments.get(index + 1).cloned()
		} else {
			continue;
		};

		let Some(text) = text else {
			warn!("{flag} needs an address to connect to");

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
		filed = net.filed(),
		garbled = net.garbled(),
		dropped = net.dropped(),
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

/// Whether a console line that crossed is one a peer may run here.
///
/// **The gate, and it is one line of policy: a peer may run the game's
/// commands and none of the engine's.** Every entry in the table already says
/// who registered it - the engine or the module - and the two are exactly the
/// two halves that matter. `quit`, `exec`, `scene load`, `sim.pause`, `net.*`
/// and `editor.*` are the engine's and belong to whoever is at the keyboard of
/// the machine that is running; `game.freeze`, `game.spawn` and the rest are
/// gameplay, which is the thing a player is entitled to ask for. Nothing new
/// had to be declared for this: the distinction was already there for what a
/// reload does to a table, and it turns out to be the same line.
///
/// **A variable is not a command and is refused with them.** Setting a value
/// is configuration - how fast, how loud, how many - and configuration is the
/// operator's even when the game registered it. What a peer may do is *call*
/// something the game exposed.
///
/// **A line is allowed whole or refused whole.** A console line may hold
/// several statements, and running the half in front of a refusal is the worst
/// of the three answers available - the same argument the reliable ring makes
/// about a block it could not finish reading.
///
/// **This is a gate, not a security boundary**, and the difference is worth
/// saying out loud: nothing on this wire is authenticated, so what it decides
/// is which *names* a stranger may call rather than which strangers may call.
/// A game that registers a command a stranger must not run is a game that has
/// published it, and there is no flag here that says otherwise yet.
///
/// @param cvars - the table this process has
/// @param text - the line exactly as it crossed
/// @return whether to run it
pub(crate) fn allowed(cvars: &Cvars, text: &str) -> bool {
	let statements = console::statements(text);

	// a line with nothing in it is not a line to run. Refused rather than
	// passed through, because "every statement is allowed" is trivially true
	// of none of them and this reads as an allowance rather than a nothing.
	if statements.is_empty() {
		return false;
	}

	statements.iter().all(|words| {
		words.first().is_some_and(|name| {
			cvars
				.get(name)
				.is_some_and(|entry| entry.is_command() && entry.owner() == Owner::Module)
		})
	})
}

/// Runs what a peer asked for, if a peer is allowed to ask for it.
///
/// A host's, and only a host's. A client running what a host typed at it would
/// be the authority reaching through the wire into a machine it does not own,
/// which is a different feature with a different argument and nobody has made
/// it. @ref [`allowed`] for what a host will run.
///
/// @param world - the world commands are run against
/// @param net - the endpoint whose last receive is being answered
pub(crate) fn obey(world: &mut World, net: &Net) {
	if !net.hosting() {
		return;
	}

	// one at a time, and each one asked about against the table **as it now
	// stands**. There is no borrow to escape here - what crossed lives in the
	// endpoint, which arrives as a shared reference, and a command is run
	// against the world, which is a different object - so collecting first
	// would buy nothing and cost something: a command may register or forget
	// entries, and a batch decided up front would gate every line after it
	// against a table that no longer exists.
	for said in net.said() {
		if !allowed(&world.cvars, &said.text) {
			warn!(from = %said.from, text = %said.text, "a peer asked for something it may not");

			continue;
		}

		// **a line nobody said is not run.** A peer that has not been seated
		// has no player for a command to be about, and running it anyway is
		// what used to make every crossed line act on whoever is at the host's
		// keyboard.
		let who = net.whose(said.from);

		if !who.is_some() {
			warn!(from = %said.from, text = %said.text, "a line from nobody");

			continue;
		}

		// **and it is run as its sender.** Every command in this game reads
		// `World::peer` to find out whose player it is about - `played`, the
		// spawn point in front of somebody, which crosshair a nameless
		// `game.grab` means. Swapping the field around the call is what turns
		// the console from the host's keyboard into an RPC layer, which is what
		// the design said it was for. Put back afterwards whatever the command
		// did, because a command that ended the process leaves the world half
		// written and the field has to name this machine again either way.
		let was = world.peer;

		world.peer = who;
		info!(from = %said.from, slot = who.slot(), text = %said.text, "running what a peer asked for");
		crate::console::run(world, &said.text);
		world.peer = was;
	}
}

/// Says that this process has gone looking for somebody else's world.
///
/// A world is born believing it is the authority, because that is true of
/// every world that is on its own, and most worlds are. A window that was
/// told to connect is the one case where it is false, and nothing else in the
/// engine can notice - so it is said here, once, at the moment the socket is
/// opened.
///
/// *Which* peer this process is, the host says, and this is the value it says
/// it into: a window is nobody from the moment it opens a socket until the
/// first message that names it arrives. @ref [`Net::seat`], which is where
/// that happens and where a host claiming this window is the *host* is
/// refused.
///
/// Being nobody in the meantime is not a placeholder. "Not the host" is the
/// whole of what [`Role::of`](colby_core::abi::Role) needs to call every body
/// a proxy, and a proxy is a body this end draws rather than decides - so a
/// window that is never named still shows the world correctly and only fails
/// to have anything of its own in it.
///
/// @param world - the world this process is running
pub(crate) fn joined(world: &mut World) { world.peer = PeerId::NONE; }

/// How far behind to draw, as the table has it.
///
/// @param cvars - the world's own table
/// @return the delay, clamped to something the ring can answer
#[must_use]
pub(crate) fn behind(cvars: &Cvars) -> Duration {
	let seconds = cvars.float(INTERP).unwrap_or(INTERP_DEFAULT);

	Duration::from_secs_f32(seconds.clamp(0.0, MAX_INTERP))
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

	/// Who this peer is, as *this end* has named them.
	///
	/// A host mints one out of [`World::players`] the first time it seats
	/// somebody, and says it in every message afterwards. A client names
	/// nobody and leaves this [`PeerId::NONE`] forever: naming is the
	/// authority's, like everything else.
	id: PeerId,

	/// Who the far end says *this end* is.
	///
	/// The other direction of the same field, and the whole of the naming
	/// there is. A client copies this into
	/// [`World::peer`](colby_core::abi::World::peer) and stops being nobody; a
	/// host is told nothing and ignores it.
	named: PeerId,

	/// What this peer has asked for and this end has not filed yet.
	///
	/// A short queue rather than a straight write into the world, because the
	/// wire is drained where there is no world to write into - and because a
	/// peer's *first* message arrives in the same pass that admits it, so its
	/// commands have to wait somewhere for a name to be minted. Bounded at a
	/// ring's depth, since anything older than that is something the world's
	/// own ring would drop the moment it arrived.
	asked: Vec<Command>,

	/// The highest of *their* command numbers this end is done with.
	///
	/// What goes in the head of every block sent to them. Read back out of
	/// the world after each filing, so the number on the wire is the world's
	/// answer rather than a second copy of it.
	settled: u32,

	/// The highest of *ours* they have said they are done with.
	confirmed: u32,

	/// The same, as it stood in the message that brought the newest world.
	///
	/// **Not [`confirmed`](Self::confirmed), and the difference is the whole of
	/// whether a client's guess fights the host.** A client predicts forward
	/// from a snapshot, and it has to know which of its commands that snapshot
	/// already accounts for. Those two facts arrive together - a message
	/// carries a block of commands and then a block of world - but they do not
	/// arrive at the same *rate*: a snapshot goes with one message in every
	/// [`every`](colby_net::every) of them, and an acknowledgement with all of
	/// them.
	///
	/// Measuring the newest mark against a world several messages older is
	/// asking the guess to cover a stretch the base already covers, which reads
	/// as the player having been in two places and is corrected every time a
	/// snapshot lands. That was a smear of whole units on a wire with a
	/// hundred milliseconds on it.
	stamped: u32,

	/// What was told to this peer, to write the next difference against.
	told: Ring,

	/// Scratch for what the far end will hold once it has read what was just
	/// written, so working it out costs no allocation per peer per snapshot.
	///
	/// A field rather than a local because it is refilled twenty times a
	/// second forever, which is the same reason [`Ring`] keeps one.
	kept: Vec<Slot>,

	/// What was taken *from* it, which is a thing of its own so that reading a
	/// block can borrow it while the block itself is still borrowed from the
	/// channel beside it.
	taken: Taken,
}

/// What one peer has said about the world, and what this end made of it.
struct Taken {
	/// The worlds taken from that peer, and when each of them arrived.
	///
	/// It holds a ring for the same reason the sending half does: a block is
	/// written against a *numbered* snapshot, and the number it names is a
	/// round trip old whenever snapshots go out faster than one per round
	/// trip - which at twenty a second they do. Most of the time the newest
	/// world would answer the same way, because most fields that changed
	/// since then changed once. The case it would not is a field that changed
	/// and changed *back*, which the writer sees as unchanged and does not
	/// send; a reader holding the value from halfway would keep it forever.
	///
	/// The times beside it are what lets a world be drawn a moment behind
	/// rather than in jumps. @ref [`Heard`].
	heard: Heard,

	/// The newest snapshot *it* says it has, which is what the next difference
	/// to it is written against.
	theirs: u32,

	/// How many snapshots in a row have been cut short of the world's end.
	///
	/// A cut snapshot is **ordinary**: a world bigger than one block is told
	/// over several, and the count going up for a moment is a peer catching
	/// up rather than a peer in trouble. What is not ordinary is the count
	/// never coming back down, which is the tail of the slot table never being
	/// reached because the front of it fills every block on its own. @ref
	/// [`STARVED`].
	behind: u32,

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
	/// Which address it came from.
	pub(crate) from: SocketAddr,

	/// The console line, exactly as it was queued.
	pub(crate) text: String,
}

/// One endpoint: a wire, and a conversation with everybody on the far end.
pub(crate) struct Net {
	post: Box<dyn Post>,
	peers: Vec<Peer>,
	/// The address a client went looking for, if it is a client.
	///
	/// Kept so that a host which was slow to start, or quiet for longer than
	/// [`QUIET`], is somebody this end will still listen to. @ref
	/// [`Net::find`].
	looking: Option<SocketAddr>,

	/// Whether a datagram from a stranger becomes a peer or is thrown away.
	hosting: bool,
	conditions: Conditions,
	seed: u64,
	scratch: Box<[u8; MAX_DATAGRAM]>,
	payload: Vec<u8>,
	commands: Vec<String>,
	said: Vec<Said>,

	/// What this end is asking for, as of the last step.
	///
	/// One window for the endpoint rather than one per peer, because the thing
	/// asking is the person at this keyboard and there is one of them. A host
	/// leaves it empty: a host asks nobody for anything.
	asking: Vec<Command>,

	/// Where an arriving block's commands are read before they are queued.
	arrived: Vec<Command>,

	/// Peers that have gone, waiting for a world to be told.
	///
	/// The wire is drained with no world in hand, so a peer can be dropped in
	/// a pass that cannot let its slot go. @ref [`seat`](Self::seat).
	departed: Vec<PeerId>,
	/// Which bodies a world that arrived is about, and what it said of them.
	///
	/// Gathered in one walk over the body table and applied in another,
	/// because a handle cannot be made from a slot number outside the table's
	/// own crate and nothing here may forge one.
	proxies: Vec<(BodyId, Solid)>,

	/// The newest world a client has been told about, kept beside the blend.
	///
	/// Two tables rather than one because a client wants two answers at once:
	/// where everybody else is a fixed delay ago, and where *it* is right now.
	/// @ref [`Net::arrive`].
	freshest: Vec<Slot>,

	/// The world a delay behind the newest, which is where a proxy is drawn.
	///
	/// A copy rather than the borrow `Heard::at` hands out, because the walk
	/// that reads it reads [`Net::freshest`] in the same breath and the two
	/// come out of the same peer.
	drawn: Vec<Slot>,

	/// Which bodies the far end has stopped describing, gathered in the same
	/// walk as the proxies and applied after them.
	removed: Vec<BodyId>,

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
	/// How many snapshots went out cut short of the world's end, counted once
	/// per peer per attempt rather than once per world.
	///
	/// Not a fault on its own - a world bigger than one block is told over
	/// several, and this is how many of those there were. @ref [`STARVED`] for
	/// the shape of it that is a fault.
	crowded: u32,

	/// How many commands were filed into a world.
	filed: u32,

	/// How many messages were thrown away for carrying a block of commands
	/// that is not one.
	///
	/// A message rather than a peer, which is the difference from every other
	/// block on this wire. Named apart from [`Net::refused`], which counts
	/// datagrams the socket itself would not take. @ref [`Net::next_message`].
	garbled: u32,

	/// How many bodies were despawned because the far end stopped describing
	/// them.
	dropped: u32,
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
			looking: None,
			conditions: Conditions::PERFECT,
			seed,
			scratch: Box::new([0; MAX_DATAGRAM]),
			payload: Vec::new(),
			commands: Vec::new(),
			said: Vec::new(),
			asking: Vec::new(),
			arrived: Vec::new(),
			departed: Vec::new(),
			proxies: Vec::new(),
			freshest: Vec::new(),
			drawn: Vec::new(),
			removed: Vec::new(),
			applying: Vec::new(),
			sent: 0,
			delivered: 0,
			ignored: 0,
			strangers: 0,
			forgotten: 0,
			crowded: 0,
			filed: 0,
			garbled: 0,
			dropped: 0,
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

		net.introduce(address);
		info!(%ours, %address, "talking to a host");
		Ok(net)
	}

	/// Starts a conversation with an address, without binding anything.
	///
	/// What [`connect`](Self::connect) does once it has a socket, for the
	/// endpoints that already have a wire - a test, and the two-endpoint run.
	///
	/// @param address - where the far end is
	pub(crate) fn introduce(&mut self, address: SocketAddr) {
		self.looking = Some(address);
		self.add(address, Duration::ZERO);
	}

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

	/// Which peer an address is, as this end has it now.
	///
	/// **Asked when a line is run rather than stamped when it arrived**, and
	/// the difference is the very first message somebody sends. A peer is
	/// given its name by the block of commands in that message, which is read
	/// *after* the reliable ring in front of it - so a line stamped as it
	/// crossed would be stamped by a peer that had no name yet, and the first
	/// thing anybody ever asked for would be the one thing that ran as
	/// nobody. @ref [`Net::file`], which is where a name is minted.
	///
	/// @param address - where it came from
	/// @return that peer's name, or nobody if the address is nobody's
	pub(crate) fn whose(&self, address: SocketAddr) -> PeerId {
		self.peers
			.iter()
			.find(|peer| peer.address == address)
			.map_or(PeerId::NONE, |peer| peer.id)
	}

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

	/// How many bodies were taken away because the far end stopped describing
	/// them.
	pub(crate) const fn dropped(&self) -> u32 { self.dropped }

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
			.map(|peer| peer.taken.heard.holding())
		else {
			return &[];
		};

		self.peers[index].taken.heard.base(holding).0
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
			.map_or(NOTHING, |peer| peer.taken.heard.holding())
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

	/// Sets what this end is asking for, for every message from here on.
	///
	/// The window rather than one command: the last few unsettled go in
	/// *every* message on purpose, so this is replaced each step rather than
	/// appended to. A host calls it with nothing, because a host asks nobody
	/// for anything.
	///
	/// @param commands - this end's unsettled window, oldest first
	pub(crate) fn ask(&mut self, commands: &[Command]) {
		self.asking.clear();
		self.asking.extend_from_slice(commands);
	}

	/// Seats whoever has arrived, files what they asked for, and lets go of
	/// whoever has gone.
	///
	/// **Where a peer's identity meets the world**, which is not the only
	/// place the wire touches it - [`arrive`](Self::arrive) writes what a
	/// snapshot said and [`obey`](crate::net::obey) runs what a peer typed -
	/// but is the only one that mints or lets go of anything. A call of its
	/// own rather than part of [`receive`](Self::receive) for a reason worth
	/// stating: the wire is drained in places where there is no world at all -
	/// the two-endpoint run has two endpoints and no `World` between them -
	/// and a signature that demanded one would make the cheapest checkable
	/// thing in this subsystem impossible to build. So the socket half runs on
	/// its own and this is where the two are put together.
	///
	/// On a **host** it mints a [`PeerId`] for anybody who has not got one,
	/// pushes what each peer asked for into the world's own ring under that
	/// name, and reads back what the world has settled so the next message can
	/// say so. On a **client** it takes the name the host gave and settles its
	/// own ring by what the host has confirmed.
	///
	/// @note: a peer whose commands arrive in the same pass that admits it is
	/// the ordinary case rather than a corner - the first datagram is what
	/// makes it a peer at all. That is why a peer holds its commands rather
	/// than the world doing it: at the moment they arrive there is no name to
	/// file them under.
	///
	/// @param world - the world this process is running
	pub(crate) fn seat(&mut self, world: &mut World) {
		for peer in self.departed.drain(..) {
			// the slot goes back and the ring does not, which is deliberate:
			// nothing lets one peer's commands go on purpose, because the
			// thing that used to is what let a departed peer hand a command
			// back for a second run. The next occupant of the slot arrives at
			// a higher generation and clears the ring on its first command.
			// @ref `colby_core::abi::Commands`.
			world.players.forget(peer);
		}

		if !self.hosting {
			let Some(peer) = self.peers.first() else {
				// nobody is talking to this window, so it is nobody's client.
				// Said rather than left, because a name that outlives the
				// conversation it came from is a window that goes on believing
				// it owns things after the host it was owned by has gone.
				world.peer = PeerId::NONE;

				return;
			};

			// **and not whatever the wire says.** This is the one place a
			// peer's identity crosses an unauthenticated wire, and the value
			// it lands in is the one the whole authority model turns on:
			// `Role::of` hands `PeerId::HOST` authority over every body there
			// is, and the step settles commands for a world whose peer is the
			// host. Eight bytes from a stranger must not be able to say so.
			//
			// The rule is the slot rather than the generation, which is the
			// same rule `Commands::push` uses and for the same reason: slot
			// zero is the host's, no table ever mints a client into it, and a
			// generation is a thing a liar picks freely.
			if peer.named.slot() == PeerId::HOST.slot() && peer.named.is_some() {
				warn!(
					address = %peer.address,
					"a host that says this window is the host, which it is not"
				);
			} else {
				world.peer = peer.named;
				// and the table takes the name as well, which is what gives a
				// window a block of its own to keep a player in. A host mints
				// its peers; a window is told, and until it can seat what it
				// was told it has nowhere to put a command, nothing to be
				// corrected about, and no player at all. @ref
				// `Players::seat`, and note this costs nothing on the sixty
				// messages a second that say the same name again.
				world.players.seat(world.peer);
			}

			// **two marks, because a client asks two questions.** What it may
			// stop re-sending is everything the host has acknowledged, which is
			// every message. What it may stop *guessing about* is only what the
			// world it is looking at already accounts for, which is one message
			// in [`every`](colby_net::every). @ref `Peer::stamped` and
			// `Commands::base`.
			world.commands.settle(world.peer, peer.confirmed);
			world.commands.base(world.peer, peer.stamped);

			return;
		}

		for peer in &mut self.peers {
			self.filed = self.filed.saturating_add(file(peer, world));
		}
	}

	/// Whether this endpoint is the one that decides.
	pub(crate) const fn hosting(&self) -> bool { self.hosting }

	/// How many commands have been filed into a world.
	pub(crate) const fn filed(&self) -> u32 { self.filed }

	/// How many messages were thrown away over their block of commands.
	pub(crate) const fn garbled(&self) -> u32 { self.garbled }

	/// How many peers are waiting for a world to be told they have gone.
	///
	/// For the test that pins the claim [`forget`](Self::forget) makes about
	/// this list staying bounded where nothing drains it. Nothing else reads
	/// it: what it is *for* is [`seat`](Self::seat), which empties it.
	#[cfg(test)]
	pub(crate) fn departed(&self) -> usize { self.departed.len() }

	/// What one peer has asked for and not been filed into a world.
	///
	/// For the two-endpoint run, which has no world to file into and is
	/// therefore the one place this queue can be looked at directly. A host
	/// with a world empties it every pass. @ref [`seat`](Self::seat).
	pub(crate) fn holding_commands(&self, index: usize) -> &[Command] {
		self.peers
			.get(index)
			.map_or(&[], |peer| peer.asked.as_slice())
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

	/// Puts a snapshot block into what the first peer holds, with no wire.
	///
	/// For a test in another module that has a world and a step to run but no
	/// far end to hear from. Everything the real path does to a block before
	/// this - the datagram, the channel, the reliable ring - is tested here.
	///
	/// @param bytes - the block
	/// @param now - when it is to count as having arrived
	/// @return whether it was something a working peer would have sent
	#[cfg(test)]
	pub(crate) fn absorbed(&mut self, bytes: &[u8], now: Duration) -> bool {
		let Some(peer) = self.peers.first_mut() else {
			return false;
		};

		absorb(&mut peer.taken, bytes, &mut self.applying, now)
	}

	/// Puts the world a host described into the world this end is running.
	///
	/// **A body this end does not own stops being simulated and is driven
	/// instead.** That is one line here because the engine already had the
	/// idea: [`BodyKind::Kinematic`] means the solver leaves a body's
	/// transform alone and something else writes it, which is what freezing a
	/// prop already used. So a proxy is a frozen body with the wire holding the
	/// pen.
	///
	/// **What is written is the entity, not the body.** The solver copies a
	/// body it does not move *from* the entity it drives, so writing the entity
	/// is writing both - and writing the body instead would be overwritten a
	/// moment later by the entity it was supposed to have moved.
	///
	/// A slot the far end describes and this end has no body in is passed over
	/// in silence. Nothing here can make one: what a body *is* - its shape, its
	/// mass, its surface - is not in a snapshot by design, so a thing that
	/// appears on the host appears here when something says what it is, and
	/// that message does not exist yet.
	///
	/// @param world - the world this end is running
	/// @param when - the moment to draw, already behind the clock
	pub(crate) fn arrive(&mut self, world: &mut World, when: Duration) {
		// a host is the one that says what the world is; nothing tells it.
		if self.hosting {
			return;
		}

		let Some(peer) = self.peers.first_mut() else {
			return;
		};

		// **two worlds, not one, and which body reads which is the whole of
		// what a client draws.** Everybody else's body is drawn a fixed delay
		// behind the newest, because a proxy drawn at the newest stutters every
		// time a message is late. This window's *own* body is corrected against
		// the newest with no delay at all, because it is not drawn from this at
		// all - it is the base its own prediction runs forward from, and a base
		// a tenth of a second in the past is a guess that starts a tenth of a
		// second wrong. @ref `colby_net::Heard::newest`.
		peer.taken.heard.newest(&mut self.freshest);
		// copied out rather than borrowed, because the newest above is the
		// endpoint's own field and the loop below reads both at once.
		self.drawn.clear();
		self.drawn
			.extend_from_slice(peer.taken.heard.at(when));

		let told = &self.drawn;

		self.proxies.clear();
		self.removed.clear();

		// **a window that has not been told who it is takes nothing away.**
		// `Role::of` answers `SimulatedProxy` for every body in the world until
		// this end holds a name - deliberately, @ref `colby_core::abi::Role` -
		// so a client acting on empty slots before it is seated would delete
		// its own map on the first snapshot it ever received. Driving proxies
		// before a name is harmless and useful; removing them is neither.
		let named = world.peer.is_some();

		for (id, body) in world.bodies.iter() {
			// **whether a thing is here at all is asked of the newest, and
			// where it is of the blend.** They are different questions and the
			// blend cannot answer the first one: it reaches as far as the
			// *earlier* of the two snapshots and keeps whatever that one held,
			// so a body the far end has dropped is present in every blend
			// forever rather than for a delay. @ref `colby_net::Heard::blend`,
			// which says so - existence is a fact and not something to
			// interpolate towards.
			let Some(here) = self.freshest.get(id.slot()) else {
				// past the end of what the far end holds, which is a body this
				// end has and the other has never described. Left alone: it is
				// not a removal, it is a table that has not caught up.
				continue;
			};

			let Some((generation, newest)) = *here else {
				self.removed
					.extend(dropping(world.peer, named, id, body));

				continue;
			};

			// and now where it is: the blend when it has an answer, and the
			// newest when the blend does not reach this far yet.
			let solid = told
				.get(id.slot())
				.copied()
				.flatten()
				.filter(|(was, _)| *was == generation)
				.map_or(newest, |(_, drawn)| drawn);

			// the generation is what says this is the same body rather than
			// the one that took its place, and a wrong answer here would drive
			// a live body to where a dead one was.
			//
			// **A body this window owns is written too**, which is the one
			// place the role is *not* the question. It used to be skipped -
			// `local()` is the two roles that decide, and a window decides
			// about its own - and that left a client with nothing to be
			// corrected against: the wire never said where the machine that
			// decides had actually put it, so a wrong guess was a wrong guess
			// forever. What the wire writes is the truth as of the last
			// command the host confirmed, and predicting forward from it is
			// the game's. @ref `colby_core::abi::character::replay`.
			if generation != id.generation() || body.role(world.peer) == Role::Authority {
				continue;
			}

			// and this is where the two worlds part: a body this window owns
			// takes the newest record of itself when there is one, and falls
			// back to the delayed blend when the ring is not holding one yet.
			//
			// **Whose it is comes from the record and not from the body**, for
			// the reason the whole field exists: the far end is the authority
			// on ownership and the copy sitting here is one message stale.
			// Asking the body would read every one of them as nobody's on the
			// very first snapshot, which is a window predicting its own player
			// off the delayed blend until a second message arrived.
			let solid = if world.peer.is_some() && world.peer == owner(&newest) {
				newest
			} else {
				solid
			};

			self.proxies.push((id, solid));
		}

		// the entities in one pass and the bodies in another, because a handle
		// is only had by walking the table and the walk borrows it.
		for (id, solid) in &self.proxies {
			drive(world, *id, solid);
		}

		// and the removals last, so that nothing is driven after it is gone.
		for id in &self.removed {
			let entity = world
				.bodies
				.get(*id)
				.map_or(EntityId::NONE, |body| body.entity);

			world.joints.forget(*id);
			world.bodies.despawn(*id);
			world.entities.despawn(entity);
		}

		if !self.removed.is_empty() {
			self.dropped = self
				.dropped
				.saturating_add(u32::try_from(self.removed.len()).unwrap_or(u32::MAX));
		}
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
			// begins; then the commands, which say the same of themselves;
			// then a snapshot block, always, even when it describes nothing at
			// all. Three blocks in every message, each one a head at worst.
			peer.reliable.write(&mut self.payload);
			// what this end is asking for and what it has done about what the
			// far end asked. A host writes the name it minted and an empty
			// body; a client writes nobody and its own unsettled window.
			Block::write(peer.id, peer.settled, &self.asking, &mut self.payload);

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
			let gone = self.peers.remove(index);

			// as in `forget`: a peer that never had a name took no slot to
			// give back, and pushing one would grow a list nothing drains.
			if gone.id.is_some() {
				self.departed.push(gone.id);
			}
			self.forgotten = self.forgotten.saturating_add(1);
			warn!(address = %gone.address, "a peer that is not talking sense");
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

				// the ring said how far it read, so what follows it is the
				// block of commands. `taken` and `channel` are different
				// fields, which is what lets this borrow one while the block
				// is still borrowed from the other.
				let rest = message.get(used..).unwrap_or(&[]);

				// **a block that is not one costs what is left of the
				// message and not the peer**, which is the opposite of what
				// the reliable ring above it does, and the difference is that
				// this block carries no state between one and the next. The
				// ring is numbered, so a bad one loses the numbering and there
				// is nothing to do with the conversation afterwards; a block
				// of commands is whole every time, so throwing it away costs
				// whatever it was carrying and nothing else. Dropping a peer
				// for it would let one flipped bit end a session, which is the
				// thing the world's own ring has a window to avoid.
				//
				// *What is left of* the message, precisely: the reliable block
				// in front of this one has already been read and already been
				// acknowledged, so whatever console lines it carried are
				// already on their way to being run. Undoing that is not
				// available - the numbering has moved - and it is the same
				// bargain the ring makes with itself. What does go is the
				// commands here and the *snapshot* behind them, because this
				// block is what says where that one begins.
				let Ok(told) = Block::read(rest, &mut self.arrived) else {
					self.garbled = self.garbled.saturating_add(1);
					debug!(%from, "a peer said something that is not a block of commands");

					return Some(true);
				};

				peer.named = told.yours;
				peer.confirmed = told.settled;

				// kept before the block behind it is read, because that is
				// where the two facts are still known to have come out of one
				// message. @ref [`Peer::stamped`].
				let settled = told.settled;
				let held = peer.taken.heard.holding();

				queue(&mut peer.asked, &mut self.arrived);

				let rest = rest.get(told.used..).unwrap_or(&[]);
				let read = absorb(&mut peer.taken, rest, &mut self.applying, now);

				if peer.taken.heard.holding() != held {
					peer.stamped = settled;
				}

				Some(read)
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
		let departed = &mut self.departed;

		self.peers.retain(|peer| {
			let here = now.saturating_sub(peer.heard) < QUIET;

			// only a peer that was given a name, which is what keeps this
			// list bounded on an endpoint that never seats anybody: the
			// two-endpoint run drains it from nowhere, and on that run nothing
			// is ever named, so nothing is ever pushed.
			if !here && peer.id.is_some() {
				departed.push(peer.id);
			}

			here
		});

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

		// **a host takes a datagram from anybody; a client takes one only from
		// the address it went looking for.** Which is not the same as taking
		// none, and used to be: a client whose host had not started yet went
		// quiet for ten seconds, forgot the only peer it had, and then refused
		// every datagram the host ever sent because the address was no longer
		// one it knew. It would sit there for the rest of the process saying
		// nothing to a host that was answering.
		//
		// @note: this makes a *new* conversation with the same address, its
		// rings numbering from one at this end. That converges when the far end also
		// gave up on this one and does not when only this end did - which is
		// the connection handshake's job and is not written. @ref [`Net::add`].
		let looked = self
			.looking
			.is_some_and(|looking| looking == address);

		if (!self.hosting && !looked) || self.peers.len() >= MAX_PEERS {
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
			id: PeerId::NONE,
			named: PeerId::NONE,
			asked: Vec::new(),
			settled: 0,
			confirmed: 0,
			stamped: 0,
			told: Ring::new(),
			kept: Vec::new(),
			taken: Taken {
				heard: Heard::new(),
				theirs: NOTHING,
				baselines: 0,
				behind: 0,
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

/// How many snapshots in a row may be cut before it is worth saying so.
///
/// Three seconds at twenty a second. Not tidiness: a peer joining a world of
/// the largest table this engine allows is told it over about a dozen blocks,
/// and a number under that would report every big join as a fault. Sixty is
/// comfortably past any honest catch-up and comfortably short of a person
/// wondering why nothing is arriving.
const STARVED: u32 = 60;

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
		nothing(peer.taken.heard.holding(), out);

		return true;
	};

	let number = peer.told.next();
	let holding = peer.taken.heard.holding();
	let (base, against) = peer.told.base(peer.taken.theirs);
	let written = Snapshot::write(number, against, holding, base, world, out);
	let reached = written.reached.min(world.len());
	let whole = written.reached >= world.len();

	// **what is kept is what the far end will hold, which is not the same as
	// what this end wanted to send.** As far up as the block got, that is the
	// world; above that, it is whatever the peer already had, because a slot
	// this block never mentioned is a slot nothing about this block changed.
	//
	// Keeping the whole world instead would lose a removal for good: the base
	// would say the peer holds nothing above the cut, so a slot that emptied
	// up there would never be written as gone and the far end would keep the
	// body forever. Keeping only the front instead would lose it the same way.
	peer.kept.clear();
	peer.kept.extend_from_slice(&world[..reached]);

	if let Some(above) = base.get(written.reached..) {
		peer.kept.extend_from_slice(above);
	}

	peer.told.keep(number, &peer.kept);

	// a cut block is ordinary. A run of them that never ends is the tail of
	// the table never being reached, and that is worth one line by address -
	// the count is what a person would otherwise have to guess at.
	peer.taken.behind = if whole { 0 } else { peer.taken.behind.saturating_add(1) };

	if peer.taken.behind == STARVED {
		warn!(
			address = %peer.address,
			bodies = world.len(),
			told = reached,
			snapshots = STARVED,
			"every snapshot to this peer for three seconds has been cut short: the front of \
			 the body table fills each one on its own and the end is never reached"
		);
	}

	whole
}

/// A block that describes nothing and says only what this end holds.
fn nothing(holding: u32, out: &mut Vec<u8>) {
	let written = Snapshot::write(NOTHING, NOTHING, holding, &[], &[], out);

	debug_assert_eq!(written.spoken, 0, "a block of nothing speaks about nothing");
}

/// Names one peer if it has no name, and files what it asked for.
///
/// Its own function rather than the body of the loop in
/// [`Net::seat`](Net::seat), because a branch inside a branch inside a loop
/// inside a method is one level deeper than the house rules allow - and they
/// are right, since what this does is two sentences and reads as two here.
///
/// @param peer - whose conversation
/// @param world - the world to file into
/// @return how many commands were taken
fn file(peer: &mut Peer, world: &mut World) -> u32 {
	// **asked of the world rather than of the handle**, and the difference is
	// a whole class of bug. A cached name says only that this endpoint minted
	// one once; whether the world still knows it is the world's to answer, and
	// there is one thing in the engine that makes the answer change under a
	// live conversation: a restore puts the peer table back as some file had
	// it, marking every slot the file did not describe as empty. A save taken
	// before anybody connected describes nobody.
	//
	// With a cached name that peer becomes a ghost - the world's own walk over
	// its players never reaches it, so its ring is never settled and it
	// resends the same window for the rest of the session - and worse, the
	// next slot the table hands out is the very same handle, so two live peers
	// share one ring and one block. `Players::restore`'s own doc says
	// reconciling this is owed by whoever restored; this is the first thing in
	// the tree that holds a peer's identity outside the world, so it is owed
	// here.
	if !world.players.here(peer.id) {
		peer.id = world.players.admit();

		if peer.id.is_some() {
			info!(address = %peer.address, slot = peer.id.slot(), "a peer is somebody");
		} else {
			// every slot is taken. It is still talked to - a peer with nowhere
			// to file its commands still gets the world, which is the
			// difference between watching and not being here at all - and
			// what it asks for goes on the floor, which is the drain below
			// rather than anything here: a nameless peer's commands are all
			// refused, and the drain empties the queue either way.
			warn!(address = %peer.address, "a peer with nowhere to put what it asks for");
		}
	}

	let mut filed = 0;

	// drained rather than walked, and that is the line that keeps this queue
	// from growing for the length of a conversation: what the world refuses -
	// a repeat, a stray number, anything at all from a peer with no name - is
	// gone from here whether the world took it or not. What holds a command is
	// the world's ring; this is a doorstep.
	for command in peer.asked.drain(..) {
		if world.commands.push(peer.id, command) {
			filed += 1;
		}
	}

	// read back out of the world rather than tracked beside it, so the number
	// on the wire is the world's answer and cannot drift from it.
	peer.settled = world.commands.settled(peer.id);

	filed
}

/// Moves what a block carried onto the end of a peer's queue.
///
/// Bounded here as well as in the world's own ring, because this queue is what
/// holds a peer's commands while it has no name. The case that needs it is an
/// endpoint with **no world at all** - the two-endpoint run is two endpoints
/// and nothing else, and never seats anybody, so nothing ever drains this. A
/// host whose peer table is full is *not* that case: it seats every pass and
/// the drain there empties the queue whether or not the world took anything.
///
/// The oldest goes, which is the same choice made everywhere else this window
/// is trimmed and for the same reason.
///
/// @param asked - the peer's queue
/// @param arrived - what the block held, emptied here
fn queue(asked: &mut Vec<Command>, arrived: &mut Vec<Command>) {
	for command in arrived.drain(..) {
		if asked.len() >= BACKUP {
			asked.remove(0);
		}

		asked.push(command);
	}
}

/// Whether a body the far end no longer describes is one to take away.
///
/// **In range and empty is the far end saying this slot holds nothing.**
/// Removals have always crossed - a snapshot writes a `GONE` mark for a slot
/// that emptied - and nothing ever acted on one, so a prop the host cleaned up
/// stayed on a client forever, kinematic and frozen where the last message left
/// it.
///
/// Nothing at all for a window that has not been told who it is, and nothing
/// for a body this window has authority over, which on a client is its own
/// player: neither is somebody else's to remove by saying nothing.
///
/// @param peer - who this process is
/// @param named - whether that is anybody yet
/// @param id - the body in question
/// @param body - and the body itself, for its owner
/// @return the handle to drop, or nothing
fn dropping(peer: PeerId, named: bool, id: BodyId, body: &Body) -> Option<BodyId> {
	if !named || body.role(peer) == Role::AutonomousProxy {
		return None;
	}

	Some(id)
}

/// Whose a record says its body is.
///
/// The pair of words the wire carries, put back together through the only door
/// there is: a `PeerId`'s fields are the table's to mint and nothing out here
/// may forge one. @ref `PeerId::to_bits`, which is the other half.
fn owner(solid: &Solid) -> PeerId {
	PeerId::from_bits((u64::from(solid.owner[1]) << 32) | u64::from(solid.owner[0]))
}

/// Puts one body's share of a world that arrived into the world.
///
/// Its own function because the body and the entity it drives are two writes
/// through two tables, and a `let Some` for each inside a loop is one level
/// deeper than the house rules allow.
///
/// @param world - the world this end is running
/// @param id - which body
/// @param solid - what the far end said of it
fn drive(world: &mut World, id: BodyId, solid: &Solid) {
	let Some(body) = world.bodies.get_mut(id) else {
		return;
	};

	// **the kind the far end sent, except that nothing this end does not own
	// may be simulated here.** A body the host is simulating has to be
	// kinematic locally or the solver fights the wire sixty times a second;
	// one the host is *not* simulating is passed through as it is, so a static
	// map body stays static and a prop the host froze reads as frozen.
	//
	// @note: what this cannot express is a host-simulated prop, which arrives
	// as kinematic and is indistinguishable from a frozen one to anything that
	// reads only the kind - and the sandbox's own `frozen` is exactly that.
	// Which is why the owner below is written: with it a game can ask the
	// *role* instead of the kind, and the two questions part company exactly
	// here.
	let was = body.kind;

	// **whose it is, which used to be sent and thrown away.** Every authority
	// question in the engine is `(World::peer, Body::owner)` and nothing else
	// (@ref `colby_core::abi::Role`), so a receiver that dropped this left
	// every body on a client owned by nobody - a client could not tell its own
	// player from a stranger's, and `Role::of` could only ever answer
	// `SimulatedProxy`. It is a plain copy because the far end's slot numbering
	// for peers is the far end's, and it is the one both ends agree on.
	body.owner = owner(solid);

	body.kind = match solid.kind {
		| 0 => BodyKind::Static,
		// one is kinematic and two is dynamic, and both arrive as kinematic:
		// the first because that is what it is, the second because this end
		// may not simulate it.
		| _ => BodyKind::Kinematic,
	};
	// and the state, which is on the wire and used to be dropped. A body that
	// fell asleep while this end was still simulating it would otherwise stay
	// asleep as a proxy, and two sleeping bodies are a pair the broadphase
	// refuses - so nothing would ever wake it again.
	body.sleeping = solid.sleeping != 0;
	// the speeds as well as the place. A body that is not simulated is not
	// integrated from them, but the solver reads them to work out what it does
	// to whatever it is pushing - so a proxy carrying a stale speed shoves
	// things the wrong way.
	body.velocity = Vec3::from_array(solid.velocity);
	body.angular = Vec3::from_array(solid.angular);

	let entity = body.entity;
	let cut = matches!(was, BodyKind::Dynamic);

	// **the entity when there is one, and the body itself when there is not.**
	// The solver copies a body it does not move *from* the entity it drives, so
	// for a body that drives one the entity is the only write that lasts. A
	// body that drives none - a ragdoll limb, a collision brush a scene gave no
	// thing to - is never copied over, so writing it directly is what places
	// it. Setting the kind and then failing to place it would be a body the
	// solver has stopped moving and the wire cannot move either.
	let Some(transform) = world.entities.transform_mut(entity) else {
		if let Some(body) = world.bodies.get_mut(id) {
			body.transform = solid.transform();
		}

		return;
	};

	*transform = solid.transform();

	// the first step a body is driven, it comes from wherever this end's own
	// solver had left it, which is anywhere at all. Blending towards the truth
	// from there is a slide across the map, so that one write is a cut.
	if cut {
		world.entities.snap(entity);
	}
}

/// Reads the snapshot block at the end of a message into what a peer holds.
///
/// @param taken - what this end has made of that peer's side of it
/// @param bytes - the block, and nothing is expected after it
/// @param applying - the endpoint's scratch, where the world is put together
/// @param now - how long this endpoint has been running, which is the only
/// clock a snapshot can be stamped with
/// @return whether it was something a working peer would have sent
fn absorb(taken: &mut Taken, bytes: &[u8], applying: &mut Vec<Slot>, now: Duration) -> bool {
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
	let (base, _) = taken.heard.base(against);

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
	if !taken.heard.take(number, applying, now) {
		return true;
	}

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
		INTERP,
		Value::Float(INTERP_DEFAULT),
		"draw the world this many seconds behind what has arrived, to draw it smoothly",
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

	use colby_core::abi::{Transform, net::MAX_PEERS as PLAYERS};
	use colby_net::{MAX_BASELINE, MAX_COMMAND};

	use super::*;

	/// An address nobody has to be able to reach.
	fn somewhere(port: u16) -> SocketAddr {
		SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
	}

	/// What a client asks for on one step, with four different numbers in it.
	fn wanted(number: u32) -> Command {
		Command {
			// not the number, and not a multiple of it: the two fields are
			// different fields and a test that could not tell them apart would
			// not notice them swapping.
			step: u64::from(number) * 5 + 900,
			number,
			buttons: 1 << (number % 6),
			yaw: f32::from(u16::try_from(number % 11).unwrap_or(0)) * 0.75,
			pitch: -1.25,
		}
	}

	/// A world with nobody in it but the host.
	fn empty_world() -> Box<World> { Box::<World>::default() }

	/// One peer with nothing said either way yet.
	fn fresh_peer() -> Peer {
		Peer {
			address: somewhere(7),
			channel: Channel::new(),
			reliable: Reliable::new(),
			link: Link::new(1),
			heard: Duration::ZERO,
			id: PeerId::NONE,
			named: PeerId::NONE,
			asked: Vec::new(),
			settled: 0,
			confirmed: 0,
			stamped: 0,
			told: Ring::new(),
			kept: Vec::new(),
			taken: Taken {
				heard: Heard::new(),
				theirs: NOTHING,
				baselines: 0,
				behind: 0,
			},
		}
	}

	/// A world of this many bodies, every one of them different from the last.
	fn crowd(bodies: usize) -> Vec<Slot> {
		(0..bodies)
			.map(|slot| {
				let along = f32::from(u16::try_from(slot % 500).unwrap_or(0));

				Some((1, Solid {
					position: [along, 1.5, -along],
					rotation: [0.5, 0.5, 0.5, 0.5],
					velocity: [along, -along, 1.0],
					angular: [1.0, along, 2.0],
					scale: [1.0, 1.0, 1.0],
					sleeping: 1,
					entity: [7, 1],
					owner: [1, 1],
					kind: u32::try_from(slot + 1).unwrap_or(1),
					..Solid::default()
				}))
			})
			.collect()
	}

	/// Answers one snapshot from a peer's point of view: what it now holds.
	///
	/// The receiving half of the conversation, written out here rather than
	/// stood up as a second `Net`, because what these tests are about is
	/// whether the *sender's* idea of what the far end holds is the truth.
	fn applied(held: &mut Vec<Slot>, block: &[u8]) {
		let (snapshot, _) = Snapshot::read(held, block).expect("what was written reads");

		// a baseline describes the whole world, so what is held is emptied
		// before it is applied. @ref `colby_net::snapshot`.
		if snapshot.against == NOTHING {
			held.clear();
		}

		for change in &snapshot.changes {
			let slot = usize::from(change.slot);

			if held.len() <= slot {
				held.resize(slot + 1, None);
			}

			held[slot] = change
				.solid
				.map(|solid| (change.generation, solid));
		}
	}

	/// Whether two tables hold the same bodies, ignoring trailing holes.
	fn alike(left: &[Slot], right: &[Slot]) -> bool {
		let reach = left.len().max(right.len());

		(0..reach)
			.all(|slot| left.get(slot).copied().flatten() == right.get(slot).copied().flatten())
	}

	#[test]
	fn a_world_bigger_than_one_snapshot_reaches_a_peer_over_several() {
		// **the whole of NET-4.** A world past what one block holds used to be
		// refused whole: the peer was owed a baseline that could never be
		// written, so it never learned of a base to answer with and every
		// attempt after it failed the same way, forever.
		let mut peer = fresh_peer();
		let world = crowd(MAX_BASELINE * 2 + 7);
		let mut theirs: Vec<Slot> = Vec::new();
		let mut blocks = 0;

		while !alike(&theirs, &world) {
			blocks += 1;
			assert!(blocks < 10, "it has to converge, and it took {blocks} blocks");

			let mut out = Vec::new();
			let whole = describe(&mut peer, Some(&world), &mut out);

			applied(&mut theirs, &out);
			// and the far end says so, which is what moves the base along.
			// The number just used is one behind the one to come.
			peer.taken.theirs = peer.told.next().saturating_sub(1);

			assert_eq!(whole, alike(&theirs, &world), "it says whether it got to the end");
		}

		assert_eq!(blocks, 3, "two baselines and a bit takes three blocks");
		assert_eq!(peer.taken.behind, 0, "and the count of cut ones is back to nothing");
	}

	#[test]
	fn a_body_that_vanishes_above_the_cut_is_still_reported_gone() {
		// **the trap the carrying over creates and has to answer.** What is
		// kept is what the far end *will hold*, which is the world as far as
		// the block got and whatever the peer already had above that. Keeping
		// only the front instead would say the peer holds nothing up there -
		// so a slot that emptied above the cut would never be written as gone,
		// and the far end would keep the body for the rest of the session.
		let mut peer = fresh_peer();
		let world = crowd(MAX_BASELINE * 2 + 7);
		let mut theirs: Vec<Slot> = Vec::new();

		// first, tell it the whole world, however many blocks that takes.
		for _ in 0..6 {
			let mut out = Vec::new();
			describe(&mut peer, Some(&world), &mut out);
			applied(&mut theirs, &out);
			peer.taken.theirs = peer.told.next().saturating_sub(1);
		}

		assert!(alike(&theirs, &world), "the far end has all of it to start with");

		// now empty a slot near the top, and put a **new occupant** in every
		// slot at the bottom. A new generation is written in full rather than
		// as a difference, so the block fills long before it reaches the
		// removal - which is the only arrangement that exercises this at all.
		// Nudging the bottom bodies instead writes deltas of sixteen bytes,
		// the whole world fits after all, and the test passes whatever the
		// code does.
		let mut shrunk = world.clone();
		let gone = shrunk.len() - 1;
		shrunk[gone] = None;

		for slot in &mut shrunk[..MAX_BASELINE + 10] {
			if let Some((generation, solid)) = slot.as_mut() {
				*generation = 2;
				solid.position[1] += 1.0;
			}
		}

		for _ in 0..6 {
			let mut out = Vec::new();
			describe(&mut peer, Some(&shrunk), &mut out);
			applied(&mut theirs, &out);
			peer.taken.theirs = peer.told.next().saturating_sub(1);
		}

		assert!(
			theirs.get(gone).copied().flatten().is_none(),
			"the body that went away above the cut is gone at the far end too"
		);
		assert!(alike(&theirs, &shrunk), "and the two ends agree about all of it");
	}

	#[test]
	fn a_peer_whose_tail_is_never_reached_is_named_once() {
		// a cut snapshot is ordinary. A run of them that never ends is the
		// front of the table filling every block on its own, and that is worth
		// saying - once, by address, rather than twenty times a second.
		let mut peer = fresh_peer();
		let world = crowd(MAX_BASELINE * 3);

		for _ in 0..STARVED {
			let mut out = Vec::new();

			// never acknowledged, so every block is a baseline and every
			// baseline is cut at the same place: the shape of a tail nothing
			// ever reaches.
			assert!(!describe(&mut peer, Some(&world), &mut out), "cut every time");
		}

		assert_eq!(peer.taken.behind, STARVED, "counted, and this is the moment it is said");

		// and one whole snapshot puts the count back, so the next run of cut
		// ones is reported as its own rather than never again.
		let mut out = Vec::new();
		describe(&mut peer, Some(&crowd(4)), &mut out);

		assert_eq!(peer.taken.behind, 0, "a whole world clears it");
	}

	#[test]
	fn a_world_that_fits_is_told_whole_in_one_block() {
		// the ordinary case, stated so that a change which cut every snapshot
		// would be caught rather than merely slower.
		let mut peer = fresh_peer();
		let world = crowd(20);
		let mut out = Vec::new();

		assert!(describe(&mut peer, Some(&world), &mut out), "it fits");
		assert_eq!(peer.taken.behind, 0, "so nothing was carried over");

		let mut theirs: Vec<Slot> = Vec::new();
		applied(&mut theirs, &out);

		assert!(alike(&theirs, &world), "and the far end has all of it at once");
	}

	/// Whoever the host has admitted that is not the host.
	fn seated(world: &World) -> PeerId {
		world
			.players
			.iter()
			.map(|(peer, _)| peer)
			.find(|peer| !peer.is_host())
			.expect("somebody was admitted")
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
			heard: Heard::new(),
			theirs: NOTHING,
			baselines: 0,
			behind: 0,
		};
		let mut applying = Vec::new();
		let one = world(1.0);
		let two = world(2.0);
		let mut out = Vec::new();

		// snapshot one, a baseline.
		Snapshot::write(1, NOTHING, NOTHING, &[], &one, &mut out);
		assert!(absorb(&mut taken, &out, &mut applying, Duration::ZERO));

		// snapshot two, moved away, taken against one.
		out.clear();
		Snapshot::write(2, 1, NOTHING, &one, &two, &mut out);
		assert!(absorb(&mut taken, &out, &mut applying, Duration::ZERO));
		assert_eq!(taken.heard.holding(), 2);

		// snapshot three, back where it started - and written against ONE,
		// because that is the last the sender heard of. Nothing about the
		// body is in this block at all.
		out.clear();

		let spoken = Snapshot::write(3, 1, NOTHING, &one, &one, &mut out);

		assert_eq!(spoken.spoken, 0, "the sender sees nothing to say, which is the whole trap");
		assert!(absorb(&mut taken, &out, &mut applying, Duration::ZERO));
		assert_eq!(taken.heard.holding(), 3);
		assert_eq!(
			taken.heard.base(3).0,
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
			heard: Heard::new(),
			theirs: NOTHING,
			baselines: 0,
			behind: 0,
		};
		let mut applying = Vec::new();
		let mut out = Vec::new();

		for number in 1..=3 {
			out.clear();
			Snapshot::write(number, NOTHING, NOTHING, &[], &world(1.0), &mut out);
			assert!(absorb(&mut taken, &out, &mut applying, Duration::ZERO));
		}

		assert_eq!(taken.heard.holding(), 3);
		assert_eq!(taken.baselines, 3);

		// and now snapshot two again, which the ring still has a place for.
		out.clear();
		Snapshot::write(2, NOTHING, NOTHING, &[], &world(9.0), &mut out);

		assert!(
			absorb(&mut taken, &out, &mut applying, Duration::ZERO),
			"it is not a fault, only old news"
		);
		assert_eq!(taken.heard.holding(), 3, "and what this end holds did not move");
		assert_eq!(taken.baselines, 3, "nor did the count of worlds it was sent whole");
	}

	/// A client with one host peer, and a world with one body driving one
	/// entity in it.
	///
	/// The body is `Dynamic` on purpose: what the apply has to do first is
	/// stop the solver from simulating something the far end is in charge of,
	/// and a body that was already kinematic could not show that.
	fn client() -> (Net, World, BodyId) {
		let wire = Rc::new(RefCell::new(Wire::default()));
		let mut net = Net::over(Box::new(Loopback::at(somewhere(2), &wire)), false, 2);
		let mut world = World::new();

		// something in the slots before it, so that a slot number, a walk's
		// index and "the first one" are three different numbers. One body in
		// slot nought makes all three the same and hides a mix-up of any two.
		for _ in 0..3 {
			world.bodies.spawn(Body::default());
		}

		let entity = world.entities.spawn();
		let body = world.bodies.spawn(Body {
			kind: BodyKind::Dynamic,
			entity,
			transform: Transform::IDENTITY,
			..Body::default()
		});

		// a window that went looking for a host is not the authority, which is
		// what makes every body in its world a proxy. @ref `App::resumed`.
		world.peer = PeerId::NONE;
		net.introduce(somewhere(1));
		(net, world, body)
	}

	/// A world of one body somewhere along an axis, in the slot a handle names.
	///
	/// The kind is the one a host is simulating, because that is the case the
	/// apply has work to do in - `Solid::default` is a *static* body, and a
	/// fixture built from one would say nothing about a proxy. The speed is
	/// not the position either: they were the same number once and a test
	/// cannot tell two fields apart when they hold one value.
	fn told(body: BodyId, along: f32) -> Vec<Slot> {
		let mut world = vec![None; body.slot() + 1];
		let solid = Solid {
			position: [along, 0.0, 0.0],
			rotation: [0.0, 0.0, 0.0, 1.0],
			scale: [1.0, 1.0, 1.0],
			velocity: [0.0, along * 2.0, 0.0],
			angular: [0.0, 0.0, along * 3.0],
			kind: 2,
			..Solid::default()
		};

		world[body.slot()] = Some((body.generation(), solid));
		world
	}

	/// The same, saying whose it is.
	fn owned(body: BodyId, along: f32, owner: PeerId) -> Vec<Slot> {
		let mut world = told(body, along);

		if let Some(Some((_, solid))) = world.get_mut(body.slot()) {
			solid.owner = [u32::try_from(owner.slot()).unwrap_or(0), owner.generation()];
		}

		world
	}

	/// A peer that is somebody, at a slot nothing in these fixtures sits in.
	fn somebody() -> PeerId { PeerId::from_bits((7_u64 << 32) | 4) }

	/// Every body a world holds, described where it stands.
	fn everything(world: &World) -> Vec<Slot> {
		let mut said = vec![None; world.bodies.slots()];

		for (id, body) in world.bodies.iter() {
			if let Some(slot) = said.get_mut(id.slot()) {
				*slot = Some((id.generation(), Solid::of(body)));
			}
		}

		said
	}

	/// What one endpoint's peer was told, put in as though it had arrived.
	fn arrived(net: &mut Net, number: u32, world: &[Slot], at: Duration) {
		let mut out = Vec::new();
		let (base, against) = net.peers[0].taken.heard.base(NOTHING);

		assert_eq!(against, NOTHING);
		Snapshot::write(number, against, NOTHING, base, world, &mut out);

		let mut applying = Vec::new();

		assert!(absorb(&mut net.peers[0].taken, &out, &mut applying, at));
	}

	#[test]
	fn a_body_the_far_end_owns_is_driven_rather_than_simulated() {
		let (mut net, mut world, body) = client();

		assert_eq!(
			world.bodies.get(body).map(|it| it.kind),
			Some(BodyKind::Dynamic),
			"it starts as something the solver moves"
		);

		arrived(&mut net, 1, &told(body, 4.0), Duration::ZERO);
		net.arrive(&mut world, Duration::ZERO);

		assert_eq!(
			world.bodies.get(body).map(|it| it.kind),
			Some(BodyKind::Kinematic),
			"and stops being one, because the far end is in charge of it"
		);
		assert_eq!(
			world
				.entities
				.transform(
					world
						.bodies
						.get(body)
						.expect("it is there")
						.entity
				)
				.map(|it| it.position.x),
			Some(4.0),
			"what is written is the entity, which is what the solver copies from"
		);
		assert_eq!(
			world.bodies.get(body).map(|it| it.velocity.y),
			Some(8.0),
			"and the speed, which is what it pushes with"
		);
		assert_eq!(world.bodies.get(body).map(|it| it.angular.z), Some(12.0), "and the spin");
	}

	#[test]
	fn a_world_is_put_back_at_the_moment_it_is_asked_for_and_not_the_newest() {
		// the whole of what the delay buys, seen from the world's side: two
		// snapshots a tenth apart, and a moment half way between them puts the
		// body half way rather than at either end.
		let (mut net, mut world, body) = client();
		let tenth = Duration::from_millis(100);

		arrived(&mut net, 1, &told(body, 0.0), Duration::ZERO);
		arrived(&mut net, 2, &told(body, 10.0), tenth);
		net.arrive(&mut world, Duration::from_millis(50));

		assert_eq!(
			world
				.entities
				.transform(
					world
						.bodies
						.get(body)
						.expect("it is there")
						.entity
				)
				.map(|it| it.position.x),
			Some(5.0),
			"half way between the two it was told about"
		);
	}

	#[test]
	fn a_body_that_drives_no_entity_is_written_where_it_stands() {
		// a ragdoll limb, or a collision brush a scene gave no thing to. The
		// solver only ever copies a body *from* an entity, so one with none is
		// never copied over and the body itself is the write that lasts.
		// Setting its kind and then failing to place it would be a body
		// neither the solver nor the wire could move.
		let (mut net, mut world, _) = client();
		let alone = world.bodies.spawn(Body {
			kind: BodyKind::Dynamic,
			..Body::default()
		});

		assert_eq!(
			world.bodies.get(alone).map(|it| it.entity),
			Some(EntityId::NONE),
			"it drives nothing"
		);

		arrived(&mut net, 1, &told(alone, 6.0), Duration::ZERO);
		net.arrive(&mut world, Duration::ZERO);

		assert_eq!(
			world
				.bodies
				.get(alone)
				.map(|it| it.transform.position.x),
			Some(6.0),
			"so it is put where it was said to be, directly"
		);
	}

	#[test]
	fn the_kind_the_far_end_sent_is_kept_unless_it_is_one_the_solver_would_move() {
		// a static map body stays static, and a prop the host froze reads as
		// frozen. Only a body the *host* is simulating has to become kinematic
		// here, and only so that this end's solver does not fight the wire.
		let (mut net, mut world, body) = client();

		for (sent, want) in
			[(0_u32, BodyKind::Static), (1, BodyKind::Kinematic), (2, BodyKind::Kinematic)]
		{
			let mut said = told(body, 1.0);

			said[body.slot()] = said[body.slot()].map(|(generation, mut solid)| {
				solid.kind = sent;
				(generation, solid)
			});
			arrived(&mut net, sent + 1, &said, Duration::ZERO);
			net.arrive(&mut world, Duration::ZERO);

			assert_eq!(
				world.bodies.get(body).map(|it| it.kind),
				Some(want),
				"kind {sent} arrived"
			);
		}
	}

	#[test]
	fn whether_a_proxy_is_asleep_is_the_far_ends_answer_and_not_this_ends() {
		// two sleeping bodies are a pair the broadphase refuses, and nothing
		// wakes a body it never makes a contact for. A body that fell asleep
		// while this end was still simulating it would otherwise stay asleep
		// as a proxy for the life of the process.
		let (mut net, mut world, body) = client();

		if let Some(it) = world.bodies.get_mut(body) {
			it.sleeping = true;
		}

		arrived(&mut net, 1, &told(body, 1.0), Duration::ZERO);
		net.arrive(&mut world, Duration::ZERO);

		assert_eq!(
			world.bodies.get(body).map(|it| it.sleeping),
			Some(false),
			"the far end says it is moving, so it is awake here"
		);
	}

	#[test]
	fn a_slot_this_end_has_no_body_in_is_passed_over_in_silence() {
		// a snapshot says where a thing is and never what it is, so a body
		// that exists on the host and not here cannot be made from one. What
		// must not happen is a panic or a body driven to somebody else's
		// place.
		let (mut net, mut world, body) = client();
		let far = 900;
		let mut said = vec![None; far + 1];

		said[far] = told(body, 7.0)[body.slot()];
		arrived(&mut net, 1, &said, Duration::ZERO);
		net.arrive(&mut world, Duration::ZERO);

		assert_eq!(
			world.bodies.get(body).map(|it| it.kind),
			Some(BodyKind::Dynamic),
			"the body this end does have was not spoken about and was left alone"
		);
	}

	#[test]
	fn a_slot_the_far_end_has_emptied_is_emptied_here_too() {
		// removals have always crossed - the snapshot writes a mark for a slot
		// that went from occupied to empty - and nothing ever acted on one, so
		// a prop the host cleaned up stayed here forever, kinematic and frozen
		// where the last message left it.
		let (mut net, mut world, body) = client();
		let entity = world
			.bodies
			.get(body)
			.map_or(EntityId::NONE, |it| it.entity);

		// **a body behind it in the table**, so that the world which no longer
		// mentions the one under test still reaches past its slot. A table that
		// simply stops short is a far end that has not caught up, which is the
		// other test and the other answer.
		let behind = world.bodies.spawn(Body::default());

		// named, because a window that does not know who it is takes nothing
		// away. The slot is one nobody sits in, so nothing here owns anything.
		world.peer = somebody();

		// **everything this end holds, described**, so that the only slot the
		// second world empties is the one under test. A fixture that quietly
		// dropped the fillers as well would pass on a rule that removed
		// whatever it liked.
		let all = everything(&world);

		arrived(&mut net, 1, &all, Duration::ZERO);
		net.arrive(&mut world, Duration::ZERO);
		assert!(world.bodies.get(body).is_some(), "described, so it is here");
		assert_eq!(net.dropped(), 0, "and nothing else went with it");

		// and now the same world with one slot emptied, which is the far end
		// saying that one is gone rather than saying nothing.
		let mut gone = all;

		gone[body.slot()] = None;
		arrived(&mut net, 2, &gone, colby_core::time::STEP);
		net.arrive(&mut world, colby_core::time::STEP);

		assert!(world.bodies.get(body).is_none(), "and now it is not");
		assert!(!world.entities.alive(entity), "the thing it drove went with it");
		assert!(world.bodies.get(behind).is_some(), "and the one still described stayed");
		assert_eq!(net.dropped(), 1);
	}

	#[test]
	fn a_window_with_no_name_yet_takes_nothing_away() {
		// **the fixture that matters**: every body reads as somebody else's
		// until this end has a name, so a client acting on empty slots before
		// it is seated would delete its own map on the first snapshot.
		let (mut net, mut world, body) = client();

		assert_eq!(world.peer, PeerId::NONE, "a window is nobody until it is told");

		let behind = world.bodies.spawn(Body::default());

		arrived(&mut net, 1, &told(behind, 1.0), Duration::ZERO);
		net.arrive(&mut world, Duration::ZERO);

		assert!(world.bodies.get(body).is_some(), "still here");
		assert_eq!(net.dropped(), 0);
	}

	#[test]
	fn a_body_past_the_end_of_what_the_far_end_holds_is_left_alone() {
		// in range and empty is a removal; past the end is a table that has
		// not caught up, and the two must not be the same answer.
		let (mut net, mut world, body) = client();

		world.peer = somebody();

		// a world that stops before this end's body ever begins. One occupied
		// slot in it, so that what is being tested is the length rather than a
		// snapshot describing nothing at all. `client` puts three fillers in
		// front of its body for exactly this: a slot number and an index are
		// two different numbers here.
		let short = told(
			world
				.bodies
				.iter()
				.next()
				.expect("the fillers are there")
				.0,
			5.0,
		);

		assert!(short.len() <= body.slot(), "the fixture has to stop short to mean anything");
		arrived(&mut net, 1, &short, Duration::ZERO);
		net.arrive(&mut world, Duration::ZERO);

		assert!(world.bodies.get(body).is_some(), "not spoken about is not removed");
		assert_eq!(net.dropped(), 0);
	}

	#[test]
	fn a_window_takes_the_newest_word_about_itself_and_the_delayed_one_about_everybody_else() {
		// what a proxy is drawn at and what an own body is predicted from are
		// two different questions, and asking one of them twice is either a
		// stutter or a guess that starts late.
		let (mut net, mut world, body) = client();
		let mine = somebody();
		let entity = world
			.bodies
			.get(body)
			.map_or(EntityId::NONE, |it| it.entity);

		world.peer = mine;

		// two snapshots a step apart, at three and then at nine, both saying
		// this is the window's own. Asked for the moment of the first, the
		// blend answers three - and an own body has to answer nine.
		arrived(&mut net, 1, &owned(body, 3.0, mine), Duration::ZERO);
		arrived(&mut net, 2, &owned(body, 9.0, mine), colby_core::time::STEP);
		net.arrive(&mut world, Duration::ZERO);

		// where a driven body is actually written: the solver copies a body it
		// does not move *from* the entity, so the entity is the write that
		// lasts. @ref `drive`.
		let at = world
			.entities
			.transform(entity)
			.map_or(f32::NAN, |it| it.position.x);

		assert!(
			(at - 9.0).abs() < 1.0e-4,
			"an own body is corrected against the newest, not the blend: {at}"
		);

		// and the same wire, the same moment, the same two snapshots - with
		// the records saying the body belongs to somebody else.
		let (mut net, mut world, body) = client();
		let entity = world
			.bodies
			.get(body)
			.map_or(EntityId::NONE, |it| it.entity);

		world.peer = mine;
		arrived(&mut net, 1, &told(body, 3.0), Duration::ZERO);
		arrived(&mut net, 2, &told(body, 9.0), colby_core::time::STEP);
		net.arrive(&mut world, Duration::ZERO);

		let at = world
			.entities
			.transform(entity)
			.map_or(f32::NAN, |it| it.position.x);

		assert!(
			(at - 3.0).abs() < 1.0e-4,
			"and everybody else's is drawn where it was a delay ago: {at}"
		);
	}

	#[test]
	fn a_body_arrives_knowing_whose_it_is() {
		// every authority question in the engine is the pair (this peer, that
		// owner), so a receiver that dropped the owner left a client unable to
		// tell its own player from a stranger's.
		let (mut net, mut world, body) = client();
		let mine = somebody();

		world.peer = mine;
		arrived(&mut net, 1, &owned(body, 1.0, mine), Duration::ZERO);
		net.arrive(&mut world, Duration::ZERO);

		assert_eq!(world.bodies.get(body).map(|it| it.owner), Some(mine));
		assert_eq!(
			world.bodies.get(body).map(|it| it.role(mine)),
			Some(Role::AutonomousProxy),
			"and so a client can finally tell its own from everybody else's"
		);
	}

	#[test]
	fn a_slot_whose_occupant_changed_is_not_driven_to_the_dead_ones_place() {
		// the generation is the whole of what says these are two bodies. A
		// snapshot about the one that used to be in this slot must not move
		// the one that is there now.
		let (mut net, mut world, body) = client();
		let mut said = told(body, 7.0);

		said[body.slot()] = said[body.slot()].map(|(generation, solid)| (generation + 1, solid));
		arrived(&mut net, 1, &said, Duration::ZERO);
		net.arrive(&mut world, Duration::ZERO);

		assert_eq!(
			world.bodies.get(body).map(|it| it.kind),
			Some(BodyKind::Dynamic),
			"a different occupant, so this one is not touched"
		);
	}

	#[test]
	fn a_world_that_joined_one_is_no_longer_the_authority_over_anything() {
		let mut world = World::new();

		assert!(world.peer.is_host(), "a world on its own is the authority");
		assert!(
			world.bodies.spawn(Body::default()).is_some(),
			"and a body in it is one this end decides"
		);

		joined(&mut world);

		assert!(!world.peer.is_host(), "and a world that went looking for one is not");

		let body = world.bodies.spawn(Body::default());

		assert!(
			!world
				.bodies
				.get(body)
				.expect("it is there")
				.role(world.peer)
				.local(),
			"so every body in it is somebody else's to decide"
		);
	}

	#[test]
	fn a_body_this_end_has_authority_over_is_left_alone() {
		// the role is what decides, not the wire. Today a client's world says
		// it is nobody, so nothing is local and this arm is unreachable from
		// the outside - the commit that gives a client its own name is what
		// makes it ordinary, and the rule has to be right before then.
		let (mut net, mut world, body) = client();

		// this end is the authority again, which is what a peer owning a body
		// will look like once there is a handshake to say so.
		world.peer = PeerId::HOST;

		arrived(&mut net, 1, &told(body, 4.0), Duration::ZERO);
		net.arrive(&mut world, Duration::ZERO);

		assert_eq!(
			world.bodies.get(body).map(|it| it.kind),
			Some(BodyKind::Dynamic),
			"a body this end decides is not driven by what somebody said about it"
		);
	}

	#[test]
	fn a_host_puts_nothing_back_even_if_its_world_forgot_it_is_one() {
		// the endpoint knows what it is, and that is what has to stand between
		// a client's claim and a host's world. `World::peer` is a public field
		// a game module writes, so it cannot be the only thing that does.
		let wire = Rc::new(RefCell::new(Wire::default()));
		let mut host = Net::over(Box::new(Loopback::at(somewhere(1), &wire)), true, 1);
		let mut world = World::new();
		let entity = world.entities.spawn();
		let body = world.bodies.spawn(Body {
			kind: BodyKind::Dynamic,
			entity,
			..Body::default()
		});

		// the inconsistent state on purpose: a host whose world does not say
		// so. Nothing produces this today and something might.
		world.peer = PeerId::NONE;
		host.introduce(somewhere(2));
		arrived(&mut host, 1, &told(body, 4.0), Duration::ZERO);
		host.arrive(&mut world, Duration::ZERO);

		assert_eq!(
			world.bodies.get(body).map(|it| it.kind),
			Some(BodyKind::Dynamic),
			"a host is the end that says what the world is, whatever its world thinks"
		);
	}

	#[test]
	fn a_host_is_told_nothing_and_puts_nothing_back() {
		// the one direction this does not go. A host is the end that says what
		// the world is, so a world arriving at one would be the world
		// overwriting itself with whatever a client claimed.
		let wire = Rc::new(RefCell::new(Wire::default()));
		let mut host = Net::over(Box::new(Loopback::at(somewhere(1), &wire)), true, 1);
		let mut world = World::new();
		let entity = world.entities.spawn();
		let body = world.bodies.spawn(Body {
			kind: BodyKind::Dynamic,
			entity,
			..Body::default()
		});

		host.introduce(somewhere(2));
		arrived(&mut host, 1, &told(body, 4.0), Duration::ZERO);
		host.arrive(&mut world, Duration::ZERO);

		assert_eq!(
			world.bodies.get(body).map(|it| it.kind),
			Some(BodyKind::Dynamic),
			"nothing a peer said moved anything here"
		);
	}

	#[test]
	fn how_far_behind_to_draw_is_read_from_the_table_and_kept_sensible() {
		let mut cvars = table();

		assert_eq!(behind(&cvars), Duration::from_secs_f32(INTERP_DEFAULT));

		assert!(cvars.set(INTERP, "0.25"));
		assert_eq!(behind(&cvars), Duration::from_millis(250));

		// a delay past what the ring can answer would ask for a moment older
		// than anything still here, which reads as the world having stopped.
		assert!(cvars.set(INTERP, "90"));
		assert_eq!(behind(&cvars), Duration::from_secs_f32(MAX_INTERP));

		assert!(cvars.set(INTERP, "-3"));
		assert_eq!(behind(&cvars), Duration::ZERO, "and behind is never ahead");
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
	fn a_message_that_stops_after_the_ring_costs_the_message_and_not_the_peer() {
		// a peer that stops after its reliable block hands the reader a block
		// of commands that is not there. That is refused - and **the peer is
		// kept**, which is the one place this wire treats a fault as the
		// message's rather than the conversation's. A block of commands
		// carries no state from one to the next, so throwing it away costs
		// what it was carrying and nothing else; dropping the peer instead
		// would let one flipped bit end a session.
		//
		// The snapshot in the same message goes with it, because the block of
		// commands is what says where the snapshot begins.
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

		assert_eq!(host.peers(), 1, "the peer is still here");
		assert_eq!(host.forgotten(), 0, "and was not dropped for it");
		assert_eq!(host.garbled(), 1, "the message was thrown away and counted");
		assert_eq!(host.holding(0), NOTHING, "and the snapshot in it went too");
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
		// a block of commands that reads: nobody, nothing settled, none of
		// them. Fourteen bytes of head, which is what almost every message
		// actually carries - and it has to be here, because it is what says
		// where the snapshot below begins.
		broken.extend_from_slice(&[0; 14]);
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
		// an address after it rather than a port, so that the flag is what
		// this turns on: a word no reader could parse would refuse either way.
		assert_eq!(
			asked_for(&["--join".to_owned(), "127.0.0.1:27015".to_owned()]),
			None,
			"and not somebody else's flag"
		);
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

	#[test]
	fn a_host_names_whoever_turns_up_and_a_client_takes_the_name() {
		let (mut host, mut client, _wire) = two();
		let (mut served, mut window) = (empty_world(), empty_world());

		joined(&mut window);
		assert_eq!(window.peer, PeerId::NONE, "a client is nobody until it is told");

		// the client says hello, the host hears it and seats it.
		round(&mut host, &mut client, 1);
		host.seat(&mut served);

		let named = seated(&served);

		assert!(!named.is_host(), "it is not handed the host's own name");
		assert_eq!(named.slot(), 1, "the lowest slot above the host's");

		// and then the host says so, and the client hears it.
		round(&mut host, &mut client, 2);
		client.seat(&mut window);

		assert_eq!(window.peer, named, "the client is who the host says it is");
		assert!(!window.peer.is_host(), "which is not the host");
	}

	#[test]
	fn what_a_client_asks_for_is_filed_under_the_name_the_host_minted() {
		let (mut host, mut client, _wire) = two();
		let mut served = empty_world();

		client.ask(&[wanted(4), wanted(5), wanted(6)]);
		round(&mut host, &mut client, 1);
		host.seat(&mut served);

		let peer = seated(&served);

		assert_eq!(served.commands.kept(peer), &[wanted(4), wanted(5), wanted(6)]);
		assert_eq!(host.filed(), 3);

		// the same window again, which is what the next message carries: every
		// one of them is a repeat and none of them is filed twice.
		round(&mut host, &mut client, 2);
		host.seat(&mut served);

		assert_eq!(served.commands.kept(peer).len(), 3, "the same three and no more");
		assert_eq!(host.filed(), 3, "and nothing was taken the second time");

		// and one more on the end, which is not a repeat.
		client.ask(&[wanted(5), wanted(6), wanted(7)]);
		round(&mut host, &mut client, 3);
		host.seat(&mut served);

		assert_eq!(served.commands.newest(peer), 7);
		assert_eq!(host.filed(), 4);
	}

	#[test]
	fn a_peer_that_goes_gives_its_slot_back() {
		let (mut host, mut client, _wire) = two();
		let mut served = empty_world();

		round(&mut host, &mut client, 1);
		host.seat(&mut served);

		let peer = seated(&served);

		assert!(served.players.here(peer));

		// long enough that it has said nothing for a while. The world is told
		// on the next seat rather than when the wire notices, because the wire
		// is drained where there is no world to tell.
		host.receive(QUIET + Duration::from_secs(1));
		assert_eq!(host.peers(), 0, "the wire has let it go");
		assert!(served.players.here(peer), "and the world has not been told yet");

		host.seat(&mut served);
		assert!(!served.players.here(peer), "and now it has");

		// and the slot comes back, to somebody else.
		client.introduce(somewhere(1));
		round(&mut host, &mut client, 100);
		host.seat(&mut served);

		let next = seated(&served);

		assert_eq!(next.slot(), peer.slot(), "the same slot");
		assert_ne!(next, peer, "and not the same peer");
	}

	/// The whole round trip, in the order the runner really does it.
	///
	/// **A client cannot record its own commands until it has been named**,
	/// and that falls out of the ring being keyed by peer: nobody has no ring,
	/// so the first messages a window sends carry nothing at all. Two round
	/// trips of silence at the start of a session is the cost, and it is the
	/// handshake's to remove rather than this commit's.
	#[test]
	fn a_host_says_what_it_has_settled_and_a_client_stops_asking_for_it() {
		let (mut host, mut client, _wire) = two();
		let (mut served, mut window) = (empty_world(), empty_world());

		joined(&mut window);

		// nobody yet, so there is nowhere to put a command and nothing to ask
		// for.
		assert!(!window.commands.push(window.peer, wanted(1)), "nobody has no ring");

		round(&mut host, &mut client, 1);
		host.seat(&mut served);
		round(&mut host, &mut client, 2);
		client.seat(&mut window);
		assert!(window.peer.is_some(), "and now the window is somebody");

		// which is where the game would put its own move. Two of them, so that
		// a mark landing on the wrong one is visible.
		for number in [1_u32, 2] {
			assert!(window.commands.push(window.peer, wanted(number)));
		}

		client.ask(window.commands.unsettled(window.peer));
		round(&mut host, &mut client, 3);
		host.seat(&mut served);

		let peer = seated(&served);

		assert_eq!(served.commands.kept(peer), &[wanted(1), wanted(2)]);
		assert_eq!(window.commands.settled(window.peer), 0, "and nothing has run them yet");

		// the host runs them, which is what a step does at the end of itself.
		assert!(
			served
				.commands
				.settle(peer, served.commands.newest(peer))
		);

		host.seat(&mut served);
		round(&mut host, &mut client, 4);
		client.seat(&mut window);

		assert_eq!(window.commands.settled(window.peer), 2, "the host has said so");
		assert!(
			window.commands.unsettled(window.peer).is_empty(),
			"so the window stops asking for them"
		);

		// and a third, which is asked for on its own rather than behind the
		// two that are done with.
		assert!(window.commands.push(window.peer, wanted(3)));
		assert_eq!(window.commands.unsettled(window.peer), &[wanted(3)]);
	}

	#[test]
	fn a_window_stops_resending_what_was_heard_and_stops_guessing_at_what_was_drawn() {
		// **the two marks come from two different places and only a wire where
		// they disagree can tell.** An acknowledgement rides every message and
		// a world rides one in a handful, so a client that took the newest
		// acknowledgement as the base for its guess would be running that guess
		// forward from a picture the acknowledgement has already overtaken -
		// which is a smear of whole units on any wire with delay on it.
		let (mut host, mut client, _wire) = two();
		let (mut served, mut window) = (empty_world(), empty_world());

		joined(&mut window);
		round(&mut host, &mut client, 1);
		host.seat(&mut served);
		round(&mut host, &mut client, 2);
		client.seat(&mut window);

		for number in 1..=3 {
			assert!(window.commands.push(window.peer, wanted(number)));
		}

		client.ask(window.commands.unsettled(window.peer));
		round(&mut host, &mut client, 3);
		host.seat(&mut served);

		let peer = seated(&served);

		assert_eq!(served.commands.kept(peer).len(), 3, "all three arrived");

		// the host runs two of them and describes the world it made.
		assert!(served.commands.settle(peer, 2));
		host.seat(&mut served);

		let described = vec![None; 4];
		let at = colby_core::time::STEP * 4;

		host.send(at, Some(&described));
		client.send(at, None);
		host.receive(at);
		client.receive(at);
		client.seat(&mut window);

		assert_eq!(window.commands.settled(window.peer), 2, "two were heard");
		assert_eq!(window.commands.based(window.peer), 2, "and the picture shows two");

		// and then it runs the third and says so in a message carrying no
		// world at all, which is what four messages in five look like.
		assert!(served.commands.settle(peer, 3));
		host.seat(&mut served);
		round(&mut host, &mut client, 5);
		client.seat(&mut window);

		assert_eq!(window.commands.settled(window.peer), 3, "three were heard");
		assert_eq!(
			window.commands.based(window.peer),
			2,
			"but the newest picture is still the one that showed two"
		);
		assert_eq!(
			window.commands.unbased(window.peer),
			&[wanted(3)],
			"so the third is still a guess"
		);
		assert!(
			window.commands.unsettled(window.peer).is_empty(),
			"and none of them is worth sending again"
		);
	}

	#[test]
	fn a_block_of_commands_that_is_nonsense_costs_the_message_and_not_the_peer() {
		let wire = Rc::new(RefCell::new(Wire::default()));
		let (one, two) = (somewhere(1), somewhere(2));
		let mut host = Net::over(Box::new(Loopback::at(one, &wire)), true, 1);
		let mut liar = Net::over(Box::new(Loopback::at(two, &wire)), false, 2);

		liar.introduce(one);

		let mut lying = Vec::new();

		// a reliable block that parses and owes nothing.
		lying.extend_from_slice(&0_u32.to_le_bytes());
		lying.extend_from_slice(&1_u32.to_le_bytes());
		lying.extend_from_slice(&0_u16.to_le_bytes());
		// then a block of commands claiming more than a ring can hold, which
		// is the one a peer could send by having a bit flipped rather than by
		// being written to lie.
		lying.extend_from_slice(&[0; 12]);
		lying.extend_from_slice(&u16::MAX.to_le_bytes());

		liar.hand(one, &lying, Duration::ZERO);
		host.receive(Duration::ZERO);

		assert_eq!(host.peers(), 1, "the peer is kept");
		assert_eq!(host.forgotten(), 0);
		assert_eq!(host.garbled(), 1);

		// and the conversation carries on: the next honest message is taken.
		let mut served = empty_world();

		liar.ask(&[wanted(3)]);
		round(&mut host, &mut liar, 1);
		host.seat(&mut served);

		assert_eq!(
			served.commands.kept(seated(&served)),
			&[wanted(3)],
			"one bad block is one bad block"
		);
	}

	#[test]
	fn a_peer_may_call_the_games_commands_and_none_of_the_engines() {
		let mut world = empty_world();

		world.cvars.attribute(Owner::Engine);
		world
			.cvars
			.command("quit", nothing, "the engine's own");
		world
			.cvars
			.var("sim.speed", Value::Float(1.0), "a number the engine owns");
		world.cvars.attribute(Owner::Module);
		world
			.cvars
			.command("game.poke", nothing, "a thing a game exposed");
		world
			.cvars
			.var("game.speed", Value::Float(1.0), "a number a game exposed");
		world.cvars.attribute(Owner::Engine);

		assert!(allowed(&world.cvars, "game.poke"), "a module command is a player's to call");
		assert!(allowed(&world.cvars, "game.poke 3 upper"), "arguments and all");

		assert!(!allowed(&world.cvars, "quit"), "and the engine's are not");
		assert!(!allowed(&world.cvars, "sim.speed 4"));
		assert!(
			!allowed(&world.cvars, "game.speed 100"),
			"nor is setting a value, even one the game registered"
		);
		assert!(!allowed(&world.cvars, "game.nosuch"), "nor a name nothing answers to");
		assert!(!allowed(&world.cvars, ""), "nor nothing at all");
		assert!(!allowed(&world.cvars, "   "), "however it is spelled");

		// a line is several statements, and running the half in front of a
		// refusal is the worst answer available.
		assert!(allowed(&world.cvars, "game.poke; game.poke"), "two of its own is fine");
		assert!(
			!allowed(&world.cvars, "game.poke; quit"),
			"and one of somebody else's refuses the line rather than half of it"
		);
		assert!(!allowed(&world.cvars, "quit; game.poke"), "whichever end it is on");
	}

	/// A restore is the one thing that takes a name away under a live
	/// conversation, and the endpoint holds names.
	///
	/// `scene load` puts the peer table back as a file had it, and a file
	/// written before anybody connected describes nobody - so every peer on
	/// the wire is holding a handle the world no longer knows. Two things go
	/// wrong if the endpoint trusts its own copy: the world's walk over its
	/// players never reaches that peer, so its ring is never settled and it
	/// resends forever; and the next slot handed out is the very same handle,
	/// so two live peers share one ring.
	#[test]
	fn a_restore_reseats_the_peers_that_are_still_here_rather_than_leaving_ghosts() {
		let wire = Rc::new(RefCell::new(Wire::default()));
		let (at_host, at_one, at_two) = (somewhere(1), somewhere(2), somewhere(3));
		let mut host = Net::over(Box::new(Loopback::at(at_host, &wire)), true, 1);
		let mut one = Net::over(Box::new(Loopback::at(at_one, &wire)), false, 2);
		let mut two = Net::over(Box::new(Loopback::at(at_two, &wire)), false, 3);
		let mut served = empty_world();

		one.introduce(at_host);
		one.ask(&[wanted(1)]);
		round(&mut host, &mut one, 1);
		host.seat(&mut served);

		let before = seated(&served);

		assert!(served.players.here(before));

		// what `scene::restore` does, in the order it does it: the rings go,
		// then the peer table comes back as some file had it. Every generation
		// nothing and no blocks is what a description written before this peer
		// existed hands over.
		served.commands.clear();
		served.players.restore(&[0; PLAYERS], &[]);
		assert!(!served.players.here(before), "the world has forgotten it");

		// and now a second window turns up, which is the case that used to
		// hand out a name somebody was already using.
		two.introduce(at_host);
		two.ask(&[wanted(9)]);
		one.ask(&[wanted(2)]);
		round(&mut host, &mut one, 2);
		round(&mut host, &mut two, 3);
		host.seat(&mut served);

		let names: Vec<PeerId> = served
			.players
			.iter()
			.map(|(peer, _)| peer)
			.filter(|peer| !peer.is_host())
			.collect();

		assert_eq!(names.len(), 2, "both windows are somebody");
		assert_ne!(names[0], names[1], "and they are not the same somebody");

		for peer in &names {
			assert!(served.players.here(*peer), "each is one the world knows");
			assert_eq!(served.commands.kept(*peer).len(), 1, "with a ring of its own");
		}
	}

	/// The one value on this wire that the authority model turns on.
	#[test]
	fn a_window_refuses_a_host_that_says_the_window_is_the_host() {
		let wire = Rc::new(RefCell::new(Wire::default()));
		let (at_host, at_window) = (somewhere(1), somewhere(2));
		let mut liar = Net::over(Box::new(Loopback::at(at_host, &wire)), true, 1);
		let mut client = Net::over(Box::new(Loopback::at(at_window, &wire)), false, 2);
		let mut window = empty_world();

		joined(&mut window);
		client.introduce(at_host);

		// so the liar has somebody to answer.
		round(&mut liar, &mut client, 1);
		client.seat(&mut window);
		assert_eq!(window.peer, PeerId::NONE, "nothing honest has named it yet");

		// every name at the host's slot, whatever generation it wears: the one
		// nothing mints, one a table in the house style would mint first, and
		// one in between.
		for claim in [PeerId::HOST, PeerId::from_bits(1 << 32), PeerId::from_bits(7 << 32)] {
			let mut payload = Vec::new();

			// a reliable block owing nothing
			payload.extend_from_slice(&0_u32.to_le_bytes());
			payload.extend_from_slice(&1_u32.to_le_bytes());
			payload.extend_from_slice(&0_u16.to_le_bytes());
			Block::write(claim, 0, &[], &mut payload);
			// and a snapshot head describing nothing
			payload.extend_from_slice(&NOTHING.to_le_bytes());
			payload.extend_from_slice(&NOTHING.to_le_bytes());
			payload.extend_from_slice(&NOTHING.to_le_bytes());
			payload.extend_from_slice(&0_u16.to_le_bytes());

			liar.hand(at_window, &payload, Duration::ZERO);
			client.receive(Duration::ZERO);
			client.seat(&mut window);

			assert_eq!(window.peer, PeerId::NONE, "{claim:?} is refused");
			assert!(!window.peer.is_host(), "and the window is not the authority");
		}

		// and an honest name at a slot a table really does hand out is taken,
		// so the guard is the slot rather than a refusal of everything.
		let mut payload = Vec::new();

		payload.extend_from_slice(&0_u32.to_le_bytes());
		payload.extend_from_slice(&1_u32.to_le_bytes());
		payload.extend_from_slice(&0_u16.to_le_bytes());
		Block::write(PeerId::from_bits((4 << 32) | 3), 0, &[], &mut payload);
		payload.extend_from_slice(&NOTHING.to_le_bytes());
		payload.extend_from_slice(&NOTHING.to_le_bytes());
		payload.extend_from_slice(&NOTHING.to_le_bytes());
		payload.extend_from_slice(&0_u16.to_le_bytes());

		liar.hand(at_window, &payload, Duration::ZERO);
		client.receive(Duration::ZERO);
		client.seat(&mut window);

		assert_eq!(window.peer.slot(), 3, "a name at an ordinary slot is taken");
		assert_eq!(window.peer.generation(), 4);
	}

	/// The list of who has gone is only ever about somebody who had a name.
	///
	/// Which is what keeps it bounded where nothing drains it: the
	/// two-endpoint run seats nobody and empties this from nowhere, so an
	/// endpoint that never names anybody must never put anything in it.
	#[test]
	fn a_peer_that_was_never_named_is_not_remembered_as_having_gone() {
		let (mut host, mut client, _wire) = two();

		round(&mut host, &mut client, 1);
		assert_eq!(host.peers(), 1, "somebody turned up");
		assert_eq!(host.departed(), 0);

		// and goes again without ever having been seated, which is every peer
		// on an endpoint with no world at all.
		host.receive(QUIET + Duration::from_secs(1));

		assert_eq!(host.peers(), 0, "and went");
		assert_eq!(
			host.departed(),
			0,
			"with no name to give back, so nothing is waiting for a world that may never come"
		);

		// a peer that *was* named is remembered, which is what says the guard
		// above is about the name rather than about nothing being remembered.
		let mut served = empty_world();

		client.introduce(somewhere(1));
		round(&mut host, &mut client, 100);
		host.seat(&mut served);
		assert_eq!(host.departed(), 0, "seating empties it");

		host.receive(Duration::from_secs(200) + QUIET);
		assert_eq!(host.departed(), 1, "and a peer with a name is waiting to be given back");
	}

	#[test]
	fn a_window_whose_host_has_gone_is_nobody_again() {
		let (mut host, mut client, _wire) = two();
		let (mut served, mut window) = (empty_world(), empty_world());

		joined(&mut window);
		round(&mut host, &mut client, 1);
		host.seat(&mut served);
		round(&mut host, &mut client, 2);
		client.seat(&mut window);

		assert!(window.peer.is_some(), "named while somebody was talking");

		// the host says nothing for long enough that the window forgets it.
		client.receive(QUIET + Duration::from_secs(1));
		assert_eq!(client.peers(), 0);

		client.seat(&mut window);
		assert_eq!(
			window.peer,
			PeerId::NONE,
			"a name that outlived the conversation it came from is a window that thinks it \
			 still owns things"
		);
	}

	#[test]
	fn a_peer_with_nowhere_to_put_its_commands_is_still_talked_to() {
		let mut served = empty_world();

		// every slot above the host's, so the next peer to turn up cannot be
		// named at all.
		while served.players.admit().is_some() {}

		let (mut host, mut client, _wire) = two();

		client.ask(&[wanted(1), wanted(2)]);
		round(&mut host, &mut client, 1);
		host.seat(&mut served);

		assert_eq!(host.filed(), 0, "there was nowhere to file them");
		assert!(
			host.holding_commands(0).is_empty(),
			"and they are dropped rather than queued until the conversation ends"
		);
		assert_eq!(host.peers(), 1, "but the peer is still talked to");

		// and it keeps being talked to rather than filling a queue for as long
		// as it keeps asking.
		client.ask(&[wanted(3)]);
		round(&mut host, &mut client, 2);
		host.seat(&mut served);

		assert!(host.holding_commands(0).is_empty());
		assert_eq!(host.peers(), 1);
	}

	#[test]
	fn a_line_that_crossed_runs_as_the_peer_that_said_it() {
		// **the console is this engine's RPC layer**, and a layer that ran
		// every call as whoever is at the host's keyboard is not one. Every
		// command in the sandbox reads `World::peer` to find out whose player
		// it is about - which crosshair a nameless `game.grab` means, where in
		// front of somebody a `game.spawn` lands - so this is the difference
		// between a peer asking for something and a peer making the host do
		// something to itself.
		let (mut host, mut client, _wire) = two();
		let (mut served, mut window) = (empty_world(), empty_world());

		served.cvars.attribute(Owner::Module);
		served
			.cvars
			.command("game.whoami", whoami, "writes down whose the world said it was");
		served.cvars.attribute(Owner::Engine);

		joined(&mut window);
		round(&mut host, &mut client, 1);
		client
			.say("game.whoami")
			.expect("the ring took it");
		round(&mut host, &mut client, 2);

		host.seat(&mut served);

		let named = served
			.players
			.iter()
			.map(|(peer, _)| peer)
			.find(|peer| !peer.is_host())
			.expect("the host seated whoever turned up");

		assert!(served.peer.is_host(), "the machine running it is still the host");
		obey(&mut served, &host);

		let ran =
			PeerId::from_bits((u64::from(served.owed_steps) << 32) | u64::from(served.contacts));

		assert_eq!(served.steps, 1, "it ran, once");
		assert_eq!(ran, named, "and as the peer that asked, not as the host");
		assert!(
			served.peer.is_host(),
			"and the field naming this machine is put back afterwards"
		);
	}

	#[test]
	fn a_line_from_a_peer_with_no_name_is_not_run_as_somebody_else() {
		// a peer is given its name by the block of commands in a message, and
		// the reliable ring in front of that block is read first - so the very
		// first thing anybody ever says arrives before they are anybody.
		// Running it would run it as whoever the host is.
		let (mut host, mut client, _wire) = two();
		let mut served = empty_world();

		served.cvars.attribute(Owner::Module);
		served
			.cvars
			.command("game.whoami", whoami, "writes down whose the world said it was");
		served.cvars.attribute(Owner::Engine);

		round(&mut host, &mut client, 1);
		client
			.say("game.whoami")
			.expect("the ring took it");
		round(&mut host, &mut client, 2);

		assert_eq!(host.said().len(), 1, "it crossed");

		// and deliberately *not* seated, which is the state the first message
		// of a conversation arrives in.
		obey(&mut served, &host);
		assert_eq!(served.steps, 0, "a line from nobody is not run at all");
		assert_eq!(served.contacts, 0, "and so leaves no mark of any kind");
	}

	#[test]
	fn a_client_still_listens_to_the_host_it_went_looking_for() {
		// a host that took a moment longer to start than the client did used
		// to cost the whole process: the client went quiet, forgot the only
		// peer it had, and then refused every datagram that address ever sent
		// because it was no longer one it knew.
		let (mut host, mut client, _wire) = two();
		let quiet = QUIET + Duration::from_secs(1);

		// they find each other first, so that the host has somebody to answer.
		round(&mut host, &mut client, 1);
		assert_eq!(client.peers(), 1, "it found a host");

		client.receive(quiet);
		assert_eq!(client.peers(), 0, "and gave up on one that then said nothing");

		// and now the host speaks, from the same address as before.
		host.send(quiet, None);
		client.receive(quiet);

		assert_eq!(client.peers(), 1, "the address it went looking for is still somebody");
	}

	#[test]
	fn a_host_runs_what_a_peer_may_ask_for_and_a_client_runs_nothing() {
		let (mut host, mut client, _wire) = two();
		let (mut served, mut window) = (empty_world(), empty_world());

		for world in [&mut served, &mut window] {
			world.cvars.attribute(Owner::Module);
			world
				.cvars
				.command("game.mark", mark, "leaves a mark");
			world.cvars.attribute(Owner::Engine);
			world
				.cvars
				.command("quit", nothing, "the engine's own");
		}

		joined(&mut window);

		// one round first: a host queues a line for every peer it has, and it
		// has none until somebody has said something to it.
		round(&mut host, &mut client, 1);

		// the refused one first, so a gate that stopped after the first line
		// would never reach the one that has to run.
		client.say("quit").expect("the ring took it");
		client.say("game.mark").expect("the ring took it");
		host.say("game.mark").expect("the ring took it");
		round(&mut host, &mut client, 2);

		// what the host does with what a peer asked for.
		assert_eq!(host.said().len(), 2, "both crossed");

		// seated first, exactly as the loop does it: a line is run *as* the
		// peer that said it, and a peer with no name has no player for a
		// command to be about. @ref `crate::host`, where the order is
		// receive, seat, obey.
		host.seat(&mut served);
		obey(&mut served, &host);
		assert_eq!(served.contacts, MARK, "the one it may run, ran");
		assert_eq!(served.owed_steps, 0, "and the one it may not, did not");

		// and what a client does with what a host asked for, which is nothing:
		// the same line, the same table, the other end of the wire.
		assert_eq!(client.said().len(), 1, "it crossed the other way too");
		obey(&mut window, &client);
		assert_eq!(window.contacts, 0, "a client runs nothing a host types at it");
	}

	/// What `mark` leaves behind, which is not a number anything else here
	/// writes.
	const MARK: u32 = 0x00BE_EF00;

	/// A console command that leaves a mark, so a test can tell that it ran.
	///
	/// # Safety
	///
	/// As `ConsoleFn`: both pointers are live for the duration of the call.
	unsafe extern "C-unwind" fn mark(world: *mut World, _args: *const colby_core::abi::Args) {
		// SAFETY: the console hands over a live world for the duration of the
		// call, exactly as the host's own commands are handed one.
		let world = unsafe { &mut *world };

		world.contacts = MARK;
	}

	/// A console command that writes down whose the world said it was.
	///
	/// The whole of what "the console is the RPC layer" means, made into a
	/// number a test can compare: every command in the game reads
	/// [`World::peer`] to find out whose player it is about, so a line run with
	/// the wrong one in place is a line that acted on the wrong person.
	///
	/// # Safety
	///
	/// As `ConsoleFn`: both pointers are live for the duration of the call.
	unsafe extern "C-unwind" fn whoami(world: *mut World, _args: *const colby_core::abi::Args) {
		// SAFETY: as `mark`.
		let world = unsafe { &mut *world };

		// **and a mark that says it ran at all**, which is separate from the
		// name it ran under on purpose: a peer of nobody writes zeroes, and a
		// test that read only those could not tell a line that was refused
		// from one that ran as nobody. The two are the whole question here.
		world.steps = world.steps.saturating_add(1);
		world.contacts = u32::try_from(world.peer.to_bits() & 0xFFFF_FFFF).unwrap_or(0);
		world.owed_steps = world.peer.generation();
	}

	/// A console command that does nothing, so the gate has a name to find.
	///
	/// # Safety
	///
	/// As `ConsoleFn`: both pointers are live for the duration of the call.
	unsafe extern "C-unwind" fn nothing(world: *mut World, _args: *const colby_core::abi::Args) {
		// not nothing at all: a command that really did nothing could not tell
		// "it was refused" from "it ran and had no effect".
		//
		// SAFETY: as above.
		let world = unsafe { &mut *world };

		world.owed_steps = 1;
	}
}
