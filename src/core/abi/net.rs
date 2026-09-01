//! Who somebody is, what part this process plays in what they own, and what
//! they are asking for.
//!
//! Four types. A [`PeerId`] names one endpoint of a conversation; a [`Role`]
//! says what this process is entitled to do about a thing that peer owns; a
//! [`Command`] is the whole of what one client asks for during one step, and
//! [`Commands`] is the ring of them kept per peer. Nothing here opens a socket
//! and nothing here is sent: the wire is a crate of its own, and putting a
//! command on it is a later commit. What is here is the vocabulary the rest of
//! it is written in.
//!
//! **Authority runs one way and asking runs the other**, and these are the two
//! halves. A snapshot goes out from the machine that simulates and says where
//! everything ended up; a command comes back from a machine that does not and
//! says what somebody wanted. Neither is the other's answer, and neither may
//! be built out of the other.
//!
//! **The host is the only thing that simulates.** That is the whole authority
//! model in one line, and every role below follows from it: a host is the
//! [`Role::Authority`] over everything in its world, including a body a client
//! owns, and a client works out nothing except what it owns itself.
//!
//! **A role is worked out, never stored.** A body carries an owner and the
//! world carries who this process is; those two are the whole of the input,
//! and a third field saying "and my role is" is one that can disagree with
//! them, with one of the two then having to win. That is the argument that
//! took a redundant flag bit back out of the mesh format.
//!
//! Of the five systems read for this, exactly one stores a role - and it
//! stores a **pair**, its own and the far end's, replicates both, and swaps
//! them as they are read so that each side's "mine and yours" comes out right.
//! It has to: its notion of an owner is a pointer to another object, and
//! asking who that is means walking the chain of them up to a connection,
//! which exists only on the machine that accepted it. A client cannot work the
//! answer out from what it holds, so it is told instead. Here an owner is
//! a [`PeerId`], which is the same number on both machines, so both machines
//! reach the same answer out of what they already hold. That engine also pays
//! for the stored pair in a way worth knowing about: the far end's role is
//! *projected per connection* on the way out, so one thing is written as
//! autonomous to the peer that owns it and as simulated to everybody else, and
//! the real value is put back afterwards. A derived role gets that for
//! nothing, because each peer works out its own answer and no answer is ever
//! sent. Of the other four, one recomputes cached booleans inside the setter
//! that changes the owner - a derivation with a memo in front of it - and one
//! derives from which markers a receiver was handed. The last two have no
//! per-object owner at all: one compares two integers at each of the several
//! dozen places it wants to know whether something is the local player, and
//! the other is host-authoritative with no such notion in it anywhere.
//!
//! **Three roles and not four.** The engine with the enum has a fourth
//! meaning "replicated to nobody", and it earns its place there by gating two
//! real things: whether a remote call is absorbed, and whether a thing is in
//! the replication list at all. Both of those are deferred here with named
//! triggers rather than fields, so a fourth would be a variant every match
//! had to carry and nothing could ever produce.
//!
//! ```text
//!   let peer = world.peer;
//!
//!   for (_, body) in world.bodies.iter() {
//!       match body.role(peer) {
//!           | Role::SimulatedProxy  => ..  // be told where it went
//!           | Role::AutonomousProxy => ..  // work it out; expect a correction
//!           | Role::Authority       => ..  // work it out; tell everybody
//!       }
//!   }
//! ```

/// How many peers one world keeps a block of gameplay state for.
///
/// The host and eight clients. The runner's own ceiling is eight *clients* on
/// one socket, and this is that number seen from the other side, with the
/// host's own slot added: slot zero is [`PeerId::HOST`]'s and a client is never
/// given it. Bounded for the reason every table here is bounded, and the cost
/// of the bound is nine blocks of arena whether or not anybody is in them.
///
/// @note: the two numbers agree by arithmetic and nothing more. The runner's
/// is a private constant in its own crate and there is no import, no
/// assertion and no test tying the two together, so either can be changed on
/// its own. The commit that gives a connecting client a slot is where that
/// has to stop being true.
pub const MAX_PEERS: usize = 9;

/// A handle to one endpoint of a conversation.
///
/// Generational, like a body's handle and unlike an asset's, and for a
/// stronger reason than either: a peer disconnects and its slot is handed to
/// whoever turns up next, while a body it owned may still be lying where it
/// was dropped. A bare slot number there is a prop that changes hands the
/// moment a stranger connects.
///
/// The field agrees from both sides. The two systems read that name an owner
/// on the object identify a peer with something nothing ever reuses - one
/// mints a fresh unique value per connection and says in a comment that it is
/// avoiding allocated sequential ids on purpose, the other counts upwards from
/// past a reserved range and recycles nothing until it wraps. The one whose
/// peer slots *are* reused is a bare array index with no generation anywhere
/// on it, and that system has no per-client owner on an entity at all - so it
/// has nothing that can go stale. What it does carry is an owner that is an
/// *entity* number, kept on the server side of its own structure and never put
/// on the wire.
///
/// Which makes a numeric peer on a body a step past what any of them do, and
/// what makes it workable here is the property the whole wire format will
/// stand on: the *world's* tables - entities, bodies, joints - are slotted and
/// generational, and a restore rebuilds them slot for slot and generation for
/// generation, so a number written down on one machine names the same thing
/// when it is read on another.
///
/// `Pod` for the reason every other handle here is: a game keeps its handles
/// in the arena, and a zeroed arena has to read back as [`NONE`](Self::NONE)
/// rather than as something that could resolve.
///
/// **A generation of zero means nobody, and nothing in this file makes that
/// true.** Every handle here is guaranteed it by the table that hands it out -
/// a body's generation is bumped before the handle leaves and lifted to one
/// where a slot is reused, so zero can never escape. For a peer that table is
/// [`Players`](crate::abi::state::Players), which mints nothing at zero and
/// refuses to seat anybody at a slot whose generation is zero. `Pod` means none
/// of this is an enforcement in any case: any crate can cast eight bytes into
/// one of these, so a table's checks are a way of not being wrong by accident
/// rather than a way of stopping somebody being wrong on purpose.
#[repr(C)]
#[derive(
	Clone,
	Copy,
	Debug,
	Default,
	PartialEq,
	Eq,
	Hash,
	crate::bytemuck::Pod,
	crate::bytemuck::Zeroable,
)]
pub struct PeerId {
	index: u32,
	generation: u32,
}

impl PeerId {
	/// The peer that simulates the world.
	///
	/// Slot zero, at a generation nothing hands out. Both halves are
	/// deliberate and they defend different things.
	///
	/// The **slot** is reserved so that a block kept per peer has somewhere to
	/// put the host's. Nothing in this file enforces it; the table that hands
	/// slots out does, by seating the host there before anybody can ask and by
	/// starting its search one slot later. @ref
	/// [`Players`](crate::abi::state::Players).
	///
	/// The **generation** is `u32::MAX` because the tables in this engine all
	/// mint a handle the same way - take the lowest free slot, add one to its
	/// generation - and the first occupant of the first slot in such a table
	/// is therefore `{0, 1}`. Had that been this value, a peer table written
	/// in the house style would hand the *first client to connect* an identity
	/// that reads back as the host, and with it authority over every body in
	/// the world. So the two defenses are kept separate on purpose: getting
	/// the reservation wrong is then a client sharing the host's block, which
	/// is a bug, rather than a client being the host, which is the end of the
	/// authority model.
	///
	/// A world nobody has networked holds this in
	/// [`World::peer`](crate::abi::World::peer), because a process playing on
	/// its own is its own authority. That is a deliberate value rather than
	/// the zero one, and the difference is the whole of the safety here: see
	/// [`NONE`](Self::NONE).
	pub const HOST: Self = Self { index: 0, generation: u32::MAX };
	/// Nobody.
	///
	/// What a body nobody owns says - the map, a prop lying in a corner, every
	/// body in a world that has never heard of a network. Also what a client
	/// that has connected but has not yet been told who it is is meant to say
	/// about itself, and the reason this is the *zero* value rather than
	/// [`HOST`](Self::HOST): a peer read out of a freshly zeroed arena works
	/// nothing out and owns nothing. The powerless answer is the one a zero
	/// should give.
	pub const NONE: Self = Self { index: 0, generation: 0 };

	/// Which occupant of that slot this names.
	#[must_use]
	pub const fn generation(self) -> u32 { self.generation }

	/// Whether this is the peer that simulates the world.
	#[must_use]
	pub const fn is_host(self) -> bool {
		self.index == Self::HOST.index && self.generation == Self::HOST.generation
	}

	/// Whether this names anybody at all.
	///
	/// A `true` here does not mean that peer is still connected - only the
	/// table of them answers that.
	#[must_use]
	pub const fn is_some(self) -> bool { self.generation != 0 }

	/// The slot this addresses, whatever peer sits there now.
	///
	/// Unvalidated, and it has to be: every bit pattern is a legal `PeerId`,
	/// so this can be any number a `u32` holds. It is also meaningless unless
	/// [`is_some`](Self::is_some), because a peer naming nobody reports slot
	/// zero, which is the host's. Whoever indexes a table with this bounds
	/// checks it and asks that question first - the same arrangement a body's
	/// slot has, where every consumer goes through a lookup that rejects a
	/// generation of zero.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		reason = "lossless here, and try_from is not const"
	)]
	pub const fn slot(self) -> usize { self.index as usize }

	/// The peer at a slot, as the table that hands them out sees it.
	///
	/// Crate-visible and no wider, which is the arrangement every other handle
	/// here has: the only thing that may mint one is the table that owns the
	/// slot, and for a peer that is [`Players`](crate::abi::state::Players). A
	/// game reads a peer out of the world; it never builds one.
	///
	/// @param index - which slot
	/// @param generation - which occupant of it
	pub(crate) const fn at(index: u32, generation: u32) -> Self { Self { index, generation } }
}

/// What part this process plays in one thing.
///
/// Not stored anywhere: it is [`of`](Self::of) two peers, worked out wherever
/// somebody needs it. @ref the module comment for why, and note there is
/// deliberately no `Default` - a role is an answer to a question, and there is
/// no question to answer until both peers are in hand.
///
/// **Declared in order of how much this process decides**, which is the order
/// the one engine that has this enum declares its own in and compares them by.
/// Nothing here compares two roles, so nothing derives an ordering. What the
/// order does buy is that the discriminant a zero would read as is
/// `SimulatedProxy`, the powerless one - which matters not at all today,
/// because this is never stored, never sent and not `Pod`, and would matter a
/// great deal the day any of those three stopped being true.
///
/// `#[repr(C)]` for consistency with every other enum at this boundary rather
/// than because anything needs it: this crosses no `extern "C"` signature and
/// sits in no `#[repr(C)]` struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
	/// This process is told where it went.
	///
	/// Everything a client can see that is not its own. Nothing local moves
	/// it; it is driven towards the place the last snapshot put it.
	SimulatedProxy,

	/// This process works out where it goes and expects to be corrected.
	///
	/// What a client holds over the things it owns. The prediction is the
	/// client's own move re-run over what the host has not acknowledged yet,
	/// and the correction is a decay rather than a snap.
	AutonomousProxy,

	/// This process decides where it goes, and everybody else is told.
	///
	/// What the host holds over everything in its world, a client's own player
	/// included: one machine simulates, and that is the whole model.
	Authority,
}

impl Role {
	/// Whether a thing in this role is worked out here rather than reported to
	/// here.
	///
	/// True for the two that decide and false for the one that is told. That
	/// is the branch anything driving a body towards a snapshot actually
	/// wants, because a whole class of code cares which side of the line a
	/// role falls on rather than which of the two it is.
	///
	/// @note: written as the two that decide rather than as "not the one that
	/// is told", which is the same answer today and a different one the day a
	/// fourth role is added. A role nobody has thought about yet should fail
	/// to this side of the line, and `matches!` is checked for exhaustiveness
	/// in neither direction.
	#[must_use]
	pub const fn local(self) -> bool { matches!(self, Self::AutonomousProxy | Self::Authority) }

	/// The part one peer plays in a thing another peer owns.
	///
	/// @param peer - who this process is, @ref
	/// [`World::peer`](crate::abi::World::peer)
	/// @param owner - who owns the thing, or [`PeerId::NONE`] for nobody
	/// @return what this process may do about it
	#[must_use]
	pub fn of(peer: PeerId, owner: PeerId) -> Self {
		if peer.is_host() {
			return Self::Authority;
		}

		// @note: both halves are load-bearing, and the first is deliberately
		// `is_some` rather than a comparison against `PeerId::NONE`. The
		// obvious case it stops is a client that has connected and has not
		// been told who it is: it holds nobody, and every unowned body in the
		// world - which is most of them - would otherwise answer that it owns
		// it. The case only `is_some` stops is a peer of any slot at all with
		// a generation of zero, which is a handle naming nobody at a slot
		// somebody could be sitting in, and every bit pattern is one of those.
		if peer.is_some() && peer == owner {
			return Self::AutonomousProxy;
		}

		Self::SimulatedProxy
	}
}

/// How many of one peer's commands are kept at once.
///
/// The window everything about a command is measured against: how far back a
/// re-send may reach, how many a host may find waiting after a stall, and how
/// many moves a client may have outstanding before the oldest is forgotten.
///
/// Thirty-two at sixty a second is a little over half a second. What sets the
/// floor is the round trip: a command is made every step and settled a round
/// trip later, so the number outstanding is the round trip measured in steps -
/// eighteen at three hundred milliseconds, which is past what anybody plays
/// over. The rest is margin for a stall at either end.
///
/// The cost is [`MAX_PEERS`] rings of this many commands whether or not
/// anybody is in them, which is nine times thirty-two times twenty-four bytes,
/// or under seven kilobytes for the whole table. That figure is the *whole*
/// cost and not a starting point, which took two deliberate choices rather
/// than none: a ring reserves its room when the table is built, and a full one
/// makes space before it takes rather than after, so nothing here ever grows
/// past this and no peer's traffic decides when an allocation happens.
///
/// The field spends between twenty and ninety-six on the same question, and
/// the spread is about what the history is *for*: the one that keeps twenty
/// treats it as a floor and grows it to whatever its rollback window needs;
/// the one that keeps ninety-six re-simulates from it and warns rather than
/// dropping when it fills. Nothing here re-simulates, so this is the low end
/// on purpose, and the one system that keeps nothing at all is the one whose
/// receiver never runs a command either.
///
/// @note: this is what is *kept*, not what is sent. How many of them go in one
/// datagram is the wire's business and is bounded by the datagram rather than
/// by this.
pub const BACKUP: usize = 32;

/// How far a number may stand from what is held and still be the same
/// conversation.
///
/// A ring's depth, said as a command number, and it is the same number for a
/// reason rather than by coincidence: a receiver that has missed more than a
/// ring's worth has lost them whatever it does next, and a sender's redundancy
/// never reaches back further than its own ring either, so nothing honest is
/// ever this far out.
///
/// **What it is for is that one number cannot mute a peer forever.** The rule
/// above it - a number not above the newest held is refused - is what makes
/// the redundancy free, and on its own it is also a permanent silent wedge:
/// nothing here is authenticated and a datagram's checksum is sixteen bits, so
/// one flipped bit turning command five into two billion becomes the newest
/// held, and every real command that peer sends afterwards is below it and
/// refused, for the rest of the session, with no counter moving and nothing
/// said. A number this far out is therefore read as a discontinuity rather
/// than as a lie: what is held is dropped and the count starts again from it.
/// Two of the systems in the field bound the same field the same way - one
/// clamps a command's time into a window around its own clock, the other
/// refuses a step larger than half its reset period - and both do it because
/// the field they order by is one a client chooses.
pub const WINDOW: u32 = 32;

// the two are one number, and a check rather than a cast so that changing
// either alone does not compile.
#[expect(
	clippy::as_conversions,
	reason = "a u32 to usize is lossless on every target this builds for, and try_from is not \
	          available in a const item"
)]
const _: () = assert!(BACKUP == WINDOW as usize, "a ring's depth and its window are one number");

/// What one client asked for during one step.
///
/// The only thing a client says about itself, and deliberately small: what was
/// held down, which way the person was looking, which command this is and
/// which step they thought the world was on when they made it. Everything a
/// move *becomes* - a direction, a speed, a jump, a shot - is worked out from
/// this by whoever runs it, and both machines work it out with the same code.
/// That is what makes a client's guess and the host's answer comparable at
/// all. A client that sent the result of its own arithmetic instead would not
/// be asking for anything; it would be telling the host where it had decided
/// to go, which is the model this engine is not.
///
/// **The buttons are a bitmask this module gives no meaning to**, the way
/// [`Layers`](super::Layers) and the state arena carry none either. Nothing
/// here knows what a jump is, and the day something did, every game would have
/// to agree with it.
///
/// Four systems were read that have a per-tick command at all, and they land
/// in three places. Two carry movement *apart* from the buttons - one as three
/// signed bytes, the other as an already-resolved acceleration vector - and
/// both do it because they take an analog stick. One folds the whole of it
/// into a single sixty-four-bit action mask and rebuilds the direction locally
/// from the bits, which is what is done here, and it pays a price worth
/// knowing: a stick's *magnitude* does not survive the crossing, so a remote
/// viewer sees every walk at full speed. The fourth has no bitmask anywhere
/// and sends a typed value per named action instead. A game whose walk is four
/// keys spends four bits and no bytes; the day one has a stick, the axes are
/// what this struct grows, and the note above is the reason.
///
/// **Two angles and not three, and that is a departure.** Of the four, two put
/// the look on the command itself and one replicates it as an ordinary
/// property beside the command - and all three of those carry *three* angles.
/// The fourth has no notion of a look at all. Two here because a person's look
/// is a yaw and a pitch: a roll is a camera effect, and a camera lives in the
/// one arena a snapshot never touches, so a client that rolls its own view has
/// already got the answer where it needs it. The day something leans, this is
/// where the third goes.
///
/// **No owner, and on this the field is unanimous.** Which peer asked for this
/// is not in here, because it is in none of the four either: every one of them
/// takes the sender from the conversation the message arrived over, and one of
/// them goes further and *strips* a target the sending peer does not own. A
/// host knows a command's peer from the connection; a client has only its own.
/// A field saying it as well is a field that can disagree with the table it is
/// filed in, and it is a field a stranger could lie in - the same argument
/// that keeps a [`Role`] derived.
///
/// `Pod`, so a command is read straight out of a datagram's bytes and written
/// straight back into one. Twenty-four bytes with no padding, which the build
/// checks.
#[repr(C)]
#[derive(
	Clone, Copy, Debug, Default, PartialEq, crate::bytemuck::Pod, crate::bytemuck::Zeroable,
)]
pub struct Command {
	/// Which step the client believed the world was on when it made this.
	///
	/// The client's own [`World::steps`](crate::abi::World::steps), which is
	/// not the host's and is not meant to be: two processes started at
	/// different moments, and neither number says anything on the other
	/// machine as an absolute. What is read is the *difference* between two
	/// commands, which is how long the later one covers - one step usually,
	/// and more when the client made none for a while.
	///
	/// **Not the same number as [`number`](Self::number), and worth knowing
	/// where they part.** One command is made per step, so the two move
	/// together while everything is ordinary. They separate whenever a step
	/// runs and no command is made - a world being edited, a window with no
	/// focus - and they start from different places, because a command is
	/// numbered from one per conversation while a step is counted from when
	/// the process started.
	///
	/// **Nobody in the field carries both, and it is worth saying why this
	/// does.** Of the four systems read, two have only a *time* on the
	/// command, a millisecond count in one and an accumulating float of
	/// seconds in the other, and both let it double as the ordering key; one
	/// has only a *number*, and needs no time because its receiver never
	/// simulates a command and only reads the buttons off it; and one has only
	/// a *tick*, which in a fixed-step engine is both at once and is the
	/// nearest thing in the field to what colby has.
	///
	/// **And it is the sender's to choose, like every ordering field in the
	/// field.** A client can claim any step it likes and so can claim any
	/// length of move; both systems that order by a time clamp it on arrival
	/// for exactly that reason, one against its own clock and one against half
	/// its reset period. Nothing here clamps anything, because nothing here
	/// runs a command yet - the commit that does is where the clamp belongs,
	/// and it is named here so that it is a thing to write rather than a thing
	/// to discover.
	///
	/// Carrying the pair buys exactly one thing, and it is a thing the others
	/// cannot ask: with both, a gap tells the receiver *which kind* of gap it
	/// is. Numbers that run on with steps that jump is a client that made no
	/// command for a while, and it is nothing to worry about; numbers that
	/// jump is a wire that lost some, and it is. A step alone cannot tell
	/// those apart, and one of the two systems above pays for it by dropping
	/// every command that shares a millisecond with the one before it.
	pub step: u64,

	/// Which command of this peer's this is, counting from one.
	///
	/// The identity of the thing, and the only field anything sorts, refuses
	/// or acknowledges by. **Zero is not a command**, which is the reading a
	/// zeroed twenty-four bytes has to have (@ref
	/// [`is_some`](Self::is_some)), and it is also what a receiving end
	/// compares against before it has heard anything at all.
	pub number: u32,

	/// What was held down, in whatever bits the game gave meaning to.
	pub buttons: u32,

	/// Which way the person was looking about the vertical, in radians.
	pub yaw: f32,

	/// And how far up or down, in radians.
	pub pitch: f32,
}

// twenty-four bytes with no padding. `Pod` already refuses padding, so what
// this adds is the literal number, which is what a wire-format change would
// have to move on purpose rather than by rearranging a struct.
const _: () = assert!(size_of::<Command>() == 24, "a command is twenty-four bytes and no more");
// and eight, from the step. Worth pinning beside the size because it is the
// half that decides how a command may be read: a datagram puts one wherever
// the bytes before it end, so nothing may cast a slice to this - it is read
// and written a copy at a time, whatever the offset.
const _: () =
	assert!(align_of::<Command>() == 8, "a command is aligned like the step it starts with");

impl Command {
	/// Whether this names a command at all.
	///
	/// False for the zero value, and for the same reason [`PeerId::NONE`] is
	/// the zero one: what twenty-four zeroed bytes should say is "nothing was
	/// asked for", not "a command from step nought looking straight ahead".
	#[must_use]
	pub const fn is_some(self) -> bool { self.number != 0 }
}

/// What every peer recently asked for, one ring each.
///
/// Beside [`World::input`](crate::abi::World::input) and deliberately not part
/// of it. Input is this machine's keyboard and mouse - the surface size, the
/// cursor in pixels, what was typed - and none of that crosses to anybody or
/// ever should. A command is the small part of it that another machine has to
/// be told, said in a form that means the same thing on both.
///
/// **A ring per peer, keyed by [`PeerId::slot`], and the ring remembers whose
/// it is.** That is the same problem [`Players`](super::state::Players) has
/// and very nearly the same answer, with one difference worth stating: that
/// table hands out identities and so has to be told when somebody leaves, and
/// this one only holds what somebody asked for, so it works the identity out.
/// A slot's generation is only ever raised by the table that mints peers, so
/// **a later occupant takes the ring over and clears it, and an earlier one is
/// refused.**
///
/// **And nothing here is told when a peer leaves**, which was a deliberate
/// removal rather than an omission. There was a `forget` taking one peer, and
/// what it bought was that a departed peer's last few commands did not sit in
/// a ring until a replacement turned up - nothing at all, since the ring is a
/// fixed slot that frees no memory and nobody walks a peer the table of
/// identities has let go. What it *cost* was the sentence above: it handed the
/// ring back to nobody, which lowered the generation `push` compares against,
/// and from there the peer that had just left could take its own ring again
/// with the mark reset - so a command the host had already run came back round
/// on the next redundant datagram and ran twice. One method that bought
/// nothing was the only thing in the file that broke the rule everything else
/// stands on. What is left that gives a ring up is [`clear`](Self::clear),
/// which is a whole world being replaced and is the one moment every identity
/// here is being rewound anyway.
///
/// **The host holds one too.** The person playing on the machine that
/// simulates is a player, and their move has to reach the game the way
/// everybody else's does or the game has two paths through the same question -
/// which is the shape every one of these that ships avoids.
///
/// **A command arriving twice is taken once**, and that is not a nicety, it is
/// what the whole redundancy scheme stands on: the last few unsettled commands
/// go out in *every* datagram, so a wire dropping one in ten still delivers
/// each command, and every copy after the first is an old number. The same
/// rule throws away a datagram that overtook the one in front of it, whose
/// commands the newer one was carrying anyway. One comparison answers both,
/// and it is the same comparison all four systems read make - each against its
/// own record of the last one it took, each with the older or equal one
/// dropped. How many go out at once is where they part: one sends up to
/// thirty-two, one sends five, one sends the new move plus at most one old
/// one, and one sends a single command over an unreliable channel and simply
/// loses it.
///
/// **Not written down by a save**, and for [`Body::owner`](super::Body)'s
/// reason: what somebody was asking for a moment ago is a moment rather than a
/// world. A restore leaves the rings where they are, which costs nothing
/// because whatever was in them is already settled or already stale.
#[derive(Debug)]
pub struct Commands {
	rings: Vec<Ring>,
}

/// One peer's recent commands, oldest first and strictly increasing.
///
/// Deliberately not `Clone`, and neither is the table above it. Cloning a
/// `Vec` allocates for its *length* rather than its capacity, so a copied
/// table would hand back rings that had reserved nothing - which is how the
/// room [`Ring::empty`] reserves came to be reserved for one ring out of nine
/// the first time this was written, `vec![Ring::empty(); MAX_PEERS]` having
/// cloned the one it was given. Nothing copies a table of commands, and the
/// day something does, this is the line that will make it say so.
#[derive(Debug)]
struct Ring {
	/// Whose these are, or [`PeerId::NONE`] while nobody has used the slot.
	owner: PeerId,

	/// At most [`BACKUP`] of them, sorted by [`Command::number`].
	kept: Vec<Command>,

	/// The highest number this end is done with. @ref [`Commands::settle`].
	settled: u32,
}

impl Ring {
	/// A slot nobody has used, with its room already reserved.
	///
	/// `with_capacity` rather than `new`, which is what makes [`BACKUP`]'s
	/// arithmetic the whole cost rather than a floor: a growing run would
	/// allocate on a datagram's arrival, five times over on the way up, and
	/// settle at twice the number that paragraph quotes. Reserving here and
	/// making room before taking in [`Commands::push`] are two halves of one
	/// decision and neither is worth much alone.
	fn empty() -> Self {
		Self {
			owner: PeerId::NONE,
			kept: Vec::with_capacity(BACKUP),
			settled: 0,
		}
	}

	/// Gives the ring to somebody, throwing away what was in it.
	///
	/// Called from two places and the emptying is observable from exactly one
	/// of them. In [`Commands::push`] it is the whole point: a slot changing
	/// hands must not read the last occupant's commands as this one's. In
	/// [`Commands::clear`] it cannot be seen at all, because that hands every
	/// ring to nobody and **a ring nobody owns is a ring nothing can reach** -
	/// [`Commands::ring`] refuses it, and the next peer to claim it comes back
	/// through here and empties it anyway. A mutation that stops `clear`
	/// emptying breaks no test and none can be written for it.
	///
	/// It is one function all the same, and the reason is that "a ring nobody
	/// owns is empty" is the invariant every unobservable line above is
	/// unobservable *because of*. Split it into an emptying reset and a
	/// disowning one and the invariant becomes something two call sites have
	/// to remember rather than something this function is.
	fn reset(&mut self, owner: PeerId) {
		self.owner = owner;
		self.kept.clear();
		self.settled = 0;
	}

	/// The commands above a number, oldest first.
	///
	/// A subslice rather than a filter, which is what the numbers being
	/// increasing buys: everything above a number is a run at the end.
	fn after(&self, number: u32) -> &[Command] {
		&self.kept[self
			.kept
			.partition_point(|command| command.number <= number)..]
	}

	/// The highest number held, or nought for an empty ring.
	fn newest(&self) -> u32 {
		self.kept
			.last()
			.map_or(0, |command| command.number)
	}
}

impl Commands {
	/// A table with a ring for every slot and nothing in any of them.
	#[must_use]
	pub fn new() -> Self {
		Self {
			// one at a time rather than `vec![..; MAX_PEERS]`, which clones,
			// and a cloned `Vec` reserves its length rather than its capacity.
			// @ref [`Ring`].
			rings: core::iter::repeat_with(Ring::empty)
				.take(MAX_PEERS)
				.collect(),
		}
	}

	/// Takes one command from a peer, if it is news.
	///
	/// Refused for a peer naming nobody, a command numbered nothing, anybody
	/// but the host at the host's slot, a slot past the end of the table, a
	/// peer an occupant of its slot has already replaced, and a number that is
	/// not above the newest held. Taken, but after throwing the ring away
	/// first, for a number further than [`WINDOW`] from the newest held.
	///
	/// @param peer - who is asking
	/// @param command - what they asked for
	/// @return whether it was taken
	pub fn push(&mut self, peer: PeerId, command: Command) -> bool {
		if !peer.is_some() || !command.is_some() {
			return false;
		}

		// slot zero is the host's, and this table says so itself rather than
		// leaning on the one that mints peers never handing it out. Two
		// independent defenses is what [`PeerId::HOST`]'s own doc asks for,
		// and until this line there was only one of them here: the host's
		// generation kept a stranger out of its ring, which is true right up
		// until the ring is given up by a [`clear`](Self::clear) and is then
		// nobody's for as long as it takes the host to ask for something.
		if peer.slot() == PeerId::HOST.slot() && !peer.is_host() {
			return false;
		}

		let Some(ring) = self.rings.get_mut(peer.slot()) else {
			return false;
		};

		// the two halves of "whose ring is this", and they are one comparison
		// because a slot's generation only ever goes up. Equal is the ordinary
		// case; above is a peer that has replaced whoever was here, and what
		// the one before asked for is not this one's to run; below is a peer
		// that has itself been replaced, and it gets nothing.
		if peer.generation() < ring.owner.generation() {
			return false;
		}

		if peer.generation() > ring.owner.generation() {
			ring.reset(peer);
		}

		// a number this far from what is held is not this conversation going
		// on, in either direction: too far above and everything between is
		// lost whatever happens next, too far below and the far end is not
		// counting from where this end thinks it is. What is held goes, and
		// the count starts again from here. @ref [`WINDOW`] for the wedge this
		// exists to stop.
		//
		// **And the mark is put just under it rather than back to nothing**,
		// which is the whole of what keeps the promise through a
		// discontinuity: everything below a number this end will never see
		// again is done with by definition, so nothing before it can be handed
		// out to be run a second time - and nothing after it is swallowed
		// either.
		if command.number.abs_diff(ring.newest()) > WINDOW {
			ring.kept.clear();
			ring.settled = command.number.saturating_sub(1);
		} else if command.number <= ring.newest() {
			// and the line the redundancy stands on. @ref the type comment.
			return false;
		}

		// room is made before the command is taken rather than after, so the
		// run never grows past its reserved length and no peer's traffic
		// decides when this allocates. The oldest goes rather than the newest
		// being refused: a peer that has fallen far enough behind for this to
		// fire has lost the old ones either way, and refusing the new one
		// would lose the present as well as the past.
		if ring.kept.len() >= BACKUP {
			ring.kept.remove(0);
		}

		ring.kept.push(command);

		true
	}

	/// Everything held for one peer, oldest first.
	///
	/// Empty for anybody the ring at their slot does not belong to, which
	/// includes [`PeerId::NONE`] - whose slot is the host's, and who must not
	/// be answered with the host's commands.
	#[must_use]
	pub fn kept(&self, peer: PeerId) -> &[Command] {
		match self.ring(peer) {
			| Some(ring) => &ring.kept,
			| None => &[],
		}
	}

	/// What this end has not settled yet, oldest first.
	///
	/// The answer to all three questions asked of this table, which is why
	/// there is one method rather than three: what a host has still to run,
	/// what a client has still to be corrected about, and what goes in the
	/// next datagram are the same list read from different places.
	#[must_use]
	pub fn unsettled(&self, peer: PeerId) -> &[Command] {
		match self.ring(peer) {
			| Some(ring) => ring.after(ring.settled),
			| None => &[],
		}
	}

	/// Says that everything up to a number is done with.
	///
	/// One field with one meaning read from two sides. On the machine that
	/// simulates, settling a number is having *run* it, and what is left is
	/// what has still to happen. On a client it is the host having said it
	/// arrived, and what is left is what the client is still guessing about.
	/// Both are "below here these no longer matter", and giving the two ends
	/// separate words for it would be two things to keep in step.
	///
	/// **Every system read keeps this mark, and every one of them keeps it
	/// beside the ring rather than in gameplay's own memory.** Two of the four
	/// mark what was *run* and two mark what merely *arrived*, and the split
	/// is not arbitrary: the two that mark what was run are the two whose
	/// receiver simulates the command, which is what this one does. Keeping it
	/// here rather than in the game's arena is what makes "a command runs
	/// once" a promise the engine makes rather than one every game has to
	/// remember to keep - and it is also what makes the promise survive a
	/// module being swapped out mid-session.
	///
	/// **Never past what has arrived.** A far end that says it has settled a
	/// command this side has never sent would otherwise settle every command
	/// there will ever be, and nothing here is authenticated. Capped instead,
	/// so a lie costs the commands already held and no more.
	///
	/// **Never backwards**, so two acknowledgements that overtook each other
	/// do not un-settle anything.
	///
	/// @param peer - whose ring
	/// @param number - the highest command that no longer matters
	/// @return whether the mark moved
	pub fn settle(&mut self, peer: PeerId, number: u32) -> bool {
		let Some(ring) = self.ring_mut(peer) else {
			return false;
		};

		let reached = number.min(ring.newest());

		if reached <= ring.settled {
			return false;
		}

		ring.settled = reached;

		true
	}

	/// The highest number this end has settled for a peer.
	#[must_use]
	pub fn settled(&self, peer: PeerId) -> u32 { self.ring(peer).map_or(0, |ring| ring.settled) }

	/// The highest number held for a peer, or nought if none is.
	#[must_use]
	pub fn newest(&self, peer: PeerId) -> u32 { self.ring(peer).map_or(0, Ring::newest) }

	/// The lowest number held for a peer, or nought if none is.
	///
	/// The only way to tell "nothing new has come" from "something was lost":
	/// a caller that has settled `n` and finds this above `n + 1` is looking
	/// at a ring that forgot commands before anybody ran them.
	#[must_use]
	pub fn oldest(&self, peer: PeerId) -> u32 {
		self.ring(peer).map_or(0, |ring| {
			ring.kept
				.first()
				.map_or(0, |command| command.number)
		})
	}

	/// Gives up every ring, whoever holds it.
	///
	/// For putting a world back, and that is not tidiness. Everything else
	/// here leans on a slot's generation only ever going *up*, which is true
	/// of [`Players::admit`](super::state::Players::admit) and is not true of
	/// [`Players::restore`](super::state::Players::restore) - a restore puts
	/// generations back as they were and therefore rewinds them, and its own
	/// doc says a handle minted after a description was captured can be minted
	/// a second time afterwards. Minted a second time, that handle would find
	/// its ring still held by a peer of the same name, with the last one's
	/// commands in it, and every one of them would read as news.
	///
	/// The merits agree with the mechanism, which is the part worth saying:
	/// what somebody was asking for is a request against the world that was
	/// there when they asked. Replace the world and the request is not late,
	/// it is about somewhere else.
	///
	/// @note: that this *empties* rather than only disowning cannot be
	/// observed, and @ref [`Ring::reset`] for why the line stays. What is
	/// observable, and tested, is that it reaches the host's ring too and that
	/// it happens on every restore.
	pub fn clear(&mut self) {
		for ring in &mut self.rings {
			ring.reset(PeerId::NONE);
		}
	}

	/// Who holds the ring at a slot, whatever they have asked for.
	///
	/// @param slot - which slot
	#[must_use]
	pub fn owner(&self, slot: usize) -> PeerId {
		self.rings
			.get(slot)
			.map_or(PeerId::NONE, |ring| ring.owner)
	}

	/// The ring at a peer's slot, if it is theirs.
	///
	/// @note: **this `is_some` cannot be observed and is kept anyway**, which
	/// is worth being exact about because the identical line in
	/// [`ring_mut`](Self::ring_mut) *can* be. Without it, a peer naming nobody
	/// matches an unused ring, whose owner is nobody too - but a ring nobody
	/// owns is always empty and always unsettled, @ref [`Ring::reset`], so
	/// every reader below answers nothing either way. A mutation removing this
	/// line breaks no test and none can be written for it. It stays because it
	/// is the line that says what the rule is, and because what makes it
	/// unobservable is somebody else's invariant rather than this function's.
	/// Two somebodies, in fact, and both are tested: [`Ring::reset`] is what
	/// empties a ring on the way to nobody, and the `is_some` in
	/// [`push`](Self::push) is what stops a peer at a *non-zero* slot naming
	/// nobody from filing commands into a ring nobody owns - which would make
	/// an unowned ring non-empty and this line load-bearing at once. The day
	/// either goes, this is the guard that was already right.
	fn ring(&self, peer: PeerId) -> Option<&Ring> {
		if !peer.is_some() {
			return None;
		}

		self.rings
			.get(peer.slot())
			.filter(|ring| ring.owner == peer)
	}

	/// The same, to write into.
	///
	/// @note: this `is_some` cannot be observed either, for
	/// [`ring`](Self::ring)'s reason and one more: the only thing reached
	/// through here that writes is [`settle`](Self::settle), and a mark on a
	/// ring nobody owns cannot move, because the mark is capped at the newest
	/// held and a ring nobody owns holds nothing. It stays for the same reason
	/// the other does.
	fn ring_mut(&mut self, peer: PeerId) -> Option<&mut Ring> {
		if !peer.is_some() {
			return None;
		}

		self.rings
			.get_mut(peer.slot())
			.filter(|ring| ring.owner == peer)
	}
}

impl Default for Commands {
	fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Some client, as the table handing slots out would mint one.
	const fn client(index: u32, generation: u32) -> PeerId { PeerId::at(index, generation) }

	/// A command whose four numbers are four different numbers.
	///
	/// The step is deliberately not the command number and not a multiple of
	/// it, the buttons are not empty and the two angles differ - so a test
	/// that reads the wrong word out of a record cannot be answered by
	/// arithmetic that happens to agree.
	///
	/// @note: the step here *is* a fixed function of the number, which is
	/// right for every test that is about ordering and wrong for the one test
	/// that is about the two of them saying different things. @ref
	/// [`asked_at`].
	fn asked(number: u32) -> Command { asked_at(number, u64::from(number) * 7 + 1000) }

	/// The same, with the step said rather than derived.
	///
	/// For the one question the pair of fields exists to answer, which no
	/// fixture whose step follows its number can even express.
	fn asked_at(number: u32, step: u64) -> Command {
		let turn = [0.3_f32, -1.4, 2.05, 0.75, -2.9];
		let yaw = turn[usize::try_from(number % 5).unwrap_or(0)];

		Command {
			step,
			number,
			buttons: 0b1001 << (number % 5),
			yaw,
			pitch: yaw * -0.375,
		}
	}

	#[test]
	fn nobody_is_the_zero_value_and_the_host_is_not() {
		// the bytes a game's own memory actually keeps a handle as, rather
		// than a constructor's opinion of them.
		let arena = [0_u8; 8];
		let read: PeerId = bytemuck::pod_read_unaligned(&arena);

		assert_eq!(size_of::<PeerId>(), size_of::<u64>(), "two words, like a BodyId");
		assert_eq!(PeerId::default(), PeerId::NONE);
		assert_eq!(read, PeerId::NONE, "a freshly zeroed arena reads as nobody");

		assert!(!PeerId::NONE.is_some());
		assert!(!PeerId::NONE.is_host(), "the powerless answer is the one a zero gives");
		assert!(PeerId::HOST.is_some());
		assert!(PeerId::HOST.is_host());
		assert_ne!(PeerId::HOST, PeerId::NONE);
	}

	#[test]
	fn a_slot_with_no_occupant_in_it_is_nobody_whatever_slot_it_is() {
		let empty = client(3, 0);

		assert!(!empty.is_some(), "a generation of zero is nobody at any slot at all");
		assert!(!empty.is_host());
		assert_eq!(
			Role::of(empty, empty),
			Role::SimulatedProxy,
			"so a handle naming nobody does not own itself"
		);
	}

	/// The trap this constant exists to avoid, written as a test.
	///
	/// Every table in this engine mints the first occupant of a slot at
	/// generation one, so `{0, 1}` is what a peer table in the house style
	/// hands the first client that connects. If that were the host, that
	/// client would be handed the world.
	#[test]
	fn what_a_table_would_mint_for_a_first_client_is_not_the_host() {
		let first = client(0, 1);

		assert!(!first.is_host(), "the host is at a generation nothing hands out");
		assert_ne!(first, PeerId::HOST);
		assert_eq!(
			Role::of(first, PeerId::NONE),
			Role::SimulatedProxy,
			"so the first peer to connect decides nothing until it is given something"
		);
		assert_eq!(PeerId::HOST.generation(), u32::MAX);
	}

	#[test]
	fn a_slot_alone_is_not_an_identity() {
		assert_eq!(PeerId::HOST.slot(), 0, "the host's is the first");
		assert_eq!(client(3, 1).slot(), 3, "and a client's is the one it was minted at");
		assert_eq!(client(3, 2).generation(), 2, "as is the occupant");

		let stranger = client(0, 2);

		assert_eq!(stranger.slot(), PeerId::HOST.slot(), "the same slot as the host");
		assert!(!stranger.is_host(), "and not the host");
		assert!(
			!client(7, u32::MAX).is_host(),
			"and neither is the host's generation somewhere else, so both halves are read"
		);
		assert_eq!(
			PeerId::NONE.slot(),
			PeerId::HOST.slot(),
			"and nobody reports the host's slot too, which is why a slot is read only once \
			 is_some has been asked"
		);
	}

	#[test]
	fn a_slot_that_changed_hands_is_not_the_peer_that_left_it() {
		let before = client(3, 1);
		let after = client(3, 2);

		assert_eq!(before.slot(), after.slot());
		assert_ne!(before, after, "which is the whole reason a generation is carried");
		assert_eq!(
			Role::of(after, before),
			Role::SimulatedProxy,
			"so a body the old peer owned does not change hands with the slot"
		);
	}

	#[test]
	fn a_host_is_the_authority_over_everything_it_can_see() {
		for owner in [PeerId::NONE, PeerId::HOST, client(1, 1), client(7, 4)] {
			assert_eq!(
				Role::of(PeerId::HOST, owner),
				Role::Authority,
				"including what a client owns"
			);
		}
	}

	#[test]
	fn a_client_works_out_its_own_and_is_told_the_rest() {
		let mine = client(1, 1);
		let theirs = client(2, 1);

		assert_eq!(Role::of(mine, mine), Role::AutonomousProxy);
		assert_eq!(Role::of(mine, theirs), Role::SimulatedProxy);
		assert_eq!(
			Role::of(mine, PeerId::NONE),
			Role::SimulatedProxy,
			"and a prop nobody owns is the host's business"
		);
		assert_eq!(
			Role::of(mine, PeerId::HOST),
			Role::SimulatedProxy,
			"as is anything the host holds itself"
		);
	}

	#[test]
	fn a_client_that_has_not_been_told_who_it_is_owns_nothing() {
		assert_eq!(
			Role::of(PeerId::NONE, PeerId::NONE),
			Role::SimulatedProxy,
			"nobody owning nothing is not a peer owning its own"
		);
		assert_eq!(Role::of(PeerId::NONE, client(1, 1)), Role::SimulatedProxy);
		assert_eq!(Role::of(PeerId::NONE, PeerId::HOST), Role::SimulatedProxy);
	}

	#[test]
	fn the_two_roles_that_decide_are_the_two_that_are_local() {
		assert!(Role::Authority.local());
		assert!(Role::AutonomousProxy.local());
		assert!(!Role::SimulatedProxy.local(), "and the one that is told is not");
	}

	/// The layout, written down as the bytes a datagram would carry.
	///
	/// Every other test here round-trips through the same struct definition
	/// and would not notice two fields swapping places. This one would: it is
	/// a known answer rather than a comparison of the code with itself.
	#[test]
	fn a_command_is_twenty_four_bytes_in_the_order_it_declares() {
		let command = Command {
			step: 0x0102_0304_0506_0708,
			number: 0x090A_0B0C,
			buttons: 0x0D0E_0F10,
			// 0x3F80_0000 and 0xC000_0000, so a swapped pair is visible.
			yaw: 1.0,
			pitch: -2.0,
		};

		assert_eq!(size_of::<Command>(), 24);
		assert_eq!(bytemuck::bytes_of(&command), &[
			0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // step
			0x0C, 0x0B, 0x0A, 0x09, // number
			0x10, 0x0F, 0x0E, 0x0D, // buttons
			0x00, 0x00, 0x80, 0x3F, // yaw
			0x00, 0x00, 0x00, 0xC0, // pitch
		]);

		let zeroed: Command = bytemuck::pod_read_unaligned(&[0_u8; 24]);

		assert_eq!(zeroed, Command::default());
		assert!(!zeroed.is_some(), "nothing asked for is what zeroed bytes say");
		assert!(asked(1).is_some(), "and the first command there is, is one");
	}

	#[test]
	fn what_was_asked_for_comes_back_word_for_word() {
		let mut commands = Commands::new();
		let peer = client(4, 3);

		for number in [2_u32, 9, 10] {
			assert!(commands.push(peer, asked(number)), "each is above the last");
		}

		assert_eq!(commands.kept(peer), &[asked(2), asked(9), asked(10)]);
		assert_eq!(commands.oldest(peer), 2);
		assert_eq!(commands.newest(peer), 10);
		assert_eq!(commands.owner(peer.slot()), peer);
	}

	/// The line the whole redundancy scheme stands on.
	#[test]
	fn a_command_that_has_already_been_taken_is_not_taken_again() {
		let mut commands = Commands::new();
		let peer = client(2, 6);

		for number in [4_u32, 5, 6] {
			assert!(commands.push(peer, asked(number)));
		}

		// the same three again, which is what the next datagram carries.
		for number in [4_u32, 5, 6] {
			assert!(!commands.push(peer, asked(number)), "{number} was already taken");
		}

		// and one that overtook the datagram in front of it: its own commands
		// came with the newer one, so there is nothing here to lose.
		assert!(!commands.push(peer, asked(3)));
		assert_eq!(commands.kept(peer), &[asked(4), asked(5), asked(6)]);
		assert!(commands.push(peer, asked(7)), "and the next one still lands");
	}

	#[test]
	fn a_full_ring_forgets_its_oldest_rather_than_refusing_the_newest() {
		let mut commands = Commands::new();
		let peer = client(5, 2);

		// five past the ceiling, so the boundary is crossed rather than
		// landed on: a ring exactly full and a ring that has wrapped are
		// different questions.
		for number in 1..=u32::try_from(BACKUP + 5).expect("small") {
			assert!(commands.push(peer, asked(number)), "{number} is the newest so far");
		}

		let kept = commands.kept(peer);

		assert_eq!(kept.len(), BACKUP);
		assert_eq!(commands.oldest(peer), 6, "the first five went");
		assert_eq!(commands.newest(peer), u32::try_from(BACKUP + 5).expect("small"));
		assert_eq!(kept.first().copied(), Some(asked(6)));

		// nothing was settled, and what is held starts above where the last
		// settled number leaves off - which is the only way a caller can tell
		// commands were lost from nothing having arrived.
		assert_eq!(commands.settled(peer), 0);
		assert!(commands.oldest(peer) > commands.settled(peer) + 1, "five were lost");
	}

	#[test]
	fn a_later_occupant_of_a_slot_does_not_inherit_what_the_last_one_asked_for() {
		let mut commands = Commands::new();
		let before = client(3, 2);
		// not the next generation: a slot may be handed on more than once
		// before anybody looks.
		let after = client(3, 4);

		for number in [11_u32, 12, 13] {
			assert!(commands.push(before, asked(number)));
		}

		assert!(commands.settle(before, 12));

		// numbered from one, which is below everything the peer before it
		// sent. Without the ring changing hands this would read as an old
		// command and be dropped.
		assert!(commands.push(after, asked(1)));
		assert_eq!(commands.kept(after), &[asked(1)]);
		assert_eq!(commands.settled(after), 0, "and the mark started over with it");
		assert_eq!(commands.unsettled(after), &[asked(1)]);
		assert!(commands.kept(before).is_empty(), "the peer that left is answered with nothing");
		assert_eq!(commands.newest(before), 0);
	}

	#[test]
	fn a_peer_its_slot_has_already_replaced_is_refused() {
		let mut commands = Commands::new();
		let stale = client(3, 2);
		let here = client(3, 4);

		assert!(commands.push(here, asked(20)));
		assert!(!commands.push(stale, asked(21)), "it is not this peer's ring any more");
		assert_eq!(commands.kept(here), &[asked(20)], "and nothing of theirs was disturbed");
		assert!(!commands.settle(stale, 20), "nor can they settle it");
		assert_eq!(commands.owner(3), here);
	}

	/// Nobody's slot is the host's slot, which is the whole of why this is a
	/// test.
	#[test]
	fn nobody_is_never_answered_with_the_hosts_commands() {
		let mut commands = Commands::new();

		for number in [1_u32, 2] {
			assert!(commands.push(PeerId::HOST, asked(number)));
		}

		assert_eq!(PeerId::NONE.slot(), PeerId::HOST.slot(), "the trap this guards");
		assert!(commands.kept(PeerId::NONE).is_empty());
		assert!(commands.unsettled(PeerId::NONE).is_empty());
		assert_eq!(commands.newest(PeerId::NONE), 0);
		assert_eq!(commands.oldest(PeerId::NONE), 0);
		assert_eq!(commands.settled(PeerId::NONE), 0);
		assert!(!commands.push(PeerId::NONE, asked(3)), "and nobody asks for nothing");
		assert!(!commands.settle(PeerId::NONE, 1));
		assert_eq!(commands.kept(PeerId::HOST).len(), 2, "the host's are untouched");
	}

	/// Nobody against a table nobody has touched, which is a different question
	/// from nobody against the host's ring.
	///
	/// Every ring starts unowned, so on a fresh table an unused slot's owner
	/// and a peer naming nobody are **the same value** - and the generation
	/// comparison that refuses a stale peer cannot tell them apart either,
	/// because both are nothing. So this is the only state in which the
	/// `is_some` checks are the thing doing the work. Asked after the host has
	/// pushed, every one of these is refused by a guard other than the one it
	/// is about.
	#[test]
	fn a_ring_of_nobodys_is_not_a_ring_anybody_can_reach() {
		let mut commands = Commands::new();

		for slot in 0..MAX_PEERS {
			assert_eq!(commands.owner(slot), PeerId::NONE);
		}

		assert!(!commands.push(PeerId::NONE, asked(1)), "nobody may not put one in");
		assert!(!commands.settle(PeerId::NONE, 5), "nor mark one done");
		assert_eq!(
			commands.owner(PeerId::HOST.slot()),
			PeerId::NONE,
			"and the host's slot was not claimed on nobody's behalf"
		);

		assert!(commands.kept(PeerId::NONE).is_empty());
		assert!(commands.kept(PeerId::HOST).is_empty(), "least of all for the host");
		assert!(commands.kept(client(4, 1)).is_empty(), "and nobody has asked for anything");
	}

	/// Nobody at a slot that is not the host's, which is the one place the
	/// `is_some` in `push` is the only thing standing.
	///
	/// Every bit pattern is a legal [`PeerId`], so a generation of nothing can
	/// sit at any slot at all - and at slot zero the host's own rule refuses
	/// it, which is why that is the wrong slot to ask about. Here the slot
	/// rule does not apply, and the generation comparison cannot help either:
	/// an unused ring is owned by nobody, so nobody's generation neither
	/// exceeds it nor falls below it, and the command lands. What it would
	/// land *in* is a ring nobody owns, which nothing can read - so the
	/// command is invisible, and it is also the thing that makes the "a ring
	/// nobody owns is empty" invariant false, which is what two other guards
	/// in this file are unobservable because of.
	#[test]
	fn nobody_at_somebody_elses_slot_files_nothing() {
		let mut commands = Commands::new();
		let nobody = client(3, 0);

		assert!(!nobody.is_some(), "a generation of nothing, at a slot that is not the host's");
		assert_ne!(nobody.slot(), PeerId::HOST.slot());

		assert!(!commands.push(nobody, asked(5)));
		assert_eq!(commands.owner(3), PeerId::NONE, "the ring is still nobody's");
		assert!(commands.kept(nobody).is_empty());

		// and the number that proves nothing landed rather than landing
		// somewhere unreadable: five would be the newest held, and a real
		// peer's first command is one.
		let peer = client(3, 1);

		assert!(commands.push(peer, asked(1)), "so the first real command is not an old one");
		assert_eq!(commands.kept(peer), &[asked(1)]);
	}

	/// The table's whole cost is paid when it is built, which is what makes
	/// the figure in `BACKUP`'s doc a ceiling rather than a starting point.
	#[test]
	fn every_ring_reserves_its_room_and_never_asks_for_more() {
		let mut commands = Commands::new();

		for ring in &commands.rings {
			assert_eq!(ring.kept.capacity(), BACKUP, "reserved before anybody has asked");
			assert!(ring.kept.is_empty());
		}

		// well past the depth, so the run is trimmed many times over. Room is
		// made before a command is taken rather than after, so the length
		// never reaches the size that would double this.
		let peer = client(4, 3);

		for number in 1..=u32::try_from(BACKUP * 2).expect("small") {
			assert!(commands.push(peer, asked(number)));
		}

		let ring = &commands.rings[peer.slot()];

		assert_eq!(ring.kept.len(), BACKUP);
		assert_eq!(ring.kept.capacity(), BACKUP, "and it never grew past what it reserved");
	}

	#[test]
	fn a_command_numbered_nothing_is_not_a_command() {
		let mut commands = Commands::new();
		let peer = client(6, 3);

		assert!(!commands.push(peer, Command { number: 0, ..asked(9) }));
		assert!(commands.kept(peer).is_empty());
		assert_eq!(commands.owner(6), PeerId::NONE, "and the ring was not even claimed");
	}

	#[test]
	fn what_is_left_is_what_stands_above_the_mark() {
		let mut commands = Commands::new();
		let peer = client(7, 5);

		// gapped on purpose, and the gaps here are a wire that lost some
		// rather than a client that made none - which is the reading `asked`
		// can express and the other is not, since its step follows its number.
		// @ref the test below for the pair that tells those apart. What this
		// one is about is that a mark is a number rather than a position.
		for number in [3_u32, 7, 8, 12, 15] {
			assert!(commands.push(peer, asked(number)));
		}

		assert_eq!(commands.unsettled(peer), commands.kept(peer), "nothing settled yet");
		assert!(commands.settle(peer, 8));
		assert_eq!(commands.settled(peer), 8);
		assert_eq!(commands.unsettled(peer), &[asked(12), asked(15)]);

		// a number nothing was ever sent under, between two that were. The
		// mark is where it was put, and what is left is what is above it.
		assert!(commands.settle(peer, 9));
		assert_eq!(commands.settled(peer), 9);
		assert_eq!(commands.unsettled(peer), &[asked(12), asked(15)]);
	}

	/// The one thing carrying a number *and* a step buys, and the only test
	/// that can express it.
	///
	/// A client makes one command per step, so the two move together while
	/// nothing is wrong. When something is wrong they part, and which way they
	/// part says what went wrong: a client that made no command for a while
	/// spends no numbers, so its numbers stay consecutive across a jump in
	/// steps; a wire that lost some loses the numbers *and* the steps
	/// together, because the client did make them.
	#[test]
	fn a_gap_in_the_steps_and_a_gap_in_the_numbers_are_different_news() {
		let mut commands = Commands::new();
		let peer = client(5, 4);

		// three in a row, then the client is left alone for forty steps and
		// makes none, then two more, then the wire loses four.
		for (number, step) in
			[(7_u32, 200_u64), (8, 201), (9, 202), (10, 243), (11, 244), (16, 249)]
		{
			assert!(commands.push(peer, asked_at(number, step)));
		}

		let apart: Vec<(u32, u64)> = commands
			.kept(peer)
			.windows(2)
			.map(|pair| (pair[1].number - pair[0].number, pair[1].step - pair[0].step))
			.collect();

		assert_eq!(apart, vec![(1, 1), (1, 1), (1, 41), (1, 1), (5, 5)]);

		let idle = apart
			.iter()
			.filter(|(numbers, steps)| *numbers == 1 && *steps > 1)
			.count();
		let lost: u64 = apart
			.iter()
			.filter(|(numbers, steps)| u64::from(*numbers) == *steps && *numbers > 1)
			.map(|(numbers, _)| u64::from(*numbers) - 1)
			.sum();

		assert_eq!(idle, 1, "one pause, and it cost nobody a command");
		assert_eq!(lost, 4, "and four commands the client made that never arrived");
	}

	/// A number far enough out is a discontinuity rather than a lie about
	/// order, and it must not be able to mute a peer.
	#[test]
	fn one_number_out_of_all_reason_costs_a_ring_rather_than_a_session() {
		let mut commands = Commands::new();
		let peer = client(6, 2);

		for number in [4_u32, 5] {
			assert!(commands.push(peer, asked(number)));
		}

		assert!(commands.settle(peer, 5), "the host has run both");

		// one flipped bit. Taken, because nothing here can tell it from a peer
		// that has jumped a long way, and the alternative to taking it is
		// refusing everything after it forever.
		let stray = 0x8000_0005;

		assert!(commands.push(peer, asked(stray)));
		assert_eq!(commands.kept(peer), &[asked(stray)], "what was held is gone");
		assert_eq!(
			commands.settled(peer),
			stray - 1,
			"and the mark went with it rather than back to nothing"
		);
		assert_eq!(commands.unsettled(peer), &[asked(stray)], "the stray runs once, and once");

		// and the very next honest command puts it right, which is the whole
		// point: without this the peer is silent for the rest of the session,
		// because every number it will ever send is below two billion.
		assert!(commands.push(peer, asked(6)));
		assert_eq!(commands.kept(peer), &[asked(6)]);
		assert_eq!(commands.unsettled(peer), &[asked(6)], "and is handed out to be run");
		assert_eq!(commands.settled(peer), 5, "having settled everything below it");

		// the window is a window and not a ceiling: a step short of it is
		// still this conversation, so an ordinary duplicate inside a ring's
		// depth is refused rather than starting the count over.
		assert!(!commands.push(peer, asked(6)), "a duplicate is still a duplicate");
		assert!(
			commands.push(peer, asked(6 + WINDOW)),
			"and the edge of the window is inside it"
		);
		assert_eq!(commands.kept(peer).len(), 2, "so nothing was thrown away for it");
	}

	#[test]
	fn a_mark_goes_neither_backwards_nor_past_what_has_arrived() {
		let mut commands = Commands::new();
		let peer = client(1, 8);

		for number in [3_u32, 7, 8, 12, 15] {
			assert!(commands.push(peer, asked(number)));
		}

		assert!(commands.settle(peer, 12));
		assert!(
			!commands.settle(peer, 12),
			"a mark already where it is being put has not moved, and the same acknowledgement \
			 arrives many times over"
		);
		assert!(!commands.settle(peer, 7), "two acknowledgements can overtake each other");
		assert_eq!(commands.settled(peer), 12);

		// a far end claiming to have settled a command nobody has sent. Capped
		// at what is here, so what it costs is the ring and not the future.
		assert!(commands.settle(peer, u32::MAX));
		assert_eq!(commands.settled(peer), 15, "the newest held, and no further");
		assert!(commands.unsettled(peer).is_empty());

		assert!(commands.push(peer, asked(16)));
		assert_eq!(
			commands.unsettled(peer),
			&[asked(16)],
			"so the next command asked for is still the client's own to be told about"
		);
	}

	/// The whole reason there is no way to let one peer go.
	///
	/// Handing a ring back to nobody lowers the generation `push` compares
	/// against, and from there the peer that had just left passes every guard
	/// and takes its own ring back with the mark reset - so the next redundant
	/// datagram carrying a command the host had already run puts it back in
	/// front of the host to run a second time. There is no call that can do
	/// this now, and this test is what says so: a departed peer's ring stays
	/// its own, with its mark, until somebody with a later generation claims
	/// it.
	#[test]
	fn a_peer_that_has_gone_cannot_hand_back_a_command_the_host_already_ran() {
		let mut commands = Commands::new();
		let gone = client(2, 3);

		for number in [4_u32, 5] {
			assert!(commands.push(gone, asked(number)));
		}

		assert!(commands.settle(gone, 5), "the host has run both");
		assert!(commands.unsettled(gone).is_empty());

		// the departed peer's last datagrams are still in flight, and every
		// one of them carries the last few commands over again.
		for number in [4_u32, 5] {
			assert!(!commands.push(gone, asked(number)), "{number} has been run once already");
		}

		assert!(commands.unsettled(gone).is_empty(), "and is not handed out a second time");
		assert_eq!(commands.owner(2), gone, "the ring is still the peer's that left");

		let next = client(2, 4);

		assert!(commands.push(next, asked(1)), "until somebody later takes it");
		assert_eq!(commands.kept(next), &[asked(1)]);
		assert_eq!(commands.settled(next), 0, "starting over, mark and all");
	}

	/// Slot zero is the host's, and this table says so on its own.
	#[test]
	fn nothing_but_the_host_may_ask_at_the_hosts_slot() {
		let mut commands = Commands::new();

		// a generation above the host's does not exist, so the interesting
		// ones are below it and either side of what a table in the house style
		// would mint first.
		for generation in [1_u32, 2, u32::MAX - 1] {
			assert!(
				!commands.push(client(0, generation), asked(2)),
				"generation {generation} is not the host"
			);
		}

		assert_eq!(commands.owner(PeerId::HOST.slot()), PeerId::NONE, "nothing was claimed");
		assert!(commands.push(PeerId::HOST, asked(2)), "and the host itself may");
		assert_eq!(commands.owner(PeerId::HOST.slot()), PeerId::HOST);

		// and the guard is the slot rather than the generation, which is what
		// the empty ring above already showed and this shows the other way:
		// with the ring given up, the generation comparison would let anybody
		// in, and the slot rule still does not.
		commands.clear();
		assert!(!commands.push(client(0, 1), asked(9)), "not even a ring nobody holds");
		assert_eq!(commands.owner(PeerId::HOST.slot()), PeerId::NONE);
	}

	#[test]
	fn clearing_the_table_gives_up_every_ring_including_the_hosts() {
		let mut commands = Commands::new();
		let mine = client(3, 2);
		let theirs = client(7, 5);

		for (peer, number) in [(mine, 40_u32), (theirs, 9), (PeerId::HOST, 12)] {
			assert!(commands.push(peer, asked(number)));
		}

		assert!(commands.settle(mine, 40));
		commands.clear();

		for slot in 0..MAX_PEERS {
			assert_eq!(commands.owner(slot), PeerId::NONE, "slot {slot} is nobody's again");
		}

		for peer in [mine, theirs, PeerId::HOST] {
			assert!(commands.kept(peer).is_empty());
			assert_eq!(commands.settled(peer), 0, "and the mark went with the commands");
			assert_eq!(commands.newest(peer), 0);
		}

		// the number that proves the ring was emptied rather than only
		// disowned: one is far below the forty that was in it, so a ring still
		// holding what it held would read this as an old command and drop it.
		assert!(commands.push(mine, asked(1)), "and a ring given up takes anybody's first");
		assert_eq!(commands.kept(mine), &[asked(1)]);

		// and the mark is read back *through the peer that holds the ring*,
		// which is the only way to see it: asked about a ring nobody holds,
		// `settled` answers nought because the lookup failed rather than
		// because the mark was cleared, and a mark left at forty would have
		// swallowed this command whole.
		assert_eq!(commands.settled(mine), 0);
		assert_eq!(commands.unsettled(mine), &[asked(1)], "so the new one is still to run");
	}

	#[test]
	fn two_peers_do_not_share_a_ring() {
		let mut commands = Commands::new();
		let mine = client(2, 6);
		let theirs = client(6, 2);

		assert!(commands.push(mine, asked(4)));
		assert!(commands.push(theirs, asked(9)));
		assert!(commands.settle(mine, 4));

		assert_eq!(commands.kept(mine), &[asked(4)]);
		assert_eq!(commands.kept(theirs), &[asked(9)]);
		assert!(commands.unsettled(mine).is_empty());
		assert_eq!(commands.unsettled(theirs), &[asked(9)], "and a mark is one peer's too");
	}

	#[test]
	fn a_slot_past_the_end_of_the_table_takes_nothing() {
		let mut commands = Commands::new();
		// the boundary and the slot below it, which is the pair rather than
		// the one: a table a slot too short refuses them both and looks from
		// the outside exactly like a table of the right size.
		let last = client(u32::try_from(MAX_PEERS - 1).expect("small"), 3);
		let past = client(u32::try_from(MAX_PEERS).expect("small"), 1);

		assert!(commands.push(last, asked(2)), "the last slot there is takes one");
		assert_eq!(commands.kept(last), &[asked(2)]);

		assert!(!commands.push(past, asked(1)));
		assert!(commands.kept(past).is_empty());
		assert!(!commands.settle(past, 1));
		assert_eq!(commands.owner(MAX_PEERS), PeerId::NONE, "and reading it is not a panic");
	}
}
