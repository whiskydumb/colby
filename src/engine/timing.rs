//! What a frame costs, in milliseconds, per part of it.
//!
//! **This exists because nothing measured step five.** Five cards landed
//! rendering work with no timing at all: up to thirty-two local lights walked
//! by every fragment with no tiles, a full-screen composite every frame, nine
//! tiny passes when the eye meters and eleven more when there is bloom, a
//! color target that went from eight bits a channel to sixteen, and four
//! samples on the geometry. Which of those costs anything is a question
//! nobody could answer, and guessing at it is how a renderer ends up
//! optimized in the wrong place.
//!
//! **Two clocks, because they answer different questions.** A GPU timestamp
//! written at the boundaries of a pass says how long the hardware spent
//! executing it; a wall clock around the same code says how long this process
//! spent *recording* it. A frame that is slow because the CPU cannot build the
//! instance buffer fast enough and a frame that is slow because thirty-two
//! lights are shaded per pixel look identical from one of the two and obvious
//! from both. Godot writes both at every mark it takes
//! (`rendering_device.cpp:8957-8970`), and bevy keeps the wall clock as the
//! answer for adapters with no timestamps at all
//! (`diagnostic/internal.rs:523-545`).
//!
//! **Five spans rather than twenty-six.** The question the debt asks is which
//! of the five things step five built is expensive, so the labels are those
//! five: the shadow cascades, the scene, the eye's ladder, the glow chain and
//! the composite. Cutting the ladder into its eight rungs is what to do once
//! one of the five is guilty, and a table of twenty-six rows would hide the
//! answer rather than give it. **A sixth since parity card B1**: the pass
//! that makes a multisampled depth readable, which is a price a frame pays
//! only while something reads the depth, and a difference between two runs
//! is worth less than a number measured directly.
//!
//! **Off until asked, and it has to be asked for twice.** The adapter feature
//! is requested when the device is made - it cannot be asked for later, and
//! the device is made long before anybody wants a number - but no query set
//! exists until [`Timings::start`] is called. Godot gates the whole mechanism
//! behind one bool (`storage/utilities.h:159`), Wicked ships it disabled
//! (`wiProfiler.cpp:27`), and bevy makes it a plugin somebody adds.
//!
//! **Reading a frame's numbers blocks.** [`Timings::settle`] waits for the
//! queue rather than picking the answer up two frames later the way every
//! reference does, and that is a deliberate trade: a run that stalls between
//! frames measures each frame on its own, with nothing from the frame before
//! it still in flight to be attributed to this one. It is the right shape for
//! a measuring run and the wrong shape for a window, which is why the mode
//! that uses it opens no window. What it costs is real and worth knowing: a
//! GPU idled between frames may clock differently from one that never is.

use std::{
	cell::Cell,
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	time::{Duration, Instant},
};

use wgpu::{
	Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, MapMode, PollType, QuerySet,
	QuerySetDescriptor, QueryType, RenderPassTimestampWrites,
};

/// How many nanoseconds a timestamp tick is worth when the queue will not say.
///
/// `get_timestamp_period` returns zero on an adapter with no timestamps, and a
/// period of zero would turn every duration into nought - which reads as "this
/// pass is free" rather than as "nothing was measured".
const UNKNOWN_PERIOD: f32 = 1.0;

/// Eight bytes a timestamp, which is what `resolve_query_set` writes.
const QUERY_SIZE: u64 = 8;

/// The parts of a frame the hardware is timed over.
///
/// In the order the frame runs them, which is the order they are reported in:
/// a table whose rows are shuffled relative to the work is one more thing to
/// hold in your head while reading it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pass {
	/// Every shadow cascade, as one span: four passes that are one feature.
	Shadow,

	/// Every local light's shadow map, as one span: one pass over the atlas's
	/// last layer, with a viewport per tile. None at all in a frame where no
	/// lamp asked for one. @ref [`shadow`](crate::shadow).
	Lamps,

	/// The world itself - the geometry, the sky, the debug lines and the
	/// blended half. Where the lights and the samples are paid for.
	Scene,

	/// The depth made readable: the one pass that writes the nearest of each
	/// pixel's samples into a buffer of one. Only at four samples a pixel, and
	/// only in a frame something reads the depth in. @ref
	/// [`depth`](crate::depth).
	Depth,

	/// The eye's ladder and the one texel it settles into: nine passes.
	Meter,

	/// The light the air catches around the sun: a mask and a smear at a
	/// quarter of the picture, and one pass putting the smear back over it.
	/// None at all while a world asks for no shafts or the sun is behind the
	/// camera. @ref [`shaft`](crate::shaft).
	Shaft,

	/// The lens out of focus: two blurs at half the picture on each axis, and
	/// one pass blending the result back over it by how far off the plane in
	/// focus each pixel is. None at all while the camera focuses on nothing.
	/// @ref [`focus`](crate::focus).
	Focus,

	/// The bloom chain, down and back up: eleven passes at a window's size,
	/// and none at all in a frame that does not bloom.
	Glow,

	/// The one full-screen pass that squeezes the float target onto the
	/// screen, or that draws the depth there instead while somebody is looking
	/// at it. The only thing here that runs in every frame without exception.
	Composite,
}

impl Pass {
	/// Every span, in the order a frame runs them.
	pub const ALL: [Self; 9] = [
		Self::Shadow,
		Self::Lamps,
		Self::Scene,
		Self::Depth,
		Self::Shaft,
		Self::Focus,
		Self::Meter,
		Self::Glow,
		Self::Composite,
	];

	/// The word for it in a report.
	#[must_use]
	pub const fn name(self) -> &'static str {
		match self {
			| Self::Shadow => "shadow",
			| Self::Lamps => "lamps",
			| Self::Scene => "scene",
			| Self::Depth => "depth",
			| Self::Shaft => "shaft",
			| Self::Focus => "focus",
			| Self::Meter => "meter",
			| Self::Glow => "glow",
			| Self::Composite => "composite",
		}
	}

	/// Which row of a [`Frame`] this span is, and which bit of the mask.
	const fn slot(self) -> usize {
		match self {
			| Self::Shadow => 0,
			| Self::Lamps => 1,
			| Self::Scene => 2,
			| Self::Depth => 3,
			| Self::Shaft => 4,
			| Self::Focus => 5,
			| Self::Meter => 6,
			| Self::Glow => 7,
			| Self::Composite => 8,
		}
	}

	/// Where this span's pair of timestamps begins in the query set.
	///
	/// Written out rather than `slot() * 2`, because the query set is indexed
	/// in `u32` and the tables are indexed in `usize`, and a silent conversion
	/// between the two is one this workspace does not allow. A test pins the
	/// two tables to each other.
	const fn query(self) -> u32 {
		match self {
			| Self::Shadow => 0,
			| Self::Lamps => 2,
			| Self::Scene => 4,
			| Self::Depth => 6,
			| Self::Shaft => 8,
			| Self::Focus => 10,
			| Self::Meter => 12,
			| Self::Glow => 14,
			| Self::Composite => 16,
		}
	}
}

/// The parts of a frame this process is timed over.
///
/// Deliberately short. Everything else a frame does - stepping the simulation,
/// laying the interface out - belongs to whoever calls the renderer, and a
/// measurement of it taken from in here would be a measurement of the wrong
/// call stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Work {
	/// Walking the world and writing this frame's buffers: the instances, the
	/// lights, the cascades, the materials that changed.
	Upload,

	/// Recording every pass into the encoder, from the first cascade to the
	/// submit. Not how long the GPU took - how long this thread took to
	/// describe it.
	Record,
}

impl Work {
	/// Both, in the order a frame runs them.
	pub const ALL: [Self; 2] = [Self::Upload, Self::Record];

	/// The word for it in a report.
	#[must_use]
	pub const fn name(self) -> &'static str {
		match self {
			| Self::Upload => "upload",
			| Self::Record => "record",
		}
	}

	/// Where this span is kept.
	const fn slot(self) -> usize {
		match self {
			| Self::Upload => 0,
			| Self::Record => 1,
		}
	}
}

/// Which end of a span a pass carries.
///
/// A span is often several passes - four shadow cascades, eight ladder rungs
/// and the eye, six halvings and five additions - so the first of them opens
/// it, the last closes it, and the ones between write nothing at all. Godot's
/// marks nest the same way, with `>` and `<` in the name
/// (`environment/fog.cpp:572,1228`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ends {
	/// The first pass of several.
	Open,

	/// One in between, which writes no timestamp and is only counted.
	Middle,

	/// The last of them.
	Close,

	/// A span that is one pass.
	Both,
}

/// How long each part of one frame took.
///
/// A span that did not run in the frame this describes is `None` rather than
/// zero: a glow chain nobody asked for and a glow chain that took no time at
/// all are different answers, and at ten nanoseconds a tick the second one
/// does not happen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Frame {
	/// What the hardware spent, per [`Pass`], in its `slot` order.
	passes: [Option<Duration>; 9],

	/// What this thread spent, per [`Work`], in its `slot` order.
	work: [Option<Duration>; 2],

	/// How many render passes the frame recorded.
	///
	/// Not a duration and the only thing here that does not move between two
	/// runs of the same scene, which is what makes it the part of a
	/// measurement worth comparing: a frame that grew a pass grew it for a
	/// reason somebody can name.
	count: u32,
}

impl Frame {
	/// What the hardware spent on one span.
	#[must_use]
	pub fn pass(&self, pass: Pass) -> Option<Duration> {
		self.passes.get(pass.slot()).copied().flatten()
	}

	/// What this thread spent on one span.
	#[must_use]
	pub fn work(&self, work: Work) -> Option<Duration> {
		self.work.get(work.slot()).copied().flatten()
	}

	/// How many render passes the frame recorded.
	#[must_use]
	pub const fn passes(&self) -> u32 { self.count }
}

/// Eight bytes a slot, little-endian, as `resolve_query_set` wrote them.
///
/// A short buffer leaves the rest of the slots at nought, which the mask then
/// keeps anybody from reading as an answer.
///
/// @param view - the mapped read buffer
/// @return one tick per slot
fn unpack(view: &[u8]) -> [u64; Timings::TICKS] {
	let mut ticks = [0_u64; Timings::TICKS];

	for (slot, eight) in view.chunks_exact(8).enumerate() {
		let (Some(held), Ok(bytes)) = (ticks.get_mut(slot), eight.try_into()) else {
			continue;
		};

		*held = u64::from_le_bytes(bytes);
	}

	ticks
}

/// The measuring apparatus, inert until somebody asks for it.
///
/// Lives on a [`Scene`](crate::Scene), so a window and a capture both have one
/// and neither pays for it: with nothing started there is no query set, no
/// buffer and no timestamp written, and every pass descriptor gets the `None`
/// it got before this module existed.
#[derive(Debug)]
pub struct Timings {
	/// Two slots per [`Pass`], or `None` while nobody is measuring and on an
	/// adapter that cannot.
	set: Option<QuerySet>,

	/// Where `resolve_query_set` writes, and what is copied out of.
	resolved: Option<Buffer>,

	/// The mappable copy the numbers are read from.
	read: Option<Buffer>,

	/// Nanoseconds a tick is worth on this queue.
	period: f32,

	/// Which spans wrote a timestamp this frame, one bit per `Pass::slot`.
	///
	/// A slot that was never written holds whatever the query set held before,
	/// which is not defined - so a span nobody ran has to be known to be
	/// absent rather than read and believed.
	///
	/// A [`Cell`] because the alternative is worse: a pass descriptor names
	/// the query set *and* the target *and* the bind groups, all of which live
	/// on the same struct, so handing one out through `&mut self` locks the
	/// rest of the frame out of its own fields. One bit set behind a shared
	/// reference is the smaller of the two costs.
	ran: Cell<u32>,

	/// How many passes this frame has recorded so far.
	///
	/// Counted through the same call every pass already makes, so it cannot
	/// drift from the frame the way a second tally computed from the world
	/// would. Behind a [`Cell`] for the reason `ran` is.
	passes: Cell<u32>,

	/// Where each wall-clock span started, until it ends.
	marks: [Option<Instant>; 2],

	/// The frame most recently settled.
	last: Frame,

	/// The frame whose timestamps are being read back, and the mask of which
	/// spans ran in it, held until the numbers arrive.
	///
	/// **A frame's answers stay together.** The readback lands two or three
	/// frames after the frame it is about, and the wall-clock half of that
	/// frame is long gone from [`last`](Self::last), which [`begin`] clears -
	/// so the whole of it is stashed here when the map is asked for and handed
	/// back complete. The alternative, GPU numbers from one frame beside CPU
	/// numbers from another, is a table describing two things.
	awaiting: Option<(Frame, u32)>,

	/// Whether a map has been asked for and not yet read.
	///
	/// A [`Cell`] because [`resolve`](Self::resolve) reads it through a shared
	/// reference, for the reason `ran` and `passes` are cells: the encoder,
	/// the target and the bind groups are fields of one struct and a `&mut`
	/// here would lock the rest of the frame out.
	mapping: Cell<bool>,

	/// Set by the map callback when the buffer is readable.
	///
	/// Shared with a closure wgpu keeps, so an `Arc` rather than a `Cell`.
	ready: Arc<AtomicBool>,
}

impl Timings {
	/// How many timestamps the set holds: two per span.
	const QUERIES: u32 = 18;
	/// The same number where a length is wanted. @ref [`Pass::query`].
	const TICKS: usize = 18;

	/// An apparatus that measures nothing.
	///
	/// @param period - what the queue says a tick is worth in nanoseconds,
	/// which is zero on an adapter with no timestamps
	pub(crate) fn new(period: f32) -> Self {
		Self {
			set: None,
			resolved: None,
			read: None,
			period: if period > 0.0 { period } else { UNKNOWN_PERIOD },
			ran: Cell::new(0),
			passes: Cell::new(0),
			marks: [None; 2],
			last: Frame::default(),
			awaiting: None,
			mapping: Cell::new(false),
			ready: Arc::new(AtomicBool::new(false)),
		}
	}

	/// Whether the hardware side of this is switched on.
	#[must_use]
	pub const fn timing(&self) -> bool { self.set.is_some() }

	/// Builds the query set and the two buffers, if the device can.
	///
	/// Idempotent: asking twice is asking once. A device whose feature was not
	/// requested keeps the wall clock and nothing else, which is bevy's
	/// fallback and is honest - half the question is answerable without any
	/// hardware support at all.
	///
	/// @param device - the device the set is made on
	/// @return whether the hardware side came up
	pub fn start(&mut self, device: &Device) -> bool {
		if self.set.is_some() {
			return true;
		}

		if !device
			.features()
			.contains(wgpu::Features::TIMESTAMP_QUERY)
		{
			return false;
		}

		let size = u64::from(Self::QUERIES) * QUERY_SIZE;

		self.set = Some(device.create_query_set(&QuerySetDescriptor {
			label: Some("frame timings"),
			ty: QueryType::Timestamp,
			count: Self::QUERIES,
		}));
		self.resolved = Some(device.create_buffer(&BufferDescriptor {
			label: Some("frame timings"),
			size,
			usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
			mapped_at_creation: false,
		}));
		self.read = Some(device.create_buffer(&BufferDescriptor {
			label: Some("frame timings read"),
			size,
			usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		}));

		true
	}

	/// Forgets the frame before this one.
	///
	/// Called at the top of every frame whether anything is being measured or
	/// not, so that a frame drawn with the apparatus off does not leave a
	/// stale answer behind for the next one that has it on.
	pub(crate) fn begin(&mut self) {
		self.ran.set(0);
		self.passes.set(0);
		self.marks = [None; 2];
		self.last = Frame::default();
	}

	/// Starts a wall-clock span.
	pub(crate) fn open(&mut self, work: Work) { self.marks[work.slot()] = Some(Instant::now()); }

	/// Ends one, and keeps what it took.
	///
	/// A span closed without being opened is nothing rather than a panic: the
	/// renderer has early returns in it, and a measurement is not worth
	/// stopping a frame over.
	pub(crate) fn close(&mut self, work: Work) {
		let slot = work.slot();

		if let Some(started) = self.marks[slot].take() {
			self.last.work[slot] = Some(started.elapsed());
		}
	}

	/// What a pass descriptor writes for one end of one span.
	///
	/// `None` when nothing is being measured, which is what the field held
	/// before this module existed - so a call site reads the same either way
	/// and there is no second path through the frame to keep working.
	///
	/// @param pass - which span
	/// @param ends - whether this pass opens it, closes it, or is all of it
	pub(crate) fn writes(&self, pass: Pass, ends: Ends) -> Option<RenderPassTimestampWrites<'_>> {
		let slot = pass.slot();

		// every pass calls this, including the ones in the middle of a span
		// that write no timestamp at all, which is what makes the count exact
		// rather than derived.
		self.passes
			.set(self.passes.get().saturating_add(1));

		// on the way in rather than on the way out: a span whose opening pass
		// ran is a span that ran, and the closing one is not always reached.
		if matches!(ends, Ends::Open | Ends::Both) {
			self.ran.set(self.ran.get() | (1 << slot));
		}

		// wgpu refuses a descriptor with neither end, so a pass in the middle
		// is counted and then handed the `None` it would have had anyway.
		if matches!(ends, Ends::Middle) {
			return None;
		}

		Some(RenderPassTimestampWrites {
			query_set: self.set.as_ref()?,
			beginning_of_pass_write_index: match ends {
				| Ends::Open | Ends::Both => Some(pass.query()),
				| Ends::Middle | Ends::Close => None,
			},
			end_of_pass_write_index: match ends {
				| Ends::Close | Ends::Both => Some(pass.query() + 1),
				| Ends::Middle | Ends::Open => None,
			},
		})
	}

	/// Records the resolve and the copy out, at the end of the frame's
	/// encoder.
	///
	/// Every slot is resolved rather than only the ones that were written -
	/// the range has to be one range, and which spans ran is already known
	/// from the mask.
	///
	/// @param encoder - the frame's own, before it is finished
	pub(crate) fn resolve(&self, encoder: &mut CommandEncoder) {
		let (Some(set), Some(resolved), Some(read)) =
			(self.set.as_ref(), self.resolved.as_ref(), self.read.as_ref())
		else {
			return;
		};

		encoder.resolve_query_set(set, 0..Self::QUERIES, resolved, 0);

		// **not into a buffer somebody is mapping.** The query set is resolved
		// every frame whatever happens, because it is the hardware's own
		// staging and nothing else reads it; the copy out is what would be a
		// write to a mapped buffer, and wgpu refuses that. While a readback is
		// in flight this frame's numbers are simply not collected, which is
		// what makes the live path sample rather than take every frame. @ref
		// [`poll`](Self::poll).
		if self.mapping.get() {
			return;
		}

		encoder.copy_buffer_to_buffer(
			resolved,
			0,
			read,
			0,
			u64::from(Self::QUERIES) * QUERY_SIZE,
		);
	}

	/// This frame's wall-clock spans, as they stand.
	///
	/// No device and no waiting: the two recording spans are written by
	/// [`close`](Self::close) as the frame is recorded, so they are already
	/// here. The pass count is taken from the same tally the frame kept.
	#[must_use]
	pub fn spans(&self) -> Frame { Frame { count: self.passes.get(), ..self.last } }

	/// Gives the query set and both buffers back, and forgets what was in
	/// flight.
	///
	/// The other end of [`start`](Self::start), so that a panel that turns
	/// measuring on when it opens can turn it off when it closes. A buffer
	/// still mapped is unmapped first: wgpu will not free one that is, and a
	/// readback nobody is going to read is not worth waiting for.
	pub fn stop(&mut self) {
		if self.mapping.get()
			&& let Some(read) = self.read.as_ref()
			&& self.ready.load(Ordering::Acquire)
		{
			read.unmap();
		}

		self.set = None;
		self.resolved = None;
		self.read = None;
		self.awaiting = None;
		self.mapping.set(false);
		self.ready.store(false, Ordering::Release);
		self.last = Frame::default();
	}

	/// Asks for the numbers without waiting, and hands back a frame's worth
	/// when one has arrived.
	///
	/// **The other half of [`settle`](Self::settle), and the difference is the
	/// whole point.** `settle` blocks on the queue so that each frame's
	/// numbers are its own, which is what a measuring run wants and what a
	/// window cannot afford: a window that waits on the queue every frame has
	/// given up the pipelining that makes it a window, and the numbers it then
	/// reads are about a stalled frame rather than a real one.
	///
	/// So this does what every engine read for `PERF-1` does and colby did not
	/// need until now - bevy collects on a later frame's `begin_frame`, Godot
	/// after that frame's fence, Wicked from the previous swapchain cycle:
	/// **the answer arrives late and nothing waits.** A frame's numbers come
	/// back two or three frames after it, complete, or not at all.
	///
	/// Call it once a frame, after the frame has been submitted.
	///
	/// @param device - the device whose callbacks are pumped
	/// @return the frame a readback has just completed for, if one has
	pub fn poll(&mut self, device: &Device) -> Option<Frame> {
		// nothing to read from means nothing was ever started, which is what
		// every frame of every window that never opened the panel looks like
		self.read.as_ref()?;

		// callbacks fire on a poll and nowhere else, and this is the
		// non-blocking one: a queue with nothing finished says so and the
		// frame goes on.
		drop(device.poll(PollType::Poll));

		if !self.mapping.get() {
			self.request();

			return None;
		}

		if !self.ready.load(Ordering::Acquire) {
			return None;
		}

		self.collect()
	}

	/// Asks for the map that a later [`poll`](Self::poll) reads.
	///
	/// The frame being stashed is the one whose copy the encoder has just
	/// recorded, so what comes back later is that frame and not the one it
	/// arrives in.
	fn request(&mut self) {
		let Some(read) = self.read.as_ref() else {
			return;
		};

		self.last.count = self.passes.get();
		self.awaiting = Some((self.last, self.ran.get()));
		self.ready.store(false, Ordering::Release);
		self.mapping.set(true);

		let ready = Arc::clone(&self.ready);

		read.slice(..)
			.map_async(MapMode::Read, move |outcome| {
				// a failed map is still an answer: the flag says the buffer is
				// no longer in flight, and `collect` finds no numbers in it
				// and hands back the wall-clock half alone.
				drop(outcome);
				ready.store(true, Ordering::Release);
			});
	}

	/// Reads the mapped buffer, frees it, and completes the stashed frame.
	fn collect(&mut self) -> Option<Frame> {
		let ticks = {
			let read = self.read.as_ref()?;
			let ticks = read
				.slice(..)
				.get_mapped_range()
				.map_or_else(|_| [0_u64; Self::TICKS], |view| unpack(&view));

			read.unmap();

			ticks
		};

		self.mapping.set(false);
		self.ready.store(false, Ordering::Release);

		let (mut frame, ran) = self.awaiting.take()?;
		// against the mask the stashed frame was taken with, not the one this
		// frame left behind: a span that ran two frames ago is what these
		// ticks are about.
		let was = self.ran.replace(ran);

		for pass in Pass::ALL {
			if let Some(held) = frame.passes.get_mut(pass.slot()) {
				*held = self.span(&ticks, pass);
			}
		}

		self.ran.set(was);

		Some(frame)
	}

	/// Waits for the queue and reads what the hardware said.
	///
	/// **This blocks**, which is the module note's whole caveat. Call it after
	/// the frame has been submitted and before the next one is recorded.
	///
	/// @param device - the device to wait on
	/// @return what the frame that has just been submitted cost
	pub fn settle(&mut self, device: &Device) -> Frame {
		self.last.count = self.passes.get();

		let Some(ticks) = self.download(device) else {
			return self.last;
		};
		// read into a table of its own and then written over the frame in one
		// go: `span` reads the mask off `self` and the frame is a field of
		// the same `self`, so the two cannot be borrowed at once.
		let mut spans = [None; 9];

		for pass in Pass::ALL {
			if let Some(held) = spans.get_mut(pass.slot()) {
				*held = self.span(&ticks, pass);
			}
		}

		self.last.passes = spans;

		self.last
	}

	/// Waits for the queue and maps the set's numbers out of the read buffer.
	///
	/// Split off the caller so that the borrow of the buffer ends before the
	/// frame is written into: they are two fields of one struct, and one is
	/// read while the other is written.
	///
	/// @param device - the device to wait on
	/// @return the ticks, or nothing at all when there is no set and when the
	/// wait failed
	fn download(&self, device: &Device) -> Option<[u64; Self::TICKS]> {
		let read = self.read.as_ref()?;
		let slice = read.slice(..);

		slice.map_async(MapMode::Read, |_| {});
		device
			.poll(PollType::Wait { submission_index: None, timeout: None })
			.ok()?;

		let ticks = slice
			.get_mapped_range()
			.map_or_else(|_| [0_u64; Self::TICKS], |view| unpack(&view));

		read.unmap();

		Some(ticks)
	}

	/// One span's pair of ticks turned into a length of time.
	///
	/// `None` for a span that did not run this frame, and for a pair that came
	/// back the wrong way round: a timestamp on a tiled adapter that drew
	/// nothing can be nonsense, which Wicked guards against by throwing away
	/// anything absurd (`wiProfiler.cpp:130-134`).
	#[expect(
		clippy::as_conversions,
		clippy::cast_precision_loss,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "a pass is nanoseconds to milliseconds long, nowhere near where f64 or u64 \
		          stop holding it"
	)]
	fn span(&self, ticks: &[u64; Self::TICKS], pass: Pass) -> Option<Duration> {
		let slot = pass.slot();

		if self.ran.get() & (1 << slot) == 0 {
			return None;
		}

		let start = *ticks.get(slot * 2)?;
		let end = *ticks.get(slot * 2 + 1)?;
		let elapsed = end.checked_sub(start)?;

		Some(Duration::from_nanos((elapsed as f64 * f64::from(self.period)) as u64))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn polling_an_apparatus_nobody_started_is_nothing_rather_than_a_panic() {
		// the window calls this every frame whether the tab is open or not,
		// and on an adapter with no timestamps it never becomes anything
		let mut timings = Timings::new(1.0);
		let Some(gpu) = device() else {
			return;
		};

		assert!(timings.poll(gpu.device()).is_none(), "nothing was asked for");
		assert!(!timings.mapping.get(), "and nothing is in flight");
	}

	#[test]
	fn a_frame_polled_for_comes_back_late_and_whole() {
		let Some(gpu) = device() else {
			return;
		};
		let mut capture = match crate::Capture::new(gpu, 32, 32) {
			| Ok(capture) => capture,
			| Err(error) => panic!("building the capture failed: {error}"),
		};

		if !capture.scene_mut().measure() {
			// no timestamp queries on this adapter, which the module treats as
			// the wall clock alone
			return;
		}

		let mut world = colby_core::abi::World::new();
		let mut got = None;

		// twenty frames is far more than the two or three a readback takes,
		// and a loop that never gets one is the failure this asserts against
		for _ in 0..20 {
			capture.draw(&mut world, &mut []);

			if let Some(frame) = capture.scene_mut().collect() {
				got = Some(frame);

				break;
			}
		}

		let frame = got.expect("a readback landed inside twenty frames without anything waiting");

		assert!(frame.passes() > 0, "the frame recorded passes: {}", frame.passes());
		assert!(
			Pass::ALL
				.into_iter()
				.any(|pass| frame.pass(pass).is_some()),
			"and at least one hardware span came back with a number in it"
		);
		assert!(
			frame.work(Work::Record).is_some(),
			"with the wall-clock half of the very same frame beside it, which is what the stash \
			 is for"
		);
	}

	#[test]
	fn nothing_is_copied_into_the_buffer_while_it_is_being_mapped() {
		// the one hazard in the arrangement: wgpu refuses a copy into a mapped
		// buffer, and `resolve` runs every frame whatever the readback is doing
		let mut timings = Timings::new(1.0);
		let Some(gpu) = device() else {
			return;
		};

		if !timings.start(gpu.device()) {
			return;
		}

		let mut encoder = gpu
			.device()
			.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

		timings.mapping.set(false);
		timings.resolve(&mut encoder);
		timings.mapping.set(true);
		timings.resolve(&mut encoder);

		// the assertion is that the second call returned without recording a
		// copy, which is what keeps the submit below legal
		gpu.queue().submit([encoder.finish()]);
		drop(
			gpu.device()
				.poll(PollType::Wait { submission_index: None, timeout: None }),
		);
	}

	/// A device, or nothing on a machine with no usable adapter.
	fn device() -> Option<&'static crate::Gpu> { crate::gpu::shared() }

	#[test]
	fn every_span_has_a_slot_of_its_own_and_the_set_is_big_enough_for_all_of_them() {
		// the one invariant the whole thing rests on: two slots per span, and
		// no two spans sharing one. A collision here is a glow chain reported
		// as a composite, which reads as a plausible number.
		let mut slots: Vec<usize> = Pass::ALL.iter().map(|pass| pass.slot()).collect();
		slots.sort_unstable();
		slots.dedup();

		assert_eq!(slots.len(), Pass::ALL.len(), "two spans share a slot");
		assert_eq!(
			Timings::TICKS,
			Pass::ALL.len() * 2,
			"the set holds a beginning and an end for every span and nothing else"
		);
		assert_eq!(
			usize::try_from(Timings::QUERIES).expect("a small count fits"),
			Timings::TICKS,
			"the count the set is made with and the array it is read into"
		);

		for pass in Pass::ALL {
			// the two tables are written out separately because one is `u32`
			// and the other `usize`; this is what keeps them one table.
			assert_eq!(
				usize::try_from(pass.query()).expect("a slot fits"),
				pass.slot() * 2,
				"{} disagrees with itself about where its ticks are",
				pass.name()
			);
			assert!(pass.query() + 1 < Timings::QUERIES, "{} writes past the set", pass.name());
		}
	}

	#[test]
	fn a_span_nobody_ran_is_absent_rather_than_free() {
		// the reason the mask exists. An unwritten slot holds whatever was in
		// the query set, so a glow chain that did not run must not come back
		// as a duration at all - least of all as a convincing zero.
		let timings = Timings::new(1.0);
		let ticks = [7_u64; Timings::TICKS];

		for pass in Pass::ALL {
			assert_eq!(timings.span(&ticks, pass), None, "{} was never recorded", pass.name());
		}
	}

	#[test]
	fn a_span_that_ran_is_the_difference_between_its_two_marks_in_real_time() {
		let timings = Timings::new(10.0);
		let mut ticks = [0_u64; Timings::TICKS];

		// a hundred thousand ticks at ten nanoseconds each is a millisecond,
		// which is the size of answer this is for.
		timings.ran.set(1 << Pass::Scene.slot());
		ticks[Pass::Scene.slot() * 2] = 500;
		ticks[Pass::Scene.slot() * 2 + 1] = 100_500;

		assert_eq!(timings.span(&ticks, Pass::Scene), Some(Duration::from_millis(1)));
		assert_eq!(timings.span(&ticks, Pass::Glow), None, "a neighbor is not dragged in");
	}

	#[test]
	fn a_pair_that_came_back_backwards_is_no_answer_rather_than_a_huge_one() {
		// subtracting the other way round would be about six hundred years,
		// which is the shape of the number a tiled adapter hands back when a
		// pass drew nothing.
		let timings = Timings::new(10.0);
		let mut ticks = [0_u64; Timings::TICKS];

		timings.ran.set(1 << Pass::Meter.slot());
		ticks[Pass::Meter.slot() * 2] = 900;
		ticks[Pass::Meter.slot() * 2 + 1] = 100;

		assert_eq!(timings.span(&ticks, Pass::Meter), None);
	}

	#[test]
	fn an_adapter_that_will_not_say_what_a_tick_is_worth_does_not_make_every_pass_free() {
		// `get_timestamp_period` answers zero where there are no timestamps,
		// and a period of zero multiplies every measurement into nothing -
		// which reads as a renderer that costs nothing at all.
		assert!(Timings::new(0.0).period > 0.0);
		assert!(Timings::new(-1.0).period > 0.0);
		assert!((Timings::new(10.0).period - 10.0).abs() < f32::EPSILON);
	}

	#[test]
	fn a_wall_clock_span_is_only_reported_once_it_has_been_closed() {
		let mut timings = Timings::new(1.0);

		timings.begin();
		timings.open(Work::Upload);

		assert_eq!(timings.last.work(Work::Upload), None, "an open span has no length yet");

		timings.close(Work::Upload);

		assert!(timings.last.work(Work::Upload).is_some(), "a closed one does");
	}

	#[test]
	fn closing_a_span_nobody_opened_is_nothing_rather_than_a_panic() {
		// the renderer returns early in the middle of a frame when its
		// viewport has nothing inside the target, and a measurement is not
		// worth stopping a frame over.
		let mut timings = Timings::new(1.0);

		timings.begin();
		timings.close(Work::Record);

		assert_eq!(timings.last.work(Work::Record), None);
	}

	#[test]
	fn nothing_is_built_until_it_is_started() {
		let timings = Timings::new(10.0);

		assert!(!timings.timing(), "an apparatus nobody asked for holds no query set");
	}

	#[test]
	fn the_names_are_all_different_and_all_lowercase() {
		// they end up in one table beside each other, so two spans with one
		// word is a row nobody can read.
		let mut names: Vec<&str> = Pass::ALL
			.iter()
			.map(|pass| pass.name())
			.chain(Work::ALL.iter().map(|work| work.name()))
			.collect();
		let all = names.len();
		names.sort_unstable();
		names.dedup();

		assert_eq!(names.len(), all, "two spans answer to one word");
		assert!(
			names.iter().all(|name| name
				.chars()
				.all(|letter| letter.is_ascii_lowercase())),
			"a name is one lowercase word"
		);
	}
}
