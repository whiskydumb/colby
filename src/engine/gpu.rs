//! One device for the process, however many things draw with it.
//!
//! A window and an offscreen picture used to bring up an adapter and a device
//! each, and a thumbnail or a second viewport would have been a third. What a
//! second device costs is not the memory: two devices cannot share a texture,
//! so a picture drawn by one can never be shown by the other without a trip
//! through the CPU, and a material preview inside an editor panel is exactly
//! that picture. So there is one. [`Gpu`] is the instance, the adapter it
//! chose, the device and its queue; a [`Renderer`](crate::Renderer) per window
//! and a [`Capture`](crate::Capture) per picture are built against it and share
//! it, each with a [`Scene`](crate::Scene) of its own.
//!
//! **Which APIs are considered is decided here and once.** Nobody switches API
//! after startup, and neither does this: the variable [`BACKEND`] is read when
//! the device is made and never again, the environment's `WGPU_BACKEND` is
//! read over it for one run, and everything built on the device lives with the
//! answer. @ref [`backends`], which is the whole of that rule.

use std::{env, sync::Arc};

use colby_core::{Result, debug, err, info, warn};
use wgpu::{
	Adapter, BackendOptions, Backends, Device, DeviceDescriptor, ExperimentalFeatures, Features,
	Instance, InstanceDescriptor, InstanceFlags, Limits, MemoryHints, NoopBackendOptions,
	PowerPreference, Queue, RequestAdapterOptions, Trace,
};
use winit::window::Window;

use crate::brdf::Split;

/// The APIs considered when nobody says otherwise.
///
/// wgpu's own first tier - Vulkan, Metal, DirectX 12 and WebGPU - of which a
/// build only ever has the ones it compiled in, so on Windows this is DirectX
/// 12 and Vulkan and the adapter that wins is the one wgpu ranks first. GL is
/// left out on purpose: wgpu calls it a secondary backend, and a machine that
/// has nothing else is a machine to hear about rather than to draw on quietly.
pub const DEFAULT: Backends = Backends::PRIMARY;

/// The variable that says which graphics APIs to consider.
///
/// Saved, because which API a machine draws with is a property of the machine
/// rather than of a session; and read once, so a value typed at a running
/// window takes effect at the next start, which is what every engine does
/// with this setting. Its value is [`AUTO`] or a comma list of the words wgpu
/// itself reads - `vulkan`, `dx12`, `metal`, `gl` - so that the variable and
/// the environment say the same things in the same words.
pub const BACKEND: &str = "r.backend";

/// The word for "whatever wgpu ranks first among the APIs this build has".
pub const AUTO: &str = "auto";

/// The environment variable that overrides [`BACKEND`] for one run.
///
/// wgpu's own, read by its examples and by the engines built on it, so a
/// person who knows one knows this one. Read here rather than through wgpu so
/// that what it wins over and what happens to a word nobody knows are decided
/// in one place and said in the log.
const BACKEND_VAR: &str = "WGPU_BACKEND";

/// Who decided which APIs to consider, for the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum By {
	/// `WGPU_BACKEND` in the environment.
	Environment,

	/// The variable, from the config or typed at the console.
	Variable,

	/// Nobody, so [`DEFAULT`].
	Default,
}

impl By {
	const fn as_str(self) -> &'static str {
		match self {
			| Self::Environment => BACKEND_VAR,
			| Self::Variable => BACKEND,
			| Self::Default => "the default",
		}
	}
}

/// Which APIs to consider: the environment over the variable over the default.
///
/// A level that says something this build does not understand is warned about
/// and skipped rather than taken as nothing: an override made of a word nobody
/// knows still draws, with whatever the variable says.
///
/// @param asked - what [`BACKEND`] holds, if there is a console to hold it
/// @return the set to hand to [`Gpu::open`]
#[must_use]
pub fn backends(asked: Option<&str>) -> Backends {
	let forced = env::var(BACKEND_VAR).ok();
	let (chosen, by) = choose(asked, forced.as_deref(), Instance::enabled_backend_features());
	debug!(backends = ?chosen, by = by.as_str(), "graphics APIs to consider");

	chosen
}

/// The precedence, with the environment and the build's own set handed in so
/// that a test can say what each holds.
///
/// A level is skipped when nothing in it is an API this build has - `dx12` in
/// a `settings.cfg` carried over from Windows would otherwise stop a Linux
/// window with "no adapter" - and a list is narrowed to what is here, because
/// a person who lists alternatives means whichever of them there is.
///
/// @param asked - what [`BACKEND`] holds, if there is a console to hold it
/// @param forced - what [`BACKEND_VAR`] holds, if it is set
/// @param have - the APIs this build compiled in for this platform
fn choose(asked: Option<&str>, forced: Option<&str>, have: Backends) -> (Backends, By) {
	if let Some(text) = forced.filter(|text| !text.trim().is_empty())
		&& let Some(set) = taken(BACKEND_VAR, text, have)
	{
		return (set, By::Environment);
	}

	if let Some(text) = asked
		&& let Some(set) = taken(BACKEND, text, have)
	{
		return (set, By::Variable);
	}

	(DEFAULT & have, By::Default)
}

/// The one device a test binary draws with.
///
/// **Because the alternative was measured and it crashes.** Every rendered
/// test in this workspace used to open a device of its own; the engine's
/// suite alone opened more than fifty of them, up to sixteen alive at once on
/// this machine, and the driver fell over on `STATUS_ACCESS_VIOLATION` with no
/// panic and no failing test. Measured 2026-09-08 before this existed: **six
/// crashes in twenty runs** of the engine's binary back to back.
///
/// One `OnceLock` a test binary, so three of them in this workspace - the
/// engine's, the interface's and the editor's - because a `cfg(test)` item is
/// not visible to another crate and the alternative was one more shipped
/// function that only a test calls. @ref `colby-gate-gotchas` for the
/// measurement, which is the whole argument for this.
///
/// It is never dropped, which for a test binary is the point: a device that
/// outlives every test on it cannot be the thing a later test is waiting for.
///
/// **The same shape of fault came back once and was something else**, so the
/// device is the first thing to check and not the only one: on 2026-09-12 the
/// engine's suite faulted two runs in five with every test on this one device,
/// and what it was that time was the graphics API's own debug layer. @ref
/// [`layers`].
///
/// @return the device, or `None` on a machine with no adapter - which every
/// caller skips on rather than failing
#[cfg(test)]
pub(crate) fn shared() -> Option<&'static Gpu> {
	static SHARED: std::sync::OnceLock<Option<Gpu>> = std::sync::OnceLock::new();

	SHARED
		.get_or_init(|| match Gpu::open(backends(None), None) {
			| Ok(gpu) => gpu,
			| Err(error) => panic!("opening the device failed: {error}"),
		})
		.as_ref()
}

/// What one level comes to, with a line said when it comes to nothing.
///
/// **Three refusals and not one**, which is the whole of why this is a
/// function: `gl` is a word wgpu knows that this build was not compiled with,
/// `noop` is a word wgpu knows that draws nothing anybody could look at, and
/// `dx11` is nobody's word at all. All three used to say "names no graphics
/// API this build has", which is true of one of them and misleading about the
/// other two - and the one a person hits is the one the answer matters for.
///
/// @param source - what the value came from, for the log
/// @param text - the value
/// @param have - the APIs this build compiled in for this platform
/// @return the part of it this build has, or nothing with a reason said
fn taken(source: &str, text: &str, have: Backends) -> Option<Backends> {
	match parse(text) {
		| Read::Apis(set) => {
			let kept = set & have;

			if !kept.is_empty() {
				return Some(kept);
			}

			warn!(
				source,
				text,
				missing = ?set.difference(have),
				"this build was not compiled with that graphics API"
			);
		},
		| Read::NotDrawn(why) => warn!(source, text, "{why}"),
		| Read::Unknown => warn!(source, text, "no graphics API goes by that name"),
	}

	None
}

/// What a value of [`BACKEND`] turned out to say.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Read {
	/// wgpu's words for APIs a picture could come out of.
	Apis(Backends),

	/// A word wgpu knows for something no picture comes out of, and why.
	NotDrawn(&'static str),

	/// A word nothing here knows.
	Unknown,
}

/// Reads a value of [`BACKEND`]: [`AUTO`], or a comma list of wgpu's words.
///
/// wgpu's own reader takes the same words, warns through a logger this process
/// does not listen to, and hands back an empty set for a value made of nothing
/// it knows; this one refuses the whole value on the first word it does not
/// know, so that a list with a misspelling in it is a warning rather than a
/// quiet subset. `webgpu` and `noop` are wgpu's words too and are refused on
/// purpose, each with its own sentence: one is a browser and the other draws
/// nothing.
///
/// @param text - the value
/// @return the set, or which kind of word it stopped on
fn parse(text: &str) -> Read {
	if text.trim().eq_ignore_ascii_case(AUTO) {
		return Read::Apis(DEFAULT);
	}

	let mut set = Backends::empty();
	for word in text.split(',') {
		set |= match word.trim().to_ascii_lowercase().as_str() {
			| "vulkan" | "vk" => Backends::VULKAN,
			| "dx12" | "d3d12" => Backends::DX12,
			| "metal" | "mtl" => Backends::METAL,
			| "gl" | "gles" | "opengl" => Backends::GL,
			| "webgpu" =>
				return Read::NotDrawn("webgpu is a browser's API, and this is not one"),
			| "noop" =>
				return Read::NotDrawn(
					"noop draws nothing; it is what the headless tests build pipelines on",
				),
			| _ => return Read::Unknown,
		};
	}

	Read::Apis(set)
}

/// The instance, the adapter, the device and its queue.
///
/// Held together because they are made together and in this order, and
/// because nothing below the device makes sense without the instance it came
/// from: a surface is made on the instance and configured against the adapter,
/// and both have to be the ones the device belongs to.
pub struct Gpu {
	instance: Instance,
	adapter: Adapter,
	device: Device,
	queue: Queue,

	/// The split-sum table of the shading lobe, baked once for this device.
	///
	/// **Here rather than on the scene that reads it**, because the table
	/// depends on the device and on nothing else: it is the same one for
	/// every world, every camera and every size a device ever draws, so a copy
	/// per scene would be a texture, a view and a sampler apiece for one
	/// answer. @ref [`brdf`](crate::brdf).
	split: Arc<Split>,
}

impl Gpu {
	/// Brings up an adapter and a device on the APIs asked for.
	///
	/// @param backends - which APIs to consider; @ref [`backends`]
	/// @param present_to - a window the adapter has to be able to present to,
	/// if there is one yet. A surface is made for it and dropped again here:
	/// all that is wanted from it is the adapter's answer to "can you present
	/// to this", which on Windows every adapter gives and on Linux with two
	/// GPUs not every adapter does. A picture with no window passes nothing.
	/// @return the device, or `None` when no adapter could be found - which a
	/// test skips on and a window stops on
	pub fn open(backends: Backends, present_to: Option<&Window>) -> Result<Option<Self>> {
		pollster::block_on(Self::create(backends, present_to))
	}

	/// A device that validates everything and draws nothing.
	///
	/// wgpu's `noop` backend: every operation is an empty body except making
	/// and mapping a buffer, so a [`Capture`](crate::Capture) on this comes
	/// back all zeros and no test may look at a pixel of it. What still runs
	/// is all of wgpu's validation and all of naga's - a shader that does not
	/// compile, a bind group that does not match the layout it was built
	/// against, a pipeline whose fragment stage writes a format its target
	/// does not have - so the question "does every pipeline this engine builds
	/// still build" has an answer here, **on a machine with no graphics
	/// hardware at all**.
	///
	/// That is the whole of what it is for, and it is why it is a function of
	/// its own rather than a word [`BACKEND`] takes: a person who typed
	/// `r.backend noop` at a window would get a black screen and no reason
	/// for it, so the variable refuses the word and says why.
	///
	/// @return the device, or `None` when this build has no `noop` feature -
	/// which a test skips on, the same way it skips on a machine with no GPU
	pub fn headless() -> Result<Option<Self>> { Self::open(Backends::NOOP, None) }

	/// The device everything built on this draws with.
	#[must_use]
	pub const fn device(&self) -> &Device { &self.device }

	/// The queue every upload and every frame's work is submitted on.
	#[must_use]
	pub const fn queue(&self) -> &Queue { &self.queue }

	/// The split-sum table this device's scenes read their ambient term out of.
	pub(crate) fn split(&self) -> &Arc<Split> { &self.split }

	/// The adapter the device was made on, for configuring a surface against.
	pub(crate) const fn adapter(&self) -> &Adapter { &self.adapter }

	/// The instance the adapter came from, for making a surface on.
	pub(crate) const fn instance(&self) -> &Instance { &self.instance }

	/// The async half of [`open`](Self::open).
	async fn create(backends: Backends, present_to: Option<&Window>) -> Result<Option<Self>> {
		let instance = Instance::new(InstanceDescriptor {
			backends,
			flags: layers(),
			// **the one backend that has to be asked for twice**, and that is
			// wgpu's decision rather than this one: a stub that draws nothing
			// must not be reachable by a set of flags somebody widened, so it
			// stays out until an instance says the word. Which means the only
			// way here is [`headless`](Self::headless) - `DEFAULT` does not
			// hold it and `parse` refuses to read it.
			backend_options: BackendOptions {
				noop: NoopBackendOptions {
					enable: backends.contains(Backends::NOOP),
					..NoopBackendOptions::default()
				},
				..BackendOptions::default()
			},
			..InstanceDescriptor::new_without_display_handle()
		});

		let surface = match present_to {
			| Some(window) => Some(
				instance
					.create_surface(window)
					.map_err(|error| err!(Graphics("creating the surface: {error}")))?,
			),
			| None => None,
		};

		let adapter = match instance
			.request_adapter(&RequestAdapterOptions {
				power_preference: PowerPreference::HighPerformance,
				compatible_surface: surface.as_ref(),
				..Default::default()
			})
			.await
		{
			| Ok(adapter) => adapter,
			| Err(error) => {
				warn!(%error, ?backends, "no usable adapter");

				return Ok(None);
			},
		};

		// **at `info`, and once a process.** Which graphics API a machine ended
		// up drawing with is the first thing anybody asks about a picture that
		// looks wrong or a frame that is slow, and it is not something a person
		// should have to raise the log level to find out afterwards. The driver
		// with it, because "Vulkan on this AMD card" and "Vulkan on that AMD
		// card six months later" are different answers.
		//
		// Two strings joined and not one, because the backends do not agree
		// which of them holds it: DX12 writes the version into `driver` and
		// leaves `driver_info` empty, Vulkan fills both. Either alone reads as
		// a machine with no driver on the other one.
		let info = adapter.get_info();
		let driver = [info.driver.as_str(), info.driver_info.as_str()]
			.into_iter()
			.filter(|part| !part.is_empty())
			.collect::<Vec<_>>()
			.join(" ");

		info!(adapter = %info.name, backend = %info.backend, driver, "drawing with");

		let (device, queue) = adapter
			.request_device(&DeviceDescriptor {
				label: Some("colby"),
				required_features: timing_features(&adapter),
				required_limits: Limits::default(),
				experimental_features: ExperimentalFeatures::disabled(),
				memory_hints: MemoryHints::Performance,
				trace: Trace::Off,
			})
			.await
			.map_err(|error| err!(Graphics("requesting a device: {error}")))?;

		let split = Arc::new(Split::new(&device, &queue)?);

		Ok(Some(Self { instance, adapter, device, queue, split }))
	}
}

/// Whether to ask each graphics API for its own debug layer.
///
/// wgpu's answer in a debug build is yes, and this says **no in a test
/// binary**. The layers are a machine's, not this project's: on Windows the
/// one that matters arrives with whatever Vulkan SDK is installed, and it is
/// loaded into the process by name. The one on this machine dereferences a
/// null pointer and takes the process with it - no panic, no failing test, the
/// harness simply stopping mid-list - when a test binary holds a device on a
/// second graphics API beside the one it draws with, which is what the test
/// that compares two of them does. **Measured 2026-09-12: twelve faults in
/// thirty runs of the engine's suite with the layers on and none in twenty
/// with them off**, everything else the same.
///
/// What that costs the gate is nothing it could see. The layers report through
/// `log`, a test binary installs no subscriber, and no test reads or fails on
/// what they say; what still runs is the whole of wgpu's own validation and
/// the whole of naga's, which is what every mistake in this tree has actually
/// been caught by. A real run keeps them, because there a person is reading
/// the console, and that is where they are worth having.
///
/// `cfg!(test)` is this crate's own test binary, which is the one that holds
/// two devices. The interface's and the editor's hold one each and have never
/// faulted; if either ever grows a second, this is the line to widen.
///
/// **And never wgpu's check of indirect draws, in any build.** The one draw
/// taken from a buffer here is the scene's lists through what the test for
/// what is behind something nearer kept, and the scene writes every command
/// itself: the index count of the mesh it binds, a first index, a base vertex
/// and a first instance of nought, and an instance count the copy can only
/// lower. What the check does is copy every command through a compute pass of
/// its own before the render pass, and it asserts that storage offsets align
/// to at least thirty-two bytes, which the stub device's do not: every
/// pipeline test on it panicked inside wgpu the day the draw went in. @ref
/// [`cover`](crate::cover).
///
/// @return the flags the instance is made with
fn layers() -> InstanceFlags {
	let asked =
		InstanceFlags::from_build_config().difference(InstanceFlags::VALIDATION_INDIRECT_CALL);

	if cfg!(test) {
		asked.difference(InstanceFlags::VALIDATION)
	} else {
		asked
	}
}

/// The one feature this asks for beyond what every device has.
///
/// **Asked for at every start, whether or not anybody measures anything.** A
/// feature can only be requested when the device is made, and the device is
/// made before there is a console to type at, before a project is open and
/// long before anybody has a frame they think is slow - so requesting it later
/// is not a thing that exists. What it costs when nothing measures is the flag
/// and nothing else: no query set is created until [`Timings::start`] is
/// called, and every pass writes the same `None` it did before.
///
/// Asked for only where the adapter has it, because a device asked for a
/// feature its adapter lacks is refused outright - and a machine that cannot
/// time a pass should still draw. That is bevy's arrangement too: it never
/// requests this by name, it checks `device.features()` and keeps a wall clock
/// where the answer is no (`diagnostic/internal.rs:54,248,523-545`).
///
/// @param adapter - the adapter the device is about to be made on
/// @return the feature when it is there, and nothing when it is not
fn timing_features(adapter: &Adapter) -> Features {
	adapter.features() & Features::TIMESTAMP_QUERY
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_default_is_the_first_tier_and_leaves_gl_out() {
		// the one thing the default promises beyond "what wgpu ranks first": a
		// machine with nothing but GL is a machine to hear about, not to draw
		// on quietly. The rest of the set is wgpu's to define.
		assert!(!DEFAULT.contains(Backends::GL), "GL is a secondary backend");
		assert!(DEFAULT.contains(Backends::VULKAN | Backends::DX12), "and these are not");
	}

	/// A build that has every API there is, so that the words alone decide.
	const EVERY: Backends = Backends::all();

	#[test]
	fn auto_and_nothing_at_all_are_both_the_default() {
		assert_eq!(choose(Some(AUTO), None, EVERY), (DEFAULT, By::Variable));
		assert_eq!(choose(Some(" Auto "), None, EVERY), (DEFAULT, By::Variable));
		assert_eq!(choose(None, None, EVERY), (DEFAULT, By::Default));
	}

	#[test]
	fn an_api_this_build_lacks_is_skipped_and_a_list_is_narrowed_to_what_is_here() {
		let vulkan = Backends::VULKAN;

		assert_eq!(
			choose(Some("dx12"), None, vulkan),
			(vulkan, By::Default),
			"a variable naming only what is not here falls to the default"
		);
		assert_eq!(
			choose(Some("dx12,vulkan"), None, vulkan),
			(vulkan, By::Variable),
			"a list keeps what is here"
		);
		assert_eq!(
			choose(Some("vulkan"), Some("dx12"), vulkan),
			(vulkan, By::Variable),
			"an override naming only what is not here falls to the variable"
		);
		assert_eq!(
			choose(Some(AUTO), None, vulkan),
			(vulkan, By::Variable),
			"and auto is the default narrowed to what is here"
		);
		assert_eq!(
			choose(None, None, vulkan),
			(vulkan, By::Default),
			"which the default itself is too"
		);
	}

	#[test]
	fn the_words_are_wgpus_own_and_a_list_is_a_union() {
		assert_eq!(parse("vulkan"), Read::Apis(Backends::VULKAN));
		assert_eq!(parse("DX12"), Read::Apis(Backends::DX12));
		assert_eq!(parse("dx12, vk"), Read::Apis(Backends::DX12 | Backends::VULKAN));
		assert_eq!(parse("metal"), Read::Apis(Backends::METAL));
		assert_eq!(parse("opengl"), Read::Apis(Backends::GL));
	}

	#[test]
	fn a_word_nobody_knows_refuses_the_whole_value() {
		assert_eq!(parse("dx11"), Read::Unknown);
		assert_eq!(
			parse("vulkan,dx11"),
			Read::Unknown,
			"one bad word is a bad value, not vulkan"
		);
		assert_eq!(parse(""), Read::Unknown);
		assert_eq!(
			choose(Some("dx11"), None, EVERY),
			(DEFAULT, By::Default),
			"a bad variable draws with the default"
		);
	}

	#[test]
	fn a_word_that_draws_nothing_is_refused_with_a_reason_of_its_own() {
		// the three refusals are three, and this is the pair that used to be
		// told apart from a typo by nothing: both are words wgpu reads, and
		// neither is a thing a person could look at.
		let Read::NotDrawn(browser) = parse("webgpu") else {
			panic!("a browser is not an API this draws with");
		};
		let Read::NotDrawn(nothing) = parse("noop") else {
			panic!("and nothing is not one either");
		};

		assert_ne!(browser, nothing, "and the two say different things");
		assert_eq!(
			parse("vulkan,noop"),
			Read::NotDrawn(nothing),
			"a list stops on the first word it cannot draw with, as it does on a typo"
		);
		assert_eq!(
			choose(Some("noop"), None, EVERY),
			(DEFAULT, By::Default),
			"and the run draws with the default rather than with nothing"
		);
	}

	#[test]
	fn a_build_without_an_api_says_so_and_is_not_a_typo() {
		// what `taken` exists for: `dx12` in a `settings.cfg` carried over
		// from Windows and `dx11` typed by hand both end at the default, and
		// used to say the same sentence on the way. The set survives the
		// parse in one case and not the other, which is what the two lines
		// are drawn from.
		assert_eq!(parse("dx12"), Read::Apis(Backends::DX12), "the word is known");
		assert_eq!(
			taken(BACKEND, "dx12", Backends::VULKAN),
			None,
			"and this build still has no such API"
		);
		assert_eq!(
			taken(BACKEND, "dx12", Backends::DX12 | Backends::VULKAN),
			Some(Backends::DX12),
			"where it does, the same value is taken"
		);
	}

	#[test]
	fn the_environment_wins_and_a_bad_one_is_skipped_rather_than_taken() {
		assert_eq!(
			choose(Some("dx12"), Some("vulkan"), EVERY),
			(Backends::VULKAN, By::Environment)
		);
		assert_eq!(choose(None, Some("dx12"), EVERY), (Backends::DX12, By::Environment));
		assert_eq!(
			choose(Some("dx12"), Some("dx11"), EVERY),
			(Backends::DX12, By::Variable),
			"a bad override falls through to the variable"
		);
		assert_eq!(
			choose(Some("dx12"), Some("  "), EVERY),
			(Backends::DX12, By::Variable),
			"an empty override is no override"
		);
	}
}
