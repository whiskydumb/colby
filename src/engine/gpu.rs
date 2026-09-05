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

use std::env;

use colby_core::{Result, debug, err, warn};
use wgpu::{
	Adapter, Backends, Device, DeviceDescriptor, ExperimentalFeatures, Features, Instance,
	InstanceDescriptor, Limits, MemoryHints, PowerPreference, Queue, RequestAdapterOptions,
	Trace,
};
use winit::window::Window;

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
	if let Some(text) = forced.filter(|text| !text.trim().is_empty()) {
		match parse(text).and_then(|set| here(set, have)) {
			| Some(set) => return (set, By::Environment),
			| None =>
				warn!(text, "{BACKEND_VAR} names no graphics API this build has; ignoring it"),
		}
	}

	if let Some(text) = asked {
		match parse(text).and_then(|set| here(set, have)) {
			| Some(set) => return (set, By::Variable),
			| None => warn!(text, "{BACKEND} names no graphics API this build has; using {AUTO}"),
		}
	}

	(DEFAULT & have, By::Default)
}

/// The part of a set this build has, or nothing when none of it is.
fn here(set: Backends, have: Backends) -> Option<Backends> {
	let kept = set & have;

	(!kept.is_empty()).then_some(kept)
}

/// Reads a value of [`BACKEND`]: [`AUTO`], or a comma list of wgpu's words.
///
/// wgpu's own reader takes the same words, warns through a logger this process
/// does not listen to, and hands back an empty set for a value made of nothing
/// it knows; this one refuses the whole value on the first word it does not
/// know, so that a list with a misspelling in it is a warning rather than a
/// quiet subset. `webgpu` and `noop` are wgpu's words too and are refused on
/// purpose: one is a browser and the other draws nothing.
///
/// @param text - the value
/// @return the set, or `None` for a word nothing here knows
fn parse(text: &str) -> Option<Backends> {
	if text.trim().eq_ignore_ascii_case(AUTO) {
		return Some(DEFAULT);
	}

	let mut set = Backends::empty();
	for word in text.split(',') {
		set |= match word.trim().to_ascii_lowercase().as_str() {
			| "vulkan" | "vk" => Backends::VULKAN,
			| "dx12" | "d3d12" => Backends::DX12,
			| "metal" | "mtl" => Backends::METAL,
			| "gl" | "gles" | "opengl" => Backends::GL,
			| _ => return None,
		};
	}

	Some(set)
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

	/// The device everything built on this draws with.
	#[must_use]
	pub const fn device(&self) -> &Device { &self.device }

	/// The queue every upload and every frame's work is submitted on.
	#[must_use]
	pub const fn queue(&self) -> &Queue { &self.queue }

	/// The adapter the device was made on, for configuring a surface against.
	pub(crate) const fn adapter(&self) -> &Adapter { &self.adapter }

	/// The instance the adapter came from, for making a surface on.
	pub(crate) const fn instance(&self) -> &Instance { &self.instance }

	/// The async half of [`open`](Self::open).
	async fn create(backends: Backends, present_to: Option<&Window>) -> Result<Option<Self>> {
		let instance = Instance::new(InstanceDescriptor {
			backends,
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

		let info = adapter.get_info();
		debug!(adapter = %info.name, backend = ?info.backend, "adapter selected");

		let (device, queue) = adapter
			.request_device(&DeviceDescriptor {
				label: Some("colby"),
				required_features: Features::empty(),
				required_limits: Limits::default(),
				experimental_features: ExperimentalFeatures::disabled(),
				memory_hints: MemoryHints::Performance,
				trace: Trace::Off,
			})
			.await
			.map_err(|error| err!(Graphics("requesting a device: {error}")))?;

		Ok(Some(Self { instance, adapter, device, queue }))
	}
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
		assert_eq!(parse("vulkan"), Some(Backends::VULKAN));
		assert_eq!(parse("DX12"), Some(Backends::DX12));
		assert_eq!(parse("dx12, vk"), Some(Backends::DX12 | Backends::VULKAN));
		assert_eq!(parse("metal"), Some(Backends::METAL));
		assert_eq!(parse("opengl"), Some(Backends::GL));
	}

	#[test]
	fn a_word_nobody_knows_refuses_the_whole_value() {
		assert_eq!(parse("dx11"), None);
		assert_eq!(parse("vulkan,dx11"), None, "one bad word is a bad value, not vulkan");
		assert_eq!(parse(""), None);
		assert_eq!(parse("webgpu"), None, "a browser is not an API this draws with");
		assert_eq!(parse("noop"), None, "and nothing is not one either");
		assert_eq!(
			choose(Some("dx11"), None, EVERY),
			(DEFAULT, By::Default),
			"a bad variable draws with the default"
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
