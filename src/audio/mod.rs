//! The host's mixer: voices and samples in, a buffer of stereo floats out.
//!
//! What this crate owns is what is *derived* from the voice table, and nothing
//! else. The authoritative state of every playing sound - what it is, where it
//! is, how loud, how far along - lives in `World::audio`, plain data in
//! `colby_core` that the editor could show and a console command can write.
//! This crate owns a copy of the samples and a playhead per slot, both of which
//! can be thrown away and rebuilt from that table. The same arrangement the
//! solver has, and for the same reason.
//!
//! ```text
//!   World::audio ---[Snapshot::take]--> Snapshot   (finished gains, once a step)
//!   World::sounds --[Bank::sync]------> Bank       (the samples, when one changes)
//!                   Mixer::render(bank, snapshot, out)  -> [f32] stereo
//! ```
//!
//! **Two clocks, and which is which is the design.** The simulation's playhead
//! moves by `dt` a step and is the one that decides when a voice *ends*; a
//! one-second sound therefore lasts the same number of steps on every machine
//! and a screenshot stays reproducible. The mixer's own playhead moves one
//! output frame at a time and decides which sample is *heard*. They are allowed
//! to disagree by a block, which is what stops every block starting with a
//! jump; past that the mixer follows, which is what makes a game writing
//! `Voice::head` a seek rather than a suggestion.
//!
//! **There is no device in this crate.** [`Mixer::render`] fills a buffer
//! somebody else asked for, which is what makes all of it testable by running a
//! test: what a driver does with the buffer is one commit further out, and it
//! is the only part of the subsystem that cannot be checked without hardware.
//!
//! **Everything crossing to the mixer is already a number.** Distance,
//! direction, category and volume are worked out on the simulation's side and
//! arrive as two gains. Nothing on the far side knows where the listener is,
//! which is deliberate: a value computed on somebody else's thread at somebody
//! else's cadence is a value that cannot be unit-tested.

/// The variable that scales everything.
pub const MASTER: &str = "snd.volume";

/// The variable that scales sounds in the world.
pub const EFFECTS: &str = "snd.effects";

/// The variable that scales music.
pub const MUSIC: &str = "snd.music";

/// The variable that scales clicks and beeps.
pub const INTERFACE: &str = "snd.interface";

pub mod bank;
pub mod device;
pub mod mix;
pub mod pan;
#[cfg(test)]
mod real;
pub mod snapshot;

pub use self::{
	bank::{Bank, Recording},
	device::Device,
	mix::{CHANNELS, Mixer, SLEW_SECONDS},
	snapshot::{Playing, Snapshot},
};
