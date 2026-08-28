//! The parts of colby that a change to forces a full restart.
//!
//! Window and renderer live here. The split from `colby_game` is the one from
//! the architecture notes: touch the engine and the process has to come back
//! up, touch the game and it does not.
//!
//! Geometry is not here. [`MeshData`] and the built-in primitives live in
//! `colby_core::abi::mesh`, next to the [`MeshId`](colby_core::abi::MeshId)
//! that addresses them and the registry the world holds - the engine is a
//! consumer of that table like everything else, and reading a mesh off disk is
//! `colby_asset`'s job. Nor is timing: the step length is something a game
//! reads, so the loop's clock lives in [`colby_core::time`] with the rest of
//! the contract rather than beside the thing it happens to pace.

pub mod capture;
pub mod image;
mod lines;
pub mod overlay;
pub mod renderer;
pub mod scene;
pub mod shader;

pub use ::wgpu;
pub use ::winit;
pub use colby_core::abi::mesh::{MeshData, MeshVertex};

pub use self::{
	capture::Capture, image::Image, overlay::Overlay, renderer::Renderer, scene::Scene,
	shader::Shader,
};
