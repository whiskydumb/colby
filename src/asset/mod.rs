//! The asset pipeline: source formats in, colby's own binary formats out.
//!
//! Two halves that never meet at runtime. The **importer** ([`obj`]) reads a
//! format a modeling tool writes and is only ever run by the compiler. The
//! **runtime format** ([`format`]) is what the engine loads, and it is designed
//! to be read rather than parsed: a versioned header, little-endian, then
//! `#[repr(C)]` blocks that are already the layout the GPU wants. Loading a
//! mesh is a read into an aligned buffer and two `bytemuck` casts, with no
//! decoding step in between.
//!
//! The compiler ([`compile`]) is the seam. It walks a source tree, skips
//! anything whose output is newer than its input, and writes the results into a
//! mirrored output tree. It is a library first and a command second: `just
//! assets` runs it through `colby_assetc`, and the runner calls the same
//! function in-process every quarter second so that editing a mesh reloads it
//! without a restart.
//!
//! ```text
//! assets/meshes/crystal.obj  --[obj::import]-->  MeshData
//!                            --[format::encode]-->  target/assets/meshes/crystal.cmesh
//!                            --[format::MeshFile]-->  World::meshes["meshes/crystal"]
//! ```
//!
//! Nothing here depends on wgpu, so the compiler builds and runs without a GPU.

pub mod anim;
pub mod bytes;
pub mod compile;
pub mod css;
pub mod document;
pub mod font;
pub mod format;
pub mod gltf;
pub mod html;
pub mod jpeg;
pub mod json;
pub mod level;
pub mod lua;
pub mod model;
pub mod obj;
pub mod png;
pub mod project;
pub mod scene;
pub mod script;
pub mod sdf;
pub mod skeleton;
pub mod sound;
pub mod texture;
pub mod ttf;
pub mod wav;

pub use self::{
	bytes::AlignedBytes,
	compile::{Report, compile_dir, compile_file, output_root, source_root},
	format::{FORMAT_VERSION, MAGIC, MeshFile, MeshHeader, encode},
	project::Project,
	scene::{SceneFile, SceneHeader},
	texture::{TextureFile, TextureHeader},
};
