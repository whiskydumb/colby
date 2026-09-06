//! Pictures of what is in the asset tree: a mesh drawn once into a small
//! target, a texture shrunk to the same size, each kept as a png beside the
//! compiled tree.
//!
//! **One scene, one world, one entity, reused.** A [`Capture`] is a table of
//! pipelines and everything it has uploaded, and making one per picture
//! would compile the shaders per picture; a [`World`] reused across meshes
//! hands every mesh the same registry slot, which the scene caches by
//! revision - so the mesh is rewritten under one name and the revision moves,
//! which is the engine's own reload path and the only one that works. **One
//! picture a frame at most**, so that a tree of a hundred meshes costs a
//! hundred frames rather than one long one.
//!
//! The cache is `.colby/thumbs/<name>.png`, fresh while it is newer than the
//! compiled file it is a picture of. The field caches by mtime or by md5;
//! this is the mtime half, and a stale picture is drawn again the next time
//! it is asked for.

use std::{
	collections::HashMap,
	fs,
	path::{Path, PathBuf},
};

use colby_asset::{MeshFile, TextureFile, compile::Kind, png};
use colby_core::{
	Err, Result,
	abi::{
		Camera, EntityId, Renderable, World,
		texture::{Texel, TextureData},
	},
	debug,
	glam::Vec3,
	warn,
};
use colby_engine::{Capture, Gpu, Image};
use egui::{ColorImage, Context, TextureHandle, TextureId, TextureOptions};

use crate::catalog::{self, Entry, State};

/// How wide and tall a picture is, in pixels.
pub(crate) const SIZE: u32 = 64;

/// The name the one mesh is registered under in the thumbnail world, again
/// and again.
const MESH_NAME: &str = "thumbnail";

/// From which way a mesh is looked at: a little above, from the front and
/// the side, so that three faces of a box show.
const EYE: Vec3 = Vec3::new(1.0, 0.8, 1.3);

/// The pictures, made once each and kept.
pub(crate) struct Thumbs {
	/// Where the pngs go.
	dir: PathBuf,

	/// Every picture handed to egui so far, by asset name.
	loaded: HashMap<String, TextureHandle>,

	/// Names whose picture could not be made, so that a broken file is tried
	/// once rather than every frame.
	refused: Vec<String>,

	/// The world a mesh is drawn in: one entity, whichever mesh is asked for.
	world: Box<World>,

	/// The entity.
	entity: EntityId,

	/// The target, made the first time a mesh is drawn.
	capture: Option<Capture>,
}

impl Thumbs {
	/// A cache under a directory, with nothing loaded yet.
	///
	/// @param dir - where the pngs go, made when the first one is written
	pub(crate) fn new(dir: PathBuf) -> Self {
		let mut world = Box::new(World::new());
		world.clear = Vec3::splat(0.14);
		let entity = world.entities.spawn();

		Self {
			dir,
			loaded: HashMap::new(),
			refused: Vec::new(),
			world,
			entity,
			capture: None,
		}
	}

	/// The picture for an entry, if there is one or can be one.
	///
	/// A picture already handed to egui is answered at once; one on disk
	/// and fresh is read; one missing or stale is made, and only when
	/// nothing else was made this frame.
	///
	/// @param context - egui, to hand a new picture to
	/// @param gpu - the device to draw a mesh with, if there is one
	/// @param entry - what the picture is of
	/// @param made - whether a picture was made this frame already; set when
	/// this call makes one
	pub(crate) fn get(
		&mut self,
		context: &Context,
		gpu: Option<&Gpu>,
		entry: &Entry,
		made: &mut bool,
	) -> Option<TextureId> {
		if !matches!(entry.kind, Kind::Mesh | Kind::Texture) || entry.state == State::Uncompiled {
			return None;
		}

		if let Some(handle) = self.loaded.get(&entry.name) {
			return Some(handle.id());
		}

		if *made || self.refused.contains(&entry.name) {
			return None;
		}

		*made = true;

		let image = match self.picture(gpu, entry) {
			| Ok(image) => image,
			| Err(error) => {
				warn!(name = entry.name, %error, "no thumbnail");
				self.refused.push(entry.name.clone());

				return None;
			},
		};

		let Ok(width) = usize::try_from(image.width) else {
			return None;
		};
		let Ok(height) = usize::try_from(image.height) else {
			return None;
		};

		let handle = context.load_texture(
			format!("thumb {}", entry.name),
			ColorImage::from_rgba_unmultiplied([width, height], &image.pixels),
			TextureOptions::LINEAR,
		);
		let id = handle.id();
		self.loaded.insert(entry.name.clone(), handle);

		Some(id)
	}

	/// The picture, from the cache when it is fresh and drawn otherwise.
	fn picture(&mut self, gpu: Option<&Gpu>, entry: &Entry) -> Result<Image> {
		let path = self.dir.join(&entry.name).with_extension("png");

		if fresh(&path, &entry.output)
			&& let Ok(image) = read(&path)
		{
			return Ok(image);
		}

		let image = match entry.kind {
			| Kind::Mesh => self.render(gpu, &entry.output)?,
			| Kind::Texture => shrink(&entry.output)?,
			| _ =>
				return Err!(Asset("nothing draws a picture of a {}", catalog::word(entry.kind))),
		};

		// the cache is a convenience: a picture that cannot be kept is still
		// a picture
		if let Err(error) = keep(&path, &image) {
			debug!(path = %path.display(), %error, "the thumbnail was not kept");
		}

		Ok(image)
	}

	/// Draws a mesh into the target.
	fn render(&mut self, gpu: Option<&Gpu>, output: &Path) -> Result<Image> {
		let Some(gpu) = gpu else {
			return Err!(Graphics("no device to draw a thumbnail with"));
		};

		let data = MeshFile::open(output)?.to_mesh_data();
		let (min, max) = data.bounds();
		let mesh = self.world.meshes.insert(MESH_NAME, data);

		self.world
			.entities
			.set_renderable(self.entity, Renderable::new(mesh, Vec3::splat(0.85)));
		frame(&mut self.world.camera, min, max);

		if self.capture.is_none() {
			self.capture = Some(Capture::new(gpu, SIZE, SIZE)?);
		}

		let Some(capture) = self.capture.as_mut() else {
			return Err!(Graphics("no target to draw a thumbnail into"));
		};

		capture.shoot(&mut self.world)
	}
}

/// Points the camera at a box so that the whole of it is in the picture.
///
/// @param camera - the camera to move
/// @param min - the box's near corner
/// @param max - its far corner
pub(crate) fn frame(camera: &mut Camera, min: Vec3, max: Vec3) {
	let center = (min + max) * 0.5;
	let radius = ((max - min).length() * 0.5).max(1.0e-3);
	// a little further than the sphere around the box needs, so that a
	// corner does not touch the edge
	let distance = radius / (camera.fov_y * 0.5).tan() * 1.15;

	camera.target = center;
	camera.position = center + EYE.normalize() * distance;
	camera.near = (distance - radius * 2.0).max(0.01);
	camera.far = distance + radius * 4.0;
}

/// A texture, shrunk to the picture's size by taking one texel per pixel.
fn shrink(output: &Path) -> Result<Image> {
	let data = TextureFile::open(output)?.to_texture_data();

	shrunk(&data)
}

/// The same, given the texels.
pub(crate) fn shrunk(data: &TextureData) -> Result<Image> {
	let Some(level) = data.levels.first() else {
		return Err!(Asset("a texture with no levels"));
	};
	let (width, height) = (data.width.max(1), data.height.max(1));
	let stride = usize::try_from(width).unwrap_or(1) * 4;

	if level.len() < stride * usize::try_from(height).unwrap_or(1) {
		return Err!(Asset("a texture shorter than its size says"));
	}

	let mut pixels = Vec::with_capacity(usize::try_from(SIZE * SIZE * 4).unwrap_or(0));

	for y in 0..SIZE {
		let row = usize::try_from(y * height / SIZE).unwrap_or(0) * stride;

		for x in 0..SIZE {
			let at = row + usize::try_from(x * width / SIZE).unwrap_or(0) * 4;

			pixels.extend_from_slice(level.get(at..at + 4).unwrap_or(&[0, 0, 0, 255]));
		}
	}

	Ok(Image { width: SIZE, height: SIZE, pixels })
}

/// Whether a picture on disk is newer than the compiled file it is of.
pub(crate) fn fresh(thumb: &Path, output: &Path) -> bool {
	match (catalog::modified(thumb), catalog::modified(output)) {
		| (Some(kept), Some(made)) => kept >= made,
		| _ => false,
	}
}

/// A picture read back from the cache.
fn read(path: &Path) -> Result<Image> {
	let data = png::import_file(path, Texel::Rgba8Srgb)?;
	let Some(level) = data.levels.first() else {
		return Err!(Asset("a png with nothing in it"));
	};

	Ok(Image {
		width: data.width,
		height: data.height,
		pixels: level.clone(),
	})
}

/// Writes a picture into the cache, making the directory.
fn keep(path: &Path, image: &Image) -> Result {
	if let Some(dir) = path.parent() {
		fs::create_dir_all(dir)?;
	}

	image.write_png(path)
}

#[cfg(test)]
mod tests {
	use std::{
		env,
		fs::File,
		time::{Duration, SystemTime},
	};

	use colby_core::abi::mesh;
	use colby_engine::gpu;

	use super::*;

	#[test]
	fn the_camera_is_moved_back_far_enough_to_see_the_whole_box() {
		let mut camera = Camera::DEFAULT;

		frame(&mut camera, Vec3::splat(-1.0), Vec3::splat(1.0));

		let radius = Vec3::splat(1.0).length();
		let distance = (camera.position - camera.target).length();
		assert_eq!(camera.target, Vec3::ZERO, "looking at the middle");
		assert!(distance > radius, "from outside the sphere around it: {distance} > {radius}");
		assert!(camera.near < distance - radius, "with the near plane in front of it");
		assert!(camera.far > distance + radius, "and the far plane behind it");
	}

	#[test]
	fn a_texture_is_shrunk_by_taking_one_texel_per_pixel() {
		// two by two: red, green over blue, white; every quarter of the
		// picture is one of them
		let data = TextureData {
			width: 2,
			height: 2,
			texel: Texel::Rgba8Srgb,
			levels: vec![vec![
				255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
			]],
		};

		let image = shrunk(&data).expect("it shrinks");

		assert_eq!((image.width, image.height), (SIZE, SIZE));
		assert_eq!(image.pixel(0, 0), [255, 0, 0, 255], "top left is red");
		assert_eq!(image.pixel(SIZE - 1, 0), [0, 255, 0, 255], "top right is green");
		assert_eq!(image.pixel(0, SIZE - 1), [0, 0, 255, 255], "bottom left is blue");
		assert_eq!(
			image.pixel(SIZE - 1, SIZE - 1),
			[255, 255, 255, 255],
			"bottom right is white"
		);
	}

	#[test]
	fn a_picture_is_fresh_while_it_is_newer_than_what_it_is_of() {
		let dir = env::temp_dir().join("colby_thumbs_fresh");
		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(&dir).expect("a directory to work in");
		let output = dir.join("thing.cmesh");
		let thumb = dir.join("thing.png");
		fs::write(&output, b"mesh").expect("the output");

		assert!(!fresh(&thumb, &output), "no picture yet");

		fs::write(&thumb, b"png").expect("the picture");
		assert!(fresh(&thumb, &output), "a picture made after the output");

		File::options()
			.write(true)
			.open(&thumb)
			.expect("the picture opens")
			.set_modified(SystemTime::now() - Duration::from_mins(1))
			.expect("and takes a time");
		assert!(!fresh(&thumb, &output), "and stale once the output is newer");
	}

	#[test]
	fn a_mesh_is_drawn_into_the_picture_when_there_is_a_device() {
		let Some(gpu) = Gpu::open(gpu::backends(None), None).expect("the adapter query works")
		else {
			return;
		};

		let dir = env::temp_dir().join("colby_thumbs_render");
		drop(fs::remove_dir_all(&dir));
		let mut thumbs = Thumbs::new(dir.join("thumbs"));

		// a compiled cube, the way the asset loop would have written it
		let output = dir.join("cube.cmesh");
		fs::create_dir_all(&dir).expect("a directory to work in");
		fs::write(&output, colby_asset::format::encode(&mesh::cube()).expect("a cube encodes"))
			.expect("the file");

		let image = thumbs
			.render(Some(&gpu), &output)
			.expect("the cube draws");

		assert_eq!((image.width, image.height), (SIZE, SIZE));
		let middle = image.pixel(SIZE / 2, SIZE / 2);
		let corner = image.pixel(0, 0);
		assert_ne!(middle, corner, "the cube is in the middle and the clear color in the corner");

		// and again, with a different mesh under the same name: the picture
		// changes, which is the registry's revision moving
		let output = dir.join("sphere.cmesh");
		fs::write(
			&output,
			colby_asset::format::encode(&mesh::sphere()).expect("a sphere encodes"),
		)
		.expect("the file");
		let again = thumbs
			.render(Some(&gpu), &output)
			.expect("the sphere draws");
		assert_ne!(again.pixels, image.pixels, "a sphere is not a cube");
	}
}
