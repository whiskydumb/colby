//! Pictures of what is in the asset tree: a mesh drawn once into a small
//! target, a material worn by a ball, a texture shrunk to the same size, each
//! kept as a png beside the compiled tree.
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

use colby_asset::{
	MeshFile, TextureFile, compile::Kind, material::MaterialFile, model::ModelFile, png,
};
use colby_core::{
	Err, Result,
	abi::{
		Camera, EntityId, MaterialId, MeshId, Renderable, World,
		texture::{self, Texel, TextureData},
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

/// The same for the one material.
const MATERIAL_NAME: &str = "thumbnail";

/// From which way a mesh is looked at: a little above, from the front and
/// the side, so that three faces of a box show.
const EYE: Vec3 = Vec3::new(1.0, 0.8, 1.3);

/// The pictures, made once each and kept.
pub(crate) struct Thumbs {
	/// Where the pngs go.
	dir: PathBuf,

	/// The compiled tree, for the pictures a material names.
	///
	/// A material is the one asset that points at *another* one, and a
	/// thumbnail world has no loader watching a directory - so its pictures
	/// are opened by hand, from here.
	compiled: PathBuf,

	/// Every picture handed to egui so far, by asset name.
	loaded: HashMap<String, TextureHandle>,

	/// Names whose picture could not be made, so that a broken file is tried
	/// once rather than every frame.
	refused: Vec<String>,

	/// The world a mesh is drawn in: one entity, whichever mesh is asked for.
	world: Box<World>,

	/// The entity a mesh or a material is drawn on.
	entity: EntityId,

	/// One more entity per piece, for a model, grown as models get bigger and
	/// never shrunk.
	///
	/// A pool rather than a spawn and a despawn per picture, for the reason
	/// the world itself is reused: an entity table that churns hands every
	/// picture different slots, and a spare is free once it draws nothing.
	pieces: Vec<EntityId>,

	/// The target, made the first time a mesh is drawn.
	capture: Option<Capture>,
}

impl Thumbs {
	/// A cache under a directory, with nothing loaded yet.
	///
	/// @param dir - where the pngs go, made when the first one is written
	/// @param compiled - the compiled tree, for a material's own pictures
	pub(crate) fn new(dir: PathBuf, compiled: PathBuf) -> Self {
		let mut world = Box::new(World::new());
		world.clear = Vec3::splat(0.14);
		let entity = world.entities.spawn();

		Self {
			dir,
			compiled,
			loaded: HashMap::new(),
			refused: Vec::new(),
			world,
			entity,
			pieces: Vec::new(),
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
		if !matches!(entry.kind, Kind::Mesh | Kind::Texture | Kind::Material | Kind::Model)
			|| entry.state == State::Uncompiled
		{
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
			| Kind::Material => self.wearing(gpu, &entry.output, &entry.name)?,
			| Kind::Model => self.standing(gpu, &entry.output)?,
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

	/// A ball wearing a material, drawn once.
	///
	/// **A ball rather than the cube a mesh gets**, and it is the field's
	/// shape: Fyrox's material panel has a preview sphere in it, and a sphere
	/// is the one surface that shows a roughness and a metal at every
	/// angle at once - a flat face shows one highlight or none.
	///
	/// The pictures it may name are opened from the compiled tree by hand.
	/// A material is the only asset that points at another, and a thumbnail
	/// world has no loader watching a directory; one that cannot be read is
	/// left out rather than refusing the picture, because a material with a
	/// missing texture is still a material worth looking at.
	///
	/// @param gpu - the device, or nothing
	/// @param output - the `.cmat` on disk
	/// @param name - its asset name, which the file does not carry
	fn wearing(&mut self, gpu: Option<&Gpu>, output: &Path, name: &str) -> Result<Image> {
		let Some(gpu) = gpu else {
			return Err!(Graphics("no device to draw a thumbnail with"));
		};

		let described = MaterialFile::open(output)?.to_material(name);
		let live = described.live(|picture| self.picture_named(picture));
		let material = self.world.materials.insert(MATERIAL_NAME, live);

		self.world
			.entities
			.set_renderable(self.entity, Renderable::of(MeshId::SPHERE, material, Vec3::ONE));

		let (min, max) = self
			.world
			.meshes
			.get(MeshId::SPHERE)
			.map_or((Vec3::ZERO, Vec3::ZERO), |entry| entry.value().bounds());
		frame(&mut self.world.camera, min, max);

		if self.capture.is_none() {
			self.capture = Some(Capture::new(gpu, SIZE, SIZE)?);
		}

		let Some(capture) = self.capture.as_mut() else {
			return Err!(Graphics("no target to draw a thumbnail into"));
		};

		capture.shoot(&mut self.world)
	}

	/// A whole model, every piece of it standing where the model says.
	///
	/// **The one picture that is more than one entity**, which is what the
	/// pool is for: a model is a list of placements and drawing one of them
	/// would be a picture of a lamp's shade rather than of a lamp. The
	/// entity a mesh and a material use draws nothing while this runs.
	///
	/// Its meshes and its materials are opened from the compiled tree by
	/// hand, the way a material's pictures are, and for the same reason:
	/// a thumbnail world has no loader watching a directory. A piece whose
	/// mesh is not there is left out rather than refusing the picture.
	///
	/// @param gpu - the device, or nothing
	/// @param output - the `.cmodel` on disk
	fn standing(&mut self, gpu: Option<&Gpu>, output: &Path) -> Result<Image> {
		let Some(gpu) = gpu else {
			return Err!(Graphics("no device to draw a thumbnail with"));
		};

		let data = ModelFile::open(output)?.to_model_data();

		// the single entity steps aside: a model draws on the pool
		self.world
			.entities
			.set_renderable(self.entity, Renderable::NOTHING);

		while self.pieces.len() < data.placements.len() {
			let made = self.world.entities.spawn();

			self.pieces.push(made);
		}

		let mut low = Vec3::splat(f32::INFINITY);
		let mut high = Vec3::splat(f32::NEG_INFINITY);

		// by index rather than over the pool, because the body reaches back
		// into `self` to open a mesh
		for slot in 0..self.pieces.len() {
			let id = self.pieces[slot];
			let Some(placement) = data.placements.get(slot) else {
				self.world
					.entities
					.set_renderable(id, Renderable::NOTHING);

				continue;
			};
			let mesh = self.mesh_named(&placement.mesh);
			let material = self.coat_named(&placement.material, &data);

			self.world
				.entities
				.set_renderable(id, Renderable::of(mesh, material, Vec3::ONE));
			self.world
				.entities
				.set_placed(id, placement.transform);

			let (min, max) = self
				.world
				.meshes
				.get(mesh)
				.map_or((Vec3::ZERO, Vec3::ZERO), |entry| entry.value().bounds());

			// the eight corners through the placement, not the two: a box
			// turned by forty-five degrees has corners its own two do not
			// reach, and a camera framed on those would cut the model off
			for corner in corners(min, max) {
				let at = placement
					.transform
					.matrix()
					.transform_point3(corner);

				low = low.min(at);
				high = high.max(at);
			}
		}

		if !low.is_finite() || !high.is_finite() {
			return Err!(Asset("this model stands nothing there is geometry for"));
		}

		// a placement is written straight in rather than stepped toward, so
		// the one frame this draws is the frame it is meant to be
		self.world.entities.snap_all();
		self.world.entities.settle();
		frame(&mut self.world.camera, low, high);

		if self.capture.is_none() {
			self.capture = Some(Capture::new(gpu, SIZE, SIZE)?);
		}

		let Some(capture) = self.capture.as_mut() else {
			return Err!(Graphics("no target to draw a thumbnail into"));
		};

		capture.shoot(&mut self.world)
	}

	/// One of a model's meshes, put in the thumbnail world.
	///
	/// @param name - the mesh's asset name
	/// @return its handle, or [`MeshId::NONE`] for one that is not on disk
	fn mesh_named(&mut self, name: &str) -> MeshId {
		let held = self.world.meshes.find(name);

		if held.is_some() {
			return held;
		}

		let path = self
			.compiled
			.join(name)
			.with_extension(colby_asset::format::EXTENSION);

		match MeshFile::open(&path) {
			| Ok(file) => self
				.world
				.meshes
				.insert(name, file.to_mesh_data()),
			| Err(error) => {
				debug!(name, %error, "a model's mesh is not on disk");

				MeshId::NONE
			},
		}
	}

	/// One of a model's materials, put in the thumbnail world.
	///
	/// **Two places to look, and the order is the point.** A material the
	/// model declares lives inside the `.cmodel` and has no file of its own;
	/// one a sidecar sent somewhere else is a `.cmat` in the compiled tree.
	/// Looking in the model first and on disk second is what makes a remapped
	/// model draw wearing the thing somebody wrote. @ref
	/// `colby_asset::import`.
	///
	/// @param name - the material's asset name, or empty for the built-in one
	/// @param data - the model, for the surfaces it declares itself
	fn coat_named(&mut self, name: &str, data: &colby_asset::model::ModelData) -> MaterialId {
		if name.is_empty() {
			return MaterialId::DEFAULT;
		}

		let held = self.world.materials.find(name);

		if held.is_some() {
			return held;
		}

		let described = data
			.materials
			.iter()
			.find(|surface| surface.name == name)
			.cloned()
			.or_else(|| {
				let path = self
					.compiled
					.join(name)
					.with_extension(colby_asset::material::EXTENSION);

				MaterialFile::open(&path)
					.map(|file| file.to_material(name))
					.map_err(|error| debug!(name, %error, "a model's material is not on disk"))
					.ok()
			});
		let Some(described) = described else {
			return MaterialId::DEFAULT;
		};
		let live = described.live(|picture| self.picture_named(picture));

		self.world.materials.insert(name, live)
	}

	/// One of a material's pictures, put in the thumbnail world.
	///
	/// @param name - the texture's asset name, or empty for none
	/// @return its handle, or [`texture::TextureId::NONE`] for a name that is
	/// empty or that nothing on disk answers to
	///
	/// @note: the *engine's* handle, not egui's - this module speaks both, and
	/// the two are one word apart.
	fn picture_named(&mut self, name: &str) -> texture::TextureId {
		if name.is_empty() {
			return texture::TextureId::NONE;
		}

		// already in the world, from a material drawn before this one: a
		// thumbnail world is reused across pictures, which is the whole
		// reason it is a field.
		let held = self.world.textures.find(name);
		if held.is_some() {
			return held;
		}

		let path = self
			.compiled
			.join(name)
			.with_extension(colby_asset::texture::EXTENSION);

		match TextureFile::open(&path) {
			| Ok(file) => self
				.world
				.textures
				.insert(name, file.to_texture_data()),
			| Err(error) => {
				debug!(name, %error, "a material's picture is not on disk");

				texture::TextureId::NONE
			},
		}
	}
}

/// The eight corners of a box.
fn corners(min: Vec3, max: Vec3) -> [Vec3; 8] {
	[
		Vec3::new(min.x, min.y, min.z),
		Vec3::new(max.x, min.y, min.z),
		Vec3::new(min.x, max.y, min.z),
		Vec3::new(max.x, max.y, min.z),
		Vec3::new(min.x, min.y, max.z),
		Vec3::new(max.x, min.y, max.z),
		Vec3::new(min.x, max.y, max.z),
		Vec3::new(max.x, max.y, max.z),
	]
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
///
/// **A cube shows its first face, and a value above one is held down to it.**
/// The browser wants a square of eight-bit color, and an environment is six
/// faces whose channels go up to the tens of thousands, so something has to be
/// chosen rather than read. What is chosen is the face a direction of `+x`
/// lands on, exposed by nothing at all: a sky's own numbers, clipped, which is
/// the same picture a camera pointing that way with the exposure at one would
/// take.
pub(crate) fn shrunk(data: &TextureData) -> Result<Image> {
	let Some(level) = data.levels.first() else {
		return Err!(Asset("a texture with no levels"));
	};
	let (width, height) = (data.width.max(1), data.height.max(1));
	let bytes = data.texel.bytes();
	let stride = usize::try_from(width).unwrap_or(1) * bytes;

	if level.len() < stride * usize::try_from(height).unwrap_or(1) {
		return Err!(Asset("a texture shorter than its size says"));
	}

	let mut pixels = Vec::with_capacity(usize::try_from(SIZE * SIZE * 4).unwrap_or(0));

	for y in 0..SIZE {
		let row = usize::try_from(y * height / SIZE).unwrap_or(0) * stride;

		for x in 0..SIZE {
			let at = row + usize::try_from(x * width / SIZE).unwrap_or(0) * bytes;
			let texel = level.get(at..at + bytes).unwrap_or_default();

			pixels.extend_from_slice(&flatten(texel, data.texel));
		}
	}

	Ok(Image { width: SIZE, height: SIZE, pixels })
}

/// One texel of any layout as four eight-bit channels.
fn flatten(texel: &[u8], layout: Texel) -> [u8; 4] {
	if layout != Texel::Rgba16Float {
		return <[u8; 4]>::try_from(texel).unwrap_or([0, 0, 0, 255]);
	}

	let mut out = [0, 0, 0, 255];
	for (channel, slot) in out.iter_mut().enumerate() {
		let pair = texel
			.get(channel * 2..channel * 2 + 2)
			.and_then(|slice| <[u8; 2]>::try_from(slice).ok())
			.unwrap_or([0, 0]);

		*slot = encoded(widened(u16::from_le_bytes(pair)));
	}

	out
}

/// One of the sixteen-bit values a high-range texture holds, as a float.
fn widened(bits: u16) -> f32 {
	let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
	let exponent = i32::from((bits >> 10) & 0x1F);
	let mantissa = f32::from(bits & 0x03FF);

	if exponent == 0 {
		return sign * mantissa * 2.0_f32.powi(-24);
	}

	sign * (1.0 + mantissa / 1024.0) * 2.0_f32.powi(exponent - 15)
}

/// A linear value as the sRGB byte a browser shows.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "held inside nought and two hundred and fifty-five on the line above the cast"
)]
fn encoded(value: f32) -> u8 {
	let held = value.clamp(0.0, 1.0);
	let curved = if held <= 0.003_130_8 {
		held * 12.92
	} else {
		1.055_f32.mul_add(held.powf(1.0 / 2.4), -0.055)
	};

	(curved * 255.0).round().clamp(0.0, 255.0) as u8
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
		sync::OnceLock,
		time::{Duration, SystemTime},
	};

	use colby_core::abi::{Material, mesh};
	use colby_engine::gpu;

	use super::*;

	/// The one device this test binary draws with.
	///
	/// A `OnceLock` and not a device a test, because a suite that opens one
	/// per test opens several at once and the driver falls over on it - which
	/// was measured rather than guessed. @ref [`colby_engine::gpu::shared`]'s
	/// own note, and `colby-gate-gotchas`; the renderer's tests and the
	/// interface's each keep a copy of this, because a `cfg(test)` item is not
	/// visible to another crate.
	///
	/// @return the device, or `None` on a machine with no adapter
	fn shared() -> Option<&'static Gpu> {
		static SHARED: OnceLock<Option<Gpu>> = OnceLock::new();

		SHARED
			.get_or_init(|| {
				Gpu::open(gpu::backends(None), None).expect("the adapter query works")
			})
			.as_ref()
	}

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
			faces: 1,
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

	/// The described form of a material, off its defaults.
	fn brass() -> colby_asset::model::Material {
		colby_asset::model::Material {
			name: "materials/brass".to_owned(),
			surface: Material {
				base_color: Vec3::new(0.85, 0.62, 0.22),
				metallic: 1.0,
				roughness: 0.2,
				..Material::DEFAULT
			},
			..colby_asset::model::Material::default()
		}
	}

	#[test]
	fn a_material_is_drawn_on_a_ball_and_two_materials_are_two_pictures() {
		let Some(gpu) = shared() else {
			return;
		};

		let dir = env::temp_dir().join("colby_thumbs_material");
		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(&dir).expect("a directory to work in");
		let mut thumbs = Thumbs::new(dir.join("thumbs"), dir.join("assets"));

		let output = dir.join("brass.cmat");
		fs::write(&output, colby_asset::material::encode(&brass())).expect("the file");

		let image = thumbs
			.wearing(Some(gpu), &output, "materials/brass")
			.expect("the material draws");

		assert_eq!((image.width, image.height), (SIZE, SIZE));

		let middle = image.pixel(SIZE / 2, SIZE / 2);
		let corner = image.pixel(0, 0);

		assert_ne!(middle, corner, "the ball is in the middle and the clear color in the corner");
		assert!(
			middle[0] > middle[2],
			"and it is the material's own color rather than the default's white: {middle:?}"
		);

		// the same name, another material: the picture has to move, which is
		// the registry's revision doing for a material what it does for a mesh
		let pale = colby_asset::model::Material {
			surface: Material {
				base_color: Vec3::new(0.15, 0.3, 0.9),
				metallic: 0.0,
				roughness: 0.9,
				..brass().surface
			},
			..brass()
		};
		fs::write(&output, colby_asset::material::encode(&pale)).expect("the second file");

		let again = thumbs
			.wearing(Some(gpu), &output, "materials/brass")
			.expect("the second material draws");

		assert_ne!(again.pixels, image.pixels, "a blue matte ball is not a brass one");
		assert!(
			again.pixel(SIZE / 2, SIZE / 2)[2] > again.pixel(SIZE / 2, SIZE / 2)[0],
			"and the second one is the blue"
		);
	}

	#[test]
	fn a_model_is_drawn_as_every_piece_of_itself_standing_where_it_stands() {
		let Some(gpu) = shared() else {
			return;
		};

		let dir = env::temp_dir().join("colby_thumbs_model");
		drop(fs::remove_dir_all(&dir));
		let compiled = dir.join("assets");
		fs::create_dir_all(compiled.join("models").join("tower"))
			.expect("a directory to work in");

		let mut thumbs = Thumbs::new(dir.join("thumbs"), compiled.clone());

		// two cubes, one above the other and well apart, so that a picture of
		// one of them is a different picture from one of both
		for piece in ["low", "high"] {
			fs::write(
				compiled
					.join("models")
					.join("tower")
					.join(format!("{piece}.cmesh")),
				colby_asset::format::encode(&mesh::cube()).expect("a cube encodes"),
			)
			.expect("the mesh");
		}

		let both = colby_asset::model::ModelData {
			guided: false,
			materials: vec![colby_asset::model::Material {
				name: "models/tower/paint".to_owned(),
				..brass()
			}],
			placements: [("low", 0.0), ("high", 4.0)]
				.into_iter()
				.map(|(piece, height)| colby_asset::model::Placement {
					name: piece.to_owned(),
					mesh: format!("models/tower/{piece}"),
					material: "models/tower/paint".to_owned(),
					skeleton: String::new(),
					transform: colby_core::abi::Transform::at(Vec3::Y * height),
				})
				.collect(),
		};
		let output = dir.join("tower.cmodel");
		fs::write(&output, colby_asset::model::encode(&both).expect("the model encodes"))
			.expect("the file");

		let image = thumbs
			.standing(Some(gpu), &output)
			.expect("the model draws");

		assert_eq!((image.width, image.height), (SIZE, SIZE));
		assert_eq!(thumbs.pieces.len(), 2, "an entity per piece");

		// **against each picture's own corner, not against a number.** A
		// `Capture` meters what it drew and moves its eye, and that eye
		// carries from one shot to the next - so the same clear color comes
		// out 136 on a fresh one and 140 on a reused one. Anything compared
		// across two shots has to be a shape rather than a value.
		let covered = |image: &Image| {
			let clear = image.pixel(0, 0);

			image
				.pixels
				.chunks_exact(4)
				.filter(|texel| texel[..3] != clear[..3])
				.count()
		};

		assert!(covered(&image) > 0, "something drew: the picture is not one flat color");

		// the negative control, and it is the whole test. The camera frames
		// whatever the model spans, so a tower of two puts each cube in a
		// quarter of the height and a tower of one fills the frame with it -
		// if `standing` had drawn only the first piece both times, the two
		// pictures would cover the same ground.
		let one = colby_asset::model::ModelData {
			placements: both.placements[..1].to_vec(),
			..both.clone()
		};
		fs::write(&output, colby_asset::model::encode(&one).expect("the model encodes"))
			.expect("the second file");

		let alone = thumbs
			.standing(Some(gpu), &output)
			.expect("the shorter model draws");

		assert!(covered(&alone) > 0, "the shorter one drew something too");
		assert!(
			covered(&alone) > covered(&image) * 2,
			"one cube alone fills far more of the frame than either of two does: {} against {}",
			covered(&alone),
			covered(&image)
		);

		// and the entity the second piece used draws nothing rather than what
		// it drew last time, which is what a pool costs and what it has to pay
		assert_eq!(thumbs.pieces.len(), 2, "the pool is not shrunk");
		assert_eq!(
			thumbs
				.world
				.entities
				.renderable(thumbs.pieces[1])
				.map(|renderable| renderable.mesh),
			Some(MeshId::NONE),
			"the spare draws nothing"
		);
	}

	#[test]
	fn a_material_naming_a_picture_that_is_not_there_still_draws() {
		// a material with a missing texture is still a material worth looking
		// at, and the browser is exactly where somebody would find out that
		// the texture is missing.
		let Some(gpu) = shared() else {
			return;
		};

		let dir = env::temp_dir().join("colby_thumbs_missing");
		drop(fs::remove_dir_all(&dir));
		fs::create_dir_all(&dir).expect("a directory to work in");
		let mut thumbs = Thumbs::new(dir.join("thumbs"), dir.join("assets"));

		let output = dir.join("lost.cmat");
		let lost = colby_asset::model::Material {
			albedo: "textures/nowhere".to_owned(),
			..brass()
		};
		fs::write(&output, colby_asset::material::encode(&lost)).expect("the file");

		let image = thumbs
			.wearing(Some(gpu), &output, "materials/lost")
			.expect("it draws all the same");

		assert_eq!((image.width, image.height), (SIZE, SIZE));
		assert_ne!(
			image.pixel(SIZE / 2, SIZE / 2),
			image.pixel(0, 0),
			"a ball is still in the middle of it"
		);
	}

	#[test]
	fn a_mesh_is_drawn_into_the_picture_when_there_is_a_device() {
		let Some(gpu) = shared() else {
			return;
		};

		let dir = env::temp_dir().join("colby_thumbs_render");
		drop(fs::remove_dir_all(&dir));
		let mut thumbs = Thumbs::new(dir.join("thumbs"), dir.join("assets"));

		// a compiled cube, the way the asset loop would have written it
		let output = dir.join("cube.cmesh");
		fs::create_dir_all(&dir).expect("a directory to work in");
		fs::write(&output, colby_asset::format::encode(&mesh::cube()).expect("a cube encodes"))
			.expect("the file");

		let image = thumbs
			.render(Some(gpu), &output)
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
			.render(Some(gpu), &output)
			.expect("the sphere draws");
		assert_ne!(again.pixels, image.pixels, "a sphere is not a cube");
	}
}
